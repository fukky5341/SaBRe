## Execution arguments:
Dataset: Dataset.ACAS
Network: ds/onnx/acasxu_op11/ACASXU_2_7.onnx
Relational property: GLOBAL_ROBUSTNESS
LP Analysis: True
Epsilon: None
Delta epsilon: 0.1
execution index: (10, None, 8)
Time budget: 420 seconds
Split limit: 100
Threshold: 20.6039733455


## IAR start

### BASE IAR bounds
Layer (inp1_lb, inp1_ub, inp2_lb, inp2_ub, d_lb, d_ub)
0: (-121.3309937, 166.6516571, -121.3309937, 166.6516571, -287.9826355, 287.9826050)
1: (-15.4075546, 14.3914642, -15.4075546, 14.3914642, -29.7990189, 29.7990170)
2: (-8.8996906, 14.6720200, -8.8996906, 14.6720200, -23.5717106, 23.5717106)
3: (-7.2209902, 16.0608215, -7.2209902, 16.0608215, -23.2818108, 23.2818108)
4: (-10.9869347, 13.1586323, -10.9869347, 13.1586323, -24.1455631, 24.1455631)

## BASE Relational Analysis

## BASE Result
execution time: IAR + RelationalAnalysis = 2.59 + 1.64 = 4.23 seconds
status: Status.UNKNOWN
relational distance
Output dim: 3, lower bound: -20.7075109, upper bound: 20.7075081

# Neuron Split (NS) starts

## BFS NS instance: NS

Time for backsubstitution: 0.01 seconds

### NS candidates at layer 1
type: A, layer: 1, pos: 37
type: B, layer: 1, pos: 37
type: A, layer: 1, pos: 7
type: B, layer: 1, pos: 7
type: A, layer: 1, pos: 21
type: B, layer: 1, pos: 21
type: A, layer: 1, pos: 36
type: B, layer: 1, pos: 36
type: A, layer: 1, pos: 6
type: B, layer: 1, pos: 6
type: B, layer: 1, pos: 35
type: A, layer: 1, pos: 35
type: B, layer: 1, pos: 0
type: A, layer: 1, pos: 0
type: A, layer: 1, pos: 28
type: B, layer: 1, pos: 28
type: A, layer: 1, pos: 24
type: B, layer: 1, pos: 24
type: B, layer: 1, pos: 47
type: A, layer: 1, pos: 47

Time for candidate selection: 0.20 seconds

### Candidate
type: A, layer: 1, pos: 37

## Relational analysis of NS_A1

### Relational analysis result of NS_A1
Status: Status.UNKNOWN
Output dim: 3, lower bound: -20.7063113, upper bound: 20.7002068
time: 0.61 seconds

## Relational analysis of NS_A2

### Relational analysis result of NS_A2
Status: Status.UNKNOWN
Output dim: 3, lower bound: -20.7063135, upper bound: 20.7063125
time: 0.63 seconds

## Summary of splitting at layer (split count: 0)
- Time for NS candidates: 1.46 seconds
NS_A1, status: Status.UNKNOWN, split count: 1, time: 1.46
Output dim: 3, lower bound: -20.7063113, upper bound: 20.7002068
NS_A2, status: Status.UNKNOWN, split count: 1, time: 1.46
Output dim: 3, lower bound: -20.7063135, upper bound: 20.7063125

## BFS NS instance: NS_A1

### Backsubstitution after applying NS history:
0: -115.6280289, 159.7565613, -121.3309937, 166.6516571, -282.2796326, 281.0875549
1: -14.7722130, 13.7954998, -15.4075546, 14.3914642, -29.1636753, 29.2030544
2: -8.5106497, 14.0549011, -8.8996906, 14.6720200, -23.1826649, 22.9545918
3: -6.8926716, 15.4024286, -7.2209902, 16.0608215, -22.9534931, 22.6234169
4: -10.5123882, 12.6043329, -10.9869347, 13.1586323, -23.6710148, 23.5912666

Time for backsubstitution: 2.38 seconds

### NS candidates at layer 1
type: A, layer: 1, pos: 7
type: B, layer: 1, pos: 35
type: B, layer: 1, pos: 7
type: A, layer: 1, pos: 21
type: B, layer: 1, pos: 0
type: B, layer: 1, pos: 36
type: B, layer: 1, pos: 28
type: A, layer: 1, pos: 36
type: B, layer: 1, pos: 21
type: A, layer: 1, pos: 28
type: A, layer: 1, pos: 6
type: B, layer: 1, pos: 6
type: B, layer: 1, pos: 47
type: A, layer: 1, pos: 24
type: B, layer: 1, pos: 24
type: A, layer: 1, pos: 35
type: A, layer: 1, pos: 0
type: A, layer: 1, pos: 47
type: B, layer: 1, pos: 37

Time for candidate selection: 0.20 seconds

### Candidate
type: A, layer: 1, pos: 7

## Relational analysis of NS_A1_A1

### Relational analysis result of NS_A1_A1
Status: Status.UNKNOWN
Output dim: 3, lower bound: -20.6936786, upper bound: 20.6971247
time: 0.62 seconds

## Relational analysis of NS_A1_A2
Optimization infeasible because this subproblem isn't reachable.

### Candidate
type: B, layer: 1, pos: 35

## Relational analysis of NS_A1_B1

### Relational analysis result of NS_A1_B1
Status: Status.UNKNOWN
Output dim: 3, lower bound: -20.5664895, upper bound: 20.7001801
time: 0.63 seconds

## Relational analysis of NS_A1_B2

### Relational analysis result of NS_A1_B2
Status: Status.UNKNOWN
Output dim: 3, lower bound: -20.7063010, upper bound: 20.7001959
time: 0.59 seconds

## BFS NS instance: NS_A2

### Backsubstitution after applying NS history:
0: -114.0270386, 157.7888947, -121.3309937, 166.6516571, -280.6786499, 279.1198730
1: -14.6086082, 13.5764160, -15.4075546, 14.3914642, -29.0000706, 28.9839706
2: -8.3833780, 13.8703585, -8.8996906, 14.6720200, -23.0553970, 22.7700500
3: -6.7724304, 15.2083950, -7.2209902, 16.0608215, -22.8332500, 22.4293861
4: -10.3577290, 12.4345798, -10.9869347, 13.1586323, -23.5163555, 23.4215145

Time for backsubstitution: 2.38 seconds

### NS candidates at layer 1
type: B, layer: 1, pos: 35
type: A, layer: 1, pos: 7
type: B, layer: 1, pos: 0
type: B, layer: 1, pos: 7
type: A, layer: 1, pos: 21
type: B, layer: 1, pos: 36
type: A, layer: 1, pos: 36
type: A, layer: 1, pos: 6
type: B, layer: 1, pos: 6
type: B, layer: 1, pos: 21
type: B, layer: 1, pos: 28
type: A, layer: 1, pos: 28
type: B, layer: 1, pos: 24
type: A, layer: 1, pos: 24
type: B, layer: 1, pos: 47
type: B, layer: 1, pos: 37
type: A, layer: 1, pos: 35
type: A, layer: 1, pos: 0
type: A, layer: 1, pos: 47

Time for candidate selection: 0.20 seconds

### Candidate
type: B, layer: 1, pos: 35

## Relational analysis of NS_A2_B1

### Relational analysis result of NS_A2_B1
Status: Status.UNKNOWN
Output dim: 3, lower bound: -20.5664919, upper bound: 20.7062856
time: 0.54 seconds

## Relational analysis of NS_A2_B2

### Relational analysis result of NS_A2_B2
Status: Status.UNKNOWN
Output dim: 3, lower bound: -20.7063020, upper bound: 20.7063035
time: 0.58 seconds

## Summary of splitting at layer (split count: 1)
- Time for NS candidates: 3.72 seconds
NS_A1_B1, status: Status.UNKNOWN, split count: 2, time: 3.72
Output dim: 3, lower bound: -20.5664895, upper bound: 20.7001801
NS_A1_B2, status: Status.UNKNOWN, split count: 2, time: 3.72
Output dim: 3, lower bound: -20.7063010, upper bound: 20.7001959
NS_A2_B1, status: Status.UNKNOWN, split count: 2, time: 3.72
Output dim: 3, lower bound: -20.5664919, upper bound: 20.7062856
NS_A2_B2, status: Status.UNKNOWN, split count: 2, time: 3.72
Output dim: 3, lower bound: -20.7063020, upper bound: 20.7063035

## BFS NS instance: NS_A1_B1

### Backsubstitution after applying NS history:
0: -115.6280289, 159.7565613, -64.4378891, 83.7772598, -199.4052887, 224.1944580
1: -14.7722130, 13.7954998, -7.6196566, 7.6589622, -22.4311752, 21.4151573
2: -8.5106497, 14.0549011, -4.6781135, 7.2810550, -15.7917032, 18.7330132
3: -6.8926716, 15.4024286, -3.8368852, 7.8101020, -14.7027721, 19.2393093
4: -10.5123882, 12.6043329, -5.8599577, 6.5999441, -17.1123295, 18.4642887

Time for backsubstitution: 2.39 seconds

### NS candidates at layer 1
type: A, layer: 1, pos: 35
type: A, layer: 1, pos: 0
type: B, layer: 1, pos: 21
type: A, layer: 1, pos: 47
type: B, layer: 1, pos: 7
type: A, layer: 1, pos: 7
type: A, layer: 1, pos: 36
type: A, layer: 1, pos: 28
type: B, layer: 1, pos: 37
type: A, layer: 1, pos: 24
type: B, layer: 1, pos: 24
type: A, layer: 1, pos: 6
type: B, layer: 1, pos: 28
type: B, layer: 1, pos: 36
type: B, layer: 1, pos: 6
type: A, layer: 1, pos: 21
type: B, layer: 1, pos: 47
type: B, layer: 1, pos: 0

Time for candidate selection: 0.20 seconds

### Candidate
type: A, layer: 1, pos: 35

## Relational analysis of NS_A1_B1_A1

### Relational analysis result of NS_A1_B1_A1
Status: Status.VERIFIED
Output dim: 3, lower bound: -20.5664705, upper bound: 20.5589725
time: 0.60 seconds

## Relational analysis of NS_A1_B1_A2

### Relational analysis result of NS_A1_B1_A2
Status: Status.UNKNOWN
Output dim: 3, lower bound: -20.5664705, upper bound: 20.7001819
time: 0.51 seconds

## BFS NS instance: NS_A1_B2

### Backsubstitution after applying NS history:
0: -115.6280289, 159.7565613, -120.8285675, 166.0415649, -281.6695251, 280.5851440
1: -14.7722130, 13.7954998, -15.3544054, 14.3327837, -29.1049957, 29.1499062
2: -8.5106497, 14.0549011, -8.8637476, 14.6180887, -23.1287365, 22.9186478
3: -6.8926716, 15.4024286, -7.1897345, 16.0026474, -22.8953152, 22.5921593
4: -10.5123882, 12.6043329, -10.9433813, 13.1088514, -23.6212349, 23.5477142

Time for backsubstitution: 2.38 seconds

### NS candidates at layer 1
type: A, layer: 1, pos: 21
type: A, layer: 1, pos: 7
type: B, layer: 1, pos: 7
type: B, layer: 1, pos: 0
type: B, layer: 1, pos: 36
type: B, layer: 1, pos: 28
type: A, layer: 1, pos: 36
type: B, layer: 1, pos: 21
type: A, layer: 1, pos: 28
type: A, layer: 1, pos: 6
type: B, layer: 1, pos: 6
type: B, layer: 1, pos: 47
type: A, layer: 1, pos: 24
type: B, layer: 1, pos: 24
type: A, layer: 1, pos: 0
type: A, layer: 1, pos: 35
type: A, layer: 1, pos: 47
type: B, layer: 1, pos: 37

Time for candidate selection: 0.20 seconds

### Candidate
type: A, layer: 1, pos: 21

## Relational analysis of NS_A1_B2_A1

### Relational analysis result of NS_A1_B2_A1
Status: Status.UNKNOWN
Output dim: 3, lower bound: -20.6129566, upper bound: 20.6993624
time: 0.53 seconds

## Relational analysis of NS_A1_B2_A2

### Relational analysis result of NS_A1_B2_A2
Status: Status.UNKNOWN
Output dim: 3, lower bound: -20.6129429, upper bound: 20.6068234
time: 0.52 seconds

## BFS NS instance: NS_A2_B1

### Backsubstitution after applying NS history:
0: -114.0270386, 157.7888947, -64.4378891, 83.7772598, -197.8042603, 222.2267761
1: -14.6086082, 13.5764160, -7.6196566, 7.6589622, -22.2675705, 21.1960716
2: -8.3833780, 13.8703585, -4.6781135, 7.2810550, -15.6644316, 18.5484676
3: -6.7724304, 15.2083950, -3.8368852, 7.8101020, -14.5825310, 19.0452766
4: -10.3577290, 12.4345798, -5.8599577, 6.5999441, -16.9576721, 18.2945366

Time for backsubstitution: 2.38 seconds

### NS candidates at layer 1
type: A, layer: 1, pos: 35
type: A, layer: 1, pos: 0
type: B, layer: 1, pos: 21
type: A, layer: 1, pos: 47
type: B, layer: 1, pos: 37
type: A, layer: 1, pos: 36
type: B, layer: 1, pos: 7
type: A, layer: 1, pos: 7
type: A, layer: 1, pos: 28
type: A, layer: 1, pos: 6
type: A, layer: 1, pos: 24
type: B, layer: 1, pos: 24
type: B, layer: 1, pos: 36
type: B, layer: 1, pos: 28
type: B, layer: 1, pos: 6
type: A, layer: 1, pos: 21
type: B, layer: 1, pos: 47
type: B, layer: 1, pos: 0

Time for candidate selection: 0.20 seconds

### Candidate
type: A, layer: 1, pos: 35

## Relational analysis of NS_A2_B1_A1

### Relational analysis result of NS_A2_B1_A1
Status: Status.VERIFIED
Output dim: 3, lower bound: -20.5664753, upper bound: 20.5664713
time: 0.62 seconds

## Relational analysis of NS_A2_B1_A2

### Relational analysis result of NS_A2_B1_A2
Status: Status.UNKNOWN
Output dim: 3, lower bound: -20.5664753, upper bound: 20.7062822
time: 0.60 seconds

## BFS NS instance: NS_A2_B2

### Backsubstitution after applying NS history:
0: -114.0270386, 157.7888947, -120.8285675, 166.0415649, -280.0685730, 278.6174316
1: -14.6086082, 13.5764160, -15.3544054, 14.3327837, -28.9413910, 28.9308205
2: -8.3833780, 13.8703585, -8.8637476, 14.6180887, -23.0014668, 22.7341061
3: -6.7724304, 15.2083950, -7.1897345, 16.0026474, -22.7750759, 22.3981285
4: -10.3577290, 12.4345798, -10.9433813, 13.1088514, -23.4665775, 23.3779602

Time for backsubstitution: 2.39 seconds

### NS candidates at layer 1
type: A, layer: 1, pos: 7
type: B, layer: 1, pos: 7
type: B, layer: 1, pos: 0
type: A, layer: 1, pos: 21
type: B, layer: 1, pos: 36
type: A, layer: 1, pos: 36
type: A, layer: 1, pos: 6
type: B, layer: 1, pos: 6
type: B, layer: 1, pos: 21
type: B, layer: 1, pos: 28
type: A, layer: 1, pos: 28
type: B, layer: 1, pos: 24
type: A, layer: 1, pos: 24
type: B, layer: 1, pos: 47
type: B, layer: 1, pos: 37
type: A, layer: 1, pos: 0
type: A, layer: 1, pos: 47
type: A, layer: 1, pos: 35

Time for candidate selection: 0.20 seconds

### Candidate
type: A, layer: 1, pos: 7

## Relational analysis of NS_A2_B2_A1

### Relational analysis result of NS_A2_B2_A1
Status: Status.UNKNOWN
Output dim: 3, lower bound: -20.6936500, upper bound: 20.7024360
time: 0.58 seconds

## Relational analysis of NS_A2_B2_A2

### Relational analysis result of NS_A2_B2_A2
Status: Status.UNKNOWN
Output dim: 3, lower bound: -20.6936514, upper bound: 20.6936515
time: 0.71 seconds

## Summary of splitting at layer (split count: 2)
- Time for NS candidates: 3.89 seconds
NS_A1_B1_A1, status: Status.VERIFIED, split count: 3, time: 3.89
Output dim: 3, lower bound: -20.5664705, upper bound: 20.5589725
NS_A1_B1_A2, status: Status.UNKNOWN, split count: 3, time: 3.89
Output dim: 3, lower bound: -20.5664705, upper bound: 20.7001819
NS_A1_B2_A1, status: Status.UNKNOWN, split count: 3, time: 3.89
Output dim: 3, lower bound: -20.6129566, upper bound: 20.6993624
NS_A1_B2_A2, status: Status.UNKNOWN, split count: 3, time: 3.89
Output dim: 3, lower bound: -20.6129429, upper bound: 20.6068234
NS_A2_B1_A1, status: Status.VERIFIED, split count: 3, time: 3.89
Output dim: 3, lower bound: -20.5664753, upper bound: 20.5664713
NS_A2_B1_A2, status: Status.UNKNOWN, split count: 3, time: 3.89
Output dim: 3, lower bound: -20.5664753, upper bound: 20.7062822
NS_A2_B2_A1, status: Status.UNKNOWN, split count: 3, time: 3.89
Output dim: 3, lower bound: -20.6936500, upper bound: 20.7024360
NS_A2_B2_A2, status: Status.UNKNOWN, split count: 3, time: 3.89
Output dim: 3, lower bound: -20.6936514, upper bound: 20.6936515

## BFS NS instance: NS_A1_B1_A2

### Backsubstitution after applying NS history:
0: -115.2759705, 159.3167419, -64.4378891, 83.7772598, -199.0531921, 223.7546387
1: -14.7341137, 13.7537832, -7.6196566, 7.6589622, -22.3930759, 21.3734341
2: -8.4850483, 14.0161543, -4.6781135, 7.2810550, -15.7661037, 18.6942654
3: -6.8703575, 15.3609219, -3.8368852, 7.8101020, -14.6804581, 19.1978035
4: -10.4817038, 12.5685396, -5.8599577, 6.5999441, -17.0816460, 18.4284954

Time for backsubstitution: 2.38 seconds

### NS candidates at layer 1
type: A, layer: 1, pos: 0
type: B, layer: 1, pos: 21
type: A, layer: 1, pos: 47
type: B, layer: 1, pos: 7
type: A, layer: 1, pos: 7
type: A, layer: 1, pos: 36
type: A, layer: 1, pos: 28
type: B, layer: 1, pos: 37
type: A, layer: 1, pos: 24
type: B, layer: 1, pos: 24
type: A, layer: 1, pos: 6
type: B, layer: 1, pos: 28
type: B, layer: 1, pos: 36
type: B, layer: 1, pos: 6
type: A, layer: 1, pos: 21
type: B, layer: 1, pos: 47
type: B, layer: 1, pos: 0

Time for candidate selection: 0.20 seconds

### Candidate
type: A, layer: 1, pos: 0

## Relational analysis of NS_A1_B1_A2_A1
Optimization infeasible because this subproblem isn't reachable.

### Candidate
type: B, layer: 1, pos: 21

## Relational analysis of NS_A1_B1_A2_B1
Optimization infeasible because this subproblem isn't reachable.

### Candidate
type: A, layer: 1, pos: 47

## Relational analysis of NS_A1_B1_A2_A1
Optimization infeasible because this subproblem isn't reachable.

### Candidate
type: B, layer: 1, pos: 7

### Candidate
type: A, layer: 1, pos: 7

### Candidate
type: A, layer: 1, pos: 36

## Relational analysis of NS_A1_B1_A2_A1

### Relational analysis result of NS_A1_B1_A2_A1
Status: Status.UNKNOWN
Output dim: 3, lower bound: -20.5384859, upper bound: 20.7001772
time: 0.55 seconds

## Relational analysis of NS_A1_B1_A2_A2

### Relational analysis result of NS_A1_B1_A2_A2
Status: Status.VERIFIED
Output dim: 3, lower bound: -20.5384812, upper bound: 20.5249424
time: 0.54 seconds

## BFS NS instance: NS_A1_B2_A1

### Backsubstitution after applying NS history:
0: -107.4569092, 149.4062958, -120.8285675, 166.0415649, -273.4983521, 270.2348328
1: -13.8525238, 12.8498917, -15.3544054, 14.3327837, -28.1853065, 28.2042923
2: -7.9272017, 13.1349926, -8.8637476, 14.6180887, -22.5452881, 21.9987411
3: -6.3961639, 14.4240589, -7.1897345, 16.0026474, -22.3988094, 21.6137924
4: -9.8063383, 11.7648029, -10.9433813, 13.1088514, -22.9151840, 22.7081833

Time for backsubstitution: 2.41 seconds

### NS candidates at layer 1
type: B, layer: 1, pos: 0
type: A, layer: 1, pos: 7
type: B, layer: 1, pos: 7
type: B, layer: 1, pos: 36
type: A, layer: 1, pos: 36
type: B, layer: 1, pos: 28
type: A, layer: 1, pos: 28
type: B, layer: 1, pos: 47
type: B, layer: 1, pos: 6
type: A, layer: 1, pos: 6
type: B, layer: 1, pos: 24
type: A, layer: 1, pos: 24
type: B, layer: 1, pos: 21
type: A, layer: 1, pos: 47
type: A, layer: 1, pos: 35
type: A, layer: 1, pos: 0
type: B, layer: 1, pos: 37

Time for candidate selection: 0.20 seconds

### Candidate
type: B, layer: 1, pos: 0

## Relational analysis of NS_A1_B2_A1_B1
Optimization infeasible because this subproblem isn't reachable.

### Candidate
type: A, layer: 1, pos: 7

### Candidate
type: B, layer: 1, pos: 7

### Candidate
type: B, layer: 1, pos: 36

## Relational analysis of NS_A1_B2_A1_B1

### Relational analysis result of NS_A1_B2_A1_B1
Status: Status.UNKNOWN
Output dim: 3, lower bound: -20.6129574, upper bound: 20.6838649
time: 0.68 seconds

## Relational analysis of NS_A1_B2_A1_B2

### Relational analysis result of NS_A1_B2_A1_B2
Status: Status.UNKNOWN
Output dim: 3, lower bound: -20.5833225, upper bound: 20.6838609
time: 0.57 seconds

## BFS NS instance: NS_A1_B2_A2

### Backsubstitution after applying NS history:
0: -127.9776459, 179.3025665, -120.8285675, 166.0415649, -294.0191650, 300.1311035
1: -16.6276455, 15.3680820, -15.3544054, 14.3327837, -30.9604282, 30.7224846
2: -9.4743786, 15.7429419, -8.8637476, 14.6180887, -24.0924664, 24.6066895
3: -7.6515388, 17.2871590, -7.1897345, 16.0026474, -23.6541862, 24.4768906
4: -11.7098789, 14.1008635, -10.9433813, 13.1088514, -24.8187294, 25.0442448

Time for backsubstitution: 2.40 seconds

### NS candidates at layer 1
type: B, layer: 1, pos: 21
type: A, layer: 1, pos: 7
type: B, layer: 1, pos: 7
type: A, layer: 1, pos: 0
type: A, layer: 1, pos: 35
type: B, layer: 1, pos: 36
type: A, layer: 1, pos: 36
type: B, layer: 1, pos: 28
type: A, layer: 1, pos: 28
type: A, layer: 1, pos: 6
type: B, layer: 1, pos: 6
type: B, layer: 1, pos: 0
type: A, layer: 1, pos: 24
type: B, layer: 1, pos: 24
type: B, layer: 1, pos: 47
type: A, layer: 1, pos: 47
type: B, layer: 1, pos: 37

Time for candidate selection: 0.20 seconds

### Candidate
type: B, layer: 1, pos: 21

## Relational analysis of NS_A1_B2_A2_B1

### Relational analysis result of NS_A1_B2_A2_B1
Status: Status.UNKNOWN
Output dim: 3, lower bound: -20.6129459, upper bound: 20.6068211
time: 0.59 seconds

## Relational analysis of NS_A1_B2_A2_B2

### Relational analysis result of NS_A1_B2_A2_B2
Status: Status.UNKNOWN
Output dim: 3, lower bound: -20.6129459, upper bound: 20.6068217
time: 0.54 seconds

## BFS NS instance: NS_A2_B1_A2

### Backsubstitution after applying NS history:
0: -113.5251083, 157.1798248, -64.4378891, 83.7772598, -197.3023682, 221.6177063
1: -14.5556822, 13.5179958, -7.6196566, 7.6589622, -22.2146454, 21.1376514
2: -8.3479624, 13.8164997, -4.6781135, 7.2810550, -15.6290159, 18.4946136
3: -6.7410970, 15.1504660, -3.8368852, 7.8101020, -14.5511990, 18.9873466
4: -10.3142529, 12.3849545, -5.8599577, 6.5999441, -16.9141960, 18.2449112

Time for backsubstitution: 2.41 seconds

### NS candidates at layer 1
type: A, layer: 1, pos: 0
type: B, layer: 1, pos: 21
type: A, layer: 1, pos: 47
type: B, layer: 1, pos: 37
type: A, layer: 1, pos: 36
type: B, layer: 1, pos: 7
type: A, layer: 1, pos: 7
type: A, layer: 1, pos: 28
type: A, layer: 1, pos: 6
type: A, layer: 1, pos: 24
type: B, layer: 1, pos: 24
type: B, layer: 1, pos: 36
type: B, layer: 1, pos: 28
type: B, layer: 1, pos: 6
type: A, layer: 1, pos: 21
type: B, layer: 1, pos: 47
type: B, layer: 1, pos: 0

Time for candidate selection: 0.20 seconds

### Candidate
type: A, layer: 1, pos: 0

## Relational analysis of NS_A2_B1_A2_A1
Optimization infeasible because this subproblem isn't reachable.

### Candidate
type: B, layer: 1, pos: 21

## Relational analysis of NS_A2_B1_A2_B1
Optimization infeasible because this subproblem isn't reachable.

### Candidate
type: A, layer: 1, pos: 47

## Relational analysis of NS_A2_B1_A2_A1

### Relational analysis result of NS_A2_B1_A2_A1
Status: Status.VERIFIED
Output dim: 3, lower bound: -20.5652193, upper bound: 20.5893648
time: 0.57 seconds

## Relational analysis of NS_A2_B1_A2_A2

### Relational analysis result of NS_A2_B1_A2_A2
Status: Status.UNKNOWN
Output dim: 3, lower bound: -20.5663953, upper bound: 20.7062711
time: 0.57 seconds

## BFS NS instance: NS_A2_B2_A1

### Backsubstitution after applying NS history:
0: -102.6300964, 142.3890839, -120.8285675, 166.0415649, -268.6716003, 263.2176514
1: -13.1932011, 12.2476940, -15.3544054, 14.3327837, -27.5259857, 27.6020966
2: -7.5800185, 12.4953012, -8.8637476, 14.6180887, -22.1981030, 21.3590488
3: -6.0822811, 13.7058611, -7.1897345, 16.0026474, -22.0849228, 20.8955956
4: -9.3485603, 11.1931620, -10.9433813, 13.1088514, -22.4574127, 22.1365433

Time for backsubstitution: 2.40 seconds

### NS candidates at layer 1
type: B, layer: 1, pos: 0
type: B, layer: 1, pos: 36
type: A, layer: 1, pos: 21
type: B, layer: 1, pos: 7
type: A, layer: 1, pos: 36
type: B, layer: 1, pos: 47
type: B, layer: 1, pos: 6
type: A, layer: 1, pos: 6
type: B, layer: 1, pos: 28
type: A, layer: 1, pos: 28
type: B, layer: 1, pos: 24
type: A, layer: 1, pos: 24
type: B, layer: 1, pos: 21
type: B, layer: 1, pos: 37
type: A, layer: 1, pos: 47
type: A, layer: 1, pos: 0
type: A, layer: 1, pos: 35

Time for candidate selection: 0.20 seconds

### Candidate
type: B, layer: 1, pos: 0

## Relational analysis of NS_A2_B2_A1_B1
Optimization infeasible because this subproblem isn't reachable.

### Candidate
type: B, layer: 1, pos: 36

## Relational analysis of NS_A2_B2_A1_B1

### Relational analysis result of NS_A2_B2_A1_B1
Status: Status.UNKNOWN
Output dim: 3, lower bound: -20.6936477, upper bound: 20.6837275
time: 0.61 seconds

## Relational analysis of NS_A2_B2_A1_B2

### Relational analysis result of NS_A2_B2_A1_B2
Status: Status.UNKNOWN
Output dim: 3, lower bound: -20.6736842, upper bound: 20.6837248
time: 0.67 seconds

## BFS NS instance: NS_A2_B2_A2

### Backsubstitution after applying NS history:
0: -102.3066483, 142.2346344, -120.8285675, 166.0415649, -268.3482056, 263.0632019
1: -13.1595669, 12.2468472, -15.3544054, 14.3327837, -27.4923496, 27.6012497
2: -7.5773954, 12.4816761, -8.8637476, 14.6180887, -22.1954842, 21.3454247
3: -6.1004505, 13.6574812, -7.1897345, 16.0026474, -22.1030941, 20.8472099
4: -9.3307514, 11.1743279, -10.9433813, 13.1088514, -22.4396019, 22.1177044

Time for backsubstitution: 2.42 seconds

### NS candidates at layer 1
type: B, layer: 1, pos: 0
type: A, layer: 1, pos: 21
type: B, layer: 1, pos: 7
type: B, layer: 1, pos: 47
type: B, layer: 1, pos: 28
type: B, layer: 1, pos: 36
type: A, layer: 1, pos: 28
type: B, layer: 1, pos: 6
type: A, layer: 1, pos: 36
type: A, layer: 1, pos: 6
type: A, layer: 1, pos: 24
type: B, layer: 1, pos: 24
type: B, layer: 1, pos: 37
type: B, layer: 1, pos: 21
type: A, layer: 1, pos: 47
type: A, layer: 1, pos: 0
type: A, layer: 1, pos: 35

Time for candidate selection: 0.20 seconds

### Candidate
type: B, layer: 1, pos: 0

## Relational analysis of NS_A2_B2_A2_B1
Optimization infeasible because this subproblem isn't reachable.

### Candidate
type: A, layer: 1, pos: 21

## Relational analysis of NS_A2_B2_A2_A1

### Relational analysis result of NS_A2_B2_A2_A1
Status: Status.UNKNOWN
Output dim: 3, lower bound: -20.5937076, upper bound: 20.6928904
time: 0.60 seconds

## Relational analysis of NS_A2_B2_A2_A2

### Relational analysis result of NS_A2_B2_A2_A2
Status: Status.VERIFIED
Output dim: 3, lower bound: -20.5936955, upper bound: 20.5936923
time: 0.61 seconds

## Summary of splitting at layer (split count: 3)
- Time for NS candidates: 4.76 seconds
NS_A1_B1_A2_A1, status: Status.UNKNOWN, split count: 4, time: 4.76
Output dim: 3, lower bound: -20.5384859, upper bound: 20.7001772
NS_A1_B1_A2_A2, status: Status.VERIFIED, split count: 4, time: 4.76
Output dim: 3, lower bound: -20.5384812, upper bound: 20.5249424
NS_A1_B2_A1_B1, status: Status.UNKNOWN, split count: 4, time: 4.76
Output dim: 3, lower bound: -20.6129574, upper bound: 20.6838649
NS_A1_B2_A1_B2, status: Status.UNKNOWN, split count: 4, time: 4.76
Output dim: 3, lower bound: -20.5833225, upper bound: 20.6838609
NS_A1_B2_A2_B1, status: Status.UNKNOWN, split count: 4, time: 4.76
Output dim: 3, lower bound: -20.6129459, upper bound: 20.6068211
NS_A1_B2_A2_B2, status: Status.UNKNOWN, split count: 4, time: 4.76
Output dim: 3, lower bound: -20.6129459, upper bound: 20.6068217
NS_A2_B1_A2_A1, status: Status.VERIFIED, split count: 4, time: 4.76
Output dim: 3, lower bound: -20.5652193, upper bound: 20.5893648
NS_A2_B1_A2_A2, status: Status.UNKNOWN, split count: 4, time: 4.76
Output dim: 3, lower bound: -20.5663953, upper bound: 20.7062711
NS_A2_B2_A1_B1, status: Status.UNKNOWN, split count: 4, time: 4.76
Output dim: 3, lower bound: -20.6936477, upper bound: 20.6837275
NS_A2_B2_A1_B2, status: Status.UNKNOWN, split count: 4, time: 4.76
Output dim: 3, lower bound: -20.6736842, upper bound: 20.6837248
NS_A2_B2_A2_A1, status: Status.UNKNOWN, split count: 4, time: 4.76
Output dim: 3, lower bound: -20.5937076, upper bound: 20.6928904
NS_A2_B2_A2_A2, status: Status.VERIFIED, split count: 4, time: 4.76
Output dim: 3, lower bound: -20.5936955, upper bound: 20.5936923

## BFS NS instance: NS_A1_B1_A2_A1

### Backsubstitution after applying NS history:
0: -110.1612396, 153.0794373, -64.4378891, 83.7772598, -193.9384918, 217.5173340
1: -14.1676292, 13.1946115, -7.6196566, 7.6589622, -21.8265915, 20.8142681
2: -8.1353064, 13.4398422, -4.6781135, 7.2810550, -15.4163609, 18.1179543
3: -6.5701790, 14.7620602, -3.8368852, 7.8101020, -14.3802814, 18.5989437
4: -10.0504532, 12.0519505, -5.8599577, 6.5999441, -16.6503983, 17.9119053

Time for backsubstitution: 2.41 seconds

### NS candidates at layer 1
type: A, layer: 1, pos: 0
type: B, layer: 1, pos: 21
type: A, layer: 1, pos: 47
type: B, layer: 1, pos: 7
type: A, layer: 1, pos: 28
type: A, layer: 1, pos: 7
type: B, layer: 1, pos: 37
type: A, layer: 1, pos: 24
type: B, layer: 1, pos: 24
type: B, layer: 1, pos: 6
type: B, layer: 1, pos: 28
type: A, layer: 1, pos: 6
type: B, layer: 1, pos: 36
type: B, layer: 1, pos: 47
type: A, layer: 1, pos: 21
type: B, layer: 1, pos: 0

Time for candidate selection: 0.20 seconds

### Candidate
type: A, layer: 1, pos: 0

### Candidate
type: B, layer: 1, pos: 21

### Candidate
type: A, layer: 1, pos: 47

### Candidate
type: B, layer: 1, pos: 7

### Candidate
type: A, layer: 1, pos: 28

## Relational analysis of NS_A1_B1_A2_A1_A1

### Relational analysis result of NS_A1_B1_A2_A1_A1
Status: Status.UNKNOWN
Output dim: 3, lower bound: -20.5300532, upper bound: 20.6343364
time: 0.62 seconds

## Relational analysis of NS_A1_B1_A2_A1_A2

### Relational analysis result of NS_A1_B1_A2_A1_A2
Status: Status.UNKNOWN
Output dim: 3, lower bound: -20.5300592, upper bound: 20.6980175
time: 0.63 seconds

## BFS NS instance: NS_A1_B2_A1_B1

### Backsubstitution after applying NS history:
0: -107.4569092, 149.4062958, -116.0162811, 160.4569397, -267.9137878, 265.4225159
1: -13.8525238, 12.8498917, -14.8514814, 13.8080673, -27.6605911, 27.7013702
2: -7.9272017, 13.1349926, -8.5386076, 14.1044064, -22.0316048, 21.6735992
3: -6.3961639, 14.4240589, -6.9060659, 15.4590082, -21.8551712, 21.3301239
4: -9.8063383, 11.7648029, -10.5402803, 12.6461687, -22.4525051, 22.3050823

Time for backsubstitution: 2.40 seconds

### NS candidates at layer 1
type: A, layer: 1, pos: 7
type: B, layer: 1, pos: 7
type: B, layer: 1, pos: 0
type: B, layer: 1, pos: 28
type: A, layer: 1, pos: 28
type: A, layer: 1, pos: 35
type: A, layer: 1, pos: 0
type: A, layer: 1, pos: 36
type: B, layer: 1, pos: 24
type: B, layer: 1, pos: 47
type: A, layer: 1, pos: 6
type: B, layer: 1, pos: 6
type: A, layer: 1, pos: 24
type: B, layer: 1, pos: 21
type: A, layer: 1, pos: 47
type: B, layer: 1, pos: 37

Time for candidate selection: 0.20 seconds

### Candidate
type: A, layer: 1, pos: 7

### Candidate
type: B, layer: 1, pos: 7

### Candidate
type: B, layer: 1, pos: 0

### Candidate
type: B, layer: 1, pos: 28

## Relational analysis of NS_A1_B2_A1_B1_B1

### Relational analysis result of NS_A1_B2_A1_B1_B1
Status: Status.UNKNOWN
Output dim: 3, lower bound: -20.6094994, upper bound: 20.6827621
time: 0.59 seconds

## Relational analysis of NS_A1_B2_A1_B1_B2

### Relational analysis result of NS_A1_B2_A1_B1_B2
Status: Status.UNKNOWN
Output dim: 3, lower bound: -20.6033761, upper bound: 20.6827606
time: 0.61 seconds

## BFS NS instance: NS_A1_B2_A1_B2

### Backsubstitution after applying NS history:
0: -107.4569092, 149.4062958, -117.1993408, 161.3668365, -268.8237305, 266.6056213
1: -13.8525238, 12.8498917, -14.9286995, 13.9060850, -27.7586040, 27.7785892
2: -7.9272017, 13.1349926, -8.6000643, 14.2070026, -22.1342049, 21.7350578
3: -6.3961639, 14.4240589, -6.9678402, 15.5446568, -21.9408207, 21.3918991
4: -9.8063383, 11.7648029, -10.6234360, 12.7364960, -22.5428333, 22.3882389

Time for backsubstitution: 2.41 seconds

### NS candidates at layer 1
type: A, layer: 1, pos: 7
type: B, layer: 1, pos: 7
type: B, layer: 1, pos: 0
type: B, layer: 1, pos: 28
type: A, layer: 1, pos: 28
type: B, layer: 1, pos: 47
type: A, layer: 1, pos: 36
type: B, layer: 1, pos: 6
type: B, layer: 1, pos: 21
type: B, layer: 1, pos: 24
type: A, layer: 1, pos: 6
type: A, layer: 1, pos: 24
type: A, layer: 1, pos: 35
type: A, layer: 1, pos: 0
type: A, layer: 1, pos: 47
type: B, layer: 1, pos: 37

Time for candidate selection: 0.20 seconds

### Candidate
type: A, layer: 1, pos: 7

### Candidate
type: B, layer: 1, pos: 7

### Candidate
type: B, layer: 1, pos: 0

### Candidate
type: B, layer: 1, pos: 28

## Relational analysis of NS_A1_B2_A1_B2_B1

### Relational analysis result of NS_A1_B2_A1_B2_B1
Status: Status.UNKNOWN
Output dim: 3, lower bound: -20.5821564, upper bound: 20.6827564
time: 0.64 seconds

## Relational analysis of NS_A1_B2_A1_B2_B2

### Relational analysis result of NS_A1_B2_A1_B2_B2
Status: Status.UNKNOWN
Output dim: 3, lower bound: -20.5779195, upper bound: 20.6827564
time: 0.60 seconds

## BFS NS instance: NS_A1_B2_A2_B1

### Backsubstitution after applying NS history:
0: -127.9776459, 179.3025665, -112.5631027, 155.5566559, -283.5343018, 291.8655090
1: -16.6276455, 15.3680820, -14.4233618, 13.3771133, -30.0047531, 29.7914429
2: -9.4743786, 15.7429419, -8.2733898, 13.6866093, -23.1609879, 24.0163307
3: -7.6515388, 17.2871590, -6.6859941, 15.0133543, -22.6648941, 23.9731522
4: -11.7098789, 14.1008635, -10.2299709, 12.2601938, -23.9700737, 24.3308334

Time for backsubstitution: 2.41 seconds

### NS candidates at layer 1
type: A, layer: 1, pos: 35
type: A, layer: 1, pos: 0
type: A, layer: 1, pos: 7
type: B, layer: 1, pos: 7
type: B, layer: 1, pos: 36
type: A, layer: 1, pos: 36
type: B, layer: 1, pos: 28
type: A, layer: 1, pos: 28
type: A, layer: 1, pos: 6
type: A, layer: 1, pos: 24
type: B, layer: 1, pos: 24
type: B, layer: 1, pos: 6
type: B, layer: 1, pos: 47
type: A, layer: 1, pos: 47
type: B, layer: 1, pos: 0
type: B, layer: 1, pos: 37

Time for candidate selection: 0.20 seconds

### Candidate
type: A, layer: 1, pos: 35

## Relational analysis of NS_A1_B2_A2_B1_A1
Optimization infeasible because this subproblem isn't reachable.

### Candidate
type: A, layer: 1, pos: 0

## Relational analysis of NS_A1_B2_A2_B1_A1
Optimization infeasible because this subproblem isn't reachable.

### Candidate
type: A, layer: 1, pos: 7

### Candidate
type: B, layer: 1, pos: 7

### Candidate
type: B, layer: 1, pos: 36

## Relational analysis of NS_A1_B2_A2_B1_B1

### Relational analysis result of NS_A1_B2_A2_B1_B1
Status: Status.UNKNOWN
Output dim: 3, lower bound: -20.6129425, upper bound: 20.5705919
time: 0.58 seconds

## Relational analysis of NS_A1_B2_A2_B1_B2

### Relational analysis result of NS_A1_B2_A2_B1_B2
Status: Status.VERIFIED
Output dim: 3, lower bound: -20.5833051, upper bound: 20.5705843
time: 0.60 seconds

## BFS NS instance: NS_A1_B2_A2_B2

### Backsubstitution after applying NS history:
0: -127.9776459, 179.3025665, -133.7400055, 186.2889862, -314.2666321, 313.0424805
1: -16.6276455, 15.3680820, -17.2766457, 15.9703350, -32.5979729, 32.6447220
2: -9.4743786, 15.7429419, -9.8654881, 16.3711472, -25.8455238, 25.6084290
3: -7.6515388, 17.2871590, -7.9798994, 17.9552975, -25.6068363, 25.2670536
4: -11.7098789, 14.1008635, -12.1915236, 14.6639843, -26.3738632, 26.2923870

Time for backsubstitution: 2.41 seconds

### NS candidates at layer 1
type: A, layer: 1, pos: 7
type: B, layer: 1, pos: 0
type: B, layer: 1, pos: 7
type: B, layer: 1, pos: 36
type: A, layer: 1, pos: 36
type: B, layer: 1, pos: 28
type: A, layer: 1, pos: 28
type: A, layer: 1, pos: 6
type: B, layer: 1, pos: 6
type: A, layer: 1, pos: 24
type: B, layer: 1, pos: 24
type: B, layer: 1, pos: 47
type: A, layer: 1, pos: 0
type: A, layer: 1, pos: 35
type: A, layer: 1, pos: 47
type: B, layer: 1, pos: 37

Time for candidate selection: 0.20 seconds

### Candidate
type: A, layer: 1, pos: 7

### Candidate
type: B, layer: 1, pos: 0

## Relational analysis of NS_A1_B2_A2_B2_B1
Optimization infeasible because this subproblem isn't reachable.

### Candidate
type: B, layer: 1, pos: 7

### Candidate
type: B, layer: 1, pos: 36

## Relational analysis of NS_A1_B2_A2_B2_B1

### Relational analysis result of NS_A1_B2_A2_B2_B1
Status: Status.UNKNOWN
Output dim: 3, lower bound: -20.6129393, upper bound: 20.5705957
time: 0.67 seconds

## Relational analysis of NS_A1_B2_A2_B2_B2

### Relational analysis result of NS_A1_B2_A2_B2_B2
Status: Status.VERIFIED
Output dim: 3, lower bound: -20.5833068, upper bound: 20.5705876
time: 0.64 seconds

## BFS NS instance: NS_A2_B1_A2_A2

### Backsubstitution after applying NS history:
0: -113.3691940, 156.9696960, -64.4378891, 83.7772598, -197.1464539, 221.4075928
1: -14.5364876, 13.5001736, -7.6196566, 7.6589622, -22.1954479, 21.1198273
2: -8.3368378, 13.7978535, -4.6781135, 7.2810550, -15.6178904, 18.4759674
3: -6.7319307, 15.1299477, -3.8368852, 7.8101020, -14.5420322, 18.9668293
4: -10.3006001, 12.3681145, -5.8599577, 6.5999441, -16.9005432, 18.2280693

Time for backsubstitution: 2.43 seconds

### NS candidates at layer 1
type: A, layer: 1, pos: 0
type: B, layer: 1, pos: 21
type: B, layer: 1, pos: 37
type: A, layer: 1, pos: 36
type: B, layer: 1, pos: 7
type: A, layer: 1, pos: 7
type: A, layer: 1, pos: 28
type: A, layer: 1, pos: 6
type: A, layer: 1, pos: 24
type: B, layer: 1, pos: 24
type: B, layer: 1, pos: 36
type: B, layer: 1, pos: 28
type: B, layer: 1, pos: 6
type: A, layer: 1, pos: 21
type: B, layer: 1, pos: 47
type: B, layer: 1, pos: 0

Time for candidate selection: 0.20 seconds

### Candidate
type: A, layer: 1, pos: 0

### Candidate
type: B, layer: 1, pos: 21

### Candidate
type: B, layer: 1, pos: 37

## Relational analysis of NS_A2_B1_A2_A2_B1

### Relational analysis result of NS_A2_B1_A2_A2_B1
Status: Status.UNKNOWN
Output dim: 3, lower bound: -20.5589840, upper bound: 20.7062685
time: 0.63 seconds

## Relational analysis of NS_A2_B1_A2_A2_B2

### Relational analysis result of NS_A2_B1_A2_A2_B2
Status: Status.UNKNOWN
Output dim: 3, lower bound: -20.5589840, upper bound: 20.7062711
time: 0.69 seconds

## BFS NS instance: NS_A2_B2_A1_B1

### Backsubstitution after applying NS history:
0: -102.6300964, 142.3890839, -116.0162811, 160.4569397, -263.0870361, 258.4053650
1: -13.1932011, 12.2476940, -14.8514814, 13.8080673, -27.0012665, 27.0991745
2: -7.5800185, 12.4953012, -8.5386076, 14.1044064, -21.6844215, 21.0339088
3: -6.0822811, 13.7058611, -6.9060659, 15.4590082, -21.5412884, 20.6119270
4: -9.3485603, 11.1931620, -10.5402803, 12.6461687, -21.9947281, 21.7334423

Time for backsubstitution: 2.43 seconds

### NS candidates at layer 1
type: B, layer: 1, pos: 0
type: A, layer: 1, pos: 21
type: B, layer: 1, pos: 7
type: B, layer: 1, pos: 47
type: B, layer: 1, pos: 6
type: B, layer: 1, pos: 28
type: A, layer: 1, pos: 6
type: A, layer: 1, pos: 28
type: A, layer: 1, pos: 36
type: B, layer: 1, pos: 24
type: A, layer: 1, pos: 24
type: B, layer: 1, pos: 21
type: B, layer: 1, pos: 37
type: A, layer: 1, pos: 47
type: A, layer: 1, pos: 0
type: A, layer: 1, pos: 35

Time for candidate selection: 0.20 seconds

### Candidate
type: B, layer: 1, pos: 0

### Candidate
type: A, layer: 1, pos: 21

## Relational analysis of NS_A2_B2_A1_B1_A1

### Relational analysis result of NS_A2_B2_A1_B1_A1
Status: Status.UNKNOWN
Output dim: 3, lower bound: -20.5935250, upper bound: 20.6831312
time: 0.60 seconds

## Relational analysis of NS_A2_B2_A1_B1_A2

### Relational analysis result of NS_A2_B2_A1_B1_A2
Status: Status.VERIFIED
Output dim: 3, lower bound: -20.5935095, upper bound: 20.5718250
time: 0.60 seconds

## BFS NS instance: NS_A2_B2_A1_B2

### Backsubstitution after applying NS history:
0: -102.6300964, 142.3890839, -117.1993408, 161.3668365, -263.9969482, 259.5884399
1: -13.1932011, 12.2476940, -14.9286995, 13.9060850, -27.0992851, 27.1763935
2: -7.5800185, 12.4953012, -8.6000643, 14.2070026, -21.7870197, 21.0953655
3: -6.0822811, 13.7058611, -6.9678402, 15.5446568, -21.6269360, 20.6737022
4: -9.3485603, 11.1931620, -10.6234360, 12.7364960, -22.0850563, 21.8165970

Time for backsubstitution: 2.44 seconds

### NS candidates at layer 1
type: B, layer: 1, pos: 0
type: A, layer: 1, pos: 21
type: B, layer: 1, pos: 7
type: B, layer: 1, pos: 47
type: B, layer: 1, pos: 6
type: A, layer: 1, pos: 6
type: B, layer: 1, pos: 28
type: A, layer: 1, pos: 36
type: A, layer: 1, pos: 28
type: B, layer: 1, pos: 24
type: A, layer: 1, pos: 24
type: B, layer: 1, pos: 21
type: B, layer: 1, pos: 37
type: A, layer: 1, pos: 47
type: A, layer: 1, pos: 0
type: A, layer: 1, pos: 35

Time for candidate selection: 0.20 seconds

### Candidate
type: B, layer: 1, pos: 0

### Candidate
type: A, layer: 1, pos: 21

## Relational analysis of NS_A2_B2_A1_B2_A1

### Relational analysis result of NS_A2_B2_A1_B2_A1
Status: Status.UNKNOWN
Output dim: 3, lower bound: -20.5669690, upper bound: 20.6831258
time: 0.63 seconds

## Relational analysis of NS_A2_B2_A1_B2_A2

### Relational analysis result of NS_A2_B2_A1_B2_A2
Status: Status.VERIFIED
Output dim: 3, lower bound: -20.5669533, upper bound: 20.5718200
time: 0.54 seconds

## BFS NS instance: NS_A2_B2_A2_A1

### Backsubstitution after applying NS history:
0: -95.0689392, 132.8676910, -120.8285675, 166.0415649, -261.1104431, 253.6962585
1: -12.3253155, 11.3997688, -15.3544054, 14.3327837, -26.6581001, 26.7541733
2: -7.0606232, 11.6485138, -8.8637476, 14.6180887, -21.6787109, 20.5122604
3: -5.6620207, 12.7705164, -7.1897345, 16.0026474, -21.6646671, 19.9602470
4: -8.7019453, 10.4130878, -10.9433813, 13.1088514, -21.8107929, 21.3564682

Time for backsubstitution: 2.43 seconds

### NS candidates at layer 1
type: B, layer: 1, pos: 0
type: B, layer: 1, pos: 7
type: B, layer: 1, pos: 47
type: B, layer: 1, pos: 36
type: B, layer: 1, pos: 28
type: A, layer: 1, pos: 28
type: B, layer: 1, pos: 6
type: A, layer: 1, pos: 36
type: B, layer: 1, pos: 24
type: B, layer: 1, pos: 21
type: A, layer: 1, pos: 24
type: B, layer: 1, pos: 37
type: A, layer: 1, pos: 6
type: A, layer: 1, pos: 47
type: A, layer: 1, pos: 0
type: A, layer: 1, pos: 35

Time for candidate selection: 0.20 seconds

### Candidate
type: B, layer: 1, pos: 0

### Candidate
type: B, layer: 1, pos: 7

## Relational analysis of NS_A2_B2_A2_A1_B1

### Relational analysis result of NS_A2_B2_A2_A1_B1
Status: Status.UNKNOWN
Output dim: 3, lower bound: -20.5937064, upper bound: 20.6928908
time: 0.54 seconds

## Relational analysis of NS_A2_B2_A2_A1_B2

### Relational analysis result of NS_A2_B2_A2_A1_B2
Status: Status.UNKNOWN
Output dim: 3, lower bound: -20.5937044, upper bound: 20.6928892
time: 0.55 seconds

## Summary of splitting at layer (split count: 4)
- Time for NS candidates: 3.74 seconds
NS_A1_B1_A2_A1_A1, status: Status.UNKNOWN, split count: 5, time: 3.74
Output dim: 3, lower bound: -20.5300532, upper bound: 20.6343364
NS_A1_B1_A2_A1_A2, status: Status.UNKNOWN, split count: 5, time: 3.74
Output dim: 3, lower bound: -20.5300592, upper bound: 20.6980175
NS_A1_B2_A1_B1_B1, status: Status.UNKNOWN, split count: 5, time: 3.74
Output dim: 3, lower bound: -20.6094994, upper bound: 20.6827621
NS_A1_B2_A1_B1_B2, status: Status.UNKNOWN, split count: 5, time: 3.74
Output dim: 3, lower bound: -20.6033761, upper bound: 20.6827606
NS_A1_B2_A1_B2_B1, status: Status.UNKNOWN, split count: 5, time: 3.74
Output dim: 3, lower bound: -20.5821564, upper bound: 20.6827564
NS_A1_B2_A1_B2_B2, status: Status.UNKNOWN, split count: 5, time: 3.74
Output dim: 3, lower bound: -20.5779195, upper bound: 20.6827564
NS_A1_B2_A2_B1_B1, status: Status.UNKNOWN, split count: 5, time: 3.74
Output dim: 3, lower bound: -20.6129425, upper bound: 20.5705919
NS_A1_B2_A2_B1_B2, status: Status.VERIFIED, split count: 5, time: 3.74
Output dim: 3, lower bound: -20.5833051, upper bound: 20.5705843
NS_A1_B2_A2_B2_B1, status: Status.UNKNOWN, split count: 5, time: 3.74
Output dim: 3, lower bound: -20.6129393, upper bound: 20.5705957
NS_A1_B2_A2_B2_B2, status: Status.VERIFIED, split count: 5, time: 3.74
Output dim: 3, lower bound: -20.5833068, upper bound: 20.5705876
NS_A2_B1_A2_A2_B1, status: Status.UNKNOWN, split count: 5, time: 3.74
Output dim: 3, lower bound: -20.5589840, upper bound: 20.7062685
NS_A2_B1_A2_A2_B2, status: Status.UNKNOWN, split count: 5, time: 3.74
Output dim: 3, lower bound: -20.5589840, upper bound: 20.7062711
NS_A2_B2_A1_B1_A1, status: Status.UNKNOWN, split count: 5, time: 3.74
Output dim: 3, lower bound: -20.5935250, upper bound: 20.6831312
NS_A2_B2_A1_B1_A2, status: Status.VERIFIED, split count: 5, time: 3.74
Output dim: 3, lower bound: -20.5935095, upper bound: 20.5718250
NS_A2_B2_A1_B2_A1, status: Status.UNKNOWN, split count: 5, time: 3.74
Output dim: 3, lower bound: -20.5669690, upper bound: 20.6831258
NS_A2_B2_A1_B2_A2, status: Status.VERIFIED, split count: 5, time: 3.74
Output dim: 3, lower bound: -20.5669533, upper bound: 20.5718200
NS_A2_B2_A2_A1_B1, status: Status.UNKNOWN, split count: 5, time: 3.74
Output dim: 3, lower bound: -20.5937064, upper bound: 20.6928908
NS_A2_B2_A2_A1_B2, status: Status.UNKNOWN, split count: 5, time: 3.74
Output dim: 3, lower bound: -20.5937044, upper bound: 20.6928892

## BFS NS instance: NS_A1_B1_A2_A1_A1

### Backsubstitution after applying NS history:
0: -108.7109680, 150.9740753, -64.4378891, 83.7772598, -192.4881897, 215.4119568
1: -13.9608555, 13.0281258, -7.6196566, 7.6589622, -21.6198177, 20.6477795
2: -8.0301933, 13.2491865, -4.6781135, 7.2810550, -15.3112488, 17.9272995
3: -6.4961653, 14.5776148, -3.8368852, 7.8101020, -14.3062668, 18.4144974
4: -9.9313498, 11.8866148, -5.8599577, 6.5999441, -16.5312939, 17.7465668

Time for backsubstitution: 2.43 seconds

### NS candidates at layer 1
type: A, layer: 1, pos: 0
type: B, layer: 1, pos: 21
type: A, layer: 1, pos: 47
type: B, layer: 1, pos: 37
type: B, layer: 1, pos: 7
type: A, layer: 1, pos: 7
type: A, layer: 1, pos: 24
type: B, layer: 1, pos: 24
type: B, layer: 1, pos: 6
type: A, layer: 1, pos: 6
type: B, layer: 1, pos: 28
type: B, layer: 1, pos: 36
type: B, layer: 1, pos: 47
type: A, layer: 1, pos: 21
type: B, layer: 1, pos: 0

Time for candidate selection: 0.20 seconds

### Candidate
type: A, layer: 1, pos: 0

### Candidate
type: B, layer: 1, pos: 21

### Candidate
type: A, layer: 1, pos: 47

### Candidate
type: B, layer: 1, pos: 37

## Relational analysis of NS_A1_B1_A2_A1_A1_B1

### Relational analysis result of NS_A1_B1_A2_A1_A1_B1
Status: Status.UNKNOWN
Output dim: 3, lower bound: -20.5228503, upper bound: 20.6343366
time: 0.59 seconds

## Relational analysis of NS_A1_B1_A2_A1_A1_B2

### Relational analysis result of NS_A1_B1_A2_A1_A1_B2
Status: Status.UNKNOWN
Output dim: 3, lower bound: -20.5228503, upper bound: 20.6343359
time: 0.56 seconds

## BFS NS instance: NS_A1_B1_A2_A1_A2

### Backsubstitution after applying NS history:
0: -106.4225693, 147.9249115, -64.4378891, 83.7772598, -190.1998138, 212.3627930
1: -13.6897011, 12.7627611, -7.6196566, 7.6589622, -21.3486595, 20.3824177
2: -7.8642807, 12.9812555, -4.6781135, 7.2810550, -15.1453352, 17.6593666
3: -6.3524604, 14.2819500, -3.8368852, 7.8101020, -14.1625614, 18.1188297
4: -9.7289190, 11.6428547, -5.8599577, 6.5999441, -16.3288631, 17.5028095

Time for backsubstitution: 2.44 seconds

### NS candidates at layer 1
type: A, layer: 1, pos: 0
type: B, layer: 1, pos: 21
type: A, layer: 1, pos: 47
type: B, layer: 1, pos: 7
type: B, layer: 1, pos: 37
type: A, layer: 1, pos: 7
type: A, layer: 1, pos: 24
type: B, layer: 1, pos: 24
type: B, layer: 1, pos: 6
type: A, layer: 1, pos: 6
type: B, layer: 1, pos: 28
type: B, layer: 1, pos: 36
type: B, layer: 1, pos: 47
type: A, layer: 1, pos: 21
type: B, layer: 1, pos: 0

Time for candidate selection: 0.20 seconds

### Candidate
type: A, layer: 1, pos: 0

### Candidate
type: B, layer: 1, pos: 21

### Candidate
type: A, layer: 1, pos: 47

### Candidate
type: B, layer: 1, pos: 7

### Candidate
type: B, layer: 1, pos: 37

## Relational analysis of NS_A1_B1_A2_A1_A2_B1

### Relational analysis result of NS_A1_B1_A2_A1_A2_B1
Status: Status.UNKNOWN
Output dim: 3, lower bound: -20.5228566, upper bound: 20.6980165
time: 0.53 seconds

## Relational analysis of NS_A1_B1_A2_A1_A2_B2

### Relational analysis result of NS_A1_B1_A2_A1_A2_B2
Status: Status.UNKNOWN
Output dim: 3, lower bound: -20.5228564, upper bound: 20.6980185
time: 0.62 seconds

## BFS NS instance: NS_A1_B2_A1_B1_B1

### Backsubstitution after applying NS history:
0: -107.4569092, 149.4062958, -113.3619614, 156.8386383, -264.2955017, 262.7682190
1: -13.8525238, 12.8498917, -14.5068369, 13.5073509, -27.3598728, 27.3567200
2: -7.9272017, 13.1349926, -8.3509645, 13.7784405, -21.7056389, 21.4859543
3: -6.3961639, 14.4240589, -6.7625446, 15.1243000, -21.5204639, 21.1866035
4: -9.8063383, 11.7648029, -10.3205271, 12.3568792, -22.1632156, 22.0853271

Time for backsubstitution: 2.44 seconds

### NS candidates at layer 1
type: A, layer: 1, pos: 7
type: B, layer: 1, pos: 7
type: B, layer: 1, pos: 0
type: A, layer: 1, pos: 35
type: A, layer: 1, pos: 0
type: A, layer: 1, pos: 28
type: A, layer: 1, pos: 36
type: B, layer: 1, pos: 24
type: B, layer: 1, pos: 47
type: A, layer: 1, pos: 6
type: A, layer: 1, pos: 24
type: B, layer: 1, pos: 6
type: B, layer: 1, pos: 21
type: A, layer: 1, pos: 47
type: B, layer: 1, pos: 37

Time for candidate selection: 0.20 seconds

### Candidate
type: A, layer: 1, pos: 7

### Candidate
type: B, layer: 1, pos: 7

### Candidate
type: B, layer: 1, pos: 0

### Candidate
type: A, layer: 1, pos: 35

## Relational analysis of NS_A1_B2_A1_B1_B1_A1
Optimization infeasible because this subproblem isn't reachable.

### Candidate
type: A, layer: 1, pos: 0

### Candidate
type: A, layer: 1, pos: 28

## Relational analysis of NS_A1_B2_A1_B1_B1_A1
Optimization infeasible because this subproblem isn't reachable.

### Candidate
type: A, layer: 1, pos: 36

## Relational analysis of NS_A1_B2_A1_B1_B1_A1

### Relational analysis result of NS_A1_B2_A1_B1_B1_A1
Status: Status.UNKNOWN
Output dim: 3, lower bound: -20.6095014, upper bound: 20.6827627
time: 0.57 seconds

## Relational analysis of NS_A1_B2_A1_B1_B1_A2

### Relational analysis result of NS_A1_B2_A1_B1_B1_A2
Status: Status.UNKNOWN
Output dim: 3, lower bound: -20.6095014, upper bound: 20.6827588
time: 0.55 seconds

## BFS NS instance: NS_A1_B2_A1_B1_B2

### Backsubstitution after applying NS history:
0: -107.4569092, 149.4062958, -112.4632492, 155.7105408, -263.1673584, 261.8694763
1: -13.8525238, 12.8498917, -14.4164772, 13.4019489, -27.2544727, 27.2663670
2: -7.9272017, 13.1349926, -8.2836676, 13.6824551, -21.6096573, 21.4186592
3: -6.3961639, 14.4240589, -6.6981082, 15.0158939, -21.4120579, 21.1221676
4: -9.8063383, 11.7648029, -10.2392101, 12.2670708, -22.0734081, 22.0040092

Time for backsubstitution: 2.45 seconds

### NS candidates at layer 1
type: A, layer: 1, pos: 7
type: B, layer: 1, pos: 7
type: A, layer: 1, pos: 35
type: A, layer: 1, pos: 0
type: B, layer: 1, pos: 0
type: A, layer: 1, pos: 28
type: A, layer: 1, pos: 36
type: B, layer: 1, pos: 24
type: A, layer: 1, pos: 6
type: B, layer: 1, pos: 47
type: A, layer: 1, pos: 24
type: B, layer: 1, pos: 21
type: B, layer: 1, pos: 6
type: A, layer: 1, pos: 47
type: B, layer: 1, pos: 37

Time for candidate selection: 0.20 seconds

### Candidate
type: A, layer: 1, pos: 7

### Candidate
type: B, layer: 1, pos: 7

### Candidate
type: A, layer: 1, pos: 35

## Relational analysis of NS_A1_B2_A1_B1_B2_A1
Optimization infeasible because this subproblem isn't reachable.

### Candidate
type: A, layer: 1, pos: 0

### Candidate
type: B, layer: 1, pos: 0

### Candidate
type: A, layer: 1, pos: 28

## Relational analysis of NS_A1_B2_A1_B1_B2_A1
Optimization infeasible because this subproblem isn't reachable.

### Candidate
type: A, layer: 1, pos: 36

## Relational analysis of NS_A1_B2_A1_B1_B2_A1

### Relational analysis result of NS_A1_B2_A1_B1_B2_A1
Status: Status.UNKNOWN
Output dim: 3, lower bound: -20.6033760, upper bound: 20.6827614
time: 0.70 seconds

## Relational analysis of NS_A1_B2_A1_B1_B2_A2

### Relational analysis result of NS_A1_B2_A1_B1_B2_A2
Status: Status.UNKNOWN
Output dim: 3, lower bound: -20.6033760, upper bound: 20.6827558
time: 0.62 seconds

## BFS NS instance: NS_A1_B2_A1_B2_B1

### Backsubstitution after applying NS history:
0: -107.4569092, 149.4062958, -115.7473679, 159.5204010, -266.9772949, 265.1536560
1: -13.8525238, 12.8498917, -14.7541456, 13.7527189, -27.6052399, 27.6040344
2: -7.9272017, 13.1349926, -8.5021629, 14.0355921, -21.9627934, 21.6371536
3: -6.3961639, 14.4240589, -6.8954983, 15.3830976, -21.7792606, 21.3195572
4: -9.8063383, 11.7648029, -10.5128937, 12.5846434, -22.3909817, 22.2776966

Time for backsubstitution: 2.45 seconds

### NS candidates at layer 1
type: A, layer: 1, pos: 7
type: B, layer: 1, pos: 0
type: B, layer: 1, pos: 7
type: B, layer: 1, pos: 47
type: A, layer: 1, pos: 28
type: A, layer: 1, pos: 36
type: B, layer: 1, pos: 24
type: B, layer: 1, pos: 6
type: A, layer: 1, pos: 6
type: B, layer: 1, pos: 21
type: A, layer: 1, pos: 24
type: A, layer: 1, pos: 35
type: A, layer: 1, pos: 0
type: A, layer: 1, pos: 47
type: B, layer: 1, pos: 37

Time for candidate selection: 0.20 seconds

### Candidate
type: A, layer: 1, pos: 7

### Candidate
type: B, layer: 1, pos: 0

### Candidate
type: B, layer: 1, pos: 7

### Candidate
type: B, layer: 1, pos: 47

## Relational analysis of NS_A1_B2_A1_B2_B1_B1

### Relational analysis result of NS_A1_B2_A1_B2_B1_B1
Status: Status.UNKNOWN
Output dim: 3, lower bound: -20.4844757, upper bound: 20.6659401
time: 0.56 seconds

## Relational analysis of NS_A1_B2_A1_B2_B1_B2

### Relational analysis result of NS_A1_B2_A1_B2_B1_B2
Status: Status.UNKNOWN
Output dim: 3, lower bound: -20.5819093, upper bound: 20.6827172
time: 0.56 seconds

## BFS NS instance: NS_A1_B2_A1_B2_B2

### Backsubstitution after applying NS history:
0: -107.4569092, 149.4062958, -112.9994278, 155.6864014, -263.1432495, 262.4057312
1: -13.8525238, 12.8498917, -14.4079819, 13.4225550, -27.2750778, 27.2578697
2: -7.9272017, 13.1349926, -8.2972803, 13.7014332, -21.6286335, 21.4322739
3: -6.3961639, 14.4240589, -6.7209225, 15.0139751, -21.4101391, 21.1449814
4: -9.8063383, 11.7648029, -10.2640667, 12.2832479, -22.0895844, 22.0288696

Time for backsubstitution: 2.45 seconds

### NS candidates at layer 1
type: A, layer: 1, pos: 7
type: B, layer: 1, pos: 7
type: B, layer: 1, pos: 0
type: A, layer: 1, pos: 35
type: A, layer: 1, pos: 0
type: A, layer: 1, pos: 28
type: A, layer: 1, pos: 36
type: B, layer: 1, pos: 47
type: B, layer: 1, pos: 21
type: A, layer: 1, pos: 6
type: B, layer: 1, pos: 6
type: B, layer: 1, pos: 24
type: A, layer: 1, pos: 24
type: A, layer: 1, pos: 47
type: B, layer: 1, pos: 37

Time for candidate selection: 0.20 seconds

### Candidate
type: A, layer: 1, pos: 7

### Candidate
type: B, layer: 1, pos: 7

### Candidate
type: B, layer: 1, pos: 0

### Candidate
type: A, layer: 1, pos: 35

## Relational analysis of NS_A1_B2_A1_B2_B2_A1
Optimization infeasible because this subproblem isn't reachable.

### Candidate
type: A, layer: 1, pos: 0

### Candidate
type: A, layer: 1, pos: 28

## Relational analysis of NS_A1_B2_A1_B2_B2_A1
Optimization infeasible because this subproblem isn't reachable.

### Candidate
type: A, layer: 1, pos: 36

## Relational analysis of NS_A1_B2_A1_B2_B2_A1

### Relational analysis result of NS_A1_B2_A1_B2_B2_A1
Status: Status.UNKNOWN
Output dim: 3, lower bound: -20.5779165, upper bound: 20.6827574
time: 0.66 seconds

## Relational analysis of NS_A1_B2_A1_B2_B2_A2

### Relational analysis result of NS_A1_B2_A1_B2_B2_A2
Status: Status.UNKNOWN
Output dim: 3, lower bound: -20.5779194, upper bound: 20.6827566
time: 0.55 seconds

## BFS NS instance: NS_A1_B2_A2_B1_B1

### Backsubstitution after applying NS history:
0: -127.9776459, 179.3025665, -107.7582932, 149.9104309, -277.8880615, 287.0607605
1: -16.6276455, 15.3680820, -13.9115181, 12.8534250, -29.4810715, 29.2796001
2: -9.4743786, 15.7429419, -7.9473271, 13.1672535, -22.6416302, 23.6902695
3: -7.6515388, 17.2871590, -6.4039869, 14.4634514, -22.1149902, 23.6911449
4: -11.7098789, 14.1008635, -9.8263731, 11.7925348, -23.5024109, 23.9272366

Time for backsubstitution: 2.45 seconds

### NS candidates at layer 1
type: A, layer: 1, pos: 35
type: A, layer: 1, pos: 0
type: A, layer: 1, pos: 7
type: B, layer: 1, pos: 7
type: B, layer: 1, pos: 28
type: A, layer: 1, pos: 28
type: A, layer: 1, pos: 6
type: A, layer: 1, pos: 36
type: B, layer: 1, pos: 24
type: A, layer: 1, pos: 24
type: B, layer: 1, pos: 6
type: A, layer: 1, pos: 47
type: B, layer: 1, pos: 47
type: B, layer: 1, pos: 37
type: B, layer: 1, pos: 0

Time for candidate selection: 0.20 seconds

### Candidate
type: A, layer: 1, pos: 35

### Candidate
type: A, layer: 1, pos: 0

### Candidate
type: A, layer: 1, pos: 7

### Candidate
type: B, layer: 1, pos: 7

### Candidate
type: B, layer: 1, pos: 28

## Relational analysis of NS_A1_B2_A2_B1_B1_B1

### Relational analysis result of NS_A1_B2_A2_B1_B1_B1
Status: Status.UNKNOWN
Output dim: 3, lower bound: -20.7036463, upper bound: 20.5688187
time: 0.65 seconds

## Relational analysis of NS_A1_B2_A2_B1_B1_B2

### Relational analysis result of NS_A1_B2_A2_B1_B1_B2
Status: Status.UNKNOWN
Output dim: 3, lower bound: -20.7024975, upper bound: 20.5688147
time: 0.60 seconds

## BFS NS instance: NS_A1_B2_A2_B2_B1

### Backsubstitution after applying NS history:
0: -127.9776459, 179.3025665, -128.6947479, 180.3825989, -308.3602295, 307.9972839
1: -16.6276455, 15.3680820, -16.7398109, 15.4197369, -32.0473824, 32.1078873
2: -9.4743786, 15.7429419, -9.5231218, 15.8290501, -25.3034286, 25.2660580
3: -7.6515388, 17.2871590, -7.6825891, 17.3809319, -25.0324707, 24.9697437
4: -11.7098789, 14.1008635, -11.7697344, 14.1758413, -25.8857193, 25.8705978

Time for backsubstitution: 2.46 seconds

### NS candidates at layer 1
type: A, layer: 1, pos: 7
type: B, layer: 1, pos: 7
type: A, layer: 1, pos: 0
type: A, layer: 1, pos: 35
type: B, layer: 1, pos: 28
type: B, layer: 1, pos: 0
type: A, layer: 1, pos: 28
type: A, layer: 1, pos: 6
type: B, layer: 1, pos: 6
type: B, layer: 1, pos: 24
type: A, layer: 1, pos: 24
type: A, layer: 1, pos: 36
type: B, layer: 1, pos: 47
type: A, layer: 1, pos: 47
type: B, layer: 1, pos: 37

Time for candidate selection: 0.21 seconds

### Candidate
type: A, layer: 1, pos: 7

### Candidate
type: B, layer: 1, pos: 7

### Candidate
type: A, layer: 1, pos: 0

### Candidate
type: A, layer: 1, pos: 35

## Relational analysis of NS_A1_B2_A2_B2_B1_A1

### Relational analysis result of NS_A1_B2_A2_B2_B1_A1
Status: Status.VERIFIED
Output dim: 3, lower bound: -20.5629565, upper bound: 20.5241827
time: 0.59 seconds

## Relational analysis of NS_A1_B2_A2_B2_B1_A2

### Relational analysis result of NS_A1_B2_A2_B2_B1_A2
Status: Status.VERIFIED
Output dim: 3, lower bound: -20.5629606, upper bound: 20.5241807
time: 0.54 seconds

## BFS NS instance: NS_A2_B1_A2_A2_B1

### Backsubstitution after applying NS history:
0: -113.3691940, 156.9696960, -60.7692795, 78.7800903, -192.1492920, 217.7389679
1: -14.5364876, 13.5001736, -7.1476021, 7.2403092, -21.7767963, 20.6477718
2: -8.3368378, 13.7978535, -4.4023685, 6.8436527, -15.1804905, 18.2002220
3: -6.7319307, 15.1299477, -3.6247139, 7.3066416, -14.0385723, 18.7546616
4: -10.3006001, 12.3681145, -5.5420227, 6.2030392, -16.5036392, 17.9101372

Time for backsubstitution: 2.47 seconds

### NS candidates at layer 1
type: A, layer: 1, pos: 0
type: B, layer: 1, pos: 21
type: A, layer: 1, pos: 36
type: A, layer: 1, pos: 6
type: B, layer: 1, pos: 36
type: A, layer: 1, pos: 24
type: A, layer: 1, pos: 7
type: B, layer: 1, pos: 24
type: A, layer: 1, pos: 28
type: B, layer: 1, pos: 28
type: B, layer: 1, pos: 7
type: B, layer: 1, pos: 6
type: A, layer: 1, pos: 21
type: B, layer: 1, pos: 47
type: B, layer: 1, pos: 0

Time for candidate selection: 0.20 seconds

### Candidate
type: A, layer: 1, pos: 0

### Candidate
type: B, layer: 1, pos: 21

### Candidate
type: A, layer: 1, pos: 36

## Relational analysis of NS_A2_B1_A2_A2_B1_A1

### Relational analysis result of NS_A2_B1_A2_A2_B1_A1
Status: Status.UNKNOWN
Output dim: 3, lower bound: -20.5248911, upper bound: 20.7062633
time: 0.72 seconds

## Relational analysis of NS_A2_B1_A2_A2_B1_A2

### Relational analysis result of NS_A2_B1_A2_A2_B1_A2
Status: Status.UNKNOWN
Output dim: 3, lower bound: -20.5248904, upper bound: 20.6899470
time: 0.79 seconds

## BFS NS instance: NS_A2_B1_A2_A2_B2

### Backsubstitution after applying NS history:
0: -113.3691940, 156.9696960, -59.1519318, 76.8340149, -190.2032166, 216.1216278
1: -14.5364876, 13.5001736, -6.9958315, 7.0370650, -21.5735531, 20.4960003
2: -8.3368378, 13.7978535, -4.2961965, 6.6580710, -14.9949093, 18.0940495
3: -6.7319307, 15.1299477, -3.5116916, 7.1283474, -13.8602781, 18.6416359
4: -10.3006001, 12.3681145, -5.3880043, 6.0388904, -16.3394909, 17.7561188

Time for backsubstitution: 2.47 seconds

### NS candidates at layer 1
type: A, layer: 1, pos: 0
type: B, layer: 1, pos: 21
type: A, layer: 1, pos: 36
type: A, layer: 1, pos: 7
type: A, layer: 1, pos: 6
type: A, layer: 1, pos: 24
type: A, layer: 1, pos: 28
type: B, layer: 1, pos: 24
type: B, layer: 1, pos: 36
type: B, layer: 1, pos: 7
type: B, layer: 1, pos: 28
type: B, layer: 1, pos: 6
type: B, layer: 1, pos: 47
type: A, layer: 1, pos: 21
type: B, layer: 1, pos: 0

Time for candidate selection: 0.21 seconds

### Candidate
type: A, layer: 1, pos: 0

### Candidate
type: B, layer: 1, pos: 21

### Candidate
type: A, layer: 1, pos: 36

## Relational analysis of NS_A2_B1_A2_A2_B2_A1

### Relational analysis result of NS_A2_B1_A2_A2_B2_A1
Status: Status.UNKNOWN
Output dim: 3, lower bound: -20.5248911, upper bound: 20.7062664
time: 0.56 seconds

## Relational analysis of NS_A2_B1_A2_A2_B2_A2

### Relational analysis result of NS_A2_B1_A2_A2_B2_A2
Status: Status.UNKNOWN
Output dim: 3, lower bound: -20.5248861, upper bound: 20.6899493
time: 0.64 seconds

## BFS NS instance: NS_A2_B2_A1_B1_A1

### Backsubstitution after applying NS history:
0: -95.0735168, 132.7543030, -116.0162811, 160.4569397, -255.5304565, 248.7705536
1: -12.3337259, 11.3693285, -14.8514814, 13.8080673, -26.1417923, 26.2208061
2: -7.0452828, 11.6368580, -8.5386076, 14.1044064, -21.1496849, 20.1754646
3: -5.6269884, 12.7925835, -6.9060659, 15.4590082, -21.0859966, 19.6986427
4: -8.6930714, 10.4093132, -10.5402803, 12.6461687, -21.3392410, 20.9495926

Time for backsubstitution: 2.47 seconds

### NS candidates at layer 1
type: B, layer: 1, pos: 0
type: B, layer: 1, pos: 7
type: B, layer: 1, pos: 47
type: B, layer: 1, pos: 6
type: B, layer: 1, pos: 28
type: A, layer: 1, pos: 36
type: A, layer: 1, pos: 28
type: A, layer: 1, pos: 6
type: B, layer: 1, pos: 24
type: A, layer: 1, pos: 24
type: B, layer: 1, pos: 21
type: B, layer: 1, pos: 37
type: A, layer: 1, pos: 47
type: A, layer: 1, pos: 0
type: A, layer: 1, pos: 35

Time for candidate selection: 0.21 seconds

### Candidate
type: B, layer: 1, pos: 0

### Candidate
type: B, layer: 1, pos: 7

## Relational analysis of NS_A2_B2_A1_B1_A1_B1

### Relational analysis result of NS_A2_B2_A1_B1_A1_B1
Status: Status.UNKNOWN
Output dim: 3, lower bound: -20.5935213, upper bound: 20.6831342
time: 0.64 seconds

## Relational analysis of NS_A2_B2_A1_B1_A1_B2

### Relational analysis result of NS_A2_B2_A1_B1_A1_B2
Status: Status.UNKNOWN
Output dim: 3, lower bound: -20.5935213, upper bound: 20.6831331
time: 0.62 seconds

## BFS NS instance: NS_A2_B2_A1_B2_A1

### Backsubstitution after applying NS history:
0: -95.0735168, 132.7543030, -117.1993408, 161.3668365, -256.4403687, 249.9536438
1: -12.3337259, 11.3693285, -14.9286995, 13.9060850, -26.2398109, 26.2980270
2: -7.0452828, 11.6368580, -8.6000643, 14.2070026, -21.2522850, 20.2369232
3: -5.6269884, 12.7925835, -6.9678402, 15.5446568, -21.1716461, 19.7604198
4: -8.6930714, 10.4093132, -10.6234360, 12.7364960, -21.4295673, 21.0327492

Time for backsubstitution: 2.48 seconds

### NS candidates at layer 1
type: B, layer: 1, pos: 0
type: B, layer: 1, pos: 7
type: B, layer: 1, pos: 47
type: B, layer: 1, pos: 6
type: B, layer: 1, pos: 28
type: A, layer: 1, pos: 36
type: A, layer: 1, pos: 28
type: A, layer: 1, pos: 6
type: B, layer: 1, pos: 24
type: A, layer: 1, pos: 24
type: B, layer: 1, pos: 21
type: B, layer: 1, pos: 37
type: A, layer: 1, pos: 47
type: A, layer: 1, pos: 0
type: A, layer: 1, pos: 35

Time for candidate selection: 0.20 seconds

### Candidate
type: B, layer: 1, pos: 0

### Candidate
type: B, layer: 1, pos: 7

## Relational analysis of NS_A2_B2_A1_B2_A1_B1

### Relational analysis result of NS_A2_B2_A1_B2_A1_B1
Status: Status.UNKNOWN
Output dim: 3, lower bound: -20.5664678, upper bound: 20.6831259
time: 0.63 seconds

## Relational analysis of NS_A2_B2_A1_B2_A1_B2

### Relational analysis result of NS_A2_B2_A1_B2_A1_B2
Status: Status.UNKNOWN
Output dim: 3, lower bound: -20.5664677, upper bound: 20.6831276
time: 0.63 seconds

## BFS NS instance: NS_A2_B2_A2_A1_B1

### Backsubstitution after applying NS history:
0: -95.0689392, 132.8676910, -109.1594696, 150.3858337, -245.4547729, 242.0271454
1: -12.3253155, 11.3997688, -13.9188223, 12.9745989, -25.2999153, 25.3185883
2: -7.0606232, 11.6485138, -8.0244865, 13.2249870, -20.2856102, 19.6729984
3: -5.6620207, 12.7705164, -6.4771557, 14.4798832, -20.1419029, 19.2476692
4: -8.7019453, 10.4130878, -9.9148054, 11.8508644, -20.5528069, 20.3278923

Time for backsubstitution: 2.49 seconds

### NS candidates at layer 1
type: B, layer: 1, pos: 0
type: B, layer: 1, pos: 28
type: A, layer: 1, pos: 28
type: B, layer: 1, pos: 37
type: B, layer: 1, pos: 36
type: B, layer: 1, pos: 47
type: B, layer: 1, pos: 21
type: B, layer: 1, pos: 6
type: A, layer: 1, pos: 36
type: B, layer: 1, pos: 24
type: A, layer: 1, pos: 24
type: A, layer: 1, pos: 6
type: A, layer: 1, pos: 47
type: A, layer: 1, pos: 0
type: A, layer: 1, pos: 35

Time for candidate selection: 0.21 seconds

### Candidate
type: B, layer: 1, pos: 0

### Candidate
type: B, layer: 1, pos: 28

## Relational analysis of NS_A2_B2_A2_A1_B1_B1

### Relational analysis result of NS_A2_B2_A2_A1_B1_B1
Status: Status.UNKNOWN
Output dim: 3, lower bound: -20.5910108, upper bound: 20.6804304
time: 0.68 seconds

## Relational analysis of NS_A2_B2_A2_A1_B1_B2

### Relational analysis result of NS_A2_B2_A2_A1_B1_B2
Status: Status.UNKNOWN
Output dim: 3, lower bound: -20.5654292, upper bound: 20.6804245
time: 0.63 seconds

## BFS NS instance: NS_A2_B2_A2_A1_B2

### Backsubstitution after applying NS history:
0: -95.0689392, 132.8676910, -108.0869522, 149.2170715, -244.2860107, 240.9546356
1: -12.3253155, 11.3997688, -13.7921286, 12.8874111, -25.2127266, 25.1918964
2: -7.0606232, 11.6485138, -7.9716330, 13.1191387, -20.1797619, 19.6201458
3: -5.6620207, 12.7705164, -6.4524317, 14.3341846, -19.9962025, 19.2229462
4: -8.7019453, 10.4130878, -9.8296480, 11.7491283, -20.4510727, 20.2427349

Time for backsubstitution: 2.49 seconds

### NS candidates at layer 1
type: B, layer: 1, pos: 0
type: B, layer: 1, pos: 28
type: A, layer: 1, pos: 28
type: B, layer: 1, pos: 21
type: B, layer: 1, pos: 47
type: B, layer: 1, pos: 36
type: B, layer: 1, pos: 6
type: B, layer: 1, pos: 37
type: A, layer: 1, pos: 36
type: A, layer: 1, pos: 6
type: B, layer: 1, pos: 24
type: A, layer: 1, pos: 24
type: A, layer: 1, pos: 47
type: A, layer: 1, pos: 0
type: A, layer: 1, pos: 35

Time for candidate selection: 0.21 seconds

### Candidate
type: B, layer: 1, pos: 0

### Candidate
type: B, layer: 1, pos: 28

## Relational analysis of NS_A2_B2_A2_A1_B2_B1

### Relational analysis result of NS_A2_B2_A2_A1_B2_B1
Status: Status.UNKNOWN
Output dim: 3, lower bound: -20.5910148, upper bound: 20.6804308
time: 0.65 seconds

## Relational analysis of NS_A2_B2_A2_A1_B2_B2

### Relational analysis result of NS_A2_B2_A2_A1_B2_B2
Status: Status.UNKNOWN
Output dim: 3, lower bound: -20.5654292, upper bound: 20.6804270
time: 0.58 seconds

## Summary of splitting at layer (split count: 5)
- Time for NS candidates: 3.94 seconds
NS_A1_B1_A2_A1_A1_B1, status: Status.UNKNOWN, split count: 6, time: 3.94
Output dim: 3, lower bound: -20.5228503, upper bound: 20.6343366
NS_A1_B1_A2_A1_A1_B2, status: Status.UNKNOWN, split count: 6, time: 3.94
Output dim: 3, lower bound: -20.5228503, upper bound: 20.6343359
NS_A1_B1_A2_A1_A2_B1, status: Status.UNKNOWN, split count: 6, time: 3.94
Output dim: 3, lower bound: -20.5228566, upper bound: 20.6980165
NS_A1_B1_A2_A1_A2_B2, status: Status.UNKNOWN, split count: 6, time: 3.94
Output dim: 3, lower bound: -20.5228564, upper bound: 20.6980185
NS_A1_B2_A1_B1_B1_A1, status: Status.UNKNOWN, split count: 6, time: 3.94
Output dim: 3, lower bound: -20.6095014, upper bound: 20.6827627
NS_A1_B2_A1_B1_B1_A2, status: Status.UNKNOWN, split count: 6, time: 3.94
Output dim: 3, lower bound: -20.6095014, upper bound: 20.6827588
NS_A1_B2_A1_B1_B2_A1, status: Status.UNKNOWN, split count: 6, time: 3.94
Output dim: 3, lower bound: -20.6033760, upper bound: 20.6827614
NS_A1_B2_A1_B1_B2_A2, status: Status.UNKNOWN, split count: 6, time: 3.94
Output dim: 3, lower bound: -20.6033760, upper bound: 20.6827558
NS_A1_B2_A1_B2_B1_B1, status: Status.UNKNOWN, split count: 6, time: 3.94
Output dim: 3, lower bound: -20.4844757, upper bound: 20.6659401
NS_A1_B2_A1_B2_B1_B2, status: Status.UNKNOWN, split count: 6, time: 3.94
Output dim: 3, lower bound: -20.5819093, upper bound: 20.6827172
NS_A1_B2_A1_B2_B2_A1, status: Status.UNKNOWN, split count: 6, time: 3.94
Output dim: 3, lower bound: -20.5779165, upper bound: 20.6827574
NS_A1_B2_A1_B2_B2_A2, status: Status.UNKNOWN, split count: 6, time: 3.94
Output dim: 3, lower bound: -20.5779194, upper bound: 20.6827566
NS_A1_B2_A2_B1_B1_B1, status: Status.UNKNOWN, split count: 6, time: 3.94
Output dim: 3, lower bound: -20.7036463, upper bound: 20.5688187
NS_A1_B2_A2_B1_B1_B2, status: Status.UNKNOWN, split count: 6, time: 3.94
Output dim: 3, lower bound: -20.7024975, upper bound: 20.5688147
NS_A1_B2_A2_B2_B1_A1, status: Status.VERIFIED, split count: 6, time: 3.94
Output dim: 3, lower bound: -20.5629565, upper bound: 20.5241827
NS_A1_B2_A2_B2_B1_A2, status: Status.VERIFIED, split count: 6, time: 3.94
Output dim: 3, lower bound: -20.5629606, upper bound: 20.5241807
NS_A2_B1_A2_A2_B1_A1, status: Status.UNKNOWN, split count: 6, time: 3.94
Output dim: 3, lower bound: -20.5248911, upper bound: 20.7062633
NS_A2_B1_A2_A2_B1_A2, status: Status.UNKNOWN, split count: 6, time: 3.94
Output dim: 3, lower bound: -20.5248904, upper bound: 20.6899470
NS_A2_B1_A2_A2_B2_A1, status: Status.UNKNOWN, split count: 6, time: 3.94
Output dim: 3, lower bound: -20.5248911, upper bound: 20.7062664
NS_A2_B1_A2_A2_B2_A2, status: Status.UNKNOWN, split count: 6, time: 3.94
Output dim: 3, lower bound: -20.5248861, upper bound: 20.6899493
NS_A2_B2_A1_B1_A1_B1, status: Status.UNKNOWN, split count: 6, time: 3.94
Output dim: 3, lower bound: -20.5935213, upper bound: 20.6831342
NS_A2_B2_A1_B1_A1_B2, status: Status.UNKNOWN, split count: 6, time: 3.94
Output dim: 3, lower bound: -20.5935213, upper bound: 20.6831331
NS_A2_B2_A1_B2_A1_B1, status: Status.UNKNOWN, split count: 6, time: 3.94
Output dim: 3, lower bound: -20.5664678, upper bound: 20.6831259
NS_A2_B2_A1_B2_A1_B2, status: Status.UNKNOWN, split count: 6, time: 3.94
Output dim: 3, lower bound: -20.5664677, upper bound: 20.6831276
NS_A2_B2_A2_A1_B1_B1, status: Status.UNKNOWN, split count: 6, time: 3.94
Output dim: 3, lower bound: -20.5910108, upper bound: 20.6804304
NS_A2_B2_A2_A1_B1_B2, status: Status.UNKNOWN, split count: 6, time: 3.94
Output dim: 3, lower bound: -20.5654292, upper bound: 20.6804245
NS_A2_B2_A2_A1_B2_B1, status: Status.UNKNOWN, split count: 6, time: 3.94
Output dim: 3, lower bound: -20.5910148, upper bound: 20.6804308
NS_A2_B2_A2_A1_B2_B2, status: Status.UNKNOWN, split count: 6, time: 3.94
Output dim: 3, lower bound: -20.5654292, upper bound: 20.6804270

## BFS NS instance: NS_A1_B1_A2_A1_A1_B1

### Backsubstitution after applying NS history:
0: -108.7109680, 150.9740753, -60.7692795, 78.7800903, -187.4910431, 211.7433472
1: -13.9608555, 13.0281258, -7.1476021, 7.2403092, -21.2011642, 20.1757240
2: -8.0301933, 13.2491865, -4.4023685, 6.8436527, -14.8738461, 17.6515541
3: -6.4961653, 14.5776148, -3.6247139, 7.3066416, -13.8028069, 18.2023277
4: -9.9313498, 11.8866148, -5.5420227, 6.2030392, -16.1343880, 17.4286385

Time for backsubstitution: 2.49 seconds

### NS candidates at layer 1
type: A, layer: 1, pos: 0
type: B, layer: 1, pos: 21
type: A, layer: 1, pos: 47
type: A, layer: 1, pos: 7
type: A, layer: 1, pos: 6
type: A, layer: 1, pos: 24
type: B, layer: 1, pos: 24
type: B, layer: 1, pos: 7
type: B, layer: 1, pos: 36
type: B, layer: 1, pos: 28
type: B, layer: 1, pos: 6
type: B, layer: 1, pos: 47
type: A, layer: 1, pos: 21
type: B, layer: 1, pos: 0

Time for candidate selection: 0.20 seconds

### Candidate
type: A, layer: 1, pos: 0

### Candidate
type: B, layer: 1, pos: 21

### Candidate
type: A, layer: 1, pos: 47

### Candidate
type: A, layer: 1, pos: 7

### Candidate
type: A, layer: 1, pos: 6

## Relational analysis of NS_A1_B1_A2_A1_A1_B1_A1
Optimization infeasible because this subproblem isn't reachable.

### Candidate
type: A, layer: 1, pos: 24

## Relational analysis of NS_A1_B1_A2_A1_A1_B1_A1

### Relational analysis result of NS_A1_B1_A2_A1_A1_B1_A1
Status: Status.UNKNOWN
Output dim: 3, lower bound: -20.5018316, upper bound: 20.6317517
time: 0.65 seconds

## Relational analysis of NS_A1_B1_A2_A1_A1_B1_A2
Optimization infeasible because this subproblem isn't reachable.

### Candidate
type: B, layer: 1, pos: 24

### Candidate
type: B, layer: 1, pos: 7

### Candidate
type: B, layer: 1, pos: 36

## Relational analysis of NS_A1_B1_A2_A1_A1_B1_B1

### Relational analysis result of NS_A1_B1_A2_A1_A1_B1_B1
Status: Status.UNKNOWN
Output dim: 3, lower bound: -20.5228503, upper bound: 20.6343363
time: 0.65 seconds

## Relational analysis of NS_A1_B1_A2_A1_A1_B1_B2

### Relational analysis result of NS_A1_B1_A2_A1_A1_B1_B2
Status: Status.UNKNOWN
Output dim: 3, lower bound: -20.5228503, upper bound: 20.6343322
time: 0.60 seconds

## BFS NS instance: NS_A1_B1_A2_A1_A1_B2

### Backsubstitution after applying NS history:
0: -108.7109680, 150.9740753, -59.1519318, 76.8340149, -185.5449829, 210.1260071
1: -13.9608555, 13.0281258, -6.9958315, 7.0370650, -20.9979210, 20.0239525
2: -8.0301933, 13.2491865, -4.2961965, 6.6580710, -14.6882648, 17.5453835
3: -6.4961653, 14.5776148, -3.5116916, 7.1283474, -13.6245127, 18.0893040
4: -9.9313498, 11.8866148, -5.3880043, 6.0388904, -15.9702396, 17.2746162

Time for backsubstitution: 2.48 seconds

### NS candidates at layer 1
type: A, layer: 1, pos: 0
type: B, layer: 1, pos: 21
type: A, layer: 1, pos: 47
type: A, layer: 1, pos: 7
type: A, layer: 1, pos: 24
type: A, layer: 1, pos: 6
type: B, layer: 1, pos: 24
type: B, layer: 1, pos: 7
type: B, layer: 1, pos: 28
type: B, layer: 1, pos: 36
type: B, layer: 1, pos: 6
type: B, layer: 1, pos: 47
type: A, layer: 1, pos: 21
type: B, layer: 1, pos: 0

Time for candidate selection: 0.21 seconds

### Candidate
type: A, layer: 1, pos: 0

### Candidate
type: B, layer: 1, pos: 21

### Candidate
type: A, layer: 1, pos: 47

### Candidate
type: A, layer: 1, pos: 7

### Candidate
type: A, layer: 1, pos: 24

## Relational analysis of NS_A1_B1_A2_A1_A1_B2_A1

### Relational analysis result of NS_A1_B1_A2_A1_A1_B2_A1
Status: Status.UNKNOWN
Output dim: 3, lower bound: -20.5018316, upper bound: 20.6317531
time: 0.60 seconds

## Relational analysis of NS_A1_B1_A2_A1_A1_B2_A2
Optimization infeasible because this subproblem isn't reachable.

### Candidate
type: A, layer: 1, pos: 6

## Relational analysis of NS_A1_B1_A2_A1_A1_B2_A1
Optimization infeasible because this subproblem isn't reachable.

### Candidate
type: B, layer: 1, pos: 24

### Candidate
type: B, layer: 1, pos: 7

### Candidate
type: B, layer: 1, pos: 28

## Relational analysis of NS_A1_B1_A2_A1_A1_B2_B1
Optimization infeasible because this subproblem isn't reachable.

### Candidate
type: B, layer: 1, pos: 36

## Relational analysis of NS_A1_B1_A2_A1_A1_B2_B1

### Relational analysis result of NS_A1_B1_A2_A1_A1_B2_B1
Status: Status.UNKNOWN
Output dim: 3, lower bound: -20.5228503, upper bound: 20.6343335
time: 0.61 seconds

## Relational analysis of NS_A1_B1_A2_A1_A1_B2_B2

### Relational analysis result of NS_A1_B1_A2_A1_A1_B2_B2
Status: Status.UNKNOWN
Output dim: 3, lower bound: -20.5228503, upper bound: 20.6343359
time: 0.56 seconds

## BFS NS instance: NS_A1_B1_A2_A1_A2_B1

### Backsubstitution after applying NS history:
0: -106.4225693, 147.9249115, -60.7692795, 78.7800903, -185.2026520, 208.6941528
1: -13.6897011, 12.7627611, -7.1476021, 7.2403092, -20.9300098, 19.9103622
2: -7.8642807, 12.9812555, -4.4023685, 6.8436527, -14.7079325, 17.3836250
3: -6.3524604, 14.2819500, -3.6247139, 7.3066416, -13.6591015, 17.9066620
4: -9.7289190, 11.6428547, -5.5420227, 6.2030392, -15.9319582, 17.1848774

Time for backsubstitution: 2.48 seconds

### NS candidates at layer 1
type: A, layer: 1, pos: 0
type: B, layer: 1, pos: 21
type: A, layer: 1, pos: 47
type: A, layer: 1, pos: 7
type: A, layer: 1, pos: 6
type: A, layer: 1, pos: 24
type: B, layer: 1, pos: 24
type: B, layer: 1, pos: 7
type: B, layer: 1, pos: 36
type: B, layer: 1, pos: 28
type: B, layer: 1, pos: 6
type: B, layer: 1, pos: 47
type: A, layer: 1, pos: 21
type: B, layer: 1, pos: 0

Time for candidate selection: 0.21 seconds

### Candidate
type: A, layer: 1, pos: 0

### Candidate
type: B, layer: 1, pos: 21

### Candidate
type: A, layer: 1, pos: 47

### Candidate
type: A, layer: 1, pos: 7

### Candidate
type: A, layer: 1, pos: 6

## Relational analysis of NS_A1_B1_A2_A1_A2_B1_A1
Optimization infeasible because this subproblem isn't reachable.

### Candidate
type: A, layer: 1, pos: 24

## Relational analysis of NS_A1_B1_A2_A1_A2_B1_A1

### Relational analysis result of NS_A1_B1_A2_A1_A2_B1_A1
Status: Status.UNKNOWN
Output dim: 3, lower bound: -20.5018379, upper bound: 20.6964961
time: 0.54 seconds

## Relational analysis of NS_A1_B1_A2_A1_A2_B1_A2

### Relational analysis result of NS_A1_B1_A2_A1_A2_B1_A2
Status: Status.UNKNOWN
Output dim: 3, lower bound: -20.5018374, upper bound: 20.6670930
time: 0.58 seconds

## BFS NS instance: NS_A1_B1_A2_A1_A2_B2

### Backsubstitution after applying NS history:
0: -106.4225693, 147.9249115, -59.1519318, 76.8340149, -183.2565918, 207.0768433
1: -13.6897011, 12.7627611, -6.9958315, 7.0370650, -20.7267666, 19.7585907
2: -7.8642807, 12.9812555, -4.2961965, 6.6580710, -14.5223513, 17.2774506
3: -6.3524604, 14.2819500, -3.5116916, 7.1283474, -13.4808064, 17.7936363
4: -9.7289190, 11.6428547, -5.3880043, 6.0388904, -15.7678089, 17.0308590

Time for backsubstitution: 2.49 seconds

### NS candidates at layer 1
type: A, layer: 1, pos: 0
type: B, layer: 1, pos: 21
type: A, layer: 1, pos: 47
type: A, layer: 1, pos: 7
type: A, layer: 1, pos: 24
type: A, layer: 1, pos: 6
type: B, layer: 1, pos: 24
type: B, layer: 1, pos: 7
type: B, layer: 1, pos: 28
type: B, layer: 1, pos: 36
type: B, layer: 1, pos: 6
type: B, layer: 1, pos: 47
type: A, layer: 1, pos: 21
type: B, layer: 1, pos: 0

Time for candidate selection: 0.21 seconds

### Candidate
type: A, layer: 1, pos: 0

### Candidate
type: B, layer: 1, pos: 21

### Candidate
type: A, layer: 1, pos: 47

### Candidate
type: A, layer: 1, pos: 7

### Candidate
type: A, layer: 1, pos: 24

## Relational analysis of NS_A1_B1_A2_A1_A2_B2_A1

### Relational analysis result of NS_A1_B1_A2_A1_A2_B2_A1
Status: Status.UNKNOWN
Output dim: 3, lower bound: -20.5018379, upper bound: 20.6964973
time: 0.66 seconds

## Relational analysis of NS_A1_B1_A2_A1_A2_B2_A2

### Relational analysis result of NS_A1_B1_A2_A1_A2_B2_A2
Status: Status.UNKNOWN
Output dim: 3, lower bound: -20.5018377, upper bound: 20.6670934
time: 0.58 seconds

## BFS NS instance: NS_A1_B2_A1_B1_B1_A1

### Backsubstitution after applying NS history:
0: -102.5664444, 143.4198303, -113.3619614, 156.8386383, -259.4049988, 256.7817993
1: -13.3056774, 12.3136339, -14.5068369, 13.5073509, -26.8130245, 26.8204708
2: -7.5930424, 12.5812550, -8.3509645, 13.7784405, -21.3714790, 20.9322128
3: -6.1116810, 13.8463879, -6.7625446, 15.1243000, -21.2359810, 20.6089306
4: -9.3920126, 11.2692289, -10.3205271, 12.3568792, -21.7488918, 21.5897522

Time for backsubstitution: 2.49 seconds

### NS candidates at layer 1
type: B, layer: 1, pos: 0
type: A, layer: 1, pos: 7
type: B, layer: 1, pos: 7
type: B, layer: 1, pos: 47
type: A, layer: 1, pos: 28
type: B, layer: 1, pos: 24
type: A, layer: 1, pos: 6
type: B, layer: 1, pos: 6
type: A, layer: 1, pos: 24
type: B, layer: 1, pos: 21
type: A, layer: 1, pos: 47
type: A, layer: 1, pos: 35
type: A, layer: 1, pos: 0
type: B, layer: 1, pos: 37

Time for candidate selection: 0.21 seconds

### Candidate
type: B, layer: 1, pos: 0

### Candidate
type: A, layer: 1, pos: 7

### Candidate
type: B, layer: 1, pos: 7

### Candidate
type: B, layer: 1, pos: 47

## Relational analysis of NS_A1_B2_A1_B1_B1_A1_B1
Optimization infeasible because this subproblem isn't reachable.

### Candidate
type: A, layer: 1, pos: 28

### Candidate
type: B, layer: 1, pos: 24

## Relational analysis of NS_A1_B2_A1_B1_B1_A1_B1

### Relational analysis result of NS_A1_B2_A1_B1_B1_A1_B1
Status: Status.UNKNOWN
Output dim: 3, lower bound: -20.6078631, upper bound: 20.6662791
time: 0.67 seconds

## Relational analysis of NS_A1_B2_A1_B1_B1_A1_B2
Optimization infeasible because this subproblem isn't reachable.

### Candidate
type: A, layer: 1, pos: 6

## Relational analysis of NS_A1_B2_A1_B1_B1_A1_A1
Optimization infeasible because this subproblem isn't reachable.

### Candidate
type: B, layer: 1, pos: 6

### Candidate
type: A, layer: 1, pos: 24

### Candidate
type: B, layer: 1, pos: 21

## Relational analysis of NS_A1_B2_A1_B1_B1_A1_B1

### Relational analysis result of NS_A1_B2_A1_B1_B1_A1_B1
Status: Status.UNKNOWN
Output dim: 3, lower bound: -20.6095015, upper bound: 20.6827620
time: 0.70 seconds

## Relational analysis of NS_A1_B2_A1_B1_B1_A1_B2

### Relational analysis result of NS_A1_B2_A1_B1_B1_A1_B2
Status: Status.UNKNOWN
Output dim: 3, lower bound: -20.6094975, upper bound: 20.6827622
time: 0.59 seconds

## BFS NS instance: NS_A1_B2_A1_B1_B1_A2

### Backsubstitution after applying NS history:
0: -105.0573273, 146.1588287, -113.3619614, 156.8386383, -261.8959656, 259.5207520
1: -13.5524206, 12.5621405, -14.5068369, 13.5073509, -27.0597725, 27.0689774
2: -7.7492042, 12.8493176, -8.3509645, 13.7784405, -21.5276413, 21.2002811
3: -6.2506890, 14.1001730, -6.7625446, 15.1243000, -21.3749866, 20.8627129
4: -9.5892010, 11.5065594, -10.3205271, 12.3568792, -21.9460793, 21.8270836

Time for backsubstitution: 2.51 seconds

### NS candidates at layer 1
type: A, layer: 1, pos: 7
type: B, layer: 1, pos: 0
type: B, layer: 1, pos: 7
type: B, layer: 1, pos: 47
type: A, layer: 1, pos: 28
type: B, layer: 1, pos: 24
type: A, layer: 1, pos: 6
type: A, layer: 1, pos: 24
type: B, layer: 1, pos: 6
type: A, layer: 1, pos: 35
type: A, layer: 1, pos: 0
type: B, layer: 1, pos: 21
type: A, layer: 1, pos: 47
type: B, layer: 1, pos: 37

Time for candidate selection: 0.21 seconds

### Candidate
type: A, layer: 1, pos: 7

### Candidate
type: B, layer: 1, pos: 0

### Candidate
type: B, layer: 1, pos: 7

### Candidate
type: B, layer: 1, pos: 47

## Relational analysis of NS_A1_B2_A1_B1_B1_A2_B1
Optimization infeasible because this subproblem isn't reachable.

### Candidate
type: A, layer: 1, pos: 28

### Candidate
type: B, layer: 1, pos: 24

## Relational analysis of NS_A1_B2_A1_B1_B1_A2_B1

### Relational analysis result of NS_A1_B2_A1_B1_B1_A2_B1
Status: Status.UNKNOWN
Output dim: 3, lower bound: -20.6078631, upper bound: 20.6662776
time: 0.67 seconds

## Relational analysis of NS_A1_B2_A1_B1_B1_A2_B2
Optimization infeasible because this subproblem isn't reachable.

### Candidate
type: A, layer: 1, pos: 6

## Relational analysis of NS_A1_B2_A1_B1_B1_A2_A1
Optimization infeasible because this subproblem isn't reachable.

### Candidate
type: A, layer: 1, pos: 24

### Candidate
type: B, layer: 1, pos: 6

### Candidate
type: A, layer: 1, pos: 35

### Candidate
type: A, layer: 1, pos: 0

### Candidate
type: B, layer: 1, pos: 21

## Relational analysis of NS_A1_B2_A1_B1_B1_A2_B1

### Relational analysis result of NS_A1_B2_A1_B1_B1_A2_B1
Status: Status.UNKNOWN
Output dim: 3, lower bound: -20.6095010, upper bound: 20.6827631
time: 0.61 seconds

## Relational analysis of NS_A1_B2_A1_B1_B1_A2_B2

### Relational analysis result of NS_A1_B2_A1_B1_B1_A2_B2
Status: Status.UNKNOWN
Output dim: 3, lower bound: -20.6094993, upper bound: 20.6827557
time: 0.57 seconds

## BFS NS instance: NS_A1_B2_A1_B1_B2_A1

### Backsubstitution after applying NS history:
0: -102.5664444, 143.4198303, -112.4632492, 155.7105408, -258.2768860, 255.8830566
1: -13.3056774, 12.3136339, -14.4164772, 13.4019489, -26.7076244, 26.7301102
2: -7.5930424, 12.5812550, -8.2836676, 13.6824551, -21.2754974, 20.8649178
3: -6.1116810, 13.8463879, -6.6981082, 15.0158939, -21.1275749, 20.5444965
4: -9.3920126, 11.2692289, -10.2392101, 12.2670708, -21.6590824, 21.5084343

Time for backsubstitution: 2.50 seconds

### NS candidates at layer 1
type: B, layer: 1, pos: 0
type: A, layer: 1, pos: 7
type: B, layer: 1, pos: 7
type: B, layer: 1, pos: 47
type: A, layer: 1, pos: 28
type: B, layer: 1, pos: 24
type: A, layer: 1, pos: 6
type: B, layer: 1, pos: 6
type: A, layer: 1, pos: 24
type: B, layer: 1, pos: 21
type: A, layer: 1, pos: 35
type: A, layer: 1, pos: 0
type: A, layer: 1, pos: 47
type: B, layer: 1, pos: 37

Time for candidate selection: 0.21 seconds

### Candidate
type: B, layer: 1, pos: 0

### Candidate
type: A, layer: 1, pos: 7

### Candidate
type: B, layer: 1, pos: 7

### Candidate
type: B, layer: 1, pos: 47

## Relational analysis of NS_A1_B2_A1_B1_B2_A1_B1
Optimization infeasible because this subproblem isn't reachable.

### Candidate
type: A, layer: 1, pos: 28

### Candidate
type: B, layer: 1, pos: 24

## Relational analysis of NS_A1_B2_A1_B1_B2_A1_B1

### Relational analysis result of NS_A1_B2_A1_B1_B2_A1_B1
Status: Status.UNKNOWN
Output dim: 3, lower bound: -20.6023319, upper bound: 20.6662785
time: 0.62 seconds

## Relational analysis of NS_A1_B2_A1_B1_B2_A1_B2

### Relational analysis result of NS_A1_B2_A1_B1_B2_A1_B2
Status: Status.UNKNOWN
Output dim: 3, lower bound: -20.5489739, upper bound: 20.6662697
time: 0.63 seconds

## BFS NS instance: NS_A1_B2_A1_B1_B2_A2

### Backsubstitution after applying NS history:
0: -105.0573273, 146.1588287, -112.4632492, 155.7105408, -260.7678833, 258.6220398
1: -13.5524206, 12.5621405, -14.4164772, 13.4019489, -26.9543686, 26.9786186
2: -7.7492042, 12.8493176, -8.2836676, 13.6824551, -21.4316597, 21.1329842
3: -6.2506890, 14.1001730, -6.6981082, 15.0158939, -21.2665825, 20.7982788
4: -9.5892010, 11.5065594, -10.2392101, 12.2670708, -21.8562717, 21.7457676

Time for backsubstitution: 2.51 seconds

### NS candidates at layer 1
type: A, layer: 1, pos: 7
type: B, layer: 1, pos: 7
type: B, layer: 1, pos: 0
type: A, layer: 1, pos: 28
type: A, layer: 1, pos: 35
type: B, layer: 1, pos: 47
type: A, layer: 1, pos: 0
type: B, layer: 1, pos: 24
type: A, layer: 1, pos: 6
type: A, layer: 1, pos: 24
type: B, layer: 1, pos: 6
type: B, layer: 1, pos: 21
type: A, layer: 1, pos: 47
type: B, layer: 1, pos: 37

Time for candidate selection: 0.21 seconds

### Candidate
type: A, layer: 1, pos: 7

### Candidate
type: B, layer: 1, pos: 7

### Candidate
type: B, layer: 1, pos: 0

### Candidate
type: A, layer: 1, pos: 28

### Candidate
type: A, layer: 1, pos: 35

### Candidate
type: B, layer: 1, pos: 47

## Relational analysis of NS_A1_B2_A1_B1_B2_A2_B1
Optimization infeasible because this subproblem isn't reachable.

### Candidate
type: A, layer: 1, pos: 0

### Candidate
type: B, layer: 1, pos: 24

## Relational analysis of NS_A1_B2_A1_B1_B2_A2_B1

### Relational analysis result of NS_A1_B2_A1_B1_B2_A2_B1
Status: Status.UNKNOWN
Output dim: 3, lower bound: -20.6023319, upper bound: 20.6662777
time: 0.53 seconds

## Relational analysis of NS_A1_B2_A1_B1_B2_A2_B2

### Relational analysis result of NS_A1_B2_A1_B1_B2_A2_B2
Status: Status.UNKNOWN
Output dim: 3, lower bound: -20.5489739, upper bound: 20.6662698
time: 0.64 seconds

## BFS NS instance: NS_A1_B2_A1_B2_B1_B1

### Backsubstitution after applying NS history:
0: -107.4569092, 149.4062958, -106.9470139, 147.2560730, -254.7129517, 256.3533020
1: -13.8525238, 12.8498917, -13.6055450, 12.7431421, -26.5956650, 26.4554291
2: -7.9272017, 13.1349926, -7.8803043, 12.9435129, -20.8707142, 21.0152969
3: -6.3961639, 14.4240589, -6.3831148, 14.1405191, -20.5366821, 20.8071747
4: -9.8063383, 11.7648029, -9.7215338, 11.5995846, -21.4059200, 21.4863300

Time for backsubstitution: 2.51 seconds

### NS candidates at layer 1
type: A, layer: 1, pos: 35
type: A, layer: 1, pos: 0
type: A, layer: 1, pos: 7
type: B, layer: 1, pos: 7
type: B, layer: 1, pos: 21
type: A, layer: 1, pos: 47
type: A, layer: 1, pos: 36
type: A, layer: 1, pos: 28
type: A, layer: 1, pos: 6
type: A, layer: 1, pos: 24
type: B, layer: 1, pos: 24
type: B, layer: 1, pos: 6
type: B, layer: 1, pos: 37
type: B, layer: 1, pos: 0

Time for candidate selection: 0.21 seconds

### Candidate
type: A, layer: 1, pos: 35

## Relational analysis of NS_A1_B2_A1_B2_B1_B1_A1
Optimization infeasible because this subproblem isn't reachable.

### Candidate
type: A, layer: 1, pos: 0

### Candidate
type: A, layer: 1, pos: 7

### Candidate
type: B, layer: 1, pos: 7

### Candidate
type: B, layer: 1, pos: 21

## Relational analysis of NS_A1_B2_A1_B2_B1_B1_B1

### Relational analysis result of NS_A1_B2_A1_B2_B1_B1_B1
Status: Status.UNKNOWN
Output dim: 3, lower bound: -20.4844750, upper bound: 20.6657644
time: 0.57 seconds

## Relational analysis of NS_A1_B2_A1_B2_B1_B1_B2

### Relational analysis result of NS_A1_B2_A1_B2_B1_B1_B2
Status: Status.UNKNOWN
Output dim: 3, lower bound: -20.4844792, upper bound: 20.6659409
time: 0.57 seconds

## BFS NS instance: NS_A1_B2_A1_B2_B1_B2

### Backsubstitution after applying NS history:
0: -107.4569092, 149.4062958, -115.5967560, 159.3176727, -266.7745667, 265.0030518
1: -13.8525238, 12.8498917, -14.7356567, 13.7355738, -27.5880947, 27.5855446
2: -7.9272017, 13.1349926, -8.4914293, 14.0176039, -21.9448032, 21.6264229
3: -6.3961639, 14.4240589, -6.8866749, 15.3633566, -21.7595177, 21.3107338
4: -9.8063383, 11.7648029, -10.4997616, 12.5683928, -22.3747272, 22.2645569

Time for backsubstitution: 2.50 seconds

### NS candidates at layer 1
type: A, layer: 1, pos: 7
type: B, layer: 1, pos: 7
type: B, layer: 1, pos: 0
type: A, layer: 1, pos: 28
type: A, layer: 1, pos: 36
type: B, layer: 1, pos: 24
type: B, layer: 1, pos: 6
type: A, layer: 1, pos: 6
type: B, layer: 1, pos: 21
type: A, layer: 1, pos: 24
type: A, layer: 1, pos: 35
type: A, layer: 1, pos: 0
type: A, layer: 1, pos: 47
type: B, layer: 1, pos: 37

Time for candidate selection: 0.21 seconds

### Candidate
type: A, layer: 1, pos: 7

### Candidate
type: B, layer: 1, pos: 7

### Candidate
type: B, layer: 1, pos: 0

### Candidate
type: A, layer: 1, pos: 28

## Relational analysis of NS_A1_B2_A1_B2_B1_B2_A1
Optimization infeasible because this subproblem isn't reachable.

### Candidate
type: A, layer: 1, pos: 36

## Relational analysis of NS_A1_B2_A1_B2_B1_B2_A1

### Relational analysis result of NS_A1_B2_A1_B2_B1_B2_A1
Status: Status.UNKNOWN
Output dim: 3, lower bound: -20.5819115, upper bound: 20.6827185
time: 0.65 seconds

## Relational analysis of NS_A1_B2_A1_B2_B1_B2_A2

### Relational analysis result of NS_A1_B2_A1_B2_B1_B2_A2
Status: Status.UNKNOWN
Output dim: 3, lower bound: -20.5819115, upper bound: 20.6827173
time: 0.64 seconds

## BFS NS instance: NS_A1_B2_A1_B2_B2_A1

### Backsubstitution after applying NS history:
0: -102.5664444, 143.4198303, -112.9994278, 155.6864014, -258.2527466, 256.4192505
1: -13.3056774, 12.3136339, -14.4079819, 13.4225550, -26.7282333, 26.7216148
2: -7.5930424, 12.5812550, -8.2972803, 13.7014332, -21.2944756, 20.8785286
3: -6.1116810, 13.8463879, -6.7209225, 15.0139751, -21.1256561, 20.5673084
4: -9.3920126, 11.2692289, -10.2640667, 12.2832479, -21.6752605, 21.5332947

Time for backsubstitution: 2.52 seconds

### NS candidates at layer 1
type: B, layer: 1, pos: 0
type: A, layer: 1, pos: 7
type: B, layer: 1, pos: 7
type: B, layer: 1, pos: 47
type: A, layer: 1, pos: 28
type: B, layer: 1, pos: 6
type: B, layer: 1, pos: 21
type: A, layer: 1, pos: 6
type: A, layer: 1, pos: 24
type: B, layer: 1, pos: 24
type: A, layer: 1, pos: 47
type: A, layer: 1, pos: 35
type: A, layer: 1, pos: 0
type: B, layer: 1, pos: 37

Time for candidate selection: 0.21 seconds

### Candidate
type: B, layer: 1, pos: 0

### Candidate
type: A, layer: 1, pos: 7

### Candidate
type: B, layer: 1, pos: 7

### Candidate
type: B, layer: 1, pos: 47

## Relational analysis of NS_A1_B2_A1_B2_B2_A1_B1
Optimization infeasible because this subproblem isn't reachable.

### Candidate
type: A, layer: 1, pos: 28

### Candidate
type: B, layer: 1, pos: 6

## Relational analysis of NS_A1_B2_A1_B2_B2_A1_B1
Optimization infeasible because this subproblem isn't reachable.

### Candidate
type: B, layer: 1, pos: 21

## Relational analysis of NS_A1_B2_A1_B2_B2_A1_B1

### Relational analysis result of NS_A1_B2_A1_B2_B2_A1_B1
Status: Status.UNKNOWN
Output dim: 3, lower bound: -20.5779191, upper bound: 20.6827544
time: 0.59 seconds

## Relational analysis of NS_A1_B2_A1_B2_B2_A1_B2

### Relational analysis result of NS_A1_B2_A1_B2_B2_A1_B2
Status: Status.UNKNOWN
Output dim: 3, lower bound: -20.5779186, upper bound: 20.6827531
time: 0.67 seconds

## BFS NS instance: NS_A1_B2_A1_B2_B2_A2

### Backsubstitution after applying NS history:
0: -105.0573273, 146.1588287, -112.9994278, 155.6864014, -260.7437134, 259.1582336
1: -13.5524206, 12.5621405, -14.4079819, 13.4225550, -26.9749756, 26.9701214
2: -7.7492042, 12.8493176, -8.2972803, 13.7014332, -21.4506359, 21.1465988
3: -6.2506890, 14.1001730, -6.7209225, 15.0139751, -21.2646618, 20.8210907
4: -9.5892010, 11.5065594, -10.2640667, 12.2832479, -21.8724480, 21.7706261

Time for backsubstitution: 2.52 seconds

### NS candidates at layer 1
type: A, layer: 1, pos: 7
type: B, layer: 1, pos: 7
type: B, layer: 1, pos: 0
type: B, layer: 1, pos: 47
type: A, layer: 1, pos: 28
type: B, layer: 1, pos: 6
type: B, layer: 1, pos: 24
type: B, layer: 1, pos: 21
type: A, layer: 1, pos: 6
type: A, layer: 1, pos: 24
type: A, layer: 1, pos: 35
type: A, layer: 1, pos: 0
type: A, layer: 1, pos: 47
type: B, layer: 1, pos: 37

Time for candidate selection: 0.21 seconds

### Candidate
type: A, layer: 1, pos: 7

### Candidate
type: B, layer: 1, pos: 7

### Candidate
type: B, layer: 1, pos: 0

### Candidate
type: B, layer: 1, pos: 47

## Relational analysis of NS_A1_B2_A1_B2_B2_A2_B1
Optimization infeasible because this subproblem isn't reachable.

### Candidate
type: A, layer: 1, pos: 28

### Candidate
type: B, layer: 1, pos: 6

## Relational analysis of NS_A1_B2_A1_B2_B2_A2_B1
Optimization infeasible because this subproblem isn't reachable.

### Candidate
type: B, layer: 1, pos: 24

## Relational analysis of NS_A1_B2_A1_B2_B2_A2_B1

### Relational analysis result of NS_A1_B2_A1_B2_B2_A2_B1
Status: Status.UNKNOWN
Output dim: 3, lower bound: -20.5735387, upper bound: 20.6662721
time: 0.56 seconds

## Relational analysis of NS_A1_B2_A1_B2_B2_A2_B2

### Relational analysis result of NS_A1_B2_A1_B2_B2_A2_B2
Status: Status.UNKNOWN
Output dim: 3, lower bound: -20.5489739, upper bound: 20.6662697
time: 0.59 seconds

## BFS NS instance: NS_A1_B2_A2_B1_B1_B1

### Backsubstitution after applying NS history:
0: -127.9776459, 179.3025665, -105.7482376, 147.0162506, -274.9938660, 285.0507202
1: -16.6276455, 15.3680820, -13.6290569, 12.6251478, -29.2527905, 28.9971390
2: -9.4743786, 15.7429419, -7.8028665, 12.9075041, -22.3818817, 23.5458069
3: -7.6515388, 17.2871590, -6.3008537, 14.1975565, -21.8490925, 23.5880108
4: -11.7098789, 14.1008635, -9.6589079, 11.5644083, -23.2742882, 23.7597713

Time for backsubstitution: 2.53 seconds

### NS candidates at layer 1
type: A, layer: 1, pos: 35
type: A, layer: 1, pos: 0
type: A, layer: 1, pos: 7
type: B, layer: 1, pos: 7
type: A, layer: 1, pos: 6
type: A, layer: 1, pos: 36
type: A, layer: 1, pos: 28
type: B, layer: 1, pos: 24
type: B, layer: 1, pos: 6
type: A, layer: 1, pos: 24
type: A, layer: 1, pos: 47
type: B, layer: 1, pos: 47
type: B, layer: 1, pos: 37
type: B, layer: 1, pos: 0

Time for candidate selection: 0.21 seconds

### Candidate
type: A, layer: 1, pos: 35

### Candidate
type: A, layer: 1, pos: 0

### Candidate
type: A, layer: 1, pos: 7

### Candidate
type: B, layer: 1, pos: 7

### Candidate
type: A, layer: 1, pos: 6

## Relational analysis of NS_A1_B2_A2_B1_B1_B1_A1
Optimization infeasible because this subproblem isn't reachable.

### Candidate
type: A, layer: 1, pos: 36

## Relational analysis of NS_A1_B2_A2_B1_B1_B1_A1

### Relational analysis result of NS_A1_B2_A2_B1_B1_B1_A1
Status: Status.UNKNOWN
Output dim: 3, lower bound: -20.7036472, upper bound: 20.5688159
time: 0.68 seconds

## Relational analysis of NS_A1_B2_A2_B1_B1_B1_A2

### Relational analysis result of NS_A1_B2_A2_B1_B1_B1_A2
Status: Status.UNKNOWN
Output dim: 3, lower bound: -20.7036429, upper bound: 20.5688182
time: 0.65 seconds

## BFS NS instance: NS_A1_B2_A2_B1_B1_B2

### Backsubstitution after applying NS history:
0: -127.9776459, 179.3025665, -104.5640869, 145.6174622, -273.5950928, 283.8665161
1: -16.6276455, 15.3680820, -13.5162983, 12.4906998, -29.1183453, 28.8843803
2: -9.4743786, 15.7429419, -7.7180171, 12.7851391, -22.2595177, 23.4609585
3: -7.6515388, 17.2871590, -6.2191501, 14.0629053, -21.7144432, 23.5063076
4: -11.7098789, 14.1008635, -9.5563126, 11.4491491, -23.1590271, 23.6571770

Time for backsubstitution: 2.52 seconds

### NS candidates at layer 1
type: A, layer: 1, pos: 35
type: A, layer: 1, pos: 0
type: A, layer: 1, pos: 7
type: B, layer: 1, pos: 7
type: A, layer: 1, pos: 6
type: A, layer: 1, pos: 36
type: A, layer: 1, pos: 28
type: B, layer: 1, pos: 24
type: A, layer: 1, pos: 47
type: B, layer: 1, pos: 6
type: A, layer: 1, pos: 24
type: B, layer: 1, pos: 47
type: B, layer: 1, pos: 37
type: B, layer: 1, pos: 0

Time for candidate selection: 0.21 seconds

### Candidate
type: A, layer: 1, pos: 35

### Candidate
type: A, layer: 1, pos: 0

### Candidate
type: A, layer: 1, pos: 7

### Candidate
type: B, layer: 1, pos: 7

### Candidate
type: A, layer: 1, pos: 6

## Relational analysis of NS_A1_B2_A2_B1_B1_B2_A1
Optimization infeasible because this subproblem isn't reachable.

### Candidate
type: A, layer: 1, pos: 36

## Relational analysis of NS_A1_B2_A2_B1_B1_B2_A1

### Relational analysis result of NS_A1_B2_A2_B1_B1_B2_A1
Status: Status.UNKNOWN
Output dim: 3, lower bound: -20.7024944, upper bound: 20.5688187
time: 0.61 seconds

## Relational analysis of NS_A1_B2_A2_B1_B1_B2_A2

### Relational analysis result of NS_A1_B2_A2_B1_B1_B2_A2
Status: Status.UNKNOWN
Output dim: 3, lower bound: -20.7024944, upper bound: 20.5688180
time: 0.56 seconds

## BFS NS instance: NS_A2_B1_A2_A2_B1_A1

### Backsubstitution after applying NS history:
0: -109.4565582, 152.5658569, -60.7692795, 78.7800903, -188.2366180, 213.3350983
1: -14.1440659, 13.0789051, -7.1476021, 7.2403092, -21.3843746, 20.2265034
2: -8.0750504, 13.3887863, -4.4023685, 6.8436527, -14.9187031, 17.7911549
3: -6.5016389, 14.6994257, -3.6247139, 7.3066416, -13.8082809, 18.3241386
4: -9.9753170, 11.9980631, -5.5420227, 6.2030392, -16.1783562, 17.5400848

Time for backsubstitution: 2.54 seconds

### NS candidates at layer 1
type: A, layer: 1, pos: 0
type: B, layer: 1, pos: 21
type: A, layer: 1, pos: 7
type: A, layer: 1, pos: 28
type: A, layer: 1, pos: 6
type: A, layer: 1, pos: 24
type: B, layer: 1, pos: 24
type: B, layer: 1, pos: 7
type: B, layer: 1, pos: 28
type: B, layer: 1, pos: 36
type: B, layer: 1, pos: 6
type: A, layer: 1, pos: 21
type: B, layer: 1, pos: 47
type: B, layer: 1, pos: 0

Time for candidate selection: 0.21 seconds

### Candidate
type: A, layer: 1, pos: 0

### Candidate
type: B, layer: 1, pos: 21

### Candidate
type: A, layer: 1, pos: 7

## Relational analysis of NS_A2_B1_A2_A2_B1_A1_A1
Optimization infeasible because this subproblem isn't reachable.

### Candidate
type: A, layer: 1, pos: 28

## Relational analysis of NS_A2_B1_A2_A2_B1_A1_A1

### Relational analysis result of NS_A2_B1_A2_A2_B1_A1_A1
Status: Status.UNKNOWN
Output dim: 3, lower bound: -20.5227843, upper bound: 20.7042775
time: 0.77 seconds

## Relational analysis of NS_A2_B1_A2_A2_B1_A1_A2

### Relational analysis result of NS_A2_B1_A2_A2_B1_A1_A2
Status: Status.UNKNOWN
Output dim: 3, lower bound: -20.5227833, upper bound: 20.7031628
time: 0.57 seconds

## BFS NS instance: NS_A2_B1_A2_A2_B1_A2

### Backsubstitution after applying NS history:
0: -109.9886093, 152.5753632, -60.7692795, 78.7800903, -188.7687073, 213.3446198
1: -14.1351700, 13.1023397, -7.1476021, 7.2403092, -21.3754787, 20.2499371
2: -8.0956335, 13.4106274, -4.4023685, 6.8436527, -14.9392862, 17.8129959
3: -6.5262361, 14.6979609, -3.6247139, 7.3066416, -13.8328781, 18.3226738
4: -10.0008602, 12.0177679, -5.5420227, 6.2030392, -16.2038994, 17.5597916

Time for backsubstitution: 2.55 seconds

### NS candidates at layer 1
type: A, layer: 1, pos: 0
type: B, layer: 1, pos: 21
type: A, layer: 1, pos: 6
type: A, layer: 1, pos: 7
type: A, layer: 1, pos: 24
type: B, layer: 1, pos: 24
type: A, layer: 1, pos: 28
type: B, layer: 1, pos: 36
type: B, layer: 1, pos: 28
type: B, layer: 1, pos: 7
type: B, layer: 1, pos: 6
type: A, layer: 1, pos: 21
type: B, layer: 1, pos: 47
type: B, layer: 1, pos: 0

Time for candidate selection: 0.22 seconds

### Candidate
type: A, layer: 1, pos: 0

### Candidate
type: B, layer: 1, pos: 21

### Candidate
type: A, layer: 1, pos: 6

## Relational analysis of NS_A2_B1_A2_A2_B1_A2_A1
Optimization infeasible because this subproblem isn't reachable.

### Candidate
type: A, layer: 1, pos: 7

## Relational analysis of NS_A2_B1_A2_A2_B1_A2_A1
Optimization infeasible because this subproblem isn't reachable.

### Candidate
type: A, layer: 1, pos: 24

## Relational analysis of NS_A2_B1_A2_A2_B1_A2_A1

### Relational analysis result of NS_A2_B1_A2_A2_B1_A2_A1
Status: Status.UNKNOWN
Output dim: 3, lower bound: -20.5039903, upper bound: 20.6870095
time: 0.63 seconds

## Relational analysis of NS_A2_B1_A2_A2_B1_A2_A2

### Relational analysis result of NS_A2_B1_A2_A2_B1_A2_A2
Status: Status.UNKNOWN
Output dim: 3, lower bound: -20.5039886, upper bound: 20.6696066
time: 0.61 seconds

## BFS NS instance: NS_A2_B1_A2_A2_B2_A1

### Backsubstitution after applying NS history:
0: -109.4565582, 152.5658569, -59.1519318, 76.8340149, -186.2905579, 211.7177887
1: -14.1440659, 13.0789051, -6.9958315, 7.0370650, -21.1811314, 20.0747299
2: -8.0750504, 13.3887863, -4.2961965, 6.6580710, -14.7331219, 17.6849823
3: -6.5016389, 14.6994257, -3.5116916, 7.1283474, -13.6299858, 18.2111168
4: -9.9753170, 11.9980631, -5.3880043, 6.0388904, -16.0142059, 17.3860664

Time for backsubstitution: 2.54 seconds

### NS candidates at layer 1
type: A, layer: 1, pos: 0
type: B, layer: 1, pos: 21
type: A, layer: 1, pos: 7
type: A, layer: 1, pos: 28
type: A, layer: 1, pos: 24
type: A, layer: 1, pos: 6
type: B, layer: 1, pos: 24
type: B, layer: 1, pos: 7
type: B, layer: 1, pos: 28
type: B, layer: 1, pos: 6
type: B, layer: 1, pos: 36
type: B, layer: 1, pos: 47
type: A, layer: 1, pos: 21
type: B, layer: 1, pos: 0

Time for candidate selection: 0.21 seconds

### Candidate
type: A, layer: 1, pos: 0

### Candidate
type: B, layer: 1, pos: 21

### Candidate
type: A, layer: 1, pos: 7

## Relational analysis of NS_A2_B1_A2_A2_B2_A1_A1

### Relational analysis result of NS_A2_B1_A2_A2_B2_A1_A1
Status: Status.UNKNOWN
Output dim: 3, lower bound: -20.5214582, upper bound: 20.7022298
time: 0.60 seconds

## Relational analysis of NS_A2_B1_A2_A2_B2_A1_A2

### Relational analysis result of NS_A2_B1_A2_A2_B2_A1_A2
Status: Status.UNKNOWN
Output dim: 3, lower bound: -20.5214597, upper bound: 20.6935607
time: 0.59 seconds

## BFS NS instance: NS_A2_B1_A2_A2_B2_A2

### Backsubstitution after applying NS history:
0: -109.9886093, 152.5753632, -59.1519318, 76.8340149, -186.8226318, 211.7272949
1: -14.1351700, 13.1023397, -6.9958315, 7.0370650, -21.1722355, 20.0981655
2: -8.0956335, 13.4106274, -4.2961965, 6.6580710, -14.7537041, 17.7068233
3: -6.5262361, 14.6979609, -3.5116916, 7.1283474, -13.6545830, 18.2096500
4: -10.0008602, 12.0177679, -5.3880043, 6.0388904, -16.0397511, 17.4057713

Time for backsubstitution: 2.55 seconds

### NS candidates at layer 1
type: A, layer: 1, pos: 0
type: B, layer: 1, pos: 21
type: A, layer: 1, pos: 7
type: A, layer: 1, pos: 6
type: A, layer: 1, pos: 24
type: A, layer: 1, pos: 28
type: B, layer: 1, pos: 24
type: B, layer: 1, pos: 7
type: B, layer: 1, pos: 28
type: B, layer: 1, pos: 36
type: B, layer: 1, pos: 6
type: A, layer: 1, pos: 21
type: B, layer: 1, pos: 47
type: B, layer: 1, pos: 0

Time for candidate selection: 0.21 seconds

### Candidate
type: A, layer: 1, pos: 0

### Candidate
type: B, layer: 1, pos: 21

### Candidate
type: A, layer: 1, pos: 7

## Relational analysis of NS_A2_B1_A2_A2_B2_A2_A1

### Relational analysis result of NS_A2_B1_A2_A2_B2_A2_A1
Status: Status.UNKNOWN
Output dim: 3, lower bound: -20.5214563, upper bound: 20.6836290
time: 0.64 seconds

## Relational analysis of NS_A2_B1_A2_A2_B2_A2_A2

### Relational analysis result of NS_A2_B1_A2_A2_B2_A2_A2
Status: Status.UNKNOWN
Output dim: 3, lower bound: -20.5214582, upper bound: 20.6735237
time: 0.80 seconds

## BFS NS instance: NS_A2_B2_A1_B1_A1_B1

### Backsubstitution after applying NS history:
0: -95.0735168, 132.7543030, -103.6279755, 143.8854523, -238.9589691, 236.3822784
1: -12.3337259, 11.3693285, -13.3326559, 12.3727036, -24.7064285, 24.7019844
2: -7.0452828, 11.6368580, -7.6497664, 12.6256332, -19.6709137, 19.2866249
3: -5.6269884, 12.7925835, -6.1554966, 13.8497925, -19.4767799, 18.9480762
4: -8.6930714, 10.4093132, -9.4506826, 11.3116140, -20.0046844, 19.8599911

Time for backsubstitution: 2.57 seconds

### NS candidates at layer 1
type: B, layer: 1, pos: 0
type: B, layer: 1, pos: 6
type: A, layer: 1, pos: 36
type: B, layer: 1, pos: 28
type: A, layer: 1, pos: 6
type: A, layer: 1, pos: 0
type: A, layer: 1, pos: 35
type: A, layer: 1, pos: 28
type: B, layer: 1, pos: 24
type: B, layer: 1, pos: 47
type: B, layer: 1, pos: 21
type: A, layer: 1, pos: 24
type: B, layer: 1, pos: 37
type: A, layer: 1, pos: 47

Time for candidate selection: 0.21 seconds

### Candidate
type: B, layer: 1, pos: 0

### Candidate
type: B, layer: 1, pos: 6

## Relational analysis of NS_A2_B2_A1_B1_A1_B1_B1

### Relational analysis result of NS_A2_B2_A1_B1_A1_B1_B1
Status: Status.UNKNOWN
Output dim: 3, lower bound: -20.5927473, upper bound: 20.6168489
time: 0.60 seconds

## Relational analysis of NS_A2_B2_A1_B1_A1_B1_B2
Optimization infeasible because this subproblem isn't reachable.

### Candidate
type: A, layer: 1, pos: 36

## Relational analysis of NS_A2_B2_A1_B1_A1_B1_A1

### Relational analysis result of NS_A2_B2_A1_B1_A1_B1_A1
Status: Status.UNKNOWN
Output dim: 3, lower bound: -20.5935250, upper bound: 20.6831325
time: 0.63 seconds

## Relational analysis of NS_A2_B2_A1_B1_A1_B1_A2

### Relational analysis result of NS_A2_B2_A1_B1_A1_B1_A2
Status: Status.UNKNOWN
Output dim: 3, lower bound: -20.5935250, upper bound: 20.6831340
time: 0.58 seconds

## BFS NS instance: NS_A2_B2_A1_B1_A1_B2

### Backsubstitution after applying NS history:
0: -95.0735168, 132.7543030, -104.9739761, 145.7171326, -240.7906494, 237.7282715
1: -12.3337259, 11.3693285, -13.4792881, 12.5555696, -24.8892956, 24.8486176
2: -7.0452828, 11.6368580, -7.7664585, 12.7877102, -19.8329926, 19.4033165
3: -5.6269884, 12.7925835, -6.2719631, 13.9889660, -19.6159554, 19.0645428
4: -8.6930714, 10.4093132, -9.5714073, 11.4501390, -20.1432095, 19.9807186

Time for backsubstitution: 2.56 seconds

### NS candidates at layer 1
type: B, layer: 1, pos: 0
type: A, layer: 1, pos: 35
type: A, layer: 1, pos: 0
type: B, layer: 1, pos: 6
type: A, layer: 1, pos: 36
type: B, layer: 1, pos: 28
type: A, layer: 1, pos: 6
type: A, layer: 1, pos: 28
type: B, layer: 1, pos: 24
type: B, layer: 1, pos: 47
type: B, layer: 1, pos: 21
type: A, layer: 1, pos: 24
type: B, layer: 1, pos: 37
type: A, layer: 1, pos: 47

Time for candidate selection: 0.21 seconds

### Candidate
type: B, layer: 1, pos: 0

### Candidate
type: A, layer: 1, pos: 35

## Relational analysis of NS_A2_B2_A1_B1_A1_B2_A1
Optimization infeasible because this subproblem isn't reachable.

### Candidate
type: A, layer: 1, pos: 0

### Candidate
type: B, layer: 1, pos: 6

## Relational analysis of NS_A2_B2_A1_B1_A1_B2_B1

### Relational analysis result of NS_A2_B2_A1_B1_A1_B2_B1
Status: Status.UNKNOWN
Output dim: 3, lower bound: -20.5927453, upper bound: 20.6168521
time: 0.56 seconds

## Relational analysis of NS_A2_B2_A1_B1_A1_B2_B2
Optimization infeasible because this subproblem isn't reachable.

### Candidate
type: A, layer: 1, pos: 36

## Relational analysis of NS_A2_B2_A1_B1_A1_B2_A1

### Relational analysis result of NS_A2_B2_A1_B1_A1_B2_A1
Status: Status.UNKNOWN
Output dim: 3, lower bound: -20.5935250, upper bound: 20.6831332
time: 0.60 seconds

## Relational analysis of NS_A2_B2_A1_B1_A1_B2_A2

### Relational analysis result of NS_A2_B2_A1_B1_A1_B2_A2
Status: Status.UNKNOWN
Output dim: 3, lower bound: -20.5935250, upper bound: 20.6831345
time: 0.62 seconds

## BFS NS instance: NS_A2_B2_A1_B2_A1_B1

### Backsubstitution after applying NS history:
0: -95.0735168, 132.7543030, -105.9509964, 146.1819763, -241.2554932, 238.7052917
1: -12.3337259, 11.3693285, -13.5341396, 12.5975742, -24.9313011, 24.9034672
2: -7.0452828, 11.6368580, -7.7957573, 12.8536930, -19.8989697, 19.4326153
3: -5.6269884, 12.7925835, -6.2829480, 14.0652695, -19.6922569, 19.0755310
4: -8.6930714, 10.4093132, -9.6299067, 11.5145483, -20.2076149, 20.0392151

Time for backsubstitution: 2.57 seconds

### NS candidates at layer 1
type: B, layer: 1, pos: 0
type: B, layer: 1, pos: 6
type: A, layer: 1, pos: 36
type: A, layer: 1, pos: 6
type: B, layer: 1, pos: 28
type: B, layer: 1, pos: 47
type: A, layer: 1, pos: 28
type: B, layer: 1, pos: 21
type: B, layer: 1, pos: 24
type: A, layer: 1, pos: 24
type: B, layer: 1, pos: 37
type: A, layer: 1, pos: 0
type: A, layer: 1, pos: 35
type: A, layer: 1, pos: 47

Time for candidate selection: 0.21 seconds

### Candidate
type: B, layer: 1, pos: 0

### Candidate
type: B, layer: 1, pos: 6

## Relational analysis of NS_A2_B2_A1_B2_A1_B1_B1

### Relational analysis result of NS_A2_B2_A1_B2_A1_B1_B1
Status: Status.UNKNOWN
Output dim: 3, lower bound: -20.5664600, upper bound: 20.6168460
time: 0.58 seconds

## Relational analysis of NS_A2_B2_A1_B2_A1_B1_B2
Optimization infeasible because this subproblem isn't reachable.

### Candidate
type: A, layer: 1, pos: 36

## Relational analysis of NS_A2_B2_A1_B2_A1_B1_A1

### Relational analysis result of NS_A2_B2_A1_B2_A1_B1_A1
Status: Status.UNKNOWN
Output dim: 3, lower bound: -20.5664651, upper bound: 20.6831271
time: 0.58 seconds

## Relational analysis of NS_A2_B2_A1_B2_A1_B1_A2

### Relational analysis result of NS_A2_B2_A1_B2_A1_B1_A2
Status: Status.UNKNOWN
Output dim: 3, lower bound: -20.5664677, upper bound: 20.6831277
time: 0.64 seconds

## BFS NS instance: NS_A2_B2_A1_B2_A1_B2

### Backsubstitution after applying NS history:
0: -95.0735168, 132.7543030, -105.4650345, 145.7380829, -240.8115997, 238.2193298
1: -12.3337259, 11.3693285, -13.4724770, 12.5768051, -24.9105301, 24.8418045
2: -7.0452828, 11.6368580, -7.7824373, 12.8118734, -19.8571568, 19.4192963
3: -5.6269884, 12.7925835, -6.2930226, 13.9913244, -19.6183128, 19.0855999
4: -8.6930714, 10.4093132, -9.5947895, 11.4718676, -20.1649361, 20.0041027

Time for backsubstitution: 2.56 seconds

### NS candidates at layer 1
type: B, layer: 1, pos: 0
type: B, layer: 1, pos: 28
type: A, layer: 1, pos: 28
type: B, layer: 1, pos: 21
type: B, layer: 1, pos: 47
type: B, layer: 1, pos: 37
type: A, layer: 1, pos: 36
type: B, layer: 1, pos: 6
type: A, layer: 1, pos: 35
type: B, layer: 1, pos: 24
type: A, layer: 1, pos: 24
type: A, layer: 1, pos: 0
type: A, layer: 1, pos: 6
type: A, layer: 1, pos: 47

Time for candidate selection: 0.21 seconds

### Candidate
type: B, layer: 1, pos: 0

### Candidate
type: B, layer: 1, pos: 28

## Relational analysis of NS_A2_B2_A1_B2_A1_B2_B1

### Relational analysis result of NS_A2_B2_A1_B2_A1_B2_B1
Status: Status.UNKNOWN
Output dim: 3, lower bound: -20.5650441, upper bound: 20.6812461
time: 1.02 seconds

## Relational analysis of NS_A2_B2_A1_B2_A1_B2_B2

### Relational analysis result of NS_A2_B2_A1_B2_A1_B2_B2
Status: Status.UNKNOWN
Output dim: 3, lower bound: -20.5564971, upper bound: 20.6812446
time: 0.70 seconds

## BFS NS instance: NS_A2_B2_A2_A1_B1_B1

### Backsubstitution after applying NS history:
0: -95.0689392, 132.8676910, -107.8430099, 148.7107391, -243.7796783, 240.7106628
1: -12.3253155, 11.3997688, -13.7578459, 12.8343000, -25.1596146, 25.1576138
2: -7.0606232, 11.6485138, -7.9261131, 13.0691128, -20.1297359, 19.5746269
3: -5.6620207, 12.7705164, -6.4137383, 14.3320303, -19.9940510, 19.1842518
4: -8.7019453, 10.4130878, -9.8160381, 11.7135620, -20.4155045, 20.2291241

Time for backsubstitution: 2.57 seconds

### NS candidates at layer 1
type: B, layer: 1, pos: 0
type: B, layer: 1, pos: 47
type: B, layer: 1, pos: 36
type: B, layer: 1, pos: 37
type: A, layer: 1, pos: 28
type: B, layer: 1, pos: 21
type: A, layer: 1, pos: 36
type: B, layer: 1, pos: 6
type: B, layer: 1, pos: 24
type: A, layer: 1, pos: 24
type: A, layer: 1, pos: 6
type: A, layer: 1, pos: 47
type: A, layer: 1, pos: 0
type: A, layer: 1, pos: 35

Time for candidate selection: 0.21 seconds

### Candidate
type: B, layer: 1, pos: 0

### Candidate
type: B, layer: 1, pos: 47

## Relational analysis of NS_A2_B2_A2_A1_B1_B1_B1
Optimization infeasible because this subproblem isn't reachable.

### Candidate
type: B, layer: 1, pos: 36

## Relational analysis of NS_A2_B2_A2_A1_B1_B1_B1

### Relational analysis result of NS_A2_B2_A2_A1_B1_B1_B1
Status: Status.UNKNOWN
Output dim: 3, lower bound: -20.6012258, upper bound: 20.6706618
time: 0.61 seconds

## Relational analysis of NS_A2_B2_A2_A1_B1_B1_B2

### Relational analysis result of NS_A2_B2_A2_A1_B1_B1_B2
Status: Status.UNKNOWN
Output dim: 3, lower bound: -20.5695071, upper bound: 20.6706566
time: 0.66 seconds

## BFS NS instance: NS_A2_B2_A2_A1_B1_B2

### Backsubstitution after applying NS history:
0: -95.0689392, 132.8676910, -105.5044022, 145.4144592, -240.4833984, 238.3720856
1: -12.3253155, 11.3997688, -13.4587669, 12.5549593, -24.8802700, 24.8585339
2: -7.0606232, 11.6485138, -7.7575073, 12.7811861, -19.8418064, 19.4060211
3: -5.6620207, 12.7705164, -6.2657366, 14.0150375, -19.6770592, 19.0362511
4: -8.7019453, 10.4130878, -9.6030197, 11.4533844, -20.1553268, 20.0161076

Time for backsubstitution: 2.58 seconds

### NS candidates at layer 1
type: B, layer: 1, pos: 0
type: B, layer: 1, pos: 37
type: B, layer: 1, pos: 36
type: B, layer: 1, pos: 47
type: A, layer: 1, pos: 28
type: B, layer: 1, pos: 21
type: A, layer: 1, pos: 36
type: B, layer: 1, pos: 6
type: B, layer: 1, pos: 24
type: A, layer: 1, pos: 6
type: A, layer: 1, pos: 24
type: A, layer: 1, pos: 47
type: A, layer: 1, pos: 0
type: A, layer: 1, pos: 35

Time for candidate selection: 0.21 seconds

### Candidate
type: B, layer: 1, pos: 0

### Candidate
type: B, layer: 1, pos: 37

## Relational analysis of NS_A2_B2_A2_A1_B1_B2_B1

### Relational analysis result of NS_A2_B2_A2_A1_B1_B2_B1
Status: Status.UNKNOWN
Output dim: 3, lower bound: -20.5936389, upper bound: 20.6804284
time: 0.56 seconds

## Relational analysis of NS_A2_B2_A2_A1_B1_B2_B2

### Relational analysis result of NS_A2_B2_A2_A1_B1_B2_B2
Status: Status.UNKNOWN
Output dim: 3, lower bound: -20.5936389, upper bound: 20.6804274
time: 0.61 seconds

## BFS NS instance: NS_A2_B2_A2_A1_B2_B1

### Backsubstitution after applying NS history:
0: -95.0689392, 132.8676910, -106.1239853, 146.5814514, -241.6503906, 238.9916687
1: -12.3253155, 11.3997688, -13.5391121, 12.6741037, -24.9994202, 24.9388809
2: -7.0606232, 11.6485138, -7.8372231, 12.8767986, -19.9374180, 19.4857330
3: -5.6620207, 12.7705164, -6.3524704, 14.0922661, -19.7542877, 19.1229858
4: -8.7019453, 10.4130878, -9.6710129, 11.5341730, -20.2361183, 20.0841007

Time for backsubstitution: 2.58 seconds

### NS candidates at layer 1
type: B, layer: 1, pos: 0
type: B, layer: 1, pos: 47
type: B, layer: 1, pos: 21
type: A, layer: 1, pos: 28
type: B, layer: 1, pos: 36
type: B, layer: 1, pos: 37
type: A, layer: 1, pos: 36
type: B, layer: 1, pos: 6
type: B, layer: 1, pos: 24
type: A, layer: 1, pos: 24
type: A, layer: 1, pos: 6
type: A, layer: 1, pos: 47
type: A, layer: 1, pos: 0
type: A, layer: 1, pos: 35

Time for candidate selection: 0.21 seconds

### Candidate
type: B, layer: 1, pos: 0

### Candidate
type: B, layer: 1, pos: 47

## Relational analysis of NS_A2_B2_A2_A1_B2_B1_B1
Optimization infeasible because this subproblem isn't reachable.

### Candidate
type: B, layer: 1, pos: 21

## Relational analysis of NS_A2_B2_A2_A1_B2_B1_B1

### Relational analysis result of NS_A2_B2_A2_A1_B2_B1_B1
Status: Status.UNKNOWN
Output dim: 3, lower bound: -20.5910133, upper bound: 20.6804302
time: 1.02 seconds

## Relational analysis of NS_A2_B2_A2_A1_B2_B1_B2

### Relational analysis result of NS_A2_B2_A2_A1_B2_B1_B2
Status: Status.UNKNOWN
Output dim: 3, lower bound: -20.5910129, upper bound: 20.6804307
time: 0.69 seconds

## BFS NS instance: NS_A2_B2_A2_A1_B2_B2

### Backsubstitution after applying NS history:
0: -95.0689392, 132.8676910, -103.9513626, 143.6722870, -238.7412262, 236.8190308
1: -12.3253155, 11.3997688, -13.2803516, 12.4157372, -24.7410488, 24.6801205
2: -7.0606232, 11.6485138, -7.6763110, 12.6231537, -19.6837769, 19.3248253
3: -5.6620207, 12.7705164, -6.2133050, 13.8134508, -19.4754715, 18.9838181
4: -8.7019453, 10.4130878, -9.4775314, 11.3042612, -20.0062027, 19.8906155

Time for backsubstitution: 2.58 seconds

### NS candidates at layer 1
type: B, layer: 1, pos: 0
type: B, layer: 1, pos: 21
type: B, layer: 1, pos: 36
type: B, layer: 1, pos: 47
type: A, layer: 1, pos: 28
type: B, layer: 1, pos: 6
type: B, layer: 1, pos: 37
type: A, layer: 1, pos: 36
type: A, layer: 1, pos: 6
type: B, layer: 1, pos: 24
type: A, layer: 1, pos: 24
type: A, layer: 1, pos: 35
type: A, layer: 1, pos: 0
type: A, layer: 1, pos: 47

Time for candidate selection: 0.21 seconds

### Candidate
type: B, layer: 1, pos: 0

### Candidate
type: B, layer: 1, pos: 21

## Relational analysis of NS_A2_B2_A2_A1_B2_B2_B1

### Relational analysis result of NS_A2_B2_A2_A1_B2_B2_B1
Status: Status.UNKNOWN
Output dim: 3, lower bound: -20.5654298, upper bound: 20.6804264
time: 0.75 seconds

## Relational analysis of NS_A2_B2_A2_A1_B2_B2_B2
Optimization infeasible because this subproblem isn't reachable.

### Candidate
type: B, layer: 1, pos: 36

## Relational analysis of NS_A2_B2_A2_A1_B2_B2_B1

### Relational analysis result of NS_A2_B2_A2_A1_B2_B2_B1
Status: Status.UNKNOWN
Output dim: 3, lower bound: -20.5648507, upper bound: 20.6706561
time: 0.62 seconds

## Relational analysis of NS_A2_B2_A2_A1_B2_B2_B2

### Relational analysis result of NS_A2_B2_A2_A1_B2_B2_B2
Status: Status.UNKNOWN
Output dim: 3, lower bound: -20.5564970, upper bound: 20.6706544
time: 0.63 seconds

## Summary of splitting at layer (split count: 6)
- Time for NS candidates: 5.46 seconds
NS_A1_B1_A2_A1_A1_B1_B1, status: Status.UNKNOWN, split count: 7, time: 5.46
Output dim: 3, lower bound: -20.5228503, upper bound: 20.6343363
NS_A1_B1_A2_A1_A1_B1_B2, status: Status.UNKNOWN, split count: 7, time: 5.46
Output dim: 3, lower bound: -20.5228503, upper bound: 20.6343322
NS_A1_B1_A2_A1_A1_B2_B1, status: Status.UNKNOWN, split count: 7, time: 5.46
Output dim: 3, lower bound: -20.5228503, upper bound: 20.6343335
NS_A1_B1_A2_A1_A1_B2_B2, status: Status.UNKNOWN, split count: 7, time: 5.46
Output dim: 3, lower bound: -20.5228503, upper bound: 20.6343359
NS_A1_B1_A2_A1_A2_B1_A1, status: Status.UNKNOWN, split count: 7, time: 5.46
Output dim: 3, lower bound: -20.5018379, upper bound: 20.6964961
NS_A1_B1_A2_A1_A2_B1_A2, status: Status.UNKNOWN, split count: 7, time: 5.46
Output dim: 3, lower bound: -20.5018374, upper bound: 20.6670930
NS_A1_B1_A2_A1_A2_B2_A1, status: Status.UNKNOWN, split count: 7, time: 5.46
Output dim: 3, lower bound: -20.5018379, upper bound: 20.6964973
NS_A1_B1_A2_A1_A2_B2_A2, status: Status.UNKNOWN, split count: 7, time: 5.46
Output dim: 3, lower bound: -20.5018377, upper bound: 20.6670934
NS_A1_B2_A1_B1_B1_A1_B1, status: Status.UNKNOWN, split count: 7, time: 5.46
Output dim: 3, lower bound: -20.6095015, upper bound: 20.6827620
NS_A1_B2_A1_B1_B1_A1_B2, status: Status.UNKNOWN, split count: 7, time: 5.46
Output dim: 3, lower bound: -20.6094975, upper bound: 20.6827622
NS_A1_B2_A1_B1_B1_A2_B1, status: Status.UNKNOWN, split count: 7, time: 5.46
Output dim: 3, lower bound: -20.6095010, upper bound: 20.6827631
NS_A1_B2_A1_B1_B1_A2_B2, status: Status.UNKNOWN, split count: 7, time: 5.46
Output dim: 3, lower bound: -20.6094993, upper bound: 20.6827557
NS_A1_B2_A1_B1_B2_A1_B1, status: Status.UNKNOWN, split count: 7, time: 5.46
Output dim: 3, lower bound: -20.6023319, upper bound: 20.6662785
NS_A1_B2_A1_B1_B2_A1_B2, status: Status.UNKNOWN, split count: 7, time: 5.46
Output dim: 3, lower bound: -20.5489739, upper bound: 20.6662697
NS_A1_B2_A1_B1_B2_A2_B1, status: Status.UNKNOWN, split count: 7, time: 5.46
Output dim: 3, lower bound: -20.6023319, upper bound: 20.6662777
NS_A1_B2_A1_B1_B2_A2_B2, status: Status.UNKNOWN, split count: 7, time: 5.46
Output dim: 3, lower bound: -20.5489739, upper bound: 20.6662698
NS_A1_B2_A1_B2_B1_B1_B1, status: Status.UNKNOWN, split count: 7, time: 5.46
Output dim: 3, lower bound: -20.4844750, upper bound: 20.6657644
NS_A1_B2_A1_B2_B1_B1_B2, status: Status.UNKNOWN, split count: 7, time: 5.46
Output dim: 3, lower bound: -20.4844792, upper bound: 20.6659409
NS_A1_B2_A1_B2_B1_B2_A1, status: Status.UNKNOWN, split count: 7, time: 5.46
Output dim: 3, lower bound: -20.5819115, upper bound: 20.6827185
NS_A1_B2_A1_B2_B1_B2_A2, status: Status.UNKNOWN, split count: 7, time: 5.46
Output dim: 3, lower bound: -20.5819115, upper bound: 20.6827173
NS_A1_B2_A1_B2_B2_A1_B1, status: Status.UNKNOWN, split count: 7, time: 5.46
Output dim: 3, lower bound: -20.5779191, upper bound: 20.6827544
NS_A1_B2_A1_B2_B2_A1_B2, status: Status.UNKNOWN, split count: 7, time: 5.46
Output dim: 3, lower bound: -20.5779186, upper bound: 20.6827531
NS_A1_B2_A1_B2_B2_A2_B1, status: Status.UNKNOWN, split count: 7, time: 5.46
Output dim: 3, lower bound: -20.5735387, upper bound: 20.6662721
NS_A1_B2_A1_B2_B2_A2_B2, status: Status.UNKNOWN, split count: 7, time: 5.46
Output dim: 3, lower bound: -20.5489739, upper bound: 20.6662697
NS_A1_B2_A2_B1_B1_B1_A1, status: Status.UNKNOWN, split count: 7, time: 5.46
Output dim: 3, lower bound: -20.7036472, upper bound: 20.5688159
NS_A1_B2_A2_B1_B1_B1_A2, status: Status.UNKNOWN, split count: 7, time: 5.46
Output dim: 3, lower bound: -20.7036429, upper bound: 20.5688182
NS_A1_B2_A2_B1_B1_B2_A1, status: Status.UNKNOWN, split count: 7, time: 5.46
Output dim: 3, lower bound: -20.7024944, upper bound: 20.5688187
NS_A1_B2_A2_B1_B1_B2_A2, status: Status.UNKNOWN, split count: 7, time: 5.46
Output dim: 3, lower bound: -20.7024944, upper bound: 20.5688180
NS_A2_B1_A2_A2_B1_A1_A1, status: Status.UNKNOWN, split count: 7, time: 5.46
Output dim: 3, lower bound: -20.5227843, upper bound: 20.7042775
NS_A2_B1_A2_A2_B1_A1_A2, status: Status.UNKNOWN, split count: 7, time: 5.46
Output dim: 3, lower bound: -20.5227833, upper bound: 20.7031628
NS_A2_B1_A2_A2_B1_A2_A1, status: Status.UNKNOWN, split count: 7, time: 5.46
Output dim: 3, lower bound: -20.5039903, upper bound: 20.6870095
NS_A2_B1_A2_A2_B1_A2_A2, status: Status.UNKNOWN, split count: 7, time: 5.46
Output dim: 3, lower bound: -20.5039886, upper bound: 20.6696066
NS_A2_B1_A2_A2_B2_A1_A1, status: Status.UNKNOWN, split count: 7, time: 5.46
Output dim: 3, lower bound: -20.5214582, upper bound: 20.7022298
NS_A2_B1_A2_A2_B2_A1_A2, status: Status.UNKNOWN, split count: 7, time: 5.46
Output dim: 3, lower bound: -20.5214597, upper bound: 20.6935607
NS_A2_B1_A2_A2_B2_A2_A1, status: Status.UNKNOWN, split count: 7, time: 5.46
Output dim: 3, lower bound: -20.5214563, upper bound: 20.6836290
NS_A2_B1_A2_A2_B2_A2_A2, status: Status.UNKNOWN, split count: 7, time: 5.46
Output dim: 3, lower bound: -20.5214582, upper bound: 20.6735237
NS_A2_B2_A1_B1_A1_B1_A1, status: Status.UNKNOWN, split count: 7, time: 5.46
Output dim: 3, lower bound: -20.5935250, upper bound: 20.6831325
NS_A2_B2_A1_B1_A1_B1_A2, status: Status.UNKNOWN, split count: 7, time: 5.46
Output dim: 3, lower bound: -20.5935250, upper bound: 20.6831340
NS_A2_B2_A1_B1_A1_B2_A1, status: Status.UNKNOWN, split count: 7, time: 5.46
Output dim: 3, lower bound: -20.5935250, upper bound: 20.6831332
NS_A2_B2_A1_B1_A1_B2_A2, status: Status.UNKNOWN, split count: 7, time: 5.46
Output dim: 3, lower bound: -20.5935250, upper bound: 20.6831345
NS_A2_B2_A1_B2_A1_B1_A1, status: Status.UNKNOWN, split count: 7, time: 5.46
Output dim: 3, lower bound: -20.5664651, upper bound: 20.6831271
NS_A2_B2_A1_B2_A1_B1_A2, status: Status.UNKNOWN, split count: 7, time: 5.46
Output dim: 3, lower bound: -20.5664677, upper bound: 20.6831277
NS_A2_B2_A1_B2_A1_B2_B1, status: Status.UNKNOWN, split count: 7, time: 5.46
Output dim: 3, lower bound: -20.5650441, upper bound: 20.6812461
NS_A2_B2_A1_B2_A1_B2_B2, status: Status.UNKNOWN, split count: 7, time: 5.46
Output dim: 3, lower bound: -20.5564971, upper bound: 20.6812446
NS_A2_B2_A2_A1_B1_B1_B1, status: Status.UNKNOWN, split count: 7, time: 5.46
Output dim: 3, lower bound: -20.6012258, upper bound: 20.6706618
NS_A2_B2_A2_A1_B1_B1_B2, status: Status.UNKNOWN, split count: 7, time: 5.46
Output dim: 3, lower bound: -20.5695071, upper bound: 20.6706566
NS_A2_B2_A2_A1_B1_B2_B1, status: Status.UNKNOWN, split count: 7, time: 5.46
Output dim: 3, lower bound: -20.5936389, upper bound: 20.6804284
NS_A2_B2_A2_A1_B1_B2_B2, status: Status.UNKNOWN, split count: 7, time: 5.46
Output dim: 3, lower bound: -20.5936389, upper bound: 20.6804274
NS_A2_B2_A2_A1_B2_B1_B1, status: Status.UNKNOWN, split count: 7, time: 5.46
Output dim: 3, lower bound: -20.5910133, upper bound: 20.6804302
NS_A2_B2_A2_A1_B2_B1_B2, status: Status.UNKNOWN, split count: 7, time: 5.46
Output dim: 3, lower bound: -20.5910129, upper bound: 20.6804307
NS_A2_B2_A2_A1_B2_B2_B1, status: Status.UNKNOWN, split count: 7, time: 5.46
Output dim: 3, lower bound: -20.5648507, upper bound: 20.6706561
NS_A2_B2_A2_A1_B2_B2_B2, status: Status.UNKNOWN, split count: 7, time: 5.46
Output dim: 3, lower bound: -20.5564970, upper bound: 20.6706544

## BFS NS instance: NS_A1_B1_A2_A1_A1_B1_B1

### Backsubstitution after applying NS history:
0: -108.7109680, 150.9740753, -57.0456734, 73.9323425, -182.6433105, 208.0197449
1: -13.9608555, 13.0281258, -6.7156248, 6.8167634, -20.7776184, 19.7437458
2: -8.0301933, 13.2491865, -4.1401720, 6.4069657, -14.4371586, 17.3893585
3: -6.4961653, 14.5776148, -3.3985119, 6.8698354, -13.3660011, 17.9761257
4: -9.9313498, 11.8866148, -5.2150559, 5.8094478, -15.7407970, 17.1016693

Time for backsubstitution: 2.54 seconds

### NS candidates at layer 1
type: A, layer: 1, pos: 0
type: B, layer: 1, pos: 21
type: A, layer: 1, pos: 47
type: A, layer: 1, pos: 7
type: A, layer: 1, pos: 24
type: A, layer: 1, pos: 6
type: B, layer: 1, pos: 24
type: B, layer: 1, pos: 7
type: B, layer: 1, pos: 28
type: B, layer: 1, pos: 6
type: B, layer: 1, pos: 47
type: A, layer: 1, pos: 21
type: B, layer: 1, pos: 0

Time for candidate selection: 0.21 seconds

### Candidate
type: A, layer: 1, pos: 0

### Candidate
type: B, layer: 1, pos: 21

### Candidate
type: A, layer: 1, pos: 47

### Candidate
type: A, layer: 1, pos: 7

### Candidate
type: A, layer: 1, pos: 24

### Candidate
type: A, layer: 1, pos: 6

### Candidate
type: B, layer: 1, pos: 24

### Candidate
type: B, layer: 1, pos: 7

### Candidate
type: B, layer: 1, pos: 28

## Relational analysis of NS_A1_B1_A2_A1_A1_B1_B1_B1
Optimization infeasible because this subproblem isn't reachable.

### Candidate
type: B, layer: 1, pos: 6

### Candidate
type: B, layer: 1, pos: 47

### Candidate
type: A, layer: 1, pos: 21

### Candidate
type: B, layer: 1, pos: 0

## BFS NS instance: NS_A1_B1_A2_A1_A1_B1_B2

### Backsubstitution after applying NS history:
0: -108.7109680, 150.9740753, -59.1718903, 76.3112793, -185.0222321, 210.1459656
1: -13.9608555, 13.0281258, -6.9227810, 7.0372496, -20.9981041, 19.9509029
2: -8.0301933, 13.2491865, -4.2806969, 6.6354141, -14.6656075, 17.5298843
3: -6.4961653, 14.5776148, -3.5207918, 7.0747633, -13.5709276, 18.0984058
4: -9.9313498, 11.8866148, -5.3875523, 6.0142002, -15.9455500, 17.2741642

Time for backsubstitution: 2.54 seconds

### NS candidates at layer 1
type: A, layer: 1, pos: 0
type: B, layer: 1, pos: 21
type: A, layer: 1, pos: 47
type: A, layer: 1, pos: 7
type: A, layer: 1, pos: 6
type: A, layer: 1, pos: 24
type: B, layer: 1, pos: 24
type: B, layer: 1, pos: 7
type: B, layer: 1, pos: 28
type: B, layer: 1, pos: 6
type: B, layer: 1, pos: 47
type: A, layer: 1, pos: 21
type: B, layer: 1, pos: 0

Time for candidate selection: 0.21 seconds

### Candidate
type: A, layer: 1, pos: 0

### Candidate
type: B, layer: 1, pos: 21

### Candidate
type: A, layer: 1, pos: 47

### Candidate
type: A, layer: 1, pos: 7

### Candidate
type: A, layer: 1, pos: 6

### Candidate
type: A, layer: 1, pos: 24

### Candidate
type: B, layer: 1, pos: 24

### Candidate
type: B, layer: 1, pos: 7

### Candidate
type: B, layer: 1, pos: 28

## Relational analysis of NS_A1_B1_A2_A1_A1_B1_B2_B1
Optimization infeasible because this subproblem isn't reachable.

### Candidate
type: B, layer: 1, pos: 6

### Candidate
type: B, layer: 1, pos: 47

### Candidate
type: A, layer: 1, pos: 21

### Candidate
type: B, layer: 1, pos: 0

## BFS NS instance: NS_A1_B1_A2_A1_A1_B2_B1

### Backsubstitution after applying NS history:
0: -108.7109680, 150.9740753, -56.4241676, 73.5239182, -182.2348480, 207.3982391
1: -13.9608555, 13.0281258, -6.7107825, 6.7340088, -20.6948624, 19.7389069
2: -8.0301933, 13.2491865, -4.1111112, 6.3606029, -14.3907967, 17.3602982
3: -6.4961653, 14.5776148, -3.3499129, 6.8255262, -13.3216915, 17.9275246
4: -9.9313498, 11.8866148, -5.1579680, 5.7657614, -15.6971111, 17.0445786

Time for backsubstitution: 2.53 seconds

### NS candidates at layer 1
type: A, layer: 1, pos: 0
type: B, layer: 1, pos: 21
type: A, layer: 1, pos: 47
type: A, layer: 1, pos: 7
type: A, layer: 1, pos: 6
type: A, layer: 1, pos: 24
type: B, layer: 1, pos: 24
type: B, layer: 1, pos: 7
type: B, layer: 1, pos: 28
type: B, layer: 1, pos: 6
type: B, layer: 1, pos: 47
type: A, layer: 1, pos: 21
type: B, layer: 1, pos: 0

Time for candidate selection: 0.21 seconds

### Candidate
type: A, layer: 1, pos: 0

### Candidate
type: B, layer: 1, pos: 21

### Candidate
type: A, layer: 1, pos: 47

### Candidate
type: A, layer: 1, pos: 7

### Candidate
type: A, layer: 1, pos: 6

### Candidate
type: A, layer: 1, pos: 24

### Candidate
type: B, layer: 1, pos: 24

### Candidate
type: B, layer: 1, pos: 7

### Candidate
type: B, layer: 1, pos: 28

### Candidate
type: B, layer: 1, pos: 6

### Candidate
type: B, layer: 1, pos: 47

### Candidate
type: A, layer: 1, pos: 21

### Candidate
type: B, layer: 1, pos: 0

## BFS NS instance: NS_A1_B1_A2_A1_A1_B2_B2

### Backsubstitution after applying NS history:
0: -108.7109680, 150.9740753, -56.8997726, 73.5524826, -182.2634583, 207.8738403
1: -13.9608555, 13.0281258, -6.7002521, 6.7609510, -20.7218056, 19.7283745
2: -8.0301933, 13.2491865, -4.1285939, 6.3786354, -14.4088287, 17.3777809
3: -6.4961653, 14.5776148, -3.3739226, 6.8207431, -13.3169079, 17.9515381
4: -9.9313498, 11.8866148, -5.1795106, 5.7839584, -15.7153082, 17.0661240

Time for backsubstitution: 2.53 seconds

### NS candidates at layer 1
type: A, layer: 1, pos: 0
type: B, layer: 1, pos: 21
type: A, layer: 1, pos: 47
type: A, layer: 1, pos: 7
type: A, layer: 1, pos: 6
type: A, layer: 1, pos: 24
type: B, layer: 1, pos: 24
type: B, layer: 1, pos: 7
type: B, layer: 1, pos: 28
type: B, layer: 1, pos: 6
type: B, layer: 1, pos: 47
type: A, layer: 1, pos: 21
type: B, layer: 1, pos: 0

Time for candidate selection: 0.21 seconds

### Candidate
type: A, layer: 1, pos: 0

### Candidate
type: B, layer: 1, pos: 21

### Candidate
type: A, layer: 1, pos: 47

### Candidate
type: A, layer: 1, pos: 7

### Candidate
type: A, layer: 1, pos: 6

### Candidate
type: A, layer: 1, pos: 24

### Candidate
type: B, layer: 1, pos: 24

### Candidate
type: B, layer: 1, pos: 7

### Candidate
type: B, layer: 1, pos: 28

### Candidate
type: B, layer: 1, pos: 6

### Candidate
type: B, layer: 1, pos: 47

### Candidate
type: A, layer: 1, pos: 21

### Candidate
type: B, layer: 1, pos: 0

## BFS NS instance: NS_A1_B1_A2_A1_A2_B1_A1

### Backsubstitution after applying NS history:
0: -97.8743515, 135.7075958, -60.7692795, 78.7800903, -176.6544342, 196.4768524
1: -12.5542641, 11.7495852, -7.1476021, 7.2403092, -19.7945728, 18.8971863
2: -7.2430701, 11.8971634, -4.4023685, 6.8436527, -14.0867233, 16.2995319
3: -5.8512888, 13.1052275, -3.6247139, 7.3066416, -13.1579304, 16.7299423
4: -8.9598989, 10.6746407, -5.5420227, 6.2030392, -15.1629362, 16.2166634

Time for backsubstitution: 2.53 seconds

### NS candidates at layer 1
type: A, layer: 1, pos: 0
type: B, layer: 1, pos: 21
type: A, layer: 1, pos: 47
type: A, layer: 1, pos: 7
type: A, layer: 1, pos: 6
type: B, layer: 1, pos: 7
type: B, layer: 1, pos: 24
type: B, layer: 1, pos: 36
type: B, layer: 1, pos: 28
type: B, layer: 1, pos: 6
type: B, layer: 1, pos: 47
type: A, layer: 1, pos: 21
type: B, layer: 1, pos: 0

Time for candidate selection: 0.21 seconds

### Candidate
type: A, layer: 1, pos: 0

### Candidate
type: B, layer: 1, pos: 21

### Candidate
type: A, layer: 1, pos: 47

### Candidate
type: A, layer: 1, pos: 7

### Candidate
type: A, layer: 1, pos: 6

### Candidate
type: B, layer: 1, pos: 7

### Candidate
type: B, layer: 1, pos: 24

## Relational analysis of NS_A1_B1_A2_A1_A2_B1_A1_B1

### Relational analysis result of NS_A1_B1_A2_A1_A2_B1_A1_B1
Status: Status.UNKNOWN
Output dim: 3, lower bound: -20.5018371, upper bound: 20.6670910
time: 0.58 seconds

## Relational analysis of NS_A1_B1_A2_A1_A2_B1_A1_B2

### Relational analysis result of NS_A1_B1_A2_A1_A2_B1_A1_B2
Status: Status.UNKNOWN
Output dim: 3, lower bound: -20.5018371, upper bound: 20.6670930
time: 0.58 seconds

## BFS NS instance: NS_A1_B1_A2_A1_A2_B1_A2

### Backsubstitution after applying NS history:
0: -99.7806015, 138.4137115, -60.7692795, 78.7800903, -178.5606995, 199.1829529
1: -12.8008957, 11.9618216, -7.1476021, 7.2403092, -20.0412045, 19.1094227
2: -7.3644843, 12.1489353, -4.4023685, 6.8436527, -14.2081375, 16.5513039
3: -5.9537787, 13.3352900, -3.6247139, 7.3066416, -13.2604198, 16.9600029
4: -9.1147022, 10.8933620, -5.5420227, 6.2030392, -15.3177414, 16.4353848

Time for backsubstitution: 2.54 seconds

### NS candidates at layer 1
type: A, layer: 1, pos: 0
type: B, layer: 1, pos: 21
type: A, layer: 1, pos: 47
type: A, layer: 1, pos: 7
type: A, layer: 1, pos: 6
type: B, layer: 1, pos: 7
type: B, layer: 1, pos: 24
type: B, layer: 1, pos: 36
type: B, layer: 1, pos: 28
type: B, layer: 1, pos: 6
type: B, layer: 1, pos: 47
type: A, layer: 1, pos: 21
type: B, layer: 1, pos: 0

Time for candidate selection: 0.21 seconds

### Candidate
type: A, layer: 1, pos: 0

### Candidate
type: B, layer: 1, pos: 21

### Candidate
type: A, layer: 1, pos: 47

### Candidate
type: A, layer: 1, pos: 7

### Candidate
type: A, layer: 1, pos: 6

### Candidate
type: B, layer: 1, pos: 7

### Candidate
type: B, layer: 1, pos: 24

## Relational analysis of NS_A1_B1_A2_A1_A2_B1_A2_B1

### Relational analysis result of NS_A1_B1_A2_A1_A2_B1_A2_B1
Status: Status.UNKNOWN
Output dim: 3, lower bound: -20.5018348, upper bound: 20.6670880
time: 0.58 seconds

## Relational analysis of NS_A1_B1_A2_A1_A2_B1_A2_B2

### Relational analysis result of NS_A1_B1_A2_A1_A2_B1_A2_B2
Status: Status.UNKNOWN
Output dim: 3, lower bound: -20.5018348, upper bound: 20.6670909
time: 0.53 seconds

## BFS NS instance: NS_A1_B1_A2_A1_A2_B2_A1

### Backsubstitution after applying NS history:
0: -97.8743515, 135.7075958, -59.1519318, 76.8340149, -174.7083740, 194.8595276
1: -12.5542641, 11.7495852, -6.9958315, 7.0370650, -19.5913296, 18.7454147
2: -7.2430701, 11.8971634, -4.2961965, 6.6580710, -13.9011412, 16.1933594
3: -5.8512888, 13.1052275, -3.5116916, 7.1283474, -12.9796362, 16.6169186
4: -8.9598989, 10.6746407, -5.3880043, 6.0388904, -14.9987879, 16.0626450

Time for backsubstitution: 2.54 seconds

### NS candidates at layer 1
type: A, layer: 1, pos: 0
type: B, layer: 1, pos: 21
type: A, layer: 1, pos: 47
type: A, layer: 1, pos: 7
type: A, layer: 1, pos: 6
type: B, layer: 1, pos: 24
type: B, layer: 1, pos: 7
type: B, layer: 1, pos: 28
type: B, layer: 1, pos: 36
type: B, layer: 1, pos: 6
type: B, layer: 1, pos: 47
type: A, layer: 1, pos: 21
type: B, layer: 1, pos: 0

Time for candidate selection: 0.21 seconds

### Candidate
type: A, layer: 1, pos: 0

### Candidate
type: B, layer: 1, pos: 21

### Candidate
type: A, layer: 1, pos: 47

### Candidate
type: A, layer: 1, pos: 7

### Candidate
type: A, layer: 1, pos: 6

## Relational analysis of NS_A1_B1_A2_A1_A2_B2_A1_A1

### Relational analysis result of NS_A1_B1_A2_A1_A2_B2_A1_A1
Status: Status.VERIFIED
Output dim: 3, lower bound: -20.3248176, upper bound: 20.5959364
time: 0.67 seconds

## Relational analysis of NS_A1_B1_A2_A1_A2_B2_A1_A2
Optimization infeasible because this subproblem isn't reachable.

### Candidate
type: B, layer: 1, pos: 24

## Relational analysis of NS_A1_B1_A2_A1_A2_B2_A1_B1

### Relational analysis result of NS_A1_B1_A2_A1_A2_B2_A1_B1
Status: Status.UNKNOWN
Output dim: 3, lower bound: -20.5056991, upper bound: 20.6670931
time: 0.61 seconds

## Relational analysis of NS_A1_B1_A2_A1_A2_B2_A1_B2

### Relational analysis result of NS_A1_B1_A2_A1_A2_B2_A1_B2
Status: Status.UNKNOWN
Output dim: 3, lower bound: -20.5056991, upper bound: 20.6670935
time: 0.55 seconds

## BFS NS instance: NS_A1_B1_A2_A1_A2_B2_A2

### Backsubstitution after applying NS history:
0: -99.7806015, 138.4137115, -59.1519318, 76.8340149, -176.6146240, 197.5656433
1: -12.8008957, 11.9618216, -6.9958315, 7.0370650, -19.8379612, 18.9576511
2: -7.3644843, 12.1489353, -4.2961965, 6.6580710, -14.0225534, 16.4451294
3: -5.9537787, 13.3352900, -3.5116916, 7.1283474, -13.0821257, 16.8469791
4: -9.1147022, 10.8933620, -5.3880043, 6.0388904, -15.1535931, 16.2813663

Time for backsubstitution: 2.54 seconds

### NS candidates at layer 1
type: A, layer: 1, pos: 0
type: B, layer: 1, pos: 21
type: A, layer: 1, pos: 47
type: A, layer: 1, pos: 7
type: A, layer: 1, pos: 6
type: B, layer: 1, pos: 24
type: B, layer: 1, pos: 7
type: B, layer: 1, pos: 28
type: B, layer: 1, pos: 36
type: B, layer: 1, pos: 6
type: B, layer: 1, pos: 47
type: A, layer: 1, pos: 21
type: B, layer: 1, pos: 0

Time for candidate selection: 0.21 seconds

### Candidate
type: A, layer: 1, pos: 0

### Candidate
type: B, layer: 1, pos: 21

### Candidate
type: A, layer: 1, pos: 47

### Candidate
type: A, layer: 1, pos: 7

### Candidate
type: A, layer: 1, pos: 6

## Relational analysis of NS_A1_B1_A2_A1_A2_B2_A2_A1

### Relational analysis result of NS_A1_B1_A2_A1_A2_B2_A2_A1
Status: Status.VERIFIED
Output dim: 3, lower bound: -20.3248119, upper bound: 20.5714100
time: 0.60 seconds

## Relational analysis of NS_A1_B1_A2_A1_A2_B2_A2_A2
Optimization infeasible because this subproblem isn't reachable.

### Candidate
type: B, layer: 1, pos: 24

## Relational analysis of NS_A1_B1_A2_A1_A2_B2_A2_B1

### Relational analysis result of NS_A1_B1_A2_A1_A2_B2_A2_B1
Status: Status.UNKNOWN
Output dim: 3, lower bound: -20.5057012, upper bound: 20.6670934
time: 0.66 seconds

## Relational analysis of NS_A1_B1_A2_A1_A2_B2_A2_B2

### Relational analysis result of NS_A1_B1_A2_A1_A2_B2_A2_B2
Status: Status.UNKNOWN
Output dim: 3, lower bound: -20.5057012, upper bound: 20.6670935
time: 0.66 seconds

## BFS NS instance: NS_A1_B2_A1_B1_B1_A1_B1

### Backsubstitution after applying NS history:
0: -102.5664444, 143.4198303, -105.7482376, 147.0162506, -249.5826111, 249.1680603
1: -13.3056774, 12.3136339, -13.6290569, 12.6251478, -25.9308186, 25.9426918
2: -7.5930424, 12.5812550, -7.8028665, 12.9075041, -20.5005455, 20.3841171
3: -6.1116810, 13.8463879, -6.3008537, 14.1975565, -20.3092308, 20.1472416
4: -9.3920126, 11.2692289, -9.6589079, 11.5644083, -20.9564190, 20.9281349

Time for backsubstitution: 2.56 seconds

### NS candidates at layer 1
type: B, layer: 1, pos: 0
type: A, layer: 1, pos: 7
type: B, layer: 1, pos: 7
type: B, layer: 1, pos: 47
type: A, layer: 1, pos: 28
type: A, layer: 1, pos: 6
type: B, layer: 1, pos: 24
type: A, layer: 1, pos: 24
type: B, layer: 1, pos: 6
type: A, layer: 1, pos: 35
type: A, layer: 1, pos: 0
type: A, layer: 1, pos: 47
type: B, layer: 1, pos: 37

Time for candidate selection: 0.21 seconds

### Candidate
type: B, layer: 1, pos: 0

### Candidate
type: A, layer: 1, pos: 7

### Candidate
type: B, layer: 1, pos: 7

### Candidate
type: B, layer: 1, pos: 47

### Candidate
type: A, layer: 1, pos: 28

### Candidate
type: A, layer: 1, pos: 6

### Candidate
type: B, layer: 1, pos: 24

### Candidate
type: A, layer: 1, pos: 24

### Candidate
type: B, layer: 1, pos: 6

### Candidate
type: A, layer: 1, pos: 35

### Candidate
type: A, layer: 1, pos: 0

### Candidate
type: A, layer: 1, pos: 47

### Candidate
type: B, layer: 1, pos: 37

## Relational analysis of NS_A1_B2_A1_B1_B1_A1_B1_B1

### Relational analysis result of NS_A1_B2_A1_B1_B1_A1_B1_B1
Status: Status.UNKNOWN
Output dim: 3, lower bound: -20.6010745, upper bound: 20.6971449
time: 0.72 seconds

## Relational analysis of NS_A1_B2_A1_B1_B1_A1_B1_B2

### Relational analysis result of NS_A1_B2_A1_B1_B1_A1_B1_B2
Status: Status.UNKNOWN
Output dim: 3, lower bound: -20.6010746, upper bound: 20.6971487
time: 0.56 seconds

## BFS NS instance: NS_A1_B2_A1_B1_B1_A1_B2

### Backsubstitution after applying NS history:
0: -102.5664444, 143.4198303, -126.4183044, 177.1730347, -279.7394104, 269.8380737
1: -13.3056774, 12.3136339, -16.4308586, 15.1561155, -28.4617901, 28.7444916
2: -7.5930424, 12.5812550, -9.3592567, 15.5399828, -23.1330242, 21.9405079
3: -6.1116810, 13.8463879, -7.5596809, 17.0870991, -23.1987801, 21.4060688
4: -9.3920126, 11.2692289, -11.5780315, 13.9205914, -23.3126030, 22.8472595

Time for backsubstitution: 2.55 seconds

### NS candidates at layer 1
type: B, layer: 1, pos: 0
type: A, layer: 1, pos: 7
type: B, layer: 1, pos: 7
type: B, layer: 1, pos: 47
type: B, layer: 1, pos: 6
type: A, layer: 1, pos: 6
type: A, layer: 1, pos: 28
type: B, layer: 1, pos: 24
type: A, layer: 1, pos: 24
type: A, layer: 1, pos: 47
type: B, layer: 1, pos: 37
type: A, layer: 1, pos: 0
type: A, layer: 1, pos: 35

Time for candidate selection: 0.21 seconds

### Candidate
type: B, layer: 1, pos: 0

### Candidate
type: A, layer: 1, pos: 7

### Candidate
type: B, layer: 1, pos: 7

### Candidate
type: B, layer: 1, pos: 47

### Candidate
type: B, layer: 1, pos: 6

### Candidate
type: A, layer: 1, pos: 6

### Candidate
type: A, layer: 1, pos: 28

### Candidate
type: B, layer: 1, pos: 24

### Candidate
type: A, layer: 1, pos: 24

### Candidate
type: A, layer: 1, pos: 47

### Candidate
type: B, layer: 1, pos: 37

## Relational analysis of NS_A1_B2_A1_B1_B1_A1_B2_B1

### Relational analysis result of NS_A1_B2_A1_B1_B1_A1_B2_B1
Status: Status.UNKNOWN
Output dim: 3, lower bound: -20.6010746, upper bound: 20.6971430
time: 0.56 seconds

## Relational analysis of NS_A1_B2_A1_B1_B1_A1_B2_B2

### Relational analysis result of NS_A1_B2_A1_B1_B1_A1_B2_B2
Status: Status.UNKNOWN
Output dim: 3, lower bound: -20.6010746, upper bound: 20.6971475
time: 0.63 seconds

## BFS NS instance: NS_A1_B2_A1_B1_B1_A2_B1

### Backsubstitution after applying NS history:
0: -105.0573273, 146.1588287, -105.7482376, 147.0162506, -252.0735779, 251.9070435
1: -13.5524206, 12.5621405, -13.6290569, 12.6251478, -26.1775684, 26.1911964
2: -7.7492042, 12.8493176, -7.8028665, 12.9075041, -20.6567039, 20.6521835
3: -6.2506890, 14.1001730, -6.3008537, 14.1975565, -20.4482384, 20.4010239
4: -9.5892010, 11.5065594, -9.6589079, 11.5644083, -21.1536102, 21.1654663

Time for backsubstitution: 2.56 seconds

### NS candidates at layer 1
type: A, layer: 1, pos: 7
type: B, layer: 1, pos: 0
type: B, layer: 1, pos: 7
type: A, layer: 1, pos: 35
type: A, layer: 1, pos: 0
type: B, layer: 1, pos: 47
type: A, layer: 1, pos: 28
type: A, layer: 1, pos: 6
type: B, layer: 1, pos: 24
type: A, layer: 1, pos: 24
type: B, layer: 1, pos: 6
type: A, layer: 1, pos: 47
type: B, layer: 1, pos: 37

Time for candidate selection: 0.21 seconds

### Candidate
type: A, layer: 1, pos: 7

### Candidate
type: B, layer: 1, pos: 0

### Candidate
type: B, layer: 1, pos: 7

### Candidate
type: A, layer: 1, pos: 35

### Candidate
type: A, layer: 1, pos: 0

### Candidate
type: B, layer: 1, pos: 47

### Candidate
type: A, layer: 1, pos: 28

### Candidate
type: A, layer: 1, pos: 6

### Candidate
type: B, layer: 1, pos: 24

### Candidate
type: A, layer: 1, pos: 24

### Candidate
type: B, layer: 1, pos: 6

### Candidate
type: A, layer: 1, pos: 47

### Candidate
type: B, layer: 1, pos: 37

## Relational analysis of NS_A1_B2_A1_B1_B1_A2_B1_B1

### Relational analysis result of NS_A1_B2_A1_B1_B1_A2_B1_B1
Status: Status.UNKNOWN
Output dim: 3, lower bound: -20.6010742, upper bound: 20.6827565
time: 0.65 seconds

## Relational analysis of NS_A1_B2_A1_B1_B1_A2_B1_B2

### Relational analysis result of NS_A1_B2_A1_B1_B1_A2_B1_B2
Status: Status.UNKNOWN
Output dim: 3, lower bound: -20.6010742, upper bound: 20.6827608
time: 0.61 seconds

## BFS NS instance: NS_A1_B2_A1_B1_B1_A2_B2

### Backsubstitution after applying NS history:
0: -105.0573273, 146.1588287, -126.4183044, 177.1730347, -282.2303467, 272.5770874
1: -13.5524206, 12.5621405, -16.4308586, 15.1561155, -28.7085361, 28.9930000
2: -7.7492042, 12.8493176, -9.3592567, 15.5399828, -23.2891865, 22.2085743
3: -6.2506890, 14.1001730, -7.5596809, 17.0870991, -23.3377857, 21.6598549
4: -9.5892010, 11.5065594, -11.5780315, 13.9205914, -23.5097923, 23.0845909

Time for backsubstitution: 2.56 seconds

### NS candidates at layer 1
type: B, layer: 1, pos: 0
type: B, layer: 1, pos: 7
type: A, layer: 1, pos: 7
type: B, layer: 1, pos: 47
type: A, layer: 1, pos: 28
type: A, layer: 1, pos: 6
type: B, layer: 1, pos: 24
type: B, layer: 1, pos: 6
type: A, layer: 1, pos: 24
type: A, layer: 1, pos: 47
type: B, layer: 1, pos: 37
type: A, layer: 1, pos: 0
type: A, layer: 1, pos: 35

Time for candidate selection: 0.21 seconds

### Candidate
type: B, layer: 1, pos: 0

### Candidate
type: B, layer: 1, pos: 7

### Candidate
type: A, layer: 1, pos: 7

### Candidate
type: B, layer: 1, pos: 47

### Candidate
type: A, layer: 1, pos: 28

### Candidate
type: A, layer: 1, pos: 6

### Candidate
type: B, layer: 1, pos: 24

### Candidate
type: B, layer: 1, pos: 6

### Candidate
type: A, layer: 1, pos: 24

### Candidate
type: A, layer: 1, pos: 47

### Candidate
type: B, layer: 1, pos: 37

## Relational analysis of NS_A1_B2_A1_B1_B1_A2_B2_B1

### Relational analysis result of NS_A1_B2_A1_B1_B1_A2_B2_B1
Status: Status.UNKNOWN
Output dim: 3, lower bound: -20.6010742, upper bound: 20.6827571
time: 0.60 seconds

## Relational analysis of NS_A1_B2_A1_B1_B1_A2_B2_B2

### Relational analysis result of NS_A1_B2_A1_B1_B1_A2_B2_B2
Status: Status.UNKNOWN
Output dim: 3, lower bound: -20.6010742, upper bound: 20.6827590
time: 0.63 seconds

## BFS NS instance: NS_A1_B2_A1_B1_B2_A1_B1

### Backsubstitution after applying NS history:
0: -102.5664444, 143.4198303, -102.9929428, 142.2880096, -244.8544159, 246.4127808
1: -13.3056774, 12.3136339, -13.1744347, 12.2857018, -25.5913792, 25.4880676
2: -7.5930424, 12.5812550, -7.5945411, 12.4928875, -20.0859299, 20.1757927
3: -6.1116810, 13.8463879, -6.1373358, 13.7262249, -19.8379059, 19.9837227
4: -9.3920126, 11.2692289, -9.3911743, 11.2031307, -20.5951424, 20.6604004

Time for backsubstitution: 2.58 seconds

### NS candidates at layer 1
type: A, layer: 1, pos: 35
type: A, layer: 1, pos: 0
type: A, layer: 1, pos: 7
type: B, layer: 1, pos: 7
type: B, layer: 1, pos: 21
type: A, layer: 1, pos: 6
type: A, layer: 1, pos: 28
type: B, layer: 1, pos: 6
type: A, layer: 1, pos: 47
type: A, layer: 1, pos: 24
type: B, layer: 1, pos: 47
type: B, layer: 1, pos: 0
type: B, layer: 1, pos: 37

Time for candidate selection: 0.21 seconds

### Candidate
type: A, layer: 1, pos: 35

### Candidate
type: A, layer: 1, pos: 0

### Candidate
type: A, layer: 1, pos: 7

### Candidate
type: B, layer: 1, pos: 7

### Candidate
type: B, layer: 1, pos: 21

## Relational analysis of NS_A1_B2_A1_B1_B2_A1_B1_B1

### Relational analysis result of NS_A1_B2_A1_B1_B2_A1_B1_B1
Status: Status.UNKNOWN
Output dim: 3, lower bound: -20.6023339, upper bound: 20.6664754
time: 0.64 seconds

## Relational analysis of NS_A1_B2_A1_B1_B2_A1_B1_B2

### Relational analysis result of NS_A1_B2_A1_B1_B2_A1_B1_B2
Status: Status.UNKNOWN
Output dim: 3, lower bound: -20.6023316, upper bound: 20.6664735
time: 0.58 seconds

## BFS NS instance: NS_A1_B2_A1_B1_B2_A1_B2

### Backsubstitution after applying NS history:
0: -102.5664444, 143.4198303, -105.3224335, 145.5297089, -248.0960999, 248.7422638
1: -13.3056774, 12.3136339, -13.4735508, 12.5471287, -25.8528004, 25.7871857
2: -7.5930424, 12.5812550, -7.7453299, 12.7934780, -20.3865204, 20.3265781
3: -6.1116810, 13.8463879, -6.2631497, 14.0149574, -20.1266384, 20.1095371
4: -9.3920126, 11.2692289, -9.5815601, 11.4662514, -20.8582649, 20.8507862

Time for backsubstitution: 2.57 seconds

### NS candidates at layer 1
type: A, layer: 1, pos: 35
type: A, layer: 1, pos: 0
type: A, layer: 1, pos: 7
type: B, layer: 1, pos: 7
type: B, layer: 1, pos: 21
type: A, layer: 1, pos: 28
type: A, layer: 1, pos: 6
type: B, layer: 1, pos: 6
type: B, layer: 1, pos: 47
type: A, layer: 1, pos: 24
type: A, layer: 1, pos: 47
type: B, layer: 1, pos: 0
type: B, layer: 1, pos: 37

Time for candidate selection: 0.21 seconds

### Candidate
type: A, layer: 1, pos: 35

### Candidate
type: A, layer: 1, pos: 0

### Candidate
type: A, layer: 1, pos: 7

### Candidate
type: B, layer: 1, pos: 7

### Candidate
type: B, layer: 1, pos: 21

## Relational analysis of NS_A1_B2_A1_B1_B2_A1_B2_B1

### Relational analysis result of NS_A1_B2_A1_B1_B2_A1_B2_B1
Status: Status.UNKNOWN
Output dim: 3, lower bound: -20.5489743, upper bound: 20.6664620
time: 0.58 seconds

## Relational analysis of NS_A1_B2_A1_B1_B2_A1_B2_B2

### Relational analysis result of NS_A1_B2_A1_B1_B2_A1_B2_B2
Status: Status.UNKNOWN
Output dim: 3, lower bound: -20.5489738, upper bound: 20.6664696
time: 0.57 seconds

## BFS NS instance: NS_A1_B2_A1_B1_B2_A2_B1

### Backsubstitution after applying NS history:
0: -105.0573273, 146.1588287, -102.9929428, 142.2880096, -247.3453369, 249.1517487
1: -13.5524206, 12.5621405, -13.1744347, 12.2857018, -25.8381233, 25.7365761
2: -7.7492042, 12.8493176, -7.5945411, 12.4928875, -20.2420921, 20.4438572
3: -6.2506890, 14.1001730, -6.1373358, 13.7262249, -19.9769135, 20.2375031
4: -9.5892010, 11.5065594, -9.3911743, 11.2031307, -20.7923317, 20.8977337

Time for backsubstitution: 2.58 seconds

### NS candidates at layer 1
type: A, layer: 1, pos: 35
type: A, layer: 1, pos: 0
type: A, layer: 1, pos: 7
type: B, layer: 1, pos: 7
type: A, layer: 1, pos: 6
type: B, layer: 1, pos: 21
type: A, layer: 1, pos: 28
type: A, layer: 1, pos: 47
type: B, layer: 1, pos: 6
type: A, layer: 1, pos: 24
type: B, layer: 1, pos: 47
type: B, layer: 1, pos: 37
type: B, layer: 1, pos: 0

Time for candidate selection: 0.22 seconds

### Candidate
type: A, layer: 1, pos: 35

### Candidate
type: A, layer: 1, pos: 0

### Candidate
type: A, layer: 1, pos: 7

### Candidate
type: B, layer: 1, pos: 7

### Candidate
type: A, layer: 1, pos: 6

## Relational analysis of NS_A1_B2_A1_B1_B2_A2_B1_A1
Optimization infeasible because this subproblem isn't reachable.

### Candidate
type: B, layer: 1, pos: 21

## Relational analysis of NS_A1_B2_A1_B1_B2_A2_B1_B1

### Relational analysis result of NS_A1_B2_A1_B1_B2_A2_B1_B1
Status: Status.UNKNOWN
Output dim: 3, lower bound: -20.6023344, upper bound: 20.6662786
time: 0.57 seconds

## Relational analysis of NS_A1_B2_A1_B1_B2_A2_B1_B2

### Relational analysis result of NS_A1_B2_A1_B1_B2_A2_B1_B2
Status: Status.UNKNOWN
Output dim: 3, lower bound: -20.6023339, upper bound: 20.6662756
time: 0.63 seconds

## BFS NS instance: NS_A1_B2_A1_B1_B2_A2_B2

### Backsubstitution after applying NS history:
0: -105.0573273, 146.1588287, -105.3224335, 145.5297089, -250.5870361, 251.4812622
1: -13.5524206, 12.5621405, -13.4735508, 12.5471287, -26.0995483, 26.0356903
2: -7.7492042, 12.8493176, -7.7453299, 12.7934780, -20.5426826, 20.5946465
3: -6.2506890, 14.1001730, -6.2631497, 14.0149574, -20.2656441, 20.3633175
4: -9.5892010, 11.5065594, -9.5815601, 11.4662514, -21.0554523, 21.0881176

Time for backsubstitution: 2.58 seconds

### NS candidates at layer 1
type: A, layer: 1, pos: 35
type: A, layer: 1, pos: 0
type: A, layer: 1, pos: 7
type: B, layer: 1, pos: 7
type: B, layer: 1, pos: 21
type: A, layer: 1, pos: 6
type: A, layer: 1, pos: 28
type: B, layer: 1, pos: 6
type: A, layer: 1, pos: 47
type: A, layer: 1, pos: 24
type: B, layer: 1, pos: 47
type: B, layer: 1, pos: 0
type: B, layer: 1, pos: 37

Time for candidate selection: 0.22 seconds

### Candidate
type: A, layer: 1, pos: 35

### Candidate
type: A, layer: 1, pos: 0

### Candidate
type: A, layer: 1, pos: 7

### Candidate
type: B, layer: 1, pos: 7

### Candidate
type: B, layer: 1, pos: 21

## Relational analysis of NS_A1_B2_A1_B1_B2_A2_B2_B1

### Relational analysis result of NS_A1_B2_A1_B1_B2_A2_B2_B1
Status: Status.UNKNOWN
Output dim: 3, lower bound: -20.5489743, upper bound: 20.6662721
time: 0.58 seconds

## Relational analysis of NS_A1_B2_A1_B1_B2_A2_B2_B2

### Relational analysis result of NS_A1_B2_A1_B1_B2_A2_B2_B2
Status: Status.UNKNOWN
Output dim: 3, lower bound: -20.5489717, upper bound: 20.6662724
time: 0.66 seconds

## BFS NS instance: NS_A1_B2_A1_B2_B1_B1_B1

### Backsubstitution after applying NS history:
0: -107.4569092, 149.4062958, -99.8934860, 138.2301636, -245.6870728, 249.2997742
1: -13.8525238, 12.8498917, -12.8038139, 11.9218225, -25.7743454, 25.6537018
2: -7.9272017, 13.1349926, -7.3766441, 12.1441517, -20.0713539, 20.5116367
3: -6.3961639, 14.4240589, -5.9548283, 13.2885685, -19.6847305, 20.3788872
4: -9.8063383, 11.7648029, -9.1090450, 10.8691931, -20.6755276, 20.8738441

Time for backsubstitution: 2.59 seconds

### NS candidates at layer 1
type: A, layer: 1, pos: 35
type: A, layer: 1, pos: 0
type: A, layer: 1, pos: 7
type: B, layer: 1, pos: 7
type: A, layer: 1, pos: 47
type: A, layer: 1, pos: 36
type: A, layer: 1, pos: 6
type: A, layer: 1, pos: 28
type: A, layer: 1, pos: 24
type: B, layer: 1, pos: 24
type: B, layer: 1, pos: 6
type: B, layer: 1, pos: 37
type: B, layer: 1, pos: 0

Time for candidate selection: 0.21 seconds

### Candidate
type: A, layer: 1, pos: 35

### Candidate
type: A, layer: 1, pos: 0

### Candidate
type: A, layer: 1, pos: 7

### Candidate
type: B, layer: 1, pos: 7

### Candidate
type: A, layer: 1, pos: 47

## Relational analysis of NS_A1_B2_A1_B2_B1_B1_B1_A1
Optimization infeasible because this subproblem isn't reachable.

### Candidate
type: A, layer: 1, pos: 36

## Relational analysis of NS_A1_B2_A1_B2_B1_B1_B1_A1
Optimization infeasible because this subproblem isn't reachable.

### Candidate
type: A, layer: 1, pos: 6

## Relational analysis of NS_A1_B2_A1_B2_B1_B1_B1_A1
Optimization infeasible because this subproblem isn't reachable.

### Candidate
type: A, layer: 1, pos: 28

## Relational analysis of NS_A1_B2_A1_B2_B1_B1_B1_A1
Optimization infeasible because this subproblem isn't reachable.

### Candidate
type: A, layer: 1, pos: 24

## Relational analysis of NS_A1_B2_A1_B2_B1_B1_B1_A1

### Relational analysis result of NS_A1_B2_A1_B2_B1_B1_B1_A1
Status: Status.UNKNOWN
Output dim: 3, lower bound: -20.4784560, upper bound: 20.6513858
time: 0.55 seconds

## Relational analysis of NS_A1_B2_A1_B2_B1_B1_B1_A2

### Relational analysis result of NS_A1_B2_A1_B2_B1_B1_B1_A2
Status: Status.UNKNOWN
Output dim: 3, lower bound: -20.4784572, upper bound: 20.6597328
time: 0.65 seconds

## BFS NS instance: NS_A1_B2_A1_B2_B1_B1_B2

### Backsubstitution after applying NS history:
0: -107.4569092, 149.4062958, -120.1708679, 167.9312897, -275.3881836, 269.5771179
1: -13.8525238, 12.8498917, -15.5617046, 14.4091454, -28.2616673, 28.4115906
2: -7.9272017, 13.1349926, -8.9050169, 14.7326345, -22.6598358, 22.0400085
3: -6.3961639, 14.4240589, -7.1947503, 16.1275349, -22.5236988, 21.6188087
4: -9.8063383, 11.7648029, -10.9935894, 13.1865959, -22.9929333, 22.7583885

Time for backsubstitution: 2.60 seconds

### NS candidates at layer 1
type: A, layer: 1, pos: 35
type: A, layer: 1, pos: 0
type: B, layer: 1, pos: 7
type: A, layer: 1, pos: 7
type: A, layer: 1, pos: 36
type: A, layer: 1, pos: 47
type: A, layer: 1, pos: 6
type: A, layer: 1, pos: 24
type: B, layer: 1, pos: 6
type: A, layer: 1, pos: 28
type: B, layer: 1, pos: 24
type: B, layer: 1, pos: 0
type: B, layer: 1, pos: 37

Time for candidate selection: 0.22 seconds

### Candidate
type: A, layer: 1, pos: 35

### Candidate
type: A, layer: 1, pos: 0

### Candidate
type: B, layer: 1, pos: 7

### Candidate
type: A, layer: 1, pos: 7

### Candidate
type: A, layer: 1, pos: 36

## Relational analysis of NS_A1_B2_A1_B2_B1_B1_B2_A1
Optimization infeasible because this subproblem isn't reachable.

### Candidate
type: A, layer: 1, pos: 47

## Relational analysis of NS_A1_B2_A1_B2_B1_B1_B2_A1
Optimization infeasible because this subproblem isn't reachable.

### Candidate
type: A, layer: 1, pos: 6

## Relational analysis of NS_A1_B2_A1_B2_B1_B1_B2_A1
Optimization infeasible because this subproblem isn't reachable.

### Candidate
type: A, layer: 1, pos: 24

## Relational analysis of NS_A1_B2_A1_B2_B1_B1_B2_A1

### Relational analysis result of NS_A1_B2_A1_B2_B1_B1_B2_A1
Status: Status.UNKNOWN
Output dim: 3, lower bound: -20.4784536, upper bound: 20.6526970
time: 0.67 seconds

## Relational analysis of NS_A1_B2_A1_B2_B1_B1_B2_A2

### Relational analysis result of NS_A1_B2_A1_B2_B1_B1_B2_A2
Status: Status.UNKNOWN
Output dim: 3, lower bound: -20.4784563, upper bound: 20.6598939
time: 0.58 seconds

## BFS NS instance: NS_A1_B2_A1_B2_B1_B2_A1

### Backsubstitution after applying NS history:
0: -102.5664444, 143.4198303, -115.5967560, 159.3176727, -261.8840027, 259.0166016
1: -13.3056774, 12.3136339, -14.7356567, 13.7355738, -27.0412464, 27.0492897
2: -7.5930424, 12.5812550, -8.4914293, 14.0176039, -21.6106434, 21.0726852
3: -6.1116810, 13.8463879, -6.8866749, 15.3633566, -21.4750366, 20.7330627
4: -9.3920126, 11.2692289, -10.4997616, 12.5683928, -21.9604034, 21.7689819

Time for backsubstitution: 2.61 seconds

### NS candidates at layer 1
type: B, layer: 1, pos: 0
type: A, layer: 1, pos: 7
type: B, layer: 1, pos: 7
type: A, layer: 1, pos: 28
type: B, layer: 1, pos: 6
type: A, layer: 1, pos: 24
type: A, layer: 1, pos: 6
type: B, layer: 1, pos: 24
type: B, layer: 1, pos: 21
type: A, layer: 1, pos: 47
type: A, layer: 1, pos: 0
type: B, layer: 1, pos: 37
type: A, layer: 1, pos: 35

Time for candidate selection: 0.22 seconds

### Candidate
type: B, layer: 1, pos: 0

### Candidate
type: A, layer: 1, pos: 7

### Candidate
type: B, layer: 1, pos: 7

### Candidate
type: A, layer: 1, pos: 28

### Candidate
type: B, layer: 1, pos: 6

## Relational analysis of NS_A1_B2_A1_B2_B1_B2_A1_B1
Optimization infeasible because this subproblem isn't reachable.

### Candidate
type: A, layer: 1, pos: 24

## Relational analysis of NS_A1_B2_A1_B2_B1_B2_A1_A1
Optimization infeasible because this subproblem isn't reachable.

### Candidate
type: A, layer: 1, pos: 6

### Candidate
type: B, layer: 1, pos: 24

### Candidate
type: B, layer: 1, pos: 21

## Relational analysis of NS_A1_B2_A1_B2_B1_B2_A1_B1

### Relational analysis result of NS_A1_B2_A1_B2_B1_B2_A1_B1
Status: Status.UNKNOWN
Output dim: 3, lower bound: -20.5819115, upper bound: 20.6827185
time: 0.71 seconds

## Relational analysis of NS_A1_B2_A1_B2_B1_B2_A1_B2

### Relational analysis result of NS_A1_B2_A1_B2_B1_B2_A1_B2
Status: Status.UNKNOWN
Output dim: 3, lower bound: -20.5819076, upper bound: 20.6827133
time: 0.57 seconds

## BFS NS instance: NS_A1_B2_A1_B2_B1_B2_A2

### Backsubstitution after applying NS history:
0: -105.0573273, 146.1588287, -115.5967560, 159.3176727, -264.3750000, 261.7555237
1: -13.5524206, 12.5621405, -14.7356567, 13.7355738, -27.2879944, 27.2977982
2: -7.7492042, 12.8493176, -8.4914293, 14.0176039, -21.7668056, 21.3407478
3: -6.2506890, 14.1001730, -6.8866749, 15.3633566, -21.6140423, 20.9868431
4: -9.5892010, 11.5065594, -10.4997616, 12.5683928, -22.1575928, 22.0063133

Time for backsubstitution: 2.60 seconds

### NS candidates at layer 1
type: B, layer: 1, pos: 0
type: A, layer: 1, pos: 7
type: B, layer: 1, pos: 7
type: A, layer: 1, pos: 28
type: B, layer: 1, pos: 24
type: B, layer: 1, pos: 6
type: A, layer: 1, pos: 24
type: A, layer: 1, pos: 6
type: B, layer: 1, pos: 21
type: A, layer: 1, pos: 35
type: A, layer: 1, pos: 47
type: A, layer: 1, pos: 0
type: B, layer: 1, pos: 37

Time for candidate selection: 0.22 seconds

### Candidate
type: B, layer: 1, pos: 0

### Candidate
type: A, layer: 1, pos: 7

### Candidate
type: B, layer: 1, pos: 7

### Candidate
type: A, layer: 1, pos: 28

### Candidate
type: B, layer: 1, pos: 24

## Relational analysis of NS_A1_B2_A1_B2_B1_B2_A2_B1

### Relational analysis result of NS_A1_B2_A1_B2_B1_B2_A2_B1
Status: Status.UNKNOWN
Output dim: 3, lower bound: -20.5780607, upper bound: 20.6662122
time: 0.57 seconds

## Relational analysis of NS_A1_B2_A1_B2_B1_B2_A2_B2
Optimization infeasible because this subproblem isn't reachable.

### Candidate
type: B, layer: 1, pos: 6

## Relational analysis of NS_A1_B2_A1_B2_B1_B2_A2_B1
Optimization infeasible because this subproblem isn't reachable.

### Candidate
type: A, layer: 1, pos: 24

### Candidate
type: A, layer: 1, pos: 6

### Candidate
type: B, layer: 1, pos: 21

## Relational analysis of NS_A1_B2_A1_B2_B1_B2_A2_B1

### Relational analysis result of NS_A1_B2_A1_B2_B1_B2_A2_B1
Status: Status.UNKNOWN
Output dim: 3, lower bound: -20.5819097, upper bound: 20.6827179
time: 0.61 seconds

## Relational analysis of NS_A1_B2_A1_B2_B1_B2_A2_B2

### Relational analysis result of NS_A1_B2_A1_B2_B1_B2_A2_B2
Status: Status.UNKNOWN
Output dim: 3, lower bound: -20.5819047, upper bound: 20.6827188
time: 0.55 seconds

## BFS NS instance: NS_A1_B2_A1_B2_B2_A1_B1

### Backsubstitution after applying NS history:
0: -102.5664444, 143.4198303, -105.3472824, 145.9782257, -248.5446320, 248.7671204
1: -13.3056774, 12.3136339, -13.5449104, 12.5412617, -25.8469315, 25.8585434
2: -7.5930424, 12.5812550, -7.7514648, 12.8365688, -20.4296112, 20.3327141
3: -6.1116810, 13.8463879, -6.2571592, 14.0979071, -20.2095852, 20.1035461
4: -9.3920126, 11.2692289, -9.6030216, 11.4954319, -20.8874435, 20.8722477

Time for backsubstitution: 2.61 seconds

### NS candidates at layer 1
type: B, layer: 1, pos: 0
type: A, layer: 1, pos: 7
type: B, layer: 1, pos: 7
type: B, layer: 1, pos: 47
type: A, layer: 1, pos: 28
type: A, layer: 1, pos: 6
type: B, layer: 1, pos: 6
type: A, layer: 1, pos: 24
type: B, layer: 1, pos: 24
type: A, layer: 1, pos: 35
type: A, layer: 1, pos: 0
type: A, layer: 1, pos: 47
type: B, layer: 1, pos: 37

Time for candidate selection: 0.22 seconds

### Candidate
type: B, layer: 1, pos: 0

### Candidate
type: A, layer: 1, pos: 7

### Candidate
type: B, layer: 1, pos: 7

### Candidate
type: B, layer: 1, pos: 47

### Candidate
type: A, layer: 1, pos: 28

### Candidate
type: A, layer: 1, pos: 6

### Candidate
type: B, layer: 1, pos: 6

### Candidate
type: A, layer: 1, pos: 24

## Relational analysis of NS_A1_B2_A1_B2_B2_A1_B1_A1

### Relational analysis result of NS_A1_B2_A1_B2_B2_A1_B1_A1
Status: Status.UNKNOWN
Output dim: 3, lower bound: -20.5527583, upper bound: 20.6956143
time: 0.65 seconds

## Relational analysis of NS_A1_B2_A1_B2_B2_A1_B1_A2

### Relational analysis result of NS_A1_B2_A1_B2_B2_A1_B1_A2
Status: Status.UNKNOWN
Output dim: 3, lower bound: -20.5527590, upper bound: 20.6664671
time: 0.58 seconds

## BFS NS instance: NS_A1_B2_A1_B2_B2_A1_B2

### Backsubstitution after applying NS history:
0: -102.5664444, 143.4198303, -125.4651947, 175.4957886, -278.0621948, 268.8850098
1: -13.3056774, 12.3136339, -16.2982426, 14.9983511, -28.3040276, 28.6118774
2: -7.5930424, 12.5812550, -9.2701159, 15.4215355, -23.0145721, 21.8513680
3: -6.1116810, 13.8463879, -7.4745045, 16.9310951, -23.0427742, 21.3208923
4: -9.3920126, 11.2692289, -11.4773617, 13.8075666, -23.1995773, 22.7465897

Time for backsubstitution: 2.61 seconds

### NS candidates at layer 1
type: B, layer: 1, pos: 0
type: B, layer: 1, pos: 7
type: A, layer: 1, pos: 7
type: B, layer: 1, pos: 47
type: B, layer: 1, pos: 6
type: A, layer: 1, pos: 28
type: A, layer: 1, pos: 6
type: A, layer: 1, pos: 24
type: B, layer: 1, pos: 24
type: A, layer: 1, pos: 47
type: B, layer: 1, pos: 37
type: A, layer: 1, pos: 0
type: A, layer: 1, pos: 35

Time for candidate selection: 0.22 seconds

### Candidate
type: B, layer: 1, pos: 0

### Candidate
type: B, layer: 1, pos: 7

### Candidate
type: A, layer: 1, pos: 7

### Candidate
type: B, layer: 1, pos: 47

### Candidate
type: B, layer: 1, pos: 6

### Candidate
type: A, layer: 1, pos: 28

### Candidate
type: A, layer: 1, pos: 6

### Candidate
type: A, layer: 1, pos: 24

## Relational analysis of NS_A1_B2_A1_B2_B2_A1_B2_A1

### Relational analysis result of NS_A1_B2_A1_B2_B2_A1_B2_A1
Status: Status.UNKNOWN
Output dim: 3, lower bound: -20.5527595, upper bound: 20.6956144
time: 0.58 seconds

## Relational analysis of NS_A1_B2_A1_B2_B2_A1_B2_A2

### Relational analysis result of NS_A1_B2_A1_B2_B2_A1_B2_A2
Status: Status.UNKNOWN
Output dim: 3, lower bound: -20.5527590, upper bound: 20.6664690
time: 0.57 seconds

## BFS NS instance: NS_A1_B2_A1_B2_B2_A2_B1

### Backsubstitution after applying NS history:
0: -105.0573273, 146.1588287, -103.0337677, 141.5471497, -246.6044769, 249.1925964
1: -13.5524206, 12.5621405, -13.0950356, 12.2496815, -25.8021011, 25.6571770
2: -7.7492042, 12.8493176, -7.5708485, 12.4477148, -20.1969147, 20.4201660
3: -6.2506890, 14.1001730, -6.1326370, 13.6562023, -19.9068909, 20.2328072
4: -9.5892010, 11.5065594, -9.3719139, 11.1653881, -20.7545891, 20.8784733

Time for backsubstitution: 2.62 seconds

### NS candidates at layer 1
type: A, layer: 1, pos: 35
type: A, layer: 1, pos: 0
type: A, layer: 1, pos: 7
type: B, layer: 1, pos: 7
type: B, layer: 1, pos: 21
type: A, layer: 1, pos: 6
type: A, layer: 1, pos: 28
type: B, layer: 1, pos: 6
type: A, layer: 1, pos: 47
type: A, layer: 1, pos: 24
type: B, layer: 1, pos: 47
type: B, layer: 1, pos: 37
type: B, layer: 1, pos: 0

Time for candidate selection: 0.22 seconds

### Candidate
type: A, layer: 1, pos: 35

### Candidate
type: A, layer: 1, pos: 0

### Candidate
type: A, layer: 1, pos: 7

### Candidate
type: B, layer: 1, pos: 7

### Candidate
type: B, layer: 1, pos: 21

## Relational analysis of NS_A1_B2_A1_B2_B2_A2_B1_B1

### Relational analysis result of NS_A1_B2_A1_B2_B2_A2_B1_B1
Status: Status.UNKNOWN
Output dim: 3, lower bound: -20.5735368, upper bound: 20.6662719
time: 0.59 seconds

## Relational analysis of NS_A1_B2_A1_B2_B2_A2_B1_B2

### Relational analysis result of NS_A1_B2_A1_B2_B2_A2_B1_B2
Status: Status.UNKNOWN
Output dim: 3, lower bound: -20.5735387, upper bound: 20.6662735
time: 0.61 seconds

## BFS NS instance: NS_A1_B2_A1_B2_B2_A2_B2

### Backsubstitution after applying NS history:
0: -105.0573273, 146.1588287, -105.3449936, 144.8378601, -249.8951874, 251.5038147
1: -13.5524206, 12.5621405, -13.4001780, 12.5108299, -26.0632515, 25.9623184
2: -7.7492042, 12.8493176, -7.7231021, 12.7516079, -20.5008106, 20.5724201
3: -6.2506890, 14.1001730, -6.2564716, 13.9476671, -20.1983566, 20.3566399
4: -9.5892010, 11.5065594, -9.5623960, 11.4285898, -21.0177917, 21.0689487

Time for backsubstitution: 2.62 seconds

### NS candidates at layer 1
type: A, layer: 1, pos: 35
type: A, layer: 1, pos: 0
type: A, layer: 1, pos: 7
type: B, layer: 1, pos: 7
type: B, layer: 1, pos: 21
type: A, layer: 1, pos: 28
type: A, layer: 1, pos: 6
type: B, layer: 1, pos: 6
type: A, layer: 1, pos: 24
type: A, layer: 1, pos: 47
type: B, layer: 1, pos: 47
type: B, layer: 1, pos: 0
type: B, layer: 1, pos: 37

Time for candidate selection: 0.22 seconds

### Candidate
type: A, layer: 1, pos: 35

### Candidate
type: A, layer: 1, pos: 0

### Candidate
type: A, layer: 1, pos: 7

### Candidate
type: B, layer: 1, pos: 7

### Candidate
type: B, layer: 1, pos: 21

## Relational analysis of NS_A1_B2_A1_B2_B2_A2_B2_B1

### Relational analysis result of NS_A1_B2_A1_B2_B2_A2_B2_B1
Status: Status.UNKNOWN
Output dim: 3, lower bound: -20.5527544, upper bound: 20.6662726
time: 0.56 seconds

## Relational analysis of NS_A1_B2_A1_B2_B2_A2_B2_B2

### Relational analysis result of NS_A1_B2_A1_B2_B2_A2_B2_B2
Status: Status.UNKNOWN
Output dim: 3, lower bound: -20.5527567, upper bound: 20.6662688
time: 0.64 seconds

## BFS NS instance: NS_A1_B2_A2_B1_B1_B1_A1

### Backsubstitution after applying NS history:
0: -122.3504181, 172.4206543, -105.7482376, 147.0162506, -269.3666687, 278.1688232
1: -16.0028038, 14.7454338, -13.6290569, 12.6251478, -28.6279488, 28.3744907
2: -9.0890217, 15.1113281, -7.8028665, 12.9075041, -21.9965248, 22.9141922
3: -7.3192620, 16.6259670, -6.3008537, 14.1975565, -21.5168114, 22.9268169
4: -11.2329483, 13.5337877, -9.6589079, 11.5644083, -22.7973557, 23.1926956

Time for backsubstitution: 2.64 seconds

### NS candidates at layer 1
type: A, layer: 1, pos: 35
type: A, layer: 1, pos: 0
type: A, layer: 1, pos: 7
type: B, layer: 1, pos: 7
type: A, layer: 1, pos: 6
type: A, layer: 1, pos: 28
type: B, layer: 1, pos: 24
type: A, layer: 1, pos: 24
type: B, layer: 1, pos: 6
type: B, layer: 1, pos: 47
type: A, layer: 1, pos: 47
type: B, layer: 1, pos: 0
type: B, layer: 1, pos: 37

Time for candidate selection: 0.22 seconds

### Candidate
type: A, layer: 1, pos: 35

### Candidate
type: A, layer: 1, pos: 0

### Candidate
type: A, layer: 1, pos: 7

### Candidate
type: B, layer: 1, pos: 7

### Candidate
type: A, layer: 1, pos: 6

### Candidate
type: A, layer: 1, pos: 28

## Relational analysis of NS_A1_B2_A2_B1_B1_B1_A1_A1
Optimization infeasible because this subproblem isn't reachable.

### Candidate
type: B, layer: 1, pos: 24

## Relational analysis of NS_A1_B2_A2_B1_B1_B1_A1_B1

### Relational analysis result of NS_A1_B2_A2_B1_B1_B1_A1_B1
Status: Status.UNKNOWN
Output dim: 3, lower bound: -20.7031067, upper bound: 20.5484796
time: 0.66 seconds

## Relational analysis of NS_A1_B2_A2_B1_B1_B1_A1_B2
Optimization infeasible because this subproblem isn't reachable.

### Candidate
type: A, layer: 1, pos: 24

### Candidate
type: B, layer: 1, pos: 6

### Candidate
type: B, layer: 1, pos: 47

## Relational analysis of NS_A1_B2_A2_B1_B1_B1_A1_B1
Optimization infeasible because this subproblem isn't reachable.

### Candidate
type: A, layer: 1, pos: 47

### Candidate
type: B, layer: 1, pos: 0

### Candidate
type: B, layer: 1, pos: 37

## Relational analysis of NS_A1_B2_A2_B1_B1_B1_A1_B1
Optimization infeasible because this subproblem isn't reachable.

## BFS NS instance: NS_A1_B2_A2_B1_B1_B1_A2

### Backsubstitution after applying NS history:
0: -125.4466095, 175.9087677, -105.7482376, 147.0162506, -272.4628296, 281.6570129
1: -16.3155556, 15.0648985, -13.6290569, 12.6251478, -28.9407005, 28.6939507
2: -9.2879438, 15.4446859, -7.8028665, 12.9075041, -22.1954441, 23.2475529
3: -7.4973702, 16.9498234, -6.3008537, 14.1975565, -21.6949158, 23.2506771
4: -11.4820595, 13.8310213, -9.6589079, 11.5644083, -23.0464668, 23.4899254

Time for backsubstitution: 2.64 seconds

### NS candidates at layer 1
type: A, layer: 1, pos: 35
type: A, layer: 1, pos: 0
type: A, layer: 1, pos: 7
type: B, layer: 1, pos: 7
type: A, layer: 1, pos: 6
type: A, layer: 1, pos: 28
type: B, layer: 1, pos: 24
type: A, layer: 1, pos: 24
type: B, layer: 1, pos: 6
type: A, layer: 1, pos: 47
type: B, layer: 1, pos: 47
type: B, layer: 1, pos: 0
type: B, layer: 1, pos: 37

Time for candidate selection: 0.22 seconds

### Candidate
type: A, layer: 1, pos: 35

### Candidate
type: A, layer: 1, pos: 0

### Candidate
type: A, layer: 1, pos: 7

### Candidate
type: B, layer: 1, pos: 7

### Candidate
type: A, layer: 1, pos: 6

### Candidate
type: A, layer: 1, pos: 28

## Relational analysis of NS_A1_B2_A2_B1_B1_B1_A2_A1
Optimization infeasible because this subproblem isn't reachable.

### Candidate
type: B, layer: 1, pos: 24

## Relational analysis of NS_A1_B2_A2_B1_B1_B1_A2_B1

### Relational analysis result of NS_A1_B2_A2_B1_B1_B1_A2_B1
Status: Status.UNKNOWN
Output dim: 3, lower bound: -20.7031138, upper bound: 20.5484797
time: 0.57 seconds

## Relational analysis of NS_A1_B2_A2_B1_B1_B1_A2_B2
Optimization infeasible because this subproblem isn't reachable.

### Candidate
type: A, layer: 1, pos: 24

### Candidate
type: B, layer: 1, pos: 6

### Candidate
type: A, layer: 1, pos: 47

## Relational analysis of NS_A1_B2_A2_B1_B1_B1_A2_A1
Optimization infeasible because this subproblem isn't reachable.

### Candidate
type: B, layer: 1, pos: 47

### Candidate
type: B, layer: 1, pos: 0

### Candidate
type: B, layer: 1, pos: 37

## Relational analysis of NS_A1_B2_A2_B1_B1_B1_A2_B1

### Relational analysis result of NS_A1_B2_A2_B1_B1_B1_A2_B1
Status: Status.UNKNOWN
Output dim: 3, lower bound: -20.6313885, upper bound: 20.5688050
time: 0.66 seconds

## Relational analysis of NS_A1_B2_A2_B1_B1_B1_A2_B2
Optimization infeasible because this subproblem isn't reachable.

## BFS NS instance: NS_A1_B2_A2_B1_B1_B2_A1

### Backsubstitution after applying NS history:
0: -122.3504181, 172.4206543, -104.5640869, 145.6174622, -267.9678955, 276.9846191
1: -16.0028038, 14.7454338, -13.5162983, 12.4906998, -28.4934998, 28.2617321
2: -9.0890217, 15.1113281, -7.7180171, 12.7851391, -21.8741608, 22.8293438
3: -7.3192620, 16.6259670, -6.2191501, 14.0629053, -21.3821640, 22.8451157
4: -11.2329483, 13.5337877, -9.5563126, 11.4491491, -22.6820984, 23.0900993

Time for backsubstitution: 2.64 seconds

### NS candidates at layer 1
type: A, layer: 1, pos: 35
type: A, layer: 1, pos: 0
type: A, layer: 1, pos: 7
type: B, layer: 1, pos: 7
type: A, layer: 1, pos: 6
type: A, layer: 1, pos: 28
type: B, layer: 1, pos: 24
type: A, layer: 1, pos: 24
type: B, layer: 1, pos: 6
type: A, layer: 1, pos: 47
type: B, layer: 1, pos: 47
type: B, layer: 1, pos: 0
type: B, layer: 1, pos: 37

Time for candidate selection: 0.22 seconds

### Candidate
type: A, layer: 1, pos: 35

### Candidate
type: A, layer: 1, pos: 0

### Candidate
type: A, layer: 1, pos: 7

### Candidate
type: B, layer: 1, pos: 7

### Candidate
type: A, layer: 1, pos: 6

### Candidate
type: A, layer: 1, pos: 28

## Relational analysis of NS_A1_B2_A2_B1_B1_B2_A1_A1
Optimization infeasible because this subproblem isn't reachable.

### Candidate
type: B, layer: 1, pos: 24

## Relational analysis of NS_A1_B2_A2_B1_B1_B2_A1_B1

### Relational analysis result of NS_A1_B2_A2_B1_B1_B2_A1_B1
Status: Status.UNKNOWN
Output dim: 3, lower bound: -20.7022592, upper bound: 20.5484789
time: 0.64 seconds

## Relational analysis of NS_A1_B2_A2_B1_B1_B2_A1_B2

### Relational analysis result of NS_A1_B2_A2_B1_B1_B2_A1_B2
Status: Status.UNKNOWN
Output dim: 3, lower bound: -20.6665011, upper bound: 20.5484783
time: 0.70 seconds

## BFS NS instance: NS_A1_B2_A2_B1_B1_B2_A2

### Backsubstitution after applying NS history:
0: -125.4466095, 175.9087677, -104.5640869, 145.6174622, -271.0640869, 280.4728088
1: -16.3155556, 15.0648985, -13.5162983, 12.4906998, -28.8062515, 28.5811958
2: -9.2879438, 15.4446859, -7.7180171, 12.7851391, -22.0730820, 23.1627026
3: -7.4973702, 16.9498234, -6.2191501, 14.0629053, -21.5602703, 23.1689739
4: -11.4820595, 13.8310213, -9.5563126, 11.4491491, -22.9312096, 23.3873291

Time for backsubstitution: 2.65 seconds

### NS candidates at layer 1
type: A, layer: 1, pos: 35
type: A, layer: 1, pos: 0
type: A, layer: 1, pos: 7
type: B, layer: 1, pos: 7
type: A, layer: 1, pos: 6
type: A, layer: 1, pos: 28
type: B, layer: 1, pos: 24
type: A, layer: 1, pos: 24
type: B, layer: 1, pos: 6
type: A, layer: 1, pos: 47
type: B, layer: 1, pos: 47
type: B, layer: 1, pos: 37
type: B, layer: 1, pos: 0

Time for candidate selection: 0.22 seconds

### Candidate
type: A, layer: 1, pos: 35

### Candidate
type: A, layer: 1, pos: 0

### Candidate
type: A, layer: 1, pos: 7

### Candidate
type: B, layer: 1, pos: 7

### Candidate
type: A, layer: 1, pos: 6

### Candidate
type: A, layer: 1, pos: 28

## Relational analysis of NS_A1_B2_A2_B1_B1_B2_A2_A1
Optimization infeasible because this subproblem isn't reachable.

### Candidate
type: B, layer: 1, pos: 24

## Relational analysis of NS_A1_B2_A2_B1_B1_B2_A2_B1

### Relational analysis result of NS_A1_B2_A2_B1_B1_B2_A2_B1
Status: Status.UNKNOWN
Output dim: 3, lower bound: -20.7022596, upper bound: 20.5484794
time: 0.62 seconds

## Relational analysis of NS_A1_B2_A2_B1_B1_B2_A2_B2

### Relational analysis result of NS_A1_B2_A2_B1_B1_B2_A2_B2
Status: Status.UNKNOWN
Output dim: 3, lower bound: -20.6665011, upper bound: 20.5484782
time: 0.62 seconds

## BFS NS instance: NS_A2_B1_A2_A2_B1_A1_A1

### Backsubstitution after applying NS history:
0: -106.8512573, 148.9349670, -60.7692795, 78.7800903, -185.6313477, 209.7042236
1: -13.7961617, 12.7832594, -7.1476021, 7.2403092, -21.0364685, 19.9308605
2: -7.8885875, 13.0623493, -4.4023685, 6.8436527, -14.7322388, 17.4647179
3: -6.3611565, 14.3635292, -3.6247139, 7.3066416, -13.6677980, 17.9882412
4: -9.7572527, 11.7084198, -5.5420227, 6.2030392, -15.9602919, 17.2504406

Time for backsubstitution: 2.65 seconds

### NS candidates at layer 1
type: A, layer: 1, pos: 0
type: B, layer: 1, pos: 21
type: A, layer: 1, pos: 7
type: A, layer: 1, pos: 24
type: A, layer: 1, pos: 6
type: B, layer: 1, pos: 24
type: B, layer: 1, pos: 7
type: B, layer: 1, pos: 36
type: B, layer: 1, pos: 28
type: B, layer: 1, pos: 6
type: A, layer: 1, pos: 21
type: B, layer: 1, pos: 47
type: B, layer: 1, pos: 0

Time for candidate selection: 0.22 seconds

### Candidate
type: A, layer: 1, pos: 0

### Candidate
type: B, layer: 1, pos: 21

### Candidate
type: A, layer: 1, pos: 7

### Candidate
type: A, layer: 1, pos: 24

## Relational analysis of NS_A2_B1_A2_A2_B1_A1_A1_A1

### Relational analysis result of NS_A2_B1_A2_A2_B1_A1_A1_A1
Status: Status.UNKNOWN
Output dim: 3, lower bound: -20.5017471, upper bound: 20.7037593
time: 0.72 seconds

## Relational analysis of NS_A2_B1_A2_A2_B1_A1_A1_A2
Optimization infeasible because this subproblem isn't reachable.

### Candidate
type: A, layer: 1, pos: 6

## Relational analysis of NS_A2_B1_A2_A2_B1_A1_A1_A1
Optimization infeasible because this subproblem isn't reachable.

### Candidate
type: B, layer: 1, pos: 24

### Candidate
type: B, layer: 1, pos: 7

### Candidate
type: B, layer: 1, pos: 36

## Relational analysis of NS_A2_B1_A2_A2_B1_A1_A1_B1

### Relational analysis result of NS_A2_B1_A2_A2_B1_A1_A1_B1
Status: Status.UNKNOWN
Output dim: 3, lower bound: -20.5227832, upper bound: 20.7042800
time: 0.63 seconds

## Relational analysis of NS_A2_B1_A2_A2_B1_A1_A1_B2

### Relational analysis result of NS_A2_B1_A2_A2_B1_A1_A1_B2
Status: Status.UNKNOWN
Output dim: 3, lower bound: -20.5227842, upper bound: 20.7042781
time: 0.63 seconds

## BFS NS instance: NS_A2_B1_A2_A2_B1_A1_A2

### Backsubstitution after applying NS history:
0: -105.7592010, 147.6298065, -60.7692795, 78.7800903, -184.5392914, 208.3990479
1: -13.6911354, 12.6574459, -7.1476021, 7.2403092, -20.9314442, 19.8050480
2: -7.8084731, 12.9498215, -4.4023685, 6.8436527, -14.6521254, 17.3521881
3: -6.2859244, 14.2387390, -3.6247139, 7.3066416, -13.5925655, 17.8634529
4: -9.6611700, 11.6032963, -5.5420227, 6.2030392, -15.8642092, 17.1453190

Time for backsubstitution: 2.65 seconds

### NS candidates at layer 1
type: A, layer: 1, pos: 0
type: B, layer: 1, pos: 21
type: A, layer: 1, pos: 7
type: A, layer: 1, pos: 6
type: A, layer: 1, pos: 24
type: B, layer: 1, pos: 24
type: B, layer: 1, pos: 7
type: B, layer: 1, pos: 36
type: B, layer: 1, pos: 28
type: B, layer: 1, pos: 6
type: A, layer: 1, pos: 21
type: B, layer: 1, pos: 47
type: B, layer: 1, pos: 0

Time for candidate selection: 0.22 seconds

### Candidate
type: A, layer: 1, pos: 0

### Candidate
type: B, layer: 1, pos: 21

### Candidate
type: A, layer: 1, pos: 7

### Candidate
type: A, layer: 1, pos: 6

## Relational analysis of NS_A2_B1_A2_A2_B1_A1_A2_A1
Optimization infeasible because this subproblem isn't reachable.

### Candidate
type: A, layer: 1, pos: 24

## Relational analysis of NS_A2_B1_A2_A2_B1_A1_A2_A1

### Relational analysis result of NS_A2_B1_A2_A2_B1_A1_A2_A1
Status: Status.UNKNOWN
Output dim: 3, lower bound: -20.5017456, upper bound: 20.7028973
time: 0.63 seconds

## Relational analysis of NS_A2_B1_A2_A2_B1_A1_A2_A2

### Relational analysis result of NS_A2_B1_A2_A2_B1_A1_A2_A2
Status: Status.UNKNOWN
Output dim: 3, lower bound: -20.5017444, upper bound: 20.6669936
time: 0.66 seconds

## BFS NS instance: NS_A2_B1_A2_A2_B1_A2_A1

### Backsubstitution after applying NS history:
0: -100.2234039, 138.7128448, -60.7692795, 78.7800903, -179.0034790, 199.4821014
1: -12.8489332, 11.9514885, -7.1476021, 7.2403092, -20.0892429, 19.0990906
2: -7.3790054, 12.1810884, -4.4023685, 6.8436527, -14.2226582, 16.5834579
3: -5.9505777, 13.3626137, -3.6247139, 7.3066416, -13.2572193, 16.9873276
4: -9.1261005, 10.9195204, -5.5420227, 6.2030392, -15.3291397, 16.4615440

Time for backsubstitution: 2.65 seconds

### NS candidates at layer 1
type: A, layer: 1, pos: 0
type: B, layer: 1, pos: 21
type: A, layer: 1, pos: 6
type: A, layer: 1, pos: 7
type: A, layer: 1, pos: 28
type: B, layer: 1, pos: 24
type: B, layer: 1, pos: 36
type: B, layer: 1, pos: 28
type: B, layer: 1, pos: 7
type: B, layer: 1, pos: 6
type: A, layer: 1, pos: 21
type: B, layer: 1, pos: 47
type: B, layer: 1, pos: 0

Time for candidate selection: 0.22 seconds

### Candidate
type: A, layer: 1, pos: 0

### Candidate
type: B, layer: 1, pos: 21

### Candidate
type: A, layer: 1, pos: 6

### Candidate
type: A, layer: 1, pos: 7

### Candidate
type: A, layer: 1, pos: 28

## Relational analysis of NS_A2_B1_A2_A2_B1_A2_A1_A1

### Relational analysis result of NS_A2_B1_A2_A2_B1_A2_A1_A1
Status: Status.UNKNOWN
Output dim: 3, lower bound: -20.5017443, upper bound: 20.6852296
time: 0.63 seconds

## Relational analysis of NS_A2_B1_A2_A2_B1_A2_A1_A2

### Relational analysis result of NS_A2_B1_A2_A2_B1_A2_A1_A2
Status: Status.UNKNOWN
Output dim: 3, lower bound: -20.5017418, upper bound: 20.6850343
time: 0.67 seconds

## BFS NS instance: NS_A2_B1_A2_A2_B1_A2_A2

### Backsubstitution after applying NS history:
0: -102.0461884, 141.2780304, -60.7692795, 78.7800903, -180.8262634, 202.0472870
1: -13.0849609, 12.1524773, -7.1476021, 7.2403092, -20.3252697, 19.3000774
2: -7.5124521, 12.4186974, -4.4023685, 6.8436527, -14.3561049, 16.8210640
3: -6.0464311, 13.5862017, -3.6247139, 7.3066416, -13.3530731, 17.2109146
4: -9.2703857, 11.1262703, -5.5420227, 6.2030392, -15.4734240, 16.6682930

Time for backsubstitution: 2.66 seconds

### NS candidates at layer 1
type: A, layer: 1, pos: 0
type: B, layer: 1, pos: 21
type: A, layer: 1, pos: 6
type: A, layer: 1, pos: 7
type: A, layer: 1, pos: 28
type: B, layer: 1, pos: 36
type: B, layer: 1, pos: 24
type: B, layer: 1, pos: 28
type: B, layer: 1, pos: 7
type: B, layer: 1, pos: 6
type: A, layer: 1, pos: 21
type: B, layer: 1, pos: 47
type: B, layer: 1, pos: 0

Time for candidate selection: 0.22 seconds

### Candidate
type: A, layer: 1, pos: 0

### Candidate
type: B, layer: 1, pos: 21

### Candidate
type: A, layer: 1, pos: 6

### Candidate
type: A, layer: 1, pos: 7

### Candidate
type: A, layer: 1, pos: 28

## Relational analysis of NS_A2_B1_A2_A2_B1_A2_A2_A1

### Relational analysis result of NS_A2_B1_A2_A2_B1_A2_A2_A1
Status: Status.UNKNOWN
Output dim: 3, lower bound: -20.5017454, upper bound: 20.6671352
time: 0.65 seconds

## Relational analysis of NS_A2_B1_A2_A2_B1_A2_A2_A2

### Relational analysis result of NS_A2_B1_A2_A2_B1_A2_A2_A2
Status: Status.UNKNOWN
Output dim: 3, lower bound: -20.5017441, upper bound: 20.6680266
time: 0.60 seconds

## NS Result
status: Status.UNKNOWN
execution time: (base) + (ns) = 4.23 + 415.90 = 420.13 seconds
