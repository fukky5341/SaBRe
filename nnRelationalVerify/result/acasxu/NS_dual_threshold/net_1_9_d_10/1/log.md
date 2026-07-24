## Execution arguments:
Dataset: Dataset.ACAS
Network: ds/onnx/acasxu_op11/ACASXU_1_9.onnx
Relational property: GLOBAL_ROBUSTNESS
LP Analysis: True
Epsilon: None
Delta epsilon: 0.1
execution index: (10, None, 1)
Time budget: 420 seconds
Split limit: 100
Threshold: 63.39241200129601


## IAR start

### BASE IAR bounds
Layer (inp1_lb, inp1_ub, inp2_lb, inp2_ub, d_lb, d_ub)
0: (-38.4153824, 40.2786446, -38.4153824, 40.2786446, -78.6940308, 78.6940308)
1: (-271.9440918, 95.2166214, -271.9440918, 95.2166214, -367.1607056, 367.1607056)
2: (-152.0317841, 87.0516129, -152.0317841, 87.0516129, -239.0833893, 239.0833893)
3: (-189.9531097, 69.9066696, -189.9531097, 69.9066696, -259.8597412, 259.8597412)
4: (-109.4088516, 76.1420975, -109.4088516, 76.1420975, -185.5509186, 185.5509338)

## BASE Relational Analysis

## BASE Result
execution time: IAR + RelationalAnalysis = 0.99 + 1.99 = 2.98 seconds
status: Status.UNKNOWN
relational distance
Output dim: 0, lower bound: -63.3974838, upper bound: 63.3974838

# Neuron Split (NS) starts

## BFS NS instance: NS

Time for backsubstitution: 0.01 seconds

### NS candidates at layer 1
type: B, layer: 1, pos: 40
type: A, layer: 1, pos: 40
type: A, layer: 1, pos: 7
type: B, layer: 1, pos: 7
type: B, layer: 1, pos: 4
type: A, layer: 1, pos: 4
type: A, layer: 1, pos: 9
type: B, layer: 1, pos: 9
type: B, layer: 1, pos: 35
type: A, layer: 1, pos: 35
type: A, layer: 1, pos: 17
type: B, layer: 1, pos: 17
type: B, layer: 1, pos: 19
type: A, layer: 1, pos: 19
type: B, layer: 1, pos: 45
type: A, layer: 1, pos: 45
type: B, layer: 1, pos: 3
type: A, layer: 1, pos: 3
type: B, layer: 1, pos: 11
type: A, layer: 1, pos: 11
type: A, layer: 1, pos: 10
type: B, layer: 1, pos: 10
type: A, layer: 1, pos: 15
type: B, layer: 1, pos: 15
type: B, layer: 1, pos: 18
type: A, layer: 1, pos: 18
type: B, layer: 1, pos: 39
type: A, layer: 1, pos: 39
type: A, layer: 1, pos: 32
type: B, layer: 1, pos: 32
type: B, layer: 1, pos: 43
type: A, layer: 1, pos: 43
type: A, layer: 1, pos: 38
type: B, layer: 1, pos: 38
type: B, layer: 1, pos: 37
type: A, layer: 1, pos: 37
type: A, layer: 1, pos: 29
type: B, layer: 1, pos: 29
type: A, layer: 1, pos: 41
type: B, layer: 1, pos: 41
type: A, layer: 1, pos: 20
type: B, layer: 1, pos: 20
type: A, layer: 1, pos: 21
type: B, layer: 1, pos: 21
type: A, layer: 1, pos: 5
type: B, layer: 1, pos: 5
type: B, layer: 1, pos: 23
type: A, layer: 1, pos: 23
type: A, layer: 1, pos: 16
type: B, layer: 1, pos: 16

Time for candidate selection: 0.07 seconds

### Candidate
type: B, layer: 1, pos: 40

## Relational analysis of NS_B1

### Relational analysis result of NS_B1
Status: Status.UNKNOWN
Output dim: 0, lower bound: -63.3963488, upper bound: 63.3961981
time: 0.69 seconds

## Relational analysis of NS_B2

### Relational analysis result of NS_B2
Status: Status.UNKNOWN
Output dim: 0, lower bound: -63.3958632, upper bound: 63.3958632
time: 0.71 seconds

## Summary of splitting at layer (split count: 0)
- Time for NS candidates: 1.49 seconds
NS_B1, status: Status.UNKNOWN, split count: 1, time: 1.49
Output dim: 0, lower bound: -63.3963488, upper bound: 63.3961981
NS_B2, status: Status.UNKNOWN, split count: 1, time: 1.49
Output dim: 0, lower bound: -63.3958632, upper bound: 63.3958632

## BFS NS instance: NS_B1

### Backsubstitution after applying NS history:
0: -36.6631546, 38.4677963, -35.6584015, 37.4214935, -74.0846481, 74.1261902
1: -260.1878967, 90.8589401, -253.4253845, 88.3672638, -348.5551758, 344.2843018
2: -145.2130890, 83.1253891, -141.3148041, 80.8721390, -226.0852356, 224.4401855
3: -181.7225800, 66.7120132, -176.9852448, 64.8847427, -246.6073303, 243.6972656
4: -104.4755707, 72.6576462, -101.6394196, 70.6504669, -175.1260223, 174.2970581

Time for backsubstitution: 0.83 seconds

### NS candidates at layer 1
type: B, layer: 1, pos: 7
type: A, layer: 1, pos: 7
type: A, layer: 1, pos: 40
type: B, layer: 1, pos: 4
type: A, layer: 1, pos: 4
type: B, layer: 1, pos: 9
type: A, layer: 1, pos: 9
type: A, layer: 1, pos: 35
type: B, layer: 1, pos: 35
type: A, layer: 1, pos: 17
type: A, layer: 1, pos: 19
type: B, layer: 1, pos: 17
type: B, layer: 1, pos: 45
type: B, layer: 1, pos: 3
type: B, layer: 1, pos: 19
type: A, layer: 1, pos: 45
type: A, layer: 1, pos: 3
type: B, layer: 1, pos: 11
type: A, layer: 1, pos: 11
type: B, layer: 1, pos: 10
type: A, layer: 1, pos: 10
type: A, layer: 1, pos: 15
type: B, layer: 1, pos: 15
type: A, layer: 1, pos: 18
type: B, layer: 1, pos: 18
type: A, layer: 1, pos: 39
type: B, layer: 1, pos: 32
type: A, layer: 1, pos: 32
type: A, layer: 1, pos: 43
type: B, layer: 1, pos: 39
type: B, layer: 1, pos: 38
type: B, layer: 1, pos: 43
type: B, layer: 1, pos: 37
type: B, layer: 1, pos: 29
type: A, layer: 1, pos: 38
type: A, layer: 1, pos: 37
type: B, layer: 1, pos: 41
type: A, layer: 1, pos: 41
type: B, layer: 1, pos: 20
type: A, layer: 1, pos: 29
type: A, layer: 1, pos: 20
type: A, layer: 1, pos: 23
type: A, layer: 1, pos: 5
type: B, layer: 1, pos: 21
type: A, layer: 1, pos: 21
type: B, layer: 1, pos: 5
type: B, layer: 1, pos: 16
type: B, layer: 1, pos: 23
type: A, layer: 1, pos: 16

Time for candidate selection: 0.07 seconds

### Candidate
type: B, layer: 1, pos: 7

## Relational analysis of NS_B1_B1

### Relational analysis result of NS_B1_B1
Status: Status.UNKNOWN
Output dim: 0, lower bound: -63.3946059, upper bound: 63.3945244
time: 0.70 seconds

## Relational analysis of NS_B1_B2

### Relational analysis result of NS_B1_B2
Status: Status.UNKNOWN
Output dim: 0, lower bound: -63.3954223, upper bound: 63.3952755
time: 0.95 seconds

## BFS NS instance: NS_B2

### Backsubstitution after applying NS history:
0: -37.2165909, 39.2005997, -56.4413948, 61.1914101, -98.4080048, 95.6419983
1: -265.3079529, 92.1755524, -416.5941772, 139.3072815, -404.6152344, 508.7697144
2: -147.3638763, 84.6269760, -222.9568634, 131.2333984, -278.5972290, 307.5838318
3: -185.0496063, 67.9713440, -287.4306335, 105.1853409, -290.2349548, 355.4019775
4: -105.9015121, 74.0802994, -159.8457642, 115.9607849, -221.8623047, 233.9260406

Time for backsubstitution: 0.87 seconds

### NS candidates at layer 1
type: A, layer: 1, pos: 7
type: A, layer: 1, pos: 4
type: A, layer: 1, pos: 40
type: B, layer: 1, pos: 7
type: A, layer: 1, pos: 9
type: A, layer: 1, pos: 17
type: B, layer: 1, pos: 35
type: A, layer: 1, pos: 19
type: A, layer: 1, pos: 3
type: B, layer: 1, pos: 9
type: A, layer: 1, pos: 38
type: A, layer: 1, pos: 29
type: A, layer: 1, pos: 35
type: B, layer: 1, pos: 11
type: A, layer: 1, pos: 45
type: B, layer: 1, pos: 19
type: A, layer: 1, pos: 11
type: A, layer: 1, pos: 10
type: B, layer: 1, pos: 10
type: B, layer: 1, pos: 15
type: B, layer: 1, pos: 3
type: A, layer: 1, pos: 37
type: B, layer: 1, pos: 18
type: B, layer: 1, pos: 45
type: B, layer: 1, pos: 16
type: A, layer: 1, pos: 32
type: B, layer: 1, pos: 17
type: B, layer: 1, pos: 43
type: A, layer: 1, pos: 18
type: B, layer: 1, pos: 39
type: A, layer: 1, pos: 21
type: A, layer: 1, pos: 15
type: A, layer: 1, pos: 39
type: A, layer: 1, pos: 41
type: B, layer: 1, pos: 32
type: A, layer: 1, pos: 5
type: B, layer: 1, pos: 23
type: B, layer: 1, pos: 41
type: B, layer: 1, pos: 20
type: A, layer: 1, pos: 20
type: A, layer: 1, pos: 23
type: A, layer: 1, pos: 43
type: B, layer: 1, pos: 5
type: B, layer: 1, pos: 21
type: B, layer: 1, pos: 4
type: B, layer: 1, pos: 37
type: A, layer: 1, pos: 16
type: B, layer: 1, pos: 38
type: B, layer: 1, pos: 29

Time for candidate selection: 0.07 seconds

### Candidate
type: A, layer: 1, pos: 7

## Relational analysis of NS_B2_A1

### Relational analysis result of NS_B2_A1
Status: Status.UNKNOWN
Output dim: 0, lower bound: -63.3940963, upper bound: 63.3937680
time: 0.70 seconds

## Relational analysis of NS_B2_A2

### Relational analysis result of NS_B2_A2
Status: Status.UNKNOWN
Output dim: 0, lower bound: -63.3949094, upper bound: 63.3949094
time: 0.75 seconds

## Summary of splitting at layer (split count: 1)
- Time for NS candidates: 2.41 seconds
NS_B1_B1, status: Status.UNKNOWN, split count: 2, time: 2.41
Output dim: 0, lower bound: -63.3946059, upper bound: 63.3945244
NS_B1_B2, status: Status.UNKNOWN, split count: 2, time: 2.41
Output dim: 0, lower bound: -63.3954223, upper bound: 63.3952755
NS_B2_A1, status: Status.UNKNOWN, split count: 2, time: 2.41
Output dim: 0, lower bound: -63.3940963, upper bound: 63.3937680
NS_B2_A2, status: Status.UNKNOWN, split count: 2, time: 2.41
Output dim: 0, lower bound: -63.3949094, upper bound: 63.3949094

## BFS NS instance: NS_B1_B1

### Backsubstitution after applying NS history:
0: -36.0036278, 37.7448730, -33.5843048, 35.1677933, -71.1714172, 71.3291779
1: -255.4349365, 89.2145386, -238.2765503, 83.1981659, -338.6331177, 327.4910583
2: -142.6184845, 81.5017166, -133.0587158, 75.9643021, -218.5827637, 214.5604095
3: -178.4406433, 65.4211426, -166.4944458, 60.9163361, -239.3569794, 231.9155884
4: -102.6114731, 71.2552490, -95.7579269, 66.2982178, -168.9096985, 167.0131683

Time for backsubstitution: 0.85 seconds

### NS candidates at layer 1
type: A, layer: 1, pos: 40
type: A, layer: 1, pos: 7
type: B, layer: 1, pos: 4
type: A, layer: 1, pos: 9
type: B, layer: 1, pos: 9
type: A, layer: 1, pos: 4
type: A, layer: 1, pos: 35
type: B, layer: 1, pos: 35
type: B, layer: 1, pos: 45
type: A, layer: 1, pos: 17
type: B, layer: 1, pos: 3
type: B, layer: 1, pos: 17
type: A, layer: 1, pos: 19
type: B, layer: 1, pos: 19
type: A, layer: 1, pos: 11
type: A, layer: 1, pos: 45
type: A, layer: 1, pos: 3
type: B, layer: 1, pos: 11
type: B, layer: 1, pos: 10
type: A, layer: 1, pos: 10
type: A, layer: 1, pos: 15
type: B, layer: 1, pos: 18
type: B, layer: 1, pos: 38
type: A, layer: 1, pos: 39
type: A, layer: 1, pos: 18
type: B, layer: 1, pos: 15
type: A, layer: 1, pos: 32
type: B, layer: 1, pos: 32
type: B, layer: 1, pos: 43
type: B, layer: 1, pos: 37
type: A, layer: 1, pos: 43
type: B, layer: 1, pos: 39
type: B, layer: 1, pos: 29
type: A, layer: 1, pos: 37
type: B, layer: 1, pos: 41
type: A, layer: 1, pos: 41
type: B, layer: 1, pos: 20
type: A, layer: 1, pos: 23
type: A, layer: 1, pos: 29
type: A, layer: 1, pos: 21
type: A, layer: 1, pos: 20
type: A, layer: 1, pos: 5
type: B, layer: 1, pos: 5
type: B, layer: 1, pos: 21
type: B, layer: 1, pos: 16
type: A, layer: 1, pos: 16
type: A, layer: 1, pos: 38
type: B, layer: 1, pos: 23

Time for candidate selection: 0.07 seconds

### Candidate
type: A, layer: 1, pos: 40

## Relational analysis of NS_B1_B1_A1

### Relational analysis result of NS_B1_B1_A1
Status: Status.UNKNOWN
Output dim: 0, lower bound: -63.3946059, upper bound: 63.3945244
time: 0.71 seconds

## Relational analysis of NS_B1_B1_A2

### Relational analysis result of NS_B1_B1_A2
Status: Status.UNKNOWN
Output dim: 0, lower bound: -63.3946059, upper bound: 63.3945244
time: 0.72 seconds

## BFS NS instance: NS_B1_B2

### Backsubstitution after applying NS history:
0: -34.9757843, 36.8191261, -33.1110268, 34.9088249, -69.8845978, 69.9301529
1: -249.2049713, 86.5966187, -236.7059174, 81.9441833, -331.1491699, 323.3025513
2: -138.3972015, 79.5439301, -130.8000793, 75.3684692, -213.7656708, 210.3440094
3: -173.4491882, 63.7782097, -164.3968353, 60.3954544, -233.8446198, 228.1750488
4: -99.5938263, 69.6211548, -94.1242828, 65.9827042, -165.5765381, 163.7454376

Time for backsubstitution: 0.86 seconds

### NS candidates at layer 1
type: A, layer: 1, pos: 40
type: A, layer: 1, pos: 7
type: B, layer: 1, pos: 4
type: A, layer: 1, pos: 4
type: A, layer: 1, pos: 9
type: B, layer: 1, pos: 9
type: B, layer: 1, pos: 35
type: A, layer: 1, pos: 35
type: B, layer: 1, pos: 17
type: A, layer: 1, pos: 19
type: A, layer: 1, pos: 17
type: B, layer: 1, pos: 3
type: B, layer: 1, pos: 45
type: B, layer: 1, pos: 19
type: A, layer: 1, pos: 45
type: A, layer: 1, pos: 3
type: B, layer: 1, pos: 11
type: A, layer: 1, pos: 11
type: B, layer: 1, pos: 10
type: A, layer: 1, pos: 10
type: A, layer: 1, pos: 18
type: B, layer: 1, pos: 15
type: A, layer: 1, pos: 15
type: B, layer: 1, pos: 18
type: B, layer: 1, pos: 32
type: A, layer: 1, pos: 39
type: A, layer: 1, pos: 43
type: B, layer: 1, pos: 39
type: A, layer: 1, pos: 38
type: B, layer: 1, pos: 29
type: A, layer: 1, pos: 32
type: A, layer: 1, pos: 37
type: B, layer: 1, pos: 38
type: B, layer: 1, pos: 43
type: B, layer: 1, pos: 37
type: B, layer: 1, pos: 41
type: A, layer: 1, pos: 41
type: B, layer: 1, pos: 20
type: A, layer: 1, pos: 5
type: A, layer: 1, pos: 20
type: B, layer: 1, pos: 21
type: A, layer: 1, pos: 23
type: B, layer: 1, pos: 16
type: A, layer: 1, pos: 29
type: A, layer: 1, pos: 21
type: B, layer: 1, pos: 5
type: B, layer: 1, pos: 23
type: A, layer: 1, pos: 16

Time for candidate selection: 0.07 seconds

### Candidate
type: A, layer: 1, pos: 40

## Relational analysis of NS_B1_B2_A1

### Relational analysis result of NS_B1_B2_A1
Status: Status.UNKNOWN
Output dim: 0, lower bound: -63.3954223, upper bound: 63.3952755
time: 0.74 seconds

## Relational analysis of NS_B1_B2_A2

### Relational analysis result of NS_B1_B2_A2
Status: Status.UNKNOWN
Output dim: 0, lower bound: -63.3954223, upper bound: 63.3952755
time: 0.88 seconds

## BFS NS instance: NS_B2_A1

### Backsubstitution after applying NS history:
0: -35.4014130, 37.2153969, -55.2682686, 59.9139137, -95.3153229, 92.4836655
1: -252.6767578, 87.7712631, -407.2115173, 136.3405151, -389.0172729, 494.4415588
2: -140.3321838, 80.4171143, -218.0220642, 128.4799652, -268.8121033, 298.4391785
3: -176.2659454, 64.5553894, -281.0300293, 102.9374008, -279.2033386, 345.5143738
4: -100.8727875, 70.2676086, -156.6577911, 113.5915985, -214.4643555, 226.9253998

Time for backsubstitution: 1.05 seconds

### NS candidates at layer 1
type: A, layer: 1, pos: 4
type: A, layer: 1, pos: 40
type: A, layer: 1, pos: 9
type: B, layer: 1, pos: 35
type: A, layer: 1, pos: 19
type: A, layer: 1, pos: 17
type: B, layer: 1, pos: 7
type: A, layer: 1, pos: 38
type: A, layer: 1, pos: 3
type: B, layer: 1, pos: 9
type: A, layer: 1, pos: 29
type: B, layer: 1, pos: 11
type: A, layer: 1, pos: 45
type: B, layer: 1, pos: 15
type: A, layer: 1, pos: 35
type: A, layer: 1, pos: 10
type: B, layer: 1, pos: 19
type: A, layer: 1, pos: 11
type: B, layer: 1, pos: 10
type: B, layer: 1, pos: 3
type: A, layer: 1, pos: 37
type: B, layer: 1, pos: 16
type: A, layer: 1, pos: 18
type: B, layer: 1, pos: 17
type: B, layer: 1, pos: 45
type: B, layer: 1, pos: 39
type: B, layer: 1, pos: 18
type: A, layer: 1, pos: 32
type: B, layer: 1, pos: 43
type: A, layer: 1, pos: 21
type: B, layer: 1, pos: 32
type: A, layer: 1, pos: 41
type: A, layer: 1, pos: 39
type: B, layer: 1, pos: 23
type: A, layer: 1, pos: 5
type: A, layer: 1, pos: 15
type: B, layer: 1, pos: 41
type: B, layer: 1, pos: 20
type: A, layer: 1, pos: 20
type: A, layer: 1, pos: 43
type: B, layer: 1, pos: 21
type: B, layer: 1, pos: 5
type: A, layer: 1, pos: 23
type: A, layer: 1, pos: 16
type: B, layer: 1, pos: 37
type: B, layer: 1, pos: 4
type: B, layer: 1, pos: 38
type: B, layer: 1, pos: 29

Time for candidate selection: 0.08 seconds

### Candidate
type: A, layer: 1, pos: 4

## Relational analysis of NS_B2_A1_A1
Optimization infeasible because this subproblem isn't reachable.

### Candidate
type: A, layer: 1, pos: 40

## Relational analysis of NS_B2_A1_A1

### Relational analysis result of NS_B2_A1_A1
Status: Status.UNKNOWN
Output dim: 0, lower bound: -63.3940963, upper bound: 63.3937680
time: 0.71 seconds

## Relational analysis of NS_B2_A1_A2

### Relational analysis result of NS_B2_A1_A2
Status: Status.UNKNOWN
Output dim: 0, lower bound: -63.3940963, upper bound: 63.3937680
time: 0.67 seconds

## BFS NS instance: NS_B2_A2

### Backsubstitution after applying NS history:
0: -34.8330917, 36.7908058, -54.8922081, 59.6490822, -94.4821625, 91.6830139
1: -248.2947388, 86.1418076, -406.4594421, 135.5396118, -383.8343506, 492.6012573
2: -137.5112610, 79.3509903, -217.0846710, 127.8905411, -265.4017639, 296.4356689
3: -172.5016022, 63.6426582, -280.2118225, 102.5400391, -275.0416260, 343.8544922
4: -99.0283432, 69.6066818, -155.4643555, 113.1344757, -212.1627960, 225.0710297

Time for backsubstitution: 0.86 seconds

### NS candidates at layer 1
type: A, layer: 1, pos: 4
type: A, layer: 1, pos: 40
type: A, layer: 1, pos: 3
type: B, layer: 1, pos: 35
type: A, layer: 1, pos: 9
type: A, layer: 1, pos: 19
type: A, layer: 1, pos: 35
type: B, layer: 1, pos: 7
type: B, layer: 1, pos: 3
type: B, layer: 1, pos: 9
type: A, layer: 1, pos: 29
type: A, layer: 1, pos: 17
type: A, layer: 1, pos: 38
type: B, layer: 1, pos: 19
type: B, layer: 1, pos: 43
type: B, layer: 1, pos: 15
type: B, layer: 1, pos: 11
type: B, layer: 1, pos: 18
type: A, layer: 1, pos: 45
type: A, layer: 1, pos: 11
type: A, layer: 1, pos: 10
type: B, layer: 1, pos: 10
type: B, layer: 1, pos: 4
type: A, layer: 1, pos: 21
type: B, layer: 1, pos: 45
type: A, layer: 1, pos: 18
type: A, layer: 1, pos: 43
type: A, layer: 1, pos: 37
type: B, layer: 1, pos: 16
type: A, layer: 1, pos: 32
type: B, layer: 1, pos: 23
type: A, layer: 1, pos: 15
type: B, layer: 1, pos: 17
type: B, layer: 1, pos: 21
type: A, layer: 1, pos: 23
type: B, layer: 1, pos: 39
type: B, layer: 1, pos: 32
type: A, layer: 1, pos: 41
type: A, layer: 1, pos: 20
type: B, layer: 1, pos: 20
type: A, layer: 1, pos: 39
type: A, layer: 1, pos: 5
type: B, layer: 1, pos: 41
type: B, layer: 1, pos: 38
type: B, layer: 1, pos: 5
type: A, layer: 1, pos: 16
type: B, layer: 1, pos: 37
type: B, layer: 1, pos: 29

Time for candidate selection: 0.07 seconds

### Candidate
type: A, layer: 1, pos: 4

## Relational analysis of NS_B2_A2_A1
Optimization infeasible because this subproblem isn't reachable.

### Candidate
type: A, layer: 1, pos: 40

## Relational analysis of NS_B2_A2_A1

### Relational analysis result of NS_B2_A2_A1
Status: Status.UNKNOWN
Output dim: 0, lower bound: -63.3949094, upper bound: 63.3949094
time: 0.88 seconds

## Relational analysis of NS_B2_A2_A2

### Relational analysis result of NS_B2_A2_A2
Status: Status.UNKNOWN
Output dim: 0, lower bound: -63.3949094, upper bound: 63.3949094
time: 1.11 seconds

## Summary of splitting at layer (split count: 2)
- Time for NS candidates: 3.19 seconds
NS_B1_B1_A1, status: Status.UNKNOWN, split count: 3, time: 3.19
Output dim: 0, lower bound: -63.3946059, upper bound: 63.3945244
NS_B1_B1_A2, status: Status.UNKNOWN, split count: 3, time: 3.19
Output dim: 0, lower bound: -63.3946059, upper bound: 63.3945244
NS_B1_B2_A1, status: Status.UNKNOWN, split count: 3, time: 3.19
Output dim: 0, lower bound: -63.3954223, upper bound: 63.3952755
NS_B1_B2_A2, status: Status.UNKNOWN, split count: 3, time: 3.19
Output dim: 0, lower bound: -63.3954223, upper bound: 63.3952755
NS_B2_A1_A1, status: Status.UNKNOWN, split count: 3, time: 3.19
Output dim: 0, lower bound: -63.3940963, upper bound: 63.3937680
NS_B2_A1_A2, status: Status.UNKNOWN, split count: 3, time: 3.19
Output dim: 0, lower bound: -63.3940963, upper bound: 63.3937680
NS_B2_A2_A1, status: Status.UNKNOWN, split count: 3, time: 3.19
Output dim: 0, lower bound: -63.3949094, upper bound: 63.3949094
NS_B2_A2_A2, status: Status.UNKNOWN, split count: 3, time: 3.19
Output dim: 0, lower bound: -63.3949094, upper bound: 63.3949094

## BFS NS instance: NS_B1_B1_A1

### Backsubstitution after applying NS history:
0: -34.9201393, 36.6235657, -33.5843048, 35.1677933, -70.0879211, 70.2078705
1: -248.0101776, 86.5205154, -238.2765503, 83.1981659, -331.2083130, 324.7969666
2: -138.3804932, 79.0562668, -133.0587158, 75.9643021, -214.3447876, 212.1149902
3: -173.2475739, 63.4533463, -166.4944458, 60.9163361, -234.1639099, 229.9477844
4: -99.5461960, 69.1046753, -95.7579269, 66.2982178, -165.8443909, 164.8625641

Time for backsubstitution: 0.87 seconds

### NS candidates at layer 1
type: A, layer: 1, pos: 7
type: B, layer: 1, pos: 4
type: A, layer: 1, pos: 9
type: A, layer: 1, pos: 4
type: B, layer: 1, pos: 9
type: A, layer: 1, pos: 35
type: B, layer: 1, pos: 35
type: A, layer: 1, pos: 17
type: B, layer: 1, pos: 19
type: B, layer: 1, pos: 17
type: B, layer: 1, pos: 45
type: B, layer: 1, pos: 3
type: A, layer: 1, pos: 19
type: A, layer: 1, pos: 3
type: A, layer: 1, pos: 45
type: A, layer: 1, pos: 11
type: B, layer: 1, pos: 11
type: B, layer: 1, pos: 10
type: A, layer: 1, pos: 10
type: A, layer: 1, pos: 15
type: B, layer: 1, pos: 18
type: B, layer: 1, pos: 38
type: A, layer: 1, pos: 18
type: B, layer: 1, pos: 15
type: A, layer: 1, pos: 39
type: A, layer: 1, pos: 32
type: B, layer: 1, pos: 43
type: B, layer: 1, pos: 39
type: B, layer: 1, pos: 32
type: B, layer: 1, pos: 37
type: A, layer: 1, pos: 43
type: A, layer: 1, pos: 37
type: A, layer: 1, pos: 29
type: B, layer: 1, pos: 29
type: B, layer: 1, pos: 41
type: A, layer: 1, pos: 41
type: B, layer: 1, pos: 20
type: A, layer: 1, pos: 20
type: A, layer: 1, pos: 21
type: A, layer: 1, pos: 23
type: A, layer: 1, pos: 5
type: B, layer: 1, pos: 5
type: A, layer: 1, pos: 16
type: B, layer: 1, pos: 21
type: A, layer: 1, pos: 38
type: B, layer: 1, pos: 23
type: B, layer: 1, pos: 16

Time for candidate selection: 0.07 seconds

### Candidate
type: A, layer: 1, pos: 7

## Relational analysis of NS_B1_B1_A1_A1

### Relational analysis result of NS_B1_B1_A1_A1
Status: Status.UNKNOWN
Output dim: 0, lower bound: -63.3945363, upper bound: 63.3943452
time: 0.71 seconds

## Relational analysis of NS_B1_B1_A1_A2

### Relational analysis result of NS_B1_B1_A1_A2
Status: Status.UNKNOWN
Output dim: 0, lower bound: -63.3945363, upper bound: 63.3945244
time: 0.67 seconds

## BFS NS instance: NS_B1_B1_A2

### Backsubstitution after applying NS history:
0: -55.2682686, 59.9139137, -33.5843048, 35.1677933, -90.4360657, 93.4982147
1: -407.2115173, 136.3405151, -238.2765503, 83.1981659, -489.5598145, 374.6169739
2: -218.0220642, 128.4799652, -133.0587158, 75.9643021, -293.9863586, 261.5386047
3: -281.0300293, 102.9374008, -166.4944458, 60.9163361, -341.8305969, 269.4318237
4: -156.6577911, 113.5915985, -95.7579269, 66.2982178, -222.9560089, 209.3495026

Time for backsubstitution: 0.86 seconds

### NS candidates at layer 1
type: B, layer: 1, pos: 4
type: B, layer: 1, pos: 9
type: A, layer: 1, pos: 35
type: B, layer: 1, pos: 38
type: B, layer: 1, pos: 19
type: B, layer: 1, pos: 17
type: B, layer: 1, pos: 3
type: A, layer: 1, pos: 7
type: B, layer: 1, pos: 29
type: A, layer: 1, pos: 9
type: B, layer: 1, pos: 45
type: A, layer: 1, pos: 11
type: A, layer: 1, pos: 15
type: A, layer: 1, pos: 19
type: B, layer: 1, pos: 10
type: B, layer: 1, pos: 11
type: A, layer: 1, pos: 10
type: B, layer: 1, pos: 35
type: A, layer: 1, pos: 16
type: A, layer: 1, pos: 39
type: B, layer: 1, pos: 37
type: B, layer: 1, pos: 18
type: A, layer: 1, pos: 17
type: A, layer: 1, pos: 3
type: A, layer: 1, pos: 43
type: A, layer: 1, pos: 18
type: B, layer: 1, pos: 32
type: A, layer: 1, pos: 23
type: A, layer: 1, pos: 32
type: A, layer: 1, pos: 45
type: B, layer: 1, pos: 21
type: B, layer: 1, pos: 41
type: B, layer: 1, pos: 15
type: B, layer: 1, pos: 5
type: A, layer: 1, pos: 41
type: B, layer: 1, pos: 39
type: A, layer: 1, pos: 20
type: B, layer: 1, pos: 20
type: A, layer: 1, pos: 21
type: A, layer: 1, pos: 5
type: B, layer: 1, pos: 43
type: B, layer: 1, pos: 23
type: A, layer: 1, pos: 37
type: B, layer: 1, pos: 16
type: A, layer: 1, pos: 38
type: A, layer: 1, pos: 4
type: A, layer: 1, pos: 29

Time for candidate selection: 0.07 seconds

### Candidate
type: B, layer: 1, pos: 4

## Relational analysis of NS_B1_B1_A2_B1
Optimization infeasible because this subproblem isn't reachable.

### Candidate
type: B, layer: 1, pos: 9

## Relational analysis of NS_B1_B1_A2_B1
Optimization infeasible because this subproblem isn't reachable.

### Candidate
type: A, layer: 1, pos: 35

## Relational analysis of NS_B1_B1_A2_A1
Optimization infeasible because this subproblem isn't reachable.

### Candidate
type: B, layer: 1, pos: 38

## Relational analysis of NS_B1_B1_A2_B1
Optimization infeasible because this subproblem isn't reachable.

### Candidate
type: B, layer: 1, pos: 19

## Relational analysis of NS_B1_B1_A2_B1

### Relational analysis result of NS_B1_B1_A2_B1
Status: Status.UNKNOWN
Output dim: 0, lower bound: -63.3943650, upper bound: 63.3944516
time: 0.68 seconds

## Relational analysis of NS_B1_B1_A2_B2

### Relational analysis result of NS_B1_B1_A2_B2
Status: Status.UNKNOWN
Output dim: 0, lower bound: -63.3943431, upper bound: 63.3944609
time: 0.71 seconds

## BFS NS instance: NS_B1_B2_A1

### Backsubstitution after applying NS history:
0: -33.9761581, 35.7755737, -33.1110268, 34.9088249, -68.8849792, 68.8865967
1: -242.5032349, 84.1374435, -236.7059174, 81.9441833, -324.4474182, 320.8433533
2: -134.5047455, 77.2907333, -130.8000793, 75.3684692, -209.8732147, 208.0908203
3: -168.7405090, 61.9489403, -164.3968353, 60.3954544, -229.1359558, 226.3457794
4: -96.7570953, 67.6140366, -94.1242828, 65.9827042, -162.7398071, 161.7383118

Time for backsubstitution: 0.87 seconds

### NS candidates at layer 1
type: A, layer: 1, pos: 7
type: B, layer: 1, pos: 4
type: A, layer: 1, pos: 4
type: A, layer: 1, pos: 9
type: B, layer: 1, pos: 9
type: B, layer: 1, pos: 35
type: A, layer: 1, pos: 35
type: B, layer: 1, pos: 17
type: A, layer: 1, pos: 19
type: A, layer: 1, pos: 17
type: B, layer: 1, pos: 19
type: A, layer: 1, pos: 45
type: B, layer: 1, pos: 45
type: B, layer: 1, pos: 3
type: A, layer: 1, pos: 3
type: B, layer: 1, pos: 11
type: A, layer: 1, pos: 11
type: B, layer: 1, pos: 10
type: A, layer: 1, pos: 10
type: B, layer: 1, pos: 15
type: A, layer: 1, pos: 18
type: A, layer: 1, pos: 15
type: B, layer: 1, pos: 18
type: B, layer: 1, pos: 32
type: B, layer: 1, pos: 39
type: A, layer: 1, pos: 39
type: A, layer: 1, pos: 38
type: A, layer: 1, pos: 43
type: A, layer: 1, pos: 32
type: A, layer: 1, pos: 37
type: B, layer: 1, pos: 43
type: B, layer: 1, pos: 29
type: B, layer: 1, pos: 38
type: B, layer: 1, pos: 37
type: A, layer: 1, pos: 41
type: B, layer: 1, pos: 41
type: A, layer: 1, pos: 29
type: B, layer: 1, pos: 20
type: A, layer: 1, pos: 20
type: B, layer: 1, pos: 21
type: A, layer: 1, pos: 5
type: A, layer: 1, pos: 21
type: B, layer: 1, pos: 5
type: B, layer: 1, pos: 23
type: A, layer: 1, pos: 23
type: A, layer: 1, pos: 16
type: B, layer: 1, pos: 16

Time for candidate selection: 0.09 seconds

### Candidate
type: A, layer: 1, pos: 7

## Relational analysis of NS_B1_B2_A1_A1

### Relational analysis result of NS_B1_B2_A1_A1
Status: Status.UNKNOWN
Output dim: 0, lower bound: -63.3945203, upper bound: 63.3943221
time: 0.77 seconds

## Relational analysis of NS_B1_B2_A1_A2

### Relational analysis result of NS_B1_B2_A1_A2
Status: Status.UNKNOWN
Output dim: 0, lower bound: -63.3945203, upper bound: 63.3952755
time: 0.71 seconds

## BFS NS instance: NS_B1_B2_A2

### Backsubstitution after applying NS history:
0: -54.8922081, 59.6490822, -33.1110268, 34.9088249, -89.8010330, 92.7601013
1: -406.4594421, 135.5396118, -236.7059174, 81.9441833, -488.4036255, 372.2455444
2: -217.0846710, 127.8905411, -130.8000793, 75.3684692, -292.4531250, 258.6906128
3: -280.2118225, 102.5400391, -164.3968353, 60.3954544, -340.6072693, 266.9368286
4: -155.4643555, 113.1344757, -94.1242828, 65.9827042, -221.4470520, 207.2587433

Time for backsubstitution: 0.88 seconds

### NS candidates at layer 1
type: B, layer: 1, pos: 4
type: B, layer: 1, pos: 3
type: A, layer: 1, pos: 35
type: B, layer: 1, pos: 9
type: B, layer: 1, pos: 19
type: B, layer: 1, pos: 35
type: A, layer: 1, pos: 3
type: B, layer: 1, pos: 29
type: A, layer: 1, pos: 7
type: A, layer: 1, pos: 9
type: B, layer: 1, pos: 38
type: B, layer: 1, pos: 17
type: A, layer: 1, pos: 43
type: A, layer: 1, pos: 19
type: A, layer: 1, pos: 15
type: B, layer: 1, pos: 45
type: A, layer: 1, pos: 4
type: A, layer: 1, pos: 11
type: B, layer: 1, pos: 11
type: A, layer: 1, pos: 18
type: B, layer: 1, pos: 10
type: A, layer: 1, pos: 10
type: B, layer: 1, pos: 21
type: B, layer: 1, pos: 18
type: B, layer: 1, pos: 43
type: A, layer: 1, pos: 23
type: B, layer: 1, pos: 37
type: A, layer: 1, pos: 45
type: A, layer: 1, pos: 17
type: A, layer: 1, pos: 16
type: A, layer: 1, pos: 21
type: A, layer: 1, pos: 39
type: B, layer: 1, pos: 32
type: B, layer: 1, pos: 15
type: B, layer: 1, pos: 23
type: A, layer: 1, pos: 32
type: B, layer: 1, pos: 41
type: A, layer: 1, pos: 20
type: B, layer: 1, pos: 20
type: A, layer: 1, pos: 41
type: B, layer: 1, pos: 5
type: B, layer: 1, pos: 39
type: A, layer: 1, pos: 38
type: A, layer: 1, pos: 5
type: B, layer: 1, pos: 16
type: A, layer: 1, pos: 37
type: A, layer: 1, pos: 29

Time for candidate selection: 0.07 seconds

### Candidate
type: B, layer: 1, pos: 4

## Relational analysis of NS_B1_B2_A2_B1
Optimization infeasible because this subproblem isn't reachable.

### Candidate
type: B, layer: 1, pos: 3

## Relational analysis of NS_B1_B2_A2_B1
Optimization infeasible because this subproblem isn't reachable.

### Candidate
type: A, layer: 1, pos: 35

## Relational analysis of NS_B1_B2_A2_A1
Optimization infeasible because this subproblem isn't reachable.

### Candidate
type: B, layer: 1, pos: 9

## Relational analysis of NS_B1_B2_A2_B1

### Relational analysis result of NS_B1_B2_A2_B1
Status: Status.UNKNOWN
Output dim: 0, lower bound: -63.3940130, upper bound: 63.3942140
time: 0.80 seconds

## Relational analysis of NS_B1_B2_A2_B2

### Relational analysis result of NS_B1_B2_A2_B2
Status: Status.UNKNOWN
Output dim: 0, lower bound: -63.3941823, upper bound: 63.3942861
time: 0.81 seconds

## BFS NS instance: NS_B2_A1_A1

### Backsubstitution after applying NS history:
0: -33.5843048, 35.1677933, -55.2682686, 59.9139137, -93.4982147, 90.4360657
1: -238.2765503, 83.1981659, -407.2115173, 136.3405151, -374.6169739, 489.5598145
2: -133.0587158, 75.9643021, -218.0220642, 128.4799652, -261.5386047, 293.9863281
3: -166.4944458, 60.9163361, -281.0300293, 102.9374008, -269.4317932, 341.8305969
4: -95.7579269, 66.2982178, -156.6577911, 113.5915985, -209.3495178, 222.9560089

Time for backsubstitution: 0.88 seconds

### NS candidates at layer 1
type: A, layer: 1, pos: 4
type: A, layer: 1, pos: 9
type: B, layer: 1, pos: 35
type: A, layer: 1, pos: 38
type: A, layer: 1, pos: 19
type: A, layer: 1, pos: 17
type: A, layer: 1, pos: 3
type: B, layer: 1, pos: 7
type: A, layer: 1, pos: 29
type: B, layer: 1, pos: 9
type: A, layer: 1, pos: 45
type: B, layer: 1, pos: 11
type: B, layer: 1, pos: 15
type: B, layer: 1, pos: 19
type: A, layer: 1, pos: 10
type: A, layer: 1, pos: 11
type: B, layer: 1, pos: 10
type: A, layer: 1, pos: 35
type: B, layer: 1, pos: 16
type: B, layer: 1, pos: 39
type: A, layer: 1, pos: 37
type: A, layer: 1, pos: 18
type: B, layer: 1, pos: 17
type: B, layer: 1, pos: 3
type: B, layer: 1, pos: 43
type: B, layer: 1, pos: 18
type: A, layer: 1, pos: 32
type: B, layer: 1, pos: 23
type: B, layer: 1, pos: 32
type: B, layer: 1, pos: 45
type: A, layer: 1, pos: 21
type: A, layer: 1, pos: 41
type: A, layer: 1, pos: 15
type: A, layer: 1, pos: 5
type: B, layer: 1, pos: 41
type: A, layer: 1, pos: 39
type: B, layer: 1, pos: 20
type: A, layer: 1, pos: 20
type: B, layer: 1, pos: 21
type: B, layer: 1, pos: 5
type: A, layer: 1, pos: 43
type: A, layer: 1, pos: 23
type: B, layer: 1, pos: 37
type: A, layer: 1, pos: 16
type: B, layer: 1, pos: 38
type: B, layer: 1, pos: 4
type: B, layer: 1, pos: 29

Time for candidate selection: 0.07 seconds

### Candidate
type: A, layer: 1, pos: 4

### Candidate
type: A, layer: 1, pos: 9

## Relational analysis of NS_B2_A1_A1_A1
Optimization infeasible because this subproblem isn't reachable.

### Candidate
type: B, layer: 1, pos: 35

## Relational analysis of NS_B2_A1_A1_B1
Optimization infeasible because this subproblem isn't reachable.

### Candidate
type: A, layer: 1, pos: 38

## Relational analysis of NS_B2_A1_A1_A1
Optimization infeasible because this subproblem isn't reachable.

### Candidate
type: A, layer: 1, pos: 19

## Relational analysis of NS_B2_A1_A1_A1
Optimization infeasible because this subproblem isn't reachable.

### Candidate
type: A, layer: 1, pos: 17

## Relational analysis of NS_B2_A1_A1_A1
Optimization infeasible because this subproblem isn't reachable.

### Candidate
type: A, layer: 1, pos: 3

## Relational analysis of NS_B2_A1_A1_A1
Optimization infeasible because this subproblem isn't reachable.

### Candidate
type: B, layer: 1, pos: 7

## Relational analysis of NS_B2_A1_A1_B1

### Relational analysis result of NS_B2_A1_A1_B1
Status: Status.UNKNOWN
Output dim: 0, lower bound: -63.3933976, upper bound: 63.3933976
time: 0.69 seconds

## Relational analysis of NS_B2_A1_A1_B2

### Relational analysis result of NS_B2_A1_A1_B2
Status: Status.UNKNOWN
Output dim: 0, lower bound: -63.3933976, upper bound: 63.3933976
time: 0.97 seconds

## BFS NS instance: NS_B2_A1_A2

### Backsubstitution after applying NS history:
0: -54.0104523, 58.5131683, -55.2682686, 59.9139137, -113.9243546, 113.7814178
1: -397.0915527, 133.2104645, -407.2115173, 136.3405151, -532.4822998, 539.5666504
2: -212.8295288, 125.4976883, -218.0220642, 128.4799652, -341.3094482, 343.5197144
3: -274.0873108, 100.5429916, -281.0300293, 102.9374008, -376.1135559, 380.7726746
4: -153.0611725, 110.9844284, -156.6577911, 113.5915985, -266.6527710, 267.6422119

Time for backsubstitution: 0.87 seconds

### NS candidates at layer 1
type: B, layer: 1, pos: 7
type: A, layer: 1, pos: 4
type: B, layer: 1, pos: 4
type: B, layer: 1, pos: 9
type: A, layer: 1, pos: 9
type: B, layer: 1, pos: 35
type: A, layer: 1, pos: 35
type: B, layer: 1, pos: 17
type: A, layer: 1, pos: 17
type: A, layer: 1, pos: 45
type: A, layer: 1, pos: 19
type: A, layer: 1, pos: 3
type: B, layer: 1, pos: 45
type: B, layer: 1, pos: 19
type: B, layer: 1, pos: 3
type: A, layer: 1, pos: 10
type: B, layer: 1, pos: 10
type: B, layer: 1, pos: 11
type: A, layer: 1, pos: 11
type: A, layer: 1, pos: 18
type: B, layer: 1, pos: 15
type: A, layer: 1, pos: 38
type: B, layer: 1, pos: 18
type: A, layer: 1, pos: 15
type: B, layer: 1, pos: 39
type: A, layer: 1, pos: 43
type: B, layer: 1, pos: 32
type: A, layer: 1, pos: 39
type: A, layer: 1, pos: 37
type: B, layer: 1, pos: 29
type: B, layer: 1, pos: 37
type: A, layer: 1, pos: 41
type: B, layer: 1, pos: 43
type: B, layer: 1, pos: 41
type: A, layer: 1, pos: 32
type: A, layer: 1, pos: 29
type: A, layer: 1, pos: 20
type: B, layer: 1, pos: 16
type: B, layer: 1, pos: 38
type: B, layer: 1, pos: 5
type: A, layer: 1, pos: 5
type: B, layer: 1, pos: 20
type: B, layer: 1, pos: 21
type: B, layer: 1, pos: 23
type: A, layer: 1, pos: 16
type: A, layer: 1, pos: 21
type: A, layer: 1, pos: 23

Time for candidate selection: 0.07 seconds

### Candidate
type: B, layer: 1, pos: 7

## Relational analysis of NS_B2_A1_A2_B1

### Relational analysis result of NS_B2_A1_A2_B1
Status: Status.UNKNOWN
Output dim: 0, lower bound: -63.3933976, upper bound: 63.3933976
time: 0.70 seconds

## Relational analysis of NS_B2_A1_A2_B2

### Relational analysis result of NS_B2_A1_A2_B2
Status: Status.UNKNOWN
Output dim: 0, lower bound: -63.3933976, upper bound: 63.3933976
time: 0.69 seconds

## BFS NS instance: NS_B2_A2_A1

### Backsubstitution after applying NS history:
0: -33.1092682, 34.9069939, -54.8922081, 59.6490822, -92.7583313, 89.7992020
1: -236.6929016, 81.9397430, -406.4594421, 135.5396118, -372.2325134, 488.3991699
2: -130.7926941, 75.3644485, -217.0846710, 127.8905411, -258.6832275, 292.4490967
3: -164.3877411, 60.3922920, -280.2118225, 102.5400391, -266.9277344, 340.6041260
4: -94.1191254, 65.9793015, -155.4643555, 113.1344757, -207.2535858, 221.4436340

Time for backsubstitution: 0.89 seconds

### NS candidates at layer 1
type: A, layer: 1, pos: 4
type: A, layer: 1, pos: 3
type: B, layer: 1, pos: 35
type: A, layer: 1, pos: 9
type: A, layer: 1, pos: 19
type: A, layer: 1, pos: 35
type: B, layer: 1, pos: 3
type: A, layer: 1, pos: 29
type: B, layer: 1, pos: 7
type: B, layer: 1, pos: 9
type: A, layer: 1, pos: 38
type: A, layer: 1, pos: 17
type: B, layer: 1, pos: 43
type: B, layer: 1, pos: 19
type: B, layer: 1, pos: 15
type: A, layer: 1, pos: 45
type: B, layer: 1, pos: 4
type: B, layer: 1, pos: 11
type: A, layer: 1, pos: 11
type: B, layer: 1, pos: 18
type: A, layer: 1, pos: 10
type: B, layer: 1, pos: 10
type: A, layer: 1, pos: 21
type: A, layer: 1, pos: 18
type: A, layer: 1, pos: 43
type: B, layer: 1, pos: 23
type: A, layer: 1, pos: 37
type: B, layer: 1, pos: 45
type: B, layer: 1, pos: 17
type: B, layer: 1, pos: 16
type: B, layer: 1, pos: 21
type: B, layer: 1, pos: 39
type: A, layer: 1, pos: 32
type: A, layer: 1, pos: 15
type: A, layer: 1, pos: 23
type: B, layer: 1, pos: 32
type: A, layer: 1, pos: 41
type: B, layer: 1, pos: 20
type: A, layer: 1, pos: 20
type: B, layer: 1, pos: 41
type: A, layer: 1, pos: 5
type: A, layer: 1, pos: 39
type: B, layer: 1, pos: 38
type: B, layer: 1, pos: 5
type: A, layer: 1, pos: 16
type: B, layer: 1, pos: 37
type: B, layer: 1, pos: 29

Time for candidate selection: 0.07 seconds

### Candidate
type: A, layer: 1, pos: 4

### Candidate
type: A, layer: 1, pos: 3

## Relational analysis of NS_B2_A2_A1_A1
Optimization infeasible because this subproblem isn't reachable.

### Candidate
type: B, layer: 1, pos: 35

## Relational analysis of NS_B2_A2_A1_B1
Optimization infeasible because this subproblem isn't reachable.

### Candidate
type: A, layer: 1, pos: 9

## Relational analysis of NS_B2_A2_A1_A1
Optimization infeasible because this subproblem isn't reachable.

### Candidate
type: A, layer: 1, pos: 19

## Relational analysis of NS_B2_A2_A1_A1
Optimization infeasible because this subproblem isn't reachable.

### Candidate
type: A, layer: 1, pos: 35

### Candidate
type: B, layer: 1, pos: 3

### Candidate
type: A, layer: 1, pos: 29

## Relational analysis of NS_B2_A2_A1_A1

### Relational analysis result of NS_B2_A2_A1_A1
Status: Status.UNKNOWN
Output dim: 0, lower bound: -63.3946560, upper bound: 63.3947170
time: 0.65 seconds

## Relational analysis of NS_B2_A2_A1_A2

### Relational analysis result of NS_B2_A2_A1_A2
Status: Status.UNKNOWN
Output dim: 0, lower bound: -63.3947778, upper bound: 63.3947778
time: 0.76 seconds

## BFS NS instance: NS_B2_A2_A2

### Backsubstitution after applying NS history:
0: -54.0260849, 58.7897339, -54.8922081, 59.6490822, -113.6751251, 113.6819458
1: -400.6096802, 133.4270172, -406.4594421, 135.5396118, -536.1492920, 539.8864746
2: -213.7019806, 126.0158386, -217.0846710, 127.8905411, -341.5924683, 343.1005249
3: -276.0601196, 101.0406876, -280.2118225, 102.5400391, -378.6001587, 381.2525024
4: -152.9686127, 111.5431519, -155.4643555, 113.1344757, -266.1030579, 267.0074768

Time for backsubstitution: 0.91 seconds

### NS candidates at layer 1
type: B, layer: 1, pos: 7
type: A, layer: 1, pos: 35
type: B, layer: 1, pos: 35
type: A, layer: 1, pos: 9
type: B, layer: 1, pos: 9
type: A, layer: 1, pos: 19
type: B, layer: 1, pos: 19
type: A, layer: 1, pos: 3
type: B, layer: 1, pos: 3
type: A, layer: 1, pos: 17
type: B, layer: 1, pos: 17
type: B, layer: 1, pos: 10
type: A, layer: 1, pos: 10
type: B, layer: 1, pos: 18
type: A, layer: 1, pos: 11
type: A, layer: 1, pos: 15
type: B, layer: 1, pos: 11
type: A, layer: 1, pos: 45
type: B, layer: 1, pos: 43
type: A, layer: 1, pos: 18
type: B, layer: 1, pos: 45
type: A, layer: 1, pos: 43
type: A, layer: 1, pos: 29
type: B, layer: 1, pos: 15
type: B, layer: 1, pos: 29
type: A, layer: 1, pos: 21
type: B, layer: 1, pos: 21
type: A, layer: 1, pos: 23
type: B, layer: 1, pos: 23
type: A, layer: 1, pos: 32
type: B, layer: 1, pos: 37
type: B, layer: 1, pos: 32
type: A, layer: 1, pos: 20
type: B, layer: 1, pos: 38
type: A, layer: 1, pos: 37
type: A, layer: 1, pos: 39
type: B, layer: 1, pos: 20
type: B, layer: 1, pos: 41
type: B, layer: 1, pos: 39
type: A, layer: 1, pos: 41
type: B, layer: 1, pos: 16
type: A, layer: 1, pos: 16
type: A, layer: 1, pos: 5
type: A, layer: 1, pos: 38
type: B, layer: 1, pos: 5
type: A, layer: 1, pos: 4
type: B, layer: 1, pos: 4

Time for candidate selection: 0.07 seconds

### Candidate
type: B, layer: 1, pos: 7

## Relational analysis of NS_B2_A2_A2_B1

### Relational analysis result of NS_B2_A2_A2_B1
Status: Status.UNKNOWN
Output dim: 0, lower bound: -63.3937680, upper bound: 63.3940963
time: 0.73 seconds

## Relational analysis of NS_B2_A2_A2_B2

### Relational analysis result of NS_B2_A2_A2_B2
Status: Status.UNKNOWN
Output dim: 0, lower bound: -63.3937680, upper bound: 63.3949094
time: 0.77 seconds

## Summary of splitting at layer (split count: 3)
- Time for NS candidates: 2.48 seconds
NS_B1_B1_A1_A1, status: Status.UNKNOWN, split count: 4, time: 2.48
Output dim: 0, lower bound: -63.3945363, upper bound: 63.3943452
NS_B1_B1_A1_A2, status: Status.UNKNOWN, split count: 4, time: 2.48
Output dim: 0, lower bound: -63.3945363, upper bound: 63.3945244
NS_B1_B1_A2_B1, status: Status.UNKNOWN, split count: 4, time: 2.48
Output dim: 0, lower bound: -63.3943650, upper bound: 63.3944516
NS_B1_B1_A2_B2, status: Status.UNKNOWN, split count: 4, time: 2.48
Output dim: 0, lower bound: -63.3943431, upper bound: 63.3944609
NS_B1_B2_A1_A1, status: Status.UNKNOWN, split count: 4, time: 2.48
Output dim: 0, lower bound: -63.3945203, upper bound: 63.3943221
NS_B1_B2_A1_A2, status: Status.UNKNOWN, split count: 4, time: 2.48
Output dim: 0, lower bound: -63.3945203, upper bound: 63.3952755
NS_B1_B2_A2_B1, status: Status.UNKNOWN, split count: 4, time: 2.48
Output dim: 0, lower bound: -63.3940130, upper bound: 63.3942140
NS_B1_B2_A2_B2, status: Status.UNKNOWN, split count: 4, time: 2.48
Output dim: 0, lower bound: -63.3941823, upper bound: 63.3942861
NS_B2_A1_A1_B1, status: Status.UNKNOWN, split count: 4, time: 2.48
Output dim: 0, lower bound: -63.3933976, upper bound: 63.3933976
NS_B2_A1_A1_B2, status: Status.UNKNOWN, split count: 4, time: 2.48
Output dim: 0, lower bound: -63.3933976, upper bound: 63.3933976
NS_B2_A1_A2_B1, status: Status.UNKNOWN, split count: 4, time: 2.48
Output dim: 0, lower bound: -63.3933976, upper bound: 63.3933976
NS_B2_A1_A2_B2, status: Status.UNKNOWN, split count: 4, time: 2.48
Output dim: 0, lower bound: -63.3933976, upper bound: 63.3933976
NS_B2_A2_A1_A1, status: Status.UNKNOWN, split count: 4, time: 2.48
Output dim: 0, lower bound: -63.3946560, upper bound: 63.3947170
NS_B2_A2_A1_A2, status: Status.UNKNOWN, split count: 4, time: 2.48
Output dim: 0, lower bound: -63.3947778, upper bound: 63.3947778
NS_B2_A2_A2_B1, status: Status.UNKNOWN, split count: 4, time: 2.48
Output dim: 0, lower bound: -63.3937680, upper bound: 63.3940963
NS_B2_A2_A2_B2, status: Status.UNKNOWN, split count: 4, time: 2.48
Output dim: 0, lower bound: -63.3937680, upper bound: 63.3949094

## BFS NS instance: NS_B1_B1_A1_A1

### Backsubstitution after applying NS history:
0: -33.5843048, 35.1677933, -33.5843048, 35.1677933, -68.7520981, 68.7520981
1: -238.2765503, 83.1981659, -238.2765503, 83.1981659, -321.4746399, 321.4746399
2: -133.0587158, 75.9643021, -133.0587158, 75.9643021, -209.0229950, 209.0229950
3: -166.4944458, 60.9163361, -166.4944458, 60.9163361, -227.4107819, 227.4107819
4: -95.7579269, 66.2982178, -95.7579269, 66.2982178, -162.0561371, 162.0561218

Time for backsubstitution: 0.91 seconds

### NS candidates at layer 1
type: B, layer: 1, pos: 4
type: A, layer: 1, pos: 4
type: A, layer: 1, pos: 9
type: B, layer: 1, pos: 9
type: A, layer: 1, pos: 35
type: B, layer: 1, pos: 35
type: B, layer: 1, pos: 17
type: A, layer: 1, pos: 17
type: A, layer: 1, pos: 19
type: B, layer: 1, pos: 19
type: A, layer: 1, pos: 45
type: B, layer: 1, pos: 45
type: A, layer: 1, pos: 3
type: B, layer: 1, pos: 3
type: A, layer: 1, pos: 11
type: B, layer: 1, pos: 11
type: A, layer: 1, pos: 10
type: B, layer: 1, pos: 10
type: A, layer: 1, pos: 15
type: B, layer: 1, pos: 15
type: A, layer: 1, pos: 18
type: B, layer: 1, pos: 18
type: A, layer: 1, pos: 39
type: B, layer: 1, pos: 39
type: B, layer: 1, pos: 32
type: A, layer: 1, pos: 32
type: A, layer: 1, pos: 43
type: B, layer: 1, pos: 43
type: A, layer: 1, pos: 38
type: B, layer: 1, pos: 38
type: A, layer: 1, pos: 37
type: B, layer: 1, pos: 37
type: A, layer: 1, pos: 29
type: B, layer: 1, pos: 29
type: B, layer: 1, pos: 41
type: A, layer: 1, pos: 41
type: A, layer: 1, pos: 20
type: B, layer: 1, pos: 20
type: B, layer: 1, pos: 21
type: A, layer: 1, pos: 21
type: B, layer: 1, pos: 5
type: A, layer: 1, pos: 5
type: A, layer: 1, pos: 23
type: B, layer: 1, pos: 23
type: A, layer: 1, pos: 16
type: B, layer: 1, pos: 16

Time for candidate selection: 0.07 seconds

### Candidate
type: B, layer: 1, pos: 4

## Relational analysis of NS_B1_B1_A1_A1_B1

### Relational analysis result of NS_B1_B1_A1_A1_B1
Status: Status.UNKNOWN
Output dim: 0, lower bound: -63.3958876, upper bound: 63.3958122
time: 0.80 seconds

## Relational analysis of NS_B1_B1_A1_A1_B2

### Relational analysis result of NS_B1_B1_A1_A1_B2
Status: Status.UNKNOWN
Output dim: 0, lower bound: -63.3958450, upper bound: 63.3957590
time: 0.71 seconds

## BFS NS instance: NS_B1_B1_A1_A2

### Backsubstitution after applying NS history:
0: -33.1110268, 34.9088249, -33.5843048, 35.1677933, -68.2788239, 68.4931335
1: -236.7059174, 81.9441833, -238.2765503, 83.1981659, -319.9040527, 320.2206726
2: -130.8000793, 75.3684692, -133.0587158, 75.9643021, -206.7643738, 208.4271851
3: -164.3968353, 60.3954544, -166.4944458, 60.9163361, -225.3131714, 226.8898926
4: -94.1242828, 65.9827042, -95.7579269, 66.2982178, -160.4224854, 161.7406311

Time for backsubstitution: 0.86 seconds

### NS candidates at layer 1
type: B, layer: 1, pos: 4
type: A, layer: 1, pos: 4
type: B, layer: 1, pos: 9
type: A, layer: 1, pos: 9
type: A, layer: 1, pos: 35
type: B, layer: 1, pos: 45
type: B, layer: 1, pos: 35
type: B, layer: 1, pos: 19
type: B, layer: 1, pos: 3
type: B, layer: 1, pos: 17
type: A, layer: 1, pos: 17
type: A, layer: 1, pos: 19
type: A, layer: 1, pos: 11
type: B, layer: 1, pos: 11
type: A, layer: 1, pos: 10
type: A, layer: 1, pos: 3
type: A, layer: 1, pos: 45
type: B, layer: 1, pos: 10
type: A, layer: 1, pos: 15
type: B, layer: 1, pos: 18
type: B, layer: 1, pos: 15
type: B, layer: 1, pos: 38
type: A, layer: 1, pos: 39
type: A, layer: 1, pos: 18
type: A, layer: 1, pos: 32
type: B, layer: 1, pos: 43
type: B, layer: 1, pos: 32
type: B, layer: 1, pos: 39
type: B, layer: 1, pos: 37
type: A, layer: 1, pos: 43
type: A, layer: 1, pos: 29
type: B, layer: 1, pos: 41
type: A, layer: 1, pos: 37
type: A, layer: 1, pos: 41
type: B, layer: 1, pos: 29
type: B, layer: 1, pos: 20
type: A, layer: 1, pos: 23
type: A, layer: 1, pos: 20
type: B, layer: 1, pos: 5
type: A, layer: 1, pos: 21
type: A, layer: 1, pos: 38
type: A, layer: 1, pos: 16
type: B, layer: 1, pos: 21
type: A, layer: 1, pos: 5
type: B, layer: 1, pos: 23
type: B, layer: 1, pos: 16

Time for candidate selection: 0.07 seconds

### Candidate
type: B, layer: 1, pos: 4

## Relational analysis of NS_B1_B1_A1_A2_B1

### Relational analysis result of NS_B1_B1_A1_A2_B1
Status: Status.UNKNOWN
Output dim: 0, lower bound: -63.3958876, upper bound: 63.3958122
time: 1.06 seconds

## Relational analysis of NS_B1_B1_A1_A2_B2

### Relational analysis result of NS_B1_B1_A1_A2_B2
Status: Status.UNKNOWN
Output dim: 0, lower bound: -63.3958450, upper bound: 63.3957590
time: 0.67 seconds

## BFS NS instance: NS_B1_B1_A2_B1

### Backsubstitution after applying NS history:
0: -54.7686424, 59.4249878, -32.7632294, 34.3331223, -89.1017609, 92.1882172
1: -404.0156250, 135.1101379, -232.5169525, 81.1931534, -484.2805481, 367.6269531
2: -216.1138153, 127.4112854, -129.8301544, 74.1390152, -290.2528381, 257.2414551
3: -278.7328796, 102.0758667, -162.4364471, 59.4611740, -338.0152283, 264.5122986
4: -155.2354126, 112.6637726, -93.4227905, 64.7076111, -219.9430237, 206.0865631

Time for backsubstitution: 0.93 seconds

### NS candidates at layer 1
type: B, layer: 1, pos: 4
type: B, layer: 1, pos: 9
type: A, layer: 1, pos: 35
type: B, layer: 1, pos: 38
type: B, layer: 1, pos: 17
type: B, layer: 1, pos: 3
type: B, layer: 1, pos: 29
type: A, layer: 1, pos: 7
type: A, layer: 1, pos: 9
type: B, layer: 1, pos: 45
type: A, layer: 1, pos: 11
type: A, layer: 1, pos: 15
type: B, layer: 1, pos: 11
type: B, layer: 1, pos: 10
type: A, layer: 1, pos: 10
type: B, layer: 1, pos: 35
type: A, layer: 1, pos: 16
type: A, layer: 1, pos: 39
type: B, layer: 1, pos: 37
type: A, layer: 1, pos: 19
type: B, layer: 1, pos: 18
type: A, layer: 1, pos: 3
type: A, layer: 1, pos: 17
type: A, layer: 1, pos: 43
type: B, layer: 1, pos: 32
type: A, layer: 1, pos: 18
type: A, layer: 1, pos: 45
type: A, layer: 1, pos: 23
type: B, layer: 1, pos: 21
type: A, layer: 1, pos: 32
type: B, layer: 1, pos: 41
type: B, layer: 1, pos: 15
type: B, layer: 1, pos: 5
type: A, layer: 1, pos: 41
type: A, layer: 1, pos: 20
type: B, layer: 1, pos: 39
type: B, layer: 1, pos: 20
type: A, layer: 1, pos: 21
type: A, layer: 1, pos: 5
type: B, layer: 1, pos: 43
type: B, layer: 1, pos: 23
type: A, layer: 1, pos: 37
type: B, layer: 1, pos: 16
type: A, layer: 1, pos: 38
type: A, layer: 1, pos: 4
type: A, layer: 1, pos: 29

Time for candidate selection: 0.10 seconds

### Candidate
type: B, layer: 1, pos: 4

### Candidate
type: B, layer: 1, pos: 9

### Candidate
type: A, layer: 1, pos: 35

### Candidate
type: B, layer: 1, pos: 38

### Candidate
type: B, layer: 1, pos: 17

## Relational analysis of NS_B1_B1_A2_B1_B1
Optimization infeasible because this subproblem isn't reachable.

### Candidate
type: B, layer: 1, pos: 3

## Relational analysis of NS_B1_B1_A2_B1_B1
Optimization infeasible because this subproblem isn't reachable.

### Candidate
type: B, layer: 1, pos: 29

## Relational analysis of NS_B1_B1_A2_B1_B1

### Relational analysis result of NS_B1_B1_A2_B1_B1
Status: Status.UNKNOWN
Output dim: 0, lower bound: -63.3943110, upper bound: 63.3942986
time: 0.69 seconds

## Relational analysis of NS_B1_B1_A2_B1_B2

### Relational analysis result of NS_B1_B1_A2_B1_B2
Status: Status.UNKNOWN
Output dim: 0, lower bound: -63.3941184, upper bound: 63.3941425
time: 0.68 seconds

## BFS NS instance: NS_B1_B1_A2_B2

### Backsubstitution after applying NS history:
0: -54.7002220, 59.3067627, -35.4041939, 37.2144241, -91.9146423, 94.7109375
1: -403.1087341, 134.9294434, -253.3442535, 87.7649612, -489.9754333, 388.2736206
2: -215.7784271, 127.1595306, -140.9738159, 80.3180618, -296.0964966, 268.1333008
3: -278.2004395, 101.8846817, -176.9305115, 64.4613800, -342.4980164, 278.8151550
4: -155.0654449, 112.4233398, -101.1564407, 70.0503616, -225.1158142, 213.5797729

Time for backsubstitution: 0.91 seconds

### NS candidates at layer 1
type: B, layer: 1, pos: 4
type: B, layer: 1, pos: 3
type: A, layer: 1, pos: 35
type: B, layer: 1, pos: 9
type: B, layer: 1, pos: 29
type: B, layer: 1, pos: 38
type: B, layer: 1, pos: 17
type: A, layer: 1, pos: 7
type: B, layer: 1, pos: 45
type: A, layer: 1, pos: 9
type: A, layer: 1, pos: 11
type: B, layer: 1, pos: 11
type: A, layer: 1, pos: 15
type: A, layer: 1, pos: 10
type: B, layer: 1, pos: 35
type: B, layer: 1, pos: 10
type: A, layer: 1, pos: 39
type: A, layer: 1, pos: 3
type: A, layer: 1, pos: 17
type: B, layer: 1, pos: 18
type: A, layer: 1, pos: 16
type: B, layer: 1, pos: 37
type: A, layer: 1, pos: 45
type: A, layer: 1, pos: 18
type: B, layer: 1, pos: 15
type: A, layer: 1, pos: 19
type: A, layer: 1, pos: 43
type: A, layer: 1, pos: 23
type: A, layer: 1, pos: 32
type: B, layer: 1, pos: 41
type: B, layer: 1, pos: 32
type: B, layer: 1, pos: 5
type: B, layer: 1, pos: 21
type: A, layer: 1, pos: 41
type: A, layer: 1, pos: 20
type: B, layer: 1, pos: 20
type: A, layer: 1, pos: 21
type: A, layer: 1, pos: 5
type: B, layer: 1, pos: 39
type: B, layer: 1, pos: 43
type: A, layer: 1, pos: 4
type: B, layer: 1, pos: 23
type: A, layer: 1, pos: 37
type: B, layer: 1, pos: 16
type: A, layer: 1, pos: 38
type: A, layer: 1, pos: 29

Time for candidate selection: 0.07 seconds

### Candidate
type: B, layer: 1, pos: 4

### Candidate
type: B, layer: 1, pos: 3

## Relational analysis of NS_B1_B1_A2_B2_B1
Optimization infeasible because this subproblem isn't reachable.

### Candidate
type: A, layer: 1, pos: 35

### Candidate
type: B, layer: 1, pos: 9

### Candidate
type: B, layer: 1, pos: 29

## Relational analysis of NS_B1_B1_A2_B2_B1

### Relational analysis result of NS_B1_B1_A2_B2_B1
Status: Status.UNKNOWN
Output dim: 0, lower bound: -63.3942807, upper bound: 63.3943366
time: 0.90 seconds

## Relational analysis of NS_B1_B1_A2_B2_B2

### Relational analysis result of NS_B1_B1_A2_B2_B2
Status: Status.UNKNOWN
Output dim: 0, lower bound: -63.3942638, upper bound: 63.3943223
time: 1.22 seconds

## BFS NS instance: NS_B1_B2_A1_A1

### Backsubstitution after applying NS history:
0: -33.5843048, 35.1677933, -33.1110268, 34.9088249, -68.4931335, 68.2788239
1: -238.2765503, 83.1981659, -236.7059174, 81.9441833, -320.2206726, 319.9040527
2: -133.0587158, 75.9643021, -130.8000793, 75.3684692, -208.4271851, 206.7643738
3: -166.4944458, 60.9163361, -164.3968353, 60.3954544, -226.8898926, 225.3131714
4: -95.7579269, 66.2982178, -94.1242828, 65.9827042, -161.7406158, 160.4224854

Time for backsubstitution: 0.91 seconds

### NS candidates at layer 1
type: A, layer: 1, pos: 4
type: B, layer: 1, pos: 4
type: A, layer: 1, pos: 9
type: B, layer: 1, pos: 9
type: B, layer: 1, pos: 35
type: A, layer: 1, pos: 45
type: A, layer: 1, pos: 35
type: A, layer: 1, pos: 19
type: A, layer: 1, pos: 3
type: A, layer: 1, pos: 17
type: B, layer: 1, pos: 17
type: B, layer: 1, pos: 19
type: B, layer: 1, pos: 11
type: A, layer: 1, pos: 11
type: B, layer: 1, pos: 10
type: B, layer: 1, pos: 3
type: B, layer: 1, pos: 45
type: A, layer: 1, pos: 10
type: B, layer: 1, pos: 15
type: A, layer: 1, pos: 18
type: A, layer: 1, pos: 15
type: A, layer: 1, pos: 38
type: B, layer: 1, pos: 39
type: B, layer: 1, pos: 18
type: B, layer: 1, pos: 32
type: A, layer: 1, pos: 43
type: A, layer: 1, pos: 32
type: A, layer: 1, pos: 39
type: A, layer: 1, pos: 37
type: B, layer: 1, pos: 43
type: B, layer: 1, pos: 29
type: A, layer: 1, pos: 41
type: B, layer: 1, pos: 37
type: B, layer: 1, pos: 41
type: A, layer: 1, pos: 29
type: A, layer: 1, pos: 20
type: B, layer: 1, pos: 23
type: B, layer: 1, pos: 20
type: A, layer: 1, pos: 5
type: B, layer: 1, pos: 21
type: B, layer: 1, pos: 38
type: B, layer: 1, pos: 16
type: A, layer: 1, pos: 21
type: B, layer: 1, pos: 5
type: A, layer: 1, pos: 23
type: A, layer: 1, pos: 16

Time for candidate selection: 0.07 seconds

### Candidate
type: A, layer: 1, pos: 4

## Relational analysis of NS_B1_B2_A1_A1_A1

### Relational analysis result of NS_B1_B2_A1_A1_A1
Status: Status.UNKNOWN
Output dim: 0, lower bound: -63.3957216, upper bound: 63.3958435
time: 0.65 seconds

## Relational analysis of NS_B1_B2_A1_A1_A2

### Relational analysis result of NS_B1_B2_A1_A1_A2
Status: Status.UNKNOWN
Output dim: 0, lower bound: -63.3954598, upper bound: 63.3954598
time: 0.81 seconds

## BFS NS instance: NS_B1_B2_A1_A2

### Backsubstitution after applying NS history:
0: -33.1110268, 34.9088249, -33.1110268, 34.9088249, -68.0198517, 68.0198517
1: -236.7059174, 81.9441833, -236.7059174, 81.9441833, -318.6500854, 318.6500549
2: -130.8000793, 75.3684692, -130.8000793, 75.3684692, -206.1685486, 206.1685486
3: -164.3968353, 60.3954544, -164.3968353, 60.3954544, -224.7922516, 224.7922516
4: -94.1242828, 65.9827042, -94.1242828, 65.9827042, -160.1069794, 160.1069794

Time for backsubstitution: 1.07 seconds

### NS candidates at layer 1
type: B, layer: 1, pos: 4
type: A, layer: 1, pos: 4
type: B, layer: 1, pos: 9
type: A, layer: 1, pos: 9
type: A, layer: 1, pos: 35
type: B, layer: 1, pos: 35
type: B, layer: 1, pos: 17
type: A, layer: 1, pos: 17
type: A, layer: 1, pos: 19
type: B, layer: 1, pos: 19
type: B, layer: 1, pos: 45
type: A, layer: 1, pos: 45
type: B, layer: 1, pos: 3
type: A, layer: 1, pos: 3
type: A, layer: 1, pos: 11
type: B, layer: 1, pos: 11
type: B, layer: 1, pos: 10
type: A, layer: 1, pos: 10
type: A, layer: 1, pos: 15
type: B, layer: 1, pos: 15
type: A, layer: 1, pos: 18
type: B, layer: 1, pos: 18
type: A, layer: 1, pos: 39
type: B, layer: 1, pos: 39
type: B, layer: 1, pos: 32
type: A, layer: 1, pos: 32
type: A, layer: 1, pos: 43
type: B, layer: 1, pos: 43
type: B, layer: 1, pos: 38
type: A, layer: 1, pos: 38
type: B, layer: 1, pos: 37
type: A, layer: 1, pos: 37
type: B, layer: 1, pos: 29
type: A, layer: 1, pos: 29
type: B, layer: 1, pos: 41
type: A, layer: 1, pos: 41
type: A, layer: 1, pos: 20
type: B, layer: 1, pos: 20
type: B, layer: 1, pos: 21
type: A, layer: 1, pos: 21
type: B, layer: 1, pos: 5
type: A, layer: 1, pos: 5
type: A, layer: 1, pos: 23
type: B, layer: 1, pos: 23
type: A, layer: 1, pos: 16
type: B, layer: 1, pos: 16

Time for candidate selection: 0.13 seconds

### Candidate
type: B, layer: 1, pos: 4

## Relational analysis of NS_B1_B2_A1_A2_B1

### Relational analysis result of NS_B1_B2_A1_A2_B1
Status: Status.UNKNOWN
Output dim: 0, lower bound: -63.3958435, upper bound: 63.3957216
time: 1.26 seconds

## Relational analysis of NS_B1_B2_A1_A2_B2

### Relational analysis result of NS_B1_B2_A1_A2_B2
Status: Status.UNKNOWN
Output dim: 0, lower bound: -63.3954598, upper bound: 63.3954598
time: 0.67 seconds

## BFS NS instance: NS_B1_B2_A2_B1

### Backsubstitution after applying NS history:
0: -53.5134697, 58.3027115, -31.7075348, 33.5421410, -87.0556030, 90.0102310
1: -397.7839661, 132.1765289, -228.2451782, 78.4206696, -476.2046509, 360.4216309
2: -211.6885529, 124.9430847, -125.0585175, 72.3499680, -284.0385132, 250.0016022
3: -273.8537598, 100.1888733, -157.9605103, 57.9945068, -331.8482666, 258.1493835
4: -151.4457550, 110.6002960, -89.8834915, 63.3862991, -214.8320465, 200.4837799

Time for backsubstitution: 1.05 seconds

### NS candidates at layer 1
type: B, layer: 1, pos: 4
type: B, layer: 1, pos: 29
type: B, layer: 1, pos: 3
type: A, layer: 1, pos: 4
type: B, layer: 1, pos: 19
type: A, layer: 1, pos: 35
type: B, layer: 1, pos: 35
type: B, layer: 1, pos: 38
type: A, layer: 1, pos: 19
type: A, layer: 1, pos: 3
type: A, layer: 1, pos: 7
type: B, layer: 1, pos: 45
type: A, layer: 1, pos: 29
type: A, layer: 1, pos: 9
type: B, layer: 1, pos: 17
type: A, layer: 1, pos: 15
type: A, layer: 1, pos: 43
type: A, layer: 1, pos: 45
type: A, layer: 1, pos: 11
type: B, layer: 1, pos: 21
type: A, layer: 1, pos: 18
type: A, layer: 1, pos: 38
type: B, layer: 1, pos: 11
type: B, layer: 1, pos: 10
type: A, layer: 1, pos: 23
type: A, layer: 1, pos: 21
type: A, layer: 1, pos: 39
type: A, layer: 1, pos: 10
type: B, layer: 1, pos: 18
type: B, layer: 1, pos: 37
type: A, layer: 1, pos: 16
type: B, layer: 1, pos: 23
type: B, layer: 1, pos: 20
type: B, layer: 1, pos: 15
type: B, layer: 1, pos: 32
type: A, layer: 1, pos: 20
type: B, layer: 1, pos: 43
type: A, layer: 1, pos: 32
type: B, layer: 1, pos: 41
type: B, layer: 1, pos: 39
type: A, layer: 1, pos: 41
type: A, layer: 1, pos: 17
type: B, layer: 1, pos: 5
type: A, layer: 1, pos: 5
type: B, layer: 1, pos: 16
type: A, layer: 1, pos: 37

Time for candidate selection: 0.09 seconds

### Candidate
type: B, layer: 1, pos: 4

### Candidate
type: B, layer: 1, pos: 29

## Relational analysis of NS_B1_B2_A2_B1_B1

### Relational analysis result of NS_B1_B2_A2_B1_B1
Status: Status.UNKNOWN
Output dim: 0, lower bound: -63.3939212, upper bound: 63.3941125
time: 0.74 seconds

## Relational analysis of NS_B1_B2_A2_B1_B2

### Relational analysis result of NS_B1_B2_A2_B1_B2
Status: Status.UNKNOWN
Output dim: 0, lower bound: -63.3936884, upper bound: 63.3935634
time: 0.67 seconds

## BFS NS instance: NS_B1_B2_A2_B2

### Backsubstitution after applying NS history:
0: -54.1864319, 58.9121284, -31.9944019, 33.7464180, -87.9328461, 90.9065247
1: -401.2415771, 133.7657928, -228.5129700, 79.2402725, -480.4818420, 362.2787476
2: -214.2349548, 126.2974319, -126.8174133, 72.9059601, -287.1408691, 253.1148376
3: -276.5735779, 101.2141190, -159.0127563, 58.4444389, -335.0180054, 260.2268677
4: -153.4540100, 111.7632599, -91.2179718, 63.8624268, -217.3164215, 202.9812012

Time for backsubstitution: 1.03 seconds

### NS candidates at layer 1
type: B, layer: 1, pos: 4
type: B, layer: 1, pos: 3
type: A, layer: 1, pos: 35
type: B, layer: 1, pos: 19
type: B, layer: 1, pos: 35
type: A, layer: 1, pos: 3
type: B, layer: 1, pos: 29
type: A, layer: 1, pos: 7
type: B, layer: 1, pos: 38
type: B, layer: 1, pos: 17
type: A, layer: 1, pos: 9
type: A, layer: 1, pos: 43
type: A, layer: 1, pos: 19
type: A, layer: 1, pos: 15
type: B, layer: 1, pos: 45
type: A, layer: 1, pos: 4
type: A, layer: 1, pos: 11
type: B, layer: 1, pos: 10
type: A, layer: 1, pos: 18
type: B, layer: 1, pos: 11
type: A, layer: 1, pos: 10
type: B, layer: 1, pos: 21
type: B, layer: 1, pos: 18
type: B, layer: 1, pos: 43
type: A, layer: 1, pos: 23
type: A, layer: 1, pos: 45
type: B, layer: 1, pos: 37
type: A, layer: 1, pos: 16
type: A, layer: 1, pos: 17
type: A, layer: 1, pos: 21
type: A, layer: 1, pos: 39
type: B, layer: 1, pos: 32
type: B, layer: 1, pos: 15
type: B, layer: 1, pos: 23
type: A, layer: 1, pos: 32
type: B, layer: 1, pos: 41
type: B, layer: 1, pos: 20
type: A, layer: 1, pos: 20
type: B, layer: 1, pos: 5
type: A, layer: 1, pos: 41
type: B, layer: 1, pos: 39
type: A, layer: 1, pos: 38
type: A, layer: 1, pos: 5
type: B, layer: 1, pos: 16
type: A, layer: 1, pos: 37
type: A, layer: 1, pos: 29

Time for candidate selection: 0.09 seconds

### Candidate
type: B, layer: 1, pos: 4

### Candidate
type: B, layer: 1, pos: 3

### Candidate
type: A, layer: 1, pos: 35

### Candidate
type: B, layer: 1, pos: 19

## Relational analysis of NS_B1_B2_A2_B2_B1

### Relational analysis result of NS_B1_B2_A2_B2_B1
Status: Status.UNKNOWN
Output dim: 0, lower bound: -63.3935961, upper bound: 63.3942849
time: 0.73 seconds

## Relational analysis of NS_B1_B2_A2_B2_B2

### Relational analysis result of NS_B1_B2_A2_B2_B2
Status: Status.UNKNOWN
Output dim: 0, lower bound: -63.3941813, upper bound: 63.3942861
time: 0.81 seconds

## BFS NS instance: NS_B2_A1_A1_B1

### Backsubstitution after applying NS history:
0: -33.5843048, 35.1677933, -54.0113716, 58.5142365, -92.0985413, 89.1791687
1: -238.2765503, 83.1981659, -397.1004944, 133.2129364, -371.4894714, 479.2958069
2: -133.0587158, 75.9643021, -212.8338928, 125.4999237, -258.5586243, 288.7981873
3: -166.4944458, 60.9163361, -274.0934448, 100.5448303, -267.0392456, 334.7619934
4: -95.7579269, 66.2982178, -153.0641937, 110.9862289, -206.7441559, 219.3623962

Time for backsubstitution: 1.00 seconds

### NS candidates at layer 1
type: A, layer: 1, pos: 4
type: A, layer: 1, pos: 9
type: B, layer: 1, pos: 35
type: A, layer: 1, pos: 17
type: A, layer: 1, pos: 19
type: A, layer: 1, pos: 3
type: A, layer: 1, pos: 38
type: A, layer: 1, pos: 29
type: A, layer: 1, pos: 45
type: B, layer: 1, pos: 9
type: B, layer: 1, pos: 11
type: B, layer: 1, pos: 19
type: A, layer: 1, pos: 11
type: A, layer: 1, pos: 10
type: B, layer: 1, pos: 15
type: B, layer: 1, pos: 10
type: A, layer: 1, pos: 35
type: B, layer: 1, pos: 16
type: B, layer: 1, pos: 43
type: B, layer: 1, pos: 39
type: B, layer: 1, pos: 18
type: A, layer: 1, pos: 37
type: B, layer: 1, pos: 3
type: A, layer: 1, pos: 32
type: B, layer: 1, pos: 17
type: A, layer: 1, pos: 18
type: A, layer: 1, pos: 21
type: B, layer: 1, pos: 45
type: B, layer: 1, pos: 23
type: A, layer: 1, pos: 41
type: A, layer: 1, pos: 15
type: B, layer: 1, pos: 32
type: A, layer: 1, pos: 5
type: A, layer: 1, pos: 39
type: B, layer: 1, pos: 41
type: B, layer: 1, pos: 20
type: A, layer: 1, pos: 20
type: B, layer: 1, pos: 5
type: B, layer: 1, pos: 21
type: A, layer: 1, pos: 23
type: A, layer: 1, pos: 43
type: B, layer: 1, pos: 37
type: A, layer: 1, pos: 16
type: B, layer: 1, pos: 38
type: B, layer: 1, pos: 4
type: B, layer: 1, pos: 29

Time for candidate selection: 0.08 seconds

### Candidate
type: A, layer: 1, pos: 4

### Candidate
type: A, layer: 1, pos: 9

### Candidate
type: B, layer: 1, pos: 35

### Candidate
type: A, layer: 1, pos: 17

### Candidate
type: A, layer: 1, pos: 19

### Candidate
type: A, layer: 1, pos: 3

### Candidate
type: A, layer: 1, pos: 38

### Candidate
type: A, layer: 1, pos: 29

## Relational analysis of NS_B2_A1_A1_B1_A1

### Relational analysis result of NS_B2_A1_A1_B1_A1
Status: Status.UNKNOWN
Output dim: 0, lower bound: -63.3942734, upper bound: 63.3945363
time: 0.69 seconds

## Relational analysis of NS_B2_A1_A1_B1_A2

### Relational analysis result of NS_B2_A1_A1_B1_A2
Status: Status.UNKNOWN
Output dim: 0, lower bound: -63.3941174, upper bound: 63.3943349
time: 0.67 seconds

## BFS NS instance: NS_B2_A1_A1_B2

### Backsubstitution after applying NS history:
0: -33.5843048, 35.1677933, -54.0260849, 58.7897339, -92.3740387, 89.1938705
1: -238.2765503, 83.1981659, -400.6096802, 133.4270172, -371.7035217, 483.2600403
2: -133.0587158, 75.9643021, -213.7019806, 126.0158386, -259.0745239, 289.6662292
3: -166.4944458, 60.9163361, -276.0601196, 101.0406876, -267.5351257, 336.9764404
4: -95.7579269, 66.2982178, -152.9686127, 111.5431519, -207.3010712, 219.2668304

Time for backsubstitution: 0.92 seconds

### NS candidates at layer 1
type: A, layer: 1, pos: 4
type: A, layer: 1, pos: 3
type: B, layer: 1, pos: 35
type: A, layer: 1, pos: 9
type: A, layer: 1, pos: 19
type: A, layer: 1, pos: 38
type: B, layer: 1, pos: 3
type: A, layer: 1, pos: 35
type: B, layer: 1, pos: 9
type: A, layer: 1, pos: 29
type: A, layer: 1, pos: 17
type: B, layer: 1, pos: 43
type: B, layer: 1, pos: 19
type: B, layer: 1, pos: 15
type: A, layer: 1, pos: 45
type: B, layer: 1, pos: 4
type: B, layer: 1, pos: 11
type: B, layer: 1, pos: 10
type: A, layer: 1, pos: 11
type: A, layer: 1, pos: 10
type: A, layer: 1, pos: 18
type: B, layer: 1, pos: 18
type: A, layer: 1, pos: 43
type: B, layer: 1, pos: 23
type: A, layer: 1, pos: 37
type: A, layer: 1, pos: 21
type: B, layer: 1, pos: 16
type: B, layer: 1, pos: 39
type: B, layer: 1, pos: 17
type: B, layer: 1, pos: 21
type: B, layer: 1, pos: 32
type: A, layer: 1, pos: 32
type: B, layer: 1, pos: 45
type: A, layer: 1, pos: 15
type: A, layer: 1, pos: 23
type: A, layer: 1, pos: 41
type: A, layer: 1, pos: 20
type: B, layer: 1, pos: 20
type: A, layer: 1, pos: 5
type: B, layer: 1, pos: 41
type: A, layer: 1, pos: 39
type: B, layer: 1, pos: 5
type: B, layer: 1, pos: 38
type: A, layer: 1, pos: 16
type: B, layer: 1, pos: 37
type: B, layer: 1, pos: 29

Time for candidate selection: 0.08 seconds

### Candidate
type: A, layer: 1, pos: 4

### Candidate
type: A, layer: 1, pos: 3

### Candidate
type: B, layer: 1, pos: 35

### Candidate
type: A, layer: 1, pos: 9

### Candidate
type: A, layer: 1, pos: 19

### Candidate
type: A, layer: 1, pos: 38

### Candidate
type: B, layer: 1, pos: 3

### Candidate
type: A, layer: 1, pos: 35

### Candidate
type: B, layer: 1, pos: 9

### Candidate
type: A, layer: 1, pos: 29

## Relational analysis of NS_B2_A1_A1_B2_A1

### Relational analysis result of NS_B2_A1_A1_B2_A1
Status: Status.UNKNOWN
Output dim: 0, lower bound: -63.3942734, upper bound: 63.3945955
time: 0.78 seconds

## Relational analysis of NS_B2_A1_A1_B2_A2

### Relational analysis result of NS_B2_A1_A1_B2_A2
Status: Status.UNKNOWN
Output dim: 0, lower bound: -63.3941174, upper bound: 63.3945137
time: 0.78 seconds

## BFS NS instance: NS_B2_A1_A2_B1

### Backsubstitution after applying NS history:
0: -54.0104523, 58.5131683, -54.0113716, 58.5142365, -112.5246735, 112.5245285
1: -397.0915527, 133.2104645, -397.1004944, 133.2129364, -529.2963257, 529.3025513
2: -212.8295288, 125.4976883, -212.8338928, 125.4999237, -338.3294373, 338.3315430
3: -274.0873108, 100.5429916, -274.0934448, 100.5448303, -373.6997375, 373.7040405
4: -153.0611725, 110.9844284, -153.0641937, 110.9862289, -264.0473938, 264.0486145

Time for backsubstitution: 1.04 seconds

### NS candidates at layer 1
type: B, layer: 1, pos: 4
type: A, layer: 1, pos: 4
type: B, layer: 1, pos: 9
type: A, layer: 1, pos: 9
type: A, layer: 1, pos: 35
type: B, layer: 1, pos: 35
type: A, layer: 1, pos: 17
type: B, layer: 1, pos: 17
type: B, layer: 1, pos: 45
type: A, layer: 1, pos: 45
type: B, layer: 1, pos: 19
type: A, layer: 1, pos: 19
type: B, layer: 1, pos: 3
type: A, layer: 1, pos: 3
type: A, layer: 1, pos: 10
type: B, layer: 1, pos: 10
type: A, layer: 1, pos: 11
type: B, layer: 1, pos: 11
type: B, layer: 1, pos: 18
type: A, layer: 1, pos: 18
type: A, layer: 1, pos: 15
type: B, layer: 1, pos: 15
type: B, layer: 1, pos: 39
type: A, layer: 1, pos: 39
type: B, layer: 1, pos: 43
type: B, layer: 1, pos: 37
type: A, layer: 1, pos: 43
type: B, layer: 1, pos: 38
type: A, layer: 1, pos: 37
type: A, layer: 1, pos: 32
type: A, layer: 1, pos: 38
type: B, layer: 1, pos: 32
type: A, layer: 1, pos: 41
type: B, layer: 1, pos: 41
type: B, layer: 1, pos: 29
type: A, layer: 1, pos: 29
type: B, layer: 1, pos: 20
type: A, layer: 1, pos: 20
type: A, layer: 1, pos: 5
type: B, layer: 1, pos: 5
type: A, layer: 1, pos: 16
type: B, layer: 1, pos: 16
type: A, layer: 1, pos: 21
type: B, layer: 1, pos: 21
type: A, layer: 1, pos: 23
type: B, layer: 1, pos: 23

Time for candidate selection: 0.07 seconds

### Candidate
type: B, layer: 1, pos: 4

### Candidate
type: A, layer: 1, pos: 4

### Candidate
type: B, layer: 1, pos: 9

## Relational analysis of NS_B2_A1_A2_B1_B1
Optimization infeasible because this subproblem isn't reachable.

### Candidate
type: A, layer: 1, pos: 9

### Candidate
type: A, layer: 1, pos: 35

## Relational analysis of NS_B2_A1_A2_B1_A1
Optimization infeasible because this subproblem isn't reachable.

### Candidate
type: B, layer: 1, pos: 35

### Candidate
type: A, layer: 1, pos: 17

## Relational analysis of NS_B2_A1_A2_B1_A1
Optimization infeasible because this subproblem isn't reachable.

### Candidate
type: B, layer: 1, pos: 17

### Candidate
type: B, layer: 1, pos: 45

## Relational analysis of NS_B2_A1_A2_B1_B1
Optimization infeasible because this subproblem isn't reachable.

### Candidate
type: A, layer: 1, pos: 45

### Candidate
type: B, layer: 1, pos: 19

## Relational analysis of NS_B2_A1_A2_B1_B1
Optimization infeasible because this subproblem isn't reachable.

### Candidate
type: A, layer: 1, pos: 19

### Candidate
type: B, layer: 1, pos: 3

## Relational analysis of NS_B2_A1_A2_B1_B1
Optimization infeasible because this subproblem isn't reachable.

### Candidate
type: A, layer: 1, pos: 3

### Candidate
type: A, layer: 1, pos: 10

## Relational analysis of NS_B2_A1_A2_B1_A1
Optimization infeasible because this subproblem isn't reachable.

### Candidate
type: B, layer: 1, pos: 10

### Candidate
type: A, layer: 1, pos: 11

## Relational analysis of NS_B2_A1_A2_B1_A1

### Relational analysis result of NS_B2_A1_A2_B1_A1
Status: Status.UNKNOWN
Output dim: 0, lower bound: -63.3930077, upper bound: 63.3930551
time: 0.71 seconds

## Relational analysis of NS_B2_A1_A2_B1_A2

### Relational analysis result of NS_B2_A1_A2_B1_A2
Status: Status.UNKNOWN
Output dim: 0, lower bound: -63.3931305, upper bound: 63.3931305
time: 0.82 seconds

## BFS NS instance: NS_B2_A1_A2_B2

### Backsubstitution after applying NS history:
0: -54.0104523, 58.5131683, -54.0260849, 58.7897339, -112.8001862, 112.5392151
1: -397.0915527, 133.2104645, -400.6096802, 133.4270172, -529.6282349, 533.2668457
2: -212.8295288, 125.4976883, -213.7019806, 126.0158386, -338.8453674, 339.1996155
3: -274.0873108, 100.5429916, -276.0601196, 101.0406876, -374.2534485, 376.0151672
4: -153.0611725, 110.9844284, -152.9686127, 111.5431519, -264.6043091, 263.9530029

Time for backsubstitution: 0.98 seconds

### NS candidates at layer 1
type: B, layer: 1, pos: 35
type: A, layer: 1, pos: 35
type: B, layer: 1, pos: 9
type: A, layer: 1, pos: 9
type: A, layer: 1, pos: 19
type: B, layer: 1, pos: 19
type: A, layer: 1, pos: 3
type: B, layer: 1, pos: 3
type: A, layer: 1, pos: 17
type: B, layer: 1, pos: 17
type: B, layer: 1, pos: 10
type: A, layer: 1, pos: 18
type: A, layer: 1, pos: 10
type: A, layer: 1, pos: 45
type: B, layer: 1, pos: 11
type: A, layer: 1, pos: 11
type: B, layer: 1, pos: 15
type: A, layer: 1, pos: 43
type: B, layer: 1, pos: 18
type: A, layer: 1, pos: 15
type: B, layer: 1, pos: 29
type: B, layer: 1, pos: 45
type: B, layer: 1, pos: 43
type: A, layer: 1, pos: 29
type: B, layer: 1, pos: 21
type: A, layer: 1, pos: 21
type: B, layer: 1, pos: 23
type: A, layer: 1, pos: 37
type: A, layer: 1, pos: 23
type: B, layer: 1, pos: 32
type: A, layer: 1, pos: 32
type: B, layer: 1, pos: 39
type: A, layer: 1, pos: 38
type: B, layer: 1, pos: 37
type: A, layer: 1, pos: 20
type: A, layer: 1, pos: 4
type: A, layer: 1, pos: 41
type: B, layer: 1, pos: 41
type: B, layer: 1, pos: 20
type: A, layer: 1, pos: 39
type: B, layer: 1, pos: 16
type: A, layer: 1, pos: 5
type: B, layer: 1, pos: 5
type: A, layer: 1, pos: 16
type: B, layer: 1, pos: 4
type: B, layer: 1, pos: 38

Time for candidate selection: 0.08 seconds

### Candidate
type: B, layer: 1, pos: 35

## Relational analysis of NS_B2_A1_A2_B2_B1
Optimization infeasible because this subproblem isn't reachable.

### Candidate
type: A, layer: 1, pos: 35

### Candidate
type: B, layer: 1, pos: 9

## Relational analysis of NS_B2_A1_A2_B2_B1
Optimization infeasible because this subproblem isn't reachable.

### Candidate
type: A, layer: 1, pos: 9

### Candidate
type: A, layer: 1, pos: 19

## Relational analysis of NS_B2_A1_A2_B2_A1
Optimization infeasible because this subproblem isn't reachable.

### Candidate
type: B, layer: 1, pos: 19

### Candidate
type: A, layer: 1, pos: 3

## Relational analysis of NS_B2_A1_A2_B2_A1
Optimization infeasible because this subproblem isn't reachable.

### Candidate
type: B, layer: 1, pos: 3

### Candidate
type: A, layer: 1, pos: 17

## Relational analysis of NS_B2_A1_A2_B2_A1
Optimization infeasible because this subproblem isn't reachable.

### Candidate
type: B, layer: 1, pos: 17

### Candidate
type: B, layer: 1, pos: 10

## Relational analysis of NS_B2_A1_A2_B2_B1
Optimization infeasible because this subproblem isn't reachable.

### Candidate
type: A, layer: 1, pos: 18

## Relational analysis of NS_B2_A1_A2_B2_A1
Optimization infeasible because this subproblem isn't reachable.

### Candidate
type: A, layer: 1, pos: 10

### Candidate
type: A, layer: 1, pos: 45

## Relational analysis of NS_B2_A1_A2_B2_A1
Optimization infeasible because this subproblem isn't reachable.

### Candidate
type: B, layer: 1, pos: 11

## Relational analysis of NS_B2_A1_A2_B2_B1

### Relational analysis result of NS_B2_A1_A2_B2_B1
Status: Status.UNKNOWN
Output dim: 0, lower bound: -63.3930551, upper bound: 63.3930898
time: 0.73 seconds

## Relational analysis of NS_B2_A1_A2_B2_B2

### Relational analysis result of NS_B2_A1_A2_B2_B2
Status: Status.UNKNOWN
Output dim: 0, lower bound: -63.3931305, upper bound: 63.3935402
time: 0.79 seconds

## BFS NS instance: NS_B2_A2_A1_A1

### Backsubstitution after applying NS history:
0: -31.2933064, 32.9650536, -53.9953995, 58.7144966, -90.0078049, 86.9604263
1: -223.5657806, 77.4652863, -400.3821106, 133.3540344, -356.9197693, 477.8474121
2: -123.6531296, 71.1243896, -213.6814728, 125.8472977, -249.5004272, 284.8058472
3: -155.3525391, 57.0316010, -275.9838257, 100.9185486, -256.2710876, 333.0154114
4: -88.9686127, 62.2370262, -152.9810638, 111.3096771, -200.2782745, 215.2180939

Time for backsubstitution: 0.93 seconds

### NS candidates at layer 1
type: A, layer: 1, pos: 4
type: A, layer: 1, pos: 3
type: B, layer: 1, pos: 35
type: A, layer: 1, pos: 19
type: A, layer: 1, pos: 9
type: A, layer: 1, pos: 35
type: B, layer: 1, pos: 3
type: A, layer: 1, pos: 38
type: B, layer: 1, pos: 7
type: B, layer: 1, pos: 9
type: A, layer: 1, pos: 17
type: B, layer: 1, pos: 43
type: B, layer: 1, pos: 19
type: B, layer: 1, pos: 15
type: A, layer: 1, pos: 45
type: B, layer: 1, pos: 4
type: B, layer: 1, pos: 11
type: A, layer: 1, pos: 11
type: A, layer: 1, pos: 10
type: B, layer: 1, pos: 18
type: B, layer: 1, pos: 10
type: A, layer: 1, pos: 18
type: A, layer: 1, pos: 43
type: A, layer: 1, pos: 21
type: B, layer: 1, pos: 23
type: A, layer: 1, pos: 37
type: B, layer: 1, pos: 16
type: B, layer: 1, pos: 45
type: B, layer: 1, pos: 17
type: B, layer: 1, pos: 39
type: B, layer: 1, pos: 21
type: B, layer: 1, pos: 32
type: A, layer: 1, pos: 23
type: A, layer: 1, pos: 15
type: A, layer: 1, pos: 32
type: B, layer: 1, pos: 20
type: A, layer: 1, pos: 41
type: A, layer: 1, pos: 20
type: B, layer: 1, pos: 41
type: A, layer: 1, pos: 5
type: A, layer: 1, pos: 39
type: B, layer: 1, pos: 5
type: B, layer: 1, pos: 38
type: A, layer: 1, pos: 16
type: B, layer: 1, pos: 37
type: B, layer: 1, pos: 29

Time for candidate selection: 0.08 seconds

### Candidate
type: A, layer: 1, pos: 4

### Candidate
type: A, layer: 1, pos: 3

### Candidate
type: B, layer: 1, pos: 35

### Candidate
type: A, layer: 1, pos: 19

### Candidate
type: A, layer: 1, pos: 9

### Candidate
type: A, layer: 1, pos: 35

### Candidate
type: B, layer: 1, pos: 3

### Candidate
type: A, layer: 1, pos: 38

## Relational analysis of NS_B2_A2_A1_A1_A1
Optimization infeasible because this subproblem isn't reachable.

### Candidate
type: B, layer: 1, pos: 7

## Relational analysis of NS_B2_A2_A1_A1_B1

### Relational analysis result of NS_B2_A2_A1_A1_B1
Status: Status.UNKNOWN
Output dim: 0, lower bound: -63.3942459, upper bound: 63.3945203
time: 0.78 seconds

## Relational analysis of NS_B2_A2_A1_A1_B2

### Relational analysis result of NS_B2_A2_A1_A1_B2
Status: Status.UNKNOWN
Output dim: 0, lower bound: -63.3942459, upper bound: 63.3952135
time: 0.75 seconds

## BFS NS instance: NS_B2_A2_A1_A2

### Backsubstitution after applying NS history:
0: -41.4766655, 42.6777725, -53.3992004, 57.9703140, -99.4469528, 96.0769501
1: -280.1462708, 102.6841431, -393.9755554, 131.7529907, -411.8992310, 496.6596985
2: -161.1961517, 92.3508987, -210.8420868, 124.2164688, -285.4126282, 303.1929626
3: -197.2792816, 73.9040070, -271.7917480, 99.5846939, -296.8639832, 345.6956787
4: -117.7232513, 80.8190765, -151.1241608, 109.9499054, -227.6731567, 231.9432373

Time for backsubstitution: 1.13 seconds

### NS candidates at layer 1
type: A, layer: 1, pos: 4
type: A, layer: 1, pos: 3
type: B, layer: 1, pos: 35
type: A, layer: 1, pos: 19
type: A, layer: 1, pos: 9
type: A, layer: 1, pos: 35
type: B, layer: 1, pos: 9
type: B, layer: 1, pos: 7
type: A, layer: 1, pos: 17
type: B, layer: 1, pos: 3
type: A, layer: 1, pos: 38
type: B, layer: 1, pos: 19
type: B, layer: 1, pos: 10
type: B, layer: 1, pos: 15
type: A, layer: 1, pos: 45
type: A, layer: 1, pos: 18
type: B, layer: 1, pos: 11
type: A, layer: 1, pos: 11
type: B, layer: 1, pos: 4
type: A, layer: 1, pos: 43
type: B, layer: 1, pos: 43
type: B, layer: 1, pos: 17
type: B, layer: 1, pos: 29
type: B, layer: 1, pos: 32
type: A, layer: 1, pos: 21
type: A, layer: 1, pos: 37
type: A, layer: 1, pos: 10
type: B, layer: 1, pos: 16
type: B, layer: 1, pos: 18
type: A, layer: 1, pos: 15
type: B, layer: 1, pos: 21
type: B, layer: 1, pos: 20
type: B, layer: 1, pos: 45
type: B, layer: 1, pos: 23
type: B, layer: 1, pos: 39
type: A, layer: 1, pos: 23
type: A, layer: 1, pos: 32
type: A, layer: 1, pos: 41
type: B, layer: 1, pos: 41
type: A, layer: 1, pos: 39
type: A, layer: 1, pos: 5
type: A, layer: 1, pos: 20
type: B, layer: 1, pos: 5
type: B, layer: 1, pos: 38
type: B, layer: 1, pos: 37
type: A, layer: 1, pos: 16

Time for candidate selection: 0.09 seconds

### Candidate
type: A, layer: 1, pos: 4

### Candidate
type: A, layer: 1, pos: 3

### Candidate
type: B, layer: 1, pos: 35

### Candidate
type: A, layer: 1, pos: 19

### Candidate
type: A, layer: 1, pos: 9

### Candidate
type: A, layer: 1, pos: 35

### Candidate
type: B, layer: 1, pos: 9

### Candidate
type: B, layer: 1, pos: 7

## Relational analysis of NS_B2_A2_A1_A2_B1

### Relational analysis result of NS_B2_A2_A1_A2_B1
Status: Status.UNKNOWN
Output dim: 0, lower bound: -63.3940284, upper bound: 63.3944068
time: 0.78 seconds

## Relational analysis of NS_B2_A2_A1_A2_B2

### Relational analysis result of NS_B2_A2_A1_A2_B2
Status: Status.UNKNOWN
Output dim: 0, lower bound: -63.3940284, upper bound: 63.3949890
time: 0.77 seconds

## BFS NS instance: NS_B2_A2_A2_B1

### Backsubstitution after applying NS history:
0: -54.0260849, 58.7897339, -53.9753265, 58.4763908, -112.5024490, 112.7650604
1: -400.6096802, 133.4270172, -396.8188477, 133.1195221, -533.1740723, 529.3531494
2: -213.7019806, 126.0158386, -212.6795044, 125.4174194, -339.1193848, 338.6953430
3: -276.0601196, 101.0406876, -273.8961182, 100.4767838, -375.9484253, 374.0577393
4: -152.9686127, 111.5431519, -152.9559479, 110.9175262, -263.8861389, 264.4990845

Time for backsubstitution: 1.17 seconds

### NS candidates at layer 1
type: A, layer: 1, pos: 35
type: B, layer: 1, pos: 35
type: A, layer: 1, pos: 9
type: B, layer: 1, pos: 9
type: B, layer: 1, pos: 19
type: A, layer: 1, pos: 19
type: B, layer: 1, pos: 3
type: A, layer: 1, pos: 3
type: B, layer: 1, pos: 17
type: A, layer: 1, pos: 17
type: A, layer: 1, pos: 10
type: B, layer: 1, pos: 18
type: B, layer: 1, pos: 10
type: B, layer: 1, pos: 45
type: A, layer: 1, pos: 11
type: B, layer: 1, pos: 11
type: A, layer: 1, pos: 15
type: B, layer: 1, pos: 43
type: A, layer: 1, pos: 18
type: B, layer: 1, pos: 15
type: A, layer: 1, pos: 29
type: A, layer: 1, pos: 45
type: A, layer: 1, pos: 43
type: B, layer: 1, pos: 29
type: A, layer: 1, pos: 21
type: A, layer: 1, pos: 23
type: B, layer: 1, pos: 21
type: B, layer: 1, pos: 37
type: A, layer: 1, pos: 32
type: B, layer: 1, pos: 23
type: B, layer: 1, pos: 4
type: B, layer: 1, pos: 38
type: B, layer: 1, pos: 32
type: A, layer: 1, pos: 39
type: B, layer: 1, pos: 20
type: A, layer: 1, pos: 37
type: B, layer: 1, pos: 41
type: A, layer: 1, pos: 41
type: A, layer: 1, pos: 20
type: B, layer: 1, pos: 39
type: A, layer: 1, pos: 16
type: B, layer: 1, pos: 5
type: A, layer: 1, pos: 4
type: B, layer: 1, pos: 16
type: A, layer: 1, pos: 5
type: A, layer: 1, pos: 38

Time for candidate selection: 0.10 seconds

### Candidate
type: A, layer: 1, pos: 35

## Relational analysis of NS_B2_A2_A2_B1_A1
Optimization infeasible because this subproblem isn't reachable.

### Candidate
type: B, layer: 1, pos: 35

### Candidate
type: A, layer: 1, pos: 9

## Relational analysis of NS_B2_A2_A2_B1_A1
Optimization infeasible because this subproblem isn't reachable.

### Candidate
type: B, layer: 1, pos: 9

### Candidate
type: B, layer: 1, pos: 19

## Relational analysis of NS_B2_A2_A2_B1_B1
Optimization infeasible because this subproblem isn't reachable.

### Candidate
type: A, layer: 1, pos: 19

### Candidate
type: B, layer: 1, pos: 3

## Relational analysis of NS_B2_A2_A2_B1_B1
Optimization infeasible because this subproblem isn't reachable.

### Candidate
type: A, layer: 1, pos: 3

### Candidate
type: B, layer: 1, pos: 17

## Relational analysis of NS_B2_A2_A2_B1_B1
Optimization infeasible because this subproblem isn't reachable.

### Candidate
type: A, layer: 1, pos: 17

### Candidate
type: A, layer: 1, pos: 10

## Relational analysis of NS_B2_A2_A2_B1_A1
Optimization infeasible because this subproblem isn't reachable.

### Candidate
type: B, layer: 1, pos: 18

## Relational analysis of NS_B2_A2_A2_B1_B1
Optimization infeasible because this subproblem isn't reachable.

### Candidate
type: B, layer: 1, pos: 10

### Candidate
type: B, layer: 1, pos: 45

## Relational analysis of NS_B2_A2_A2_B1_B1
Optimization infeasible because this subproblem isn't reachable.

### Candidate
type: A, layer: 1, pos: 11

## Relational analysis of NS_B2_A2_A2_B1_A1

### Relational analysis result of NS_B2_A2_A2_B1_A1
Status: Status.UNKNOWN
Output dim: 0, lower bound: -63.3930898, upper bound: 63.3931718
time: 0.77 seconds

## Relational analysis of NS_B2_A2_A2_B1_A2

### Relational analysis result of NS_B2_A2_A2_B1_A2
Status: Status.UNKNOWN
Output dim: 0, lower bound: -63.3935402, upper bound: 63.3938446
time: 0.76 seconds

## BFS NS instance: NS_B2_A2_A2_B2

### Backsubstitution after applying NS history:
0: -54.0260849, 58.7897339, -54.0260849, 58.7897339, -112.8158112, 112.8158112
1: -400.6096802, 133.4270172, -400.6096802, 133.4270172, -534.0366821, 534.0366821
2: -213.7019806, 126.0158386, -213.7019806, 126.0158386, -339.7178040, 339.7178040
3: -276.0601196, 101.0406876, -276.0601196, 101.0406876, -377.1007996, 377.1007996
4: -152.9686127, 111.5431519, -152.9686127, 111.5431519, -264.5117493, 264.5117493

Time for backsubstitution: 1.07 seconds

### NS candidates at layer 1
type: A, layer: 1, pos: 35
type: B, layer: 1, pos: 35
type: B, layer: 1, pos: 9
type: A, layer: 1, pos: 9
type: A, layer: 1, pos: 19
type: B, layer: 1, pos: 19
type: B, layer: 1, pos: 3
type: A, layer: 1, pos: 3
type: B, layer: 1, pos: 17
type: A, layer: 1, pos: 17
type: A, layer: 1, pos: 10
type: B, layer: 1, pos: 10
type: A, layer: 1, pos: 11
type: B, layer: 1, pos: 11
type: B, layer: 1, pos: 18
type: A, layer: 1, pos: 18
type: B, layer: 1, pos: 45
type: A, layer: 1, pos: 15
type: B, layer: 1, pos: 15
type: A, layer: 1, pos: 45
type: B, layer: 1, pos: 43
type: A, layer: 1, pos: 43
type: B, layer: 1, pos: 29
type: A, layer: 1, pos: 29
type: A, layer: 1, pos: 21
type: B, layer: 1, pos: 21
type: A, layer: 1, pos: 23
type: B, layer: 1, pos: 23
type: A, layer: 1, pos: 32
type: B, layer: 1, pos: 37
type: B, layer: 1, pos: 32
type: A, layer: 1, pos: 37
type: B, layer: 1, pos: 20
type: A, layer: 1, pos: 20
type: B, layer: 1, pos: 39
type: A, layer: 1, pos: 39
type: A, layer: 1, pos: 41
type: B, layer: 1, pos: 41
type: B, layer: 1, pos: 38
type: A, layer: 1, pos: 38
type: A, layer: 1, pos: 16
type: B, layer: 1, pos: 16
type: B, layer: 1, pos: 5
type: A, layer: 1, pos: 5
type: B, layer: 1, pos: 4
type: A, layer: 1, pos: 4

Time for candidate selection: 0.08 seconds

### Candidate
type: A, layer: 1, pos: 35

## Relational analysis of NS_B2_A2_A2_B2_A1
Optimization infeasible because this subproblem isn't reachable.

### Candidate
type: B, layer: 1, pos: 35

### Candidate
type: B, layer: 1, pos: 9

## Relational analysis of NS_B2_A2_A2_B2_B1
Optimization infeasible because this subproblem isn't reachable.

### Candidate
type: A, layer: 1, pos: 9

### Candidate
type: A, layer: 1, pos: 19

## Relational analysis of NS_B2_A2_A2_B2_A1
Optimization infeasible because this subproblem isn't reachable.

### Candidate
type: B, layer: 1, pos: 19

### Candidate
type: B, layer: 1, pos: 3

## Relational analysis of NS_B2_A2_A2_B2_B1
Optimization infeasible because this subproblem isn't reachable.

### Candidate
type: A, layer: 1, pos: 3

### Candidate
type: B, layer: 1, pos: 17

## Relational analysis of NS_B2_A2_A2_B2_B1
Optimization infeasible because this subproblem isn't reachable.

### Candidate
type: A, layer: 1, pos: 17

### Candidate
type: A, layer: 1, pos: 10

## Relational analysis of NS_B2_A2_A2_B2_A1
Optimization infeasible because this subproblem isn't reachable.

### Candidate
type: B, layer: 1, pos: 10

### Candidate
type: A, layer: 1, pos: 11

## Relational analysis of NS_B2_A2_A2_B2_A1

### Relational analysis result of NS_B2_A2_A2_B2_A1
Status: Status.UNKNOWN
Output dim: 0, lower bound: -63.3930898, upper bound: 63.3933078
time: 0.78 seconds

## Relational analysis of NS_B2_A2_A2_B2_A2

### Relational analysis result of NS_B2_A2_A2_B2_A2
Status: Status.UNKNOWN
Output dim: 0, lower bound: -63.3935402, upper bound: 63.3948172
time: 0.75 seconds

## Summary of splitting at layer (split count: 4)
- Time for NS candidates: 4.70 seconds
NS_B1_B1_A1_A1_B1, status: Status.UNKNOWN, split count: 5, time: 4.70
Output dim: 0, lower bound: -63.3958876, upper bound: 63.3958122
NS_B1_B1_A1_A1_B2, status: Status.UNKNOWN, split count: 5, time: 4.70
Output dim: 0, lower bound: -63.3958450, upper bound: 63.3957590
NS_B1_B1_A1_A2_B1, status: Status.UNKNOWN, split count: 5, time: 4.70
Output dim: 0, lower bound: -63.3958876, upper bound: 63.3958122
NS_B1_B1_A1_A2_B2, status: Status.UNKNOWN, split count: 5, time: 4.70
Output dim: 0, lower bound: -63.3958450, upper bound: 63.3957590
NS_B1_B1_A2_B1_B1, status: Status.UNKNOWN, split count: 5, time: 4.70
Output dim: 0, lower bound: -63.3943110, upper bound: 63.3942986
NS_B1_B1_A2_B1_B2, status: Status.UNKNOWN, split count: 5, time: 4.70
Output dim: 0, lower bound: -63.3941184, upper bound: 63.3941425
NS_B1_B1_A2_B2_B1, status: Status.UNKNOWN, split count: 5, time: 4.70
Output dim: 0, lower bound: -63.3942807, upper bound: 63.3943366
NS_B1_B1_A2_B2_B2, status: Status.UNKNOWN, split count: 5, time: 4.70
Output dim: 0, lower bound: -63.3942638, upper bound: 63.3943223
NS_B1_B2_A1_A1_A1, status: Status.UNKNOWN, split count: 5, time: 4.70
Output dim: 0, lower bound: -63.3957216, upper bound: 63.3958435
NS_B1_B2_A1_A1_A2, status: Status.UNKNOWN, split count: 5, time: 4.70
Output dim: 0, lower bound: -63.3954598, upper bound: 63.3954598
NS_B1_B2_A1_A2_B1, status: Status.UNKNOWN, split count: 5, time: 4.70
Output dim: 0, lower bound: -63.3958435, upper bound: 63.3957216
NS_B1_B2_A1_A2_B2, status: Status.UNKNOWN, split count: 5, time: 4.70
Output dim: 0, lower bound: -63.3954598, upper bound: 63.3954598
NS_B1_B2_A2_B1_B1, status: Status.UNKNOWN, split count: 5, time: 4.70
Output dim: 0, lower bound: -63.3939212, upper bound: 63.3941125
NS_B1_B2_A2_B1_B2, status: Status.UNKNOWN, split count: 5, time: 4.70
Output dim: 0, lower bound: -63.3936884, upper bound: 63.3935634
NS_B1_B2_A2_B2_B1, status: Status.UNKNOWN, split count: 5, time: 4.70
Output dim: 0, lower bound: -63.3935961, upper bound: 63.3942849
NS_B1_B2_A2_B2_B2, status: Status.UNKNOWN, split count: 5, time: 4.70
Output dim: 0, lower bound: -63.3941813, upper bound: 63.3942861
NS_B2_A1_A1_B1_A1, status: Status.UNKNOWN, split count: 5, time: 4.70
Output dim: 0, lower bound: -63.3942734, upper bound: 63.3945363
NS_B2_A1_A1_B1_A2, status: Status.UNKNOWN, split count: 5, time: 4.70
Output dim: 0, lower bound: -63.3941174, upper bound: 63.3943349
NS_B2_A1_A1_B2_A1, status: Status.UNKNOWN, split count: 5, time: 4.70
Output dim: 0, lower bound: -63.3942734, upper bound: 63.3945955
NS_B2_A1_A1_B2_A2, status: Status.UNKNOWN, split count: 5, time: 4.70
Output dim: 0, lower bound: -63.3941174, upper bound: 63.3945137
NS_B2_A1_A2_B1_A1, status: Status.UNKNOWN, split count: 5, time: 4.70
Output dim: 0, lower bound: -63.3930077, upper bound: 63.3930551
NS_B2_A1_A2_B1_A2, status: Status.UNKNOWN, split count: 5, time: 4.70
Output dim: 0, lower bound: -63.3931305, upper bound: 63.3931305
NS_B2_A1_A2_B2_B1, status: Status.UNKNOWN, split count: 5, time: 4.70
Output dim: 0, lower bound: -63.3930551, upper bound: 63.3930898
NS_B2_A1_A2_B2_B2, status: Status.UNKNOWN, split count: 5, time: 4.70
Output dim: 0, lower bound: -63.3931305, upper bound: 63.3935402
NS_B2_A2_A1_A1_B1, status: Status.UNKNOWN, split count: 5, time: 4.70
Output dim: 0, lower bound: -63.3942459, upper bound: 63.3945203
NS_B2_A2_A1_A1_B2, status: Status.UNKNOWN, split count: 5, time: 4.70
Output dim: 0, lower bound: -63.3942459, upper bound: 63.3952135
NS_B2_A2_A1_A2_B1, status: Status.UNKNOWN, split count: 5, time: 4.70
Output dim: 0, lower bound: -63.3940284, upper bound: 63.3944068
NS_B2_A2_A1_A2_B2, status: Status.UNKNOWN, split count: 5, time: 4.70
Output dim: 0, lower bound: -63.3940284, upper bound: 63.3949890
NS_B2_A2_A2_B1_A1, status: Status.UNKNOWN, split count: 5, time: 4.70
Output dim: 0, lower bound: -63.3930898, upper bound: 63.3931718
NS_B2_A2_A2_B1_A2, status: Status.UNKNOWN, split count: 5, time: 4.70
Output dim: 0, lower bound: -63.3935402, upper bound: 63.3938446
NS_B2_A2_A2_B2_A1, status: Status.UNKNOWN, split count: 5, time: 4.70
Output dim: 0, lower bound: -63.3930898, upper bound: 63.3933078
NS_B2_A2_A2_B2_A2, status: Status.UNKNOWN, split count: 5, time: 4.70
Output dim: 0, lower bound: -63.3935402, upper bound: 63.3948172

## BFS NS instance: NS_B1_B1_A1_A1_B1

### Backsubstitution after applying NS history:
0: -32.1090889, 33.6279716, -31.2985306, 32.7712555, -64.8803406, 64.9264984
1: -227.5665436, 79.5726700, -221.7436371, 77.5785599, -305.1451111, 301.3163147
2: -127.1797180, 72.6204758, -123.9428406, 70.7823410, -197.9620667, 196.5633240
3: -159.0883789, 58.2320824, -155.0498047, 56.7560463, -215.8444214, 213.2818909
4: -91.5803223, 63.3711128, -89.2801895, 61.7543640, -153.3346710, 152.6512909

Time for backsubstitution: 0.98 seconds

### NS candidates at layer 1
type: B, layer: 1, pos: 9
type: A, layer: 1, pos: 9
type: A, layer: 1, pos: 4
type: A, layer: 1, pos: 35
type: B, layer: 1, pos: 35
type: B, layer: 1, pos: 17
type: A, layer: 1, pos: 17
type: A, layer: 1, pos: 19
type: B, layer: 1, pos: 19
type: A, layer: 1, pos: 3
type: B, layer: 1, pos: 45
type: A, layer: 1, pos: 45
type: B, layer: 1, pos: 3
type: B, layer: 1, pos: 11
type: A, layer: 1, pos: 11
type: B, layer: 1, pos: 10
type: A, layer: 1, pos: 10
type: A, layer: 1, pos: 15
type: B, layer: 1, pos: 15
type: A, layer: 1, pos: 18
type: B, layer: 1, pos: 18
type: B, layer: 1, pos: 32
type: A, layer: 1, pos: 39
type: A, layer: 1, pos: 38
type: B, layer: 1, pos: 39
type: A, layer: 1, pos: 43
type: A, layer: 1, pos: 32
type: A, layer: 1, pos: 37
type: B, layer: 1, pos: 43
type: A, layer: 1, pos: 29
type: B, layer: 1, pos: 37
type: B, layer: 1, pos: 38
type: B, layer: 1, pos: 41
type: A, layer: 1, pos: 41
type: B, layer: 1, pos: 29
type: B, layer: 1, pos: 20
type: A, layer: 1, pos: 20
type: B, layer: 1, pos: 21
type: B, layer: 1, pos: 5
type: A, layer: 1, pos: 5
type: B, layer: 1, pos: 23
type: A, layer: 1, pos: 23
type: A, layer: 1, pos: 21
type: B, layer: 1, pos: 16
type: A, layer: 1, pos: 16

Time for candidate selection: 0.08 seconds

### Candidate
type: B, layer: 1, pos: 9

## Relational analysis of NS_B1_B1_A1_A1_B1_B1
Optimization infeasible because this subproblem isn't reachable.

### Candidate
type: A, layer: 1, pos: 9

### Candidate
type: A, layer: 1, pos: 4

## Relational analysis of NS_B1_B1_A1_A1_B1_A1

### Relational analysis result of NS_B1_B1_A1_A1_B1_A1
Status: Status.UNKNOWN
Output dim: 0, lower bound: -63.3963898, upper bound: 63.3963898
time: 0.75 seconds

## Relational analysis of NS_B1_B1_A1_A1_B1_A2

### Relational analysis result of NS_B1_B1_A1_A1_B1_A2
Status: Status.UNKNOWN
Output dim: 0, lower bound: -63.3963898, upper bound: 63.3963898
time: 0.72 seconds

## BFS NS instance: NS_B1_B1_A1_A1_B2

### Backsubstitution after applying NS history:
0: -32.0274887, 33.4556084, -43.3717232, 44.9401779, -76.9676666, 76.8273315
1: -226.3287659, 79.3363266, -303.9672852, 107.6393127, -333.9680481, 382.7997131
2: -126.7314911, 72.2876282, -170.4709625, 98.0167847, -224.7482758, 242.0802002
3: -158.3087463, 57.9702454, -213.3660736, 78.5642319, -236.8729858, 269.9973755
4: -91.3053436, 63.0636482, -123.1605453, 85.3021240, -176.6074677, 185.8313141

Time for backsubstitution: 0.92 seconds

### NS candidates at layer 1
type: A, layer: 1, pos: 4
type: A, layer: 1, pos: 29
type: B, layer: 1, pos: 9
type: A, layer: 1, pos: 38
type: A, layer: 1, pos: 9
type: A, layer: 1, pos: 17
type: B, layer: 1, pos: 35
type: A, layer: 1, pos: 3
type: A, layer: 1, pos: 35
type: B, layer: 1, pos: 19
type: B, layer: 1, pos: 45
type: A, layer: 1, pos: 10
type: A, layer: 1, pos: 19
type: B, layer: 1, pos: 11
type: A, layer: 1, pos: 11
type: A, layer: 1, pos: 45
type: B, layer: 1, pos: 17
type: B, layer: 1, pos: 18
type: B, layer: 1, pos: 15
type: B, layer: 1, pos: 43
type: B, layer: 1, pos: 39
type: B, layer: 1, pos: 3
type: A, layer: 1, pos: 37
type: B, layer: 1, pos: 10
type: A, layer: 1, pos: 15
type: B, layer: 1, pos: 32
type: B, layer: 1, pos: 23
type: A, layer: 1, pos: 18
type: A, layer: 1, pos: 32
type: B, layer: 1, pos: 41
type: B, layer: 1, pos: 16
type: B, layer: 1, pos: 5
type: A, layer: 1, pos: 20
type: A, layer: 1, pos: 41
type: A, layer: 1, pos: 39
type: B, layer: 1, pos: 21
type: A, layer: 1, pos: 21
type: B, layer: 1, pos: 20
type: A, layer: 1, pos: 5
type: B, layer: 1, pos: 37
type: A, layer: 1, pos: 43
type: A, layer: 1, pos: 16
type: A, layer: 1, pos: 23
type: B, layer: 1, pos: 38
type: B, layer: 1, pos: 29

Time for candidate selection: 0.07 seconds

### Candidate
type: A, layer: 1, pos: 4

## Relational analysis of NS_B1_B1_A1_A1_B2_A1

### Relational analysis result of NS_B1_B1_A1_A1_B2_A1
Status: Status.UNKNOWN
Output dim: 0, lower bound: -63.3963898, upper bound: 63.3963898
time: 0.76 seconds

## Relational analysis of NS_B1_B1_A1_A1_B2_A2

### Relational analysis result of NS_B1_B1_A1_A1_B2_A2
Status: Status.UNKNOWN
Output dim: 0, lower bound: -63.3963898, upper bound: 63.3963898
time: 0.71 seconds

## BFS NS instance: NS_B1_B1_A1_A2_B1

### Backsubstitution after applying NS history:
0: -31.2985344, 32.9879265, -31.2985306, 32.7712555, -64.0697937, 64.2864456
1: -223.0827179, 77.4370422, -221.7436371, 77.5785599, -300.6612854, 299.1806030
2: -123.5101624, 71.1551437, -123.9428406, 70.7823410, -194.2925110, 195.0979919
3: -155.0326538, 57.0326843, -155.0498047, 56.7560463, -211.7886810, 212.0824738
4: -88.9413681, 62.3415108, -89.2801895, 61.7543640, -150.6957397, 151.6217041

Time for backsubstitution: 1.03 seconds

### NS candidates at layer 1
type: B, layer: 1, pos: 9
type: A, layer: 1, pos: 9
type: A, layer: 1, pos: 35
type: B, layer: 1, pos: 45
type: B, layer: 1, pos: 35
type: B, layer: 1, pos: 17
type: B, layer: 1, pos: 19
type: B, layer: 1, pos: 3
type: A, layer: 1, pos: 17
type: A, layer: 1, pos: 19
type: A, layer: 1, pos: 4
type: A, layer: 1, pos: 3
type: A, layer: 1, pos: 11
type: B, layer: 1, pos: 11
type: A, layer: 1, pos: 10
type: A, layer: 1, pos: 45
type: B, layer: 1, pos: 10
type: A, layer: 1, pos: 15
type: B, layer: 1, pos: 18
type: B, layer: 1, pos: 15
type: A, layer: 1, pos: 18
type: A, layer: 1, pos: 39
type: B, layer: 1, pos: 38
type: B, layer: 1, pos: 32
type: A, layer: 1, pos: 32
type: B, layer: 1, pos: 43
type: B, layer: 1, pos: 39
type: A, layer: 1, pos: 43
type: B, layer: 1, pos: 37
type: A, layer: 1, pos: 29
type: A, layer: 1, pos: 37
type: B, layer: 1, pos: 41
type: B, layer: 1, pos: 20
type: A, layer: 1, pos: 38
type: A, layer: 1, pos: 41
type: B, layer: 1, pos: 29
type: A, layer: 1, pos: 23
type: B, layer: 1, pos: 5
type: B, layer: 1, pos: 21
type: A, layer: 1, pos: 20
type: A, layer: 1, pos: 21
type: A, layer: 1, pos: 16
type: A, layer: 1, pos: 5
type: B, layer: 1, pos: 23
type: B, layer: 1, pos: 16

Time for candidate selection: 0.09 seconds

### Candidate
type: B, layer: 1, pos: 9

## Relational analysis of NS_B1_B1_A1_A2_B1_B1
Optimization infeasible because this subproblem isn't reachable.

### Candidate
type: A, layer: 1, pos: 9

### Candidate
type: A, layer: 1, pos: 35

## Relational analysis of NS_B1_B1_A1_A2_B1_A1
Optimization infeasible because this subproblem isn't reachable.

### Candidate
type: B, layer: 1, pos: 45

## Relational analysis of NS_B1_B1_A1_A2_B1_B1
Optimization infeasible because this subproblem isn't reachable.

### Candidate
type: B, layer: 1, pos: 35

### Candidate
type: B, layer: 1, pos: 17

## Relational analysis of NS_B1_B1_A1_A2_B1_B1
Optimization infeasible because this subproblem isn't reachable.

### Candidate
type: B, layer: 1, pos: 19

## Relational analysis of NS_B1_B1_A1_A2_B1_B1
Optimization infeasible because this subproblem isn't reachable.

### Candidate
type: B, layer: 1, pos: 3

## Relational analysis of NS_B1_B1_A1_A2_B1_B1

### Relational analysis result of NS_B1_B1_A1_A2_B1_B1
Status: Status.UNKNOWN
Output dim: 0, lower bound: -63.3958702, upper bound: 63.3958122
time: 0.68 seconds

## Relational analysis of NS_B1_B1_A1_A2_B1_B2

### Relational analysis result of NS_B1_B1_A1_A2_B1_B2
Status: Status.UNKNOWN
Output dim: 0, lower bound: -63.3958754, upper bound: 63.3958122
time: 0.75 seconds

## BFS NS instance: NS_B1_B1_A1_A2_B2

### Backsubstitution after applying NS history:
0: -31.9320335, 33.5925255, -43.3717232, 44.9401779, -76.8722076, 76.9642334
1: -227.7259827, 79.0248489, -303.9672852, 107.6393127, -335.3652649, 382.9920654
2: -126.1115570, 72.5627213, -170.4709625, 98.0167847, -224.1283264, 242.5304260
3: -158.2653961, 58.1356277, -213.3660736, 78.5642319, -236.8296204, 270.2504578
4: -90.7752762, 63.4691505, -123.1605453, 85.3021240, -176.0773773, 186.3454742

Time for backsubstitution: 0.92 seconds

### NS candidates at layer 1
type: A, layer: 1, pos: 4
type: A, layer: 1, pos: 29
type: B, layer: 1, pos: 9
type: A, layer: 1, pos: 35
type: A, layer: 1, pos: 17
type: A, layer: 1, pos: 9
type: B, layer: 1, pos: 45
type: B, layer: 1, pos: 35
type: B, layer: 1, pos: 19
type: A, layer: 1, pos: 3
type: A, layer: 1, pos: 38
type: A, layer: 1, pos: 10
type: A, layer: 1, pos: 19
type: A, layer: 1, pos: 11
type: B, layer: 1, pos: 11
type: B, layer: 1, pos: 18
type: B, layer: 1, pos: 17
type: A, layer: 1, pos: 45
type: B, layer: 1, pos: 43
type: B, layer: 1, pos: 3
type: B, layer: 1, pos: 15
type: A, layer: 1, pos: 15
type: B, layer: 1, pos: 39
type: B, layer: 1, pos: 10
type: A, layer: 1, pos: 37
type: A, layer: 1, pos: 32
type: B, layer: 1, pos: 32
type: B, layer: 1, pos: 23
type: B, layer: 1, pos: 41
type: B, layer: 1, pos: 5
type: A, layer: 1, pos: 20
type: A, layer: 1, pos: 18
type: A, layer: 1, pos: 39
type: B, layer: 1, pos: 16
type: A, layer: 1, pos: 41
type: B, layer: 1, pos: 21
type: A, layer: 1, pos: 21
type: B, layer: 1, pos: 37
type: B, layer: 1, pos: 20
type: A, layer: 1, pos: 5
type: A, layer: 1, pos: 23
type: A, layer: 1, pos: 16
type: A, layer: 1, pos: 43
type: B, layer: 1, pos: 38
type: B, layer: 1, pos: 29

Time for candidate selection: 0.07 seconds

### Candidate
type: A, layer: 1, pos: 4

## Relational analysis of NS_B1_B1_A1_A2_B2_A1

### Relational analysis result of NS_B1_B1_A1_A2_B2_A1
Status: Status.UNKNOWN
Output dim: 0, lower bound: -63.3958450, upper bound: 63.3957590
time: 0.68 seconds

## Relational analysis of NS_B1_B1_A1_A2_B2_A2

### Relational analysis result of NS_B1_B1_A1_A2_B2_A2
Status: Status.UNKNOWN
Output dim: 0, lower bound: -63.3958450, upper bound: 63.3957590
time: 0.74 seconds

## BFS NS instance: NS_B1_B1_A2_B1_B1

### Backsubstitution after applying NS history:
0: -53.7749062, 58.3826141, -30.8998337, 32.3675804, -86.1424866, 89.2824326
1: -397.2376709, 132.6823425, -219.2346802, 76.5160141, -472.7992554, 351.9169617
2: -212.4016876, 125.1452332, -122.6432190, 69.8588181, -282.2604980, 247.7884521
3: -274.0435181, 100.2597046, -153.3082123, 56.0272484, -329.8058472, 253.5678711
4: -152.4556427, 110.6469498, -88.1771545, 60.9467621, -213.4023895, 198.8240967

Time for backsubstitution: 0.92 seconds

### NS candidates at layer 1
type: B, layer: 1, pos: 4
type: B, layer: 1, pos: 38
type: B, layer: 1, pos: 9
type: A, layer: 1, pos: 35
type: B, layer: 1, pos: 3
type: B, layer: 1, pos: 17
type: A, layer: 1, pos: 7
type: B, layer: 1, pos: 45
type: A, layer: 1, pos: 9
type: A, layer: 1, pos: 11
type: A, layer: 1, pos: 15
type: A, layer: 1, pos: 10
type: B, layer: 1, pos: 11
type: B, layer: 1, pos: 10
type: B, layer: 1, pos: 35
type: A, layer: 1, pos: 16
type: A, layer: 1, pos: 39
type: B, layer: 1, pos: 37
type: B, layer: 1, pos: 18
type: A, layer: 1, pos: 17
type: A, layer: 1, pos: 3
type: A, layer: 1, pos: 19
type: A, layer: 1, pos: 43
type: B, layer: 1, pos: 32
type: A, layer: 1, pos: 18
type: A, layer: 1, pos: 32
type: A, layer: 1, pos: 23
type: A, layer: 1, pos: 45
type: B, layer: 1, pos: 21
type: B, layer: 1, pos: 41
type: B, layer: 1, pos: 15
type: B, layer: 1, pos: 5
type: A, layer: 1, pos: 41
type: A, layer: 1, pos: 20
type: B, layer: 1, pos: 39
type: B, layer: 1, pos: 20
type: A, layer: 1, pos: 5
type: A, layer: 1, pos: 21
type: B, layer: 1, pos: 43
type: B, layer: 1, pos: 23
type: B, layer: 1, pos: 16
type: A, layer: 1, pos: 37
type: A, layer: 1, pos: 38
type: A, layer: 1, pos: 4
type: A, layer: 1, pos: 29

Time for candidate selection: 0.07 seconds

### Candidate
type: B, layer: 1, pos: 4

### Candidate
type: B, layer: 1, pos: 38

### Candidate
type: B, layer: 1, pos: 9

### Candidate
type: A, layer: 1, pos: 35

### Candidate
type: B, layer: 1, pos: 3

### Candidate
type: B, layer: 1, pos: 17

### Candidate
type: A, layer: 1, pos: 7

## Relational analysis of NS_B1_B1_A2_B1_B1_A1

### Relational analysis result of NS_B1_B1_A2_B1_B1_A1
Status: Status.UNKNOWN
Output dim: 0, lower bound: -63.3942257, upper bound: 63.3940786
time: 0.66 seconds

## Relational analysis of NS_B1_B1_A2_B1_B1_A2

### Relational analysis result of NS_B1_B1_A2_B1_B1_A2
Status: Status.UNKNOWN
Output dim: 0, lower bound: -63.3942257, upper bound: 63.3942986
time: 0.72 seconds

## BFS NS instance: NS_B1_B1_A2_B1_B2

### Backsubstitution after applying NS history:
0: -53.2943115, 57.7648315, -40.1849136, 41.3175163, -94.6118317, 97.8713074
1: -391.6769714, 131.4059601, -270.5454102, 99.4554672, -489.2981567, 401.9513550
2: -209.9710083, 123.7719040, -155.8203583, 89.4233932, -299.3944092, 278.8367004
3: -270.4146118, 99.1839218, -190.4992371, 71.5364838, -341.6929932, 289.5463257
4: -150.9994507, 109.5045471, -113.9260864, 78.2623062, -229.2617340, 222.4403229

Time for backsubstitution: 1.07 seconds

### NS candidates at layer 1
type: B, layer: 1, pos: 4
type: B, layer: 1, pos: 9
type: A, layer: 1, pos: 35
type: A, layer: 1, pos: 7
type: B, layer: 1, pos: 3
type: B, layer: 1, pos: 17
type: A, layer: 1, pos: 9
type: B, layer: 1, pos: 38
type: A, layer: 1, pos: 10
type: B, layer: 1, pos: 45
type: B, layer: 1, pos: 35
type: A, layer: 1, pos: 11
type: A, layer: 1, pos: 15
type: B, layer: 1, pos: 18
type: B, layer: 1, pos: 11
type: A, layer: 1, pos: 17
type: A, layer: 1, pos: 29
type: A, layer: 1, pos: 16
type: A, layer: 1, pos: 3
type: A, layer: 1, pos: 32
type: B, layer: 1, pos: 37
type: B, layer: 1, pos: 10
type: A, layer: 1, pos: 4
type: A, layer: 1, pos: 39
type: A, layer: 1, pos: 19
type: A, layer: 1, pos: 45
type: B, layer: 1, pos: 43
type: A, layer: 1, pos: 20
type: B, layer: 1, pos: 21
type: B, layer: 1, pos: 41
type: B, layer: 1, pos: 15
type: B, layer: 1, pos: 39
type: A, layer: 1, pos: 18
type: B, layer: 1, pos: 32
type: A, layer: 1, pos: 23
type: B, layer: 1, pos: 5
type: A, layer: 1, pos: 41
type: A, layer: 1, pos: 43
type: A, layer: 1, pos: 5
type: A, layer: 1, pos: 21
type: B, layer: 1, pos: 23
type: B, layer: 1, pos: 20
type: A, layer: 1, pos: 37
type: B, layer: 1, pos: 16
type: A, layer: 1, pos: 38

Time for candidate selection: 0.07 seconds

### Candidate
type: B, layer: 1, pos: 4

### Candidate
type: B, layer: 1, pos: 9

### Candidate
type: A, layer: 1, pos: 35

### Candidate
type: A, layer: 1, pos: 7

## Relational analysis of NS_B1_B1_A2_B1_B2_A1

### Relational analysis result of NS_B1_B1_A2_B1_B2_A1
Status: Status.UNKNOWN
Output dim: 0, lower bound: -63.3939945, upper bound: 63.3939685
time: 0.73 seconds

## Relational analysis of NS_B1_B1_A2_B1_B2_A2

### Relational analysis result of NS_B1_B1_A2_B1_B2_A2
Status: Status.UNKNOWN
Output dim: 0, lower bound: -63.3939945, upper bound: 63.3941425
time: 0.65 seconds

## BFS NS instance: NS_B1_B1_A2_B2_B1

### Backsubstitution after applying NS history:
0: -53.7296906, 58.2884941, -33.7412872, 35.4420433, -89.1717377, 92.0297852
1: -396.4836731, 132.5462341, -241.4537048, 83.6052399, -479.1410217, 373.9999390
2: -212.1580200, 124.9503784, -134.5909882, 76.4633179, -288.6213074, 259.5413513
3: -273.6259460, 100.1009369, -168.7952271, 61.3642120, -334.7517395, 268.8961182
4: -152.3504181, 110.4550476, -96.5188828, 66.6376648, -218.9880829, 206.9739227

Time for backsubstitution: 0.92 seconds

### NS candidates at layer 1
type: B, layer: 1, pos: 4
type: B, layer: 1, pos: 38
type: B, layer: 1, pos: 3
type: A, layer: 1, pos: 35
type: B, layer: 1, pos: 9
type: B, layer: 1, pos: 17
type: A, layer: 1, pos: 7
type: B, layer: 1, pos: 45
type: A, layer: 1, pos: 9
type: A, layer: 1, pos: 11
type: B, layer: 1, pos: 11
type: A, layer: 1, pos: 15
type: A, layer: 1, pos: 10
type: B, layer: 1, pos: 35
type: B, layer: 1, pos: 10
type: A, layer: 1, pos: 39
type: A, layer: 1, pos: 3
type: B, layer: 1, pos: 18
type: A, layer: 1, pos: 17
type: A, layer: 1, pos: 16
type: B, layer: 1, pos: 37
type: A, layer: 1, pos: 45
type: A, layer: 1, pos: 32
type: A, layer: 1, pos: 23
type: B, layer: 1, pos: 15
type: A, layer: 1, pos: 43
type: A, layer: 1, pos: 18
type: A, layer: 1, pos: 19
type: B, layer: 1, pos: 41
type: B, layer: 1, pos: 32
type: A, layer: 1, pos: 41
type: B, layer: 1, pos: 21
type: B, layer: 1, pos: 5
type: A, layer: 1, pos: 20
type: B, layer: 1, pos: 20
type: A, layer: 1, pos: 5
type: A, layer: 1, pos: 21
type: B, layer: 1, pos: 39
type: B, layer: 1, pos: 43
type: B, layer: 1, pos: 23
type: B, layer: 1, pos: 16
type: A, layer: 1, pos: 37
type: A, layer: 1, pos: 4
type: A, layer: 1, pos: 38
type: A, layer: 1, pos: 29

Time for candidate selection: 0.07 seconds

### Candidate
type: B, layer: 1, pos: 4

### Candidate
type: B, layer: 1, pos: 38

### Candidate
type: B, layer: 1, pos: 3

### Candidate
type: A, layer: 1, pos: 35

### Candidate
type: B, layer: 1, pos: 9

### Candidate
type: B, layer: 1, pos: 17

## Relational analysis of NS_B1_B1_A2_B2_B1_B1
Optimization infeasible because this subproblem isn't reachable.

### Candidate
type: A, layer: 1, pos: 7

## Relational analysis of NS_B1_B1_A2_B2_B1_A1

### Relational analysis result of NS_B1_B1_A2_B2_B1_A1
Status: Status.UNKNOWN
Output dim: 0, lower bound: -63.3940763, upper bound: 63.3939509
time: 0.77 seconds

## Relational analysis of NS_B1_B1_A2_B2_B1_A2

### Relational analysis result of NS_B1_B1_A2_B2_B1_A2
Status: Status.UNKNOWN
Output dim: 0, lower bound: -63.3940763, upper bound: 63.3943366
time: 0.72 seconds

## BFS NS instance: NS_B1_B1_A2_B2_B2

### Backsubstitution after applying NS history:
0: -53.1750488, 57.5974617, -43.8191643, 45.3083954, -98.4834442, 101.2977448
1: -390.4071960, 131.1018524, -299.1305847, 108.4494934, -496.9378357, 430.2324219
2: -209.4720612, 123.4078064, -171.1938019, 97.9577637, -307.4298096, 293.4460449
3: -269.6227417, 98.9051590, -210.4305420, 78.3455505, -347.7146912, 308.6618347
4: -150.6838074, 109.1703568, -124.5197067, 85.6121521, -236.2959595, 232.3986816

Time for backsubstitution: 0.98 seconds

### NS candidates at layer 1
type: B, layer: 1, pos: 4
type: B, layer: 1, pos: 9
type: A, layer: 1, pos: 35
type: A, layer: 1, pos: 7
type: B, layer: 1, pos: 3
type: A, layer: 1, pos: 9
type: B, layer: 1, pos: 17
type: A, layer: 1, pos: 10
type: B, layer: 1, pos: 38
type: B, layer: 1, pos: 35
type: B, layer: 1, pos: 45
type: B, layer: 1, pos: 18
type: A, layer: 1, pos: 11
type: B, layer: 1, pos: 11
type: A, layer: 1, pos: 15
type: A, layer: 1, pos: 4
type: A, layer: 1, pos: 17
type: A, layer: 1, pos: 3
type: A, layer: 1, pos: 29
type: B, layer: 1, pos: 37
type: B, layer: 1, pos: 10
type: A, layer: 1, pos: 32
type: A, layer: 1, pos: 39
type: A, layer: 1, pos: 45
type: B, layer: 1, pos: 43
type: A, layer: 1, pos: 16
type: A, layer: 1, pos: 19
type: B, layer: 1, pos: 15
type: A, layer: 1, pos: 20
type: B, layer: 1, pos: 39
type: B, layer: 1, pos: 41
type: B, layer: 1, pos: 21
type: B, layer: 1, pos: 32
type: A, layer: 1, pos: 23
type: A, layer: 1, pos: 18
type: A, layer: 1, pos: 41
type: B, layer: 1, pos: 5
type: A, layer: 1, pos: 21
type: A, layer: 1, pos: 5
type: B, layer: 1, pos: 23
type: B, layer: 1, pos: 20
type: A, layer: 1, pos: 43
type: A, layer: 1, pos: 37
type: B, layer: 1, pos: 16
type: A, layer: 1, pos: 38

Time for candidate selection: 0.07 seconds

### Candidate
type: B, layer: 1, pos: 4

### Candidate
type: B, layer: 1, pos: 9

### Candidate
type: A, layer: 1, pos: 35

### Candidate
type: A, layer: 1, pos: 7

## Relational analysis of NS_B1_B1_A2_B2_B2_A1

### Relational analysis result of NS_B1_B1_A2_B2_B2_A1
Status: Status.UNKNOWN
Output dim: 0, lower bound: -63.3939964, upper bound: 63.3939342
time: 0.69 seconds

## Relational analysis of NS_B1_B1_A2_B2_B2_A2

### Relational analysis result of NS_B1_B1_A2_B2_B2_A2
Status: Status.UNKNOWN
Output dim: 0, lower bound: -63.3939964, upper bound: 63.3943223
time: 0.70 seconds

## BFS NS instance: NS_B1_B2_A1_A1_A1

### Backsubstitution after applying NS history:
0: -31.2985306, 32.7712555, -31.2985344, 32.9879265, -64.2864532, 64.0697937
1: -221.7436371, 77.5785599, -223.0827179, 77.4370422, -299.1806030, 300.6612854
2: -123.9428406, 70.7823410, -123.5101624, 71.1551437, -195.0979919, 194.2925110
3: -155.0498047, 56.7560463, -155.0326538, 57.0326843, -212.0824738, 211.7886810
4: -89.2801895, 61.7543640, -88.9413681, 62.3415108, -151.6217041, 150.6957397

Time for backsubstitution: 0.92 seconds

### NS candidates at layer 1
type: A, layer: 1, pos: 9
type: B, layer: 1, pos: 9
type: B, layer: 1, pos: 35
type: A, layer: 1, pos: 45
type: A, layer: 1, pos: 35
type: A, layer: 1, pos: 17
type: A, layer: 1, pos: 19
type: A, layer: 1, pos: 3
type: B, layer: 1, pos: 17
type: B, layer: 1, pos: 19
type: B, layer: 1, pos: 4
type: B, layer: 1, pos: 3
type: B, layer: 1, pos: 11
type: A, layer: 1, pos: 11
type: B, layer: 1, pos: 10
type: B, layer: 1, pos: 45
type: A, layer: 1, pos: 10
type: B, layer: 1, pos: 15
type: A, layer: 1, pos: 18
type: A, layer: 1, pos: 15
type: B, layer: 1, pos: 18
type: B, layer: 1, pos: 39
type: A, layer: 1, pos: 38
type: A, layer: 1, pos: 32
type: B, layer: 1, pos: 32
type: A, layer: 1, pos: 43
type: A, layer: 1, pos: 39
type: B, layer: 1, pos: 43
type: A, layer: 1, pos: 37
type: B, layer: 1, pos: 29
type: B, layer: 1, pos: 37
type: A, layer: 1, pos: 41
type: A, layer: 1, pos: 20
type: B, layer: 1, pos: 38
type: B, layer: 1, pos: 41
type: A, layer: 1, pos: 29
type: B, layer: 1, pos: 23
type: A, layer: 1, pos: 5
type: A, layer: 1, pos: 21
type: B, layer: 1, pos: 20
type: B, layer: 1, pos: 21
type: B, layer: 1, pos: 16
type: B, layer: 1, pos: 5
type: A, layer: 1, pos: 23
type: A, layer: 1, pos: 16

Time for candidate selection: 0.08 seconds

### Candidate
type: A, layer: 1, pos: 9

## Relational analysis of NS_B1_B2_A1_A1_A1_A1
Optimization infeasible because this subproblem isn't reachable.

### Candidate
type: B, layer: 1, pos: 9

### Candidate
type: B, layer: 1, pos: 35

## Relational analysis of NS_B1_B2_A1_A1_A1_B1
Optimization infeasible because this subproblem isn't reachable.

### Candidate
type: A, layer: 1, pos: 45

## Relational analysis of NS_B1_B2_A1_A1_A1_A1
Optimization infeasible because this subproblem isn't reachable.

### Candidate
type: A, layer: 1, pos: 35

### Candidate
type: A, layer: 1, pos: 17

## Relational analysis of NS_B1_B2_A1_A1_A1_A1
Optimization infeasible because this subproblem isn't reachable.

### Candidate
type: A, layer: 1, pos: 19

## Relational analysis of NS_B1_B2_A1_A1_A1_A1
Optimization infeasible because this subproblem isn't reachable.

### Candidate
type: A, layer: 1, pos: 3

## Relational analysis of NS_B1_B2_A1_A1_A1_A1

### Relational analysis result of NS_B1_B2_A1_A1_A1_A1
Status: Status.UNKNOWN
Output dim: 0, lower bound: -63.3958122, upper bound: 63.3958702
time: 0.69 seconds

## Relational analysis of NS_B1_B2_A1_A1_A1_A2

### Relational analysis result of NS_B1_B2_A1_A1_A1_A2
Status: Status.UNKNOWN
Output dim: 0, lower bound: -63.3958122, upper bound: 63.3958754
time: 0.63 seconds

## BFS NS instance: NS_B1_B2_A1_A1_A2

### Backsubstitution after applying NS history:
0: -43.3717232, 44.9401779, -31.9320335, 33.5925255, -76.9642334, 76.8722076
1: -303.9672852, 107.6393127, -227.7259827, 79.0248489, -382.9920654, 335.3652344
2: -170.4709625, 98.0167847, -126.1115570, 72.5627213, -242.5304260, 224.1283264
3: -213.3660736, 78.5642319, -158.2653961, 58.1356277, -270.2504272, 236.8296204
4: -123.1605453, 85.3021240, -90.7752762, 63.4691505, -186.3454742, 176.0773926

Time for backsubstitution: 0.92 seconds

### NS candidates at layer 1
type: B, layer: 1, pos: 4
type: B, layer: 1, pos: 29
type: A, layer: 1, pos: 9
type: B, layer: 1, pos: 35
type: B, layer: 1, pos: 17
type: B, layer: 1, pos: 9
type: A, layer: 1, pos: 45
type: A, layer: 1, pos: 35
type: A, layer: 1, pos: 19
type: B, layer: 1, pos: 3
type: B, layer: 1, pos: 38
type: B, layer: 1, pos: 10
type: B, layer: 1, pos: 19
type: B, layer: 1, pos: 11
type: A, layer: 1, pos: 11
type: A, layer: 1, pos: 18
type: A, layer: 1, pos: 17
type: B, layer: 1, pos: 45
type: A, layer: 1, pos: 43
type: A, layer: 1, pos: 3
type: A, layer: 1, pos: 15
type: B, layer: 1, pos: 15
type: A, layer: 1, pos: 39
type: A, layer: 1, pos: 10
type: B, layer: 1, pos: 37
type: B, layer: 1, pos: 32
type: A, layer: 1, pos: 32
type: A, layer: 1, pos: 23
type: A, layer: 1, pos: 41
type: A, layer: 1, pos: 5
type: B, layer: 1, pos: 20
type: B, layer: 1, pos: 18
type: B, layer: 1, pos: 39
type: A, layer: 1, pos: 16
type: B, layer: 1, pos: 41
type: A, layer: 1, pos: 21
type: B, layer: 1, pos: 21
type: A, layer: 1, pos: 37
type: A, layer: 1, pos: 20
type: B, layer: 1, pos: 5
type: B, layer: 1, pos: 23
type: B, layer: 1, pos: 16
type: B, layer: 1, pos: 43
type: A, layer: 1, pos: 38
type: A, layer: 1, pos: 29

Time for candidate selection: 0.11 seconds

### Candidate
type: B, layer: 1, pos: 4

## Relational analysis of NS_B1_B2_A1_A1_A2_B1

### Relational analysis result of NS_B1_B2_A1_A1_A2_B1
Status: Status.UNKNOWN
Output dim: 0, lower bound: -63.3957590, upper bound: 63.3958450
time: 0.71 seconds

## Relational analysis of NS_B1_B2_A1_A1_A2_B2

### Relational analysis result of NS_B1_B2_A1_A1_A2_B2
Status: Status.UNKNOWN
Output dim: 0, lower bound: -63.3957590, upper bound: 63.3958450
time: 0.76 seconds

## BFS NS instance: NS_B1_B2_A1_A2_B1

### Backsubstitution after applying NS history:
0: -31.2985344, 32.9879265, -30.1823769, 31.8147984, -63.1133347, 63.1703033
1: -223.0827179, 77.4370422, -214.8651276, 74.6638260, -297.7465515, 292.3021545
2: -123.5101624, 71.1551437, -119.0411377, 68.5830307, -192.0932007, 190.1962891
3: -155.0326538, 57.0326843, -149.3636475, 54.9800568, -210.0127106, 206.3963318
4: -88.9413681, 62.3415108, -85.7541580, 60.1135139, -149.0548859, 148.0956726

Time for backsubstitution: 0.92 seconds

### NS candidates at layer 1
type: B, layer: 1, pos: 9
type: A, layer: 1, pos: 4
type: A, layer: 1, pos: 9
type: A, layer: 1, pos: 35
type: B, layer: 1, pos: 35
type: A, layer: 1, pos: 17
type: B, layer: 1, pos: 17
type: A, layer: 1, pos: 19
type: B, layer: 1, pos: 45
type: B, layer: 1, pos: 19
type: A, layer: 1, pos: 3
type: A, layer: 1, pos: 45
type: B, layer: 1, pos: 3
type: B, layer: 1, pos: 11
type: A, layer: 1, pos: 11
type: B, layer: 1, pos: 10
type: A, layer: 1, pos: 10
type: A, layer: 1, pos: 15
type: A, layer: 1, pos: 18
type: B, layer: 1, pos: 15
type: B, layer: 1, pos: 18
type: B, layer: 1, pos: 32
type: A, layer: 1, pos: 39
type: A, layer: 1, pos: 38
type: B, layer: 1, pos: 39
type: A, layer: 1, pos: 43
type: A, layer: 1, pos: 37
type: B, layer: 1, pos: 43
type: A, layer: 1, pos: 32
type: A, layer: 1, pos: 29
type: B, layer: 1, pos: 37
type: B, layer: 1, pos: 41
type: B, layer: 1, pos: 38
type: A, layer: 1, pos: 41
type: B, layer: 1, pos: 20
type: B, layer: 1, pos: 29
type: B, layer: 1, pos: 21
type: A, layer: 1, pos: 20
type: B, layer: 1, pos: 5
type: A, layer: 1, pos: 23
type: A, layer: 1, pos: 5
type: A, layer: 1, pos: 21
type: B, layer: 1, pos: 23
type: B, layer: 1, pos: 16
type: A, layer: 1, pos: 16

Time for candidate selection: 0.08 seconds

### Candidate
type: B, layer: 1, pos: 9

## Relational analysis of NS_B1_B2_A1_A2_B1_B1

### Relational analysis result of NS_B1_B2_A1_A2_B1_B1
Status: Status.UNKNOWN
Output dim: 0, lower bound: -63.3938505, upper bound: 63.3932262
time: 0.71 seconds

## Relational analysis of NS_B1_B2_A1_A2_B1_B2

### Relational analysis result of NS_B1_B2_A1_A2_B1_B2
Status: Status.UNKNOWN
Output dim: 0, lower bound: -63.3949434, upper bound: 63.3947308
time: 0.92 seconds

## BFS NS instance: NS_B1_B2_A1_A2_B2

### Backsubstitution after applying NS history:
0: -31.9320335, 33.5925255, -44.6238937, 46.3271980, -78.2592316, 78.2164154
1: -227.7259827, 79.0248489, -313.9445801, 110.7814026, -338.5073853, 392.9693604
2: -126.1115570, 72.5627213, -175.7769775, 100.9844360, -227.0959778, 248.3396912
3: -158.2653961, 58.1356277, -220.2172699, 80.9449158, -239.2103119, 278.3528442
4: -90.7752762, 63.4691505, -126.8017273, 87.9186707, -178.6939392, 190.2708740

Time for backsubstitution: 0.96 seconds

### NS candidates at layer 1
type: A, layer: 1, pos: 4
type: A, layer: 1, pos: 29
type: B, layer: 1, pos: 9
type: A, layer: 1, pos: 38
type: A, layer: 1, pos: 35
type: A, layer: 1, pos: 17
type: A, layer: 1, pos: 9
type: B, layer: 1, pos: 35
type: B, layer: 1, pos: 45
type: A, layer: 1, pos: 3
type: B, layer: 1, pos: 19
type: A, layer: 1, pos: 10
type: A, layer: 1, pos: 19
type: B, layer: 1, pos: 11
type: A, layer: 1, pos: 11
type: B, layer: 1, pos: 17
type: A, layer: 1, pos: 45
type: B, layer: 1, pos: 18
type: B, layer: 1, pos: 15
type: B, layer: 1, pos: 3
type: B, layer: 1, pos: 39
type: B, layer: 1, pos: 43
type: B, layer: 1, pos: 10
type: A, layer: 1, pos: 15
type: A, layer: 1, pos: 37
type: B, layer: 1, pos: 32
type: A, layer: 1, pos: 18
type: B, layer: 1, pos: 23
type: A, layer: 1, pos: 32
type: B, layer: 1, pos: 41
type: B, layer: 1, pos: 5
type: A, layer: 1, pos: 20
type: B, layer: 1, pos: 16
type: A, layer: 1, pos: 39
type: A, layer: 1, pos: 41
type: B, layer: 1, pos: 21
type: A, layer: 1, pos: 21
type: B, layer: 1, pos: 20
type: B, layer: 1, pos: 37
type: A, layer: 1, pos: 43
type: A, layer: 1, pos: 5
type: A, layer: 1, pos: 23
type: A, layer: 1, pos: 16
type: B, layer: 1, pos: 38
type: B, layer: 1, pos: 29

Time for candidate selection: 0.08 seconds

### Candidate
type: A, layer: 1, pos: 4

## Relational analysis of NS_B1_B2_A1_A2_B2_A1

### Relational analysis result of NS_B1_B2_A1_A2_B2_A1
Status: Status.UNKNOWN
Output dim: 0, lower bound: -63.3954598, upper bound: 63.3954598
time: 0.71 seconds

## Relational analysis of NS_B1_B2_A1_A2_B2_A2

### Relational analysis result of NS_B1_B2_A1_A2_B2_A2
Status: Status.UNKNOWN
Output dim: 0, lower bound: -63.3954598, upper bound: 63.3954598
time: 0.74 seconds

## BFS NS instance: NS_B1_B2_A2_B1_B1

### Backsubstitution after applying NS history:
0: -52.6585197, 57.3923950, -30.0790844, 31.7514000, -84.4099197, 87.4714737
1: -391.8822021, 130.0857697, -215.8428955, 74.3573303, -466.2395325, 345.9286499
2: -208.4651794, 122.9702530, -118.6634903, 68.4873657, -276.9525452, 241.6337433
3: -269.7965698, 98.6160355, -149.5511017, 54.8825340, -324.6791077, 248.1671448
4: -149.1065216, 108.8269043, -85.3397293, 59.9776192, -209.0841217, 194.1666260

Time for backsubstitution: 0.95 seconds

### NS candidates at layer 1
type: B, layer: 1, pos: 4
type: B, layer: 1, pos: 3
type: A, layer: 1, pos: 4
type: B, layer: 1, pos: 19
type: A, layer: 1, pos: 35
type: B, layer: 1, pos: 35
type: B, layer: 1, pos: 38
type: A, layer: 1, pos: 3
type: A, layer: 1, pos: 19
type: A, layer: 1, pos: 7
type: B, layer: 1, pos: 45
type: A, layer: 1, pos: 9
type: B, layer: 1, pos: 17
type: A, layer: 1, pos: 15
type: A, layer: 1, pos: 43
type: A, layer: 1, pos: 45
type: A, layer: 1, pos: 11
type: B, layer: 1, pos: 21
type: A, layer: 1, pos: 18
type: B, layer: 1, pos: 11
type: A, layer: 1, pos: 23
type: A, layer: 1, pos: 38
type: A, layer: 1, pos: 39
type: B, layer: 1, pos: 10
type: A, layer: 1, pos: 10
type: A, layer: 1, pos: 21
type: B, layer: 1, pos: 18
type: B, layer: 1, pos: 37
type: A, layer: 1, pos: 16
type: B, layer: 1, pos: 23
type: B, layer: 1, pos: 20
type: B, layer: 1, pos: 43
type: B, layer: 1, pos: 15
type: A, layer: 1, pos: 20
type: B, layer: 1, pos: 32
type: A, layer: 1, pos: 32
type: B, layer: 1, pos: 41
type: B, layer: 1, pos: 39
type: A, layer: 1, pos: 41
type: A, layer: 1, pos: 17
type: B, layer: 1, pos: 5
type: A, layer: 1, pos: 5
type: B, layer: 1, pos: 16
type: A, layer: 1, pos: 37
type: A, layer: 1, pos: 29

Time for candidate selection: 0.07 seconds

### Candidate
type: B, layer: 1, pos: 4

### Candidate
type: B, layer: 1, pos: 3

### Candidate
type: A, layer: 1, pos: 4

### Candidate
type: B, layer: 1, pos: 19

## Relational analysis of NS_B1_B2_A2_B1_B1_B1

### Relational analysis result of NS_B1_B2_A2_B1_B1_B1
Status: Status.UNKNOWN
Output dim: 0, lower bound: -63.3936396, upper bound: 63.3940085
time: 0.76 seconds

## Relational analysis of NS_B1_B2_A2_B1_B1_B2

### Relational analysis result of NS_B1_B2_A2_B1_B1_B2
Status: Status.UNKNOWN
Output dim: 0, lower bound: -63.3939212, upper bound: 63.3941125
time: 1.12 seconds

## BFS NS instance: NS_B1_B2_A2_B1_B2

### Backsubstitution after applying NS history:
0: -52.0513229, 56.6567726, -39.6053352, 40.7887802, -92.8401031, 96.2621078
1: -385.4307556, 128.4622650, -267.8989563, 98.0441742, -482.7828369, 396.3611450
2: -205.5552826, 121.3393707, -153.9289856, 88.2534256, -293.8087158, 275.2525635
3: -265.5540466, 97.2910156, -188.3889313, 70.6072540, -336.1613159, 285.6798706
4: -147.1788177, 107.4819183, -112.4031372, 77.2734604, -224.4522705, 219.3874359

Time for backsubstitution: 1.20 seconds

### NS candidates at layer 1
type: B, layer: 1, pos: 4
type: B, layer: 1, pos: 3
type: A, layer: 1, pos: 35
type: B, layer: 1, pos: 19
type: B, layer: 1, pos: 35
type: A, layer: 1, pos: 7
type: B, layer: 1, pos: 17
type: A, layer: 1, pos: 3
type: A, layer: 1, pos: 9
type: B, layer: 1, pos: 38
type: A, layer: 1, pos: 19
type: A, layer: 1, pos: 10
type: A, layer: 1, pos: 15
type: B, layer: 1, pos: 45
type: B, layer: 1, pos: 18
type: A, layer: 1, pos: 11
type: A, layer: 1, pos: 4
type: B, layer: 1, pos: 11
type: A, layer: 1, pos: 43
type: B, layer: 1, pos: 43
type: A, layer: 1, pos: 29
type: A, layer: 1, pos: 17
type: A, layer: 1, pos: 32
type: B, layer: 1, pos: 37
type: B, layer: 1, pos: 10
type: B, layer: 1, pos: 21
type: A, layer: 1, pos: 16
type: A, layer: 1, pos: 18
type: B, layer: 1, pos: 15
type: A, layer: 1, pos: 21
type: A, layer: 1, pos: 23
type: A, layer: 1, pos: 20
type: A, layer: 1, pos: 39
type: A, layer: 1, pos: 45
type: B, layer: 1, pos: 23
type: B, layer: 1, pos: 41
type: B, layer: 1, pos: 32
type: A, layer: 1, pos: 41
type: B, layer: 1, pos: 39
type: B, layer: 1, pos: 5
type: B, layer: 1, pos: 20
type: A, layer: 1, pos: 5
type: A, layer: 1, pos: 38
type: A, layer: 1, pos: 37
type: B, layer: 1, pos: 16

Time for candidate selection: 0.10 seconds

### Candidate
type: B, layer: 1, pos: 4

### Candidate
type: B, layer: 1, pos: 3

### Candidate
type: A, layer: 1, pos: 35

### Candidate
type: B, layer: 1, pos: 19

## Relational analysis of NS_B1_B2_A2_B1_B2_B1
Optimization infeasible because this subproblem isn't reachable.

### Candidate
type: B, layer: 1, pos: 35

### Candidate
type: A, layer: 1, pos: 7

## Relational analysis of NS_B1_B2_A2_B1_B2_A1
Optimization infeasible because this subproblem isn't reachable.

### Candidate
type: B, layer: 1, pos: 17

## Relational analysis of NS_B1_B2_A2_B1_B2_B1
Optimization infeasible because this subproblem isn't reachable.

### Candidate
type: A, layer: 1, pos: 3

### Candidate
type: A, layer: 1, pos: 9

## Relational analysis of NS_B1_B2_A2_B1_B2_A1

### Relational analysis result of NS_B1_B2_A2_B1_B2_A1
Status: Status.UNKNOWN
Output dim: 0, lower bound: -63.3936783, upper bound: 63.3935557
time: 0.69 seconds

## Relational analysis of NS_B1_B2_A2_B1_B2_A2

### Relational analysis result of NS_B1_B2_A2_B1_B2_A2
Status: Status.UNKNOWN
Output dim: 0, lower bound: -63.3936783, upper bound: 63.3935634
time: 0.65 seconds

## BFS NS instance: NS_B1_B2_A2_B2_B1

### Backsubstitution after applying NS history:
0: -53.6934738, 58.4299927, -31.2356853, 32.9815636, -86.6750336, 89.6656723
1: -398.1136169, 132.5605774, -223.3799896, 77.3887100, -475.5023193, 355.9405518
2: -212.3375549, 125.2409058, -123.8424377, 71.2410431, -283.5786133, 249.0833435
3: -274.3168640, 100.3724899, -155.3355408, 57.1053543, -331.4222107, 255.7080231
4: -152.0602875, 110.8469315, -89.0727921, 62.4052467, -214.4654999, 199.9197235

Time for backsubstitution: 0.94 seconds

### NS candidates at layer 1
type: B, layer: 1, pos: 4
type: B, layer: 1, pos: 3
type: A, layer: 1, pos: 35
type: B, layer: 1, pos: 35
type: A, layer: 1, pos: 3
type: B, layer: 1, pos: 29
type: A, layer: 1, pos: 7
type: B, layer: 1, pos: 38
type: B, layer: 1, pos: 17
type: A, layer: 1, pos: 9
type: A, layer: 1, pos: 43
type: A, layer: 1, pos: 15
type: A, layer: 1, pos: 4
type: B, layer: 1, pos: 45
type: A, layer: 1, pos: 11
type: B, layer: 1, pos: 10
type: A, layer: 1, pos: 18
type: B, layer: 1, pos: 11
type: A, layer: 1, pos: 10
type: A, layer: 1, pos: 19
type: B, layer: 1, pos: 21
type: B, layer: 1, pos: 18
type: B, layer: 1, pos: 43
type: A, layer: 1, pos: 23
type: A, layer: 1, pos: 16
type: A, layer: 1, pos: 45
type: B, layer: 1, pos: 37
type: A, layer: 1, pos: 39
type: B, layer: 1, pos: 32
type: A, layer: 1, pos: 17
type: A, layer: 1, pos: 21
type: B, layer: 1, pos: 15
type: B, layer: 1, pos: 23
type: A, layer: 1, pos: 32
type: B, layer: 1, pos: 41
type: B, layer: 1, pos: 20
type: A, layer: 1, pos: 20
type: B, layer: 1, pos: 5
type: A, layer: 1, pos: 41
type: B, layer: 1, pos: 39
type: A, layer: 1, pos: 38
type: A, layer: 1, pos: 5
type: A, layer: 1, pos: 37
type: B, layer: 1, pos: 16
type: A, layer: 1, pos: 29

Time for candidate selection: 0.08 seconds

### Candidate
type: B, layer: 1, pos: 4

### Candidate
type: B, layer: 1, pos: 3

### Candidate
type: A, layer: 1, pos: 35

### Candidate
type: B, layer: 1, pos: 35

### Candidate
type: A, layer: 1, pos: 3

### Candidate
type: B, layer: 1, pos: 29

## Relational analysis of NS_B1_B2_A2_B2_B1_B1

### Relational analysis result of NS_B1_B2_A2_B2_B1_B1
Status: Status.UNKNOWN
Output dim: 0, lower bound: -63.3940453, upper bound: 63.3941899
time: 0.77 seconds

## Relational analysis of NS_B1_B2_A2_B2_B1_B2

### Relational analysis result of NS_B1_B2_A2_B2_B1_B2
Status: Status.UNKNOWN
Output dim: 0, lower bound: -63.3931911, upper bound: 63.3933291
time: 0.78 seconds

## BFS NS instance: NS_B1_B2_A2_B2_B2

### Backsubstitution after applying NS history:
0: -53.6087799, 58.2966270, -33.7474594, 35.7593155, -89.3680801, 92.0440826
1: -397.0691223, 132.3298340, -244.1085052, 83.6298599, -480.6989441, 376.4383545
2: -211.9619293, 124.9605103, -134.4412231, 77.1542892, -289.1161804, 259.4017029
3: -273.7013550, 100.1439285, -169.4770813, 61.8347511, -335.5360413, 269.6209717
4: -151.8250275, 110.5817719, -96.3958893, 67.4860153, -219.3110046, 206.9776611

Time for backsubstitution: 0.95 seconds

### NS candidates at layer 1
type: B, layer: 1, pos: 4
type: B, layer: 1, pos: 3
type: A, layer: 1, pos: 35
type: B, layer: 1, pos: 29
type: B, layer: 1, pos: 35
type: A, layer: 1, pos: 7
type: B, layer: 1, pos: 38
type: A, layer: 1, pos: 9
type: B, layer: 1, pos: 17
type: A, layer: 1, pos: 3
type: B, layer: 1, pos: 45
type: A, layer: 1, pos: 43
type: A, layer: 1, pos: 15
type: A, layer: 1, pos: 11
type: B, layer: 1, pos: 11
type: A, layer: 1, pos: 18
type: B, layer: 1, pos: 10
type: A, layer: 1, pos: 10
type: A, layer: 1, pos: 45
type: A, layer: 1, pos: 4
type: B, layer: 1, pos: 15
type: B, layer: 1, pos: 18
type: A, layer: 1, pos: 23
type: A, layer: 1, pos: 39
type: A, layer: 1, pos: 17
type: A, layer: 1, pos: 19
type: B, layer: 1, pos: 43
type: B, layer: 1, pos: 21
type: A, layer: 1, pos: 16
type: A, layer: 1, pos: 21
type: B, layer: 1, pos: 37
type: A, layer: 1, pos: 32
type: B, layer: 1, pos: 23
type: B, layer: 1, pos: 32
type: B, layer: 1, pos: 20
type: B, layer: 1, pos: 41
type: A, layer: 1, pos: 20
type: A, layer: 1, pos: 41
type: B, layer: 1, pos: 5
type: A, layer: 1, pos: 5
type: B, layer: 1, pos: 39
type: A, layer: 1, pos: 37
type: A, layer: 1, pos: 38
type: B, layer: 1, pos: 16
type: A, layer: 1, pos: 29

Time for candidate selection: 0.08 seconds

### Candidate
type: B, layer: 1, pos: 4

### Candidate
type: B, layer: 1, pos: 3

### Candidate
type: A, layer: 1, pos: 35

### Candidate
type: B, layer: 1, pos: 29

## Relational analysis of NS_B1_B2_A2_B2_B2_B1

### Relational analysis result of NS_B1_B2_A2_B2_B2_B1
Status: Status.UNKNOWN
Output dim: 0, lower bound: -63.3940682, upper bound: 63.3941989
time: 0.86 seconds

## Relational analysis of NS_B1_B2_A2_B2_B2_B2

### Relational analysis result of NS_B1_B2_A2_B2_B2_B2
Status: Status.UNKNOWN
Output dim: 0, lower bound: -63.3937801, upper bound: 63.3937700
time: 0.76 seconds

## BFS NS instance: NS_B2_A1_A1_B1_A1

### Backsubstitution after applying NS history:
0: -31.7064209, 33.1850853, -52.9641418, 57.4213943, -89.1278152, 86.1492310
1: -224.8632965, 78.4888840, -390.0163269, 130.6372681, -355.5005493, 467.4760437
2: -125.8085251, 71.6483917, -208.8492889, 123.1231308, -248.9316559, 280.4976807
3: -157.2832489, 57.4549141, -269.1730042, 98.6250229, -255.9082336, 326.2878113
4: -90.4712524, 62.5022736, -150.1322632, 108.8735657, -199.3447876, 212.6345215

Time for backsubstitution: 1.01 seconds

### NS candidates at layer 1
type: A, layer: 1, pos: 4
type: A, layer: 1, pos: 9
type: B, layer: 1, pos: 35
type: A, layer: 1, pos: 19
type: A, layer: 1, pos: 38
type: A, layer: 1, pos: 17
type: A, layer: 1, pos: 3
type: A, layer: 1, pos: 45
type: B, layer: 1, pos: 9
type: B, layer: 1, pos: 11
type: B, layer: 1, pos: 19
type: B, layer: 1, pos: 15
type: A, layer: 1, pos: 11
type: A, layer: 1, pos: 10
type: B, layer: 1, pos: 10
type: A, layer: 1, pos: 35
type: B, layer: 1, pos: 16
type: B, layer: 1, pos: 39
type: A, layer: 1, pos: 37
type: B, layer: 1, pos: 43
type: B, layer: 1, pos: 17
type: B, layer: 1, pos: 3
type: B, layer: 1, pos: 18
type: A, layer: 1, pos: 18
type: A, layer: 1, pos: 32
type: A, layer: 1, pos: 21
type: B, layer: 1, pos: 45
type: B, layer: 1, pos: 23
type: B, layer: 1, pos: 32
type: A, layer: 1, pos: 41
type: A, layer: 1, pos: 15
type: A, layer: 1, pos: 5
type: B, layer: 1, pos: 41
type: B, layer: 1, pos: 20
type: A, layer: 1, pos: 39
type: A, layer: 1, pos: 20
type: B, layer: 1, pos: 5
type: B, layer: 1, pos: 21
type: A, layer: 1, pos: 43
type: A, layer: 1, pos: 23
type: B, layer: 1, pos: 37
type: A, layer: 1, pos: 16
type: B, layer: 1, pos: 38
type: B, layer: 1, pos: 4
type: B, layer: 1, pos: 29

Time for candidate selection: 0.08 seconds

### Candidate
type: A, layer: 1, pos: 4

### Candidate
type: A, layer: 1, pos: 9

### Candidate
type: B, layer: 1, pos: 35

### Candidate
type: A, layer: 1, pos: 19

### Candidate
type: A, layer: 1, pos: 38

### Candidate
type: A, layer: 1, pos: 17

### Candidate
type: A, layer: 1, pos: 3

### Candidate
type: A, layer: 1, pos: 45

## Relational analysis of NS_B2_A1_A1_B1_A1_A1

### Relational analysis result of NS_B2_A1_A1_B1_A1_A1
Status: Status.UNKNOWN
Output dim: 0, lower bound: -63.3940047, upper bound: 63.3945002
time: 0.73 seconds

## Relational analysis of NS_B2_A1_A1_B1_A1_A2

### Relational analysis result of NS_B2_A1_A1_B1_A1_A2
Status: Status.UNKNOWN
Output dim: 0, lower bound: -63.3940047, upper bound: 63.3944632
time: 0.72 seconds

## BFS NS instance: NS_B2_A1_A1_B1_A2

### Backsubstitution after applying NS history:
0: -40.8103180, 41.9722328, -52.4516983, 56.7775230, -97.4968338, 94.4239273
1: -275.1538086, 100.9877777, -384.3140564, 129.3001862, -404.4539795, 483.3817749
2: -158.3198853, 90.8393936, -206.4783630, 121.6884079, -279.1997986, 297.3176575
3: -193.7020264, 72.6721802, -265.4350586, 97.5187225, -291.0138245, 337.7642822
4: -115.7052841, 79.5005569, -148.5830231, 107.6768570, -222.3908234, 228.0835419

Time for backsubstitution: 1.04 seconds

### NS candidates at layer 1
type: A, layer: 1, pos: 4
type: A, layer: 1, pos: 9
type: A, layer: 1, pos: 19
type: B, layer: 1, pos: 35
type: A, layer: 1, pos: 3
type: A, layer: 1, pos: 17
type: B, layer: 1, pos: 9
type: A, layer: 1, pos: 45
type: B, layer: 1, pos: 10
type: A, layer: 1, pos: 35
type: B, layer: 1, pos: 11
type: A, layer: 1, pos: 38
type: A, layer: 1, pos: 11
type: B, layer: 1, pos: 15
type: B, layer: 1, pos: 17
type: B, layer: 1, pos: 19
type: A, layer: 1, pos: 18
type: B, layer: 1, pos: 3
type: B, layer: 1, pos: 4
type: B, layer: 1, pos: 29
type: B, layer: 1, pos: 16
type: A, layer: 1, pos: 10
type: A, layer: 1, pos: 37
type: B, layer: 1, pos: 32
type: B, layer: 1, pos: 39
type: B, layer: 1, pos: 45
type: A, layer: 1, pos: 15
type: A, layer: 1, pos: 43
type: A, layer: 1, pos: 39
type: B, layer: 1, pos: 18
type: A, layer: 1, pos: 21
type: B, layer: 1, pos: 20
type: A, layer: 1, pos: 32
type: A, layer: 1, pos: 41
type: B, layer: 1, pos: 43
type: B, layer: 1, pos: 41
type: A, layer: 1, pos: 5
type: B, layer: 1, pos: 23
type: A, layer: 1, pos: 23
type: B, layer: 1, pos: 5
type: B, layer: 1, pos: 21
type: A, layer: 1, pos: 20
type: B, layer: 1, pos: 37
type: A, layer: 1, pos: 16
type: B, layer: 1, pos: 38

Time for candidate selection: 0.08 seconds

### Candidate
type: A, layer: 1, pos: 4

### Candidate
type: A, layer: 1, pos: 9

### Candidate
type: A, layer: 1, pos: 19

### Candidate
type: B, layer: 1, pos: 35

### Candidate
type: A, layer: 1, pos: 3

### Candidate
type: A, layer: 1, pos: 17

### Candidate
type: B, layer: 1, pos: 9

### Candidate
type: A, layer: 1, pos: 45

## Relational analysis of NS_B2_A1_A1_B1_A2_A1

### Relational analysis result of NS_B2_A1_A1_B1_A2_A1
Status: Status.UNKNOWN
Output dim: 0, lower bound: -63.3935219, upper bound: 63.3938497
time: 0.70 seconds

## Relational analysis of NS_B2_A1_A1_B1_A2_A2

### Relational analysis result of NS_B2_A1_A1_B1_A2_A2
Status: Status.UNKNOWN
Output dim: 0, lower bound: -63.3939658, upper bound: 63.3943218
time: 0.85 seconds

## BFS NS instance: NS_B2_A1_A1_B2_A1

### Backsubstitution after applying NS history:
0: -31.7064209, 33.1850853, -53.1764832, 57.9081841, -89.6146088, 86.3615723
1: -224.8632965, 78.4888840, -394.9996948, 131.3654633, -356.2286987, 472.9119873
2: -125.8085251, 71.6483917, -210.4864197, 124.0891571, -249.8976746, 282.1348267
3: -157.2832489, 57.4549141, -272.1233826, 99.5140686, -256.7972717, 329.5782776
4: -90.4712524, 62.5022736, -150.6161346, 109.8162842, -200.2875061, 213.1184082

Time for backsubstitution: 0.97 seconds

### NS candidates at layer 1
type: A, layer: 1, pos: 4
type: A, layer: 1, pos: 3
type: B, layer: 1, pos: 35
type: A, layer: 1, pos: 9
type: A, layer: 1, pos: 19
type: A, layer: 1, pos: 38
type: B, layer: 1, pos: 3
type: A, layer: 1, pos: 35
type: B, layer: 1, pos: 9
type: A, layer: 1, pos: 17
type: B, layer: 1, pos: 43
type: B, layer: 1, pos: 15
type: B, layer: 1, pos: 19
type: B, layer: 1, pos: 4
type: A, layer: 1, pos: 45
type: B, layer: 1, pos: 11
type: B, layer: 1, pos: 10
type: A, layer: 1, pos: 11
type: A, layer: 1, pos: 10
type: A, layer: 1, pos: 18
type: A, layer: 1, pos: 43
type: B, layer: 1, pos: 18
type: B, layer: 1, pos: 23
type: A, layer: 1, pos: 37
type: B, layer: 1, pos: 16
type: A, layer: 1, pos: 21
type: B, layer: 1, pos: 39
type: B, layer: 1, pos: 32
type: B, layer: 1, pos: 17
type: B, layer: 1, pos: 21
type: B, layer: 1, pos: 45
type: A, layer: 1, pos: 32
type: A, layer: 1, pos: 23
type: A, layer: 1, pos: 15
type: A, layer: 1, pos: 41
type: B, layer: 1, pos: 20
type: A, layer: 1, pos: 20
type: A, layer: 1, pos: 5
type: B, layer: 1, pos: 41
type: A, layer: 1, pos: 39
type: B, layer: 1, pos: 5
type: B, layer: 1, pos: 38
type: A, layer: 1, pos: 16
type: B, layer: 1, pos: 37
type: B, layer: 1, pos: 29

Time for candidate selection: 0.08 seconds

### Candidate
type: A, layer: 1, pos: 4

### Candidate
type: A, layer: 1, pos: 3

### Candidate
type: B, layer: 1, pos: 35

### Candidate
type: A, layer: 1, pos: 9

### Candidate
type: A, layer: 1, pos: 19

### Candidate
type: A, layer: 1, pos: 38

### Candidate
type: B, layer: 1, pos: 3

### Candidate
type: A, layer: 1, pos: 35

### Candidate
type: B, layer: 1, pos: 9

### Candidate
type: A, layer: 1, pos: 17

### Candidate
type: B, layer: 1, pos: 43

## Relational analysis of NS_B2_A1_A1_B2_A1_B1
Optimization infeasible because this subproblem isn't reachable.

### Candidate
type: B, layer: 1, pos: 15

## Relational analysis of NS_B2_A1_A1_B2_A1_B1
Optimization infeasible because this subproblem isn't reachable.

### Candidate
type: B, layer: 1, pos: 19

### Candidate
type: B, layer: 1, pos: 4

### Candidate
type: A, layer: 1, pos: 45

## Relational analysis of NS_B2_A1_A1_B2_A1_A1

### Relational analysis result of NS_B2_A1_A1_B2_A1_A1
Status: Status.UNKNOWN
Output dim: 0, lower bound: -63.3940047, upper bound: 63.3945531
time: 0.69 seconds

## Relational analysis of NS_B2_A1_A1_B2_A1_A2

### Relational analysis result of NS_B2_A1_A1_B2_A1_A2
Status: Status.UNKNOWN
Output dim: 0, lower bound: -63.3942132, upper bound: 63.3945397
time: 0.72 seconds

## BFS NS instance: NS_B2_A1_A1_B2_A2

### Backsubstitution after applying NS history:
0: -40.8103180, 41.9722328, -52.5505524, 57.1262398, -97.8766022, 94.5227814
1: -275.1538086, 100.9877777, -388.2085876, 129.6735077, -404.8273010, 487.7508240
2: -158.3198853, 90.8393936, -207.5194092, 122.3719025, -279.9708862, 298.3587646
3: -193.7020264, 72.6721802, -267.7059937, 98.1092606, -291.6645813, 340.3781738
4: -115.7052841, 79.5005569, -148.6719513, 108.3880539, -223.1401672, 228.1725159

Time for backsubstitution: 0.97 seconds

### NS candidates at layer 1
type: A, layer: 1, pos: 4
type: A, layer: 1, pos: 3
type: B, layer: 1, pos: 35
type: A, layer: 1, pos: 19
type: A, layer: 1, pos: 9
type: A, layer: 1, pos: 35
type: B, layer: 1, pos: 9
type: A, layer: 1, pos: 38
type: A, layer: 1, pos: 17
type: B, layer: 1, pos: 3
type: B, layer: 1, pos: 15
type: B, layer: 1, pos: 10
type: B, layer: 1, pos: 19
type: A, layer: 1, pos: 45
type: A, layer: 1, pos: 18
type: B, layer: 1, pos: 11
type: B, layer: 1, pos: 4
type: A, layer: 1, pos: 43
type: A, layer: 1, pos: 11
type: B, layer: 1, pos: 17
type: B, layer: 1, pos: 43
type: B, layer: 1, pos: 29
type: B, layer: 1, pos: 32
type: B, layer: 1, pos: 16
type: A, layer: 1, pos: 37
type: A, layer: 1, pos: 10
type: A, layer: 1, pos: 21
type: B, layer: 1, pos: 18
type: B, layer: 1, pos: 21
type: A, layer: 1, pos: 15
type: B, layer: 1, pos: 20
type: B, layer: 1, pos: 23
type: B, layer: 1, pos: 39
type: B, layer: 1, pos: 45
type: A, layer: 1, pos: 23
type: A, layer: 1, pos: 41
type: A, layer: 1, pos: 32
type: B, layer: 1, pos: 41
type: A, layer: 1, pos: 5
type: A, layer: 1, pos: 39
type: A, layer: 1, pos: 20
type: B, layer: 1, pos: 5
type: B, layer: 1, pos: 37
type: B, layer: 1, pos: 38
type: A, layer: 1, pos: 16

Time for candidate selection: 0.08 seconds

### Candidate
type: A, layer: 1, pos: 4

### Candidate
type: A, layer: 1, pos: 3

### Candidate
type: B, layer: 1, pos: 35

### Candidate
type: A, layer: 1, pos: 19

### Candidate
type: A, layer: 1, pos: 9

### Candidate
type: A, layer: 1, pos: 35

### Candidate
type: B, layer: 1, pos: 9

### Candidate
type: A, layer: 1, pos: 38

### Candidate
type: A, layer: 1, pos: 17

### Candidate
type: B, layer: 1, pos: 3

### Candidate
type: B, layer: 1, pos: 15

## Relational analysis of NS_B2_A1_A1_B2_A2_B1
Optimization infeasible because this subproblem isn't reachable.

### Candidate
type: B, layer: 1, pos: 10

## Relational analysis of NS_B2_A1_A1_B2_A2_B1
Optimization infeasible because this subproblem isn't reachable.

### Candidate
type: B, layer: 1, pos: 19

### Candidate
type: A, layer: 1, pos: 45

## Relational analysis of NS_B2_A1_A1_B2_A2_A1
Optimization infeasible because this subproblem isn't reachable.

### Candidate
type: A, layer: 1, pos: 18

## Relational analysis of NS_B2_A1_A1_B2_A2_A1
Optimization infeasible because this subproblem isn't reachable.

### Candidate
type: B, layer: 1, pos: 11

## Relational analysis of NS_B2_A1_A1_B2_A2_B1

### Relational analysis result of NS_B2_A1_A1_B2_A2_B1
Status: Status.UNKNOWN
Output dim: 0, lower bound: -63.3939171, upper bound: 63.3939666
time: 0.75 seconds

## Relational analysis of NS_B2_A1_A1_B2_A2_B2

### Relational analysis result of NS_B2_A1_A1_B2_A2_B2
Status: Status.UNKNOWN
Output dim: 0, lower bound: -63.3941648, upper bound: 63.3942476
time: 0.72 seconds

## BFS NS instance: NS_B2_A1_A2_B1_A1

### Backsubstitution after applying NS history:
0: -53.9737701, 58.4838371, -53.9128532, 58.4085197, -112.3822861, 112.3966904
1: -396.7337952, 133.1474609, -396.3260193, 132.9687500, -528.5906372, 528.4053955
2: -212.7236023, 125.4101639, -212.4430084, 125.2666550, -337.9902649, 337.8531494
3: -273.8752441, 100.4784775, -273.5679626, 100.3590775, -373.2450867, 373.0678101
4: -152.9991302, 110.9224167, -152.7918243, 110.7855988, -263.7846985, 263.7142334

Time for backsubstitution: 0.96 seconds

### NS candidates at layer 1
type: B, layer: 1, pos: 4
type: A, layer: 1, pos: 4
type: A, layer: 1, pos: 9
type: B, layer: 1, pos: 9
type: B, layer: 1, pos: 35
type: A, layer: 1, pos: 35
type: B, layer: 1, pos: 17
type: A, layer: 1, pos: 17
type: A, layer: 1, pos: 45
type: B, layer: 1, pos: 45
type: B, layer: 1, pos: 19
type: A, layer: 1, pos: 19
type: A, layer: 1, pos: 3
type: B, layer: 1, pos: 3
type: B, layer: 1, pos: 10
type: A, layer: 1, pos: 10
type: B, layer: 1, pos: 11
type: A, layer: 1, pos: 15
type: A, layer: 1, pos: 18
type: B, layer: 1, pos: 18
type: B, layer: 1, pos: 15
type: B, layer: 1, pos: 39
type: A, layer: 1, pos: 39
type: B, layer: 1, pos: 37
type: A, layer: 1, pos: 43
type: B, layer: 1, pos: 43
type: A, layer: 1, pos: 38
type: B, layer: 1, pos: 32
type: A, layer: 1, pos: 37
type: B, layer: 1, pos: 38
type: A, layer: 1, pos: 32
type: B, layer: 1, pos: 41
type: A, layer: 1, pos: 41
type: A, layer: 1, pos: 29
type: B, layer: 1, pos: 29
type: B, layer: 1, pos: 20
type: A, layer: 1, pos: 20
type: A, layer: 1, pos: 5
type: B, layer: 1, pos: 16
type: B, layer: 1, pos: 5
type: A, layer: 1, pos: 16
type: A, layer: 1, pos: 21
type: B, layer: 1, pos: 21
type: B, layer: 1, pos: 23
type: A, layer: 1, pos: 23

Time for candidate selection: 0.08 seconds

### Candidate
type: B, layer: 1, pos: 4

### Candidate
type: A, layer: 1, pos: 4

### Candidate
type: A, layer: 1, pos: 9

### Candidate
type: B, layer: 1, pos: 9

### Candidate
type: B, layer: 1, pos: 35

### Candidate
type: A, layer: 1, pos: 35

### Candidate
type: B, layer: 1, pos: 17

### Candidate
type: A, layer: 1, pos: 17

### Candidate
type: A, layer: 1, pos: 45

### Candidate
type: B, layer: 1, pos: 45

### Candidate
type: B, layer: 1, pos: 19

### Candidate
type: A, layer: 1, pos: 19

### Candidate
type: A, layer: 1, pos: 3

### Candidate
type: B, layer: 1, pos: 3

### Candidate
type: B, layer: 1, pos: 10

### Candidate
type: A, layer: 1, pos: 10

### Candidate
type: B, layer: 1, pos: 11

## Relational analysis of NS_B2_A1_A2_B1_A1_B1

### Relational analysis result of NS_B2_A1_A2_B1_A1_B1
Status: Status.UNKNOWN
Output dim: 0, lower bound: -63.3929322, upper bound: 63.3929322
time: 0.73 seconds

## Relational analysis of NS_B2_A1_A2_B1_A1_B2

### Relational analysis result of NS_B2_A1_A2_B1_A1_B2
Status: Status.UNKNOWN
Output dim: 0, lower bound: -63.3929322, upper bound: 63.3930551
time: 0.70 seconds

## BFS NS instance: NS_B2_A1_A2_B1_A2

### Backsubstitution after applying NS history:
0: -53.8703156, 58.3821106, -54.0113716, 58.5142365, -112.3845520, 112.3934784
1: -396.2604980, 132.8612976, -397.1004944, 133.2129364, -528.4453735, 528.9489746
2: -212.2722015, 125.2070465, -212.8338928, 125.4999237, -337.7720947, 338.0409546
3: -273.4829102, 100.3024445, -274.0934448, 100.5448303, -373.0795898, 373.4604797
4: -152.6544037, 110.7377243, -153.0641937, 110.9862289, -263.6406250, 263.8019104

Time for backsubstitution: 0.96 seconds

### NS candidates at layer 1
type: B, layer: 1, pos: 4
type: A, layer: 1, pos: 4
type: A, layer: 1, pos: 9
type: B, layer: 1, pos: 9
type: B, layer: 1, pos: 35
type: A, layer: 1, pos: 35
type: A, layer: 1, pos: 17
type: B, layer: 1, pos: 17
type: A, layer: 1, pos: 45
type: B, layer: 1, pos: 45
type: B, layer: 1, pos: 19
type: B, layer: 1, pos: 3
type: A, layer: 1, pos: 19
type: A, layer: 1, pos: 3
type: B, layer: 1, pos: 10
type: A, layer: 1, pos: 10
type: B, layer: 1, pos: 11
type: B, layer: 1, pos: 18
type: A, layer: 1, pos: 18
type: A, layer: 1, pos: 15
type: B, layer: 1, pos: 15
type: B, layer: 1, pos: 39
type: A, layer: 1, pos: 39
type: B, layer: 1, pos: 37
type: B, layer: 1, pos: 38
type: B, layer: 1, pos: 43
type: A, layer: 1, pos: 32
type: A, layer: 1, pos: 43
type: A, layer: 1, pos: 37
type: A, layer: 1, pos: 38
type: B, layer: 1, pos: 32
type: B, layer: 1, pos: 29
type: B, layer: 1, pos: 41
type: A, layer: 1, pos: 41
type: A, layer: 1, pos: 29
type: A, layer: 1, pos: 20
type: B, layer: 1, pos: 20
type: A, layer: 1, pos: 5
type: B, layer: 1, pos: 5
type: B, layer: 1, pos: 16
type: A, layer: 1, pos: 16
type: A, layer: 1, pos: 21
type: B, layer: 1, pos: 23
type: B, layer: 1, pos: 21
type: A, layer: 1, pos: 23

Time for candidate selection: 0.08 seconds

### Candidate
type: B, layer: 1, pos: 4

### Candidate
type: A, layer: 1, pos: 4

### Candidate
type: A, layer: 1, pos: 9

### Candidate
type: B, layer: 1, pos: 9

### Candidate
type: B, layer: 1, pos: 35

### Candidate
type: A, layer: 1, pos: 35

### Candidate
type: A, layer: 1, pos: 17

### Candidate
type: B, layer: 1, pos: 17

### Candidate
type: A, layer: 1, pos: 45

### Candidate
type: B, layer: 1, pos: 45

### Candidate
type: B, layer: 1, pos: 19

### Candidate
type: B, layer: 1, pos: 3

### Candidate
type: A, layer: 1, pos: 19

### Candidate
type: A, layer: 1, pos: 3

### Candidate
type: B, layer: 1, pos: 10

### Candidate
type: A, layer: 1, pos: 10

### Candidate
type: B, layer: 1, pos: 11

## Relational analysis of NS_B2_A1_A2_B1_A2_B1

### Relational analysis result of NS_B2_A1_A2_B1_A2_B1
Status: Status.UNKNOWN
Output dim: 0, lower bound: -63.3930551, upper bound: 63.3930077
time: 0.74 seconds

## Relational analysis of NS_B2_A1_A2_B1_A2_B2

### Relational analysis result of NS_B2_A1_A2_B1_A2_B2
Status: Status.UNKNOWN
Output dim: 0, lower bound: -63.3930551, upper bound: 63.3931305
time: 0.74 seconds

## BFS NS instance: NS_B2_A1_A2_B2_B1

### Backsubstitution after applying NS history:
0: -53.9119186, 58.4074745, -54.1881714, 58.9965973, -112.9085159, 112.5956421
1: -396.3174438, 132.9662628, -402.2130432, 133.8377991, -529.2075806, 534.5039673
2: -212.4386292, 125.2644272, -214.3997955, 126.4415131, -338.8800964, 339.6642151
3: -273.5619812, 100.3572235, -277.1029663, 101.3699265, -374.0090332, 376.8019714
4: -152.7888184, 110.7837906, -153.4725342, 111.9196091, -264.7084351, 264.2563171

Time for backsubstitution: 0.97 seconds

### NS candidates at layer 1
type: B, layer: 1, pos: 35
type: A, layer: 1, pos: 35
type: B, layer: 1, pos: 9
type: A, layer: 1, pos: 9
type: A, layer: 1, pos: 19
type: B, layer: 1, pos: 19
type: A, layer: 1, pos: 3
type: B, layer: 1, pos: 3
type: A, layer: 1, pos: 17
type: B, layer: 1, pos: 17
type: B, layer: 1, pos: 10
type: A, layer: 1, pos: 10
type: A, layer: 1, pos: 18
type: A, layer: 1, pos: 45
type: B, layer: 1, pos: 15
type: A, layer: 1, pos: 43
type: B, layer: 1, pos: 18
type: A, layer: 1, pos: 15
type: B, layer: 1, pos: 29
type: B, layer: 1, pos: 45
type: B, layer: 1, pos: 43
type: A, layer: 1, pos: 11
type: A, layer: 1, pos: 29
type: B, layer: 1, pos: 21
type: A, layer: 1, pos: 21
type: B, layer: 1, pos: 23
type: A, layer: 1, pos: 37
type: A, layer: 1, pos: 23
type: B, layer: 1, pos: 32
type: A, layer: 1, pos: 32
type: A, layer: 1, pos: 4
type: A, layer: 1, pos: 38
type: A, layer: 1, pos: 20
type: B, layer: 1, pos: 39
type: B, layer: 1, pos: 37
type: A, layer: 1, pos: 41
type: B, layer: 1, pos: 41
type: B, layer: 1, pos: 20
type: A, layer: 1, pos: 39
type: B, layer: 1, pos: 16
type: A, layer: 1, pos: 5
type: A, layer: 1, pos: 16
type: B, layer: 1, pos: 5
type: B, layer: 1, pos: 4
type: B, layer: 1, pos: 38

Time for candidate selection: 0.08 seconds

### Candidate
type: B, layer: 1, pos: 35

### Candidate
type: A, layer: 1, pos: 35

### Candidate
type: B, layer: 1, pos: 9

### Candidate
type: A, layer: 1, pos: 9

### Candidate
type: A, layer: 1, pos: 19

### Candidate
type: B, layer: 1, pos: 19

### Candidate
type: A, layer: 1, pos: 3

### Candidate
type: B, layer: 1, pos: 3

### Candidate
type: A, layer: 1, pos: 17

### Candidate
type: B, layer: 1, pos: 17

### Candidate
type: B, layer: 1, pos: 10

### Candidate
type: A, layer: 1, pos: 10

### Candidate
type: A, layer: 1, pos: 18

### Candidate
type: A, layer: 1, pos: 45

### Candidate
type: B, layer: 1, pos: 15

## Relational analysis of NS_B2_A1_A2_B2_B1_B1
Optimization infeasible because this subproblem isn't reachable.

### Candidate
type: A, layer: 1, pos: 43

## Relational analysis of NS_B2_A1_A2_B2_B1_A1
Optimization infeasible because this subproblem isn't reachable.

### Candidate
type: B, layer: 1, pos: 18

### Candidate
type: A, layer: 1, pos: 15

### Candidate
type: B, layer: 1, pos: 29

## Relational analysis of NS_B2_A1_A2_B2_B1_B1
Optimization infeasible because this subproblem isn't reachable.

### Candidate
type: B, layer: 1, pos: 45

### Candidate
type: B, layer: 1, pos: 43

### Candidate
type: A, layer: 1, pos: 11

## Relational analysis of NS_B2_A1_A2_B2_B1_A1

### Relational analysis result of NS_B2_A1_A2_B2_B1_A1
Status: Status.UNKNOWN
Output dim: 0, lower bound: -63.3930511, upper bound: 63.3930163
time: 0.71 seconds

## Relational analysis of NS_B2_A1_A2_B2_B1_A2

### Relational analysis result of NS_B2_A1_A2_B2_B1_A2
Status: Status.UNKNOWN
Output dim: 0, lower bound: -63.3930511, upper bound: 63.3930898
time: 0.72 seconds

## BFS NS instance: NS_B2_A1_A2_B2_B2

### Backsubstitution after applying NS history:
0: -54.0104523, 58.5131683, -53.8847351, 58.6575432, -112.6679840, 112.3978729
1: -397.0915527, 133.2104645, -399.7374878, 133.0946503, -529.2918091, 532.3772583
2: -212.8295288, 125.4976883, -213.1640167, 125.7145767, -338.5440979, 338.6616516
3: -274.0873108, 100.5429916, -275.4341431, 100.8090820, -374.0199890, 375.3732300
4: -153.0611725, 110.9844284, -152.5790253, 111.2873230, -264.3484802, 263.5633850

Time for backsubstitution: 0.97 seconds

### NS candidates at layer 1
type: B, layer: 1, pos: 35
type: A, layer: 1, pos: 35
type: B, layer: 1, pos: 9
type: A, layer: 1, pos: 9
type: A, layer: 1, pos: 19
type: B, layer: 1, pos: 19
type: A, layer: 1, pos: 3
type: B, layer: 1, pos: 3
type: A, layer: 1, pos: 17
type: B, layer: 1, pos: 17
type: B, layer: 1, pos: 10
type: A, layer: 1, pos: 18
type: A, layer: 1, pos: 10
type: A, layer: 1, pos: 45
type: B, layer: 1, pos: 15
type: A, layer: 1, pos: 43
type: A, layer: 1, pos: 11
type: B, layer: 1, pos: 18
type: A, layer: 1, pos: 15
type: B, layer: 1, pos: 29
type: B, layer: 1, pos: 45
type: B, layer: 1, pos: 43
type: A, layer: 1, pos: 29
type: B, layer: 1, pos: 21
type: A, layer: 1, pos: 21
type: B, layer: 1, pos: 23
type: B, layer: 1, pos: 32
type: A, layer: 1, pos: 37
type: A, layer: 1, pos: 23
type: A, layer: 1, pos: 38
type: A, layer: 1, pos: 32
type: B, layer: 1, pos: 39
type: A, layer: 1, pos: 20
type: B, layer: 1, pos: 37
type: A, layer: 1, pos: 4
type: A, layer: 1, pos: 41
type: B, layer: 1, pos: 20
type: B, layer: 1, pos: 41
type: A, layer: 1, pos: 39
type: B, layer: 1, pos: 16
type: A, layer: 1, pos: 5
type: B, layer: 1, pos: 5
type: A, layer: 1, pos: 16
type: B, layer: 1, pos: 4
type: B, layer: 1, pos: 38

Time for candidate selection: 0.08 seconds

### Candidate
type: B, layer: 1, pos: 35

### Candidate
type: A, layer: 1, pos: 35

### Candidate
type: B, layer: 1, pos: 9

### Candidate
type: A, layer: 1, pos: 9

### Candidate
type: A, layer: 1, pos: 19

### Candidate
type: B, layer: 1, pos: 19

### Candidate
type: A, layer: 1, pos: 3

### Candidate
type: B, layer: 1, pos: 3

### Candidate
type: A, layer: 1, pos: 17

### Candidate
type: B, layer: 1, pos: 17

### Candidate
type: B, layer: 1, pos: 10

### Candidate
type: A, layer: 1, pos: 18

### Candidate
type: A, layer: 1, pos: 10

### Candidate
type: A, layer: 1, pos: 45

### Candidate
type: B, layer: 1, pos: 15

## Relational analysis of NS_B2_A1_A2_B2_B2_B1
Optimization infeasible because this subproblem isn't reachable.

### Candidate
type: A, layer: 1, pos: 43

## Relational analysis of NS_B2_A1_A2_B2_B2_A1
Optimization infeasible because this subproblem isn't reachable.

### Candidate
type: A, layer: 1, pos: 11

## Relational analysis of NS_B2_A1_A2_B2_B2_A1

### Relational analysis result of NS_B2_A1_A2_B2_B2_A1
Status: Status.UNKNOWN
Output dim: 0, lower bound: -63.3937162, upper bound: 63.3934490
time: 0.70 seconds

## Relational analysis of NS_B2_A1_A2_B2_B2_A2

### Relational analysis result of NS_B2_A1_A2_B2_B2_A2
Status: Status.UNKNOWN
Output dim: 0, lower bound: -63.3937162, upper bound: 63.3935402
time: 0.75 seconds

## BFS NS instance: NS_B2_A2_A1_A1_B1

### Backsubstitution after applying NS history:
0: -31.2933064, 32.9650536, -52.9280472, 57.3836327, -88.6769409, 85.8930740
1: -223.5657806, 77.4652863, -389.7350159, 130.5437317, -354.1094971, 466.6008606
2: -123.6531296, 71.1243896, -208.6952515, 123.0406799, -246.6938171, 279.8196411
3: -155.3525391, 57.0316010, -268.9759521, 98.5570068, -253.9095459, 325.7637634
4: -88.9686127, 62.2370262, -150.0240326, 108.8049927, -197.7735901, 212.2610626

Time for backsubstitution: 0.97 seconds

### NS candidates at layer 1
type: A, layer: 1, pos: 4
type: A, layer: 1, pos: 9
type: B, layer: 1, pos: 35
type: A, layer: 1, pos: 17
type: A, layer: 1, pos: 19
type: A, layer: 1, pos: 3
type: A, layer: 1, pos: 38
type: B, layer: 1, pos: 9
type: A, layer: 1, pos: 45
type: B, layer: 1, pos: 11
type: B, layer: 1, pos: 19
type: A, layer: 1, pos: 35
type: A, layer: 1, pos: 10
type: A, layer: 1, pos: 11
type: B, layer: 1, pos: 10
type: B, layer: 1, pos: 15
type: B, layer: 1, pos: 18
type: B, layer: 1, pos: 43
type: B, layer: 1, pos: 16
type: B, layer: 1, pos: 3
type: A, layer: 1, pos: 37
type: B, layer: 1, pos: 39
type: B, layer: 1, pos: 17
type: A, layer: 1, pos: 32
type: B, layer: 1, pos: 45
type: A, layer: 1, pos: 18
type: A, layer: 1, pos: 21
type: A, layer: 1, pos: 15
type: B, layer: 1, pos: 23
type: A, layer: 1, pos: 41
type: B, layer: 1, pos: 32
type: A, layer: 1, pos: 39
type: B, layer: 1, pos: 20
type: B, layer: 1, pos: 41
type: A, layer: 1, pos: 5
type: B, layer: 1, pos: 5
type: A, layer: 1, pos: 20
type: B, layer: 1, pos: 21
type: A, layer: 1, pos: 23
type: A, layer: 1, pos: 43
type: B, layer: 1, pos: 37
type: A, layer: 1, pos: 16
type: B, layer: 1, pos: 38
type: B, layer: 1, pos: 4
type: B, layer: 1, pos: 29

Time for candidate selection: 0.08 seconds

### Candidate
type: A, layer: 1, pos: 4

### Candidate
type: A, layer: 1, pos: 9

### Candidate
type: B, layer: 1, pos: 35

### Candidate
type: A, layer: 1, pos: 17

## Relational analysis of NS_B2_A2_A1_A1_B1_A1
Optimization infeasible because this subproblem isn't reachable.

### Candidate
type: A, layer: 1, pos: 19

### Candidate
type: A, layer: 1, pos: 3

### Candidate
type: A, layer: 1, pos: 38

### Candidate
type: B, layer: 1, pos: 9

### Candidate
type: A, layer: 1, pos: 45

## Relational analysis of NS_B2_A2_A1_A1_B1_A1

### Relational analysis result of NS_B2_A2_A1_A1_B1_A1
Status: Status.UNKNOWN
Output dim: 0, lower bound: -63.3939698, upper bound: 63.3944786
time: 0.70 seconds

## Relational analysis of NS_B2_A2_A1_A1_B1_A2

### Relational analysis result of NS_B2_A2_A1_A1_B1_A2
Status: Status.UNKNOWN
Output dim: 0, lower bound: -63.3939587, upper bound: 63.3944746
time: 0.75 seconds

## BFS NS instance: NS_B2_A2_A1_A1_B2

### Backsubstitution after applying NS history:
0: -31.2933064, 32.9650536, -53.1764832, 57.9081841, -89.2014923, 86.1415100
1: -223.5657806, 77.4652863, -394.9996948, 131.3654633, -354.9312439, 472.4649658
2: -123.6531296, 71.1243896, -210.4864197, 124.0891571, -247.7422791, 281.6108093
3: -155.3525391, 57.0316010, -272.1233826, 99.5140686, -254.8665924, 329.1549683
4: -88.9686127, 62.2370262, -150.6161346, 109.8162842, -198.7848511, 212.8531647

Time for backsubstitution: 0.99 seconds

### NS candidates at layer 1
type: A, layer: 1, pos: 4
type: A, layer: 1, pos: 3
type: B, layer: 1, pos: 35
type: A, layer: 1, pos: 19
type: A, layer: 1, pos: 9
type: B, layer: 1, pos: 3
type: A, layer: 1, pos: 35
type: A, layer: 1, pos: 38
type: B, layer: 1, pos: 9
type: A, layer: 1, pos: 17
type: B, layer: 1, pos: 43
type: B, layer: 1, pos: 19
type: B, layer: 1, pos: 15
type: B, layer: 1, pos: 4
type: A, layer: 1, pos: 45
type: B, layer: 1, pos: 11
type: A, layer: 1, pos: 11
type: B, layer: 1, pos: 10
type: A, layer: 1, pos: 10
type: B, layer: 1, pos: 18
type: A, layer: 1, pos: 18
type: A, layer: 1, pos: 43
type: A, layer: 1, pos: 37
type: B, layer: 1, pos: 23
type: A, layer: 1, pos: 21
type: B, layer: 1, pos: 16
type: B, layer: 1, pos: 45
type: B, layer: 1, pos: 17
type: B, layer: 1, pos: 39
type: B, layer: 1, pos: 32
type: B, layer: 1, pos: 21
type: A, layer: 1, pos: 23
type: A, layer: 1, pos: 15
type: A, layer: 1, pos: 32
type: B, layer: 1, pos: 20
type: A, layer: 1, pos: 41
type: A, layer: 1, pos: 20
type: B, layer: 1, pos: 41
type: A, layer: 1, pos: 5
type: A, layer: 1, pos: 39
type: B, layer: 1, pos: 5
type: B, layer: 1, pos: 38
type: A, layer: 1, pos: 16
type: B, layer: 1, pos: 37
type: B, layer: 1, pos: 29

Time for candidate selection: 0.08 seconds

### Candidate
type: A, layer: 1, pos: 4

### Candidate
type: A, layer: 1, pos: 3

### Candidate
type: B, layer: 1, pos: 35

### Candidate
type: A, layer: 1, pos: 19

### Candidate
type: A, layer: 1, pos: 9

### Candidate
type: B, layer: 1, pos: 3

### Candidate
type: A, layer: 1, pos: 35

### Candidate
type: A, layer: 1, pos: 38

### Candidate
type: B, layer: 1, pos: 9

### Candidate
type: A, layer: 1, pos: 17

## Relational analysis of NS_B2_A2_A1_A1_B2_A1
Optimization infeasible because this subproblem isn't reachable.

### Candidate
type: B, layer: 1, pos: 43

## Relational analysis of NS_B2_A2_A1_A1_B2_B1
Optimization infeasible because this subproblem isn't reachable.

### Candidate
type: B, layer: 1, pos: 19

### Candidate
type: B, layer: 1, pos: 15

## Relational analysis of NS_B2_A2_A1_A1_B2_B1
Optimization infeasible because this subproblem isn't reachable.

### Candidate
type: B, layer: 1, pos: 4

### Candidate
type: A, layer: 1, pos: 45

## Relational analysis of NS_B2_A2_A1_A1_B2_A1

### Relational analysis result of NS_B2_A2_A1_A1_B2_A1
Status: Status.UNKNOWN
Output dim: 0, lower bound: -63.3939698, upper bound: 63.3947612
time: 0.69 seconds

## Relational analysis of NS_B2_A2_A1_A1_B2_A2

### Relational analysis result of NS_B2_A2_A1_A1_B2_A2
Status: Status.UNKNOWN
Output dim: 0, lower bound: -63.3939587, upper bound: 63.3947644
time: 0.78 seconds

## BFS NS instance: NS_B2_A2_A1_A2_B1

### Backsubstitution after applying NS history:
0: -41.4766655, 42.6777725, -52.4175224, 56.7414017, -98.1967850, 95.0952911
1: -280.1462708, 102.6841431, -384.0408325, 129.2115936, -409.3577881, 484.9897156
2: -161.1961517, 92.3508987, -206.3283691, 121.6101837, -282.3533936, 298.6792297
3: -197.2792816, 73.9040070, -265.2448730, 97.4538727, -294.7331543, 338.8872986
4: -117.7232513, 80.8190765, -148.4796600, 107.6116180, -224.5513153, 229.2987366

Time for backsubstitution: 0.98 seconds

### NS candidates at layer 1
type: A, layer: 1, pos: 4
type: A, layer: 1, pos: 9
type: A, layer: 1, pos: 19
type: B, layer: 1, pos: 35
type: A, layer: 1, pos: 3
type: A, layer: 1, pos: 17
type: B, layer: 1, pos: 9
type: A, layer: 1, pos: 35
type: A, layer: 1, pos: 45
type: B, layer: 1, pos: 10
type: B, layer: 1, pos: 11
type: A, layer: 1, pos: 11
type: B, layer: 1, pos: 19
type: B, layer: 1, pos: 15
type: B, layer: 1, pos: 17
type: A, layer: 1, pos: 38
type: A, layer: 1, pos: 18
type: B, layer: 1, pos: 3
type: B, layer: 1, pos: 4
type: A, layer: 1, pos: 10
type: B, layer: 1, pos: 29
type: B, layer: 1, pos: 16
type: A, layer: 1, pos: 37
type: B, layer: 1, pos: 39
type: B, layer: 1, pos: 32
type: B, layer: 1, pos: 45
type: B, layer: 1, pos: 18
type: A, layer: 1, pos: 15
type: A, layer: 1, pos: 32
type: A, layer: 1, pos: 21
type: A, layer: 1, pos: 39
type: B, layer: 1, pos: 20
type: A, layer: 1, pos: 41
type: A, layer: 1, pos: 43
type: B, layer: 1, pos: 43
type: B, layer: 1, pos: 41
type: A, layer: 1, pos: 5
type: B, layer: 1, pos: 23
type: A, layer: 1, pos: 23
type: B, layer: 1, pos: 5
type: B, layer: 1, pos: 21
type: A, layer: 1, pos: 20
type: B, layer: 1, pos: 37
type: B, layer: 1, pos: 38
type: A, layer: 1, pos: 16

Time for candidate selection: 0.08 seconds

### Candidate
type: A, layer: 1, pos: 4

### Candidate
type: A, layer: 1, pos: 9

### Candidate
type: A, layer: 1, pos: 19

### Candidate
type: B, layer: 1, pos: 35

### Candidate
type: A, layer: 1, pos: 3

### Candidate
type: A, layer: 1, pos: 17

## Relational analysis of NS_B2_A2_A1_A2_B1_A1
Optimization infeasible because this subproblem isn't reachable.

### Candidate
type: B, layer: 1, pos: 9

### Candidate
type: A, layer: 1, pos: 35

### Candidate
type: A, layer: 1, pos: 45

## Relational analysis of NS_B2_A2_A1_A2_B1_A1
Optimization infeasible because this subproblem isn't reachable.

### Candidate
type: B, layer: 1, pos: 10

## Relational analysis of NS_B2_A2_A1_A2_B1_B1
Optimization infeasible because this subproblem isn't reachable.

### Candidate
type: B, layer: 1, pos: 11

## Relational analysis of NS_B2_A2_A1_A2_B1_B1

### Relational analysis result of NS_B2_A2_A1_A2_B1_B1
Status: Status.UNKNOWN
Output dim: 0, lower bound: -63.3937079, upper bound: 63.3940319
time: 0.73 seconds

## Relational analysis of NS_B2_A2_A1_A2_B1_B2

### Relational analysis result of NS_B2_A2_A1_A2_B1_B2
Status: Status.UNKNOWN
Output dim: 0, lower bound: -63.3937826, upper bound: 63.3941691
time: 0.74 seconds

## BFS NS instance: NS_B2_A2_A1_A2_B2

### Backsubstitution after applying NS history:
0: -41.4766655, 42.6777725, -52.5505524, 57.1262398, -98.6028976, 95.2283249
1: -280.1462708, 102.6841431, -388.2085876, 129.6735077, -409.8197327, 490.8927307
2: -161.1961517, 92.3508987, -207.5194092, 122.3719025, -283.5680542, 299.8702698
3: -197.2792816, 73.9040070, -267.7059937, 98.1092606, -295.3885193, 341.6099854
4: -117.7232513, 80.8190765, -148.6719513, 108.3880539, -226.1112976, 229.4910278

Time for backsubstitution: 0.98 seconds

### NS candidates at layer 1
type: A, layer: 1, pos: 4
type: A, layer: 1, pos: 3
type: B, layer: 1, pos: 35
type: A, layer: 1, pos: 19
type: A, layer: 1, pos: 9
type: A, layer: 1, pos: 35
type: B, layer: 1, pos: 9
type: A, layer: 1, pos: 17
type: A, layer: 1, pos: 38
type: B, layer: 1, pos: 3
type: B, layer: 1, pos: 19
type: B, layer: 1, pos: 10
type: B, layer: 1, pos: 15
type: A, layer: 1, pos: 18
type: A, layer: 1, pos: 45
type: B, layer: 1, pos: 11
type: B, layer: 1, pos: 4
type: A, layer: 1, pos: 43
type: A, layer: 1, pos: 11
type: B, layer: 1, pos: 17
type: B, layer: 1, pos: 43
type: B, layer: 1, pos: 29
type: B, layer: 1, pos: 32
type: A, layer: 1, pos: 37
type: A, layer: 1, pos: 21
type: A, layer: 1, pos: 10
type: B, layer: 1, pos: 16
type: B, layer: 1, pos: 18
type: A, layer: 1, pos: 15
type: B, layer: 1, pos: 21
type: B, layer: 1, pos: 45
type: B, layer: 1, pos: 20
type: B, layer: 1, pos: 23
type: B, layer: 1, pos: 39
type: A, layer: 1, pos: 23
type: A, layer: 1, pos: 41
type: A, layer: 1, pos: 32
type: B, layer: 1, pos: 41
type: A, layer: 1, pos: 39
type: A, layer: 1, pos: 5
type: A, layer: 1, pos: 20
type: B, layer: 1, pos: 5
type: B, layer: 1, pos: 38
type: B, layer: 1, pos: 37
type: A, layer: 1, pos: 16

Time for candidate selection: 0.08 seconds

### Candidate
type: A, layer: 1, pos: 4

### Candidate
type: A, layer: 1, pos: 3

### Candidate
type: B, layer: 1, pos: 35

### Candidate
type: A, layer: 1, pos: 19

### Candidate
type: A, layer: 1, pos: 9

### Candidate
type: A, layer: 1, pos: 35

### Candidate
type: B, layer: 1, pos: 9

### Candidate
type: A, layer: 1, pos: 17

## Relational analysis of NS_B2_A2_A1_A2_B2_A1
Optimization infeasible because this subproblem isn't reachable.

### Candidate
type: A, layer: 1, pos: 38

## Relational analysis of NS_B2_A2_A1_A2_B2_A1
Optimization infeasible because this subproblem isn't reachable.

### Candidate
type: B, layer: 1, pos: 3

### Candidate
type: B, layer: 1, pos: 19

### Candidate
type: B, layer: 1, pos: 10

## Relational analysis of NS_B2_A2_A1_A2_B2_B1
Optimization infeasible because this subproblem isn't reachable.

### Candidate
type: B, layer: 1, pos: 15

## Relational analysis of NS_B2_A2_A1_A2_B2_B1
Optimization infeasible because this subproblem isn't reachable.

### Candidate
type: A, layer: 1, pos: 18

## Relational analysis of NS_B2_A2_A1_A2_B2_A1
Optimization infeasible because this subproblem isn't reachable.

### Candidate
type: A, layer: 1, pos: 45

## Relational analysis of NS_B2_A2_A1_A2_B2_A1
Optimization infeasible because this subproblem isn't reachable.

### Candidate
type: B, layer: 1, pos: 11

## Relational analysis of NS_B2_A2_A1_A2_B2_B1

### Relational analysis result of NS_B2_A2_A1_A2_B2_B1
Status: Status.UNKNOWN
Output dim: 0, lower bound: -63.3937079, upper bound: 63.3941582
time: 0.76 seconds

## Relational analysis of NS_B2_A2_A1_A2_B2_B2

### Relational analysis result of NS_B2_A2_A1_A2_B2_B2
Status: Status.UNKNOWN
Output dim: 0, lower bound: -63.3937826, upper bound: 63.3947862
time: 0.76 seconds

## BFS NS instance: NS_B2_A2_A2_B1_A1

### Backsubstitution after applying NS history:
0: -54.1881714, 58.9965973, -53.8769455, 58.3707047, -112.5588760, 112.8735428
1: -402.2130432, 133.8377991, -396.0427246, 132.8756409, -534.4114380, 528.9307251
2: -214.3997955, 126.4415131, -212.2888184, 125.1844635, -339.5842590, 338.7303467
3: -277.1029663, 101.3699265, -273.3699341, 100.2912750, -376.7356262, 373.8124695
4: -153.4725342, 111.9196091, -152.6836853, 110.7171249, -264.1896667, 264.6033020

Time for backsubstitution: 1.00 seconds

### NS candidates at layer 1
type: A, layer: 1, pos: 35
type: B, layer: 1, pos: 35
type: A, layer: 1, pos: 9
type: B, layer: 1, pos: 9
type: B, layer: 1, pos: 19
type: A, layer: 1, pos: 19
type: B, layer: 1, pos: 3
type: A, layer: 1, pos: 3
type: B, layer: 1, pos: 17
type: A, layer: 1, pos: 17
type: A, layer: 1, pos: 10
type: B, layer: 1, pos: 18
type: B, layer: 1, pos: 10
type: B, layer: 1, pos: 45
type: A, layer: 1, pos: 15
type: B, layer: 1, pos: 43
type: A, layer: 1, pos: 18
type: B, layer: 1, pos: 15
type: A, layer: 1, pos: 45
type: A, layer: 1, pos: 29
type: A, layer: 1, pos: 43
type: B, layer: 1, pos: 11
type: B, layer: 1, pos: 29
type: A, layer: 1, pos: 21
type: A, layer: 1, pos: 23
type: B, layer: 1, pos: 21
type: B, layer: 1, pos: 4
type: B, layer: 1, pos: 37
type: A, layer: 1, pos: 32
type: B, layer: 1, pos: 23
type: B, layer: 1, pos: 38
type: B, layer: 1, pos: 20
type: B, layer: 1, pos: 32
type: A, layer: 1, pos: 39
type: A, layer: 1, pos: 37
type: B, layer: 1, pos: 41
type: A, layer: 1, pos: 41
type: A, layer: 1, pos: 20
type: B, layer: 1, pos: 39
type: A, layer: 1, pos: 16
type: B, layer: 1, pos: 5
type: B, layer: 1, pos: 16
type: A, layer: 1, pos: 5
type: A, layer: 1, pos: 4
type: A, layer: 1, pos: 38

Time for candidate selection: 0.08 seconds

### Candidate
type: A, layer: 1, pos: 35

### Candidate
type: B, layer: 1, pos: 35

### Candidate
type: A, layer: 1, pos: 9

### Candidate
type: B, layer: 1, pos: 9

### Candidate
type: B, layer: 1, pos: 19

### Candidate
type: A, layer: 1, pos: 19

### Candidate
type: B, layer: 1, pos: 3

### Candidate
type: A, layer: 1, pos: 3

### Candidate
type: B, layer: 1, pos: 17

### Candidate
type: A, layer: 1, pos: 17

### Candidate
type: A, layer: 1, pos: 10

### Candidate
type: B, layer: 1, pos: 18

### Candidate
type: B, layer: 1, pos: 10

### Candidate
type: B, layer: 1, pos: 45

### Candidate
type: A, layer: 1, pos: 15

## Relational analysis of NS_B2_A2_A2_B1_A1_A1
Optimization infeasible because this subproblem isn't reachable.

### Candidate
type: B, layer: 1, pos: 43

## Relational analysis of NS_B2_A2_A2_B1_A1_B1
Optimization infeasible because this subproblem isn't reachable.

### Candidate
type: A, layer: 1, pos: 18

### Candidate
type: B, layer: 1, pos: 15

### Candidate
type: A, layer: 1, pos: 45

### Candidate
type: A, layer: 1, pos: 29

## Relational analysis of NS_B2_A2_A2_B1_A1_A1
Optimization infeasible because this subproblem isn't reachable.

### Candidate
type: A, layer: 1, pos: 43

### Candidate
type: B, layer: 1, pos: 11

## Relational analysis of NS_B2_A2_A2_B1_A1_B1

### Relational analysis result of NS_B2_A2_A2_B1_A1_B1
Status: Status.UNKNOWN
Output dim: 0, lower bound: -63.3930163, upper bound: 63.3930511
time: 0.78 seconds

## Relational analysis of NS_B2_A2_A2_B1_A1_B2

### Relational analysis result of NS_B2_A2_A2_B1_A1_B2
Status: Status.UNKNOWN
Output dim: 0, lower bound: -63.3930163, upper bound: 63.3931718
time: 1.05 seconds

## BFS NS instance: NS_B2_A2_A2_B1_A2

### Backsubstitution after applying NS history:
0: -53.8847351, 58.6575432, -53.9753265, 58.4763908, -112.3611069, 112.6328735
1: -399.7374878, 133.0946503, -396.8188477, 133.1195221, -532.2844849, 529.0166626
2: -213.1640167, 125.7145767, -212.6795044, 125.4174194, -338.5814209, 338.3940735
3: -275.4341431, 100.8090820, -273.8961182, 100.4767838, -375.3064575, 373.8242798
4: -152.5790253, 111.2873230, -152.9559479, 110.9175262, -263.4965515, 264.2432556

Time for backsubstitution: 0.99 seconds

### NS candidates at layer 1
type: A, layer: 1, pos: 35
type: B, layer: 1, pos: 35
type: A, layer: 1, pos: 9
type: B, layer: 1, pos: 9
type: B, layer: 1, pos: 19
type: A, layer: 1, pos: 19
type: B, layer: 1, pos: 3
type: A, layer: 1, pos: 3
type: B, layer: 1, pos: 17
type: A, layer: 1, pos: 17
type: A, layer: 1, pos: 10
type: B, layer: 1, pos: 18
type: B, layer: 1, pos: 10
type: B, layer: 1, pos: 45
type: A, layer: 1, pos: 15
type: B, layer: 1, pos: 43
type: B, layer: 1, pos: 11
type: A, layer: 1, pos: 18
type: B, layer: 1, pos: 15
type: A, layer: 1, pos: 45
type: A, layer: 1, pos: 29
type: A, layer: 1, pos: 43
type: B, layer: 1, pos: 29
type: A, layer: 1, pos: 21
type: A, layer: 1, pos: 23
type: B, layer: 1, pos: 21
type: B, layer: 1, pos: 37
type: A, layer: 1, pos: 32
type: B, layer: 1, pos: 4
type: B, layer: 1, pos: 23
type: B, layer: 1, pos: 38
type: A, layer: 1, pos: 39
type: B, layer: 1, pos: 20
type: B, layer: 1, pos: 32
type: A, layer: 1, pos: 37
type: B, layer: 1, pos: 41
type: A, layer: 1, pos: 41
type: A, layer: 1, pos: 20
type: B, layer: 1, pos: 39
type: A, layer: 1, pos: 16
type: B, layer: 1, pos: 5
type: A, layer: 1, pos: 5
type: B, layer: 1, pos: 16
type: A, layer: 1, pos: 4
type: A, layer: 1, pos: 38

Time for candidate selection: 0.08 seconds

### Candidate
type: A, layer: 1, pos: 35

### Candidate
type: B, layer: 1, pos: 35

### Candidate
type: A, layer: 1, pos: 9

### Candidate
type: B, layer: 1, pos: 9

### Candidate
type: B, layer: 1, pos: 19

### Candidate
type: A, layer: 1, pos: 19

### Candidate
type: B, layer: 1, pos: 3

### Candidate
type: A, layer: 1, pos: 3

### Candidate
type: B, layer: 1, pos: 17

### Candidate
type: A, layer: 1, pos: 17

### Candidate
type: A, layer: 1, pos: 10

### Candidate
type: B, layer: 1, pos: 18

### Candidate
type: B, layer: 1, pos: 10

### Candidate
type: B, layer: 1, pos: 45

### Candidate
type: A, layer: 1, pos: 15

## Relational analysis of NS_B2_A2_A2_B1_A2_A1
Optimization infeasible because this subproblem isn't reachable.

### Candidate
type: B, layer: 1, pos: 43

## Relational analysis of NS_B2_A2_A2_B1_A2_B1
Optimization infeasible because this subproblem isn't reachable.

### Candidate
type: B, layer: 1, pos: 11

## Relational analysis of NS_B2_A2_A2_B1_A2_B1

### Relational analysis result of NS_B2_A2_A2_B1_A2_B1
Status: Status.UNKNOWN
Output dim: 0, lower bound: -63.3934490, upper bound: 63.3937162
time: 0.69 seconds

## Relational analysis of NS_B2_A2_A2_B1_A2_B2

### Relational analysis result of NS_B2_A2_A2_B1_A2_B2
Status: Status.UNKNOWN
Output dim: 0, lower bound: -63.3934490, upper bound: 63.3938446
time: 0.72 seconds

## BFS NS instance: NS_B2_A2_A2_B2_A1

### Backsubstitution after applying NS history:
0: -54.1881714, 58.9965973, -53.9260216, 58.6826210, -112.8707886, 112.9226151
1: -402.2130432, 133.8377991, -399.8506775, 133.1753235, -535.3883667, 533.6884155
2: -214.3997955, 126.4415131, -213.3021240, 125.7830124, -340.1828003, 339.7436523
3: -277.1029663, 101.3699265, -275.5415955, 100.8500671, -377.9530334, 376.9114380
4: -153.4725342, 111.9196091, -152.6899719, 111.3419495, -264.8144836, 264.6095886

Time for backsubstitution: 0.99 seconds

### NS candidates at layer 1
type: B, layer: 1, pos: 35
type: A, layer: 1, pos: 35
type: A, layer: 1, pos: 9
type: B, layer: 1, pos: 9
type: B, layer: 1, pos: 19
type: A, layer: 1, pos: 19
type: B, layer: 1, pos: 3
type: A, layer: 1, pos: 3
type: B, layer: 1, pos: 17
type: A, layer: 1, pos: 17
type: B, layer: 1, pos: 10
type: A, layer: 1, pos: 10
type: B, layer: 1, pos: 18
type: A, layer: 1, pos: 18
type: A, layer: 1, pos: 45
type: A, layer: 1, pos: 15
type: B, layer: 1, pos: 43
type: B, layer: 1, pos: 15
type: B, layer: 1, pos: 45
type: A, layer: 1, pos: 43
type: A, layer: 1, pos: 29
type: B, layer: 1, pos: 29
type: B, layer: 1, pos: 11
type: A, layer: 1, pos: 21
type: B, layer: 1, pos: 21
type: B, layer: 1, pos: 23
type: A, layer: 1, pos: 23
type: B, layer: 1, pos: 37
type: B, layer: 1, pos: 32
type: A, layer: 1, pos: 32
type: A, layer: 1, pos: 37
type: B, layer: 1, pos: 20
type: B, layer: 1, pos: 39
type: A, layer: 1, pos: 20
type: B, layer: 1, pos: 41
type: A, layer: 1, pos: 39
type: A, layer: 1, pos: 41
type: B, layer: 1, pos: 38
type: A, layer: 1, pos: 38
type: B, layer: 1, pos: 16
type: A, layer: 1, pos: 5
type: A, layer: 1, pos: 16
type: B, layer: 1, pos: 5
type: B, layer: 1, pos: 4
type: A, layer: 1, pos: 4

Time for candidate selection: 0.08 seconds

### Candidate
type: B, layer: 1, pos: 35

### Candidate
type: A, layer: 1, pos: 35

### Candidate
type: A, layer: 1, pos: 9

### Candidate
type: B, layer: 1, pos: 9

### Candidate
type: B, layer: 1, pos: 19

### Candidate
type: A, layer: 1, pos: 19

### Candidate
type: B, layer: 1, pos: 3

### Candidate
type: A, layer: 1, pos: 3

### Candidate
type: B, layer: 1, pos: 17

### Candidate
type: A, layer: 1, pos: 17

### Candidate
type: B, layer: 1, pos: 10

### Candidate
type: A, layer: 1, pos: 10

### Candidate
type: B, layer: 1, pos: 18

## Relational analysis of NS_B2_A2_A2_B2_A1_B1
Optimization infeasible because this subproblem isn't reachable.

### Candidate
type: A, layer: 1, pos: 18

### Candidate
type: A, layer: 1, pos: 45

## Relational analysis of NS_B2_A2_A2_B2_A1_A1
Optimization infeasible because this subproblem isn't reachable.

### Candidate
type: A, layer: 1, pos: 15

## Relational analysis of NS_B2_A2_A2_B2_A1_A1

### Relational analysis result of NS_B2_A2_A2_B2_A1_A1
Status: Status.VERIFIED
Output dim: 0, lower bound: -63.3862433, upper bound: 63.3898573
time: 0.77 seconds

## Relational analysis of NS_B2_A2_A2_B2_A1_A2

### Relational analysis result of NS_B2_A2_A2_B2_A1_A2
Status: Status.UNKNOWN
Output dim: 0, lower bound: -63.3930372, upper bound: 63.3930663
time: 0.79 seconds

## BFS NS instance: NS_B2_A2_A2_B2_A2

### Backsubstitution after applying NS history:
0: -53.8847351, 58.6575432, -54.0260849, 58.7897339, -112.6744690, 112.6836090
1: -399.7374878, 133.0946503, -400.6096802, 133.4270172, -533.1644287, 533.7043457
2: -213.1640167, 125.7145767, -213.7019806, 126.0158386, -339.1798096, 339.4165344
3: -275.4341431, 100.8090820, -276.0601196, 101.0406876, -376.4747925, 376.8692017
4: -152.5790253, 111.2873230, -152.9686127, 111.5431519, -264.1221313, 264.2558899

Time for backsubstitution: 0.99 seconds

### NS candidates at layer 1
type: A, layer: 1, pos: 35
type: B, layer: 1, pos: 35
type: A, layer: 1, pos: 9
type: B, layer: 1, pos: 9
type: B, layer: 1, pos: 19
type: A, layer: 1, pos: 19
type: B, layer: 1, pos: 3
type: A, layer: 1, pos: 3
type: A, layer: 1, pos: 17
type: B, layer: 1, pos: 17
type: B, layer: 1, pos: 10
type: A, layer: 1, pos: 10
type: B, layer: 1, pos: 18
type: A, layer: 1, pos: 18
type: B, layer: 1, pos: 11
type: A, layer: 1, pos: 15
type: B, layer: 1, pos: 45
type: A, layer: 1, pos: 45
type: B, layer: 1, pos: 15
type: B, layer: 1, pos: 43
type: A, layer: 1, pos: 43
type: B, layer: 1, pos: 29
type: A, layer: 1, pos: 29
type: A, layer: 1, pos: 21
type: B, layer: 1, pos: 21
type: B, layer: 1, pos: 23
type: A, layer: 1, pos: 23
type: A, layer: 1, pos: 32
type: B, layer: 1, pos: 37
type: B, layer: 1, pos: 32
type: A, layer: 1, pos: 37
type: A, layer: 1, pos: 20
type: B, layer: 1, pos: 20
type: B, layer: 1, pos: 39
type: A, layer: 1, pos: 39
type: B, layer: 1, pos: 41
type: A, layer: 1, pos: 41
type: B, layer: 1, pos: 38
type: A, layer: 1, pos: 38
type: B, layer: 1, pos: 16
type: A, layer: 1, pos: 16
type: A, layer: 1, pos: 5
type: B, layer: 1, pos: 5
type: B, layer: 1, pos: 4
type: A, layer: 1, pos: 4

Time for candidate selection: 0.08 seconds

### Candidate
type: A, layer: 1, pos: 35

### Candidate
type: B, layer: 1, pos: 35

### Candidate
type: A, layer: 1, pos: 9

### Candidate
type: B, layer: 1, pos: 9

### Candidate
type: B, layer: 1, pos: 19

### Candidate
type: A, layer: 1, pos: 19

### Candidate
type: B, layer: 1, pos: 3

### Candidate
type: A, layer: 1, pos: 3

### Candidate
type: A, layer: 1, pos: 17

### Candidate
type: B, layer: 1, pos: 17

### Candidate
type: B, layer: 1, pos: 10

### Candidate
type: A, layer: 1, pos: 10

### Candidate
type: B, layer: 1, pos: 18

## Relational analysis of NS_B2_A2_A2_B2_A2_B1

### Relational analysis result of NS_B2_A2_A2_B2_A2_B1
Status: Status.UNKNOWN
Output dim: 0, lower bound: -63.3941377, upper bound: 63.3941202
time: 0.85 seconds

## Relational analysis of NS_B2_A2_A2_B2_A2_B2
Optimization infeasible because this subproblem isn't reachable.

### Candidate
type: A, layer: 1, pos: 18

### Candidate
type: B, layer: 1, pos: 11

## Relational analysis of NS_B2_A2_A2_B2_A2_B1

### Relational analysis result of NS_B2_A2_A2_B2_A2_B1
Status: Status.UNKNOWN
Output dim: 0, lower bound: -63.3930551, upper bound: 63.3939188
time: 0.91 seconds

## Relational analysis of NS_B2_A2_A2_B2_A2_B2

### Relational analysis result of NS_B2_A2_A2_B2_A2_B2
Status: Status.UNKNOWN
Output dim: 0, lower bound: -63.3936129, upper bound: 63.3948172
time: 0.87 seconds

## Summary of splitting at layer (split count: 5)
- Time for NS candidates: 4.09 seconds
NS_B1_B1_A1_A1_B1_A1, status: Status.UNKNOWN, split count: 6, time: 4.09
Output dim: 0, lower bound: -63.3963898, upper bound: 63.3963898
NS_B1_B1_A1_A1_B1_A2, status: Status.UNKNOWN, split count: 6, time: 4.09
Output dim: 0, lower bound: -63.3963898, upper bound: 63.3963898
NS_B1_B1_A1_A1_B2_A1, status: Status.UNKNOWN, split count: 6, time: 4.09
Output dim: 0, lower bound: -63.3963898, upper bound: 63.3963898
NS_B1_B1_A1_A1_B2_A2, status: Status.UNKNOWN, split count: 6, time: 4.09
Output dim: 0, lower bound: -63.3963898, upper bound: 63.3963898
NS_B1_B1_A1_A2_B1_B1, status: Status.UNKNOWN, split count: 6, time: 4.09
Output dim: 0, lower bound: -63.3958702, upper bound: 63.3958122
NS_B1_B1_A1_A2_B1_B2, status: Status.UNKNOWN, split count: 6, time: 4.09
Output dim: 0, lower bound: -63.3958754, upper bound: 63.3958122
NS_B1_B1_A1_A2_B2_A1, status: Status.UNKNOWN, split count: 6, time: 4.09
Output dim: 0, lower bound: -63.3958450, upper bound: 63.3957590
NS_B1_B1_A1_A2_B2_A2, status: Status.UNKNOWN, split count: 6, time: 4.09
Output dim: 0, lower bound: -63.3958450, upper bound: 63.3957590
NS_B1_B1_A2_B1_B1_A1, status: Status.UNKNOWN, split count: 6, time: 4.09
Output dim: 0, lower bound: -63.3942257, upper bound: 63.3940786
NS_B1_B1_A2_B1_B1_A2, status: Status.UNKNOWN, split count: 6, time: 4.09
Output dim: 0, lower bound: -63.3942257, upper bound: 63.3942986
NS_B1_B1_A2_B1_B2_A1, status: Status.UNKNOWN, split count: 6, time: 4.09
Output dim: 0, lower bound: -63.3939945, upper bound: 63.3939685
NS_B1_B1_A2_B1_B2_A2, status: Status.UNKNOWN, split count: 6, time: 4.09
Output dim: 0, lower bound: -63.3939945, upper bound: 63.3941425
NS_B1_B1_A2_B2_B1_A1, status: Status.UNKNOWN, split count: 6, time: 4.09
Output dim: 0, lower bound: -63.3940763, upper bound: 63.3939509
NS_B1_B1_A2_B2_B1_A2, status: Status.UNKNOWN, split count: 6, time: 4.09
Output dim: 0, lower bound: -63.3940763, upper bound: 63.3943366
NS_B1_B1_A2_B2_B2_A1, status: Status.UNKNOWN, split count: 6, time: 4.09
Output dim: 0, lower bound: -63.3939964, upper bound: 63.3939342
NS_B1_B1_A2_B2_B2_A2, status: Status.UNKNOWN, split count: 6, time: 4.09
Output dim: 0, lower bound: -63.3939964, upper bound: 63.3943223
NS_B1_B2_A1_A1_A1_A1, status: Status.UNKNOWN, split count: 6, time: 4.09
Output dim: 0, lower bound: -63.3958122, upper bound: 63.3958702
NS_B1_B2_A1_A1_A1_A2, status: Status.UNKNOWN, split count: 6, time: 4.09
Output dim: 0, lower bound: -63.3958122, upper bound: 63.3958754
NS_B1_B2_A1_A1_A2_B1, status: Status.UNKNOWN, split count: 6, time: 4.09
Output dim: 0, lower bound: -63.3957590, upper bound: 63.3958450
NS_B1_B2_A1_A1_A2_B2, status: Status.UNKNOWN, split count: 6, time: 4.09
Output dim: 0, lower bound: -63.3957590, upper bound: 63.3958450
NS_B1_B2_A1_A2_B1_B1, status: Status.UNKNOWN, split count: 6, time: 4.09
Output dim: 0, lower bound: -63.3938505, upper bound: 63.3932262
NS_B1_B2_A1_A2_B1_B2, status: Status.UNKNOWN, split count: 6, time: 4.09
Output dim: 0, lower bound: -63.3949434, upper bound: 63.3947308
NS_B1_B2_A1_A2_B2_A1, status: Status.UNKNOWN, split count: 6, time: 4.09
Output dim: 0, lower bound: -63.3954598, upper bound: 63.3954598
NS_B1_B2_A1_A2_B2_A2, status: Status.UNKNOWN, split count: 6, time: 4.09
Output dim: 0, lower bound: -63.3954598, upper bound: 63.3954598
NS_B1_B2_A2_B1_B1_B1, status: Status.UNKNOWN, split count: 6, time: 4.09
Output dim: 0, lower bound: -63.3936396, upper bound: 63.3940085
NS_B1_B2_A2_B1_B1_B2, status: Status.UNKNOWN, split count: 6, time: 4.09
Output dim: 0, lower bound: -63.3939212, upper bound: 63.3941125
NS_B1_B2_A2_B1_B2_A1, status: Status.UNKNOWN, split count: 6, time: 4.09
Output dim: 0, lower bound: -63.3936783, upper bound: 63.3935557
NS_B1_B2_A2_B1_B2_A2, status: Status.UNKNOWN, split count: 6, time: 4.09
Output dim: 0, lower bound: -63.3936783, upper bound: 63.3935634
NS_B1_B2_A2_B2_B1_B1, status: Status.UNKNOWN, split count: 6, time: 4.09
Output dim: 0, lower bound: -63.3940453, upper bound: 63.3941899
NS_B1_B2_A2_B2_B1_B2, status: Status.UNKNOWN, split count: 6, time: 4.09
Output dim: 0, lower bound: -63.3931911, upper bound: 63.3933291
NS_B1_B2_A2_B2_B2_B1, status: Status.UNKNOWN, split count: 6, time: 4.09
Output dim: 0, lower bound: -63.3940682, upper bound: 63.3941989
NS_B1_B2_A2_B2_B2_B2, status: Status.UNKNOWN, split count: 6, time: 4.09
Output dim: 0, lower bound: -63.3937801, upper bound: 63.3937700
NS_B2_A1_A1_B1_A1_A1, status: Status.UNKNOWN, split count: 6, time: 4.09
Output dim: 0, lower bound: -63.3940047, upper bound: 63.3945002
NS_B2_A1_A1_B1_A1_A2, status: Status.UNKNOWN, split count: 6, time: 4.09
Output dim: 0, lower bound: -63.3940047, upper bound: 63.3944632
NS_B2_A1_A1_B1_A2_A1, status: Status.UNKNOWN, split count: 6, time: 4.09
Output dim: 0, lower bound: -63.3935219, upper bound: 63.3938497
NS_B2_A1_A1_B1_A2_A2, status: Status.UNKNOWN, split count: 6, time: 4.09
Output dim: 0, lower bound: -63.3939658, upper bound: 63.3943218
NS_B2_A1_A1_B2_A1_A1, status: Status.UNKNOWN, split count: 6, time: 4.09
Output dim: 0, lower bound: -63.3940047, upper bound: 63.3945531
NS_B2_A1_A1_B2_A1_A2, status: Status.UNKNOWN, split count: 6, time: 4.09
Output dim: 0, lower bound: -63.3942132, upper bound: 63.3945397
NS_B2_A1_A1_B2_A2_B1, status: Status.UNKNOWN, split count: 6, time: 4.09
Output dim: 0, lower bound: -63.3939171, upper bound: 63.3939666
NS_B2_A1_A1_B2_A2_B2, status: Status.UNKNOWN, split count: 6, time: 4.09
Output dim: 0, lower bound: -63.3941648, upper bound: 63.3942476
NS_B2_A1_A2_B1_A1_B1, status: Status.UNKNOWN, split count: 6, time: 4.09
Output dim: 0, lower bound: -63.3929322, upper bound: 63.3929322
NS_B2_A1_A2_B1_A1_B2, status: Status.UNKNOWN, split count: 6, time: 4.09
Output dim: 0, lower bound: -63.3929322, upper bound: 63.3930551
NS_B2_A1_A2_B1_A2_B1, status: Status.UNKNOWN, split count: 6, time: 4.09
Output dim: 0, lower bound: -63.3930551, upper bound: 63.3930077
NS_B2_A1_A2_B1_A2_B2, status: Status.UNKNOWN, split count: 6, time: 4.09
Output dim: 0, lower bound: -63.3930551, upper bound: 63.3931305
NS_B2_A1_A2_B2_B1_A1, status: Status.UNKNOWN, split count: 6, time: 4.09
Output dim: 0, lower bound: -63.3930511, upper bound: 63.3930163
NS_B2_A1_A2_B2_B1_A2, status: Status.UNKNOWN, split count: 6, time: 4.09
Output dim: 0, lower bound: -63.3930511, upper bound: 63.3930898
NS_B2_A1_A2_B2_B2_A1, status: Status.UNKNOWN, split count: 6, time: 4.09
Output dim: 0, lower bound: -63.3937162, upper bound: 63.3934490
NS_B2_A1_A2_B2_B2_A2, status: Status.UNKNOWN, split count: 6, time: 4.09
Output dim: 0, lower bound: -63.3937162, upper bound: 63.3935402
NS_B2_A2_A1_A1_B1_A1, status: Status.UNKNOWN, split count: 6, time: 4.09
Output dim: 0, lower bound: -63.3939698, upper bound: 63.3944786
NS_B2_A2_A1_A1_B1_A2, status: Status.UNKNOWN, split count: 6, time: 4.09
Output dim: 0, lower bound: -63.3939587, upper bound: 63.3944746
NS_B2_A2_A1_A1_B2_A1, status: Status.UNKNOWN, split count: 6, time: 4.09
Output dim: 0, lower bound: -63.3939698, upper bound: 63.3947612
NS_B2_A2_A1_A1_B2_A2, status: Status.UNKNOWN, split count: 6, time: 4.09
Output dim: 0, lower bound: -63.3939587, upper bound: 63.3947644
NS_B2_A2_A1_A2_B1_B1, status: Status.UNKNOWN, split count: 6, time: 4.09
Output dim: 0, lower bound: -63.3937079, upper bound: 63.3940319
NS_B2_A2_A1_A2_B1_B2, status: Status.UNKNOWN, split count: 6, time: 4.09
Output dim: 0, lower bound: -63.3937826, upper bound: 63.3941691
NS_B2_A2_A1_A2_B2_B1, status: Status.UNKNOWN, split count: 6, time: 4.09
Output dim: 0, lower bound: -63.3937079, upper bound: 63.3941582
NS_B2_A2_A1_A2_B2_B2, status: Status.UNKNOWN, split count: 6, time: 4.09
Output dim: 0, lower bound: -63.3937826, upper bound: 63.3947862
NS_B2_A2_A2_B1_A1_B1, status: Status.UNKNOWN, split count: 6, time: 4.09
Output dim: 0, lower bound: -63.3930163, upper bound: 63.3930511
NS_B2_A2_A2_B1_A1_B2, status: Status.UNKNOWN, split count: 6, time: 4.09
Output dim: 0, lower bound: -63.3930163, upper bound: 63.3931718
NS_B2_A2_A2_B1_A2_B1, status: Status.UNKNOWN, split count: 6, time: 4.09
Output dim: 0, lower bound: -63.3934490, upper bound: 63.3937162
NS_B2_A2_A2_B1_A2_B2, status: Status.UNKNOWN, split count: 6, time: 4.09
Output dim: 0, lower bound: -63.3934490, upper bound: 63.3938446
NS_B2_A2_A2_B2_A1_A1, status: Status.VERIFIED, split count: 6, time: 4.09
Output dim: 0, lower bound: -63.3862433, upper bound: 63.3898573
NS_B2_A2_A2_B2_A1_A2, status: Status.UNKNOWN, split count: 6, time: 4.09
Output dim: 0, lower bound: -63.3930372, upper bound: 63.3930663
NS_B2_A2_A2_B2_A2_B1, status: Status.UNKNOWN, split count: 6, time: 4.09
Output dim: 0, lower bound: -63.3930551, upper bound: 63.3939188
NS_B2_A2_A2_B2_A2_B2, status: Status.UNKNOWN, split count: 6, time: 4.09
Output dim: 0, lower bound: -63.3936129, upper bound: 63.3948172

## BFS NS instance: NS_B1_B1_A1_A1_B1_A1

### Backsubstitution after applying NS history:
0: -31.2985306, 32.7712555, -31.2985306, 32.7712555, -64.0697784, 64.0697784
1: -221.7436371, 77.5785599, -221.7436371, 77.5785599, -299.3221741, 299.3221741
2: -123.9428406, 70.7823410, -123.9428406, 70.7823410, -194.7251892, 194.7251892
3: -155.0498047, 56.7560463, -155.0498047, 56.7560463, -211.8058472, 211.8058472
4: -89.2801895, 61.7543640, -89.2801895, 61.7543640, -151.0345459, 151.0345459

Time for backsubstitution: 1.14 seconds

### NS candidates at layer 1
type: A, layer: 1, pos: 9
type: B, layer: 1, pos: 9
type: A, layer: 1, pos: 35
type: B, layer: 1, pos: 35
type: A, layer: 1, pos: 17
type: B, layer: 1, pos: 17
type: B, layer: 1, pos: 19
type: A, layer: 1, pos: 19
type: B, layer: 1, pos: 45
type: A, layer: 1, pos: 45
type: B, layer: 1, pos: 3
type: A, layer: 1, pos: 3
type: B, layer: 1, pos: 11
type: A, layer: 1, pos: 11
type: A, layer: 1, pos: 10
type: B, layer: 1, pos: 10
type: B, layer: 1, pos: 15
type: A, layer: 1, pos: 15
type: A, layer: 1, pos: 18
type: B, layer: 1, pos: 18
type: B, layer: 1, pos: 39
type: A, layer: 1, pos: 39
type: A, layer: 1, pos: 32
type: B, layer: 1, pos: 32
type: A, layer: 1, pos: 43
type: B, layer: 1, pos: 43
type: B, layer: 1, pos: 38
type: A, layer: 1, pos: 38
type: A, layer: 1, pos: 37
type: B, layer: 1, pos: 37
type: B, layer: 1, pos: 29
type: A, layer: 1, pos: 29
type: A, layer: 1, pos: 41
type: B, layer: 1, pos: 41
type: A, layer: 1, pos: 20
type: B, layer: 1, pos: 20
type: A, layer: 1, pos: 21
type: B, layer: 1, pos: 21
type: B, layer: 1, pos: 5
type: A, layer: 1, pos: 5
type: B, layer: 1, pos: 23
type: A, layer: 1, pos: 23
type: A, layer: 1, pos: 16
type: B, layer: 1, pos: 16

Time for candidate selection: 0.08 seconds

### Candidate
type: A, layer: 1, pos: 9

### Candidate
type: B, layer: 1, pos: 9

### Candidate
type: A, layer: 1, pos: 35

## Relational analysis of NS_B1_B1_A1_A1_B1_A1_A1
Optimization infeasible because this subproblem isn't reachable.

### Candidate
type: B, layer: 1, pos: 35

### Candidate
type: A, layer: 1, pos: 17

## Relational analysis of NS_B1_B1_A1_A1_B1_A1_A1
Optimization infeasible because this subproblem isn't reachable.

### Candidate
type: B, layer: 1, pos: 17

### Candidate
type: B, layer: 1, pos: 19

## Relational analysis of NS_B1_B1_A1_A1_B1_A1_B1
Optimization infeasible because this subproblem isn't reachable.

### Candidate
type: A, layer: 1, pos: 19

### Candidate
type: B, layer: 1, pos: 45

## Relational analysis of NS_B1_B1_A1_A1_B1_A1_B1
Optimization infeasible because this subproblem isn't reachable.

### Candidate
type: A, layer: 1, pos: 45

### Candidate
type: B, layer: 1, pos: 3

## Relational analysis of NS_B1_B1_A1_A1_B1_A1_B1

### Relational analysis result of NS_B1_B1_A1_A1_B1_A1_B1
Status: Status.UNKNOWN
Output dim: 0, lower bound: -63.3964140, upper bound: 63.3964064
time: 0.87 seconds

## Relational analysis of NS_B1_B1_A1_A1_B1_A1_B2

### Relational analysis result of NS_B1_B1_A1_A1_B1_A1_B2
Status: Status.UNKNOWN
Output dim: 0, lower bound: -63.3964120, upper bound: 63.3964073
time: 0.76 seconds

## BFS NS instance: NS_B1_B1_A1_A1_B1_A2

### Backsubstitution after applying NS history:
0: -43.3717232, 44.9401779, -31.2985306, 32.7712555, -76.1429749, 76.2387085
1: -303.9672852, 107.6393127, -221.7436371, 77.5785599, -381.0203857, 329.3828735
2: -170.4709625, 98.0167847, -123.9428406, 70.7823410, -240.5885010, 221.9596252
3: -213.3660736, 78.5642319, -155.0498047, 56.7560463, -268.8029480, 233.6140442
4: -123.1605453, 85.3021240, -89.2801895, 61.7543640, -184.5613861, 174.5823059

Time for backsubstitution: 0.97 seconds

### NS candidates at layer 1
type: B, layer: 1, pos: 29
type: A, layer: 1, pos: 9
type: B, layer: 1, pos: 17
type: B, layer: 1, pos: 9
type: B, layer: 1, pos: 35
type: A, layer: 1, pos: 35
type: B, layer: 1, pos: 3
type: A, layer: 1, pos: 19
type: B, layer: 1, pos: 10
type: A, layer: 1, pos: 45
type: B, layer: 1, pos: 38
type: B, layer: 1, pos: 19
type: A, layer: 1, pos: 18
type: B, layer: 1, pos: 45
type: B, layer: 1, pos: 11
type: A, layer: 1, pos: 11
type: A, layer: 1, pos: 43
type: A, layer: 1, pos: 17
type: A, layer: 1, pos: 15
type: A, layer: 1, pos: 3
type: A, layer: 1, pos: 39
type: B, layer: 1, pos: 32
type: A, layer: 1, pos: 10
type: B, layer: 1, pos: 15
type: B, layer: 1, pos: 37
type: B, layer: 1, pos: 20
type: A, layer: 1, pos: 23
type: A, layer: 1, pos: 16
type: B, layer: 1, pos: 39
type: A, layer: 1, pos: 41
type: B, layer: 1, pos: 18
type: B, layer: 1, pos: 41
type: A, layer: 1, pos: 5
type: A, layer: 1, pos: 32
type: A, layer: 1, pos: 37
type: B, layer: 1, pos: 21
type: A, layer: 1, pos: 21
type: B, layer: 1, pos: 5
type: A, layer: 1, pos: 20
type: B, layer: 1, pos: 23
type: B, layer: 1, pos: 16
type: B, layer: 1, pos: 43
type: A, layer: 1, pos: 38
type: A, layer: 1, pos: 29

Time for candidate selection: 0.08 seconds

### Candidate
type: B, layer: 1, pos: 29

## Relational analysis of NS_B1_B1_A1_A1_B1_A2_B1
Optimization infeasible because this subproblem isn't reachable.

### Candidate
type: A, layer: 1, pos: 9

### Candidate
type: B, layer: 1, pos: 17

## Relational analysis of NS_B1_B1_A1_A1_B1_A2_B1
Optimization infeasible because this subproblem isn't reachable.

### Candidate
type: B, layer: 1, pos: 9

### Candidate
type: B, layer: 1, pos: 35

## Relational analysis of NS_B1_B1_A1_A1_B1_A2_B1
Optimization infeasible because this subproblem isn't reachable.

### Candidate
type: A, layer: 1, pos: 35

### Candidate
type: B, layer: 1, pos: 3

## Relational analysis of NS_B1_B1_A1_A1_B1_A2_B1

### Relational analysis result of NS_B1_B1_A1_A1_B1_A2_B1
Status: Status.UNKNOWN
Output dim: 0, lower bound: -63.3964140, upper bound: 63.3964064
time: 0.65 seconds

## Relational analysis of NS_B1_B1_A1_A1_B1_A2_B2

### Relational analysis result of NS_B1_B1_A1_A1_B1_A2_B2
Status: Status.UNKNOWN
Output dim: 0, lower bound: -63.3964120, upper bound: 63.3964073
time: 0.81 seconds

## BFS NS instance: NS_B1_B1_A1_A1_B2_A1

### Backsubstitution after applying NS history:
0: -31.2985306, 32.7712555, -43.3717232, 44.9401779, -76.2387085, 76.1429749
1: -221.7436371, 77.5785599, -303.9672852, 107.6393127, -329.3829041, 381.0203857
2: -123.9428406, 70.7823410, -170.4709625, 98.0167847, -221.9596252, 240.5885010
3: -155.0498047, 56.7560463, -213.3660736, 78.5642319, -233.6140442, 268.8029480
4: -89.2801895, 61.7543640, -123.1605453, 85.3021240, -174.5823059, 184.5613861

Time for backsubstitution: 0.96 seconds

### NS candidates at layer 1
type: A, layer: 1, pos: 29
type: B, layer: 1, pos: 9
type: A, layer: 1, pos: 17
type: A, layer: 1, pos: 9
type: A, layer: 1, pos: 35
type: B, layer: 1, pos: 35
type: A, layer: 1, pos: 3
type: B, layer: 1, pos: 19
type: A, layer: 1, pos: 10
type: B, layer: 1, pos: 45
type: A, layer: 1, pos: 38
type: A, layer: 1, pos: 19
type: B, layer: 1, pos: 18
type: A, layer: 1, pos: 45
type: A, layer: 1, pos: 11
type: B, layer: 1, pos: 11
type: B, layer: 1, pos: 43
type: B, layer: 1, pos: 17
type: B, layer: 1, pos: 15
type: B, layer: 1, pos: 3
type: B, layer: 1, pos: 39
type: A, layer: 1, pos: 32
type: B, layer: 1, pos: 10
type: A, layer: 1, pos: 15
type: A, layer: 1, pos: 37
type: A, layer: 1, pos: 20
type: B, layer: 1, pos: 23
type: B, layer: 1, pos: 16
type: A, layer: 1, pos: 39
type: B, layer: 1, pos: 41
type: A, layer: 1, pos: 18
type: A, layer: 1, pos: 41
type: B, layer: 1, pos: 5
type: B, layer: 1, pos: 32
type: B, layer: 1, pos: 37
type: A, layer: 1, pos: 21
type: B, layer: 1, pos: 21
type: A, layer: 1, pos: 5
type: B, layer: 1, pos: 20
type: A, layer: 1, pos: 23
type: A, layer: 1, pos: 16
type: A, layer: 1, pos: 43
type: B, layer: 1, pos: 38
type: B, layer: 1, pos: 29

Time for candidate selection: 0.08 seconds

### Candidate
type: A, layer: 1, pos: 29

## Relational analysis of NS_B1_B1_A1_A1_B2_A1_A1
Optimization infeasible because this subproblem isn't reachable.

### Candidate
type: B, layer: 1, pos: 9

## Relational analysis of NS_B1_B1_A1_A1_B2_A1_B1
Optimization infeasible because this subproblem isn't reachable.

### Candidate
type: A, layer: 1, pos: 17

## Relational analysis of NS_B1_B1_A1_A1_B2_A1_A1
Optimization infeasible because this subproblem isn't reachable.

### Candidate
type: A, layer: 1, pos: 9

### Candidate
type: A, layer: 1, pos: 35

## Relational analysis of NS_B1_B1_A1_A1_B2_A1_A1
Optimization infeasible because this subproblem isn't reachable.

### Candidate
type: B, layer: 1, pos: 35

### Candidate
type: A, layer: 1, pos: 3

## Relational analysis of NS_B1_B1_A1_A1_B2_A1_A1
Optimization infeasible because this subproblem isn't reachable.

### Candidate
type: B, layer: 1, pos: 19

## Relational analysis of NS_B1_B1_A1_A1_B2_A1_B1
Optimization infeasible because this subproblem isn't reachable.

### Candidate
type: A, layer: 1, pos: 10

## Relational analysis of NS_B1_B1_A1_A1_B2_A1_A1

### Relational analysis result of NS_B1_B1_A1_A1_B2_A1_A1
Status: Status.UNKNOWN
Output dim: 0, lower bound: -63.3959779, upper bound: 63.3960336
time: 0.74 seconds

## Relational analysis of NS_B1_B1_A1_A1_B2_A1_A2

### Relational analysis result of NS_B1_B1_A1_A1_B2_A1_A2
Status: Status.UNKNOWN
Output dim: 0, lower bound: -63.3956942, upper bound: 63.3956942
time: 0.70 seconds

## BFS NS instance: NS_B1_B1_A1_A1_B2_A2

### Backsubstitution after applying NS history:
0: -43.3717232, 44.9401779, -43.3717232, 44.9401779, -88.2596130, 88.2596130
1: -303.9672852, 107.6393127, -303.9672852, 107.6393127, -410.1252747, 410.1252747
2: -170.4709625, 98.0167847, -170.4709625, 98.0167847, -267.2682495, 267.2682495
3: -213.3660736, 78.5642319, -213.3660736, 78.5642319, -290.1622009, 290.1622009
4: -123.1605453, 85.3021240, -123.1605453, 85.3021240, -207.7959747, 207.7959747

Time for backsubstitution: 1.00 seconds

### NS candidates at layer 1
type: B, layer: 1, pos: 9
type: A, layer: 1, pos: 9
type: A, layer: 1, pos: 35
type: B, layer: 1, pos: 35
type: A, layer: 1, pos: 17
type: B, layer: 1, pos: 17
type: A, layer: 1, pos: 19
type: B, layer: 1, pos: 19
type: B, layer: 1, pos: 45
type: A, layer: 1, pos: 45
type: B, layer: 1, pos: 3
type: A, layer: 1, pos: 3
type: B, layer: 1, pos: 11
type: A, layer: 1, pos: 11
type: A, layer: 1, pos: 10
type: B, layer: 1, pos: 10
type: A, layer: 1, pos: 15
type: B, layer: 1, pos: 15
type: A, layer: 1, pos: 18
type: B, layer: 1, pos: 18
type: A, layer: 1, pos: 39
type: B, layer: 1, pos: 39
type: B, layer: 1, pos: 32
type: A, layer: 1, pos: 32
type: A, layer: 1, pos: 43
type: B, layer: 1, pos: 43
type: A, layer: 1, pos: 38
type: B, layer: 1, pos: 38
type: A, layer: 1, pos: 37
type: B, layer: 1, pos: 37
type: A, layer: 1, pos: 29
type: B, layer: 1, pos: 29
type: B, layer: 1, pos: 41
type: A, layer: 1, pos: 41
type: A, layer: 1, pos: 20
type: B, layer: 1, pos: 20
type: B, layer: 1, pos: 21
type: A, layer: 1, pos: 21
type: B, layer: 1, pos: 5
type: A, layer: 1, pos: 5
type: A, layer: 1, pos: 23
type: B, layer: 1, pos: 23
type: A, layer: 1, pos: 16
type: B, layer: 1, pos: 16

Time for candidate selection: 0.08 seconds

### Candidate
type: B, layer: 1, pos: 9

## Relational analysis of NS_B1_B1_A1_A1_B2_A2_B1
Optimization infeasible because this subproblem isn't reachable.

### Candidate
type: A, layer: 1, pos: 9

### Candidate
type: A, layer: 1, pos: 35

## Relational analysis of NS_B1_B1_A1_A1_B2_A2_A1
Optimization infeasible because this subproblem isn't reachable.

### Candidate
type: B, layer: 1, pos: 35

### Candidate
type: A, layer: 1, pos: 17

## Relational analysis of NS_B1_B1_A1_A1_B2_A2_A1
Optimization infeasible because this subproblem isn't reachable.

### Candidate
type: B, layer: 1, pos: 17

### Candidate
type: A, layer: 1, pos: 19

## Relational analysis of NS_B1_B1_A1_A1_B2_A2_A1
Optimization infeasible because this subproblem isn't reachable.

### Candidate
type: B, layer: 1, pos: 19

### Candidate
type: B, layer: 1, pos: 45

## Relational analysis of NS_B1_B1_A1_A1_B2_A2_B1
Optimization infeasible because this subproblem isn't reachable.

### Candidate
type: A, layer: 1, pos: 45

### Candidate
type: B, layer: 1, pos: 3

## Relational analysis of NS_B1_B1_A1_A1_B2_A2_B1
Optimization infeasible because this subproblem isn't reachable.

### Candidate
type: A, layer: 1, pos: 3

### Candidate
type: B, layer: 1, pos: 11

## Relational analysis of NS_B1_B1_A1_A1_B2_A2_B1

### Relational analysis result of NS_B1_B1_A1_A1_B2_A2_B1
Status: Status.UNKNOWN
Output dim: 0, lower bound: -63.3954147, upper bound: 63.3953984
time: 0.67 seconds

## Relational analysis of NS_B1_B1_A1_A1_B2_A2_B2

### Relational analysis result of NS_B1_B1_A1_A1_B2_A2_B2
Status: Status.UNKNOWN
Output dim: 0, lower bound: -63.3954147, upper bound: 63.3961970
time: 0.71 seconds

## BFS NS instance: NS_B1_B1_A1_A2_B1_B1

### Backsubstitution after applying NS history:
0: -29.8957691, 31.5139313, -29.5365906, 30.9115620, -60.8073311, 61.0505219
1: -213.3718414, 73.9482346, -209.6969757, 73.2576218, -286.6294556, 283.6451416
2: -117.9968338, 68.0444641, -117.0093307, 66.9293823, -184.9262085, 185.0537720
3: -148.1968842, 54.5233231, -146.5489655, 53.6311874, -201.8280640, 201.0722809
4: -84.9133301, 59.6252861, -84.2029343, 58.3682022, -143.2815247, 143.8282166

Time for backsubstitution: 0.97 seconds

### NS candidates at layer 1
type: A, layer: 1, pos: 3
type: B, layer: 1, pos: 9
type: A, layer: 1, pos: 9
type: A, layer: 1, pos: 35
type: B, layer: 1, pos: 45
type: B, layer: 1, pos: 35
type: B, layer: 1, pos: 17
type: B, layer: 1, pos: 19
type: A, layer: 1, pos: 17
type: A, layer: 1, pos: 19
type: A, layer: 1, pos: 4
type: A, layer: 1, pos: 11
type: B, layer: 1, pos: 11
type: A, layer: 1, pos: 45
type: B, layer: 1, pos: 10
type: A, layer: 1, pos: 10
type: A, layer: 1, pos: 15
type: B, layer: 1, pos: 18
type: A, layer: 1, pos: 18
type: B, layer: 1, pos: 15
type: B, layer: 1, pos: 32
type: A, layer: 1, pos: 39
type: B, layer: 1, pos: 38
type: A, layer: 1, pos: 43
type: B, layer: 1, pos: 39
type: B, layer: 1, pos: 43
type: A, layer: 1, pos: 32
type: B, layer: 1, pos: 37
type: A, layer: 1, pos: 37
type: B, layer: 1, pos: 41
type: A, layer: 1, pos: 38
type: A, layer: 1, pos: 29
type: B, layer: 1, pos: 29
type: B, layer: 1, pos: 20
type: A, layer: 1, pos: 41
type: B, layer: 1, pos: 5
type: A, layer: 1, pos: 23
type: B, layer: 1, pos: 21
type: A, layer: 1, pos: 20
type: A, layer: 1, pos: 16
type: A, layer: 1, pos: 21
type: A, layer: 1, pos: 5
type: B, layer: 1, pos: 23
type: B, layer: 1, pos: 16

Time for candidate selection: 0.08 seconds

### Candidate
type: A, layer: 1, pos: 3

## Relational analysis of NS_B1_B1_A1_A2_B1_B1_A1
Optimization infeasible because this subproblem isn't reachable.

### Candidate
type: B, layer: 1, pos: 9

### Candidate
type: A, layer: 1, pos: 9

### Candidate
type: A, layer: 1, pos: 35

### Candidate
type: B, layer: 1, pos: 45

### Candidate
type: B, layer: 1, pos: 35

### Candidate
type: B, layer: 1, pos: 17

### Candidate
type: B, layer: 1, pos: 19

### Candidate
type: A, layer: 1, pos: 17

### Candidate
type: A, layer: 1, pos: 19

### Candidate
type: A, layer: 1, pos: 4

## Relational analysis of NS_B1_B1_A1_A2_B1_B1_A1

### Relational analysis result of NS_B1_B1_A1_A2_B1_B1_A1
Status: Status.UNKNOWN
Output dim: 0, lower bound: -63.3958702, upper bound: 63.3958122
time: 0.66 seconds

## Relational analysis of NS_B1_B1_A1_A2_B1_B1_A2

### Relational analysis result of NS_B1_B1_A1_A2_B1_B1_A2
Status: Status.UNKNOWN
Output dim: 0, lower bound: -63.3958702, upper bound: 63.3958122
time: 0.64 seconds

## BFS NS instance: NS_B1_B1_A1_A2_B1_B2

### Backsubstitution after applying NS history:
0: -29.2645760, 30.8694839, -29.2935848, 30.6672192, -59.9317932, 60.1630707
1: -208.9285583, 72.4774933, -207.5810242, 72.7790451, -281.7075806, 280.0585327
2: -115.4824677, 66.6106720, -115.9507675, 66.4279785, -181.9104462, 182.5614319
3: -145.1338348, 53.4042702, -145.0855560, 53.2599449, -198.3937836, 198.4898071
4: -83.1748199, 58.3728714, -83.5118332, 57.9175110, -141.0923309, 141.8847046

Time for backsubstitution: 0.97 seconds

### NS candidates at layer 1
type: A, layer: 1, pos: 3
type: A, layer: 1, pos: 4
type: A, layer: 1, pos: 9
type: B, layer: 1, pos: 9
type: A, layer: 1, pos: 35
type: B, layer: 1, pos: 35
type: A, layer: 1, pos: 45
type: A, layer: 1, pos: 17
type: B, layer: 1, pos: 17
type: B, layer: 1, pos: 19
type: A, layer: 1, pos: 19
type: A, layer: 1, pos: 11
type: B, layer: 1, pos: 11
type: A, layer: 1, pos: 10
type: B, layer: 1, pos: 10
type: B, layer: 1, pos: 45
type: A, layer: 1, pos: 15
type: B, layer: 1, pos: 18
type: A, layer: 1, pos: 29
type: B, layer: 1, pos: 32
type: B, layer: 1, pos: 15
type: A, layer: 1, pos: 39
type: B, layer: 1, pos: 43
type: A, layer: 1, pos: 18
type: B, layer: 1, pos: 39
type: A, layer: 1, pos: 38
type: A, layer: 1, pos: 37
type: A, layer: 1, pos: 32
type: B, layer: 1, pos: 38
type: B, layer: 1, pos: 41
type: B, layer: 1, pos: 37
type: A, layer: 1, pos: 43
type: B, layer: 1, pos: 20
type: A, layer: 1, pos: 41
type: B, layer: 1, pos: 21
type: A, layer: 1, pos: 20
type: A, layer: 1, pos: 5
type: B, layer: 1, pos: 5
type: A, layer: 1, pos: 23
type: B, layer: 1, pos: 23
type: A, layer: 1, pos: 16
type: A, layer: 1, pos: 21
type: B, layer: 1, pos: 16
type: B, layer: 1, pos: 29

Time for candidate selection: 0.08 seconds

### Candidate
type: A, layer: 1, pos: 3

## Relational analysis of NS_B1_B1_A1_A2_B1_B2_A1
Optimization infeasible because this subproblem isn't reachable.

### Candidate
type: A, layer: 1, pos: 4

## Relational analysis of NS_B1_B1_A1_A2_B1_B2_A1

### Relational analysis result of NS_B1_B1_A1_A2_B1_B2_A1
Status: Status.UNKNOWN
Output dim: 0, lower bound: -63.3958754, upper bound: 63.3958122
time: 0.75 seconds

## Relational analysis of NS_B1_B1_A1_A2_B1_B2_A2

### Relational analysis result of NS_B1_B1_A1_A2_B1_B2_A2
Status: Status.UNKNOWN
Output dim: 0, lower bound: -63.3958754, upper bound: 63.3958122
time: 0.75 seconds

## BFS NS instance: NS_B1_B1_A1_A2_B2_A1

### Backsubstitution after applying NS history:
0: -30.1823769, 31.8147984, -43.3717232, 44.9401779, -75.1225433, 75.1865234
1: -214.8651276, 74.6638260, -303.9672852, 107.6393127, -322.5044250, 378.6311035
2: -119.0411377, 68.5830307, -170.4709625, 98.0167847, -217.0579224, 238.5750427
3: -149.3636475, 54.9800568, -213.3660736, 78.5642319, -227.9278870, 267.1139526
4: -85.7541580, 60.1135139, -123.1605453, 85.3021240, -171.0562744, 183.0218353

Time for backsubstitution: 0.97 seconds

### NS candidates at layer 1
type: A, layer: 1, pos: 29
type: B, layer: 1, pos: 9
type: A, layer: 1, pos: 35
type: A, layer: 1, pos: 17
type: A, layer: 1, pos: 9
type: B, layer: 1, pos: 45
type: B, layer: 1, pos: 19
type: A, layer: 1, pos: 10
type: B, layer: 1, pos: 35
type: A, layer: 1, pos: 3
type: B, layer: 1, pos: 18
type: A, layer: 1, pos: 19
type: A, layer: 1, pos: 11
type: A, layer: 1, pos: 38
type: B, layer: 1, pos: 11
type: B, layer: 1, pos: 43
type: B, layer: 1, pos: 17
type: A, layer: 1, pos: 45
type: B, layer: 1, pos: 3
type: B, layer: 1, pos: 15
type: A, layer: 1, pos: 32
type: B, layer: 1, pos: 39
type: A, layer: 1, pos: 15
type: B, layer: 1, pos: 10
type: A, layer: 1, pos: 37
type: A, layer: 1, pos: 20
type: A, layer: 1, pos: 39
type: B, layer: 1, pos: 23
type: B, layer: 1, pos: 41
type: B, layer: 1, pos: 5
type: B, layer: 1, pos: 37
type: B, layer: 1, pos: 16
type: A, layer: 1, pos: 41
type: A, layer: 1, pos: 18
type: B, layer: 1, pos: 32
type: A, layer: 1, pos: 21
type: B, layer: 1, pos: 21
type: B, layer: 1, pos: 20
type: A, layer: 1, pos: 5
type: A, layer: 1, pos: 23
type: A, layer: 1, pos: 16
type: B, layer: 1, pos: 38
type: A, layer: 1, pos: 43
type: B, layer: 1, pos: 29

Time for candidate selection: 0.08 seconds

### Candidate
type: A, layer: 1, pos: 29

## Relational analysis of NS_B1_B1_A1_A2_B2_A1_A1
Optimization infeasible because this subproblem isn't reachable.

### Candidate
type: B, layer: 1, pos: 9

## Relational analysis of NS_B1_B1_A1_A2_B2_A1_B1
Optimization infeasible because this subproblem isn't reachable.

### Candidate
type: A, layer: 1, pos: 35

## Relational analysis of NS_B1_B1_A1_A2_B2_A1_A1
Optimization infeasible because this subproblem isn't reachable.

### Candidate
type: A, layer: 1, pos: 17

## Relational analysis of NS_B1_B1_A1_A2_B2_A1_A1

### Relational analysis result of NS_B1_B1_A1_A2_B2_A1_A1
Status: Status.UNKNOWN
Output dim: 0, lower bound: -63.3955360, upper bound: 63.3953415
time: 0.69 seconds

## Relational analysis of NS_B1_B1_A1_A2_B2_A1_A2

### Relational analysis result of NS_B1_B1_A1_A2_B2_A1_A2
Status: Status.UNKNOWN
Output dim: 0, lower bound: -63.3955360, upper bound: 63.3957590
time: 0.79 seconds

## BFS NS instance: NS_B1_B1_A1_A2_B2_A2

### Backsubstitution after applying NS history:
0: -44.6238937, 46.3271980, -43.3717232, 44.9401779, -89.5640640, 89.6979980
1: -313.9445801, 110.7814026, -303.9672852, 107.6393127, -420.6076660, 413.4206543
2: -175.7769775, 100.9844360, -170.4709625, 98.0167847, -272.9084167, 270.3331604
3: -220.2172699, 80.9449158, -213.3660736, 78.5642319, -297.3772583, 292.6127319
4: -126.8017273, 87.9186707, -123.1605453, 85.3021240, -211.6746979, 210.5055847

Time for backsubstitution: 0.97 seconds

### NS candidates at layer 1
type: A, layer: 1, pos: 9
type: B, layer: 1, pos: 9
type: A, layer: 1, pos: 35
type: B, layer: 1, pos: 35
type: B, layer: 1, pos: 17
type: B, layer: 1, pos: 19
type: A, layer: 1, pos: 17
type: B, layer: 1, pos: 45
type: A, layer: 1, pos: 19
type: A, layer: 1, pos: 45
type: A, layer: 1, pos: 3
type: B, layer: 1, pos: 3
type: A, layer: 1, pos: 11
type: B, layer: 1, pos: 11
type: A, layer: 1, pos: 10
type: B, layer: 1, pos: 10
type: B, layer: 1, pos: 18
type: A, layer: 1, pos: 15
type: B, layer: 1, pos: 15
type: B, layer: 1, pos: 38
type: A, layer: 1, pos: 32
type: A, layer: 1, pos: 18
type: A, layer: 1, pos: 39
type: B, layer: 1, pos: 43
type: B, layer: 1, pos: 39
type: A, layer: 1, pos: 29
type: B, layer: 1, pos: 32
type: B, layer: 1, pos: 37
type: A, layer: 1, pos: 37
type: A, layer: 1, pos: 43
type: B, layer: 1, pos: 41
type: A, layer: 1, pos: 41
type: B, layer: 1, pos: 20
type: A, layer: 1, pos: 38
type: A, layer: 1, pos: 21
type: A, layer: 1, pos: 20
type: B, layer: 1, pos: 5
type: B, layer: 1, pos: 29
type: A, layer: 1, pos: 23
type: A, layer: 1, pos: 5
type: B, layer: 1, pos: 21
type: B, layer: 1, pos: 23
type: A, layer: 1, pos: 16
type: B, layer: 1, pos: 16

Time for candidate selection: 0.08 seconds

### Candidate
type: A, layer: 1, pos: 9

## Relational analysis of NS_B1_B1_A1_A2_B2_A2_A1

### Relational analysis result of NS_B1_B1_A1_A2_B2_A2_A1
Status: Status.UNKNOWN
Output dim: 0, lower bound: -63.3935634, upper bound: 63.3933890
time: 0.76 seconds

## Relational analysis of NS_B1_B1_A1_A2_B2_A2_A2

### Relational analysis result of NS_B1_B1_A1_A2_B2_A2_A2
Status: Status.UNKNOWN
Output dim: 0, lower bound: -63.3950467, upper bound: 63.3948832
time: 0.68 seconds

## BFS NS instance: NS_B1_B1_A2_B1_B1_A1

### Backsubstitution after applying NS history:
0: -52.4417572, 56.9100952, -30.8998337, 32.3675804, -84.8093338, 87.8099213
1: -386.6705322, 129.3537598, -219.2346802, 76.5160141, -462.0713806, 348.5883789
2: -206.8216400, 121.9994202, -122.6432190, 69.8588181, -276.6804504, 244.6426392
3: -266.7668457, 97.7206726, -153.3082123, 56.0272484, -322.3842163, 251.0288849
4: -148.6499023, 107.9011002, -88.1771545, 60.9467621, -209.5966339, 196.0782471

Time for backsubstitution: 1.08 seconds

### NS candidates at layer 1
type: B, layer: 1, pos: 4
type: B, layer: 1, pos: 9
type: A, layer: 1, pos: 35
type: B, layer: 1, pos: 38
type: B, layer: 1, pos: 17
type: B, layer: 1, pos: 3
type: B, layer: 1, pos: 45
type: A, layer: 1, pos: 9
type: A, layer: 1, pos: 11
type: B, layer: 1, pos: 11
type: A, layer: 1, pos: 15
type: A, layer: 1, pos: 10
type: B, layer: 1, pos: 10
type: B, layer: 1, pos: 35
type: A, layer: 1, pos: 16
type: A, layer: 1, pos: 39
type: B, layer: 1, pos: 37
type: A, layer: 1, pos: 19
type: A, layer: 1, pos: 43
type: A, layer: 1, pos: 17
type: A, layer: 1, pos: 3
type: A, layer: 1, pos: 18
type: B, layer: 1, pos: 32
type: B, layer: 1, pos: 18
type: A, layer: 1, pos: 45
type: B, layer: 1, pos: 21
type: A, layer: 1, pos: 23
type: B, layer: 1, pos: 41
type: B, layer: 1, pos: 15
type: A, layer: 1, pos: 32
type: B, layer: 1, pos: 5
type: A, layer: 1, pos: 20
type: A, layer: 1, pos: 41
type: B, layer: 1, pos: 39
type: B, layer: 1, pos: 20
type: A, layer: 1, pos: 5
type: A, layer: 1, pos: 21
type: B, layer: 1, pos: 43
type: B, layer: 1, pos: 23
type: A, layer: 1, pos: 37
type: B, layer: 1, pos: 16
type: A, layer: 1, pos: 38
type: A, layer: 1, pos: 4
type: A, layer: 1, pos: 29

Time for candidate selection: 0.08 seconds

### Candidate
type: B, layer: 1, pos: 4

### Candidate
type: B, layer: 1, pos: 9

### Candidate
type: A, layer: 1, pos: 35

### Candidate
type: B, layer: 1, pos: 38

### Candidate
type: B, layer: 1, pos: 17

### Candidate
type: B, layer: 1, pos: 3

### Candidate
type: B, layer: 1, pos: 45

## Relational analysis of NS_B1_B1_A2_B1_B1_A1_B1

### Relational analysis result of NS_B1_B1_A2_B1_B1_A1_B1
Status: Status.UNKNOWN
Output dim: 0, lower bound: -63.3941710, upper bound: 63.3937844
time: 0.74 seconds

## Relational analysis of NS_B1_B1_A2_B1_B1_A1_B2

### Relational analysis result of NS_B1_B1_A2_B1_B1_A1_B2
Status: Status.UNKNOWN
Output dim: 0, lower bound: -63.3941380, upper bound: 63.3937792
time: 0.71 seconds

## BFS NS instance: NS_B1_B1_A2_B1_B1_A2

### Backsubstitution after applying NS history:
0: -52.6876831, 57.4347305, -30.8998337, 32.3675804, -85.0552597, 88.3345642
1: -391.9079895, 130.1788025, -219.2346802, 76.5160141, -467.7626953, 349.4133911
2: -208.6153564, 123.0446472, -122.6432190, 69.8588181, -278.4741516, 245.6878662
3: -269.8934021, 98.6852036, -153.3082123, 56.0272484, -325.8593140, 251.9934082
4: -149.2469482, 108.9122238, -88.1771545, 60.9467621, -210.1937103, 197.0893860

Time for backsubstitution: 0.98 seconds

### NS candidates at layer 1
type: B, layer: 1, pos: 4
type: B, layer: 1, pos: 3
type: A, layer: 1, pos: 35
type: B, layer: 1, pos: 9
type: B, layer: 1, pos: 38
type: A, layer: 1, pos: 3
type: B, layer: 1, pos: 35
type: A, layer: 1, pos: 9
type: B, layer: 1, pos: 17
type: A, layer: 1, pos: 43
type: A, layer: 1, pos: 15
type: A, layer: 1, pos: 4
type: B, layer: 1, pos: 45
type: A, layer: 1, pos: 11
type: A, layer: 1, pos: 10
type: B, layer: 1, pos: 11
type: B, layer: 1, pos: 10
type: B, layer: 1, pos: 18
type: B, layer: 1, pos: 43
type: A, layer: 1, pos: 18
type: A, layer: 1, pos: 23
type: A, layer: 1, pos: 19
type: A, layer: 1, pos: 16
type: B, layer: 1, pos: 37
type: A, layer: 1, pos: 39
type: B, layer: 1, pos: 21
type: A, layer: 1, pos: 17
type: A, layer: 1, pos: 32
type: A, layer: 1, pos: 21
type: A, layer: 1, pos: 45
type: B, layer: 1, pos: 32
type: B, layer: 1, pos: 23
type: B, layer: 1, pos: 15
type: B, layer: 1, pos: 41
type: A, layer: 1, pos: 20
type: B, layer: 1, pos: 20
type: B, layer: 1, pos: 5
type: A, layer: 1, pos: 41
type: B, layer: 1, pos: 39
type: A, layer: 1, pos: 5
type: A, layer: 1, pos: 38
type: B, layer: 1, pos: 16
type: A, layer: 1, pos: 37
type: A, layer: 1, pos: 29

Time for candidate selection: 0.08 seconds

### Candidate
type: B, layer: 1, pos: 4

### Candidate
type: B, layer: 1, pos: 3

### Candidate
type: A, layer: 1, pos: 35

### Candidate
type: B, layer: 1, pos: 9

### Candidate
type: B, layer: 1, pos: 38

### Candidate
type: A, layer: 1, pos: 3

### Candidate
type: B, layer: 1, pos: 35

### Candidate
type: A, layer: 1, pos: 9

### Candidate
type: B, layer: 1, pos: 17

### Candidate
type: A, layer: 1, pos: 43

## Relational analysis of NS_B1_B1_A2_B1_B1_A2_A1
Optimization infeasible because this subproblem isn't reachable.

### Candidate
type: A, layer: 1, pos: 15

## Relational analysis of NS_B1_B1_A2_B1_B1_A2_A1
Optimization infeasible because this subproblem isn't reachable.

### Candidate
type: A, layer: 1, pos: 4

### Candidate
type: B, layer: 1, pos: 45

## Relational analysis of NS_B1_B1_A2_B1_B1_A2_B1

### Relational analysis result of NS_B1_B1_A2_B1_B1_A2_B1
Status: Status.UNKNOWN
Output dim: 0, lower bound: -63.3941710, upper bound: 63.3938891
time: 0.74 seconds

## Relational analysis of NS_B1_B1_A2_B1_B1_A2_B2

### Relational analysis result of NS_B1_B1_A2_B1_B1_A2_B2
Status: Status.UNKNOWN
Output dim: 0, lower bound: -63.3941380, upper bound: 63.3940286
time: 0.67 seconds

## BFS NS instance: NS_B1_B1_A2_B1_B2_A1

### Backsubstitution after applying NS history:
0: -51.9699173, 56.3084450, -40.1849136, 41.3175163, -93.2874298, 96.3959198
1: -381.2426453, 128.1190796, -270.5454102, 99.4554672, -478.6930237, 398.6644897
2: -204.5754395, 120.6591721, -155.8203583, 89.4233932, -293.9988403, 275.6827698
3: -263.2272949, 96.6899796, -190.4992371, 71.5364838, -334.3562622, 287.0283813
4: -147.2113037, 106.7868118, -113.9260864, 78.2623062, -225.4736023, 219.7004089

Time for backsubstitution: 0.98 seconds

### NS candidates at layer 1
type: B, layer: 1, pos: 4
type: B, layer: 1, pos: 9
type: A, layer: 1, pos: 35
type: B, layer: 1, pos: 3
type: B, layer: 1, pos: 17
type: A, layer: 1, pos: 9
type: B, layer: 1, pos: 45
type: A, layer: 1, pos: 10
type: B, layer: 1, pos: 35
type: A, layer: 1, pos: 11
type: B, layer: 1, pos: 38
type: B, layer: 1, pos: 11
type: A, layer: 1, pos: 15
type: A, layer: 1, pos: 17
type: B, layer: 1, pos: 18
type: A, layer: 1, pos: 3
type: A, layer: 1, pos: 4
type: A, layer: 1, pos: 29
type: A, layer: 1, pos: 16
type: B, layer: 1, pos: 10
type: B, layer: 1, pos: 37
type: A, layer: 1, pos: 32
type: A, layer: 1, pos: 19
type: A, layer: 1, pos: 39
type: A, layer: 1, pos: 45
type: B, layer: 1, pos: 15
type: B, layer: 1, pos: 21
type: A, layer: 1, pos: 20
type: A, layer: 1, pos: 18
type: B, layer: 1, pos: 43
type: B, layer: 1, pos: 39
type: B, layer: 1, pos: 32
type: B, layer: 1, pos: 41
type: A, layer: 1, pos: 43
type: B, layer: 1, pos: 5
type: A, layer: 1, pos: 41
type: A, layer: 1, pos: 23
type: B, layer: 1, pos: 23
type: A, layer: 1, pos: 5
type: A, layer: 1, pos: 21
type: B, layer: 1, pos: 20
type: A, layer: 1, pos: 37
type: B, layer: 1, pos: 16
type: A, layer: 1, pos: 38

Time for candidate selection: 0.08 seconds

### Candidate
type: B, layer: 1, pos: 4

### Candidate
type: B, layer: 1, pos: 9

### Candidate
type: A, layer: 1, pos: 35

### Candidate
type: B, layer: 1, pos: 3

### Candidate
type: B, layer: 1, pos: 17

### Candidate
type: A, layer: 1, pos: 9

### Candidate
type: B, layer: 1, pos: 45

## Relational analysis of NS_B1_B1_A2_B1_B2_A1_B1
Optimization infeasible because this subproblem isn't reachable.

### Candidate
type: A, layer: 1, pos: 10

## Relational analysis of NS_B1_B1_A2_B1_B2_A1_A1
Optimization infeasible because this subproblem isn't reachable.

### Candidate
type: B, layer: 1, pos: 35

### Candidate
type: A, layer: 1, pos: 11

## Relational analysis of NS_B1_B1_A2_B1_B2_A1_A1

### Relational analysis result of NS_B1_B1_A2_B1_B2_A1_A1
Status: Status.UNKNOWN
Output dim: 0, lower bound: -63.3934695, upper bound: 63.3935876
time: 0.71 seconds

## Relational analysis of NS_B1_B1_A2_B1_B2_A1_A2

### Relational analysis result of NS_B1_B1_A2_B1_B2_A1_A2
Status: Status.UNKNOWN
Output dim: 0, lower bound: -63.3936346, upper bound: 63.3936782
time: 0.74 seconds

## BFS NS instance: NS_B1_B1_A2_B1_B2_A2

### Backsubstitution after applying NS history:
0: -52.0833855, 56.6725082, -40.1849136, 41.3175163, -93.4009018, 96.7914581
1: -385.2969971, 128.5437622, -270.5454102, 99.4554672, -483.2335205, 399.0891113
2: -205.7564697, 121.3837280, -155.8203583, 89.4233932, -295.1798706, 276.4957275
3: -265.6109009, 97.3231201, -190.4992371, 71.5364838, -337.1045532, 287.7229919
4: -147.3715973, 107.5238266, -113.9260864, 78.2623062, -225.6338959, 220.4762268

Time for backsubstitution: 1.08 seconds

### NS candidates at layer 1
type: B, layer: 1, pos: 4
type: B, layer: 1, pos: 3
type: A, layer: 1, pos: 35
type: B, layer: 1, pos: 9
type: B, layer: 1, pos: 35
type: A, layer: 1, pos: 9
type: B, layer: 1, pos: 17
type: B, layer: 1, pos: 38
type: A, layer: 1, pos: 3
type: A, layer: 1, pos: 15
type: A, layer: 1, pos: 10
type: B, layer: 1, pos: 45
type: B, layer: 1, pos: 18
type: A, layer: 1, pos: 4
type: A, layer: 1, pos: 11
type: B, layer: 1, pos: 11
type: B, layer: 1, pos: 43
type: A, layer: 1, pos: 29
type: A, layer: 1, pos: 43
type: A, layer: 1, pos: 17
type: A, layer: 1, pos: 32
type: A, layer: 1, pos: 19
type: A, layer: 1, pos: 16
type: B, layer: 1, pos: 37
type: B, layer: 1, pos: 10
type: B, layer: 1, pos: 21
type: A, layer: 1, pos: 18
type: A, layer: 1, pos: 21
type: B, layer: 1, pos: 15
type: A, layer: 1, pos: 20
type: A, layer: 1, pos: 23
type: A, layer: 1, pos: 39
type: A, layer: 1, pos: 45
type: B, layer: 1, pos: 23
type: B, layer: 1, pos: 41
type: B, layer: 1, pos: 32
type: B, layer: 1, pos: 5
type: A, layer: 1, pos: 41
type: B, layer: 1, pos: 39
type: B, layer: 1, pos: 20
type: A, layer: 1, pos: 5
type: A, layer: 1, pos: 37
type: A, layer: 1, pos: 38
type: B, layer: 1, pos: 16

Time for candidate selection: 0.08 seconds

### Candidate
type: B, layer: 1, pos: 4

### Candidate
type: B, layer: 1, pos: 3

### Candidate
type: A, layer: 1, pos: 35

### Candidate
type: B, layer: 1, pos: 9

### Candidate
type: B, layer: 1, pos: 35

### Candidate
type: A, layer: 1, pos: 9

### Candidate
type: B, layer: 1, pos: 17

### Candidate
type: B, layer: 1, pos: 38

### Candidate
type: A, layer: 1, pos: 3

### Candidate
type: A, layer: 1, pos: 15

## Relational analysis of NS_B1_B1_A2_B1_B2_A2_A1
Optimization infeasible because this subproblem isn't reachable.

### Candidate
type: A, layer: 1, pos: 10

## Relational analysis of NS_B1_B1_A2_B1_B2_A2_A1
Optimization infeasible because this subproblem isn't reachable.

### Candidate
type: B, layer: 1, pos: 45

## Relational analysis of NS_B1_B1_A2_B1_B2_A2_B1
Optimization infeasible because this subproblem isn't reachable.

### Candidate
type: B, layer: 1, pos: 18

## Relational analysis of NS_B1_B1_A2_B1_B2_A2_B1
Optimization infeasible because this subproblem isn't reachable.

### Candidate
type: A, layer: 1, pos: 4

### Candidate
type: A, layer: 1, pos: 11

## Relational analysis of NS_B1_B1_A2_B1_B2_A2_A1

### Relational analysis result of NS_B1_B1_A2_B1_B2_A2_A1
Status: Status.UNKNOWN
Output dim: 0, lower bound: -63.3934695, upper bound: 63.3937074
time: 0.72 seconds

## Relational analysis of NS_B1_B1_A2_B1_B2_A2_A2

### Relational analysis result of NS_B1_B1_A2_B1_B2_A2_A2
Status: Status.UNKNOWN
Output dim: 0, lower bound: -63.3934695, upper bound: 63.3938612
time: 0.76 seconds

## BFS NS instance: NS_B1_B1_A2_B2_B1_A1

### Backsubstitution after applying NS history:
0: -52.4790688, 56.8977356, -33.7412872, 35.4420433, -87.9211121, 90.6389923
1: -386.5005493, 129.4375458, -241.4537048, 83.6052399, -469.0086975, 370.8912354
2: -206.9522552, 121.9908752, -134.5909882, 76.4633179, -283.4154968, 256.5818481
3: -266.7574463, 97.7221451, -168.7952271, 61.3642120, -327.7478638, 266.5173645
4: -148.7790222, 107.8636475, -96.5188828, 66.6376648, -215.4166870, 204.3825378

Time for backsubstitution: 1.00 seconds

### NS candidates at layer 1
type: B, layer: 1, pos: 4
type: B, layer: 1, pos: 9
type: B, layer: 1, pos: 3
type: A, layer: 1, pos: 35
type: B, layer: 1, pos: 38
type: B, layer: 1, pos: 17
type: B, layer: 1, pos: 45
type: A, layer: 1, pos: 9
type: A, layer: 1, pos: 11
type: B, layer: 1, pos: 11
type: A, layer: 1, pos: 10
type: B, layer: 1, pos: 35
type: A, layer: 1, pos: 15
type: B, layer: 1, pos: 10
type: A, layer: 1, pos: 39
type: A, layer: 1, pos: 3
type: A, layer: 1, pos: 17
type: A, layer: 1, pos: 16
type: B, layer: 1, pos: 18
type: B, layer: 1, pos: 37
type: A, layer: 1, pos: 18
type: A, layer: 1, pos: 43
type: A, layer: 1, pos: 45
type: B, layer: 1, pos: 15
type: A, layer: 1, pos: 19
type: A, layer: 1, pos: 23
type: A, layer: 1, pos: 32
type: B, layer: 1, pos: 32
type: B, layer: 1, pos: 41
type: B, layer: 1, pos: 21
type: A, layer: 1, pos: 41
type: B, layer: 1, pos: 5
type: A, layer: 1, pos: 20
type: B, layer: 1, pos: 20
type: A, layer: 1, pos: 5
type: A, layer: 1, pos: 21
type: B, layer: 1, pos: 39
type: B, layer: 1, pos: 43
type: B, layer: 1, pos: 23
type: A, layer: 1, pos: 37
type: B, layer: 1, pos: 16
type: A, layer: 1, pos: 4
type: A, layer: 1, pos: 38
type: A, layer: 1, pos: 29

Time for candidate selection: 0.08 seconds

### Candidate
type: B, layer: 1, pos: 4

### Candidate
type: B, layer: 1, pos: 9

### Candidate
type: B, layer: 1, pos: 3

### Candidate
type: A, layer: 1, pos: 35

### Candidate
type: B, layer: 1, pos: 38

### Candidate
type: B, layer: 1, pos: 17

### Candidate
type: B, layer: 1, pos: 45

## Relational analysis of NS_B1_B1_A2_B2_B1_A1_B1
Optimization infeasible because this subproblem isn't reachable.

### Candidate
type: A, layer: 1, pos: 9

### Candidate
type: A, layer: 1, pos: 11

## Relational analysis of NS_B1_B1_A2_B2_B1_A1_A1

### Relational analysis result of NS_B1_B1_A2_B2_B1_A1_A1
Status: Status.UNKNOWN
Output dim: 0, lower bound: -63.3935559, upper bound: 63.3935835
time: 0.65 seconds

## Relational analysis of NS_B1_B1_A2_B2_B1_A1_A2

### Relational analysis result of NS_B1_B1_A2_B2_B1_A1_A2
Status: Status.UNKNOWN
Output dim: 0, lower bound: -63.3937217, upper bound: 63.3936811
time: 0.78 seconds

## BFS NS instance: NS_B1_B1_A2_B2_B1_A2

### Backsubstitution after applying NS history:
0: -52.6066322, 57.2988358, -33.7412872, 35.4420433, -88.0486679, 91.0401077
1: -390.8960571, 129.9486084, -241.4537048, 83.6052399, -473.8500366, 371.4022827
2: -208.2440948, 122.7669373, -134.5909882, 76.4633179, -284.7073975, 257.3579102
3: -269.2930603, 98.4558945, -168.7952271, 61.3642120, -330.6251831, 267.2510986
4: -149.0089111, 108.6464310, -96.5188828, 66.6376648, -215.6465759, 205.1653137

Time for backsubstitution: 1.01 seconds

### NS candidates at layer 1
type: B, layer: 1, pos: 4
type: B, layer: 1, pos: 3
type: A, layer: 1, pos: 35
type: B, layer: 1, pos: 9
type: B, layer: 1, pos: 38
type: A, layer: 1, pos: 9
type: B, layer: 1, pos: 35
type: A, layer: 1, pos: 3
type: B, layer: 1, pos: 17
type: B, layer: 1, pos: 45
type: A, layer: 1, pos: 43
type: A, layer: 1, pos: 15
type: A, layer: 1, pos: 11
type: B, layer: 1, pos: 11
type: A, layer: 1, pos: 4
type: A, layer: 1, pos: 10
type: B, layer: 1, pos: 18
type: B, layer: 1, pos: 10
type: A, layer: 1, pos: 23
type: A, layer: 1, pos: 39
type: A, layer: 1, pos: 18
type: B, layer: 1, pos: 43
type: A, layer: 1, pos: 45
type: A, layer: 1, pos: 17
type: A, layer: 1, pos: 32
type: B, layer: 1, pos: 37
type: A, layer: 1, pos: 16
type: B, layer: 1, pos: 15
type: B, layer: 1, pos: 21
type: A, layer: 1, pos: 21
type: A, layer: 1, pos: 19
type: B, layer: 1, pos: 20
type: B, layer: 1, pos: 41
type: A, layer: 1, pos: 20
type: B, layer: 1, pos: 23
type: A, layer: 1, pos: 41
type: B, layer: 1, pos: 5
type: B, layer: 1, pos: 32
type: A, layer: 1, pos: 5
type: B, layer: 1, pos: 39
type: B, layer: 1, pos: 16
type: A, layer: 1, pos: 37
type: A, layer: 1, pos: 38
type: A, layer: 1, pos: 29

Time for candidate selection: 0.08 seconds

### Candidate
type: B, layer: 1, pos: 4

### Candidate
type: B, layer: 1, pos: 3

### Candidate
type: A, layer: 1, pos: 35

### Candidate
type: B, layer: 1, pos: 9

### Candidate
type: B, layer: 1, pos: 38

### Candidate
type: A, layer: 1, pos: 9

### Candidate
type: B, layer: 1, pos: 35

### Candidate
type: A, layer: 1, pos: 3

### Candidate
type: B, layer: 1, pos: 17

### Candidate
type: B, layer: 1, pos: 45

## Relational analysis of NS_B1_B1_A2_B2_B1_A2_B1
Optimization infeasible because this subproblem isn't reachable.

### Candidate
type: A, layer: 1, pos: 43

## Relational analysis of NS_B1_B1_A2_B2_B1_A2_A1
Optimization infeasible because this subproblem isn't reachable.

### Candidate
type: A, layer: 1, pos: 15

## Relational analysis of NS_B1_B1_A2_B2_B1_A2_A1
Optimization infeasible because this subproblem isn't reachable.

### Candidate
type: A, layer: 1, pos: 11

## Relational analysis of NS_B1_B1_A2_B2_B1_A2_A1

### Relational analysis result of NS_B1_B1_A2_B2_B1_A2_A1
Status: Status.UNKNOWN
Output dim: 0, lower bound: -63.3935559, upper bound: 63.3937464
time: 0.70 seconds

## Relational analysis of NS_B1_B1_A2_B2_B1_A2_A2

### Relational analysis result of NS_B1_B1_A2_B2_B1_A2_A2
Status: Status.UNKNOWN
Output dim: 0, lower bound: -63.3937217, upper bound: 63.3940841
time: 0.68 seconds

## BFS NS instance: NS_B1_B1_A2_B2_B2_A1

### Backsubstitution after applying NS history:
0: -51.9288788, 56.2117043, -43.8191643, 45.3083954, -97.2372742, 99.8938904
1: -380.5144348, 128.0065308, -299.1305847, 108.4494934, -486.8936768, 427.1370544
2: -204.4586792, 120.4639511, -171.1938019, 97.9577637, -302.4164124, 290.4625854
3: -262.8247681, 96.5425644, -210.4305420, 78.3455505, -340.7835999, 306.2767334
4: -147.1262665, 106.5851288, -124.5197067, 85.6121521, -232.7384186, 229.7925110

Time for backsubstitution: 1.01 seconds

### NS candidates at layer 1
type: B, layer: 1, pos: 4
type: B, layer: 1, pos: 9
type: A, layer: 1, pos: 35
type: B, layer: 1, pos: 17
type: B, layer: 1, pos: 3
type: A, layer: 1, pos: 9
type: B, layer: 1, pos: 35
type: A, layer: 1, pos: 10
type: B, layer: 1, pos: 45
type: A, layer: 1, pos: 4
type: A, layer: 1, pos: 11
type: B, layer: 1, pos: 11
type: B, layer: 1, pos: 38
type: B, layer: 1, pos: 18
type: A, layer: 1, pos: 17
type: A, layer: 1, pos: 3
type: A, layer: 1, pos: 15
type: B, layer: 1, pos: 10
type: A, layer: 1, pos: 29
type: A, layer: 1, pos: 45
type: B, layer: 1, pos: 37
type: A, layer: 1, pos: 39
type: A, layer: 1, pos: 32
type: A, layer: 1, pos: 16
type: A, layer: 1, pos: 19
type: B, layer: 1, pos: 43
type: B, layer: 1, pos: 15
type: B, layer: 1, pos: 32
type: A, layer: 1, pos: 20
type: B, layer: 1, pos: 39
type: B, layer: 1, pos: 21
type: A, layer: 1, pos: 18
type: B, layer: 1, pos: 41
type: A, layer: 1, pos: 23
type: A, layer: 1, pos: 41
type: B, layer: 1, pos: 5
type: A, layer: 1, pos: 5
type: A, layer: 1, pos: 43
type: B, layer: 1, pos: 23
type: A, layer: 1, pos: 21
type: B, layer: 1, pos: 20
type: A, layer: 1, pos: 37
type: B, layer: 1, pos: 16
type: A, layer: 1, pos: 38

Time for candidate selection: 0.08 seconds

### Candidate
type: B, layer: 1, pos: 4

### Candidate
type: B, layer: 1, pos: 9

### Candidate
type: A, layer: 1, pos: 35

### Candidate
type: B, layer: 1, pos: 17

## Relational analysis of NS_B1_B1_A2_B2_B2_A1_B1
Optimization infeasible because this subproblem isn't reachable.

### Candidate
type: B, layer: 1, pos: 3

### Candidate
type: A, layer: 1, pos: 9

### Candidate
type: B, layer: 1, pos: 35

### Candidate
type: A, layer: 1, pos: 10

## Relational analysis of NS_B1_B1_A2_B2_B2_A1_A1
Optimization infeasible because this subproblem isn't reachable.

### Candidate
type: B, layer: 1, pos: 45

## Relational analysis of NS_B1_B1_A2_B2_B2_A1_B1
Optimization infeasible because this subproblem isn't reachable.

### Candidate
type: A, layer: 1, pos: 4

### Candidate
type: A, layer: 1, pos: 11

## Relational analysis of NS_B1_B1_A2_B2_B2_A1_A1

### Relational analysis result of NS_B1_B1_A2_B2_B2_A1_A1
Status: Status.UNKNOWN
Output dim: 0, lower bound: -63.3934748, upper bound: 63.3935350
time: 0.71 seconds

## Relational analysis of NS_B1_B1_A2_B2_B2_A1_A2

### Relational analysis result of NS_B1_B1_A2_B2_B2_A1_A2
Status: Status.UNKNOWN
Output dim: 0, lower bound: -63.3936405, upper bound: 63.3936336
time: 0.70 seconds

## BFS NS instance: NS_B1_B1_A2_B2_B2_A2

### Backsubstitution after applying NS history:
0: -51.9615936, 56.4974899, -43.8191643, 45.3083954, -97.2699890, 100.2106171
1: -383.9544983, 128.2036591, -299.1305847, 108.4494934, -490.8065186, 427.3342285
2: -205.1951294, 121.0046310, -171.1938019, 97.9577637, -303.1528931, 291.0913391
3: -264.7727661, 97.0129776, -210.4305420, 78.3455505, -343.0846252, 306.8079529
4: -147.0079651, 107.1813049, -124.5197067, 85.6121521, -232.6201172, 230.4271088

Time for backsubstitution: 1.14 seconds

### NS candidates at layer 1
type: B, layer: 1, pos: 4
type: A, layer: 1, pos: 35
type: B, layer: 1, pos: 3
type: B, layer: 1, pos: 9
type: A, layer: 1, pos: 9
type: B, layer: 1, pos: 35
type: B, layer: 1, pos: 17
type: B, layer: 1, pos: 38
type: A, layer: 1, pos: 3
type: B, layer: 1, pos: 18
type: A, layer: 1, pos: 10
type: A, layer: 1, pos: 15
type: B, layer: 1, pos: 43
type: A, layer: 1, pos: 11
type: B, layer: 1, pos: 45
type: B, layer: 1, pos: 11
type: A, layer: 1, pos: 17
type: A, layer: 1, pos: 4
type: A, layer: 1, pos: 29
type: A, layer: 1, pos: 32
type: B, layer: 1, pos: 37
type: B, layer: 1, pos: 10
type: A, layer: 1, pos: 43
type: B, layer: 1, pos: 21
type: A, layer: 1, pos: 19
type: B, layer: 1, pos: 15
type: A, layer: 1, pos: 16
type: A, layer: 1, pos: 45
type: A, layer: 1, pos: 18
type: A, layer: 1, pos: 21
type: A, layer: 1, pos: 39
type: A, layer: 1, pos: 23
type: A, layer: 1, pos: 20
type: B, layer: 1, pos: 41
type: B, layer: 1, pos: 23
type: B, layer: 1, pos: 32
type: A, layer: 1, pos: 41
type: B, layer: 1, pos: 5
type: B, layer: 1, pos: 39
type: B, layer: 1, pos: 20
type: A, layer: 1, pos: 5
type: A, layer: 1, pos: 37
type: B, layer: 1, pos: 16
type: A, layer: 1, pos: 38

Time for candidate selection: 0.08 seconds

### Candidate
type: B, layer: 1, pos: 4

### Candidate
type: A, layer: 1, pos: 35

### Candidate
type: B, layer: 1, pos: 3

### Candidate
type: B, layer: 1, pos: 9

### Candidate
type: A, layer: 1, pos: 9

### Candidate
type: B, layer: 1, pos: 35

### Candidate
type: B, layer: 1, pos: 17

## Relational analysis of NS_B1_B1_A2_B2_B2_A2_B1
Optimization infeasible because this subproblem isn't reachable.

### Candidate
type: B, layer: 1, pos: 38

### Candidate
type: A, layer: 1, pos: 3

### Candidate
type: B, layer: 1, pos: 18

## Relational analysis of NS_B1_B1_A2_B2_B2_A2_B1
Optimization infeasible because this subproblem isn't reachable.

### Candidate
type: A, layer: 1, pos: 10

## Relational analysis of NS_B1_B1_A2_B2_B2_A2_A1
Optimization infeasible because this subproblem isn't reachable.

### Candidate
type: A, layer: 1, pos: 15

## Relational analysis of NS_B1_B1_A2_B2_B2_A2_A1
Optimization infeasible because this subproblem isn't reachable.

### Candidate
type: B, layer: 1, pos: 43

## Relational analysis of NS_B1_B1_A2_B2_B2_A2_B1
Optimization infeasible because this subproblem isn't reachable.

### Candidate
type: A, layer: 1, pos: 11

## Relational analysis of NS_B1_B1_A2_B2_B2_A2_A1

### Relational analysis result of NS_B1_B1_A2_B2_B2_A2_A1
Status: Status.UNKNOWN
Output dim: 0, lower bound: -63.3934748, upper bound: 63.3936938
time: 0.74 seconds

## Relational analysis of NS_B1_B1_A2_B2_B2_A2_A2

### Relational analysis result of NS_B1_B1_A2_B2_B2_A2_A2
Status: Status.UNKNOWN
Output dim: 0, lower bound: -63.3936405, upper bound: 63.3940548
time: 0.80 seconds

## BFS NS instance: NS_B1_B2_A1_A1_A1_A1

### Backsubstitution after applying NS history:
0: -29.5365906, 30.9115620, -29.8957691, 31.5139313, -61.0505219, 60.8073311
1: -209.6969757, 73.2576218, -213.3718414, 73.9482346, -283.6451721, 286.6294556
2: -117.0093307, 66.9293823, -117.9968338, 68.0444641, -185.0537720, 184.9262085
3: -146.5489655, 53.6311874, -148.1968842, 54.5233231, -201.0722809, 201.8280640
4: -84.2029343, 58.3682022, -84.9133301, 59.6252861, -143.8282166, 143.2815247

Time for backsubstitution: 1.03 seconds

### NS candidates at layer 1
type: B, layer: 1, pos: 3
type: A, layer: 1, pos: 9
type: B, layer: 1, pos: 9
type: B, layer: 1, pos: 35
type: A, layer: 1, pos: 45
type: A, layer: 1, pos: 35
type: A, layer: 1, pos: 17
type: A, layer: 1, pos: 19
type: B, layer: 1, pos: 17
type: B, layer: 1, pos: 19
type: B, layer: 1, pos: 4
type: B, layer: 1, pos: 11
type: A, layer: 1, pos: 11
type: B, layer: 1, pos: 45
type: A, layer: 1, pos: 10
type: B, layer: 1, pos: 10
type: B, layer: 1, pos: 15
type: A, layer: 1, pos: 18
type: B, layer: 1, pos: 18
type: A, layer: 1, pos: 15
type: A, layer: 1, pos: 32
type: B, layer: 1, pos: 39
type: A, layer: 1, pos: 38
type: B, layer: 1, pos: 43
type: A, layer: 1, pos: 39
type: A, layer: 1, pos: 43
type: B, layer: 1, pos: 32
type: A, layer: 1, pos: 37
type: B, layer: 1, pos: 37
type: A, layer: 1, pos: 41
type: B, layer: 1, pos: 38
type: B, layer: 1, pos: 29
type: A, layer: 1, pos: 29
type: A, layer: 1, pos: 20
type: B, layer: 1, pos: 41
type: A, layer: 1, pos: 5
type: B, layer: 1, pos: 23
type: A, layer: 1, pos: 21
type: B, layer: 1, pos: 20
type: B, layer: 1, pos: 16
type: B, layer: 1, pos: 21
type: B, layer: 1, pos: 5
type: A, layer: 1, pos: 23
type: A, layer: 1, pos: 16

Time for candidate selection: 0.08 seconds

### Candidate
type: B, layer: 1, pos: 3

## Relational analysis of NS_B1_B2_A1_A1_A1_A1_B1
Optimization infeasible because this subproblem isn't reachable.

### Candidate
type: A, layer: 1, pos: 9

### Candidate
type: B, layer: 1, pos: 9

### Candidate
type: B, layer: 1, pos: 35

### Candidate
type: A, layer: 1, pos: 45

### Candidate
type: A, layer: 1, pos: 35

### Candidate
type: A, layer: 1, pos: 17

### Candidate
type: A, layer: 1, pos: 19

### Candidate
type: B, layer: 1, pos: 17

### Candidate
type: B, layer: 1, pos: 19

### Candidate
type: B, layer: 1, pos: 4

## Relational analysis of NS_B1_B2_A1_A1_A1_A1_B1

### Relational analysis result of NS_B1_B2_A1_A1_A1_A1_B1
Status: Status.UNKNOWN
Output dim: 0, lower bound: -63.3958122, upper bound: 63.3958702
time: 0.73 seconds

## Relational analysis of NS_B1_B2_A1_A1_A1_A1_B2

### Relational analysis result of NS_B1_B2_A1_A1_A1_A1_B2
Status: Status.UNKNOWN
Output dim: 0, lower bound: -63.3958122, upper bound: 63.3958702
time: 0.67 seconds

## BFS NS instance: NS_B1_B2_A1_A1_A1_A2

### Backsubstitution after applying NS history:
0: -29.2935848, 30.6672192, -29.2645760, 30.8694839, -60.1630707, 59.9317932
1: -207.5810242, 72.7790451, -208.9285583, 72.4774933, -280.0585327, 281.7075806
2: -115.9507675, 66.4279785, -115.4824677, 66.6106720, -182.5614319, 181.9104462
3: -145.0855560, 53.2599449, -145.1338348, 53.4042702, -198.4897919, 198.3937836
4: -83.5118332, 57.9175110, -83.1748199, 58.3728714, -141.8847046, 141.0923309

Time for backsubstitution: 1.02 seconds

### NS candidates at layer 1
type: B, layer: 1, pos: 3
type: B, layer: 1, pos: 4
type: B, layer: 1, pos: 9
type: A, layer: 1, pos: 9
type: B, layer: 1, pos: 35
type: A, layer: 1, pos: 35
type: B, layer: 1, pos: 45
type: B, layer: 1, pos: 17
type: A, layer: 1, pos: 17
type: A, layer: 1, pos: 19
type: B, layer: 1, pos: 19
type: B, layer: 1, pos: 11
type: A, layer: 1, pos: 11
type: B, layer: 1, pos: 10
type: A, layer: 1, pos: 10
type: A, layer: 1, pos: 45
type: B, layer: 1, pos: 15
type: A, layer: 1, pos: 18
type: B, layer: 1, pos: 29
type: A, layer: 1, pos: 32
type: A, layer: 1, pos: 15
type: B, layer: 1, pos: 39
type: A, layer: 1, pos: 43
type: B, layer: 1, pos: 18
type: A, layer: 1, pos: 39
type: B, layer: 1, pos: 38
type: B, layer: 1, pos: 37
type: B, layer: 1, pos: 32
type: A, layer: 1, pos: 38
type: A, layer: 1, pos: 41
type: A, layer: 1, pos: 37
type: B, layer: 1, pos: 43
type: A, layer: 1, pos: 20
type: B, layer: 1, pos: 41
type: A, layer: 1, pos: 21
type: B, layer: 1, pos: 20
type: B, layer: 1, pos: 5
type: A, layer: 1, pos: 5
type: B, layer: 1, pos: 23
type: A, layer: 1, pos: 23
type: B, layer: 1, pos: 16
type: B, layer: 1, pos: 21
type: A, layer: 1, pos: 16
type: A, layer: 1, pos: 29

Time for candidate selection: 0.10 seconds

### Candidate
type: B, layer: 1, pos: 3

## Relational analysis of NS_B1_B2_A1_A1_A1_A2_B1
Optimization infeasible because this subproblem isn't reachable.

### Candidate
type: B, layer: 1, pos: 4

## Relational analysis of NS_B1_B2_A1_A1_A1_A2_B1

### Relational analysis result of NS_B1_B2_A1_A1_A1_A2_B1
Status: Status.UNKNOWN
Output dim: 0, lower bound: -63.3958122, upper bound: 63.3958754
time: 0.76 seconds

## Relational analysis of NS_B1_B2_A1_A1_A1_A2_B2

### Relational analysis result of NS_B1_B2_A1_A1_A1_A2_B2
Status: Status.UNKNOWN
Output dim: 0, lower bound: -63.3958122, upper bound: 63.3958754
time: 0.64 seconds

## BFS NS instance: NS_B1_B2_A1_A1_A2_B1

### Backsubstitution after applying NS history:
0: -43.3717232, 44.9401779, -30.1823769, 31.8147984, -75.1865234, 75.1225433
1: -303.9672852, 107.6393127, -214.8651276, 74.6638260, -378.6311035, 322.5044556
2: -170.4709625, 98.0167847, -119.0411377, 68.5830307, -238.5750427, 217.0579224
3: -213.3660736, 78.5642319, -149.3636475, 54.9800568, -267.1139526, 227.9278870
4: -123.1605453, 85.3021240, -85.7541580, 60.1135139, -183.0218353, 171.0562744

Time for backsubstitution: 1.31 seconds

### NS candidates at layer 1
type: B, layer: 1, pos: 29
type: A, layer: 1, pos: 9
type: B, layer: 1, pos: 35
type: B, layer: 1, pos: 17
type: B, layer: 1, pos: 9
type: A, layer: 1, pos: 45
type: A, layer: 1, pos: 19
type: B, layer: 1, pos: 10
type: A, layer: 1, pos: 35
type: B, layer: 1, pos: 3
type: A, layer: 1, pos: 18
type: B, layer: 1, pos: 19
type: B, layer: 1, pos: 11
type: B, layer: 1, pos: 38
type: A, layer: 1, pos: 11
type: A, layer: 1, pos: 43
type: A, layer: 1, pos: 17
type: B, layer: 1, pos: 45
type: A, layer: 1, pos: 3
type: A, layer: 1, pos: 15
type: B, layer: 1, pos: 32
type: A, layer: 1, pos: 39
type: B, layer: 1, pos: 15
type: A, layer: 1, pos: 10
type: B, layer: 1, pos: 37
type: B, layer: 1, pos: 20
type: B, layer: 1, pos: 39
type: A, layer: 1, pos: 23
type: A, layer: 1, pos: 41
type: A, layer: 1, pos: 5
type: A, layer: 1, pos: 37
type: A, layer: 1, pos: 16
type: B, layer: 1, pos: 41
type: B, layer: 1, pos: 18
type: A, layer: 1, pos: 32
type: B, layer: 1, pos: 21
type: A, layer: 1, pos: 21
type: A, layer: 1, pos: 20
type: B, layer: 1, pos: 5
type: B, layer: 1, pos: 23
type: B, layer: 1, pos: 16
type: A, layer: 1, pos: 38
type: B, layer: 1, pos: 43
type: A, layer: 1, pos: 29

Time for candidate selection: 0.10 seconds

### Candidate
type: B, layer: 1, pos: 29

## Relational analysis of NS_B1_B2_A1_A1_A2_B1_B1
Optimization infeasible because this subproblem isn't reachable.

### Candidate
type: A, layer: 1, pos: 9

## Relational analysis of NS_B1_B2_A1_A1_A2_B1_A1
Optimization infeasible because this subproblem isn't reachable.

### Candidate
type: B, layer: 1, pos: 35

## Relational analysis of NS_B1_B2_A1_A1_A2_B1_B1
Optimization infeasible because this subproblem isn't reachable.

### Candidate
type: B, layer: 1, pos: 17

## Relational analysis of NS_B1_B2_A1_A1_A2_B1_B1

### Relational analysis result of NS_B1_B2_A1_A1_A2_B1_B1
Status: Status.UNKNOWN
Output dim: 0, lower bound: -63.3953415, upper bound: 63.3955360
time: 0.76 seconds

## Relational analysis of NS_B1_B2_A1_A1_A2_B1_B2

### Relational analysis result of NS_B1_B2_A1_A1_A2_B1_B2
Status: Status.UNKNOWN
Output dim: 0, lower bound: -63.3957590, upper bound: 63.3958450
time: 0.80 seconds

## BFS NS instance: NS_B1_B2_A1_A1_A2_B2

### Backsubstitution after applying NS history:
0: -43.3717232, 44.9401779, -44.6238937, 46.3271980, -89.6979904, 89.5640640
1: -303.9672852, 107.6393127, -313.9445801, 110.7814026, -413.4206848, 420.6076355
2: -170.4709625, 98.0167847, -175.7769775, 100.9844360, -270.3331604, 272.9084167
3: -213.3660736, 78.5642319, -220.2172699, 80.9449158, -292.6127319, 297.3772583
4: -123.1605453, 85.3021240, -126.8017273, 87.9186707, -210.5055847, 211.6747131

Time for backsubstitution: 1.30 seconds

### NS candidates at layer 1
type: B, layer: 1, pos: 9
type: A, layer: 1, pos: 9
type: B, layer: 1, pos: 35
type: A, layer: 1, pos: 35
type: A, layer: 1, pos: 17
type: A, layer: 1, pos: 19
type: B, layer: 1, pos: 17
type: A, layer: 1, pos: 45
type: B, layer: 1, pos: 19
type: B, layer: 1, pos: 45
type: B, layer: 1, pos: 3
type: A, layer: 1, pos: 3
type: B, layer: 1, pos: 11
type: A, layer: 1, pos: 11
type: B, layer: 1, pos: 10
type: A, layer: 1, pos: 10
type: A, layer: 1, pos: 18
type: B, layer: 1, pos: 15
type: A, layer: 1, pos: 15
type: A, layer: 1, pos: 38
type: B, layer: 1, pos: 32
type: B, layer: 1, pos: 18
type: B, layer: 1, pos: 39
type: A, layer: 1, pos: 43
type: A, layer: 1, pos: 39
type: B, layer: 1, pos: 29
type: A, layer: 1, pos: 32
type: A, layer: 1, pos: 37
type: B, layer: 1, pos: 37
type: B, layer: 1, pos: 43
type: A, layer: 1, pos: 41
type: B, layer: 1, pos: 41
type: A, layer: 1, pos: 20
type: B, layer: 1, pos: 38
type: B, layer: 1, pos: 21
type: B, layer: 1, pos: 20
type: A, layer: 1, pos: 5
type: A, layer: 1, pos: 29
type: B, layer: 1, pos: 23
type: B, layer: 1, pos: 5
type: A, layer: 1, pos: 21
type: A, layer: 1, pos: 23
type: B, layer: 1, pos: 16
type: A, layer: 1, pos: 16

Time for candidate selection: 0.11 seconds

### Candidate
type: B, layer: 1, pos: 9

## Relational analysis of NS_B1_B2_A1_A1_A2_B2_B1

### Relational analysis result of NS_B1_B2_A1_A1_A2_B2_B1
Status: Status.UNKNOWN
Output dim: 0, lower bound: -63.3933914, upper bound: 63.3935634
time: 0.65 seconds

## Relational analysis of NS_B1_B2_A1_A1_A2_B2_B2

### Relational analysis result of NS_B1_B2_A1_A1_A2_B2_B2
Status: Status.UNKNOWN
Output dim: 0, lower bound: -63.3948832, upper bound: 63.3950468
time: 0.77 seconds

## BFS NS instance: NS_B1_B2_A1_A2_B1_B1

### Backsubstitution after applying NS history:
0: -29.9004688, 31.5799618, -28.9615479, 30.5737095, -60.4741783, 60.5415115
1: -213.7520294, 73.9493179, -207.0983124, 71.5728531, -285.3248901, 281.0476379
2: -117.7341156, 68.0566254, -113.8917847, 65.9328995, -183.6670227, 181.9483795
3: -148.2603760, 54.5692177, -143.5340271, 52.8391800, -201.0995331, 198.1032410
4: -84.7556610, 59.6533127, -81.9813232, 57.7735481, -142.5291748, 141.6346130

Time for backsubstitution: 1.02 seconds

### NS candidates at layer 1
type: A, layer: 1, pos: 29
type: B, layer: 1, pos: 29
type: B, layer: 1, pos: 35
type: A, layer: 1, pos: 35
type: A, layer: 1, pos: 19
type: B, layer: 1, pos: 19
type: A, layer: 1, pos: 45
type: B, layer: 1, pos: 45
type: A, layer: 1, pos: 4
type: A, layer: 1, pos: 9
type: A, layer: 1, pos: 3
type: B, layer: 1, pos: 3
type: A, layer: 1, pos: 38
type: B, layer: 1, pos: 38
type: A, layer: 1, pos: 15
type: B, layer: 1, pos: 15
type: B, layer: 1, pos: 11
type: A, layer: 1, pos: 11
type: B, layer: 1, pos: 21
type: B, layer: 1, pos: 17
type: A, layer: 1, pos: 21
type: A, layer: 1, pos: 18
type: B, layer: 1, pos: 18
type: B, layer: 1, pos: 23
type: B, layer: 1, pos: 10
type: A, layer: 1, pos: 17
type: A, layer: 1, pos: 23
type: B, layer: 1, pos: 20
type: B, layer: 1, pos: 32
type: A, layer: 1, pos: 10
type: A, layer: 1, pos: 20
type: B, layer: 1, pos: 43
type: A, layer: 1, pos: 37
type: A, layer: 1, pos: 43
type: A, layer: 1, pos: 16
type: A, layer: 1, pos: 32
type: B, layer: 1, pos: 16
type: B, layer: 1, pos: 41
type: B, layer: 1, pos: 37
type: A, layer: 1, pos: 41
type: A, layer: 1, pos: 39
type: B, layer: 1, pos: 39
type: A, layer: 1, pos: 5
type: B, layer: 1, pos: 5

Time for candidate selection: 0.08 seconds

### Candidate
type: A, layer: 1, pos: 29

## Relational analysis of NS_B1_B2_A1_A2_B1_B1_A1
Optimization infeasible because this subproblem isn't reachable.

### Candidate
type: B, layer: 1, pos: 29

### Candidate
type: B, layer: 1, pos: 35

## Relational analysis of NS_B1_B2_A1_A2_B1_B1_B1
Optimization infeasible because this subproblem isn't reachable.

### Candidate
type: A, layer: 1, pos: 35

### Candidate
type: A, layer: 1, pos: 19

## Relational analysis of NS_B1_B2_A1_A2_B1_B1_A1
Optimization infeasible because this subproblem isn't reachable.

### Candidate
type: B, layer: 1, pos: 19

### Candidate
type: A, layer: 1, pos: 45

## Relational analysis of NS_B1_B2_A1_A2_B1_B1_A1
Optimization infeasible because this subproblem isn't reachable.

### Candidate
type: B, layer: 1, pos: 45

### Candidate
type: A, layer: 1, pos: 4

## Relational analysis of NS_B1_B2_A1_A2_B1_B1_A1

### Relational analysis result of NS_B1_B2_A1_A2_B1_B1_A1
Status: Status.UNKNOWN
Output dim: 0, lower bound: -63.3938505, upper bound: 63.3932262
time: 0.77 seconds

## Relational analysis of NS_B1_B2_A1_A2_B1_B1_A2

### Relational analysis result of NS_B1_B2_A1_A2_B1_B1_A2
Status: Status.UNKNOWN
Output dim: 0, lower bound: -63.3938505, upper bound: 63.3932262
time: 0.68 seconds

## BFS NS instance: NS_B1_B2_A1_A2_B1_B2

### Backsubstitution after applying NS history:
0: -30.6833210, 32.3495865, -29.0219860, 30.6075287, -61.2908478, 61.3715591
1: -218.6328583, 75.9537354, -206.3379974, 71.8889771, -290.5218201, 282.2917175
2: -121.3715286, 69.8045197, -114.9485168, 66.0204163, -187.3919373, 184.7530060
3: -152.1492310, 55.9572372, -143.8393707, 52.9667053, -205.1159210, 199.7966003
4: -87.3655548, 61.1791573, -82.7645950, 57.8989754, -145.2645264, 143.9437408

Time for backsubstitution: 1.06 seconds

### NS candidates at layer 1
type: A, layer: 1, pos: 35
type: B, layer: 1, pos: 35
type: A, layer: 1, pos: 9
type: A, layer: 1, pos: 19
type: B, layer: 1, pos: 19
type: B, layer: 1, pos: 17
type: A, layer: 1, pos: 17
type: A, layer: 1, pos: 4
type: A, layer: 1, pos: 11
type: B, layer: 1, pos: 11
type: A, layer: 1, pos: 3
type: B, layer: 1, pos: 3
type: A, layer: 1, pos: 15
type: B, layer: 1, pos: 10
type: B, layer: 1, pos: 15
type: A, layer: 1, pos: 43
type: A, layer: 1, pos: 10
type: B, layer: 1, pos: 43
type: A, layer: 1, pos: 18
type: B, layer: 1, pos: 45
type: B, layer: 1, pos: 21
type: B, layer: 1, pos: 18
type: A, layer: 1, pos: 21
type: B, layer: 1, pos: 32
type: A, layer: 1, pos: 45
type: A, layer: 1, pos: 37
type: B, layer: 1, pos: 29
type: A, layer: 1, pos: 23
type: B, layer: 1, pos: 23
type: A, layer: 1, pos: 38
type: A, layer: 1, pos: 29
type: A, layer: 1, pos: 32
type: B, layer: 1, pos: 20
type: B, layer: 1, pos: 37
type: B, layer: 1, pos: 38
type: B, layer: 1, pos: 41
type: A, layer: 1, pos: 39
type: A, layer: 1, pos: 20
type: A, layer: 1, pos: 41
type: B, layer: 1, pos: 39
type: B, layer: 1, pos: 5
type: B, layer: 1, pos: 16
type: A, layer: 1, pos: 5
type: A, layer: 1, pos: 16

Time for candidate selection: 0.08 seconds

### Candidate
type: A, layer: 1, pos: 35

## Relational analysis of NS_B1_B2_A1_A2_B1_B2_A1
Optimization infeasible because this subproblem isn't reachable.

### Candidate
type: B, layer: 1, pos: 35

### Candidate
type: A, layer: 1, pos: 9

## Relational analysis of NS_B1_B2_A1_A2_B1_B2_A1

### Relational analysis result of NS_B1_B2_A1_A2_B1_B2_A1
Status: Status.UNKNOWN
Output dim: 0, lower bound: -63.3939272, upper bound: 63.3934434
time: 0.73 seconds

## Relational analysis of NS_B1_B2_A1_A2_B1_B2_A2

### Relational analysis result of NS_B1_B2_A1_A2_B1_B2_A2
Status: Status.UNKNOWN
Output dim: 0, lower bound: -63.3939272, upper bound: 63.3947308
time: 0.77 seconds

## BFS NS instance: NS_B1_B2_A1_A2_B2_A1

### Backsubstitution after applying NS history:
0: -30.1823769, 31.8147984, -44.6238937, 46.3271980, -76.5095673, 76.4386902
1: -214.8651276, 74.6638260, -313.9445801, 110.7814026, -325.6465454, 388.6083984
2: -119.0411377, 68.5830307, -175.7769775, 100.9844360, -220.0255737, 244.3600159
3: -149.3636475, 54.9800568, -220.2172699, 80.9449158, -230.3085632, 275.1972656
4: -85.7541580, 60.1135139, -126.8017273, 87.9186707, -173.6728210, 186.9152374

Time for backsubstitution: 1.02 seconds

### NS candidates at layer 1
type: A, layer: 1, pos: 29
type: B, layer: 1, pos: 9
type: A, layer: 1, pos: 17
type: A, layer: 1, pos: 35
type: A, layer: 1, pos: 9
type: B, layer: 1, pos: 35
type: B, layer: 1, pos: 45
type: B, layer: 1, pos: 19
type: A, layer: 1, pos: 10
type: A, layer: 1, pos: 3
type: A, layer: 1, pos: 38
type: A, layer: 1, pos: 19
type: B, layer: 1, pos: 18
type: A, layer: 1, pos: 11
type: B, layer: 1, pos: 11
type: A, layer: 1, pos: 45
type: B, layer: 1, pos: 17
type: B, layer: 1, pos: 3
type: B, layer: 1, pos: 43
type: B, layer: 1, pos: 15
type: B, layer: 1, pos: 39
type: A, layer: 1, pos: 32
type: B, layer: 1, pos: 10
type: A, layer: 1, pos: 15
type: A, layer: 1, pos: 37
type: A, layer: 1, pos: 20
type: B, layer: 1, pos: 23
type: A, layer: 1, pos: 39
type: B, layer: 1, pos: 41
type: A, layer: 1, pos: 18
type: B, layer: 1, pos: 5
type: B, layer: 1, pos: 32
type: B, layer: 1, pos: 16
type: B, layer: 1, pos: 37
type: A, layer: 1, pos: 41
type: B, layer: 1, pos: 21
type: A, layer: 1, pos: 21
type: B, layer: 1, pos: 20
type: A, layer: 1, pos: 5
type: A, layer: 1, pos: 23
type: A, layer: 1, pos: 16
type: A, layer: 1, pos: 43
type: B, layer: 1, pos: 38
type: B, layer: 1, pos: 29

Time for candidate selection: 0.09 seconds

### Candidate
type: A, layer: 1, pos: 29

## Relational analysis of NS_B1_B2_A1_A2_B2_A1_A1
Optimization infeasible because this subproblem isn't reachable.

### Candidate
type: B, layer: 1, pos: 9

## Relational analysis of NS_B1_B2_A1_A2_B2_A1_B1

### Relational analysis result of NS_B1_B2_A1_A2_B2_A1_B1
Status: Status.UNKNOWN
Output dim: 0, lower bound: -63.3927605, upper bound: 63.3926815
time: 0.69 seconds

## Relational analysis of NS_B1_B2_A1_A2_B2_A1_B2

### Relational analysis result of NS_B1_B2_A1_A2_B2_A1_B2
Status: Status.UNKNOWN
Output dim: 0, lower bound: -63.3942151, upper bound: 63.3942151
time: 0.77 seconds

## BFS NS instance: NS_B1_B2_A1_A2_B2_A2

### Backsubstitution after applying NS history:
0: -44.6238937, 46.3271980, -44.6238937, 46.3271980, -90.9510880, 90.9510880
1: -313.9445801, 110.7814026, -313.9445801, 110.7814026, -424.7259827, 424.7259827
2: -175.7769775, 100.9844360, -175.7769775, 100.9844360, -276.7614136, 276.7614136
3: -220.2172699, 80.9449158, -220.2172699, 80.9449158, -301.1621704, 301.1621704
4: -126.8017273, 87.9186707, -126.8017273, 87.9186707, -214.7203979, 214.7203979

Time for backsubstitution: 1.16 seconds

### NS candidates at layer 1
type: A, layer: 1, pos: 9
type: B, layer: 1, pos: 9
type: A, layer: 1, pos: 35
type: B, layer: 1, pos: 35
type: A, layer: 1, pos: 17
type: B, layer: 1, pos: 17
type: B, layer: 1, pos: 19
type: A, layer: 1, pos: 19
type: B, layer: 1, pos: 45
type: A, layer: 1, pos: 45
type: A, layer: 1, pos: 3
type: B, layer: 1, pos: 3
type: B, layer: 1, pos: 11
type: A, layer: 1, pos: 11
type: A, layer: 1, pos: 10
type: B, layer: 1, pos: 10
type: A, layer: 1, pos: 15
type: B, layer: 1, pos: 15
type: B, layer: 1, pos: 18
type: A, layer: 1, pos: 18
type: A, layer: 1, pos: 39
type: B, layer: 1, pos: 39
type: A, layer: 1, pos: 32
type: B, layer: 1, pos: 32
type: B, layer: 1, pos: 43
type: A, layer: 1, pos: 43
type: B, layer: 1, pos: 38
type: A, layer: 1, pos: 38
type: A, layer: 1, pos: 37
type: B, layer: 1, pos: 37
type: A, layer: 1, pos: 29
type: B, layer: 1, pos: 29
type: A, layer: 1, pos: 41
type: B, layer: 1, pos: 41
type: A, layer: 1, pos: 20
type: B, layer: 1, pos: 20
type: A, layer: 1, pos: 21
type: B, layer: 1, pos: 21
type: B, layer: 1, pos: 5
type: A, layer: 1, pos: 5
type: B, layer: 1, pos: 23
type: A, layer: 1, pos: 23
type: A, layer: 1, pos: 16
type: B, layer: 1, pos: 16

Time for candidate selection: 0.08 seconds

### Candidate
type: A, layer: 1, pos: 9

## Relational analysis of NS_B1_B2_A1_A2_B2_A2_A1

### Relational analysis result of NS_B1_B2_A1_A2_B2_A2_A1
Status: Status.UNKNOWN
Output dim: 0, lower bound: -63.3935071, upper bound: 63.3933773
time: 1.19 seconds

## Relational analysis of NS_B1_B2_A1_A2_B2_A2_A2

### Relational analysis result of NS_B1_B2_A1_A2_B2_A2_A2
Status: Status.UNKNOWN
Output dim: 0, lower bound: -63.3942151, upper bound: 63.3942151
time: 0.72 seconds

## BFS NS instance: NS_B1_B2_A2_B1_B1_B1

### Backsubstitution after applying NS history:
0: -52.1692619, 56.9168472, -29.3263874, 30.9956894, -83.1649475, 86.2432098
1: -388.7693787, 128.8978271, -210.7784882, 72.5411301, -461.3104553, 339.6763000
2: -206.5955811, 121.9235916, -115.7335205, 66.8332214, -273.4288025, 237.6571045
3: -267.5608521, 97.7861481, -145.9232483, 53.5703850, -321.1312256, 243.7093658
4: -147.7406464, 107.9198608, -83.2501450, 58.5407257, -206.2813263, 191.1699677

Time for backsubstitution: 1.04 seconds

### NS candidates at layer 1
type: B, layer: 1, pos: 4
type: B, layer: 1, pos: 3
type: A, layer: 1, pos: 4
type: A, layer: 1, pos: 35
type: B, layer: 1, pos: 38
type: B, layer: 1, pos: 35
type: A, layer: 1, pos: 3
type: A, layer: 1, pos: 7
type: B, layer: 1, pos: 45
type: B, layer: 1, pos: 17
type: A, layer: 1, pos: 43
type: A, layer: 1, pos: 9
type: A, layer: 1, pos: 15
type: A, layer: 1, pos: 19
type: A, layer: 1, pos: 45
type: B, layer: 1, pos: 21
type: A, layer: 1, pos: 11
type: A, layer: 1, pos: 38
type: B, layer: 1, pos: 11
type: A, layer: 1, pos: 23
type: B, layer: 1, pos: 43
type: A, layer: 1, pos: 21
type: A, layer: 1, pos: 39
type: B, layer: 1, pos: 10
type: A, layer: 1, pos: 18
type: A, layer: 1, pos: 10
type: B, layer: 1, pos: 37
type: A, layer: 1, pos: 16
type: B, layer: 1, pos: 23
type: B, layer: 1, pos: 15
type: B, layer: 1, pos: 20
type: B, layer: 1, pos: 18
type: A, layer: 1, pos: 20
type: B, layer: 1, pos: 32
type: A, layer: 1, pos: 32
type: B, layer: 1, pos: 41
type: A, layer: 1, pos: 17
type: B, layer: 1, pos: 39
type: A, layer: 1, pos: 41
type: B, layer: 1, pos: 5
type: A, layer: 1, pos: 5
type: A, layer: 1, pos: 37
type: B, layer: 1, pos: 16
type: A, layer: 1, pos: 29

Time for candidate selection: 0.08 seconds

### Candidate
type: B, layer: 1, pos: 4

### Candidate
type: B, layer: 1, pos: 3

### Candidate
type: A, layer: 1, pos: 4

### Candidate
type: A, layer: 1, pos: 35

### Candidate
type: B, layer: 1, pos: 38

## Relational analysis of NS_B1_B2_A2_B1_B1_B1_B1
Optimization infeasible because this subproblem isn't reachable.

### Candidate
type: B, layer: 1, pos: 35

### Candidate
type: A, layer: 1, pos: 3

### Candidate
type: A, layer: 1, pos: 7

## Relational analysis of NS_B1_B2_A2_B1_B1_B1_A1
Optimization infeasible because this subproblem isn't reachable.

### Candidate
type: B, layer: 1, pos: 45

## Relational analysis of NS_B1_B2_A2_B1_B1_B1_B1

### Relational analysis result of NS_B1_B2_A2_B1_B1_B1_B1
Status: Status.UNKNOWN
Output dim: 0, lower bound: -63.3930653, upper bound: 63.3933406
time: 0.67 seconds

## Relational analysis of NS_B1_B2_A2_B1_B1_B1_B2
Optimization infeasible because this subproblem isn't reachable.

### Candidate
type: B, layer: 1, pos: 17

## Relational analysis of NS_B1_B2_A2_B1_B1_B1_B1
Optimization infeasible because this subproblem isn't reachable.

### Candidate
type: A, layer: 1, pos: 43

## Relational analysis of NS_B1_B2_A2_B1_B1_B1_A1

### Relational analysis result of NS_B1_B2_A2_B1_B1_B1_A1
Status: Status.UNKNOWN
Output dim: 0, lower bound: -63.3936230, upper bound: 63.3939962
time: 0.73 seconds

## Relational analysis of NS_B1_B2_A2_B1_B1_B1_A2
Optimization infeasible because this subproblem isn't reachable.

### Candidate
type: A, layer: 1, pos: 9

## Relational analysis of NS_B1_B2_A2_B1_B1_B1_A1

### Relational analysis result of NS_B1_B2_A2_B1_B1_B1_A1
Status: Status.UNKNOWN
Output dim: 0, lower bound: -63.3935115, upper bound: 63.3939136
time: 0.85 seconds

## Relational analysis of NS_B1_B2_A2_B1_B1_B1_A2

### Relational analysis result of NS_B1_B2_A2_B1_B1_B1_A2
Status: Status.UNKNOWN
Output dim: 0, lower bound: -63.3935115, upper bound: 63.3940085
time: 0.67 seconds

## BFS NS instance: NS_B1_B2_A2_B1_B1_B2

### Backsubstitution after applying NS history:
0: -52.0881042, 56.7848129, -31.6310120, 33.5251617, -85.6132660, 88.4158173
1: -387.7741089, 128.6701813, -229.6852875, 78.3447952, -466.1188660, 358.3554688
2: -206.2195435, 121.6504974, -126.0184631, 72.2825851, -278.5021362, 247.6689606
3: -266.9662170, 97.5603180, -159.2089539, 57.9216194, -324.8878479, 256.7692566
4: -147.4947510, 107.6608124, -90.2369766, 63.2144165, -210.7091217, 197.8977814

Time for backsubstitution: 1.02 seconds

### NS candidates at layer 1
type: B, layer: 1, pos: 4
type: B, layer: 1, pos: 3
type: A, layer: 1, pos: 35
type: B, layer: 1, pos: 38
type: B, layer: 1, pos: 35
type: A, layer: 1, pos: 7
type: A, layer: 1, pos: 3
type: B, layer: 1, pos: 17
type: A, layer: 1, pos: 9
type: B, layer: 1, pos: 45
type: A, layer: 1, pos: 43
type: A, layer: 1, pos: 15
type: A, layer: 1, pos: 11
type: B, layer: 1, pos: 11
type: A, layer: 1, pos: 18
type: B, layer: 1, pos: 10
type: A, layer: 1, pos: 10
type: A, layer: 1, pos: 4
type: A, layer: 1, pos: 45
type: A, layer: 1, pos: 23
type: B, layer: 1, pos: 18
type: A, layer: 1, pos: 39
type: B, layer: 1, pos: 43
type: B, layer: 1, pos: 15
type: B, layer: 1, pos: 21
type: A, layer: 1, pos: 19
type: A, layer: 1, pos: 17
type: A, layer: 1, pos: 16
type: A, layer: 1, pos: 21
type: B, layer: 1, pos: 37
type: A, layer: 1, pos: 32
type: B, layer: 1, pos: 20
type: B, layer: 1, pos: 23
type: B, layer: 1, pos: 41
type: B, layer: 1, pos: 32
type: A, layer: 1, pos: 41
type: A, layer: 1, pos: 20
type: B, layer: 1, pos: 5
type: A, layer: 1, pos: 5
type: B, layer: 1, pos: 39
type: A, layer: 1, pos: 37
type: B, layer: 1, pos: 16
type: A, layer: 1, pos: 38
type: A, layer: 1, pos: 29

Time for candidate selection: 0.08 seconds

### Candidate
type: B, layer: 1, pos: 4

### Candidate
type: B, layer: 1, pos: 3

### Candidate
type: A, layer: 1, pos: 35

### Candidate
type: B, layer: 1, pos: 38

## Relational analysis of NS_B1_B2_A2_B1_B1_B2_B1
Optimization infeasible because this subproblem isn't reachable.

### Candidate
type: B, layer: 1, pos: 35

### Candidate
type: A, layer: 1, pos: 7

## Relational analysis of NS_B1_B2_A2_B1_B1_B2_A1
Optimization infeasible because this subproblem isn't reachable.

### Candidate
type: A, layer: 1, pos: 3

### Candidate
type: B, layer: 1, pos: 17

## Relational analysis of NS_B1_B2_A2_B1_B1_B2_B1
Optimization infeasible because this subproblem isn't reachable.

### Candidate
type: A, layer: 1, pos: 9

## Relational analysis of NS_B1_B2_A2_B1_B1_B2_A1

### Relational analysis result of NS_B1_B2_A2_B1_B1_B2_A1
Status: Status.UNKNOWN
Output dim: 0, lower bound: -63.3935115, upper bound: 63.3940906
time: 0.84 seconds

## Relational analysis of NS_B1_B2_A2_B1_B1_B2_A2

### Relational analysis result of NS_B1_B2_A2_B1_B1_B2_A2
Status: Status.UNKNOWN
Output dim: 0, lower bound: -63.3939212, upper bound: 63.3941125
time: 0.79 seconds

## BFS NS instance: NS_B1_B2_A2_B1_B2_A1

### Backsubstitution after applying NS history:
0: -51.6358986, 56.2629089, -39.6053352, 40.7887802, -92.4246826, 95.8682404
1: -383.5046692, 127.3618622, -267.8989563, 98.0441742, -480.7815247, 395.2607727
2: -203.9217682, 120.5108871, -153.9289856, 88.2534256, -292.1751404, 274.4091492
3: -263.8349609, 96.5557404, -188.3889313, 70.6072540, -334.4421997, 284.9446106
4: -145.8617401, 106.7511292, -112.4031372, 77.2734604, -223.1351929, 218.6411133

Time for backsubstitution: 1.05 seconds

### NS candidates at layer 1
type: B, layer: 1, pos: 4
type: B, layer: 1, pos: 3
type: A, layer: 1, pos: 35
type: B, layer: 1, pos: 19
type: B, layer: 1, pos: 35
type: A, layer: 1, pos: 7
type: B, layer: 1, pos: 17
type: B, layer: 1, pos: 38
type: A, layer: 1, pos: 3
type: A, layer: 1, pos: 10
type: A, layer: 1, pos: 19
type: A, layer: 1, pos: 15
type: B, layer: 1, pos: 45
type: A, layer: 1, pos: 4
type: B, layer: 1, pos: 18
type: A, layer: 1, pos: 11
type: A, layer: 1, pos: 43
type: B, layer: 1, pos: 11
type: B, layer: 1, pos: 43
type: A, layer: 1, pos: 17
type: A, layer: 1, pos: 29
type: A, layer: 1, pos: 32
type: B, layer: 1, pos: 37
type: B, layer: 1, pos: 21
type: A, layer: 1, pos: 16
type: A, layer: 1, pos: 18
type: B, layer: 1, pos: 10
type: A, layer: 1, pos: 21
type: B, layer: 1, pos: 15
type: A, layer: 1, pos: 23
type: A, layer: 1, pos: 20
type: A, layer: 1, pos: 39
type: B, layer: 1, pos: 23
type: A, layer: 1, pos: 45
type: B, layer: 1, pos: 41
type: B, layer: 1, pos: 32
type: A, layer: 1, pos: 41
type: B, layer: 1, pos: 39
type: B, layer: 1, pos: 5
type: A, layer: 1, pos: 5
type: B, layer: 1, pos: 20
type: A, layer: 1, pos: 38
type: B, layer: 1, pos: 16
type: A, layer: 1, pos: 37

Time for candidate selection: 0.09 seconds

### Candidate
type: B, layer: 1, pos: 4

### Candidate
type: B, layer: 1, pos: 3

### Candidate
type: A, layer: 1, pos: 35

### Candidate
type: B, layer: 1, pos: 19

### Candidate
type: B, layer: 1, pos: 35

### Candidate
type: A, layer: 1, pos: 7

### Candidate
type: B, layer: 1, pos: 17

### Candidate
type: B, layer: 1, pos: 38

## Relational analysis of NS_B1_B2_A2_B1_B2_A1_B1
Optimization infeasible because this subproblem isn't reachable.

### Candidate
type: A, layer: 1, pos: 3

### Candidate
type: A, layer: 1, pos: 10

## Relational analysis of NS_B1_B2_A2_B1_B2_A1_A1

### Relational analysis result of NS_B1_B2_A2_B1_B2_A1_A1
Status: Status.UNKNOWN
Output dim: 0, lower bound: -63.3935359, upper bound: 63.3934718
time: 0.69 seconds

## Relational analysis of NS_B1_B2_A2_B1_B2_A1_A2

### Relational analysis result of NS_B1_B2_A2_B1_B2_A1_A2
Status: Status.UNKNOWN
Output dim: 0, lower bound: -63.3935272, upper bound: 63.3934717
time: 0.72 seconds

## BFS NS instance: NS_B1_B2_A2_B1_B2_A2

### Backsubstitution after applying NS history:
0: -52.0477753, 56.5673981, -39.6053352, 40.7887802, -92.8365555, 96.1727295
1: -383.9574585, 128.3494415, -267.8989563, 98.0441742, -481.3737488, 396.2484131
2: -205.3651123, 121.1593781, -153.9289856, 88.2534256, -293.6185303, 275.0788879
3: -264.8219299, 97.0561752, -188.3889313, 70.6072540, -335.4291687, 285.4450378
4: -147.3216705, 107.3162003, -112.4031372, 77.2734604, -224.5951233, 219.2324219

Time for backsubstitution: 1.04 seconds

### NS candidates at layer 1
type: B, layer: 1, pos: 4
type: B, layer: 1, pos: 3
type: A, layer: 1, pos: 35
type: B, layer: 1, pos: 19
type: B, layer: 1, pos: 35
type: A, layer: 1, pos: 7
type: B, layer: 1, pos: 17
type: A, layer: 1, pos: 3
type: B, layer: 1, pos: 38
type: A, layer: 1, pos: 19
type: A, layer: 1, pos: 15
type: A, layer: 1, pos: 10
type: B, layer: 1, pos: 45
type: A, layer: 1, pos: 11
type: B, layer: 1, pos: 18
type: B, layer: 1, pos: 11
type: A, layer: 1, pos: 43
type: A, layer: 1, pos: 4
type: B, layer: 1, pos: 43
type: A, layer: 1, pos: 17
type: A, layer: 1, pos: 29
type: A, layer: 1, pos: 32
type: B, layer: 1, pos: 10
type: B, layer: 1, pos: 37
type: B, layer: 1, pos: 21
type: A, layer: 1, pos: 16
type: A, layer: 1, pos: 18
type: B, layer: 1, pos: 15
type: A, layer: 1, pos: 21
type: A, layer: 1, pos: 20
type: A, layer: 1, pos: 23
type: A, layer: 1, pos: 45
type: A, layer: 1, pos: 39
type: B, layer: 1, pos: 23
type: B, layer: 1, pos: 32
type: B, layer: 1, pos: 41
type: A, layer: 1, pos: 41
type: B, layer: 1, pos: 39
type: B, layer: 1, pos: 5
type: B, layer: 1, pos: 20
type: A, layer: 1, pos: 5
type: A, layer: 1, pos: 38
type: A, layer: 1, pos: 37
type: B, layer: 1, pos: 16

Time for candidate selection: 0.08 seconds

### Candidate
type: B, layer: 1, pos: 4

### Candidate
type: B, layer: 1, pos: 3

### Candidate
type: A, layer: 1, pos: 35

### Candidate
type: B, layer: 1, pos: 19

### Candidate
type: B, layer: 1, pos: 35

### Candidate
type: A, layer: 1, pos: 7

### Candidate
type: B, layer: 1, pos: 17

### Candidate
type: A, layer: 1, pos: 3

### Candidate
type: B, layer: 1, pos: 38

## Relational analysis of NS_B1_B2_A2_B1_B2_A2_B1
Optimization infeasible because this subproblem isn't reachable.

### Candidate
type: A, layer: 1, pos: 19

### Candidate
type: A, layer: 1, pos: 15

## Relational analysis of NS_B1_B2_A2_B1_B2_A2_A1
Optimization infeasible because this subproblem isn't reachable.

### Candidate
type: A, layer: 1, pos: 10

## Relational analysis of NS_B1_B2_A2_B1_B2_A2_A1

### Relational analysis result of NS_B1_B2_A2_B1_B2_A2_A1
Status: Status.UNKNOWN
Output dim: 0, lower bound: -63.3935359, upper bound: 63.3934797
time: 0.72 seconds

## Relational analysis of NS_B1_B2_A2_B1_B2_A2_A2

### Relational analysis result of NS_B1_B2_A2_B1_B2_A2_A2
Status: Status.UNKNOWN
Output dim: 0, lower bound: -63.3935272, upper bound: 63.3934797
time: 0.66 seconds

## BFS NS instance: NS_B1_B2_A2_B2_B1_B1

### Backsubstitution after applying NS history:
0: -52.7859306, 57.4743652, -29.3598194, 30.9858704, -83.7717896, 86.8341827
1: -391.9863281, 130.3419800, -210.1095276, 72.7058563, -464.6921387, 340.4515076
2: -208.8900909, 123.1569595, -116.5262222, 66.8786011, -275.7686768, 239.6831818
3: -270.0416260, 98.7194366, -146.1479340, 53.5919342, -323.6334534, 244.8673401
4: -149.5399780, 108.9858856, -83.7368774, 58.5537834, -208.0937347, 192.7227631

Time for backsubstitution: 1.05 seconds

### NS candidates at layer 1
type: B, layer: 1, pos: 4
type: B, layer: 1, pos: 3
type: A, layer: 1, pos: 35
type: A, layer: 1, pos: 3
type: B, layer: 1, pos: 35
type: B, layer: 1, pos: 38
type: A, layer: 1, pos: 7
type: B, layer: 1, pos: 17
type: A, layer: 1, pos: 43
type: A, layer: 1, pos: 9
type: A, layer: 1, pos: 4
type: A, layer: 1, pos: 15
type: B, layer: 1, pos: 45
type: A, layer: 1, pos: 11
type: B, layer: 1, pos: 11
type: B, layer: 1, pos: 43
type: B, layer: 1, pos: 10
type: A, layer: 1, pos: 10
type: A, layer: 1, pos: 19
type: A, layer: 1, pos: 18
type: B, layer: 1, pos: 21
type: A, layer: 1, pos: 23
type: A, layer: 1, pos: 16
type: A, layer: 1, pos: 17
type: A, layer: 1, pos: 45
type: B, layer: 1, pos: 37
type: A, layer: 1, pos: 21
type: A, layer: 1, pos: 39
type: B, layer: 1, pos: 18
type: B, layer: 1, pos: 23
type: B, layer: 1, pos: 15
type: B, layer: 1, pos: 32
type: A, layer: 1, pos: 32
type: A, layer: 1, pos: 20
type: B, layer: 1, pos: 41
type: B, layer: 1, pos: 20
type: A, layer: 1, pos: 41
type: B, layer: 1, pos: 5
type: B, layer: 1, pos: 39
type: A, layer: 1, pos: 5
type: A, layer: 1, pos: 38
type: A, layer: 1, pos: 37
type: B, layer: 1, pos: 16
type: A, layer: 1, pos: 29

Time for candidate selection: 0.08 seconds

### Candidate
type: B, layer: 1, pos: 4

### Candidate
type: B, layer: 1, pos: 3

### Candidate
type: A, layer: 1, pos: 35

### Candidate
type: A, layer: 1, pos: 3

### Candidate
type: B, layer: 1, pos: 35

### Candidate
type: B, layer: 1, pos: 38

## Relational analysis of NS_B1_B2_A2_B2_B1_B1_B1
Optimization infeasible because this subproblem isn't reachable.

### Candidate
type: A, layer: 1, pos: 7

## Relational analysis of NS_B1_B2_A2_B2_B1_B1_A1

### Relational analysis result of NS_B1_B2_A2_B2_B1_B1_A1
Status: Status.UNKNOWN
Output dim: 0, lower bound: -63.3934536, upper bound: 63.3934069
time: 0.70 seconds

## Relational analysis of NS_B1_B2_A2_B2_B1_B1_A2

### Relational analysis result of NS_B1_B2_A2_B2_B1_B1_A2
Status: Status.UNKNOWN
Output dim: 0, lower bound: -63.3934536, upper bound: 63.3941899
time: 0.84 seconds

## BFS NS instance: NS_B1_B2_A2_B2_B1_B2

### Backsubstitution after applying NS history:
0: -52.2217522, 56.7749939, -39.1467552, 40.2921753, -92.5139236, 95.9217300
1: -385.7852783, 128.8337402, -263.9763794, 96.9063492, -482.6916199, 392.8100586
2: -206.2005920, 121.6190414, -151.9536285, 87.1560898, -293.3566589, 273.5726624
3: -266.0208740, 97.4611511, -185.7883911, 69.7485275, -335.7693787, 283.2495422
4: -147.7929688, 107.7015762, -111.0705872, 76.3432465, -224.1362152, 218.7721252

Time for backsubstitution: 1.05 seconds

### NS candidates at layer 1
type: B, layer: 1, pos: 4
type: B, layer: 1, pos: 3
type: A, layer: 1, pos: 35
type: B, layer: 1, pos: 35
type: A, layer: 1, pos: 7
type: A, layer: 1, pos: 9
type: B, layer: 1, pos: 17
type: A, layer: 1, pos: 3
type: B, layer: 1, pos: 38
type: A, layer: 1, pos: 10
type: A, layer: 1, pos: 15
type: B, layer: 1, pos: 45
type: A, layer: 1, pos: 4
type: B, layer: 1, pos: 18
type: A, layer: 1, pos: 11
type: B, layer: 1, pos: 11
type: A, layer: 1, pos: 43
type: B, layer: 1, pos: 43
type: A, layer: 1, pos: 29
type: A, layer: 1, pos: 17
type: A, layer: 1, pos: 19
type: A, layer: 1, pos: 16
type: B, layer: 1, pos: 10
type: B, layer: 1, pos: 21
type: A, layer: 1, pos: 32
type: A, layer: 1, pos: 18
type: B, layer: 1, pos: 37
type: B, layer: 1, pos: 15
type: A, layer: 1, pos: 21
type: A, layer: 1, pos: 45
type: A, layer: 1, pos: 20
type: A, layer: 1, pos: 23
type: A, layer: 1, pos: 39
type: B, layer: 1, pos: 23
type: B, layer: 1, pos: 32
type: B, layer: 1, pos: 41
type: A, layer: 1, pos: 41
type: B, layer: 1, pos: 5
type: B, layer: 1, pos: 39
type: B, layer: 1, pos: 20
type: A, layer: 1, pos: 5
type: A, layer: 1, pos: 38
type: A, layer: 1, pos: 37
type: B, layer: 1, pos: 16

Time for candidate selection: 0.09 seconds

### Candidate
type: B, layer: 1, pos: 4

### Candidate
type: B, layer: 1, pos: 3

### Candidate
type: A, layer: 1, pos: 35

### Candidate
type: B, layer: 1, pos: 35

### Candidate
type: A, layer: 1, pos: 7

## Relational analysis of NS_B1_B2_A2_B2_B1_B2_A1

### Relational analysis result of NS_B1_B2_A2_B2_B1_B2_A1
Status: Status.UNKNOWN
Output dim: 0, lower bound: -63.3929721, upper bound: 63.3929847
time: 0.68 seconds

## Relational analysis of NS_B1_B2_A2_B2_B1_B2_A2

### Relational analysis result of NS_B1_B2_A2_B2_B1_B2_A2
Status: Status.UNKNOWN
Output dim: 0, lower bound: -63.3929721, upper bound: 63.3933291
time: 0.76 seconds

## BFS NS instance: NS_B1_B2_A2_B2_B2_B1

### Backsubstitution after applying NS history:
0: -52.7145691, 57.3530884, -32.1617851, 34.0683327, -86.7828903, 89.5148621
1: -391.0660706, 130.1454468, -232.7899628, 79.6942673, -470.7603455, 362.9353943
2: -208.5744019, 122.9083939, -128.2648773, 73.4637985, -282.0381775, 251.1732788
3: -269.5091858, 98.5153122, -161.6835480, 58.8755074, -328.3847046, 260.1988220
4: -149.3443909, 108.7455902, -91.9323349, 64.2209549, -213.5653076, 200.6779175

Time for backsubstitution: 1.02 seconds

### NS candidates at layer 1
type: B, layer: 1, pos: 4
type: B, layer: 1, pos: 3
type: A, layer: 1, pos: 35
type: B, layer: 1, pos: 35
type: B, layer: 1, pos: 38
type: A, layer: 1, pos: 7
type: A, layer: 1, pos: 3
type: A, layer: 1, pos: 9
type: B, layer: 1, pos: 17
type: B, layer: 1, pos: 45
type: A, layer: 1, pos: 43
type: A, layer: 1, pos: 15
type: A, layer: 1, pos: 11
type: B, layer: 1, pos: 11
type: A, layer: 1, pos: 18
type: A, layer: 1, pos: 10
type: B, layer: 1, pos: 10
type: A, layer: 1, pos: 45
type: A, layer: 1, pos: 4
type: A, layer: 1, pos: 23
type: B, layer: 1, pos: 18
type: B, layer: 1, pos: 15
type: A, layer: 1, pos: 39
type: B, layer: 1, pos: 43
type: A, layer: 1, pos: 17
type: B, layer: 1, pos: 21
type: A, layer: 1, pos: 19
type: A, layer: 1, pos: 16
type: A, layer: 1, pos: 21
type: B, layer: 1, pos: 37
type: A, layer: 1, pos: 32
type: B, layer: 1, pos: 23
type: A, layer: 1, pos: 20
type: B, layer: 1, pos: 20
type: B, layer: 1, pos: 41
type: B, layer: 1, pos: 32
type: A, layer: 1, pos: 41
type: B, layer: 1, pos: 5
type: A, layer: 1, pos: 5
type: B, layer: 1, pos: 39
type: A, layer: 1, pos: 37
type: B, layer: 1, pos: 16
type: A, layer: 1, pos: 38
type: A, layer: 1, pos: 29

Time for candidate selection: 0.08 seconds

### Candidate
type: B, layer: 1, pos: 4

### Candidate
type: B, layer: 1, pos: 3

### Candidate
type: A, layer: 1, pos: 35

### Candidate
type: B, layer: 1, pos: 35

### Candidate
type: B, layer: 1, pos: 38

## Relational analysis of NS_B1_B2_A2_B2_B2_B1_B1
Optimization infeasible because this subproblem isn't reachable.

### Candidate
type: A, layer: 1, pos: 7

## Relational analysis of NS_B1_B2_A2_B2_B2_B1_A1

### Relational analysis result of NS_B1_B2_A2_B2_B2_B1_A1
Status: Status.UNKNOWN
Output dim: 0, lower bound: -63.3932456, upper bound: 63.3932652
time: 0.72 seconds

## Relational analysis of NS_B1_B2_A2_B2_B2_B1_A2

### Relational analysis result of NS_B1_B2_A2_B2_B2_B1_A2
Status: Status.UNKNOWN
Output dim: 0, lower bound: -63.3932456, upper bound: 63.3941989
time: 0.73 seconds

## BFS NS instance: NS_B1_B2_A2_B2_B2_B2

### Backsubstitution after applying NS history:
0: -52.1011620, 56.6095123, -42.7220459, 44.2243690, -96.3255157, 99.3315582
1: -384.4843140, 128.5108948, -291.8194580, 105.7542801, -490.2385864, 420.3302612
2: -205.6697693, 121.2567215, -167.0193329, 95.5304642, -301.2002258, 288.2760010
3: -265.2196350, 97.1680374, -205.2453308, 76.4457169, -341.6653442, 302.4133606
4: -147.4522400, 107.3706512, -121.4646149, 83.5994034, -231.0516357, 228.8352509

Time for backsubstitution: 1.15 seconds

### NS candidates at layer 1
type: B, layer: 1, pos: 4
type: A, layer: 1, pos: 35
type: B, layer: 1, pos: 3
type: B, layer: 1, pos: 35
type: A, layer: 1, pos: 7
type: A, layer: 1, pos: 9
type: B, layer: 1, pos: 17
type: A, layer: 1, pos: 3
type: B, layer: 1, pos: 38
type: A, layer: 1, pos: 10
type: A, layer: 1, pos: 15
type: B, layer: 1, pos: 18
type: A, layer: 1, pos: 11
type: B, layer: 1, pos: 43
type: B, layer: 1, pos: 11
type: B, layer: 1, pos: 45
type: A, layer: 1, pos: 17
type: A, layer: 1, pos: 29
type: A, layer: 1, pos: 4
type: A, layer: 1, pos: 43
type: B, layer: 1, pos: 10
type: B, layer: 1, pos: 37
type: B, layer: 1, pos: 15
type: A, layer: 1, pos: 19
type: B, layer: 1, pos: 21
type: A, layer: 1, pos: 32
type: A, layer: 1, pos: 18
type: A, layer: 1, pos: 16
type: A, layer: 1, pos: 45
type: A, layer: 1, pos: 21
type: A, layer: 1, pos: 39
type: A, layer: 1, pos: 20
type: A, layer: 1, pos: 23
type: B, layer: 1, pos: 32
type: B, layer: 1, pos: 23
type: B, layer: 1, pos: 41
type: A, layer: 1, pos: 41
type: B, layer: 1, pos: 5
type: B, layer: 1, pos: 39
type: B, layer: 1, pos: 20
type: A, layer: 1, pos: 5
type: A, layer: 1, pos: 37
type: A, layer: 1, pos: 38
type: B, layer: 1, pos: 16

Time for candidate selection: 0.08 seconds

### Candidate
type: B, layer: 1, pos: 4

### Candidate
type: A, layer: 1, pos: 35

### Candidate
type: B, layer: 1, pos: 3

### Candidate
type: B, layer: 1, pos: 35

### Candidate
type: A, layer: 1, pos: 7

## Relational analysis of NS_B1_B2_A2_B2_B2_B2_A1

### Relational analysis result of NS_B1_B2_A2_B2_B2_B2_A1
Status: Status.UNKNOWN
Output dim: 0, lower bound: -63.3931347, upper bound: 63.3930939
time: 0.74 seconds

## Relational analysis of NS_B1_B2_A2_B2_B2_B2_A2

### Relational analysis result of NS_B1_B2_A2_B2_B2_B2_A2
Status: Status.UNKNOWN
Output dim: 0, lower bound: -63.3929721, upper bound: 63.3937700
time: 0.80 seconds

## BFS NS instance: NS_B2_A1_A1_B1_A1_A1

### Backsubstitution after applying NS history:
0: -28.6123600, 29.9811668, -51.8972397, 56.3605003, -84.9728622, 81.8784027
1: -203.3975525, 70.9110107, -383.1071472, 128.0625305, -331.4600525, 452.9517822
2: -113.8226547, 64.7897034, -204.7822723, 120.8237000, -234.6463623, 269.5719604
3: -142.2896118, 51.9520988, -264.2325439, 96.7753143, -239.0649261, 315.7929382
4: -81.7880096, 56.5346222, -147.1542511, 106.8489990, -188.6370087, 203.6888580

Time for backsubstitution: 1.05 seconds

### NS candidates at layer 1
type: A, layer: 1, pos: 4
type: A, layer: 1, pos: 9
type: A, layer: 1, pos: 38
type: B, layer: 1, pos: 45
type: B, layer: 1, pos: 35
type: A, layer: 1, pos: 19
type: A, layer: 1, pos: 17
type: A, layer: 1, pos: 3
type: B, layer: 1, pos: 9
type: B, layer: 1, pos: 11
type: B, layer: 1, pos: 19
type: A, layer: 1, pos: 10
type: A, layer: 1, pos: 11
type: B, layer: 1, pos: 15
type: B, layer: 1, pos: 10
type: A, layer: 1, pos: 35
type: B, layer: 1, pos: 16
type: B, layer: 1, pos: 39
type: A, layer: 1, pos: 37
type: B, layer: 1, pos: 43
type: B, layer: 1, pos: 3
type: B, layer: 1, pos: 17
type: B, layer: 1, pos: 18
type: A, layer: 1, pos: 18
type: A, layer: 1, pos: 32
type: B, layer: 1, pos: 23
type: A, layer: 1, pos: 21
type: B, layer: 1, pos: 32
type: A, layer: 1, pos: 41
type: A, layer: 1, pos: 15
type: B, layer: 1, pos: 41
type: B, layer: 1, pos: 20
type: A, layer: 1, pos: 5
type: B, layer: 1, pos: 5
type: A, layer: 1, pos: 20
type: A, layer: 1, pos: 39
type: B, layer: 1, pos: 21
type: A, layer: 1, pos: 43
type: A, layer: 1, pos: 23
type: A, layer: 1, pos: 16
type: B, layer: 1, pos: 37
type: B, layer: 1, pos: 38
type: B, layer: 1, pos: 4
type: B, layer: 1, pos: 29

Time for candidate selection: 0.08 seconds

### Candidate
type: A, layer: 1, pos: 4

### Candidate
type: A, layer: 1, pos: 9

### Candidate
type: A, layer: 1, pos: 38

### Candidate
type: B, layer: 1, pos: 45

## Relational analysis of NS_B2_A1_A1_B1_A1_A1_B1
Optimization infeasible because this subproblem isn't reachable.

### Candidate
type: B, layer: 1, pos: 35

### Candidate
type: A, layer: 1, pos: 19

### Candidate
type: A, layer: 1, pos: 17

### Candidate
type: A, layer: 1, pos: 3

### Candidate
type: B, layer: 1, pos: 9

### Candidate
type: B, layer: 1, pos: 11

## Relational analysis of NS_B2_A1_A1_B1_A1_A1_B1

### Relational analysis result of NS_B2_A1_A1_B1_A1_A1_B1
Status: Status.UNKNOWN
Output dim: 0, lower bound: -63.3937038, upper bound: 63.3940937
time: 0.77 seconds

## Relational analysis of NS_B2_A1_A1_B1_A1_A1_B2

### Relational analysis result of NS_B2_A1_A1_B1_A1_A1_B2
Status: Status.UNKNOWN
Output dim: 0, lower bound: -63.3937808, upper bound: 63.3942339
time: 0.75 seconds

## BFS NS instance: NS_B2_A1_A1_B1_A1_A2

### Backsubstitution after applying NS history:
0: -32.1717300, 33.8845825, -49.0315132, 53.2506561, -85.4223862, 82.9160843
1: -232.4073181, 79.8427124, -362.6061096, 121.0304184, -353.4377441, 441.8114624
2: -128.4413452, 73.6523666, -193.8548584, 114.2220535, -242.6633911, 267.5072327
3: -161.6919708, 58.9059448, -250.1602631, 91.4705429, -253.1625061, 308.7703247
4: -92.0508575, 64.0470734, -139.3379364, 101.0630875, -193.1139221, 203.3850098

Time for backsubstitution: 1.05 seconds

### NS candidates at layer 1
type: A, layer: 1, pos: 4
type: B, layer: 1, pos: 45
type: A, layer: 1, pos: 9
type: B, layer: 1, pos: 35
type: A, layer: 1, pos: 17
type: A, layer: 1, pos: 38
type: A, layer: 1, pos: 19
type: A, layer: 1, pos: 3
type: B, layer: 1, pos: 9
type: B, layer: 1, pos: 19
type: B, layer: 1, pos: 11
type: A, layer: 1, pos: 35
type: A, layer: 1, pos: 10
type: A, layer: 1, pos: 11
type: B, layer: 1, pos: 15
type: B, layer: 1, pos: 10
type: B, layer: 1, pos: 3
type: B, layer: 1, pos: 17
type: B, layer: 1, pos: 39
type: A, layer: 1, pos: 32
type: B, layer: 1, pos: 16
type: A, layer: 1, pos: 37
type: A, layer: 1, pos: 18
type: B, layer: 1, pos: 43
type: B, layer: 1, pos: 18
type: A, layer: 1, pos: 21
type: A, layer: 1, pos: 15
type: B, layer: 1, pos: 23
type: A, layer: 1, pos: 41
type: B, layer: 1, pos: 32
type: B, layer: 1, pos: 20
type: A, layer: 1, pos: 5
type: A, layer: 1, pos: 39
type: B, layer: 1, pos: 41
type: A, layer: 1, pos: 20
type: B, layer: 1, pos: 5
type: B, layer: 1, pos: 21
type: A, layer: 1, pos: 43
type: A, layer: 1, pos: 23
type: B, layer: 1, pos: 37
type: A, layer: 1, pos: 16
type: B, layer: 1, pos: 4
type: B, layer: 1, pos: 38
type: B, layer: 1, pos: 29

Time for candidate selection: 0.09 seconds

### Candidate
type: A, layer: 1, pos: 4

### Candidate
type: B, layer: 1, pos: 45

## Relational analysis of NS_B2_A1_A1_B1_A1_A2_B1
Optimization infeasible because this subproblem isn't reachable.

### Candidate
type: A, layer: 1, pos: 9

### Candidate
type: B, layer: 1, pos: 35

### Candidate
type: A, layer: 1, pos: 17

### Candidate
type: A, layer: 1, pos: 38

### Candidate
type: A, layer: 1, pos: 19

### Candidate
type: A, layer: 1, pos: 3

### Candidate
type: B, layer: 1, pos: 9

### Candidate
type: B, layer: 1, pos: 19

### Candidate
type: B, layer: 1, pos: 11

## Relational analysis of NS_B2_A1_A1_B1_A1_A2_B1

### Relational analysis result of NS_B2_A1_A1_B1_A1_A2_B1
Status: Status.UNKNOWN
Output dim: 0, lower bound: -63.3936915, upper bound: 63.3940600
time: 0.72 seconds

## Relational analysis of NS_B2_A1_A1_B1_A1_A2_B2

### Relational analysis result of NS_B2_A1_A1_B1_A1_A2_B2
Status: Status.UNKNOWN
Output dim: 0, lower bound: -63.3937689, upper bound: 63.3942007
time: 1.13 seconds

## BFS NS instance: NS_B2_A1_A1_B1_A2_A1

### Backsubstitution after applying NS history:
0: -39.1999359, 40.2295341, -51.4681511, 55.8039970, -94.8895721, 91.6976852
1: -262.8308716, 96.9850388, -377.9686584, 126.9298859, -389.7607117, 472.9980469
2: -151.7921295, 87.1059418, -202.6948242, 119.5828781, -270.5632019, 289.8006592
3: -185.1978149, 69.6738281, -260.8946838, 95.8214035, -280.8345947, 330.1843262
4: -111.0881882, 76.2246246, -145.8291473, 105.8247452, -215.8088226, 222.0537720

Time for backsubstitution: 1.05 seconds

### NS candidates at layer 1
type: A, layer: 1, pos: 4
type: A, layer: 1, pos: 9
type: B, layer: 1, pos: 45
type: A, layer: 1, pos: 19
type: B, layer: 1, pos: 35
type: A, layer: 1, pos: 3
type: A, layer: 1, pos: 17
type: B, layer: 1, pos: 9
type: A, layer: 1, pos: 35
type: B, layer: 1, pos: 10
type: A, layer: 1, pos: 38
type: B, layer: 1, pos: 11
type: B, layer: 1, pos: 15
type: A, layer: 1, pos: 11
type: B, layer: 1, pos: 19
type: A, layer: 1, pos: 18
type: B, layer: 1, pos: 17
type: B, layer: 1, pos: 29
type: B, layer: 1, pos: 3
type: B, layer: 1, pos: 4
type: A, layer: 1, pos: 10
type: A, layer: 1, pos: 37
type: B, layer: 1, pos: 16
type: B, layer: 1, pos: 32
type: B, layer: 1, pos: 39
type: A, layer: 1, pos: 21
type: B, layer: 1, pos: 18
type: B, layer: 1, pos: 20
type: A, layer: 1, pos: 43
type: A, layer: 1, pos: 15
type: A, layer: 1, pos: 32
type: A, layer: 1, pos: 39
type: A, layer: 1, pos: 41
type: B, layer: 1, pos: 43
type: B, layer: 1, pos: 23
type: B, layer: 1, pos: 41
type: A, layer: 1, pos: 5
type: B, layer: 1, pos: 5
type: A, layer: 1, pos: 23
type: B, layer: 1, pos: 21
type: A, layer: 1, pos: 20
type: B, layer: 1, pos: 37
type: A, layer: 1, pos: 16
type: B, layer: 1, pos: 38

Time for candidate selection: 0.08 seconds

### Candidate
type: A, layer: 1, pos: 4

### Candidate
type: A, layer: 1, pos: 9

### Candidate
type: B, layer: 1, pos: 45

## Relational analysis of NS_B2_A1_A1_B1_A2_A1_B1
Optimization infeasible because this subproblem isn't reachable.

### Candidate
type: A, layer: 1, pos: 19

### Candidate
type: B, layer: 1, pos: 35

### Candidate
type: A, layer: 1, pos: 3

### Candidate
type: A, layer: 1, pos: 17

### Candidate
type: B, layer: 1, pos: 9

### Candidate
type: A, layer: 1, pos: 35

### Candidate
type: B, layer: 1, pos: 10

## Relational analysis of NS_B2_A1_A1_B1_A2_A1_B1
Optimization infeasible because this subproblem isn't reachable.

### Candidate
type: A, layer: 1, pos: 38

### Candidate
type: B, layer: 1, pos: 11

## Relational analysis of NS_B2_A1_A1_B1_A2_A1_B1

### Relational analysis result of NS_B2_A1_A1_B1_A2_A1_B1
Status: Status.UNKNOWN
Output dim: 0, lower bound: -63.3931915, upper bound: 63.3935011
time: 0.71 seconds

## Relational analysis of NS_B2_A1_A1_B1_A2_A1_B2

### Relational analysis result of NS_B2_A1_A1_B1_A2_A1_B2
Status: Status.UNKNOWN
Output dim: 0, lower bound: -63.3932008, upper bound: 63.3935008
time: 0.63 seconds

## BFS NS instance: NS_B2_A1_A1_B1_A2_A2

### Backsubstitution after applying NS history:
0: -40.2247047, 41.6142464, -48.5056076, 52.5619240, -92.6558838, 90.1198578
1: -276.9466553, 99.8336334, -356.4946899, 119.6866150, -396.6332703, 454.6684570
2: -157.8598022, 90.5087128, -191.5168610, 112.6963272, -269.4329834, 282.0255432
3: -194.3886261, 72.2380142, -246.1923828, 90.2981110, -284.0122986, 318.2066956
4: -114.4939880, 79.0424728, -137.7221985, 99.7798843, -213.1783447, 216.7646790

Time for backsubstitution: 1.08 seconds

### NS candidates at layer 1
type: A, layer: 1, pos: 4
type: B, layer: 1, pos: 45
type: A, layer: 1, pos: 9
type: B, layer: 1, pos: 9
type: B, layer: 1, pos: 35
type: A, layer: 1, pos: 19
type: A, layer: 1, pos: 17
type: A, layer: 1, pos: 35
type: B, layer: 1, pos: 10
type: A, layer: 1, pos: 3
type: B, layer: 1, pos: 19
type: B, layer: 1, pos: 3
type: B, layer: 1, pos: 11
type: B, layer: 1, pos: 17
type: A, layer: 1, pos: 11
type: A, layer: 1, pos: 18
type: A, layer: 1, pos: 38
type: B, layer: 1, pos: 4
type: B, layer: 1, pos: 29
type: B, layer: 1, pos: 15
type: A, layer: 1, pos: 10
type: B, layer: 1, pos: 16
type: A, layer: 1, pos: 37
type: A, layer: 1, pos: 43
type: B, layer: 1, pos: 32
type: B, layer: 1, pos: 39
type: A, layer: 1, pos: 15
type: B, layer: 1, pos: 20
type: A, layer: 1, pos: 32
type: A, layer: 1, pos: 21
type: A, layer: 1, pos: 39
type: A, layer: 1, pos: 41
type: B, layer: 1, pos: 18
type: B, layer: 1, pos: 41
type: B, layer: 1, pos: 43
type: A, layer: 1, pos: 23
type: A, layer: 1, pos: 5
type: B, layer: 1, pos: 5
type: B, layer: 1, pos: 23
type: B, layer: 1, pos: 37
type: A, layer: 1, pos: 20
type: B, layer: 1, pos: 21
type: A, layer: 1, pos: 16
type: B, layer: 1, pos: 38

Time for candidate selection: 0.09 seconds

### Candidate
type: A, layer: 1, pos: 4

### Candidate
type: B, layer: 1, pos: 45

## Relational analysis of NS_B2_A1_A1_B1_A2_A2_B1
Optimization infeasible because this subproblem isn't reachable.

### Candidate
type: A, layer: 1, pos: 9

### Candidate
type: B, layer: 1, pos: 9

### Candidate
type: B, layer: 1, pos: 35

### Candidate
type: A, layer: 1, pos: 19

### Candidate
type: A, layer: 1, pos: 17

### Candidate
type: A, layer: 1, pos: 35

### Candidate
type: B, layer: 1, pos: 10

## Relational analysis of NS_B2_A1_A1_B1_A2_A2_B1
Optimization infeasible because this subproblem isn't reachable.

### Candidate
type: A, layer: 1, pos: 3

### Candidate
type: B, layer: 1, pos: 19

### Candidate
type: B, layer: 1, pos: 3

### Candidate
type: B, layer: 1, pos: 11

## Relational analysis of NS_B2_A1_A1_B1_A2_A2_B1

### Relational analysis result of NS_B2_A1_A1_B1_A2_A2_B1
Status: Status.UNKNOWN
Output dim: 0, lower bound: -63.3936381, upper bound: 63.3938858
time: 0.67 seconds

## Relational analysis of NS_B2_A1_A1_B1_A2_A2_B2

### Relational analysis result of NS_B2_A1_A1_B1_A2_A2_B2
Status: Status.UNKNOWN
Output dim: 0, lower bound: -63.3937065, upper bound: 63.3940240
time: 0.74 seconds

## BFS NS instance: NS_B2_A1_A1_B2_A1_A1

### Backsubstitution after applying NS history:
0: -28.6123600, 29.9811668, -52.2746696, 56.9984779, -85.6108398, 82.2558365
1: -203.3975525, 70.9110107, -389.0621033, 129.1828613, -332.5804138, 459.3416443
2: -113.8226547, 64.7897034, -207.0110779, 122.1272202, -235.9498749, 271.8007507
3: -142.2896118, 51.9520988, -267.8845215, 97.9457474, -240.2353516, 319.7839661
4: -81.7880096, 56.5346222, -148.0931091, 108.0790558, -189.8670654, 204.6277008

Time for backsubstitution: 1.30 seconds

### NS candidates at layer 1
type: A, layer: 1, pos: 4
type: A, layer: 1, pos: 3
type: B, layer: 1, pos: 35
type: A, layer: 1, pos: 38
type: B, layer: 1, pos: 3
type: A, layer: 1, pos: 9
type: A, layer: 1, pos: 19
type: A, layer: 1, pos: 35
type: B, layer: 1, pos: 9
type: B, layer: 1, pos: 45
type: A, layer: 1, pos: 17
type: B, layer: 1, pos: 43
type: B, layer: 1, pos: 19
type: B, layer: 1, pos: 15
type: B, layer: 1, pos: 4
type: B, layer: 1, pos: 11
type: A, layer: 1, pos: 10
type: A, layer: 1, pos: 11
type: B, layer: 1, pos: 10
type: A, layer: 1, pos: 43
type: A, layer: 1, pos: 18
type: B, layer: 1, pos: 23
type: B, layer: 1, pos: 39
type: A, layer: 1, pos: 37
type: B, layer: 1, pos: 18
type: B, layer: 1, pos: 16
type: A, layer: 1, pos: 21
type: B, layer: 1, pos: 32
type: B, layer: 1, pos: 17
type: B, layer: 1, pos: 21
type: A, layer: 1, pos: 23
type: A, layer: 1, pos: 32
type: A, layer: 1, pos: 15
type: B, layer: 1, pos: 20
type: A, layer: 1, pos: 20
type: A, layer: 1, pos: 41
type: B, layer: 1, pos: 41
type: A, layer: 1, pos: 5
type: B, layer: 1, pos: 5
type: A, layer: 1, pos: 39
type: A, layer: 1, pos: 16
type: B, layer: 1, pos: 38
type: B, layer: 1, pos: 37
type: B, layer: 1, pos: 29

Time for candidate selection: 0.10 seconds

### Candidate
type: A, layer: 1, pos: 4

### Candidate
type: A, layer: 1, pos: 3

### Candidate
type: B, layer: 1, pos: 35

### Candidate
type: A, layer: 1, pos: 38

### Candidate
type: B, layer: 1, pos: 3

### Candidate
type: A, layer: 1, pos: 9

### Candidate
type: A, layer: 1, pos: 19

### Candidate
type: A, layer: 1, pos: 35

### Candidate
type: B, layer: 1, pos: 9

### Candidate
type: B, layer: 1, pos: 45

## Relational analysis of NS_B2_A1_A1_B2_A1_A1_B1
Optimization infeasible because this subproblem isn't reachable.

### Candidate
type: A, layer: 1, pos: 17

### Candidate
type: B, layer: 1, pos: 43

### Candidate
type: B, layer: 1, pos: 19

### Candidate
type: B, layer: 1, pos: 15

### Candidate
type: B, layer: 1, pos: 4

### Candidate
type: B, layer: 1, pos: 11

## Relational analysis of NS_B2_A1_A1_B2_A1_A1_B1

### Relational analysis result of NS_B2_A1_A1_B2_A1_A1_B1
Status: Status.UNKNOWN
Output dim: 0, lower bound: -63.3938091, upper bound: 63.3941592
time: 0.75 seconds

## Relational analysis of NS_B2_A1_A1_B2_A1_A1_B2

### Relational analysis result of NS_B2_A1_A1_B2_A1_A1_B2
Status: Status.UNKNOWN
Output dim: 0, lower bound: -63.3938763, upper bound: 63.3942808
time: 0.79 seconds

## BFS NS instance: NS_B2_A1_A1_B2_A1_A2

### Backsubstitution after applying NS history:
0: -32.1717300, 33.8845825, -49.1084404, 53.5609856, -85.7327118, 82.9930038
1: -232.4073181, 79.8427124, -365.8971252, 121.2925720, -353.6998596, 445.5918274
2: -128.4413452, 73.6523666, -194.7173767, 114.7962875, -243.2376404, 268.3697510
3: -161.6919708, 58.9059448, -252.0432892, 91.9607239, -253.6526947, 310.9492188
4: -92.0508575, 64.0470734, -139.2741394, 101.6966629, -193.7475281, 203.3212128

Time for backsubstitution: 1.05 seconds

### NS candidates at layer 1
type: A, layer: 1, pos: 4
type: A, layer: 1, pos: 3
type: B, layer: 1, pos: 35
type: A, layer: 1, pos: 9
type: B, layer: 1, pos: 3
type: A, layer: 1, pos: 38
type: A, layer: 1, pos: 19
type: A, layer: 1, pos: 35
type: B, layer: 1, pos: 45
type: B, layer: 1, pos: 9
type: A, layer: 1, pos: 17
type: B, layer: 1, pos: 19
type: B, layer: 1, pos: 43
type: B, layer: 1, pos: 15
type: B, layer: 1, pos: 4
type: B, layer: 1, pos: 11
type: A, layer: 1, pos: 10
type: A, layer: 1, pos: 11
type: A, layer: 1, pos: 18
type: B, layer: 1, pos: 10
type: A, layer: 1, pos: 43
type: B, layer: 1, pos: 18
type: B, layer: 1, pos: 23
type: A, layer: 1, pos: 21
type: B, layer: 1, pos: 17
type: A, layer: 1, pos: 37
type: B, layer: 1, pos: 39
type: B, layer: 1, pos: 16
type: B, layer: 1, pos: 32
type: B, layer: 1, pos: 21
type: A, layer: 1, pos: 32
type: A, layer: 1, pos: 15
type: A, layer: 1, pos: 23
type: B, layer: 1, pos: 20
type: A, layer: 1, pos: 41
type: A, layer: 1, pos: 20
type: A, layer: 1, pos: 5
type: B, layer: 1, pos: 41
type: A, layer: 1, pos: 39
type: B, layer: 1, pos: 5
type: A, layer: 1, pos: 16
type: B, layer: 1, pos: 37
type: B, layer: 1, pos: 38
type: B, layer: 1, pos: 29

Time for candidate selection: 0.08 seconds

### Candidate
type: A, layer: 1, pos: 4

### Candidate
type: A, layer: 1, pos: 3

### Candidate
type: B, layer: 1, pos: 35

### Candidate
type: A, layer: 1, pos: 9

### Candidate
type: B, layer: 1, pos: 3

### Candidate
type: A, layer: 1, pos: 38

### Candidate
type: A, layer: 1, pos: 19

### Candidate
type: A, layer: 1, pos: 35

### Candidate
type: B, layer: 1, pos: 45

## Relational analysis of NS_B2_A1_A1_B2_A1_A2_B1
Optimization infeasible because this subproblem isn't reachable.

### Candidate
type: B, layer: 1, pos: 9

### Candidate
type: A, layer: 1, pos: 17

### Candidate
type: B, layer: 1, pos: 19

### Candidate
type: B, layer: 1, pos: 43

### Candidate
type: B, layer: 1, pos: 15

### Candidate
type: B, layer: 1, pos: 4

### Candidate
type: B, layer: 1, pos: 11

## Relational analysis of NS_B2_A1_A1_B2_A1_A2_B1

### Relational analysis result of NS_B2_A1_A1_B2_A1_A2_B1
Status: Status.UNKNOWN
Output dim: 0, lower bound: -63.3938102, upper bound: 63.3941281
time: 0.74 seconds

## Relational analysis of NS_B2_A1_A1_B2_A1_A2_B2

### Relational analysis result of NS_B2_A1_A1_B2_A1_A2_B2
Status: Status.UNKNOWN
Output dim: 0, lower bound: -63.3939816, upper bound: 63.3942695
time: 0.73 seconds

## BFS NS instance: NS_B2_A1_A1_B2_A2_B1

### Backsubstitution after applying NS history:
0: -40.6671410, 41.8246460, -52.5613327, 57.1557770, -97.7474670, 94.3859787
1: -274.0611572, 100.6335068, -388.3494568, 129.7206573, -403.7817993, 487.4248657
2: -157.7368164, 90.5137711, -207.6318054, 122.4168320, -279.3742065, 298.1455688
3: -192.9522858, 72.4099655, -267.8355713, 98.1381149, -290.8802795, 340.1970825
4: -115.2987747, 79.2247696, -148.7674866, 108.4405518, -222.7550201, 227.9922485

Time for backsubstitution: 1.04 seconds

### NS candidates at layer 1
type: A, layer: 1, pos: 4
type: A, layer: 1, pos: 3
type: B, layer: 1, pos: 35
type: A, layer: 1, pos: 19
type: A, layer: 1, pos: 9
type: A, layer: 1, pos: 35
type: B, layer: 1, pos: 9
type: A, layer: 1, pos: 38
type: A, layer: 1, pos: 17
type: B, layer: 1, pos: 3
type: B, layer: 1, pos: 15
type: B, layer: 1, pos: 19
type: B, layer: 1, pos: 10
type: A, layer: 1, pos: 45
type: A, layer: 1, pos: 18
type: B, layer: 1, pos: 4
type: A, layer: 1, pos: 43
type: B, layer: 1, pos: 43
type: B, layer: 1, pos: 17
type: B, layer: 1, pos: 29
type: B, layer: 1, pos: 32
type: A, layer: 1, pos: 11
type: A, layer: 1, pos: 37
type: A, layer: 1, pos: 10
type: B, layer: 1, pos: 16
type: A, layer: 1, pos: 21
type: B, layer: 1, pos: 18
type: B, layer: 1, pos: 21
type: B, layer: 1, pos: 45
type: A, layer: 1, pos: 15
type: B, layer: 1, pos: 20
type: B, layer: 1, pos: 23
type: B, layer: 1, pos: 39
type: A, layer: 1, pos: 23
type: A, layer: 1, pos: 41
type: A, layer: 1, pos: 32
type: B, layer: 1, pos: 41
type: A, layer: 1, pos: 5
type: A, layer: 1, pos: 39
type: A, layer: 1, pos: 20
type: B, layer: 1, pos: 5
type: B, layer: 1, pos: 38
type: B, layer: 1, pos: 37
type: A, layer: 1, pos: 16

Time for candidate selection: 0.08 seconds

### Candidate
type: A, layer: 1, pos: 4

### Candidate
type: A, layer: 1, pos: 3

### Candidate
type: B, layer: 1, pos: 35

### Candidate
type: A, layer: 1, pos: 19

### Candidate
type: A, layer: 1, pos: 9

### Candidate
type: A, layer: 1, pos: 35

### Candidate
type: B, layer: 1, pos: 9

### Candidate
type: A, layer: 1, pos: 38

### Candidate
type: A, layer: 1, pos: 17

### Candidate
type: B, layer: 1, pos: 3

### Candidate
type: B, layer: 1, pos: 15

### Candidate
type: B, layer: 1, pos: 19

### Candidate
type: B, layer: 1, pos: 10

### Candidate
type: A, layer: 1, pos: 45

### Candidate
type: A, layer: 1, pos: 18

### Candidate
type: B, layer: 1, pos: 4

### Candidate
type: A, layer: 1, pos: 43

## Relational analysis of NS_B2_A1_A1_B2_A2_B1_A1
Optimization infeasible because this subproblem isn't reachable.

### Candidate
type: B, layer: 1, pos: 43

### Candidate
type: B, layer: 1, pos: 17

### Candidate
type: B, layer: 1, pos: 29

## Relational analysis of NS_B2_A1_A1_B2_A2_B1_B1
Optimization infeasible because this subproblem isn't reachable.

### Candidate
type: B, layer: 1, pos: 32

## Relational analysis of NS_B2_A1_A1_B2_A2_B1_B1
Optimization infeasible because this subproblem isn't reachable.

### Candidate
type: A, layer: 1, pos: 11

## Relational analysis of NS_B2_A1_A1_B2_A2_B1_A1
Optimization infeasible because this subproblem isn't reachable.

### Candidate
type: A, layer: 1, pos: 37

## Relational analysis of NS_B2_A1_A1_B2_A2_B1_A1
Optimization infeasible because this subproblem isn't reachable.

### Candidate
type: A, layer: 1, pos: 10

### Candidate
type: B, layer: 1, pos: 16

## Relational analysis of NS_B2_A1_A1_B2_A2_B1_B1
Optimization infeasible because this subproblem isn't reachable.

### Candidate
type: A, layer: 1, pos: 21

## Relational analysis of NS_B2_A1_A1_B2_A2_B1_A1
Optimization infeasible because this subproblem isn't reachable.

### Candidate
type: B, layer: 1, pos: 18

### Candidate
type: B, layer: 1, pos: 21

### Candidate
type: B, layer: 1, pos: 45

### Candidate
type: A, layer: 1, pos: 15

### Candidate
type: B, layer: 1, pos: 20

## Relational analysis of NS_B2_A1_A1_B2_A2_B1_B1
Optimization infeasible because this subproblem isn't reachable.

### Candidate
type: B, layer: 1, pos: 23

## Relational analysis of NS_B2_A1_A1_B2_A2_B1_B1
Optimization infeasible because this subproblem isn't reachable.

### Candidate
type: B, layer: 1, pos: 39

## Relational analysis of NS_B2_A1_A1_B2_A2_B1_B1
Optimization infeasible because this subproblem isn't reachable.

### Candidate
type: A, layer: 1, pos: 23

### Candidate
type: A, layer: 1, pos: 41

## Relational analysis of NS_B2_A1_A1_B2_A2_B1_A1
Optimization infeasible because this subproblem isn't reachable.

### Candidate
type: A, layer: 1, pos: 32

### Candidate
type: B, layer: 1, pos: 41

### Candidate
type: A, layer: 1, pos: 5

## Relational analysis of NS_B2_A1_A1_B2_A2_B1_A1
Optimization infeasible because this subproblem isn't reachable.

### Candidate
type: A, layer: 1, pos: 39

### Candidate
type: A, layer: 1, pos: 20

### Candidate
type: B, layer: 1, pos: 5

### Candidate
type: B, layer: 1, pos: 38

### Candidate
type: B, layer: 1, pos: 37

### Candidate
type: A, layer: 1, pos: 16

## BFS NS instance: NS_B2_A1_A1_B2_A2_B2

### Backsubstitution after applying NS history:
0: -40.8103180, 41.9722328, -52.4143295, 56.9992371, -97.7481003, 94.3865585
1: -275.1538086, 100.9877777, -387.3854065, 129.3546753, -404.5084534, 486.9083862
2: -158.3198853, 90.8393936, -207.0046539, 122.0845871, -279.6778870, 297.8440247
3: -193.7020264, 72.6721802, -267.1123047, 97.8885498, -291.4414368, 339.7837524
4: -115.7052841, 79.5005569, -148.2979126, 108.1422577, -222.8901062, 227.7984161

Time for backsubstitution: 1.11 seconds

### NS candidates at layer 1
type: A, layer: 1, pos: 4
type: A, layer: 1, pos: 3
type: B, layer: 1, pos: 35
type: A, layer: 1, pos: 19
type: A, layer: 1, pos: 9
type: A, layer: 1, pos: 35
type: B, layer: 1, pos: 9
type: A, layer: 1, pos: 38
type: A, layer: 1, pos: 17
type: B, layer: 1, pos: 3
type: B, layer: 1, pos: 15
type: B, layer: 1, pos: 10
type: B, layer: 1, pos: 19
type: A, layer: 1, pos: 45
type: A, layer: 1, pos: 18
type: B, layer: 1, pos: 4
type: A, layer: 1, pos: 43
type: B, layer: 1, pos: 17
type: B, layer: 1, pos: 29
type: A, layer: 1, pos: 11
type: B, layer: 1, pos: 32
type: B, layer: 1, pos: 43
type: A, layer: 1, pos: 37
type: B, layer: 1, pos: 16
type: A, layer: 1, pos: 10
type: A, layer: 1, pos: 21
type: B, layer: 1, pos: 18
type: B, layer: 1, pos: 21
type: A, layer: 1, pos: 15
type: B, layer: 1, pos: 20
type: B, layer: 1, pos: 23
type: B, layer: 1, pos: 45
type: B, layer: 1, pos: 39
type: A, layer: 1, pos: 23
type: A, layer: 1, pos: 41
type: A, layer: 1, pos: 32
type: B, layer: 1, pos: 41
type: A, layer: 1, pos: 5
type: A, layer: 1, pos: 39
type: A, layer: 1, pos: 20
type: B, layer: 1, pos: 5
type: B, layer: 1, pos: 37
type: B, layer: 1, pos: 38
type: A, layer: 1, pos: 16

Time for candidate selection: 0.09 seconds

### Candidate
type: A, layer: 1, pos: 4

### Candidate
type: A, layer: 1, pos: 3

### Candidate
type: B, layer: 1, pos: 35

### Candidate
type: A, layer: 1, pos: 19

### Candidate
type: A, layer: 1, pos: 9

### Candidate
type: A, layer: 1, pos: 35

### Candidate
type: B, layer: 1, pos: 9

### Candidate
type: A, layer: 1, pos: 38

### Candidate
type: A, layer: 1, pos: 17

### Candidate
type: B, layer: 1, pos: 3

### Candidate
type: B, layer: 1, pos: 15

### Candidate
type: B, layer: 1, pos: 10

### Candidate
type: B, layer: 1, pos: 19

### Candidate
type: A, layer: 1, pos: 45

### Candidate
type: A, layer: 1, pos: 18

### Candidate
type: B, layer: 1, pos: 4

### Candidate
type: A, layer: 1, pos: 43

## Relational analysis of NS_B2_A1_A1_B2_A2_B2_A1
Optimization infeasible because this subproblem isn't reachable.

### Candidate
type: B, layer: 1, pos: 17

### Candidate
type: B, layer: 1, pos: 29

## Relational analysis of NS_B2_A1_A1_B2_A2_B2_B1
Optimization infeasible because this subproblem isn't reachable.

### Candidate
type: A, layer: 1, pos: 11

## Relational analysis of NS_B2_A1_A1_B2_A2_B2_A1

### Relational analysis result of NS_B2_A1_A1_B2_A2_B2_A1
Status: Status.UNKNOWN
Output dim: 0, lower bound: -63.3937747, upper bound: 63.3937608
time: 0.75 seconds

## Relational analysis of NS_B2_A1_A1_B2_A2_B2_A2

### Relational analysis result of NS_B2_A1_A1_B2_A2_B2_A2
Status: Status.UNKNOWN
Output dim: 0, lower bound: -63.3937747, upper bound: 63.3942476
time: 0.76 seconds

## BFS NS instance: NS_B2_A1_A2_B1_A1_B1

### Backsubstitution after applying NS history:
0: -53.9737701, 58.4838371, -53.9752045, 58.4854774, -112.4592438, 112.4590378
1: -396.7337952, 133.1474609, -396.7471008, 133.1512756, -528.7742310, 528.7833862
2: -212.7236023, 125.4101639, -212.7303009, 125.4136429, -338.1372375, 338.1404419
3: -273.8752441, 100.4784775, -273.8845825, 100.4812927, -373.3616333, 373.3680420
4: -152.9991302, 110.9224167, -153.0037079, 110.9251785, -263.9243164, 263.9260864

Time for backsubstitution: 1.08 seconds

### NS candidates at layer 1
type: B, layer: 1, pos: 4
type: A, layer: 1, pos: 4
type: B, layer: 1, pos: 9
type: A, layer: 1, pos: 9
type: A, layer: 1, pos: 35
type: B, layer: 1, pos: 35
type: A, layer: 1, pos: 17
type: B, layer: 1, pos: 17
type: B, layer: 1, pos: 45
type: A, layer: 1, pos: 45
type: B, layer: 1, pos: 19
type: A, layer: 1, pos: 19
type: B, layer: 1, pos: 3
type: A, layer: 1, pos: 3
type: A, layer: 1, pos: 10
type: B, layer: 1, pos: 10
type: B, layer: 1, pos: 18
type: A, layer: 1, pos: 18
type: A, layer: 1, pos: 15
type: B, layer: 1, pos: 15
type: B, layer: 1, pos: 39
type: A, layer: 1, pos: 39
type: B, layer: 1, pos: 43
type: B, layer: 1, pos: 37
type: A, layer: 1, pos: 43
type: B, layer: 1, pos: 38
type: A, layer: 1, pos: 37
type: A, layer: 1, pos: 32
type: A, layer: 1, pos: 38
type: B, layer: 1, pos: 32
type: A, layer: 1, pos: 41
type: B, layer: 1, pos: 41
type: B, layer: 1, pos: 29
type: A, layer: 1, pos: 29
type: B, layer: 1, pos: 20
type: A, layer: 1, pos: 20
type: A, layer: 1, pos: 5
type: B, layer: 1, pos: 5
type: A, layer: 1, pos: 16
type: B, layer: 1, pos: 16
type: A, layer: 1, pos: 21
type: B, layer: 1, pos: 21
type: A, layer: 1, pos: 23
type: B, layer: 1, pos: 23

Time for candidate selection: 0.09 seconds

### Candidate
type: B, layer: 1, pos: 4

### Candidate
type: A, layer: 1, pos: 4

### Candidate
type: B, layer: 1, pos: 9

### Candidate
type: A, layer: 1, pos: 9

### Candidate
type: A, layer: 1, pos: 35

### Candidate
type: B, layer: 1, pos: 35

### Candidate
type: A, layer: 1, pos: 17

### Candidate
type: B, layer: 1, pos: 17

### Candidate
type: B, layer: 1, pos: 45

### Candidate
type: A, layer: 1, pos: 45

### Candidate
type: B, layer: 1, pos: 19

### Candidate
type: A, layer: 1, pos: 19

### Candidate
type: B, layer: 1, pos: 3

### Candidate
type: A, layer: 1, pos: 3

### Candidate
type: A, layer: 1, pos: 10

### Candidate
type: B, layer: 1, pos: 10

### Candidate
type: B, layer: 1, pos: 18

## Relational analysis of NS_B2_A1_A2_B1_A1_B1_B1
Optimization infeasible because this subproblem isn't reachable.

### Candidate
type: A, layer: 1, pos: 18

### Candidate
type: A, layer: 1, pos: 15

## Relational analysis of NS_B2_A1_A2_B1_A1_B1_A1
Optimization infeasible because this subproblem isn't reachable.

### Candidate
type: B, layer: 1, pos: 15

### Candidate
type: B, layer: 1, pos: 39

## Relational analysis of NS_B2_A1_A2_B1_A1_B1_B1
Optimization infeasible because this subproblem isn't reachable.

### Candidate
type: A, layer: 1, pos: 39

### Candidate
type: B, layer: 1, pos: 43

## Relational analysis of NS_B2_A1_A2_B1_A1_B1_B1
Optimization infeasible because this subproblem isn't reachable.

### Candidate
type: B, layer: 1, pos: 37

## Relational analysis of NS_B2_A1_A2_B1_A1_B1_B1
Optimization infeasible because this subproblem isn't reachable.

### Candidate
type: A, layer: 1, pos: 43

### Candidate
type: B, layer: 1, pos: 38

## Relational analysis of NS_B2_A1_A2_B1_A1_B1_B1
Optimization infeasible because this subproblem isn't reachable.

### Candidate
type: A, layer: 1, pos: 37

### Candidate
type: A, layer: 1, pos: 32

## Relational analysis of NS_B2_A1_A2_B1_A1_B1_A1
Optimization infeasible because this subproblem isn't reachable.

### Candidate
type: A, layer: 1, pos: 38

### Candidate
type: B, layer: 1, pos: 32

### Candidate
type: A, layer: 1, pos: 41

## Relational analysis of NS_B2_A1_A2_B1_A1_B1_A1
Optimization infeasible because this subproblem isn't reachable.

### Candidate
type: B, layer: 1, pos: 41

### Candidate
type: B, layer: 1, pos: 29

## Relational analysis of NS_B2_A1_A2_B1_A1_B1_B1
Optimization infeasible because this subproblem isn't reachable.

### Candidate
type: A, layer: 1, pos: 29

### Candidate
type: B, layer: 1, pos: 20

## Relational analysis of NS_B2_A1_A2_B1_A1_B1_B1
Optimization infeasible because this subproblem isn't reachable.

### Candidate
type: A, layer: 1, pos: 20

### Candidate
type: A, layer: 1, pos: 5

## Relational analysis of NS_B2_A1_A2_B1_A1_B1_A1
Optimization infeasible because this subproblem isn't reachable.

### Candidate
type: B, layer: 1, pos: 5

### Candidate
type: A, layer: 1, pos: 16

## Relational analysis of NS_B2_A1_A2_B1_A1_B1_A1
Optimization infeasible because this subproblem isn't reachable.

### Candidate
type: B, layer: 1, pos: 16

### Candidate
type: A, layer: 1, pos: 21

## Relational analysis of NS_B2_A1_A2_B1_A1_B1_A1
Optimization infeasible because this subproblem isn't reachable.

### Candidate
type: B, layer: 1, pos: 21

### Candidate
type: A, layer: 1, pos: 23

## Relational analysis of NS_B2_A1_A2_B1_A1_B1_A1

### Relational analysis result of NS_B2_A1_A2_B1_A1_B1_A1
Status: Status.VERIFIED
Output dim: 0, lower bound: -63.3859320, upper bound: 63.3853893
time: 0.70 seconds

## Relational analysis of NS_B2_A1_A2_B1_A1_B1_A2

### Relational analysis result of NS_B2_A1_A2_B1_A1_B1_A2
Status: Status.UNKNOWN
Output dim: 0, lower bound: -63.3929322, upper bound: 63.3929322
time: 0.81 seconds

## BFS NS instance: NS_B2_A1_A2_B1_A1_B2

### Backsubstitution after applying NS history:
0: -53.9737701, 58.4838371, -53.8712502, 58.3831711, -112.3569183, 112.3550873
1: -396.7337952, 133.1474609, -396.2691650, 132.8637543, -528.4718018, 528.3070068
2: -212.7236023, 125.4101639, -212.2765808, 125.2092590, -337.9328613, 337.6867371
3: -273.8752441, 100.4784775, -273.4888916, 100.3042755, -373.1825256, 372.9573059
4: -152.9991302, 110.9224167, -152.6573334, 110.7395172, -263.7386475, 263.5797424

Time for backsubstitution: 1.13 seconds

### NS candidates at layer 1
type: B, layer: 1, pos: 4
type: A, layer: 1, pos: 4
type: A, layer: 1, pos: 9
type: B, layer: 1, pos: 9
type: B, layer: 1, pos: 35
type: A, layer: 1, pos: 35
type: A, layer: 1, pos: 17
type: B, layer: 1, pos: 17
type: A, layer: 1, pos: 45
type: B, layer: 1, pos: 45
type: B, layer: 1, pos: 19
type: A, layer: 1, pos: 19
type: A, layer: 1, pos: 3
type: B, layer: 1, pos: 3
type: B, layer: 1, pos: 10
type: A, layer: 1, pos: 10
type: A, layer: 1, pos: 18
type: A, layer: 1, pos: 15
type: B, layer: 1, pos: 18
type: B, layer: 1, pos: 15
type: B, layer: 1, pos: 39
type: A, layer: 1, pos: 39
type: B, layer: 1, pos: 37
type: A, layer: 1, pos: 43
type: B, layer: 1, pos: 43
type: A, layer: 1, pos: 38
type: B, layer: 1, pos: 32
type: A, layer: 1, pos: 37
type: B, layer: 1, pos: 38
type: A, layer: 1, pos: 32
type: B, layer: 1, pos: 41
type: B, layer: 1, pos: 29
type: A, layer: 1, pos: 41
type: A, layer: 1, pos: 29
type: B, layer: 1, pos: 20
type: A, layer: 1, pos: 20
type: A, layer: 1, pos: 5
type: B, layer: 1, pos: 16
type: B, layer: 1, pos: 5
type: A, layer: 1, pos: 16
type: A, layer: 1, pos: 21
type: B, layer: 1, pos: 23
type: B, layer: 1, pos: 21
type: A, layer: 1, pos: 23

Time for candidate selection: 0.09 seconds

### Candidate
type: B, layer: 1, pos: 4

### Candidate
type: A, layer: 1, pos: 4

### Candidate
type: A, layer: 1, pos: 9

### Candidate
type: B, layer: 1, pos: 9

### Candidate
type: B, layer: 1, pos: 35

### Candidate
type: A, layer: 1, pos: 35

### Candidate
type: A, layer: 1, pos: 17

### Candidate
type: B, layer: 1, pos: 17

### Candidate
type: A, layer: 1, pos: 45

### Candidate
type: B, layer: 1, pos: 45

### Candidate
type: B, layer: 1, pos: 19

### Candidate
type: A, layer: 1, pos: 19

### Candidate
type: A, layer: 1, pos: 3

### Candidate
type: B, layer: 1, pos: 3

### Candidate
type: B, layer: 1, pos: 10

### Candidate
type: A, layer: 1, pos: 10

### Candidate
type: A, layer: 1, pos: 18

## Relational analysis of NS_B2_A1_A2_B1_A1_B2_A1
Optimization infeasible because this subproblem isn't reachable.

### Candidate
type: A, layer: 1, pos: 15

## Relational analysis of NS_B2_A1_A2_B1_A1_B2_A1
Optimization infeasible because this subproblem isn't reachable.

### Candidate
type: B, layer: 1, pos: 18

### Candidate
type: B, layer: 1, pos: 15

### Candidate
type: B, layer: 1, pos: 39

## Relational analysis of NS_B2_A1_A2_B1_A1_B2_B1
Optimization infeasible because this subproblem isn't reachable.

### Candidate
type: A, layer: 1, pos: 39

### Candidate
type: B, layer: 1, pos: 37

## Relational analysis of NS_B2_A1_A2_B1_A1_B2_B1
Optimization infeasible because this subproblem isn't reachable.

### Candidate
type: A, layer: 1, pos: 43

## Relational analysis of NS_B2_A1_A2_B1_A1_B2_A1
Optimization infeasible because this subproblem isn't reachable.

### Candidate
type: B, layer: 1, pos: 43

### Candidate
type: A, layer: 1, pos: 38

## Relational analysis of NS_B2_A1_A2_B1_A1_B2_A1
Optimization infeasible because this subproblem isn't reachable.

### Candidate
type: B, layer: 1, pos: 32

## Relational analysis of NS_B2_A1_A2_B1_A1_B2_B1
Optimization infeasible because this subproblem isn't reachable.

### Candidate
type: A, layer: 1, pos: 37

### Candidate
type: B, layer: 1, pos: 38

### Candidate
type: A, layer: 1, pos: 32

### Candidate
type: B, layer: 1, pos: 41

## Relational analysis of NS_B2_A1_A2_B1_A1_B2_B1
Optimization infeasible because this subproblem isn't reachable.

### Candidate
type: B, layer: 1, pos: 29

## Relational analysis of NS_B2_A1_A2_B1_A1_B2_B1
Optimization infeasible because this subproblem isn't reachable.

### Candidate
type: A, layer: 1, pos: 41

### Candidate
type: A, layer: 1, pos: 29

### Candidate
type: B, layer: 1, pos: 20

## Relational analysis of NS_B2_A1_A2_B1_A1_B2_B1
Optimization infeasible because this subproblem isn't reachable.

### Candidate
type: A, layer: 1, pos: 20

### Candidate
type: A, layer: 1, pos: 5

## Relational analysis of NS_B2_A1_A2_B1_A1_B2_A1
Optimization infeasible because this subproblem isn't reachable.

### Candidate
type: B, layer: 1, pos: 16

## Relational analysis of NS_B2_A1_A2_B1_A1_B2_B1

### Relational analysis result of NS_B2_A1_A2_B1_A1_B2_B1
Status: Status.VERIFIED
Output dim: 0, lower bound: -63.3883271, upper bound: 63.3873208
time: 0.76 seconds

## Relational analysis of NS_B2_A1_A2_B1_A1_B2_B2
Optimization infeasible because this subproblem isn't reachable.

### Candidate
type: B, layer: 1, pos: 5

### Candidate
type: A, layer: 1, pos: 16

### Candidate
type: A, layer: 1, pos: 21

## Relational analysis of NS_B2_A1_A2_B1_A1_B2_A1
Optimization infeasible because this subproblem isn't reachable.

### Candidate
type: B, layer: 1, pos: 23

## Relational analysis of NS_B2_A1_A2_B1_A1_B2_B1
Optimization infeasible because this subproblem isn't reachable.

### Candidate
type: B, layer: 1, pos: 21

### Candidate
type: A, layer: 1, pos: 23

## BFS NS instance: NS_B2_A1_A2_B1_A2_B1

### Backsubstitution after applying NS history:
0: -53.8703156, 58.3821106, -53.9752045, 58.4854774, -112.3557892, 112.3572922
1: -396.2604980, 132.8612976, -396.7471008, 133.1512756, -528.3022461, 528.4822998
2: -212.2722015, 125.2070465, -212.7303009, 125.4136429, -337.6858215, 337.9373169
3: -273.4829102, 100.3024445, -273.8845825, 100.4812927, -372.9541931, 373.1899719
4: -152.6544037, 110.7377243, -153.0037079, 110.9251785, -263.5795593, 263.7414246

Time for backsubstitution: 1.09 seconds

### NS candidates at layer 1
type: B, layer: 1, pos: 4
type: A, layer: 1, pos: 4
type: B, layer: 1, pos: 9
type: A, layer: 1, pos: 9
type: A, layer: 1, pos: 35
type: B, layer: 1, pos: 35
type: A, layer: 1, pos: 17
type: B, layer: 1, pos: 17
type: B, layer: 1, pos: 45
type: B, layer: 1, pos: 3
type: A, layer: 1, pos: 19
type: A, layer: 1, pos: 45
type: B, layer: 1, pos: 19
type: A, layer: 1, pos: 3
type: A, layer: 1, pos: 10
type: B, layer: 1, pos: 10
type: B, layer: 1, pos: 18
type: B, layer: 1, pos: 15
type: A, layer: 1, pos: 18
type: A, layer: 1, pos: 15
type: A, layer: 1, pos: 39
type: B, layer: 1, pos: 39
type: B, layer: 1, pos: 43
type: B, layer: 1, pos: 38
type: A, layer: 1, pos: 32
type: B, layer: 1, pos: 37
type: A, layer: 1, pos: 37
type: A, layer: 1, pos: 43
type: A, layer: 1, pos: 38
type: B, layer: 1, pos: 32
type: A, layer: 1, pos: 41
type: B, layer: 1, pos: 29
type: B, layer: 1, pos: 41
type: A, layer: 1, pos: 29
type: A, layer: 1, pos: 20
type: B, layer: 1, pos: 20
type: B, layer: 1, pos: 5
type: A, layer: 1, pos: 16
type: A, layer: 1, pos: 5
type: B, layer: 1, pos: 16
type: A, layer: 1, pos: 23
type: B, layer: 1, pos: 21
type: A, layer: 1, pos: 21
type: B, layer: 1, pos: 23

Time for candidate selection: 0.09 seconds

### Candidate
type: B, layer: 1, pos: 4

### Candidate
type: A, layer: 1, pos: 4

### Candidate
type: B, layer: 1, pos: 9

### Candidate
type: A, layer: 1, pos: 9

### Candidate
type: A, layer: 1, pos: 35

### Candidate
type: B, layer: 1, pos: 35

### Candidate
type: A, layer: 1, pos: 17

### Candidate
type: B, layer: 1, pos: 17

### Candidate
type: B, layer: 1, pos: 45

### Candidate
type: B, layer: 1, pos: 3

### Candidate
type: A, layer: 1, pos: 19

### Candidate
type: A, layer: 1, pos: 45

### Candidate
type: B, layer: 1, pos: 19

### Candidate
type: A, layer: 1, pos: 3

### Candidate
type: A, layer: 1, pos: 10

### Candidate
type: B, layer: 1, pos: 10

### Candidate
type: B, layer: 1, pos: 18

## Relational analysis of NS_B2_A1_A2_B1_A2_B1_B1
Optimization infeasible because this subproblem isn't reachable.

### Candidate
type: B, layer: 1, pos: 15

## Relational analysis of NS_B2_A1_A2_B1_A2_B1_B1
Optimization infeasible because this subproblem isn't reachable.

### Candidate
type: A, layer: 1, pos: 18

### Candidate
type: A, layer: 1, pos: 15

### Candidate
type: A, layer: 1, pos: 39

## Relational analysis of NS_B2_A1_A2_B1_A2_B1_A1
Optimization infeasible because this subproblem isn't reachable.

### Candidate
type: B, layer: 1, pos: 39

### Candidate
type: B, layer: 1, pos: 43

## Relational analysis of NS_B2_A1_A2_B1_A2_B1_B1
Optimization infeasible because this subproblem isn't reachable.

### Candidate
type: B, layer: 1, pos: 38

## Relational analysis of NS_B2_A1_A2_B1_A2_B1_B1
Optimization infeasible because this subproblem isn't reachable.

### Candidate
type: A, layer: 1, pos: 32

## Relational analysis of NS_B2_A1_A2_B1_A2_B1_A1
Optimization infeasible because this subproblem isn't reachable.

### Candidate
type: B, layer: 1, pos: 37

## Relational analysis of NS_B2_A1_A2_B1_A2_B1_B1
Optimization infeasible because this subproblem isn't reachable.

### Candidate
type: A, layer: 1, pos: 37

### Candidate
type: A, layer: 1, pos: 43

### Candidate
type: A, layer: 1, pos: 38

### Candidate
type: B, layer: 1, pos: 32

### Candidate
type: A, layer: 1, pos: 41

## Relational analysis of NS_B2_A1_A2_B1_A2_B1_A1
Optimization infeasible because this subproblem isn't reachable.

### Candidate
type: B, layer: 1, pos: 29

## Relational analysis of NS_B2_A1_A2_B1_A2_B1_B1
Optimization infeasible because this subproblem isn't reachable.

### Candidate
type: B, layer: 1, pos: 41

### Candidate
type: A, layer: 1, pos: 29

### Candidate
type: A, layer: 1, pos: 20

## Relational analysis of NS_B2_A1_A2_B1_A2_B1_A1
Optimization infeasible because this subproblem isn't reachable.

### Candidate
type: B, layer: 1, pos: 20

### Candidate
type: B, layer: 1, pos: 5

## Relational analysis of NS_B2_A1_A2_B1_A2_B1_B1
Optimization infeasible because this subproblem isn't reachable.

### Candidate
type: A, layer: 1, pos: 16

## Relational analysis of NS_B2_A1_A2_B1_A2_B1_A1
Optimization infeasible because this subproblem isn't reachable.

### Candidate
type: A, layer: 1, pos: 5

### Candidate
type: B, layer: 1, pos: 16

### Candidate
type: A, layer: 1, pos: 23

## Relational analysis of NS_B2_A1_A2_B1_A2_B1_A1
Optimization infeasible because this subproblem isn't reachable.

### Candidate
type: B, layer: 1, pos: 21

## Relational analysis of NS_B2_A1_A2_B1_A2_B1_B1
Optimization infeasible because this subproblem isn't reachable.

### Candidate
type: A, layer: 1, pos: 21

### Candidate
type: B, layer: 1, pos: 23

## BFS NS instance: NS_B2_A1_A2_B1_A2_B2

### Backsubstitution after applying NS history:
0: -53.8703156, 58.3821106, -53.8712502, 58.3831711, -112.2534790, 112.2533417
1: -396.2604980, 132.8612976, -396.2691650, 132.8637543, -528.0918579, 528.0979004
2: -212.2722015, 125.2070465, -212.2765808, 125.2092590, -337.4814453, 337.4836121
3: -273.4829102, 100.3024445, -273.4888916, 100.3042755, -372.8360291, 372.8401489
4: -152.6544037, 110.7377243, -152.6573334, 110.7395172, -263.3939209, 263.3950500

Time for backsubstitution: 1.06 seconds

### NS candidates at layer 1
type: B, layer: 1, pos: 4
type: A, layer: 1, pos: 4
type: B, layer: 1, pos: 9
type: A, layer: 1, pos: 9
type: A, layer: 1, pos: 35
type: B, layer: 1, pos: 35
type: A, layer: 1, pos: 17
type: B, layer: 1, pos: 17
type: B, layer: 1, pos: 45
type: A, layer: 1, pos: 45
type: B, layer: 1, pos: 19
type: A, layer: 1, pos: 19
type: B, layer: 1, pos: 3
type: A, layer: 1, pos: 3
type: A, layer: 1, pos: 10
type: B, layer: 1, pos: 10
type: B, layer: 1, pos: 18
type: A, layer: 1, pos: 18
type: A, layer: 1, pos: 15
type: B, layer: 1, pos: 15
type: B, layer: 1, pos: 39
type: A, layer: 1, pos: 39
type: B, layer: 1, pos: 43
type: B, layer: 1, pos: 37
type: A, layer: 1, pos: 43
type: B, layer: 1, pos: 38
type: A, layer: 1, pos: 32
type: A, layer: 1, pos: 37
type: A, layer: 1, pos: 38
type: B, layer: 1, pos: 32
type: A, layer: 1, pos: 41
type: B, layer: 1, pos: 41
type: B, layer: 1, pos: 29
type: A, layer: 1, pos: 29
type: B, layer: 1, pos: 20
type: A, layer: 1, pos: 20
type: A, layer: 1, pos: 5
type: B, layer: 1, pos: 5
type: A, layer: 1, pos: 16
type: B, layer: 1, pos: 16
type: A, layer: 1, pos: 21
type: B, layer: 1, pos: 21
type: A, layer: 1, pos: 23
type: B, layer: 1, pos: 23

Time for candidate selection: 0.09 seconds

### Candidate
type: B, layer: 1, pos: 4

### Candidate
type: A, layer: 1, pos: 4

### Candidate
type: B, layer: 1, pos: 9

### Candidate
type: A, layer: 1, pos: 9

### Candidate
type: A, layer: 1, pos: 35

### Candidate
type: B, layer: 1, pos: 35

### Candidate
type: A, layer: 1, pos: 17

### Candidate
type: B, layer: 1, pos: 17

### Candidate
type: B, layer: 1, pos: 45

### Candidate
type: A, layer: 1, pos: 45

### Candidate
type: B, layer: 1, pos: 19

### Candidate
type: A, layer: 1, pos: 19

### Candidate
type: B, layer: 1, pos: 3

### Candidate
type: A, layer: 1, pos: 3

### Candidate
type: A, layer: 1, pos: 10

### Candidate
type: B, layer: 1, pos: 10

### Candidate
type: B, layer: 1, pos: 18

## Relational analysis of NS_B2_A1_A2_B1_A2_B2_B1
Optimization infeasible because this subproblem isn't reachable.

### Candidate
type: A, layer: 1, pos: 18

### Candidate
type: A, layer: 1, pos: 15

## Relational analysis of NS_B2_A1_A2_B1_A2_B2_A1
Optimization infeasible because this subproblem isn't reachable.

### Candidate
type: B, layer: 1, pos: 15

### Candidate
type: B, layer: 1, pos: 39

## Relational analysis of NS_B2_A1_A2_B1_A2_B2_B1
Optimization infeasible because this subproblem isn't reachable.

### Candidate
type: A, layer: 1, pos: 39

### Candidate
type: B, layer: 1, pos: 43

## Relational analysis of NS_B2_A1_A2_B1_A2_B2_B1
Optimization infeasible because this subproblem isn't reachable.

### Candidate
type: B, layer: 1, pos: 37

## Relational analysis of NS_B2_A1_A2_B1_A2_B2_B1
Optimization infeasible because this subproblem isn't reachable.

### Candidate
type: A, layer: 1, pos: 43

### Candidate
type: B, layer: 1, pos: 38

## Relational analysis of NS_B2_A1_A2_B1_A2_B2_B1
Optimization infeasible because this subproblem isn't reachable.

### Candidate
type: A, layer: 1, pos: 32

## Relational analysis of NS_B2_A1_A2_B1_A2_B2_A1
Optimization infeasible because this subproblem isn't reachable.

### Candidate
type: A, layer: 1, pos: 37

### Candidate
type: A, layer: 1, pos: 38

### Candidate
type: B, layer: 1, pos: 32

### Candidate
type: A, layer: 1, pos: 41

## Relational analysis of NS_B2_A1_A2_B1_A2_B2_A1
Optimization infeasible because this subproblem isn't reachable.

### Candidate
type: B, layer: 1, pos: 41

### Candidate
type: B, layer: 1, pos: 29

## Relational analysis of NS_B2_A1_A2_B1_A2_B2_B1
Optimization infeasible because this subproblem isn't reachable.

### Candidate
type: A, layer: 1, pos: 29

### Candidate
type: B, layer: 1, pos: 20

## Relational analysis of NS_B2_A1_A2_B1_A2_B2_B1
Optimization infeasible because this subproblem isn't reachable.

### Candidate
type: A, layer: 1, pos: 20

### Candidate
type: A, layer: 1, pos: 5

## Relational analysis of NS_B2_A1_A2_B1_A2_B2_A1
Optimization infeasible because this subproblem isn't reachable.

### Candidate
type: B, layer: 1, pos: 5

### Candidate
type: A, layer: 1, pos: 16

## Relational analysis of NS_B2_A1_A2_B1_A2_B2_A1
Optimization infeasible because this subproblem isn't reachable.

### Candidate
type: B, layer: 1, pos: 16

### Candidate
type: A, layer: 1, pos: 21

## Relational analysis of NS_B2_A1_A2_B1_A2_B2_A1
Optimization infeasible because this subproblem isn't reachable.

### Candidate
type: B, layer: 1, pos: 21

### Candidate
type: A, layer: 1, pos: 23

## Relational analysis of NS_B2_A1_A2_B1_A2_B2_A1
Optimization infeasible because this subproblem isn't reachable.

### Candidate
type: B, layer: 1, pos: 23

## BFS NS instance: NS_B2_A1_A2_B2_B1_A1

### Backsubstitution after applying NS history:
0: -53.9737701, 58.4838371, -54.1881714, 58.9965973, -112.9703598, 112.6720123
1: -396.7337952, 133.1474609, -402.2130432, 133.8377991, -529.5810547, 534.6861572
2: -212.7236023, 125.4101639, -214.3997955, 126.4415131, -339.1651001, 339.8099670
3: -273.8752441, 100.4784775, -277.1029663, 101.3699265, -374.3060608, 376.9176025
4: -152.9991302, 110.9224167, -153.4725342, 111.9196091, -264.9187317, 264.3949585

Time for backsubstitution: 1.41 seconds

### NS candidates at layer 1
type: B, layer: 1, pos: 35
type: A, layer: 1, pos: 35
type: B, layer: 1, pos: 9
type: A, layer: 1, pos: 9
type: A, layer: 1, pos: 19
type: B, layer: 1, pos: 19
type: A, layer: 1, pos: 3
type: B, layer: 1, pos: 3
type: A, layer: 1, pos: 17
type: B, layer: 1, pos: 17
type: B, layer: 1, pos: 10
type: A, layer: 1, pos: 18
type: A, layer: 1, pos: 10
type: A, layer: 1, pos: 45
type: B, layer: 1, pos: 15
type: A, layer: 1, pos: 43
type: B, layer: 1, pos: 18
type: A, layer: 1, pos: 15
type: B, layer: 1, pos: 29
type: B, layer: 1, pos: 43
type: B, layer: 1, pos: 45
type: A, layer: 1, pos: 29
type: B, layer: 1, pos: 21
type: A, layer: 1, pos: 21
type: B, layer: 1, pos: 23
type: B, layer: 1, pos: 32
type: A, layer: 1, pos: 37
type: A, layer: 1, pos: 23
type: A, layer: 1, pos: 38
type: A, layer: 1, pos: 32
type: A, layer: 1, pos: 4
type: B, layer: 1, pos: 39
type: A, layer: 1, pos: 20
type: B, layer: 1, pos: 37
type: A, layer: 1, pos: 41
type: B, layer: 1, pos: 41
type: B, layer: 1, pos: 20
type: A, layer: 1, pos: 39
type: B, layer: 1, pos: 16
type: A, layer: 1, pos: 5
type: B, layer: 1, pos: 5
type: A, layer: 1, pos: 16
type: B, layer: 1, pos: 4
type: B, layer: 1, pos: 38

Time for candidate selection: 0.10 seconds

### Candidate
type: B, layer: 1, pos: 35

### Candidate
type: A, layer: 1, pos: 35

### Candidate
type: B, layer: 1, pos: 9

### Candidate
type: A, layer: 1, pos: 9

### Candidate
type: A, layer: 1, pos: 19

### Candidate
type: B, layer: 1, pos: 19

### Candidate
type: A, layer: 1, pos: 3

### Candidate
type: B, layer: 1, pos: 3

### Candidate
type: A, layer: 1, pos: 17

### Candidate
type: B, layer: 1, pos: 17

### Candidate
type: B, layer: 1, pos: 10

### Candidate
type: A, layer: 1, pos: 18

### Candidate
type: A, layer: 1, pos: 10

### Candidate
type: A, layer: 1, pos: 45

### Candidate
type: B, layer: 1, pos: 15

### Candidate
type: A, layer: 1, pos: 43

### Candidate
type: B, layer: 1, pos: 18

### Candidate
type: A, layer: 1, pos: 15

### Candidate
type: B, layer: 1, pos: 29

### Candidate
type: B, layer: 1, pos: 43

### Candidate
type: B, layer: 1, pos: 45

### Candidate
type: A, layer: 1, pos: 29

### Candidate
type: B, layer: 1, pos: 21

## Relational analysis of NS_B2_A1_A2_B2_B1_A1_B1
Optimization infeasible because this subproblem isn't reachable.

### Candidate
type: A, layer: 1, pos: 21

### Candidate
type: B, layer: 1, pos: 23

## Relational analysis of NS_B2_A1_A2_B2_B1_A1_B1
Optimization infeasible because this subproblem isn't reachable.

### Candidate
type: B, layer: 1, pos: 32

## Relational analysis of NS_B2_A1_A2_B2_B1_A1_B1
Optimization infeasible because this subproblem isn't reachable.

### Candidate
type: A, layer: 1, pos: 37

## Relational analysis of NS_B2_A1_A2_B2_B1_A1_A1
Optimization infeasible because this subproblem isn't reachable.

### Candidate
type: A, layer: 1, pos: 23

### Candidate
type: A, layer: 1, pos: 38

## Relational analysis of NS_B2_A1_A2_B2_B1_A1_A1
Optimization infeasible because this subproblem isn't reachable.

### Candidate
type: A, layer: 1, pos: 32

### Candidate
type: A, layer: 1, pos: 4

### Candidate
type: B, layer: 1, pos: 39

## Relational analysis of NS_B2_A1_A2_B2_B1_A1_B1
Optimization infeasible because this subproblem isn't reachable.

### Candidate
type: A, layer: 1, pos: 20

## Relational analysis of NS_B2_A1_A2_B2_B1_A1_A1
Optimization infeasible because this subproblem isn't reachable.

### Candidate
type: B, layer: 1, pos: 37

### Candidate
type: A, layer: 1, pos: 41

## Relational analysis of NS_B2_A1_A2_B2_B1_A1_A1
Optimization infeasible because this subproblem isn't reachable.

### Candidate
type: B, layer: 1, pos: 41

### Candidate
type: B, layer: 1, pos: 20

### Candidate
type: A, layer: 1, pos: 39

### Candidate
type: B, layer: 1, pos: 16

## Relational analysis of NS_B2_A1_A2_B2_B1_A1_B1
Optimization infeasible because this subproblem isn't reachable.

### Candidate
type: A, layer: 1, pos: 5

## Relational analysis of NS_B2_A1_A2_B2_B1_A1_A1
Optimization infeasible because this subproblem isn't reachable.

### Candidate
type: B, layer: 1, pos: 5

### Candidate
type: A, layer: 1, pos: 16

### Candidate
type: B, layer: 1, pos: 4

### Candidate
type: B, layer: 1, pos: 38

## BFS NS instance: NS_B2_A1_A2_B2_B1_A2

### Backsubstitution after applying NS history:
0: -53.8703156, 58.3821106, -54.1881714, 58.9965973, -112.8669128, 112.5702820
1: -396.2604980, 132.8612976, -402.2130432, 133.8377991, -529.1090698, 534.3850098
2: -212.2722015, 125.2070465, -214.3997955, 126.4415131, -338.7137146, 339.6068420
3: -273.4829102, 100.3024445, -277.1029663, 101.3699265, -373.8986511, 376.7395020
4: -152.6544037, 110.7377243, -153.4725342, 111.9196091, -264.5739746, 264.2102661

Time for backsubstitution: 1.19 seconds

### NS candidates at layer 1
type: B, layer: 1, pos: 35
type: A, layer: 1, pos: 35
type: B, layer: 1, pos: 9
type: A, layer: 1, pos: 9
type: A, layer: 1, pos: 19
type: B, layer: 1, pos: 19
type: A, layer: 1, pos: 3
type: B, layer: 1, pos: 3
type: A, layer: 1, pos: 17
type: B, layer: 1, pos: 17
type: B, layer: 1, pos: 10
type: A, layer: 1, pos: 10
type: A, layer: 1, pos: 18
type: A, layer: 1, pos: 45
type: B, layer: 1, pos: 15
type: A, layer: 1, pos: 43
type: B, layer: 1, pos: 18
type: A, layer: 1, pos: 15
type: B, layer: 1, pos: 45
type: B, layer: 1, pos: 29
type: B, layer: 1, pos: 43
type: A, layer: 1, pos: 29
type: B, layer: 1, pos: 21
type: A, layer: 1, pos: 21
type: B, layer: 1, pos: 23
type: A, layer: 1, pos: 37
type: A, layer: 1, pos: 23
type: B, layer: 1, pos: 32
type: A, layer: 1, pos: 32
type: A, layer: 1, pos: 38
type: A, layer: 1, pos: 20
type: A, layer: 1, pos: 4
type: B, layer: 1, pos: 39
type: B, layer: 1, pos: 37
type: A, layer: 1, pos: 41
type: B, layer: 1, pos: 41
type: B, layer: 1, pos: 20
type: A, layer: 1, pos: 39
type: B, layer: 1, pos: 16
type: A, layer: 1, pos: 5
type: A, layer: 1, pos: 16
type: B, layer: 1, pos: 5
type: B, layer: 1, pos: 38
type: B, layer: 1, pos: 4

Time for candidate selection: 0.10 seconds

### Candidate
type: B, layer: 1, pos: 35

### Candidate
type: A, layer: 1, pos: 35

### Candidate
type: B, layer: 1, pos: 9

### Candidate
type: A, layer: 1, pos: 9

### Candidate
type: A, layer: 1, pos: 19

### Candidate
type: B, layer: 1, pos: 19

### Candidate
type: A, layer: 1, pos: 3

### Candidate
type: B, layer: 1, pos: 3

### Candidate
type: A, layer: 1, pos: 17

### Candidate
type: B, layer: 1, pos: 17

### Candidate
type: B, layer: 1, pos: 10

### Candidate
type: A, layer: 1, pos: 10

### Candidate
type: A, layer: 1, pos: 18

### Candidate
type: A, layer: 1, pos: 45

### Candidate
type: B, layer: 1, pos: 15

### Candidate
type: A, layer: 1, pos: 43

### Candidate
type: B, layer: 1, pos: 18

### Candidate
type: A, layer: 1, pos: 15

### Candidate
type: B, layer: 1, pos: 45

### Candidate
type: B, layer: 1, pos: 29

### Candidate
type: B, layer: 1, pos: 43

### Candidate
type: A, layer: 1, pos: 29

### Candidate
type: B, layer: 1, pos: 21

## Relational analysis of NS_B2_A1_A2_B2_B1_A2_B1
Optimization infeasible because this subproblem isn't reachable.

### Candidate
type: A, layer: 1, pos: 21

### Candidate
type: B, layer: 1, pos: 23

## Relational analysis of NS_B2_A1_A2_B2_B1_A2_B1
Optimization infeasible because this subproblem isn't reachable.

### Candidate
type: A, layer: 1, pos: 37

## Relational analysis of NS_B2_A1_A2_B2_B1_A2_A1
Optimization infeasible because this subproblem isn't reachable.

### Candidate
type: A, layer: 1, pos: 23

### Candidate
type: B, layer: 1, pos: 32

## Relational analysis of NS_B2_A1_A2_B2_B1_A2_B1
Optimization infeasible because this subproblem isn't reachable.

### Candidate
type: A, layer: 1, pos: 32

### Candidate
type: A, layer: 1, pos: 38

## Relational analysis of NS_B2_A1_A2_B2_B1_A2_A1
Optimization infeasible because this subproblem isn't reachable.

### Candidate
type: A, layer: 1, pos: 20

## Relational analysis of NS_B2_A1_A2_B2_B1_A2_A1
Optimization infeasible because this subproblem isn't reachable.

### Candidate
type: A, layer: 1, pos: 4

### Candidate
type: B, layer: 1, pos: 39

## Relational analysis of NS_B2_A1_A2_B2_B1_A2_B1
Optimization infeasible because this subproblem isn't reachable.

### Candidate
type: B, layer: 1, pos: 37

### Candidate
type: A, layer: 1, pos: 41

## Relational analysis of NS_B2_A1_A2_B2_B1_A2_A1
Optimization infeasible because this subproblem isn't reachable.

### Candidate
type: B, layer: 1, pos: 41

### Candidate
type: B, layer: 1, pos: 20

### Candidate
type: A, layer: 1, pos: 39

### Candidate
type: B, layer: 1, pos: 16

## Relational analysis of NS_B2_A1_A2_B2_B1_A2_B1
Optimization infeasible because this subproblem isn't reachable.

### Candidate
type: A, layer: 1, pos: 5

## Relational analysis of NS_B2_A1_A2_B2_B1_A2_A1
Optimization infeasible because this subproblem isn't reachable.

### Candidate
type: A, layer: 1, pos: 16

### Candidate
type: B, layer: 1, pos: 5

### Candidate
type: B, layer: 1, pos: 38

### Candidate
type: B, layer: 1, pos: 4

## BFS NS instance: NS_B2_A1_A2_B2_B2_A1

### Backsubstitution after applying NS history:
0: -53.9737701, 58.4838371, -53.8847351, 58.6575432, -112.6313019, 112.3685608
1: -396.7337952, 133.1474609, -399.7374878, 133.0946503, -528.8206177, 532.2303467
2: -212.7236023, 125.4101639, -213.1640167, 125.7145767, -338.4381714, 338.5741577
3: -273.8752441, 100.4784775, -275.4341431, 100.8090820, -373.7459717, 375.2457275
4: -152.9991302, 110.9224167, -152.5790253, 111.2873230, -264.2864380, 263.5014038

Time for backsubstitution: 1.25 seconds

### NS candidates at layer 1
type: B, layer: 1, pos: 35
type: A, layer: 1, pos: 35
type: B, layer: 1, pos: 9
type: A, layer: 1, pos: 9
type: A, layer: 1, pos: 19
type: B, layer: 1, pos: 19
type: A, layer: 1, pos: 3
type: B, layer: 1, pos: 3
type: A, layer: 1, pos: 17
type: B, layer: 1, pos: 17
type: B, layer: 1, pos: 10
type: A, layer: 1, pos: 45
type: A, layer: 1, pos: 18
type: A, layer: 1, pos: 10
type: B, layer: 1, pos: 15
type: A, layer: 1, pos: 43
type: B, layer: 1, pos: 18
type: A, layer: 1, pos: 15
type: B, layer: 1, pos: 29
type: B, layer: 1, pos: 43
type: B, layer: 1, pos: 45
type: A, layer: 1, pos: 29
type: B, layer: 1, pos: 21
type: B, layer: 1, pos: 23
type: A, layer: 1, pos: 21
type: B, layer: 1, pos: 32
type: A, layer: 1, pos: 37
type: A, layer: 1, pos: 23
type: A, layer: 1, pos: 38
type: B, layer: 1, pos: 39
type: A, layer: 1, pos: 32
type: B, layer: 1, pos: 37
type: A, layer: 1, pos: 20
type: A, layer: 1, pos: 4
type: B, layer: 1, pos: 41
type: B, layer: 1, pos: 20
type: A, layer: 1, pos: 41
type: A, layer: 1, pos: 39
type: B, layer: 1, pos: 16
type: A, layer: 1, pos: 5
type: B, layer: 1, pos: 5
type: B, layer: 1, pos: 4
type: A, layer: 1, pos: 16
type: B, layer: 1, pos: 38

Time for candidate selection: 0.10 seconds

### Candidate
type: B, layer: 1, pos: 35

### Candidate
type: A, layer: 1, pos: 35

### Candidate
type: B, layer: 1, pos: 9

### Candidate
type: A, layer: 1, pos: 9

### Candidate
type: A, layer: 1, pos: 19

### Candidate
type: B, layer: 1, pos: 19

### Candidate
type: A, layer: 1, pos: 3

### Candidate
type: B, layer: 1, pos: 3

### Candidate
type: A, layer: 1, pos: 17

### Candidate
type: B, layer: 1, pos: 17

### Candidate
type: B, layer: 1, pos: 10

### Candidate
type: A, layer: 1, pos: 45

### Candidate
type: A, layer: 1, pos: 18

### Candidate
type: A, layer: 1, pos: 10

### Candidate
type: B, layer: 1, pos: 15

### Candidate
type: A, layer: 1, pos: 43

### Candidate
type: B, layer: 1, pos: 18

### Candidate
type: A, layer: 1, pos: 15

### Candidate
type: B, layer: 1, pos: 29

## Relational analysis of NS_B2_A1_A2_B2_B2_A1_B1
Optimization infeasible because this subproblem isn't reachable.

### Candidate
type: B, layer: 1, pos: 43

### Candidate
type: B, layer: 1, pos: 45

### Candidate
type: A, layer: 1, pos: 29

### Candidate
type: B, layer: 1, pos: 21

## Relational analysis of NS_B2_A1_A2_B2_B2_A1_B1
Optimization infeasible because this subproblem isn't reachable.

### Candidate
type: B, layer: 1, pos: 23

## Relational analysis of NS_B2_A1_A2_B2_B2_A1_B1
Optimization infeasible because this subproblem isn't reachable.

### Candidate
type: A, layer: 1, pos: 21

### Candidate
type: B, layer: 1, pos: 32

## Relational analysis of NS_B2_A1_A2_B2_B2_A1_B1
Optimization infeasible because this subproblem isn't reachable.

### Candidate
type: A, layer: 1, pos: 37

## Relational analysis of NS_B2_A1_A2_B2_B2_A1_A1
Optimization infeasible because this subproblem isn't reachable.

### Candidate
type: A, layer: 1, pos: 23

### Candidate
type: A, layer: 1, pos: 38

## Relational analysis of NS_B2_A1_A2_B2_B2_A1_A1
Optimization infeasible because this subproblem isn't reachable.

### Candidate
type: B, layer: 1, pos: 39

## Relational analysis of NS_B2_A1_A2_B2_B2_A1_B1
Optimization infeasible because this subproblem isn't reachable.

### Candidate
type: A, layer: 1, pos: 32

### Candidate
type: B, layer: 1, pos: 37

### Candidate
type: A, layer: 1, pos: 20

## Relational analysis of NS_B2_A1_A2_B2_B2_A1_A1
Optimization infeasible because this subproblem isn't reachable.

### Candidate
type: A, layer: 1, pos: 4

### Candidate
type: B, layer: 1, pos: 41

## Relational analysis of NS_B2_A1_A2_B2_B2_A1_B1
Optimization infeasible because this subproblem isn't reachable.

### Candidate
type: B, layer: 1, pos: 20

### Candidate
type: A, layer: 1, pos: 41

### Candidate
type: A, layer: 1, pos: 39

### Candidate
type: B, layer: 1, pos: 16

## Relational analysis of NS_B2_A1_A2_B2_B2_A1_B1
Optimization infeasible because this subproblem isn't reachable.

### Candidate
type: A, layer: 1, pos: 5

## Relational analysis of NS_B2_A1_A2_B2_B2_A1_A1
Optimization infeasible because this subproblem isn't reachable.

### Candidate
type: B, layer: 1, pos: 5

### Candidate
type: B, layer: 1, pos: 4

### Candidate
type: A, layer: 1, pos: 16

### Candidate
type: B, layer: 1, pos: 38

## BFS NS instance: NS_B2_A1_A2_B2_B2_A2

### Backsubstitution after applying NS history:
0: -53.8703156, 58.3821106, -53.8847351, 58.6575432, -112.5278625, 112.2668228
1: -396.2604980, 132.8612976, -399.7374878, 133.0946503, -528.4409180, 532.0236206
2: -212.2722015, 125.2070465, -213.1640167, 125.7145767, -337.9867859, 338.3710327
3: -273.4829102, 100.3024445, -275.4341431, 100.8090820, -373.3998413, 375.1296692
4: -152.6544037, 110.7377243, -152.5790253, 111.2873230, -263.9416809, 263.3167419

Time for backsubstitution: 1.18 seconds

### NS candidates at layer 1
type: B, layer: 1, pos: 35
type: A, layer: 1, pos: 35
type: B, layer: 1, pos: 9
type: A, layer: 1, pos: 9
type: A, layer: 1, pos: 19
type: B, layer: 1, pos: 19
type: A, layer: 1, pos: 3
type: B, layer: 1, pos: 3
type: A, layer: 1, pos: 17
type: B, layer: 1, pos: 17
type: B, layer: 1, pos: 10
type: A, layer: 1, pos: 18
type: A, layer: 1, pos: 10
type: A, layer: 1, pos: 45
type: B, layer: 1, pos: 15
type: A, layer: 1, pos: 43
type: B, layer: 1, pos: 18
type: A, layer: 1, pos: 15
type: B, layer: 1, pos: 29
type: B, layer: 1, pos: 43
type: B, layer: 1, pos: 45
type: A, layer: 1, pos: 29
type: B, layer: 1, pos: 21
type: A, layer: 1, pos: 21
type: B, layer: 1, pos: 23
type: B, layer: 1, pos: 32
type: A, layer: 1, pos: 37
type: A, layer: 1, pos: 23
type: A, layer: 1, pos: 38
type: A, layer: 1, pos: 32
type: B, layer: 1, pos: 39
type: A, layer: 1, pos: 20
type: B, layer: 1, pos: 37
type: A, layer: 1, pos: 4
type: A, layer: 1, pos: 41
type: B, layer: 1, pos: 41
type: B, layer: 1, pos: 20
type: A, layer: 1, pos: 39
type: B, layer: 1, pos: 16
type: A, layer: 1, pos: 5
type: B, layer: 1, pos: 5
type: A, layer: 1, pos: 16
type: B, layer: 1, pos: 38
type: B, layer: 1, pos: 4

Time for candidate selection: 0.10 seconds

### Candidate
type: B, layer: 1, pos: 35

### Candidate
type: A, layer: 1, pos: 35

### Candidate
type: B, layer: 1, pos: 9

### Candidate
type: A, layer: 1, pos: 9

### Candidate
type: A, layer: 1, pos: 19

### Candidate
type: B, layer: 1, pos: 19

### Candidate
type: A, layer: 1, pos: 3

### Candidate
type: B, layer: 1, pos: 3

### Candidate
type: A, layer: 1, pos: 17

### Candidate
type: B, layer: 1, pos: 17

### Candidate
type: B, layer: 1, pos: 10

### Candidate
type: A, layer: 1, pos: 18

### Candidate
type: A, layer: 1, pos: 10

### Candidate
type: A, layer: 1, pos: 45

### Candidate
type: B, layer: 1, pos: 15

### Candidate
type: A, layer: 1, pos: 43

### Candidate
type: B, layer: 1, pos: 18

### Candidate
type: A, layer: 1, pos: 15

### Candidate
type: B, layer: 1, pos: 29

## Relational analysis of NS_B2_A1_A2_B2_B2_A2_B1
Optimization infeasible because this subproblem isn't reachable.

### Candidate
type: B, layer: 1, pos: 43

### Candidate
type: B, layer: 1, pos: 45

### Candidate
type: A, layer: 1, pos: 29

### Candidate
type: B, layer: 1, pos: 21

## Relational analysis of NS_B2_A1_A2_B2_B2_A2_B1
Optimization infeasible because this subproblem isn't reachable.

### Candidate
type: A, layer: 1, pos: 21

### Candidate
type: B, layer: 1, pos: 23

## Relational analysis of NS_B2_A1_A2_B2_B2_A2_B1
Optimization infeasible because this subproblem isn't reachable.

### Candidate
type: B, layer: 1, pos: 32

## Relational analysis of NS_B2_A1_A2_B2_B2_A2_B1
Optimization infeasible because this subproblem isn't reachable.

### Candidate
type: A, layer: 1, pos: 37

## Relational analysis of NS_B2_A1_A2_B2_B2_A2_A1
Optimization infeasible because this subproblem isn't reachable.

### Candidate
type: A, layer: 1, pos: 23

### Candidate
type: A, layer: 1, pos: 38

## Relational analysis of NS_B2_A1_A2_B2_B2_A2_A1
Optimization infeasible because this subproblem isn't reachable.

### Candidate
type: A, layer: 1, pos: 32

### Candidate
type: B, layer: 1, pos: 39

## Relational analysis of NS_B2_A1_A2_B2_B2_A2_B1
Optimization infeasible because this subproblem isn't reachable.

### Candidate
type: A, layer: 1, pos: 20

## Relational analysis of NS_B2_A1_A2_B2_B2_A2_A1
Optimization infeasible because this subproblem isn't reachable.

### Candidate
type: B, layer: 1, pos: 37

### Candidate
type: A, layer: 1, pos: 4

### Candidate
type: A, layer: 1, pos: 41

## Relational analysis of NS_B2_A1_A2_B2_B2_A2_A1
Optimization infeasible because this subproblem isn't reachable.

### Candidate
type: B, layer: 1, pos: 41

### Candidate
type: B, layer: 1, pos: 20

### Candidate
type: A, layer: 1, pos: 39

### Candidate
type: B, layer: 1, pos: 16

## Relational analysis of NS_B2_A1_A2_B2_B2_A2_B1
Optimization infeasible because this subproblem isn't reachable.

### Candidate
type: A, layer: 1, pos: 5

## Relational analysis of NS_B2_A1_A2_B2_B2_A2_A1
Optimization infeasible because this subproblem isn't reachable.

### Candidate
type: B, layer: 1, pos: 5

### Candidate
type: A, layer: 1, pos: 16

### Candidate
type: B, layer: 1, pos: 38

### Candidate
type: B, layer: 1, pos: 4

## BFS NS instance: NS_B2_A2_A1_A1_B1_A1

### Backsubstitution after applying NS history:
0: -28.5664749, 30.1689701, -51.8698730, 56.3319435, -84.8984222, 82.0388412
1: -205.3779755, 70.8297882, -382.8943176, 127.9916611, -333.3695984, 453.0242920
2: -113.2364197, 65.0554352, -204.6689301, 120.7612076, -233.9976196, 269.7243347
3: -142.5589294, 52.1684952, -264.0834351, 96.7237778, -239.2827148, 315.9387512
4: -81.4878922, 56.8995934, -147.0722046, 106.7969894, -188.2848816, 203.9717865

Time for backsubstitution: 1.09 seconds

### NS candidates at layer 1
type: A, layer: 1, pos: 4
type: B, layer: 1, pos: 45
type: A, layer: 1, pos: 3
type: A, layer: 1, pos: 9
type: A, layer: 1, pos: 17
type: A, layer: 1, pos: 38
type: A, layer: 1, pos: 19
type: B, layer: 1, pos: 35
type: B, layer: 1, pos: 43
type: B, layer: 1, pos: 9
type: B, layer: 1, pos: 19
type: B, layer: 1, pos: 3
type: B, layer: 1, pos: 11
type: A, layer: 1, pos: 10
type: A, layer: 1, pos: 11
type: A, layer: 1, pos: 35
type: B, layer: 1, pos: 10
type: B, layer: 1, pos: 15
type: B, layer: 1, pos: 17
type: B, layer: 1, pos: 16
type: A, layer: 1, pos: 37
type: B, layer: 1, pos: 39
type: A, layer: 1, pos: 32
type: A, layer: 1, pos: 43
type: B, layer: 1, pos: 18
type: A, layer: 1, pos: 21
type: A, layer: 1, pos: 15
type: B, layer: 1, pos: 23
type: A, layer: 1, pos: 41
type: A, layer: 1, pos: 18
type: B, layer: 1, pos: 20
type: B, layer: 1, pos: 41
type: B, layer: 1, pos: 32
type: A, layer: 1, pos: 5
type: B, layer: 1, pos: 5
type: A, layer: 1, pos: 39
type: B, layer: 1, pos: 21
type: A, layer: 1, pos: 20
type: A, layer: 1, pos: 23
type: B, layer: 1, pos: 37
type: A, layer: 1, pos: 16
type: B, layer: 1, pos: 38
type: B, layer: 1, pos: 4
type: B, layer: 1, pos: 29

Time for candidate selection: 0.10 seconds

### Candidate
type: A, layer: 1, pos: 4

### Candidate
type: B, layer: 1, pos: 45

## Relational analysis of NS_B2_A2_A1_A1_B1_A1_B1
Optimization infeasible because this subproblem isn't reachable.

### Candidate
type: A, layer: 1, pos: 3

### Candidate
type: A, layer: 1, pos: 9

### Candidate
type: A, layer: 1, pos: 17

### Candidate
type: A, layer: 1, pos: 38

### Candidate
type: A, layer: 1, pos: 19

### Candidate
type: B, layer: 1, pos: 35

### Candidate
type: B, layer: 1, pos: 43

## Relational analysis of NS_B2_A2_A1_A1_B1_A1_B1
Optimization infeasible because this subproblem isn't reachable.

### Candidate
type: B, layer: 1, pos: 9

### Candidate
type: B, layer: 1, pos: 19

### Candidate
type: B, layer: 1, pos: 3

### Candidate
type: B, layer: 1, pos: 11

## Relational analysis of NS_B2_A2_A1_A1_B1_A1_B1

### Relational analysis result of NS_B2_A2_A1_A1_B1_A1_B1
Status: Status.UNKNOWN
Output dim: 0, lower bound: -63.3936520, upper bound: 63.3940690
time: 0.94 seconds

## Relational analysis of NS_B2_A2_A1_A1_B1_A1_B2

### Relational analysis result of NS_B2_A2_A1_A1_B1_A1_B2
Status: Status.UNKNOWN
Output dim: 0, lower bound: -63.3937463, upper bound: 63.3942170
time: 0.97 seconds

## BFS NS instance: NS_B2_A2_A1_A1_B1_A2

### Backsubstitution after applying NS history:
0: -31.3312283, 33.2471046, -49.0097733, 53.2279167, -84.5591354, 82.2568817
1: -229.0442810, 77.7797470, -362.4377441, 120.9739456, -350.0182190, 439.6603699
2: -125.1618118, 72.0901031, -193.7615662, 114.1725082, -239.3343201, 265.8516235
3: -158.3682098, 57.6655502, -250.0415344, 91.4295197, -249.7977295, 307.4492188
4: -89.2824326, 62.8486900, -139.2727051, 101.0217896, -190.3041992, 202.1213837

Time for backsubstitution: 1.09 seconds

### NS candidates at layer 1
type: A, layer: 1, pos: 4
type: A, layer: 1, pos: 3
type: B, layer: 1, pos: 35
type: A, layer: 1, pos: 9
type: A, layer: 1, pos: 35
type: B, layer: 1, pos: 45
type: B, layer: 1, pos: 3
type: A, layer: 1, pos: 19
type: B, layer: 1, pos: 9
type: A, layer: 1, pos: 38
type: B, layer: 1, pos: 19
type: A, layer: 1, pos: 17
type: B, layer: 1, pos: 43
type: B, layer: 1, pos: 15
type: A, layer: 1, pos: 10
type: B, layer: 1, pos: 11
type: B, layer: 1, pos: 4
type: A, layer: 1, pos: 11
type: B, layer: 1, pos: 18
type: B, layer: 1, pos: 10
type: A, layer: 1, pos: 18
type: B, layer: 1, pos: 17
type: A, layer: 1, pos: 43
type: A, layer: 1, pos: 21
type: B, layer: 1, pos: 23
type: A, layer: 1, pos: 15
type: A, layer: 1, pos: 37
type: A, layer: 1, pos: 32
type: B, layer: 1, pos: 16
type: B, layer: 1, pos: 21
type: B, layer: 1, pos: 39
type: A, layer: 1, pos: 23
type: B, layer: 1, pos: 32
type: B, layer: 1, pos: 20
type: A, layer: 1, pos: 41
type: A, layer: 1, pos: 20
type: B, layer: 1, pos: 41
type: A, layer: 1, pos: 39
type: A, layer: 1, pos: 5
type: B, layer: 1, pos: 5
type: B, layer: 1, pos: 38
type: B, layer: 1, pos: 37
type: A, layer: 1, pos: 16
type: B, layer: 1, pos: 29

Time for candidate selection: 0.09 seconds

### Candidate
type: A, layer: 1, pos: 4

### Candidate
type: A, layer: 1, pos: 3

### Candidate
type: B, layer: 1, pos: 35

### Candidate
type: A, layer: 1, pos: 9

### Candidate
type: A, layer: 1, pos: 35

### Candidate
type: B, layer: 1, pos: 45

## Relational analysis of NS_B2_A2_A1_A1_B1_A2_B1
Optimization infeasible because this subproblem isn't reachable.

### Candidate
type: B, layer: 1, pos: 3

### Candidate
type: A, layer: 1, pos: 19

### Candidate
type: B, layer: 1, pos: 9

### Candidate
type: A, layer: 1, pos: 38

### Candidate
type: B, layer: 1, pos: 19

### Candidate
type: A, layer: 1, pos: 17

### Candidate
type: B, layer: 1, pos: 43

## Relational analysis of NS_B2_A2_A1_A1_B1_A2_B1
Optimization infeasible because this subproblem isn't reachable.

### Candidate
type: B, layer: 1, pos: 15

## Relational analysis of NS_B2_A2_A1_A1_B1_A2_B1
Optimization infeasible because this subproblem isn't reachable.

### Candidate
type: A, layer: 1, pos: 10

## Relational analysis of NS_B2_A2_A1_A1_B1_A2_A1
Optimization infeasible because this subproblem isn't reachable.

### Candidate
type: B, layer: 1, pos: 11

## Relational analysis of NS_B2_A2_A1_A1_B1_A2_B1

### Relational analysis result of NS_B2_A2_A1_A1_B1_A2_B1
Status: Status.UNKNOWN
Output dim: 0, lower bound: -63.3936447, upper bound: 63.3940670
time: 0.77 seconds

## Relational analysis of NS_B2_A2_A1_A1_B1_A2_B2

### Relational analysis result of NS_B2_A2_A1_A1_B1_A2_B2
Status: Status.UNKNOWN
Output dim: 0, lower bound: -63.3937338, upper bound: 63.3942133
time: 0.65 seconds

## BFS NS instance: NS_B2_A2_A1_A1_B2_A1

### Backsubstitution after applying NS history:
0: -28.5664749, 30.1689701, -52.2746696, 56.9984779, -85.5649490, 82.4436417
1: -205.3779755, 70.8297882, -389.0621033, 129.1828613, -334.5608521, 459.8918457
2: -113.2364197, 65.0554352, -207.0110779, 122.1272202, -235.3636322, 272.0664978
3: -142.5589294, 52.1684952, -267.8845215, 97.9457474, -240.5046692, 320.0530090
4: -81.4878922, 56.8995934, -148.0931091, 108.0790558, -189.5669403, 204.9926910

Time for backsubstitution: 1.20 seconds

### NS candidates at layer 1
type: A, layer: 1, pos: 4
type: A, layer: 1, pos: 3
type: B, layer: 1, pos: 35
type: B, layer: 1, pos: 3
type: A, layer: 1, pos: 19
type: A, layer: 1, pos: 9
type: A, layer: 1, pos: 38
type: A, layer: 1, pos: 35
type: B, layer: 1, pos: 45
type: B, layer: 1, pos: 9
type: A, layer: 1, pos: 17
type: B, layer: 1, pos: 43
type: B, layer: 1, pos: 19
type: B, layer: 1, pos: 4
type: B, layer: 1, pos: 15
type: A, layer: 1, pos: 43
type: B, layer: 1, pos: 11
type: A, layer: 1, pos: 11
type: A, layer: 1, pos: 10
type: B, layer: 1, pos: 10
type: B, layer: 1, pos: 23
type: A, layer: 1, pos: 37
type: A, layer: 1, pos: 21
type: B, layer: 1, pos: 17
type: B, layer: 1, pos: 16
type: B, layer: 1, pos: 18
type: B, layer: 1, pos: 39
type: A, layer: 1, pos: 18
type: B, layer: 1, pos: 21
type: B, layer: 1, pos: 32
type: A, layer: 1, pos: 23
type: A, layer: 1, pos: 15
type: B, layer: 1, pos: 20
type: A, layer: 1, pos: 32
type: A, layer: 1, pos: 20
type: A, layer: 1, pos: 41
type: B, layer: 1, pos: 41
type: A, layer: 1, pos: 5
type: B, layer: 1, pos: 5
type: A, layer: 1, pos: 39
type: A, layer: 1, pos: 16
type: B, layer: 1, pos: 38
type: B, layer: 1, pos: 37
type: B, layer: 1, pos: 29

Time for candidate selection: 0.09 seconds

### Candidate
type: A, layer: 1, pos: 4

### Candidate
type: A, layer: 1, pos: 3

### Candidate
type: B, layer: 1, pos: 35

### Candidate
type: B, layer: 1, pos: 3

### Candidate
type: A, layer: 1, pos: 19

### Candidate
type: A, layer: 1, pos: 9

### Candidate
type: A, layer: 1, pos: 38

### Candidate
type: A, layer: 1, pos: 35

### Candidate
type: B, layer: 1, pos: 45

## Relational analysis of NS_B2_A2_A1_A1_B2_A1_B1
Optimization infeasible because this subproblem isn't reachable.

### Candidate
type: B, layer: 1, pos: 9

### Candidate
type: A, layer: 1, pos: 17

### Candidate
type: B, layer: 1, pos: 43

### Candidate
type: B, layer: 1, pos: 19

### Candidate
type: B, layer: 1, pos: 4

### Candidate
type: B, layer: 1, pos: 15

### Candidate
type: A, layer: 1, pos: 43

### Candidate
type: B, layer: 1, pos: 11

## Relational analysis of NS_B2_A2_A1_A1_B2_A1_B1

### Relational analysis result of NS_B2_A2_A1_A1_B2_A1_B1
Status: Status.UNKNOWN
Output dim: 0, lower bound: -63.3937633, upper bound: 63.3942024
time: 0.86 seconds

## Relational analysis of NS_B2_A2_A1_A1_B2_A1_B2

### Relational analysis result of NS_B2_A2_A1_A1_B2_A1_B2
Status: Status.UNKNOWN
Output dim: 0, lower bound: -63.3942786, upper bound: 63.3945493
time: 0.76 seconds

## BFS NS instance: NS_B2_A2_A1_A1_B2_A2

### Backsubstitution after applying NS history:
0: -31.3312283, 33.2471046, -49.1084404, 53.5609856, -84.8922119, 82.3555450
1: -229.0442810, 77.7797470, -365.8971252, 121.2925720, -350.3368530, 443.6768799
2: -125.1618118, 72.0901031, -194.7173767, 114.7962875, -239.9580841, 266.8074646
3: -158.3682098, 57.6655502, -252.0432892, 91.9607239, -250.3289337, 309.7088318
4: -89.2824326, 62.8486900, -139.2741394, 101.6966629, -190.9790955, 202.1228333

Time for backsubstitution: 1.07 seconds

### NS candidates at layer 1
type: A, layer: 1, pos: 4
type: A, layer: 1, pos: 3
type: B, layer: 1, pos: 35
type: A, layer: 1, pos: 9
type: B, layer: 1, pos: 3
type: A, layer: 1, pos: 35
type: A, layer: 1, pos: 19
type: B, layer: 1, pos: 45
type: A, layer: 1, pos: 38
type: B, layer: 1, pos: 9
type: A, layer: 1, pos: 17
type: B, layer: 1, pos: 19
type: B, layer: 1, pos: 43
type: B, layer: 1, pos: 4
type: B, layer: 1, pos: 15
type: B, layer: 1, pos: 11
type: A, layer: 1, pos: 10
type: A, layer: 1, pos: 11
type: A, layer: 1, pos: 18
type: A, layer: 1, pos: 43
type: B, layer: 1, pos: 10
type: B, layer: 1, pos: 18
type: B, layer: 1, pos: 17
type: A, layer: 1, pos: 21
type: A, layer: 1, pos: 37
type: B, layer: 1, pos: 23
type: B, layer: 1, pos: 32
type: B, layer: 1, pos: 39
type: B, layer: 1, pos: 21
type: B, layer: 1, pos: 16
type: A, layer: 1, pos: 15
type: A, layer: 1, pos: 23
type: A, layer: 1, pos: 32
type: B, layer: 1, pos: 20
type: A, layer: 1, pos: 41
type: A, layer: 1, pos: 20
type: B, layer: 1, pos: 41
type: A, layer: 1, pos: 5
type: A, layer: 1, pos: 39
type: B, layer: 1, pos: 5
type: A, layer: 1, pos: 16
type: B, layer: 1, pos: 37
type: B, layer: 1, pos: 38
type: B, layer: 1, pos: 29

Time for candidate selection: 0.09 seconds

### Candidate
type: A, layer: 1, pos: 4

### Candidate
type: A, layer: 1, pos: 3

### Candidate
type: B, layer: 1, pos: 35

### Candidate
type: A, layer: 1, pos: 9

### Candidate
type: B, layer: 1, pos: 3

### Candidate
type: A, layer: 1, pos: 35

### Candidate
type: A, layer: 1, pos: 19

### Candidate
type: B, layer: 1, pos: 45

## Relational analysis of NS_B2_A2_A1_A1_B2_A2_B1
Optimization infeasible because this subproblem isn't reachable.

### Candidate
type: A, layer: 1, pos: 38

### Candidate
type: B, layer: 1, pos: 9

### Candidate
type: A, layer: 1, pos: 17

### Candidate
type: B, layer: 1, pos: 19

### Candidate
type: B, layer: 1, pos: 43

### Candidate
type: B, layer: 1, pos: 4

### Candidate
type: B, layer: 1, pos: 15

### Candidate
type: B, layer: 1, pos: 11

## Relational analysis of NS_B2_A2_A1_A1_B2_A2_B1

### Relational analysis result of NS_B2_A2_A1_A1_B2_A2_B1
Status: Status.UNKNOWN
Output dim: 0, lower bound: -63.3937664, upper bound: 63.3942169
time: 0.73 seconds

## Relational analysis of NS_B2_A2_A1_A1_B2_A2_B2

### Relational analysis result of NS_B2_A2_A1_A1_B2_A2_B2
Status: Status.UNKNOWN
Output dim: 0, lower bound: -63.3942575, upper bound: 63.3945567
time: 0.74 seconds

## BFS NS instance: NS_B2_A2_A1_A2_B1_B1

### Backsubstitution after applying NS history:
0: -41.3362274, 42.5331116, -52.3831139, 56.7113609, -98.0101166, 94.9162064
1: -279.1030579, 102.3374252, -383.6714172, 129.1607971, -408.2638550, 484.1629639
2: -160.6312256, 92.0329819, -206.2644196, 121.5206833, -281.6412659, 298.2973938
3: -196.5596924, 73.6476669, -265.0341492, 97.3871384, -293.9468384, 338.3572083
4: -117.3278351, 80.5484772, -148.4291382, 107.5487595, -224.0611420, 228.9776154

Time for backsubstitution: 1.14 seconds

### NS candidates at layer 1
type: A, layer: 1, pos: 4
type: A, layer: 1, pos: 9
type: A, layer: 1, pos: 19
type: B, layer: 1, pos: 35
type: A, layer: 1, pos: 3
type: A, layer: 1, pos: 17
type: B, layer: 1, pos: 9
type: A, layer: 1, pos: 35
type: A, layer: 1, pos: 45
type: B, layer: 1, pos: 10
type: B, layer: 1, pos: 19
type: B, layer: 1, pos: 15
type: B, layer: 1, pos: 17
type: A, layer: 1, pos: 38
type: A, layer: 1, pos: 18
type: B, layer: 1, pos: 4
type: B, layer: 1, pos: 3
type: A, layer: 1, pos: 11
type: A, layer: 1, pos: 10
type: B, layer: 1, pos: 29
type: B, layer: 1, pos: 16
type: A, layer: 1, pos: 37
type: B, layer: 1, pos: 45
type: B, layer: 1, pos: 39
type: B, layer: 1, pos: 32
type: B, layer: 1, pos: 18
type: A, layer: 1, pos: 32
type: A, layer: 1, pos: 15
type: A, layer: 1, pos: 21
type: A, layer: 1, pos: 39
type: B, layer: 1, pos: 20
type: A, layer: 1, pos: 41
type: B, layer: 1, pos: 43
type: A, layer: 1, pos: 43
type: B, layer: 1, pos: 41
type: A, layer: 1, pos: 5
type: B, layer: 1, pos: 23
type: A, layer: 1, pos: 23
type: B, layer: 1, pos: 5
type: B, layer: 1, pos: 21
type: A, layer: 1, pos: 20
type: B, layer: 1, pos: 38
type: B, layer: 1, pos: 37
type: A, layer: 1, pos: 16

Time for candidate selection: 0.09 seconds

### Candidate
type: A, layer: 1, pos: 4

### Candidate
type: A, layer: 1, pos: 9

### Candidate
type: A, layer: 1, pos: 19

### Candidate
type: B, layer: 1, pos: 35

### Candidate
type: A, layer: 1, pos: 3

### Candidate
type: A, layer: 1, pos: 17

### Candidate
type: B, layer: 1, pos: 9

### Candidate
type: A, layer: 1, pos: 35

### Candidate
type: A, layer: 1, pos: 45

### Candidate
type: B, layer: 1, pos: 10

### Candidate
type: B, layer: 1, pos: 19

### Candidate
type: B, layer: 1, pos: 15

## Relational analysis of NS_B2_A2_A1_A2_B1_B1_B1
Optimization infeasible because this subproblem isn't reachable.

### Candidate
type: B, layer: 1, pos: 17

### Candidate
type: A, layer: 1, pos: 38

## Relational analysis of NS_B2_A2_A1_A2_B1_B1_A1
Optimization infeasible because this subproblem isn't reachable.

### Candidate
type: A, layer: 1, pos: 18

## Relational analysis of NS_B2_A2_A1_A2_B1_B1_A1
Optimization infeasible because this subproblem isn't reachable.

### Candidate
type: B, layer: 1, pos: 4

### Candidate
type: B, layer: 1, pos: 3

### Candidate
type: A, layer: 1, pos: 11

## Relational analysis of NS_B2_A2_A1_A2_B1_B1_A1

### Relational analysis result of NS_B2_A2_A1_A2_B1_B1_A1
Status: Status.UNKNOWN
Output dim: 0, lower bound: -63.3931573, upper bound: 63.3932422
time: 0.81 seconds

## Relational analysis of NS_B2_A2_A1_A2_B1_B1_A2

### Relational analysis result of NS_B2_A2_A1_A2_B1_B1_A2
Status: Status.UNKNOWN
Output dim: 0, lower bound: -63.3931573, upper bound: 63.3940319
time: 0.75 seconds

## BFS NS instance: NS_B2_A2_A1_A2_B1_B2

### Backsubstitution after applying NS history:
0: -41.4766655, 42.6777725, -52.2842216, 56.6177788, -98.0716095, 94.9619904
1: -280.1462708, 102.6841431, -383.2715454, 128.8799133, -409.0261230, 484.1998596
2: -161.1961517, 92.3508987, -205.7797241, 121.3362656, -282.0737610, 298.1306152
3: -197.2792816, 73.9040070, -264.6797180, 97.2263565, -294.5056458, 338.3065186
4: -117.7232513, 80.8190765, -148.0908661, 107.3788910, -224.3143158, 228.9099426

Time for backsubstitution: 1.08 seconds

### NS candidates at layer 1
type: A, layer: 1, pos: 4
type: A, layer: 1, pos: 9
type: A, layer: 1, pos: 19
type: B, layer: 1, pos: 35
type: A, layer: 1, pos: 3
type: A, layer: 1, pos: 17
type: B, layer: 1, pos: 9
type: A, layer: 1, pos: 35
type: A, layer: 1, pos: 45
type: B, layer: 1, pos: 10
type: B, layer: 1, pos: 19
type: B, layer: 1, pos: 15
type: A, layer: 1, pos: 11
type: B, layer: 1, pos: 17
type: A, layer: 1, pos: 38
type: A, layer: 1, pos: 18
type: B, layer: 1, pos: 4
type: B, layer: 1, pos: 3
type: A, layer: 1, pos: 10
type: B, layer: 1, pos: 29
type: B, layer: 1, pos: 16
type: A, layer: 1, pos: 37
type: B, layer: 1, pos: 39
type: B, layer: 1, pos: 32
type: B, layer: 1, pos: 45
type: B, layer: 1, pos: 18
type: A, layer: 1, pos: 15
type: A, layer: 1, pos: 32
type: A, layer: 1, pos: 21
type: A, layer: 1, pos: 39
type: B, layer: 1, pos: 20
type: A, layer: 1, pos: 41
type: A, layer: 1, pos: 43
type: B, layer: 1, pos: 43
type: B, layer: 1, pos: 41
type: A, layer: 1, pos: 5
type: B, layer: 1, pos: 23
type: A, layer: 1, pos: 23
type: B, layer: 1, pos: 5
type: B, layer: 1, pos: 21
type: A, layer: 1, pos: 20
type: B, layer: 1, pos: 37
type: B, layer: 1, pos: 38
type: A, layer: 1, pos: 16

Time for candidate selection: 0.09 seconds

### Candidate
type: A, layer: 1, pos: 4

### Candidate
type: A, layer: 1, pos: 9

### Candidate
type: A, layer: 1, pos: 19

### Candidate
type: B, layer: 1, pos: 35

### Candidate
type: A, layer: 1, pos: 3

### Candidate
type: A, layer: 1, pos: 17

### Candidate
type: B, layer: 1, pos: 9

### Candidate
type: A, layer: 1, pos: 35

### Candidate
type: A, layer: 1, pos: 45

### Candidate
type: B, layer: 1, pos: 10

### Candidate
type: B, layer: 1, pos: 19

### Candidate
type: B, layer: 1, pos: 15

## Relational analysis of NS_B2_A2_A1_A2_B1_B2_B1
Optimization infeasible because this subproblem isn't reachable.

### Candidate
type: A, layer: 1, pos: 11

## Relational analysis of NS_B2_A2_A1_A2_B1_B2_A1

### Relational analysis result of NS_B2_A2_A1_A2_B1_B2_A1
Status: Status.UNKNOWN
Output dim: 0, lower bound: -63.3932236, upper bound: 63.3933818
time: 0.76 seconds

## Relational analysis of NS_B2_A2_A1_A2_B1_B2_A2

### Relational analysis result of NS_B2_A2_A1_A2_B1_B2_A2
Status: Status.UNKNOWN
Output dim: 0, lower bound: -63.3932236, upper bound: 63.3941691
time: 0.79 seconds

## BFS NS instance: NS_B2_A2_A1_A2_B2_B1

### Backsubstitution after applying NS history:
0: -41.3362274, 42.5331116, -52.5613327, 57.1557770, -98.4919968, 95.0944443
1: -279.1030579, 102.3374252, -388.3494568, 129.7206573, -408.8237305, 490.6868896
2: -160.6312256, 92.0329819, -207.6318054, 122.4168320, -283.0480347, 299.6647949
3: -196.5596924, 73.6476669, -267.8355713, 98.1381149, -294.6978149, 341.4832458
4: -117.3278351, 80.5484772, -148.7674866, 108.4405518, -225.7683716, 229.3159637

Time for backsubstitution: 1.10 seconds

### NS candidates at layer 1
type: A, layer: 1, pos: 4
type: A, layer: 1, pos: 3
type: B, layer: 1, pos: 35
type: A, layer: 1, pos: 19
type: A, layer: 1, pos: 9
type: A, layer: 1, pos: 35
type: B, layer: 1, pos: 9
type: A, layer: 1, pos: 38
type: A, layer: 1, pos: 17
type: B, layer: 1, pos: 3
type: B, layer: 1, pos: 15
type: B, layer: 1, pos: 19
type: B, layer: 1, pos: 10
type: A, layer: 1, pos: 18
type: A, layer: 1, pos: 45
type: B, layer: 1, pos: 4
type: A, layer: 1, pos: 43
type: B, layer: 1, pos: 43
type: B, layer: 1, pos: 17
type: B, layer: 1, pos: 29
type: B, layer: 1, pos: 32
type: A, layer: 1, pos: 11
type: A, layer: 1, pos: 37
type: A, layer: 1, pos: 10
type: A, layer: 1, pos: 21
type: B, layer: 1, pos: 16
type: B, layer: 1, pos: 18
type: B, layer: 1, pos: 45
type: A, layer: 1, pos: 15
type: B, layer: 1, pos: 21
type: B, layer: 1, pos: 20
type: B, layer: 1, pos: 23
type: B, layer: 1, pos: 39
type: A, layer: 1, pos: 23
type: A, layer: 1, pos: 41
type: A, layer: 1, pos: 32
type: B, layer: 1, pos: 41
type: A, layer: 1, pos: 39
type: A, layer: 1, pos: 5
type: A, layer: 1, pos: 20
type: B, layer: 1, pos: 5
type: B, layer: 1, pos: 38
type: A, layer: 1, pos: 16
type: B, layer: 1, pos: 37

Time for candidate selection: 0.09 seconds

### Candidate
type: A, layer: 1, pos: 4

### Candidate
type: A, layer: 1, pos: 3

### Candidate
type: B, layer: 1, pos: 35

### Candidate
type: A, layer: 1, pos: 19

### Candidate
type: A, layer: 1, pos: 9

### Candidate
type: A, layer: 1, pos: 35

### Candidate
type: B, layer: 1, pos: 9

### Candidate
type: A, layer: 1, pos: 38

### Candidate
type: A, layer: 1, pos: 17

### Candidate
type: B, layer: 1, pos: 3

### Candidate
type: B, layer: 1, pos: 15

### Candidate
type: B, layer: 1, pos: 19

### Candidate
type: B, layer: 1, pos: 10

### Candidate
type: A, layer: 1, pos: 18

### Candidate
type: A, layer: 1, pos: 45

### Candidate
type: B, layer: 1, pos: 4

### Candidate
type: A, layer: 1, pos: 43

## Relational analysis of NS_B2_A2_A1_A2_B2_B1_A1
Optimization infeasible because this subproblem isn't reachable.

### Candidate
type: B, layer: 1, pos: 43

### Candidate
type: B, layer: 1, pos: 17

### Candidate
type: B, layer: 1, pos: 29

## Relational analysis of NS_B2_A2_A1_A2_B2_B1_B1
Optimization infeasible because this subproblem isn't reachable.

### Candidate
type: B, layer: 1, pos: 32

## Relational analysis of NS_B2_A2_A1_A2_B2_B1_B1
Optimization infeasible because this subproblem isn't reachable.

### Candidate
type: A, layer: 1, pos: 11

## Relational analysis of NS_B2_A2_A1_A2_B2_B1_A1

### Relational analysis result of NS_B2_A2_A1_A2_B2_B1_A1
Status: Status.UNKNOWN
Output dim: 0, lower bound: -63.3931573, upper bound: 63.3932422
time: 0.80 seconds

## Relational analysis of NS_B2_A2_A1_A2_B2_B1_A2

### Relational analysis result of NS_B2_A2_A1_A2_B2_B1_A2
Status: Status.UNKNOWN
Output dim: 0, lower bound: -63.3931573, upper bound: 63.3941582
time: 0.74 seconds

## BFS NS instance: NS_B2_A2_A1_A2_B2_B2

### Backsubstitution after applying NS history:
0: -41.4766655, 42.6777725, -52.4143295, 56.9992371, -98.4758759, 95.0921021
1: -280.1462708, 102.6841431, -387.3854065, 129.3546753, -409.5008850, 490.0695496
2: -161.1961517, 92.3508987, -207.0046539, 122.0845871, -283.2807312, 299.3555298
3: -197.2792816, 73.9040070, -267.1123047, 97.8885498, -295.1678467, 341.0162964
4: -117.7232513, 80.8190765, -148.2979126, 108.1422577, -225.8655090, 229.1169891

Time for backsubstitution: 1.09 seconds

### NS candidates at layer 1
type: A, layer: 1, pos: 4
type: A, layer: 1, pos: 3
type: B, layer: 1, pos: 35
type: A, layer: 1, pos: 19
type: A, layer: 1, pos: 9
type: A, layer: 1, pos: 35
type: B, layer: 1, pos: 9
type: A, layer: 1, pos: 38
type: A, layer: 1, pos: 17
type: B, layer: 1, pos: 3
type: B, layer: 1, pos: 15
type: B, layer: 1, pos: 19
type: B, layer: 1, pos: 10
type: A, layer: 1, pos: 18
type: A, layer: 1, pos: 45
type: B, layer: 1, pos: 4
type: A, layer: 1, pos: 43
type: B, layer: 1, pos: 17
type: A, layer: 1, pos: 11
type: B, layer: 1, pos: 43
type: B, layer: 1, pos: 32
type: B, layer: 1, pos: 29
type: A, layer: 1, pos: 37
type: A, layer: 1, pos: 10
type: A, layer: 1, pos: 21
type: B, layer: 1, pos: 16
type: B, layer: 1, pos: 18
type: B, layer: 1, pos: 21
type: A, layer: 1, pos: 15
type: B, layer: 1, pos: 45
type: B, layer: 1, pos: 20
type: B, layer: 1, pos: 23
type: B, layer: 1, pos: 39
type: A, layer: 1, pos: 23
type: A, layer: 1, pos: 41
type: A, layer: 1, pos: 32
type: B, layer: 1, pos: 41
type: A, layer: 1, pos: 39
type: A, layer: 1, pos: 5
type: A, layer: 1, pos: 20
type: B, layer: 1, pos: 5
type: B, layer: 1, pos: 38
type: B, layer: 1, pos: 37
type: A, layer: 1, pos: 16

Time for candidate selection: 0.09 seconds

### Candidate
type: A, layer: 1, pos: 4

### Candidate
type: A, layer: 1, pos: 3

### Candidate
type: B, layer: 1, pos: 35

### Candidate
type: A, layer: 1, pos: 19

### Candidate
type: A, layer: 1, pos: 9

### Candidate
type: A, layer: 1, pos: 35

### Candidate
type: B, layer: 1, pos: 9

### Candidate
type: A, layer: 1, pos: 38

### Candidate
type: A, layer: 1, pos: 17

### Candidate
type: B, layer: 1, pos: 3

### Candidate
type: B, layer: 1, pos: 15

### Candidate
type: B, layer: 1, pos: 19

### Candidate
type: B, layer: 1, pos: 10

### Candidate
type: A, layer: 1, pos: 18

### Candidate
type: A, layer: 1, pos: 45

### Candidate
type: B, layer: 1, pos: 4

### Candidate
type: A, layer: 1, pos: 43

## Relational analysis of NS_B2_A2_A1_A2_B2_B2_A1
Optimization infeasible because this subproblem isn't reachable.

### Candidate
type: B, layer: 1, pos: 17

### Candidate
type: A, layer: 1, pos: 11

## Relational analysis of NS_B2_A2_A1_A2_B2_B2_A1

### Relational analysis result of NS_B2_A2_A1_A2_B2_B2_A1
Status: Status.UNKNOWN
Output dim: 0, lower bound: -63.3932285, upper bound: 63.3933819
time: 1.27 seconds

## Relational analysis of NS_B2_A2_A1_A2_B2_B2_A2

### Relational analysis result of NS_B2_A2_A1_A2_B2_B2_A2
Status: Status.UNKNOWN
Output dim: 0, lower bound: -63.3932285, upper bound: 63.3947862
time: 0.82 seconds

## BFS NS instance: NS_B2_A2_A2_B1_A1_B1

### Backsubstitution after applying NS history:
0: -54.1881714, 58.9965973, -53.9392204, 58.4476395, -112.6358109, 112.9358139
1: -402.2130432, 133.8377991, -396.4638672, 133.0583038, -534.5951538, 529.3087769
2: -214.3997955, 126.4415131, -212.5763245, 125.3313828, -339.7311707, 339.0178223
3: -277.1029663, 101.3699265, -273.6869812, 100.4134979, -376.8520813, 374.1130371
4: -153.4725342, 111.9196091, -152.8955994, 110.8564682, -264.3290100, 264.8152161

Time for backsubstitution: 1.09 seconds

### NS candidates at layer 1
type: A, layer: 1, pos: 35
type: B, layer: 1, pos: 35
type: A, layer: 1, pos: 9
type: B, layer: 1, pos: 9
type: B, layer: 1, pos: 19
type: A, layer: 1, pos: 19
type: B, layer: 1, pos: 3
type: A, layer: 1, pos: 3
type: B, layer: 1, pos: 17
type: A, layer: 1, pos: 17
type: A, layer: 1, pos: 10
type: B, layer: 1, pos: 18
type: B, layer: 1, pos: 45
type: B, layer: 1, pos: 10
type: A, layer: 1, pos: 15
type: B, layer: 1, pos: 43
type: A, layer: 1, pos: 18
type: B, layer: 1, pos: 15
type: A, layer: 1, pos: 29
type: A, layer: 1, pos: 43
type: A, layer: 1, pos: 45
type: B, layer: 1, pos: 29
type: A, layer: 1, pos: 21
type: A, layer: 1, pos: 23
type: B, layer: 1, pos: 21
type: B, layer: 1, pos: 4
type: B, layer: 1, pos: 37
type: A, layer: 1, pos: 32
type: B, layer: 1, pos: 23
type: B, layer: 1, pos: 38
type: A, layer: 1, pos: 39
type: B, layer: 1, pos: 20
type: B, layer: 1, pos: 32
type: A, layer: 1, pos: 37
type: A, layer: 1, pos: 41
type: B, layer: 1, pos: 41
type: A, layer: 1, pos: 20
type: B, layer: 1, pos: 39
type: A, layer: 1, pos: 16
type: B, layer: 1, pos: 5
type: A, layer: 1, pos: 5
type: B, layer: 1, pos: 16
type: A, layer: 1, pos: 4
type: A, layer: 1, pos: 38

Time for candidate selection: 0.09 seconds

### Candidate
type: A, layer: 1, pos: 35

### Candidate
type: B, layer: 1, pos: 35

### Candidate
type: A, layer: 1, pos: 9

### Candidate
type: B, layer: 1, pos: 9

### Candidate
type: B, layer: 1, pos: 19

### Candidate
type: A, layer: 1, pos: 19

### Candidate
type: B, layer: 1, pos: 3

### Candidate
type: A, layer: 1, pos: 3

### Candidate
type: B, layer: 1, pos: 17

### Candidate
type: A, layer: 1, pos: 17

### Candidate
type: A, layer: 1, pos: 10

### Candidate
type: B, layer: 1, pos: 18

### Candidate
type: B, layer: 1, pos: 45

### Candidate
type: B, layer: 1, pos: 10

### Candidate
type: A, layer: 1, pos: 15

### Candidate
type: B, layer: 1, pos: 43

### Candidate
type: A, layer: 1, pos: 18

### Candidate
type: B, layer: 1, pos: 15

### Candidate
type: A, layer: 1, pos: 29

### Candidate
type: A, layer: 1, pos: 43

### Candidate
type: A, layer: 1, pos: 45

### Candidate
type: B, layer: 1, pos: 29

### Candidate
type: A, layer: 1, pos: 21

## Relational analysis of NS_B2_A2_A2_B1_A1_B1_A1
Optimization infeasible because this subproblem isn't reachable.

### Candidate
type: A, layer: 1, pos: 23

## Relational analysis of NS_B2_A2_A2_B1_A1_B1_A1
Optimization infeasible because this subproblem isn't reachable.

### Candidate
type: B, layer: 1, pos: 21

### Candidate
type: B, layer: 1, pos: 4

### Candidate
type: B, layer: 1, pos: 37

## Relational analysis of NS_B2_A2_A2_B1_A1_B1_B1
Optimization infeasible because this subproblem isn't reachable.

### Candidate
type: A, layer: 1, pos: 32

## Relational analysis of NS_B2_A2_A2_B1_A1_B1_A1
Optimization infeasible because this subproblem isn't reachable.

### Candidate
type: B, layer: 1, pos: 23

### Candidate
type: B, layer: 1, pos: 38

## Relational analysis of NS_B2_A2_A2_B1_A1_B1_B1
Optimization infeasible because this subproblem isn't reachable.

### Candidate
type: A, layer: 1, pos: 39

## Relational analysis of NS_B2_A2_A2_B1_A1_B1_A1
Optimization infeasible because this subproblem isn't reachable.

### Candidate
type: B, layer: 1, pos: 20

## Relational analysis of NS_B2_A2_A2_B1_A1_B1_B1
Optimization infeasible because this subproblem isn't reachable.

### Candidate
type: B, layer: 1, pos: 32

### Candidate
type: A, layer: 1, pos: 37

### Candidate
type: A, layer: 1, pos: 41

## Relational analysis of NS_B2_A2_A2_B1_A1_B1_A1
Optimization infeasible because this subproblem isn't reachable.

### Candidate
type: B, layer: 1, pos: 41

### Candidate
type: A, layer: 1, pos: 20

### Candidate
type: B, layer: 1, pos: 39

### Candidate
type: A, layer: 1, pos: 16

## Relational analysis of NS_B2_A2_A2_B1_A1_B1_A1
Optimization infeasible because this subproblem isn't reachable.

### Candidate
type: B, layer: 1, pos: 5

## Relational analysis of NS_B2_A2_A2_B1_A1_B1_B1
Optimization infeasible because this subproblem isn't reachable.

### Candidate
type: A, layer: 1, pos: 5

### Candidate
type: B, layer: 1, pos: 16

### Candidate
type: A, layer: 1, pos: 4

### Candidate
type: A, layer: 1, pos: 38

## BFS NS instance: NS_B2_A2_A2_B1_A1_B2

### Backsubstitution after applying NS history:
0: -54.1881714, 58.9965973, -53.8370018, 58.3472595, -112.5354309, 112.8336029
1: -402.2130432, 133.8377991, -396.0016785, 132.7749634, -534.2969971, 528.8482056
2: -214.3997955, 126.4415131, -212.1299591, 125.1308899, -339.5307007, 338.5714722
3: -277.1029663, 101.3699265, -273.3017578, 100.2396317, -376.6761169, 373.7130737
4: -153.4725342, 111.9196091, -152.5545502, 110.6742859, -264.1468201, 264.4741211

Time for backsubstitution: 1.11 seconds

### NS candidates at layer 1
type: A, layer: 1, pos: 35
type: B, layer: 1, pos: 35
type: A, layer: 1, pos: 9
type: B, layer: 1, pos: 9
type: B, layer: 1, pos: 19
type: A, layer: 1, pos: 19
type: B, layer: 1, pos: 3
type: A, layer: 1, pos: 3
type: B, layer: 1, pos: 17
type: A, layer: 1, pos: 17
type: A, layer: 1, pos: 10
type: B, layer: 1, pos: 18
type: B, layer: 1, pos: 10
type: B, layer: 1, pos: 45
type: A, layer: 1, pos: 15
type: B, layer: 1, pos: 43
type: A, layer: 1, pos: 18
type: B, layer: 1, pos: 15
type: A, layer: 1, pos: 45
type: A, layer: 1, pos: 29
type: A, layer: 1, pos: 43
type: B, layer: 1, pos: 29
type: A, layer: 1, pos: 21
type: B, layer: 1, pos: 21
type: A, layer: 1, pos: 23
type: B, layer: 1, pos: 4
type: B, layer: 1, pos: 37
type: A, layer: 1, pos: 32
type: B, layer: 1, pos: 23
type: B, layer: 1, pos: 38
type: B, layer: 1, pos: 32
type: B, layer: 1, pos: 20
type: A, layer: 1, pos: 39
type: A, layer: 1, pos: 37
type: B, layer: 1, pos: 41
type: A, layer: 1, pos: 41
type: A, layer: 1, pos: 20
type: B, layer: 1, pos: 39
type: A, layer: 1, pos: 16
type: B, layer: 1, pos: 5
type: B, layer: 1, pos: 16
type: A, layer: 1, pos: 5
type: A, layer: 1, pos: 4
type: A, layer: 1, pos: 38

Time for candidate selection: 0.09 seconds

### Candidate
type: A, layer: 1, pos: 35

### Candidate
type: B, layer: 1, pos: 35

### Candidate
type: A, layer: 1, pos: 9

### Candidate
type: B, layer: 1, pos: 9

### Candidate
type: B, layer: 1, pos: 19

### Candidate
type: A, layer: 1, pos: 19

### Candidate
type: B, layer: 1, pos: 3

### Candidate
type: A, layer: 1, pos: 3

### Candidate
type: B, layer: 1, pos: 17

### Candidate
type: A, layer: 1, pos: 17

### Candidate
type: A, layer: 1, pos: 10

### Candidate
type: B, layer: 1, pos: 18

### Candidate
type: B, layer: 1, pos: 10

### Candidate
type: B, layer: 1, pos: 45

### Candidate
type: A, layer: 1, pos: 15

### Candidate
type: B, layer: 1, pos: 43

### Candidate
type: A, layer: 1, pos: 18

### Candidate
type: B, layer: 1, pos: 15

### Candidate
type: A, layer: 1, pos: 45

### Candidate
type: A, layer: 1, pos: 29

### Candidate
type: A, layer: 1, pos: 43

### Candidate
type: B, layer: 1, pos: 29

### Candidate
type: A, layer: 1, pos: 21

## Relational analysis of NS_B2_A2_A2_B1_A1_B2_A1
Optimization infeasible because this subproblem isn't reachable.

### Candidate
type: B, layer: 1, pos: 21

### Candidate
type: A, layer: 1, pos: 23

## Relational analysis of NS_B2_A2_A2_B1_A1_B2_A1
Optimization infeasible because this subproblem isn't reachable.

### Candidate
type: B, layer: 1, pos: 4

### Candidate
type: B, layer: 1, pos: 37

## Relational analysis of NS_B2_A2_A2_B1_A1_B2_B1
Optimization infeasible because this subproblem isn't reachable.

### Candidate
type: A, layer: 1, pos: 32

## Relational analysis of NS_B2_A2_A2_B1_A1_B2_A1
Optimization infeasible because this subproblem isn't reachable.

### Candidate
type: B, layer: 1, pos: 23

### Candidate
type: B, layer: 1, pos: 38

## Relational analysis of NS_B2_A2_A2_B1_A1_B2_B1
Optimization infeasible because this subproblem isn't reachable.

### Candidate
type: B, layer: 1, pos: 32

### Candidate
type: B, layer: 1, pos: 20

## Relational analysis of NS_B2_A2_A2_B1_A1_B2_B1
Optimization infeasible because this subproblem isn't reachable.

### Candidate
type: A, layer: 1, pos: 39

## Relational analysis of NS_B2_A2_A2_B1_A1_B2_A1
Optimization infeasible because this subproblem isn't reachable.

### Candidate
type: A, layer: 1, pos: 37

### Candidate
type: B, layer: 1, pos: 41

## Relational analysis of NS_B2_A2_A2_B1_A1_B2_B1
Optimization infeasible because this subproblem isn't reachable.

### Candidate
type: A, layer: 1, pos: 41

### Candidate
type: A, layer: 1, pos: 20

### Candidate
type: B, layer: 1, pos: 39

### Candidate
type: A, layer: 1, pos: 16

## Relational analysis of NS_B2_A2_A2_B1_A1_B2_A1
Optimization infeasible because this subproblem isn't reachable.

### Candidate
type: B, layer: 1, pos: 5

## Relational analysis of NS_B2_A2_A2_B1_A1_B2_B1
Optimization infeasible because this subproblem isn't reachable.

### Candidate
type: B, layer: 1, pos: 16

### Candidate
type: A, layer: 1, pos: 5

### Candidate
type: A, layer: 1, pos: 4

### Candidate
type: A, layer: 1, pos: 38

## BFS NS instance: NS_B2_A2_A2_B1_A2_B1

### Backsubstitution after applying NS history:
0: -53.8847351, 58.6575432, -53.9392204, 58.4476395, -112.3323593, 112.5967484
1: -399.7374878, 133.0946503, -396.4638672, 133.0583038, -532.1393433, 528.5483398
2: -213.1640167, 125.7145767, -212.5763245, 125.3313828, -338.4953613, 338.2908936
3: -275.4341431, 100.8090820, -273.6869812, 100.4134979, -375.1802063, 373.5529480
4: -152.5790253, 111.2873230, -152.8955994, 110.8564682, -263.4354858, 264.1829224

Time for backsubstitution: 1.09 seconds

### NS candidates at layer 1
type: A, layer: 1, pos: 35
type: B, layer: 1, pos: 35
type: A, layer: 1, pos: 9
type: B, layer: 1, pos: 9
type: B, layer: 1, pos: 19
type: A, layer: 1, pos: 19
type: B, layer: 1, pos: 3
type: A, layer: 1, pos: 3
type: B, layer: 1, pos: 17
type: A, layer: 1, pos: 17
type: A, layer: 1, pos: 10
type: B, layer: 1, pos: 45
type: B, layer: 1, pos: 18
type: B, layer: 1, pos: 10
type: B, layer: 1, pos: 43
type: A, layer: 1, pos: 15
type: B, layer: 1, pos: 15
type: A, layer: 1, pos: 18
type: A, layer: 1, pos: 29
type: A, layer: 1, pos: 43
type: A, layer: 1, pos: 45
type: B, layer: 1, pos: 29
type: A, layer: 1, pos: 21
type: A, layer: 1, pos: 23
type: B, layer: 1, pos: 21
type: A, layer: 1, pos: 32
type: B, layer: 1, pos: 37
type: B, layer: 1, pos: 23
type: B, layer: 1, pos: 4
type: B, layer: 1, pos: 38
type: A, layer: 1, pos: 39
type: B, layer: 1, pos: 32
type: B, layer: 1, pos: 20
type: A, layer: 1, pos: 37
type: A, layer: 1, pos: 41
type: B, layer: 1, pos: 41
type: A, layer: 1, pos: 20
type: B, layer: 1, pos: 39
type: A, layer: 1, pos: 16
type: B, layer: 1, pos: 5
type: A, layer: 1, pos: 4
type: A, layer: 1, pos: 5
type: B, layer: 1, pos: 16
type: A, layer: 1, pos: 38

Time for candidate selection: 0.09 seconds

### Candidate
type: A, layer: 1, pos: 35

### Candidate
type: B, layer: 1, pos: 35

### Candidate
type: A, layer: 1, pos: 9

### Candidate
type: B, layer: 1, pos: 9

### Candidate
type: B, layer: 1, pos: 19

### Candidate
type: A, layer: 1, pos: 19

### Candidate
type: B, layer: 1, pos: 3

### Candidate
type: A, layer: 1, pos: 3

### Candidate
type: B, layer: 1, pos: 17

### Candidate
type: A, layer: 1, pos: 17

### Candidate
type: A, layer: 1, pos: 10

### Candidate
type: B, layer: 1, pos: 45

### Candidate
type: B, layer: 1, pos: 18

### Candidate
type: B, layer: 1, pos: 10

### Candidate
type: B, layer: 1, pos: 43

### Candidate
type: A, layer: 1, pos: 15

### Candidate
type: B, layer: 1, pos: 15

### Candidate
type: A, layer: 1, pos: 18

### Candidate
type: A, layer: 1, pos: 29

## Relational analysis of NS_B2_A2_A2_B1_A2_B1_A1
Optimization infeasible because this subproblem isn't reachable.

### Candidate
type: A, layer: 1, pos: 43

### Candidate
type: A, layer: 1, pos: 45

### Candidate
type: B, layer: 1, pos: 29

### Candidate
type: A, layer: 1, pos: 21

## Relational analysis of NS_B2_A2_A2_B1_A2_B1_A1
Optimization infeasible because this subproblem isn't reachable.

### Candidate
type: A, layer: 1, pos: 23

## Relational analysis of NS_B2_A2_A2_B1_A2_B1_A1
Optimization infeasible because this subproblem isn't reachable.

### Candidate
type: B, layer: 1, pos: 21

### Candidate
type: A, layer: 1, pos: 32

## Relational analysis of NS_B2_A2_A2_B1_A2_B1_A1
Optimization infeasible because this subproblem isn't reachable.

### Candidate
type: B, layer: 1, pos: 37

## Relational analysis of NS_B2_A2_A2_B1_A2_B1_B1
Optimization infeasible because this subproblem isn't reachable.

### Candidate
type: B, layer: 1, pos: 23

### Candidate
type: B, layer: 1, pos: 4

### Candidate
type: B, layer: 1, pos: 38

## Relational analysis of NS_B2_A2_A2_B1_A2_B1_B1
Optimization infeasible because this subproblem isn't reachable.

### Candidate
type: A, layer: 1, pos: 39

## Relational analysis of NS_B2_A2_A2_B1_A2_B1_A1
Optimization infeasible because this subproblem isn't reachable.

### Candidate
type: B, layer: 1, pos: 32

### Candidate
type: B, layer: 1, pos: 20

## Relational analysis of NS_B2_A2_A2_B1_A2_B1_B1
Optimization infeasible because this subproblem isn't reachable.

### Candidate
type: A, layer: 1, pos: 37

### Candidate
type: A, layer: 1, pos: 41

## Relational analysis of NS_B2_A2_A2_B1_A2_B1_A1
Optimization infeasible because this subproblem isn't reachable.

### Candidate
type: B, layer: 1, pos: 41

### Candidate
type: A, layer: 1, pos: 20

### Candidate
type: B, layer: 1, pos: 39

### Candidate
type: A, layer: 1, pos: 16

## Relational analysis of NS_B2_A2_A2_B1_A2_B1_A1
Optimization infeasible because this subproblem isn't reachable.

### Candidate
type: B, layer: 1, pos: 5

## Relational analysis of NS_B2_A2_A2_B1_A2_B1_B1
Optimization infeasible because this subproblem isn't reachable.

### Candidate
type: A, layer: 1, pos: 4

### Candidate
type: A, layer: 1, pos: 5

### Candidate
type: B, layer: 1, pos: 16

### Candidate
type: A, layer: 1, pos: 38

## BFS NS instance: NS_B2_A2_A2_B1_A2_B2

### Backsubstitution after applying NS history:
0: -53.8847351, 58.6575432, -53.8370018, 58.3472595, -112.2319946, 112.4945450
1: -399.7374878, 133.0946503, -396.0016785, 132.7749634, -531.9356689, 528.1801147
2: -213.1640167, 125.7145767, -212.1299591, 125.1308899, -338.2949219, 337.8445435
3: -275.4341431, 100.8090820, -273.3017578, 100.2396317, -375.0663147, 373.2143250
4: -152.5790253, 111.2873230, -152.5545502, 110.6742859, -263.2532349, 263.8418274

Time for backsubstitution: 1.12 seconds

### NS candidates at layer 1
type: A, layer: 1, pos: 35
type: B, layer: 1, pos: 35
type: A, layer: 1, pos: 9
type: B, layer: 1, pos: 9
type: B, layer: 1, pos: 19
type: A, layer: 1, pos: 19
type: B, layer: 1, pos: 3
type: A, layer: 1, pos: 3
type: B, layer: 1, pos: 17
type: A, layer: 1, pos: 17
type: A, layer: 1, pos: 10
type: B, layer: 1, pos: 18
type: B, layer: 1, pos: 45
type: B, layer: 1, pos: 10
type: A, layer: 1, pos: 15
type: B, layer: 1, pos: 43
type: A, layer: 1, pos: 18
type: B, layer: 1, pos: 15
type: A, layer: 1, pos: 29
type: A, layer: 1, pos: 43
type: A, layer: 1, pos: 45
type: B, layer: 1, pos: 29
type: A, layer: 1, pos: 21
type: A, layer: 1, pos: 23
type: B, layer: 1, pos: 21
type: B, layer: 1, pos: 4
type: B, layer: 1, pos: 37
type: A, layer: 1, pos: 32
type: B, layer: 1, pos: 23
type: B, layer: 1, pos: 38
type: A, layer: 1, pos: 39
type: B, layer: 1, pos: 20
type: B, layer: 1, pos: 32
type: A, layer: 1, pos: 37
type: A, layer: 1, pos: 41
type: B, layer: 1, pos: 41
type: A, layer: 1, pos: 20
type: B, layer: 1, pos: 39
type: A, layer: 1, pos: 16
type: B, layer: 1, pos: 5
type: A, layer: 1, pos: 5
type: B, layer: 1, pos: 16
type: A, layer: 1, pos: 4
type: A, layer: 1, pos: 38

Time for candidate selection: 0.09 seconds

### Candidate
type: A, layer: 1, pos: 35

### Candidate
type: B, layer: 1, pos: 35

### Candidate
type: A, layer: 1, pos: 9

### Candidate
type: B, layer: 1, pos: 9

### Candidate
type: B, layer: 1, pos: 19

### Candidate
type: A, layer: 1, pos: 19

### Candidate
type: B, layer: 1, pos: 3

### Candidate
type: A, layer: 1, pos: 3

### Candidate
type: B, layer: 1, pos: 17

### Candidate
type: A, layer: 1, pos: 17

### Candidate
type: A, layer: 1, pos: 10

### Candidate
type: B, layer: 1, pos: 18

### Candidate
type: B, layer: 1, pos: 45

### Candidate
type: B, layer: 1, pos: 10

### Candidate
type: A, layer: 1, pos: 15

### Candidate
type: B, layer: 1, pos: 43

### Candidate
type: A, layer: 1, pos: 18

### Candidate
type: B, layer: 1, pos: 15

### Candidate
type: A, layer: 1, pos: 29

## Relational analysis of NS_B2_A2_A2_B1_A2_B2_A1
Optimization infeasible because this subproblem isn't reachable.

### Candidate
type: A, layer: 1, pos: 43

### Candidate
type: A, layer: 1, pos: 45

### Candidate
type: B, layer: 1, pos: 29

### Candidate
type: A, layer: 1, pos: 21

## Relational analysis of NS_B2_A2_A2_B1_A2_B2_A1
Optimization infeasible because this subproblem isn't reachable.

### Candidate
type: A, layer: 1, pos: 23

## Relational analysis of NS_B2_A2_A2_B1_A2_B2_A1
Optimization infeasible because this subproblem isn't reachable.

### Candidate
type: B, layer: 1, pos: 21

### Candidate
type: B, layer: 1, pos: 4

### Candidate
type: B, layer: 1, pos: 37

## Relational analysis of NS_B2_A2_A2_B1_A2_B2_B1
Optimization infeasible because this subproblem isn't reachable.

### Candidate
type: A, layer: 1, pos: 32

## Relational analysis of NS_B2_A2_A2_B1_A2_B2_A1
Optimization infeasible because this subproblem isn't reachable.

### Candidate
type: B, layer: 1, pos: 23

### Candidate
type: B, layer: 1, pos: 38

## Relational analysis of NS_B2_A2_A2_B1_A2_B2_B1
Optimization infeasible because this subproblem isn't reachable.

### Candidate
type: A, layer: 1, pos: 39

## Relational analysis of NS_B2_A2_A2_B1_A2_B2_A1
Optimization infeasible because this subproblem isn't reachable.

### Candidate
type: B, layer: 1, pos: 20

## Relational analysis of NS_B2_A2_A2_B1_A2_B2_B1
Optimization infeasible because this subproblem isn't reachable.

### Candidate
type: B, layer: 1, pos: 32

### Candidate
type: A, layer: 1, pos: 37

### Candidate
type: A, layer: 1, pos: 41

## Relational analysis of NS_B2_A2_A2_B1_A2_B2_A1
Optimization infeasible because this subproblem isn't reachable.

### Candidate
type: B, layer: 1, pos: 41

### Candidate
type: A, layer: 1, pos: 20

### Candidate
type: B, layer: 1, pos: 39

### Candidate
type: A, layer: 1, pos: 16

## Relational analysis of NS_B2_A2_A2_B1_A2_B2_A1
Optimization infeasible because this subproblem isn't reachable.

### Candidate
type: B, layer: 1, pos: 5

## Relational analysis of NS_B2_A2_A2_B1_A2_B2_B1
Optimization infeasible because this subproblem isn't reachable.

### Candidate
type: A, layer: 1, pos: 5

### Candidate
type: B, layer: 1, pos: 16

### Candidate
type: A, layer: 1, pos: 4

### Candidate
type: A, layer: 1, pos: 38

## BFS NS instance: NS_B2_A2_A2_B2_A1_A2

### Backsubstitution after applying NS history:
0: -54.0465584, 58.8513603, -53.9227829, 58.6793022, -112.7258606, 112.7741394
1: -401.3183899, 133.4969940, -399.8303223, 133.1675415, -534.4859619, 533.3273315
2: -213.8758240, 126.1286774, -213.2902069, 125.7758636, -339.6515808, 339.4188843
3: -276.4712830, 101.1208725, -275.5272217, 100.8443680, -377.3156433, 376.6481018
4: -153.0874481, 111.6378708, -152.6812134, 111.3355026, -264.4229431, 264.3190918

Time for backsubstitution: 1.12 seconds

### NS candidates at layer 1
type: B, layer: 1, pos: 35
type: A, layer: 1, pos: 35
type: B, layer: 1, pos: 9
type: A, layer: 1, pos: 9
type: B, layer: 1, pos: 19
type: A, layer: 1, pos: 19
type: A, layer: 1, pos: 3
type: B, layer: 1, pos: 3
type: B, layer: 1, pos: 17
type: A, layer: 1, pos: 17
type: B, layer: 1, pos: 15
type: A, layer: 1, pos: 10
type: B, layer: 1, pos: 10
type: A, layer: 1, pos: 45
type: B, layer: 1, pos: 18
type: B, layer: 1, pos: 43
type: A, layer: 1, pos: 18
type: A, layer: 1, pos: 43
type: B, layer: 1, pos: 29
type: A, layer: 1, pos: 29
type: B, layer: 1, pos: 45
type: B, layer: 1, pos: 11
type: A, layer: 1, pos: 21
type: B, layer: 1, pos: 21
type: A, layer: 1, pos: 23
type: A, layer: 1, pos: 32
type: B, layer: 1, pos: 23
type: B, layer: 1, pos: 37
type: A, layer: 1, pos: 37
type: B, layer: 1, pos: 20
type: B, layer: 1, pos: 32
type: A, layer: 1, pos: 39
type: A, layer: 1, pos: 20
type: B, layer: 1, pos: 38
type: B, layer: 1, pos: 41
type: A, layer: 1, pos: 41
type: B, layer: 1, pos: 39
type: B, layer: 1, pos: 16
type: A, layer: 1, pos: 38
type: A, layer: 1, pos: 5
type: A, layer: 1, pos: 16
type: B, layer: 1, pos: 5
type: A, layer: 1, pos: 4
type: B, layer: 1, pos: 4

Time for candidate selection: 0.09 seconds

### Candidate
type: B, layer: 1, pos: 35

### Candidate
type: A, layer: 1, pos: 35

### Candidate
type: B, layer: 1, pos: 9

### Candidate
type: A, layer: 1, pos: 9

### Candidate
type: B, layer: 1, pos: 19

### Candidate
type: A, layer: 1, pos: 19

### Candidate
type: A, layer: 1, pos: 3

### Candidate
type: B, layer: 1, pos: 3

### Candidate
type: B, layer: 1, pos: 17

### Candidate
type: A, layer: 1, pos: 17

### Candidate
type: B, layer: 1, pos: 15

## Relational analysis of NS_B2_A2_A2_B2_A1_A2_B1

### Relational analysis result of NS_B2_A2_A2_B2_A1_A2_B1
Status: Status.VERIFIED
Output dim: 0, lower bound: -63.3881947, upper bound: 63.3881910
time: 0.80 seconds

## Relational analysis of NS_B2_A2_A2_B2_A1_A2_B2

### Relational analysis result of NS_B2_A2_A2_B2_A1_A2_B2
Status: Status.UNKNOWN
Output dim: 0, lower bound: -63.3881947, upper bound: 63.3930663
time: 0.84 seconds

## BFS NS instance: NS_B2_A2_A2_B2_A2_B1

### Backsubstitution after applying NS history:
0: -53.8847351, 58.6575432, -54.1881714, 58.9965973, -112.8813248, 112.8457184
1: -399.7374878, 133.0946503, -402.2130432, 133.8377991, -533.5752563, 535.3076782
2: -213.1640167, 125.7145767, -214.3997955, 126.4415131, -339.6054993, 340.1143799
3: -275.4341431, 100.8090820, -277.1029663, 101.3699265, -376.8040466, 377.9120483
4: -152.5790253, 111.2873230, -153.4725342, 111.9196091, -264.4985962, 264.7598572

Time for backsubstitution: 1.14 seconds

### NS candidates at layer 1
type: A, layer: 1, pos: 35
type: B, layer: 1, pos: 35
type: B, layer: 1, pos: 9
type: A, layer: 1, pos: 9
type: A, layer: 1, pos: 19
type: B, layer: 1, pos: 19
type: B, layer: 1, pos: 3
type: A, layer: 1, pos: 3
type: A, layer: 1, pos: 17
type: B, layer: 1, pos: 17
type: A, layer: 1, pos: 10
type: B, layer: 1, pos: 10
type: B, layer: 1, pos: 18
type: A, layer: 1, pos: 18
type: B, layer: 1, pos: 45
type: B, layer: 1, pos: 15
type: A, layer: 1, pos: 43
type: A, layer: 1, pos: 15
type: B, layer: 1, pos: 43
type: A, layer: 1, pos: 45
type: B, layer: 1, pos: 29
type: A, layer: 1, pos: 29
type: B, layer: 1, pos: 21
type: A, layer: 1, pos: 21
type: A, layer: 1, pos: 23
type: B, layer: 1, pos: 23
type: A, layer: 1, pos: 32
type: A, layer: 1, pos: 37
type: B, layer: 1, pos: 32
type: B, layer: 1, pos: 37
type: A, layer: 1, pos: 20
type: A, layer: 1, pos: 39
type: B, layer: 1, pos: 20
type: A, layer: 1, pos: 41
type: B, layer: 1, pos: 39
type: B, layer: 1, pos: 41
type: B, layer: 1, pos: 38
type: A, layer: 1, pos: 38
type: A, layer: 1, pos: 16
type: B, layer: 1, pos: 16
type: B, layer: 1, pos: 5
type: A, layer: 1, pos: 5
type: B, layer: 1, pos: 4
type: A, layer: 1, pos: 4

Time for candidate selection: 0.11 seconds

### Candidate
type: A, layer: 1, pos: 35

### Candidate
type: B, layer: 1, pos: 35

### Candidate
type: B, layer: 1, pos: 9

### Candidate
type: A, layer: 1, pos: 9

### Candidate
type: A, layer: 1, pos: 19

### Candidate
type: B, layer: 1, pos: 19

### Candidate
type: B, layer: 1, pos: 3

### Candidate
type: A, layer: 1, pos: 3

### Candidate
type: A, layer: 1, pos: 17

### Candidate
type: B, layer: 1, pos: 17

### Candidate
type: A, layer: 1, pos: 10

### Candidate
type: B, layer: 1, pos: 10

### Candidate
type: B, layer: 1, pos: 18

### Candidate
type: A, layer: 1, pos: 18

### Candidate
type: B, layer: 1, pos: 45

## Relational analysis of NS_B2_A2_A2_B2_A2_B1_B1
Optimization infeasible because this subproblem isn't reachable.

### Candidate
type: B, layer: 1, pos: 15

## Relational analysis of NS_B2_A2_A2_B2_A2_B1_B1
Optimization infeasible because this subproblem isn't reachable.

### Candidate
type: A, layer: 1, pos: 43

## Relational analysis of NS_B2_A2_A2_B2_A2_B1_A1
Optimization infeasible because this subproblem isn't reachable.

### Candidate
type: A, layer: 1, pos: 15

### Candidate
type: B, layer: 1, pos: 43

### Candidate
type: A, layer: 1, pos: 45

### Candidate
type: B, layer: 1, pos: 29

## Relational analysis of NS_B2_A2_A2_B2_A2_B1_B1
Optimization infeasible because this subproblem isn't reachable.

### Candidate
type: A, layer: 1, pos: 29

### Candidate
type: B, layer: 1, pos: 21

## Relational analysis of NS_B2_A2_A2_B2_A2_B1_B1
Optimization infeasible because this subproblem isn't reachable.

### Candidate
type: A, layer: 1, pos: 21

### Candidate
type: A, layer: 1, pos: 23

## Relational analysis of NS_B2_A2_A2_B2_A2_B1_A1
Optimization infeasible because this subproblem isn't reachable.

### Candidate
type: B, layer: 1, pos: 23

### Candidate
type: A, layer: 1, pos: 32

## Relational analysis of NS_B2_A2_A2_B2_A2_B1_A1
Optimization infeasible because this subproblem isn't reachable.

### Candidate
type: A, layer: 1, pos: 37

## Relational analysis of NS_B2_A2_A2_B2_A2_B1_A1
Optimization infeasible because this subproblem isn't reachable.

### Candidate
type: B, layer: 1, pos: 32

### Candidate
type: B, layer: 1, pos: 37

### Candidate
type: A, layer: 1, pos: 20

## Relational analysis of NS_B2_A2_A2_B2_A2_B1_A1
Optimization infeasible because this subproblem isn't reachable.

### Candidate
type: A, layer: 1, pos: 39

## Relational analysis of NS_B2_A2_A2_B2_A2_B1_A1
Optimization infeasible because this subproblem isn't reachable.

### Candidate
type: B, layer: 1, pos: 20

### Candidate
type: A, layer: 1, pos: 41

## Relational analysis of NS_B2_A2_A2_B2_A2_B1_A1
Optimization infeasible because this subproblem isn't reachable.

## NS Result
status: Status.UNKNOWN
execution time: (base) + (ns) = 2.98 + 417.21 = 420.18 seconds
