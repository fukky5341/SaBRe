## Execution arguments:
Dataset: Dataset.ACAS
Network: ds/onnx/acasxu_op11/ACASXU_1_9.onnx
Relational property: GLOBAL_ROBUSTNESS
LP Analysis: True
Epsilon: None
Delta epsilon: 0.1
execution index: (10, None, 0)
Time budget: 420 seconds
Split limit: 100
Threshold: 63.39367995097201


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
execution time: IAR + RelationalAnalysis = 1.24 + 2.02 = 3.26 seconds
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

Time for candidate selection: 0.13 seconds

### Candidate
type: B, layer: 1, pos: 40

## Relational analysis of NS_B1

### Relational analysis result of NS_B1
Status: Status.UNKNOWN
Output dim: 0, lower bound: -63.3963488, upper bound: 63.3961981
time: 0.71 seconds

## Relational analysis of NS_B2

### Relational analysis result of NS_B2
Status: Status.UNKNOWN
Output dim: 0, lower bound: -63.3958632, upper bound: 63.3958632
time: 0.73 seconds

## Summary of splitting at layer (split count: 0)
- Time for NS candidates: 1.59 seconds
NS_B1, status: Status.UNKNOWN, split count: 1, time: 1.59
Output dim: 0, lower bound: -63.3963488, upper bound: 63.3961981
NS_B2, status: Status.UNKNOWN, split count: 1, time: 1.59
Output dim: 0, lower bound: -63.3958632, upper bound: 63.3958632

## BFS NS instance: NS_B1

### Backsubstitution after applying NS history:
0: -36.6631546, 38.4677963, -35.6584015, 37.4214935, -74.0846481, 74.1261902
1: -260.1878967, 90.8589401, -253.4253845, 88.3672638, -348.5551758, 344.2843018
2: -145.2130890, 83.1253891, -141.3148041, 80.8721390, -226.0852356, 224.4401855
3: -181.7225800, 66.7120132, -176.9852448, 64.8847427, -246.6073303, 243.6972656
4: -104.4755707, 72.6576462, -101.6394196, 70.6504669, -175.1260223, 174.2970581

Time for backsubstitution: 1.11 seconds

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

Time for candidate selection: 0.08 seconds

### Candidate
type: B, layer: 1, pos: 7

## Relational analysis of NS_B1_B1

### Relational analysis result of NS_B1_B1
Status: Status.UNKNOWN
Output dim: 0, lower bound: -63.3946059, upper bound: 63.3945244
time: 0.72 seconds

## Relational analysis of NS_B1_B2

### Relational analysis result of NS_B1_B2
Status: Status.UNKNOWN
Output dim: 0, lower bound: -63.3954223, upper bound: 63.3952755
time: 0.96 seconds

## BFS NS instance: NS_B2

### Backsubstitution after applying NS history:
0: -37.2165909, 39.2005997, -56.4413948, 61.1914101, -98.4080048, 95.6419983
1: -265.3079529, 92.1755524, -416.5941772, 139.3072815, -404.6152344, 508.7697144
2: -147.3638763, 84.6269760, -222.9568634, 131.2333984, -278.5972290, 307.5838318
3: -185.0496063, 67.9713440, -287.4306335, 105.1853409, -290.2349548, 355.4019775
4: -105.9015121, 74.0802994, -159.8457642, 115.9607849, -221.8623047, 233.9260406

Time for backsubstitution: 1.11 seconds

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

Time for candidate selection: 0.08 seconds

### Candidate
type: A, layer: 1, pos: 7

## Relational analysis of NS_B2_A1

### Relational analysis result of NS_B2_A1
Status: Status.UNKNOWN
Output dim: 0, lower bound: -63.3940963, upper bound: 63.3937680
time: 0.72 seconds

## Relational analysis of NS_B2_A2

### Relational analysis result of NS_B2_A2
Status: Status.UNKNOWN
Output dim: 0, lower bound: -63.3949094, upper bound: 63.3949094
time: 0.76 seconds

## Summary of splitting at layer (split count: 1)
- Time for NS candidates: 2.70 seconds
NS_B1_B1, status: Status.UNKNOWN, split count: 2, time: 2.70
Output dim: 0, lower bound: -63.3946059, upper bound: 63.3945244
NS_B1_B2, status: Status.UNKNOWN, split count: 2, time: 2.70
Output dim: 0, lower bound: -63.3954223, upper bound: 63.3952755
NS_B2_A1, status: Status.UNKNOWN, split count: 2, time: 2.70
Output dim: 0, lower bound: -63.3940963, upper bound: 63.3937680
NS_B2_A2, status: Status.UNKNOWN, split count: 2, time: 2.70
Output dim: 0, lower bound: -63.3949094, upper bound: 63.3949094

## BFS NS instance: NS_B1_B1

### Backsubstitution after applying NS history:
0: -36.0036278, 37.7448730, -33.5843048, 35.1677933, -71.1714172, 71.3291779
1: -255.4349365, 89.2145386, -238.2765503, 83.1981659, -338.6331177, 327.4910583
2: -142.6184845, 81.5017166, -133.0587158, 75.9643021, -218.5827637, 214.5604095
3: -178.4406433, 65.4211426, -166.4944458, 60.9163361, -239.3569794, 231.9155884
4: -102.6114731, 71.2552490, -95.7579269, 66.2982178, -168.9096985, 167.0131683

Time for backsubstitution: 1.06 seconds

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

Time for candidate selection: 0.08 seconds

### Candidate
type: A, layer: 1, pos: 40

## Relational analysis of NS_B1_B1_A1

### Relational analysis result of NS_B1_B1_A1
Status: Status.UNKNOWN
Output dim: 0, lower bound: -63.3946059, upper bound: 63.3945244
time: 0.73 seconds

## Relational analysis of NS_B1_B1_A2

### Relational analysis result of NS_B1_B1_A2
Status: Status.UNKNOWN
Output dim: 0, lower bound: -63.3946059, upper bound: 63.3945244
time: 0.74 seconds

## BFS NS instance: NS_B1_B2

### Backsubstitution after applying NS history:
0: -34.9757843, 36.8191261, -33.1110268, 34.9088249, -69.8845978, 69.9301529
1: -249.2049713, 86.5966187, -236.7059174, 81.9441833, -331.1491699, 323.3025513
2: -138.3972015, 79.5439301, -130.8000793, 75.3684692, -213.7656708, 210.3440094
3: -173.4491882, 63.7782097, -164.3968353, 60.3954544, -233.8446198, 228.1750488
4: -99.5938263, 69.6211548, -94.1242828, 65.9827042, -165.5765381, 163.7454376

Time for backsubstitution: 1.33 seconds

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

Time for candidate selection: 0.10 seconds

### Candidate
type: A, layer: 1, pos: 40

## Relational analysis of NS_B1_B2_A1

### Relational analysis result of NS_B1_B2_A1
Status: Status.UNKNOWN
Output dim: 0, lower bound: -63.3954223, upper bound: 63.3952755
time: 0.75 seconds

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

Time for backsubstitution: 1.11 seconds

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

Time for candidate selection: 0.09 seconds

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
time: 0.74 seconds

## Relational analysis of NS_B2_A1_A2

### Relational analysis result of NS_B2_A1_A2
Status: Status.UNKNOWN
Output dim: 0, lower bound: -63.3940963, upper bound: 63.3937680
time: 0.69 seconds

## BFS NS instance: NS_B2_A2

### Backsubstitution after applying NS history:
0: -34.8330917, 36.7908058, -54.8922081, 59.6490822, -94.4821625, 91.6830139
1: -248.2947388, 86.1418076, -406.4594421, 135.5396118, -383.8343506, 492.6012573
2: -137.5112610, 79.3509903, -217.0846710, 127.8905411, -265.4017639, 296.4356689
3: -172.5016022, 63.6426582, -280.2118225, 102.5400391, -275.0416260, 343.8544922
4: -99.0283432, 69.6066818, -155.4643555, 113.1344757, -212.1627960, 225.0710297

Time for backsubstitution: 1.09 seconds

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

Time for candidate selection: 0.08 seconds

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
time: 0.89 seconds

## Relational analysis of NS_B2_A2_A2

### Relational analysis result of NS_B2_A2_A2
Status: Status.UNKNOWN
Output dim: 0, lower bound: -63.3949094, upper bound: 63.3949094
time: 1.12 seconds

## Summary of splitting at layer (split count: 2)
- Time for NS candidates: 3.45 seconds
NS_B1_B1_A1, status: Status.UNKNOWN, split count: 3, time: 3.45
Output dim: 0, lower bound: -63.3946059, upper bound: 63.3945244
NS_B1_B1_A2, status: Status.UNKNOWN, split count: 3, time: 3.45
Output dim: 0, lower bound: -63.3946059, upper bound: 63.3945244
NS_B1_B2_A1, status: Status.UNKNOWN, split count: 3, time: 3.45
Output dim: 0, lower bound: -63.3954223, upper bound: 63.3952755
NS_B1_B2_A2, status: Status.UNKNOWN, split count: 3, time: 3.45
Output dim: 0, lower bound: -63.3954223, upper bound: 63.3952755
NS_B2_A1_A1, status: Status.UNKNOWN, split count: 3, time: 3.45
Output dim: 0, lower bound: -63.3940963, upper bound: 63.3937680
NS_B2_A1_A2, status: Status.UNKNOWN, split count: 3, time: 3.45
Output dim: 0, lower bound: -63.3940963, upper bound: 63.3937680
NS_B2_A2_A1, status: Status.UNKNOWN, split count: 3, time: 3.45
Output dim: 0, lower bound: -63.3949094, upper bound: 63.3949094
NS_B2_A2_A2, status: Status.UNKNOWN, split count: 3, time: 3.45
Output dim: 0, lower bound: -63.3949094, upper bound: 63.3949094

## BFS NS instance: NS_B1_B1_A1

### Backsubstitution after applying NS history:
0: -34.9201393, 36.6235657, -33.5843048, 35.1677933, -70.0879211, 70.2078705
1: -248.0101776, 86.5205154, -238.2765503, 83.1981659, -331.2083130, 324.7969666
2: -138.3804932, 79.0562668, -133.0587158, 75.9643021, -214.3447876, 212.1149902
3: -173.2475739, 63.4533463, -166.4944458, 60.9163361, -234.1639099, 229.9477844
4: -99.5461960, 69.1046753, -95.7579269, 66.2982178, -165.8443909, 164.8625641

Time for backsubstitution: 1.19 seconds

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

Time for candidate selection: 0.10 seconds

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
time: 0.69 seconds

## BFS NS instance: NS_B1_B1_A2

### Backsubstitution after applying NS history:
0: -55.2682686, 59.9139137, -33.5843048, 35.1677933, -90.4360657, 93.4982147
1: -407.2115173, 136.3405151, -238.2765503, 83.1981659, -489.5598145, 374.6169739
2: -218.0220642, 128.4799652, -133.0587158, 75.9643021, -293.9863586, 261.5386047
3: -281.0300293, 102.9374008, -166.4944458, 60.9163361, -341.8305969, 269.4318237
4: -156.6577911, 113.5915985, -95.7579269, 66.2982178, -222.9560089, 209.3495026

Time for backsubstitution: 1.22 seconds

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

Time for candidate selection: 0.10 seconds

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
time: 0.69 seconds

## Relational analysis of NS_B1_B1_A2_B2

### Relational analysis result of NS_B1_B1_A2_B2
Status: Status.UNKNOWN
Output dim: 0, lower bound: -63.3943431, upper bound: 63.3944609
time: 0.72 seconds

## BFS NS instance: NS_B1_B2_A1

### Backsubstitution after applying NS history:
0: -33.9761581, 35.7755737, -33.1110268, 34.9088249, -68.8849792, 68.8865967
1: -242.5032349, 84.1374435, -236.7059174, 81.9441833, -324.4474182, 320.8433533
2: -134.5047455, 77.2907333, -130.8000793, 75.3684692, -209.8732147, 208.0908203
3: -168.7405090, 61.9489403, -164.3968353, 60.3954544, -229.1359558, 226.3457794
4: -96.7570953, 67.6140366, -94.1242828, 65.9827042, -162.7398071, 161.7383118

Time for backsubstitution: 1.23 seconds

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
time: 0.79 seconds

## Relational analysis of NS_B1_B2_A1_A2

### Relational analysis result of NS_B1_B2_A1_A2
Status: Status.UNKNOWN
Output dim: 0, lower bound: -63.3945203, upper bound: 63.3952755
time: 0.73 seconds

## BFS NS instance: NS_B1_B2_A2

### Backsubstitution after applying NS history:
0: -54.8922081, 59.6490822, -33.1110268, 34.9088249, -89.8010330, 92.7601013
1: -406.4594421, 135.5396118, -236.7059174, 81.9441833, -488.4036255, 372.2455444
2: -217.0846710, 127.8905411, -130.8000793, 75.3684692, -292.4531250, 258.6906128
3: -280.2118225, 102.5400391, -164.3968353, 60.3954544, -340.6072693, 266.9368286
4: -155.4643555, 113.1344757, -94.1242828, 65.9827042, -221.4470520, 207.2587433

Time for backsubstitution: 1.10 seconds

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

Time for candidate selection: 0.08 seconds

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
time: 0.82 seconds

## Relational analysis of NS_B1_B2_A2_B2

### Relational analysis result of NS_B1_B2_A2_B2
Status: Status.UNKNOWN
Output dim: 0, lower bound: -63.3941823, upper bound: 63.3942861
time: 0.83 seconds

## BFS NS instance: NS_B2_A1_A1

### Backsubstitution after applying NS history:
0: -33.5843048, 35.1677933, -55.2682686, 59.9139137, -93.4982147, 90.4360657
1: -238.2765503, 83.1981659, -407.2115173, 136.3405151, -374.6169739, 489.5598145
2: -133.0587158, 75.9643021, -218.0220642, 128.4799652, -261.5386047, 293.9863281
3: -166.4944458, 60.9163361, -281.0300293, 102.9374008, -269.4317932, 341.8305969
4: -95.7579269, 66.2982178, -156.6577911, 113.5915985, -209.3495178, 222.9560089

Time for backsubstitution: 1.17 seconds

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

Time for candidate selection: 0.09 seconds

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
Status: Status.VERIFIED
Output dim: 0, lower bound: -63.3933976, upper bound: 63.3933976
time: 0.72 seconds

## Relational analysis of NS_B2_A1_A1_B2

### Relational analysis result of NS_B2_A1_A1_B2
Status: Status.VERIFIED
Output dim: 0, lower bound: -63.3933976, upper bound: 63.3933976
time: 0.99 seconds

## BFS NS instance: NS_B2_A1_A2

### Backsubstitution after applying NS history:
0: -54.0104523, 58.5131683, -55.2682686, 59.9139137, -113.9243546, 113.7814178
1: -397.0915527, 133.2104645, -407.2115173, 136.3405151, -532.4822998, 539.5666504
2: -212.8295288, 125.4976883, -218.0220642, 128.4799652, -341.3094482, 343.5197144
3: -274.0873108, 100.5429916, -281.0300293, 102.9374008, -376.1135559, 380.7726746
4: -153.0611725, 110.9844284, -156.6577911, 113.5915985, -266.6527710, 267.6422119

Time for backsubstitution: 1.01 seconds

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

Time for candidate selection: 0.08 seconds

### Candidate
type: B, layer: 1, pos: 7

## Relational analysis of NS_B2_A1_A2_B1

### Relational analysis result of NS_B2_A1_A2_B1
Status: Status.VERIFIED
Output dim: 0, lower bound: -63.3933976, upper bound: 63.3933976
time: 0.72 seconds

## Relational analysis of NS_B2_A1_A2_B2

### Relational analysis result of NS_B2_A1_A2_B2
Status: Status.VERIFIED
Output dim: 0, lower bound: -63.3933976, upper bound: 63.3933976
time: 0.70 seconds

## BFS NS instance: NS_B2_A2_A1

### Backsubstitution after applying NS history:
0: -33.1092682, 34.9069939, -54.8922081, 59.6490822, -92.7583313, 89.7992020
1: -236.6929016, 81.9397430, -406.4594421, 135.5396118, -372.2325134, 488.3991699
2: -130.7926941, 75.3644485, -217.0846710, 127.8905411, -258.6832275, 292.4490967
3: -164.3877411, 60.3922920, -280.2118225, 102.5400391, -266.9277344, 340.6041260
4: -94.1191254, 65.9793015, -155.4643555, 113.1344757, -207.2535858, 221.4436340

Time for backsubstitution: 1.11 seconds

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

Time for candidate selection: 0.10 seconds

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
time: 0.68 seconds

## Relational analysis of NS_B2_A2_A1_A2

### Relational analysis result of NS_B2_A2_A1_A2
Status: Status.UNKNOWN
Output dim: 0, lower bound: -63.3947778, upper bound: 63.3947778
time: 0.78 seconds

## BFS NS instance: NS_B2_A2_A2

### Backsubstitution after applying NS history:
0: -54.0260849, 58.7897339, -54.8922081, 59.6490822, -113.6751251, 113.6819458
1: -400.6096802, 133.4270172, -406.4594421, 135.5396118, -536.1492920, 539.8864746
2: -213.7019806, 126.0158386, -217.0846710, 127.8905411, -341.5924683, 343.1005249
3: -276.0601196, 101.0406876, -280.2118225, 102.5400391, -378.6001587, 381.2525024
4: -152.9686127, 111.5431519, -155.4643555, 113.1344757, -266.1030579, 267.0074768

Time for backsubstitution: 1.18 seconds

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

Time for candidate selection: 0.09 seconds

### Candidate
type: B, layer: 1, pos: 7

## Relational analysis of NS_B2_A2_A2_B1

### Relational analysis result of NS_B2_A2_A2_B1
Status: Status.UNKNOWN
Output dim: 0, lower bound: -63.3937680, upper bound: 63.3940963
time: 0.75 seconds

## Relational analysis of NS_B2_A2_A2_B2

### Relational analysis result of NS_B2_A2_A2_B2
Status: Status.UNKNOWN
Output dim: 0, lower bound: -63.3937680, upper bound: 63.3949094
time: 0.79 seconds

## Summary of splitting at layer (split count: 3)
- Time for NS candidates: 2.82 seconds
NS_B1_B1_A1_A1, status: Status.UNKNOWN, split count: 4, time: 2.82
Output dim: 0, lower bound: -63.3945363, upper bound: 63.3943452
NS_B1_B1_A1_A2, status: Status.UNKNOWN, split count: 4, time: 2.82
Output dim: 0, lower bound: -63.3945363, upper bound: 63.3945244
NS_B1_B1_A2_B1, status: Status.UNKNOWN, split count: 4, time: 2.82
Output dim: 0, lower bound: -63.3943650, upper bound: 63.3944516
NS_B1_B1_A2_B2, status: Status.UNKNOWN, split count: 4, time: 2.82
Output dim: 0, lower bound: -63.3943431, upper bound: 63.3944609
NS_B1_B2_A1_A1, status: Status.UNKNOWN, split count: 4, time: 2.82
Output dim: 0, lower bound: -63.3945203, upper bound: 63.3943221
NS_B1_B2_A1_A2, status: Status.UNKNOWN, split count: 4, time: 2.82
Output dim: 0, lower bound: -63.3945203, upper bound: 63.3952755
NS_B1_B2_A2_B1, status: Status.UNKNOWN, split count: 4, time: 2.82
Output dim: 0, lower bound: -63.3940130, upper bound: 63.3942140
NS_B1_B2_A2_B2, status: Status.UNKNOWN, split count: 4, time: 2.82
Output dim: 0, lower bound: -63.3941823, upper bound: 63.3942861
NS_B2_A1_A1_B1, status: Status.VERIFIED, split count: 4, time: 2.82
Output dim: 0, lower bound: -63.3933976, upper bound: 63.3933976
NS_B2_A1_A1_B2, status: Status.VERIFIED, split count: 4, time: 2.82
Output dim: 0, lower bound: -63.3933976, upper bound: 63.3933976
NS_B2_A1_A2_B1, status: Status.VERIFIED, split count: 4, time: 2.82
Output dim: 0, lower bound: -63.3933976, upper bound: 63.3933976
NS_B2_A1_A2_B2, status: Status.VERIFIED, split count: 4, time: 2.82
Output dim: 0, lower bound: -63.3933976, upper bound: 63.3933976
NS_B2_A2_A1_A1, status: Status.UNKNOWN, split count: 4, time: 2.82
Output dim: 0, lower bound: -63.3946560, upper bound: 63.3947170
NS_B2_A2_A1_A2, status: Status.UNKNOWN, split count: 4, time: 2.82
Output dim: 0, lower bound: -63.3947778, upper bound: 63.3947778
NS_B2_A2_A2_B1, status: Status.UNKNOWN, split count: 4, time: 2.82
Output dim: 0, lower bound: -63.3937680, upper bound: 63.3940963
NS_B2_A2_A2_B2, status: Status.UNKNOWN, split count: 4, time: 2.82
Output dim: 0, lower bound: -63.3937680, upper bound: 63.3949094

## BFS NS instance: NS_B1_B1_A1_A1

### Backsubstitution after applying NS history:
0: -33.5843048, 35.1677933, -33.5843048, 35.1677933, -68.7520981, 68.7520981
1: -238.2765503, 83.1981659, -238.2765503, 83.1981659, -321.4746399, 321.4746399
2: -133.0587158, 75.9643021, -133.0587158, 75.9643021, -209.0229950, 209.0229950
3: -166.4944458, 60.9163361, -166.4944458, 60.9163361, -227.4107819, 227.4107819
4: -95.7579269, 66.2982178, -95.7579269, 66.2982178, -162.0561371, 162.0561218

Time for backsubstitution: 1.23 seconds

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

Time for candidate selection: 0.10 seconds

### Candidate
type: B, layer: 1, pos: 4

## Relational analysis of NS_B1_B1_A1_A1_B1

### Relational analysis result of NS_B1_B1_A1_A1_B1
Status: Status.UNKNOWN
Output dim: 0, lower bound: -63.3958876, upper bound: 63.3958122
time: 0.82 seconds

## Relational analysis of NS_B1_B1_A1_A1_B2

### Relational analysis result of NS_B1_B1_A1_A1_B2
Status: Status.UNKNOWN
Output dim: 0, lower bound: -63.3958450, upper bound: 63.3957590
time: 0.73 seconds

## BFS NS instance: NS_B1_B1_A1_A2

### Backsubstitution after applying NS history:
0: -33.1110268, 34.9088249, -33.5843048, 35.1677933, -68.2788239, 68.4931335
1: -236.7059174, 81.9441833, -238.2765503, 83.1981659, -319.9040527, 320.2206726
2: -130.8000793, 75.3684692, -133.0587158, 75.9643021, -206.7643738, 208.4271851
3: -164.3968353, 60.3954544, -166.4944458, 60.9163361, -225.3131714, 226.8898926
4: -94.1242828, 65.9827042, -95.7579269, 66.2982178, -160.4224854, 161.7406311

Time for backsubstitution: 1.11 seconds

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

Time for candidate selection: 0.10 seconds

### Candidate
type: B, layer: 1, pos: 4

## Relational analysis of NS_B1_B1_A1_A2_B1

### Relational analysis result of NS_B1_B1_A1_A2_B1
Status: Status.UNKNOWN
Output dim: 0, lower bound: -63.3958876, upper bound: 63.3958122
time: 1.08 seconds

## Relational analysis of NS_B1_B1_A1_A2_B2

### Relational analysis result of NS_B1_B1_A1_A2_B2
Status: Status.UNKNOWN
Output dim: 0, lower bound: -63.3958450, upper bound: 63.3957590
time: 0.69 seconds

## BFS NS instance: NS_B1_B1_A2_B1

### Backsubstitution after applying NS history:
0: -54.7686424, 59.4249878, -32.7632294, 34.3331223, -89.1017609, 92.1882172
1: -404.0156250, 135.1101379, -232.5169525, 81.1931534, -484.2805481, 367.6269531
2: -216.1138153, 127.4112854, -129.8301544, 74.1390152, -290.2528381, 257.2414551
3: -278.7328796, 102.0758667, -162.4364471, 59.4611740, -338.0152283, 264.5122986
4: -155.2354126, 112.6637726, -93.4227905, 64.7076111, -219.9430237, 206.0865631

Time for backsubstitution: 1.23 seconds

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

Time for candidate selection: 0.11 seconds

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
time: 0.71 seconds

## Relational analysis of NS_B1_B1_A2_B1_B2

### Relational analysis result of NS_B1_B1_A2_B1_B2
Status: Status.UNKNOWN
Output dim: 0, lower bound: -63.3941184, upper bound: 63.3941425
time: 0.71 seconds

## BFS NS instance: NS_B1_B1_A2_B2

### Backsubstitution after applying NS history:
0: -54.7002220, 59.3067627, -35.4041939, 37.2144241, -91.9146423, 94.7109375
1: -403.1087341, 134.9294434, -253.3442535, 87.7649612, -489.9754333, 388.2736206
2: -215.7784271, 127.1595306, -140.9738159, 80.3180618, -296.0964966, 268.1333008
3: -278.2004395, 101.8846817, -176.9305115, 64.4613800, -342.4980164, 278.8151550
4: -155.0654449, 112.4233398, -101.1564407, 70.0503616, -225.1158142, 213.5797729

Time for backsubstitution: 1.12 seconds

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

Time for candidate selection: 0.09 seconds

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
time: 0.88 seconds

## Relational analysis of NS_B1_B1_A2_B2_B2

### Relational analysis result of NS_B1_B1_A2_B2_B2
Status: Status.UNKNOWN
Output dim: 0, lower bound: -63.3942638, upper bound: 63.3943223
time: 1.23 seconds

## BFS NS instance: NS_B1_B2_A1_A1

### Backsubstitution after applying NS history:
0: -33.5843048, 35.1677933, -33.1110268, 34.9088249, -68.4931335, 68.2788239
1: -238.2765503, 83.1981659, -236.7059174, 81.9441833, -320.2206726, 319.9040527
2: -133.0587158, 75.9643021, -130.8000793, 75.3684692, -208.4271851, 206.7643738
3: -166.4944458, 60.9163361, -164.3968353, 60.3954544, -226.8898926, 225.3131714
4: -95.7579269, 66.2982178, -94.1242828, 65.9827042, -161.7406158, 160.4224854

Time for backsubstitution: 1.17 seconds

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

Time for candidate selection: 0.09 seconds

### Candidate
type: A, layer: 1, pos: 4

## Relational analysis of NS_B1_B2_A1_A1_A1

### Relational analysis result of NS_B1_B2_A1_A1_A1
Status: Status.UNKNOWN
Output dim: 0, lower bound: -63.3957216, upper bound: 63.3958435
time: 0.66 seconds

## Relational analysis of NS_B1_B2_A1_A1_A2

### Relational analysis result of NS_B1_B2_A1_A1_A2
Status: Status.UNKNOWN
Output dim: 0, lower bound: -63.3954598, upper bound: 63.3954598
time: 0.82 seconds

## BFS NS instance: NS_B1_B2_A1_A2

### Backsubstitution after applying NS history:
0: -33.1110268, 34.9088249, -33.1110268, 34.9088249, -68.0198517, 68.0198517
1: -236.7059174, 81.9441833, -236.7059174, 81.9441833, -318.6500854, 318.6500549
2: -130.8000793, 75.3684692, -130.8000793, 75.3684692, -206.1685486, 206.1685486
3: -164.3968353, 60.3954544, -164.3968353, 60.3954544, -224.7922516, 224.7922516
4: -94.1242828, 65.9827042, -94.1242828, 65.9827042, -160.1069794, 160.1069794

Time for backsubstitution: 1.19 seconds

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

Time for candidate selection: 0.10 seconds

### Candidate
type: B, layer: 1, pos: 4

## Relational analysis of NS_B1_B2_A1_A2_B1

### Relational analysis result of NS_B1_B2_A1_A2_B1
Status: Status.UNKNOWN
Output dim: 0, lower bound: -63.3958435, upper bound: 63.3957216
time: 1.29 seconds

## Relational analysis of NS_B1_B2_A1_A2_B2

### Relational analysis result of NS_B1_B2_A1_A2_B2
Status: Status.UNKNOWN
Output dim: 0, lower bound: -63.3954598, upper bound: 63.3954598
time: 0.68 seconds

## BFS NS instance: NS_B1_B2_A2_B1

### Backsubstitution after applying NS history:
0: -53.5134697, 58.3027115, -31.7075348, 33.5421410, -87.0556030, 90.0102310
1: -397.7839661, 132.1765289, -228.2451782, 78.4206696, -476.2046509, 360.4216309
2: -211.6885529, 124.9430847, -125.0585175, 72.3499680, -284.0385132, 250.0016022
3: -273.8537598, 100.1888733, -157.9605103, 57.9945068, -331.8482666, 258.1493835
4: -151.4457550, 110.6002960, -89.8834915, 63.3862991, -214.8320465, 200.4837799

Time for backsubstitution: 1.26 seconds

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
time: 0.76 seconds

## Relational analysis of NS_B1_B2_A2_B1_B2

### Relational analysis result of NS_B1_B2_A2_B1_B2
Status: Status.UNKNOWN
Output dim: 0, lower bound: -63.3936884, upper bound: 63.3935634
time: 0.69 seconds

## BFS NS instance: NS_B1_B2_A2_B2

### Backsubstitution after applying NS history:
0: -54.1864319, 58.9121284, -31.9944019, 33.7464180, -87.9328461, 90.9065247
1: -401.2415771, 133.7657928, -228.5129700, 79.2402725, -480.4818420, 362.2787476
2: -214.2349548, 126.2974319, -126.8174133, 72.9059601, -287.1408691, 253.1148376
3: -276.5735779, 101.2141190, -159.0127563, 58.4444389, -335.0180054, 260.2268677
4: -153.4540100, 111.7632599, -91.2179718, 63.8624268, -217.3164215, 202.9812012

Time for backsubstitution: 1.17 seconds

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

Time for candidate selection: 0.10 seconds

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
time: 0.75 seconds

## Relational analysis of NS_B1_B2_A2_B2_B2

### Relational analysis result of NS_B1_B2_A2_B2_B2
Status: Status.UNKNOWN
Output dim: 0, lower bound: -63.3941813, upper bound: 63.3942861
time: 0.84 seconds

## BFS NS instance: NS_B2_A2_A1_A1

### Backsubstitution after applying NS history:
0: -31.2933064, 32.9650536, -53.9953995, 58.7144966, -90.0078049, 86.9604263
1: -223.5657806, 77.4652863, -400.3821106, 133.3540344, -356.9197693, 477.8474121
2: -123.6531296, 71.1243896, -213.6814728, 125.8472977, -249.5004272, 284.8058472
3: -155.3525391, 57.0316010, -275.9838257, 100.9185486, -256.2710876, 333.0154114
4: -88.9686127, 62.2370262, -152.9810638, 111.3096771, -200.2782745, 215.2180939

Time for backsubstitution: 1.41 seconds

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

Time for candidate selection: 0.11 seconds

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
time: 0.81 seconds

## Relational analysis of NS_B2_A2_A1_A1_B2

### Relational analysis result of NS_B2_A2_A1_A1_B2
Status: Status.UNKNOWN
Output dim: 0, lower bound: -63.3942459, upper bound: 63.3952135
time: 0.76 seconds

## BFS NS instance: NS_B2_A2_A1_A2

### Backsubstitution after applying NS history:
0: -41.4766655, 42.6777725, -53.3992004, 57.9703140, -99.4469528, 96.0769501
1: -280.1462708, 102.6841431, -393.9755554, 131.7529907, -411.8992310, 496.6596985
2: -161.1961517, 92.3508987, -210.8420868, 124.2164688, -285.4126282, 303.1929626
3: -197.2792816, 73.9040070, -271.7917480, 99.5846939, -296.8639832, 345.6956787
4: -117.7232513, 80.8190765, -151.1241608, 109.9499054, -227.6731567, 231.9432373

Time for backsubstitution: 1.38 seconds

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

Time for candidate selection: 0.10 seconds

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
time: 0.79 seconds

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

Time for backsubstitution: 1.13 seconds

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

Time for candidate selection: 0.08 seconds

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
Status: Status.VERIFIED
Output dim: 0, lower bound: -63.3930898, upper bound: 63.3931718
time: 0.78 seconds

## Relational analysis of NS_B2_A2_A2_B1_A2

### Relational analysis result of NS_B2_A2_A2_B1_A2
Status: Status.UNKNOWN
Output dim: 0, lower bound: -63.3935402, upper bound: 63.3938446
time: 0.79 seconds

## BFS NS instance: NS_B2_A2_A2_B2

### Backsubstitution after applying NS history:
0: -54.0260849, 58.7897339, -54.0260849, 58.7897339, -112.8158112, 112.8158112
1: -400.6096802, 133.4270172, -400.6096802, 133.4270172, -534.0366821, 534.0366821
2: -213.7019806, 126.0158386, -213.7019806, 126.0158386, -339.7178040, 339.7178040
3: -276.0601196, 101.0406876, -276.0601196, 101.0406876, -377.1007996, 377.1007996
4: -152.9686127, 111.5431519, -152.9686127, 111.5431519, -264.5117493, 264.5117493

Time for backsubstitution: 1.17 seconds

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

Time for candidate selection: 0.11 seconds

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
Status: Status.VERIFIED
Output dim: 0, lower bound: -63.3930898, upper bound: 63.3933078
time: 0.78 seconds

## Relational analysis of NS_B2_A2_A2_B2_A2

### Relational analysis result of NS_B2_A2_A2_B2_A2
Status: Status.UNKNOWN
Output dim: 0, lower bound: -63.3935402, upper bound: 63.3948172
time: 0.77 seconds

## Summary of splitting at layer (split count: 4)
- Time for NS candidates: 4.98 seconds
NS_B1_B1_A1_A1_B1, status: Status.UNKNOWN, split count: 5, time: 4.98
Output dim: 0, lower bound: -63.3958876, upper bound: 63.3958122
NS_B1_B1_A1_A1_B2, status: Status.UNKNOWN, split count: 5, time: 4.98
Output dim: 0, lower bound: -63.3958450, upper bound: 63.3957590
NS_B1_B1_A1_A2_B1, status: Status.UNKNOWN, split count: 5, time: 4.98
Output dim: 0, lower bound: -63.3958876, upper bound: 63.3958122
NS_B1_B1_A1_A2_B2, status: Status.UNKNOWN, split count: 5, time: 4.98
Output dim: 0, lower bound: -63.3958450, upper bound: 63.3957590
NS_B1_B1_A2_B1_B1, status: Status.UNKNOWN, split count: 5, time: 4.98
Output dim: 0, lower bound: -63.3943110, upper bound: 63.3942986
NS_B1_B1_A2_B1_B2, status: Status.UNKNOWN, split count: 5, time: 4.98
Output dim: 0, lower bound: -63.3941184, upper bound: 63.3941425
NS_B1_B1_A2_B2_B1, status: Status.UNKNOWN, split count: 5, time: 4.98
Output dim: 0, lower bound: -63.3942807, upper bound: 63.3943366
NS_B1_B1_A2_B2_B2, status: Status.UNKNOWN, split count: 5, time: 4.98
Output dim: 0, lower bound: -63.3942638, upper bound: 63.3943223
NS_B1_B2_A1_A1_A1, status: Status.UNKNOWN, split count: 5, time: 4.98
Output dim: 0, lower bound: -63.3957216, upper bound: 63.3958435
NS_B1_B2_A1_A1_A2, status: Status.UNKNOWN, split count: 5, time: 4.98
Output dim: 0, lower bound: -63.3954598, upper bound: 63.3954598
NS_B1_B2_A1_A2_B1, status: Status.UNKNOWN, split count: 5, time: 4.98
Output dim: 0, lower bound: -63.3958435, upper bound: 63.3957216
NS_B1_B2_A1_A2_B2, status: Status.UNKNOWN, split count: 5, time: 4.98
Output dim: 0, lower bound: -63.3954598, upper bound: 63.3954598
NS_B1_B2_A2_B1_B1, status: Status.UNKNOWN, split count: 5, time: 4.98
Output dim: 0, lower bound: -63.3939212, upper bound: 63.3941125
NS_B1_B2_A2_B1_B2, status: Status.UNKNOWN, split count: 5, time: 4.98
Output dim: 0, lower bound: -63.3936884, upper bound: 63.3935634
NS_B1_B2_A2_B2_B1, status: Status.UNKNOWN, split count: 5, time: 4.98
Output dim: 0, lower bound: -63.3935961, upper bound: 63.3942849
NS_B1_B2_A2_B2_B2, status: Status.UNKNOWN, split count: 5, time: 4.98
Output dim: 0, lower bound: -63.3941813, upper bound: 63.3942861
NS_B2_A2_A1_A1_B1, status: Status.UNKNOWN, split count: 5, time: 4.98
Output dim: 0, lower bound: -63.3942459, upper bound: 63.3945203
NS_B2_A2_A1_A1_B2, status: Status.UNKNOWN, split count: 5, time: 4.98
Output dim: 0, lower bound: -63.3942459, upper bound: 63.3952135
NS_B2_A2_A1_A2_B1, status: Status.UNKNOWN, split count: 5, time: 4.98
Output dim: 0, lower bound: -63.3940284, upper bound: 63.3944068
NS_B2_A2_A1_A2_B2, status: Status.UNKNOWN, split count: 5, time: 4.98
Output dim: 0, lower bound: -63.3940284, upper bound: 63.3949890
NS_B2_A2_A2_B1_A1, status: Status.VERIFIED, split count: 5, time: 4.98
Output dim: 0, lower bound: -63.3930898, upper bound: 63.3931718
NS_B2_A2_A2_B1_A2, status: Status.UNKNOWN, split count: 5, time: 4.98
Output dim: 0, lower bound: -63.3935402, upper bound: 63.3938446
NS_B2_A2_A2_B2_A1, status: Status.VERIFIED, split count: 5, time: 4.98
Output dim: 0, lower bound: -63.3930898, upper bound: 63.3933078
NS_B2_A2_A2_B2_A2, status: Status.UNKNOWN, split count: 5, time: 4.98
Output dim: 0, lower bound: -63.3935402, upper bound: 63.3948172

## BFS NS instance: NS_B1_B1_A1_A1_B1

### Backsubstitution after applying NS history:
0: -32.1090889, 33.6279716, -31.2985306, 32.7712555, -64.8803406, 64.9264984
1: -227.5665436, 79.5726700, -221.7436371, 77.5785599, -305.1451111, 301.3163147
2: -127.1797180, 72.6204758, -123.9428406, 70.7823410, -197.9620667, 196.5633240
3: -159.0883789, 58.2320824, -155.0498047, 56.7560463, -215.8444214, 213.2818909
4: -91.5803223, 63.3711128, -89.2801895, 61.7543640, -153.3346710, 152.6512909

Time for backsubstitution: 1.13 seconds

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

Time for candidate selection: 0.09 seconds

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
time: 0.76 seconds

## Relational analysis of NS_B1_B1_A1_A1_B1_A2

### Relational analysis result of NS_B1_B1_A1_A1_B1_A2
Status: Status.UNKNOWN
Output dim: 0, lower bound: -63.3963898, upper bound: 63.3963898
time: 0.74 seconds

## BFS NS instance: NS_B1_B1_A1_A1_B2

### Backsubstitution after applying NS history:
0: -32.0274887, 33.4556084, -43.3717232, 44.9401779, -76.9676666, 76.8273315
1: -226.3287659, 79.3363266, -303.9672852, 107.6393127, -333.9680481, 382.7997131
2: -126.7314911, 72.2876282, -170.4709625, 98.0167847, -224.7482758, 242.0802002
3: -158.3087463, 57.9702454, -213.3660736, 78.5642319, -236.8729858, 269.9973755
4: -91.3053436, 63.0636482, -123.1605453, 85.3021240, -176.6074677, 185.8313141

Time for backsubstitution: 1.27 seconds

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

Time for candidate selection: 0.10 seconds

### Candidate
type: A, layer: 1, pos: 4

## Relational analysis of NS_B1_B1_A1_A1_B2_A1

### Relational analysis result of NS_B1_B1_A1_A1_B2_A1
Status: Status.UNKNOWN
Output dim: 0, lower bound: -63.3963898, upper bound: 63.3963898
time: 0.77 seconds

## Relational analysis of NS_B1_B1_A1_A1_B2_A2

### Relational analysis result of NS_B1_B1_A1_A1_B2_A2
Status: Status.UNKNOWN
Output dim: 0, lower bound: -63.3963898, upper bound: 63.3963898
time: 0.72 seconds

## BFS NS instance: NS_B1_B1_A1_A2_B1

### Backsubstitution after applying NS history:
0: -31.2985344, 32.9879265, -31.2985306, 32.7712555, -64.0697937, 64.2864456
1: -223.0827179, 77.4370422, -221.7436371, 77.5785599, -300.6612854, 299.1806030
2: -123.5101624, 71.1551437, -123.9428406, 70.7823410, -194.2925110, 195.0979919
3: -155.0326538, 57.0326843, -155.0498047, 56.7560463, -211.7886810, 212.0824738
4: -88.9413681, 62.3415108, -89.2801895, 61.7543640, -150.6957397, 151.6217041

Time for backsubstitution: 1.19 seconds

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
time: 0.70 seconds

## Relational analysis of NS_B1_B1_A1_A2_B1_B2

### Relational analysis result of NS_B1_B1_A1_A2_B1_B2
Status: Status.UNKNOWN
Output dim: 0, lower bound: -63.3958754, upper bound: 63.3958122
time: 0.76 seconds

## BFS NS instance: NS_B1_B1_A1_A2_B2

### Backsubstitution after applying NS history:
0: -31.9320335, 33.5925255, -43.3717232, 44.9401779, -76.8722076, 76.9642334
1: -227.7259827, 79.0248489, -303.9672852, 107.6393127, -335.3652649, 382.9920654
2: -126.1115570, 72.5627213, -170.4709625, 98.0167847, -224.1283264, 242.5304260
3: -158.2653961, 58.1356277, -213.3660736, 78.5642319, -236.8296204, 270.2504578
4: -90.7752762, 63.4691505, -123.1605453, 85.3021240, -176.0773773, 186.3454742

Time for backsubstitution: 1.09 seconds

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

Time for candidate selection: 0.09 seconds

### Candidate
type: A, layer: 1, pos: 4

## Relational analysis of NS_B1_B1_A1_A2_B2_A1

### Relational analysis result of NS_B1_B1_A1_A2_B2_A1
Status: Status.UNKNOWN
Output dim: 0, lower bound: -63.3958450, upper bound: 63.3957590
time: 0.70 seconds

## Relational analysis of NS_B1_B1_A1_A2_B2_A2

### Relational analysis result of NS_B1_B1_A1_A2_B2_A2
Status: Status.UNKNOWN
Output dim: 0, lower bound: -63.3958450, upper bound: 63.3957590
time: 0.76 seconds

## BFS NS instance: NS_B1_B1_A2_B1_B1

### Backsubstitution after applying NS history:
0: -53.7749062, 58.3826141, -30.8998337, 32.3675804, -86.1424866, 89.2824326
1: -397.2376709, 132.6823425, -219.2346802, 76.5160141, -472.7992554, 351.9169617
2: -212.4016876, 125.1452332, -122.6432190, 69.8588181, -282.2604980, 247.7884521
3: -274.0435181, 100.2597046, -153.3082123, 56.0272484, -329.8058472, 253.5678711
4: -152.4556427, 110.6469498, -88.1771545, 60.9467621, -213.4023895, 198.8240967

Time for backsubstitution: 1.26 seconds

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

Time for candidate selection: 0.09 seconds

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
time: 0.73 seconds

## BFS NS instance: NS_B1_B1_A2_B1_B2

### Backsubstitution after applying NS history:
0: -53.2943115, 57.7648315, -40.1849136, 41.3175163, -94.6118317, 97.8713074
1: -391.6769714, 131.4059601, -270.5454102, 99.4554672, -489.2981567, 401.9513550
2: -209.9710083, 123.7719040, -155.8203583, 89.4233932, -299.3944092, 278.8367004
3: -270.4146118, 99.1839218, -190.4992371, 71.5364838, -341.6929932, 289.5463257
4: -150.9994507, 109.5045471, -113.9260864, 78.2623062, -229.2617340, 222.4403229

Time for backsubstitution: 1.21 seconds

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

Time for candidate selection: 0.10 seconds

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
time: 0.75 seconds

## Relational analysis of NS_B1_B1_A2_B1_B2_A2

### Relational analysis result of NS_B1_B1_A2_B1_B2_A2
Status: Status.UNKNOWN
Output dim: 0, lower bound: -63.3939945, upper bound: 63.3941425
time: 0.68 seconds

## BFS NS instance: NS_B1_B1_A2_B2_B1

### Backsubstitution after applying NS history:
0: -53.7296906, 58.2884941, -33.7412872, 35.4420433, -89.1717377, 92.0297852
1: -396.4836731, 132.5462341, -241.4537048, 83.6052399, -479.1410217, 373.9999390
2: -212.1580200, 124.9503784, -134.5909882, 76.4633179, -288.6213074, 259.5413513
3: -273.6259460, 100.1009369, -168.7952271, 61.3642120, -334.7517395, 268.8961182
4: -152.3504181, 110.4550476, -96.5188828, 66.6376648, -218.9880829, 206.9739227

Time for backsubstitution: 1.29 seconds

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

Time for candidate selection: 0.10 seconds

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
time: 0.79 seconds

## Relational analysis of NS_B1_B1_A2_B2_B1_A2

### Relational analysis result of NS_B1_B1_A2_B2_B1_A2
Status: Status.UNKNOWN
Output dim: 0, lower bound: -63.3940763, upper bound: 63.3943366
time: 0.73 seconds

## BFS NS instance: NS_B1_B1_A2_B2_B2

### Backsubstitution after applying NS history:
0: -53.1750488, 57.5974617, -43.8191643, 45.3083954, -98.4834442, 101.2977448
1: -390.4071960, 131.1018524, -299.1305847, 108.4494934, -496.9378357, 430.2324219
2: -209.4720612, 123.4078064, -171.1938019, 97.9577637, -307.4298096, 293.4460449
3: -269.6227417, 98.9051590, -210.4305420, 78.3455505, -347.7146912, 308.6618347
4: -150.6838074, 109.1703568, -124.5197067, 85.6121521, -236.2959595, 232.3986816

Time for backsubstitution: 1.30 seconds

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

Time for candidate selection: 0.10 seconds

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
time: 0.71 seconds

## Relational analysis of NS_B1_B1_A2_B2_B2_A2

### Relational analysis result of NS_B1_B1_A2_B2_B2_A2
Status: Status.UNKNOWN
Output dim: 0, lower bound: -63.3939964, upper bound: 63.3943223
time: 0.71 seconds

## BFS NS instance: NS_B1_B2_A1_A1_A1

### Backsubstitution after applying NS history:
0: -31.2985306, 32.7712555, -31.2985344, 32.9879265, -64.2864532, 64.0697937
1: -221.7436371, 77.5785599, -223.0827179, 77.4370422, -299.1806030, 300.6612854
2: -123.9428406, 70.7823410, -123.5101624, 71.1551437, -195.0979919, 194.2925110
3: -155.0498047, 56.7560463, -155.0326538, 57.0326843, -212.0824738, 211.7886810
4: -89.2801895, 61.7543640, -88.9413681, 62.3415108, -151.6217041, 150.6957397

Time for backsubstitution: 1.32 seconds

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

Time for candidate selection: 0.10 seconds

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
time: 0.70 seconds

## Relational analysis of NS_B1_B2_A1_A1_A1_A2

### Relational analysis result of NS_B1_B2_A1_A1_A1_A2
Status: Status.UNKNOWN
Output dim: 0, lower bound: -63.3958122, upper bound: 63.3958754
time: 0.65 seconds

## BFS NS instance: NS_B1_B2_A1_A1_A2

### Backsubstitution after applying NS history:
0: -43.3717232, 44.9401779, -31.9320335, 33.5925255, -76.9642334, 76.8722076
1: -303.9672852, 107.6393127, -227.7259827, 79.0248489, -382.9920654, 335.3652344
2: -170.4709625, 98.0167847, -126.1115570, 72.5627213, -242.5304260, 224.1283264
3: -213.3660736, 78.5642319, -158.2653961, 58.1356277, -270.2504272, 236.8296204
4: -123.1605453, 85.3021240, -90.7752762, 63.4691505, -186.3454742, 176.0773926

Time for backsubstitution: 1.24 seconds

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

Time for candidate selection: 0.10 seconds

### Candidate
type: B, layer: 1, pos: 4

## Relational analysis of NS_B1_B2_A1_A1_A2_B1

### Relational analysis result of NS_B1_B2_A1_A1_A2_B1
Status: Status.UNKNOWN
Output dim: 0, lower bound: -63.3957590, upper bound: 63.3958450
time: 0.72 seconds

## Relational analysis of NS_B1_B2_A1_A1_A2_B2

### Relational analysis result of NS_B1_B2_A1_A1_A2_B2
Status: Status.UNKNOWN
Output dim: 0, lower bound: -63.3957590, upper bound: 63.3958450
time: 0.77 seconds

## BFS NS instance: NS_B1_B2_A1_A2_B1

### Backsubstitution after applying NS history:
0: -31.2985344, 32.9879265, -30.1823769, 31.8147984, -63.1133347, 63.1703033
1: -223.0827179, 77.4370422, -214.8651276, 74.6638260, -297.7465515, 292.3021545
2: -123.5101624, 71.1551437, -119.0411377, 68.5830307, -192.0932007, 190.1962891
3: -155.0326538, 57.0326843, -149.3636475, 54.9800568, -210.0127106, 206.3963318
4: -88.9413681, 62.3415108, -85.7541580, 60.1135139, -149.0548859, 148.0956726

Time for backsubstitution: 1.13 seconds

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

Time for candidate selection: 0.10 seconds

### Candidate
type: B, layer: 1, pos: 9

## Relational analysis of NS_B1_B2_A1_A2_B1_B1

### Relational analysis result of NS_B1_B2_A1_A2_B1_B1
Status: Status.UNKNOWN
Output dim: 0, lower bound: -63.3938505, upper bound: 63.3932262
time: 0.72 seconds

## Relational analysis of NS_B1_B2_A1_A2_B1_B2

### Relational analysis result of NS_B1_B2_A1_A2_B1_B2
Status: Status.UNKNOWN
Output dim: 0, lower bound: -63.3949434, upper bound: 63.3947308
time: 0.94 seconds

## BFS NS instance: NS_B1_B2_A1_A2_B2

### Backsubstitution after applying NS history:
0: -31.9320335, 33.5925255, -44.6238937, 46.3271980, -78.2592316, 78.2164154
1: -227.7259827, 79.0248489, -313.9445801, 110.7814026, -338.5073853, 392.9693604
2: -126.1115570, 72.5627213, -175.7769775, 100.9844360, -227.0959778, 248.3396912
3: -158.2653961, 58.1356277, -220.2172699, 80.9449158, -239.2103119, 278.3528442
4: -90.7752762, 63.4691505, -126.8017273, 87.9186707, -178.6939392, 190.2708740

Time for backsubstitution: 1.24 seconds

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

Time for candidate selection: 0.10 seconds

### Candidate
type: A, layer: 1, pos: 4

## Relational analysis of NS_B1_B2_A1_A2_B2_A1

### Relational analysis result of NS_B1_B2_A1_A2_B2_A1
Status: Status.UNKNOWN
Output dim: 0, lower bound: -63.3954598, upper bound: 63.3954598
time: 0.72 seconds

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

Time for backsubstitution: 1.26 seconds

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

Time for candidate selection: 0.11 seconds

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
time: 0.77 seconds

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

Time for backsubstitution: 1.25 seconds

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

Time for candidate selection: 0.12 seconds

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
Status: Status.VERIFIED
Output dim: 0, lower bound: -63.3936783, upper bound: 63.3935557
time: 0.71 seconds

## Relational analysis of NS_B1_B2_A2_B1_B2_A2

### Relational analysis result of NS_B1_B2_A2_B1_B2_A2
Status: Status.VERIFIED
Output dim: 0, lower bound: -63.3936783, upper bound: 63.3935634
time: 0.70 seconds

## BFS NS instance: NS_B1_B2_A2_B2_B1

### Backsubstitution after applying NS history:
0: -53.6934738, 58.4299927, -31.2356853, 32.9815636, -86.6750336, 89.6656723
1: -398.1136169, 132.5605774, -223.3799896, 77.3887100, -475.5023193, 355.9405518
2: -212.3375549, 125.2409058, -123.8424377, 71.2410431, -283.5786133, 249.0833435
3: -274.3168640, 100.3724899, -155.3355408, 57.1053543, -331.4222107, 255.7080231
4: -152.0602875, 110.8469315, -89.0727921, 62.4052467, -214.4654999, 199.9197235

Time for backsubstitution: 1.30 seconds

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

Time for candidate selection: 0.12 seconds

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
time: 0.78 seconds

## Relational analysis of NS_B1_B2_A2_B2_B1_B2

### Relational analysis result of NS_B1_B2_A2_B2_B1_B2
Status: Status.VERIFIED
Output dim: 0, lower bound: -63.3931911, upper bound: 63.3933291
time: 0.79 seconds

## BFS NS instance: NS_B1_B2_A2_B2_B2

### Backsubstitution after applying NS history:
0: -53.6087799, 58.2966270, -33.7474594, 35.7593155, -89.3680801, 92.0440826
1: -397.0691223, 132.3298340, -244.1085052, 83.6298599, -480.6989441, 376.4383545
2: -211.9619293, 124.9605103, -134.4412231, 77.1542892, -289.1161804, 259.4017029
3: -273.7013550, 100.1439285, -169.4770813, 61.8347511, -335.5360413, 269.6209717
4: -151.8250275, 110.5817719, -96.3958893, 67.4860153, -219.3110046, 206.9776611

Time for backsubstitution: 1.27 seconds

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

Time for candidate selection: 0.10 seconds

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
time: 0.88 seconds

## Relational analysis of NS_B1_B2_A2_B2_B2_B2

### Relational analysis result of NS_B1_B2_A2_B2_B2_B2
Status: Status.UNKNOWN
Output dim: 0, lower bound: -63.3937801, upper bound: 63.3937700
time: 0.76 seconds

## BFS NS instance: NS_B2_A2_A1_A1_B1

### Backsubstitution after applying NS history:
0: -31.2933064, 32.9650536, -52.9280472, 57.3836327, -88.6769409, 85.8930740
1: -223.5657806, 77.4652863, -389.7350159, 130.5437317, -354.1094971, 466.6008606
2: -123.6531296, 71.1243896, -208.6952515, 123.0406799, -246.6938171, 279.8196411
3: -155.3525391, 57.0316010, -268.9759521, 98.5570068, -253.9095459, 325.7637634
4: -88.9686127, 62.2370262, -150.0240326, 108.8049927, -197.7735901, 212.2610626

Time for backsubstitution: 1.24 seconds

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

Time for candidate selection: 0.11 seconds

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
time: 0.72 seconds

## Relational analysis of NS_B2_A2_A1_A1_B1_A2

### Relational analysis result of NS_B2_A2_A1_A1_B1_A2
Status: Status.UNKNOWN
Output dim: 0, lower bound: -63.3939587, upper bound: 63.3944746
time: 0.78 seconds

## BFS NS instance: NS_B2_A2_A1_A1_B2

### Backsubstitution after applying NS history:
0: -31.2933064, 32.9650536, -53.1764832, 57.9081841, -89.2014923, 86.1415100
1: -223.5657806, 77.4652863, -394.9996948, 131.3654633, -354.9312439, 472.4649658
2: -123.6531296, 71.1243896, -210.4864197, 124.0891571, -247.7422791, 281.6108093
3: -155.3525391, 57.0316010, -272.1233826, 99.5140686, -254.8665924, 329.1549683
4: -88.9686127, 62.2370262, -150.6161346, 109.8162842, -198.7848511, 212.8531647

Time for backsubstitution: 1.30 seconds

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

Time for candidate selection: 0.10 seconds

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
time: 0.71 seconds

## Relational analysis of NS_B2_A2_A1_A1_B2_A2

### Relational analysis result of NS_B2_A2_A1_A1_B2_A2
Status: Status.UNKNOWN
Output dim: 0, lower bound: -63.3939587, upper bound: 63.3947644
time: 0.79 seconds

## BFS NS instance: NS_B2_A2_A1_A2_B1

### Backsubstitution after applying NS history:
0: -41.4766655, 42.6777725, -52.4175224, 56.7414017, -98.1967850, 95.0952911
1: -280.1462708, 102.6841431, -384.0408325, 129.2115936, -409.3577881, 484.9897156
2: -161.1961517, 92.3508987, -206.3283691, 121.6101837, -282.3533936, 298.6792297
3: -197.2792816, 73.9040070, -265.2448730, 97.4538727, -294.7331543, 338.8872986
4: -117.7232513, 80.8190765, -148.4796600, 107.6116180, -224.5513153, 229.2987366

Time for backsubstitution: 1.30 seconds

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

Time for candidate selection: 0.11 seconds

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
time: 0.76 seconds

## Relational analysis of NS_B2_A2_A1_A2_B1_B2

### Relational analysis result of NS_B2_A2_A1_A2_B1_B2
Status: Status.UNKNOWN
Output dim: 0, lower bound: -63.3937826, upper bound: 63.3941691
time: 0.77 seconds

## BFS NS instance: NS_B2_A2_A1_A2_B2

### Backsubstitution after applying NS history:
0: -41.4766655, 42.6777725, -52.5505524, 57.1262398, -98.6028976, 95.2283249
1: -280.1462708, 102.6841431, -388.2085876, 129.6735077, -409.8197327, 490.8927307
2: -161.1961517, 92.3508987, -207.5194092, 122.3719025, -283.5680542, 299.8702698
3: -197.2792816, 73.9040070, -267.7059937, 98.1092606, -295.3885193, 341.6099854
4: -117.7232513, 80.8190765, -148.6719513, 108.3880539, -226.1112976, 229.4910278

Time for backsubstitution: 1.29 seconds

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

Time for candidate selection: 0.11 seconds

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
time: 0.78 seconds

## Relational analysis of NS_B2_A2_A1_A2_B2_B2

### Relational analysis result of NS_B2_A2_A1_A2_B2_B2
Status: Status.UNKNOWN
Output dim: 0, lower bound: -63.3937826, upper bound: 63.3947862
time: 0.78 seconds

## BFS NS instance: NS_B2_A2_A2_B1_A2

### Backsubstitution after applying NS history:
0: -53.8847351, 58.6575432, -53.9753265, 58.4763908, -112.3611069, 112.6328735
1: -399.7374878, 133.0946503, -396.8188477, 133.1195221, -532.2844849, 529.0166626
2: -213.1640167, 125.7145767, -212.6795044, 125.4174194, -338.5814209, 338.3940735
3: -275.4341431, 100.8090820, -273.8961182, 100.4767838, -375.3064575, 373.8242798
4: -152.5790253, 111.2873230, -152.9559479, 110.9175262, -263.4965515, 264.2432556

Time for backsubstitution: 1.43 seconds

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

Time for candidate selection: 0.12 seconds

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
time: 0.71 seconds

## Relational analysis of NS_B2_A2_A2_B1_A2_B2

### Relational analysis result of NS_B2_A2_A2_B1_A2_B2
Status: Status.UNKNOWN
Output dim: 0, lower bound: -63.3934490, upper bound: 63.3938446
time: 0.73 seconds

## BFS NS instance: NS_B2_A2_A2_B2_A2

### Backsubstitution after applying NS history:
0: -53.8847351, 58.6575432, -54.0260849, 58.7897339, -112.6744690, 112.6836090
1: -399.7374878, 133.0946503, -400.6096802, 133.4270172, -533.1644287, 533.7043457
2: -213.1640167, 125.7145767, -213.7019806, 126.0158386, -339.1798096, 339.4165344
3: -275.4341431, 100.8090820, -276.0601196, 101.0406876, -376.4747925, 376.8692017
4: -152.5790253, 111.2873230, -152.9686127, 111.5431519, -264.1221313, 264.2558899

Time for backsubstitution: 1.41 seconds

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
time: 0.88 seconds

## Summary of splitting at layer (split count: 5)
- Time for NS candidates: 4.57 seconds
NS_B1_B1_A1_A1_B1_A1, status: Status.UNKNOWN, split count: 6, time: 4.57
Output dim: 0, lower bound: -63.3963898, upper bound: 63.3963898
NS_B1_B1_A1_A1_B1_A2, status: Status.UNKNOWN, split count: 6, time: 4.57
Output dim: 0, lower bound: -63.3963898, upper bound: 63.3963898
NS_B1_B1_A1_A1_B2_A1, status: Status.UNKNOWN, split count: 6, time: 4.57
Output dim: 0, lower bound: -63.3963898, upper bound: 63.3963898
NS_B1_B1_A1_A1_B2_A2, status: Status.UNKNOWN, split count: 6, time: 4.57
Output dim: 0, lower bound: -63.3963898, upper bound: 63.3963898
NS_B1_B1_A1_A2_B1_B1, status: Status.UNKNOWN, split count: 6, time: 4.57
Output dim: 0, lower bound: -63.3958702, upper bound: 63.3958122
NS_B1_B1_A1_A2_B1_B2, status: Status.UNKNOWN, split count: 6, time: 4.57
Output dim: 0, lower bound: -63.3958754, upper bound: 63.3958122
NS_B1_B1_A1_A2_B2_A1, status: Status.UNKNOWN, split count: 6, time: 4.57
Output dim: 0, lower bound: -63.3958450, upper bound: 63.3957590
NS_B1_B1_A1_A2_B2_A2, status: Status.UNKNOWN, split count: 6, time: 4.57
Output dim: 0, lower bound: -63.3958450, upper bound: 63.3957590
NS_B1_B1_A2_B1_B1_A1, status: Status.UNKNOWN, split count: 6, time: 4.57
Output dim: 0, lower bound: -63.3942257, upper bound: 63.3940786
NS_B1_B1_A2_B1_B1_A2, status: Status.UNKNOWN, split count: 6, time: 4.57
Output dim: 0, lower bound: -63.3942257, upper bound: 63.3942986
NS_B1_B1_A2_B1_B2_A1, status: Status.UNKNOWN, split count: 6, time: 4.57
Output dim: 0, lower bound: -63.3939945, upper bound: 63.3939685
NS_B1_B1_A2_B1_B2_A2, status: Status.UNKNOWN, split count: 6, time: 4.57
Output dim: 0, lower bound: -63.3939945, upper bound: 63.3941425
NS_B1_B1_A2_B2_B1_A1, status: Status.UNKNOWN, split count: 6, time: 4.57
Output dim: 0, lower bound: -63.3940763, upper bound: 63.3939509
NS_B1_B1_A2_B2_B1_A2, status: Status.UNKNOWN, split count: 6, time: 4.57
Output dim: 0, lower bound: -63.3940763, upper bound: 63.3943366
NS_B1_B1_A2_B2_B2_A1, status: Status.UNKNOWN, split count: 6, time: 4.57
Output dim: 0, lower bound: -63.3939964, upper bound: 63.3939342
NS_B1_B1_A2_B2_B2_A2, status: Status.UNKNOWN, split count: 6, time: 4.57
Output dim: 0, lower bound: -63.3939964, upper bound: 63.3943223
NS_B1_B2_A1_A1_A1_A1, status: Status.UNKNOWN, split count: 6, time: 4.57
Output dim: 0, lower bound: -63.3958122, upper bound: 63.3958702
NS_B1_B2_A1_A1_A1_A2, status: Status.UNKNOWN, split count: 6, time: 4.57
Output dim: 0, lower bound: -63.3958122, upper bound: 63.3958754
NS_B1_B2_A1_A1_A2_B1, status: Status.UNKNOWN, split count: 6, time: 4.57
Output dim: 0, lower bound: -63.3957590, upper bound: 63.3958450
NS_B1_B2_A1_A1_A2_B2, status: Status.UNKNOWN, split count: 6, time: 4.57
Output dim: 0, lower bound: -63.3957590, upper bound: 63.3958450
NS_B1_B2_A1_A2_B1_B1, status: Status.UNKNOWN, split count: 6, time: 4.57
Output dim: 0, lower bound: -63.3938505, upper bound: 63.3932262
NS_B1_B2_A1_A2_B1_B2, status: Status.UNKNOWN, split count: 6, time: 4.57
Output dim: 0, lower bound: -63.3949434, upper bound: 63.3947308
NS_B1_B2_A1_A2_B2_A1, status: Status.UNKNOWN, split count: 6, time: 4.57
Output dim: 0, lower bound: -63.3954598, upper bound: 63.3954598
NS_B1_B2_A1_A2_B2_A2, status: Status.UNKNOWN, split count: 6, time: 4.57
Output dim: 0, lower bound: -63.3954598, upper bound: 63.3954598
NS_B1_B2_A2_B1_B1_B1, status: Status.UNKNOWN, split count: 6, time: 4.57
Output dim: 0, lower bound: -63.3936396, upper bound: 63.3940085
NS_B1_B2_A2_B1_B1_B2, status: Status.UNKNOWN, split count: 6, time: 4.57
Output dim: 0, lower bound: -63.3939212, upper bound: 63.3941125
NS_B1_B2_A2_B1_B2_A1, status: Status.VERIFIED, split count: 6, time: 4.57
Output dim: 0, lower bound: -63.3936783, upper bound: 63.3935557
NS_B1_B2_A2_B1_B2_A2, status: Status.VERIFIED, split count: 6, time: 4.57
Output dim: 0, lower bound: -63.3936783, upper bound: 63.3935634
NS_B1_B2_A2_B2_B1_B1, status: Status.UNKNOWN, split count: 6, time: 4.57
Output dim: 0, lower bound: -63.3940453, upper bound: 63.3941899
NS_B1_B2_A2_B2_B1_B2, status: Status.VERIFIED, split count: 6, time: 4.57
Output dim: 0, lower bound: -63.3931911, upper bound: 63.3933291
NS_B1_B2_A2_B2_B2_B1, status: Status.UNKNOWN, split count: 6, time: 4.57
Output dim: 0, lower bound: -63.3940682, upper bound: 63.3941989
NS_B1_B2_A2_B2_B2_B2, status: Status.UNKNOWN, split count: 6, time: 4.57
Output dim: 0, lower bound: -63.3937801, upper bound: 63.3937700
NS_B2_A2_A1_A1_B1_A1, status: Status.UNKNOWN, split count: 6, time: 4.57
Output dim: 0, lower bound: -63.3939698, upper bound: 63.3944786
NS_B2_A2_A1_A1_B1_A2, status: Status.UNKNOWN, split count: 6, time: 4.57
Output dim: 0, lower bound: -63.3939587, upper bound: 63.3944746
NS_B2_A2_A1_A1_B2_A1, status: Status.UNKNOWN, split count: 6, time: 4.57
Output dim: 0, lower bound: -63.3939698, upper bound: 63.3947612
NS_B2_A2_A1_A1_B2_A2, status: Status.UNKNOWN, split count: 6, time: 4.57
Output dim: 0, lower bound: -63.3939587, upper bound: 63.3947644
NS_B2_A2_A1_A2_B1_B1, status: Status.UNKNOWN, split count: 6, time: 4.57
Output dim: 0, lower bound: -63.3937079, upper bound: 63.3940319
NS_B2_A2_A1_A2_B1_B2, status: Status.UNKNOWN, split count: 6, time: 4.57
Output dim: 0, lower bound: -63.3937826, upper bound: 63.3941691
NS_B2_A2_A1_A2_B2_B1, status: Status.UNKNOWN, split count: 6, time: 4.57
Output dim: 0, lower bound: -63.3937079, upper bound: 63.3941582
NS_B2_A2_A1_A2_B2_B2, status: Status.UNKNOWN, split count: 6, time: 4.57
Output dim: 0, lower bound: -63.3937826, upper bound: 63.3947862
NS_B2_A2_A2_B1_A2_B1, status: Status.UNKNOWN, split count: 6, time: 4.57
Output dim: 0, lower bound: -63.3934490, upper bound: 63.3937162
NS_B2_A2_A2_B1_A2_B2, status: Status.UNKNOWN, split count: 6, time: 4.57
Output dim: 0, lower bound: -63.3934490, upper bound: 63.3938446
NS_B2_A2_A2_B2_A2_B1, status: Status.UNKNOWN, split count: 6, time: 4.57
Output dim: 0, lower bound: -63.3930551, upper bound: 63.3939188
NS_B2_A2_A2_B2_A2_B2, status: Status.UNKNOWN, split count: 6, time: 4.57
Output dim: 0, lower bound: -63.3936129, upper bound: 63.3948172

## BFS NS instance: NS_B1_B1_A1_A1_B1_A1

### Backsubstitution after applying NS history:
0: -31.2985306, 32.7712555, -31.2985306, 32.7712555, -64.0697784, 64.0697784
1: -221.7436371, 77.5785599, -221.7436371, 77.5785599, -299.3221741, 299.3221741
2: -123.9428406, 70.7823410, -123.9428406, 70.7823410, -194.7251892, 194.7251892
3: -155.0498047, 56.7560463, -155.0498047, 56.7560463, -211.8058472, 211.8058472
4: -89.2801895, 61.7543640, -89.2801895, 61.7543640, -151.0345459, 151.0345459

Time for backsubstitution: 1.24 seconds

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

Time for candidate selection: 0.09 seconds

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
time: 0.89 seconds

## Relational analysis of NS_B1_B1_A1_A1_B1_A1_B2

### Relational analysis result of NS_B1_B1_A1_A1_B1_A1_B2
Status: Status.UNKNOWN
Output dim: 0, lower bound: -63.3964120, upper bound: 63.3964073
time: 0.78 seconds

## BFS NS instance: NS_B1_B1_A1_A1_B1_A2

### Backsubstitution after applying NS history:
0: -43.3717232, 44.9401779, -31.2985306, 32.7712555, -76.1429749, 76.2387085
1: -303.9672852, 107.6393127, -221.7436371, 77.5785599, -381.0203857, 329.3828735
2: -170.4709625, 98.0167847, -123.9428406, 70.7823410, -240.5885010, 221.9596252
3: -213.3660736, 78.5642319, -155.0498047, 56.7560463, -268.8029480, 233.6140442
4: -123.1605453, 85.3021240, -89.2801895, 61.7543640, -184.5613861, 174.5823059

Time for backsubstitution: 1.20 seconds

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

Time for candidate selection: 0.09 seconds

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
time: 0.67 seconds

## Relational analysis of NS_B1_B1_A1_A1_B1_A2_B2

### Relational analysis result of NS_B1_B1_A1_A1_B1_A2_B2
Status: Status.UNKNOWN
Output dim: 0, lower bound: -63.3964120, upper bound: 63.3964073
time: 0.83 seconds

## BFS NS instance: NS_B1_B1_A1_A1_B2_A1

### Backsubstitution after applying NS history:
0: -31.2985306, 32.7712555, -43.3717232, 44.9401779, -76.2387085, 76.1429749
1: -221.7436371, 77.5785599, -303.9672852, 107.6393127, -329.3829041, 381.0203857
2: -123.9428406, 70.7823410, -170.4709625, 98.0167847, -221.9596252, 240.5885010
3: -155.0498047, 56.7560463, -213.3660736, 78.5642319, -233.6140442, 268.8029480
4: -89.2801895, 61.7543640, -123.1605453, 85.3021240, -174.5823059, 184.5613861

Time for backsubstitution: 1.26 seconds

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

Time for candidate selection: 0.09 seconds

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
time: 0.76 seconds

## Relational analysis of NS_B1_B1_A1_A1_B2_A1_A2

### Relational analysis result of NS_B1_B1_A1_A1_B2_A1_A2
Status: Status.UNKNOWN
Output dim: 0, lower bound: -63.3956942, upper bound: 63.3956942
time: 0.71 seconds

## BFS NS instance: NS_B1_B1_A1_A1_B2_A2

### Backsubstitution after applying NS history:
0: -43.3717232, 44.9401779, -43.3717232, 44.9401779, -88.2596130, 88.2596130
1: -303.9672852, 107.6393127, -303.9672852, 107.6393127, -410.1252747, 410.1252747
2: -170.4709625, 98.0167847, -170.4709625, 98.0167847, -267.2682495, 267.2682495
3: -213.3660736, 78.5642319, -213.3660736, 78.5642319, -290.1622009, 290.1622009
4: -123.1605453, 85.3021240, -123.1605453, 85.3021240, -207.7959747, 207.7959747

Time for backsubstitution: 1.30 seconds

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

Time for candidate selection: 0.10 seconds

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
time: 0.69 seconds

## Relational analysis of NS_B1_B1_A1_A1_B2_A2_B2

### Relational analysis result of NS_B1_B1_A1_A1_B2_A2_B2
Status: Status.UNKNOWN
Output dim: 0, lower bound: -63.3954147, upper bound: 63.3961970
time: 0.73 seconds

## BFS NS instance: NS_B1_B1_A1_A2_B1_B1

### Backsubstitution after applying NS history:
0: -29.8957691, 31.5139313, -29.5365906, 30.9115620, -60.8073311, 61.0505219
1: -213.3718414, 73.9482346, -209.6969757, 73.2576218, -286.6294556, 283.6451416
2: -117.9968338, 68.0444641, -117.0093307, 66.9293823, -184.9262085, 185.0537720
3: -148.1968842, 54.5233231, -146.5489655, 53.6311874, -201.8280640, 201.0722809
4: -84.9133301, 59.6252861, -84.2029343, 58.3682022, -143.2815247, 143.8282166

Time for backsubstitution: 1.27 seconds

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

Time for candidate selection: 0.11 seconds

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
time: 0.67 seconds

## Relational analysis of NS_B1_B1_A1_A2_B1_B1_A2

### Relational analysis result of NS_B1_B1_A1_A2_B1_B1_A2
Status: Status.UNKNOWN
Output dim: 0, lower bound: -63.3958702, upper bound: 63.3958122
time: 0.65 seconds

## BFS NS instance: NS_B1_B1_A1_A2_B1_B2

### Backsubstitution after applying NS history:
0: -29.2645760, 30.8694839, -29.2935848, 30.6672192, -59.9317932, 60.1630707
1: -208.9285583, 72.4774933, -207.5810242, 72.7790451, -281.7075806, 280.0585327
2: -115.4824677, 66.6106720, -115.9507675, 66.4279785, -181.9104462, 182.5614319
3: -145.1338348, 53.4042702, -145.0855560, 53.2599449, -198.3937836, 198.4898071
4: -83.1748199, 58.3728714, -83.5118332, 57.9175110, -141.0923309, 141.8847046

Time for backsubstitution: 1.30 seconds

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

Time for candidate selection: 0.11 seconds

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
time: 0.77 seconds

## Relational analysis of NS_B1_B1_A1_A2_B1_B2_A2

### Relational analysis result of NS_B1_B1_A1_A2_B1_B2_A2
Status: Status.UNKNOWN
Output dim: 0, lower bound: -63.3958754, upper bound: 63.3958122
time: 0.78 seconds

## BFS NS instance: NS_B1_B1_A1_A2_B2_A1

### Backsubstitution after applying NS history:
0: -30.1823769, 31.8147984, -43.3717232, 44.9401779, -75.1225433, 75.1865234
1: -214.8651276, 74.6638260, -303.9672852, 107.6393127, -322.5044250, 378.6311035
2: -119.0411377, 68.5830307, -170.4709625, 98.0167847, -217.0579224, 238.5750427
3: -149.3636475, 54.9800568, -213.3660736, 78.5642319, -227.9278870, 267.1139526
4: -85.7541580, 60.1135139, -123.1605453, 85.3021240, -171.0562744, 183.0218353

Time for backsubstitution: 1.17 seconds

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

Time for candidate selection: 0.11 seconds

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
time: 0.73 seconds

## Relational analysis of NS_B1_B1_A1_A2_B2_A1_A2

### Relational analysis result of NS_B1_B1_A1_A2_B2_A1_A2
Status: Status.UNKNOWN
Output dim: 0, lower bound: -63.3955360, upper bound: 63.3957590
time: 0.83 seconds

## BFS NS instance: NS_B1_B1_A1_A2_B2_A2

### Backsubstitution after applying NS history:
0: -44.6238937, 46.3271980, -43.3717232, 44.9401779, -89.5640640, 89.6979980
1: -313.9445801, 110.7814026, -303.9672852, 107.6393127, -420.6076660, 413.4206543
2: -175.7769775, 100.9844360, -170.4709625, 98.0167847, -272.9084167, 270.3331604
3: -220.2172699, 80.9449158, -213.3660736, 78.5642319, -297.3772583, 292.6127319
4: -126.8017273, 87.9186707, -123.1605453, 85.3021240, -211.6746979, 210.5055847

Time for backsubstitution: 1.43 seconds

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

Time for candidate selection: 0.11 seconds

### Candidate
type: A, layer: 1, pos: 9

## Relational analysis of NS_B1_B1_A1_A2_B2_A2_A1

### Relational analysis result of NS_B1_B1_A1_A2_B2_A2_A1
Status: Status.VERIFIED
Output dim: 0, lower bound: -63.3935634, upper bound: 63.3933890
time: 0.77 seconds

## Relational analysis of NS_B1_B1_A1_A2_B2_A2_A2

### Relational analysis result of NS_B1_B1_A1_A2_B2_A2_A2
Status: Status.UNKNOWN
Output dim: 0, lower bound: -63.3950467, upper bound: 63.3948832
time: 0.69 seconds

## BFS NS instance: NS_B1_B1_A2_B1_B1_A1

### Backsubstitution after applying NS history:
0: -52.4417572, 56.9100952, -30.8998337, 32.3675804, -84.8093338, 87.8099213
1: -386.6705322, 129.3537598, -219.2346802, 76.5160141, -462.0713806, 348.5883789
2: -206.8216400, 121.9994202, -122.6432190, 69.8588181, -276.6804504, 244.6426392
3: -266.7668457, 97.7206726, -153.3082123, 56.0272484, -322.3842163, 251.0288849
4: -148.6499023, 107.9011002, -88.1771545, 60.9467621, -209.5966339, 196.0782471

Time for backsubstitution: 1.30 seconds

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

### Candidate
type: B, layer: 1, pos: 3

### Candidate
type: B, layer: 1, pos: 45

## Relational analysis of NS_B1_B1_A2_B1_B1_A1_B1

### Relational analysis result of NS_B1_B1_A2_B1_B1_A1_B1
Status: Status.UNKNOWN
Output dim: 0, lower bound: -63.3941710, upper bound: 63.3937844
time: 0.76 seconds

## Relational analysis of NS_B1_B1_A2_B1_B1_A1_B2

### Relational analysis result of NS_B1_B1_A2_B1_B1_A1_B2
Status: Status.UNKNOWN
Output dim: 0, lower bound: -63.3941380, upper bound: 63.3937792
time: 0.72 seconds

## BFS NS instance: NS_B1_B1_A2_B1_B1_A2

### Backsubstitution after applying NS history:
0: -52.6876831, 57.4347305, -30.8998337, 32.3675804, -85.0552597, 88.3345642
1: -391.9079895, 130.1788025, -219.2346802, 76.5160141, -467.7626953, 349.4133911
2: -208.6153564, 123.0446472, -122.6432190, 69.8588181, -278.4741516, 245.6878662
3: -269.8934021, 98.6852036, -153.3082123, 56.0272484, -325.8593140, 251.9934082
4: -149.2469482, 108.9122238, -88.1771545, 60.9467621, -210.1937103, 197.0893860

Time for backsubstitution: 1.21 seconds

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

Time for candidate selection: 0.10 seconds

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
time: 0.75 seconds

## Relational analysis of NS_B1_B1_A2_B1_B1_A2_B2

### Relational analysis result of NS_B1_B1_A2_B1_B1_A2_B2
Status: Status.UNKNOWN
Output dim: 0, lower bound: -63.3941380, upper bound: 63.3940286
time: 0.68 seconds

## BFS NS instance: NS_B1_B1_A2_B1_B2_A1

### Backsubstitution after applying NS history:
0: -51.9699173, 56.3084450, -40.1849136, 41.3175163, -93.2874298, 96.3959198
1: -381.2426453, 128.1190796, -270.5454102, 99.4554672, -478.6930237, 398.6644897
2: -204.5754395, 120.6591721, -155.8203583, 89.4233932, -293.9988403, 275.6827698
3: -263.2272949, 96.6899796, -190.4992371, 71.5364838, -334.3562622, 287.0283813
4: -147.2113037, 106.7868118, -113.9260864, 78.2623062, -225.4736023, 219.7004089

Time for backsubstitution: 1.33 seconds

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

Time for candidate selection: 0.11 seconds

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
Status: Status.VERIFIED
Output dim: 0, lower bound: -63.3934695, upper bound: 63.3935876
time: 0.72 seconds

## Relational analysis of NS_B1_B1_A2_B1_B2_A1_A2

### Relational analysis result of NS_B1_B1_A2_B1_B2_A1_A2
Status: Status.VERIFIED
Output dim: 0, lower bound: -63.3936346, upper bound: 63.3936782
time: 0.73 seconds

## BFS NS instance: NS_B1_B1_A2_B1_B2_A2

### Backsubstitution after applying NS history:
0: -52.0833855, 56.6725082, -40.1849136, 41.3175163, -93.4009018, 96.7914581
1: -385.2969971, 128.5437622, -270.5454102, 99.4554672, -483.2335205, 399.0891113
2: -205.7564697, 121.3837280, -155.8203583, 89.4233932, -295.1798706, 276.4957275
3: -265.6109009, 97.3231201, -190.4992371, 71.5364838, -337.1045532, 287.7229919
4: -147.3715973, 107.5238266, -113.9260864, 78.2623062, -225.6338959, 220.4762268

Time for backsubstitution: 1.21 seconds

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

Time for candidate selection: 0.10 seconds

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
time: 0.73 seconds

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

Time for backsubstitution: 1.18 seconds

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

Time for candidate selection: 0.10 seconds

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
Status: Status.VERIFIED
Output dim: 0, lower bound: -63.3935559, upper bound: 63.3935835
time: 0.66 seconds

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

Time for backsubstitution: 1.39 seconds

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

Time for candidate selection: 0.11 seconds

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
time: 0.71 seconds

## Relational analysis of NS_B1_B1_A2_B2_B1_A2_A2

### Relational analysis result of NS_B1_B1_A2_B2_B1_A2_A2
Status: Status.UNKNOWN
Output dim: 0, lower bound: -63.3937217, upper bound: 63.3940841
time: 0.70 seconds

## BFS NS instance: NS_B1_B1_A2_B2_B2_A1

### Backsubstitution after applying NS history:
0: -51.9288788, 56.2117043, -43.8191643, 45.3083954, -97.2372742, 99.8938904
1: -380.5144348, 128.0065308, -299.1305847, 108.4494934, -486.8936768, 427.1370544
2: -204.4586792, 120.4639511, -171.1938019, 97.9577637, -302.4164124, 290.4625854
3: -262.8247681, 96.5425644, -210.4305420, 78.3455505, -340.7835999, 306.2767334
4: -147.1262665, 106.5851288, -124.5197067, 85.6121521, -232.7384186, 229.7925110

Time for backsubstitution: 1.26 seconds

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

Time for candidate selection: 0.11 seconds

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
Status: Status.VERIFIED
Output dim: 0, lower bound: -63.3934748, upper bound: 63.3935350
time: 0.75 seconds

## Relational analysis of NS_B1_B1_A2_B2_B2_A1_A2

### Relational analysis result of NS_B1_B1_A2_B2_B2_A1_A2
Status: Status.VERIFIED
Output dim: 0, lower bound: -63.3936405, upper bound: 63.3936336
time: 0.72 seconds

## BFS NS instance: NS_B1_B1_A2_B2_B2_A2

### Backsubstitution after applying NS history:
0: -51.9615936, 56.4974899, -43.8191643, 45.3083954, -97.2699890, 100.2106171
1: -383.9544983, 128.2036591, -299.1305847, 108.4494934, -490.8065186, 427.3342285
2: -205.1951294, 121.0046310, -171.1938019, 97.9577637, -303.1528931, 291.0913391
3: -264.7727661, 97.0129776, -210.4305420, 78.3455505, -343.0846252, 306.8079529
4: -147.0079651, 107.1813049, -124.5197067, 85.6121521, -232.6201172, 230.4271088

Time for backsubstitution: 1.20 seconds

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

Time for candidate selection: 0.11 seconds

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
time: 0.76 seconds

## Relational analysis of NS_B1_B1_A2_B2_B2_A2_A2

### Relational analysis result of NS_B1_B1_A2_B2_B2_A2_A2
Status: Status.UNKNOWN
Output dim: 0, lower bound: -63.3936405, upper bound: 63.3940548
time: 0.82 seconds

## BFS NS instance: NS_B1_B2_A1_A1_A1_A1

### Backsubstitution after applying NS history:
0: -29.5365906, 30.9115620, -29.8957691, 31.5139313, -61.0505219, 60.8073311
1: -209.6969757, 73.2576218, -213.3718414, 73.9482346, -283.6451721, 286.6294556
2: -117.0093307, 66.9293823, -117.9968338, 68.0444641, -185.0537720, 184.9262085
3: -146.5489655, 53.6311874, -148.1968842, 54.5233231, -201.0722809, 201.8280640
4: -84.2029343, 58.3682022, -84.9133301, 59.6252861, -143.8282166, 143.2815247

Time for backsubstitution: 1.30 seconds

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

Time for candidate selection: 0.11 seconds

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
time: 0.75 seconds

## Relational analysis of NS_B1_B2_A1_A1_A1_A1_B2

### Relational analysis result of NS_B1_B2_A1_A1_A1_A1_B2
Status: Status.UNKNOWN
Output dim: 0, lower bound: -63.3958122, upper bound: 63.3958702
time: 0.69 seconds

## BFS NS instance: NS_B1_B2_A1_A1_A1_A2

### Backsubstitution after applying NS history:
0: -29.2935848, 30.6672192, -29.2645760, 30.8694839, -60.1630707, 59.9317932
1: -207.5810242, 72.7790451, -208.9285583, 72.4774933, -280.0585327, 281.7075806
2: -115.9507675, 66.4279785, -115.4824677, 66.6106720, -182.5614319, 181.9104462
3: -145.0855560, 53.2599449, -145.1338348, 53.4042702, -198.4897919, 198.3937836
4: -83.5118332, 57.9175110, -83.1748199, 58.3728714, -141.8847046, 141.0923309

Time for backsubstitution: 1.44 seconds

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
time: 0.78 seconds

## Relational analysis of NS_B1_B2_A1_A1_A1_A2_B2

### Relational analysis result of NS_B1_B2_A1_A1_A1_A2_B2
Status: Status.UNKNOWN
Output dim: 0, lower bound: -63.3958122, upper bound: 63.3958754
time: 0.62 seconds

## BFS NS instance: NS_B1_B2_A1_A1_A2_B1

### Backsubstitution after applying NS history:
0: -43.3717232, 44.9401779, -30.1823769, 31.8147984, -75.1865234, 75.1225433
1: -303.9672852, 107.6393127, -214.8651276, 74.6638260, -378.6311035, 322.5044556
2: -170.4709625, 98.0167847, -119.0411377, 68.5830307, -238.5750427, 217.0579224
3: -213.3660736, 78.5642319, -149.3636475, 54.9800568, -267.1139526, 227.9278870
4: -123.1605453, 85.3021240, -85.7541580, 60.1135139, -183.0218353, 171.0562744

Time for backsubstitution: 1.30 seconds

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

Time for candidate selection: 0.12 seconds

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
time: 0.75 seconds

## Relational analysis of NS_B1_B2_A1_A1_A2_B1_B2

### Relational analysis result of NS_B1_B2_A1_A1_A2_B1_B2
Status: Status.UNKNOWN
Output dim: 0, lower bound: -63.3957590, upper bound: 63.3958450
time: 0.81 seconds

## BFS NS instance: NS_B1_B2_A1_A1_A2_B2

### Backsubstitution after applying NS history:
0: -43.3717232, 44.9401779, -44.6238937, 46.3271980, -89.6979904, 89.5640640
1: -303.9672852, 107.6393127, -313.9445801, 110.7814026, -413.4206848, 420.6076355
2: -170.4709625, 98.0167847, -175.7769775, 100.9844360, -270.3331604, 272.9084167
3: -213.3660736, 78.5642319, -220.2172699, 80.9449158, -292.6127319, 297.3772583
4: -123.1605453, 85.3021240, -126.8017273, 87.9186707, -210.5055847, 211.6747131

Time for backsubstitution: 1.29 seconds

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
Status: Status.VERIFIED
Output dim: 0, lower bound: -63.3933914, upper bound: 63.3935634
time: 0.66 seconds

## Relational analysis of NS_B1_B2_A1_A1_A2_B2_B2

### Relational analysis result of NS_B1_B2_A1_A1_A2_B2_B2
Status: Status.UNKNOWN
Output dim: 0, lower bound: -63.3948832, upper bound: 63.3950468
time: 0.78 seconds

## BFS NS instance: NS_B1_B2_A1_A2_B1_B1

### Backsubstitution after applying NS history:
0: -29.9004688, 31.5799618, -28.9615479, 30.5737095, -60.4741783, 60.5415115
1: -213.7520294, 73.9493179, -207.0983124, 71.5728531, -285.3248901, 281.0476379
2: -117.7341156, 68.0566254, -113.8917847, 65.9328995, -183.6670227, 181.9483795
3: -148.2603760, 54.5692177, -143.5340271, 52.8391800, -201.0995331, 198.1032410
4: -84.7556610, 59.6533127, -81.9813232, 57.7735481, -142.5291748, 141.6346130

Time for backsubstitution: 1.30 seconds

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

Time for candidate selection: 0.13 seconds

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
time: 0.78 seconds

## Relational analysis of NS_B1_B2_A1_A2_B1_B1_A2

### Relational analysis result of NS_B1_B2_A1_A2_B1_B1_A2
Status: Status.UNKNOWN
Output dim: 0, lower bound: -63.3938505, upper bound: 63.3932262
time: 0.69 seconds

## BFS NS instance: NS_B1_B2_A1_A2_B1_B2

### Backsubstitution after applying NS history:
0: -30.6833210, 32.3495865, -29.0219860, 30.6075287, -61.2908478, 61.3715591
1: -218.6328583, 75.9537354, -206.3379974, 71.8889771, -290.5218201, 282.2917175
2: -121.3715286, 69.8045197, -114.9485168, 66.0204163, -187.3919373, 184.7530060
3: -152.1492310, 55.9572372, -143.8393707, 52.9667053, -205.1159210, 199.7966003
4: -87.3655548, 61.1791573, -82.7645950, 57.8989754, -145.2645264, 143.9437408

Time for backsubstitution: 1.32 seconds

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

Time for candidate selection: 0.09 seconds

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
time: 0.74 seconds

## Relational analysis of NS_B1_B2_A1_A2_B1_B2_A2

### Relational analysis result of NS_B1_B2_A1_A2_B1_B2_A2
Status: Status.UNKNOWN
Output dim: 0, lower bound: -63.3939272, upper bound: 63.3947308
time: 0.80 seconds

## BFS NS instance: NS_B1_B2_A1_A2_B2_A1

### Backsubstitution after applying NS history:
0: -30.1823769, 31.8147984, -44.6238937, 46.3271980, -76.5095673, 76.4386902
1: -214.8651276, 74.6638260, -313.9445801, 110.7814026, -325.6465454, 388.6083984
2: -119.0411377, 68.5830307, -175.7769775, 100.9844360, -220.0255737, 244.3600159
3: -149.3636475, 54.9800568, -220.2172699, 80.9449158, -230.3085632, 275.1972656
4: -85.7541580, 60.1135139, -126.8017273, 87.9186707, -173.6728210, 186.9152374

Time for backsubstitution: 1.34 seconds

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

Time for candidate selection: 0.11 seconds

### Candidate
type: A, layer: 1, pos: 29

## Relational analysis of NS_B1_B2_A1_A2_B2_A1_A1
Optimization infeasible because this subproblem isn't reachable.

### Candidate
type: B, layer: 1, pos: 9

## Relational analysis of NS_B1_B2_A1_A2_B2_A1_B1

### Relational analysis result of NS_B1_B2_A1_A2_B2_A1_B1
Status: Status.VERIFIED
Output dim: 0, lower bound: -63.3927605, upper bound: 63.3926815
time: 0.71 seconds

## Relational analysis of NS_B1_B2_A1_A2_B2_A1_B2

### Relational analysis result of NS_B1_B2_A1_A2_B2_A1_B2
Status: Status.UNKNOWN
Output dim: 0, lower bound: -63.3942151, upper bound: 63.3942151
time: 0.79 seconds

## BFS NS instance: NS_B1_B2_A1_A2_B2_A2

### Backsubstitution after applying NS history:
0: -44.6238937, 46.3271980, -44.6238937, 46.3271980, -90.9510880, 90.9510880
1: -313.9445801, 110.7814026, -313.9445801, 110.7814026, -424.7259827, 424.7259827
2: -175.7769775, 100.9844360, -175.7769775, 100.9844360, -276.7614136, 276.7614136
3: -220.2172699, 80.9449158, -220.2172699, 80.9449158, -301.1621704, 301.1621704
4: -126.8017273, 87.9186707, -126.8017273, 87.9186707, -214.7203979, 214.7203979

Time for backsubstitution: 1.35 seconds

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

Time for candidate selection: 0.11 seconds

### Candidate
type: A, layer: 1, pos: 9

## Relational analysis of NS_B1_B2_A1_A2_B2_A2_A1

### Relational analysis result of NS_B1_B2_A1_A2_B2_A2_A1
Status: Status.VERIFIED
Output dim: 0, lower bound: -63.3935071, upper bound: 63.3933773
time: 1.22 seconds

## Relational analysis of NS_B1_B2_A1_A2_B2_A2_A2

### Relational analysis result of NS_B1_B2_A1_A2_B2_A2_A2
Status: Status.UNKNOWN
Output dim: 0, lower bound: -63.3942151, upper bound: 63.3942151
time: 0.75 seconds

## BFS NS instance: NS_B1_B2_A2_B1_B1_B1

### Backsubstitution after applying NS history:
0: -52.1692619, 56.9168472, -29.3263874, 30.9956894, -83.1649475, 86.2432098
1: -388.7693787, 128.8978271, -210.7784882, 72.5411301, -461.3104553, 339.6763000
2: -206.5955811, 121.9235916, -115.7335205, 66.8332214, -273.4288025, 237.6571045
3: -267.5608521, 97.7861481, -145.9232483, 53.5703850, -321.1312256, 243.7093658
4: -147.7406464, 107.9198608, -83.2501450, 58.5407257, -206.2813263, 191.1699677

Time for backsubstitution: 1.52 seconds

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

Time for candidate selection: 0.10 seconds

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
Status: Status.VERIFIED
Output dim: 0, lower bound: -63.3930653, upper bound: 63.3933406
time: 0.69 seconds

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
time: 0.87 seconds

## Relational analysis of NS_B1_B2_A2_B1_B1_B1_A2

### Relational analysis result of NS_B1_B2_A2_B1_B1_B1_A2
Status: Status.UNKNOWN
Output dim: 0, lower bound: -63.3935115, upper bound: 63.3940085
time: 0.70 seconds

## BFS NS instance: NS_B1_B2_A2_B1_B1_B2

### Backsubstitution after applying NS history:
0: -52.0881042, 56.7848129, -31.6310120, 33.5251617, -85.6132660, 88.4158173
1: -387.7741089, 128.6701813, -229.6852875, 78.3447952, -466.1188660, 358.3554688
2: -206.2195435, 121.6504974, -126.0184631, 72.2825851, -278.5021362, 247.6689606
3: -266.9662170, 97.5603180, -159.2089539, 57.9216194, -324.8878479, 256.7692566
4: -147.4947510, 107.6608124, -90.2369766, 63.2144165, -210.7091217, 197.8977814

Time for backsubstitution: 1.32 seconds

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

Time for candidate selection: 0.09 seconds

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
time: 0.82 seconds

## BFS NS instance: NS_B1_B2_A2_B2_B1_B1

### Backsubstitution after applying NS history:
0: -52.7859306, 57.4743652, -29.3598194, 30.9858704, -83.7717896, 86.8341827
1: -391.9863281, 130.3419800, -210.1095276, 72.7058563, -464.6921387, 340.4515076
2: -208.8900909, 123.1569595, -116.5262222, 66.8786011, -275.7686768, 239.6831818
3: -270.0416260, 98.7194366, -146.1479340, 53.5919342, -323.6334534, 244.8673401
4: -149.5399780, 108.9858856, -83.7368774, 58.5537834, -208.0937347, 192.7227631

Time for backsubstitution: 1.40 seconds

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

Time for candidate selection: 0.10 seconds

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
Status: Status.VERIFIED
Output dim: 0, lower bound: -63.3934536, upper bound: 63.3934069
time: 0.70 seconds

## Relational analysis of NS_B1_B2_A2_B2_B1_B1_A2

### Relational analysis result of NS_B1_B2_A2_B2_B1_B1_A2
Status: Status.UNKNOWN
Output dim: 0, lower bound: -63.3934536, upper bound: 63.3941899
time: 0.85 seconds

## BFS NS instance: NS_B1_B2_A2_B2_B2_B1

### Backsubstitution after applying NS history:
0: -52.7145691, 57.3530884, -32.1617851, 34.0683327, -86.7828903, 89.5148621
1: -391.0660706, 130.1454468, -232.7899628, 79.6942673, -470.7603455, 362.9353943
2: -208.5744019, 122.9083939, -128.2648773, 73.4637985, -282.0381775, 251.1732788
3: -269.5091858, 98.5153122, -161.6835480, 58.8755074, -328.3847046, 260.1988220
4: -149.3443909, 108.7455902, -91.9323349, 64.2209549, -213.5653076, 200.6779175

Time for backsubstitution: 1.30 seconds

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

Time for candidate selection: 0.11 seconds

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
Status: Status.VERIFIED
Output dim: 0, lower bound: -63.3932456, upper bound: 63.3932652
time: 0.74 seconds

## Relational analysis of NS_B1_B2_A2_B2_B2_B1_A2

### Relational analysis result of NS_B1_B2_A2_B2_B2_B1_A2
Status: Status.UNKNOWN
Output dim: 0, lower bound: -63.3932456, upper bound: 63.3941989
time: 0.71 seconds

## BFS NS instance: NS_B1_B2_A2_B2_B2_B2

### Backsubstitution after applying NS history:
0: -52.1011620, 56.6095123, -42.7220459, 44.2243690, -96.3255157, 99.3315582
1: -384.4843140, 128.5108948, -291.8194580, 105.7542801, -490.2385864, 420.3302612
2: -205.6697693, 121.2567215, -167.0193329, 95.5304642, -301.2002258, 288.2760010
3: -265.2196350, 97.1680374, -205.2453308, 76.4457169, -341.6653442, 302.4133606
4: -147.4522400, 107.3706512, -121.4646149, 83.5994034, -231.0516357, 228.8352509

Time for backsubstitution: 1.29 seconds

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

Time for candidate selection: 0.11 seconds

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
Status: Status.VERIFIED
Output dim: 0, lower bound: -63.3931347, upper bound: 63.3930939
time: 0.78 seconds

## Relational analysis of NS_B1_B2_A2_B2_B2_B2_A2

### Relational analysis result of NS_B1_B2_A2_B2_B2_B2_A2
Status: Status.UNKNOWN
Output dim: 0, lower bound: -63.3929721, upper bound: 63.3937700
time: 0.83 seconds

## BFS NS instance: NS_B2_A2_A1_A1_B1_A1

### Backsubstitution after applying NS history:
0: -28.5664749, 30.1689701, -51.8698730, 56.3319435, -84.8984222, 82.0388412
1: -205.3779755, 70.8297882, -382.8943176, 127.9916611, -333.3695984, 453.0242920
2: -113.2364197, 65.0554352, -204.6689301, 120.7612076, -233.9976196, 269.7243347
3: -142.5589294, 52.1684952, -264.0834351, 96.7237778, -239.2827148, 315.9387512
4: -81.4878922, 56.8995934, -147.0722046, 106.7969894, -188.2848816, 203.9717865

Time for backsubstitution: 1.35 seconds

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

Time for candidate selection: 0.11 seconds

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
time: 0.97 seconds

## Relational analysis of NS_B2_A2_A1_A1_B1_A1_B2

### Relational analysis result of NS_B2_A2_A1_A1_B1_A1_B2
Status: Status.UNKNOWN
Output dim: 0, lower bound: -63.3937463, upper bound: 63.3942170
time: 1.01 seconds

## BFS NS instance: NS_B2_A2_A1_A1_B1_A2

### Backsubstitution after applying NS history:
0: -31.3312283, 33.2471046, -49.0097733, 53.2279167, -84.5591354, 82.2568817
1: -229.0442810, 77.7797470, -362.4377441, 120.9739456, -350.0182190, 439.6603699
2: -125.1618118, 72.0901031, -193.7615662, 114.1725082, -239.3343201, 265.8516235
3: -158.3682098, 57.6655502, -250.0415344, 91.4295197, -249.7977295, 307.4492188
4: -89.2824326, 62.8486900, -139.2727051, 101.0217896, -190.3041992, 202.1213837

Time for backsubstitution: 1.32 seconds

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

Time for candidate selection: 0.12 seconds

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
time: 0.66 seconds

## BFS NS instance: NS_B2_A2_A1_A1_B2_A1

### Backsubstitution after applying NS history:
0: -28.5664749, 30.1689701, -52.2746696, 56.9984779, -85.5649490, 82.4436417
1: -205.3779755, 70.8297882, -389.0621033, 129.1828613, -334.5608521, 459.8918457
2: -113.2364197, 65.0554352, -207.0110779, 122.1272202, -235.3636322, 272.0664978
3: -142.5589294, 52.1684952, -267.8845215, 97.9457474, -240.5046692, 320.0530090
4: -81.4878922, 56.8995934, -148.0931091, 108.0790558, -189.5669403, 204.9926910

Time for backsubstitution: 1.35 seconds

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

Time for candidate selection: 0.11 seconds

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
time: 0.87 seconds

## Relational analysis of NS_B2_A2_A1_A1_B2_A1_B2

### Relational analysis result of NS_B2_A2_A1_A1_B2_A1_B2
Status: Status.UNKNOWN
Output dim: 0, lower bound: -63.3942786, upper bound: 63.3945493
time: 0.77 seconds

## BFS NS instance: NS_B2_A2_A1_A1_B2_A2

### Backsubstitution after applying NS history:
0: -31.3312283, 33.2471046, -49.1084404, 53.5609856, -84.8922119, 82.3555450
1: -229.0442810, 77.7797470, -365.8971252, 121.2925720, -350.3368530, 443.6768799
2: -125.1618118, 72.0901031, -194.7173767, 114.7962875, -239.9580841, 266.8074646
3: -158.3682098, 57.6655502, -252.0432892, 91.9607239, -250.3289337, 309.7088318
4: -89.2824326, 62.8486900, -139.2741394, 101.6966629, -190.9790955, 202.1228333

Time for backsubstitution: 1.24 seconds

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

Time for candidate selection: 0.11 seconds

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
time: 0.75 seconds

## Relational analysis of NS_B2_A2_A1_A1_B2_A2_B2

### Relational analysis result of NS_B2_A2_A1_A1_B2_A2_B2
Status: Status.UNKNOWN
Output dim: 0, lower bound: -63.3942575, upper bound: 63.3945567
time: 0.75 seconds

## BFS NS instance: NS_B2_A2_A1_A2_B1_B1

### Backsubstitution after applying NS history:
0: -41.3362274, 42.5331116, -52.3831139, 56.7113609, -98.0101166, 94.9162064
1: -279.1030579, 102.3374252, -383.6714172, 129.1607971, -408.2638550, 484.1629639
2: -160.6312256, 92.0329819, -206.2644196, 121.5206833, -281.6412659, 298.2973938
3: -196.5596924, 73.6476669, -265.0341492, 97.3871384, -293.9468384, 338.3572083
4: -117.3278351, 80.5484772, -148.4291382, 107.5487595, -224.0611420, 228.9776154

Time for backsubstitution: 1.45 seconds

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

Time for candidate selection: 0.11 seconds

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
Status: Status.VERIFIED
Output dim: 0, lower bound: -63.3931573, upper bound: 63.3932422
time: 0.82 seconds

## Relational analysis of NS_B2_A2_A1_A2_B1_B1_A2

### Relational analysis result of NS_B2_A2_A1_A2_B1_B1_A2
Status: Status.UNKNOWN
Output dim: 0, lower bound: -63.3931573, upper bound: 63.3940319
time: 0.76 seconds

## BFS NS instance: NS_B2_A2_A1_A2_B1_B2

### Backsubstitution after applying NS history:
0: -41.4766655, 42.6777725, -52.2842216, 56.6177788, -98.0716095, 94.9619904
1: -280.1462708, 102.6841431, -383.2715454, 128.8799133, -409.0261230, 484.1998596
2: -161.1961517, 92.3508987, -205.7797241, 121.3362656, -282.0737610, 298.1306152
3: -197.2792816, 73.9040070, -264.6797180, 97.2263565, -294.5056458, 338.3065186
4: -117.7232513, 80.8190765, -148.0908661, 107.3788910, -224.3143158, 228.9099426

Time for backsubstitution: 1.52 seconds

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

Time for candidate selection: 0.11 seconds

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
Status: Status.VERIFIED
Output dim: 0, lower bound: -63.3932236, upper bound: 63.3933818
time: 0.79 seconds

## Relational analysis of NS_B2_A2_A1_A2_B1_B2_A2

### Relational analysis result of NS_B2_A2_A1_A2_B1_B2_A2
Status: Status.UNKNOWN
Output dim: 0, lower bound: -63.3932236, upper bound: 63.3941691
time: 0.80 seconds

## BFS NS instance: NS_B2_A2_A1_A2_B2_B1

### Backsubstitution after applying NS history:
0: -41.3362274, 42.5331116, -52.5613327, 57.1557770, -98.4919968, 95.0944443
1: -279.1030579, 102.3374252, -388.3494568, 129.7206573, -408.8237305, 490.6868896
2: -160.6312256, 92.0329819, -207.6318054, 122.4168320, -283.0480347, 299.6647949
3: -196.5596924, 73.6476669, -267.8355713, 98.1381149, -294.6978149, 341.4832458
4: -117.3278351, 80.5484772, -148.7674866, 108.4405518, -225.7683716, 229.3159637

Time for backsubstitution: 1.46 seconds

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

Time for candidate selection: 0.10 seconds

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
Status: Status.VERIFIED
Output dim: 0, lower bound: -63.3931573, upper bound: 63.3932422
time: 0.81 seconds

## Relational analysis of NS_B2_A2_A1_A2_B2_B1_A2

### Relational analysis result of NS_B2_A2_A1_A2_B2_B1_A2
Status: Status.UNKNOWN
Output dim: 0, lower bound: -63.3931573, upper bound: 63.3941582
time: 0.75 seconds

## BFS NS instance: NS_B2_A2_A1_A2_B2_B2

### Backsubstitution after applying NS history:
0: -41.4766655, 42.6777725, -52.4143295, 56.9992371, -98.4758759, 95.0921021
1: -280.1462708, 102.6841431, -387.3854065, 129.3546753, -409.5008850, 490.0695496
2: -161.1961517, 92.3508987, -207.0046539, 122.0845871, -283.2807312, 299.3555298
3: -197.2792816, 73.9040070, -267.1123047, 97.8885498, -295.1678467, 341.0162964
4: -117.7232513, 80.8190765, -148.2979126, 108.1422577, -225.8655090, 229.1169891

Time for backsubstitution: 1.28 seconds

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
Status: Status.VERIFIED
Output dim: 0, lower bound: -63.3932285, upper bound: 63.3933819
time: 1.28 seconds

## Relational analysis of NS_B2_A2_A1_A2_B2_B2_A2

### Relational analysis result of NS_B2_A2_A1_A2_B2_B2_A2
Status: Status.UNKNOWN
Output dim: 0, lower bound: -63.3932285, upper bound: 63.3947862
time: 0.84 seconds

## BFS NS instance: NS_B2_A2_A2_B1_A2_B1

### Backsubstitution after applying NS history:
0: -53.8847351, 58.6575432, -53.9392204, 58.4476395, -112.3323593, 112.5967484
1: -399.7374878, 133.0946503, -396.4638672, 133.0583038, -532.1393433, 528.5483398
2: -213.1640167, 125.7145767, -212.5763245, 125.3313828, -338.4953613, 338.2908936
3: -275.4341431, 100.8090820, -273.6869812, 100.4134979, -375.1802063, 373.5529480
4: -152.5790253, 111.2873230, -152.8955994, 110.8564682, -263.4354858, 264.1829224

Time for backsubstitution: 1.23 seconds

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

Time for backsubstitution: 1.38 seconds

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

Time for candidate selection: 0.11 seconds

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

## BFS NS instance: NS_B2_A2_A2_B2_A2_B1

### Backsubstitution after applying NS history:
0: -53.8847351, 58.6575432, -54.1881714, 58.9965973, -112.8813248, 112.8457184
1: -399.7374878, 133.0946503, -402.2130432, 133.8377991, -533.5752563, 535.3076782
2: -213.1640167, 125.7145767, -214.3997955, 126.4415131, -339.6054993, 340.1143799
3: -275.4341431, 100.8090820, -277.1029663, 101.3699265, -376.8040466, 377.9120483
4: -152.5790253, 111.2873230, -153.4725342, 111.9196091, -264.4985962, 264.7598572

Time for backsubstitution: 1.29 seconds

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

Time for candidate selection: 0.13 seconds

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

### Candidate
type: B, layer: 1, pos: 39

### Candidate
type: B, layer: 1, pos: 41

### Candidate
type: B, layer: 1, pos: 38

## Relational analysis of NS_B2_A2_A2_B2_A2_B1_B1
Optimization infeasible because this subproblem isn't reachable.

### Candidate
type: A, layer: 1, pos: 38

### Candidate
type: A, layer: 1, pos: 16

## Relational analysis of NS_B2_A2_A2_B2_A2_B1_A1
Optimization infeasible because this subproblem isn't reachable.

### Candidate
type: B, layer: 1, pos: 16

### Candidate
type: B, layer: 1, pos: 5

## Relational analysis of NS_B2_A2_A2_B2_A2_B1_B1
Optimization infeasible because this subproblem isn't reachable.

### Candidate
type: A, layer: 1, pos: 5

### Candidate
type: B, layer: 1, pos: 4

### Candidate
type: A, layer: 1, pos: 4

## BFS NS instance: NS_B2_A2_A2_B2_A2_B2

### Backsubstitution after applying NS history:
0: -53.8847351, 58.6575432, -53.8847351, 58.6575432, -112.5422668, 112.5422668
1: -399.7374878, 133.0946503, -399.7374878, 133.0946503, -532.8320923, 532.8320923
2: -213.1640167, 125.7145767, -213.1640167, 125.7145767, -338.8785706, 338.8785706
3: -275.4341431, 100.8090820, -275.4341431, 100.8090820, -376.2432251, 376.2432251
4: -152.5790253, 111.2873230, -152.5790253, 111.2873230, -263.8663025, 263.8663025

Time for backsubstitution: 1.25 seconds

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
type: A, layer: 1, pos: 5
type: B, layer: 1, pos: 5
type: B, layer: 1, pos: 4
type: A, layer: 1, pos: 4

Time for candidate selection: 0.10 seconds

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
type: B, layer: 1, pos: 17

### Candidate
type: A, layer: 1, pos: 17

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

## Relational analysis of NS_B2_A2_A2_B2_A2_B2_B1
Optimization infeasible because this subproblem isn't reachable.

### Candidate
type: A, layer: 1, pos: 15

## Relational analysis of NS_B2_A2_A2_B2_A2_B2_A1

### Relational analysis result of NS_B2_A2_A2_B2_A2_B2_A1
Status: Status.VERIFIED
Output dim: 0, lower bound: -63.3903239, upper bound: 63.3934325
time: 0.86 seconds

## Relational analysis of NS_B2_A2_A2_B2_A2_B2_A2

### Relational analysis result of NS_B2_A2_A2_B2_A2_B2_A2
Status: Status.UNKNOWN
Output dim: 0, lower bound: -63.3903239, upper bound: 63.3944484
time: 0.82 seconds

## Summary of splitting at layer (split count: 6)
- Time for NS candidates: 3.44 seconds
NS_B1_B1_A1_A1_B1_A1_B1, status: Status.UNKNOWN, split count: 7, time: 3.44
Output dim: 0, lower bound: -63.3964140, upper bound: 63.3964064
NS_B1_B1_A1_A1_B1_A1_B2, status: Status.UNKNOWN, split count: 7, time: 3.44
Output dim: 0, lower bound: -63.3964120, upper bound: 63.3964073
NS_B1_B1_A1_A1_B1_A2_B1, status: Status.UNKNOWN, split count: 7, time: 3.44
Output dim: 0, lower bound: -63.3964140, upper bound: 63.3964064
NS_B1_B1_A1_A1_B1_A2_B2, status: Status.UNKNOWN, split count: 7, time: 3.44
Output dim: 0, lower bound: -63.3964120, upper bound: 63.3964073
NS_B1_B1_A1_A1_B2_A1_A1, status: Status.UNKNOWN, split count: 7, time: 3.44
Output dim: 0, lower bound: -63.3959779, upper bound: 63.3960336
NS_B1_B1_A1_A1_B2_A1_A2, status: Status.UNKNOWN, split count: 7, time: 3.44
Output dim: 0, lower bound: -63.3956942, upper bound: 63.3956942
NS_B1_B1_A1_A1_B2_A2_B1, status: Status.UNKNOWN, split count: 7, time: 3.44
Output dim: 0, lower bound: -63.3954147, upper bound: 63.3953984
NS_B1_B1_A1_A1_B2_A2_B2, status: Status.UNKNOWN, split count: 7, time: 3.44
Output dim: 0, lower bound: -63.3954147, upper bound: 63.3961970
NS_B1_B1_A1_A2_B1_B1_A1, status: Status.UNKNOWN, split count: 7, time: 3.44
Output dim: 0, lower bound: -63.3958702, upper bound: 63.3958122
NS_B1_B1_A1_A2_B1_B1_A2, status: Status.UNKNOWN, split count: 7, time: 3.44
Output dim: 0, lower bound: -63.3958702, upper bound: 63.3958122
NS_B1_B1_A1_A2_B1_B2_A1, status: Status.UNKNOWN, split count: 7, time: 3.44
Output dim: 0, lower bound: -63.3958754, upper bound: 63.3958122
NS_B1_B1_A1_A2_B1_B2_A2, status: Status.UNKNOWN, split count: 7, time: 3.44
Output dim: 0, lower bound: -63.3958754, upper bound: 63.3958122
NS_B1_B1_A1_A2_B2_A1_A1, status: Status.UNKNOWN, split count: 7, time: 3.44
Output dim: 0, lower bound: -63.3955360, upper bound: 63.3953415
NS_B1_B1_A1_A2_B2_A1_A2, status: Status.UNKNOWN, split count: 7, time: 3.44
Output dim: 0, lower bound: -63.3955360, upper bound: 63.3957590
NS_B1_B1_A1_A2_B2_A2_A1, status: Status.VERIFIED, split count: 7, time: 3.44
Output dim: 0, lower bound: -63.3935634, upper bound: 63.3933890
NS_B1_B1_A1_A2_B2_A2_A2, status: Status.UNKNOWN, split count: 7, time: 3.44
Output dim: 0, lower bound: -63.3950467, upper bound: 63.3948832
NS_B1_B1_A2_B1_B1_A1_B1, status: Status.UNKNOWN, split count: 7, time: 3.44
Output dim: 0, lower bound: -63.3941710, upper bound: 63.3937844
NS_B1_B1_A2_B1_B1_A1_B2, status: Status.UNKNOWN, split count: 7, time: 3.44
Output dim: 0, lower bound: -63.3941380, upper bound: 63.3937792
NS_B1_B1_A2_B1_B1_A2_B1, status: Status.UNKNOWN, split count: 7, time: 3.44
Output dim: 0, lower bound: -63.3941710, upper bound: 63.3938891
NS_B1_B1_A2_B1_B1_A2_B2, status: Status.UNKNOWN, split count: 7, time: 3.44
Output dim: 0, lower bound: -63.3941380, upper bound: 63.3940286
NS_B1_B1_A2_B1_B2_A1_A1, status: Status.VERIFIED, split count: 7, time: 3.44
Output dim: 0, lower bound: -63.3934695, upper bound: 63.3935876
NS_B1_B1_A2_B1_B2_A1_A2, status: Status.VERIFIED, split count: 7, time: 3.44
Output dim: 0, lower bound: -63.3936346, upper bound: 63.3936782
NS_B1_B1_A2_B1_B2_A2_A1, status: Status.UNKNOWN, split count: 7, time: 3.44
Output dim: 0, lower bound: -63.3934695, upper bound: 63.3937074
NS_B1_B1_A2_B1_B2_A2_A2, status: Status.UNKNOWN, split count: 7, time: 3.44
Output dim: 0, lower bound: -63.3934695, upper bound: 63.3938612
NS_B1_B1_A2_B2_B1_A1_A1, status: Status.VERIFIED, split count: 7, time: 3.44
Output dim: 0, lower bound: -63.3935559, upper bound: 63.3935835
NS_B1_B1_A2_B2_B1_A1_A2, status: Status.UNKNOWN, split count: 7, time: 3.44
Output dim: 0, lower bound: -63.3937217, upper bound: 63.3936811
NS_B1_B1_A2_B2_B1_A2_A1, status: Status.UNKNOWN, split count: 7, time: 3.44
Output dim: 0, lower bound: -63.3935559, upper bound: 63.3937464
NS_B1_B1_A2_B2_B1_A2_A2, status: Status.UNKNOWN, split count: 7, time: 3.44
Output dim: 0, lower bound: -63.3937217, upper bound: 63.3940841
NS_B1_B1_A2_B2_B2_A1_A1, status: Status.VERIFIED, split count: 7, time: 3.44
Output dim: 0, lower bound: -63.3934748, upper bound: 63.3935350
NS_B1_B1_A2_B2_B2_A1_A2, status: Status.VERIFIED, split count: 7, time: 3.44
Output dim: 0, lower bound: -63.3936405, upper bound: 63.3936336
NS_B1_B1_A2_B2_B2_A2_A1, status: Status.UNKNOWN, split count: 7, time: 3.44
Output dim: 0, lower bound: -63.3934748, upper bound: 63.3936938
NS_B1_B1_A2_B2_B2_A2_A2, status: Status.UNKNOWN, split count: 7, time: 3.44
Output dim: 0, lower bound: -63.3936405, upper bound: 63.3940548
NS_B1_B2_A1_A1_A1_A1_B1, status: Status.UNKNOWN, split count: 7, time: 3.44
Output dim: 0, lower bound: -63.3958122, upper bound: 63.3958702
NS_B1_B2_A1_A1_A1_A1_B2, status: Status.UNKNOWN, split count: 7, time: 3.44
Output dim: 0, lower bound: -63.3958122, upper bound: 63.3958702
NS_B1_B2_A1_A1_A1_A2_B1, status: Status.UNKNOWN, split count: 7, time: 3.44
Output dim: 0, lower bound: -63.3958122, upper bound: 63.3958754
NS_B1_B2_A1_A1_A1_A2_B2, status: Status.UNKNOWN, split count: 7, time: 3.44
Output dim: 0, lower bound: -63.3958122, upper bound: 63.3958754
NS_B1_B2_A1_A1_A2_B1_B1, status: Status.UNKNOWN, split count: 7, time: 3.44
Output dim: 0, lower bound: -63.3953415, upper bound: 63.3955360
NS_B1_B2_A1_A1_A2_B1_B2, status: Status.UNKNOWN, split count: 7, time: 3.44
Output dim: 0, lower bound: -63.3957590, upper bound: 63.3958450
NS_B1_B2_A1_A1_A2_B2_B1, status: Status.VERIFIED, split count: 7, time: 3.44
Output dim: 0, lower bound: -63.3933914, upper bound: 63.3935634
NS_B1_B2_A1_A1_A2_B2_B2, status: Status.UNKNOWN, split count: 7, time: 3.44
Output dim: 0, lower bound: -63.3948832, upper bound: 63.3950468
NS_B1_B2_A1_A2_B1_B1_A1, status: Status.UNKNOWN, split count: 7, time: 3.44
Output dim: 0, lower bound: -63.3938505, upper bound: 63.3932262
NS_B1_B2_A1_A2_B1_B1_A2, status: Status.UNKNOWN, split count: 7, time: 3.44
Output dim: 0, lower bound: -63.3938505, upper bound: 63.3932262
NS_B1_B2_A1_A2_B1_B2_A1, status: Status.UNKNOWN, split count: 7, time: 3.44
Output dim: 0, lower bound: -63.3939272, upper bound: 63.3934434
NS_B1_B2_A1_A2_B1_B2_A2, status: Status.UNKNOWN, split count: 7, time: 3.44
Output dim: 0, lower bound: -63.3939272, upper bound: 63.3947308
NS_B1_B2_A1_A2_B2_A1_B1, status: Status.VERIFIED, split count: 7, time: 3.44
Output dim: 0, lower bound: -63.3927605, upper bound: 63.3926815
NS_B1_B2_A1_A2_B2_A1_B2, status: Status.UNKNOWN, split count: 7, time: 3.44
Output dim: 0, lower bound: -63.3942151, upper bound: 63.3942151
NS_B1_B2_A1_A2_B2_A2_A1, status: Status.VERIFIED, split count: 7, time: 3.44
Output dim: 0, lower bound: -63.3935071, upper bound: 63.3933773
NS_B1_B2_A1_A2_B2_A2_A2, status: Status.UNKNOWN, split count: 7, time: 3.44
Output dim: 0, lower bound: -63.3942151, upper bound: 63.3942151
NS_B1_B2_A2_B1_B1_B1_A1, status: Status.UNKNOWN, split count: 7, time: 3.44
Output dim: 0, lower bound: -63.3935115, upper bound: 63.3939136
NS_B1_B2_A2_B1_B1_B1_A2, status: Status.UNKNOWN, split count: 7, time: 3.44
Output dim: 0, lower bound: -63.3935115, upper bound: 63.3940085
NS_B1_B2_A2_B1_B1_B2_A1, status: Status.UNKNOWN, split count: 7, time: 3.44
Output dim: 0, lower bound: -63.3935115, upper bound: 63.3940906
NS_B1_B2_A2_B1_B1_B2_A2, status: Status.UNKNOWN, split count: 7, time: 3.44
Output dim: 0, lower bound: -63.3939212, upper bound: 63.3941125
NS_B1_B2_A2_B2_B1_B1_A1, status: Status.VERIFIED, split count: 7, time: 3.44
Output dim: 0, lower bound: -63.3934536, upper bound: 63.3934069
NS_B1_B2_A2_B2_B1_B1_A2, status: Status.UNKNOWN, split count: 7, time: 3.44
Output dim: 0, lower bound: -63.3934536, upper bound: 63.3941899
NS_B1_B2_A2_B2_B2_B1_A1, status: Status.VERIFIED, split count: 7, time: 3.44
Output dim: 0, lower bound: -63.3932456, upper bound: 63.3932652
NS_B1_B2_A2_B2_B2_B1_A2, status: Status.UNKNOWN, split count: 7, time: 3.44
Output dim: 0, lower bound: -63.3932456, upper bound: 63.3941989
NS_B1_B2_A2_B2_B2_B2_A1, status: Status.VERIFIED, split count: 7, time: 3.44
Output dim: 0, lower bound: -63.3931347, upper bound: 63.3930939
NS_B1_B2_A2_B2_B2_B2_A2, status: Status.UNKNOWN, split count: 7, time: 3.44
Output dim: 0, lower bound: -63.3929721, upper bound: 63.3937700
NS_B2_A2_A1_A1_B1_A1_B1, status: Status.UNKNOWN, split count: 7, time: 3.44
Output dim: 0, lower bound: -63.3936520, upper bound: 63.3940690
NS_B2_A2_A1_A1_B1_A1_B2, status: Status.UNKNOWN, split count: 7, time: 3.44
Output dim: 0, lower bound: -63.3937463, upper bound: 63.3942170
NS_B2_A2_A1_A1_B1_A2_B1, status: Status.UNKNOWN, split count: 7, time: 3.44
Output dim: 0, lower bound: -63.3936447, upper bound: 63.3940670
NS_B2_A2_A1_A1_B1_A2_B2, status: Status.UNKNOWN, split count: 7, time: 3.44
Output dim: 0, lower bound: -63.3937338, upper bound: 63.3942133
NS_B2_A2_A1_A1_B2_A1_B1, status: Status.UNKNOWN, split count: 7, time: 3.44
Output dim: 0, lower bound: -63.3937633, upper bound: 63.3942024
NS_B2_A2_A1_A1_B2_A1_B2, status: Status.UNKNOWN, split count: 7, time: 3.44
Output dim: 0, lower bound: -63.3942786, upper bound: 63.3945493
NS_B2_A2_A1_A1_B2_A2_B1, status: Status.UNKNOWN, split count: 7, time: 3.44
Output dim: 0, lower bound: -63.3937664, upper bound: 63.3942169
NS_B2_A2_A1_A1_B2_A2_B2, status: Status.UNKNOWN, split count: 7, time: 3.44
Output dim: 0, lower bound: -63.3942575, upper bound: 63.3945567
NS_B2_A2_A1_A2_B1_B1_A1, status: Status.VERIFIED, split count: 7, time: 3.44
Output dim: 0, lower bound: -63.3931573, upper bound: 63.3932422
NS_B2_A2_A1_A2_B1_B1_A2, status: Status.UNKNOWN, split count: 7, time: 3.44
Output dim: 0, lower bound: -63.3931573, upper bound: 63.3940319
NS_B2_A2_A1_A2_B1_B2_A1, status: Status.VERIFIED, split count: 7, time: 3.44
Output dim: 0, lower bound: -63.3932236, upper bound: 63.3933818
NS_B2_A2_A1_A2_B1_B2_A2, status: Status.UNKNOWN, split count: 7, time: 3.44
Output dim: 0, lower bound: -63.3932236, upper bound: 63.3941691
NS_B2_A2_A1_A2_B2_B1_A1, status: Status.VERIFIED, split count: 7, time: 3.44
Output dim: 0, lower bound: -63.3931573, upper bound: 63.3932422
NS_B2_A2_A1_A2_B2_B1_A2, status: Status.UNKNOWN, split count: 7, time: 3.44
Output dim: 0, lower bound: -63.3931573, upper bound: 63.3941582
NS_B2_A2_A1_A2_B2_B2_A1, status: Status.VERIFIED, split count: 7, time: 3.44
Output dim: 0, lower bound: -63.3932285, upper bound: 63.3933819
NS_B2_A2_A1_A2_B2_B2_A2, status: Status.UNKNOWN, split count: 7, time: 3.44
Output dim: 0, lower bound: -63.3932285, upper bound: 63.3947862
NS_B2_A2_A2_B2_A2_B2_A1, status: Status.VERIFIED, split count: 7, time: 3.44
Output dim: 0, lower bound: -63.3903239, upper bound: 63.3934325
NS_B2_A2_A2_B2_A2_B2_A2, status: Status.UNKNOWN, split count: 7, time: 3.44
Output dim: 0, lower bound: -63.3903239, upper bound: 63.3944484

## BFS NS instance: NS_B1_B1_A1_A1_B1_A1_B1

### Backsubstitution after applying NS history:
0: -30.1897888, 31.5982895, -29.5365906, 30.9115620, -61.1013489, 61.1348801
1: -214.1289520, 74.8528442, -209.6969757, 73.2576218, -287.3865662, 284.5497437
2: -119.5830994, 68.3489609, -117.0093307, 66.9293823, -186.5124664, 185.3582611
3: -149.6831360, 54.7830582, -146.5489655, 53.6311874, -203.3143311, 201.3320312
4: -86.0862274, 59.6165848, -84.2029343, 58.3682022, -144.4544373, 143.8195038

Time for backsubstitution: 1.35 seconds

### NS candidates at layer 1
type: A, layer: 1, pos: 3
type: B, layer: 1, pos: 9
type: A, layer: 1, pos: 9
type: A, layer: 1, pos: 35
type: B, layer: 1, pos: 35
type: B, layer: 1, pos: 17
type: A, layer: 1, pos: 17
type: A, layer: 1, pos: 19
type: B, layer: 1, pos: 19
type: A, layer: 1, pos: 45
type: B, layer: 1, pos: 45
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
type: B, layer: 1, pos: 39
type: A, layer: 1, pos: 43
type: A, layer: 1, pos: 38
type: A, layer: 1, pos: 32
type: B, layer: 1, pos: 43
type: A, layer: 1, pos: 37
type: B, layer: 1, pos: 38
type: B, layer: 1, pos: 29
type: B, layer: 1, pos: 37
type: B, layer: 1, pos: 41
type: A, layer: 1, pos: 41
type: B, layer: 1, pos: 20
type: A, layer: 1, pos: 29
type: A, layer: 1, pos: 20
type: B, layer: 1, pos: 21
type: B, layer: 1, pos: 5
type: A, layer: 1, pos: 5
type: A, layer: 1, pos: 21
type: A, layer: 1, pos: 23
type: B, layer: 1, pos: 23
type: B, layer: 1, pos: 16
type: A, layer: 1, pos: 16

Time for candidate selection: 0.12 seconds

### Candidate
type: A, layer: 1, pos: 3

## Relational analysis of NS_B1_B1_A1_A1_B1_A1_B1_A1

### Relational analysis result of NS_B1_B1_A1_A1_B1_A1_B1_A1
Status: Status.UNKNOWN
Output dim: 0, lower bound: -63.3964605, upper bound: 63.3964605
time: 0.76 seconds

## Relational analysis of NS_B1_B1_A1_A1_B1_A1_B1_A2

### Relational analysis result of NS_B1_B1_A1_A1_B1_A1_B1_A2
Status: Status.UNKNOWN
Output dim: 0, lower bound: -63.3964605, upper bound: 63.3964605
time: 0.79 seconds

## BFS NS instance: NS_B1_B1_A1_A1_B1_A1_B2

### Backsubstitution after applying NS history:
0: -28.6635017, 30.0046329, -29.2935848, 30.6672192, -59.3307152, 59.2982178
1: -203.0244446, 71.1200867, -207.5810242, 72.7790451, -275.8034973, 278.7011108
2: -113.5666733, 64.8913727, -115.9507675, 66.4279785, -179.9946442, 180.8421173
3: -142.0295868, 52.0181732, -145.0855560, 53.2599449, -195.2895355, 197.1037140
4: -81.7767563, 56.6121597, -83.5118332, 57.9175110, -139.6942291, 140.1239777

Time for backsubstitution: 1.30 seconds

### NS candidates at layer 1
type: A, layer: 1, pos: 3
type: A, layer: 1, pos: 9
type: B, layer: 1, pos: 9
type: B, layer: 1, pos: 35
type: A, layer: 1, pos: 45
type: A, layer: 1, pos: 35
type: A, layer: 1, pos: 17
type: A, layer: 1, pos: 19
type: B, layer: 1, pos: 17
type: B, layer: 1, pos: 19
type: B, layer: 1, pos: 11
type: A, layer: 1, pos: 11
type: B, layer: 1, pos: 10
type: A, layer: 1, pos: 10
type: B, layer: 1, pos: 15
type: B, layer: 1, pos: 45
type: B, layer: 1, pos: 18
type: A, layer: 1, pos: 38
type: A, layer: 1, pos: 15
type: A, layer: 1, pos: 18
type: A, layer: 1, pos: 29
type: B, layer: 1, pos: 32
type: B, layer: 1, pos: 39
type: B, layer: 1, pos: 43
type: A, layer: 1, pos: 37
type: A, layer: 1, pos: 39
type: A, layer: 1, pos: 32
type: A, layer: 1, pos: 43
type: B, layer: 1, pos: 41
type: A, layer: 1, pos: 41
type: B, layer: 1, pos: 37
type: B, layer: 1, pos: 23
type: A, layer: 1, pos: 20
type: B, layer: 1, pos: 21
type: A, layer: 1, pos: 5
type: B, layer: 1, pos: 20
type: B, layer: 1, pos: 5
type: B, layer: 1, pos: 38
type: B, layer: 1, pos: 16
type: A, layer: 1, pos: 21
type: A, layer: 1, pos: 16
type: A, layer: 1, pos: 23
type: B, layer: 1, pos: 29

Time for candidate selection: 0.11 seconds

### Candidate
type: A, layer: 1, pos: 3

## Relational analysis of NS_B1_B1_A1_A1_B1_A1_B2_A1

### Relational analysis result of NS_B1_B1_A1_A1_B1_A1_B2_A1
Status: Status.UNKNOWN
Output dim: 0, lower bound: -63.3964605, upper bound: 63.3964605
time: 0.71 seconds

## Relational analysis of NS_B1_B1_A1_A1_B1_A1_B2_A2

### Relational analysis result of NS_B1_B1_A1_A1_B1_A1_B2_A2
Status: Status.UNKNOWN
Output dim: 0, lower bound: -63.3964605, upper bound: 63.3964605
time: 0.66 seconds

## BFS NS instance: NS_B1_B1_A1_A1_B1_A2_B1

### Backsubstitution after applying NS history:
0: -42.3474426, 43.8335266, -29.5365906, 30.9115620, -73.2590027, 73.3701172
1: -296.5514221, 105.1179886, -209.6969757, 73.2576218, -369.0158691, 314.8148499
2: -166.3531342, 95.6677856, -117.0093307, 66.9293823, -232.4356842, 212.6770935
3: -208.2189026, 76.6767426, -146.5489655, 53.6311874, -260.3650818, 223.2257080
4: -120.2055283, 83.2302704, -84.2029343, 58.3682022, -178.0145264, 167.4331970

Time for backsubstitution: 1.22 seconds

### NS candidates at layer 1
type: B, layer: 1, pos: 29
type: A, layer: 1, pos: 9
type: B, layer: 1, pos: 17
type: B, layer: 1, pos: 9
type: A, layer: 1, pos: 3
type: B, layer: 1, pos: 35
type: A, layer: 1, pos: 35
type: A, layer: 1, pos: 19
type: B, layer: 1, pos: 10
type: A, layer: 1, pos: 45
type: B, layer: 1, pos: 38
type: A, layer: 1, pos: 18
type: B, layer: 1, pos: 19
type: B, layer: 1, pos: 45
type: A, layer: 1, pos: 43
type: B, layer: 1, pos: 11
type: A, layer: 1, pos: 11
type: A, layer: 1, pos: 15
type: A, layer: 1, pos: 17
type: B, layer: 1, pos: 32
type: A, layer: 1, pos: 39
type: A, layer: 1, pos: 10
type: B, layer: 1, pos: 15
type: B, layer: 1, pos: 37
type: A, layer: 1, pos: 23
type: B, layer: 1, pos: 20
type: A, layer: 1, pos: 16
type: B, layer: 1, pos: 41
type: A, layer: 1, pos: 41
type: A, layer: 1, pos: 5
type: A, layer: 1, pos: 37
type: B, layer: 1, pos: 39
type: B, layer: 1, pos: 18
type: B, layer: 1, pos: 21
type: A, layer: 1, pos: 32
type: A, layer: 1, pos: 21
type: B, layer: 1, pos: 5
type: A, layer: 1, pos: 20
type: B, layer: 1, pos: 23
type: B, layer: 1, pos: 16
type: B, layer: 1, pos: 43
type: A, layer: 1, pos: 38
type: A, layer: 1, pos: 29

Time for candidate selection: 0.13 seconds

### Candidate
type: B, layer: 1, pos: 29

### Candidate
type: A, layer: 1, pos: 9

### Candidate
type: B, layer: 1, pos: 17

### Candidate
type: B, layer: 1, pos: 9

### Candidate
type: A, layer: 1, pos: 3

## Relational analysis of NS_B1_B1_A1_A1_B1_A2_B1_A1
Optimization infeasible because this subproblem isn't reachable.

### Candidate
type: B, layer: 1, pos: 35

### Candidate
type: A, layer: 1, pos: 35

### Candidate
type: A, layer: 1, pos: 19

## Relational analysis of NS_B1_B1_A1_A1_B1_A2_B1_A1
Optimization infeasible because this subproblem isn't reachable.

### Candidate
type: B, layer: 1, pos: 10

## Relational analysis of NS_B1_B1_A1_A1_B1_A2_B1_B1

### Relational analysis result of NS_B1_B1_A1_A1_B1_A2_B1_B1
Status: Status.UNKNOWN
Output dim: 0, lower bound: -63.3960552, upper bound: 63.3959763
time: 0.73 seconds

## Relational analysis of NS_B1_B1_A1_A1_B1_A2_B1_B2

### Relational analysis result of NS_B1_B1_A1_A1_B1_A2_B1_B2
Status: Status.UNKNOWN
Output dim: 0, lower bound: -63.3958584, upper bound: 63.3958706
time: 0.80 seconds

## BFS NS instance: NS_B1_B1_A1_A1_B1_A2_B2

### Backsubstitution after applying NS history:
0: -40.5391808, 42.0468941, -29.2935848, 30.6672192, -71.2063904, 71.3404770
1: -285.2591248, 100.7088699, -207.5810242, 72.7790451, -357.5968628, 308.2898865
2: -159.7291260, 91.8024902, -115.9507675, 66.4279785, -225.5702972, 207.7532654
3: -200.1235809, 73.5756607, -145.0855560, 53.2599449, -252.1767120, 218.6612244
4: -115.2630692, 79.8833542, -83.5118332, 57.9175110, -172.9458923, 163.3951721

Time for backsubstitution: 1.28 seconds

### NS candidates at layer 1
type: B, layer: 1, pos: 29
type: A, layer: 1, pos: 9
type: A, layer: 1, pos: 3
type: A, layer: 1, pos: 45
type: B, layer: 1, pos: 35
type: B, layer: 1, pos: 17
type: B, layer: 1, pos: 9
type: A, layer: 1, pos: 19
type: A, layer: 1, pos: 35
type: B, layer: 1, pos: 10
type: B, layer: 1, pos: 19
type: A, layer: 1, pos: 11
type: B, layer: 1, pos: 11
type: B, layer: 1, pos: 38
type: A, layer: 1, pos: 18
type: A, layer: 1, pos: 17
type: A, layer: 1, pos: 15
type: A, layer: 1, pos: 43
type: B, layer: 1, pos: 32
type: B, layer: 1, pos: 45
type: A, layer: 1, pos: 39
type: A, layer: 1, pos: 10
type: B, layer: 1, pos: 15
type: B, layer: 1, pos: 37
type: B, layer: 1, pos: 20
type: B, layer: 1, pos: 39
type: B, layer: 1, pos: 18
type: A, layer: 1, pos: 23
type: A, layer: 1, pos: 37
type: B, layer: 1, pos: 41
type: A, layer: 1, pos: 5
type: A, layer: 1, pos: 41
type: A, layer: 1, pos: 16
type: B, layer: 1, pos: 21
type: A, layer: 1, pos: 32
type: A, layer: 1, pos: 21
type: B, layer: 1, pos: 5
type: A, layer: 1, pos: 20
type: B, layer: 1, pos: 23
type: B, layer: 1, pos: 16
type: B, layer: 1, pos: 43
type: A, layer: 1, pos: 38
type: A, layer: 1, pos: 29

Time for candidate selection: 0.11 seconds

### Candidate
type: B, layer: 1, pos: 29

### Candidate
type: A, layer: 1, pos: 9

### Candidate
type: A, layer: 1, pos: 3

## Relational analysis of NS_B1_B1_A1_A1_B1_A2_B2_A1
Optimization infeasible because this subproblem isn't reachable.

### Candidate
type: A, layer: 1, pos: 45

## Relational analysis of NS_B1_B1_A1_A1_B1_A2_B2_A1
Optimization infeasible because this subproblem isn't reachable.

### Candidate
type: B, layer: 1, pos: 35

### Candidate
type: B, layer: 1, pos: 17

### Candidate
type: B, layer: 1, pos: 9

### Candidate
type: A, layer: 1, pos: 19

## Relational analysis of NS_B1_B1_A1_A1_B1_A2_B2_A1
Optimization infeasible because this subproblem isn't reachable.

### Candidate
type: A, layer: 1, pos: 35

### Candidate
type: B, layer: 1, pos: 10

## Relational analysis of NS_B1_B1_A1_A1_B1_A2_B2_B1

### Relational analysis result of NS_B1_B1_A1_A1_B1_A2_B2_B1
Status: Status.UNKNOWN
Output dim: 0, lower bound: -63.3960641, upper bound: 63.3959851
time: 0.78 seconds

## Relational analysis of NS_B1_B1_A1_A1_B1_A2_B2_B2

### Relational analysis result of NS_B1_B1_A1_A1_B1_A2_B2_B2
Status: Status.UNKNOWN
Output dim: 0, lower bound: -63.3957853, upper bound: 63.3958234
time: 0.78 seconds

## BFS NS instance: NS_B1_B1_A1_A1_B2_A1_A1

### Backsubstitution after applying NS history:
0: -30.7523174, 32.2144966, -43.0804024, 44.6405792, -75.3928757, 75.2948990
1: -217.8811188, 76.2049332, -301.9946594, 106.9033508, -324.7844238, 377.6895447
2: -121.7813797, 69.5103455, -169.3493805, 97.3467484, -219.1281281, 238.2093201
3: -152.3731079, 55.7684937, -211.9939117, 78.0273743, -230.4004669, 266.4191589
4: -87.7194824, 60.6922379, -122.3427811, 84.7143784, -172.4338684, 182.6511536

Time for backsubstitution: 1.27 seconds

### NS candidates at layer 1
type: A, layer: 1, pos: 29
type: B, layer: 1, pos: 9
type: A, layer: 1, pos: 17
type: A, layer: 1, pos: 3
type: A, layer: 1, pos: 9
type: A, layer: 1, pos: 35
type: B, layer: 1, pos: 35
type: A, layer: 1, pos: 38
type: B, layer: 1, pos: 19
type: B, layer: 1, pos: 45
type: A, layer: 1, pos: 45
type: A, layer: 1, pos: 19
type: B, layer: 1, pos: 18
type: B, layer: 1, pos: 11
type: A, layer: 1, pos: 11
type: B, layer: 1, pos: 3
type: B, layer: 1, pos: 43
type: B, layer: 1, pos: 17
type: B, layer: 1, pos: 15
type: B, layer: 1, pos: 39
type: A, layer: 1, pos: 15
type: A, layer: 1, pos: 32
type: A, layer: 1, pos: 20
type: B, layer: 1, pos: 23
type: A, layer: 1, pos: 37
type: B, layer: 1, pos: 16
type: B, layer: 1, pos: 41
type: B, layer: 1, pos: 5
type: B, layer: 1, pos: 32
type: A, layer: 1, pos: 41
type: B, layer: 1, pos: 10
type: A, layer: 1, pos: 18
type: B, layer: 1, pos: 37
type: A, layer: 1, pos: 39
type: B, layer: 1, pos: 21
type: A, layer: 1, pos: 5
type: A, layer: 1, pos: 21
type: B, layer: 1, pos: 20
type: A, layer: 1, pos: 23
type: A, layer: 1, pos: 16
type: A, layer: 1, pos: 43
type: B, layer: 1, pos: 38
type: B, layer: 1, pos: 29

Time for candidate selection: 0.09 seconds

### Candidate
type: A, layer: 1, pos: 29

### Candidate
type: B, layer: 1, pos: 9

### Candidate
type: A, layer: 1, pos: 17

### Candidate
type: A, layer: 1, pos: 3

### Candidate
type: A, layer: 1, pos: 9

### Candidate
type: A, layer: 1, pos: 35

### Candidate
type: B, layer: 1, pos: 35

### Candidate
type: A, layer: 1, pos: 38

## Relational analysis of NS_B1_B1_A1_A1_B2_A1_A1_A1

### Relational analysis result of NS_B1_B1_A1_A1_B2_A1_A1_A1
Status: Status.UNKNOWN
Output dim: 0, lower bound: -63.3955244, upper bound: 63.3953745
time: 0.71 seconds

## Relational analysis of NS_B1_B1_A1_A1_B2_A1_A1_A2

### Relational analysis result of NS_B1_B1_A1_A1_B2_A1_A1_A2
Status: Status.UNKNOWN
Output dim: 0, lower bound: -63.3956007, upper bound: 63.3954687
time: 0.74 seconds

## BFS NS instance: NS_B1_B1_A1_A1_B2_A1_A2

### Backsubstitution after applying NS history:
0: -30.4671669, 31.8833961, -43.1612892, 44.7047577, -75.1719208, 75.0446854
1: -215.1082611, 75.3826675, -302.2045593, 107.1012421, -322.2095032, 377.1367798
2: -120.1630402, 68.7502365, -169.5616455, 97.4849930, -217.6480255, 237.6129150
3: -150.3408051, 55.0720863, -212.1918488, 78.1470184, -228.4878235, 265.8919373
4: -86.6366043, 60.0038681, -122.5466461, 84.8386154, -171.4751740, 182.2073517

Time for backsubstitution: 1.23 seconds

### NS candidates at layer 1
type: A, layer: 1, pos: 29
type: B, layer: 1, pos: 9
type: A, layer: 1, pos: 9
type: A, layer: 1, pos: 17
type: A, layer: 1, pos: 35
type: B, layer: 1, pos: 35
type: A, layer: 1, pos: 3
type: A, layer: 1, pos: 38
type: B, layer: 1, pos: 19
type: B, layer: 1, pos: 45
type: B, layer: 1, pos: 10
type: A, layer: 1, pos: 19
type: A, layer: 1, pos: 45
type: B, layer: 1, pos: 11
type: B, layer: 1, pos: 18
type: A, layer: 1, pos: 11
type: B, layer: 1, pos: 17
type: B, layer: 1, pos: 43
type: B, layer: 1, pos: 15
type: B, layer: 1, pos: 3
type: B, layer: 1, pos: 39
type: A, layer: 1, pos: 32
type: A, layer: 1, pos: 15
type: A, layer: 1, pos: 37
type: B, layer: 1, pos: 16
type: B, layer: 1, pos: 23
type: A, layer: 1, pos: 20
type: B, layer: 1, pos: 5
type: B, layer: 1, pos: 41
type: A, layer: 1, pos: 18
type: A, layer: 1, pos: 39
type: B, layer: 1, pos: 32
type: A, layer: 1, pos: 41
type: B, layer: 1, pos: 37
type: B, layer: 1, pos: 21
type: A, layer: 1, pos: 21
type: B, layer: 1, pos: 20
type: A, layer: 1, pos: 5
type: A, layer: 1, pos: 23
type: A, layer: 1, pos: 16
type: A, layer: 1, pos: 43
type: B, layer: 1, pos: 38
type: B, layer: 1, pos: 29

Time for candidate selection: 0.11 seconds

### Candidate
type: A, layer: 1, pos: 29

### Candidate
type: B, layer: 1, pos: 9

### Candidate
type: A, layer: 1, pos: 9

### Candidate
type: A, layer: 1, pos: 17

### Candidate
type: A, layer: 1, pos: 35

### Candidate
type: B, layer: 1, pos: 35

### Candidate
type: A, layer: 1, pos: 3

### Candidate
type: A, layer: 1, pos: 38

## Relational analysis of NS_B1_B1_A1_A1_B2_A1_A2_A1

### Relational analysis result of NS_B1_B1_A1_A1_B2_A1_A2_A1
Status: Status.UNKNOWN
Output dim: 0, lower bound: -63.3953176, upper bound: 63.3951373
time: 0.81 seconds

## Relational analysis of NS_B1_B1_A1_A1_B2_A1_A2_A2

### Relational analysis result of NS_B1_B1_A1_A1_B2_A1_A2_A2
Status: Status.UNKNOWN
Output dim: 0, lower bound: -63.3951157, upper bound: 63.3950635
time: 0.71 seconds

## BFS NS instance: NS_B1_B1_A1_A1_B2_A2_B1

### Backsubstitution after applying NS history:
0: -43.2815247, 44.8428879, -42.7765808, 44.2917786, -87.4930115, 87.5373688
1: -303.2672424, 107.4137878, -299.1881409, 106.1467361, -407.8079834, 404.9270020
2: -170.1005707, 97.8036041, -168.0145874, 96.6080780, -265.4073792, 264.4916992
3: -212.8860779, 78.3923187, -210.1160736, 77.4256058, -288.4560852, 286.6106873
4: -122.9020538, 85.1178970, -121.4461975, 84.0942764, -206.2808533, 205.8409119

Time for backsubstitution: 1.37 seconds

### NS candidates at layer 1
type: B, layer: 1, pos: 9
type: A, layer: 1, pos: 9
type: A, layer: 1, pos: 35
type: B, layer: 1, pos: 35
type: B, layer: 1, pos: 17
type: A, layer: 1, pos: 17
type: A, layer: 1, pos: 19
type: B, layer: 1, pos: 19
type: B, layer: 1, pos: 45
type: B, layer: 1, pos: 3
type: A, layer: 1, pos: 45
type: A, layer: 1, pos: 3
type: A, layer: 1, pos: 10
type: B, layer: 1, pos: 10
type: A, layer: 1, pos: 11
type: B, layer: 1, pos: 15
type: A, layer: 1, pos: 15
type: A, layer: 1, pos: 18
type: B, layer: 1, pos: 18
type: A, layer: 1, pos: 39
type: B, layer: 1, pos: 32
type: B, layer: 1, pos: 39
type: A, layer: 1, pos: 32
type: A, layer: 1, pos: 43
type: B, layer: 1, pos: 43
type: B, layer: 1, pos: 38
type: A, layer: 1, pos: 38
type: A, layer: 1, pos: 37
type: B, layer: 1, pos: 37
type: B, layer: 1, pos: 29
type: A, layer: 1, pos: 41
type: B, layer: 1, pos: 41
type: A, layer: 1, pos: 29
type: A, layer: 1, pos: 20
type: B, layer: 1, pos: 20
type: B, layer: 1, pos: 21
type: B, layer: 1, pos: 5
type: A, layer: 1, pos: 21
type: A, layer: 1, pos: 5
type: A, layer: 1, pos: 23
type: B, layer: 1, pos: 23
type: A, layer: 1, pos: 16
type: B, layer: 1, pos: 16

Time for candidate selection: 0.11 seconds

### Candidate
type: B, layer: 1, pos: 9

### Candidate
type: A, layer: 1, pos: 9

### Candidate
type: A, layer: 1, pos: 35

### Candidate
type: B, layer: 1, pos: 35

### Candidate
type: B, layer: 1, pos: 17

### Candidate
type: A, layer: 1, pos: 17

### Candidate
type: A, layer: 1, pos: 19

### Candidate
type: B, layer: 1, pos: 19

### Candidate
type: B, layer: 1, pos: 45

### Candidate
type: B, layer: 1, pos: 3

### Candidate
type: A, layer: 1, pos: 45

### Candidate
type: A, layer: 1, pos: 3

### Candidate
type: A, layer: 1, pos: 10

## Relational analysis of NS_B1_B1_A1_A1_B2_A2_B1_A1

### Relational analysis result of NS_B1_B1_A1_A1_B2_A2_B1_A1
Status: Status.UNKNOWN
Output dim: 0, lower bound: -63.3954147, upper bound: 63.3953984
time: 0.90 seconds

## Relational analysis of NS_B1_B1_A1_A1_B2_A2_B1_A2

### Relational analysis result of NS_B1_B1_A1_A1_B2_A2_B1_A2
Status: Status.UNKNOWN
Output dim: 0, lower bound: -63.3950458, upper bound: 63.3950052
time: 0.77 seconds

## BFS NS instance: NS_B1_B1_A1_A1_B2_A2_B2

### Backsubstitution after applying NS history:
0: -43.3717232, 44.9401779, -43.1568336, 44.7299004, -88.0543823, 88.0559235
1: -303.9672852, 107.6393127, -302.5183716, 107.1138229, -409.6247253, 408.7366028
2: -170.4709625, 98.0167847, -169.6340790, 97.5491638, -266.8119202, 266.4760437
3: -213.3660736, 78.5642319, -212.3484802, 78.1876984, -289.7970276, 289.1901245
4: -123.1605453, 85.3021240, -122.5672989, 84.9033813, -207.4030457, 207.2294769

Time for backsubstitution: 1.29 seconds

### NS candidates at layer 1
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
type: A, layer: 1, pos: 10
type: A, layer: 1, pos: 11
type: B, layer: 1, pos: 10
type: B, layer: 1, pos: 15
type: A, layer: 1, pos: 15
type: A, layer: 1, pos: 18
type: B, layer: 1, pos: 18
type: A, layer: 1, pos: 39
type: B, layer: 1, pos: 39
type: B, layer: 1, pos: 32
type: A, layer: 1, pos: 32
type: B, layer: 1, pos: 43
type: A, layer: 1, pos: 43
type: A, layer: 1, pos: 38
type: B, layer: 1, pos: 38
type: A, layer: 1, pos: 37
type: B, layer: 1, pos: 37
type: A, layer: 1, pos: 29
type: A, layer: 1, pos: 41
type: B, layer: 1, pos: 29
type: B, layer: 1, pos: 41
type: B, layer: 1, pos: 20
type: A, layer: 1, pos: 20
type: B, layer: 1, pos: 21
type: B, layer: 1, pos: 5
type: A, layer: 1, pos: 21
type: A, layer: 1, pos: 5
type: A, layer: 1, pos: 23
type: B, layer: 1, pos: 23
type: A, layer: 1, pos: 16
type: B, layer: 1, pos: 16

Time for candidate selection: 0.11 seconds

### Candidate
type: B, layer: 1, pos: 9

### Candidate
type: A, layer: 1, pos: 9

### Candidate
type: A, layer: 1, pos: 35

### Candidate
type: B, layer: 1, pos: 35

### Candidate
type: B, layer: 1, pos: 17

### Candidate
type: A, layer: 1, pos: 17

### Candidate
type: A, layer: 1, pos: 19

### Candidate
type: B, layer: 1, pos: 19

### Candidate
type: B, layer: 1, pos: 45

### Candidate
type: A, layer: 1, pos: 45

### Candidate
type: B, layer: 1, pos: 3

### Candidate
type: A, layer: 1, pos: 3

### Candidate
type: A, layer: 1, pos: 10

## Relational analysis of NS_B1_B1_A1_A1_B2_A2_B2_A1

### Relational analysis result of NS_B1_B1_A1_A1_B2_A2_B2_A1
Status: Status.UNKNOWN
Output dim: 0, lower bound: -63.3957456, upper bound: 63.3957699
time: 0.89 seconds

## Relational analysis of NS_B1_B1_A1_A1_B2_A2_B2_A2

### Relational analysis result of NS_B1_B1_A1_A1_B2_A2_B2_A2
Status: Status.UNKNOWN
Output dim: 0, lower bound: -63.3953744, upper bound: 63.3953744
time: 0.83 seconds

## BFS NS instance: NS_B1_B1_A1_A2_B1_B1_A1

### Backsubstitution after applying NS history:
0: -28.9203854, 30.4928493, -29.5365906, 30.9115620, -59.8319473, 60.0294418
1: -206.3644409, 71.5317383, -209.6969757, 73.2576218, -279.6220703, 281.2286682
2: -114.1110687, 65.8139496, -117.0093307, 66.9293823, -181.0404358, 182.8232727
3: -143.3531952, 52.7365494, -146.5489655, 53.6311874, -196.9843750, 199.2855225
4: -82.1403580, 57.6763458, -84.2029343, 58.3682022, -140.5085602, 141.8792725

Time for backsubstitution: 1.30 seconds

### NS candidates at layer 1
type: A, layer: 1, pos: 3
type: A, layer: 1, pos: 9
type: B, layer: 1, pos: 9
type: A, layer: 1, pos: 35
type: B, layer: 1, pos: 35
type: B, layer: 1, pos: 17
type: B, layer: 1, pos: 45
type: B, layer: 1, pos: 19
type: A, layer: 1, pos: 17
type: A, layer: 1, pos: 19
type: A, layer: 1, pos: 11
type: A, layer: 1, pos: 45
type: B, layer: 1, pos: 11
type: A, layer: 1, pos: 10
type: B, layer: 1, pos: 10
type: A, layer: 1, pos: 15
type: B, layer: 1, pos: 18
type: B, layer: 1, pos: 15
type: A, layer: 1, pos: 18
type: B, layer: 1, pos: 38
type: A, layer: 1, pos: 39
type: B, layer: 1, pos: 32
type: A, layer: 1, pos: 32
type: B, layer: 1, pos: 39
type: B, layer: 1, pos: 43
type: A, layer: 1, pos: 43
type: B, layer: 1, pos: 37
type: B, layer: 1, pos: 41
type: A, layer: 1, pos: 37
type: B, layer: 1, pos: 29
type: A, layer: 1, pos: 29
type: A, layer: 1, pos: 41
type: B, layer: 1, pos: 20
type: B, layer: 1, pos: 5
type: A, layer: 1, pos: 23
type: A, layer: 1, pos: 20
type: A, layer: 1, pos: 38
type: B, layer: 1, pos: 21
type: A, layer: 1, pos: 21
type: A, layer: 1, pos: 16
type: A, layer: 1, pos: 5
type: B, layer: 1, pos: 23
type: B, layer: 1, pos: 16

Time for candidate selection: 0.11 seconds

### Candidate
type: A, layer: 1, pos: 3

### Candidate
type: A, layer: 1, pos: 9

### Candidate
type: B, layer: 1, pos: 9

### Candidate
type: A, layer: 1, pos: 35

### Candidate
type: B, layer: 1, pos: 35

### Candidate
type: B, layer: 1, pos: 17

### Candidate
type: B, layer: 1, pos: 45

### Candidate
type: B, layer: 1, pos: 19

### Candidate
type: A, layer: 1, pos: 17

### Candidate
type: A, layer: 1, pos: 19

### Candidate
type: A, layer: 1, pos: 11

## Relational analysis of NS_B1_B1_A1_A2_B1_B1_A1_A1
Optimization infeasible because this subproblem isn't reachable.

### Candidate
type: A, layer: 1, pos: 45

### Candidate
type: B, layer: 1, pos: 11

### Candidate
type: A, layer: 1, pos: 10

## Relational analysis of NS_B1_B1_A1_A2_B1_B1_A1_A1
Optimization infeasible because this subproblem isn't reachable.

### Candidate
type: B, layer: 1, pos: 10

### Candidate
type: A, layer: 1, pos: 15

## Relational analysis of NS_B1_B1_A1_A2_B1_B1_A1_A1
Optimization infeasible because this subproblem isn't reachable.

### Candidate
type: B, layer: 1, pos: 18

## Relational analysis of NS_B1_B1_A1_A2_B1_B1_A1_B1
Optimization infeasible because this subproblem isn't reachable.

### Candidate
type: B, layer: 1, pos: 15

### Candidate
type: A, layer: 1, pos: 18

### Candidate
type: B, layer: 1, pos: 38

## Relational analysis of NS_B1_B1_A1_A2_B1_B1_A1_B1

### Relational analysis result of NS_B1_B1_A1_A2_B1_B1_A1_B1
Status: Status.UNKNOWN
Output dim: 0, lower bound: -63.3957790, upper bound: 63.3956037
time: 0.72 seconds

## Relational analysis of NS_B1_B1_A1_A2_B1_B1_A1_B2

### Relational analysis result of NS_B1_B1_A1_A2_B1_B1_A1_B2
Status: Status.UNKNOWN
Output dim: 0, lower bound: -63.3957790, upper bound: 63.3956545
time: 0.70 seconds

## BFS NS instance: NS_B1_B1_A1_A2_B1_B1_A2

### Backsubstitution after applying NS history:
0: -43.5455475, 45.1661339, -29.5365906, 30.9115620, -74.4571075, 74.7027283
1: -306.2635803, 108.1363754, -209.6969757, 73.2576218, -379.2349548, 317.8332214
2: -171.5144958, 98.5134354, -117.0093307, 66.9293823, -237.9598541, 215.5227509
3: -214.8825073, 78.9636917, -146.5489655, 53.6311874, -267.3988953, 225.5126648
4: -123.7212448, 85.7286911, -84.2029343, 58.3682022, -181.7980042, 169.9316101

Time for backsubstitution: 1.37 seconds

### NS candidates at layer 1
type: B, layer: 1, pos: 29
type: A, layer: 1, pos: 9
type: B, layer: 1, pos: 17
type: B, layer: 1, pos: 9
type: A, layer: 1, pos: 3
type: A, layer: 1, pos: 35
type: B, layer: 1, pos: 35
type: B, layer: 1, pos: 38
type: B, layer: 1, pos: 10
type: A, layer: 1, pos: 19
type: A, layer: 1, pos: 45
type: B, layer: 1, pos: 19
type: B, layer: 1, pos: 45
type: A, layer: 1, pos: 18
type: A, layer: 1, pos: 11
type: B, layer: 1, pos: 11
type: A, layer: 1, pos: 15
type: A, layer: 1, pos: 43
type: A, layer: 1, pos: 17
type: A, layer: 1, pos: 39
type: B, layer: 1, pos: 32
type: A, layer: 1, pos: 10
type: B, layer: 1, pos: 15
type: B, layer: 1, pos: 37
type: A, layer: 1, pos: 23
type: B, layer: 1, pos: 20
type: A, layer: 1, pos: 16
type: B, layer: 1, pos: 18
type: A, layer: 1, pos: 41
type: B, layer: 1, pos: 41
type: A, layer: 1, pos: 5
type: A, layer: 1, pos: 37
type: B, layer: 1, pos: 39
type: A, layer: 1, pos: 32
type: A, layer: 1, pos: 21
type: B, layer: 1, pos: 21
type: B, layer: 1, pos: 5
type: A, layer: 1, pos: 20
type: B, layer: 1, pos: 23
type: B, layer: 1, pos: 16
type: B, layer: 1, pos: 43
type: A, layer: 1, pos: 38
type: A, layer: 1, pos: 29

Time for candidate selection: 0.12 seconds

### Candidate
type: B, layer: 1, pos: 29

## Relational analysis of NS_B1_B1_A1_A2_B1_B1_A2_B1
Optimization infeasible because this subproblem isn't reachable.

### Candidate
type: A, layer: 1, pos: 9

### Candidate
type: B, layer: 1, pos: 17

### Candidate
type: B, layer: 1, pos: 9

### Candidate
type: A, layer: 1, pos: 3

### Candidate
type: A, layer: 1, pos: 35

### Candidate
type: B, layer: 1, pos: 35

### Candidate
type: B, layer: 1, pos: 38

## Relational analysis of NS_B1_B1_A1_A2_B1_B1_A2_B1

### Relational analysis result of NS_B1_B1_A1_A2_B1_B1_A2_B1
Status: Status.UNKNOWN
Output dim: 0, lower bound: -63.3957790, upper bound: 63.3956037
time: 0.81 seconds

## Relational analysis of NS_B1_B1_A1_A2_B1_B1_A2_B2

### Relational analysis result of NS_B1_B1_A1_A2_B1_B1_A2_B2
Status: Status.UNKNOWN
Output dim: 0, lower bound: -63.3957790, upper bound: 63.3956545
time: 0.69 seconds

## BFS NS instance: NS_B1_B1_A1_A2_B1_B2_A1

### Backsubstitution after applying NS history:
0: -27.9795895, 29.5172138, -29.2935848, 30.6672192, -58.6468086, 58.8107986
1: -199.4865875, 69.2859421, -207.5810242, 72.7790451, -272.2656250, 276.8669739
2: -110.3211212, 63.6366425, -115.9507675, 66.4279785, -176.7490997, 179.5874023
3: -138.6365662, 51.0289268, -145.0855560, 53.2599449, -191.8965149, 196.1144562
4: -79.5082932, 55.8040314, -83.5118332, 57.9175110, -137.4257660, 139.3158569

Time for backsubstitution: 1.38 seconds

### NS candidates at layer 1
type: A, layer: 1, pos: 3
type: A, layer: 1, pos: 9
type: B, layer: 1, pos: 43
type: B, layer: 1, pos: 9
type: A, layer: 1, pos: 17
type: B, layer: 1, pos: 17
type: A, layer: 1, pos: 43
type: A, layer: 1, pos: 35
type: B, layer: 1, pos: 35
type: A, layer: 1, pos: 45
type: B, layer: 1, pos: 19
type: A, layer: 1, pos: 11
type: B, layer: 1, pos: 11
type: A, layer: 1, pos: 19
type: A, layer: 1, pos: 10
type: B, layer: 1, pos: 10
type: A, layer: 1, pos: 15
type: B, layer: 1, pos: 15
type: B, layer: 1, pos: 45
type: A, layer: 1, pos: 29
type: B, layer: 1, pos: 18
type: B, layer: 1, pos: 32
type: A, layer: 1, pos: 32
type: A, layer: 1, pos: 39
type: B, layer: 1, pos: 38
type: B, layer: 1, pos: 39
type: B, layer: 1, pos: 37
type: A, layer: 1, pos: 37
type: B, layer: 1, pos: 41
type: A, layer: 1, pos: 38
type: B, layer: 1, pos: 23
type: B, layer: 1, pos: 21
type: A, layer: 1, pos: 23
type: A, layer: 1, pos: 41
type: B, layer: 1, pos: 20
type: A, layer: 1, pos: 20
type: A, layer: 1, pos: 18
type: A, layer: 1, pos: 16
type: A, layer: 1, pos: 21
type: A, layer: 1, pos: 5
type: B, layer: 1, pos: 5
type: B, layer: 1, pos: 16
type: B, layer: 1, pos: 29

Time for candidate selection: 0.11 seconds

### Candidate
type: A, layer: 1, pos: 3

### Candidate
type: A, layer: 1, pos: 9

### Candidate
type: B, layer: 1, pos: 43

## Relational analysis of NS_B1_B1_A1_A2_B1_B2_A1_B1
Optimization infeasible because this subproblem isn't reachable.

### Candidate
type: B, layer: 1, pos: 9

### Candidate
type: A, layer: 1, pos: 17

### Candidate
type: B, layer: 1, pos: 17

### Candidate
type: A, layer: 1, pos: 43

### Candidate
type: A, layer: 1, pos: 35

### Candidate
type: B, layer: 1, pos: 35

### Candidate
type: A, layer: 1, pos: 45

### Candidate
type: B, layer: 1, pos: 19

### Candidate
type: A, layer: 1, pos: 11

## Relational analysis of NS_B1_B1_A1_A2_B1_B2_A1_A1
Optimization infeasible because this subproblem isn't reachable.

### Candidate
type: B, layer: 1, pos: 11

### Candidate
type: A, layer: 1, pos: 19

### Candidate
type: A, layer: 1, pos: 10

## Relational analysis of NS_B1_B1_A1_A2_B1_B2_A1_A1
Optimization infeasible because this subproblem isn't reachable.

### Candidate
type: B, layer: 1, pos: 10

### Candidate
type: A, layer: 1, pos: 15

## Relational analysis of NS_B1_B1_A1_A2_B1_B2_A1_A1
Optimization infeasible because this subproblem isn't reachable.

### Candidate
type: B, layer: 1, pos: 15

### Candidate
type: B, layer: 1, pos: 45

### Candidate
type: A, layer: 1, pos: 29

## Relational analysis of NS_B1_B1_A1_A2_B1_B2_A1_A1
Optimization infeasible because this subproblem isn't reachable.

### Candidate
type: B, layer: 1, pos: 18

## Relational analysis of NS_B1_B1_A1_A2_B1_B2_A1_B1
Optimization infeasible because this subproblem isn't reachable.

### Candidate
type: B, layer: 1, pos: 32

## Relational analysis of NS_B1_B1_A1_A2_B1_B2_A1_B1
Optimization infeasible because this subproblem isn't reachable.

### Candidate
type: A, layer: 1, pos: 32

### Candidate
type: A, layer: 1, pos: 39

## Relational analysis of NS_B1_B1_A1_A2_B1_B2_A1_A1

### Relational analysis result of NS_B1_B1_A1_A2_B1_B2_A1_A1
Status: Status.UNKNOWN
Output dim: 0, lower bound: -63.3954394, upper bound: 63.3954791
time: 0.78 seconds

## Relational analysis of NS_B1_B1_A1_A2_B1_B2_A1_A2

### Relational analysis result of NS_B1_B1_A1_A2_B1_B2_A1_A2
Status: Status.UNKNOWN
Output dim: 0, lower bound: -63.3952984, upper bound: 63.3952597
time: 0.78 seconds

## BFS NS instance: NS_B1_B1_A1_A2_B1_B2_A2

### Backsubstitution after applying NS history:
0: -41.8346672, 43.4684563, -29.2935848, 30.6672192, -72.5018845, 72.7620316
1: -295.4231262, 103.9650803, -207.5810242, 72.7790451, -368.2021484, 311.5461121
2: -165.1257629, 94.8444672, -115.9507675, 66.4279785, -231.3134460, 210.7952271
3: -207.1293488, 76.0368118, -145.0855560, 53.2599449, -259.5586853, 221.1223755
4: -118.9920349, 82.5691147, -83.5118332, 57.9175110, -176.9095306, 166.0809479

Time for backsubstitution: 1.18 seconds

### NS candidates at layer 1
type: A, layer: 1, pos: 9
type: A, layer: 1, pos: 3
type: B, layer: 1, pos: 29
type: A, layer: 1, pos: 45
type: B, layer: 1, pos: 17
type: B, layer: 1, pos: 35
type: B, layer: 1, pos: 9
type: A, layer: 1, pos: 35
type: A, layer: 1, pos: 19
type: B, layer: 1, pos: 10
type: B, layer: 1, pos: 38
type: B, layer: 1, pos: 19
type: A, layer: 1, pos: 11
type: B, layer: 1, pos: 11
type: A, layer: 1, pos: 17
type: A, layer: 1, pos: 15
type: A, layer: 1, pos: 18
type: B, layer: 1, pos: 45
type: A, layer: 1, pos: 43
type: B, layer: 1, pos: 32
type: A, layer: 1, pos: 39
type: A, layer: 1, pos: 10
type: B, layer: 1, pos: 15
type: B, layer: 1, pos: 18
type: B, layer: 1, pos: 37
type: B, layer: 1, pos: 20
type: A, layer: 1, pos: 23
type: B, layer: 1, pos: 39
type: A, layer: 1, pos: 5
type: B, layer: 1, pos: 41
type: A, layer: 1, pos: 41
type: A, layer: 1, pos: 37
type: A, layer: 1, pos: 16
type: A, layer: 1, pos: 32
type: B, layer: 1, pos: 21
type: A, layer: 1, pos: 21
type: B, layer: 1, pos: 43
type: B, layer: 1, pos: 5
type: A, layer: 1, pos: 20
type: B, layer: 1, pos: 16
type: B, layer: 1, pos: 23
type: A, layer: 1, pos: 38
type: A, layer: 1, pos: 29

Time for candidate selection: 0.08 seconds

### Candidate
type: A, layer: 1, pos: 9

### Candidate
type: A, layer: 1, pos: 3

### Candidate
type: B, layer: 1, pos: 29

## Relational analysis of NS_B1_B1_A1_A2_B1_B2_A2_B1
Optimization infeasible because this subproblem isn't reachable.

### Candidate
type: A, layer: 1, pos: 45

### Candidate
type: B, layer: 1, pos: 17

### Candidate
type: B, layer: 1, pos: 35

### Candidate
type: B, layer: 1, pos: 9

### Candidate
type: A, layer: 1, pos: 35

### Candidate
type: A, layer: 1, pos: 19

### Candidate
type: B, layer: 1, pos: 10

## Relational analysis of NS_B1_B1_A1_A2_B1_B2_A2_B1
Optimization infeasible because this subproblem isn't reachable.

### Candidate
type: B, layer: 1, pos: 38

## Relational analysis of NS_B1_B1_A1_A2_B1_B2_A2_B1
Optimization infeasible because this subproblem isn't reachable.

### Candidate
type: B, layer: 1, pos: 19

### Candidate
type: A, layer: 1, pos: 11

## Relational analysis of NS_B1_B1_A1_A2_B1_B2_A2_A1
Optimization infeasible because this subproblem isn't reachable.

### Candidate
type: B, layer: 1, pos: 11

### Candidate
type: A, layer: 1, pos: 17

### Candidate
type: A, layer: 1, pos: 15

## Relational analysis of NS_B1_B1_A1_A2_B1_B2_A2_A1
Optimization infeasible because this subproblem isn't reachable.

### Candidate
type: A, layer: 1, pos: 18

## Relational analysis of NS_B1_B1_A1_A2_B1_B2_A2_A1
Optimization infeasible because this subproblem isn't reachable.

### Candidate
type: B, layer: 1, pos: 45

### Candidate
type: A, layer: 1, pos: 43

## Relational analysis of NS_B1_B1_A1_A2_B1_B2_A2_A1
Optimization infeasible because this subproblem isn't reachable.

### Candidate
type: B, layer: 1, pos: 32

## Relational analysis of NS_B1_B1_A1_A2_B1_B2_A2_B1
Optimization infeasible because this subproblem isn't reachable.

### Candidate
type: A, layer: 1, pos: 39

## Relational analysis of NS_B1_B1_A1_A2_B1_B2_A2_A1

### Relational analysis result of NS_B1_B1_A1_A2_B1_B2_A2_A1
Status: Status.UNKNOWN
Output dim: 0, lower bound: -63.3954394, upper bound: 63.3954791
time: 0.67 seconds

## Relational analysis of NS_B1_B1_A1_A2_B1_B2_A2_A2

### Relational analysis result of NS_B1_B1_A1_A2_B1_B2_A2_A2
Status: Status.UNKNOWN
Output dim: 0, lower bound: -63.3952984, upper bound: 63.3952598
time: 0.75 seconds

## BFS NS instance: NS_B1_B1_A1_A2_B2_A1_A1

### Backsubstitution after applying NS history:
0: -29.1075058, 30.6168957, -42.4086761, 43.8786583, -72.9861603, 73.0255737
1: -207.8951569, 72.0322952, -297.0866089, 105.2603531, -313.1555176, 369.1188965
2: -114.9896011, 66.0805664, -166.7182007, 95.7784424, -210.7680359, 232.3902283
3: -144.6104736, 52.9815865, -208.6051178, 76.7622681, -221.3727417, 260.3976135
4: -82.6823883, 57.7715912, -120.4414139, 83.2792130, -165.9616089, 178.0212708

Time for backsubstitution: 1.08 seconds

### NS candidates at layer 1
type: A, layer: 1, pos: 29
type: B, layer: 1, pos: 9
type: A, layer: 1, pos: 9
type: A, layer: 1, pos: 35
type: B, layer: 1, pos: 35
type: B, layer: 1, pos: 19
type: A, layer: 1, pos: 10
type: B, layer: 1, pos: 45
type: A, layer: 1, pos: 3
type: B, layer: 1, pos: 18
type: A, layer: 1, pos: 11
type: A, layer: 1, pos: 19
type: B, layer: 1, pos: 43
type: A, layer: 1, pos: 38
type: A, layer: 1, pos: 45
type: B, layer: 1, pos: 11
type: B, layer: 1, pos: 17
type: B, layer: 1, pos: 3
type: B, layer: 1, pos: 15
type: A, layer: 1, pos: 32
type: B, layer: 1, pos: 39
type: A, layer: 1, pos: 15
type: B, layer: 1, pos: 10
type: A, layer: 1, pos: 37
type: A, layer: 1, pos: 20
type: B, layer: 1, pos: 23
type: B, layer: 1, pos: 41
type: A, layer: 1, pos: 39
type: B, layer: 1, pos: 5
type: A, layer: 1, pos: 41
type: B, layer: 1, pos: 37
type: B, layer: 1, pos: 16
type: B, layer: 1, pos: 32
type: A, layer: 1, pos: 18
type: A, layer: 1, pos: 21
type: B, layer: 1, pos: 21
type: A, layer: 1, pos: 5
type: B, layer: 1, pos: 20
type: A, layer: 1, pos: 16
type: A, layer: 1, pos: 23
type: B, layer: 1, pos: 38
type: A, layer: 1, pos: 43
type: B, layer: 1, pos: 29

Time for candidate selection: 0.10 seconds

### Candidate
type: A, layer: 1, pos: 29

### Candidate
type: B, layer: 1, pos: 9

### Candidate
type: A, layer: 1, pos: 9

### Candidate
type: A, layer: 1, pos: 35

### Candidate
type: B, layer: 1, pos: 35

### Candidate
type: B, layer: 1, pos: 19

## Relational analysis of NS_B1_B1_A1_A2_B2_A1_A1_B1
Optimization infeasible because this subproblem isn't reachable.

### Candidate
type: A, layer: 1, pos: 10

## Relational analysis of NS_B1_B1_A1_A2_B2_A1_A1_A1
Optimization infeasible because this subproblem isn't reachable.

### Candidate
type: B, layer: 1, pos: 45

## Relational analysis of NS_B1_B1_A1_A2_B2_A1_A1_B1
Optimization infeasible because this subproblem isn't reachable.

### Candidate
type: A, layer: 1, pos: 3

## Relational analysis of NS_B1_B1_A1_A2_B2_A1_A1_A1
Optimization infeasible because this subproblem isn't reachable.

### Candidate
type: B, layer: 1, pos: 18

## Relational analysis of NS_B1_B1_A1_A2_B2_A1_A1_B1
Optimization infeasible because this subproblem isn't reachable.

### Candidate
type: A, layer: 1, pos: 11

## Relational analysis of NS_B1_B1_A1_A2_B2_A1_A1_A1
Optimization infeasible because this subproblem isn't reachable.

### Candidate
type: A, layer: 1, pos: 19

### Candidate
type: B, layer: 1, pos: 43

## Relational analysis of NS_B1_B1_A1_A2_B2_A1_A1_B1
Optimization infeasible because this subproblem isn't reachable.

### Candidate
type: A, layer: 1, pos: 38

## Relational analysis of NS_B1_B1_A1_A2_B2_A1_A1_A1
Optimization infeasible because this subproblem isn't reachable.

### Candidate
type: A, layer: 1, pos: 45

### Candidate
type: B, layer: 1, pos: 11

### Candidate
type: B, layer: 1, pos: 17

## Relational analysis of NS_B1_B1_A1_A2_B2_A1_A1_B1

### Relational analysis result of NS_B1_B1_A1_A2_B2_A1_A1_B1
Status: Status.UNKNOWN
Output dim: 0, lower bound: -63.3943225, upper bound: 63.3946255
time: 0.78 seconds

## Relational analysis of NS_B1_B1_A1_A2_B2_A1_A1_B2

### Relational analysis result of NS_B1_B1_A1_A2_B2_A1_A1_B2
Status: Status.UNKNOWN
Output dim: 0, lower bound: -63.3943225, upper bound: 63.3953713
time: 0.78 seconds

## BFS NS instance: NS_B1_B1_A1_A2_B2_A1_A2

### Backsubstitution after applying NS history:
0: -29.9918919, 31.6059914, -43.2364655, 44.7933006, -74.7851868, 74.8424530
1: -213.4465332, 74.1882706, -302.9176025, 107.3049316, -320.7514343, 377.0272217
2: -118.2696838, 68.1327744, -169.9059296, 97.6940842, -215.9637756, 237.4962616
3: -148.3811798, 54.6175232, -212.6481628, 78.3096619, -226.6908417, 265.9521790
4: -85.2141724, 59.7166481, -122.7699356, 85.0215454, -170.2356873, 182.1769409

Time for backsubstitution: 1.37 seconds

### NS candidates at layer 1
type: A, layer: 1, pos: 29
type: B, layer: 1, pos: 9
type: A, layer: 1, pos: 35
type: A, layer: 1, pos: 9
type: B, layer: 1, pos: 45
type: B, layer: 1, pos: 19
type: A, layer: 1, pos: 10
type: B, layer: 1, pos: 35
type: B, layer: 1, pos: 17
type: A, layer: 1, pos: 3
type: B, layer: 1, pos: 18
type: A, layer: 1, pos: 19
type: A, layer: 1, pos: 11
type: A, layer: 1, pos: 38
type: B, layer: 1, pos: 11
type: B, layer: 1, pos: 43
type: B, layer: 1, pos: 3
type: A, layer: 1, pos: 45
type: B, layer: 1, pos: 15
type: A, layer: 1, pos: 32
type: B, layer: 1, pos: 39
type: A, layer: 1, pos: 15
type: B, layer: 1, pos: 10
type: A, layer: 1, pos: 37
type: A, layer: 1, pos: 39
type: A, layer: 1, pos: 20
type: B, layer: 1, pos: 5
type: B, layer: 1, pos: 23
type: B, layer: 1, pos: 37
type: B, layer: 1, pos: 41
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
type: A, layer: 1, pos: 43
type: B, layer: 1, pos: 38
type: B, layer: 1, pos: 29

Time for candidate selection: 0.09 seconds

### Candidate
type: A, layer: 1, pos: 29

### Candidate
type: B, layer: 1, pos: 9

### Candidate
type: A, layer: 1, pos: 35

### Candidate
type: A, layer: 1, pos: 9

### Candidate
type: B, layer: 1, pos: 45

## Relational analysis of NS_B1_B1_A1_A2_B2_A1_A2_B1
Optimization infeasible because this subproblem isn't reachable.

### Candidate
type: B, layer: 1, pos: 19

## Relational analysis of NS_B1_B1_A1_A2_B2_A1_A2_B1
Optimization infeasible because this subproblem isn't reachable.

### Candidate
type: A, layer: 1, pos: 10

## Relational analysis of NS_B1_B1_A1_A2_B2_A1_A2_A1
Optimization infeasible because this subproblem isn't reachable.

### Candidate
type: B, layer: 1, pos: 35

### Candidate
type: B, layer: 1, pos: 17

## Relational analysis of NS_B1_B1_A1_A2_B2_A1_A2_B1

### Relational analysis result of NS_B1_B1_A1_A2_B2_A1_A2_B1
Status: Status.UNKNOWN
Output dim: 0, lower bound: -63.3947200, upper bound: 63.3952202
time: 0.65 seconds

## Relational analysis of NS_B1_B1_A1_A2_B2_A1_A2_B2

### Relational analysis result of NS_B1_B1_A1_A2_B2_A1_A2_B2
Status: Status.UNKNOWN
Output dim: 0, lower bound: -63.3947200, upper bound: 63.3959908
time: 0.69 seconds

## BFS NS instance: NS_B1_B1_A1_A2_B2_A2_A2

### Backsubstitution after applying NS history:
0: -43.0588684, 44.7279816, -42.6379204, 44.1991959, -87.2580643, 87.3659058
1: -303.0512390, 106.8752518, -298.8344116, 105.8017044, -408.2455750, 404.7100525
2: -169.5615540, 97.4988174, -167.5728760, 96.3819962, -265.3379211, 264.1490479
3: -212.4954224, 78.1508484, -209.7274780, 77.2568741, -288.7063599, 286.4180603
4: -122.3466568, 84.9117966, -121.0736923, 83.8969269, -206.0624695, 205.5648956

Time for backsubstitution: 1.30 seconds

### NS candidates at layer 1
type: B, layer: 1, pos: 9
type: A, layer: 1, pos: 35
type: B, layer: 1, pos: 35
type: B, layer: 1, pos: 19
type: B, layer: 1, pos: 17
type: A, layer: 1, pos: 17
type: B, layer: 1, pos: 45
type: B, layer: 1, pos: 3
type: A, layer: 1, pos: 19
type: A, layer: 1, pos: 45
type: A, layer: 1, pos: 3
type: A, layer: 1, pos: 11
type: A, layer: 1, pos: 10
type: B, layer: 1, pos: 11
type: B, layer: 1, pos: 10
type: B, layer: 1, pos: 18
type: B, layer: 1, pos: 15
type: A, layer: 1, pos: 15
type: A, layer: 1, pos: 32
type: B, layer: 1, pos: 43
type: A, layer: 1, pos: 39
type: B, layer: 1, pos: 38
type: A, layer: 1, pos: 18
type: B, layer: 1, pos: 39
type: A, layer: 1, pos: 29
type: B, layer: 1, pos: 37
type: B, layer: 1, pos: 32
type: A, layer: 1, pos: 37
type: B, layer: 1, pos: 41
type: A, layer: 1, pos: 43
type: A, layer: 1, pos: 41
type: B, layer: 1, pos: 20
type: A, layer: 1, pos: 38
type: B, layer: 1, pos: 29
type: A, layer: 1, pos: 21
type: A, layer: 1, pos: 23
type: B, layer: 1, pos: 5
type: A, layer: 1, pos: 20
type: B, layer: 1, pos: 21
type: A, layer: 1, pos: 5
type: A, layer: 1, pos: 16
type: B, layer: 1, pos: 23
type: B, layer: 1, pos: 16

Time for candidate selection: 0.11 seconds

### Candidate
type: B, layer: 1, pos: 9

## Relational analysis of NS_B1_B1_A1_A2_B2_A2_A2_B1
Optimization infeasible because this subproblem isn't reachable.

### Candidate
type: A, layer: 1, pos: 35

## Relational analysis of NS_B1_B1_A1_A2_B2_A2_A2_A1
Optimization infeasible because this subproblem isn't reachable.

### Candidate
type: B, layer: 1, pos: 35

### Candidate
type: B, layer: 1, pos: 19

## Relational analysis of NS_B1_B1_A1_A2_B2_A2_A2_B1
Optimization infeasible because this subproblem isn't reachable.

### Candidate
type: B, layer: 1, pos: 17

## Relational analysis of NS_B1_B1_A1_A2_B2_A2_A2_B1

### Relational analysis result of NS_B1_B1_A1_A2_B2_A2_A2_B1
Status: Status.VERIFIED
Output dim: 0, lower bound: -63.3924704, upper bound: 63.3923936
time: 0.74 seconds

## Relational analysis of NS_B1_B1_A1_A2_B2_A2_A2_B2
Optimization infeasible because this subproblem isn't reachable.

### Candidate
type: A, layer: 1, pos: 17

### Candidate
type: B, layer: 1, pos: 45

## Relational analysis of NS_B1_B1_A1_A2_B2_A2_A2_B1
Optimization infeasible because this subproblem isn't reachable.

### Candidate
type: B, layer: 1, pos: 3

## Relational analysis of NS_B1_B1_A1_A2_B2_A2_A2_B1
Optimization infeasible because this subproblem isn't reachable.

### Candidate
type: A, layer: 1, pos: 19

### Candidate
type: A, layer: 1, pos: 45

### Candidate
type: A, layer: 1, pos: 3

### Candidate
type: A, layer: 1, pos: 11

## Relational analysis of NS_B1_B1_A1_A2_B2_A2_A2_A1
Optimization infeasible because this subproblem isn't reachable.

### Candidate
type: A, layer: 1, pos: 10

## Relational analysis of NS_B1_B1_A1_A2_B2_A2_A2_A1
Optimization infeasible because this subproblem isn't reachable.

### Candidate
type: B, layer: 1, pos: 11

### Candidate
type: B, layer: 1, pos: 10

### Candidate
type: B, layer: 1, pos: 18

## Relational analysis of NS_B1_B1_A1_A2_B2_A2_A2_B1
Optimization infeasible because this subproblem isn't reachable.

### Candidate
type: B, layer: 1, pos: 15

## Relational analysis of NS_B1_B1_A1_A2_B2_A2_A2_B1
Optimization infeasible because this subproblem isn't reachable.

### Candidate
type: A, layer: 1, pos: 15

### Candidate
type: A, layer: 1, pos: 32

## Relational analysis of NS_B1_B1_A1_A2_B2_A2_A2_A1
Optimization infeasible because this subproblem isn't reachable.

### Candidate
type: B, layer: 1, pos: 43

## Relational analysis of NS_B1_B1_A1_A2_B2_A2_A2_B1
Optimization infeasible because this subproblem isn't reachable.

### Candidate
type: A, layer: 1, pos: 39

## Relational analysis of NS_B1_B1_A1_A2_B2_A2_A2_A1
Optimization infeasible because this subproblem isn't reachable.

### Candidate
type: B, layer: 1, pos: 38

## Relational analysis of NS_B1_B1_A1_A2_B2_A2_A2_B1
Optimization infeasible because this subproblem isn't reachable.

### Candidate
type: A, layer: 1, pos: 18

### Candidate
type: B, layer: 1, pos: 39

### Candidate
type: A, layer: 1, pos: 29

## Relational analysis of NS_B1_B1_A1_A2_B2_A2_A2_A1
Optimization infeasible because this subproblem isn't reachable.

### Candidate
type: B, layer: 1, pos: 37

## Relational analysis of NS_B1_B1_A1_A2_B2_A2_A2_B1

### Relational analysis result of NS_B1_B1_A1_A2_B2_A2_A2_B1
Status: Status.VERIFIED
Output dim: 0, lower bound: -63.3792120, upper bound: 63.3790508
time: 0.74 seconds

## Relational analysis of NS_B1_B1_A1_A2_B2_A2_A2_B2

### Relational analysis result of NS_B1_B1_A1_A2_B2_A2_A2_B2
Status: Status.UNKNOWN
Output dim: 0, lower bound: -63.3948749, upper bound: 63.3946019
time: 0.77 seconds

## BFS NS instance: NS_B1_B1_A2_B1_B1_A1_B1

### Backsubstitution after applying NS history:
0: -51.3486900, 55.8240395, -27.8771610, 29.2590733, -80.6077652, 83.7011795
1: -379.5923462, 126.7126923, -198.3176727, 69.1243134, -447.5756226, 325.0303040
2: -202.6836090, 119.6470337, -110.9151154, 63.1946526, -265.8782349, 230.5621338
3: -261.6979675, 95.8248215, -138.6566772, 50.6811104, -311.9244080, 234.4814911
4: -145.6045074, 105.8283157, -79.7050247, 55.1526260, -200.7571259, 185.5333252

Time for backsubstitution: 1.22 seconds

### NS candidates at layer 1
type: B, layer: 1, pos: 4
type: B, layer: 1, pos: 9
type: B, layer: 1, pos: 38
type: A, layer: 1, pos: 45
type: A, layer: 1, pos: 35
type: B, layer: 1, pos: 17
type: B, layer: 1, pos: 3
type: A, layer: 1, pos: 9
type: A, layer: 1, pos: 11
type: B, layer: 1, pos: 11
type: B, layer: 1, pos: 10
type: A, layer: 1, pos: 10
type: A, layer: 1, pos: 16
type: A, layer: 1, pos: 15
type: A, layer: 1, pos: 39
type: B, layer: 1, pos: 35
type: B, layer: 1, pos: 37
type: A, layer: 1, pos: 43
type: A, layer: 1, pos: 3
type: A, layer: 1, pos: 17
type: A, layer: 1, pos: 19
type: B, layer: 1, pos: 32
type: B, layer: 1, pos: 18
type: A, layer: 1, pos: 18
type: A, layer: 1, pos: 23
type: B, layer: 1, pos: 21
type: B, layer: 1, pos: 15
type: B, layer: 1, pos: 41
type: A, layer: 1, pos: 32
type: A, layer: 1, pos: 20
type: A, layer: 1, pos: 41
type: B, layer: 1, pos: 5
type: A, layer: 1, pos: 5
type: B, layer: 1, pos: 20
type: B, layer: 1, pos: 39
type: A, layer: 1, pos: 21
type: B, layer: 1, pos: 43
type: B, layer: 1, pos: 23
type: A, layer: 1, pos: 37
type: B, layer: 1, pos: 16
type: A, layer: 1, pos: 38
type: A, layer: 1, pos: 4
type: A, layer: 1, pos: 29

Time for candidate selection: 0.09 seconds

### Candidate
type: B, layer: 1, pos: 4

### Candidate
type: B, layer: 1, pos: 9

### Candidate
type: B, layer: 1, pos: 38

### Candidate
type: A, layer: 1, pos: 45

## Relational analysis of NS_B1_B1_A2_B1_B1_A1_B1_A1
Optimization infeasible because this subproblem isn't reachable.

### Candidate
type: A, layer: 1, pos: 35

### Candidate
type: B, layer: 1, pos: 17

### Candidate
type: B, layer: 1, pos: 3

### Candidate
type: A, layer: 1, pos: 9

### Candidate
type: A, layer: 1, pos: 11

## Relational analysis of NS_B1_B1_A2_B1_B1_A1_B1_A1

### Relational analysis result of NS_B1_B1_A2_B1_B1_A1_B1_A1
Status: Status.VERIFIED
Output dim: 0, lower bound: -63.3936616, upper bound: 63.3934311
time: 0.83 seconds

## Relational analysis of NS_B1_B1_A2_B1_B1_A1_B1_A2

### Relational analysis result of NS_B1_B1_A2_B1_B1_A1_B1_A2
Status: Status.UNKNOWN
Output dim: 0, lower bound: -63.3938288, upper bound: 63.3935210
time: 0.81 seconds

## BFS NS instance: NS_B1_B1_A2_B1_B1_A1_B2

### Backsubstitution after applying NS history:
0: -48.5488396, 52.7849350, -31.0816956, 32.8034973, -81.3523331, 83.8666229
1: -359.5079651, 119.8391647, -225.2608032, 77.2006378, -435.9344788, 345.0999756
2: -191.9619293, 113.1905823, -124.1734848, 71.2927322, -263.2546692, 237.3640747
3: -247.9404144, 90.6391602, -156.4770966, 57.0056229, -304.5505981, 247.1162567
4: -137.9670410, 100.1775360, -88.9962845, 61.9981041, -199.9651489, 189.1737976

Time for backsubstitution: 1.20 seconds

### NS candidates at layer 1
type: B, layer: 1, pos: 4
type: A, layer: 1, pos: 45
type: B, layer: 1, pos: 9
type: A, layer: 1, pos: 35
type: B, layer: 1, pos: 17
type: B, layer: 1, pos: 38
type: A, layer: 1, pos: 9
type: B, layer: 1, pos: 3
type: A, layer: 1, pos: 11
type: B, layer: 1, pos: 10
type: B, layer: 1, pos: 35
type: B, layer: 1, pos: 11
type: A, layer: 1, pos: 15
type: A, layer: 1, pos: 10
type: A, layer: 1, pos: 3
type: A, layer: 1, pos: 19
type: A, layer: 1, pos: 17
type: B, layer: 1, pos: 32
type: A, layer: 1, pos: 39
type: A, layer: 1, pos: 16
type: B, layer: 1, pos: 18
type: B, layer: 1, pos: 37
type: A, layer: 1, pos: 43
type: A, layer: 1, pos: 18
type: B, layer: 1, pos: 21
type: B, layer: 1, pos: 15
type: A, layer: 1, pos: 23
type: B, layer: 1, pos: 41
type: A, layer: 1, pos: 20
type: B, layer: 1, pos: 5
type: A, layer: 1, pos: 32
type: A, layer: 1, pos: 41
type: B, layer: 1, pos: 39
type: B, layer: 1, pos: 20
type: A, layer: 1, pos: 5
type: B, layer: 1, pos: 43
type: A, layer: 1, pos: 21
type: B, layer: 1, pos: 23
type: A, layer: 1, pos: 37
type: B, layer: 1, pos: 16
type: A, layer: 1, pos: 4
type: A, layer: 1, pos: 38
type: A, layer: 1, pos: 29

Time for candidate selection: 0.10 seconds

### Candidate
type: B, layer: 1, pos: 4

### Candidate
type: A, layer: 1, pos: 45

## Relational analysis of NS_B1_B1_A2_B1_B1_A1_B2_A1
Optimization infeasible because this subproblem isn't reachable.

### Candidate
type: B, layer: 1, pos: 9

### Candidate
type: A, layer: 1, pos: 35

### Candidate
type: B, layer: 1, pos: 17

### Candidate
type: B, layer: 1, pos: 38

### Candidate
type: A, layer: 1, pos: 9

### Candidate
type: B, layer: 1, pos: 3

### Candidate
type: A, layer: 1, pos: 11

## Relational analysis of NS_B1_B1_A2_B1_B1_A1_B2_A1

### Relational analysis result of NS_B1_B1_A2_B1_B1_A1_B2_A1
Status: Status.VERIFIED
Output dim: 0, lower bound: -63.3936276, upper bound: 63.3934150
time: 0.75 seconds

## Relational analysis of NS_B1_B1_A2_B1_B1_A1_B2_A2

### Relational analysis result of NS_B1_B1_A2_B1_B1_A1_B2_A2
Status: Status.UNKNOWN
Output dim: 0, lower bound: -63.3937960, upper bound: 63.3935081
time: 0.73 seconds

## BFS NS instance: NS_B1_B1_A2_B1_B1_A2_B1

### Backsubstitution after applying NS history:
0: -51.7545166, 56.4923897, -27.8771610, 29.2590733, -81.0135880, 84.3695450
1: -385.7383118, 127.9144745, -198.3176727, 69.1243134, -454.1490173, 326.2320862
2: -205.0044250, 121.0119858, -110.9151154, 63.1946526, -268.1990662, 231.9270935
3: -265.4873657, 97.0590515, -138.6566772, 50.6811104, -316.0455322, 235.7157135
4: -146.6306000, 107.1134415, -79.7050247, 55.1526260, -201.7832336, 186.8184357

Time for backsubstitution: 1.15 seconds

### NS candidates at layer 1
type: B, layer: 1, pos: 4
type: B, layer: 1, pos: 3
type: A, layer: 1, pos: 35
type: B, layer: 1, pos: 38
type: B, layer: 1, pos: 9
type: A, layer: 1, pos: 3
type: B, layer: 1, pos: 35
type: A, layer: 1, pos: 9
type: A, layer: 1, pos: 45
type: B, layer: 1, pos: 17
type: A, layer: 1, pos: 43
type: A, layer: 1, pos: 4
type: A, layer: 1, pos: 15
type: A, layer: 1, pos: 11
type: B, layer: 1, pos: 11
type: B, layer: 1, pos: 10
type: A, layer: 1, pos: 10
type: B, layer: 1, pos: 18
type: A, layer: 1, pos: 23
type: B, layer: 1, pos: 43
type: A, layer: 1, pos: 39
type: A, layer: 1, pos: 16
type: A, layer: 1, pos: 18
type: A, layer: 1, pos: 19
type: B, layer: 1, pos: 37
type: B, layer: 1, pos: 21
type: A, layer: 1, pos: 21
type: A, layer: 1, pos: 17
type: A, layer: 1, pos: 32
type: B, layer: 1, pos: 32
type: B, layer: 1, pos: 23
type: B, layer: 1, pos: 15
type: A, layer: 1, pos: 20
type: B, layer: 1, pos: 41
type: B, layer: 1, pos: 20
type: A, layer: 1, pos: 41
type: B, layer: 1, pos: 5
type: A, layer: 1, pos: 5
type: B, layer: 1, pos: 39
type: B, layer: 1, pos: 16
type: A, layer: 1, pos: 38
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
type: B, layer: 1, pos: 38

### Candidate
type: B, layer: 1, pos: 9

### Candidate
type: A, layer: 1, pos: 3

### Candidate
type: B, layer: 1, pos: 35

### Candidate
type: A, layer: 1, pos: 9

### Candidate
type: A, layer: 1, pos: 45

## Relational analysis of NS_B1_B1_A2_B1_B1_A2_B1_A1
Optimization infeasible because this subproblem isn't reachable.

### Candidate
type: B, layer: 1, pos: 17

### Candidate
type: A, layer: 1, pos: 43

### Candidate
type: A, layer: 1, pos: 4

### Candidate
type: A, layer: 1, pos: 15

### Candidate
type: A, layer: 1, pos: 11

## Relational analysis of NS_B1_B1_A2_B1_B1_A2_B1_A1

### Relational analysis result of NS_B1_B1_A2_B1_B1_A2_B1_A1
Status: Status.UNKNOWN
Output dim: 0, lower bound: -63.3937402, upper bound: 63.3935365
time: 0.77 seconds

## Relational analysis of NS_B1_B1_A2_B1_B1_A2_B1_A2

### Relational analysis result of NS_B1_B1_A2_B1_B1_A2_B1_A2
Status: Status.UNKNOWN
Output dim: 0, lower bound: -63.3938754, upper bound: 63.3936259
time: 1.40 seconds

## BFS NS instance: NS_B1_B1_A2_B1_B1_A2_B2

### Backsubstitution after applying NS history:
0: -48.6396942, 53.1130867, -31.0816956, 32.8034973, -81.4431839, 84.1947708
1: -362.9782104, 120.1607895, -225.2608032, 77.2006378, -439.8862000, 345.4215393
2: -192.9386444, 113.8085403, -124.1734848, 71.2927322, -264.2313843, 237.9820251
3: -249.9385223, 91.1783218, -156.4770966, 57.0056229, -306.9193115, 247.6553802
4: -137.9716644, 100.8438568, -88.9962845, 61.9981041, -199.9697723, 189.8401337

Time for backsubstitution: 1.06 seconds

### NS candidates at layer 1
type: B, layer: 1, pos: 4
type: B, layer: 1, pos: 3
type: A, layer: 1, pos: 35
type: B, layer: 1, pos: 9
type: A, layer: 1, pos: 3
type: B, layer: 1, pos: 38
type: A, layer: 1, pos: 45
type: B, layer: 1, pos: 35
type: A, layer: 1, pos: 9
type: B, layer: 1, pos: 17
type: A, layer: 1, pos: 43
type: A, layer: 1, pos: 4
type: A, layer: 1, pos: 15
type: A, layer: 1, pos: 11
type: B, layer: 1, pos: 10
type: B, layer: 1, pos: 11
type: B, layer: 1, pos: 18
type: B, layer: 1, pos: 43
type: A, layer: 1, pos: 10
type: A, layer: 1, pos: 19
type: A, layer: 1, pos: 23
type: A, layer: 1, pos: 18
type: B, layer: 1, pos: 21
type: A, layer: 1, pos: 39
type: A, layer: 1, pos: 17
type: B, layer: 1, pos: 37
type: A, layer: 1, pos: 16
type: B, layer: 1, pos: 32
type: A, layer: 1, pos: 21
type: A, layer: 1, pos: 32
type: B, layer: 1, pos: 15
type: B, layer: 1, pos: 23
type: A, layer: 1, pos: 20
type: B, layer: 1, pos: 41
type: B, layer: 1, pos: 20
type: B, layer: 1, pos: 5
type: A, layer: 1, pos: 41
type: B, layer: 1, pos: 39
type: A, layer: 1, pos: 5
type: B, layer: 1, pos: 16
type: A, layer: 1, pos: 37
type: A, layer: 1, pos: 38
type: A, layer: 1, pos: 29

Time for candidate selection: 0.09 seconds

### Candidate
type: B, layer: 1, pos: 4

### Candidate
type: B, layer: 1, pos: 3

### Candidate
type: A, layer: 1, pos: 35

### Candidate
type: B, layer: 1, pos: 9

### Candidate
type: A, layer: 1, pos: 3

### Candidate
type: B, layer: 1, pos: 38

### Candidate
type: A, layer: 1, pos: 45

## Relational analysis of NS_B1_B1_A2_B1_B1_A2_B2_A1
Optimization infeasible because this subproblem isn't reachable.

### Candidate
type: B, layer: 1, pos: 35

### Candidate
type: A, layer: 1, pos: 9

### Candidate
type: B, layer: 1, pos: 17

### Candidate
type: A, layer: 1, pos: 43

### Candidate
type: A, layer: 1, pos: 4

### Candidate
type: A, layer: 1, pos: 15

### Candidate
type: A, layer: 1, pos: 11

## Relational analysis of NS_B1_B1_A2_B1_B1_A2_B2_A1

### Relational analysis result of NS_B1_B1_A2_B1_B1_A2_B2_A1
Status: Status.UNKNOWN
Output dim: 0, lower bound: -63.3937031, upper bound: 63.3935409
time: 0.78 seconds

## Relational analysis of NS_B1_B1_A2_B1_B1_A2_B2_A2

### Relational analysis result of NS_B1_B1_A2_B1_B1_A2_B2_A2
Status: Status.UNKNOWN
Output dim: 0, lower bound: -63.3937031, upper bound: 63.3937738
time: 1.34 seconds

## BFS NS instance: NS_B1_B1_A2_B1_B2_A2_A1

### Backsubstitution after applying NS history:
0: -52.0763054, 56.6840820, -40.0382233, 41.1658821, -93.2421799, 96.6414795
1: -385.3255920, 128.5486145, -269.4214783, 99.0919952, -482.7860718, 397.9700317
2: -205.7944641, 121.3870697, -155.2223511, 89.0888367, -294.8832397, 275.8476257
3: -265.6591492, 97.3198700, -189.7288361, 71.2669220, -336.8185425, 286.8923340
4: -147.4133911, 107.5415268, -113.5094452, 77.9790039, -225.3923950, 220.0486755

Time for backsubstitution: 1.26 seconds

### NS candidates at layer 1
type: B, layer: 1, pos: 4
type: B, layer: 1, pos: 3
type: A, layer: 1, pos: 35
type: B, layer: 1, pos: 9
type: B, layer: 1, pos: 35
type: A, layer: 1, pos: 9
type: B, layer: 1, pos: 38
type: B, layer: 1, pos: 17
type: A, layer: 1, pos: 3
type: A, layer: 1, pos: 15
type: A, layer: 1, pos: 10
type: B, layer: 1, pos: 45
type: B, layer: 1, pos: 18
type: A, layer: 1, pos: 4
type: B, layer: 1, pos: 43
type: A, layer: 1, pos: 43
type: A, layer: 1, pos: 29
type: A, layer: 1, pos: 17
type: A, layer: 1, pos: 32
type: B, layer: 1, pos: 11
type: A, layer: 1, pos: 19
type: B, layer: 1, pos: 37
type: A, layer: 1, pos: 16
type: B, layer: 1, pos: 10
type: B, layer: 1, pos: 21
type: A, layer: 1, pos: 18
type: A, layer: 1, pos: 21
type: A, layer: 1, pos: 45
type: B, layer: 1, pos: 15
type: A, layer: 1, pos: 20
type: A, layer: 1, pos: 39
type: A, layer: 1, pos: 23
type: B, layer: 1, pos: 23
type: B, layer: 1, pos: 41
type: B, layer: 1, pos: 32
type: B, layer: 1, pos: 5
type: A, layer: 1, pos: 41
type: B, layer: 1, pos: 39
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
type: B, layer: 1, pos: 9

### Candidate
type: B, layer: 1, pos: 35

### Candidate
type: A, layer: 1, pos: 9

### Candidate
type: B, layer: 1, pos: 38

### Candidate
type: B, layer: 1, pos: 17

### Candidate
type: A, layer: 1, pos: 3

### Candidate
type: A, layer: 1, pos: 15

### Candidate
type: A, layer: 1, pos: 10

### Candidate
type: B, layer: 1, pos: 45

### Candidate
type: B, layer: 1, pos: 18

### Candidate
type: A, layer: 1, pos: 4

### Candidate
type: B, layer: 1, pos: 43

## Relational analysis of NS_B1_B1_A2_B1_B2_A2_A1_B1
Optimization infeasible because this subproblem isn't reachable.

### Candidate
type: A, layer: 1, pos: 43

### Candidate
type: A, layer: 1, pos: 29

## Relational analysis of NS_B1_B1_A2_B1_B2_A2_A1_A1
Optimization infeasible because this subproblem isn't reachable.

### Candidate
type: A, layer: 1, pos: 17

### Candidate
type: A, layer: 1, pos: 32

## Relational analysis of NS_B1_B1_A2_B1_B2_A2_A1_A1
Optimization infeasible because this subproblem isn't reachable.

### Candidate
type: B, layer: 1, pos: 11

## Relational analysis of NS_B1_B1_A2_B1_B2_A2_A1_B1

### Relational analysis result of NS_B1_B1_A2_B1_B2_A2_A1_B1
Status: Status.VERIFIED
Output dim: 0, lower bound: -63.3932232, upper bound: 63.3933566
time: 0.75 seconds

## Relational analysis of NS_B1_B1_A2_B1_B2_A2_A1_B2

### Relational analysis result of NS_B1_B1_A2_B1_B2_A2_A1_B2
Status: Status.UNKNOWN
Output dim: 0, lower bound: -63.3932232, upper bound: 63.3937074
time: 0.67 seconds

## BFS NS instance: NS_B1_B1_A2_B1_B2_A2_A2

### Backsubstitution after applying NS history:
0: -51.9449615, 56.5429306, -40.1849136, 41.3175163, -93.2624817, 96.6601486
1: -384.4512024, 128.2194519, -270.5454102, 99.4554672, -482.3673096, 398.7648621
2: -205.2312012, 121.0895844, -155.8203583, 89.4233932, -294.6546021, 276.1959534
3: -265.0025635, 97.0972214, -190.4992371, 71.5364838, -336.4794006, 287.4949036
4: -146.9905548, 107.2729721, -113.9260864, 78.2623062, -225.2528381, 220.2207947

Time for backsubstitution: 1.11 seconds

### NS candidates at layer 1
type: B, layer: 1, pos: 4
type: B, layer: 1, pos: 3
type: A, layer: 1, pos: 35
type: B, layer: 1, pos: 9
type: B, layer: 1, pos: 35
type: A, layer: 1, pos: 9
type: B, layer: 1, pos: 38
type: B, layer: 1, pos: 17
type: A, layer: 1, pos: 3
type: A, layer: 1, pos: 15
type: A, layer: 1, pos: 10
type: B, layer: 1, pos: 45
type: B, layer: 1, pos: 18
type: A, layer: 1, pos: 4
type: B, layer: 1, pos: 43
type: A, layer: 1, pos: 29
type: B, layer: 1, pos: 11
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
type: A, layer: 1, pos: 41
type: B, layer: 1, pos: 5
type: B, layer: 1, pos: 39
type: B, layer: 1, pos: 20
type: A, layer: 1, pos: 5
type: A, layer: 1, pos: 37
type: A, layer: 1, pos: 38
type: B, layer: 1, pos: 16

Time for candidate selection: 0.11 seconds

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
type: B, layer: 1, pos: 38

### Candidate
type: B, layer: 1, pos: 17

### Candidate
type: A, layer: 1, pos: 3

### Candidate
type: A, layer: 1, pos: 15

### Candidate
type: A, layer: 1, pos: 10

### Candidate
type: B, layer: 1, pos: 45

### Candidate
type: B, layer: 1, pos: 18

### Candidate
type: A, layer: 1, pos: 4

### Candidate
type: B, layer: 1, pos: 43

## Relational analysis of NS_B1_B1_A2_B1_B2_A2_A2_B1
Optimization infeasible because this subproblem isn't reachable.

### Candidate
type: A, layer: 1, pos: 29

## Relational analysis of NS_B1_B1_A2_B1_B2_A2_A2_A1
Optimization infeasible because this subproblem isn't reachable.

### Candidate
type: B, layer: 1, pos: 11

## Relational analysis of NS_B1_B1_A2_B1_B2_A2_A2_B1

### Relational analysis result of NS_B1_B1_A2_B1_B2_A2_A2_B1
Status: Status.VERIFIED
Output dim: 0, lower bound: -63.3933754, upper bound: 63.3934403
time: 0.80 seconds

## Relational analysis of NS_B1_B1_A2_B1_B2_A2_A2_B2

### Relational analysis result of NS_B1_B1_A2_B1_B2_A2_A2_B2
Status: Status.UNKNOWN
Output dim: 0, lower bound: -63.3933754, upper bound: 63.3938612
time: 0.77 seconds

## BFS NS instance: NS_B1_B1_A2_B2_B1_A1_A2

### Backsubstitution after applying NS history:
0: -52.3392639, 56.7671509, -33.7412872, 35.4420433, -87.7813034, 90.5084229
1: -385.6745911, 129.0892334, -241.4537048, 83.6052399, -468.1608887, 370.5429382
2: -206.3956909, 121.7010422, -134.5909882, 76.4633179, -282.8589783, 256.2919617
3: -266.1554260, 97.4824905, -168.7952271, 61.3642120, -327.1290283, 266.2776794
4: -148.3731384, 107.6178589, -96.5188828, 66.6376648, -215.0108032, 204.1367493

Time for backsubstitution: 1.16 seconds

### NS candidates at layer 1
type: B, layer: 1, pos: 4
type: B, layer: 1, pos: 9
type: B, layer: 1, pos: 3
type: A, layer: 1, pos: 35
type: B, layer: 1, pos: 38
type: B, layer: 1, pos: 17
type: B, layer: 1, pos: 45
type: A, layer: 1, pos: 9
type: B, layer: 1, pos: 11
type: A, layer: 1, pos: 10
type: B, layer: 1, pos: 35
type: A, layer: 1, pos: 15
type: B, layer: 1, pos: 10
type: A, layer: 1, pos: 39
type: A, layer: 1, pos: 3
type: A, layer: 1, pos: 17
type: A, layer: 1, pos: 16
type: B, layer: 1, pos: 37
type: B, layer: 1, pos: 18
type: A, layer: 1, pos: 18
type: A, layer: 1, pos: 45
type: A, layer: 1, pos: 43
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

Time for candidate selection: 0.09 seconds

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

### Candidate
type: A, layer: 1, pos: 9

### Candidate
type: B, layer: 1, pos: 11

## Relational analysis of NS_B1_B1_A2_B2_B1_A1_A2_B1
Optimization infeasible because this subproblem isn't reachable.

### Candidate
type: A, layer: 1, pos: 10

## Relational analysis of NS_B1_B1_A2_B2_B1_A1_A2_A1
Optimization infeasible because this subproblem isn't reachable.

### Candidate
type: B, layer: 1, pos: 35

### Candidate
type: A, layer: 1, pos: 15

## Relational analysis of NS_B1_B1_A2_B2_B1_A1_A2_A1
Optimization infeasible because this subproblem isn't reachable.

### Candidate
type: B, layer: 1, pos: 10

### Candidate
type: A, layer: 1, pos: 39

## Relational analysis of NS_B1_B1_A2_B2_B1_A1_A2_A1
Optimization infeasible because this subproblem isn't reachable.

### Candidate
type: A, layer: 1, pos: 3

### Candidate
type: A, layer: 1, pos: 17

### Candidate
type: A, layer: 1, pos: 16

## Relational analysis of NS_B1_B1_A2_B2_B1_A1_A2_A1
Optimization infeasible because this subproblem isn't reachable.

### Candidate
type: B, layer: 1, pos: 37

## Relational analysis of NS_B1_B1_A2_B2_B1_A1_A2_B1
Optimization infeasible because this subproblem isn't reachable.

### Candidate
type: B, layer: 1, pos: 18

## Relational analysis of NS_B1_B1_A2_B2_B1_A1_A2_B1
Optimization infeasible because this subproblem isn't reachable.

### Candidate
type: A, layer: 1, pos: 18

### Candidate
type: A, layer: 1, pos: 45

### Candidate
type: A, layer: 1, pos: 43

## Relational analysis of NS_B1_B1_A2_B2_B1_A1_A2_A1
Optimization infeasible because this subproblem isn't reachable.

### Candidate
type: B, layer: 1, pos: 15

### Candidate
type: A, layer: 1, pos: 19

## Relational analysis of NS_B1_B1_A2_B2_B1_A1_A2_A1
Optimization infeasible because this subproblem isn't reachable.

### Candidate
type: A, layer: 1, pos: 23

## Relational analysis of NS_B1_B1_A2_B2_B1_A1_A2_A1
Optimization infeasible because this subproblem isn't reachable.

### Candidate
type: A, layer: 1, pos: 32

## Relational analysis of NS_B1_B1_A2_B2_B1_A1_A2_A1
Optimization infeasible because this subproblem isn't reachable.

### Candidate
type: B, layer: 1, pos: 32

### Candidate
type: B, layer: 1, pos: 41

## Relational analysis of NS_B1_B1_A2_B2_B1_A1_A2_B1
Optimization infeasible because this subproblem isn't reachable.

### Candidate
type: B, layer: 1, pos: 21

## Relational analysis of NS_B1_B1_A2_B2_B1_A1_A2_B1
Optimization infeasible because this subproblem isn't reachable.

### Candidate
type: A, layer: 1, pos: 41

### Candidate
type: B, layer: 1, pos: 5

## Relational analysis of NS_B1_B1_A2_B2_B1_A1_A2_B1
Optimization infeasible because this subproblem isn't reachable.

### Candidate
type: A, layer: 1, pos: 20

## Relational analysis of NS_B1_B1_A2_B2_B1_A1_A2_A1
Optimization infeasible because this subproblem isn't reachable.

### Candidate
type: B, layer: 1, pos: 20

### Candidate
type: A, layer: 1, pos: 5

### Candidate
type: A, layer: 1, pos: 21

### Candidate
type: B, layer: 1, pos: 39

### Candidate
type: B, layer: 1, pos: 43

### Candidate
type: B, layer: 1, pos: 23

### Candidate
type: A, layer: 1, pos: 37

### Candidate
type: B, layer: 1, pos: 16

### Candidate
type: A, layer: 1, pos: 4

### Candidate
type: A, layer: 1, pos: 38

### Candidate
type: A, layer: 1, pos: 29

## Relational analysis of NS_B1_B1_A2_B2_B1_A1_A2_A1
Optimization infeasible because this subproblem isn't reachable.

## BFS NS instance: NS_B1_B1_A2_B2_B1_A2_A1

### Backsubstitution after applying NS history:
0: -52.8124008, 57.5542374, -33.6359291, 35.3283958, -88.1407928, 91.1901627
1: -392.8023071, 130.4674835, -240.6488953, 83.3413162, -475.3539734, 371.1163940
2: -209.0969238, 123.2988205, -134.1670227, 76.2076187, -285.3045044, 257.4658508
3: -270.5476379, 98.8694534, -168.2424164, 61.1583519, -331.5967102, 267.1117859
4: -149.6274109, 109.1186142, -96.2193832, 66.4189072, -216.0463257, 205.3379974

Time for backsubstitution: 1.07 seconds

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
type: A, layer: 1, pos: 4
type: A, layer: 1, pos: 10
type: B, layer: 1, pos: 18
type: B, layer: 1, pos: 10
type: A, layer: 1, pos: 23
type: A, layer: 1, pos: 39
type: B, layer: 1, pos: 11
type: A, layer: 1, pos: 18
type: B, layer: 1, pos: 43
type: A, layer: 1, pos: 45
type: A, layer: 1, pos: 32
type: A, layer: 1, pos: 17
type: B, layer: 1, pos: 37
type: B, layer: 1, pos: 15
type: B, layer: 1, pos: 21
type: A, layer: 1, pos: 16
type: A, layer: 1, pos: 21
type: A, layer: 1, pos: 19
type: B, layer: 1, pos: 20
type: B, layer: 1, pos: 41
type: B, layer: 1, pos: 23
type: A, layer: 1, pos: 20
type: A, layer: 1, pos: 41
type: B, layer: 1, pos: 5
type: B, layer: 1, pos: 32
type: A, layer: 1, pos: 5
type: B, layer: 1, pos: 39
type: B, layer: 1, pos: 16
type: A, layer: 1, pos: 37
type: A, layer: 1, pos: 38
type: A, layer: 1, pos: 29

Time for candidate selection: 0.09 seconds

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

### Candidate
type: A, layer: 1, pos: 43

### Candidate
type: A, layer: 1, pos: 15

### Candidate
type: A, layer: 1, pos: 4

### Candidate
type: A, layer: 1, pos: 10

## Relational analysis of NS_B1_B1_A2_B2_B1_A2_A1_A1
Optimization infeasible because this subproblem isn't reachable.

### Candidate
type: B, layer: 1, pos: 18

## Relational analysis of NS_B1_B1_A2_B2_B1_A2_A1_B1
Optimization infeasible because this subproblem isn't reachable.

### Candidate
type: B, layer: 1, pos: 10

### Candidate
type: A, layer: 1, pos: 23

## Relational analysis of NS_B1_B1_A2_B2_B1_A2_A1_A1
Optimization infeasible because this subproblem isn't reachable.

### Candidate
type: A, layer: 1, pos: 39

## Relational analysis of NS_B1_B1_A2_B2_B1_A2_A1_A1
Optimization infeasible because this subproblem isn't reachable.

### Candidate
type: B, layer: 1, pos: 11

## Relational analysis of NS_B1_B1_A2_B2_B1_A2_A1_B1
Optimization infeasible because this subproblem isn't reachable.

### Candidate
type: A, layer: 1, pos: 18

### Candidate
type: B, layer: 1, pos: 43

### Candidate
type: A, layer: 1, pos: 45

### Candidate
type: A, layer: 1, pos: 32

## Relational analysis of NS_B1_B1_A2_B2_B1_A2_A1_A1
Optimization infeasible because this subproblem isn't reachable.

### Candidate
type: A, layer: 1, pos: 17

### Candidate
type: B, layer: 1, pos: 37

## Relational analysis of NS_B1_B1_A2_B2_B1_A2_A1_B1
Optimization infeasible because this subproblem isn't reachable.

### Candidate
type: B, layer: 1, pos: 15

### Candidate
type: B, layer: 1, pos: 21

## Relational analysis of NS_B1_B1_A2_B2_B1_A2_A1_B1
Optimization infeasible because this subproblem isn't reachable.

### Candidate
type: A, layer: 1, pos: 16

## Relational analysis of NS_B1_B1_A2_B2_B1_A2_A1_A1
Optimization infeasible because this subproblem isn't reachable.

### Candidate
type: A, layer: 1, pos: 21

### Candidate
type: A, layer: 1, pos: 19

## Relational analysis of NS_B1_B1_A2_B2_B1_A2_A1_A1
Optimization infeasible because this subproblem isn't reachable.

### Candidate
type: B, layer: 1, pos: 20

## Relational analysis of NS_B1_B1_A2_B2_B1_A2_A1_B1
Optimization infeasible because this subproblem isn't reachable.

### Candidate
type: B, layer: 1, pos: 41

## Relational analysis of NS_B1_B1_A2_B2_B1_A2_A1_B1
Optimization infeasible because this subproblem isn't reachable.

### Candidate
type: B, layer: 1, pos: 23

### Candidate
type: A, layer: 1, pos: 20

### Candidate
type: A, layer: 1, pos: 41

### Candidate
type: B, layer: 1, pos: 5

## Relational analysis of NS_B1_B1_A2_B2_B1_A2_A1_B1
Optimization infeasible because this subproblem isn't reachable.

### Candidate
type: B, layer: 1, pos: 32

### Candidate
type: A, layer: 1, pos: 5

### Candidate
type: B, layer: 1, pos: 39

### Candidate
type: B, layer: 1, pos: 16

### Candidate
type: A, layer: 1, pos: 37

### Candidate
type: A, layer: 1, pos: 38

### Candidate
type: A, layer: 1, pos: 29

## Relational analysis of NS_B1_B1_A2_B2_B1_A2_A1_A1
Optimization infeasible because this subproblem isn't reachable.

## BFS NS instance: NS_B1_B1_A2_B2_B1_A2_A2

### Backsubstitution after applying NS history:
0: -52.4561806, 57.1548615, -33.7412872, 35.4420433, -87.8982239, 90.8961334
1: -389.9511414, 129.5919495, -241.4537048, 83.6052399, -472.8885498, 371.0456238
2: -207.6702271, 122.4413910, -134.5909882, 76.4633179, -284.1334839, 257.0323792
3: -268.6185303, 98.2042770, -168.7952271, 61.3642120, -329.9345093, 266.9994812
4: -148.5932465, 108.3678131, -96.5188828, 66.6376648, -215.2309113, 204.8866882

Time for backsubstitution: 1.27 seconds

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
type: A, layer: 1, pos: 4
type: A, layer: 1, pos: 10
type: B, layer: 1, pos: 18
type: B, layer: 1, pos: 11
type: B, layer: 1, pos: 10
type: A, layer: 1, pos: 23
type: A, layer: 1, pos: 39
type: A, layer: 1, pos: 18
type: B, layer: 1, pos: 43
type: A, layer: 1, pos: 45
type: A, layer: 1, pos: 17
type: A, layer: 1, pos: 32
type: B, layer: 1, pos: 37
type: B, layer: 1, pos: 15
type: A, layer: 1, pos: 16
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

Time for candidate selection: 0.11 seconds

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

### Candidate
type: A, layer: 1, pos: 43

### Candidate
type: A, layer: 1, pos: 15

### Candidate
type: A, layer: 1, pos: 4

### Candidate
type: A, layer: 1, pos: 10

## Relational analysis of NS_B1_B1_A2_B2_B1_A2_A2_A1
Optimization infeasible because this subproblem isn't reachable.

### Candidate
type: B, layer: 1, pos: 18

## Relational analysis of NS_B1_B1_A2_B2_B1_A2_A2_B1
Optimization infeasible because this subproblem isn't reachable.

## NS Result
status: Status.UNKNOWN
execution time: (base) + (ns) = 3.26 + 416.85 = 420.12 seconds
