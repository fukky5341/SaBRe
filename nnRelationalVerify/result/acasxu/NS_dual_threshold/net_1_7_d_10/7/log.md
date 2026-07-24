## Execution arguments:
Dataset: Dataset.ACAS
Network: ds/onnx/acasxu_op11/ACASXU_1_7.onnx
Relational property: GLOBAL_ROBUSTNESS
LP Analysis: True
Epsilon: None
Delta epsilon: 0.1
execution index: (10, None, 7)
Time budget: 420 seconds
Split limit: 100
Threshold: 81.860902399251


## IAR start

### BASE IAR bounds
Layer (inp1_lb, inp1_ub, inp2_lb, inp2_ub, d_lb, d_ub)
0: (-20.8385620, 76.1094437, -20.8385620, 76.1094437, -96.9480057, 96.9480057)
1: (-55.3152542, 171.7260437, -55.3152542, 171.7260437, -227.0412903, 227.0412903)
2: (-82.9352112, 152.8937378, -82.9352112, 152.8937378, -235.8289490, 235.8289490)
3: (-47.5881386, 183.4647217, -47.5881386, 183.4647217, -231.0528564, 231.0528564)
4: (-75.8024445, 134.6393127, -75.8024445, 134.6393127, -210.4417572, 210.4417572)

## BASE Relational Analysis

## BASE Result
execution time: IAR + RelationalAnalysis = 1.35 + 3.52 = 4.86 seconds
status: Status.UNKNOWN
relational distance
Output dim: 0, lower bound: -81.8633583, upper bound: 81.8633583

# Neuron Split (NS) starts

## BFS NS instance: NS

Time for backsubstitution: 0.01 seconds

### NS candidates at layer 1
type: A, layer: 1, pos: 7
type: B, layer: 1, pos: 7
type: A, layer: 1, pos: 43
type: B, layer: 1, pos: 43
type: B, layer: 1, pos: 41
type: A, layer: 1, pos: 41
type: A, layer: 1, pos: 13
type: B, layer: 1, pos: 13
type: B, layer: 1, pos: 19
type: A, layer: 1, pos: 19
type: A, layer: 1, pos: 25
type: B, layer: 1, pos: 25
type: A, layer: 1, pos: 16
type: B, layer: 1, pos: 16
type: B, layer: 1, pos: 26
type: A, layer: 1, pos: 26
type: B, layer: 1, pos: 46
type: A, layer: 1, pos: 46
type: B, layer: 1, pos: 4
type: A, layer: 1, pos: 4
type: A, layer: 1, pos: 29
type: B, layer: 1, pos: 29
type: B, layer: 1, pos: 21
type: A, layer: 1, pos: 21
type: A, layer: 1, pos: 38
type: B, layer: 1, pos: 38
type: B, layer: 1, pos: 22
type: A, layer: 1, pos: 22
type: A, layer: 1, pos: 12
type: B, layer: 1, pos: 12
type: A, layer: 1, pos: 37
type: B, layer: 1, pos: 37
type: B, layer: 1, pos: 14
type: A, layer: 1, pos: 14
type: A, layer: 1, pos: 8
type: B, layer: 1, pos: 8
type: A, layer: 1, pos: 6
type: B, layer: 1, pos: 6
type: A, layer: 1, pos: 31
type: B, layer: 1, pos: 31
type: B, layer: 1, pos: 39
type: A, layer: 1, pos: 39
type: B, layer: 1, pos: 20
type: A, layer: 1, pos: 20
type: A, layer: 1, pos: 23
type: B, layer: 1, pos: 23
type: A, layer: 1, pos: 42
type: B, layer: 1, pos: 42
type: B, layer: 1, pos: 11
type: A, layer: 1, pos: 11
type: A, layer: 1, pos: 15
type: B, layer: 1, pos: 15

Time for candidate selection: 0.10 seconds

### Candidate
type: A, layer: 1, pos: 7

## Relational analysis of NS_A1

### Relational analysis result of NS_A1
Status: Status.UNKNOWN
Output dim: 0, lower bound: -81.8633583, upper bound: 81.8632828
time: 1.60 seconds

## Relational analysis of NS_A2

### Relational analysis result of NS_A2
Status: Status.UNKNOWN
Output dim: 0, lower bound: -81.8633583, upper bound: 81.8633583
time: 0.76 seconds

## Summary of splitting at layer (split count: 0)
- Time for NS candidates: 2.49 seconds
NS_A1, status: Status.UNKNOWN, split count: 1, time: 2.49
Output dim: 0, lower bound: -81.8633583, upper bound: 81.8632828
NS_A2, status: Status.UNKNOWN, split count: 1, time: 2.49
Output dim: 0, lower bound: -81.8633583, upper bound: 81.8633583

## BFS NS instance: NS_A1

### Backsubstitution after applying NS history:
0: -19.8369217, 72.4063263, -20.8385620, 76.1094437, -95.9463501, 93.2448883
1: -52.7576790, 163.3856506, -55.3152542, 171.7260437, -224.4836884, 218.7008972
2: -79.4177856, 145.0854797, -82.9352112, 152.8937378, -232.3115082, 228.0206909
3: -45.3945847, 174.7305145, -47.5881386, 183.4647217, -228.8593140, 222.3186493
4: -72.5078278, 127.8521042, -75.8024445, 134.6393127, -207.1471405, 203.6545410

Time for backsubstitution: 0.84 seconds

### NS candidates at layer 1
type: B, layer: 1, pos: 43
type: A, layer: 1, pos: 43
type: B, layer: 1, pos: 41
type: B, layer: 1, pos: 7
type: B, layer: 1, pos: 13
type: A, layer: 1, pos: 13
type: A, layer: 1, pos: 41
type: B, layer: 1, pos: 19
type: A, layer: 1, pos: 19
type: A, layer: 1, pos: 25
type: B, layer: 1, pos: 25
type: B, layer: 1, pos: 16
type: A, layer: 1, pos: 16
type: B, layer: 1, pos: 26
type: A, layer: 1, pos: 26
type: A, layer: 1, pos: 4
type: B, layer: 1, pos: 46
type: A, layer: 1, pos: 46
type: B, layer: 1, pos: 29
type: B, layer: 1, pos: 21
type: B, layer: 1, pos: 4
type: B, layer: 1, pos: 38
type: A, layer: 1, pos: 29
type: A, layer: 1, pos: 38
type: A, layer: 1, pos: 21
type: B, layer: 1, pos: 22
type: B, layer: 1, pos: 12
type: A, layer: 1, pos: 12
type: A, layer: 1, pos: 22
type: B, layer: 1, pos: 37
type: A, layer: 1, pos: 37
type: B, layer: 1, pos: 14
type: B, layer: 1, pos: 8
type: A, layer: 1, pos: 8
type: A, layer: 1, pos: 6
type: A, layer: 1, pos: 14
type: B, layer: 1, pos: 6
type: B, layer: 1, pos: 39
type: A, layer: 1, pos: 31
type: B, layer: 1, pos: 31
type: A, layer: 1, pos: 20
type: A, layer: 1, pos: 23
type: A, layer: 1, pos: 39
type: A, layer: 1, pos: 42
type: B, layer: 1, pos: 23
type: A, layer: 1, pos: 11
type: B, layer: 1, pos: 20
type: B, layer: 1, pos: 42
type: B, layer: 1, pos: 11
type: B, layer: 1, pos: 15
type: A, layer: 1, pos: 15

Time for candidate selection: 0.07 seconds

### Candidate
type: B, layer: 1, pos: 43

## Relational analysis of NS_A1_B1

### Relational analysis result of NS_A1_B1
Status: Status.UNKNOWN
Output dim: 0, lower bound: -81.8616233, upper bound: 81.8610823
time: 0.84 seconds

## Relational analysis of NS_A1_B2

### Relational analysis result of NS_A1_B2
Status: Status.VERIFIED
Output dim: 0, lower bound: -81.8605070, upper bound: 81.8605891
time: 0.82 seconds

## BFS NS instance: NS_A2

### Backsubstitution after applying NS history:
0: -20.6764908, 75.5103683, -20.8385620, 76.1094437, -96.7859344, 96.3489304
1: -54.8931732, 170.3698120, -55.3152542, 171.7260437, -226.6192169, 225.6850586
2: -82.3216171, 151.6371460, -82.9352112, 152.8937378, -235.2153625, 234.5723419
3: -47.2267761, 182.0128632, -47.5881386, 183.4647217, -230.6914825, 229.6009979
4: -75.2349548, 133.5453339, -75.8024445, 134.6393127, -209.8742676, 209.3477631

Time for backsubstitution: 0.80 seconds

### NS candidates at layer 1
type: A, layer: 1, pos: 43
type: B, layer: 1, pos: 43
type: B, layer: 1, pos: 7
type: B, layer: 1, pos: 41
type: A, layer: 1, pos: 41
type: B, layer: 1, pos: 13
type: A, layer: 1, pos: 13
type: B, layer: 1, pos: 19
type: A, layer: 1, pos: 19
type: A, layer: 1, pos: 25
type: B, layer: 1, pos: 25
type: B, layer: 1, pos: 16
type: A, layer: 1, pos: 16
type: B, layer: 1, pos: 26
type: A, layer: 1, pos: 26
type: A, layer: 1, pos: 4
type: B, layer: 1, pos: 46
type: A, layer: 1, pos: 46
type: B, layer: 1, pos: 4
type: B, layer: 1, pos: 29
type: A, layer: 1, pos: 29
type: A, layer: 1, pos: 21
type: B, layer: 1, pos: 21
type: A, layer: 1, pos: 38
type: B, layer: 1, pos: 38
type: B, layer: 1, pos: 12
type: A, layer: 1, pos: 22
type: B, layer: 1, pos: 22
type: A, layer: 1, pos: 12
type: A, layer: 1, pos: 37
type: B, layer: 1, pos: 37
type: B, layer: 1, pos: 14
type: A, layer: 1, pos: 8
type: A, layer: 1, pos: 14
type: B, layer: 1, pos: 8
type: A, layer: 1, pos: 6
type: B, layer: 1, pos: 6
type: A, layer: 1, pos: 31
type: B, layer: 1, pos: 31
type: B, layer: 1, pos: 39
type: A, layer: 1, pos: 39
type: A, layer: 1, pos: 20
type: A, layer: 1, pos: 23
type: B, layer: 1, pos: 23
type: A, layer: 1, pos: 42
type: B, layer: 1, pos: 20
type: B, layer: 1, pos: 42
type: A, layer: 1, pos: 11
type: B, layer: 1, pos: 11
type: B, layer: 1, pos: 15
type: A, layer: 1, pos: 15

Time for candidate selection: 0.07 seconds

### Candidate
type: A, layer: 1, pos: 43

## Relational analysis of NS_A2_A1

### Relational analysis result of NS_A2_A1
Status: Status.UNKNOWN
Output dim: 0, lower bound: -81.8608128, upper bound: 81.8614179
time: 1.02 seconds

## Relational analysis of NS_A2_A2

### Relational analysis result of NS_A2_A2
Status: Status.VERIFIED
Output dim: 0, lower bound: -81.8603873, upper bound: 81.8603873
time: 0.73 seconds

## Summary of splitting at layer (split count: 1)
- Time for NS candidates: 2.63 seconds
NS_A1_B1, status: Status.UNKNOWN, split count: 2, time: 2.63
Output dim: 0, lower bound: -81.8616233, upper bound: 81.8610823
NS_A1_B2, status: Status.VERIFIED, split count: 2, time: 2.63
Output dim: 0, lower bound: -81.8605070, upper bound: 81.8605891
NS_A2_A1, status: Status.UNKNOWN, split count: 2, time: 2.63
Output dim: 0, lower bound: -81.8608128, upper bound: 81.8614179
NS_A2_A2, status: Status.VERIFIED, split count: 2, time: 2.63
Output dim: 0, lower bound: -81.8603873, upper bound: 81.8603873

## BFS NS instance: NS_A1_B1

### Backsubstitution after applying NS history:
0: -19.8369217, 72.4063263, -20.7887135, 75.9263153, -95.7632294, 93.1950302
1: -52.7576790, 163.3856506, -55.1828003, 171.3142395, -224.0718994, 218.5684509
2: -79.4177856, 145.0854797, -82.7363739, 152.5190277, -231.9367981, 227.8218536
3: -45.3945847, 174.7305145, -47.4739609, 183.0247803, -228.4193726, 222.2044678
4: -72.5078278, 127.8521042, -75.6199112, 134.3123627, -206.8201904, 203.4720154

Time for backsubstitution: 0.96 seconds

### NS candidates at layer 1
type: B, layer: 1, pos: 41
type: B, layer: 1, pos: 7
type: B, layer: 1, pos: 13
type: A, layer: 1, pos: 13
type: A, layer: 1, pos: 41
type: B, layer: 1, pos: 19
type: A, layer: 1, pos: 19
type: A, layer: 1, pos: 25
type: B, layer: 1, pos: 25
type: A, layer: 1, pos: 16
type: B, layer: 1, pos: 16
type: B, layer: 1, pos: 26
type: A, layer: 1, pos: 26
type: A, layer: 1, pos: 4
type: B, layer: 1, pos: 46
type: A, layer: 1, pos: 46
type: B, layer: 1, pos: 29
type: B, layer: 1, pos: 21
type: B, layer: 1, pos: 38
type: B, layer: 1, pos: 4
type: A, layer: 1, pos: 29
type: A, layer: 1, pos: 21
type: A, layer: 1, pos: 38
type: B, layer: 1, pos: 22
type: A, layer: 1, pos: 43
type: B, layer: 1, pos: 12
type: A, layer: 1, pos: 12
type: A, layer: 1, pos: 22
type: B, layer: 1, pos: 37
type: A, layer: 1, pos: 37
type: B, layer: 1, pos: 14
type: B, layer: 1, pos: 8
type: A, layer: 1, pos: 8
type: A, layer: 1, pos: 14
type: A, layer: 1, pos: 6
type: B, layer: 1, pos: 6
type: B, layer: 1, pos: 39
type: A, layer: 1, pos: 31
type: B, layer: 1, pos: 31
type: A, layer: 1, pos: 20
type: A, layer: 1, pos: 39
type: A, layer: 1, pos: 23
type: A, layer: 1, pos: 42
type: B, layer: 1, pos: 23
type: A, layer: 1, pos: 11
type: B, layer: 1, pos: 42
type: B, layer: 1, pos: 20
type: B, layer: 1, pos: 11
type: B, layer: 1, pos: 15
type: A, layer: 1, pos: 15

Time for candidate selection: 0.08 seconds

### Candidate
type: B, layer: 1, pos: 41

## Relational analysis of NS_A1_B1_B1

### Relational analysis result of NS_A1_B1_B1
Status: Status.VERIFIED
Output dim: 0, lower bound: -81.8605862, upper bound: 81.8597364
time: 0.64 seconds

## Relational analysis of NS_A1_B1_B2

### Relational analysis result of NS_A1_B1_B2
Status: Status.VERIFIED
Output dim: 0, lower bound: -81.8600311, upper bound: 81.8595173
time: 1.22 seconds

## BFS NS instance: NS_A2_A1

### Backsubstitution after applying NS history:
0: -20.6270142, 75.3285370, -20.8385620, 76.1094437, -96.7364502, 96.1670990
1: -54.7614517, 169.9616241, -55.3152542, 171.7260437, -226.4874878, 225.2768860
2: -82.1236191, 151.2655640, -82.9352112, 152.8937378, -235.0173340, 234.2007751
3: -47.1132736, 181.5767212, -47.5881386, 183.4647217, -230.5779877, 229.1648560
4: -75.0532150, 133.2210693, -75.8024445, 134.6393127, -209.6925354, 209.0235138

Time for backsubstitution: 1.46 seconds

### NS candidates at layer 1
type: B, layer: 1, pos: 7
type: B, layer: 1, pos: 41
type: A, layer: 1, pos: 41
type: B, layer: 1, pos: 13
type: A, layer: 1, pos: 13
type: B, layer: 1, pos: 19
type: A, layer: 1, pos: 19
type: A, layer: 1, pos: 25
type: B, layer: 1, pos: 25
type: B, layer: 1, pos: 16
type: A, layer: 1, pos: 16
type: B, layer: 1, pos: 26
type: A, layer: 1, pos: 26
type: B, layer: 1, pos: 46
type: A, layer: 1, pos: 4
type: A, layer: 1, pos: 46
type: B, layer: 1, pos: 4
type: A, layer: 1, pos: 29
type: B, layer: 1, pos: 29
type: B, layer: 1, pos: 21
type: A, layer: 1, pos: 38
type: A, layer: 1, pos: 21
type: B, layer: 1, pos: 38
type: B, layer: 1, pos: 43
type: A, layer: 1, pos: 22
type: B, layer: 1, pos: 12
type: B, layer: 1, pos: 22
type: A, layer: 1, pos: 12
type: A, layer: 1, pos: 37
type: B, layer: 1, pos: 37
type: B, layer: 1, pos: 14
type: A, layer: 1, pos: 8
type: B, layer: 1, pos: 8
type: A, layer: 1, pos: 14
type: B, layer: 1, pos: 6
type: A, layer: 1, pos: 6
type: A, layer: 1, pos: 31
type: B, layer: 1, pos: 31
type: B, layer: 1, pos: 39
type: A, layer: 1, pos: 39
type: A, layer: 1, pos: 20
type: A, layer: 1, pos: 23
type: B, layer: 1, pos: 23
type: B, layer: 1, pos: 20
type: A, layer: 1, pos: 42
type: B, layer: 1, pos: 42
type: A, layer: 1, pos: 11
type: B, layer: 1, pos: 11
type: B, layer: 1, pos: 15
type: A, layer: 1, pos: 15

Time for candidate selection: 0.12 seconds

### Candidate
type: B, layer: 1, pos: 7

## Relational analysis of NS_A2_A1_B1

### Relational analysis result of NS_A2_A1_B1
Status: Status.UNKNOWN
Output dim: 0, lower bound: -81.8608128, upper bound: 81.8614179
time: 0.81 seconds

## Relational analysis of NS_A2_A1_B2

### Relational analysis result of NS_A2_A1_B2
Status: Status.UNKNOWN
Output dim: 0, lower bound: -81.8608128, upper bound: 81.8614179
time: 1.22 seconds

## Summary of splitting at layer (split count: 2)
- Time for NS candidates: 3.63 seconds
NS_A1_B1_B1, status: Status.VERIFIED, split count: 3, time: 3.63
Output dim: 0, lower bound: -81.8605862, upper bound: 81.8597364
NS_A1_B1_B2, status: Status.VERIFIED, split count: 3, time: 3.63
Output dim: 0, lower bound: -81.8600311, upper bound: 81.8595173
NS_A2_A1_B1, status: Status.UNKNOWN, split count: 3, time: 3.63
Output dim: 0, lower bound: -81.8608128, upper bound: 81.8614179
NS_A2_A1_B2, status: Status.UNKNOWN, split count: 3, time: 3.63
Output dim: 0, lower bound: -81.8608128, upper bound: 81.8614179

## BFS NS instance: NS_A2_A1_B1

### Backsubstitution after applying NS history:
0: -20.6270142, 75.3285370, -19.8369217, 72.4063263, -93.0333328, 95.1654510
1: -54.7614517, 169.9616241, -52.7576790, 163.3856506, -218.1470947, 222.7192841
2: -82.1236191, 151.2655640, -79.4177856, 145.0854797, -227.2090912, 230.6833191
3: -47.1132736, 181.5767212, -45.3945847, 174.7305145, -221.8437805, 226.9713135
4: -75.0532150, 133.2210693, -72.5078278, 127.8521042, -202.9053192, 205.7288971

Time for backsubstitution: 1.47 seconds

### NS candidates at layer 1
type: A, layer: 1, pos: 41
type: A, layer: 1, pos: 13
type: B, layer: 1, pos: 13
type: B, layer: 1, pos: 41
type: A, layer: 1, pos: 19
type: B, layer: 1, pos: 19
type: B, layer: 1, pos: 25
type: A, layer: 1, pos: 25
type: B, layer: 1, pos: 16
type: A, layer: 1, pos: 16
type: A, layer: 1, pos: 26
type: B, layer: 1, pos: 26
type: B, layer: 1, pos: 4
type: A, layer: 1, pos: 46
type: B, layer: 1, pos: 46
type: A, layer: 1, pos: 29
type: A, layer: 1, pos: 4
type: A, layer: 1, pos: 21
type: A, layer: 1, pos: 38
type: B, layer: 1, pos: 29
type: B, layer: 1, pos: 21
type: B, layer: 1, pos: 38
type: A, layer: 1, pos: 22
type: B, layer: 1, pos: 43
type: A, layer: 1, pos: 12
type: B, layer: 1, pos: 12
type: B, layer: 1, pos: 22
type: A, layer: 1, pos: 37
type: B, layer: 1, pos: 37
type: A, layer: 1, pos: 14
type: A, layer: 1, pos: 8
type: B, layer: 1, pos: 8
type: B, layer: 1, pos: 14
type: B, layer: 1, pos: 6
type: A, layer: 1, pos: 6
type: A, layer: 1, pos: 39
type: B, layer: 1, pos: 31
type: A, layer: 1, pos: 31
type: B, layer: 1, pos: 20
type: B, layer: 1, pos: 39
type: B, layer: 1, pos: 23
type: B, layer: 1, pos: 42
type: A, layer: 1, pos: 23
type: B, layer: 1, pos: 11
type: A, layer: 1, pos: 42
type: A, layer: 1, pos: 20
type: A, layer: 1, pos: 11
type: B, layer: 1, pos: 15
type: A, layer: 1, pos: 15

Time for candidate selection: 0.12 seconds

### Candidate
type: A, layer: 1, pos: 41

## Relational analysis of NS_A2_A1_B1_A1
Optimization infeasible because this subproblem isn't reachable.

### Candidate
type: A, layer: 1, pos: 13

## Relational analysis of NS_A2_A1_B1_A1

### Relational analysis result of NS_A2_A1_B1_A1
Status: Status.VERIFIED
Output dim: 0, lower bound: -81.8605164, upper bound: 81.8608939
time: 0.87 seconds

## Relational analysis of NS_A2_A1_B1_A2

### Relational analysis result of NS_A2_A1_B1_A2
Status: Status.UNKNOWN
Output dim: 0, lower bound: -81.8608126, upper bound: 81.8613877
time: 1.04 seconds

## BFS NS instance: NS_A2_A1_B2

### Backsubstitution after applying NS history:
0: -20.6270142, 75.3285370, -20.6764908, 75.5103683, -96.1373672, 96.0050278
1: -54.7614517, 169.9616241, -54.8931732, 170.3698120, -225.1312561, 224.8547974
2: -82.1236191, 151.2655640, -82.3216171, 151.6371460, -233.7607269, 233.5871735
3: -47.1132736, 181.5767212, -47.2267761, 182.0128632, -229.1261139, 228.8034821
4: -75.0532150, 133.2210693, -75.2349548, 133.5453339, -208.5985260, 208.4560242

Time for backsubstitution: 1.43 seconds

### NS candidates at layer 1
type: B, layer: 1, pos: 41
type: A, layer: 1, pos: 41
type: A, layer: 1, pos: 13
type: B, layer: 1, pos: 13
type: B, layer: 1, pos: 19
type: A, layer: 1, pos: 19
type: A, layer: 1, pos: 25
type: B, layer: 1, pos: 25
type: B, layer: 1, pos: 16
type: A, layer: 1, pos: 16
type: B, layer: 1, pos: 26
type: A, layer: 1, pos: 26
type: B, layer: 1, pos: 46
type: A, layer: 1, pos: 46
type: B, layer: 1, pos: 4
type: A, layer: 1, pos: 4
type: A, layer: 1, pos: 29
type: B, layer: 1, pos: 29
type: B, layer: 1, pos: 21
type: A, layer: 1, pos: 38
type: A, layer: 1, pos: 21
type: B, layer: 1, pos: 38
type: B, layer: 1, pos: 43
type: B, layer: 1, pos: 22
type: A, layer: 1, pos: 22
type: A, layer: 1, pos: 12
type: B, layer: 1, pos: 12
type: A, layer: 1, pos: 37
type: B, layer: 1, pos: 37
type: B, layer: 1, pos: 14
type: B, layer: 1, pos: 8
type: A, layer: 1, pos: 8
type: A, layer: 1, pos: 14
type: B, layer: 1, pos: 6
type: A, layer: 1, pos: 6
type: B, layer: 1, pos: 31
type: A, layer: 1, pos: 31
type: B, layer: 1, pos: 39
type: A, layer: 1, pos: 39
type: B, layer: 1, pos: 20
type: A, layer: 1, pos: 20
type: A, layer: 1, pos: 23
type: B, layer: 1, pos: 23
type: A, layer: 1, pos: 42
type: B, layer: 1, pos: 42
type: B, layer: 1, pos: 11
type: A, layer: 1, pos: 11
type: B, layer: 1, pos: 15
type: A, layer: 1, pos: 15

Time for candidate selection: 0.12 seconds

### Candidate
type: B, layer: 1, pos: 41

## Relational analysis of NS_A2_A1_B2_B1

### Relational analysis result of NS_A2_A1_B2_B1
Status: Status.VERIFIED
Output dim: 0, lower bound: -81.8594920, upper bound: 81.8597996
time: 1.01 seconds

## Relational analysis of NS_A2_A1_B2_B2
Optimization infeasible because this subproblem isn't reachable.

### Candidate
type: A, layer: 1, pos: 41

### Candidate
type: A, layer: 1, pos: 13

## Relational analysis of NS_A2_A1_B2_A1

### Relational analysis result of NS_A2_A1_B2_A1
Status: Status.VERIFIED
Output dim: 0, lower bound: -81.8605823, upper bound: 81.8608939
time: 0.89 seconds

## Relational analysis of NS_A2_A1_B2_A2

### Relational analysis result of NS_A2_A1_B2_A2
Status: Status.UNKNOWN
Output dim: 0, lower bound: -81.8608126, upper bound: 81.8613877
time: 0.86 seconds

## Summary of splitting at layer (split count: 3)
- Time for NS candidates: 4.65 seconds
NS_A2_A1_B1_A1, status: Status.VERIFIED, split count: 4, time: 4.65
Output dim: 0, lower bound: -81.8605164, upper bound: 81.8608939
NS_A2_A1_B1_A2, status: Status.UNKNOWN, split count: 4, time: 4.65
Output dim: 0, lower bound: -81.8608126, upper bound: 81.8613877
NS_A2_A1_B2_A1, status: Status.VERIFIED, split count: 4, time: 4.65
Output dim: 0, lower bound: -81.8605823, upper bound: 81.8608939
NS_A2_A1_B2_A2, status: Status.UNKNOWN, split count: 4, time: 4.65
Output dim: 0, lower bound: -81.8608126, upper bound: 81.8613877

## BFS NS instance: NS_A2_A1_B1_A2

### Backsubstitution after applying NS history:
0: -20.4099464, 74.5275269, -19.8369217, 72.4063263, -92.8162689, 94.3644333
1: -54.1815529, 168.1692810, -52.7576790, 163.3856506, -217.5671997, 220.9269409
2: -81.2650299, 149.6428528, -79.4177856, 145.0854797, -226.3505096, 229.0606232
3: -46.6135674, 179.6675720, -45.3945847, 174.7305145, -221.3440857, 225.0621643
4: -74.2615967, 131.7928925, -72.5078278, 127.8521042, -202.1137085, 204.3007202

Time for backsubstitution: 1.51 seconds

### NS candidates at layer 1
type: A, layer: 1, pos: 41
type: B, layer: 1, pos: 41
type: A, layer: 1, pos: 19
type: B, layer: 1, pos: 19
type: B, layer: 1, pos: 6
type: A, layer: 1, pos: 6
type: B, layer: 1, pos: 25
type: B, layer: 1, pos: 21
type: A, layer: 1, pos: 25
type: B, layer: 1, pos: 26
type: B, layer: 1, pos: 4
type: A, layer: 1, pos: 26
type: A, layer: 1, pos: 21
type: A, layer: 1, pos: 46
type: B, layer: 1, pos: 46
type: A, layer: 1, pos: 38
type: A, layer: 1, pos: 4
type: B, layer: 1, pos: 13
type: A, layer: 1, pos: 29
type: B, layer: 1, pos: 38
type: B, layer: 1, pos: 14
type: B, layer: 1, pos: 29
type: A, layer: 1, pos: 14
type: A, layer: 1, pos: 12
type: B, layer: 1, pos: 12
type: B, layer: 1, pos: 43
type: B, layer: 1, pos: 11
type: A, layer: 1, pos: 31
type: B, layer: 1, pos: 31
type: A, layer: 1, pos: 11
type: B, layer: 1, pos: 16
type: A, layer: 1, pos: 39
type: B, layer: 1, pos: 39
type: B, layer: 1, pos: 15
type: B, layer: 1, pos: 20
type: A, layer: 1, pos: 15
type: A, layer: 1, pos: 16
type: A, layer: 1, pos: 37
type: B, layer: 1, pos: 37
type: A, layer: 1, pos: 22
type: B, layer: 1, pos: 23
type: B, layer: 1, pos: 22
type: A, layer: 1, pos: 8
type: B, layer: 1, pos: 42
type: A, layer: 1, pos: 23
type: B, layer: 1, pos: 8
type: A, layer: 1, pos: 20
type: A, layer: 1, pos: 42

Time for candidate selection: 0.12 seconds

### Candidate
type: A, layer: 1, pos: 41

### Candidate
type: B, layer: 1, pos: 41

### Candidate
type: A, layer: 1, pos: 19

## Relational analysis of NS_A2_A1_B1_A2_A1
Optimization infeasible because this subproblem isn't reachable.

### Candidate
type: B, layer: 1, pos: 19

### Candidate
type: B, layer: 1, pos: 6

## Relational analysis of NS_A2_A1_B1_A2_B1
Optimization infeasible because this subproblem isn't reachable.

### Candidate
type: A, layer: 1, pos: 6

### Candidate
type: B, layer: 1, pos: 25

## Relational analysis of NS_A2_A1_B1_A2_B1

### Relational analysis result of NS_A2_A1_B1_A2_B1
Status: Status.UNKNOWN
Output dim: 0, lower bound: -81.8608927, upper bound: 81.8612455
time: 0.82 seconds

## Relational analysis of NS_A2_A1_B1_A2_B2
Optimization infeasible because this subproblem isn't reachable.

### Candidate
type: B, layer: 1, pos: 21

## Relational analysis of NS_A2_A1_B1_A2_B1
Optimization infeasible because this subproblem isn't reachable.

### Candidate
type: A, layer: 1, pos: 25

### Candidate
type: B, layer: 1, pos: 26

## Relational analysis of NS_A2_A1_B1_A2_B1

### Relational analysis result of NS_A2_A1_B1_A2_B1
Status: Status.UNKNOWN
Output dim: 0, lower bound: -81.8608351, upper bound: 81.8614238
time: 0.93 seconds

## Relational analysis of NS_A2_A1_B1_A2_B2

### Relational analysis result of NS_A2_A1_B1_A2_B2
Status: Status.UNKNOWN
Output dim: 0, lower bound: -81.8610569, upper bound: 81.8615675
time: 0.76 seconds

## BFS NS instance: NS_A2_A1_B2_A2

### Backsubstitution after applying NS history:
0: -20.4099464, 74.5275269, -20.6764908, 75.5103683, -95.9203110, 95.2040176
1: -54.1815529, 168.1692810, -54.8931732, 170.3698120, -224.5513611, 223.0624542
2: -81.2650299, 149.6428528, -82.3216171, 151.6371460, -232.9021606, 231.9644775
3: -46.6135674, 179.6675720, -47.2267761, 182.0128632, -228.6264191, 226.8943329
4: -74.2615967, 131.7928925, -75.2349548, 133.5453339, -207.8069153, 207.0278473

Time for backsubstitution: 1.49 seconds

### NS candidates at layer 1
type: B, layer: 1, pos: 41
type: A, layer: 1, pos: 41
type: A, layer: 1, pos: 19
type: B, layer: 1, pos: 19
type: B, layer: 1, pos: 6
type: A, layer: 1, pos: 6
type: A, layer: 1, pos: 25
type: B, layer: 1, pos: 25
type: B, layer: 1, pos: 26
type: B, layer: 1, pos: 21
type: A, layer: 1, pos: 26
type: A, layer: 1, pos: 4
type: A, layer: 1, pos: 46
type: A, layer: 1, pos: 21
type: B, layer: 1, pos: 46
type: B, layer: 1, pos: 4
type: A, layer: 1, pos: 38
type: B, layer: 1, pos: 38
type: B, layer: 1, pos: 13
type: A, layer: 1, pos: 29
type: B, layer: 1, pos: 14
type: B, layer: 1, pos: 43
type: B, layer: 1, pos: 29
type: A, layer: 1, pos: 12
type: A, layer: 1, pos: 14
type: B, layer: 1, pos: 12
type: B, layer: 1, pos: 11
type: B, layer: 1, pos: 31
type: A, layer: 1, pos: 31
type: A, layer: 1, pos: 11
type: B, layer: 1, pos: 39
type: A, layer: 1, pos: 39
type: B, layer: 1, pos: 16
type: A, layer: 1, pos: 15
type: B, layer: 1, pos: 15
type: A, layer: 1, pos: 16
type: B, layer: 1, pos: 37
type: B, layer: 1, pos: 20
type: A, layer: 1, pos: 37
type: B, layer: 1, pos: 23
type: B, layer: 1, pos: 22
type: A, layer: 1, pos: 22
type: A, layer: 1, pos: 20
type: A, layer: 1, pos: 23
type: A, layer: 1, pos: 42
type: B, layer: 1, pos: 8
type: A, layer: 1, pos: 8
type: B, layer: 1, pos: 42

Time for candidate selection: 0.12 seconds

### Candidate
type: B, layer: 1, pos: 41

### Candidate
type: A, layer: 1, pos: 41

### Candidate
type: A, layer: 1, pos: 19

## Relational analysis of NS_A2_A1_B2_A2_A1
Optimization infeasible because this subproblem isn't reachable.

### Candidate
type: B, layer: 1, pos: 19

### Candidate
type: B, layer: 1, pos: 6

## Relational analysis of NS_A2_A1_B2_A2_B1
Optimization infeasible because this subproblem isn't reachable.

### Candidate
type: A, layer: 1, pos: 6

### Candidate
type: A, layer: 1, pos: 25

## Relational analysis of NS_A2_A1_B2_A2_A1
Optimization infeasible because this subproblem isn't reachable.

### Candidate
type: B, layer: 1, pos: 25

### Candidate
type: B, layer: 1, pos: 26

## Relational analysis of NS_A2_A1_B2_A2_B1
Optimization infeasible because this subproblem isn't reachable.

### Candidate
type: B, layer: 1, pos: 21

## Relational analysis of NS_A2_A1_B2_A2_B1
Optimization infeasible because this subproblem isn't reachable.

### Candidate
type: A, layer: 1, pos: 26

### Candidate
type: A, layer: 1, pos: 4

## Relational analysis of NS_A2_A1_B2_A2_A1

### Relational analysis result of NS_A2_A1_B2_A2_A1
Status: Status.VERIFIED
Output dim: 0, lower bound: -81.8596946, upper bound: 81.8596057
time: 1.07 seconds

## Relational analysis of NS_A2_A1_B2_A2_A2

### Relational analysis result of NS_A2_A1_B2_A2_A2
Status: Status.UNKNOWN
Output dim: 0, lower bound: -81.8608126, upper bound: 81.8613877
time: 0.99 seconds

## Summary of splitting at layer (split count: 4)
- Time for NS candidates: 5.54 seconds
NS_A2_A1_B1_A2_B1, status: Status.UNKNOWN, split count: 5, time: 5.54
Output dim: 0, lower bound: -81.8608351, upper bound: 81.8614238
NS_A2_A1_B1_A2_B2, status: Status.UNKNOWN, split count: 5, time: 5.54
Output dim: 0, lower bound: -81.8610569, upper bound: 81.8615675
NS_A2_A1_B2_A2_A1, status: Status.VERIFIED, split count: 5, time: 5.54
Output dim: 0, lower bound: -81.8596946, upper bound: 81.8596057
NS_A2_A1_B2_A2_A2, status: Status.UNKNOWN, split count: 5, time: 5.54
Output dim: 0, lower bound: -81.8608126, upper bound: 81.8613877

## BFS NS instance: NS_A2_A1_B1_A2_B1

### Backsubstitution after applying NS history:
0: -20.0273113, 73.1337204, -18.9973679, 69.3533783, -89.3806763, 92.1310806
1: -53.1638145, 165.0083466, -50.5398903, 156.4445648, -209.6083832, 215.5482330
2: -79.7408829, 146.8376160, -76.1148834, 138.9211273, -218.6619720, 222.9524994
3: -45.7397995, 176.2969208, -43.4817924, 167.3243256, -213.0640869, 219.7786865
4: -72.8656387, 129.3282471, -69.4807663, 122.4303818, -195.2960205, 198.8090210

Time for backsubstitution: 1.51 seconds

### NS candidates at layer 1
type: A, layer: 1, pos: 26
type: B, layer: 1, pos: 41
type: A, layer: 1, pos: 41
type: A, layer: 1, pos: 19
type: B, layer: 1, pos: 19
type: B, layer: 1, pos: 25
type: A, layer: 1, pos: 6
type: B, layer: 1, pos: 6
type: A, layer: 1, pos: 21
type: B, layer: 1, pos: 4
type: B, layer: 1, pos: 46
type: A, layer: 1, pos: 25
type: B, layer: 1, pos: 21
type: A, layer: 1, pos: 38
type: A, layer: 1, pos: 46
type: A, layer: 1, pos: 4
type: B, layer: 1, pos: 13
type: A, layer: 1, pos: 14
type: B, layer: 1, pos: 38
type: A, layer: 1, pos: 29
type: B, layer: 1, pos: 11
type: B, layer: 1, pos: 12
type: B, layer: 1, pos: 29
type: B, layer: 1, pos: 43
type: A, layer: 1, pos: 12
type: B, layer: 1, pos: 31
type: B, layer: 1, pos: 14
type: A, layer: 1, pos: 31
type: B, layer: 1, pos: 16
type: B, layer: 1, pos: 39
type: B, layer: 1, pos: 15
type: A, layer: 1, pos: 39
type: A, layer: 1, pos: 11
type: A, layer: 1, pos: 37
type: A, layer: 1, pos: 15
type: B, layer: 1, pos: 20
type: A, layer: 1, pos: 22
type: A, layer: 1, pos: 16
type: A, layer: 1, pos: 20
type: B, layer: 1, pos: 23
type: A, layer: 1, pos: 23
type: B, layer: 1, pos: 8
type: B, layer: 1, pos: 37
type: B, layer: 1, pos: 22
type: A, layer: 1, pos: 42
type: A, layer: 1, pos: 8
type: B, layer: 1, pos: 42

Time for candidate selection: 0.14 seconds

### Candidate
type: A, layer: 1, pos: 26

## Relational analysis of NS_A2_A1_B1_A2_B1_A1

### Relational analysis result of NS_A2_A1_B1_A2_B1_A1
Status: Status.UNKNOWN
Output dim: 0, lower bound: -81.8608383, upper bound: 81.8614238
time: 0.93 seconds

## Relational analysis of NS_A2_A1_B1_A2_B1_A2

### Relational analysis result of NS_A2_A1_B1_A2_B1_A2
Status: Status.UNKNOWN
Output dim: 0, lower bound: -81.8608383, upper bound: 81.8614238
time: 0.76 seconds

## BFS NS instance: NS_A2_A1_B1_A2_B2

### Backsubstitution after applying NS history:
0: -20.2480984, 73.9418488, -22.3477135, 81.5668259, -101.8149261, 96.2895508
1: -53.7638664, 166.8047180, -59.0682335, 185.8462830, -239.6101532, 225.8729553
2: -80.6519775, 148.4456329, -88.6273117, 163.8466644, -244.4986420, 237.0729370
3: -46.2542229, 178.2196350, -50.8008537, 198.3957214, -244.6499176, 229.0204926
4: -73.6978226, 130.7500000, -80.9325562, 144.4432983, -218.1411133, 211.6825104

Time for backsubstitution: 1.47 seconds

### NS candidates at layer 1
type: A, layer: 1, pos: 26
type: A, layer: 1, pos: 19
type: B, layer: 1, pos: 41
type: A, layer: 1, pos: 41
type: A, layer: 1, pos: 21
type: A, layer: 1, pos: 6
type: B, layer: 1, pos: 6
type: B, layer: 1, pos: 46
type: B, layer: 1, pos: 25
type: B, layer: 1, pos: 11
type: A, layer: 1, pos: 14
type: B, layer: 1, pos: 4
type: A, layer: 1, pos: 25
type: B, layer: 1, pos: 38
type: B, layer: 1, pos: 16
type: A, layer: 1, pos: 29
type: B, layer: 1, pos: 13
type: B, layer: 1, pos: 43
type: A, layer: 1, pos: 38
type: B, layer: 1, pos: 12
type: A, layer: 1, pos: 37
type: B, layer: 1, pos: 15
type: A, layer: 1, pos: 12
type: B, layer: 1, pos: 19
type: A, layer: 1, pos: 4
type: A, layer: 1, pos: 39
type: B, layer: 1, pos: 31
type: A, layer: 1, pos: 46
type: A, layer: 1, pos: 31
type: B, layer: 1, pos: 21
type: B, layer: 1, pos: 29
type: A, layer: 1, pos: 20
type: B, layer: 1, pos: 39
type: B, layer: 1, pos: 8
type: A, layer: 1, pos: 23
type: A, layer: 1, pos: 22
type: B, layer: 1, pos: 14
type: B, layer: 1, pos: 20
type: B, layer: 1, pos: 42
type: B, layer: 1, pos: 23
type: A, layer: 1, pos: 8
type: A, layer: 1, pos: 15
type: B, layer: 1, pos: 37
type: A, layer: 1, pos: 42
type: B, layer: 1, pos: 22
type: A, layer: 1, pos: 16
type: A, layer: 1, pos: 11

Time for candidate selection: 0.12 seconds

### Candidate
type: A, layer: 1, pos: 26

## Relational analysis of NS_A2_A1_B1_A2_B2_A1

### Relational analysis result of NS_A2_A1_B1_A2_B2_A1
Status: Status.UNKNOWN
Output dim: 0, lower bound: -81.8610504, upper bound: 81.8615675
time: 0.90 seconds

## Relational analysis of NS_A2_A1_B1_A2_B2_A2

### Relational analysis result of NS_A2_A1_B1_A2_B2_A2
Status: Status.UNKNOWN
Output dim: 0, lower bound: -81.8610504, upper bound: 81.8615675
time: 0.94 seconds

## BFS NS instance: NS_A2_A1_B2_A2_A2

### Backsubstitution after applying NS history:
0: -20.1414471, 73.5443802, -20.6713696, 75.4915924, -95.6330414, 94.2157516
1: -53.4771767, 165.9335632, -54.8797112, 170.3271790, -223.8043365, 220.8132782
2: -80.1817780, 147.6730042, -82.3009415, 151.5994720, -231.7812500, 229.9739380
3: -46.0048676, 177.2870331, -47.2151451, 181.9674988, -227.9723663, 224.5021667
4: -73.2760468, 130.0593109, -75.2161407, 133.5122681, -206.7882690, 205.2754517

Time for backsubstitution: 1.46 seconds

### NS candidates at layer 1
type: B, layer: 1, pos: 41
type: A, layer: 1, pos: 41
type: B, layer: 1, pos: 4
type: B, layer: 1, pos: 19
type: A, layer: 1, pos: 19
type: A, layer: 1, pos: 6
type: B, layer: 1, pos: 6
type: B, layer: 1, pos: 25
type: B, layer: 1, pos: 26
type: A, layer: 1, pos: 25
type: B, layer: 1, pos: 21
type: A, layer: 1, pos: 26
type: B, layer: 1, pos: 46
type: A, layer: 1, pos: 21
type: A, layer: 1, pos: 46
type: B, layer: 1, pos: 38
type: A, layer: 1, pos: 38
type: B, layer: 1, pos: 13
type: A, layer: 1, pos: 29
type: B, layer: 1, pos: 14
type: B, layer: 1, pos: 43
type: A, layer: 1, pos: 12
type: B, layer: 1, pos: 29
type: B, layer: 1, pos: 11
type: A, layer: 1, pos: 14
type: B, layer: 1, pos: 12
type: A, layer: 1, pos: 31
type: B, layer: 1, pos: 31
type: B, layer: 1, pos: 39
type: A, layer: 1, pos: 11
type: A, layer: 1, pos: 39
type: A, layer: 1, pos: 16
type: A, layer: 1, pos: 15
type: B, layer: 1, pos: 15
type: B, layer: 1, pos: 16
type: B, layer: 1, pos: 20
type: A, layer: 1, pos: 37
type: B, layer: 1, pos: 37
type: B, layer: 1, pos: 22
type: A, layer: 1, pos: 23
type: B, layer: 1, pos: 23
type: A, layer: 1, pos: 20
type: A, layer: 1, pos: 22
type: A, layer: 1, pos: 42
type: B, layer: 1, pos: 8
type: A, layer: 1, pos: 8
type: B, layer: 1, pos: 42

Time for candidate selection: 0.12 seconds

### Candidate
type: B, layer: 1, pos: 41

### Candidate
type: A, layer: 1, pos: 41

### Candidate
type: B, layer: 1, pos: 4

## Relational analysis of NS_A2_A1_B2_A2_A2_B1
Optimization infeasible because this subproblem isn't reachable.

### Candidate
type: B, layer: 1, pos: 19

### Candidate
type: A, layer: 1, pos: 19

### Candidate
type: A, layer: 1, pos: 6

### Candidate
type: B, layer: 1, pos: 6

### Candidate
type: B, layer: 1, pos: 25

### Candidate
type: B, layer: 1, pos: 26

### Candidate
type: A, layer: 1, pos: 25

### Candidate
type: B, layer: 1, pos: 21

### Candidate
type: A, layer: 1, pos: 26

### Candidate
type: B, layer: 1, pos: 46

## Relational analysis of NS_A2_A1_B2_A2_A2_B1
Optimization infeasible because this subproblem isn't reachable.

### Candidate
type: A, layer: 1, pos: 21

### Candidate
type: A, layer: 1, pos: 46

### Candidate
type: B, layer: 1, pos: 38

## Relational analysis of NS_A2_A1_B2_A2_A2_B1
Optimization infeasible because this subproblem isn't reachable.

### Candidate
type: A, layer: 1, pos: 38

### Candidate
type: B, layer: 1, pos: 13

## Relational analysis of NS_A2_A1_B2_A2_A2_B1
Optimization infeasible because this subproblem isn't reachable.

### Candidate
type: A, layer: 1, pos: 29

## Relational analysis of NS_A2_A1_B2_A2_A2_A1
Optimization infeasible because this subproblem isn't reachable.

### Candidate
type: B, layer: 1, pos: 14

## Relational analysis of NS_A2_A1_B2_A2_A2_B1
Optimization infeasible because this subproblem isn't reachable.

### Candidate
type: B, layer: 1, pos: 43

## Relational analysis of NS_A2_A1_B2_A2_A2_B1

### Relational analysis result of NS_A2_A1_B2_A2_A2_B1
Status: Status.UNKNOWN
Output dim: 0, lower bound: -81.8605823, upper bound: 81.8613877
time: 1.39 seconds

## Relational analysis of NS_A2_A1_B2_A2_A2_B2

### Relational analysis result of NS_A2_A1_B2_A2_A2_B2
Status: Status.UNKNOWN
Output dim: 0, lower bound: -81.8605823, upper bound: 81.8613877
time: 0.83 seconds

## Summary of splitting at layer (split count: 5)
- Time for NS candidates: 5.90 seconds
NS_A2_A1_B1_A2_B1_A1, status: Status.UNKNOWN, split count: 6, time: 5.90
Output dim: 0, lower bound: -81.8608383, upper bound: 81.8614238
NS_A2_A1_B1_A2_B1_A2, status: Status.UNKNOWN, split count: 6, time: 5.90
Output dim: 0, lower bound: -81.8608383, upper bound: 81.8614238
NS_A2_A1_B1_A2_B2_A1, status: Status.UNKNOWN, split count: 6, time: 5.90
Output dim: 0, lower bound: -81.8610504, upper bound: 81.8615675
NS_A2_A1_B1_A2_B2_A2, status: Status.UNKNOWN, split count: 6, time: 5.90
Output dim: 0, lower bound: -81.8610504, upper bound: 81.8615675
NS_A2_A1_B2_A2_A2_B1, status: Status.UNKNOWN, split count: 6, time: 5.90
Output dim: 0, lower bound: -81.8605823, upper bound: 81.8613877
NS_A2_A1_B2_A2_A2_B2, status: Status.UNKNOWN, split count: 6, time: 5.90
Output dim: 0, lower bound: -81.8605823, upper bound: 81.8613877

## BFS NS instance: NS_A2_A1_B1_A2_B1_A1

### Backsubstitution after applying NS history:
0: -19.5514183, 71.4037170, -18.9973679, 69.3533783, -88.9047852, 90.4010696
1: -51.9093094, 161.0701752, -50.5398903, 156.4445648, -208.3538666, 211.6100616
2: -77.8657913, 143.3444519, -76.1148834, 138.9211273, -216.7869263, 219.4593201
3: -44.6583481, 172.0934601, -43.4817924, 167.3243256, -211.9826508, 215.5752563
4: -71.1484833, 126.2585678, -69.4807663, 122.4303818, -193.5788574, 195.7393341

Time for backsubstitution: 1.04 seconds

### NS candidates at layer 1
type: A, layer: 1, pos: 41
type: B, layer: 1, pos: 41
type: A, layer: 1, pos: 19
type: B, layer: 1, pos: 19
type: B, layer: 1, pos: 6
type: A, layer: 1, pos: 6
type: B, layer: 1, pos: 25
type: B, layer: 1, pos: 21
type: A, layer: 1, pos: 25
type: B, layer: 1, pos: 4
type: A, layer: 1, pos: 46
type: A, layer: 1, pos: 21
type: B, layer: 1, pos: 46
type: A, layer: 1, pos: 38
type: A, layer: 1, pos: 4
type: B, layer: 1, pos: 13
type: A, layer: 1, pos: 29
type: B, layer: 1, pos: 38
type: B, layer: 1, pos: 14
type: B, layer: 1, pos: 29
type: A, layer: 1, pos: 14
type: A, layer: 1, pos: 12
type: B, layer: 1, pos: 12
type: B, layer: 1, pos: 43
type: B, layer: 1, pos: 11
type: A, layer: 1, pos: 31
type: B, layer: 1, pos: 31
type: A, layer: 1, pos: 11
type: B, layer: 1, pos: 16
type: A, layer: 1, pos: 39
type: B, layer: 1, pos: 39
type: B, layer: 1, pos: 15
type: B, layer: 1, pos: 20
type: A, layer: 1, pos: 15
type: A, layer: 1, pos: 16
type: A, layer: 1, pos: 37
type: B, layer: 1, pos: 37
type: B, layer: 1, pos: 23
type: A, layer: 1, pos: 22
type: B, layer: 1, pos: 22
type: A, layer: 1, pos: 8
type: B, layer: 1, pos: 42
type: A, layer: 1, pos: 23
type: B, layer: 1, pos: 8
type: A, layer: 1, pos: 20
type: A, layer: 1, pos: 42

Time for candidate selection: 0.08 seconds

### Candidate
type: A, layer: 1, pos: 41

### Candidate
type: B, layer: 1, pos: 41

### Candidate
type: A, layer: 1, pos: 19

### Candidate
type: B, layer: 1, pos: 19

### Candidate
type: B, layer: 1, pos: 6

### Candidate
type: A, layer: 1, pos: 6

### Candidate
type: B, layer: 1, pos: 25

### Candidate
type: B, layer: 1, pos: 21

### Candidate
type: A, layer: 1, pos: 25

### Candidate
type: B, layer: 1, pos: 4

## Relational analysis of NS_A2_A1_B1_A2_B1_A1_B1
Optimization infeasible because this subproblem isn't reachable.

### Candidate
type: A, layer: 1, pos: 46

## Relational analysis of NS_A2_A1_B1_A2_B1_A1_A1
Optimization infeasible because this subproblem isn't reachable.

### Candidate
type: A, layer: 1, pos: 21

### Candidate
type: B, layer: 1, pos: 46

### Candidate
type: A, layer: 1, pos: 38

## Relational analysis of NS_A2_A1_B1_A2_B1_A1_A1
Optimization infeasible because this subproblem isn't reachable.

### Candidate
type: A, layer: 1, pos: 4

### Candidate
type: B, layer: 1, pos: 13

## Relational analysis of NS_A2_A1_B1_A2_B1_A1_B1
Optimization infeasible because this subproblem isn't reachable.

### Candidate
type: A, layer: 1, pos: 29

## Relational analysis of NS_A2_A1_B1_A2_B1_A1_A1
Optimization infeasible because this subproblem isn't reachable.

### Candidate
type: B, layer: 1, pos: 38

### Candidate
type: B, layer: 1, pos: 14

## Relational analysis of NS_A2_A1_B1_A2_B1_A1_B1
Optimization infeasible because this subproblem isn't reachable.

### Candidate
type: B, layer: 1, pos: 29

### Candidate
type: A, layer: 1, pos: 14

### Candidate
type: A, layer: 1, pos: 12

## Relational analysis of NS_A2_A1_B1_A2_B1_A1_A1
Optimization infeasible because this subproblem isn't reachable.

### Candidate
type: B, layer: 1, pos: 12

### Candidate
type: B, layer: 1, pos: 43

## Relational analysis of NS_A2_A1_B1_A2_B1_A1_B1

### Relational analysis result of NS_A2_A1_B1_A2_B1_A1_B1
Status: Status.UNKNOWN
Output dim: 0, lower bound: -81.8608383, upper bound: 81.8614238
time: 0.81 seconds

## Relational analysis of NS_A2_A1_B1_A2_B1_A1_B2

### Relational analysis result of NS_A2_A1_B1_A2_B1_A1_B2
Status: Status.UNKNOWN
Output dim: 0, lower bound: -81.8608383, upper bound: 81.8614238
time: 0.86 seconds

## BFS NS instance: NS_A2_A1_B1_A2_B1_A2

### Backsubstitution after applying NS history:
0: -23.0393181, 84.1384583, -18.9973679, 69.3533783, -92.3926926, 103.1358185
1: -60.7940979, 191.6210938, -50.5398903, 156.4445648, -217.2386627, 242.1609802
2: -90.8825989, 169.5263519, -76.1148834, 138.9211273, -229.8037109, 245.6412048
3: -52.2806168, 204.3468628, -43.4817924, 167.3243256, -219.6049042, 247.8286133
4: -83.0892868, 149.3332672, -69.4807663, 122.4303818, -205.5196686, 218.8140259

Time for backsubstitution: 0.96 seconds

### NS candidates at layer 1
type: A, layer: 1, pos: 41
type: B, layer: 1, pos: 41
type: B, layer: 1, pos: 19
type: B, layer: 1, pos: 21
type: A, layer: 1, pos: 6
type: B, layer: 1, pos: 6
type: A, layer: 1, pos: 19
type: B, layer: 1, pos: 25
type: A, layer: 1, pos: 46
type: A, layer: 1, pos: 25
type: A, layer: 1, pos: 38
type: B, layer: 1, pos: 14
type: A, layer: 1, pos: 4
type: A, layer: 1, pos: 11
type: B, layer: 1, pos: 4
type: A, layer: 1, pos: 21
type: B, layer: 1, pos: 13
type: B, layer: 1, pos: 12
type: B, layer: 1, pos: 38
type: A, layer: 1, pos: 12
type: B, layer: 1, pos: 39
type: B, layer: 1, pos: 29
type: B, layer: 1, pos: 46
type: A, layer: 1, pos: 29
type: B, layer: 1, pos: 20
type: A, layer: 1, pos: 16
type: A, layer: 1, pos: 31
type: B, layer: 1, pos: 31
type: A, layer: 1, pos: 15
type: A, layer: 1, pos: 14
type: B, layer: 1, pos: 37
type: B, layer: 1, pos: 43
type: B, layer: 1, pos: 15
type: B, layer: 1, pos: 23
type: B, layer: 1, pos: 16
type: B, layer: 1, pos: 8
type: A, layer: 1, pos: 39
type: B, layer: 1, pos: 22
type: A, layer: 1, pos: 8
type: A, layer: 1, pos: 42
type: A, layer: 1, pos: 23
type: A, layer: 1, pos: 20
type: B, layer: 1, pos: 11
type: A, layer: 1, pos: 22
type: B, layer: 1, pos: 42
type: A, layer: 1, pos: 37

Time for candidate selection: 0.07 seconds

### Candidate
type: A, layer: 1, pos: 41

### Candidate
type: B, layer: 1, pos: 41

### Candidate
type: B, layer: 1, pos: 19

### Candidate
type: B, layer: 1, pos: 21

### Candidate
type: A, layer: 1, pos: 6

### Candidate
type: B, layer: 1, pos: 6

### Candidate
type: A, layer: 1, pos: 19

### Candidate
type: B, layer: 1, pos: 25

### Candidate
type: A, layer: 1, pos: 46

## Relational analysis of NS_A2_A1_B1_A2_B1_A2_A1
Optimization infeasible because this subproblem isn't reachable.

### Candidate
type: A, layer: 1, pos: 25

### Candidate
type: A, layer: 1, pos: 38

## Relational analysis of NS_A2_A1_B1_A2_B1_A2_A1
Optimization infeasible because this subproblem isn't reachable.

### Candidate
type: B, layer: 1, pos: 14

## Relational analysis of NS_A2_A1_B1_A2_B1_A2_B1
Optimization infeasible because this subproblem isn't reachable.

### Candidate
type: A, layer: 1, pos: 4

## Relational analysis of NS_A2_A1_B1_A2_B1_A2_A1
Optimization infeasible because this subproblem isn't reachable.

### Candidate
type: A, layer: 1, pos: 11

## Relational analysis of NS_A2_A1_B1_A2_B1_A2_A1
Optimization infeasible because this subproblem isn't reachable.

### Candidate
type: B, layer: 1, pos: 4

### Candidate
type: A, layer: 1, pos: 21

### Candidate
type: B, layer: 1, pos: 13

## Relational analysis of NS_A2_A1_B1_A2_B1_A2_B1
Optimization infeasible because this subproblem isn't reachable.

### Candidate
type: B, layer: 1, pos: 12

## Relational analysis of NS_A2_A1_B1_A2_B1_A2_B1
Optimization infeasible because this subproblem isn't reachable.

### Candidate
type: B, layer: 1, pos: 38

### Candidate
type: A, layer: 1, pos: 12

### Candidate
type: B, layer: 1, pos: 39

## Relational analysis of NS_A2_A1_B1_A2_B1_A2_B1
Optimization infeasible because this subproblem isn't reachable.

### Candidate
type: B, layer: 1, pos: 29

## Relational analysis of NS_A2_A1_B1_A2_B1_A2_B1
Optimization infeasible because this subproblem isn't reachable.

### Candidate
type: B, layer: 1, pos: 46

### Candidate
type: A, layer: 1, pos: 29

### Candidate
type: B, layer: 1, pos: 20

## Relational analysis of NS_A2_A1_B1_A2_B1_A2_B1
Optimization infeasible because this subproblem isn't reachable.

### Candidate
type: A, layer: 1, pos: 16

## Relational analysis of NS_A2_A1_B1_A2_B1_A2_A1

### Relational analysis result of NS_A2_A1_B1_A2_B1_A2_A1
Status: Status.UNKNOWN
Output dim: 0, lower bound: -81.8608383, upper bound: 81.8614238
time: 1.08 seconds

## Relational analysis of NS_A2_A1_B1_A2_B1_A2_A2

### Relational analysis result of NS_A2_A1_B1_A2_B1_A2_A2
Status: Status.UNKNOWN
Output dim: 0, lower bound: -81.8607763, upper bound: 81.8614029
time: 1.04 seconds

## BFS NS instance: NS_A2_A1_B1_A2_B2_A1

### Backsubstitution after applying NS history:
0: -19.5407543, 71.3658142, -22.3477135, 81.5668259, -101.1075821, 93.7135162
1: -51.8812523, 160.9836731, -59.0682335, 185.8462830, -237.7275391, 220.0518799
2: -77.8219833, 143.2691956, -88.6273117, 163.8466644, -241.6686249, 231.8965149
3: -44.6342278, 172.0003357, -50.8008537, 198.3957214, -243.0299377, 222.8011780
4: -71.1089172, 126.1929550, -80.9325562, 144.4432983, -215.5522156, 207.1254883

Time for backsubstitution: 0.93 seconds

### NS candidates at layer 1
type: A, layer: 1, pos: 41
type: B, layer: 1, pos: 41
type: A, layer: 1, pos: 19
type: B, layer: 1, pos: 6
type: A, layer: 1, pos: 6
type: A, layer: 1, pos: 21
type: B, layer: 1, pos: 25
type: A, layer: 1, pos: 25
type: B, layer: 1, pos: 19
type: B, layer: 1, pos: 46
type: B, layer: 1, pos: 4
type: B, layer: 1, pos: 11
type: A, layer: 1, pos: 14
type: B, layer: 1, pos: 21
type: A, layer: 1, pos: 38
type: B, layer: 1, pos: 38
type: A, layer: 1, pos: 29
type: B, layer: 1, pos: 13
type: A, layer: 1, pos: 46
type: A, layer: 1, pos: 12
type: B, layer: 1, pos: 16
type: B, layer: 1, pos: 12
type: A, layer: 1, pos: 39
type: A, layer: 1, pos: 4
type: B, layer: 1, pos: 43
type: B, layer: 1, pos: 15
type: B, layer: 1, pos: 14
type: B, layer: 1, pos: 31
type: A, layer: 1, pos: 31
type: B, layer: 1, pos: 29
type: A, layer: 1, pos: 37
type: A, layer: 1, pos: 20
type: B, layer: 1, pos: 20
type: A, layer: 1, pos: 8
type: A, layer: 1, pos: 15
type: B, layer: 1, pos: 23
type: A, layer: 1, pos: 22
type: B, layer: 1, pos: 42
type: B, layer: 1, pos: 39
type: A, layer: 1, pos: 23
type: B, layer: 1, pos: 8
type: A, layer: 1, pos: 16
type: A, layer: 1, pos: 11
type: B, layer: 1, pos: 37
type: B, layer: 1, pos: 22
type: A, layer: 1, pos: 42

Time for candidate selection: 0.07 seconds

### Candidate
type: A, layer: 1, pos: 41

### Candidate
type: B, layer: 1, pos: 41

### Candidate
type: A, layer: 1, pos: 19

### Candidate
type: B, layer: 1, pos: 6

### Candidate
type: A, layer: 1, pos: 6

### Candidate
type: A, layer: 1, pos: 21

### Candidate
type: B, layer: 1, pos: 25

### Candidate
type: A, layer: 1, pos: 25

### Candidate
type: B, layer: 1, pos: 19

### Candidate
type: B, layer: 1, pos: 46

## Relational analysis of NS_A2_A1_B1_A2_B2_A1_B1
Optimization infeasible because this subproblem isn't reachable.

### Candidate
type: B, layer: 1, pos: 4

## Relational analysis of NS_A2_A1_B1_A2_B2_A1_B1
Optimization infeasible because this subproblem isn't reachable.

### Candidate
type: B, layer: 1, pos: 11

## Relational analysis of NS_A2_A1_B1_A2_B2_A1_B1
Optimization infeasible because this subproblem isn't reachable.

### Candidate
type: A, layer: 1, pos: 14

## Relational analysis of NS_A2_A1_B1_A2_B2_A1_A1
Optimization infeasible because this subproblem isn't reachable.

### Candidate
type: B, layer: 1, pos: 21

### Candidate
type: A, layer: 1, pos: 38

## Relational analysis of NS_A2_A1_B1_A2_B2_A1_A1
Optimization infeasible because this subproblem isn't reachable.

### Candidate
type: B, layer: 1, pos: 38

### Candidate
type: A, layer: 1, pos: 29

## Relational analysis of NS_A2_A1_B1_A2_B2_A1_A1
Optimization infeasible because this subproblem isn't reachable.

### Candidate
type: B, layer: 1, pos: 13

## Relational analysis of NS_A2_A1_B1_A2_B2_A1_B1
Optimization infeasible because this subproblem isn't reachable.

### Candidate
type: A, layer: 1, pos: 46

### Candidate
type: A, layer: 1, pos: 12

## Relational analysis of NS_A2_A1_B1_A2_B2_A1_A1
Optimization infeasible because this subproblem isn't reachable.

### Candidate
type: B, layer: 1, pos: 16

## Relational analysis of NS_A2_A1_B1_A2_B2_A1_B1
Optimization infeasible because this subproblem isn't reachable.

### Candidate
type: B, layer: 1, pos: 12

### Candidate
type: A, layer: 1, pos: 39

## Relational analysis of NS_A2_A1_B1_A2_B2_A1_A1
Optimization infeasible because this subproblem isn't reachable.

### Candidate
type: A, layer: 1, pos: 4

### Candidate
type: B, layer: 1, pos: 43

## Relational analysis of NS_A2_A1_B1_A2_B2_A1_B1

### Relational analysis result of NS_A2_A1_B1_A2_B2_A1_B1
Status: Status.UNKNOWN
Output dim: 0, lower bound: -81.8608351, upper bound: 81.8615675
time: 0.83 seconds

## Relational analysis of NS_A2_A1_B1_A2_B2_A1_B2

### Relational analysis result of NS_A2_A1_B1_A2_B2_A1_B2
Status: Status.UNKNOWN
Output dim: 0, lower bound: -81.8608351, upper bound: 81.8615675
time: 0.70 seconds

## BFS NS instance: NS_A2_A1_B1_A2_B2_A2

### Backsubstitution after applying NS history:
0: -23.0393181, 84.1384583, -22.3477135, 81.5668259, -104.6061401, 106.4861603
1: -60.7940979, 191.6210938, -59.0682335, 185.8462830, -246.6403809, 250.6893311
2: -90.8825989, 169.5263519, -88.6273117, 163.8466644, -254.7292480, 258.1536255
3: -52.2806168, 204.3468628, -50.8008537, 198.3957214, -250.6763153, 255.1477051
4: -83.0892868, 149.3332672, -80.9325562, 144.4432983, -227.5325928, 230.2657776

Time for backsubstitution: 0.92 seconds

### NS candidates at layer 1
type: A, layer: 1, pos: 41
type: B, layer: 1, pos: 41
type: A, layer: 1, pos: 19
type: B, layer: 1, pos: 19
type: B, layer: 1, pos: 6
type: A, layer: 1, pos: 6
type: B, layer: 1, pos: 25
type: B, layer: 1, pos: 21
type: A, layer: 1, pos: 25
type: A, layer: 1, pos: 21
type: A, layer: 1, pos: 46
type: B, layer: 1, pos: 46
type: B, layer: 1, pos: 4
type: A, layer: 1, pos: 38
type: B, layer: 1, pos: 38
type: A, layer: 1, pos: 12
type: B, layer: 1, pos: 13
type: A, layer: 1, pos: 29
type: A, layer: 1, pos: 4
type: B, layer: 1, pos: 14
type: B, layer: 1, pos: 12
type: A, layer: 1, pos: 14
type: B, layer: 1, pos: 29
type: A, layer: 1, pos: 31
type: B, layer: 1, pos: 31
type: B, layer: 1, pos: 43
type: B, layer: 1, pos: 20
type: B, layer: 1, pos: 11
type: B, layer: 1, pos: 16
type: A, layer: 1, pos: 39
type: A, layer: 1, pos: 11
type: B, layer: 1, pos: 39
type: B, layer: 1, pos: 15
type: A, layer: 1, pos: 15
type: A, layer: 1, pos: 16
type: B, layer: 1, pos: 23
type: A, layer: 1, pos: 8
type: A, layer: 1, pos: 20
type: B, layer: 1, pos: 8
type: A, layer: 1, pos: 22
type: A, layer: 1, pos: 23
type: B, layer: 1, pos: 42
type: A, layer: 1, pos: 37
type: B, layer: 1, pos: 22
type: B, layer: 1, pos: 37
type: A, layer: 1, pos: 42

Time for candidate selection: 0.07 seconds

### Candidate
type: A, layer: 1, pos: 41

### Candidate
type: B, layer: 1, pos: 41

### Candidate
type: A, layer: 1, pos: 19

### Candidate
type: B, layer: 1, pos: 19

### Candidate
type: B, layer: 1, pos: 6

### Candidate
type: A, layer: 1, pos: 6

### Candidate
type: B, layer: 1, pos: 25

### Candidate
type: B, layer: 1, pos: 21

### Candidate
type: A, layer: 1, pos: 25

### Candidate
type: A, layer: 1, pos: 21

### Candidate
type: A, layer: 1, pos: 46

## Relational analysis of NS_A2_A1_B1_A2_B2_A2_A1
Optimization infeasible because this subproblem isn't reachable.

### Candidate
type: B, layer: 1, pos: 46

### Candidate
type: B, layer: 1, pos: 4

## Relational analysis of NS_A2_A1_B1_A2_B2_A2_B1
Optimization infeasible because this subproblem isn't reachable.

### Candidate
type: A, layer: 1, pos: 38

## Relational analysis of NS_A2_A1_B1_A2_B2_A2_A1
Optimization infeasible because this subproblem isn't reachable.

### Candidate
type: B, layer: 1, pos: 38

### Candidate
type: A, layer: 1, pos: 12

## Relational analysis of NS_A2_A1_B1_A2_B2_A2_A1
Optimization infeasible because this subproblem isn't reachable.

### Candidate
type: B, layer: 1, pos: 13

## Relational analysis of NS_A2_A1_B1_A2_B2_A2_B1
Optimization infeasible because this subproblem isn't reachable.

### Candidate
type: A, layer: 1, pos: 29

## Relational analysis of NS_A2_A1_B1_A2_B2_A2_A1
Optimization infeasible because this subproblem isn't reachable.

### Candidate
type: A, layer: 1, pos: 4

### Candidate
type: B, layer: 1, pos: 14

## Relational analysis of NS_A2_A1_B1_A2_B2_A2_B1
Optimization infeasible because this subproblem isn't reachable.

### Candidate
type: B, layer: 1, pos: 12

### Candidate
type: A, layer: 1, pos: 14

### Candidate
type: B, layer: 1, pos: 29

### Candidate
type: A, layer: 1, pos: 31

## Relational analysis of NS_A2_A1_B1_A2_B2_A2_A1
Optimization infeasible because this subproblem isn't reachable.

### Candidate
type: B, layer: 1, pos: 31

### Candidate
type: B, layer: 1, pos: 43

## Relational analysis of NS_A2_A1_B1_A2_B2_A2_B1

### Relational analysis result of NS_A2_A1_B1_A2_B2_A2_B1
Status: Status.UNKNOWN
Output dim: 0, lower bound: -81.8608351, upper bound: 81.8615610
time: 0.78 seconds

## Relational analysis of NS_A2_A1_B1_A2_B2_A2_B2

### Relational analysis result of NS_A2_A1_B1_A2_B2_A2_B2
Status: Status.UNKNOWN
Output dim: 0, lower bound: -81.8608351, upper bound: 81.8615610
time: 1.57 seconds

## BFS NS instance: NS_A2_A1_B2_A2_A2_B1

### Backsubstitution after applying NS history:
0: -20.1414471, 73.5443802, -20.6219311, 75.3099213, -95.4513702, 94.1663132
1: -53.4771767, 165.9335632, -54.7481003, 169.9193115, -223.3964844, 220.6816559
2: -80.1817780, 147.6730042, -82.1030884, 151.2281952, -231.4099579, 229.7760925
3: -46.0048676, 177.2870331, -47.1017303, 181.5316620, -227.5365295, 224.3887482
4: -73.2760468, 130.0593109, -75.0345535, 133.1882935, -206.4642944, 205.0938721

Time for backsubstitution: 1.23 seconds

### NS candidates at layer 1
type: B, layer: 1, pos: 41
type: A, layer: 1, pos: 41
type: B, layer: 1, pos: 4
type: B, layer: 1, pos: 19
type: A, layer: 1, pos: 19
type: A, layer: 1, pos: 6
type: B, layer: 1, pos: 6
type: B, layer: 1, pos: 25
type: B, layer: 1, pos: 26
type: A, layer: 1, pos: 25
type: B, layer: 1, pos: 21
type: A, layer: 1, pos: 26
type: B, layer: 1, pos: 46
type: A, layer: 1, pos: 21
type: A, layer: 1, pos: 46
type: B, layer: 1, pos: 38
type: A, layer: 1, pos: 38
type: B, layer: 1, pos: 13
type: A, layer: 1, pos: 29
type: B, layer: 1, pos: 14
type: A, layer: 1, pos: 12
type: B, layer: 1, pos: 29
type: B, layer: 1, pos: 11
type: A, layer: 1, pos: 14
type: B, layer: 1, pos: 12
type: A, layer: 1, pos: 31
type: B, layer: 1, pos: 31
type: B, layer: 1, pos: 39
type: A, layer: 1, pos: 11
type: A, layer: 1, pos: 39
type: A, layer: 1, pos: 16
type: A, layer: 1, pos: 15
type: B, layer: 1, pos: 15
type: B, layer: 1, pos: 16
type: B, layer: 1, pos: 20
type: A, layer: 1, pos: 37
type: B, layer: 1, pos: 37
type: B, layer: 1, pos: 22
type: A, layer: 1, pos: 23
type: A, layer: 1, pos: 20
type: B, layer: 1, pos: 23
type: A, layer: 1, pos: 22
type: A, layer: 1, pos: 42
type: B, layer: 1, pos: 8
type: A, layer: 1, pos: 8
type: B, layer: 1, pos: 42

Time for candidate selection: 0.09 seconds

### Candidate
type: B, layer: 1, pos: 41

### Candidate
type: A, layer: 1, pos: 41

### Candidate
type: B, layer: 1, pos: 4

### Candidate
type: B, layer: 1, pos: 19

### Candidate
type: A, layer: 1, pos: 19

### Candidate
type: A, layer: 1, pos: 6

### Candidate
type: B, layer: 1, pos: 6

### Candidate
type: B, layer: 1, pos: 25

### Candidate
type: B, layer: 1, pos: 26

### Candidate
type: A, layer: 1, pos: 25

### Candidate
type: B, layer: 1, pos: 21

### Candidate
type: A, layer: 1, pos: 26

### Candidate
type: B, layer: 1, pos: 46

### Candidate
type: A, layer: 1, pos: 21

### Candidate
type: A, layer: 1, pos: 46

### Candidate
type: B, layer: 1, pos: 38

### Candidate
type: A, layer: 1, pos: 38

### Candidate
type: B, layer: 1, pos: 13

### Candidate
type: A, layer: 1, pos: 29

### Candidate
type: B, layer: 1, pos: 14

### Candidate
type: A, layer: 1, pos: 12

## Relational analysis of NS_A2_A1_B2_A2_A2_B1_A1
Optimization infeasible because this subproblem isn't reachable.

### Candidate
type: B, layer: 1, pos: 29

### Candidate
type: B, layer: 1, pos: 11

## Relational analysis of NS_A2_A1_B2_A2_A2_B1_B1
Optimization infeasible because this subproblem isn't reachable.

### Candidate
type: A, layer: 1, pos: 14

### Candidate
type: B, layer: 1, pos: 12

### Candidate
type: A, layer: 1, pos: 31

## Relational analysis of NS_A2_A1_B2_A2_A2_B1_A1
Optimization infeasible because this subproblem isn't reachable.

### Candidate
type: B, layer: 1, pos: 31

### Candidate
type: B, layer: 1, pos: 39

## Relational analysis of NS_A2_A1_B2_A2_A2_B1_B1
Optimization infeasible because this subproblem isn't reachable.

### Candidate
type: A, layer: 1, pos: 11

### Candidate
type: A, layer: 1, pos: 39

### Candidate
type: A, layer: 1, pos: 16

## Relational analysis of NS_A2_A1_B2_A2_A2_B1_A1

### Relational analysis result of NS_A2_A1_B2_A2_A2_B1_A1
Status: Status.UNKNOWN
Output dim: 0, lower bound: -81.8605284, upper bound: 81.8611365
time: 0.79 seconds

## Relational analysis of NS_A2_A1_B2_A2_A2_B1_A2
Optimization infeasible because this subproblem isn't reachable.

### Candidate
type: A, layer: 1, pos: 15

## Relational analysis of NS_A2_A1_B2_A2_A2_B1_A1

### Relational analysis result of NS_A2_A1_B2_A2_A2_B1_A1
Status: Status.VERIFIED
Output dim: 0, lower bound: -81.8595408, upper bound: 81.8598082
time: 0.81 seconds

## Relational analysis of NS_A2_A1_B2_A2_A2_B1_A2

### Relational analysis result of NS_A2_A1_B2_A2_A2_B1_A2
Status: Status.UNKNOWN
Output dim: 0, lower bound: -81.8608126, upper bound: 81.8613877
time: 0.91 seconds

## BFS NS instance: NS_A2_A1_B2_A2_A2_B2

### Backsubstitution after applying NS history:
0: -20.1414471, 73.5443802, -20.6279507, 75.2376175, -95.3790665, 94.1723099
1: -53.4771767, 165.9335632, -54.8578377, 169.6214447, -223.0986176, 220.7913971
2: -80.1817780, 147.6730042, -82.3189621, 151.2639465, -231.4457245, 229.9919739
3: -46.0048676, 177.2870331, -47.1946831, 181.4897461, -227.4946136, 224.4816895
4: -73.2760468, 130.0593109, -75.2171707, 133.2788696, -206.5549164, 205.2764740

Time for backsubstitution: 1.00 seconds

### NS candidates at layer 1
type: A, layer: 1, pos: 41
type: A, layer: 1, pos: 25
type: B, layer: 1, pos: 25
type: B, layer: 1, pos: 41
type: B, layer: 1, pos: 12
type: B, layer: 1, pos: 19
type: A, layer: 1, pos: 14
type: A, layer: 1, pos: 12
type: B, layer: 1, pos: 4
type: B, layer: 1, pos: 14
type: B, layer: 1, pos: 26
type: B, layer: 1, pos: 29
type: A, layer: 1, pos: 26
type: A, layer: 1, pos: 19
type: A, layer: 1, pos: 23
type: A, layer: 1, pos: 29
type: A, layer: 1, pos: 8
type: B, layer: 1, pos: 37
type: B, layer: 1, pos: 23
type: A, layer: 1, pos: 37
type: B, layer: 1, pos: 8
type: A, layer: 1, pos: 11
type: B, layer: 1, pos: 22
type: A, layer: 1, pos: 6
type: A, layer: 1, pos: 46
type: B, layer: 1, pos: 6
type: A, layer: 1, pos: 20
type: A, layer: 1, pos: 21
type: A, layer: 1, pos: 22
type: B, layer: 1, pos: 21
type: B, layer: 1, pos: 46
type: B, layer: 1, pos: 13
type: A, layer: 1, pos: 38
type: B, layer: 1, pos: 38
type: B, layer: 1, pos: 20
type: A, layer: 1, pos: 16
type: A, layer: 1, pos: 39
type: B, layer: 1, pos: 11
type: A, layer: 1, pos: 31
type: B, layer: 1, pos: 31
type: A, layer: 1, pos: 15
type: B, layer: 1, pos: 42
type: B, layer: 1, pos: 39
type: B, layer: 1, pos: 15
type: B, layer: 1, pos: 16
type: A, layer: 1, pos: 42

Time for candidate selection: 0.07 seconds

### Candidate
type: A, layer: 1, pos: 41

### Candidate
type: A, layer: 1, pos: 25

### Candidate
type: B, layer: 1, pos: 25

### Candidate
type: B, layer: 1, pos: 41

### Candidate
type: B, layer: 1, pos: 12

## Relational analysis of NS_A2_A1_B2_A2_A2_B2_B1
Optimization infeasible because this subproblem isn't reachable.

### Candidate
type: B, layer: 1, pos: 19

### Candidate
type: A, layer: 1, pos: 14

### Candidate
type: A, layer: 1, pos: 12

### Candidate
type: B, layer: 1, pos: 4

### Candidate
type: B, layer: 1, pos: 14

### Candidate
type: B, layer: 1, pos: 26

### Candidate
type: B, layer: 1, pos: 29

### Candidate
type: A, layer: 1, pos: 26

### Candidate
type: A, layer: 1, pos: 19

### Candidate
type: A, layer: 1, pos: 23

## Relational analysis of NS_A2_A1_B2_A2_A2_B2_A1
Optimization infeasible because this subproblem isn't reachable.

### Candidate
type: A, layer: 1, pos: 29

### Candidate
type: A, layer: 1, pos: 8

## Relational analysis of NS_A2_A1_B2_A2_A2_B2_A1

### Relational analysis result of NS_A2_A1_B2_A2_A2_B2_A1
Status: Status.UNKNOWN
Output dim: 0, lower bound: -81.8607546, upper bound: 81.8613032
time: 0.95 seconds

## Relational analysis of NS_A2_A1_B2_A2_A2_B2_A2

### Relational analysis result of NS_A2_A1_B2_A2_A2_B2_A2
Status: Status.UNKNOWN
Output dim: 0, lower bound: -81.8607546, upper bound: 81.8613863
time: 0.83 seconds

## Summary of splitting at layer (split count: 6)
- Time for NS candidates: 3.39 seconds
NS_A2_A1_B1_A2_B1_A1_B1, status: Status.UNKNOWN, split count: 7, time: 3.39
Output dim: 0, lower bound: -81.8608383, upper bound: 81.8614238
NS_A2_A1_B1_A2_B1_A1_B2, status: Status.UNKNOWN, split count: 7, time: 3.39
Output dim: 0, lower bound: -81.8608383, upper bound: 81.8614238
NS_A2_A1_B1_A2_B1_A2_A1, status: Status.UNKNOWN, split count: 7, time: 3.39
Output dim: 0, lower bound: -81.8608383, upper bound: 81.8614238
NS_A2_A1_B1_A2_B1_A2_A2, status: Status.UNKNOWN, split count: 7, time: 3.39
Output dim: 0, lower bound: -81.8607763, upper bound: 81.8614029
NS_A2_A1_B1_A2_B2_A1_B1, status: Status.UNKNOWN, split count: 7, time: 3.39
Output dim: 0, lower bound: -81.8608351, upper bound: 81.8615675
NS_A2_A1_B1_A2_B2_A1_B2, status: Status.UNKNOWN, split count: 7, time: 3.39
Output dim: 0, lower bound: -81.8608351, upper bound: 81.8615675
NS_A2_A1_B1_A2_B2_A2_B1, status: Status.UNKNOWN, split count: 7, time: 3.39
Output dim: 0, lower bound: -81.8608351, upper bound: 81.8615610
NS_A2_A1_B1_A2_B2_A2_B2, status: Status.UNKNOWN, split count: 7, time: 3.39
Output dim: 0, lower bound: -81.8608351, upper bound: 81.8615610
NS_A2_A1_B2_A2_A2_B1_A1, status: Status.VERIFIED, split count: 7, time: 3.39
Output dim: 0, lower bound: -81.8595408, upper bound: 81.8598082
NS_A2_A1_B2_A2_A2_B1_A2, status: Status.UNKNOWN, split count: 7, time: 3.39
Output dim: 0, lower bound: -81.8608126, upper bound: 81.8613877
NS_A2_A1_B2_A2_A2_B2_A1, status: Status.UNKNOWN, split count: 7, time: 3.39
Output dim: 0, lower bound: -81.8607546, upper bound: 81.8613032
NS_A2_A1_B2_A2_A2_B2_A2, status: Status.UNKNOWN, split count: 7, time: 3.39
Output dim: 0, lower bound: -81.8607546, upper bound: 81.8613863

## BFS NS instance: NS_A2_A1_B1_A2_B1_A1_B1

### Backsubstitution after applying NS history:
0: -19.5514183, 71.4037170, -18.9485092, 69.1741562, -88.7255630, 90.3522263
1: -51.9093094, 161.0701752, -50.4105797, 156.0391998, -207.9485016, 211.4807587
2: -77.8657913, 143.3444519, -75.9221573, 138.5529175, -216.4186707, 219.2666016
3: -44.6583481, 172.0934601, -43.3704910, 166.8921967, -211.5505219, 215.4639587
4: -71.1484833, 126.2585678, -69.3036575, 122.1093903, -193.2578735, 195.5622253

Time for backsubstitution: 0.90 seconds

### NS candidates at layer 1
type: A, layer: 1, pos: 41
type: B, layer: 1, pos: 41
type: A, layer: 1, pos: 19
type: B, layer: 1, pos: 19
type: B, layer: 1, pos: 6
type: A, layer: 1, pos: 6
type: B, layer: 1, pos: 25
type: B, layer: 1, pos: 21
type: A, layer: 1, pos: 25
type: B, layer: 1, pos: 4
type: A, layer: 1, pos: 46
type: A, layer: 1, pos: 21
type: A, layer: 1, pos: 38
type: B, layer: 1, pos: 46
type: A, layer: 1, pos: 4
type: B, layer: 1, pos: 13
type: A, layer: 1, pos: 29
type: B, layer: 1, pos: 38
type: B, layer: 1, pos: 14
type: A, layer: 1, pos: 14
type: B, layer: 1, pos: 29
type: A, layer: 1, pos: 12
type: B, layer: 1, pos: 12
type: B, layer: 1, pos: 11
type: A, layer: 1, pos: 31
type: B, layer: 1, pos: 31
type: A, layer: 1, pos: 11
type: B, layer: 1, pos: 16
type: A, layer: 1, pos: 39
type: B, layer: 1, pos: 39
type: B, layer: 1, pos: 15
type: B, layer: 1, pos: 20
type: A, layer: 1, pos: 15
type: A, layer: 1, pos: 16
type: A, layer: 1, pos: 37
type: B, layer: 1, pos: 37
type: B, layer: 1, pos: 23
type: A, layer: 1, pos: 22
type: B, layer: 1, pos: 22
type: A, layer: 1, pos: 8
type: B, layer: 1, pos: 42
type: A, layer: 1, pos: 23
type: B, layer: 1, pos: 8
type: A, layer: 1, pos: 20
type: A, layer: 1, pos: 42

Time for candidate selection: 0.07 seconds

### Candidate
type: A, layer: 1, pos: 41

### Candidate
type: B, layer: 1, pos: 41

### Candidate
type: A, layer: 1, pos: 19

### Candidate
type: B, layer: 1, pos: 19

### Candidate
type: B, layer: 1, pos: 6

### Candidate
type: A, layer: 1, pos: 6

### Candidate
type: B, layer: 1, pos: 25

### Candidate
type: B, layer: 1, pos: 21

### Candidate
type: A, layer: 1, pos: 25

### Candidate
type: B, layer: 1, pos: 4

### Candidate
type: A, layer: 1, pos: 46

### Candidate
type: A, layer: 1, pos: 21

### Candidate
type: A, layer: 1, pos: 38

### Candidate
type: B, layer: 1, pos: 46

### Candidate
type: A, layer: 1, pos: 4

### Candidate
type: B, layer: 1, pos: 13

### Candidate
type: A, layer: 1, pos: 29

### Candidate
type: B, layer: 1, pos: 38

### Candidate
type: B, layer: 1, pos: 14

### Candidate
type: A, layer: 1, pos: 14

### Candidate
type: B, layer: 1, pos: 29

### Candidate
type: A, layer: 1, pos: 12

### Candidate
type: B, layer: 1, pos: 12

### Candidate
type: B, layer: 1, pos: 11

## Relational analysis of NS_A2_A1_B1_A2_B1_A1_B1_B1
Optimization infeasible because this subproblem isn't reachable.

### Candidate
type: A, layer: 1, pos: 31

## Relational analysis of NS_A2_A1_B1_A2_B1_A1_B1_A1
Optimization infeasible because this subproblem isn't reachable.

### Candidate
type: B, layer: 1, pos: 31

### Candidate
type: A, layer: 1, pos: 11

### Candidate
type: B, layer: 1, pos: 16

## Relational analysis of NS_A2_A1_B1_A2_B1_A1_B1_B1
Optimization infeasible because this subproblem isn't reachable.

### Candidate
type: A, layer: 1, pos: 39

## Relational analysis of NS_A2_A1_B1_A2_B1_A1_B1_A1
Optimization infeasible because this subproblem isn't reachable.

### Candidate
type: B, layer: 1, pos: 39

### Candidate
type: B, layer: 1, pos: 15

## Relational analysis of NS_A2_A1_B1_A2_B1_A1_B1_B1
Optimization infeasible because this subproblem isn't reachable.

### Candidate
type: B, layer: 1, pos: 20

## Relational analysis of NS_A2_A1_B1_A2_B1_A1_B1_B1
Optimization infeasible because this subproblem isn't reachable.

### Candidate
type: A, layer: 1, pos: 15

### Candidate
type: A, layer: 1, pos: 16

### Candidate
type: A, layer: 1, pos: 37

## Relational analysis of NS_A2_A1_B1_A2_B1_A1_B1_A1
Optimization infeasible because this subproblem isn't reachable.

### Candidate
type: B, layer: 1, pos: 37

### Candidate
type: B, layer: 1, pos: 23

## Relational analysis of NS_A2_A1_B1_A2_B1_A1_B1_B1
Optimization infeasible because this subproblem isn't reachable.

### Candidate
type: A, layer: 1, pos: 22

## Relational analysis of NS_A2_A1_B1_A2_B1_A1_B1_A1

### Relational analysis result of NS_A2_A1_B1_A2_B1_A1_B1_A1
Status: Status.UNKNOWN
Output dim: 0, lower bound: -81.8605141, upper bound: 81.8611964
time: 1.22 seconds

## Relational analysis of NS_A2_A1_B1_A2_B1_A1_B1_A2

### Relational analysis result of NS_A2_A1_B1_A2_B1_A1_B1_A2
Status: Status.UNKNOWN
Output dim: 0, lower bound: -81.8604268, upper bound: 81.8610514
time: 0.87 seconds

## BFS NS instance: NS_A2_A1_B1_A2_B1_A1_B2

### Backsubstitution after applying NS history:
0: -19.5514183, 71.4037170, -19.0046711, 69.3115234, -88.8629456, 90.4083710
1: -51.9093094, 161.0701752, -50.6510124, 156.2105865, -208.1198883, 211.7211914
2: -77.8657913, 143.3444519, -76.3190994, 138.9543457, -216.8200989, 219.6635284
3: -44.6583481, 172.0934601, -43.5779457, 167.3686371, -212.0269623, 215.6714020
4: -71.1484833, 126.2585678, -69.6552200, 122.5172653, -193.6657410, 195.9137878

Time for backsubstitution: 0.83 seconds

### NS candidates at layer 1
type: A, layer: 1, pos: 41
type: A, layer: 1, pos: 25
type: B, layer: 1, pos: 25
type: B, layer: 1, pos: 12
type: A, layer: 1, pos: 14
type: B, layer: 1, pos: 41
type: B, layer: 1, pos: 19
type: A, layer: 1, pos: 12
type: B, layer: 1, pos: 14
type: B, layer: 1, pos: 29
type: A, layer: 1, pos: 19
type: A, layer: 1, pos: 8
type: A, layer: 1, pos: 29
type: A, layer: 1, pos: 23
type: B, layer: 1, pos: 37
type: B, layer: 1, pos: 23
type: A, layer: 1, pos: 37
type: A, layer: 1, pos: 11
type: B, layer: 1, pos: 8
type: B, layer: 1, pos: 22
type: A, layer: 1, pos: 46
type: A, layer: 1, pos: 6
type: B, layer: 1, pos: 6
type: B, layer: 1, pos: 4
type: A, layer: 1, pos: 21
type: A, layer: 1, pos: 22
type: A, layer: 1, pos: 20
type: A, layer: 1, pos: 4
type: A, layer: 1, pos: 38
type: B, layer: 1, pos: 13
type: B, layer: 1, pos: 21
type: B, layer: 1, pos: 20
type: B, layer: 1, pos: 38
type: B, layer: 1, pos: 46
type: A, layer: 1, pos: 39
type: A, layer: 1, pos: 16
type: A, layer: 1, pos: 31
type: B, layer: 1, pos: 11
type: B, layer: 1, pos: 42
type: B, layer: 1, pos: 31
type: A, layer: 1, pos: 15
type: B, layer: 1, pos: 39
type: B, layer: 1, pos: 15
type: B, layer: 1, pos: 16
type: A, layer: 1, pos: 42

Time for candidate selection: 0.07 seconds

### Candidate
type: A, layer: 1, pos: 41

### Candidate
type: A, layer: 1, pos: 25

### Candidate
type: B, layer: 1, pos: 25

### Candidate
type: B, layer: 1, pos: 12

### Candidate
type: A, layer: 1, pos: 14

### Candidate
type: B, layer: 1, pos: 41

### Candidate
type: B, layer: 1, pos: 19

### Candidate
type: A, layer: 1, pos: 12

### Candidate
type: B, layer: 1, pos: 14

### Candidate
type: B, layer: 1, pos: 29

### Candidate
type: A, layer: 1, pos: 19

### Candidate
type: A, layer: 1, pos: 8

## Relational analysis of NS_A2_A1_B1_A2_B1_A1_B2_A1

### Relational analysis result of NS_A2_A1_B1_A2_B1_A1_B2_A1
Status: Status.UNKNOWN
Output dim: 0, lower bound: -81.8607922, upper bound: 81.8613221
time: 0.90 seconds

## Relational analysis of NS_A2_A1_B1_A2_B1_A1_B2_A2
Optimization infeasible because this subproblem isn't reachable.

### Candidate
type: A, layer: 1, pos: 29

### Candidate
type: A, layer: 1, pos: 23

## Relational analysis of NS_A2_A1_B1_A2_B1_A1_B2_A1
Optimization infeasible because this subproblem isn't reachable.

### Candidate
type: B, layer: 1, pos: 37

## Relational analysis of NS_A2_A1_B1_A2_B1_A1_B2_B1
Optimization infeasible because this subproblem isn't reachable.

### Candidate
type: B, layer: 1, pos: 23

### Candidate
type: A, layer: 1, pos: 37

### Candidate
type: A, layer: 1, pos: 11

## Relational analysis of NS_A2_A1_B1_A2_B1_A1_B2_A1
Optimization infeasible because this subproblem isn't reachable.

### Candidate
type: B, layer: 1, pos: 8

### Candidate
type: B, layer: 1, pos: 22

## Relational analysis of NS_A2_A1_B1_A2_B1_A1_B2_B1

### Relational analysis result of NS_A2_A1_B1_A2_B1_A1_B2_B1
Status: Status.UNKNOWN
Output dim: 0, lower bound: -81.8605715, upper bound: 81.8611176
time: 1.06 seconds

## Relational analysis of NS_A2_A1_B1_A2_B1_A1_B2_B2

### Relational analysis result of NS_A2_A1_B1_A2_B1_A1_B2_B2
Status: Status.UNKNOWN
Output dim: 0, lower bound: -81.8604268, upper bound: 81.8610514
time: 0.85 seconds

## BFS NS instance: NS_A2_A1_B1_A2_B1_A2_A1

### Backsubstitution after applying NS history:
0: -21.6661549, 79.1037064, -18.8034267, 68.6416092, -90.3077621, 97.9071198
1: -57.1756744, 180.2160187, -50.0376701, 154.7985229, -211.9741516, 230.2536926
2: -85.4210587, 159.3976746, -75.3637695, 137.4893494, -222.9104004, 234.7614288
3: -49.1567116, 192.1778412, -43.0481796, 165.5693054, -214.7260132, 235.2260132
4: -78.1009827, 140.3668823, -68.7939148, 121.1632767, -199.2642365, 209.1607666

Time for backsubstitution: 0.82 seconds

### NS candidates at layer 1
type: A, layer: 1, pos: 41
type: B, layer: 1, pos: 19
type: B, layer: 1, pos: 41
type: B, layer: 1, pos: 21
type: A, layer: 1, pos: 6
type: B, layer: 1, pos: 6
type: A, layer: 1, pos: 46
type: A, layer: 1, pos: 25
type: A, layer: 1, pos: 11
type: B, layer: 1, pos: 25
type: A, layer: 1, pos: 4
type: A, layer: 1, pos: 19
type: A, layer: 1, pos: 38
type: B, layer: 1, pos: 14
type: A, layer: 1, pos: 12
type: B, layer: 1, pos: 38
type: B, layer: 1, pos: 12
type: B, layer: 1, pos: 13
type: B, layer: 1, pos: 39
type: B, layer: 1, pos: 29
type: B, layer: 1, pos: 4
type: A, layer: 1, pos: 29
type: B, layer: 1, pos: 20
type: A, layer: 1, pos: 21
type: B, layer: 1, pos: 46
type: A, layer: 1, pos: 15
type: A, layer: 1, pos: 31
type: B, layer: 1, pos: 31
type: A, layer: 1, pos: 14
type: B, layer: 1, pos: 37
type: B, layer: 1, pos: 23
type: B, layer: 1, pos: 22
type: B, layer: 1, pos: 43
type: B, layer: 1, pos: 16
type: B, layer: 1, pos: 15
type: A, layer: 1, pos: 42
type: B, layer: 1, pos: 8
type: A, layer: 1, pos: 8
type: A, layer: 1, pos: 39
type: A, layer: 1, pos: 20
type: A, layer: 1, pos: 23
type: A, layer: 1, pos: 37
type: B, layer: 1, pos: 42
type: A, layer: 1, pos: 22
type: B, layer: 1, pos: 11

Time for candidate selection: 0.07 seconds

### Candidate
type: A, layer: 1, pos: 41

### Candidate
type: B, layer: 1, pos: 19

### Candidate
type: B, layer: 1, pos: 41

### Candidate
type: B, layer: 1, pos: 21

### Candidate
type: A, layer: 1, pos: 6

### Candidate
type: B, layer: 1, pos: 6

### Candidate
type: A, layer: 1, pos: 46

### Candidate
type: A, layer: 1, pos: 25

### Candidate
type: A, layer: 1, pos: 11

### Candidate
type: B, layer: 1, pos: 25

### Candidate
type: A, layer: 1, pos: 4

### Candidate
type: A, layer: 1, pos: 19

### Candidate
type: A, layer: 1, pos: 38

### Candidate
type: B, layer: 1, pos: 14

### Candidate
type: A, layer: 1, pos: 12

### Candidate
type: B, layer: 1, pos: 38

### Candidate
type: B, layer: 1, pos: 12

### Candidate
type: B, layer: 1, pos: 13

### Candidate
type: B, layer: 1, pos: 39

### Candidate
type: B, layer: 1, pos: 29

### Candidate
type: B, layer: 1, pos: 4

### Candidate
type: A, layer: 1, pos: 29

### Candidate
type: B, layer: 1, pos: 20

### Candidate
type: A, layer: 1, pos: 21

### Candidate
type: B, layer: 1, pos: 46

### Candidate
type: A, layer: 1, pos: 15

## Relational analysis of NS_A2_A1_B1_A2_B1_A2_A1_A1
Optimization infeasible because this subproblem isn't reachable.

### Candidate
type: A, layer: 1, pos: 31

## Relational analysis of NS_A2_A1_B1_A2_B1_A2_A1_A1
Optimization infeasible because this subproblem isn't reachable.

### Candidate
type: B, layer: 1, pos: 31

### Candidate
type: A, layer: 1, pos: 14

### Candidate
type: B, layer: 1, pos: 37

## Relational analysis of NS_A2_A1_B1_A2_B1_A2_A1_B1
Optimization infeasible because this subproblem isn't reachable.

### Candidate
type: B, layer: 1, pos: 23

## Relational analysis of NS_A2_A1_B1_A2_B1_A2_A1_B1
Optimization infeasible because this subproblem isn't reachable.

### Candidate
type: B, layer: 1, pos: 22

## Relational analysis of NS_A2_A1_B1_A2_B1_A2_A1_B1

### Relational analysis result of NS_A2_A1_B1_A2_B1_A2_A1_B1
Status: Status.UNKNOWN
Output dim: 0, lower bound: -81.8605726, upper bound: 81.8611340
time: 0.82 seconds

## Relational analysis of NS_A2_A1_B1_A2_B1_A2_A1_B2

### Relational analysis result of NS_A2_A1_B1_A2_B1_A2_A1_B2
Status: Status.UNKNOWN
Output dim: 0, lower bound: -81.8604279, upper bound: 81.8610697
time: 0.85 seconds

## BFS NS instance: NS_A2_A1_B1_A2_B1_A2_A2

### Backsubstitution after applying NS history:
0: -22.6331539, 82.6661758, -18.8741322, 68.9050751, -91.5382309, 101.5402908
1: -59.7047958, 188.2940369, -50.2090187, 155.4353638, -215.1401672, 238.5030518
2: -89.2544403, 166.6350403, -75.6109161, 138.0349426, -227.2893829, 242.2459412
3: -51.3380013, 200.7817230, -43.1954498, 166.2319794, -217.5699463, 243.9771271
4: -81.5932159, 146.7880249, -69.0207138, 121.6462173, -203.2394104, 215.8087158

Time for backsubstitution: 0.84 seconds

### NS candidates at layer 1
type: B, layer: 1, pos: 41
type: A, layer: 1, pos: 41
type: B, layer: 1, pos: 19
type: B, layer: 1, pos: 21
type: A, layer: 1, pos: 6
type: B, layer: 1, pos: 6
type: A, layer: 1, pos: 19
type: A, layer: 1, pos: 46
type: A, layer: 1, pos: 25
type: B, layer: 1, pos: 25
type: A, layer: 1, pos: 38
type: A, layer: 1, pos: 4
type: B, layer: 1, pos: 14
type: B, layer: 1, pos: 16
type: A, layer: 1, pos: 11
type: B, layer: 1, pos: 4
type: B, layer: 1, pos: 20
type: B, layer: 1, pos: 12
type: A, layer: 1, pos: 21
type: B, layer: 1, pos: 13
type: B, layer: 1, pos: 39
type: A, layer: 1, pos: 12
type: A, layer: 1, pos: 29
type: B, layer: 1, pos: 38
type: B, layer: 1, pos: 29
type: B, layer: 1, pos: 46
type: B, layer: 1, pos: 31
type: A, layer: 1, pos: 31
type: B, layer: 1, pos: 15
type: A, layer: 1, pos: 14
type: B, layer: 1, pos: 43
type: A, layer: 1, pos: 15
type: B, layer: 1, pos: 37
type: B, layer: 1, pos: 23
type: A, layer: 1, pos: 8
type: A, layer: 1, pos: 23
type: A, layer: 1, pos: 42
type: A, layer: 1, pos: 39
type: B, layer: 1, pos: 11
type: B, layer: 1, pos: 8
type: B, layer: 1, pos: 22
type: A, layer: 1, pos: 22
type: B, layer: 1, pos: 42
type: A, layer: 1, pos: 37
type: A, layer: 1, pos: 20

Time for candidate selection: 0.06 seconds

### Candidate
type: B, layer: 1, pos: 41

### Candidate
type: A, layer: 1, pos: 41

### Candidate
type: B, layer: 1, pos: 19

### Candidate
type: B, layer: 1, pos: 21

### Candidate
type: A, layer: 1, pos: 6

### Candidate
type: B, layer: 1, pos: 6

### Candidate
type: A, layer: 1, pos: 19

### Candidate
type: A, layer: 1, pos: 46

### Candidate
type: A, layer: 1, pos: 25

### Candidate
type: B, layer: 1, pos: 25

### Candidate
type: A, layer: 1, pos: 38

### Candidate
type: A, layer: 1, pos: 4

### Candidate
type: B, layer: 1, pos: 14

### Candidate
type: B, layer: 1, pos: 16

## Relational analysis of NS_A2_A1_B1_A2_B1_A2_A2_B1
Optimization infeasible because this subproblem isn't reachable.

### Candidate
type: A, layer: 1, pos: 11

### Candidate
type: B, layer: 1, pos: 4

### Candidate
type: B, layer: 1, pos: 20

### Candidate
type: B, layer: 1, pos: 12

### Candidate
type: A, layer: 1, pos: 21

### Candidate
type: B, layer: 1, pos: 13

### Candidate
type: B, layer: 1, pos: 39

### Candidate
type: A, layer: 1, pos: 12

### Candidate
type: A, layer: 1, pos: 29

### Candidate
type: B, layer: 1, pos: 38

### Candidate
type: B, layer: 1, pos: 29

### Candidate
type: B, layer: 1, pos: 46

### Candidate
type: B, layer: 1, pos: 31

## Relational analysis of NS_A2_A1_B1_A2_B1_A2_A2_B1
Optimization infeasible because this subproblem isn't reachable.

### Candidate
type: A, layer: 1, pos: 31

### Candidate
type: B, layer: 1, pos: 15

## Relational analysis of NS_A2_A1_B1_A2_B1_A2_A2_B1
Optimization infeasible because this subproblem isn't reachable.

### Candidate
type: A, layer: 1, pos: 14

### Candidate
type: B, layer: 1, pos: 43

## Relational analysis of NS_A2_A1_B1_A2_B1_A2_A2_B1

### Relational analysis result of NS_A2_A1_B1_A2_B1_A2_A2_B1
Status: Status.UNKNOWN
Output dim: 0, lower bound: -81.8607932, upper bound: 81.8614029
time: 0.85 seconds

## Relational analysis of NS_A2_A1_B1_A2_B1_A2_A2_B2

### Relational analysis result of NS_A2_A1_B1_A2_B1_A2_A2_B2
Status: Status.UNKNOWN
Output dim: 0, lower bound: -81.8607932, upper bound: 81.8614029
time: 0.79 seconds

## BFS NS instance: NS_A2_A1_B1_A2_B2_A1_B1

### Backsubstitution after applying NS history:
0: -19.5407543, 71.3658142, -22.2980099, 81.3815079, -100.9222641, 93.6638260
1: -51.8812523, 160.9836731, -58.9381638, 185.4232025, -237.3044586, 219.9218140
2: -77.8219833, 143.2691956, -88.4381561, 163.4754028, -241.2973938, 231.7073517
3: -44.6342278, 172.0003357, -50.6893501, 197.9528656, -242.5870819, 222.6896820
4: -71.1089172, 126.1929550, -80.7585754, 144.1177216, -215.2266235, 206.9515381

Time for backsubstitution: 0.82 seconds

### NS candidates at layer 1
type: A, layer: 1, pos: 41
type: B, layer: 1, pos: 41
type: A, layer: 1, pos: 19
type: B, layer: 1, pos: 6
type: A, layer: 1, pos: 6
type: A, layer: 1, pos: 21
type: B, layer: 1, pos: 25
type: A, layer: 1, pos: 25
type: B, layer: 1, pos: 19
type: B, layer: 1, pos: 46
type: B, layer: 1, pos: 4
type: A, layer: 1, pos: 14
type: B, layer: 1, pos: 11
type: A, layer: 1, pos: 38
type: B, layer: 1, pos: 21
type: B, layer: 1, pos: 38
type: A, layer: 1, pos: 29
type: B, layer: 1, pos: 13
type: A, layer: 1, pos: 46
type: A, layer: 1, pos: 12
type: B, layer: 1, pos: 16
type: B, layer: 1, pos: 12
type: A, layer: 1, pos: 39
type: A, layer: 1, pos: 4
type: B, layer: 1, pos: 15
type: B, layer: 1, pos: 31
type: B, layer: 1, pos: 14
type: A, layer: 1, pos: 31
type: B, layer: 1, pos: 29
type: A, layer: 1, pos: 37
type: A, layer: 1, pos: 20
type: B, layer: 1, pos: 20
type: A, layer: 1, pos: 8
type: A, layer: 1, pos: 15
type: B, layer: 1, pos: 23
type: B, layer: 1, pos: 42
type: A, layer: 1, pos: 22
type: B, layer: 1, pos: 39
type: A, layer: 1, pos: 23
type: B, layer: 1, pos: 8
type: A, layer: 1, pos: 11
type: A, layer: 1, pos: 16
type: B, layer: 1, pos: 37
type: B, layer: 1, pos: 22
type: A, layer: 1, pos: 42

Time for candidate selection: 0.07 seconds

### Candidate
type: A, layer: 1, pos: 41

### Candidate
type: B, layer: 1, pos: 41

### Candidate
type: A, layer: 1, pos: 19

### Candidate
type: B, layer: 1, pos: 6

### Candidate
type: A, layer: 1, pos: 6

### Candidate
type: A, layer: 1, pos: 21

### Candidate
type: B, layer: 1, pos: 25

### Candidate
type: A, layer: 1, pos: 25

### Candidate
type: B, layer: 1, pos: 19

### Candidate
type: B, layer: 1, pos: 46

### Candidate
type: B, layer: 1, pos: 4

### Candidate
type: A, layer: 1, pos: 14

### Candidate
type: B, layer: 1, pos: 11

### Candidate
type: A, layer: 1, pos: 38

### Candidate
type: B, layer: 1, pos: 21

### Candidate
type: B, layer: 1, pos: 38

### Candidate
type: A, layer: 1, pos: 29

### Candidate
type: B, layer: 1, pos: 13

### Candidate
type: A, layer: 1, pos: 46

### Candidate
type: A, layer: 1, pos: 12

### Candidate
type: B, layer: 1, pos: 16

### Candidate
type: B, layer: 1, pos: 12

### Candidate
type: A, layer: 1, pos: 39

### Candidate
type: A, layer: 1, pos: 4

### Candidate
type: B, layer: 1, pos: 15

## Relational analysis of NS_A2_A1_B1_A2_B2_A1_B1_B1
Optimization infeasible because this subproblem isn't reachable.

### Candidate
type: B, layer: 1, pos: 31

## Relational analysis of NS_A2_A1_B1_A2_B2_A1_B1_B1
Optimization infeasible because this subproblem isn't reachable.

### Candidate
type: B, layer: 1, pos: 14

### Candidate
type: A, layer: 1, pos: 31

### Candidate
type: B, layer: 1, pos: 29

### Candidate
type: A, layer: 1, pos: 37

## Relational analysis of NS_A2_A1_B1_A2_B2_A1_B1_A1
Optimization infeasible because this subproblem isn't reachable.

### Candidate
type: A, layer: 1, pos: 20

## Relational analysis of NS_A2_A1_B1_A2_B2_A1_B1_A1
Optimization infeasible because this subproblem isn't reachable.

### Candidate
type: B, layer: 1, pos: 20

### Candidate
type: A, layer: 1, pos: 8

## Relational analysis of NS_A2_A1_B1_A2_B2_A1_B1_A1
Optimization infeasible because this subproblem isn't reachable.

### Candidate
type: A, layer: 1, pos: 15

### Candidate
type: B, layer: 1, pos: 23

## Relational analysis of NS_A2_A1_B1_A2_B2_A1_B1_B1
Optimization infeasible because this subproblem isn't reachable.

### Candidate
type: B, layer: 1, pos: 42

## Relational analysis of NS_A2_A1_B1_A2_B2_A1_B1_B1

### Relational analysis result of NS_A2_A1_B1_A2_B2_A1_B1_B1
Status: Status.UNKNOWN
Output dim: 0, lower bound: -81.8606652, upper bound: 81.8614183
time: 0.85 seconds

## Relational analysis of NS_A2_A1_B1_A2_B2_A1_B1_B2

### Relational analysis result of NS_A2_A1_B1_A2_B2_A1_B1_B2
Status: Status.UNKNOWN
Output dim: 0, lower bound: -81.8610575, upper bound: 81.8615892
time: 0.82 seconds

## BFS NS instance: NS_A2_A1_B1_A2_B2_A1_B2

### Backsubstitution after applying NS history:
0: -19.5407543, 71.3658142, -22.3063660, 81.3184967, -100.8592529, 93.6721802
1: -51.8812523, 160.9836731, -59.0154343, 185.2219543, -237.1032104, 219.9990997
2: -77.8219833, 143.2691956, -88.5713730, 163.5092316, -241.3312073, 231.8405609
3: -44.6342278, 172.0003357, -50.7731934, 197.9091797, -242.5433960, 222.7735291
4: -71.1089172, 126.1929550, -80.8973312, 144.1884460, -215.2973480, 207.0902863

Time for backsubstitution: 0.81 seconds

### NS candidates at layer 1
type: A, layer: 1, pos: 41
type: A, layer: 1, pos: 25
type: B, layer: 1, pos: 25
type: A, layer: 1, pos: 14
type: A, layer: 1, pos: 19
type: B, layer: 1, pos: 12
type: B, layer: 1, pos: 41
type: A, layer: 1, pos: 12
type: B, layer: 1, pos: 29
type: B, layer: 1, pos: 14
type: A, layer: 1, pos: 8
type: A, layer: 1, pos: 29
type: B, layer: 1, pos: 19
type: A, layer: 1, pos: 23
type: A, layer: 1, pos: 37
type: B, layer: 1, pos: 23
type: B, layer: 1, pos: 37
type: A, layer: 1, pos: 21
type: B, layer: 1, pos: 8
type: A, layer: 1, pos: 11
type: B, layer: 1, pos: 22
type: A, layer: 1, pos: 6
type: B, layer: 1, pos: 4
type: A, layer: 1, pos: 20
type: B, layer: 1, pos: 6
type: A, layer: 1, pos: 22
type: A, layer: 1, pos: 46
type: B, layer: 1, pos: 46
type: A, layer: 1, pos: 4
type: B, layer: 1, pos: 13
type: B, layer: 1, pos: 38
type: A, layer: 1, pos: 38
type: A, layer: 1, pos: 39
type: B, layer: 1, pos: 11
type: B, layer: 1, pos: 20
type: B, layer: 1, pos: 42
type: B, layer: 1, pos: 21
type: A, layer: 1, pos: 31
type: B, layer: 1, pos: 31
type: B, layer: 1, pos: 16
type: B, layer: 1, pos: 15
type: A, layer: 1, pos: 16
type: A, layer: 1, pos: 15
type: B, layer: 1, pos: 39
type: A, layer: 1, pos: 42

Time for candidate selection: 0.07 seconds

### Candidate
type: A, layer: 1, pos: 41

### Candidate
type: A, layer: 1, pos: 25

### Candidate
type: B, layer: 1, pos: 25

### Candidate
type: A, layer: 1, pos: 14

### Candidate
type: A, layer: 1, pos: 19

### Candidate
type: B, layer: 1, pos: 12

### Candidate
type: B, layer: 1, pos: 41

### Candidate
type: A, layer: 1, pos: 12

### Candidate
type: B, layer: 1, pos: 29

### Candidate
type: B, layer: 1, pos: 14

### Candidate
type: A, layer: 1, pos: 8

## Relational analysis of NS_A2_A1_B1_A2_B2_A1_B2_A1

### Relational analysis result of NS_A2_A1_B1_A2_B2_A1_B2_A1
Status: Status.UNKNOWN
Output dim: 0, lower bound: -81.8610447, upper bound: 81.8615827
time: 0.71 seconds

## Relational analysis of NS_A2_A1_B1_A2_B2_A1_B2_A2
Optimization infeasible because this subproblem isn't reachable.

### Candidate
type: A, layer: 1, pos: 29

### Candidate
type: B, layer: 1, pos: 19

### Candidate
type: A, layer: 1, pos: 23

## Relational analysis of NS_A2_A1_B1_A2_B2_A1_B2_A1
Optimization infeasible because this subproblem isn't reachable.

### Candidate
type: A, layer: 1, pos: 37

## Relational analysis of NS_A2_A1_B1_A2_B2_A1_B2_A1
Optimization infeasible because this subproblem isn't reachable.

### Candidate
type: B, layer: 1, pos: 23

### Candidate
type: B, layer: 1, pos: 37

### Candidate
type: A, layer: 1, pos: 21

### Candidate
type: B, layer: 1, pos: 8

### Candidate
type: A, layer: 1, pos: 11

### Candidate
type: B, layer: 1, pos: 22

## Relational analysis of NS_A2_A1_B1_A2_B2_A1_B2_B1

### Relational analysis result of NS_A2_A1_B1_A2_B2_A1_B2_B1
Status: Status.UNKNOWN
Output dim: 0, lower bound: -81.8605715, upper bound: 81.8611810
time: 0.83 seconds

## Relational analysis of NS_A2_A1_B1_A2_B2_A1_B2_B2

### Relational analysis result of NS_A2_A1_B1_A2_B2_A1_B2_B2
Status: Status.UNKNOWN
Output dim: 0, lower bound: -81.8605486, upper bound: 81.8611519
time: 0.92 seconds

## BFS NS instance: NS_A2_A1_B1_A2_B2_A2_B1

### Backsubstitution after applying NS history:
0: -23.0393181, 84.1384583, -22.2980099, 81.3815079, -104.4208145, 106.4364700
1: -60.7940979, 191.6210938, -58.9381638, 185.4232025, -246.2173004, 250.5592651
2: -90.8825989, 169.5263519, -88.4381561, 163.4754028, -254.3580017, 257.9645081
3: -52.2806168, 204.3468628, -50.6893501, 197.9528656, -250.2334595, 255.0362091
4: -83.0892868, 149.3332672, -80.7585754, 144.1177216, -227.2070007, 230.0918427

Time for backsubstitution: 0.84 seconds

### NS candidates at layer 1
type: A, layer: 1, pos: 41
type: B, layer: 1, pos: 41
type: A, layer: 1, pos: 19
type: B, layer: 1, pos: 19
type: B, layer: 1, pos: 6
type: A, layer: 1, pos: 6
type: B, layer: 1, pos: 25
type: B, layer: 1, pos: 21
type: A, layer: 1, pos: 25
type: A, layer: 1, pos: 21
type: A, layer: 1, pos: 46
type: B, layer: 1, pos: 46
type: B, layer: 1, pos: 4
type: A, layer: 1, pos: 38
type: B, layer: 1, pos: 38
type: B, layer: 1, pos: 13
type: A, layer: 1, pos: 12
type: A, layer: 1, pos: 4
type: A, layer: 1, pos: 29
type: B, layer: 1, pos: 14
type: B, layer: 1, pos: 12
type: A, layer: 1, pos: 14
type: B, layer: 1, pos: 29
type: A, layer: 1, pos: 31
type: B, layer: 1, pos: 31
type: B, layer: 1, pos: 20
type: B, layer: 1, pos: 11
type: B, layer: 1, pos: 16
type: A, layer: 1, pos: 39
type: A, layer: 1, pos: 11
type: B, layer: 1, pos: 39
type: B, layer: 1, pos: 15
type: A, layer: 1, pos: 15
type: A, layer: 1, pos: 16
type: B, layer: 1, pos: 23
type: A, layer: 1, pos: 8
type: A, layer: 1, pos: 20
type: B, layer: 1, pos: 8
type: A, layer: 1, pos: 22
type: A, layer: 1, pos: 23
type: B, layer: 1, pos: 42
type: A, layer: 1, pos: 37
type: B, layer: 1, pos: 22
type: B, layer: 1, pos: 37
type: A, layer: 1, pos: 42

Time for candidate selection: 0.07 seconds

### Candidate
type: A, layer: 1, pos: 41

### Candidate
type: B, layer: 1, pos: 41

### Candidate
type: A, layer: 1, pos: 19

### Candidate
type: B, layer: 1, pos: 19

### Candidate
type: B, layer: 1, pos: 6

### Candidate
type: A, layer: 1, pos: 6

### Candidate
type: B, layer: 1, pos: 25

### Candidate
type: B, layer: 1, pos: 21

### Candidate
type: A, layer: 1, pos: 25

### Candidate
type: A, layer: 1, pos: 21

### Candidate
type: A, layer: 1, pos: 46

### Candidate
type: B, layer: 1, pos: 46

### Candidate
type: B, layer: 1, pos: 4

### Candidate
type: A, layer: 1, pos: 38

### Candidate
type: B, layer: 1, pos: 38

### Candidate
type: B, layer: 1, pos: 13

### Candidate
type: A, layer: 1, pos: 12

### Candidate
type: A, layer: 1, pos: 4

### Candidate
type: A, layer: 1, pos: 29

### Candidate
type: B, layer: 1, pos: 14

### Candidate
type: B, layer: 1, pos: 12

### Candidate
type: A, layer: 1, pos: 14

### Candidate
type: B, layer: 1, pos: 29

### Candidate
type: A, layer: 1, pos: 31

### Candidate
type: B, layer: 1, pos: 31

### Candidate
type: B, layer: 1, pos: 20

## Relational analysis of NS_A2_A1_B1_A2_B2_A2_B1_B1
Optimization infeasible because this subproblem isn't reachable.

### Candidate
type: B, layer: 1, pos: 11

## Relational analysis of NS_A2_A1_B1_A2_B2_A2_B1_B1
Optimization infeasible because this subproblem isn't reachable.

### Candidate
type: B, layer: 1, pos: 16

## Relational analysis of NS_A2_A1_B1_A2_B2_A2_B1_B1

### Relational analysis result of NS_A2_A1_B1_A2_B2_A2_B1_B1
Status: Status.UNKNOWN
Output dim: 0, lower bound: -81.8609240, upper bound: 81.8614205
time: 1.47 seconds

## Relational analysis of NS_A2_A1_B1_A2_B2_A2_B1_B2

### Relational analysis result of NS_A2_A1_B1_A2_B2_A2_B1_B2
Status: Status.UNKNOWN
Output dim: 0, lower bound: -81.8608459, upper bound: 81.8614131
time: 0.86 seconds

## BFS NS instance: NS_A2_A1_B1_A2_B2_A2_B2

### Backsubstitution after applying NS history:
0: -23.0393181, 84.1384583, -22.3063660, 81.3184967, -104.3578033, 106.4448242
1: -60.7940979, 191.6210938, -59.0154343, 185.2219543, -246.0160522, 250.6365356
2: -90.8825989, 169.5263519, -88.5713730, 163.5092316, -254.3918152, 258.0976868
3: -52.2806168, 204.3468628, -50.7731934, 197.9091797, -250.1897736, 255.1200562
4: -83.0892868, 149.3332672, -80.8973312, 144.1884460, -227.2777405, 230.2305908

Time for backsubstitution: 0.85 seconds

### NS candidates at layer 1
type: A, layer: 1, pos: 41
type: A, layer: 1, pos: 25
type: B, layer: 1, pos: 25
type: B, layer: 1, pos: 12
type: B, layer: 1, pos: 41
type: A, layer: 1, pos: 14
type: B, layer: 1, pos: 19
type: B, layer: 1, pos: 29
type: A, layer: 1, pos: 12
type: A, layer: 1, pos: 19
type: B, layer: 1, pos: 14
type: A, layer: 1, pos: 8
type: B, layer: 1, pos: 37
type: A, layer: 1, pos: 23
type: A, layer: 1, pos: 29
type: B, layer: 1, pos: 23
type: A, layer: 1, pos: 11
type: A, layer: 1, pos: 37
type: B, layer: 1, pos: 8
type: B, layer: 1, pos: 22
type: A, layer: 1, pos: 46
type: A, layer: 1, pos: 6
type: A, layer: 1, pos: 20
type: A, layer: 1, pos: 21
type: B, layer: 1, pos: 6
type: B, layer: 1, pos: 4
type: A, layer: 1, pos: 4
type: A, layer: 1, pos: 22
type: B, layer: 1, pos: 13
type: A, layer: 1, pos: 38
type: B, layer: 1, pos: 38
type: B, layer: 1, pos: 21
type: B, layer: 1, pos: 46
type: B, layer: 1, pos: 20
type: A, layer: 1, pos: 39
type: A, layer: 1, pos: 16
type: A, layer: 1, pos: 31
type: B, layer: 1, pos: 11
type: B, layer: 1, pos: 31
type: B, layer: 1, pos: 42
type: A, layer: 1, pos: 15
type: B, layer: 1, pos: 39
type: B, layer: 1, pos: 16
type: B, layer: 1, pos: 15
type: A, layer: 1, pos: 42

Time for candidate selection: 0.07 seconds

### Candidate
type: A, layer: 1, pos: 41

### Candidate
type: A, layer: 1, pos: 25

### Candidate
type: B, layer: 1, pos: 25

### Candidate
type: B, layer: 1, pos: 12

### Candidate
type: B, layer: 1, pos: 41

### Candidate
type: A, layer: 1, pos: 14

### Candidate
type: B, layer: 1, pos: 19

### Candidate
type: B, layer: 1, pos: 29

### Candidate
type: A, layer: 1, pos: 12

### Candidate
type: A, layer: 1, pos: 19

### Candidate
type: B, layer: 1, pos: 14

### Candidate
type: A, layer: 1, pos: 8

## Relational analysis of NS_A2_A1_B1_A2_B2_A2_B2_A1

### Relational analysis result of NS_A2_A1_B1_A2_B2_A2_B2_A1
Status: Status.UNKNOWN
Output dim: 0, lower bound: -81.8610381, upper bound: 81.8615567
time: 1.13 seconds

## Relational analysis of NS_A2_A1_B1_A2_B2_A2_B2_A2

### Relational analysis result of NS_A2_A1_B1_A2_B2_A2_B2_A2
Status: Status.UNKNOWN
Output dim: 0, lower bound: -81.8610429, upper bound: 81.8615547
time: 0.90 seconds

## BFS NS instance: NS_A2_A1_B2_A2_A2_B1_A2

### Backsubstitution after applying NS history:
0: -19.6327686, 71.7022171, -20.5378704, 75.0052719, -94.6380386, 92.2400894
1: -52.1400604, 161.7694092, -54.5269051, 169.2305298, -221.3705902, 216.2963104
2: -78.1802368, 143.9759064, -81.7714844, 150.6178589, -228.7980652, 225.7473755
3: -44.8481369, 172.8265839, -46.9105797, 180.7904968, -225.6386414, 219.7371521
4: -71.4469528, 126.8020782, -74.7318115, 132.6495361, -204.0964813, 201.5338898

Time for backsubstitution: 0.83 seconds

### NS candidates at layer 1
type: B, layer: 1, pos: 41
type: A, layer: 1, pos: 41
type: B, layer: 1, pos: 20
type: B, layer: 1, pos: 39
type: B, layer: 1, pos: 19
type: A, layer: 1, pos: 20
type: A, layer: 1, pos: 39
type: A, layer: 1, pos: 19
type: B, layer: 1, pos: 4
type: A, layer: 1, pos: 26
type: A, layer: 1, pos: 25
type: B, layer: 1, pos: 14
type: B, layer: 1, pos: 26
type: A, layer: 1, pos: 14
type: A, layer: 1, pos: 23
type: B, layer: 1, pos: 25
type: B, layer: 1, pos: 23
type: B, layer: 1, pos: 15
type: A, layer: 1, pos: 6
type: B, layer: 1, pos: 6
type: B, layer: 1, pos: 21
type: A, layer: 1, pos: 21
type: B, layer: 1, pos: 46
type: A, layer: 1, pos: 29
type: B, layer: 1, pos: 13
type: A, layer: 1, pos: 46
type: A, layer: 1, pos: 37
type: A, layer: 1, pos: 38
type: B, layer: 1, pos: 38
type: A, layer: 1, pos: 12
type: B, layer: 1, pos: 11
type: A, layer: 1, pos: 16
type: B, layer: 1, pos: 16
type: B, layer: 1, pos: 29
type: B, layer: 1, pos: 37
type: A, layer: 1, pos: 42
type: B, layer: 1, pos: 12
type: A, layer: 1, pos: 31
type: B, layer: 1, pos: 31
type: B, layer: 1, pos: 42
type: A, layer: 1, pos: 11
type: B, layer: 1, pos: 22
type: A, layer: 1, pos: 8
type: A, layer: 1, pos: 22
type: B, layer: 1, pos: 8

Time for candidate selection: 0.07 seconds

### Candidate
type: B, layer: 1, pos: 41

### Candidate
type: A, layer: 1, pos: 41

### Candidate
type: B, layer: 1, pos: 20

## Relational analysis of NS_A2_A1_B2_A2_A2_B1_A2_B1

### Relational analysis result of NS_A2_A1_B2_A2_A2_B1_A2_B1
Status: Status.UNKNOWN
Output dim: 0, lower bound: -81.8632979, upper bound: 81.8633526
time: 1.02 seconds

## Relational analysis of NS_A2_A1_B2_A2_A2_B1_A2_B2

### Relational analysis result of NS_A2_A1_B2_A2_A2_B1_A2_B2
Status: Status.UNKNOWN
Output dim: 0, lower bound: -81.8633128, upper bound: 81.8633536
time: 0.79 seconds

## BFS NS instance: NS_A2_A1_B2_A2_A2_B2_A1

### Backsubstitution after applying NS history:
0: -19.6868839, 71.8857117, -20.5253410, 74.8603363, -94.5472183, 92.4110565
1: -52.3027802, 162.1246490, -54.5889549, 168.7613373, -221.0641022, 216.7135925
2: -78.4741592, 144.2440796, -81.9313965, 150.4774017, -228.9515686, 226.1754608
3: -44.9923248, 173.2512360, -46.9648743, 180.5753937, -225.5677032, 220.2160950
4: -71.6982269, 127.0506744, -74.8585205, 132.5899811, -204.2882080, 201.9091949

Time for backsubstitution: 0.92 seconds

### NS candidates at layer 1
type: A, layer: 1, pos: 41
type: A, layer: 1, pos: 25
type: B, layer: 1, pos: 25
type: B, layer: 1, pos: 41
type: B, layer: 1, pos: 12
type: B, layer: 1, pos: 19
type: A, layer: 1, pos: 14
type: A, layer: 1, pos: 12
type: B, layer: 1, pos: 4
type: B, layer: 1, pos: 14
type: B, layer: 1, pos: 26
type: B, layer: 1, pos: 29
type: A, layer: 1, pos: 26
type: A, layer: 1, pos: 19
type: A, layer: 1, pos: 23
type: A, layer: 1, pos: 29
type: B, layer: 1, pos: 37
type: B, layer: 1, pos: 23
type: A, layer: 1, pos: 37
type: B, layer: 1, pos: 22
type: A, layer: 1, pos: 11
type: A, layer: 1, pos: 6
type: A, layer: 1, pos: 46
type: B, layer: 1, pos: 6
type: B, layer: 1, pos: 8
type: A, layer: 1, pos: 20
type: B, layer: 1, pos: 21
type: A, layer: 1, pos: 22
type: A, layer: 1, pos: 21
type: B, layer: 1, pos: 46
type: B, layer: 1, pos: 13
type: A, layer: 1, pos: 38
type: B, layer: 1, pos: 20
type: B, layer: 1, pos: 38
type: A, layer: 1, pos: 16
type: B, layer: 1, pos: 11
type: A, layer: 1, pos: 31
type: A, layer: 1, pos: 39
type: B, layer: 1, pos: 39
type: B, layer: 1, pos: 31
type: A, layer: 1, pos: 15
type: B, layer: 1, pos: 42
type: B, layer: 1, pos: 15
type: A, layer: 1, pos: 42
type: B, layer: 1, pos: 16

Time for candidate selection: 0.07 seconds

### Candidate
type: A, layer: 1, pos: 41

### Candidate
type: A, layer: 1, pos: 25

### Candidate
type: B, layer: 1, pos: 25

### Candidate
type: B, layer: 1, pos: 41

### Candidate
type: B, layer: 1, pos: 12

### Candidate
type: B, layer: 1, pos: 19

### Candidate
type: A, layer: 1, pos: 14

### Candidate
type: A, layer: 1, pos: 12

### Candidate
type: B, layer: 1, pos: 4

### Candidate
type: B, layer: 1, pos: 14

### Candidate
type: B, layer: 1, pos: 26

### Candidate
type: B, layer: 1, pos: 29

### Candidate
type: A, layer: 1, pos: 26

### Candidate
type: A, layer: 1, pos: 19

### Candidate
type: A, layer: 1, pos: 23

### Candidate
type: A, layer: 1, pos: 29

### Candidate
type: B, layer: 1, pos: 37

## Relational analysis of NS_A2_A1_B2_A2_A2_B2_A1_B1
Optimization infeasible because this subproblem isn't reachable.

### Candidate
type: B, layer: 1, pos: 23

### Candidate
type: A, layer: 1, pos: 37

### Candidate
type: B, layer: 1, pos: 22

## Relational analysis of NS_A2_A1_B2_A2_A2_B2_A1_B1
Optimization infeasible because this subproblem isn't reachable.

### Candidate
type: A, layer: 1, pos: 11

## Relational analysis of NS_A2_A1_B2_A2_A2_B2_A1_A1
Optimization infeasible because this subproblem isn't reachable.

### Candidate
type: A, layer: 1, pos: 6

### Candidate
type: A, layer: 1, pos: 46

### Candidate
type: B, layer: 1, pos: 6

### Candidate
type: B, layer: 1, pos: 8

## Relational analysis of NS_A2_A1_B2_A2_A2_B2_A1_B1
Optimization infeasible because this subproblem isn't reachable.

### Candidate
type: A, layer: 1, pos: 20

## Relational analysis of NS_A2_A1_B2_A2_A2_B2_A1_A1
Optimization infeasible because this subproblem isn't reachable.

### Candidate
type: B, layer: 1, pos: 21

### Candidate
type: A, layer: 1, pos: 22

### Candidate
type: A, layer: 1, pos: 21

### Candidate
type: B, layer: 1, pos: 46

### Candidate
type: B, layer: 1, pos: 13

### Candidate
type: A, layer: 1, pos: 38

### Candidate
type: B, layer: 1, pos: 20

### Candidate
type: B, layer: 1, pos: 38

### Candidate
type: A, layer: 1, pos: 16

## Relational analysis of NS_A2_A1_B2_A2_A2_B2_A1_A1

### Relational analysis result of NS_A2_A1_B2_A2_A2_B2_A1_A1
Status: Status.UNKNOWN
Output dim: 0, lower bound: -81.8604240, upper bound: 81.8610228
time: 0.83 seconds

## Relational analysis of NS_A2_A1_B2_A2_A2_B2_A1_A2
Optimization infeasible because this subproblem isn't reachable.

### Candidate
type: B, layer: 1, pos: 11

### Candidate
type: A, layer: 1, pos: 31

## Relational analysis of NS_A2_A1_B2_A2_A2_B2_A1_A1
Optimization infeasible because this subproblem isn't reachable.

### Candidate
type: A, layer: 1, pos: 39

## Relational analysis of NS_A2_A1_B2_A2_A2_B2_A1_A1
Optimization infeasible because this subproblem isn't reachable.

### Candidate
type: B, layer: 1, pos: 39

### Candidate
type: B, layer: 1, pos: 31

### Candidate
type: A, layer: 1, pos: 15

## Relational analysis of NS_A2_A1_B2_A2_A2_B2_A1_A1
Optimization infeasible because this subproblem isn't reachable.

### Candidate
type: B, layer: 1, pos: 42

## Relational analysis of NS_A2_A1_B2_A2_A2_B2_A1_B1

### Relational analysis result of NS_A2_A1_B2_A2_A2_B2_A1_B1
Status: Status.UNKNOWN
Output dim: 0, lower bound: -81.8605982, upper bound: 81.8611741
time: 0.95 seconds

## Relational analysis of NS_A2_A1_B2_A2_A2_B2_A1_B2

### Relational analysis result of NS_A2_A1_B2_A2_A2_B2_A1_B2
Status: Status.UNKNOWN
Output dim: 0, lower bound: -81.8607351, upper bound: 81.8612842
time: 0.86 seconds

## BFS NS instance: NS_A2_A1_B2_A2_A2_B2_A2

### Backsubstitution after applying NS history:
0: -20.0459213, 73.1830368, -20.1489220, 73.4936752, -93.5395966, 93.3319550
1: -53.1450500, 165.3082275, -53.5930939, 165.6540985, -218.7991333, 218.9013062
2: -79.5760498, 147.2333374, -80.4273300, 147.7969818, -227.3730316, 227.6606750
3: -45.7573967, 176.5436859, -46.1105843, 177.2835388, -223.0409393, 222.6542664
4: -72.7589798, 129.6905060, -73.4860153, 130.2353363, -202.9942932, 203.1765137

Time for backsubstitution: 0.85 seconds

### NS candidates at layer 1
type: A, layer: 1, pos: 41
type: B, layer: 1, pos: 41
type: A, layer: 1, pos: 25
type: B, layer: 1, pos: 25
type: B, layer: 1, pos: 12
type: B, layer: 1, pos: 19
type: B, layer: 1, pos: 4
type: B, layer: 1, pos: 14
type: A, layer: 1, pos: 14
type: B, layer: 1, pos: 26
type: A, layer: 1, pos: 12
type: B, layer: 1, pos: 29
type: A, layer: 1, pos: 26
type: A, layer: 1, pos: 19
type: A, layer: 1, pos: 23
type: B, layer: 1, pos: 37
type: A, layer: 1, pos: 29
type: B, layer: 1, pos: 22
type: B, layer: 1, pos: 23
type: A, layer: 1, pos: 37
type: A, layer: 1, pos: 11
type: A, layer: 1, pos: 6
type: B, layer: 1, pos: 21
type: B, layer: 1, pos: 6
type: A, layer: 1, pos: 46
type: B, layer: 1, pos: 8
type: B, layer: 1, pos: 20
type: B, layer: 1, pos: 46
type: B, layer: 1, pos: 13
type: B, layer: 1, pos: 38
type: A, layer: 1, pos: 20
type: A, layer: 1, pos: 38
type: A, layer: 1, pos: 22
type: A, layer: 1, pos: 16
type: A, layer: 1, pos: 21
type: B, layer: 1, pos: 39
type: B, layer: 1, pos: 11
type: A, layer: 1, pos: 15
type: A, layer: 1, pos: 31
type: B, layer: 1, pos: 31
type: B, layer: 1, pos: 42
type: A, layer: 1, pos: 42
type: A, layer: 1, pos: 39
type: B, layer: 1, pos: 15
type: B, layer: 1, pos: 16

Time for candidate selection: 0.07 seconds

### Candidate
type: A, layer: 1, pos: 41

### Candidate
type: B, layer: 1, pos: 41

### Candidate
type: A, layer: 1, pos: 25

### Candidate
type: B, layer: 1, pos: 25

### Candidate
type: B, layer: 1, pos: 12

### Candidate
type: B, layer: 1, pos: 19

### Candidate
type: B, layer: 1, pos: 4

### Candidate
type: B, layer: 1, pos: 14

### Candidate
type: A, layer: 1, pos: 14

### Candidate
type: B, layer: 1, pos: 26

### Candidate
type: A, layer: 1, pos: 12

### Candidate
type: B, layer: 1, pos: 29

### Candidate
type: A, layer: 1, pos: 26

### Candidate
type: A, layer: 1, pos: 19

### Candidate
type: A, layer: 1, pos: 23

### Candidate
type: B, layer: 1, pos: 37

## Relational analysis of NS_A2_A1_B2_A2_A2_B2_A2_B1
Optimization infeasible because this subproblem isn't reachable.

### Candidate
type: A, layer: 1, pos: 29

### Candidate
type: B, layer: 1, pos: 22

## Relational analysis of NS_A2_A1_B2_A2_A2_B2_A2_B1

### Relational analysis result of NS_A2_A1_B2_A2_A2_B2_A2_B1
Status: Status.UNKNOWN
Output dim: 0, lower bound: -81.8605364, upper bound: 81.8611001
time: 0.88 seconds

## Relational analysis of NS_A2_A1_B2_A2_A2_B2_A2_B2

### Relational analysis result of NS_A2_A1_B2_A2_A2_B2_A2_B2
Status: Status.UNKNOWN
Output dim: 0, lower bound: -81.8604938, upper bound: 81.8610934
time: 0.96 seconds

## Summary of splitting at layer (split count: 7)
- Time for NS candidates: 3.03 seconds
NS_A2_A1_B1_A2_B1_A1_B1_A1, status: Status.UNKNOWN, split count: 8, time: 3.03
Output dim: 0, lower bound: -81.8605141, upper bound: 81.8611964
NS_A2_A1_B1_A2_B1_A1_B1_A2, status: Status.UNKNOWN, split count: 8, time: 3.03
Output dim: 0, lower bound: -81.8604268, upper bound: 81.8610514
NS_A2_A1_B1_A2_B1_A1_B2_B1, status: Status.UNKNOWN, split count: 8, time: 3.03
Output dim: 0, lower bound: -81.8605715, upper bound: 81.8611176
NS_A2_A1_B1_A2_B1_A1_B2_B2, status: Status.UNKNOWN, split count: 8, time: 3.03
Output dim: 0, lower bound: -81.8604268, upper bound: 81.8610514
NS_A2_A1_B1_A2_B1_A2_A1_B1, status: Status.UNKNOWN, split count: 8, time: 3.03
Output dim: 0, lower bound: -81.8605726, upper bound: 81.8611340
NS_A2_A1_B1_A2_B1_A2_A1_B2, status: Status.UNKNOWN, split count: 8, time: 3.03
Output dim: 0, lower bound: -81.8604279, upper bound: 81.8610697
NS_A2_A1_B1_A2_B1_A2_A2_B1, status: Status.UNKNOWN, split count: 8, time: 3.03
Output dim: 0, lower bound: -81.8607932, upper bound: 81.8614029
NS_A2_A1_B1_A2_B1_A2_A2_B2, status: Status.UNKNOWN, split count: 8, time: 3.03
Output dim: 0, lower bound: -81.8607932, upper bound: 81.8614029
NS_A2_A1_B1_A2_B2_A1_B1_B1, status: Status.UNKNOWN, split count: 8, time: 3.03
Output dim: 0, lower bound: -81.8606652, upper bound: 81.8614183
NS_A2_A1_B1_A2_B2_A1_B1_B2, status: Status.UNKNOWN, split count: 8, time: 3.03
Output dim: 0, lower bound: -81.8610575, upper bound: 81.8615892
NS_A2_A1_B1_A2_B2_A1_B2_B1, status: Status.UNKNOWN, split count: 8, time: 3.03
Output dim: 0, lower bound: -81.8605715, upper bound: 81.8611810
NS_A2_A1_B1_A2_B2_A1_B2_B2, status: Status.UNKNOWN, split count: 8, time: 3.03
Output dim: 0, lower bound: -81.8605486, upper bound: 81.8611519
NS_A2_A1_B1_A2_B2_A2_B1_B1, status: Status.UNKNOWN, split count: 8, time: 3.03
Output dim: 0, lower bound: -81.8609240, upper bound: 81.8614205
NS_A2_A1_B1_A2_B2_A2_B1_B2, status: Status.UNKNOWN, split count: 8, time: 3.03
Output dim: 0, lower bound: -81.8608459, upper bound: 81.8614131
NS_A2_A1_B1_A2_B2_A2_B2_A1, status: Status.UNKNOWN, split count: 8, time: 3.03
Output dim: 0, lower bound: -81.8610381, upper bound: 81.8615567
NS_A2_A1_B1_A2_B2_A2_B2_A2, status: Status.UNKNOWN, split count: 8, time: 3.03
Output dim: 0, lower bound: -81.8610429, upper bound: 81.8615547
NS_A2_A1_B2_A2_A2_B1_A2_B1, status: Status.UNKNOWN, split count: 8, time: 3.03
Output dim: 0, lower bound: -81.8632979, upper bound: 81.8633526
NS_A2_A1_B2_A2_A2_B1_A2_B2, status: Status.UNKNOWN, split count: 8, time: 3.03
Output dim: 0, lower bound: -81.8633128, upper bound: 81.8633536
NS_A2_A1_B2_A2_A2_B2_A1_B1, status: Status.UNKNOWN, split count: 8, time: 3.03
Output dim: 0, lower bound: -81.8605982, upper bound: 81.8611741
NS_A2_A1_B2_A2_A2_B2_A1_B2, status: Status.UNKNOWN, split count: 8, time: 3.03
Output dim: 0, lower bound: -81.8607351, upper bound: 81.8612842
NS_A2_A1_B2_A2_A2_B2_A2_B1, status: Status.UNKNOWN, split count: 8, time: 3.03
Output dim: 0, lower bound: -81.8605364, upper bound: 81.8611001
NS_A2_A1_B2_A2_A2_B2_A2_B2, status: Status.UNKNOWN, split count: 8, time: 3.03
Output dim: 0, lower bound: -81.8604938, upper bound: 81.8610934

## BFS NS instance: NS_A2_A1_B1_A2_B1_A1_B1_A1

### Backsubstitution after applying NS history:
0: -19.2080135, 70.1533127, -18.9485092, 69.1741562, -88.3821640, 89.1018219
1: -51.0122948, 158.1988220, -50.4105797, 156.0391998, -207.0514984, 208.6093903
2: -76.5601349, 140.7829285, -75.9221573, 138.5529175, -215.1129761, 216.7050781
3: -43.8815269, 169.0470276, -43.3704910, 166.8921967, -210.7737122, 212.4175110
4: -69.9408417, 124.0000458, -69.3036575, 122.1093903, -192.0501862, 193.3037109

Time for backsubstitution: 0.99 seconds

### NS candidates at layer 1
type: B, layer: 1, pos: 41
type: A, layer: 1, pos: 41
type: B, layer: 1, pos: 20
type: B, layer: 1, pos: 39
type: A, layer: 1, pos: 20
type: A, layer: 1, pos: 39
type: B, layer: 1, pos: 19
type: A, layer: 1, pos: 19
type: B, layer: 1, pos: 14
type: B, layer: 1, pos: 25
type: A, layer: 1, pos: 25
type: A, layer: 1, pos: 14
type: B, layer: 1, pos: 23
type: A, layer: 1, pos: 23
type: B, layer: 1, pos: 21
type: B, layer: 1, pos: 6
type: A, layer: 1, pos: 6
type: B, layer: 1, pos: 4
type: A, layer: 1, pos: 21
type: A, layer: 1, pos: 46
type: B, layer: 1, pos: 13
type: A, layer: 1, pos: 4
type: A, layer: 1, pos: 15
type: A, layer: 1, pos: 29
type: B, layer: 1, pos: 46
type: A, layer: 1, pos: 38
type: B, layer: 1, pos: 15
type: B, layer: 1, pos: 16
type: A, layer: 1, pos: 16
type: B, layer: 1, pos: 29
type: B, layer: 1, pos: 38
type: A, layer: 1, pos: 37
type: B, layer: 1, pos: 37
type: A, layer: 1, pos: 12
type: A, layer: 1, pos: 11
type: B, layer: 1, pos: 12
type: B, layer: 1, pos: 42
type: B, layer: 1, pos: 22
type: A, layer: 1, pos: 42
type: A, layer: 1, pos: 31
type: B, layer: 1, pos: 11
type: B, layer: 1, pos: 31
type: A, layer: 1, pos: 8
type: B, layer: 1, pos: 8

Time for candidate selection: 0.07 seconds

### Candidate
type: B, layer: 1, pos: 41

### Candidate
type: A, layer: 1, pos: 41

### Candidate
type: B, layer: 1, pos: 20

### Candidate
type: B, layer: 1, pos: 39

### Candidate
type: A, layer: 1, pos: 20

### Candidate
type: A, layer: 1, pos: 39

### Candidate
type: B, layer: 1, pos: 19

### Candidate
type: A, layer: 1, pos: 19

### Candidate
type: B, layer: 1, pos: 14

### Candidate
type: B, layer: 1, pos: 25

### Candidate
type: A, layer: 1, pos: 25

### Candidate
type: A, layer: 1, pos: 14

### Candidate
type: B, layer: 1, pos: 23

### Candidate
type: A, layer: 1, pos: 23

### Candidate
type: B, layer: 1, pos: 21

### Candidate
type: B, layer: 1, pos: 6

### Candidate
type: A, layer: 1, pos: 6

### Candidate
type: B, layer: 1, pos: 4

### Candidate
type: A, layer: 1, pos: 21

### Candidate
type: A, layer: 1, pos: 46

### Candidate
type: B, layer: 1, pos: 13

### Candidate
type: A, layer: 1, pos: 4

### Candidate
type: A, layer: 1, pos: 15

### Candidate
type: A, layer: 1, pos: 29

### Candidate
type: B, layer: 1, pos: 46

### Candidate
type: A, layer: 1, pos: 38

### Candidate
type: B, layer: 1, pos: 15

### Candidate
type: B, layer: 1, pos: 16

### Candidate
type: A, layer: 1, pos: 16

### Candidate
type: B, layer: 1, pos: 29

### Candidate
type: B, layer: 1, pos: 38

### Candidate
type: A, layer: 1, pos: 37

### Candidate
type: B, layer: 1, pos: 37

### Candidate
type: A, layer: 1, pos: 12

### Candidate
type: A, layer: 1, pos: 11

### Candidate
type: B, layer: 1, pos: 12

### Candidate
type: B, layer: 1, pos: 42

## Relational analysis of NS_A2_A1_B1_A2_B1_A1_B1_A1_B1

### Relational analysis result of NS_A2_A1_B1_A2_B1_A1_B1_A1_B1
Status: Status.UNKNOWN
Output dim: 0, lower bound: -81.8628103, upper bound: 81.8629251
time: 0.98 seconds

## Relational analysis of NS_A2_A1_B1_A2_B1_A1_B1_A1_B2
Optimization infeasible because this subproblem isn't reachable.

### Candidate
type: B, layer: 1, pos: 22

## Relational analysis of NS_A2_A1_B1_A2_B1_A1_B1_A1_B1

### Relational analysis result of NS_A2_A1_B1_A2_B1_A1_B1_A1_B1
Status: Status.UNKNOWN
Output dim: 0, lower bound: -81.8629221, upper bound: 81.8629398
time: 1.49 seconds

## Relational analysis of NS_A2_A1_B1_A2_B1_A1_B1_A1_B2

### Relational analysis result of NS_A2_A1_B1_A2_B1_A1_B1_A1_B2
Status: Status.UNKNOWN
Output dim: 0, lower bound: -81.8629221, upper bound: 81.8629398
time: 0.97 seconds

## BFS NS instance: NS_A2_A1_B1_A2_B1_A1_B1_A2

### Backsubstitution after applying NS history:
0: -19.3761635, 70.6416855, -18.6662045, 68.1297913, -87.5059509, 89.3078918
1: -51.1780510, 160.0721283, -49.6206551, 153.7447510, -204.9227905, 209.6927490
2: -76.4557343, 142.3144379, -74.6735458, 136.5794830, -213.0352173, 216.9879761
3: -44.0854263, 170.8686676, -42.6957550, 164.4387970, -208.5242310, 213.5644073
4: -69.9495621, 125.3343735, -68.1795273, 120.3657761, -190.3153381, 193.5138702

Time for backsubstitution: 0.84 seconds

### NS candidates at layer 1
type: A, layer: 1, pos: 41
type: B, layer: 1, pos: 41
type: B, layer: 1, pos: 19
type: A, layer: 1, pos: 19
type: B, layer: 1, pos: 6
type: A, layer: 1, pos: 6
type: B, layer: 1, pos: 21
type: A, layer: 1, pos: 25
type: B, layer: 1, pos: 25
type: A, layer: 1, pos: 46
type: A, layer: 1, pos: 21
type: B, layer: 1, pos: 46
type: B, layer: 1, pos: 4
type: B, layer: 1, pos: 38
type: A, layer: 1, pos: 38
type: B, layer: 1, pos: 14
type: A, layer: 1, pos: 4
type: B, layer: 1, pos: 13
type: B, layer: 1, pos: 12
type: A, layer: 1, pos: 12
type: B, layer: 1, pos: 29
type: A, layer: 1, pos: 14
type: A, layer: 1, pos: 29
type: B, layer: 1, pos: 39
type: A, layer: 1, pos: 31
type: B, layer: 1, pos: 31
type: A, layer: 1, pos: 11
type: A, layer: 1, pos: 15
type: B, layer: 1, pos: 11
type: A, layer: 1, pos: 16
type: B, layer: 1, pos: 22
type: B, layer: 1, pos: 20
type: B, layer: 1, pos: 16
type: A, layer: 1, pos: 39
type: B, layer: 1, pos: 15
type: A, layer: 1, pos: 20
type: A, layer: 1, pos: 23
type: B, layer: 1, pos: 8
type: B, layer: 1, pos: 37
type: B, layer: 1, pos: 42
type: A, layer: 1, pos: 8
type: B, layer: 1, pos: 23
type: A, layer: 1, pos: 42
type: A, layer: 1, pos: 37

Time for candidate selection: 0.09 seconds

### Candidate
type: A, layer: 1, pos: 41

### Candidate
type: B, layer: 1, pos: 41

### Candidate
type: B, layer: 1, pos: 19

### Candidate
type: A, layer: 1, pos: 19

### Candidate
type: B, layer: 1, pos: 6

### Candidate
type: A, layer: 1, pos: 6

### Candidate
type: B, layer: 1, pos: 21

### Candidate
type: A, layer: 1, pos: 25

### Candidate
type: B, layer: 1, pos: 25

### Candidate
type: A, layer: 1, pos: 46

### Candidate
type: A, layer: 1, pos: 21

### Candidate
type: B, layer: 1, pos: 46

### Candidate
type: B, layer: 1, pos: 4

### Candidate
type: B, layer: 1, pos: 38

### Candidate
type: A, layer: 1, pos: 38

### Candidate
type: B, layer: 1, pos: 14

### Candidate
type: A, layer: 1, pos: 4

### Candidate
type: B, layer: 1, pos: 13

### Candidate
type: B, layer: 1, pos: 12

### Candidate
type: A, layer: 1, pos: 12

### Candidate
type: B, layer: 1, pos: 29

### Candidate
type: A, layer: 1, pos: 14

### Candidate
type: A, layer: 1, pos: 29

### Candidate
type: B, layer: 1, pos: 39

### Candidate
type: A, layer: 1, pos: 31

### Candidate
type: B, layer: 1, pos: 31

### Candidate
type: A, layer: 1, pos: 11

### Candidate
type: A, layer: 1, pos: 15

### Candidate
type: B, layer: 1, pos: 11

### Candidate
type: A, layer: 1, pos: 16

### Candidate
type: B, layer: 1, pos: 22

## Relational analysis of NS_A2_A1_B1_A2_B1_A1_B1_A2_B1

### Relational analysis result of NS_A2_A1_B1_A2_B1_A1_B1_A2_B1
Status: Status.UNKNOWN
Output dim: 0, lower bound: -81.8629221, upper bound: 81.8629398
time: 1.13 seconds

## Relational analysis of NS_A2_A1_B1_A2_B1_A1_B1_A2_B2

### Relational analysis result of NS_A2_A1_B1_A2_B1_A1_B1_A2_B2
Status: Status.UNKNOWN
Output dim: 0, lower bound: -81.8629221, upper bound: 81.8629399
time: 1.11 seconds

## BFS NS instance: NS_A2_A1_B1_A2_B1_A1_B2_B1

### Backsubstitution after applying NS history:
0: -19.5514183, 71.4037170, -18.6935673, 68.1446991, -87.6961136, 90.0972824
1: -51.9093094, 161.0701752, -49.8248100, 153.5952301, -205.5045166, 210.8949738
2: -77.8657913, 143.3444519, -75.1386414, 136.5925293, -214.4583130, 218.4830933
3: -44.6583481, 172.0934601, -42.8763771, 164.5938721, -209.2522125, 214.9698334
4: -71.1484833, 126.2585678, -68.5641785, 120.4410019, -191.5894775, 194.8227539

Time for backsubstitution: 0.92 seconds

### NS candidates at layer 1
type: A, layer: 1, pos: 41
type: A, layer: 1, pos: 14
type: B, layer: 1, pos: 14
type: A, layer: 1, pos: 19
type: A, layer: 1, pos: 39
type: B, layer: 1, pos: 19
type: A, layer: 1, pos: 25
type: B, layer: 1, pos: 12
type: B, layer: 1, pos: 39
type: B, layer: 1, pos: 25
type: A, layer: 1, pos: 12
type: A, layer: 1, pos: 8
type: B, layer: 1, pos: 37
type: A, layer: 1, pos: 20
type: B, layer: 1, pos: 29
type: A, layer: 1, pos: 37
type: B, layer: 1, pos: 20
type: B, layer: 1, pos: 8
type: B, layer: 1, pos: 41
type: A, layer: 1, pos: 29
type: A, layer: 1, pos: 11
type: A, layer: 1, pos: 21
type: A, layer: 1, pos: 46
type: A, layer: 1, pos: 6
type: B, layer: 1, pos: 6
type: B, layer: 1, pos: 4
type: B, layer: 1, pos: 21
type: A, layer: 1, pos: 15
type: B, layer: 1, pos: 42
type: A, layer: 1, pos: 4
type: B, layer: 1, pos: 13
type: B, layer: 1, pos: 46
type: B, layer: 1, pos: 15
type: A, layer: 1, pos: 38
type: A, layer: 1, pos: 22
type: B, layer: 1, pos: 23
type: A, layer: 1, pos: 23
type: B, layer: 1, pos: 38
type: B, layer: 1, pos: 16
type: A, layer: 1, pos: 16
type: A, layer: 1, pos: 31
type: A, layer: 1, pos: 42
type: B, layer: 1, pos: 31
type: B, layer: 1, pos: 11

Time for candidate selection: 0.07 seconds

### Candidate
type: A, layer: 1, pos: 41

### Candidate
type: A, layer: 1, pos: 14

### Candidate
type: B, layer: 1, pos: 14

### Candidate
type: A, layer: 1, pos: 19

### Candidate
type: A, layer: 1, pos: 39

## Relational analysis of NS_A2_A1_B1_A2_B1_A1_B2_B1_A1
Optimization infeasible because this subproblem isn't reachable.

### Candidate
type: B, layer: 1, pos: 19

### Candidate
type: A, layer: 1, pos: 25

### Candidate
type: B, layer: 1, pos: 12

### Candidate
type: B, layer: 1, pos: 39

### Candidate
type: B, layer: 1, pos: 25

### Candidate
type: A, layer: 1, pos: 12

### Candidate
type: A, layer: 1, pos: 8

### Candidate
type: B, layer: 1, pos: 37

### Candidate
type: A, layer: 1, pos: 20

## Relational analysis of NS_A2_A1_B1_A2_B1_A1_B2_B1_A1
Optimization infeasible because this subproblem isn't reachable.

### Candidate
type: B, layer: 1, pos: 29

### Candidate
type: A, layer: 1, pos: 37

### Candidate
type: B, layer: 1, pos: 20

### Candidate
type: B, layer: 1, pos: 8

### Candidate
type: B, layer: 1, pos: 41

### Candidate
type: A, layer: 1, pos: 29

### Candidate
type: A, layer: 1, pos: 11

### Candidate
type: A, layer: 1, pos: 21

### Candidate
type: A, layer: 1, pos: 46

### Candidate
type: A, layer: 1, pos: 6

### Candidate
type: B, layer: 1, pos: 6

### Candidate
type: B, layer: 1, pos: 4

### Candidate
type: B, layer: 1, pos: 21

### Candidate
type: A, layer: 1, pos: 15

## Relational analysis of NS_A2_A1_B1_A2_B1_A1_B2_B1_A1
Optimization infeasible because this subproblem isn't reachable.

### Candidate
type: B, layer: 1, pos: 42

## Relational analysis of NS_A2_A1_B1_A2_B1_A1_B2_B1_B1

### Relational analysis result of NS_A2_A1_B1_A2_B1_A1_B2_B1_B1
Status: Status.UNKNOWN
Output dim: 0, lower bound: -81.8604385, upper bound: 81.8610384
time: 0.73 seconds

## Relational analysis of NS_A2_A1_B1_A2_B1_A1_B2_B1_B2

### Relational analysis result of NS_A2_A1_B1_A2_B1_A1_B2_B1_B2
Status: Status.UNKNOWN
Output dim: 0, lower bound: -81.8605636, upper bound: 81.8611060
time: 0.71 seconds

## BFS NS instance: NS_A2_A1_B1_A2_B1_A1_B2_B2

### Backsubstitution after applying NS history:
0: -19.2612724, 70.3088608, -18.7856750, 68.4834824, -87.7447510, 89.0945358
1: -51.0843353, 158.7248383, -49.8460236, 154.8673096, -205.9516449, 208.5708618
2: -76.5974350, 141.2996521, -74.7570572, 137.6546631, -214.2521057, 216.0567017
3: -43.9664841, 169.5800476, -42.9090462, 165.7720337, -209.7384949, 212.4890900
4: -70.0057755, 124.4529114, -68.2975235, 121.3659363, -191.3717041, 192.7504272

Time for backsubstitution: 0.87 seconds

### NS candidates at layer 1
type: A, layer: 1, pos: 41
type: A, layer: 1, pos: 14
type: B, layer: 1, pos: 14
type: A, layer: 1, pos: 39
type: A, layer: 1, pos: 19
type: B, layer: 1, pos: 19
type: A, layer: 1, pos: 25
type: B, layer: 1, pos: 12
type: B, layer: 1, pos: 39
type: B, layer: 1, pos: 25
type: A, layer: 1, pos: 12
type: A, layer: 1, pos: 8
type: A, layer: 1, pos: 37
type: B, layer: 1, pos: 37
type: A, layer: 1, pos: 20
type: B, layer: 1, pos: 29
type: B, layer: 1, pos: 20
type: A, layer: 1, pos: 29
type: B, layer: 1, pos: 41
type: B, layer: 1, pos: 8
type: A, layer: 1, pos: 21
type: A, layer: 1, pos: 11
type: A, layer: 1, pos: 6
type: A, layer: 1, pos: 46
type: B, layer: 1, pos: 6
type: B, layer: 1, pos: 4
type: B, layer: 1, pos: 15
type: B, layer: 1, pos: 42
type: A, layer: 1, pos: 15
type: B, layer: 1, pos: 21
type: A, layer: 1, pos: 4
type: B, layer: 1, pos: 46
type: A, layer: 1, pos: 38
type: B, layer: 1, pos: 13
type: B, layer: 1, pos: 23
type: B, layer: 1, pos: 16
type: A, layer: 1, pos: 22
type: A, layer: 1, pos: 23
type: B, layer: 1, pos: 38
type: A, layer: 1, pos: 16
type: A, layer: 1, pos: 42
type: A, layer: 1, pos: 31
type: B, layer: 1, pos: 31
type: B, layer: 1, pos: 11

Time for candidate selection: 0.07 seconds

### Candidate
type: A, layer: 1, pos: 41

### Candidate
type: A, layer: 1, pos: 14

### Candidate
type: B, layer: 1, pos: 14

### Candidate
type: A, layer: 1, pos: 39

## Relational analysis of NS_A2_A1_B1_A2_B1_A1_B2_B2_A1
Optimization infeasible because this subproblem isn't reachable.

### Candidate
type: A, layer: 1, pos: 19

### Candidate
type: B, layer: 1, pos: 19

### Candidate
type: A, layer: 1, pos: 25

### Candidate
type: B, layer: 1, pos: 12

### Candidate
type: B, layer: 1, pos: 39

### Candidate
type: B, layer: 1, pos: 25

### Candidate
type: A, layer: 1, pos: 12

### Candidate
type: A, layer: 1, pos: 8

### Candidate
type: A, layer: 1, pos: 37

### Candidate
type: B, layer: 1, pos: 37

### Candidate
type: A, layer: 1, pos: 20

## Relational analysis of NS_A2_A1_B1_A2_B1_A1_B2_B2_A1
Optimization infeasible because this subproblem isn't reachable.

### Candidate
type: B, layer: 1, pos: 29

### Candidate
type: B, layer: 1, pos: 20

### Candidate
type: A, layer: 1, pos: 29

### Candidate
type: B, layer: 1, pos: 41

### Candidate
type: B, layer: 1, pos: 8

### Candidate
type: A, layer: 1, pos: 21

### Candidate
type: A, layer: 1, pos: 11

### Candidate
type: A, layer: 1, pos: 6

### Candidate
type: A, layer: 1, pos: 46

### Candidate
type: B, layer: 1, pos: 6

### Candidate
type: B, layer: 1, pos: 4

### Candidate
type: B, layer: 1, pos: 15

## Relational analysis of NS_A2_A1_B1_A2_B1_A1_B2_B2_B1
Optimization infeasible because this subproblem isn't reachable.

### Candidate
type: B, layer: 1, pos: 42

## Relational analysis of NS_A2_A1_B1_A2_B1_A1_B2_B2_B1

### Relational analysis result of NS_A2_A1_B1_A2_B1_A1_B2_B2_B1
Status: Status.UNKNOWN
Output dim: 0, lower bound: -81.8604233, upper bound: 81.8610251
time: 0.68 seconds

## Relational analysis of NS_A2_A1_B1_A2_B1_A1_B2_B2_B2
Optimization infeasible because this subproblem isn't reachable.

### Candidate
type: A, layer: 1, pos: 15

### Candidate
type: B, layer: 1, pos: 21

### Candidate
type: A, layer: 1, pos: 4

### Candidate
type: B, layer: 1, pos: 46

### Candidate
type: A, layer: 1, pos: 38

### Candidate
type: B, layer: 1, pos: 13

### Candidate
type: B, layer: 1, pos: 23

### Candidate
type: B, layer: 1, pos: 16

## Relational analysis of NS_A2_A1_B1_A2_B1_A1_B2_B2_B1
Optimization infeasible because this subproblem isn't reachable.

### Candidate
type: A, layer: 1, pos: 22

## Relational analysis of NS_A2_A1_B1_A2_B1_A1_B2_B2_A1

### Relational analysis result of NS_A2_A1_B1_A2_B1_A1_B2_B2_A1
Status: Status.UNKNOWN
Output dim: 0, lower bound: -81.8604268, upper bound: 81.8610514
time: 0.76 seconds

## Relational analysis of NS_A2_A1_B1_A2_B1_A1_B2_B2_A2

### Relational analysis result of NS_A2_A1_B1_A2_B1_A1_B2_B2_A2
Status: Status.UNKNOWN
Output dim: 0, lower bound: -81.8604268, upper bound: 81.8610514
time: 0.76 seconds

## BFS NS instance: NS_A2_A1_B1_A2_B1_A2_A1_B1

### Backsubstitution after applying NS history:
0: -21.6661549, 79.1037064, -18.4410477, 67.3194046, -88.9855576, 97.5447311
1: -57.1756744, 180.2160187, -49.0906525, 151.7741394, -208.9497986, 229.3066711
2: -85.4210587, 159.3976746, -73.9905624, 134.7157288, -220.1367798, 233.3882141
3: -49.1567116, 192.1778412, -42.2283630, 162.3587189, -211.5154266, 234.4061890
4: -78.1009827, 140.3668823, -67.5197830, 118.7400513, -196.8410034, 207.8866272

Time for backsubstitution: 0.90 seconds

### NS candidates at layer 1
type: A, layer: 1, pos: 41
type: B, layer: 1, pos: 19
type: B, layer: 1, pos: 41
type: B, layer: 1, pos: 21
type: A, layer: 1, pos: 6
type: B, layer: 1, pos: 6
type: A, layer: 1, pos: 46
type: A, layer: 1, pos: 25
type: B, layer: 1, pos: 25
type: A, layer: 1, pos: 19
type: A, layer: 1, pos: 11
type: A, layer: 1, pos: 4
type: A, layer: 1, pos: 38
type: B, layer: 1, pos: 14
type: A, layer: 1, pos: 12
type: B, layer: 1, pos: 12
type: B, layer: 1, pos: 38
type: B, layer: 1, pos: 13
type: A, layer: 1, pos: 21
type: B, layer: 1, pos: 4
type: B, layer: 1, pos: 29
type: A, layer: 1, pos: 29
type: B, layer: 1, pos: 39
type: B, layer: 1, pos: 20
type: B, layer: 1, pos: 46
type: A, layer: 1, pos: 31
type: A, layer: 1, pos: 15
type: B, layer: 1, pos: 31
type: A, layer: 1, pos: 14
type: B, layer: 1, pos: 23
type: B, layer: 1, pos: 16
type: B, layer: 1, pos: 37
type: B, layer: 1, pos: 43
type: A, layer: 1, pos: 39
type: B, layer: 1, pos: 15
type: A, layer: 1, pos: 8
type: A, layer: 1, pos: 42
type: B, layer: 1, pos: 8
type: A, layer: 1, pos: 20
type: A, layer: 1, pos: 23
type: A, layer: 1, pos: 22
type: B, layer: 1, pos: 42
type: A, layer: 1, pos: 37
type: B, layer: 1, pos: 11

Time for candidate selection: 0.07 seconds

### Candidate
type: A, layer: 1, pos: 41

### Candidate
type: B, layer: 1, pos: 19

### Candidate
type: B, layer: 1, pos: 41

### Candidate
type: B, layer: 1, pos: 21

### Candidate
type: A, layer: 1, pos: 6

### Candidate
type: B, layer: 1, pos: 6

### Candidate
type: A, layer: 1, pos: 46

### Candidate
type: A, layer: 1, pos: 25

### Candidate
type: B, layer: 1, pos: 25

### Candidate
type: A, layer: 1, pos: 19

### Candidate
type: A, layer: 1, pos: 11

### Candidate
type: A, layer: 1, pos: 4

### Candidate
type: A, layer: 1, pos: 38

### Candidate
type: B, layer: 1, pos: 14

### Candidate
type: A, layer: 1, pos: 12

### Candidate
type: B, layer: 1, pos: 12

### Candidate
type: B, layer: 1, pos: 38

### Candidate
type: B, layer: 1, pos: 13

### Candidate
type: A, layer: 1, pos: 21

### Candidate
type: B, layer: 1, pos: 4

### Candidate
type: B, layer: 1, pos: 29

### Candidate
type: A, layer: 1, pos: 29

### Candidate
type: B, layer: 1, pos: 39

### Candidate
type: B, layer: 1, pos: 20

### Candidate
type: B, layer: 1, pos: 46

### Candidate
type: A, layer: 1, pos: 31

### Candidate
type: A, layer: 1, pos: 15

### Candidate
type: B, layer: 1, pos: 31

### Candidate
type: A, layer: 1, pos: 14

### Candidate
type: B, layer: 1, pos: 23

### Candidate
type: B, layer: 1, pos: 16

## Relational analysis of NS_A2_A1_B1_A2_B1_A2_A1_B1_B1
Optimization infeasible because this subproblem isn't reachable.

### Candidate
type: B, layer: 1, pos: 37

### Candidate
type: B, layer: 1, pos: 43

## Relational analysis of NS_A2_A1_B1_A2_B1_A2_A1_B1_B1

### Relational analysis result of NS_A2_A1_B1_A2_B1_A2_A1_B1_B1
Status: Status.UNKNOWN
Output dim: 0, lower bound: -81.8605726, upper bound: 81.8611340
time: 0.78 seconds

## Relational analysis of NS_A2_A1_B1_A2_B1_A2_A1_B1_B2

### Relational analysis result of NS_A2_A1_B1_A2_B1_A2_A1_B1_B2
Status: Status.UNKNOWN
Output dim: 0, lower bound: -81.8605726, upper bound: 81.8611340
time: 0.89 seconds

## BFS NS instance: NS_A2_A1_B1_A2_B1_A2_A1_B2

### Backsubstitution after applying NS history:
0: -21.2976780, 77.7213821, -18.6291218, 67.9447556, -89.2424316, 96.3505020
1: -56.1462212, 177.2727966, -49.3322220, 153.7995758, -209.9458008, 226.6050110
2: -83.8302002, 156.6667328, -73.9622803, 136.4961243, -220.3263245, 230.6289673
3: -48.2794685, 189.0195312, -42.4831963, 164.3551941, -212.6346436, 231.5026550
4: -76.6548920, 137.9883423, -67.5930634, 120.2783661, -196.9332581, 205.5813446

Time for backsubstitution: 1.08 seconds

### NS candidates at layer 1
type: A, layer: 1, pos: 41
type: B, layer: 1, pos: 41
type: B, layer: 1, pos: 19
type: A, layer: 1, pos: 6
type: B, layer: 1, pos: 6
type: B, layer: 1, pos: 21
type: A, layer: 1, pos: 46
type: B, layer: 1, pos: 25
type: A, layer: 1, pos: 19
type: A, layer: 1, pos: 25
type: A, layer: 1, pos: 11
type: A, layer: 1, pos: 38
type: A, layer: 1, pos: 4
type: B, layer: 1, pos: 14
type: A, layer: 1, pos: 21
type: A, layer: 1, pos: 12
type: A, layer: 1, pos: 29
type: B, layer: 1, pos: 12
type: B, layer: 1, pos: 13
type: B, layer: 1, pos: 38
type: B, layer: 1, pos: 4
type: B, layer: 1, pos: 20
type: B, layer: 1, pos: 29
type: B, layer: 1, pos: 46
type: B, layer: 1, pos: 39
type: A, layer: 1, pos: 14
type: A, layer: 1, pos: 31
type: B, layer: 1, pos: 31
type: B, layer: 1, pos: 23
type: A, layer: 1, pos: 15
type: B, layer: 1, pos: 43
type: B, layer: 1, pos: 16
type: B, layer: 1, pos: 15
type: A, layer: 1, pos: 39
type: A, layer: 1, pos: 8
type: A, layer: 1, pos: 42
type: A, layer: 1, pos: 22
type: B, layer: 1, pos: 37
type: B, layer: 1, pos: 8
type: A, layer: 1, pos: 20
type: A, layer: 1, pos: 37
type: A, layer: 1, pos: 23
type: B, layer: 1, pos: 42
type: B, layer: 1, pos: 11

Time for candidate selection: 0.08 seconds

### Candidate
type: A, layer: 1, pos: 41

### Candidate
type: B, layer: 1, pos: 41

### Candidate
type: B, layer: 1, pos: 19

### Candidate
type: A, layer: 1, pos: 6

### Candidate
type: B, layer: 1, pos: 6

### Candidate
type: B, layer: 1, pos: 21

### Candidate
type: A, layer: 1, pos: 46

### Candidate
type: B, layer: 1, pos: 25

### Candidate
type: A, layer: 1, pos: 19

### Candidate
type: A, layer: 1, pos: 25

### Candidate
type: A, layer: 1, pos: 11

### Candidate
type: A, layer: 1, pos: 38

### Candidate
type: A, layer: 1, pos: 4

### Candidate
type: B, layer: 1, pos: 14

### Candidate
type: A, layer: 1, pos: 21

### Candidate
type: A, layer: 1, pos: 12

### Candidate
type: A, layer: 1, pos: 29

### Candidate
type: B, layer: 1, pos: 12

### Candidate
type: B, layer: 1, pos: 13

### Candidate
type: B, layer: 1, pos: 38

### Candidate
type: B, layer: 1, pos: 4

### Candidate
type: B, layer: 1, pos: 20

### Candidate
type: B, layer: 1, pos: 29

### Candidate
type: B, layer: 1, pos: 46

### Candidate
type: B, layer: 1, pos: 39

### Candidate
type: A, layer: 1, pos: 14

### Candidate
type: A, layer: 1, pos: 31

### Candidate
type: B, layer: 1, pos: 31

### Candidate
type: B, layer: 1, pos: 23

### Candidate
type: A, layer: 1, pos: 15

### Candidate
type: B, layer: 1, pos: 43

## Relational analysis of NS_A2_A1_B1_A2_B1_A2_A1_B2_B1

### Relational analysis result of NS_A2_A1_B1_A2_B1_A2_A1_B2_B1
Status: Status.UNKNOWN
Output dim: 0, lower bound: -81.8603972, upper bound: 81.8610697
time: 0.90 seconds

## Relational analysis of NS_A2_A1_B1_A2_B1_A2_A1_B2_B2

### Relational analysis result of NS_A2_A1_B1_A2_B1_A2_A1_B2_B2
Status: Status.UNKNOWN
Output dim: 0, lower bound: -81.8603972, upper bound: 81.8610697
time: 0.94 seconds

## BFS NS instance: NS_A2_A1_B1_A2_B1_A2_A2_B1

### Backsubstitution after applying NS history:
0: -22.6331539, 82.6661758, -18.8255348, 68.7267151, -91.3598709, 101.4917145
1: -59.7047958, 188.2940369, -50.0805817, 155.0319061, -214.7366943, 238.3746033
2: -89.2544403, 166.6350403, -75.4192276, 137.6687622, -226.9232025, 242.0542297
3: -51.3380013, 200.7817230, -43.0847549, 165.8019409, -217.1399384, 243.8664398
4: -81.5932159, 146.7880249, -68.8445129, 121.3269653, -202.9201508, 215.6325226

Time for backsubstitution: 0.91 seconds

### NS candidates at layer 1
type: A, layer: 1, pos: 41
type: B, layer: 1, pos: 41
type: B, layer: 1, pos: 19
type: B, layer: 1, pos: 21
type: A, layer: 1, pos: 6
type: B, layer: 1, pos: 6
type: A, layer: 1, pos: 19
type: A, layer: 1, pos: 46
type: A, layer: 1, pos: 25
type: B, layer: 1, pos: 25
type: A, layer: 1, pos: 38
type: A, layer: 1, pos: 4
type: B, layer: 1, pos: 14
type: A, layer: 1, pos: 11
type: B, layer: 1, pos: 16
type: B, layer: 1, pos: 4
type: A, layer: 1, pos: 21
type: B, layer: 1, pos: 12
type: B, layer: 1, pos: 20
type: B, layer: 1, pos: 13
type: B, layer: 1, pos: 39
type: A, layer: 1, pos: 12
type: A, layer: 1, pos: 29
type: B, layer: 1, pos: 38
type: B, layer: 1, pos: 29
type: B, layer: 1, pos: 46
type: B, layer: 1, pos: 31
type: A, layer: 1, pos: 31
type: A, layer: 1, pos: 14
type: B, layer: 1, pos: 15
type: A, layer: 1, pos: 15
type: B, layer: 1, pos: 37
type: B, layer: 1, pos: 23
type: A, layer: 1, pos: 8
type: A, layer: 1, pos: 23
type: A, layer: 1, pos: 42
type: A, layer: 1, pos: 39
type: B, layer: 1, pos: 8
type: B, layer: 1, pos: 11
type: B, layer: 1, pos: 22
type: A, layer: 1, pos: 22
type: B, layer: 1, pos: 42
type: A, layer: 1, pos: 37
type: A, layer: 1, pos: 20

Time for candidate selection: 0.07 seconds

### Candidate
type: A, layer: 1, pos: 41

### Candidate
type: B, layer: 1, pos: 41

### Candidate
type: B, layer: 1, pos: 19

### Candidate
type: B, layer: 1, pos: 21

### Candidate
type: A, layer: 1, pos: 6

### Candidate
type: B, layer: 1, pos: 6

### Candidate
type: A, layer: 1, pos: 19

### Candidate
type: A, layer: 1, pos: 46

### Candidate
type: A, layer: 1, pos: 25

### Candidate
type: B, layer: 1, pos: 25

### Candidate
type: A, layer: 1, pos: 38

### Candidate
type: A, layer: 1, pos: 4

### Candidate
type: B, layer: 1, pos: 14

### Candidate
type: A, layer: 1, pos: 11

### Candidate
type: B, layer: 1, pos: 16

### Candidate
type: B, layer: 1, pos: 4

### Candidate
type: A, layer: 1, pos: 21

### Candidate
type: B, layer: 1, pos: 12

### Candidate
type: B, layer: 1, pos: 20

### Candidate
type: B, layer: 1, pos: 13

### Candidate
type: B, layer: 1, pos: 39

### Candidate
type: A, layer: 1, pos: 12

### Candidate
type: A, layer: 1, pos: 29

### Candidate
type: B, layer: 1, pos: 38

### Candidate
type: B, layer: 1, pos: 29

### Candidate
type: B, layer: 1, pos: 46

### Candidate
type: B, layer: 1, pos: 31

### Candidate
type: A, layer: 1, pos: 31

### Candidate
type: A, layer: 1, pos: 14

### Candidate
type: B, layer: 1, pos: 15

### Candidate
type: A, layer: 1, pos: 15

### Candidate
type: B, layer: 1, pos: 37

## Relational analysis of NS_A2_A1_B1_A2_B1_A2_A2_B1_B1
Optimization infeasible because this subproblem isn't reachable.

### Candidate
type: B, layer: 1, pos: 23

## Relational analysis of NS_A2_A1_B1_A2_B1_A2_A2_B1_B1
Optimization infeasible because this subproblem isn't reachable.

### Candidate
type: A, layer: 1, pos: 8

## Relational analysis of NS_A2_A1_B1_A2_B1_A2_A2_B1_A1
Optimization infeasible because this subproblem isn't reachable.

### Candidate
type: A, layer: 1, pos: 23

### Candidate
type: A, layer: 1, pos: 42

## Relational analysis of NS_A2_A1_B1_A2_B1_A2_A2_B1_A1
Optimization infeasible because this subproblem isn't reachable.

### Candidate
type: A, layer: 1, pos: 39

### Candidate
type: B, layer: 1, pos: 8

### Candidate
type: B, layer: 1, pos: 11

### Candidate
type: B, layer: 1, pos: 22

## Relational analysis of NS_A2_A1_B1_A2_B1_A2_A2_B1_B1

### Relational analysis result of NS_A2_A1_B1_A2_B1_A2_A2_B1_B1
Status: Status.UNKNOWN
Output dim: 0, lower bound: -81.8605726, upper bound: 81.8611305
time: 1.35 seconds

## Relational analysis of NS_A2_A1_B1_A2_B1_A2_A2_B1_B2

### Relational analysis result of NS_A2_A1_B1_A2_B1_A2_A2_B1_B2
Status: Status.UNKNOWN
Output dim: 0, lower bound: -81.8604279, upper bound: 81.8610642
time: 1.08 seconds

## BFS NS instance: NS_A2_A1_B1_A2_B1_A2_A2_B2

### Backsubstitution after applying NS history:
0: -22.6331539, 82.6661758, -18.8893452, 68.8913422, -91.5244980, 101.5555191
1: -59.7047958, 188.2940369, -50.3433876, 155.2543488, -214.9591370, 238.6374207
2: -89.2544403, 166.6350403, -75.8517761, 138.1201935, -227.3746338, 242.4868164
3: -51.3380013, 200.7817230, -43.3109741, 166.3342133, -217.6722107, 244.0926971
4: -81.5932159, 146.7880249, -69.2284470, 121.7789001, -203.3721008, 216.0164795

Time for backsubstitution: 0.86 seconds

### NS candidates at layer 1
type: A, layer: 1, pos: 41
type: A, layer: 1, pos: 25
type: B, layer: 1, pos: 25
type: B, layer: 1, pos: 19
type: B, layer: 1, pos: 12
type: B, layer: 1, pos: 41
type: B, layer: 1, pos: 14
type: A, layer: 1, pos: 14
type: B, layer: 1, pos: 29
type: A, layer: 1, pos: 12
type: B, layer: 1, pos: 37
type: A, layer: 1, pos: 8
type: A, layer: 1, pos: 11
type: A, layer: 1, pos: 23
type: A, layer: 1, pos: 29
type: A, layer: 1, pos: 19
type: B, layer: 1, pos: 23
type: A, layer: 1, pos: 46
type: B, layer: 1, pos: 8
type: A, layer: 1, pos: 37
type: B, layer: 1, pos: 22
type: A, layer: 1, pos: 6
type: B, layer: 1, pos: 21
type: B, layer: 1, pos: 6
type: A, layer: 1, pos: 4
type: B, layer: 1, pos: 20
type: A, layer: 1, pos: 38
type: A, layer: 1, pos: 22
type: B, layer: 1, pos: 4
type: A, layer: 1, pos: 20
type: B, layer: 1, pos: 13
type: A, layer: 1, pos: 21
type: B, layer: 1, pos: 38
type: B, layer: 1, pos: 39
type: B, layer: 1, pos: 46
type: A, layer: 1, pos: 31
type: B, layer: 1, pos: 16
type: B, layer: 1, pos: 31
type: A, layer: 1, pos: 39
type: A, layer: 1, pos: 15
type: B, layer: 1, pos: 42
type: B, layer: 1, pos: 15
type: B, layer: 1, pos: 11
type: A, layer: 1, pos: 42

Time for candidate selection: 0.07 seconds

### Candidate
type: A, layer: 1, pos: 41

### Candidate
type: A, layer: 1, pos: 25

### Candidate
type: B, layer: 1, pos: 25

### Candidate
type: B, layer: 1, pos: 19

### Candidate
type: B, layer: 1, pos: 12

### Candidate
type: B, layer: 1, pos: 41

### Candidate
type: B, layer: 1, pos: 14

### Candidate
type: A, layer: 1, pos: 14

### Candidate
type: B, layer: 1, pos: 29

### Candidate
type: A, layer: 1, pos: 12

### Candidate
type: B, layer: 1, pos: 37

## Relational analysis of NS_A2_A1_B1_A2_B1_A2_A2_B2_B1
Optimization infeasible because this subproblem isn't reachable.

### Candidate
type: A, layer: 1, pos: 8

## Relational analysis of NS_A2_A1_B1_A2_B1_A2_A2_B2_A1
Optimization infeasible because this subproblem isn't reachable.

### Candidate
type: A, layer: 1, pos: 11

### Candidate
type: A, layer: 1, pos: 23

## Relational analysis of NS_A2_A1_B1_A2_B1_A2_A2_B2_A1
Optimization infeasible because this subproblem isn't reachable.

### Candidate
type: A, layer: 1, pos: 29

### Candidate
type: A, layer: 1, pos: 19

### Candidate
type: B, layer: 1, pos: 23

### Candidate
type: A, layer: 1, pos: 46

### Candidate
type: B, layer: 1, pos: 8

### Candidate
type: A, layer: 1, pos: 37

### Candidate
type: B, layer: 1, pos: 22

## Relational analysis of NS_A2_A1_B1_A2_B1_A2_A2_B2_B1

### Relational analysis result of NS_A2_A1_B1_A2_B1_A2_A2_B2_B1
Status: Status.UNKNOWN
Output dim: 0, lower bound: -81.8605726, upper bound: 81.8611305
time: 0.83 seconds

## Relational analysis of NS_A2_A1_B1_A2_B1_A2_A2_B2_B2

### Relational analysis result of NS_A2_A1_B1_A2_B1_A2_A2_B2_B2
Status: Status.UNKNOWN
Output dim: 0, lower bound: -81.8604279, upper bound: 81.8610642
time: 0.65 seconds

## BFS NS instance: NS_A2_A1_B1_A2_B2_A1_B1_B1

### Backsubstitution after applying NS history:
0: -19.2562885, 70.3274536, -22.7525654, 83.2576599, -102.5139465, 93.0800171
1: -51.1238251, 158.6177673, -60.1493530, 189.8360443, -240.9598694, 218.7671051
2: -76.6867676, 141.2121277, -90.1654434, 166.8709564, -243.5577087, 231.3775635
3: -43.9839020, 169.4877319, -51.7128868, 202.6529388, -246.6368408, 221.2006073
4: -70.0636826, 124.3861465, -82.3338470, 147.1033783, -217.1670380, 206.7199707

Time for backsubstitution: 0.84 seconds

### NS candidates at layer 1
type: B, layer: 1, pos: 41
type: A, layer: 1, pos: 19
type: A, layer: 1, pos: 41
type: A, layer: 1, pos: 6
type: B, layer: 1, pos: 6
type: A, layer: 1, pos: 21
type: B, layer: 1, pos: 25
type: B, layer: 1, pos: 46
type: B, layer: 1, pos: 19
type: A, layer: 1, pos: 25
type: B, layer: 1, pos: 4
type: A, layer: 1, pos: 38
type: B, layer: 1, pos: 11
type: A, layer: 1, pos: 14
type: A, layer: 1, pos: 29
type: B, layer: 1, pos: 21
type: B, layer: 1, pos: 13
type: A, layer: 1, pos: 46
type: B, layer: 1, pos: 12
type: B, layer: 1, pos: 38
type: A, layer: 1, pos: 4
type: A, layer: 1, pos: 12
type: B, layer: 1, pos: 16
type: A, layer: 1, pos: 20
type: A, layer: 1, pos: 39
type: B, layer: 1, pos: 14
type: B, layer: 1, pos: 31
type: A, layer: 1, pos: 31
type: B, layer: 1, pos: 15
type: B, layer: 1, pos: 29
type: A, layer: 1, pos: 37
type: B, layer: 1, pos: 39
type: B, layer: 1, pos: 8
type: A, layer: 1, pos: 15
type: B, layer: 1, pos: 23
type: A, layer: 1, pos: 22
type: A, layer: 1, pos: 16
type: B, layer: 1, pos: 20
type: A, layer: 1, pos: 23
type: A, layer: 1, pos: 42
type: A, layer: 1, pos: 8
type: A, layer: 1, pos: 11
type: B, layer: 1, pos: 37
type: B, layer: 1, pos: 22

Time for candidate selection: 0.08 seconds

### Candidate
type: B, layer: 1, pos: 41

### Candidate
type: A, layer: 1, pos: 19

### Candidate
type: A, layer: 1, pos: 41

### Candidate
type: A, layer: 1, pos: 6

### Candidate
type: B, layer: 1, pos: 6

### Candidate
type: A, layer: 1, pos: 21

### Candidate
type: B, layer: 1, pos: 25

### Candidate
type: B, layer: 1, pos: 46

### Candidate
type: B, layer: 1, pos: 19

### Candidate
type: A, layer: 1, pos: 25

### Candidate
type: B, layer: 1, pos: 4

### Candidate
type: A, layer: 1, pos: 38

### Candidate
type: B, layer: 1, pos: 11

### Candidate
type: A, layer: 1, pos: 14

### Candidate
type: A, layer: 1, pos: 29

### Candidate
type: B, layer: 1, pos: 21

### Candidate
type: B, layer: 1, pos: 13

### Candidate
type: A, layer: 1, pos: 46

### Candidate
type: B, layer: 1, pos: 12

### Candidate
type: B, layer: 1, pos: 38

### Candidate
type: A, layer: 1, pos: 4

### Candidate
type: A, layer: 1, pos: 12

### Candidate
type: B, layer: 1, pos: 16

### Candidate
type: A, layer: 1, pos: 20

### Candidate
type: A, layer: 1, pos: 39

### Candidate
type: B, layer: 1, pos: 14

### Candidate
type: B, layer: 1, pos: 31

### Candidate
type: A, layer: 1, pos: 31

### Candidate
type: B, layer: 1, pos: 15

### Candidate
type: B, layer: 1, pos: 29

### Candidate
type: A, layer: 1, pos: 37

### Candidate
type: B, layer: 1, pos: 39

### Candidate
type: B, layer: 1, pos: 8

### Candidate
type: A, layer: 1, pos: 15

### Candidate
type: B, layer: 1, pos: 23

### Candidate
type: A, layer: 1, pos: 22

## Relational analysis of NS_A2_A1_B1_A2_B2_A1_B1_B1_A1

### Relational analysis result of NS_A2_A1_B1_A2_B2_A1_B1_B1_A1
Status: Status.UNKNOWN
Output dim: 0, lower bound: -81.8628103, upper bound: 81.8629777
time: 0.80 seconds

## Relational analysis of NS_A2_A1_B1_A2_B2_A1_B1_B1_A2

### Relational analysis result of NS_A2_A1_B1_A2_B2_A1_B1_B1_A2
Status: Status.UNKNOWN
Output dim: 0, lower bound: -81.8627895, upper bound: 81.8628393
time: 1.01 seconds

## BFS NS instance: NS_A2_A1_B1_A2_B2_A1_B1_B2

### Backsubstitution after applying NS history:
0: -19.5407543, 71.3658142, -22.1935444, 81.0053101, -100.5460587, 93.5593491
1: -51.8812523, 160.9836731, -58.6595573, 184.5610046, -236.4422607, 219.6432037
2: -77.8219833, 143.2691956, -88.0264587, 162.7339783, -240.5559692, 231.2956543
3: -44.6342278, 172.0003357, -50.4481354, 197.0260162, -241.6602325, 222.4484711
4: -71.1089172, 126.1929550, -80.3808289, 143.4608917, -214.5698090, 206.5737762

Time for backsubstitution: 0.85 seconds

### NS candidates at layer 1
type: A, layer: 1, pos: 41
type: B, layer: 1, pos: 41
type: A, layer: 1, pos: 19
type: B, layer: 1, pos: 6
type: A, layer: 1, pos: 21
type: A, layer: 1, pos: 6
type: B, layer: 1, pos: 25
type: B, layer: 1, pos: 19
type: B, layer: 1, pos: 46
type: A, layer: 1, pos: 25
type: B, layer: 1, pos: 4
type: A, layer: 1, pos: 14
type: A, layer: 1, pos: 38
type: B, layer: 1, pos: 11
type: B, layer: 1, pos: 21
type: B, layer: 1, pos: 38
type: B, layer: 1, pos: 13
type: A, layer: 1, pos: 46
type: A, layer: 1, pos: 29
type: B, layer: 1, pos: 12
type: B, layer: 1, pos: 16
type: A, layer: 1, pos: 12
type: A, layer: 1, pos: 39
type: A, layer: 1, pos: 4
type: B, layer: 1, pos: 15
type: B, layer: 1, pos: 29
type: B, layer: 1, pos: 31
type: A, layer: 1, pos: 31
type: B, layer: 1, pos: 14
type: A, layer: 1, pos: 37
type: A, layer: 1, pos: 20
type: B, layer: 1, pos: 20
type: A, layer: 1, pos: 22
type: A, layer: 1, pos: 8
type: B, layer: 1, pos: 39
type: A, layer: 1, pos: 15
type: A, layer: 1, pos: 23
type: B, layer: 1, pos: 23
type: B, layer: 1, pos: 8
type: A, layer: 1, pos: 11
type: A, layer: 1, pos: 16
type: B, layer: 1, pos: 37
type: A, layer: 1, pos: 42
type: B, layer: 1, pos: 22

Time for candidate selection: 0.07 seconds

### Candidate
type: A, layer: 1, pos: 41

### Candidate
type: B, layer: 1, pos: 41

### Candidate
type: A, layer: 1, pos: 19

### Candidate
type: B, layer: 1, pos: 6

### Candidate
type: A, layer: 1, pos: 21

### Candidate
type: A, layer: 1, pos: 6

### Candidate
type: B, layer: 1, pos: 25

### Candidate
type: B, layer: 1, pos: 19

### Candidate
type: B, layer: 1, pos: 46

### Candidate
type: A, layer: 1, pos: 25

### Candidate
type: B, layer: 1, pos: 4

### Candidate
type: A, layer: 1, pos: 14

### Candidate
type: A, layer: 1, pos: 38

### Candidate
type: B, layer: 1, pos: 11

### Candidate
type: B, layer: 1, pos: 21

### Candidate
type: B, layer: 1, pos: 38

### Candidate
type: B, layer: 1, pos: 13

### Candidate
type: A, layer: 1, pos: 46

### Candidate
type: A, layer: 1, pos: 29

### Candidate
type: B, layer: 1, pos: 12

### Candidate
type: B, layer: 1, pos: 16

### Candidate
type: A, layer: 1, pos: 12

### Candidate
type: A, layer: 1, pos: 39

### Candidate
type: A, layer: 1, pos: 4

### Candidate
type: B, layer: 1, pos: 15

### Candidate
type: B, layer: 1, pos: 29

### Candidate
type: B, layer: 1, pos: 31

### Candidate
type: A, layer: 1, pos: 31

### Candidate
type: B, layer: 1, pos: 14

### Candidate
type: A, layer: 1, pos: 37

### Candidate
type: A, layer: 1, pos: 20

### Candidate
type: B, layer: 1, pos: 20

### Candidate
type: A, layer: 1, pos: 22

## Relational analysis of NS_A2_A1_B1_A2_B2_A1_B1_B2_A1
Optimization infeasible because this subproblem isn't reachable.

### Candidate
type: A, layer: 1, pos: 8

### Candidate
type: B, layer: 1, pos: 39

### Candidate
type: A, layer: 1, pos: 15

### Candidate
type: A, layer: 1, pos: 23

### Candidate
type: B, layer: 1, pos: 23

### Candidate
type: B, layer: 1, pos: 8

### Candidate
type: A, layer: 1, pos: 11

### Candidate
type: A, layer: 1, pos: 16

### Candidate
type: B, layer: 1, pos: 37

### Candidate
type: A, layer: 1, pos: 42

## Relational analysis of NS_A2_A1_B1_A2_B2_A1_B1_B2_A1
Optimization infeasible because this subproblem isn't reachable.

### Candidate
type: B, layer: 1, pos: 22

## BFS NS instance: NS_A2_A1_B1_A2_B2_A1_B2_B1

### Backsubstitution after applying NS history:
0: -19.5407543, 71.3658142, -22.0341949, 80.3250885, -99.8658447, 93.4000015
1: -51.8812523, 160.9836731, -58.2994423, 182.9301910, -234.8114471, 219.2830963
2: -77.8219833, 143.2691956, -87.5515976, 161.5136566, -239.3356018, 230.8208008
3: -44.6342278, 172.0003357, -50.1588783, 195.4825897, -240.1168060, 222.1592102
4: -71.1089172, 126.1929550, -79.9561539, 142.4262848, -213.5352020, 206.1491089

Time for backsubstitution: 0.86 seconds

### NS candidates at layer 1
type: A, layer: 1, pos: 41
type: A, layer: 1, pos: 14
type: A, layer: 1, pos: 19
type: A, layer: 1, pos: 39
type: B, layer: 1, pos: 14
type: A, layer: 1, pos: 25
type: B, layer: 1, pos: 12
type: B, layer: 1, pos: 19
type: B, layer: 1, pos: 25
type: B, layer: 1, pos: 39
type: A, layer: 1, pos: 20
type: A, layer: 1, pos: 8
type: A, layer: 1, pos: 12
type: A, layer: 1, pos: 37
type: B, layer: 1, pos: 29
type: B, layer: 1, pos: 37
type: A, layer: 1, pos: 29
type: B, layer: 1, pos: 20
type: B, layer: 1, pos: 8
type: A, layer: 1, pos: 21
type: B, layer: 1, pos: 41
type: A, layer: 1, pos: 6
type: B, layer: 1, pos: 6
type: B, layer: 1, pos: 4
type: A, layer: 1, pos: 11
type: A, layer: 1, pos: 46
type: B, layer: 1, pos: 42
type: B, layer: 1, pos: 46
type: B, layer: 1, pos: 15
type: A, layer: 1, pos: 15
type: B, layer: 1, pos: 13
type: A, layer: 1, pos: 4
type: A, layer: 1, pos: 22
type: B, layer: 1, pos: 16
type: B, layer: 1, pos: 38
type: B, layer: 1, pos: 11
type: B, layer: 1, pos: 23
type: A, layer: 1, pos: 38
type: A, layer: 1, pos: 23
type: B, layer: 1, pos: 21
type: A, layer: 1, pos: 31
type: B, layer: 1, pos: 31
type: A, layer: 1, pos: 16
type: A, layer: 1, pos: 42

Time for candidate selection: 0.07 seconds

### Candidate
type: A, layer: 1, pos: 41

### Candidate
type: A, layer: 1, pos: 14

### Candidate
type: A, layer: 1, pos: 19

### Candidate
type: A, layer: 1, pos: 39

### Candidate
type: B, layer: 1, pos: 14

### Candidate
type: A, layer: 1, pos: 25

### Candidate
type: B, layer: 1, pos: 12

### Candidate
type: B, layer: 1, pos: 19

### Candidate
type: B, layer: 1, pos: 25

### Candidate
type: B, layer: 1, pos: 39

### Candidate
type: A, layer: 1, pos: 20

## Relational analysis of NS_A2_A1_B1_A2_B2_A1_B2_B1_A1
Optimization infeasible because this subproblem isn't reachable.

### Candidate
type: A, layer: 1, pos: 8

### Candidate
type: A, layer: 1, pos: 12

### Candidate
type: A, layer: 1, pos: 37

### Candidate
type: B, layer: 1, pos: 29

### Candidate
type: B, layer: 1, pos: 37

### Candidate
type: A, layer: 1, pos: 29

### Candidate
type: B, layer: 1, pos: 20

### Candidate
type: B, layer: 1, pos: 8

### Candidate
type: A, layer: 1, pos: 21

### Candidate
type: B, layer: 1, pos: 41

### Candidate
type: A, layer: 1, pos: 6

### Candidate
type: B, layer: 1, pos: 6

### Candidate
type: B, layer: 1, pos: 4

### Candidate
type: A, layer: 1, pos: 11

### Candidate
type: A, layer: 1, pos: 46

### Candidate
type: B, layer: 1, pos: 42

## Relational analysis of NS_A2_A1_B1_A2_B2_A1_B2_B1_B1

### Relational analysis result of NS_A2_A1_B1_A2_B2_A1_B2_B1_B1
Status: Status.UNKNOWN
Output dim: 0, lower bound: -81.8604805, upper bound: 81.8610756
time: 0.85 seconds

## Relational analysis of NS_A2_A1_B1_A2_B2_A1_B2_B1_B2

### Relational analysis result of NS_A2_A1_B1_A2_B2_A1_B2_B1_B2
Status: Status.UNKNOWN
Output dim: 0, lower bound: -81.8607244, upper bound: 81.8611686
time: 0.96 seconds

## BFS NS instance: NS_A2_A1_B1_A2_B2_A1_B2_B2

### Backsubstitution after applying NS history:
0: -19.2612724, 70.3088608, -21.7932167, 79.2595901, -98.5208588, 92.1020813
1: -51.0843353, 158.7248383, -57.3422089, 181.5820465, -232.6663818, 216.0670471
2: -76.5974350, 141.2996521, -85.8580017, 159.6853485, -236.2827759, 227.1576538
3: -43.9664841, 169.5800476, -49.3844452, 193.9513397, -237.9178162, 218.9644928
4: -70.0057755, 124.4529114, -78.4333725, 140.9108582, -210.9166260, 202.8862610

Time for backsubstitution: 0.87 seconds

### NS candidates at layer 1
type: A, layer: 1, pos: 41
type: A, layer: 1, pos: 14
type: A, layer: 1, pos: 19
type: A, layer: 1, pos: 39
type: B, layer: 1, pos: 14
type: B, layer: 1, pos: 12
type: A, layer: 1, pos: 12
type: B, layer: 1, pos: 39
type: B, layer: 1, pos: 19
type: A, layer: 1, pos: 8
type: A, layer: 1, pos: 37
type: A, layer: 1, pos: 25
type: A, layer: 1, pos: 21
type: B, layer: 1, pos: 29
type: B, layer: 1, pos: 37
type: A, layer: 1, pos: 29
type: B, layer: 1, pos: 25
type: B, layer: 1, pos: 8
type: A, layer: 1, pos: 20
type: B, layer: 1, pos: 41
type: B, layer: 1, pos: 20
type: A, layer: 1, pos: 6
type: B, layer: 1, pos: 15
type: A, layer: 1, pos: 46
type: B, layer: 1, pos: 46
type: B, layer: 1, pos: 6
type: B, layer: 1, pos: 16
type: A, layer: 1, pos: 11
type: B, layer: 1, pos: 21
type: A, layer: 1, pos: 22
type: B, layer: 1, pos: 4
type: A, layer: 1, pos: 15
type: B, layer: 1, pos: 42
type: B, layer: 1, pos: 13
type: A, layer: 1, pos: 38
type: B, layer: 1, pos: 23
type: B, layer: 1, pos: 38
type: B, layer: 1, pos: 11
type: A, layer: 1, pos: 4
type: A, layer: 1, pos: 23
type: A, layer: 1, pos: 31
type: B, layer: 1, pos: 31
type: A, layer: 1, pos: 16
type: A, layer: 1, pos: 42

Time for candidate selection: 0.07 seconds

### Candidate
type: A, layer: 1, pos: 41

### Candidate
type: A, layer: 1, pos: 14

### Candidate
type: A, layer: 1, pos: 19

### Candidate
type: A, layer: 1, pos: 39

### Candidate
type: B, layer: 1, pos: 14

### Candidate
type: B, layer: 1, pos: 12

### Candidate
type: A, layer: 1, pos: 12

### Candidate
type: B, layer: 1, pos: 39

### Candidate
type: B, layer: 1, pos: 19

### Candidate
type: A, layer: 1, pos: 8

### Candidate
type: A, layer: 1, pos: 37

### Candidate
type: A, layer: 1, pos: 25

### Candidate
type: A, layer: 1, pos: 21

### Candidate
type: B, layer: 1, pos: 29

### Candidate
type: B, layer: 1, pos: 37

### Candidate
type: A, layer: 1, pos: 29

### Candidate
type: B, layer: 1, pos: 25

### Candidate
type: B, layer: 1, pos: 8

### Candidate
type: A, layer: 1, pos: 20

## Relational analysis of NS_A2_A1_B1_A2_B2_A1_B2_B2_A1
Optimization infeasible because this subproblem isn't reachable.

### Candidate
type: B, layer: 1, pos: 41

### Candidate
type: B, layer: 1, pos: 20

### Candidate
type: A, layer: 1, pos: 6

### Candidate
type: B, layer: 1, pos: 15

## Relational analysis of NS_A2_A1_B1_A2_B2_A1_B2_B2_B1
Optimization infeasible because this subproblem isn't reachable.

### Candidate
type: A, layer: 1, pos: 46

### Candidate
type: B, layer: 1, pos: 46

### Candidate
type: B, layer: 1, pos: 6

### Candidate
type: B, layer: 1, pos: 16

### Candidate
type: A, layer: 1, pos: 11

### Candidate
type: B, layer: 1, pos: 21

### Candidate
type: A, layer: 1, pos: 22

## Relational analysis of NS_A2_A1_B1_A2_B2_A1_B2_B2_A1

### Relational analysis result of NS_A2_A1_B1_A2_B2_A1_B2_B2_A1
Status: Status.UNKNOWN
Output dim: 0, lower bound: -81.8605006, upper bound: 81.8611519
time: 0.82 seconds

## Relational analysis of NS_A2_A1_B1_A2_B2_A1_B2_B2_A2

### Relational analysis result of NS_A2_A1_B1_A2_B2_A1_B2_B2_A2
Status: Status.UNKNOWN
Output dim: 0, lower bound: -81.8605486, upper bound: 81.8611519
time: 0.72 seconds

## BFS NS instance: NS_A2_A1_B1_A2_B2_A2_B1_B1

### Backsubstitution after applying NS history:
0: -22.8255653, 83.3522263, -20.8587322, 76.1052017, -98.9307404, 104.2109604
1: -60.2388191, 189.8150330, -55.1490402, 173.4452515, -233.6840668, 244.9640656
2: -90.0339966, 167.9370880, -82.7351837, 152.8569946, -242.8909912, 250.6722717
3: -51.7993279, 202.4369354, -47.4171410, 185.1742249, -236.9735565, 249.8540802
4: -82.3146667, 147.9259338, -75.5455551, 134.7193604, -217.0340271, 223.4714813

Time for backsubstitution: 0.86 seconds

### NS candidates at layer 1
type: A, layer: 1, pos: 41
type: B, layer: 1, pos: 41
type: A, layer: 1, pos: 19
type: B, layer: 1, pos: 19
type: B, layer: 1, pos: 6
type: A, layer: 1, pos: 6
type: B, layer: 1, pos: 25
type: A, layer: 1, pos: 21
type: B, layer: 1, pos: 4
type: A, layer: 1, pos: 25
type: B, layer: 1, pos: 46
type: B, layer: 1, pos: 21
type: A, layer: 1, pos: 46
type: A, layer: 1, pos: 38
type: B, layer: 1, pos: 38
type: A, layer: 1, pos: 12
type: B, layer: 1, pos: 12
type: A, layer: 1, pos: 29
type: B, layer: 1, pos: 13
type: A, layer: 1, pos: 14
type: B, layer: 1, pos: 14
type: B, layer: 1, pos: 11
type: A, layer: 1, pos: 4
type: B, layer: 1, pos: 29
type: B, layer: 1, pos: 20
type: B, layer: 1, pos: 31
type: A, layer: 1, pos: 31
type: B, layer: 1, pos: 15
type: A, layer: 1, pos: 39
type: B, layer: 1, pos: 39
type: A, layer: 1, pos: 16
type: A, layer: 1, pos: 8
type: A, layer: 1, pos: 22
type: B, layer: 1, pos: 23
type: A, layer: 1, pos: 15
type: A, layer: 1, pos: 11
type: A, layer: 1, pos: 23
type: B, layer: 1, pos: 42
type: A, layer: 1, pos: 37
type: A, layer: 1, pos: 20
type: B, layer: 1, pos: 8
type: A, layer: 1, pos: 42
type: B, layer: 1, pos: 37
type: B, layer: 1, pos: 22

Time for candidate selection: 0.07 seconds

### Candidate
type: A, layer: 1, pos: 41

### Candidate
type: B, layer: 1, pos: 41

### Candidate
type: A, layer: 1, pos: 19

### Candidate
type: B, layer: 1, pos: 19

### Candidate
type: B, layer: 1, pos: 6

### Candidate
type: A, layer: 1, pos: 6

### Candidate
type: B, layer: 1, pos: 25

### Candidate
type: A, layer: 1, pos: 21

### Candidate
type: B, layer: 1, pos: 4

### Candidate
type: A, layer: 1, pos: 25

### Candidate
type: B, layer: 1, pos: 46

### Candidate
type: B, layer: 1, pos: 21

### Candidate
type: A, layer: 1, pos: 46

### Candidate
type: A, layer: 1, pos: 38

### Candidate
type: B, layer: 1, pos: 38

### Candidate
type: A, layer: 1, pos: 12

### Candidate
type: B, layer: 1, pos: 12

### Candidate
type: A, layer: 1, pos: 29

### Candidate
type: B, layer: 1, pos: 13

### Candidate
type: A, layer: 1, pos: 14

### Candidate
type: B, layer: 1, pos: 14

### Candidate
type: B, layer: 1, pos: 11

### Candidate
type: A, layer: 1, pos: 4

### Candidate
type: B, layer: 1, pos: 29

### Candidate
type: B, layer: 1, pos: 20

### Candidate
type: B, layer: 1, pos: 31

### Candidate
type: A, layer: 1, pos: 31

### Candidate
type: B, layer: 1, pos: 15

## Relational analysis of NS_A2_A1_B1_A2_B2_A2_B1_B1_B1
Optimization infeasible because this subproblem isn't reachable.

### Candidate
type: A, layer: 1, pos: 39

## Relational analysis of NS_A2_A1_B1_A2_B2_A2_B1_B1_A1

### Relational analysis result of NS_A2_A1_B1_A2_B2_A2_B1_B1_A1
Status: Status.UNKNOWN
Output dim: 0, lower bound: -81.8631372, upper bound: 81.8632948
time: 0.84 seconds

## Relational analysis of NS_A2_A1_B1_A2_B2_A2_B1_B1_A2

### Relational analysis result of NS_A2_A1_B1_A2_B2_A2_B1_B1_A2
Status: Status.UNKNOWN
Output dim: 0, lower bound: -81.8632284, upper bound: 81.8633019
time: 0.80 seconds

## BFS NS instance: NS_A2_A1_B1_A2_B2_A2_B1_B2

### Backsubstitution after applying NS history:
0: -22.9382153, 83.7714310, -21.8679695, 79.8196640, -102.7578659, 105.6393967
1: -60.5223961, 190.7922668, -57.7817192, 181.9031372, -242.4255371, 248.5739746
2: -90.4763870, 168.8052826, -86.7036438, 160.4056091, -250.8819885, 255.5089264
3: -52.0459023, 203.4579620, -49.6888580, 194.1599426, -246.2058411, 253.1468201
4: -82.7164154, 148.6980896, -79.1684113, 141.4126129, -224.1290131, 227.8665009

Time for backsubstitution: 0.86 seconds

### NS candidates at layer 1
type: A, layer: 1, pos: 41
type: B, layer: 1, pos: 41
type: B, layer: 1, pos: 19
type: A, layer: 1, pos: 19
type: B, layer: 1, pos: 6
type: A, layer: 1, pos: 6
type: B, layer: 1, pos: 25
type: B, layer: 1, pos: 21
type: A, layer: 1, pos: 21
type: A, layer: 1, pos: 25
type: B, layer: 1, pos: 46
type: A, layer: 1, pos: 46
type: B, layer: 1, pos: 4
type: B, layer: 1, pos: 38
type: A, layer: 1, pos: 38
type: A, layer: 1, pos: 16
type: A, layer: 1, pos: 12
type: A, layer: 1, pos: 4
type: B, layer: 1, pos: 13
type: B, layer: 1, pos: 12
type: B, layer: 1, pos: 14
type: A, layer: 1, pos: 14
type: A, layer: 1, pos: 29
type: B, layer: 1, pos: 29
type: A, layer: 1, pos: 11
type: A, layer: 1, pos: 31
type: B, layer: 1, pos: 31
type: A, layer: 1, pos: 15
type: A, layer: 1, pos: 39
type: B, layer: 1, pos: 20
type: B, layer: 1, pos: 39
type: A, layer: 1, pos: 20
type: B, layer: 1, pos: 11
type: B, layer: 1, pos: 23
type: B, layer: 1, pos: 15
type: B, layer: 1, pos: 8
type: A, layer: 1, pos: 8
type: B, layer: 1, pos: 22
type: B, layer: 1, pos: 42
type: A, layer: 1, pos: 23
type: B, layer: 1, pos: 37
type: A, layer: 1, pos: 37
type: A, layer: 1, pos: 22
type: A, layer: 1, pos: 42

Time for candidate selection: 0.07 seconds

### Candidate
type: A, layer: 1, pos: 41

### Candidate
type: B, layer: 1, pos: 41

### Candidate
type: B, layer: 1, pos: 19

### Candidate
type: A, layer: 1, pos: 19

### Candidate
type: B, layer: 1, pos: 6

### Candidate
type: A, layer: 1, pos: 6

### Candidate
type: B, layer: 1, pos: 25

### Candidate
type: B, layer: 1, pos: 21

### Candidate
type: A, layer: 1, pos: 21

### Candidate
type: A, layer: 1, pos: 25

### Candidate
type: B, layer: 1, pos: 46

### Candidate
type: A, layer: 1, pos: 46

### Candidate
type: B, layer: 1, pos: 4

### Candidate
type: B, layer: 1, pos: 38

### Candidate
type: A, layer: 1, pos: 38

### Candidate
type: A, layer: 1, pos: 16

## Relational analysis of NS_A2_A1_B1_A2_B2_A2_B1_B2_A1

### Relational analysis result of NS_A2_A1_B1_A2_B2_A2_B1_B2_A1
Status: Status.UNKNOWN
Output dim: 0, lower bound: -81.8632007, upper bound: 81.8632840
time: 1.31 seconds

## Relational analysis of NS_A2_A1_B1_A2_B2_A2_B1_B2_A2

### Relational analysis result of NS_A2_A1_B1_A2_B2_A2_B1_B2_A2
Status: Status.UNKNOWN
Output dim: 0, lower bound: -81.8632007, upper bound: 81.8632840
time: 0.82 seconds

## BFS NS instance: NS_A2_A1_B1_A2_B2_A2_B2_A1

### Backsubstitution after applying NS history:
0: -22.5824738, 82.4527893, -22.2047100, 80.9417038, -103.5241699, 104.6574936
1: -59.5856590, 187.8150940, -58.7459526, 184.3729553, -243.9586029, 246.5610504
2: -89.1177292, 166.1107941, -88.1776047, 162.7483826, -251.8661194, 254.2883911
3: -51.2497063, 200.3224335, -50.5433006, 197.0122223, -248.2619019, 250.8657379
4: -81.4681931, 146.3343201, -80.5360565, 143.5206604, -224.9888611, 226.8703766

Time for backsubstitution: 0.86 seconds

### NS candidates at layer 1
type: A, layer: 1, pos: 41
type: A, layer: 1, pos: 25
type: B, layer: 1, pos: 25
type: B, layer: 1, pos: 41
type: B, layer: 1, pos: 12
type: A, layer: 1, pos: 14
type: B, layer: 1, pos: 19
type: B, layer: 1, pos: 29
type: A, layer: 1, pos: 12
type: B, layer: 1, pos: 14
type: A, layer: 1, pos: 19
type: A, layer: 1, pos: 23
type: B, layer: 1, pos: 37
type: A, layer: 1, pos: 29
type: B, layer: 1, pos: 23
type: A, layer: 1, pos: 11
type: A, layer: 1, pos: 37
type: B, layer: 1, pos: 22
type: A, layer: 1, pos: 6
type: A, layer: 1, pos: 46
type: A, layer: 1, pos: 20
type: B, layer: 1, pos: 6
type: A, layer: 1, pos: 21
type: B, layer: 1, pos: 4
type: A, layer: 1, pos: 4
type: B, layer: 1, pos: 8
type: A, layer: 1, pos: 22
type: B, layer: 1, pos: 21
type: B, layer: 1, pos: 13
type: A, layer: 1, pos: 38
type: B, layer: 1, pos: 38
type: B, layer: 1, pos: 46
type: B, layer: 1, pos: 20
type: A, layer: 1, pos: 16
type: A, layer: 1, pos: 31
type: B, layer: 1, pos: 11
type: A, layer: 1, pos: 39
type: B, layer: 1, pos: 31
type: A, layer: 1, pos: 15
type: B, layer: 1, pos: 39
type: B, layer: 1, pos: 42
type: B, layer: 1, pos: 16
type: B, layer: 1, pos: 15
type: A, layer: 1, pos: 42

Time for candidate selection: 0.07 seconds

### Candidate
type: A, layer: 1, pos: 41

### Candidate
type: A, layer: 1, pos: 25

### Candidate
type: B, layer: 1, pos: 25

### Candidate
type: B, layer: 1, pos: 41

### Candidate
type: B, layer: 1, pos: 12

### Candidate
type: A, layer: 1, pos: 14

### Candidate
type: B, layer: 1, pos: 19

### Candidate
type: B, layer: 1, pos: 29

### Candidate
type: A, layer: 1, pos: 12

### Candidate
type: B, layer: 1, pos: 14

### Candidate
type: A, layer: 1, pos: 19

### Candidate
type: A, layer: 1, pos: 23

## Relational analysis of NS_A2_A1_B1_A2_B2_A2_B2_A1_A1
Optimization infeasible because this subproblem isn't reachable.

### Candidate
type: B, layer: 1, pos: 37

## Relational analysis of NS_A2_A1_B1_A2_B2_A2_B2_A1_B1
Optimization infeasible because this subproblem isn't reachable.

### Candidate
type: A, layer: 1, pos: 29

### Candidate
type: B, layer: 1, pos: 23

### Candidate
type: A, layer: 1, pos: 11

## Relational analysis of NS_A2_A1_B1_A2_B2_A2_B2_A1_A1
Optimization infeasible because this subproblem isn't reachable.

### Candidate
type: A, layer: 1, pos: 37

### Candidate
type: B, layer: 1, pos: 22

## Relational analysis of NS_A2_A1_B1_A2_B2_A2_B2_A1_B1
Optimization infeasible because this subproblem isn't reachable.

### Candidate
type: A, layer: 1, pos: 6

### Candidate
type: A, layer: 1, pos: 46

### Candidate
type: A, layer: 1, pos: 20

## Relational analysis of NS_A2_A1_B1_A2_B2_A2_B2_A1_A1
Optimization infeasible because this subproblem isn't reachable.

### Candidate
type: B, layer: 1, pos: 6

### Candidate
type: A, layer: 1, pos: 21

### Candidate
type: B, layer: 1, pos: 4

### Candidate
type: A, layer: 1, pos: 4

### Candidate
type: B, layer: 1, pos: 8

## Relational analysis of NS_A2_A1_B1_A2_B2_A2_B2_A1_B1
Optimization infeasible because this subproblem isn't reachable.

### Candidate
type: A, layer: 1, pos: 22

### Candidate
type: B, layer: 1, pos: 21

### Candidate
type: B, layer: 1, pos: 13

### Candidate
type: A, layer: 1, pos: 38

### Candidate
type: B, layer: 1, pos: 38

### Candidate
type: B, layer: 1, pos: 46

### Candidate
type: B, layer: 1, pos: 20

### Candidate
type: A, layer: 1, pos: 16

## Relational analysis of NS_A2_A1_B1_A2_B2_A2_B2_A1_A1
Optimization infeasible because this subproblem isn't reachable.

### Candidate
type: A, layer: 1, pos: 31

### Candidate
type: B, layer: 1, pos: 11

### Candidate
type: A, layer: 1, pos: 39

## Relational analysis of NS_A2_A1_B1_A2_B2_A2_B2_A1_A1

### Relational analysis result of NS_A2_A1_B1_A2_B2_A2_B2_A1_A1
Status: Status.UNKNOWN
Output dim: 0, lower bound: -81.8610369, upper bound: 81.8615567
time: 0.93 seconds

## Relational analysis of NS_A2_A1_B1_A2_B2_A2_B2_A1_A2

### Relational analysis result of NS_A2_A1_B1_A2_B2_A2_B2_A1_A2
Status: Status.UNKNOWN
Output dim: 0, lower bound: -81.8610171, upper bound: 81.8615472
time: 0.79 seconds

## BFS NS instance: NS_A2_A1_B1_A2_B2_A2_B2_A2

### Backsubstitution after applying NS history:
0: -23.0073433, 84.0065536, -21.8331680, 79.5910492, -102.5983887, 105.8397217
1: -60.6873665, 191.3380280, -57.7683601, 181.3033447, -241.9907074, 249.1063843
2: -90.6737289, 169.3656616, -86.7372589, 160.0077972, -250.6815186, 256.1029053
3: -52.2170639, 204.0262451, -49.7066650, 193.7279816, -245.9450378, 253.7328796
4: -82.9182129, 149.2326202, -79.2135391, 141.1222076, -224.0404205, 228.4461517

Time for backsubstitution: 0.87 seconds

### NS candidates at layer 1
type: A, layer: 1, pos: 41
type: B, layer: 1, pos: 41
type: B, layer: 1, pos: 25
type: A, layer: 1, pos: 25
type: B, layer: 1, pos: 12
type: A, layer: 1, pos: 14
type: B, layer: 1, pos: 14
type: B, layer: 1, pos: 19
type: B, layer: 1, pos: 29
type: A, layer: 1, pos: 19
type: A, layer: 1, pos: 12
type: B, layer: 1, pos: 37
type: A, layer: 1, pos: 23
type: A, layer: 1, pos: 29
type: B, layer: 1, pos: 22
type: B, layer: 1, pos: 23
type: A, layer: 1, pos: 11
type: A, layer: 1, pos: 37
type: A, layer: 1, pos: 6
type: B, layer: 1, pos: 21
type: B, layer: 1, pos: 4
type: A, layer: 1, pos: 46
type: B, layer: 1, pos: 6
type: B, layer: 1, pos: 8
type: B, layer: 1, pos: 20
type: B, layer: 1, pos: 13
type: A, layer: 1, pos: 4
type: B, layer: 1, pos: 46
type: A, layer: 1, pos: 38
type: A, layer: 1, pos: 22
type: B, layer: 1, pos: 38
type: A, layer: 1, pos: 21
type: A, layer: 1, pos: 20
type: B, layer: 1, pos: 39
type: A, layer: 1, pos: 16
type: B, layer: 1, pos: 11
type: A, layer: 1, pos: 15
type: A, layer: 1, pos: 31
type: B, layer: 1, pos: 31
type: B, layer: 1, pos: 42
type: A, layer: 1, pos: 39
type: A, layer: 1, pos: 42
type: B, layer: 1, pos: 15
type: B, layer: 1, pos: 16

Time for candidate selection: 0.10 seconds

### Candidate
type: A, layer: 1, pos: 41

### Candidate
type: B, layer: 1, pos: 41

### Candidate
type: B, layer: 1, pos: 25

### Candidate
type: A, layer: 1, pos: 25

### Candidate
type: B, layer: 1, pos: 12

### Candidate
type: A, layer: 1, pos: 14

### Candidate
type: B, layer: 1, pos: 14

### Candidate
type: B, layer: 1, pos: 19

### Candidate
type: B, layer: 1, pos: 29

### Candidate
type: A, layer: 1, pos: 19

### Candidate
type: A, layer: 1, pos: 12

### Candidate
type: B, layer: 1, pos: 37

## Relational analysis of NS_A2_A1_B1_A2_B2_A2_B2_A2_B1
Optimization infeasible because this subproblem isn't reachable.

### Candidate
type: A, layer: 1, pos: 23

## Relational analysis of NS_A2_A1_B1_A2_B2_A2_B2_A2_A1
Optimization infeasible because this subproblem isn't reachable.

### Candidate
type: A, layer: 1, pos: 29

### Candidate
type: B, layer: 1, pos: 22

## Relational analysis of NS_A2_A1_B1_A2_B2_A2_B2_A2_B1

### Relational analysis result of NS_A2_A1_B1_A2_B2_A2_B2_A2_B1
Status: Status.UNKNOWN
Output dim: 0, lower bound: -81.8605432, upper bound: 81.8612180
time: 1.07 seconds

## Relational analysis of NS_A2_A1_B1_A2_B2_A2_B2_A2_B2

### Relational analysis result of NS_A2_A1_B1_A2_B2_A2_B2_A2_B2
Status: Status.UNKNOWN
Output dim: 0, lower bound: -81.8605873, upper bound: 81.8612151
time: 0.89 seconds

## BFS NS instance: NS_A2_A1_B2_A2_A2_B1_A2_B1

### Backsubstitution after applying NS history:
0: -18.6952496, 68.3234482, -19.5003433, 71.3841171, -90.0793686, 87.8237839
1: -49.6338654, 154.1445160, -51.7450981, 160.7055054, -210.3393707, 205.8895874
2: -74.3940125, 137.2646027, -77.2753830, 143.7020874, -218.0960999, 214.5399780
3: -42.6771584, 164.6407623, -44.4789925, 171.3450165, -214.0221710, 209.1197357
4: -67.9915771, 120.8721466, -70.7467728, 126.4680481, -194.4596252, 191.6188812

Time for backsubstitution: 0.87 seconds

### NS candidates at layer 1
type: A, layer: 1, pos: 41
type: B, layer: 1, pos: 41
type: A, layer: 1, pos: 39
type: B, layer: 1, pos: 19
type: B, layer: 1, pos: 39
type: A, layer: 1, pos: 20
type: A, layer: 1, pos: 19
type: B, layer: 1, pos: 4
type: A, layer: 1, pos: 26
type: A, layer: 1, pos: 14
type: A, layer: 1, pos: 25
type: B, layer: 1, pos: 25
type: A, layer: 1, pos: 23
type: B, layer: 1, pos: 14
type: A, layer: 1, pos: 21
type: B, layer: 1, pos: 6
type: B, layer: 1, pos: 26
type: A, layer: 1, pos: 6
type: B, layer: 1, pos: 23
type: B, layer: 1, pos: 15
type: B, layer: 1, pos: 46
type: B, layer: 1, pos: 38
type: B, layer: 1, pos: 21
type: B, layer: 1, pos: 29
type: B, layer: 1, pos: 13
type: A, layer: 1, pos: 46
type: A, layer: 1, pos: 12
type: A, layer: 1, pos: 16
type: B, layer: 1, pos: 12
type: B, layer: 1, pos: 22
type: A, layer: 1, pos: 11
type: A, layer: 1, pos: 38
type: B, layer: 1, pos: 16
type: A, layer: 1, pos: 8
type: A, layer: 1, pos: 42
type: B, layer: 1, pos: 37
type: A, layer: 1, pos: 29
type: A, layer: 1, pos: 37
type: A, layer: 1, pos: 31
type: B, layer: 1, pos: 11
type: B, layer: 1, pos: 42
type: B, layer: 1, pos: 31
type: A, layer: 1, pos: 22
type: B, layer: 1, pos: 8

Time for candidate selection: 0.07 seconds

### Candidate
type: A, layer: 1, pos: 41

### Candidate
type: B, layer: 1, pos: 41

### Candidate
type: A, layer: 1, pos: 39

### Candidate
type: B, layer: 1, pos: 19

### Candidate
type: B, layer: 1, pos: 39

### Candidate
type: A, layer: 1, pos: 20

## Relational analysis of NS_A2_A1_B2_A2_A2_B1_A2_B1_A1

### Relational analysis result of NS_A2_A1_B2_A2_A2_B1_A2_B1_A1
Status: Status.UNKNOWN
Output dim: 0, lower bound: -81.8632979, upper bound: 81.8633474
time: 0.75 seconds

## Relational analysis of NS_A2_A1_B2_A2_A2_B1_A2_B1_A2
Optimization infeasible because this subproblem isn't reachable.

### Candidate
type: A, layer: 1, pos: 19

### Candidate
type: B, layer: 1, pos: 4

### Candidate
type: A, layer: 1, pos: 26

### Candidate
type: A, layer: 1, pos: 14

### Candidate
type: A, layer: 1, pos: 25

### Candidate
type: B, layer: 1, pos: 25

### Candidate
type: A, layer: 1, pos: 23

## Relational analysis of NS_A2_A1_B2_A2_A2_B1_A2_B1_A1
Optimization infeasible because this subproblem isn't reachable.

### Candidate
type: B, layer: 1, pos: 14

### Candidate
type: A, layer: 1, pos: 21

### Candidate
type: B, layer: 1, pos: 6

### Candidate
type: B, layer: 1, pos: 26

### Candidate
type: A, layer: 1, pos: 6

### Candidate
type: B, layer: 1, pos: 23

### Candidate
type: B, layer: 1, pos: 15

## Relational analysis of NS_A2_A1_B2_A2_A2_B1_A2_B1_B1

### Relational analysis result of NS_A2_A1_B2_A2_A2_B1_A2_B1_B1
Status: Status.UNKNOWN
Output dim: 0, lower bound: -81.8631486, upper bound: 81.8631192
time: 0.93 seconds

## Relational analysis of NS_A2_A1_B2_A2_A2_B1_A2_B1_B2

### Relational analysis result of NS_A2_A1_B2_A2_A2_B1_A2_B1_B2
Status: Status.UNKNOWN
Output dim: 0, lower bound: -81.8632292, upper bound: 81.8633521
time: 1.80 seconds

## BFS NS instance: NS_A2_A1_B2_A2_A2_B1_A2_B2

### Backsubstitution after applying NS history:
0: -19.4070683, 70.8927689, -20.1860847, 73.7450256, -93.1520920, 91.0788574
1: -51.5499992, 159.9037323, -53.6063957, 166.3255768, -217.8755798, 213.5101318
2: -77.2870712, 142.3648529, -80.3843460, 148.0995483, -225.3866272, 222.7492065
3: -44.3350830, 170.8355713, -46.1106377, 177.6928101, -222.0278931, 216.9461975
4: -70.6320572, 125.3745117, -73.4653473, 130.4210968, -201.0531464, 198.8398590

Time for backsubstitution: 0.87 seconds

### NS candidates at layer 1
type: B, layer: 1, pos: 41
type: A, layer: 1, pos: 41
type: B, layer: 1, pos: 14
type: A, layer: 1, pos: 14
type: A, layer: 1, pos: 26
type: B, layer: 1, pos: 19
type: A, layer: 1, pos: 19
type: A, layer: 1, pos: 39
type: A, layer: 1, pos: 25
type: B, layer: 1, pos: 39
type: B, layer: 1, pos: 26
type: A, layer: 1, pos: 20
type: A, layer: 1, pos: 12
type: A, layer: 1, pos: 37
type: B, layer: 1, pos: 25
type: B, layer: 1, pos: 4
type: B, layer: 1, pos: 12
type: B, layer: 1, pos: 8
type: B, layer: 1, pos: 37
type: A, layer: 1, pos: 29
type: A, layer: 1, pos: 8
type: B, layer: 1, pos: 29
type: A, layer: 1, pos: 21
type: B, layer: 1, pos: 15
type: B, layer: 1, pos: 46
type: B, layer: 1, pos: 6
type: A, layer: 1, pos: 6
type: B, layer: 1, pos: 22
type: A, layer: 1, pos: 46
type: B, layer: 1, pos: 11
type: A, layer: 1, pos: 11
type: B, layer: 1, pos: 13
type: A, layer: 1, pos: 22
type: A, layer: 1, pos: 42
type: B, layer: 1, pos: 23
type: B, layer: 1, pos: 38
type: B, layer: 1, pos: 42
type: A, layer: 1, pos: 38
type: B, layer: 1, pos: 21
type: A, layer: 1, pos: 23
type: A, layer: 1, pos: 16
type: A, layer: 1, pos: 31
type: B, layer: 1, pos: 31
type: B, layer: 1, pos: 16

Time for candidate selection: 0.07 seconds

### Candidate
type: B, layer: 1, pos: 41

### Candidate
type: A, layer: 1, pos: 41

### Candidate
type: B, layer: 1, pos: 14

### Candidate
type: A, layer: 1, pos: 14

### Candidate
type: A, layer: 1, pos: 26

### Candidate
type: B, layer: 1, pos: 19

### Candidate
type: A, layer: 1, pos: 19

### Candidate
type: A, layer: 1, pos: 39

### Candidate
type: A, layer: 1, pos: 25

### Candidate
type: B, layer: 1, pos: 39

### Candidate
type: B, layer: 1, pos: 26

### Candidate
type: A, layer: 1, pos: 20

## Relational analysis of NS_A2_A1_B2_A2_A2_B1_A2_B2_A1

### Relational analysis result of NS_A2_A1_B2_A2_A2_B1_A2_B2_A1
Status: Status.UNKNOWN
Output dim: 0, lower bound: -81.8633116, upper bound: 81.8633480
time: 0.89 seconds

## Relational analysis of NS_A2_A1_B2_A2_A2_B1_A2_B2_A2

### Relational analysis result of NS_A2_A1_B2_A2_A2_B1_A2_B2_A2
Status: Status.UNKNOWN
Output dim: 0, lower bound: -81.8633116, upper bound: 81.8633535
time: 0.89 seconds

## BFS NS instance: NS_A2_A1_B2_A2_A2_B2_A1_B1

### Backsubstitution after applying NS history:
0: -19.4434166, 70.9989700, -20.9419403, 76.5952759, -96.0386810, 91.9409103
1: -51.6599655, 160.0805817, -55.6774902, 172.7837982, -224.4437561, 215.7580719
2: -77.5124054, 142.4694824, -83.4180908, 153.7117767, -231.2241669, 225.8875580
3: -44.4397621, 171.0796814, -47.8805275, 184.7951202, -229.2348785, 218.9602051
4: -70.8106613, 125.4927444, -76.2180252, 135.4127655, -206.2234192, 201.7107697

Time for backsubstitution: 0.90 seconds

### NS candidates at layer 1
type: A, layer: 1, pos: 41
type: A, layer: 1, pos: 14
type: B, layer: 1, pos: 19
type: B, layer: 1, pos: 14
type: A, layer: 1, pos: 26
type: B, layer: 1, pos: 39
type: B, layer: 1, pos: 41
type: B, layer: 1, pos: 26
type: A, layer: 1, pos: 19
type: A, layer: 1, pos: 39
type: A, layer: 1, pos: 25
type: B, layer: 1, pos: 12
type: B, layer: 1, pos: 25
type: A, layer: 1, pos: 20
type: A, layer: 1, pos: 12
type: B, layer: 1, pos: 37
type: B, layer: 1, pos: 4
type: B, layer: 1, pos: 29
type: A, layer: 1, pos: 37
type: A, layer: 1, pos: 29
type: B, layer: 1, pos: 20
type: B, layer: 1, pos: 8
type: A, layer: 1, pos: 6
type: A, layer: 1, pos: 11
type: B, layer: 1, pos: 6
type: A, layer: 1, pos: 21
type: A, layer: 1, pos: 46
type: B, layer: 1, pos: 21
type: B, layer: 1, pos: 22
type: A, layer: 1, pos: 15
type: B, layer: 1, pos: 46
type: A, layer: 1, pos: 16
type: B, layer: 1, pos: 13
type: B, layer: 1, pos: 15
type: A, layer: 1, pos: 23
type: A, layer: 1, pos: 38
type: B, layer: 1, pos: 38
type: B, layer: 1, pos: 23
type: A, layer: 1, pos: 42
type: A, layer: 1, pos: 22
type: A, layer: 1, pos: 31
type: B, layer: 1, pos: 11
type: B, layer: 1, pos: 31
type: B, layer: 1, pos: 16

Time for candidate selection: 0.08 seconds

### Candidate
type: A, layer: 1, pos: 41

### Candidate
type: A, layer: 1, pos: 14

### Candidate
type: B, layer: 1, pos: 19

### Candidate
type: B, layer: 1, pos: 14

### Candidate
type: A, layer: 1, pos: 26

### Candidate
type: B, layer: 1, pos: 39

### Candidate
type: B, layer: 1, pos: 41

### Candidate
type: B, layer: 1, pos: 26

### Candidate
type: A, layer: 1, pos: 19

### Candidate
type: A, layer: 1, pos: 39

### Candidate
type: A, layer: 1, pos: 25

### Candidate
type: B, layer: 1, pos: 12

### Candidate
type: B, layer: 1, pos: 25

### Candidate
type: A, layer: 1, pos: 20

### Candidate
type: A, layer: 1, pos: 12

### Candidate
type: B, layer: 1, pos: 37

### Candidate
type: B, layer: 1, pos: 4

### Candidate
type: B, layer: 1, pos: 29

### Candidate
type: A, layer: 1, pos: 37

### Candidate
type: A, layer: 1, pos: 29

### Candidate
type: B, layer: 1, pos: 20

### Candidate
type: B, layer: 1, pos: 8

### Candidate
type: A, layer: 1, pos: 6

### Candidate
type: A, layer: 1, pos: 11

### Candidate
type: B, layer: 1, pos: 6

### Candidate
type: A, layer: 1, pos: 21

### Candidate
type: A, layer: 1, pos: 46

### Candidate
type: B, layer: 1, pos: 21

### Candidate
type: B, layer: 1, pos: 22

### Candidate
type: A, layer: 1, pos: 15

### Candidate
type: B, layer: 1, pos: 46

### Candidate
type: A, layer: 1, pos: 16

### Candidate
type: B, layer: 1, pos: 13

### Candidate
type: B, layer: 1, pos: 15

### Candidate
type: A, layer: 1, pos: 23

### Candidate
type: A, layer: 1, pos: 38

### Candidate
type: B, layer: 1, pos: 38

### Candidate
type: B, layer: 1, pos: 23

### Candidate
type: A, layer: 1, pos: 42

## Relational analysis of NS_A2_A1_B2_A2_A2_B2_A1_B1_A1
Optimization infeasible because this subproblem isn't reachable.

### Candidate
type: A, layer: 1, pos: 22

### Candidate
type: A, layer: 1, pos: 31

### Candidate
type: B, layer: 1, pos: 11

### Candidate
type: B, layer: 1, pos: 31

### Candidate
type: B, layer: 1, pos: 16

## BFS NS instance: NS_A2_A1_B2_A2_A2_B2_A1_B2

### Backsubstitution after applying NS history:
0: -19.6868839, 71.8857117, -20.4000645, 74.4081726, -94.0950546, 92.2857666
1: -52.3027802, 162.1246490, -54.2541618, 167.7283173, -220.0310974, 216.3787689
2: -78.4741592, 144.2440796, -81.4258728, 149.5800323, -228.0541992, 225.6699524
3: -44.9923248, 173.2512360, -46.6753235, 179.4617462, -224.4540710, 219.9265289
4: -71.6982269, 127.0506744, -74.3975449, 131.7936401, -203.4918671, 201.4482117

Time for backsubstitution: 0.97 seconds

### NS candidates at layer 1
type: A, layer: 1, pos: 41
type: A, layer: 1, pos: 25
type: B, layer: 1, pos: 25
type: B, layer: 1, pos: 41
type: B, layer: 1, pos: 12
type: B, layer: 1, pos: 19
type: A, layer: 1, pos: 14
type: B, layer: 1, pos: 4
type: A, layer: 1, pos: 12
type: B, layer: 1, pos: 14
type: B, layer: 1, pos: 29
type: B, layer: 1, pos: 26
type: A, layer: 1, pos: 26
type: A, layer: 1, pos: 19
type: A, layer: 1, pos: 23
type: A, layer: 1, pos: 29
type: B, layer: 1, pos: 37
type: B, layer: 1, pos: 23
type: A, layer: 1, pos: 37
type: A, layer: 1, pos: 11
type: B, layer: 1, pos: 22
type: A, layer: 1, pos: 6
type: A, layer: 1, pos: 46
type: B, layer: 1, pos: 6
type: B, layer: 1, pos: 8
type: A, layer: 1, pos: 20
type: A, layer: 1, pos: 22
type: B, layer: 1, pos: 21
type: A, layer: 1, pos: 21
type: B, layer: 1, pos: 46
type: A, layer: 1, pos: 38
type: B, layer: 1, pos: 13
type: B, layer: 1, pos: 20
type: B, layer: 1, pos: 38
type: A, layer: 1, pos: 16
type: B, layer: 1, pos: 11
type: B, layer: 1, pos: 39
type: A, layer: 1, pos: 31
type: B, layer: 1, pos: 31
type: A, layer: 1, pos: 39
type: A, layer: 1, pos: 15
type: B, layer: 1, pos: 15
type: A, layer: 1, pos: 42
type: B, layer: 1, pos: 16

Time for candidate selection: 0.10 seconds

### Candidate
type: A, layer: 1, pos: 41

### Candidate
type: A, layer: 1, pos: 25

### Candidate
type: B, layer: 1, pos: 25

### Candidate
type: B, layer: 1, pos: 41

### Candidate
type: B, layer: 1, pos: 12

### Candidate
type: B, layer: 1, pos: 19

### Candidate
type: A, layer: 1, pos: 14

### Candidate
type: B, layer: 1, pos: 4

### Candidate
type: A, layer: 1, pos: 12

### Candidate
type: B, layer: 1, pos: 14

### Candidate
type: B, layer: 1, pos: 29

### Candidate
type: B, layer: 1, pos: 26

### Candidate
type: A, layer: 1, pos: 26

### Candidate
type: A, layer: 1, pos: 19

### Candidate
type: A, layer: 1, pos: 23

### Candidate
type: A, layer: 1, pos: 29

### Candidate
type: B, layer: 1, pos: 37

### Candidate
type: B, layer: 1, pos: 23

### Candidate
type: A, layer: 1, pos: 37

### Candidate
type: A, layer: 1, pos: 11

### Candidate
type: B, layer: 1, pos: 22

### Candidate
type: A, layer: 1, pos: 6

### Candidate
type: A, layer: 1, pos: 46

### Candidate
type: B, layer: 1, pos: 6

### Candidate
type: B, layer: 1, pos: 8

### Candidate
type: A, layer: 1, pos: 20

### Candidate
type: A, layer: 1, pos: 22

### Candidate
type: B, layer: 1, pos: 21

### Candidate
type: A, layer: 1, pos: 21

### Candidate
type: B, layer: 1, pos: 46

### Candidate
type: A, layer: 1, pos: 38

### Candidate
type: B, layer: 1, pos: 13

### Candidate
type: B, layer: 1, pos: 20

### Candidate
type: B, layer: 1, pos: 38

### Candidate
type: A, layer: 1, pos: 16

### Candidate
type: B, layer: 1, pos: 11

### Candidate
type: B, layer: 1, pos: 39

### Candidate
type: A, layer: 1, pos: 31

### Candidate
type: B, layer: 1, pos: 31

### Candidate
type: A, layer: 1, pos: 39

### Candidate
type: A, layer: 1, pos: 15

### Candidate
type: B, layer: 1, pos: 15

### Candidate
type: A, layer: 1, pos: 42

## Relational analysis of NS_A2_A1_B2_A2_A2_B2_A1_B2_A1
Optimization infeasible because this subproblem isn't reachable.

### Candidate
type: B, layer: 1, pos: 16

## BFS NS instance: NS_A2_A1_B2_A2_A2_B2_A2_B1

### Backsubstitution after applying NS history:
0: -20.0459213, 73.1830368, -19.8240700, 72.2916412, -92.3375549, 93.0070877
1: -53.1450500, 165.3082275, -52.7287102, 162.9240723, -216.0691071, 218.0369415
2: -79.5760498, 147.2333374, -79.1922989, 145.3326416, -224.9086914, 226.4256287
3: -45.7573967, 176.5436859, -45.3761864, 174.3841248, -220.1415253, 221.9198761
4: -72.7589798, 129.6905060, -72.3446503, 128.0694885, -200.8284302, 202.0351562

Time for backsubstitution: 1.13 seconds

### NS candidates at layer 1
type: A, layer: 1, pos: 41
type: A, layer: 1, pos: 14
type: B, layer: 1, pos: 14
type: B, layer: 1, pos: 19
type: B, layer: 1, pos: 41
type: B, layer: 1, pos: 39
type: B, layer: 1, pos: 26
type: A, layer: 1, pos: 26
type: A, layer: 1, pos: 19
type: A, layer: 1, pos: 39
type: B, layer: 1, pos: 12
type: B, layer: 1, pos: 25
type: A, layer: 1, pos: 25
type: B, layer: 1, pos: 37
type: B, layer: 1, pos: 4
type: A, layer: 1, pos: 12
type: A, layer: 1, pos: 20
type: B, layer: 1, pos: 20
type: B, layer: 1, pos: 29
type: A, layer: 1, pos: 37
type: A, layer: 1, pos: 29
type: A, layer: 1, pos: 6
type: B, layer: 1, pos: 21
type: B, layer: 1, pos: 8
type: B, layer: 1, pos: 6
type: A, layer: 1, pos: 11
type: A, layer: 1, pos: 46
type: A, layer: 1, pos: 15
type: A, layer: 1, pos: 21
type: B, layer: 1, pos: 46
type: B, layer: 1, pos: 13
type: A, layer: 1, pos: 16
type: B, layer: 1, pos: 15
type: A, layer: 1, pos: 23
type: A, layer: 1, pos: 38
type: B, layer: 1, pos: 42
type: A, layer: 1, pos: 42
type: B, layer: 1, pos: 38
type: B, layer: 1, pos: 11
type: A, layer: 1, pos: 31
type: B, layer: 1, pos: 31
type: B, layer: 1, pos: 23
type: A, layer: 1, pos: 22
type: B, layer: 1, pos: 16

Time for candidate selection: 0.09 seconds

### Candidate
type: A, layer: 1, pos: 41

### Candidate
type: A, layer: 1, pos: 14

### Candidate
type: B, layer: 1, pos: 14

### Candidate
type: B, layer: 1, pos: 19

### Candidate
type: B, layer: 1, pos: 41

### Candidate
type: B, layer: 1, pos: 39

## Relational analysis of NS_A2_A1_B2_A2_A2_B2_A2_B1_B1
Optimization infeasible because this subproblem isn't reachable.

### Candidate
type: B, layer: 1, pos: 26

### Candidate
type: A, layer: 1, pos: 26

### Candidate
type: A, layer: 1, pos: 19

### Candidate
type: A, layer: 1, pos: 39

### Candidate
type: B, layer: 1, pos: 12

### Candidate
type: B, layer: 1, pos: 25

### Candidate
type: A, layer: 1, pos: 25

### Candidate
type: B, layer: 1, pos: 37

### Candidate
type: B, layer: 1, pos: 4

### Candidate
type: A, layer: 1, pos: 12

### Candidate
type: A, layer: 1, pos: 20

## Relational analysis of NS_A2_A1_B2_A2_A2_B2_A2_B1_A1
Optimization infeasible because this subproblem isn't reachable.

### Candidate
type: B, layer: 1, pos: 20

### Candidate
type: B, layer: 1, pos: 29

### Candidate
type: A, layer: 1, pos: 37

### Candidate
type: A, layer: 1, pos: 29

### Candidate
type: A, layer: 1, pos: 6

### Candidate
type: B, layer: 1, pos: 21

### Candidate
type: B, layer: 1, pos: 8

## Relational analysis of NS_A2_A1_B2_A2_A2_B2_A2_B1_B1
Optimization infeasible because this subproblem isn't reachable.

### Candidate
type: B, layer: 1, pos: 6

### Candidate
type: A, layer: 1, pos: 11

## Relational analysis of NS_A2_A1_B2_A2_A2_B2_A2_B1_A1
Optimization infeasible because this subproblem isn't reachable.

### Candidate
type: A, layer: 1, pos: 46

### Candidate
type: A, layer: 1, pos: 15

## Relational analysis of NS_A2_A1_B2_A2_A2_B2_A2_B1_A1
Optimization infeasible because this subproblem isn't reachable.

### Candidate
type: A, layer: 1, pos: 21

### Candidate
type: B, layer: 1, pos: 46

### Candidate
type: B, layer: 1, pos: 13

### Candidate
type: A, layer: 1, pos: 16

## Relational analysis of NS_A2_A1_B2_A2_A2_B2_A2_B1_A1
Optimization infeasible because this subproblem isn't reachable.

### Candidate
type: B, layer: 1, pos: 15

### Candidate
type: A, layer: 1, pos: 23

### Candidate
type: A, layer: 1, pos: 38

### Candidate
type: B, layer: 1, pos: 42

## Relational analysis of NS_A2_A1_B2_A2_A2_B2_A2_B1_B1
Optimization infeasible because this subproblem isn't reachable.

### Candidate
type: A, layer: 1, pos: 42

### Candidate
type: B, layer: 1, pos: 38

### Candidate
type: B, layer: 1, pos: 11

### Candidate
type: A, layer: 1, pos: 31

## Relational analysis of NS_A2_A1_B2_A2_A2_B2_A2_B1_A1
Optimization infeasible because this subproblem isn't reachable.

### Candidate
type: B, layer: 1, pos: 31

### Candidate
type: B, layer: 1, pos: 23

### Candidate
type: A, layer: 1, pos: 22

## Relational analysis of NS_A2_A1_B2_A2_A2_B2_A2_B1_A1

### Relational analysis result of NS_A2_A1_B2_A2_A2_B2_A2_B1_A1
Status: Status.UNKNOWN
Output dim: 0, lower bound: -81.8604938, upper bound: 81.8610934
time: 0.97 seconds

## Relational analysis of NS_A2_A1_B2_A2_A2_B2_A2_B1_A2

### Relational analysis result of NS_A2_A1_B2_A2_A2_B2_A2_B1_A2
Status: Status.UNKNOWN
Output dim: 0, lower bound: -81.8604938, upper bound: 81.8610934
time: 0.78 seconds

## BFS NS instance: NS_A2_A1_B2_A2_A2_B2_A2_B2

### Backsubstitution after applying NS history:
0: -19.7194881, 71.9420166, -19.9869442, 72.8922119, -92.6117020, 91.9289627
1: -52.2444572, 162.6484222, -52.9518051, 164.7993774, -217.0438385, 215.6002197
2: -78.2039719, 144.8011169, -79.1051712, 146.9074249, -225.1113586, 223.9062653
3: -44.9858093, 173.7129059, -45.5881691, 176.1671753, -221.1529846, 219.3010559
4: -71.5087280, 127.5537109, -72.3572540, 129.4321136, -200.9408417, 199.9109344

Time for backsubstitution: 0.92 seconds

### NS candidates at layer 1
type: A, layer: 1, pos: 41
type: A, layer: 1, pos: 14
type: B, layer: 1, pos: 14
type: B, layer: 1, pos: 19
type: B, layer: 1, pos: 41
type: B, layer: 1, pos: 39
type: B, layer: 1, pos: 26
type: A, layer: 1, pos: 19
type: A, layer: 1, pos: 26
type: A, layer: 1, pos: 39
type: B, layer: 1, pos: 25
type: B, layer: 1, pos: 12
type: A, layer: 1, pos: 25
type: B, layer: 1, pos: 37
type: B, layer: 1, pos: 4
type: A, layer: 1, pos: 12
type: B, layer: 1, pos: 20
type: A, layer: 1, pos: 20
type: B, layer: 1, pos: 29
type: A, layer: 1, pos: 37
type: A, layer: 1, pos: 29
type: A, layer: 1, pos: 6
type: B, layer: 1, pos: 8
type: B, layer: 1, pos: 21
type: A, layer: 1, pos: 46
type: B, layer: 1, pos: 6
type: A, layer: 1, pos: 11
type: A, layer: 1, pos: 21
type: A, layer: 1, pos: 15
type: B, layer: 1, pos: 46
type: B, layer: 1, pos: 13
type: B, layer: 1, pos: 15
type: A, layer: 1, pos: 38
type: A, layer: 1, pos: 16
type: A, layer: 1, pos: 42
type: A, layer: 1, pos: 23
type: B, layer: 1, pos: 42
type: B, layer: 1, pos: 38
type: B, layer: 1, pos: 11
type: A, layer: 1, pos: 31
type: B, layer: 1, pos: 23
type: B, layer: 1, pos: 31
type: A, layer: 1, pos: 22
type: B, layer: 1, pos: 16

Time for candidate selection: 0.08 seconds

### Candidate
type: A, layer: 1, pos: 41

### Candidate
type: A, layer: 1, pos: 14

### Candidate
type: B, layer: 1, pos: 14

### Candidate
type: B, layer: 1, pos: 19

### Candidate
type: B, layer: 1, pos: 41

### Candidate
type: B, layer: 1, pos: 39

## Relational analysis of NS_A2_A1_B2_A2_A2_B2_A2_B2_B1
Optimization infeasible because this subproblem isn't reachable.

### Candidate
type: B, layer: 1, pos: 26

### Candidate
type: A, layer: 1, pos: 19

### Candidate
type: A, layer: 1, pos: 26

### Candidate
type: A, layer: 1, pos: 39

### Candidate
type: B, layer: 1, pos: 25

### Candidate
type: B, layer: 1, pos: 12

### Candidate
type: A, layer: 1, pos: 25

### Candidate
type: B, layer: 1, pos: 37

### Candidate
type: B, layer: 1, pos: 4

### Candidate
type: A, layer: 1, pos: 12

### Candidate
type: B, layer: 1, pos: 20

## Relational analysis of NS_A2_A1_B2_A2_A2_B2_A2_B2_B1
Optimization infeasible because this subproblem isn't reachable.

### Candidate
type: A, layer: 1, pos: 20

### Candidate
type: B, layer: 1, pos: 29

### Candidate
type: A, layer: 1, pos: 37

### Candidate
type: A, layer: 1, pos: 29

### Candidate
type: A, layer: 1, pos: 6

### Candidate
type: B, layer: 1, pos: 8

## Relational analysis of NS_A2_A1_B2_A2_A2_B2_A2_B2_B1
Optimization infeasible because this subproblem isn't reachable.

### Candidate
type: B, layer: 1, pos: 21

### Candidate
type: A, layer: 1, pos: 46

### Candidate
type: B, layer: 1, pos: 6

### Candidate
type: A, layer: 1, pos: 11

## Relational analysis of NS_A2_A1_B2_A2_A2_B2_A2_B2_A1

### Relational analysis result of NS_A2_A1_B2_A2_A2_B2_A2_B2_A1
Status: Status.VERIFIED
Output dim: 0, lower bound: -81.8599185, upper bound: 81.8600156
time: 0.92 seconds

## Relational analysis of NS_A2_A1_B2_A2_A2_B2_A2_B2_A2

### Relational analysis result of NS_A2_A1_B2_A2_A2_B2_A2_B2_A2
Status: Status.UNKNOWN
Output dim: 0, lower bound: -81.8604938, upper bound: 81.8610934
time: 1.05 seconds

## Summary of splitting at layer (split count: 8)
- Time for NS candidates: 3.76 seconds
NS_A2_A1_B1_A2_B1_A1_B1_A1_B1, status: Status.UNKNOWN, split count: 9, time: 3.76
Output dim: 0, lower bound: -81.8629221, upper bound: 81.8629398
NS_A2_A1_B1_A2_B1_A1_B1_A1_B2, status: Status.UNKNOWN, split count: 9, time: 3.76
Output dim: 0, lower bound: -81.8629221, upper bound: 81.8629398
NS_A2_A1_B1_A2_B1_A1_B1_A2_B1, status: Status.UNKNOWN, split count: 9, time: 3.76
Output dim: 0, lower bound: -81.8629221, upper bound: 81.8629398
NS_A2_A1_B1_A2_B1_A1_B1_A2_B2, status: Status.UNKNOWN, split count: 9, time: 3.76
Output dim: 0, lower bound: -81.8629221, upper bound: 81.8629399
NS_A2_A1_B1_A2_B1_A1_B2_B1_B1, status: Status.UNKNOWN, split count: 9, time: 3.76
Output dim: 0, lower bound: -81.8604385, upper bound: 81.8610384
NS_A2_A1_B1_A2_B1_A1_B2_B1_B2, status: Status.UNKNOWN, split count: 9, time: 3.76
Output dim: 0, lower bound: -81.8605636, upper bound: 81.8611060
NS_A2_A1_B1_A2_B1_A1_B2_B2_A1, status: Status.UNKNOWN, split count: 9, time: 3.76
Output dim: 0, lower bound: -81.8604268, upper bound: 81.8610514
NS_A2_A1_B1_A2_B1_A1_B2_B2_A2, status: Status.UNKNOWN, split count: 9, time: 3.76
Output dim: 0, lower bound: -81.8604268, upper bound: 81.8610514
NS_A2_A1_B1_A2_B1_A2_A1_B1_B1, status: Status.UNKNOWN, split count: 9, time: 3.76
Output dim: 0, lower bound: -81.8605726, upper bound: 81.8611340
NS_A2_A1_B1_A2_B1_A2_A1_B1_B2, status: Status.UNKNOWN, split count: 9, time: 3.76
Output dim: 0, lower bound: -81.8605726, upper bound: 81.8611340
NS_A2_A1_B1_A2_B1_A2_A1_B2_B1, status: Status.UNKNOWN, split count: 9, time: 3.76
Output dim: 0, lower bound: -81.8603972, upper bound: 81.8610697
NS_A2_A1_B1_A2_B1_A2_A1_B2_B2, status: Status.UNKNOWN, split count: 9, time: 3.76
Output dim: 0, lower bound: -81.8603972, upper bound: 81.8610697
NS_A2_A1_B1_A2_B1_A2_A2_B1_B1, status: Status.UNKNOWN, split count: 9, time: 3.76
Output dim: 0, lower bound: -81.8605726, upper bound: 81.8611305
NS_A2_A1_B1_A2_B1_A2_A2_B1_B2, status: Status.UNKNOWN, split count: 9, time: 3.76
Output dim: 0, lower bound: -81.8604279, upper bound: 81.8610642
NS_A2_A1_B1_A2_B1_A2_A2_B2_B1, status: Status.UNKNOWN, split count: 9, time: 3.76
Output dim: 0, lower bound: -81.8605726, upper bound: 81.8611305
NS_A2_A1_B1_A2_B1_A2_A2_B2_B2, status: Status.UNKNOWN, split count: 9, time: 3.76
Output dim: 0, lower bound: -81.8604279, upper bound: 81.8610642
NS_A2_A1_B1_A2_B2_A1_B1_B1_A1, status: Status.UNKNOWN, split count: 9, time: 3.76
Output dim: 0, lower bound: -81.8628103, upper bound: 81.8629777
NS_A2_A1_B1_A2_B2_A1_B1_B1_A2, status: Status.UNKNOWN, split count: 9, time: 3.76
Output dim: 0, lower bound: -81.8627895, upper bound: 81.8628393
NS_A2_A1_B1_A2_B2_A1_B2_B1_B1, status: Status.UNKNOWN, split count: 9, time: 3.76
Output dim: 0, lower bound: -81.8604805, upper bound: 81.8610756
NS_A2_A1_B1_A2_B2_A1_B2_B1_B2, status: Status.UNKNOWN, split count: 9, time: 3.76
Output dim: 0, lower bound: -81.8607244, upper bound: 81.8611686
NS_A2_A1_B1_A2_B2_A1_B2_B2_A1, status: Status.UNKNOWN, split count: 9, time: 3.76
Output dim: 0, lower bound: -81.8605006, upper bound: 81.8611519
NS_A2_A1_B1_A2_B2_A1_B2_B2_A2, status: Status.UNKNOWN, split count: 9, time: 3.76
Output dim: 0, lower bound: -81.8605486, upper bound: 81.8611519
NS_A2_A1_B1_A2_B2_A2_B1_B1_A1, status: Status.UNKNOWN, split count: 9, time: 3.76
Output dim: 0, lower bound: -81.8631372, upper bound: 81.8632948
NS_A2_A1_B1_A2_B2_A2_B1_B1_A2, status: Status.UNKNOWN, split count: 9, time: 3.76
Output dim: 0, lower bound: -81.8632284, upper bound: 81.8633019
NS_A2_A1_B1_A2_B2_A2_B1_B2_A1, status: Status.UNKNOWN, split count: 9, time: 3.76
Output dim: 0, lower bound: -81.8632007, upper bound: 81.8632840
NS_A2_A1_B1_A2_B2_A2_B1_B2_A2, status: Status.UNKNOWN, split count: 9, time: 3.76
Output dim: 0, lower bound: -81.8632007, upper bound: 81.8632840
NS_A2_A1_B1_A2_B2_A2_B2_A1_A1, status: Status.UNKNOWN, split count: 9, time: 3.76
Output dim: 0, lower bound: -81.8610369, upper bound: 81.8615567
NS_A2_A1_B1_A2_B2_A2_B2_A1_A2, status: Status.UNKNOWN, split count: 9, time: 3.76
Output dim: 0, lower bound: -81.8610171, upper bound: 81.8615472
NS_A2_A1_B1_A2_B2_A2_B2_A2_B1, status: Status.UNKNOWN, split count: 9, time: 3.76
Output dim: 0, lower bound: -81.8605432, upper bound: 81.8612180
NS_A2_A1_B1_A2_B2_A2_B2_A2_B2, status: Status.UNKNOWN, split count: 9, time: 3.76
Output dim: 0, lower bound: -81.8605873, upper bound: 81.8612151
NS_A2_A1_B2_A2_A2_B1_A2_B1_B1, status: Status.UNKNOWN, split count: 9, time: 3.76
Output dim: 0, lower bound: -81.8631486, upper bound: 81.8631192
NS_A2_A1_B2_A2_A2_B1_A2_B1_B2, status: Status.UNKNOWN, split count: 9, time: 3.76
Output dim: 0, lower bound: -81.8632292, upper bound: 81.8633521
NS_A2_A1_B2_A2_A2_B1_A2_B2_A1, status: Status.UNKNOWN, split count: 9, time: 3.76
Output dim: 0, lower bound: -81.8633116, upper bound: 81.8633480
NS_A2_A1_B2_A2_A2_B1_A2_B2_A2, status: Status.UNKNOWN, split count: 9, time: 3.76
Output dim: 0, lower bound: -81.8633116, upper bound: 81.8633535
NS_A2_A1_B2_A2_A2_B2_A2_B1_A1, status: Status.UNKNOWN, split count: 9, time: 3.76
Output dim: 0, lower bound: -81.8604938, upper bound: 81.8610934
NS_A2_A1_B2_A2_A2_B2_A2_B1_A2, status: Status.UNKNOWN, split count: 9, time: 3.76
Output dim: 0, lower bound: -81.8604938, upper bound: 81.8610934
NS_A2_A1_B2_A2_A2_B2_A2_B2_A1, status: Status.VERIFIED, split count: 9, time: 3.76
Output dim: 0, lower bound: -81.8599185, upper bound: 81.8600156
NS_A2_A1_B2_A2_A2_B2_A2_B2_A2, status: Status.UNKNOWN, split count: 9, time: 3.76
Output dim: 0, lower bound: -81.8604938, upper bound: 81.8610934

## BFS NS instance: NS_A2_A1_B1_A2_B1_A1_B1_A1_B1

### Backsubstitution after applying NS history:
0: -19.2080135, 70.1533127, -18.5884132, 67.8597412, -87.0677567, 88.7417297
1: -51.0122948, 158.1988220, -49.4699135, 153.0329895, -204.0452728, 207.6687012
2: -76.5601349, 140.7829285, -74.5580063, 135.7942352, -212.3543549, 215.3409424
3: -43.8815269, 169.0470276, -42.5573502, 163.7023773, -207.5839081, 211.6043701
4: -69.9408417, 124.0000458, -68.0377960, 119.6994858, -189.6403046, 192.0378265

Time for backsubstitution: 0.94 seconds

### NS candidates at layer 1
type: A, layer: 1, pos: 41
type: B, layer: 1, pos: 41
type: B, layer: 1, pos: 20
type: A, layer: 1, pos: 39
type: A, layer: 1, pos: 20
type: B, layer: 1, pos: 39
type: A, layer: 1, pos: 19
type: B, layer: 1, pos: 19
type: B, layer: 1, pos: 14
type: B, layer: 1, pos: 25
type: A, layer: 1, pos: 14
type: A, layer: 1, pos: 25
type: B, layer: 1, pos: 23
type: A, layer: 1, pos: 23
type: B, layer: 1, pos: 6
type: A, layer: 1, pos: 6
type: B, layer: 1, pos: 21
type: A, layer: 1, pos: 21
type: B, layer: 1, pos: 4
type: A, layer: 1, pos: 46
type: B, layer: 1, pos: 13
type: A, layer: 1, pos: 4
type: A, layer: 1, pos: 15
type: A, layer: 1, pos: 29
type: B, layer: 1, pos: 46
type: A, layer: 1, pos: 38
type: B, layer: 1, pos: 15
type: B, layer: 1, pos: 16
type: A, layer: 1, pos: 37
type: B, layer: 1, pos: 29
type: B, layer: 1, pos: 38
type: B, layer: 1, pos: 37
type: A, layer: 1, pos: 16
type: A, layer: 1, pos: 12
type: B, layer: 1, pos: 12
type: A, layer: 1, pos: 11
type: B, layer: 1, pos: 42
type: B, layer: 1, pos: 11
type: A, layer: 1, pos: 42
type: A, layer: 1, pos: 31
type: B, layer: 1, pos: 31
type: A, layer: 1, pos: 8
type: B, layer: 1, pos: 8

Time for candidate selection: 0.07 seconds

### Candidate
type: A, layer: 1, pos: 41

### Candidate
type: B, layer: 1, pos: 41

### Candidate
type: B, layer: 1, pos: 20

### Candidate
type: A, layer: 1, pos: 39

### Candidate
type: A, layer: 1, pos: 20

### Candidate
type: B, layer: 1, pos: 39

### Candidate
type: A, layer: 1, pos: 19

### Candidate
type: B, layer: 1, pos: 19

### Candidate
type: B, layer: 1, pos: 14

### Candidate
type: B, layer: 1, pos: 25

### Candidate
type: A, layer: 1, pos: 14

### Candidate
type: A, layer: 1, pos: 25

### Candidate
type: B, layer: 1, pos: 23

### Candidate
type: A, layer: 1, pos: 23

### Candidate
type: B, layer: 1, pos: 6

### Candidate
type: A, layer: 1, pos: 6

### Candidate
type: B, layer: 1, pos: 21

### Candidate
type: A, layer: 1, pos: 21

### Candidate
type: B, layer: 1, pos: 4

### Candidate
type: A, layer: 1, pos: 46

### Candidate
type: B, layer: 1, pos: 13

### Candidate
type: A, layer: 1, pos: 4

### Candidate
type: A, layer: 1, pos: 15

### Candidate
type: A, layer: 1, pos: 29

### Candidate
type: B, layer: 1, pos: 46

### Candidate
type: A, layer: 1, pos: 38

### Candidate
type: B, layer: 1, pos: 15

### Candidate
type: B, layer: 1, pos: 16

### Candidate
type: A, layer: 1, pos: 37

### Candidate
type: B, layer: 1, pos: 29

### Candidate
type: B, layer: 1, pos: 38

### Candidate
type: B, layer: 1, pos: 37

### Candidate
type: A, layer: 1, pos: 16

### Candidate
type: A, layer: 1, pos: 12

### Candidate
type: B, layer: 1, pos: 12

### Candidate
type: A, layer: 1, pos: 11

### Candidate
type: B, layer: 1, pos: 42

### Candidate
type: B, layer: 1, pos: 11

### Candidate
type: A, layer: 1, pos: 42

### Candidate
type: A, layer: 1, pos: 31

### Candidate
type: B, layer: 1, pos: 31

### Candidate
type: A, layer: 1, pos: 8

## Relational analysis of NS_A2_A1_B1_A2_B1_A1_B1_A1_B1_A1

### Relational analysis result of NS_A2_A1_B1_A2_B1_A1_B1_A1_B1_A1
Status: Status.UNKNOWN
Output dim: 0, lower bound: -81.8628677, upper bound: 81.8629717
time: 0.84 seconds

## Relational analysis of NS_A2_A1_B1_A2_B1_A1_B1_A1_B1_A2

### Relational analysis result of NS_A2_A1_B1_A2_B1_A1_B1_A1_B1_A2
Status: Status.UNKNOWN
Output dim: 0, lower bound: -81.8626178, upper bound: 81.8626890
time: 0.92 seconds

## BFS NS instance: NS_A2_A1_B1_A2_B1_A1_B1_A1_B2

### Backsubstitution after applying NS history:
0: -19.2080135, 70.1533127, -18.7781601, 68.4837570, -87.6917725, 88.9314728
1: -51.0122948, 158.1988220, -49.7113342, 155.0832672, -206.0955353, 207.9101257
2: -76.5601349, 140.7829285, -74.5315704, 137.5883636, -214.1484680, 215.3144989
3: -43.8815269, 169.0470276, -42.8129158, 165.7190399, -209.6005707, 211.8599243
4: -69.9408417, 124.0000458, -68.1134262, 121.2504272, -191.1912537, 192.1134644

Time for backsubstitution: 1.08 seconds

### NS candidates at layer 1
type: A, layer: 1, pos: 41
type: B, layer: 1, pos: 41
type: B, layer: 1, pos: 20
type: A, layer: 1, pos: 39
type: B, layer: 1, pos: 39
type: A, layer: 1, pos: 20
type: A, layer: 1, pos: 19
type: B, layer: 1, pos: 19
type: B, layer: 1, pos: 25
type: A, layer: 1, pos: 14
type: B, layer: 1, pos: 14
type: A, layer: 1, pos: 25
type: B, layer: 1, pos: 23
type: A, layer: 1, pos: 6
type: A, layer: 1, pos: 23
type: B, layer: 1, pos: 6
type: A, layer: 1, pos: 21
type: B, layer: 1, pos: 21
type: B, layer: 1, pos: 15
type: A, layer: 1, pos: 46
type: B, layer: 1, pos: 4
type: A, layer: 1, pos: 38
type: B, layer: 1, pos: 46
type: B, layer: 1, pos: 16
type: B, layer: 1, pos: 13
type: A, layer: 1, pos: 29
type: A, layer: 1, pos: 15
type: A, layer: 1, pos: 4
type: A, layer: 1, pos: 12
type: B, layer: 1, pos: 12
type: A, layer: 1, pos: 37
type: B, layer: 1, pos: 38
type: A, layer: 1, pos: 16
type: B, layer: 1, pos: 29
type: A, layer: 1, pos: 11
type: A, layer: 1, pos: 42
type: A, layer: 1, pos: 8
type: B, layer: 1, pos: 42
type: B, layer: 1, pos: 11
type: B, layer: 1, pos: 31
type: A, layer: 1, pos: 31
type: B, layer: 1, pos: 37
type: B, layer: 1, pos: 8

Time for candidate selection: 0.08 seconds

### Candidate
type: A, layer: 1, pos: 41

### Candidate
type: B, layer: 1, pos: 41

### Candidate
type: B, layer: 1, pos: 20

### Candidate
type: A, layer: 1, pos: 39

### Candidate
type: B, layer: 1, pos: 39

### Candidate
type: A, layer: 1, pos: 20

### Candidate
type: A, layer: 1, pos: 19

### Candidate
type: B, layer: 1, pos: 19

### Candidate
type: B, layer: 1, pos: 25

### Candidate
type: A, layer: 1, pos: 14

### Candidate
type: B, layer: 1, pos: 14

### Candidate
type: A, layer: 1, pos: 25

### Candidate
type: B, layer: 1, pos: 23

### Candidate
type: A, layer: 1, pos: 6

### Candidate
type: A, layer: 1, pos: 23

### Candidate
type: B, layer: 1, pos: 6

### Candidate
type: A, layer: 1, pos: 21

### Candidate
type: B, layer: 1, pos: 21

### Candidate
type: B, layer: 1, pos: 15

### Candidate
type: A, layer: 1, pos: 46

### Candidate
type: B, layer: 1, pos: 4

### Candidate
type: A, layer: 1, pos: 38

### Candidate
type: B, layer: 1, pos: 46

### Candidate
type: B, layer: 1, pos: 16

### Candidate
type: B, layer: 1, pos: 13

### Candidate
type: A, layer: 1, pos: 29

### Candidate
type: A, layer: 1, pos: 15

### Candidate
type: A, layer: 1, pos: 4

### Candidate
type: A, layer: 1, pos: 12

### Candidate
type: B, layer: 1, pos: 12

### Candidate
type: A, layer: 1, pos: 37

### Candidate
type: B, layer: 1, pos: 38

### Candidate
type: A, layer: 1, pos: 16

### Candidate
type: B, layer: 1, pos: 29

### Candidate
type: A, layer: 1, pos: 11

### Candidate
type: A, layer: 1, pos: 42

### Candidate
type: A, layer: 1, pos: 8

## Relational analysis of NS_A2_A1_B1_A2_B1_A1_B1_A1_B2_A1

### Relational analysis result of NS_A2_A1_B1_A2_B1_A1_B1_A1_B2_A1
Status: Status.UNKNOWN
Output dim: 0, lower bound: -81.8628677, upper bound: 81.8629716
time: 1.00 seconds

## Relational analysis of NS_A2_A1_B1_A2_B1_A1_B1_A1_B2_A2

### Relational analysis result of NS_A2_A1_B1_A2_B1_A1_B1_A1_B2_A2
Status: Status.UNKNOWN
Output dim: 0, lower bound: -81.8626178, upper bound: 81.8626891
time: 0.78 seconds

## BFS NS instance: NS_A2_A1_B1_A2_B1_A1_B1_A2_B1

### Backsubstitution after applying NS history:
0: -19.3761635, 70.6416855, -18.5802975, 67.8309326, -87.2070923, 89.2219696
1: -51.1780510, 160.0721283, -49.4485245, 152.9668579, -204.1449127, 209.5206146
2: -76.4557343, 142.3144379, -74.5244980, 135.7377319, -212.1934204, 216.8389282
3: -44.0854263, 170.8686676, -42.5389595, 163.6310272, -207.7164612, 213.4075928
4: -69.9495621, 125.3343735, -68.0075836, 119.6497345, -189.5993042, 193.3419495

Time for backsubstitution: 0.98 seconds

### NS candidates at layer 1
type: A, layer: 1, pos: 41
type: B, layer: 1, pos: 41
type: A, layer: 1, pos: 19
type: B, layer: 1, pos: 19
type: B, layer: 1, pos: 6
type: A, layer: 1, pos: 6
type: B, layer: 1, pos: 21
type: A, layer: 1, pos: 25
type: B, layer: 1, pos: 25
type: A, layer: 1, pos: 21
type: A, layer: 1, pos: 46
type: B, layer: 1, pos: 46
type: B, layer: 1, pos: 4
type: B, layer: 1, pos: 38
type: A, layer: 1, pos: 38
type: B, layer: 1, pos: 14
type: B, layer: 1, pos: 13
type: A, layer: 1, pos: 4
type: B, layer: 1, pos: 12
type: A, layer: 1, pos: 12
type: B, layer: 1, pos: 29
type: A, layer: 1, pos: 14
type: A, layer: 1, pos: 29
type: B, layer: 1, pos: 11
type: A, layer: 1, pos: 31
type: B, layer: 1, pos: 31
type: A, layer: 1, pos: 15
type: A, layer: 1, pos: 39
type: B, layer: 1, pos: 39
type: B, layer: 1, pos: 16
type: A, layer: 1, pos: 11
type: A, layer: 1, pos: 16
type: A, layer: 1, pos: 20
type: B, layer: 1, pos: 20
type: B, layer: 1, pos: 15
type: B, layer: 1, pos: 8
type: A, layer: 1, pos: 23
type: B, layer: 1, pos: 42
type: B, layer: 1, pos: 23
type: A, layer: 1, pos: 8
type: B, layer: 1, pos: 37
type: A, layer: 1, pos: 37
type: A, layer: 1, pos: 42

Time for candidate selection: 0.08 seconds

### Candidate
type: A, layer: 1, pos: 41

### Candidate
type: B, layer: 1, pos: 41

### Candidate
type: A, layer: 1, pos: 19

### Candidate
type: B, layer: 1, pos: 19

### Candidate
type: B, layer: 1, pos: 6

### Candidate
type: A, layer: 1, pos: 6

### Candidate
type: B, layer: 1, pos: 21

### Candidate
type: A, layer: 1, pos: 25

### Candidate
type: B, layer: 1, pos: 25

### Candidate
type: A, layer: 1, pos: 21

### Candidate
type: A, layer: 1, pos: 46

### Candidate
type: B, layer: 1, pos: 46

### Candidate
type: B, layer: 1, pos: 4

### Candidate
type: B, layer: 1, pos: 38

### Candidate
type: A, layer: 1, pos: 38

### Candidate
type: B, layer: 1, pos: 14

### Candidate
type: B, layer: 1, pos: 13

### Candidate
type: A, layer: 1, pos: 4

### Candidate
type: B, layer: 1, pos: 12

### Candidate
type: A, layer: 1, pos: 12

### Candidate
type: B, layer: 1, pos: 29

### Candidate
type: A, layer: 1, pos: 14

### Candidate
type: A, layer: 1, pos: 29

### Candidate
type: B, layer: 1, pos: 11

### Candidate
type: A, layer: 1, pos: 31

### Candidate
type: B, layer: 1, pos: 31

### Candidate
type: A, layer: 1, pos: 15

### Candidate
type: A, layer: 1, pos: 39

### Candidate
type: B, layer: 1, pos: 39

### Candidate
type: B, layer: 1, pos: 16

### Candidate
type: A, layer: 1, pos: 11

### Candidate
type: A, layer: 1, pos: 16

### Candidate
type: A, layer: 1, pos: 20

### Candidate
type: B, layer: 1, pos: 20

### Candidate
type: B, layer: 1, pos: 15

### Candidate
type: B, layer: 1, pos: 8

## Relational analysis of NS_A2_A1_B1_A2_B1_A1_B1_A2_B1_B1

### Relational analysis result of NS_A2_A1_B1_A2_B1_A1_B1_A2_B1_B1
Status: Status.UNKNOWN
Output dim: 0, lower bound: -81.8627495, upper bound: 81.8627730
time: 0.84 seconds

## Relational analysis of NS_A2_A1_B1_A2_B1_A1_B1_A2_B1_B2

### Relational analysis result of NS_A2_A1_B1_A2_B1_A1_B1_A2_B1_B2
Status: Status.UNKNOWN
Output dim: 0, lower bound: -81.8626021, upper bound: 81.8626291
time: 0.80 seconds

## BFS NS instance: NS_A2_A1_B1_A2_B1_A1_B1_A2_B2

### Backsubstitution after applying NS history:
0: -19.3761635, 70.6416855, -18.7781601, 68.4837570, -87.8599243, 89.4198456
1: -51.1780510, 160.0721283, -49.7113342, 155.0832672, -206.2613220, 209.7834167
2: -76.4557343, 142.3144379, -74.5315704, 137.5883636, -214.0440979, 216.8460083
3: -44.0854263, 170.8686676, -42.8129158, 165.7190399, -209.8044739, 213.6815643
4: -69.9495621, 125.3343735, -68.1134262, 121.2504272, -191.1999817, 193.4477997

Time for backsubstitution: 0.95 seconds

### NS candidates at layer 1
type: A, layer: 1, pos: 41
type: B, layer: 1, pos: 41
type: A, layer: 1, pos: 19
type: B, layer: 1, pos: 19
type: B, layer: 1, pos: 6
type: A, layer: 1, pos: 6
type: B, layer: 1, pos: 25
type: A, layer: 1, pos: 25
type: B, layer: 1, pos: 21
type: A, layer: 1, pos: 21
type: A, layer: 1, pos: 46
type: B, layer: 1, pos: 46
type: B, layer: 1, pos: 4
type: A, layer: 1, pos: 38
type: B, layer: 1, pos: 38
type: B, layer: 1, pos: 13
type: A, layer: 1, pos: 12
type: A, layer: 1, pos: 4
type: A, layer: 1, pos: 29
type: B, layer: 1, pos: 14
type: B, layer: 1, pos: 12
type: A, layer: 1, pos: 14
type: B, layer: 1, pos: 29
type: A, layer: 1, pos: 31
type: B, layer: 1, pos: 31
type: B, layer: 1, pos: 11
type: A, layer: 1, pos: 39
type: B, layer: 1, pos: 16
type: B, layer: 1, pos: 20
type: A, layer: 1, pos: 11
type: B, layer: 1, pos: 15
type: B, layer: 1, pos: 39
type: A, layer: 1, pos: 15
type: A, layer: 1, pos: 16
type: B, layer: 1, pos: 23
type: A, layer: 1, pos: 8
type: A, layer: 1, pos: 20
type: A, layer: 1, pos: 23
type: B, layer: 1, pos: 8
type: B, layer: 1, pos: 42
type: A, layer: 1, pos: 37
type: B, layer: 1, pos: 37
type: A, layer: 1, pos: 42

Time for candidate selection: 0.08 seconds

### Candidate
type: A, layer: 1, pos: 41

### Candidate
type: B, layer: 1, pos: 41

### Candidate
type: A, layer: 1, pos: 19

### Candidate
type: B, layer: 1, pos: 19

### Candidate
type: B, layer: 1, pos: 6

### Candidate
type: A, layer: 1, pos: 6

### Candidate
type: B, layer: 1, pos: 25

### Candidate
type: A, layer: 1, pos: 25

### Candidate
type: B, layer: 1, pos: 21

### Candidate
type: A, layer: 1, pos: 21

### Candidate
type: A, layer: 1, pos: 46

### Candidate
type: B, layer: 1, pos: 46

### Candidate
type: B, layer: 1, pos: 4

### Candidate
type: A, layer: 1, pos: 38

### Candidate
type: B, layer: 1, pos: 38

### Candidate
type: B, layer: 1, pos: 13

### Candidate
type: A, layer: 1, pos: 12

### Candidate
type: A, layer: 1, pos: 4

### Candidate
type: A, layer: 1, pos: 29

### Candidate
type: B, layer: 1, pos: 14

### Candidate
type: B, layer: 1, pos: 12

### Candidate
type: A, layer: 1, pos: 14

### Candidate
type: B, layer: 1, pos: 29

### Candidate
type: A, layer: 1, pos: 31

### Candidate
type: B, layer: 1, pos: 31

### Candidate
type: B, layer: 1, pos: 11

### Candidate
type: A, layer: 1, pos: 39

### Candidate
type: B, layer: 1, pos: 16

### Candidate
type: B, layer: 1, pos: 20

### Candidate
type: A, layer: 1, pos: 11

### Candidate
type: B, layer: 1, pos: 15

### Candidate
type: B, layer: 1, pos: 39

### Candidate
type: A, layer: 1, pos: 15

### Candidate
type: A, layer: 1, pos: 16

### Candidate
type: B, layer: 1, pos: 23

### Candidate
type: A, layer: 1, pos: 8

## Relational analysis of NS_A2_A1_B1_A2_B1_A1_B1_A2_B2_A1

### Relational analysis result of NS_A2_A1_B1_A2_B1_A1_B1_A2_B2_A1
Status: Status.UNKNOWN
Output dim: 0, lower bound: -81.8627703, upper bound: 81.8627925
time: 0.94 seconds

## Relational analysis of NS_A2_A1_B1_A2_B1_A1_B1_A2_B2_A2

### Relational analysis result of NS_A2_A1_B1_A2_B1_A1_B1_A2_B2_A2
Status: Status.UNKNOWN
Output dim: 0, lower bound: -81.8626021, upper bound: 81.8626291
time: 0.81 seconds

## BFS NS instance: NS_A2_A1_B1_A2_B1_A1_B2_B1_B1

### Backsubstitution after applying NS history:
0: -19.2622166, 70.3484726, -19.0926151, 69.8261414, -89.0883560, 89.4410858
1: -51.1395035, 158.6659393, -50.8617516, 157.4954071, -208.6349030, 209.5276947
2: -76.7113037, 141.2533112, -76.5586243, 139.6645355, -216.3758240, 217.8119354
3: -43.9973488, 169.5396729, -43.7430496, 168.6731262, -212.6704712, 213.2826996
4: -70.0857925, 124.4224167, -69.8557816, 123.1292496, -193.2150421, 194.2781982

Time for backsubstitution: 0.95 seconds

### NS candidates at layer 1
type: A, layer: 1, pos: 41
type: A, layer: 1, pos: 14
type: B, layer: 1, pos: 14
type: A, layer: 1, pos: 19
type: A, layer: 1, pos: 39
type: B, layer: 1, pos: 19
type: B, layer: 1, pos: 12
type: B, layer: 1, pos: 39
type: B, layer: 1, pos: 25
type: A, layer: 1, pos: 25
type: A, layer: 1, pos: 20
type: B, layer: 1, pos: 37
type: A, layer: 1, pos: 12
type: B, layer: 1, pos: 29
type: A, layer: 1, pos: 8
type: A, layer: 1, pos: 37
type: B, layer: 1, pos: 41
type: B, layer: 1, pos: 8
type: B, layer: 1, pos: 20
type: A, layer: 1, pos: 29
type: A, layer: 1, pos: 21
type: A, layer: 1, pos: 11
type: A, layer: 1, pos: 46
type: A, layer: 1, pos: 6
type: B, layer: 1, pos: 6
type: B, layer: 1, pos: 4
type: A, layer: 1, pos: 15
type: B, layer: 1, pos: 21
type: A, layer: 1, pos: 4
type: B, layer: 1, pos: 13
type: B, layer: 1, pos: 46
type: A, layer: 1, pos: 38
type: B, layer: 1, pos: 15
type: A, layer: 1, pos: 16
type: A, layer: 1, pos: 22
type: B, layer: 1, pos: 23
type: A, layer: 1, pos: 23
type: B, layer: 1, pos: 38
type: B, layer: 1, pos: 16
type: A, layer: 1, pos: 31
type: A, layer: 1, pos: 42
type: B, layer: 1, pos: 31
type: B, layer: 1, pos: 11

Time for candidate selection: 0.08 seconds

### Candidate
type: A, layer: 1, pos: 41

### Candidate
type: A, layer: 1, pos: 14

### Candidate
type: B, layer: 1, pos: 14

### Candidate
type: A, layer: 1, pos: 19

### Candidate
type: A, layer: 1, pos: 39

### Candidate
type: B, layer: 1, pos: 19

### Candidate
type: B, layer: 1, pos: 12

### Candidate
type: B, layer: 1, pos: 39

### Candidate
type: B, layer: 1, pos: 25

### Candidate
type: A, layer: 1, pos: 25

### Candidate
type: A, layer: 1, pos: 20

### Candidate
type: B, layer: 1, pos: 37

### Candidate
type: A, layer: 1, pos: 12

### Candidate
type: B, layer: 1, pos: 29

### Candidate
type: A, layer: 1, pos: 8

### Candidate
type: A, layer: 1, pos: 37

### Candidate
type: B, layer: 1, pos: 41

### Candidate
type: B, layer: 1, pos: 8

### Candidate
type: B, layer: 1, pos: 20

### Candidate
type: A, layer: 1, pos: 29

### Candidate
type: A, layer: 1, pos: 21

### Candidate
type: A, layer: 1, pos: 11

### Candidate
type: A, layer: 1, pos: 46

### Candidate
type: A, layer: 1, pos: 6

### Candidate
type: B, layer: 1, pos: 6

### Candidate
type: B, layer: 1, pos: 4

### Candidate
type: A, layer: 1, pos: 15

### Candidate
type: B, layer: 1, pos: 21

### Candidate
type: A, layer: 1, pos: 4

### Candidate
type: B, layer: 1, pos: 13

### Candidate
type: B, layer: 1, pos: 46

### Candidate
type: A, layer: 1, pos: 38

### Candidate
type: B, layer: 1, pos: 15

### Candidate
type: A, layer: 1, pos: 16

## Relational analysis of NS_A2_A1_B1_A2_B1_A1_B2_B1_B1_A1
Optimization infeasible because this subproblem isn't reachable.

### Candidate
type: A, layer: 1, pos: 22

## Relational analysis of NS_A2_A1_B1_A2_B1_A1_B2_B1_B1_A1

### Relational analysis result of NS_A2_A1_B1_A2_B1_A1_B2_B1_B1_A1
Status: Status.UNKNOWN
Output dim: 0, lower bound: -81.8604385, upper bound: 81.8610384
time: 0.70 seconds

## Relational analysis of NS_A2_A1_B1_A2_B1_A1_B2_B1_B1_A2

### Relational analysis result of NS_A2_A1_B1_A2_B1_A1_B2_B1_B1_A2
Status: Status.UNKNOWN
Output dim: 0, lower bound: -81.8604385, upper bound: 81.8610384
time: 0.78 seconds

## BFS NS instance: NS_A2_A1_B1_A2_B1_A1_B2_B1_B2

### Backsubstitution after applying NS history:
0: -19.5514183, 71.4037170, -18.5805836, 67.7363052, -87.2877197, 89.9842987
1: -51.9093094, 161.0701752, -49.5240135, 152.6590118, -204.5683136, 210.5941925
2: -77.8657913, 143.3444519, -74.6881332, 135.7837677, -213.6495514, 218.0325623
3: -44.6583481, 172.0934601, -42.6170883, 163.5825653, -208.2409058, 214.7105408
4: -71.1484833, 126.2585678, -68.1534424, 119.7229919, -190.8714752, 194.4120026

Time for backsubstitution: 0.96 seconds

### NS candidates at layer 1
type: A, layer: 1, pos: 41
type: A, layer: 1, pos: 14
type: B, layer: 1, pos: 14
type: A, layer: 1, pos: 19
type: A, layer: 1, pos: 39
type: B, layer: 1, pos: 19
type: B, layer: 1, pos: 12
type: A, layer: 1, pos: 25
type: B, layer: 1, pos: 39
type: B, layer: 1, pos: 25
type: A, layer: 1, pos: 12
type: B, layer: 1, pos: 37
type: A, layer: 1, pos: 8
type: B, layer: 1, pos: 29
type: A, layer: 1, pos: 20
type: A, layer: 1, pos: 37
type: B, layer: 1, pos: 20
type: B, layer: 1, pos: 41
type: B, layer: 1, pos: 8
type: A, layer: 1, pos: 29
type: A, layer: 1, pos: 11
type: A, layer: 1, pos: 21
type: A, layer: 1, pos: 46
type: A, layer: 1, pos: 6
type: B, layer: 1, pos: 6
type: B, layer: 1, pos: 4
type: B, layer: 1, pos: 21
type: A, layer: 1, pos: 15
type: B, layer: 1, pos: 15
type: B, layer: 1, pos: 13
type: A, layer: 1, pos: 4
type: B, layer: 1, pos: 46
type: A, layer: 1, pos: 38
type: A, layer: 1, pos: 22
type: A, layer: 1, pos: 23
type: B, layer: 1, pos: 23
type: B, layer: 1, pos: 38
type: B, layer: 1, pos: 16
type: A, layer: 1, pos: 16
type: A, layer: 1, pos: 31
type: A, layer: 1, pos: 42
type: B, layer: 1, pos: 31
type: B, layer: 1, pos: 11

Time for candidate selection: 0.07 seconds

### Candidate
type: A, layer: 1, pos: 41

### Candidate
type: A, layer: 1, pos: 14

### Candidate
type: B, layer: 1, pos: 14

### Candidate
type: A, layer: 1, pos: 19

### Candidate
type: A, layer: 1, pos: 39

### Candidate
type: B, layer: 1, pos: 19

### Candidate
type: B, layer: 1, pos: 12

### Candidate
type: A, layer: 1, pos: 25

### Candidate
type: B, layer: 1, pos: 39

### Candidate
type: B, layer: 1, pos: 25

### Candidate
type: A, layer: 1, pos: 12

### Candidate
type: B, layer: 1, pos: 37

### Candidate
type: A, layer: 1, pos: 8

### Candidate
type: B, layer: 1, pos: 29

### Candidate
type: A, layer: 1, pos: 20

### Candidate
type: A, layer: 1, pos: 37

### Candidate
type: B, layer: 1, pos: 20

### Candidate
type: B, layer: 1, pos: 41

### Candidate
type: B, layer: 1, pos: 8

### Candidate
type: A, layer: 1, pos: 29

### Candidate
type: A, layer: 1, pos: 11

### Candidate
type: A, layer: 1, pos: 21

### Candidate
type: A, layer: 1, pos: 46

### Candidate
type: A, layer: 1, pos: 6

### Candidate
type: B, layer: 1, pos: 6

### Candidate
type: B, layer: 1, pos: 4

### Candidate
type: B, layer: 1, pos: 21

### Candidate
type: A, layer: 1, pos: 15

### Candidate
type: B, layer: 1, pos: 15

### Candidate
type: B, layer: 1, pos: 13

### Candidate
type: A, layer: 1, pos: 4

### Candidate
type: B, layer: 1, pos: 46

### Candidate
type: A, layer: 1, pos: 38

### Candidate
type: A, layer: 1, pos: 22

## Relational analysis of NS_A2_A1_B1_A2_B1_A1_B2_B1_B2_A1

### Relational analysis result of NS_A2_A1_B1_A2_B1_A1_B2_B1_B2_A1
Status: Status.UNKNOWN
Output dim: 0, lower bound: -81.8605636, upper bound: 81.8611060
time: 0.84 seconds

## Relational analysis of NS_A2_A1_B1_A2_B1_A1_B2_B1_B2_A2

### Relational analysis result of NS_A2_A1_B1_A2_B1_A1_B2_B1_B2_A2
Status: Status.UNKNOWN
Output dim: 0, lower bound: -81.8605636, upper bound: 81.8611060
time: 0.83 seconds

## BFS NS instance: NS_A2_A1_B1_A2_B1_A1_B2_B2_A1

### Backsubstitution after applying NS history:
0: -19.1982937, 70.1187820, -18.7856750, 68.4834824, -87.6817703, 88.9044571
1: -50.9867439, 158.1197510, -49.8460236, 154.8673096, -205.8540497, 207.9657440
2: -76.5202637, 140.7139435, -74.7570572, 137.6546631, -214.1749268, 215.4709778
3: -43.8595810, 168.9619751, -42.9090462, 165.7720337, -209.6315918, 211.8710175
4: -69.9047928, 123.9398575, -68.2975235, 121.3659363, -191.2707214, 192.2373810

Time for backsubstitution: 0.90 seconds

### NS candidates at layer 1
type: A, layer: 1, pos: 41
type: A, layer: 1, pos: 14
type: B, layer: 1, pos: 14
type: B, layer: 1, pos: 19
type: A, layer: 1, pos: 39
type: A, layer: 1, pos: 19
type: A, layer: 1, pos: 25
type: B, layer: 1, pos: 39
type: B, layer: 1, pos: 12
type: B, layer: 1, pos: 25
type: A, layer: 1, pos: 12
type: A, layer: 1, pos: 8
type: B, layer: 1, pos: 37
type: A, layer: 1, pos: 37
type: B, layer: 1, pos: 20
type: A, layer: 1, pos: 20
type: B, layer: 1, pos: 29
type: B, layer: 1, pos: 41
type: A, layer: 1, pos: 29
type: B, layer: 1, pos: 8
type: A, layer: 1, pos: 11
type: A, layer: 1, pos: 46
type: A, layer: 1, pos: 21
type: A, layer: 1, pos: 6
type: B, layer: 1, pos: 6
type: B, layer: 1, pos: 4
type: B, layer: 1, pos: 21
type: B, layer: 1, pos: 15
type: A, layer: 1, pos: 15
type: B, layer: 1, pos: 42
type: A, layer: 1, pos: 4
type: A, layer: 1, pos: 38
type: B, layer: 1, pos: 46
type: B, layer: 1, pos: 13
type: B, layer: 1, pos: 23
type: A, layer: 1, pos: 23
type: B, layer: 1, pos: 16
type: A, layer: 1, pos: 16
type: B, layer: 1, pos: 38
type: A, layer: 1, pos: 42
type: A, layer: 1, pos: 31
type: B, layer: 1, pos: 31
type: B, layer: 1, pos: 11

Time for candidate selection: 0.08 seconds

### Candidate
type: A, layer: 1, pos: 41

### Candidate
type: A, layer: 1, pos: 14

### Candidate
type: B, layer: 1, pos: 14

### Candidate
type: B, layer: 1, pos: 19

### Candidate
type: A, layer: 1, pos: 39

### Candidate
type: A, layer: 1, pos: 19

### Candidate
type: A, layer: 1, pos: 25

### Candidate
type: B, layer: 1, pos: 39

### Candidate
type: B, layer: 1, pos: 12

### Candidate
type: B, layer: 1, pos: 25

### Candidate
type: A, layer: 1, pos: 12

### Candidate
type: A, layer: 1, pos: 8

### Candidate
type: B, layer: 1, pos: 37

### Candidate
type: A, layer: 1, pos: 37

### Candidate
type: B, layer: 1, pos: 20

### Candidate
type: A, layer: 1, pos: 20

### Candidate
type: B, layer: 1, pos: 29

### Candidate
type: B, layer: 1, pos: 41

### Candidate
type: A, layer: 1, pos: 29

### Candidate
type: B, layer: 1, pos: 8

### Candidate
type: A, layer: 1, pos: 11

### Candidate
type: A, layer: 1, pos: 46

### Candidate
type: A, layer: 1, pos: 21

### Candidate
type: A, layer: 1, pos: 6

### Candidate
type: B, layer: 1, pos: 6

### Candidate
type: B, layer: 1, pos: 4

### Candidate
type: B, layer: 1, pos: 21

### Candidate
type: B, layer: 1, pos: 15

### Candidate
type: A, layer: 1, pos: 15

### Candidate
type: B, layer: 1, pos: 42

### Candidate
type: A, layer: 1, pos: 4

### Candidate
type: A, layer: 1, pos: 38

### Candidate
type: B, layer: 1, pos: 46

### Candidate
type: B, layer: 1, pos: 13

### Candidate
type: B, layer: 1, pos: 23

### Candidate
type: A, layer: 1, pos: 23

### Candidate
type: B, layer: 1, pos: 16

### Candidate
type: A, layer: 1, pos: 16

### Candidate
type: B, layer: 1, pos: 38

### Candidate
type: A, layer: 1, pos: 42

### Candidate
type: A, layer: 1, pos: 31

## Relational analysis of NS_A2_A1_B1_A2_B1_A1_B2_B2_A1_A1
Optimization infeasible because this subproblem isn't reachable.

### Candidate
type: B, layer: 1, pos: 31

### Candidate
type: B, layer: 1, pos: 11

## BFS NS instance: NS_A2_A1_B1_A2_B1_A1_B2_B2_A2

### Backsubstitution after applying NS history:
0: -19.3761635, 70.6416855, -18.7856750, 68.4834824, -87.8596497, 89.4273605
1: -51.1780510, 160.0721283, -49.8460236, 154.8673096, -206.0453644, 209.9181366
2: -76.4557343, 142.3144379, -74.7570572, 137.6546631, -214.1103973, 217.0715027
3: -44.0854263, 170.8686676, -42.9090462, 165.7720337, -209.8574524, 213.7777100
4: -69.9495621, 125.3343735, -68.2975235, 121.3659363, -191.3154907, 193.6318665

Time for backsubstitution: 0.91 seconds

### NS candidates at layer 1
type: A, layer: 1, pos: 41
type: A, layer: 1, pos: 14
type: B, layer: 1, pos: 14
type: B, layer: 1, pos: 19
type: A, layer: 1, pos: 39
type: A, layer: 1, pos: 19
type: B, layer: 1, pos: 39
type: A, layer: 1, pos: 25
type: B, layer: 1, pos: 12
type: B, layer: 1, pos: 25
type: B, layer: 1, pos: 37
type: A, layer: 1, pos: 20
type: A, layer: 1, pos: 12
type: B, layer: 1, pos: 29
type: A, layer: 1, pos: 8
type: B, layer: 1, pos: 20
type: A, layer: 1, pos: 37
type: B, layer: 1, pos: 41
type: B, layer: 1, pos: 8
type: A, layer: 1, pos: 29
type: A, layer: 1, pos: 11
type: A, layer: 1, pos: 46
type: A, layer: 1, pos: 6
type: A, layer: 1, pos: 21
type: B, layer: 1, pos: 6
type: B, layer: 1, pos: 4
type: B, layer: 1, pos: 21
type: A, layer: 1, pos: 15
type: B, layer: 1, pos: 42
type: A, layer: 1, pos: 4
type: B, layer: 1, pos: 46
type: B, layer: 1, pos: 13
type: B, layer: 1, pos: 15
type: A, layer: 1, pos: 38
type: B, layer: 1, pos: 38
type: A, layer: 1, pos: 16
type: B, layer: 1, pos: 23
type: B, layer: 1, pos: 16
type: A, layer: 1, pos: 23
type: A, layer: 1, pos: 31
type: B, layer: 1, pos: 31
type: A, layer: 1, pos: 42
type: B, layer: 1, pos: 11

Time for candidate selection: 0.07 seconds

### Candidate
type: A, layer: 1, pos: 41

### Candidate
type: A, layer: 1, pos: 14

### Candidate
type: B, layer: 1, pos: 14

### Candidate
type: B, layer: 1, pos: 19

### Candidate
type: A, layer: 1, pos: 39

### Candidate
type: A, layer: 1, pos: 19

### Candidate
type: B, layer: 1, pos: 39

### Candidate
type: A, layer: 1, pos: 25

### Candidate
type: B, layer: 1, pos: 12

### Candidate
type: B, layer: 1, pos: 25

### Candidate
type: B, layer: 1, pos: 37

### Candidate
type: A, layer: 1, pos: 20

### Candidate
type: A, layer: 1, pos: 12

### Candidate
type: B, layer: 1, pos: 29

### Candidate
type: A, layer: 1, pos: 8

### Candidate
type: B, layer: 1, pos: 20

### Candidate
type: A, layer: 1, pos: 37

### Candidate
type: B, layer: 1, pos: 41

### Candidate
type: B, layer: 1, pos: 8

### Candidate
type: A, layer: 1, pos: 29

### Candidate
type: A, layer: 1, pos: 11

### Candidate
type: A, layer: 1, pos: 46

### Candidate
type: A, layer: 1, pos: 6

### Candidate
type: A, layer: 1, pos: 21

### Candidate
type: B, layer: 1, pos: 6

### Candidate
type: B, layer: 1, pos: 4

### Candidate
type: B, layer: 1, pos: 21

### Candidate
type: A, layer: 1, pos: 15

### Candidate
type: B, layer: 1, pos: 42

### Candidate
type: A, layer: 1, pos: 4

### Candidate
type: B, layer: 1, pos: 46

### Candidate
type: B, layer: 1, pos: 13

### Candidate
type: B, layer: 1, pos: 15

### Candidate
type: A, layer: 1, pos: 38

### Candidate
type: B, layer: 1, pos: 38

### Candidate
type: A, layer: 1, pos: 16

### Candidate
type: B, layer: 1, pos: 23

### Candidate
type: B, layer: 1, pos: 16

### Candidate
type: A, layer: 1, pos: 23

### Candidate
type: A, layer: 1, pos: 31

## Relational analysis of NS_A2_A1_B1_A2_B1_A1_B2_B2_A2_A1
Optimization infeasible because this subproblem isn't reachable.

### Candidate
type: B, layer: 1, pos: 31

### Candidate
type: A, layer: 1, pos: 42

### Candidate
type: B, layer: 1, pos: 11

## BFS NS instance: NS_A2_A1_B1_A2_B1_A2_A1_B1_B1

### Backsubstitution after applying NS history:
0: -21.6661549, 79.1037064, -18.3910942, 67.1361313, -88.8022842, 97.4947968
1: -57.1756744, 180.2160187, -48.9588051, 151.3595581, -208.5352173, 229.1748199
2: -85.4210587, 159.3976746, -73.7940826, 134.3392334, -219.7602844, 233.1917114
3: -49.1567116, 192.1778412, -42.1147232, 161.9170685, -211.0737762, 234.2925415
4: -78.1009827, 140.3668823, -67.3392639, 118.4117813, -196.5127563, 207.7061462

Time for backsubstitution: 0.89 seconds

### NS candidates at layer 1
type: A, layer: 1, pos: 41
type: B, layer: 1, pos: 19
type: B, layer: 1, pos: 41
type: B, layer: 1, pos: 21
type: A, layer: 1, pos: 6
type: B, layer: 1, pos: 6
type: A, layer: 1, pos: 46
type: A, layer: 1, pos: 25
type: B, layer: 1, pos: 25
type: A, layer: 1, pos: 19
type: A, layer: 1, pos: 11
type: A, layer: 1, pos: 4
type: A, layer: 1, pos: 38
type: B, layer: 1, pos: 14
type: B, layer: 1, pos: 12
type: A, layer: 1, pos: 12
type: B, layer: 1, pos: 38
type: B, layer: 1, pos: 13
type: A, layer: 1, pos: 21
type: B, layer: 1, pos: 4
type: B, layer: 1, pos: 29
type: A, layer: 1, pos: 29
type: B, layer: 1, pos: 39
type: B, layer: 1, pos: 20
type: B, layer: 1, pos: 46
type: A, layer: 1, pos: 31
type: A, layer: 1, pos: 15
type: B, layer: 1, pos: 31
type: A, layer: 1, pos: 14
type: B, layer: 1, pos: 23
type: B, layer: 1, pos: 16
type: B, layer: 1, pos: 37
type: A, layer: 1, pos: 39
type: B, layer: 1, pos: 15
type: A, layer: 1, pos: 8
type: A, layer: 1, pos: 42
type: B, layer: 1, pos: 8
type: A, layer: 1, pos: 20
type: A, layer: 1, pos: 22
type: A, layer: 1, pos: 23
type: B, layer: 1, pos: 42
type: A, layer: 1, pos: 37
type: B, layer: 1, pos: 11

Time for candidate selection: 0.07 seconds

### Candidate
type: A, layer: 1, pos: 41

### Candidate
type: B, layer: 1, pos: 19

### Candidate
type: B, layer: 1, pos: 41

### Candidate
type: B, layer: 1, pos: 21

### Candidate
type: A, layer: 1, pos: 6

### Candidate
type: B, layer: 1, pos: 6

### Candidate
type: A, layer: 1, pos: 46

### Candidate
type: A, layer: 1, pos: 25

### Candidate
type: B, layer: 1, pos: 25

### Candidate
type: A, layer: 1, pos: 19

### Candidate
type: A, layer: 1, pos: 11

### Candidate
type: A, layer: 1, pos: 4

### Candidate
type: A, layer: 1, pos: 38

### Candidate
type: B, layer: 1, pos: 14

### Candidate
type: B, layer: 1, pos: 12

### Candidate
type: A, layer: 1, pos: 12

### Candidate
type: B, layer: 1, pos: 38

### Candidate
type: B, layer: 1, pos: 13

### Candidate
type: A, layer: 1, pos: 21

### Candidate
type: B, layer: 1, pos: 4

### Candidate
type: B, layer: 1, pos: 29

### Candidate
type: A, layer: 1, pos: 29

### Candidate
type: B, layer: 1, pos: 39

### Candidate
type: B, layer: 1, pos: 20

### Candidate
type: B, layer: 1, pos: 46

### Candidate
type: A, layer: 1, pos: 31

### Candidate
type: A, layer: 1, pos: 15

### Candidate
type: B, layer: 1, pos: 31

### Candidate
type: A, layer: 1, pos: 14

### Candidate
type: B, layer: 1, pos: 23

### Candidate
type: B, layer: 1, pos: 16

### Candidate
type: B, layer: 1, pos: 37

### Candidate
type: A, layer: 1, pos: 39

### Candidate
type: B, layer: 1, pos: 15

### Candidate
type: A, layer: 1, pos: 8

## Relational analysis of NS_A2_A1_B1_A2_B1_A2_A1_B1_B1_A1
Optimization infeasible because this subproblem isn't reachable.

### Candidate
type: A, layer: 1, pos: 42

## Relational analysis of NS_A2_A1_B1_A2_B1_A2_A1_B1_B1_A1

### Relational analysis result of NS_A2_A1_B1_A2_B1_A2_A1_B1_B1_A1
Status: Status.VERIFIED
Output dim: 0, lower bound: -81.8604179, upper bound: 81.8608600
time: 0.85 seconds

## Relational analysis of NS_A2_A1_B1_A2_B1_A2_A1_B1_B1_A2

### Relational analysis result of NS_A2_A1_B1_A2_B1_A2_A1_B1_B1_A2
Status: Status.UNKNOWN
Output dim: 0, lower bound: -81.8605647, upper bound: 81.8611203
time: 1.32 seconds

## BFS NS instance: NS_A2_A1_B1_A2_B1_A2_A1_B1_B2

### Backsubstitution after applying NS history:
0: -21.6661549, 79.1037064, -18.5065250, 67.4602966, -89.1264496, 97.6102142
1: -57.1756744, 180.2160187, -49.3399620, 152.0112915, -209.1869659, 229.5559845
2: -85.4210587, 159.3976746, -74.4119873, 135.2107544, -220.6318054, 233.8096619
3: -49.1567116, 192.1778412, -42.4566727, 162.9008026, -212.0575104, 234.6344910
4: -78.1009827, 140.3668823, -67.8973923, 119.2207794, -197.3217621, 208.2642670

Time for backsubstitution: 0.89 seconds

### NS candidates at layer 1
type: A, layer: 1, pos: 41
type: B, layer: 1, pos: 19
type: B, layer: 1, pos: 14
type: A, layer: 1, pos: 14
type: B, layer: 1, pos: 39
type: A, layer: 1, pos: 39
type: A, layer: 1, pos: 25
type: B, layer: 1, pos: 12
type: B, layer: 1, pos: 25
type: B, layer: 1, pos: 37
type: A, layer: 1, pos: 19
type: A, layer: 1, pos: 12
type: B, layer: 1, pos: 29
type: A, layer: 1, pos: 8
type: A, layer: 1, pos: 20
type: B, layer: 1, pos: 20
type: A, layer: 1, pos: 11
type: A, layer: 1, pos: 37
type: B, layer: 1, pos: 8
type: A, layer: 1, pos: 46
type: B, layer: 1, pos: 41
type: A, layer: 1, pos: 29
type: B, layer: 1, pos: 21
type: A, layer: 1, pos: 6
type: B, layer: 1, pos: 6
type: A, layer: 1, pos: 4
type: A, layer: 1, pos: 15
type: A, layer: 1, pos: 21
type: B, layer: 1, pos: 4
type: A, layer: 1, pos: 38
type: B, layer: 1, pos: 13
type: B, layer: 1, pos: 42
type: B, layer: 1, pos: 15
type: B, layer: 1, pos: 23
type: A, layer: 1, pos: 42
type: B, layer: 1, pos: 38
type: A, layer: 1, pos: 23
type: B, layer: 1, pos: 46
type: A, layer: 1, pos: 31
type: B, layer: 1, pos: 31
type: A, layer: 1, pos: 22
type: B, layer: 1, pos: 16
type: B, layer: 1, pos: 11

Time for candidate selection: 0.08 seconds

### Candidate
type: A, layer: 1, pos: 41

### Candidate
type: B, layer: 1, pos: 19

### Candidate
type: B, layer: 1, pos: 14

### Candidate
type: A, layer: 1, pos: 14

### Candidate
type: B, layer: 1, pos: 39

### Candidate
type: A, layer: 1, pos: 39

### Candidate
type: A, layer: 1, pos: 25

### Candidate
type: B, layer: 1, pos: 12

### Candidate
type: B, layer: 1, pos: 25

### Candidate
type: B, layer: 1, pos: 37

### Candidate
type: A, layer: 1, pos: 19

### Candidate
type: A, layer: 1, pos: 12

### Candidate
type: B, layer: 1, pos: 29

### Candidate
type: A, layer: 1, pos: 8

## Relational analysis of NS_A2_A1_B1_A2_B1_A2_A1_B1_B2_A1
Optimization infeasible because this subproblem isn't reachable.

### Candidate
type: A, layer: 1, pos: 20

### Candidate
type: B, layer: 1, pos: 20

### Candidate
type: A, layer: 1, pos: 11

### Candidate
type: A, layer: 1, pos: 37

### Candidate
type: B, layer: 1, pos: 8

### Candidate
type: A, layer: 1, pos: 46

### Candidate
type: B, layer: 1, pos: 41

### Candidate
type: A, layer: 1, pos: 29

### Candidate
type: B, layer: 1, pos: 21

### Candidate
type: A, layer: 1, pos: 6

### Candidate
type: B, layer: 1, pos: 6

### Candidate
type: A, layer: 1, pos: 4

### Candidate
type: A, layer: 1, pos: 15

### Candidate
type: A, layer: 1, pos: 21

### Candidate
type: B, layer: 1, pos: 4

### Candidate
type: A, layer: 1, pos: 38

### Candidate
type: B, layer: 1, pos: 13

### Candidate
type: B, layer: 1, pos: 42

## Relational analysis of NS_A2_A1_B1_A2_B1_A2_A1_B1_B2_B1

### Relational analysis result of NS_A2_A1_B1_A2_B1_A2_A1_B1_B2_B1
Status: Status.UNKNOWN
Output dim: 0, lower bound: -81.8604397, upper bound: 81.8610556
time: 0.74 seconds

## Relational analysis of NS_A2_A1_B1_A2_B1_A2_A1_B1_B2_B2

### Relational analysis result of NS_A2_A1_B1_A2_B1_A2_A1_B1_B2_B2
Status: Status.UNKNOWN
Output dim: 0, lower bound: -81.8605647, upper bound: 81.8611203
time: 0.74 seconds

## BFS NS instance: NS_A2_A1_B1_A2_B1_A2_A1_B2_B1

### Backsubstitution after applying NS history:
0: -21.2976780, 77.7213821, -18.5815697, 67.7649078, -89.0625839, 96.3029480
1: -56.1462212, 177.2727966, -49.2043877, 153.4045715, -209.5507965, 226.4771881
2: -83.8302002, 156.6667328, -73.7736893, 136.1362610, -219.9664459, 230.4404144
3: -48.2794685, 189.0195312, -42.3743095, 163.9343567, -212.2138214, 231.3938293
4: -76.6548920, 137.9883423, -67.4200287, 119.9641190, -196.6190186, 205.4083405

Time for backsubstitution: 0.98 seconds

### NS candidates at layer 1
type: A, layer: 1, pos: 41
type: B, layer: 1, pos: 41
type: B, layer: 1, pos: 19
type: A, layer: 1, pos: 6
type: B, layer: 1, pos: 6
type: A, layer: 1, pos: 46
type: B, layer: 1, pos: 21
type: B, layer: 1, pos: 25
type: A, layer: 1, pos: 19
type: A, layer: 1, pos: 25
type: A, layer: 1, pos: 11
type: A, layer: 1, pos: 38
type: A, layer: 1, pos: 4
type: B, layer: 1, pos: 14
type: A, layer: 1, pos: 21
type: A, layer: 1, pos: 12
type: B, layer: 1, pos: 12
type: A, layer: 1, pos: 29
type: B, layer: 1, pos: 13
type: B, layer: 1, pos: 38
type: B, layer: 1, pos: 4
type: B, layer: 1, pos: 29
type: B, layer: 1, pos: 20
type: B, layer: 1, pos: 46
type: B, layer: 1, pos: 39
type: A, layer: 1, pos: 14
type: A, layer: 1, pos: 31
type: B, layer: 1, pos: 31
type: B, layer: 1, pos: 23
type: A, layer: 1, pos: 15
type: B, layer: 1, pos: 16
type: B, layer: 1, pos: 15
type: A, layer: 1, pos: 39
type: A, layer: 1, pos: 8
type: A, layer: 1, pos: 42
type: A, layer: 1, pos: 22
type: B, layer: 1, pos: 37
type: A, layer: 1, pos: 20
type: B, layer: 1, pos: 8
type: A, layer: 1, pos: 37
type: A, layer: 1, pos: 23
type: B, layer: 1, pos: 42
type: B, layer: 1, pos: 11

Time for candidate selection: 0.10 seconds

### Candidate
type: A, layer: 1, pos: 41

### Candidate
type: B, layer: 1, pos: 41

### Candidate
type: B, layer: 1, pos: 19

### Candidate
type: A, layer: 1, pos: 6

### Candidate
type: B, layer: 1, pos: 6

### Candidate
type: A, layer: 1, pos: 46

### Candidate
type: B, layer: 1, pos: 21

### Candidate
type: B, layer: 1, pos: 25

### Candidate
type: A, layer: 1, pos: 19

### Candidate
type: A, layer: 1, pos: 25

### Candidate
type: A, layer: 1, pos: 11

### Candidate
type: A, layer: 1, pos: 38

### Candidate
type: A, layer: 1, pos: 4

### Candidate
type: B, layer: 1, pos: 14

### Candidate
type: A, layer: 1, pos: 21

### Candidate
type: A, layer: 1, pos: 12

### Candidate
type: B, layer: 1, pos: 12

### Candidate
type: A, layer: 1, pos: 29

### Candidate
type: B, layer: 1, pos: 13

### Candidate
type: B, layer: 1, pos: 38

### Candidate
type: B, layer: 1, pos: 4

### Candidate
type: B, layer: 1, pos: 29

### Candidate
type: B, layer: 1, pos: 20

### Candidate
type: B, layer: 1, pos: 46

### Candidate
type: B, layer: 1, pos: 39

### Candidate
type: A, layer: 1, pos: 14

### Candidate
type: A, layer: 1, pos: 31

### Candidate
type: B, layer: 1, pos: 31

### Candidate
type: B, layer: 1, pos: 23

### Candidate
type: A, layer: 1, pos: 15

### Candidate
type: B, layer: 1, pos: 16

## Relational analysis of NS_A2_A1_B1_A2_B1_A2_A1_B2_B1_B1
Optimization infeasible because this subproblem isn't reachable.

### Candidate
type: B, layer: 1, pos: 15

### Candidate
type: A, layer: 1, pos: 39

### Candidate
type: A, layer: 1, pos: 8

## Relational analysis of NS_A2_A1_B1_A2_B1_A2_A1_B2_B1_A1
Optimization infeasible because this subproblem isn't reachable.

### Candidate
type: A, layer: 1, pos: 42

## Relational analysis of NS_A2_A1_B1_A2_B1_A2_A1_B2_B1_A1
Optimization infeasible because this subproblem isn't reachable.

### Candidate
type: A, layer: 1, pos: 22

## Relational analysis of NS_A2_A1_B1_A2_B1_A2_A1_B2_B1_A1

### Relational analysis result of NS_A2_A1_B1_A2_B1_A2_A1_B2_B1_A1
Status: Status.UNKNOWN
Output dim: 0, lower bound: -81.8604279, upper bound: 81.8610697
time: 0.85 seconds

## Relational analysis of NS_A2_A1_B1_A2_B1_A2_A1_B2_B1_A2

### Relational analysis result of NS_A2_A1_B1_A2_B1_A2_A1_B2_B1_A2
Status: Status.UNKNOWN
Output dim: 0, lower bound: -81.8604279, upper bound: 81.8610697
time: 0.80 seconds

## BFS NS instance: NS_A2_A1_B1_A2_B1_A2_A1_B2_B2

### Backsubstitution after applying NS history:
0: -21.2976780, 77.7213821, -18.5464497, 67.6095352, -88.9072113, 96.2678146
1: -56.1462212, 177.2727966, -49.2204208, 152.8471222, -208.9933319, 226.4932251
2: -83.8302002, 156.6667328, -73.8128738, 135.9040527, -219.7342529, 230.4795837
3: -48.2794685, 189.0195312, -42.3677864, 163.5677032, -211.8471680, 231.3872833
4: -76.6548920, 137.9883423, -67.4343491, 119.8176804, -196.4725647, 205.4226837

Time for backsubstitution: 0.98 seconds

### NS candidates at layer 1
type: A, layer: 1, pos: 41
type: B, layer: 1, pos: 19
type: A, layer: 1, pos: 14
type: B, layer: 1, pos: 14
type: A, layer: 1, pos: 39
type: B, layer: 1, pos: 39
type: A, layer: 1, pos: 25
type: B, layer: 1, pos: 12
type: B, layer: 1, pos: 25
type: A, layer: 1, pos: 19
type: B, layer: 1, pos: 37
type: A, layer: 1, pos: 12
type: A, layer: 1, pos: 8
type: B, layer: 1, pos: 29
type: A, layer: 1, pos: 20
type: B, layer: 1, pos: 20
type: A, layer: 1, pos: 11
type: A, layer: 1, pos: 37
type: B, layer: 1, pos: 8
type: A, layer: 1, pos: 46
type: A, layer: 1, pos: 29
type: B, layer: 1, pos: 41
type: A, layer: 1, pos: 6
type: B, layer: 1, pos: 21
type: B, layer: 1, pos: 6
type: A, layer: 1, pos: 4
type: A, layer: 1, pos: 15
type: A, layer: 1, pos: 21
type: A, layer: 1, pos: 38
type: B, layer: 1, pos: 4
type: B, layer: 1, pos: 15
type: B, layer: 1, pos: 13
type: B, layer: 1, pos: 42
type: B, layer: 1, pos: 23
type: A, layer: 1, pos: 42
type: B, layer: 1, pos: 38
type: A, layer: 1, pos: 23
type: B, layer: 1, pos: 46
type: A, layer: 1, pos: 31
type: A, layer: 1, pos: 22
type: B, layer: 1, pos: 31
type: B, layer: 1, pos: 16
type: B, layer: 1, pos: 11

Time for candidate selection: 0.08 seconds

### Candidate
type: A, layer: 1, pos: 41

### Candidate
type: B, layer: 1, pos: 19

### Candidate
type: A, layer: 1, pos: 14

### Candidate
type: B, layer: 1, pos: 14

### Candidate
type: A, layer: 1, pos: 39

### Candidate
type: B, layer: 1, pos: 39

### Candidate
type: A, layer: 1, pos: 25

### Candidate
type: B, layer: 1, pos: 12

### Candidate
type: B, layer: 1, pos: 25

### Candidate
type: A, layer: 1, pos: 19

### Candidate
type: B, layer: 1, pos: 37

### Candidate
type: A, layer: 1, pos: 12

### Candidate
type: A, layer: 1, pos: 8

## Relational analysis of NS_A2_A1_B1_A2_B1_A2_A1_B2_B2_A1
Optimization infeasible because this subproblem isn't reachable.

### Candidate
type: B, layer: 1, pos: 29

### Candidate
type: A, layer: 1, pos: 20

### Candidate
type: B, layer: 1, pos: 20

### Candidate
type: A, layer: 1, pos: 11

### Candidate
type: A, layer: 1, pos: 37

### Candidate
type: B, layer: 1, pos: 8

### Candidate
type: A, layer: 1, pos: 46

### Candidate
type: A, layer: 1, pos: 29

### Candidate
type: B, layer: 1, pos: 41

### Candidate
type: A, layer: 1, pos: 6

### Candidate
type: B, layer: 1, pos: 21

### Candidate
type: B, layer: 1, pos: 6

### Candidate
type: A, layer: 1, pos: 4

### Candidate
type: A, layer: 1, pos: 15

### Candidate
type: A, layer: 1, pos: 21

### Candidate
type: A, layer: 1, pos: 38

### Candidate
type: B, layer: 1, pos: 4

### Candidate
type: B, layer: 1, pos: 15

### Candidate
type: B, layer: 1, pos: 13

### Candidate
type: B, layer: 1, pos: 42

## Relational analysis of NS_A2_A1_B1_A2_B1_A2_A1_B2_B2_B1

### Relational analysis result of NS_A2_A1_B1_A2_B1_A2_A1_B2_B2_B1
Status: Status.UNKNOWN
Output dim: 0, lower bound: -81.8604245, upper bound: 81.8610423
time: 0.83 seconds

## Relational analysis of NS_A2_A1_B1_A2_B1_A2_A1_B2_B2_B2
Optimization infeasible because this subproblem isn't reachable.

### Candidate
type: B, layer: 1, pos: 23

### Candidate
type: A, layer: 1, pos: 42

### Candidate
type: B, layer: 1, pos: 38

### Candidate
type: A, layer: 1, pos: 23

### Candidate
type: B, layer: 1, pos: 46

### Candidate
type: A, layer: 1, pos: 31

### Candidate
type: A, layer: 1, pos: 22

## Relational analysis of NS_A2_A1_B1_A2_B1_A2_A1_B2_B2_A1

### Relational analysis result of NS_A2_A1_B1_A2_B1_A2_A1_B2_B2_A1
Status: Status.UNKNOWN
Output dim: 0, lower bound: -81.8604279, upper bound: 81.8610697
time: 0.74 seconds

## Relational analysis of NS_A2_A1_B1_A2_B1_A2_A1_B2_B2_A2

### Relational analysis result of NS_A2_A1_B1_A2_B1_A2_A1_B2_B2_A2
Status: Status.UNKNOWN
Output dim: 0, lower bound: -81.8604279, upper bound: 81.8610697
time: 0.82 seconds

## BFS NS instance: NS_A2_A1_B1_A2_B1_A2_A2_B1_B1

### Backsubstitution after applying NS history:
0: -22.6331539, 82.6661758, -18.4655514, 67.4126968, -90.0458527, 101.1317215
1: -59.7047958, 188.2940369, -49.1400375, 152.0275879, -211.7323608, 237.4340668
2: -89.2544403, 166.6350403, -74.0549774, 134.9131470, -224.1675873, 240.6900177
3: -51.3380013, 200.7817230, -42.2714348, 162.6137848, -213.9517670, 243.0531464
4: -81.5932159, 146.7880249, -67.5785675, 118.9201355, -200.5133514, 214.3665924

Time for backsubstitution: 0.89 seconds

### NS candidates at layer 1
type: A, layer: 1, pos: 41
type: B, layer: 1, pos: 41
type: B, layer: 1, pos: 19
type: A, layer: 1, pos: 19
type: A, layer: 1, pos: 6
type: B, layer: 1, pos: 6
type: B, layer: 1, pos: 21
type: A, layer: 1, pos: 46
type: A, layer: 1, pos: 25
type: B, layer: 1, pos: 25
type: A, layer: 1, pos: 38
type: A, layer: 1, pos: 4
type: B, layer: 1, pos: 14
type: A, layer: 1, pos: 21
type: B, layer: 1, pos: 16
type: B, layer: 1, pos: 4
type: B, layer: 1, pos: 12
type: A, layer: 1, pos: 11
type: B, layer: 1, pos: 13
type: B, layer: 1, pos: 20
type: A, layer: 1, pos: 12
type: A, layer: 1, pos: 29
type: B, layer: 1, pos: 39
type: B, layer: 1, pos: 29
type: B, layer: 1, pos: 46
type: B, layer: 1, pos: 38
type: B, layer: 1, pos: 31
type: A, layer: 1, pos: 14
type: A, layer: 1, pos: 31
type: B, layer: 1, pos: 15
type: A, layer: 1, pos: 15
type: B, layer: 1, pos: 23
type: A, layer: 1, pos: 22
type: A, layer: 1, pos: 8
type: A, layer: 1, pos: 39
type: B, layer: 1, pos: 37
type: B, layer: 1, pos: 11
type: A, layer: 1, pos: 42
type: A, layer: 1, pos: 23
type: B, layer: 1, pos: 8
type: B, layer: 1, pos: 42
type: A, layer: 1, pos: 37
type: A, layer: 1, pos: 20

Time for candidate selection: 0.08 seconds

### Candidate
type: A, layer: 1, pos: 41

### Candidate
type: B, layer: 1, pos: 41

### Candidate
type: B, layer: 1, pos: 19

### Candidate
type: A, layer: 1, pos: 19

### Candidate
type: A, layer: 1, pos: 6

### Candidate
type: B, layer: 1, pos: 6

### Candidate
type: B, layer: 1, pos: 21

### Candidate
type: A, layer: 1, pos: 46

### Candidate
type: A, layer: 1, pos: 25

### Candidate
type: B, layer: 1, pos: 25

### Candidate
type: A, layer: 1, pos: 38

### Candidate
type: A, layer: 1, pos: 4

### Candidate
type: B, layer: 1, pos: 14

### Candidate
type: A, layer: 1, pos: 21

### Candidate
type: B, layer: 1, pos: 16

### Candidate
type: B, layer: 1, pos: 4

### Candidate
type: B, layer: 1, pos: 12

### Candidate
type: A, layer: 1, pos: 11

### Candidate
type: B, layer: 1, pos: 13

### Candidate
type: B, layer: 1, pos: 20

### Candidate
type: A, layer: 1, pos: 12

### Candidate
type: A, layer: 1, pos: 29

### Candidate
type: B, layer: 1, pos: 39

### Candidate
type: B, layer: 1, pos: 29

### Candidate
type: B, layer: 1, pos: 46

### Candidate
type: B, layer: 1, pos: 38

### Candidate
type: B, layer: 1, pos: 31

### Candidate
type: A, layer: 1, pos: 14

### Candidate
type: A, layer: 1, pos: 31

### Candidate
type: B, layer: 1, pos: 15

### Candidate
type: A, layer: 1, pos: 15

### Candidate
type: B, layer: 1, pos: 23

### Candidate
type: A, layer: 1, pos: 22

## Relational analysis of NS_A2_A1_B1_A2_B1_A2_A2_B1_B1_A1

### Relational analysis result of NS_A2_A1_B1_A2_B1_A2_A2_B1_B1_A1
Status: Status.UNKNOWN
Output dim: 0, lower bound: -81.8629228, upper bound: 81.8629875
time: 0.85 seconds

## Relational analysis of NS_A2_A1_B1_A2_B1_A2_A2_B1_B1_A2

### Relational analysis result of NS_A2_A1_B1_A2_B1_A2_A2_B1_B1_A2
Status: Status.UNKNOWN
Output dim: 0, lower bound: -81.8629228, upper bound: 81.8629875
time: 0.82 seconds

## BFS NS instance: NS_A2_A1_B1_A2_B1_A2_A2_B1_B2

### Backsubstitution after applying NS history:
0: -22.2672043, 81.2917252, -18.6594925, 68.0497360, -90.3169327, 99.9512100
1: -58.6808510, 185.3720551, -49.3909416, 154.1109314, -212.7917786, 234.7630005
2: -87.6690369, 163.9244537, -74.0428772, 136.7345276, -224.4035645, 237.9672852
3: -50.4653702, 197.6452942, -42.5357437, 164.6624908, -215.1278687, 240.1810150
4: -80.1520844, 144.4278717, -67.6671448, 120.4946899, -200.6467743, 212.0950165

Time for backsubstitution: 1.00 seconds

### NS candidates at layer 1
type: A, layer: 1, pos: 41
type: B, layer: 1, pos: 41
type: B, layer: 1, pos: 19
type: A, layer: 1, pos: 19
type: A, layer: 1, pos: 6
type: B, layer: 1, pos: 6
type: A, layer: 1, pos: 46
type: B, layer: 1, pos: 21
type: B, layer: 1, pos: 25
type: A, layer: 1, pos: 25
type: A, layer: 1, pos: 38
type: A, layer: 1, pos: 4
type: A, layer: 1, pos: 21
type: B, layer: 1, pos: 16
type: B, layer: 1, pos: 14
type: A, layer: 1, pos: 11
type: A, layer: 1, pos: 12
type: A, layer: 1, pos: 29
type: B, layer: 1, pos: 4
type: B, layer: 1, pos: 20
type: B, layer: 1, pos: 12
type: B, layer: 1, pos: 13
type: B, layer: 1, pos: 46
type: B, layer: 1, pos: 38
type: A, layer: 1, pos: 14
type: B, layer: 1, pos: 29
type: B, layer: 1, pos: 39
type: B, layer: 1, pos: 15
type: B, layer: 1, pos: 31
type: A, layer: 1, pos: 31
type: B, layer: 1, pos: 23
type: A, layer: 1, pos: 22
type: A, layer: 1, pos: 8
type: A, layer: 1, pos: 39
type: A, layer: 1, pos: 15
type: A, layer: 1, pos: 42
type: B, layer: 1, pos: 11
type: A, layer: 1, pos: 37
type: B, layer: 1, pos: 37
type: A, layer: 1, pos: 23
type: B, layer: 1, pos: 8
type: B, layer: 1, pos: 42
type: A, layer: 1, pos: 20

Time for candidate selection: 0.08 seconds

### Candidate
type: A, layer: 1, pos: 41

### Candidate
type: B, layer: 1, pos: 41

### Candidate
type: B, layer: 1, pos: 19

### Candidate
type: A, layer: 1, pos: 19

### Candidate
type: A, layer: 1, pos: 6

### Candidate
type: B, layer: 1, pos: 6

### Candidate
type: A, layer: 1, pos: 46

### Candidate
type: B, layer: 1, pos: 21

### Candidate
type: B, layer: 1, pos: 25

### Candidate
type: A, layer: 1, pos: 25

### Candidate
type: A, layer: 1, pos: 38

### Candidate
type: A, layer: 1, pos: 4

### Candidate
type: A, layer: 1, pos: 21

### Candidate
type: B, layer: 1, pos: 16

### Candidate
type: B, layer: 1, pos: 14

### Candidate
type: A, layer: 1, pos: 11

### Candidate
type: A, layer: 1, pos: 12

### Candidate
type: A, layer: 1, pos: 29

### Candidate
type: B, layer: 1, pos: 4

### Candidate
type: B, layer: 1, pos: 20

### Candidate
type: B, layer: 1, pos: 12

### Candidate
type: B, layer: 1, pos: 13

### Candidate
type: B, layer: 1, pos: 46

### Candidate
type: B, layer: 1, pos: 38

### Candidate
type: A, layer: 1, pos: 14

### Candidate
type: B, layer: 1, pos: 29

### Candidate
type: B, layer: 1, pos: 39

### Candidate
type: B, layer: 1, pos: 15

### Candidate
type: B, layer: 1, pos: 31

### Candidate
type: A, layer: 1, pos: 31

### Candidate
type: B, layer: 1, pos: 23

### Candidate
type: A, layer: 1, pos: 22

## Relational analysis of NS_A2_A1_B1_A2_B1_A2_A2_B1_B2_A1
Optimization infeasible because this subproblem isn't reachable.

### Candidate
type: A, layer: 1, pos: 8

### Candidate
type: A, layer: 1, pos: 39

### Candidate
type: A, layer: 1, pos: 15

### Candidate
type: A, layer: 1, pos: 42

### Candidate
type: B, layer: 1, pos: 11

### Candidate
type: A, layer: 1, pos: 37

### Candidate
type: B, layer: 1, pos: 37

### Candidate
type: A, layer: 1, pos: 23

### Candidate
type: B, layer: 1, pos: 8

### Candidate
type: B, layer: 1, pos: 42

### Candidate
type: A, layer: 1, pos: 20

## BFS NS instance: NS_A2_A1_B1_A2_B1_A2_A2_B2_B1

### Backsubstitution after applying NS history:
0: -22.6331539, 82.6661758, -18.5787945, 67.7266159, -90.3597717, 101.2449722
1: -59.7047958, 188.2940369, -49.5185394, 152.6435699, -212.3483582, 237.8125763
2: -89.2544403, 166.6350403, -74.6734467, 135.7630005, -225.0174408, 241.3084717
3: -51.3380013, 200.7817230, -42.6107178, 163.5640869, -214.9020844, 243.3924408
4: -81.5932159, 146.7880249, -68.1391220, 119.7065964, -201.2997742, 214.9271545

Time for backsubstitution: 1.06 seconds

### NS candidates at layer 1
type: A, layer: 1, pos: 41
type: B, layer: 1, pos: 19
type: A, layer: 1, pos: 14
type: B, layer: 1, pos: 14
type: B, layer: 1, pos: 39
type: A, layer: 1, pos: 39
type: A, layer: 1, pos: 25
type: A, layer: 1, pos: 19
type: B, layer: 1, pos: 12
type: B, layer: 1, pos: 25
type: B, layer: 1, pos: 37
type: A, layer: 1, pos: 8
type: A, layer: 1, pos: 12
type: B, layer: 1, pos: 20
type: B, layer: 1, pos: 29
type: A, layer: 1, pos: 20
type: A, layer: 1, pos: 37
type: B, layer: 1, pos: 8
type: B, layer: 1, pos: 41
type: A, layer: 1, pos: 11
type: A, layer: 1, pos: 29
type: A, layer: 1, pos: 46
type: A, layer: 1, pos: 6
type: B, layer: 1, pos: 21
type: B, layer: 1, pos: 6
type: A, layer: 1, pos: 21
type: A, layer: 1, pos: 4
type: B, layer: 1, pos: 4
type: B, layer: 1, pos: 15
type: A, layer: 1, pos: 38
type: A, layer: 1, pos: 15
type: B, layer: 1, pos: 13
type: B, layer: 1, pos: 42
type: B, layer: 1, pos: 16
type: A, layer: 1, pos: 23
type: A, layer: 1, pos: 22
type: B, layer: 1, pos: 23
type: A, layer: 1, pos: 42
type: B, layer: 1, pos: 46
type: B, layer: 1, pos: 38
type: A, layer: 1, pos: 31
type: B, layer: 1, pos: 31
type: B, layer: 1, pos: 11

Time for candidate selection: 0.09 seconds

### Candidate
type: A, layer: 1, pos: 41

### Candidate
type: B, layer: 1, pos: 19

### Candidate
type: A, layer: 1, pos: 14

### Candidate
type: B, layer: 1, pos: 14

### Candidate
type: B, layer: 1, pos: 39

### Candidate
type: A, layer: 1, pos: 39

### Candidate
type: A, layer: 1, pos: 25

### Candidate
type: A, layer: 1, pos: 19

### Candidate
type: B, layer: 1, pos: 12

### Candidate
type: B, layer: 1, pos: 25

### Candidate
type: B, layer: 1, pos: 37

### Candidate
type: A, layer: 1, pos: 8

### Candidate
type: A, layer: 1, pos: 12

### Candidate
type: B, layer: 1, pos: 20

### Candidate
type: B, layer: 1, pos: 29

### Candidate
type: A, layer: 1, pos: 20

### Candidate
type: A, layer: 1, pos: 37

### Candidate
type: B, layer: 1, pos: 8

### Candidate
type: B, layer: 1, pos: 41

### Candidate
type: A, layer: 1, pos: 11

### Candidate
type: A, layer: 1, pos: 29

### Candidate
type: A, layer: 1, pos: 46

### Candidate
type: A, layer: 1, pos: 6

### Candidate
type: B, layer: 1, pos: 21

### Candidate
type: B, layer: 1, pos: 6

### Candidate
type: A, layer: 1, pos: 21

### Candidate
type: A, layer: 1, pos: 4

### Candidate
type: B, layer: 1, pos: 4

### Candidate
type: B, layer: 1, pos: 15

### Candidate
type: A, layer: 1, pos: 38

### Candidate
type: A, layer: 1, pos: 15

### Candidate
type: B, layer: 1, pos: 13

### Candidate
type: B, layer: 1, pos: 42

## Relational analysis of NS_A2_A1_B1_A2_B1_A2_A2_B2_B1_B1
Optimization infeasible because this subproblem isn't reachable.

### Candidate
type: B, layer: 1, pos: 16

### Candidate
type: A, layer: 1, pos: 23

### Candidate
type: A, layer: 1, pos: 22

## Relational analysis of NS_A2_A1_B1_A2_B1_A2_A2_B2_B1_A1

### Relational analysis result of NS_A2_A1_B1_A2_B1_A2_A2_B2_B1_A1
Status: Status.UNKNOWN
Output dim: 0, lower bound: -81.8603983, upper bound: 81.8610642
time: 0.86 seconds

## Relational analysis of NS_A2_A1_B1_A2_B1_A2_A2_B2_B1_A2

### Relational analysis result of NS_A2_A1_B1_A2_B1_A2_A2_B2_B1_A2
Status: Status.UNKNOWN
Output dim: 0, lower bound: -81.8604279, upper bound: 81.8610642
time: 0.97 seconds

## BFS NS instance: NS_A2_A1_B1_A2_B1_A2_A2_B2_B2

### Backsubstitution after applying NS history:
0: -22.2672043, 81.2917252, -18.6771259, 68.0861969, -90.3534012, 99.9688492
1: -58.6808510, 185.3720551, -49.5544319, 153.9679413, -212.6487732, 234.9264832
2: -87.6690369, 163.9244537, -74.3130569, 136.8685760, -224.5376129, 238.2374878
3: -50.4653702, 197.6452942, -42.6558762, 164.7959595, -215.2613220, 240.3011322
4: -80.1520844, 144.4278717, -67.8917389, 120.6695709, -200.8216553, 212.3195953

Time for backsubstitution: 0.94 seconds

### NS candidates at layer 1
type: A, layer: 1, pos: 41
type: A, layer: 1, pos: 14
type: B, layer: 1, pos: 19
type: B, layer: 1, pos: 14
type: B, layer: 1, pos: 39
type: A, layer: 1, pos: 39
type: B, layer: 1, pos: 12
type: A, layer: 1, pos: 19
type: A, layer: 1, pos: 12
type: A, layer: 1, pos: 8
type: B, layer: 1, pos: 37
type: A, layer: 1, pos: 25
type: B, layer: 1, pos: 29
type: A, layer: 1, pos: 37
type: B, layer: 1, pos: 8
type: A, layer: 1, pos: 29
type: B, layer: 1, pos: 25
type: A, layer: 1, pos: 46
type: A, layer: 1, pos: 11
type: B, layer: 1, pos: 20
type: B, layer: 1, pos: 21
type: B, layer: 1, pos: 41
type: A, layer: 1, pos: 21
type: A, layer: 1, pos: 20
type: A, layer: 1, pos: 6
type: B, layer: 1, pos: 15
type: B, layer: 1, pos: 6
type: A, layer: 1, pos: 15
type: A, layer: 1, pos: 38
type: A, layer: 1, pos: 4
type: B, layer: 1, pos: 13
type: A, layer: 1, pos: 22
type: B, layer: 1, pos: 16
type: B, layer: 1, pos: 42
type: B, layer: 1, pos: 23
type: B, layer: 1, pos: 4
type: B, layer: 1, pos: 46
type: A, layer: 1, pos: 23
type: B, layer: 1, pos: 38
type: A, layer: 1, pos: 31
type: B, layer: 1, pos: 31
type: A, layer: 1, pos: 42
type: B, layer: 1, pos: 11

Time for candidate selection: 0.08 seconds

### Candidate
type: A, layer: 1, pos: 41

### Candidate
type: A, layer: 1, pos: 14

### Candidate
type: B, layer: 1, pos: 19

### Candidate
type: B, layer: 1, pos: 14

### Candidate
type: B, layer: 1, pos: 39

### Candidate
type: A, layer: 1, pos: 39

### Candidate
type: B, layer: 1, pos: 12

### Candidate
type: A, layer: 1, pos: 19

### Candidate
type: A, layer: 1, pos: 12

### Candidate
type: A, layer: 1, pos: 8

### Candidate
type: B, layer: 1, pos: 37

### Candidate
type: A, layer: 1, pos: 25

### Candidate
type: B, layer: 1, pos: 29

### Candidate
type: A, layer: 1, pos: 37

### Candidate
type: B, layer: 1, pos: 8

### Candidate
type: A, layer: 1, pos: 29

### Candidate
type: B, layer: 1, pos: 25

### Candidate
type: A, layer: 1, pos: 46

### Candidate
type: A, layer: 1, pos: 11

### Candidate
type: B, layer: 1, pos: 20

### Candidate
type: B, layer: 1, pos: 21

### Candidate
type: B, layer: 1, pos: 41

### Candidate
type: A, layer: 1, pos: 21

### Candidate
type: A, layer: 1, pos: 20

### Candidate
type: A, layer: 1, pos: 6

### Candidate
type: B, layer: 1, pos: 15

### Candidate
type: B, layer: 1, pos: 6

### Candidate
type: A, layer: 1, pos: 15

### Candidate
type: A, layer: 1, pos: 38

### Candidate
type: A, layer: 1, pos: 4

### Candidate
type: B, layer: 1, pos: 13

### Candidate
type: A, layer: 1, pos: 22

## Relational analysis of NS_A2_A1_B1_A2_B1_A2_A2_B2_B2_A1

### Relational analysis result of NS_A2_A1_B1_A2_B1_A2_A2_B2_B2_A1
Status: Status.UNKNOWN
Output dim: 0, lower bound: -81.8604279, upper bound: 81.8610642
time: 0.74 seconds

## Relational analysis of NS_A2_A1_B1_A2_B1_A2_A2_B2_B2_A2

### Relational analysis result of NS_A2_A1_B1_A2_B1_A2_A2_B2_B2_A2
Status: Status.UNKNOWN
Output dim: 0, lower bound: -81.8604279, upper bound: 81.8610642
time: 0.92 seconds

## BFS NS instance: NS_A2_A1_B1_A2_B2_A1_B1_B1_A1

### Backsubstitution after applying NS history:
0: -18.9003239, 69.0304260, -22.7525654, 83.2576599, -102.1579819, 91.7829895
1: -50.1970177, 155.6475983, -60.1493530, 189.8360443, -240.0330505, 215.7969360
2: -75.3415909, 138.5106201, -90.1654434, 166.8709564, -242.2125397, 228.6760559
3: -43.1801605, 166.3397522, -51.7128868, 202.6529388, -245.8330536, 218.0526428
4: -68.8156433, 122.0141144, -82.3338470, 147.1033783, -215.9189911, 204.3479309

Time for backsubstitution: 1.06 seconds

### NS candidates at layer 1
type: B, layer: 1, pos: 41
type: A, layer: 1, pos: 20
type: A, layer: 1, pos: 19
type: B, layer: 1, pos: 20
type: A, layer: 1, pos: 39
type: B, layer: 1, pos: 39
type: A, layer: 1, pos: 41
type: B, layer: 1, pos: 19
type: B, layer: 1, pos: 25
type: A, layer: 1, pos: 14
type: A, layer: 1, pos: 25
type: B, layer: 1, pos: 14
type: B, layer: 1, pos: 23
type: A, layer: 1, pos: 23
type: A, layer: 1, pos: 6
type: A, layer: 1, pos: 21
type: B, layer: 1, pos: 6
type: B, layer: 1, pos: 46
type: B, layer: 1, pos: 21
type: B, layer: 1, pos: 4
type: B, layer: 1, pos: 15
type: B, layer: 1, pos: 13
type: A, layer: 1, pos: 29
type: A, layer: 1, pos: 38
type: B, layer: 1, pos: 16
type: A, layer: 1, pos: 15
type: B, layer: 1, pos: 11
type: A, layer: 1, pos: 46
type: B, layer: 1, pos: 12
type: A, layer: 1, pos: 4
type: A, layer: 1, pos: 37
type: A, layer: 1, pos: 12
type: B, layer: 1, pos: 38
type: A, layer: 1, pos: 16
type: B, layer: 1, pos: 29
type: B, layer: 1, pos: 8
type: A, layer: 1, pos: 31
type: B, layer: 1, pos: 31
type: A, layer: 1, pos: 42
type: B, layer: 1, pos: 22
type: B, layer: 1, pos: 37
type: A, layer: 1, pos: 8
type: A, layer: 1, pos: 11

Time for candidate selection: 0.08 seconds

### Candidate
type: B, layer: 1, pos: 41

### Candidate
type: A, layer: 1, pos: 20

### Candidate
type: A, layer: 1, pos: 19

### Candidate
type: B, layer: 1, pos: 20

### Candidate
type: A, layer: 1, pos: 39

### Candidate
type: B, layer: 1, pos: 39

### Candidate
type: A, layer: 1, pos: 41

### Candidate
type: B, layer: 1, pos: 19

### Candidate
type: B, layer: 1, pos: 25

### Candidate
type: A, layer: 1, pos: 14

### Candidate
type: A, layer: 1, pos: 25

### Candidate
type: B, layer: 1, pos: 14

### Candidate
type: B, layer: 1, pos: 23

### Candidate
type: A, layer: 1, pos: 23

### Candidate
type: A, layer: 1, pos: 6

### Candidate
type: A, layer: 1, pos: 21

### Candidate
type: B, layer: 1, pos: 6

### Candidate
type: B, layer: 1, pos: 46

### Candidate
type: B, layer: 1, pos: 21

### Candidate
type: B, layer: 1, pos: 4

### Candidate
type: B, layer: 1, pos: 15

### Candidate
type: B, layer: 1, pos: 13

### Candidate
type: A, layer: 1, pos: 29

### Candidate
type: A, layer: 1, pos: 38

### Candidate
type: B, layer: 1, pos: 16

### Candidate
type: A, layer: 1, pos: 15

### Candidate
type: B, layer: 1, pos: 11

### Candidate
type: A, layer: 1, pos: 46

### Candidate
type: B, layer: 1, pos: 12

### Candidate
type: A, layer: 1, pos: 4

### Candidate
type: A, layer: 1, pos: 37

### Candidate
type: A, layer: 1, pos: 12

### Candidate
type: B, layer: 1, pos: 38

### Candidate
type: A, layer: 1, pos: 16

### Candidate
type: B, layer: 1, pos: 29

### Candidate
type: B, layer: 1, pos: 8

### Candidate
type: A, layer: 1, pos: 31

### Candidate
type: B, layer: 1, pos: 31

### Candidate
type: A, layer: 1, pos: 42

## Relational analysis of NS_A2_A1_B1_A2_B2_A1_B1_B1_A1_A1

### Relational analysis result of NS_A2_A1_B1_A2_B2_A1_B1_B1_A1_A1
Status: Status.UNKNOWN
Output dim: 0, lower bound: -81.8626243, upper bound: 81.8625774
time: 0.85 seconds

## Relational analysis of NS_A2_A1_B1_A2_B2_A1_B1_B1_A1_A2

### Relational analysis result of NS_A2_A1_B1_A2_B2_A1_B1_B1_A1_A2
Status: Status.UNKNOWN
Output dim: 0, lower bound: -81.8626243, upper bound: 81.8625774
time: 1.09 seconds

## BFS NS instance: NS_A2_A1_B1_A2_B2_A1_B1_B1_A2

### Backsubstitution after applying NS history:
0: -19.1695518, 69.8992538, -22.3819027, 81.8603592, -101.0299072, 92.2811584
1: -50.6409225, 158.3228149, -59.1157532, 186.8670502, -237.5079651, 217.4385529
2: -75.6577301, 140.8316345, -88.5823746, 164.1040192, -239.7617493, 229.4140015
3: -43.6225853, 169.0185852, -50.8336487, 199.4884644, -243.1110535, 219.8522186
4: -69.2205124, 124.0199509, -80.8856812, 144.7041016, -213.9246216, 204.9056396

Time for backsubstitution: 1.06 seconds

### NS candidates at layer 1
type: B, layer: 1, pos: 41
type: A, layer: 1, pos: 19
type: A, layer: 1, pos: 41
type: B, layer: 1, pos: 6
type: A, layer: 1, pos: 6
type: B, layer: 1, pos: 19
type: B, layer: 1, pos: 25
type: A, layer: 1, pos: 21
type: B, layer: 1, pos: 46
type: A, layer: 1, pos: 25
type: B, layer: 1, pos: 21
type: B, layer: 1, pos: 4
type: B, layer: 1, pos: 11
type: A, layer: 1, pos: 38
type: B, layer: 1, pos: 38
type: B, layer: 1, pos: 13
type: B, layer: 1, pos: 12
type: A, layer: 1, pos: 46
type: A, layer: 1, pos: 14
type: A, layer: 1, pos: 29
type: A, layer: 1, pos: 4
type: A, layer: 1, pos: 12
type: B, layer: 1, pos: 14
type: B, layer: 1, pos: 29
type: A, layer: 1, pos: 20
type: A, layer: 1, pos: 31
type: B, layer: 1, pos: 31
type: B, layer: 1, pos: 39
type: B, layer: 1, pos: 16
type: B, layer: 1, pos: 8
type: A, layer: 1, pos: 39
type: B, layer: 1, pos: 15
type: A, layer: 1, pos: 15
type: A, layer: 1, pos: 16
type: A, layer: 1, pos: 23
type: B, layer: 1, pos: 22
type: A, layer: 1, pos: 37
type: B, layer: 1, pos: 20
type: A, layer: 1, pos: 42
type: B, layer: 1, pos: 23
type: A, layer: 1, pos: 11
type: B, layer: 1, pos: 37
type: A, layer: 1, pos: 8

Time for candidate selection: 0.08 seconds

### Candidate
type: B, layer: 1, pos: 41

### Candidate
type: A, layer: 1, pos: 19

### Candidate
type: A, layer: 1, pos: 41

### Candidate
type: B, layer: 1, pos: 6

### Candidate
type: A, layer: 1, pos: 6

### Candidate
type: B, layer: 1, pos: 19

### Candidate
type: B, layer: 1, pos: 25

### Candidate
type: A, layer: 1, pos: 21

### Candidate
type: B, layer: 1, pos: 46

### Candidate
type: A, layer: 1, pos: 25

### Candidate
type: B, layer: 1, pos: 21

### Candidate
type: B, layer: 1, pos: 4

### Candidate
type: B, layer: 1, pos: 11

### Candidate
type: A, layer: 1, pos: 38

### Candidate
type: B, layer: 1, pos: 38

### Candidate
type: B, layer: 1, pos: 13

### Candidate
type: B, layer: 1, pos: 12

### Candidate
type: A, layer: 1, pos: 46

### Candidate
type: A, layer: 1, pos: 14

### Candidate
type: A, layer: 1, pos: 29

### Candidate
type: A, layer: 1, pos: 4

### Candidate
type: A, layer: 1, pos: 12

### Candidate
type: B, layer: 1, pos: 14

### Candidate
type: B, layer: 1, pos: 29

### Candidate
type: A, layer: 1, pos: 20

### Candidate
type: A, layer: 1, pos: 31

### Candidate
type: B, layer: 1, pos: 31

### Candidate
type: B, layer: 1, pos: 39

### Candidate
type: B, layer: 1, pos: 16

### Candidate
type: B, layer: 1, pos: 8

### Candidate
type: A, layer: 1, pos: 39

### Candidate
type: B, layer: 1, pos: 15

### Candidate
type: A, layer: 1, pos: 15

### Candidate
type: A, layer: 1, pos: 16

### Candidate
type: A, layer: 1, pos: 23

### Candidate
type: B, layer: 1, pos: 22

## Relational analysis of NS_A2_A1_B1_A2_B2_A1_B1_B1_A2_B1
Optimization infeasible because this subproblem isn't reachable.

### Candidate
type: A, layer: 1, pos: 37

### Candidate
type: B, layer: 1, pos: 20

### Candidate
type: A, layer: 1, pos: 42

## Relational analysis of NS_A2_A1_B1_A2_B2_A1_B1_B1_A2_A1

### Relational analysis result of NS_A2_A1_B1_A2_B2_A1_B1_B1_A2_A1
Status: Status.UNKNOWN
Output dim: 0, lower bound: -81.8625776, upper bound: 81.8626573
time: 0.91 seconds

## Relational analysis of NS_A2_A1_B1_A2_B2_A1_B1_B1_A2_A2

### Relational analysis result of NS_A2_A1_B1_A2_B2_A1_B1_B1_A2_A2
Status: Status.UNKNOWN
Output dim: 0, lower bound: -81.8625776, upper bound: 81.8628393
time: 0.92 seconds

## BFS NS instance: NS_A2_A1_B1_A2_B2_A1_B2_B1_B1

### Backsubstitution after applying NS history:
0: -19.2562885, 70.3274536, -22.3502445, 81.6966095, -100.9528961, 92.6776962
1: -51.1238251, 158.6177673, -59.1535072, 186.1858215, -237.3096466, 217.7712402
2: -76.6867676, 141.2121277, -88.7366409, 163.8814545, -240.5682220, 229.9487610
3: -43.9839020, 169.4877319, -50.8682022, 198.9301300, -242.9140320, 220.3559113
4: -70.0636826, 124.3861465, -81.0366821, 144.4750519, -214.5387268, 205.4228210

Time for backsubstitution: 1.17 seconds

### NS candidates at layer 1
type: A, layer: 1, pos: 41
type: A, layer: 1, pos: 14
type: A, layer: 1, pos: 19
type: A, layer: 1, pos: 39
type: B, layer: 1, pos: 14
type: B, layer: 1, pos: 12
type: A, layer: 1, pos: 25
type: B, layer: 1, pos: 25
type: B, layer: 1, pos: 39
type: A, layer: 1, pos: 20
type: B, layer: 1, pos: 19
type: A, layer: 1, pos: 37
type: A, layer: 1, pos: 12
type: B, layer: 1, pos: 41
type: A, layer: 1, pos: 8
type: B, layer: 1, pos: 37
type: B, layer: 1, pos: 29
type: B, layer: 1, pos: 8
type: A, layer: 1, pos: 29
type: A, layer: 1, pos: 21
type: B, layer: 1, pos: 20
type: A, layer: 1, pos: 6
type: B, layer: 1, pos: 6
type: A, layer: 1, pos: 46
type: B, layer: 1, pos: 4
type: A, layer: 1, pos: 11
type: B, layer: 1, pos: 46
type: B, layer: 1, pos: 15
type: B, layer: 1, pos: 13
type: A, layer: 1, pos: 4
type: A, layer: 1, pos: 15
type: A, layer: 1, pos: 22
type: A, layer: 1, pos: 38
type: B, layer: 1, pos: 11
type: B, layer: 1, pos: 16
type: B, layer: 1, pos: 23
type: A, layer: 1, pos: 23
type: B, layer: 1, pos: 38
type: B, layer: 1, pos: 21
type: A, layer: 1, pos: 16
type: A, layer: 1, pos: 31
type: B, layer: 1, pos: 31
type: A, layer: 1, pos: 42

Time for candidate selection: 0.10 seconds

### Candidate
type: A, layer: 1, pos: 41

### Candidate
type: A, layer: 1, pos: 14

### Candidate
type: A, layer: 1, pos: 19

### Candidate
type: A, layer: 1, pos: 39

### Candidate
type: B, layer: 1, pos: 14

### Candidate
type: B, layer: 1, pos: 12

### Candidate
type: A, layer: 1, pos: 25

### Candidate
type: B, layer: 1, pos: 25

### Candidate
type: B, layer: 1, pos: 39

### Candidate
type: A, layer: 1, pos: 20

### Candidate
type: B, layer: 1, pos: 19

### Candidate
type: A, layer: 1, pos: 37

### Candidate
type: A, layer: 1, pos: 12

### Candidate
type: B, layer: 1, pos: 41

### Candidate
type: A, layer: 1, pos: 8

### Candidate
type: B, layer: 1, pos: 37

### Candidate
type: B, layer: 1, pos: 29

### Candidate
type: B, layer: 1, pos: 8

### Candidate
type: A, layer: 1, pos: 29

### Candidate
type: A, layer: 1, pos: 21

### Candidate
type: B, layer: 1, pos: 20

### Candidate
type: A, layer: 1, pos: 6

### Candidate
type: B, layer: 1, pos: 6

### Candidate
type: A, layer: 1, pos: 46

### Candidate
type: B, layer: 1, pos: 4

### Candidate
type: A, layer: 1, pos: 11

### Candidate
type: B, layer: 1, pos: 46

### Candidate
type: B, layer: 1, pos: 15

## Relational analysis of NS_A2_A1_B1_A2_B2_A1_B2_B1_B1_B1
Optimization infeasible because this subproblem isn't reachable.

### Candidate
type: B, layer: 1, pos: 13

### Candidate
type: A, layer: 1, pos: 4

### Candidate
type: A, layer: 1, pos: 15

### Candidate
type: A, layer: 1, pos: 22

## Relational analysis of NS_A2_A1_B1_A2_B2_A1_B2_B1_B1_A1

### Relational analysis result of NS_A2_A1_B1_A2_B2_A1_B2_B1_B1_A1
Status: Status.UNKNOWN
Output dim: 0, lower bound: -81.8604805, upper bound: 81.8610756
time: 1.06 seconds

## Relational analysis of NS_A2_A1_B1_A2_B2_A1_B2_B1_B1_A2

### Relational analysis result of NS_A2_A1_B1_A2_B2_A1_B2_B1_B1_A2
Status: Status.UNKNOWN
Output dim: 0, lower bound: -81.8604805, upper bound: 81.8610756
time: 0.87 seconds

## BFS NS instance: NS_A2_A1_B1_A2_B2_A1_B2_B1_B2

### Backsubstitution after applying NS history:
0: -19.5407543, 71.3658142, -21.9304276, 79.9511108, -99.4918671, 93.2962265
1: -51.8812523, 160.9836731, -58.0220451, 182.0756073, -233.9568481, 219.0056763
2: -77.8219833, 143.2691956, -87.1399994, 160.7810974, -238.6030884, 230.4091797
3: -44.6342278, 172.0003357, -49.9183846, 194.5646057, -239.1988220, 221.9186859
4: -71.1089172, 126.1929550, -79.5788498, 141.7767334, -212.8856201, 205.7717743

Time for backsubstitution: 0.97 seconds

### NS candidates at layer 1
type: A, layer: 1, pos: 41
type: A, layer: 1, pos: 14
type: A, layer: 1, pos: 19
type: A, layer: 1, pos: 39
type: B, layer: 1, pos: 14
type: A, layer: 1, pos: 25
type: B, layer: 1, pos: 12
type: B, layer: 1, pos: 19
type: B, layer: 1, pos: 25
type: B, layer: 1, pos: 39
type: A, layer: 1, pos: 20
type: A, layer: 1, pos: 8
type: A, layer: 1, pos: 37
type: A, layer: 1, pos: 12
type: B, layer: 1, pos: 29
type: B, layer: 1, pos: 37
type: B, layer: 1, pos: 20
type: A, layer: 1, pos: 21
type: A, layer: 1, pos: 29
type: B, layer: 1, pos: 8
type: B, layer: 1, pos: 41
type: A, layer: 1, pos: 6
type: B, layer: 1, pos: 6
type: B, layer: 1, pos: 4
type: A, layer: 1, pos: 11
type: A, layer: 1, pos: 46
type: B, layer: 1, pos: 46
type: B, layer: 1, pos: 15
type: A, layer: 1, pos: 22
type: B, layer: 1, pos: 13
type: A, layer: 1, pos: 15
type: B, layer: 1, pos: 16
type: A, layer: 1, pos: 4
type: B, layer: 1, pos: 38
type: A, layer: 1, pos: 38
type: A, layer: 1, pos: 23
type: B, layer: 1, pos: 11
type: B, layer: 1, pos: 23
type: B, layer: 1, pos: 21
type: A, layer: 1, pos: 31
type: B, layer: 1, pos: 31
type: A, layer: 1, pos: 16
type: A, layer: 1, pos: 42

Time for candidate selection: 0.07 seconds

### Candidate
type: A, layer: 1, pos: 41

### Candidate
type: A, layer: 1, pos: 14

### Candidate
type: A, layer: 1, pos: 19

### Candidate
type: A, layer: 1, pos: 39

### Candidate
type: B, layer: 1, pos: 14

### Candidate
type: A, layer: 1, pos: 25

### Candidate
type: B, layer: 1, pos: 12

### Candidate
type: B, layer: 1, pos: 19

### Candidate
type: B, layer: 1, pos: 25

### Candidate
type: B, layer: 1, pos: 39

### Candidate
type: A, layer: 1, pos: 20

### Candidate
type: A, layer: 1, pos: 8

### Candidate
type: A, layer: 1, pos: 37

### Candidate
type: A, layer: 1, pos: 12

### Candidate
type: B, layer: 1, pos: 29

### Candidate
type: B, layer: 1, pos: 37

### Candidate
type: B, layer: 1, pos: 20

### Candidate
type: A, layer: 1, pos: 21

### Candidate
type: A, layer: 1, pos: 29

### Candidate
type: B, layer: 1, pos: 8

### Candidate
type: B, layer: 1, pos: 41

### Candidate
type: A, layer: 1, pos: 6

### Candidate
type: B, layer: 1, pos: 6

### Candidate
type: B, layer: 1, pos: 4

### Candidate
type: A, layer: 1, pos: 11

### Candidate
type: A, layer: 1, pos: 46

### Candidate
type: B, layer: 1, pos: 46

### Candidate
type: B, layer: 1, pos: 15

## Relational analysis of NS_A2_A1_B1_A2_B2_A1_B2_B1_B2_B1
Optimization infeasible because this subproblem isn't reachable.

### Candidate
type: A, layer: 1, pos: 22

## Relational analysis of NS_A2_A1_B1_A2_B2_A1_B2_B1_B2_A1

### Relational analysis result of NS_A2_A1_B1_A2_B2_A1_B2_B1_B2_A1
Status: Status.UNKNOWN
Output dim: 0, lower bound: -81.8607244, upper bound: 81.8611686
time: 0.92 seconds

## Relational analysis of NS_A2_A1_B1_A2_B2_A1_B2_B1_B2_A2

### Relational analysis result of NS_A2_A1_B1_A2_B2_A1_B2_B1_B2_A2
Status: Status.UNKNOWN
Output dim: 0, lower bound: -81.8607244, upper bound: 81.8611686
time: 0.92 seconds

## BFS NS instance: NS_A2_A1_B1_A2_B2_A1_B2_B2_A1

### Backsubstitution after applying NS history:
0: -19.1982937, 70.1187820, -21.7932167, 79.2595901, -98.4578781, 91.9120026
1: -50.9867439, 158.1197510, -57.3422089, 181.5820465, -232.5687866, 215.4619598
2: -76.5202637, 140.7139435, -85.8580017, 159.6853485, -236.2055969, 226.5719452
3: -43.8595810, 168.9619751, -49.3844452, 193.9513397, -237.8108978, 218.3464203
4: -69.9047928, 123.9398575, -78.4333725, 140.9108582, -210.8156128, 202.3732300

Time for backsubstitution: 0.95 seconds

### NS candidates at layer 1
type: A, layer: 1, pos: 41
type: A, layer: 1, pos: 14
type: A, layer: 1, pos: 19
type: A, layer: 1, pos: 39
type: B, layer: 1, pos: 14
type: B, layer: 1, pos: 12
type: B, layer: 1, pos: 19
type: B, layer: 1, pos: 39
type: A, layer: 1, pos: 12
type: A, layer: 1, pos: 8
type: A, layer: 1, pos: 37
type: A, layer: 1, pos: 25
type: B, layer: 1, pos: 29
type: B, layer: 1, pos: 37
type: A, layer: 1, pos: 29
type: A, layer: 1, pos: 21
type: B, layer: 1, pos: 8
type: B, layer: 1, pos: 25
type: A, layer: 1, pos: 20
type: B, layer: 1, pos: 20
type: B, layer: 1, pos: 41
type: B, layer: 1, pos: 15
type: A, layer: 1, pos: 6
type: A, layer: 1, pos: 46
type: B, layer: 1, pos: 6
type: B, layer: 1, pos: 46
type: B, layer: 1, pos: 21
type: A, layer: 1, pos: 11
type: B, layer: 1, pos: 16
type: B, layer: 1, pos: 4
type: A, layer: 1, pos: 15
type: B, layer: 1, pos: 42
type: B, layer: 1, pos: 13
type: A, layer: 1, pos: 38
type: B, layer: 1, pos: 23
type: B, layer: 1, pos: 38
type: A, layer: 1, pos: 4
type: B, layer: 1, pos: 11
type: A, layer: 1, pos: 23
type: A, layer: 1, pos: 31
type: B, layer: 1, pos: 31
type: A, layer: 1, pos: 16
type: A, layer: 1, pos: 42

Time for candidate selection: 0.08 seconds

### Candidate
type: A, layer: 1, pos: 41

### Candidate
type: A, layer: 1, pos: 14

### Candidate
type: A, layer: 1, pos: 19

### Candidate
type: A, layer: 1, pos: 39

### Candidate
type: B, layer: 1, pos: 14

### Candidate
type: B, layer: 1, pos: 12

### Candidate
type: B, layer: 1, pos: 19

### Candidate
type: B, layer: 1, pos: 39

### Candidate
type: A, layer: 1, pos: 12

### Candidate
type: A, layer: 1, pos: 8

### Candidate
type: A, layer: 1, pos: 37

### Candidate
type: A, layer: 1, pos: 25

### Candidate
type: B, layer: 1, pos: 29

### Candidate
type: B, layer: 1, pos: 37

### Candidate
type: A, layer: 1, pos: 29

### Candidate
type: A, layer: 1, pos: 21

### Candidate
type: B, layer: 1, pos: 8

### Candidate
type: B, layer: 1, pos: 25

### Candidate
type: A, layer: 1, pos: 20

### Candidate
type: B, layer: 1, pos: 20

### Candidate
type: B, layer: 1, pos: 41

### Candidate
type: B, layer: 1, pos: 15

### Candidate
type: A, layer: 1, pos: 6

### Candidate
type: A, layer: 1, pos: 46

### Candidate
type: B, layer: 1, pos: 6

### Candidate
type: B, layer: 1, pos: 46

### Candidate
type: B, layer: 1, pos: 21

### Candidate
type: A, layer: 1, pos: 11

### Candidate
type: B, layer: 1, pos: 16

### Candidate
type: B, layer: 1, pos: 4

### Candidate
type: A, layer: 1, pos: 15

### Candidate
type: B, layer: 1, pos: 42

## Relational analysis of NS_A2_A1_B1_A2_B2_A1_B2_B2_A1_B1

### Relational analysis result of NS_A2_A1_B1_A2_B2_A1_B2_B2_A1_B1
Status: Status.UNKNOWN
Output dim: 0, lower bound: -81.8604805, upper bound: 81.8611192
time: 0.77 seconds

## Relational analysis of NS_A2_A1_B1_A2_B2_A1_B2_B2_A1_B2

### Relational analysis result of NS_A2_A1_B1_A2_B2_A1_B2_B2_A1_B2
Status: Status.UNKNOWN
Output dim: 0, lower bound: -81.8605459, upper bound: 81.8611290
time: 0.92 seconds

## BFS NS instance: NS_A2_A1_B1_A2_B2_A1_B2_B2_A2

### Backsubstitution after applying NS history:
0: -19.3761635, 70.6416855, -21.7932167, 79.2595901, -98.6357574, 92.4349060
1: -51.1780510, 160.0721283, -57.3422089, 181.5820465, -232.7601013, 217.4143372
2: -76.4557343, 142.3144379, -85.8580017, 159.6853485, -236.1410522, 228.1724396
3: -44.0854263, 170.8686676, -49.3844452, 193.9513397, -238.0367737, 220.2531128
4: -69.9495621, 125.3343735, -78.4333725, 140.9108582, -210.8604126, 203.7677155

Time for backsubstitution: 1.10 seconds

### NS candidates at layer 1
type: A, layer: 1, pos: 41
type: A, layer: 1, pos: 14
type: A, layer: 1, pos: 19
type: A, layer: 1, pos: 39
type: B, layer: 1, pos: 14
type: B, layer: 1, pos: 12
type: B, layer: 1, pos: 19
type: B, layer: 1, pos: 39
type: A, layer: 1, pos: 12
type: A, layer: 1, pos: 8
type: A, layer: 1, pos: 37
type: B, layer: 1, pos: 37
type: B, layer: 1, pos: 29
type: A, layer: 1, pos: 25
type: A, layer: 1, pos: 21
type: B, layer: 1, pos: 8
type: A, layer: 1, pos: 29
type: A, layer: 1, pos: 20
type: B, layer: 1, pos: 25
type: B, layer: 1, pos: 41
type: B, layer: 1, pos: 20
type: A, layer: 1, pos: 6
type: B, layer: 1, pos: 15
type: B, layer: 1, pos: 6
type: B, layer: 1, pos: 46
type: A, layer: 1, pos: 46
type: B, layer: 1, pos: 21
type: A, layer: 1, pos: 15
type: A, layer: 1, pos: 11
type: B, layer: 1, pos: 4
type: B, layer: 1, pos: 16
type: B, layer: 1, pos: 42
type: B, layer: 1, pos: 13
type: B, layer: 1, pos: 38
type: A, layer: 1, pos: 38
type: A, layer: 1, pos: 4
type: B, layer: 1, pos: 11
type: A, layer: 1, pos: 16
type: B, layer: 1, pos: 23
type: A, layer: 1, pos: 23
type: A, layer: 1, pos: 31
type: B, layer: 1, pos: 31
type: A, layer: 1, pos: 42

Time for candidate selection: 0.10 seconds

### Candidate
type: A, layer: 1, pos: 41

### Candidate
type: A, layer: 1, pos: 14

### Candidate
type: A, layer: 1, pos: 19

### Candidate
type: A, layer: 1, pos: 39

### Candidate
type: B, layer: 1, pos: 14

### Candidate
type: B, layer: 1, pos: 12

### Candidate
type: B, layer: 1, pos: 19

### Candidate
type: B, layer: 1, pos: 39

### Candidate
type: A, layer: 1, pos: 12

### Candidate
type: A, layer: 1, pos: 8

### Candidate
type: A, layer: 1, pos: 37

### Candidate
type: B, layer: 1, pos: 37

### Candidate
type: B, layer: 1, pos: 29

### Candidate
type: A, layer: 1, pos: 25

### Candidate
type: A, layer: 1, pos: 21

### Candidate
type: B, layer: 1, pos: 8

### Candidate
type: A, layer: 1, pos: 29

### Candidate
type: A, layer: 1, pos: 20

### Candidate
type: B, layer: 1, pos: 25

### Candidate
type: B, layer: 1, pos: 41

### Candidate
type: B, layer: 1, pos: 20

### Candidate
type: A, layer: 1, pos: 6

### Candidate
type: B, layer: 1, pos: 15

### Candidate
type: B, layer: 1, pos: 6

### Candidate
type: B, layer: 1, pos: 46

### Candidate
type: A, layer: 1, pos: 46

### Candidate
type: B, layer: 1, pos: 21

### Candidate
type: A, layer: 1, pos: 15

### Candidate
type: A, layer: 1, pos: 11

### Candidate
type: B, layer: 1, pos: 4

### Candidate
type: B, layer: 1, pos: 16

### Candidate
type: B, layer: 1, pos: 42

## Relational analysis of NS_A2_A1_B1_A2_B2_A1_B2_B2_A2_B1

### Relational analysis result of NS_A2_A1_B1_A2_B2_A1_B2_B2_A2_B1
Status: Status.UNKNOWN
Output dim: 0, lower bound: -81.8604547, upper bound: 81.8611026
time: 0.75 seconds

## Relational analysis of NS_A2_A1_B1_A2_B2_A1_B2_B2_A2_B2
Optimization infeasible because this subproblem isn't reachable.

### Candidate
type: B, layer: 1, pos: 13

### Candidate
type: B, layer: 1, pos: 38

### Candidate
type: A, layer: 1, pos: 38

### Candidate
type: A, layer: 1, pos: 4

### Candidate
type: B, layer: 1, pos: 11

### Candidate
type: A, layer: 1, pos: 16

### Candidate
type: B, layer: 1, pos: 23

### Candidate
type: A, layer: 1, pos: 23

### Candidate
type: A, layer: 1, pos: 31

## Relational analysis of NS_A2_A1_B1_A2_B2_A1_B2_B2_A2_A1
Optimization infeasible because this subproblem isn't reachable.

### Candidate
type: B, layer: 1, pos: 31

### Candidate
type: A, layer: 1, pos: 42

## BFS NS instance: NS_A2_A1_B1_A2_B2_A2_B1_B1_A1

### Backsubstitution after applying NS history:
0: -21.8176060, 79.6616898, -20.3303623, 74.1747437, -95.9923477, 99.9920502
1: -57.5466347, 181.4291687, -53.7455330, 169.0512543, -226.5978851, 235.1746979
2: -86.0756149, 160.5619507, -80.6776886, 148.9853363, -235.0609436, 241.2396393
3: -49.4931564, 193.4996185, -46.2123299, 180.5061493, -229.9992981, 239.7119446
4: -78.6817398, 141.4108582, -73.6563263, 131.2958069, -209.9775085, 215.0671844

Time for backsubstitution: 1.08 seconds

### NS candidates at layer 1
type: B, layer: 1, pos: 41
type: B, layer: 1, pos: 20
type: A, layer: 1, pos: 19
type: A, layer: 1, pos: 20
type: A, layer: 1, pos: 41
type: B, layer: 1, pos: 19
type: B, layer: 1, pos: 39
type: B, layer: 1, pos: 25
type: B, layer: 1, pos: 14
type: A, layer: 1, pos: 23
type: A, layer: 1, pos: 14
type: A, layer: 1, pos: 25
type: B, layer: 1, pos: 21
type: B, layer: 1, pos: 23
type: A, layer: 1, pos: 6
type: B, layer: 1, pos: 6
type: B, layer: 1, pos: 15
type: B, layer: 1, pos: 4
type: B, layer: 1, pos: 46
type: A, layer: 1, pos: 38
type: B, layer: 1, pos: 13
type: A, layer: 1, pos: 46
type: A, layer: 1, pos: 29
type: A, layer: 1, pos: 21
type: A, layer: 1, pos: 15
type: A, layer: 1, pos: 12
type: B, layer: 1, pos: 11
type: B, layer: 1, pos: 12
type: A, layer: 1, pos: 4
type: B, layer: 1, pos: 38
type: B, layer: 1, pos: 29
type: A, layer: 1, pos: 42
type: A, layer: 1, pos: 37
type: A, layer: 1, pos: 16
type: B, layer: 1, pos: 8
type: B, layer: 1, pos: 37
type: B, layer: 1, pos: 42
type: B, layer: 1, pos: 31
type: A, layer: 1, pos: 31
type: A, layer: 1, pos: 22
type: A, layer: 1, pos: 11
type: A, layer: 1, pos: 8
type: B, layer: 1, pos: 22

Time for candidate selection: 0.08 seconds

### Candidate
type: B, layer: 1, pos: 41

### Candidate
type: B, layer: 1, pos: 20

### Candidate
type: A, layer: 1, pos: 19

### Candidate
type: A, layer: 1, pos: 20

### Candidate
type: A, layer: 1, pos: 41

### Candidate
type: B, layer: 1, pos: 19

### Candidate
type: B, layer: 1, pos: 39

## Relational analysis of NS_A2_A1_B1_A2_B2_A2_B1_B1_A1_B1

### Relational analysis result of NS_A2_A1_B1_A2_B2_A2_B1_B1_A1_B1
Status: Status.UNKNOWN
Output dim: 0, lower bound: -81.8631372, upper bound: 81.8632948
time: 0.81 seconds

## Relational analysis of NS_A2_A1_B1_A2_B2_A2_B1_B1_A1_B2

### Relational analysis result of NS_A2_A1_B1_A2_B2_A2_B1_B1_A1_B2
Status: Status.UNKNOWN
Output dim: 0, lower bound: -81.8631372, upper bound: 81.8632948
time: 0.93 seconds

## BFS NS instance: NS_A2_A1_B1_A2_B2_A2_B1_B1_A2

### Backsubstitution after applying NS history:
0: -25.4717045, 92.7767258, -20.0697346, 73.2081604, -98.6798630, 112.8464508
1: -67.0863876, 210.8640747, -53.0901680, 166.7341003, -233.8204956, 263.9542236
2: -100.6374817, 187.2869263, -79.7464066, 146.9233856, -247.5608368, 267.0332947
3: -57.7826309, 224.8646393, -45.6617775, 177.8588867, -235.6415100, 270.5263977
4: -91.9857330, 164.9750671, -72.7989349, 129.5023499, -221.4880829, 237.7740021

Time for backsubstitution: 1.04 seconds

### NS candidates at layer 1
type: B, layer: 1, pos: 41
type: B, layer: 1, pos: 20
type: A, layer: 1, pos: 19
type: B, layer: 1, pos: 19
type: B, layer: 1, pos: 21
type: B, layer: 1, pos: 39
type: B, layer: 1, pos: 25
type: A, layer: 1, pos: 23
type: B, layer: 1, pos: 14
type: A, layer: 1, pos: 20
type: A, layer: 1, pos: 14
type: A, layer: 1, pos: 6
type: B, layer: 1, pos: 6
type: A, layer: 1, pos: 25
type: B, layer: 1, pos: 4
type: B, layer: 1, pos: 15
type: A, layer: 1, pos: 38
type: B, layer: 1, pos: 46
type: B, layer: 1, pos: 23
type: B, layer: 1, pos: 13
type: B, layer: 1, pos: 8
type: A, layer: 1, pos: 41
type: B, layer: 1, pos: 12
type: A, layer: 1, pos: 29
type: A, layer: 1, pos: 46
type: A, layer: 1, pos: 42
type: B, layer: 1, pos: 11
type: A, layer: 1, pos: 15
type: B, layer: 1, pos: 29
type: A, layer: 1, pos: 12
type: B, layer: 1, pos: 37
type: A, layer: 1, pos: 16
type: B, layer: 1, pos: 22
type: A, layer: 1, pos: 37
type: B, layer: 1, pos: 31
type: B, layer: 1, pos: 38
type: A, layer: 1, pos: 31
type: A, layer: 1, pos: 4
type: A, layer: 1, pos: 11
type: A, layer: 1, pos: 22
type: B, layer: 1, pos: 42
type: A, layer: 1, pos: 8
type: A, layer: 1, pos: 21

Time for candidate selection: 0.09 seconds

### Candidate
type: B, layer: 1, pos: 41

### Candidate
type: B, layer: 1, pos: 20

### Candidate
type: A, layer: 1, pos: 19

### Candidate
type: B, layer: 1, pos: 19

### Candidate
type: B, layer: 1, pos: 21

### Candidate
type: B, layer: 1, pos: 39

## Relational analysis of NS_A2_A1_B1_A2_B2_A2_B1_B1_A2_B1

### Relational analysis result of NS_A2_A1_B1_A2_B2_A2_B1_B1_A2_B1
Status: Status.UNKNOWN
Output dim: 0, lower bound: -81.8632284, upper bound: 81.8633019
time: 1.03 seconds

## Relational analysis of NS_A2_A1_B1_A2_B2_A2_B1_B1_A2_B2

### Relational analysis result of NS_A2_A1_B1_A2_B2_A2_B1_B1_A2_B2
Status: Status.UNKNOWN
Output dim: 0, lower bound: -81.8632284, upper bound: 81.8633019
time: 0.99 seconds

## BFS NS instance: NS_A2_A1_B1_A2_B2_A2_B1_B2_A1

### Backsubstitution after applying NS history:
0: -21.6661549, 79.1037064, -21.8679695, 79.8196640, -101.4858170, 100.9716644
1: -57.1756744, 180.2160187, -57.7817192, 181.9031372, -239.0787964, 237.9977417
2: -85.4210587, 159.3976746, -86.7036438, 160.4056091, -245.8266602, 246.1013184
3: -49.1567116, 192.1778412, -49.6888580, 194.1599426, -243.3166504, 241.8666992
4: -78.1009827, 140.3668823, -79.1684113, 141.4126129, -219.5135956, 219.5352936

Time for backsubstitution: 1.01 seconds

### NS candidates at layer 1
type: B, layer: 1, pos: 41
type: A, layer: 1, pos: 41
type: B, layer: 1, pos: 19
type: A, layer: 1, pos: 19
type: A, layer: 1, pos: 6
type: B, layer: 1, pos: 6
type: B, layer: 1, pos: 25
type: B, layer: 1, pos: 21
type: A, layer: 1, pos: 25
type: A, layer: 1, pos: 46
type: A, layer: 1, pos: 21
type: B, layer: 1, pos: 46
type: B, layer: 1, pos: 4
type: A, layer: 1, pos: 38
type: B, layer: 1, pos: 38
type: A, layer: 1, pos: 4
type: A, layer: 1, pos: 12
type: A, layer: 1, pos: 29
type: B, layer: 1, pos: 13
type: B, layer: 1, pos: 14
type: B, layer: 1, pos: 12
type: A, layer: 1, pos: 14
type: A, layer: 1, pos: 11
type: B, layer: 1, pos: 29
type: A, layer: 1, pos: 31
type: B, layer: 1, pos: 31
type: A, layer: 1, pos: 15
type: B, layer: 1, pos: 39
type: B, layer: 1, pos: 20
type: A, layer: 1, pos: 39
type: B, layer: 1, pos: 11
type: B, layer: 1, pos: 15
type: B, layer: 1, pos: 23
type: A, layer: 1, pos: 20
type: B, layer: 1, pos: 8
type: B, layer: 1, pos: 22
type: A, layer: 1, pos: 8
type: A, layer: 1, pos: 23
type: A, layer: 1, pos: 42
type: A, layer: 1, pos: 37
type: B, layer: 1, pos: 42
type: B, layer: 1, pos: 37
type: A, layer: 1, pos: 22

Time for candidate selection: 0.07 seconds

### Candidate
type: B, layer: 1, pos: 41

### Candidate
type: A, layer: 1, pos: 41

### Candidate
type: B, layer: 1, pos: 19

### Candidate
type: A, layer: 1, pos: 19

### Candidate
type: A, layer: 1, pos: 6

### Candidate
type: B, layer: 1, pos: 6

### Candidate
type: B, layer: 1, pos: 25

### Candidate
type: B, layer: 1, pos: 21

### Candidate
type: A, layer: 1, pos: 25

### Candidate
type: A, layer: 1, pos: 46

### Candidate
type: A, layer: 1, pos: 21

### Candidate
type: B, layer: 1, pos: 46

### Candidate
type: B, layer: 1, pos: 4

### Candidate
type: A, layer: 1, pos: 38

### Candidate
type: B, layer: 1, pos: 38

### Candidate
type: A, layer: 1, pos: 4

### Candidate
type: A, layer: 1, pos: 12

### Candidate
type: A, layer: 1, pos: 29

### Candidate
type: B, layer: 1, pos: 13

### Candidate
type: B, layer: 1, pos: 14

### Candidate
type: B, layer: 1, pos: 12

### Candidate
type: A, layer: 1, pos: 14

### Candidate
type: A, layer: 1, pos: 11

### Candidate
type: B, layer: 1, pos: 29

### Candidate
type: A, layer: 1, pos: 31

### Candidate
type: B, layer: 1, pos: 31

### Candidate
type: A, layer: 1, pos: 15

## Relational analysis of NS_A2_A1_B1_A2_B2_A2_B1_B2_A1_A1
Optimization infeasible because this subproblem isn't reachable.

### Candidate
type: B, layer: 1, pos: 39

## Relational analysis of NS_A2_A1_B1_A2_B2_A2_B1_B2_A1_B1
Optimization infeasible because this subproblem isn't reachable.

### Candidate
type: B, layer: 1, pos: 20

### Candidate
type: A, layer: 1, pos: 39

### Candidate
type: B, layer: 1, pos: 11

### Candidate
type: B, layer: 1, pos: 15

### Candidate
type: B, layer: 1, pos: 23

## Relational analysis of NS_A2_A1_B1_A2_B2_A2_B1_B2_A1_B1

### Relational analysis result of NS_A2_A1_B1_A2_B2_A2_B1_B2_A1_B1
Status: Status.UNKNOWN
Output dim: 0, lower bound: -81.8629460, upper bound: 81.8630767
time: 0.78 seconds

## Relational analysis of NS_A2_A1_B1_A2_B2_A2_B1_B2_A1_B2
Optimization infeasible because this subproblem isn't reachable.

### Candidate
type: A, layer: 1, pos: 20

### Candidate
type: B, layer: 1, pos: 8

## Relational analysis of NS_A2_A1_B1_A2_B2_A2_B1_B2_A1_B1

### Relational analysis result of NS_A2_A1_B1_A2_B2_A2_B1_B2_A1_B1
Status: Status.UNKNOWN
Output dim: 0, lower bound: -81.8626869, upper bound: 81.8626522
time: 0.94 seconds

## Relational analysis of NS_A2_A1_B1_A2_B2_A2_B1_B2_A1_B2

### Relational analysis result of NS_A2_A1_B1_A2_B2_A2_B1_B2_A1_B2
Status: Status.UNKNOWN
Output dim: 0, lower bound: -81.8623963, upper bound: 81.8624728
time: 0.89 seconds

## BFS NS instance: NS_A2_A1_B1_A2_B2_A2_B1_B2_A2

### Backsubstitution after applying NS history:
0: -22.6331539, 82.6661758, -21.8679695, 79.8196640, -102.4528198, 104.5341415
1: -59.7047958, 188.2940369, -57.7817192, 181.9031372, -241.6079407, 246.0757599
2: -89.2544403, 166.6350403, -86.7036438, 160.4056091, -249.6600494, 253.3386841
3: -51.3380013, 200.7817230, -49.6888580, 194.1599426, -245.4979401, 250.4705811
4: -81.5932159, 146.7880249, -79.1684113, 141.4126129, -223.0058136, 225.9564362

Time for backsubstitution: 0.99 seconds

### NS candidates at layer 1
type: A, layer: 1, pos: 41
type: B, layer: 1, pos: 41
type: A, layer: 1, pos: 19
type: B, layer: 1, pos: 19
type: A, layer: 1, pos: 6
type: B, layer: 1, pos: 6
type: B, layer: 1, pos: 25
type: A, layer: 1, pos: 25
type: B, layer: 1, pos: 21
type: A, layer: 1, pos: 21
type: A, layer: 1, pos: 46
type: B, layer: 1, pos: 46
type: B, layer: 1, pos: 4
type: A, layer: 1, pos: 38
type: B, layer: 1, pos: 38
type: A, layer: 1, pos: 12
type: A, layer: 1, pos: 4
type: A, layer: 1, pos: 29
type: B, layer: 1, pos: 13
type: B, layer: 1, pos: 12
type: A, layer: 1, pos: 14
type: B, layer: 1, pos: 14
type: B, layer: 1, pos: 29
type: A, layer: 1, pos: 31
type: B, layer: 1, pos: 31
type: B, layer: 1, pos: 20
type: A, layer: 1, pos: 11
type: A, layer: 1, pos: 39
type: B, layer: 1, pos: 15
type: B, layer: 1, pos: 11
type: B, layer: 1, pos: 39
type: A, layer: 1, pos: 15
type: B, layer: 1, pos: 23
type: A, layer: 1, pos: 8
type: A, layer: 1, pos: 20
type: A, layer: 1, pos: 23
type: A, layer: 1, pos: 22
type: A, layer: 1, pos: 37
type: B, layer: 1, pos: 42
type: B, layer: 1, pos: 8
type: B, layer: 1, pos: 22
type: A, layer: 1, pos: 42
type: B, layer: 1, pos: 37

Time for candidate selection: 0.08 seconds

### Candidate
type: A, layer: 1, pos: 41

### Candidate
type: B, layer: 1, pos: 41

### Candidate
type: A, layer: 1, pos: 19

### Candidate
type: B, layer: 1, pos: 19

### Candidate
type: A, layer: 1, pos: 6

### Candidate
type: B, layer: 1, pos: 6

### Candidate
type: B, layer: 1, pos: 25

### Candidate
type: A, layer: 1, pos: 25

### Candidate
type: B, layer: 1, pos: 21

### Candidate
type: A, layer: 1, pos: 21

### Candidate
type: A, layer: 1, pos: 46

### Candidate
type: B, layer: 1, pos: 46

### Candidate
type: B, layer: 1, pos: 4

### Candidate
type: A, layer: 1, pos: 38

### Candidate
type: B, layer: 1, pos: 38

### Candidate
type: A, layer: 1, pos: 12

### Candidate
type: A, layer: 1, pos: 4

### Candidate
type: A, layer: 1, pos: 29

### Candidate
type: B, layer: 1, pos: 13

### Candidate
type: B, layer: 1, pos: 12

### Candidate
type: A, layer: 1, pos: 14

### Candidate
type: B, layer: 1, pos: 14

### Candidate
type: B, layer: 1, pos: 29

### Candidate
type: A, layer: 1, pos: 31

### Candidate
type: B, layer: 1, pos: 31

### Candidate
type: B, layer: 1, pos: 20

### Candidate
type: A, layer: 1, pos: 11

### Candidate
type: A, layer: 1, pos: 39

## Relational analysis of NS_A2_A1_B1_A2_B2_A2_B1_B2_A2_A1

### Relational analysis result of NS_A2_A1_B1_A2_B2_A2_B1_B2_A2_A1
Status: Status.UNKNOWN
Output dim: 0, lower bound: -81.8631082, upper bound: 81.8632765
time: 0.87 seconds

## Relational analysis of NS_A2_A1_B1_A2_B2_A2_B1_B2_A2_A2
Optimization infeasible because this subproblem isn't reachable.

### Candidate
type: B, layer: 1, pos: 15

## Relational analysis of NS_A2_A1_B1_A2_B2_A2_B1_B2_A2_B1
Optimization infeasible because this subproblem isn't reachable.

### Candidate
type: B, layer: 1, pos: 11

### Candidate
type: B, layer: 1, pos: 39

### Candidate
type: A, layer: 1, pos: 15

### Candidate
type: B, layer: 1, pos: 23

## Relational analysis of NS_A2_A1_B1_A2_B2_A2_B1_B2_A2_B1

### Relational analysis result of NS_A2_A1_B1_A2_B2_A2_B1_B2_A2_B1
Status: Status.UNKNOWN
Output dim: 0, lower bound: -81.8629460, upper bound: 81.8630767
time: 0.84 seconds

## Relational analysis of NS_A2_A1_B1_A2_B2_A2_B1_B2_A2_B2
Optimization infeasible because this subproblem isn't reachable.

### Candidate
type: A, layer: 1, pos: 8

## Relational analysis of NS_A2_A1_B1_A2_B2_A2_B1_B2_A2_A1

### Relational analysis result of NS_A2_A1_B1_A2_B2_A2_B1_B2_A2_A1
Status: Status.UNKNOWN
Output dim: 0, lower bound: -81.8625846, upper bound: 81.8628301
time: 0.89 seconds

## Relational analysis of NS_A2_A1_B1_A2_B2_A2_B1_B2_A2_A2

### Relational analysis result of NS_A2_A1_B1_A2_B2_A2_B1_B2_A2_A2
Status: Status.UNKNOWN
Output dim: 0, lower bound: -81.8623963, upper bound: 81.8625689
time: 0.82 seconds

## BFS NS instance: NS_A2_A1_B1_A2_B2_A2_B2_A1_A1

### Backsubstitution after applying NS history:
0: -21.5765705, 78.7623978, -21.6943512, 79.0675583, -100.6441269, 100.4567490
1: -56.8920670, 179.4511108, -57.3755074, 180.1337128, -237.0257874, 236.8266144
2: -85.1523132, 158.7488708, -86.1597443, 159.0279999, -244.1803131, 244.9086151
3: -48.9436722, 191.3975830, -49.3702393, 192.4675446, -241.4111786, 240.7678223
4: -77.8289566, 139.8345032, -78.6882248, 140.2301636, -218.0591125, 218.5227203

Time for backsubstitution: 0.96 seconds

### NS candidates at layer 1
type: A, layer: 1, pos: 41
type: A, layer: 1, pos: 14
type: B, layer: 1, pos: 19
type: B, layer: 1, pos: 14
type: B, layer: 1, pos: 41
type: A, layer: 1, pos: 19
type: A, layer: 1, pos: 25
type: B, layer: 1, pos: 12
type: B, layer: 1, pos: 25
type: B, layer: 1, pos: 39
type: B, layer: 1, pos: 37
type: A, layer: 1, pos: 12
type: A, layer: 1, pos: 20
type: B, layer: 1, pos: 29
type: B, layer: 1, pos: 20
type: A, layer: 1, pos: 37
type: A, layer: 1, pos: 29
type: B, layer: 1, pos: 8
type: B, layer: 1, pos: 21
type: A, layer: 1, pos: 11
type: A, layer: 1, pos: 6
type: A, layer: 1, pos: 46
type: B, layer: 1, pos: 6
type: B, layer: 1, pos: 22
type: A, layer: 1, pos: 15
type: B, layer: 1, pos: 4
type: A, layer: 1, pos: 4
type: B, layer: 1, pos: 15
type: B, layer: 1, pos: 46
type: A, layer: 1, pos: 21
type: A, layer: 1, pos: 38
type: B, layer: 1, pos: 13
type: A, layer: 1, pos: 23
type: B, layer: 1, pos: 42
type: A, layer: 1, pos: 16
type: A, layer: 1, pos: 42
type: B, layer: 1, pos: 38
type: A, layer: 1, pos: 31
type: B, layer: 1, pos: 31
type: B, layer: 1, pos: 16
type: B, layer: 1, pos: 23
type: A, layer: 1, pos: 22
type: B, layer: 1, pos: 11

Time for candidate selection: 0.08 seconds

### Candidate
type: A, layer: 1, pos: 41

### Candidate
type: A, layer: 1, pos: 14

### Candidate
type: B, layer: 1, pos: 19

### Candidate
type: B, layer: 1, pos: 14

### Candidate
type: B, layer: 1, pos: 41

### Candidate
type: A, layer: 1, pos: 19

### Candidate
type: A, layer: 1, pos: 25

### Candidate
type: B, layer: 1, pos: 12

### Candidate
type: B, layer: 1, pos: 25

### Candidate
type: B, layer: 1, pos: 39

## Relational analysis of NS_A2_A1_B1_A2_B2_A2_B2_A1_A1_B1
Optimization infeasible because this subproblem isn't reachable.

### Candidate
type: B, layer: 1, pos: 37

### Candidate
type: A, layer: 1, pos: 12

### Candidate
type: A, layer: 1, pos: 20

### Candidate
type: B, layer: 1, pos: 29

### Candidate
type: B, layer: 1, pos: 20

### Candidate
type: A, layer: 1, pos: 37

### Candidate
type: A, layer: 1, pos: 29

### Candidate
type: B, layer: 1, pos: 8

### Candidate
type: B, layer: 1, pos: 21

### Candidate
type: A, layer: 1, pos: 11

### Candidate
type: A, layer: 1, pos: 6

### Candidate
type: A, layer: 1, pos: 46

### Candidate
type: B, layer: 1, pos: 6

### Candidate
type: B, layer: 1, pos: 22

### Candidate
type: A, layer: 1, pos: 15

## Relational analysis of NS_A2_A1_B1_A2_B2_A2_B2_A1_A1_A1
Optimization infeasible because this subproblem isn't reachable.

### Candidate
type: B, layer: 1, pos: 4

### Candidate
type: A, layer: 1, pos: 4

### Candidate
type: B, layer: 1, pos: 15

### Candidate
type: B, layer: 1, pos: 46

### Candidate
type: A, layer: 1, pos: 21

### Candidate
type: A, layer: 1, pos: 38

### Candidate
type: B, layer: 1, pos: 13

### Candidate
type: A, layer: 1, pos: 23

### Candidate
type: B, layer: 1, pos: 42

## Relational analysis of NS_A2_A1_B1_A2_B2_A2_B2_A1_A1_B1
Optimization infeasible because this subproblem isn't reachable.

### Candidate
type: A, layer: 1, pos: 16

### Candidate
type: A, layer: 1, pos: 42

### Candidate
type: B, layer: 1, pos: 38

### Candidate
type: A, layer: 1, pos: 31

### Candidate
type: B, layer: 1, pos: 31

### Candidate
type: B, layer: 1, pos: 16

### Candidate
type: B, layer: 1, pos: 23

### Candidate
type: A, layer: 1, pos: 22

### Candidate
type: B, layer: 1, pos: 11

## BFS NS instance: NS_A2_A1_B1_A2_B2_A2_B2_A1_A2

### Backsubstitution after applying NS history:
0: -25.2634239, 92.0110474, -21.3698139, 77.8816223, -103.1450500, 113.3808594
1: -66.5193863, 209.1605072, -56.5534439, 177.3153992, -243.8347778, 265.7139282
2: -99.8456650, 185.7543793, -84.9894714, 156.4965515, -256.3421631, 270.7438354
3: -57.3076057, 223.0561829, -48.6742172, 189.3233185, -246.6309204, 271.7303772
4: -91.2523117, 163.6518707, -77.6057510, 138.0177765, -229.2700653, 241.2575989

Time for backsubstitution: 1.05 seconds

### NS candidates at layer 1
type: B, layer: 1, pos: 41
type: B, layer: 1, pos: 14
type: A, layer: 1, pos: 14
type: B, layer: 1, pos: 19
type: B, layer: 1, pos: 12
type: A, layer: 1, pos: 19
type: B, layer: 1, pos: 25
type: B, layer: 1, pos: 20
type: A, layer: 1, pos: 41
type: A, layer: 1, pos: 25
type: B, layer: 1, pos: 37
type: B, layer: 1, pos: 39
type: B, layer: 1, pos: 29
type: B, layer: 1, pos: 21
type: A, layer: 1, pos: 12
type: B, layer: 1, pos: 8
type: A, layer: 1, pos: 37
type: A, layer: 1, pos: 29
type: A, layer: 1, pos: 23
type: A, layer: 1, pos: 6
type: A, layer: 1, pos: 11
type: A, layer: 1, pos: 20
type: B, layer: 1, pos: 22
type: A, layer: 1, pos: 46
type: B, layer: 1, pos: 4
type: B, layer: 1, pos: 6
type: A, layer: 1, pos: 38
type: B, layer: 1, pos: 46
type: A, layer: 1, pos: 15
type: B, layer: 1, pos: 15
type: B, layer: 1, pos: 13
type: A, layer: 1, pos: 42
type: A, layer: 1, pos: 4
type: A, layer: 1, pos: 16
type: B, layer: 1, pos: 16
type: B, layer: 1, pos: 42
type: A, layer: 1, pos: 31
type: B, layer: 1, pos: 31
type: B, layer: 1, pos: 38
type: B, layer: 1, pos: 11
type: A, layer: 1, pos: 22
type: A, layer: 1, pos: 21
type: B, layer: 1, pos: 23

Time for candidate selection: 0.07 seconds

### Candidate
type: B, layer: 1, pos: 41

### Candidate
type: B, layer: 1, pos: 14

### Candidate
type: A, layer: 1, pos: 14

### Candidate
type: B, layer: 1, pos: 19

### Candidate
type: B, layer: 1, pos: 12

### Candidate
type: A, layer: 1, pos: 19

### Candidate
type: B, layer: 1, pos: 25

### Candidate
type: B, layer: 1, pos: 20

### Candidate
type: A, layer: 1, pos: 41

### Candidate
type: A, layer: 1, pos: 25

### Candidate
type: B, layer: 1, pos: 37

### Candidate
type: B, layer: 1, pos: 39

## Relational analysis of NS_A2_A1_B1_A2_B2_A2_B2_A1_A2_B1
Optimization infeasible because this subproblem isn't reachable.

### Candidate
type: B, layer: 1, pos: 29

### Candidate
type: B, layer: 1, pos: 21

### Candidate
type: A, layer: 1, pos: 12

### Candidate
type: B, layer: 1, pos: 8

### Candidate
type: A, layer: 1, pos: 37

### Candidate
type: A, layer: 1, pos: 29

### Candidate
type: A, layer: 1, pos: 23

### Candidate
type: A, layer: 1, pos: 6

### Candidate
type: A, layer: 1, pos: 11

### Candidate
type: A, layer: 1, pos: 20

### Candidate
type: B, layer: 1, pos: 22

### Candidate
type: A, layer: 1, pos: 46

### Candidate
type: B, layer: 1, pos: 4

### Candidate
type: B, layer: 1, pos: 6

### Candidate
type: A, layer: 1, pos: 38

### Candidate
type: B, layer: 1, pos: 46

### Candidate
type: A, layer: 1, pos: 15

## Relational analysis of NS_A2_A1_B1_A2_B2_A2_B2_A1_A2_A1
Optimization infeasible because this subproblem isn't reachable.

### Candidate
type: B, layer: 1, pos: 15

### Candidate
type: B, layer: 1, pos: 13

### Candidate
type: A, layer: 1, pos: 42

## Relational analysis of NS_A2_A1_B1_A2_B2_A2_B2_A1_A2_A1
Optimization infeasible because this subproblem isn't reachable.

### Candidate
type: A, layer: 1, pos: 4

### Candidate
type: A, layer: 1, pos: 16

### Candidate
type: B, layer: 1, pos: 16

### Candidate
type: B, layer: 1, pos: 42

### Candidate
type: A, layer: 1, pos: 31

### Candidate
type: B, layer: 1, pos: 31

### Candidate
type: B, layer: 1, pos: 38

### Candidate
type: B, layer: 1, pos: 11

### Candidate
type: A, layer: 1, pos: 22

### Candidate
type: A, layer: 1, pos: 21

### Candidate
type: B, layer: 1, pos: 23

## BFS NS instance: NS_A2_A1_B1_A2_B2_A2_B2_A2_B1

### Backsubstitution after applying NS history:
0: -23.0073433, 84.0065536, -21.5613537, 78.5990067, -101.6063538, 105.5679016
1: -60.6873665, 191.3380280, -57.0527496, 179.0166779, -239.7040405, 248.3907776
2: -90.6737289, 169.3656616, -85.7162704, 158.0159607, -248.6896973, 255.0818939
3: -52.2170639, 204.0262451, -49.0925903, 191.3048096, -243.5218811, 253.1188354
4: -82.9182129, 149.2326202, -78.2719574, 139.3632202, -222.2814331, 227.5045776

Time for backsubstitution: 1.01 seconds

### NS candidates at layer 1
type: A, layer: 1, pos: 41
type: A, layer: 1, pos: 14
type: B, layer: 1, pos: 14
type: A, layer: 1, pos: 19
type: B, layer: 1, pos: 39
type: B, layer: 1, pos: 19
type: B, layer: 1, pos: 41
type: A, layer: 1, pos: 39
type: B, layer: 1, pos: 12
type: B, layer: 1, pos: 25
type: A, layer: 1, pos: 25
type: B, layer: 1, pos: 37
type: B, layer: 1, pos: 20
type: A, layer: 1, pos: 20
type: B, layer: 1, pos: 29
type: A, layer: 1, pos: 12
type: A, layer: 1, pos: 37
type: A, layer: 1, pos: 29
type: A, layer: 1, pos: 6
type: B, layer: 1, pos: 21
type: B, layer: 1, pos: 8
type: A, layer: 1, pos: 46
type: B, layer: 1, pos: 6
type: A, layer: 1, pos: 11
type: B, layer: 1, pos: 4
type: A, layer: 1, pos: 21
type: A, layer: 1, pos: 15
type: B, layer: 1, pos: 46
type: B, layer: 1, pos: 13
type: A, layer: 1, pos: 4
type: B, layer: 1, pos: 42
type: A, layer: 1, pos: 38
type: A, layer: 1, pos: 16
type: B, layer: 1, pos: 15
type: B, layer: 1, pos: 38
type: A, layer: 1, pos: 23
type: A, layer: 1, pos: 42
type: B, layer: 1, pos: 11
type: B, layer: 1, pos: 16
type: A, layer: 1, pos: 31
type: B, layer: 1, pos: 31
type: A, layer: 1, pos: 22
type: B, layer: 1, pos: 23

Time for candidate selection: 0.07 seconds

### Candidate
type: A, layer: 1, pos: 41

### Candidate
type: A, layer: 1, pos: 14

### Candidate
type: B, layer: 1, pos: 14

### Candidate
type: A, layer: 1, pos: 19

### Candidate
type: B, layer: 1, pos: 39

## Relational analysis of NS_A2_A1_B1_A2_B2_A2_B2_A2_B1_B1
Optimization infeasible because this subproblem isn't reachable.

### Candidate
type: B, layer: 1, pos: 19

### Candidate
type: B, layer: 1, pos: 41

### Candidate
type: A, layer: 1, pos: 39

### Candidate
type: B, layer: 1, pos: 12

### Candidate
type: B, layer: 1, pos: 25

### Candidate
type: A, layer: 1, pos: 25

### Candidate
type: B, layer: 1, pos: 37

### Candidate
type: B, layer: 1, pos: 20

## Relational analysis of NS_A2_A1_B1_A2_B2_A2_B2_A2_B1_B1
Optimization infeasible because this subproblem isn't reachable.

### Candidate
type: A, layer: 1, pos: 20

### Candidate
type: B, layer: 1, pos: 29

### Candidate
type: A, layer: 1, pos: 12

### Candidate
type: A, layer: 1, pos: 37

### Candidate
type: A, layer: 1, pos: 29

### Candidate
type: A, layer: 1, pos: 6

### Candidate
type: B, layer: 1, pos: 21

### Candidate
type: B, layer: 1, pos: 8

## Relational analysis of NS_A2_A1_B1_A2_B2_A2_B2_A2_B1_B1
Optimization infeasible because this subproblem isn't reachable.

### Candidate
type: A, layer: 1, pos: 46

### Candidate
type: B, layer: 1, pos: 6

### Candidate
type: A, layer: 1, pos: 11

## Relational analysis of NS_A2_A1_B1_A2_B2_A2_B2_A2_B1_A1

### Relational analysis result of NS_A2_A1_B1_A2_B2_A2_B2_A2_B1_A1
Status: Status.VERIFIED
Output dim: 0, lower bound: -81.8602045, upper bound: 81.8602833
time: 1.81 seconds

## Relational analysis of NS_A2_A1_B1_A2_B2_A2_B2_A2_B1_A2

### Relational analysis result of NS_A2_A1_B1_A2_B2_A2_B2_A2_B1_A2
Status: Status.UNKNOWN
Output dim: 0, lower bound: -81.8607798, upper bound: 81.8612180
time: 1.04 seconds

## BFS NS instance: NS_A2_A1_B1_A2_B2_A2_B2_A2_B2

### Backsubstitution after applying NS history:
0: -22.6294613, 82.5926514, -21.3220959, 77.5369644, -100.1664276, 103.9147415
1: -59.6350594, 188.3110962, -56.0987015, 177.6692657, -237.3043213, 244.4097748
2: -89.0393600, 166.5645752, -84.0279846, 156.1920319, -245.2313843, 250.5925598
3: -51.3182487, 200.7823029, -48.3225136, 189.7791748, -241.0974274, 249.1048126
4: -81.4321136, 146.7905273, -76.7525940, 137.8507843, -219.2828979, 223.5431061

Time for backsubstitution: 1.01 seconds

### NS candidates at layer 1
type: A, layer: 1, pos: 41
type: A, layer: 1, pos: 14
type: B, layer: 1, pos: 14
type: A, layer: 1, pos: 19
type: B, layer: 1, pos: 39
type: B, layer: 1, pos: 19
type: B, layer: 1, pos: 12
type: A, layer: 1, pos: 39
type: B, layer: 1, pos: 41
type: A, layer: 1, pos: 12
type: B, layer: 1, pos: 37
type: B, layer: 1, pos: 29
type: B, layer: 1, pos: 25
type: A, layer: 1, pos: 37
type: B, layer: 1, pos: 20
type: A, layer: 1, pos: 29
type: A, layer: 1, pos: 25
type: B, layer: 1, pos: 21
type: A, layer: 1, pos: 21
type: A, layer: 1, pos: 20
type: A, layer: 1, pos: 6
type: A, layer: 1, pos: 46
type: B, layer: 1, pos: 8
type: A, layer: 1, pos: 15
type: B, layer: 1, pos: 6
type: A, layer: 1, pos: 11
type: B, layer: 1, pos: 15
type: B, layer: 1, pos: 46
type: B, layer: 1, pos: 4
type: A, layer: 1, pos: 38
type: B, layer: 1, pos: 13
type: A, layer: 1, pos: 16
type: A, layer: 1, pos: 22
type: B, layer: 1, pos: 16
type: A, layer: 1, pos: 4
type: B, layer: 1, pos: 38
type: A, layer: 1, pos: 23
type: B, layer: 1, pos: 42
type: A, layer: 1, pos: 42
type: A, layer: 1, pos: 31
type: B, layer: 1, pos: 31
type: B, layer: 1, pos: 11
type: B, layer: 1, pos: 23

Time for candidate selection: 0.09 seconds

### Candidate
type: A, layer: 1, pos: 41

### Candidate
type: A, layer: 1, pos: 14

### Candidate
type: B, layer: 1, pos: 14

### Candidate
type: A, layer: 1, pos: 19

### Candidate
type: B, layer: 1, pos: 39

## Relational analysis of NS_A2_A1_B1_A2_B2_A2_B2_A2_B2_B1
Optimization infeasible because this subproblem isn't reachable.

### Candidate
type: B, layer: 1, pos: 19

### Candidate
type: B, layer: 1, pos: 12

### Candidate
type: A, layer: 1, pos: 39

### Candidate
type: B, layer: 1, pos: 41

### Candidate
type: A, layer: 1, pos: 12

### Candidate
type: B, layer: 1, pos: 37

### Candidate
type: B, layer: 1, pos: 29

### Candidate
type: B, layer: 1, pos: 25

### Candidate
type: A, layer: 1, pos: 37

### Candidate
type: B, layer: 1, pos: 20

## Relational analysis of NS_A2_A1_B1_A2_B2_A2_B2_A2_B2_B1
Optimization infeasible because this subproblem isn't reachable.

### Candidate
type: A, layer: 1, pos: 29

### Candidate
type: A, layer: 1, pos: 25

### Candidate
type: B, layer: 1, pos: 21

### Candidate
type: A, layer: 1, pos: 21

### Candidate
type: A, layer: 1, pos: 20

### Candidate
type: A, layer: 1, pos: 6

### Candidate
type: A, layer: 1, pos: 46

### Candidate
type: B, layer: 1, pos: 8

## Relational analysis of NS_A2_A1_B1_A2_B2_A2_B2_A2_B2_B1
Optimization infeasible because this subproblem isn't reachable.

### Candidate
type: A, layer: 1, pos: 15

## Relational analysis of NS_A2_A1_B1_A2_B2_A2_B2_A2_B2_A1

### Relational analysis result of NS_A2_A1_B1_A2_B2_A2_B2_A2_B2_A1
Status: Status.VERIFIED
Output dim: 0, lower bound: -81.8600707, upper bound: 81.8605823
time: 0.89 seconds

## Relational analysis of NS_A2_A1_B1_A2_B2_A2_B2_A2_B2_A2

### Relational analysis result of NS_A2_A1_B1_A2_B2_A2_B2_A2_B2_A2
Status: Status.UNKNOWN
Output dim: 0, lower bound: -81.8605873, upper bound: 81.8612151
time: 0.88 seconds

## BFS NS instance: NS_A2_A1_B2_A2_A2_B1_A2_B1_B1

### Backsubstitution after applying NS history:
0: -18.6952496, 68.3234482, -18.3124771, 67.1370773, -85.8323288, 86.6359177
1: -49.6338654, 154.1445160, -48.8487778, 150.3119659, -199.9458160, 202.9932556
2: -74.3940125, 137.2646027, -73.5225601, 134.3159027, -208.7099152, 210.7871704
3: -42.6771584, 164.6407623, -41.9497757, 160.5357361, -203.2128906, 206.5905457
4: -67.9915771, 120.8721466, -67.1424255, 118.2522125, -186.2437897, 188.0145569

Time for backsubstitution: 0.99 seconds

### NS candidates at layer 1
type: B, layer: 1, pos: 41
type: A, layer: 1, pos: 41
type: A, layer: 1, pos: 39
type: B, layer: 1, pos: 39
type: B, layer: 1, pos: 19
type: A, layer: 1, pos: 19
type: A, layer: 1, pos: 20
type: B, layer: 1, pos: 4
type: B, layer: 1, pos: 25
type: B, layer: 1, pos: 26
type: A, layer: 1, pos: 14
type: A, layer: 1, pos: 26
type: A, layer: 1, pos: 23
type: B, layer: 1, pos: 14
type: B, layer: 1, pos: 6
type: A, layer: 1, pos: 25
type: B, layer: 1, pos: 23
type: A, layer: 1, pos: 6
type: A, layer: 1, pos: 21
type: B, layer: 1, pos: 21
type: B, layer: 1, pos: 29
type: B, layer: 1, pos: 46
type: B, layer: 1, pos: 38
type: B, layer: 1, pos: 13
type: A, layer: 1, pos: 46
type: A, layer: 1, pos: 16
type: B, layer: 1, pos: 12
type: B, layer: 1, pos: 37
type: A, layer: 1, pos: 11
type: B, layer: 1, pos: 22
type: A, layer: 1, pos: 38
type: A, layer: 1, pos: 12
type: A, layer: 1, pos: 42
type: B, layer: 1, pos: 16
type: A, layer: 1, pos: 8
type: A, layer: 1, pos: 31
type: B, layer: 1, pos: 42
type: B, layer: 1, pos: 31
type: A, layer: 1, pos: 29
type: B, layer: 1, pos: 11
type: B, layer: 1, pos: 8
type: A, layer: 1, pos: 37
type: A, layer: 1, pos: 22

Time for candidate selection: 0.10 seconds

### Candidate
type: B, layer: 1, pos: 41

### Candidate
type: A, layer: 1, pos: 41

### Candidate
type: A, layer: 1, pos: 39

### Candidate
type: B, layer: 1, pos: 39

### Candidate
type: B, layer: 1, pos: 19

### Candidate
type: A, layer: 1, pos: 19

### Candidate
type: A, layer: 1, pos: 20

### Candidate
type: B, layer: 1, pos: 4

### Candidate
type: B, layer: 1, pos: 25

### Candidate
type: B, layer: 1, pos: 26

### Candidate
type: A, layer: 1, pos: 14

### Candidate
type: A, layer: 1, pos: 26

### Candidate
type: A, layer: 1, pos: 23

### Candidate
type: B, layer: 1, pos: 14

### Candidate
type: B, layer: 1, pos: 6

### Candidate
type: A, layer: 1, pos: 25

### Candidate
type: B, layer: 1, pos: 23

### Candidate
type: A, layer: 1, pos: 6

### Candidate
type: A, layer: 1, pos: 21

### Candidate
type: B, layer: 1, pos: 21

### Candidate
type: B, layer: 1, pos: 29

### Candidate
type: B, layer: 1, pos: 46

### Candidate
type: B, layer: 1, pos: 38

### Candidate
type: B, layer: 1, pos: 13

### Candidate
type: A, layer: 1, pos: 46

### Candidate
type: A, layer: 1, pos: 16

### Candidate
type: B, layer: 1, pos: 12

### Candidate
type: B, layer: 1, pos: 37

## Relational analysis of NS_A2_A1_B2_A2_A2_B1_A2_B1_B1_B1

### Relational analysis result of NS_A2_A1_B2_A2_A2_B1_A2_B1_B1_B1
Status: Status.UNKNOWN
Output dim: 0, lower bound: -81.8627886, upper bound: 81.8627585
time: 0.86 seconds

## Relational analysis of NS_A2_A1_B2_A2_A2_B1_A2_B1_B1_B2
Optimization infeasible because this subproblem isn't reachable.

### Candidate
type: A, layer: 1, pos: 11

### Candidate
type: B, layer: 1, pos: 22

## Relational analysis of NS_A2_A1_B2_A2_A2_B1_A2_B1_B1_B1
Optimization infeasible because this subproblem isn't reachable.

### Candidate
type: A, layer: 1, pos: 38

### Candidate
type: A, layer: 1, pos: 12

### Candidate
type: A, layer: 1, pos: 42

## Relational analysis of NS_A2_A1_B2_A2_A2_B1_A2_B1_B1_A1
Optimization infeasible because this subproblem isn't reachable.

### Candidate
type: B, layer: 1, pos: 16

### Candidate
type: A, layer: 1, pos: 8

## Relational analysis of NS_A2_A1_B2_A2_A2_B1_A2_B1_B1_A1
Optimization infeasible because this subproblem isn't reachable.

### Candidate
type: A, layer: 1, pos: 31

### Candidate
type: B, layer: 1, pos: 42

### Candidate
type: B, layer: 1, pos: 31

### Candidate
type: A, layer: 1, pos: 29

### Candidate
type: B, layer: 1, pos: 11

### Candidate
type: B, layer: 1, pos: 8

### Candidate
type: A, layer: 1, pos: 37

### Candidate
type: A, layer: 1, pos: 22

## BFS NS instance: NS_A2_A1_B2_A2_A2_B1_A2_B1_B2

### Backsubstitution after applying NS history:
0: -18.6952496, 68.3234482, -19.0650463, 69.8031311, -88.4983826, 87.3884811
1: -49.6338654, 154.1445160, -50.6033974, 157.1330719, -206.7669373, 204.7478790
2: -74.3940125, 137.2646027, -75.5729370, 140.5175629, -214.9115601, 212.8375244
3: -42.6771584, 164.6407623, -43.4897385, 167.5281982, -210.2053528, 208.1304932
4: -67.9915771, 120.8721466, -69.1845322, 123.6557465, -191.6473236, 190.0566711

Time for backsubstitution: 1.08 seconds

### NS candidates at layer 1
type: B, layer: 1, pos: 41
type: A, layer: 1, pos: 41
type: A, layer: 1, pos: 39
type: B, layer: 1, pos: 39
type: B, layer: 1, pos: 19
type: A, layer: 1, pos: 19
type: A, layer: 1, pos: 20
type: B, layer: 1, pos: 4
type: A, layer: 1, pos: 26
type: A, layer: 1, pos: 14
type: B, layer: 1, pos: 25
type: A, layer: 1, pos: 23
type: A, layer: 1, pos: 25
type: B, layer: 1, pos: 26
type: B, layer: 1, pos: 14
type: B, layer: 1, pos: 6
type: A, layer: 1, pos: 21
type: B, layer: 1, pos: 23
type: A, layer: 1, pos: 6
type: B, layer: 1, pos: 46
type: B, layer: 1, pos: 38
type: B, layer: 1, pos: 29
type: B, layer: 1, pos: 21
type: B, layer: 1, pos: 13
type: A, layer: 1, pos: 46
type: A, layer: 1, pos: 16
type: B, layer: 1, pos: 12
type: A, layer: 1, pos: 12
type: A, layer: 1, pos: 11
type: B, layer: 1, pos: 37
type: B, layer: 1, pos: 22
type: B, layer: 1, pos: 16
type: A, layer: 1, pos: 38
type: A, layer: 1, pos: 8
type: A, layer: 1, pos: 42
type: A, layer: 1, pos: 29
type: B, layer: 1, pos: 42
type: A, layer: 1, pos: 31
type: B, layer: 1, pos: 31
type: B, layer: 1, pos: 11
type: A, layer: 1, pos: 37
type: A, layer: 1, pos: 22
type: B, layer: 1, pos: 8

Time for candidate selection: 0.09 seconds

### Candidate
type: B, layer: 1, pos: 41

### Candidate
type: A, layer: 1, pos: 41

### Candidate
type: A, layer: 1, pos: 39

### Candidate
type: B, layer: 1, pos: 39

### Candidate
type: B, layer: 1, pos: 19

### Candidate
type: A, layer: 1, pos: 19

### Candidate
type: A, layer: 1, pos: 20

### Candidate
type: B, layer: 1, pos: 4

### Candidate
type: A, layer: 1, pos: 26

### Candidate
type: A, layer: 1, pos: 14

### Candidate
type: B, layer: 1, pos: 25

### Candidate
type: A, layer: 1, pos: 23

### Candidate
type: A, layer: 1, pos: 25

### Candidate
type: B, layer: 1, pos: 26

### Candidate
type: B, layer: 1, pos: 14

### Candidate
type: B, layer: 1, pos: 6

### Candidate
type: A, layer: 1, pos: 21

### Candidate
type: B, layer: 1, pos: 23

### Candidate
type: A, layer: 1, pos: 6

### Candidate
type: B, layer: 1, pos: 46

### Candidate
type: B, layer: 1, pos: 38

### Candidate
type: B, layer: 1, pos: 29

### Candidate
type: B, layer: 1, pos: 21

### Candidate
type: B, layer: 1, pos: 13

### Candidate
type: A, layer: 1, pos: 46

### Candidate
type: A, layer: 1, pos: 16

### Candidate
type: B, layer: 1, pos: 12

### Candidate
type: A, layer: 1, pos: 12

### Candidate
type: A, layer: 1, pos: 11

### Candidate
type: B, layer: 1, pos: 37

## Relational analysis of NS_A2_A1_B2_A2_A2_B1_A2_B1_B2_B1
Optimization infeasible because this subproblem isn't reachable.

### Candidate
type: B, layer: 1, pos: 22

## Relational analysis of NS_A2_A1_B2_A2_A2_B1_A2_B1_B2_B1
Optimization infeasible because this subproblem isn't reachable.

### Candidate
type: B, layer: 1, pos: 16

### Candidate
type: A, layer: 1, pos: 38

### Candidate
type: A, layer: 1, pos: 8

## Relational analysis of NS_A2_A1_B2_A2_A2_B1_A2_B1_B2_A1
Optimization infeasible because this subproblem isn't reachable.

### Candidate
type: A, layer: 1, pos: 42

## Relational analysis of NS_A2_A1_B2_A2_A2_B1_A2_B1_B2_A1
Optimization infeasible because this subproblem isn't reachable.

### Candidate
type: A, layer: 1, pos: 29

### Candidate
type: B, layer: 1, pos: 42

### Candidate
type: A, layer: 1, pos: 31

### Candidate
type: B, layer: 1, pos: 31

### Candidate
type: B, layer: 1, pos: 11

### Candidate
type: A, layer: 1, pos: 37

### Candidate
type: A, layer: 1, pos: 22

### Candidate
type: B, layer: 1, pos: 8

## BFS NS instance: NS_A2_A1_B2_A2_A2_B1_A2_B2_A1

### Backsubstitution after applying NS history:
0: -18.4144020, 67.4136810, -20.1860847, 73.7450256, -92.1594238, 87.5997620
1: -48.8962708, 151.8069458, -53.6063957, 166.3255768, -215.2218475, 205.4133453
2: -72.9979477, 135.7030182, -80.3843460, 148.0995483, -221.0975037, 216.0873566
3: -42.0129814, 161.9240112, -46.1106377, 177.6928101, -219.7057190, 208.0346527
4: -66.8281021, 119.4361267, -73.4653473, 130.4210968, -197.2492065, 192.9014740

Time for backsubstitution: 0.95 seconds

### NS candidates at layer 1
type: B, layer: 1, pos: 41
type: A, layer: 1, pos: 25
type: A, layer: 1, pos: 41
type: B, layer: 1, pos: 25
type: B, layer: 1, pos: 19
type: A, layer: 1, pos: 12
type: B, layer: 1, pos: 14
type: B, layer: 1, pos: 26
type: A, layer: 1, pos: 29
type: B, layer: 1, pos: 4
type: B, layer: 1, pos: 12
type: A, layer: 1, pos: 14
type: A, layer: 1, pos: 37
type: B, layer: 1, pos: 21
type: B, layer: 1, pos: 23
type: A, layer: 1, pos: 26
type: A, layer: 1, pos: 23
type: B, layer: 1, pos: 8
type: A, layer: 1, pos: 19
type: A, layer: 1, pos: 8
type: B, layer: 1, pos: 29
type: B, layer: 1, pos: 37
type: B, layer: 1, pos: 15
type: A, layer: 1, pos: 6
type: A, layer: 1, pos: 22
type: B, layer: 1, pos: 11
type: B, layer: 1, pos: 22
type: B, layer: 1, pos: 6
type: B, layer: 1, pos: 46
type: A, layer: 1, pos: 38
type: A, layer: 1, pos: 46
type: B, layer: 1, pos: 39
type: B, layer: 1, pos: 13
type: B, layer: 1, pos: 38
type: A, layer: 1, pos: 11
type: B, layer: 1, pos: 16
type: A, layer: 1, pos: 31
type: B, layer: 1, pos: 31
type: A, layer: 1, pos: 16
type: B, layer: 1, pos: 42
type: A, layer: 1, pos: 42
type: A, layer: 1, pos: 39
type: A, layer: 1, pos: 21

Time for candidate selection: 0.08 seconds

### Candidate
type: B, layer: 1, pos: 41

### Candidate
type: A, layer: 1, pos: 25

### Candidate
type: A, layer: 1, pos: 41

### Candidate
type: B, layer: 1, pos: 25

### Candidate
type: B, layer: 1, pos: 19

### Candidate
type: A, layer: 1, pos: 12

### Candidate
type: B, layer: 1, pos: 14

### Candidate
type: B, layer: 1, pos: 26

### Candidate
type: A, layer: 1, pos: 29

### Candidate
type: B, layer: 1, pos: 4

### Candidate
type: B, layer: 1, pos: 12

### Candidate
type: A, layer: 1, pos: 14

### Candidate
type: A, layer: 1, pos: 37

## Relational analysis of NS_A2_A1_B2_A2_A2_B1_A2_B2_A1_A1

### Relational analysis result of NS_A2_A1_B2_A2_A2_B1_A2_B2_A1_A1
Status: Status.UNKNOWN
Output dim: 0, lower bound: -81.8629138, upper bound: 81.8631198
time: 0.95 seconds

## Relational analysis of NS_A2_A1_B2_A2_A2_B1_A2_B2_A1_A2

### Relational analysis result of NS_A2_A1_B2_A2_A2_B1_A2_B2_A1_A2
Status: Status.UNKNOWN
Output dim: 0, lower bound: -81.8627166, upper bound: 81.8627916
time: 1.14 seconds

## BFS NS instance: NS_A2_A1_B2_A2_A2_B1_A2_B2_A2

### Backsubstitution after applying NS history:
0: -19.2693539, 70.4004974, -20.1860847, 73.7450256, -93.0143814, 90.5865784
1: -51.1902962, 158.7684021, -53.6063957, 166.3255768, -217.5158691, 212.3748016
2: -76.7452240, 141.3838196, -80.3843460, 148.0995483, -224.8447723, 221.7681580
3: -44.0220375, 169.6260834, -46.1106377, 177.6928101, -221.7148285, 215.7367249
4: -70.1366730, 124.5034332, -73.4653473, 130.4210968, -200.5577698, 197.9687805

Time for backsubstitution: 0.94 seconds

### NS candidates at layer 1
type: B, layer: 1, pos: 41
type: A, layer: 1, pos: 41
type: B, layer: 1, pos: 19
type: B, layer: 1, pos: 14
type: A, layer: 1, pos: 14
type: B, layer: 1, pos: 39
type: A, layer: 1, pos: 26
type: B, layer: 1, pos: 26
type: A, layer: 1, pos: 19
type: A, layer: 1, pos: 39
type: A, layer: 1, pos: 25
type: B, layer: 1, pos: 25
type: B, layer: 1, pos: 4
type: A, layer: 1, pos: 12
type: A, layer: 1, pos: 37
type: B, layer: 1, pos: 12
type: A, layer: 1, pos: 8
type: B, layer: 1, pos: 37
type: B, layer: 1, pos: 8
type: A, layer: 1, pos: 29
type: B, layer: 1, pos: 29
type: B, layer: 1, pos: 15
type: B, layer: 1, pos: 46
type: A, layer: 1, pos: 6
type: B, layer: 1, pos: 21
type: B, layer: 1, pos: 6
type: B, layer: 1, pos: 11
type: A, layer: 1, pos: 46
type: B, layer: 1, pos: 22
type: A, layer: 1, pos: 21
type: A, layer: 1, pos: 22
type: B, layer: 1, pos: 13
type: A, layer: 1, pos: 38
type: A, layer: 1, pos: 42
type: A, layer: 1, pos: 23
type: A, layer: 1, pos: 11
type: B, layer: 1, pos: 42
type: B, layer: 1, pos: 38
type: B, layer: 1, pos: 23
type: A, layer: 1, pos: 31
type: B, layer: 1, pos: 16
type: B, layer: 1, pos: 31
type: A, layer: 1, pos: 16

Time for candidate selection: 0.08 seconds

### Candidate
type: B, layer: 1, pos: 41

### Candidate
type: A, layer: 1, pos: 41

### Candidate
type: B, layer: 1, pos: 19

### Candidate
type: B, layer: 1, pos: 14

### Candidate
type: A, layer: 1, pos: 14

### Candidate
type: B, layer: 1, pos: 39

### Candidate
type: A, layer: 1, pos: 26

### Candidate
type: B, layer: 1, pos: 26

### Candidate
type: A, layer: 1, pos: 19

### Candidate
type: A, layer: 1, pos: 39

### Candidate
type: A, layer: 1, pos: 25

### Candidate
type: B, layer: 1, pos: 25

### Candidate
type: B, layer: 1, pos: 4

### Candidate
type: A, layer: 1, pos: 12

### Candidate
type: A, layer: 1, pos: 37

## Relational analysis of NS_A2_A1_B2_A2_A2_B1_A2_B2_A2_A1

### Relational analysis result of NS_A2_A1_B2_A2_A2_B1_A2_B2_A2_A1
Status: Status.UNKNOWN
Output dim: 0, lower bound: -81.8629138, upper bound: 81.8631260
time: 0.83 seconds

## Relational analysis of NS_A2_A1_B2_A2_A2_B1_A2_B2_A2_A2

### Relational analysis result of NS_A2_A1_B2_A2_A2_B1_A2_B2_A2_A2
Status: Status.UNKNOWN
Output dim: 0, lower bound: -81.8627166, upper bound: 81.8627600
time: 0.94 seconds

## BFS NS instance: NS_A2_A1_B2_A2_A2_B2_A2_B1_A1

### Backsubstitution after applying NS history:
0: -19.7194328, 71.9988174, -19.8240700, 72.2916412, -92.0110626, 91.8228683
1: -52.2890434, 162.5952606, -52.7287102, 162.9240723, -215.2131195, 215.3239746
2: -78.3183746, 144.8326874, -79.1922989, 145.3326416, -223.6510162, 224.0249634
3: -45.0192795, 173.6589966, -45.3761864, 174.3841248, -219.4034119, 219.0351562
4: -71.6043396, 127.5748749, -72.3446503, 128.0694885, -199.6737976, 199.9194946

Time for backsubstitution: 0.96 seconds

### NS candidates at layer 1
type: A, layer: 1, pos: 41
type: B, layer: 1, pos: 19
type: A, layer: 1, pos: 14
type: B, layer: 1, pos: 14
type: B, layer: 1, pos: 41
type: B, layer: 1, pos: 39
type: B, layer: 1, pos: 26
type: A, layer: 1, pos: 26
type: A, layer: 1, pos: 19
type: A, layer: 1, pos: 39
type: B, layer: 1, pos: 12
type: A, layer: 1, pos: 25
type: B, layer: 1, pos: 25
type: B, layer: 1, pos: 37
type: B, layer: 1, pos: 4
type: A, layer: 1, pos: 12
type: B, layer: 1, pos: 20
type: A, layer: 1, pos: 20
type: B, layer: 1, pos: 29
type: A, layer: 1, pos: 37
type: A, layer: 1, pos: 29
type: B, layer: 1, pos: 21
type: A, layer: 1, pos: 6
type: B, layer: 1, pos: 8
type: A, layer: 1, pos: 11
type: B, layer: 1, pos: 6
type: A, layer: 1, pos: 46
type: A, layer: 1, pos: 15
type: B, layer: 1, pos: 46
type: A, layer: 1, pos: 21
type: A, layer: 1, pos: 16
type: B, layer: 1, pos: 13
type: A, layer: 1, pos: 23
type: B, layer: 1, pos: 15
type: A, layer: 1, pos: 38
type: A, layer: 1, pos: 42
type: B, layer: 1, pos: 38
type: B, layer: 1, pos: 42
type: B, layer: 1, pos: 11
type: A, layer: 1, pos: 31
type: B, layer: 1, pos: 31
type: B, layer: 1, pos: 23
type: B, layer: 1, pos: 16

Time for candidate selection: 0.08 seconds

### Candidate
type: A, layer: 1, pos: 41

### Candidate
type: B, layer: 1, pos: 19

### Candidate
type: A, layer: 1, pos: 14

### Candidate
type: B, layer: 1, pos: 14

### Candidate
type: B, layer: 1, pos: 41

### Candidate
type: B, layer: 1, pos: 39

### Candidate
type: B, layer: 1, pos: 26

### Candidate
type: A, layer: 1, pos: 26

### Candidate
type: A, layer: 1, pos: 19

### Candidate
type: A, layer: 1, pos: 39

### Candidate
type: B, layer: 1, pos: 12

### Candidate
type: A, layer: 1, pos: 25

### Candidate
type: B, layer: 1, pos: 25

### Candidate
type: B, layer: 1, pos: 37

### Candidate
type: B, layer: 1, pos: 4

### Candidate
type: A, layer: 1, pos: 12

### Candidate
type: B, layer: 1, pos: 20

### Candidate
type: A, layer: 1, pos: 20

### Candidate
type: B, layer: 1, pos: 29

### Candidate
type: A, layer: 1, pos: 37

### Candidate
type: A, layer: 1, pos: 29

### Candidate
type: B, layer: 1, pos: 21

### Candidate
type: A, layer: 1, pos: 6

### Candidate
type: B, layer: 1, pos: 8

### Candidate
type: A, layer: 1, pos: 11

### Candidate
type: B, layer: 1, pos: 6

### Candidate
type: A, layer: 1, pos: 46

### Candidate
type: A, layer: 1, pos: 15

### Candidate
type: B, layer: 1, pos: 46

### Candidate
type: A, layer: 1, pos: 21

### Candidate
type: A, layer: 1, pos: 16

### Candidate
type: B, layer: 1, pos: 13

### Candidate
type: A, layer: 1, pos: 23

### Candidate
type: B, layer: 1, pos: 15

### Candidate
type: A, layer: 1, pos: 38

### Candidate
type: A, layer: 1, pos: 42

### Candidate
type: B, layer: 1, pos: 38

### Candidate
type: B, layer: 1, pos: 42

### Candidate
type: B, layer: 1, pos: 11

### Candidate
type: A, layer: 1, pos: 31

### Candidate
type: B, layer: 1, pos: 31

### Candidate
type: B, layer: 1, pos: 23

### Candidate
type: B, layer: 1, pos: 16

## BFS NS instance: NS_A2_A1_B2_A2_A2_B2_A2_B1_A2

### Backsubstitution after applying NS history:
0: -19.8083668, 72.1840134, -19.8240700, 72.2916412, -92.1000061, 92.0080643
1: -52.2716103, 163.8082581, -52.7287102, 162.9240723, -215.1956787, 216.5369720
2: -77.9788666, 145.6064148, -79.1922989, 145.3326416, -223.3114929, 224.7987061
3: -45.0483627, 174.8100128, -45.3761864, 174.3841248, -219.4324951, 220.1862030
4: -71.3640823, 128.2841187, -72.3446503, 128.0694885, -199.4335632, 200.6287689

Time for backsubstitution: 0.96 seconds

### NS candidates at layer 1
type: B, layer: 1, pos: 19
type: B, layer: 1, pos: 39
type: B, layer: 1, pos: 41
type: B, layer: 1, pos: 14
type: A, layer: 1, pos: 41
type: A, layer: 1, pos: 14
type: B, layer: 1, pos: 26
type: B, layer: 1, pos: 25
type: A, layer: 1, pos: 26
type: B, layer: 1, pos: 37
type: A, layer: 1, pos: 19
type: A, layer: 1, pos: 25
type: B, layer: 1, pos: 12
type: A, layer: 1, pos: 39
type: B, layer: 1, pos: 8
type: A, layer: 1, pos: 20
type: B, layer: 1, pos: 29
type: B, layer: 1, pos: 4
type: A, layer: 1, pos: 12
type: A, layer: 1, pos: 37
type: B, layer: 1, pos: 20
type: B, layer: 1, pos: 13
type: A, layer: 1, pos: 29
type: A, layer: 1, pos: 16
type: B, layer: 1, pos: 21
type: A, layer: 1, pos: 46
type: A, layer: 1, pos: 11
type: A, layer: 1, pos: 15
type: B, layer: 1, pos: 46
type: B, layer: 1, pos: 16
type: B, layer: 1, pos: 42
type: B, layer: 1, pos: 38
type: A, layer: 1, pos: 23
type: B, layer: 1, pos: 15
type: A, layer: 1, pos: 42
type: A, layer: 1, pos: 6
type: A, layer: 1, pos: 21
type: A, layer: 1, pos: 38
type: B, layer: 1, pos: 6
type: B, layer: 1, pos: 23
type: A, layer: 1, pos: 31
type: B, layer: 1, pos: 31
type: B, layer: 1, pos: 11

Time for candidate selection: 0.08 seconds

### Candidate
type: B, layer: 1, pos: 19

### Candidate
type: B, layer: 1, pos: 39

### Candidate
type: B, layer: 1, pos: 41

### Candidate
type: B, layer: 1, pos: 14

### Candidate
type: A, layer: 1, pos: 41

### Candidate
type: A, layer: 1, pos: 14

### Candidate
type: B, layer: 1, pos: 26

### Candidate
type: B, layer: 1, pos: 25

### Candidate
type: A, layer: 1, pos: 26

### Candidate
type: B, layer: 1, pos: 37

### Candidate
type: A, layer: 1, pos: 19

### Candidate
type: A, layer: 1, pos: 25

### Candidate
type: B, layer: 1, pos: 12

### Candidate
type: A, layer: 1, pos: 39

### Candidate
type: B, layer: 1, pos: 8

### Candidate
type: A, layer: 1, pos: 20

### Candidate
type: B, layer: 1, pos: 29

### Candidate
type: B, layer: 1, pos: 4

### Candidate
type: A, layer: 1, pos: 12

### Candidate
type: A, layer: 1, pos: 37

### Candidate
type: B, layer: 1, pos: 20

### Candidate
type: B, layer: 1, pos: 13

### Candidate
type: A, layer: 1, pos: 29

### Candidate
type: A, layer: 1, pos: 16

### Candidate
type: B, layer: 1, pos: 21

### Candidate
type: A, layer: 1, pos: 46

### Candidate
type: A, layer: 1, pos: 11

### Candidate
type: A, layer: 1, pos: 15

### Candidate
type: B, layer: 1, pos: 46

### Candidate
type: B, layer: 1, pos: 16

### Candidate
type: B, layer: 1, pos: 42

### Candidate
type: B, layer: 1, pos: 38

### Candidate
type: A, layer: 1, pos: 23

### Candidate
type: B, layer: 1, pos: 15

### Candidate
type: A, layer: 1, pos: 42

### Candidate
type: A, layer: 1, pos: 6

### Candidate
type: A, layer: 1, pos: 21

### Candidate
type: A, layer: 1, pos: 38

### Candidate
type: B, layer: 1, pos: 6

### Candidate
type: B, layer: 1, pos: 23

### Candidate
type: A, layer: 1, pos: 31

### Candidate
type: B, layer: 1, pos: 31

### Candidate
type: B, layer: 1, pos: 11

## BFS NS instance: NS_A2_A1_B2_A2_A2_B2_A2_B2_A2

### Backsubstitution after applying NS history:
0: -19.3557186, 70.6018448, -19.9155350, 72.6295166, -91.9852371, 90.5173798
1: -51.2920113, 159.5831757, -52.7646866, 164.1979218, -215.4899292, 212.3478546
2: -76.8068695, 142.0757141, -78.8302460, 146.3729095, -223.1797638, 220.9059601
3: -44.1711006, 170.4850922, -45.4274750, 175.5338593, -219.7049561, 215.9125671
4: -70.2195587, 125.1684570, -72.1023483, 128.9653168, -199.1848755, 197.2708130

Time for backsubstitution: 0.96 seconds

### NS candidates at layer 1
type: B, layer: 1, pos: 41
type: A, layer: 1, pos: 41
type: A, layer: 1, pos: 14
type: B, layer: 1, pos: 14
type: A, layer: 1, pos: 19
type: B, layer: 1, pos: 39
type: A, layer: 1, pos: 26
type: B, layer: 1, pos: 26
type: B, layer: 1, pos: 19
type: A, layer: 1, pos: 39
type: B, layer: 1, pos: 25
type: B, layer: 1, pos: 12
type: A, layer: 1, pos: 25
type: B, layer: 1, pos: 4
type: B, layer: 1, pos: 37
type: A, layer: 1, pos: 12
type: A, layer: 1, pos: 20
type: B, layer: 1, pos: 20
type: A, layer: 1, pos: 37
type: B, layer: 1, pos: 29
type: A, layer: 1, pos: 29
type: B, layer: 1, pos: 11
type: A, layer: 1, pos: 6
type: B, layer: 1, pos: 8
type: B, layer: 1, pos: 21
type: B, layer: 1, pos: 6
type: B, layer: 1, pos: 46
type: A, layer: 1, pos: 21
type: A, layer: 1, pos: 46
type: A, layer: 1, pos: 15
type: B, layer: 1, pos: 15
type: B, layer: 1, pos: 13
type: A, layer: 1, pos: 38
type: A, layer: 1, pos: 23
type: A, layer: 1, pos: 42
type: A, layer: 1, pos: 16
type: B, layer: 1, pos: 42
type: B, layer: 1, pos: 38
type: A, layer: 1, pos: 31
type: B, layer: 1, pos: 16
type: B, layer: 1, pos: 31
type: A, layer: 1, pos: 22
type: B, layer: 1, pos: 23

Time for candidate selection: 0.08 seconds

### Candidate
type: B, layer: 1, pos: 41

### Candidate
type: A, layer: 1, pos: 41

### Candidate
type: A, layer: 1, pos: 14

### Candidate
type: B, layer: 1, pos: 14

### Candidate
type: A, layer: 1, pos: 19

### Candidate
type: B, layer: 1, pos: 39

### Candidate
type: A, layer: 1, pos: 26

### Candidate
type: B, layer: 1, pos: 26

### Candidate
type: B, layer: 1, pos: 19

### Candidate
type: A, layer: 1, pos: 39

### Candidate
type: B, layer: 1, pos: 25

### Candidate
type: B, layer: 1, pos: 12

### Candidate
type: A, layer: 1, pos: 25

### Candidate
type: B, layer: 1, pos: 4

### Candidate
type: B, layer: 1, pos: 37

### Candidate
type: A, layer: 1, pos: 12

### Candidate
type: A, layer: 1, pos: 20

### Candidate
type: B, layer: 1, pos: 20

### Candidate
type: A, layer: 1, pos: 37

### Candidate
type: B, layer: 1, pos: 29

### Candidate
type: A, layer: 1, pos: 29

### Candidate
type: B, layer: 1, pos: 11

## Relational analysis of NS_A2_A1_B2_A2_A2_B2_A2_B2_A2_B1
Optimization infeasible because this subproblem isn't reachable.

### Candidate
type: A, layer: 1, pos: 6

### Candidate
type: B, layer: 1, pos: 8

### Candidate
type: B, layer: 1, pos: 21

### Candidate
type: B, layer: 1, pos: 6

### Candidate
type: B, layer: 1, pos: 46

### Candidate
type: A, layer: 1, pos: 21

### Candidate
type: A, layer: 1, pos: 46

### Candidate
type: A, layer: 1, pos: 15

## Relational analysis of NS_A2_A1_B2_A2_A2_B2_A2_B2_A2_A1
Optimization infeasible because this subproblem isn't reachable.

### Candidate
type: B, layer: 1, pos: 15

### Candidate
type: B, layer: 1, pos: 13

### Candidate
type: A, layer: 1, pos: 38

### Candidate
type: A, layer: 1, pos: 23

### Candidate
type: A, layer: 1, pos: 42

## Relational analysis of NS_A2_A1_B2_A2_A2_B2_A2_B2_A2_A1

### Relational analysis result of NS_A2_A1_B2_A2_A2_B2_A2_B2_A2_A1
Status: Status.VERIFIED
Output dim: 0, lower bound: -81.8603468, upper bound: 81.8608172
time: 0.93 seconds

## Relational analysis of NS_A2_A1_B2_A2_A2_B2_A2_B2_A2_A2

### Relational analysis result of NS_A2_A1_B2_A2_A2_B2_A2_B2_A2_A2
Status: Status.UNKNOWN
Output dim: 0, lower bound: -81.8604926, upper bound: 81.8610738
time: 0.95 seconds

## Summary of splitting at layer (split count: 9)
- Time for NS candidates: 3.62 seconds
NS_A2_A1_B1_A2_B1_A1_B1_A1_B1_A1, status: Status.UNKNOWN, split count: 10, time: 3.62
Output dim: 0, lower bound: -81.8628677, upper bound: 81.8629717
NS_A2_A1_B1_A2_B1_A1_B1_A1_B1_A2, status: Status.UNKNOWN, split count: 10, time: 3.62
Output dim: 0, lower bound: -81.8626178, upper bound: 81.8626890
NS_A2_A1_B1_A2_B1_A1_B1_A1_B2_A1, status: Status.UNKNOWN, split count: 10, time: 3.62
Output dim: 0, lower bound: -81.8628677, upper bound: 81.8629716
NS_A2_A1_B1_A2_B1_A1_B1_A1_B2_A2, status: Status.UNKNOWN, split count: 10, time: 3.62
Output dim: 0, lower bound: -81.8626178, upper bound: 81.8626891
NS_A2_A1_B1_A2_B1_A1_B1_A2_B1_B1, status: Status.UNKNOWN, split count: 10, time: 3.62
Output dim: 0, lower bound: -81.8627495, upper bound: 81.8627730
NS_A2_A1_B1_A2_B1_A1_B1_A2_B1_B2, status: Status.UNKNOWN, split count: 10, time: 3.62
Output dim: 0, lower bound: -81.8626021, upper bound: 81.8626291
NS_A2_A1_B1_A2_B1_A1_B1_A2_B2_A1, status: Status.UNKNOWN, split count: 10, time: 3.62
Output dim: 0, lower bound: -81.8627703, upper bound: 81.8627925
NS_A2_A1_B1_A2_B1_A1_B1_A2_B2_A2, status: Status.UNKNOWN, split count: 10, time: 3.62
Output dim: 0, lower bound: -81.8626021, upper bound: 81.8626291
NS_A2_A1_B1_A2_B1_A1_B2_B1_B1_A1, status: Status.UNKNOWN, split count: 10, time: 3.62
Output dim: 0, lower bound: -81.8604385, upper bound: 81.8610384
NS_A2_A1_B1_A2_B1_A1_B2_B1_B1_A2, status: Status.UNKNOWN, split count: 10, time: 3.62
Output dim: 0, lower bound: -81.8604385, upper bound: 81.8610384
NS_A2_A1_B1_A2_B1_A1_B2_B1_B2_A1, status: Status.UNKNOWN, split count: 10, time: 3.62
Output dim: 0, lower bound: -81.8605636, upper bound: 81.8611060
NS_A2_A1_B1_A2_B1_A1_B2_B1_B2_A2, status: Status.UNKNOWN, split count: 10, time: 3.62
Output dim: 0, lower bound: -81.8605636, upper bound: 81.8611060
NS_A2_A1_B1_A2_B1_A2_A1_B1_B1_A1, status: Status.VERIFIED, split count: 10, time: 3.62
Output dim: 0, lower bound: -81.8604179, upper bound: 81.8608600
NS_A2_A1_B1_A2_B1_A2_A1_B1_B1_A2, status: Status.UNKNOWN, split count: 10, time: 3.62
Output dim: 0, lower bound: -81.8605647, upper bound: 81.8611203
NS_A2_A1_B1_A2_B1_A2_A1_B1_B2_B1, status: Status.UNKNOWN, split count: 10, time: 3.62
Output dim: 0, lower bound: -81.8604397, upper bound: 81.8610556
NS_A2_A1_B1_A2_B1_A2_A1_B1_B2_B2, status: Status.UNKNOWN, split count: 10, time: 3.62
Output dim: 0, lower bound: -81.8605647, upper bound: 81.8611203
NS_A2_A1_B1_A2_B1_A2_A1_B2_B1_A1, status: Status.UNKNOWN, split count: 10, time: 3.62
Output dim: 0, lower bound: -81.8604279, upper bound: 81.8610697
NS_A2_A1_B1_A2_B1_A2_A1_B2_B1_A2, status: Status.UNKNOWN, split count: 10, time: 3.62
Output dim: 0, lower bound: -81.8604279, upper bound: 81.8610697
NS_A2_A1_B1_A2_B1_A2_A1_B2_B2_A1, status: Status.UNKNOWN, split count: 10, time: 3.62
Output dim: 0, lower bound: -81.8604279, upper bound: 81.8610697
NS_A2_A1_B1_A2_B1_A2_A1_B2_B2_A2, status: Status.UNKNOWN, split count: 10, time: 3.62
Output dim: 0, lower bound: -81.8604279, upper bound: 81.8610697
NS_A2_A1_B1_A2_B1_A2_A2_B1_B1_A1, status: Status.UNKNOWN, split count: 10, time: 3.62
Output dim: 0, lower bound: -81.8629228, upper bound: 81.8629875
NS_A2_A1_B1_A2_B1_A2_A2_B1_B1_A2, status: Status.UNKNOWN, split count: 10, time: 3.62
Output dim: 0, lower bound: -81.8629228, upper bound: 81.8629875
NS_A2_A1_B1_A2_B1_A2_A2_B2_B1_A1, status: Status.UNKNOWN, split count: 10, time: 3.62
Output dim: 0, lower bound: -81.8603983, upper bound: 81.8610642
NS_A2_A1_B1_A2_B1_A2_A2_B2_B1_A2, status: Status.UNKNOWN, split count: 10, time: 3.62
Output dim: 0, lower bound: -81.8604279, upper bound: 81.8610642
NS_A2_A1_B1_A2_B1_A2_A2_B2_B2_A1, status: Status.UNKNOWN, split count: 10, time: 3.62
Output dim: 0, lower bound: -81.8604279, upper bound: 81.8610642
NS_A2_A1_B1_A2_B1_A2_A2_B2_B2_A2, status: Status.UNKNOWN, split count: 10, time: 3.62
Output dim: 0, lower bound: -81.8604279, upper bound: 81.8610642
NS_A2_A1_B1_A2_B2_A1_B1_B1_A1_A1, status: Status.UNKNOWN, split count: 10, time: 3.62
Output dim: 0, lower bound: -81.8626243, upper bound: 81.8625774
NS_A2_A1_B1_A2_B2_A1_B1_B1_A1_A2, status: Status.UNKNOWN, split count: 10, time: 3.62
Output dim: 0, lower bound: -81.8626243, upper bound: 81.8625774
NS_A2_A1_B1_A2_B2_A1_B1_B1_A2_A1, status: Status.UNKNOWN, split count: 10, time: 3.62
Output dim: 0, lower bound: -81.8625776, upper bound: 81.8626573
NS_A2_A1_B1_A2_B2_A1_B1_B1_A2_A2, status: Status.UNKNOWN, split count: 10, time: 3.62
Output dim: 0, lower bound: -81.8625776, upper bound: 81.8628393
NS_A2_A1_B1_A2_B2_A1_B2_B1_B1_A1, status: Status.UNKNOWN, split count: 10, time: 3.62
Output dim: 0, lower bound: -81.8604805, upper bound: 81.8610756
NS_A2_A1_B1_A2_B2_A1_B2_B1_B1_A2, status: Status.UNKNOWN, split count: 10, time: 3.62
Output dim: 0, lower bound: -81.8604805, upper bound: 81.8610756
NS_A2_A1_B1_A2_B2_A1_B2_B1_B2_A1, status: Status.UNKNOWN, split count: 10, time: 3.62
Output dim: 0, lower bound: -81.8607244, upper bound: 81.8611686
NS_A2_A1_B1_A2_B2_A1_B2_B1_B2_A2, status: Status.UNKNOWN, split count: 10, time: 3.62
Output dim: 0, lower bound: -81.8607244, upper bound: 81.8611686
NS_A2_A1_B1_A2_B2_A1_B2_B2_A1_B1, status: Status.UNKNOWN, split count: 10, time: 3.62
Output dim: 0, lower bound: -81.8604805, upper bound: 81.8611192
NS_A2_A1_B1_A2_B2_A1_B2_B2_A1_B2, status: Status.UNKNOWN, split count: 10, time: 3.62
Output dim: 0, lower bound: -81.8605459, upper bound: 81.8611290
NS_A2_A1_B1_A2_B2_A2_B1_B1_A1_B1, status: Status.UNKNOWN, split count: 10, time: 3.62
Output dim: 0, lower bound: -81.8631372, upper bound: 81.8632948
NS_A2_A1_B1_A2_B2_A2_B1_B1_A1_B2, status: Status.UNKNOWN, split count: 10, time: 3.62
Output dim: 0, lower bound: -81.8631372, upper bound: 81.8632948
NS_A2_A1_B1_A2_B2_A2_B1_B1_A2_B1, status: Status.UNKNOWN, split count: 10, time: 3.62
Output dim: 0, lower bound: -81.8632284, upper bound: 81.8633019
NS_A2_A1_B1_A2_B2_A2_B1_B1_A2_B2, status: Status.UNKNOWN, split count: 10, time: 3.62
Output dim: 0, lower bound: -81.8632284, upper bound: 81.8633019
NS_A2_A1_B1_A2_B2_A2_B1_B2_A1_B1, status: Status.UNKNOWN, split count: 10, time: 3.62
Output dim: 0, lower bound: -81.8626869, upper bound: 81.8626522
NS_A2_A1_B1_A2_B2_A2_B1_B2_A1_B2, status: Status.UNKNOWN, split count: 10, time: 3.62
Output dim: 0, lower bound: -81.8623963, upper bound: 81.8624728
NS_A2_A1_B1_A2_B2_A2_B1_B2_A2_A1, status: Status.UNKNOWN, split count: 10, time: 3.62
Output dim: 0, lower bound: -81.8625846, upper bound: 81.8628301
NS_A2_A1_B1_A2_B2_A2_B1_B2_A2_A2, status: Status.UNKNOWN, split count: 10, time: 3.62
Output dim: 0, lower bound: -81.8623963, upper bound: 81.8625689
NS_A2_A1_B1_A2_B2_A2_B2_A2_B1_A1, status: Status.VERIFIED, split count: 10, time: 3.62
Output dim: 0, lower bound: -81.8602045, upper bound: 81.8602833
NS_A2_A1_B1_A2_B2_A2_B2_A2_B1_A2, status: Status.UNKNOWN, split count: 10, time: 3.62
Output dim: 0, lower bound: -81.8607798, upper bound: 81.8612180
NS_A2_A1_B1_A2_B2_A2_B2_A2_B2_A1, status: Status.VERIFIED, split count: 10, time: 3.62
Output dim: 0, lower bound: -81.8600707, upper bound: 81.8605823
NS_A2_A1_B1_A2_B2_A2_B2_A2_B2_A2, status: Status.UNKNOWN, split count: 10, time: 3.62
Output dim: 0, lower bound: -81.8605873, upper bound: 81.8612151
NS_A2_A1_B2_A2_A2_B1_A2_B2_A1_A1, status: Status.UNKNOWN, split count: 10, time: 3.62
Output dim: 0, lower bound: -81.8629138, upper bound: 81.8631198
NS_A2_A1_B2_A2_A2_B1_A2_B2_A1_A2, status: Status.UNKNOWN, split count: 10, time: 3.62
Output dim: 0, lower bound: -81.8627166, upper bound: 81.8627916
NS_A2_A1_B2_A2_A2_B1_A2_B2_A2_A1, status: Status.UNKNOWN, split count: 10, time: 3.62
Output dim: 0, lower bound: -81.8629138, upper bound: 81.8631260
NS_A2_A1_B2_A2_A2_B1_A2_B2_A2_A2, status: Status.UNKNOWN, split count: 10, time: 3.62
Output dim: 0, lower bound: -81.8627166, upper bound: 81.8627600
NS_A2_A1_B2_A2_A2_B2_A2_B2_A2_A1, status: Status.VERIFIED, split count: 10, time: 3.62
Output dim: 0, lower bound: -81.8603468, upper bound: 81.8608172
NS_A2_A1_B2_A2_A2_B2_A2_B2_A2_A2, status: Status.UNKNOWN, split count: 10, time: 3.62
Output dim: 0, lower bound: -81.8604926, upper bound: 81.8610738

## BFS NS instance: NS_A2_A1_B1_A2_B1_A1_B1_A1_B1_A1

### Backsubstitution after applying NS history:
0: -18.7508411, 68.4890060, -18.4944057, 67.5162964, -86.2671356, 86.9834137
1: -49.8245735, 154.3710022, -49.2275085, 152.2429962, -202.0675354, 203.5985107
2: -74.8435745, 137.3560791, -74.2057419, 135.0674896, -209.9110718, 211.5618286
3: -42.8591805, 164.9938202, -42.3476486, 162.8658600, -205.7250366, 207.3414459
4: -68.3577652, 120.9827576, -67.7113876, 119.0659943, -187.4237518, 188.6941223

Time for backsubstitution: 1.18 seconds

### NS candidates at layer 1
type: B, layer: 1, pos: 41
type: A, layer: 1, pos: 41
type: B, layer: 1, pos: 20
type: B, layer: 1, pos: 39
type: A, layer: 1, pos: 20
type: A, layer: 1, pos: 39
type: A, layer: 1, pos: 19
type: B, layer: 1, pos: 19
type: B, layer: 1, pos: 14
type: B, layer: 1, pos: 25
type: A, layer: 1, pos: 14
type: A, layer: 1, pos: 25
type: A, layer: 1, pos: 23
type: B, layer: 1, pos: 23
type: B, layer: 1, pos: 21
type: A, layer: 1, pos: 6
type: B, layer: 1, pos: 6
type: B, layer: 1, pos: 4
type: A, layer: 1, pos: 21
type: A, layer: 1, pos: 46
type: B, layer: 1, pos: 13
type: A, layer: 1, pos: 29
type: A, layer: 1, pos: 15
type: B, layer: 1, pos: 46
type: A, layer: 1, pos: 4
type: A, layer: 1, pos: 38
type: B, layer: 1, pos: 15
type: B, layer: 1, pos: 16
type: A, layer: 1, pos: 37
type: A, layer: 1, pos: 16
type: B, layer: 1, pos: 29
type: B, layer: 1, pos: 37
type: B, layer: 1, pos: 38
type: B, layer: 1, pos: 12
type: A, layer: 1, pos: 12
type: A, layer: 1, pos: 11
type: B, layer: 1, pos: 11
type: A, layer: 1, pos: 42
type: B, layer: 1, pos: 42
type: A, layer: 1, pos: 31
type: B, layer: 1, pos: 31
type: B, layer: 1, pos: 8

Time for candidate selection: 0.08 seconds

### Candidate
type: B, layer: 1, pos: 41

### Candidate
type: A, layer: 1, pos: 41

### Candidate
type: B, layer: 1, pos: 20

### Candidate
type: B, layer: 1, pos: 39

### Candidate
type: A, layer: 1, pos: 20

### Candidate
type: A, layer: 1, pos: 39

### Candidate
type: A, layer: 1, pos: 19

### Candidate
type: B, layer: 1, pos: 19

### Candidate
type: B, layer: 1, pos: 14

### Candidate
type: B, layer: 1, pos: 25

### Candidate
type: A, layer: 1, pos: 14

### Candidate
type: A, layer: 1, pos: 25

### Candidate
type: A, layer: 1, pos: 23

### Candidate
type: B, layer: 1, pos: 23

### Candidate
type: B, layer: 1, pos: 21

### Candidate
type: A, layer: 1, pos: 6

### Candidate
type: B, layer: 1, pos: 6

### Candidate
type: B, layer: 1, pos: 4

### Candidate
type: A, layer: 1, pos: 21

### Candidate
type: A, layer: 1, pos: 46

### Candidate
type: B, layer: 1, pos: 13

### Candidate
type: A, layer: 1, pos: 29

### Candidate
type: A, layer: 1, pos: 15

### Candidate
type: B, layer: 1, pos: 46

### Candidate
type: A, layer: 1, pos: 4

### Candidate
type: A, layer: 1, pos: 38

### Candidate
type: B, layer: 1, pos: 15

### Candidate
type: B, layer: 1, pos: 16

### Candidate
type: A, layer: 1, pos: 37

### Candidate
type: A, layer: 1, pos: 16

### Candidate
type: B, layer: 1, pos: 29

### Candidate
type: B, layer: 1, pos: 37

### Candidate
type: B, layer: 1, pos: 38

### Candidate
type: B, layer: 1, pos: 12

### Candidate
type: A, layer: 1, pos: 12

### Candidate
type: A, layer: 1, pos: 11

### Candidate
type: B, layer: 1, pos: 11

### Candidate
type: A, layer: 1, pos: 42

### Candidate
type: B, layer: 1, pos: 42

### Candidate
type: A, layer: 1, pos: 31

### Candidate
type: B, layer: 1, pos: 31

### Candidate
type: B, layer: 1, pos: 8

## Relational analysis of NS_A2_A1_B1_A2_B1_A1_B1_A1_B1_A1_B1
Optimization infeasible because this subproblem isn't reachable.

## BFS NS instance: NS_A2_A1_B1_A2_B1_A1_B1_A1_B1_A2

### Backsubstitution after applying NS history:
0: -19.1569290, 69.9650497, -18.0725880, 65.9674606, -85.1243820, 88.0376358
1: -50.8049240, 157.9555054, -48.0959282, 148.7796478, -199.5845642, 206.0514221
2: -76.1308975, 140.7245026, -72.5140762, 132.0465240, -208.1773987, 213.2385406
3: -43.7400742, 168.7090302, -41.3882027, 159.1763916, -202.9164276, 210.0971985
4: -69.5919113, 123.9626083, -66.1706238, 116.4095078, -186.0014038, 190.1332397

Time for backsubstitution: 0.96 seconds

### NS candidates at layer 1
type: B, layer: 1, pos: 41
type: B, layer: 1, pos: 20
type: B, layer: 1, pos: 39
type: A, layer: 1, pos: 20
type: A, layer: 1, pos: 19
type: B, layer: 1, pos: 19
type: A, layer: 1, pos: 41
type: A, layer: 1, pos: 39
type: B, layer: 1, pos: 25
type: B, layer: 1, pos: 14
type: A, layer: 1, pos: 23
type: A, layer: 1, pos: 14
type: B, layer: 1, pos: 21
type: A, layer: 1, pos: 25
type: A, layer: 1, pos: 6
type: B, layer: 1, pos: 6
type: B, layer: 1, pos: 23
type: A, layer: 1, pos: 15
type: B, layer: 1, pos: 4
type: B, layer: 1, pos: 46
type: A, layer: 1, pos: 46
type: B, layer: 1, pos: 8
type: B, layer: 1, pos: 13
type: A, layer: 1, pos: 38
type: A, layer: 1, pos: 16
type: A, layer: 1, pos: 21
type: A, layer: 1, pos: 29
type: B, layer: 1, pos: 12
type: B, layer: 1, pos: 15
type: A, layer: 1, pos: 4
type: B, layer: 1, pos: 38
type: A, layer: 1, pos: 42
type: B, layer: 1, pos: 16
type: A, layer: 1, pos: 12
type: B, layer: 1, pos: 29
type: B, layer: 1, pos: 37
type: B, layer: 1, pos: 11
type: A, layer: 1, pos: 31
type: B, layer: 1, pos: 31
type: A, layer: 1, pos: 11
type: A, layer: 1, pos: 37
type: B, layer: 1, pos: 42

Time for candidate selection: 0.07 seconds

### Candidate
type: B, layer: 1, pos: 41

### Candidate
type: B, layer: 1, pos: 20

### Candidate
type: B, layer: 1, pos: 39

### Candidate
type: A, layer: 1, pos: 20

### Candidate
type: A, layer: 1, pos: 19

### Candidate
type: B, layer: 1, pos: 19

### Candidate
type: A, layer: 1, pos: 41

### Candidate
type: A, layer: 1, pos: 39

### Candidate
type: B, layer: 1, pos: 25

### Candidate
type: B, layer: 1, pos: 14

### Candidate
type: A, layer: 1, pos: 23

### Candidate
type: A, layer: 1, pos: 14

### Candidate
type: B, layer: 1, pos: 21

### Candidate
type: A, layer: 1, pos: 25

### Candidate
type: A, layer: 1, pos: 6

### Candidate
type: B, layer: 1, pos: 6

### Candidate
type: B, layer: 1, pos: 23

### Candidate
type: A, layer: 1, pos: 15

### Candidate
type: B, layer: 1, pos: 4

### Candidate
type: B, layer: 1, pos: 46

### Candidate
type: A, layer: 1, pos: 46

### Candidate
type: B, layer: 1, pos: 8

## Relational analysis of NS_A2_A1_B1_A2_B1_A1_B1_A1_B1_A2_B1
Optimization infeasible because this subproblem isn't reachable.

### Candidate
type: B, layer: 1, pos: 13

### Candidate
type: A, layer: 1, pos: 38

### Candidate
type: A, layer: 1, pos: 16

### Candidate
type: A, layer: 1, pos: 21

### Candidate
type: A, layer: 1, pos: 29

### Candidate
type: B, layer: 1, pos: 12

### Candidate
type: B, layer: 1, pos: 15

### Candidate
type: A, layer: 1, pos: 4

### Candidate
type: B, layer: 1, pos: 38

### Candidate
type: A, layer: 1, pos: 42

### Candidate
type: B, layer: 1, pos: 16

### Candidate
type: A, layer: 1, pos: 12

### Candidate
type: B, layer: 1, pos: 29

### Candidate
type: B, layer: 1, pos: 37

### Candidate
type: B, layer: 1, pos: 11

### Candidate
type: A, layer: 1, pos: 31

### Candidate
type: B, layer: 1, pos: 31

### Candidate
type: A, layer: 1, pos: 11

### Candidate
type: A, layer: 1, pos: 37

### Candidate
type: B, layer: 1, pos: 42

## BFS NS instance: NS_A2_A1_B1_A2_B1_A1_B1_A1_B2_A1

### Backsubstitution after applying NS history:
0: -18.7508411, 68.4890060, -18.6717911, 68.0852661, -86.8361053, 87.1607971
1: -49.8245735, 154.3710022, -49.4309769, 154.1967163, -204.0212708, 203.8019714
2: -74.8435745, 137.3560791, -74.1288147, 136.7682495, -211.6118164, 211.4848938
3: -42.8591805, 164.9938202, -42.5748444, 164.7793274, -207.6385040, 207.5686646
4: -68.3577652, 120.9827576, -67.7418365, 120.5343704, -188.8921204, 188.7245789

Time for backsubstitution: 0.86 seconds

### NS candidates at layer 1
type: B, layer: 1, pos: 41
type: B, layer: 1, pos: 20
type: A, layer: 1, pos: 41
type: B, layer: 1, pos: 39
type: A, layer: 1, pos: 39
type: A, layer: 1, pos: 20
type: A, layer: 1, pos: 19
type: B, layer: 1, pos: 19
type: B, layer: 1, pos: 25
type: A, layer: 1, pos: 14
type: B, layer: 1, pos: 14
type: A, layer: 1, pos: 25
type: B, layer: 1, pos: 23
type: A, layer: 1, pos: 23
type: A, layer: 1, pos: 6
type: B, layer: 1, pos: 6
type: B, layer: 1, pos: 21
type: A, layer: 1, pos: 21
type: B, layer: 1, pos: 15
type: A, layer: 1, pos: 38
type: B, layer: 1, pos: 4
type: A, layer: 1, pos: 46
type: B, layer: 1, pos: 46
type: A, layer: 1, pos: 29
type: B, layer: 1, pos: 13
type: B, layer: 1, pos: 16
type: A, layer: 1, pos: 15
type: A, layer: 1, pos: 4
type: A, layer: 1, pos: 12
type: B, layer: 1, pos: 12
type: A, layer: 1, pos: 16
type: A, layer: 1, pos: 37
type: B, layer: 1, pos: 38
type: A, layer: 1, pos: 42
type: B, layer: 1, pos: 29
type: A, layer: 1, pos: 11
type: B, layer: 1, pos: 11
type: B, layer: 1, pos: 8
type: B, layer: 1, pos: 31
type: A, layer: 1, pos: 31
type: B, layer: 1, pos: 37
type: B, layer: 1, pos: 42

Time for candidate selection: 0.07 seconds

### Candidate
type: B, layer: 1, pos: 41

### Candidate
type: B, layer: 1, pos: 20

### Candidate
type: A, layer: 1, pos: 41

### Candidate
type: B, layer: 1, pos: 39

### Candidate
type: A, layer: 1, pos: 39

### Candidate
type: A, layer: 1, pos: 20

### Candidate
type: A, layer: 1, pos: 19

### Candidate
type: B, layer: 1, pos: 19

### Candidate
type: B, layer: 1, pos: 25

### Candidate
type: A, layer: 1, pos: 14

### Candidate
type: B, layer: 1, pos: 14

### Candidate
type: A, layer: 1, pos: 25

### Candidate
type: B, layer: 1, pos: 23

### Candidate
type: A, layer: 1, pos: 23

### Candidate
type: A, layer: 1, pos: 6

### Candidate
type: B, layer: 1, pos: 6

### Candidate
type: B, layer: 1, pos: 21

### Candidate
type: A, layer: 1, pos: 21

### Candidate
type: B, layer: 1, pos: 15

### Candidate
type: A, layer: 1, pos: 38

### Candidate
type: B, layer: 1, pos: 4

### Candidate
type: A, layer: 1, pos: 46

### Candidate
type: B, layer: 1, pos: 46

### Candidate
type: A, layer: 1, pos: 29

### Candidate
type: B, layer: 1, pos: 13

### Candidate
type: B, layer: 1, pos: 16

### Candidate
type: A, layer: 1, pos: 15

### Candidate
type: A, layer: 1, pos: 4

### Candidate
type: A, layer: 1, pos: 12

### Candidate
type: B, layer: 1, pos: 12

### Candidate
type: A, layer: 1, pos: 16

### Candidate
type: A, layer: 1, pos: 37

### Candidate
type: B, layer: 1, pos: 38

### Candidate
type: A, layer: 1, pos: 42

### Candidate
type: B, layer: 1, pos: 29

### Candidate
type: A, layer: 1, pos: 11

### Candidate
type: B, layer: 1, pos: 11

### Candidate
type: B, layer: 1, pos: 8

## Relational analysis of NS_A2_A1_B1_A2_B1_A1_B1_A1_B2_A1_B1

### Relational analysis result of NS_A2_A1_B1_A2_B1_A1_B1_A1_B2_A1_B1
Status: Status.UNKNOWN
Output dim: 0, lower bound: -81.8626178, upper bound: 81.8626891
time: 0.85 seconds

## Relational analysis of NS_A2_A1_B1_A2_B1_A1_B1_A1_B2_A1_B2

### Relational analysis result of NS_A2_A1_B1_A2_B1_A1_B1_A1_B2_A1_B2
Status: Status.UNKNOWN
Output dim: 0, lower bound: -81.8626178, upper bound: 81.8626890
time: 0.84 seconds

## BFS NS instance: NS_A2_A1_B1_A2_B1_A1_B1_A1_B2_A2

### Backsubstitution after applying NS history:
0: -19.1569290, 69.9650497, -18.3149700, 66.8060226, -85.9629517, 88.2800217
1: -50.8049240, 157.9555054, -48.4981194, 151.2578888, -202.0628052, 206.4535980
2: -76.1308975, 140.7245026, -72.6938629, 134.2282867, -210.3591461, 213.4183655
3: -43.7400742, 168.7090302, -41.7621117, 161.6397858, -205.3798370, 210.4710999
4: -69.5919113, 123.9626083, -66.4339600, 118.3033295, -187.8952332, 190.3965759

Time for backsubstitution: 0.86 seconds

### NS candidates at layer 1
type: B, layer: 1, pos: 41
type: B, layer: 1, pos: 20
type: B, layer: 1, pos: 39
type: A, layer: 1, pos: 20
type: A, layer: 1, pos: 19
type: B, layer: 1, pos: 19
type: A, layer: 1, pos: 39
type: A, layer: 1, pos: 41
type: B, layer: 1, pos: 25
type: B, layer: 1, pos: 14
type: A, layer: 1, pos: 14
type: A, layer: 1, pos: 23
type: A, layer: 1, pos: 25
type: B, layer: 1, pos: 21
type: A, layer: 1, pos: 6
type: B, layer: 1, pos: 23
type: B, layer: 1, pos: 6
type: B, layer: 1, pos: 4
type: A, layer: 1, pos: 38
type: B, layer: 1, pos: 46
type: A, layer: 1, pos: 46
type: A, layer: 1, pos: 21
type: B, layer: 1, pos: 15
type: B, layer: 1, pos: 13
type: A, layer: 1, pos: 15
type: A, layer: 1, pos: 29
type: B, layer: 1, pos: 8
type: A, layer: 1, pos: 16
type: B, layer: 1, pos: 12
type: A, layer: 1, pos: 42
type: A, layer: 1, pos: 4
type: B, layer: 1, pos: 16
type: A, layer: 1, pos: 12
type: B, layer: 1, pos: 38
type: B, layer: 1, pos: 29
type: B, layer: 1, pos: 11
type: B, layer: 1, pos: 37
type: A, layer: 1, pos: 37
type: A, layer: 1, pos: 11
type: A, layer: 1, pos: 31
type: B, layer: 1, pos: 31
type: B, layer: 1, pos: 42

Time for candidate selection: 0.07 seconds

### Candidate
type: B, layer: 1, pos: 41

### Candidate
type: B, layer: 1, pos: 20

### Candidate
type: B, layer: 1, pos: 39

### Candidate
type: A, layer: 1, pos: 20

### Candidate
type: A, layer: 1, pos: 19

### Candidate
type: B, layer: 1, pos: 19

### Candidate
type: A, layer: 1, pos: 39

### Candidate
type: A, layer: 1, pos: 41

### Candidate
type: B, layer: 1, pos: 25

### Candidate
type: B, layer: 1, pos: 14

### Candidate
type: A, layer: 1, pos: 14

### Candidate
type: A, layer: 1, pos: 23

### Candidate
type: A, layer: 1, pos: 25

### Candidate
type: B, layer: 1, pos: 21

### Candidate
type: A, layer: 1, pos: 6

### Candidate
type: B, layer: 1, pos: 23

### Candidate
type: B, layer: 1, pos: 6

### Candidate
type: B, layer: 1, pos: 4

### Candidate
type: A, layer: 1, pos: 38

### Candidate
type: B, layer: 1, pos: 46

### Candidate
type: A, layer: 1, pos: 46

### Candidate
type: A, layer: 1, pos: 21

### Candidate
type: B, layer: 1, pos: 15

### Candidate
type: B, layer: 1, pos: 13

### Candidate
type: A, layer: 1, pos: 15

### Candidate
type: A, layer: 1, pos: 29

### Candidate
type: B, layer: 1, pos: 8

## Relational analysis of NS_A2_A1_B1_A2_B1_A1_B1_A1_B2_A2_B1
Optimization infeasible because this subproblem isn't reachable.

### Candidate
type: A, layer: 1, pos: 16

### Candidate
type: B, layer: 1, pos: 12

### Candidate
type: A, layer: 1, pos: 42

### Candidate
type: A, layer: 1, pos: 4

### Candidate
type: B, layer: 1, pos: 16

### Candidate
type: A, layer: 1, pos: 12

### Candidate
type: B, layer: 1, pos: 38

### Candidate
type: B, layer: 1, pos: 29

### Candidate
type: B, layer: 1, pos: 11

### Candidate
type: B, layer: 1, pos: 37

### Candidate
type: A, layer: 1, pos: 37

### Candidate
type: A, layer: 1, pos: 11

### Candidate
type: A, layer: 1, pos: 31

### Candidate
type: B, layer: 1, pos: 31

### Candidate
type: B, layer: 1, pos: 42

## BFS NS instance: NS_A2_A1_B1_A2_B1_A1_B1_A2_B1_B1

### Backsubstitution after applying NS history:
0: -19.2726421, 70.2619247, -18.1283054, 66.1793213, -85.4519653, 88.3902283
1: -50.9067268, 159.2095032, -48.2829247, 149.1696320, -200.0763397, 207.4924316
2: -76.0648041, 141.5390320, -72.8342361, 132.2471313, -208.3119354, 214.3732605
3: -43.8539200, 169.9546509, -41.5333443, 159.6132355, -203.4671631, 211.4879913
4: -69.5887299, 124.6562119, -66.4418259, 116.6022110, -186.1909027, 191.0979919

Time for backsubstitution: 0.87 seconds

### NS candidates at layer 1
type: A, layer: 1, pos: 41
type: B, layer: 1, pos: 41
type: A, layer: 1, pos: 19
type: B, layer: 1, pos: 19
type: B, layer: 1, pos: 6
type: A, layer: 1, pos: 6
type: A, layer: 1, pos: 25
type: B, layer: 1, pos: 25
type: A, layer: 1, pos: 21
type: B, layer: 1, pos: 21
type: A, layer: 1, pos: 46
type: B, layer: 1, pos: 4
type: B, layer: 1, pos: 46
type: B, layer: 1, pos: 38
type: A, layer: 1, pos: 38
type: B, layer: 1, pos: 13
type: B, layer: 1, pos: 14
type: A, layer: 1, pos: 4
type: A, layer: 1, pos: 12
type: B, layer: 1, pos: 29
type: B, layer: 1, pos: 12
type: A, layer: 1, pos: 14
type: A, layer: 1, pos: 39
type: A, layer: 1, pos: 8
type: A, layer: 1, pos: 29
type: A, layer: 1, pos: 31
type: B, layer: 1, pos: 31
type: B, layer: 1, pos: 11
type: A, layer: 1, pos: 15
type: B, layer: 1, pos: 16
type: A, layer: 1, pos: 20
type: A, layer: 1, pos: 11
type: A, layer: 1, pos: 16
type: B, layer: 1, pos: 15
type: B, layer: 1, pos: 39
type: B, layer: 1, pos: 42
type: B, layer: 1, pos: 23
type: B, layer: 1, pos: 20
type: A, layer: 1, pos: 23
type: B, layer: 1, pos: 37
type: A, layer: 1, pos: 37
type: A, layer: 1, pos: 42

Time for candidate selection: 0.07 seconds

### Candidate
type: A, layer: 1, pos: 41

### Candidate
type: B, layer: 1, pos: 41

### Candidate
type: A, layer: 1, pos: 19

### Candidate
type: B, layer: 1, pos: 19

### Candidate
type: B, layer: 1, pos: 6

### Candidate
type: A, layer: 1, pos: 6

### Candidate
type: A, layer: 1, pos: 25

### Candidate
type: B, layer: 1, pos: 25

### Candidate
type: A, layer: 1, pos: 21

### Candidate
type: B, layer: 1, pos: 21

### Candidate
type: A, layer: 1, pos: 46

### Candidate
type: B, layer: 1, pos: 4

### Candidate
type: B, layer: 1, pos: 46

### Candidate
type: B, layer: 1, pos: 38

### Candidate
type: A, layer: 1, pos: 38

### Candidate
type: B, layer: 1, pos: 13

### Candidate
type: B, layer: 1, pos: 14

### Candidate
type: A, layer: 1, pos: 4

### Candidate
type: A, layer: 1, pos: 12

### Candidate
type: B, layer: 1, pos: 29

### Candidate
type: B, layer: 1, pos: 12

### Candidate
type: A, layer: 1, pos: 14

### Candidate
type: A, layer: 1, pos: 39

### Candidate
type: A, layer: 1, pos: 8

## Relational analysis of NS_A2_A1_B1_A2_B1_A1_B1_A2_B1_B1_A1
Optimization infeasible because this subproblem isn't reachable.

### Candidate
type: A, layer: 1, pos: 29

### Candidate
type: A, layer: 1, pos: 31

### Candidate
type: B, layer: 1, pos: 31

### Candidate
type: B, layer: 1, pos: 11

### Candidate
type: A, layer: 1, pos: 15

### Candidate
type: B, layer: 1, pos: 16

### Candidate
type: A, layer: 1, pos: 20

### Candidate
type: A, layer: 1, pos: 11

### Candidate
type: A, layer: 1, pos: 16

### Candidate
type: B, layer: 1, pos: 15

### Candidate
type: B, layer: 1, pos: 39

### Candidate
type: B, layer: 1, pos: 42

## Relational analysis of NS_A2_A1_B1_A2_B1_A1_B1_A2_B1_B1_B1
Optimization infeasible because this subproblem isn't reachable.

### Candidate
type: B, layer: 1, pos: 23

### Candidate
type: B, layer: 1, pos: 20

### Candidate
type: A, layer: 1, pos: 23

### Candidate
type: B, layer: 1, pos: 37

### Candidate
type: A, layer: 1, pos: 37

### Candidate
type: A, layer: 1, pos: 42

## BFS NS instance: NS_A2_A1_B1_A2_B1_A1_B1_A2_B1_B2

### Backsubstitution after applying NS history:
0: -18.9129696, 68.9487915, -18.5660954, 67.7696533, -86.6826248, 87.5148697
1: -49.9474106, 156.2555389, -49.3537445, 153.0117950, -202.9591980, 205.6092224
2: -74.6084900, 138.9383698, -74.3062286, 135.8249969, -210.4334717, 213.2445984
3: -43.0312271, 166.7962036, -42.4972496, 163.6280670, -206.6592712, 209.2934570
4: -68.2602692, 122.3663864, -67.8385010, 119.7530518, -188.0132904, 190.2048950

Time for backsubstitution: 0.91 seconds

### NS candidates at layer 1
type: A, layer: 1, pos: 41
type: A, layer: 1, pos: 21
type: A, layer: 1, pos: 20
type: A, layer: 1, pos: 25
type: B, layer: 1, pos: 21
type: B, layer: 1, pos: 25
type: B, layer: 1, pos: 20
type: B, layer: 1, pos: 41
type: B, layer: 1, pos: 19
type: A, layer: 1, pos: 19
type: A, layer: 1, pos: 39
type: A, layer: 1, pos: 14
type: B, layer: 1, pos: 14
type: B, layer: 1, pos: 4
type: B, layer: 1, pos: 39
type: B, layer: 1, pos: 16
type: A, layer: 1, pos: 4
type: A, layer: 1, pos: 8
type: A, layer: 1, pos: 11
type: B, layer: 1, pos: 6
type: A, layer: 1, pos: 16
type: B, layer: 1, pos: 11
type: A, layer: 1, pos: 6
type: A, layer: 1, pos: 46
type: B, layer: 1, pos: 15
type: B, layer: 1, pos: 46
type: A, layer: 1, pos: 15
type: B, layer: 1, pos: 13
type: B, layer: 1, pos: 38
type: A, layer: 1, pos: 12
type: B, layer: 1, pos: 29
type: A, layer: 1, pos: 38
type: A, layer: 1, pos: 29
type: B, layer: 1, pos: 12
type: B, layer: 1, pos: 42
type: B, layer: 1, pos: 23
type: A, layer: 1, pos: 31
type: B, layer: 1, pos: 31
type: A, layer: 1, pos: 37
type: B, layer: 1, pos: 37
type: A, layer: 1, pos: 23
type: A, layer: 1, pos: 42

Time for candidate selection: 0.07 seconds

### Candidate
type: A, layer: 1, pos: 41

### Candidate
type: A, layer: 1, pos: 21

### Candidate
type: A, layer: 1, pos: 20

### Candidate
type: A, layer: 1, pos: 25

### Candidate
type: B, layer: 1, pos: 21

### Candidate
type: B, layer: 1, pos: 25

### Candidate
type: B, layer: 1, pos: 20

### Candidate
type: B, layer: 1, pos: 41

### Candidate
type: B, layer: 1, pos: 19

### Candidate
type: A, layer: 1, pos: 19

### Candidate
type: A, layer: 1, pos: 39

### Candidate
type: A, layer: 1, pos: 14

### Candidate
type: B, layer: 1, pos: 14

### Candidate
type: B, layer: 1, pos: 4

### Candidate
type: B, layer: 1, pos: 39

### Candidate
type: B, layer: 1, pos: 16

### Candidate
type: A, layer: 1, pos: 4

### Candidate
type: A, layer: 1, pos: 8

## Relational analysis of NS_A2_A1_B1_A2_B1_A1_B1_A2_B1_B2_A1

### Relational analysis result of NS_A2_A1_B1_A2_B1_A1_B1_A2_B1_B2_A1
Status: Status.UNKNOWN
Output dim: 0, lower bound: -81.8626580, upper bound: 81.8626387
time: 1.31 seconds

## Relational analysis of NS_A2_A1_B1_A2_B1_A1_B1_A2_B1_B2_A2

### Relational analysis result of NS_A2_A1_B1_A2_B1_A1_B1_A2_B1_B2_A2
Status: Status.UNKNOWN
Output dim: 0, lower bound: -81.8626579, upper bound: 81.8626386
time: 0.81 seconds

## BFS NS instance: NS_A2_A1_B1_A2_B1_A1_B1_A2_B2_A1

### Backsubstitution after applying NS history:
0: -18.8804264, 68.8203964, -18.6717911, 68.0852661, -86.9656906, 87.4921875
1: -49.8772011, 155.9365845, -49.4309769, 154.1967163, -204.0739136, 205.3675537
2: -74.5861359, 138.5931549, -74.1288147, 136.7682495, -211.3543549, 212.7219696
3: -42.9775162, 166.4864044, -42.5748444, 164.7793274, -207.7568359, 209.0612488
4: -68.2231598, 122.0781555, -67.7418365, 120.5343704, -188.7575378, 189.8199768

Time for backsubstitution: 0.90 seconds

### NS candidates at layer 1
type: B, layer: 1, pos: 41
type: A, layer: 1, pos: 41
type: A, layer: 1, pos: 19
type: B, layer: 1, pos: 19
type: A, layer: 1, pos: 6
type: B, layer: 1, pos: 6
type: B, layer: 1, pos: 25
type: B, layer: 1, pos: 21
type: A, layer: 1, pos: 25
type: A, layer: 1, pos: 21
type: B, layer: 1, pos: 46
type: A, layer: 1, pos: 46
type: B, layer: 1, pos: 4
type: A, layer: 1, pos: 38
type: B, layer: 1, pos: 38
type: A, layer: 1, pos: 29
type: B, layer: 1, pos: 13
type: A, layer: 1, pos: 12
type: B, layer: 1, pos: 14
type: A, layer: 1, pos: 4
type: B, layer: 1, pos: 12
type: A, layer: 1, pos: 14
type: B, layer: 1, pos: 39
type: B, layer: 1, pos: 29
type: B, layer: 1, pos: 20
type: A, layer: 1, pos: 31
type: B, layer: 1, pos: 8
type: B, layer: 1, pos: 11
type: B, layer: 1, pos: 31
type: B, layer: 1, pos: 16
type: A, layer: 1, pos: 11
type: B, layer: 1, pos: 15
type: A, layer: 1, pos: 15
type: A, layer: 1, pos: 16
type: A, layer: 1, pos: 39
type: A, layer: 1, pos: 23
type: B, layer: 1, pos: 23
type: A, layer: 1, pos: 20
type: A, layer: 1, pos: 42
type: A, layer: 1, pos: 37
type: B, layer: 1, pos: 37
type: B, layer: 1, pos: 42

Time for candidate selection: 0.07 seconds

### Candidate
type: B, layer: 1, pos: 41

### Candidate
type: A, layer: 1, pos: 41

### Candidate
type: A, layer: 1, pos: 19

### Candidate
type: B, layer: 1, pos: 19

### Candidate
type: A, layer: 1, pos: 6

### Candidate
type: B, layer: 1, pos: 6

### Candidate
type: B, layer: 1, pos: 25

### Candidate
type: B, layer: 1, pos: 21

### Candidate
type: A, layer: 1, pos: 25

### Candidate
type: A, layer: 1, pos: 21

### Candidate
type: B, layer: 1, pos: 46

### Candidate
type: A, layer: 1, pos: 46

### Candidate
type: B, layer: 1, pos: 4

### Candidate
type: A, layer: 1, pos: 38

### Candidate
type: B, layer: 1, pos: 38

### Candidate
type: A, layer: 1, pos: 29

### Candidate
type: B, layer: 1, pos: 13

### Candidate
type: A, layer: 1, pos: 12

### Candidate
type: B, layer: 1, pos: 14

### Candidate
type: A, layer: 1, pos: 4

### Candidate
type: B, layer: 1, pos: 12

### Candidate
type: A, layer: 1, pos: 14

### Candidate
type: B, layer: 1, pos: 39

### Candidate
type: B, layer: 1, pos: 29

### Candidate
type: B, layer: 1, pos: 20

### Candidate
type: A, layer: 1, pos: 31

### Candidate
type: B, layer: 1, pos: 8

## Relational analysis of NS_A2_A1_B1_A2_B1_A1_B1_A2_B2_A1_B1

### Relational analysis result of NS_A2_A1_B1_A2_B1_A1_B1_A2_B2_A1_B1
Status: Status.UNKNOWN
Output dim: 0, lower bound: -81.8626021, upper bound: 81.8626291
time: 1.25 seconds

## Relational analysis of NS_A2_A1_B1_A2_B1_A1_B1_A2_B2_A1_B2

### Relational analysis result of NS_A2_A1_B1_A2_B1_A1_B1_A2_B2_A1_B2
Status: Status.UNKNOWN
Output dim: 0, lower bound: -81.8626021, upper bound: 81.8626291
time: 1.01 seconds

## BFS NS instance: NS_A2_A1_B1_A2_B1_A1_B1_A2_B2_A2

### Backsubstitution after applying NS history:
0: -19.2631512, 70.2086716, -18.3149700, 66.8060226, -86.0691681, 88.5236435
1: -50.8365173, 159.3043976, -48.4981194, 151.2578888, -202.0943909, 207.8025055
2: -75.8894196, 141.5539398, -72.6938629, 134.2282867, -210.1176758, 214.2478027
3: -43.8118935, 170.0169830, -41.7621117, 161.6397858, -205.4516754, 211.7790680
4: -69.4336929, 124.7251511, -66.4339600, 118.3033295, -187.7370300, 191.1591034

Time for backsubstitution: 0.90 seconds

### NS candidates at layer 1
type: B, layer: 1, pos: 41
type: A, layer: 1, pos: 19
type: B, layer: 1, pos: 13
type: B, layer: 1, pos: 19
type: B, layer: 1, pos: 25
type: A, layer: 1, pos: 41
type: A, layer: 1, pos: 25
type: A, layer: 1, pos: 16
type: B, layer: 1, pos: 21
type: B, layer: 1, pos: 16
type: B, layer: 1, pos: 8
type: B, layer: 1, pos: 46
type: B, layer: 1, pos: 4
type: A, layer: 1, pos: 46
type: B, layer: 1, pos: 12
type: A, layer: 1, pos: 38
type: A, layer: 1, pos: 29
type: A, layer: 1, pos: 6
type: B, layer: 1, pos: 38
type: B, layer: 1, pos: 29
type: B, layer: 1, pos: 39
type: A, layer: 1, pos: 4
type: A, layer: 1, pos: 12
type: B, layer: 1, pos: 6
type: B, layer: 1, pos: 14
type: A, layer: 1, pos: 21
type: B, layer: 1, pos: 20
type: B, layer: 1, pos: 37
type: A, layer: 1, pos: 14
type: A, layer: 1, pos: 23
type: A, layer: 1, pos: 37
type: A, layer: 1, pos: 42
type: A, layer: 1, pos: 31
type: B, layer: 1, pos: 31
type: A, layer: 1, pos: 15
type: B, layer: 1, pos: 23
type: A, layer: 1, pos: 20
type: B, layer: 1, pos: 11
type: A, layer: 1, pos: 11
type: B, layer: 1, pos: 42
type: B, layer: 1, pos: 15
type: A, layer: 1, pos: 39

Time for candidate selection: 0.07 seconds

### Candidate
type: B, layer: 1, pos: 41

### Candidate
type: A, layer: 1, pos: 19

### Candidate
type: B, layer: 1, pos: 13

### Candidate
type: B, layer: 1, pos: 19

### Candidate
type: B, layer: 1, pos: 25

### Candidate
type: A, layer: 1, pos: 41

### Candidate
type: A, layer: 1, pos: 25

### Candidate
type: A, layer: 1, pos: 16

### Candidate
type: B, layer: 1, pos: 21

### Candidate
type: B, layer: 1, pos: 16

### Candidate
type: B, layer: 1, pos: 8

## Relational analysis of NS_A2_A1_B1_A2_B1_A1_B1_A2_B2_A2_B1

### Relational analysis result of NS_A2_A1_B1_A2_B1_A1_B1_A2_B2_A2_B1
Status: Status.UNKNOWN
Output dim: 0, lower bound: -81.8626021, upper bound: 81.8626291
time: 0.88 seconds

## Relational analysis of NS_A2_A1_B1_A2_B1_A1_B1_A2_B2_A2_B2

### Relational analysis result of NS_A2_A1_B1_A2_B1_A1_B1_A2_B2_A2_B2
Status: Status.UNKNOWN
Output dim: 0, lower bound: -81.8626021, upper bound: 81.8626291
time: 0.90 seconds

## BFS NS instance: NS_A2_A1_B1_A2_B1_A1_B2_B1_B1_A1

### Backsubstitution after applying NS history:
0: -18.9055080, 69.0488586, -19.0926151, 69.8261414, -88.7316513, 88.1414719
1: -50.2107277, 155.6897736, -50.8617516, 157.4954071, -207.7061310, 206.5515289
2: -75.3631592, 138.5468445, -76.5586243, 139.6645355, -215.0276794, 215.1054688
3: -43.1919365, 166.3853455, -43.7430496, 168.6731262, -211.8650360, 210.1283875
4: -68.8351059, 122.0457687, -69.8557816, 123.1292496, -191.9643555, 191.9015503

Time for backsubstitution: 0.88 seconds

### NS candidates at layer 1
type: A, layer: 1, pos: 41
type: A, layer: 1, pos: 14
type: B, layer: 1, pos: 14
type: B, layer: 1, pos: 19
type: A, layer: 1, pos: 19
type: A, layer: 1, pos: 39
type: B, layer: 1, pos: 39
type: B, layer: 1, pos: 12
type: A, layer: 1, pos: 25
type: B, layer: 1, pos: 25
type: A, layer: 1, pos: 20
type: B, layer: 1, pos: 37
type: A, layer: 1, pos: 12
type: B, layer: 1, pos: 41
type: B, layer: 1, pos: 29
type: A, layer: 1, pos: 8
type: A, layer: 1, pos: 37
type: B, layer: 1, pos: 8
type: B, layer: 1, pos: 20
type: A, layer: 1, pos: 29
type: A, layer: 1, pos: 11
type: A, layer: 1, pos: 21
type: A, layer: 1, pos: 46
type: A, layer: 1, pos: 6
type: B, layer: 1, pos: 6
type: B, layer: 1, pos: 4
type: A, layer: 1, pos: 15
type: B, layer: 1, pos: 21
type: A, layer: 1, pos: 4
type: B, layer: 1, pos: 13
type: B, layer: 1, pos: 46
type: A, layer: 1, pos: 38
type: B, layer: 1, pos: 15
type: A, layer: 1, pos: 16
type: A, layer: 1, pos: 23
type: B, layer: 1, pos: 23
type: B, layer: 1, pos: 38
type: A, layer: 1, pos: 31
type: A, layer: 1, pos: 42
type: B, layer: 1, pos: 31
type: B, layer: 1, pos: 16
type: B, layer: 1, pos: 11

Time for candidate selection: 0.07 seconds

### Candidate
type: A, layer: 1, pos: 41

### Candidate
type: A, layer: 1, pos: 14

### Candidate
type: B, layer: 1, pos: 14

### Candidate
type: B, layer: 1, pos: 19

### Candidate
type: A, layer: 1, pos: 19

### Candidate
type: A, layer: 1, pos: 39

### Candidate
type: B, layer: 1, pos: 39

### Candidate
type: B, layer: 1, pos: 12

### Candidate
type: A, layer: 1, pos: 25

### Candidate
type: B, layer: 1, pos: 25

### Candidate
type: A, layer: 1, pos: 20

### Candidate
type: B, layer: 1, pos: 37

### Candidate
type: A, layer: 1, pos: 12

### Candidate
type: B, layer: 1, pos: 41

### Candidate
type: B, layer: 1, pos: 29

### Candidate
type: A, layer: 1, pos: 8

### Candidate
type: A, layer: 1, pos: 37

### Candidate
type: B, layer: 1, pos: 8

### Candidate
type: B, layer: 1, pos: 20

### Candidate
type: A, layer: 1, pos: 29

### Candidate
type: A, layer: 1, pos: 11

### Candidate
type: A, layer: 1, pos: 21

### Candidate
type: A, layer: 1, pos: 46

### Candidate
type: A, layer: 1, pos: 6

### Candidate
type: B, layer: 1, pos: 6

### Candidate
type: B, layer: 1, pos: 4

### Candidate
type: A, layer: 1, pos: 15

### Candidate
type: B, layer: 1, pos: 21

### Candidate
type: A, layer: 1, pos: 4

### Candidate
type: B, layer: 1, pos: 13

### Candidate
type: B, layer: 1, pos: 46

### Candidate
type: A, layer: 1, pos: 38

### Candidate
type: B, layer: 1, pos: 15

### Candidate
type: A, layer: 1, pos: 16

### Candidate
type: A, layer: 1, pos: 23

### Candidate
type: B, layer: 1, pos: 23

### Candidate
type: B, layer: 1, pos: 38

### Candidate
type: A, layer: 1, pos: 31

## Relational analysis of NS_A2_A1_B1_A2_B1_A1_B2_B1_B1_A1_A1
Optimization infeasible because this subproblem isn't reachable.

### Candidate
type: A, layer: 1, pos: 42

## Relational analysis of NS_A2_A1_B1_A2_B1_A1_B2_B1_B1_A1_A1

### Relational analysis result of NS_A2_A1_B1_A2_B1_A1_B2_B1_B1_A1_A1
Status: Status.VERIFIED
Output dim: 0, lower bound: -81.8602981, upper bound: 81.8606100
time: 0.78 seconds

## Relational analysis of NS_A2_A1_B1_A2_B1_A1_B2_B1_B1_A1_A2

### Relational analysis result of NS_A2_A1_B1_A2_B1_A1_B2_B1_B1_A1_A2
Status: Status.UNKNOWN
Output dim: 0, lower bound: -81.8602981, upper bound: 81.8610384
time: 0.77 seconds

## BFS NS instance: NS_A2_A1_B1_A2_B1_A1_B2_B1_B1_A2

### Backsubstitution after applying NS history:
0: -19.1695518, 69.8992538, -19.0926151, 69.8261414, -88.9956894, 88.9918671
1: -50.6409225, 158.3228149, -50.8617516, 157.4954071, -208.1363220, 209.1845703
2: -75.6577301, 140.8316345, -76.5586243, 139.6645355, -215.3222656, 217.3902435
3: -43.6225853, 169.0185852, -43.7430496, 168.6731262, -212.2957153, 212.7616272
4: -69.2205124, 124.0199509, -69.8557816, 123.1292496, -192.3497620, 193.8757324

Time for backsubstitution: 0.88 seconds

### NS candidates at layer 1
type: A, layer: 1, pos: 41
type: A, layer: 1, pos: 14
type: B, layer: 1, pos: 14
type: B, layer: 1, pos: 19
type: A, layer: 1, pos: 39
type: A, layer: 1, pos: 19
type: B, layer: 1, pos: 39
type: B, layer: 1, pos: 12
type: A, layer: 1, pos: 20
type: A, layer: 1, pos: 25
type: B, layer: 1, pos: 25
type: B, layer: 1, pos: 37
type: B, layer: 1, pos: 29
type: A, layer: 1, pos: 12
type: B, layer: 1, pos: 41
type: A, layer: 1, pos: 8
type: B, layer: 1, pos: 8
type: A, layer: 1, pos: 37
type: B, layer: 1, pos: 20
type: A, layer: 1, pos: 29
type: A, layer: 1, pos: 11
type: A, layer: 1, pos: 21
type: A, layer: 1, pos: 46
type: B, layer: 1, pos: 6
type: A, layer: 1, pos: 6
type: A, layer: 1, pos: 15
type: B, layer: 1, pos: 4
type: B, layer: 1, pos: 21
type: B, layer: 1, pos: 13
type: A, layer: 1, pos: 4
type: B, layer: 1, pos: 46
type: A, layer: 1, pos: 16
type: A, layer: 1, pos: 38
type: B, layer: 1, pos: 38
type: B, layer: 1, pos: 15
type: A, layer: 1, pos: 23
type: B, layer: 1, pos: 23
type: A, layer: 1, pos: 31
type: B, layer: 1, pos: 31
type: A, layer: 1, pos: 42
type: B, layer: 1, pos: 16
type: B, layer: 1, pos: 11

Time for candidate selection: 0.07 seconds

### Candidate
type: A, layer: 1, pos: 41

### Candidate
type: A, layer: 1, pos: 14

### Candidate
type: B, layer: 1, pos: 14

### Candidate
type: B, layer: 1, pos: 19

### Candidate
type: A, layer: 1, pos: 39

### Candidate
type: A, layer: 1, pos: 19

### Candidate
type: B, layer: 1, pos: 39

### Candidate
type: B, layer: 1, pos: 12

### Candidate
type: A, layer: 1, pos: 20

### Candidate
type: A, layer: 1, pos: 25

### Candidate
type: B, layer: 1, pos: 25

### Candidate
type: B, layer: 1, pos: 37

### Candidate
type: B, layer: 1, pos: 29

### Candidate
type: A, layer: 1, pos: 12

### Candidate
type: B, layer: 1, pos: 41

### Candidate
type: A, layer: 1, pos: 8

### Candidate
type: B, layer: 1, pos: 8

### Candidate
type: A, layer: 1, pos: 37

### Candidate
type: B, layer: 1, pos: 20

### Candidate
type: A, layer: 1, pos: 29

### Candidate
type: A, layer: 1, pos: 11

### Candidate
type: A, layer: 1, pos: 21

### Candidate
type: A, layer: 1, pos: 46

### Candidate
type: B, layer: 1, pos: 6

### Candidate
type: A, layer: 1, pos: 6

### Candidate
type: A, layer: 1, pos: 15

### Candidate
type: B, layer: 1, pos: 4

### Candidate
type: B, layer: 1, pos: 21

### Candidate
type: B, layer: 1, pos: 13

### Candidate
type: A, layer: 1, pos: 4

### Candidate
type: B, layer: 1, pos: 46

### Candidate
type: A, layer: 1, pos: 16

### Candidate
type: A, layer: 1, pos: 38

### Candidate
type: B, layer: 1, pos: 38

### Candidate
type: B, layer: 1, pos: 15

### Candidate
type: A, layer: 1, pos: 23

### Candidate
type: B, layer: 1, pos: 23

### Candidate
type: A, layer: 1, pos: 31

## Relational analysis of NS_A2_A1_B1_A2_B1_A1_B2_B1_B1_A2_A1
Optimization infeasible because this subproblem isn't reachable.

### Candidate
type: B, layer: 1, pos: 31

### Candidate
type: A, layer: 1, pos: 42

## Relational analysis of NS_A2_A1_B1_A2_B1_A1_B2_B1_B1_A2_A1

### Relational analysis result of NS_A2_A1_B1_A2_B1_A1_B2_B1_B1_A2_A1
Status: Status.VERIFIED
Output dim: 0, lower bound: -81.8602981, upper bound: 81.8608015
time: 0.81 seconds

## Relational analysis of NS_A2_A1_B1_A2_B1_A1_B2_B1_B1_A2_A2

### Relational analysis result of NS_A2_A1_B1_A2_B1_A1_B2_B1_B1_A2_A2
Status: Status.UNKNOWN
Output dim: 0, lower bound: -81.8602981, upper bound: 81.8610384
time: 1.30 seconds

## BFS NS instance: NS_A2_A1_B1_A2_B1_A1_B2_B1_B2_A1

### Backsubstitution after applying NS history:
0: -19.2080135, 70.1533127, -18.5805836, 67.7363052, -86.9443207, 88.7338943
1: -51.0122948, 158.1988220, -49.5240135, 152.6590118, -203.6713104, 207.7228394
2: -76.5601349, 140.7829285, -74.6881332, 135.7837677, -212.3438568, 215.4710236
3: -43.8815269, 169.0470276, -42.6170883, 163.5825653, -207.4640961, 211.6640930
4: -69.9408417, 124.0000458, -68.1534424, 119.7229919, -189.6638031, 192.1534882

Time for backsubstitution: 0.89 seconds

### NS candidates at layer 1
type: A, layer: 1, pos: 41
type: A, layer: 1, pos: 14
type: B, layer: 1, pos: 14
type: B, layer: 1, pos: 19
type: A, layer: 1, pos: 39
type: A, layer: 1, pos: 19
type: B, layer: 1, pos: 39
type: A, layer: 1, pos: 25
type: B, layer: 1, pos: 12
type: B, layer: 1, pos: 25
type: B, layer: 1, pos: 37
type: A, layer: 1, pos: 12
type: B, layer: 1, pos: 29
type: A, layer: 1, pos: 20
type: A, layer: 1, pos: 8
type: A, layer: 1, pos: 37
type: B, layer: 1, pos: 20
type: B, layer: 1, pos: 41
type: B, layer: 1, pos: 8
type: A, layer: 1, pos: 29
type: A, layer: 1, pos: 11
type: A, layer: 1, pos: 46
type: A, layer: 1, pos: 21
type: A, layer: 1, pos: 6
type: B, layer: 1, pos: 6
type: B, layer: 1, pos: 4
type: B, layer: 1, pos: 21
type: A, layer: 1, pos: 15
type: A, layer: 1, pos: 4
type: B, layer: 1, pos: 15
type: B, layer: 1, pos: 13
type: B, layer: 1, pos: 46
type: A, layer: 1, pos: 38
type: A, layer: 1, pos: 23
type: A, layer: 1, pos: 16
type: B, layer: 1, pos: 38
type: B, layer: 1, pos: 23
type: B, layer: 1, pos: 16
type: A, layer: 1, pos: 42
type: A, layer: 1, pos: 31
type: B, layer: 1, pos: 31
type: B, layer: 1, pos: 11

Time for candidate selection: 0.07 seconds

### Candidate
type: A, layer: 1, pos: 41

### Candidate
type: A, layer: 1, pos: 14

### Candidate
type: B, layer: 1, pos: 14

### Candidate
type: B, layer: 1, pos: 19

### Candidate
type: A, layer: 1, pos: 39

### Candidate
type: A, layer: 1, pos: 19

### Candidate
type: B, layer: 1, pos: 39

### Candidate
type: A, layer: 1, pos: 25

### Candidate
type: B, layer: 1, pos: 12

### Candidate
type: B, layer: 1, pos: 25

### Candidate
type: B, layer: 1, pos: 37

### Candidate
type: A, layer: 1, pos: 12

### Candidate
type: B, layer: 1, pos: 29

### Candidate
type: A, layer: 1, pos: 20

### Candidate
type: A, layer: 1, pos: 8

### Candidate
type: A, layer: 1, pos: 37

### Candidate
type: B, layer: 1, pos: 20

### Candidate
type: B, layer: 1, pos: 41

### Candidate
type: B, layer: 1, pos: 8

### Candidate
type: A, layer: 1, pos: 29

### Candidate
type: A, layer: 1, pos: 11

### Candidate
type: A, layer: 1, pos: 46

### Candidate
type: A, layer: 1, pos: 21

### Candidate
type: A, layer: 1, pos: 6

### Candidate
type: B, layer: 1, pos: 6

### Candidate
type: B, layer: 1, pos: 4

### Candidate
type: B, layer: 1, pos: 21

### Candidate
type: A, layer: 1, pos: 15

### Candidate
type: A, layer: 1, pos: 4

### Candidate
type: B, layer: 1, pos: 15

### Candidate
type: B, layer: 1, pos: 13

### Candidate
type: B, layer: 1, pos: 46

### Candidate
type: A, layer: 1, pos: 38

### Candidate
type: A, layer: 1, pos: 23

### Candidate
type: A, layer: 1, pos: 16

## Relational analysis of NS_A2_A1_B1_A2_B1_A1_B2_B1_B2_A1_A1
Optimization infeasible because this subproblem isn't reachable.

### Candidate
type: B, layer: 1, pos: 38

### Candidate
type: B, layer: 1, pos: 23

### Candidate
type: B, layer: 1, pos: 16

### Candidate
type: A, layer: 1, pos: 42

## Relational analysis of NS_A2_A1_B1_A2_B1_A1_B2_B1_B2_A1_A1

### Relational analysis result of NS_A2_A1_B1_A2_B1_A1_B2_B1_B2_A1_A1
Status: Status.VERIFIED
Output dim: 0, lower bound: -81.8602981, upper bound: 81.8606638
time: 0.93 seconds

## Relational analysis of NS_A2_A1_B1_A2_B1_A1_B2_B1_B2_A1_A2

### Relational analysis result of NS_A2_A1_B1_A2_B1_A1_B2_B1_B2_A1_A2
Status: Status.UNKNOWN
Output dim: 0, lower bound: -81.8602981, upper bound: 81.8611060
time: 0.76 seconds

## BFS NS instance: NS_A2_A1_B1_A2_B1_A1_B2_B1_B2_A2

### Backsubstitution after applying NS history:
0: -19.3761635, 70.6416855, -18.5805836, 67.7363052, -87.1124725, 89.2222672
1: -51.1780510, 160.0721283, -49.5240135, 152.6590118, -203.8370667, 209.5961304
2: -76.4557343, 142.3144379, -74.6881332, 135.7837677, -212.2394562, 217.0025482
3: -44.0854263, 170.8686676, -42.6170883, 163.5825653, -207.6679993, 213.4857330
4: -69.9495621, 125.3343735, -68.1534424, 119.7229919, -189.6725464, 193.4877930

Time for backsubstitution: 0.90 seconds

### NS candidates at layer 1
type: A, layer: 1, pos: 41
type: A, layer: 1, pos: 14
type: B, layer: 1, pos: 14
type: B, layer: 1, pos: 19
type: A, layer: 1, pos: 39
type: A, layer: 1, pos: 19
type: B, layer: 1, pos: 39
type: B, layer: 1, pos: 12
type: A, layer: 1, pos: 25
type: B, layer: 1, pos: 25
type: A, layer: 1, pos: 20
type: B, layer: 1, pos: 37
type: B, layer: 1, pos: 29
type: A, layer: 1, pos: 12
type: A, layer: 1, pos: 8
type: A, layer: 1, pos: 37
type: B, layer: 1, pos: 20
type: B, layer: 1, pos: 41
type: B, layer: 1, pos: 8
type: A, layer: 1, pos: 29
type: A, layer: 1, pos: 11
type: A, layer: 1, pos: 46
type: B, layer: 1, pos: 6
type: A, layer: 1, pos: 6
type: A, layer: 1, pos: 21
type: B, layer: 1, pos: 21
type: B, layer: 1, pos: 4
type: A, layer: 1, pos: 15
type: B, layer: 1, pos: 13
type: A, layer: 1, pos: 4
type: B, layer: 1, pos: 46
type: B, layer: 1, pos: 38
type: B, layer: 1, pos: 15
type: A, layer: 1, pos: 16
type: A, layer: 1, pos: 38
type: A, layer: 1, pos: 23
type: B, layer: 1, pos: 23
type: B, layer: 1, pos: 16
type: A, layer: 1, pos: 31
type: B, layer: 1, pos: 31
type: A, layer: 1, pos: 42
type: B, layer: 1, pos: 11

Time for candidate selection: 0.07 seconds

### Candidate
type: A, layer: 1, pos: 41

### Candidate
type: A, layer: 1, pos: 14

### Candidate
type: B, layer: 1, pos: 14

### Candidate
type: B, layer: 1, pos: 19

### Candidate
type: A, layer: 1, pos: 39

### Candidate
type: A, layer: 1, pos: 19

### Candidate
type: B, layer: 1, pos: 39

### Candidate
type: B, layer: 1, pos: 12

### Candidate
type: A, layer: 1, pos: 25

### Candidate
type: B, layer: 1, pos: 25

### Candidate
type: A, layer: 1, pos: 20

### Candidate
type: B, layer: 1, pos: 37

### Candidate
type: B, layer: 1, pos: 29

### Candidate
type: A, layer: 1, pos: 12

### Candidate
type: A, layer: 1, pos: 8

### Candidate
type: A, layer: 1, pos: 37

### Candidate
type: B, layer: 1, pos: 20

### Candidate
type: B, layer: 1, pos: 41

### Candidate
type: B, layer: 1, pos: 8

### Candidate
type: A, layer: 1, pos: 29

### Candidate
type: A, layer: 1, pos: 11

### Candidate
type: A, layer: 1, pos: 46

### Candidate
type: B, layer: 1, pos: 6

### Candidate
type: A, layer: 1, pos: 6

### Candidate
type: A, layer: 1, pos: 21

### Candidate
type: B, layer: 1, pos: 21

### Candidate
type: B, layer: 1, pos: 4

### Candidate
type: A, layer: 1, pos: 15

### Candidate
type: B, layer: 1, pos: 13

### Candidate
type: A, layer: 1, pos: 4

### Candidate
type: B, layer: 1, pos: 46

### Candidate
type: B, layer: 1, pos: 38

### Candidate
type: B, layer: 1, pos: 15

### Candidate
type: A, layer: 1, pos: 16

## Relational analysis of NS_A2_A1_B1_A2_B1_A1_B2_B1_B2_A2_A1
Optimization infeasible because this subproblem isn't reachable.

### Candidate
type: A, layer: 1, pos: 38

### Candidate
type: A, layer: 1, pos: 23

### Candidate
type: B, layer: 1, pos: 23

### Candidate
type: B, layer: 1, pos: 16

### Candidate
type: A, layer: 1, pos: 31

## Relational analysis of NS_A2_A1_B1_A2_B1_A1_B2_B1_B2_A2_A1
Optimization infeasible because this subproblem isn't reachable.

### Candidate
type: B, layer: 1, pos: 31

### Candidate
type: A, layer: 1, pos: 42

## Relational analysis of NS_A2_A1_B1_A2_B1_A1_B2_B1_B2_A2_A1

### Relational analysis result of NS_A2_A1_B1_A2_B1_A1_B2_B1_B2_A2_A1
Status: Status.VERIFIED
Output dim: 0, lower bound: -81.8602981, upper bound: 81.8608595
time: 0.94 seconds

## Relational analysis of NS_A2_A1_B1_A2_B1_A1_B2_B1_B2_A2_A2

### Relational analysis result of NS_A2_A1_B1_A2_B1_A1_B2_B1_B2_A2_A2
Status: Status.UNKNOWN
Output dim: 0, lower bound: -81.8602981, upper bound: 81.8611060
time: 0.79 seconds

## BFS NS instance: NS_A2_A1_B1_A2_B1_A2_A1_B1_B1_A2

### Backsubstitution after applying NS history:
0: -21.5581799, 78.7153015, -18.3910942, 67.1361313, -88.6943130, 97.1063995
1: -56.8885002, 179.3242798, -48.9588051, 151.3595581, -208.2480621, 228.2830811
2: -84.9954147, 158.6314087, -73.7940826, 134.3392334, -219.3346252, 232.4254456
3: -48.9080620, 191.2201691, -42.1147232, 161.9170685, -210.8250885, 233.3348541
4: -77.7107773, 139.6879883, -67.3392639, 118.4117813, -196.1225586, 207.0272522

Time for backsubstitution: 0.99 seconds

### NS candidates at layer 1
type: A, layer: 1, pos: 41
type: B, layer: 1, pos: 19
type: B, layer: 1, pos: 41
type: B, layer: 1, pos: 21
type: A, layer: 1, pos: 6
type: B, layer: 1, pos: 6
type: A, layer: 1, pos: 46
type: A, layer: 1, pos: 25
type: A, layer: 1, pos: 19
type: B, layer: 1, pos: 25
type: A, layer: 1, pos: 11
type: A, layer: 1, pos: 4
type: A, layer: 1, pos: 38
type: B, layer: 1, pos: 14
type: A, layer: 1, pos: 12
type: B, layer: 1, pos: 38
type: A, layer: 1, pos: 21
type: B, layer: 1, pos: 13
type: B, layer: 1, pos: 12
type: A, layer: 1, pos: 29
type: B, layer: 1, pos: 4
type: B, layer: 1, pos: 29
type: B, layer: 1, pos: 20
type: B, layer: 1, pos: 46
type: A, layer: 1, pos: 15
type: B, layer: 1, pos: 39
type: A, layer: 1, pos: 31
type: B, layer: 1, pos: 31
type: A, layer: 1, pos: 14
type: B, layer: 1, pos: 23
type: B, layer: 1, pos: 16
type: A, layer: 1, pos: 39
type: B, layer: 1, pos: 37
type: A, layer: 1, pos: 8
type: B, layer: 1, pos: 15
type: A, layer: 1, pos: 20
type: B, layer: 1, pos: 8
type: B, layer: 1, pos: 42
type: A, layer: 1, pos: 22
type: A, layer: 1, pos: 23
type: A, layer: 1, pos: 37
type: B, layer: 1, pos: 11

Time for candidate selection: 0.09 seconds

### Candidate
type: A, layer: 1, pos: 41

### Candidate
type: B, layer: 1, pos: 19

### Candidate
type: B, layer: 1, pos: 41

### Candidate
type: B, layer: 1, pos: 21

### Candidate
type: A, layer: 1, pos: 6

### Candidate
type: B, layer: 1, pos: 6

### Candidate
type: A, layer: 1, pos: 46

### Candidate
type: A, layer: 1, pos: 25

### Candidate
type: A, layer: 1, pos: 19

### Candidate
type: B, layer: 1, pos: 25

### Candidate
type: A, layer: 1, pos: 11

### Candidate
type: A, layer: 1, pos: 4

### Candidate
type: A, layer: 1, pos: 38

### Candidate
type: B, layer: 1, pos: 14

### Candidate
type: A, layer: 1, pos: 12

### Candidate
type: B, layer: 1, pos: 38

### Candidate
type: A, layer: 1, pos: 21

### Candidate
type: B, layer: 1, pos: 13

### Candidate
type: B, layer: 1, pos: 12

### Candidate
type: A, layer: 1, pos: 29

### Candidate
type: B, layer: 1, pos: 4

### Candidate
type: B, layer: 1, pos: 29

### Candidate
type: B, layer: 1, pos: 20

### Candidate
type: B, layer: 1, pos: 46

### Candidate
type: A, layer: 1, pos: 15

### Candidate
type: B, layer: 1, pos: 39

### Candidate
type: A, layer: 1, pos: 31

### Candidate
type: B, layer: 1, pos: 31

### Candidate
type: A, layer: 1, pos: 14

### Candidate
type: B, layer: 1, pos: 23

### Candidate
type: B, layer: 1, pos: 16

### Candidate
type: A, layer: 1, pos: 39

### Candidate
type: B, layer: 1, pos: 37

### Candidate
type: A, layer: 1, pos: 8

### Candidate
type: B, layer: 1, pos: 15

### Candidate
type: A, layer: 1, pos: 20

### Candidate
type: B, layer: 1, pos: 8

### Candidate
type: B, layer: 1, pos: 42

## Relational analysis of NS_A2_A1_B1_A2_B1_A2_A1_B1_B1_A2_B1
Optimization infeasible because this subproblem isn't reachable.

### Candidate
type: A, layer: 1, pos: 22

## Relational analysis of NS_A2_A1_B1_A2_B1_A2_A1_B1_B1_A2_A1

### Relational analysis result of NS_A2_A1_B1_A2_B1_A2_A1_B1_B1_A2_A1
Status: Status.UNKNOWN
Output dim: 0, lower bound: -81.8630666, upper bound: 81.8631093
time: 0.89 seconds

## Relational analysis of NS_A2_A1_B1_A2_B1_A2_A1_B1_B1_A2_A2

### Relational analysis result of NS_A2_A1_B1_A2_B1_A2_A1_B1_B1_A2_A2
Status: Status.UNKNOWN
Output dim: 0, lower bound: -81.8630666, upper bound: 81.8631093
time: 0.91 seconds

## BFS NS instance: NS_A2_A1_B1_A2_B1_A2_A1_B1_B2_B1

### Backsubstitution after applying NS history:
0: -21.3427429, 77.9242706, -18.9016666, 69.1292267, -90.4719696, 96.8259277
1: -56.3315735, 177.5464630, -50.3671761, 155.8801880, -212.2117310, 227.9136200
2: -84.1739655, 156.9538116, -75.8190689, 138.2523651, -222.4263306, 232.7728882
3: -48.4298248, 189.3373413, -43.3155518, 166.9483185, -215.3781433, 232.6528625
4: -76.9488678, 138.2232819, -69.1784210, 121.8822937, -198.8311615, 207.4017029

Time for backsubstitution: 1.07 seconds

### NS candidates at layer 1
type: A, layer: 1, pos: 41
type: B, layer: 1, pos: 19
type: A, layer: 1, pos: 14
type: B, layer: 1, pos: 14
type: B, layer: 1, pos: 39
type: A, layer: 1, pos: 39
type: B, layer: 1, pos: 12
type: A, layer: 1, pos: 25
type: B, layer: 1, pos: 25
type: A, layer: 1, pos: 19
type: B, layer: 1, pos: 37
type: A, layer: 1, pos: 20
type: B, layer: 1, pos: 29
type: A, layer: 1, pos: 12
type: A, layer: 1, pos: 8
type: A, layer: 1, pos: 11
type: B, layer: 1, pos: 8
type: B, layer: 1, pos: 41
type: A, layer: 1, pos: 37
type: B, layer: 1, pos: 20
type: A, layer: 1, pos: 46
type: A, layer: 1, pos: 29
type: A, layer: 1, pos: 6
type: B, layer: 1, pos: 21
type: B, layer: 1, pos: 6
type: A, layer: 1, pos: 15
type: A, layer: 1, pos: 4
type: A, layer: 1, pos: 21
type: A, layer: 1, pos: 38
type: B, layer: 1, pos: 4
type: B, layer: 1, pos: 13
type: B, layer: 1, pos: 15
type: B, layer: 1, pos: 23
type: B, layer: 1, pos: 38
type: A, layer: 1, pos: 23
type: B, layer: 1, pos: 46
type: A, layer: 1, pos: 42
type: A, layer: 1, pos: 31
type: B, layer: 1, pos: 31
type: A, layer: 1, pos: 22
type: B, layer: 1, pos: 16
type: B, layer: 1, pos: 11

Time for candidate selection: 0.09 seconds

### Candidate
type: A, layer: 1, pos: 41

### Candidate
type: B, layer: 1, pos: 19

### Candidate
type: A, layer: 1, pos: 14

### Candidate
type: B, layer: 1, pos: 14

### Candidate
type: B, layer: 1, pos: 39

### Candidate
type: A, layer: 1, pos: 39

### Candidate
type: B, layer: 1, pos: 12

### Candidate
type: A, layer: 1, pos: 25

### Candidate
type: B, layer: 1, pos: 25

### Candidate
type: A, layer: 1, pos: 19

### Candidate
type: B, layer: 1, pos: 37

### Candidate
type: A, layer: 1, pos: 20

### Candidate
type: B, layer: 1, pos: 29

### Candidate
type: A, layer: 1, pos: 12

### Candidate
type: A, layer: 1, pos: 8

### Candidate
type: A, layer: 1, pos: 11

### Candidate
type: B, layer: 1, pos: 8

### Candidate
type: B, layer: 1, pos: 41

### Candidate
type: A, layer: 1, pos: 37

### Candidate
type: B, layer: 1, pos: 20

### Candidate
type: A, layer: 1, pos: 46

### Candidate
type: A, layer: 1, pos: 29

### Candidate
type: A, layer: 1, pos: 6

### Candidate
type: B, layer: 1, pos: 21

### Candidate
type: B, layer: 1, pos: 6

### Candidate
type: A, layer: 1, pos: 15

### Candidate
type: A, layer: 1, pos: 4

### Candidate
type: A, layer: 1, pos: 21

### Candidate
type: A, layer: 1, pos: 38

### Candidate
type: B, layer: 1, pos: 4

### Candidate
type: B, layer: 1, pos: 13

### Candidate
type: B, layer: 1, pos: 15

### Candidate
type: B, layer: 1, pos: 23

### Candidate
type: B, layer: 1, pos: 38

### Candidate
type: A, layer: 1, pos: 23

### Candidate
type: B, layer: 1, pos: 46

### Candidate
type: A, layer: 1, pos: 42

## Relational analysis of NS_A2_A1_B1_A2_B1_A2_A1_B1_B2_B1_A1

### Relational analysis result of NS_A2_A1_B1_A2_B1_A2_A1_B1_B2_B1_A1
Status: Status.VERIFIED
Output dim: 0, lower bound: -81.8602949, upper bound: 81.8607957
time: 0.83 seconds

## Relational analysis of NS_A2_A1_B1_A2_B1_A2_A1_B1_B2_B1_A2

### Relational analysis result of NS_A2_A1_B1_A2_B1_A2_A1_B1_B2_B1_A2
Status: Status.UNKNOWN
Output dim: 0, lower bound: -81.8602949, upper bound: 81.8610556
time: 1.04 seconds

## BFS NS instance: NS_A2_A1_B1_A2_B1_A2_A1_B1_B2_B2

### Backsubstitution after applying NS history:
0: -21.6661549, 79.1037064, -18.3934708, 67.0515137, -88.7176666, 97.4971695
1: -57.1756744, 180.2160187, -49.0389175, 151.0746155, -208.2502899, 229.2549438
2: -85.4210587, 159.3976746, -73.9607849, 134.4019775, -219.8230286, 233.3584442
3: -49.1567116, 192.1778412, -42.1971512, 161.8887634, -211.0454712, 234.3750000
4: -78.1009827, 140.3668823, -67.4861298, 118.5025024, -196.6034698, 207.8529663

Time for backsubstitution: 1.15 seconds

### NS candidates at layer 1
type: A, layer: 1, pos: 41
type: B, layer: 1, pos: 19
type: A, layer: 1, pos: 14
type: B, layer: 1, pos: 14
type: B, layer: 1, pos: 39
type: A, layer: 1, pos: 39
type: A, layer: 1, pos: 25
type: B, layer: 1, pos: 12
type: B, layer: 1, pos: 25
type: B, layer: 1, pos: 37
type: B, layer: 1, pos: 29
type: A, layer: 1, pos: 19
type: A, layer: 1, pos: 12
type: A, layer: 1, pos: 8
type: A, layer: 1, pos: 20
type: A, layer: 1, pos: 11
type: B, layer: 1, pos: 20
type: A, layer: 1, pos: 37
type: B, layer: 1, pos: 8
type: A, layer: 1, pos: 46
type: B, layer: 1, pos: 41
type: A, layer: 1, pos: 29
type: B, layer: 1, pos: 21
type: A, layer: 1, pos: 6
type: B, layer: 1, pos: 6
type: A, layer: 1, pos: 15
type: A, layer: 1, pos: 4
type: A, layer: 1, pos: 21
type: B, layer: 1, pos: 4
type: A, layer: 1, pos: 38
type: B, layer: 1, pos: 13
type: B, layer: 1, pos: 15
type: A, layer: 1, pos: 42
type: B, layer: 1, pos: 23
type: A, layer: 1, pos: 23
type: B, layer: 1, pos: 38
type: A, layer: 1, pos: 22
type: B, layer: 1, pos: 46
type: A, layer: 1, pos: 31
type: B, layer: 1, pos: 31
type: B, layer: 1, pos: 16
type: B, layer: 1, pos: 11

Time for candidate selection: 0.08 seconds

### Candidate
type: A, layer: 1, pos: 41

### Candidate
type: B, layer: 1, pos: 19

### Candidate
type: A, layer: 1, pos: 14

### Candidate
type: B, layer: 1, pos: 14

### Candidate
type: B, layer: 1, pos: 39

### Candidate
type: A, layer: 1, pos: 39

### Candidate
type: A, layer: 1, pos: 25

### Candidate
type: B, layer: 1, pos: 12

### Candidate
type: B, layer: 1, pos: 25

### Candidate
type: B, layer: 1, pos: 37

### Candidate
type: B, layer: 1, pos: 29

### Candidate
type: A, layer: 1, pos: 19

### Candidate
type: A, layer: 1, pos: 12

### Candidate
type: A, layer: 1, pos: 8

### Candidate
type: A, layer: 1, pos: 20

### Candidate
type: A, layer: 1, pos: 11

### Candidate
type: B, layer: 1, pos: 20

### Candidate
type: A, layer: 1, pos: 37

### Candidate
type: B, layer: 1, pos: 8

### Candidate
type: A, layer: 1, pos: 46

### Candidate
type: B, layer: 1, pos: 41

### Candidate
type: A, layer: 1, pos: 29

### Candidate
type: B, layer: 1, pos: 21

### Candidate
type: A, layer: 1, pos: 6

### Candidate
type: B, layer: 1, pos: 6

### Candidate
type: A, layer: 1, pos: 15

### Candidate
type: A, layer: 1, pos: 4

### Candidate
type: A, layer: 1, pos: 21

### Candidate
type: B, layer: 1, pos: 4

### Candidate
type: A, layer: 1, pos: 38

### Candidate
type: B, layer: 1, pos: 13

### Candidate
type: B, layer: 1, pos: 15

### Candidate
type: A, layer: 1, pos: 42

## Relational analysis of NS_A2_A1_B1_A2_B1_A2_A1_B1_B2_B2_A1

### Relational analysis result of NS_A2_A1_B1_A2_B1_A2_A1_B1_B2_B2_A1
Status: Status.VERIFIED
Output dim: 0, lower bound: -81.8604179, upper bound: 81.8608600
time: 0.81 seconds

## Relational analysis of NS_A2_A1_B1_A2_B1_A2_A1_B1_B2_B2_A2

### Relational analysis result of NS_A2_A1_B1_A2_B1_A2_A1_B1_B2_B2_A2
Status: Status.UNKNOWN
Output dim: 0, lower bound: -81.8604179, upper bound: 81.8611203
time: 0.82 seconds

## BFS NS instance: NS_A2_A1_B1_A2_B1_A2_A1_B2_B1_A1

### Backsubstitution after applying NS history:
0: -21.3804379, 78.0639191, -18.5815697, 67.7649078, -89.1453476, 96.6454926
1: -56.4302750, 177.7992554, -49.2043877, 153.4045715, -209.8348389, 227.0036469
2: -84.3612900, 157.2890778, -73.7736893, 136.1362610, -220.4975128, 231.0627747
3: -48.5161324, 189.6266174, -42.3743095, 163.9343567, -212.4504852, 232.0009155
4: -77.1220779, 138.5061798, -67.4200287, 119.9641190, -197.0861969, 205.9261475

Time for backsubstitution: 1.05 seconds

### NS candidates at layer 1
type: A, layer: 1, pos: 41
type: B, layer: 1, pos: 20
type: B, layer: 1, pos: 19
type: B, layer: 1, pos: 41
type: B, layer: 1, pos: 39
type: A, layer: 1, pos: 20
type: A, layer: 1, pos: 39
type: B, layer: 1, pos: 14
type: A, layer: 1, pos: 19
type: A, layer: 1, pos: 25
type: B, layer: 1, pos: 25
type: B, layer: 1, pos: 23
type: B, layer: 1, pos: 21
type: A, layer: 1, pos: 14
type: A, layer: 1, pos: 6
type: A, layer: 1, pos: 23
type: B, layer: 1, pos: 6
type: A, layer: 1, pos: 46
type: A, layer: 1, pos: 11
type: A, layer: 1, pos: 38
type: A, layer: 1, pos: 4
type: A, layer: 1, pos: 15
type: B, layer: 1, pos: 13
type: B, layer: 1, pos: 15
type: A, layer: 1, pos: 29
type: A, layer: 1, pos: 21
type: A, layer: 1, pos: 12
type: B, layer: 1, pos: 12
type: A, layer: 1, pos: 42
type: B, layer: 1, pos: 4
type: B, layer: 1, pos: 29
type: B, layer: 1, pos: 37
type: B, layer: 1, pos: 38
type: B, layer: 1, pos: 46
type: A, layer: 1, pos: 8
type: A, layer: 1, pos: 37
type: A, layer: 1, pos: 31
type: B, layer: 1, pos: 31
type: B, layer: 1, pos: 16
type: B, layer: 1, pos: 8
type: B, layer: 1, pos: 42
type: B, layer: 1, pos: 11

Time for candidate selection: 0.08 seconds

### Candidate
type: A, layer: 1, pos: 41

### Candidate
type: B, layer: 1, pos: 20

### Candidate
type: B, layer: 1, pos: 19

### Candidate
type: B, layer: 1, pos: 41

### Candidate
type: B, layer: 1, pos: 39

### Candidate
type: A, layer: 1, pos: 20

### Candidate
type: A, layer: 1, pos: 39

### Candidate
type: B, layer: 1, pos: 14

### Candidate
type: A, layer: 1, pos: 19

### Candidate
type: A, layer: 1, pos: 25

### Candidate
type: B, layer: 1, pos: 25

### Candidate
type: B, layer: 1, pos: 23

### Candidate
type: B, layer: 1, pos: 21

### Candidate
type: A, layer: 1, pos: 14

### Candidate
type: A, layer: 1, pos: 6

### Candidate
type: A, layer: 1, pos: 23

### Candidate
type: B, layer: 1, pos: 6

### Candidate
type: A, layer: 1, pos: 46

### Candidate
type: A, layer: 1, pos: 11

### Candidate
type: A, layer: 1, pos: 38

### Candidate
type: A, layer: 1, pos: 4

### Candidate
type: A, layer: 1, pos: 15

### Candidate
type: B, layer: 1, pos: 13

### Candidate
type: B, layer: 1, pos: 15

### Candidate
type: A, layer: 1, pos: 29

### Candidate
type: A, layer: 1, pos: 21

### Candidate
type: A, layer: 1, pos: 12

### Candidate
type: B, layer: 1, pos: 12

### Candidate
type: A, layer: 1, pos: 42

### Candidate
type: B, layer: 1, pos: 4

### Candidate
type: B, layer: 1, pos: 29

### Candidate
type: B, layer: 1, pos: 37

### Candidate
type: B, layer: 1, pos: 38

### Candidate
type: B, layer: 1, pos: 46

### Candidate
type: A, layer: 1, pos: 8

### Candidate
type: A, layer: 1, pos: 37

### Candidate
type: A, layer: 1, pos: 31

### Candidate
type: B, layer: 1, pos: 31

### Candidate
type: B, layer: 1, pos: 16

### Candidate
type: B, layer: 1, pos: 8

### Candidate
type: B, layer: 1, pos: 42

### Candidate
type: B, layer: 1, pos: 11

## BFS NS instance: NS_A2_A1_B1_A2_B1_A2_A1_B2_B1_A2

### Backsubstitution after applying NS history:
0: -21.2478695, 77.4019089, -18.5815697, 67.7649078, -89.0127792, 95.9834747
1: -55.7632408, 177.2924652, -49.2043877, 153.4045715, -209.1678162, 226.4968567
2: -83.0239944, 156.3711090, -73.7736893, 136.1362610, -219.1602173, 230.1448059
3: -48.0032959, 188.9327240, -42.3743095, 163.9343567, -211.9376526, 231.3070374
4: -75.9488525, 137.7809143, -67.4200287, 119.9641190, -195.9129639, 205.2009277

Time for backsubstitution: 0.99 seconds

### NS candidates at layer 1
type: B, layer: 1, pos: 41
type: B, layer: 1, pos: 19
type: A, layer: 1, pos: 41
type: B, layer: 1, pos: 21
type: A, layer: 1, pos: 6
type: B, layer: 1, pos: 6
type: A, layer: 1, pos: 46
type: A, layer: 1, pos: 25
type: A, layer: 1, pos: 11
type: B, layer: 1, pos: 25
type: A, layer: 1, pos: 4
type: A, layer: 1, pos: 38
type: B, layer: 1, pos: 14
type: A, layer: 1, pos: 19
type: B, layer: 1, pos: 38
type: A, layer: 1, pos: 12
type: B, layer: 1, pos: 13
type: B, layer: 1, pos: 39
type: B, layer: 1, pos: 12
type: B, layer: 1, pos: 4
type: B, layer: 1, pos: 29
type: A, layer: 1, pos: 29
type: A, layer: 1, pos: 21
type: A, layer: 1, pos: 15
type: B, layer: 1, pos: 46
type: A, layer: 1, pos: 31
type: B, layer: 1, pos: 31
type: B, layer: 1, pos: 20
type: A, layer: 1, pos: 14
type: B, layer: 1, pos: 37
type: B, layer: 1, pos: 23
type: A, layer: 1, pos: 8
type: A, layer: 1, pos: 20
type: A, layer: 1, pos: 42
type: B, layer: 1, pos: 16
type: B, layer: 1, pos: 15
type: A, layer: 1, pos: 39
type: B, layer: 1, pos: 8
type: A, layer: 1, pos: 23
type: B, layer: 1, pos: 42
type: A, layer: 1, pos: 37
type: B, layer: 1, pos: 11

Time for candidate selection: 0.08 seconds

### Candidate
type: B, layer: 1, pos: 41

### Candidate
type: B, layer: 1, pos: 19

### Candidate
type: A, layer: 1, pos: 41

### Candidate
type: B, layer: 1, pos: 21

### Candidate
type: A, layer: 1, pos: 6

### Candidate
type: B, layer: 1, pos: 6

### Candidate
type: A, layer: 1, pos: 46

### Candidate
type: A, layer: 1, pos: 25

### Candidate
type: A, layer: 1, pos: 11

### Candidate
type: B, layer: 1, pos: 25

### Candidate
type: A, layer: 1, pos: 4

### Candidate
type: A, layer: 1, pos: 38

### Candidate
type: B, layer: 1, pos: 14

### Candidate
type: A, layer: 1, pos: 19

### Candidate
type: B, layer: 1, pos: 38

### Candidate
type: A, layer: 1, pos: 12

### Candidate
type: B, layer: 1, pos: 13

### Candidate
type: B, layer: 1, pos: 39

### Candidate
type: B, layer: 1, pos: 12

### Candidate
type: B, layer: 1, pos: 4

### Candidate
type: B, layer: 1, pos: 29

### Candidate
type: A, layer: 1, pos: 29

### Candidate
type: A, layer: 1, pos: 21

### Candidate
type: A, layer: 1, pos: 15

### Candidate
type: B, layer: 1, pos: 46

### Candidate
type: A, layer: 1, pos: 31

### Candidate
type: B, layer: 1, pos: 31

### Candidate
type: B, layer: 1, pos: 20

### Candidate
type: A, layer: 1, pos: 14

### Candidate
type: B, layer: 1, pos: 37

### Candidate
type: B, layer: 1, pos: 23

### Candidate
type: A, layer: 1, pos: 8

### Candidate
type: A, layer: 1, pos: 20

### Candidate
type: A, layer: 1, pos: 42

### Candidate
type: B, layer: 1, pos: 16

### Candidate
type: B, layer: 1, pos: 15

### Candidate
type: A, layer: 1, pos: 39

### Candidate
type: B, layer: 1, pos: 8

### Candidate
type: A, layer: 1, pos: 23

### Candidate
type: B, layer: 1, pos: 42

### Candidate
type: A, layer: 1, pos: 37

### Candidate
type: B, layer: 1, pos: 11

## BFS NS instance: NS_A2_A1_B1_A2_B1_A2_A1_B2_B2_A1

### Backsubstitution after applying NS history:
0: -21.3804379, 78.0639191, -18.5464497, 67.6095352, -88.9899750, 96.6103592
1: -56.4302750, 177.7992554, -49.2204208, 152.8471222, -209.2773590, 227.0196838
2: -84.3612900, 157.2890778, -73.8128738, 135.9040527, -220.2653351, 231.1019440
3: -48.5161324, 189.6266174, -42.3677864, 163.5677032, -212.0838318, 231.9943695
4: -77.1220779, 138.5061798, -67.4343491, 119.8176804, -196.9397278, 205.9405060

Time for backsubstitution: 0.93 seconds

### NS candidates at layer 1
type: A, layer: 1, pos: 41
type: B, layer: 1, pos: 19
type: B, layer: 1, pos: 14
type: A, layer: 1, pos: 14
type: B, layer: 1, pos: 39
type: A, layer: 1, pos: 39
type: A, layer: 1, pos: 25
type: B, layer: 1, pos: 12
type: B, layer: 1, pos: 25
type: B, layer: 1, pos: 37
type: A, layer: 1, pos: 19
type: A, layer: 1, pos: 12
type: B, layer: 1, pos: 20
type: A, layer: 1, pos: 8
type: B, layer: 1, pos: 29
type: A, layer: 1, pos: 11
type: A, layer: 1, pos: 20
type: A, layer: 1, pos: 37
type: B, layer: 1, pos: 8
type: B, layer: 1, pos: 41
type: A, layer: 1, pos: 46
type: A, layer: 1, pos: 29
type: B, layer: 1, pos: 21
type: A, layer: 1, pos: 6
type: B, layer: 1, pos: 6
type: A, layer: 1, pos: 4
type: A, layer: 1, pos: 15
type: A, layer: 1, pos: 38
type: A, layer: 1, pos: 21
type: B, layer: 1, pos: 4
type: B, layer: 1, pos: 15
type: B, layer: 1, pos: 13
type: B, layer: 1, pos: 42
type: B, layer: 1, pos: 23
type: A, layer: 1, pos: 42
type: A, layer: 1, pos: 23
type: B, layer: 1, pos: 38
type: B, layer: 1, pos: 46
type: A, layer: 1, pos: 31
type: B, layer: 1, pos: 31
type: B, layer: 1, pos: 16
type: B, layer: 1, pos: 11

Time for candidate selection: 0.07 seconds

### Candidate
type: A, layer: 1, pos: 41

### Candidate
type: B, layer: 1, pos: 19

### Candidate
type: B, layer: 1, pos: 14

### Candidate
type: A, layer: 1, pos: 14

### Candidate
type: B, layer: 1, pos: 39

### Candidate
type: A, layer: 1, pos: 39

### Candidate
type: A, layer: 1, pos: 25

### Candidate
type: B, layer: 1, pos: 12

### Candidate
type: B, layer: 1, pos: 25

### Candidate
type: B, layer: 1, pos: 37

### Candidate
type: A, layer: 1, pos: 19

### Candidate
type: A, layer: 1, pos: 12

### Candidate
type: B, layer: 1, pos: 20

### Candidate
type: A, layer: 1, pos: 8

### Candidate
type: B, layer: 1, pos: 29

### Candidate
type: A, layer: 1, pos: 11

### Candidate
type: A, layer: 1, pos: 20

### Candidate
type: A, layer: 1, pos: 37

### Candidate
type: B, layer: 1, pos: 8

### Candidate
type: B, layer: 1, pos: 41

### Candidate
type: A, layer: 1, pos: 46

### Candidate
type: A, layer: 1, pos: 29

### Candidate
type: B, layer: 1, pos: 21

### Candidate
type: A, layer: 1, pos: 6

### Candidate
type: B, layer: 1, pos: 6

### Candidate
type: A, layer: 1, pos: 4

### Candidate
type: A, layer: 1, pos: 15

### Candidate
type: A, layer: 1, pos: 38

### Candidate
type: A, layer: 1, pos: 21

### Candidate
type: B, layer: 1, pos: 4

### Candidate
type: B, layer: 1, pos: 15

### Candidate
type: B, layer: 1, pos: 13

### Candidate
type: B, layer: 1, pos: 42

### Candidate
type: B, layer: 1, pos: 23

### Candidate
type: A, layer: 1, pos: 42

### Candidate
type: A, layer: 1, pos: 23

### Candidate
type: B, layer: 1, pos: 38

### Candidate
type: B, layer: 1, pos: 46

### Candidate
type: A, layer: 1, pos: 31

### Candidate
type: B, layer: 1, pos: 31

### Candidate
type: B, layer: 1, pos: 16

## Relational analysis of NS_A2_A1_B1_A2_B1_A2_A1_B2_B2_A1_B1
Optimization infeasible because this subproblem isn't reachable.

### Candidate
type: B, layer: 1, pos: 11

## BFS NS instance: NS_A2_A1_B1_A2_B1_A2_A1_B2_B2_A2

### Backsubstitution after applying NS history:
0: -21.2478695, 77.4019089, -18.5464497, 67.6095352, -88.8574066, 95.9483414
1: -55.7632408, 177.2924652, -49.2204208, 152.8471222, -208.6103516, 226.5128784
2: -83.0239944, 156.3711090, -73.8128738, 135.9040527, -218.9280396, 230.1839905
3: -48.0032959, 188.9327240, -42.3677864, 163.5677032, -211.5709991, 231.3005066
4: -75.9488525, 137.7809143, -67.4343491, 119.8176804, -195.7665405, 205.2152710

Time for backsubstitution: 0.95 seconds

### NS candidates at layer 1
type: A, layer: 1, pos: 41
type: B, layer: 1, pos: 19
type: B, layer: 1, pos: 14
type: A, layer: 1, pos: 14
type: B, layer: 1, pos: 39
type: A, layer: 1, pos: 39
type: A, layer: 1, pos: 25
type: B, layer: 1, pos: 12
type: B, layer: 1, pos: 37
type: B, layer: 1, pos: 25
type: A, layer: 1, pos: 20
type: A, layer: 1, pos: 12
type: B, layer: 1, pos: 29
type: A, layer: 1, pos: 19
type: B, layer: 1, pos: 20
type: A, layer: 1, pos: 8
type: A, layer: 1, pos: 11
type: B, layer: 1, pos: 41
type: B, layer: 1, pos: 8
type: A, layer: 1, pos: 37
type: A, layer: 1, pos: 46
type: A, layer: 1, pos: 29
type: B, layer: 1, pos: 21
type: A, layer: 1, pos: 6
type: B, layer: 1, pos: 6
type: A, layer: 1, pos: 15
type: A, layer: 1, pos: 4
type: A, layer: 1, pos: 21
type: B, layer: 1, pos: 4
type: B, layer: 1, pos: 42
type: A, layer: 1, pos: 38
type: B, layer: 1, pos: 13
type: B, layer: 1, pos: 15
type: B, layer: 1, pos: 38
type: B, layer: 1, pos: 23
type: A, layer: 1, pos: 42
type: B, layer: 1, pos: 46
type: A, layer: 1, pos: 31
type: B, layer: 1, pos: 31
type: A, layer: 1, pos: 23
type: B, layer: 1, pos: 16
type: B, layer: 1, pos: 11

Time for candidate selection: 0.07 seconds

### Candidate
type: A, layer: 1, pos: 41

### Candidate
type: B, layer: 1, pos: 19

### Candidate
type: B, layer: 1, pos: 14

### Candidate
type: A, layer: 1, pos: 14

### Candidate
type: B, layer: 1, pos: 39

### Candidate
type: A, layer: 1, pos: 39

### Candidate
type: A, layer: 1, pos: 25

### Candidate
type: B, layer: 1, pos: 12

### Candidate
type: B, layer: 1, pos: 37

### Candidate
type: B, layer: 1, pos: 25

### Candidate
type: A, layer: 1, pos: 20

### Candidate
type: A, layer: 1, pos: 12

### Candidate
type: B, layer: 1, pos: 29

### Candidate
type: A, layer: 1, pos: 19

### Candidate
type: B, layer: 1, pos: 20

### Candidate
type: A, layer: 1, pos: 8

### Candidate
type: A, layer: 1, pos: 11

### Candidate
type: B, layer: 1, pos: 41

### Candidate
type: B, layer: 1, pos: 8

### Candidate
type: A, layer: 1, pos: 37

### Candidate
type: A, layer: 1, pos: 46

### Candidate
type: A, layer: 1, pos: 29

### Candidate
type: B, layer: 1, pos: 21

### Candidate
type: A, layer: 1, pos: 6

### Candidate
type: B, layer: 1, pos: 6

### Candidate
type: A, layer: 1, pos: 15

### Candidate
type: A, layer: 1, pos: 4

### Candidate
type: A, layer: 1, pos: 21

### Candidate
type: B, layer: 1, pos: 4

### Candidate
type: B, layer: 1, pos: 42

### Candidate
type: A, layer: 1, pos: 38

### Candidate
type: B, layer: 1, pos: 13

### Candidate
type: B, layer: 1, pos: 15

### Candidate
type: B, layer: 1, pos: 38

### Candidate
type: B, layer: 1, pos: 23

### Candidate
type: A, layer: 1, pos: 42

### Candidate
type: B, layer: 1, pos: 46

### Candidate
type: A, layer: 1, pos: 31

### Candidate
type: B, layer: 1, pos: 31

### Candidate
type: A, layer: 1, pos: 23

### Candidate
type: B, layer: 1, pos: 16

## Relational analysis of NS_A2_A1_B1_A2_B1_A2_A1_B2_B2_A2_B1
Optimization infeasible because this subproblem isn't reachable.

### Candidate
type: B, layer: 1, pos: 11

## BFS NS instance: NS_A2_A1_B1_A2_B1_A2_A2_B1_B1_A1

### Backsubstitution after applying NS history:
0: -22.3518906, 81.6434860, -18.4655514, 67.4126968, -89.7645874, 100.1090317
1: -58.9702797, 185.9140167, -49.1400375, 152.0275879, -210.9978333, 235.0540466
2: -88.2109146, 164.5646362, -74.0549774, 134.9131470, -223.1240540, 238.6195984
3: -50.7075233, 198.2664185, -42.2714348, 162.6137848, -213.3213043, 240.5378418
4: -80.6300812, 144.9612579, -67.5785675, 118.9201355, -199.5502167, 212.5398254

Time for backsubstitution: 0.91 seconds

### NS candidates at layer 1
type: B, layer: 1, pos: 20
type: A, layer: 1, pos: 41
type: B, layer: 1, pos: 41
type: B, layer: 1, pos: 39
type: B, layer: 1, pos: 19
type: A, layer: 1, pos: 20
type: A, layer: 1, pos: 39
type: A, layer: 1, pos: 19
type: B, layer: 1, pos: 14
type: A, layer: 1, pos: 25
type: B, layer: 1, pos: 25
type: B, layer: 1, pos: 21
type: B, layer: 1, pos: 23
type: A, layer: 1, pos: 23
type: A, layer: 1, pos: 14
type: A, layer: 1, pos: 6
type: B, layer: 1, pos: 6
type: A, layer: 1, pos: 46
type: A, layer: 1, pos: 38
type: B, layer: 1, pos: 15
type: A, layer: 1, pos: 4
type: B, layer: 1, pos: 13
type: B, layer: 1, pos: 16
type: A, layer: 1, pos: 21
type: A, layer: 1, pos: 11
type: A, layer: 1, pos: 15
type: B, layer: 1, pos: 4
type: A, layer: 1, pos: 29
type: B, layer: 1, pos: 12
type: A, layer: 1, pos: 12
type: B, layer: 1, pos: 29
type: B, layer: 1, pos: 37
type: A, layer: 1, pos: 42
type: B, layer: 1, pos: 46
type: B, layer: 1, pos: 38
type: A, layer: 1, pos: 8
type: B, layer: 1, pos: 31
type: A, layer: 1, pos: 31
type: A, layer: 1, pos: 37
type: B, layer: 1, pos: 42
type: B, layer: 1, pos: 8
type: B, layer: 1, pos: 11

Time for candidate selection: 0.08 seconds

### Candidate
type: B, layer: 1, pos: 20

### Candidate
type: A, layer: 1, pos: 41

### Candidate
type: B, layer: 1, pos: 41

### Candidate
type: B, layer: 1, pos: 39

### Candidate
type: B, layer: 1, pos: 19

### Candidate
type: A, layer: 1, pos: 20

### Candidate
type: A, layer: 1, pos: 39

### Candidate
type: A, layer: 1, pos: 19

### Candidate
type: B, layer: 1, pos: 14

### Candidate
type: A, layer: 1, pos: 25

### Candidate
type: B, layer: 1, pos: 25

### Candidate
type: B, layer: 1, pos: 21

### Candidate
type: B, layer: 1, pos: 23

### Candidate
type: A, layer: 1, pos: 23

### Candidate
type: A, layer: 1, pos: 14

### Candidate
type: A, layer: 1, pos: 6

### Candidate
type: B, layer: 1, pos: 6

### Candidate
type: A, layer: 1, pos: 46

### Candidate
type: A, layer: 1, pos: 38

### Candidate
type: B, layer: 1, pos: 15

### Candidate
type: A, layer: 1, pos: 4

### Candidate
type: B, layer: 1, pos: 13

### Candidate
type: B, layer: 1, pos: 16

### Candidate
type: A, layer: 1, pos: 21

### Candidate
type: A, layer: 1, pos: 11

### Candidate
type: A, layer: 1, pos: 15

### Candidate
type: B, layer: 1, pos: 4

### Candidate
type: A, layer: 1, pos: 29

### Candidate
type: B, layer: 1, pos: 12

### Candidate
type: A, layer: 1, pos: 12

### Candidate
type: B, layer: 1, pos: 29

### Candidate
type: B, layer: 1, pos: 37

### Candidate
type: A, layer: 1, pos: 42

### Candidate
type: B, layer: 1, pos: 46

### Candidate
type: B, layer: 1, pos: 38

### Candidate
type: A, layer: 1, pos: 8

### Candidate
type: B, layer: 1, pos: 31

### Candidate
type: A, layer: 1, pos: 31

### Candidate
type: A, layer: 1, pos: 37

### Candidate
type: B, layer: 1, pos: 42

### Candidate
type: B, layer: 1, pos: 8

### Candidate
type: B, layer: 1, pos: 11

## BFS NS instance: NS_A2_A1_B1_A2_B1_A2_A2_B1_B1_A2

### Backsubstitution after applying NS history:
0: -22.2197933, 80.9836044, -18.4655514, 67.4126968, -89.6324921, 99.4491577
1: -58.3009148, 185.4321747, -49.1400375, 152.0275879, -210.3284607, 234.5722046
2: -86.8606567, 163.6519470, -74.0549774, 134.9131470, -221.7738037, 237.7069244
3: -50.1980286, 197.5921021, -42.2714348, 162.6137848, -212.8118134, 239.8635254
4: -79.4424744, 144.2436371, -67.5785675, 118.9201355, -198.3626099, 211.8222046

Time for backsubstitution: 0.94 seconds

### NS candidates at layer 1
type: B, layer: 1, pos: 41
type: A, layer: 1, pos: 41
type: B, layer: 1, pos: 20
type: B, layer: 1, pos: 39
type: B, layer: 1, pos: 19
type: A, layer: 1, pos: 20
type: A, layer: 1, pos: 39
type: A, layer: 1, pos: 19
type: B, layer: 1, pos: 14
type: A, layer: 1, pos: 25
type: B, layer: 1, pos: 25
type: A, layer: 1, pos: 23
type: B, layer: 1, pos: 21
type: B, layer: 1, pos: 23
type: A, layer: 1, pos: 14
type: B, layer: 1, pos: 6
type: A, layer: 1, pos: 6
type: A, layer: 1, pos: 46
type: A, layer: 1, pos: 4
type: A, layer: 1, pos: 15
type: A, layer: 1, pos: 21
type: A, layer: 1, pos: 38
type: B, layer: 1, pos: 13
type: A, layer: 1, pos: 11
type: B, layer: 1, pos: 15
type: B, layer: 1, pos: 29
type: B, layer: 1, pos: 16
type: B, layer: 1, pos: 37
type: B, layer: 1, pos: 12
type: B, layer: 1, pos: 4
type: A, layer: 1, pos: 12
type: B, layer: 1, pos: 38
type: A, layer: 1, pos: 29
type: B, layer: 1, pos: 46
type: A, layer: 1, pos: 42
type: B, layer: 1, pos: 42
type: A, layer: 1, pos: 31
type: A, layer: 1, pos: 8
type: B, layer: 1, pos: 31
type: B, layer: 1, pos: 8
type: A, layer: 1, pos: 37
type: B, layer: 1, pos: 11

Time for candidate selection: 0.07 seconds

### Candidate
type: B, layer: 1, pos: 41

### Candidate
type: A, layer: 1, pos: 41

### Candidate
type: B, layer: 1, pos: 20

### Candidate
type: B, layer: 1, pos: 39

### Candidate
type: B, layer: 1, pos: 19

### Candidate
type: A, layer: 1, pos: 20

### Candidate
type: A, layer: 1, pos: 39

### Candidate
type: A, layer: 1, pos: 19

### Candidate
type: B, layer: 1, pos: 14

### Candidate
type: A, layer: 1, pos: 25

### Candidate
type: B, layer: 1, pos: 25

### Candidate
type: A, layer: 1, pos: 23

### Candidate
type: B, layer: 1, pos: 21

### Candidate
type: B, layer: 1, pos: 23

### Candidate
type: A, layer: 1, pos: 14

### Candidate
type: B, layer: 1, pos: 6

### Candidate
type: A, layer: 1, pos: 6

### Candidate
type: A, layer: 1, pos: 46

### Candidate
type: A, layer: 1, pos: 4

### Candidate
type: A, layer: 1, pos: 15

### Candidate
type: A, layer: 1, pos: 21

### Candidate
type: A, layer: 1, pos: 38

### Candidate
type: B, layer: 1, pos: 13

### Candidate
type: A, layer: 1, pos: 11

### Candidate
type: B, layer: 1, pos: 15

### Candidate
type: B, layer: 1, pos: 29

### Candidate
type: B, layer: 1, pos: 16

### Candidate
type: B, layer: 1, pos: 37

### Candidate
type: B, layer: 1, pos: 12

### Candidate
type: B, layer: 1, pos: 4

### Candidate
type: A, layer: 1, pos: 12

### Candidate
type: B, layer: 1, pos: 38

### Candidate
type: A, layer: 1, pos: 29

### Candidate
type: B, layer: 1, pos: 46

### Candidate
type: A, layer: 1, pos: 42

### Candidate
type: B, layer: 1, pos: 42

### Candidate
type: A, layer: 1, pos: 31

### Candidate
type: A, layer: 1, pos: 8

### Candidate
type: B, layer: 1, pos: 31

### Candidate
type: B, layer: 1, pos: 8

### Candidate
type: A, layer: 1, pos: 37

### Candidate
type: B, layer: 1, pos: 11

## BFS NS instance: NS_A2_A1_B1_A2_B1_A2_A2_B2_B1_A1

### Backsubstitution after applying NS history:
0: -22.3518906, 81.6434860, -18.5787945, 67.7266159, -90.0784988, 100.2222748
1: -58.9702797, 185.9140167, -49.5185394, 152.6435699, -211.6138458, 235.4325562
2: -88.2109146, 164.5646362, -74.6734467, 135.7630005, -223.9739075, 239.2380524
3: -50.7075233, 198.2664185, -42.6107178, 163.5640869, -214.2716064, 240.8771362
4: -80.6300812, 144.9612579, -68.1391220, 119.7065964, -200.3366699, 213.1003723

Time for backsubstitution: 0.90 seconds

### NS candidates at layer 1
type: A, layer: 1, pos: 41
type: B, layer: 1, pos: 19
type: B, layer: 1, pos: 14
type: A, layer: 1, pos: 14
type: B, layer: 1, pos: 39
type: A, layer: 1, pos: 25
type: A, layer: 1, pos: 39
type: B, layer: 1, pos: 12
type: A, layer: 1, pos: 19
type: B, layer: 1, pos: 25
type: B, layer: 1, pos: 37
type: A, layer: 1, pos: 12
type: B, layer: 1, pos: 20
type: A, layer: 1, pos: 8
type: B, layer: 1, pos: 29
type: A, layer: 1, pos: 20
type: B, layer: 1, pos: 41
type: A, layer: 1, pos: 37
type: B, layer: 1, pos: 8
type: A, layer: 1, pos: 11
type: A, layer: 1, pos: 29
type: A, layer: 1, pos: 46
type: B, layer: 1, pos: 21
type: A, layer: 1, pos: 6
type: B, layer: 1, pos: 6
type: A, layer: 1, pos: 4
type: A, layer: 1, pos: 21
type: B, layer: 1, pos: 4
type: A, layer: 1, pos: 15
type: A, layer: 1, pos: 38
type: B, layer: 1, pos: 15
type: B, layer: 1, pos: 13
type: B, layer: 1, pos: 42
type: B, layer: 1, pos: 16
type: A, layer: 1, pos: 23
type: A, layer: 1, pos: 42
type: B, layer: 1, pos: 23
type: B, layer: 1, pos: 46
type: B, layer: 1, pos: 38
type: A, layer: 1, pos: 31
type: B, layer: 1, pos: 31
type: B, layer: 1, pos: 11

Time for candidate selection: 0.07 seconds

### Candidate
type: A, layer: 1, pos: 41

### Candidate
type: B, layer: 1, pos: 19

### Candidate
type: B, layer: 1, pos: 14

### Candidate
type: A, layer: 1, pos: 14

### Candidate
type: B, layer: 1, pos: 39

### Candidate
type: A, layer: 1, pos: 25

### Candidate
type: A, layer: 1, pos: 39

### Candidate
type: B, layer: 1, pos: 12

### Candidate
type: A, layer: 1, pos: 19

### Candidate
type: B, layer: 1, pos: 25

### Candidate
type: B, layer: 1, pos: 37

### Candidate
type: A, layer: 1, pos: 12

### Candidate
type: B, layer: 1, pos: 20

### Candidate
type: A, layer: 1, pos: 8

### Candidate
type: B, layer: 1, pos: 29

### Candidate
type: A, layer: 1, pos: 20

### Candidate
type: B, layer: 1, pos: 41

### Candidate
type: A, layer: 1, pos: 37

### Candidate
type: B, layer: 1, pos: 8

### Candidate
type: A, layer: 1, pos: 11

### Candidate
type: A, layer: 1, pos: 29

### Candidate
type: A, layer: 1, pos: 46

### Candidate
type: B, layer: 1, pos: 21

### Candidate
type: A, layer: 1, pos: 6

### Candidate
type: B, layer: 1, pos: 6

### Candidate
type: A, layer: 1, pos: 4

### Candidate
type: A, layer: 1, pos: 21

### Candidate
type: B, layer: 1, pos: 4

### Candidate
type: A, layer: 1, pos: 15

### Candidate
type: A, layer: 1, pos: 38

### Candidate
type: B, layer: 1, pos: 15

### Candidate
type: B, layer: 1, pos: 13

### Candidate
type: B, layer: 1, pos: 42

### Candidate
type: B, layer: 1, pos: 16

### Candidate
type: A, layer: 1, pos: 23

### Candidate
type: A, layer: 1, pos: 42

### Candidate
type: B, layer: 1, pos: 23

### Candidate
type: B, layer: 1, pos: 46

### Candidate
type: B, layer: 1, pos: 38

### Candidate
type: A, layer: 1, pos: 31

### Candidate
type: B, layer: 1, pos: 31

### Candidate
type: B, layer: 1, pos: 11

## BFS NS instance: NS_A2_A1_B1_A2_B1_A2_A2_B2_B1_A2

### Backsubstitution after applying NS history:
0: -22.2197933, 80.9836044, -18.5787945, 67.7266159, -89.9464111, 99.5624008
1: -58.3009148, 185.4321747, -49.5185394, 152.6435699, -210.9444885, 234.9507141
2: -86.8606567, 163.6519470, -74.6734467, 135.7630005, -222.6236572, 238.3253937
3: -50.1980286, 197.5921021, -42.6107178, 163.5640869, -213.7621155, 240.2028198
4: -79.4424744, 144.2436371, -68.1391220, 119.7065964, -199.1490479, 212.3827515

Time for backsubstitution: 0.90 seconds

### NS candidates at layer 1
type: A, layer: 1, pos: 41
type: B, layer: 1, pos: 19
type: B, layer: 1, pos: 14
type: A, layer: 1, pos: 14
type: B, layer: 1, pos: 39
type: A, layer: 1, pos: 39
type: A, layer: 1, pos: 25
type: B, layer: 1, pos: 12
type: B, layer: 1, pos: 37
type: A, layer: 1, pos: 19
type: A, layer: 1, pos: 20
type: B, layer: 1, pos: 25
type: B, layer: 1, pos: 29
type: A, layer: 1, pos: 12
type: A, layer: 1, pos: 8
type: B, layer: 1, pos: 20
type: B, layer: 1, pos: 41
type: B, layer: 1, pos: 8
type: A, layer: 1, pos: 37
type: A, layer: 1, pos: 11
type: A, layer: 1, pos: 46
type: A, layer: 1, pos: 29
type: B, layer: 1, pos: 21
type: A, layer: 1, pos: 6
type: B, layer: 1, pos: 6
type: A, layer: 1, pos: 4
type: A, layer: 1, pos: 15
type: A, layer: 1, pos: 21
type: B, layer: 1, pos: 42
type: B, layer: 1, pos: 4
type: B, layer: 1, pos: 13
type: A, layer: 1, pos: 38
type: B, layer: 1, pos: 15
type: B, layer: 1, pos: 38
type: B, layer: 1, pos: 46
type: B, layer: 1, pos: 16
type: A, layer: 1, pos: 23
type: A, layer: 1, pos: 42
type: A, layer: 1, pos: 31
type: B, layer: 1, pos: 23
type: B, layer: 1, pos: 31
type: B, layer: 1, pos: 11

Time for candidate selection: 0.07 seconds

### Candidate
type: A, layer: 1, pos: 41

### Candidate
type: B, layer: 1, pos: 19

### Candidate
type: B, layer: 1, pos: 14

### Candidate
type: A, layer: 1, pos: 14

### Candidate
type: B, layer: 1, pos: 39

### Candidate
type: A, layer: 1, pos: 39

### Candidate
type: A, layer: 1, pos: 25

### Candidate
type: B, layer: 1, pos: 12

### Candidate
type: B, layer: 1, pos: 37

### Candidate
type: A, layer: 1, pos: 19

### Candidate
type: A, layer: 1, pos: 20

### Candidate
type: B, layer: 1, pos: 25

### Candidate
type: B, layer: 1, pos: 29

### Candidate
type: A, layer: 1, pos: 12

### Candidate
type: A, layer: 1, pos: 8

### Candidate
type: B, layer: 1, pos: 20

### Candidate
type: B, layer: 1, pos: 41

### Candidate
type: B, layer: 1, pos: 8

### Candidate
type: A, layer: 1, pos: 37

### Candidate
type: A, layer: 1, pos: 11

### Candidate
type: A, layer: 1, pos: 46

### Candidate
type: A, layer: 1, pos: 29

### Candidate
type: B, layer: 1, pos: 21

### Candidate
type: A, layer: 1, pos: 6

### Candidate
type: B, layer: 1, pos: 6

### Candidate
type: A, layer: 1, pos: 4

### Candidate
type: A, layer: 1, pos: 15

### Candidate
type: A, layer: 1, pos: 21

### Candidate
type: B, layer: 1, pos: 42

### Candidate
type: B, layer: 1, pos: 4

### Candidate
type: B, layer: 1, pos: 13

### Candidate
type: A, layer: 1, pos: 38

### Candidate
type: B, layer: 1, pos: 15

### Candidate
type: B, layer: 1, pos: 38

### Candidate
type: B, layer: 1, pos: 46

### Candidate
type: B, layer: 1, pos: 16

### Candidate
type: A, layer: 1, pos: 23

### Candidate
type: A, layer: 1, pos: 42

### Candidate
type: A, layer: 1, pos: 31

### Candidate
type: B, layer: 1, pos: 23

### Candidate
type: B, layer: 1, pos: 31

### Candidate
type: B, layer: 1, pos: 11

## BFS NS instance: NS_A2_A1_B1_A2_B1_A2_A2_B2_B2_A1

### Backsubstitution after applying NS history:
0: -22.3518906, 81.6434860, -18.6771259, 68.0861969, -90.4380798, 100.3206100
1: -58.9702797, 185.9140167, -49.5544319, 153.9679413, -212.9381714, 235.4684448
2: -88.2109146, 164.5646362, -74.3130569, 136.8685760, -225.0794830, 238.8776550
3: -50.7075233, 198.2664185, -42.6558762, 164.7959595, -215.5034790, 240.9222565
4: -80.6300812, 144.9612579, -67.8917389, 120.6695709, -201.2996521, 212.8529816

Time for backsubstitution: 0.90 seconds

### NS candidates at layer 1
type: A, layer: 1, pos: 41
type: B, layer: 1, pos: 19
type: B, layer: 1, pos: 14
type: A, layer: 1, pos: 14
type: B, layer: 1, pos: 39
type: B, layer: 1, pos: 12
type: A, layer: 1, pos: 39
type: A, layer: 1, pos: 19
type: A, layer: 1, pos: 12
type: B, layer: 1, pos: 37
type: A, layer: 1, pos: 8
type: A, layer: 1, pos: 25
type: B, layer: 1, pos: 29
type: A, layer: 1, pos: 37
type: B, layer: 1, pos: 8
type: B, layer: 1, pos: 20
type: A, layer: 1, pos: 29
type: B, layer: 1, pos: 25
type: A, layer: 1, pos: 46
type: B, layer: 1, pos: 41
type: A, layer: 1, pos: 11
type: B, layer: 1, pos: 21
type: A, layer: 1, pos: 20
type: A, layer: 1, pos: 6
type: A, layer: 1, pos: 21
type: B, layer: 1, pos: 15
type: B, layer: 1, pos: 6
type: A, layer: 1, pos: 38
type: A, layer: 1, pos: 15
type: A, layer: 1, pos: 4
type: B, layer: 1, pos: 13
type: B, layer: 1, pos: 16
type: B, layer: 1, pos: 4
type: B, layer: 1, pos: 42
type: B, layer: 1, pos: 23
type: B, layer: 1, pos: 46
type: A, layer: 1, pos: 23
type: A, layer: 1, pos: 42
type: A, layer: 1, pos: 31
type: B, layer: 1, pos: 31
type: B, layer: 1, pos: 38
type: B, layer: 1, pos: 11

Time for candidate selection: 0.07 seconds

### Candidate
type: A, layer: 1, pos: 41

### Candidate
type: B, layer: 1, pos: 19

### Candidate
type: B, layer: 1, pos: 14

### Candidate
type: A, layer: 1, pos: 14

### Candidate
type: B, layer: 1, pos: 39

### Candidate
type: B, layer: 1, pos: 12

### Candidate
type: A, layer: 1, pos: 39

### Candidate
type: A, layer: 1, pos: 19

### Candidate
type: A, layer: 1, pos: 12

### Candidate
type: B, layer: 1, pos: 37

### Candidate
type: A, layer: 1, pos: 8

### Candidate
type: A, layer: 1, pos: 25

### Candidate
type: B, layer: 1, pos: 29

### Candidate
type: A, layer: 1, pos: 37

### Candidate
type: B, layer: 1, pos: 8

### Candidate
type: B, layer: 1, pos: 20

### Candidate
type: A, layer: 1, pos: 29

### Candidate
type: B, layer: 1, pos: 25

### Candidate
type: A, layer: 1, pos: 46

### Candidate
type: B, layer: 1, pos: 41

### Candidate
type: A, layer: 1, pos: 11

### Candidate
type: B, layer: 1, pos: 21

### Candidate
type: A, layer: 1, pos: 20

### Candidate
type: A, layer: 1, pos: 6

### Candidate
type: A, layer: 1, pos: 21

### Candidate
type: B, layer: 1, pos: 15

### Candidate
type: B, layer: 1, pos: 6

### Candidate
type: A, layer: 1, pos: 38

### Candidate
type: A, layer: 1, pos: 15

### Candidate
type: A, layer: 1, pos: 4

### Candidate
type: B, layer: 1, pos: 13

### Candidate
type: B, layer: 1, pos: 16

### Candidate
type: B, layer: 1, pos: 4

### Candidate
type: B, layer: 1, pos: 42

## Relational analysis of NS_A2_A1_B1_A2_B1_A2_A2_B2_B2_A1_B1

### Relational analysis result of NS_A2_A1_B1_A2_B1_A2_A2_B2_B2_A1_B1
Status: Status.UNKNOWN
Output dim: 0, lower bound: -81.8604245, upper bound: 81.8610367
time: 1.47 seconds

## Relational analysis of NS_A2_A1_B1_A2_B1_A2_A2_B2_B2_A1_B2
Optimization infeasible because this subproblem isn't reachable.

### Candidate
type: B, layer: 1, pos: 23

### Candidate
type: B, layer: 1, pos: 46

### Candidate
type: A, layer: 1, pos: 23

### Candidate
type: A, layer: 1, pos: 42

### Candidate
type: A, layer: 1, pos: 31

### Candidate
type: B, layer: 1, pos: 31

### Candidate
type: B, layer: 1, pos: 38

### Candidate
type: B, layer: 1, pos: 11

## BFS NS instance: NS_A2_A1_B1_A2_B1_A2_A2_B2_B2_A2

### Backsubstitution after applying NS history:
0: -22.2197933, 80.9836044, -18.6771259, 68.0861969, -90.3059845, 99.6607285
1: -58.3009148, 185.4321747, -49.5544319, 153.9679413, -212.2688141, 234.9866028
2: -86.8606567, 163.6519470, -74.3130569, 136.8685760, -223.7292023, 237.9649963
3: -50.1980286, 197.5921021, -42.6558762, 164.7959595, -214.9939880, 240.2479248
4: -79.4424744, 144.2436371, -67.8917389, 120.6695709, -200.1120453, 212.1353455

Time for backsubstitution: 0.98 seconds

### NS candidates at layer 1
type: A, layer: 1, pos: 41
type: B, layer: 1, pos: 19
type: B, layer: 1, pos: 14
type: A, layer: 1, pos: 14
type: B, layer: 1, pos: 39
type: A, layer: 1, pos: 39
type: B, layer: 1, pos: 12
type: A, layer: 1, pos: 19
type: A, layer: 1, pos: 12
type: B, layer: 1, pos: 37
type: B, layer: 1, pos: 29
type: A, layer: 1, pos: 8
type: A, layer: 1, pos: 25
type: B, layer: 1, pos: 8
type: A, layer: 1, pos: 37
type: B, layer: 1, pos: 41
type: A, layer: 1, pos: 46
type: B, layer: 1, pos: 21
type: A, layer: 1, pos: 11
type: A, layer: 1, pos: 29
type: B, layer: 1, pos: 20
type: A, layer: 1, pos: 20
type: B, layer: 1, pos: 25
type: A, layer: 1, pos: 21
type: A, layer: 1, pos: 6
type: B, layer: 1, pos: 6
type: A, layer: 1, pos: 15
type: B, layer: 1, pos: 15
type: A, layer: 1, pos: 4
type: A, layer: 1, pos: 38
type: B, layer: 1, pos: 13
type: B, layer: 1, pos: 42
type: B, layer: 1, pos: 4
type: B, layer: 1, pos: 16
type: B, layer: 1, pos: 46
type: B, layer: 1, pos: 38
type: B, layer: 1, pos: 23
type: A, layer: 1, pos: 31
type: B, layer: 1, pos: 31
type: A, layer: 1, pos: 23
type: A, layer: 1, pos: 42
type: B, layer: 1, pos: 11

Time for candidate selection: 0.07 seconds

### Candidate
type: A, layer: 1, pos: 41

### Candidate
type: B, layer: 1, pos: 19

### Candidate
type: B, layer: 1, pos: 14

### Candidate
type: A, layer: 1, pos: 14

### Candidate
type: B, layer: 1, pos: 39

### Candidate
type: A, layer: 1, pos: 39

### Candidate
type: B, layer: 1, pos: 12

### Candidate
type: A, layer: 1, pos: 19

### Candidate
type: A, layer: 1, pos: 12

### Candidate
type: B, layer: 1, pos: 37

### Candidate
type: B, layer: 1, pos: 29

### Candidate
type: A, layer: 1, pos: 8

### Candidate
type: A, layer: 1, pos: 25

### Candidate
type: B, layer: 1, pos: 8

### Candidate
type: A, layer: 1, pos: 37

### Candidate
type: B, layer: 1, pos: 41

### Candidate
type: A, layer: 1, pos: 46

### Candidate
type: B, layer: 1, pos: 21

### Candidate
type: A, layer: 1, pos: 11

### Candidate
type: A, layer: 1, pos: 29

### Candidate
type: B, layer: 1, pos: 20

### Candidate
type: A, layer: 1, pos: 20

### Candidate
type: B, layer: 1, pos: 25

### Candidate
type: A, layer: 1, pos: 21

### Candidate
type: A, layer: 1, pos: 6

### Candidate
type: B, layer: 1, pos: 6

### Candidate
type: A, layer: 1, pos: 15

### Candidate
type: B, layer: 1, pos: 15

### Candidate
type: A, layer: 1, pos: 4

### Candidate
type: A, layer: 1, pos: 38

### Candidate
type: B, layer: 1, pos: 13

### Candidate
type: B, layer: 1, pos: 42

## Relational analysis of NS_A2_A1_B1_A2_B1_A2_A2_B2_B2_A2_B1

### Relational analysis result of NS_A2_A1_B1_A2_B1_A2_A2_B2_B2_A2_B1
Status: Status.UNKNOWN
Output dim: 0, lower bound: -81.8604245, upper bound: 81.8610367
time: 0.85 seconds

## Relational analysis of NS_A2_A1_B1_A2_B1_A2_A2_B2_B2_A2_B2
Optimization infeasible because this subproblem isn't reachable.

### Candidate
type: B, layer: 1, pos: 4

### Candidate
type: B, layer: 1, pos: 16

### Candidate
type: B, layer: 1, pos: 46

### Candidate
type: B, layer: 1, pos: 38

### Candidate
type: B, layer: 1, pos: 23

### Candidate
type: A, layer: 1, pos: 31

### Candidate
type: B, layer: 1, pos: 31

### Candidate
type: A, layer: 1, pos: 23

### Candidate
type: A, layer: 1, pos: 42

### Candidate
type: B, layer: 1, pos: 11

## BFS NS instance: NS_A2_A1_B1_A2_B2_A1_B1_B1_A1_A1

### Backsubstitution after applying NS history:
0: -19.6321354, 71.9451523, -22.7525654, 83.2576599, -102.8897781, 94.6977158
1: -52.1180496, 162.3968811, -60.1493530, 189.8360443, -241.9541016, 222.5462189
2: -78.0430450, 144.1837769, -90.1654434, 166.8709564, -244.9140015, 234.3492126
3: -44.8063202, 173.4655457, -51.7128868, 202.6529388, -247.4592285, 225.1784363
4: -71.3078003, 126.9616776, -82.3338470, 147.1033783, -218.4111786, 209.2954865

Time for backsubstitution: 0.90 seconds

### NS candidates at layer 1
type: B, layer: 1, pos: 41
type: A, layer: 1, pos: 41
type: A, layer: 1, pos: 19
type: A, layer: 1, pos: 20
type: B, layer: 1, pos: 20
type: A, layer: 1, pos: 39
type: B, layer: 1, pos: 39
type: B, layer: 1, pos: 19
type: A, layer: 1, pos: 14
type: A, layer: 1, pos: 25
type: B, layer: 1, pos: 25
type: B, layer: 1, pos: 14
type: A, layer: 1, pos: 23
type: B, layer: 1, pos: 23
type: A, layer: 1, pos: 6
type: B, layer: 1, pos: 6
type: A, layer: 1, pos: 21
type: B, layer: 1, pos: 46
type: B, layer: 1, pos: 21
type: B, layer: 1, pos: 16
type: B, layer: 1, pos: 4
type: B, layer: 1, pos: 15
type: A, layer: 1, pos: 29
type: B, layer: 1, pos: 13
type: A, layer: 1, pos: 38
type: A, layer: 1, pos: 15
type: B, layer: 1, pos: 11
type: A, layer: 1, pos: 46
type: A, layer: 1, pos: 4
type: B, layer: 1, pos: 38
type: A, layer: 1, pos: 12
type: A, layer: 1, pos: 37
type: B, layer: 1, pos: 12
type: B, layer: 1, pos: 29
type: A, layer: 1, pos: 16
type: A, layer: 1, pos: 31
type: B, layer: 1, pos: 31
type: A, layer: 1, pos: 8
type: B, layer: 1, pos: 8
type: B, layer: 1, pos: 22
type: B, layer: 1, pos: 37
type: A, layer: 1, pos: 11

Time for candidate selection: 0.07 seconds

### Candidate
type: B, layer: 1, pos: 41

### Candidate
type: A, layer: 1, pos: 41

### Candidate
type: A, layer: 1, pos: 19

### Candidate
type: A, layer: 1, pos: 20

### Candidate
type: B, layer: 1, pos: 20

### Candidate
type: A, layer: 1, pos: 39

### Candidate
type: B, layer: 1, pos: 39

### Candidate
type: B, layer: 1, pos: 19

### Candidate
type: A, layer: 1, pos: 14

### Candidate
type: A, layer: 1, pos: 25

### Candidate
type: B, layer: 1, pos: 25

### Candidate
type: B, layer: 1, pos: 14

### Candidate
type: A, layer: 1, pos: 23

### Candidate
type: B, layer: 1, pos: 23

### Candidate
type: A, layer: 1, pos: 6

### Candidate
type: B, layer: 1, pos: 6

### Candidate
type: A, layer: 1, pos: 21

### Candidate
type: B, layer: 1, pos: 46

### Candidate
type: B, layer: 1, pos: 21

### Candidate
type: B, layer: 1, pos: 16

### Candidate
type: B, layer: 1, pos: 4

### Candidate
type: B, layer: 1, pos: 15

### Candidate
type: A, layer: 1, pos: 29

### Candidate
type: B, layer: 1, pos: 13

### Candidate
type: A, layer: 1, pos: 38

### Candidate
type: A, layer: 1, pos: 15

### Candidate
type: B, layer: 1, pos: 11

### Candidate
type: A, layer: 1, pos: 46

### Candidate
type: A, layer: 1, pos: 4

### Candidate
type: B, layer: 1, pos: 38

### Candidate
type: A, layer: 1, pos: 12

### Candidate
type: A, layer: 1, pos: 37

### Candidate
type: B, layer: 1, pos: 12

### Candidate
type: B, layer: 1, pos: 29

### Candidate
type: A, layer: 1, pos: 16

### Candidate
type: A, layer: 1, pos: 31

### Candidate
type: B, layer: 1, pos: 31

### Candidate
type: A, layer: 1, pos: 8

### Candidate
type: B, layer: 1, pos: 8

### Candidate
type: B, layer: 1, pos: 22

## Relational analysis of NS_A2_A1_B1_A2_B2_A1_B1_B1_A1_A1_B1

### Relational analysis result of NS_A2_A1_B1_A2_B2_A1_B1_B1_A1_A1_B1
Status: Status.UNKNOWN
Output dim: 0, lower bound: -81.8625266, upper bound: 81.8625729
time: 0.88 seconds

## Relational analysis of NS_A2_A1_B1_A2_B2_A1_B1_B1_A1_A1_B2

### Relational analysis result of NS_A2_A1_B1_A2_B2_A1_B1_B1_A1_A1_B2
Status: Status.UNKNOWN
Output dim: 0, lower bound: -81.8625266, upper bound: 81.8625774
time: 1.14 seconds

## BFS NS instance: NS_A2_A1_B1_A2_B2_A1_B1_B1_A1_A2

### Backsubstitution after applying NS history:
0: -19.0841656, 69.7065048, -22.7525654, 83.2576599, -102.3418198, 92.4590683
1: -50.6829987, 157.1727448, -60.1493530, 189.8360443, -240.5190430, 217.3220825
2: -76.0643387, 139.9002533, -90.1654434, 166.8709564, -242.9352875, 230.0657043
3: -43.5968208, 167.9398499, -51.7128868, 202.6529388, -246.2497101, 219.6527405
4: -69.4886246, 123.2182236, -82.3338470, 147.1033783, -216.5920105, 205.5520477

Time for backsubstitution: 0.91 seconds

### NS candidates at layer 1
type: B, layer: 1, pos: 41
type: A, layer: 1, pos: 20
type: A, layer: 1, pos: 19
type: B, layer: 1, pos: 20
type: B, layer: 1, pos: 39
type: A, layer: 1, pos: 39
type: A, layer: 1, pos: 41
type: B, layer: 1, pos: 19
type: B, layer: 1, pos: 25
type: A, layer: 1, pos: 14
type: A, layer: 1, pos: 25
type: B, layer: 1, pos: 14
type: A, layer: 1, pos: 23
type: B, layer: 1, pos: 23
type: A, layer: 1, pos: 6
type: B, layer: 1, pos: 6
type: A, layer: 1, pos: 21
type: B, layer: 1, pos: 21
type: B, layer: 1, pos: 46
type: B, layer: 1, pos: 4
type: B, layer: 1, pos: 15
type: B, layer: 1, pos: 13
type: A, layer: 1, pos: 38
type: A, layer: 1, pos: 29
type: B, layer: 1, pos: 16
type: A, layer: 1, pos: 46
type: A, layer: 1, pos: 15
type: B, layer: 1, pos: 12
type: B, layer: 1, pos: 11
type: A, layer: 1, pos: 4
type: A, layer: 1, pos: 37
type: B, layer: 1, pos: 38
type: A, layer: 1, pos: 16
type: A, layer: 1, pos: 12
type: B, layer: 1, pos: 29
type: B, layer: 1, pos: 8
type: A, layer: 1, pos: 31
type: B, layer: 1, pos: 31
type: B, layer: 1, pos: 22
type: B, layer: 1, pos: 37
type: A, layer: 1, pos: 8
type: A, layer: 1, pos: 11

Time for candidate selection: 0.07 seconds

### Candidate
type: B, layer: 1, pos: 41

### Candidate
type: A, layer: 1, pos: 20

### Candidate
type: A, layer: 1, pos: 19

### Candidate
type: B, layer: 1, pos: 20

### Candidate
type: B, layer: 1, pos: 39

### Candidate
type: A, layer: 1, pos: 39

### Candidate
type: A, layer: 1, pos: 41

### Candidate
type: B, layer: 1, pos: 19

### Candidate
type: B, layer: 1, pos: 25

### Candidate
type: A, layer: 1, pos: 14

### Candidate
type: A, layer: 1, pos: 25

### Candidate
type: B, layer: 1, pos: 14

### Candidate
type: A, layer: 1, pos: 23

### Candidate
type: B, layer: 1, pos: 23

### Candidate
type: A, layer: 1, pos: 6

### Candidate
type: B, layer: 1, pos: 6

### Candidate
type: A, layer: 1, pos: 21

### Candidate
type: B, layer: 1, pos: 21

### Candidate
type: B, layer: 1, pos: 46

### Candidate
type: B, layer: 1, pos: 4

### Candidate
type: B, layer: 1, pos: 15

### Candidate
type: B, layer: 1, pos: 13

### Candidate
type: A, layer: 1, pos: 38

### Candidate
type: A, layer: 1, pos: 29

### Candidate
type: B, layer: 1, pos: 16

### Candidate
type: A, layer: 1, pos: 46

### Candidate
type: A, layer: 1, pos: 15

### Candidate
type: B, layer: 1, pos: 12

### Candidate
type: B, layer: 1, pos: 11

### Candidate
type: A, layer: 1, pos: 4

### Candidate
type: A, layer: 1, pos: 37

### Candidate
type: B, layer: 1, pos: 38

### Candidate
type: A, layer: 1, pos: 16

### Candidate
type: A, layer: 1, pos: 12

### Candidate
type: B, layer: 1, pos: 29

### Candidate
type: B, layer: 1, pos: 8

### Candidate
type: A, layer: 1, pos: 31

### Candidate
type: B, layer: 1, pos: 31

### Candidate
type: B, layer: 1, pos: 22

## Relational analysis of NS_A2_A1_B1_A2_B2_A1_B1_B1_A1_A2_B1

### Relational analysis result of NS_A2_A1_B1_A2_B2_A1_B1_B1_A1_A2_B1
Status: Status.UNKNOWN
Output dim: 0, lower bound: -81.8625266, upper bound: 81.8629716
time: 0.81 seconds

## Relational analysis of NS_A2_A1_B1_A2_B2_A1_B1_B1_A1_A2_B2

### Relational analysis result of NS_A2_A1_B1_A2_B2_A1_B1_B1_A1_A2_B2
Status: Status.UNKNOWN
Output dim: 0, lower bound: -81.8625266, upper bound: 81.8629776
time: 0.80 seconds

## BFS NS instance: NS_A2_A1_B1_A2_B2_A1_B1_B1_A2_A1

### Backsubstitution after applying NS history:
0: -19.8337536, 72.5302277, -22.3819027, 81.8603592, -101.6941147, 94.9121323
1: -52.3728676, 164.4721069, -59.1157532, 186.8670502, -239.2399139, 223.5878601
2: -78.1087875, 145.9137421, -88.5823746, 164.1040192, -242.2127991, 234.4960785
3: -45.1153412, 175.5155945, -50.8336487, 199.4884644, -244.6038055, 226.3492432
4: -71.4802246, 128.5213470, -80.8856812, 144.7041016, -216.1842957, 209.4070282

Time for backsubstitution: 1.10 seconds

### NS candidates at layer 1
type: B, layer: 1, pos: 41
type: A, layer: 1, pos: 41
type: A, layer: 1, pos: 20
type: B, layer: 1, pos: 20
type: A, layer: 1, pos: 19
type: A, layer: 1, pos: 39
type: B, layer: 1, pos: 39
type: B, layer: 1, pos: 19
type: A, layer: 1, pos: 25
type: A, layer: 1, pos: 14
type: B, layer: 1, pos: 25
type: B, layer: 1, pos: 14
type: A, layer: 1, pos: 23
type: B, layer: 1, pos: 6
type: B, layer: 1, pos: 23
type: A, layer: 1, pos: 6
type: A, layer: 1, pos: 21
type: B, layer: 1, pos: 21
type: B, layer: 1, pos: 46
type: B, layer: 1, pos: 4
type: B, layer: 1, pos: 16
type: A, layer: 1, pos: 15
type: B, layer: 1, pos: 13
type: B, layer: 1, pos: 15
type: B, layer: 1, pos: 38
type: B, layer: 1, pos: 11
type: A, layer: 1, pos: 29
type: A, layer: 1, pos: 46
type: A, layer: 1, pos: 38
type: B, layer: 1, pos: 12
type: A, layer: 1, pos: 4
type: A, layer: 1, pos: 12
type: B, layer: 1, pos: 29
type: A, layer: 1, pos: 16
type: A, layer: 1, pos: 37
type: B, layer: 1, pos: 22
type: B, layer: 1, pos: 37
type: A, layer: 1, pos: 31
type: B, layer: 1, pos: 8
type: B, layer: 1, pos: 31
type: A, layer: 1, pos: 8
type: A, layer: 1, pos: 11

Time for candidate selection: 0.09 seconds

### Candidate
type: B, layer: 1, pos: 41

### Candidate
type: A, layer: 1, pos: 41

### Candidate
type: A, layer: 1, pos: 20

### Candidate
type: B, layer: 1, pos: 20

### Candidate
type: A, layer: 1, pos: 19

### Candidate
type: A, layer: 1, pos: 39

### Candidate
type: B, layer: 1, pos: 39

### Candidate
type: B, layer: 1, pos: 19

### Candidate
type: A, layer: 1, pos: 25

### Candidate
type: A, layer: 1, pos: 14

### Candidate
type: B, layer: 1, pos: 25

### Candidate
type: B, layer: 1, pos: 14

### Candidate
type: A, layer: 1, pos: 23

### Candidate
type: B, layer: 1, pos: 6

### Candidate
type: B, layer: 1, pos: 23

### Candidate
type: A, layer: 1, pos: 6

### Candidate
type: A, layer: 1, pos: 21

### Candidate
type: B, layer: 1, pos: 21

### Candidate
type: B, layer: 1, pos: 46

### Candidate
type: B, layer: 1, pos: 4

### Candidate
type: B, layer: 1, pos: 16

### Candidate
type: A, layer: 1, pos: 15

### Candidate
type: B, layer: 1, pos: 13

### Candidate
type: B, layer: 1, pos: 15

### Candidate
type: B, layer: 1, pos: 38

### Candidate
type: B, layer: 1, pos: 11

### Candidate
type: A, layer: 1, pos: 29

### Candidate
type: A, layer: 1, pos: 46

### Candidate
type: A, layer: 1, pos: 38

### Candidate
type: B, layer: 1, pos: 12

### Candidate
type: A, layer: 1, pos: 4

### Candidate
type: A, layer: 1, pos: 12

### Candidate
type: B, layer: 1, pos: 29

### Candidate
type: A, layer: 1, pos: 16

### Candidate
type: A, layer: 1, pos: 37

### Candidate
type: B, layer: 1, pos: 22

### Candidate
type: B, layer: 1, pos: 37

### Candidate
type: A, layer: 1, pos: 31

### Candidate
type: B, layer: 1, pos: 8

### Candidate
type: B, layer: 1, pos: 31

### Candidate
type: A, layer: 1, pos: 8

### Candidate
type: A, layer: 1, pos: 11

## BFS NS instance: NS_A2_A1_B1_A2_B2_A1_B1_B1_A2_A2

### Backsubstitution after applying NS history:
0: -19.2441559, 70.1604004, -22.3819027, 81.8603592, -101.1045074, 92.5423050
1: -50.8282700, 158.9814758, -59.1157532, 186.8670502, -237.6953125, 218.0972137
2: -75.9408569, 141.3431091, -88.5823746, 164.1040192, -240.0448761, 229.9254761
3: -43.7825432, 169.7017059, -50.8336487, 199.4884644, -243.2710114, 220.5353546
4: -69.4735336, 124.4805603, -80.8856812, 144.7041016, -214.1776428, 205.3662415

Time for backsubstitution: 1.13 seconds

### NS candidates at layer 1
type: B, layer: 1, pos: 41
type: A, layer: 1, pos: 19
type: A, layer: 1, pos: 41
type: B, layer: 1, pos: 6
type: A, layer: 1, pos: 6
type: B, layer: 1, pos: 19
type: B, layer: 1, pos: 25
type: A, layer: 1, pos: 25
type: B, layer: 1, pos: 46
type: A, layer: 1, pos: 21
type: B, layer: 1, pos: 21
type: B, layer: 1, pos: 4
type: A, layer: 1, pos: 38
type: B, layer: 1, pos: 11
type: B, layer: 1, pos: 38
type: B, layer: 1, pos: 13
type: A, layer: 1, pos: 46
type: B, layer: 1, pos: 12
type: A, layer: 1, pos: 14
type: A, layer: 1, pos: 29
type: A, layer: 1, pos: 4
type: A, layer: 1, pos: 12
type: B, layer: 1, pos: 14
type: B, layer: 1, pos: 39
type: B, layer: 1, pos: 29
type: A, layer: 1, pos: 31
type: B, layer: 1, pos: 31
type: A, layer: 1, pos: 20
type: B, layer: 1, pos: 8
type: B, layer: 1, pos: 15
type: B, layer: 1, pos: 16
type: A, layer: 1, pos: 15
type: A, layer: 1, pos: 16
type: A, layer: 1, pos: 39
type: A, layer: 1, pos: 23
type: B, layer: 1, pos: 22
type: B, layer: 1, pos: 20
type: A, layer: 1, pos: 37
type: A, layer: 1, pos: 11
type: B, layer: 1, pos: 23
type: B, layer: 1, pos: 37
type: A, layer: 1, pos: 8

Time for candidate selection: 0.10 seconds

### Candidate
type: B, layer: 1, pos: 41

### Candidate
type: A, layer: 1, pos: 19

### Candidate
type: A, layer: 1, pos: 41

### Candidate
type: B, layer: 1, pos: 6

### Candidate
type: A, layer: 1, pos: 6

### Candidate
type: B, layer: 1, pos: 19

### Candidate
type: B, layer: 1, pos: 25

### Candidate
type: A, layer: 1, pos: 25

### Candidate
type: B, layer: 1, pos: 46

### Candidate
type: A, layer: 1, pos: 21

### Candidate
type: B, layer: 1, pos: 21

### Candidate
type: B, layer: 1, pos: 4

### Candidate
type: A, layer: 1, pos: 38

### Candidate
type: B, layer: 1, pos: 11

### Candidate
type: B, layer: 1, pos: 38

### Candidate
type: B, layer: 1, pos: 13

### Candidate
type: A, layer: 1, pos: 46

### Candidate
type: B, layer: 1, pos: 12

### Candidate
type: A, layer: 1, pos: 14

### Candidate
type: A, layer: 1, pos: 29

### Candidate
type: A, layer: 1, pos: 4

### Candidate
type: A, layer: 1, pos: 12

### Candidate
type: B, layer: 1, pos: 14

### Candidate
type: B, layer: 1, pos: 39

### Candidate
type: B, layer: 1, pos: 29

### Candidate
type: A, layer: 1, pos: 31

### Candidate
type: B, layer: 1, pos: 31

### Candidate
type: A, layer: 1, pos: 20

### Candidate
type: B, layer: 1, pos: 8

### Candidate
type: B, layer: 1, pos: 15

### Candidate
type: B, layer: 1, pos: 16

### Candidate
type: A, layer: 1, pos: 15

### Candidate
type: A, layer: 1, pos: 16

### Candidate
type: A, layer: 1, pos: 39

### Candidate
type: A, layer: 1, pos: 23

### Candidate
type: B, layer: 1, pos: 22

### Candidate
type: B, layer: 1, pos: 20

### Candidate
type: A, layer: 1, pos: 37

### Candidate
type: A, layer: 1, pos: 11

### Candidate
type: B, layer: 1, pos: 23

### Candidate
type: B, layer: 1, pos: 37

### Candidate
type: A, layer: 1, pos: 8

## BFS NS instance: NS_A2_A1_B1_A2_B2_A1_B2_B1_B1_A1

### Backsubstitution after applying NS history:
0: -18.9003239, 69.0304260, -22.3502445, 81.6966095, -100.5969238, 91.3806458
1: -50.1970177, 155.6475983, -59.1535072, 186.1858215, -236.3828430, 214.8010712
2: -75.3415909, 138.5106201, -88.7366409, 163.8814545, -239.2230377, 227.2472382
3: -43.1801605, 166.3397522, -50.8682022, 198.9301300, -242.1102600, 217.2079468
4: -68.8156433, 122.0141144, -81.0366821, 144.4750519, -213.2906799, 203.0507965

Time for backsubstitution: 1.13 seconds

### NS candidates at layer 1
type: A, layer: 1, pos: 41
type: A, layer: 1, pos: 14
type: A, layer: 1, pos: 19
type: A, layer: 1, pos: 39
type: B, layer: 1, pos: 14
type: B, layer: 1, pos: 12
type: A, layer: 1, pos: 25
type: B, layer: 1, pos: 39
type: B, layer: 1, pos: 19
type: B, layer: 1, pos: 25
type: A, layer: 1, pos: 20
type: B, layer: 1, pos: 41
type: A, layer: 1, pos: 37
type: A, layer: 1, pos: 12
type: B, layer: 1, pos: 37
type: A, layer: 1, pos: 8
type: B, layer: 1, pos: 29
type: B, layer: 1, pos: 8
type: A, layer: 1, pos: 29
type: A, layer: 1, pos: 21
type: B, layer: 1, pos: 20
type: A, layer: 1, pos: 6
type: B, layer: 1, pos: 6
type: A, layer: 1, pos: 11
type: A, layer: 1, pos: 46
type: B, layer: 1, pos: 4
type: B, layer: 1, pos: 46
type: A, layer: 1, pos: 15
type: A, layer: 1, pos: 4
type: B, layer: 1, pos: 13
type: B, layer: 1, pos: 15
type: A, layer: 1, pos: 38
type: B, layer: 1, pos: 11
type: A, layer: 1, pos: 23
type: B, layer: 1, pos: 21
type: B, layer: 1, pos: 16
type: B, layer: 1, pos: 23
type: B, layer: 1, pos: 38
type: A, layer: 1, pos: 16
type: A, layer: 1, pos: 31
type: B, layer: 1, pos: 31
type: A, layer: 1, pos: 42

Time for candidate selection: 0.09 seconds

### Candidate
type: A, layer: 1, pos: 41

### Candidate
type: A, layer: 1, pos: 14

### Candidate
type: A, layer: 1, pos: 19

### Candidate
type: A, layer: 1, pos: 39

### Candidate
type: B, layer: 1, pos: 14

### Candidate
type: B, layer: 1, pos: 12

### Candidate
type: A, layer: 1, pos: 25

### Candidate
type: B, layer: 1, pos: 39

### Candidate
type: B, layer: 1, pos: 19

### Candidate
type: B, layer: 1, pos: 25

### Candidate
type: A, layer: 1, pos: 20

### Candidate
type: B, layer: 1, pos: 41

### Candidate
type: A, layer: 1, pos: 37

### Candidate
type: A, layer: 1, pos: 12

### Candidate
type: B, layer: 1, pos: 37

### Candidate
type: A, layer: 1, pos: 8

### Candidate
type: B, layer: 1, pos: 29

### Candidate
type: B, layer: 1, pos: 8

### Candidate
type: A, layer: 1, pos: 29

### Candidate
type: A, layer: 1, pos: 21

### Candidate
type: B, layer: 1, pos: 20

### Candidate
type: A, layer: 1, pos: 6

### Candidate
type: B, layer: 1, pos: 6

### Candidate
type: A, layer: 1, pos: 11

### Candidate
type: A, layer: 1, pos: 46

### Candidate
type: B, layer: 1, pos: 4

### Candidate
type: B, layer: 1, pos: 46

### Candidate
type: A, layer: 1, pos: 15

### Candidate
type: A, layer: 1, pos: 4

### Candidate
type: B, layer: 1, pos: 13

### Candidate
type: B, layer: 1, pos: 15

### Candidate
type: A, layer: 1, pos: 38

### Candidate
type: B, layer: 1, pos: 11

### Candidate
type: A, layer: 1, pos: 23

### Candidate
type: B, layer: 1, pos: 21

### Candidate
type: B, layer: 1, pos: 16

### Candidate
type: B, layer: 1, pos: 23

### Candidate
type: B, layer: 1, pos: 38

### Candidate
type: A, layer: 1, pos: 16

### Candidate
type: A, layer: 1, pos: 31

## Relational analysis of NS_A2_A1_B1_A2_B2_A1_B2_B1_B1_A1_A1
Optimization infeasible because this subproblem isn't reachable.

### Candidate
type: B, layer: 1, pos: 31

### Candidate
type: A, layer: 1, pos: 42

## Relational analysis of NS_A2_A1_B1_A2_B2_A1_B2_B1_B1_A1_A1

### Relational analysis result of NS_A2_A1_B1_A2_B2_A1_B2_B1_B1_A1_A1
Status: Status.VERIFIED
Output dim: 0, lower bound: -81.8603240, upper bound: 81.8606420
time: 0.97 seconds

## Relational analysis of NS_A2_A1_B1_A2_B2_A1_B2_B1_B1_A1_A2
Optimization infeasible because this subproblem isn't reachable.

## BFS NS instance: NS_A2_A1_B1_A2_B2_A1_B2_B1_B1_A2

### Backsubstitution after applying NS history:
0: -19.1695518, 69.8992538, -22.3502445, 81.6966095, -100.8661575, 92.2494812
1: -50.6409225, 158.3228149, -59.1535072, 186.1858215, -236.8267365, 217.4762878
2: -75.6577301, 140.8316345, -88.7366409, 163.8814545, -239.5391846, 229.5682678
3: -43.6225853, 169.0185852, -50.8682022, 198.9301300, -242.5527191, 219.8867798
4: -69.2205124, 124.0199509, -81.0366821, 144.4750519, -213.6955566, 205.0566406

Time for backsubstitution: 0.99 seconds

### NS candidates at layer 1
type: A, layer: 1, pos: 41
type: A, layer: 1, pos: 14
type: A, layer: 1, pos: 19
type: A, layer: 1, pos: 39
type: B, layer: 1, pos: 14
type: B, layer: 1, pos: 12
type: A, layer: 1, pos: 20
type: B, layer: 1, pos: 19
type: B, layer: 1, pos: 39
type: A, layer: 1, pos: 25
type: B, layer: 1, pos: 25
type: B, layer: 1, pos: 37
type: B, layer: 1, pos: 41
type: B, layer: 1, pos: 29
type: A, layer: 1, pos: 12
type: A, layer: 1, pos: 37
type: A, layer: 1, pos: 8
type: B, layer: 1, pos: 8
type: A, layer: 1, pos: 29
type: A, layer: 1, pos: 21
type: B, layer: 1, pos: 20
type: A, layer: 1, pos: 6
type: B, layer: 1, pos: 6
type: B, layer: 1, pos: 4
type: A, layer: 1, pos: 46
type: B, layer: 1, pos: 46
type: A, layer: 1, pos: 11
type: A, layer: 1, pos: 15
type: B, layer: 1, pos: 13
type: A, layer: 1, pos: 4
type: B, layer: 1, pos: 15
type: B, layer: 1, pos: 38
type: B, layer: 1, pos: 11
type: B, layer: 1, pos: 21
type: A, layer: 1, pos: 38
type: A, layer: 1, pos: 16
type: B, layer: 1, pos: 16
type: A, layer: 1, pos: 23
type: B, layer: 1, pos: 23
type: A, layer: 1, pos: 31
type: B, layer: 1, pos: 31
type: A, layer: 1, pos: 42

Time for candidate selection: 0.08 seconds

### Candidate
type: A, layer: 1, pos: 41

### Candidate
type: A, layer: 1, pos: 14

### Candidate
type: A, layer: 1, pos: 19

### Candidate
type: A, layer: 1, pos: 39

### Candidate
type: B, layer: 1, pos: 14

### Candidate
type: B, layer: 1, pos: 12

### Candidate
type: A, layer: 1, pos: 20

### Candidate
type: B, layer: 1, pos: 19

### Candidate
type: B, layer: 1, pos: 39

### Candidate
type: A, layer: 1, pos: 25

### Candidate
type: B, layer: 1, pos: 25

### Candidate
type: B, layer: 1, pos: 37

### Candidate
type: B, layer: 1, pos: 41

### Candidate
type: B, layer: 1, pos: 29

### Candidate
type: A, layer: 1, pos: 12

### Candidate
type: A, layer: 1, pos: 37

### Candidate
type: A, layer: 1, pos: 8

### Candidate
type: B, layer: 1, pos: 8

### Candidate
type: A, layer: 1, pos: 29

### Candidate
type: A, layer: 1, pos: 21

### Candidate
type: B, layer: 1, pos: 20

### Candidate
type: A, layer: 1, pos: 6

### Candidate
type: B, layer: 1, pos: 6

### Candidate
type: B, layer: 1, pos: 4

### Candidate
type: A, layer: 1, pos: 46

### Candidate
type: B, layer: 1, pos: 46

### Candidate
type: A, layer: 1, pos: 11

### Candidate
type: A, layer: 1, pos: 15

### Candidate
type: B, layer: 1, pos: 13

### Candidate
type: A, layer: 1, pos: 4

### Candidate
type: B, layer: 1, pos: 15

### Candidate
type: B, layer: 1, pos: 38

### Candidate
type: B, layer: 1, pos: 11

### Candidate
type: B, layer: 1, pos: 21

### Candidate
type: A, layer: 1, pos: 38

### Candidate
type: A, layer: 1, pos: 16

### Candidate
type: B, layer: 1, pos: 16

### Candidate
type: A, layer: 1, pos: 23

### Candidate
type: B, layer: 1, pos: 23

### Candidate
type: A, layer: 1, pos: 31

## Relational analysis of NS_A2_A1_B1_A2_B2_A1_B2_B1_B1_A2_A1
Optimization infeasible because this subproblem isn't reachable.

### Candidate
type: B, layer: 1, pos: 31

### Candidate
type: A, layer: 1, pos: 42

## Relational analysis of NS_A2_A1_B1_A2_B2_A1_B2_B1_B1_A2_A1

### Relational analysis result of NS_A2_A1_B1_A2_B2_A1_B2_B1_B1_A2_A1
Status: Status.VERIFIED
Output dim: 0, lower bound: -81.8603240, upper bound: 81.8608257
time: 0.80 seconds

## Relational analysis of NS_A2_A1_B1_A2_B2_A1_B2_B1_B1_A2_A2
Optimization infeasible because this subproblem isn't reachable.

## BFS NS instance: NS_A2_A1_B1_A2_B2_A1_B2_B1_B2_A1

### Backsubstitution after applying NS history:
0: -19.1982937, 70.1187820, -21.9304276, 79.9511108, -99.1494064, 92.0491867
1: -50.9867439, 158.1197510, -58.0220451, 182.0756073, -233.0623474, 216.1417694
2: -76.5202637, 140.7139435, -87.1399994, 160.7810974, -237.3013611, 227.8539276
3: -43.8595810, 168.9619751, -49.9183846, 194.5646057, -238.4241791, 218.8803253
4: -69.9047928, 123.9398575, -79.5788498, 141.7767334, -211.6814880, 203.5187073

Time for backsubstitution: 0.91 seconds

### NS candidates at layer 1
type: A, layer: 1, pos: 41
type: A, layer: 1, pos: 14
type: A, layer: 1, pos: 19
type: A, layer: 1, pos: 39
type: B, layer: 1, pos: 14
type: A, layer: 1, pos: 25
type: B, layer: 1, pos: 12
type: B, layer: 1, pos: 19
type: B, layer: 1, pos: 25
type: B, layer: 1, pos: 39
type: A, layer: 1, pos: 20
type: A, layer: 1, pos: 12
type: A, layer: 1, pos: 8
type: A, layer: 1, pos: 37
type: B, layer: 1, pos: 29
type: B, layer: 1, pos: 37
type: B, layer: 1, pos: 41
type: B, layer: 1, pos: 20
type: B, layer: 1, pos: 8
type: A, layer: 1, pos: 29
type: A, layer: 1, pos: 21
type: A, layer: 1, pos: 6
type: B, layer: 1, pos: 6
type: A, layer: 1, pos: 11
type: B, layer: 1, pos: 4
type: A, layer: 1, pos: 46
type: B, layer: 1, pos: 46
type: B, layer: 1, pos: 15
type: B, layer: 1, pos: 13
type: A, layer: 1, pos: 15
type: A, layer: 1, pos: 4
type: B, layer: 1, pos: 16
type: B, layer: 1, pos: 38
type: A, layer: 1, pos: 23
type: A, layer: 1, pos: 38
type: B, layer: 1, pos: 21
type: B, layer: 1, pos: 11
type: B, layer: 1, pos: 23
type: A, layer: 1, pos: 31
type: B, layer: 1, pos: 31
type: A, layer: 1, pos: 16
type: A, layer: 1, pos: 42

Time for candidate selection: 0.07 seconds

### Candidate
type: A, layer: 1, pos: 41

### Candidate
type: A, layer: 1, pos: 14

### Candidate
type: A, layer: 1, pos: 19

### Candidate
type: A, layer: 1, pos: 39

### Candidate
type: B, layer: 1, pos: 14

### Candidate
type: A, layer: 1, pos: 25

### Candidate
type: B, layer: 1, pos: 12

### Candidate
type: B, layer: 1, pos: 19

### Candidate
type: B, layer: 1, pos: 25

### Candidate
type: B, layer: 1, pos: 39

### Candidate
type: A, layer: 1, pos: 20

### Candidate
type: A, layer: 1, pos: 12

### Candidate
type: A, layer: 1, pos: 8

### Candidate
type: A, layer: 1, pos: 37

### Candidate
type: B, layer: 1, pos: 29

### Candidate
type: B, layer: 1, pos: 37

### Candidate
type: B, layer: 1, pos: 41

### Candidate
type: B, layer: 1, pos: 20

### Candidate
type: B, layer: 1, pos: 8

### Candidate
type: A, layer: 1, pos: 29

### Candidate
type: A, layer: 1, pos: 21

### Candidate
type: A, layer: 1, pos: 6

### Candidate
type: B, layer: 1, pos: 6

### Candidate
type: A, layer: 1, pos: 11

### Candidate
type: B, layer: 1, pos: 4

### Candidate
type: A, layer: 1, pos: 46

### Candidate
type: B, layer: 1, pos: 46

### Candidate
type: B, layer: 1, pos: 15

### Candidate
type: B, layer: 1, pos: 13

### Candidate
type: A, layer: 1, pos: 15

### Candidate
type: A, layer: 1, pos: 4

### Candidate
type: B, layer: 1, pos: 16

### Candidate
type: B, layer: 1, pos: 38

### Candidate
type: A, layer: 1, pos: 23

### Candidate
type: A, layer: 1, pos: 38

### Candidate
type: B, layer: 1, pos: 21

### Candidate
type: B, layer: 1, pos: 11

### Candidate
type: B, layer: 1, pos: 23

### Candidate
type: A, layer: 1, pos: 31

## Relational analysis of NS_A2_A1_B1_A2_B2_A1_B2_B1_B2_A1_A1
Optimization infeasible because this subproblem isn't reachable.

### Candidate
type: B, layer: 1, pos: 31

### Candidate
type: A, layer: 1, pos: 16

### Candidate
type: A, layer: 1, pos: 42

## Relational analysis of NS_A2_A1_B1_A2_B2_A1_B2_B1_B2_A1_A1

### Relational analysis result of NS_A2_A1_B1_A2_B2_A1_B2_B1_B2_A1_A1
Status: Status.VERIFIED
Output dim: 0, lower bound: -81.8603240, upper bound: 81.8606638
time: 0.80 seconds

## Relational analysis of NS_A2_A1_B1_A2_B2_A1_B2_B1_B2_A1_A2

### Relational analysis result of NS_A2_A1_B1_A2_B2_A1_B2_B1_B2_A1_A2
Status: Status.UNKNOWN
Output dim: 0, lower bound: -81.8603240, upper bound: 81.8611676
time: 1.56 seconds

## BFS NS instance: NS_A2_A1_B1_A2_B2_A1_B2_B1_B2_A2

### Backsubstitution after applying NS history:
0: -19.3761635, 70.6416855, -21.9304276, 79.9511108, -99.3272705, 92.5720978
1: -51.1780510, 160.0721283, -58.0220451, 182.0756073, -233.2536621, 218.0941315
2: -76.4557343, 142.3144379, -87.1399994, 160.7810974, -237.2368317, 229.4544220
3: -44.0854263, 170.8686676, -49.9183846, 194.5646057, -238.6500244, 220.7870331
4: -69.9495621, 125.3343735, -79.5788498, 141.7767334, -211.7262726, 204.9131622

Time for backsubstitution: 0.95 seconds

### NS candidates at layer 1
type: A, layer: 1, pos: 41
type: A, layer: 1, pos: 14
type: A, layer: 1, pos: 19
type: A, layer: 1, pos: 39
type: B, layer: 1, pos: 14
type: B, layer: 1, pos: 19
type: B, layer: 1, pos: 12
type: A, layer: 1, pos: 25
type: A, layer: 1, pos: 20
type: B, layer: 1, pos: 39
type: B, layer: 1, pos: 25
type: B, layer: 1, pos: 37
type: B, layer: 1, pos: 29
type: A, layer: 1, pos: 12
type: A, layer: 1, pos: 8
type: A, layer: 1, pos: 37
type: B, layer: 1, pos: 8
type: B, layer: 1, pos: 41
type: B, layer: 1, pos: 20
type: A, layer: 1, pos: 21
type: A, layer: 1, pos: 29
type: B, layer: 1, pos: 6
type: A, layer: 1, pos: 6
type: B, layer: 1, pos: 4
type: A, layer: 1, pos: 11
type: B, layer: 1, pos: 46
type: A, layer: 1, pos: 46
type: A, layer: 1, pos: 15
type: B, layer: 1, pos: 13
type: B, layer: 1, pos: 38
type: B, layer: 1, pos: 15
type: A, layer: 1, pos: 4
type: B, layer: 1, pos: 21
type: B, layer: 1, pos: 11
type: B, layer: 1, pos: 16
type: A, layer: 1, pos: 38
type: A, layer: 1, pos: 23
type: A, layer: 1, pos: 16
type: B, layer: 1, pos: 23
type: A, layer: 1, pos: 31
type: B, layer: 1, pos: 31
type: A, layer: 1, pos: 42

Time for candidate selection: 0.09 seconds

### Candidate
type: A, layer: 1, pos: 41

### Candidate
type: A, layer: 1, pos: 14

### Candidate
type: A, layer: 1, pos: 19

### Candidate
type: A, layer: 1, pos: 39

### Candidate
type: B, layer: 1, pos: 14

### Candidate
type: B, layer: 1, pos: 19

### Candidate
type: B, layer: 1, pos: 12

### Candidate
type: A, layer: 1, pos: 25

### Candidate
type: A, layer: 1, pos: 20

### Candidate
type: B, layer: 1, pos: 39

### Candidate
type: B, layer: 1, pos: 25

### Candidate
type: B, layer: 1, pos: 37

### Candidate
type: B, layer: 1, pos: 29

### Candidate
type: A, layer: 1, pos: 12

### Candidate
type: A, layer: 1, pos: 8

### Candidate
type: A, layer: 1, pos: 37

### Candidate
type: B, layer: 1, pos: 8

### Candidate
type: B, layer: 1, pos: 41

### Candidate
type: B, layer: 1, pos: 20

### Candidate
type: A, layer: 1, pos: 21

### Candidate
type: A, layer: 1, pos: 29

### Candidate
type: B, layer: 1, pos: 6

### Candidate
type: A, layer: 1, pos: 6

### Candidate
type: B, layer: 1, pos: 4

### Candidate
type: A, layer: 1, pos: 11

### Candidate
type: B, layer: 1, pos: 46

### Candidate
type: A, layer: 1, pos: 46

### Candidate
type: A, layer: 1, pos: 15

### Candidate
type: B, layer: 1, pos: 13

### Candidate
type: B, layer: 1, pos: 38

### Candidate
type: B, layer: 1, pos: 15

### Candidate
type: A, layer: 1, pos: 4

### Candidate
type: B, layer: 1, pos: 21

### Candidate
type: B, layer: 1, pos: 11

### Candidate
type: B, layer: 1, pos: 16

### Candidate
type: A, layer: 1, pos: 38

### Candidate
type: A, layer: 1, pos: 23

### Candidate
type: A, layer: 1, pos: 16

### Candidate
type: B, layer: 1, pos: 23

### Candidate
type: A, layer: 1, pos: 31

## Relational analysis of NS_A2_A1_B1_A2_B2_A1_B2_B1_B2_A2_A1
Optimization infeasible because this subproblem isn't reachable.

### Candidate
type: B, layer: 1, pos: 31

### Candidate
type: A, layer: 1, pos: 42

## Relational analysis of NS_A2_A1_B1_A2_B2_A1_B2_B1_B2_A2_A1

### Relational analysis result of NS_A2_A1_B1_A2_B2_A1_B2_B1_B2_A2_A1
Status: Status.VERIFIED
Output dim: 0, lower bound: -81.8603240, upper bound: 81.8608595
time: 1.02 seconds

## Relational analysis of NS_A2_A1_B1_A2_B2_A1_B2_B1_B2_A2_A2

### Relational analysis result of NS_A2_A1_B1_A2_B2_A1_B2_B1_B2_A2_A2
Status: Status.UNKNOWN
Output dim: 0, lower bound: -81.8603240, upper bound: 81.8611676
time: 0.84 seconds

## BFS NS instance: NS_A2_A1_B1_A2_B2_A1_B2_B2_A1_B1

### Backsubstitution after applying NS history:
0: -18.9003239, 69.0304260, -22.0191746, 80.2793198, -99.1796417, 91.0495834
1: -50.1970177, 155.6475983, -57.9374886, 184.0004578, -234.1974792, 213.5850677
2: -75.3415909, 138.5106201, -86.6770935, 161.4652710, -236.8068542, 225.1877136
3: -43.1801605, 166.3397522, -49.8879509, 196.4913177, -239.6714478, 216.2276764
4: -68.8156433, 122.0141144, -79.1761398, 142.4474640, -211.2630615, 201.1902466

Time for backsubstitution: 1.06 seconds

### NS candidates at layer 1
type: A, layer: 1, pos: 41
type: A, layer: 1, pos: 20
type: A, layer: 1, pos: 19
type: A, layer: 1, pos: 39
type: B, layer: 1, pos: 20
type: B, layer: 1, pos: 39
type: A, layer: 1, pos: 14
type: B, layer: 1, pos: 19
type: B, layer: 1, pos: 25
type: B, layer: 1, pos: 41
type: A, layer: 1, pos: 21
type: A, layer: 1, pos: 25
type: B, layer: 1, pos: 23
type: A, layer: 1, pos: 23
type: A, layer: 1, pos: 6
type: B, layer: 1, pos: 14
type: B, layer: 1, pos: 6
type: B, layer: 1, pos: 4
type: A, layer: 1, pos: 46
type: A, layer: 1, pos: 38
type: B, layer: 1, pos: 15
type: B, layer: 1, pos: 16
type: B, layer: 1, pos: 12
type: B, layer: 1, pos: 46
type: B, layer: 1, pos: 21
type: B, layer: 1, pos: 13
type: A, layer: 1, pos: 11
type: A, layer: 1, pos: 15
type: A, layer: 1, pos: 4
type: A, layer: 1, pos: 16
type: A, layer: 1, pos: 29
type: B, layer: 1, pos: 29
type: A, layer: 1, pos: 37
type: B, layer: 1, pos: 38
type: A, layer: 1, pos: 12
type: A, layer: 1, pos: 8
type: A, layer: 1, pos: 31
type: B, layer: 1, pos: 31
type: A, layer: 1, pos: 42
type: B, layer: 1, pos: 37
type: B, layer: 1, pos: 11
type: B, layer: 1, pos: 8

Time for candidate selection: 0.12 seconds

### Candidate
type: A, layer: 1, pos: 41

### Candidate
type: A, layer: 1, pos: 20

### Candidate
type: A, layer: 1, pos: 19

### Candidate
type: A, layer: 1, pos: 39

### Candidate
type: B, layer: 1, pos: 20

### Candidate
type: B, layer: 1, pos: 39

### Candidate
type: A, layer: 1, pos: 14

### Candidate
type: B, layer: 1, pos: 19

### Candidate
type: B, layer: 1, pos: 25

### Candidate
type: B, layer: 1, pos: 41

### Candidate
type: A, layer: 1, pos: 21

### Candidate
type: A, layer: 1, pos: 25

### Candidate
type: B, layer: 1, pos: 23

### Candidate
type: A, layer: 1, pos: 23

### Candidate
type: A, layer: 1, pos: 6

### Candidate
type: B, layer: 1, pos: 14

### Candidate
type: B, layer: 1, pos: 6

### Candidate
type: B, layer: 1, pos: 4

### Candidate
type: A, layer: 1, pos: 46

### Candidate
type: A, layer: 1, pos: 38

### Candidate
type: B, layer: 1, pos: 15

### Candidate
type: B, layer: 1, pos: 16

### Candidate
type: B, layer: 1, pos: 12

### Candidate
type: B, layer: 1, pos: 46

### Candidate
type: B, layer: 1, pos: 21

### Candidate
type: B, layer: 1, pos: 13

### Candidate
type: A, layer: 1, pos: 11

### Candidate
type: A, layer: 1, pos: 15

### Candidate
type: A, layer: 1, pos: 4

### Candidate
type: A, layer: 1, pos: 16

### Candidate
type: A, layer: 1, pos: 29

### Candidate
type: B, layer: 1, pos: 29

### Candidate
type: A, layer: 1, pos: 37

### Candidate
type: B, layer: 1, pos: 38

### Candidate
type: A, layer: 1, pos: 12

### Candidate
type: A, layer: 1, pos: 8

### Candidate
type: A, layer: 1, pos: 31

## Relational analysis of NS_A2_A1_B1_A2_B2_A1_B2_B2_A1_B1_A1
Optimization infeasible because this subproblem isn't reachable.

### Candidate
type: B, layer: 1, pos: 31

### Candidate
type: A, layer: 1, pos: 42

## Relational analysis of NS_A2_A1_B1_A2_B2_A1_B2_B2_A1_B1_A1

### Relational analysis result of NS_A2_A1_B1_A2_B2_A1_B2_B2_A1_B1_A1
Status: Status.VERIFIED
Output dim: 0, lower bound: -81.8603194, upper bound: 81.8606284
time: 0.95 seconds

## Relational analysis of NS_A2_A1_B1_A2_B2_A1_B2_B2_A1_B1_A2

### Relational analysis result of NS_A2_A1_B1_A2_B2_A1_B2_B2_A1_B1_A2
Status: Status.UNKNOWN
Output dim: 0, lower bound: -81.8603194, upper bound: 81.8613420
time: 0.88 seconds

## BFS NS instance: NS_A2_A1_B1_A2_B2_A1_B2_B2_A1_B2

### Backsubstitution after applying NS history:
0: -19.1982937, 70.1187820, -21.6846695, 78.8668900, -98.0651779, 91.8034363
1: -50.9867439, 158.1197510, -57.0523071, 180.6849213, -231.6716614, 215.1720276
2: -76.5202637, 140.7139435, -85.4215088, 158.9122162, -235.4324799, 226.1354523
3: -43.8595810, 168.9619751, -49.1329575, 192.9877472, -236.8473053, 218.0949402
4: -69.9047928, 123.9398575, -78.0352020, 140.2251129, -210.1298981, 201.9750519

Time for backsubstitution: 1.23 seconds

### NS candidates at layer 1
type: A, layer: 1, pos: 41
type: A, layer: 1, pos: 14
type: A, layer: 1, pos: 19
type: A, layer: 1, pos: 39
type: B, layer: 1, pos: 14
type: B, layer: 1, pos: 12
type: B, layer: 1, pos: 19
type: B, layer: 1, pos: 39
type: A, layer: 1, pos: 12
type: A, layer: 1, pos: 8
type: A, layer: 1, pos: 37
type: A, layer: 1, pos: 25
type: B, layer: 1, pos: 29
type: B, layer: 1, pos: 37
type: A, layer: 1, pos: 29
type: A, layer: 1, pos: 21
type: B, layer: 1, pos: 25
type: B, layer: 1, pos: 8
type: A, layer: 1, pos: 20
type: B, layer: 1, pos: 20
type: B, layer: 1, pos: 41
type: B, layer: 1, pos: 15
type: A, layer: 1, pos: 6
type: A, layer: 1, pos: 46
type: B, layer: 1, pos: 6
type: B, layer: 1, pos: 46
type: A, layer: 1, pos: 11
type: B, layer: 1, pos: 21
type: B, layer: 1, pos: 4
type: B, layer: 1, pos: 16
type: A, layer: 1, pos: 15
type: B, layer: 1, pos: 13
type: A, layer: 1, pos: 38
type: B, layer: 1, pos: 38
type: B, layer: 1, pos: 23
type: A, layer: 1, pos: 23
type: A, layer: 1, pos: 4
type: B, layer: 1, pos: 11
type: A, layer: 1, pos: 31
type: B, layer: 1, pos: 31
type: A, layer: 1, pos: 16
type: A, layer: 1, pos: 42

Time for candidate selection: 0.08 seconds

### Candidate
type: A, layer: 1, pos: 41

### Candidate
type: A, layer: 1, pos: 14

### Candidate
type: A, layer: 1, pos: 19

### Candidate
type: A, layer: 1, pos: 39

### Candidate
type: B, layer: 1, pos: 14

### Candidate
type: B, layer: 1, pos: 12

### Candidate
type: B, layer: 1, pos: 19

### Candidate
type: B, layer: 1, pos: 39

### Candidate
type: A, layer: 1, pos: 12

### Candidate
type: A, layer: 1, pos: 8

### Candidate
type: A, layer: 1, pos: 37

### Candidate
type: A, layer: 1, pos: 25

### Candidate
type: B, layer: 1, pos: 29

### Candidate
type: B, layer: 1, pos: 37

### Candidate
type: A, layer: 1, pos: 29

### Candidate
type: A, layer: 1, pos: 21

### Candidate
type: B, layer: 1, pos: 25

### Candidate
type: B, layer: 1, pos: 8

### Candidate
type: A, layer: 1, pos: 20

### Candidate
type: B, layer: 1, pos: 20

### Candidate
type: B, layer: 1, pos: 41

### Candidate
type: B, layer: 1, pos: 15

### Candidate
type: A, layer: 1, pos: 6

### Candidate
type: A, layer: 1, pos: 46

### Candidate
type: B, layer: 1, pos: 6

### Candidate
type: B, layer: 1, pos: 46

### Candidate
type: A, layer: 1, pos: 11

### Candidate
type: B, layer: 1, pos: 21

### Candidate
type: B, layer: 1, pos: 4

### Candidate
type: B, layer: 1, pos: 16

### Candidate
type: A, layer: 1, pos: 15

### Candidate
type: B, layer: 1, pos: 13

### Candidate
type: A, layer: 1, pos: 38

### Candidate
type: B, layer: 1, pos: 38

### Candidate
type: B, layer: 1, pos: 23

### Candidate
type: A, layer: 1, pos: 23

### Candidate
type: A, layer: 1, pos: 4

### Candidate
type: B, layer: 1, pos: 11

### Candidate
type: A, layer: 1, pos: 31

## Relational analysis of NS_A2_A1_B1_A2_B2_A1_B2_B2_A1_B2_A1
Optimization infeasible because this subproblem isn't reachable.

### Candidate
type: B, layer: 1, pos: 31

### Candidate
type: A, layer: 1, pos: 16

### Candidate
type: A, layer: 1, pos: 42

## Relational analysis of NS_A2_A1_B1_A2_B2_A1_B2_B2_A1_B2_A1
Optimization infeasible because this subproblem isn't reachable.

## BFS NS instance: NS_A2_A1_B1_A2_B2_A2_B1_B1_A1_B1

### Backsubstitution after applying NS history:
0: -21.8176060, 79.6616898, -19.8372517, 72.3583832, -94.1759796, 99.4989395
1: -57.5466347, 181.4291687, -52.4216766, 164.9348755, -222.4815063, 233.8508453
2: -86.0756149, 160.5619507, -78.7271881, 145.3789215, -231.4545135, 239.2891388
3: -49.4931564, 193.4996185, -45.0796814, 176.1155243, -225.6086731, 238.5792999
4: -78.6817398, 141.4108582, -71.8689957, 128.1148376, -206.7965698, 213.2798462

Time for backsubstitution: 0.97 seconds

### NS candidates at layer 1
type: A, layer: 1, pos: 41
type: B, layer: 1, pos: 20
type: B, layer: 1, pos: 41
type: A, layer: 1, pos: 19
type: A, layer: 1, pos: 20
type: B, layer: 1, pos: 19
type: B, layer: 1, pos: 25
type: A, layer: 1, pos: 14
type: B, layer: 1, pos: 14
type: A, layer: 1, pos: 25
type: B, layer: 1, pos: 23
type: A, layer: 1, pos: 23
type: B, layer: 1, pos: 6
type: A, layer: 1, pos: 6
type: A, layer: 1, pos: 21
type: B, layer: 1, pos: 21
type: B, layer: 1, pos: 4
type: B, layer: 1, pos: 15
type: B, layer: 1, pos: 46
type: B, layer: 1, pos: 13
type: A, layer: 1, pos: 46
type: A, layer: 1, pos: 38
type: A, layer: 1, pos: 29
type: A, layer: 1, pos: 15
type: B, layer: 1, pos: 11
type: B, layer: 1, pos: 38
type: A, layer: 1, pos: 12
type: B, layer: 1, pos: 12
type: B, layer: 1, pos: 29
type: A, layer: 1, pos: 4
type: B, layer: 1, pos: 42
type: A, layer: 1, pos: 37
type: A, layer: 1, pos: 22
type: A, layer: 1, pos: 8
type: B, layer: 1, pos: 37
type: B, layer: 1, pos: 31
type: A, layer: 1, pos: 42
type: A, layer: 1, pos: 31
type: A, layer: 1, pos: 16
type: A, layer: 1, pos: 11
type: B, layer: 1, pos: 8
type: B, layer: 1, pos: 22

Time for candidate selection: 0.08 seconds

### Candidate
type: A, layer: 1, pos: 41

### Candidate
type: B, layer: 1, pos: 20

### Candidate
type: B, layer: 1, pos: 41

### Candidate
type: A, layer: 1, pos: 19

### Candidate
type: A, layer: 1, pos: 20

### Candidate
type: B, layer: 1, pos: 19

### Candidate
type: B, layer: 1, pos: 25

### Candidate
type: A, layer: 1, pos: 14

### Candidate
type: B, layer: 1, pos: 14

### Candidate
type: A, layer: 1, pos: 25

### Candidate
type: B, layer: 1, pos: 23

## Relational analysis of NS_A2_A1_B1_A2_B2_A2_B1_B1_A1_B1_B1

### Relational analysis result of NS_A2_A1_B1_A2_B2_A2_B1_B1_A1_B1_B1
Status: Status.UNKNOWN
Output dim: 0, lower bound: -81.8629550, upper bound: 81.8630840
time: 0.88 seconds

## Relational analysis of NS_A2_A1_B1_A2_B2_A2_B1_B1_A1_B1_B2

### Relational analysis result of NS_A2_A1_B1_A2_B2_A2_B1_B1_A1_B1_B2
Status: Status.UNKNOWN
Output dim: 0, lower bound: -81.8629727, upper bound: 81.8630885
time: 1.12 seconds

## BFS NS instance: NS_A2_A1_B1_A2_B2_A2_B1_B1_A1_B2

### Backsubstitution after applying NS history:
0: -21.8176060, 79.6616898, -23.5648251, 85.7438583, -107.5614624, 103.2265167
1: -57.5466347, 181.4291687, -62.1873856, 194.8871918, -252.4338226, 243.6165466
2: -86.0756149, 160.5619507, -93.6776581, 172.6041718, -258.6797791, 254.2396088
3: -49.4931564, 193.4996185, -53.5600662, 208.0747833, -257.5679321, 247.0596924
4: -78.6817398, 141.4108582, -85.5229263, 152.1083221, -230.7900391, 226.9337769

Time for backsubstitution: 0.95 seconds

### NS candidates at layer 1
type: A, layer: 1, pos: 41
type: A, layer: 1, pos: 20
type: B, layer: 1, pos: 20
type: A, layer: 1, pos: 19
type: B, layer: 1, pos: 19
type: B, layer: 1, pos: 41
type: A, layer: 1, pos: 25
type: B, layer: 1, pos: 23
type: A, layer: 1, pos: 14
type: A, layer: 1, pos: 21
type: B, layer: 1, pos: 14
type: B, layer: 1, pos: 25
type: A, layer: 1, pos: 6
type: B, layer: 1, pos: 6
type: A, layer: 1, pos: 23
type: B, layer: 1, pos: 15
type: A, layer: 1, pos: 4
type: A, layer: 1, pos: 46
type: A, layer: 1, pos: 29
type: B, layer: 1, pos: 46
type: A, layer: 1, pos: 12
type: B, layer: 1, pos: 13
type: B, layer: 1, pos: 38
type: A, layer: 1, pos: 15
type: A, layer: 1, pos: 8
type: B, layer: 1, pos: 4
type: A, layer: 1, pos: 38
type: A, layer: 1, pos: 37
type: B, layer: 1, pos: 42
type: B, layer: 1, pos: 11
type: B, layer: 1, pos: 21
type: B, layer: 1, pos: 29
type: A, layer: 1, pos: 22
type: B, layer: 1, pos: 12
type: A, layer: 1, pos: 16
type: A, layer: 1, pos: 11
type: A, layer: 1, pos: 31
type: B, layer: 1, pos: 31
type: B, layer: 1, pos: 37
type: A, layer: 1, pos: 42
type: B, layer: 1, pos: 22
type: B, layer: 1, pos: 8

Time for candidate selection: 0.08 seconds

### Candidate
type: A, layer: 1, pos: 41

### Candidate
type: A, layer: 1, pos: 20

### Candidate
type: B, layer: 1, pos: 20

### Candidate
type: A, layer: 1, pos: 19

### Candidate
type: B, layer: 1, pos: 19

### Candidate
type: B, layer: 1, pos: 41

### Candidate
type: A, layer: 1, pos: 25

### Candidate
type: B, layer: 1, pos: 23

## Relational analysis of NS_A2_A1_B1_A2_B2_A2_B1_B1_A1_B2_B1

### Relational analysis result of NS_A2_A1_B1_A2_B2_A2_B1_B1_A1_B2_B1
Status: Status.UNKNOWN
Output dim: 0, lower bound: -81.8629550, upper bound: 81.8630867
time: 0.83 seconds

## Relational analysis of NS_A2_A1_B1_A2_B2_A2_B1_B1_A1_B2_B2

### Relational analysis result of NS_A2_A1_B1_A2_B2_A2_B1_B1_A1_B2_B2
Status: Status.UNKNOWN
Output dim: 0, lower bound: -81.8629727, upper bound: 81.8630885
time: 1.43 seconds

## BFS NS instance: NS_A2_A1_B1_A2_B2_A2_B1_B1_A2_B1

### Backsubstitution after applying NS history:
0: -25.4717045, 92.7767258, -19.8372517, 72.3583832, -97.8300858, 112.6139755
1: -67.0863876, 210.8640747, -52.4216766, 164.9348755, -232.0212708, 263.2857666
2: -100.6374817, 187.2869263, -78.7271881, 145.3789215, -246.0163879, 266.0140686
3: -57.7826309, 224.8646393, -45.0796814, 176.1155243, -233.8981476, 269.9443359
4: -91.9857330, 164.9750671, -71.8689957, 128.1148376, -220.1005707, 236.8440552

Time for backsubstitution: 1.00 seconds

### NS candidates at layer 1
type: B, layer: 1, pos: 41
type: B, layer: 1, pos: 20
type: A, layer: 1, pos: 19
type: A, layer: 1, pos: 41
type: B, layer: 1, pos: 25
type: A, layer: 1, pos: 20
type: B, layer: 1, pos: 19
type: A, layer: 1, pos: 23
type: B, layer: 1, pos: 14
type: A, layer: 1, pos: 14
type: B, layer: 1, pos: 21
type: B, layer: 1, pos: 6
type: A, layer: 1, pos: 6
type: B, layer: 1, pos: 4
type: B, layer: 1, pos: 23
type: A, layer: 1, pos: 25
type: B, layer: 1, pos: 46
type: B, layer: 1, pos: 15
type: B, layer: 1, pos: 13
type: A, layer: 1, pos: 38
type: B, layer: 1, pos: 12
type: B, layer: 1, pos: 11
type: A, layer: 1, pos: 46
type: A, layer: 1, pos: 15
type: B, layer: 1, pos: 29
type: A, layer: 1, pos: 29
type: A, layer: 1, pos: 21
type: B, layer: 1, pos: 38
type: A, layer: 1, pos: 42
type: A, layer: 1, pos: 12
type: B, layer: 1, pos: 37
type: B, layer: 1, pos: 8
type: A, layer: 1, pos: 22
type: A, layer: 1, pos: 37
type: B, layer: 1, pos: 31
type: B, layer: 1, pos: 42
type: A, layer: 1, pos: 31
type: A, layer: 1, pos: 16
type: A, layer: 1, pos: 4
type: A, layer: 1, pos: 8
type: B, layer: 1, pos: 22
type: A, layer: 1, pos: 11

Time for candidate selection: 0.08 seconds

### Candidate
type: B, layer: 1, pos: 41

### Candidate
type: B, layer: 1, pos: 20

### Candidate
type: A, layer: 1, pos: 19

### Candidate
type: A, layer: 1, pos: 41

### Candidate
type: B, layer: 1, pos: 25

### Candidate
type: A, layer: 1, pos: 20

### Candidate
type: B, layer: 1, pos: 19

### Candidate
type: A, layer: 1, pos: 23

## Relational analysis of NS_A2_A1_B1_A2_B2_A2_B1_B1_A2_B1_A1

### Relational analysis result of NS_A2_A1_B1_A2_B2_A2_B1_B1_A2_B1_A1
Status: Status.UNKNOWN
Output dim: 0, lower bound: -81.8630763, upper bound: 81.8631075
time: 0.78 seconds

## Relational analysis of NS_A2_A1_B1_A2_B2_A2_B1_B1_A2_B1_A2
Optimization infeasible because this subproblem isn't reachable.

### Candidate
type: B, layer: 1, pos: 14

### Candidate
type: A, layer: 1, pos: 14

### Candidate
type: B, layer: 1, pos: 21

### Candidate
type: B, layer: 1, pos: 6

### Candidate
type: A, layer: 1, pos: 6

### Candidate
type: B, layer: 1, pos: 4

### Candidate
type: B, layer: 1, pos: 23

### Candidate
type: A, layer: 1, pos: 25

### Candidate
type: B, layer: 1, pos: 46

### Candidate
type: B, layer: 1, pos: 15

### Candidate
type: B, layer: 1, pos: 13

### Candidate
type: A, layer: 1, pos: 38

### Candidate
type: B, layer: 1, pos: 12

### Candidate
type: B, layer: 1, pos: 11

### Candidate
type: A, layer: 1, pos: 46

### Candidate
type: A, layer: 1, pos: 15

### Candidate
type: B, layer: 1, pos: 29

### Candidate
type: A, layer: 1, pos: 29

### Candidate
type: A, layer: 1, pos: 21

### Candidate
type: B, layer: 1, pos: 38

### Candidate
type: A, layer: 1, pos: 42

## Relational analysis of NS_A2_A1_B1_A2_B2_A2_B1_B1_A2_B1_A1
Optimization infeasible because this subproblem isn't reachable.

### Candidate
type: A, layer: 1, pos: 12

### Candidate
type: B, layer: 1, pos: 37

## Relational analysis of NS_A2_A1_B1_A2_B2_A2_B1_B1_A2_B1_B1
Optimization infeasible because this subproblem isn't reachable.

### Candidate
type: B, layer: 1, pos: 8

## Relational analysis of NS_A2_A1_B1_A2_B2_A2_B1_B1_A2_B1_B1

### Relational analysis result of NS_A2_A1_B1_A2_B2_A2_B1_B1_A2_B1_B1
Status: Status.UNKNOWN
Output dim: 0, lower bound: -81.8627345, upper bound: 81.8627360
time: 1.02 seconds

## Relational analysis of NS_A2_A1_B1_A2_B2_A2_B1_B1_A2_B1_B2

### Relational analysis result of NS_A2_A1_B1_A2_B2_A2_B1_B1_A2_B1_B2
Status: Status.UNKNOWN
Output dim: 0, lower bound: -81.8620061, upper bound: 81.8622308
time: 0.90 seconds

## BFS NS instance: NS_A2_A1_B1_A2_B2_A2_B1_B1_A2_B2

### Backsubstitution after applying NS history:
0: -25.4717045, 92.7767258, -23.5663586, 85.7493134, -111.2210159, 116.3430862
1: -67.0863876, 210.8640747, -62.1913643, 194.8991852, -261.9855347, 273.0554199
2: -100.6374817, 187.2869263, -93.6835403, 172.6158600, -273.2533264, 280.9703979
3: -57.7826309, 224.8646393, -53.5635490, 208.0873413, -265.8699646, 278.4281616
4: -91.9857330, 164.9750671, -85.5283432, 152.1186066, -244.1043396, 250.5034180

Time for backsubstitution: 1.03 seconds

### NS candidates at layer 1
type: B, layer: 1, pos: 41
type: B, layer: 1, pos: 20
type: A, layer: 1, pos: 41
type: A, layer: 1, pos: 20
type: A, layer: 1, pos: 19
type: B, layer: 1, pos: 19
type: B, layer: 1, pos: 25
type: B, layer: 1, pos: 14
type: A, layer: 1, pos: 14
type: A, layer: 1, pos: 25
type: B, layer: 1, pos: 23
type: A, layer: 1, pos: 23
type: B, layer: 1, pos: 6
type: A, layer: 1, pos: 6
type: A, layer: 1, pos: 21
type: B, layer: 1, pos: 21
type: B, layer: 1, pos: 15
type: B, layer: 1, pos: 4
type: B, layer: 1, pos: 46
type: A, layer: 1, pos: 46
type: B, layer: 1, pos: 13
type: A, layer: 1, pos: 38
type: A, layer: 1, pos: 29
type: A, layer: 1, pos: 15
type: A, layer: 1, pos: 12
type: B, layer: 1, pos: 11
type: B, layer: 1, pos: 12
type: B, layer: 1, pos: 38
type: A, layer: 1, pos: 4
type: B, layer: 1, pos: 29
type: A, layer: 1, pos: 37
type: B, layer: 1, pos: 42
type: A, layer: 1, pos: 22
type: A, layer: 1, pos: 8
type: A, layer: 1, pos: 42
type: B, layer: 1, pos: 37
type: B, layer: 1, pos: 31
type: A, layer: 1, pos: 16
type: A, layer: 1, pos: 31
type: A, layer: 1, pos: 11
type: B, layer: 1, pos: 8
type: B, layer: 1, pos: 22

Time for candidate selection: 0.08 seconds

### Candidate
type: B, layer: 1, pos: 41

### Candidate
type: B, layer: 1, pos: 20

### Candidate
type: A, layer: 1, pos: 41

### Candidate
type: A, layer: 1, pos: 20

### Candidate
type: A, layer: 1, pos: 19

### Candidate
type: B, layer: 1, pos: 19

### Candidate
type: B, layer: 1, pos: 25

### Candidate
type: B, layer: 1, pos: 14

### Candidate
type: A, layer: 1, pos: 14

### Candidate
type: A, layer: 1, pos: 25

### Candidate
type: B, layer: 1, pos: 23

## Relational analysis of NS_A2_A1_B1_A2_B2_A2_B1_B1_A2_B2_B1
Optimization infeasible because this subproblem isn't reachable.

### Candidate
type: A, layer: 1, pos: 23

### Candidate
type: B, layer: 1, pos: 6

### Candidate
type: A, layer: 1, pos: 6

### Candidate
type: A, layer: 1, pos: 21

### Candidate
type: B, layer: 1, pos: 21

### Candidate
type: B, layer: 1, pos: 15

### Candidate
type: B, layer: 1, pos: 4

### Candidate
type: B, layer: 1, pos: 46

### Candidate
type: A, layer: 1, pos: 46

### Candidate
type: B, layer: 1, pos: 13

### Candidate
type: A, layer: 1, pos: 38

### Candidate
type: A, layer: 1, pos: 29

### Candidate
type: A, layer: 1, pos: 15

### Candidate
type: A, layer: 1, pos: 12

### Candidate
type: B, layer: 1, pos: 11

### Candidate
type: B, layer: 1, pos: 12

### Candidate
type: B, layer: 1, pos: 38

### Candidate
type: A, layer: 1, pos: 4

### Candidate
type: B, layer: 1, pos: 29

### Candidate
type: A, layer: 1, pos: 37

## Relational analysis of NS_A2_A1_B1_A2_B2_A2_B1_B1_A2_B2_A1

### Relational analysis result of NS_A2_A1_B1_A2_B2_A2_B1_B1_A2_B2_A1
Status: Status.UNKNOWN
Output dim: 0, lower bound: -81.8628029, upper bound: 81.8629873
time: 1.13 seconds

## Relational analysis of NS_A2_A1_B1_A2_B2_A2_B1_B1_A2_B2_A2
Optimization infeasible because this subproblem isn't reachable.

### Candidate
type: B, layer: 1, pos: 42

## Relational analysis of NS_A2_A1_B1_A2_B2_A2_B1_B1_A2_B2_B1
Optimization infeasible because this subproblem isn't reachable.

### Candidate
type: A, layer: 1, pos: 22

## Relational analysis of NS_A2_A1_B1_A2_B2_A2_B1_B1_A2_B2_A1
Optimization infeasible because this subproblem isn't reachable.

### Candidate
type: A, layer: 1, pos: 8

## Relational analysis of NS_A2_A1_B1_A2_B2_A2_B1_B1_A2_B2_A1

### Relational analysis result of NS_A2_A1_B1_A2_B2_A2_B1_B1_A2_B2_A1
Status: Status.UNKNOWN
Output dim: 0, lower bound: -81.8620332, upper bound: 81.8626166
time: 0.96 seconds

## Relational analysis of NS_A2_A1_B1_A2_B2_A2_B1_B1_A2_B2_A2

### Relational analysis result of NS_A2_A1_B1_A2_B2_A2_B1_B1_A2_B2_A2
Status: Status.UNKNOWN
Output dim: 0, lower bound: -81.8620061, upper bound: 81.8624370
time: 0.98 seconds

## BFS NS instance: NS_A2_A1_B1_A2_B2_A2_B1_B2_A1_B1

### Backsubstitution after applying NS history:
0: -21.5657101, 78.7337189, -21.4209633, 78.1690826, -99.7347946, 100.1546783
1: -56.9098625, 179.3781738, -56.6007881, 178.1797485, -235.0896149, 235.9789429
2: -85.0303497, 158.6518860, -84.9844589, 157.0603333, -242.0906677, 243.6363373
3: -48.9293861, 191.2919006, -48.6832047, 190.2256317, -239.1549988, 239.9750824
4: -77.7424393, 139.7112885, -77.5880814, 138.4752502, -216.2176819, 217.2993774

Time for backsubstitution: 1.15 seconds

### NS candidates at layer 1
type: A, layer: 1, pos: 41
type: B, layer: 1, pos: 41
type: B, layer: 1, pos: 19
type: A, layer: 1, pos: 19
type: A, layer: 1, pos: 6
type: B, layer: 1, pos: 6
type: B, layer: 1, pos: 25
type: A, layer: 1, pos: 25
type: A, layer: 1, pos: 21
type: B, layer: 1, pos: 21
type: A, layer: 1, pos: 46
type: B, layer: 1, pos: 46
type: B, layer: 1, pos: 4
type: B, layer: 1, pos: 38
type: A, layer: 1, pos: 38
type: A, layer: 1, pos: 4
type: A, layer: 1, pos: 12
type: A, layer: 1, pos: 29
type: B, layer: 1, pos: 13
type: B, layer: 1, pos: 14
type: B, layer: 1, pos: 12
type: A, layer: 1, pos: 14
type: A, layer: 1, pos: 11
type: B, layer: 1, pos: 29
type: A, layer: 1, pos: 39
type: A, layer: 1, pos: 8
type: A, layer: 1, pos: 31
type: B, layer: 1, pos: 31
type: A, layer: 1, pos: 15
type: B, layer: 1, pos: 20
type: B, layer: 1, pos: 23
type: A, layer: 1, pos: 20
type: B, layer: 1, pos: 15
type: B, layer: 1, pos: 39
type: B, layer: 1, pos: 11
type: B, layer: 1, pos: 42
type: A, layer: 1, pos: 37
type: B, layer: 1, pos: 22
type: A, layer: 1, pos: 22
type: A, layer: 1, pos: 23
type: B, layer: 1, pos: 37
type: A, layer: 1, pos: 42

Time for candidate selection: 0.08 seconds

### Candidate
type: A, layer: 1, pos: 41

### Candidate
type: B, layer: 1, pos: 41

### Candidate
type: B, layer: 1, pos: 19

### Candidate
type: A, layer: 1, pos: 19

### Candidate
type: A, layer: 1, pos: 6

### Candidate
type: B, layer: 1, pos: 6

### Candidate
type: B, layer: 1, pos: 25

### Candidate
type: A, layer: 1, pos: 25

### Candidate
type: A, layer: 1, pos: 21

### Candidate
type: B, layer: 1, pos: 21

### Candidate
type: A, layer: 1, pos: 46

### Candidate
type: B, layer: 1, pos: 46

### Candidate
type: B, layer: 1, pos: 4

### Candidate
type: B, layer: 1, pos: 38

### Candidate
type: A, layer: 1, pos: 38

### Candidate
type: A, layer: 1, pos: 4

### Candidate
type: A, layer: 1, pos: 12

### Candidate
type: A, layer: 1, pos: 29

### Candidate
type: B, layer: 1, pos: 13

### Candidate
type: B, layer: 1, pos: 14

### Candidate
type: B, layer: 1, pos: 12

### Candidate
type: A, layer: 1, pos: 14

### Candidate
type: A, layer: 1, pos: 11

### Candidate
type: B, layer: 1, pos: 29

### Candidate
type: A, layer: 1, pos: 39

### Candidate
type: A, layer: 1, pos: 8

## Relational analysis of NS_A2_A1_B1_A2_B2_A2_B1_B2_A1_B1_A1

### Relational analysis result of NS_A2_A1_B1_A2_B2_A2_B1_B2_A1_B1_A1
Status: Status.UNKNOWN
Output dim: 0, lower bound: -81.8623854, upper bound: 81.8624728
time: 1.32 seconds

## Relational analysis of NS_A2_A1_B1_A2_B2_A2_B1_B2_A1_B1_A2

### Relational analysis result of NS_A2_A1_B1_A2_B2_A2_B1_B2_A1_B1_A2
Status: Status.UNKNOWN
Output dim: 0, lower bound: -81.8623854, upper bound: 81.8624728
time: 1.44 seconds

## BFS NS instance: NS_A2_A1_B1_A2_B2_A2_B1_B2_A1_B2

### Backsubstitution after applying NS history:
0: -21.1879272, 77.3554916, -21.8043365, 79.5688095, -100.7567368, 99.1598129
1: -55.9140129, 176.2540436, -57.5986404, 181.3307495, -237.2447662, 233.8526764
2: -83.5615540, 155.8548737, -86.3956757, 159.9920044, -243.5535431, 242.2505493
3: -48.0784798, 187.9532928, -49.5589447, 193.5500641, -241.6285095, 237.5122070
4: -76.3956604, 137.2661896, -78.9043732, 141.0879822, -217.4836426, 216.1705627

Time for backsubstitution: 1.00 seconds

### NS candidates at layer 1
type: A, layer: 1, pos: 41
type: B, layer: 1, pos: 19
type: A, layer: 1, pos: 19
type: B, layer: 1, pos: 6
type: A, layer: 1, pos: 6
type: B, layer: 1, pos: 41
type: A, layer: 1, pos: 21
type: A, layer: 1, pos: 25
type: A, layer: 1, pos: 46
type: B, layer: 1, pos: 25
type: A, layer: 1, pos: 8
type: A, layer: 1, pos: 4
type: B, layer: 1, pos: 38
type: A, layer: 1, pos: 38
type: A, layer: 1, pos: 11
type: B, layer: 1, pos: 46
type: A, layer: 1, pos: 14
type: A, layer: 1, pos: 39
type: A, layer: 1, pos: 12
type: B, layer: 1, pos: 21
type: B, layer: 1, pos: 4
type: A, layer: 1, pos: 29
type: B, layer: 1, pos: 13
type: B, layer: 1, pos: 12
type: A, layer: 1, pos: 20
type: B, layer: 1, pos: 29
type: B, layer: 1, pos: 14
type: A, layer: 1, pos: 31
type: B, layer: 1, pos: 31
type: B, layer: 1, pos: 23
type: B, layer: 1, pos: 15
type: A, layer: 1, pos: 37
type: A, layer: 1, pos: 15
type: B, layer: 1, pos: 42
type: A, layer: 1, pos: 22
type: B, layer: 1, pos: 20
type: B, layer: 1, pos: 11
type: B, layer: 1, pos: 39
type: B, layer: 1, pos: 22
type: B, layer: 1, pos: 37
type: A, layer: 1, pos: 42
type: A, layer: 1, pos: 23

Time for candidate selection: 0.08 seconds

### Candidate
type: A, layer: 1, pos: 41

### Candidate
type: B, layer: 1, pos: 19

### Candidate
type: A, layer: 1, pos: 19

### Candidate
type: B, layer: 1, pos: 6

### Candidate
type: A, layer: 1, pos: 6

### Candidate
type: B, layer: 1, pos: 41

### Candidate
type: A, layer: 1, pos: 21

### Candidate
type: A, layer: 1, pos: 25

### Candidate
type: A, layer: 1, pos: 46

### Candidate
type: B, layer: 1, pos: 25

### Candidate
type: A, layer: 1, pos: 8

## Relational analysis of NS_A2_A1_B1_A2_B2_A2_B1_B2_A1_B2_A1

### Relational analysis result of NS_A2_A1_B1_A2_B2_A2_B1_B2_A1_B2_A1
Status: Status.UNKNOWN
Output dim: 0, lower bound: -81.8623854, upper bound: 81.8624728
time: 1.27 seconds

## Relational analysis of NS_A2_A1_B1_A2_B2_A2_B1_B2_A1_B2_A2

### Relational analysis result of NS_A2_A1_B1_A2_B2_A2_B1_B2_A1_B2_A2
Status: Status.UNKNOWN
Output dim: 0, lower bound: -81.8623854, upper bound: 81.8624728
time: 0.89 seconds

## BFS NS instance: NS_A2_A1_B1_A2_B2_A2_B1_B2_A2_A1

### Backsubstitution after applying NS history:
0: -22.1734371, 80.9674911, -21.7754898, 79.4776306, -101.6510620, 102.7429657
1: -58.4890480, 184.4613800, -57.5375023, 181.1314850, -239.6204987, 241.9988861
2: -87.4751282, 163.1868286, -86.3481827, 159.7120819, -247.1871796, 249.5350037
3: -50.3009987, 196.7294006, -49.4804726, 193.3454590, -243.6464539, 246.2098694
4: -79.9599762, 143.7593842, -78.8416901, 140.8039246, -220.7639008, 222.6010742

Time for backsubstitution: 1.18 seconds

### NS candidates at layer 1
type: B, layer: 1, pos: 41
type: A, layer: 1, pos: 41
type: B, layer: 1, pos: 19
type: A, layer: 1, pos: 19
type: A, layer: 1, pos: 6
type: B, layer: 1, pos: 6
type: B, layer: 1, pos: 21
type: B, layer: 1, pos: 25
type: A, layer: 1, pos: 25
type: A, layer: 1, pos: 46
type: A, layer: 1, pos: 21
type: B, layer: 1, pos: 46
type: B, layer: 1, pos: 4
type: A, layer: 1, pos: 38
type: B, layer: 1, pos: 38
type: A, layer: 1, pos: 29
type: A, layer: 1, pos: 4
type: A, layer: 1, pos: 12
type: B, layer: 1, pos: 13
type: B, layer: 1, pos: 12
type: B, layer: 1, pos: 14
type: A, layer: 1, pos: 14
type: B, layer: 1, pos: 39
type: B, layer: 1, pos: 20
type: B, layer: 1, pos: 29
type: A, layer: 1, pos: 31
type: B, layer: 1, pos: 31
type: B, layer: 1, pos: 8
type: B, layer: 1, pos: 11
type: A, layer: 1, pos: 11
type: B, layer: 1, pos: 15
type: A, layer: 1, pos: 15
type: A, layer: 1, pos: 39
type: A, layer: 1, pos: 23
type: B, layer: 1, pos: 23
type: A, layer: 1, pos: 20
type: B, layer: 1, pos: 22
type: A, layer: 1, pos: 42
type: A, layer: 1, pos: 37
type: A, layer: 1, pos: 22
type: B, layer: 1, pos: 37
type: B, layer: 1, pos: 42

Time for candidate selection: 0.10 seconds

### Candidate
type: B, layer: 1, pos: 41

### Candidate
type: A, layer: 1, pos: 41

### Candidate
type: B, layer: 1, pos: 19

### Candidate
type: A, layer: 1, pos: 19

### Candidate
type: A, layer: 1, pos: 6

### Candidate
type: B, layer: 1, pos: 6

### Candidate
type: B, layer: 1, pos: 21

### Candidate
type: B, layer: 1, pos: 25

### Candidate
type: A, layer: 1, pos: 25

### Candidate
type: A, layer: 1, pos: 46

### Candidate
type: A, layer: 1, pos: 21

### Candidate
type: B, layer: 1, pos: 46

### Candidate
type: B, layer: 1, pos: 4

### Candidate
type: A, layer: 1, pos: 38

### Candidate
type: B, layer: 1, pos: 38

### Candidate
type: A, layer: 1, pos: 29

### Candidate
type: A, layer: 1, pos: 4

### Candidate
type: A, layer: 1, pos: 12

### Candidate
type: B, layer: 1, pos: 13

### Candidate
type: B, layer: 1, pos: 12

### Candidate
type: B, layer: 1, pos: 14

### Candidate
type: A, layer: 1, pos: 14

### Candidate
type: B, layer: 1, pos: 39

### Candidate
type: B, layer: 1, pos: 20

### Candidate
type: B, layer: 1, pos: 29

### Candidate
type: A, layer: 1, pos: 31

### Candidate
type: B, layer: 1, pos: 31

### Candidate
type: B, layer: 1, pos: 8

## Relational analysis of NS_A2_A1_B1_A2_B2_A2_B1_B2_A2_A1_B1

### Relational analysis result of NS_A2_A1_B1_A2_B2_A2_B1_B2_A2_A1_B1
Status: Status.UNKNOWN
Output dim: 0, lower bound: -81.8623963, upper bound: 81.8625689
time: 1.19 seconds

## Relational analysis of NS_A2_A1_B1_A2_B2_A2_B1_B2_A2_A1_B2

### Relational analysis result of NS_A2_A1_B1_A2_B2_A2_B1_B2_A2_A1_B2
Status: Status.UNKNOWN
Output dim: 0, lower bound: -81.8623963, upper bound: 81.8625689
time: 1.05 seconds

## NS Result
status: Status.UNKNOWN
execution time: (base) + (ns) = 4.86 + 415.93 = 420.80 seconds
