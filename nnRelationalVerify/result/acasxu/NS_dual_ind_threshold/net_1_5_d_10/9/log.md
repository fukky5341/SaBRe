## Execution arguments:
Dataset: Dataset.ACAS
Network: ds/onnx/acasxu_op11/ACASXU_1_5.onnx
Relational property: GLOBAL_ROBUSTNESS
LP Analysis: True
Epsilon: None
Delta epsilon: 0.1
execution index: (10, None, 9)
Time budget: 420 seconds
Split limit: 100
Threshold: 27.7691976323


## IAR start

### BASE IAR bounds
Layer (inp1_lb, inp1_ub, inp2_lb, inp2_ub, d_lb, d_ub)
0: (-6.9769545, 23.4826069, -6.9769545, 23.4826069, -30.4595604, 30.4595604)
1: (-11.4056101, 24.0275745, -11.4056101, 24.0275745, -35.4331856, 35.4331856)
2: (-9.3722210, 25.8551235, -9.3722210, 25.8551235, -35.2273369, 35.2273369)
3: (-10.1196079, 35.4739799, -10.1196079, 35.4739799, -45.5935898, 45.5935898)
4: (-8.2898979, 33.6252289, -8.2898979, 33.6252289, -41.9151268, 41.9151268)

## BASE Relational Analysis

## BASE Result
execution time: IAR + RelationalAnalysis = 0.80 + 1.60 = 2.40 seconds
status: Status.UNKNOWN
relational distance
Output dim: 0, lower bound: -27.8527559, upper bound: 27.8527559

# Neuron Split (NS) starts

## BFS NS instance: NS

Time for backsubstitution: 0.01 seconds

### NS candidates at layer 1
type: A, layer: 1, pos: 45
type: A, layer: 1, pos: 14
type: A, layer: 1, pos: 13
type: A, layer: 1, pos: 16
type: A, layer: 1, pos: 22
type: A, layer: 1, pos: 0
type: A, layer: 1, pos: 9
type: A, layer: 1, pos: 10
type: A, layer: 1, pos: 41
type: A, layer: 1, pos: 21
type: A, layer: 1, pos: 36
type: A, layer: 1, pos: 17

Time for candidate selection: 0.05 seconds

### Candidate
type: A, layer: 1, pos: 45

## Relational analysis of NS_A1

### Relational analysis result of NS_A1
Status: Status.UNKNOWN
Output dim: 0, lower bound: -27.8498972, upper bound: 27.8472336
time: 0.62 seconds

## Relational analysis of NS_A2

### Relational analysis result of NS_A2
Status: Status.UNKNOWN
Output dim: 0, lower bound: -27.8466200, upper bound: 27.8466200
time: 1.10 seconds

## Summary of splitting at layer (split count: 0)
- Time for NS candidates: 1.79 seconds
NS_A1, status: Status.UNKNOWN, split count: 1, time: 1.79
Output dim: 0, lower bound: -27.8498972, upper bound: 27.8472336
NS_A2, status: Status.UNKNOWN, split count: 1, time: 1.79
Output dim: 0, lower bound: -27.8466200, upper bound: 27.8466200

## BFS NS instance: NS_A1

### Backsubstitution after applying NS history:
0: -4.8150167, 17.2120781, -6.1745024, 21.1960850, -26.0110989, 23.3865814
1: -7.9723043, 17.4862099, -10.1439810, 21.6423035, -29.6146088, 27.6301918
2: -6.4574275, 18.9246101, -8.3049488, 23.3549442, -29.8123722, 27.2295589
3: -7.0645366, 26.1020927, -9.0087423, 32.0837708, -39.1483078, 35.1108360
4: -5.6939859, 24.4059410, -7.3409567, 30.3513966, -36.0453835, 31.7468891

Time for backsubstitution: 0.70 seconds

### NS candidates at layer 1
type: B, layer: 1, pos: 14
type: B, layer: 1, pos: 13
type: B, layer: 1, pos: 45
type: B, layer: 1, pos: 16
type: B, layer: 1, pos: 22
type: B, layer: 1, pos: 9
type: B, layer: 1, pos: 10
type: B, layer: 1, pos: 0
type: B, layer: 1, pos: 21
type: B, layer: 1, pos: 41
type: B, layer: 1, pos: 17
type: B, layer: 1, pos: 36

Time for candidate selection: 0.05 seconds

### Candidate
type: B, layer: 1, pos: 14

## Relational analysis of NS_A1_B1

### Relational analysis result of NS_A1_B1
Status: Status.UNKNOWN
Output dim: 0, lower bound: -27.8118463, upper bound: 27.8079184
time: 0.74 seconds

## Relational analysis of NS_A1_B2

### Relational analysis result of NS_A1_B2
Status: Status.UNKNOWN
Output dim: 0, lower bound: -27.7817977, upper bound: 27.7829412
time: 0.68 seconds

## BFS NS instance: NS_A2

### Backsubstitution after applying NS history:
0: -6.2154808, 21.2638779, -6.9639506, 23.4443073, -29.6597881, 28.2278271
1: -10.2115059, 21.7044182, -11.3851805, 23.9874458, -34.1989441, 33.0895996
2: -8.3589554, 23.4411602, -9.3549051, 25.8134308, -34.1723862, 32.7960663
3: -9.0649576, 32.1635170, -10.1015215, 35.4167786, -44.4817276, 42.2650375
4: -7.4015422, 30.4624748, -8.2747221, 33.5706329, -40.9721756, 38.7371979

Time for backsubstitution: 0.70 seconds

### NS candidates at layer 1
type: B, layer: 1, pos: 45
type: B, layer: 1, pos: 14
type: B, layer: 1, pos: 13
type: B, layer: 1, pos: 16
type: B, layer: 1, pos: 22
type: B, layer: 1, pos: 0
type: B, layer: 1, pos: 9
type: B, layer: 1, pos: 10
type: B, layer: 1, pos: 41
type: B, layer: 1, pos: 21
type: B, layer: 1, pos: 36
type: B, layer: 1, pos: 17

Time for candidate selection: 0.05 seconds

### Candidate
type: B, layer: 1, pos: 45

## Relational analysis of NS_A2_B1

### Relational analysis result of NS_A2_B1
Status: Status.UNKNOWN
Output dim: 0, lower bound: -27.8466200, upper bound: 27.8466200
time: 0.38 seconds

## Relational analysis of NS_A2_B2

### Relational analysis result of NS_A2_B2
Status: Status.UNKNOWN
Output dim: 0, lower bound: -27.8466200, upper bound: 27.8466200
time: 0.60 seconds

## Summary of splitting at layer (split count: 1)
- Time for NS candidates: 1.75 seconds
NS_A1_B1, status: Status.UNKNOWN, split count: 2, time: 1.75
Output dim: 0, lower bound: -27.8118463, upper bound: 27.8079184
NS_A1_B2, status: Status.UNKNOWN, split count: 2, time: 1.75
Output dim: 0, lower bound: -27.7817977, upper bound: 27.7829412
NS_A2_B1, status: Status.UNKNOWN, split count: 2, time: 1.75
Output dim: 0, lower bound: -27.8466200, upper bound: 27.8466200
NS_A2_B2, status: Status.UNKNOWN, split count: 2, time: 1.75
Output dim: 0, lower bound: -27.8466200, upper bound: 27.8466200

## BFS NS instance: NS_A1_B1

### Backsubstitution after applying NS history:
0: -4.7503119, 17.0180416, -5.3882785, 18.7908363, -23.5411453, 22.4063206
1: -7.8693061, 17.2813454, -8.8957186, 19.1404037, -27.0097103, 26.1770630
2: -6.3699632, 18.7084370, -7.2432556, 20.7321072, -27.1020699, 25.9516926
3: -6.9733434, 25.8055115, -7.9095993, 28.4771652, -35.4505081, 33.7151108
4: -5.6170487, 24.1190872, -6.4142904, 26.8979416, -32.5149918, 30.5333786

Time for backsubstitution: 0.70 seconds

### NS candidates at layer 1
type: A, layer: 1, pos: 13
type: A, layer: 1, pos: 14
type: A, layer: 1, pos: 16
type: A, layer: 1, pos: 22
type: A, layer: 1, pos: 9
type: A, layer: 1, pos: 10
type: A, layer: 1, pos: 17
type: A, layer: 1, pos: 21
type: A, layer: 1, pos: 41
type: A, layer: 1, pos: 0
type: A, layer: 1, pos: 36

Time for candidate selection: 0.05 seconds

### Candidate
type: A, layer: 1, pos: 13

## Relational analysis of NS_A1_B1_A1

### Relational analysis result of NS_A1_B1_A1
Status: Status.VERIFIED
Output dim: 0, lower bound: -27.7489732, upper bound: 27.7340198
time: 0.56 seconds

## Relational analysis of NS_A1_B1_A2
Optimization infeasible because this subproblem isn't reachable.

### Candidate
type: A, layer: 1, pos: 14

## Relational analysis of NS_A1_B1_A1

### Relational analysis result of NS_A1_B1_A1
Status: Status.UNKNOWN
Output dim: 0, lower bound: -27.7817977, upper bound: 27.7829412
time: 0.43 seconds

## Relational analysis of NS_A1_B1_A2

### Relational analysis result of NS_A1_B1_A2
Status: Status.UNKNOWN
Output dim: 0, lower bound: -27.7817977, upper bound: 27.7829412
time: 0.70 seconds

## BFS NS instance: NS_A1_B2

### Backsubstitution after applying NS history:
0: -4.5029621, 16.2596722, -6.2351542, 21.7795181, -26.2824802, 22.4948273
1: -7.4736395, 16.4933014, -10.2239170, 22.0891743, -29.5628128, 26.7172184
2: -6.0349174, 17.8740559, -8.3566227, 23.9043789, -29.9392929, 26.2306786
3: -6.6256905, 24.6575470, -9.0479250, 32.8376541, -39.4633446, 33.7054710
4: -5.3236103, 23.0057850, -7.3892879, 30.9044914, -36.2281036, 30.3950710

Time for backsubstitution: 0.70 seconds

### NS candidates at layer 1
type: A, layer: 1, pos: 13
type: A, layer: 1, pos: 14
type: A, layer: 1, pos: 16
type: A, layer: 1, pos: 9
type: A, layer: 1, pos: 22
type: A, layer: 1, pos: 10
type: A, layer: 1, pos: 17
type: A, layer: 1, pos: 21
type: A, layer: 1, pos: 41
type: A, layer: 1, pos: 0
type: A, layer: 1, pos: 36

Time for candidate selection: 0.05 seconds

### Candidate
type: A, layer: 1, pos: 13

## Relational analysis of NS_A1_B2_A1
Optimization infeasible because this subproblem isn't reachable.

### Candidate
type: A, layer: 1, pos: 14

## Relational analysis of NS_A1_B2_A1

### Relational analysis result of NS_A1_B2_A1
Status: Status.UNKNOWN
Output dim: 0, lower bound: -27.7817977, upper bound: 27.7829412
time: 0.93 seconds

## Relational analysis of NS_A1_B2_A2

### Relational analysis result of NS_A1_B2_A2
Status: Status.UNKNOWN
Output dim: 0, lower bound: -27.7817977, upper bound: 27.7829412
time: 0.44 seconds

## BFS NS instance: NS_A2_B1

### Backsubstitution after applying NS history:
0: -6.2154808, 21.2638779, -4.8150167, 17.2120781, -23.4275589, 26.0788918
1: -10.2115059, 21.7044182, -7.9723043, 17.4862099, -27.6977158, 29.6767235
2: -8.3589554, 23.4411602, -6.4574275, 18.9246101, -27.2835655, 29.8985825
3: -9.0649576, 32.1635170, -7.0645366, 26.1020927, -35.1670456, 39.2280540
4: -7.4015422, 30.4624748, -5.6939859, 24.4059410, -31.8074837, 36.1564598

Time for backsubstitution: 0.71 seconds

### NS candidates at layer 1
type: A, layer: 1, pos: 14
type: A, layer: 1, pos: 13
type: A, layer: 1, pos: 16
type: A, layer: 1, pos: 22
type: A, layer: 1, pos: 9
type: A, layer: 1, pos: 10
type: A, layer: 1, pos: 0
type: A, layer: 1, pos: 41
type: A, layer: 1, pos: 21
type: A, layer: 1, pos: 17
type: A, layer: 1, pos: 36

Time for candidate selection: 0.05 seconds

### Candidate
type: A, layer: 1, pos: 14

## Relational analysis of NS_A2_B1_A1

### Relational analysis result of NS_A2_B1_A1
Status: Status.UNKNOWN
Output dim: 0, lower bound: -27.7875726, upper bound: 27.7935425
time: 0.71 seconds

## Relational analysis of NS_A2_B1_A2

### Relational analysis result of NS_A2_B1_A2
Status: Status.VERIFIED
Output dim: 0, lower bound: -27.7601637, upper bound: 27.7601637
time: 0.64 seconds

## BFS NS instance: NS_A2_B2

### Backsubstitution after applying NS history:
0: -6.2154808, 21.2638779, -6.2154808, 21.2638779, -27.4793587, 27.4793587
1: -10.2115059, 21.7044182, -10.2115059, 21.7044182, -31.9159241, 31.9159241
2: -8.3589554, 23.4411602, -8.3589554, 23.4411602, -31.8001156, 31.8001156
3: -9.0649576, 32.1635170, -9.0649576, 32.1635170, -41.2284698, 41.2284698
4: -7.4015422, 30.4624748, -7.4015422, 30.4624748, -37.8640175, 37.8640175

Time for backsubstitution: 0.71 seconds

### NS candidates at layer 1
type: A, layer: 1, pos: 14
type: A, layer: 1, pos: 13
type: A, layer: 1, pos: 16
type: A, layer: 1, pos: 22
type: A, layer: 1, pos: 9
type: A, layer: 1, pos: 10
type: A, layer: 1, pos: 0
type: A, layer: 1, pos: 41
type: A, layer: 1, pos: 21
type: A, layer: 1, pos: 17
type: A, layer: 1, pos: 36

Time for candidate selection: 0.05 seconds

### Candidate
type: A, layer: 1, pos: 14

## Relational analysis of NS_A2_B2_A1

### Relational analysis result of NS_A2_B2_A1
Status: Status.UNKNOWN
Output dim: 0, lower bound: -27.7875726, upper bound: 27.7935425
time: 0.74 seconds

## Relational analysis of NS_A2_B2_A2

### Relational analysis result of NS_A2_B2_A2
Status: Status.VERIFIED
Output dim: 0, lower bound: -27.7601637, upper bound: 27.7601637
time: 0.70 seconds

## Summary of splitting at layer (split count: 2)
- Time for NS candidates: 2.21 seconds
NS_A1_B1_A1, status: Status.UNKNOWN, split count: 3, time: 2.21
Output dim: 0, lower bound: -27.7817977, upper bound: 27.7829412
NS_A1_B1_A2, status: Status.UNKNOWN, split count: 3, time: 2.21
Output dim: 0, lower bound: -27.7817977, upper bound: 27.7829412
NS_A1_B2_A1, status: Status.UNKNOWN, split count: 3, time: 2.21
Output dim: 0, lower bound: -27.7817977, upper bound: 27.7829412
NS_A1_B2_A2, status: Status.UNKNOWN, split count: 3, time: 2.21
Output dim: 0, lower bound: -27.7817977, upper bound: 27.7829412
NS_A2_B1_A1, status: Status.UNKNOWN, split count: 3, time: 2.21
Output dim: 0, lower bound: -27.7875726, upper bound: 27.7935425
NS_A2_B1_A2, status: Status.VERIFIED, split count: 3, time: 2.21
Output dim: 0, lower bound: -27.7601637, upper bound: 27.7601637
NS_A2_B2_A1, status: Status.UNKNOWN, split count: 3, time: 2.21
Output dim: 0, lower bound: -27.7875726, upper bound: 27.7935425
NS_A2_B2_A2, status: Status.VERIFIED, split count: 3, time: 2.21
Output dim: 0, lower bound: -27.7601637, upper bound: 27.7601637

## BFS NS instance: NS_A1_B1_A1

### Backsubstitution after applying NS history:
0: -4.1239586, 15.1485586, -5.3882785, 18.7908363, -22.9147949, 20.5368347
1: -6.8710823, 15.3317356, -8.8957186, 19.1404037, -26.0114861, 24.2274551
2: -5.5255399, 16.6497955, -7.2432556, 20.7321072, -26.2576466, 23.8930511
3: -6.0924091, 22.9905643, -7.9095993, 28.4771652, -34.5695724, 30.9001560
4: -4.8740869, 21.3502178, -6.4142904, 26.8979416, -31.7720261, 27.7645073

Time for backsubstitution: 0.71 seconds

### NS candidates at layer 1
type: B, layer: 1, pos: 13
type: B, layer: 1, pos: 45
type: B, layer: 1, pos: 16
type: B, layer: 1, pos: 22
type: B, layer: 1, pos: 9
type: B, layer: 1, pos: 10
type: B, layer: 1, pos: 17
type: B, layer: 1, pos: 41
type: B, layer: 1, pos: 0
type: B, layer: 1, pos: 21
type: B, layer: 1, pos: 36

Time for candidate selection: 0.05 seconds

### Candidate
type: B, layer: 1, pos: 13

### Candidate
type: B, layer: 1, pos: 45

## Relational analysis of NS_A1_B1_A1_B1

### Relational analysis result of NS_A1_B1_A1_B1
Status: Status.UNKNOWN
Output dim: 0, lower bound: -27.8118463, upper bound: 27.8079184
time: 0.43 seconds

## Relational analysis of NS_A1_B1_A1_B2

### Relational analysis result of NS_A1_B1_A1_B2
Status: Status.UNKNOWN
Output dim: 0, lower bound: -27.8118463, upper bound: 27.8079184
time: 0.54 seconds

## BFS NS instance: NS_A1_B1_A2

### Backsubstitution after applying NS history:
0: -5.2290773, 18.9489193, -5.3882785, 18.7908363, -24.0199108, 24.3371983
1: -8.6279545, 19.1228676, -8.8957186, 19.1404037, -27.7683582, 28.0185852
2: -6.9899049, 20.7296162, -7.2432556, 20.7321072, -27.7220097, 27.9728718
3: -7.5971055, 28.6222916, -7.9095993, 28.4771652, -36.0742722, 36.5318832
4: -6.1555519, 26.5859127, -6.4142904, 26.8979416, -33.0534897, 33.0001984

Time for backsubstitution: 0.72 seconds

### NS candidates at layer 1
type: B, layer: 1, pos: 13
type: B, layer: 1, pos: 45
type: B, layer: 1, pos: 16
type: B, layer: 1, pos: 22
type: B, layer: 1, pos: 9
type: B, layer: 1, pos: 10
type: B, layer: 1, pos: 17
type: B, layer: 1, pos: 41
type: B, layer: 1, pos: 0
type: B, layer: 1, pos: 21
type: B, layer: 1, pos: 36

Time for candidate selection: 0.05 seconds

### Candidate
type: B, layer: 1, pos: 13

### Candidate
type: B, layer: 1, pos: 45

## Relational analysis of NS_A1_B1_A2_B1

### Relational analysis result of NS_A1_B1_A2_B1
Status: Status.UNKNOWN
Output dim: 0, lower bound: -27.8118463, upper bound: 27.8079184
time: 0.91 seconds

## Relational analysis of NS_A1_B1_A2_B2

### Relational analysis result of NS_A1_B1_A2_B2
Status: Status.UNKNOWN
Output dim: 0, lower bound: -27.8118463, upper bound: 27.8079184
time: 0.72 seconds

## BFS NS instance: NS_A1_B2_A1

### Backsubstitution after applying NS history:
0: -4.1239586, 15.1485586, -6.2351542, 21.7795181, -25.9034767, 21.3837090
1: -6.8710823, 15.3317356, -10.2239170, 22.0891743, -28.9602566, 25.5556526
2: -5.5255399, 16.6497955, -8.3566227, 23.9043789, -29.4299164, 25.0064182
3: -6.0924091, 22.9905643, -9.0479250, 32.8376541, -38.9300575, 32.0384827
4: -4.8740869, 21.3502178, -7.3892879, 30.9044914, -35.7785721, 28.7395039

Time for backsubstitution: 0.72 seconds

### NS candidates at layer 1
type: B, layer: 1, pos: 45
type: B, layer: 1, pos: 13
type: B, layer: 1, pos: 16
type: B, layer: 1, pos: 9
type: B, layer: 1, pos: 22
type: B, layer: 1, pos: 10
type: B, layer: 1, pos: 41
type: B, layer: 1, pos: 17
type: B, layer: 1, pos: 0
type: B, layer: 1, pos: 21
type: B, layer: 1, pos: 36

Time for candidate selection: 0.05 seconds

### Candidate
type: B, layer: 1, pos: 45

## Relational analysis of NS_A1_B2_A1_B1

### Relational analysis result of NS_A1_B2_A1_B1
Status: Status.UNKNOWN
Output dim: 0, lower bound: -27.7817977, upper bound: 27.7829412
time: 0.62 seconds

## Relational analysis of NS_A1_B2_A1_B2

### Relational analysis result of NS_A1_B2_A1_B2
Status: Status.UNKNOWN
Output dim: 0, lower bound: -27.7817977, upper bound: 27.7829412
time: 0.56 seconds

## BFS NS instance: NS_A1_B2_A2

### Backsubstitution after applying NS history:
0: -5.2290773, 18.9489193, -6.2351542, 21.7795181, -27.0085926, 25.1840744
1: -8.6279545, 19.1228676, -10.2239170, 22.0891743, -30.7171288, 29.3467846
2: -6.9899049, 20.7296162, -8.3566227, 23.9043789, -30.8942795, 29.0862389
3: -7.5971055, 28.6222916, -9.0479250, 32.8376541, -40.4347534, 37.6702042
4: -6.1555519, 26.5859127, -7.3892879, 30.9044914, -37.0600395, 33.9751968

Time for backsubstitution: 0.72 seconds

### NS candidates at layer 1
type: B, layer: 1, pos: 45
type: B, layer: 1, pos: 13
type: B, layer: 1, pos: 16
type: B, layer: 1, pos: 9
type: B, layer: 1, pos: 22
type: B, layer: 1, pos: 10
type: B, layer: 1, pos: 41
type: B, layer: 1, pos: 17
type: B, layer: 1, pos: 0
type: B, layer: 1, pos: 21
type: B, layer: 1, pos: 36

Time for candidate selection: 0.05 seconds

### Candidate
type: B, layer: 1, pos: 45

## Relational analysis of NS_A1_B2_A2_B1

### Relational analysis result of NS_A1_B2_A2_B1
Status: Status.UNKNOWN
Output dim: 0, lower bound: -27.7817977, upper bound: 27.7829412
time: 0.75 seconds

## Relational analysis of NS_A1_B2_A2_B2

### Relational analysis result of NS_A1_B2_A2_B2
Status: Status.UNKNOWN
Output dim: 0, lower bound: -27.7817977, upper bound: 27.7829412
time: 0.63 seconds

## BFS NS instance: NS_A2_B1_A1

### Backsubstitution after applying NS history:
0: -5.4554358, 18.9849701, -4.7503119, 17.0180416, -22.4734745, 23.7352753
1: -9.0068264, 19.3360481, -7.8693061, 17.2813454, -26.2881718, 27.2053547
2: -7.3360786, 20.9468498, -6.3699632, 18.7084370, -26.0445156, 27.3168125
3: -8.0077810, 28.7492008, -6.9733434, 25.8055115, -33.8132935, 35.7225456
4: -6.5097780, 27.1807137, -5.6170487, 24.1190872, -30.6288605, 32.7977638

Time for backsubstitution: 0.71 seconds

### NS candidates at layer 1
type: B, layer: 1, pos: 13
type: B, layer: 1, pos: 14
type: B, layer: 1, pos: 16
type: B, layer: 1, pos: 22
type: B, layer: 1, pos: 9
type: B, layer: 1, pos: 10
type: B, layer: 1, pos: 17
type: B, layer: 1, pos: 21
type: B, layer: 1, pos: 41
type: B, layer: 1, pos: 0
type: B, layer: 1, pos: 36

Time for candidate selection: 0.05 seconds

### Candidate
type: B, layer: 1, pos: 13

## Relational analysis of NS_A2_B1_A1_B1

### Relational analysis result of NS_A2_B1_A1_B1
Status: Status.VERIFIED
Output dim: 0, lower bound: -27.7340198, upper bound: 27.7489732
time: 0.66 seconds

## Relational analysis of NS_A2_B1_A1_B2
Optimization infeasible because this subproblem isn't reachable.

### Candidate
type: B, layer: 1, pos: 14

## Relational analysis of NS_A2_B1_A1_B1

### Relational analysis result of NS_A2_B1_A1_B1
Status: Status.UNKNOWN
Output dim: 0, lower bound: -27.7829412, upper bound: 27.7817977
time: 0.69 seconds

## Relational analysis of NS_A2_B1_A1_B2

### Relational analysis result of NS_A2_B1_A1_B2
Status: Status.UNKNOWN
Output dim: 0, lower bound: -27.7829412, upper bound: 27.7817977
time: 1.27 seconds

## BFS NS instance: NS_A2_B2_A1

### Backsubstitution after applying NS history:
0: -5.4554358, 18.9849701, -6.1406865, 21.0467625, -26.5021973, 25.1256523
1: -9.0068264, 19.3360481, -10.0939865, 21.4784050, -30.4852276, 29.4300327
2: -7.3360786, 20.9468498, -8.2584724, 23.2039051, -30.5399837, 29.2053223
3: -8.0077810, 28.7492008, -8.9619379, 31.8389492, -39.8467255, 37.7111397
4: -6.5097780, 27.1807137, -7.3137531, 30.1513252, -36.6610985, 34.4944649

Time for backsubstitution: 0.72 seconds

### NS candidates at layer 1
type: B, layer: 1, pos: 14
type: B, layer: 1, pos: 13
type: B, layer: 1, pos: 16
type: B, layer: 1, pos: 22
type: B, layer: 1, pos: 9
type: B, layer: 1, pos: 10
type: B, layer: 1, pos: 0
type: B, layer: 1, pos: 41
type: B, layer: 1, pos: 21
type: B, layer: 1, pos: 17
type: B, layer: 1, pos: 36

Time for candidate selection: 0.05 seconds

### Candidate
type: B, layer: 1, pos: 14

## Relational analysis of NS_A2_B2_A1_B1

### Relational analysis result of NS_A2_B2_A1_B1
Status: Status.VERIFIED
Output dim: 0, lower bound: -27.7601637, upper bound: 27.7601637
time: 0.41 seconds

## Relational analysis of NS_A2_B2_A1_B2

### Relational analysis result of NS_A2_B2_A1_B2
Status: Status.VERIFIED
Output dim: 0, lower bound: -27.7601637, upper bound: 27.7601637
time: 0.56 seconds

## Summary of splitting at layer (split count: 3)
- Time for NS candidates: 1.75 seconds
NS_A1_B1_A1_B1, status: Status.UNKNOWN, split count: 4, time: 1.75
Output dim: 0, lower bound: -27.8118463, upper bound: 27.8079184
NS_A1_B1_A1_B2, status: Status.UNKNOWN, split count: 4, time: 1.75
Output dim: 0, lower bound: -27.8118463, upper bound: 27.8079184
NS_A1_B1_A2_B1, status: Status.UNKNOWN, split count: 4, time: 1.75
Output dim: 0, lower bound: -27.8118463, upper bound: 27.8079184
NS_A1_B1_A2_B2, status: Status.UNKNOWN, split count: 4, time: 1.75
Output dim: 0, lower bound: -27.8118463, upper bound: 27.8079184
NS_A1_B2_A1_B1, status: Status.UNKNOWN, split count: 4, time: 1.75
Output dim: 0, lower bound: -27.7817977, upper bound: 27.7829412
NS_A1_B2_A1_B2, status: Status.UNKNOWN, split count: 4, time: 1.75
Output dim: 0, lower bound: -27.7817977, upper bound: 27.7829412
NS_A1_B2_A2_B1, status: Status.UNKNOWN, split count: 4, time: 1.75
Output dim: 0, lower bound: -27.7817977, upper bound: 27.7829412
NS_A1_B2_A2_B2, status: Status.UNKNOWN, split count: 4, time: 1.75
Output dim: 0, lower bound: -27.7817977, upper bound: 27.7829412
NS_A2_B1_A1_B1, status: Status.UNKNOWN, split count: 4, time: 1.75
Output dim: 0, lower bound: -27.7829412, upper bound: 27.7817977
NS_A2_B1_A1_B2, status: Status.UNKNOWN, split count: 4, time: 1.75
Output dim: 0, lower bound: -27.7829412, upper bound: 27.7817977
NS_A2_B2_A1_B1, status: Status.VERIFIED, split count: 4, time: 1.75
Output dim: 0, lower bound: -27.7601637, upper bound: 27.7601637
NS_A2_B2_A1_B2, status: Status.VERIFIED, split count: 4, time: 1.75
Output dim: 0, lower bound: -27.7601637, upper bound: 27.7601637

## BFS NS instance: NS_A1_B1_A1_B1

### Backsubstitution after applying NS history:
0: -4.1239586, 15.1485586, -4.1239586, 15.1485586, -19.2725143, 19.2725124
1: -6.8710823, 15.3317356, -6.8710823, 15.3317356, -22.2028179, 22.2028179
2: -5.5255399, 16.6497955, -5.5255399, 16.6497955, -22.1753349, 22.1753349
3: -6.0924091, 22.9905643, -6.0924091, 22.9905643, -29.0829735, 29.0829735
4: -4.8740869, 21.3502178, -4.8740869, 21.3502178, -26.2243042, 26.2243042

Time for backsubstitution: 0.72 seconds

### NS candidates at layer 1
type: A, layer: 1, pos: 13
type: A, layer: 1, pos: 16
type: A, layer: 1, pos: 9
type: A, layer: 1, pos: 22
type: A, layer: 1, pos: 17
type: A, layer: 1, pos: 10
type: A, layer: 1, pos: 21
type: A, layer: 1, pos: 41
type: A, layer: 1, pos: 0
type: A, layer: 1, pos: 36

Time for candidate selection: 0.05 seconds

### Candidate
type: A, layer: 1, pos: 13

### Candidate
type: A, layer: 1, pos: 16

## Relational analysis of NS_A1_B1_A1_B1_A1
Optimization infeasible because this subproblem isn't reachable.

### Candidate
type: A, layer: 1, pos: 9

## Relational analysis of NS_A1_B1_A1_B1_A1

### Relational analysis result of NS_A1_B1_A1_B1_A1
Status: Status.UNKNOWN
Output dim: 0, lower bound: -27.8452373, upper bound: 27.8429820
time: 0.49 seconds

## Relational analysis of NS_A1_B1_A1_B1_A2

### Relational analysis result of NS_A1_B1_A1_B1_A2
Status: Status.UNKNOWN
Output dim: 0, lower bound: -27.8498972, upper bound: 27.8472336
time: 0.71 seconds

## BFS NS instance: NS_A1_B1_A1_B2

### Backsubstitution after applying NS history:
0: -4.1239586, 15.1485586, -5.4476757, 18.9605465, -23.0845032, 20.5962296
1: -6.8710823, 15.3317356, -8.9945908, 19.3103657, -26.1814480, 24.3263264
2: -5.5255399, 16.6497955, -7.3259263, 20.9208145, -26.4463539, 23.9757214
3: -6.0924091, 22.9905643, -7.9968929, 28.7140160, -34.8064270, 30.9874458
4: -4.8740869, 21.3502178, -6.5010571, 27.1483727, -32.0224533, 27.8512726

Time for backsubstitution: 0.72 seconds

### NS candidates at layer 1
type: A, layer: 1, pos: 13
type: A, layer: 1, pos: 16
type: A, layer: 1, pos: 9
type: A, layer: 1, pos: 22
type: A, layer: 1, pos: 17
type: A, layer: 1, pos: 10
type: A, layer: 1, pos: 21
type: A, layer: 1, pos: 41
type: A, layer: 1, pos: 0
type: A, layer: 1, pos: 36

Time for candidate selection: 0.05 seconds

### Candidate
type: A, layer: 1, pos: 13

### Candidate
type: A, layer: 1, pos: 16

## Relational analysis of NS_A1_B1_A1_B2_A1
Optimization infeasible because this subproblem isn't reachable.

### Candidate
type: A, layer: 1, pos: 9

## Relational analysis of NS_A1_B1_A1_B2_A1

### Relational analysis result of NS_A1_B1_A1_B2_A1
Status: Status.UNKNOWN
Output dim: 0, lower bound: -27.8452373, upper bound: 27.8429820
time: 0.70 seconds

## Relational analysis of NS_A1_B1_A1_B2_A2

### Relational analysis result of NS_A1_B1_A1_B2_A2
Status: Status.UNKNOWN
Output dim: 0, lower bound: -27.8498972, upper bound: 27.8472336
time: 0.71 seconds

## BFS NS instance: NS_A1_B1_A2_B1

### Backsubstitution after applying NS history:
0: -5.2290773, 18.9489193, -4.1239586, 15.1485586, -20.3776302, 23.0728779
1: -8.6279545, 19.1228676, -6.8710823, 15.3317356, -23.9596901, 25.9939499
2: -6.9899049, 20.7296162, -5.5255399, 16.6497955, -23.6396961, 26.2551556
3: -7.5971055, 28.6222916, -6.0924091, 22.9905643, -30.5876656, 34.7146988
4: -6.1555519, 26.5859127, -4.8740869, 21.3502178, -27.5057697, 31.4599972

Time for backsubstitution: 0.72 seconds

### NS candidates at layer 1
type: A, layer: 1, pos: 13
type: A, layer: 1, pos: 16
type: A, layer: 1, pos: 9
type: A, layer: 1, pos: 22
type: A, layer: 1, pos: 10
type: A, layer: 1, pos: 17
type: A, layer: 1, pos: 41
type: A, layer: 1, pos: 21
type: A, layer: 1, pos: 0
type: A, layer: 1, pos: 36

Time for candidate selection: 0.05 seconds

### Candidate
type: A, layer: 1, pos: 13

### Candidate
type: A, layer: 1, pos: 16

## Relational analysis of NS_A1_B1_A2_B1_A1
Optimization infeasible because this subproblem isn't reachable.

### Candidate
type: A, layer: 1, pos: 9

## Relational analysis of NS_A1_B1_A2_B1_A1

### Relational analysis result of NS_A1_B1_A2_B1_A1
Status: Status.VERIFIED
Output dim: 0, lower bound: -27.7208208, upper bound: 27.7038050
time: 0.85 seconds

## Relational analysis of NS_A1_B1_A2_B1_A2

### Relational analysis result of NS_A1_B1_A2_B1_A2
Status: Status.UNKNOWN
Output dim: 0, lower bound: -27.8118463, upper bound: 27.8079184
time: 0.57 seconds

## BFS NS instance: NS_A1_B1_A2_B2

### Backsubstitution after applying NS history:
0: -5.2290773, 18.9489193, -5.4476757, 18.9605465, -24.1896210, 24.3965950
1: -8.6279545, 19.1228676, -8.9945908, 19.3103657, -27.9383202, 28.1174583
2: -6.9899049, 20.7296162, -7.3259263, 20.9208145, -27.9107170, 28.0555420
3: -7.5971055, 28.6222916, -7.9968929, 28.7140160, -36.3111191, 36.6191788
4: -6.1555519, 26.5859127, -6.5010571, 27.1483727, -33.3039169, 33.0869675

Time for backsubstitution: 0.72 seconds

### NS candidates at layer 1
type: A, layer: 1, pos: 13
type: A, layer: 1, pos: 16
type: A, layer: 1, pos: 9
type: A, layer: 1, pos: 22
type: A, layer: 1, pos: 10
type: A, layer: 1, pos: 17
type: A, layer: 1, pos: 41
type: A, layer: 1, pos: 21
type: A, layer: 1, pos: 0
type: A, layer: 1, pos: 36

Time for candidate selection: 0.05 seconds

### Candidate
type: A, layer: 1, pos: 13

### Candidate
type: A, layer: 1, pos: 16

## Relational analysis of NS_A1_B1_A2_B2_A1
Optimization infeasible because this subproblem isn't reachable.

### Candidate
type: A, layer: 1, pos: 9

## Relational analysis of NS_A1_B1_A2_B2_A1

### Relational analysis result of NS_A1_B1_A2_B2_A1
Status: Status.VERIFIED
Output dim: 0, lower bound: -27.7208208, upper bound: 27.7038050
time: 0.69 seconds

## Relational analysis of NS_A1_B1_A2_B2_A2

### Relational analysis result of NS_A1_B1_A2_B2_A2
Status: Status.UNKNOWN
Output dim: 0, lower bound: -27.8118463, upper bound: 27.8079184
time: 0.53 seconds

## BFS NS instance: NS_A1_B2_A1_B1

### Backsubstitution after applying NS history:
0: -4.1239586, 15.1485586, -5.3676920, 19.4284458, -23.5524025, 20.5162487
1: -6.8710823, 15.3317356, -8.8535709, 19.6024017, -26.4734840, 24.1853065
2: -5.5255399, 16.6497955, -7.1736655, 21.2490845, -26.7746239, 23.8234596
3: -6.0924091, 22.9905643, -7.7958689, 29.3602962, -35.4527054, 30.7864265
4: -4.8740869, 21.3502178, -6.3186235, 27.2952156, -32.1692963, 27.6688385

Time for backsubstitution: 0.74 seconds

### NS candidates at layer 1
type: A, layer: 1, pos: 13
type: A, layer: 1, pos: 16
type: A, layer: 1, pos: 9
type: A, layer: 1, pos: 22
type: A, layer: 1, pos: 17
type: A, layer: 1, pos: 10
type: A, layer: 1, pos: 21
type: A, layer: 1, pos: 41
type: A, layer: 1, pos: 0
type: A, layer: 1, pos: 36

Time for candidate selection: 0.05 seconds

### Candidate
type: A, layer: 1, pos: 13

### Candidate
type: A, layer: 1, pos: 16

## Relational analysis of NS_A1_B2_A1_B1_A1
Optimization infeasible because this subproblem isn't reachable.

### Candidate
type: A, layer: 1, pos: 9

## Relational analysis of NS_A1_B2_A1_B1_A1

### Relational analysis result of NS_A1_B2_A1_B1_A1
Status: Status.UNKNOWN
Output dim: 0, lower bound: -27.7680185, upper bound: 27.7762063
time: 0.75 seconds

## Relational analysis of NS_A1_B2_A1_B1_A2

### Relational analysis result of NS_A1_B2_A1_B1_A2
Status: Status.UNKNOWN
Output dim: 0, lower bound: -27.7911695, upper bound: 27.7939982
time: 0.59 seconds

## BFS NS instance: NS_A1_B2_A1_B2

### Backsubstitution after applying NS history:
0: -4.1239586, 15.1485586, -6.3037281, 22.0140495, -26.1380081, 21.4522800
1: -6.8710823, 15.3317356, -10.3375835, 22.3265495, -29.1976318, 25.6693192
2: -5.5255399, 16.6497955, -8.4509039, 24.1487713, -29.6743107, 25.1006985
3: -6.0924091, 22.9905643, -9.1481276, 33.1661034, -39.2585144, 32.1386871
4: -4.8740869, 21.3502178, -7.4770923, 31.2366295, -36.1107101, 28.8273106

Time for backsubstitution: 0.73 seconds

### NS candidates at layer 1
type: A, layer: 1, pos: 13
type: A, layer: 1, pos: 16
type: A, layer: 1, pos: 9
type: A, layer: 1, pos: 22
type: A, layer: 1, pos: 17
type: A, layer: 1, pos: 10
type: A, layer: 1, pos: 21
type: A, layer: 1, pos: 41
type: A, layer: 1, pos: 0
type: A, layer: 1, pos: 36

Time for candidate selection: 0.05 seconds

### Candidate
type: A, layer: 1, pos: 13

### Candidate
type: A, layer: 1, pos: 16

## Relational analysis of NS_A1_B2_A1_B2_A1
Optimization infeasible because this subproblem isn't reachable.

### Candidate
type: A, layer: 1, pos: 9

## Relational analysis of NS_A1_B2_A1_B2_A1

### Relational analysis result of NS_A1_B2_A1_B2_A1
Status: Status.UNKNOWN
Output dim: 0, lower bound: -27.7680185, upper bound: 27.7762063
time: 0.66 seconds

## Relational analysis of NS_A1_B2_A1_B2_A2

### Relational analysis result of NS_A1_B2_A1_B2_A2
Status: Status.UNKNOWN
Output dim: 0, lower bound: -27.7911695, upper bound: 27.7939982
time: 0.52 seconds

## BFS NS instance: NS_A1_B2_A2_B1

### Backsubstitution after applying NS history:
0: -5.2290773, 18.9489193, -5.3676920, 19.4284458, -24.6575184, 24.3166122
1: -8.6279545, 19.1228676, -8.8535709, 19.6024017, -28.2303562, 27.9764366
2: -6.9899049, 20.7296162, -7.1736655, 21.2490845, -28.2389851, 27.9032822
3: -7.5971055, 28.6222916, -7.7958689, 29.3602962, -36.9574013, 36.4181519
4: -6.1555519, 26.5859127, -6.3186235, 27.2952156, -33.4507599, 32.9045372

Time for backsubstitution: 0.73 seconds

### NS candidates at layer 1
type: A, layer: 1, pos: 13
type: A, layer: 1, pos: 16
type: A, layer: 1, pos: 9
type: A, layer: 1, pos: 22
type: A, layer: 1, pos: 10
type: A, layer: 1, pos: 17
type: A, layer: 1, pos: 41
type: A, layer: 1, pos: 21
type: A, layer: 1, pos: 0
type: A, layer: 1, pos: 36

Time for candidate selection: 0.05 seconds

### Candidate
type: A, layer: 1, pos: 13

### Candidate
type: A, layer: 1, pos: 16

## Relational analysis of NS_A1_B2_A2_B1_A1
Optimization infeasible because this subproblem isn't reachable.

### Candidate
type: A, layer: 1, pos: 9

## Relational analysis of NS_A1_B2_A2_B1_A1

### Relational analysis result of NS_A1_B2_A2_B1_A1
Status: Status.VERIFIED
Output dim: 0, lower bound: -27.4954318, upper bound: 27.4304726
time: 0.60 seconds

## Relational analysis of NS_A1_B2_A2_B1_A2

### Relational analysis result of NS_A1_B2_A2_B1_A2
Status: Status.UNKNOWN
Output dim: 0, lower bound: -27.7817977, upper bound: 27.7829412
time: 0.55 seconds

## BFS NS instance: NS_A1_B2_A2_B2

### Backsubstitution after applying NS history:
0: -5.2290773, 18.9489193, -6.3037281, 22.0140495, -27.2431240, 25.2526455
1: -8.6279545, 19.1228676, -10.3375835, 22.3265495, -30.9545040, 29.4604473
2: -6.9899049, 20.7296162, -8.4509039, 24.1487713, -31.1386719, 29.1805191
3: -7.5971055, 28.6222916, -9.1481276, 33.1661034, -40.7632065, 37.7704163
4: -6.1555519, 26.5859127, -7.4770923, 31.2366295, -37.3921814, 34.0630035

Time for backsubstitution: 0.73 seconds

### NS candidates at layer 1
type: A, layer: 1, pos: 13
type: A, layer: 1, pos: 16
type: A, layer: 1, pos: 9
type: A, layer: 1, pos: 22
type: A, layer: 1, pos: 10
type: A, layer: 1, pos: 17
type: A, layer: 1, pos: 41
type: A, layer: 1, pos: 21
type: A, layer: 1, pos: 0
type: A, layer: 1, pos: 36

Time for candidate selection: 0.05 seconds

### Candidate
type: A, layer: 1, pos: 13

### Candidate
type: A, layer: 1, pos: 16

## Relational analysis of NS_A1_B2_A2_B2_A1
Optimization infeasible because this subproblem isn't reachable.

### Candidate
type: A, layer: 1, pos: 9

## Relational analysis of NS_A1_B2_A2_B2_A1

### Relational analysis result of NS_A1_B2_A2_B2_A1
Status: Status.VERIFIED
Output dim: 0, lower bound: -27.4954318, upper bound: 27.4304726
time: 0.49 seconds

## Relational analysis of NS_A1_B2_A2_B2_A2

### Relational analysis result of NS_A1_B2_A2_B2_A2
Status: Status.UNKNOWN
Output dim: 0, lower bound: -27.7817977, upper bound: 27.7829412
time: 0.72 seconds

## BFS NS instance: NS_A2_B1_A1_B1

### Backsubstitution after applying NS history:
0: -5.4554358, 18.9849701, -4.1239586, 15.1485586, -20.6039886, 23.1089268
1: -9.0068264, 19.3360481, -6.8710823, 15.3317356, -24.3385620, 26.2071304
2: -7.3360786, 20.9468498, -5.5255399, 16.6497955, -23.9858723, 26.4723892
3: -8.0077810, 28.7492008, -6.0924091, 22.9905643, -30.9983368, 34.8416100
4: -6.5097780, 27.1807137, -4.8740869, 21.3502178, -27.8599949, 32.0547943

Time for backsubstitution: 0.73 seconds

### NS candidates at layer 1
type: A, layer: 1, pos: 13
type: A, layer: 1, pos: 16
type: A, layer: 1, pos: 22
type: A, layer: 1, pos: 9
type: A, layer: 1, pos: 10
type: A, layer: 1, pos: 0
type: A, layer: 1, pos: 41
type: A, layer: 1, pos: 17
type: A, layer: 1, pos: 21
type: A, layer: 1, pos: 36

Time for candidate selection: 0.05 seconds

### Candidate
type: A, layer: 1, pos: 13

### Candidate
type: A, layer: 1, pos: 16

## Relational analysis of NS_A2_B1_A1_B1_A1
Optimization infeasible because this subproblem isn't reachable.

### Candidate
type: A, layer: 1, pos: 22

## Relational analysis of NS_A2_B1_A1_B1_A1
Optimization infeasible because this subproblem isn't reachable.

### Candidate
type: A, layer: 1, pos: 9

## Relational analysis of NS_A2_B1_A1_B1_A1
Optimization infeasible because this subproblem isn't reachable.

### Candidate
type: A, layer: 1, pos: 10

## Relational analysis of NS_A2_B1_A1_B1_A1

### Relational analysis result of NS_A2_B1_A1_B1_A1
Status: Status.UNKNOWN
Output dim: 0, lower bound: -27.8079184, upper bound: 27.8118463
time: 0.54 seconds

## Relational analysis of NS_A2_B1_A1_B1_A2

### Relational analysis result of NS_A2_B1_A1_B1_A2
Status: Status.UNKNOWN
Output dim: 0, lower bound: -27.8066735, upper bound: 27.8100959
time: 0.74 seconds

## BFS NS instance: NS_A2_B1_A1_B2

### Backsubstitution after applying NS history:
0: -5.4554358, 18.9849701, -5.2290773, 18.9489193, -24.4043541, 24.2140408
1: -9.0068264, 19.3360481, -8.6279545, 19.1228676, -28.1296940, 27.9640026
2: -7.3360786, 20.9468498, -6.9899049, 20.7296162, -28.0656948, 27.9367523
3: -8.0077810, 28.7492008, -7.5971055, 28.6222916, -36.6300621, 36.3463058
4: -6.5097780, 27.1807137, -6.1555519, 26.5859127, -33.0956841, 33.3362617

Time for backsubstitution: 0.74 seconds

### NS candidates at layer 1
type: A, layer: 1, pos: 13
type: A, layer: 1, pos: 16
type: A, layer: 1, pos: 22
type: A, layer: 1, pos: 9
type: A, layer: 1, pos: 10
type: A, layer: 1, pos: 0
type: A, layer: 1, pos: 41
type: A, layer: 1, pos: 17
type: A, layer: 1, pos: 21
type: A, layer: 1, pos: 36

Time for candidate selection: 0.05 seconds

### Candidate
type: A, layer: 1, pos: 13

### Candidate
type: A, layer: 1, pos: 16

## Relational analysis of NS_A2_B1_A1_B2_A1
Optimization infeasible because this subproblem isn't reachable.

### Candidate
type: A, layer: 1, pos: 22

## Relational analysis of NS_A2_B1_A1_B2_A1
Optimization infeasible because this subproblem isn't reachable.

### Candidate
type: A, layer: 1, pos: 9

## Relational analysis of NS_A2_B1_A1_B2_A1
Optimization infeasible because this subproblem isn't reachable.

### Candidate
type: A, layer: 1, pos: 10

## Relational analysis of NS_A2_B1_A1_B2_A1

### Relational analysis result of NS_A2_B1_A1_B2_A1
Status: Status.UNKNOWN
Output dim: 0, lower bound: -27.8079184, upper bound: 27.8118463
time: 0.56 seconds

## Relational analysis of NS_A2_B1_A1_B2_A2

### Relational analysis result of NS_A2_B1_A1_B2_A2
Status: Status.UNKNOWN
Output dim: 0, lower bound: -27.8066735, upper bound: 27.8100959
time: 0.50 seconds

## Summary of splitting at layer (split count: 4)
- Time for NS candidates: 2.71 seconds
NS_A1_B1_A1_B1_A1, status: Status.UNKNOWN, split count: 5, time: 2.71
Output dim: 0, lower bound: -27.8452373, upper bound: 27.8429820
NS_A1_B1_A1_B1_A2, status: Status.UNKNOWN, split count: 5, time: 2.71
Output dim: 0, lower bound: -27.8498972, upper bound: 27.8472336
NS_A1_B1_A1_B2_A1, status: Status.UNKNOWN, split count: 5, time: 2.71
Output dim: 0, lower bound: -27.8452373, upper bound: 27.8429820
NS_A1_B1_A1_B2_A2, status: Status.UNKNOWN, split count: 5, time: 2.71
Output dim: 0, lower bound: -27.8498972, upper bound: 27.8472336
NS_A1_B1_A2_B1_A1, status: Status.VERIFIED, split count: 5, time: 2.71
Output dim: 0, lower bound: -27.7208208, upper bound: 27.7038050
NS_A1_B1_A2_B1_A2, status: Status.UNKNOWN, split count: 5, time: 2.71
Output dim: 0, lower bound: -27.8118463, upper bound: 27.8079184
NS_A1_B1_A2_B2_A1, status: Status.VERIFIED, split count: 5, time: 2.71
Output dim: 0, lower bound: -27.7208208, upper bound: 27.7038050
NS_A1_B1_A2_B2_A2, status: Status.UNKNOWN, split count: 5, time: 2.71
Output dim: 0, lower bound: -27.8118463, upper bound: 27.8079184
NS_A1_B2_A1_B1_A1, status: Status.UNKNOWN, split count: 5, time: 2.71
Output dim: 0, lower bound: -27.7680185, upper bound: 27.7762063
NS_A1_B2_A1_B1_A2, status: Status.UNKNOWN, split count: 5, time: 2.71
Output dim: 0, lower bound: -27.7911695, upper bound: 27.7939982
NS_A1_B2_A1_B2_A1, status: Status.UNKNOWN, split count: 5, time: 2.71
Output dim: 0, lower bound: -27.7680185, upper bound: 27.7762063
NS_A1_B2_A1_B2_A2, status: Status.UNKNOWN, split count: 5, time: 2.71
Output dim: 0, lower bound: -27.7911695, upper bound: 27.7939982
NS_A1_B2_A2_B1_A1, status: Status.VERIFIED, split count: 5, time: 2.71
Output dim: 0, lower bound: -27.4954318, upper bound: 27.4304726
NS_A1_B2_A2_B1_A2, status: Status.UNKNOWN, split count: 5, time: 2.71
Output dim: 0, lower bound: -27.7817977, upper bound: 27.7829412
NS_A1_B2_A2_B2_A1, status: Status.VERIFIED, split count: 5, time: 2.71
Output dim: 0, lower bound: -27.4954318, upper bound: 27.4304726
NS_A1_B2_A2_B2_A2, status: Status.UNKNOWN, split count: 5, time: 2.71
Output dim: 0, lower bound: -27.7817977, upper bound: 27.7829412
NS_A2_B1_A1_B1_A1, status: Status.UNKNOWN, split count: 5, time: 2.71
Output dim: 0, lower bound: -27.8079184, upper bound: 27.8118463
NS_A2_B1_A1_B1_A2, status: Status.UNKNOWN, split count: 5, time: 2.71
Output dim: 0, lower bound: -27.8066735, upper bound: 27.8100959
NS_A2_B1_A1_B2_A1, status: Status.UNKNOWN, split count: 5, time: 2.71
Output dim: 0, lower bound: -27.8079184, upper bound: 27.8118463
NS_A2_B1_A1_B2_A2, status: Status.UNKNOWN, split count: 5, time: 2.71
Output dim: 0, lower bound: -27.8066735, upper bound: 27.8100959

## BFS NS instance: NS_A1_B1_A1_B1_A1

### Backsubstitution after applying NS history:
0: -3.9672580, 14.7736979, -4.0048380, 14.7744207, -18.7416782, 18.7785339
1: -6.6120090, 14.9280148, -6.6800752, 14.9509392, -21.5629425, 21.6080894
2: -5.3131099, 16.2373619, -5.3646793, 16.2427387, -21.5558453, 21.6020374
3: -5.8528652, 22.4279957, -5.9247437, 22.4313984, -28.2842617, 28.3527393
4: -4.6880665, 20.7485371, -4.7332287, 20.8038006, -25.4918671, 25.4817657

Time for backsubstitution: 0.73 seconds

### NS candidates at layer 1
type: B, layer: 1, pos: 13
type: B, layer: 1, pos: 16
type: B, layer: 1, pos: 22
type: B, layer: 1, pos: 9
type: B, layer: 1, pos: 17
type: B, layer: 1, pos: 10
type: B, layer: 1, pos: 21
type: B, layer: 1, pos: 41
type: B, layer: 1, pos: 0
type: B, layer: 1, pos: 36

Time for candidate selection: 0.05 seconds

### Candidate
type: B, layer: 1, pos: 13

### Candidate
type: B, layer: 1, pos: 16

### Candidate
type: B, layer: 1, pos: 22

## Relational analysis of NS_A1_B1_A1_B1_A1_B1

### Relational analysis result of NS_A1_B1_A1_B1_A1_B1
Status: Status.UNKNOWN
Output dim: 0, lower bound: -27.8510354, upper bound: 27.8506168
time: 0.56 seconds

## Relational analysis of NS_A1_B1_A1_B1_A1_B2

### Relational analysis result of NS_A1_B1_A1_B1_A1_B2
Status: Status.UNKNOWN
Output dim: 0, lower bound: -27.8510354, upper bound: 27.8503726
time: 0.53 seconds

## BFS NS instance: NS_A1_B1_A1_B1_A2

### Backsubstitution after applying NS history:
0: -4.0236449, 14.8443575, -4.1239586, 15.1485586, -19.1721935, 18.9683132
1: -6.7107162, 15.0167494, -6.8710823, 15.3317356, -22.0424519, 21.8878307
2: -5.3908873, 16.3171444, -5.5255399, 16.6497955, -22.0406837, 21.8426838
3: -5.9503188, 22.5346546, -6.0924091, 22.9905643, -28.9408836, 28.6270638
4: -4.7566547, 20.9033737, -4.8740869, 21.3502178, -26.1068726, 25.7774601

Time for backsubstitution: 0.73 seconds

### NS candidates at layer 1
type: B, layer: 1, pos: 13
type: B, layer: 1, pos: 16
type: B, layer: 1, pos: 9
type: B, layer: 1, pos: 22
type: B, layer: 1, pos: 17
type: B, layer: 1, pos: 10
type: B, layer: 1, pos: 21
type: B, layer: 1, pos: 41
type: B, layer: 1, pos: 0
type: B, layer: 1, pos: 36

Time for candidate selection: 0.05 seconds

### Candidate
type: B, layer: 1, pos: 13

### Candidate
type: B, layer: 1, pos: 16

### Candidate
type: B, layer: 1, pos: 9

## Relational analysis of NS_A1_B1_A1_B1_A2_B1

### Relational analysis result of NS_A1_B1_A1_B1_A2_B1
Status: Status.UNKNOWN
Output dim: 0, lower bound: -27.8521223, upper bound: 27.8521345
time: 0.66 seconds

## Relational analysis of NS_A1_B1_A1_B1_A2_B2

### Relational analysis result of NS_A1_B1_A1_B1_A2_B2
Status: Status.UNKNOWN
Output dim: 0, lower bound: -27.8521223, upper bound: 27.8521345
time: 0.51 seconds

## BFS NS instance: NS_A1_B1_A1_B2_A1

### Backsubstitution after applying NS history:
0: -3.9672580, 14.7736979, -5.3109922, 18.5251369, -22.4923954, 20.0846863
1: -6.6120090, 14.9280148, -8.7747793, 18.8638172, -25.4758263, 23.7027931
2: -5.3131099, 16.2373619, -7.1408863, 20.4450340, -25.7581444, 23.3782482
3: -5.8528652, 22.4279957, -7.8019438, 28.0624371, -33.9153023, 30.2299385
4: -4.6880665, 20.7485371, -6.3360844, 26.5145111, -31.2025738, 27.0846214

Time for backsubstitution: 0.74 seconds

### NS candidates at layer 1
type: B, layer: 1, pos: 13
type: B, layer: 1, pos: 16
type: B, layer: 1, pos: 22
type: B, layer: 1, pos: 10
type: B, layer: 1, pos: 9
type: B, layer: 1, pos: 0
type: B, layer: 1, pos: 41
type: B, layer: 1, pos: 17
type: B, layer: 1, pos: 21
type: B, layer: 1, pos: 36

Time for candidate selection: 0.05 seconds

### Candidate
type: B, layer: 1, pos: 13

### Candidate
type: B, layer: 1, pos: 16

### Candidate
type: B, layer: 1, pos: 22

## Relational analysis of NS_A1_B1_A1_B2_A1_B1
Optimization infeasible because this subproblem isn't reachable.

### Candidate
type: B, layer: 1, pos: 10

## Relational analysis of NS_A1_B1_A1_B2_A1_B1

### Relational analysis result of NS_A1_B1_A1_B2_A1_B1
Status: Status.UNKNOWN
Output dim: 0, lower bound: -27.8303058, upper bound: 27.8288729
time: 0.80 seconds

## Relational analysis of NS_A1_B1_A1_B2_A1_B2

### Relational analysis result of NS_A1_B1_A1_B2_A1_B2
Status: Status.UNKNOWN
Output dim: 0, lower bound: -27.8234250, upper bound: 27.8218229
time: 0.46 seconds

## BFS NS instance: NS_A1_B1_A1_B2_A2

### Backsubstitution after applying NS history:
0: -4.0236449, 14.8443575, -5.4476757, 18.9605465, -22.9841862, 20.2920303
1: -6.7107162, 15.0167494, -8.9945908, 19.3103657, -26.0210800, 24.0113411
2: -5.3908873, 16.3171444, -7.3259263, 20.9208145, -26.3117027, 23.6430702
3: -5.9503188, 22.5346546, -7.9968929, 28.7140160, -34.6643333, 30.5315399
4: -4.7566547, 20.9033737, -6.5010571, 27.1483727, -31.9050236, 27.4044304

Time for backsubstitution: 0.74 seconds

### NS candidates at layer 1
type: B, layer: 1, pos: 13
type: B, layer: 1, pos: 16
type: B, layer: 1, pos: 22
type: B, layer: 1, pos: 9
type: B, layer: 1, pos: 10
type: B, layer: 1, pos: 0
type: B, layer: 1, pos: 41
type: B, layer: 1, pos: 17
type: B, layer: 1, pos: 21
type: B, layer: 1, pos: 36

Time for candidate selection: 0.05 seconds

### Candidate
type: B, layer: 1, pos: 13

### Candidate
type: B, layer: 1, pos: 16

### Candidate
type: B, layer: 1, pos: 22

## Relational analysis of NS_A1_B1_A1_B2_A2_B1
Optimization infeasible because this subproblem isn't reachable.

### Candidate
type: B, layer: 1, pos: 9

## Relational analysis of NS_A1_B1_A1_B2_A2_B1

### Relational analysis result of NS_A1_B1_A1_B2_A2_B1
Status: Status.UNKNOWN
Output dim: 0, lower bound: -27.7755511, upper bound: 27.7687815
time: 0.52 seconds

## Relational analysis of NS_A1_B1_A1_B2_A2_B2

### Relational analysis result of NS_A1_B1_A1_B2_A2_B2
Status: Status.UNKNOWN
Output dim: 0, lower bound: -27.7755511, upper bound: 27.8472336
time: 0.62 seconds

## BFS NS instance: NS_A1_B1_A2_B1_A2

### Backsubstitution after applying NS history:
0: -5.1221609, 18.6184425, -4.1239586, 15.1485586, -20.2707195, 22.7424011
1: -8.4564943, 18.7807064, -6.8710823, 15.3317356, -23.7882309, 25.6517887
2: -6.8467531, 20.3681755, -5.5255399, 16.6497955, -23.4965458, 25.8937149
3: -7.4440994, 28.1251602, -6.0924091, 22.9905643, -30.4346581, 34.2175674
4: -6.0290127, 26.1052856, -4.8740869, 21.3502178, -27.3792305, 30.9793720

Time for backsubstitution: 0.74 seconds

### NS candidates at layer 1
type: B, layer: 1, pos: 13
type: B, layer: 1, pos: 16
type: B, layer: 1, pos: 9
type: B, layer: 1, pos: 22
type: B, layer: 1, pos: 17
type: B, layer: 1, pos: 10
type: B, layer: 1, pos: 21
type: B, layer: 1, pos: 41
type: B, layer: 1, pos: 0
type: B, layer: 1, pos: 36

Time for candidate selection: 0.05 seconds

### Candidate
type: B, layer: 1, pos: 13

### Candidate
type: B, layer: 1, pos: 16

### Candidate
type: B, layer: 1, pos: 9

## Relational analysis of NS_A1_B1_A2_B1_A2_B1

### Relational analysis result of NS_A1_B1_A2_B1_A2_B1
Status: Status.UNKNOWN
Output dim: 0, lower bound: -27.8079817, upper bound: 27.8050375
time: 0.61 seconds

## Relational analysis of NS_A1_B1_A2_B1_A2_B2

### Relational analysis result of NS_A1_B1_A2_B1_A2_B2
Status: Status.UNKNOWN
Output dim: 0, lower bound: -27.8079817, upper bound: 27.8144338
time: 0.83 seconds

## BFS NS instance: NS_A1_B1_A2_B2_A2

### Backsubstitution after applying NS history:
0: -5.1221609, 18.6184425, -5.4476757, 18.9605465, -24.0827065, 24.0661182
1: -8.4564943, 18.7807064, -8.9945908, 19.3103657, -27.7668591, 27.7752972
2: -6.8467531, 20.3681755, -7.3259263, 20.9208145, -27.7675667, 27.6941013
3: -7.4440994, 28.1251602, -7.9968929, 28.7140160, -36.1581078, 36.1220436
4: -6.0290127, 26.1052856, -6.5010571, 27.1483727, -33.1773834, 32.6063423

Time for backsubstitution: 0.74 seconds

### NS candidates at layer 1
type: B, layer: 1, pos: 13
type: B, layer: 1, pos: 16
type: B, layer: 1, pos: 22
type: B, layer: 1, pos: 9
type: B, layer: 1, pos: 10
type: B, layer: 1, pos: 0
type: B, layer: 1, pos: 41
type: B, layer: 1, pos: 17
type: B, layer: 1, pos: 21
type: B, layer: 1, pos: 36

Time for candidate selection: 0.05 seconds

### Candidate
type: B, layer: 1, pos: 13

### Candidate
type: B, layer: 1, pos: 16

### Candidate
type: B, layer: 1, pos: 22

## Relational analysis of NS_A1_B1_A2_B2_A2_B1
Optimization infeasible because this subproblem isn't reachable.

### Candidate
type: B, layer: 1, pos: 9

## Relational analysis of NS_A1_B1_A2_B2_A2_B1
Optimization infeasible because this subproblem isn't reachable.

### Candidate
type: B, layer: 1, pos: 10

## Relational analysis of NS_A1_B1_A2_B2_A2_B1

### Relational analysis result of NS_A1_B1_A2_B2_A2_B1
Status: Status.UNKNOWN
Output dim: 0, lower bound: -27.8118463, upper bound: 27.8079184
time: 0.56 seconds

## Relational analysis of NS_A1_B1_A2_B2_A2_B2

### Relational analysis result of NS_A1_B1_A2_B2_A2_B2
Status: Status.UNKNOWN
Output dim: 0, lower bound: -27.8100959, upper bound: 27.8066735
time: 0.59 seconds

## BFS NS instance: NS_A1_B2_A1_B1_A1

### Backsubstitution after applying NS history:
0: -3.9672580, 14.7736979, -5.2224030, 18.9615154, -22.9287739, 19.9960976
1: -6.6120090, 14.9280148, -8.6213322, 19.1280823, -25.7400913, 23.5493469
2: -5.3131099, 16.2373619, -6.9774370, 20.7433205, -26.0564308, 23.2147942
3: -5.8528652, 22.4279957, -7.5894547, 28.6626644, -34.5155296, 30.0174465
4: -4.6880665, 20.7485371, -6.1466241, 26.6249580, -31.3130207, 26.8951607

Time for backsubstitution: 0.74 seconds

### NS candidates at layer 1
type: B, layer: 1, pos: 13
type: B, layer: 1, pos: 16
type: B, layer: 1, pos: 9
type: B, layer: 1, pos: 22
type: B, layer: 1, pos: 10
type: B, layer: 1, pos: 17
type: B, layer: 1, pos: 41
type: B, layer: 1, pos: 0
type: B, layer: 1, pos: 21
type: B, layer: 1, pos: 36

Time for candidate selection: 0.05 seconds

### Candidate
type: B, layer: 1, pos: 13

### Candidate
type: B, layer: 1, pos: 16

### Candidate
type: B, layer: 1, pos: 9

## Relational analysis of NS_A1_B2_A1_B1_A1_B1

### Relational analysis result of NS_A1_B2_A1_B1_A1_B1
Status: Status.VERIFIED
Output dim: 0, lower bound: -27.7369988, upper bound: 27.7491186
time: 0.59 seconds

## Relational analysis of NS_A1_B2_A1_B1_A1_B2

### Relational analysis result of NS_A1_B2_A1_B1_A1_B2
Status: Status.UNKNOWN
Output dim: 0, lower bound: -27.7369988, upper bound: 27.8079817
time: 0.65 seconds

## BFS NS instance: NS_A1_B2_A1_B1_A2

### Backsubstitution after applying NS history:
0: -4.0236449, 14.8443575, -5.3676920, 19.4284458, -23.4520855, 20.2120476
1: -6.7107162, 15.0167494, -8.8535709, 19.6024017, -26.3131180, 23.8703194
2: -5.3908873, 16.3171444, -7.1736655, 21.2490845, -26.6399727, 23.4908104
3: -5.9503188, 22.5346546, -7.7958689, 29.3602962, -35.3106155, 30.3305206
4: -4.7566547, 20.9033737, -6.3186235, 27.2952156, -32.0518684, 27.2219963

Time for backsubstitution: 0.74 seconds

### NS candidates at layer 1
type: B, layer: 1, pos: 13
type: B, layer: 1, pos: 16
type: B, layer: 1, pos: 9
type: B, layer: 1, pos: 22
type: B, layer: 1, pos: 10
type: B, layer: 1, pos: 17
type: B, layer: 1, pos: 41
type: B, layer: 1, pos: 0
type: B, layer: 1, pos: 21
type: B, layer: 1, pos: 36

Time for candidate selection: 0.05 seconds

### Candidate
type: B, layer: 1, pos: 13

### Candidate
type: B, layer: 1, pos: 16

### Candidate
type: B, layer: 1, pos: 9

## Relational analysis of NS_A1_B2_A1_B1_A2_B1

### Relational analysis result of NS_A1_B2_A1_B1_A2_B1
Status: Status.VERIFIED
Output dim: 0, lower bound: -27.7369988, upper bound: 27.7491237
time: 0.44 seconds

## Relational analysis of NS_A1_B2_A1_B1_A2_B2

### Relational analysis result of NS_A1_B2_A1_B1_A2_B2
Status: Status.UNKNOWN
Output dim: 0, lower bound: -27.7369988, upper bound: 27.8161528
time: 0.61 seconds

## BFS NS instance: NS_A1_B2_A1_B2_A1

### Backsubstitution after applying NS history:
0: -3.9672580, 14.7736979, -6.1593466, 21.5621471, -25.5294037, 20.9330368
1: -6.6120090, 14.9280148, -10.1067486, 21.8576832, -28.4696922, 25.0347633
2: -5.3131099, 16.2373619, -8.2562380, 23.6580582, -28.9711685, 24.4935970
3: -5.8528652, 22.4279957, -8.9429302, 32.4946060, -38.3474731, 31.3709183
4: -4.6880665, 20.7485371, -7.3059959, 30.5865879, -35.2746544, 28.0545330

Time for backsubstitution: 0.75 seconds

### NS candidates at layer 1
type: B, layer: 1, pos: 13
type: B, layer: 1, pos: 16
type: B, layer: 1, pos: 9
type: B, layer: 1, pos: 22
type: B, layer: 1, pos: 41
type: B, layer: 1, pos: 10
type: B, layer: 1, pos: 17
type: B, layer: 1, pos: 0
type: B, layer: 1, pos: 21
type: B, layer: 1, pos: 36

Time for candidate selection: 0.05 seconds

### Candidate
type: B, layer: 1, pos: 13

### Candidate
type: B, layer: 1, pos: 16

### Candidate
type: B, layer: 1, pos: 9

## Relational analysis of NS_A1_B2_A1_B2_A1_B1
Optimization infeasible because this subproblem isn't reachable.

### Candidate
type: B, layer: 1, pos: 22

## Relational analysis of NS_A1_B2_A1_B2_A1_B1
Optimization infeasible because this subproblem isn't reachable.

### Candidate
type: B, layer: 1, pos: 41

## Relational analysis of NS_A1_B2_A1_B2_A1_B1

### Relational analysis result of NS_A1_B2_A1_B2_A1_B1
Status: Status.VERIFIED
Output dim: 0, lower bound: -27.1658621, upper bound: 27.1400320
time: 0.74 seconds

## Relational analysis of NS_A1_B2_A1_B2_A1_B2

### Relational analysis result of NS_A1_B2_A1_B2_A1_B2
Status: Status.VERIFIED
Output dim: 0, lower bound: -27.1292155, upper bound: 27.0607941
time: 1.04 seconds

## BFS NS instance: NS_A1_B2_A1_B2_A2

### Backsubstitution after applying NS history:
0: -4.0236449, 14.8443575, -6.3037281, 22.0140495, -26.0376911, 21.1480808
1: -6.7107162, 15.0167494, -10.3375835, 22.3265495, -29.0372658, 25.3543320
2: -5.3908873, 16.3171444, -8.4509039, 24.1487713, -29.5396576, 24.7680473
3: -5.9503188, 22.5346546, -9.1481276, 33.1661034, -39.1164207, 31.6827812
4: -4.7566547, 20.9033737, -7.4770923, 31.2366295, -35.9932785, 28.3804665

Time for backsubstitution: 0.75 seconds

### NS candidates at layer 1
type: B, layer: 1, pos: 13
type: B, layer: 1, pos: 16
type: B, layer: 1, pos: 9
type: B, layer: 1, pos: 22
type: B, layer: 1, pos: 41
type: B, layer: 1, pos: 10
type: B, layer: 1, pos: 0
type: B, layer: 1, pos: 17
type: B, layer: 1, pos: 21
type: B, layer: 1, pos: 36

Time for candidate selection: 0.05 seconds

### Candidate
type: B, layer: 1, pos: 13

### Candidate
type: B, layer: 1, pos: 16

### Candidate
type: B, layer: 1, pos: 9

## Relational analysis of NS_A1_B2_A1_B2_A2_B1
Optimization infeasible because this subproblem isn't reachable.

### Candidate
type: B, layer: 1, pos: 22

## Relational analysis of NS_A1_B2_A1_B2_A2_B1
Optimization infeasible because this subproblem isn't reachable.

### Candidate
type: B, layer: 1, pos: 41

## Relational analysis of NS_A1_B2_A1_B2_A2_B1

### Relational analysis result of NS_A1_B2_A1_B2_A2_B1
Status: Status.VERIFIED
Output dim: 0, lower bound: -27.3466310, upper bound: 27.4313909
time: 0.86 seconds

## Relational analysis of NS_A1_B2_A1_B2_A2_B2

### Relational analysis result of NS_A1_B2_A1_B2_A2_B2
Status: Status.VERIFIED
Output dim: 0, lower bound: -27.1567376, upper bound: 27.0754124
time: 0.42 seconds

## BFS NS instance: NS_A1_B2_A2_B1_A2

### Backsubstitution after applying NS history:
0: -5.1221609, 18.6184425, -5.3676920, 19.4284458, -24.5506058, 23.9861336
1: -8.4564943, 18.7807064, -8.8535709, 19.6024017, -28.0588951, 27.6342754
2: -6.8467531, 20.3681755, -7.1736655, 21.2490845, -28.0958347, 27.5418415
3: -7.4440994, 28.1251602, -7.7958689, 29.3602962, -36.8043938, 35.9210281
4: -6.0290127, 26.1052856, -6.3186235, 27.2952156, -33.3242264, 32.4239082

Time for backsubstitution: 0.75 seconds

### NS candidates at layer 1
type: B, layer: 1, pos: 13
type: B, layer: 1, pos: 16
type: B, layer: 1, pos: 9
type: B, layer: 1, pos: 22
type: B, layer: 1, pos: 10
type: B, layer: 1, pos: 17
type: B, layer: 1, pos: 41
type: B, layer: 1, pos: 0
type: B, layer: 1, pos: 21
type: B, layer: 1, pos: 36

Time for candidate selection: 0.05 seconds

### Candidate
type: B, layer: 1, pos: 13

### Candidate
type: B, layer: 1, pos: 16

### Candidate
type: B, layer: 1, pos: 9

## Relational analysis of NS_A1_B2_A2_B1_A2_B1

### Relational analysis result of NS_A1_B2_A2_B1_A2_B1
Status: Status.VERIFIED
Output dim: 0, lower bound: -27.6794362, upper bound: 27.6810159
time: 0.59 seconds

## Relational analysis of NS_A1_B2_A2_B1_A2_B2

### Relational analysis result of NS_A1_B2_A2_B1_A2_B2
Status: Status.UNKNOWN
Output dim: 0, lower bound: -27.6794362, upper bound: 27.8045752
time: 1.10 seconds

## BFS NS instance: NS_A1_B2_A2_B2_A2

### Backsubstitution after applying NS history:
0: -5.1221609, 18.6184425, -6.3037281, 22.0140495, -27.1362114, 24.9221687
1: -8.4564943, 18.7807064, -10.3375835, 22.3265495, -30.7830429, 29.1182899
2: -6.8467531, 20.3681755, -8.4509039, 24.1487713, -30.9955254, 28.8190804
3: -7.4440994, 28.1251602, -9.1481276, 33.1661034, -40.6102028, 37.2732849
4: -6.0290127, 26.1052856, -7.4770923, 31.2366295, -37.2656403, 33.5823784

Time for backsubstitution: 0.75 seconds

### NS candidates at layer 1
type: B, layer: 1, pos: 13
type: B, layer: 1, pos: 16
type: B, layer: 1, pos: 9
type: B, layer: 1, pos: 22
type: B, layer: 1, pos: 41
type: B, layer: 1, pos: 10
type: B, layer: 1, pos: 0
type: B, layer: 1, pos: 17
type: B, layer: 1, pos: 21
type: B, layer: 1, pos: 36

Time for candidate selection: 0.05 seconds

### Candidate
type: B, layer: 1, pos: 13

### Candidate
type: B, layer: 1, pos: 16

### Candidate
type: B, layer: 1, pos: 9

## Relational analysis of NS_A1_B2_A2_B2_A2_B1
Optimization infeasible because this subproblem isn't reachable.

### Candidate
type: B, layer: 1, pos: 22

## Relational analysis of NS_A1_B2_A2_B2_A2_B1
Optimization infeasible because this subproblem isn't reachable.

### Candidate
type: B, layer: 1, pos: 41

## Relational analysis of NS_A1_B2_A2_B2_A2_B1

### Relational analysis result of NS_A1_B2_A2_B2_A2_B1
Status: Status.VERIFIED
Output dim: 0, lower bound: -27.3630762, upper bound: 27.4417000
time: 1.02 seconds

## Relational analysis of NS_A1_B2_A2_B2_A2_B2

### Relational analysis result of NS_A1_B2_A2_B2_A2_B2
Status: Status.VERIFIED
Output dim: 0, lower bound: -27.1625318, upper bound: 27.0791871
time: 0.60 seconds

## BFS NS instance: NS_A2_B1_A1_B1_A1

### Backsubstitution after applying NS history:
0: -5.3486147, 18.6461754, -4.1239586, 15.1485586, -20.4971714, 22.7701321
1: -8.8365345, 18.9885597, -6.8710823, 15.3317356, -24.1682701, 25.8596420
2: -7.1932240, 20.5773335, -5.5255399, 16.6497955, -23.8430195, 26.1028728
3: -7.8570271, 28.2433796, -6.0924091, 22.9905643, -30.8475876, 34.3357887
4: -6.3842969, 26.6930981, -4.8740869, 21.3502178, -27.7345104, 31.5671844

Time for backsubstitution: 0.75 seconds

### NS candidates at layer 1
type: B, layer: 1, pos: 13
type: B, layer: 1, pos: 16
type: B, layer: 1, pos: 9
type: B, layer: 1, pos: 22
type: B, layer: 1, pos: 17
type: B, layer: 1, pos: 10
type: B, layer: 1, pos: 21
type: B, layer: 1, pos: 41
type: B, layer: 1, pos: 0
type: B, layer: 1, pos: 36

Time for candidate selection: 0.05 seconds

### Candidate
type: B, layer: 1, pos: 13

### Candidate
type: B, layer: 1, pos: 16

### Candidate
type: B, layer: 1, pos: 9

### Candidate
type: B, layer: 1, pos: 22

### Candidate
type: B, layer: 1, pos: 17

## Relational analysis of NS_A2_B1_A1_B1_A1_B1
Optimization infeasible because this subproblem isn't reachable.

### Candidate
type: B, layer: 1, pos: 10

## Relational analysis of NS_A2_B1_A1_B1_A1_B1

### Relational analysis result of NS_A2_B1_A1_B1_A1_B1
Status: Status.UNKNOWN
Output dim: 0, lower bound: -27.8222555, upper bound: 27.8225243
time: 0.88 seconds

## Relational analysis of NS_A2_B1_A1_B1_A1_B2

### Relational analysis result of NS_A2_B1_A1_B1_A1_B2
Status: Status.UNKNOWN
Output dim: 0, lower bound: -27.8222555, upper bound: 27.8234250
time: 0.63 seconds

## BFS NS instance: NS_A2_B1_A1_B1_A2

### Backsubstitution after applying NS history:
0: -6.2936206, 21.7869453, -4.0356045, 14.8624039, -21.1560230, 25.8225498
1: -10.3439865, 22.1925621, -6.7305045, 15.0457191, -25.3897057, 28.9230671
2: -8.4501743, 23.9575081, -5.4062943, 16.3423939, -24.7925625, 29.3638000
3: -9.1837044, 32.9127502, -5.9694557, 22.5781479, -31.7618523, 38.8822060
4: -7.4634824, 31.1625271, -4.7701931, 20.9538345, -28.4173164, 35.9327126

Time for backsubstitution: 0.76 seconds

### NS candidates at layer 1
type: B, layer: 1, pos: 13
type: B, layer: 1, pos: 16
type: B, layer: 1, pos: 9
type: B, layer: 1, pos: 22
type: B, layer: 1, pos: 17
type: B, layer: 1, pos: 10
type: B, layer: 1, pos: 21
type: B, layer: 1, pos: 41
type: B, layer: 1, pos: 0
type: B, layer: 1, pos: 36

Time for candidate selection: 0.05 seconds

### Candidate
type: B, layer: 1, pos: 13

### Candidate
type: B, layer: 1, pos: 16

### Candidate
type: B, layer: 1, pos: 9

### Candidate
type: B, layer: 1, pos: 22

### Candidate
type: B, layer: 1, pos: 17

## Relational analysis of NS_A2_B1_A1_B1_A2_B1
Optimization infeasible because this subproblem isn't reachable.

### Candidate
type: B, layer: 1, pos: 10

## Relational analysis of NS_A2_B1_A1_B1_A2_B1

### Relational analysis result of NS_A2_B1_A1_B1_A2_B1
Status: Status.UNKNOWN
Output dim: 0, lower bound: -27.8222555, upper bound: 27.8225243
time: 0.66 seconds

## Relational analysis of NS_A2_B1_A1_B1_A2_B2

### Relational analysis result of NS_A2_B1_A1_B1_A2_B2
Status: Status.UNKNOWN
Output dim: 0, lower bound: -27.8222555, upper bound: 27.8234250
time: 0.86 seconds

## BFS NS instance: NS_A2_B1_A1_B2_A1

### Backsubstitution after applying NS history:
0: -5.3486147, 18.6461754, -5.2290773, 18.9489193, -24.2975349, 23.8752480
1: -8.8365345, 18.9885597, -8.6279545, 19.1228676, -27.9594021, 27.6165142
2: -7.1932240, 20.5773335, -6.9899049, 20.7296162, -27.9228401, 27.5672359
3: -7.8570271, 28.2433796, -7.5971055, 28.6222916, -36.4793167, 35.8404846
4: -6.3842969, 26.6930981, -6.1555519, 26.5859127, -32.9702072, 32.8486481

Time for backsubstitution: 0.77 seconds

### NS candidates at layer 1
type: B, layer: 1, pos: 13
type: B, layer: 1, pos: 16
type: B, layer: 1, pos: 9
type: B, layer: 1, pos: 22
type: B, layer: 1, pos: 10
type: B, layer: 1, pos: 17
type: B, layer: 1, pos: 41
type: B, layer: 1, pos: 21
type: B, layer: 1, pos: 0
type: B, layer: 1, pos: 36

Time for candidate selection: 0.05 seconds

### Candidate
type: B, layer: 1, pos: 13

### Candidate
type: B, layer: 1, pos: 16

### Candidate
type: B, layer: 1, pos: 9

### Candidate
type: B, layer: 1, pos: 22

### Candidate
type: B, layer: 1, pos: 10

## Relational analysis of NS_A2_B1_A1_B2_A1_B1

### Relational analysis result of NS_A2_B1_A1_B2_A1_B1
Status: Status.VERIFIED
Output dim: 0, lower bound: -27.7530056, upper bound: 27.7609829
time: 0.90 seconds

## Relational analysis of NS_A2_B1_A1_B2_A1_B2

### Relational analysis result of NS_A2_B1_A1_B2_A1_B2
Status: Status.VERIFIED
Output dim: 0, lower bound: -27.7530056, upper bound: 27.7609829
time: 0.48 seconds

## BFS NS instance: NS_A2_B1_A1_B2_A2

### Backsubstitution after applying NS history:
0: -6.2936206, 21.7869453, -5.1042194, 18.5488262, -24.8424473, 26.8911648
1: -10.3439865, 22.1925621, -8.4335775, 18.7205429, -29.0645294, 30.6261368
2: -8.4501743, 23.9575081, -6.8225837, 20.3026943, -28.7528648, 30.7800922
3: -9.1837044, 32.9127502, -7.4257655, 28.0532227, -37.2369270, 40.3385124
4: -7.4634824, 31.1625271, -6.0125823, 26.0352364, -33.4987183, 37.1751099

Time for backsubstitution: 0.77 seconds

### NS candidates at layer 1
type: B, layer: 1, pos: 13
type: B, layer: 1, pos: 16
type: B, layer: 1, pos: 9
type: B, layer: 1, pos: 22
type: B, layer: 1, pos: 10
type: B, layer: 1, pos: 17
type: B, layer: 1, pos: 41
type: B, layer: 1, pos: 21
type: B, layer: 1, pos: 0
type: B, layer: 1, pos: 36

Time for candidate selection: 0.05 seconds

### Candidate
type: B, layer: 1, pos: 13

### Candidate
type: B, layer: 1, pos: 16

### Candidate
type: B, layer: 1, pos: 9

### Candidate
type: B, layer: 1, pos: 22

### Candidate
type: B, layer: 1, pos: 10

## Relational analysis of NS_A2_B1_A1_B2_A2_B1

### Relational analysis result of NS_A2_B1_A1_B2_A2_B1
Status: Status.VERIFIED
Output dim: 0, lower bound: -27.7530056, upper bound: 27.7609829
time: 0.54 seconds

## Relational analysis of NS_A2_B1_A1_B2_A2_B2

### Relational analysis result of NS_A2_B1_A1_B2_A2_B2
Status: Status.UNKNOWN
Output dim: 0, lower bound: -27.7530056, upper bound: 27.8100959
time: 0.87 seconds

## Summary of splitting at layer (split count: 5)
- Time for NS candidates: 2.25 seconds
NS_A1_B1_A1_B1_A1_B1, status: Status.UNKNOWN, split count: 6, time: 2.25
Output dim: 0, lower bound: -27.8510354, upper bound: 27.8506168
NS_A1_B1_A1_B1_A1_B2, status: Status.UNKNOWN, split count: 6, time: 2.25
Output dim: 0, lower bound: -27.8510354, upper bound: 27.8503726
NS_A1_B1_A1_B1_A2_B1, status: Status.UNKNOWN, split count: 6, time: 2.25
Output dim: 0, lower bound: -27.8521223, upper bound: 27.8521345
NS_A1_B1_A1_B1_A2_B2, status: Status.UNKNOWN, split count: 6, time: 2.25
Output dim: 0, lower bound: -27.8521223, upper bound: 27.8521345
NS_A1_B1_A1_B2_A1_B1, status: Status.UNKNOWN, split count: 6, time: 2.25
Output dim: 0, lower bound: -27.8303058, upper bound: 27.8288729
NS_A1_B1_A1_B2_A1_B2, status: Status.UNKNOWN, split count: 6, time: 2.25
Output dim: 0, lower bound: -27.8234250, upper bound: 27.8218229
NS_A1_B1_A1_B2_A2_B1, status: Status.UNKNOWN, split count: 6, time: 2.25
Output dim: 0, lower bound: -27.7755511, upper bound: 27.7687815
NS_A1_B1_A1_B2_A2_B2, status: Status.UNKNOWN, split count: 6, time: 2.25
Output dim: 0, lower bound: -27.7755511, upper bound: 27.8472336
NS_A1_B1_A2_B1_A2_B1, status: Status.UNKNOWN, split count: 6, time: 2.25
Output dim: 0, lower bound: -27.8079817, upper bound: 27.8050375
NS_A1_B1_A2_B1_A2_B2, status: Status.UNKNOWN, split count: 6, time: 2.25
Output dim: 0, lower bound: -27.8079817, upper bound: 27.8144338
NS_A1_B1_A2_B2_A2_B1, status: Status.UNKNOWN, split count: 6, time: 2.25
Output dim: 0, lower bound: -27.8118463, upper bound: 27.8079184
NS_A1_B1_A2_B2_A2_B2, status: Status.UNKNOWN, split count: 6, time: 2.25
Output dim: 0, lower bound: -27.8100959, upper bound: 27.8066735
NS_A1_B2_A1_B1_A1_B1, status: Status.VERIFIED, split count: 6, time: 2.25
Output dim: 0, lower bound: -27.7369988, upper bound: 27.7491186
NS_A1_B2_A1_B1_A1_B2, status: Status.UNKNOWN, split count: 6, time: 2.25
Output dim: 0, lower bound: -27.7369988, upper bound: 27.8079817
NS_A1_B2_A1_B1_A2_B1, status: Status.VERIFIED, split count: 6, time: 2.25
Output dim: 0, lower bound: -27.7369988, upper bound: 27.7491237
NS_A1_B2_A1_B1_A2_B2, status: Status.UNKNOWN, split count: 6, time: 2.25
Output dim: 0, lower bound: -27.7369988, upper bound: 27.8161528
NS_A1_B2_A1_B2_A1_B1, status: Status.VERIFIED, split count: 6, time: 2.25
Output dim: 0, lower bound: -27.1658621, upper bound: 27.1400320
NS_A1_B2_A1_B2_A1_B2, status: Status.VERIFIED, split count: 6, time: 2.25
Output dim: 0, lower bound: -27.1292155, upper bound: 27.0607941
NS_A1_B2_A1_B2_A2_B1, status: Status.VERIFIED, split count: 6, time: 2.25
Output dim: 0, lower bound: -27.3466310, upper bound: 27.4313909
NS_A1_B2_A1_B2_A2_B2, status: Status.VERIFIED, split count: 6, time: 2.25
Output dim: 0, lower bound: -27.1567376, upper bound: 27.0754124
NS_A1_B2_A2_B1_A2_B1, status: Status.VERIFIED, split count: 6, time: 2.25
Output dim: 0, lower bound: -27.6794362, upper bound: 27.6810159
NS_A1_B2_A2_B1_A2_B2, status: Status.UNKNOWN, split count: 6, time: 2.25
Output dim: 0, lower bound: -27.6794362, upper bound: 27.8045752
NS_A1_B2_A2_B2_A2_B1, status: Status.VERIFIED, split count: 6, time: 2.25
Output dim: 0, lower bound: -27.3630762, upper bound: 27.4417000
NS_A1_B2_A2_B2_A2_B2, status: Status.VERIFIED, split count: 6, time: 2.25
Output dim: 0, lower bound: -27.1625318, upper bound: 27.0791871
NS_A2_B1_A1_B1_A1_B1, status: Status.UNKNOWN, split count: 6, time: 2.25
Output dim: 0, lower bound: -27.8222555, upper bound: 27.8225243
NS_A2_B1_A1_B1_A1_B2, status: Status.UNKNOWN, split count: 6, time: 2.25
Output dim: 0, lower bound: -27.8222555, upper bound: 27.8234250
NS_A2_B1_A1_B1_A2_B1, status: Status.UNKNOWN, split count: 6, time: 2.25
Output dim: 0, lower bound: -27.8222555, upper bound: 27.8225243
NS_A2_B1_A1_B1_A2_B2, status: Status.UNKNOWN, split count: 6, time: 2.25
Output dim: 0, lower bound: -27.8222555, upper bound: 27.8234250
NS_A2_B1_A1_B2_A1_B1, status: Status.VERIFIED, split count: 6, time: 2.25
Output dim: 0, lower bound: -27.7530056, upper bound: 27.7609829
NS_A2_B1_A1_B2_A1_B2, status: Status.VERIFIED, split count: 6, time: 2.25
Output dim: 0, lower bound: -27.7530056, upper bound: 27.7609829
NS_A2_B1_A1_B2_A2_B1, status: Status.VERIFIED, split count: 6, time: 2.25
Output dim: 0, lower bound: -27.7530056, upper bound: 27.7609829
NS_A2_B1_A1_B2_A2_B2, status: Status.UNKNOWN, split count: 6, time: 2.25
Output dim: 0, lower bound: -27.7530056, upper bound: 27.8100959

## BFS NS instance: NS_A1_B1_A1_B1_A1_B1

### Backsubstitution after applying NS history:
0: -3.6825645, 13.8960981, -3.5030324, 13.2311544, -16.9137192, 17.3991299
1: -6.1600418, 14.0308790, -5.8693361, 13.3828163, -19.5428581, 19.9002151
2: -4.9269824, 15.2807989, -4.6870651, 14.5938587, -19.5208416, 19.9678612
3: -5.4549265, 21.1124554, -5.2032495, 20.0847931, -25.5397186, 26.3157043
4: -4.3532619, 19.4837170, -4.1517558, 18.5818901, -22.9351521, 23.6354713

Time for backsubstitution: 0.75 seconds

### NS candidates at layer 1
type: A, layer: 1, pos: 13
type: A, layer: 1, pos: 16
type: A, layer: 1, pos: 22
type: A, layer: 1, pos: 10
type: A, layer: 1, pos: 17
type: A, layer: 1, pos: 21
type: A, layer: 1, pos: 41
type: A, layer: 1, pos: 36
type: A, layer: 1, pos: 0

Time for candidate selection: 0.05 seconds

### Candidate
type: A, layer: 1, pos: 13

### Candidate
type: A, layer: 1, pos: 16

### Candidate
type: A, layer: 1, pos: 22

## Relational analysis of NS_A1_B1_A1_B1_A1_B1_A1

### Relational analysis result of NS_A1_B1_A1_B1_A1_B1_A1
Status: Status.UNKNOWN
Output dim: 0, lower bound: -27.8510354, upper bound: 27.8503726
time: 0.63 seconds

## Relational analysis of NS_A1_B1_A1_B1_A1_B1_A2

### Relational analysis result of NS_A1_B1_A1_B1_A1_B1_A2
Status: Status.UNKNOWN
Output dim: 0, lower bound: -27.8510354, upper bound: 27.8503726
time: 0.61 seconds

## BFS NS instance: NS_A1_B1_A1_B1_A1_B2

### Backsubstitution after applying NS history:
0: -3.9672580, 14.7736979, -3.8925047, 14.4426603, -18.4099140, 18.6661987
1: -6.6120090, 14.9280148, -6.5009913, 14.6081524, -21.2201595, 21.4290066
2: -5.3131099, 16.2373619, -5.2121353, 15.8789501, -21.1920605, 21.4494934
3: -5.8528652, 22.4279957, -5.7661185, 21.9363670, -27.7892303, 28.1941128
4: -4.6880665, 20.7485371, -4.5990372, 20.3096542, -24.9977207, 25.3475742

Time for backsubstitution: 0.75 seconds

### NS candidates at layer 1
type: A, layer: 1, pos: 13
type: A, layer: 1, pos: 16
type: A, layer: 1, pos: 22
type: A, layer: 1, pos: 10
type: A, layer: 1, pos: 17
type: A, layer: 1, pos: 21
type: A, layer: 1, pos: 41
type: A, layer: 1, pos: 36
type: A, layer: 1, pos: 0

Time for candidate selection: 0.05 seconds

### Candidate
type: A, layer: 1, pos: 13

### Candidate
type: A, layer: 1, pos: 16

### Candidate
type: A, layer: 1, pos: 22

## Relational analysis of NS_A1_B1_A1_B1_A1_B2_A1

### Relational analysis result of NS_A1_B1_A1_B1_A1_B2_A1
Status: Status.UNKNOWN
Output dim: 0, lower bound: -27.8510354, upper bound: 27.8503726
time: 0.69 seconds

## Relational analysis of NS_A1_B1_A1_B1_A1_B2_A2

### Relational analysis result of NS_A1_B1_A1_B1_A1_B2_A2
Status: Status.UNKNOWN
Output dim: 0, lower bound: -27.8510354, upper bound: 27.8503726
time: 0.68 seconds

## BFS NS instance: NS_A1_B1_A1_B1_A2_B1

### Backsubstitution after applying NS history:
0: -4.0236449, 14.8443575, -3.9672580, 14.7736979, -18.7973366, 18.8116150
1: -6.7107162, 15.0167494, -6.6120090, 14.9280148, -21.6387310, 21.6287575
2: -5.3908873, 16.3171444, -5.3131099, 16.2373619, -21.6282501, 21.6302547
3: -5.9503188, 22.5346546, -5.8528652, 22.4279957, -28.3783150, 28.3875179
4: -4.7566547, 20.9033737, -4.6880665, 20.7485371, -25.5051918, 25.5914402

Time for backsubstitution: 0.75 seconds

### NS candidates at layer 1
type: A, layer: 1, pos: 13
type: A, layer: 1, pos: 16
type: A, layer: 1, pos: 22
type: A, layer: 1, pos: 17
type: A, layer: 1, pos: 10
type: A, layer: 1, pos: 21
type: A, layer: 1, pos: 41
type: A, layer: 1, pos: 0
type: A, layer: 1, pos: 36

Time for candidate selection: 0.05 seconds

### Candidate
type: A, layer: 1, pos: 13

### Candidate
type: A, layer: 1, pos: 16

### Candidate
type: A, layer: 1, pos: 22

## Relational analysis of NS_A1_B1_A1_B1_A2_B1_A1

### Relational analysis result of NS_A1_B1_A1_B1_A2_B1_A1
Status: Status.UNKNOWN
Output dim: 0, lower bound: -27.8506168, upper bound: 27.8510354
time: 0.81 seconds

## Relational analysis of NS_A1_B1_A1_B1_A2_B1_A2

### Relational analysis result of NS_A1_B1_A1_B1_A2_B1_A2
Status: Status.UNKNOWN
Output dim: 0, lower bound: -27.8503726, upper bound: 27.8510354
time: 0.55 seconds

## BFS NS instance: NS_A1_B1_A1_B1_A2_B2

### Backsubstitution after applying NS history:
0: -4.0236449, 14.8443575, -4.0236449, 14.8443575, -18.8679943, 18.8679943
1: -6.7107162, 15.0167494, -6.7107162, 15.0167494, -21.7274647, 21.7274628
2: -5.3908873, 16.3171444, -5.3908873, 16.3171444, -21.7080307, 21.7080307
3: -5.9503188, 22.5346546, -5.9503188, 22.5346546, -28.4849739, 28.4849739
4: -4.7566547, 20.9033737, -4.7566547, 20.9033737, -25.6600285, 25.6600285

Time for backsubstitution: 0.76 seconds

### NS candidates at layer 1
type: A, layer: 1, pos: 13
type: A, layer: 1, pos: 16
type: A, layer: 1, pos: 22
type: A, layer: 1, pos: 17
type: A, layer: 1, pos: 10
type: A, layer: 1, pos: 21
type: A, layer: 1, pos: 41
type: A, layer: 1, pos: 0
type: A, layer: 1, pos: 36

Time for candidate selection: 0.05 seconds

### Candidate
type: A, layer: 1, pos: 13

### Candidate
type: A, layer: 1, pos: 16

### Candidate
type: A, layer: 1, pos: 22

## Relational analysis of NS_A1_B1_A1_B1_A2_B2_A1

### Relational analysis result of NS_A1_B1_A1_B1_A2_B2_A1
Status: Status.UNKNOWN
Output dim: 0, lower bound: -27.8506168, upper bound: 27.8518903
time: 0.80 seconds

## Relational analysis of NS_A1_B1_A1_B1_A2_B2_A2

### Relational analysis result of NS_A1_B1_A1_B1_A2_B2_A2
Status: Status.UNKNOWN
Output dim: 0, lower bound: -27.8503726, upper bound: 27.8518903
time: 0.46 seconds

## BFS NS instance: NS_A1_B1_A1_B2_A1_B1

### Backsubstitution after applying NS history:
0: -3.9672580, 14.7736979, -5.2065444, 18.1934853, -22.1607418, 19.9802341
1: -6.6120090, 14.9280148, -8.6081333, 18.5242100, -25.1362190, 23.5361481
2: -5.3131099, 16.2373619, -7.0009108, 20.0829887, -25.3960991, 23.2382679
3: -5.8528652, 22.4279957, -7.6543884, 27.5664101, -33.4192734, 30.0823784
4: -4.6880665, 20.7485371, -6.2131224, 26.0368614, -30.7249260, 26.9616585

Time for backsubstitution: 0.76 seconds

### NS candidates at layer 1
type: A, layer: 1, pos: 13
type: A, layer: 1, pos: 16
type: A, layer: 1, pos: 22
type: A, layer: 1, pos: 10
type: A, layer: 1, pos: 17
type: A, layer: 1, pos: 21
type: A, layer: 1, pos: 41
type: A, layer: 1, pos: 36
type: A, layer: 1, pos: 0

Time for candidate selection: 0.05 seconds

### Candidate
type: A, layer: 1, pos: 13

### Candidate
type: A, layer: 1, pos: 16

### Candidate
type: A, layer: 1, pos: 22

### Candidate
type: A, layer: 1, pos: 10

## Relational analysis of NS_A1_B1_A1_B2_A1_B1_A1

### Relational analysis result of NS_A1_B1_A1_B2_A1_B1_A1
Status: Status.UNKNOWN
Output dim: 0, lower bound: -27.8218115, upper bound: 27.8210412
time: 0.48 seconds

## Relational analysis of NS_A1_B1_A1_B2_A1_B1_A2

### Relational analysis result of NS_A1_B1_A1_B2_A1_B1_A2
Status: Status.UNKNOWN
Output dim: 0, lower bound: -27.8218115, upper bound: 27.8218229
time: 0.50 seconds

## BFS NS instance: NS_A1_B1_A1_B2_A1_B2

### Backsubstitution after applying NS history:
0: -3.8808694, 14.4860821, -6.0672770, 21.1454773, -25.0263462, 20.5533562
1: -6.4740901, 14.6423807, -9.9897528, 21.4899788, -27.9640675, 24.6321335
2: -5.1966004, 15.9301434, -8.1532803, 23.2542458, -28.4508438, 24.0834236
3: -5.7323523, 22.0144463, -8.8703241, 31.9812069, -37.7135582, 30.8847656
4: -4.5869827, 20.3543396, -7.2070513, 30.2580280, -34.8450089, 27.5613899

Time for backsubstitution: 0.78 seconds

### NS candidates at layer 1
type: A, layer: 1, pos: 13
type: A, layer: 1, pos: 16
type: A, layer: 1, pos: 22
type: A, layer: 1, pos: 17
type: A, layer: 1, pos: 10
type: A, layer: 1, pos: 21
type: A, layer: 1, pos: 41
type: A, layer: 1, pos: 36
type: A, layer: 1, pos: 0

Time for candidate selection: 0.05 seconds

### Candidate
type: A, layer: 1, pos: 13

### Candidate
type: A, layer: 1, pos: 16

### Candidate
type: A, layer: 1, pos: 22

### Candidate
type: A, layer: 1, pos: 17

## Relational analysis of NS_A1_B1_A1_B2_A1_B2_A1
Optimization infeasible because this subproblem isn't reachable.

### Candidate
type: A, layer: 1, pos: 10

## Relational analysis of NS_A1_B1_A1_B2_A1_B2_A1

### Relational analysis result of NS_A1_B1_A1_B2_A1_B2_A1
Status: Status.UNKNOWN
Output dim: 0, lower bound: -27.8218115, upper bound: 27.8210412
time: 0.66 seconds

## Relational analysis of NS_A1_B1_A1_B2_A1_B2_A2

### Relational analysis result of NS_A1_B1_A1_B2_A1_B2_A2
Status: Status.UNKNOWN
Output dim: 0, lower bound: -27.8218115, upper bound: 27.8218229
time: 0.76 seconds

## BFS NS instance: NS_A1_B1_A1_B2_A2_B1

### Backsubstitution after applying NS history:
0: -4.0236449, 14.8443575, -5.3342867, 18.5256844, -22.5493240, 20.1786442
1: -6.7107162, 15.0167494, -8.7927580, 18.8779011, -25.5886154, 23.8095055
2: -5.3908873, 16.3171444, -7.1675220, 20.4197502, -25.8106384, 23.4846668
3: -5.9503188, 22.5346546, -7.8134527, 28.0398369, -33.9901543, 30.3481026
4: -4.7566547, 20.9033737, -6.3561907, 26.5359650, -31.2926197, 27.2595634

Time for backsubstitution: 0.77 seconds

### NS candidates at layer 1
type: A, layer: 1, pos: 13
type: A, layer: 1, pos: 16
type: A, layer: 1, pos: 22
type: A, layer: 1, pos: 17
type: A, layer: 1, pos: 10
type: A, layer: 1, pos: 21
type: A, layer: 1, pos: 41
type: A, layer: 1, pos: 0
type: A, layer: 1, pos: 36

Time for candidate selection: 0.05 seconds

### Candidate
type: A, layer: 1, pos: 13

### Candidate
type: A, layer: 1, pos: 16

### Candidate
type: A, layer: 1, pos: 22

### Candidate
type: A, layer: 1, pos: 17

## Relational analysis of NS_A1_B1_A1_B2_A2_B1_A1

### Relational analysis result of NS_A1_B1_A1_B2_A2_B1_A1
Status: Status.VERIFIED
Output dim: 0, lower bound: -27.6112512, upper bound: 27.6228780
time: 0.70 seconds

## Relational analysis of NS_A1_B1_A1_B2_A2_B1_A2

### Relational analysis result of NS_A1_B1_A1_B2_A2_B1_A2
Status: Status.UNKNOWN
Output dim: 0, lower bound: -27.7755511, upper bound: 27.7687815
time: 1.14 seconds

## BFS NS instance: NS_A1_B1_A1_B2_A2_B2

### Backsubstitution after applying NS history:
0: -4.0236449, 14.8443575, -5.3180194, 18.5575352, -22.5811729, 20.1623745
1: -6.7107162, 15.0167494, -8.7866850, 18.8954010, -25.6061134, 23.8034344
2: -5.3908873, 16.3171444, -7.1526365, 20.4805336, -25.8714218, 23.4697762
3: -5.9503188, 22.5346546, -7.8126078, 28.1088581, -34.0591736, 30.3472576
4: -4.7566547, 20.9033737, -6.3490210, 26.5637913, -31.3204460, 27.2523956

Time for backsubstitution: 0.77 seconds

### NS candidates at layer 1
type: A, layer: 1, pos: 13
type: A, layer: 1, pos: 16
type: A, layer: 1, pos: 22
type: A, layer: 1, pos: 17
type: A, layer: 1, pos: 10
type: A, layer: 1, pos: 21
type: A, layer: 1, pos: 41
type: A, layer: 1, pos: 0
type: A, layer: 1, pos: 36

Time for candidate selection: 0.05 seconds

### Candidate
type: A, layer: 1, pos: 13

### Candidate
type: A, layer: 1, pos: 16

### Candidate
type: A, layer: 1, pos: 22

### Candidate
type: A, layer: 1, pos: 17

## Relational analysis of NS_A1_B1_A1_B2_A2_B2_A1

### Relational analysis result of NS_A1_B1_A1_B2_A2_B2_A1
Status: Status.VERIFIED
Output dim: 0, lower bound: -27.6112512, upper bound: 27.6228780
time: 0.62 seconds

## Relational analysis of NS_A1_B1_A1_B2_A2_B2_A2

### Relational analysis result of NS_A1_B1_A1_B2_A2_B2_A2
Status: Status.UNKNOWN
Output dim: 0, lower bound: -27.7755511, upper bound: 27.8466200
time: 0.63 seconds

## BFS NS instance: NS_A1_B1_A2_B1_A2_B1

### Backsubstitution after applying NS history:
0: -5.1221609, 18.6184425, -3.9672580, 14.7736979, -19.8958588, 22.5857010
1: -8.4564943, 18.7807064, -6.6120090, 14.9280148, -23.3845100, 25.3927155
2: -6.8467531, 20.3681755, -5.3131099, 16.2373619, -23.0841103, 25.6812859
3: -7.4440994, 28.1251602, -5.8528652, 22.4279957, -29.8720913, 33.9780273
4: -6.0290127, 26.1052856, -4.6880665, 20.7485371, -26.7775497, 30.7933521

Time for backsubstitution: 0.77 seconds

### NS candidates at layer 1
type: A, layer: 1, pos: 13
type: A, layer: 1, pos: 16
type: A, layer: 1, pos: 22
type: A, layer: 1, pos: 10
type: A, layer: 1, pos: 17
type: A, layer: 1, pos: 41
type: A, layer: 1, pos: 21
type: A, layer: 1, pos: 0
type: A, layer: 1, pos: 36

Time for candidate selection: 0.05 seconds

### Candidate
type: A, layer: 1, pos: 13

### Candidate
type: A, layer: 1, pos: 16

### Candidate
type: A, layer: 1, pos: 22

## Relational analysis of NS_A1_B1_A2_B1_A2_B1_A1

### Relational analysis result of NS_A1_B1_A2_B1_A2_B1_A1
Status: Status.VERIFIED
Output dim: 0, lower bound: -27.7549919, upper bound: 27.7419981
time: 0.69 seconds

## Relational analysis of NS_A1_B1_A2_B1_A2_B1_A2

### Relational analysis result of NS_A1_B1_A2_B1_A2_B1_A2
Status: Status.UNKNOWN
Output dim: 0, lower bound: -27.8079817, upper bound: 27.8050375
time: 0.62 seconds

## BFS NS instance: NS_A1_B1_A2_B1_A2_B2

### Backsubstitution after applying NS history:
0: -5.1221609, 18.6184425, -4.0236449, 14.8443575, -19.9665184, 22.6420822
1: -8.4564943, 18.7807064, -6.7107162, 15.0167494, -23.4732437, 25.4914207
2: -6.8467531, 20.3681755, -5.3908873, 16.3171444, -23.1638947, 25.7590637
3: -7.4440994, 28.1251602, -5.9503188, 22.5346546, -29.9787521, 34.0754776
4: -6.0290127, 26.1052856, -4.7566547, 20.9033737, -26.9323864, 30.8619404

Time for backsubstitution: 0.78 seconds

### NS candidates at layer 1
type: A, layer: 1, pos: 13
type: A, layer: 1, pos: 16
type: A, layer: 1, pos: 22
type: A, layer: 1, pos: 10
type: A, layer: 1, pos: 17
type: A, layer: 1, pos: 41
type: A, layer: 1, pos: 21
type: A, layer: 1, pos: 0
type: A, layer: 1, pos: 36

Time for candidate selection: 0.05 seconds

### Candidate
type: A, layer: 1, pos: 13

### Candidate
type: A, layer: 1, pos: 16

### Candidate
type: A, layer: 1, pos: 22

## Relational analysis of NS_A1_B1_A2_B1_A2_B2_A1

### Relational analysis result of NS_A1_B1_A2_B1_A2_B2_A1
Status: Status.VERIFIED
Output dim: 0, lower bound: -27.7549919, upper bound: 27.7423736
time: 0.76 seconds

## Relational analysis of NS_A1_B1_A2_B1_A2_B2_A2

### Relational analysis result of NS_A1_B1_A2_B1_A2_B2_A2
Status: Status.UNKNOWN
Output dim: 0, lower bound: -27.8079817, upper bound: 27.8144338
time: 0.63 seconds

## BFS NS instance: NS_A1_B1_A2_B2_A2_B1

### Backsubstitution after applying NS history:
0: -5.1221609, 18.6184425, -5.3430967, 18.6277962, -23.7499580, 23.9615402
1: -8.4564943, 18.7807064, -8.8277473, 18.9698372, -27.4263306, 27.6084538
2: -6.8467531, 20.3681755, -7.1858878, 20.5576668, -27.4044189, 27.5540619
3: -7.4440994, 28.1251602, -7.8491683, 28.2164726, -35.6605682, 35.9743271
4: -6.0290127, 26.1052856, -6.3779392, 26.6686516, -32.6976624, 32.4832230

Time for backsubstitution: 0.79 seconds

### NS candidates at layer 1
type: A, layer: 1, pos: 13
type: A, layer: 1, pos: 16
type: A, layer: 1, pos: 22
type: A, layer: 1, pos: 10
type: A, layer: 1, pos: 17
type: A, layer: 1, pos: 41
type: A, layer: 1, pos: 21
type: A, layer: 1, pos: 0
type: A, layer: 1, pos: 36

Time for candidate selection: 0.05 seconds

### Candidate
type: A, layer: 1, pos: 13

### Candidate
type: A, layer: 1, pos: 16

### Candidate
type: A, layer: 1, pos: 22

### Candidate
type: A, layer: 1, pos: 10

## Relational analysis of NS_A1_B1_A2_B2_A2_B1_A1

### Relational analysis result of NS_A1_B1_A2_B2_A2_B1_A1
Status: Status.VERIFIED
Output dim: 0, lower bound: -27.7609829, upper bound: 27.7530056
time: 0.67 seconds

## Relational analysis of NS_A1_B1_A2_B2_A2_B1_A2

### Relational analysis result of NS_A1_B1_A2_B2_A2_B1_A2
Status: Status.UNKNOWN
Output dim: 0, lower bound: -27.7609829, upper bound: 27.8066735
time: 0.78 seconds

## BFS NS instance: NS_A1_B1_A2_B2_A2_B2

### Backsubstitution after applying NS history:
0: -4.9966965, 18.2156467, -6.1962891, 21.5450344, -26.5417290, 24.4119339
1: -8.2608452, 18.3756657, -10.1972399, 21.9049397, -30.1657848, 28.5729027
2: -6.6783481, 19.9384537, -8.3275728, 23.6974449, -30.3757935, 28.2660255
3: -7.2715921, 27.5524273, -9.0545511, 32.5837708, -39.8553581, 36.6069794
4: -5.8851523, 25.5509300, -7.3623266, 30.8463898, -36.7315407, 32.9132576

Time for backsubstitution: 0.78 seconds

### NS candidates at layer 1
type: A, layer: 1, pos: 13
type: A, layer: 1, pos: 16
type: A, layer: 1, pos: 22
type: A, layer: 1, pos: 10
type: A, layer: 1, pos: 17
type: A, layer: 1, pos: 41
type: A, layer: 1, pos: 21
type: A, layer: 1, pos: 0
type: A, layer: 1, pos: 36

Time for candidate selection: 0.05 seconds

### Candidate
type: A, layer: 1, pos: 13

### Candidate
type: A, layer: 1, pos: 16

### Candidate
type: A, layer: 1, pos: 22

### Candidate
type: A, layer: 1, pos: 10

## Relational analysis of NS_A1_B1_A2_B2_A2_B2_A1

### Relational analysis result of NS_A1_B1_A2_B2_A2_B2_A1
Status: Status.VERIFIED
Output dim: 0, lower bound: -27.7609829, upper bound: 27.7530056
time: 0.66 seconds

## Relational analysis of NS_A1_B1_A2_B2_A2_B2_A2

### Relational analysis result of NS_A1_B1_A2_B2_A2_B2_A2
Status: Status.UNKNOWN
Output dim: 0, lower bound: -27.7609829, upper bound: 27.8066735
time: 0.60 seconds

## BFS NS instance: NS_A1_B2_A1_B1_A1_B2

### Backsubstitution after applying NS history:
0: -3.9672580, 14.7736979, -5.2570248, 19.0853691, -23.0526276, 20.0307198
1: -6.6120090, 14.9280148, -8.6756134, 19.2474518, -25.8594608, 23.6036282
2: -5.3131099, 16.2373619, -7.0249491, 20.8739109, -26.1870213, 23.2623081
3: -5.8528652, 22.4279957, -7.6366882, 28.8440228, -34.6968842, 30.0646763
4: -4.6880665, 20.7485371, -6.1873622, 26.7960167, -31.4840832, 26.9358997

Time for backsubstitution: 0.78 seconds

### NS candidates at layer 1
type: A, layer: 1, pos: 13
type: A, layer: 1, pos: 16
type: A, layer: 1, pos: 22
type: A, layer: 1, pos: 10
type: A, layer: 1, pos: 17
type: A, layer: 1, pos: 21
type: A, layer: 1, pos: 41
type: A, layer: 1, pos: 36
type: A, layer: 1, pos: 0

Time for candidate selection: 0.05 seconds

### Candidate
type: A, layer: 1, pos: 13

### Candidate
type: A, layer: 1, pos: 16

### Candidate
type: A, layer: 1, pos: 22

## Relational analysis of NS_A1_B2_A1_B1_A1_B2_A1
Optimization infeasible because this subproblem isn't reachable.

### Candidate
type: A, layer: 1, pos: 10

## Relational analysis of NS_A1_B2_A1_B1_A1_B2_A1
Optimization infeasible because this subproblem isn't reachable.

### Candidate
type: A, layer: 1, pos: 17

## Relational analysis of NS_A1_B2_A1_B1_A1_B2_A1

### Relational analysis result of NS_A1_B2_A1_B1_A1_B2_A1
Status: Status.VERIFIED
Output dim: 0, lower bound: -27.6945105, upper bound: 27.7344059
time: 0.61 seconds

## Relational analysis of NS_A1_B2_A1_B1_A1_B2_A2

### Relational analysis result of NS_A1_B2_A1_B1_A1_B2_A2
Status: Status.UNKNOWN
Output dim: 0, lower bound: -27.7376273, upper bound: 27.8079817
time: 0.54 seconds

## BFS NS instance: NS_A1_B2_A1_B1_A2_B2

### Backsubstitution after applying NS history:
0: -4.0236449, 14.8443575, -5.2570248, 19.0853691, -23.1090088, 20.1013775
1: -6.7107162, 15.0167494, -8.6756134, 19.2474518, -25.9581661, 23.6923637
2: -5.3908873, 16.3171444, -7.0249491, 20.8739109, -26.2647972, 23.3420887
3: -5.9503188, 22.5346546, -7.6366882, 28.8440228, -34.7943344, 30.1713390
4: -4.7566547, 20.9033737, -6.1873622, 26.7960167, -31.5526714, 27.0907364

Time for backsubstitution: 0.79 seconds

### NS candidates at layer 1
type: A, layer: 1, pos: 13
type: A, layer: 1, pos: 16
type: A, layer: 1, pos: 22
type: A, layer: 1, pos: 17
type: A, layer: 1, pos: 10
type: A, layer: 1, pos: 21
type: A, layer: 1, pos: 41
type: A, layer: 1, pos: 0
type: A, layer: 1, pos: 36

Time for candidate selection: 0.05 seconds

### Candidate
type: A, layer: 1, pos: 13

### Candidate
type: A, layer: 1, pos: 16

### Candidate
type: A, layer: 1, pos: 22

## Relational analysis of NS_A1_B2_A1_B1_A2_B2_A1
Optimization infeasible because this subproblem isn't reachable.

### Candidate
type: A, layer: 1, pos: 17

## Relational analysis of NS_A1_B2_A1_B1_A2_B2_A1

### Relational analysis result of NS_A1_B2_A1_B1_A2_B2_A1
Status: Status.VERIFIED
Output dim: 0, lower bound: -27.4392437, upper bound: 27.5083911
time: 0.48 seconds

## Relational analysis of NS_A1_B2_A1_B1_A2_B2_A2

### Relational analysis result of NS_A1_B2_A1_B1_A2_B2_A2
Status: Status.UNKNOWN
Output dim: 0, lower bound: -27.7369988, upper bound: 27.8160649
time: 0.80 seconds

## BFS NS instance: NS_A1_B2_A2_B1_A2_B2

### Backsubstitution after applying NS history:
0: -5.1221609, 18.6184425, -5.2570248, 19.0853691, -24.2075310, 23.8754654
1: -8.4564943, 18.7807064, -8.6756134, 19.2474518, -27.7039452, 27.4563198
2: -6.8467531, 20.3681755, -7.0249491, 20.8739109, -27.7206631, 27.3931198
3: -7.4440994, 28.1251602, -7.6366882, 28.8440228, -36.2881088, 35.7618484
4: -6.0290127, 26.1052856, -6.1873622, 26.7960167, -32.8250275, 32.2926483

Time for backsubstitution: 0.79 seconds

### NS candidates at layer 1
type: A, layer: 1, pos: 13
type: A, layer: 1, pos: 16
type: A, layer: 1, pos: 22
type: A, layer: 1, pos: 10
type: A, layer: 1, pos: 17
type: A, layer: 1, pos: 41
type: A, layer: 1, pos: 21
type: A, layer: 1, pos: 0
type: A, layer: 1, pos: 36

Time for candidate selection: 0.05 seconds

### Candidate
type: A, layer: 1, pos: 13

### Candidate
type: A, layer: 1, pos: 16

### Candidate
type: A, layer: 1, pos: 22

## Relational analysis of NS_A1_B2_A2_B1_A2_B2_A1
Optimization infeasible because this subproblem isn't reachable.

### Candidate
type: A, layer: 1, pos: 10

## Relational analysis of NS_A1_B2_A2_B1_A2_B2_A1
Optimization infeasible because this subproblem isn't reachable.

### Candidate
type: A, layer: 1, pos: 17

## Relational analysis of NS_A1_B2_A2_B1_A2_B2_A1

### Relational analysis result of NS_A1_B2_A2_B1_A2_B2_A1
Status: Status.VERIFIED
Output dim: 0, lower bound: -27.4099857, upper bound: 27.4858591
time: 0.78 seconds

## Relational analysis of NS_A1_B2_A2_B1_A2_B2_A2

### Relational analysis result of NS_A1_B2_A2_B1_A2_B2_A2
Status: Status.UNKNOWN
Output dim: 0, lower bound: -27.4099857, upper bound: 27.8045752
time: 0.50 seconds

## BFS NS instance: NS_A2_B1_A1_B1_A1_B1

### Backsubstitution after applying NS history:
0: -5.3486147, 18.6461754, -4.0425730, 14.9028101, -20.2514248, 22.6887455
1: -8.8365345, 18.9885597, -6.7421403, 15.0785389, -23.9150734, 25.7306995
2: -7.1932240, 20.5773335, -5.4165335, 16.3820667, -23.5752907, 25.9938660
3: -7.8570271, 28.2433796, -5.9786677, 22.6255608, -30.4825878, 34.2220459
4: -6.3842969, 26.6930981, -4.7795992, 20.9909840, -27.3752785, 31.4726925

Time for backsubstitution: 0.79 seconds

### NS candidates at layer 1
type: A, layer: 1, pos: 13
type: A, layer: 1, pos: 16
type: A, layer: 1, pos: 9
type: A, layer: 1, pos: 22
type: A, layer: 1, pos: 41
type: A, layer: 1, pos: 0
type: A, layer: 1, pos: 17
type: A, layer: 1, pos: 21
type: A, layer: 1, pos: 36

Time for candidate selection: 0.05 seconds

### Candidate
type: A, layer: 1, pos: 13

### Candidate
type: A, layer: 1, pos: 16

### Candidate
type: A, layer: 1, pos: 9

### Candidate
type: A, layer: 1, pos: 22

### Candidate
type: A, layer: 1, pos: 41

## Relational analysis of NS_A2_B1_A1_B1_A1_B1_A1
Optimization infeasible because this subproblem isn't reachable.

### Candidate
type: A, layer: 1, pos: 0

## Relational analysis of NS_A2_B1_A1_B1_A1_B1_A1
Optimization infeasible because this subproblem isn't reachable.

### Candidate
type: A, layer: 1, pos: 17

### Candidate
type: A, layer: 1, pos: 21

## Relational analysis of NS_A2_B1_A1_B1_A1_B1_A1

### Relational analysis result of NS_A2_B1_A1_B1_A1_B1_A1
Status: Status.UNKNOWN
Output dim: 0, lower bound: -27.8196818, upper bound: 27.8245054
time: 0.80 seconds

## Relational analysis of NS_A2_B1_A1_B1_A1_B1_A2
Optimization infeasible because this subproblem isn't reachable.

### Candidate
type: A, layer: 1, pos: 36

## Relational analysis of NS_A2_B1_A1_B1_A1_B1_A1

### Relational analysis result of NS_A2_B1_A1_B1_A1_B1_A1
Status: Status.UNKNOWN
Output dim: 0, lower bound: -27.8242323, upper bound: 27.8269793
time: 0.61 seconds

## Relational analysis of NS_A2_B1_A1_B1_A1_B1_A2

### Relational analysis result of NS_A2_B1_A1_B1_A1_B1_A2
Status: Status.UNKNOWN
Output dim: 0, lower bound: -27.8207193, upper bound: 27.8221306
time: 0.72 seconds

## BFS NS instance: NS_A2_B1_A1_B1_A1_B2

### Backsubstitution after applying NS history:
0: -5.3486147, 18.6461754, -4.9090581, 17.7465897, -23.0952034, 23.5552330
1: -8.8365345, 18.9885597, -8.1334858, 17.9675713, -26.8041058, 27.1220436
2: -7.1932240, 20.5773335, -6.5688643, 19.4793873, -26.6726112, 27.1461945
3: -7.8570271, 28.2433796, -7.2016802, 26.9309406, -34.7879677, 35.4450569
4: -6.3842969, 26.6930981, -5.7876644, 25.0691452, -31.4534416, 32.4807625

Time for backsubstitution: 0.79 seconds

### NS candidates at layer 1
type: A, layer: 1, pos: 13
type: A, layer: 1, pos: 16
type: A, layer: 1, pos: 9
type: A, layer: 1, pos: 22
type: A, layer: 1, pos: 41
type: A, layer: 1, pos: 0
type: A, layer: 1, pos: 17
type: A, layer: 1, pos: 21
type: A, layer: 1, pos: 36

Time for candidate selection: 0.05 seconds

### Candidate
type: A, layer: 1, pos: 13

### Candidate
type: A, layer: 1, pos: 16

### Candidate
type: A, layer: 1, pos: 9

### Candidate
type: A, layer: 1, pos: 22

### Candidate
type: A, layer: 1, pos: 41

## Relational analysis of NS_A2_B1_A1_B1_A1_B2_A1
Optimization infeasible because this subproblem isn't reachable.

### Candidate
type: A, layer: 1, pos: 0

## Relational analysis of NS_A2_B1_A1_B1_A1_B2_A1
Optimization infeasible because this subproblem isn't reachable.

### Candidate
type: A, layer: 1, pos: 17

### Candidate
type: A, layer: 1, pos: 21

## Relational analysis of NS_A2_B1_A1_B1_A1_B2_A1

### Relational analysis result of NS_A2_B1_A1_B1_A1_B2_A1
Status: Status.UNKNOWN
Output dim: 0, lower bound: -27.8196818, upper bound: 27.8245054
time: 0.58 seconds

## Relational analysis of NS_A2_B1_A1_B1_A1_B2_A2
Optimization infeasible because this subproblem isn't reachable.

### Candidate
type: A, layer: 1, pos: 36

## Relational analysis of NS_A2_B1_A1_B1_A1_B2_A1

### Relational analysis result of NS_A2_B1_A1_B1_A1_B2_A1
Status: Status.UNKNOWN
Output dim: 0, lower bound: -27.8242323, upper bound: 27.8269793
time: 0.93 seconds

## Relational analysis of NS_A2_B1_A1_B1_A1_B2_A2

### Relational analysis result of NS_A2_B1_A1_B1_A1_B2_A2
Status: Status.UNKNOWN
Output dim: 0, lower bound: -27.8207193, upper bound: 27.8237610
time: 0.78 seconds

## BFS NS instance: NS_A2_B1_A1_B1_A2_B1

### Backsubstitution after applying NS history:
0: -6.2936206, 21.7869453, -4.0425730, 14.9028101, -21.1964302, 25.8295174
1: -10.3439865, 22.1925621, -6.7421403, 15.0785389, -25.4225254, 28.9347019
2: -8.4501743, 23.9575081, -5.4165335, 16.3820667, -24.8322392, 29.3740406
3: -9.1837044, 32.9127502, -5.9786677, 22.6255608, -31.8092651, 38.8914146
4: -7.4634824, 31.1625271, -4.7795992, 20.9909840, -28.4544659, 35.9421234

Time for backsubstitution: 0.79 seconds

### NS candidates at layer 1
type: A, layer: 1, pos: 13
type: A, layer: 1, pos: 16
type: A, layer: 1, pos: 9
type: A, layer: 1, pos: 22
type: A, layer: 1, pos: 0
type: A, layer: 1, pos: 41
type: A, layer: 1, pos: 21
type: A, layer: 1, pos: 17
type: A, layer: 1, pos: 36

Time for candidate selection: 0.05 seconds

### Candidate
type: A, layer: 1, pos: 13

### Candidate
type: A, layer: 1, pos: 16

### Candidate
type: A, layer: 1, pos: 9

### Candidate
type: A, layer: 1, pos: 22

### Candidate
type: A, layer: 1, pos: 0

## Relational analysis of NS_A2_B1_A1_B1_A2_B1_A1
Optimization infeasible because this subproblem isn't reachable.

### Candidate
type: A, layer: 1, pos: 41

## Relational analysis of NS_A2_B1_A1_B1_A2_B1_A1
Optimization infeasible because this subproblem isn't reachable.

### Candidate
type: A, layer: 1, pos: 21

## Relational analysis of NS_A2_B1_A1_B1_A2_B1_A1
Optimization infeasible because this subproblem isn't reachable.

### Candidate
type: A, layer: 1, pos: 17

### Candidate
type: A, layer: 1, pos: 36

## Relational analysis of NS_A2_B1_A1_B1_A2_B1_A1
Optimization infeasible because this subproblem isn't reachable.

## BFS NS instance: NS_A2_B1_A1_B1_A2_B2

### Backsubstitution after applying NS history:
0: -6.2936206, 21.7869453, -4.8150897, 17.4743633, -23.7679844, 26.6020355
1: -10.3439865, 22.1925621, -7.9864111, 17.6775780, -28.0215626, 30.1789742
2: -8.4501743, 23.9575081, -6.4391880, 19.1782894, -27.6284637, 30.3966961
3: -9.1837044, 32.9127502, -7.0671067, 26.5339699, -35.7176743, 39.9798508
4: -7.4634824, 31.1625271, -5.6701784, 24.6798344, -32.1433144, 36.8326988

Time for backsubstitution: 0.80 seconds

### NS candidates at layer 1
type: A, layer: 1, pos: 13
type: A, layer: 1, pos: 16
type: A, layer: 1, pos: 9
type: A, layer: 1, pos: 22
type: A, layer: 1, pos: 0
type: A, layer: 1, pos: 41
type: A, layer: 1, pos: 21
type: A, layer: 1, pos: 17
type: A, layer: 1, pos: 36

Time for candidate selection: 0.05 seconds

### Candidate
type: A, layer: 1, pos: 13

### Candidate
type: A, layer: 1, pos: 16

### Candidate
type: A, layer: 1, pos: 9

### Candidate
type: A, layer: 1, pos: 22

### Candidate
type: A, layer: 1, pos: 0

## Relational analysis of NS_A2_B1_A1_B1_A2_B2_A1
Optimization infeasible because this subproblem isn't reachable.

### Candidate
type: A, layer: 1, pos: 41

## Relational analysis of NS_A2_B1_A1_B1_A2_B2_A1
Optimization infeasible because this subproblem isn't reachable.

### Candidate
type: A, layer: 1, pos: 21

## Relational analysis of NS_A2_B1_A1_B1_A2_B2_A1
Optimization infeasible because this subproblem isn't reachable.

### Candidate
type: A, layer: 1, pos: 17

### Candidate
type: A, layer: 1, pos: 36

## Relational analysis of NS_A2_B1_A1_B1_A2_B2_A1
Optimization infeasible because this subproblem isn't reachable.

## BFS NS instance: NS_A2_B1_A1_B2_A2_B2

### Backsubstitution after applying NS history:
0: -6.2936206, 21.7869453, -5.9018812, 21.2415199, -27.5351391, 27.6888275
1: -10.3439865, 22.1925621, -9.7087307, 21.4427299, -31.7867107, 31.9012928
2: -8.4501743, 23.9575081, -7.8944397, 23.2121582, -31.6623325, 31.8519478
3: -9.1837044, 32.9127502, -8.5615578, 32.1011086, -41.2848091, 41.4743042
4: -7.4634824, 31.1625271, -6.9596791, 29.8811989, -37.3446770, 38.1222038

Time for backsubstitution: 0.79 seconds

### NS candidates at layer 1
type: A, layer: 1, pos: 13
type: A, layer: 1, pos: 16
type: A, layer: 1, pos: 9
type: A, layer: 1, pos: 22
type: A, layer: 1, pos: 0
type: A, layer: 1, pos: 41
type: A, layer: 1, pos: 21
type: A, layer: 1, pos: 17
type: A, layer: 1, pos: 36

Time for candidate selection: 0.05 seconds

### Candidate
type: A, layer: 1, pos: 13

### Candidate
type: A, layer: 1, pos: 16

### Candidate
type: A, layer: 1, pos: 9

### Candidate
type: A, layer: 1, pos: 22

### Candidate
type: A, layer: 1, pos: 0

## Relational analysis of NS_A2_B1_A1_B2_A2_B2_A1
Optimization infeasible because this subproblem isn't reachable.

### Candidate
type: A, layer: 1, pos: 41

## Relational analysis of NS_A2_B1_A1_B2_A2_B2_A1
Optimization infeasible because this subproblem isn't reachable.

### Candidate
type: A, layer: 1, pos: 21

## Relational analysis of NS_A2_B1_A1_B2_A2_B2_A1
Optimization infeasible because this subproblem isn't reachable.

### Candidate
type: A, layer: 1, pos: 17

## Relational analysis of NS_A2_B1_A1_B2_A2_B2_A1
Optimization infeasible because this subproblem isn't reachable.

### Candidate
type: A, layer: 1, pos: 36

## Relational analysis of NS_A2_B1_A1_B2_A2_B2_A1
Optimization infeasible because this subproblem isn't reachable.

## NS Result
status: Status.UNKNOWN
execution time: (base) + (ns) = 2.40 + 138.25 = 140.65 seconds
