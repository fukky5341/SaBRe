## Execution arguments:
Dataset: Dataset.ACAS
Network: ds/onnx/acasxu_op11/ACASXU_1_8.onnx
Relational property: GLOBAL_ROBUSTNESS
LP Analysis: True
Epsilon: None
Delta epsilon: 0.1
execution index: (10, None, 8)
Time budget: 420 seconds
Split limit: 100
Threshold: 20.60317678965


## IAR start

### BASE IAR bounds
Layer (inp1_lb, inp1_ub, inp2_lb, inp2_ub, d_lb, d_ub)
0: (-3.7843938, 3.3221977, -3.7843938, 3.3221977, -7.1065907, 7.1065907)
1: (-14.9787064, 12.8845959, -14.9787064, 12.8845959, -27.8633022, 27.8633022)
2: (-7.4894867, 12.0534105, -7.4894867, 12.0534105, -19.5428963, 19.5428963)
3: (-13.1016846, 11.7157326, -13.1016846, 11.7157326, -24.8174152, 24.8174152)
4: (-9.5921707, 12.2164268, -9.5921707, 12.2164268, -21.8085976, 21.8085976)

## BASE Relational Analysis

## BASE Result
execution time: IAR + RelationalAnalysis = 1.64 + 1.89 = 3.53 seconds
status: Status.UNKNOWN
relational distance
Output dim: 3, lower bound: -20.6042070, upper bound: 20.6042070

# Neuron Split (NS) starts

## BFS NS instance: NS

Time for backsubstitution: 0.01 seconds

### NS candidates at layer 1
type: A, layer: 1, pos: 24
type: B, layer: 1, pos: 24
type: A, layer: 1, pos: 9
type: B, layer: 1, pos: 9
type: B, layer: 1, pos: 11
type: A, layer: 1, pos: 11
type: A, layer: 1, pos: 22
type: B, layer: 1, pos: 22
type: A, layer: 1, pos: 42
type: B, layer: 1, pos: 42
type: A, layer: 1, pos: 34
type: B, layer: 1, pos: 34
type: A, layer: 1, pos: 30
type: B, layer: 1, pos: 30
type: A, layer: 1, pos: 4
type: B, layer: 1, pos: 4
type: A, layer: 1, pos: 12
type: B, layer: 1, pos: 12
type: A, layer: 1, pos: 47
type: B, layer: 1, pos: 47
type: A, layer: 1, pos: 16
type: B, layer: 1, pos: 16
type: A, layer: 1, pos: 39
type: B, layer: 1, pos: 39
type: A, layer: 1, pos: 31
type: B, layer: 1, pos: 31
type: A, layer: 1, pos: 23
type: B, layer: 1, pos: 23
type: A, layer: 1, pos: 33
type: B, layer: 1, pos: 33

Time for candidate selection: 0.12 seconds

### Candidate
type: A, layer: 1, pos: 24

## Relational analysis of NS_A1

### Relational analysis result of NS_A1
Status: Status.UNKNOWN
Output dim: 3, lower bound: -20.6040137, upper bound: 20.6028333
time: 0.74 seconds

## Relational analysis of NS_A2

### Relational analysis result of NS_A2
Status: Status.VERIFIED
Output dim: 3, lower bound: -20.6026400, upper bound: 20.6026400
time: 0.76 seconds

## Summary of splitting at layer (split count: 0)
- Time for NS candidates: 1.65 seconds
NS_A1, status: Status.UNKNOWN, split count: 1, time: 1.65
Output dim: 3, lower bound: -20.6040137, upper bound: 20.6028333
NS_A2, status: Status.VERIFIED, split count: 1, time: 1.65
Output dim: 3, lower bound: -20.6026400, upper bound: 20.6026400

## BFS NS instance: NS_A1

### Backsubstitution after applying NS history:
0: -3.6480763, 3.2057531, -3.7843938, 3.3221977, -6.9702740, 6.9901462
1: -14.4352322, 12.4338837, -14.9787064, 12.8845959, -27.3198261, 27.4125881
2: -7.2177253, 11.6256628, -7.4894867, 12.0534105, -19.2711334, 19.1151485
3: -12.6268978, 11.3142090, -13.1016846, 11.7157326, -24.3426304, 24.4158936
4: -9.2462864, 11.7903395, -9.5921707, 12.2164268, -21.4627132, 21.3825111

Time for backsubstitution: 0.95 seconds

### NS candidates at layer 1
type: A, layer: 1, pos: 9
type: B, layer: 1, pos: 9
type: B, layer: 1, pos: 11
type: A, layer: 1, pos: 11
type: B, layer: 1, pos: 22
type: A, layer: 1, pos: 22
type: B, layer: 1, pos: 30
type: B, layer: 1, pos: 34
type: B, layer: 1, pos: 42
type: A, layer: 1, pos: 42
type: A, layer: 1, pos: 34
type: A, layer: 1, pos: 4
type: B, layer: 1, pos: 4
type: A, layer: 1, pos: 12
type: B, layer: 1, pos: 12
type: A, layer: 1, pos: 30
type: B, layer: 1, pos: 24
type: B, layer: 1, pos: 47
type: B, layer: 1, pos: 16
type: A, layer: 1, pos: 16
type: A, layer: 1, pos: 47
type: A, layer: 1, pos: 39
type: B, layer: 1, pos: 39
type: A, layer: 1, pos: 31
type: B, layer: 1, pos: 23
type: B, layer: 1, pos: 31
type: A, layer: 1, pos: 23
type: A, layer: 1, pos: 33
type: B, layer: 1, pos: 33

Time for candidate selection: 0.07 seconds

### Candidate
type: A, layer: 1, pos: 9

## Relational analysis of NS_A1_A1

### Relational analysis result of NS_A1_A1
Status: Status.VERIFIED
Output dim: 3, lower bound: -20.6004034, upper bound: 20.6026494
time: 0.66 seconds

## Relational analysis of NS_A1_A2

### Relational analysis result of NS_A1_A2
Status: Status.UNKNOWN
Output dim: 3, lower bound: -20.6040137, upper bound: 20.6028333
time: 0.91 seconds

## Summary of splitting at layer (split count: 1)
- Time for NS candidates: 2.61 seconds
NS_A1_A1, status: Status.VERIFIED, split count: 2, time: 2.61
Output dim: 3, lower bound: -20.6004034, upper bound: 20.6026494
NS_A1_A2, status: Status.UNKNOWN, split count: 2, time: 2.61
Output dim: 3, lower bound: -20.6040137, upper bound: 20.6028333

## BFS NS instance: NS_A1_A2

### Backsubstitution after applying NS history:
0: -3.5923164, 3.1570973, -3.7843938, 3.3221977, -6.9145141, 6.9414897
1: -14.2131710, 12.2455807, -14.9787064, 12.8845959, -27.0977669, 27.2242832
2: -7.1067591, 11.4467535, -7.4894867, 12.0534105, -19.1601696, 18.9362411
3: -12.4342022, 11.1442776, -13.1016846, 11.7157326, -24.1499348, 24.2459583
4: -9.1061058, 11.6097670, -9.5921707, 12.2164268, -21.3225327, 21.2019386

Time for backsubstitution: 0.85 seconds

### NS candidates at layer 1
type: B, layer: 1, pos: 11
type: A, layer: 1, pos: 11
type: B, layer: 1, pos: 22
type: A, layer: 1, pos: 22
type: B, layer: 1, pos: 34
type: B, layer: 1, pos: 30
type: B, layer: 1, pos: 42
type: A, layer: 1, pos: 42
type: A, layer: 1, pos: 34
type: A, layer: 1, pos: 4
type: B, layer: 1, pos: 4
type: A, layer: 1, pos: 12
type: B, layer: 1, pos: 12
type: A, layer: 1, pos: 30
type: B, layer: 1, pos: 24
type: B, layer: 1, pos: 9
type: A, layer: 1, pos: 47
type: B, layer: 1, pos: 16
type: B, layer: 1, pos: 47
type: A, layer: 1, pos: 16
type: A, layer: 1, pos: 39
type: B, layer: 1, pos: 39
type: A, layer: 1, pos: 31
type: B, layer: 1, pos: 23
type: B, layer: 1, pos: 31
type: A, layer: 1, pos: 23
type: A, layer: 1, pos: 33
type: B, layer: 1, pos: 33

Time for candidate selection: 0.07 seconds

### Candidate
type: B, layer: 1, pos: 11

## Relational analysis of NS_A1_A2_B1

### Relational analysis result of NS_A1_A2_B1
Status: Status.VERIFIED
Output dim: 3, lower bound: -20.5927310, upper bound: 20.6021486
time: 0.72 seconds

## Relational analysis of NS_A1_A2_B2
Optimization infeasible because this subproblem isn't reachable.

### Candidate
type: A, layer: 1, pos: 11

### Candidate
type: B, layer: 1, pos: 22

## Relational analysis of NS_A1_A2_B1

### Relational analysis result of NS_A1_A2_B1
Status: Status.UNKNOWN
Output dim: 3, lower bound: -20.6036034, upper bound: 20.5995919
time: 0.69 seconds

## Relational analysis of NS_A1_A2_B2

### Relational analysis result of NS_A1_A2_B2
Status: Status.UNKNOWN
Output dim: 3, lower bound: -20.6040040, upper bound: 20.6028273
time: 0.59 seconds

## Summary of splitting at layer (split count: 2)
- Time for NS candidates: 3.15 seconds
NS_A1_A2_B1, status: Status.UNKNOWN, split count: 3, time: 3.15
Output dim: 3, lower bound: -20.6036034, upper bound: 20.5995919
NS_A1_A2_B2, status: Status.UNKNOWN, split count: 3, time: 3.15
Output dim: 3, lower bound: -20.6040040, upper bound: 20.6028273

## BFS NS instance: NS_A1_A2_B1

### Backsubstitution after applying NS history:
0: -3.5923164, 3.1570973, -3.3461347, 2.9654913, -6.5578074, 6.5032315
1: -14.2131710, 12.2455807, -13.2283525, 11.5012493, -25.7144203, 25.4739304
2: -7.1067591, 11.4467535, -6.6200395, 10.7311440, -17.8379021, 18.0667934
3: -12.4342022, 11.1442776, -11.5760069, 10.4962263, -22.9304276, 22.7202816
4: -9.1061058, 11.6097670, -8.4712791, 10.9263763, -20.0324821, 20.0810471

Time for backsubstitution: 0.87 seconds

### NS candidates at layer 1
type: A, layer: 1, pos: 11
type: B, layer: 1, pos: 11
type: B, layer: 1, pos: 30
type: B, layer: 1, pos: 34
type: A, layer: 1, pos: 42
type: B, layer: 1, pos: 42
type: B, layer: 1, pos: 4
type: B, layer: 1, pos: 12
type: A, layer: 1, pos: 4
type: A, layer: 1, pos: 34
type: B, layer: 1, pos: 47
type: B, layer: 1, pos: 24
type: A, layer: 1, pos: 12
type: A, layer: 1, pos: 22
type: B, layer: 1, pos: 9
type: A, layer: 1, pos: 30
type: A, layer: 1, pos: 16
type: B, layer: 1, pos: 16
type: A, layer: 1, pos: 39
type: B, layer: 1, pos: 39
type: A, layer: 1, pos: 23
type: B, layer: 1, pos: 33
type: B, layer: 1, pos: 31
type: A, layer: 1, pos: 31
type: B, layer: 1, pos: 23
type: A, layer: 1, pos: 33
type: A, layer: 1, pos: 47

Time for candidate selection: 0.07 seconds

### Candidate
type: A, layer: 1, pos: 11

### Candidate
type: B, layer: 1, pos: 11

### Candidate
type: B, layer: 1, pos: 30

## Relational analysis of NS_A1_A2_B1_B1

### Relational analysis result of NS_A1_A2_B1_B1
Status: Status.UNKNOWN
Output dim: 3, lower bound: -20.6036034, upper bound: 20.5995919
time: 0.70 seconds

## Relational analysis of NS_A1_A2_B1_B2

### Relational analysis result of NS_A1_A2_B1_B2
Status: Status.UNKNOWN
Output dim: 3, lower bound: -20.6034243, upper bound: 20.5985397
time: 0.76 seconds

## BFS NS instance: NS_A1_A2_B2

### Backsubstitution after applying NS history:
0: -3.5923164, 3.1570973, -4.0484486, 3.5925496, -7.1848660, 7.2055445
1: -14.2131710, 12.2455807, -16.0087757, 13.9060564, -28.1192284, 28.2543526
2: -7.1067591, 11.4467535, -7.9802833, 13.0727425, -20.1795006, 19.4270363
3: -12.4342022, 11.1442776, -13.9787140, 12.6847849, -25.1189880, 25.1229897
4: -9.1061058, 11.6097670, -10.2256079, 13.2608871, -22.3669930, 21.8353748

Time for backsubstitution: 0.96 seconds

### NS candidates at layer 1
type: A, layer: 1, pos: 11
type: B, layer: 1, pos: 30
type: B, layer: 1, pos: 34
type: A, layer: 1, pos: 42
type: B, layer: 1, pos: 42
type: B, layer: 1, pos: 24
type: B, layer: 1, pos: 4
type: B, layer: 1, pos: 12
type: A, layer: 1, pos: 4
type: A, layer: 1, pos: 22
type: A, layer: 1, pos: 34
type: B, layer: 1, pos: 11
type: A, layer: 1, pos: 12
type: B, layer: 1, pos: 9
type: A, layer: 1, pos: 30
type: A, layer: 1, pos: 47
type: A, layer: 1, pos: 16
type: B, layer: 1, pos: 16
type: B, layer: 1, pos: 39
type: A, layer: 1, pos: 39
type: A, layer: 1, pos: 31
type: B, layer: 1, pos: 31
type: A, layer: 1, pos: 23
type: B, layer: 1, pos: 23
type: A, layer: 1, pos: 33
type: B, layer: 1, pos: 33
type: B, layer: 1, pos: 47

Time for candidate selection: 0.10 seconds

### Candidate
type: A, layer: 1, pos: 11

### Candidate
type: B, layer: 1, pos: 30

## Relational analysis of NS_A1_A2_B2_B1

### Relational analysis result of NS_A1_A2_B2_B1
Status: Status.UNKNOWN
Output dim: 3, lower bound: -20.6040040, upper bound: 20.6028273
time: 0.66 seconds

## Relational analysis of NS_A1_A2_B2_B2

### Relational analysis result of NS_A1_A2_B2_B2
Status: Status.UNKNOWN
Output dim: 3, lower bound: -20.6039518, upper bound: 20.6022874
time: 0.83 seconds

## Summary of splitting at layer (split count: 3)
- Time for NS candidates: 2.57 seconds
NS_A1_A2_B1_B1, status: Status.UNKNOWN, split count: 4, time: 2.57
Output dim: 3, lower bound: -20.6036034, upper bound: 20.5995919
NS_A1_A2_B1_B2, status: Status.UNKNOWN, split count: 4, time: 2.57
Output dim: 3, lower bound: -20.6034243, upper bound: 20.5985397
NS_A1_A2_B2_B1, status: Status.UNKNOWN, split count: 4, time: 2.57
Output dim: 3, lower bound: -20.6040040, upper bound: 20.6028273
NS_A1_A2_B2_B2, status: Status.UNKNOWN, split count: 4, time: 2.57
Output dim: 3, lower bound: -20.6039518, upper bound: 20.6022874

## BFS NS instance: NS_A1_A2_B1_B1

### Backsubstitution after applying NS history:
0: -3.5923164, 3.1570973, -3.1073444, 2.7737439, -6.3660603, 6.2644415
1: -14.2131710, 12.2455807, -12.2408829, 10.7699928, -24.9831638, 24.4864597
2: -7.1067591, 11.4467535, -6.1448307, 10.0647478, -17.1715069, 17.5915813
3: -12.4342022, 11.1442776, -10.7172527, 9.8717442, -22.3059464, 21.8615265
4: -9.1061058, 11.6097670, -7.8571095, 10.2771721, -19.3832779, 19.4668751

Time for backsubstitution: 1.07 seconds

### NS candidates at layer 1
type: B, layer: 1, pos: 11
type: B, layer: 1, pos: 47
type: A, layer: 1, pos: 11
type: B, layer: 1, pos: 34
type: B, layer: 1, pos: 4
type: A, layer: 1, pos: 4
type: B, layer: 1, pos: 12
type: A, layer: 1, pos: 42
type: B, layer: 1, pos: 42
type: A, layer: 1, pos: 22
type: B, layer: 1, pos: 9
type: A, layer: 1, pos: 12
type: A, layer: 1, pos: 34
type: B, layer: 1, pos: 24
type: B, layer: 1, pos: 33
type: A, layer: 1, pos: 23
type: A, layer: 1, pos: 39
type: A, layer: 1, pos: 31
type: B, layer: 1, pos: 31
type: B, layer: 1, pos: 16
type: A, layer: 1, pos: 16
type: B, layer: 1, pos: 23
type: B, layer: 1, pos: 39
type: A, layer: 1, pos: 33
type: A, layer: 1, pos: 47
type: A, layer: 1, pos: 30

Time for candidate selection: 0.09 seconds

### Candidate
type: B, layer: 1, pos: 11

### Candidate
type: B, layer: 1, pos: 47

## Relational analysis of NS_A1_A2_B1_B1_B1

### Relational analysis result of NS_A1_A2_B1_B1_B1
Status: Status.UNKNOWN
Output dim: 3, lower bound: -20.6032768, upper bound: 20.5970383
time: 0.82 seconds

## Relational analysis of NS_A1_A2_B1_B1_B2

### Relational analysis result of NS_A1_A2_B1_B1_B2
Status: Status.UNKNOWN
Output dim: 3, lower bound: -20.6036034, upper bound: 20.5995744
time: 0.79 seconds

## BFS NS instance: NS_A1_A2_B1_B2

### Backsubstitution after applying NS history:
0: -3.5923164, 3.1570973, -3.2859685, 2.9183435, -6.5106602, 6.4430656
1: -14.2131710, 12.2455807, -12.9850140, 11.3200541, -25.5332260, 25.2305908
2: -7.1067591, 11.4467535, -6.5043592, 10.5621500, -17.6689091, 17.9511127
3: -12.4342022, 11.1442776, -11.3635283, 10.3372755, -22.7714767, 22.5078049
4: -9.1061058, 11.6097670, -8.3101521, 10.7604237, -19.8665295, 19.9199162

Time for backsubstitution: 0.99 seconds

### NS candidates at layer 1
type: B, layer: 1, pos: 11
type: A, layer: 1, pos: 11
type: B, layer: 1, pos: 34
type: A, layer: 1, pos: 42
type: B, layer: 1, pos: 42
type: B, layer: 1, pos: 4
type: A, layer: 1, pos: 4
type: B, layer: 1, pos: 12
type: A, layer: 1, pos: 22
type: A, layer: 1, pos: 34
type: B, layer: 1, pos: 24
type: B, layer: 1, pos: 47
type: A, layer: 1, pos: 12
type: B, layer: 1, pos: 9
type: A, layer: 1, pos: 39
type: B, layer: 1, pos: 39
type: A, layer: 1, pos: 16
type: B, layer: 1, pos: 33
type: B, layer: 1, pos: 16
type: B, layer: 1, pos: 31
type: A, layer: 1, pos: 31
type: A, layer: 1, pos: 23
type: A, layer: 1, pos: 33
type: B, layer: 1, pos: 23
type: A, layer: 1, pos: 30
type: A, layer: 1, pos: 47

Time for candidate selection: 0.07 seconds

### Candidate
type: B, layer: 1, pos: 11

### Candidate
type: A, layer: 1, pos: 11

### Candidate
type: B, layer: 1, pos: 34

## Relational analysis of NS_A1_A2_B1_B2_B1
Optimization infeasible because this subproblem isn't reachable.

### Candidate
type: A, layer: 1, pos: 42

## Relational analysis of NS_A1_A2_B1_B2_A1

### Relational analysis result of NS_A1_A2_B1_B2_A1
Status: Status.VERIFIED
Output dim: 3, lower bound: -20.6009425, upper bound: 20.5983351
time: 0.63 seconds

## Relational analysis of NS_A1_A2_B1_B2_A2

### Relational analysis result of NS_A1_A2_B1_B2_A2
Status: Status.UNKNOWN
Output dim: 3, lower bound: -20.6033932, upper bound: 20.5985397
time: 0.73 seconds

## BFS NS instance: NS_A1_A2_B2_B1

### Backsubstitution after applying NS history:
0: -3.5923164, 3.1570973, -3.8009360, 3.3910289, -6.9833450, 6.9580331
1: -14.2131710, 12.2455807, -14.9908495, 13.1372862, -27.3504562, 27.2364273
2: -7.1067591, 11.4467535, -7.4865479, 12.3588858, -19.4656448, 18.9333019
3: -12.4342022, 11.1442776, -13.0861444, 12.0394907, -24.4736900, 24.2304192
4: -9.1061058, 11.6097670, -9.5920515, 12.5576630, -21.6637688, 21.2018185

Time for backsubstitution: 0.87 seconds

### NS candidates at layer 1
type: A, layer: 1, pos: 11
type: B, layer: 1, pos: 34
type: B, layer: 1, pos: 47
type: B, layer: 1, pos: 11
type: B, layer: 1, pos: 4
type: B, layer: 1, pos: 12
type: A, layer: 1, pos: 42
type: B, layer: 1, pos: 42
type: B, layer: 1, pos: 9
type: A, layer: 1, pos: 4
type: A, layer: 1, pos: 22
type: A, layer: 1, pos: 12
type: B, layer: 1, pos: 24
type: A, layer: 1, pos: 34
type: B, layer: 1, pos: 16
type: A, layer: 1, pos: 31
type: A, layer: 1, pos: 23
type: A, layer: 1, pos: 39
type: B, layer: 1, pos: 31
type: A, layer: 1, pos: 16
type: B, layer: 1, pos: 23
type: B, layer: 1, pos: 39
type: B, layer: 1, pos: 33
type: A, layer: 1, pos: 33
type: A, layer: 1, pos: 47
type: A, layer: 1, pos: 30

Time for candidate selection: 0.07 seconds

### Candidate
type: A, layer: 1, pos: 11

### Candidate
type: B, layer: 1, pos: 34

## Relational analysis of NS_A1_A2_B2_B1_B1
Optimization infeasible because this subproblem isn't reachable.

### Candidate
type: B, layer: 1, pos: 47

## Relational analysis of NS_A1_A2_B2_B1_B1

### Relational analysis result of NS_A1_A2_B2_B1_B1
Status: Status.UNKNOWN
Output dim: 3, lower bound: -20.6038471, upper bound: 20.6006222
time: 0.66 seconds

## Relational analysis of NS_A1_A2_B2_B1_B2

### Relational analysis result of NS_A1_A2_B2_B1_B2
Status: Status.UNKNOWN
Output dim: 3, lower bound: -20.6040001, upper bound: 20.6028097
time: 0.67 seconds

## BFS NS instance: NS_A1_A2_B2_B2

### Backsubstitution after applying NS history:
0: -3.5923164, 3.1570973, -3.9835007, 3.5414548, -7.1337709, 7.1405973
1: -14.2131710, 12.2455807, -15.7446146, 13.7117996, -27.9249706, 27.9901943
2: -7.1067591, 11.4467535, -7.8555064, 12.8895645, -19.9963226, 19.3022575
3: -12.4342022, 11.1442776, -13.7488871, 12.5138626, -24.9480648, 24.8931599
4: -9.1061058, 11.6097670, -10.0535917, 13.0814514, -22.1875572, 21.6633587

Time for backsubstitution: 0.88 seconds

### NS candidates at layer 1
type: A, layer: 1, pos: 11
type: B, layer: 1, pos: 34
type: A, layer: 1, pos: 42
type: B, layer: 1, pos: 42
type: B, layer: 1, pos: 24
type: B, layer: 1, pos: 4
type: B, layer: 1, pos: 12
type: B, layer: 1, pos: 11
type: A, layer: 1, pos: 22
type: A, layer: 1, pos: 4
type: A, layer: 1, pos: 34
type: A, layer: 1, pos: 12
type: B, layer: 1, pos: 9
type: A, layer: 1, pos: 47
type: B, layer: 1, pos: 39
type: A, layer: 1, pos: 16
type: B, layer: 1, pos: 16
type: A, layer: 1, pos: 39
type: B, layer: 1, pos: 47
type: A, layer: 1, pos: 31
type: B, layer: 1, pos: 31
type: A, layer: 1, pos: 23
type: B, layer: 1, pos: 23
type: A, layer: 1, pos: 33
type: B, layer: 1, pos: 33
type: A, layer: 1, pos: 30

Time for candidate selection: 0.07 seconds

### Candidate
type: A, layer: 1, pos: 11

### Candidate
type: B, layer: 1, pos: 34

## Relational analysis of NS_A1_A2_B2_B2_B1
Optimization infeasible because this subproblem isn't reachable.

### Candidate
type: A, layer: 1, pos: 42

## Relational analysis of NS_A1_A2_B2_B2_A1

### Relational analysis result of NS_A1_A2_B2_B2_A1
Status: Status.VERIFIED
Output dim: 3, lower bound: -20.6015011, upper bound: 20.6020827
time: 0.87 seconds

## Relational analysis of NS_A1_A2_B2_B2_A2

### Relational analysis result of NS_A1_A2_B2_B2_A2
Status: Status.UNKNOWN
Output dim: 3, lower bound: -20.6039518, upper bound: 20.6022874
time: 0.78 seconds

## Summary of splitting at layer (split count: 4)
- Time for NS candidates: 2.93 seconds
NS_A1_A2_B1_B1_B1, status: Status.UNKNOWN, split count: 5, time: 2.93
Output dim: 3, lower bound: -20.6032768, upper bound: 20.5970383
NS_A1_A2_B1_B1_B2, status: Status.UNKNOWN, split count: 5, time: 2.93
Output dim: 3, lower bound: -20.6036034, upper bound: 20.5995744
NS_A1_A2_B1_B2_A1, status: Status.VERIFIED, split count: 5, time: 2.93
Output dim: 3, lower bound: -20.6009425, upper bound: 20.5983351
NS_A1_A2_B1_B2_A2, status: Status.UNKNOWN, split count: 5, time: 2.93
Output dim: 3, lower bound: -20.6033932, upper bound: 20.5985397
NS_A1_A2_B2_B1_B1, status: Status.UNKNOWN, split count: 5, time: 2.93
Output dim: 3, lower bound: -20.6038471, upper bound: 20.6006222
NS_A1_A2_B2_B1_B2, status: Status.UNKNOWN, split count: 5, time: 2.93
Output dim: 3, lower bound: -20.6040001, upper bound: 20.6028097
NS_A1_A2_B2_B2_A1, status: Status.VERIFIED, split count: 5, time: 2.93
Output dim: 3, lower bound: -20.6015011, upper bound: 20.6020827
NS_A1_A2_B2_B2_A2, status: Status.UNKNOWN, split count: 5, time: 2.93
Output dim: 3, lower bound: -20.6039518, upper bound: 20.6022874

## BFS NS instance: NS_A1_A2_B1_B1_B1

### Backsubstitution after applying NS history:
0: -3.5923164, 3.1570973, -2.9222052, 2.6103210, -6.2026377, 6.0793023
1: -14.2131710, 12.2455807, -11.5062580, 10.1043606, -24.3175316, 23.7518368
2: -7.1067591, 11.4467535, -5.7777696, 9.4893036, -16.5960617, 17.2245197
3: -12.4342022, 11.1442776, -10.0655785, 9.2635679, -21.6977692, 21.2098560
4: -9.1061058, 11.6097670, -7.3661041, 9.6684942, -18.7746010, 18.9758720

Time for backsubstitution: 0.88 seconds

### NS candidates at layer 1
type: A, layer: 1, pos: 11
type: B, layer: 1, pos: 11
type: B, layer: 1, pos: 34
type: A, layer: 1, pos: 34
type: B, layer: 1, pos: 24
type: A, layer: 1, pos: 22
type: B, layer: 1, pos: 12
type: B, layer: 1, pos: 4
type: A, layer: 1, pos: 12
type: B, layer: 1, pos: 9
type: A, layer: 1, pos: 4
type: B, layer: 1, pos: 42
type: A, layer: 1, pos: 42
type: B, layer: 1, pos: 31
type: A, layer: 1, pos: 31
type: A, layer: 1, pos: 39
type: B, layer: 1, pos: 16
type: B, layer: 1, pos: 33
type: A, layer: 1, pos: 16
type: A, layer: 1, pos: 23
type: A, layer: 1, pos: 33
type: B, layer: 1, pos: 39
type: B, layer: 1, pos: 23
type: A, layer: 1, pos: 47
type: A, layer: 1, pos: 30

Time for candidate selection: 0.07 seconds

### Candidate
type: A, layer: 1, pos: 11

### Candidate
type: B, layer: 1, pos: 11

### Candidate
type: B, layer: 1, pos: 34

## Relational analysis of NS_A1_A2_B1_B1_B1_B1
Optimization infeasible because this subproblem isn't reachable.

### Candidate
type: A, layer: 1, pos: 34

### Candidate
type: B, layer: 1, pos: 24

## Relational analysis of NS_A1_A2_B1_B1_B1_B1

### Relational analysis result of NS_A1_A2_B1_B1_B1_B1
Status: Status.UNKNOWN
Output dim: 3, lower bound: -20.6032768, upper bound: 20.5970383
time: 0.67 seconds

## Relational analysis of NS_A1_A2_B1_B1_B1_B2

### Relational analysis result of NS_A1_A2_B1_B1_B1_B2
Status: Status.UNKNOWN
Output dim: 3, lower bound: -20.6032768, upper bound: 20.5970383
time: 0.79 seconds

## BFS NS instance: NS_A1_A2_B1_B1_B2

### Backsubstitution after applying NS history:
0: -3.5923164, 3.1570973, -3.0748661, 2.7457089, -6.3380251, 6.2319617
1: -14.2131710, 12.2455807, -12.1110201, 10.6581306, -24.8713017, 24.3565979
2: -7.1067591, 11.4467535, -6.0808282, 9.9635391, -17.0702972, 17.5275764
3: -12.4342022, 11.1442776, -10.6057835, 9.7708673, -22.2050705, 21.7500591
4: -9.1061058, 11.6097670, -7.7756262, 10.1745081, -19.2806129, 19.3853931

Time for backsubstitution: 1.08 seconds

### NS candidates at layer 1
type: B, layer: 1, pos: 11
type: A, layer: 1, pos: 11
type: B, layer: 1, pos: 34
type: B, layer: 1, pos: 4
type: A, layer: 1, pos: 4
type: B, layer: 1, pos: 12
type: A, layer: 1, pos: 42
type: B, layer: 1, pos: 42
type: A, layer: 1, pos: 22
type: B, layer: 1, pos: 9
type: A, layer: 1, pos: 12
type: A, layer: 1, pos: 34
type: B, layer: 1, pos: 24
type: A, layer: 1, pos: 39
type: B, layer: 1, pos: 33
type: A, layer: 1, pos: 31
type: A, layer: 1, pos: 23
type: B, layer: 1, pos: 31
type: B, layer: 1, pos: 16
type: A, layer: 1, pos: 16
type: B, layer: 1, pos: 39
type: B, layer: 1, pos: 23
type: A, layer: 1, pos: 33
type: A, layer: 1, pos: 30
type: A, layer: 1, pos: 47

Time for candidate selection: 0.07 seconds

### Candidate
type: B, layer: 1, pos: 11

### Candidate
type: A, layer: 1, pos: 11

### Candidate
type: B, layer: 1, pos: 34

## Relational analysis of NS_A1_A2_B1_B1_B2_B1
Optimization infeasible because this subproblem isn't reachable.

### Candidate
type: B, layer: 1, pos: 4

## Relational analysis of NS_A1_A2_B1_B1_B2_B1

### Relational analysis result of NS_A1_A2_B1_B1_B2_B1
Status: Status.UNKNOWN
Output dim: 3, lower bound: -20.6035522, upper bound: 20.5994820
time: 0.74 seconds

## Relational analysis of NS_A1_A2_B1_B1_B2_B2

### Relational analysis result of NS_A1_A2_B1_B1_B2_B2
Status: Status.UNKNOWN
Output dim: 3, lower bound: -20.6035806, upper bound: 20.5993457
time: 0.63 seconds

## BFS NS instance: NS_A1_A2_B1_B2_A2

### Backsubstitution after applying NS history:
0: -3.4072533, 2.9987469, -3.2859685, 2.9183435, -6.3255968, 6.2847147
1: -13.4742098, 11.6417627, -12.9850140, 11.3200541, -24.7942600, 24.6267738
2: -6.7381449, 10.8741417, -6.5043592, 10.5621500, -17.3002949, 17.3785019
3: -11.7894468, 10.6054029, -11.3635283, 10.3372755, -22.1267223, 21.9689274
4: -8.6332855, 11.0345154, -8.3101521, 10.7604237, -19.3937092, 19.3446655

Time for backsubstitution: 1.04 seconds

### NS candidates at layer 1
type: A, layer: 1, pos: 11
type: B, layer: 1, pos: 11
type: B, layer: 1, pos: 34
type: A, layer: 1, pos: 34
type: B, layer: 1, pos: 24
type: B, layer: 1, pos: 12
type: A, layer: 1, pos: 22
type: B, layer: 1, pos: 47
type: A, layer: 1, pos: 12
type: B, layer: 1, pos: 4
type: B, layer: 1, pos: 42
type: B, layer: 1, pos: 9
type: A, layer: 1, pos: 4
type: B, layer: 1, pos: 31
type: A, layer: 1, pos: 31
type: B, layer: 1, pos: 33
type: A, layer: 1, pos: 39
type: B, layer: 1, pos: 39
type: A, layer: 1, pos: 33
type: B, layer: 1, pos: 16
type: A, layer: 1, pos: 16
type: A, layer: 1, pos: 23
type: B, layer: 1, pos: 23
type: A, layer: 1, pos: 47
type: A, layer: 1, pos: 30

Time for candidate selection: 0.11 seconds

### Candidate
type: A, layer: 1, pos: 11

### Candidate
type: B, layer: 1, pos: 11

### Candidate
type: B, layer: 1, pos: 34

### Candidate
type: A, layer: 1, pos: 34

### Candidate
type: B, layer: 1, pos: 24

## Relational analysis of NS_A1_A2_B1_B2_A2_B1

### Relational analysis result of NS_A1_A2_B1_B2_A2_B1
Status: Status.UNKNOWN
Output dim: 3, lower bound: -20.6033932, upper bound: 20.5985397
time: 0.86 seconds

## Relational analysis of NS_A1_A2_B1_B2_A2_B2

### Relational analysis result of NS_A1_A2_B1_B2_A2_B2
Status: Status.UNKNOWN
Output dim: 3, lower bound: -20.6033932, upper bound: 20.5985397
time: 0.74 seconds

## BFS NS instance: NS_A1_A2_B2_B1_B1

### Backsubstitution after applying NS history:
0: -3.5923164, 3.1570973, -3.5576327, 3.1796906, -6.7720065, 6.7147303
1: -14.2131710, 12.2455807, -14.0135107, 12.3078413, -26.5210114, 26.2590885
2: -7.1067591, 11.4467535, -7.0061011, 11.5805044, -18.6872616, 18.4528542
3: -12.4342022, 11.1442776, -12.2298727, 11.2740774, -23.7082787, 23.3741493
4: -9.1061058, 11.6097670, -8.9676342, 11.7635689, -20.8696747, 20.5773983

Time for backsubstitution: 0.95 seconds

### NS candidates at layer 1
type: A, layer: 1, pos: 11
type: B, layer: 1, pos: 34
type: B, layer: 1, pos: 24
type: B, layer: 1, pos: 4
type: A, layer: 1, pos: 22
type: B, layer: 1, pos: 9
type: B, layer: 1, pos: 12
type: A, layer: 1, pos: 4
type: A, layer: 1, pos: 34
type: A, layer: 1, pos: 12
type: B, layer: 1, pos: 11
type: A, layer: 1, pos: 42
type: B, layer: 1, pos: 42
type: B, layer: 1, pos: 39
type: A, layer: 1, pos: 39
type: B, layer: 1, pos: 31
type: B, layer: 1, pos: 16
type: A, layer: 1, pos: 31
type: A, layer: 1, pos: 16
type: A, layer: 1, pos: 23
type: A, layer: 1, pos: 33
type: B, layer: 1, pos: 23
type: B, layer: 1, pos: 33
type: A, layer: 1, pos: 47
type: A, layer: 1, pos: 30

Time for candidate selection: 0.07 seconds

### Candidate
type: A, layer: 1, pos: 11

### Candidate
type: B, layer: 1, pos: 34

### Candidate
type: B, layer: 1, pos: 24

## Relational analysis of NS_A1_A2_B2_B1_B1_B1

### Relational analysis result of NS_A1_A2_B2_B1_B1_B1
Status: Status.UNKNOWN
Output dim: 3, lower bound: -20.6038471, upper bound: 20.6006222
time: 0.76 seconds

## Relational analysis of NS_A1_A2_B2_B1_B1_B2

### Relational analysis result of NS_A1_A2_B2_B1_B1_B2
Status: Status.UNKNOWN
Output dim: 3, lower bound: -20.6038471, upper bound: 20.6006222
time: 0.75 seconds

## BFS NS instance: NS_A1_A2_B2_B1_B2

### Backsubstitution after applying NS history:
0: -3.5923164, 3.1570973, -3.7686472, 3.3631830, -6.9554992, 6.9257441
1: -14.2131710, 12.2455807, -14.8608608, 13.0287399, -27.2419109, 27.1064415
2: -7.1067591, 11.4467535, -7.4229341, 12.2570028, -19.3637619, 18.8696842
3: -12.4342022, 11.1442776, -12.9745588, 11.9410810, -24.3752823, 24.1188335
4: -9.1061058, 11.6097670, -9.5114717, 12.4552126, -21.5613174, 21.1212368

Time for backsubstitution: 1.18 seconds

### NS candidates at layer 1
type: A, layer: 1, pos: 11
type: B, layer: 1, pos: 34
type: B, layer: 1, pos: 11
type: B, layer: 1, pos: 4
type: B, layer: 1, pos: 12
type: A, layer: 1, pos: 42
type: B, layer: 1, pos: 42
type: B, layer: 1, pos: 9
type: A, layer: 1, pos: 22
type: A, layer: 1, pos: 4
type: A, layer: 1, pos: 12
type: B, layer: 1, pos: 24
type: A, layer: 1, pos: 34
type: A, layer: 1, pos: 39
type: A, layer: 1, pos: 31
type: B, layer: 1, pos: 16
type: B, layer: 1, pos: 39
type: A, layer: 1, pos: 23
type: B, layer: 1, pos: 31
type: A, layer: 1, pos: 16
type: B, layer: 1, pos: 33
type: B, layer: 1, pos: 23
type: A, layer: 1, pos: 33
type: A, layer: 1, pos: 47
type: A, layer: 1, pos: 30

Time for candidate selection: 0.07 seconds

### Candidate
type: A, layer: 1, pos: 11

### Candidate
type: B, layer: 1, pos: 34

### Candidate
type: B, layer: 1, pos: 11

### Candidate
type: B, layer: 1, pos: 4

## Relational analysis of NS_A1_A2_B2_B1_B2_B1

### Relational analysis result of NS_A1_A2_B2_B1_B2_B1
Status: Status.UNKNOWN
Output dim: 3, lower bound: -20.6039446, upper bound: 20.6026866
time: 0.63 seconds

## Relational analysis of NS_A1_A2_B2_B1_B2_B2

### Relational analysis result of NS_A1_A2_B2_B1_B2_B2
Status: Status.UNKNOWN
Output dim: 3, lower bound: -20.6037862, upper bound: 20.6002504
time: 0.69 seconds

## BFS NS instance: NS_A1_A2_B2_B2_A2

### Backsubstitution after applying NS history:
0: -3.4072533, 2.9987469, -3.9835007, 3.5414548, -6.9487081, 6.9822464
1: -13.4742098, 11.6417627, -15.7446146, 13.7117996, -27.1860085, 27.3863773
2: -6.7381449, 10.8741417, -7.8555064, 12.8895645, -19.6277084, 18.7296467
3: -11.7894468, 10.6054029, -13.7488871, 12.5138626, -24.3033104, 24.3542843
4: -8.6332855, 11.0345154, -10.0535917, 13.0814514, -21.7147369, 21.0881081

Time for backsubstitution: 1.03 seconds

### NS candidates at layer 1
type: A, layer: 1, pos: 11
type: B, layer: 1, pos: 34
type: B, layer: 1, pos: 24
type: B, layer: 1, pos: 11
type: B, layer: 1, pos: 12
type: A, layer: 1, pos: 34
type: A, layer: 1, pos: 22
type: A, layer: 1, pos: 12
type: A, layer: 1, pos: 47
type: B, layer: 1, pos: 9
type: B, layer: 1, pos: 42
type: B, layer: 1, pos: 4
type: A, layer: 1, pos: 31
type: B, layer: 1, pos: 31
type: A, layer: 1, pos: 4
type: B, layer: 1, pos: 39
type: B, layer: 1, pos: 16
type: A, layer: 1, pos: 39
type: A, layer: 1, pos: 16
type: A, layer: 1, pos: 33
type: B, layer: 1, pos: 23
type: B, layer: 1, pos: 33
type: A, layer: 1, pos: 23
type: A, layer: 1, pos: 30
type: B, layer: 1, pos: 47

Time for candidate selection: 0.08 seconds

### Candidate
type: A, layer: 1, pos: 11

### Candidate
type: B, layer: 1, pos: 34

### Candidate
type: B, layer: 1, pos: 24

## Relational analysis of NS_A1_A2_B2_B2_A2_B1

### Relational analysis result of NS_A1_A2_B2_B2_A2_B1
Status: Status.UNKNOWN
Output dim: 3, lower bound: -20.6039518, upper bound: 20.6020827
time: 1.03 seconds

## Relational analysis of NS_A1_A2_B2_B2_A2_B2

### Relational analysis result of NS_A1_A2_B2_B2_A2_B2
Status: Status.UNKNOWN
Output dim: 3, lower bound: -20.6039518, upper bound: 20.6022874
time: 1.09 seconds

## Summary of splitting at layer (split count: 5)
- Time for NS candidates: 3.24 seconds
NS_A1_A2_B1_B1_B1_B1, status: Status.UNKNOWN, split count: 6, time: 3.24
Output dim: 3, lower bound: -20.6032768, upper bound: 20.5970383
NS_A1_A2_B1_B1_B1_B2, status: Status.UNKNOWN, split count: 6, time: 3.24
Output dim: 3, lower bound: -20.6032768, upper bound: 20.5970383
NS_A1_A2_B1_B1_B2_B1, status: Status.UNKNOWN, split count: 6, time: 3.24
Output dim: 3, lower bound: -20.6035522, upper bound: 20.5994820
NS_A1_A2_B1_B1_B2_B2, status: Status.UNKNOWN, split count: 6, time: 3.24
Output dim: 3, lower bound: -20.6035806, upper bound: 20.5993457
NS_A1_A2_B1_B2_A2_B1, status: Status.UNKNOWN, split count: 6, time: 3.24
Output dim: 3, lower bound: -20.6033932, upper bound: 20.5985397
NS_A1_A2_B1_B2_A2_B2, status: Status.UNKNOWN, split count: 6, time: 3.24
Output dim: 3, lower bound: -20.6033932, upper bound: 20.5985397
NS_A1_A2_B2_B1_B1_B1, status: Status.UNKNOWN, split count: 6, time: 3.24
Output dim: 3, lower bound: -20.6038471, upper bound: 20.6006222
NS_A1_A2_B2_B1_B1_B2, status: Status.UNKNOWN, split count: 6, time: 3.24
Output dim: 3, lower bound: -20.6038471, upper bound: 20.6006222
NS_A1_A2_B2_B1_B2_B1, status: Status.UNKNOWN, split count: 6, time: 3.24
Output dim: 3, lower bound: -20.6039446, upper bound: 20.6026866
NS_A1_A2_B2_B1_B2_B2, status: Status.UNKNOWN, split count: 6, time: 3.24
Output dim: 3, lower bound: -20.6037862, upper bound: 20.6002504
NS_A1_A2_B2_B2_A2_B1, status: Status.UNKNOWN, split count: 6, time: 3.24
Output dim: 3, lower bound: -20.6039518, upper bound: 20.6020827
NS_A1_A2_B2_B2_A2_B2, status: Status.UNKNOWN, split count: 6, time: 3.24
Output dim: 3, lower bound: -20.6039518, upper bound: 20.6022874

## BFS NS instance: NS_A1_A2_B1_B1_B1_B1

### Backsubstitution after applying NS history:
0: -3.5923164, 3.1570973, -2.8369715, 2.5377181, -6.1300344, 5.9940677
1: -14.2131710, 12.2455807, -11.1648302, 9.8173466, -24.0305176, 23.4104061
2: -7.1067591, 11.4467535, -5.6087904, 9.2279978, -16.3347569, 17.0555439
3: -12.4342022, 11.1442776, -9.7681313, 9.0039520, -21.4381542, 20.9124050
4: -9.1061058, 11.6097670, -7.1507564, 9.4026794, -18.5087852, 18.7605228

Time for backsubstitution: 1.43 seconds

### NS candidates at layer 1
type: A, layer: 1, pos: 11
type: B, layer: 1, pos: 11
type: B, layer: 1, pos: 34
type: A, layer: 1, pos: 34
type: A, layer: 1, pos: 22
type: B, layer: 1, pos: 12
type: B, layer: 1, pos: 4
type: A, layer: 1, pos: 12
type: B, layer: 1, pos: 9
type: A, layer: 1, pos: 4
type: B, layer: 1, pos: 42
type: A, layer: 1, pos: 42
type: B, layer: 1, pos: 31
type: A, layer: 1, pos: 31
type: A, layer: 1, pos: 39
type: B, layer: 1, pos: 33
type: B, layer: 1, pos: 16
type: A, layer: 1, pos: 16
type: A, layer: 1, pos: 23
type: A, layer: 1, pos: 33
type: B, layer: 1, pos: 39
type: B, layer: 1, pos: 23
type: A, layer: 1, pos: 47
type: A, layer: 1, pos: 30

Time for candidate selection: 0.11 seconds

### Candidate
type: A, layer: 1, pos: 11

### Candidate
type: B, layer: 1, pos: 11

### Candidate
type: B, layer: 1, pos: 34

### Candidate
type: A, layer: 1, pos: 34

### Candidate
type: A, layer: 1, pos: 22

## Relational analysis of NS_A1_A2_B1_B1_B1_B1_A1

### Relational analysis result of NS_A1_A2_B1_B1_B1_B1_A1
Status: Status.UNKNOWN
Output dim: 3, lower bound: -20.6032768, upper bound: 20.5970383
time: 0.69 seconds

## Relational analysis of NS_A1_A2_B1_B1_B1_B1_A2

### Relational analysis result of NS_A1_A2_B1_B1_B1_B1_A2
Status: Status.UNKNOWN
Output dim: 3, lower bound: -20.6032768, upper bound: 20.5970383
time: 0.91 seconds

## BFS NS instance: NS_A1_A2_B1_B1_B1_B2

### Backsubstitution after applying NS history:
0: -3.5923164, 3.1570973, -3.4805706, 3.1200867, -6.7124033, 6.6376667
1: -14.2131710, 12.2455807, -13.7218876, 12.0228958, -26.2360668, 25.9674606
2: -7.1067591, 11.4467535, -6.8483577, 11.4872770, -18.5940361, 18.2951107
3: -12.4342022, 11.1442776, -11.9613485, 11.0371389, -23.4713402, 23.1056252
4: -9.1061058, 11.6097670, -8.7219229, 11.6184912, -20.7245979, 20.3316879

Time for backsubstitution: 1.31 seconds

### NS candidates at layer 1
type: A, layer: 1, pos: 11
type: B, layer: 1, pos: 11
type: A, layer: 1, pos: 34
type: B, layer: 1, pos: 34
type: B, layer: 1, pos: 9
type: A, layer: 1, pos: 22
type: B, layer: 1, pos: 12
type: B, layer: 1, pos: 4
type: A, layer: 1, pos: 47
type: A, layer: 1, pos: 4
type: A, layer: 1, pos: 12
type: B, layer: 1, pos: 42
type: A, layer: 1, pos: 42
type: B, layer: 1, pos: 31
type: A, layer: 1, pos: 31
type: B, layer: 1, pos: 39
type: A, layer: 1, pos: 16
type: B, layer: 1, pos: 16
type: A, layer: 1, pos: 33
type: B, layer: 1, pos: 33
type: A, layer: 1, pos: 39
type: B, layer: 1, pos: 23
type: A, layer: 1, pos: 23
type: A, layer: 1, pos: 30

Time for candidate selection: 0.11 seconds

### Candidate
type: A, layer: 1, pos: 11

### Candidate
type: B, layer: 1, pos: 11

### Candidate
type: A, layer: 1, pos: 34

### Candidate
type: B, layer: 1, pos: 34

### Candidate
type: B, layer: 1, pos: 9

## Relational analysis of NS_A1_A2_B1_B1_B1_B2_B1
Optimization infeasible because this subproblem isn't reachable.

### Candidate
type: A, layer: 1, pos: 22

## Relational analysis of NS_A1_A2_B1_B1_B1_B2_A1

### Relational analysis result of NS_A1_A2_B1_B1_B1_B2_A1
Status: Status.UNKNOWN
Output dim: 3, lower bound: -20.6032768, upper bound: 20.5970383
time: 0.81 seconds

## Relational analysis of NS_A1_A2_B1_B1_B1_B2_A2

### Relational analysis result of NS_A1_A2_B1_B1_B1_B2_A2
Status: Status.UNKNOWN
Output dim: 3, lower bound: -20.6032768, upper bound: 20.5970383
time: 0.64 seconds

## BFS NS instance: NS_A1_A2_B1_B1_B2_B1

### Backsubstitution after applying NS history:
0: -3.5923164, 3.1570973, -2.6776226, 2.3901095, -5.9824257, 5.8347197
1: -14.2131710, 12.2455807, -10.5423784, 9.2746677, -23.4878387, 22.7879581
2: -7.1067591, 11.4467535, -5.3035603, 8.6706009, -15.7773600, 16.7503128
3: -12.4342022, 11.1442776, -9.2467489, 8.5087490, -20.9429493, 20.3910255
4: -9.1061058, 11.6097670, -6.7716413, 8.8663626, -17.9724693, 18.3814068

Time for backsubstitution: 1.23 seconds

### NS candidates at layer 1
type: B, layer: 1, pos: 11
type: A, layer: 1, pos: 11
type: B, layer: 1, pos: 34
type: A, layer: 1, pos: 42
type: B, layer: 1, pos: 42
type: A, layer: 1, pos: 34
type: B, layer: 1, pos: 12
type: A, layer: 1, pos: 22
type: A, layer: 1, pos: 12
type: B, layer: 1, pos: 9
type: A, layer: 1, pos: 39
type: B, layer: 1, pos: 33
type: B, layer: 1, pos: 31
type: A, layer: 1, pos: 31
type: B, layer: 1, pos: 24
type: A, layer: 1, pos: 23
type: B, layer: 1, pos: 16
type: A, layer: 1, pos: 16
type: A, layer: 1, pos: 4
type: B, layer: 1, pos: 39
type: B, layer: 1, pos: 23
type: A, layer: 1, pos: 33
type: A, layer: 1, pos: 30
type: A, layer: 1, pos: 47

Time for candidate selection: 0.10 seconds

### Candidate
type: B, layer: 1, pos: 11

### Candidate
type: A, layer: 1, pos: 11

### Candidate
type: B, layer: 1, pos: 34

### Candidate
type: A, layer: 1, pos: 42

## Relational analysis of NS_A1_A2_B1_B1_B2_B1_A1

### Relational analysis result of NS_A1_A2_B1_B1_B2_B1_A1
Status: Status.VERIFIED
Output dim: 3, lower bound: -20.6011146, upper bound: 20.5993392
time: 0.77 seconds

## Relational analysis of NS_A1_A2_B1_B1_B2_B1_A2

### Relational analysis result of NS_A1_A2_B1_B1_B2_B1_A2
Status: Status.UNKNOWN
Output dim: 3, lower bound: -20.6035437, upper bound: 20.5994820
time: 0.77 seconds

## BFS NS instance: NS_A1_A2_B1_B1_B2_B2

### Backsubstitution after applying NS history:
0: -3.5923164, 3.1570973, -3.0070853, 2.6840725, -6.2763886, 6.1641817
1: -14.2131710, 12.2455807, -11.8437071, 10.4111977, -24.6243687, 24.0892868
2: -7.1067591, 11.4467535, -5.9487362, 9.7418251, -16.8485832, 17.3954887
3: -12.4342022, 11.1442776, -10.3784885, 9.5441866, -21.9783859, 21.5227661
4: -9.1061058, 11.6097670, -7.6077647, 9.9460669, -19.0521736, 19.2175312

Time for backsubstitution: 1.23 seconds

### NS candidates at layer 1
type: B, layer: 1, pos: 11
type: A, layer: 1, pos: 11
type: B, layer: 1, pos: 34
type: B, layer: 1, pos: 12
type: A, layer: 1, pos: 34
type: A, layer: 1, pos: 22
type: B, layer: 1, pos: 9
type: A, layer: 1, pos: 12
type: A, layer: 1, pos: 42
type: B, layer: 1, pos: 42
type: B, layer: 1, pos: 24
type: B, layer: 1, pos: 33
type: A, layer: 1, pos: 39
type: A, layer: 1, pos: 31
type: A, layer: 1, pos: 23
type: B, layer: 1, pos: 31
type: A, layer: 1, pos: 4
type: B, layer: 1, pos: 16
type: A, layer: 1, pos: 16
type: B, layer: 1, pos: 39
type: B, layer: 1, pos: 23
type: A, layer: 1, pos: 33
type: A, layer: 1, pos: 30
type: A, layer: 1, pos: 47

Time for candidate selection: 0.11 seconds

### Candidate
type: B, layer: 1, pos: 11

### Candidate
type: A, layer: 1, pos: 11

### Candidate
type: B, layer: 1, pos: 34

### Candidate
type: B, layer: 1, pos: 12

## Relational analysis of NS_A1_A2_B1_B1_B2_B2_B1
Optimization infeasible because this subproblem isn't reachable.

### Candidate
type: A, layer: 1, pos: 34

### Candidate
type: A, layer: 1, pos: 22

## Relational analysis of NS_A1_A2_B1_B1_B2_B2_A1

### Relational analysis result of NS_A1_A2_B1_B1_B2_B2_A1
Status: Status.UNKNOWN
Output dim: 3, lower bound: -20.6035806, upper bound: 20.5993457
time: 0.74 seconds

## Relational analysis of NS_A1_A2_B1_B1_B2_B2_A2

### Relational analysis result of NS_A1_A2_B1_B1_B2_B2_A2
Status: Status.UNKNOWN
Output dim: 3, lower bound: -20.6035806, upper bound: 20.5993457
time: 0.69 seconds

## BFS NS instance: NS_A1_A2_B1_B2_A2_B1

### Backsubstitution after applying NS history:
0: -3.4072533, 2.9987469, -3.1507802, 2.8027632, -6.2100163, 6.1495266
1: -13.4742098, 11.6417627, -12.4441538, 10.8765450, -24.3507519, 24.0859165
2: -6.7381449, 10.8741417, -6.2334828, 10.1443806, -16.8825226, 17.1076241
3: -11.7894468, 10.6054029, -10.8921318, 9.9383621, -21.7278099, 21.4975319
4: -8.6332855, 11.0345154, -7.9639950, 10.3374100, -18.9706955, 18.9985104

Time for backsubstitution: 1.25 seconds

### NS candidates at layer 1
type: A, layer: 1, pos: 11
type: B, layer: 1, pos: 11
type: B, layer: 1, pos: 34
type: A, layer: 1, pos: 34
type: B, layer: 1, pos: 12
type: A, layer: 1, pos: 22
type: A, layer: 1, pos: 12
type: B, layer: 1, pos: 47
type: B, layer: 1, pos: 4
type: B, layer: 1, pos: 9
type: B, layer: 1, pos: 42
type: A, layer: 1, pos: 4
type: B, layer: 1, pos: 31
type: A, layer: 1, pos: 31
type: B, layer: 1, pos: 33
type: A, layer: 1, pos: 39
type: B, layer: 1, pos: 39
type: A, layer: 1, pos: 33
type: A, layer: 1, pos: 16
type: B, layer: 1, pos: 16
type: A, layer: 1, pos: 30
type: A, layer: 1, pos: 23
type: A, layer: 1, pos: 47
type: B, layer: 1, pos: 23

Time for candidate selection: 0.09 seconds

### Candidate
type: A, layer: 1, pos: 11

### Candidate
type: B, layer: 1, pos: 11

### Candidate
type: B, layer: 1, pos: 34

### Candidate
type: A, layer: 1, pos: 34

### Candidate
type: B, layer: 1, pos: 12

## Relational analysis of NS_A1_A2_B1_B2_A2_B1_B1
Optimization infeasible because this subproblem isn't reachable.

### Candidate
type: A, layer: 1, pos: 22

## Relational analysis of NS_A1_A2_B1_B2_A2_B1_A1

### Relational analysis result of NS_A1_A2_B1_B2_A2_B1_A1
Status: Status.UNKNOWN
Output dim: 3, lower bound: -20.6033932, upper bound: 20.5985397
time: 0.76 seconds

## Relational analysis of NS_A1_A2_B1_B2_A2_B1_A2

### Relational analysis result of NS_A1_A2_B1_B2_A2_B1_A2
Status: Status.UNKNOWN
Output dim: 3, lower bound: -20.6033932, upper bound: 20.5985397
time: 0.70 seconds

## BFS NS instance: NS_A1_A2_B1_B2_A2_B2

### Backsubstitution after applying NS history:
0: -3.4072533, 2.9987469, -3.7443836, 3.3308897, -6.7381420, 6.7431293
1: -13.4742098, 11.6417627, -14.8040266, 12.8840399, -26.3582497, 26.4457893
2: -6.7381449, 10.8741417, -7.3696198, 12.1977873, -18.9359322, 18.2437611
3: -11.7894468, 10.6054029, -12.9243069, 11.7908812, -23.5803280, 23.5297070
4: -8.6332855, 11.0345154, -9.4258900, 12.3335075, -20.9667931, 20.4604053

Time for backsubstitution: 1.28 seconds

### NS candidates at layer 1
type: A, layer: 1, pos: 11
type: B, layer: 1, pos: 11
type: A, layer: 1, pos: 47
type: A, layer: 1, pos: 34
type: B, layer: 1, pos: 34
type: B, layer: 1, pos: 12
type: B, layer: 1, pos: 9
type: A, layer: 1, pos: 22
type: A, layer: 1, pos: 30
type: A, layer: 1, pos: 12
type: B, layer: 1, pos: 31
type: A, layer: 1, pos: 31
type: B, layer: 1, pos: 4
type: B, layer: 1, pos: 42
type: A, layer: 1, pos: 4
type: A, layer: 1, pos: 16
type: B, layer: 1, pos: 16
type: B, layer: 1, pos: 39
type: B, layer: 1, pos: 23
type: A, layer: 1, pos: 33
type: B, layer: 1, pos: 33
type: A, layer: 1, pos: 23
type: A, layer: 1, pos: 39
type: B, layer: 1, pos: 47

Time for candidate selection: 0.11 seconds

### Candidate
type: A, layer: 1, pos: 11

### Candidate
type: B, layer: 1, pos: 11

### Candidate
type: A, layer: 1, pos: 47

## Relational analysis of NS_A1_A2_B1_B2_A2_B2_A1

### Relational analysis result of NS_A1_A2_B1_B2_A2_B2_A1
Status: Status.VERIFIED
Output dim: 3, lower bound: -20.6002979, upper bound: 20.5982526
time: 0.72 seconds

## Relational analysis of NS_A1_A2_B1_B2_A2_B2_A2

### Relational analysis result of NS_A1_A2_B1_B2_A2_B2_A2
Status: Status.UNKNOWN
Output dim: 3, lower bound: -20.6033932, upper bound: 20.5985222
time: 1.08 seconds

## BFS NS instance: NS_A1_A2_B2_B1_B1_B1

### Backsubstitution after applying NS history:
0: -3.5923164, 3.1570973, -3.4812961, 3.1150770, -6.7073936, 6.6383924
1: -14.2131710, 12.2455807, -13.7059050, 12.0545502, -26.2677212, 25.9514847
2: -7.1067591, 11.4467535, -6.8541398, 11.3458900, -18.4526482, 18.3008919
3: -12.4342022, 11.1442776, -11.9621525, 11.0443764, -23.4785767, 23.1064281
4: -9.1061058, 11.6097670, -8.7754803, 11.5260696, -20.6321754, 20.3852425

Time for backsubstitution: 1.29 seconds

### NS candidates at layer 1
type: A, layer: 1, pos: 11
type: B, layer: 1, pos: 34
type: A, layer: 1, pos: 22
type: B, layer: 1, pos: 9
type: B, layer: 1, pos: 4
type: B, layer: 1, pos: 12
type: A, layer: 1, pos: 34
type: A, layer: 1, pos: 4
type: A, layer: 1, pos: 12
type: A, layer: 1, pos: 42
type: B, layer: 1, pos: 42
type: B, layer: 1, pos: 11
type: B, layer: 1, pos: 39
type: A, layer: 1, pos: 39
type: B, layer: 1, pos: 31
type: B, layer: 1, pos: 16
type: A, layer: 1, pos: 31
type: A, layer: 1, pos: 16
type: A, layer: 1, pos: 23
type: A, layer: 1, pos: 33
type: B, layer: 1, pos: 23
type: B, layer: 1, pos: 33
type: A, layer: 1, pos: 47
type: A, layer: 1, pos: 30

Time for candidate selection: 0.10 seconds

### Candidate
type: A, layer: 1, pos: 11

### Candidate
type: B, layer: 1, pos: 34

### Candidate
type: A, layer: 1, pos: 22

## Relational analysis of NS_A1_A2_B2_B1_B1_B1_A1

### Relational analysis result of NS_A1_A2_B2_B1_B1_B1_A1
Status: Status.UNKNOWN
Output dim: 3, lower bound: -20.6032768, upper bound: 20.6006222
time: 0.61 seconds

## Relational analysis of NS_A1_A2_B2_B1_B1_B1_A2

### Relational analysis result of NS_A1_A2_B2_B1_B1_B1_A2
Status: Status.UNKNOWN
Output dim: 3, lower bound: -20.6032768, upper bound: 20.6003978
time: 0.64 seconds

## BFS NS instance: NS_A1_A2_B2_B1_B1_B2

### Backsubstitution after applying NS history:
0: -3.5923164, 3.1570973, -4.1259956, 3.6959655, -7.2882819, 7.2830920
1: -14.2131710, 12.2455807, -16.2677040, 14.2814875, -28.4946594, 28.5132828
2: -7.1067591, 11.4467535, -8.1034117, 13.5844889, -20.6912479, 19.5501614
3: -12.4342022, 11.1442776, -14.1629162, 13.1021299, -25.5363312, 25.3071918
4: -9.1061058, 11.6097670, -10.3536272, 13.7522688, -22.8583755, 21.9633942

Time for backsubstitution: 1.27 seconds

### NS candidates at layer 1
type: A, layer: 1, pos: 11
type: A, layer: 1, pos: 34
type: A, layer: 1, pos: 47
type: B, layer: 1, pos: 34
type: A, layer: 1, pos: 22
type: B, layer: 1, pos: 9
type: B, layer: 1, pos: 12
type: B, layer: 1, pos: 4
type: A, layer: 1, pos: 4
type: A, layer: 1, pos: 12
type: A, layer: 1, pos: 42
type: B, layer: 1, pos: 42
type: B, layer: 1, pos: 39
type: B, layer: 1, pos: 31
type: A, layer: 1, pos: 33
type: A, layer: 1, pos: 31
type: A, layer: 1, pos: 16
type: B, layer: 1, pos: 16
type: B, layer: 1, pos: 23
type: A, layer: 1, pos: 23
type: A, layer: 1, pos: 39
type: B, layer: 1, pos: 33
type: A, layer: 1, pos: 30
type: B, layer: 1, pos: 11

Time for candidate selection: 0.10 seconds

### Candidate
type: A, layer: 1, pos: 11

### Candidate
type: A, layer: 1, pos: 34

### Candidate
type: A, layer: 1, pos: 47

## Relational analysis of NS_A1_A2_B2_B1_B1_B2_A1

### Relational analysis result of NS_A1_A2_B2_B1_B1_B2_A1
Status: Status.VERIFIED
Output dim: 3, lower bound: -20.6007518, upper bound: 20.6003526
time: 0.66 seconds

## Relational analysis of NS_A1_A2_B2_B1_B1_B2_A2

### Relational analysis result of NS_A1_A2_B2_B1_B1_B2_A2
Status: Status.VERIFIED
Output dim: 3, lower bound: -20.6007518, upper bound: 20.6006222
time: 0.77 seconds

## BFS NS instance: NS_A1_A2_B2_B1_B2_B1

### Backsubstitution after applying NS history:
0: -3.5923164, 3.1570973, -3.2224360, 2.8752885, -6.4676046, 6.3795314
1: -14.2131710, 12.2455807, -12.6998873, 11.1310387, -25.3442097, 24.9454651
2: -7.1067591, 11.4467535, -6.3695655, 10.4611368, -17.5678959, 17.8163185
3: -12.4342022, 11.1442776, -11.1067505, 10.2140846, -22.6482868, 22.2510262
4: -9.1061058, 11.6097670, -8.1406956, 10.6549606, -19.7610664, 19.7504597

Time for backsubstitution: 1.23 seconds

### NS candidates at layer 1
type: A, layer: 1, pos: 11
type: B, layer: 1, pos: 11
type: B, layer: 1, pos: 34
type: A, layer: 1, pos: 42
type: B, layer: 1, pos: 42
type: B, layer: 1, pos: 12
type: A, layer: 1, pos: 22
type: A, layer: 1, pos: 34
type: A, layer: 1, pos: 12
type: B, layer: 1, pos: 9
type: B, layer: 1, pos: 24
type: B, layer: 1, pos: 31
type: A, layer: 1, pos: 31
type: A, layer: 1, pos: 39
type: B, layer: 1, pos: 33
type: B, layer: 1, pos: 39
type: B, layer: 1, pos: 16
type: A, layer: 1, pos: 16
type: A, layer: 1, pos: 23
type: A, layer: 1, pos: 4
type: B, layer: 1, pos: 23
type: A, layer: 1, pos: 33
type: A, layer: 1, pos: 47
type: A, layer: 1, pos: 30

Time for candidate selection: 0.10 seconds

### Candidate
type: A, layer: 1, pos: 11

### Candidate
type: B, layer: 1, pos: 11

### Candidate
type: B, layer: 1, pos: 34

### Candidate
type: A, layer: 1, pos: 42

## Relational analysis of NS_A1_A2_B2_B1_B2_B1_A1

### Relational analysis result of NS_A1_A2_B2_B1_B2_B1_A1
Status: Status.VERIFIED
Output dim: 3, lower bound: -20.6015155, upper bound: 20.6025437
time: 0.99 seconds

## Relational analysis of NS_A1_A2_B2_B1_B2_B1_A2

### Relational analysis result of NS_A1_A2_B2_B1_B2_B1_A2
Status: Status.UNKNOWN
Output dim: 3, lower bound: -20.6039446, upper bound: 20.6026866
time: 1.05 seconds

## BFS NS instance: NS_A1_A2_B2_B1_B2_B2

### Backsubstitution after applying NS history:
0: -3.5923164, 3.1570973, -3.6991673, 3.3018584, -6.8941746, 6.8562646
1: -14.2131710, 12.2455807, -14.5843849, 12.7891378, -27.0023079, 26.8299580
2: -7.1067591, 11.4467535, -7.2860594, 12.0324087, -19.1391678, 18.7328110
3: -12.4342022, 11.1442776, -12.7376213, 11.7208414, -24.1550446, 23.8818989
4: -9.1061058, 11.6097670, -9.3409872, 12.2283535, -21.3344593, 20.9507504

Time for backsubstitution: 0.94 seconds

### NS candidates at layer 1
type: A, layer: 1, pos: 11
type: B, layer: 1, pos: 34
type: B, layer: 1, pos: 11
type: B, layer: 1, pos: 12
type: A, layer: 1, pos: 42
type: B, layer: 1, pos: 42
type: B, layer: 1, pos: 9
type: A, layer: 1, pos: 22
type: A, layer: 1, pos: 12
type: B, layer: 1, pos: 24
type: A, layer: 1, pos: 34
type: A, layer: 1, pos: 39
type: A, layer: 1, pos: 31
type: B, layer: 1, pos: 16
type: B, layer: 1, pos: 39
type: A, layer: 1, pos: 23
type: B, layer: 1, pos: 31
type: A, layer: 1, pos: 4
type: A, layer: 1, pos: 16
type: B, layer: 1, pos: 33
type: B, layer: 1, pos: 23
type: A, layer: 1, pos: 33
type: A, layer: 1, pos: 47
type: A, layer: 1, pos: 30

Time for candidate selection: 0.08 seconds

### Candidate
type: A, layer: 1, pos: 11

### Candidate
type: B, layer: 1, pos: 34

### Candidate
type: B, layer: 1, pos: 11

### Candidate
type: B, layer: 1, pos: 12

## Relational analysis of NS_A1_A2_B2_B1_B2_B2_B1
Optimization infeasible because this subproblem isn't reachable.

### Candidate
type: A, layer: 1, pos: 42

## Relational analysis of NS_A1_A2_B2_B1_B2_B2_A1

### Relational analysis result of NS_A1_A2_B2_B1_B2_B2_A1
Status: Status.VERIFIED
Output dim: 3, lower bound: -20.6012900, upper bound: 20.6001075
time: 0.69 seconds

## Relational analysis of NS_A1_A2_B2_B1_B2_B2_A2

### Relational analysis result of NS_A1_A2_B2_B1_B2_B2_A2
Status: Status.UNKNOWN
Output dim: 3, lower bound: -20.6037190, upper bound: 20.6002504
time: 0.71 seconds

## BFS NS instance: NS_A1_A2_B2_B2_A2_B1

### Backsubstitution after applying NS history:
0: -3.4072533, 2.9987469, -3.8540320, 3.4321170, -6.8393703, 6.8527789
1: -13.4742098, 11.6417627, -15.2264605, 13.2877932, -26.7619991, 26.8682232
2: -6.7381449, 10.8741417, -7.5972466, 12.4870577, -19.2252026, 18.4713860
3: -11.7894468, 10.6054029, -13.2969398, 12.1337633, -23.9232101, 23.9023399
4: -8.6332855, 11.0345154, -9.7240124, 12.6810350, -21.3143196, 20.7585258

Time for backsubstitution: 1.19 seconds

### NS candidates at layer 1
type: A, layer: 1, pos: 11
type: B, layer: 1, pos: 34
type: A, layer: 1, pos: 34
type: B, layer: 1, pos: 12
type: B, layer: 1, pos: 11
type: A, layer: 1, pos: 22
type: A, layer: 1, pos: 47
type: B, layer: 1, pos: 9
type: A, layer: 1, pos: 12
type: B, layer: 1, pos: 42
type: B, layer: 1, pos: 4
type: B, layer: 1, pos: 31
type: A, layer: 1, pos: 31
type: A, layer: 1, pos: 4
type: B, layer: 1, pos: 39
type: A, layer: 1, pos: 16
type: B, layer: 1, pos: 16
type: A, layer: 1, pos: 39
type: A, layer: 1, pos: 33
type: A, layer: 1, pos: 30
type: B, layer: 1, pos: 23
type: B, layer: 1, pos: 33
type: A, layer: 1, pos: 23
type: B, layer: 1, pos: 47

Time for candidate selection: 0.10 seconds

### Candidate
type: A, layer: 1, pos: 11

### Candidate
type: B, layer: 1, pos: 34

### Candidate
type: A, layer: 1, pos: 34

### Candidate
type: B, layer: 1, pos: 12

## Relational analysis of NS_A1_A2_B2_B2_A2_B1_B1
Optimization infeasible because this subproblem isn't reachable.

### Candidate
type: B, layer: 1, pos: 11

### Candidate
type: A, layer: 1, pos: 22

## Relational analysis of NS_A1_A2_B2_B2_A2_B1_A1

### Relational analysis result of NS_A1_A2_B2_B2_A2_B1_A1
Status: Status.UNKNOWN
Output dim: 3, lower bound: -20.6033932, upper bound: 20.6022874
time: 0.79 seconds

## Relational analysis of NS_A1_A2_B2_B2_A2_B1_A2

### Relational analysis result of NS_A1_A2_B2_B2_A2_B1_A2
Status: Status.UNKNOWN
Output dim: 3, lower bound: -20.6033932, upper bound: 20.6020680
time: 0.92 seconds

## BFS NS instance: NS_A1_A2_B2_B2_A2_B2

### Backsubstitution after applying NS history:
0: -3.4072533, 2.9987469, -4.4449782, 3.9616370, -7.3688903, 7.4437242
1: -13.4742098, 11.6417627, -17.5713806, 15.3260794, -28.8002872, 29.2131424
2: -6.7381449, 10.8741417, -8.7266226, 14.5362911, -21.2744370, 19.6007652
3: -11.7894468, 10.6054029, -15.3107166, 14.0289297, -25.8183765, 25.9161129
4: -8.6332855, 11.0345154, -11.1794167, 14.6911469, -23.3244324, 22.2139320

Time for backsubstitution: 1.22 seconds

### NS candidates at layer 1
type: A, layer: 1, pos: 11
type: A, layer: 1, pos: 47
type: A, layer: 1, pos: 34
type: B, layer: 1, pos: 34
type: B, layer: 1, pos: 9
type: B, layer: 1, pos: 12
type: A, layer: 1, pos: 22
type: A, layer: 1, pos: 30
type: B, layer: 1, pos: 31
type: A, layer: 1, pos: 12
type: A, layer: 1, pos: 31
type: B, layer: 1, pos: 11
type: B, layer: 1, pos: 4
type: A, layer: 1, pos: 16
type: B, layer: 1, pos: 42
type: B, layer: 1, pos: 16
type: A, layer: 1, pos: 4
type: B, layer: 1, pos: 39
type: A, layer: 1, pos: 33
type: B, layer: 1, pos: 23
type: A, layer: 1, pos: 23
type: B, layer: 1, pos: 33
type: A, layer: 1, pos: 39
type: B, layer: 1, pos: 47

Time for candidate selection: 0.08 seconds

### Candidate
type: A, layer: 1, pos: 11

### Candidate
type: A, layer: 1, pos: 47

## Relational analysis of NS_A1_A2_B2_B2_A2_B2_A1

### Relational analysis result of NS_A1_A2_B2_B2_A2_B2_A1
Status: Status.VERIFIED
Output dim: 3, lower bound: -20.6008533, upper bound: 20.6020003
time: 0.66 seconds

## Relational analysis of NS_A1_A2_B2_B2_A2_B2_A2

### Relational analysis result of NS_A1_A2_B2_B2_A2_B2_A2
Status: Status.UNKNOWN
Output dim: 3, lower bound: -20.6039487, upper bound: 20.6022699
time: 0.77 seconds

## Summary of splitting at layer (split count: 6)
- Time for NS candidates: 2.73 seconds
NS_A1_A2_B1_B1_B1_B1_A1, status: Status.UNKNOWN, split count: 7, time: 2.73
Output dim: 3, lower bound: -20.6032768, upper bound: 20.5970383
NS_A1_A2_B1_B1_B1_B1_A2, status: Status.UNKNOWN, split count: 7, time: 2.73
Output dim: 3, lower bound: -20.6032768, upper bound: 20.5970383
NS_A1_A2_B1_B1_B1_B2_A1, status: Status.UNKNOWN, split count: 7, time: 2.73
Output dim: 3, lower bound: -20.6032768, upper bound: 20.5970383
NS_A1_A2_B1_B1_B1_B2_A2, status: Status.UNKNOWN, split count: 7, time: 2.73
Output dim: 3, lower bound: -20.6032768, upper bound: 20.5970383
NS_A1_A2_B1_B1_B2_B1_A1, status: Status.VERIFIED, split count: 7, time: 2.73
Output dim: 3, lower bound: -20.6011146, upper bound: 20.5993392
NS_A1_A2_B1_B1_B2_B1_A2, status: Status.UNKNOWN, split count: 7, time: 2.73
Output dim: 3, lower bound: -20.6035437, upper bound: 20.5994820
NS_A1_A2_B1_B1_B2_B2_A1, status: Status.UNKNOWN, split count: 7, time: 2.73
Output dim: 3, lower bound: -20.6035806, upper bound: 20.5993457
NS_A1_A2_B1_B1_B2_B2_A2, status: Status.UNKNOWN, split count: 7, time: 2.73
Output dim: 3, lower bound: -20.6035806, upper bound: 20.5993457
NS_A1_A2_B1_B2_A2_B1_A1, status: Status.UNKNOWN, split count: 7, time: 2.73
Output dim: 3, lower bound: -20.6033932, upper bound: 20.5985397
NS_A1_A2_B1_B2_A2_B1_A2, status: Status.UNKNOWN, split count: 7, time: 2.73
Output dim: 3, lower bound: -20.6033932, upper bound: 20.5985397
NS_A1_A2_B1_B2_A2_B2_A1, status: Status.VERIFIED, split count: 7, time: 2.73
Output dim: 3, lower bound: -20.6002979, upper bound: 20.5982526
NS_A1_A2_B1_B2_A2_B2_A2, status: Status.UNKNOWN, split count: 7, time: 2.73
Output dim: 3, lower bound: -20.6033932, upper bound: 20.5985222
NS_A1_A2_B2_B1_B1_B1_A1, status: Status.UNKNOWN, split count: 7, time: 2.73
Output dim: 3, lower bound: -20.6032768, upper bound: 20.6006222
NS_A1_A2_B2_B1_B1_B1_A2, status: Status.UNKNOWN, split count: 7, time: 2.73
Output dim: 3, lower bound: -20.6032768, upper bound: 20.6003978
NS_A1_A2_B2_B1_B1_B2_A1, status: Status.VERIFIED, split count: 7, time: 2.73
Output dim: 3, lower bound: -20.6007518, upper bound: 20.6003526
NS_A1_A2_B2_B1_B1_B2_A2, status: Status.VERIFIED, split count: 7, time: 2.73
Output dim: 3, lower bound: -20.6007518, upper bound: 20.6006222
NS_A1_A2_B2_B1_B2_B1_A1, status: Status.VERIFIED, split count: 7, time: 2.73
Output dim: 3, lower bound: -20.6015155, upper bound: 20.6025437
NS_A1_A2_B2_B1_B2_B1_A2, status: Status.UNKNOWN, split count: 7, time: 2.73
Output dim: 3, lower bound: -20.6039446, upper bound: 20.6026866
NS_A1_A2_B2_B1_B2_B2_A1, status: Status.VERIFIED, split count: 7, time: 2.73
Output dim: 3, lower bound: -20.6012900, upper bound: 20.6001075
NS_A1_A2_B2_B1_B2_B2_A2, status: Status.UNKNOWN, split count: 7, time: 2.73
Output dim: 3, lower bound: -20.6037190, upper bound: 20.6002504
NS_A1_A2_B2_B2_A2_B1_A1, status: Status.UNKNOWN, split count: 7, time: 2.73
Output dim: 3, lower bound: -20.6033932, upper bound: 20.6022874
NS_A1_A2_B2_B2_A2_B1_A2, status: Status.UNKNOWN, split count: 7, time: 2.73
Output dim: 3, lower bound: -20.6033932, upper bound: 20.6020680
NS_A1_A2_B2_B2_A2_B2_A1, status: Status.VERIFIED, split count: 7, time: 2.73
Output dim: 3, lower bound: -20.6008533, upper bound: 20.6020003
NS_A1_A2_B2_B2_A2_B2_A2, status: Status.UNKNOWN, split count: 7, time: 2.73
Output dim: 3, lower bound: -20.6039487, upper bound: 20.6022699

## BFS NS instance: NS_A1_A2_B1_B1_B1_B1_A1

### Backsubstitution after applying NS history:
0: -3.1597602, 2.8049958, -2.8369715, 2.5377181, -5.6974773, 5.6419668
1: -12.4845533, 10.8861303, -11.1648302, 9.8173466, -22.3018990, 22.0509605
2: -6.2473922, 10.1505280, -5.6087904, 9.2279978, -15.4753895, 15.7593184
3: -10.9279404, 9.9419422, -9.7681313, 9.0039520, -19.9318924, 19.7100716
4: -7.9961886, 10.3389921, -7.1507564, 9.4026794, -17.3988667, 17.4897480

Time for backsubstitution: 1.37 seconds

### NS candidates at layer 1
type: A, layer: 1, pos: 11
type: B, layer: 1, pos: 11
type: B, layer: 1, pos: 34
type: A, layer: 1, pos: 34
type: A, layer: 1, pos: 12
type: B, layer: 1, pos: 12
type: B, layer: 1, pos: 4
type: A, layer: 1, pos: 4
type: B, layer: 1, pos: 9
type: B, layer: 1, pos: 42
type: A, layer: 1, pos: 42
type: A, layer: 1, pos: 31
type: B, layer: 1, pos: 31
type: B, layer: 1, pos: 16
type: A, layer: 1, pos: 39
type: A, layer: 1, pos: 16
type: B, layer: 1, pos: 33
type: A, layer: 1, pos: 33
type: A, layer: 1, pos: 23
type: B, layer: 1, pos: 39
type: B, layer: 1, pos: 23
type: A, layer: 1, pos: 47
type: A, layer: 1, pos: 30

Time for candidate selection: 0.10 seconds

### Candidate
type: A, layer: 1, pos: 11

### Candidate
type: B, layer: 1, pos: 11

### Candidate
type: B, layer: 1, pos: 34

### Candidate
type: A, layer: 1, pos: 34

### Candidate
type: A, layer: 1, pos: 12

## Relational analysis of NS_A1_A2_B1_B1_B1_B1_A1_A1

### Relational analysis result of NS_A1_A2_B1_B1_B1_B1_A1_A1
Status: Status.VERIFIED
Output dim: 3, lower bound: -20.5973723, upper bound: 20.5968767
time: 0.56 seconds

## Relational analysis of NS_A1_A2_B1_B1_B1_B1_A1_A2

### Relational analysis result of NS_A1_A2_B1_B1_B1_B1_A1_A2
Status: Status.UNKNOWN
Output dim: 3, lower bound: -20.6033073, upper bound: 20.5972723
time: 0.71 seconds

## BFS NS instance: NS_A1_A2_B1_B1_B1_B1_A2

### Backsubstitution after applying NS history:
0: -3.8558705, 3.4283504, -2.8369715, 2.5377181, -6.3935885, 6.2653217
1: -15.2394800, 13.2685280, -11.1648302, 9.8173466, -25.0568237, 24.4333553
2: -7.5982504, 12.4685383, -5.6087904, 9.2279978, -16.8262482, 18.0773277
3: -13.3093185, 12.1124258, -9.7681313, 9.0039520, -22.3132706, 21.8805542
4: -9.7378912, 12.6584482, -7.1507564, 9.4026794, -19.1405659, 19.8092041

Time for backsubstitution: 1.28 seconds

### NS candidates at layer 1
type: B, layer: 1, pos: 11
type: A, layer: 1, pos: 11
type: B, layer: 1, pos: 34
type: A, layer: 1, pos: 34
type: A, layer: 1, pos: 12
type: B, layer: 1, pos: 12
type: B, layer: 1, pos: 4
type: A, layer: 1, pos: 4
type: B, layer: 1, pos: 9
type: B, layer: 1, pos: 42
type: A, layer: 1, pos: 42
type: B, layer: 1, pos: 31
type: A, layer: 1, pos: 31
type: A, layer: 1, pos: 39
type: B, layer: 1, pos: 16
type: B, layer: 1, pos: 33
type: A, layer: 1, pos: 16
type: A, layer: 1, pos: 23
type: A, layer: 1, pos: 33
type: B, layer: 1, pos: 39
type: B, layer: 1, pos: 23
type: A, layer: 1, pos: 30
type: A, layer: 1, pos: 47

Time for candidate selection: 0.12 seconds

### Candidate
type: B, layer: 1, pos: 11

### Candidate
type: A, layer: 1, pos: 11

### Candidate
type: B, layer: 1, pos: 34

### Candidate
type: A, layer: 1, pos: 34

### Candidate
type: A, layer: 1, pos: 12

## Relational analysis of NS_A1_A2_B1_B1_B1_B1_A2_A1

### Relational analysis result of NS_A1_A2_B1_B1_B1_B1_A2_A1
Status: Status.VERIFIED
Output dim: 3, lower bound: -20.5973723, upper bound: 20.5968767
time: 0.57 seconds

## Relational analysis of NS_A1_A2_B1_B1_B1_B1_A2_A2

### Relational analysis result of NS_A1_A2_B1_B1_B1_B1_A2_A2
Status: Status.UNKNOWN
Output dim: 3, lower bound: -20.6033073, upper bound: 20.5972723
time: 0.78 seconds

## BFS NS instance: NS_A1_A2_B1_B1_B1_B2_A1

### Backsubstitution after applying NS history:
0: -3.1597602, 2.8049958, -3.4805706, 3.1200867, -6.2798471, 6.2855654
1: -12.4845533, 10.8861303, -13.7218876, 12.0228958, -24.5074482, 24.6080170
2: -6.2473922, 10.1505280, -6.8483577, 11.4872770, -17.7346687, 16.9988861
3: -10.9279404, 9.9419422, -11.9613485, 11.0371389, -21.9650784, 21.9032898
4: -7.9961886, 10.3389921, -8.7219229, 11.6184912, -19.6146793, 19.0609150

Time for backsubstitution: 1.39 seconds

### NS candidates at layer 1
type: A, layer: 1, pos: 11
type: A, layer: 1, pos: 34
type: B, layer: 1, pos: 11
type: B, layer: 1, pos: 34
type: A, layer: 1, pos: 47
type: B, layer: 1, pos: 9
type: B, layer: 1, pos: 12
type: B, layer: 1, pos: 4
type: A, layer: 1, pos: 4
type: A, layer: 1, pos: 12
type: B, layer: 1, pos: 42
type: A, layer: 1, pos: 42
type: B, layer: 1, pos: 31
type: A, layer: 1, pos: 31
type: B, layer: 1, pos: 39
type: A, layer: 1, pos: 33
type: B, layer: 1, pos: 16
type: A, layer: 1, pos: 16
type: B, layer: 1, pos: 23
type: B, layer: 1, pos: 33
type: A, layer: 1, pos: 39
type: A, layer: 1, pos: 23
type: A, layer: 1, pos: 30

Time for candidate selection: 0.11 seconds

### Candidate
type: A, layer: 1, pos: 11

### Candidate
type: A, layer: 1, pos: 34

### Candidate
type: B, layer: 1, pos: 11

### Candidate
type: B, layer: 1, pos: 34

### Candidate
type: A, layer: 1, pos: 47

## Relational analysis of NS_A1_A2_B1_B1_B1_B2_A1_A1

### Relational analysis result of NS_A1_A2_B1_B1_B1_B2_A1_A1
Status: Status.VERIFIED
Output dim: 3, lower bound: -20.5958956, upper bound: 20.5961608
time: 0.64 seconds

## Relational analysis of NS_A1_A2_B1_B1_B1_B2_A1_A2

### Relational analysis result of NS_A1_A2_B1_B1_B1_B2_A1_A2
Status: Status.VERIFIED
Output dim: 3, lower bound: -20.5958956, upper bound: 20.5970383
time: 0.71 seconds

## BFS NS instance: NS_A1_A2_B1_B1_B1_B2_A2

### Backsubstitution after applying NS history:
0: -3.8558705, 3.4283504, -3.4805706, 3.1200867, -6.9759569, 6.9089208
1: -15.2394800, 13.2685280, -13.7218876, 12.0228958, -27.2623730, 26.9904137
2: -7.5982504, 12.4685383, -6.8483577, 11.4872770, -19.0855274, 19.3168964
3: -13.3093185, 12.1124258, -11.9613485, 11.0371389, -24.3464565, 24.0737724
4: -9.7378912, 12.6584482, -8.7219229, 11.6184912, -21.3563805, 21.3803711

Time for backsubstitution: 1.38 seconds

### NS candidates at layer 1
type: A, layer: 1, pos: 11
type: B, layer: 1, pos: 11
type: A, layer: 1, pos: 34
type: B, layer: 1, pos: 34
type: B, layer: 1, pos: 9
type: B, layer: 1, pos: 12
type: A, layer: 1, pos: 12
type: B, layer: 1, pos: 4
type: A, layer: 1, pos: 4
type: A, layer: 1, pos: 47
type: B, layer: 1, pos: 42
type: A, layer: 1, pos: 42
type: B, layer: 1, pos: 31
type: A, layer: 1, pos: 31
type: B, layer: 1, pos: 16
type: A, layer: 1, pos: 16
type: B, layer: 1, pos: 33
type: B, layer: 1, pos: 39
type: A, layer: 1, pos: 39
type: A, layer: 1, pos: 33
type: B, layer: 1, pos: 23
type: A, layer: 1, pos: 23
type: A, layer: 1, pos: 30

Time for candidate selection: 0.10 seconds

### Candidate
type: A, layer: 1, pos: 11

### Candidate
type: B, layer: 1, pos: 11

### Candidate
type: A, layer: 1, pos: 34

### Candidate
type: B, layer: 1, pos: 34

### Candidate
type: B, layer: 1, pos: 9

### Candidate
type: B, layer: 1, pos: 12

## Relational analysis of NS_A1_A2_B1_B1_B1_B2_A2_B1
Optimization infeasible because this subproblem isn't reachable.

### Candidate
type: A, layer: 1, pos: 12

### Candidate
type: B, layer: 1, pos: 4

## Relational analysis of NS_A1_A2_B1_B1_B1_B2_A2_B1

### Relational analysis result of NS_A1_A2_B1_B1_B1_B2_A2_B1
Status: Status.UNKNOWN
Output dim: 3, lower bound: -20.6032218, upper bound: 20.5966779
time: 0.68 seconds

## Relational analysis of NS_A1_A2_B1_B1_B1_B2_A2_B2
Optimization infeasible because this subproblem isn't reachable.

### Candidate
type: A, layer: 1, pos: 4

### Candidate
type: A, layer: 1, pos: 47

## Relational analysis of NS_A1_A2_B1_B1_B1_B2_A2_A1

### Relational analysis result of NS_A1_A2_B1_B1_B1_B2_A2_A1
Status: Status.VERIFIED
Output dim: 3, lower bound: -20.5958956, upper bound: 20.5967687
time: 0.78 seconds

## Relational analysis of NS_A1_A2_B1_B1_B1_B2_A2_A2

### Relational analysis result of NS_A1_A2_B1_B1_B1_B2_A2_A2
Status: Status.VERIFIED
Output dim: 3, lower bound: -20.5958956, upper bound: 20.5970383
time: 0.94 seconds

## BFS NS instance: NS_A1_A2_B1_B1_B2_B1_A2

### Backsubstitution after applying NS history:
0: -3.4072533, 2.9987469, -2.6776226, 2.3901095, -5.7973628, 5.6763687
1: -13.4742098, 11.6417627, -10.5423784, 9.2746677, -22.7488766, 22.1841412
2: -6.7381449, 10.8741417, -5.3035603, 8.6706009, -15.4087458, 16.1777020
3: -11.7894468, 10.6054029, -9.2467489, 8.5087490, -20.2981949, 19.8521519
4: -8.6332855, 11.0345154, -6.7716413, 8.8663626, -17.4996490, 17.8061543

Time for backsubstitution: 1.25 seconds

### NS candidates at layer 1
type: B, layer: 1, pos: 11
type: A, layer: 1, pos: 11
type: B, layer: 1, pos: 34
type: A, layer: 1, pos: 34
type: B, layer: 1, pos: 12
type: A, layer: 1, pos: 22
type: A, layer: 1, pos: 12
type: B, layer: 1, pos: 9
type: A, layer: 1, pos: 39
type: B, layer: 1, pos: 33
type: B, layer: 1, pos: 31
type: B, layer: 1, pos: 24
type: A, layer: 1, pos: 31
type: B, layer: 1, pos: 42
type: B, layer: 1, pos: 16
type: A, layer: 1, pos: 23
type: A, layer: 1, pos: 16
type: B, layer: 1, pos: 39
type: A, layer: 1, pos: 4
type: B, layer: 1, pos: 23
type: A, layer: 1, pos: 33
type: A, layer: 1, pos: 30
type: A, layer: 1, pos: 47

Time for candidate selection: 0.10 seconds

### Candidate
type: B, layer: 1, pos: 11

### Candidate
type: A, layer: 1, pos: 11

### Candidate
type: B, layer: 1, pos: 34

### Candidate
type: A, layer: 1, pos: 34

### Candidate
type: B, layer: 1, pos: 12

## Relational analysis of NS_A1_A2_B1_B1_B2_B1_A2_B1
Optimization infeasible because this subproblem isn't reachable.

### Candidate
type: A, layer: 1, pos: 22

## Relational analysis of NS_A1_A2_B1_B1_B2_B1_A2_A1

### Relational analysis result of NS_A1_A2_B1_B1_B2_B1_A2_A1
Status: Status.UNKNOWN
Output dim: 3, lower bound: -20.6035437, upper bound: 20.5994820
time: 0.66 seconds

## Relational analysis of NS_A1_A2_B1_B1_B2_B1_A2_A2

### Relational analysis result of NS_A1_A2_B1_B1_B2_B1_A2_A2
Status: Status.UNKNOWN
Output dim: 3, lower bound: -20.6035437, upper bound: 20.5994820
time: 0.68 seconds

## BFS NS instance: NS_A1_A2_B1_B1_B2_B2_A1

### Backsubstitution after applying NS history:
0: -3.1597602, 2.8049958, -3.0070853, 2.6840725, -5.8438320, 5.8120809
1: -12.4845533, 10.8861303, -11.8437071, 10.4111977, -22.8957520, 22.7298374
2: -6.2473922, 10.1505280, -5.9487362, 9.7418251, -15.9892159, 16.0992641
3: -10.9279404, 9.9419422, -10.3784885, 9.5441866, -20.4721222, 20.3204288
4: -7.9961886, 10.3389921, -7.6077647, 9.9460669, -17.9422550, 17.9467564

Time for backsubstitution: 1.26 seconds

### NS candidates at layer 1
type: B, layer: 1, pos: 11
type: A, layer: 1, pos: 11
type: B, layer: 1, pos: 34
type: A, layer: 1, pos: 34
type: A, layer: 1, pos: 12
type: B, layer: 1, pos: 12
type: B, layer: 1, pos: 42
type: A, layer: 1, pos: 42
type: B, layer: 1, pos: 9
type: A, layer: 1, pos: 39
type: B, layer: 1, pos: 33
type: A, layer: 1, pos: 31
type: B, layer: 1, pos: 16
type: A, layer: 1, pos: 4
type: B, layer: 1, pos: 31
type: A, layer: 1, pos: 23
type: B, layer: 1, pos: 24
type: A, layer: 1, pos: 16
type: B, layer: 1, pos: 23
type: B, layer: 1, pos: 39
type: A, layer: 1, pos: 33
type: A, layer: 1, pos: 47
type: A, layer: 1, pos: 30

Time for candidate selection: 0.09 seconds

### Candidate
type: B, layer: 1, pos: 11

### Candidate
type: A, layer: 1, pos: 11

### Candidate
type: B, layer: 1, pos: 34

### Candidate
type: A, layer: 1, pos: 34

### Candidate
type: A, layer: 1, pos: 12

### Candidate
type: B, layer: 1, pos: 12

### Candidate
type: B, layer: 1, pos: 42

## Relational analysis of NS_A1_A2_B1_B1_B2_B2_A1_B1

### Relational analysis result of NS_A1_A2_B1_B1_B2_B2_A1_B1
Status: Status.UNKNOWN
Output dim: 3, lower bound: -20.6034656, upper bound: 20.5987112
time: 0.74 seconds

## Relational analysis of NS_A1_A2_B1_B1_B2_B2_A1_B2

### Relational analysis result of NS_A1_A2_B1_B1_B2_B2_A1_B2
Status: Status.UNKNOWN
Output dim: 3, lower bound: -20.6035433, upper bound: 20.5993457
time: 0.90 seconds

## BFS NS instance: NS_A1_A2_B1_B1_B2_B2_A2

### Backsubstitution after applying NS history:
0: -3.8558705, 3.4283504, -3.0070853, 2.6840725, -6.5399427, 6.4354358
1: -15.2394800, 13.2685280, -11.8437071, 10.4111977, -25.6506767, 25.1122360
2: -7.5982504, 12.4685383, -5.9487362, 9.7418251, -17.3400764, 18.4172745
3: -13.3093185, 12.1124258, -10.3784885, 9.5441866, -22.8535023, 22.4909115
4: -9.7378912, 12.6584482, -7.6077647, 9.9460669, -19.6839561, 20.2662125

Time for backsubstitution: 1.17 seconds

### NS candidates at layer 1
type: B, layer: 1, pos: 11
type: B, layer: 1, pos: 34
type: A, layer: 1, pos: 34
type: A, layer: 1, pos: 12
type: B, layer: 1, pos: 12
type: A, layer: 1, pos: 42
type: B, layer: 1, pos: 42
type: B, layer: 1, pos: 9
type: A, layer: 1, pos: 11
type: A, layer: 1, pos: 39
type: B, layer: 1, pos: 33
type: A, layer: 1, pos: 4
type: B, layer: 1, pos: 16
type: A, layer: 1, pos: 23
type: A, layer: 1, pos: 31
type: B, layer: 1, pos: 31
type: A, layer: 1, pos: 16
type: B, layer: 1, pos: 23
type: B, layer: 1, pos: 24
type: B, layer: 1, pos: 39
type: A, layer: 1, pos: 33
type: A, layer: 1, pos: 30
type: A, layer: 1, pos: 47

Time for candidate selection: 0.09 seconds

### Candidate
type: B, layer: 1, pos: 11

### Candidate
type: B, layer: 1, pos: 34

### Candidate
type: A, layer: 1, pos: 34

### Candidate
type: A, layer: 1, pos: 12

### Candidate
type: B, layer: 1, pos: 12

### Candidate
type: A, layer: 1, pos: 42

## Relational analysis of NS_A1_A2_B1_B1_B2_B2_A2_A1

### Relational analysis result of NS_A1_A2_B1_B1_B2_B2_A2_A1
Status: Status.VERIFIED
Output dim: 3, lower bound: -20.6011143, upper bound: 20.5992028
time: 1.31 seconds

## Relational analysis of NS_A1_A2_B1_B1_B2_B2_A2_A2

### Relational analysis result of NS_A1_A2_B1_B1_B2_B2_A2_A2
Status: Status.UNKNOWN
Output dim: 3, lower bound: -20.6035433, upper bound: 20.5993457
time: 0.76 seconds

## BFS NS instance: NS_A1_A2_B1_B2_A2_B1_A1

### Backsubstitution after applying NS history:
0: -2.9960527, 2.6623325, -3.1507802, 2.8027632, -5.7988157, 5.8131123
1: -11.8343382, 10.3386192, -12.4441538, 10.8765450, -22.7108803, 22.7827721
2: -5.9199672, 9.6447563, -6.2334828, 10.1443806, -16.0643482, 15.8782387
3: -10.3606873, 9.4523716, -10.8921318, 9.9383621, -20.2990494, 20.3444996
4: -7.5751920, 9.8309212, -7.9639950, 10.3374100, -17.9126015, 17.7949162

Time for backsubstitution: 1.17 seconds

### NS candidates at layer 1
type: A, layer: 1, pos: 11
type: B, layer: 1, pos: 11
type: B, layer: 1, pos: 34
type: A, layer: 1, pos: 34
type: B, layer: 1, pos: 12
type: A, layer: 1, pos: 12
type: A, layer: 1, pos: 47
type: B, layer: 1, pos: 42
type: B, layer: 1, pos: 4
type: B, layer: 1, pos: 9
type: A, layer: 1, pos: 4
type: A, layer: 1, pos: 31
type: B, layer: 1, pos: 31
type: A, layer: 1, pos: 30
type: B, layer: 1, pos: 39
type: A, layer: 1, pos: 33
type: A, layer: 1, pos: 39
type: B, layer: 1, pos: 33
type: B, layer: 1, pos: 16
type: B, layer: 1, pos: 47
type: A, layer: 1, pos: 16
type: B, layer: 1, pos: 23
type: A, layer: 1, pos: 23

Time for candidate selection: 0.10 seconds

### Candidate
type: A, layer: 1, pos: 11

### Candidate
type: B, layer: 1, pos: 11

### Candidate
type: B, layer: 1, pos: 34

### Candidate
type: A, layer: 1, pos: 34

### Candidate
type: B, layer: 1, pos: 12

### Candidate
type: A, layer: 1, pos: 12

### Candidate
type: A, layer: 1, pos: 47

## Relational analysis of NS_A1_A2_B1_B2_A2_B1_A1_A1

### Relational analysis result of NS_A1_A2_B1_B2_A2_B1_A1_A1
Status: Status.UNKNOWN
Output dim: 3, lower bound: -20.5967996, upper bound: 20.6033041
time: 0.85 seconds

## Relational analysis of NS_A1_A2_B1_B2_A2_B1_A1_A2

### Relational analysis result of NS_A1_A2_B1_B2_A2_B1_A1_A2
Status: Status.UNKNOWN
Output dim: 3, lower bound: -20.6041808, upper bound: 20.6041816
time: 0.89 seconds

## BFS NS instance: NS_A1_A2_B1_B2_A2_B1_A2

### Backsubstitution after applying NS history:
0: -3.5902476, 3.1974847, -3.1507802, 2.8027632, -6.3930106, 6.3482647
1: -14.1819506, 12.3807278, -12.4441538, 10.8765450, -25.0584908, 24.8248787
2: -7.0806346, 11.6148405, -6.2334828, 10.1443806, -17.2250118, 17.8483238
3: -12.3865166, 11.3239698, -10.8921318, 9.9383621, -22.3248768, 22.2160988
4: -9.0666447, 11.8067245, -7.9639950, 10.3374100, -19.4040546, 19.7707195

Time for backsubstitution: 1.30 seconds

### NS candidates at layer 1
type: B, layer: 1, pos: 11
type: A, layer: 1, pos: 11
type: A, layer: 1, pos: 34
type: B, layer: 1, pos: 34
type: B, layer: 1, pos: 4
type: A, layer: 1, pos: 4
type: A, layer: 1, pos: 12
type: B, layer: 1, pos: 12
type: B, layer: 1, pos: 47
type: B, layer: 1, pos: 9
type: B, layer: 1, pos: 42
type: A, layer: 1, pos: 30
type: A, layer: 1, pos: 39
type: B, layer: 1, pos: 33
type: B, layer: 1, pos: 31
type: A, layer: 1, pos: 31
type: B, layer: 1, pos: 16
type: A, layer: 1, pos: 16
type: A, layer: 1, pos: 33
type: B, layer: 1, pos: 39
type: A, layer: 1, pos: 23
type: B, layer: 1, pos: 23
type: A, layer: 1, pos: 47

Time for candidate selection: 0.09 seconds

### Candidate
type: B, layer: 1, pos: 11

### Candidate
type: A, layer: 1, pos: 11

### Candidate
type: A, layer: 1, pos: 34

### Candidate
type: B, layer: 1, pos: 34

### Candidate
type: B, layer: 1, pos: 4

## Relational analysis of NS_A1_A2_B1_B2_A2_B1_A2_B1

### Relational analysis result of NS_A1_A2_B1_B2_A2_B1_A2_B1
Status: Status.UNKNOWN
Output dim: 3, lower bound: -20.6037517, upper bound: 20.6002704
time: 0.82 seconds

## Relational analysis of NS_A1_A2_B1_B2_A2_B1_A2_B2

### Relational analysis result of NS_A1_A2_B1_B2_A2_B1_A2_B2
Status: Status.UNKNOWN
Output dim: 3, lower bound: -20.6041379, upper bound: 20.6041557
time: 0.85 seconds

## BFS NS instance: NS_A1_A2_B1_B2_A2_B2_A2

### Backsubstitution after applying NS history:
0: -3.3629475, 2.9603455, -3.7443836, 3.3308897, -6.6938357, 6.7047286
1: -13.2969484, 11.4919004, -14.8040266, 12.8840399, -26.1809883, 26.2959270
2: -6.6500711, 10.7346029, -7.3696198, 12.1977873, -18.8478584, 18.1042233
3: -11.6361809, 10.4710007, -12.9243069, 11.7908812, -23.4270630, 23.3953075
4: -8.5214405, 10.8940716, -9.4258900, 12.3335075, -20.8549423, 20.3199615

Time for backsubstitution: 1.22 seconds

### NS candidates at layer 1
type: A, layer: 1, pos: 11
type: B, layer: 1, pos: 11
type: A, layer: 1, pos: 34
type: B, layer: 1, pos: 34
type: B, layer: 1, pos: 12
type: A, layer: 1, pos: 22
type: B, layer: 1, pos: 9
type: A, layer: 1, pos: 30
type: B, layer: 1, pos: 31
type: A, layer: 1, pos: 12
type: A, layer: 1, pos: 31
type: B, layer: 1, pos: 4
type: B, layer: 1, pos: 42
type: B, layer: 1, pos: 39
type: A, layer: 1, pos: 4
type: A, layer: 1, pos: 16
type: B, layer: 1, pos: 23
type: B, layer: 1, pos: 16
type: A, layer: 1, pos: 33
type: A, layer: 1, pos: 23
type: B, layer: 1, pos: 33
type: A, layer: 1, pos: 39
type: B, layer: 1, pos: 47

Time for candidate selection: 0.10 seconds

### Candidate
type: A, layer: 1, pos: 11

### Candidate
type: B, layer: 1, pos: 11

### Candidate
type: A, layer: 1, pos: 34

### Candidate
type: B, layer: 1, pos: 34

### Candidate
type: B, layer: 1, pos: 12

## Relational analysis of NS_A1_A2_B1_B2_A2_B2_A2_B1
Optimization infeasible because this subproblem isn't reachable.

### Candidate
type: A, layer: 1, pos: 22

## Relational analysis of NS_A1_A2_B1_B2_A2_B2_A2_A1

### Relational analysis result of NS_A1_A2_B1_B2_A2_B2_A2_A1
Status: Status.UNKNOWN
Output dim: 3, lower bound: -20.6033932, upper bound: 20.5985222
time: 0.69 seconds

## Relational analysis of NS_A1_A2_B1_B2_A2_B2_A2_A2

### Relational analysis result of NS_A1_A2_B1_B2_A2_B2_A2_A2
Status: Status.UNKNOWN
Output dim: 3, lower bound: -20.6033932, upper bound: 20.5985222
time: 0.85 seconds

## BFS NS instance: NS_A1_A2_B2_B1_B1_B1_A1

### Backsubstitution after applying NS history:
0: -3.1597602, 2.8049958, -3.4812961, 3.1150770, -6.2748375, 6.2862916
1: -12.4845533, 10.8861303, -13.7059050, 12.0545502, -24.5391045, 24.5920353
2: -6.2473922, 10.1505280, -6.8541398, 11.3458900, -17.5932827, 17.0046673
3: -10.9279404, 9.9419422, -11.9621525, 11.0443764, -21.9723167, 21.9040890
4: -7.9961886, 10.3389921, -8.7754803, 11.5260696, -19.5222569, 19.1144676

Time for backsubstitution: 1.10 seconds

### NS candidates at layer 1
type: A, layer: 1, pos: 11
type: B, layer: 1, pos: 34
type: A, layer: 1, pos: 34
type: B, layer: 1, pos: 9
type: B, layer: 1, pos: 4
type: B, layer: 1, pos: 12
type: A, layer: 1, pos: 4
type: A, layer: 1, pos: 12
type: A, layer: 1, pos: 47
type: B, layer: 1, pos: 11
type: B, layer: 1, pos: 42
type: A, layer: 1, pos: 42
type: B, layer: 1, pos: 39
type: A, layer: 1, pos: 39
type: B, layer: 1, pos: 16
type: B, layer: 1, pos: 31
type: A, layer: 1, pos: 31
type: A, layer: 1, pos: 33
type: A, layer: 1, pos: 16
type: B, layer: 1, pos: 23
type: A, layer: 1, pos: 23
type: B, layer: 1, pos: 33
type: A, layer: 1, pos: 30

Time for candidate selection: 0.08 seconds

### Candidate
type: A, layer: 1, pos: 11

### Candidate
type: B, layer: 1, pos: 34

### Candidate
type: A, layer: 1, pos: 34

### Candidate
type: B, layer: 1, pos: 9

## Relational analysis of NS_A1_A2_B2_B1_B1_B1_A1_B1
Optimization infeasible because this subproblem isn't reachable.

### Candidate
type: B, layer: 1, pos: 4

## Relational analysis of NS_A1_A2_B2_B1_B1_B1_A1_B1

### Relational analysis result of NS_A1_A2_B2_B1_B1_B1_A1_B1
Status: Status.UNKNOWN
Output dim: 3, lower bound: -20.6032731, upper bound: 20.6005546
time: 0.82 seconds

## Relational analysis of NS_A1_A2_B2_B1_B1_B1_A1_B2
Optimization infeasible because this subproblem isn't reachable.

### Candidate
type: B, layer: 1, pos: 12

## Relational analysis of NS_A1_A2_B2_B1_B1_B1_A1_B1
Optimization infeasible because this subproblem isn't reachable.

### Candidate
type: A, layer: 1, pos: 4

### Candidate
type: A, layer: 1, pos: 12

### Candidate
type: A, layer: 1, pos: 47

## Relational analysis of NS_A1_A2_B2_B1_B1_B1_A1_A1

### Relational analysis result of NS_A1_A2_B2_B1_B1_B1_A1_A1
Status: Status.VERIFIED
Output dim: 3, lower bound: -20.5959490, upper bound: 20.6003045
time: 0.65 seconds

## Relational analysis of NS_A1_A2_B2_B1_B1_B1_A1_A2

### Relational analysis result of NS_A1_A2_B2_B1_B1_B1_A1_A2
Status: Status.VERIFIED
Output dim: 3, lower bound: -20.5959490, upper bound: 20.6011821
time: 1.03 seconds

## BFS NS instance: NS_A1_A2_B2_B1_B1_B1_A2

### Backsubstitution after applying NS history:
0: -3.8558705, 3.4283504, -3.4812961, 3.1150770, -6.9709473, 6.9096465
1: -15.2394800, 13.2685280, -13.7059050, 12.0545502, -27.2940292, 26.9744339
2: -7.5982504, 12.4685383, -6.8541398, 11.3458900, -18.9441414, 19.3226776
3: -13.3093185, 12.1124258, -11.9621525, 11.0443764, -24.3536930, 24.0745735
4: -9.7378912, 12.6584482, -8.7754803, 11.5260696, -21.2639599, 21.4339275

Time for backsubstitution: 1.00 seconds

### NS candidates at layer 1
type: A, layer: 1, pos: 11
type: B, layer: 1, pos: 11
type: B, layer: 1, pos: 34
type: A, layer: 1, pos: 34
type: B, layer: 1, pos: 4
type: A, layer: 1, pos: 4
type: B, layer: 1, pos: 9
type: A, layer: 1, pos: 12
type: B, layer: 1, pos: 12
type: B, layer: 1, pos: 42
type: A, layer: 1, pos: 42
type: A, layer: 1, pos: 39
type: B, layer: 1, pos: 39
type: B, layer: 1, pos: 16
type: B, layer: 1, pos: 31
type: A, layer: 1, pos: 16
type: A, layer: 1, pos: 31
type: A, layer: 1, pos: 23
type: A, layer: 1, pos: 33
type: B, layer: 1, pos: 33
type: B, layer: 1, pos: 23
type: A, layer: 1, pos: 47
type: A, layer: 1, pos: 30

Time for candidate selection: 0.07 seconds

### Candidate
type: A, layer: 1, pos: 11

### Candidate
type: B, layer: 1, pos: 11

### Candidate
type: B, layer: 1, pos: 34

### Candidate
type: A, layer: 1, pos: 34

### Candidate
type: B, layer: 1, pos: 4

## Relational analysis of NS_A1_A2_B2_B1_B1_B1_A2_B1

### Relational analysis result of NS_A1_A2_B2_B1_B1_B1_A2_B1
Status: Status.UNKNOWN
Output dim: 3, lower bound: -20.6032731, upper bound: 20.6003233
time: 0.80 seconds

## Relational analysis of NS_A1_A2_B2_B1_B1_B1_A2_B2
Optimization infeasible because this subproblem isn't reachable.

### Candidate
type: A, layer: 1, pos: 4

### Candidate
type: B, layer: 1, pos: 9

## Relational analysis of NS_A1_A2_B2_B1_B1_B1_A2_B1
Optimization infeasible because this subproblem isn't reachable.

### Candidate
type: A, layer: 1, pos: 12

## Relational analysis of NS_A1_A2_B2_B1_B1_B1_A2_A1

### Relational analysis result of NS_A1_A2_B2_B1_B1_B1_A2_A1
Status: Status.VERIFIED
Output dim: 3, lower bound: -20.5973723, upper bound: 20.6006040
time: 0.92 seconds

## Relational analysis of NS_A1_A2_B2_B1_B1_B1_A2_A2

### Relational analysis result of NS_A1_A2_B2_B1_B1_B1_A2_A2
Status: Status.UNKNOWN
Output dim: 3, lower bound: -20.6033073, upper bound: 20.6010027
time: 0.64 seconds

## BFS NS instance: NS_A1_A2_B2_B1_B2_B1_A2

### Backsubstitution after applying NS history:
0: -3.4072533, 2.9987469, -3.2224360, 2.8752885, -6.2825418, 6.2211814
1: -13.4742098, 11.6417627, -12.6998873, 11.1310387, -24.6052475, 24.3416500
2: -6.7381449, 10.8741417, -6.3695655, 10.4611368, -17.1992817, 17.2437077
3: -11.7894468, 10.6054029, -11.1067505, 10.2140846, -22.0035324, 21.7121487
4: -8.6332855, 11.0345154, -8.1406956, 10.6549606, -19.2882442, 19.1752071

Time for backsubstitution: 1.15 seconds

### NS candidates at layer 1
type: A, layer: 1, pos: 11
type: B, layer: 1, pos: 11
type: B, layer: 1, pos: 34
type: B, layer: 1, pos: 12
type: A, layer: 1, pos: 34
type: A, layer: 1, pos: 22
type: B, layer: 1, pos: 9
type: A, layer: 1, pos: 12
type: B, layer: 1, pos: 24
type: B, layer: 1, pos: 31
type: A, layer: 1, pos: 31
type: B, layer: 1, pos: 42
type: A, layer: 1, pos: 39
type: B, layer: 1, pos: 33
type: B, layer: 1, pos: 39
type: B, layer: 1, pos: 16
type: A, layer: 1, pos: 16
type: A, layer: 1, pos: 23
type: A, layer: 1, pos: 4
type: B, layer: 1, pos: 23
type: A, layer: 1, pos: 33
type: A, layer: 1, pos: 47
type: A, layer: 1, pos: 30

Time for candidate selection: 0.12 seconds

### Candidate
type: A, layer: 1, pos: 11

### Candidate
type: B, layer: 1, pos: 11

### Candidate
type: B, layer: 1, pos: 34

### Candidate
type: B, layer: 1, pos: 12

## Relational analysis of NS_A1_A2_B2_B1_B2_B1_A2_B1
Optimization infeasible because this subproblem isn't reachable.

### Candidate
type: A, layer: 1, pos: 34

### Candidate
type: A, layer: 1, pos: 22

## Relational analysis of NS_A1_A2_B2_B1_B2_B1_A2_A1

### Relational analysis result of NS_A1_A2_B2_B1_B2_B1_A2_A1
Status: Status.UNKNOWN
Output dim: 3, lower bound: -20.6035437, upper bound: 20.6026866
time: 0.78 seconds

## Relational analysis of NS_A1_A2_B2_B1_B2_B1_A2_A2

### Relational analysis result of NS_A1_A2_B2_B1_B2_B1_A2_A2
Status: Status.UNKNOWN
Output dim: 3, lower bound: -20.6035437, upper bound: 20.6024620
time: 0.74 seconds

## BFS NS instance: NS_A1_A2_B2_B1_B2_B2_A2

### Backsubstitution after applying NS history:
0: -3.4072533, 2.9987469, -3.6991673, 3.3018584, -6.7091117, 6.6979136
1: -13.4742098, 11.6417627, -14.5843849, 12.7891378, -26.2633438, 26.2261448
2: -6.7381449, 10.8741417, -7.2860594, 12.0324087, -18.7705536, 18.1602001
3: -11.7894468, 10.6054029, -12.7376213, 11.7208414, -23.5102882, 23.3430195
4: -8.6332855, 11.0345154, -9.3409872, 12.2283535, -20.8616390, 20.3754997

Time for backsubstitution: 1.27 seconds

### NS candidates at layer 1
type: A, layer: 1, pos: 11
type: B, layer: 1, pos: 34
type: B, layer: 1, pos: 11
type: B, layer: 1, pos: 12
type: B, layer: 1, pos: 24
type: A, layer: 1, pos: 34
type: A, layer: 1, pos: 22
type: A, layer: 1, pos: 12
type: B, layer: 1, pos: 9
type: A, layer: 1, pos: 31
type: B, layer: 1, pos: 31
type: B, layer: 1, pos: 42
type: A, layer: 1, pos: 39
type: B, layer: 1, pos: 39
type: B, layer: 1, pos: 16
type: A, layer: 1, pos: 16
type: B, layer: 1, pos: 33
type: A, layer: 1, pos: 23
type: B, layer: 1, pos: 23
type: A, layer: 1, pos: 4
type: A, layer: 1, pos: 33
type: A, layer: 1, pos: 47
type: A, layer: 1, pos: 30

Time for candidate selection: 0.09 seconds

### Candidate
type: A, layer: 1, pos: 11

### Candidate
type: B, layer: 1, pos: 34

### Candidate
type: B, layer: 1, pos: 11

### Candidate
type: B, layer: 1, pos: 12

### Candidate
type: B, layer: 1, pos: 24

## Relational analysis of NS_A1_A2_B2_B1_B2_B2_A2_B1

### Relational analysis result of NS_A1_A2_B2_B1_B2_B2_A2_B1
Status: Status.UNKNOWN
Output dim: 3, lower bound: -20.6037190, upper bound: 20.6002504
time: 0.73 seconds

## Relational analysis of NS_A1_A2_B2_B1_B2_B2_A2_B2

### Relational analysis result of NS_A1_A2_B2_B1_B2_B2_A2_B2
Status: Status.UNKNOWN
Output dim: 3, lower bound: -20.6037190, upper bound: 20.6002504
time: 0.74 seconds

## BFS NS instance: NS_A1_A2_B2_B2_A2_B1_A1

### Backsubstitution after applying NS history:
0: -2.9960527, 2.6623325, -3.8540320, 3.4321170, -6.4281697, 6.5163636
1: -11.8343382, 10.3386192, -15.2264605, 13.2877932, -25.1221275, 25.5650787
2: -5.9199672, 9.6447563, -7.5972466, 12.4870577, -18.4070244, 17.2420006
3: -10.3606873, 9.4523716, -13.2969398, 12.1337633, -22.4944496, 22.7493057
4: -7.5751920, 9.8309212, -9.7240124, 12.6810350, -20.2562237, 19.5549335

Time for backsubstitution: 1.11 seconds

### NS candidates at layer 1
type: A, layer: 1, pos: 11
type: B, layer: 1, pos: 11
type: A, layer: 1, pos: 47
type: B, layer: 1, pos: 34
type: A, layer: 1, pos: 34
type: B, layer: 1, pos: 12
type: A, layer: 1, pos: 12
type: B, layer: 1, pos: 9
type: B, layer: 1, pos: 4
type: B, layer: 1, pos: 42
type: A, layer: 1, pos: 4
type: A, layer: 1, pos: 31
type: B, layer: 1, pos: 31
type: A, layer: 1, pos: 30
type: B, layer: 1, pos: 39
type: A, layer: 1, pos: 33
type: B, layer: 1, pos: 23
type: A, layer: 1, pos: 39
type: B, layer: 1, pos: 16
type: B, layer: 1, pos: 33
type: A, layer: 1, pos: 16
type: A, layer: 1, pos: 23
type: B, layer: 1, pos: 47

Time for candidate selection: 0.10 seconds

### Candidate
type: A, layer: 1, pos: 11

### Candidate
type: B, layer: 1, pos: 11

### Candidate
type: A, layer: 1, pos: 47

## Relational analysis of NS_A1_A2_B2_B2_A2_B1_A1_A1

### Relational analysis result of NS_A1_A2_B2_B2_A2_B1_A1_A1
Status: Status.UNKNOWN
Output dim: 3, lower bound: -20.5967926, upper bound: 20.6032962
time: 0.63 seconds

## Relational analysis of NS_A1_A2_B2_B2_A2_B1_A1_A2

### Relational analysis result of NS_A1_A2_B2_B2_A2_B1_A1_A2
Status: Status.UNKNOWN
Output dim: 3, lower bound: -20.6041737, upper bound: 20.6041737
time: 0.62 seconds

## BFS NS instance: NS_A1_A2_B2_B2_A2_B1_A2

### Backsubstitution after applying NS history:
0: -3.5902476, 3.1974847, -3.8540320, 3.4321170, -7.0223646, 7.0515165
1: -14.1819506, 12.3807278, -15.2264605, 13.2877932, -27.4697380, 27.6071873
2: -7.0806346, 11.6148405, -7.5972466, 12.4870577, -19.5676918, 19.2120857
3: -12.3865166, 11.3239698, -13.2969398, 12.1337633, -24.5202789, 24.6209106
4: -9.0666447, 11.8067245, -9.7240124, 12.6810350, -21.7476788, 21.5307350

Time for backsubstitution: 0.94 seconds

### NS candidates at layer 1
type: A, layer: 1, pos: 11
type: B, layer: 1, pos: 11
type: B, layer: 1, pos: 34
type: A, layer: 1, pos: 34
type: B, layer: 1, pos: 4
type: A, layer: 1, pos: 4
type: B, layer: 1, pos: 12
type: A, layer: 1, pos: 12
type: B, layer: 1, pos: 9
type: B, layer: 1, pos: 42
type: A, layer: 1, pos: 47
type: A, layer: 1, pos: 30
type: B, layer: 1, pos: 39
type: A, layer: 1, pos: 39
type: B, layer: 1, pos: 16
type: A, layer: 1, pos: 16
type: A, layer: 1, pos: 33
type: A, layer: 1, pos: 31
type: B, layer: 1, pos: 31
type: B, layer: 1, pos: 33
type: B, layer: 1, pos: 23
type: A, layer: 1, pos: 23
type: B, layer: 1, pos: 47

Time for candidate selection: 0.08 seconds

### Candidate
type: A, layer: 1, pos: 11

### Candidate
type: B, layer: 1, pos: 11

### Candidate
type: B, layer: 1, pos: 34

### Candidate
type: A, layer: 1, pos: 34

### Candidate
type: B, layer: 1, pos: 4

## Relational analysis of NS_A1_A2_B2_B2_A2_B1_A2_B1

### Relational analysis result of NS_A1_A2_B2_B2_A2_B1_A2_B1
Status: Status.UNKNOWN
Output dim: 3, lower bound: -20.6037517, upper bound: 20.6032445
time: 0.68 seconds

## Relational analysis of NS_A1_A2_B2_B2_A2_B1_A2_B2

### Relational analysis result of NS_A1_A2_B2_B2_A2_B1_A2_B2
Status: Status.UNKNOWN
Output dim: 3, lower bound: -20.6041261, upper bound: 20.6041261
time: 0.80 seconds

## BFS NS instance: NS_A1_A2_B2_B2_A2_B2_A2

### Backsubstitution after applying NS history:
0: -3.3629475, 2.9603455, -4.4449782, 3.9616370, -7.3245835, 7.4053235
1: -13.2969484, 11.4919004, -17.5713806, 15.3260794, -28.6230278, 29.0632782
2: -6.6500711, 10.7346029, -8.7266226, 14.5362911, -21.1863632, 19.4612255
3: -11.6361809, 10.4710007, -15.3107166, 14.0289297, -25.6651115, 25.7817154
4: -8.5214405, 10.8940716, -11.1794167, 14.6911469, -23.2125797, 22.0734863

Time for backsubstitution: 1.08 seconds

### NS candidates at layer 1
type: A, layer: 1, pos: 11
type: A, layer: 1, pos: 34
type: B, layer: 1, pos: 34
type: B, layer: 1, pos: 11
type: A, layer: 1, pos: 22
type: B, layer: 1, pos: 9
type: B, layer: 1, pos: 12
type: A, layer: 1, pos: 30
type: B, layer: 1, pos: 31
type: A, layer: 1, pos: 12
type: A, layer: 1, pos: 31
type: B, layer: 1, pos: 4
type: A, layer: 1, pos: 4
type: B, layer: 1, pos: 42
type: B, layer: 1, pos: 39
type: A, layer: 1, pos: 16
type: A, layer: 1, pos: 33
type: B, layer: 1, pos: 23
type: B, layer: 1, pos: 16
type: A, layer: 1, pos: 23
type: B, layer: 1, pos: 33
type: A, layer: 1, pos: 39
type: B, layer: 1, pos: 47

Time for candidate selection: 0.07 seconds

### Candidate
type: A, layer: 1, pos: 11

### Candidate
type: A, layer: 1, pos: 34

### Candidate
type: B, layer: 1, pos: 34

### Candidate
type: B, layer: 1, pos: 11

### Candidate
type: A, layer: 1, pos: 22

## Relational analysis of NS_A1_A2_B2_B2_A2_B2_A2_A1

### Relational analysis result of NS_A1_A2_B2_B2_A2_B2_A2_A1
Status: Status.UNKNOWN
Output dim: 3, lower bound: -20.6033932, upper bound: 20.6022699
time: 0.68 seconds

## Relational analysis of NS_A1_A2_B2_B2_A2_B2_A2_A2

### Relational analysis result of NS_A1_A2_B2_B2_A2_B2_A2_A2
Status: Status.UNKNOWN
Output dim: 3, lower bound: -20.6033932, upper bound: 20.6020629
time: 0.72 seconds

## Summary of splitting at layer (split count: 7)
- Time for NS candidates: 2.57 seconds
NS_A1_A2_B1_B1_B1_B1_A1_A1, status: Status.VERIFIED, split count: 8, time: 2.57
Output dim: 3, lower bound: -20.5973723, upper bound: 20.5968767
NS_A1_A2_B1_B1_B1_B1_A1_A2, status: Status.UNKNOWN, split count: 8, time: 2.57
Output dim: 3, lower bound: -20.6033073, upper bound: 20.5972723
NS_A1_A2_B1_B1_B1_B1_A2_A1, status: Status.VERIFIED, split count: 8, time: 2.57
Output dim: 3, lower bound: -20.5973723, upper bound: 20.5968767
NS_A1_A2_B1_B1_B1_B1_A2_A2, status: Status.UNKNOWN, split count: 8, time: 2.57
Output dim: 3, lower bound: -20.6033073, upper bound: 20.5972723
NS_A1_A2_B1_B1_B1_B2_A1_A1, status: Status.VERIFIED, split count: 8, time: 2.57
Output dim: 3, lower bound: -20.5958956, upper bound: 20.5961608
NS_A1_A2_B1_B1_B1_B2_A1_A2, status: Status.VERIFIED, split count: 8, time: 2.57
Output dim: 3, lower bound: -20.5958956, upper bound: 20.5970383
NS_A1_A2_B1_B1_B1_B2_A2_A1, status: Status.VERIFIED, split count: 8, time: 2.57
Output dim: 3, lower bound: -20.5958956, upper bound: 20.5967687
NS_A1_A2_B1_B1_B1_B2_A2_A2, status: Status.VERIFIED, split count: 8, time: 2.57
Output dim: 3, lower bound: -20.5958956, upper bound: 20.5970383
NS_A1_A2_B1_B1_B2_B1_A2_A1, status: Status.UNKNOWN, split count: 8, time: 2.57
Output dim: 3, lower bound: -20.6035437, upper bound: 20.5994820
NS_A1_A2_B1_B1_B2_B1_A2_A2, status: Status.UNKNOWN, split count: 8, time: 2.57
Output dim: 3, lower bound: -20.6035437, upper bound: 20.5994820
NS_A1_A2_B1_B1_B2_B2_A1_B1, status: Status.UNKNOWN, split count: 8, time: 2.57
Output dim: 3, lower bound: -20.6034656, upper bound: 20.5987112
NS_A1_A2_B1_B1_B2_B2_A1_B2, status: Status.UNKNOWN, split count: 8, time: 2.57
Output dim: 3, lower bound: -20.6035433, upper bound: 20.5993457
NS_A1_A2_B1_B1_B2_B2_A2_A1, status: Status.VERIFIED, split count: 8, time: 2.57
Output dim: 3, lower bound: -20.6011143, upper bound: 20.5992028
NS_A1_A2_B1_B1_B2_B2_A2_A2, status: Status.UNKNOWN, split count: 8, time: 2.57
Output dim: 3, lower bound: -20.6035433, upper bound: 20.5993457
NS_A1_A2_B1_B2_A2_B1_A1_A1, status: Status.UNKNOWN, split count: 8, time: 2.57
Output dim: 3, lower bound: -20.5967996, upper bound: 20.6033041
NS_A1_A2_B1_B2_A2_B1_A1_A2, status: Status.UNKNOWN, split count: 8, time: 2.57
Output dim: 3, lower bound: -20.6041808, upper bound: 20.6041816
NS_A1_A2_B1_B2_A2_B1_A2_B1, status: Status.UNKNOWN, split count: 8, time: 2.57
Output dim: 3, lower bound: -20.6037517, upper bound: 20.6002704
NS_A1_A2_B1_B2_A2_B1_A2_B2, status: Status.UNKNOWN, split count: 8, time: 2.57
Output dim: 3, lower bound: -20.6041379, upper bound: 20.6041557
NS_A1_A2_B1_B2_A2_B2_A2_A1, status: Status.UNKNOWN, split count: 8, time: 2.57
Output dim: 3, lower bound: -20.6033932, upper bound: 20.5985222
NS_A1_A2_B1_B2_A2_B2_A2_A2, status: Status.UNKNOWN, split count: 8, time: 2.57
Output dim: 3, lower bound: -20.6033932, upper bound: 20.5985222
NS_A1_A2_B2_B1_B1_B1_A1_A1, status: Status.VERIFIED, split count: 8, time: 2.57
Output dim: 3, lower bound: -20.5959490, upper bound: 20.6003045
NS_A1_A2_B2_B1_B1_B1_A1_A2, status: Status.VERIFIED, split count: 8, time: 2.57
Output dim: 3, lower bound: -20.5959490, upper bound: 20.6011821
NS_A1_A2_B2_B1_B1_B1_A2_A1, status: Status.VERIFIED, split count: 8, time: 2.57
Output dim: 3, lower bound: -20.5973723, upper bound: 20.6006040
NS_A1_A2_B2_B1_B1_B1_A2_A2, status: Status.UNKNOWN, split count: 8, time: 2.57
Output dim: 3, lower bound: -20.6033073, upper bound: 20.6010027
NS_A1_A2_B2_B1_B2_B1_A2_A1, status: Status.UNKNOWN, split count: 8, time: 2.57
Output dim: 3, lower bound: -20.6035437, upper bound: 20.6026866
NS_A1_A2_B2_B1_B2_B1_A2_A2, status: Status.UNKNOWN, split count: 8, time: 2.57
Output dim: 3, lower bound: -20.6035437, upper bound: 20.6024620
NS_A1_A2_B2_B1_B2_B2_A2_B1, status: Status.UNKNOWN, split count: 8, time: 2.57
Output dim: 3, lower bound: -20.6037190, upper bound: 20.6002504
NS_A1_A2_B2_B1_B2_B2_A2_B2, status: Status.UNKNOWN, split count: 8, time: 2.57
Output dim: 3, lower bound: -20.6037190, upper bound: 20.6002504
NS_A1_A2_B2_B2_A2_B1_A1_A1, status: Status.UNKNOWN, split count: 8, time: 2.57
Output dim: 3, lower bound: -20.5967926, upper bound: 20.6032962
NS_A1_A2_B2_B2_A2_B1_A1_A2, status: Status.UNKNOWN, split count: 8, time: 2.57
Output dim: 3, lower bound: -20.6041737, upper bound: 20.6041737
NS_A1_A2_B2_B2_A2_B1_A2_B1, status: Status.UNKNOWN, split count: 8, time: 2.57
Output dim: 3, lower bound: -20.6037517, upper bound: 20.6032445
NS_A1_A2_B2_B2_A2_B1_A2_B2, status: Status.UNKNOWN, split count: 8, time: 2.57
Output dim: 3, lower bound: -20.6041261, upper bound: 20.6041261
NS_A1_A2_B2_B2_A2_B2_A2_A1, status: Status.UNKNOWN, split count: 8, time: 2.57
Output dim: 3, lower bound: -20.6033932, upper bound: 20.6022699
NS_A1_A2_B2_B2_A2_B2_A2_A2, status: Status.UNKNOWN, split count: 8, time: 2.57
Output dim: 3, lower bound: -20.6033932, upper bound: 20.6020629

## BFS NS instance: NS_A1_A2_B1_B1_B1_B1_A1_A2

### Backsubstitution after applying NS history:
0: -3.0719039, 2.7303421, -2.8369715, 2.5377181, -5.6096220, 5.5673137
1: -12.1338568, 10.5984812, -11.1648302, 9.8173466, -21.9512024, 21.7633114
2: -6.0763721, 9.8739471, -5.6087904, 9.2279978, -15.3043671, 15.4827356
3: -10.6220951, 9.6835747, -9.7681313, 9.0039520, -19.6260471, 19.4517059
4: -7.7725878, 10.0674229, -7.1507564, 9.4026794, -17.1752605, 17.2181797

Time for backsubstitution: 0.94 seconds

### NS candidates at layer 1
type: A, layer: 1, pos: 11
type: B, layer: 1, pos: 11
type: B, layer: 1, pos: 34
type: A, layer: 1, pos: 34
type: B, layer: 1, pos: 4
type: B, layer: 1, pos: 9
type: A, layer: 1, pos: 4
type: B, layer: 1, pos: 42
type: A, layer: 1, pos: 42
type: B, layer: 1, pos: 12
type: A, layer: 1, pos: 31
type: B, layer: 1, pos: 31
type: B, layer: 1, pos: 16
type: A, layer: 1, pos: 39
type: A, layer: 1, pos: 16
type: B, layer: 1, pos: 33
type: A, layer: 1, pos: 33
type: B, layer: 1, pos: 39
type: A, layer: 1, pos: 23
type: B, layer: 1, pos: 23
type: A, layer: 1, pos: 47
type: A, layer: 1, pos: 30

Time for candidate selection: 0.08 seconds

### Candidate
type: A, layer: 1, pos: 11

### Candidate
type: B, layer: 1, pos: 11

### Candidate
type: B, layer: 1, pos: 34

### Candidate
type: A, layer: 1, pos: 34

### Candidate
type: B, layer: 1, pos: 4

## Relational analysis of NS_A1_A2_B1_B1_B1_B1_A1_A2_B1

### Relational analysis result of NS_A1_A2_B1_B1_B1_B1_A1_A2_B1
Status: Status.UNKNOWN
Output dim: 3, lower bound: -20.6032705, upper bound: 20.5969143
time: 0.74 seconds

## Relational analysis of NS_A1_A2_B1_B1_B1_B1_A1_A2_B2
Optimization infeasible because this subproblem isn't reachable.

### Candidate
type: B, layer: 1, pos: 9

## Relational analysis of NS_A1_A2_B1_B1_B1_B1_A1_A2_B1
Optimization infeasible because this subproblem isn't reachable.

### Candidate
type: A, layer: 1, pos: 4

### Candidate
type: B, layer: 1, pos: 42

## Relational analysis of NS_A1_A2_B1_B1_B1_B1_A1_A2_B1
Optimization infeasible because this subproblem isn't reachable.

### Candidate
type: A, layer: 1, pos: 42

### Candidate
type: B, layer: 1, pos: 12

## Relational analysis of NS_A1_A2_B1_B1_B1_B1_A1_A2_B1
Optimization infeasible because this subproblem isn't reachable.

### Candidate
type: A, layer: 1, pos: 31

## Relational analysis of NS_A1_A2_B1_B1_B1_B1_A1_A2_A1

### Relational analysis result of NS_A1_A2_B1_B1_B1_B1_A1_A2_A1
Status: Status.UNKNOWN
Output dim: 3, lower bound: -20.6033197, upper bound: 20.5972793
time: 0.62 seconds

## Relational analysis of NS_A1_A2_B1_B1_B1_B1_A1_A2_A2

### Relational analysis result of NS_A1_A2_B1_B1_B1_B1_A1_A2_A2
Status: Status.VERIFIED
Output dim: 3, lower bound: -20.6030875, upper bound: 20.5966088
time: 0.80 seconds

## BFS NS instance: NS_A1_A2_B1_B1_B1_B1_A2_A2

### Backsubstitution after applying NS history:
0: -3.7551029, 3.3418815, -2.8369715, 2.5377181, -6.2928209, 6.1788530
1: -14.8365059, 12.9373684, -11.1648302, 9.8173466, -24.6538525, 24.1021938
2: -7.4026179, 12.1471291, -5.6087904, 9.2279978, -16.6306152, 17.7559185
3: -12.9592552, 11.8153782, -9.7681313, 9.0039520, -21.9632034, 21.5835094
4: -9.4836941, 12.3391991, -7.1507564, 9.4026794, -18.8863735, 19.4899559

Time for backsubstitution: 1.23 seconds

### NS candidates at layer 1
type: B, layer: 1, pos: 11
type: A, layer: 1, pos: 11
type: B, layer: 1, pos: 34
type: A, layer: 1, pos: 34
type: B, layer: 1, pos: 4
type: A, layer: 1, pos: 4
type: B, layer: 1, pos: 9
type: B, layer: 1, pos: 42
type: A, layer: 1, pos: 42
type: B, layer: 1, pos: 12
type: A, layer: 1, pos: 31
type: B, layer: 1, pos: 31
type: A, layer: 1, pos: 39
type: B, layer: 1, pos: 16
type: B, layer: 1, pos: 33
type: A, layer: 1, pos: 16
type: A, layer: 1, pos: 23
type: A, layer: 1, pos: 33
type: B, layer: 1, pos: 39
type: B, layer: 1, pos: 23
type: A, layer: 1, pos: 30
type: A, layer: 1, pos: 47

Time for candidate selection: 0.11 seconds

### Candidate
type: B, layer: 1, pos: 11

### Candidate
type: A, layer: 1, pos: 11

### Candidate
type: B, layer: 1, pos: 34

### Candidate
type: A, layer: 1, pos: 34

### Candidate
type: B, layer: 1, pos: 4

## Relational analysis of NS_A1_A2_B1_B1_B1_B1_A2_A2_B1

### Relational analysis result of NS_A1_A2_B1_B1_B1_B1_A2_A2_B1
Status: Status.UNKNOWN
Output dim: 3, lower bound: -20.6031924, upper bound: 20.5969028
time: 0.69 seconds

## Relational analysis of NS_A1_A2_B1_B1_B1_B1_A2_A2_B2
Optimization infeasible because this subproblem isn't reachable.

### Candidate
type: A, layer: 1, pos: 4

### Candidate
type: B, layer: 1, pos: 9

## Relational analysis of NS_A1_A2_B1_B1_B1_B1_A2_A2_B1
Optimization infeasible because this subproblem isn't reachable.

### Candidate
type: B, layer: 1, pos: 42

## Relational analysis of NS_A1_A2_B1_B1_B1_B1_A2_A2_B1
Optimization infeasible because this subproblem isn't reachable.

### Candidate
type: A, layer: 1, pos: 42

### Candidate
type: B, layer: 1, pos: 12

## Relational analysis of NS_A1_A2_B1_B1_B1_B1_A2_A2_B1
Optimization infeasible because this subproblem isn't reachable.

### Candidate
type: A, layer: 1, pos: 31

## Relational analysis of NS_A1_A2_B1_B1_B1_B1_A2_A2_A1

### Relational analysis result of NS_A1_A2_B1_B1_B1_B1_A2_A2_A1
Status: Status.UNKNOWN
Output dim: 3, lower bound: -20.6033073, upper bound: 20.5972557
time: 0.72 seconds

## Relational analysis of NS_A1_A2_B1_B1_B1_B1_A2_A2_A2

### Relational analysis result of NS_A1_A2_B1_B1_B1_B1_A2_A2_A2
Status: Status.UNKNOWN
Output dim: 3, lower bound: -20.6031824, upper bound: 20.5972723
time: 0.76 seconds

## BFS NS instance: NS_A1_A2_B1_B1_B2_B1_A2_A1

### Backsubstitution after applying NS history:
0: -2.9960527, 2.6623325, -2.6776226, 2.3901095, -5.3861623, 5.3399544
1: -11.8343382, 10.3386192, -10.5423784, 9.2746677, -21.1090050, 20.8809967
2: -5.9199672, 9.6447563, -5.3035603, 8.6706009, -14.5905685, 14.9483166
3: -10.3606873, 9.4523716, -9.2467489, 8.5087490, -18.8694363, 18.6991177
4: -7.5751920, 9.8309212, -6.7716413, 8.8663626, -16.4415531, 16.6025620

Time for backsubstitution: 1.16 seconds

### NS candidates at layer 1
type: B, layer: 1, pos: 11
type: A, layer: 1, pos: 11
type: B, layer: 1, pos: 34
type: A, layer: 1, pos: 34
type: A, layer: 1, pos: 12
type: B, layer: 1, pos: 12
type: B, layer: 1, pos: 9
type: A, layer: 1, pos: 31
type: A, layer: 1, pos: 39
type: B, layer: 1, pos: 42
type: B, layer: 1, pos: 31
type: B, layer: 1, pos: 33
type: B, layer: 1, pos: 16
type: A, layer: 1, pos: 16
type: A, layer: 1, pos: 23
type: B, layer: 1, pos: 24
type: B, layer: 1, pos: 39
type: A, layer: 1, pos: 4
type: B, layer: 1, pos: 23
type: A, layer: 1, pos: 33
type: A, layer: 1, pos: 47
type: A, layer: 1, pos: 30

Time for candidate selection: 0.11 seconds

### Candidate
type: B, layer: 1, pos: 11

### Candidate
type: A, layer: 1, pos: 11

### Candidate
type: B, layer: 1, pos: 34

### Candidate
type: A, layer: 1, pos: 34

### Candidate
type: A, layer: 1, pos: 12

### Candidate
type: B, layer: 1, pos: 12

### Candidate
type: B, layer: 1, pos: 9

## Relational analysis of NS_A1_A2_B1_B1_B2_B1_A2_A1_B1
Optimization infeasible because this subproblem isn't reachable.

### Candidate
type: A, layer: 1, pos: 31

## Relational analysis of NS_A1_A2_B1_B1_B2_B1_A2_A1_A1

### Relational analysis result of NS_A1_A2_B1_B1_B2_B1_A2_A1_A1
Status: Status.UNKNOWN
Output dim: 3, lower bound: -20.6035437, upper bound: 20.5994695
time: 0.77 seconds

## Relational analysis of NS_A1_A2_B1_B1_B2_B1_A2_A1_A2

### Relational analysis result of NS_A1_A2_B1_B1_B2_B1_A2_A1_A2
Status: Status.UNKNOWN
Output dim: 3, lower bound: -20.6033286, upper bound: 20.5994820
time: 0.65 seconds

## BFS NS instance: NS_A1_A2_B1_B1_B2_B1_A2_A2

### Backsubstitution after applying NS history:
0: -3.5902476, 3.1974847, -2.6776226, 2.3901095, -5.9803572, 5.8751073
1: -14.1819506, 12.3807278, -10.5423784, 9.2746677, -23.4566154, 22.9231071
2: -7.0806346, 11.6148405, -5.3035603, 8.6706009, -15.7512331, 16.9183998
3: -12.3865166, 11.3239698, -9.2467489, 8.5087490, -20.8952637, 20.5707188
4: -9.0666447, 11.8067245, -6.7716413, 8.8663626, -17.9330063, 18.5783634

Time for backsubstitution: 1.02 seconds

### NS candidates at layer 1
type: B, layer: 1, pos: 11
type: B, layer: 1, pos: 34
type: A, layer: 1, pos: 34
type: A, layer: 1, pos: 12
type: A, layer: 1, pos: 11
type: B, layer: 1, pos: 12
type: B, layer: 1, pos: 9
type: A, layer: 1, pos: 39
type: B, layer: 1, pos: 33
type: A, layer: 1, pos: 31
type: B, layer: 1, pos: 31
type: B, layer: 1, pos: 42
type: B, layer: 1, pos: 16
type: A, layer: 1, pos: 23
type: A, layer: 1, pos: 16
type: A, layer: 1, pos: 4
type: B, layer: 1, pos: 39
type: B, layer: 1, pos: 24
type: B, layer: 1, pos: 23
type: A, layer: 1, pos: 33
type: A, layer: 1, pos: 30
type: A, layer: 1, pos: 47

Time for candidate selection: 0.07 seconds

### Candidate
type: B, layer: 1, pos: 11

### Candidate
type: B, layer: 1, pos: 34

### Candidate
type: A, layer: 1, pos: 34

### Candidate
type: A, layer: 1, pos: 12

### Candidate
type: A, layer: 1, pos: 11

### Candidate
type: B, layer: 1, pos: 12

### Candidate
type: B, layer: 1, pos: 9

## Relational analysis of NS_A1_A2_B1_B1_B2_B1_A2_A2_B1
Optimization infeasible because this subproblem isn't reachable.

### Candidate
type: A, layer: 1, pos: 39

## Relational analysis of NS_A1_A2_B1_B1_B2_B1_A2_A2_A1
Optimization infeasible because this subproblem isn't reachable.

### Candidate
type: B, layer: 1, pos: 33

## Relational analysis of NS_A1_A2_B1_B1_B2_B1_A2_A2_B1

### Relational analysis result of NS_A1_A2_B1_B1_B2_B1_A2_A2_B1
Status: Status.UNKNOWN
Output dim: 3, lower bound: -20.6033752, upper bound: 20.5990013
time: 0.79 seconds

## Relational analysis of NS_A1_A2_B1_B1_B2_B1_A2_A2_B2

### Relational analysis result of NS_A1_A2_B1_B1_B2_B1_A2_A2_B2
Status: Status.UNKNOWN
Output dim: 3, lower bound: -20.6034812, upper bound: 20.5994820
time: 0.74 seconds

## BFS NS instance: NS_A1_A2_B1_B1_B2_B2_A1_B1

### Backsubstitution after applying NS history:
0: -3.1597602, 2.8049958, -2.1720495, 1.9849454, -5.1447048, 4.9770446
1: -12.4845533, 10.8861303, -8.5200977, 7.6689053, -20.1534576, 19.4062271
2: -6.2473922, 10.1505280, -4.3270049, 7.1652737, -13.4126663, 14.4775324
3: -10.9279404, 9.9419422, -7.4694524, 7.0567627, -17.9847031, 17.4113941
4: -7.9961886, 10.3389921, -5.4614787, 7.3901196, -15.3863087, 15.8004704

Time for backsubstitution: 0.94 seconds

### NS candidates at layer 1
type: B, layer: 1, pos: 11
type: B, layer: 1, pos: 34
type: A, layer: 1, pos: 34
type: B, layer: 1, pos: 12
type: A, layer: 1, pos: 12
type: A, layer: 1, pos: 39
type: B, layer: 1, pos: 9
type: A, layer: 1, pos: 31
type: A, layer: 1, pos: 11
type: B, layer: 1, pos: 31
type: A, layer: 1, pos: 42
type: B, layer: 1, pos: 33
type: A, layer: 1, pos: 4
type: B, layer: 1, pos: 16
type: A, layer: 1, pos: 23
type: A, layer: 1, pos: 16
type: B, layer: 1, pos: 39
type: B, layer: 1, pos: 24
type: B, layer: 1, pos: 23
type: A, layer: 1, pos: 33
type: A, layer: 1, pos: 30
type: A, layer: 1, pos: 47

Time for candidate selection: 0.07 seconds

### Candidate
type: B, layer: 1, pos: 11

### Candidate
type: B, layer: 1, pos: 34

### Candidate
type: A, layer: 1, pos: 34

### Candidate
type: B, layer: 1, pos: 12

### Candidate
type: A, layer: 1, pos: 12

### Candidate
type: A, layer: 1, pos: 39

## Relational analysis of NS_A1_A2_B1_B1_B2_B2_A1_B1_A1
Optimization infeasible because this subproblem isn't reachable.

### Candidate
type: B, layer: 1, pos: 9

## Relational analysis of NS_A1_A2_B1_B1_B2_B2_A1_B1_B1
Optimization infeasible because this subproblem isn't reachable.

### Candidate
type: A, layer: 1, pos: 31

## Relational analysis of NS_A1_A2_B1_B1_B2_B2_A1_B1_A1

### Relational analysis result of NS_A1_A2_B1_B1_B2_B2_A1_B1_A1
Status: Status.UNKNOWN
Output dim: 3, lower bound: -20.6034951, upper bound: 20.5987226
time: 0.65 seconds

## Relational analysis of NS_A1_A2_B1_B1_B2_B2_A1_B1_A2

### Relational analysis result of NS_A1_A2_B1_B1_B2_B2_A1_B1_A2
Status: Status.UNKNOWN
Output dim: 3, lower bound: -20.6032505, upper bound: 20.5987197
time: 0.78 seconds

## BFS NS instance: NS_A1_A2_B1_B1_B2_B2_A1_B2

### Backsubstitution after applying NS history:
0: -3.1597602, 2.8049958, -2.8515317, 2.5491934, -5.7089524, 5.6565275
1: -12.4845533, 10.8861303, -11.2261553, 9.8856926, -22.3702431, 22.1122856
2: -6.2473922, 10.1505280, -5.6394186, 9.2582951, -15.5056868, 15.7899466
3: -10.9279404, 9.9419422, -9.8397264, 9.0728273, -20.0007668, 19.7816658
4: -7.9961886, 10.3389921, -7.2097111, 9.4593086, -17.4554977, 17.5487022

Time for backsubstitution: 0.94 seconds

### NS candidates at layer 1
type: B, layer: 1, pos: 11
type: B, layer: 1, pos: 34
type: A, layer: 1, pos: 11
type: A, layer: 1, pos: 34
type: A, layer: 1, pos: 12
type: B, layer: 1, pos: 12
type: B, layer: 1, pos: 9
type: A, layer: 1, pos: 39
type: B, layer: 1, pos: 33
type: A, layer: 1, pos: 31
type: A, layer: 1, pos: 42
type: B, layer: 1, pos: 16
type: A, layer: 1, pos: 4
type: B, layer: 1, pos: 31
type: A, layer: 1, pos: 23
type: A, layer: 1, pos: 16
type: B, layer: 1, pos: 24
type: B, layer: 1, pos: 23
type: B, layer: 1, pos: 39
type: A, layer: 1, pos: 33
type: A, layer: 1, pos: 47
type: A, layer: 1, pos: 30

Time for candidate selection: 0.07 seconds

### Candidate
type: B, layer: 1, pos: 11

### Candidate
type: B, layer: 1, pos: 34

### Candidate
type: A, layer: 1, pos: 11

### Candidate
type: A, layer: 1, pos: 34

### Candidate
type: A, layer: 1, pos: 12

### Candidate
type: B, layer: 1, pos: 12

### Candidate
type: B, layer: 1, pos: 9

## Relational analysis of NS_A1_A2_B1_B1_B2_B2_A1_B2_B1
Optimization infeasible because this subproblem isn't reachable.

### Candidate
type: A, layer: 1, pos: 39

## Relational analysis of NS_A1_A2_B1_B1_B2_B2_A1_B2_A1
Optimization infeasible because this subproblem isn't reachable.

### Candidate
type: B, layer: 1, pos: 33

## Relational analysis of NS_A1_A2_B1_B1_B2_B2_A1_B2_B1
Optimization infeasible because this subproblem isn't reachable.

### Candidate
type: A, layer: 1, pos: 31

## Relational analysis of NS_A1_A2_B1_B1_B2_B2_A1_B2_A1

### Relational analysis result of NS_A1_A2_B1_B1_B2_B2_A1_B2_A1
Status: Status.UNKNOWN
Output dim: 3, lower bound: -20.6035729, upper bound: 20.5993572
time: 0.81 seconds

## Relational analysis of NS_A1_A2_B1_B1_B2_B2_A1_B2_A2

### Relational analysis result of NS_A1_A2_B1_B1_B2_B2_A1_B2_A2
Status: Status.UNKNOWN
Output dim: 3, lower bound: -20.6033283, upper bound: 20.5993543
time: 0.82 seconds

## BFS NS instance: NS_A1_A2_B1_B1_B2_B2_A2_A2

### Backsubstitution after applying NS history:
0: -3.6803167, 3.2757277, -3.0070853, 2.6840725, -6.3643894, 6.2828131
1: -14.5396061, 12.6777191, -11.8437071, 10.4111977, -24.9508018, 24.5214272
2: -7.2490587, 11.9156590, -5.9487362, 9.7418251, -16.9908829, 17.8643951
3: -12.6998749, 11.5894537, -10.3784885, 9.5441866, -22.2440548, 21.9679413
4: -9.2898760, 12.1022100, -7.6077647, 9.9460669, -19.2359409, 19.7099724

Time for backsubstitution: 1.00 seconds

### NS candidates at layer 1
type: B, layer: 1, pos: 11
type: B, layer: 1, pos: 34
type: A, layer: 1, pos: 34
type: A, layer: 1, pos: 11
type: A, layer: 1, pos: 12
type: B, layer: 1, pos: 12
type: B, layer: 1, pos: 9
type: A, layer: 1, pos: 39
type: B, layer: 1, pos: 33
type: B, layer: 1, pos: 16
type: B, layer: 1, pos: 42
type: A, layer: 1, pos: 4
type: A, layer: 1, pos: 23
type: A, layer: 1, pos: 31
type: B, layer: 1, pos: 31
type: A, layer: 1, pos: 16
type: B, layer: 1, pos: 23
type: B, layer: 1, pos: 24
type: B, layer: 1, pos: 39
type: A, layer: 1, pos: 33
type: A, layer: 1, pos: 30
type: A, layer: 1, pos: 47

Time for candidate selection: 0.08 seconds

### Candidate
type: B, layer: 1, pos: 11

### Candidate
type: B, layer: 1, pos: 34

### Candidate
type: A, layer: 1, pos: 34

### Candidate
type: A, layer: 1, pos: 11

### Candidate
type: A, layer: 1, pos: 12

### Candidate
type: B, layer: 1, pos: 12

### Candidate
type: B, layer: 1, pos: 9

## Relational analysis of NS_A1_A2_B1_B1_B2_B2_A2_A2_B1
Optimization infeasible because this subproblem isn't reachable.

### Candidate
type: A, layer: 1, pos: 39

## Relational analysis of NS_A1_A2_B1_B1_B2_B2_A2_A2_A1
Optimization infeasible because this subproblem isn't reachable.

### Candidate
type: B, layer: 1, pos: 33

## Relational analysis of NS_A1_A2_B1_B1_B2_B2_A2_A2_B1
Optimization infeasible because this subproblem isn't reachable.

### Candidate
type: B, layer: 1, pos: 16

## Relational analysis of NS_A1_A2_B1_B1_B2_B2_A2_A2_B1
Optimization infeasible because this subproblem isn't reachable.

### Candidate
type: B, layer: 1, pos: 42

## Relational analysis of NS_A1_A2_B1_B1_B2_B2_A2_A2_B1

### Relational analysis result of NS_A1_A2_B1_B1_B2_B2_A2_A2_B1
Status: Status.UNKNOWN
Output dim: 3, lower bound: -20.6034656, upper bound: 20.5987112
time: 0.75 seconds

## Relational analysis of NS_A1_A2_B1_B1_B2_B2_A2_A2_B2

### Relational analysis result of NS_A1_A2_B1_B1_B2_B2_A2_A2_B2
Status: Status.UNKNOWN
Output dim: 3, lower bound: -20.6034656, upper bound: 20.5993457
time: 0.78 seconds

## BFS NS instance: NS_A1_A2_B1_B2_A2_B1_A1_A1

### Backsubstitution after applying NS history:
0: -2.9575121, 2.6303887, -3.1507802, 2.8027632, -5.7602754, 5.7811689
1: -11.6835804, 10.1886177, -12.4441538, 10.8765450, -22.5601254, 22.6327705
2: -5.8330021, 9.5619545, -6.2334828, 10.1443806, -15.9773827, 15.7954369
3: -10.2288876, 9.3145103, -10.8921318, 9.9383621, -20.1672497, 20.2066402
4: -7.4482050, 9.7160215, -7.9639950, 10.3374100, -17.7856140, 17.6800156

Time for backsubstitution: 1.10 seconds

### NS candidates at layer 1
type: B, layer: 1, pos: 11
type: A, layer: 1, pos: 11
type: B, layer: 1, pos: 34
type: A, layer: 1, pos: 34
type: B, layer: 1, pos: 47
type: A, layer: 1, pos: 31
type: B, layer: 1, pos: 31
type: B, layer: 1, pos: 12
type: A, layer: 1, pos: 12
type: B, layer: 1, pos: 9
type: A, layer: 1, pos: 39
type: B, layer: 1, pos: 4
type: B, layer: 1, pos: 42
type: B, layer: 1, pos: 39
type: A, layer: 1, pos: 4
type: B, layer: 1, pos: 16
type: A, layer: 1, pos: 16
type: B, layer: 1, pos: 33
type: A, layer: 1, pos: 23
type: B, layer: 1, pos: 23
type: A, layer: 1, pos: 33
type: A, layer: 1, pos: 30

Time for candidate selection: 0.08 seconds

### Candidate
type: B, layer: 1, pos: 11

### Candidate
type: A, layer: 1, pos: 11

### Candidate
type: B, layer: 1, pos: 34

### Candidate
type: A, layer: 1, pos: 34

### Candidate
type: B, layer: 1, pos: 47

## Relational analysis of NS_A1_A2_B1_B2_A2_B1_A1_A1_B1

### Relational analysis result of NS_A1_A2_B1_B2_A2_B1_A1_A1_B1
Status: Status.VERIFIED
Output dim: 3, lower bound: -20.5959150, upper bound: 20.5959150
time: 0.52 seconds

## Relational analysis of NS_A1_A2_B1_B2_A2_B1_A1_A1_B2

### Relational analysis result of NS_A1_A2_B1_B2_A2_B1_A1_A1_B2
Status: Status.UNKNOWN
Output dim: 3, lower bound: -20.5959150, upper bound: 20.6033041
time: 0.62 seconds

## BFS NS instance: NS_A1_A2_B1_B2_A2_B1_A1_A2

### Backsubstitution after applying NS history:
0: -2.9541757, 2.6261964, -3.1507802, 2.8027632, -5.7569389, 5.7769766
1: -11.6670923, 10.1961460, -12.4441538, 10.8765450, -22.5436344, 22.6402969
2: -5.8364959, 9.5153542, -6.2334828, 10.1443806, -15.9808769, 15.7488365
3: -10.2159939, 9.3246737, -10.8921318, 9.9383621, -20.1543541, 20.2168045
4: -7.4689107, 9.7003708, -7.9639950, 10.3374100, -17.8063202, 17.6643658

Time for backsubstitution: 0.96 seconds

### NS candidates at layer 1
type: A, layer: 1, pos: 11
type: B, layer: 1, pos: 11
type: B, layer: 1, pos: 34
type: A, layer: 1, pos: 34
type: B, layer: 1, pos: 12
type: A, layer: 1, pos: 12
type: B, layer: 1, pos: 9
type: B, layer: 1, pos: 31
type: A, layer: 1, pos: 30
type: A, layer: 1, pos: 31
type: B, layer: 1, pos: 42
type: B, layer: 1, pos: 4
type: A, layer: 1, pos: 4
type: B, layer: 1, pos: 16
type: A, layer: 1, pos: 16
type: B, layer: 1, pos: 39
type: B, layer: 1, pos: 23
type: A, layer: 1, pos: 39
type: A, layer: 1, pos: 23
type: A, layer: 1, pos: 33
type: B, layer: 1, pos: 33
type: B, layer: 1, pos: 47

Time for candidate selection: 0.08 seconds

### Candidate
type: A, layer: 1, pos: 11

### Candidate
type: B, layer: 1, pos: 11

### Candidate
type: B, layer: 1, pos: 34

### Candidate
type: A, layer: 1, pos: 34

### Candidate
type: B, layer: 1, pos: 12

### Candidate
type: A, layer: 1, pos: 12

### Candidate
type: B, layer: 1, pos: 9

## Relational analysis of NS_A1_A2_B1_B2_A2_B1_A1_A2_B1
Optimization infeasible because this subproblem isn't reachable.

### Candidate
type: B, layer: 1, pos: 31

## Relational analysis of NS_A1_A2_B1_B2_A2_B1_A1_A2_B1

### Relational analysis result of NS_A1_A2_B1_B2_A2_B1_A1_A2_B1
Status: Status.UNKNOWN
Output dim: 3, lower bound: -20.6039454, upper bound: 20.6041886
time: 0.83 seconds

## Relational analysis of NS_A1_A2_B1_B2_A2_B1_A1_A2_B2

### Relational analysis result of NS_A1_A2_B1_B2_A2_B1_A1_A2_B2
Status: Status.UNKNOWN
Output dim: 3, lower bound: -20.6039453, upper bound: 20.6039453
time: 0.96 seconds

## BFS NS instance: NS_A1_A2_B1_B2_A2_B1_A2_B1

### Backsubstitution after applying NS history:
0: -3.5902476, 3.1974847, -2.8267651, 2.5123618, -6.1026096, 6.0242481
1: -14.1819506, 12.3807278, -11.1671572, 9.7449017, -23.9268532, 23.5478840
2: -7.0806346, 11.6148405, -5.6000242, 9.0970116, -16.1776428, 17.2148647
3: -12.3865166, 11.3239698, -9.7856503, 8.9046326, -21.2911491, 21.1096153
4: -9.0666447, 11.8067245, -7.1425700, 9.2732248, -18.3398705, 18.9492931

Time for backsubstitution: 1.01 seconds

### NS candidates at layer 1
type: B, layer: 1, pos: 11
type: A, layer: 1, pos: 11
type: B, layer: 1, pos: 47
type: A, layer: 1, pos: 34
type: B, layer: 1, pos: 34
type: A, layer: 1, pos: 12
type: B, layer: 1, pos: 12
type: A, layer: 1, pos: 30
type: B, layer: 1, pos: 9
type: A, layer: 1, pos: 31
type: B, layer: 1, pos: 31
type: B, layer: 1, pos: 42
type: A, layer: 1, pos: 4
type: A, layer: 1, pos: 16
type: B, layer: 1, pos: 16
type: A, layer: 1, pos: 39
type: B, layer: 1, pos: 33
type: A, layer: 1, pos: 23
type: B, layer: 1, pos: 23
type: A, layer: 1, pos: 33
type: B, layer: 1, pos: 39
type: A, layer: 1, pos: 47

Time for candidate selection: 0.08 seconds

### Candidate
type: B, layer: 1, pos: 11

### Candidate
type: A, layer: 1, pos: 11

### Candidate
type: B, layer: 1, pos: 47

## Relational analysis of NS_A1_A2_B1_B2_A2_B1_A2_B1_B1

### Relational analysis result of NS_A1_A2_B1_B2_A2_B1_A2_B1_B1
Status: Status.UNKNOWN
Output dim: 3, lower bound: -20.6032393, upper bound: 20.5964160
time: 0.85 seconds

## Relational analysis of NS_A1_A2_B1_B2_A2_B1_A2_B1_B2

### Relational analysis result of NS_A1_A2_B1_B2_A2_B1_A2_B1_B2
Status: Status.UNKNOWN
Output dim: 3, lower bound: -20.6037242, upper bound: 20.6002121
time: 0.62 seconds

## BFS NS instance: NS_A1_A2_B1_B2_A2_B1_A2_B2

### Backsubstitution after applying NS history:
0: -3.5902476, 3.1974847, -3.0771875, 2.7368279, -6.3270755, 6.2746720
1: -14.1819506, 12.3807278, -12.1525078, 10.6178522, -24.7998009, 24.5332355
2: -7.0806346, 11.6148405, -6.0879107, 9.9097996, -16.9904346, 17.7027512
3: -12.3865166, 11.3239698, -10.6401510, 9.7027712, -22.0892849, 21.9641209
4: -9.0666447, 11.8067245, -7.7787948, 10.0979300, -19.1645737, 19.5855179

Time for backsubstitution: 1.18 seconds

### NS candidates at layer 1
type: B, layer: 1, pos: 11
type: A, layer: 1, pos: 11
type: A, layer: 1, pos: 34
type: B, layer: 1, pos: 34
type: B, layer: 1, pos: 47
type: A, layer: 1, pos: 12
type: B, layer: 1, pos: 12
type: B, layer: 1, pos: 9
type: A, layer: 1, pos: 30
type: A, layer: 1, pos: 4
type: B, layer: 1, pos: 42
type: B, layer: 1, pos: 16
type: A, layer: 1, pos: 16
type: A, layer: 1, pos: 39
type: B, layer: 1, pos: 33
type: B, layer: 1, pos: 31
type: A, layer: 1, pos: 31
type: A, layer: 1, pos: 23
type: B, layer: 1, pos: 39
type: B, layer: 1, pos: 23
type: A, layer: 1, pos: 33
type: A, layer: 1, pos: 47

Time for candidate selection: 0.09 seconds

### Candidate
type: B, layer: 1, pos: 11

### Candidate
type: A, layer: 1, pos: 11

### Candidate
type: A, layer: 1, pos: 34

### Candidate
type: B, layer: 1, pos: 34

### Candidate
type: B, layer: 1, pos: 47

## Relational analysis of NS_A1_A2_B1_B2_A2_B1_A2_B2_B1
Optimization infeasible because this subproblem isn't reachable.

### Candidate
type: A, layer: 1, pos: 12

### Candidate
type: B, layer: 1, pos: 12

### Candidate
type: B, layer: 1, pos: 9

## Relational analysis of NS_A1_A2_B1_B2_A2_B1_A2_B2_B1
Optimization infeasible because this subproblem isn't reachable.

### Candidate
type: A, layer: 1, pos: 30

## Relational analysis of NS_A1_A2_B1_B2_A2_B1_A2_B2_A1

### Relational analysis result of NS_A1_A2_B1_B2_A2_B1_A2_B2_A1
Status: Status.UNKNOWN
Output dim: 3, lower bound: -20.6037382, upper bound: 20.6041557
time: 0.74 seconds

## Relational analysis of NS_A1_A2_B1_B2_A2_B1_A2_B2_A2

### Relational analysis result of NS_A1_A2_B1_B2_A2_B1_A2_B2_A2
Status: Status.UNKNOWN
Output dim: 3, lower bound: -20.6037382, upper bound: 20.6040943
time: 0.79 seconds

## BFS NS instance: NS_A1_A2_B1_B2_A2_B2_A2_A1

### Backsubstitution after applying NS history:
0: -2.9541757, 2.6261964, -3.7443836, 3.3308897, -6.2850657, 6.3705797
1: -11.6670923, 10.1961460, -14.8040266, 12.8840399, -24.5511322, 25.0001717
2: -5.8364959, 9.5153542, -7.3696198, 12.1977873, -18.0342827, 16.8849735
3: -10.2159939, 9.3246737, -12.9243069, 11.7908812, -22.0068741, 22.2489815
4: -7.4689107, 9.7003708, -9.4258900, 12.3335075, -19.8024178, 19.1262608

Time for backsubstitution: 1.36 seconds

### NS candidates at layer 1
type: A, layer: 1, pos: 11
type: B, layer: 1, pos: 11
type: A, layer: 1, pos: 34
type: B, layer: 1, pos: 34
type: A, layer: 1, pos: 30
type: B, layer: 1, pos: 9
type: B, layer: 1, pos: 12
type: A, layer: 1, pos: 12
type: B, layer: 1, pos: 31
type: A, layer: 1, pos: 31
type: B, layer: 1, pos: 42
type: B, layer: 1, pos: 4
type: A, layer: 1, pos: 4
type: B, layer: 1, pos: 39
type: B, layer: 1, pos: 23
type: A, layer: 1, pos: 16
type: A, layer: 1, pos: 33
type: B, layer: 1, pos: 16
type: A, layer: 1, pos: 23
type: A, layer: 1, pos: 39
type: B, layer: 1, pos: 33
type: B, layer: 1, pos: 47

Time for candidate selection: 0.09 seconds

### Candidate
type: A, layer: 1, pos: 11

### Candidate
type: B, layer: 1, pos: 11

### Candidate
type: A, layer: 1, pos: 34

### Candidate
type: B, layer: 1, pos: 34

### Candidate
type: A, layer: 1, pos: 30

## Relational analysis of NS_A1_A2_B1_B2_A2_B2_A2_A1_A1

### Relational analysis result of NS_A1_A2_B1_B2_A2_B2_A2_A1_A1
Status: Status.VERIFIED
Output dim: 3, lower bound: -20.6029467, upper bound: 20.5985222
time: 0.72 seconds

## Relational analysis of NS_A1_A2_B1_B2_A2_B2_A2_A1_A2

### Relational analysis result of NS_A1_A2_B1_B2_A2_B2_A2_A1_A2
Status: Status.VERIFIED
Output dim: 3, lower bound: -20.6029467, upper bound: 20.5985222
time: 0.69 seconds

## BFS NS instance: NS_A1_A2_B1_B2_A2_B2_A2_A2

### Backsubstitution after applying NS history:
0: -3.5432069, 3.1574054, -3.7443836, 3.3308897, -6.8740969, 6.9017882
1: -13.9932346, 12.2254572, -14.8040266, 12.8840399, -26.8772736, 27.0294838
2: -6.9867640, 11.4690914, -7.3696198, 12.1977873, -19.1845512, 18.8387108
3: -12.2230520, 11.1845713, -12.9243069, 11.7908812, -24.0139332, 24.1088791
4: -8.9469519, 11.6595716, -9.4258900, 12.3335075, -21.2804604, 21.0854607

Time for backsubstitution: 0.99 seconds

### NS candidates at layer 1
type: A, layer: 1, pos: 11
type: B, layer: 1, pos: 11
type: A, layer: 1, pos: 34
type: B, layer: 1, pos: 34
type: A, layer: 1, pos: 30
type: B, layer: 1, pos: 12
type: B, layer: 1, pos: 9
type: B, layer: 1, pos: 4
type: A, layer: 1, pos: 4
type: A, layer: 1, pos: 12
type: B, layer: 1, pos: 42
type: A, layer: 1, pos: 16
type: B, layer: 1, pos: 16
type: B, layer: 1, pos: 39
type: B, layer: 1, pos: 23
type: A, layer: 1, pos: 33
type: A, layer: 1, pos: 31
type: B, layer: 1, pos: 31
type: A, layer: 1, pos: 39
type: B, layer: 1, pos: 33
type: A, layer: 1, pos: 23
type: B, layer: 1, pos: 47

Time for candidate selection: 0.08 seconds

### Candidate
type: A, layer: 1, pos: 11

### Candidate
type: B, layer: 1, pos: 11

### Candidate
type: A, layer: 1, pos: 34

### Candidate
type: B, layer: 1, pos: 34

### Candidate
type: A, layer: 1, pos: 30

## Relational analysis of NS_A1_A2_B1_B2_A2_B2_A2_A2_A1

### Relational analysis result of NS_A1_A2_B1_B2_A2_B2_A2_A2_A1
Status: Status.VERIFIED
Output dim: 3, lower bound: -20.6029467, upper bound: 20.5985222
time: 0.85 seconds

## Relational analysis of NS_A1_A2_B1_B2_A2_B2_A2_A2_A2

### Relational analysis result of NS_A1_A2_B1_B2_A2_B2_A2_A2_A2
Status: Status.VERIFIED
Output dim: 3, lower bound: -20.6029467, upper bound: 20.5985222
time: 0.67 seconds

## BFS NS instance: NS_A1_A2_B2_B1_B1_B1_A2_A2

### Backsubstitution after applying NS history:
0: -3.7551029, 3.3418815, -3.4812961, 3.1150770, -6.8701801, 6.8231773
1: -14.8365059, 12.9373684, -13.7059050, 12.0545502, -26.8910561, 26.6432724
2: -7.4026179, 12.1471291, -6.8541398, 11.3458900, -18.7485085, 19.0012665
3: -12.9592552, 11.8153782, -11.9621525, 11.0443764, -24.0036240, 23.7775307
4: -9.4836941, 12.3391991, -8.7754803, 11.5260696, -21.0097637, 21.1146755

Time for backsubstitution: 0.96 seconds

### NS candidates at layer 1
type: A, layer: 1, pos: 11
type: B, layer: 1, pos: 11
type: B, layer: 1, pos: 34
type: A, layer: 1, pos: 34
type: B, layer: 1, pos: 4
type: B, layer: 1, pos: 9
type: A, layer: 1, pos: 4
type: B, layer: 1, pos: 42
type: A, layer: 1, pos: 42
type: B, layer: 1, pos: 12
type: A, layer: 1, pos: 39
type: B, layer: 1, pos: 39
type: B, layer: 1, pos: 16
type: B, layer: 1, pos: 31
type: A, layer: 1, pos: 16
type: A, layer: 1, pos: 31
type: A, layer: 1, pos: 33
type: A, layer: 1, pos: 23
type: B, layer: 1, pos: 33
type: B, layer: 1, pos: 23
type: A, layer: 1, pos: 47
type: A, layer: 1, pos: 30

Time for candidate selection: 0.08 seconds

### Candidate
type: A, layer: 1, pos: 11

### Candidate
type: B, layer: 1, pos: 11

### Candidate
type: B, layer: 1, pos: 34

### Candidate
type: A, layer: 1, pos: 34

### Candidate
type: B, layer: 1, pos: 4

### Candidate
type: B, layer: 1, pos: 9

### Candidate
type: A, layer: 1, pos: 4

### Candidate
type: B, layer: 1, pos: 42

## Relational analysis of NS_A1_A2_B2_B1_B1_B1_A2_A2_B1
Optimization infeasible because this subproblem isn't reachable.

### Candidate
type: A, layer: 1, pos: 42

### Candidate
type: B, layer: 1, pos: 12

## Relational analysis of NS_A1_A2_B2_B1_B1_B1_A2_A2_B1
Optimization infeasible because this subproblem isn't reachable.

### Candidate
type: A, layer: 1, pos: 39

## Relational analysis of NS_A1_A2_B2_B1_B1_B1_A2_A2_A1
Optimization infeasible because this subproblem isn't reachable.

### Candidate
type: B, layer: 1, pos: 39

### Candidate
type: B, layer: 1, pos: 16

## Relational analysis of NS_A1_A2_B2_B1_B1_B1_A2_A2_B1
Optimization infeasible because this subproblem isn't reachable.

### Candidate
type: B, layer: 1, pos: 31

## Relational analysis of NS_A1_A2_B2_B1_B1_B1_A2_A2_B1
Optimization infeasible because this subproblem isn't reachable.

### Candidate
type: A, layer: 1, pos: 16

### Candidate
type: A, layer: 1, pos: 31

### Candidate
type: A, layer: 1, pos: 33

## Relational analysis of NS_A1_A2_B2_B1_B1_B1_A2_A2_A1
Optimization infeasible because this subproblem isn't reachable.

### Candidate
type: A, layer: 1, pos: 23

## Relational analysis of NS_A1_A2_B2_B1_B1_B1_A2_A2_A1

### Relational analysis result of NS_A1_A2_B2_B1_B1_B1_A2_A2_A1
Status: Status.VERIFIED
Output dim: 3, lower bound: -20.6012840, upper bound: 20.5997932
time: 0.67 seconds

## Relational analysis of NS_A1_A2_B2_B1_B1_B1_A2_A2_A2

### Relational analysis result of NS_A1_A2_B2_B1_B1_B1_A2_A2_A2
Status: Status.VERIFIED
Output dim: 3, lower bound: -20.6017844, upper bound: 20.5998665
time: 0.69 seconds

## BFS NS instance: NS_A1_A2_B2_B1_B2_B1_A2_A1

### Backsubstitution after applying NS history:
0: -2.9960527, 2.6623325, -3.2224360, 2.8752885, -5.8713412, 5.8847671
1: -11.8343382, 10.3386192, -12.6998873, 11.1310387, -22.9653759, 23.0385056
2: -5.9199672, 9.6447563, -6.3695655, 10.4611368, -16.3811016, 16.0143223
3: -10.3606873, 9.4523716, -11.1067505, 10.2140846, -20.5747719, 20.5591145
4: -7.5751920, 9.8309212, -8.1406956, 10.6549606, -18.2301521, 17.9716148

Time for backsubstitution: 1.01 seconds

### NS candidates at layer 1
type: A, layer: 1, pos: 11
type: B, layer: 1, pos: 11
type: B, layer: 1, pos: 34
type: A, layer: 1, pos: 34
type: B, layer: 1, pos: 12
type: A, layer: 1, pos: 12
type: B, layer: 1, pos: 9
type: B, layer: 1, pos: 24
type: A, layer: 1, pos: 31
type: B, layer: 1, pos: 31
type: B, layer: 1, pos: 42
type: A, layer: 1, pos: 39
type: B, layer: 1, pos: 16
type: B, layer: 1, pos: 33
type: B, layer: 1, pos: 39
type: A, layer: 1, pos: 16
type: A, layer: 1, pos: 4
type: B, layer: 1, pos: 23
type: A, layer: 1, pos: 23
type: A, layer: 1, pos: 33
type: A, layer: 1, pos: 47
type: A, layer: 1, pos: 30

Time for candidate selection: 0.08 seconds

### Candidate
type: A, layer: 1, pos: 11

### Candidate
type: B, layer: 1, pos: 11

### Candidate
type: B, layer: 1, pos: 34

### Candidate
type: A, layer: 1, pos: 34

### Candidate
type: B, layer: 1, pos: 12

### Candidate
type: A, layer: 1, pos: 12

### Candidate
type: B, layer: 1, pos: 9

## Relational analysis of NS_A1_A2_B2_B1_B2_B1_A2_A1_B1
Optimization infeasible because this subproblem isn't reachable.

### Candidate
type: B, layer: 1, pos: 24

## Relational analysis of NS_A1_A2_B2_B1_B2_B1_A2_A1_B1

### Relational analysis result of NS_A1_A2_B2_B1_B2_B1_A2_A1_B1
Status: Status.UNKNOWN
Output dim: 3, lower bound: -20.6035437, upper bound: 20.6026866
time: 0.60 seconds

## Relational analysis of NS_A1_A2_B2_B1_B2_B1_A2_A1_B2

### Relational analysis result of NS_A1_A2_B2_B1_B2_B1_A2_A1_B2
Status: Status.UNKNOWN
Output dim: 3, lower bound: -20.6035437, upper bound: 20.6026866
time: 0.72 seconds

## BFS NS instance: NS_A1_A2_B2_B1_B2_B1_A2_A2

### Backsubstitution after applying NS history:
0: -3.5902476, 3.1974847, -3.2224360, 2.8752885, -6.4655361, 6.4199195
1: -14.1819506, 12.3807278, -12.6998873, 11.1310387, -25.3129883, 25.0806141
2: -7.0806346, 11.6148405, -6.3695655, 10.4611368, -17.5417690, 17.9844055
3: -12.3865166, 11.3239698, -11.1067505, 10.2140846, -22.6005993, 22.4307194
4: -9.0666447, 11.8067245, -8.1406956, 10.6549606, -19.7216034, 19.9474163

Time for backsubstitution: 1.14 seconds

### NS candidates at layer 1
type: B, layer: 1, pos: 11
type: A, layer: 1, pos: 11
type: B, layer: 1, pos: 34
type: A, layer: 1, pos: 34
type: A, layer: 1, pos: 12
type: B, layer: 1, pos: 12
type: B, layer: 1, pos: 9
type: A, layer: 1, pos: 31
type: B, layer: 1, pos: 31
type: B, layer: 1, pos: 42
type: A, layer: 1, pos: 39
type: B, layer: 1, pos: 16
type: B, layer: 1, pos: 24
type: B, layer: 1, pos: 33
type: B, layer: 1, pos: 39
type: A, layer: 1, pos: 16
type: A, layer: 1, pos: 4
type: A, layer: 1, pos: 23
type: B, layer: 1, pos: 23
type: A, layer: 1, pos: 33
type: A, layer: 1, pos: 47
type: A, layer: 1, pos: 30

Time for candidate selection: 0.08 seconds

### Candidate
type: B, layer: 1, pos: 11

### Candidate
type: A, layer: 1, pos: 11

### Candidate
type: B, layer: 1, pos: 34

### Candidate
type: A, layer: 1, pos: 34

### Candidate
type: A, layer: 1, pos: 12

### Candidate
type: B, layer: 1, pos: 12

### Candidate
type: B, layer: 1, pos: 9

## Relational analysis of NS_A1_A2_B2_B1_B2_B1_A2_A2_B1
Optimization infeasible because this subproblem isn't reachable.

### Candidate
type: A, layer: 1, pos: 31

## Relational analysis of NS_A1_A2_B2_B1_B2_B1_A2_A2_A1

### Relational analysis result of NS_A1_A2_B2_B1_B2_B1_A2_A2_A1
Status: Status.UNKNOWN
Output dim: 3, lower bound: -20.6035437, upper bound: 20.6024183
time: 0.63 seconds

## Relational analysis of NS_A1_A2_B2_B1_B2_B1_A2_A2_A2

### Relational analysis result of NS_A1_A2_B2_B1_B2_B1_A2_A2_A2
Status: Status.UNKNOWN
Output dim: 3, lower bound: -20.6033286, upper bound: 20.6024620
time: 0.69 seconds

## BFS NS instance: NS_A1_A2_B2_B1_B2_B2_A2_B1

### Backsubstitution after applying NS history:
0: -3.4072533, 2.9987469, -3.6223240, 3.2366309, -6.6438842, 6.6210694
1: -13.4742098, 11.6417627, -14.2748737, 12.5350981, -26.0093079, 25.9166374
2: -6.7381449, 10.8741417, -7.1335411, 11.7950106, -18.5331554, 18.0076828
3: -11.7894468, 10.6054029, -12.4678802, 11.4905529, -23.2799988, 23.0732784
4: -8.6332855, 11.0345154, -9.1481209, 11.9893732, -20.6226578, 20.1826363

Time for backsubstitution: 0.99 seconds

### NS candidates at layer 1
type: A, layer: 1, pos: 11
type: B, layer: 1, pos: 34
type: B, layer: 1, pos: 11
type: B, layer: 1, pos: 12
type: A, layer: 1, pos: 34
type: A, layer: 1, pos: 22
type: B, layer: 1, pos: 9
type: A, layer: 1, pos: 12
type: A, layer: 1, pos: 31
type: B, layer: 1, pos: 31
type: B, layer: 1, pos: 42
type: A, layer: 1, pos: 39
type: B, layer: 1, pos: 39
type: B, layer: 1, pos: 16
type: A, layer: 1, pos: 23
type: B, layer: 1, pos: 23
type: A, layer: 1, pos: 16
type: B, layer: 1, pos: 33
type: A, layer: 1, pos: 4
type: A, layer: 1, pos: 33
type: A, layer: 1, pos: 47
type: A, layer: 1, pos: 30

Time for candidate selection: 0.08 seconds

### Candidate
type: A, layer: 1, pos: 11

### Candidate
type: B, layer: 1, pos: 34

### Candidate
type: B, layer: 1, pos: 11

### Candidate
type: B, layer: 1, pos: 12

### Candidate
type: A, layer: 1, pos: 34

### Candidate
type: A, layer: 1, pos: 22

## Relational analysis of NS_A1_A2_B2_B1_B2_B2_A2_B1_A1

### Relational analysis result of NS_A1_A2_B2_B1_B2_B2_A2_B1_A1
Status: Status.UNKNOWN
Output dim: 3, lower bound: -20.6035433, upper bound: 20.6002504
time: 0.76 seconds

## Relational analysis of NS_A1_A2_B2_B1_B2_B2_A2_B1_A2

### Relational analysis result of NS_A1_A2_B2_B1_B2_B2_A2_B1_A2
Status: Status.UNKNOWN
Output dim: 3, lower bound: -20.6035433, upper bound: 20.6002210
time: 0.71 seconds

## BFS NS instance: NS_A1_A2_B2_B1_B2_B2_A2_B2

### Backsubstitution after applying NS history:
0: -3.4072533, 2.9987469, -4.1345482, 3.7068682, -7.1141214, 7.1332946
1: -13.4742098, 11.6417627, -16.2946053, 14.3344345, -27.8086433, 27.9363670
2: -6.7381449, 10.8741417, -8.1139383, 13.6177435, -20.3558884, 18.9880791
3: -11.7894468, 10.6054029, -14.1805086, 13.1594715, -24.9489174, 24.7859097
4: -8.6332855, 11.0345154, -10.3973160, 13.7970152, -22.4302998, 21.4318295

Time for backsubstitution: 1.06 seconds

### NS candidates at layer 1
type: A, layer: 1, pos: 11
type: B, layer: 1, pos: 34
type: A, layer: 1, pos: 34
type: B, layer: 1, pos: 11
type: B, layer: 1, pos: 12
type: A, layer: 1, pos: 22
type: B, layer: 1, pos: 9
type: A, layer: 1, pos: 47
type: B, layer: 1, pos: 31
type: A, layer: 1, pos: 12
type: A, layer: 1, pos: 31
type: B, layer: 1, pos: 42
type: B, layer: 1, pos: 39
type: A, layer: 1, pos: 4
type: B, layer: 1, pos: 16
type: A, layer: 1, pos: 16
type: B, layer: 1, pos: 33
type: A, layer: 1, pos: 33
type: A, layer: 1, pos: 39
type: B, layer: 1, pos: 23
type: A, layer: 1, pos: 23
type: A, layer: 1, pos: 30

Time for candidate selection: 0.10 seconds

### Candidate
type: A, layer: 1, pos: 11

### Candidate
type: B, layer: 1, pos: 34

### Candidate
type: A, layer: 1, pos: 34

### Candidate
type: B, layer: 1, pos: 11

### Candidate
type: B, layer: 1, pos: 12

### Candidate
type: A, layer: 1, pos: 22

## Relational analysis of NS_A1_A2_B2_B1_B2_B2_A2_B2_A1

### Relational analysis result of NS_A1_A2_B2_B1_B2_B2_A2_B2_A1
Status: Status.UNKNOWN
Output dim: 3, lower bound: -20.6035433, upper bound: 20.6002504
time: 0.94 seconds

## Relational analysis of NS_A1_A2_B2_B1_B2_B2_A2_B2_A2

### Relational analysis result of NS_A1_A2_B2_B1_B2_B2_A2_B2_A2
Status: Status.UNKNOWN
Output dim: 3, lower bound: -20.6035433, upper bound: 20.6002210
time: 0.83 seconds

## BFS NS instance: NS_A1_A2_B2_B2_A2_B1_A1_A1

### Backsubstitution after applying NS history:
0: -2.9575121, 2.6303887, -3.8540320, 3.4321170, -6.3896294, 6.4844208
1: -11.6835804, 10.1886177, -15.2264605, 13.2877932, -24.9713707, 25.4150772
2: -5.8330021, 9.5619545, -7.5972466, 12.4870577, -18.3200607, 17.1592007
3: -10.2288876, 9.3145103, -13.2969398, 12.1337633, -22.3626518, 22.6114483
4: -7.4482050, 9.7160215, -9.7240124, 12.6810350, -20.1292343, 19.4400311

Time for backsubstitution: 1.00 seconds

### NS candidates at layer 1
type: A, layer: 1, pos: 11
type: B, layer: 1, pos: 11
type: B, layer: 1, pos: 34
type: A, layer: 1, pos: 34
type: A, layer: 1, pos: 31
type: B, layer: 1, pos: 31
type: B, layer: 1, pos: 12
type: A, layer: 1, pos: 12
type: B, layer: 1, pos: 9
type: B, layer: 1, pos: 39
type: B, layer: 1, pos: 4
type: B, layer: 1, pos: 42
type: A, layer: 1, pos: 4
type: A, layer: 1, pos: 39
type: B, layer: 1, pos: 16
type: B, layer: 1, pos: 23
type: A, layer: 1, pos: 16
type: A, layer: 1, pos: 23
type: A, layer: 1, pos: 33
type: B, layer: 1, pos: 33
type: A, layer: 1, pos: 30
type: B, layer: 1, pos: 47

Time for candidate selection: 0.08 seconds

### Candidate
type: A, layer: 1, pos: 11

### Candidate
type: B, layer: 1, pos: 11

### Candidate
type: B, layer: 1, pos: 34

### Candidate
type: A, layer: 1, pos: 34

### Candidate
type: A, layer: 1, pos: 31

## Relational analysis of NS_A1_A2_B2_B2_A2_B1_A1_A1_A1
Optimization infeasible because this subproblem isn't reachable.

### Candidate
type: B, layer: 1, pos: 31

### Candidate
type: B, layer: 1, pos: 12

### Candidate
type: A, layer: 1, pos: 12

### Candidate
type: B, layer: 1, pos: 9

## Relational analysis of NS_A1_A2_B2_B2_A2_B1_A1_A1_B1
Optimization infeasible because this subproblem isn't reachable.

### Candidate
type: B, layer: 1, pos: 39

## Relational analysis of NS_A1_A2_B2_B2_A2_B1_A1_A1_B1
Optimization infeasible because this subproblem isn't reachable.

### Candidate
type: B, layer: 1, pos: 4

## Relational analysis of NS_A1_A2_B2_B2_A2_B1_A1_A1_B1
Optimization infeasible because this subproblem isn't reachable.

### Candidate
type: B, layer: 1, pos: 42

## Relational analysis of NS_A1_A2_B2_B2_A2_B1_A1_A1_B1

### Relational analysis result of NS_A1_A2_B2_B2_A2_B1_A1_A1_B1
Status: Status.VERIFIED
Output dim: 3, lower bound: -20.5966054, upper bound: 20.6008455
time: 0.69 seconds

## Relational analysis of NS_A1_A2_B2_B2_A2_B1_A1_A1_B2

### Relational analysis result of NS_A1_A2_B2_B2_A2_B1_A1_A1_B2
Status: Status.VERIFIED
Output dim: 3, lower bound: -20.5966054, upper bound: 20.6008455
time: 0.70 seconds

## BFS NS instance: NS_A1_A2_B2_B2_A2_B1_A1_A2

### Backsubstitution after applying NS history:
0: -2.9541757, 2.6261964, -3.8540320, 3.4321170, -6.3862925, 6.4802284
1: -11.6670923, 10.1961460, -15.2264605, 13.2877932, -24.9548836, 25.4226036
2: -5.8364959, 9.5153542, -7.5972466, 12.4870577, -18.3235531, 17.1126003
3: -10.2159939, 9.3246737, -13.2969398, 12.1337633, -22.3497562, 22.6216125
4: -7.4689107, 9.7003708, -9.7240124, 12.6810350, -20.1499462, 19.4243813

Time for backsubstitution: 1.30 seconds

### NS candidates at layer 1
type: A, layer: 1, pos: 11
type: B, layer: 1, pos: 11
type: B, layer: 1, pos: 34
type: A, layer: 1, pos: 34
type: B, layer: 1, pos: 12
type: B, layer: 1, pos: 9
type: A, layer: 1, pos: 12
type: A, layer: 1, pos: 30
type: B, layer: 1, pos: 31
type: A, layer: 1, pos: 31
type: B, layer: 1, pos: 4
type: B, layer: 1, pos: 42
type: A, layer: 1, pos: 4
type: B, layer: 1, pos: 39
type: B, layer: 1, pos: 16
type: B, layer: 1, pos: 23
type: A, layer: 1, pos: 16
type: A, layer: 1, pos: 33
type: A, layer: 1, pos: 23
type: A, layer: 1, pos: 39
type: B, layer: 1, pos: 33
type: B, layer: 1, pos: 47

Time for candidate selection: 0.10 seconds

### Candidate
type: A, layer: 1, pos: 11

### Candidate
type: B, layer: 1, pos: 11

### Candidate
type: B, layer: 1, pos: 34

### Candidate
type: A, layer: 1, pos: 34

### Candidate
type: B, layer: 1, pos: 12

### Candidate
type: B, layer: 1, pos: 9

## Relational analysis of NS_A1_A2_B2_B2_A2_B1_A1_A2_B1
Optimization infeasible because this subproblem isn't reachable.

### Candidate
type: A, layer: 1, pos: 12

### Candidate
type: A, layer: 1, pos: 30

## Relational analysis of NS_A1_A2_B2_B2_A2_B1_A1_A2_A1

### Relational analysis result of NS_A1_A2_B2_B2_A2_B1_A1_A2_A1
Status: Status.UNKNOWN
Output dim: 3, lower bound: -20.6037272, upper bound: 20.6041808
time: 0.66 seconds

## Relational analysis of NS_A1_A2_B2_B2_A2_B1_A1_A2_A2

### Relational analysis result of NS_A1_A2_B2_B2_A2_B1_A1_A2_A2
Status: Status.UNKNOWN
Output dim: 3, lower bound: -20.6037272, upper bound: 20.6041081
time: 0.64 seconds

## BFS NS instance: NS_A1_A2_B2_B2_A2_B1_A2_B1

### Backsubstitution after applying NS history:
0: -3.5902476, 3.1974847, -3.3638229, 2.9935117, -6.5837593, 6.5613079
1: -14.1819506, 12.3807278, -13.2885780, 11.5806503, -25.7625980, 25.6693039
2: -7.0806346, 11.6148405, -6.6508455, 10.8754978, -17.9561329, 18.2656860
3: -12.3865166, 11.3239698, -11.6201696, 10.5826540, -22.9691639, 22.9441395
4: -9.0666447, 11.8067245, -8.4928770, 11.0615835, -20.1282272, 20.2996006

Time for backsubstitution: 1.02 seconds

### NS candidates at layer 1
type: B, layer: 1, pos: 11
type: A, layer: 1, pos: 11
type: A, layer: 1, pos: 34
type: B, layer: 1, pos: 34
type: B, layer: 1, pos: 47
type: A, layer: 1, pos: 12
type: B, layer: 1, pos: 12
type: A, layer: 1, pos: 47
type: A, layer: 1, pos: 30
type: B, layer: 1, pos: 9
type: B, layer: 1, pos: 31
type: A, layer: 1, pos: 31
type: B, layer: 1, pos: 42
type: A, layer: 1, pos: 4
type: B, layer: 1, pos: 16
type: A, layer: 1, pos: 16
type: A, layer: 1, pos: 23
type: A, layer: 1, pos: 39
type: B, layer: 1, pos: 33
type: B, layer: 1, pos: 23
type: A, layer: 1, pos: 33
type: B, layer: 1, pos: 39

Time for candidate selection: 0.08 seconds

### Candidate
type: B, layer: 1, pos: 11

### Candidate
type: A, layer: 1, pos: 11

### Candidate
type: A, layer: 1, pos: 34

### Candidate
type: B, layer: 1, pos: 34

### Candidate
type: B, layer: 1, pos: 47

## Relational analysis of NS_A1_A2_B2_B2_A2_B1_A2_B1_B1

### Relational analysis result of NS_A1_A2_B2_B2_A2_B1_A2_B1_B1
Status: Status.UNKNOWN
Output dim: 3, lower bound: -20.6033417, upper bound: 20.6001261
time: 0.63 seconds

## Relational analysis of NS_A1_A2_B2_B2_A2_B1_A2_B1_B2

### Relational analysis result of NS_A1_A2_B2_B2_A2_B1_A2_B1_B2
Status: Status.UNKNOWN
Output dim: 3, lower bound: -20.6037400, upper bound: 20.6032333
time: 0.75 seconds

## BFS NS instance: NS_A1_A2_B2_B2_A2_B1_A2_B2

### Backsubstitution after applying NS history:
0: -3.5902476, 3.1974847, -3.7804952, 3.3682485, -6.9584961, 6.9779787
1: -14.1819506, 12.3807278, -14.9342661, 13.0380468, -27.2199955, 27.3149948
2: -7.0806346, 11.6148405, -7.4507456, 12.2512989, -19.3319340, 19.0655861
3: -12.3865166, 11.3239698, -13.0427217, 11.9087887, -24.2953033, 24.3666897
4: -9.0666447, 11.8067245, -9.5405416, 12.4435196, -21.5101643, 21.3472652

Time for backsubstitution: 0.99 seconds

### NS candidates at layer 1
type: A, layer: 1, pos: 11
type: B, layer: 1, pos: 11
type: B, layer: 1, pos: 34
type: A, layer: 1, pos: 34
type: B, layer: 1, pos: 12
type: A, layer: 1, pos: 12
type: B, layer: 1, pos: 9
type: B, layer: 1, pos: 42
type: A, layer: 1, pos: 4
type: A, layer: 1, pos: 30
type: A, layer: 1, pos: 47
type: B, layer: 1, pos: 39
type: A, layer: 1, pos: 39
type: B, layer: 1, pos: 16
type: A, layer: 1, pos: 16
type: A, layer: 1, pos: 33
type: B, layer: 1, pos: 33
type: A, layer: 1, pos: 31
type: B, layer: 1, pos: 31
type: B, layer: 1, pos: 23
type: A, layer: 1, pos: 23
type: B, layer: 1, pos: 47

Time for candidate selection: 0.10 seconds

### Candidate
type: A, layer: 1, pos: 11

### Candidate
type: B, layer: 1, pos: 11

### Candidate
type: B, layer: 1, pos: 34

### Candidate
type: A, layer: 1, pos: 34

### Candidate
type: B, layer: 1, pos: 12

### Candidate
type: A, layer: 1, pos: 12

### Candidate
type: B, layer: 1, pos: 9

## Relational analysis of NS_A1_A2_B2_B2_A2_B1_A2_B2_B1
Optimization infeasible because this subproblem isn't reachable.

### Candidate
type: B, layer: 1, pos: 42

## Relational analysis of NS_A1_A2_B2_B2_A2_B1_A2_B2_B1

### Relational analysis result of NS_A1_A2_B2_B2_A2_B1_A2_B2_B1
Status: Status.UNKNOWN
Output dim: 3, lower bound: -20.6039649, upper bound: 20.6016971
time: 0.80 seconds

## Relational analysis of NS_A1_A2_B2_B2_A2_B1_A2_B2_B2

### Relational analysis result of NS_A1_A2_B2_B2_A2_B1_A2_B2_B2
Status: Status.UNKNOWN
Output dim: 3, lower bound: -20.6039649, upper bound: 20.6041258
time: 0.79 seconds

## BFS NS instance: NS_A1_A2_B2_B2_A2_B2_A2_A1

### Backsubstitution after applying NS history:
0: -2.9541757, 2.6261964, -4.4449782, 3.9616370, -6.9158125, 7.0711746
1: -11.6670923, 10.1961460, -17.5713806, 15.3260794, -26.9931698, 27.7675266
2: -5.8364959, 9.5153542, -8.7266226, 14.5362911, -20.3727875, 18.2419777
3: -10.2159939, 9.3246737, -15.3107166, 14.0289297, -24.2449226, 24.6353893
4: -7.4689107, 9.7003708, -11.1794167, 14.6911469, -22.1600571, 20.8797874

Time for backsubstitution: 1.03 seconds

### NS candidates at layer 1
type: A, layer: 1, pos: 11
type: A, layer: 1, pos: 34
type: B, layer: 1, pos: 34
type: A, layer: 1, pos: 30
type: B, layer: 1, pos: 11
type: B, layer: 1, pos: 9
type: B, layer: 1, pos: 12
type: B, layer: 1, pos: 31
type: A, layer: 1, pos: 12
type: A, layer: 1, pos: 31
type: B, layer: 1, pos: 4
type: B, layer: 1, pos: 39
type: A, layer: 1, pos: 4
type: B, layer: 1, pos: 42
type: A, layer: 1, pos: 33
type: B, layer: 1, pos: 23
type: A, layer: 1, pos: 16
type: B, layer: 1, pos: 16
type: A, layer: 1, pos: 23
type: B, layer: 1, pos: 33
type: A, layer: 1, pos: 39
type: B, layer: 1, pos: 47

Time for candidate selection: 0.08 seconds

### Candidate
type: A, layer: 1, pos: 11

### Candidate
type: A, layer: 1, pos: 34

### Candidate
type: B, layer: 1, pos: 34

### Candidate
type: A, layer: 1, pos: 30

## Relational analysis of NS_A1_A2_B2_B2_A2_B2_A2_A1_A1

### Relational analysis result of NS_A1_A2_B2_B2_A2_B2_A2_A1_A1
Status: Status.VERIFIED
Output dim: 3, lower bound: -20.6029467, upper bound: 20.6022699
time: 0.95 seconds

## Relational analysis of NS_A1_A2_B2_B2_A2_B2_A2_A1_A2

### Relational analysis result of NS_A1_A2_B2_B2_A2_B2_A2_A1_A2
Status: Status.VERIFIED
Output dim: 3, lower bound: -20.6029467, upper bound: 20.6022699
time: 0.88 seconds

## BFS NS instance: NS_A1_A2_B2_B2_A2_B2_A2_A2

### Backsubstitution after applying NS history:
0: -3.5432069, 3.1574054, -4.4449782, 3.9616370, -7.5048437, 7.6023831
1: -13.9932346, 12.2254572, -17.5713806, 15.3260794, -29.3193130, 29.7968369
2: -6.9867640, 11.4690914, -8.7266226, 14.5362911, -21.5230541, 20.1957130
3: -12.2230520, 11.1845713, -15.3107166, 14.0289297, -26.2519817, 26.4952888
4: -8.9469519, 11.6595716, -11.1794167, 14.6911469, -23.6380978, 22.8389874

Time for backsubstitution: 1.01 seconds

### NS candidates at layer 1
type: A, layer: 1, pos: 11
type: B, layer: 1, pos: 11
type: A, layer: 1, pos: 34
type: B, layer: 1, pos: 34
type: B, layer: 1, pos: 9
type: B, layer: 1, pos: 12
type: A, layer: 1, pos: 30
type: B, layer: 1, pos: 4
type: A, layer: 1, pos: 4
type: A, layer: 1, pos: 12
type: B, layer: 1, pos: 42
type: A, layer: 1, pos: 16
type: B, layer: 1, pos: 16
type: B, layer: 1, pos: 39
type: A, layer: 1, pos: 33
type: A, layer: 1, pos: 31
type: B, layer: 1, pos: 31
type: B, layer: 1, pos: 23
type: A, layer: 1, pos: 23
type: B, layer: 1, pos: 33
type: A, layer: 1, pos: 39
type: B, layer: 1, pos: 47

Time for candidate selection: 0.08 seconds

### Candidate
type: A, layer: 1, pos: 11

### Candidate
type: B, layer: 1, pos: 11

### Candidate
type: A, layer: 1, pos: 34

### Candidate
type: B, layer: 1, pos: 34

### Candidate
type: B, layer: 1, pos: 9

## Relational analysis of NS_A1_A2_B2_B2_A2_B2_A2_A2_B1
Optimization infeasible because this subproblem isn't reachable.

### Candidate
type: B, layer: 1, pos: 12

## Relational analysis of NS_A1_A2_B2_B2_A2_B2_A2_A2_B1
Optimization infeasible because this subproblem isn't reachable.

### Candidate
type: A, layer: 1, pos: 30

## Relational analysis of NS_A1_A2_B2_B2_A2_B2_A2_A2_A1

### Relational analysis result of NS_A1_A2_B2_B2_A2_B2_A2_A2_A1
Status: Status.VERIFIED
Output dim: 3, lower bound: -20.6029467, upper bound: 20.6020629
time: 0.70 seconds

## Relational analysis of NS_A1_A2_B2_B2_A2_B2_A2_A2_A2

### Relational analysis result of NS_A1_A2_B2_B2_A2_B2_A2_A2_A2
Status: Status.VERIFIED
Output dim: 3, lower bound: -20.6029467, upper bound: 20.6020629
time: 0.79 seconds

## Summary of splitting at layer (split count: 8)
- Time for NS candidates: 3.18 seconds
NS_A1_A2_B1_B1_B1_B1_A1_A2_A1, status: Status.UNKNOWN, split count: 9, time: 3.18
Output dim: 3, lower bound: -20.6033197, upper bound: 20.5972793
NS_A1_A2_B1_B1_B1_B1_A1_A2_A2, status: Status.VERIFIED, split count: 9, time: 3.18
Output dim: 3, lower bound: -20.6030875, upper bound: 20.5966088
NS_A1_A2_B1_B1_B1_B1_A2_A2_A1, status: Status.UNKNOWN, split count: 9, time: 3.18
Output dim: 3, lower bound: -20.6033073, upper bound: 20.5972557
NS_A1_A2_B1_B1_B1_B1_A2_A2_A2, status: Status.UNKNOWN, split count: 9, time: 3.18
Output dim: 3, lower bound: -20.6031824, upper bound: 20.5972723
NS_A1_A2_B1_B1_B2_B1_A2_A1_A1, status: Status.UNKNOWN, split count: 9, time: 3.18
Output dim: 3, lower bound: -20.6035437, upper bound: 20.5994695
NS_A1_A2_B1_B1_B2_B1_A2_A1_A2, status: Status.UNKNOWN, split count: 9, time: 3.18
Output dim: 3, lower bound: -20.6033286, upper bound: 20.5994820
NS_A1_A2_B1_B1_B2_B1_A2_A2_B1, status: Status.UNKNOWN, split count: 9, time: 3.18
Output dim: 3, lower bound: -20.6033752, upper bound: 20.5990013
NS_A1_A2_B1_B1_B2_B1_A2_A2_B2, status: Status.UNKNOWN, split count: 9, time: 3.18
Output dim: 3, lower bound: -20.6034812, upper bound: 20.5994820
NS_A1_A2_B1_B1_B2_B2_A1_B1_A1, status: Status.UNKNOWN, split count: 9, time: 3.18
Output dim: 3, lower bound: -20.6034951, upper bound: 20.5987226
NS_A1_A2_B1_B1_B2_B2_A1_B1_A2, status: Status.UNKNOWN, split count: 9, time: 3.18
Output dim: 3, lower bound: -20.6032505, upper bound: 20.5987197
NS_A1_A2_B1_B1_B2_B2_A1_B2_A1, status: Status.UNKNOWN, split count: 9, time: 3.18
Output dim: 3, lower bound: -20.6035729, upper bound: 20.5993572
NS_A1_A2_B1_B1_B2_B2_A1_B2_A2, status: Status.UNKNOWN, split count: 9, time: 3.18
Output dim: 3, lower bound: -20.6033283, upper bound: 20.5993543
NS_A1_A2_B1_B1_B2_B2_A2_A2_B1, status: Status.UNKNOWN, split count: 9, time: 3.18
Output dim: 3, lower bound: -20.6034656, upper bound: 20.5987112
NS_A1_A2_B1_B1_B2_B2_A2_A2_B2, status: Status.UNKNOWN, split count: 9, time: 3.18
Output dim: 3, lower bound: -20.6034656, upper bound: 20.5993457
NS_A1_A2_B1_B2_A2_B1_A1_A1_B1, status: Status.VERIFIED, split count: 9, time: 3.18
Output dim: 3, lower bound: -20.5959150, upper bound: 20.5959150
NS_A1_A2_B1_B2_A2_B1_A1_A1_B2, status: Status.UNKNOWN, split count: 9, time: 3.18
Output dim: 3, lower bound: -20.5959150, upper bound: 20.6033041
NS_A1_A2_B1_B2_A2_B1_A1_A2_B1, status: Status.UNKNOWN, split count: 9, time: 3.18
Output dim: 3, lower bound: -20.6039454, upper bound: 20.6041886
NS_A1_A2_B1_B2_A2_B1_A1_A2_B2, status: Status.UNKNOWN, split count: 9, time: 3.18
Output dim: 3, lower bound: -20.6039453, upper bound: 20.6039453
NS_A1_A2_B1_B2_A2_B1_A2_B1_B1, status: Status.UNKNOWN, split count: 9, time: 3.18
Output dim: 3, lower bound: -20.6032393, upper bound: 20.5964160
NS_A1_A2_B1_B2_A2_B1_A2_B1_B2, status: Status.UNKNOWN, split count: 9, time: 3.18
Output dim: 3, lower bound: -20.6037242, upper bound: 20.6002121
NS_A1_A2_B1_B2_A2_B1_A2_B2_A1, status: Status.UNKNOWN, split count: 9, time: 3.18
Output dim: 3, lower bound: -20.6037382, upper bound: 20.6041557
NS_A1_A2_B1_B2_A2_B1_A2_B2_A2, status: Status.UNKNOWN, split count: 9, time: 3.18
Output dim: 3, lower bound: -20.6037382, upper bound: 20.6040943
NS_A1_A2_B1_B2_A2_B2_A2_A1_A1, status: Status.VERIFIED, split count: 9, time: 3.18
Output dim: 3, lower bound: -20.6029467, upper bound: 20.5985222
NS_A1_A2_B1_B2_A2_B2_A2_A1_A2, status: Status.VERIFIED, split count: 9, time: 3.18
Output dim: 3, lower bound: -20.6029467, upper bound: 20.5985222
NS_A1_A2_B1_B2_A2_B2_A2_A2_A1, status: Status.VERIFIED, split count: 9, time: 3.18
Output dim: 3, lower bound: -20.6029467, upper bound: 20.5985222
NS_A1_A2_B1_B2_A2_B2_A2_A2_A2, status: Status.VERIFIED, split count: 9, time: 3.18
Output dim: 3, lower bound: -20.6029467, upper bound: 20.5985222
NS_A1_A2_B2_B1_B1_B1_A2_A2_A1, status: Status.VERIFIED, split count: 9, time: 3.18
Output dim: 3, lower bound: -20.6012840, upper bound: 20.5997932
NS_A1_A2_B2_B1_B1_B1_A2_A2_A2, status: Status.VERIFIED, split count: 9, time: 3.18
Output dim: 3, lower bound: -20.6017844, upper bound: 20.5998665
NS_A1_A2_B2_B1_B2_B1_A2_A1_B1, status: Status.UNKNOWN, split count: 9, time: 3.18
Output dim: 3, lower bound: -20.6035437, upper bound: 20.6026866
NS_A1_A2_B2_B1_B2_B1_A2_A1_B2, status: Status.UNKNOWN, split count: 9, time: 3.18
Output dim: 3, lower bound: -20.6035437, upper bound: 20.6026866
NS_A1_A2_B2_B1_B2_B1_A2_A2_A1, status: Status.UNKNOWN, split count: 9, time: 3.18
Output dim: 3, lower bound: -20.6035437, upper bound: 20.6024183
NS_A1_A2_B2_B1_B2_B1_A2_A2_A2, status: Status.UNKNOWN, split count: 9, time: 3.18
Output dim: 3, lower bound: -20.6033286, upper bound: 20.6024620
NS_A1_A2_B2_B1_B2_B2_A2_B1_A1, status: Status.UNKNOWN, split count: 9, time: 3.18
Output dim: 3, lower bound: -20.6035433, upper bound: 20.6002504
NS_A1_A2_B2_B1_B2_B2_A2_B1_A2, status: Status.UNKNOWN, split count: 9, time: 3.18
Output dim: 3, lower bound: -20.6035433, upper bound: 20.6002210
NS_A1_A2_B2_B1_B2_B2_A2_B2_A1, status: Status.UNKNOWN, split count: 9, time: 3.18
Output dim: 3, lower bound: -20.6035433, upper bound: 20.6002504
NS_A1_A2_B2_B1_B2_B2_A2_B2_A2, status: Status.UNKNOWN, split count: 9, time: 3.18
Output dim: 3, lower bound: -20.6035433, upper bound: 20.6002210
NS_A1_A2_B2_B2_A2_B1_A1_A1_B1, status: Status.VERIFIED, split count: 9, time: 3.18
Output dim: 3, lower bound: -20.5966054, upper bound: 20.6008455
NS_A1_A2_B2_B2_A2_B1_A1_A1_B2, status: Status.VERIFIED, split count: 9, time: 3.18
Output dim: 3, lower bound: -20.5966054, upper bound: 20.6008455
NS_A1_A2_B2_B2_A2_B1_A1_A2_A1, status: Status.UNKNOWN, split count: 9, time: 3.18
Output dim: 3, lower bound: -20.6037272, upper bound: 20.6041808
NS_A1_A2_B2_B2_A2_B1_A1_A2_A2, status: Status.UNKNOWN, split count: 9, time: 3.18
Output dim: 3, lower bound: -20.6037272, upper bound: 20.6041081
NS_A1_A2_B2_B2_A2_B1_A2_B1_B1, status: Status.UNKNOWN, split count: 9, time: 3.18
Output dim: 3, lower bound: -20.6033417, upper bound: 20.6001261
NS_A1_A2_B2_B2_A2_B1_A2_B1_B2, status: Status.UNKNOWN, split count: 9, time: 3.18
Output dim: 3, lower bound: -20.6037400, upper bound: 20.6032333
NS_A1_A2_B2_B2_A2_B1_A2_B2_B1, status: Status.UNKNOWN, split count: 9, time: 3.18
Output dim: 3, lower bound: -20.6039649, upper bound: 20.6016971
NS_A1_A2_B2_B2_A2_B1_A2_B2_B2, status: Status.UNKNOWN, split count: 9, time: 3.18
Output dim: 3, lower bound: -20.6039649, upper bound: 20.6041258
NS_A1_A2_B2_B2_A2_B2_A2_A1_A1, status: Status.VERIFIED, split count: 9, time: 3.18
Output dim: 3, lower bound: -20.6029467, upper bound: 20.6022699
NS_A1_A2_B2_B2_A2_B2_A2_A1_A2, status: Status.VERIFIED, split count: 9, time: 3.18
Output dim: 3, lower bound: -20.6029467, upper bound: 20.6022699
NS_A1_A2_B2_B2_A2_B2_A2_A2_A1, status: Status.VERIFIED, split count: 9, time: 3.18
Output dim: 3, lower bound: -20.6029467, upper bound: 20.6020629
NS_A1_A2_B2_B2_A2_B2_A2_A2_A2, status: Status.VERIFIED, split count: 9, time: 3.18
Output dim: 3, lower bound: -20.6029467, upper bound: 20.6020629

## BFS NS instance: NS_A1_A2_B1_B1_B1_B1_A1_A2_A1

### Backsubstitution after applying NS history:
0: -2.8579593, 2.5493798, -2.8369715, 2.5377181, -5.3956771, 5.3863511
1: -11.2709818, 9.8977585, -11.1648302, 9.8173466, -21.0883274, 21.0625877
2: -5.6504035, 9.2356186, -5.6087904, 9.2279978, -14.8783998, 14.8444071
3: -9.8734846, 9.0577898, -9.7681313, 9.0039520, -18.8774376, 18.8259201
4: -7.2260165, 9.4269829, -7.1507564, 9.4026794, -16.6286945, 16.5777397

Time for backsubstitution: 1.05 seconds

### NS candidates at layer 1
type: A, layer: 1, pos: 11
type: B, layer: 1, pos: 11
type: B, layer: 1, pos: 34
type: A, layer: 1, pos: 34
type: B, layer: 1, pos: 9
type: A, layer: 1, pos: 4
type: B, layer: 1, pos: 4
type: B, layer: 1, pos: 42
type: A, layer: 1, pos: 42
type: A, layer: 1, pos: 47
type: B, layer: 1, pos: 12
type: B, layer: 1, pos: 16
type: B, layer: 1, pos: 31
type: A, layer: 1, pos: 16
type: A, layer: 1, pos: 39
type: B, layer: 1, pos: 39
type: A, layer: 1, pos: 33
type: B, layer: 1, pos: 33
type: A, layer: 1, pos: 23
type: B, layer: 1, pos: 23
type: A, layer: 1, pos: 30

Time for candidate selection: 0.08 seconds

### Candidate
type: A, layer: 1, pos: 11

### Candidate
type: B, layer: 1, pos: 11

### Candidate
type: B, layer: 1, pos: 34

### Candidate
type: A, layer: 1, pos: 34

### Candidate
type: B, layer: 1, pos: 9

### Candidate
type: A, layer: 1, pos: 4

### Candidate
type: B, layer: 1, pos: 4

### Candidate
type: B, layer: 1, pos: 42

### Candidate
type: A, layer: 1, pos: 42

### Candidate
type: A, layer: 1, pos: 47

## Relational analysis of NS_A1_A2_B1_B1_B1_B1_A1_A2_A1_A1
Optimization infeasible because this subproblem isn't reachable.

### Candidate
type: B, layer: 1, pos: 12

### Candidate
type: B, layer: 1, pos: 16

## Relational analysis of NS_A1_A2_B1_B1_B1_B1_A1_A2_A1_B1
Optimization infeasible because this subproblem isn't reachable.

### Candidate
type: B, layer: 1, pos: 31

## Relational analysis of NS_A1_A2_B1_B1_B1_B1_A1_A2_A1_B1
Optimization infeasible because this subproblem isn't reachable.

### Candidate
type: A, layer: 1, pos: 16

### Candidate
type: A, layer: 1, pos: 39

## Relational analysis of NS_A1_A2_B1_B1_B1_B1_A1_A2_A1_A1
Optimization infeasible because this subproblem isn't reachable.

### Candidate
type: B, layer: 1, pos: 39

### Candidate
type: A, layer: 1, pos: 33

## Relational analysis of NS_A1_A2_B1_B1_B1_B1_A1_A2_A1_A1
Optimization infeasible because this subproblem isn't reachable.

### Candidate
type: B, layer: 1, pos: 33

### Candidate
type: A, layer: 1, pos: 23

## Relational analysis of NS_A1_A2_B1_B1_B1_B1_A1_A2_A1_A1
Optimization infeasible because this subproblem isn't reachable.

### Candidate
type: B, layer: 1, pos: 23

### Candidate
type: A, layer: 1, pos: 30

## Relational analysis of NS_A1_A2_B1_B1_B1_B1_A1_A2_A1_A1

### Relational analysis result of NS_A1_A2_B1_B1_B1_B1_A1_A2_A1_A1
Status: Status.VERIFIED
Output dim: 3, lower bound: -20.6028099, upper bound: 20.5972793
time: 0.77 seconds

## Relational analysis of NS_A1_A2_B1_B1_B1_B1_A1_A2_A1_A2

### Relational analysis result of NS_A1_A2_B1_B1_B1_B1_A1_A2_A1_A2
Status: Status.VERIFIED
Output dim: 3, lower bound: -20.6028099, upper bound: 20.5972793
time: 0.86 seconds

## BFS NS instance: NS_A1_A2_B1_B1_B1_B1_A2_A2_A1

### Backsubstitution after applying NS history:
0: -3.5583677, 3.1767392, -2.8369715, 2.5377181, -6.0960846, 6.0137110
1: -14.0411711, 12.2968454, -11.1648302, 9.8173466, -23.8585167, 23.4616699
2: -7.0125341, 11.5522718, -5.6087904, 9.2279978, -16.2405319, 17.1610584
3: -12.2686119, 11.2447872, -9.7681313, 9.0039520, -21.2725639, 21.0129185
4: -8.9827728, 11.7397652, -7.1507564, 9.4026794, -18.3854465, 18.8905220

Time for backsubstitution: 0.97 seconds

### NS candidates at layer 1
type: B, layer: 1, pos: 11
type: A, layer: 1, pos: 11
type: B, layer: 1, pos: 34
type: A, layer: 1, pos: 34
type: A, layer: 1, pos: 4
type: B, layer: 1, pos: 4
type: B, layer: 1, pos: 9
type: B, layer: 1, pos: 42
type: A, layer: 1, pos: 42
type: A, layer: 1, pos: 39
type: B, layer: 1, pos: 31
type: B, layer: 1, pos: 16
type: B, layer: 1, pos: 12
type: A, layer: 1, pos: 16
type: B, layer: 1, pos: 33
type: B, layer: 1, pos: 39
type: A, layer: 1, pos: 23
type: A, layer: 1, pos: 33
type: B, layer: 1, pos: 23
type: A, layer: 1, pos: 30
type: A, layer: 1, pos: 47

Time for candidate selection: 0.09 seconds

### Candidate
type: B, layer: 1, pos: 11

### Candidate
type: A, layer: 1, pos: 11

### Candidate
type: B, layer: 1, pos: 34

### Candidate
type: A, layer: 1, pos: 34

### Candidate
type: A, layer: 1, pos: 4

### Candidate
type: B, layer: 1, pos: 4

### Candidate
type: B, layer: 1, pos: 9

### Candidate
type: B, layer: 1, pos: 42

### Candidate
type: A, layer: 1, pos: 42

### Candidate
type: A, layer: 1, pos: 39

## Relational analysis of NS_A1_A2_B1_B1_B1_B1_A2_A2_A1_A1
Optimization infeasible because this subproblem isn't reachable.

### Candidate
type: B, layer: 1, pos: 31

## Relational analysis of NS_A1_A2_B1_B1_B1_B1_A2_A2_A1_B1
Optimization infeasible because this subproblem isn't reachable.

### Candidate
type: B, layer: 1, pos: 16

## Relational analysis of NS_A1_A2_B1_B1_B1_B1_A2_A2_A1_B1
Optimization infeasible because this subproblem isn't reachable.

### Candidate
type: B, layer: 1, pos: 12

### Candidate
type: A, layer: 1, pos: 16

### Candidate
type: B, layer: 1, pos: 33

## Relational analysis of NS_A1_A2_B1_B1_B1_B1_A2_A2_A1_B1

### Relational analysis result of NS_A1_A2_B1_B1_B1_B1_A2_A2_A1_B1
Status: Status.UNKNOWN
Output dim: 3, lower bound: -20.6032340, upper bound: 20.5971874
time: 0.56 seconds

## Relational analysis of NS_A1_A2_B1_B1_B1_B1_A2_A2_A1_B2
Optimization infeasible because this subproblem isn't reachable.

### Candidate
type: B, layer: 1, pos: 39

### Candidate
type: A, layer: 1, pos: 23

## Relational analysis of NS_A1_A2_B1_B1_B1_B1_A2_A2_A1_A1
Optimization infeasible because this subproblem isn't reachable.

### Candidate
type: A, layer: 1, pos: 33

### Candidate
type: B, layer: 1, pos: 23

### Candidate
type: A, layer: 1, pos: 30

## Relational analysis of NS_A1_A2_B1_B1_B1_B1_A2_A2_A1_A1

### Relational analysis result of NS_A1_A2_B1_B1_B1_B1_A2_A2_A1_A1
Status: Status.VERIFIED
Output dim: 3, lower bound: -20.6016093, upper bound: 20.5971645
time: 0.69 seconds

## Relational analysis of NS_A1_A2_B1_B1_B1_B1_A2_A2_A1_A2

### Relational analysis result of NS_A1_A2_B1_B1_B1_B1_A2_A2_A1_A2
Status: Status.VERIFIED
Output dim: 3, lower bound: -20.6016093, upper bound: 20.5972557
time: 0.75 seconds

## BFS NS instance: NS_A1_A2_B1_B1_B1_B1_A2_A2_A2

### Backsubstitution after applying NS history:
0: -3.7766862, 3.3656306, -2.8369715, 2.5377181, -6.3144045, 6.2026024
1: -14.9201612, 12.9954309, -11.1648302, 9.8173466, -24.7375069, 24.1602612
2: -7.4469337, 12.2636309, -5.6087904, 9.2279978, -16.6749306, 17.8724194
3: -13.0294085, 11.8641272, -9.7681313, 9.0039520, -22.0333595, 21.6322594
4: -9.5356007, 12.4469366, -7.1507564, 9.4026794, -18.9382801, 19.5976925

Time for backsubstitution: 1.20 seconds

### NS candidates at layer 1
type: B, layer: 1, pos: 11
type: A, layer: 1, pos: 34
type: B, layer: 1, pos: 34
type: A, layer: 1, pos: 11
type: A, layer: 1, pos: 4
type: B, layer: 1, pos: 4
type: B, layer: 1, pos: 9
type: B, layer: 1, pos: 42
type: A, layer: 1, pos: 42
type: A, layer: 1, pos: 39
type: B, layer: 1, pos: 31
type: B, layer: 1, pos: 16
type: B, layer: 1, pos: 12
type: A, layer: 1, pos: 16
type: B, layer: 1, pos: 33
type: A, layer: 1, pos: 23
type: B, layer: 1, pos: 39
type: A, layer: 1, pos: 33
type: B, layer: 1, pos: 23
type: A, layer: 1, pos: 30
type: A, layer: 1, pos: 47

Time for candidate selection: 0.10 seconds

### Candidate
type: B, layer: 1, pos: 11

### Candidate
type: A, layer: 1, pos: 34

### Candidate
type: B, layer: 1, pos: 34

### Candidate
type: A, layer: 1, pos: 11

### Candidate
type: A, layer: 1, pos: 4

### Candidate
type: B, layer: 1, pos: 4

### Candidate
type: B, layer: 1, pos: 9

### Candidate
type: B, layer: 1, pos: 42

### Candidate
type: A, layer: 1, pos: 42

### Candidate
type: A, layer: 1, pos: 39

## Relational analysis of NS_A1_A2_B1_B1_B1_B1_A2_A2_A2_A1
Optimization infeasible because this subproblem isn't reachable.

### Candidate
type: B, layer: 1, pos: 31

## Relational analysis of NS_A1_A2_B1_B1_B1_B1_A2_A2_A2_B1
Optimization infeasible because this subproblem isn't reachable.

### Candidate
type: B, layer: 1, pos: 16

## Relational analysis of NS_A1_A2_B1_B1_B1_B1_A2_A2_A2_B1
Optimization infeasible because this subproblem isn't reachable.

### Candidate
type: B, layer: 1, pos: 12

### Candidate
type: A, layer: 1, pos: 16

### Candidate
type: B, layer: 1, pos: 33

## Relational analysis of NS_A1_A2_B1_B1_B1_B1_A2_A2_A2_B1

### Relational analysis result of NS_A1_A2_B1_B1_B1_B1_A2_A2_A2_B1
Status: Status.VERIFIED
Output dim: 3, lower bound: -20.6031203, upper bound: 20.5972040
time: 0.65 seconds

## Relational analysis of NS_A1_A2_B1_B1_B1_B1_A2_A2_A2_B2
Optimization infeasible because this subproblem isn't reachable.

### Candidate
type: A, layer: 1, pos: 23

## Relational analysis of NS_A1_A2_B1_B1_B1_B1_A2_A2_A2_A1
Optimization infeasible because this subproblem isn't reachable.

### Candidate
type: B, layer: 1, pos: 39

### Candidate
type: A, layer: 1, pos: 33

### Candidate
type: B, layer: 1, pos: 23

### Candidate
type: A, layer: 1, pos: 30

## Relational analysis of NS_A1_A2_B1_B1_B1_B1_A2_A2_A2_A1

### Relational analysis result of NS_A1_A2_B1_B1_B1_B1_A2_A2_A2_A1
Status: Status.VERIFIED
Output dim: 3, lower bound: -20.6029094, upper bound: 20.5972723
time: 0.75 seconds

## Relational analysis of NS_A1_A2_B1_B1_B1_B1_A2_A2_A2_A2

### Relational analysis result of NS_A1_A2_B1_B1_B1_B1_A2_A2_A2_A2
Status: Status.VERIFIED
Output dim: 3, lower bound: -20.6029094, upper bound: 20.5972723
time: 0.81 seconds

## BFS NS instance: NS_A1_A2_B1_B1_B2_B1_A2_A1_A1

### Backsubstitution after applying NS history:
0: -2.7816570, 2.4818509, -2.6776226, 2.3901095, -5.1717663, 5.1594734
1: -10.9689932, 9.6364336, -10.5423784, 9.2746677, -20.2436562, 20.1788120
2: -5.4935513, 9.0054016, -5.3035603, 8.6706009, -14.1641521, 14.3089619
3: -9.6097250, 8.8251762, -9.2467489, 8.5087490, -18.1184731, 18.0719242
4: -7.0272222, 9.1884327, -6.7716413, 8.8663626, -15.8935814, 15.9600735

Time for backsubstitution: 0.98 seconds

### NS candidates at layer 1
type: B, layer: 1, pos: 11
type: A, layer: 1, pos: 11
type: B, layer: 1, pos: 34
type: A, layer: 1, pos: 34
type: A, layer: 1, pos: 12
type: B, layer: 1, pos: 12
type: A, layer: 1, pos: 39
type: B, layer: 1, pos: 9
type: B, layer: 1, pos: 42
type: B, layer: 1, pos: 39
type: B, layer: 1, pos: 16
type: B, layer: 1, pos: 33
type: A, layer: 1, pos: 16
type: A, layer: 1, pos: 4
type: B, layer: 1, pos: 31
type: A, layer: 1, pos: 23
type: B, layer: 1, pos: 23
type: B, layer: 1, pos: 24
type: A, layer: 1, pos: 33
type: A, layer: 1, pos: 47
type: A, layer: 1, pos: 30

Time for candidate selection: 0.08 seconds

### Candidate
type: B, layer: 1, pos: 11

### Candidate
type: A, layer: 1, pos: 11

### Candidate
type: B, layer: 1, pos: 34

### Candidate
type: A, layer: 1, pos: 34

### Candidate
type: A, layer: 1, pos: 12

### Candidate
type: B, layer: 1, pos: 12

### Candidate
type: A, layer: 1, pos: 39

## Relational analysis of NS_A1_A2_B1_B1_B2_B1_A2_A1_A1_A1
Optimization infeasible because this subproblem isn't reachable.

### Candidate
type: B, layer: 1, pos: 9

### Candidate
type: B, layer: 1, pos: 42

## Relational analysis of NS_A1_A2_B1_B1_B2_B1_A2_A1_A1_B1
Optimization infeasible because this subproblem isn't reachable.

### Candidate
type: B, layer: 1, pos: 39

### Candidate
type: B, layer: 1, pos: 16

## Relational analysis of NS_A1_A2_B1_B1_B2_B1_A2_A1_A1_B1
Optimization infeasible because this subproblem isn't reachable.

### Candidate
type: B, layer: 1, pos: 33

## Relational analysis of NS_A1_A2_B1_B1_B2_B1_A2_A1_A1_B1

### Relational analysis result of NS_A1_A2_B1_B1_B2_B1_A2_A1_A1_B1
Status: Status.UNKNOWN
Output dim: 3, lower bound: -20.6033943, upper bound: 20.5990128
time: 0.75 seconds

## Relational analysis of NS_A1_A2_B1_B1_B2_B1_A2_A1_A1_B2

### Relational analysis result of NS_A1_A2_B1_B1_B2_B1_A2_A1_A1_B2
Status: Status.UNKNOWN
Output dim: 3, lower bound: -20.6035002, upper bound: 20.5994935
time: 0.73 seconds

## BFS NS instance: NS_A1_A2_B1_B1_B2_B1_A2_A1_A2

### Backsubstitution after applying NS history:
0: -2.9391685, 2.6167705, -2.6776226, 2.3901095, -5.3292770, 5.2943931
1: -11.6126900, 10.1238680, -10.5423784, 9.2746677, -20.8873539, 20.6662464
2: -5.8127203, 9.5147934, -5.3035603, 8.6706009, -14.4833193, 14.8183537
3: -10.1638746, 9.2626143, -9.2467489, 8.5087490, -18.6726227, 18.5093632
4: -7.4215121, 9.6846666, -6.7716413, 8.8663626, -16.2878742, 16.4563065

Time for backsubstitution: 0.97 seconds

### NS candidates at layer 1
type: B, layer: 1, pos: 11
type: B, layer: 1, pos: 34
type: A, layer: 1, pos: 11
type: A, layer: 1, pos: 34
type: A, layer: 1, pos: 12
type: B, layer: 1, pos: 12
type: A, layer: 1, pos: 39
type: B, layer: 1, pos: 9
type: B, layer: 1, pos: 42
type: B, layer: 1, pos: 16
type: B, layer: 1, pos: 39
type: B, layer: 1, pos: 33
type: B, layer: 1, pos: 31
type: B, layer: 1, pos: 24
type: A, layer: 1, pos: 4
type: A, layer: 1, pos: 16
type: A, layer: 1, pos: 23
type: B, layer: 1, pos: 23
type: A, layer: 1, pos: 33
type: A, layer: 1, pos: 47
type: A, layer: 1, pos: 30

Time for candidate selection: 0.08 seconds

### Candidate
type: B, layer: 1, pos: 11

### Candidate
type: B, layer: 1, pos: 34

### Candidate
type: A, layer: 1, pos: 11

### Candidate
type: A, layer: 1, pos: 34

### Candidate
type: A, layer: 1, pos: 12

### Candidate
type: B, layer: 1, pos: 12

### Candidate
type: A, layer: 1, pos: 39

## Relational analysis of NS_A1_A2_B1_B1_B2_B1_A2_A1_A2_A1
Optimization infeasible because this subproblem isn't reachable.

### Candidate
type: B, layer: 1, pos: 9

### Candidate
type: B, layer: 1, pos: 42

## Relational analysis of NS_A1_A2_B1_B1_B2_B1_A2_A1_A2_B1
Optimization infeasible because this subproblem isn't reachable.

### Candidate
type: B, layer: 1, pos: 16

## Relational analysis of NS_A1_A2_B1_B1_B2_B1_A2_A1_A2_B1
Optimization infeasible because this subproblem isn't reachable.

### Candidate
type: B, layer: 1, pos: 39

### Candidate
type: B, layer: 1, pos: 33

## Relational analysis of NS_A1_A2_B1_B1_B2_B1_A2_A1_A2_B1

### Relational analysis result of NS_A1_A2_B1_B1_B2_B1_A2_A1_A2_B1
Status: Status.VERIFIED
Output dim: 3, lower bound: -20.6031175, upper bound: 20.5990099
time: 1.14 seconds

## Relational analysis of NS_A1_A2_B1_B1_B2_B1_A2_A1_A2_B2

### Relational analysis result of NS_A1_A2_B1_B1_B2_B1_A2_A1_A2_B2
Status: Status.UNKNOWN
Output dim: 3, lower bound: -20.6032235, upper bound: 20.5994906
time: 0.79 seconds

## BFS NS instance: NS_A1_A2_B1_B1_B2_B1_A2_A2_B1

### Backsubstitution after applying NS history:
0: -3.5902476, 3.1974847, -2.5164056, 2.2540786, -5.8443260, 5.7138901
1: -14.1819506, 12.3807278, -9.9001131, 8.7325802, -22.9145279, 22.2808399
2: -7.0806346, 11.6148405, -4.9823089, 8.1783438, -15.2589779, 16.5971470
3: -12.3865166, 11.3239698, -8.6768465, 8.0154171, -20.4019279, 20.0008144
4: -9.0666447, 11.8067245, -6.3510180, 8.3670273, -17.4336720, 18.1577377

Time for backsubstitution: 1.16 seconds

### NS candidates at layer 1
type: B, layer: 1, pos: 11
type: B, layer: 1, pos: 34
type: A, layer: 1, pos: 34
type: A, layer: 1, pos: 12
type: B, layer: 1, pos: 12
type: A, layer: 1, pos: 11
type: B, layer: 1, pos: 9
type: A, layer: 1, pos: 39
type: A, layer: 1, pos: 31
type: B, layer: 1, pos: 42
type: B, layer: 1, pos: 31
type: B, layer: 1, pos: 16
type: A, layer: 1, pos: 23
type: A, layer: 1, pos: 16
type: A, layer: 1, pos: 4
type: B, layer: 1, pos: 24
type: B, layer: 1, pos: 39
type: B, layer: 1, pos: 23
type: A, layer: 1, pos: 33
type: A, layer: 1, pos: 30
type: A, layer: 1, pos: 47

Time for candidate selection: 0.08 seconds

### Candidate
type: B, layer: 1, pos: 11

### Candidate
type: B, layer: 1, pos: 34

### Candidate
type: A, layer: 1, pos: 34

### Candidate
type: A, layer: 1, pos: 12

### Candidate
type: B, layer: 1, pos: 12

### Candidate
type: A, layer: 1, pos: 11

### Candidate
type: B, layer: 1, pos: 9

### Candidate
type: A, layer: 1, pos: 39

### Candidate
type: A, layer: 1, pos: 31

## Relational analysis of NS_A1_A2_B1_B1_B2_B1_A2_A2_B1_A1

### Relational analysis result of NS_A1_A2_B1_B1_B2_B1_A2_A2_B1_A1
Status: Status.UNKNOWN
Output dim: 3, lower bound: -20.6033752, upper bound: 20.5989888
time: 0.72 seconds

## Relational analysis of NS_A1_A2_B1_B1_B2_B1_A2_A2_B1_A2

### Relational analysis result of NS_A1_A2_B1_B1_B2_B1_A2_A2_B1_A2
Status: Status.UNKNOWN
Output dim: 3, lower bound: -20.6032433, upper bound: 20.5990013
time: 0.63 seconds

## BFS NS instance: NS_A1_A2_B1_B1_B2_B1_A2_A2_B2

### Backsubstitution after applying NS history:
0: -3.5902476, 3.1974847, -2.6532226, 2.3691103, -5.9593582, 5.8507071
1: -14.1819506, 12.3807278, -10.4453049, 9.1917686, -23.3737144, 22.8260288
2: -7.0806346, 11.6148405, -5.2544699, 8.5943823, -15.6750154, 16.8693104
3: -12.3865166, 11.3239698, -9.1631756, 8.4334030, -20.8199196, 20.4871445
4: -9.0666447, 11.8067245, -6.7102489, 8.7898760, -17.8565216, 18.5169735

Time for backsubstitution: 0.99 seconds

### NS candidates at layer 1
type: B, layer: 1, pos: 11
type: B, layer: 1, pos: 34
type: A, layer: 1, pos: 34
type: A, layer: 1, pos: 12
type: B, layer: 1, pos: 12
type: A, layer: 1, pos: 11
type: B, layer: 1, pos: 9
type: A, layer: 1, pos: 39
type: A, layer: 1, pos: 31
type: B, layer: 1, pos: 42
type: B, layer: 1, pos: 31
type: B, layer: 1, pos: 16
type: A, layer: 1, pos: 23
type: A, layer: 1, pos: 16
type: A, layer: 1, pos: 4
type: B, layer: 1, pos: 39
type: B, layer: 1, pos: 24
type: B, layer: 1, pos: 23
type: A, layer: 1, pos: 33
type: A, layer: 1, pos: 30
type: A, layer: 1, pos: 47

Time for candidate selection: 0.08 seconds

### Candidate
type: B, layer: 1, pos: 11

### Candidate
type: B, layer: 1, pos: 34

### Candidate
type: A, layer: 1, pos: 34

### Candidate
type: A, layer: 1, pos: 12

### Candidate
type: B, layer: 1, pos: 12

### Candidate
type: A, layer: 1, pos: 11

### Candidate
type: B, layer: 1, pos: 9

### Candidate
type: A, layer: 1, pos: 39

### Candidate
type: A, layer: 1, pos: 31

## Relational analysis of NS_A1_A2_B1_B1_B2_B1_A2_A2_B2_A1

### Relational analysis result of NS_A1_A2_B1_B1_B2_B1_A2_A2_B2_A1
Status: Status.UNKNOWN
Output dim: 3, lower bound: -20.6034812, upper bound: 20.5994695
time: 0.56 seconds

## Relational analysis of NS_A1_A2_B1_B1_B2_B1_A2_A2_B2_A2

### Relational analysis result of NS_A1_A2_B1_B1_B2_B1_A2_A2_B2_A2
Status: Status.UNKNOWN
Output dim: 3, lower bound: -20.6033492, upper bound: 20.5994820
time: 0.70 seconds

## BFS NS instance: NS_A1_A2_B1_B1_B2_B2_A1_B1_A1

### Backsubstitution after applying NS history:
0: -2.9449172, 2.6235266, -2.1720495, 1.9849454, -4.9298625, 4.7955756
1: -11.6181107, 10.1830626, -8.5200977, 7.6689053, -19.2870121, 18.7031593
2: -5.8201113, 9.5109978, -4.3270049, 7.1652737, -12.9853849, 13.8380013
3: -10.1757755, 9.3143005, -7.4694524, 7.0567627, -17.2325382, 16.7837524
4: -7.4468493, 9.6967487, -5.4614787, 7.3901196, -14.8369694, 15.1582279

Time for backsubstitution: 1.13 seconds

### NS candidates at layer 1
type: B, layer: 1, pos: 11
type: B, layer: 1, pos: 34
type: A, layer: 1, pos: 11
type: A, layer: 1, pos: 34
type: B, layer: 1, pos: 12
type: A, layer: 1, pos: 12
type: A, layer: 1, pos: 39
type: B, layer: 1, pos: 9
type: A, layer: 1, pos: 42
type: B, layer: 1, pos: 33
type: A, layer: 1, pos: 4
type: B, layer: 1, pos: 16
type: B, layer: 1, pos: 39
type: A, layer: 1, pos: 16
type: B, layer: 1, pos: 31
type: A, layer: 1, pos: 23
type: B, layer: 1, pos: 24
type: B, layer: 1, pos: 23
type: A, layer: 1, pos: 33
type: A, layer: 1, pos: 30
type: A, layer: 1, pos: 47

Time for candidate selection: 0.08 seconds

### Candidate
type: B, layer: 1, pos: 11

### Candidate
type: B, layer: 1, pos: 34

### Candidate
type: A, layer: 1, pos: 11

### Candidate
type: A, layer: 1, pos: 34

### Candidate
type: B, layer: 1, pos: 12

### Candidate
type: A, layer: 1, pos: 12

### Candidate
type: A, layer: 1, pos: 39

### Candidate
type: B, layer: 1, pos: 9

### Candidate
type: A, layer: 1, pos: 42

## Relational analysis of NS_A1_A2_B1_B1_B2_B2_A1_B1_A1_A1

### Relational analysis result of NS_A1_A2_B1_B1_B2_B2_A1_B1_A1_A1
Status: Status.VERIFIED
Output dim: 3, lower bound: -20.6014707, upper bound: 20.5986244
time: 0.66 seconds

## Relational analysis of NS_A1_A2_B1_B1_B2_B2_A1_B1_A1_A2

### Relational analysis result of NS_A1_A2_B1_B1_B2_B2_A1_B1_A1_A2
Status: Status.VERIFIED
Output dim: 3, lower bound: -20.6014707, upper bound: 20.5987226
time: 0.79 seconds

## BFS NS instance: NS_A1_A2_B1_B1_B2_B2_A1_B1_A2

### Backsubstitution after applying NS history:
0: -3.2134719, 2.8538458, -2.1720495, 1.9849454, -5.1984167, 5.0258946
1: -12.6991587, 11.0321903, -8.5200977, 7.6689053, -20.3680592, 19.5522842
2: -6.3571825, 10.3699150, -4.3270049, 7.1652737, -13.5224562, 14.6969194
3: -11.1127367, 10.0746346, -7.4694524, 7.0567627, -18.1694965, 17.5440865
4: -8.1241493, 10.5364218, -5.4614787, 7.3901196, -15.5142689, 15.9978991

Time for backsubstitution: 1.02 seconds

### NS candidates at layer 1
type: B, layer: 1, pos: 11
type: B, layer: 1, pos: 34
type: A, layer: 1, pos: 34
type: A, layer: 1, pos: 39
type: B, layer: 1, pos: 12
type: A, layer: 1, pos: 12
type: B, layer: 1, pos: 9
type: A, layer: 1, pos: 42
type: B, layer: 1, pos: 33
type: A, layer: 1, pos: 4
type: B, layer: 1, pos: 39
type: B, layer: 1, pos: 16
type: A, layer: 1, pos: 16
type: A, layer: 1, pos: 23
type: B, layer: 1, pos: 31
type: B, layer: 1, pos: 24
type: A, layer: 1, pos: 33
type: B, layer: 1, pos: 23
type: A, layer: 1, pos: 11
type: A, layer: 1, pos: 30
type: A, layer: 1, pos: 47

Time for candidate selection: 0.08 seconds

### Candidate
type: B, layer: 1, pos: 11

### Candidate
type: B, layer: 1, pos: 34

### Candidate
type: A, layer: 1, pos: 34

### Candidate
type: A, layer: 1, pos: 39

### Candidate
type: B, layer: 1, pos: 12

### Candidate
type: A, layer: 1, pos: 12

### Candidate
type: B, layer: 1, pos: 9

### Candidate
type: A, layer: 1, pos: 42

## Relational analysis of NS_A1_A2_B1_B1_B2_B2_A1_B1_A2_A1

### Relational analysis result of NS_A1_A2_B1_B1_B2_B2_A1_B1_A2_A1
Status: Status.VERIFIED
Output dim: 3, lower bound: -20.6009570, upper bound: 20.5985516
time: 0.83 seconds

## Relational analysis of NS_A1_A2_B1_B1_B2_B2_A1_B1_A2_A2

### Relational analysis result of NS_A1_A2_B1_B1_B2_B2_A1_B1_A2_A2
Status: Status.VERIFIED
Output dim: 3, lower bound: -20.6009570, upper bound: 20.5985516
time: 1.02 seconds

## BFS NS instance: NS_A1_A2_B1_B1_B2_B2_A1_B2_A1

### Backsubstitution after applying NS history:
0: -2.9449172, 2.6235266, -2.8515317, 2.5491934, -5.4941101, 5.4750586
1: -11.6181107, 10.1830626, -11.2261553, 9.8856926, -21.5037994, 21.4092178
2: -5.8201113, 9.5109978, -5.6394186, 9.2582951, -15.0784063, 15.1504164
3: -10.1757755, 9.3143005, -9.8397264, 9.0728273, -19.2486019, 19.1540260
4: -7.4468493, 9.6967487, -7.2097111, 9.4593086, -16.9061584, 16.9064560

Time for backsubstitution: 1.24 seconds

### NS candidates at layer 1
type: B, layer: 1, pos: 11
type: A, layer: 1, pos: 11
type: B, layer: 1, pos: 34
type: A, layer: 1, pos: 34
type: A, layer: 1, pos: 12
type: B, layer: 1, pos: 12
type: B, layer: 1, pos: 9
type: A, layer: 1, pos: 39
type: B, layer: 1, pos: 33
type: B, layer: 1, pos: 16
type: A, layer: 1, pos: 4
type: A, layer: 1, pos: 42
type: A, layer: 1, pos: 23
type: A, layer: 1, pos: 16
type: B, layer: 1, pos: 23
type: B, layer: 1, pos: 39
type: B, layer: 1, pos: 24
type: B, layer: 1, pos: 31
type: A, layer: 1, pos: 33
type: A, layer: 1, pos: 47
type: A, layer: 1, pos: 30

Time for candidate selection: 0.09 seconds

### Candidate
type: B, layer: 1, pos: 11

### Candidate
type: A, layer: 1, pos: 11

### Candidate
type: B, layer: 1, pos: 34

### Candidate
type: A, layer: 1, pos: 34

### Candidate
type: A, layer: 1, pos: 12

### Candidate
type: B, layer: 1, pos: 12

### Candidate
type: B, layer: 1, pos: 9

### Candidate
type: A, layer: 1, pos: 39

### Candidate
type: B, layer: 1, pos: 33

### Candidate
type: B, layer: 1, pos: 16

## Relational analysis of NS_A1_A2_B1_B1_B2_B2_A1_B2_A1_B1
Optimization infeasible because this subproblem isn't reachable.

### Candidate
type: A, layer: 1, pos: 4

## Relational analysis of NS_A1_A2_B1_B1_B2_B2_A1_B2_A1_A1
Optimization infeasible because this subproblem isn't reachable.

### Candidate
type: A, layer: 1, pos: 42

## Relational analysis of NS_A1_A2_B1_B1_B2_B2_A1_B2_A1_A1

### Relational analysis result of NS_A1_A2_B1_B1_B2_B2_A1_B2_A1_A1
Status: Status.VERIFIED
Output dim: 3, lower bound: -20.6014707, upper bound: 20.5992590
time: 1.09 seconds

## Relational analysis of NS_A1_A2_B1_B1_B2_B2_A1_B2_A1_A2

### Relational analysis result of NS_A1_A2_B1_B1_B2_B2_A1_B2_A1_A2
Status: Status.VERIFIED
Output dim: 3, lower bound: -20.6014707, upper bound: 20.5993572
time: 0.76 seconds

## BFS NS instance: NS_A1_A2_B1_B1_B2_B2_A1_B2_A2

### Backsubstitution after applying NS history:
0: -3.2134719, 2.8538458, -2.8515317, 2.5491934, -5.7626643, 5.7053771
1: -12.6991587, 11.0321903, -11.2261553, 9.8856926, -22.5848484, 22.2583427
2: -6.3571825, 10.3699150, -5.6394186, 9.2582951, -15.6154776, 16.0093346
3: -11.1127367, 10.0746346, -9.8397264, 9.0728273, -20.1855621, 19.9143600
4: -8.1241493, 10.5364218, -7.2097111, 9.4593086, -17.5834579, 17.7461319

Time for backsubstitution: 1.08 seconds

### NS candidates at layer 1
type: B, layer: 1, pos: 11
type: B, layer: 1, pos: 34
type: A, layer: 1, pos: 11
type: A, layer: 1, pos: 34
type: A, layer: 1, pos: 12
type: B, layer: 1, pos: 12
type: B, layer: 1, pos: 9
type: A, layer: 1, pos: 39
type: A, layer: 1, pos: 42
type: A, layer: 1, pos: 4
type: B, layer: 1, pos: 33
type: B, layer: 1, pos: 16
type: B, layer: 1, pos: 39
type: A, layer: 1, pos: 23
type: A, layer: 1, pos: 16
type: B, layer: 1, pos: 31
type: B, layer: 1, pos: 23
type: B, layer: 1, pos: 24
type: A, layer: 1, pos: 33
type: A, layer: 1, pos: 47
type: A, layer: 1, pos: 30

Time for candidate selection: 0.08 seconds

### Candidate
type: B, layer: 1, pos: 11

### Candidate
type: B, layer: 1, pos: 34

### Candidate
type: A, layer: 1, pos: 11

### Candidate
type: A, layer: 1, pos: 34

### Candidate
type: A, layer: 1, pos: 12

### Candidate
type: B, layer: 1, pos: 12

### Candidate
type: B, layer: 1, pos: 9

### Candidate
type: A, layer: 1, pos: 39

### Candidate
type: A, layer: 1, pos: 42

## Relational analysis of NS_A1_A2_B1_B1_B2_B2_A1_B2_A2_A1

### Relational analysis result of NS_A1_A2_B1_B1_B2_B2_A1_B2_A2_A1
Status: Status.VERIFIED
Output dim: 3, lower bound: -20.6009570, upper bound: 20.5991862
time: 0.75 seconds

## Relational analysis of NS_A1_A2_B1_B1_B2_B2_A1_B2_A2_A2

### Relational analysis result of NS_A1_A2_B1_B1_B2_B2_A1_B2_A2_A2
Status: Status.VERIFIED
Output dim: 3, lower bound: -20.6009570, upper bound: 20.5985516
time: 0.83 seconds

## BFS NS instance: NS_A1_A2_B1_B1_B2_B2_A2_A2_B1

### Backsubstitution after applying NS history:
0: -3.6803167, 3.2757277, -2.1720495, 1.9849454, -5.6652622, 5.4477768
1: -14.5396061, 12.6777191, -8.5200977, 7.6689053, -22.2085094, 21.1978168
2: -7.2490587, 11.9156590, -4.3270049, 7.1652737, -14.4143324, 16.2426624
3: -12.6998749, 11.5894537, -7.4694524, 7.0567627, -19.7566319, 19.0589046
4: -9.2898760, 12.1022100, -5.4614787, 7.3901196, -16.6799965, 17.5636826

Time for backsubstitution: 1.03 seconds

### NS candidates at layer 1
type: B, layer: 1, pos: 11
type: B, layer: 1, pos: 34
type: A, layer: 1, pos: 34
type: A, layer: 1, pos: 12
type: B, layer: 1, pos: 12
type: A, layer: 1, pos: 39
type: B, layer: 1, pos: 9
type: A, layer: 1, pos: 31
type: B, layer: 1, pos: 33
type: B, layer: 1, pos: 31
type: A, layer: 1, pos: 4
type: B, layer: 1, pos: 16
type: A, layer: 1, pos: 23
type: A, layer: 1, pos: 16
type: B, layer: 1, pos: 39
type: B, layer: 1, pos: 23
type: B, layer: 1, pos: 24
type: A, layer: 1, pos: 33
type: A, layer: 1, pos: 11
type: A, layer: 1, pos: 30
type: A, layer: 1, pos: 47

Time for candidate selection: 0.08 seconds

### Candidate
type: B, layer: 1, pos: 11

### Candidate
type: B, layer: 1, pos: 34

### Candidate
type: A, layer: 1, pos: 34

### Candidate
type: A, layer: 1, pos: 12

### Candidate
type: B, layer: 1, pos: 12

### Candidate
type: A, layer: 1, pos: 39

### Candidate
type: B, layer: 1, pos: 9

### Candidate
type: A, layer: 1, pos: 31

## Relational analysis of NS_A1_A2_B1_B1_B2_B2_A2_A2_B1_A1

### Relational analysis result of NS_A1_A2_B1_B1_B2_B2_A2_A2_B1_A1
Status: Status.UNKNOWN
Output dim: 3, lower bound: -20.6034656, upper bound: 20.5986986
time: 0.89 seconds

## Relational analysis of NS_A1_A2_B1_B1_B2_B2_A2_A2_B1_A2

### Relational analysis result of NS_A1_A2_B1_B1_B2_B2_A2_A2_B1_A2
Status: Status.UNKNOWN
Output dim: 3, lower bound: -20.6033252, upper bound: 20.5987112
time: 0.74 seconds

## BFS NS instance: NS_A1_A2_B1_B1_B2_B2_A2_A2_B2

### Backsubstitution after applying NS history:
0: -3.6803167, 3.2757277, -2.8515317, 2.5491934, -6.2295103, 6.1272593
1: -14.5396061, 12.6777191, -11.2261553, 9.8856926, -24.4252930, 23.9038734
2: -7.2490587, 11.9156590, -5.6394186, 9.2582951, -16.5073490, 17.5550766
3: -12.6998749, 11.5894537, -9.8397264, 9.0728273, -21.7726974, 21.4291801
4: -9.2898760, 12.1022100, -7.2097111, 9.4593086, -18.7491837, 19.3119144

Time for backsubstitution: 1.24 seconds

### NS candidates at layer 1
type: B, layer: 1, pos: 11
type: B, layer: 1, pos: 34
type: A, layer: 1, pos: 34
type: A, layer: 1, pos: 12
type: B, layer: 1, pos: 12
type: B, layer: 1, pos: 9
type: A, layer: 1, pos: 11
type: A, layer: 1, pos: 39
type: B, layer: 1, pos: 33
type: A, layer: 1, pos: 4
type: B, layer: 1, pos: 16
type: A, layer: 1, pos: 23
type: A, layer: 1, pos: 31
type: B, layer: 1, pos: 31
type: A, layer: 1, pos: 16
type: B, layer: 1, pos: 23
type: B, layer: 1, pos: 24
type: B, layer: 1, pos: 39
type: A, layer: 1, pos: 33
type: A, layer: 1, pos: 30
type: A, layer: 1, pos: 47

Time for candidate selection: 0.09 seconds

### Candidate
type: B, layer: 1, pos: 11

### Candidate
type: B, layer: 1, pos: 34

### Candidate
type: A, layer: 1, pos: 34

### Candidate
type: A, layer: 1, pos: 12

### Candidate
type: B, layer: 1, pos: 12

### Candidate
type: B, layer: 1, pos: 9

### Candidate
type: A, layer: 1, pos: 11

### Candidate
type: A, layer: 1, pos: 39

### Candidate
type: B, layer: 1, pos: 33

### Candidate
type: A, layer: 1, pos: 4

## Relational analysis of NS_A1_A2_B1_B1_B2_B2_A2_A2_B2_A1
Optimization infeasible because this subproblem isn't reachable.

### Candidate
type: B, layer: 1, pos: 16

### Candidate
type: A, layer: 1, pos: 23

## Relational analysis of NS_A1_A2_B1_B1_B2_B2_A2_A2_B2_A1
Optimization infeasible because this subproblem isn't reachable.

### Candidate
type: A, layer: 1, pos: 31

## Relational analysis of NS_A1_A2_B1_B1_B2_B2_A2_A2_B2_A1

### Relational analysis result of NS_A1_A2_B1_B1_B2_B2_A2_A2_B2_A1
Status: Status.UNKNOWN
Output dim: 3, lower bound: -20.6034656, upper bound: 20.5986986
time: 0.81 seconds

## Relational analysis of NS_A1_A2_B1_B1_B2_B2_A2_A2_B2_A2

### Relational analysis result of NS_A1_A2_B1_B1_B2_B2_A2_A2_B2_A2
Status: Status.UNKNOWN
Output dim: 3, lower bound: -20.6033252, upper bound: 20.5993457
time: 0.69 seconds

## BFS NS instance: NS_A1_A2_B1_B2_A2_B1_A1_A1_B2

### Backsubstitution after applying NS history:
0: -2.9575121, 2.6303887, -3.1073406, 2.7655692, -5.7230816, 5.7377291
1: -11.6835804, 10.1886177, -12.2705278, 10.7313585, -22.4149399, 22.4591408
2: -5.8330021, 9.5619545, -6.1469731, 10.0111504, -15.8441525, 15.7089272
3: -10.2288876, 9.3145103, -10.7416000, 9.8079548, -20.0368404, 20.0561085
4: -7.4482050, 9.7160215, -7.8535166, 10.2031994, -17.6514053, 17.5695381

Time for backsubstitution: 1.30 seconds

### NS candidates at layer 1
type: B, layer: 1, pos: 11
type: A, layer: 1, pos: 11
type: B, layer: 1, pos: 34
type: A, layer: 1, pos: 34
type: A, layer: 1, pos: 31
type: B, layer: 1, pos: 31
type: B, layer: 1, pos: 12
type: A, layer: 1, pos: 12
type: B, layer: 1, pos: 9
type: A, layer: 1, pos: 39
type: B, layer: 1, pos: 4
type: B, layer: 1, pos: 42
type: B, layer: 1, pos: 39
type: A, layer: 1, pos: 4
type: B, layer: 1, pos: 16
type: A, layer: 1, pos: 16
type: B, layer: 1, pos: 33
type: A, layer: 1, pos: 23
type: B, layer: 1, pos: 23
type: A, layer: 1, pos: 33
type: A, layer: 1, pos: 30

Time for candidate selection: 0.09 seconds

### Candidate
type: B, layer: 1, pos: 11

### Candidate
type: A, layer: 1, pos: 11

### Candidate
type: B, layer: 1, pos: 34

### Candidate
type: A, layer: 1, pos: 34

### Candidate
type: A, layer: 1, pos: 31

## Relational analysis of NS_A1_A2_B1_B2_A2_B1_A1_A1_B2_A1
Optimization infeasible because this subproblem isn't reachable.

### Candidate
type: B, layer: 1, pos: 31

### Candidate
type: B, layer: 1, pos: 12

### Candidate
type: A, layer: 1, pos: 12

### Candidate
type: B, layer: 1, pos: 9

## Relational analysis of NS_A1_A2_B1_B2_A2_B1_A1_A1_B2_B1
Optimization infeasible because this subproblem isn't reachable.

### Candidate
type: A, layer: 1, pos: 39

## Relational analysis of NS_A1_A2_B1_B2_A2_B1_A1_A1_B2_A1
Optimization infeasible because this subproblem isn't reachable.

### Candidate
type: B, layer: 1, pos: 4

## Relational analysis of NS_A1_A2_B1_B2_A2_B1_A1_A1_B2_B1
Optimization infeasible because this subproblem isn't reachable.

### Candidate
type: B, layer: 1, pos: 42

## Relational analysis of NS_A1_A2_B1_B2_A2_B1_A1_A1_B2_B1
Optimization infeasible because this subproblem isn't reachable.

### Candidate
type: B, layer: 1, pos: 39

### Candidate
type: A, layer: 1, pos: 4

### Candidate
type: B, layer: 1, pos: 16

## Relational analysis of NS_A1_A2_B1_B2_A2_B1_A1_A1_B2_B1
Optimization infeasible because this subproblem isn't reachable.

### Candidate
type: A, layer: 1, pos: 16

### Candidate
type: B, layer: 1, pos: 33

## Relational analysis of NS_A1_A2_B1_B2_A2_B1_A1_A1_B2_B1
Optimization infeasible because this subproblem isn't reachable.

### Candidate
type: A, layer: 1, pos: 23

## Relational analysis of NS_A1_A2_B1_B2_A2_B1_A1_A1_B2_A1
Optimization infeasible because this subproblem isn't reachable.

### Candidate
type: B, layer: 1, pos: 23

### Candidate
type: A, layer: 1, pos: 33

### Candidate
type: A, layer: 1, pos: 30

## Relational analysis of NS_A1_A2_B1_B2_A2_B1_A1_A1_B2_A1

### Relational analysis result of NS_A1_A2_B1_B2_A2_B1_A1_A1_B2_A1
Status: Status.UNKNOWN
Output dim: 3, lower bound: -20.5959150, upper bound: 20.6033041
time: 0.67 seconds

## Relational analysis of NS_A1_A2_B1_B2_A2_B1_A1_A1_B2_A2

### Relational analysis result of NS_A1_A2_B1_B2_A2_B1_A1_A1_B2_A2
Status: Status.VERIFIED
Output dim: 3, lower bound: -20.5959150, upper bound: 20.6028985
time: 0.67 seconds

## BFS NS instance: NS_A1_A2_B1_B2_A2_B1_A1_A2_B1

### Backsubstitution after applying NS history:
0: -2.9541757, 2.6261964, -2.9369402, 2.6221790, -5.5763550, 5.5631366
1: -11.6670923, 10.1961460, -11.5816984, 10.1788731, -21.8459663, 21.7778397
2: -5.8364959, 9.5153542, -5.8078890, 9.5083771, -15.3448715, 15.3232422
3: -10.2159939, 9.3246737, -10.1429844, 9.3147154, -19.5307064, 19.4676590
4: -7.4689107, 9.7003708, -7.4170961, 9.6984949, -17.1674061, 17.1174660

Time for backsubstitution: 1.20 seconds

### NS candidates at layer 1
type: B, layer: 1, pos: 11
type: A, layer: 1, pos: 11
type: B, layer: 1, pos: 34
type: A, layer: 1, pos: 34
type: B, layer: 1, pos: 12
type: A, layer: 1, pos: 12
type: B, layer: 1, pos: 9
type: A, layer: 1, pos: 30
type: B, layer: 1, pos: 47
type: B, layer: 1, pos: 42
type: B, layer: 1, pos: 4
type: A, layer: 1, pos: 4
type: B, layer: 1, pos: 16
type: A, layer: 1, pos: 16
type: A, layer: 1, pos: 39
type: A, layer: 1, pos: 31
type: B, layer: 1, pos: 23
type: A, layer: 1, pos: 23
type: B, layer: 1, pos: 39
type: B, layer: 1, pos: 33
type: A, layer: 1, pos: 33

Time for candidate selection: 0.08 seconds

### Candidate
type: B, layer: 1, pos: 11

### Candidate
type: A, layer: 1, pos: 11

### Candidate
type: B, layer: 1, pos: 34

### Candidate
type: A, layer: 1, pos: 34

### Candidate
type: B, layer: 1, pos: 12

### Candidate
type: A, layer: 1, pos: 12

### Candidate
type: B, layer: 1, pos: 9

### Candidate
type: A, layer: 1, pos: 30

## Relational analysis of NS_A1_A2_B1_B2_A2_B1_A1_A2_B1_A1

### Relational analysis result of NS_A1_A2_B1_B2_A2_B1_A1_A2_B1_A1
Status: Status.UNKNOWN
Output dim: 3, lower bound: -20.6037343, upper bound: 20.6041886
time: 0.68 seconds

## Relational analysis of NS_A1_A2_B1_B2_A2_B1_A1_A2_B1_A2

### Relational analysis result of NS_A1_A2_B1_B2_A2_B1_A1_A2_B1_A2
Status: Status.UNKNOWN
Output dim: 3, lower bound: -20.6037343, upper bound: 20.6040916
time: 0.87 seconds

## BFS NS instance: NS_A1_A2_B1_B2_A2_B1_A1_A2_B2

### Backsubstitution after applying NS history:
0: -2.9541757, 2.6261964, -3.2003164, 2.8472798, -5.8014550, 5.8265123
1: -11.6670923, 10.1961460, -12.6413479, 11.0089865, -22.6760788, 22.8374920
2: -5.8364959, 9.5153542, -6.3340435, 10.3492575, -16.1857529, 15.8493977
3: -10.2159939, 9.3246737, -11.0609941, 10.0579910, -20.2739830, 20.3856678
4: -7.4689107, 9.7003708, -8.0821466, 10.5208540, -17.9897652, 17.7825165

Time for backsubstitution: 1.17 seconds

### NS candidates at layer 1
type: A, layer: 1, pos: 11
type: B, layer: 1, pos: 11
type: B, layer: 1, pos: 34
type: A, layer: 1, pos: 34
type: B, layer: 1, pos: 12
type: A, layer: 1, pos: 12
type: B, layer: 1, pos: 9
type: A, layer: 1, pos: 30
type: A, layer: 1, pos: 31
type: B, layer: 1, pos: 4
type: B, layer: 1, pos: 16
type: B, layer: 1, pos: 42
type: A, layer: 1, pos: 16
type: B, layer: 1, pos: 39
type: A, layer: 1, pos: 4
type: A, layer: 1, pos: 39
type: B, layer: 1, pos: 23
type: A, layer: 1, pos: 23
type: A, layer: 1, pos: 33
type: B, layer: 1, pos: 33
type: B, layer: 1, pos: 47

Time for candidate selection: 0.11 seconds

### Candidate
type: A, layer: 1, pos: 11

### Candidate
type: B, layer: 1, pos: 11

### Candidate
type: B, layer: 1, pos: 34

### Candidate
type: A, layer: 1, pos: 34

### Candidate
type: B, layer: 1, pos: 12

### Candidate
type: A, layer: 1, pos: 12

### Candidate
type: B, layer: 1, pos: 9

### Candidate
type: A, layer: 1, pos: 30

## Relational analysis of NS_A1_A2_B1_B2_A2_B1_A1_A2_B2_A1

### Relational analysis result of NS_A1_A2_B1_B2_A2_B1_A1_A2_B2_A1
Status: Status.UNKNOWN
Output dim: 3, lower bound: -20.6037342, upper bound: 20.6039453
time: 0.62 seconds

## Relational analysis of NS_A1_A2_B1_B2_A2_B1_A1_A2_B2_A2

### Relational analysis result of NS_A1_A2_B1_B2_A2_B1_A1_A2_B2_A2
Status: Status.UNKNOWN
Output dim: 3, lower bound: -20.6037342, upper bound: 20.6039190
time: 0.91 seconds

## BFS NS instance: NS_A1_A2_B1_B2_A2_B1_A2_B1_B1

### Backsubstitution after applying NS history:
0: -3.5902476, 3.1974847, -2.8702092, 2.5531759, -6.1434236, 6.0676937
1: -14.1819506, 12.3807278, -11.3381014, 9.8738308, -24.0557823, 23.7188263
2: -7.0806346, 11.6148405, -5.6711969, 9.3008661, -16.3814983, 17.2860374
3: -12.3865166, 11.3239698, -9.9279318, 9.0253582, -21.4118748, 21.2519016
4: -9.0666447, 11.8067245, -7.2295156, 9.4379730, -18.5046177, 19.0362358

Time for backsubstitution: 1.08 seconds

### NS candidates at layer 1
type: B, layer: 1, pos: 11
type: A, layer: 1, pos: 11
type: A, layer: 1, pos: 34
type: B, layer: 1, pos: 34
type: A, layer: 1, pos: 30
type: B, layer: 1, pos: 31
type: A, layer: 1, pos: 31
type: A, layer: 1, pos: 12
type: B, layer: 1, pos: 12
type: B, layer: 1, pos: 9
type: A, layer: 1, pos: 39
type: B, layer: 1, pos: 39
type: B, layer: 1, pos: 42
type: A, layer: 1, pos: 23
type: B, layer: 1, pos: 23
type: A, layer: 1, pos: 4
type: A, layer: 1, pos: 16
type: A, layer: 1, pos: 47
type: B, layer: 1, pos: 16
type: A, layer: 1, pos: 33
type: B, layer: 1, pos: 33

Time for candidate selection: 0.08 seconds

### Candidate
type: B, layer: 1, pos: 11

### Candidate
type: A, layer: 1, pos: 11

### Candidate
type: A, layer: 1, pos: 34

### Candidate
type: B, layer: 1, pos: 34

### Candidate
type: A, layer: 1, pos: 30

## Relational analysis of NS_A1_A2_B1_B2_A2_B1_A2_B1_B1_A1

### Relational analysis result of NS_A1_A2_B1_B2_A2_B1_A2_B1_B1_A1
Status: Status.VERIFIED
Output dim: 3, lower bound: -20.6028402, upper bound: 20.5964160
time: 0.74 seconds

## Relational analysis of NS_A1_A2_B1_B2_A2_B1_A2_B1_B1_A2

### Relational analysis result of NS_A1_A2_B1_B2_A2_B1_A2_B1_B1_A2
Status: Status.VERIFIED
Output dim: 3, lower bound: -20.6028402, upper bound: 20.5964160
time: 0.78 seconds

## BFS NS instance: NS_A1_A2_B1_B2_A2_B1_A2_B1_B2

### Backsubstitution after applying NS history:
0: -3.5902476, 3.1974847, -2.7827682, 2.4748209, -6.0650682, 5.9802527
1: -14.1819506, 12.3807278, -10.9915371, 9.5970364, -23.7789879, 23.3722649
2: -7.0806346, 11.6148405, -5.5127335, 8.9621267, -16.0427589, 17.1275749
3: -12.3865166, 11.3239698, -9.6334667, 8.7718945, -21.1584091, 20.9574356
4: -9.0666447, 11.8067245, -7.0310092, 9.1374693, -18.2041130, 18.8377228

Time for backsubstitution: 1.08 seconds

### NS candidates at layer 1
type: B, layer: 1, pos: 11
type: A, layer: 1, pos: 11
type: A, layer: 1, pos: 34
type: B, layer: 1, pos: 34
type: A, layer: 1, pos: 12
type: B, layer: 1, pos: 12
type: A, layer: 1, pos: 30
type: B, layer: 1, pos: 9
type: A, layer: 1, pos: 31
type: B, layer: 1, pos: 31
type: B, layer: 1, pos: 42
type: A, layer: 1, pos: 4
type: A, layer: 1, pos: 16
type: B, layer: 1, pos: 16
type: A, layer: 1, pos: 39
type: B, layer: 1, pos: 33
type: A, layer: 1, pos: 23
type: B, layer: 1, pos: 23
type: A, layer: 1, pos: 33
type: B, layer: 1, pos: 39
type: A, layer: 1, pos: 47

Time for candidate selection: 0.08 seconds

### Candidate
type: B, layer: 1, pos: 11

### Candidate
type: A, layer: 1, pos: 11

### Candidate
type: A, layer: 1, pos: 34

### Candidate
type: B, layer: 1, pos: 34

### Candidate
type: A, layer: 1, pos: 12

### Candidate
type: B, layer: 1, pos: 12

### Candidate
type: A, layer: 1, pos: 30

## Relational analysis of NS_A1_A2_B1_B2_A2_B1_A2_B1_B2_A1

### Relational analysis result of NS_A1_A2_B1_B2_A2_B1_A2_B1_B2_A1
Status: Status.UNKNOWN
Output dim: 3, lower bound: -20.6033251, upper bound: 20.6002121
time: 0.66 seconds

## Relational analysis of NS_A1_A2_B1_B2_A2_B1_A2_B1_B2_A2

### Relational analysis result of NS_A1_A2_B1_B2_A2_B1_A2_B1_B2_A2
Status: Status.UNKNOWN
Output dim: 3, lower bound: -20.6033251, upper bound: 20.6002121
time: 0.67 seconds

## BFS NS instance: NS_A1_A2_B1_B2_A2_B1_A2_B2_A1

### Backsubstitution after applying NS history:
0: -3.3409936, 2.9953606, -3.0771875, 2.7368279, -6.0778217, 6.0725474
1: -13.1568880, 11.6057796, -12.1525078, 10.6178522, -23.7747383, 23.7582836
2: -6.5867920, 10.9051161, -6.0879107, 9.9097996, -16.4965916, 16.9930229
3: -11.4867058, 10.6629086, -10.6401510, 9.7027712, -21.1894760, 21.3030586
4: -8.4299479, 11.1081009, -7.7787948, 10.0979300, -18.5278778, 18.8868961

Time for backsubstitution: 1.05 seconds

### NS candidates at layer 1
type: B, layer: 1, pos: 11
type: A, layer: 1, pos: 11
type: A, layer: 1, pos: 34
type: A, layer: 1, pos: 47
type: B, layer: 1, pos: 34
type: A, layer: 1, pos: 12
type: B, layer: 1, pos: 12
type: B, layer: 1, pos: 9
type: A, layer: 1, pos: 4
type: B, layer: 1, pos: 42
type: A, layer: 1, pos: 33
type: B, layer: 1, pos: 39
type: A, layer: 1, pos: 16
type: B, layer: 1, pos: 31
type: B, layer: 1, pos: 16
type: A, layer: 1, pos: 39
type: A, layer: 1, pos: 31
type: A, layer: 1, pos: 23
type: B, layer: 1, pos: 23
type: B, layer: 1, pos: 33
type: B, layer: 1, pos: 47

Time for candidate selection: 0.09 seconds

### Candidate
type: B, layer: 1, pos: 11

### Candidate
type: A, layer: 1, pos: 11

### Candidate
type: A, layer: 1, pos: 34

### Candidate
type: A, layer: 1, pos: 47

### Candidate
type: B, layer: 1, pos: 34

### Candidate
type: A, layer: 1, pos: 12

### Candidate
type: B, layer: 1, pos: 12

### Candidate
type: B, layer: 1, pos: 9

### Candidate
type: A, layer: 1, pos: 4

## Relational analysis of NS_A1_A2_B1_B2_A2_B1_A2_B2_A1_A1

### Relational analysis result of NS_A1_A2_B1_B2_A2_B1_A2_B2_A1_A1
Status: Status.UNKNOWN
Output dim: 3, lower bound: -20.6029630, upper bound: 20.6040652
time: 0.74 seconds

## Relational analysis of NS_A1_A2_B1_B2_A2_B1_A2_B2_A1_A2

### Relational analysis result of NS_A1_A2_B1_B2_A2_B1_A2_B2_A1_A2
Status: Status.UNKNOWN
Output dim: 3, lower bound: -20.6029630, upper bound: 20.6037493
time: 0.71 seconds

## BFS NS instance: NS_A1_A2_B1_B2_A2_B1_A2_B2_A2

### Backsubstitution after applying NS history:
0: -3.5319197, 3.1517057, -3.0771875, 2.7368279, -6.2687473, 6.2288933
1: -13.9454231, 12.2059650, -12.1525078, 10.6178522, -24.5632744, 24.3584690
2: -6.9684982, 11.4514275, -6.0879107, 9.9097996, -16.8782978, 17.5393333
3: -12.1795673, 11.1709251, -10.6401510, 9.7027712, -21.8823376, 21.8110771
4: -8.9091740, 11.6463661, -7.7787948, 10.0979300, -19.0071030, 19.4251614

Time for backsubstitution: 1.10 seconds

### NS candidates at layer 1
type: B, layer: 1, pos: 11
type: A, layer: 1, pos: 11
type: A, layer: 1, pos: 34
type: B, layer: 1, pos: 34
type: B, layer: 1, pos: 47
type: A, layer: 1, pos: 12
type: B, layer: 1, pos: 12
type: B, layer: 1, pos: 9
type: A, layer: 1, pos: 4
type: B, layer: 1, pos: 42
type: B, layer: 1, pos: 16
type: A, layer: 1, pos: 16
type: A, layer: 1, pos: 39
type: B, layer: 1, pos: 33
type: B, layer: 1, pos: 31
type: A, layer: 1, pos: 31
type: A, layer: 1, pos: 23
type: A, layer: 1, pos: 47
type: B, layer: 1, pos: 39
type: B, layer: 1, pos: 23
type: A, layer: 1, pos: 33

Time for candidate selection: 0.10 seconds

### Candidate
type: B, layer: 1, pos: 11

### Candidate
type: A, layer: 1, pos: 11

### Candidate
type: A, layer: 1, pos: 34

### Candidate
type: B, layer: 1, pos: 34

### Candidate
type: B, layer: 1, pos: 47

### Candidate
type: A, layer: 1, pos: 12

### Candidate
type: B, layer: 1, pos: 12

### Candidate
type: B, layer: 1, pos: 9

### Candidate
type: A, layer: 1, pos: 4

## Relational analysis of NS_A1_A2_B1_B2_A2_B1_A2_B2_A2_A1

### Relational analysis result of NS_A1_A2_B1_B2_A2_B1_A2_B2_A2_A1
Status: Status.UNKNOWN
Output dim: 3, lower bound: -20.6029630, upper bound: 20.6039567
time: 0.82 seconds

## Relational analysis of NS_A1_A2_B1_B2_A2_B1_A2_B2_A2_A2

### Relational analysis result of NS_A1_A2_B1_B2_A2_B1_A2_B2_A2_A2
Status: Status.UNKNOWN
Output dim: 3, lower bound: -20.6029630, upper bound: 20.6040673
time: 0.84 seconds

## BFS NS instance: NS_A1_A2_B2_B1_B2_B1_A2_A1_B1

### Backsubstitution after applying NS history:
0: -2.9960527, 2.6623325, -3.1513410, 2.8146360, -5.8106875, 5.8136716
1: -11.8343382, 10.3386192, -12.4139872, 10.8930225, -22.7273598, 22.7526054
2: -5.9199672, 9.6447563, -6.2274179, 10.2425604, -16.1625271, 15.8721743
3: -10.3606873, 9.4523716, -10.8576565, 9.9981375, -20.3588257, 20.3100166
4: -7.5751920, 9.8309212, -7.9622679, 10.4328938, -18.0080814, 17.7931900

Time for backsubstitution: 1.03 seconds

### NS candidates at layer 1
type: A, layer: 1, pos: 11
type: B, layer: 1, pos: 11
type: B, layer: 1, pos: 34
type: A, layer: 1, pos: 34
type: B, layer: 1, pos: 12
type: A, layer: 1, pos: 12
type: B, layer: 1, pos: 9
type: A, layer: 1, pos: 31
type: B, layer: 1, pos: 31
type: B, layer: 1, pos: 42
type: A, layer: 1, pos: 39
type: B, layer: 1, pos: 33
type: B, layer: 1, pos: 16
type: B, layer: 1, pos: 39
type: A, layer: 1, pos: 16
type: A, layer: 1, pos: 4
type: A, layer: 1, pos: 23
type: B, layer: 1, pos: 23
type: A, layer: 1, pos: 33
type: A, layer: 1, pos: 47
type: A, layer: 1, pos: 30

Time for candidate selection: 0.08 seconds

### Candidate
type: A, layer: 1, pos: 11

### Candidate
type: B, layer: 1, pos: 11

### Candidate
type: B, layer: 1, pos: 34

### Candidate
type: A, layer: 1, pos: 34

### Candidate
type: B, layer: 1, pos: 12

### Candidate
type: A, layer: 1, pos: 12

### Candidate
type: B, layer: 1, pos: 9

### Candidate
type: A, layer: 1, pos: 31

## Relational analysis of NS_A1_A2_B2_B1_B2_B1_A2_A1_B1_A1

### Relational analysis result of NS_A1_A2_B2_B1_B2_B1_A2_A1_B1_A1
Status: Status.UNKNOWN
Output dim: 3, lower bound: -20.6039741, upper bound: 20.6026981
time: 0.70 seconds

## Relational analysis of NS_A1_A2_B2_B1_B2_B1_A2_A1_B1_A2

### Relational analysis result of NS_A1_A2_B2_B1_B2_B1_A2_A1_B1_A2
Status: Status.UNKNOWN
Output dim: 3, lower bound: -20.6037295, upper bound: 20.6026952
time: 0.71 seconds

## BFS NS instance: NS_A1_A2_B2_B1_B2_B1_A2_A1_B2

### Backsubstitution after applying NS history:
0: -2.9960527, 2.6623325, -3.5829484, 3.2091122, -6.2051649, 6.2452798
1: -11.8343382, 10.3386192, -14.1194181, 12.3827600, -24.2170944, 24.4580383
2: -5.9199672, 9.6447563, -7.0389814, 11.7711287, -17.6910915, 16.6837387
3: -10.3606873, 9.4523716, -12.3018827, 11.3730946, -21.7337818, 21.7542439
4: -7.5751920, 9.8309212, -9.0094128, 11.9351234, -19.5103149, 18.8403320

Time for backsubstitution: 1.04 seconds

### NS candidates at layer 1
type: A, layer: 1, pos: 11
type: B, layer: 1, pos: 11
type: A, layer: 1, pos: 34
type: B, layer: 1, pos: 34
type: A, layer: 1, pos: 47
type: B, layer: 1, pos: 12
type: B, layer: 1, pos: 9
type: A, layer: 1, pos: 12
type: B, layer: 1, pos: 31
type: B, layer: 1, pos: 42
type: A, layer: 1, pos: 31
type: B, layer: 1, pos: 39
type: A, layer: 1, pos: 4
type: B, layer: 1, pos: 16
type: A, layer: 1, pos: 33
type: A, layer: 1, pos: 16
type: B, layer: 1, pos: 23
type: B, layer: 1, pos: 33
type: A, layer: 1, pos: 39
type: A, layer: 1, pos: 23
type: A, layer: 1, pos: 30

Time for candidate selection: 0.08 seconds

### Candidate
type: A, layer: 1, pos: 11

### Candidate
type: B, layer: 1, pos: 11

### Candidate
type: A, layer: 1, pos: 34

### Candidate
type: B, layer: 1, pos: 34

### Candidate
type: A, layer: 1, pos: 47

## Relational analysis of NS_A1_A2_B2_B1_B2_B1_A2_A1_B2_A1
Optimization infeasible because this subproblem isn't reachable.

### Candidate
type: B, layer: 1, pos: 12

### Candidate
type: B, layer: 1, pos: 9

### Candidate
type: A, layer: 1, pos: 12

### Candidate
type: B, layer: 1, pos: 31

## Relational analysis of NS_A1_A2_B2_B1_B2_B1_A2_A1_B2_B1
Optimization infeasible because this subproblem isn't reachable.

### Candidate
type: B, layer: 1, pos: 42

## Relational analysis of NS_A1_A2_B2_B1_B2_B1_A2_A1_B2_B1
Optimization infeasible because this subproblem isn't reachable.

### Candidate
type: A, layer: 1, pos: 31

### Candidate
type: B, layer: 1, pos: 39

## Relational analysis of NS_A1_A2_B2_B1_B2_B1_A2_A1_B2_B1

### Relational analysis result of NS_A1_A2_B2_B1_B2_B1_A2_A1_B2_B1
Status: Status.UNKNOWN
Output dim: 3, lower bound: -20.6039741, upper bound: 20.6026423
time: 0.69 seconds

## Relational analysis of NS_A1_A2_B2_B1_B2_B1_A2_A1_B2_B2
Optimization infeasible because this subproblem isn't reachable.

### Candidate
type: A, layer: 1, pos: 4

## Relational analysis of NS_A1_A2_B2_B1_B2_B1_A2_A1_B2_A1

### Relational analysis result of NS_A1_A2_B2_B1_B2_B1_A2_A1_B2_A1
Status: Status.VERIFIED
Output dim: 3, lower bound: -20.6000489, upper bound: 20.6023030
time: 0.65 seconds

## Relational analysis of NS_A1_A2_B2_B1_B2_B1_A2_A1_B2_A2

### Relational analysis result of NS_A1_A2_B2_B1_B2_B1_A2_A1_B2_A2
Status: Status.VERIFIED
Output dim: 3, lower bound: -20.6000489, upper bound: 20.6026981
time: 0.90 seconds

## BFS NS instance: NS_A1_A2_B2_B1_B2_B1_A2_A2_A1

### Backsubstitution after applying NS history:
0: -3.3850455, 3.0251527, -3.2224360, 2.8752885, -6.2603335, 6.2475877
1: -13.3516960, 11.7186394, -12.6998873, 11.1310387, -24.4827347, 24.4185238
2: -6.6723394, 10.9979820, -6.3695655, 10.4611368, -17.1334763, 17.3675480
3: -11.6664848, 10.7302942, -11.1067505, 10.2140846, -21.8805676, 21.8370419
4: -8.5420465, 11.1814547, -8.1406956, 10.6549606, -19.1970024, 19.3221474

Time for backsubstitution: 1.11 seconds

### NS candidates at layer 1
type: B, layer: 1, pos: 11
type: A, layer: 1, pos: 11
type: B, layer: 1, pos: 34
type: A, layer: 1, pos: 34
type: A, layer: 1, pos: 12
type: B, layer: 1, pos: 12
type: B, layer: 1, pos: 9
type: A, layer: 1, pos: 39
type: B, layer: 1, pos: 42
type: B, layer: 1, pos: 39
type: B, layer: 1, pos: 24
type: B, layer: 1, pos: 16
type: B, layer: 1, pos: 31
type: B, layer: 1, pos: 33
type: A, layer: 1, pos: 4
type: A, layer: 1, pos: 16
type: A, layer: 1, pos: 23
type: B, layer: 1, pos: 23
type: A, layer: 1, pos: 33
type: A, layer: 1, pos: 47
type: A, layer: 1, pos: 30

Time for candidate selection: 0.08 seconds

### Candidate
type: B, layer: 1, pos: 11

### Candidate
type: A, layer: 1, pos: 11

### Candidate
type: B, layer: 1, pos: 34

### Candidate
type: A, layer: 1, pos: 34

### Candidate
type: A, layer: 1, pos: 12

### Candidate
type: B, layer: 1, pos: 12

### Candidate
type: B, layer: 1, pos: 9

### Candidate
type: A, layer: 1, pos: 39

## Relational analysis of NS_A1_A2_B2_B1_B2_B1_A2_A2_A1_A1
Optimization infeasible because this subproblem isn't reachable.

### Candidate
type: B, layer: 1, pos: 42

## Relational analysis of NS_A1_A2_B2_B1_B2_B1_A2_A2_A1_B1
Optimization infeasible because this subproblem isn't reachable.

### Candidate
type: B, layer: 1, pos: 39

### Candidate
type: B, layer: 1, pos: 24

## Relational analysis of NS_A1_A2_B2_B1_B2_B1_A2_A2_A1_B1

### Relational analysis result of NS_A1_A2_B2_B1_B2_B1_A2_A2_A1_B1
Status: Status.UNKNOWN
Output dim: 3, lower bound: -20.6035637, upper bound: 20.6024183
time: 0.79 seconds

## Relational analysis of NS_A1_A2_B2_B1_B2_B1_A2_A2_A1_B2

### Relational analysis result of NS_A1_A2_B2_B1_B2_B1_A2_A2_A1_B2
Status: Status.UNKNOWN
Output dim: 3, lower bound: -20.6035637, upper bound: 20.6024183
time: 0.65 seconds

## BFS NS instance: NS_A1_A2_B2_B1_B2_B1_A2_A2_A2

### Backsubstitution after applying NS history:
0: -3.4960272, 3.1204393, -3.2224360, 2.8752885, -6.3713136, 6.3428741
1: -13.8108044, 12.0488758, -12.6998873, 11.1310387, -24.9418430, 24.7487640
2: -6.8975406, 11.3577156, -6.3695655, 10.4611368, -17.3586750, 17.7272816
3: -12.0608492, 11.0254507, -11.1067505, 10.2140846, -22.2749329, 22.1322002
4: -8.8241854, 11.5409737, -8.1406956, 10.6549606, -19.4791431, 19.6816673

Time for backsubstitution: 1.06 seconds

### NS candidates at layer 1
type: B, layer: 1, pos: 11
type: A, layer: 1, pos: 11
type: B, layer: 1, pos: 34
type: A, layer: 1, pos: 34
type: A, layer: 1, pos: 12
type: B, layer: 1, pos: 12
type: B, layer: 1, pos: 9
type: A, layer: 1, pos: 39
type: B, layer: 1, pos: 42
type: B, layer: 1, pos: 39
type: B, layer: 1, pos: 24
type: B, layer: 1, pos: 16
type: B, layer: 1, pos: 31
type: A, layer: 1, pos: 4
type: B, layer: 1, pos: 33
type: A, layer: 1, pos: 16
type: A, layer: 1, pos: 23
type: B, layer: 1, pos: 23
type: A, layer: 1, pos: 33
type: A, layer: 1, pos: 47
type: A, layer: 1, pos: 30

Time for candidate selection: 0.08 seconds

### Candidate
type: B, layer: 1, pos: 11

### Candidate
type: A, layer: 1, pos: 11

### Candidate
type: B, layer: 1, pos: 34

### Candidate
type: A, layer: 1, pos: 34

### Candidate
type: A, layer: 1, pos: 12

### Candidate
type: B, layer: 1, pos: 12

### Candidate
type: B, layer: 1, pos: 9

### Candidate
type: A, layer: 1, pos: 39

## Relational analysis of NS_A1_A2_B2_B1_B2_B1_A2_A2_A2_A1
Optimization infeasible because this subproblem isn't reachable.

### Candidate
type: B, layer: 1, pos: 42

## Relational analysis of NS_A1_A2_B2_B1_B2_B1_A2_A2_A2_B1
Optimization infeasible because this subproblem isn't reachable.

### Candidate
type: B, layer: 1, pos: 39

### Candidate
type: B, layer: 1, pos: 24

## Relational analysis of NS_A1_A2_B2_B1_B2_B1_A2_A2_A2_B1

### Relational analysis result of NS_A1_A2_B2_B1_B2_B1_A2_A2_A2_B1
Status: Status.UNKNOWN
Output dim: 3, lower bound: -20.6035622, upper bound: 20.6024620
time: 0.68 seconds

## Relational analysis of NS_A1_A2_B2_B1_B2_B1_A2_A2_A2_B2

### Relational analysis result of NS_A1_A2_B2_B1_B2_B1_A2_A2_A2_B2
Status: Status.UNKNOWN
Output dim: 3, lower bound: -20.6035622, upper bound: 20.6024620
time: 1.28 seconds

## BFS NS instance: NS_A1_A2_B2_B1_B2_B2_A2_B1_A1

### Backsubstitution after applying NS history:
0: -2.9960527, 2.6623325, -3.6223240, 3.2366309, -6.2326832, 6.2846551
1: -11.8343382, 10.3386192, -14.2748737, 12.5350981, -24.3694344, 24.6134930
2: -5.9199672, 9.6447563, -7.1335411, 11.7950106, -17.7149773, 16.7782974
3: -10.3606873, 9.4523716, -12.4678802, 11.4905529, -21.8512402, 21.9202442
4: -7.5751920, 9.8309212, -9.1481209, 11.9893732, -19.5645638, 18.9790421

Time for backsubstitution: 1.22 seconds

### NS candidates at layer 1
type: A, layer: 1, pos: 11
type: B, layer: 1, pos: 11
type: B, layer: 1, pos: 34
type: A, layer: 1, pos: 34
type: B, layer: 1, pos: 12
type: A, layer: 1, pos: 12
type: B, layer: 1, pos: 9
type: A, layer: 1, pos: 31
type: B, layer: 1, pos: 42
type: B, layer: 1, pos: 31
type: B, layer: 1, pos: 39
type: B, layer: 1, pos: 16
type: A, layer: 1, pos: 39
type: B, layer: 1, pos: 33
type: A, layer: 1, pos: 16
type: B, layer: 1, pos: 23
type: A, layer: 1, pos: 33
type: A, layer: 1, pos: 4
type: A, layer: 1, pos: 23
type: A, layer: 1, pos: 47
type: A, layer: 1, pos: 30

Time for candidate selection: 0.09 seconds

### Candidate
type: A, layer: 1, pos: 11

### Candidate
type: B, layer: 1, pos: 11

### Candidate
type: B, layer: 1, pos: 34

### Candidate
type: A, layer: 1, pos: 34

### Candidate
type: B, layer: 1, pos: 12

### Candidate
type: A, layer: 1, pos: 12

### Candidate
type: B, layer: 1, pos: 9

## Relational analysis of NS_A1_A2_B2_B1_B2_B2_A2_B1_A1_B1
Optimization infeasible because this subproblem isn't reachable.

### Candidate
type: A, layer: 1, pos: 31

## Relational analysis of NS_A1_A2_B2_B1_B2_B2_A2_B1_A1_A1

### Relational analysis result of NS_A1_A2_B2_B1_B2_B2_A2_B1_A1_A1
Status: Status.UNKNOWN
Output dim: 3, lower bound: -20.6041077, upper bound: 20.6036961
time: 0.72 seconds

## Relational analysis of NS_A1_A2_B2_B1_B2_B2_A2_B1_A1_A2

### Relational analysis result of NS_A1_A2_B2_B1_B2_B2_A2_B1_A1_A2
Status: Status.UNKNOWN
Output dim: 3, lower bound: -20.6038927, upper bound: 20.6037086
time: 0.73 seconds

## BFS NS instance: NS_A1_A2_B2_B1_B2_B2_A2_B1_A2

### Backsubstitution after applying NS history:
0: -3.5902476, 3.1974847, -3.6223240, 3.2366309, -6.8268785, 6.8198080
1: -14.1819506, 12.3807278, -14.2748737, 12.5350981, -26.7170448, 26.6555996
2: -7.0806346, 11.6148405, -7.1335411, 11.7950106, -18.8756447, 18.7483826
3: -12.3865166, 11.3239698, -12.4678802, 11.4905529, -23.8770657, 23.7918472
4: -9.0666447, 11.8067245, -9.1481209, 11.9893732, -21.0560188, 20.9548435

Time for backsubstitution: 1.05 seconds

### NS candidates at layer 1
type: B, layer: 1, pos: 11
type: A, layer: 1, pos: 11
type: B, layer: 1, pos: 34
type: A, layer: 1, pos: 34
type: B, layer: 1, pos: 12
type: A, layer: 1, pos: 12
type: B, layer: 1, pos: 9
type: A, layer: 1, pos: 39
type: B, layer: 1, pos: 42
type: A, layer: 1, pos: 4
type: B, layer: 1, pos: 16
type: A, layer: 1, pos: 31
type: A, layer: 1, pos: 23
type: B, layer: 1, pos: 33
type: B, layer: 1, pos: 31
type: B, layer: 1, pos: 39
type: A, layer: 1, pos: 16
type: B, layer: 1, pos: 23
type: A, layer: 1, pos: 33
type: A, layer: 1, pos: 47
type: A, layer: 1, pos: 30

Time for candidate selection: 0.08 seconds

### Candidate
type: B, layer: 1, pos: 11

### Candidate
type: A, layer: 1, pos: 11

### Candidate
type: B, layer: 1, pos: 34

### Candidate
type: A, layer: 1, pos: 34

### Candidate
type: B, layer: 1, pos: 12

### Candidate
type: A, layer: 1, pos: 12

### Candidate
type: B, layer: 1, pos: 9

## Relational analysis of NS_A1_A2_B2_B1_B2_B2_A2_B1_A2_B1
Optimization infeasible because this subproblem isn't reachable.

### Candidate
type: A, layer: 1, pos: 39

## Relational analysis of NS_A1_A2_B2_B1_B2_B2_A2_B1_A2_A1
Optimization infeasible because this subproblem isn't reachable.

### Candidate
type: B, layer: 1, pos: 42

## Relational analysis of NS_A1_A2_B2_B1_B2_B2_A2_B1_A2_B1

### Relational analysis result of NS_A1_A2_B2_B1_B2_B2_A2_B1_A2_B1
Status: Status.UNKNOWN
Output dim: 3, lower bound: -20.6039654, upper bound: 20.6017344
time: 0.66 seconds

## Relational analysis of NS_A1_A2_B2_B1_B2_B2_A2_B1_A2_B2

### Relational analysis result of NS_A1_A2_B2_B1_B2_B2_A2_B1_A2_B2
Status: Status.UNKNOWN
Output dim: 3, lower bound: -20.6039654, upper bound: 20.6037002
time: 0.65 seconds

## BFS NS instance: NS_A1_A2_B2_B1_B2_B2_A2_B2_A1

### Backsubstitution after applying NS history:
0: -2.9960527, 2.6623325, -4.1345482, 3.7068682, -6.7029209, 6.7968802
1: -11.8343382, 10.3386192, -16.2946053, 14.3344345, -26.1687737, 26.6332245
2: -5.9199672, 9.6447563, -8.1139383, 13.6177435, -19.5377102, 17.7586937
3: -10.3606873, 9.4523716, -14.1805086, 13.1594715, -23.5201588, 23.6328754
4: -7.5751920, 9.8309212, -10.3973160, 13.7970152, -21.3722057, 20.2282372

Time for backsubstitution: 1.05 seconds

### NS candidates at layer 1
type: A, layer: 1, pos: 11
type: B, layer: 1, pos: 34
type: A, layer: 1, pos: 34
type: A, layer: 1, pos: 47
type: B, layer: 1, pos: 12
type: B, layer: 1, pos: 9
type: B, layer: 1, pos: 11
type: B, layer: 1, pos: 31
type: A, layer: 1, pos: 12
type: A, layer: 1, pos: 31
type: B, layer: 1, pos: 42
type: B, layer: 1, pos: 39
type: A, layer: 1, pos: 4
type: A, layer: 1, pos: 33
type: B, layer: 1, pos: 23
type: B, layer: 1, pos: 16
type: A, layer: 1, pos: 16
type: B, layer: 1, pos: 33
type: A, layer: 1, pos: 39
type: A, layer: 1, pos: 23
type: A, layer: 1, pos: 30

Time for candidate selection: 0.11 seconds

### Candidate
type: A, layer: 1, pos: 11

### Candidate
type: B, layer: 1, pos: 34

### Candidate
type: A, layer: 1, pos: 34

### Candidate
type: A, layer: 1, pos: 47

## Relational analysis of NS_A1_A2_B2_B1_B2_B2_A2_B2_A1_A1
Optimization infeasible because this subproblem isn't reachable.

### Candidate
type: B, layer: 1, pos: 12

### Candidate
type: B, layer: 1, pos: 9

## Relational analysis of NS_A1_A2_B2_B1_B2_B2_A2_B2_A1_B1
Optimization infeasible because this subproblem isn't reachable.

### Candidate
type: B, layer: 1, pos: 11

### Candidate
type: B, layer: 1, pos: 31

## Relational analysis of NS_A1_A2_B2_B1_B2_B2_A2_B2_A1_B1
Optimization infeasible because this subproblem isn't reachable.

### Candidate
type: A, layer: 1, pos: 12

### Candidate
type: A, layer: 1, pos: 31

### Candidate
type: B, layer: 1, pos: 42

## Relational analysis of NS_A1_A2_B2_B1_B2_B2_A2_B2_A1_B1

### Relational analysis result of NS_A1_A2_B2_B1_B2_B2_A2_B2_A1_B1
Status: Status.UNKNOWN
Output dim: 3, lower bound: -20.6034656, upper bound: 20.5995481
time: 0.82 seconds

## Relational analysis of NS_A1_A2_B2_B1_B2_B2_A2_B2_A1_B2

### Relational analysis result of NS_A1_A2_B2_B1_B2_B2_A2_B2_A1_B2
Status: Status.UNKNOWN
Output dim: 3, lower bound: -20.6034656, upper bound: 20.6002504
time: 0.80 seconds

## BFS NS instance: NS_A1_A2_B2_B1_B2_B2_A2_B2_A2

### Backsubstitution after applying NS history:
0: -3.5902476, 3.1974847, -4.1345482, 3.7068682, -7.2971158, 7.3320332
1: -14.1819506, 12.3807278, -16.2946053, 14.3344345, -28.5163841, 28.6753311
2: -7.0806346, 11.6148405, -8.1139383, 13.6177435, -20.6983776, 19.7287788
3: -12.3865166, 11.3239698, -14.1805086, 13.1594715, -25.5459843, 25.5044765
4: -9.0666447, 11.8067245, -10.3973160, 13.7970152, -22.8636589, 22.2040367

Time for backsubstitution: 1.44 seconds

### NS candidates at layer 1
type: A, layer: 1, pos: 11
type: B, layer: 1, pos: 11
type: A, layer: 1, pos: 34
type: B, layer: 1, pos: 34
type: B, layer: 1, pos: 12
type: B, layer: 1, pos: 9
type: A, layer: 1, pos: 12
type: A, layer: 1, pos: 47
type: B, layer: 1, pos: 42
type: A, layer: 1, pos: 4
type: B, layer: 1, pos: 16
type: B, layer: 1, pos: 39
type: A, layer: 1, pos: 16
type: A, layer: 1, pos: 31
type: B, layer: 1, pos: 31
type: B, layer: 1, pos: 33
type: B, layer: 1, pos: 23
type: A, layer: 1, pos: 39
type: A, layer: 1, pos: 33
type: A, layer: 1, pos: 23
type: A, layer: 1, pos: 30

Time for candidate selection: 0.10 seconds

### Candidate
type: A, layer: 1, pos: 11

### Candidate
type: B, layer: 1, pos: 11

### Candidate
type: A, layer: 1, pos: 34

### Candidate
type: B, layer: 1, pos: 34

### Candidate
type: B, layer: 1, pos: 12

### Candidate
type: B, layer: 1, pos: 9

## Relational analysis of NS_A1_A2_B2_B1_B2_B2_A2_B2_A2_B1
Optimization infeasible because this subproblem isn't reachable.

### Candidate
type: A, layer: 1, pos: 12

### Candidate
type: A, layer: 1, pos: 47

## Relational analysis of NS_A1_A2_B2_B1_B2_B2_A2_B2_A2_A1
Optimization infeasible because this subproblem isn't reachable.

### Candidate
type: B, layer: 1, pos: 42

## Relational analysis of NS_A1_A2_B2_B1_B2_B2_A2_B2_A2_B1

### Relational analysis result of NS_A1_A2_B2_B1_B2_B2_A2_B2_A2_B1
Status: Status.UNKNOWN
Output dim: 3, lower bound: -20.6034656, upper bound: 20.5995144
time: 0.66 seconds

## Relational analysis of NS_A1_A2_B2_B1_B2_B2_A2_B2_A2_B2

### Relational analysis result of NS_A1_A2_B2_B1_B2_B2_A2_B2_A2_B2
Status: Status.UNKNOWN
Output dim: 3, lower bound: -20.6034656, upper bound: 20.6002210
time: 0.80 seconds

## BFS NS instance: NS_A1_A2_B2_B2_A2_B1_A1_A2_A1

### Backsubstitution after applying NS history:
0: -2.7367759, 2.4551795, -3.8540320, 3.4321170, -6.1688924, 6.3092117
1: -10.7669916, 9.5255136, -15.2264605, 13.2877932, -24.0547810, 24.7519684
2: -5.4099050, 8.9155455, -7.5972466, 12.4870577, -17.8969631, 16.5127926
3: -9.4317722, 8.7528687, -13.2969398, 12.1337633, -21.5655365, 22.0498085
4: -6.9135633, 9.1204062, -9.7240124, 12.6810350, -19.5945930, 18.8444176

Time for backsubstitution: 1.07 seconds

### NS candidates at layer 1
type: A, layer: 1, pos: 11
type: A, layer: 1, pos: 34
type: B, layer: 1, pos: 34
type: B, layer: 1, pos: 9
type: A, layer: 1, pos: 4
type: B, layer: 1, pos: 4
type: B, layer: 1, pos: 12
type: A, layer: 1, pos: 12
type: B, layer: 1, pos: 39
type: A, layer: 1, pos: 33
type: B, layer: 1, pos: 42
type: B, layer: 1, pos: 23
type: A, layer: 1, pos: 16
type: B, layer: 1, pos: 11
type: B, layer: 1, pos: 31
type: A, layer: 1, pos: 31
type: B, layer: 1, pos: 16
type: A, layer: 1, pos: 23
type: A, layer: 1, pos: 39
type: B, layer: 1, pos: 33
type: B, layer: 1, pos: 47

Time for candidate selection: 0.09 seconds

### Candidate
type: A, layer: 1, pos: 11

### Candidate
type: A, layer: 1, pos: 34

### Candidate
type: B, layer: 1, pos: 34

### Candidate
type: B, layer: 1, pos: 9

### Candidate
type: A, layer: 1, pos: 4

## Relational analysis of NS_A1_A2_B2_B2_A2_B1_A1_A2_A1_A1

### Relational analysis result of NS_A1_A2_B2_B2_A2_B1_A1_A2_A1_A1
Status: Status.UNKNOWN
Output dim: 3, lower bound: -20.6002121, upper bound: 20.6037242
time: 0.85 seconds

## Relational analysis of NS_A1_A2_B2_B2_A2_B1_A1_A2_A1_A2

### Relational analysis result of NS_A1_A2_B2_B2_A2_B1_A1_A2_A1_A2
Status: Status.UNKNOWN
Output dim: 3, lower bound: -20.6036829, upper bound: 20.6041185
time: 0.79 seconds

## BFS NS instance: NS_A1_A2_B2_B2_A2_B1_A1_A2_A2

### Backsubstitution after applying NS history:
0: -2.8930047, 2.5780895, -3.8540320, 3.4321170, -6.3251219, 6.4321213
1: -11.4196548, 10.0121975, -15.2264605, 13.2877932, -24.7074432, 25.2386551
2: -5.7184653, 9.3445778, -7.5972466, 12.4870577, -18.2055225, 16.9418221
3: -9.9995356, 9.1629286, -13.2969398, 12.1337633, -22.1332989, 22.4598694
4: -7.3038015, 9.5328236, -9.7240124, 12.6810350, -19.9848309, 19.2568302

Time for backsubstitution: 1.15 seconds

### NS candidates at layer 1
type: A, layer: 1, pos: 11
type: B, layer: 1, pos: 11
type: B, layer: 1, pos: 34
type: A, layer: 1, pos: 34
type: B, layer: 1, pos: 12
type: B, layer: 1, pos: 9
type: A, layer: 1, pos: 12
type: B, layer: 1, pos: 31
type: A, layer: 1, pos: 31
type: B, layer: 1, pos: 42
type: B, layer: 1, pos: 4
type: A, layer: 1, pos: 4
type: B, layer: 1, pos: 39
type: B, layer: 1, pos: 23
type: B, layer: 1, pos: 16
type: A, layer: 1, pos: 16
type: A, layer: 1, pos: 33
type: A, layer: 1, pos: 23
type: A, layer: 1, pos: 39
type: B, layer: 1, pos: 33
type: B, layer: 1, pos: 47

Time for candidate selection: 0.08 seconds

### Candidate
type: A, layer: 1, pos: 11

### Candidate
type: B, layer: 1, pos: 11

### Candidate
type: B, layer: 1, pos: 34

### Candidate
type: A, layer: 1, pos: 34

### Candidate
type: B, layer: 1, pos: 12

### Candidate
type: B, layer: 1, pos: 9

### Candidate
type: A, layer: 1, pos: 12

### Candidate
type: B, layer: 1, pos: 31

## Relational analysis of NS_A1_A2_B2_B2_A2_B1_A1_A2_A2_B1

### Relational analysis result of NS_A1_A2_B2_B2_A2_B1_A1_A2_A2_B1
Status: Status.UNKNOWN
Output dim: 3, lower bound: -20.6035983, upper bound: 20.6040746
time: 0.69 seconds

## Relational analysis of NS_A1_A2_B2_B2_A2_B1_A1_A2_A2_B2

### Relational analysis result of NS_A1_A2_B2_B2_A2_B1_A1_A2_A2_B2
Status: Status.UNKNOWN
Output dim: 3, lower bound: -20.6037272, upper bound: 20.6040054
time: 0.74 seconds

## BFS NS instance: NS_A1_A2_B2_B2_A2_B1_A2_B1_B1

### Backsubstitution after applying NS history:
0: -3.5902476, 3.1974847, -3.4100599, 3.0330069, -6.6232548, 6.6075444
1: -14.1819506, 12.3807278, -13.4700336, 11.7069979, -25.8889465, 25.8507576
2: -7.0806346, 11.6148405, -6.7193437, 11.0487976, -18.1294327, 18.3341846
3: -12.3865166, 11.3239698, -11.7811317, 10.6865797, -23.0730953, 23.1051006
4: -9.0666447, 11.8067245, -8.5973883, 11.2034492, -20.2700939, 20.4041080

Time for backsubstitution: 1.08 seconds

### NS candidates at layer 1
type: A, layer: 1, pos: 11
type: B, layer: 1, pos: 11
type: A, layer: 1, pos: 34
type: B, layer: 1, pos: 34
type: A, layer: 1, pos: 47
type: A, layer: 1, pos: 30
type: A, layer: 1, pos: 31
type: B, layer: 1, pos: 31
type: A, layer: 1, pos: 12
type: B, layer: 1, pos: 12
type: B, layer: 1, pos: 9
type: B, layer: 1, pos: 39
type: A, layer: 1, pos: 39
type: A, layer: 1, pos: 33
type: B, layer: 1, pos: 23
type: A, layer: 1, pos: 23
type: B, layer: 1, pos: 42
type: A, layer: 1, pos: 16
type: B, layer: 1, pos: 16
type: A, layer: 1, pos: 4
type: B, layer: 1, pos: 33

Time for candidate selection: 0.09 seconds

### Candidate
type: A, layer: 1, pos: 11

### Candidate
type: B, layer: 1, pos: 11

### Candidate
type: A, layer: 1, pos: 34

### Candidate
type: B, layer: 1, pos: 34

### Candidate
type: A, layer: 1, pos: 47

## Relational analysis of NS_A1_A2_B2_B2_A2_B1_A2_B1_B1_A1
Optimization infeasible because this subproblem isn't reachable.

### Candidate
type: A, layer: 1, pos: 30

## Relational analysis of NS_A1_A2_B2_B2_A2_B1_A2_B1_B1_A1

### Relational analysis result of NS_A1_A2_B2_B2_A2_B1_A2_B1_B1_A1
Status: Status.VERIFIED
Output dim: 3, lower bound: -20.6031079, upper bound: 20.5992681
time: 0.75 seconds

## Relational analysis of NS_A1_A2_B2_B2_A2_B1_A2_B1_B1_A2

### Relational analysis result of NS_A1_A2_B2_B2_A2_B1_A2_B1_B1_A2
Status: Status.VERIFIED
Output dim: 3, lower bound: -20.6031079, upper bound: 20.6001261
time: 0.68 seconds

## BFS NS instance: NS_A1_A2_B2_B2_A2_B1_A2_B1_B2

### Backsubstitution after applying NS history:
0: -3.5902476, 3.1974847, -3.3247313, 2.9603162, -6.5505638, 6.5222154
1: -14.1819506, 12.3807278, -13.1322536, 11.4495230, -25.6314735, 25.5129795
2: -7.0806346, 11.6148405, -6.5729632, 10.7548695, -17.8355045, 18.1878033
3: -12.3865166, 11.3239698, -11.4845371, 10.4657965, -22.8523102, 22.8085060
4: -9.0666447, 11.8067245, -8.3935518, 10.9394827, -20.0061264, 20.2002754

Time for backsubstitution: 1.33 seconds

### NS candidates at layer 1
type: B, layer: 1, pos: 11
type: A, layer: 1, pos: 11
type: A, layer: 1, pos: 34
type: B, layer: 1, pos: 34
type: A, layer: 1, pos: 30
type: A, layer: 1, pos: 12
type: B, layer: 1, pos: 12
type: B, layer: 1, pos: 9
type: B, layer: 1, pos: 31
type: A, layer: 1, pos: 31
type: B, layer: 1, pos: 42
type: A, layer: 1, pos: 4
type: A, layer: 1, pos: 16
type: B, layer: 1, pos: 16
type: A, layer: 1, pos: 39
type: A, layer: 1, pos: 23
type: B, layer: 1, pos: 39
type: B, layer: 1, pos: 23
type: B, layer: 1, pos: 33
type: A, layer: 1, pos: 33
type: A, layer: 1, pos: 47

Time for candidate selection: 0.09 seconds

### Candidate
type: B, layer: 1, pos: 11

### Candidate
type: A, layer: 1, pos: 11

### Candidate
type: A, layer: 1, pos: 34

### Candidate
type: B, layer: 1, pos: 34

### Candidate
type: A, layer: 1, pos: 30

## Relational analysis of NS_A1_A2_B2_B2_A2_B1_A2_B1_B2_A1

### Relational analysis result of NS_A1_A2_B2_B2_A2_B1_A2_B1_B2_A1
Status: Status.UNKNOWN
Output dim: 3, lower bound: -20.6034811, upper bound: 20.6032333
time: 0.70 seconds

## Relational analysis of NS_A1_A2_B2_B2_A2_B1_A2_B1_B2_A2

### Relational analysis result of NS_A1_A2_B2_B2_A2_B1_A2_B1_B2_A2
Status: Status.UNKNOWN
Output dim: 3, lower bound: -20.6034811, upper bound: 20.6032333
time: 0.66 seconds

## BFS NS instance: NS_A1_A2_B2_B2_A2_B1_A2_B2_B1

### Backsubstitution after applying NS history:
0: -3.5902476, 3.1974847, -2.9268007, 2.6424587, -6.2327061, 6.1242857
1: -14.1819506, 12.3807278, -11.5282898, 10.2406979, -24.4226456, 23.9090157
2: -7.0806346, 11.6148405, -5.7881970, 9.5416584, -16.6222935, 17.4030361
3: -12.3865166, 11.3239698, -10.0749216, 9.3865290, -21.7730446, 21.3988895
4: -9.0666447, 11.8067245, -7.3511415, 9.7661915, -18.8328362, 19.1578655

Time for backsubstitution: 1.50 seconds

### NS candidates at layer 1
type: B, layer: 1, pos: 11
type: A, layer: 1, pos: 11
type: B, layer: 1, pos: 47
type: B, layer: 1, pos: 34
type: A, layer: 1, pos: 34
type: B, layer: 1, pos: 12
type: A, layer: 1, pos: 12
type: B, layer: 1, pos: 9
type: A, layer: 1, pos: 4
type: A, layer: 1, pos: 16
type: B, layer: 1, pos: 16
type: A, layer: 1, pos: 39
type: A, layer: 1, pos: 31
type: B, layer: 1, pos: 31
type: A, layer: 1, pos: 23
type: B, layer: 1, pos: 33
type: A, layer: 1, pos: 30
type: B, layer: 1, pos: 39
type: B, layer: 1, pos: 23
type: A, layer: 1, pos: 33
type: A, layer: 1, pos: 47

Time for candidate selection: 0.13 seconds

### Candidate
type: B, layer: 1, pos: 11

### Candidate
type: A, layer: 1, pos: 11

### Candidate
type: B, layer: 1, pos: 47

## Relational analysis of NS_A1_A2_B2_B2_A2_B1_A2_B2_B1_B1
Optimization infeasible because this subproblem isn't reachable.

### Candidate
type: B, layer: 1, pos: 34

### Candidate
type: A, layer: 1, pos: 34

### Candidate
type: B, layer: 1, pos: 12

### Candidate
type: A, layer: 1, pos: 12

### Candidate
type: B, layer: 1, pos: 9

### Candidate
type: A, layer: 1, pos: 4

## Relational analysis of NS_A1_A2_B2_B2_A2_B1_A2_B2_B1_A1
Optimization infeasible because this subproblem isn't reachable.

### Candidate
type: A, layer: 1, pos: 16

## Relational analysis of NS_A1_A2_B2_B2_A2_B1_A2_B2_B1_A1

### Relational analysis result of NS_A1_A2_B2_B2_A2_B1_A2_B2_B1_A1
Status: Status.VERIFIED
Output dim: 3, lower bound: -20.5994015, upper bound: 20.6012676
time: 0.77 seconds

## Relational analysis of NS_A1_A2_B2_B2_A2_B1_A2_B2_B1_A2

### Relational analysis result of NS_A1_A2_B2_B2_A2_B1_A2_B2_B1_A2
Status: Status.UNKNOWN
Output dim: 3, lower bound: -20.6039411, upper bound: 20.6016956
time: 0.72 seconds

## BFS NS instance: NS_A1_A2_B2_B2_A2_B1_A2_B2_B2

### Backsubstitution after applying NS history:
0: -3.5902476, 3.1974847, -3.6080756, 3.2179179, -6.8081656, 6.8055601
1: -14.1819506, 12.3807278, -14.2473698, 12.4547148, -26.6366653, 26.6280975
2: -7.0806346, 11.6148405, -7.1079121, 11.7080612, -18.7886925, 18.7227516
3: -12.3865166, 11.3239698, -12.4447784, 11.3908911, -23.7774048, 23.7687492
4: -9.0666447, 11.8067245, -9.0990839, 11.8952971, -20.9619408, 20.9058075

Time for backsubstitution: 1.32 seconds

### NS candidates at layer 1
type: B, layer: 1, pos: 11
type: A, layer: 1, pos: 11
type: B, layer: 1, pos: 34
type: A, layer: 1, pos: 34
type: B, layer: 1, pos: 12
type: A, layer: 1, pos: 12
type: B, layer: 1, pos: 9
type: A, layer: 1, pos: 30
type: A, layer: 1, pos: 4
type: B, layer: 1, pos: 39
type: A, layer: 1, pos: 39
type: A, layer: 1, pos: 16
type: B, layer: 1, pos: 16
type: A, layer: 1, pos: 33
type: B, layer: 1, pos: 33
type: A, layer: 1, pos: 31
type: B, layer: 1, pos: 31
type: B, layer: 1, pos: 47
type: A, layer: 1, pos: 23
type: B, layer: 1, pos: 23
type: A, layer: 1, pos: 47

Time for candidate selection: 0.09 seconds

### Candidate
type: B, layer: 1, pos: 11

### Candidate
type: A, layer: 1, pos: 11

### Candidate
type: B, layer: 1, pos: 34

### Candidate
type: A, layer: 1, pos: 34

### Candidate
type: B, layer: 1, pos: 12

### Candidate
type: A, layer: 1, pos: 12

### Candidate
type: B, layer: 1, pos: 9

### Candidate
type: A, layer: 1, pos: 30

## Relational analysis of NS_A1_A2_B2_B2_A2_B1_A2_B2_B2_A1

### Relational analysis result of NS_A1_A2_B2_B2_A2_B1_A2_B2_B2_A1
Status: Status.UNKNOWN
Output dim: 3, lower bound: -20.6035657, upper bound: 20.6041258
time: 0.65 seconds

## Relational analysis of NS_A1_A2_B2_B2_A2_B1_A2_B2_B2_A2

### Relational analysis result of NS_A1_A2_B2_B2_A2_B1_A2_B2_B2_A2
Status: Status.UNKNOWN
Output dim: 3, lower bound: -20.6035657, upper bound: 20.6040657
time: 0.78 seconds

## Summary of splitting at layer (split count: 9)
- Time for NS candidates: 2.85 seconds
NS_A1_A2_B1_B1_B1_B1_A1_A2_A1_A1, status: Status.VERIFIED, split count: 10, time: 2.85
Output dim: 3, lower bound: -20.6028099, upper bound: 20.5972793
NS_A1_A2_B1_B1_B1_B1_A1_A2_A1_A2, status: Status.VERIFIED, split count: 10, time: 2.85
Output dim: 3, lower bound: -20.6028099, upper bound: 20.5972793
NS_A1_A2_B1_B1_B1_B1_A2_A2_A1_A1, status: Status.VERIFIED, split count: 10, time: 2.85
Output dim: 3, lower bound: -20.6016093, upper bound: 20.5971645
NS_A1_A2_B1_B1_B1_B1_A2_A2_A1_A2, status: Status.VERIFIED, split count: 10, time: 2.85
Output dim: 3, lower bound: -20.6016093, upper bound: 20.5972557
NS_A1_A2_B1_B1_B1_B1_A2_A2_A2_A1, status: Status.VERIFIED, split count: 10, time: 2.85
Output dim: 3, lower bound: -20.6029094, upper bound: 20.5972723
NS_A1_A2_B1_B1_B1_B1_A2_A2_A2_A2, status: Status.VERIFIED, split count: 10, time: 2.85
Output dim: 3, lower bound: -20.6029094, upper bound: 20.5972723
NS_A1_A2_B1_B1_B2_B1_A2_A1_A1_B1, status: Status.UNKNOWN, split count: 10, time: 2.85
Output dim: 3, lower bound: -20.6033943, upper bound: 20.5990128
NS_A1_A2_B1_B1_B2_B1_A2_A1_A1_B2, status: Status.UNKNOWN, split count: 10, time: 2.85
Output dim: 3, lower bound: -20.6035002, upper bound: 20.5994935
NS_A1_A2_B1_B1_B2_B1_A2_A1_A2_B1, status: Status.VERIFIED, split count: 10, time: 2.85
Output dim: 3, lower bound: -20.6031175, upper bound: 20.5990099
NS_A1_A2_B1_B1_B2_B1_A2_A1_A2_B2, status: Status.UNKNOWN, split count: 10, time: 2.85
Output dim: 3, lower bound: -20.6032235, upper bound: 20.5994906
NS_A1_A2_B1_B1_B2_B1_A2_A2_B1_A1, status: Status.UNKNOWN, split count: 10, time: 2.85
Output dim: 3, lower bound: -20.6033752, upper bound: 20.5989888
NS_A1_A2_B1_B1_B2_B1_A2_A2_B1_A2, status: Status.UNKNOWN, split count: 10, time: 2.85
Output dim: 3, lower bound: -20.6032433, upper bound: 20.5990013
NS_A1_A2_B1_B1_B2_B1_A2_A2_B2_A1, status: Status.UNKNOWN, split count: 10, time: 2.85
Output dim: 3, lower bound: -20.6034812, upper bound: 20.5994695
NS_A1_A2_B1_B1_B2_B1_A2_A2_B2_A2, status: Status.UNKNOWN, split count: 10, time: 2.85
Output dim: 3, lower bound: -20.6033492, upper bound: 20.5994820
NS_A1_A2_B1_B1_B2_B2_A1_B1_A1_A1, status: Status.VERIFIED, split count: 10, time: 2.85
Output dim: 3, lower bound: -20.6014707, upper bound: 20.5986244
NS_A1_A2_B1_B1_B2_B2_A1_B1_A1_A2, status: Status.VERIFIED, split count: 10, time: 2.85
Output dim: 3, lower bound: -20.6014707, upper bound: 20.5987226
NS_A1_A2_B1_B1_B2_B2_A1_B1_A2_A1, status: Status.VERIFIED, split count: 10, time: 2.85
Output dim: 3, lower bound: -20.6009570, upper bound: 20.5985516
NS_A1_A2_B1_B1_B2_B2_A1_B1_A2_A2, status: Status.VERIFIED, split count: 10, time: 2.85
Output dim: 3, lower bound: -20.6009570, upper bound: 20.5985516
NS_A1_A2_B1_B1_B2_B2_A1_B2_A1_A1, status: Status.VERIFIED, split count: 10, time: 2.85
Output dim: 3, lower bound: -20.6014707, upper bound: 20.5992590
NS_A1_A2_B1_B1_B2_B2_A1_B2_A1_A2, status: Status.VERIFIED, split count: 10, time: 2.85
Output dim: 3, lower bound: -20.6014707, upper bound: 20.5993572
NS_A1_A2_B1_B1_B2_B2_A1_B2_A2_A1, status: Status.VERIFIED, split count: 10, time: 2.85
Output dim: 3, lower bound: -20.6009570, upper bound: 20.5991862
NS_A1_A2_B1_B1_B2_B2_A1_B2_A2_A2, status: Status.VERIFIED, split count: 10, time: 2.85
Output dim: 3, lower bound: -20.6009570, upper bound: 20.5985516
NS_A1_A2_B1_B1_B2_B2_A2_A2_B1_A1, status: Status.UNKNOWN, split count: 10, time: 2.85
Output dim: 3, lower bound: -20.6034656, upper bound: 20.5986986
NS_A1_A2_B1_B1_B2_B2_A2_A2_B1_A2, status: Status.UNKNOWN, split count: 10, time: 2.85
Output dim: 3, lower bound: -20.6033252, upper bound: 20.5987112
NS_A1_A2_B1_B1_B2_B2_A2_A2_B2_A1, status: Status.UNKNOWN, split count: 10, time: 2.85
Output dim: 3, lower bound: -20.6034656, upper bound: 20.5986986
NS_A1_A2_B1_B1_B2_B2_A2_A2_B2_A2, status: Status.UNKNOWN, split count: 10, time: 2.85
Output dim: 3, lower bound: -20.6033252, upper bound: 20.5993457
NS_A1_A2_B1_B2_A2_B1_A1_A1_B2_A1, status: Status.UNKNOWN, split count: 10, time: 2.85
Output dim: 3, lower bound: -20.5959150, upper bound: 20.6033041
NS_A1_A2_B1_B2_A2_B1_A1_A1_B2_A2, status: Status.VERIFIED, split count: 10, time: 2.85
Output dim: 3, lower bound: -20.5959150, upper bound: 20.6028985
NS_A1_A2_B1_B2_A2_B1_A1_A2_B1_A1, status: Status.UNKNOWN, split count: 10, time: 2.85
Output dim: 3, lower bound: -20.6037343, upper bound: 20.6041886
NS_A1_A2_B1_B2_A2_B1_A1_A2_B1_A2, status: Status.UNKNOWN, split count: 10, time: 2.85
Output dim: 3, lower bound: -20.6037343, upper bound: 20.6040916
NS_A1_A2_B1_B2_A2_B1_A1_A2_B2_A1, status: Status.UNKNOWN, split count: 10, time: 2.85
Output dim: 3, lower bound: -20.6037342, upper bound: 20.6039453
NS_A1_A2_B1_B2_A2_B1_A1_A2_B2_A2, status: Status.UNKNOWN, split count: 10, time: 2.85
Output dim: 3, lower bound: -20.6037342, upper bound: 20.6039190
NS_A1_A2_B1_B2_A2_B1_A2_B1_B1_A1, status: Status.VERIFIED, split count: 10, time: 2.85
Output dim: 3, lower bound: -20.6028402, upper bound: 20.5964160
NS_A1_A2_B1_B2_A2_B1_A2_B1_B1_A2, status: Status.VERIFIED, split count: 10, time: 2.85
Output dim: 3, lower bound: -20.6028402, upper bound: 20.5964160
NS_A1_A2_B1_B2_A2_B1_A2_B1_B2_A1, status: Status.UNKNOWN, split count: 10, time: 2.85
Output dim: 3, lower bound: -20.6033251, upper bound: 20.6002121
NS_A1_A2_B1_B2_A2_B1_A2_B1_B2_A2, status: Status.UNKNOWN, split count: 10, time: 2.85
Output dim: 3, lower bound: -20.6033251, upper bound: 20.6002121
NS_A1_A2_B1_B2_A2_B1_A2_B2_A1_A1, status: Status.UNKNOWN, split count: 10, time: 2.85
Output dim: 3, lower bound: -20.6029630, upper bound: 20.6040652
NS_A1_A2_B1_B2_A2_B1_A2_B2_A1_A2, status: Status.UNKNOWN, split count: 10, time: 2.85
Output dim: 3, lower bound: -20.6029630, upper bound: 20.6037493
NS_A1_A2_B1_B2_A2_B1_A2_B2_A2_A1, status: Status.UNKNOWN, split count: 10, time: 2.85
Output dim: 3, lower bound: -20.6029630, upper bound: 20.6039567
NS_A1_A2_B1_B2_A2_B1_A2_B2_A2_A2, status: Status.UNKNOWN, split count: 10, time: 2.85
Output dim: 3, lower bound: -20.6029630, upper bound: 20.6040673
NS_A1_A2_B2_B1_B2_B1_A2_A1_B1_A1, status: Status.UNKNOWN, split count: 10, time: 2.85
Output dim: 3, lower bound: -20.6039741, upper bound: 20.6026981
NS_A1_A2_B2_B1_B2_B1_A2_A1_B1_A2, status: Status.UNKNOWN, split count: 10, time: 2.85
Output dim: 3, lower bound: -20.6037295, upper bound: 20.6026952
NS_A1_A2_B2_B1_B2_B1_A2_A1_B2_A1, status: Status.VERIFIED, split count: 10, time: 2.85
Output dim: 3, lower bound: -20.6000489, upper bound: 20.6023030
NS_A1_A2_B2_B1_B2_B1_A2_A1_B2_A2, status: Status.VERIFIED, split count: 10, time: 2.85
Output dim: 3, lower bound: -20.6000489, upper bound: 20.6026981
NS_A1_A2_B2_B1_B2_B1_A2_A2_A1_B1, status: Status.UNKNOWN, split count: 10, time: 2.85
Output dim: 3, lower bound: -20.6035637, upper bound: 20.6024183
NS_A1_A2_B2_B1_B2_B1_A2_A2_A1_B2, status: Status.UNKNOWN, split count: 10, time: 2.85
Output dim: 3, lower bound: -20.6035637, upper bound: 20.6024183
NS_A1_A2_B2_B1_B2_B1_A2_A2_A2_B1, status: Status.UNKNOWN, split count: 10, time: 2.85
Output dim: 3, lower bound: -20.6035622, upper bound: 20.6024620
NS_A1_A2_B2_B1_B2_B1_A2_A2_A2_B2, status: Status.UNKNOWN, split count: 10, time: 2.85
Output dim: 3, lower bound: -20.6035622, upper bound: 20.6024620
NS_A1_A2_B2_B1_B2_B2_A2_B1_A1_A1, status: Status.UNKNOWN, split count: 10, time: 2.85
Output dim: 3, lower bound: -20.6041077, upper bound: 20.6036961
NS_A1_A2_B2_B1_B2_B2_A2_B1_A1_A2, status: Status.UNKNOWN, split count: 10, time: 2.85
Output dim: 3, lower bound: -20.6038927, upper bound: 20.6037086
NS_A1_A2_B2_B1_B2_B2_A2_B1_A2_B1, status: Status.UNKNOWN, split count: 10, time: 2.85
Output dim: 3, lower bound: -20.6039654, upper bound: 20.6017344
NS_A1_A2_B2_B1_B2_B2_A2_B1_A2_B2, status: Status.UNKNOWN, split count: 10, time: 2.85
Output dim: 3, lower bound: -20.6039654, upper bound: 20.6037002
NS_A1_A2_B2_B1_B2_B2_A2_B2_A1_B1, status: Status.UNKNOWN, split count: 10, time: 2.85
Output dim: 3, lower bound: -20.6034656, upper bound: 20.5995481
NS_A1_A2_B2_B1_B2_B2_A2_B2_A1_B2, status: Status.UNKNOWN, split count: 10, time: 2.85
Output dim: 3, lower bound: -20.6034656, upper bound: 20.6002504
NS_A1_A2_B2_B1_B2_B2_A2_B2_A2_B1, status: Status.UNKNOWN, split count: 10, time: 2.85
Output dim: 3, lower bound: -20.6034656, upper bound: 20.5995144
NS_A1_A2_B2_B1_B2_B2_A2_B2_A2_B2, status: Status.UNKNOWN, split count: 10, time: 2.85
Output dim: 3, lower bound: -20.6034656, upper bound: 20.6002210
NS_A1_A2_B2_B2_A2_B1_A1_A2_A1_A1, status: Status.UNKNOWN, split count: 10, time: 2.85
Output dim: 3, lower bound: -20.6002121, upper bound: 20.6037242
NS_A1_A2_B2_B2_A2_B1_A1_A2_A1_A2, status: Status.UNKNOWN, split count: 10, time: 2.85
Output dim: 3, lower bound: -20.6036829, upper bound: 20.6041185
NS_A1_A2_B2_B2_A2_B1_A1_A2_A2_B1, status: Status.UNKNOWN, split count: 10, time: 2.85
Output dim: 3, lower bound: -20.6035983, upper bound: 20.6040746
NS_A1_A2_B2_B2_A2_B1_A1_A2_A2_B2, status: Status.UNKNOWN, split count: 10, time: 2.85
Output dim: 3, lower bound: -20.6037272, upper bound: 20.6040054
NS_A1_A2_B2_B2_A2_B1_A2_B1_B1_A1, status: Status.VERIFIED, split count: 10, time: 2.85
Output dim: 3, lower bound: -20.6031079, upper bound: 20.5992681
NS_A1_A2_B2_B2_A2_B1_A2_B1_B1_A2, status: Status.VERIFIED, split count: 10, time: 2.85
Output dim: 3, lower bound: -20.6031079, upper bound: 20.6001261
NS_A1_A2_B2_B2_A2_B1_A2_B1_B2_A1, status: Status.UNKNOWN, split count: 10, time: 2.85
Output dim: 3, lower bound: -20.6034811, upper bound: 20.6032333
NS_A1_A2_B2_B2_A2_B1_A2_B1_B2_A2, status: Status.UNKNOWN, split count: 10, time: 2.85
Output dim: 3, lower bound: -20.6034811, upper bound: 20.6032333
NS_A1_A2_B2_B2_A2_B1_A2_B2_B1_A1, status: Status.VERIFIED, split count: 10, time: 2.85
Output dim: 3, lower bound: -20.5994015, upper bound: 20.6012676
NS_A1_A2_B2_B2_A2_B1_A2_B2_B1_A2, status: Status.UNKNOWN, split count: 10, time: 2.85
Output dim: 3, lower bound: -20.6039411, upper bound: 20.6016956
NS_A1_A2_B2_B2_A2_B1_A2_B2_B2_A1, status: Status.UNKNOWN, split count: 10, time: 2.85
Output dim: 3, lower bound: -20.6035657, upper bound: 20.6041258
NS_A1_A2_B2_B2_A2_B1_A2_B2_B2_A2, status: Status.UNKNOWN, split count: 10, time: 2.85
Output dim: 3, lower bound: -20.6035657, upper bound: 20.6040657

## BFS NS instance: NS_A1_A2_B1_B1_B2_B1_A2_A1_A1_B1

### Backsubstitution after applying NS history:
0: -2.7816570, 2.4818509, -2.5164056, 2.2540786, -5.0357356, 4.9982557
1: -10.9689932, 9.6364336, -9.9001131, 8.7325802, -19.7015724, 19.5365467
2: -5.4935513, 9.0054016, -4.9823089, 8.1783438, -13.6718950, 13.9877110
3: -9.6097250, 8.8251762, -8.6768465, 8.0154171, -17.6251373, 17.5020199
4: -7.0272222, 9.1884327, -6.3510180, 8.3670273, -15.3942471, 15.5394506

Time for backsubstitution: 1.06 seconds

### NS candidates at layer 1
type: B, layer: 1, pos: 11
type: A, layer: 1, pos: 11
type: B, layer: 1, pos: 34
type: A, layer: 1, pos: 34
type: A, layer: 1, pos: 12
type: B, layer: 1, pos: 12
type: A, layer: 1, pos: 39
type: B, layer: 1, pos: 9
type: B, layer: 1, pos: 42
type: B, layer: 1, pos: 39
type: B, layer: 1, pos: 16
type: A, layer: 1, pos: 16
type: A, layer: 1, pos: 4
type: B, layer: 1, pos: 24
type: B, layer: 1, pos: 31
type: A, layer: 1, pos: 23
type: B, layer: 1, pos: 23
type: A, layer: 1, pos: 33
type: A, layer: 1, pos: 47
type: A, layer: 1, pos: 30

Time for candidate selection: 0.09 seconds

### Candidate
type: B, layer: 1, pos: 11

### Candidate
type: A, layer: 1, pos: 11

### Candidate
type: B, layer: 1, pos: 34

### Candidate
type: A, layer: 1, pos: 34

### Candidate
type: A, layer: 1, pos: 12

### Candidate
type: B, layer: 1, pos: 12

### Candidate
type: A, layer: 1, pos: 39

### Candidate
type: B, layer: 1, pos: 9

### Candidate
type: B, layer: 1, pos: 42

### Candidate
type: B, layer: 1, pos: 39

### Candidate
type: B, layer: 1, pos: 16

### Candidate
type: A, layer: 1, pos: 16

### Candidate
type: A, layer: 1, pos: 4

## Relational analysis of NS_A1_A2_B1_B1_B2_B1_A2_A1_A1_B1_A1
Optimization infeasible because this subproblem isn't reachable.

### Candidate
type: B, layer: 1, pos: 24

## Relational analysis of NS_A1_A2_B1_B1_B2_B1_A2_A1_A1_B1_B1

### Relational analysis result of NS_A1_A2_B1_B1_B2_B1_A2_A1_A1_B1_B1
Status: Status.UNKNOWN
Output dim: 3, lower bound: -20.6033943, upper bound: 20.5990128
time: 0.64 seconds

## Relational analysis of NS_A1_A2_B1_B1_B2_B1_A2_A1_A1_B1_B2

### Relational analysis result of NS_A1_A2_B1_B1_B2_B1_A2_A1_A1_B1_B2
Status: Status.UNKNOWN
Output dim: 3, lower bound: -20.6033943, upper bound: 20.5990128
time: 0.62 seconds

## BFS NS instance: NS_A1_A2_B1_B1_B2_B1_A2_A1_A1_B2

### Backsubstitution after applying NS history:
0: -2.7816570, 2.4818509, -2.6532226, 2.3691103, -5.1507673, 5.1350727
1: -10.9689932, 9.6364336, -10.4453049, 9.1917686, -20.1607571, 20.0817375
2: -5.4935513, 9.0054016, -5.2544699, 8.5943823, -14.0879316, 14.2598705
3: -9.6097250, 8.8251762, -9.1631756, 8.4334030, -18.0431290, 17.9883499
4: -7.0272222, 9.1884327, -6.7102489, 8.7898760, -15.8170967, 15.8986816

Time for backsubstitution: 1.13 seconds

### NS candidates at layer 1
type: B, layer: 1, pos: 11
type: A, layer: 1, pos: 11
type: B, layer: 1, pos: 34
type: A, layer: 1, pos: 34
type: A, layer: 1, pos: 12
type: B, layer: 1, pos: 12
type: A, layer: 1, pos: 39
type: B, layer: 1, pos: 9
type: B, layer: 1, pos: 42
type: B, layer: 1, pos: 39
type: B, layer: 1, pos: 16
type: A, layer: 1, pos: 16
type: A, layer: 1, pos: 4
type: B, layer: 1, pos: 31
type: A, layer: 1, pos: 23
type: B, layer: 1, pos: 23
type: B, layer: 1, pos: 24
type: A, layer: 1, pos: 33
type: A, layer: 1, pos: 47
type: A, layer: 1, pos: 30

Time for candidate selection: 0.08 seconds

### Candidate
type: B, layer: 1, pos: 11

### Candidate
type: A, layer: 1, pos: 11

### Candidate
type: B, layer: 1, pos: 34

### Candidate
type: A, layer: 1, pos: 34

### Candidate
type: A, layer: 1, pos: 12

### Candidate
type: B, layer: 1, pos: 12

### Candidate
type: A, layer: 1, pos: 39

### Candidate
type: B, layer: 1, pos: 9

### Candidate
type: B, layer: 1, pos: 42

### Candidate
type: B, layer: 1, pos: 39

### Candidate
type: B, layer: 1, pos: 16

### Candidate
type: A, layer: 1, pos: 16

### Candidate
type: A, layer: 1, pos: 4

## Relational analysis of NS_A1_A2_B1_B1_B2_B1_A2_A1_A1_B2_A1
Optimization infeasible because this subproblem isn't reachable.

### Candidate
type: B, layer: 1, pos: 31

## Relational analysis of NS_A1_A2_B1_B1_B2_B1_A2_A1_A1_B2_B1
Optimization infeasible because this subproblem isn't reachable.

### Candidate
type: A, layer: 1, pos: 23

## Relational analysis of NS_A1_A2_B1_B1_B2_B1_A2_A1_A1_B2_A1

### Relational analysis result of NS_A1_A2_B1_B1_B2_B1_A2_A1_A1_B2_A1
Status: Status.VERIFIED
Output dim: 3, lower bound: -20.6030587, upper bound: 20.5966719
time: 0.73 seconds

## Relational analysis of NS_A1_A2_B1_B1_B2_B1_A2_A1_A1_B2_A2
Optimization infeasible because this subproblem isn't reachable.

### Candidate
type: B, layer: 1, pos: 23

### Candidate
type: B, layer: 1, pos: 24

## Relational analysis of NS_A1_A2_B1_B1_B2_B1_A2_A1_A1_B2_B1

### Relational analysis result of NS_A1_A2_B1_B1_B2_B1_A2_A1_A1_B2_B1
Status: Status.UNKNOWN
Output dim: 3, lower bound: -20.6035002, upper bound: 20.5994935
time: 0.70 seconds

## Relational analysis of NS_A1_A2_B1_B1_B2_B1_A2_A1_A1_B2_B2

### Relational analysis result of NS_A1_A2_B1_B1_B2_B1_A2_A1_A1_B2_B2
Status: Status.UNKNOWN
Output dim: 3, lower bound: -20.6035002, upper bound: 20.5994935
time: 0.75 seconds

## BFS NS instance: NS_A1_A2_B1_B1_B2_B1_A2_A1_A2_B2

### Backsubstitution after applying NS history:
0: -2.9391685, 2.6167705, -2.6532226, 2.3691103, -5.3082781, 5.2699928
1: -11.6126900, 10.1238680, -10.4453049, 9.1917686, -20.8044529, 20.5691719
2: -5.8127203, 9.5147934, -5.2544699, 8.5943823, -14.4070997, 14.7692585
3: -10.1638746, 9.2626143, -9.1631756, 8.4334030, -18.5972786, 18.4257889
4: -7.4215121, 9.6846666, -6.7102489, 8.7898760, -16.2113876, 16.3949165

Time for backsubstitution: 1.21 seconds

### NS candidates at layer 1
type: B, layer: 1, pos: 11
type: B, layer: 1, pos: 34
type: A, layer: 1, pos: 11
type: A, layer: 1, pos: 34
type: A, layer: 1, pos: 12
type: B, layer: 1, pos: 12
type: A, layer: 1, pos: 39
type: B, layer: 1, pos: 9
type: B, layer: 1, pos: 42
type: B, layer: 1, pos: 16
type: B, layer: 1, pos: 39
type: B, layer: 1, pos: 31
type: B, layer: 1, pos: 24
type: A, layer: 1, pos: 4
type: A, layer: 1, pos: 16
type: A, layer: 1, pos: 23
type: B, layer: 1, pos: 23
type: A, layer: 1, pos: 33
type: A, layer: 1, pos: 47
type: A, layer: 1, pos: 30

Time for candidate selection: 0.08 seconds

### Candidate
type: B, layer: 1, pos: 11

### Candidate
type: B, layer: 1, pos: 34

### Candidate
type: A, layer: 1, pos: 11

### Candidate
type: A, layer: 1, pos: 34

### Candidate
type: A, layer: 1, pos: 12

### Candidate
type: B, layer: 1, pos: 12

### Candidate
type: A, layer: 1, pos: 39

### Candidate
type: B, layer: 1, pos: 9

### Candidate
type: B, layer: 1, pos: 42

### Candidate
type: B, layer: 1, pos: 16

### Candidate
type: B, layer: 1, pos: 39

### Candidate
type: B, layer: 1, pos: 31

## Relational analysis of NS_A1_A2_B1_B1_B2_B1_A2_A1_A2_B2_B1
Optimization infeasible because this subproblem isn't reachable.

### Candidate
type: B, layer: 1, pos: 24

## Relational analysis of NS_A1_A2_B1_B1_B2_B1_A2_A1_A2_B2_B1

### Relational analysis result of NS_A1_A2_B1_B1_B2_B1_A2_A1_A2_B2_B1
Status: Status.UNKNOWN
Output dim: 3, lower bound: -20.6032235, upper bound: 20.5994906
time: 0.67 seconds

## Relational analysis of NS_A1_A2_B1_B1_B2_B1_A2_A1_A2_B2_B2

### Relational analysis result of NS_A1_A2_B1_B1_B2_B1_A2_A1_A2_B2_B2
Status: Status.UNKNOWN
Output dim: 3, lower bound: -20.6032235, upper bound: 20.5994906
time: 0.64 seconds

## BFS NS instance: NS_A1_A2_B1_B1_B2_B1_A2_A2_B1_A1

### Backsubstitution after applying NS history:
0: -3.3850455, 3.0251527, -2.5164056, 2.2540786, -5.6391239, 5.5415573
1: -13.3516960, 11.7186394, -9.9001131, 8.7325802, -22.0842762, 21.6187515
2: -6.6723394, 10.9979820, -4.9823089, 8.1783438, -14.8506832, 15.9802885
3: -11.6664848, 10.7302942, -8.6768465, 8.0154171, -19.6818943, 19.4071388
4: -8.5420465, 11.1814547, -6.3510180, 8.3670273, -16.9090691, 17.5324707

Time for backsubstitution: 1.30 seconds

### NS candidates at layer 1
type: B, layer: 1, pos: 11
type: B, layer: 1, pos: 34
type: A, layer: 1, pos: 34
type: A, layer: 1, pos: 12
type: B, layer: 1, pos: 12
type: A, layer: 1, pos: 39
type: B, layer: 1, pos: 42
type: B, layer: 1, pos: 9
type: B, layer: 1, pos: 16
type: B, layer: 1, pos: 31
type: B, layer: 1, pos: 39
type: A, layer: 1, pos: 16
type: A, layer: 1, pos: 4
type: A, layer: 1, pos: 23
type: A, layer: 1, pos: 11
type: B, layer: 1, pos: 24
type: B, layer: 1, pos: 23
type: A, layer: 1, pos: 33
type: A, layer: 1, pos: 30
type: A, layer: 1, pos: 47

Time for candidate selection: 0.09 seconds

### Candidate
type: B, layer: 1, pos: 11

### Candidate
type: B, layer: 1, pos: 34

### Candidate
type: A, layer: 1, pos: 34

### Candidate
type: A, layer: 1, pos: 12

### Candidate
type: B, layer: 1, pos: 12

### Candidate
type: A, layer: 1, pos: 39

### Candidate
type: B, layer: 1, pos: 42

## Relational analysis of NS_A1_A2_B1_B1_B2_B1_A2_A2_B1_A1_B1
Optimization infeasible because this subproblem isn't reachable.

### Candidate
type: B, layer: 1, pos: 9

### Candidate
type: B, layer: 1, pos: 16

## Relational analysis of NS_A1_A2_B1_B1_B2_B1_A2_A2_B1_A1_B1
Optimization infeasible because this subproblem isn't reachable.

### Candidate
type: B, layer: 1, pos: 31

## Relational analysis of NS_A1_A2_B1_B1_B2_B1_A2_A2_B1_A1_B1
Optimization infeasible because this subproblem isn't reachable.

### Candidate
type: B, layer: 1, pos: 39

### Candidate
type: A, layer: 1, pos: 16

### Candidate
type: A, layer: 1, pos: 4

## Relational analysis of NS_A1_A2_B1_B1_B2_B1_A2_A2_B1_A1_A1

### Relational analysis result of NS_A1_A2_B1_B1_B2_B1_A2_A2_B1_A1_A1
Status: Status.VERIFIED
Output dim: 3, lower bound: -20.5988658, upper bound: 20.5986191
time: 0.75 seconds

## Relational analysis of NS_A1_A2_B1_B1_B2_B1_A2_A2_B1_A1_A2

### Relational analysis result of NS_A1_A2_B1_B1_B2_B1_A2_A2_B1_A1_A2
Status: Status.VERIFIED
Output dim: 3, lower bound: -20.5988658, upper bound: 20.5989888
time: 0.74 seconds

## BFS NS instance: NS_A1_A2_B1_B1_B2_B1_A2_A2_B1_A2

### Backsubstitution after applying NS history:
0: -3.4960272, 3.1204393, -2.5164056, 2.2540786, -5.7501044, 5.6368446
1: -13.8108044, 12.0488758, -9.9001131, 8.7325802, -22.5433846, 21.9489880
2: -6.8975406, 11.3577156, -4.9823089, 8.1783438, -15.0758839, 16.3400211
3: -12.0608492, 11.0254507, -8.6768465, 8.0154171, -20.0762615, 19.7022972
4: -8.8241854, 11.5409737, -6.3510180, 8.3670273, -17.1912117, 17.8919888

Time for backsubstitution: 1.04 seconds

### NS candidates at layer 1
type: B, layer: 1, pos: 11
type: B, layer: 1, pos: 34
type: A, layer: 1, pos: 34
type: A, layer: 1, pos: 12
type: B, layer: 1, pos: 12
type: A, layer: 1, pos: 39
type: B, layer: 1, pos: 42
type: B, layer: 1, pos: 9
type: B, layer: 1, pos: 16
type: B, layer: 1, pos: 31
type: A, layer: 1, pos: 4
type: A, layer: 1, pos: 16
type: B, layer: 1, pos: 39
type: A, layer: 1, pos: 23
type: B, layer: 1, pos: 24
type: B, layer: 1, pos: 23
type: A, layer: 1, pos: 33
type: A, layer: 1, pos: 11
type: A, layer: 1, pos: 30
type: A, layer: 1, pos: 47

Time for candidate selection: 0.08 seconds

### Candidate
type: B, layer: 1, pos: 11

### Candidate
type: B, layer: 1, pos: 34

### Candidate
type: A, layer: 1, pos: 34

### Candidate
type: A, layer: 1, pos: 12

### Candidate
type: B, layer: 1, pos: 12

### Candidate
type: A, layer: 1, pos: 39

### Candidate
type: B, layer: 1, pos: 42

## Relational analysis of NS_A1_A2_B1_B1_B2_B1_A2_A2_B1_A2_B1
Optimization infeasible because this subproblem isn't reachable.

### Candidate
type: B, layer: 1, pos: 9

### Candidate
type: B, layer: 1, pos: 16

## Relational analysis of NS_A1_A2_B1_B1_B2_B1_A2_A2_B1_A2_B1
Optimization infeasible because this subproblem isn't reachable.

### Candidate
type: B, layer: 1, pos: 31

## Relational analysis of NS_A1_A2_B1_B1_B2_B1_A2_A2_B1_A2_B1
Optimization infeasible because this subproblem isn't reachable.

### Candidate
type: A, layer: 1, pos: 4

## Relational analysis of NS_A1_A2_B1_B1_B2_B1_A2_A2_B1_A2_A1

### Relational analysis result of NS_A1_A2_B1_B1_B2_B1_A2_A2_B1_A2_A1
Status: Status.VERIFIED
Output dim: 3, lower bound: -20.6026581, upper bound: 20.5989166
time: 0.71 seconds

## Relational analysis of NS_A1_A2_B1_B1_B2_B1_A2_A2_B1_A2_A2

### Relational analysis result of NS_A1_A2_B1_B1_B2_B1_A2_A2_B1_A2_A2
Status: Status.VERIFIED
Output dim: 3, lower bound: -20.6026581, upper bound: 20.5990013
time: 0.89 seconds

## BFS NS instance: NS_A1_A2_B1_B1_B2_B1_A2_A2_B2_A1

### Backsubstitution after applying NS history:
0: -3.3850455, 3.0251527, -2.6532226, 2.3691103, -5.7541561, 5.6783743
1: -13.3516960, 11.7186394, -10.4453049, 9.1917686, -22.5434647, 22.1639404
2: -6.6723394, 10.9979820, -5.2544699, 8.5943823, -15.2667189, 16.2524471
3: -11.6664848, 10.7302942, -9.1631756, 8.4334030, -20.0998859, 19.8934689
4: -8.5420465, 11.1814547, -6.7102489, 8.7898760, -17.3319206, 17.8917046

Time for backsubstitution: 1.40 seconds

### NS candidates at layer 1
type: B, layer: 1, pos: 11
type: B, layer: 1, pos: 34
type: A, layer: 1, pos: 34
type: A, layer: 1, pos: 12
type: B, layer: 1, pos: 12
type: A, layer: 1, pos: 39
type: B, layer: 1, pos: 9
type: B, layer: 1, pos: 42
type: B, layer: 1, pos: 16
type: B, layer: 1, pos: 31
type: B, layer: 1, pos: 39
type: A, layer: 1, pos: 4
type: A, layer: 1, pos: 16
type: A, layer: 1, pos: 23
type: B, layer: 1, pos: 24
type: A, layer: 1, pos: 11
type: B, layer: 1, pos: 23
type: A, layer: 1, pos: 33
type: A, layer: 1, pos: 30
type: A, layer: 1, pos: 47

Time for candidate selection: 0.09 seconds

### Candidate
type: B, layer: 1, pos: 11

### Candidate
type: B, layer: 1, pos: 34

### Candidate
type: A, layer: 1, pos: 34

### Candidate
type: A, layer: 1, pos: 12

### Candidate
type: B, layer: 1, pos: 12

### Candidate
type: A, layer: 1, pos: 39

### Candidate
type: B, layer: 1, pos: 9

### Candidate
type: B, layer: 1, pos: 42

## Relational analysis of NS_A1_A2_B1_B1_B2_B1_A2_A2_B2_A1_B1
Optimization infeasible because this subproblem isn't reachable.

### Candidate
type: B, layer: 1, pos: 16

## Relational analysis of NS_A1_A2_B1_B1_B2_B1_A2_A2_B2_A1_B1
Optimization infeasible because this subproblem isn't reachable.

### Candidate
type: B, layer: 1, pos: 31

## Relational analysis of NS_A1_A2_B1_B1_B2_B1_A2_A2_B2_A1_B1
Optimization infeasible because this subproblem isn't reachable.

### Candidate
type: B, layer: 1, pos: 39

### Candidate
type: A, layer: 1, pos: 4

## Relational analysis of NS_A1_A2_B1_B1_B2_B1_A2_A2_B2_A1_A1

### Relational analysis result of NS_A1_A2_B1_B1_B2_B1_A2_A2_B2_A1_A1
Status: Status.VERIFIED
Output dim: 3, lower bound: -20.5989717, upper bound: 20.5990998
time: 0.81 seconds

## Relational analysis of NS_A1_A2_B1_B1_B2_B1_A2_A2_B2_A1_A2

### Relational analysis result of NS_A1_A2_B1_B1_B2_B1_A2_A2_B2_A1_A2
Status: Status.VERIFIED
Output dim: 3, lower bound: -20.5989717, upper bound: 20.5994695
time: 0.69 seconds

## BFS NS instance: NS_A1_A2_B1_B1_B2_B1_A2_A2_B2_A2

### Backsubstitution after applying NS history:
0: -3.4960272, 3.1204393, -2.6532226, 2.3691103, -5.8651366, 5.7736616
1: -13.8108044, 12.0488758, -10.4453049, 9.1917686, -23.0025711, 22.4941769
2: -6.8975406, 11.3577156, -5.2544699, 8.5943823, -15.4919214, 16.6121826
3: -12.0608492, 11.0254507, -9.1631756, 8.4334030, -20.4942513, 20.1886253
4: -8.8241854, 11.5409737, -6.7102489, 8.7898760, -17.6140614, 18.2512226

Time for backsubstitution: 1.05 seconds

### NS candidates at layer 1
type: B, layer: 1, pos: 11
type: B, layer: 1, pos: 34
type: A, layer: 1, pos: 34
type: A, layer: 1, pos: 12
type: B, layer: 1, pos: 12
type: A, layer: 1, pos: 39
type: B, layer: 1, pos: 42
type: B, layer: 1, pos: 9
type: B, layer: 1, pos: 16
type: B, layer: 1, pos: 31
type: A, layer: 1, pos: 4
type: B, layer: 1, pos: 39
type: A, layer: 1, pos: 16
type: A, layer: 1, pos: 23
type: B, layer: 1, pos: 24
type: B, layer: 1, pos: 23
type: A, layer: 1, pos: 33
type: A, layer: 1, pos: 11
type: A, layer: 1, pos: 30
type: A, layer: 1, pos: 47

Time for candidate selection: 0.09 seconds

### Candidate
type: B, layer: 1, pos: 11

### Candidate
type: B, layer: 1, pos: 34

### Candidate
type: A, layer: 1, pos: 34

### Candidate
type: A, layer: 1, pos: 12

### Candidate
type: B, layer: 1, pos: 12

### Candidate
type: A, layer: 1, pos: 39

### Candidate
type: B, layer: 1, pos: 42

## Relational analysis of NS_A1_A2_B1_B1_B2_B1_A2_A2_B2_A2_B1
Optimization infeasible because this subproblem isn't reachable.

### Candidate
type: B, layer: 1, pos: 9

### Candidate
type: B, layer: 1, pos: 16

## Relational analysis of NS_A1_A2_B1_B1_B2_B1_A2_A2_B2_A2_B1
Optimization infeasible because this subproblem isn't reachable.

### Candidate
type: B, layer: 1, pos: 31

## Relational analysis of NS_A1_A2_B1_B1_B2_B1_A2_A2_B2_A2_B1
Optimization infeasible because this subproblem isn't reachable.

### Candidate
type: A, layer: 1, pos: 4

## Relational analysis of NS_A1_A2_B1_B1_B2_B1_A2_A2_B2_A2_A1

### Relational analysis result of NS_A1_A2_B1_B1_B2_B1_A2_A2_B2_A2_A1
Status: Status.VERIFIED
Output dim: 3, lower bound: -20.6027640, upper bound: 20.5993973
time: 0.65 seconds

## Relational analysis of NS_A1_A2_B1_B1_B2_B1_A2_A2_B2_A2_A2

### Relational analysis result of NS_A1_A2_B1_B1_B2_B1_A2_A2_B2_A2_A2
Status: Status.VERIFIED
Output dim: 3, lower bound: -20.6027640, upper bound: 20.5994820
time: 0.75 seconds

## BFS NS instance: NS_A1_A2_B1_B1_B2_B2_A2_A2_B1_A1

### Backsubstitution after applying NS history:
0: -3.4751813, 3.1038034, -2.1720495, 1.9849454, -5.4601269, 5.2758522
1: -13.7093725, 12.0165625, -8.5200977, 7.6689053, -21.3782749, 20.5366592
2: -6.8422880, 11.2979622, -4.3270049, 7.1652737, -14.0075617, 15.6249647
3: -11.9793720, 10.9963074, -7.4694524, 7.0567627, -19.0361328, 18.4657593
4: -8.7660084, 11.4777985, -5.4614787, 7.3901196, -16.1561279, 16.9392719

Time for backsubstitution: 1.23 seconds

### NS candidates at layer 1
type: B, layer: 1, pos: 11
type: B, layer: 1, pos: 34
type: A, layer: 1, pos: 34
type: A, layer: 1, pos: 39
type: A, layer: 1, pos: 12
type: B, layer: 1, pos: 12
type: B, layer: 1, pos: 9
type: B, layer: 1, pos: 33
type: A, layer: 1, pos: 4
type: B, layer: 1, pos: 16
type: B, layer: 1, pos: 39
type: A, layer: 1, pos: 16
type: A, layer: 1, pos: 23
type: B, layer: 1, pos: 31
type: B, layer: 1, pos: 24
type: B, layer: 1, pos: 23
type: A, layer: 1, pos: 33
type: A, layer: 1, pos: 11
type: A, layer: 1, pos: 30
type: A, layer: 1, pos: 47

Time for candidate selection: 0.13 seconds

### Candidate
type: B, layer: 1, pos: 11

### Candidate
type: B, layer: 1, pos: 34

### Candidate
type: A, layer: 1, pos: 34

### Candidate
type: A, layer: 1, pos: 39

### Candidate
type: A, layer: 1, pos: 12

### Candidate
type: B, layer: 1, pos: 12

### Candidate
type: B, layer: 1, pos: 9

### Candidate
type: B, layer: 1, pos: 33

### Candidate
type: A, layer: 1, pos: 4

## Relational analysis of NS_A1_A2_B1_B1_B2_B2_A2_A2_B1_A1_A1
Optimization infeasible because this subproblem isn't reachable.

### Candidate
type: B, layer: 1, pos: 16

### Candidate
type: B, layer: 1, pos: 39

### Candidate
type: A, layer: 1, pos: 16

### Candidate
type: A, layer: 1, pos: 23

## Relational analysis of NS_A1_A2_B1_B1_B2_B2_A2_A2_B1_A1_A1
Optimization infeasible because this subproblem isn't reachable.

### Candidate
type: B, layer: 1, pos: 31

## Relational analysis of NS_A1_A2_B1_B1_B2_B2_A2_A2_B1_A1_B1
Optimization infeasible because this subproblem isn't reachable.

### Candidate
type: B, layer: 1, pos: 24

## Relational analysis of NS_A1_A2_B1_B1_B2_B2_A2_A2_B1_A1_B1

### Relational analysis result of NS_A1_A2_B1_B1_B2_B2_A2_A2_B1_A1_B1
Status: Status.UNKNOWN
Output dim: 3, lower bound: -20.6034656, upper bound: 20.5986986
time: 0.74 seconds

## Relational analysis of NS_A1_A2_B1_B1_B2_B2_A2_A2_B1_A1_B2

### Relational analysis result of NS_A1_A2_B1_B1_B2_B2_A2_A2_B1_A1_B2
Status: Status.UNKNOWN
Output dim: 3, lower bound: -20.6034656, upper bound: 20.5986986
time: 0.84 seconds

## BFS NS instance: NS_A1_A2_B1_B1_B2_B2_A2_A2_B1_A2

### Backsubstitution after applying NS history:
0: -3.7033052, 3.3009422, -2.1720495, 1.9849454, -5.6882505, 5.4729910
1: -14.6281128, 12.7401171, -8.5200977, 7.6689053, -22.2970181, 21.2602158
2: -7.2948422, 12.0380049, -4.3270049, 7.1652737, -14.4601154, 16.3650093
3: -12.7744160, 11.6426945, -7.4694524, 7.0567627, -19.8311768, 19.1121464
4: -9.3448181, 12.2149258, -5.4614787, 7.3901196, -16.7349377, 17.6764011

Time for backsubstitution: 1.27 seconds

### NS candidates at layer 1
type: B, layer: 1, pos: 11
type: B, layer: 1, pos: 34
type: A, layer: 1, pos: 34
type: A, layer: 1, pos: 39
type: A, layer: 1, pos: 12
type: B, layer: 1, pos: 12
type: B, layer: 1, pos: 33
type: B, layer: 1, pos: 9
type: A, layer: 1, pos: 4
type: B, layer: 1, pos: 16
type: B, layer: 1, pos: 39
type: A, layer: 1, pos: 16
type: A, layer: 1, pos: 23
type: B, layer: 1, pos: 31
type: B, layer: 1, pos: 23
type: A, layer: 1, pos: 33
type: B, layer: 1, pos: 24
type: A, layer: 1, pos: 11
type: A, layer: 1, pos: 30
type: A, layer: 1, pos: 47

Time for candidate selection: 0.10 seconds

### Candidate
type: B, layer: 1, pos: 11

### Candidate
type: B, layer: 1, pos: 34

### Candidate
type: A, layer: 1, pos: 34

### Candidate
type: A, layer: 1, pos: 39

### Candidate
type: A, layer: 1, pos: 12

### Candidate
type: B, layer: 1, pos: 12

### Candidate
type: B, layer: 1, pos: 33

### Candidate
type: B, layer: 1, pos: 9

### Candidate
type: A, layer: 1, pos: 4

## Relational analysis of NS_A1_A2_B1_B1_B2_B2_A2_A2_B1_A2_A1
Optimization infeasible because this subproblem isn't reachable.

### Candidate
type: B, layer: 1, pos: 16

### Candidate
type: B, layer: 1, pos: 39

### Candidate
type: A, layer: 1, pos: 16

### Candidate
type: A, layer: 1, pos: 23

## Relational analysis of NS_A1_A2_B1_B1_B2_B2_A2_A2_B1_A2_A1
Optimization infeasible because this subproblem isn't reachable.

### Candidate
type: B, layer: 1, pos: 31

## Relational analysis of NS_A1_A2_B1_B1_B2_B2_A2_A2_B1_A2_B1
Optimization infeasible because this subproblem isn't reachable.

### Candidate
type: B, layer: 1, pos: 23

### Candidate
type: A, layer: 1, pos: 33

### Candidate
type: B, layer: 1, pos: 24

## Relational analysis of NS_A1_A2_B1_B1_B2_B2_A2_A2_B1_A2_B1

### Relational analysis result of NS_A1_A2_B1_B1_B2_B2_A2_A2_B1_A2_B1
Status: Status.UNKNOWN
Output dim: 3, lower bound: -20.6033252, upper bound: 20.5987112
time: 0.73 seconds

## Relational analysis of NS_A1_A2_B1_B1_B2_B2_A2_A2_B1_A2_B2

### Relational analysis result of NS_A1_A2_B1_B1_B2_B2_A2_A2_B1_A2_B2
Status: Status.UNKNOWN
Output dim: 3, lower bound: -20.6033252, upper bound: 20.5987112
time: 1.01 seconds

## BFS NS instance: NS_A1_A2_B1_B1_B2_B2_A2_A2_B2_A1

### Backsubstitution after applying NS history:
0: -3.4751813, 3.1038034, -2.8515317, 2.5491934, -6.0243750, 5.9553351
1: -13.7093725, 12.0165625, -11.2261553, 9.8856926, -23.5950642, 23.2427177
2: -6.8422880, 11.2979622, -5.6394186, 9.2582951, -16.1005802, 16.9373817
3: -11.9793720, 10.9963074, -9.8397264, 9.0728273, -21.0521965, 20.8360310
4: -8.7660084, 11.4777985, -7.2097111, 9.4593086, -18.2253170, 18.6875057

Time for backsubstitution: 1.06 seconds

### NS candidates at layer 1
type: B, layer: 1, pos: 11
type: B, layer: 1, pos: 34
type: A, layer: 1, pos: 34
type: A, layer: 1, pos: 12
type: B, layer: 1, pos: 12
type: B, layer: 1, pos: 9
type: A, layer: 1, pos: 39
type: B, layer: 1, pos: 33
type: A, layer: 1, pos: 4
type: B, layer: 1, pos: 16
type: B, layer: 1, pos: 39
type: A, layer: 1, pos: 23
type: A, layer: 1, pos: 16
type: B, layer: 1, pos: 31
type: B, layer: 1, pos: 23
type: A, layer: 1, pos: 11
type: B, layer: 1, pos: 24
type: A, layer: 1, pos: 33
type: A, layer: 1, pos: 30
type: A, layer: 1, pos: 47

Time for candidate selection: 0.09 seconds

### Candidate
type: B, layer: 1, pos: 11

### Candidate
type: B, layer: 1, pos: 34

### Candidate
type: A, layer: 1, pos: 34

### Candidate
type: A, layer: 1, pos: 12

### Candidate
type: B, layer: 1, pos: 12

### Candidate
type: B, layer: 1, pos: 9

### Candidate
type: A, layer: 1, pos: 39

### Candidate
type: B, layer: 1, pos: 33

### Candidate
type: A, layer: 1, pos: 4

### Candidate
type: B, layer: 1, pos: 16

### Candidate
type: B, layer: 1, pos: 39

### Candidate
type: A, layer: 1, pos: 23

### Candidate
type: A, layer: 1, pos: 16

### Candidate
type: B, layer: 1, pos: 31

## Relational analysis of NS_A1_A2_B1_B1_B2_B2_A2_A2_B2_A1_B1
Optimization infeasible because this subproblem isn't reachable.

### Candidate
type: B, layer: 1, pos: 23

### Candidate
type: A, layer: 1, pos: 11

### Candidate
type: B, layer: 1, pos: 24

## Relational analysis of NS_A1_A2_B1_B1_B2_B2_A2_A2_B2_A1_B1

### Relational analysis result of NS_A1_A2_B1_B1_B2_B2_A2_A2_B2_A1_B1
Status: Status.UNKNOWN
Output dim: 3, lower bound: -20.6035433, upper bound: 20.5993332
time: 0.65 seconds

## Relational analysis of NS_A1_A2_B1_B1_B2_B2_A2_A2_B2_A1_B2

### Relational analysis result of NS_A1_A2_B1_B1_B2_B2_A2_A2_B2_A1_B2
Status: Status.UNKNOWN
Output dim: 3, lower bound: -20.6035433, upper bound: 20.5993332
time: 0.87 seconds

## BFS NS instance: NS_A1_A2_B1_B1_B2_B2_A2_A2_B2_A2

### Backsubstitution after applying NS history:
0: -3.7033052, 3.3009422, -2.8515317, 2.5491934, -6.2524986, 6.1524734
1: -14.6281128, 12.7401171, -11.2261553, 9.8856926, -24.5138054, 23.9662724
2: -7.2948422, 12.0380049, -5.6394186, 9.2582951, -16.5531349, 17.6774235
3: -12.7744160, 11.6426945, -9.8397264, 9.0728273, -21.8472424, 21.4824200
4: -9.3448181, 12.2149258, -7.2097111, 9.4593086, -18.8041267, 19.4246349

Time for backsubstitution: 1.44 seconds

### NS candidates at layer 1
type: B, layer: 1, pos: 11
type: B, layer: 1, pos: 34
type: A, layer: 1, pos: 34
type: A, layer: 1, pos: 12
type: B, layer: 1, pos: 12
type: A, layer: 1, pos: 39
type: B, layer: 1, pos: 9
type: B, layer: 1, pos: 33
type: A, layer: 1, pos: 4
type: B, layer: 1, pos: 16
type: A, layer: 1, pos: 23
type: A, layer: 1, pos: 16
type: B, layer: 1, pos: 31
type: B, layer: 1, pos: 39
type: B, layer: 1, pos: 23
type: B, layer: 1, pos: 24
type: A, layer: 1, pos: 33
type: A, layer: 1, pos: 11
type: A, layer: 1, pos: 30
type: A, layer: 1, pos: 47

Time for candidate selection: 0.13 seconds

### Candidate
type: B, layer: 1, pos: 11

### Candidate
type: B, layer: 1, pos: 34

### Candidate
type: A, layer: 1, pos: 34

### Candidate
type: A, layer: 1, pos: 12

### Candidate
type: B, layer: 1, pos: 12

### Candidate
type: A, layer: 1, pos: 39

### Candidate
type: B, layer: 1, pos: 9

### Candidate
type: B, layer: 1, pos: 33

### Candidate
type: A, layer: 1, pos: 4

### Candidate
type: B, layer: 1, pos: 16

### Candidate
type: A, layer: 1, pos: 23

### Candidate
type: A, layer: 1, pos: 16

### Candidate
type: B, layer: 1, pos: 31

## Relational analysis of NS_A1_A2_B1_B1_B2_B2_A2_A2_B2_A2_B1
Optimization infeasible because this subproblem isn't reachable.

### Candidate
type: B, layer: 1, pos: 39

### Candidate
type: B, layer: 1, pos: 23

### Candidate
type: B, layer: 1, pos: 24

## Relational analysis of NS_A1_A2_B1_B1_B2_B2_A2_A2_B2_A2_B1

### Relational analysis result of NS_A1_A2_B1_B1_B2_B2_A2_A2_B2_A2_B1
Status: Status.UNKNOWN
Output dim: 3, lower bound: -20.6034030, upper bound: 20.5993457
time: 0.70 seconds

## Relational analysis of NS_A1_A2_B1_B1_B2_B2_A2_A2_B2_A2_B2

### Relational analysis result of NS_A1_A2_B1_B1_B2_B2_A2_A2_B2_A2_B2
Status: Status.UNKNOWN
Output dim: 3, lower bound: -20.6034030, upper bound: 20.5993457
time: 0.78 seconds

## BFS NS instance: NS_A1_A2_B1_B2_A2_B1_A1_A1_B2_A1

### Backsubstitution after applying NS history:
0: -2.6430960, 2.3687704, -3.1073406, 2.7655692, -5.4086652, 5.4761109
1: -10.3937035, 9.1559248, -12.2705278, 10.7313585, -21.1250610, 21.4264469
2: -5.2244406, 8.6165409, -6.1469731, 10.0111504, -15.2355909, 14.7635136
3: -9.0996952, 8.4087467, -10.7416000, 9.8079548, -18.9076481, 19.1503468
4: -6.6588473, 8.7898111, -7.8535166, 10.2031994, -16.8620472, 16.6433277

Time for backsubstitution: 1.48 seconds

### NS candidates at layer 1
type: A, layer: 1, pos: 11
type: B, layer: 1, pos: 11
type: A, layer: 1, pos: 34
type: B, layer: 1, pos: 34
type: B, layer: 1, pos: 12
type: A, layer: 1, pos: 12
type: B, layer: 1, pos: 31
type: A, layer: 1, pos: 31
type: B, layer: 1, pos: 9
type: B, layer: 1, pos: 42
type: A, layer: 1, pos: 4
type: B, layer: 1, pos: 4
type: B, layer: 1, pos: 39
type: A, layer: 1, pos: 33
type: A, layer: 1, pos: 16
type: B, layer: 1, pos: 33
type: B, layer: 1, pos: 16
type: B, layer: 1, pos: 23
type: A, layer: 1, pos: 39
type: A, layer: 1, pos: 23

Time for candidate selection: 0.10 seconds

### Candidate
type: A, layer: 1, pos: 11

### Candidate
type: B, layer: 1, pos: 11

### Candidate
type: A, layer: 1, pos: 34

### Candidate
type: B, layer: 1, pos: 34

### Candidate
type: B, layer: 1, pos: 12

### Candidate
type: A, layer: 1, pos: 12

### Candidate
type: B, layer: 1, pos: 31

### Candidate
type: A, layer: 1, pos: 31

### Candidate
type: B, layer: 1, pos: 9

### Candidate
type: B, layer: 1, pos: 42

### Candidate
type: A, layer: 1, pos: 4

### Candidate
type: B, layer: 1, pos: 4

### Candidate
type: B, layer: 1, pos: 39

### Candidate
type: A, layer: 1, pos: 33

### Candidate
type: A, layer: 1, pos: 16

### Candidate
type: B, layer: 1, pos: 33

### Candidate
type: B, layer: 1, pos: 16

### Candidate
type: B, layer: 1, pos: 23

### Candidate
type: A, layer: 1, pos: 39

### Candidate
type: A, layer: 1, pos: 23

## BFS NS instance: NS_A1_A2_B1_B2_A2_B1_A1_A2_B1_A1

### Backsubstitution after applying NS history:
0: -2.7367759, 2.4551795, -2.9369402, 2.6221790, -5.3589549, 5.3921194
1: -10.7669916, 9.5255136, -11.5816984, 10.1788731, -20.9458656, 21.1072044
2: -5.4099050, 8.9155455, -5.8078890, 9.5083771, -14.9182816, 14.7234344
3: -9.4317722, 8.7528687, -10.1429844, 9.3147154, -18.7464848, 18.8958530
4: -6.9135633, 9.1204062, -7.4170961, 9.6984949, -16.6120567, 16.5375023

Time for backsubstitution: 1.12 seconds

### NS candidates at layer 1
type: A, layer: 1, pos: 11
type: B, layer: 1, pos: 11
type: A, layer: 1, pos: 34
type: B, layer: 1, pos: 34
type: A, layer: 1, pos: 4
type: B, layer: 1, pos: 9
type: B, layer: 1, pos: 4
type: B, layer: 1, pos: 12
type: A, layer: 1, pos: 12
type: B, layer: 1, pos: 39
type: A, layer: 1, pos: 33
type: B, layer: 1, pos: 42
type: A, layer: 1, pos: 16
type: B, layer: 1, pos: 23
type: B, layer: 1, pos: 16
type: A, layer: 1, pos: 23
type: A, layer: 1, pos: 39
type: A, layer: 1, pos: 31
type: B, layer: 1, pos: 33
type: B, layer: 1, pos: 47

Time for candidate selection: 0.09 seconds

### Candidate
type: A, layer: 1, pos: 11

### Candidate
type: B, layer: 1, pos: 11

### Candidate
type: A, layer: 1, pos: 34

### Candidate
type: B, layer: 1, pos: 34

### Candidate
type: A, layer: 1, pos: 4

## Relational analysis of NS_A1_A2_B1_B2_A2_B1_A1_A2_B1_A1_A1

### Relational analysis result of NS_A1_A2_B1_B2_A2_B1_A1_A2_B1_A1_A1
Status: Status.UNKNOWN
Output dim: 3, lower bound: -20.6002229, upper bound: 20.6037538
time: 0.69 seconds

## Relational analysis of NS_A1_A2_B1_B2_A2_B1_A1_A2_B1_A1_A2

### Relational analysis result of NS_A1_A2_B1_B2_A2_B1_A1_A2_B1_A1_A2
Status: Status.UNKNOWN
Output dim: 3, lower bound: -20.6036937, upper bound: 20.6041459
time: 0.72 seconds

## BFS NS instance: NS_A1_A2_B1_B2_A2_B1_A1_A2_B1_A2

### Backsubstitution after applying NS history:
0: -2.8930047, 2.5780895, -2.9369402, 2.6221790, -5.5151834, 5.5150299
1: -11.4196548, 10.0121975, -11.5816984, 10.1788731, -21.5985279, 21.5938892
2: -5.7184653, 9.3445778, -5.8078890, 9.5083771, -15.2268429, 15.1524658
3: -9.9995356, 9.1629286, -10.1429844, 9.3147154, -19.3142452, 19.3059120
4: -7.3038015, 9.5328236, -7.4170961, 9.6984949, -17.0022964, 16.9499207

Time for backsubstitution: 1.62 seconds

### NS candidates at layer 1
type: A, layer: 1, pos: 11
type: B, layer: 1, pos: 11
type: B, layer: 1, pos: 34
type: A, layer: 1, pos: 34
type: B, layer: 1, pos: 12
type: A, layer: 1, pos: 12
type: B, layer: 1, pos: 9
type: B, layer: 1, pos: 42
type: B, layer: 1, pos: 4
type: A, layer: 1, pos: 4
type: B, layer: 1, pos: 16
type: A, layer: 1, pos: 16
type: B, layer: 1, pos: 47
type: A, layer: 1, pos: 39
type: A, layer: 1, pos: 31
type: B, layer: 1, pos: 23
type: B, layer: 1, pos: 39
type: A, layer: 1, pos: 23
type: A, layer: 1, pos: 33
type: B, layer: 1, pos: 33

Time for candidate selection: 0.14 seconds

### Candidate
type: A, layer: 1, pos: 11

### Candidate
type: B, layer: 1, pos: 11

### Candidate
type: B, layer: 1, pos: 34

### Candidate
type: A, layer: 1, pos: 34

### Candidate
type: B, layer: 1, pos: 12

### Candidate
type: A, layer: 1, pos: 12

### Candidate
type: B, layer: 1, pos: 9

### Candidate
type: B, layer: 1, pos: 42

## Relational analysis of NS_A1_A2_B1_B2_A2_B1_A1_A2_B1_A2_B1

### Relational analysis result of NS_A1_A2_B1_B2_A2_B1_A1_A2_B1_A2_B1
Status: Status.UNKNOWN
Output dim: 3, lower bound: -20.6036027, upper bound: 20.6021526
time: 0.86 seconds

## Relational analysis of NS_A1_A2_B1_B2_A2_B1_A1_A2_B1_A2_B2

### Relational analysis result of NS_A1_A2_B1_B2_A2_B1_A1_A2_B1_A2_B2
Status: Status.UNKNOWN
Output dim: 3, lower bound: -20.6036027, upper bound: 20.6021526
time: 1.09 seconds

## BFS NS instance: NS_A1_A2_B1_B2_A2_B1_A1_A2_B2_A1

### Backsubstitution after applying NS history:
0: -2.7367759, 2.4551795, -3.2003164, 2.8472798, -5.5840545, 5.6554956
1: -10.7669916, 9.5255136, -12.6413479, 11.0089865, -21.7759781, 22.1668568
2: -5.4099050, 8.9155455, -6.3340435, 10.3492575, -15.7591629, 15.2495890
3: -9.4317722, 8.7528687, -11.0609941, 10.0579910, -19.4897633, 19.8138618
4: -6.9135633, 9.1204062, -8.0821466, 10.5208540, -17.4344120, 17.2025528

Time for backsubstitution: 1.29 seconds

### NS candidates at layer 1
type: A, layer: 1, pos: 11
type: A, layer: 1, pos: 34
type: B, layer: 1, pos: 11
type: B, layer: 1, pos: 34
type: A, layer: 1, pos: 4
type: B, layer: 1, pos: 4
type: B, layer: 1, pos: 12
type: B, layer: 1, pos: 9
type: A, layer: 1, pos: 12
type: B, layer: 1, pos: 39
type: B, layer: 1, pos: 42
type: A, layer: 1, pos: 33
type: A, layer: 1, pos: 16
type: A, layer: 1, pos: 39
type: B, layer: 1, pos: 23
type: A, layer: 1, pos: 31
type: B, layer: 1, pos: 16
type: A, layer: 1, pos: 23
type: B, layer: 1, pos: 33
type: B, layer: 1, pos: 47

Time for candidate selection: 0.09 seconds

### Candidate
type: A, layer: 1, pos: 11

### Candidate
type: A, layer: 1, pos: 34

### Candidate
type: B, layer: 1, pos: 11

### Candidate
type: B, layer: 1, pos: 34

### Candidate
type: A, layer: 1, pos: 4

## Relational analysis of NS_A1_A2_B1_B2_A2_B1_A1_A2_B2_A1_A1

### Relational analysis result of NS_A1_A2_B1_B2_A2_B1_A1_A2_B2_A1_A1
Status: Status.UNKNOWN
Output dim: 3, lower bound: -20.6002207, upper bound: 20.6035091
time: 0.65 seconds

## Relational analysis of NS_A1_A2_B1_B2_A2_B1_A1_A2_B2_A1_A2

### Relational analysis result of NS_A1_A2_B1_B2_A2_B1_A1_A2_B2_A1_A2
Status: Status.UNKNOWN
Output dim: 3, lower bound: -20.6036915, upper bound: 20.6039012
time: 0.85 seconds

## BFS NS instance: NS_A1_A2_B1_B2_A2_B1_A1_A2_B2_A2

### Backsubstitution after applying NS history:
0: -2.8930047, 2.5780895, -3.2003164, 2.8472798, -5.7402835, 5.7784061
1: -11.4196548, 10.0121975, -12.6413479, 11.0089865, -22.4286423, 22.6535435
2: -5.7184653, 9.3445778, -6.3340435, 10.3492575, -16.0677223, 15.6786213
3: -9.9995356, 9.1629286, -11.0609941, 10.0579910, -20.0575256, 20.2239227
4: -7.3038015, 9.5328236, -8.0821466, 10.5208540, -17.8246517, 17.6149654

Time for backsubstitution: 1.60 seconds

### NS candidates at layer 1
type: A, layer: 1, pos: 11
type: B, layer: 1, pos: 11
type: B, layer: 1, pos: 34
type: A, layer: 1, pos: 34
type: B, layer: 1, pos: 12
type: A, layer: 1, pos: 12
type: B, layer: 1, pos: 9
type: A, layer: 1, pos: 31
type: B, layer: 1, pos: 4
type: B, layer: 1, pos: 16
type: B, layer: 1, pos: 42
type: B, layer: 1, pos: 39
type: A, layer: 1, pos: 16
type: A, layer: 1, pos: 4
type: A, layer: 1, pos: 39
type: B, layer: 1, pos: 23
type: A, layer: 1, pos: 23
type: A, layer: 1, pos: 33
type: B, layer: 1, pos: 33
type: B, layer: 1, pos: 47

Time for candidate selection: 0.12 seconds

### Candidate
type: A, layer: 1, pos: 11

### Candidate
type: B, layer: 1, pos: 11

### Candidate
type: B, layer: 1, pos: 34

### Candidate
type: A, layer: 1, pos: 34

### Candidate
type: B, layer: 1, pos: 12

### Candidate
type: A, layer: 1, pos: 12

### Candidate
type: B, layer: 1, pos: 9

### Candidate
type: A, layer: 1, pos: 31

## Relational analysis of NS_A1_A2_B1_B2_A2_B1_A1_A2_B2_A2_A1

### Relational analysis result of NS_A1_A2_B1_B2_A2_B1_A1_A2_B2_A2_A1
Status: Status.UNKNOWN
Output dim: 3, lower bound: -20.6036603, upper bound: 20.6039190
time: 0.76 seconds

## Relational analysis of NS_A1_A2_B1_B2_A2_B1_A1_A2_B2_A2_A2

### Relational analysis result of NS_A1_A2_B1_B2_A2_B1_A1_A2_B2_A2_A2
Status: Status.UNKNOWN
Output dim: 3, lower bound: -20.6036603, upper bound: 20.6039190
time: 0.65 seconds

## BFS NS instance: NS_A1_A2_B1_B2_A2_B1_A2_B1_B2_A1

### Backsubstitution after applying NS history:
0: -3.3409936, 2.9953606, -2.7827682, 2.4748209, -5.8158140, 5.7781286
1: -13.1568880, 11.6057796, -10.9915371, 9.5970364, -22.7539253, 22.5973148
2: -6.5867920, 10.9051161, -5.5127335, 8.9621267, -15.5489187, 16.4178467
3: -11.4867058, 10.6629086, -9.6334667, 8.7718945, -20.2586002, 20.2963753
4: -8.4299479, 11.1081009, -7.0310092, 9.1374693, -17.5674133, 18.1391048

Time for backsubstitution: 1.08 seconds

### NS candidates at layer 1
type: B, layer: 1, pos: 11
type: A, layer: 1, pos: 11
type: A, layer: 1, pos: 34
type: B, layer: 1, pos: 34
type: A, layer: 1, pos: 12
type: B, layer: 1, pos: 12
type: A, layer: 1, pos: 47
type: B, layer: 1, pos: 9
type: B, layer: 1, pos: 31
type: A, layer: 1, pos: 31
type: B, layer: 1, pos: 42
type: A, layer: 1, pos: 4
type: A, layer: 1, pos: 16
type: B, layer: 1, pos: 39
type: A, layer: 1, pos: 39
type: A, layer: 1, pos: 33
type: B, layer: 1, pos: 16
type: A, layer: 1, pos: 23
type: B, layer: 1, pos: 33
type: B, layer: 1, pos: 23

Time for candidate selection: 0.09 seconds

### Candidate
type: B, layer: 1, pos: 11

### Candidate
type: A, layer: 1, pos: 11

### Candidate
type: A, layer: 1, pos: 34

### Candidate
type: B, layer: 1, pos: 34

### Candidate
type: A, layer: 1, pos: 12

### Candidate
type: B, layer: 1, pos: 12

### Candidate
type: A, layer: 1, pos: 47

## Relational analysis of NS_A1_A2_B1_B2_A2_B1_A2_B1_B2_A1_A1
Optimization infeasible because this subproblem isn't reachable.

### Candidate
type: B, layer: 1, pos: 9

## Relational analysis of NS_A1_A2_B1_B2_A2_B1_A2_B1_B2_A1_B1
Optimization infeasible because this subproblem isn't reachable.

### Candidate
type: B, layer: 1, pos: 31

## Relational analysis of NS_A1_A2_B1_B2_A2_B1_A2_B1_B2_A1_B1
Optimization infeasible because this subproblem isn't reachable.

### Candidate
type: A, layer: 1, pos: 31

### Candidate
type: B, layer: 1, pos: 42

## Relational analysis of NS_A1_A2_B1_B2_A2_B1_A2_B1_B2_A1_B1
Optimization infeasible because this subproblem isn't reachable.

### Candidate
type: A, layer: 1, pos: 4

## Relational analysis of NS_A1_A2_B1_B2_A2_B1_A2_B1_B2_A1_A1

### Relational analysis result of NS_A1_A2_B1_B2_A2_B1_A2_B1_B2_A1_A1
Status: Status.VERIFIED
Output dim: 3, lower bound: -20.6029355, upper bound: 20.6001274
time: 0.69 seconds

## Relational analysis of NS_A1_A2_B1_B2_A2_B1_A2_B1_B2_A1_A2

### Relational analysis result of NS_A1_A2_B1_B2_A2_B1_A2_B1_B2_A1_A2
Status: Status.VERIFIED
Output dim: 3, lower bound: -20.6029355, upper bound: 20.6002121
time: 0.61 seconds

## BFS NS instance: NS_A1_A2_B1_B2_A2_B1_A2_B1_B2_A2

### Backsubstitution after applying NS history:
0: -3.5319197, 3.1517057, -2.7827682, 2.4748209, -6.0067396, 5.9344735
1: -13.9454231, 12.2059650, -10.9915371, 9.5970364, -23.5424595, 23.1975002
2: -6.9684982, 11.4514275, -5.5127335, 8.9621267, -15.9306250, 16.9641571
3: -12.1795673, 11.1709251, -9.6334667, 8.7718945, -20.9514618, 20.8043919
4: -8.9091740, 11.6463661, -7.0310092, 9.1374693, -18.0466385, 18.6773701

Time for backsubstitution: 1.43 seconds

### NS candidates at layer 1
type: B, layer: 1, pos: 11
type: A, layer: 1, pos: 11
type: A, layer: 1, pos: 34
type: B, layer: 1, pos: 34
type: A, layer: 1, pos: 12
type: B, layer: 1, pos: 12
type: B, layer: 1, pos: 9
type: A, layer: 1, pos: 31
type: B, layer: 1, pos: 31
type: B, layer: 1, pos: 42
type: A, layer: 1, pos: 4
type: A, layer: 1, pos: 16
type: B, layer: 1, pos: 16
type: A, layer: 1, pos: 39
type: B, layer: 1, pos: 33
type: A, layer: 1, pos: 23
type: B, layer: 1, pos: 23
type: A, layer: 1, pos: 33
type: B, layer: 1, pos: 39
type: A, layer: 1, pos: 47

Time for candidate selection: 0.10 seconds

### Candidate
type: B, layer: 1, pos: 11

### Candidate
type: A, layer: 1, pos: 11

### Candidate
type: A, layer: 1, pos: 34

### Candidate
type: B, layer: 1, pos: 34

### Candidate
type: A, layer: 1, pos: 12

### Candidate
type: B, layer: 1, pos: 12

### Candidate
type: B, layer: 1, pos: 9

## Relational analysis of NS_A1_A2_B1_B2_A2_B1_A2_B1_B2_A2_B1
Optimization infeasible because this subproblem isn't reachable.

### Candidate
type: A, layer: 1, pos: 31

## Relational analysis of NS_A1_A2_B1_B2_A2_B1_A2_B1_B2_A2_A1

### Relational analysis result of NS_A1_A2_B1_B2_A2_B1_A2_B1_B2_A2_A1
Status: Status.VERIFIED
Output dim: 3, lower bound: -20.6020250, upper bound: 20.6001996
time: 0.67 seconds

## Relational analysis of NS_A1_A2_B1_B2_A2_B1_A2_B1_B2_A2_A2

### Relational analysis result of NS_A1_A2_B1_B2_A2_B1_A2_B1_B2_A2_A2
Status: Status.UNKNOWN
Output dim: 3, lower bound: -20.6033251, upper bound: 20.6002121
time: 0.69 seconds

## BFS NS instance: NS_A1_A2_B1_B2_A2_B1_A2_B2_A1_A1

### Backsubstitution after applying NS history:
0: -3.0119102, 2.6933634, -3.0771875, 2.7368279, -5.7487383, 5.7705507
1: -11.8554697, 10.4257259, -12.1525078, 10.6178522, -22.4733219, 22.5782318
2: -5.9509277, 9.8081779, -6.0879107, 9.9097996, -15.8607273, 15.8960876
3: -10.3699198, 9.5781031, -10.6401510, 9.7027712, -20.0726910, 20.2182541
4: -7.6069245, 9.9983540, -7.7787948, 10.0979300, -17.7048550, 17.7771492

Time for backsubstitution: 1.24 seconds

### NS candidates at layer 1
type: B, layer: 1, pos: 11
type: A, layer: 1, pos: 11
type: A, layer: 1, pos: 47
type: A, layer: 1, pos: 34
type: B, layer: 1, pos: 34
type: A, layer: 1, pos: 12
type: B, layer: 1, pos: 12
type: B, layer: 1, pos: 9
type: B, layer: 1, pos: 31
type: A, layer: 1, pos: 31
type: B, layer: 1, pos: 42
type: B, layer: 1, pos: 39
type: A, layer: 1, pos: 33
type: A, layer: 1, pos: 16
type: A, layer: 1, pos: 39
type: B, layer: 1, pos: 16
type: B, layer: 1, pos: 23
type: B, layer: 1, pos: 33
type: A, layer: 1, pos: 23
type: B, layer: 1, pos: 47

Time for candidate selection: 0.13 seconds

### Candidate
type: B, layer: 1, pos: 11

### Candidate
type: A, layer: 1, pos: 11

### Candidate
type: A, layer: 1, pos: 47

### Candidate
type: A, layer: 1, pos: 34

### Candidate
type: B, layer: 1, pos: 34

### Candidate
type: A, layer: 1, pos: 12

### Candidate
type: B, layer: 1, pos: 12

### Candidate
type: B, layer: 1, pos: 9

### Candidate
type: B, layer: 1, pos: 31

## Relational analysis of NS_A1_A2_B1_B2_A2_B1_A2_B2_A1_A1_B1
Optimization infeasible because this subproblem isn't reachable.

### Candidate
type: A, layer: 1, pos: 31

### Candidate
type: B, layer: 1, pos: 42

## Relational analysis of NS_A1_A2_B1_B2_A2_B1_A2_B2_A1_A1_B1
Optimization infeasible because this subproblem isn't reachable.

### Candidate
type: B, layer: 1, pos: 39

## Relational analysis of NS_A1_A2_B1_B2_A2_B1_A2_B2_A1_A1_B1
Optimization infeasible because this subproblem isn't reachable.

### Candidate
type: A, layer: 1, pos: 33

## Relational analysis of NS_A1_A2_B1_B2_A2_B1_A2_B2_A1_A1_A1

### Relational analysis result of NS_A1_A2_B1_B2_A2_B1_A2_B2_A1_A1_A1
Status: Status.UNKNOWN
Output dim: 3, lower bound: -20.6028480, upper bound: 20.6039922
time: 0.74 seconds

## Relational analysis of NS_A1_A2_B1_B2_A2_B1_A2_B2_A1_A1_A2

### Relational analysis result of NS_A1_A2_B1_B2_A2_B1_A2_B2_A1_A1_A2
Status: Status.VERIFIED
Output dim: 3, lower bound: -20.6029150, upper bound: 20.6001799
time: 1.31 seconds

## BFS NS instance: NS_A1_A2_B1_B2_A2_B1_A2_B2_A1_A2

### Backsubstitution after applying NS history:
0: -3.2805936, 2.9413643, -3.0771875, 2.7368279, -6.0174217, 6.0185518
1: -12.9186125, 11.3891850, -12.1525078, 10.6178522, -23.5364647, 23.5416908
2: -6.4675436, 10.7104273, -6.0879107, 9.9097996, -16.3773422, 16.7983341
3: -11.2828960, 10.4635677, -10.6401510, 9.7027712, -20.9856682, 21.1037178
4: -8.2807093, 10.9069805, -7.7787948, 10.0979300, -18.3786392, 18.6857758

Time for backsubstitution: 1.15 seconds

### NS candidates at layer 1
type: B, layer: 1, pos: 11
type: A, layer: 1, pos: 11
type: A, layer: 1, pos: 34
type: A, layer: 1, pos: 47
type: B, layer: 1, pos: 34
type: A, layer: 1, pos: 12
type: B, layer: 1, pos: 12
type: B, layer: 1, pos: 9
type: B, layer: 1, pos: 42
type: A, layer: 1, pos: 33
type: A, layer: 1, pos: 16
type: B, layer: 1, pos: 39
type: B, layer: 1, pos: 31
type: B, layer: 1, pos: 16
type: A, layer: 1, pos: 39
type: A, layer: 1, pos: 31
type: A, layer: 1, pos: 23
type: B, layer: 1, pos: 23
type: B, layer: 1, pos: 33
type: B, layer: 1, pos: 47

Time for candidate selection: 0.09 seconds

### Candidate
type: B, layer: 1, pos: 11

### Candidate
type: A, layer: 1, pos: 11

### Candidate
type: A, layer: 1, pos: 34

### Candidate
type: A, layer: 1, pos: 47

### Candidate
type: B, layer: 1, pos: 34

### Candidate
type: A, layer: 1, pos: 12

### Candidate
type: B, layer: 1, pos: 12

### Candidate
type: B, layer: 1, pos: 9

### Candidate
type: B, layer: 1, pos: 42

## Relational analysis of NS_A1_A2_B1_B2_A2_B1_A2_B2_A1_A2_B1
Optimization infeasible because this subproblem isn't reachable.

### Candidate
type: A, layer: 1, pos: 33

## Relational analysis of NS_A1_A2_B1_B2_A2_B1_A2_B2_A1_A2_A1

### Relational analysis result of NS_A1_A2_B1_B2_A2_B1_A2_B2_A1_A2_A1
Status: Status.UNKNOWN
Output dim: 3, lower bound: -20.6028480, upper bound: 20.6035560
time: 0.72 seconds

## Relational analysis of NS_A1_A2_B1_B2_A2_B1_A2_B2_A1_A2_A2

### Relational analysis result of NS_A1_A2_B1_B2_A2_B1_A2_B2_A1_A2_A2
Status: Status.UNKNOWN
Output dim: 3, lower bound: -20.6029150, upper bound: 20.6036165
time: 0.75 seconds

## BFS NS instance: NS_A1_A2_B1_B2_A2_B1_A2_B2_A2_A1

### Backsubstitution after applying NS history:
0: -3.1844416, 2.8368254, -3.0771875, 2.7368279, -5.9212689, 5.9140129
1: -12.5715866, 10.9695587, -12.1525078, 10.6178522, -23.1894360, 23.1220646
2: -6.2930036, 10.3099813, -6.0879107, 9.9097996, -16.2028027, 16.3978901
3: -10.9970789, 10.0401611, -10.6401510, 9.7027712, -20.6998501, 20.6803131
4: -8.0355978, 10.4912605, -7.7787948, 10.0979300, -18.1335258, 18.2700558

Time for backsubstitution: 1.23 seconds

### NS candidates at layer 1
type: B, layer: 1, pos: 11
type: A, layer: 1, pos: 11
type: B, layer: 1, pos: 34
type: A, layer: 1, pos: 34
type: B, layer: 1, pos: 47
type: A, layer: 1, pos: 12
type: B, layer: 1, pos: 12
type: A, layer: 1, pos: 47
type: B, layer: 1, pos: 9
type: B, layer: 1, pos: 31
type: A, layer: 1, pos: 31
type: B, layer: 1, pos: 42
type: B, layer: 1, pos: 16
type: A, layer: 1, pos: 39
type: A, layer: 1, pos: 16
type: B, layer: 1, pos: 23
type: A, layer: 1, pos: 23
type: B, layer: 1, pos: 33
type: B, layer: 1, pos: 39
type: A, layer: 1, pos: 33

Time for candidate selection: 0.09 seconds

### Candidate
type: B, layer: 1, pos: 11

### Candidate
type: A, layer: 1, pos: 11

### Candidate
type: B, layer: 1, pos: 34

### Candidate
type: A, layer: 1, pos: 34

### Candidate
type: B, layer: 1, pos: 47

### Candidate
type: A, layer: 1, pos: 12

### Candidate
type: B, layer: 1, pos: 12

### Candidate
type: A, layer: 1, pos: 47

### Candidate
type: B, layer: 1, pos: 9

### Candidate
type: B, layer: 1, pos: 31

## Relational analysis of NS_A1_A2_B1_B2_A2_B1_A2_B2_A2_A1_B1
Optimization infeasible because this subproblem isn't reachable.

### Candidate
type: A, layer: 1, pos: 31

### Candidate
type: B, layer: 1, pos: 42

## Relational analysis of NS_A1_A2_B1_B2_A2_B1_A2_B2_A2_A1_B1
Optimization infeasible because this subproblem isn't reachable.

### Candidate
type: B, layer: 1, pos: 16

## Relational analysis of NS_A1_A2_B1_B2_A2_B1_A2_B2_A2_A1_B1
Optimization infeasible because this subproblem isn't reachable.

### Candidate
type: A, layer: 1, pos: 39

## Relational analysis of NS_A1_A2_B1_B2_A2_B1_A2_B2_A2_A1_A1

### Relational analysis result of NS_A1_A2_B1_B2_A2_B1_A2_B2_A2_A1_A1
Status: Status.UNKNOWN
Output dim: 3, lower bound: -20.6028837, upper bound: 20.6039567
time: 0.73 seconds

## Relational analysis of NS_A1_A2_B1_B2_A2_B1_A2_B2_A2_A1_A2

### Relational analysis result of NS_A1_A2_B1_B2_A2_B1_A2_B2_A2_A1_A2
Status: Status.UNKNOWN
Output dim: 3, lower bound: -20.6022695, upper bound: 20.6039479
time: 0.78 seconds

## BFS NS instance: NS_A1_A2_B1_B2_A2_B1_A2_B2_A2_A2

### Backsubstitution after applying NS history:
0: -3.4638333, 3.0918503, -3.0771875, 2.7368279, -6.2006612, 6.1690369
1: -13.6751289, 11.9739399, -12.1525078, 10.6178522, -24.2929764, 24.1264439
2: -6.8318210, 11.2329483, -6.0879107, 9.9097996, -16.7416210, 17.3208580
3: -11.9446383, 10.9590368, -10.6401510, 9.7027712, -21.6474094, 21.5991879
4: -8.7379341, 11.4238663, -7.7787948, 10.0979300, -18.8358650, 19.2026577

Time for backsubstitution: 1.32 seconds

### NS candidates at layer 1
type: B, layer: 1, pos: 11
type: A, layer: 1, pos: 11
type: A, layer: 1, pos: 34
type: B, layer: 1, pos: 34
type: B, layer: 1, pos: 47
type: A, layer: 1, pos: 12
type: B, layer: 1, pos: 12
type: B, layer: 1, pos: 9
type: B, layer: 1, pos: 42
type: B, layer: 1, pos: 16
type: A, layer: 1, pos: 16
type: A, layer: 1, pos: 39
type: B, layer: 1, pos: 33
type: B, layer: 1, pos: 31
type: A, layer: 1, pos: 31
type: A, layer: 1, pos: 47
type: A, layer: 1, pos: 23
type: B, layer: 1, pos: 39
type: B, layer: 1, pos: 23
type: A, layer: 1, pos: 33

Time for candidate selection: 0.11 seconds

### Candidate
type: B, layer: 1, pos: 11

### Candidate
type: A, layer: 1, pos: 11

### Candidate
type: A, layer: 1, pos: 34

### Candidate
type: B, layer: 1, pos: 34

### Candidate
type: B, layer: 1, pos: 47

### Candidate
type: A, layer: 1, pos: 12

### Candidate
type: B, layer: 1, pos: 12

### Candidate
type: B, layer: 1, pos: 9

### Candidate
type: B, layer: 1, pos: 42

## Relational analysis of NS_A1_A2_B1_B2_A2_B1_A2_B2_A2_A2_B1
Optimization infeasible because this subproblem isn't reachable.

### Candidate
type: B, layer: 1, pos: 16

## Relational analysis of NS_A1_A2_B1_B2_A2_B1_A2_B2_A2_A2_B1
Optimization infeasible because this subproblem isn't reachable.

### Candidate
type: A, layer: 1, pos: 16

### Candidate
type: A, layer: 1, pos: 39

## Relational analysis of NS_A1_A2_B1_B2_A2_B1_A2_B2_A2_A2_A1

### Relational analysis result of NS_A1_A2_B1_B2_A2_B1_A2_B2_A2_A2_A1
Status: Status.UNKNOWN
Output dim: 3, lower bound: -20.6028837, upper bound: 20.6040657
time: 0.95 seconds

## Relational analysis of NS_A1_A2_B1_B2_A2_B1_A2_B2_A2_A2_A2

### Relational analysis result of NS_A1_A2_B1_B2_A2_B1_A2_B2_A2_A2_A2
Status: Status.UNKNOWN
Output dim: 3, lower bound: -20.6022695, upper bound: 20.6040673
time: 0.98 seconds

## BFS NS instance: NS_A1_A2_B2_B1_B2_B1_A2_A1_B1_A1

### Backsubstitution after applying NS history:
0: -2.7816570, 2.4818509, -3.1513410, 2.8146360, -5.5962925, 5.6331906
1: -10.9689932, 9.6364336, -12.4139872, 10.8930225, -21.8620148, 22.0504208
2: -5.4935513, 9.0054016, -6.2274179, 10.2425604, -15.7361116, 15.2328196
3: -9.6097250, 8.8251762, -10.8576565, 9.9981375, -19.6078625, 19.6828251
4: -7.0272222, 9.1884327, -7.9622679, 10.4328938, -17.4601135, 17.1506996

Time for backsubstitution: 1.70 seconds

### NS candidates at layer 1
type: A, layer: 1, pos: 11
type: B, layer: 1, pos: 11
type: B, layer: 1, pos: 34
type: A, layer: 1, pos: 34
type: B, layer: 1, pos: 12
type: A, layer: 1, pos: 12
type: B, layer: 1, pos: 9
type: B, layer: 1, pos: 39
type: A, layer: 1, pos: 39
type: B, layer: 1, pos: 42
type: B, layer: 1, pos: 16
type: B, layer: 1, pos: 33
type: A, layer: 1, pos: 4
type: A, layer: 1, pos: 16
type: B, layer: 1, pos: 31
type: A, layer: 1, pos: 33
type: B, layer: 1, pos: 23
type: A, layer: 1, pos: 23
type: A, layer: 1, pos: 47
type: A, layer: 1, pos: 30

Time for candidate selection: 0.14 seconds

### Candidate
type: A, layer: 1, pos: 11

### Candidate
type: B, layer: 1, pos: 11

### Candidate
type: B, layer: 1, pos: 34

### Candidate
type: A, layer: 1, pos: 34

### Candidate
type: B, layer: 1, pos: 12

### Candidate
type: A, layer: 1, pos: 12

### Candidate
type: B, layer: 1, pos: 9

### Candidate
type: B, layer: 1, pos: 39

## Relational analysis of NS_A1_A2_B2_B1_B2_B1_A2_A1_B1_A1_B1

### Relational analysis result of NS_A1_A2_B2_B1_B2_B1_A2_A1_B1_A1_B1
Status: Status.UNKNOWN
Output dim: 3, lower bound: -20.6040526, upper bound: 20.6032748
time: 0.83 seconds

## Relational analysis of NS_A1_A2_B2_B1_B2_B1_A2_A1_B1_A1_B2
Optimization infeasible because this subproblem isn't reachable.

### Candidate
type: A, layer: 1, pos: 39

### Candidate
type: B, layer: 1, pos: 42

## Relational analysis of NS_A1_A2_B2_B1_B2_B1_A2_A1_B1_A1_B1
Optimization infeasible because this subproblem isn't reachable.

### Candidate
type: B, layer: 1, pos: 16

## Relational analysis of NS_A1_A2_B2_B1_B2_B1_A2_A1_B1_A1_B1
Optimization infeasible because this subproblem isn't reachable.

### Candidate
type: B, layer: 1, pos: 33

## Relational analysis of NS_A1_A2_B2_B1_B2_B1_A2_A1_B1_A1_B1

### Relational analysis result of NS_A1_A2_B2_B1_B2_B1_A2_A1_B1_A1_B1
Status: Status.UNKNOWN
Output dim: 3, lower bound: -20.6039791, upper bound: 20.6032449
time: 0.76 seconds

## Relational analysis of NS_A1_A2_B2_B1_B2_B1_A2_A1_B1_A1_B2

### Relational analysis result of NS_A1_A2_B2_B1_B2_B1_A2_A1_B1_A1_B2
Status: Status.UNKNOWN
Output dim: 3, lower bound: -20.6039796, upper bound: 20.6033230
time: 1.29 seconds

## BFS NS instance: NS_A1_A2_B2_B1_B2_B1_A2_A1_B1_A2

### Backsubstitution after applying NS history:
0: -2.9391685, 2.6167705, -3.1513410, 2.8146360, -5.7538028, 5.7681108
1: -11.6126900, 10.1238680, -12.4139872, 10.8930225, -22.5057106, 22.5378551
2: -5.8127203, 9.5147934, -6.2274179, 10.2425604, -16.0552807, 15.7422094
3: -10.1638746, 9.2626143, -10.8576565, 9.9981375, -20.1620121, 20.1202641
4: -7.4215121, 9.6846666, -7.9622679, 10.4328938, -17.8544006, 17.6469345

Time for backsubstitution: 1.61 seconds

### NS candidates at layer 1
type: A, layer: 1, pos: 11
type: B, layer: 1, pos: 11
type: B, layer: 1, pos: 34
type: A, layer: 1, pos: 34
type: B, layer: 1, pos: 12
type: A, layer: 1, pos: 12
type: B, layer: 1, pos: 9
type: A, layer: 1, pos: 39
type: B, layer: 1, pos: 39
type: B, layer: 1, pos: 42
type: B, layer: 1, pos: 31
type: B, layer: 1, pos: 16
type: B, layer: 1, pos: 33
type: A, layer: 1, pos: 4
type: A, layer: 1, pos: 16
type: A, layer: 1, pos: 33
type: B, layer: 1, pos: 23
type: A, layer: 1, pos: 23
type: A, layer: 1, pos: 47
type: A, layer: 1, pos: 30

Time for candidate selection: 0.14 seconds

### Candidate
type: A, layer: 1, pos: 11

### Candidate
type: B, layer: 1, pos: 11

### Candidate
type: B, layer: 1, pos: 34

### Candidate
type: A, layer: 1, pos: 34

### Candidate
type: B, layer: 1, pos: 12

### Candidate
type: A, layer: 1, pos: 12

### Candidate
type: B, layer: 1, pos: 9

### Candidate
type: A, layer: 1, pos: 39

## Relational analysis of NS_A1_A2_B2_B1_B2_B1_A2_A1_B1_A2_A1
Optimization infeasible because this subproblem isn't reachable.

### Candidate
type: B, layer: 1, pos: 39

### Candidate
type: B, layer: 1, pos: 42

## Relational analysis of NS_A1_A2_B2_B1_B2_B1_A2_A1_B1_A2_B1
Optimization infeasible because this subproblem isn't reachable.

### Candidate
type: B, layer: 1, pos: 31

## Relational analysis of NS_A1_A2_B2_B1_B2_B1_A2_A1_B1_A2_B1
Optimization infeasible because this subproblem isn't reachable.

### Candidate
type: B, layer: 1, pos: 16

## Relational analysis of NS_A1_A2_B2_B1_B2_B1_A2_A1_B1_A2_B1
Optimization infeasible because this subproblem isn't reachable.

### Candidate
type: B, layer: 1, pos: 33

## Relational analysis of NS_A1_A2_B2_B1_B2_B1_A2_A1_B1_A2_B1

### Relational analysis result of NS_A1_A2_B2_B1_B2_B1_A2_A1_B1_A2_B1
Status: Status.UNKNOWN
Output dim: 3, lower bound: -20.6037023, upper bound: 20.6032420
time: 0.71 seconds

## Relational analysis of NS_A1_A2_B2_B1_B2_B1_A2_A1_B1_A2_B2

### Relational analysis result of NS_A1_A2_B2_B1_B2_B1_A2_A1_B1_A2_B2
Status: Status.UNKNOWN
Output dim: 3, lower bound: -20.6037028, upper bound: 20.6033201
time: 0.79 seconds

## BFS NS instance: NS_A1_A2_B2_B1_B2_B1_A2_A2_A1_B1

### Backsubstitution after applying NS history:
0: -3.3850455, 3.0251527, -3.1513410, 2.8146360, -6.1996803, 6.1764927
1: -13.3516960, 11.7186394, -12.4139872, 10.8930225, -24.2447186, 24.1326237
2: -6.6723394, 10.9979820, -6.2274179, 10.2425604, -16.9148998, 17.2253990
3: -11.6664848, 10.7302942, -10.8576565, 9.9981375, -21.6646233, 21.5879440
4: -8.5420465, 11.1814547, -7.9622679, 10.4328938, -18.9749355, 19.1437225

Time for backsubstitution: 1.52 seconds

### NS candidates at layer 1
type: B, layer: 1, pos: 11
type: A, layer: 1, pos: 11
type: B, layer: 1, pos: 34
type: A, layer: 1, pos: 34
type: A, layer: 1, pos: 12
type: B, layer: 1, pos: 12
type: B, layer: 1, pos: 9
type: A, layer: 1, pos: 39
type: B, layer: 1, pos: 42
type: B, layer: 1, pos: 39
type: B, layer: 1, pos: 33
type: B, layer: 1, pos: 31
type: B, layer: 1, pos: 16
type: A, layer: 1, pos: 4
type: A, layer: 1, pos: 16
type: A, layer: 1, pos: 23
type: B, layer: 1, pos: 23
type: A, layer: 1, pos: 33
type: A, layer: 1, pos: 47
type: A, layer: 1, pos: 30

Time for candidate selection: 0.12 seconds

### Candidate
type: B, layer: 1, pos: 11

### Candidate
type: A, layer: 1, pos: 11

### Candidate
type: B, layer: 1, pos: 34

### Candidate
type: A, layer: 1, pos: 34

### Candidate
type: A, layer: 1, pos: 12

### Candidate
type: B, layer: 1, pos: 12

### Candidate
type: B, layer: 1, pos: 9

### Candidate
type: A, layer: 1, pos: 39

### Candidate
type: B, layer: 1, pos: 42

### Candidate
type: B, layer: 1, pos: 39

### Candidate
type: B, layer: 1, pos: 33

## Relational analysis of NS_A1_A2_B2_B1_B2_B1_A2_A2_A1_B1_B1

### Relational analysis result of NS_A1_A2_B2_B1_B2_B1_A2_A2_A1_B1_B1
Status: Status.UNKNOWN
Output dim: 3, lower bound: -20.6034647, upper bound: 20.6022367
time: 0.73 seconds

## Relational analysis of NS_A1_A2_B2_B1_B2_B1_A2_A2_A1_B1_B2

### Relational analysis result of NS_A1_A2_B2_B1_B2_B1_A2_A2_A1_B1_B2
Status: Status.UNKNOWN
Output dim: 3, lower bound: -20.6035274, upper bound: 20.6024183
time: 0.75 seconds

## BFS NS instance: NS_A1_A2_B2_B1_B2_B1_A2_A2_A1_B2

### Backsubstitution after applying NS history:
0: -3.3850455, 3.0251527, -3.5829484, 3.2091122, -6.5941577, 6.6081009
1: -13.3516960, 11.7186394, -14.1194181, 12.3827600, -25.7344551, 25.8380566
2: -6.6723394, 10.9979820, -7.0389814, 11.7711287, -18.4434681, 18.0369606
3: -11.6664848, 10.7302942, -12.3018827, 11.3730946, -23.0395756, 23.0321693
4: -8.5420465, 11.1814547, -9.0094128, 11.9351234, -20.4771690, 20.1908646

Time for backsubstitution: 1.56 seconds

### NS candidates at layer 1
type: A, layer: 1, pos: 11
type: A, layer: 1, pos: 34
type: B, layer: 1, pos: 11
type: B, layer: 1, pos: 34
type: B, layer: 1, pos: 12
type: B, layer: 1, pos: 9
type: A, layer: 1, pos: 12
type: A, layer: 1, pos: 47
type: B, layer: 1, pos: 42
type: A, layer: 1, pos: 4
type: B, layer: 1, pos: 31
type: B, layer: 1, pos: 39
type: B, layer: 1, pos: 16
type: A, layer: 1, pos: 16
type: A, layer: 1, pos: 39
type: B, layer: 1, pos: 33
type: B, layer: 1, pos: 23
type: A, layer: 1, pos: 33
type: A, layer: 1, pos: 23
type: A, layer: 1, pos: 30

Time for candidate selection: 0.12 seconds

### Candidate
type: A, layer: 1, pos: 11

### Candidate
type: A, layer: 1, pos: 34

### Candidate
type: B, layer: 1, pos: 11

### Candidate
type: B, layer: 1, pos: 34

### Candidate
type: B, layer: 1, pos: 12

### Candidate
type: B, layer: 1, pos: 9

### Candidate
type: A, layer: 1, pos: 12

### Candidate
type: A, layer: 1, pos: 47

## Relational analysis of NS_A1_A2_B2_B1_B2_B1_A2_A2_A1_B2_A1
Optimization infeasible because this subproblem isn't reachable.

### Candidate
type: B, layer: 1, pos: 42

### Candidate
type: A, layer: 1, pos: 4

## Relational analysis of NS_A1_A2_B2_B1_B2_B1_A2_A2_A1_B2_A1

### Relational analysis result of NS_A1_A2_B2_B1_B2_B1_A2_A2_A1_B2_A1
Status: Status.VERIFIED
Output dim: 3, lower bound: -20.5993726, upper bound: 20.6023043
time: 0.71 seconds

## Relational analysis of NS_A1_A2_B2_B1_B2_B1_A2_A2_A1_B2_A2

### Relational analysis result of NS_A1_A2_B2_B1_B2_B1_A2_A2_A1_B2_A2
Status: Status.VERIFIED
Output dim: 3, lower bound: -20.5993726, upper bound: 20.6024183
time: 2.47 seconds

## NS Result
status: Status.UNKNOWN
execution time: (base) + (ns) = 3.53 + 416.53 = 420.06 seconds
