## Execution arguments:
Dataset: Dataset.ACAS
Network: ds/onnx/acasxu_op11/ACASXU_1_6.onnx
Relational property: GLOBAL_ROBUSTNESS
LP Analysis: True
Epsilon: None
Delta epsilon: 0.1
execution index: (10, None, 9)
Time budget: 420 seconds
Split limit: 100
Threshold: 20.860446436


## IAR start

### BASE IAR bounds
Layer (inp1_lb, inp1_ub, inp2_lb, inp2_ub, d_lb, d_ub)
0: (-6.6128454, 18.3588104, -6.6128454, 18.3588104, -24.9716549, 24.9716549)
1: (-16.5121365, 28.1943169, -16.5121365, 28.1943169, -44.7064514, 44.7064514)
2: (-11.5172834, 25.5670700, -11.5172834, 25.5670700, -37.0843506, 37.0843506)
3: (-17.7977448, 31.1728382, -17.7977448, 31.1728382, -48.9705811, 48.9705811)
4: (-16.2582874, 31.5556946, -16.2582874, 31.5556946, -47.8139801, 47.8139801)

## BASE Relational Analysis

## BASE Result
execution time: IAR + RelationalAnalysis = 0.89 + 1.70 = 2.60 seconds
status: Status.UNKNOWN
relational distance
Output dim: 0, lower bound: -20.9652728, upper bound: 20.9652728

# Neuron Split (NS) starts

## BFS NS instance: NS

Time for backsubstitution: 0.01 seconds

### NS candidates at layer 1
type: A, layer: 1, pos: 12
type: A, layer: 1, pos: 1
type: A, layer: 1, pos: 47
type: A, layer: 1, pos: 9
type: A, layer: 1, pos: 6
type: A, layer: 1, pos: 38
type: A, layer: 1, pos: 39
type: A, layer: 1, pos: 28
type: A, layer: 1, pos: 48
type: A, layer: 1, pos: 0
type: A, layer: 1, pos: 24
type: A, layer: 1, pos: 37
type: A, layer: 1, pos: 41
type: A, layer: 1, pos: 20
type: A, layer: 1, pos: 15
type: A, layer: 1, pos: 45
type: A, layer: 1, pos: 26
type: A, layer: 1, pos: 31
type: A, layer: 1, pos: 5
type: A, layer: 1, pos: 8

Time for candidate selection: 0.05 seconds

### Candidate
type: A, layer: 1, pos: 12

## Relational analysis of NS_A1

### Relational analysis result of NS_A1
Status: Status.UNKNOWN
Output dim: 0, lower bound: -20.9652728, upper bound: 20.9648630
time: 0.59 seconds

## Relational analysis of NS_A2

### Relational analysis result of NS_A2
Status: Status.UNKNOWN
Output dim: 0, lower bound: -20.9648630, upper bound: 20.9648630
time: 0.54 seconds

## Summary of splitting at layer (split count: 0)
- Time for NS candidates: 1.20 seconds
NS_A1, status: Status.UNKNOWN, split count: 1, time: 1.20
Output dim: 0, lower bound: -20.9652728, upper bound: 20.9648630
NS_A2, status: Status.UNKNOWN, split count: 1, time: 1.20
Output dim: 0, lower bound: -20.9648630, upper bound: 20.9648630

## BFS NS instance: NS_A1

### Backsubstitution after applying NS history:
0: -6.2940121, 17.5322781, -6.4682894, 17.9821701, -24.2761822, 24.0005684
1: -15.6561022, 27.0123844, -16.1270905, 27.6529121, -43.3090096, 43.1394730
2: -10.9337883, 24.4664974, -11.2533112, 25.0652580, -35.9990425, 35.7197952
3: -16.9244709, 29.8317928, -17.4036999, 30.5590115, -47.4834785, 47.2354927
4: -15.5113659, 30.1773357, -15.9182625, 30.9278564, -46.4392166, 46.0955963

Time for backsubstitution: 0.79 seconds

### NS candidates at layer 1
type: B, layer: 1, pos: 1
type: B, layer: 1, pos: 12
type: B, layer: 1, pos: 47
type: B, layer: 1, pos: 9
type: B, layer: 1, pos: 6
type: B, layer: 1, pos: 39
type: B, layer: 1, pos: 28
type: B, layer: 1, pos: 38
type: B, layer: 1, pos: 48
type: B, layer: 1, pos: 0
type: B, layer: 1, pos: 24
type: B, layer: 1, pos: 37
type: B, layer: 1, pos: 41
type: B, layer: 1, pos: 20
type: B, layer: 1, pos: 15
type: B, layer: 1, pos: 26
type: B, layer: 1, pos: 45
type: B, layer: 1, pos: 8
type: B, layer: 1, pos: 31
type: B, layer: 1, pos: 5

Time for candidate selection: 0.05 seconds

### Candidate
type: B, layer: 1, pos: 1

## Relational analysis of NS_A1_B1

### Relational analysis result of NS_A1_B1
Status: Status.UNKNOWN
Output dim: 0, lower bound: -20.9651837, upper bound: 20.9642482
time: 0.55 seconds

## Relational analysis of NS_A1_B2

### Relational analysis result of NS_A1_B2
Status: Status.UNKNOWN
Output dim: 0, lower bound: -20.9652728, upper bound: 20.9648630
time: 0.61 seconds

## BFS NS instance: NS_A2

### Backsubstitution after applying NS history:
0: -6.3099775, 17.4615860, -6.4305573, 17.8925438, -24.2025223, 23.8921432
1: -15.6268415, 26.9265728, -16.0598736, 27.5127029, -43.1395454, 42.9864426
2: -10.9621468, 24.3374519, -11.1947908, 24.9361191, -35.8982620, 35.5322418
3: -16.8688698, 29.7598553, -17.3018456, 30.4061832, -47.2750473, 47.0616989
4: -15.4283810, 29.9138927, -15.7858572, 30.7771568, -46.2055283, 45.6997452

Time for backsubstitution: 0.79 seconds

### NS candidates at layer 1
type: B, layer: 1, pos: 47
type: B, layer: 1, pos: 1
type: B, layer: 1, pos: 12
type: B, layer: 1, pos: 9
type: B, layer: 1, pos: 6
type: B, layer: 1, pos: 39
type: B, layer: 1, pos: 38
type: B, layer: 1, pos: 28
type: B, layer: 1, pos: 48
type: B, layer: 1, pos: 0
type: B, layer: 1, pos: 24
type: B, layer: 1, pos: 41
type: B, layer: 1, pos: 37
type: B, layer: 1, pos: 20
type: B, layer: 1, pos: 15
type: B, layer: 1, pos: 26
type: B, layer: 1, pos: 31
type: B, layer: 1, pos: 45
type: B, layer: 1, pos: 5
type: B, layer: 1, pos: 8

Time for candidate selection: 0.05 seconds

### Candidate
type: B, layer: 1, pos: 47

## Relational analysis of NS_A2_B1

### Relational analysis result of NS_A2_B1
Status: Status.UNKNOWN
Output dim: 0, lower bound: -20.9569857, upper bound: 20.9544784
time: 0.54 seconds

## Relational analysis of NS_A2_B2

### Relational analysis result of NS_A2_B2
Status: Status.UNKNOWN
Output dim: 0, lower bound: -20.9480390, upper bound: 20.9480390
time: 0.60 seconds

## Summary of splitting at layer (split count: 1)
- Time for NS candidates: 2.01 seconds
NS_A1_B1, status: Status.UNKNOWN, split count: 2, time: 2.01
Output dim: 0, lower bound: -20.9651837, upper bound: 20.9642482
NS_A1_B2, status: Status.UNKNOWN, split count: 2, time: 2.01
Output dim: 0, lower bound: -20.9652728, upper bound: 20.9648630
NS_A2_B1, status: Status.UNKNOWN, split count: 2, time: 2.01
Output dim: 0, lower bound: -20.9569857, upper bound: 20.9544784
NS_A2_B2, status: Status.UNKNOWN, split count: 2, time: 2.01
Output dim: 0, lower bound: -20.9480390, upper bound: 20.9480390

## BFS NS instance: NS_A1_B1

### Backsubstitution after applying NS history:
0: -6.0110188, 16.7995281, -6.2949438, 17.4614010, -23.4724140, 23.0944691
1: -14.9163046, 25.9396820, -15.6868448, 26.7338161, -41.6501198, 41.6265259
2: -10.4211235, 23.4742451, -10.9559574, 24.2959251, -34.7170486, 34.4302025
3: -16.1454544, 28.6220970, -16.9406414, 29.6053753, -45.7508316, 45.5627365
4: -14.8167048, 28.9426117, -15.4891968, 29.9871178, -44.8038216, 44.4318085

Time for backsubstitution: 0.79 seconds

### NS candidates at layer 1
type: A, layer: 1, pos: 47
type: A, layer: 1, pos: 1
type: A, layer: 1, pos: 39
type: A, layer: 1, pos: 9
type: A, layer: 1, pos: 6
type: A, layer: 1, pos: 28
type: A, layer: 1, pos: 48
type: A, layer: 1, pos: 38
type: A, layer: 1, pos: 0
type: A, layer: 1, pos: 24
type: A, layer: 1, pos: 41
type: A, layer: 1, pos: 37
type: A, layer: 1, pos: 8
type: A, layer: 1, pos: 15
type: A, layer: 1, pos: 26
type: A, layer: 1, pos: 20
type: A, layer: 1, pos: 5
type: A, layer: 1, pos: 45
type: A, layer: 1, pos: 31

Time for candidate selection: 0.05 seconds

### Candidate
type: A, layer: 1, pos: 47

## Relational analysis of NS_A1_B1_A1

### Relational analysis result of NS_A1_B1_A1
Status: Status.UNKNOWN
Output dim: 0, lower bound: -20.9488332, upper bound: 20.9523051
time: 0.55 seconds

## Relational analysis of NS_A1_B1_A2

### Relational analysis result of NS_A1_B1_A2
Status: Status.UNKNOWN
Output dim: 0, lower bound: -20.9431946, upper bound: 20.9438155
time: 0.61 seconds

## BFS NS instance: NS_A1_B2

### Backsubstitution after applying NS history:
0: -6.2940121, 17.5322781, -6.1991372, 17.3003941, -23.5944061, 23.7314148
1: -15.6561022, 27.0123844, -15.4168787, 26.6559677, -42.3120689, 42.4292603
2: -10.9337883, 24.4664974, -10.7691479, 24.1451225, -35.0789108, 35.2356453
3: -16.9244709, 29.8317928, -16.6685352, 29.4332523, -46.3577232, 46.5003281
4: -15.5113659, 30.1773357, -15.2781858, 29.7823944, -45.2937584, 45.4555206

Time for backsubstitution: 0.79 seconds

### NS candidates at layer 1
type: A, layer: 1, pos: 1
type: A, layer: 1, pos: 47
type: A, layer: 1, pos: 9
type: A, layer: 1, pos: 39
type: A, layer: 1, pos: 6
type: A, layer: 1, pos: 28
type: A, layer: 1, pos: 38
type: A, layer: 1, pos: 48
type: A, layer: 1, pos: 0
type: A, layer: 1, pos: 24
type: A, layer: 1, pos: 41
type: A, layer: 1, pos: 37
type: A, layer: 1, pos: 20
type: A, layer: 1, pos: 8
type: A, layer: 1, pos: 15
type: A, layer: 1, pos: 31
type: A, layer: 1, pos: 26
type: A, layer: 1, pos: 45
type: A, layer: 1, pos: 5

Time for candidate selection: 0.05 seconds

### Candidate
type: A, layer: 1, pos: 1

## Relational analysis of NS_A1_B2_A1
Optimization infeasible because this subproblem isn't reachable.

### Candidate
type: A, layer: 1, pos: 47

## Relational analysis of NS_A1_B2_A1

### Relational analysis result of NS_A1_B2_A1
Status: Status.UNKNOWN
Output dim: 0, lower bound: -20.9548571, upper bound: 20.9569857
time: 0.62 seconds

## Relational analysis of NS_A1_B2_A2

### Relational analysis result of NS_A1_B2_A2
Status: Status.UNKNOWN
Output dim: 0, lower bound: -20.9517087, upper bound: 20.9516156
time: 0.64 seconds

## BFS NS instance: NS_A2_B1

### Backsubstitution after applying NS history:
0: -6.1309218, 17.0163708, -6.0355606, 16.9193726, -23.0502930, 23.0519295
1: -15.1835890, 26.2758141, -15.1791325, 25.9912262, -41.1748047, 41.4549446
2: -10.6458235, 23.7349014, -10.5353680, 23.5500202, -34.1958427, 34.2702713
3: -16.3854332, 29.0190868, -16.2799644, 28.7346630, -45.1200905, 45.2990494
4: -14.9737673, 29.1772766, -14.6903238, 29.1178627, -44.0916290, 43.8675995

Time for backsubstitution: 0.80 seconds

### NS candidates at layer 1
type: A, layer: 1, pos: 1
type: A, layer: 1, pos: 47
type: A, layer: 1, pos: 39
type: A, layer: 1, pos: 9
type: A, layer: 1, pos: 6
type: A, layer: 1, pos: 48
type: A, layer: 1, pos: 28
type: A, layer: 1, pos: 38
type: A, layer: 1, pos: 0
type: A, layer: 1, pos: 24
type: A, layer: 1, pos: 37
type: A, layer: 1, pos: 41
type: A, layer: 1, pos: 20
type: A, layer: 1, pos: 31
type: A, layer: 1, pos: 15
type: A, layer: 1, pos: 8
type: A, layer: 1, pos: 26
type: A, layer: 1, pos: 45
type: A, layer: 1, pos: 5

Time for candidate selection: 0.05 seconds

### Candidate
type: A, layer: 1, pos: 1

## Relational analysis of NS_A2_B1_A1

### Relational analysis result of NS_A2_B1_A1
Status: Status.UNKNOWN
Output dim: 0, lower bound: -20.9523051, upper bound: 20.9485296
time: 0.93 seconds

## Relational analysis of NS_A2_B1_A2

### Relational analysis result of NS_A2_B1_A2
Status: Status.UNKNOWN
Output dim: 0, lower bound: -20.9569857, upper bound: 20.9544784
time: 1.14 seconds

## BFS NS instance: NS_A2_B2

### Backsubstitution after applying NS history:
0: -6.3099775, 17.4615860, -6.3495493, 17.6864700, -23.9964485, 23.8111343
1: -15.6268415, 26.9265728, -15.8588676, 27.2091198, -42.8359566, 42.7854385
2: -10.9621468, 24.3374519, -11.0539274, 24.6576653, -35.6197968, 35.3913803
3: -16.8688698, 29.7598553, -17.0815296, 30.0639439, -46.9328079, 46.8413849
4: -15.4283810, 29.9138927, -15.5757790, 30.4333534, -45.8617325, 45.4896698

Time for backsubstitution: 0.81 seconds

### NS candidates at layer 1
type: A, layer: 1, pos: 1
type: A, layer: 1, pos: 47
type: A, layer: 1, pos: 39
type: A, layer: 1, pos: 9
type: A, layer: 1, pos: 6
type: A, layer: 1, pos: 28
type: A, layer: 1, pos: 48
type: A, layer: 1, pos: 38
type: A, layer: 1, pos: 0
type: A, layer: 1, pos: 24
type: A, layer: 1, pos: 37
type: A, layer: 1, pos: 41
type: A, layer: 1, pos: 20
type: A, layer: 1, pos: 15
type: A, layer: 1, pos: 31
type: A, layer: 1, pos: 8
type: A, layer: 1, pos: 26
type: A, layer: 1, pos: 45
type: A, layer: 1, pos: 5

Time for candidate selection: 0.05 seconds

### Candidate
type: A, layer: 1, pos: 1

## Relational analysis of NS_A2_B2_A1

### Relational analysis result of NS_A2_B2_A1
Status: Status.UNKNOWN
Output dim: 0, lower bound: -20.9408792, upper bound: 20.9403090
time: 0.55 seconds

## Relational analysis of NS_A2_B2_A2

### Relational analysis result of NS_A2_B2_A2
Status: Status.UNKNOWN
Output dim: 0, lower bound: -20.9480390, upper bound: 20.9480390
time: 0.83 seconds

## Summary of splitting at layer (split count: 2)
- Time for NS candidates: 2.25 seconds
NS_A1_B1_A1, status: Status.UNKNOWN, split count: 3, time: 2.25
Output dim: 0, lower bound: -20.9488332, upper bound: 20.9523051
NS_A1_B1_A2, status: Status.UNKNOWN, split count: 3, time: 2.25
Output dim: 0, lower bound: -20.9431946, upper bound: 20.9438155
NS_A1_B2_A1, status: Status.UNKNOWN, split count: 3, time: 2.25
Output dim: 0, lower bound: -20.9548571, upper bound: 20.9569857
NS_A1_B2_A2, status: Status.UNKNOWN, split count: 3, time: 2.25
Output dim: 0, lower bound: -20.9517087, upper bound: 20.9516156
NS_A2_B1_A1, status: Status.UNKNOWN, split count: 3, time: 2.25
Output dim: 0, lower bound: -20.9523051, upper bound: 20.9485296
NS_A2_B1_A2, status: Status.UNKNOWN, split count: 3, time: 2.25
Output dim: 0, lower bound: -20.9569857, upper bound: 20.9544784
NS_A2_B2_A1, status: Status.UNKNOWN, split count: 3, time: 2.25
Output dim: 0, lower bound: -20.9408792, upper bound: 20.9403090
NS_A2_B2_A2, status: Status.UNKNOWN, split count: 3, time: 2.25
Output dim: 0, lower bound: -20.9480390, upper bound: 20.9480390

## BFS NS instance: NS_A1_B1_A1

### Backsubstitution after applying NS history:
0: -5.6546221, 15.9332199, -6.1445241, 17.0822372, -22.7368526, 22.0777416
1: -14.1608744, 24.5610847, -15.3202667, 26.1773071, -40.3381805, 39.8813515
2: -9.8375568, 22.2305279, -10.6948204, 23.7860107, -33.6235657, 32.9253464
3: -15.2379446, 27.1142273, -16.5344658, 28.9753017, -44.2132416, 43.6486931
4: -13.7985134, 27.4670696, -15.1078596, 29.3623276, -43.1608429, 42.5749207

Time for backsubstitution: 0.80 seconds

### NS candidates at layer 1
type: B, layer: 1, pos: 12
type: B, layer: 1, pos: 47
type: B, layer: 1, pos: 9
type: B, layer: 1, pos: 6
type: B, layer: 1, pos: 48
type: B, layer: 1, pos: 39
type: B, layer: 1, pos: 28
type: B, layer: 1, pos: 38
type: B, layer: 1, pos: 0
type: B, layer: 1, pos: 24
type: B, layer: 1, pos: 41
type: B, layer: 1, pos: 37
type: B, layer: 1, pos: 8
type: B, layer: 1, pos: 31
type: B, layer: 1, pos: 20
type: B, layer: 1, pos: 15
type: B, layer: 1, pos: 26
type: B, layer: 1, pos: 45
type: B, layer: 1, pos: 5

Time for candidate selection: 0.05 seconds

### Candidate
type: B, layer: 1, pos: 12

## Relational analysis of NS_A1_B1_A1_B1
Optimization infeasible because this subproblem isn't reachable.

### Candidate
type: B, layer: 1, pos: 47

## Relational analysis of NS_A1_B1_A1_B1

### Relational analysis result of NS_A1_B1_A1_B1
Status: Status.UNKNOWN
Output dim: 0, lower bound: -20.9431946, upper bound: 20.9438155
time: 0.54 seconds

## Relational analysis of NS_A1_B1_A1_B2

### Relational analysis result of NS_A1_B1_A1_B2
Status: Status.UNKNOWN
Output dim: 0, lower bound: -20.9431946, upper bound: 20.9438155
time: 0.55 seconds

## BFS NS instance: NS_A1_B1_A2

### Backsubstitution after applying NS history:
0: -5.9220600, 16.5755711, -6.2949438, 17.4614010, -23.3834591, 22.8705120
1: -14.6959372, 25.6085052, -15.6868448, 26.7338161, -41.4297485, 41.2953491
2: -10.2669477, 23.1718216, -10.9559574, 24.2959251, -34.5628738, 34.1277771
3: -15.9044065, 28.2489166, -16.9406414, 29.6053753, -45.5097771, 45.1895523
4: -14.5871439, 28.5687160, -15.4891968, 29.9871178, -44.5742607, 44.0579147

Time for backsubstitution: 0.80 seconds

### NS candidates at layer 1
type: B, layer: 1, pos: 12
type: B, layer: 1, pos: 47
type: B, layer: 1, pos: 9
type: B, layer: 1, pos: 6
type: B, layer: 1, pos: 48
type: B, layer: 1, pos: 39
type: B, layer: 1, pos: 28
type: B, layer: 1, pos: 38
type: B, layer: 1, pos: 0
type: B, layer: 1, pos: 24
type: B, layer: 1, pos: 37
type: B, layer: 1, pos: 41
type: B, layer: 1, pos: 8
type: B, layer: 1, pos: 31
type: B, layer: 1, pos: 20
type: B, layer: 1, pos: 26
type: B, layer: 1, pos: 5
type: B, layer: 1, pos: 45
type: B, layer: 1, pos: 15

Time for candidate selection: 0.05 seconds

### Candidate
type: B, layer: 1, pos: 12

## Relational analysis of NS_A1_B1_A2_B1
Optimization infeasible because this subproblem isn't reachable.

### Candidate
type: B, layer: 1, pos: 47

## Relational analysis of NS_A1_B1_A2_B1

### Relational analysis result of NS_A1_B1_A2_B1
Status: Status.UNKNOWN
Output dim: 0, lower bound: -20.9431946, upper bound: 20.9438155
time: 0.54 seconds

## Relational analysis of NS_A1_B1_A2_B2

### Relational analysis result of NS_A1_B1_A2_B2
Status: Status.UNKNOWN
Output dim: 0, lower bound: -20.9431946, upper bound: 20.9438155
time: 0.50 seconds

## BFS NS instance: NS_A1_B2_A1

### Backsubstitution after applying NS history:
0: -5.9211702, 16.6253052, -6.0454345, 16.9120750, -22.8332443, 22.6707401
1: -14.8620787, 25.5718231, -15.0394192, 26.0870304, -40.9491081, 40.6112404
2: -10.3212500, 23.1647587, -10.5003700, 23.6196594, -33.9409027, 33.6651306
3: -15.9681101, 28.2507477, -16.2547684, 28.7904243, -44.7585297, 44.5055122
4: -14.4465303, 28.6296043, -14.8838062, 29.1371937, -43.5837250, 43.5134087

Time for backsubstitution: 0.80 seconds

### NS candidates at layer 1
type: B, layer: 1, pos: 12
type: B, layer: 1, pos: 47
type: B, layer: 1, pos: 9
type: B, layer: 1, pos: 6
type: B, layer: 1, pos: 39
type: B, layer: 1, pos: 28
type: B, layer: 1, pos: 38
type: B, layer: 1, pos: 48
type: B, layer: 1, pos: 0
type: B, layer: 1, pos: 24
type: B, layer: 1, pos: 41
type: B, layer: 1, pos: 37
type: B, layer: 1, pos: 20
type: B, layer: 1, pos: 8
type: B, layer: 1, pos: 26
type: B, layer: 1, pos: 15
type: B, layer: 1, pos: 31
type: B, layer: 1, pos: 45
type: B, layer: 1, pos: 5

Time for candidate selection: 0.05 seconds

### Candidate
type: B, layer: 1, pos: 12

## Relational analysis of NS_A1_B2_A1_B1

### Relational analysis result of NS_A1_B2_A1_B1
Status: Status.UNKNOWN
Output dim: 0, lower bound: -20.9548571, upper bound: 20.9569857
time: 0.59 seconds

## Relational analysis of NS_A1_B2_A1_B2

### Relational analysis result of NS_A1_B2_A1_B2
Status: Status.UNKNOWN
Output dim: 0, lower bound: -20.9548571, upper bound: 20.9569857
time: 0.60 seconds

## BFS NS instance: NS_A1_B2_A2

### Backsubstitution after applying NS history:
0: -6.2047129, 17.3081169, -6.1991372, 17.3003941, -23.5051079, 23.5072517
1: -15.4355631, 26.6807861, -15.4168787, 26.6559677, -42.0915298, 42.0976639
2: -10.7784548, 24.1633434, -10.7691479, 24.1451225, -34.9235725, 34.9324913
3: -16.6825981, 29.4580708, -16.6685352, 29.4332523, -46.1158524, 46.1266022
4: -15.2801237, 29.8028202, -15.2781858, 29.7823944, -45.0625191, 45.0810051

Time for backsubstitution: 0.81 seconds

### NS candidates at layer 1
type: B, layer: 1, pos: 12
type: B, layer: 1, pos: 47
type: B, layer: 1, pos: 9
type: B, layer: 1, pos: 6
type: B, layer: 1, pos: 39
type: B, layer: 1, pos: 28
type: B, layer: 1, pos: 38
type: B, layer: 1, pos: 48
type: B, layer: 1, pos: 0
type: B, layer: 1, pos: 24
type: B, layer: 1, pos: 41
type: B, layer: 1, pos: 37
type: B, layer: 1, pos: 20
type: B, layer: 1, pos: 15
type: B, layer: 1, pos: 8
type: B, layer: 1, pos: 26
type: B, layer: 1, pos: 31
type: B, layer: 1, pos: 45
type: B, layer: 1, pos: 5

Time for candidate selection: 0.05 seconds

### Candidate
type: B, layer: 1, pos: 12

## Relational analysis of NS_A1_B2_A2_B1

### Relational analysis result of NS_A1_B2_A2_B1
Status: Status.UNKNOWN
Output dim: 0, lower bound: -20.9517087, upper bound: 20.9516156
time: 1.11 seconds

## Relational analysis of NS_A1_B2_A2_B2

### Relational analysis result of NS_A1_B2_A2_B2
Status: Status.UNKNOWN
Output dim: 0, lower bound: -20.9517087, upper bound: 20.9516156
time: 0.64 seconds

## BFS NS instance: NS_A2_B1_A1

### Backsubstitution after applying NS history:
0: -7.1646075, 19.4706726, -5.7873869, 16.2764168, -23.4410229, 25.2580547
1: -18.0290127, 29.6563625, -14.5204782, 25.0533028, -43.0823135, 44.1768379
2: -12.4878607, 26.8986130, -10.0832005, 22.6807919, -35.1686401, 36.9818115
3: -19.2521210, 32.9131813, -15.5978880, 27.6773338, -46.9294510, 48.5110626
4: -17.3505459, 33.1918945, -14.0913458, 28.0347404, -45.3852768, 47.2832413

Time for backsubstitution: 0.81 seconds

### NS candidates at layer 1
type: B, layer: 1, pos: 12
type: B, layer: 1, pos: 9
type: B, layer: 1, pos: 48
type: B, layer: 1, pos: 6
type: B, layer: 1, pos: 1
type: B, layer: 1, pos: 38
type: B, layer: 1, pos: 28
type: B, layer: 1, pos: 39
type: B, layer: 1, pos: 41
type: B, layer: 1, pos: 0
type: B, layer: 1, pos: 24
type: B, layer: 1, pos: 37
type: B, layer: 1, pos: 8
type: B, layer: 1, pos: 15
type: B, layer: 1, pos: 20
type: B, layer: 1, pos: 5
type: B, layer: 1, pos: 45
type: B, layer: 1, pos: 26
type: B, layer: 1, pos: 31

Time for candidate selection: 0.05 seconds

### Candidate
type: B, layer: 1, pos: 12

## Relational analysis of NS_A2_B1_A1_B1

### Relational analysis result of NS_A2_B1_A1_B1
Status: Status.UNKNOWN
Output dim: 0, lower bound: -20.9523051, upper bound: 20.9485296
time: 0.97 seconds

## Relational analysis of NS_A2_B1_A1_B2

### Relational analysis result of NS_A2_B1_A1_B2
Status: Status.UNKNOWN
Output dim: 0, lower bound: -20.9523051, upper bound: 20.9485296
time: 0.57 seconds

## BFS NS instance: NS_A2_B1_A2

### Backsubstitution after applying NS history:
0: -5.8769097, 16.3671246, -6.0355606, 16.9193726, -22.7962818, 22.4026852
1: -14.5033007, 25.3214855, -15.1791325, 25.9912262, -40.4945221, 40.5006180
2: -10.1820154, 22.8613892, -10.5353680, 23.5500202, -33.7320366, 33.3967590
3: -15.6849270, 27.9441319, -16.2799644, 28.7346630, -44.4195862, 44.2240906
4: -14.3726711, 28.0883980, -14.6903238, 29.1178627, -43.4905319, 42.7787209

Time for backsubstitution: 0.81 seconds

### NS candidates at layer 1
type: B, layer: 1, pos: 12
type: B, layer: 1, pos: 1
type: B, layer: 1, pos: 9
type: B, layer: 1, pos: 6
type: B, layer: 1, pos: 48
type: B, layer: 1, pos: 28
type: B, layer: 1, pos: 38
type: B, layer: 1, pos: 39
type: B, layer: 1, pos: 41
type: B, layer: 1, pos: 0
type: B, layer: 1, pos: 20
type: B, layer: 1, pos: 24
type: B, layer: 1, pos: 37
type: B, layer: 1, pos: 31
type: B, layer: 1, pos: 15
type: B, layer: 1, pos: 8
type: B, layer: 1, pos: 45
type: B, layer: 1, pos: 5
type: B, layer: 1, pos: 26

Time for candidate selection: 0.06 seconds

### Candidate
type: B, layer: 1, pos: 12

## Relational analysis of NS_A2_B1_A2_B1

### Relational analysis result of NS_A2_B1_A2_B1
Status: Status.UNKNOWN
Output dim: 0, lower bound: -20.9523051, upper bound: 20.9544784
time: 0.94 seconds

## Relational analysis of NS_A2_B1_A2_B2

### Relational analysis result of NS_A2_B1_A2_B2
Status: Status.UNKNOWN
Output dim: 0, lower bound: -20.9569857, upper bound: 20.9544784
time: 1.13 seconds

## BFS NS instance: NS_A2_B2_A1

### Backsubstitution after applying NS history:
0: -7.3449917, 19.9250336, -6.0870357, 17.0070343, -24.3520241, 26.0120659
1: -18.4785824, 30.3181419, -15.1643047, 26.2197037, -44.6982803, 45.4824333
2: -12.8041115, 27.5158348, -10.5776482, 23.7400360, -36.5441475, 38.0934830
3: -19.7412224, 33.6655312, -16.3570976, 28.9461727, -48.6873932, 50.0226250
4: -17.8084641, 33.9477806, -14.9398727, 29.2900238, -47.0984764, 48.8876534

Time for backsubstitution: 0.81 seconds

### NS candidates at layer 1
type: B, layer: 1, pos: 12
type: B, layer: 1, pos: 9
type: B, layer: 1, pos: 1
type: B, layer: 1, pos: 6
type: B, layer: 1, pos: 39
type: B, layer: 1, pos: 38
type: B, layer: 1, pos: 48
type: B, layer: 1, pos: 28
type: B, layer: 1, pos: 0
type: B, layer: 1, pos: 24
type: B, layer: 1, pos: 41
type: B, layer: 1, pos: 37
type: B, layer: 1, pos: 20
type: B, layer: 1, pos: 15
type: B, layer: 1, pos: 31
type: B, layer: 1, pos: 26
type: B, layer: 1, pos: 5
type: B, layer: 1, pos: 8
type: B, layer: 1, pos: 45

Time for candidate selection: 0.06 seconds

### Candidate
type: B, layer: 1, pos: 12

## Relational analysis of NS_A2_B2_A1_B1

### Relational analysis result of NS_A2_B2_A1_B1
Status: Status.UNKNOWN
Output dim: 0, lower bound: -20.9408792, upper bound: 20.9403090
time: 0.62 seconds

## Relational analysis of NS_A2_B2_A1_B2

### Relational analysis result of NS_A2_B2_A1_B2
Status: Status.UNKNOWN
Output dim: 0, lower bound: -20.9408792, upper bound: 20.9403090
time: 0.61 seconds

## BFS NS instance: NS_A2_B2_A2

### Backsubstitution after applying NS history:
0: -6.0558825, 16.8136654, -6.3495493, 17.6864700, -23.7423515, 23.1632137
1: -14.9474487, 25.9741230, -15.8588676, 27.2091198, -42.1565666, 41.8329926
2: -10.4984255, 23.4648914, -11.0539274, 24.6576653, -35.1560860, 34.5188179
3: -16.1687355, 28.6861324, -17.0815296, 30.0639439, -46.2326736, 45.7676620
4: -14.8261528, 28.8265705, -15.5757790, 30.4333534, -45.2595062, 44.4023476

Time for backsubstitution: 0.82 seconds

### NS candidates at layer 1
type: B, layer: 1, pos: 1
type: B, layer: 1, pos: 12
type: B, layer: 1, pos: 9
type: B, layer: 1, pos: 6
type: B, layer: 1, pos: 39
type: B, layer: 1, pos: 38
type: B, layer: 1, pos: 28
type: B, layer: 1, pos: 48
type: B, layer: 1, pos: 0
type: B, layer: 1, pos: 24
type: B, layer: 1, pos: 41
type: B, layer: 1, pos: 37
type: B, layer: 1, pos: 20
type: B, layer: 1, pos: 15
type: B, layer: 1, pos: 26
type: B, layer: 1, pos: 31
type: B, layer: 1, pos: 45
type: B, layer: 1, pos: 5
type: B, layer: 1, pos: 8

Time for candidate selection: 0.06 seconds

### Candidate
type: B, layer: 1, pos: 1

## Relational analysis of NS_A2_B2_A2_B1

### Relational analysis result of NS_A2_B2_A2_B1
Status: Status.UNKNOWN
Output dim: 0, lower bound: -20.9403090, upper bound: 20.9408792
time: 0.55 seconds

## Relational analysis of NS_A2_B2_A2_B2

### Relational analysis result of NS_A2_B2_A2_B2
Status: Status.UNKNOWN
Output dim: 0, lower bound: -20.9403090, upper bound: 20.9480390
time: 0.60 seconds

## Summary of splitting at layer (split count: 3)
- Time for NS candidates: 2.03 seconds
NS_A1_B1_A1_B1, status: Status.UNKNOWN, split count: 4, time: 2.03
Output dim: 0, lower bound: -20.9431946, upper bound: 20.9438155
NS_A1_B1_A1_B2, status: Status.UNKNOWN, split count: 4, time: 2.03
Output dim: 0, lower bound: -20.9431946, upper bound: 20.9438155
NS_A1_B1_A2_B1, status: Status.UNKNOWN, split count: 4, time: 2.03
Output dim: 0, lower bound: -20.9431946, upper bound: 20.9438155
NS_A1_B1_A2_B2, status: Status.UNKNOWN, split count: 4, time: 2.03
Output dim: 0, lower bound: -20.9431946, upper bound: 20.9438155
NS_A1_B2_A1_B1, status: Status.UNKNOWN, split count: 4, time: 2.03
Output dim: 0, lower bound: -20.9548571, upper bound: 20.9569857
NS_A1_B2_A1_B2, status: Status.UNKNOWN, split count: 4, time: 2.03
Output dim: 0, lower bound: -20.9548571, upper bound: 20.9569857
NS_A1_B2_A2_B1, status: Status.UNKNOWN, split count: 4, time: 2.03
Output dim: 0, lower bound: -20.9517087, upper bound: 20.9516156
NS_A1_B2_A2_B2, status: Status.UNKNOWN, split count: 4, time: 2.03
Output dim: 0, lower bound: -20.9517087, upper bound: 20.9516156
NS_A2_B1_A1_B1, status: Status.UNKNOWN, split count: 4, time: 2.03
Output dim: 0, lower bound: -20.9523051, upper bound: 20.9485296
NS_A2_B1_A1_B2, status: Status.UNKNOWN, split count: 4, time: 2.03
Output dim: 0, lower bound: -20.9523051, upper bound: 20.9485296
NS_A2_B1_A2_B1, status: Status.UNKNOWN, split count: 4, time: 2.03
Output dim: 0, lower bound: -20.9523051, upper bound: 20.9544784
NS_A2_B1_A2_B2, status: Status.UNKNOWN, split count: 4, time: 2.03
Output dim: 0, lower bound: -20.9569857, upper bound: 20.9544784
NS_A2_B2_A1_B1, status: Status.UNKNOWN, split count: 4, time: 2.03
Output dim: 0, lower bound: -20.9408792, upper bound: 20.9403090
NS_A2_B2_A1_B2, status: Status.UNKNOWN, split count: 4, time: 2.03
Output dim: 0, lower bound: -20.9408792, upper bound: 20.9403090
NS_A2_B2_A2_B1, status: Status.UNKNOWN, split count: 4, time: 2.03
Output dim: 0, lower bound: -20.9403090, upper bound: 20.9408792
NS_A2_B2_A2_B2, status: Status.UNKNOWN, split count: 4, time: 2.03
Output dim: 0, lower bound: -20.9403090, upper bound: 20.9480390

## BFS NS instance: NS_A1_B1_A1_B1

### Backsubstitution after applying NS history:
0: -5.6546221, 15.9332199, -6.0437589, 16.9015656, -22.5561848, 21.9769726
1: -14.1608744, 24.5610847, -15.2260876, 25.7967682, -39.9576416, 39.7871704
2: -9.8375568, 22.2305279, -10.5630150, 23.4757061, -33.3132629, 32.7935410
3: -15.2379446, 27.1142273, -16.3216972, 28.5732136, -43.8111572, 43.4359245
4: -13.7985134, 27.4670696, -14.7304068, 29.0496750, -42.8481903, 42.1974754

Time for backsubstitution: 0.81 seconds

### NS candidates at layer 1
type: A, layer: 1, pos: 9
type: A, layer: 1, pos: 1
type: A, layer: 1, pos: 39
type: A, layer: 1, pos: 6
type: A, layer: 1, pos: 48
type: A, layer: 1, pos: 28
type: A, layer: 1, pos: 38
type: A, layer: 1, pos: 41
type: A, layer: 1, pos: 0
type: A, layer: 1, pos: 8
type: A, layer: 1, pos: 24
type: A, layer: 1, pos: 37
type: A, layer: 1, pos: 15
type: A, layer: 1, pos: 31
type: A, layer: 1, pos: 26
type: A, layer: 1, pos: 45
type: A, layer: 1, pos: 20
type: A, layer: 1, pos: 5

Time for candidate selection: 0.05 seconds

### Candidate
type: A, layer: 1, pos: 9

## Relational analysis of NS_A1_B1_A1_B1_A1
Optimization infeasible because this subproblem isn't reachable.

### Candidate
type: A, layer: 1, pos: 1

## Relational analysis of NS_A1_B1_A1_B1_A1
Optimization infeasible because this subproblem isn't reachable.

### Candidate
type: A, layer: 1, pos: 39

## Relational analysis of NS_A1_B1_A1_B1_A1
Optimization infeasible because this subproblem isn't reachable.

### Candidate
type: A, layer: 1, pos: 6

## Relational analysis of NS_A1_B1_A1_B1_A1
Optimization infeasible because this subproblem isn't reachable.

### Candidate
type: A, layer: 1, pos: 48

## Relational analysis of NS_A1_B1_A1_B1_A1
Optimization infeasible because this subproblem isn't reachable.

### Candidate
type: A, layer: 1, pos: 28

## Relational analysis of NS_A1_B1_A1_B1_A1
Optimization infeasible because this subproblem isn't reachable.

### Candidate
type: A, layer: 1, pos: 38

## Relational analysis of NS_A1_B1_A1_B1_A1
Optimization infeasible because this subproblem isn't reachable.

### Candidate
type: A, layer: 1, pos: 41

## Relational analysis of NS_A1_B1_A1_B1_A1
Optimization infeasible because this subproblem isn't reachable.

### Candidate
type: A, layer: 1, pos: 0

## Relational analysis of NS_A1_B1_A1_B1_A1
Optimization infeasible because this subproblem isn't reachable.

### Candidate
type: A, layer: 1, pos: 8

## Relational analysis of NS_A1_B1_A1_B1_A1
Optimization infeasible because this subproblem isn't reachable.

### Candidate
type: A, layer: 1, pos: 24

## Relational analysis of NS_A1_B1_A1_B1_A1
Optimization infeasible because this subproblem isn't reachable.

### Candidate
type: A, layer: 1, pos: 37

## Relational analysis of NS_A1_B1_A1_B1_A1

### Relational analysis result of NS_A1_B1_A1_B1_A1
Status: Status.UNKNOWN
Output dim: 0, lower bound: -20.9455978, upper bound: 20.9467403
time: 0.56 seconds

## Relational analysis of NS_A1_B1_A1_B1_A2

### Relational analysis result of NS_A1_B1_A1_B1_A2
Status: Status.UNKNOWN
Output dim: 0, lower bound: -20.9488332, upper bound: 20.9523051
time: 0.56 seconds

## BFS NS instance: NS_A1_B1_A1_B2

### Backsubstitution after applying NS history:
0: -5.6546221, 15.9332199, -6.2151299, 17.2624722, -22.9170933, 22.1483498
1: -14.1608744, 24.5610847, -15.4910069, 26.4429283, -40.6038017, 40.0520897
2: -9.8375568, 22.2305279, -10.8176346, 24.0293427, -33.8668976, 33.0481644
3: -15.2379446, 27.1142273, -16.7221031, 29.2755241, -44.5134621, 43.8363304
4: -13.7985134, 27.4670696, -15.2812357, 29.6582718, -43.4567871, 42.7483025

Time for backsubstitution: 0.81 seconds

### NS candidates at layer 1
type: A, layer: 1, pos: 9
type: A, layer: 1, pos: 1
type: A, layer: 1, pos: 39
type: A, layer: 1, pos: 6
type: A, layer: 1, pos: 48
type: A, layer: 1, pos: 28
type: A, layer: 1, pos: 38
type: A, layer: 1, pos: 41
type: A, layer: 1, pos: 0
type: A, layer: 1, pos: 8
type: A, layer: 1, pos: 24
type: A, layer: 1, pos: 37
type: A, layer: 1, pos: 15
type: A, layer: 1, pos: 31
type: A, layer: 1, pos: 26
type: A, layer: 1, pos: 45
type: A, layer: 1, pos: 20
type: A, layer: 1, pos: 5

Time for candidate selection: 0.05 seconds

### Candidate
type: A, layer: 1, pos: 9

## Relational analysis of NS_A1_B1_A1_B2_A1
Optimization infeasible because this subproblem isn't reachable.

### Candidate
type: A, layer: 1, pos: 1

## Relational analysis of NS_A1_B1_A1_B2_A1
Optimization infeasible because this subproblem isn't reachable.

### Candidate
type: A, layer: 1, pos: 39

## Relational analysis of NS_A1_B1_A1_B2_A1
Optimization infeasible because this subproblem isn't reachable.

### Candidate
type: A, layer: 1, pos: 6

## Relational analysis of NS_A1_B1_A1_B2_A1
Optimization infeasible because this subproblem isn't reachable.

### Candidate
type: A, layer: 1, pos: 48

## Relational analysis of NS_A1_B1_A1_B2_A1
Optimization infeasible because this subproblem isn't reachable.

### Candidate
type: A, layer: 1, pos: 28

## Relational analysis of NS_A1_B1_A1_B2_A1
Optimization infeasible because this subproblem isn't reachable.

### Candidate
type: A, layer: 1, pos: 38

## Relational analysis of NS_A1_B1_A1_B2_A1
Optimization infeasible because this subproblem isn't reachable.

### Candidate
type: A, layer: 1, pos: 41

## Relational analysis of NS_A1_B1_A1_B2_A1
Optimization infeasible because this subproblem isn't reachable.

### Candidate
type: A, layer: 1, pos: 0

## Relational analysis of NS_A1_B1_A1_B2_A1
Optimization infeasible because this subproblem isn't reachable.

### Candidate
type: A, layer: 1, pos: 8

## Relational analysis of NS_A1_B1_A1_B2_A1
Optimization infeasible because this subproblem isn't reachable.

### Candidate
type: A, layer: 1, pos: 24

## Relational analysis of NS_A1_B1_A1_B2_A1
Optimization infeasible because this subproblem isn't reachable.

### Candidate
type: A, layer: 1, pos: 37

## Relational analysis of NS_A1_B1_A1_B2_A1

### Relational analysis result of NS_A1_B1_A1_B2_A1
Status: Status.UNKNOWN
Output dim: 0, lower bound: -20.9455978, upper bound: 20.9467403
time: 1.15 seconds

## Relational analysis of NS_A1_B1_A1_B2_A2

### Relational analysis result of NS_A1_B1_A1_B2_A2
Status: Status.UNKNOWN
Output dim: 0, lower bound: -20.9488332, upper bound: 20.9523051
time: 0.79 seconds

## BFS NS instance: NS_A1_B1_A2_B1

### Backsubstitution after applying NS history:
0: -5.9220600, 16.5755711, -6.0437589, 16.9015656, -22.8236256, 22.6193237
1: -14.6959372, 25.6085052, -15.2260876, 25.7967682, -40.4926987, 40.8345947
2: -10.2669477, 23.1718216, -10.5630150, 23.4757061, -33.7426529, 33.7348366
3: -15.9044065, 28.2489166, -16.3216972, 28.5732136, -44.4776192, 44.5706100
4: -14.5871439, 28.5687160, -14.7304068, 29.0496750, -43.6368179, 43.2991219

Time for backsubstitution: 0.81 seconds

### NS candidates at layer 1
type: A, layer: 1, pos: 1
type: A, layer: 1, pos: 39
type: A, layer: 1, pos: 9
type: A, layer: 1, pos: 6
type: A, layer: 1, pos: 28
type: A, layer: 1, pos: 48
type: A, layer: 1, pos: 38
type: A, layer: 1, pos: 0
type: A, layer: 1, pos: 24
type: A, layer: 1, pos: 41
type: A, layer: 1, pos: 37
type: A, layer: 1, pos: 8
type: A, layer: 1, pos: 20
type: A, layer: 1, pos: 26
type: A, layer: 1, pos: 15
type: A, layer: 1, pos: 5
type: A, layer: 1, pos: 45
type: A, layer: 1, pos: 31

Time for candidate selection: 0.06 seconds

### Candidate
type: A, layer: 1, pos: 1

## Relational analysis of NS_A1_B1_A2_B1_A1
Optimization infeasible because this subproblem isn't reachable.

### Candidate
type: A, layer: 1, pos: 39

## Relational analysis of NS_A1_B1_A2_B1_A1
Optimization infeasible because this subproblem isn't reachable.

### Candidate
type: A, layer: 1, pos: 9

## Relational analysis of NS_A1_B1_A2_B1_A1
Optimization infeasible because this subproblem isn't reachable.

### Candidate
type: A, layer: 1, pos: 6

## Relational analysis of NS_A1_B1_A2_B1_A1
Optimization infeasible because this subproblem isn't reachable.

### Candidate
type: A, layer: 1, pos: 28

## Relational analysis of NS_A1_B1_A2_B1_A1
Optimization infeasible because this subproblem isn't reachable.

### Candidate
type: A, layer: 1, pos: 48

## Relational analysis of NS_A1_B1_A2_B1_A1
Optimization infeasible because this subproblem isn't reachable.

### Candidate
type: A, layer: 1, pos: 38

## Relational analysis of NS_A1_B1_A2_B1_A1
Optimization infeasible because this subproblem isn't reachable.

### Candidate
type: A, layer: 1, pos: 0

## Relational analysis of NS_A1_B1_A2_B1_A1
Optimization infeasible because this subproblem isn't reachable.

### Candidate
type: A, layer: 1, pos: 24

## Relational analysis of NS_A1_B1_A2_B1_A1
Optimization infeasible because this subproblem isn't reachable.

### Candidate
type: A, layer: 1, pos: 41

## Relational analysis of NS_A1_B1_A2_B1_A1
Optimization infeasible because this subproblem isn't reachable.

### Candidate
type: A, layer: 1, pos: 37

## Relational analysis of NS_A1_B1_A2_B1_A1

### Relational analysis result of NS_A1_B1_A2_B1_A1
Status: Status.UNKNOWN
Output dim: 0, lower bound: -20.9431946, upper bound: 20.9438155
time: 0.60 seconds

## Relational analysis of NS_A1_B1_A2_B1_A2

### Relational analysis result of NS_A1_B1_A2_B1_A2
Status: Status.UNKNOWN
Output dim: 0, lower bound: -20.9423008, upper bound: 20.9429799
time: 0.58 seconds

## BFS NS instance: NS_A1_B1_A2_B2

### Backsubstitution after applying NS history:
0: -5.9220600, 16.5755711, -6.2151299, 17.2624722, -23.1845322, 22.7907009
1: -14.6959372, 25.6085052, -15.4910069, 26.4429283, -41.1388588, 41.0995102
2: -10.2669477, 23.1718216, -10.8176346, 24.0293427, -34.2962914, 33.9894562
3: -15.9044065, 28.2489166, -16.7221031, 29.2755241, -45.1799202, 44.9710159
4: -14.5871439, 28.5687160, -15.2812357, 29.6582718, -44.2454147, 43.8499489

Time for backsubstitution: 0.82 seconds

### NS candidates at layer 1
type: A, layer: 1, pos: 1
type: A, layer: 1, pos: 39
type: A, layer: 1, pos: 9
type: A, layer: 1, pos: 6
type: A, layer: 1, pos: 28
type: A, layer: 1, pos: 48
type: A, layer: 1, pos: 38
type: A, layer: 1, pos: 0
type: A, layer: 1, pos: 24
type: A, layer: 1, pos: 41
type: A, layer: 1, pos: 37
type: A, layer: 1, pos: 8
type: A, layer: 1, pos: 20
type: A, layer: 1, pos: 26
type: A, layer: 1, pos: 15
type: A, layer: 1, pos: 5
type: A, layer: 1, pos: 45
type: A, layer: 1, pos: 31

Time for candidate selection: 0.06 seconds

### Candidate
type: A, layer: 1, pos: 1

## Relational analysis of NS_A1_B1_A2_B2_A1
Optimization infeasible because this subproblem isn't reachable.

### Candidate
type: A, layer: 1, pos: 39

## Relational analysis of NS_A1_B1_A2_B2_A1
Optimization infeasible because this subproblem isn't reachable.

### Candidate
type: A, layer: 1, pos: 9

## Relational analysis of NS_A1_B1_A2_B2_A1

### Relational analysis result of NS_A1_B1_A2_B2_A1
Status: Status.UNKNOWN
Output dim: 0, lower bound: -20.9293848, upper bound: 20.9248678
time: 0.73 seconds

## Relational analysis of NS_A1_B1_A2_B2_A2

### Relational analysis result of NS_A1_B1_A2_B2_A2
Status: Status.UNKNOWN
Output dim: 0, lower bound: -20.9293848, upper bound: 20.9435253
time: 0.71 seconds

## BFS NS instance: NS_A1_B2_A1_B1

### Backsubstitution after applying NS history:
0: -5.9211702, 16.6253052, -5.8933477, 16.5183525, -22.4395218, 22.5186481
1: -14.8620787, 25.5718231, -14.6265459, 25.5251884, -40.3872643, 40.1983681
2: -10.3212500, 23.1647587, -10.2202435, 23.0960236, -33.4172707, 33.3850021
3: -15.9681101, 28.2507477, -15.8352146, 28.1530037, -44.1211128, 44.0859528
4: -14.4465303, 28.6296043, -14.5272236, 28.4796982, -42.9262276, 43.1568184

Time for backsubstitution: 0.82 seconds

### NS candidates at layer 1
type: A, layer: 1, pos: 1
type: A, layer: 1, pos: 9
type: A, layer: 1, pos: 39
type: A, layer: 1, pos: 6
type: A, layer: 1, pos: 28
type: A, layer: 1, pos: 48
type: A, layer: 1, pos: 38
type: A, layer: 1, pos: 41
type: A, layer: 1, pos: 0
type: A, layer: 1, pos: 8
type: A, layer: 1, pos: 24
type: A, layer: 1, pos: 20
type: A, layer: 1, pos: 31
type: A, layer: 1, pos: 37
type: A, layer: 1, pos: 15
type: A, layer: 1, pos: 45
type: A, layer: 1, pos: 26
type: A, layer: 1, pos: 5

Time for candidate selection: 0.06 seconds

### Candidate
type: A, layer: 1, pos: 1

### Candidate
type: A, layer: 1, pos: 9

## Relational analysis of NS_A1_B2_A1_B1_A1
Optimization infeasible because this subproblem isn't reachable.

### Candidate
type: A, layer: 1, pos: 39

## Relational analysis of NS_A1_B2_A1_B1_A1

### Relational analysis result of NS_A1_B2_A1_B1_A1
Status: Status.UNKNOWN
Output dim: 0, lower bound: -20.9542795, upper bound: 20.9567436
time: 0.98 seconds

## Relational analysis of NS_A1_B2_A1_B1_A2
Optimization infeasible because this subproblem isn't reachable.

### Candidate
type: A, layer: 1, pos: 6

## Relational analysis of NS_A1_B2_A1_B1_A1
Optimization infeasible because this subproblem isn't reachable.

### Candidate
type: A, layer: 1, pos: 28

## Relational analysis of NS_A1_B2_A1_B1_A1
Optimization infeasible because this subproblem isn't reachable.

### Candidate
type: A, layer: 1, pos: 48

## Relational analysis of NS_A1_B2_A1_B1_A1
Optimization infeasible because this subproblem isn't reachable.

### Candidate
type: A, layer: 1, pos: 38

## Relational analysis of NS_A1_B2_A1_B1_A1
Optimization infeasible because this subproblem isn't reachable.

### Candidate
type: A, layer: 1, pos: 41

## Relational analysis of NS_A1_B2_A1_B1_A1
Optimization infeasible because this subproblem isn't reachable.

### Candidate
type: A, layer: 1, pos: 0

## Relational analysis of NS_A1_B2_A1_B1_A1

### Relational analysis result of NS_A1_B2_A1_B1_A1
Status: Status.UNKNOWN
Output dim: 0, lower bound: -20.9540767, upper bound: 20.9563050
time: 0.64 seconds

## Relational analysis of NS_A1_B2_A1_B1_A2

### Relational analysis result of NS_A1_B2_A1_B1_A2
Status: Status.UNKNOWN
Output dim: 0, lower bound: -20.9537253, upper bound: 20.9553052
time: 0.59 seconds

## BFS NS instance: NS_A1_B2_A1_B2

### Backsubstitution after applying NS history:
0: -5.9211702, 16.6253052, -5.8073215, 16.1819324, -22.1030998, 22.4326229
1: -14.8620787, 25.5718231, -14.3221569, 25.0498352, -39.9119072, 39.8939819
2: -10.3212500, 23.1647587, -10.0556936, 22.6109219, -32.9321709, 33.2204514
3: -15.9681101, 28.2507477, -15.4944172, 27.6358719, -43.6039772, 43.7451591
4: -14.4465303, 28.6296043, -14.2062893, 27.7788143, -42.2253456, 42.8358879

Time for backsubstitution: 0.82 seconds

### NS candidates at layer 1
type: A, layer: 1, pos: 1
type: A, layer: 1, pos: 9
type: A, layer: 1, pos: 39
type: A, layer: 1, pos: 6
type: A, layer: 1, pos: 28
type: A, layer: 1, pos: 48
type: A, layer: 1, pos: 38
type: A, layer: 1, pos: 41
type: A, layer: 1, pos: 0
type: A, layer: 1, pos: 8
type: A, layer: 1, pos: 24
type: A, layer: 1, pos: 20
type: A, layer: 1, pos: 31
type: A, layer: 1, pos: 37
type: A, layer: 1, pos: 15
type: A, layer: 1, pos: 45
type: A, layer: 1, pos: 26
type: A, layer: 1, pos: 5

Time for candidate selection: 0.06 seconds

### Candidate
type: A, layer: 1, pos: 1

### Candidate
type: A, layer: 1, pos: 9

## Relational analysis of NS_A1_B2_A1_B2_A1
Optimization infeasible because this subproblem isn't reachable.

### Candidate
type: A, layer: 1, pos: 39

## Relational analysis of NS_A1_B2_A1_B2_A1

### Relational analysis result of NS_A1_B2_A1_B2_A1
Status: Status.UNKNOWN
Output dim: 0, lower bound: -20.9542795, upper bound: 20.9567436
time: 0.64 seconds

## Relational analysis of NS_A1_B2_A1_B2_A2
Optimization infeasible because this subproblem isn't reachable.

### Candidate
type: A, layer: 1, pos: 6

## Relational analysis of NS_A1_B2_A1_B2_A1
Optimization infeasible because this subproblem isn't reachable.

### Candidate
type: A, layer: 1, pos: 28

## Relational analysis of NS_A1_B2_A1_B2_A1
Optimization infeasible because this subproblem isn't reachable.

### Candidate
type: A, layer: 1, pos: 48

## Relational analysis of NS_A1_B2_A1_B2_A1
Optimization infeasible because this subproblem isn't reachable.

### Candidate
type: A, layer: 1, pos: 38

## Relational analysis of NS_A1_B2_A1_B2_A1
Optimization infeasible because this subproblem isn't reachable.

### Candidate
type: A, layer: 1, pos: 41

## Relational analysis of NS_A1_B2_A1_B2_A1
Optimization infeasible because this subproblem isn't reachable.

### Candidate
type: A, layer: 1, pos: 0

## Relational analysis of NS_A1_B2_A1_B2_A1

### Relational analysis result of NS_A1_B2_A1_B2_A1
Status: Status.UNKNOWN
Output dim: 0, lower bound: -20.9540767, upper bound: 20.9563050
time: 0.69 seconds

## Relational analysis of NS_A1_B2_A1_B2_A2

### Relational analysis result of NS_A1_B2_A1_B2_A2
Status: Status.UNKNOWN
Output dim: 0, lower bound: -20.9537253, upper bound: 20.9553052
time: 1.12 seconds

## BFS NS instance: NS_A1_B2_A2_B1

### Backsubstitution after applying NS history:
0: -6.2047129, 17.3081169, -6.0461669, 16.9050064, -23.1097183, 23.3542805
1: -15.4355631, 26.6807861, -15.0026617, 26.0916557, -41.5272179, 41.6834488
2: -10.7784548, 24.1633434, -10.4874372, 23.6187649, -34.3972206, 34.6507797
3: -16.6825981, 29.4580708, -16.2466373, 28.7926636, -45.4752617, 45.7047081
4: -15.2801237, 29.8028202, -14.9184227, 29.1219597, -44.4020691, 44.7212448

Time for backsubstitution: 0.83 seconds

### NS candidates at layer 1
type: A, layer: 1, pos: 1
type: A, layer: 1, pos: 9
type: A, layer: 1, pos: 39
type: A, layer: 1, pos: 6
type: A, layer: 1, pos: 28
type: A, layer: 1, pos: 38
type: A, layer: 1, pos: 48
type: A, layer: 1, pos: 0
type: A, layer: 1, pos: 24
type: A, layer: 1, pos: 41
type: A, layer: 1, pos: 37
type: A, layer: 1, pos: 20
type: A, layer: 1, pos: 8
type: A, layer: 1, pos: 31
type: A, layer: 1, pos: 15
type: A, layer: 1, pos: 26
type: A, layer: 1, pos: 45
type: A, layer: 1, pos: 5

Time for candidate selection: 0.06 seconds

### Candidate
type: A, layer: 1, pos: 1

### Candidate
type: A, layer: 1, pos: 9

## Relational analysis of NS_A1_B2_A2_B1_A1

### Relational analysis result of NS_A1_B2_A2_B1_A1
Status: Status.UNKNOWN
Output dim: 0, lower bound: -20.9449241, upper bound: 20.9423952
time: 0.64 seconds

## Relational analysis of NS_A1_B2_A2_B1_A2
Optimization infeasible because this subproblem isn't reachable.

### Candidate
type: A, layer: 1, pos: 39

## Relational analysis of NS_A1_B2_A2_B1_A1

### Relational analysis result of NS_A1_B2_A2_B1_A1
Status: Status.UNKNOWN
Output dim: 0, lower bound: -20.9504369, upper bound: 20.9506183
time: 0.55 seconds

## Relational analysis of NS_A1_B2_A2_B1_A2

### Relational analysis result of NS_A1_B2_A2_B1_A2
Status: Status.UNKNOWN
Output dim: 0, lower bound: -20.9510443, upper bound: 20.9512858
time: 0.57 seconds

## BFS NS instance: NS_A1_B2_A2_B2

### Backsubstitution after applying NS history:
0: -6.2047129, 17.3081169, -6.0053267, 16.6742802, -22.8789940, 23.3134422
1: -15.4355631, 26.6807861, -14.8130960, 25.7692680, -41.2048302, 41.4938812
2: -10.7784548, 24.1633434, -10.4053373, 23.2785320, -34.0569878, 34.5686798
3: -16.6825981, 29.4580708, -16.0300694, 28.4563179, -45.1389160, 45.4881363
4: -15.2801237, 29.8028202, -14.7064686, 28.5940151, -43.8741379, 44.5092850

Time for backsubstitution: 0.84 seconds

### NS candidates at layer 1
type: A, layer: 1, pos: 1
type: A, layer: 1, pos: 9
type: A, layer: 1, pos: 39
type: A, layer: 1, pos: 6
type: A, layer: 1, pos: 28
type: A, layer: 1, pos: 38
type: A, layer: 1, pos: 48
type: A, layer: 1, pos: 0
type: A, layer: 1, pos: 24
type: A, layer: 1, pos: 41
type: A, layer: 1, pos: 37
type: A, layer: 1, pos: 20
type: A, layer: 1, pos: 8
type: A, layer: 1, pos: 31
type: A, layer: 1, pos: 15
type: A, layer: 1, pos: 26
type: A, layer: 1, pos: 45
type: A, layer: 1, pos: 5

Time for candidate selection: 0.06 seconds

### Candidate
type: A, layer: 1, pos: 1

### Candidate
type: A, layer: 1, pos: 9

## Relational analysis of NS_A1_B2_A2_B2_A1

### Relational analysis result of NS_A1_B2_A2_B2_A1
Status: Status.UNKNOWN
Output dim: 0, lower bound: -20.9449241, upper bound: 20.9423952
time: 0.58 seconds

## Relational analysis of NS_A1_B2_A2_B2_A2

### Relational analysis result of NS_A1_B2_A2_B2_A2
Status: Status.UNKNOWN
Output dim: 0, lower bound: -20.9512850, upper bound: 20.9511787
time: 0.72 seconds

## BFS NS instance: NS_A2_B1_A1_B1

### Backsubstitution after applying NS history:
0: -7.1646075, 19.4706726, -5.6865683, 16.0214005, -23.1860027, 25.1572399
1: -18.0290127, 29.6563625, -14.2480450, 24.6873779, -42.7163925, 43.9044075
2: -12.4878607, 26.8986130, -9.8974237, 22.3465786, -34.8344383, 36.7960358
3: -19.2521210, 32.9131813, -15.3260517, 27.2576427, -46.5097656, 48.2392349
4: -17.3505459, 33.1918945, -13.8730974, 27.6133747, -44.9639168, 47.0649872

Time for backsubstitution: 0.83 seconds

### NS candidates at layer 1
type: A, layer: 1, pos: 47
type: A, layer: 1, pos: 9
type: A, layer: 1, pos: 6
type: A, layer: 1, pos: 38
type: A, layer: 1, pos: 48
type: A, layer: 1, pos: 0
type: A, layer: 1, pos: 28
type: A, layer: 1, pos: 37
type: A, layer: 1, pos: 39
type: A, layer: 1, pos: 41
type: A, layer: 1, pos: 8
type: A, layer: 1, pos: 15
type: A, layer: 1, pos: 24
type: A, layer: 1, pos: 31
type: A, layer: 1, pos: 20
type: A, layer: 1, pos: 45
type: A, layer: 1, pos: 26
type: A, layer: 1, pos: 5

Time for candidate selection: 0.06 seconds

### Candidate
type: A, layer: 1, pos: 47

## Relational analysis of NS_A2_B1_A1_B1_A1

### Relational analysis result of NS_A2_B1_A1_B1_A1
Status: Status.UNKNOWN
Output dim: 0, lower bound: -20.9523051, upper bound: 20.9485296
time: 0.56 seconds

## Relational analysis of NS_A2_B1_A1_B1_A2

### Relational analysis result of NS_A2_B1_A1_B1_A2
Status: Status.UNKNOWN
Output dim: 0, lower bound: -20.9523051, upper bound: 20.9485296
time: 0.97 seconds

## BFS NS instance: NS_A2_B1_A1_B2

### Backsubstitution after applying NS history:
0: -7.1646075, 19.4706726, -5.7321081, 16.0435200, -23.2081261, 25.2027798
1: -18.0290127, 29.6563625, -14.2964621, 24.7329769, -42.7619896, 43.9528236
2: -12.4878607, 26.8986130, -9.9682322, 22.3510227, -34.8388824, 36.8668442
3: -19.2521210, 32.9131813, -15.3430071, 27.3167400, -46.5688553, 48.2561874
4: -17.3505459, 33.1918945, -13.8603640, 27.5282879, -44.8788300, 47.0522537

Time for backsubstitution: 0.84 seconds

### NS candidates at layer 1
type: A, layer: 1, pos: 47
type: A, layer: 1, pos: 9
type: A, layer: 1, pos: 6
type: A, layer: 1, pos: 38
type: A, layer: 1, pos: 48
type: A, layer: 1, pos: 0
type: A, layer: 1, pos: 28
type: A, layer: 1, pos: 37
type: A, layer: 1, pos: 39
type: A, layer: 1, pos: 41
type: A, layer: 1, pos: 8
type: A, layer: 1, pos: 15
type: A, layer: 1, pos: 24
type: A, layer: 1, pos: 31
type: A, layer: 1, pos: 20
type: A, layer: 1, pos: 45
type: A, layer: 1, pos: 26
type: A, layer: 1, pos: 5

Time for candidate selection: 0.06 seconds

### Candidate
type: A, layer: 1, pos: 47

## Relational analysis of NS_A2_B1_A1_B2_A1

### Relational analysis result of NS_A2_B1_A1_B2_A1
Status: Status.UNKNOWN
Output dim: 0, lower bound: -20.9523051, upper bound: 20.9485296
time: 0.59 seconds

## Relational analysis of NS_A2_B1_A1_B2_A2

### Relational analysis result of NS_A2_B1_A1_B2_A2
Status: Status.UNKNOWN
Output dim: 0, lower bound: -20.9523051, upper bound: 20.9485296
time: 0.55 seconds

## BFS NS instance: NS_A2_B1_A2_B1

### Backsubstitution after applying NS history:
0: -5.8769097, 16.3671246, -5.9402876, 16.6799774, -22.5568848, 22.3074112
1: -14.5033007, 25.3214855, -14.9174976, 25.6486301, -40.1519318, 40.2389755
2: -10.1820154, 22.8613892, -10.3585854, 23.2349720, -33.4169807, 33.2199745
3: -15.6849270, 27.9441319, -16.0219460, 28.3389072, -44.0238304, 43.9660759
4: -14.3726711, 28.0883980, -14.4889812, 28.7199249, -43.0925865, 42.5773773

Time for backsubstitution: 0.84 seconds

### NS candidates at layer 1
type: A, layer: 1, pos: 39
type: A, layer: 1, pos: 47
type: A, layer: 1, pos: 9
type: A, layer: 1, pos: 6
type: A, layer: 1, pos: 48
type: A, layer: 1, pos: 28
type: A, layer: 1, pos: 38
type: A, layer: 1, pos: 0
type: A, layer: 1, pos: 24
type: A, layer: 1, pos: 37
type: A, layer: 1, pos: 41
type: A, layer: 1, pos: 20
type: A, layer: 1, pos: 31
type: A, layer: 1, pos: 8
type: A, layer: 1, pos: 15
type: A, layer: 1, pos: 26
type: A, layer: 1, pos: 5
type: A, layer: 1, pos: 45

Time for candidate selection: 0.06 seconds

### Candidate
type: A, layer: 1, pos: 39

## Relational analysis of NS_A2_B1_A2_B1_A1
Optimization infeasible because this subproblem isn't reachable.

### Candidate
type: A, layer: 1, pos: 47

## Relational analysis of NS_A2_B1_A2_B1_A1

### Relational analysis result of NS_A2_B1_A2_B1_A1
Status: Status.UNKNOWN
Output dim: 0, lower bound: -20.9523051, upper bound: 20.9544784
time: 0.60 seconds

## Relational analysis of NS_A2_B1_A2_B1_A2

### Relational analysis result of NS_A2_B1_A2_B1_A2
Status: Status.UNKNOWN
Output dim: 0, lower bound: -20.9523051, upper bound: 20.9544784
time: 0.64 seconds

## BFS NS instance: NS_A2_B1_A2_B2

### Backsubstitution after applying NS history:
0: -5.8769097, 16.3671246, -6.0440979, 16.8430519, -22.7199612, 22.4112225
1: -14.5033007, 25.3214855, -15.1288452, 25.9006710, -40.4039726, 40.4503326
2: -10.1820154, 22.8613892, -10.5438957, 23.4227562, -33.6047707, 33.4052811
3: -15.6849270, 27.9441319, -16.1987095, 28.6353874, -44.3203011, 44.1428413
4: -14.3726711, 28.0883980, -14.6018887, 28.8638325, -43.2365036, 42.6902847

Time for backsubstitution: 0.84 seconds

### NS candidates at layer 1
type: A, layer: 1, pos: 39
type: A, layer: 1, pos: 47
type: A, layer: 1, pos: 9
type: A, layer: 1, pos: 6
type: A, layer: 1, pos: 48
type: A, layer: 1, pos: 28
type: A, layer: 1, pos: 38
type: A, layer: 1, pos: 0
type: A, layer: 1, pos: 24
type: A, layer: 1, pos: 37
type: A, layer: 1, pos: 41
type: A, layer: 1, pos: 20
type: A, layer: 1, pos: 31
type: A, layer: 1, pos: 8
type: A, layer: 1, pos: 15
type: A, layer: 1, pos: 26
type: A, layer: 1, pos: 5
type: A, layer: 1, pos: 45

Time for candidate selection: 0.06 seconds

### Candidate
type: A, layer: 1, pos: 39

## Relational analysis of NS_A2_B1_A2_B2_A1
Optimization infeasible because this subproblem isn't reachable.

### Candidate
type: A, layer: 1, pos: 47

## Relational analysis of NS_A2_B1_A2_B2_A1

### Relational analysis result of NS_A2_B1_A2_B2_A1
Status: Status.UNKNOWN
Output dim: 0, lower bound: -20.9569857, upper bound: 20.9544784
time: 0.67 seconds

## Relational analysis of NS_A2_B1_A2_B2_A2

### Relational analysis result of NS_A2_B1_A2_B2_A2
Status: Status.UNKNOWN
Output dim: 0, lower bound: -20.9569857, upper bound: 20.9544784
time: 1.12 seconds

## BFS NS instance: NS_A2_B2_A1_B1

### Backsubstitution after applying NS history:
0: -7.3449917, 19.9250336, -5.9208970, 16.5725956, -23.9175835, 25.8459282
1: -18.4785824, 30.3181419, -14.6930676, 25.6041012, -44.0826797, 45.0112000
2: -12.8041115, 27.5158348, -10.2649250, 23.1677380, -35.9718475, 37.7807617
3: -19.7412224, 33.6655312, -15.9012699, 28.2439499, -47.9851685, 49.5668030
4: -17.8084641, 33.9477806, -14.5842028, 28.5637188, -46.3721809, 48.5319786

Time for backsubstitution: 0.86 seconds

### NS candidates at layer 1
type: A, layer: 1, pos: 47
type: A, layer: 1, pos: 9
type: A, layer: 1, pos: 6
type: A, layer: 1, pos: 38
type: A, layer: 1, pos: 0
type: A, layer: 1, pos: 28
type: A, layer: 1, pos: 48
type: A, layer: 1, pos: 37
type: A, layer: 1, pos: 39
type: A, layer: 1, pos: 41
type: A, layer: 1, pos: 8
type: A, layer: 1, pos: 15
type: A, layer: 1, pos: 24
type: A, layer: 1, pos: 20
type: A, layer: 1, pos: 31
type: A, layer: 1, pos: 45
type: A, layer: 1, pos: 26
type: A, layer: 1, pos: 5

Time for candidate selection: 0.06 seconds

### Candidate
type: A, layer: 1, pos: 47

## Relational analysis of NS_A2_B2_A1_B1_A1

### Relational analysis result of NS_A2_B2_A1_B1_A1
Status: Status.UNKNOWN
Output dim: 0, lower bound: -20.9408792, upper bound: 20.9403090
time: 0.69 seconds

## Relational analysis of NS_A2_B2_A1_B1_A2

### Relational analysis result of NS_A2_B2_A1_B1_A2
Status: Status.UNKNOWN
Output dim: 0, lower bound: -20.9408792, upper bound: 20.9403090
time: 0.63 seconds

## BFS NS instance: NS_A2_B2_A1_B2

### Backsubstitution after applying NS history:
0: -7.3449917, 19.9250336, -5.9177461, 16.4657726, -23.8107567, 25.8427792
1: -18.4785824, 30.3181419, -14.5941172, 25.4797840, -43.9583664, 44.9122581
2: -12.8041115, 27.5158348, -10.2439365, 23.0011864, -35.8052940, 37.7597733
3: -19.7412224, 33.6655312, -15.7867498, 28.1186886, -47.8599052, 49.4522820
4: -17.8084641, 33.9477806, -14.4681406, 28.2543964, -46.0628548, 48.4159164

Time for backsubstitution: 0.86 seconds

### NS candidates at layer 1
type: A, layer: 1, pos: 47
type: A, layer: 1, pos: 9
type: A, layer: 1, pos: 6
type: A, layer: 1, pos: 38
type: A, layer: 1, pos: 0
type: A, layer: 1, pos: 28
type: A, layer: 1, pos: 48
type: A, layer: 1, pos: 37
type: A, layer: 1, pos: 39
type: A, layer: 1, pos: 41
type: A, layer: 1, pos: 8
type: A, layer: 1, pos: 15
type: A, layer: 1, pos: 24
type: A, layer: 1, pos: 20
type: A, layer: 1, pos: 31
type: A, layer: 1, pos: 45
type: A, layer: 1, pos: 26
type: A, layer: 1, pos: 5

Time for candidate selection: 0.06 seconds

### Candidate
type: A, layer: 1, pos: 47

## Relational analysis of NS_A2_B2_A1_B2_A1

### Relational analysis result of NS_A2_B2_A1_B2_A1
Status: Status.UNKNOWN
Output dim: 0, lower bound: -20.9408792, upper bound: 20.9403090
time: 0.71 seconds

## Relational analysis of NS_A2_B2_A1_B2_A2

### Relational analysis result of NS_A2_B2_A1_B2_A2
Status: Status.UNKNOWN
Output dim: 0, lower bound: -20.9408792, upper bound: 20.9403090
time: 0.58 seconds

## BFS NS instance: NS_A2_B2_A2_B1

### Backsubstitution after applying NS history:
0: -6.0558825, 16.8136654, -6.2250795, 17.2942638, -23.3501472, 23.0387459
1: -14.9474487, 25.9741230, -15.5413942, 26.4833488, -41.4307976, 41.5155029
2: -10.4984255, 23.4648914, -10.8425531, 24.0679264, -34.5663528, 34.3074455
3: -16.1687355, 28.6861324, -16.7520714, 29.3229733, -45.4917068, 45.4382019
4: -14.8261528, 28.8265705, -15.2782488, 29.7155247, -44.5416794, 44.1048164

Time for backsubstitution: 0.87 seconds

### NS candidates at layer 1
type: A, layer: 1, pos: 47
type: A, layer: 1, pos: 39
type: A, layer: 1, pos: 9
type: A, layer: 1, pos: 6
type: A, layer: 1, pos: 48
type: A, layer: 1, pos: 28
type: A, layer: 1, pos: 0
type: A, layer: 1, pos: 38
type: A, layer: 1, pos: 24
type: A, layer: 1, pos: 37
type: A, layer: 1, pos: 41
type: A, layer: 1, pos: 8
type: A, layer: 1, pos: 31
type: A, layer: 1, pos: 20
type: A, layer: 1, pos: 15
type: A, layer: 1, pos: 26
type: A, layer: 1, pos: 5
type: A, layer: 1, pos: 45

Time for candidate selection: 0.06 seconds

### Candidate
type: A, layer: 1, pos: 47

## Relational analysis of NS_A2_B2_A2_B1_A1

### Relational analysis result of NS_A2_B2_A2_B1_A1
Status: Status.UNKNOWN
Output dim: 0, lower bound: -20.9403090, upper bound: 20.9408792
time: 0.59 seconds

## Relational analysis of NS_A2_B2_A2_B1_A2

### Relational analysis result of NS_A2_B2_A2_B1_A2
Status: Status.UNKNOWN
Output dim: 0, lower bound: -20.9403090, upper bound: 20.9408792
time: 0.62 seconds

## BFS NS instance: NS_A2_B2_A2_B2

### Backsubstitution after applying NS history:
0: -6.0558825, 16.8136654, -6.0877700, 17.0216694, -23.0775528, 22.9014359
1: -14.9474487, 25.9741230, -15.1663723, 26.2364292, -41.1838760, 41.1404839
2: -10.4984255, 23.4648914, -10.5828571, 23.7615471, -34.2599716, 34.0477486
3: -16.1687355, 28.6861324, -16.3651867, 28.9663982, -45.1351318, 45.0513191
4: -14.8261528, 28.8265705, -14.9519472, 29.3174820, -44.1436310, 43.7785149

Time for backsubstitution: 0.87 seconds

### NS candidates at layer 1
type: A, layer: 1, pos: 47
type: A, layer: 1, pos: 39
type: A, layer: 1, pos: 9
type: A, layer: 1, pos: 6
type: A, layer: 1, pos: 48
type: A, layer: 1, pos: 28
type: A, layer: 1, pos: 0
type: A, layer: 1, pos: 38
type: A, layer: 1, pos: 24
type: A, layer: 1, pos: 37
type: A, layer: 1, pos: 41
type: A, layer: 1, pos: 8
type: A, layer: 1, pos: 31
type: A, layer: 1, pos: 20
type: A, layer: 1, pos: 15
type: A, layer: 1, pos: 26
type: A, layer: 1, pos: 5
type: A, layer: 1, pos: 45

Time for candidate selection: 0.06 seconds

### Candidate
type: A, layer: 1, pos: 47

## Relational analysis of NS_A2_B2_A2_B2_A1

### Relational analysis result of NS_A2_B2_A2_B2_A1
Status: Status.UNKNOWN
Output dim: 0, lower bound: -20.9361618, upper bound: 20.9480285
time: 0.65 seconds

## Relational analysis of NS_A2_B2_A2_B2_A2

### Relational analysis result of NS_A2_B2_A2_B2_A2
Status: Status.UNKNOWN
Output dim: 0, lower bound: -20.9361618, upper bound: 20.9480285
time: 0.61 seconds

## Summary of splitting at layer (split count: 4)
- Time for NS candidates: 2.20 seconds
NS_A1_B1_A1_B1_A1, status: Status.UNKNOWN, split count: 5, time: 2.20
Output dim: 0, lower bound: -20.9455978, upper bound: 20.9467403
NS_A1_B1_A1_B1_A2, status: Status.UNKNOWN, split count: 5, time: 2.20
Output dim: 0, lower bound: -20.9488332, upper bound: 20.9523051
NS_A1_B1_A1_B2_A1, status: Status.UNKNOWN, split count: 5, time: 2.20
Output dim: 0, lower bound: -20.9455978, upper bound: 20.9467403
NS_A1_B1_A1_B2_A2, status: Status.UNKNOWN, split count: 5, time: 2.20
Output dim: 0, lower bound: -20.9488332, upper bound: 20.9523051
NS_A1_B1_A2_B1_A1, status: Status.UNKNOWN, split count: 5, time: 2.20
Output dim: 0, lower bound: -20.9431946, upper bound: 20.9438155
NS_A1_B1_A2_B1_A2, status: Status.UNKNOWN, split count: 5, time: 2.20
Output dim: 0, lower bound: -20.9423008, upper bound: 20.9429799
NS_A1_B1_A2_B2_A1, status: Status.UNKNOWN, split count: 5, time: 2.20
Output dim: 0, lower bound: -20.9293848, upper bound: 20.9248678
NS_A1_B1_A2_B2_A2, status: Status.UNKNOWN, split count: 5, time: 2.20
Output dim: 0, lower bound: -20.9293848, upper bound: 20.9435253
NS_A1_B2_A1_B1_A1, status: Status.UNKNOWN, split count: 5, time: 2.20
Output dim: 0, lower bound: -20.9540767, upper bound: 20.9563050
NS_A1_B2_A1_B1_A2, status: Status.UNKNOWN, split count: 5, time: 2.20
Output dim: 0, lower bound: -20.9537253, upper bound: 20.9553052
NS_A1_B2_A1_B2_A1, status: Status.UNKNOWN, split count: 5, time: 2.20
Output dim: 0, lower bound: -20.9540767, upper bound: 20.9563050
NS_A1_B2_A1_B2_A2, status: Status.UNKNOWN, split count: 5, time: 2.20
Output dim: 0, lower bound: -20.9537253, upper bound: 20.9553052
NS_A1_B2_A2_B1_A1, status: Status.UNKNOWN, split count: 5, time: 2.20
Output dim: 0, lower bound: -20.9504369, upper bound: 20.9506183
NS_A1_B2_A2_B1_A2, status: Status.UNKNOWN, split count: 5, time: 2.20
Output dim: 0, lower bound: -20.9510443, upper bound: 20.9512858
NS_A1_B2_A2_B2_A1, status: Status.UNKNOWN, split count: 5, time: 2.20
Output dim: 0, lower bound: -20.9449241, upper bound: 20.9423952
NS_A1_B2_A2_B2_A2, status: Status.UNKNOWN, split count: 5, time: 2.20
Output dim: 0, lower bound: -20.9512850, upper bound: 20.9511787
NS_A2_B1_A1_B1_A1, status: Status.UNKNOWN, split count: 5, time: 2.20
Output dim: 0, lower bound: -20.9523051, upper bound: 20.9485296
NS_A2_B1_A1_B1_A2, status: Status.UNKNOWN, split count: 5, time: 2.20
Output dim: 0, lower bound: -20.9523051, upper bound: 20.9485296
NS_A2_B1_A1_B2_A1, status: Status.UNKNOWN, split count: 5, time: 2.20
Output dim: 0, lower bound: -20.9523051, upper bound: 20.9485296
NS_A2_B1_A1_B2_A2, status: Status.UNKNOWN, split count: 5, time: 2.20
Output dim: 0, lower bound: -20.9523051, upper bound: 20.9485296
NS_A2_B1_A2_B1_A1, status: Status.UNKNOWN, split count: 5, time: 2.20
Output dim: 0, lower bound: -20.9523051, upper bound: 20.9544784
NS_A2_B1_A2_B1_A2, status: Status.UNKNOWN, split count: 5, time: 2.20
Output dim: 0, lower bound: -20.9523051, upper bound: 20.9544784
NS_A2_B1_A2_B2_A1, status: Status.UNKNOWN, split count: 5, time: 2.20
Output dim: 0, lower bound: -20.9569857, upper bound: 20.9544784
NS_A2_B1_A2_B2_A2, status: Status.UNKNOWN, split count: 5, time: 2.20
Output dim: 0, lower bound: -20.9569857, upper bound: 20.9544784
NS_A2_B2_A1_B1_A1, status: Status.UNKNOWN, split count: 5, time: 2.20
Output dim: 0, lower bound: -20.9408792, upper bound: 20.9403090
NS_A2_B2_A1_B1_A2, status: Status.UNKNOWN, split count: 5, time: 2.20
Output dim: 0, lower bound: -20.9408792, upper bound: 20.9403090
NS_A2_B2_A1_B2_A1, status: Status.UNKNOWN, split count: 5, time: 2.20
Output dim: 0, lower bound: -20.9408792, upper bound: 20.9403090
NS_A2_B2_A1_B2_A2, status: Status.UNKNOWN, split count: 5, time: 2.20
Output dim: 0, lower bound: -20.9408792, upper bound: 20.9403090
NS_A2_B2_A2_B1_A1, status: Status.UNKNOWN, split count: 5, time: 2.20
Output dim: 0, lower bound: -20.9403090, upper bound: 20.9408792
NS_A2_B2_A2_B1_A2, status: Status.UNKNOWN, split count: 5, time: 2.20
Output dim: 0, lower bound: -20.9403090, upper bound: 20.9408792
NS_A2_B2_A2_B2_A1, status: Status.UNKNOWN, split count: 5, time: 2.20
Output dim: 0, lower bound: -20.9361618, upper bound: 20.9480285
NS_A2_B2_A2_B2_A2, status: Status.UNKNOWN, split count: 5, time: 2.20
Output dim: 0, lower bound: -20.9361618, upper bound: 20.9480285

## BFS NS instance: NS_A1_B1_A1_B1_A1

### Backsubstitution after applying NS history:
0: -5.5220685, 15.5871372, -6.0358195, 16.8812180, -22.4032841, 21.6229572
1: -13.8160563, 24.0525360, -15.2054520, 25.7668648, -39.5829201, 39.2579765
2: -9.6000080, 21.7625408, -10.5488605, 23.4481163, -33.0481262, 32.3114014
3: -14.8792744, 26.5470657, -16.3002186, 28.5397568, -43.4190292, 42.8472824
4: -13.4775257, 26.8844700, -14.7112255, 29.0153561, -42.4928780, 41.5956955

Time for backsubstitution: 0.85 seconds

### NS candidates at layer 1
type: B, layer: 1, pos: 12
type: B, layer: 1, pos: 9
type: B, layer: 1, pos: 48
type: B, layer: 1, pos: 6
type: B, layer: 1, pos: 28
type: B, layer: 1, pos: 38
type: B, layer: 1, pos: 39
type: B, layer: 1, pos: 8
type: B, layer: 1, pos: 41
type: B, layer: 1, pos: 0
type: B, layer: 1, pos: 15
type: B, layer: 1, pos: 37
type: B, layer: 1, pos: 24
type: B, layer: 1, pos: 20
type: B, layer: 1, pos: 31
type: B, layer: 1, pos: 5
type: B, layer: 1, pos: 26
type: B, layer: 1, pos: 45

Time for candidate selection: 0.06 seconds

### Candidate
type: B, layer: 1, pos: 12

### Candidate
type: B, layer: 1, pos: 9

### Candidate
type: B, layer: 1, pos: 48

### Candidate
type: B, layer: 1, pos: 6

### Candidate
type: B, layer: 1, pos: 28

### Candidate
type: B, layer: 1, pos: 38

### Candidate
type: B, layer: 1, pos: 39

### Candidate
type: B, layer: 1, pos: 8

### Candidate
type: B, layer: 1, pos: 41

### Candidate
type: B, layer: 1, pos: 0

### Candidate
type: B, layer: 1, pos: 15

## Relational analysis of NS_A1_B1_A1_B1_A1_B1
Optimization infeasible because this subproblem isn't reachable.

### Candidate
type: B, layer: 1, pos: 37

## Relational analysis of NS_A1_B1_A1_B1_A1_B1
Optimization infeasible because this subproblem isn't reachable.

### Candidate
type: B, layer: 1, pos: 24

### Candidate
type: B, layer: 1, pos: 20

## Relational analysis of NS_A1_B1_A1_B1_A1_B1
Optimization infeasible because this subproblem isn't reachable.

### Candidate
type: B, layer: 1, pos: 31

## Relational analysis of NS_A1_B1_A1_B1_A1_B1
Optimization infeasible because this subproblem isn't reachable.

### Candidate
type: B, layer: 1, pos: 5

## Relational analysis of NS_A1_B1_A1_B1_A1_B1
Optimization infeasible because this subproblem isn't reachable.

### Candidate
type: B, layer: 1, pos: 26

## Relational analysis of NS_A1_B1_A1_B1_A1_B1
Optimization infeasible because this subproblem isn't reachable.

### Candidate
type: B, layer: 1, pos: 45

## Relational analysis of NS_A1_B1_A1_B1_A1_B1
Optimization infeasible because this subproblem isn't reachable.

## BFS NS instance: NS_A1_B1_A1_B1_A2

### Backsubstitution after applying NS history:
0: -5.7029700, 16.0100746, -5.9806652, 16.7347889, -22.4377556, 21.9907398
1: -14.2405682, 24.6658897, -15.0588140, 25.5555000, -39.7960663, 39.7247047
2: -9.9136543, 22.3478088, -10.4493036, 23.2533264, -33.1669807, 32.7971115
3: -15.3522701, 27.2383041, -16.1488533, 28.3005276, -43.6527977, 43.3871498
4: -13.9597645, 27.5948181, -14.5793543, 28.7716293, -42.7313919, 42.1741714

Time for backsubstitution: 0.86 seconds

### NS candidates at layer 1
type: B, layer: 1, pos: 12
type: B, layer: 1, pos: 9
type: B, layer: 1, pos: 48
type: B, layer: 1, pos: 6
type: B, layer: 1, pos: 28
type: B, layer: 1, pos: 38
type: B, layer: 1, pos: 39
type: B, layer: 1, pos: 41
type: B, layer: 1, pos: 8
type: B, layer: 1, pos: 0
type: B, layer: 1, pos: 15
type: B, layer: 1, pos: 24
type: B, layer: 1, pos: 37
type: B, layer: 1, pos: 20
type: B, layer: 1, pos: 31
type: B, layer: 1, pos: 45
type: B, layer: 1, pos: 5
type: B, layer: 1, pos: 26

Time for candidate selection: 0.06 seconds

### Candidate
type: B, layer: 1, pos: 12

### Candidate
type: B, layer: 1, pos: 9

### Candidate
type: B, layer: 1, pos: 48

### Candidate
type: B, layer: 1, pos: 6

### Candidate
type: B, layer: 1, pos: 28

### Candidate
type: B, layer: 1, pos: 38

### Candidate
type: B, layer: 1, pos: 39

### Candidate
type: B, layer: 1, pos: 41

### Candidate
type: B, layer: 1, pos: 8

### Candidate
type: B, layer: 1, pos: 0

### Candidate
type: B, layer: 1, pos: 15

## Relational analysis of NS_A1_B1_A1_B1_A2_B1
Optimization infeasible because this subproblem isn't reachable.

### Candidate
type: B, layer: 1, pos: 24

### Candidate
type: B, layer: 1, pos: 37

## Relational analysis of NS_A1_B1_A1_B1_A2_B1
Optimization infeasible because this subproblem isn't reachable.

### Candidate
type: B, layer: 1, pos: 20

## Relational analysis of NS_A1_B1_A1_B1_A2_B1

### Relational analysis result of NS_A1_B1_A1_B1_A2_B1
Status: Status.UNKNOWN
Output dim: 0, lower bound: -20.9552630, upper bound: 20.9548567
time: 0.64 seconds

## Relational analysis of NS_A1_B1_A1_B1_A2_B2

### Relational analysis result of NS_A1_B1_A1_B1_A2_B2
Status: Status.UNKNOWN
Output dim: 0, lower bound: -20.9649475, upper bound: 20.9640737
time: 0.62 seconds

## BFS NS instance: NS_A1_B1_A1_B2_A1

### Backsubstitution after applying NS history:
0: -5.5220685, 15.5871372, -6.2049351, 17.2361221, -22.7581863, 21.7920723
1: -13.8160563, 24.0525360, -15.4642382, 26.4046478, -40.2207031, 39.5167656
2: -9.6000080, 21.7625408, -10.7993412, 23.9937649, -33.5937729, 32.5618820
3: -14.8792744, 26.5470657, -16.6942902, 29.2324123, -44.1116867, 43.2413559
4: -13.4775257, 26.8844700, -15.2564478, 29.6138382, -43.0913620, 42.1409187

Time for backsubstitution: 0.84 seconds

### NS candidates at layer 1
type: B, layer: 1, pos: 12
type: B, layer: 1, pos: 9
type: B, layer: 1, pos: 6
type: B, layer: 1, pos: 48
type: B, layer: 1, pos: 39
type: B, layer: 1, pos: 28
type: B, layer: 1, pos: 38
type: B, layer: 1, pos: 0
type: B, layer: 1, pos: 24
type: B, layer: 1, pos: 41
type: B, layer: 1, pos: 37
type: B, layer: 1, pos: 8
type: B, layer: 1, pos: 31
type: B, layer: 1, pos: 20
type: B, layer: 1, pos: 26
type: B, layer: 1, pos: 15
type: B, layer: 1, pos: 5
type: B, layer: 1, pos: 45

Time for candidate selection: 0.06 seconds

### Candidate
type: B, layer: 1, pos: 12

### Candidate
type: B, layer: 1, pos: 9

### Candidate
type: B, layer: 1, pos: 6

### Candidate
type: B, layer: 1, pos: 48

### Candidate
type: B, layer: 1, pos: 39

### Candidate
type: B, layer: 1, pos: 28

### Candidate
type: B, layer: 1, pos: 38

### Candidate
type: B, layer: 1, pos: 0

### Candidate
type: B, layer: 1, pos: 24

### Candidate
type: B, layer: 1, pos: 41

### Candidate
type: B, layer: 1, pos: 37

## Relational analysis of NS_A1_B1_A1_B2_A1_B1

### Relational analysis result of NS_A1_B1_A1_B2_A1_B1
Status: Status.UNKNOWN
Output dim: 0, lower bound: -20.9224596, upper bound: 20.9341830
time: 0.64 seconds

## Relational analysis of NS_A1_B1_A1_B2_A1_B2

### Relational analysis result of NS_A1_B1_A1_B2_A1_B2
Status: Status.UNKNOWN
Output dim: 0, lower bound: -20.9224596, upper bound: 20.9467403
time: 1.08 seconds

## BFS NS instance: NS_A1_B1_A1_B2_A2

### Backsubstitution after applying NS history:
0: -5.7029700, 16.0100746, -6.1720357, 17.1500359, -22.8530025, 22.1821098
1: -14.2405682, 24.6658897, -15.3768473, 26.2810173, -40.5215836, 40.0427361
2: -9.9136543, 22.3478088, -10.7406797, 23.8799534, -33.7936020, 33.0884895
3: -15.3522701, 27.2383041, -16.6040802, 29.0933895, -44.4456596, 43.8423729
4: -13.9597645, 27.5948181, -15.1802216, 29.4725266, -43.4322853, 42.7750359

Time for backsubstitution: 0.86 seconds

### NS candidates at layer 1
type: B, layer: 1, pos: 12
type: B, layer: 1, pos: 9
type: B, layer: 1, pos: 6
type: B, layer: 1, pos: 48
type: B, layer: 1, pos: 39
type: B, layer: 1, pos: 28
type: B, layer: 1, pos: 38
type: B, layer: 1, pos: 0
type: B, layer: 1, pos: 24
type: B, layer: 1, pos: 41
type: B, layer: 1, pos: 8
type: B, layer: 1, pos: 37
type: B, layer: 1, pos: 31
type: B, layer: 1, pos: 20
type: B, layer: 1, pos: 26
type: B, layer: 1, pos: 15
type: B, layer: 1, pos: 45
type: B, layer: 1, pos: 5

Time for candidate selection: 0.06 seconds

### Candidate
type: B, layer: 1, pos: 12

### Candidate
type: B, layer: 1, pos: 9

### Candidate
type: B, layer: 1, pos: 6

### Candidate
type: B, layer: 1, pos: 48

### Candidate
type: B, layer: 1, pos: 39

### Candidate
type: B, layer: 1, pos: 28

### Candidate
type: B, layer: 1, pos: 38

### Candidate
type: B, layer: 1, pos: 0

### Candidate
type: B, layer: 1, pos: 24

### Candidate
type: B, layer: 1, pos: 41

### Candidate
type: B, layer: 1, pos: 8

### Candidate
type: B, layer: 1, pos: 37

## Relational analysis of NS_A1_B1_A1_B2_A2_B1

### Relational analysis result of NS_A1_B1_A1_B2_A2_B1
Status: Status.UNKNOWN
Output dim: 0, lower bound: -20.9270553, upper bound: 20.9411884
time: 0.53 seconds

## Relational analysis of NS_A1_B1_A1_B2_A2_B2

### Relational analysis result of NS_A1_B1_A1_B2_A2_B2
Status: Status.UNKNOWN
Output dim: 0, lower bound: -20.9270553, upper bound: 20.9523051
time: 0.56 seconds

## BFS NS instance: NS_A1_B1_A2_B1_A1

### Backsubstitution after applying NS history:
0: -5.7593737, 16.1530533, -6.0358195, 16.8812180, -22.6405907, 22.1888733
1: -14.2705688, 24.9937592, -15.2054520, 25.7668648, -40.0374336, 40.1992111
2: -9.9754887, 22.6008472, -10.5488605, 23.4481163, -33.4236069, 33.1497078
3: -15.4613800, 27.5570297, -16.3002186, 28.5397568, -44.0011368, 43.8572464
4: -14.1926994, 27.8550491, -14.7112255, 29.0153561, -43.2080536, 42.5662766

Time for backsubstitution: 0.86 seconds

### NS candidates at layer 1
type: B, layer: 1, pos: 12
type: B, layer: 1, pos: 9
type: B, layer: 1, pos: 48
type: B, layer: 1, pos: 6
type: B, layer: 1, pos: 28
type: B, layer: 1, pos: 38
type: B, layer: 1, pos: 39
type: B, layer: 1, pos: 8
type: B, layer: 1, pos: 41
type: B, layer: 1, pos: 0
type: B, layer: 1, pos: 15
type: B, layer: 1, pos: 37
type: B, layer: 1, pos: 24
type: B, layer: 1, pos: 20
type: B, layer: 1, pos: 31
type: B, layer: 1, pos: 5
type: B, layer: 1, pos: 26
type: B, layer: 1, pos: 45

Time for candidate selection: 0.06 seconds

### Candidate
type: B, layer: 1, pos: 12

### Candidate
type: B, layer: 1, pos: 9

### Candidate
type: B, layer: 1, pos: 48

### Candidate
type: B, layer: 1, pos: 6

### Candidate
type: B, layer: 1, pos: 28

### Candidate
type: B, layer: 1, pos: 38

### Candidate
type: B, layer: 1, pos: 39

### Candidate
type: B, layer: 1, pos: 8

## Relational analysis of NS_A1_B1_A2_B1_A1_B1

### Relational analysis result of NS_A1_B1_A2_B1_A1_B1
Status: Status.UNKNOWN
Output dim: 0, lower bound: -20.9572021, upper bound: 20.9527657
time: 0.61 seconds

## Relational analysis of NS_A1_B1_A2_B1_A1_B2

### Relational analysis result of NS_A1_B1_A2_B1_A1_B2
Status: Status.UNKNOWN
Output dim: 0, lower bound: -20.9572021, upper bound: 20.9564671
time: 0.63 seconds

## BFS NS instance: NS_A1_B1_A2_B1_A2

### Backsubstitution after applying NS history:
0: -5.9965081, 16.7290058, -5.9806652, 16.7347889, -22.7312946, 22.7096691
1: -14.8499393, 25.8211803, -15.0588140, 25.5555000, -40.4054413, 40.8799934
2: -10.3930006, 23.3898659, -10.4493036, 23.2533264, -33.6463280, 33.8391685
3: -16.0963974, 28.4984932, -16.1488533, 28.3005276, -44.3969269, 44.6473389
4: -14.8098030, 28.8273373, -14.5793543, 28.7716293, -43.5814323, 43.4066925

Time for backsubstitution: 0.86 seconds

### NS candidates at layer 1
type: B, layer: 1, pos: 12
type: B, layer: 1, pos: 9
type: B, layer: 1, pos: 48
type: B, layer: 1, pos: 6
type: B, layer: 1, pos: 28
type: B, layer: 1, pos: 38
type: B, layer: 1, pos: 39
type: B, layer: 1, pos: 41
type: B, layer: 1, pos: 8
type: B, layer: 1, pos: 0
type: B, layer: 1, pos: 15
type: B, layer: 1, pos: 24
type: B, layer: 1, pos: 37
type: B, layer: 1, pos: 20
type: B, layer: 1, pos: 31
type: B, layer: 1, pos: 45
type: B, layer: 1, pos: 5
type: B, layer: 1, pos: 26

Time for candidate selection: 0.06 seconds

### Candidate
type: B, layer: 1, pos: 12

### Candidate
type: B, layer: 1, pos: 9

### Candidate
type: B, layer: 1, pos: 48

### Candidate
type: B, layer: 1, pos: 6

### Candidate
type: B, layer: 1, pos: 28

### Candidate
type: B, layer: 1, pos: 38

### Candidate
type: B, layer: 1, pos: 39

### Candidate
type: B, layer: 1, pos: 41

### Candidate
type: B, layer: 1, pos: 8

## Relational analysis of NS_A1_B1_A2_B1_A2_B1

### Relational analysis result of NS_A1_B1_A2_B1_A2_B1
Status: Status.UNKNOWN
Output dim: 0, lower bound: -20.9565773, upper bound: 20.9520676
time: 0.58 seconds

## Relational analysis of NS_A1_B1_A2_B1_A2_B2

### Relational analysis result of NS_A1_B1_A2_B1_A2_B2
Status: Status.UNKNOWN
Output dim: 0, lower bound: -20.9584735, upper bound: 20.9549741
time: 0.53 seconds

## BFS NS instance: NS_A1_B1_A2_B2_A1

### Backsubstitution after applying NS history:
0: -5.8384013, 16.3657513, -6.2151299, 17.2624722, -23.1008720, 22.5808811
1: -14.4844494, 25.3005962, -15.4910069, 26.4429283, -40.9273720, 40.7915993
2: -10.1135731, 22.8840389, -10.8176346, 24.0293427, -34.1429138, 33.7016754
3: -15.6806517, 27.9001026, -16.7221031, 29.2755241, -44.9561653, 44.6222076
4: -14.3752546, 28.2174873, -15.2812357, 29.6582718, -44.0335197, 43.4987221

Time for backsubstitution: 0.86 seconds

### NS candidates at layer 1
type: B, layer: 1, pos: 12
type: B, layer: 1, pos: 9
type: B, layer: 1, pos: 6
type: B, layer: 1, pos: 48
type: B, layer: 1, pos: 39
type: B, layer: 1, pos: 28
type: B, layer: 1, pos: 38
type: B, layer: 1, pos: 0
type: B, layer: 1, pos: 24
type: B, layer: 1, pos: 37
type: B, layer: 1, pos: 41
type: B, layer: 1, pos: 8
type: B, layer: 1, pos: 31
type: B, layer: 1, pos: 20
type: B, layer: 1, pos: 26
type: B, layer: 1, pos: 15
type: B, layer: 1, pos: 5
type: B, layer: 1, pos: 45

Time for candidate selection: 0.06 seconds

### Candidate
type: B, layer: 1, pos: 12

### Candidate
type: B, layer: 1, pos: 9

## Relational analysis of NS_A1_B1_A2_B2_A1_B1

### Relational analysis result of NS_A1_B1_A2_B2_A1_B1
Status: Status.UNKNOWN
Output dim: 0, lower bound: -20.9121909, upper bound: 20.9155805
time: 0.66 seconds

## Relational analysis of NS_A1_B1_A2_B2_A1_B2

### Relational analysis result of NS_A1_B1_A2_B2_A1_B2
Status: Status.UNKNOWN
Output dim: 0, lower bound: -20.9121909, upper bound: 20.9248678
time: 0.59 seconds

## BFS NS instance: NS_A1_B1_A2_B2_A2

### Backsubstitution after applying NS history:
0: -5.6924324, 15.9221439, -6.1602750, 17.1180325, -22.8104649, 22.0824184
1: -14.1625080, 24.5419464, -15.3579807, 26.2268734, -40.3893814, 39.8999214
2: -9.8790131, 22.2478924, -10.7235584, 23.8321934, -33.7112007, 32.9714470
3: -15.3027058, 27.1016102, -16.5769138, 29.0352364, -44.3379440, 43.6785240
4: -13.9716702, 27.4440098, -15.1459169, 29.4178200, -43.3894806, 42.5899277

Time for backsubstitution: 0.87 seconds

### NS candidates at layer 1
type: B, layer: 1, pos: 12
type: B, layer: 1, pos: 6
type: B, layer: 1, pos: 9
type: B, layer: 1, pos: 48
type: B, layer: 1, pos: 39
type: B, layer: 1, pos: 28
type: B, layer: 1, pos: 38
type: B, layer: 1, pos: 0
type: B, layer: 1, pos: 24
type: B, layer: 1, pos: 41
type: B, layer: 1, pos: 37
type: B, layer: 1, pos: 8
type: B, layer: 1, pos: 31
type: B, layer: 1, pos: 20
type: B, layer: 1, pos: 26
type: B, layer: 1, pos: 15
type: B, layer: 1, pos: 45
type: B, layer: 1, pos: 5

Time for candidate selection: 0.06 seconds

### Candidate
type: B, layer: 1, pos: 12

### Candidate
type: B, layer: 1, pos: 6

## Relational analysis of NS_A1_B1_A2_B2_A2_B1
Optimization infeasible because this subproblem isn't reachable.

### Candidate
type: B, layer: 1, pos: 9

## Relational analysis of NS_A1_B1_A2_B2_A2_B1

### Relational analysis result of NS_A1_B1_A2_B2_A2_B1
Status: Status.UNKNOWN
Output dim: 0, lower bound: -20.9138254, upper bound: 20.9189541
time: 0.58 seconds

## Relational analysis of NS_A1_B1_A2_B2_A2_B2

### Relational analysis result of NS_A1_B1_A2_B2_A2_B2
Status: Status.UNKNOWN
Output dim: 0, lower bound: -20.9138254, upper bound: 20.9435253
time: 0.60 seconds

## BFS NS instance: NS_A1_B2_A1_B1_A1

### Backsubstitution after applying NS history:
0: -5.2931418, 14.9397831, -5.8340435, 16.3648338, -21.6579762, 20.7738247
1: -13.1073370, 23.1132984, -14.4664974, 25.3032742, -38.4106102, 37.5797958
2: -9.1489487, 20.9186592, -10.1110115, 22.8896236, -32.0385628, 31.0296707
3: -14.2131996, 25.4802723, -15.6706247, 27.9005375, -42.1137314, 41.1508980
4: -13.0011435, 25.7873745, -14.3850040, 28.2209511, -41.2220917, 40.1723709

Time for backsubstitution: 0.87 seconds

### NS candidates at layer 1
type: B, layer: 1, pos: 47
type: B, layer: 1, pos: 39
type: B, layer: 1, pos: 9
type: B, layer: 1, pos: 6
type: B, layer: 1, pos: 28
type: B, layer: 1, pos: 48
type: B, layer: 1, pos: 38
type: B, layer: 1, pos: 24
type: B, layer: 1, pos: 0
type: B, layer: 1, pos: 41
type: B, layer: 1, pos: 37
type: B, layer: 1, pos: 20
type: B, layer: 1, pos: 8
type: B, layer: 1, pos: 31
type: B, layer: 1, pos: 26
type: B, layer: 1, pos: 5
type: B, layer: 1, pos: 15
type: B, layer: 1, pos: 45

Time for candidate selection: 0.06 seconds

### Candidate
type: B, layer: 1, pos: 47

## Relational analysis of NS_A1_B2_A1_B1_A1_B1

### Relational analysis result of NS_A1_B2_A1_B1_A1_B1
Status: Status.UNKNOWN
Output dim: 0, lower bound: -20.9607899, upper bound: 20.9623820
time: 0.64 seconds

## Relational analysis of NS_A1_B2_A1_B1_A1_B2

### Relational analysis result of NS_A1_B2_A1_B1_A1_B2
Status: Status.UNKNOWN
Output dim: 0, lower bound: -20.9607899, upper bound: 20.9623820
time: 0.64 seconds

## BFS NS instance: NS_A1_B2_A1_B1_A2

### Backsubstitution after applying NS history:
0: -5.5402689, 15.6422672, -5.8933477, 16.5183525, -22.0586185, 21.5356140
1: -13.8112717, 24.1865158, -14.6265459, 25.5251884, -39.3364563, 38.8130608
2: -9.6186361, 21.8514957, -10.2202435, 23.0960236, -32.7146606, 32.0717392
3: -14.9117498, 26.6818504, -15.8352146, 28.1530037, -43.0647545, 42.5170670
4: -13.5568142, 26.9629269, -14.5272236, 28.4796982, -42.0365105, 41.4901390

Time for backsubstitution: 0.88 seconds

### NS candidates at layer 1
type: B, layer: 1, pos: 47
type: B, layer: 1, pos: 9
type: B, layer: 1, pos: 39
type: B, layer: 1, pos: 6
type: B, layer: 1, pos: 28
type: B, layer: 1, pos: 48
type: B, layer: 1, pos: 38
type: B, layer: 1, pos: 0
type: B, layer: 1, pos: 24
type: B, layer: 1, pos: 41
type: B, layer: 1, pos: 37
type: B, layer: 1, pos: 20
type: B, layer: 1, pos: 8
type: B, layer: 1, pos: 31
type: B, layer: 1, pos: 26
type: B, layer: 1, pos: 45
type: B, layer: 1, pos: 5
type: B, layer: 1, pos: 15

Time for candidate selection: 0.06 seconds

### Candidate
type: B, layer: 1, pos: 47

## Relational analysis of NS_A1_B2_A1_B1_A2_B1

### Relational analysis result of NS_A1_B2_A1_B1_A2_B1
Status: Status.UNKNOWN
Output dim: 0, lower bound: -20.9604393, upper bound: 20.9612852
time: 0.59 seconds

## Relational analysis of NS_A1_B2_A1_B1_A2_B2

### Relational analysis result of NS_A1_B2_A1_B1_A2_B2
Status: Status.UNKNOWN
Output dim: 0, lower bound: -20.9604393, upper bound: 20.9612852
time: 0.63 seconds

## BFS NS instance: NS_A1_B2_A1_B2_A1

### Backsubstitution after applying NS history:
0: -5.2931418, 14.9397831, -5.7423043, 16.0126495, -21.3057861, 20.6820831
1: -13.1073370, 23.1132984, -14.1435585, 24.8067341, -37.9140663, 37.2568588
2: -9.1489487, 20.9186592, -9.9336176, 22.3848934, -31.5338402, 30.8522739
3: -14.2131996, 25.4802723, -15.3117504, 27.3586330, -41.5718269, 40.7920227
4: -13.0011435, 25.7873745, -14.0508633, 27.4949036, -40.4960365, 39.8382339

Time for backsubstitution: 0.87 seconds

### NS candidates at layer 1
type: B, layer: 1, pos: 39
type: B, layer: 1, pos: 47
type: B, layer: 1, pos: 9
type: B, layer: 1, pos: 6
type: B, layer: 1, pos: 48
type: B, layer: 1, pos: 38
type: B, layer: 1, pos: 28
type: B, layer: 1, pos: 0
type: B, layer: 1, pos: 24
type: B, layer: 1, pos: 41
type: B, layer: 1, pos: 37
type: B, layer: 1, pos: 20
type: B, layer: 1, pos: 31
type: B, layer: 1, pos: 8
type: B, layer: 1, pos: 15
type: B, layer: 1, pos: 26
type: B, layer: 1, pos: 5
type: B, layer: 1, pos: 45

Time for candidate selection: 0.06 seconds

### Candidate
type: B, layer: 1, pos: 39

### Candidate
type: B, layer: 1, pos: 47

## Relational analysis of NS_A1_B2_A1_B2_A1_B1

### Relational analysis result of NS_A1_B2_A1_B2_A1_B1
Status: Status.UNKNOWN
Output dim: 0, lower bound: -20.9540767, upper bound: 20.9563050
time: 0.59 seconds

## Relational analysis of NS_A1_B2_A1_B2_A1_B2
Optimization infeasible because this subproblem isn't reachable.

### Candidate
type: B, layer: 1, pos: 9

### Candidate
type: B, layer: 1, pos: 6

### Candidate
type: B, layer: 1, pos: 48

### Candidate
type: B, layer: 1, pos: 38

### Candidate
type: B, layer: 1, pos: 28

### Candidate
type: B, layer: 1, pos: 0

## Relational analysis of NS_A1_B2_A1_B2_A1_B1

### Relational analysis result of NS_A1_B2_A1_B2_A1_B1
Status: Status.UNKNOWN
Output dim: 0, lower bound: -20.9534221, upper bound: 20.9549159
time: 0.68 seconds

## Relational analysis of NS_A1_B2_A1_B2_A1_B2

### Relational analysis result of NS_A1_B2_A1_B2_A1_B2
Status: Status.UNKNOWN
Output dim: 0, lower bound: -20.9534221, upper bound: 20.9553052
time: 0.63 seconds

## BFS NS instance: NS_A1_B2_A1_B2_A2

### Backsubstitution after applying NS history:
0: -5.5402689, 15.6422672, -5.8073215, 16.1819324, -21.7221966, 21.4495888
1: -13.8112717, 24.1865158, -14.3221569, 25.0498352, -38.8610992, 38.5086746
2: -9.6186361, 21.8514957, -10.0556936, 22.6109219, -32.2295570, 31.9071884
3: -14.9117498, 26.6818504, -15.4944172, 27.6358719, -42.5476189, 42.1762695
4: -13.5568142, 26.9629269, -14.2062893, 27.7788143, -41.3356247, 41.1692085

Time for backsubstitution: 0.87 seconds

### NS candidates at layer 1
type: B, layer: 1, pos: 39
type: B, layer: 1, pos: 47
type: B, layer: 1, pos: 9
type: B, layer: 1, pos: 6
type: B, layer: 1, pos: 48
type: B, layer: 1, pos: 28
type: B, layer: 1, pos: 38
type: B, layer: 1, pos: 0
type: B, layer: 1, pos: 24
type: B, layer: 1, pos: 37
type: B, layer: 1, pos: 41
type: B, layer: 1, pos: 20
type: B, layer: 1, pos: 31
type: B, layer: 1, pos: 8
type: B, layer: 1, pos: 15
type: B, layer: 1, pos: 26
type: B, layer: 1, pos: 5
type: B, layer: 1, pos: 45

Time for candidate selection: 0.06 seconds

### Candidate
type: B, layer: 1, pos: 39

### Candidate
type: B, layer: 1, pos: 47

## Relational analysis of NS_A1_B2_A1_B2_A2_B1

### Relational analysis result of NS_A1_B2_A1_B2_A2_B1
Status: Status.UNKNOWN
Output dim: 0, lower bound: -20.9537253, upper bound: 20.9553052
time: 0.63 seconds

## Relational analysis of NS_A1_B2_A1_B2_A2_B2

### Relational analysis result of NS_A1_B2_A1_B2_A2_B2
Status: Status.UNKNOWN
Output dim: 0, lower bound: -20.9537253, upper bound: 20.9553052
time: 0.67 seconds

## BFS NS instance: NS_A1_B2_A2_B1_A1

### Backsubstitution after applying NS history:
0: -5.9519348, 16.6478176, -6.0137639, 16.8206902, -22.7726250, 22.6615810
1: -14.8047256, 25.6933746, -14.9219131, 25.9656925, -40.7704163, 40.6152878
2: -10.3367424, 23.2725792, -10.4307308, 23.5049591, -33.8417015, 33.7033043
3: -16.0035057, 28.3608398, -16.1594925, 28.6524944, -44.6559982, 44.5203209
4: -14.6475315, 28.7021637, -14.8368073, 28.9813328, -43.6288528, 43.5389595

Time for backsubstitution: 0.88 seconds

### NS candidates at layer 1
type: B, layer: 1, pos: 47
type: B, layer: 1, pos: 9
type: B, layer: 1, pos: 6
type: B, layer: 1, pos: 39
type: B, layer: 1, pos: 28
type: B, layer: 1, pos: 38
type: B, layer: 1, pos: 48
type: B, layer: 1, pos: 0
type: B, layer: 1, pos: 24
type: B, layer: 1, pos: 41
type: B, layer: 1, pos: 37
type: B, layer: 1, pos: 20
type: B, layer: 1, pos: 8
type: B, layer: 1, pos: 31
type: B, layer: 1, pos: 26
type: B, layer: 1, pos: 15
type: B, layer: 1, pos: 45
type: B, layer: 1, pos: 5

Time for candidate selection: 0.06 seconds

### Candidate
type: B, layer: 1, pos: 47

## Relational analysis of NS_A1_B2_A2_B1_A1_B1

### Relational analysis result of NS_A1_B2_A2_B1_A1_B1
Status: Status.UNKNOWN
Output dim: 0, lower bound: -20.9613011, upper bound: 20.9616617
time: 0.61 seconds

## Relational analysis of NS_A1_B2_A2_B1_A1_B2

### Relational analysis result of NS_A1_B2_A2_B1_A1_B2
Status: Status.UNKNOWN
Output dim: 0, lower bound: -20.9613011, upper bound: 20.9616617
time: 0.57 seconds

## BFS NS instance: NS_A1_B2_A2_B1_A2

### Backsubstitution after applying NS history:
0: -7.0105648, 19.5966530, -5.9317231, 16.6145077, -23.6250725, 25.5283718
1: -17.6516457, 30.0256901, -14.7020378, 25.6633167, -43.3149643, 44.7277298
2: -12.2167578, 27.2167797, -10.2826347, 23.2272873, -35.4440460, 37.4994087
3: -18.8770370, 33.1330910, -15.9350843, 28.3121395, -47.1891670, 49.0681725
4: -17.0309677, 33.6710854, -14.6449690, 28.6347656, -45.6657295, 48.3160553

Time for backsubstitution: 0.88 seconds

### NS candidates at layer 1
type: B, layer: 1, pos: 47
type: B, layer: 1, pos: 9
type: B, layer: 1, pos: 6
type: B, layer: 1, pos: 39
type: B, layer: 1, pos: 28
type: B, layer: 1, pos: 38
type: B, layer: 1, pos: 48
type: B, layer: 1, pos: 0
type: B, layer: 1, pos: 24
type: B, layer: 1, pos: 41
type: B, layer: 1, pos: 37
type: B, layer: 1, pos: 20
type: B, layer: 1, pos: 8
type: B, layer: 1, pos: 31
type: B, layer: 1, pos: 26
type: B, layer: 1, pos: 15
type: B, layer: 1, pos: 45
type: B, layer: 1, pos: 5

Time for candidate selection: 0.06 seconds

### Candidate
type: B, layer: 1, pos: 47

## Relational analysis of NS_A1_B2_A2_B1_A2_B1

### Relational analysis result of NS_A1_B2_A2_B1_A2_B1
Status: Status.UNKNOWN
Output dim: 0, lower bound: -20.9615105, upper bound: 20.9615105
time: 0.68 seconds

## Relational analysis of NS_A1_B2_A2_B1_A2_B2

### Relational analysis result of NS_A1_B2_A2_B1_A2_B2
Status: Status.UNKNOWN
Output dim: 0, lower bound: -20.9615105, upper bound: 20.9615105
time: 0.63 seconds

## BFS NS instance: NS_A1_B2_A2_B2_A1

### Backsubstitution after applying NS history:
0: -6.1207747, 17.0969448, -6.0053267, 16.6742802, -22.7950535, 23.1022720
1: -15.2238340, 26.3712082, -14.8130960, 25.7692680, -40.9931030, 41.1842995
2: -10.6242371, 23.8737068, -10.4053373, 23.2785320, -33.9027672, 34.2790451
3: -16.4577141, 29.1074352, -16.0300694, 28.4563179, -44.9140320, 45.1375008
4: -15.0668459, 29.4489193, -14.7064686, 28.5940151, -43.6608582, 44.1553802

Time for backsubstitution: 0.89 seconds

### NS candidates at layer 1
type: B, layer: 1, pos: 47
type: B, layer: 1, pos: 39
type: B, layer: 1, pos: 9
type: B, layer: 1, pos: 6
type: B, layer: 1, pos: 48
type: B, layer: 1, pos: 28
type: B, layer: 1, pos: 0
type: B, layer: 1, pos: 38
type: B, layer: 1, pos: 24
type: B, layer: 1, pos: 37
type: B, layer: 1, pos: 41
type: B, layer: 1, pos: 8
type: B, layer: 1, pos: 31
type: B, layer: 1, pos: 20
type: B, layer: 1, pos: 15
type: B, layer: 1, pos: 26
type: B, layer: 1, pos: 5
type: B, layer: 1, pos: 45

Time for candidate selection: 0.06 seconds

### Candidate
type: B, layer: 1, pos: 47

## Relational analysis of NS_A1_B2_A2_B2_A1_B1

### Relational analysis result of NS_A1_B2_A2_B2_A1_B1
Status: Status.UNKNOWN
Output dim: 0, lower bound: -20.9293848, upper bound: 20.9423952
time: 0.61 seconds

## Relational analysis of NS_A1_B2_A2_B2_A1_B2

### Relational analysis result of NS_A1_B2_A2_B2_A1_B2
Status: Status.UNKNOWN
Output dim: 0, lower bound: -20.9293848, upper bound: 20.9423952
time: 0.75 seconds

## BFS NS instance: NS_A1_B2_A2_B2_A2

### Backsubstitution after applying NS history:
0: -5.9430337, 16.5669880, -5.9448814, 16.5188313, -22.4618645, 22.5118675
1: -14.8138142, 25.4860477, -14.6643438, 25.5377121, -40.3515244, 40.1503906
2: -10.3298597, 23.1185207, -10.3001804, 23.0668182, -33.3966789, 33.4186935
3: -15.9877548, 28.1648884, -15.8693676, 28.1976433, -44.1853981, 44.0342560
4: -14.5869484, 28.5262318, -14.5568266, 28.3357487, -42.9226990, 43.0830574

Time for backsubstitution: 0.88 seconds

### NS candidates at layer 1
type: B, layer: 1, pos: 47
type: B, layer: 1, pos: 39
type: B, layer: 1, pos: 6
type: B, layer: 1, pos: 9
type: B, layer: 1, pos: 48
type: B, layer: 1, pos: 0
type: B, layer: 1, pos: 28
type: B, layer: 1, pos: 38
type: B, layer: 1, pos: 24
type: B, layer: 1, pos: 37
type: B, layer: 1, pos: 41
type: B, layer: 1, pos: 31
type: B, layer: 1, pos: 8
type: B, layer: 1, pos: 20
type: B, layer: 1, pos: 15
type: B, layer: 1, pos: 26
type: B, layer: 1, pos: 5
type: B, layer: 1, pos: 45

Time for candidate selection: 0.06 seconds

### Candidate
type: B, layer: 1, pos: 47

## Relational analysis of NS_A1_B2_A2_B2_A2_B1

### Relational analysis result of NS_A1_B2_A2_B2_A2_B1
Status: Status.UNKNOWN
Output dim: 0, lower bound: -20.9293848, upper bound: 20.9511787
time: 0.69 seconds

## Relational analysis of NS_A1_B2_A2_B2_A2_B2

### Relational analysis result of NS_A1_B2_A2_B2_A2_B2
Status: Status.UNKNOWN
Output dim: 0, lower bound: -20.9512850, upper bound: 20.9511787
time: 0.91 seconds

## BFS NS instance: NS_A2_B1_A1_B1_A1

### Backsubstitution after applying NS history:
0: -7.2958264, 19.9399872, -5.6865683, 16.0214005, -23.3172207, 25.6265564
1: -18.5720406, 30.2115574, -14.2480450, 24.6873779, -43.2594185, 44.4596024
2: -12.7887821, 27.4622536, -9.8974237, 22.3465786, -35.1353607, 37.3596764
3: -19.6904030, 33.5532112, -15.3260517, 27.2576427, -46.9480438, 48.8792648
4: -17.5103760, 33.9847832, -13.8730974, 27.6133747, -45.1237488, 47.8578758

Time for backsubstitution: 0.89 seconds

### NS candidates at layer 1
type: B, layer: 1, pos: 9
type: B, layer: 1, pos: 1
type: B, layer: 1, pos: 39
type: B, layer: 1, pos: 6
type: B, layer: 1, pos: 48
type: B, layer: 1, pos: 28
type: B, layer: 1, pos: 38
type: B, layer: 1, pos: 41
type: B, layer: 1, pos: 0
type: B, layer: 1, pos: 8
type: B, layer: 1, pos: 24
type: B, layer: 1, pos: 37
type: B, layer: 1, pos: 15
type: B, layer: 1, pos: 31
type: B, layer: 1, pos: 26
type: B, layer: 1, pos: 45
type: B, layer: 1, pos: 20
type: B, layer: 1, pos: 5

Time for candidate selection: 0.06 seconds

### Candidate
type: B, layer: 1, pos: 9

## Relational analysis of NS_A2_B1_A1_B1_A1_B1
Optimization infeasible because this subproblem isn't reachable.

### Candidate
type: B, layer: 1, pos: 1

## Relational analysis of NS_A2_B1_A1_B1_A1_B1
Optimization infeasible because this subproblem isn't reachable.

### Candidate
type: B, layer: 1, pos: 39

## Relational analysis of NS_A2_B1_A1_B1_A1_B1
Optimization infeasible because this subproblem isn't reachable.

### Candidate
type: B, layer: 1, pos: 6

## Relational analysis of NS_A2_B1_A1_B1_A1_B1
Optimization infeasible because this subproblem isn't reachable.

### Candidate
type: B, layer: 1, pos: 48

## Relational analysis of NS_A2_B1_A1_B1_A1_B1
Optimization infeasible because this subproblem isn't reachable.

### Candidate
type: B, layer: 1, pos: 28

## Relational analysis of NS_A2_B1_A1_B1_A1_B1
Optimization infeasible because this subproblem isn't reachable.

### Candidate
type: B, layer: 1, pos: 38

## Relational analysis of NS_A2_B1_A1_B1_A1_B1
Optimization infeasible because this subproblem isn't reachable.

### Candidate
type: B, layer: 1, pos: 41

## Relational analysis of NS_A2_B1_A1_B1_A1_B1
Optimization infeasible because this subproblem isn't reachable.

### Candidate
type: B, layer: 1, pos: 0

## Relational analysis of NS_A2_B1_A1_B1_A1_B1
Optimization infeasible because this subproblem isn't reachable.

### Candidate
type: B, layer: 1, pos: 8

## Relational analysis of NS_A2_B1_A1_B1_A1_B1
Optimization infeasible because this subproblem isn't reachable.

### Candidate
type: B, layer: 1, pos: 24

## Relational analysis of NS_A2_B1_A1_B1_A1_B1
Optimization infeasible because this subproblem isn't reachable.

### Candidate
type: B, layer: 1, pos: 37

## Relational analysis of NS_A2_B1_A1_B1_A1_B1

### Relational analysis result of NS_A2_B1_A1_B1_A1_B1
Status: Status.UNKNOWN
Output dim: 0, lower bound: -20.9467403, upper bound: 20.9455978
time: 0.61 seconds

## Relational analysis of NS_A2_B1_A1_B1_A1_B2

### Relational analysis result of NS_A2_B1_A1_B1_A1_B2
Status: Status.UNKNOWN
Output dim: 0, lower bound: -20.9523051, upper bound: 20.9488332
time: 0.69 seconds

## BFS NS instance: NS_A2_B1_A1_B1_A2

### Backsubstitution after applying NS history:
0: -7.2611141, 19.7217655, -5.6865683, 16.0214005, -23.2825127, 25.4083328
1: -18.2746468, 30.0185661, -14.2480450, 24.6873779, -42.9620247, 44.2666092
2: -12.6608963, 27.2419930, -9.8974237, 22.3465786, -35.0074768, 37.1394119
3: -19.5136757, 33.3254776, -15.3260517, 27.2576427, -46.7713165, 48.6515274
4: -17.5834770, 33.6119156, -13.8730974, 27.6133747, -45.1968460, 47.4850121

Time for backsubstitution: 0.90 seconds

### NS candidates at layer 1
type: B, layer: 1, pos: 9
type: B, layer: 1, pos: 1
type: B, layer: 1, pos: 39
type: B, layer: 1, pos: 6
type: B, layer: 1, pos: 48
type: B, layer: 1, pos: 28
type: B, layer: 1, pos: 38
type: B, layer: 1, pos: 41
type: B, layer: 1, pos: 0
type: B, layer: 1, pos: 8
type: B, layer: 1, pos: 24
type: B, layer: 1, pos: 37
type: B, layer: 1, pos: 15
type: B, layer: 1, pos: 31
type: B, layer: 1, pos: 26
type: B, layer: 1, pos: 45
type: B, layer: 1, pos: 20
type: B, layer: 1, pos: 5

Time for candidate selection: 0.06 seconds

### Candidate
type: B, layer: 1, pos: 9

## Relational analysis of NS_A2_B1_A1_B1_A2_B1
Optimization infeasible because this subproblem isn't reachable.

### Candidate
type: B, layer: 1, pos: 1

## Relational analysis of NS_A2_B1_A1_B1_A2_B1
Optimization infeasible because this subproblem isn't reachable.

### Candidate
type: B, layer: 1, pos: 39

## Relational analysis of NS_A2_B1_A1_B1_A2_B1
Optimization infeasible because this subproblem isn't reachable.

### Candidate
type: B, layer: 1, pos: 6

## Relational analysis of NS_A2_B1_A1_B1_A2_B1
Optimization infeasible because this subproblem isn't reachable.

### Candidate
type: B, layer: 1, pos: 48

## Relational analysis of NS_A2_B1_A1_B1_A2_B1
Optimization infeasible because this subproblem isn't reachable.

### Candidate
type: B, layer: 1, pos: 28

## Relational analysis of NS_A2_B1_A1_B1_A2_B1
Optimization infeasible because this subproblem isn't reachable.

### Candidate
type: B, layer: 1, pos: 38

## Relational analysis of NS_A2_B1_A1_B1_A2_B1
Optimization infeasible because this subproblem isn't reachable.

### Candidate
type: B, layer: 1, pos: 41

## Relational analysis of NS_A2_B1_A1_B1_A2_B1
Optimization infeasible because this subproblem isn't reachable.

### Candidate
type: B, layer: 1, pos: 0

## Relational analysis of NS_A2_B1_A1_B1_A2_B1
Optimization infeasible because this subproblem isn't reachable.

### Candidate
type: B, layer: 1, pos: 8

## Relational analysis of NS_A2_B1_A1_B1_A2_B1
Optimization infeasible because this subproblem isn't reachable.

### Candidate
type: B, layer: 1, pos: 24

## Relational analysis of NS_A2_B1_A1_B1_A2_B1
Optimization infeasible because this subproblem isn't reachable.

### Candidate
type: B, layer: 1, pos: 37

## Relational analysis of NS_A2_B1_A1_B1_A2_B1

### Relational analysis result of NS_A2_B1_A1_B1_A2_B1
Status: Status.UNKNOWN
Output dim: 0, lower bound: -20.9467403, upper bound: 20.9455978
time: 0.94 seconds

## Relational analysis of NS_A2_B1_A1_B1_A2_B2

### Relational analysis result of NS_A2_B1_A1_B1_A2_B2
Status: Status.UNKNOWN
Output dim: 0, lower bound: -20.9523051, upper bound: 20.9488332
time: 0.70 seconds

## BFS NS instance: NS_A2_B1_A1_B2_A1

### Backsubstitution after applying NS history:
0: -7.2958264, 19.9399872, -5.7321081, 16.0435200, -23.3393421, 25.6720963
1: -18.5720406, 30.2115574, -14.2964621, 24.7329769, -43.3050117, 44.5080185
2: -12.7887821, 27.4622536, -9.9682322, 22.3510227, -35.1398048, 37.4304848
3: -19.6904030, 33.5532112, -15.3430071, 27.3167400, -47.0071411, 48.8962173
4: -17.5103760, 33.9847832, -13.8603640, 27.5282879, -45.0386658, 47.8451385

Time for backsubstitution: 0.90 seconds

### NS candidates at layer 1
type: B, layer: 1, pos: 9
type: B, layer: 1, pos: 1
type: B, layer: 1, pos: 39
type: B, layer: 1, pos: 6
type: B, layer: 1, pos: 48
type: B, layer: 1, pos: 38
type: B, layer: 1, pos: 28
type: B, layer: 1, pos: 0
type: B, layer: 1, pos: 41
type: B, layer: 1, pos: 37
type: B, layer: 1, pos: 24
type: B, layer: 1, pos: 31
type: B, layer: 1, pos: 8
type: B, layer: 1, pos: 20
type: B, layer: 1, pos: 45
type: B, layer: 1, pos: 5
type: B, layer: 1, pos: 26
type: B, layer: 1, pos: 15

Time for candidate selection: 0.06 seconds

### Candidate
type: B, layer: 1, pos: 9

## Relational analysis of NS_A2_B1_A1_B2_A1_B1

### Relational analysis result of NS_A2_B1_A1_B2_A1_B1
Status: Status.UNKNOWN
Output dim: 0, lower bound: -20.9273867, upper bound: 20.9234893
time: 0.77 seconds

## Relational analysis of NS_A2_B1_A1_B2_A1_B2

### Relational analysis result of NS_A2_B1_A1_B2_A1_B2
Status: Status.UNKNOWN
Output dim: 0, lower bound: -20.9521644, upper bound: 20.9477517
time: 0.70 seconds

## BFS NS instance: NS_A2_B1_A1_B2_A2

### Backsubstitution after applying NS history:
0: -7.2611141, 19.7217655, -5.7321081, 16.0435200, -23.3046341, 25.4538727
1: -18.2746468, 30.0185661, -14.2964621, 24.7329769, -43.0076218, 44.3150291
2: -12.6608963, 27.2419930, -9.9682322, 22.3510227, -35.0119171, 37.2102203
3: -19.5136757, 33.3254776, -15.3430071, 27.3167400, -46.8304138, 48.6684837
4: -17.5834770, 33.6119156, -13.8603640, 27.5282879, -45.1117592, 47.4722748

Time for backsubstitution: 0.91 seconds

### NS candidates at layer 1
type: B, layer: 1, pos: 9
type: B, layer: 1, pos: 1
type: B, layer: 1, pos: 39
type: B, layer: 1, pos: 6
type: B, layer: 1, pos: 48
type: B, layer: 1, pos: 38
type: B, layer: 1, pos: 28
type: B, layer: 1, pos: 0
type: B, layer: 1, pos: 41
type: B, layer: 1, pos: 37
type: B, layer: 1, pos: 24
type: B, layer: 1, pos: 31
type: B, layer: 1, pos: 8
type: B, layer: 1, pos: 20
type: B, layer: 1, pos: 45
type: B, layer: 1, pos: 5
type: B, layer: 1, pos: 26
type: B, layer: 1, pos: 15

Time for candidate selection: 0.06 seconds

### Candidate
type: B, layer: 1, pos: 9

## Relational analysis of NS_A2_B1_A1_B2_A2_B1

### Relational analysis result of NS_A2_B1_A1_B2_A2_B1
Status: Status.UNKNOWN
Output dim: 0, lower bound: -20.9273867, upper bound: 20.9234893
time: 0.80 seconds

## Relational analysis of NS_A2_B1_A1_B2_A2_B2

### Relational analysis result of NS_A2_B1_A1_B2_A2_B2
Status: Status.UNKNOWN
Output dim: 0, lower bound: -20.9521644, upper bound: 20.9477517
time: 1.07 seconds

## BFS NS instance: NS_A2_B1_A2_B1_A1

### Backsubstitution after applying NS history:
0: -5.7942882, 16.2022667, -5.9402876, 16.6799774, -22.4742661, 22.1425533
1: -14.4589930, 24.9580421, -14.9174976, 25.6486301, -40.1076202, 39.8755302
2: -10.0880079, 22.5631218, -10.3585854, 23.2349720, -33.3229752, 32.9217072
3: -15.5135593, 27.5783768, -16.0219460, 28.3389072, -43.8524628, 43.6003227
4: -14.0118141, 27.7913380, -14.4889812, 28.7199249, -42.7317314, 42.2803192

Time for backsubstitution: 0.90 seconds

### NS candidates at layer 1
type: B, layer: 1, pos: 1
type: B, layer: 1, pos: 9
type: B, layer: 1, pos: 6
type: B, layer: 1, pos: 39
type: B, layer: 1, pos: 28
type: B, layer: 1, pos: 48
type: B, layer: 1, pos: 38
type: B, layer: 1, pos: 41
type: B, layer: 1, pos: 0
type: B, layer: 1, pos: 8
type: B, layer: 1, pos: 24
type: B, layer: 1, pos: 20
type: B, layer: 1, pos: 31
type: B, layer: 1, pos: 37
type: B, layer: 1, pos: 15
type: B, layer: 1, pos: 45
type: B, layer: 1, pos: 26
type: B, layer: 1, pos: 5

Time for candidate selection: 0.06 seconds

### Candidate
type: B, layer: 1, pos: 1

## Relational analysis of NS_A2_B1_A2_B1_A1_B1
Optimization infeasible because this subproblem isn't reachable.

### Candidate
type: B, layer: 1, pos: 9

## Relational analysis of NS_A2_B1_A2_B1_A1_B1
Optimization infeasible because this subproblem isn't reachable.

### Candidate
type: B, layer: 1, pos: 6

## Relational analysis of NS_A2_B1_A2_B1_A1_B1
Optimization infeasible because this subproblem isn't reachable.

### Candidate
type: B, layer: 1, pos: 39

### Candidate
type: B, layer: 1, pos: 28

## Relational analysis of NS_A2_B1_A2_B1_A1_B1
Optimization infeasible because this subproblem isn't reachable.

### Candidate
type: B, layer: 1, pos: 48

## Relational analysis of NS_A2_B1_A2_B1_A1_B1
Optimization infeasible because this subproblem isn't reachable.

### Candidate
type: B, layer: 1, pos: 38

## Relational analysis of NS_A2_B1_A2_B1_A1_B1
Optimization infeasible because this subproblem isn't reachable.

### Candidate
type: B, layer: 1, pos: 41

## Relational analysis of NS_A2_B1_A2_B1_A1_B1
Optimization infeasible because this subproblem isn't reachable.

### Candidate
type: B, layer: 1, pos: 0

## Relational analysis of NS_A2_B1_A2_B1_A1_B1

### Relational analysis result of NS_A2_B1_A2_B1_A1_B1
Status: Status.UNKNOWN
Output dim: 0, lower bound: -20.9563050, upper bound: 20.9540767
time: 0.73 seconds

## Relational analysis of NS_A2_B1_A2_B1_A1_B2

### Relational analysis result of NS_A2_B1_A2_B1_A1_B2
Status: Status.UNKNOWN
Output dim: 0, lower bound: -20.9553052, upper bound: 20.9537253
time: 0.59 seconds

## BFS NS instance: NS_A2_B1_A2_B1_A2

### Backsubstitution after applying NS history:
0: -5.9756260, 16.6119061, -5.9402876, 16.6799774, -22.6555996, 22.5521927
1: -14.7459917, 25.6773434, -14.9174976, 25.6486301, -40.3946190, 40.5948372
2: -10.3569841, 23.1951981, -10.3585854, 23.2349720, -33.5919533, 33.5537796
3: -15.9492168, 28.3532829, -16.0219460, 28.3389072, -44.2881241, 44.3752289
4: -14.6166353, 28.4932518, -14.4889812, 28.7199249, -43.3365479, 42.9822311

Time for backsubstitution: 0.89 seconds

### NS candidates at layer 1
type: B, layer: 1, pos: 1
type: B, layer: 1, pos: 9
type: B, layer: 1, pos: 6
type: B, layer: 1, pos: 39
type: B, layer: 1, pos: 28
type: B, layer: 1, pos: 48
type: B, layer: 1, pos: 38
type: B, layer: 1, pos: 41
type: B, layer: 1, pos: 0
type: B, layer: 1, pos: 8
type: B, layer: 1, pos: 24
type: B, layer: 1, pos: 20
type: B, layer: 1, pos: 31
type: B, layer: 1, pos: 37
type: B, layer: 1, pos: 15
type: B, layer: 1, pos: 45
type: B, layer: 1, pos: 26
type: B, layer: 1, pos: 5

Time for candidate selection: 0.06 seconds

### Candidate
type: B, layer: 1, pos: 1

## Relational analysis of NS_A2_B1_A2_B1_A2_B1
Optimization infeasible because this subproblem isn't reachable.

### Candidate
type: B, layer: 1, pos: 9

## Relational analysis of NS_A2_B1_A2_B1_A2_B1
Optimization infeasible because this subproblem isn't reachable.

### Candidate
type: B, layer: 1, pos: 6

## Relational analysis of NS_A2_B1_A2_B1_A2_B1
Optimization infeasible because this subproblem isn't reachable.

### Candidate
type: B, layer: 1, pos: 39

### Candidate
type: B, layer: 1, pos: 28

## Relational analysis of NS_A2_B1_A2_B1_A2_B1
Optimization infeasible because this subproblem isn't reachable.

### Candidate
type: B, layer: 1, pos: 48

## Relational analysis of NS_A2_B1_A2_B1_A2_B1
Optimization infeasible because this subproblem isn't reachable.

### Candidate
type: B, layer: 1, pos: 38

## Relational analysis of NS_A2_B1_A2_B1_A2_B1
Optimization infeasible because this subproblem isn't reachable.

### Candidate
type: B, layer: 1, pos: 41

## Relational analysis of NS_A2_B1_A2_B1_A2_B1
Optimization infeasible because this subproblem isn't reachable.

### Candidate
type: B, layer: 1, pos: 0

## Relational analysis of NS_A2_B1_A2_B1_A2_B1

### Relational analysis result of NS_A2_B1_A2_B1_A2_B1
Status: Status.UNKNOWN
Output dim: 0, lower bound: -20.9563050, upper bound: 20.9540767
time: 0.67 seconds

## Relational analysis of NS_A2_B1_A2_B1_A2_B2

### Relational analysis result of NS_A2_B1_A2_B1_A2_B2
Status: Status.UNKNOWN
Output dim: 0, lower bound: -20.9553052, upper bound: 20.9537253
time: 0.86 seconds

## BFS NS instance: NS_A2_B1_A2_B2_A1

### Backsubstitution after applying NS history:
0: -5.7942882, 16.2022667, -6.0440979, 16.8430519, -22.6373405, 22.2463646
1: -14.4589930, 24.9580421, -15.1288452, 25.9006710, -40.3596649, 40.0868874
2: -10.0880079, 22.5631218, -10.5438957, 23.4227562, -33.5107651, 33.1070137
3: -15.5135593, 27.5783768, -16.1987095, 28.6353874, -44.1489334, 43.7770844
4: -14.0118141, 27.7913380, -14.6018887, 28.8638325, -42.8756485, 42.3932266

Time for backsubstitution: 0.89 seconds

### NS candidates at layer 1
type: B, layer: 1, pos: 1
type: B, layer: 1, pos: 9
type: B, layer: 1, pos: 39
type: B, layer: 1, pos: 6
type: B, layer: 1, pos: 48
type: B, layer: 1, pos: 38
type: B, layer: 1, pos: 28
type: B, layer: 1, pos: 0
type: B, layer: 1, pos: 41
type: B, layer: 1, pos: 20
type: B, layer: 1, pos: 31
type: B, layer: 1, pos: 37
type: B, layer: 1, pos: 24
type: B, layer: 1, pos: 8
type: B, layer: 1, pos: 45
type: B, layer: 1, pos: 26
type: B, layer: 1, pos: 5
type: B, layer: 1, pos: 15

Time for candidate selection: 0.06 seconds

### Candidate
type: B, layer: 1, pos: 1

## Relational analysis of NS_A2_B1_A2_B2_A1_B1

### Relational analysis result of NS_A2_B1_A2_B2_A1_B1
Status: Status.UNKNOWN
Output dim: 0, lower bound: -20.9553459, upper bound: 20.9514872
time: 0.67 seconds

## Relational analysis of NS_A2_B1_A2_B2_A1_B2

### Relational analysis result of NS_A2_B1_A2_B2_A1_B2
Status: Status.UNKNOWN
Output dim: 0, lower bound: -20.9553459, upper bound: 20.9544784
time: 0.62 seconds

## BFS NS instance: NS_A2_B1_A2_B2_A2

### Backsubstitution after applying NS history:
0: -5.9756260, 16.6119061, -6.0440979, 16.8430519, -22.8186779, 22.6560040
1: -14.7459917, 25.6773434, -15.1288452, 25.9006710, -40.6466637, 40.8061905
2: -10.3569841, 23.1951981, -10.5438957, 23.4227562, -33.7797394, 33.7390900
3: -15.9492168, 28.3532829, -16.1987095, 28.6353874, -44.5846024, 44.5519943
4: -14.6166353, 28.4932518, -14.6018887, 28.8638325, -43.4804688, 43.0951385

Time for backsubstitution: 0.89 seconds

### NS candidates at layer 1
type: B, layer: 1, pos: 1
type: B, layer: 1, pos: 9
type: B, layer: 1, pos: 39
type: B, layer: 1, pos: 6
type: B, layer: 1, pos: 48
type: B, layer: 1, pos: 38
type: B, layer: 1, pos: 28
type: B, layer: 1, pos: 0
type: B, layer: 1, pos: 41
type: B, layer: 1, pos: 20
type: B, layer: 1, pos: 31
type: B, layer: 1, pos: 37
type: B, layer: 1, pos: 24
type: B, layer: 1, pos: 8
type: B, layer: 1, pos: 45
type: B, layer: 1, pos: 26
type: B, layer: 1, pos: 5
type: B, layer: 1, pos: 15

Time for candidate selection: 0.06 seconds

### Candidate
type: B, layer: 1, pos: 1

## Relational analysis of NS_A2_B1_A2_B2_A2_B1

### Relational analysis result of NS_A2_B1_A2_B2_A2_B1
Status: Status.UNKNOWN
Output dim: 0, lower bound: -20.9553459, upper bound: 20.9514872
time: 0.60 seconds

## Relational analysis of NS_A2_B1_A2_B2_A2_B2

### Relational analysis result of NS_A2_B1_A2_B2_A2_B2
Status: Status.UNKNOWN
Output dim: 0, lower bound: -20.9553459, upper bound: 20.9544784
time: 1.08 seconds

## BFS NS instance: NS_A2_B2_A1_B1_A1

### Backsubstitution after applying NS history:
0: -7.2958264, 19.9399872, -5.9208970, 16.5725956, -23.8684177, 25.8608837
1: -18.5720406, 30.2115574, -14.6930676, 25.6041012, -44.1761398, 44.9046211
2: -12.7887821, 27.4622536, -10.2649250, 23.1677380, -35.9565201, 37.7271805
3: -19.6904030, 33.5532112, -15.9012699, 28.2439499, -47.9343491, 49.4544830
4: -17.5103760, 33.9847832, -14.5842028, 28.5637188, -46.0740967, 48.5689774

Time for backsubstitution: 0.90 seconds

### NS candidates at layer 1
type: B, layer: 1, pos: 1
type: B, layer: 1, pos: 39
type: B, layer: 1, pos: 9
type: B, layer: 1, pos: 6
type: B, layer: 1, pos: 28
type: B, layer: 1, pos: 48
type: B, layer: 1, pos: 38
type: B, layer: 1, pos: 0
type: B, layer: 1, pos: 24
type: B, layer: 1, pos: 41
type: B, layer: 1, pos: 37
type: B, layer: 1, pos: 8
type: B, layer: 1, pos: 20
type: B, layer: 1, pos: 26
type: B, layer: 1, pos: 15
type: B, layer: 1, pos: 5
type: B, layer: 1, pos: 45
type: B, layer: 1, pos: 31

Time for candidate selection: 0.06 seconds

### Candidate
type: B, layer: 1, pos: 1

## Relational analysis of NS_A2_B2_A1_B1_A1_B1
Optimization infeasible because this subproblem isn't reachable.

### Candidate
type: B, layer: 1, pos: 39

## Relational analysis of NS_A2_B2_A1_B1_A1_B1
Optimization infeasible because this subproblem isn't reachable.

### Candidate
type: B, layer: 1, pos: 9

## Relational analysis of NS_A2_B2_A1_B1_A1_B1
Optimization infeasible because this subproblem isn't reachable.

### Candidate
type: B, layer: 1, pos: 6

## Relational analysis of NS_A2_B2_A1_B1_A1_B1
Optimization infeasible because this subproblem isn't reachable.

### Candidate
type: B, layer: 1, pos: 28

## Relational analysis of NS_A2_B2_A1_B1_A1_B1
Optimization infeasible because this subproblem isn't reachable.

### Candidate
type: B, layer: 1, pos: 48

## Relational analysis of NS_A2_B2_A1_B1_A1_B1
Optimization infeasible because this subproblem isn't reachable.

### Candidate
type: B, layer: 1, pos: 38

## Relational analysis of NS_A2_B2_A1_B1_A1_B1
Optimization infeasible because this subproblem isn't reachable.

### Candidate
type: B, layer: 1, pos: 0

## Relational analysis of NS_A2_B2_A1_B1_A1_B1
Optimization infeasible because this subproblem isn't reachable.

### Candidate
type: B, layer: 1, pos: 24

## Relational analysis of NS_A2_B2_A1_B1_A1_B1
Optimization infeasible because this subproblem isn't reachable.

### Candidate
type: B, layer: 1, pos: 41

## Relational analysis of NS_A2_B2_A1_B1_A1_B1
Optimization infeasible because this subproblem isn't reachable.

### Candidate
type: B, layer: 1, pos: 37

## Relational analysis of NS_A2_B2_A1_B1_A1_B1

### Relational analysis result of NS_A2_B2_A1_B1_A1_B1
Status: Status.UNKNOWN
Output dim: 0, lower bound: -20.9438155, upper bound: 20.9431946
time: 0.59 seconds

## Relational analysis of NS_A2_B2_A1_B1_A1_B2

### Relational analysis result of NS_A2_B2_A1_B1_A1_B2
Status: Status.UNKNOWN
Output dim: 0, lower bound: -20.9429799, upper bound: 20.9423008
time: 0.58 seconds

## BFS NS instance: NS_A2_B2_A1_B1_A2

### Backsubstitution after applying NS history:
0: -7.2611141, 19.7217655, -5.9208970, 16.5725956, -23.8337097, 25.6426620
1: -18.2746468, 30.0185661, -14.6930676, 25.6041012, -43.8787460, 44.7116318
2: -12.6608963, 27.2419930, -10.2649250, 23.1677380, -35.8286362, 37.5069160
3: -19.5136757, 33.3254776, -15.9012699, 28.2439499, -47.7576256, 49.2267456
4: -17.5834770, 33.6119156, -14.5842028, 28.5637188, -46.1471939, 48.1961136

Time for backsubstitution: 0.90 seconds

### NS candidates at layer 1
type: B, layer: 1, pos: 1
type: B, layer: 1, pos: 39
type: B, layer: 1, pos: 9
type: B, layer: 1, pos: 6
type: B, layer: 1, pos: 28
type: B, layer: 1, pos: 48
type: B, layer: 1, pos: 38
type: B, layer: 1, pos: 0
type: B, layer: 1, pos: 24
type: B, layer: 1, pos: 41
type: B, layer: 1, pos: 37
type: B, layer: 1, pos: 8
type: B, layer: 1, pos: 20
type: B, layer: 1, pos: 26
type: B, layer: 1, pos: 15
type: B, layer: 1, pos: 5
type: B, layer: 1, pos: 45
type: B, layer: 1, pos: 31

Time for candidate selection: 0.06 seconds

### Candidate
type: B, layer: 1, pos: 1

## Relational analysis of NS_A2_B2_A1_B1_A2_B1
Optimization infeasible because this subproblem isn't reachable.

### Candidate
type: B, layer: 1, pos: 39

## Relational analysis of NS_A2_B2_A1_B1_A2_B1
Optimization infeasible because this subproblem isn't reachable.

### Candidate
type: B, layer: 1, pos: 9

## Relational analysis of NS_A2_B2_A1_B1_A2_B1
Optimization infeasible because this subproblem isn't reachable.

### Candidate
type: B, layer: 1, pos: 6

## Relational analysis of NS_A2_B2_A1_B1_A2_B1
Optimization infeasible because this subproblem isn't reachable.

### Candidate
type: B, layer: 1, pos: 28

## Relational analysis of NS_A2_B2_A1_B1_A2_B1
Optimization infeasible because this subproblem isn't reachable.

### Candidate
type: B, layer: 1, pos: 48

## Relational analysis of NS_A2_B2_A1_B1_A2_B1
Optimization infeasible because this subproblem isn't reachable.

### Candidate
type: B, layer: 1, pos: 38

## Relational analysis of NS_A2_B2_A1_B1_A2_B1
Optimization infeasible because this subproblem isn't reachable.

### Candidate
type: B, layer: 1, pos: 0

## Relational analysis of NS_A2_B2_A1_B1_A2_B1
Optimization infeasible because this subproblem isn't reachable.

### Candidate
type: B, layer: 1, pos: 24

## Relational analysis of NS_A2_B2_A1_B1_A2_B1
Optimization infeasible because this subproblem isn't reachable.

### Candidate
type: B, layer: 1, pos: 41

## Relational analysis of NS_A2_B2_A1_B1_A2_B1
Optimization infeasible because this subproblem isn't reachable.

### Candidate
type: B, layer: 1, pos: 37

## Relational analysis of NS_A2_B2_A1_B1_A2_B1

### Relational analysis result of NS_A2_B2_A1_B1_A2_B1
Status: Status.UNKNOWN
Output dim: 0, lower bound: -20.9438155, upper bound: 20.9431946
time: 0.59 seconds

## Relational analysis of NS_A2_B2_A1_B1_A2_B2

### Relational analysis result of NS_A2_B2_A1_B1_A2_B2
Status: Status.UNKNOWN
Output dim: 0, lower bound: -20.9429799, upper bound: 20.9423008
time: 0.62 seconds

## BFS NS instance: NS_A2_B2_A1_B2_A1

### Backsubstitution after applying NS history:
0: -7.2958264, 19.9399872, -5.9177461, 16.4657726, -23.7615929, 25.8577328
1: -18.5720406, 30.2115574, -14.5941172, 25.4797840, -44.0518265, 44.8056717
2: -12.7887821, 27.4622536, -10.2439365, 23.0011864, -35.7899666, 37.7061920
3: -19.6904030, 33.5532112, -15.7867498, 28.1186886, -47.8090858, 49.3399620
4: -17.5103760, 33.9847832, -14.4681406, 28.2543964, -45.7647705, 48.4529152

Time for backsubstitution: 0.90 seconds

### NS candidates at layer 1
type: B, layer: 1, pos: 1
type: B, layer: 1, pos: 39
type: B, layer: 1, pos: 9
type: B, layer: 1, pos: 6
type: B, layer: 1, pos: 48
type: B, layer: 1, pos: 38
type: B, layer: 1, pos: 28
type: B, layer: 1, pos: 0
type: B, layer: 1, pos: 24
type: B, layer: 1, pos: 37
type: B, layer: 1, pos: 41
type: B, layer: 1, pos: 15
type: B, layer: 1, pos: 8
type: B, layer: 1, pos: 20
type: B, layer: 1, pos: 31
type: B, layer: 1, pos: 26
type: B, layer: 1, pos: 45
type: B, layer: 1, pos: 5

Time for candidate selection: 0.06 seconds

### Candidate
type: B, layer: 1, pos: 1

## Relational analysis of NS_A2_B2_A1_B2_A1_B1

### Relational analysis result of NS_A2_B2_A1_B2_A1_B1
Status: Status.UNKNOWN
Output dim: 0, lower bound: -20.9361618, upper bound: 20.9361582
time: 0.57 seconds

## Relational analysis of NS_A2_B2_A1_B2_A1_B2

### Relational analysis result of NS_A2_B2_A1_B2_A1_B2
Status: Status.UNKNOWN
Output dim: 0, lower bound: -20.9361618, upper bound: 20.9403090
time: 0.62 seconds

## BFS NS instance: NS_A2_B2_A1_B2_A2

### Backsubstitution after applying NS history:
0: -7.2611141, 19.7217655, -5.9177461, 16.4657726, -23.7268829, 25.6395111
1: -18.2746468, 30.0185661, -14.5941172, 25.4797840, -43.7544327, 44.6126823
2: -12.6608963, 27.2419930, -10.2439365, 23.0011864, -35.6620827, 37.4859314
3: -19.5136757, 33.3254776, -15.7867498, 28.1186886, -47.6323624, 49.1122284
4: -17.5834770, 33.6119156, -14.4681406, 28.2543964, -45.8378716, 48.0800514

Time for backsubstitution: 0.90 seconds

### NS candidates at layer 1
type: B, layer: 1, pos: 1
type: B, layer: 1, pos: 39
type: B, layer: 1, pos: 9
type: B, layer: 1, pos: 6
type: B, layer: 1, pos: 48
type: B, layer: 1, pos: 38
type: B, layer: 1, pos: 28
type: B, layer: 1, pos: 0
type: B, layer: 1, pos: 24
type: B, layer: 1, pos: 37
type: B, layer: 1, pos: 41
type: B, layer: 1, pos: 15
type: B, layer: 1, pos: 8
type: B, layer: 1, pos: 20
type: B, layer: 1, pos: 31
type: B, layer: 1, pos: 26
type: B, layer: 1, pos: 45
type: B, layer: 1, pos: 5

Time for candidate selection: 0.06 seconds

### Candidate
type: B, layer: 1, pos: 1

## Relational analysis of NS_A2_B2_A1_B2_A2_B1

### Relational analysis result of NS_A2_B2_A1_B2_A2_B1
Status: Status.UNKNOWN
Output dim: 0, lower bound: -20.9361618, upper bound: 20.9361582
time: 0.63 seconds

## Relational analysis of NS_A2_B2_A1_B2_A2_B2

### Relational analysis result of NS_A2_B2_A1_B2_A2_B2
Status: Status.UNKNOWN
Output dim: 0, lower bound: -20.9361618, upper bound: 20.9403090
time: 0.71 seconds

## BFS NS instance: NS_A2_B2_A2_B1_A1

### Backsubstitution after applying NS history:
0: -5.7942882, 16.2022667, -6.2250795, 17.2942638, -23.0885525, 22.4273453
1: -14.4589930, 24.9580421, -15.5413942, 26.4833488, -40.9423409, 40.4994240
2: -10.0880079, 22.5631218, -10.8425531, 24.0679264, -34.1559334, 33.4056740
3: -15.5135593, 27.5783768, -16.7520714, 29.3229733, -44.8365326, 44.3304482
4: -14.0118141, 27.7913380, -15.2782488, 29.7155247, -43.7273407, 43.0695801

Time for backsubstitution: 0.90 seconds

### NS candidates at layer 1
type: B, layer: 1, pos: 12
type: B, layer: 1, pos: 9
type: B, layer: 1, pos: 6
type: B, layer: 1, pos: 48
type: B, layer: 1, pos: 38
type: B, layer: 1, pos: 28
type: B, layer: 1, pos: 39
type: B, layer: 1, pos: 0
type: B, layer: 1, pos: 24
type: B, layer: 1, pos: 37
type: B, layer: 1, pos: 41
type: B, layer: 1, pos: 8
type: B, layer: 1, pos: 31
type: B, layer: 1, pos: 20
type: B, layer: 1, pos: 26
type: B, layer: 1, pos: 15
type: B, layer: 1, pos: 5
type: B, layer: 1, pos: 45

Time for candidate selection: 0.06 seconds

### Candidate
type: B, layer: 1, pos: 12

## Relational analysis of NS_A2_B2_A2_B1_A1_B1
Optimization infeasible because this subproblem isn't reachable.

### Candidate
type: B, layer: 1, pos: 9

## Relational analysis of NS_A2_B2_A2_B1_A1_B1

### Relational analysis result of NS_A2_B2_A2_B1_A1_B1
Status: Status.UNKNOWN
Output dim: 0, lower bound: -20.9251292, upper bound: 20.9303579
time: 0.66 seconds

## Relational analysis of NS_A2_B2_A2_B1_A1_B2

### Relational analysis result of NS_A2_B2_A2_B1_A1_B2
Status: Status.UNKNOWN
Output dim: 0, lower bound: -20.9393947, upper bound: 20.9404709
time: 0.63 seconds

## BFS NS instance: NS_A2_B2_A2_B1_A2

### Backsubstitution after applying NS history:
0: -5.9756260, 16.6119061, -6.2250795, 17.2942638, -23.2698879, 22.8369865
1: -14.7459917, 25.6773434, -15.5413942, 26.4833488, -41.2293396, 41.2187309
2: -10.3569841, 23.1951981, -10.8425531, 24.0679264, -34.4249115, 34.0377502
3: -15.9492168, 28.3532829, -16.7520714, 29.3229733, -45.2721901, 45.1053543
4: -14.6166353, 28.4932518, -15.2782488, 29.7155247, -44.3321609, 43.7714996

Time for backsubstitution: 0.91 seconds

### NS candidates at layer 1
type: B, layer: 1, pos: 12
type: B, layer: 1, pos: 9
type: B, layer: 1, pos: 6
type: B, layer: 1, pos: 48
type: B, layer: 1, pos: 38
type: B, layer: 1, pos: 28
type: B, layer: 1, pos: 39
type: B, layer: 1, pos: 0
type: B, layer: 1, pos: 24
type: B, layer: 1, pos: 37
type: B, layer: 1, pos: 41
type: B, layer: 1, pos: 8
type: B, layer: 1, pos: 31
type: B, layer: 1, pos: 20
type: B, layer: 1, pos: 26
type: B, layer: 1, pos: 15
type: B, layer: 1, pos: 5
type: B, layer: 1, pos: 45

Time for candidate selection: 0.06 seconds

### Candidate
type: B, layer: 1, pos: 12

## Relational analysis of NS_A2_B2_A2_B1_A2_B1
Optimization infeasible because this subproblem isn't reachable.

### Candidate
type: B, layer: 1, pos: 9

## Relational analysis of NS_A2_B2_A2_B1_A2_B1

### Relational analysis result of NS_A2_B2_A2_B1_A2_B1
Status: Status.UNKNOWN
Output dim: 0, lower bound: -20.9251292, upper bound: 20.9302789
time: 1.02 seconds

## Relational analysis of NS_A2_B2_A2_B1_A2_B2

### Relational analysis result of NS_A2_B2_A2_B1_A2_B2
Status: Status.UNKNOWN
Output dim: 0, lower bound: -20.9393947, upper bound: 20.9404709
time: 0.66 seconds

## BFS NS instance: NS_A2_B2_A2_B2_A1

### Backsubstitution after applying NS history:
0: -5.7942882, 16.2022667, -6.0877700, 17.0216694, -22.8159580, 22.2900352
1: -14.4589930, 24.9580421, -15.1663723, 26.2364292, -40.6954193, 40.1244049
2: -10.0880079, 22.5631218, -10.5828571, 23.7615471, -33.8495483, 33.1459808
3: -15.5135593, 27.5783768, -16.3651867, 28.9663982, -44.4799576, 43.9435654
4: -14.0118141, 27.7913380, -14.9519472, 29.3174820, -43.3292961, 42.7432785

Time for backsubstitution: 0.92 seconds

### NS candidates at layer 1
type: B, layer: 1, pos: 12
type: B, layer: 1, pos: 9
type: B, layer: 1, pos: 6
type: B, layer: 1, pos: 39
type: B, layer: 1, pos: 48
type: B, layer: 1, pos: 28
type: B, layer: 1, pos: 38
type: B, layer: 1, pos: 0
type: B, layer: 1, pos: 24
type: B, layer: 1, pos: 41
type: B, layer: 1, pos: 37
type: B, layer: 1, pos: 20
type: B, layer: 1, pos: 15
type: B, layer: 1, pos: 8
type: B, layer: 1, pos: 26
type: B, layer: 1, pos: 31
type: B, layer: 1, pos: 5
type: B, layer: 1, pos: 45

Time for candidate selection: 0.06 seconds

### Candidate
type: B, layer: 1, pos: 12

## Relational analysis of NS_A2_B2_A2_B2_A1_B1

### Relational analysis result of NS_A2_B2_A2_B2_A1_B1
Status: Status.UNKNOWN
Output dim: 0, lower bound: -20.9480390, upper bound: 20.9480285
time: 0.64 seconds

## Relational analysis of NS_A2_B2_A2_B2_A1_B2

### Relational analysis result of NS_A2_B2_A2_B2_A1_B2
Status: Status.UNKNOWN
Output dim: 0, lower bound: -20.9480390, upper bound: 20.9480285
time: 0.71 seconds

## BFS NS instance: NS_A2_B2_A2_B2_A2

### Backsubstitution after applying NS history:
0: -5.9756260, 16.6119061, -6.0877700, 17.0216694, -22.9972935, 22.6996765
1: -14.7459917, 25.6773434, -15.1663723, 26.2364292, -40.9824142, 40.8437080
2: -10.3569841, 23.1951981, -10.5828571, 23.7615471, -34.1185265, 33.7780533
3: -15.9492168, 28.3532829, -16.3651867, 28.9663982, -44.9156151, 44.7184639
4: -14.6166353, 28.4932518, -14.9519472, 29.3174820, -43.9341125, 43.4451981

Time for backsubstitution: 0.91 seconds

### NS candidates at layer 1
type: B, layer: 1, pos: 12
type: B, layer: 1, pos: 9
type: B, layer: 1, pos: 6
type: B, layer: 1, pos: 39
type: B, layer: 1, pos: 48
type: B, layer: 1, pos: 28
type: B, layer: 1, pos: 38
type: B, layer: 1, pos: 0
type: B, layer: 1, pos: 24
type: B, layer: 1, pos: 41
type: B, layer: 1, pos: 37
type: B, layer: 1, pos: 20
type: B, layer: 1, pos: 15
type: B, layer: 1, pos: 8
type: B, layer: 1, pos: 26
type: B, layer: 1, pos: 31
type: B, layer: 1, pos: 5
type: B, layer: 1, pos: 45

Time for candidate selection: 0.06 seconds

### Candidate
type: B, layer: 1, pos: 12

## Relational analysis of NS_A2_B2_A2_B2_A2_B1
Optimization infeasible because this subproblem isn't reachable.

### Candidate
type: B, layer: 1, pos: 9

## Relational analysis of NS_A2_B2_A2_B2_A2_B1

### Relational analysis result of NS_A2_B2_A2_B2_A2_B1
Status: Status.UNKNOWN
Output dim: 0, lower bound: -20.9413831, upper bound: 20.9423844
time: 0.70 seconds

## Relational analysis of NS_A2_B2_A2_B2_A2_B2

### Relational analysis result of NS_A2_B2_A2_B2_A2_B2
Status: Status.UNKNOWN
Output dim: 0, lower bound: -20.9475212, upper bound: 20.9475028
time: 0.76 seconds

## Summary of splitting at layer (split count: 5)
- Time for NS candidates: 2.85 seconds
NS_A1_B1_A1_B1_A2_B1, status: Status.UNKNOWN, split count: 6, time: 2.85
Output dim: 0, lower bound: -20.9552630, upper bound: 20.9548567
NS_A1_B1_A1_B1_A2_B2, status: Status.UNKNOWN, split count: 6, time: 2.85
Output dim: 0, lower bound: -20.9649475, upper bound: 20.9640737
NS_A1_B1_A1_B2_A1_B1, status: Status.UNKNOWN, split count: 6, time: 2.85
Output dim: 0, lower bound: -20.9224596, upper bound: 20.9341830
NS_A1_B1_A1_B2_A1_B2, status: Status.UNKNOWN, split count: 6, time: 2.85
Output dim: 0, lower bound: -20.9224596, upper bound: 20.9467403
NS_A1_B1_A1_B2_A2_B1, status: Status.UNKNOWN, split count: 6, time: 2.85
Output dim: 0, lower bound: -20.9270553, upper bound: 20.9411884
NS_A1_B1_A1_B2_A2_B2, status: Status.UNKNOWN, split count: 6, time: 2.85
Output dim: 0, lower bound: -20.9270553, upper bound: 20.9523051
NS_A1_B1_A2_B1_A1_B1, status: Status.UNKNOWN, split count: 6, time: 2.85
Output dim: 0, lower bound: -20.9572021, upper bound: 20.9527657
NS_A1_B1_A2_B1_A1_B2, status: Status.UNKNOWN, split count: 6, time: 2.85
Output dim: 0, lower bound: -20.9572021, upper bound: 20.9564671
NS_A1_B1_A2_B1_A2_B1, status: Status.UNKNOWN, split count: 6, time: 2.85
Output dim: 0, lower bound: -20.9565773, upper bound: 20.9520676
NS_A1_B1_A2_B1_A2_B2, status: Status.UNKNOWN, split count: 6, time: 2.85
Output dim: 0, lower bound: -20.9584735, upper bound: 20.9549741
NS_A1_B1_A2_B2_A1_B1, status: Status.UNKNOWN, split count: 6, time: 2.85
Output dim: 0, lower bound: -20.9121909, upper bound: 20.9155805
NS_A1_B1_A2_B2_A1_B2, status: Status.UNKNOWN, split count: 6, time: 2.85
Output dim: 0, lower bound: -20.9121909, upper bound: 20.9248678
NS_A1_B1_A2_B2_A2_B1, status: Status.UNKNOWN, split count: 6, time: 2.85
Output dim: 0, lower bound: -20.9138254, upper bound: 20.9189541
NS_A1_B1_A2_B2_A2_B2, status: Status.UNKNOWN, split count: 6, time: 2.85
Output dim: 0, lower bound: -20.9138254, upper bound: 20.9435253
NS_A1_B2_A1_B1_A1_B1, status: Status.UNKNOWN, split count: 6, time: 2.85
Output dim: 0, lower bound: -20.9607899, upper bound: 20.9623820
NS_A1_B2_A1_B1_A1_B2, status: Status.UNKNOWN, split count: 6, time: 2.85
Output dim: 0, lower bound: -20.9607899, upper bound: 20.9623820
NS_A1_B2_A1_B1_A2_B1, status: Status.UNKNOWN, split count: 6, time: 2.85
Output dim: 0, lower bound: -20.9604393, upper bound: 20.9612852
NS_A1_B2_A1_B1_A2_B2, status: Status.UNKNOWN, split count: 6, time: 2.85
Output dim: 0, lower bound: -20.9604393, upper bound: 20.9612852
NS_A1_B2_A1_B2_A1_B1, status: Status.UNKNOWN, split count: 6, time: 2.85
Output dim: 0, lower bound: -20.9534221, upper bound: 20.9549159
NS_A1_B2_A1_B2_A1_B2, status: Status.UNKNOWN, split count: 6, time: 2.85
Output dim: 0, lower bound: -20.9534221, upper bound: 20.9553052
NS_A1_B2_A1_B2_A2_B1, status: Status.UNKNOWN, split count: 6, time: 2.85
Output dim: 0, lower bound: -20.9537253, upper bound: 20.9553052
NS_A1_B2_A1_B2_A2_B2, status: Status.UNKNOWN, split count: 6, time: 2.85
Output dim: 0, lower bound: -20.9537253, upper bound: 20.9553052
NS_A1_B2_A2_B1_A1_B1, status: Status.UNKNOWN, split count: 6, time: 2.85
Output dim: 0, lower bound: -20.9613011, upper bound: 20.9616617
NS_A1_B2_A2_B1_A1_B2, status: Status.UNKNOWN, split count: 6, time: 2.85
Output dim: 0, lower bound: -20.9613011, upper bound: 20.9616617
NS_A1_B2_A2_B1_A2_B1, status: Status.UNKNOWN, split count: 6, time: 2.85
Output dim: 0, lower bound: -20.9615105, upper bound: 20.9615105
NS_A1_B2_A2_B1_A2_B2, status: Status.UNKNOWN, split count: 6, time: 2.85
Output dim: 0, lower bound: -20.9615105, upper bound: 20.9615105
NS_A1_B2_A2_B2_A1_B1, status: Status.UNKNOWN, split count: 6, time: 2.85
Output dim: 0, lower bound: -20.9293848, upper bound: 20.9423952
NS_A1_B2_A2_B2_A1_B2, status: Status.UNKNOWN, split count: 6, time: 2.85
Output dim: 0, lower bound: -20.9293848, upper bound: 20.9423952
NS_A1_B2_A2_B2_A2_B1, status: Status.UNKNOWN, split count: 6, time: 2.85
Output dim: 0, lower bound: -20.9293848, upper bound: 20.9511787
NS_A1_B2_A2_B2_A2_B2, status: Status.UNKNOWN, split count: 6, time: 2.85
Output dim: 0, lower bound: -20.9512850, upper bound: 20.9511787
NS_A2_B1_A1_B1_A1_B1, status: Status.UNKNOWN, split count: 6, time: 2.85
Output dim: 0, lower bound: -20.9467403, upper bound: 20.9455978
NS_A2_B1_A1_B1_A1_B2, status: Status.UNKNOWN, split count: 6, time: 2.85
Output dim: 0, lower bound: -20.9523051, upper bound: 20.9488332
NS_A2_B1_A1_B1_A2_B1, status: Status.UNKNOWN, split count: 6, time: 2.85
Output dim: 0, lower bound: -20.9467403, upper bound: 20.9455978
NS_A2_B1_A1_B1_A2_B2, status: Status.UNKNOWN, split count: 6, time: 2.85
Output dim: 0, lower bound: -20.9523051, upper bound: 20.9488332
NS_A2_B1_A1_B2_A1_B1, status: Status.UNKNOWN, split count: 6, time: 2.85
Output dim: 0, lower bound: -20.9273867, upper bound: 20.9234893
NS_A2_B1_A1_B2_A1_B2, status: Status.UNKNOWN, split count: 6, time: 2.85
Output dim: 0, lower bound: -20.9521644, upper bound: 20.9477517
NS_A2_B1_A1_B2_A2_B1, status: Status.UNKNOWN, split count: 6, time: 2.85
Output dim: 0, lower bound: -20.9273867, upper bound: 20.9234893
NS_A2_B1_A1_B2_A2_B2, status: Status.UNKNOWN, split count: 6, time: 2.85
Output dim: 0, lower bound: -20.9521644, upper bound: 20.9477517
NS_A2_B1_A2_B1_A1_B1, status: Status.UNKNOWN, split count: 6, time: 2.85
Output dim: 0, lower bound: -20.9563050, upper bound: 20.9540767
NS_A2_B1_A2_B1_A1_B2, status: Status.UNKNOWN, split count: 6, time: 2.85
Output dim: 0, lower bound: -20.9553052, upper bound: 20.9537253
NS_A2_B1_A2_B1_A2_B1, status: Status.UNKNOWN, split count: 6, time: 2.85
Output dim: 0, lower bound: -20.9563050, upper bound: 20.9540767
NS_A2_B1_A2_B1_A2_B2, status: Status.UNKNOWN, split count: 6, time: 2.85
Output dim: 0, lower bound: -20.9553052, upper bound: 20.9537253
NS_A2_B1_A2_B2_A1_B1, status: Status.UNKNOWN, split count: 6, time: 2.85
Output dim: 0, lower bound: -20.9553459, upper bound: 20.9514872
NS_A2_B1_A2_B2_A1_B2, status: Status.UNKNOWN, split count: 6, time: 2.85
Output dim: 0, lower bound: -20.9553459, upper bound: 20.9544784
NS_A2_B1_A2_B2_A2_B1, status: Status.UNKNOWN, split count: 6, time: 2.85
Output dim: 0, lower bound: -20.9553459, upper bound: 20.9514872
NS_A2_B1_A2_B2_A2_B2, status: Status.UNKNOWN, split count: 6, time: 2.85
Output dim: 0, lower bound: -20.9553459, upper bound: 20.9544784
NS_A2_B2_A1_B1_A1_B1, status: Status.UNKNOWN, split count: 6, time: 2.85
Output dim: 0, lower bound: -20.9438155, upper bound: 20.9431946
NS_A2_B2_A1_B1_A1_B2, status: Status.UNKNOWN, split count: 6, time: 2.85
Output dim: 0, lower bound: -20.9429799, upper bound: 20.9423008
NS_A2_B2_A1_B1_A2_B1, status: Status.UNKNOWN, split count: 6, time: 2.85
Output dim: 0, lower bound: -20.9438155, upper bound: 20.9431946
NS_A2_B2_A1_B1_A2_B2, status: Status.UNKNOWN, split count: 6, time: 2.85
Output dim: 0, lower bound: -20.9429799, upper bound: 20.9423008
NS_A2_B2_A1_B2_A1_B1, status: Status.UNKNOWN, split count: 6, time: 2.85
Output dim: 0, lower bound: -20.9361618, upper bound: 20.9361582
NS_A2_B2_A1_B2_A1_B2, status: Status.UNKNOWN, split count: 6, time: 2.85
Output dim: 0, lower bound: -20.9361618, upper bound: 20.9403090
NS_A2_B2_A1_B2_A2_B1, status: Status.UNKNOWN, split count: 6, time: 2.85
Output dim: 0, lower bound: -20.9361618, upper bound: 20.9361582
NS_A2_B2_A1_B2_A2_B2, status: Status.UNKNOWN, split count: 6, time: 2.85
Output dim: 0, lower bound: -20.9361618, upper bound: 20.9403090
NS_A2_B2_A2_B1_A1_B1, status: Status.UNKNOWN, split count: 6, time: 2.85
Output dim: 0, lower bound: -20.9251292, upper bound: 20.9303579
NS_A2_B2_A2_B1_A1_B2, status: Status.UNKNOWN, split count: 6, time: 2.85
Output dim: 0, lower bound: -20.9393947, upper bound: 20.9404709
NS_A2_B2_A2_B1_A2_B1, status: Status.UNKNOWN, split count: 6, time: 2.85
Output dim: 0, lower bound: -20.9251292, upper bound: 20.9302789
NS_A2_B2_A2_B1_A2_B2, status: Status.UNKNOWN, split count: 6, time: 2.85
Output dim: 0, lower bound: -20.9393947, upper bound: 20.9404709
NS_A2_B2_A2_B2_A1_B1, status: Status.UNKNOWN, split count: 6, time: 2.85
Output dim: 0, lower bound: -20.9480390, upper bound: 20.9480285
NS_A2_B2_A2_B2_A1_B2, status: Status.UNKNOWN, split count: 6, time: 2.85
Output dim: 0, lower bound: -20.9480390, upper bound: 20.9480285
NS_A2_B2_A2_B2_A2_B1, status: Status.UNKNOWN, split count: 6, time: 2.85
Output dim: 0, lower bound: -20.9413831, upper bound: 20.9423844
NS_A2_B2_A2_B2_A2_B2, status: Status.UNKNOWN, split count: 6, time: 2.85
Output dim: 0, lower bound: -20.9475212, upper bound: 20.9475028

## BFS NS instance: NS_A1_B1_A1_B1_A2_B1

### Backsubstitution after applying NS history:
0: -5.2203469, 14.7176580, -4.5916882, 13.0146313, -18.2349777, 19.3093452
1: -12.9302711, 22.7937088, -11.3174953, 20.1672020, -33.0974731, 34.1112061
2: -9.0256786, 20.6060429, -7.8993845, 18.2288322, -27.2545052, 28.5054283
3: -14.0143957, 25.1264000, -12.2989597, 22.2310638, -36.2454605, 37.4253616
4: -12.8208933, 25.3865643, -11.2507191, 22.4210014, -35.2418823, 36.6372833

Time for backsubstitution: 0.89 seconds

### NS candidates at layer 1
type: A, layer: 1, pos: 1
type: A, layer: 1, pos: 39
type: A, layer: 1, pos: 9
type: A, layer: 1, pos: 48
type: A, layer: 1, pos: 6
type: A, layer: 1, pos: 38
type: A, layer: 1, pos: 28
type: A, layer: 1, pos: 0
type: A, layer: 1, pos: 41
type: A, layer: 1, pos: 24
type: A, layer: 1, pos: 8
type: A, layer: 1, pos: 20
type: A, layer: 1, pos: 31
type: A, layer: 1, pos: 15
type: A, layer: 1, pos: 26
type: A, layer: 1, pos: 5
type: A, layer: 1, pos: 45

Time for candidate selection: 0.06 seconds

### Candidate
type: A, layer: 1, pos: 1

### Candidate
type: A, layer: 1, pos: 39

### Candidate
type: A, layer: 1, pos: 9

### Candidate
type: A, layer: 1, pos: 48

### Candidate
type: A, layer: 1, pos: 6

### Candidate
type: A, layer: 1, pos: 38

### Candidate
type: A, layer: 1, pos: 28

### Candidate
type: A, layer: 1, pos: 0

### Candidate
type: A, layer: 1, pos: 41

### Candidate
type: A, layer: 1, pos: 24

### Candidate
type: A, layer: 1, pos: 8

### Candidate
type: A, layer: 1, pos: 20

## Relational analysis of NS_A1_B1_A1_B1_A2_B1_A1
Optimization infeasible because this subproblem isn't reachable.

### Candidate
type: A, layer: 1, pos: 31

## Relational analysis of NS_A1_B1_A1_B1_A2_B1_A1
Optimization infeasible because this subproblem isn't reachable.

### Candidate
type: A, layer: 1, pos: 15

### Candidate
type: A, layer: 1, pos: 26

## Relational analysis of NS_A1_B1_A1_B1_A2_B1_A1
Optimization infeasible because this subproblem isn't reachable.

### Candidate
type: A, layer: 1, pos: 5

## Relational analysis of NS_A1_B1_A1_B1_A2_B1_A1
Optimization infeasible because this subproblem isn't reachable.

### Candidate
type: A, layer: 1, pos: 45

## Relational analysis of NS_A1_B1_A1_B1_A2_B1_A1
Optimization infeasible because this subproblem isn't reachable.

## BFS NS instance: NS_A1_B1_A1_B1_A2_B2

### Backsubstitution after applying NS history:
0: -5.6777620, 15.9427109, -5.9206009, 16.5693340, -22.2470894, 21.8633118
1: -14.1735039, 24.5664768, -14.9013968, 25.3112659, -39.4847717, 39.4678726
2: -9.8679581, 22.2568016, -10.3397408, 23.0329132, -32.9008636, 32.5965385
3: -15.2831621, 27.1275311, -15.9853773, 28.0273838, -43.3105469, 43.1129074
4: -13.9000025, 27.4807911, -14.4383097, 28.4950523, -42.3950500, 41.9191017

Time for backsubstitution: 0.87 seconds

### NS candidates at layer 1
type: A, layer: 1, pos: 1
type: A, layer: 1, pos: 9
type: A, layer: 1, pos: 39
type: A, layer: 1, pos: 6
type: A, layer: 1, pos: 48
type: A, layer: 1, pos: 28
type: A, layer: 1, pos: 38
type: A, layer: 1, pos: 0
type: A, layer: 1, pos: 41
type: A, layer: 1, pos: 8
type: A, layer: 1, pos: 24
type: A, layer: 1, pos: 15
type: A, layer: 1, pos: 20
type: A, layer: 1, pos: 31
type: A, layer: 1, pos: 26
type: A, layer: 1, pos: 5
type: A, layer: 1, pos: 45

Time for candidate selection: 0.06 seconds

### Candidate
type: A, layer: 1, pos: 1

### Candidate
type: A, layer: 1, pos: 9

### Candidate
type: A, layer: 1, pos: 39

### Candidate
type: A, layer: 1, pos: 6

### Candidate
type: A, layer: 1, pos: 48

### Candidate
type: A, layer: 1, pos: 28

### Candidate
type: A, layer: 1, pos: 38

### Candidate
type: A, layer: 1, pos: 0

### Candidate
type: A, layer: 1, pos: 41

### Candidate
type: A, layer: 1, pos: 8

### Candidate
type: A, layer: 1, pos: 24

### Candidate
type: A, layer: 1, pos: 15

### Candidate
type: A, layer: 1, pos: 20

## Relational analysis of NS_A1_B1_A1_B1_A2_B2_A1
Optimization infeasible because this subproblem isn't reachable.

### Candidate
type: A, layer: 1, pos: 31

## Relational analysis of NS_A1_B1_A1_B1_A2_B2_A1
Optimization infeasible because this subproblem isn't reachable.

### Candidate
type: A, layer: 1, pos: 26

## Relational analysis of NS_A1_B1_A1_B1_A2_B2_A1
Optimization infeasible because this subproblem isn't reachable.

### Candidate
type: A, layer: 1, pos: 5

## Relational analysis of NS_A1_B1_A1_B1_A2_B2_A1

### Relational analysis result of NS_A1_B1_A1_B1_A2_B2_A1
Status: Status.UNKNOWN
Output dim: 0, lower bound: -20.9647999, upper bound: 20.9639122
time: 0.76 seconds

## Relational analysis of NS_A1_B1_A1_B1_A2_B2_A2
Optimization infeasible because this subproblem isn't reachable.

### Candidate
type: A, layer: 1, pos: 45

## Relational analysis of NS_A1_B1_A1_B1_A2_B2_A1
Optimization infeasible because this subproblem isn't reachable.

## BFS NS instance: NS_A1_B1_A1_B2_A1_B1

### Backsubstitution after applying NS history:
0: -5.5220685, 15.5871372, -6.0501447, 16.8363934, -22.3584614, 21.6372814
1: -13.8160563, 24.0525360, -15.0581722, 25.8237381, -39.6397934, 39.1106987
2: -9.6000080, 21.7625408, -10.5215893, 23.4539528, -33.0539627, 32.2841301
3: -14.8792744, 26.5470657, -16.2723713, 28.5780029, -43.4572716, 42.8194313
4: -13.4775257, 26.8844700, -14.8797617, 28.9397259, -42.4172516, 41.7642326

Time for backsubstitution: 0.87 seconds

### NS candidates at layer 1
type: A, layer: 1, pos: 9
type: A, layer: 1, pos: 1
type: A, layer: 1, pos: 39
type: A, layer: 1, pos: 48
type: A, layer: 1, pos: 6
type: A, layer: 1, pos: 28
type: A, layer: 1, pos: 38
type: A, layer: 1, pos: 0
type: A, layer: 1, pos: 41
type: A, layer: 1, pos: 24
type: A, layer: 1, pos: 8
type: A, layer: 1, pos: 15
type: A, layer: 1, pos: 20
type: A, layer: 1, pos: 31
type: A, layer: 1, pos: 26
type: A, layer: 1, pos: 5
type: A, layer: 1, pos: 45

Time for candidate selection: 0.06 seconds

### Candidate
type: A, layer: 1, pos: 9

### Candidate
type: A, layer: 1, pos: 1

### Candidate
type: A, layer: 1, pos: 39

### Candidate
type: A, layer: 1, pos: 48

### Candidate
type: A, layer: 1, pos: 6

### Candidate
type: A, layer: 1, pos: 28

### Candidate
type: A, layer: 1, pos: 38

### Candidate
type: A, layer: 1, pos: 0

### Candidate
type: A, layer: 1, pos: 41

### Candidate
type: A, layer: 1, pos: 24

### Candidate
type: A, layer: 1, pos: 8

### Candidate
type: A, layer: 1, pos: 15

## Relational analysis of NS_A1_B1_A1_B2_A1_B1_A1
Optimization infeasible because this subproblem isn't reachable.

### Candidate
type: A, layer: 1, pos: 20

## Relational analysis of NS_A1_B1_A1_B2_A1_B1_A1
Optimization infeasible because this subproblem isn't reachable.

### Candidate
type: A, layer: 1, pos: 31

## Relational analysis of NS_A1_B1_A1_B2_A1_B1_A1
Optimization infeasible because this subproblem isn't reachable.

### Candidate
type: A, layer: 1, pos: 26

## Relational analysis of NS_A1_B1_A1_B2_A1_B1_A1
Optimization infeasible because this subproblem isn't reachable.

### Candidate
type: A, layer: 1, pos: 5

## Relational analysis of NS_A1_B1_A1_B2_A1_B1_A1

### Relational analysis result of NS_A1_B1_A1_B2_A1_B1_A1
Status: Status.UNKNOWN
Output dim: 0, lower bound: -20.9222662, upper bound: 20.9340596
time: 0.60 seconds

## Relational analysis of NS_A1_B1_A1_B2_A1_B1_A2
Optimization infeasible because this subproblem isn't reachable.

### Candidate
type: A, layer: 1, pos: 45

## Relational analysis of NS_A1_B1_A1_B2_A1_B1_A1
Optimization infeasible because this subproblem isn't reachable.

## BFS NS instance: NS_A1_B1_A1_B2_A1_B2

### Backsubstitution after applying NS history:
0: -5.5220685, 15.5871372, -6.3116179, 17.4869881, -23.0090542, 21.8987541
1: -13.8160563, 24.0525360, -15.6882200, 26.7641907, -40.5802422, 39.7407455
2: -9.6000080, 21.7625408, -10.9718370, 24.3357201, -33.9357262, 32.7343788
3: -14.8792744, 26.5470657, -16.9725456, 29.6498413, -44.5291138, 43.5196114
4: -13.4775257, 26.8844700, -15.5468330, 30.0257492, -43.5032692, 42.4313049

Time for backsubstitution: 0.87 seconds

### NS candidates at layer 1
type: A, layer: 1, pos: 9
type: A, layer: 1, pos: 1
type: A, layer: 1, pos: 39
type: A, layer: 1, pos: 48
type: A, layer: 1, pos: 6
type: A, layer: 1, pos: 28
type: A, layer: 1, pos: 38
type: A, layer: 1, pos: 0
type: A, layer: 1, pos: 41
type: A, layer: 1, pos: 24
type: A, layer: 1, pos: 8
type: A, layer: 1, pos: 15
type: A, layer: 1, pos: 20
type: A, layer: 1, pos: 31
type: A, layer: 1, pos: 26
type: A, layer: 1, pos: 5
type: A, layer: 1, pos: 45

Time for candidate selection: 0.06 seconds

### Candidate
type: A, layer: 1, pos: 9

### Candidate
type: A, layer: 1, pos: 1

### Candidate
type: A, layer: 1, pos: 39

### Candidate
type: A, layer: 1, pos: 48

### Candidate
type: A, layer: 1, pos: 6

### Candidate
type: A, layer: 1, pos: 28

### Candidate
type: A, layer: 1, pos: 38

### Candidate
type: A, layer: 1, pos: 0

### Candidate
type: A, layer: 1, pos: 41

### Candidate
type: A, layer: 1, pos: 24

### Candidate
type: A, layer: 1, pos: 8

### Candidate
type: A, layer: 1, pos: 15

## Relational analysis of NS_A1_B1_A1_B2_A1_B2_A1
Optimization infeasible because this subproblem isn't reachable.

### Candidate
type: A, layer: 1, pos: 20

## Relational analysis of NS_A1_B1_A1_B2_A1_B2_A1
Optimization infeasible because this subproblem isn't reachable.

### Candidate
type: A, layer: 1, pos: 31

## Relational analysis of NS_A1_B1_A1_B2_A1_B2_A1
Optimization infeasible because this subproblem isn't reachable.

### Candidate
type: A, layer: 1, pos: 26

## Relational analysis of NS_A1_B1_A1_B2_A1_B2_A1
Optimization infeasible because this subproblem isn't reachable.

### Candidate
type: A, layer: 1, pos: 5

## Relational analysis of NS_A1_B1_A1_B2_A1_B2_A1

### Relational analysis result of NS_A1_B1_A1_B2_A1_B2_A1
Status: Status.UNKNOWN
Output dim: 0, lower bound: -20.9222662, upper bound: 20.9466569
time: 1.04 seconds

## Relational analysis of NS_A1_B1_A1_B2_A1_B2_A2
Optimization infeasible because this subproblem isn't reachable.

### Candidate
type: A, layer: 1, pos: 45

## Relational analysis of NS_A1_B1_A1_B2_A1_B2_A1
Optimization infeasible because this subproblem isn't reachable.

## BFS NS instance: NS_A1_B1_A1_B2_A2_B1

### Backsubstitution after applying NS history:
0: -5.7029700, 16.0100746, -6.0501447, 16.8363934, -22.5393639, 22.0602188
1: -14.2405682, 24.6658897, -15.0581722, 25.8237381, -40.0643082, 39.7240601
2: -9.9136543, 22.3478088, -10.5215893, 23.4539528, -33.3676033, 32.8693962
3: -15.3522701, 27.2383041, -16.2723713, 28.5780029, -43.9302750, 43.5106697
4: -13.9597645, 27.5948181, -14.8797617, 28.9397259, -42.8994904, 42.4745789

Time for backsubstitution: 0.87 seconds

### NS candidates at layer 1
type: A, layer: 1, pos: 1
type: A, layer: 1, pos: 9
type: A, layer: 1, pos: 39
type: A, layer: 1, pos: 6
type: A, layer: 1, pos: 48
type: A, layer: 1, pos: 28
type: A, layer: 1, pos: 38
type: A, layer: 1, pos: 0
type: A, layer: 1, pos: 41
type: A, layer: 1, pos: 8
type: A, layer: 1, pos: 24
type: A, layer: 1, pos: 20
type: A, layer: 1, pos: 15
type: A, layer: 1, pos: 31
type: A, layer: 1, pos: 26
type: A, layer: 1, pos: 5
type: A, layer: 1, pos: 45

Time for candidate selection: 0.06 seconds

### Candidate
type: A, layer: 1, pos: 1

### Candidate
type: A, layer: 1, pos: 9

### Candidate
type: A, layer: 1, pos: 39

### Candidate
type: A, layer: 1, pos: 6

### Candidate
type: A, layer: 1, pos: 48

### Candidate
type: A, layer: 1, pos: 28

### Candidate
type: A, layer: 1, pos: 38

### Candidate
type: A, layer: 1, pos: 0

### Candidate
type: A, layer: 1, pos: 41

### Candidate
type: A, layer: 1, pos: 8

### Candidate
type: A, layer: 1, pos: 24

### Candidate
type: A, layer: 1, pos: 20

## Relational analysis of NS_A1_B1_A1_B2_A2_B1_A1

### Relational analysis result of NS_A1_B1_A1_B2_A2_B1_A1
Status: Status.UNKNOWN
Output dim: 0, lower bound: -20.9059363, upper bound: 20.9232720
time: 0.65 seconds

## Relational analysis of NS_A1_B1_A1_B2_A2_B1_A2

### Relational analysis result of NS_A1_B1_A1_B2_A2_B1_A2
Status: Status.UNKNOWN
Output dim: 0, lower bound: -20.9059363, upper bound: 20.9411884
time: 1.18 seconds

## BFS NS instance: NS_A1_B1_A1_B2_A2_B2

### Backsubstitution after applying NS history:
0: -5.7029700, 16.0100746, -6.3116179, 17.4869881, -23.1899567, 22.3216934
1: -14.2405682, 24.6658897, -15.6882200, 26.7641907, -41.0047607, 40.3541107
2: -9.9136543, 22.3478088, -10.9718370, 24.3357201, -34.2493706, 33.3196449
3: -15.3522701, 27.2383041, -16.9725456, 29.6498413, -45.0021133, 44.2108459
4: -13.9597645, 27.5948181, -15.5468330, 30.0257492, -43.9855080, 43.1416512

Time for backsubstitution: 0.88 seconds

### NS candidates at layer 1
type: A, layer: 1, pos: 1
type: A, layer: 1, pos: 9
type: A, layer: 1, pos: 39
type: A, layer: 1, pos: 6
type: A, layer: 1, pos: 48
type: A, layer: 1, pos: 28
type: A, layer: 1, pos: 38
type: A, layer: 1, pos: 0
type: A, layer: 1, pos: 41
type: A, layer: 1, pos: 8
type: A, layer: 1, pos: 24
type: A, layer: 1, pos: 20
type: A, layer: 1, pos: 15
type: A, layer: 1, pos: 31
type: A, layer: 1, pos: 26
type: A, layer: 1, pos: 5
type: A, layer: 1, pos: 45

Time for candidate selection: 0.06 seconds

### Candidate
type: A, layer: 1, pos: 1

### Candidate
type: A, layer: 1, pos: 9

### Candidate
type: A, layer: 1, pos: 39

### Candidate
type: A, layer: 1, pos: 6

### Candidate
type: A, layer: 1, pos: 48

### Candidate
type: A, layer: 1, pos: 28

### Candidate
type: A, layer: 1, pos: 38

### Candidate
type: A, layer: 1, pos: 0

### Candidate
type: A, layer: 1, pos: 41

### Candidate
type: A, layer: 1, pos: 8

### Candidate
type: A, layer: 1, pos: 24

### Candidate
type: A, layer: 1, pos: 20

## Relational analysis of NS_A1_B1_A1_B2_A2_B2_A1

### Relational analysis result of NS_A1_B1_A1_B2_A2_B2_A1
Status: Status.UNKNOWN
Output dim: 0, lower bound: -20.9059363, upper bound: 20.9460725
time: 0.67 seconds

## Relational analysis of NS_A1_B1_A1_B2_A2_B2_A2

### Relational analysis result of NS_A1_B1_A1_B2_A2_B2_A2
Status: Status.UNKNOWN
Output dim: 0, lower bound: -20.9059363, upper bound: 20.9523051
time: 0.72 seconds

## BFS NS instance: NS_A1_B1_A2_B1_A1_B1

### Backsubstitution after applying NS history:
0: -5.4812288, 15.4409924, -5.3478351, 15.0860615, -20.5672894, 20.7888222
1: -13.5658493, 23.9469128, -13.3962164, 23.1866322, -36.7524796, 37.3431244
2: -9.4828529, 21.6332130, -9.2835388, 20.9972038, -30.4800568, 30.9167480
3: -14.7106934, 26.3754158, -14.4139738, 25.6347923, -40.3454819, 40.7893906
4: -13.4949007, 26.6610184, -12.9984131, 25.9302464, -39.4251480, 39.6594315

Time for backsubstitution: 0.88 seconds

### NS candidates at layer 1
type: A, layer: 1, pos: 39
type: A, layer: 1, pos: 1
type: A, layer: 1, pos: 9
type: A, layer: 1, pos: 6
type: A, layer: 1, pos: 48
type: A, layer: 1, pos: 28
type: A, layer: 1, pos: 38
type: A, layer: 1, pos: 0
type: A, layer: 1, pos: 24
type: A, layer: 1, pos: 41
type: A, layer: 1, pos: 20
type: A, layer: 1, pos: 8
type: A, layer: 1, pos: 26
type: A, layer: 1, pos: 15
type: A, layer: 1, pos: 5
type: A, layer: 1, pos: 45
type: A, layer: 1, pos: 31

Time for candidate selection: 0.06 seconds

### Candidate
type: A, layer: 1, pos: 39

### Candidate
type: A, layer: 1, pos: 1

### Candidate
type: A, layer: 1, pos: 9

### Candidate
type: A, layer: 1, pos: 6

### Candidate
type: A, layer: 1, pos: 48

### Candidate
type: A, layer: 1, pos: 28

### Candidate
type: A, layer: 1, pos: 38

### Candidate
type: A, layer: 1, pos: 0

### Candidate
type: A, layer: 1, pos: 24

### Candidate
type: A, layer: 1, pos: 41

### Candidate
type: A, layer: 1, pos: 20

## Relational analysis of NS_A1_B1_A2_B1_A1_B1_A1
Optimization infeasible because this subproblem isn't reachable.

### Candidate
type: A, layer: 1, pos: 8

## Relational analysis of NS_A1_B1_A2_B1_A1_B1_A1
Optimization infeasible because this subproblem isn't reachable.

### Candidate
type: A, layer: 1, pos: 26

## Relational analysis of NS_A1_B1_A2_B1_A1_B1_A1
Optimization infeasible because this subproblem isn't reachable.

### Candidate
type: A, layer: 1, pos: 15

## Relational analysis of NS_A1_B1_A2_B1_A1_B1_A1
Optimization infeasible because this subproblem isn't reachable.

### Candidate
type: A, layer: 1, pos: 5

## Relational analysis of NS_A1_B1_A2_B1_A1_B1_A1

### Relational analysis result of NS_A1_B1_A2_B1_A1_B1_A1
Status: Status.UNKNOWN
Output dim: 0, lower bound: -20.9536226, upper bound: 20.9473637
time: 0.56 seconds

## Relational analysis of NS_A1_B1_A2_B1_A1_B1_A2
Optimization infeasible because this subproblem isn't reachable.

### Candidate
type: A, layer: 1, pos: 45

## Relational analysis of NS_A1_B1_A2_B1_A1_B1_A1
Optimization infeasible because this subproblem isn't reachable.

### Candidate
type: A, layer: 1, pos: 31

## Relational analysis of NS_A1_B1_A2_B1_A1_B1_A1
Optimization infeasible because this subproblem isn't reachable.

## BFS NS instance: NS_A1_B1_A2_B1_A1_B2

### Backsubstitution after applying NS history:
0: -5.7593737, 16.1530533, -5.9931288, 16.7647247, -22.5240955, 22.1461830
1: -14.2705688, 24.9937592, -15.0946112, 25.5939693, -39.8645401, 40.0883713
2: -9.9754887, 22.6008472, -10.4719324, 23.2889252, -33.2644081, 33.0727806
3: -15.4613800, 27.5570297, -16.1841564, 28.3482685, -43.8096466, 43.7411880
4: -14.1926994, 27.8550491, -14.6083412, 28.8173161, -43.0100060, 42.4633865

Time for backsubstitution: 0.88 seconds

### NS candidates at layer 1
type: A, layer: 1, pos: 1
type: A, layer: 1, pos: 39
type: A, layer: 1, pos: 9
type: A, layer: 1, pos: 6
type: A, layer: 1, pos: 48
type: A, layer: 1, pos: 28
type: A, layer: 1, pos: 38
type: A, layer: 1, pos: 0
type: A, layer: 1, pos: 24
type: A, layer: 1, pos: 41
type: A, layer: 1, pos: 8
type: A, layer: 1, pos: 20
type: A, layer: 1, pos: 26
type: A, layer: 1, pos: 15
type: A, layer: 1, pos: 5
type: A, layer: 1, pos: 45
type: A, layer: 1, pos: 31

Time for candidate selection: 0.06 seconds

### Candidate
type: A, layer: 1, pos: 1

### Candidate
type: A, layer: 1, pos: 39

### Candidate
type: A, layer: 1, pos: 9

### Candidate
type: A, layer: 1, pos: 6

### Candidate
type: A, layer: 1, pos: 48

### Candidate
type: A, layer: 1, pos: 28

### Candidate
type: A, layer: 1, pos: 38

### Candidate
type: A, layer: 1, pos: 0

### Candidate
type: A, layer: 1, pos: 24

### Candidate
type: A, layer: 1, pos: 41

### Candidate
type: A, layer: 1, pos: 8

## Relational analysis of NS_A1_B1_A2_B1_A1_B2_A1
Optimization infeasible because this subproblem isn't reachable.

### Candidate
type: A, layer: 1, pos: 20

## Relational analysis of NS_A1_B1_A2_B1_A1_B2_A1

### Relational analysis result of NS_A1_B1_A2_B1_A1_B2_A1
Status: Status.UNKNOWN
Output dim: 0, lower bound: -20.9548550, upper bound: 20.9501143
time: 0.56 seconds

## Relational analysis of NS_A1_B1_A2_B1_A1_B2_A2

### Relational analysis result of NS_A1_B1_A2_B1_A1_B2_A2
Status: Status.UNKNOWN
Output dim: 0, lower bound: -20.9548550, upper bound: 20.9564671
time: 0.63 seconds

## BFS NS instance: NS_A1_B1_A2_B1_A2_B1

### Backsubstitution after applying NS history:
0: -5.6857557, 15.9304600, -5.3057251, 14.9744167, -20.6601715, 21.2361832
1: -14.0632792, 24.6490192, -13.2811995, 23.0272408, -37.0905151, 37.9302177
2: -9.8403530, 22.3031597, -9.2066679, 20.8499794, -30.6903324, 31.5098267
3: -15.2543459, 27.1746693, -14.2973461, 25.4538841, -40.7082253, 41.4720078
4: -14.0293999, 27.4854355, -12.9021292, 25.7455139, -39.7749062, 40.3875580

Time for backsubstitution: 0.88 seconds

### NS candidates at layer 1
type: A, layer: 1, pos: 1
type: A, layer: 1, pos: 39
type: A, layer: 1, pos: 9
type: A, layer: 1, pos: 6
type: A, layer: 1, pos: 48
type: A, layer: 1, pos: 28
type: A, layer: 1, pos: 38
type: A, layer: 1, pos: 0
type: A, layer: 1, pos: 24
type: A, layer: 1, pos: 41
type: A, layer: 1, pos: 20
type: A, layer: 1, pos: 8
type: A, layer: 1, pos: 26
type: A, layer: 1, pos: 5
type: A, layer: 1, pos: 31
type: A, layer: 1, pos: 45
type: A, layer: 1, pos: 15

Time for candidate selection: 0.06 seconds

### Candidate
type: A, layer: 1, pos: 1

### Candidate
type: A, layer: 1, pos: 39

### Candidate
type: A, layer: 1, pos: 9

### Candidate
type: A, layer: 1, pos: 6

### Candidate
type: A, layer: 1, pos: 48

### Candidate
type: A, layer: 1, pos: 28

### Candidate
type: A, layer: 1, pos: 38

### Candidate
type: A, layer: 1, pos: 0

### Candidate
type: A, layer: 1, pos: 24

### Candidate
type: A, layer: 1, pos: 41

### Candidate
type: A, layer: 1, pos: 20

## Relational analysis of NS_A1_B1_A2_B1_A2_B1_A1
Optimization infeasible because this subproblem isn't reachable.

### Candidate
type: A, layer: 1, pos: 8

## Relational analysis of NS_A1_B1_A2_B1_A2_B1_A1
Optimization infeasible because this subproblem isn't reachable.

### Candidate
type: A, layer: 1, pos: 26

## Relational analysis of NS_A1_B1_A2_B1_A2_B1_A1
Optimization infeasible because this subproblem isn't reachable.

### Candidate
type: A, layer: 1, pos: 5

## Relational analysis of NS_A1_B1_A2_B1_A2_B1_A1

### Relational analysis result of NS_A1_B1_A2_B1_A2_B1_A1
Status: Status.UNKNOWN
Output dim: 0, lower bound: -20.9550818, upper bound: 20.9495722
time: 0.60 seconds

## Relational analysis of NS_A1_B1_A2_B1_A2_B1_A2
Optimization infeasible because this subproblem isn't reachable.

### Candidate
type: A, layer: 1, pos: 31

## Relational analysis of NS_A1_B1_A2_B1_A2_B1_A1
Optimization infeasible because this subproblem isn't reachable.

### Candidate
type: A, layer: 1, pos: 45

## Relational analysis of NS_A1_B1_A2_B1_A2_B1_A1
Optimization infeasible because this subproblem isn't reachable.

### Candidate
type: A, layer: 1, pos: 15

## Relational analysis of NS_A1_B1_A2_B1_A2_B1_A1
Optimization infeasible because this subproblem isn't reachable.

## BFS NS instance: NS_A1_B1_A2_B1_A2_B2

### Backsubstitution after applying NS history:
0: -5.9965081, 16.7290058, -5.9385791, 16.6198044, -22.6163101, 22.6675854
1: -14.8499393, 25.8211803, -14.9495649, 25.3848782, -40.2348175, 40.7707443
2: -10.3930006, 23.3898659, -10.3734665, 23.0961895, -33.4891891, 33.7633324
3: -16.0963974, 28.4984932, -16.0344677, 28.1116180, -44.2080154, 44.5329590
4: -14.8098030, 28.8273373, -14.4779634, 28.5761471, -43.3859406, 43.3052979

Time for backsubstitution: 0.88 seconds

### NS candidates at layer 1
type: A, layer: 1, pos: 1
type: A, layer: 1, pos: 9
type: A, layer: 1, pos: 39
type: A, layer: 1, pos: 6
type: A, layer: 1, pos: 28
type: A, layer: 1, pos: 48
type: A, layer: 1, pos: 38
type: A, layer: 1, pos: 0
type: A, layer: 1, pos: 24
type: A, layer: 1, pos: 41
type: A, layer: 1, pos: 8
type: A, layer: 1, pos: 20
type: A, layer: 1, pos: 26
type: A, layer: 1, pos: 15
type: A, layer: 1, pos: 31
type: A, layer: 1, pos: 5
type: A, layer: 1, pos: 45

Time for candidate selection: 0.06 seconds

### Candidate
type: A, layer: 1, pos: 1

### Candidate
type: A, layer: 1, pos: 9

### Candidate
type: A, layer: 1, pos: 39

### Candidate
type: A, layer: 1, pos: 6

### Candidate
type: A, layer: 1, pos: 28

### Candidate
type: A, layer: 1, pos: 48

### Candidate
type: A, layer: 1, pos: 38

### Candidate
type: A, layer: 1, pos: 0

### Candidate
type: A, layer: 1, pos: 24

### Candidate
type: A, layer: 1, pos: 41

### Candidate
type: A, layer: 1, pos: 8

## Relational analysis of NS_A1_B1_A2_B1_A2_B2_A1
Optimization infeasible because this subproblem isn't reachable.

### Candidate
type: A, layer: 1, pos: 20

## Relational analysis of NS_A1_B1_A2_B1_A2_B2_A1

### Relational analysis result of NS_A1_B1_A2_B1_A2_B2_A1
Status: Status.UNKNOWN
Output dim: 0, lower bound: -20.9551558, upper bound: 20.9505436
time: 0.60 seconds

## Relational analysis of NS_A1_B1_A2_B1_A2_B2_A2

### Relational analysis result of NS_A1_B1_A2_B1_A2_B2_A2
Status: Status.UNKNOWN
Output dim: 0, lower bound: -20.9551558, upper bound: 20.9549741
time: 1.14 seconds

## BFS NS instance: NS_A1_B1_A2_B2_A1_B1

### Backsubstitution after applying NS history:
0: -5.8384013, 16.3657513, -6.1291590, 17.0460529, -22.8844490, 22.4949112
1: -14.4844494, 25.3005962, -15.2738848, 26.1256695, -40.6101151, 40.5744820
2: -10.1135731, 22.8840389, -10.6594505, 23.7330799, -33.8466530, 33.5434875
3: -15.6806517, 27.9001026, -16.4909611, 28.9151707, -44.5958138, 44.3910637
4: -14.3752546, 28.2174873, -15.0633440, 29.2970066, -43.6722603, 43.2808304

Time for backsubstitution: 0.89 seconds

### NS candidates at layer 1
type: A, layer: 1, pos: 1
type: A, layer: 1, pos: 39
type: A, layer: 1, pos: 6
type: A, layer: 1, pos: 48
type: A, layer: 1, pos: 28
type: A, layer: 1, pos: 38
type: A, layer: 1, pos: 0
type: A, layer: 1, pos: 24
type: A, layer: 1, pos: 41
type: A, layer: 1, pos: 37
type: A, layer: 1, pos: 8
type: A, layer: 1, pos: 20
type: A, layer: 1, pos: 26
type: A, layer: 1, pos: 15
type: A, layer: 1, pos: 5
type: A, layer: 1, pos: 45
type: A, layer: 1, pos: 31

Time for candidate selection: 0.06 seconds

### Candidate
type: A, layer: 1, pos: 1

### Candidate
type: A, layer: 1, pos: 39

### Candidate
type: A, layer: 1, pos: 6

## Relational analysis of NS_A1_B1_A2_B2_A1_B1_A1
Optimization infeasible because this subproblem isn't reachable.

### Candidate
type: A, layer: 1, pos: 48

## Relational analysis of NS_A1_B1_A2_B2_A1_B1_A1
Optimization infeasible because this subproblem isn't reachable.

### Candidate
type: A, layer: 1, pos: 28

## Relational analysis of NS_A1_B1_A2_B2_A1_B1_A1
Optimization infeasible because this subproblem isn't reachable.

### Candidate
type: A, layer: 1, pos: 38

## Relational analysis of NS_A1_B1_A2_B2_A1_B1_A1
Optimization infeasible because this subproblem isn't reachable.

### Candidate
type: A, layer: 1, pos: 0

## Relational analysis of NS_A1_B1_A2_B2_A1_B1_A1
Optimization infeasible because this subproblem isn't reachable.

### Candidate
type: A, layer: 1, pos: 24

## Relational analysis of NS_A1_B1_A2_B2_A1_B1_A1
Optimization infeasible because this subproblem isn't reachable.

### Candidate
type: A, layer: 1, pos: 41

## Relational analysis of NS_A1_B1_A2_B2_A1_B1_A1
Optimization infeasible because this subproblem isn't reachable.

### Candidate
type: A, layer: 1, pos: 37

## Relational analysis of NS_A1_B1_A2_B2_A1_B1_A1

### Relational analysis result of NS_A1_B1_A2_B2_A1_B1_A1
Status: Status.UNKNOWN
Output dim: 0, lower bound: -20.9121909, upper bound: 20.9155805
time: 0.60 seconds

## Relational analysis of NS_A1_B1_A2_B2_A1_B1_A2
Optimization infeasible because this subproblem isn't reachable.

### Candidate
type: A, layer: 1, pos: 8

## Relational analysis of NS_A1_B1_A2_B2_A1_B1_A1
Optimization infeasible because this subproblem isn't reachable.

### Candidate
type: A, layer: 1, pos: 20

## Relational analysis of NS_A1_B1_A2_B2_A1_B1_A1

### Relational analysis result of NS_A1_B1_A2_B2_A1_B1_A1
Status: Status.UNKNOWN
Output dim: 0, lower bound: -20.8950918, upper bound: 20.8976329
time: 0.53 seconds

## Relational analysis of NS_A1_B1_A2_B2_A1_B1_A2

### Relational analysis result of NS_A1_B1_A2_B2_A1_B1_A2
Status: Status.UNKNOWN
Output dim: 0, lower bound: -20.8950918, upper bound: 20.9155805
time: 0.62 seconds

## BFS NS instance: NS_A1_B1_A2_B2_A1_B2

### Backsubstitution after applying NS history:
0: -5.8384013, 16.3657513, -6.0668573, 16.8530884, -22.6914864, 22.4326057
1: -14.4844494, 25.3005962, -15.1914082, 25.7236519, -40.2080956, 40.4920006
2: -10.1135731, 22.8840389, -10.5811043, 23.4376564, -33.5512276, 33.4651375
3: -15.6806517, 27.9001026, -16.3516121, 28.5066910, -44.1873283, 44.2517166
4: -14.3752546, 28.2174873, -14.8622160, 28.9597073, -43.3349609, 43.0797043

Time for backsubstitution: 0.89 seconds

### NS candidates at layer 1
type: A, layer: 1, pos: 1
type: A, layer: 1, pos: 39
type: A, layer: 1, pos: 6
type: A, layer: 1, pos: 48
type: A, layer: 1, pos: 28
type: A, layer: 1, pos: 38
type: A, layer: 1, pos: 0
type: A, layer: 1, pos: 24
type: A, layer: 1, pos: 41
type: A, layer: 1, pos: 37
type: A, layer: 1, pos: 8
type: A, layer: 1, pos: 20
type: A, layer: 1, pos: 26
type: A, layer: 1, pos: 15
type: A, layer: 1, pos: 5
type: A, layer: 1, pos: 45
type: A, layer: 1, pos: 31

Time for candidate selection: 0.06 seconds

### Candidate
type: A, layer: 1, pos: 1

### Candidate
type: A, layer: 1, pos: 39

### Candidate
type: A, layer: 1, pos: 6

## Relational analysis of NS_A1_B1_A2_B2_A1_B2_A1
Optimization infeasible because this subproblem isn't reachable.

### Candidate
type: A, layer: 1, pos: 48

## Relational analysis of NS_A1_B1_A2_B2_A1_B2_A1
Optimization infeasible because this subproblem isn't reachable.

### Candidate
type: A, layer: 1, pos: 28

## Relational analysis of NS_A1_B1_A2_B2_A1_B2_A1
Optimization infeasible because this subproblem isn't reachable.

### Candidate
type: A, layer: 1, pos: 38

## Relational analysis of NS_A1_B1_A2_B2_A1_B2_A1
Optimization infeasible because this subproblem isn't reachable.

### Candidate
type: A, layer: 1, pos: 0

## Relational analysis of NS_A1_B1_A2_B2_A1_B2_A1
Optimization infeasible because this subproblem isn't reachable.

### Candidate
type: A, layer: 1, pos: 24

## Relational analysis of NS_A1_B1_A2_B2_A1_B2_A1
Optimization infeasible because this subproblem isn't reachable.

### Candidate
type: A, layer: 1, pos: 41

## Relational analysis of NS_A1_B1_A2_B2_A1_B2_A1
Optimization infeasible because this subproblem isn't reachable.

### Candidate
type: A, layer: 1, pos: 37

## Relational analysis of NS_A1_B1_A2_B2_A1_B2_A1

### Relational analysis result of NS_A1_B1_A2_B2_A1_B2_A1
Status: Status.UNKNOWN
Output dim: 0, lower bound: -20.9121909, upper bound: 20.9248678
time: 1.17 seconds

## Relational analysis of NS_A1_B1_A2_B2_A1_B2_A2
Optimization infeasible because this subproblem isn't reachable.

### Candidate
type: A, layer: 1, pos: 8

## Relational analysis of NS_A1_B1_A2_B2_A1_B2_A1
Optimization infeasible because this subproblem isn't reachable.

### Candidate
type: A, layer: 1, pos: 20

## Relational analysis of NS_A1_B1_A2_B2_A1_B2_A1

### Relational analysis result of NS_A1_B1_A2_B2_A1_B2_A1
Status: Status.UNKNOWN
Output dim: 0, lower bound: -20.8950918, upper bound: 20.9101592
time: 0.65 seconds

## Relational analysis of NS_A1_B1_A2_B2_A1_B2_A2

### Relational analysis result of NS_A1_B1_A2_B2_A1_B2_A2
Status: Status.UNKNOWN
Output dim: 0, lower bound: -20.8950918, upper bound: 20.9248678
time: 0.72 seconds

## BFS NS instance: NS_A1_B1_A2_B2_A2_B1

### Backsubstitution after applying NS history:
0: -5.6924324, 15.9221439, -6.1291590, 17.0460529, -22.7384853, 22.0513000
1: -14.1625080, 24.5419464, -15.2738848, 26.1256695, -40.2881775, 39.8158302
2: -9.8790131, 22.2478924, -10.6594505, 23.7330799, -33.6120911, 32.9073410
3: -15.3027058, 27.1016102, -16.4909611, 28.9151707, -44.2178764, 43.5925674
4: -13.9716702, 27.4440098, -15.0633440, 29.2970066, -43.2686768, 42.5073509

Time for backsubstitution: 0.89 seconds

### NS candidates at layer 1
type: A, layer: 1, pos: 1
type: A, layer: 1, pos: 39
type: A, layer: 1, pos: 6
type: A, layer: 1, pos: 48
type: A, layer: 1, pos: 28
type: A, layer: 1, pos: 38
type: A, layer: 1, pos: 0
type: A, layer: 1, pos: 24
type: A, layer: 1, pos: 41
type: A, layer: 1, pos: 37
type: A, layer: 1, pos: 8
type: A, layer: 1, pos: 20
type: A, layer: 1, pos: 26
type: A, layer: 1, pos: 45
type: A, layer: 1, pos: 5
type: A, layer: 1, pos: 31
type: A, layer: 1, pos: 15

Time for candidate selection: 0.06 seconds

### Candidate
type: A, layer: 1, pos: 1

### Candidate
type: A, layer: 1, pos: 39

### Candidate
type: A, layer: 1, pos: 6

### Candidate
type: A, layer: 1, pos: 48

## Relational analysis of NS_A1_B1_A2_B2_A2_B1_A1
Optimization infeasible because this subproblem isn't reachable.

### Candidate
type: A, layer: 1, pos: 28

## Relational analysis of NS_A1_B1_A2_B2_A2_B1_A1
Optimization infeasible because this subproblem isn't reachable.

### Candidate
type: A, layer: 1, pos: 38

## Relational analysis of NS_A1_B1_A2_B2_A2_B1_A1
Optimization infeasible because this subproblem isn't reachable.

### Candidate
type: A, layer: 1, pos: 0

## Relational analysis of NS_A1_B1_A2_B2_A2_B1_A1
Optimization infeasible because this subproblem isn't reachable.

### Candidate
type: A, layer: 1, pos: 24

## Relational analysis of NS_A1_B1_A2_B2_A2_B1_A1
Optimization infeasible because this subproblem isn't reachable.

### Candidate
type: A, layer: 1, pos: 41

## Relational analysis of NS_A1_B1_A2_B2_A2_B1_A1
Optimization infeasible because this subproblem isn't reachable.

### Candidate
type: A, layer: 1, pos: 37

## Relational analysis of NS_A1_B1_A2_B2_A2_B1_A1

### Relational analysis result of NS_A1_B1_A2_B2_A2_B1_A1
Status: Status.UNKNOWN
Output dim: 0, lower bound: -20.9138254, upper bound: 20.9189541
time: 0.66 seconds

## Relational analysis of NS_A1_B1_A2_B2_A2_B1_A2

### Relational analysis result of NS_A1_B1_A2_B2_A2_B1_A2
Status: Status.UNKNOWN
Output dim: 0, lower bound: -20.9063213, upper bound: 20.9100199
time: 0.75 seconds

## BFS NS instance: NS_A1_B1_A2_B2_A2_B2

### Backsubstitution after applying NS history:
0: -5.6924324, 15.9221439, -6.0668573, 16.8530884, -22.5455208, 21.9889984
1: -14.1625080, 24.5419464, -15.1914082, 25.7236519, -39.8861618, 39.7333527
2: -9.8790131, 22.2478924, -10.5811043, 23.4376564, -33.3166656, 32.8289948
3: -15.3027058, 27.1016102, -16.3516121, 28.5066910, -43.8093910, 43.4532242
4: -13.9716702, 27.4440098, -14.8622160, 28.9597073, -42.9313736, 42.3062248

Time for backsubstitution: 0.90 seconds

### NS candidates at layer 1
type: A, layer: 1, pos: 1
type: A, layer: 1, pos: 39
type: A, layer: 1, pos: 6
type: A, layer: 1, pos: 48
type: A, layer: 1, pos: 28
type: A, layer: 1, pos: 38
type: A, layer: 1, pos: 0
type: A, layer: 1, pos: 24
type: A, layer: 1, pos: 41
type: A, layer: 1, pos: 37
type: A, layer: 1, pos: 8
type: A, layer: 1, pos: 20
type: A, layer: 1, pos: 26
type: A, layer: 1, pos: 45
type: A, layer: 1, pos: 5
type: A, layer: 1, pos: 31
type: A, layer: 1, pos: 15

Time for candidate selection: 0.06 seconds

### Candidate
type: A, layer: 1, pos: 1

### Candidate
type: A, layer: 1, pos: 39

### Candidate
type: A, layer: 1, pos: 6

### Candidate
type: A, layer: 1, pos: 48

## Relational analysis of NS_A1_B1_A2_B2_A2_B2_A1
Optimization infeasible because this subproblem isn't reachable.

### Candidate
type: A, layer: 1, pos: 28

## Relational analysis of NS_A1_B1_A2_B2_A2_B2_A1
Optimization infeasible because this subproblem isn't reachable.

### Candidate
type: A, layer: 1, pos: 38

## Relational analysis of NS_A1_B1_A2_B2_A2_B2_A1
Optimization infeasible because this subproblem isn't reachable.

### Candidate
type: A, layer: 1, pos: 0

## Relational analysis of NS_A1_B1_A2_B2_A2_B2_A1
Optimization infeasible because this subproblem isn't reachable.

### Candidate
type: A, layer: 1, pos: 24

## Relational analysis of NS_A1_B1_A2_B2_A2_B2_A1
Optimization infeasible because this subproblem isn't reachable.

### Candidate
type: A, layer: 1, pos: 41

## Relational analysis of NS_A1_B1_A2_B2_A2_B2_A1
Optimization infeasible because this subproblem isn't reachable.

### Candidate
type: A, layer: 1, pos: 37

## Relational analysis of NS_A1_B1_A2_B2_A2_B2_A1

### Relational analysis result of NS_A1_B1_A2_B2_A2_B2_A1
Status: Status.UNKNOWN
Output dim: 0, lower bound: -20.9138254, upper bound: 20.9435253
time: 0.58 seconds

## Relational analysis of NS_A1_B1_A2_B2_A2_B2_A2

### Relational analysis result of NS_A1_B1_A2_B2_A2_B2_A2
Status: Status.UNKNOWN
Output dim: 0, lower bound: -20.9063213, upper bound: 20.9427230
time: 0.77 seconds

## BFS NS instance: NS_A1_B2_A1_B1_A1_B1

### Backsubstitution after applying NS history:
0: -5.2931418, 14.9397831, -5.6370058, 15.8936968, -21.1868362, 20.5767860
1: -13.1073370, 23.1132984, -14.0999632, 24.5022202, -37.6095505, 37.2132607
2: -9.1489487, 20.9186592, -9.8033056, 22.1821289, -31.3310776, 30.7219658
3: -14.2131996, 25.4802723, -15.1884594, 27.0471077, -41.2603035, 40.6687241
4: -13.0011435, 25.7873745, -13.7740364, 27.4020576, -40.4031944, 39.5614090

Time for backsubstitution: 0.90 seconds

### NS candidates at layer 1
type: A, layer: 1, pos: 1
type: A, layer: 1, pos: 9
type: A, layer: 1, pos: 39
type: A, layer: 1, pos: 6
type: A, layer: 1, pos: 48
type: A, layer: 1, pos: 38
type: A, layer: 1, pos: 28
type: A, layer: 1, pos: 20
type: A, layer: 1, pos: 41
type: A, layer: 1, pos: 24
type: A, layer: 1, pos: 31
type: A, layer: 1, pos: 8
type: A, layer: 1, pos: 37
type: A, layer: 1, pos: 26
type: A, layer: 1, pos: 45
type: A, layer: 1, pos: 5
type: A, layer: 1, pos: 15

Time for candidate selection: 0.06 seconds

### Candidate
type: A, layer: 1, pos: 1

### Candidate
type: A, layer: 1, pos: 9

### Candidate
type: A, layer: 1, pos: 39

### Candidate
type: A, layer: 1, pos: 6

### Candidate
type: A, layer: 1, pos: 48

### Candidate
type: A, layer: 1, pos: 38

### Candidate
type: A, layer: 1, pos: 28

### Candidate
type: A, layer: 1, pos: 20

## Relational analysis of NS_A1_B2_A1_B1_A1_B1_A1

### Relational analysis result of NS_A1_B2_A1_B1_A1_B1_A1
Status: Status.UNKNOWN
Output dim: 0, lower bound: -20.9576311, upper bound: 20.9574376
time: 0.59 seconds

## Relational analysis of NS_A1_B2_A1_B1_A1_B1_A2

### Relational analysis result of NS_A1_B2_A1_B1_A1_B1_A2
Status: Status.UNKNOWN
Output dim: 0, lower bound: -20.9607899, upper bound: 20.9623820
time: 0.68 seconds

## BFS NS instance: NS_A1_B2_A1_B1_A1_B2

### Backsubstitution after applying NS history:
0: -5.2931418, 14.9397831, -5.8964634, 16.5236454, -21.8167877, 20.8362446
1: -13.1073370, 23.1132984, -14.6177778, 25.5334053, -38.6407394, 37.7310753
2: -9.1489487, 20.9186592, -10.2206593, 23.1049309, -32.2538681, 31.1393166
3: -14.2131996, 25.4802723, -15.8367100, 28.1615410, -42.3747368, 41.3169785
4: -13.0011435, 25.7873745, -14.5429897, 28.4831047, -41.4842415, 40.3303642

Time for backsubstitution: 0.90 seconds

### NS candidates at layer 1
type: A, layer: 1, pos: 1
type: A, layer: 1, pos: 9
type: A, layer: 1, pos: 39
type: A, layer: 1, pos: 6
type: A, layer: 1, pos: 48
type: A, layer: 1, pos: 38
type: A, layer: 1, pos: 28
type: A, layer: 1, pos: 20
type: A, layer: 1, pos: 41
type: A, layer: 1, pos: 24
type: A, layer: 1, pos: 31
type: A, layer: 1, pos: 8
type: A, layer: 1, pos: 37
type: A, layer: 1, pos: 26
type: A, layer: 1, pos: 45
type: A, layer: 1, pos: 5
type: A, layer: 1, pos: 15

Time for candidate selection: 0.06 seconds

### Candidate
type: A, layer: 1, pos: 1

### Candidate
type: A, layer: 1, pos: 9

### Candidate
type: A, layer: 1, pos: 39

### Candidate
type: A, layer: 1, pos: 6

### Candidate
type: A, layer: 1, pos: 48

### Candidate
type: A, layer: 1, pos: 38

### Candidate
type: A, layer: 1, pos: 28

### Candidate
type: A, layer: 1, pos: 20

## Relational analysis of NS_A1_B2_A1_B1_A1_B2_A1

### Relational analysis result of NS_A1_B2_A1_B1_A1_B2_A1
Status: Status.UNKNOWN
Output dim: 0, lower bound: -20.9576311, upper bound: 20.9574376
time: 0.64 seconds

## Relational analysis of NS_A1_B2_A1_B1_A1_B2_A2

### Relational analysis result of NS_A1_B2_A1_B1_A1_B2_A2
Status: Status.UNKNOWN
Output dim: 0, lower bound: -20.9607899, upper bound: 20.9623820
time: 0.77 seconds

## BFS NS instance: NS_A1_B2_A1_B1_A2_B1

### Backsubstitution after applying NS history:
0: -5.5402689, 15.6422672, -5.7027106, 16.0667610, -21.6070290, 21.3449783
1: -13.8112717, 24.1865158, -14.2820339, 24.7505398, -38.5618095, 38.4685516
2: -9.6186361, 21.8514957, -9.9260283, 22.4121838, -32.0308189, 31.7775230
3: -14.9117498, 26.6818504, -15.3725481, 27.3299770, -42.2417259, 42.0543976
4: -13.5568142, 26.9629269, -13.9278164, 27.6926327, -41.2494392, 40.8907394

Time for backsubstitution: 0.90 seconds

### NS candidates at layer 1
type: A, layer: 1, pos: 1
type: A, layer: 1, pos: 9
type: A, layer: 1, pos: 39
type: A, layer: 1, pos: 6
type: A, layer: 1, pos: 48
type: A, layer: 1, pos: 28
type: A, layer: 1, pos: 38
type: A, layer: 1, pos: 41
type: A, layer: 1, pos: 20
type: A, layer: 1, pos: 24
type: A, layer: 1, pos: 8
type: A, layer: 1, pos: 31
type: A, layer: 1, pos: 37
type: A, layer: 1, pos: 45
type: A, layer: 1, pos: 15
type: A, layer: 1, pos: 26
type: A, layer: 1, pos: 5

Time for candidate selection: 0.06 seconds

### Candidate
type: A, layer: 1, pos: 1

### Candidate
type: A, layer: 1, pos: 9

### Candidate
type: A, layer: 1, pos: 39

### Candidate
type: A, layer: 1, pos: 6

### Candidate
type: A, layer: 1, pos: 48

### Candidate
type: A, layer: 1, pos: 28

### Candidate
type: A, layer: 1, pos: 38

### Candidate
type: A, layer: 1, pos: 41

### Candidate
type: A, layer: 1, pos: 20

## Relational analysis of NS_A1_B2_A1_B1_A2_B1_A1

### Relational analysis result of NS_A1_B2_A1_B1_A2_B1_A1
Status: Status.UNKNOWN
Output dim: 0, lower bound: -20.9560333, upper bound: 20.9554724
time: 0.59 seconds

## Relational analysis of NS_A1_B2_A1_B1_A2_B1_A2

### Relational analysis result of NS_A1_B2_A1_B1_A2_B1_A2
Status: Status.UNKNOWN
Output dim: 0, lower bound: -20.9604393, upper bound: 20.9612852
time: 0.80 seconds

## BFS NS instance: NS_A1_B2_A1_B1_A2_B2

### Backsubstitution after applying NS history:
0: -5.5402689, 15.6422672, -5.9575162, 16.6820240, -22.2222939, 21.5997829
1: -13.8112717, 24.1865158, -14.7838783, 25.7614155, -39.5726852, 38.9703941
2: -9.6186361, 21.8514957, -10.3334379, 23.3171310, -32.9357605, 32.1849327
3: -14.9117498, 26.6818504, -16.0066166, 28.4209652, -43.3327103, 42.6884689
4: -13.5568142, 26.9629269, -14.6881104, 28.7494240, -42.3062363, 41.6510277

Time for backsubstitution: 0.91 seconds

### NS candidates at layer 1
type: A, layer: 1, pos: 1
type: A, layer: 1, pos: 9
type: A, layer: 1, pos: 39
type: A, layer: 1, pos: 6
type: A, layer: 1, pos: 48
type: A, layer: 1, pos: 28
type: A, layer: 1, pos: 38
type: A, layer: 1, pos: 41
type: A, layer: 1, pos: 20
type: A, layer: 1, pos: 24
type: A, layer: 1, pos: 8
type: A, layer: 1, pos: 31
type: A, layer: 1, pos: 37
type: A, layer: 1, pos: 45
type: A, layer: 1, pos: 15
type: A, layer: 1, pos: 26
type: A, layer: 1, pos: 5

Time for candidate selection: 0.06 seconds

### Candidate
type: A, layer: 1, pos: 1

### Candidate
type: A, layer: 1, pos: 9

### Candidate
type: A, layer: 1, pos: 39

### Candidate
type: A, layer: 1, pos: 6

### Candidate
type: A, layer: 1, pos: 48

### Candidate
type: A, layer: 1, pos: 28

### Candidate
type: A, layer: 1, pos: 38

### Candidate
type: A, layer: 1, pos: 41

### Candidate
type: A, layer: 1, pos: 20

## Relational analysis of NS_A1_B2_A1_B1_A2_B2_A1

### Relational analysis result of NS_A1_B2_A1_B1_A2_B2_A1
Status: Status.UNKNOWN
Output dim: 0, lower bound: -20.9560333, upper bound: 20.9554724
time: 0.65 seconds

## Relational analysis of NS_A1_B2_A1_B1_A2_B2_A2

### Relational analysis result of NS_A1_B2_A1_B1_A2_B2_A2
Status: Status.UNKNOWN
Output dim: 0, lower bound: -20.9604393, upper bound: 20.9612852
time: 0.66 seconds

## BFS NS instance: NS_A1_B2_A1_B2_A1_B1

### Backsubstitution after applying NS history:
0: -5.2931418, 14.9397831, -5.2941074, 14.8130350, -20.1061764, 20.2338848
1: -13.1073370, 23.1132984, -12.9180031, 23.0340939, -36.1414261, 36.0313034
2: -9.1489487, 20.9186592, -9.0955582, 20.7701797, -29.9191284, 30.0142155
3: -14.2131996, 25.4802723, -14.0595989, 25.3568649, -39.5700531, 39.5398712
4: -13.0011435, 25.7873745, -12.9847622, 25.4687901, -38.4699326, 38.7721367

Time for backsubstitution: 0.91 seconds

### NS candidates at layer 1
type: A, layer: 1, pos: 1
type: A, layer: 1, pos: 9
type: A, layer: 1, pos: 39
type: A, layer: 1, pos: 6
type: A, layer: 1, pos: 48
type: A, layer: 1, pos: 38
type: A, layer: 1, pos: 28
type: A, layer: 1, pos: 20
type: A, layer: 1, pos: 41
type: A, layer: 1, pos: 24
type: A, layer: 1, pos: 31
type: A, layer: 1, pos: 8
type: A, layer: 1, pos: 37
type: A, layer: 1, pos: 26
type: A, layer: 1, pos: 45
type: A, layer: 1, pos: 5
type: A, layer: 1, pos: 15

Time for candidate selection: 0.06 seconds

### Candidate
type: A, layer: 1, pos: 1

### Candidate
type: A, layer: 1, pos: 9

### Candidate
type: A, layer: 1, pos: 39

### Candidate
type: A, layer: 1, pos: 6

### Candidate
type: A, layer: 1, pos: 48

### Candidate
type: A, layer: 1, pos: 38

### Candidate
type: A, layer: 1, pos: 28

### Candidate
type: A, layer: 1, pos: 20

## Relational analysis of NS_A1_B2_A1_B2_A1_B1_A1

### Relational analysis result of NS_A1_B2_A1_B2_A1_B1_A1
Status: Status.UNKNOWN
Output dim: 0, lower bound: -20.9492798, upper bound: 20.9500829
time: 0.73 seconds

## Relational analysis of NS_A1_B2_A1_B2_A1_B1_A2

### Relational analysis result of NS_A1_B2_A1_B2_A1_B1_A2
Status: Status.UNKNOWN
Output dim: 0, lower bound: -20.9492798, upper bound: 20.9562324
time: 0.65 seconds

## BFS NS instance: NS_A1_B2_A1_B2_A1_B2

### Backsubstitution after applying NS history:
0: -5.2931418, 14.9397831, -5.4117703, 15.1280680, -20.4212093, 20.3515491
1: -13.1073370, 23.1132984, -13.2163115, 23.5520096, -36.6593475, 36.3296089
2: -9.1489487, 20.9186592, -9.3165588, 21.1963367, -30.3452854, 30.2352180
3: -14.2131996, 25.4802723, -14.3792086, 25.9359150, -40.1491013, 39.8594818
4: -13.0011435, 25.7873745, -13.2758665, 25.9836140, -38.9847527, 39.0632401

Time for backsubstitution: 0.91 seconds

### NS candidates at layer 1
type: A, layer: 1, pos: 1
type: A, layer: 1, pos: 9
type: A, layer: 1, pos: 39
type: A, layer: 1, pos: 6
type: A, layer: 1, pos: 48
type: A, layer: 1, pos: 38
type: A, layer: 1, pos: 28
type: A, layer: 1, pos: 20
type: A, layer: 1, pos: 41
type: A, layer: 1, pos: 24
type: A, layer: 1, pos: 31
type: A, layer: 1, pos: 8
type: A, layer: 1, pos: 37
type: A, layer: 1, pos: 26
type: A, layer: 1, pos: 45
type: A, layer: 1, pos: 5
type: A, layer: 1, pos: 15

Time for candidate selection: 0.06 seconds

### Candidate
type: A, layer: 1, pos: 1

### Candidate
type: A, layer: 1, pos: 9

### Candidate
type: A, layer: 1, pos: 39

### Candidate
type: A, layer: 1, pos: 6

### Candidate
type: A, layer: 1, pos: 48

### Candidate
type: A, layer: 1, pos: 38

### Candidate
type: A, layer: 1, pos: 28

### Candidate
type: A, layer: 1, pos: 20

## Relational analysis of NS_A1_B2_A1_B2_A1_B2_A1

### Relational analysis result of NS_A1_B2_A1_B2_A1_B2_A1
Status: Status.UNKNOWN
Output dim: 0, lower bound: -20.9492798, upper bound: 20.9500829
time: 0.63 seconds

## Relational analysis of NS_A1_B2_A1_B2_A1_B2_A2

### Relational analysis result of NS_A1_B2_A1_B2_A1_B2_A2
Status: Status.UNKNOWN
Output dim: 0, lower bound: -20.9492798, upper bound: 20.9563050
time: 0.60 seconds

## BFS NS instance: NS_A1_B2_A1_B2_A2_B1

### Backsubstitution after applying NS history:
0: -5.5402689, 15.6422672, -5.7361045, 16.0389690, -21.5792389, 21.3783722
1: -13.8112717, 24.1865158, -14.3010998, 24.7190742, -38.5303345, 38.4876175
2: -9.6186361, 21.8514957, -9.9806919, 22.3472767, -31.9659119, 31.8321877
3: -14.9117498, 26.6818504, -15.3529701, 27.3102798, -42.2220268, 42.0348206
4: -13.5568142, 26.9629269, -13.8759937, 27.5195332, -41.0763474, 40.8389091

Time for backsubstitution: 0.92 seconds

### NS candidates at layer 1
type: A, layer: 1, pos: 1
type: A, layer: 1, pos: 9
type: A, layer: 1, pos: 39
type: A, layer: 1, pos: 6
type: A, layer: 1, pos: 48
type: A, layer: 1, pos: 28
type: A, layer: 1, pos: 38
type: A, layer: 1, pos: 41
type: A, layer: 1, pos: 20
type: A, layer: 1, pos: 24
type: A, layer: 1, pos: 8
type: A, layer: 1, pos: 31
type: A, layer: 1, pos: 37
type: A, layer: 1, pos: 45
type: A, layer: 1, pos: 15
type: A, layer: 1, pos: 26
type: A, layer: 1, pos: 5

Time for candidate selection: 0.06 seconds

### Candidate
type: A, layer: 1, pos: 1

### Candidate
type: A, layer: 1, pos: 9

### Candidate
type: A, layer: 1, pos: 39

### Candidate
type: A, layer: 1, pos: 6

### Candidate
type: A, layer: 1, pos: 48

### Candidate
type: A, layer: 1, pos: 28

### Candidate
type: A, layer: 1, pos: 38

### Candidate
type: A, layer: 1, pos: 41

### Candidate
type: A, layer: 1, pos: 20

## Relational analysis of NS_A1_B2_A1_B2_A2_B1_A1

### Relational analysis result of NS_A1_B2_A1_B2_A2_B1_A1
Status: Status.UNKNOWN
Output dim: 0, lower bound: -20.9485935, upper bound: 20.9490286
time: 0.65 seconds

## Relational analysis of NS_A1_B2_A1_B2_A2_B1_A2

### Relational analysis result of NS_A1_B2_A1_B2_A2_B1_A2
Status: Status.UNKNOWN
Output dim: 0, lower bound: -20.9537253, upper bound: 20.9553052
time: 0.71 seconds

## BFS NS instance: NS_A1_B2_A1_B2_A2_B2

### Backsubstitution after applying NS history:
0: -5.5402689, 15.6422672, -5.8837357, 16.3726349, -21.9129028, 21.5260029
1: -13.8112717, 24.1865158, -14.5101585, 25.3265743, -39.1378403, 38.6966743
2: -9.6186361, 21.8514957, -10.1916351, 22.8688374, -32.4874687, 32.0431290
3: -14.9117498, 26.6818504, -15.6975384, 27.9520607, -42.8638115, 42.3793869
4: -13.5568142, 26.9629269, -14.3947430, 28.0928898, -41.6497040, 41.3576622

Time for backsubstitution: 0.92 seconds

### NS candidates at layer 1
type: A, layer: 1, pos: 1
type: A, layer: 1, pos: 9
type: A, layer: 1, pos: 39
type: A, layer: 1, pos: 6
type: A, layer: 1, pos: 48
type: A, layer: 1, pos: 28
type: A, layer: 1, pos: 38
type: A, layer: 1, pos: 41
type: A, layer: 1, pos: 20
type: A, layer: 1, pos: 24
type: A, layer: 1, pos: 8
type: A, layer: 1, pos: 31
type: A, layer: 1, pos: 37
type: A, layer: 1, pos: 45
type: A, layer: 1, pos: 15
type: A, layer: 1, pos: 26
type: A, layer: 1, pos: 5

Time for candidate selection: 0.06 seconds

### Candidate
type: A, layer: 1, pos: 1

### Candidate
type: A, layer: 1, pos: 9

### Candidate
type: A, layer: 1, pos: 39

### Candidate
type: A, layer: 1, pos: 6

### Candidate
type: A, layer: 1, pos: 48

### Candidate
type: A, layer: 1, pos: 28

### Candidate
type: A, layer: 1, pos: 38

### Candidate
type: A, layer: 1, pos: 41

### Candidate
type: A, layer: 1, pos: 20

## Relational analysis of NS_A1_B2_A1_B2_A2_B2_A1
Optimization infeasible because this subproblem isn't reachable.

### Candidate
type: A, layer: 1, pos: 24

## Relational analysis of NS_A1_B2_A1_B2_A2_B2_A1
Optimization infeasible because this subproblem isn't reachable.

### Candidate
type: A, layer: 1, pos: 8

## Relational analysis of NS_A1_B2_A1_B2_A2_B2_A1
Optimization infeasible because this subproblem isn't reachable.

### Candidate
type: A, layer: 1, pos: 31

## Relational analysis of NS_A1_B2_A1_B2_A2_B2_A1

### Relational analysis result of NS_A1_B2_A1_B2_A2_B2_A1
Status: Status.UNKNOWN
Output dim: 0, lower bound: -20.9391105, upper bound: 20.9414149
time: 0.63 seconds

## Relational analysis of NS_A1_B2_A1_B2_A2_B2_A2
Optimization infeasible because this subproblem isn't reachable.

### Candidate
type: A, layer: 1, pos: 37

## Relational analysis of NS_A1_B2_A1_B2_A2_B2_A1

### Relational analysis result of NS_A1_B2_A1_B2_A2_B2_A1
Status: Status.UNKNOWN
Output dim: 0, lower bound: -20.9526143, upper bound: 20.9541293
time: 0.58 seconds

## Relational analysis of NS_A1_B2_A1_B2_A2_B2_A2

### Relational analysis result of NS_A1_B2_A1_B2_A2_B2_A2
Status: Status.UNKNOWN
Output dim: 0, lower bound: -20.9524777, upper bound: 20.9544091
time: 0.62 seconds

## BFS NS instance: NS_A1_B2_A2_B1_A1_B1

### Backsubstitution after applying NS history:
0: -5.9519348, 16.6478176, -5.6711140, 15.9823341, -21.9342690, 22.3189316
1: -14.8047256, 25.6933746, -14.2006493, 24.6264076, -39.4311333, 39.8940239
2: -10.3367424, 23.2725792, -9.8695850, 22.2990055, -32.6357498, 33.1421585
3: -16.0035057, 28.3608398, -15.2870331, 27.1922836, -43.1957893, 43.6478729
4: -14.6475315, 28.7021637, -13.8500452, 27.5519333, -42.1994553, 42.5522079

Time for backsubstitution: 0.92 seconds

### NS candidates at layer 1
type: A, layer: 1, pos: 1
type: A, layer: 1, pos: 9
type: A, layer: 1, pos: 6
type: A, layer: 1, pos: 28
type: A, layer: 1, pos: 48
type: A, layer: 1, pos: 38
type: A, layer: 1, pos: 0
type: A, layer: 1, pos: 24
type: A, layer: 1, pos: 41
type: A, layer: 1, pos: 37
type: A, layer: 1, pos: 20
type: A, layer: 1, pos: 8
type: A, layer: 1, pos: 31
type: A, layer: 1, pos: 26
type: A, layer: 1, pos: 5
type: A, layer: 1, pos: 15
type: A, layer: 1, pos: 45

Time for candidate selection: 0.06 seconds

### Candidate
type: A, layer: 1, pos: 1

### Candidate
type: A, layer: 1, pos: 9

### Candidate
type: A, layer: 1, pos: 6

## Relational analysis of NS_A1_B2_A2_B1_A1_B1_A1
Optimization infeasible because this subproblem isn't reachable.

### Candidate
type: A, layer: 1, pos: 28

## Relational analysis of NS_A1_B2_A2_B1_A1_B1_A1
Optimization infeasible because this subproblem isn't reachable.

### Candidate
type: A, layer: 1, pos: 48

## Relational analysis of NS_A1_B2_A2_B1_A1_B1_A1
Optimization infeasible because this subproblem isn't reachable.

### Candidate
type: A, layer: 1, pos: 38

## Relational analysis of NS_A1_B2_A2_B1_A1_B1_A1
Optimization infeasible because this subproblem isn't reachable.

### Candidate
type: A, layer: 1, pos: 0

## Relational analysis of NS_A1_B2_A2_B1_A1_B1_A1

### Relational analysis result of NS_A1_B2_A2_B1_A1_B1_A1
Status: Status.UNKNOWN
Output dim: 0, lower bound: -20.9576419, upper bound: 20.9574161
time: 0.81 seconds

## Relational analysis of NS_A1_B2_A2_B1_A1_B1_A2

### Relational analysis result of NS_A1_B2_A2_B1_A1_B1_A2
Status: Status.UNKNOWN
Output dim: 0, lower bound: -20.9522927, upper bound: 20.9519686
time: 0.60 seconds

## BFS NS instance: NS_A1_B2_A2_B1_A1_B2

### Backsubstitution after applying NS history:
0: -5.9519348, 16.6478176, -5.9259515, 16.5996265, -22.5515614, 22.5737686
1: -14.8047256, 25.6933746, -14.7052040, 25.6382656, -40.4429932, 40.3985748
2: -10.3367424, 23.2725792, -10.2782459, 23.2060642, -33.5428085, 33.5508194
3: -16.0035057, 28.3608398, -15.9218321, 28.2842026, -44.2876968, 44.2826729
4: -14.6475315, 28.7021637, -14.6089115, 28.6121178, -43.2596436, 43.3110695

Time for backsubstitution: 0.93 seconds

### NS candidates at layer 1
type: A, layer: 1, pos: 1
type: A, layer: 1, pos: 9
type: A, layer: 1, pos: 6
type: A, layer: 1, pos: 28
type: A, layer: 1, pos: 48
type: A, layer: 1, pos: 38
type: A, layer: 1, pos: 0
type: A, layer: 1, pos: 24
type: A, layer: 1, pos: 41
type: A, layer: 1, pos: 37
type: A, layer: 1, pos: 20
type: A, layer: 1, pos: 8
type: A, layer: 1, pos: 31
type: A, layer: 1, pos: 26
type: A, layer: 1, pos: 5
type: A, layer: 1, pos: 15
type: A, layer: 1, pos: 45

Time for candidate selection: 0.06 seconds

### Candidate
type: A, layer: 1, pos: 1

### Candidate
type: A, layer: 1, pos: 9

### Candidate
type: A, layer: 1, pos: 6

## Relational analysis of NS_A1_B2_A2_B1_A1_B2_A1
Optimization infeasible because this subproblem isn't reachable.

### Candidate
type: A, layer: 1, pos: 28

## Relational analysis of NS_A1_B2_A2_B1_A1_B2_A1
Optimization infeasible because this subproblem isn't reachable.

### Candidate
type: A, layer: 1, pos: 48

## Relational analysis of NS_A1_B2_A2_B1_A1_B2_A1
Optimization infeasible because this subproblem isn't reachable.

### Candidate
type: A, layer: 1, pos: 38

## Relational analysis of NS_A1_B2_A2_B1_A1_B2_A1
Optimization infeasible because this subproblem isn't reachable.

### Candidate
type: A, layer: 1, pos: 0

## Relational analysis of NS_A1_B2_A2_B1_A1_B2_A1
Optimization infeasible because this subproblem isn't reachable.

### Candidate
type: A, layer: 1, pos: 24

## Relational analysis of NS_A1_B2_A2_B1_A1_B2_A1

### Relational analysis result of NS_A1_B2_A2_B1_A1_B2_A1
Status: Status.UNKNOWN
Output dim: 0, lower bound: -20.9604218, upper bound: 20.9609168
time: 0.91 seconds

## Relational analysis of NS_A1_B2_A2_B1_A1_B2_A2

### Relational analysis result of NS_A1_B2_A2_B1_A1_B2_A2
Status: Status.UNKNOWN
Output dim: 0, lower bound: -20.9572772, upper bound: 20.9563436
time: 1.04 seconds

## BFS NS instance: NS_A1_B2_A2_B1_A2_B1

### Backsubstitution after applying NS history:
0: -7.0105648, 19.5966530, -5.6047597, 15.8112345, -22.8217926, 25.2014122
1: -17.6516457, 30.0256901, -14.0175161, 24.3764324, -42.0280762, 44.0432053
2: -12.2167578, 27.2167797, -9.7476950, 22.0709076, -34.2876625, 36.9644737
3: -18.8770370, 33.1330910, -15.1046944, 26.9131718, -45.7902069, 48.2377853
4: -17.0309677, 33.6710854, -13.6979666, 27.2654343, -44.2963905, 47.3690529

Time for backsubstitution: 0.93 seconds

### NS candidates at layer 1
type: A, layer: 1, pos: 1
type: A, layer: 1, pos: 9
type: A, layer: 1, pos: 6
type: A, layer: 1, pos: 28
type: A, layer: 1, pos: 48
type: A, layer: 1, pos: 38
type: A, layer: 1, pos: 24
type: A, layer: 1, pos: 0
type: A, layer: 1, pos: 8
type: A, layer: 1, pos: 41
type: A, layer: 1, pos: 37
type: A, layer: 1, pos: 15
type: A, layer: 1, pos: 26
type: A, layer: 1, pos: 5
type: A, layer: 1, pos: 20
type: A, layer: 1, pos: 31
type: A, layer: 1, pos: 45

Time for candidate selection: 0.07 seconds

### Candidate
type: A, layer: 1, pos: 1

### Candidate
type: A, layer: 1, pos: 9

### Candidate
type: A, layer: 1, pos: 6

## Relational analysis of NS_A1_B2_A2_B1_A2_B1_A1
Optimization infeasible because this subproblem isn't reachable.

### Candidate
type: A, layer: 1, pos: 28

## Relational analysis of NS_A1_B2_A2_B1_A2_B1_A1
Optimization infeasible because this subproblem isn't reachable.

### Candidate
type: A, layer: 1, pos: 48

## Relational analysis of NS_A1_B2_A2_B1_A2_B1_A1
Optimization infeasible because this subproblem isn't reachable.

### Candidate
type: A, layer: 1, pos: 38

## Relational analysis of NS_A1_B2_A2_B1_A2_B1_A1
Optimization infeasible because this subproblem isn't reachable.

### Candidate
type: A, layer: 1, pos: 24

## Relational analysis of NS_A1_B2_A2_B1_A2_B1_A1

### Relational analysis result of NS_A1_B2_A2_B1_A2_B1_A1
Status: Status.UNKNOWN
Output dim: 0, lower bound: -20.9604218, upper bound: 20.9608962
time: 0.59 seconds

## Relational analysis of NS_A1_B2_A2_B1_A2_B1_A2

### Relational analysis result of NS_A1_B2_A2_B1_A2_B1_A2
Status: Status.UNKNOWN
Output dim: 0, lower bound: -20.9572772, upper bound: 20.9608789
time: 1.12 seconds

## BFS NS instance: NS_A1_B2_A2_B1_A2_B2

### Backsubstitution after applying NS history:
0: -7.0105648, 19.5966530, -5.8502169, 16.4093361, -23.4198933, 25.4468689
1: -17.6516457, 30.0256901, -14.5010986, 25.3585262, -43.0101700, 44.5267830
2: -12.2167578, 27.2167797, -10.1413927, 22.9502277, -35.1669846, 37.3581696
3: -18.8770370, 33.1330910, -15.7145090, 27.9702492, -46.8472824, 48.8475914
4: -17.0309677, 33.6710854, -14.4324970, 28.2923603, -45.3233185, 48.1035843

Time for backsubstitution: 0.93 seconds

### NS candidates at layer 1
type: A, layer: 1, pos: 1
type: A, layer: 1, pos: 9
type: A, layer: 1, pos: 6
type: A, layer: 1, pos: 28
type: A, layer: 1, pos: 48
type: A, layer: 1, pos: 38
type: A, layer: 1, pos: 24
type: A, layer: 1, pos: 0
type: A, layer: 1, pos: 8
type: A, layer: 1, pos: 41
type: A, layer: 1, pos: 37
type: A, layer: 1, pos: 15
type: A, layer: 1, pos: 26
type: A, layer: 1, pos: 5
type: A, layer: 1, pos: 20
type: A, layer: 1, pos: 31
type: A, layer: 1, pos: 45

Time for candidate selection: 0.06 seconds

### Candidate
type: A, layer: 1, pos: 1

### Candidate
type: A, layer: 1, pos: 9

### Candidate
type: A, layer: 1, pos: 6

## Relational analysis of NS_A1_B2_A2_B1_A2_B2_A1
Optimization infeasible because this subproblem isn't reachable.

### Candidate
type: A, layer: 1, pos: 28

## Relational analysis of NS_A1_B2_A2_B1_A2_B2_A1
Optimization infeasible because this subproblem isn't reachable.

### Candidate
type: A, layer: 1, pos: 48

## Relational analysis of NS_A1_B2_A2_B1_A2_B2_A1
Optimization infeasible because this subproblem isn't reachable.

### Candidate
type: A, layer: 1, pos: 38

## Relational analysis of NS_A1_B2_A2_B1_A2_B2_A1
Optimization infeasible because this subproblem isn't reachable.

### Candidate
type: A, layer: 1, pos: 24

## Relational analysis of NS_A1_B2_A2_B1_A2_B2_A1

### Relational analysis result of NS_A1_B2_A2_B1_A2_B2_A1
Status: Status.UNKNOWN
Output dim: 0, lower bound: -20.9606378, upper bound: 20.9608962
time: 0.58 seconds

## Relational analysis of NS_A1_B2_A2_B1_A2_B2_A2

### Relational analysis result of NS_A1_B2_A2_B1_A2_B2_A2
Status: Status.UNKNOWN
Output dim: 0, lower bound: -20.9572772, upper bound: 20.9608789
time: 0.68 seconds

## BFS NS instance: NS_A1_B2_A2_B2_A1_B1

### Backsubstitution after applying NS history:
0: -6.1207747, 17.0969448, -5.7361045, 16.0389690, -22.1597443, 22.8330498
1: -15.2238340, 26.3712082, -14.3010998, 24.7190742, -39.9429016, 40.6723099
2: -10.6242371, 23.8737068, -9.9806919, 22.3472767, -32.9715118, 33.8543968
3: -16.4577141, 29.1074352, -15.3529701, 27.3102798, -43.7679901, 44.4603920
4: -15.0668459, 29.4489193, -13.8759937, 27.5195332, -42.5863800, 43.3249130

Time for backsubstitution: 0.94 seconds

### NS candidates at layer 1
type: A, layer: 1, pos: 1
type: A, layer: 1, pos: 39
type: A, layer: 1, pos: 6
type: A, layer: 1, pos: 28
type: A, layer: 1, pos: 38
type: A, layer: 1, pos: 48
type: A, layer: 1, pos: 0
type: A, layer: 1, pos: 24
type: A, layer: 1, pos: 41
type: A, layer: 1, pos: 20
type: A, layer: 1, pos: 37
type: A, layer: 1, pos: 8
type: A, layer: 1, pos: 31
type: A, layer: 1, pos: 26
type: A, layer: 1, pos: 15
type: A, layer: 1, pos: 45
type: A, layer: 1, pos: 5

Time for candidate selection: 0.06 seconds

### Candidate
type: A, layer: 1, pos: 1

### Candidate
type: A, layer: 1, pos: 39

## Relational analysis of NS_A1_B2_A2_B2_A1_B1_A1

### Relational analysis result of NS_A1_B2_A2_B2_A1_B1_A1
Status: Status.UNKNOWN
Output dim: 0, lower bound: -20.9427099, upper bound: 20.9393656
time: 0.64 seconds

## Relational analysis of NS_A1_B2_A2_B2_A1_B1_A2

### Relational analysis result of NS_A1_B2_A2_B2_A1_B1_A2
Status: Status.UNKNOWN
Output dim: 0, lower bound: -20.9440769, upper bound: 20.9409453
time: 0.74 seconds

## BFS NS instance: NS_A1_B2_A2_B2_A1_B2

### Backsubstitution after applying NS history:
0: -6.1207747, 17.0969448, -5.9250507, 16.4725361, -22.5933075, 23.0219936
1: -15.2238340, 26.3712082, -14.6119261, 25.4722958, -40.6961250, 40.9831314
2: -10.6242371, 23.8737068, -10.2639380, 23.0088158, -33.6330490, 34.1376457
3: -16.4577141, 29.1074352, -15.8108330, 28.1234016, -44.5811081, 44.9182663
4: -15.0668459, 29.4489193, -14.4969978, 28.2606392, -43.3274803, 43.9459152

Time for backsubstitution: 0.94 seconds

### NS candidates at layer 1
type: A, layer: 1, pos: 1
type: A, layer: 1, pos: 39
type: A, layer: 1, pos: 6
type: A, layer: 1, pos: 28
type: A, layer: 1, pos: 38
type: A, layer: 1, pos: 48
type: A, layer: 1, pos: 0
type: A, layer: 1, pos: 24
type: A, layer: 1, pos: 41
type: A, layer: 1, pos: 20
type: A, layer: 1, pos: 37
type: A, layer: 1, pos: 8
type: A, layer: 1, pos: 31
type: A, layer: 1, pos: 26
type: A, layer: 1, pos: 15
type: A, layer: 1, pos: 45
type: A, layer: 1, pos: 5

Time for candidate selection: 0.06 seconds

### Candidate
type: A, layer: 1, pos: 1

### Candidate
type: A, layer: 1, pos: 39

## Relational analysis of NS_A1_B2_A2_B2_A1_B2_A1
Optimization infeasible because this subproblem isn't reachable.

### Candidate
type: A, layer: 1, pos: 6

## Relational analysis of NS_A1_B2_A2_B2_A1_B2_A1
Optimization infeasible because this subproblem isn't reachable.

### Candidate
type: A, layer: 1, pos: 28

## Relational analysis of NS_A1_B2_A2_B2_A1_B2_A1
Optimization infeasible because this subproblem isn't reachable.

### Candidate
type: A, layer: 1, pos: 38

## Relational analysis of NS_A1_B2_A2_B2_A1_B2_A1
Optimization infeasible because this subproblem isn't reachable.

### Candidate
type: A, layer: 1, pos: 48

## Relational analysis of NS_A1_B2_A2_B2_A1_B2_A1
Optimization infeasible because this subproblem isn't reachable.

### Candidate
type: A, layer: 1, pos: 0

## Relational analysis of NS_A1_B2_A2_B2_A1_B2_A1

### Relational analysis result of NS_A1_B2_A2_B2_A1_B2_A1
Status: Status.UNKNOWN
Output dim: 0, lower bound: -20.9402842, upper bound: 20.9344529
time: 0.67 seconds

## Relational analysis of NS_A1_B2_A2_B2_A1_B2_A2

### Relational analysis result of NS_A1_B2_A2_B2_A1_B2_A2
Status: Status.UNKNOWN
Output dim: 0, lower bound: -20.9439913, upper bound: 20.9409624
time: 0.70 seconds

## BFS NS instance: NS_A1_B2_A2_B2_A2_B1

### Backsubstitution after applying NS history:
0: -5.9430337, 16.5669880, -5.6846595, 15.9016066, -21.8446407, 22.2516479
1: -14.8138142, 25.4860477, -14.1710567, 24.5143509, -39.3281631, 39.6571045
2: -10.3298597, 23.1185207, -9.8897781, 22.1604786, -32.4903374, 33.0083008
3: -15.9877548, 28.1648884, -15.2150955, 27.0815392, -43.0692940, 43.3799820
4: -14.5869484, 28.5262318, -13.7497787, 27.2896309, -41.8765755, 42.2760086

Time for backsubstitution: 0.94 seconds

### NS candidates at layer 1
type: A, layer: 1, pos: 1
type: A, layer: 1, pos: 39
type: A, layer: 1, pos: 6
type: A, layer: 1, pos: 28
type: A, layer: 1, pos: 38
type: A, layer: 1, pos: 48
type: A, layer: 1, pos: 0
type: A, layer: 1, pos: 24
type: A, layer: 1, pos: 41
type: A, layer: 1, pos: 20
type: A, layer: 1, pos: 37
type: A, layer: 1, pos: 31
type: A, layer: 1, pos: 8
type: A, layer: 1, pos: 26
type: A, layer: 1, pos: 15
type: A, layer: 1, pos: 5
type: A, layer: 1, pos: 45

Time for candidate selection: 0.07 seconds

### Candidate
type: A, layer: 1, pos: 1

### Candidate
type: A, layer: 1, pos: 39

## Relational analysis of NS_A1_B2_A2_B2_A2_B1_A1

### Relational analysis result of NS_A1_B2_A2_B2_A2_B1_A1
Status: Status.UNKNOWN
Output dim: 0, lower bound: -20.9498795, upper bound: 20.9501715
time: 0.68 seconds

## Relational analysis of NS_A1_B2_A2_B2_A2_B1_A2

### Relational analysis result of NS_A1_B2_A2_B2_A2_B1_A2
Status: Status.UNKNOWN
Output dim: 0, lower bound: -20.9504496, upper bound: 20.9508051
time: 1.19 seconds

## BFS NS instance: NS_A1_B2_A2_B2_A2_B2

### Backsubstitution after applying NS history:
0: -5.9430337, 16.5669880, -5.8652964, 16.3187618, -22.2617950, 22.4322834
1: -14.8138142, 25.4860477, -14.4646626, 25.2431335, -40.0569458, 39.9507103
2: -10.3298597, 23.1185207, -10.1599331, 22.7990417, -33.1289024, 33.2784538
3: -15.9877548, 28.1648884, -15.6518364, 27.8670101, -43.8547630, 43.8167267
4: -14.5869484, 28.5262318, -14.3493433, 28.0047264, -42.5916748, 42.8755722

Time for backsubstitution: 0.94 seconds

### NS candidates at layer 1
type: A, layer: 1, pos: 1
type: A, layer: 1, pos: 39
type: A, layer: 1, pos: 6
type: A, layer: 1, pos: 28
type: A, layer: 1, pos: 38
type: A, layer: 1, pos: 48
type: A, layer: 1, pos: 0
type: A, layer: 1, pos: 24
type: A, layer: 1, pos: 41
type: A, layer: 1, pos: 20
type: A, layer: 1, pos: 37
type: A, layer: 1, pos: 31
type: A, layer: 1, pos: 8
type: A, layer: 1, pos: 26
type: A, layer: 1, pos: 15
type: A, layer: 1, pos: 5
type: A, layer: 1, pos: 45

Time for candidate selection: 0.07 seconds

### Candidate
type: A, layer: 1, pos: 1

### Candidate
type: A, layer: 1, pos: 39

## Relational analysis of NS_A1_B2_A2_B2_A2_B2_A1

### Relational analysis result of NS_A1_B2_A2_B2_A2_B2_A1
Status: Status.UNKNOWN
Output dim: 0, lower bound: -20.9498795, upper bound: 20.9501715
time: 0.67 seconds

## Relational analysis of NS_A1_B2_A2_B2_A2_B2_A2

### Relational analysis result of NS_A1_B2_A2_B2_A2_B2_A2
Status: Status.UNKNOWN
Output dim: 0, lower bound: -20.9504496, upper bound: 20.9508051
time: 1.06 seconds

## BFS NS instance: NS_A2_B1_A1_B1_A1_B1

### Backsubstitution after applying NS history:
0: -7.2898707, 19.9250507, -5.5521789, 15.6704454, -22.9603138, 25.4772243
1: -18.5570850, 30.1894703, -13.8982563, 24.1718674, -42.7289505, 44.0877266
2: -12.7783670, 27.4419422, -9.6564922, 21.8720379, -34.6504059, 37.0984344
3: -19.6743641, 33.5284195, -14.9623499, 26.6824760, -46.3568382, 48.4907684
4: -17.4955368, 33.9598045, -13.5478468, 27.0225201, -44.5180588, 47.5076523

Time for backsubstitution: 0.95 seconds

### NS candidates at layer 1
type: A, layer: 1, pos: 9
type: A, layer: 1, pos: 6
type: A, layer: 1, pos: 38
type: A, layer: 1, pos: 48
type: A, layer: 1, pos: 28
type: A, layer: 1, pos: 8
type: A, layer: 1, pos: 0
type: A, layer: 1, pos: 37
type: A, layer: 1, pos: 41
type: A, layer: 1, pos: 20
type: A, layer: 1, pos: 39
type: A, layer: 1, pos: 31
type: A, layer: 1, pos: 15
type: A, layer: 1, pos: 5
type: A, layer: 1, pos: 24
type: A, layer: 1, pos: 26
type: A, layer: 1, pos: 45

Time for candidate selection: 0.07 seconds

### Candidate
type: A, layer: 1, pos: 9

### Candidate
type: A, layer: 1, pos: 6

### Candidate
type: A, layer: 1, pos: 38

### Candidate
type: A, layer: 1, pos: 48

### Candidate
type: A, layer: 1, pos: 28

### Candidate
type: A, layer: 1, pos: 8

### Candidate
type: A, layer: 1, pos: 0

### Candidate
type: A, layer: 1, pos: 37

## Relational analysis of NS_A2_B1_A1_B1_A1_B1_A1
Optimization infeasible because this subproblem isn't reachable.

### Candidate
type: A, layer: 1, pos: 41

### Candidate
type: A, layer: 1, pos: 20

## Relational analysis of NS_A2_B1_A1_B1_A1_B1_A1
Optimization infeasible because this subproblem isn't reachable.

### Candidate
type: A, layer: 1, pos: 39

### Candidate
type: A, layer: 1, pos: 31

## Relational analysis of NS_A2_B1_A1_B1_A1_B1_A1
Optimization infeasible because this subproblem isn't reachable.

### Candidate
type: A, layer: 1, pos: 15

## Relational analysis of NS_A2_B1_A1_B1_A1_B1_A1
Optimization infeasible because this subproblem isn't reachable.

### Candidate
type: A, layer: 1, pos: 5

## Relational analysis of NS_A2_B1_A1_B1_A1_B1_A1
Optimization infeasible because this subproblem isn't reachable.

### Candidate
type: A, layer: 1, pos: 24

### Candidate
type: A, layer: 1, pos: 26

## Relational analysis of NS_A2_B1_A1_B1_A1_B1_A1
Optimization infeasible because this subproblem isn't reachable.

### Candidate
type: A, layer: 1, pos: 45

## Relational analysis of NS_A2_B1_A1_B1_A1_B1_A1
Optimization infeasible because this subproblem isn't reachable.

## BFS NS instance: NS_A2_B1_A1_B1_A1_B2

### Backsubstitution after applying NS history:
0: -7.2051101, 19.6986122, -5.7272520, 16.0773335, -23.2824440, 25.4258652
1: -18.3306866, 29.8652592, -14.3067293, 24.7620640, -43.0927467, 44.1719818
2: -12.6249018, 27.1386833, -9.9589520, 22.4359322, -35.0608330, 37.0976334
3: -19.4402695, 33.1619530, -15.4190779, 27.3476086, -46.7878799, 48.5810318
4: -17.2928104, 33.5811272, -14.0162172, 27.7058773, -44.9986839, 47.5973434

Time for backsubstitution: 0.95 seconds

### NS candidates at layer 1
type: A, layer: 1, pos: 9
type: A, layer: 1, pos: 6
type: A, layer: 1, pos: 38
type: A, layer: 1, pos: 48
type: A, layer: 1, pos: 28
type: A, layer: 1, pos: 0
type: A, layer: 1, pos: 8
type: A, layer: 1, pos: 41
type: A, layer: 1, pos: 37
type: A, layer: 1, pos: 20
type: A, layer: 1, pos: 39
type: A, layer: 1, pos: 31
type: A, layer: 1, pos: 15
type: A, layer: 1, pos: 24
type: A, layer: 1, pos: 5
type: A, layer: 1, pos: 45
type: A, layer: 1, pos: 26

Time for candidate selection: 0.07 seconds

### Candidate
type: A, layer: 1, pos: 9

### Candidate
type: A, layer: 1, pos: 6

### Candidate
type: A, layer: 1, pos: 38

### Candidate
type: A, layer: 1, pos: 48

### Candidate
type: A, layer: 1, pos: 28

### Candidate
type: A, layer: 1, pos: 0

### Candidate
type: A, layer: 1, pos: 8

### Candidate
type: A, layer: 1, pos: 41

### Candidate
type: A, layer: 1, pos: 37

## Relational analysis of NS_A2_B1_A1_B1_A1_B2_A1
Optimization infeasible because this subproblem isn't reachable.

### Candidate
type: A, layer: 1, pos: 20

## Relational analysis of NS_A2_B1_A1_B1_A1_B2_A1

### Relational analysis result of NS_A2_B1_A1_B1_A1_B2_A1
Status: Status.UNKNOWN
Output dim: 0, lower bound: -20.9548567, upper bound: 20.9552630
time: 0.60 seconds

## Relational analysis of NS_A2_B1_A1_B1_A1_B2_A2

### Relational analysis result of NS_A2_B1_A1_B1_A1_B2_A2
Status: Status.UNKNOWN
Output dim: 0, lower bound: -20.9640737, upper bound: 20.9649475
time: 0.84 seconds

## BFS NS instance: NS_A2_B1_A1_B1_A2_B1

### Backsubstitution after applying NS history:
0: -7.2532120, 19.7014389, -5.5521789, 15.6704454, -22.9236565, 25.2536163
1: -18.2541428, 29.9893970, -13.8982563, 24.1718674, -42.4260063, 43.8876534
2: -12.6468019, 27.2146072, -9.6564922, 21.8720379, -34.5188408, 36.8710976
3: -19.4920635, 33.2924194, -14.9623499, 26.6824760, -46.1745300, 48.2547684
4: -17.5639420, 33.5777855, -13.5478468, 27.0225201, -44.5864639, 47.1256332

Time for backsubstitution: 0.95 seconds

### NS candidates at layer 1
type: A, layer: 1, pos: 9
type: A, layer: 1, pos: 6
type: A, layer: 1, pos: 38
type: A, layer: 1, pos: 48
type: A, layer: 1, pos: 0
type: A, layer: 1, pos: 28
type: A, layer: 1, pos: 39
type: A, layer: 1, pos: 37
type: A, layer: 1, pos: 41
type: A, layer: 1, pos: 8
type: A, layer: 1, pos: 15
type: A, layer: 1, pos: 24
type: A, layer: 1, pos: 20
type: A, layer: 1, pos: 31
type: A, layer: 1, pos: 45
type: A, layer: 1, pos: 26
type: A, layer: 1, pos: 5

Time for candidate selection: 0.07 seconds

### Candidate
type: A, layer: 1, pos: 9

### Candidate
type: A, layer: 1, pos: 6

### Candidate
type: A, layer: 1, pos: 38

### Candidate
type: A, layer: 1, pos: 48

### Candidate
type: A, layer: 1, pos: 0

### Candidate
type: A, layer: 1, pos: 28

### Candidate
type: A, layer: 1, pos: 39

### Candidate
type: A, layer: 1, pos: 37

## Relational analysis of NS_A2_B1_A1_B1_A2_B1_A1

### Relational analysis result of NS_A2_B1_A1_B1_A2_B1_A1
Status: Status.UNKNOWN
Output dim: 0, lower bound: -20.9341830, upper bound: 20.9224596
time: 1.17 seconds

## Relational analysis of NS_A2_B1_A1_B1_A2_B1_A2

### Relational analysis result of NS_A2_B1_A1_B1_A2_B1_A2
Status: Status.UNKNOWN
Output dim: 0, lower bound: -20.9341830, upper bound: 20.9455978
time: 0.61 seconds

## BFS NS instance: NS_A2_B1_A1_B1_A2_B2

### Backsubstitution after applying NS history:
0: -7.1838756, 19.5241489, -5.7272520, 16.0773335, -23.2612095, 25.2514000
1: -18.0769768, 29.7314720, -14.3067293, 24.7620640, -42.8390427, 44.0382004
2: -12.5251665, 26.9769573, -9.9589520, 22.4359322, -34.9610939, 36.9359093
3: -19.3050995, 33.0020142, -15.4190779, 27.3476086, -46.6527100, 48.4210930
4: -17.3972473, 33.2842674, -14.0162172, 27.7058773, -45.1031265, 47.3004837

Time for backsubstitution: 0.95 seconds

### NS candidates at layer 1
type: A, layer: 1, pos: 9
type: A, layer: 1, pos: 6
type: A, layer: 1, pos: 38
type: A, layer: 1, pos: 48
type: A, layer: 1, pos: 0
type: A, layer: 1, pos: 28
type: A, layer: 1, pos: 39
type: A, layer: 1, pos: 37
type: A, layer: 1, pos: 41
type: A, layer: 1, pos: 8
type: A, layer: 1, pos: 15
type: A, layer: 1, pos: 24
type: A, layer: 1, pos: 31
type: A, layer: 1, pos: 20
type: A, layer: 1, pos: 45
type: A, layer: 1, pos: 26
type: A, layer: 1, pos: 5

Time for candidate selection: 0.07 seconds

### Candidate
type: A, layer: 1, pos: 9

### Candidate
type: A, layer: 1, pos: 6

### Candidate
type: A, layer: 1, pos: 38

### Candidate
type: A, layer: 1, pos: 48

### Candidate
type: A, layer: 1, pos: 0

### Candidate
type: A, layer: 1, pos: 28

### Candidate
type: A, layer: 1, pos: 39

### Candidate
type: A, layer: 1, pos: 37

## Relational analysis of NS_A2_B1_A1_B1_A2_B2_A1

### Relational analysis result of NS_A2_B1_A1_B1_A2_B2_A1
Status: Status.UNKNOWN
Output dim: 0, lower bound: -20.9411884, upper bound: 20.9270553
time: 0.54 seconds

## Relational analysis of NS_A2_B1_A1_B1_A2_B2_A2

### Relational analysis result of NS_A2_B1_A1_B1_A2_B2_A2
Status: Status.UNKNOWN
Output dim: 0, lower bound: -20.9411884, upper bound: 20.9488332
time: 0.60 seconds

## BFS NS instance: NS_A2_B1_A1_B2_A1_B1

### Backsubstitution after applying NS history:
0: -7.2958264, 19.9399872, -5.6461415, 15.8184252, -23.1142483, 25.5861282
1: -18.5720406, 30.2115574, -14.0741758, 24.3985233, -42.9705658, 44.2857323
2: -12.7887821, 27.4622536, -9.8098946, 22.0409985, -34.8297806, 37.2721481
3: -19.6904030, 33.5532112, -15.1115742, 26.9437237, -46.6341209, 48.6647873
4: -17.5103760, 33.9847832, -13.6441526, 27.1492653, -44.6596413, 47.6289291

Time for backsubstitution: 0.96 seconds

### NS candidates at layer 1
type: A, layer: 1, pos: 9
type: A, layer: 1, pos: 6
type: A, layer: 1, pos: 38
type: A, layer: 1, pos: 48
type: A, layer: 1, pos: 28
type: A, layer: 1, pos: 8
type: A, layer: 1, pos: 0
type: A, layer: 1, pos: 37
type: A, layer: 1, pos: 41
type: A, layer: 1, pos: 20
type: A, layer: 1, pos: 39
type: A, layer: 1, pos: 31
type: A, layer: 1, pos: 15
type: A, layer: 1, pos: 5
type: A, layer: 1, pos: 24
type: A, layer: 1, pos: 26
type: A, layer: 1, pos: 45

Time for candidate selection: 0.07 seconds

### Candidate
type: A, layer: 1, pos: 9

## Relational analysis of NS_A2_B1_A1_B2_A1_B1_A1
Optimization infeasible because this subproblem isn't reachable.

### Candidate
type: A, layer: 1, pos: 6

## Relational analysis of NS_A2_B1_A1_B2_A1_B1_A1

### Relational analysis result of NS_A2_B1_A1_B2_A1_B1_A1
Status: Status.UNKNOWN
Output dim: 0, lower bound: -20.9303604, upper bound: 20.9425656
time: 0.60 seconds

## Relational analysis of NS_A2_B1_A1_B2_A1_B1_A2

### Relational analysis result of NS_A2_B1_A1_B2_A1_B1_A2
Status: Status.UNKNOWN
Output dim: 0, lower bound: -20.9441063, upper bound: 20.9501995
time: 0.60 seconds

## BFS NS instance: NS_A2_B1_A1_B2_A1_B2

### Backsubstitution after applying NS history:
0: -7.2430363, 19.7985516, -5.6121440, 15.6884975, -22.9315319, 25.4106960
1: -18.4409828, 29.9991875, -14.0385275, 24.0871315, -42.5281143, 44.0377121
2: -12.6969748, 27.2693424, -9.7719812, 21.8221893, -34.5191650, 37.0413055
3: -19.5496769, 33.3174286, -15.0425911, 26.6585331, -46.2082100, 48.3600159
4: -17.3799191, 33.7483063, -13.5239801, 26.9244518, -44.3043709, 47.2722855

Time for backsubstitution: 0.96 seconds

### NS candidates at layer 1
type: A, layer: 1, pos: 9
type: A, layer: 1, pos: 6
type: A, layer: 1, pos: 38
type: A, layer: 1, pos: 48
type: A, layer: 1, pos: 28
type: A, layer: 1, pos: 0
type: A, layer: 1, pos: 8
type: A, layer: 1, pos: 37
type: A, layer: 1, pos: 41
type: A, layer: 1, pos: 20
type: A, layer: 1, pos: 39
type: A, layer: 1, pos: 31
type: A, layer: 1, pos: 15
type: A, layer: 1, pos: 5
type: A, layer: 1, pos: 26
type: A, layer: 1, pos: 45
type: A, layer: 1, pos: 24

Time for candidate selection: 0.07 seconds

### Candidate
type: A, layer: 1, pos: 9

## Relational analysis of NS_A2_B1_A1_B2_A1_B2_A1

### Relational analysis result of NS_A2_B1_A1_B2_A1_B2_A1
Status: Status.UNKNOWN
Output dim: 0, lower bound: -20.9456987, upper bound: 20.9407625
time: 0.63 seconds

## Relational analysis of NS_A2_B1_A1_B2_A1_B2_A2
Optimization infeasible because this subproblem isn't reachable.

### Candidate
type: A, layer: 1, pos: 6

## Relational analysis of NS_A2_B1_A1_B2_A1_B2_A1

### Relational analysis result of NS_A2_B1_A1_B2_A1_B2_A1
Status: Status.UNKNOWN
Output dim: 0, lower bound: -20.9570587, upper bound: 20.9601795
time: 0.87 seconds

## Relational analysis of NS_A2_B1_A1_B2_A1_B2_A2

### Relational analysis result of NS_A2_B1_A1_B2_A1_B2_A2
Status: Status.UNKNOWN
Output dim: 0, lower bound: -20.9441063, upper bound: 20.9640981
time: 0.72 seconds

## BFS NS instance: NS_A2_B1_A1_B2_A2_B1

### Backsubstitution after applying NS history:
0: -7.2611141, 19.7217655, -5.6461415, 15.8184252, -23.0795383, 25.3679066
1: -18.2746468, 30.0185661, -14.0741758, 24.3985233, -42.6731720, 44.0927429
2: -12.6608963, 27.2419930, -9.8098946, 22.0409985, -34.7018967, 37.0518875
3: -19.5136757, 33.3254776, -15.1115742, 26.9437237, -46.4573975, 48.4370499
4: -17.5834770, 33.6119156, -13.6441526, 27.1492653, -44.7327423, 47.2560654

Time for backsubstitution: 0.97 seconds

### NS candidates at layer 1
type: A, layer: 1, pos: 9
type: A, layer: 1, pos: 6
type: A, layer: 1, pos: 38
type: A, layer: 1, pos: 48
type: A, layer: 1, pos: 0
type: A, layer: 1, pos: 28
type: A, layer: 1, pos: 39
type: A, layer: 1, pos: 37
type: A, layer: 1, pos: 41
type: A, layer: 1, pos: 8
type: A, layer: 1, pos: 15
type: A, layer: 1, pos: 24
type: A, layer: 1, pos: 20
type: A, layer: 1, pos: 31
type: A, layer: 1, pos: 45
type: A, layer: 1, pos: 26
type: A, layer: 1, pos: 5

Time for candidate selection: 0.07 seconds

### Candidate
type: A, layer: 1, pos: 9

## Relational analysis of NS_A2_B1_A1_B2_A2_B1_A1

### Relational analysis result of NS_A2_B1_A1_B2_A2_B1_A1
Status: Status.UNKNOWN
Output dim: 0, lower bound: -20.9273867, upper bound: 20.9229127
time: 0.78 seconds

## Relational analysis of NS_A2_B1_A1_B2_A2_B1_A2

### Relational analysis result of NS_A2_B1_A1_B2_A2_B1_A2
Status: Status.UNKNOWN
Output dim: 0, lower bound: -20.9273867, upper bound: 20.9234893
time: 0.79 seconds

## BFS NS instance: NS_A2_B1_A1_B2_A2_B2

### Backsubstitution after applying NS history:
0: -7.2028360, 19.5686207, -5.6121440, 15.6884975, -22.8913326, 25.1807632
1: -18.1313915, 29.7912064, -14.0385275, 24.0871315, -42.2185211, 43.8297348
2: -12.5599375, 27.0314064, -9.7719812, 21.8221893, -34.3821259, 36.8033829
3: -19.3585205, 33.0713081, -15.0425911, 26.6585331, -46.0170441, 48.1138878
4: -17.4386768, 33.3554459, -13.5239801, 26.9244518, -44.3631287, 46.8794250

Time for backsubstitution: 0.96 seconds

### NS candidates at layer 1
type: A, layer: 1, pos: 9
type: A, layer: 1, pos: 6
type: A, layer: 1, pos: 38
type: A, layer: 1, pos: 0
type: A, layer: 1, pos: 48
type: A, layer: 1, pos: 28
type: A, layer: 1, pos: 39
type: A, layer: 1, pos: 37
type: A, layer: 1, pos: 41
type: A, layer: 1, pos: 8
type: A, layer: 1, pos: 15
type: A, layer: 1, pos: 24
type: A, layer: 1, pos: 31
type: A, layer: 1, pos: 20
type: A, layer: 1, pos: 45
type: A, layer: 1, pos: 26
type: A, layer: 1, pos: 5

Time for candidate selection: 0.07 seconds

### Candidate
type: A, layer: 1, pos: 9

## Relational analysis of NS_A2_B1_A1_B2_A2_B2_A1

### Relational analysis result of NS_A2_B1_A1_B2_A2_B2_A1
Status: Status.UNKNOWN
Output dim: 0, lower bound: -20.9440816, upper bound: 20.9339440
time: 1.01 seconds

## Relational analysis of NS_A2_B1_A1_B2_A2_B2_A2

### Relational analysis result of NS_A2_B1_A1_B2_A2_B2_A2
Status: Status.UNKNOWN
Output dim: 0, lower bound: -20.9440816, upper bound: 20.9477517
time: 0.66 seconds

## BFS NS instance: NS_A2_B1_A2_B1_A1_B1

### Backsubstitution after applying NS history:
0: -5.7232141, 16.0164452, -5.3149152, 15.0045300, -20.7277431, 21.3313599
1: -14.2624454, 24.6904526, -13.1715002, 23.2039242, -37.4663696, 37.8619499
2: -9.9550819, 22.3152981, -9.1920538, 21.0012665, -30.9563446, 31.5073509
3: -15.3149691, 27.2726002, -14.2750311, 25.5840683, -40.8990364, 41.5476303
4: -13.8457785, 27.4785500, -13.0492983, 25.8937531, -39.7395325, 40.5278473

Time for backsubstitution: 0.97 seconds

### NS candidates at layer 1
type: A, layer: 1, pos: 9
type: A, layer: 1, pos: 39
type: A, layer: 1, pos: 48
type: A, layer: 1, pos: 6
type: A, layer: 1, pos: 28
type: A, layer: 1, pos: 38
type: A, layer: 1, pos: 0
type: A, layer: 1, pos: 41
type: A, layer: 1, pos: 31
type: A, layer: 1, pos: 37
type: A, layer: 1, pos: 20
type: A, layer: 1, pos: 24
type: A, layer: 1, pos: 8
type: A, layer: 1, pos: 15
type: A, layer: 1, pos: 45
type: A, layer: 1, pos: 26
type: A, layer: 1, pos: 5

Time for candidate selection: 0.07 seconds

### Candidate
type: A, layer: 1, pos: 9

### Candidate
type: A, layer: 1, pos: 39

### Candidate
type: A, layer: 1, pos: 48

### Candidate
type: A, layer: 1, pos: 6

### Candidate
type: A, layer: 1, pos: 28

### Candidate
type: A, layer: 1, pos: 38

### Candidate
type: A, layer: 1, pos: 0

## Relational analysis of NS_A2_B1_A2_B1_A1_B1_A1

### Relational analysis result of NS_A2_B1_A2_B1_A1_B1_A1
Status: Status.UNKNOWN
Output dim: 0, lower bound: -20.9614214, upper bound: 20.9618822
time: 0.61 seconds

## Relational analysis of NS_A2_B1_A2_B1_A1_B1_A2

### Relational analysis result of NS_A2_B1_A2_B1_A1_B1_A2
Status: Status.UNKNOWN
Output dim: 0, lower bound: -20.9614214, upper bound: 20.9618822
time: 0.67 seconds

## BFS NS instance: NS_A2_B1_A2_B1_A1_B2

### Backsubstitution after applying NS history:
0: -5.7942882, 16.2022667, -5.5565538, 15.6899843, -21.4842720, 21.7588177
1: -14.4589930, 24.9580421, -13.8594294, 24.2532310, -38.7122192, 38.8174629
2: -10.0880079, 22.5631218, -9.6509180, 21.9123402, -32.0003471, 32.2140388
3: -15.5135593, 27.5783768, -14.9580021, 26.7585545, -42.2721138, 42.5363770
4: -14.0118141, 27.7913380, -13.5923805, 27.0415897, -41.0534019, 41.3837128

Time for backsubstitution: 0.97 seconds

### NS candidates at layer 1
type: A, layer: 1, pos: 9
type: A, layer: 1, pos: 39
type: A, layer: 1, pos: 48
type: A, layer: 1, pos: 6
type: A, layer: 1, pos: 28
type: A, layer: 1, pos: 38
type: A, layer: 1, pos: 0
type: A, layer: 1, pos: 41
type: A, layer: 1, pos: 31
type: A, layer: 1, pos: 37
type: A, layer: 1, pos: 20
type: A, layer: 1, pos: 8
type: A, layer: 1, pos: 24
type: A, layer: 1, pos: 15
type: A, layer: 1, pos: 45
type: A, layer: 1, pos: 26
type: A, layer: 1, pos: 5

Time for candidate selection: 0.07 seconds

### Candidate
type: A, layer: 1, pos: 9

### Candidate
type: A, layer: 1, pos: 39

### Candidate
type: A, layer: 1, pos: 48

### Candidate
type: A, layer: 1, pos: 6

### Candidate
type: A, layer: 1, pos: 28

### Candidate
type: A, layer: 1, pos: 38

### Candidate
type: A, layer: 1, pos: 0

## Relational analysis of NS_A2_B1_A2_B1_A1_B2_A1

### Relational analysis result of NS_A2_B1_A2_B1_A1_B2_A1
Status: Status.UNKNOWN
Output dim: 0, lower bound: -20.9614214, upper bound: 20.9618822
time: 0.59 seconds

## Relational analysis of NS_A2_B1_A2_B1_A1_B2_A2

### Relational analysis result of NS_A2_B1_A2_B1_A1_B2_A2
Status: Status.UNKNOWN
Output dim: 0, lower bound: -20.9614214, upper bound: 20.9618822
time: 0.66 seconds

## BFS NS instance: NS_A2_B1_A2_B1_A2_B1

### Backsubstitution after applying NS history:
0: -5.9101925, 16.4418354, -5.3149152, 15.0045300, -20.9147205, 21.7567501
1: -14.5664444, 25.4331913, -13.1715002, 23.2039242, -37.7703705, 38.6046829
2: -10.2347069, 22.9684448, -9.1920538, 21.0012665, -31.2359715, 32.1604996
3: -15.7661171, 28.0749893, -14.2750311, 25.5840683, -41.3501854, 42.3500214
4: -14.4613228, 28.2083759, -13.0492983, 25.8937531, -40.3550682, 41.2576752

Time for backsubstitution: 0.97 seconds

### NS candidates at layer 1
type: A, layer: 1, pos: 39
type: A, layer: 1, pos: 9
type: A, layer: 1, pos: 6
type: A, layer: 1, pos: 48
type: A, layer: 1, pos: 28
type: A, layer: 1, pos: 38
type: A, layer: 1, pos: 0
type: A, layer: 1, pos: 24
type: A, layer: 1, pos: 37
type: A, layer: 1, pos: 41
type: A, layer: 1, pos: 20
type: A, layer: 1, pos: 8
type: A, layer: 1, pos: 31
type: A, layer: 1, pos: 15
type: A, layer: 1, pos: 26
type: A, layer: 1, pos: 5
type: A, layer: 1, pos: 45

Time for candidate selection: 0.07 seconds

### Candidate
type: A, layer: 1, pos: 39

### Candidate
type: A, layer: 1, pos: 9

### Candidate
type: A, layer: 1, pos: 6

### Candidate
type: A, layer: 1, pos: 48

### Candidate
type: A, layer: 1, pos: 28

### Candidate
type: A, layer: 1, pos: 38

### Candidate
type: A, layer: 1, pos: 0

## Relational analysis of NS_A2_B1_A2_B1_A2_B1_A1

### Relational analysis result of NS_A2_B1_A2_B1_A2_B1_A1
Status: Status.UNKNOWN
Output dim: 0, lower bound: -20.9549159, upper bound: 20.9534221
time: 0.62 seconds

## Relational analysis of NS_A2_B1_A2_B1_A2_B1_A2

### Relational analysis result of NS_A2_B1_A2_B1_A2_B1_A2
Status: Status.UNKNOWN
Output dim: 0, lower bound: -20.9549159, upper bound: 20.9537253
time: 0.65 seconds

## BFS NS instance: NS_A2_B1_A2_B1_A2_B2

### Backsubstitution after applying NS history:
0: -5.9756260, 16.6119061, -5.5565538, 15.6899843, -21.6656113, 22.1684608
1: -14.7459917, 25.6773434, -13.8594294, 24.2532310, -38.9992180, 39.5367699
2: -10.3569841, 23.1951981, -9.6509180, 21.9123402, -32.2693253, 32.8461151
3: -15.9492168, 28.3532829, -14.9580021, 26.7585545, -42.7077713, 43.3112869
4: -14.6166353, 28.4932518, -13.5923805, 27.0415897, -41.6582184, 42.0856323

Time for backsubstitution: 0.98 seconds

### NS candidates at layer 1
type: A, layer: 1, pos: 39
type: A, layer: 1, pos: 9
type: A, layer: 1, pos: 6
type: A, layer: 1, pos: 48
type: A, layer: 1, pos: 28
type: A, layer: 1, pos: 0
type: A, layer: 1, pos: 38
type: A, layer: 1, pos: 24
type: A, layer: 1, pos: 37
type: A, layer: 1, pos: 41
type: A, layer: 1, pos: 8
type: A, layer: 1, pos: 31
type: A, layer: 1, pos: 20
type: A, layer: 1, pos: 15
type: A, layer: 1, pos: 26
type: A, layer: 1, pos: 5
type: A, layer: 1, pos: 45

Time for candidate selection: 0.07 seconds

### Candidate
type: A, layer: 1, pos: 39

### Candidate
type: A, layer: 1, pos: 9

### Candidate
type: A, layer: 1, pos: 6

### Candidate
type: A, layer: 1, pos: 48

### Candidate
type: A, layer: 1, pos: 28

### Candidate
type: A, layer: 1, pos: 0

## Relational analysis of NS_A2_B1_A2_B1_A2_B2_A1

### Relational analysis result of NS_A2_B1_A2_B1_A2_B2_A1
Status: Status.UNKNOWN
Output dim: 0, lower bound: -20.9549159, upper bound: 20.9534221
time: 0.58 seconds

## Relational analysis of NS_A2_B1_A2_B1_A2_B2_A2

### Relational analysis result of NS_A2_B1_A2_B1_A2_B2_A2
Status: Status.UNKNOWN
Output dim: 0, lower bound: -20.9549159, upper bound: 20.9537253
time: 0.63 seconds

## BFS NS instance: NS_A2_B1_A2_B2_A1_B1

### Backsubstitution after applying NS history:
0: -5.7942882, 16.2022667, -7.2958264, 19.9399872, -25.7342758, 23.4980869
1: -14.4589930, 24.9580421, -18.5720406, 30.2115574, -44.6705475, 43.5300789
2: -10.0880079, 22.5631218, -12.7887821, 27.4622536, -37.5502625, 35.3519058
3: -15.5135593, 27.5783768, -19.6904030, 33.5532112, -49.0667686, 47.2687798
4: -14.0118141, 27.7913380, -17.5103760, 33.9847832, -47.9965935, 45.3017120

Time for backsubstitution: 0.98 seconds

### NS candidates at layer 1
type: A, layer: 1, pos: 9
type: A, layer: 1, pos: 39
type: A, layer: 1, pos: 48
type: A, layer: 1, pos: 6
type: A, layer: 1, pos: 28
type: A, layer: 1, pos: 38
type: A, layer: 1, pos: 0
type: A, layer: 1, pos: 41
type: A, layer: 1, pos: 31
type: A, layer: 1, pos: 37
type: A, layer: 1, pos: 20
type: A, layer: 1, pos: 8
type: A, layer: 1, pos: 24
type: A, layer: 1, pos: 15
type: A, layer: 1, pos: 45
type: A, layer: 1, pos: 26
type: A, layer: 1, pos: 5

Time for candidate selection: 0.07 seconds

### Candidate
type: A, layer: 1, pos: 9

## Relational analysis of NS_A2_B1_A2_B2_A1_B1_A1

### Relational analysis result of NS_A2_B1_A2_B2_A1_B1_A1
Status: Status.UNKNOWN
Output dim: 0, lower bound: -20.9489641, upper bound: 20.9434778
time: 0.67 seconds

## Relational analysis of NS_A2_B1_A2_B2_A1_B1_A2

### Relational analysis result of NS_A2_B1_A2_B2_A1_B1_A2
Status: Status.UNKNOWN
Output dim: 0, lower bound: -20.9642419, upper bound: 20.9640326
time: 0.80 seconds

## BFS NS instance: NS_A2_B1_A2_B2_A1_B2

### Backsubstitution after applying NS history:
0: -5.7942882, 16.2022667, -5.7942882, 16.2022667, -21.9965553, 21.9965553
1: -14.4589930, 24.9580421, -14.4589930, 24.9580421, -39.4170303, 39.4170303
2: -10.0880079, 22.5631218, -10.0880079, 22.5631218, -32.6511269, 32.6511269
3: -15.5135593, 27.5783768, -15.5135593, 27.5783768, -43.0919342, 43.0919342
4: -14.0118141, 27.7913380, -14.0118141, 27.7913380, -41.8031540, 41.8031502

Time for backsubstitution: 0.98 seconds

### NS candidates at layer 1
type: A, layer: 1, pos: 9
type: A, layer: 1, pos: 39
type: A, layer: 1, pos: 48
type: A, layer: 1, pos: 6
type: A, layer: 1, pos: 28
type: A, layer: 1, pos: 38
type: A, layer: 1, pos: 0
type: A, layer: 1, pos: 41
type: A, layer: 1, pos: 31
type: A, layer: 1, pos: 37
type: A, layer: 1, pos: 20
type: A, layer: 1, pos: 8
type: A, layer: 1, pos: 24
type: A, layer: 1, pos: 15
type: A, layer: 1, pos: 45
type: A, layer: 1, pos: 26
type: A, layer: 1, pos: 5

Time for candidate selection: 0.07 seconds

### Candidate
type: A, layer: 1, pos: 9

## Relational analysis of NS_A2_B1_A2_B2_A1_B2_A1
Optimization infeasible because this subproblem isn't reachable.

### Candidate
type: A, layer: 1, pos: 39

### Candidate
type: A, layer: 1, pos: 48

## Relational analysis of NS_A2_B1_A2_B2_A1_B2_A1
Optimization infeasible because this subproblem isn't reachable.

### Candidate
type: A, layer: 1, pos: 6

## Relational analysis of NS_A2_B1_A2_B2_A1_B2_A1
Optimization infeasible because this subproblem isn't reachable.

### Candidate
type: A, layer: 1, pos: 28

## Relational analysis of NS_A2_B1_A2_B2_A1_B2_A1
Optimization infeasible because this subproblem isn't reachable.

### Candidate
type: A, layer: 1, pos: 38

## Relational analysis of NS_A2_B1_A2_B2_A1_B2_A1
Optimization infeasible because this subproblem isn't reachable.

### Candidate
type: A, layer: 1, pos: 0

## Relational analysis of NS_A2_B1_A2_B2_A1_B2_A1

### Relational analysis result of NS_A2_B1_A2_B2_A1_B2_A1
Status: Status.UNKNOWN
Output dim: 0, lower bound: -20.9525391, upper bound: 20.9614829
time: 1.24 seconds

## Relational analysis of NS_A2_B1_A2_B2_A1_B2_A2

### Relational analysis result of NS_A2_B1_A2_B2_A1_B2_A2
Status: Status.UNKNOWN
Output dim: 0, lower bound: -20.9524133, upper bound: 20.9614214
time: 0.85 seconds

## BFS NS instance: NS_A2_B1_A2_B2_A2_B1

### Backsubstitution after applying NS history:
0: -5.9756260, 16.6119061, -7.2958264, 19.9399872, -25.9156132, 23.9077301
1: -14.7459917, 25.6773434, -18.5720406, 30.2115574, -44.9575462, 44.2493820
2: -10.3569841, 23.1951981, -12.7887821, 27.4622536, -37.8192368, 35.9839783
3: -15.9492168, 28.3532829, -19.6904030, 33.5532112, -49.5024261, 48.0436859
4: -14.6166353, 28.4932518, -17.5103760, 33.9847832, -48.6014137, 46.0036278

Time for backsubstitution: 0.99 seconds

### NS candidates at layer 1
type: A, layer: 1, pos: 39
type: A, layer: 1, pos: 9
type: A, layer: 1, pos: 6
type: A, layer: 1, pos: 48
type: A, layer: 1, pos: 28
type: A, layer: 1, pos: 0
type: A, layer: 1, pos: 38
type: A, layer: 1, pos: 24
type: A, layer: 1, pos: 37
type: A, layer: 1, pos: 41
type: A, layer: 1, pos: 8
type: A, layer: 1, pos: 31
type: A, layer: 1, pos: 20
type: A, layer: 1, pos: 15
type: A, layer: 1, pos: 26
type: A, layer: 1, pos: 5
type: A, layer: 1, pos: 45

Time for candidate selection: 0.07 seconds

### Candidate
type: A, layer: 1, pos: 39

### Candidate
type: A, layer: 1, pos: 9

## Relational analysis of NS_A2_B1_A2_B2_A2_B1_A1

### Relational analysis result of NS_A2_B1_A2_B2_A2_B1_A1
Status: Status.UNKNOWN
Output dim: 0, lower bound: -20.9486676, upper bound: 20.9411943
time: 0.70 seconds

## Relational analysis of NS_A2_B1_A2_B2_A2_B1_A2

### Relational analysis result of NS_A2_B1_A2_B2_A2_B1_A2
Status: Status.UNKNOWN
Output dim: 0, lower bound: -20.9551491, upper bound: 20.9511649
time: 1.16 seconds

## BFS NS instance: NS_A2_B1_A2_B2_A2_B2

### Backsubstitution after applying NS history:
0: -5.9756260, 16.6119061, -5.7942882, 16.2022667, -22.1778889, 22.4061947
1: -14.7459917, 25.6773434, -14.4589930, 24.9580421, -39.7040291, 40.1363335
2: -10.3569841, 23.1951981, -10.0880079, 22.5631218, -32.9201050, 33.2832031
3: -15.9492168, 28.3532829, -15.5135593, 27.5783768, -43.5275955, 43.8668365
4: -14.6166353, 28.4932518, -14.0118141, 27.7913380, -42.4079666, 42.5050659

Time for backsubstitution: 0.99 seconds

### NS candidates at layer 1
type: A, layer: 1, pos: 39
type: A, layer: 1, pos: 9
type: A, layer: 1, pos: 6
type: A, layer: 1, pos: 48
type: A, layer: 1, pos: 28
type: A, layer: 1, pos: 0
type: A, layer: 1, pos: 38
type: A, layer: 1, pos: 24
type: A, layer: 1, pos: 37
type: A, layer: 1, pos: 41
type: A, layer: 1, pos: 8
type: A, layer: 1, pos: 31
type: A, layer: 1, pos: 20
type: A, layer: 1, pos: 15
type: A, layer: 1, pos: 26
type: A, layer: 1, pos: 5
type: A, layer: 1, pos: 45

Time for candidate selection: 0.07 seconds

### Candidate
type: A, layer: 1, pos: 39

### Candidate
type: A, layer: 1, pos: 9

## Relational analysis of NS_A2_B1_A2_B2_A2_B2_A1

### Relational analysis result of NS_A2_B1_A2_B2_A2_B2_A1
Status: Status.UNKNOWN
Output dim: 0, lower bound: -20.9486676, upper bound: 20.9489766
time: 0.83 seconds

## Relational analysis of NS_A2_B1_A2_B2_A2_B2_A2

### Relational analysis result of NS_A2_B1_A2_B2_A2_B2_A2
Status: Status.UNKNOWN
Output dim: 0, lower bound: -20.9551491, upper bound: 20.9541027
time: 0.84 seconds

## BFS NS instance: NS_A2_B2_A1_B1_A1_B1

### Backsubstitution after applying NS history:
0: -7.2898707, 19.9250507, -5.7593737, 16.1530533, -23.4429245, 25.6844215
1: -18.5570850, 30.1894703, -14.2705688, 24.9937592, -43.5508423, 44.4600372
2: -12.7783670, 27.4419422, -9.9754887, 22.6008472, -35.3792152, 37.4174309
3: -19.6743641, 33.5284195, -15.4613800, 27.5570297, -47.2313919, 48.9897995
4: -17.4955368, 33.9598045, -14.1926994, 27.8550491, -45.3505859, 48.1525040

Time for backsubstitution: 0.99 seconds

### NS candidates at layer 1
type: A, layer: 1, pos: 9
type: A, layer: 1, pos: 6
type: A, layer: 1, pos: 38
type: A, layer: 1, pos: 48
type: A, layer: 1, pos: 28
type: A, layer: 1, pos: 8
type: A, layer: 1, pos: 0
type: A, layer: 1, pos: 37
type: A, layer: 1, pos: 41
type: A, layer: 1, pos: 20
type: A, layer: 1, pos: 39
type: A, layer: 1, pos: 31
type: A, layer: 1, pos: 15
type: A, layer: 1, pos: 5
type: A, layer: 1, pos: 24
type: A, layer: 1, pos: 26
type: A, layer: 1, pos: 45

Time for candidate selection: 0.07 seconds

### Candidate
type: A, layer: 1, pos: 9

### Candidate
type: A, layer: 1, pos: 6

### Candidate
type: A, layer: 1, pos: 38

### Candidate
type: A, layer: 1, pos: 48

### Candidate
type: A, layer: 1, pos: 28

### Candidate
type: A, layer: 1, pos: 8

## Relational analysis of NS_A2_B2_A1_B1_A1_B1_A1

### Relational analysis result of NS_A2_B2_A1_B1_A1_B1_A1
Status: Status.UNKNOWN
Output dim: 0, lower bound: -20.9527657, upper bound: 20.9572021
time: 0.58 seconds

## Relational analysis of NS_A2_B2_A1_B1_A1_B1_A2

### Relational analysis result of NS_A2_B2_A1_B1_A1_B1_A2
Status: Status.UNKNOWN
Output dim: 0, lower bound: -20.9564671, upper bound: 20.9602771
time: 0.56 seconds

## BFS NS instance: NS_A2_B2_A1_B1_A1_B2

### Backsubstitution after applying NS history:
0: -7.2051101, 19.6986122, -5.9965081, 16.7290058, -23.9341164, 25.6951199
1: -18.3306866, 29.8652592, -14.8499393, 25.8211803, -44.1518593, 44.7151947
2: -12.6249018, 27.1386833, -10.3930006, 23.3898659, -36.0147667, 37.5316811
3: -19.4402695, 33.1619530, -16.0963974, 28.4984932, -47.9387627, 49.2583466
4: -17.2928104, 33.5811272, -14.8098030, 28.8273373, -46.1201439, 48.3909302

Time for backsubstitution: 0.99 seconds

### NS candidates at layer 1
type: A, layer: 1, pos: 9
type: A, layer: 1, pos: 6
type: A, layer: 1, pos: 38
type: A, layer: 1, pos: 48
type: A, layer: 1, pos: 28
type: A, layer: 1, pos: 0
type: A, layer: 1, pos: 8
type: A, layer: 1, pos: 41
type: A, layer: 1, pos: 37
type: A, layer: 1, pos: 20
type: A, layer: 1, pos: 39
type: A, layer: 1, pos: 31
type: A, layer: 1, pos: 15
type: A, layer: 1, pos: 24
type: A, layer: 1, pos: 5
type: A, layer: 1, pos: 45
type: A, layer: 1, pos: 26

Time for candidate selection: 0.07 seconds

### Candidate
type: A, layer: 1, pos: 9

### Candidate
type: A, layer: 1, pos: 6

### Candidate
type: A, layer: 1, pos: 38

### Candidate
type: A, layer: 1, pos: 48

### Candidate
type: A, layer: 1, pos: 28

### Candidate
type: A, layer: 1, pos: 0

### Candidate
type: A, layer: 1, pos: 8

## Relational analysis of NS_A2_B2_A1_B1_A1_B2_A1

### Relational analysis result of NS_A2_B2_A1_B1_A1_B2_A1
Status: Status.UNKNOWN
Output dim: 0, lower bound: -20.9520676, upper bound: 20.9565773
time: 0.58 seconds

## Relational analysis of NS_A2_B2_A1_B1_A1_B2_A2

### Relational analysis result of NS_A2_B2_A1_B1_A1_B2_A2
Status: Status.UNKNOWN
Output dim: 0, lower bound: -20.9549741, upper bound: 20.9584735
time: 0.93 seconds

## BFS NS instance: NS_A2_B2_A1_B1_A2_B1

### Backsubstitution after applying NS history:
0: -7.2532120, 19.7014389, -5.7593737, 16.1530533, -23.4062653, 25.4608116
1: -18.2541428, 29.9893970, -14.2705688, 24.9937592, -43.2479019, 44.2599640
2: -12.6468019, 27.2146072, -9.9754887, 22.6008472, -35.2476501, 37.1900940
3: -19.4920635, 33.2924194, -15.4613800, 27.5570297, -47.0490952, 48.7537994
4: -17.5639420, 33.5777855, -14.1926994, 27.8550491, -45.4189911, 47.7704849

Time for backsubstitution: 0.99 seconds

### NS candidates at layer 1
type: A, layer: 1, pos: 9
type: A, layer: 1, pos: 6
type: A, layer: 1, pos: 38
type: A, layer: 1, pos: 48
type: A, layer: 1, pos: 0
type: A, layer: 1, pos: 28
type: A, layer: 1, pos: 39
type: A, layer: 1, pos: 37
type: A, layer: 1, pos: 41
type: A, layer: 1, pos: 8
type: A, layer: 1, pos: 15
type: A, layer: 1, pos: 24
type: A, layer: 1, pos: 20
type: A, layer: 1, pos: 31
type: A, layer: 1, pos: 45
type: A, layer: 1, pos: 26
type: A, layer: 1, pos: 5

Time for candidate selection: 0.07 seconds

### Candidate
type: A, layer: 1, pos: 9

### Candidate
type: A, layer: 1, pos: 6

### Candidate
type: A, layer: 1, pos: 38

### Candidate
type: A, layer: 1, pos: 48

### Candidate
type: A, layer: 1, pos: 0

### Candidate
type: A, layer: 1, pos: 28

### Candidate
type: A, layer: 1, pos: 39

### Candidate
type: A, layer: 1, pos: 37

## Relational analysis of NS_A2_B2_A1_B1_A2_B1_A1

### Relational analysis result of NS_A2_B2_A1_B1_A2_B1_A1
Status: Status.UNKNOWN
Output dim: 0, lower bound: -20.9315604, upper bound: 20.9203569
time: 0.66 seconds

## Relational analysis of NS_A2_B2_A1_B1_A2_B1_A2

### Relational analysis result of NS_A2_B2_A1_B1_A2_B1_A2
Status: Status.UNKNOWN
Output dim: 0, lower bound: -20.9315604, upper bound: 20.9423008
time: 1.09 seconds

## BFS NS instance: NS_A2_B2_A1_B1_A2_B2

### Backsubstitution after applying NS history:
0: -7.1838756, 19.5241489, -5.9965081, 16.7290058, -23.9128799, 25.5206566
1: -18.0769768, 29.7314720, -14.8499393, 25.8211803, -43.8981552, 44.5814095
2: -12.5251665, 26.9769573, -10.3930006, 23.3898659, -35.9150314, 37.3699570
3: -19.3050995, 33.0020142, -16.0963974, 28.4984932, -47.8035927, 49.0984077
4: -17.3972473, 33.2842674, -14.8098030, 28.8273373, -46.2245865, 48.0940704

Time for backsubstitution: 0.99 seconds

### NS candidates at layer 1
type: A, layer: 1, pos: 9
type: A, layer: 1, pos: 6
type: A, layer: 1, pos: 38
type: A, layer: 1, pos: 48
type: A, layer: 1, pos: 0
type: A, layer: 1, pos: 28
type: A, layer: 1, pos: 39
type: A, layer: 1, pos: 37
type: A, layer: 1, pos: 41
type: A, layer: 1, pos: 8
type: A, layer: 1, pos: 15
type: A, layer: 1, pos: 24
type: A, layer: 1, pos: 31
type: A, layer: 1, pos: 20
type: A, layer: 1, pos: 45
type: A, layer: 1, pos: 26
type: A, layer: 1, pos: 5

Time for candidate selection: 0.07 seconds

### Candidate
type: A, layer: 1, pos: 9

### Candidate
type: A, layer: 1, pos: 6

### Candidate
type: A, layer: 1, pos: 38

### Candidate
type: A, layer: 1, pos: 48

### Candidate
type: A, layer: 1, pos: 0

### Candidate
type: A, layer: 1, pos: 28

### Candidate
type: A, layer: 1, pos: 39

### Candidate
type: A, layer: 1, pos: 37

## Relational analysis of NS_A2_B2_A1_B1_A2_B2_A1

### Relational analysis result of NS_A2_B2_A1_B1_A2_B2_A1
Status: Status.UNKNOWN
Output dim: 0, lower bound: -20.9315604, upper bound: 20.9203569
time: 0.65 seconds

## Relational analysis of NS_A2_B2_A1_B1_A2_B2_A2

### Relational analysis result of NS_A2_B2_A1_B1_A2_B2_A2
Status: Status.UNKNOWN
Output dim: 0, lower bound: -20.9315604, upper bound: 20.9423008
time: 0.66 seconds

## BFS NS instance: NS_A2_B2_A1_B2_A1_B1

### Backsubstitution after applying NS history:
0: -7.2958264, 19.9399872, -7.2611141, 19.7217655, -27.0175896, 27.2011013
1: -18.5720406, 30.2115574, -18.2746468, 30.0185661, -48.5906067, 48.4862061
2: -12.7887821, 27.4622536, -12.6608963, 27.2419930, -40.0307693, 40.1231499
3: -19.6904030, 33.5532112, -19.5136757, 33.3254776, -53.0158806, 53.0668869
4: -17.5103760, 33.9847832, -17.5834770, 33.6119156, -51.1222916, 51.5682564

Time for backsubstitution: 1.00 seconds

### NS candidates at layer 1
type: A, layer: 1, pos: 9
type: A, layer: 1, pos: 6
type: A, layer: 1, pos: 38
type: A, layer: 1, pos: 48
type: A, layer: 1, pos: 28
type: A, layer: 1, pos: 8
type: A, layer: 1, pos: 0
type: A, layer: 1, pos: 37
type: A, layer: 1, pos: 41
type: A, layer: 1, pos: 20
type: A, layer: 1, pos: 39
type: A, layer: 1, pos: 31
type: A, layer: 1, pos: 15
type: A, layer: 1, pos: 5
type: A, layer: 1, pos: 24
type: A, layer: 1, pos: 26
type: A, layer: 1, pos: 45

Time for candidate selection: 0.07 seconds

### Candidate
type: A, layer: 1, pos: 9

## Relational analysis of NS_A2_B2_A1_B2_A1_B1_A1

### Relational analysis result of NS_A2_B2_A1_B2_A1_B1_A1
Status: Status.UNKNOWN
Output dim: 0, lower bound: -20.9234777, upper bound: 20.9244704
time: 0.78 seconds

## Relational analysis of NS_A2_B2_A1_B2_A1_B1_A2

### Relational analysis result of NS_A2_B2_A1_B2_A1_B1_A2
Status: Status.UNKNOWN
Output dim: 0, lower bound: -20.9467965, upper bound: 20.9518643
time: 0.76 seconds

## BFS NS instance: NS_A2_B2_A1_B2_A1_B2

### Backsubstitution after applying NS history:
0: -7.2958264, 19.9399872, -5.9625483, 16.5776463, -23.8734684, 25.9025345
1: -18.5720406, 30.2115574, -14.7118168, 25.6270142, -44.1990509, 44.9233742
2: -12.7887821, 27.4622536, -10.3330498, 23.1492825, -35.9380608, 37.7953033
3: -19.6904030, 33.5532112, -15.9134026, 28.2967262, -47.9871292, 49.4666138
4: -17.5103760, 33.9847832, -14.5857401, 28.4363346, -45.9467087, 48.5705185

Time for backsubstitution: 1.01 seconds

### NS candidates at layer 1
type: A, layer: 1, pos: 9
type: A, layer: 1, pos: 6
type: A, layer: 1, pos: 38
type: A, layer: 1, pos: 48
type: A, layer: 1, pos: 28
type: A, layer: 1, pos: 8
type: A, layer: 1, pos: 0
type: A, layer: 1, pos: 37
type: A, layer: 1, pos: 41
type: A, layer: 1, pos: 20
type: A, layer: 1, pos: 39
type: A, layer: 1, pos: 31
type: A, layer: 1, pos: 15
type: A, layer: 1, pos: 5
type: A, layer: 1, pos: 24
type: A, layer: 1, pos: 26
type: A, layer: 1, pos: 45

Time for candidate selection: 0.07 seconds

### Candidate
type: A, layer: 1, pos: 9

## Relational analysis of NS_A2_B2_A1_B2_A1_B2_A1

### Relational analysis result of NS_A2_B2_A1_B2_A1_B2_A1
Status: Status.UNKNOWN
Output dim: 0, lower bound: -20.9234777, upper bound: 20.9285327
time: 0.67 seconds

## Relational analysis of NS_A2_B2_A1_B2_A1_B2_A2

### Relational analysis result of NS_A2_B2_A1_B2_A1_B2_A2
Status: Status.UNKNOWN
Output dim: 0, lower bound: -20.9467965, upper bound: 20.9564293
time: 0.67 seconds

## BFS NS instance: NS_A2_B2_A1_B2_A2_B1

### Backsubstitution after applying NS history:
0: -7.2611141, 19.7217655, -7.2611141, 19.7217655, -26.9828796, 26.9828796
1: -18.2746468, 30.0185661, -18.2746468, 30.0185661, -48.2932129, 48.2932129
2: -12.6608963, 27.2419930, -12.6608963, 27.2419930, -39.9028893, 39.9028893
3: -19.5136757, 33.3254776, -19.5136757, 33.3254776, -52.8391533, 52.8391533
4: -17.5834770, 33.6119156, -17.5834770, 33.6119156, -51.1953926, 51.1953926

Time for backsubstitution: 1.00 seconds

### NS candidates at layer 1
type: A, layer: 1, pos: 9
type: A, layer: 1, pos: 6
type: A, layer: 1, pos: 38
type: A, layer: 1, pos: 48
type: A, layer: 1, pos: 0
type: A, layer: 1, pos: 28
type: A, layer: 1, pos: 39
type: A, layer: 1, pos: 37
type: A, layer: 1, pos: 41
type: A, layer: 1, pos: 8
type: A, layer: 1, pos: 15
type: A, layer: 1, pos: 24
type: A, layer: 1, pos: 20
type: A, layer: 1, pos: 31
type: A, layer: 1, pos: 45
type: A, layer: 1, pos: 26
type: A, layer: 1, pos: 5

Time for candidate selection: 0.07 seconds

### Candidate
type: A, layer: 1, pos: 9

## Relational analysis of NS_A2_B2_A1_B2_A2_B1_A1

### Relational analysis result of NS_A2_B2_A1_B2_A2_B1_A1
Status: Status.UNKNOWN
Output dim: 0, lower bound: -20.9250963, upper bound: 20.9203729
time: 0.62 seconds

## Relational analysis of NS_A2_B2_A1_B2_A2_B1_A2

### Relational analysis result of NS_A2_B2_A1_B2_A2_B1_A2
Status: Status.UNKNOWN
Output dim: 0, lower bound: -20.9352853, upper bound: 20.9352838
time: 0.75 seconds

## BFS NS instance: NS_A2_B2_A1_B2_A2_B2

### Backsubstitution after applying NS history:
0: -7.2611141, 19.7217655, -5.9625483, 16.5776463, -23.8387585, 25.6843090
1: -18.2746468, 30.0185661, -14.7118168, 25.6270142, -43.9016571, 44.7303848
2: -12.6608963, 27.2419930, -10.3330498, 23.1492825, -35.8101730, 37.5750389
3: -19.5136757, 33.3254776, -15.9134026, 28.2967262, -47.8104019, 49.2388802
4: -17.5834770, 33.6119156, -14.5857401, 28.4363346, -46.0198097, 48.1976547

Time for backsubstitution: 1.01 seconds

### NS candidates at layer 1
type: A, layer: 1, pos: 9
type: A, layer: 1, pos: 6
type: A, layer: 1, pos: 38
type: A, layer: 1, pos: 48
type: A, layer: 1, pos: 0
type: A, layer: 1, pos: 28
type: A, layer: 1, pos: 39
type: A, layer: 1, pos: 37
type: A, layer: 1, pos: 41
type: A, layer: 1, pos: 8
type: A, layer: 1, pos: 15
type: A, layer: 1, pos: 24
type: A, layer: 1, pos: 20
type: A, layer: 1, pos: 31
type: A, layer: 1, pos: 45
type: A, layer: 1, pos: 26
type: A, layer: 1, pos: 5

Time for candidate selection: 0.07 seconds

### Candidate
type: A, layer: 1, pos: 9

## Relational analysis of NS_A2_B2_A1_B2_A2_B2_A1

### Relational analysis result of NS_A2_B2_A1_B2_A2_B2_A1
Status: Status.UNKNOWN
Output dim: 0, lower bound: -20.9250963, upper bound: 20.9251292
time: 0.66 seconds

## Relational analysis of NS_A2_B2_A1_B2_A2_B2_A2

### Relational analysis result of NS_A2_B2_A1_B2_A2_B2_A2
Status: Status.UNKNOWN
Output dim: 0, lower bound: -20.9352853, upper bound: 20.9393947
time: 0.67 seconds

## BFS NS instance: NS_A2_B2_A2_B1_A1_B1

### Backsubstitution after applying NS history:
0: -5.7942882, 16.2022667, -6.1399083, 17.0795326, -22.8738213, 22.3421726
1: -14.4589930, 24.9580421, -15.3260279, 26.1677895, -40.6267815, 40.2840691
2: -10.0880079, 22.5631218, -10.6860876, 23.7735748, -33.8615837, 33.2492027
3: -15.5135593, 27.5783768, -16.5234947, 28.9655800, -44.4791374, 44.1018715
4: -14.0118141, 27.7913380, -15.0624628, 29.3566284, -43.3684425, 42.8537941

Time for backsubstitution: 1.01 seconds

### NS candidates at layer 1
type: A, layer: 1, pos: 9
type: A, layer: 1, pos: 39
type: A, layer: 1, pos: 48
type: A, layer: 1, pos: 6
type: A, layer: 1, pos: 28
type: A, layer: 1, pos: 38
type: A, layer: 1, pos: 0
type: A, layer: 1, pos: 41
type: A, layer: 1, pos: 31
type: A, layer: 1, pos: 37
type: A, layer: 1, pos: 20
type: A, layer: 1, pos: 8
type: A, layer: 1, pos: 24
type: A, layer: 1, pos: 15
type: A, layer: 1, pos: 45
type: A, layer: 1, pos: 26
type: A, layer: 1, pos: 5

Time for candidate selection: 0.07 seconds

### Candidate
type: A, layer: 1, pos: 9

## Relational analysis of NS_A2_B2_A2_B1_A1_B1_A1
Optimization infeasible because this subproblem isn't reachable.

### Candidate
type: A, layer: 1, pos: 39

## Relational analysis of NS_A2_B2_A2_B1_A1_B1_A1
Optimization infeasible because this subproblem isn't reachable.

### Candidate
type: A, layer: 1, pos: 48

## Relational analysis of NS_A2_B2_A2_B1_A1_B1_A1
Optimization infeasible because this subproblem isn't reachable.

### Candidate
type: A, layer: 1, pos: 6

## Relational analysis of NS_A2_B2_A2_B1_A1_B1_A1
Optimization infeasible because this subproblem isn't reachable.

### Candidate
type: A, layer: 1, pos: 28

## Relational analysis of NS_A2_B2_A2_B1_A1_B1_A1
Optimization infeasible because this subproblem isn't reachable.

### Candidate
type: A, layer: 1, pos: 38

## Relational analysis of NS_A2_B2_A2_B1_A1_B1_A1
Optimization infeasible because this subproblem isn't reachable.

### Candidate
type: A, layer: 1, pos: 0

## Relational analysis of NS_A2_B2_A2_B1_A1_B1_A1

### Relational analysis result of NS_A2_B2_A2_B1_A1_B1_A1
Status: Status.UNKNOWN
Output dim: 0, lower bound: -20.9273920, upper bound: 20.9403716
time: 1.05 seconds

## Relational analysis of NS_A2_B2_A2_B1_A1_B1_A2

### Relational analysis result of NS_A2_B2_A2_B1_A1_B1_A2
Status: Status.UNKNOWN
Output dim: 0, lower bound: -20.9264154, upper bound: 20.9384249
time: 0.77 seconds

## BFS NS instance: NS_A2_B2_A2_B1_A1_B2

### Backsubstitution after applying NS history:
0: -5.7418823, 16.0635509, -6.1136212, 16.9824333, -22.7243156, 22.1771660
1: -14.3273554, 24.7510281, -15.3377342, 25.9090118, -40.2363625, 40.0887604
2: -9.9958878, 22.3741817, -10.6715889, 23.6057777, -33.6016655, 33.0457687
3: -15.3737555, 27.3469887, -16.4816246, 28.7179699, -44.0917244, 43.8286133
4: -13.8832531, 27.5586777, -14.9449558, 29.1793976, -43.0626526, 42.5036240

Time for backsubstitution: 1.02 seconds

### NS candidates at layer 1
type: A, layer: 1, pos: 39
type: A, layer: 1, pos: 9
type: A, layer: 1, pos: 48
type: A, layer: 1, pos: 6
type: A, layer: 1, pos: 38
type: A, layer: 1, pos: 28
type: A, layer: 1, pos: 0
type: A, layer: 1, pos: 41
type: A, layer: 1, pos: 31
type: A, layer: 1, pos: 37
type: A, layer: 1, pos: 20
type: A, layer: 1, pos: 8
type: A, layer: 1, pos: 24
type: A, layer: 1, pos: 15
type: A, layer: 1, pos: 45
type: A, layer: 1, pos: 5
type: A, layer: 1, pos: 26

Time for candidate selection: 0.07 seconds

### Candidate
type: A, layer: 1, pos: 39

## Relational analysis of NS_A2_B2_A2_B1_A1_B2_A1
Optimization infeasible because this subproblem isn't reachable.

### Candidate
type: A, layer: 1, pos: 9

## Relational analysis of NS_A2_B2_A2_B1_A1_B2_A1

### Relational analysis result of NS_A2_B2_A2_B1_A1_B2_A1
Status: Status.UNKNOWN
Output dim: 0, lower bound: -20.9235642, upper bound: 20.9273867
time: 0.76 seconds

## Relational analysis of NS_A2_B2_A2_B1_A1_B2_A2

### Relational analysis result of NS_A2_B2_A2_B1_A1_B2_A2
Status: Status.UNKNOWN
Output dim: 0, lower bound: -20.9235642, upper bound: 20.9521644
time: 1.14 seconds

## BFS NS instance: NS_A2_B2_A2_B1_A2_B1

### Backsubstitution after applying NS history:
0: -5.9756260, 16.6119061, -6.1399083, 17.0795326, -23.0551586, 22.7518139
1: -14.7459917, 25.6773434, -15.3260279, 26.1677895, -40.9137802, 41.0033722
2: -10.3569841, 23.1951981, -10.6860876, 23.7735748, -34.1305580, 33.8812790
3: -15.9492168, 28.3532829, -16.5234947, 28.9655800, -44.9147949, 44.8767776
4: -14.6166353, 28.4932518, -15.0624628, 29.3566284, -43.9732552, 43.5557137

Time for backsubstitution: 1.03 seconds

### NS candidates at layer 1
type: A, layer: 1, pos: 39
type: A, layer: 1, pos: 9
type: A, layer: 1, pos: 6
type: A, layer: 1, pos: 48
type: A, layer: 1, pos: 28
type: A, layer: 1, pos: 0
type: A, layer: 1, pos: 38
type: A, layer: 1, pos: 24
type: A, layer: 1, pos: 37
type: A, layer: 1, pos: 41
type: A, layer: 1, pos: 8
type: A, layer: 1, pos: 31
type: A, layer: 1, pos: 20
type: A, layer: 1, pos: 15
type: A, layer: 1, pos: 26
type: A, layer: 1, pos: 5
type: A, layer: 1, pos: 45

Time for candidate selection: 0.07 seconds

### Candidate
type: A, layer: 1, pos: 39

## Relational analysis of NS_A2_B2_A2_B1_A2_B1_A1
Optimization infeasible because this subproblem isn't reachable.

### Candidate
type: A, layer: 1, pos: 9

## Relational analysis of NS_A2_B2_A2_B1_A2_B1_A1

### Relational analysis result of NS_A2_B2_A2_B1_A2_B1_A1
Status: Status.UNKNOWN
Output dim: 0, lower bound: -20.9238226, upper bound: 20.9278839
time: 0.80 seconds

## Relational analysis of NS_A2_B2_A2_B1_A2_B1_A2

### Relational analysis result of NS_A2_B2_A2_B1_A2_B1_A2
Status: Status.UNKNOWN
Output dim: 0, lower bound: -20.9238226, upper bound: 20.9302789
time: 0.70 seconds

## BFS NS instance: NS_A2_B2_A2_B1_A2_B2

### Backsubstitution after applying NS history:
0: -5.9157062, 16.4576702, -6.1136212, 16.9824333, -22.8981400, 22.5712891
1: -14.5983000, 25.4474297, -15.3377342, 25.9090118, -40.5073090, 40.7851639
2: -10.2526245, 22.9847393, -10.6715889, 23.6057777, -33.8584023, 33.6563263
3: -15.7897406, 28.0960331, -16.4816246, 28.7179699, -44.5077057, 44.5776596
4: -14.4685001, 28.2365131, -14.9449558, 29.1793976, -43.6478958, 43.1814690

Time for backsubstitution: 1.02 seconds

### NS candidates at layer 1
type: A, layer: 1, pos: 39
type: A, layer: 1, pos: 6
type: A, layer: 1, pos: 9
type: A, layer: 1, pos: 48
type: A, layer: 1, pos: 28
type: A, layer: 1, pos: 0
type: A, layer: 1, pos: 38
type: A, layer: 1, pos: 24
type: A, layer: 1, pos: 37
type: A, layer: 1, pos: 41
type: A, layer: 1, pos: 20
type: A, layer: 1, pos: 31
type: A, layer: 1, pos: 8
type: A, layer: 1, pos: 15
type: A, layer: 1, pos: 26
type: A, layer: 1, pos: 5
type: A, layer: 1, pos: 45

Time for candidate selection: 0.07 seconds

### Candidate
type: A, layer: 1, pos: 39

## Relational analysis of NS_A2_B2_A2_B1_A2_B2_A1
Optimization infeasible because this subproblem isn't reachable.

### Candidate
type: A, layer: 1, pos: 6

## Relational analysis of NS_A2_B2_A2_B1_A2_B2_A1
Optimization infeasible because this subproblem isn't reachable.

### Candidate
type: A, layer: 1, pos: 9

## Relational analysis of NS_A2_B2_A2_B1_A2_B2_A1

### Relational analysis result of NS_A2_B2_A2_B1_A2_B2_A1
Status: Status.UNKNOWN
Output dim: 0, lower bound: -20.9288968, upper bound: 20.9278839
time: 0.69 seconds

## Relational analysis of NS_A2_B2_A2_B1_A2_B2_A2

### Relational analysis result of NS_A2_B2_A2_B1_A2_B2_A2
Status: Status.UNKNOWN
Output dim: 0, lower bound: -20.9288968, upper bound: 20.9404709
time: 0.67 seconds

## BFS NS instance: NS_A2_B2_A2_B2_A1_B1

### Backsubstitution after applying NS history:
0: -5.7942882, 16.2022667, -5.9458299, 16.6518002, -22.4460888, 22.1480942
1: -14.4589930, 24.9580421, -14.7555151, 25.7168369, -40.1758308, 39.7135582
2: -10.0880079, 22.5631218, -10.3130445, 23.2755966, -33.3635979, 32.8761673
3: -15.5135593, 27.5783768, -15.9753675, 28.3708611, -43.8844109, 43.5537453
4: -14.0118141, 27.7913380, -14.6584167, 28.6986294, -42.7104416, 42.4497528

Time for backsubstitution: 1.03 seconds

### NS candidates at layer 1
type: A, layer: 1, pos: 9
type: A, layer: 1, pos: 39
type: A, layer: 1, pos: 48
type: A, layer: 1, pos: 6
type: A, layer: 1, pos: 28
type: A, layer: 1, pos: 38
type: A, layer: 1, pos: 0
type: A, layer: 1, pos: 41
type: A, layer: 1, pos: 31
type: A, layer: 1, pos: 37
type: A, layer: 1, pos: 20
type: A, layer: 1, pos: 8
type: A, layer: 1, pos: 24
type: A, layer: 1, pos: 15
type: A, layer: 1, pos: 45
type: A, layer: 1, pos: 26
type: A, layer: 1, pos: 5

Time for candidate selection: 0.07 seconds

### Candidate
type: A, layer: 1, pos: 9

## Relational analysis of NS_A2_B2_A2_B2_A1_B1_A1
Optimization infeasible because this subproblem isn't reachable.

### Candidate
type: A, layer: 1, pos: 39

## Relational analysis of NS_A2_B2_A2_B2_A1_B1_A1

### Relational analysis result of NS_A2_B2_A2_B2_A1_B1_A1
Status: Status.UNKNOWN
Output dim: 0, lower bound: -20.9483524, upper bound: 20.9496922
time: 0.58 seconds

## Relational analysis of NS_A2_B2_A2_B2_A1_B1_A2

### Relational analysis result of NS_A2_B2_A2_B2_A1_B1_A2
Status: Status.UNKNOWN
Output dim: 0, lower bound: -20.9532529, upper bound: 20.9553064
time: 0.76 seconds

## BFS NS instance: NS_A2_B2_A2_B2_A1_B2

### Backsubstitution after applying NS history:
0: -5.7942882, 16.2022667, -5.9213266, 16.4778671, -22.2721558, 22.1235924
1: -14.4589930, 24.9580421, -14.6102200, 25.4814663, -39.9404488, 39.5682564
2: -10.0880079, 22.5631218, -10.2608318, 23.0094490, -33.0974503, 32.8239479
3: -15.5135593, 27.5783768, -15.8002529, 28.1255474, -43.6390991, 43.3786316
4: -14.0118141, 27.7913380, -14.4836226, 28.2687950, -42.2806091, 42.2749596

Time for backsubstitution: 1.03 seconds

### NS candidates at layer 1
type: A, layer: 1, pos: 9
type: A, layer: 1, pos: 39
type: A, layer: 1, pos: 48
type: A, layer: 1, pos: 6
type: A, layer: 1, pos: 28
type: A, layer: 1, pos: 38
type: A, layer: 1, pos: 0
type: A, layer: 1, pos: 41
type: A, layer: 1, pos: 31
type: A, layer: 1, pos: 37
type: A, layer: 1, pos: 20
type: A, layer: 1, pos: 8
type: A, layer: 1, pos: 24
type: A, layer: 1, pos: 15
type: A, layer: 1, pos: 45
type: A, layer: 1, pos: 26
type: A, layer: 1, pos: 5

Time for candidate selection: 0.07 seconds

### Candidate
type: A, layer: 1, pos: 9

## Relational analysis of NS_A2_B2_A2_B2_A1_B2_A1
Optimization infeasible because this subproblem isn't reachable.

### Candidate
type: A, layer: 1, pos: 39

## Relational analysis of NS_A2_B2_A2_B2_A1_B2_A1

### Relational analysis result of NS_A2_B2_A2_B2_A1_B2_A1
Status: Status.UNKNOWN
Output dim: 0, lower bound: -20.9483524, upper bound: 20.9496922
time: 0.61 seconds

## Relational analysis of NS_A2_B2_A2_B2_A1_B2_A2

### Relational analysis result of NS_A2_B2_A2_B2_A1_B2_A2
Status: Status.UNKNOWN
Output dim: 0, lower bound: -20.9532529, upper bound: 20.9553064
time: 0.79 seconds

## BFS NS instance: NS_A2_B2_A2_B2_A2_B1

### Backsubstitution after applying NS history:
0: -5.9756260, 16.6119061, -5.9954491, 16.7857399, -22.7613659, 22.6073551
1: -14.7459917, 25.6773434, -14.9325676, 25.8895206, -40.6355095, 40.6099091
2: -10.3569841, 23.1951981, -10.4139805, 23.4385681, -33.7955513, 33.6091766
3: -15.9492168, 28.3532829, -16.1189556, 28.5748539, -44.5240707, 44.4722290
4: -14.6166353, 28.4932518, -14.7187843, 28.9228477, -43.5394821, 43.2120361

Time for backsubstitution: 1.04 seconds

### NS candidates at layer 1
type: A, layer: 1, pos: 39
type: A, layer: 1, pos: 9
type: A, layer: 1, pos: 6
type: A, layer: 1, pos: 48
type: A, layer: 1, pos: 28
type: A, layer: 1, pos: 0
type: A, layer: 1, pos: 38
type: A, layer: 1, pos: 24
type: A, layer: 1, pos: 37
type: A, layer: 1, pos: 41
type: A, layer: 1, pos: 8
type: A, layer: 1, pos: 31
type: A, layer: 1, pos: 20
type: A, layer: 1, pos: 15
type: A, layer: 1, pos: 26
type: A, layer: 1, pos: 5
type: A, layer: 1, pos: 45

Time for candidate selection: 0.07 seconds

### Candidate
type: A, layer: 1, pos: 39

## Relational analysis of NS_A2_B2_A2_B2_A2_B1_A1
Optimization infeasible because this subproblem isn't reachable.

### Candidate
type: A, layer: 1, pos: 9

## Relational analysis of NS_A2_B2_A2_B2_A2_B1_A1

### Relational analysis result of NS_A2_B2_A2_B2_A2_B1_A1
Status: Status.UNKNOWN
Output dim: 0, lower bound: -20.9391816, upper bound: 20.9391312
time: 0.78 seconds

## Relational analysis of NS_A2_B2_A2_B2_A2_B1_A2

### Relational analysis result of NS_A2_B2_A2_B2_A2_B1_A2
Status: Status.UNKNOWN
Output dim: 0, lower bound: -20.9391816, upper bound: 20.9423844
time: 0.80 seconds

## BFS NS instance: NS_A2_B2_A2_B2_A2_B2

### Backsubstitution after applying NS history:
0: -5.9157062, 16.4576702, -5.8584266, 16.3653088, -22.2810154, 22.3160954
1: -14.5983000, 25.4474297, -14.6261539, 25.1673660, -39.7656631, 40.0735817
2: -10.2526245, 22.9847393, -10.1949749, 22.8333759, -33.0859985, 33.1797066
3: -15.7897406, 28.0960331, -15.7621784, 27.8196697, -43.6094017, 43.8582115
4: -14.4685001, 28.2365131, -14.3403988, 28.1881580, -42.6566582, 42.5769119

Time for backsubstitution: 1.04 seconds

### NS candidates at layer 1
type: A, layer: 1, pos: 39
type: A, layer: 1, pos: 6
type: A, layer: 1, pos: 9
type: A, layer: 1, pos: 48
type: A, layer: 1, pos: 28
type: A, layer: 1, pos: 0
type: A, layer: 1, pos: 38
type: A, layer: 1, pos: 24
type: A, layer: 1, pos: 37
type: A, layer: 1, pos: 41
type: A, layer: 1, pos: 20
type: A, layer: 1, pos: 31
type: A, layer: 1, pos: 8
type: A, layer: 1, pos: 15
type: A, layer: 1, pos: 26
type: A, layer: 1, pos: 5
type: A, layer: 1, pos: 45

Time for candidate selection: 0.07 seconds

### Candidate
type: A, layer: 1, pos: 39

## Relational analysis of NS_A2_B2_A2_B2_A2_B2_A1
Optimization infeasible because this subproblem isn't reachable.

### Candidate
type: A, layer: 1, pos: 6

## Relational analysis of NS_A2_B2_A2_B2_A2_B2_A1
Optimization infeasible because this subproblem isn't reachable.

### Candidate
type: A, layer: 1, pos: 9

## Relational analysis of NS_A2_B2_A2_B2_A2_B2_A1

### Relational analysis result of NS_A2_B2_A2_B2_A2_B2_A1
Status: Status.UNKNOWN
Output dim: 0, lower bound: -20.9443077, upper bound: 20.9423952
time: 0.91 seconds

## Relational analysis of NS_A2_B2_A2_B2_A2_B2_A2

### Relational analysis result of NS_A2_B2_A2_B2_A2_B2_A2
Status: Status.UNKNOWN
Output dim: 0, lower bound: -20.9443077, upper bound: 20.9475028
time: 0.67 seconds

## Summary of splitting at layer (split count: 6)
- Time for NS candidates: 3.26 seconds
NS_A1_B1_A1_B2_A2_B1_A1, status: Status.UNKNOWN, split count: 7, time: 3.26
Output dim: 0, lower bound: -20.9059363, upper bound: 20.9232720
NS_A1_B1_A1_B2_A2_B1_A2, status: Status.UNKNOWN, split count: 7, time: 3.26
Output dim: 0, lower bound: -20.9059363, upper bound: 20.9411884
NS_A1_B1_A1_B2_A2_B2_A1, status: Status.UNKNOWN, split count: 7, time: 3.26
Output dim: 0, lower bound: -20.9059363, upper bound: 20.9460725
NS_A1_B1_A1_B2_A2_B2_A2, status: Status.UNKNOWN, split count: 7, time: 3.26
Output dim: 0, lower bound: -20.9059363, upper bound: 20.9523051
NS_A1_B1_A2_B1_A1_B2_A1, status: Status.UNKNOWN, split count: 7, time: 3.26
Output dim: 0, lower bound: -20.9548550, upper bound: 20.9501143
NS_A1_B1_A2_B1_A1_B2_A2, status: Status.UNKNOWN, split count: 7, time: 3.26
Output dim: 0, lower bound: -20.9548550, upper bound: 20.9564671
NS_A1_B1_A2_B1_A2_B2_A1, status: Status.UNKNOWN, split count: 7, time: 3.26
Output dim: 0, lower bound: -20.9551558, upper bound: 20.9505436
NS_A1_B1_A2_B1_A2_B2_A2, status: Status.UNKNOWN, split count: 7, time: 3.26
Output dim: 0, lower bound: -20.9551558, upper bound: 20.9549741
NS_A1_B1_A2_B2_A1_B1_A1, status: Status.UNKNOWN, split count: 7, time: 3.26
Output dim: 0, lower bound: -20.8950918, upper bound: 20.8976329
NS_A1_B1_A2_B2_A1_B1_A2, status: Status.UNKNOWN, split count: 7, time: 3.26
Output dim: 0, lower bound: -20.8950918, upper bound: 20.9155805
NS_A1_B1_A2_B2_A1_B2_A1, status: Status.UNKNOWN, split count: 7, time: 3.26
Output dim: 0, lower bound: -20.8950918, upper bound: 20.9101592
NS_A1_B1_A2_B2_A1_B2_A2, status: Status.UNKNOWN, split count: 7, time: 3.26
Output dim: 0, lower bound: -20.8950918, upper bound: 20.9248678
NS_A1_B1_A2_B2_A2_B1_A1, status: Status.UNKNOWN, split count: 7, time: 3.26
Output dim: 0, lower bound: -20.9138254, upper bound: 20.9189541
NS_A1_B1_A2_B2_A2_B1_A2, status: Status.UNKNOWN, split count: 7, time: 3.26
Output dim: 0, lower bound: -20.9063213, upper bound: 20.9100199
NS_A1_B1_A2_B2_A2_B2_A1, status: Status.UNKNOWN, split count: 7, time: 3.26
Output dim: 0, lower bound: -20.9138254, upper bound: 20.9435253
NS_A1_B1_A2_B2_A2_B2_A2, status: Status.UNKNOWN, split count: 7, time: 3.26
Output dim: 0, lower bound: -20.9063213, upper bound: 20.9427230
NS_A1_B2_A1_B1_A1_B1_A1, status: Status.UNKNOWN, split count: 7, time: 3.26
Output dim: 0, lower bound: -20.9576311, upper bound: 20.9574376
NS_A1_B2_A1_B1_A1_B1_A2, status: Status.UNKNOWN, split count: 7, time: 3.26
Output dim: 0, lower bound: -20.9607899, upper bound: 20.9623820
NS_A1_B2_A1_B1_A1_B2_A1, status: Status.UNKNOWN, split count: 7, time: 3.26
Output dim: 0, lower bound: -20.9576311, upper bound: 20.9574376
NS_A1_B2_A1_B1_A1_B2_A2, status: Status.UNKNOWN, split count: 7, time: 3.26
Output dim: 0, lower bound: -20.9607899, upper bound: 20.9623820
NS_A1_B2_A1_B1_A2_B1_A1, status: Status.UNKNOWN, split count: 7, time: 3.26
Output dim: 0, lower bound: -20.9560333, upper bound: 20.9554724
NS_A1_B2_A1_B1_A2_B1_A2, status: Status.UNKNOWN, split count: 7, time: 3.26
Output dim: 0, lower bound: -20.9604393, upper bound: 20.9612852
NS_A1_B2_A1_B1_A2_B2_A1, status: Status.UNKNOWN, split count: 7, time: 3.26
Output dim: 0, lower bound: -20.9560333, upper bound: 20.9554724
NS_A1_B2_A1_B1_A2_B2_A2, status: Status.UNKNOWN, split count: 7, time: 3.26
Output dim: 0, lower bound: -20.9604393, upper bound: 20.9612852
NS_A1_B2_A1_B2_A1_B1_A1, status: Status.UNKNOWN, split count: 7, time: 3.26
Output dim: 0, lower bound: -20.9492798, upper bound: 20.9500829
NS_A1_B2_A1_B2_A1_B1_A2, status: Status.UNKNOWN, split count: 7, time: 3.26
Output dim: 0, lower bound: -20.9492798, upper bound: 20.9562324
NS_A1_B2_A1_B2_A1_B2_A1, status: Status.UNKNOWN, split count: 7, time: 3.26
Output dim: 0, lower bound: -20.9492798, upper bound: 20.9500829
NS_A1_B2_A1_B2_A1_B2_A2, status: Status.UNKNOWN, split count: 7, time: 3.26
Output dim: 0, lower bound: -20.9492798, upper bound: 20.9563050
NS_A1_B2_A1_B2_A2_B1_A1, status: Status.UNKNOWN, split count: 7, time: 3.26
Output dim: 0, lower bound: -20.9485935, upper bound: 20.9490286
NS_A1_B2_A1_B2_A2_B1_A2, status: Status.UNKNOWN, split count: 7, time: 3.26
Output dim: 0, lower bound: -20.9537253, upper bound: 20.9553052
NS_A1_B2_A1_B2_A2_B2_A1, status: Status.UNKNOWN, split count: 7, time: 3.26
Output dim: 0, lower bound: -20.9526143, upper bound: 20.9541293
NS_A1_B2_A1_B2_A2_B2_A2, status: Status.UNKNOWN, split count: 7, time: 3.26
Output dim: 0, lower bound: -20.9524777, upper bound: 20.9544091
NS_A1_B2_A2_B1_A1_B1_A1, status: Status.UNKNOWN, split count: 7, time: 3.26
Output dim: 0, lower bound: -20.9576419, upper bound: 20.9574161
NS_A1_B2_A2_B1_A1_B1_A2, status: Status.UNKNOWN, split count: 7, time: 3.26
Output dim: 0, lower bound: -20.9522927, upper bound: 20.9519686
NS_A1_B2_A2_B1_A1_B2_A1, status: Status.UNKNOWN, split count: 7, time: 3.26
Output dim: 0, lower bound: -20.9604218, upper bound: 20.9609168
NS_A1_B2_A2_B1_A1_B2_A2, status: Status.UNKNOWN, split count: 7, time: 3.26
Output dim: 0, lower bound: -20.9572772, upper bound: 20.9563436
NS_A1_B2_A2_B1_A2_B1_A1, status: Status.UNKNOWN, split count: 7, time: 3.26
Output dim: 0, lower bound: -20.9604218, upper bound: 20.9608962
NS_A1_B2_A2_B1_A2_B1_A2, status: Status.UNKNOWN, split count: 7, time: 3.26
Output dim: 0, lower bound: -20.9572772, upper bound: 20.9608789
NS_A1_B2_A2_B1_A2_B2_A1, status: Status.UNKNOWN, split count: 7, time: 3.26
Output dim: 0, lower bound: -20.9606378, upper bound: 20.9608962
NS_A1_B2_A2_B1_A2_B2_A2, status: Status.UNKNOWN, split count: 7, time: 3.26
Output dim: 0, lower bound: -20.9572772, upper bound: 20.9608789
NS_A1_B2_A2_B2_A1_B1_A1, status: Status.UNKNOWN, split count: 7, time: 3.26
Output dim: 0, lower bound: -20.9427099, upper bound: 20.9393656
NS_A1_B2_A2_B2_A1_B1_A2, status: Status.UNKNOWN, split count: 7, time: 3.26
Output dim: 0, lower bound: -20.9440769, upper bound: 20.9409453
NS_A1_B2_A2_B2_A1_B2_A1, status: Status.UNKNOWN, split count: 7, time: 3.26
Output dim: 0, lower bound: -20.9402842, upper bound: 20.9344529
NS_A1_B2_A2_B2_A1_B2_A2, status: Status.UNKNOWN, split count: 7, time: 3.26
Output dim: 0, lower bound: -20.9439913, upper bound: 20.9409624
NS_A1_B2_A2_B2_A2_B1_A1, status: Status.UNKNOWN, split count: 7, time: 3.26
Output dim: 0, lower bound: -20.9498795, upper bound: 20.9501715
NS_A1_B2_A2_B2_A2_B1_A2, status: Status.UNKNOWN, split count: 7, time: 3.26
Output dim: 0, lower bound: -20.9504496, upper bound: 20.9508051
NS_A1_B2_A2_B2_A2_B2_A1, status: Status.UNKNOWN, split count: 7, time: 3.26
Output dim: 0, lower bound: -20.9498795, upper bound: 20.9501715
NS_A1_B2_A2_B2_A2_B2_A2, status: Status.UNKNOWN, split count: 7, time: 3.26
Output dim: 0, lower bound: -20.9504496, upper bound: 20.9508051
NS_A2_B1_A1_B1_A1_B2_A1, status: Status.UNKNOWN, split count: 7, time: 3.26
Output dim: 0, lower bound: -20.9548567, upper bound: 20.9552630
NS_A2_B1_A1_B1_A1_B2_A2, status: Status.UNKNOWN, split count: 7, time: 3.26
Output dim: 0, lower bound: -20.9640737, upper bound: 20.9649475
NS_A2_B1_A1_B1_A2_B1_A1, status: Status.UNKNOWN, split count: 7, time: 3.26
Output dim: 0, lower bound: -20.9341830, upper bound: 20.9224596
NS_A2_B1_A1_B1_A2_B1_A2, status: Status.UNKNOWN, split count: 7, time: 3.26
Output dim: 0, lower bound: -20.9341830, upper bound: 20.9455978
NS_A2_B1_A1_B1_A2_B2_A1, status: Status.UNKNOWN, split count: 7, time: 3.26
Output dim: 0, lower bound: -20.9411884, upper bound: 20.9270553
NS_A2_B1_A1_B1_A2_B2_A2, status: Status.UNKNOWN, split count: 7, time: 3.26
Output dim: 0, lower bound: -20.9411884, upper bound: 20.9488332
NS_A2_B1_A1_B2_A1_B1_A1, status: Status.UNKNOWN, split count: 7, time: 3.26
Output dim: 0, lower bound: -20.9303604, upper bound: 20.9425656
NS_A2_B1_A1_B2_A1_B1_A2, status: Status.UNKNOWN, split count: 7, time: 3.26
Output dim: 0, lower bound: -20.9441063, upper bound: 20.9501995
NS_A2_B1_A1_B2_A1_B2_A1, status: Status.UNKNOWN, split count: 7, time: 3.26
Output dim: 0, lower bound: -20.9570587, upper bound: 20.9601795
NS_A2_B1_A1_B2_A1_B2_A2, status: Status.UNKNOWN, split count: 7, time: 3.26
Output dim: 0, lower bound: -20.9441063, upper bound: 20.9640981
NS_A2_B1_A1_B2_A2_B1_A1, status: Status.UNKNOWN, split count: 7, time: 3.26
Output dim: 0, lower bound: -20.9273867, upper bound: 20.9229127
NS_A2_B1_A1_B2_A2_B1_A2, status: Status.UNKNOWN, split count: 7, time: 3.26
Output dim: 0, lower bound: -20.9273867, upper bound: 20.9234893
NS_A2_B1_A1_B2_A2_B2_A1, status: Status.UNKNOWN, split count: 7, time: 3.26
Output dim: 0, lower bound: -20.9440816, upper bound: 20.9339440
NS_A2_B1_A1_B2_A2_B2_A2, status: Status.UNKNOWN, split count: 7, time: 3.26
Output dim: 0, lower bound: -20.9440816, upper bound: 20.9477517
NS_A2_B1_A2_B1_A1_B1_A1, status: Status.UNKNOWN, split count: 7, time: 3.26
Output dim: 0, lower bound: -20.9614214, upper bound: 20.9618822
NS_A2_B1_A2_B1_A1_B1_A2, status: Status.UNKNOWN, split count: 7, time: 3.26
Output dim: 0, lower bound: -20.9614214, upper bound: 20.9618822
NS_A2_B1_A2_B1_A1_B2_A1, status: Status.UNKNOWN, split count: 7, time: 3.26
Output dim: 0, lower bound: -20.9614214, upper bound: 20.9618822
NS_A2_B1_A2_B1_A1_B2_A2, status: Status.UNKNOWN, split count: 7, time: 3.26
Output dim: 0, lower bound: -20.9614214, upper bound: 20.9618822
NS_A2_B1_A2_B1_A2_B1_A1, status: Status.UNKNOWN, split count: 7, time: 3.26
Output dim: 0, lower bound: -20.9549159, upper bound: 20.9534221
NS_A2_B1_A2_B1_A2_B1_A2, status: Status.UNKNOWN, split count: 7, time: 3.26
Output dim: 0, lower bound: -20.9549159, upper bound: 20.9537253
NS_A2_B1_A2_B1_A2_B2_A1, status: Status.UNKNOWN, split count: 7, time: 3.26
Output dim: 0, lower bound: -20.9549159, upper bound: 20.9534221
NS_A2_B1_A2_B1_A2_B2_A2, status: Status.UNKNOWN, split count: 7, time: 3.26
Output dim: 0, lower bound: -20.9549159, upper bound: 20.9537253
NS_A2_B1_A2_B2_A1_B1_A1, status: Status.UNKNOWN, split count: 7, time: 3.26
Output dim: 0, lower bound: -20.9489641, upper bound: 20.9434778
NS_A2_B1_A2_B2_A1_B1_A2, status: Status.UNKNOWN, split count: 7, time: 3.26
Output dim: 0, lower bound: -20.9642419, upper bound: 20.9640326
NS_A2_B1_A2_B2_A1_B2_A1, status: Status.UNKNOWN, split count: 7, time: 3.26
Output dim: 0, lower bound: -20.9525391, upper bound: 20.9614829
NS_A2_B1_A2_B2_A1_B2_A2, status: Status.UNKNOWN, split count: 7, time: 3.26
Output dim: 0, lower bound: -20.9524133, upper bound: 20.9614214
NS_A2_B1_A2_B2_A2_B1_A1, status: Status.UNKNOWN, split count: 7, time: 3.26
Output dim: 0, lower bound: -20.9486676, upper bound: 20.9411943
NS_A2_B1_A2_B2_A2_B1_A2, status: Status.UNKNOWN, split count: 7, time: 3.26
Output dim: 0, lower bound: -20.9551491, upper bound: 20.9511649
NS_A2_B1_A2_B2_A2_B2_A1, status: Status.UNKNOWN, split count: 7, time: 3.26
Output dim: 0, lower bound: -20.9486676, upper bound: 20.9489766
NS_A2_B1_A2_B2_A2_B2_A2, status: Status.UNKNOWN, split count: 7, time: 3.26
Output dim: 0, lower bound: -20.9551491, upper bound: 20.9541027
NS_A2_B2_A1_B1_A1_B1_A1, status: Status.UNKNOWN, split count: 7, time: 3.26
Output dim: 0, lower bound: -20.9527657, upper bound: 20.9572021
NS_A2_B2_A1_B1_A1_B1_A2, status: Status.UNKNOWN, split count: 7, time: 3.26
Output dim: 0, lower bound: -20.9564671, upper bound: 20.9602771
NS_A2_B2_A1_B1_A1_B2_A1, status: Status.UNKNOWN, split count: 7, time: 3.26
Output dim: 0, lower bound: -20.9520676, upper bound: 20.9565773
NS_A2_B2_A1_B1_A1_B2_A2, status: Status.UNKNOWN, split count: 7, time: 3.26
Output dim: 0, lower bound: -20.9549741, upper bound: 20.9584735
NS_A2_B2_A1_B1_A2_B1_A1, status: Status.UNKNOWN, split count: 7, time: 3.26
Output dim: 0, lower bound: -20.9315604, upper bound: 20.9203569
NS_A2_B2_A1_B1_A2_B1_A2, status: Status.UNKNOWN, split count: 7, time: 3.26
Output dim: 0, lower bound: -20.9315604, upper bound: 20.9423008
NS_A2_B2_A1_B1_A2_B2_A1, status: Status.UNKNOWN, split count: 7, time: 3.26
Output dim: 0, lower bound: -20.9315604, upper bound: 20.9203569
NS_A2_B2_A1_B1_A2_B2_A2, status: Status.UNKNOWN, split count: 7, time: 3.26
Output dim: 0, lower bound: -20.9315604, upper bound: 20.9423008
NS_A2_B2_A1_B2_A1_B1_A1, status: Status.UNKNOWN, split count: 7, time: 3.26
Output dim: 0, lower bound: -20.9234777, upper bound: 20.9244704
NS_A2_B2_A1_B2_A1_B1_A2, status: Status.UNKNOWN, split count: 7, time: 3.26
Output dim: 0, lower bound: -20.9467965, upper bound: 20.9518643
NS_A2_B2_A1_B2_A1_B2_A1, status: Status.UNKNOWN, split count: 7, time: 3.26
Output dim: 0, lower bound: -20.9234777, upper bound: 20.9285327
NS_A2_B2_A1_B2_A1_B2_A2, status: Status.UNKNOWN, split count: 7, time: 3.26
Output dim: 0, lower bound: -20.9467965, upper bound: 20.9564293
NS_A2_B2_A1_B2_A2_B1_A1, status: Status.UNKNOWN, split count: 7, time: 3.26
Output dim: 0, lower bound: -20.9250963, upper bound: 20.9203729
NS_A2_B2_A1_B2_A2_B1_A2, status: Status.UNKNOWN, split count: 7, time: 3.26
Output dim: 0, lower bound: -20.9352853, upper bound: 20.9352838
NS_A2_B2_A1_B2_A2_B2_A1, status: Status.UNKNOWN, split count: 7, time: 3.26
Output dim: 0, lower bound: -20.9250963, upper bound: 20.9251292
NS_A2_B2_A1_B2_A2_B2_A2, status: Status.UNKNOWN, split count: 7, time: 3.26
Output dim: 0, lower bound: -20.9352853, upper bound: 20.9393947
NS_A2_B2_A2_B1_A1_B1_A1, status: Status.UNKNOWN, split count: 7, time: 3.26
Output dim: 0, lower bound: -20.9273920, upper bound: 20.9403716
NS_A2_B2_A2_B1_A1_B1_A2, status: Status.UNKNOWN, split count: 7, time: 3.26
Output dim: 0, lower bound: -20.9264154, upper bound: 20.9384249
NS_A2_B2_A2_B1_A1_B2_A1, status: Status.UNKNOWN, split count: 7, time: 3.26
Output dim: 0, lower bound: -20.9235642, upper bound: 20.9273867
NS_A2_B2_A2_B1_A1_B2_A2, status: Status.UNKNOWN, split count: 7, time: 3.26
Output dim: 0, lower bound: -20.9235642, upper bound: 20.9521644
NS_A2_B2_A2_B1_A2_B1_A1, status: Status.UNKNOWN, split count: 7, time: 3.26
Output dim: 0, lower bound: -20.9238226, upper bound: 20.9278839
NS_A2_B2_A2_B1_A2_B1_A2, status: Status.UNKNOWN, split count: 7, time: 3.26
Output dim: 0, lower bound: -20.9238226, upper bound: 20.9302789
NS_A2_B2_A2_B1_A2_B2_A1, status: Status.UNKNOWN, split count: 7, time: 3.26
Output dim: 0, lower bound: -20.9288968, upper bound: 20.9278839
NS_A2_B2_A2_B1_A2_B2_A2, status: Status.UNKNOWN, split count: 7, time: 3.26
Output dim: 0, lower bound: -20.9288968, upper bound: 20.9404709
NS_A2_B2_A2_B2_A1_B1_A1, status: Status.UNKNOWN, split count: 7, time: 3.26
Output dim: 0, lower bound: -20.9483524, upper bound: 20.9496922
NS_A2_B2_A2_B2_A1_B1_A2, status: Status.UNKNOWN, split count: 7, time: 3.26
Output dim: 0, lower bound: -20.9532529, upper bound: 20.9553064
NS_A2_B2_A2_B2_A1_B2_A1, status: Status.UNKNOWN, split count: 7, time: 3.26
Output dim: 0, lower bound: -20.9483524, upper bound: 20.9496922
NS_A2_B2_A2_B2_A1_B2_A2, status: Status.UNKNOWN, split count: 7, time: 3.26
Output dim: 0, lower bound: -20.9532529, upper bound: 20.9553064
NS_A2_B2_A2_B2_A2_B1_A1, status: Status.UNKNOWN, split count: 7, time: 3.26
Output dim: 0, lower bound: -20.9391816, upper bound: 20.9391312
NS_A2_B2_A2_B2_A2_B1_A2, status: Status.UNKNOWN, split count: 7, time: 3.26
Output dim: 0, lower bound: -20.9391816, upper bound: 20.9423844
NS_A2_B2_A2_B2_A2_B2_A1, status: Status.UNKNOWN, split count: 7, time: 3.26
Output dim: 0, lower bound: -20.9443077, upper bound: 20.9423952
NS_A2_B2_A2_B2_A2_B2_A2, status: Status.UNKNOWN, split count: 7, time: 3.26
Output dim: 0, lower bound: -20.9443077, upper bound: 20.9475028

## BFS NS instance: NS_A1_B1_A1_B2_A2_B1_A1

### Backsubstitution after applying NS history:
0: -4.3730445, 12.4010944, -5.5470247, 15.4822569, -19.8553009, 17.9481163
1: -10.6506119, 19.4448338, -13.6963949, 23.8694801, -34.5200844, 33.1412277
2: -7.4423847, 17.4804916, -9.5973005, 21.6184483, -29.0608311, 27.0777931
3: -11.6417799, 21.3417969, -14.8785763, 26.3822765, -38.0240555, 36.2203751
4: -10.7328491, 21.4377575, -13.6805954, 26.6168098, -37.3496513, 35.1183472

Time for backsubstitution: 0.97 seconds

### NS candidates at layer 1
type: B, layer: 1, pos: 12
type: B, layer: 1, pos: 9
type: B, layer: 1, pos: 39
type: B, layer: 1, pos: 6
type: B, layer: 1, pos: 48
type: B, layer: 1, pos: 38
type: B, layer: 1, pos: 28
type: B, layer: 1, pos: 0
type: B, layer: 1, pos: 24
type: B, layer: 1, pos: 41
type: B, layer: 1, pos: 8
type: B, layer: 1, pos: 31
type: B, layer: 1, pos: 26
type: B, layer: 1, pos: 45
type: B, layer: 1, pos: 15
type: B, layer: 1, pos: 20
type: B, layer: 1, pos: 5

Time for candidate selection: 0.07 seconds

### Candidate
type: B, layer: 1, pos: 12

### Candidate
type: B, layer: 1, pos: 9

### Candidate
type: B, layer: 1, pos: 39

### Candidate
type: B, layer: 1, pos: 6

### Candidate
type: B, layer: 1, pos: 48

### Candidate
type: B, layer: 1, pos: 38

### Candidate
type: B, layer: 1, pos: 28

### Candidate
type: B, layer: 1, pos: 0

### Candidate
type: B, layer: 1, pos: 24

### Candidate
type: B, layer: 1, pos: 41

### Candidate
type: B, layer: 1, pos: 8

### Candidate
type: B, layer: 1, pos: 31

## Relational analysis of NS_A1_B1_A1_B2_A2_B1_A1_B1
Optimization infeasible because this subproblem isn't reachable.

### Candidate
type: B, layer: 1, pos: 26

## Relational analysis of NS_A1_B1_A1_B2_A2_B1_A1_B1
Optimization infeasible because this subproblem isn't reachable.

### Candidate
type: B, layer: 1, pos: 45

## Relational analysis of NS_A1_B1_A1_B2_A2_B1_A1_B1
Optimization infeasible because this subproblem isn't reachable.

### Candidate
type: B, layer: 1, pos: 15

## Relational analysis of NS_A1_B1_A1_B2_A2_B1_A1_B1
Optimization infeasible because this subproblem isn't reachable.

### Candidate
type: B, layer: 1, pos: 20

## Relational analysis of NS_A1_B1_A1_B2_A2_B1_A1_B1

### Relational analysis result of NS_A1_B1_A1_B2_A2_B1_A1_B1
Status: Status.UNKNOWN
Output dim: 0, lower bound: -20.8694587, upper bound: 20.8960704
time: 0.60 seconds

## Relational analysis of NS_A1_B1_A1_B2_A2_B1_A1_B2

### Relational analysis result of NS_A1_B1_A1_B2_A2_B1_A1_B2
Status: Status.UNKNOWN
Output dim: 0, lower bound: -20.8694587, upper bound: 20.9232720
time: 0.75 seconds

## BFS NS instance: NS_A1_B1_A1_B2_A2_B1_A2

### Backsubstitution after applying NS history:
0: -5.6191449, 15.7847385, -6.0268006, 16.7735538, -22.3926945, 21.8115387
1: -14.0169811, 24.3333015, -14.9967089, 25.7297859, -39.7467651, 39.3300095
2: -9.7616282, 22.0435867, -10.4796228, 23.3693790, -33.1310081, 32.5232010
3: -15.1224251, 26.8683109, -16.2082367, 28.4722595, -43.5946846, 43.0765457
4: -13.7608461, 27.2130318, -14.8243275, 28.8347683, -42.5956154, 42.0373573

Time for backsubstitution: 0.95 seconds

### NS candidates at layer 1
type: B, layer: 1, pos: 12
type: B, layer: 1, pos: 9
type: B, layer: 1, pos: 6
type: B, layer: 1, pos: 48
type: B, layer: 1, pos: 39
type: B, layer: 1, pos: 28
type: B, layer: 1, pos: 38
type: B, layer: 1, pos: 0
type: B, layer: 1, pos: 24
type: B, layer: 1, pos: 41
type: B, layer: 1, pos: 8
type: B, layer: 1, pos: 31
type: B, layer: 1, pos: 15
type: B, layer: 1, pos: 26
type: B, layer: 1, pos: 45
type: B, layer: 1, pos: 5
type: B, layer: 1, pos: 20

Time for candidate selection: 0.07 seconds

### Candidate
type: B, layer: 1, pos: 12

### Candidate
type: B, layer: 1, pos: 9

### Candidate
type: B, layer: 1, pos: 6

### Candidate
type: B, layer: 1, pos: 48

### Candidate
type: B, layer: 1, pos: 39

### Candidate
type: B, layer: 1, pos: 28

### Candidate
type: B, layer: 1, pos: 38

### Candidate
type: B, layer: 1, pos: 0

### Candidate
type: B, layer: 1, pos: 24

### Candidate
type: B, layer: 1, pos: 41

### Candidate
type: B, layer: 1, pos: 8

### Candidate
type: B, layer: 1, pos: 31

## Relational analysis of NS_A1_B1_A1_B2_A2_B1_A2_B1
Optimization infeasible because this subproblem isn't reachable.

### Candidate
type: B, layer: 1, pos: 15

## Relational analysis of NS_A1_B1_A1_B2_A2_B1_A2_B1
Optimization infeasible because this subproblem isn't reachable.

### Candidate
type: B, layer: 1, pos: 26

## Relational analysis of NS_A1_B1_A1_B2_A2_B1_A2_B1
Optimization infeasible because this subproblem isn't reachable.

### Candidate
type: B, layer: 1, pos: 45

## Relational analysis of NS_A1_B1_A1_B2_A2_B1_A2_B1
Optimization infeasible because this subproblem isn't reachable.

### Candidate
type: B, layer: 1, pos: 5

## Relational analysis of NS_A1_B1_A1_B2_A2_B1_A2_B1
Optimization infeasible because this subproblem isn't reachable.

### Candidate
type: B, layer: 1, pos: 20

## Relational analysis of NS_A1_B1_A1_B2_A2_B1_A2_B1
Optimization infeasible because this subproblem isn't reachable.

## BFS NS instance: NS_A1_B1_A1_B2_A2_B2_A1

### Backsubstitution after applying NS history:
0: -4.3730445, 12.4010944, -5.8249183, 16.1679707, -20.5410156, 18.2260132
1: -10.6506119, 19.4448338, -14.3743057, 24.8603783, -35.5109825, 33.8191376
2: -7.4423847, 17.4804916, -10.0804796, 22.5439949, -29.9863796, 27.5609703
3: -11.6417799, 21.3417969, -15.6187658, 27.5139828, -39.1557617, 36.9605637
4: -10.7328491, 21.4377575, -14.3854094, 27.7590103, -38.4918594, 35.8231659

Time for backsubstitution: 0.95 seconds

### NS candidates at layer 1
type: B, layer: 1, pos: 12
type: B, layer: 1, pos: 9
type: B, layer: 1, pos: 6
type: B, layer: 1, pos: 39
type: B, layer: 1, pos: 48
type: B, layer: 1, pos: 38
type: B, layer: 1, pos: 0
type: B, layer: 1, pos: 28
type: B, layer: 1, pos: 24
type: B, layer: 1, pos: 41
type: B, layer: 1, pos: 8
type: B, layer: 1, pos: 31
type: B, layer: 1, pos: 26
type: B, layer: 1, pos: 45
type: B, layer: 1, pos: 5
type: B, layer: 1, pos: 20
type: B, layer: 1, pos: 15

Time for candidate selection: 0.07 seconds

### Candidate
type: B, layer: 1, pos: 12

### Candidate
type: B, layer: 1, pos: 9

### Candidate
type: B, layer: 1, pos: 6

### Candidate
type: B, layer: 1, pos: 39

### Candidate
type: B, layer: 1, pos: 48

### Candidate
type: B, layer: 1, pos: 38

### Candidate
type: B, layer: 1, pos: 0

### Candidate
type: B, layer: 1, pos: 28

### Candidate
type: B, layer: 1, pos: 24

### Candidate
type: B, layer: 1, pos: 41

### Candidate
type: B, layer: 1, pos: 8

### Candidate
type: B, layer: 1, pos: 31

## Relational analysis of NS_A1_B1_A1_B2_A2_B2_A1_B1
Optimization infeasible because this subproblem isn't reachable.

### Candidate
type: B, layer: 1, pos: 26

## Relational analysis of NS_A1_B1_A1_B2_A2_B2_A1_B1
Optimization infeasible because this subproblem isn't reachable.

### Candidate
type: B, layer: 1, pos: 45

## Relational analysis of NS_A1_B1_A1_B2_A2_B2_A1_B1
Optimization infeasible because this subproblem isn't reachable.

### Candidate
type: B, layer: 1, pos: 5

## Relational analysis of NS_A1_B1_A1_B2_A2_B2_A1_B1
Optimization infeasible because this subproblem isn't reachable.

### Candidate
type: B, layer: 1, pos: 20

## Relational analysis of NS_A1_B1_A1_B2_A2_B2_A1_B1

### Relational analysis result of NS_A1_B1_A1_B2_A2_B2_A1_B1
Status: Status.UNKNOWN
Output dim: 0, lower bound: -20.9338087, upper bound: 20.9400083
time: 0.73 seconds

## Relational analysis of NS_A1_B1_A1_B2_A2_B2_A1_B2

### Relational analysis result of NS_A1_B1_A1_B2_A2_B2_A1_B2
Status: Status.UNKNOWN
Output dim: 0, lower bound: -20.9338087, upper bound: 20.9460725
time: 0.78 seconds

## BFS NS instance: NS_A1_B1_A1_B2_A2_B2_A2

### Backsubstitution after applying NS history:
0: -5.6191449, 15.7847385, -6.2875152, 17.4228020, -23.0419464, 22.0722504
1: -14.0169811, 24.3333015, -15.6246319, 26.6675835, -40.6845627, 39.9579315
2: -9.7616282, 22.0435867, -10.9288807, 24.2501984, -34.0118256, 32.9724655
3: -15.1224251, 26.8683109, -16.9074593, 29.5406723, -44.6630974, 43.7757721
4: -13.7608461, 27.2130318, -15.4907417, 29.9188957, -43.6797409, 42.7037659

Time for backsubstitution: 0.96 seconds

### NS candidates at layer 1
type: B, layer: 1, pos: 12
type: B, layer: 1, pos: 9
type: B, layer: 1, pos: 6
type: B, layer: 1, pos: 48
type: B, layer: 1, pos: 39
type: B, layer: 1, pos: 38
type: B, layer: 1, pos: 28
type: B, layer: 1, pos: 0
type: B, layer: 1, pos: 24
type: B, layer: 1, pos: 41
type: B, layer: 1, pos: 8
type: B, layer: 1, pos: 31
type: B, layer: 1, pos: 26
type: B, layer: 1, pos: 45
type: B, layer: 1, pos: 5
type: B, layer: 1, pos: 20
type: B, layer: 1, pos: 15

Time for candidate selection: 0.07 seconds

### Candidate
type: B, layer: 1, pos: 12

### Candidate
type: B, layer: 1, pos: 9

### Candidate
type: B, layer: 1, pos: 6

### Candidate
type: B, layer: 1, pos: 48

### Candidate
type: B, layer: 1, pos: 39

### Candidate
type: B, layer: 1, pos: 38

### Candidate
type: B, layer: 1, pos: 28

### Candidate
type: B, layer: 1, pos: 0

### Candidate
type: B, layer: 1, pos: 24

### Candidate
type: B, layer: 1, pos: 41

### Candidate
type: B, layer: 1, pos: 8

### Candidate
type: B, layer: 1, pos: 31

## Relational analysis of NS_A1_B1_A1_B2_A2_B2_A2_B1
Optimization infeasible because this subproblem isn't reachable.

### Candidate
type: B, layer: 1, pos: 26

## Relational analysis of NS_A1_B1_A1_B2_A2_B2_A2_B1
Optimization infeasible because this subproblem isn't reachable.

### Candidate
type: B, layer: 1, pos: 45

## Relational analysis of NS_A1_B1_A1_B2_A2_B2_A2_B1
Optimization infeasible because this subproblem isn't reachable.

### Candidate
type: B, layer: 1, pos: 5

## Relational analysis of NS_A1_B1_A1_B2_A2_B2_A2_B1
Optimization infeasible because this subproblem isn't reachable.

### Candidate
type: B, layer: 1, pos: 20

## Relational analysis of NS_A1_B1_A1_B2_A2_B2_A2_B1

### Relational analysis result of NS_A1_B1_A1_B2_A2_B2_A2_B1
Status: Status.UNKNOWN
Output dim: 0, lower bound: -20.9346823, upper bound: 20.9406742
time: 0.64 seconds

## Relational analysis of NS_A1_B1_A1_B2_A2_B2_A2_B2

### Relational analysis result of NS_A1_B1_A1_B2_A2_B2_A2_B2
Status: Status.UNKNOWN
Output dim: 0, lower bound: -20.9346823, upper bound: 20.9523051
time: 0.69 seconds

## BFS NS instance: NS_A1_B1_A2_B1_A1_B2_A1

### Backsubstitution after applying NS history:
0: -4.4598141, 12.6236944, -5.5203495, 15.4967051, -19.9565201, 18.1440430
1: -10.7757635, 19.9015121, -13.8108549, 23.7440300, -34.5197945, 33.7123604
2: -7.5629559, 17.8345814, -9.6062298, 21.5711212, -29.1340771, 27.4408092
3: -11.8345442, 21.7949352, -14.8709297, 26.2682171, -38.1027603, 36.6658630
4: -11.0299988, 21.8339424, -13.4816484, 26.6409798, -37.6709785, 35.3155899

Time for backsubstitution: 0.96 seconds

### NS candidates at layer 1
type: B, layer: 1, pos: 12
type: B, layer: 1, pos: 9
type: B, layer: 1, pos: 48
type: B, layer: 1, pos: 6
type: B, layer: 1, pos: 39
type: B, layer: 1, pos: 38
type: B, layer: 1, pos: 28
type: B, layer: 1, pos: 0
type: B, layer: 1, pos: 41
type: B, layer: 1, pos: 15
type: B, layer: 1, pos: 24
type: B, layer: 1, pos: 37
type: B, layer: 1, pos: 31
type: B, layer: 1, pos: 20
type: B, layer: 1, pos: 45
type: B, layer: 1, pos: 5
type: B, layer: 1, pos: 26

Time for candidate selection: 0.07 seconds

### Candidate
type: B, layer: 1, pos: 12

### Candidate
type: B, layer: 1, pos: 9

### Candidate
type: B, layer: 1, pos: 48

### Candidate
type: B, layer: 1, pos: 6

### Candidate
type: B, layer: 1, pos: 39

### Candidate
type: B, layer: 1, pos: 38

### Candidate
type: B, layer: 1, pos: 28

### Candidate
type: B, layer: 1, pos: 0

### Candidate
type: B, layer: 1, pos: 41

### Candidate
type: B, layer: 1, pos: 15

## Relational analysis of NS_A1_B1_A2_B1_A1_B2_A1_B1
Optimization infeasible because this subproblem isn't reachable.

### Candidate
type: B, layer: 1, pos: 24

### Candidate
type: B, layer: 1, pos: 37

## Relational analysis of NS_A1_B1_A2_B1_A1_B2_A1_B1
Optimization infeasible because this subproblem isn't reachable.

### Candidate
type: B, layer: 1, pos: 31

## Relational analysis of NS_A1_B1_A2_B1_A1_B2_A1_B1
Optimization infeasible because this subproblem isn't reachable.

### Candidate
type: B, layer: 1, pos: 20

## Relational analysis of NS_A1_B1_A2_B1_A1_B2_A1_B1

### Relational analysis result of NS_A1_B1_A2_B1_A1_B2_A1_B1
Status: Status.UNKNOWN
Output dim: 0, lower bound: -20.9484474, upper bound: 20.9444051
time: 0.63 seconds

## Relational analysis of NS_A1_B1_A2_B1_A1_B2_A1_B2

### Relational analysis result of NS_A1_B1_A2_B1_A1_B2_A1_B2
Status: Status.UNKNOWN
Output dim: 0, lower bound: -20.9484474, upper bound: 20.9501143
time: 0.63 seconds

## BFS NS instance: NS_A1_B1_A2_B1_A1_B2_A2

### Backsubstitution after applying NS history:
0: -5.6683564, 15.9111500, -5.9749360, 16.7148342, -22.3831902, 21.8860855
1: -14.0282211, 24.6389408, -15.0470085, 25.5202827, -39.5485001, 39.6859436
2: -9.8104858, 22.2736797, -10.4386559, 23.2222977, -33.0327759, 32.7123337
3: -15.2115622, 27.1595955, -16.1346149, 28.2657928, -43.4773521, 43.2942123
4: -13.9753189, 27.4463177, -14.5654545, 28.7341194, -42.7094269, 42.0117607

Time for backsubstitution: 0.96 seconds

### NS candidates at layer 1
type: B, layer: 1, pos: 12
type: B, layer: 1, pos: 9
type: B, layer: 1, pos: 48
type: B, layer: 1, pos: 6
type: B, layer: 1, pos: 28
type: B, layer: 1, pos: 38
type: B, layer: 1, pos: 39
type: B, layer: 1, pos: 41
type: B, layer: 1, pos: 0
type: B, layer: 1, pos: 15
type: B, layer: 1, pos: 37
type: B, layer: 1, pos: 24
type: B, layer: 1, pos: 20
type: B, layer: 1, pos: 31
type: B, layer: 1, pos: 5
type: B, layer: 1, pos: 26
type: B, layer: 1, pos: 45

Time for candidate selection: 0.07 seconds

### Candidate
type: B, layer: 1, pos: 12

### Candidate
type: B, layer: 1, pos: 9

### Candidate
type: B, layer: 1, pos: 48

### Candidate
type: B, layer: 1, pos: 6

### Candidate
type: B, layer: 1, pos: 28

### Candidate
type: B, layer: 1, pos: 38

### Candidate
type: B, layer: 1, pos: 39

### Candidate
type: B, layer: 1, pos: 41

### Candidate
type: B, layer: 1, pos: 0

### Candidate
type: B, layer: 1, pos: 15

## Relational analysis of NS_A1_B1_A2_B1_A1_B2_A2_B1
Optimization infeasible because this subproblem isn't reachable.

### Candidate
type: B, layer: 1, pos: 37

## Relational analysis of NS_A1_B1_A2_B1_A1_B2_A2_B1
Optimization infeasible because this subproblem isn't reachable.

### Candidate
type: B, layer: 1, pos: 24

### Candidate
type: B, layer: 1, pos: 20

## Relational analysis of NS_A1_B1_A2_B1_A1_B2_A2_B1

### Relational analysis result of NS_A1_B1_A2_B1_A1_B2_A2_B1
Status: Status.UNKNOWN
Output dim: 0, lower bound: -20.9498692, upper bound: 20.9460628
time: 0.65 seconds

## Relational analysis of NS_A1_B1_A2_B1_A1_B2_A2_B2

### Relational analysis result of NS_A1_B1_A2_B1_A1_B2_A2_B2
Status: Status.UNKNOWN
Output dim: 0, lower bound: -20.9498692, upper bound: 20.9564671
time: 0.58 seconds

## BFS NS instance: NS_A1_B1_A2_B1_A2_B2_A1

### Backsubstitution after applying NS history:
0: -4.6641440, 13.1056528, -5.4617543, 15.3407335, -20.0048752, 18.5674057
1: -11.2629089, 20.5948009, -13.6533833, 23.5206261, -34.7835350, 34.2481842
2: -7.9238386, 18.4998074, -9.5001879, 21.3645668, -29.2884064, 27.9999962
3: -12.3741236, 22.5855350, -14.7099571, 26.0154209, -38.3895416, 37.2954941
4: -11.5703382, 22.6495628, -13.3436556, 26.3819714, -37.9522972, 35.9932175

Time for backsubstitution: 0.97 seconds

### NS candidates at layer 1
type: B, layer: 1, pos: 12
type: B, layer: 1, pos: 9
type: B, layer: 1, pos: 48
type: B, layer: 1, pos: 6
type: B, layer: 1, pos: 39
type: B, layer: 1, pos: 38
type: B, layer: 1, pos: 28
type: B, layer: 1, pos: 0
type: B, layer: 1, pos: 41
type: B, layer: 1, pos: 15
type: B, layer: 1, pos: 24
type: B, layer: 1, pos: 37
type: B, layer: 1, pos: 31
type: B, layer: 1, pos: 45
type: B, layer: 1, pos: 20
type: B, layer: 1, pos: 5
type: B, layer: 1, pos: 26

Time for candidate selection: 0.07 seconds

### Candidate
type: B, layer: 1, pos: 12

### Candidate
type: B, layer: 1, pos: 9

### Candidate
type: B, layer: 1, pos: 48

### Candidate
type: B, layer: 1, pos: 6

### Candidate
type: B, layer: 1, pos: 39

### Candidate
type: B, layer: 1, pos: 38

### Candidate
type: B, layer: 1, pos: 28

### Candidate
type: B, layer: 1, pos: 0

### Candidate
type: B, layer: 1, pos: 41

### Candidate
type: B, layer: 1, pos: 15

## Relational analysis of NS_A1_B1_A2_B1_A2_B2_A1_B1
Optimization infeasible because this subproblem isn't reachable.

### Candidate
type: B, layer: 1, pos: 24

### Candidate
type: B, layer: 1, pos: 37

## Relational analysis of NS_A1_B1_A2_B1_A2_B2_A1_B1
Optimization infeasible because this subproblem isn't reachable.

### Candidate
type: B, layer: 1, pos: 31

## Relational analysis of NS_A1_B1_A2_B1_A2_B2_A1_B1
Optimization infeasible because this subproblem isn't reachable.

### Candidate
type: B, layer: 1, pos: 45

## Relational analysis of NS_A1_B1_A2_B1_A2_B2_A1_B1
Optimization infeasible because this subproblem isn't reachable.

### Candidate
type: B, layer: 1, pos: 20

## Relational analysis of NS_A1_B1_A2_B1_A2_B2_A1_B1

### Relational analysis result of NS_A1_B1_A2_B1_A2_B2_A1_B1
Status: Status.UNKNOWN
Output dim: 0, lower bound: -20.9500116, upper bound: 20.9463470
time: 0.90 seconds

## Relational analysis of NS_A1_B1_A2_B1_A2_B2_A1_B2

### Relational analysis result of NS_A1_B1_A2_B1_A2_B2_A1_B2
Status: Status.UNKNOWN
Output dim: 0, lower bound: -20.9500116, upper bound: 20.9505436
time: 0.60 seconds

## BFS NS instance: NS_A1_B1_A2_B1_A2_B2_A2

### Backsubstitution after applying NS history:
0: -5.8973289, 16.4676323, -5.9204555, 16.5701351, -22.4674644, 22.3880882
1: -14.5865850, 25.4389725, -14.9021521, 25.3115902, -39.8981743, 40.3411255
2: -10.2133141, 23.0358047, -10.3402786, 23.0298958, -33.2432098, 33.3760796
3: -15.8239231, 28.0691185, -15.9851131, 28.0295258, -43.8534470, 44.0542221
4: -14.5728226, 28.3855171, -14.4351892, 28.4933262, -43.0661469, 42.8207054

Time for backsubstitution: 0.97 seconds

### NS candidates at layer 1
type: B, layer: 1, pos: 12
type: B, layer: 1, pos: 9
type: B, layer: 1, pos: 48
type: B, layer: 1, pos: 6
type: B, layer: 1, pos: 28
type: B, layer: 1, pos: 38
type: B, layer: 1, pos: 39
type: B, layer: 1, pos: 41
type: B, layer: 1, pos: 0
type: B, layer: 1, pos: 15
type: B, layer: 1, pos: 24
type: B, layer: 1, pos: 37
type: B, layer: 1, pos: 20
type: B, layer: 1, pos: 31
type: B, layer: 1, pos: 45
type: B, layer: 1, pos: 5
type: B, layer: 1, pos: 26

Time for candidate selection: 0.07 seconds

### Candidate
type: B, layer: 1, pos: 12

### Candidate
type: B, layer: 1, pos: 9

### Candidate
type: B, layer: 1, pos: 48

### Candidate
type: B, layer: 1, pos: 6

### Candidate
type: B, layer: 1, pos: 28

### Candidate
type: B, layer: 1, pos: 38

### Candidate
type: B, layer: 1, pos: 39

### Candidate
type: B, layer: 1, pos: 41

### Candidate
type: B, layer: 1, pos: 0

### Candidate
type: B, layer: 1, pos: 15

## Relational analysis of NS_A1_B1_A2_B1_A2_B2_A2_B1
Optimization infeasible because this subproblem isn't reachable.

### Candidate
type: B, layer: 1, pos: 24

### Candidate
type: B, layer: 1, pos: 37

## Relational analysis of NS_A1_B1_A2_B1_A2_B2_A2_B1
Optimization infeasible because this subproblem isn't reachable.

### Candidate
type: B, layer: 1, pos: 20

## Relational analysis of NS_A1_B1_A2_B1_A2_B2_A2_B1

### Relational analysis result of NS_A1_B1_A2_B1_A2_B2_A2_B1
Status: Status.UNKNOWN
Output dim: 0, lower bound: -20.9507714, upper bound: 20.9472579
time: 0.75 seconds

## Relational analysis of NS_A1_B1_A2_B1_A2_B2_A2_B2

### Relational analysis result of NS_A1_B1_A2_B1_A2_B2_A2_B2
Status: Status.UNKNOWN
Output dim: 0, lower bound: -20.9507714, upper bound: 20.9549741
time: 0.69 seconds

## BFS NS instance: NS_A1_B1_A2_B2_A1_B1_A1

### Backsubstitution after applying NS history:
0: -4.5310197, 12.8211260, -5.6238737, 15.6892004, -20.2202187, 18.4449997
1: -10.9697037, 20.1862068, -13.9036665, 24.1666298, -35.1363258, 34.0898743
2: -7.6878438, 18.0960484, -9.7329712, 21.8940754, -29.5819187, 27.8290176
3: -12.0335846, 22.1118584, -15.0919828, 26.7131882, -38.7467728, 37.2038422
4: -11.1936426, 22.1703987, -13.8608274, 26.9696217, -38.1632614, 36.0312271

Time for backsubstitution: 0.99 seconds

### NS candidates at layer 1
type: B, layer: 1, pos: 12
type: B, layer: 1, pos: 39
type: B, layer: 1, pos: 6
type: B, layer: 1, pos: 48
type: B, layer: 1, pos: 28
type: B, layer: 1, pos: 38
type: B, layer: 1, pos: 0
type: B, layer: 1, pos: 24
type: B, layer: 1, pos: 41
type: B, layer: 1, pos: 37
type: B, layer: 1, pos: 8
type: B, layer: 1, pos: 31
type: B, layer: 1, pos: 26
type: B, layer: 1, pos: 45
type: B, layer: 1, pos: 15
type: B, layer: 1, pos: 5
type: B, layer: 1, pos: 20

Time for candidate selection: 0.07 seconds

### Candidate
type: B, layer: 1, pos: 12

### Candidate
type: B, layer: 1, pos: 39

### Candidate
type: B, layer: 1, pos: 6

### Candidate
type: B, layer: 1, pos: 48

### Candidate
type: B, layer: 1, pos: 28

### Candidate
type: B, layer: 1, pos: 38

### Candidate
type: B, layer: 1, pos: 0

### Candidate
type: B, layer: 1, pos: 24

### Candidate
type: B, layer: 1, pos: 41

### Candidate
type: B, layer: 1, pos: 37

### Candidate
type: B, layer: 1, pos: 8

### Candidate
type: B, layer: 1, pos: 31

## Relational analysis of NS_A1_B1_A2_B2_A1_B1_A1_B1
Optimization infeasible because this subproblem isn't reachable.

### Candidate
type: B, layer: 1, pos: 26

## Relational analysis of NS_A1_B1_A2_B2_A1_B1_A1_B1
Optimization infeasible because this subproblem isn't reachable.

### Candidate
type: B, layer: 1, pos: 45

## Relational analysis of NS_A1_B1_A2_B2_A1_B1_A1_B1
Optimization infeasible because this subproblem isn't reachable.

### Candidate
type: B, layer: 1, pos: 15

## Relational analysis of NS_A1_B1_A2_B2_A1_B1_A1_B1
Optimization infeasible because this subproblem isn't reachable.

### Candidate
type: B, layer: 1, pos: 5

## Relational analysis of NS_A1_B1_A2_B2_A1_B1_A1_B1
Optimization infeasible because this subproblem isn't reachable.

### Candidate
type: B, layer: 1, pos: 20

## Relational analysis of NS_A1_B1_A2_B2_A1_B1_A1_B1
Optimization infeasible because this subproblem isn't reachable.

## BFS NS instance: NS_A1_B1_A2_B2_A1_B1_A2

### Backsubstitution after applying NS history:
0: -5.7451720, 16.1186409, -6.1058712, 16.9839706, -22.7291412, 22.2245121
1: -14.2366209, 24.9381962, -15.2125006, 26.0323257, -40.2689438, 40.1506882
2: -9.9447250, 22.5496464, -10.6174936, 23.6492004, -33.5939178, 33.1671371
3: -15.4248657, 27.4936295, -16.4271126, 28.8107719, -44.2356377, 43.9207382
4: -14.1521463, 27.7999382, -15.0079222, 29.1930733, -43.3452187, 42.8078537

Time for backsubstitution: 0.98 seconds

### NS candidates at layer 1
type: B, layer: 1, pos: 12
type: B, layer: 1, pos: 6
type: B, layer: 1, pos: 48
type: B, layer: 1, pos: 39
type: B, layer: 1, pos: 28
type: B, layer: 1, pos: 38
type: B, layer: 1, pos: 0
type: B, layer: 1, pos: 24
type: B, layer: 1, pos: 37
type: B, layer: 1, pos: 41
type: B, layer: 1, pos: 8
type: B, layer: 1, pos: 31
type: B, layer: 1, pos: 26
type: B, layer: 1, pos: 20
type: B, layer: 1, pos: 15
type: B, layer: 1, pos: 45
type: B, layer: 1, pos: 5

Time for candidate selection: 0.07 seconds

### Candidate
type: B, layer: 1, pos: 12

### Candidate
type: B, layer: 1, pos: 6

### Candidate
type: B, layer: 1, pos: 48

### Candidate
type: B, layer: 1, pos: 39

### Candidate
type: B, layer: 1, pos: 28

### Candidate
type: B, layer: 1, pos: 38

### Candidate
type: B, layer: 1, pos: 0

### Candidate
type: B, layer: 1, pos: 24

### Candidate
type: B, layer: 1, pos: 37

### Candidate
type: B, layer: 1, pos: 41

### Candidate
type: B, layer: 1, pos: 8

### Candidate
type: B, layer: 1, pos: 31

## Relational analysis of NS_A1_B1_A2_B2_A1_B1_A2_B1
Optimization infeasible because this subproblem isn't reachable.

### Candidate
type: B, layer: 1, pos: 26

## Relational analysis of NS_A1_B1_A2_B2_A1_B1_A2_B1
Optimization infeasible because this subproblem isn't reachable.

### Candidate
type: B, layer: 1, pos: 20

## Relational analysis of NS_A1_B1_A2_B2_A1_B1_A2_B1
Optimization infeasible because this subproblem isn't reachable.

### Candidate
type: B, layer: 1, pos: 15

## Relational analysis of NS_A1_B1_A2_B2_A1_B1_A2_B1
Optimization infeasible because this subproblem isn't reachable.

### Candidate
type: B, layer: 1, pos: 45

## Relational analysis of NS_A1_B1_A2_B2_A1_B1_A2_B1
Optimization infeasible because this subproblem isn't reachable.

### Candidate
type: B, layer: 1, pos: 5

## Relational analysis of NS_A1_B1_A2_B2_A1_B1_A2_B1
Optimization infeasible because this subproblem isn't reachable.

## BFS NS instance: NS_A1_B1_A2_B2_A1_B2_A1

### Backsubstitution after applying NS history:
0: -4.5310197, 12.8211260, -5.5398569, 15.4405241, -19.9715443, 18.3609810
1: -10.9697037, 20.1862068, -13.7631493, 23.6726151, -34.6423149, 33.9493523
2: -7.6878438, 18.0960484, -9.6167507, 21.5257835, -29.2136269, 27.7127991
3: -12.0335846, 22.1118584, -14.8913517, 26.2035637, -38.2371445, 37.0032120
4: -11.1936426, 22.1703987, -13.6092377, 26.5387630, -37.7324066, 35.7796364

Time for backsubstitution: 0.97 seconds

### NS candidates at layer 1
type: B, layer: 1, pos: 12
type: B, layer: 1, pos: 48
type: B, layer: 1, pos: 6
type: B, layer: 1, pos: 39
type: B, layer: 1, pos: 38
type: B, layer: 1, pos: 0
type: B, layer: 1, pos: 28
type: B, layer: 1, pos: 41
type: B, layer: 1, pos: 24
type: B, layer: 1, pos: 8
type: B, layer: 1, pos: 37
type: B, layer: 1, pos: 31
type: B, layer: 1, pos: 45
type: B, layer: 1, pos: 15
type: B, layer: 1, pos: 26
type: B, layer: 1, pos: 20
type: B, layer: 1, pos: 5

Time for candidate selection: 0.07 seconds

### Candidate
type: B, layer: 1, pos: 12

### Candidate
type: B, layer: 1, pos: 48

### Candidate
type: B, layer: 1, pos: 6

### Candidate
type: B, layer: 1, pos: 39

### Candidate
type: B, layer: 1, pos: 38

### Candidate
type: B, layer: 1, pos: 0

### Candidate
type: B, layer: 1, pos: 28

### Candidate
type: B, layer: 1, pos: 41

### Candidate
type: B, layer: 1, pos: 24

### Candidate
type: B, layer: 1, pos: 8

### Candidate
type: B, layer: 1, pos: 37

### Candidate
type: B, layer: 1, pos: 31

## Relational analysis of NS_A1_B1_A2_B2_A1_B2_A1_B1
Optimization infeasible because this subproblem isn't reachable.

### Candidate
type: B, layer: 1, pos: 45

## Relational analysis of NS_A1_B1_A2_B2_A1_B2_A1_B1
Optimization infeasible because this subproblem isn't reachable.

### Candidate
type: B, layer: 1, pos: 15

## Relational analysis of NS_A1_B1_A2_B2_A1_B2_A1_B1
Optimization infeasible because this subproblem isn't reachable.

### Candidate
type: B, layer: 1, pos: 26

## Relational analysis of NS_A1_B1_A2_B2_A1_B2_A1_B1
Optimization infeasible because this subproblem isn't reachable.

### Candidate
type: B, layer: 1, pos: 20

## Relational analysis of NS_A1_B1_A2_B2_A1_B2_A1_B1

### Relational analysis result of NS_A1_B1_A2_B2_A1_B2_A1_B1
Status: Status.UNKNOWN
Output dim: 0, lower bound: -20.9108025, upper bound: 20.9036356
time: 0.94 seconds

## Relational analysis of NS_A1_B1_A2_B2_A1_B2_A1_B2

### Relational analysis result of NS_A1_B1_A2_B2_A1_B2_A1_B2
Status: Status.UNKNOWN
Output dim: 0, lower bound: -20.9108025, upper bound: 20.9101592
time: 0.67 seconds

## BFS NS instance: NS_A1_B1_A2_B2_A1_B2_A2

### Backsubstitution after applying NS history:
0: -5.7451720, 16.1186409, -6.0464940, 16.7988091, -22.5439796, 22.1651344
1: -14.2366209, 24.9381962, -15.1383047, 25.6429367, -39.8795586, 40.0764999
2: -9.9447250, 22.5496464, -10.5443201, 23.3643627, -33.3090782, 33.0939636
3: -15.4248657, 27.4936295, -16.2959213, 28.4164085, -43.8412704, 43.7895508
4: -14.1521463, 27.7999382, -14.8136282, 28.8689480, -43.0210876, 42.6135635

Time for backsubstitution: 0.97 seconds

### NS candidates at layer 1
type: B, layer: 1, pos: 12
type: B, layer: 1, pos: 6
type: B, layer: 1, pos: 48
type: B, layer: 1, pos: 28
type: B, layer: 1, pos: 38
type: B, layer: 1, pos: 39
type: B, layer: 1, pos: 0
type: B, layer: 1, pos: 41
type: B, layer: 1, pos: 8
type: B, layer: 1, pos: 24
type: B, layer: 1, pos: 37
type: B, layer: 1, pos: 31
type: B, layer: 1, pos: 15
type: B, layer: 1, pos: 45
type: B, layer: 1, pos: 26
type: B, layer: 1, pos: 5
type: B, layer: 1, pos: 20

Time for candidate selection: 0.07 seconds

### Candidate
type: B, layer: 1, pos: 12

### Candidate
type: B, layer: 1, pos: 6

### Candidate
type: B, layer: 1, pos: 48

### Candidate
type: B, layer: 1, pos: 28

### Candidate
type: B, layer: 1, pos: 38

### Candidate
type: B, layer: 1, pos: 39

### Candidate
type: B, layer: 1, pos: 0

### Candidate
type: B, layer: 1, pos: 41

### Candidate
type: B, layer: 1, pos: 8

### Candidate
type: B, layer: 1, pos: 24

### Candidate
type: B, layer: 1, pos: 37

### Candidate
type: B, layer: 1, pos: 31

## Relational analysis of NS_A1_B1_A2_B2_A1_B2_A2_B1
Optimization infeasible because this subproblem isn't reachable.

### Candidate
type: B, layer: 1, pos: 15

## Relational analysis of NS_A1_B1_A2_B2_A1_B2_A2_B1
Optimization infeasible because this subproblem isn't reachable.

### Candidate
type: B, layer: 1, pos: 45

## Relational analysis of NS_A1_B1_A2_B2_A1_B2_A2_B1
Optimization infeasible because this subproblem isn't reachable.

### Candidate
type: B, layer: 1, pos: 26

## Relational analysis of NS_A1_B1_A2_B2_A1_B2_A2_B1
Optimization infeasible because this subproblem isn't reachable.

### Candidate
type: B, layer: 1, pos: 5

## Relational analysis of NS_A1_B1_A2_B2_A1_B2_A2_B1
Optimization infeasible because this subproblem isn't reachable.

### Candidate
type: B, layer: 1, pos: 20

## Relational analysis of NS_A1_B1_A2_B2_A1_B2_A2_B1

### Relational analysis result of NS_A1_B1_A2_B2_A1_B2_A2_B1
Status: Status.UNKNOWN
Output dim: 0, lower bound: -20.8834278, upper bound: 20.9058885
time: 0.68 seconds

## Relational analysis of NS_A1_B1_A2_B2_A1_B2_A2_B2

### Relational analysis result of NS_A1_B1_A2_B2_A1_B2_A2_B2
Status: Status.UNKNOWN
Output dim: 0, lower bound: -20.9130919, upper bound: 20.9248678
time: 0.76 seconds

## BFS NS instance: NS_A1_B1_A2_B2_A2_B1_A1

### Backsubstitution after applying NS history:
0: -5.5425358, 15.5357647, -6.1188755, 17.0194664, -22.5620022, 21.6546364
1: -13.7755260, 23.9755001, -15.2469034, 26.0870571, -39.8625832, 39.2224007
2: -9.6129322, 21.7249069, -10.6409988, 23.6971779, -33.3101082, 32.3659058
3: -14.8987875, 26.4675903, -16.4629097, 28.8716431, -43.7704239, 42.9305000
4: -13.6085463, 26.7931404, -15.0383358, 29.2521629, -42.8607063, 41.8314743

Time for backsubstitution: 0.98 seconds

### NS candidates at layer 1
type: B, layer: 1, pos: 12
type: B, layer: 1, pos: 6
type: B, layer: 1, pos: 48
type: B, layer: 1, pos: 39
type: B, layer: 1, pos: 28
type: B, layer: 1, pos: 38
type: B, layer: 1, pos: 0
type: B, layer: 1, pos: 24
type: B, layer: 1, pos: 41
type: B, layer: 1, pos: 8
type: B, layer: 1, pos: 37
type: B, layer: 1, pos: 31
type: B, layer: 1, pos: 20
type: B, layer: 1, pos: 26
type: B, layer: 1, pos: 15
type: B, layer: 1, pos: 45
type: B, layer: 1, pos: 5

Time for candidate selection: 0.07 seconds

### Candidate
type: B, layer: 1, pos: 12

### Candidate
type: B, layer: 1, pos: 6

### Candidate
type: B, layer: 1, pos: 48

### Candidate
type: B, layer: 1, pos: 39

### Candidate
type: B, layer: 1, pos: 28

### Candidate
type: B, layer: 1, pos: 38

### Candidate
type: B, layer: 1, pos: 0

### Candidate
type: B, layer: 1, pos: 24

### Candidate
type: B, layer: 1, pos: 41

### Candidate
type: B, layer: 1, pos: 8

## Relational analysis of NS_A1_B1_A2_B2_A2_B1_A1_B1

### Relational analysis result of NS_A1_B1_A2_B2_A2_B1_A1_B1
Status: Status.UNKNOWN
Output dim: 0, lower bound: -20.9138254, upper bound: 20.9189479
time: 0.71 seconds

## Relational analysis of NS_A1_B1_A2_B2_A2_B1_A1_B2

### Relational analysis result of NS_A1_B1_A2_B2_A2_B1_A1_B2
Status: Status.UNKNOWN
Output dim: 0, lower bound: -20.9135942, upper bound: 20.9187455
time: 0.63 seconds

## BFS NS instance: NS_A1_B1_A2_B2_A2_B1_A2

### Backsubstitution after applying NS history:
0: -5.7436028, 16.0160637, -6.0863938, 16.9345703, -22.6781731, 22.1024570
1: -14.2478695, 24.6804008, -15.1604967, 25.9652119, -40.2130814, 39.8408928
2: -9.9596272, 22.3912830, -10.5830650, 23.5849571, -33.5445824, 32.9743423
3: -15.4277849, 27.2617645, -16.3738499, 28.7345905, -44.1623764, 43.6356125
4: -14.1459274, 27.6071377, -14.9632530, 29.1128101, -43.2587357, 42.5703812

Time for backsubstitution: 0.98 seconds

### NS candidates at layer 1
type: B, layer: 1, pos: 12
type: B, layer: 1, pos: 6
type: B, layer: 1, pos: 48
type: B, layer: 1, pos: 39
type: B, layer: 1, pos: 28
type: B, layer: 1, pos: 38
type: B, layer: 1, pos: 0
type: B, layer: 1, pos: 24
type: B, layer: 1, pos: 41
type: B, layer: 1, pos: 8
type: B, layer: 1, pos: 37
type: B, layer: 1, pos: 31
type: B, layer: 1, pos: 20
type: B, layer: 1, pos: 26
type: B, layer: 1, pos: 15
type: B, layer: 1, pos: 45
type: B, layer: 1, pos: 5

Time for candidate selection: 0.07 seconds

### Candidate
type: B, layer: 1, pos: 12

### Candidate
type: B, layer: 1, pos: 6

### Candidate
type: B, layer: 1, pos: 48

### Candidate
type: B, layer: 1, pos: 39

### Candidate
type: B, layer: 1, pos: 28

### Candidate
type: B, layer: 1, pos: 38

### Candidate
type: B, layer: 1, pos: 0

### Candidate
type: B, layer: 1, pos: 24

### Candidate
type: B, layer: 1, pos: 41

### Candidate
type: B, layer: 1, pos: 8

## Relational analysis of NS_A1_B1_A2_B2_A2_B1_A2_B1
Optimization infeasible because this subproblem isn't reachable.

### Candidate
type: B, layer: 1, pos: 37

## Relational analysis of NS_A1_B1_A2_B2_A2_B1_A2_B1
Optimization infeasible because this subproblem isn't reachable.

### Candidate
type: B, layer: 1, pos: 31

## Relational analysis of NS_A1_B1_A2_B2_A2_B1_A2_B1
Optimization infeasible because this subproblem isn't reachable.

### Candidate
type: B, layer: 1, pos: 20

## Relational analysis of NS_A1_B1_A2_B2_A2_B1_A2_B1

### Relational analysis result of NS_A1_B1_A2_B2_A2_B1_A2_B1
Status: Status.UNKNOWN
Output dim: 0, lower bound: -20.8796245, upper bound: 20.8847414
time: 0.64 seconds

## Relational analysis of NS_A1_B1_A2_B2_A2_B1_A2_B2

### Relational analysis result of NS_A1_B1_A2_B2_A2_B1_A2_B2
Status: Status.UNKNOWN
Output dim: 0, lower bound: -20.9063213, upper bound: 20.9096749
time: 0.77 seconds

## BFS NS instance: NS_A1_B1_A2_B2_A2_B2_A1

### Backsubstitution after applying NS history:
0: -5.5425358, 15.5357647, -6.0574489, 16.8290138, -22.3715496, 21.5932121
1: -13.7755260, 23.9755001, -15.1670351, 25.6883774, -39.4639053, 39.1425362
2: -9.6129322, 21.7249069, -10.5643253, 23.4050484, -33.0179825, 32.2892303
3: -14.8987875, 26.4675903, -16.3260632, 28.4671326, -43.3659172, 42.7936478
4: -13.6085463, 26.7931404, -14.8392515, 28.9192009, -42.5277443, 41.6323929

Time for backsubstitution: 0.98 seconds

### NS candidates at layer 1
type: B, layer: 1, pos: 12
type: B, layer: 1, pos: 6
type: B, layer: 1, pos: 48
type: B, layer: 1, pos: 28
type: B, layer: 1, pos: 38
type: B, layer: 1, pos: 39
type: B, layer: 1, pos: 0
type: B, layer: 1, pos: 41
type: B, layer: 1, pos: 8
type: B, layer: 1, pos: 24
type: B, layer: 1, pos: 37
type: B, layer: 1, pos: 31
type: B, layer: 1, pos: 15
type: B, layer: 1, pos: 45
type: B, layer: 1, pos: 26
type: B, layer: 1, pos: 20
type: B, layer: 1, pos: 5

Time for candidate selection: 0.07 seconds

### Candidate
type: B, layer: 1, pos: 12

### Candidate
type: B, layer: 1, pos: 6

### Candidate
type: B, layer: 1, pos: 48

### Candidate
type: B, layer: 1, pos: 28

### Candidate
type: B, layer: 1, pos: 38

### Candidate
type: B, layer: 1, pos: 39

### Candidate
type: B, layer: 1, pos: 0

### Candidate
type: B, layer: 1, pos: 41

### Candidate
type: B, layer: 1, pos: 8

## Relational analysis of NS_A1_B1_A2_B2_A2_B2_A1_B1

### Relational analysis result of NS_A1_B1_A2_B2_A2_B2_A1_B1
Status: Status.UNKNOWN
Output dim: 0, lower bound: -20.9356478, upper bound: 20.9386298
time: 0.62 seconds

## Relational analysis of NS_A1_B1_A2_B2_A2_B2_A1_B2

### Relational analysis result of NS_A1_B1_A2_B2_A2_B2_A1_B2
Status: Status.UNKNOWN
Output dim: 0, lower bound: -20.9119850, upper bound: 20.9435253
time: 0.64 seconds

## BFS NS instance: NS_A1_B1_A2_B2_A2_B2_A2

### Backsubstitution after applying NS history:
0: -5.7436028, 16.0160637, -6.0158582, 16.7151527, -22.4587555, 22.0319214
1: -14.2478695, 24.6804008, -15.0505915, 25.5273800, -39.7752495, 39.7309875
2: -9.9596272, 22.3912830, -10.4876976, 23.2550869, -33.2147102, 32.8789749
3: -15.4277849, 27.2617645, -16.2088146, 28.2842636, -43.7120476, 43.4705811
4: -14.1459274, 27.6071377, -14.7432709, 28.7293701, -42.8752937, 42.3504066

Time for backsubstitution: 0.98 seconds

### NS candidates at layer 1
type: B, layer: 1, pos: 12
type: B, layer: 1, pos: 6
type: B, layer: 1, pos: 48
type: B, layer: 1, pos: 38
type: B, layer: 1, pos: 28
type: B, layer: 1, pos: 39
type: B, layer: 1, pos: 0
type: B, layer: 1, pos: 41
type: B, layer: 1, pos: 8
type: B, layer: 1, pos: 24
type: B, layer: 1, pos: 37
type: B, layer: 1, pos: 31
type: B, layer: 1, pos: 15
type: B, layer: 1, pos: 45
type: B, layer: 1, pos: 26
type: B, layer: 1, pos: 5
type: B, layer: 1, pos: 20

Time for candidate selection: 0.07 seconds

### Candidate
type: B, layer: 1, pos: 12

### Candidate
type: B, layer: 1, pos: 6

### Candidate
type: B, layer: 1, pos: 48

### Candidate
type: B, layer: 1, pos: 38

### Candidate
type: B, layer: 1, pos: 28

### Candidate
type: B, layer: 1, pos: 39

### Candidate
type: B, layer: 1, pos: 0

### Candidate
type: B, layer: 1, pos: 41

### Candidate
type: B, layer: 1, pos: 8

## Relational analysis of NS_A1_B1_A2_B2_A2_B2_A2_B1

### Relational analysis result of NS_A1_B1_A2_B2_A2_B2_A2_B1
Status: Status.UNKNOWN
Output dim: 0, lower bound: -20.9354252, upper bound: 20.9383059
time: 0.65 seconds

## Relational analysis of NS_A1_B1_A2_B2_A2_B2_A2_B2

### Relational analysis result of NS_A1_B1_A2_B2_A2_B2_A2_B2
Status: Status.UNKNOWN
Output dim: 0, lower bound: -20.9414215, upper bound: 20.9427230
time: 1.12 seconds

## BFS NS instance: NS_A1_B2_A1_B1_A1_B1_A1

### Backsubstitution after applying NS history:
0: -3.9592547, 11.3020430, -5.1674256, 14.6347809, -18.5940342, 16.4694691
1: -9.5195208, 17.8342209, -12.8268080, 22.6759892, -32.1955109, 30.6610298
2: -6.6626244, 16.0027199, -8.9399157, 20.4848652, -27.1474895, 24.9426346
3: -10.4893026, 19.5231781, -13.8867607, 24.9877224, -35.4770241, 33.4099388
4: -9.7487974, 19.5848465, -12.6648741, 25.2493019, -34.9980927, 32.2497139

Time for backsubstitution: 0.98 seconds

### NS candidates at layer 1
type: B, layer: 1, pos: 39
type: B, layer: 1, pos: 9
type: B, layer: 1, pos: 48
type: B, layer: 1, pos: 6
type: B, layer: 1, pos: 28
type: B, layer: 1, pos: 38
type: B, layer: 1, pos: 0
type: B, layer: 1, pos: 41
type: B, layer: 1, pos: 24
type: B, layer: 1, pos: 8
type: B, layer: 1, pos: 31
type: B, layer: 1, pos: 20
type: B, layer: 1, pos: 37
type: B, layer: 1, pos: 15
type: B, layer: 1, pos: 26
type: B, layer: 1, pos: 45
type: B, layer: 1, pos: 5

Time for candidate selection: 0.07 seconds

### Candidate
type: B, layer: 1, pos: 39

### Candidate
type: B, layer: 1, pos: 9

### Candidate
type: B, layer: 1, pos: 48

### Candidate
type: B, layer: 1, pos: 6

### Candidate
type: B, layer: 1, pos: 28

### Candidate
type: B, layer: 1, pos: 38

### Candidate
type: B, layer: 1, pos: 0

## Relational analysis of NS_A1_B2_A1_B1_A1_B1_A1_B1

### Relational analysis result of NS_A1_B2_A1_B1_A1_B1_A1_B1
Status: Status.UNKNOWN
Output dim: 0, lower bound: -20.9583477, upper bound: 20.9575115
time: 1.00 seconds

## Relational analysis of NS_A1_B2_A1_B1_A1_B1_A1_B2

### Relational analysis result of NS_A1_B2_A1_B1_A1_B1_A1_B2
Status: Status.UNKNOWN
Output dim: 0, lower bound: -20.9583477, upper bound: 20.9575115
time: 0.67 seconds

## BFS NS instance: NS_A1_B2_A1_B1_A1_B1_A2

### Backsubstitution after applying NS history:
0: -5.2072835, 14.7098284, -5.6119409, 15.8261948, -21.0334778, 20.3217697
1: -12.8760509, 22.7740097, -14.0332804, 24.4026737, -37.2787247, 36.8072815
2: -8.9915180, 20.6080379, -9.7576981, 22.0911427, -31.0826588, 30.3657360
3: -13.9752502, 25.1011715, -15.1196823, 26.9363022, -40.9115524, 40.2208481
4: -12.7957668, 25.3980255, -13.7143507, 27.2879791, -40.0837479, 39.1123772

Time for backsubstitution: 0.99 seconds

### NS candidates at layer 1
type: B, layer: 1, pos: 9
type: B, layer: 1, pos: 39
type: B, layer: 1, pos: 6
type: B, layer: 1, pos: 48
type: B, layer: 1, pos: 28
type: B, layer: 1, pos: 38
type: B, layer: 1, pos: 41
type: B, layer: 1, pos: 0
type: B, layer: 1, pos: 8
type: B, layer: 1, pos: 24
type: B, layer: 1, pos: 31
type: B, layer: 1, pos: 37
type: B, layer: 1, pos: 20
type: B, layer: 1, pos: 15
type: B, layer: 1, pos: 45
type: B, layer: 1, pos: 26
type: B, layer: 1, pos: 5

Time for candidate selection: 0.07 seconds

### Candidate
type: B, layer: 1, pos: 9

### Candidate
type: B, layer: 1, pos: 39

### Candidate
type: B, layer: 1, pos: 6

### Candidate
type: B, layer: 1, pos: 48

### Candidate
type: B, layer: 1, pos: 28

### Candidate
type: B, layer: 1, pos: 38

### Candidate
type: B, layer: 1, pos: 41

### Candidate
type: B, layer: 1, pos: 0

## Relational analysis of NS_A1_B2_A1_B1_A1_B1_A2_B1

### Relational analysis result of NS_A1_B2_A1_B1_A1_B1_A2_B1
Status: Status.UNKNOWN
Output dim: 0, lower bound: -20.9622383, upper bound: 20.9629604
time: 0.74 seconds

## Relational analysis of NS_A1_B2_A1_B1_A1_B1_A2_B2

### Relational analysis result of NS_A1_B2_A1_B1_A1_B1_A2_B2
Status: Status.UNKNOWN
Output dim: 0, lower bound: -20.9622383, upper bound: 20.9629604
time: 0.78 seconds

## BFS NS instance: NS_A1_B2_A1_B1_A1_B2_A1

### Backsubstitution after applying NS history:
0: -3.9592547, 11.3020430, -5.4073167, 15.2069616, -19.1662159, 16.7093601
1: -9.5195208, 17.8342209, -13.2949362, 23.6296158, -33.1491356, 31.1291523
2: -6.6626244, 16.0027199, -9.3235912, 21.3279724, -27.9905949, 25.3263111
3: -10.4893026, 19.5231781, -14.4798269, 26.0126381, -36.5019417, 34.0030060
4: -9.7487974, 19.5848465, -13.3824739, 26.2337208, -35.9825134, 32.9673119

Time for backsubstitution: 1.00 seconds

Time for candidate selection: 0.00 seconds

## NS Result
status: Status.UNKNOWN
execution time: (base) + (ns) = 2.60 + 418.14 = 420.74 seconds
