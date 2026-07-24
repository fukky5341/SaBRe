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
execution time: IAR + RelationalAnalysis = 0.80 + 1.69 = 2.48 seconds
status: Status.UNKNOWN
relational distance
Output dim: 0, lower bound: -20.9652728, upper bound: 20.9652728

# Neuron Split (NS) starts

## BFS NS instance: NS

Time for backsubstitution: 0.01 seconds

### NS candidates at layer 1
type: B, layer: 1, pos: 1
type: A, layer: 1, pos: 1
type: B, layer: 1, pos: 12
type: A, layer: 1, pos: 12
type: B, layer: 1, pos: 47
type: A, layer: 1, pos: 47
type: A, layer: 1, pos: 15
type: B, layer: 1, pos: 15
type: A, layer: 1, pos: 45
type: B, layer: 1, pos: 45
type: A, layer: 1, pos: 28
type: B, layer: 1, pos: 28
type: A, layer: 1, pos: 9
type: B, layer: 1, pos: 9
type: A, layer: 1, pos: 37
type: B, layer: 1, pos: 37
type: A, layer: 1, pos: 24
type: B, layer: 1, pos: 24
type: A, layer: 1, pos: 38
type: B, layer: 1, pos: 38
type: B, layer: 1, pos: 6
type: A, layer: 1, pos: 6
type: B, layer: 1, pos: 8
type: A, layer: 1, pos: 8
type: B, layer: 1, pos: 39
type: A, layer: 1, pos: 39
type: A, layer: 1, pos: 48
type: B, layer: 1, pos: 48
type: A, layer: 1, pos: 20
type: B, layer: 1, pos: 20
type: A, layer: 1, pos: 26
type: B, layer: 1, pos: 26
type: A, layer: 1, pos: 5
type: B, layer: 1, pos: 5
type: B, layer: 1, pos: 31
type: A, layer: 1, pos: 31
type: A, layer: 1, pos: 0
type: B, layer: 1, pos: 0
type: A, layer: 1, pos: 41
type: B, layer: 1, pos: 41

Time for candidate selection: 0.06 seconds

### Candidate
type: B, layer: 1, pos: 1

## Relational analysis of NS_B1

### Relational analysis result of NS_B1
Status: Status.UNKNOWN
Output dim: 0, lower bound: -20.9651837, upper bound: 20.9642482
time: 0.52 seconds

## Relational analysis of NS_B2

### Relational analysis result of NS_B2
Status: Status.UNKNOWN
Output dim: 0, lower bound: -20.9652728, upper bound: 20.9652728
time: 0.57 seconds

## Summary of splitting at layer (split count: 0)
- Time for NS candidates: 1.17 seconds
NS_B1, status: Status.UNKNOWN, split count: 1, time: 1.17
Output dim: 0, lower bound: -20.9651837, upper bound: 20.9642482
NS_B2, status: Status.UNKNOWN, split count: 1, time: 1.17
Output dim: 0, lower bound: -20.9652728, upper bound: 20.9652728

## BFS NS instance: NS_B1

### Backsubstitution after applying NS history:
0: -6.3216605, 17.6058750, -6.4440327, 17.8407555, -24.1624146, 24.0499077
1: -15.7496443, 27.0957870, -16.0771294, 27.2822037, -43.0318413, 43.1729088
2: -10.9911642, 24.5486259, -11.2238207, 24.8011894, -35.7923546, 35.7724457
3: -16.9977264, 29.9324017, -17.3475590, 30.2265320, -47.2242584, 47.2799492
4: -15.5509911, 30.2883797, -15.8441029, 30.6173515, -46.1683426, 46.1324768

Time for backsubstitution: 0.70 seconds

### NS candidates at layer 1
type: B, layer: 1, pos: 12
type: A, layer: 1, pos: 12
type: A, layer: 1, pos: 1
type: B, layer: 1, pos: 47
type: A, layer: 1, pos: 47
type: A, layer: 1, pos: 45
type: B, layer: 1, pos: 45
type: B, layer: 1, pos: 15
type: A, layer: 1, pos: 15
type: B, layer: 1, pos: 28
type: B, layer: 1, pos: 6
type: B, layer: 1, pos: 24
type: A, layer: 1, pos: 24
type: B, layer: 1, pos: 9
type: A, layer: 1, pos: 37
type: A, layer: 1, pos: 9
type: A, layer: 1, pos: 28
type: A, layer: 1, pos: 6
type: B, layer: 1, pos: 8
type: A, layer: 1, pos: 39
type: B, layer: 1, pos: 37
type: B, layer: 1, pos: 48
type: A, layer: 1, pos: 8
type: B, layer: 1, pos: 38
type: A, layer: 1, pos: 38
type: B, layer: 1, pos: 39
type: A, layer: 1, pos: 48
type: A, layer: 1, pos: 20
type: B, layer: 1, pos: 20
type: B, layer: 1, pos: 26
type: A, layer: 1, pos: 26
type: B, layer: 1, pos: 41
type: A, layer: 1, pos: 41
type: B, layer: 1, pos: 5
type: B, layer: 1, pos: 0
type: A, layer: 1, pos: 5
type: A, layer: 1, pos: 31
type: B, layer: 1, pos: 31
type: A, layer: 1, pos: 0

Time for candidate selection: 0.06 seconds

### Candidate
type: B, layer: 1, pos: 12

## Relational analysis of NS_B1_B1
Optimization infeasible because this subproblem isn't reachable.

### Candidate
type: A, layer: 1, pos: 12

### Candidate
type: A, layer: 1, pos: 1

## Relational analysis of NS_B1_A1

### Relational analysis result of NS_B1_A1
Status: Status.UNKNOWN
Output dim: 0, lower bound: -20.9642097, upper bound: 20.9642097
time: 0.61 seconds

## Relational analysis of NS_B1_A2

### Relational analysis result of NS_B1_A2
Status: Status.UNKNOWN
Output dim: 0, lower bound: -20.9642097, upper bound: 20.9642482
time: 0.55 seconds

## BFS NS instance: NS_B2

### Backsubstitution after applying NS history:
0: -6.6128454, 18.3588104, -6.3490982, 17.6925869, -24.3054295, 24.7079086
1: -16.5121365, 28.1943169, -15.8180981, 27.2195091, -43.7316437, 44.0124130
2: -11.5172834, 25.5670700, -11.0441561, 24.6677074, -36.1849823, 36.6112251
3: -17.7977448, 31.1728382, -17.0784149, 30.0714798, -47.8692245, 48.2512512
4: -16.2582874, 31.5556946, -15.6314316, 30.4368286, -46.6951141, 47.1871262

Time for backsubstitution: 0.70 seconds

### NS candidates at layer 1
type: B, layer: 1, pos: 12
type: A, layer: 1, pos: 12
type: A, layer: 1, pos: 1
type: A, layer: 1, pos: 47
type: B, layer: 1, pos: 47
type: A, layer: 1, pos: 15
type: B, layer: 1, pos: 15
type: B, layer: 1, pos: 45
type: A, layer: 1, pos: 45
type: B, layer: 1, pos: 28
type: A, layer: 1, pos: 28
type: A, layer: 1, pos: 6
type: B, layer: 1, pos: 9
type: A, layer: 1, pos: 9
type: B, layer: 1, pos: 24
type: A, layer: 1, pos: 24
type: B, layer: 1, pos: 37
type: B, layer: 1, pos: 6
type: A, layer: 1, pos: 37
type: A, layer: 1, pos: 48
type: B, layer: 1, pos: 8
type: B, layer: 1, pos: 38
type: A, layer: 1, pos: 38
type: B, layer: 1, pos: 48
type: A, layer: 1, pos: 8
type: A, layer: 1, pos: 39
type: B, layer: 1, pos: 39
type: A, layer: 1, pos: 20
type: B, layer: 1, pos: 20
type: B, layer: 1, pos: 26
type: A, layer: 1, pos: 26
type: A, layer: 1, pos: 5
type: B, layer: 1, pos: 5
type: A, layer: 1, pos: 31
type: B, layer: 1, pos: 31
type: B, layer: 1, pos: 41
type: A, layer: 1, pos: 41
type: B, layer: 1, pos: 0
type: A, layer: 1, pos: 0

Time for candidate selection: 0.06 seconds

### Candidate
type: B, layer: 1, pos: 12

## Relational analysis of NS_B2_B1

### Relational analysis result of NS_B2_B1
Status: Status.UNKNOWN
Output dim: 0, lower bound: -20.9648630, upper bound: 20.9652728
time: 0.53 seconds

## Relational analysis of NS_B2_B2

### Relational analysis result of NS_B2_B2
Status: Status.UNKNOWN
Output dim: 0, lower bound: -20.9648630, upper bound: 20.9648630
time: 0.53 seconds

## Summary of splitting at layer (split count: 1)
- Time for NS candidates: 1.83 seconds
NS_B1_A1, status: Status.UNKNOWN, split count: 2, time: 1.83
Output dim: 0, lower bound: -20.9642097, upper bound: 20.9642097
NS_B1_A2, status: Status.UNKNOWN, split count: 2, time: 1.83
Output dim: 0, lower bound: -20.9642097, upper bound: 20.9642482
NS_B2_B1, status: Status.UNKNOWN, split count: 2, time: 1.83
Output dim: 0, lower bound: -20.9648630, upper bound: 20.9652728
NS_B2_B2, status: Status.UNKNOWN, split count: 2, time: 1.83
Output dim: 0, lower bound: -20.9648630, upper bound: 20.9648630

## BFS NS instance: NS_B1_A1

### Backsubstitution after applying NS history:
0: -6.4440327, 17.8407555, -6.4440327, 17.8407555, -24.2847881, 24.2847881
1: -16.0771294, 27.2822037, -16.0771294, 27.2822037, -43.3593254, 43.3593216
2: -11.2238207, 24.8011894, -11.2238207, 24.8011894, -36.0250092, 36.0250092
3: -17.3475590, 30.2265320, -17.3475590, 30.2265320, -47.5740891, 47.5740891
4: -15.8441029, 30.6173515, -15.8441029, 30.6173515, -46.4614525, 46.4614525

Time for backsubstitution: 0.70 seconds

### NS candidates at layer 1
type: A, layer: 1, pos: 12
type: B, layer: 1, pos: 12
type: A, layer: 1, pos: 45
type: B, layer: 1, pos: 45
type: A, layer: 1, pos: 47
type: B, layer: 1, pos: 47
type: A, layer: 1, pos: 15
type: B, layer: 1, pos: 15
type: A, layer: 1, pos: 28
type: B, layer: 1, pos: 28
type: B, layer: 1, pos: 6
type: A, layer: 1, pos: 6
type: A, layer: 1, pos: 24
type: B, layer: 1, pos: 24
type: A, layer: 1, pos: 39
type: B, layer: 1, pos: 39
type: A, layer: 1, pos: 8
type: B, layer: 1, pos: 8
type: A, layer: 1, pos: 9
type: B, layer: 1, pos: 9
type: A, layer: 1, pos: 37
type: B, layer: 1, pos: 37
type: A, layer: 1, pos: 38
type: B, layer: 1, pos: 38
type: A, layer: 1, pos: 48
type: B, layer: 1, pos: 48
type: A, layer: 1, pos: 20
type: B, layer: 1, pos: 20
type: A, layer: 1, pos: 26
type: B, layer: 1, pos: 26
type: A, layer: 1, pos: 41
type: B, layer: 1, pos: 41
type: B, layer: 1, pos: 31
type: A, layer: 1, pos: 31
type: A, layer: 1, pos: 5
type: B, layer: 1, pos: 5
type: B, layer: 1, pos: 0
type: A, layer: 1, pos: 0

Time for candidate selection: 0.06 seconds

### Candidate
type: A, layer: 1, pos: 12

### Candidate
type: B, layer: 1, pos: 12

### Candidate
type: A, layer: 1, pos: 45

## Relational analysis of NS_B1_A1_A1

### Relational analysis result of NS_B1_A1_A1
Status: Status.UNKNOWN
Output dim: 0, lower bound: -20.9585505, upper bound: 20.9554247
time: 0.47 seconds

## Relational analysis of NS_B1_A1_A2

### Relational analysis result of NS_B1_A1_A2
Status: Status.VERIFIED
Output dim: 0, lower bound: -20.6915878, upper bound: 20.6649274
time: 0.53 seconds

## BFS NS instance: NS_B1_A2

### Backsubstitution after applying NS history:
0: -6.3490982, 17.6925869, -6.4440327, 17.8407555, -24.1898518, 24.1366196
1: -15.8180981, 27.2195091, -16.0771294, 27.2822037, -43.1002998, 43.2966385
2: -11.0441561, 24.6677074, -11.2238207, 24.8011894, -35.8453445, 35.8915291
3: -17.0784149, 30.0714798, -17.3475590, 30.2265320, -47.3049469, 47.4190331
4: -15.6314316, 30.4368286, -15.8441029, 30.6173515, -46.2487831, 46.2809219

Time for backsubstitution: 0.70 seconds

### NS candidates at layer 1
type: A, layer: 1, pos: 12
type: B, layer: 1, pos: 12
type: A, layer: 1, pos: 47
type: B, layer: 1, pos: 47
type: B, layer: 1, pos: 45
type: A, layer: 1, pos: 45
type: B, layer: 1, pos: 15
type: A, layer: 1, pos: 15
type: B, layer: 1, pos: 28
type: B, layer: 1, pos: 6
type: B, layer: 1, pos: 24
type: A, layer: 1, pos: 24
type: A, layer: 1, pos: 6
type: A, layer: 1, pos: 39
type: A, layer: 1, pos: 9
type: B, layer: 1, pos: 9
type: A, layer: 1, pos: 28
type: A, layer: 1, pos: 37
type: B, layer: 1, pos: 8
type: B, layer: 1, pos: 37
type: B, layer: 1, pos: 48
type: A, layer: 1, pos: 8
type: B, layer: 1, pos: 38
type: B, layer: 1, pos: 39
type: A, layer: 1, pos: 38
type: A, layer: 1, pos: 48
type: B, layer: 1, pos: 20
type: A, layer: 1, pos: 20
type: B, layer: 1, pos: 26
type: A, layer: 1, pos: 26
type: A, layer: 1, pos: 41
type: B, layer: 1, pos: 41
type: B, layer: 1, pos: 0
type: B, layer: 1, pos: 5
type: A, layer: 1, pos: 31
type: A, layer: 1, pos: 5
type: A, layer: 1, pos: 0
type: B, layer: 1, pos: 31

Time for candidate selection: 0.06 seconds

### Candidate
type: A, layer: 1, pos: 12

### Candidate
type: B, layer: 1, pos: 12

### Candidate
type: A, layer: 1, pos: 47

## Relational analysis of NS_B1_A2_A1

### Relational analysis result of NS_B1_A2_A1
Status: Status.UNKNOWN
Output dim: 0, lower bound: -20.9479303, upper bound: 20.9523051
time: 0.58 seconds

## Relational analysis of NS_B1_A2_A2

### Relational analysis result of NS_B1_A2_A2
Status: Status.UNKNOWN
Output dim: 0, lower bound: -20.9361618, upper bound: 20.9438155
time: 0.57 seconds

## BFS NS instance: NS_B2_B1

### Backsubstitution after applying NS history:
0: -6.4682894, 17.9821701, -6.0461669, 16.9050064, -23.3732948, 24.0283375
1: -16.1270905, 27.6529121, -15.0026617, 26.0916557, -42.2187462, 42.6555710
2: -11.2533112, 25.0652580, -10.4874372, 23.6187649, -34.8720741, 35.5526886
3: -17.4036999, 30.5590115, -16.2466373, 28.7926636, -46.1963654, 46.8056488
4: -15.9182625, 30.9278564, -14.9184227, 29.1219597, -45.0402222, 45.8462791

Time for backsubstitution: 0.70 seconds

### NS candidates at layer 1
type: A, layer: 1, pos: 1
type: A, layer: 1, pos: 12
type: A, layer: 1, pos: 15
type: B, layer: 1, pos: 15
type: A, layer: 1, pos: 47
type: B, layer: 1, pos: 47
type: B, layer: 1, pos: 45
type: A, layer: 1, pos: 45
type: A, layer: 1, pos: 39
type: B, layer: 1, pos: 39
type: A, layer: 1, pos: 38
type: B, layer: 1, pos: 38
type: B, layer: 1, pos: 8
type: B, layer: 1, pos: 9
type: A, layer: 1, pos: 9
type: B, layer: 1, pos: 20
type: B, layer: 1, pos: 37
type: A, layer: 1, pos: 37
type: A, layer: 1, pos: 8
type: B, layer: 1, pos: 28
type: A, layer: 1, pos: 20
type: A, layer: 1, pos: 6
type: A, layer: 1, pos: 28
type: B, layer: 1, pos: 6
type: A, layer: 1, pos: 48
type: B, layer: 1, pos: 48
type: A, layer: 1, pos: 24
type: A, layer: 1, pos: 31
type: B, layer: 1, pos: 31
type: B, layer: 1, pos: 24
type: B, layer: 1, pos: 26
type: B, layer: 1, pos: 41
type: A, layer: 1, pos: 26
type: A, layer: 1, pos: 41
type: A, layer: 1, pos: 0
type: A, layer: 1, pos: 5
type: B, layer: 1, pos: 5
type: B, layer: 1, pos: 0

Time for candidate selection: 0.06 seconds

### Candidate
type: A, layer: 1, pos: 1

## Relational analysis of NS_B2_B1_A1
Optimization infeasible because this subproblem isn't reachable.

### Candidate
type: A, layer: 1, pos: 12

## Relational analysis of NS_B2_B1_A1

### Relational analysis result of NS_B2_B1_A1
Status: Status.UNKNOWN
Output dim: 0, lower bound: -20.9648630, upper bound: 20.9648630
time: 0.56 seconds

## Relational analysis of NS_B2_B1_A2

### Relational analysis result of NS_B2_B1_A2
Status: Status.UNKNOWN
Output dim: 0, lower bound: -20.9648630, upper bound: 20.9648630
time: 0.54 seconds

## BFS NS instance: NS_B2_B2

### Backsubstitution after applying NS history:
0: -6.4305573, 17.8925438, -6.0427232, 16.7792358, -23.2097893, 23.9352665
1: -16.0598736, 27.5127029, -14.9130592, 25.9235287, -41.9834023, 42.4257622
2: -11.1947908, 24.9361191, -10.4743414, 23.4187317, -34.6135178, 35.4104614
3: -17.3018456, 30.4061832, -16.1326942, 28.6292896, -45.9311371, 46.5388794
4: -15.7858572, 30.7771568, -14.7950649, 28.7693596, -44.5552101, 45.5722122

Time for backsubstitution: 0.71 seconds

### NS candidates at layer 1
type: A, layer: 1, pos: 1
type: A, layer: 1, pos: 45
type: B, layer: 1, pos: 47
type: B, layer: 1, pos: 45
type: A, layer: 1, pos: 12
type: A, layer: 1, pos: 47
type: A, layer: 1, pos: 15
type: B, layer: 1, pos: 15
type: B, layer: 1, pos: 28
type: A, layer: 1, pos: 28
type: B, layer: 1, pos: 8
type: A, layer: 1, pos: 20
type: A, layer: 1, pos: 24
type: B, layer: 1, pos: 9
type: A, layer: 1, pos: 48
type: A, layer: 1, pos: 37
type: B, layer: 1, pos: 6
type: B, layer: 1, pos: 37
type: B, layer: 1, pos: 39
type: A, layer: 1, pos: 9
type: A, layer: 1, pos: 6
type: B, layer: 1, pos: 24
type: A, layer: 1, pos: 38
type: B, layer: 1, pos: 38
type: A, layer: 1, pos: 39
type: B, layer: 1, pos: 48
type: A, layer: 1, pos: 8
type: B, layer: 1, pos: 20
type: B, layer: 1, pos: 26
type: A, layer: 1, pos: 26
type: B, layer: 1, pos: 31
type: A, layer: 1, pos: 0
type: A, layer: 1, pos: 31
type: A, layer: 1, pos: 5
type: B, layer: 1, pos: 41
type: B, layer: 1, pos: 5
type: A, layer: 1, pos: 41
type: B, layer: 1, pos: 0

Time for candidate selection: 0.06 seconds

### Candidate
type: A, layer: 1, pos: 1

## Relational analysis of NS_B2_B2_A1

### Relational analysis result of NS_B2_B2_A1
Status: Status.UNKNOWN
Output dim: 0, lower bound: -20.9642097, upper bound: 20.9648221
time: 0.56 seconds

## Relational analysis of NS_B2_B2_A2

### Relational analysis result of NS_B2_B2_A2
Status: Status.UNKNOWN
Output dim: 0, lower bound: -20.9642097, upper bound: 20.9645811
time: 0.57 seconds

## Summary of splitting at layer (split count: 2)
- Time for NS candidates: 1.91 seconds
NS_B1_A1_A1, status: Status.UNKNOWN, split count: 3, time: 1.91
Output dim: 0, lower bound: -20.9585505, upper bound: 20.9554247
NS_B1_A1_A2, status: Status.VERIFIED, split count: 3, time: 1.91
Output dim: 0, lower bound: -20.6915878, upper bound: 20.6649274
NS_B1_A2_A1, status: Status.UNKNOWN, split count: 3, time: 1.91
Output dim: 0, lower bound: -20.9479303, upper bound: 20.9523051
NS_B1_A2_A2, status: Status.UNKNOWN, split count: 3, time: 1.91
Output dim: 0, lower bound: -20.9361618, upper bound: 20.9438155
NS_B2_B1_A1, status: Status.UNKNOWN, split count: 3, time: 1.91
Output dim: 0, lower bound: -20.9648630, upper bound: 20.9648630
NS_B2_B1_A2, status: Status.UNKNOWN, split count: 3, time: 1.91
Output dim: 0, lower bound: -20.9648630, upper bound: 20.9648630
NS_B2_B2_A1, status: Status.UNKNOWN, split count: 3, time: 1.91
Output dim: 0, lower bound: -20.9642097, upper bound: 20.9648221
NS_B2_B2_A2, status: Status.UNKNOWN, split count: 3, time: 1.91
Output dim: 0, lower bound: -20.9642097, upper bound: 20.9645811

## BFS NS instance: NS_B1_A1_A1

### Backsubstitution after applying NS history:
0: -6.2671232, 17.3729401, -6.4440327, 17.8407555, -24.1078777, 23.8169727
1: -15.6261692, 26.5522480, -16.0771294, 27.2822037, -42.9083672, 42.6293755
2: -10.9130974, 24.1637859, -11.2238207, 24.8011894, -35.7142868, 35.3876076
3: -16.8714848, 29.4241104, -17.3475590, 30.2265320, -47.0980148, 46.7716599
4: -15.4132824, 29.8272419, -15.8441029, 30.6173515, -46.0306320, 45.6713333

Time for backsubstitution: 0.71 seconds

### NS candidates at layer 1
type: A, layer: 1, pos: 12
type: B, layer: 1, pos: 12
type: A, layer: 1, pos: 47
type: B, layer: 1, pos: 47
type: B, layer: 1, pos: 15
type: A, layer: 1, pos: 15
type: B, layer: 1, pos: 45
type: A, layer: 1, pos: 28
type: B, layer: 1, pos: 28
type: A, layer: 1, pos: 6
type: A, layer: 1, pos: 24
type: B, layer: 1, pos: 24
type: B, layer: 1, pos: 6
type: A, layer: 1, pos: 9
type: B, layer: 1, pos: 9
type: A, layer: 1, pos: 8
type: B, layer: 1, pos: 8
type: A, layer: 1, pos: 37
type: B, layer: 1, pos: 37
type: B, layer: 1, pos: 38
type: A, layer: 1, pos: 39
type: A, layer: 1, pos: 38
type: B, layer: 1, pos: 39
type: B, layer: 1, pos: 48
type: A, layer: 1, pos: 48
type: B, layer: 1, pos: 20
type: A, layer: 1, pos: 20
type: B, layer: 1, pos: 26
type: A, layer: 1, pos: 26
type: A, layer: 1, pos: 41
type: B, layer: 1, pos: 41
type: A, layer: 1, pos: 31
type: B, layer: 1, pos: 31
type: B, layer: 1, pos: 0
type: A, layer: 1, pos: 0
type: A, layer: 1, pos: 5
type: B, layer: 1, pos: 5

Time for candidate selection: 0.06 seconds

### Candidate
type: A, layer: 1, pos: 12

### Candidate
type: B, layer: 1, pos: 12

### Candidate
type: A, layer: 1, pos: 47

## Relational analysis of NS_B1_A1_A1_A1

### Relational analysis result of NS_B1_A1_A1_A1
Status: Status.UNKNOWN
Output dim: 0, lower bound: -20.9235252, upper bound: 20.9362552
time: 0.59 seconds

## Relational analysis of NS_B1_A1_A1_A2

### Relational analysis result of NS_B1_A1_A1_A2
Status: Status.UNKNOWN
Output dim: 0, lower bound: -20.9108170, upper bound: 20.9172336
time: 0.51 seconds

## BFS NS instance: NS_B1_A2_A1

### Backsubstitution after applying NS history:
0: -6.0017405, 16.8577003, -6.2901692, 17.4559555, -23.4576950, 23.1478672
1: -15.0798769, 25.8949661, -15.7031040, 26.7174149, -41.7972908, 41.5980682
2: -10.4755583, 23.4678841, -10.9576550, 24.2829380, -34.7584953, 34.4255371
3: -16.1910629, 28.6209011, -16.9319897, 29.5866947, -45.7777557, 45.5528870
4: -14.6383791, 29.0156803, -15.4524279, 29.9828930, -44.6212692, 44.4681091

Time for backsubstitution: 0.70 seconds

### NS candidates at layer 1
type: B, layer: 1, pos: 12
type: A, layer: 1, pos: 12
type: B, layer: 1, pos: 15
type: B, layer: 1, pos: 45
type: A, layer: 1, pos: 45
type: B, layer: 1, pos: 47
type: A, layer: 1, pos: 15
type: B, layer: 1, pos: 28
type: B, layer: 1, pos: 6
type: A, layer: 1, pos: 28
type: A, layer: 1, pos: 24
type: B, layer: 1, pos: 48
type: B, layer: 1, pos: 9
type: B, layer: 1, pos: 24
type: A, layer: 1, pos: 9
type: A, layer: 1, pos: 6
type: A, layer: 1, pos: 37
type: B, layer: 1, pos: 39
type: B, layer: 1, pos: 37
type: B, layer: 1, pos: 8
type: B, layer: 1, pos: 38
type: A, layer: 1, pos: 38
type: A, layer: 1, pos: 8
type: A, layer: 1, pos: 39
type: B, layer: 1, pos: 26
type: A, layer: 1, pos: 26
type: A, layer: 1, pos: 48
type: A, layer: 1, pos: 20
type: B, layer: 1, pos: 20
type: B, layer: 1, pos: 41
type: A, layer: 1, pos: 41
type: A, layer: 1, pos: 5
type: A, layer: 1, pos: 0
type: B, layer: 1, pos: 0
type: B, layer: 1, pos: 31
type: A, layer: 1, pos: 31
type: B, layer: 1, pos: 5

Time for candidate selection: 0.06 seconds

### Candidate
type: B, layer: 1, pos: 12

### Candidate
type: A, layer: 1, pos: 12

### Candidate
type: B, layer: 1, pos: 15

## Relational analysis of NS_B1_A2_A1_B1
Optimization infeasible because this subproblem isn't reachable.

### Candidate
type: B, layer: 1, pos: 45

## Relational analysis of NS_B1_A2_A1_B1
Optimization infeasible because this subproblem isn't reachable.

### Candidate
type: A, layer: 1, pos: 45

### Candidate
type: B, layer: 1, pos: 47

## Relational analysis of NS_B1_A2_A1_B1

### Relational analysis result of NS_B1_A2_A1_B1
Status: Status.UNKNOWN
Output dim: 0, lower bound: -20.9431946, upper bound: 20.9438155
time: 0.68 seconds

## Relational analysis of NS_B1_A2_A1_B2

### Relational analysis result of NS_B1_A2_A1_B2
Status: Status.UNKNOWN
Output dim: 0, lower bound: -20.9431946, upper bound: 20.9438155
time: 0.53 seconds

## BFS NS instance: NS_B1_A2_A2

### Backsubstitution after applying NS history:
0: -6.2620363, 17.4723530, -6.4440327, 17.8407555, -24.1027889, 23.9163857
1: -15.6027746, 26.8951664, -16.0771294, 27.2822037, -42.8849678, 42.9722900
2: -10.8925056, 24.3697147, -11.2238207, 24.8011894, -35.6936951, 35.5935364
3: -16.8417645, 29.7059155, -17.3475590, 30.2265320, -47.0682983, 47.0534630
4: -15.4047327, 30.0687370, -15.8441029, 30.6173515, -46.0220833, 45.9128342

Time for backsubstitution: 0.71 seconds

### NS candidates at layer 1
type: A, layer: 1, pos: 12
type: B, layer: 1, pos: 12
type: B, layer: 1, pos: 45
type: A, layer: 1, pos: 45
type: B, layer: 1, pos: 47
type: B, layer: 1, pos: 15
type: A, layer: 1, pos: 15
type: B, layer: 1, pos: 28
type: B, layer: 1, pos: 6
type: A, layer: 1, pos: 24
type: A, layer: 1, pos: 6
type: B, layer: 1, pos: 24
type: A, layer: 1, pos: 9
type: B, layer: 1, pos: 8
type: A, layer: 1, pos: 28
type: B, layer: 1, pos: 9
type: A, layer: 1, pos: 37
type: A, layer: 1, pos: 39
type: B, layer: 1, pos: 37
type: B, layer: 1, pos: 48
type: B, layer: 1, pos: 39
type: A, layer: 1, pos: 8
type: B, layer: 1, pos: 38
type: A, layer: 1, pos: 38
type: A, layer: 1, pos: 48
type: A, layer: 1, pos: 20
type: B, layer: 1, pos: 20
type: B, layer: 1, pos: 26
type: A, layer: 1, pos: 26
type: A, layer: 1, pos: 41
type: B, layer: 1, pos: 41
type: B, layer: 1, pos: 5
type: A, layer: 1, pos: 31
type: A, layer: 1, pos: 5
type: A, layer: 1, pos: 0
type: B, layer: 1, pos: 31
type: B, layer: 1, pos: 0

Time for candidate selection: 0.06 seconds

### Candidate
type: A, layer: 1, pos: 12

### Candidate
type: B, layer: 1, pos: 12

### Candidate
type: B, layer: 1, pos: 45

## Relational analysis of NS_B1_A2_A2_B1

### Relational analysis result of NS_B1_A2_A2_B1
Status: Status.VERIFIED
Output dim: 0, lower bound: -20.6935967, upper bound: 20.6663848
time: 0.52 seconds

## Relational analysis of NS_B1_A2_A2_B2

### Relational analysis result of NS_B1_A2_A2_B2
Status: Status.VERIFIED
Output dim: 0, lower bound: -20.6903902, upper bound: 20.6639311
time: 0.53 seconds

## BFS NS instance: NS_B2_B1_A1

### Backsubstitution after applying NS history:
0: -6.2940121, 17.5322781, -6.0461669, 16.9050064, -23.1990185, 23.5784435
1: -15.6561022, 27.0123844, -15.0026617, 26.0916557, -41.7477570, 42.0150452
2: -10.9337883, 24.4664974, -10.4874372, 23.6187649, -34.5525513, 34.9539337
3: -16.9244709, 29.8317928, -16.2466373, 28.7926636, -45.7171326, 46.0784302
4: -15.5113659, 30.1773357, -14.9184227, 29.1219597, -44.6333160, 45.0957565

Time for backsubstitution: 0.71 seconds

### NS candidates at layer 1
type: A, layer: 1, pos: 1
type: A, layer: 1, pos: 15
type: B, layer: 1, pos: 15
type: A, layer: 1, pos: 47
type: B, layer: 1, pos: 47
type: B, layer: 1, pos: 45
type: A, layer: 1, pos: 45
type: A, layer: 1, pos: 39
type: B, layer: 1, pos: 39
type: B, layer: 1, pos: 38
type: A, layer: 1, pos: 38
type: A, layer: 1, pos: 8
type: B, layer: 1, pos: 8
type: B, layer: 1, pos: 9
type: A, layer: 1, pos: 9
type: B, layer: 1, pos: 20
type: B, layer: 1, pos: 37
type: A, layer: 1, pos: 20
type: A, layer: 1, pos: 37
type: A, layer: 1, pos: 6
type: B, layer: 1, pos: 28
type: A, layer: 1, pos: 28
type: B, layer: 1, pos: 6
type: B, layer: 1, pos: 48
type: A, layer: 1, pos: 48
type: A, layer: 1, pos: 31
type: B, layer: 1, pos: 31
type: A, layer: 1, pos: 24
type: B, layer: 1, pos: 24
type: B, layer: 1, pos: 26
type: A, layer: 1, pos: 26
type: B, layer: 1, pos: 41
type: A, layer: 1, pos: 41
type: A, layer: 1, pos: 5
type: B, layer: 1, pos: 5
type: B, layer: 1, pos: 0
type: A, layer: 1, pos: 0

Time for candidate selection: 0.06 seconds

### Candidate
type: A, layer: 1, pos: 1

### Candidate
type: A, layer: 1, pos: 15

## Relational analysis of NS_B2_B1_A1_A1

### Relational analysis result of NS_B2_B1_A1_A1
Status: Status.VERIFIED
Output dim: 0, lower bound: -20.7442635, upper bound: 20.7092184
time: 0.70 seconds

## Relational analysis of NS_B2_B1_A1_A2

### Relational analysis result of NS_B2_B1_A1_A2
Status: Status.UNKNOWN
Output dim: 0, lower bound: -20.9637932, upper bound: 20.9642643
time: 0.63 seconds

## BFS NS instance: NS_B2_B1_A2

### Backsubstitution after applying NS history:
0: -6.2726994, 17.3570690, -6.0461669, 16.9050064, -23.1777058, 23.4032307
1: -15.5273714, 26.7728806, -15.0026617, 26.0916557, -41.6190262, 41.7755432
2: -10.8936138, 24.1979637, -10.4874372, 23.6187649, -34.5123787, 34.6854019
3: -16.7668762, 29.5876045, -16.2466373, 28.7926636, -45.5595360, 45.8342438
4: -15.3402452, 29.7392197, -14.9184227, 29.1219597, -44.4621964, 44.6576424

Time for backsubstitution: 0.72 seconds

### NS candidates at layer 1
type: A, layer: 1, pos: 1
type: B, layer: 1, pos: 15
type: A, layer: 1, pos: 15
type: B, layer: 1, pos: 45
type: A, layer: 1, pos: 45
type: A, layer: 1, pos: 47
type: B, layer: 1, pos: 47
type: A, layer: 1, pos: 39
type: B, layer: 1, pos: 38
type: B, layer: 1, pos: 39
type: A, layer: 1, pos: 6
type: A, layer: 1, pos: 8
type: A, layer: 1, pos: 38
type: B, layer: 1, pos: 20
type: A, layer: 1, pos: 9
type: B, layer: 1, pos: 48
type: B, layer: 1, pos: 37
type: B, layer: 1, pos: 9
type: A, layer: 1, pos: 28
type: B, layer: 1, pos: 6
type: A, layer: 1, pos: 48
type: A, layer: 1, pos: 37
type: B, layer: 1, pos: 8
type: B, layer: 1, pos: 28
type: A, layer: 1, pos: 31
type: A, layer: 1, pos: 20
type: B, layer: 1, pos: 31
type: B, layer: 1, pos: 24
type: A, layer: 1, pos: 24
type: B, layer: 1, pos: 0
type: A, layer: 1, pos: 5
type: B, layer: 1, pos: 5
type: B, layer: 1, pos: 41
type: A, layer: 1, pos: 41
type: A, layer: 1, pos: 0
type: B, layer: 1, pos: 26
type: A, layer: 1, pos: 26

Time for candidate selection: 0.06 seconds

### Candidate
type: A, layer: 1, pos: 1

### Candidate
type: B, layer: 1, pos: 15

## Relational analysis of NS_B2_B1_A2_B1

### Relational analysis result of NS_B2_B1_A2_B1
Status: Status.VERIFIED
Output dim: 0, lower bound: -20.7092184, upper bound: 20.7442635
time: 0.55 seconds

## Relational analysis of NS_B2_B1_A2_B2

### Relational analysis result of NS_B2_B1_A2_B2
Status: Status.UNKNOWN
Output dim: 0, lower bound: -20.9637932, upper bound: 20.9642643
time: 0.55 seconds

## BFS NS instance: NS_B2_B2_A1

### Backsubstitution after applying NS history:
0: -6.3032446, 17.4915447, -6.0427232, 16.7792358, -23.0824757, 23.5342636
1: -15.7344370, 26.7710838, -14.9130592, 25.9235287, -41.6579666, 41.6841354
2: -10.9783573, 24.3316841, -10.4743414, 23.4187317, -34.3970833, 34.8060265
3: -16.9640579, 29.6479645, -16.1326942, 28.6292896, -45.5933456, 45.7806587
4: -15.4797134, 30.0414028, -14.7950649, 28.7693596, -44.2490616, 44.8364677

Time for backsubstitution: 0.72 seconds

### NS candidates at layer 1
type: A, layer: 1, pos: 12
type: B, layer: 1, pos: 47
type: A, layer: 1, pos: 47
type: A, layer: 1, pos: 45
type: B, layer: 1, pos: 45
type: A, layer: 1, pos: 15
type: B, layer: 1, pos: 15
type: A, layer: 1, pos: 28
type: A, layer: 1, pos: 6
type: B, layer: 1, pos: 6
type: A, layer: 1, pos: 24
type: A, layer: 1, pos: 48
type: B, layer: 1, pos: 28
type: B, layer: 1, pos: 8
type: B, layer: 1, pos: 9
type: B, layer: 1, pos: 24
type: A, layer: 1, pos: 9
type: B, layer: 1, pos: 39
type: A, layer: 1, pos: 38
type: A, layer: 1, pos: 37
type: B, layer: 1, pos: 37
type: A, layer: 1, pos: 20
type: B, layer: 1, pos: 38
type: B, layer: 1, pos: 48
type: A, layer: 1, pos: 8
type: A, layer: 1, pos: 39
type: B, layer: 1, pos: 26
type: A, layer: 1, pos: 26
type: B, layer: 1, pos: 20
type: A, layer: 1, pos: 0
type: B, layer: 1, pos: 31
type: A, layer: 1, pos: 41
type: B, layer: 1, pos: 41
type: A, layer: 1, pos: 5
type: B, layer: 1, pos: 5
type: A, layer: 1, pos: 31
type: B, layer: 1, pos: 0

Time for candidate selection: 0.06 seconds

### Candidate
type: A, layer: 1, pos: 12

## Relational analysis of NS_B2_B2_A1_A1
Optimization infeasible because this subproblem isn't reachable.

### Candidate
type: B, layer: 1, pos: 47

## Relational analysis of NS_B2_B2_A1_B1

### Relational analysis result of NS_B2_B2_A1_B1
Status: Status.UNKNOWN
Output dim: 0, lower bound: -20.9522721, upper bound: 20.9485296
time: 0.56 seconds

## Relational analysis of NS_B2_B2_A1_B2

### Relational analysis result of NS_B2_B2_A1_B2
Status: Status.UNKNOWN
Output dim: 0, lower bound: -20.9361618, upper bound: 20.9403090
time: 0.57 seconds

## BFS NS instance: NS_B2_B2_A2

### Backsubstitution after applying NS history:
0: -6.1677237, 17.2251701, -6.0427232, 16.7792358, -22.9469528, 23.2678928
1: -15.3650265, 26.5364742, -14.9130592, 25.9235287, -41.2885551, 41.4495316
2: -10.7221899, 24.0357971, -10.4743414, 23.4187317, -34.1409073, 34.5101395
3: -16.5831528, 29.3041401, -16.1326942, 28.6292896, -45.2124405, 45.4368324
4: -15.1602678, 29.6561813, -14.7950649, 28.7693596, -43.9296188, 44.4512405

Time for backsubstitution: 0.72 seconds

### NS candidates at layer 1
type: B, layer: 1, pos: 47
type: A, layer: 1, pos: 45
type: A, layer: 1, pos: 12
type: A, layer: 1, pos: 47
type: B, layer: 1, pos: 45
type: A, layer: 1, pos: 15
type: B, layer: 1, pos: 15
type: A, layer: 1, pos: 28
type: B, layer: 1, pos: 28
type: B, layer: 1, pos: 8
type: A, layer: 1, pos: 24
type: B, layer: 1, pos: 6
type: B, layer: 1, pos: 9
type: A, layer: 1, pos: 48
type: A, layer: 1, pos: 37
type: B, layer: 1, pos: 24
type: A, layer: 1, pos: 9
type: A, layer: 1, pos: 6
type: B, layer: 1, pos: 39
type: B, layer: 1, pos: 37
type: A, layer: 1, pos: 38
type: B, layer: 1, pos: 48
type: A, layer: 1, pos: 20
type: A, layer: 1, pos: 39
type: B, layer: 1, pos: 38
type: A, layer: 1, pos: 8
type: B, layer: 1, pos: 26
type: A, layer: 1, pos: 26
type: B, layer: 1, pos: 31
type: A, layer: 1, pos: 0
type: B, layer: 1, pos: 20
type: B, layer: 1, pos: 41
type: A, layer: 1, pos: 41
type: A, layer: 1, pos: 31
type: A, layer: 1, pos: 5
type: B, layer: 1, pos: 0
type: B, layer: 1, pos: 5

Time for candidate selection: 0.06 seconds

### Candidate
type: B, layer: 1, pos: 47

## Relational analysis of NS_B2_B2_A2_B1

### Relational analysis result of NS_B2_B2_A2_B1
Status: Status.UNKNOWN
Output dim: 0, lower bound: -20.9522721, upper bound: 20.9544784
time: 0.56 seconds

## Relational analysis of NS_B2_B2_A2_B2

### Relational analysis result of NS_B2_B2_A2_B2
Status: Status.UNKNOWN
Output dim: 0, lower bound: -20.9361618, upper bound: 20.9480285
time: 0.64 seconds

## Summary of splitting at layer (split count: 3)
- Time for NS candidates: 1.98 seconds
NS_B1_A1_A1_A1, status: Status.UNKNOWN, split count: 4, time: 1.98
Output dim: 0, lower bound: -20.9235252, upper bound: 20.9362552
NS_B1_A1_A1_A2, status: Status.UNKNOWN, split count: 4, time: 1.98
Output dim: 0, lower bound: -20.9108170, upper bound: 20.9172336
NS_B1_A2_A1_B1, status: Status.UNKNOWN, split count: 4, time: 1.98
Output dim: 0, lower bound: -20.9431946, upper bound: 20.9438155
NS_B1_A2_A1_B2, status: Status.UNKNOWN, split count: 4, time: 1.98
Output dim: 0, lower bound: -20.9431946, upper bound: 20.9438155
NS_B1_A2_A2_B1, status: Status.VERIFIED, split count: 4, time: 1.98
Output dim: 0, lower bound: -20.6935967, upper bound: 20.6663848
NS_B1_A2_A2_B2, status: Status.VERIFIED, split count: 4, time: 1.98
Output dim: 0, lower bound: -20.6903902, upper bound: 20.6639311
NS_B2_B1_A1_A1, status: Status.VERIFIED, split count: 4, time: 1.98
Output dim: 0, lower bound: -20.7442635, upper bound: 20.7092184
NS_B2_B1_A1_A2, status: Status.UNKNOWN, split count: 4, time: 1.98
Output dim: 0, lower bound: -20.9637932, upper bound: 20.9642643
NS_B2_B1_A2_B1, status: Status.VERIFIED, split count: 4, time: 1.98
Output dim: 0, lower bound: -20.7092184, upper bound: 20.7442635
NS_B2_B1_A2_B2, status: Status.UNKNOWN, split count: 4, time: 1.98
Output dim: 0, lower bound: -20.9637932, upper bound: 20.9642643
NS_B2_B2_A1_B1, status: Status.UNKNOWN, split count: 4, time: 1.98
Output dim: 0, lower bound: -20.9522721, upper bound: 20.9485296
NS_B2_B2_A1_B2, status: Status.UNKNOWN, split count: 4, time: 1.98
Output dim: 0, lower bound: -20.9361618, upper bound: 20.9403090
NS_B2_B2_A2_B1, status: Status.UNKNOWN, split count: 4, time: 1.98
Output dim: 0, lower bound: -20.9522721, upper bound: 20.9544784
NS_B2_B2_A2_B2, status: Status.UNKNOWN, split count: 4, time: 1.98
Output dim: 0, lower bound: -20.9361618, upper bound: 20.9480285

## BFS NS instance: NS_B1_A1_A1_A1

### Backsubstitution after applying NS history:
0: -6.0846200, 16.9946613, -6.2901692, 17.4559555, -23.5405750, 23.2848301
1: -15.3345718, 25.8991718, -15.7031040, 26.7174149, -42.0519867, 41.6022758
2: -10.6436501, 23.5850830, -10.9576550, 24.2829380, -34.9265862, 34.5427399
3: -16.4320755, 28.7094727, -16.9319897, 29.5866947, -46.0187683, 45.6414490
4: -14.8261871, 29.1902122, -15.4524279, 29.9828930, -44.8090782, 44.6426315

Time for backsubstitution: 0.71 seconds

### NS candidates at layer 1
type: B, layer: 1, pos: 12
type: A, layer: 1, pos: 12
type: B, layer: 1, pos: 15
type: A, layer: 1, pos: 15
type: B, layer: 1, pos: 47
type: A, layer: 1, pos: 28
type: B, layer: 1, pos: 28
type: B, layer: 1, pos: 45
type: B, layer: 1, pos: 6
type: A, layer: 1, pos: 24
type: A, layer: 1, pos: 6
type: B, layer: 1, pos: 39
type: B, layer: 1, pos: 9
type: B, layer: 1, pos: 24
type: B, layer: 1, pos: 48
type: A, layer: 1, pos: 37
type: A, layer: 1, pos: 9
type: B, layer: 1, pos: 37
type: B, layer: 1, pos: 8
type: A, layer: 1, pos: 8
type: B, layer: 1, pos: 38
type: A, layer: 1, pos: 38
type: A, layer: 1, pos: 20
type: A, layer: 1, pos: 39
type: A, layer: 1, pos: 48
type: B, layer: 1, pos: 26
type: A, layer: 1, pos: 26
type: B, layer: 1, pos: 20
type: A, layer: 1, pos: 5
type: B, layer: 1, pos: 41
type: A, layer: 1, pos: 41
type: A, layer: 1, pos: 0
type: B, layer: 1, pos: 31
type: B, layer: 1, pos: 0
type: B, layer: 1, pos: 5
type: A, layer: 1, pos: 31

Time for candidate selection: 0.06 seconds

### Candidate
type: B, layer: 1, pos: 12

### Candidate
type: A, layer: 1, pos: 12

### Candidate
type: B, layer: 1, pos: 15

## Relational analysis of NS_B1_A1_A1_A1_B1
Optimization infeasible because this subproblem isn't reachable.

### Candidate
type: A, layer: 1, pos: 15

### Candidate
type: B, layer: 1, pos: 47

## Relational analysis of NS_B1_A1_A1_A1_B1

### Relational analysis result of NS_B1_A1_A1_A1_B1
Status: Status.UNKNOWN
Output dim: 0, lower bound: -20.9108170, upper bound: 20.9172336
time: 0.62 seconds

## Relational analysis of NS_B1_A1_A1_A1_B2

### Relational analysis result of NS_B1_A1_A1_A1_B2
Status: Status.UNKNOWN
Output dim: 0, lower bound: -20.9108170, upper bound: 20.9172336
time: 0.52 seconds

## BFS NS instance: NS_B1_A1_A1_A2

### Backsubstitution after applying NS history:
0: -6.1871290, 17.1735039, -6.4440327, 17.8407555, -24.0278835, 23.6175365
1: -15.4273548, 26.2602158, -16.0771294, 27.2822037, -42.7095566, 42.3373299
2: -10.7745504, 23.8967419, -11.2238207, 24.8011894, -35.5757408, 35.1205635
3: -16.6525230, 29.0933285, -17.3475590, 30.2265320, -46.8790550, 46.4408836
4: -15.2049751, 29.4977074, -15.8441029, 30.6173515, -45.8223267, 45.3418083

Time for backsubstitution: 0.71 seconds

### NS candidates at layer 1
type: A, layer: 1, pos: 12
type: B, layer: 1, pos: 12
type: B, layer: 1, pos: 47
type: B, layer: 1, pos: 15
type: A, layer: 1, pos: 15
type: B, layer: 1, pos: 45
type: A, layer: 1, pos: 28
type: B, layer: 1, pos: 28
type: A, layer: 1, pos: 6
type: B, layer: 1, pos: 6
type: A, layer: 1, pos: 24
type: B, layer: 1, pos: 24
type: B, layer: 1, pos: 9
type: A, layer: 1, pos: 9
type: A, layer: 1, pos: 37
type: B, layer: 1, pos: 8
type: B, layer: 1, pos: 37
type: A, layer: 1, pos: 8
type: A, layer: 1, pos: 38
type: B, layer: 1, pos: 38
type: B, layer: 1, pos: 39
type: A, layer: 1, pos: 48
type: B, layer: 1, pos: 48
type: A, layer: 1, pos: 20
type: A, layer: 1, pos: 39
type: B, layer: 1, pos: 20
type: B, layer: 1, pos: 26
type: A, layer: 1, pos: 26
type: A, layer: 1, pos: 41
type: B, layer: 1, pos: 41
type: A, layer: 1, pos: 0
type: A, layer: 1, pos: 5
type: B, layer: 1, pos: 5
type: B, layer: 1, pos: 31
type: B, layer: 1, pos: 0
type: A, layer: 1, pos: 31

Time for candidate selection: 0.06 seconds

### Candidate
type: A, layer: 1, pos: 12

### Candidate
type: B, layer: 1, pos: 12

### Candidate
type: B, layer: 1, pos: 47

## Relational analysis of NS_B1_A1_A1_A2_B1

### Relational analysis result of NS_B1_A1_A1_A2_B1
Status: Status.UNKNOWN
Output dim: 0, lower bound: -20.9108170, upper bound: 20.9172336
time: 0.55 seconds

## Relational analysis of NS_B1_A1_A1_A2_B2

### Relational analysis result of NS_B1_A1_A1_A2_B2
Status: Status.UNKNOWN
Output dim: 0, lower bound: -20.9108170, upper bound: 20.9172336
time: 1.20 seconds

## BFS NS instance: NS_B1_A2_A1_B1

### Backsubstitution after applying NS history:
0: -6.0017405, 16.8577003, -6.2165189, 17.3560257, -23.3577614, 23.0742168
1: -15.0798769, 25.8949661, -15.6833458, 26.4522324, -41.5321083, 41.5783119
2: -10.4755583, 23.4678841, -10.8782454, 24.0774555, -34.5530128, 34.3461304
3: -16.1910629, 28.6209011, -16.7920380, 29.3136368, -45.5046997, 45.4129410
4: -14.6383791, 29.0156803, -15.1396618, 29.8034248, -44.4418030, 44.1553421

Time for backsubstitution: 0.73 seconds

### NS candidates at layer 1
type: A, layer: 1, pos: 12
type: B, layer: 1, pos: 12
type: B, layer: 1, pos: 45
type: B, layer: 1, pos: 15
type: B, layer: 1, pos: 28
type: A, layer: 1, pos: 45
type: A, layer: 1, pos: 15
type: B, layer: 1, pos: 6
type: A, layer: 1, pos: 28
type: A, layer: 1, pos: 6
type: B, layer: 1, pos: 24
type: A, layer: 1, pos: 9
type: A, layer: 1, pos: 24
type: B, layer: 1, pos: 9
type: A, layer: 1, pos: 37
type: B, layer: 1, pos: 37
type: B, layer: 1, pos: 48
type: B, layer: 1, pos: 8
type: A, layer: 1, pos: 39
type: B, layer: 1, pos: 38
type: A, layer: 1, pos: 8
type: A, layer: 1, pos: 38
type: A, layer: 1, pos: 48
type: B, layer: 1, pos: 39
type: B, layer: 1, pos: 26
type: A, layer: 1, pos: 26
type: B, layer: 1, pos: 20
type: A, layer: 1, pos: 20
type: A, layer: 1, pos: 41
type: B, layer: 1, pos: 41
type: B, layer: 1, pos: 0
type: A, layer: 1, pos: 5
type: B, layer: 1, pos: 5
type: A, layer: 1, pos: 31
type: A, layer: 1, pos: 0
type: B, layer: 1, pos: 31

Time for candidate selection: 0.06 seconds

### Candidate
type: A, layer: 1, pos: 12

### Candidate
type: B, layer: 1, pos: 12

### Candidate
type: B, layer: 1, pos: 45

### Candidate
type: B, layer: 1, pos: 15

### Candidate
type: B, layer: 1, pos: 28

## Relational analysis of NS_B1_A2_A1_B1_B1
Optimization infeasible because this subproblem isn't reachable.

### Candidate
type: A, layer: 1, pos: 45

### Candidate
type: A, layer: 1, pos: 15

### Candidate
type: B, layer: 1, pos: 6

## Relational analysis of NS_B1_A2_A1_B1_B1

### Relational analysis result of NS_B1_A2_A1_B1_B1
Status: Status.UNKNOWN
Output dim: 0, lower bound: -20.9274296, upper bound: 20.9390731
time: 0.58 seconds

## Relational analysis of NS_B1_A2_A1_B1_B2

### Relational analysis result of NS_B1_A2_A1_B1_B2
Status: Status.UNKNOWN
Output dim: 0, lower bound: -20.9484271, upper bound: 20.9515325
time: 0.61 seconds

## BFS NS instance: NS_B1_A2_A1_B2

### Backsubstitution after applying NS history:
0: -6.0017405, 16.8577003, -6.3588510, 17.6330452, -23.6347847, 23.2165508
1: -15.0798769, 25.8949661, -15.8715429, 26.9774265, -42.0573006, 41.7665100
2: -10.4755583, 23.4678841, -11.0782328, 24.5221272, -34.9976845, 34.5461159
3: -16.1910629, 28.6209011, -17.1150131, 29.8800278, -46.0710907, 45.7359161
4: -14.6383791, 29.0156803, -15.6188936, 30.2740707, -44.9124489, 44.6345749

Time for backsubstitution: 0.72 seconds

### NS candidates at layer 1
type: B, layer: 1, pos: 12
type: A, layer: 1, pos: 12
type: B, layer: 1, pos: 15
type: B, layer: 1, pos: 45
type: A, layer: 1, pos: 45
type: A, layer: 1, pos: 15
type: B, layer: 1, pos: 28
type: B, layer: 1, pos: 6
type: A, layer: 1, pos: 28
type: A, layer: 1, pos: 24
type: B, layer: 1, pos: 48
type: B, layer: 1, pos: 9
type: B, layer: 1, pos: 24
type: A, layer: 1, pos: 9
type: A, layer: 1, pos: 6
type: B, layer: 1, pos: 39
type: A, layer: 1, pos: 37
type: B, layer: 1, pos: 8
type: B, layer: 1, pos: 37
type: B, layer: 1, pos: 38
type: A, layer: 1, pos: 38
type: A, layer: 1, pos: 8
type: A, layer: 1, pos: 39
type: A, layer: 1, pos: 20
type: B, layer: 1, pos: 26
type: A, layer: 1, pos: 26
type: A, layer: 1, pos: 48
type: B, layer: 1, pos: 20
type: B, layer: 1, pos: 41
type: A, layer: 1, pos: 41
type: A, layer: 1, pos: 5
type: A, layer: 1, pos: 0
type: B, layer: 1, pos: 31
type: B, layer: 1, pos: 0
type: A, layer: 1, pos: 31
type: B, layer: 1, pos: 5

Time for candidate selection: 0.06 seconds

### Candidate
type: B, layer: 1, pos: 12

### Candidate
type: A, layer: 1, pos: 12

### Candidate
type: B, layer: 1, pos: 15

### Candidate
type: B, layer: 1, pos: 45

### Candidate
type: A, layer: 1, pos: 45

### Candidate
type: A, layer: 1, pos: 15

### Candidate
type: B, layer: 1, pos: 28

## Relational analysis of NS_B1_A2_A1_B2_B1
Optimization infeasible because this subproblem isn't reachable.

### Candidate
type: B, layer: 1, pos: 6

## Relational analysis of NS_B1_A2_A1_B2_B1

### Relational analysis result of NS_B1_A2_A1_B2_B1
Status: Status.UNKNOWN
Output dim: 0, lower bound: -20.9274296, upper bound: 20.9390731
time: 0.58 seconds

## Relational analysis of NS_B1_A2_A1_B2_B2

### Relational analysis result of NS_B1_A2_A1_B2_B2
Status: Status.UNKNOWN
Output dim: 0, lower bound: -20.9484271, upper bound: 20.9515325
time: 0.60 seconds

## BFS NS instance: NS_B2_B1_A1_A2

### Backsubstitution after applying NS history:
0: -6.1442766, 17.1283550, -6.0461669, 16.9050064, -23.0492802, 23.1745205
1: -15.2596436, 26.4177361, -15.0026617, 26.0916557, -41.3512993, 41.4203987
2: -10.6695375, 23.9184093, -10.4874372, 23.6187649, -34.2883034, 34.4058456
3: -16.5151844, 29.1772575, -16.2466373, 28.7926636, -45.3078461, 45.4238968
4: -15.1578903, 29.4901905, -14.9184227, 29.1219597, -44.2798309, 44.4086113

Time for backsubstitution: 0.72 seconds

### NS candidates at layer 1
type: A, layer: 1, pos: 1
type: A, layer: 1, pos: 47
type: B, layer: 1, pos: 47
type: B, layer: 1, pos: 45
type: B, layer: 1, pos: 15
type: A, layer: 1, pos: 45
type: A, layer: 1, pos: 39
type: B, layer: 1, pos: 39
type: B, layer: 1, pos: 38
type: A, layer: 1, pos: 8
type: A, layer: 1, pos: 38
type: B, layer: 1, pos: 20
type: A, layer: 1, pos: 9
type: B, layer: 1, pos: 8
type: B, layer: 1, pos: 9
type: B, layer: 1, pos: 37
type: A, layer: 1, pos: 37
type: A, layer: 1, pos: 20
type: A, layer: 1, pos: 6
type: A, layer: 1, pos: 28
type: B, layer: 1, pos: 28
type: B, layer: 1, pos: 48
type: B, layer: 1, pos: 6
type: A, layer: 1, pos: 48
type: A, layer: 1, pos: 31
type: B, layer: 1, pos: 31
type: B, layer: 1, pos: 24
type: A, layer: 1, pos: 24
type: B, layer: 1, pos: 0
type: B, layer: 1, pos: 26
type: A, layer: 1, pos: 26
type: B, layer: 1, pos: 41
type: A, layer: 1, pos: 41
type: A, layer: 1, pos: 5
type: B, layer: 1, pos: 5
type: A, layer: 1, pos: 0

Time for candidate selection: 0.06 seconds

### Candidate
type: A, layer: 1, pos: 1

### Candidate
type: A, layer: 1, pos: 47

## Relational analysis of NS_B2_B1_A1_A2_A1

### Relational analysis result of NS_B2_B1_A1_A2_A1
Status: Status.UNKNOWN
Output dim: 0, lower bound: -20.9615563, upper bound: 20.9629050
time: 0.73 seconds

## Relational analysis of NS_B2_B1_A1_A2_A2

### Relational analysis result of NS_B2_B1_A1_A2_A2
Status: Status.UNKNOWN
Output dim: 0, lower bound: -20.9614772, upper bound: 20.9614772
time: 0.73 seconds

## BFS NS instance: NS_B2_B1_A2_B2

### Backsubstitution after applying NS history:
0: -6.2726994, 17.3570690, -5.9021125, 16.5162315, -22.7889271, 23.2591820
1: -15.5273714, 26.7728806, -14.6226358, 25.5178680, -41.0452385, 41.3955154
2: -10.8936138, 24.1979637, -10.2336988, 23.0905800, -33.9841919, 34.4316635
3: -16.7668762, 29.5876045, -15.8537064, 28.1620789, -44.9289436, 45.4413109
4: -15.3402452, 29.7392197, -14.5781689, 28.4602432, -43.8004875, 44.3173828

Time for backsubstitution: 0.73 seconds

### NS candidates at layer 1
type: A, layer: 1, pos: 1
type: B, layer: 1, pos: 45
type: A, layer: 1, pos: 45
type: A, layer: 1, pos: 47
type: B, layer: 1, pos: 47
type: A, layer: 1, pos: 15
type: A, layer: 1, pos: 39
type: B, layer: 1, pos: 39
type: A, layer: 1, pos: 6
type: B, layer: 1, pos: 38
type: A, layer: 1, pos: 38
type: A, layer: 1, pos: 8
type: A, layer: 1, pos: 9
type: B, layer: 1, pos: 20
type: B, layer: 1, pos: 37
type: B, layer: 1, pos: 9
type: A, layer: 1, pos: 48
type: B, layer: 1, pos: 48
type: A, layer: 1, pos: 28
type: B, layer: 1, pos: 6
type: A, layer: 1, pos: 37
type: B, layer: 1, pos: 8
type: B, layer: 1, pos: 28
type: A, layer: 1, pos: 31
type: A, layer: 1, pos: 20
type: B, layer: 1, pos: 31
type: A, layer: 1, pos: 24
type: B, layer: 1, pos: 24
type: B, layer: 1, pos: 5
type: A, layer: 1, pos: 5
type: B, layer: 1, pos: 0
type: A, layer: 1, pos: 0
type: B, layer: 1, pos: 41
type: A, layer: 1, pos: 41
type: B, layer: 1, pos: 26
type: A, layer: 1, pos: 26

Time for candidate selection: 0.06 seconds

### Candidate
type: A, layer: 1, pos: 1

### Candidate
type: B, layer: 1, pos: 45

## Relational analysis of NS_B2_B1_A2_B2_B1
Optimization infeasible because this subproblem isn't reachable.

### Candidate
type: A, layer: 1, pos: 45

### Candidate
type: A, layer: 1, pos: 47

## Relational analysis of NS_B2_B1_A2_B2_A1

### Relational analysis result of NS_B2_B1_A2_B2_A1
Status: Status.UNKNOWN
Output dim: 0, lower bound: -20.9579364, upper bound: 20.9604336
time: 0.64 seconds

## Relational analysis of NS_B2_B1_A2_B2_A2

### Relational analysis result of NS_B2_B1_A2_B2_A2
Status: Status.UNKNOWN
Output dim: 0, lower bound: -20.9513168, upper bound: 20.9513119
time: 0.76 seconds

## BFS NS instance: NS_B2_B2_A1_B1

### Backsubstitution after applying NS history:
0: -6.1524014, 17.1095142, -5.7815499, 16.1679955, -22.3203964, 22.8910637
1: -15.3654442, 26.2113514, -14.4255676, 24.9067802, -40.2722244, 40.6369171
2: -10.7162285, 23.8181400, -10.0648251, 22.5165768, -33.2328033, 33.8829613
3: -16.5595512, 29.0159378, -15.4787817, 27.5215988, -44.0811501, 44.4947166
4: -15.0991631, 29.4114056, -13.9816208, 27.7338066, -42.8329697, 43.3930244

Time for backsubstitution: 0.73 seconds

### NS candidates at layer 1
type: A, layer: 1, pos: 12
type: A, layer: 1, pos: 45
type: A, layer: 1, pos: 28
type: B, layer: 1, pos: 45
type: A, layer: 1, pos: 15
type: A, layer: 1, pos: 47
type: B, layer: 1, pos: 28
type: A, layer: 1, pos: 48
type: B, layer: 1, pos: 15
type: A, layer: 1, pos: 6
type: B, layer: 1, pos: 6
type: A, layer: 1, pos: 24
type: B, layer: 1, pos: 8
type: B, layer: 1, pos: 24
type: B, layer: 1, pos: 9
type: A, layer: 1, pos: 9
type: A, layer: 1, pos: 39
type: A, layer: 1, pos: 38
type: B, layer: 1, pos: 37
type: A, layer: 1, pos: 37
type: B, layer: 1, pos: 39
type: A, layer: 1, pos: 20
type: B, layer: 1, pos: 38
type: A, layer: 1, pos: 8
type: A, layer: 1, pos: 26
type: B, layer: 1, pos: 26
type: B, layer: 1, pos: 20
type: B, layer: 1, pos: 48
type: A, layer: 1, pos: 0
type: B, layer: 1, pos: 5
type: B, layer: 1, pos: 31
type: A, layer: 1, pos: 41
type: B, layer: 1, pos: 41
type: A, layer: 1, pos: 31
type: B, layer: 1, pos: 0
type: A, layer: 1, pos: 5

Time for candidate selection: 0.06 seconds

### Candidate
type: A, layer: 1, pos: 12

### Candidate
type: A, layer: 1, pos: 45

## Relational analysis of NS_B2_B2_A1_B1_A1
Optimization infeasible because this subproblem isn't reachable.

### Candidate
type: A, layer: 1, pos: 28

## Relational analysis of NS_B2_B2_A1_B1_A1
Optimization infeasible because this subproblem isn't reachable.

### Candidate
type: B, layer: 1, pos: 45

### Candidate
type: A, layer: 1, pos: 15

## Relational analysis of NS_B2_B2_A1_B1_A1
Optimization infeasible because this subproblem isn't reachable.

### Candidate
type: A, layer: 1, pos: 47

## Relational analysis of NS_B2_B2_A1_B1_A1

### Relational analysis result of NS_B2_B2_A1_B1_A1
Status: Status.UNKNOWN
Output dim: 0, lower bound: -20.9361618, upper bound: 20.9403090
time: 0.71 seconds

## Relational analysis of NS_B2_B2_A1_B1_A2

### Relational analysis result of NS_B2_B2_A1_B1_A2
Status: Status.UNKNOWN
Output dim: 0, lower bound: -20.9408792, upper bound: 20.9403090
time: 0.60 seconds

## BFS NS instance: NS_B2_B2_A1_B2

### Backsubstitution after applying NS history:
0: -6.3032446, 17.4915447, -5.9625483, 16.5776463, -22.8808880, 23.4540844
1: -15.7344370, 26.7710838, -14.7118168, 25.6270142, -41.3614464, 41.4828987
2: -10.9783573, 24.3316841, -10.3330498, 23.1492825, -34.1276398, 34.6647339
3: -16.9640579, 29.6479645, -15.9134026, 28.2967262, -45.2607841, 45.5613670
4: -15.4797134, 30.0414028, -14.5857401, 28.4363346, -43.9160385, 44.6271439

Time for backsubstitution: 0.74 seconds

### NS candidates at layer 1
type: A, layer: 1, pos: 12
type: A, layer: 1, pos: 45
type: A, layer: 1, pos: 47
type: B, layer: 1, pos: 45
type: A, layer: 1, pos: 15
type: B, layer: 1, pos: 15
type: A, layer: 1, pos: 28
type: A, layer: 1, pos: 6
type: B, layer: 1, pos: 6
type: A, layer: 1, pos: 24
type: B, layer: 1, pos: 28
type: B, layer: 1, pos: 9
type: B, layer: 1, pos: 8
type: A, layer: 1, pos: 48
type: B, layer: 1, pos: 24
type: A, layer: 1, pos: 9
type: B, layer: 1, pos: 39
type: A, layer: 1, pos: 38
type: A, layer: 1, pos: 37
type: B, layer: 1, pos: 37
type: A, layer: 1, pos: 20
type: B, layer: 1, pos: 38
type: A, layer: 1, pos: 8
type: A, layer: 1, pos: 39
type: B, layer: 1, pos: 48
type: B, layer: 1, pos: 26
type: A, layer: 1, pos: 26
type: B, layer: 1, pos: 20
type: A, layer: 1, pos: 0
type: B, layer: 1, pos: 31
type: A, layer: 1, pos: 41
type: B, layer: 1, pos: 41
type: A, layer: 1, pos: 5
type: B, layer: 1, pos: 5
type: A, layer: 1, pos: 31
type: B, layer: 1, pos: 0

Time for candidate selection: 0.06 seconds

### Candidate
type: A, layer: 1, pos: 12

### Candidate
type: A, layer: 1, pos: 45

## Relational analysis of NS_B2_B2_A1_B2_A1
Optimization infeasible because this subproblem isn't reachable.

### Candidate
type: A, layer: 1, pos: 47

## Relational analysis of NS_B2_B2_A1_B2_A1

### Relational analysis result of NS_B2_B2_A1_B2_A1
Status: Status.UNKNOWN
Output dim: 0, lower bound: -20.9361618, upper bound: 20.9403090
time: 0.74 seconds

## Relational analysis of NS_B2_B2_A1_B2_A2

### Relational analysis result of NS_B2_B2_A1_B2_A2
Status: Status.UNKNOWN
Output dim: 0, lower bound: -20.9361618, upper bound: 20.9403090
time: 0.68 seconds

## BFS NS instance: NS_B2_B2_A2_B1

### Backsubstitution after applying NS history:
0: -6.0165682, 16.8427544, -5.7815499, 16.1679955, -22.1845627, 22.6243019
1: -14.9917011, 25.9771519, -14.4255676, 24.9067802, -39.8984718, 40.4027176
2: -10.4577951, 23.5191689, -10.0648251, 22.5165768, -32.9743729, 33.5839920
3: -16.1758804, 28.6721153, -15.4787817, 27.5215988, -43.6974754, 44.1508904
4: -14.7741776, 29.0216789, -13.9816208, 27.7338066, -42.5079803, 43.0032959

Time for backsubstitution: 0.73 seconds

### NS candidates at layer 1
type: A, layer: 1, pos: 12
type: A, layer: 1, pos: 45
type: B, layer: 1, pos: 28
type: B, layer: 1, pos: 45
type: A, layer: 1, pos: 47
type: A, layer: 1, pos: 15
type: A, layer: 1, pos: 28
type: B, layer: 1, pos: 15
type: B, layer: 1, pos: 6
type: A, layer: 1, pos: 48
type: A, layer: 1, pos: 6
type: A, layer: 1, pos: 24
type: B, layer: 1, pos: 24
type: B, layer: 1, pos: 8
type: B, layer: 1, pos: 9
type: A, layer: 1, pos: 39
type: A, layer: 1, pos: 9
type: A, layer: 1, pos: 38
type: A, layer: 1, pos: 37
type: B, layer: 1, pos: 37
type: A, layer: 1, pos: 20
type: B, layer: 1, pos: 38
type: A, layer: 1, pos: 26
type: B, layer: 1, pos: 26
type: B, layer: 1, pos: 39
type: A, layer: 1, pos: 8
type: B, layer: 1, pos: 48
type: B, layer: 1, pos: 20
type: B, layer: 1, pos: 31
type: B, layer: 1, pos: 5
type: A, layer: 1, pos: 31
type: A, layer: 1, pos: 0
type: A, layer: 1, pos: 41
type: B, layer: 1, pos: 41
type: B, layer: 1, pos: 0
type: A, layer: 1, pos: 5

Time for candidate selection: 0.06 seconds

### Candidate
type: A, layer: 1, pos: 12

## Relational analysis of NS_B2_B2_A2_B1_A1

### Relational analysis result of NS_B2_B2_A2_B1_A1
Status: Status.UNKNOWN
Output dim: 0, lower bound: -20.9569857, upper bound: 20.9544784
time: 0.63 seconds

## Relational analysis of NS_B2_B2_A2_B1_A2

### Relational analysis result of NS_B2_B2_A2_B1_A2
Status: Status.UNKNOWN
Output dim: 0, lower bound: -20.9569857, upper bound: 20.9544784
time: 0.66 seconds

## BFS NS instance: NS_B2_B2_A2_B2

### Backsubstitution after applying NS history:
0: -6.1677237, 17.2251701, -5.9625483, 16.5776463, -22.7453651, 23.1877155
1: -15.3650265, 26.5364742, -14.7118168, 25.6270142, -40.9920349, 41.2482910
2: -10.7221899, 24.0357971, -10.3330498, 23.1492825, -33.8714638, 34.3688469
3: -16.5831528, 29.3041401, -15.9134026, 28.2967262, -44.8798752, 45.2175407
4: -15.1602678, 29.6561813, -14.5857401, 28.4363346, -43.5965958, 44.2419167

Time for backsubstitution: 0.74 seconds

### NS candidates at layer 1
type: A, layer: 1, pos: 45
type: A, layer: 1, pos: 12
type: B, layer: 1, pos: 45
type: A, layer: 1, pos: 15
type: A, layer: 1, pos: 47
type: B, layer: 1, pos: 15
type: A, layer: 1, pos: 28
type: B, layer: 1, pos: 28
type: B, layer: 1, pos: 8
type: A, layer: 1, pos: 24
type: B, layer: 1, pos: 6
type: B, layer: 1, pos: 9
type: A, layer: 1, pos: 48
type: B, layer: 1, pos: 24
type: A, layer: 1, pos: 37
type: A, layer: 1, pos: 9
type: A, layer: 1, pos: 6
type: B, layer: 1, pos: 37
type: B, layer: 1, pos: 39
type: B, layer: 1, pos: 48
type: A, layer: 1, pos: 38
type: A, layer: 1, pos: 39
type: B, layer: 1, pos: 38
type: A, layer: 1, pos: 8
type: A, layer: 1, pos: 20
type: B, layer: 1, pos: 26
type: A, layer: 1, pos: 26
type: B, layer: 1, pos: 31
type: B, layer: 1, pos: 20
type: A, layer: 1, pos: 31
type: B, layer: 1, pos: 41
type: A, layer: 1, pos: 41
type: A, layer: 1, pos: 0
type: A, layer: 1, pos: 5
type: B, layer: 1, pos: 5
type: B, layer: 1, pos: 0

Time for candidate selection: 0.06 seconds

### Candidate
type: A, layer: 1, pos: 45

## Relational analysis of NS_B2_B2_A2_B2_A1

### Relational analysis result of NS_B2_B2_A2_B2_A1
Status: Status.VERIFIED
Output dim: 0, lower bound: -20.6663848, upper bound: 20.6960602
time: 0.65 seconds

## Relational analysis of NS_B2_B2_A2_B2_A2

### Relational analysis result of NS_B2_B2_A2_B2_A2
Status: Status.VERIFIED
Output dim: 0, lower bound: -20.6065368, upper bound: 20.6065369
time: 0.62 seconds

## Summary of splitting at layer (split count: 4)
- Time for NS candidates: 2.08 seconds
NS_B1_A1_A1_A1_B1, status: Status.UNKNOWN, split count: 5, time: 2.08
Output dim: 0, lower bound: -20.9108170, upper bound: 20.9172336
NS_B1_A1_A1_A1_B2, status: Status.UNKNOWN, split count: 5, time: 2.08
Output dim: 0, lower bound: -20.9108170, upper bound: 20.9172336
NS_B1_A1_A1_A2_B1, status: Status.UNKNOWN, split count: 5, time: 2.08
Output dim: 0, lower bound: -20.9108170, upper bound: 20.9172336
NS_B1_A1_A1_A2_B2, status: Status.UNKNOWN, split count: 5, time: 2.08
Output dim: 0, lower bound: -20.9108170, upper bound: 20.9172336
NS_B1_A2_A1_B1_B1, status: Status.UNKNOWN, split count: 5, time: 2.08
Output dim: 0, lower bound: -20.9274296, upper bound: 20.9390731
NS_B1_A2_A1_B1_B2, status: Status.UNKNOWN, split count: 5, time: 2.08
Output dim: 0, lower bound: -20.9484271, upper bound: 20.9515325
NS_B1_A2_A1_B2_B1, status: Status.UNKNOWN, split count: 5, time: 2.08
Output dim: 0, lower bound: -20.9274296, upper bound: 20.9390731
NS_B1_A2_A1_B2_B2, status: Status.UNKNOWN, split count: 5, time: 2.08
Output dim: 0, lower bound: -20.9484271, upper bound: 20.9515325
NS_B2_B1_A1_A2_A1, status: Status.UNKNOWN, split count: 5, time: 2.08
Output dim: 0, lower bound: -20.9615563, upper bound: 20.9629050
NS_B2_B1_A1_A2_A2, status: Status.UNKNOWN, split count: 5, time: 2.08
Output dim: 0, lower bound: -20.9614772, upper bound: 20.9614772
NS_B2_B1_A2_B2_A1, status: Status.UNKNOWN, split count: 5, time: 2.08
Output dim: 0, lower bound: -20.9579364, upper bound: 20.9604336
NS_B2_B1_A2_B2_A2, status: Status.UNKNOWN, split count: 5, time: 2.08
Output dim: 0, lower bound: -20.9513168, upper bound: 20.9513119
NS_B2_B2_A1_B1_A1, status: Status.UNKNOWN, split count: 5, time: 2.08
Output dim: 0, lower bound: -20.9361618, upper bound: 20.9403090
NS_B2_B2_A1_B1_A2, status: Status.UNKNOWN, split count: 5, time: 2.08
Output dim: 0, lower bound: -20.9408792, upper bound: 20.9403090
NS_B2_B2_A1_B2_A1, status: Status.UNKNOWN, split count: 5, time: 2.08
Output dim: 0, lower bound: -20.9361618, upper bound: 20.9403090
NS_B2_B2_A1_B2_A2, status: Status.UNKNOWN, split count: 5, time: 2.08
Output dim: 0, lower bound: -20.9361618, upper bound: 20.9403090
NS_B2_B2_A2_B1_A1, status: Status.UNKNOWN, split count: 5, time: 2.08
Output dim: 0, lower bound: -20.9569857, upper bound: 20.9544784
NS_B2_B2_A2_B1_A2, status: Status.UNKNOWN, split count: 5, time: 2.08
Output dim: 0, lower bound: -20.9569857, upper bound: 20.9544784
NS_B2_B2_A2_B2_A1, status: Status.VERIFIED, split count: 5, time: 2.08
Output dim: 0, lower bound: -20.6663848, upper bound: 20.6960602
NS_B2_B2_A2_B2_A2, status: Status.VERIFIED, split count: 5, time: 2.08
Output dim: 0, lower bound: -20.6065368, upper bound: 20.6065369

## BFS NS instance: NS_B1_A1_A1_A1_B1

### Backsubstitution after applying NS history:
0: -6.0846200, 16.9946613, -6.2165189, 17.3560257, -23.4406452, 23.2111778
1: -15.3345718, 25.8991718, -15.6833458, 26.4522324, -41.7868004, 41.5825157
2: -10.6436501, 23.5850830, -10.8782454, 24.0774555, -34.7211037, 34.4633255
3: -16.4320755, 28.7094727, -16.7920380, 29.3136368, -45.7457123, 45.5015068
4: -14.8261871, 29.1902122, -15.1396618, 29.8034248, -44.6296120, 44.3298645

Time for backsubstitution: 0.74 seconds

### NS candidates at layer 1
type: A, layer: 1, pos: 12
type: B, layer: 1, pos: 12
type: B, layer: 1, pos: 15
type: A, layer: 1, pos: 15
type: A, layer: 1, pos: 28
type: B, layer: 1, pos: 28
type: B, layer: 1, pos: 45
type: A, layer: 1, pos: 6
type: B, layer: 1, pos: 6
type: A, layer: 1, pos: 24
type: B, layer: 1, pos: 24
type: A, layer: 1, pos: 9
type: B, layer: 1, pos: 9
type: B, layer: 1, pos: 37
type: A, layer: 1, pos: 37
type: A, layer: 1, pos: 8
type: B, layer: 1, pos: 8
type: A, layer: 1, pos: 39
type: B, layer: 1, pos: 38
type: A, layer: 1, pos: 38
type: B, layer: 1, pos: 39
type: B, layer: 1, pos: 48
type: A, layer: 1, pos: 48
type: B, layer: 1, pos: 20
type: A, layer: 1, pos: 20
type: A, layer: 1, pos: 26
type: B, layer: 1, pos: 26
type: A, layer: 1, pos: 41
type: B, layer: 1, pos: 41
type: B, layer: 1, pos: 0
type: A, layer: 1, pos: 5
type: A, layer: 1, pos: 0
type: B, layer: 1, pos: 5
type: B, layer: 1, pos: 31
type: A, layer: 1, pos: 31

Time for candidate selection: 0.06 seconds

### Candidate
type: A, layer: 1, pos: 12

### Candidate
type: B, layer: 1, pos: 12

### Candidate
type: B, layer: 1, pos: 15

### Candidate
type: A, layer: 1, pos: 15

### Candidate
type: A, layer: 1, pos: 28

## Relational analysis of NS_B1_A1_A1_A1_B1_A1
Optimization infeasible because this subproblem isn't reachable.

### Candidate
type: B, layer: 1, pos: 28

### Candidate
type: B, layer: 1, pos: 45

## Relational analysis of NS_B1_A1_A1_A1_B1_B1

### Relational analysis result of NS_B1_A1_A1_A1_B1_B1
Status: Status.UNKNOWN
Output dim: 0, lower bound: -20.9235252, upper bound: 20.9362552
time: 0.58 seconds

## Relational analysis of NS_B1_A1_A1_A1_B1_B2

### Relational analysis result of NS_B1_A1_A1_A1_B1_B2
Status: Status.UNKNOWN
Output dim: 0, lower bound: -20.9235252, upper bound: 20.9362552
time: 0.53 seconds

## BFS NS instance: NS_B1_A1_A1_A1_B2

### Backsubstitution after applying NS history:
0: -6.0846200, 16.9946613, -6.3588510, 17.6330452, -23.7176647, 23.3535118
1: -15.3345718, 25.8991718, -15.8715429, 26.9774265, -42.3119965, 41.7707138
2: -10.6436501, 23.5850830, -11.0782328, 24.5221272, -35.1657715, 34.6633148
3: -16.4320755, 28.7094727, -17.1150131, 29.8800278, -46.3120995, 45.8244820
4: -14.8261871, 29.1902122, -15.6188936, 30.2740707, -45.1002579, 44.8091011

Time for backsubstitution: 0.74 seconds

### NS candidates at layer 1
type: B, layer: 1, pos: 12
type: A, layer: 1, pos: 12
type: B, layer: 1, pos: 15
type: A, layer: 1, pos: 15
type: A, layer: 1, pos: 28
type: B, layer: 1, pos: 28
type: B, layer: 1, pos: 6
type: B, layer: 1, pos: 45
type: A, layer: 1, pos: 24
type: A, layer: 1, pos: 6
type: B, layer: 1, pos: 39
type: B, layer: 1, pos: 9
type: B, layer: 1, pos: 24
type: B, layer: 1, pos: 48
type: A, layer: 1, pos: 37
type: A, layer: 1, pos: 9
type: B, layer: 1, pos: 37
type: B, layer: 1, pos: 8
type: A, layer: 1, pos: 8
type: A, layer: 1, pos: 38
type: B, layer: 1, pos: 38
type: A, layer: 1, pos: 20
type: A, layer: 1, pos: 39
type: A, layer: 1, pos: 48
type: B, layer: 1, pos: 26
type: A, layer: 1, pos: 26
type: B, layer: 1, pos: 20
type: A, layer: 1, pos: 5
type: B, layer: 1, pos: 41
type: A, layer: 1, pos: 41
type: A, layer: 1, pos: 0
type: B, layer: 1, pos: 31
type: B, layer: 1, pos: 0
type: B, layer: 1, pos: 5
type: A, layer: 1, pos: 31

Time for candidate selection: 0.06 seconds

### Candidate
type: B, layer: 1, pos: 12

### Candidate
type: A, layer: 1, pos: 12

### Candidate
type: B, layer: 1, pos: 15

### Candidate
type: A, layer: 1, pos: 15

### Candidate
type: A, layer: 1, pos: 28

## Relational analysis of NS_B1_A1_A1_A1_B2_A1
Optimization infeasible because this subproblem isn't reachable.

### Candidate
type: B, layer: 1, pos: 28

### Candidate
type: B, layer: 1, pos: 6

## Relational analysis of NS_B1_A1_A1_A1_B2_B1

### Relational analysis result of NS_B1_A1_A1_A1_B2_B1
Status: Status.UNKNOWN
Output dim: 0, lower bound: -20.9226500, upper bound: 20.9355149
time: 0.54 seconds

## Relational analysis of NS_B1_A1_A1_A1_B2_B2
Optimization infeasible because this subproblem isn't reachable.

### Candidate
type: B, layer: 1, pos: 45

## Relational analysis of NS_B1_A1_A1_A1_B2_B1

### Relational analysis result of NS_B1_A1_A1_A1_B2_B1
Status: Status.UNKNOWN
Output dim: 0, lower bound: -20.9235252, upper bound: 20.9362552
time: 0.54 seconds

## Relational analysis of NS_B1_A1_A1_A1_B2_B2

### Relational analysis result of NS_B1_A1_A1_A1_B2_B2
Status: Status.UNKNOWN
Output dim: 0, lower bound: -20.9235252, upper bound: 20.9362552
time: 1.23 seconds

## BFS NS instance: NS_B1_A1_A1_A2_B1

### Backsubstitution after applying NS history:
0: -6.1871290, 17.1735039, -6.2165189, 17.3560257, -23.5431519, 23.3900223
1: -15.4273548, 26.2602158, -15.6833458, 26.4522324, -41.8795815, 41.9435539
2: -10.7745504, 23.8967419, -10.8782454, 24.0774555, -34.8520050, 34.7749825
3: -16.6525230, 29.0933285, -16.7920380, 29.3136368, -45.9661598, 45.8853645
4: -15.2049751, 29.4977074, -15.1396618, 29.8034248, -45.0084000, 44.6373634

Time for backsubstitution: 0.74 seconds

### NS candidates at layer 1
type: A, layer: 1, pos: 12
type: B, layer: 1, pos: 12
type: A, layer: 1, pos: 15
type: B, layer: 1, pos: 15
type: B, layer: 1, pos: 28
type: A, layer: 1, pos: 6
type: A, layer: 1, pos: 28
type: B, layer: 1, pos: 45
type: B, layer: 1, pos: 24
type: A, layer: 1, pos: 39
type: B, layer: 1, pos: 6
type: A, layer: 1, pos: 9
type: A, layer: 1, pos: 24
type: B, layer: 1, pos: 37
type: B, layer: 1, pos: 9
type: A, layer: 1, pos: 48
type: A, layer: 1, pos: 37
type: A, layer: 1, pos: 8
type: B, layer: 1, pos: 8
type: B, layer: 1, pos: 38
type: A, layer: 1, pos: 38
type: B, layer: 1, pos: 20
type: B, layer: 1, pos: 48
type: A, layer: 1, pos: 26
type: B, layer: 1, pos: 39
type: B, layer: 1, pos: 26
type: A, layer: 1, pos: 20
type: B, layer: 1, pos: 0
type: A, layer: 1, pos: 41
type: B, layer: 1, pos: 41
type: B, layer: 1, pos: 5
type: A, layer: 1, pos: 31
type: A, layer: 1, pos: 0
type: A, layer: 1, pos: 5
type: B, layer: 1, pos: 31

Time for candidate selection: 0.06 seconds

### Candidate
type: A, layer: 1, pos: 12

### Candidate
type: B, layer: 1, pos: 12

### Candidate
type: A, layer: 1, pos: 15

## Relational analysis of NS_B1_A1_A1_A2_B1_A1
Optimization infeasible because this subproblem isn't reachable.

### Candidate
type: B, layer: 1, pos: 15

### Candidate
type: B, layer: 1, pos: 28

## Relational analysis of NS_B1_A1_A1_A2_B1_B1
Optimization infeasible because this subproblem isn't reachable.

### Candidate
type: A, layer: 1, pos: 6

## Relational analysis of NS_B1_A1_A1_A2_B1_A1
Optimization infeasible because this subproblem isn't reachable.

### Candidate
type: A, layer: 1, pos: 28

### Candidate
type: B, layer: 1, pos: 45

## Relational analysis of NS_B1_A1_A1_A2_B1_B1

### Relational analysis result of NS_B1_A1_A1_A2_B1_B1
Status: Status.UNKNOWN
Output dim: 0, lower bound: -20.9108170, upper bound: 20.9172336
time: 0.57 seconds

## Relational analysis of NS_B1_A1_A1_A2_B1_B2

### Relational analysis result of NS_B1_A1_A1_A2_B1_B2
Status: Status.UNKNOWN
Output dim: 0, lower bound: -20.9108170, upper bound: 20.9172336
time: 0.57 seconds

## BFS NS instance: NS_B1_A1_A1_A2_B2

### Backsubstitution after applying NS history:
0: -6.1871290, 17.1735039, -6.3588510, 17.6330452, -23.8201733, 23.5323544
1: -15.4273548, 26.2602158, -15.8715429, 26.9774265, -42.4047813, 42.1317558
2: -10.7745504, 23.8967419, -11.0782328, 24.5221272, -35.2966766, 34.9749756
3: -16.6525230, 29.0933285, -17.1150131, 29.8800278, -46.5325432, 46.2083359
4: -15.2049751, 29.4977074, -15.6188936, 30.2740707, -45.4790459, 45.1166000

Time for backsubstitution: 0.74 seconds

### NS candidates at layer 1
type: A, layer: 1, pos: 12
type: B, layer: 1, pos: 12
type: B, layer: 1, pos: 15
type: A, layer: 1, pos: 15
type: B, layer: 1, pos: 45
type: A, layer: 1, pos: 28
type: B, layer: 1, pos: 28
type: A, layer: 1, pos: 6
type: B, layer: 1, pos: 6
type: A, layer: 1, pos: 24
type: B, layer: 1, pos: 24
type: A, layer: 1, pos: 9
type: B, layer: 1, pos: 9
type: A, layer: 1, pos: 37
type: B, layer: 1, pos: 37
type: A, layer: 1, pos: 8
type: B, layer: 1, pos: 8
type: B, layer: 1, pos: 38
type: A, layer: 1, pos: 38
type: B, layer: 1, pos: 48
type: A, layer: 1, pos: 39
type: A, layer: 1, pos: 48
type: B, layer: 1, pos: 39
type: B, layer: 1, pos: 20
type: A, layer: 1, pos: 20
type: B, layer: 1, pos: 26
type: A, layer: 1, pos: 26
type: A, layer: 1, pos: 41
type: B, layer: 1, pos: 41
type: A, layer: 1, pos: 5
type: B, layer: 1, pos: 5
type: A, layer: 1, pos: 0
type: B, layer: 1, pos: 0
type: A, layer: 1, pos: 31
type: B, layer: 1, pos: 31

Time for candidate selection: 0.06 seconds

### Candidate
type: A, layer: 1, pos: 12

### Candidate
type: B, layer: 1, pos: 12

### Candidate
type: B, layer: 1, pos: 15

## Relational analysis of NS_B1_A1_A1_A2_B2_B1
Optimization infeasible because this subproblem isn't reachable.

### Candidate
type: A, layer: 1, pos: 15

### Candidate
type: B, layer: 1, pos: 45

## Relational analysis of NS_B1_A1_A1_A2_B2_B1

### Relational analysis result of NS_B1_A1_A1_A2_B2_B1
Status: Status.UNKNOWN
Output dim: 0, lower bound: -20.9108170, upper bound: 20.9172336
time: 0.68 seconds

## Relational analysis of NS_B1_A1_A1_A2_B2_B2

### Relational analysis result of NS_B1_A1_A1_A2_B2_B2
Status: Status.UNKNOWN
Output dim: 0, lower bound: -20.9108170, upper bound: 20.9172336
time: 0.56 seconds

## BFS NS instance: NS_B1_A2_A1_B1_B1

### Backsubstitution after applying NS history:
0: -5.6829767, 16.0174122, -5.4689183, 15.4409122, -21.1238899, 21.4863300
1: -14.2276907, 24.6698856, -13.7117033, 23.7303829, -37.9580727, 38.3815880
2: -9.8961363, 22.3435078, -9.5278883, 21.5665436, -31.4626808, 31.8713951
3: -15.3129988, 27.2465572, -14.7623062, 26.2340851, -41.5470734, 42.0088654
4: -13.8679800, 27.6100693, -13.3521547, 26.6765404, -40.5445213, 40.9622116

Time for backsubstitution: 0.74 seconds

### NS candidates at layer 1
type: A, layer: 1, pos: 12
type: B, layer: 1, pos: 12
type: B, layer: 1, pos: 45
type: A, layer: 1, pos: 45
type: A, layer: 1, pos: 28
type: B, layer: 1, pos: 15
type: A, layer: 1, pos: 15
type: A, layer: 1, pos: 24
type: B, layer: 1, pos: 28
type: A, layer: 1, pos: 9
type: B, layer: 1, pos: 8
type: B, layer: 1, pos: 24
type: A, layer: 1, pos: 39
type: B, layer: 1, pos: 9
type: A, layer: 1, pos: 6
type: A, layer: 1, pos: 8
type: B, layer: 1, pos: 20
type: B, layer: 1, pos: 48
type: A, layer: 1, pos: 37
type: B, layer: 1, pos: 37
type: B, layer: 1, pos: 39
type: B, layer: 1, pos: 38
type: A, layer: 1, pos: 48
type: B, layer: 1, pos: 26
type: A, layer: 1, pos: 38
type: A, layer: 1, pos: 26
type: A, layer: 1, pos: 20
type: B, layer: 1, pos: 0
type: A, layer: 1, pos: 0
type: B, layer: 1, pos: 41
type: A, layer: 1, pos: 41
type: A, layer: 1, pos: 5
type: A, layer: 1, pos: 31
type: B, layer: 1, pos: 31
type: B, layer: 1, pos: 5

Time for candidate selection: 0.06 seconds

### Candidate
type: A, layer: 1, pos: 12

### Candidate
type: B, layer: 1, pos: 12

### Candidate
type: B, layer: 1, pos: 45

### Candidate
type: A, layer: 1, pos: 45

### Candidate
type: A, layer: 1, pos: 28

### Candidate
type: B, layer: 1, pos: 15

### Candidate
type: A, layer: 1, pos: 15

### Candidate
type: A, layer: 1, pos: 24

## Relational analysis of NS_B1_A2_A1_B1_B1_A1
Optimization infeasible because this subproblem isn't reachable.

### Candidate
type: B, layer: 1, pos: 28

### Candidate
type: A, layer: 1, pos: 9

## Relational analysis of NS_B1_A2_A1_B1_B1_A1

### Relational analysis result of NS_B1_A2_A1_B1_B1_A1
Status: Status.UNKNOWN
Output dim: 0, lower bound: -20.9425656, upper bound: 20.9303604
time: 0.57 seconds

## Relational analysis of NS_B1_A2_A1_B1_B1_A2

### Relational analysis result of NS_B1_A2_A1_B1_B1_A2
Status: Status.UNKNOWN
Output dim: 0, lower bound: -20.9601795, upper bound: 20.9570587
time: 0.57 seconds

## BFS NS instance: NS_B1_A2_A1_B1_B2

### Backsubstitution after applying NS history:
0: -6.0017405, 16.8577003, -6.1569595, 17.2093010, -23.2110405, 23.0146599
1: -15.0798769, 25.8949661, -15.5148249, 26.2459545, -41.3258324, 41.4097900
2: -10.4755583, 23.4678841, -10.7693729, 23.8826771, -34.3582306, 34.2372589
3: -16.1910629, 28.6209011, -16.6235962, 29.0747719, -45.2658348, 45.2444992
4: -14.6383791, 29.0156803, -15.0020514, 29.5595055, -44.1978836, 44.0177307

Time for backsubstitution: 0.74 seconds

### NS candidates at layer 1
type: A, layer: 1, pos: 12
type: B, layer: 1, pos: 12
type: B, layer: 1, pos: 45
type: B, layer: 1, pos: 15
type: A, layer: 1, pos: 15
type: B, layer: 1, pos: 28
type: A, layer: 1, pos: 45
type: A, layer: 1, pos: 28
type: A, layer: 1, pos: 24
type: A, layer: 1, pos: 9
type: B, layer: 1, pos: 24
type: B, layer: 1, pos: 9
type: B, layer: 1, pos: 37
type: A, layer: 1, pos: 37
type: A, layer: 1, pos: 6
type: B, layer: 1, pos: 48
type: A, layer: 1, pos: 39
type: B, layer: 1, pos: 38
type: B, layer: 1, pos: 8
type: A, layer: 1, pos: 8
type: A, layer: 1, pos: 38
type: A, layer: 1, pos: 48
type: B, layer: 1, pos: 39
type: B, layer: 1, pos: 26
type: A, layer: 1, pos: 26
type: B, layer: 1, pos: 20
type: A, layer: 1, pos: 20
type: A, layer: 1, pos: 41
type: B, layer: 1, pos: 41
type: B, layer: 1, pos: 0
type: A, layer: 1, pos: 5
type: B, layer: 1, pos: 5
type: A, layer: 1, pos: 0
type: A, layer: 1, pos: 31
type: B, layer: 1, pos: 31

Time for candidate selection: 0.06 seconds

### Candidate
type: A, layer: 1, pos: 12

### Candidate
type: B, layer: 1, pos: 12

### Candidate
type: B, layer: 1, pos: 45

### Candidate
type: B, layer: 1, pos: 15

### Candidate
type: A, layer: 1, pos: 15

### Candidate
type: B, layer: 1, pos: 28

### Candidate
type: A, layer: 1, pos: 45

### Candidate
type: A, layer: 1, pos: 28

### Candidate
type: A, layer: 1, pos: 24

## Relational analysis of NS_B1_A2_A1_B1_B2_A1
Optimization infeasible because this subproblem isn't reachable.

### Candidate
type: A, layer: 1, pos: 9

## Relational analysis of NS_B1_A2_A1_B1_B2_A1

### Relational analysis result of NS_B1_A2_A1_B1_B2_A1
Status: Status.UNKNOWN
Output dim: 0, lower bound: -20.9501995, upper bound: 20.9441063
time: 0.60 seconds

## Relational analysis of NS_B1_A2_A1_B1_B2_A2

### Relational analysis result of NS_B1_A2_A1_B1_B2_A2
Status: Status.UNKNOWN
Output dim: 0, lower bound: -20.9601795, upper bound: 20.9633596
time: 0.64 seconds

## BFS NS instance: NS_B1_A2_A1_B2_B1

### Backsubstitution after applying NS history:
0: -5.6829767, 16.0174122, -5.5659547, 15.6098299, -21.2928066, 21.5833645
1: -14.2276907, 24.6698856, -13.8243294, 24.1012421, -38.3289299, 38.4942169
2: -9.8961363, 22.3435078, -9.6580257, 21.8680553, -31.7641907, 32.0015335
3: -15.3129988, 27.2465572, -14.9626017, 26.6162720, -41.9292641, 42.2091599
4: -13.8679800, 27.6100693, -13.7018490, 26.9766998, -40.8446808, 41.3119125

Time for backsubstitution: 0.75 seconds

### NS candidates at layer 1
type: A, layer: 1, pos: 12
type: B, layer: 1, pos: 12
type: B, layer: 1, pos: 45
type: B, layer: 1, pos: 15
type: A, layer: 1, pos: 15
type: A, layer: 1, pos: 45
type: A, layer: 1, pos: 9
type: B, layer: 1, pos: 9
type: B, layer: 1, pos: 8
type: B, layer: 1, pos: 48
type: A, layer: 1, pos: 24
type: A, layer: 1, pos: 20
type: B, layer: 1, pos: 39
type: B, layer: 1, pos: 20
type: A, layer: 1, pos: 6
type: A, layer: 1, pos: 8
type: B, layer: 1, pos: 38
type: A, layer: 1, pos: 37
type: B, layer: 1, pos: 24
type: A, layer: 1, pos: 28
type: B, layer: 1, pos: 37
type: A, layer: 1, pos: 38
type: B, layer: 1, pos: 28
type: A, layer: 1, pos: 39
type: A, layer: 1, pos: 48
type: A, layer: 1, pos: 0
type: B, layer: 1, pos: 0
type: B, layer: 1, pos: 26
type: A, layer: 1, pos: 5
type: B, layer: 1, pos: 31
type: A, layer: 1, pos: 26
type: A, layer: 1, pos: 41
type: B, layer: 1, pos: 41
type: A, layer: 1, pos: 31
type: B, layer: 1, pos: 5

Time for candidate selection: 0.06 seconds

### Candidate
type: A, layer: 1, pos: 12

### Candidate
type: B, layer: 1, pos: 12

### Candidate
type: B, layer: 1, pos: 45

### Candidate
type: B, layer: 1, pos: 15

### Candidate
type: A, layer: 1, pos: 15

### Candidate
type: A, layer: 1, pos: 45

### Candidate
type: A, layer: 1, pos: 9

## Relational analysis of NS_B1_A2_A1_B2_B1_A1

### Relational analysis result of NS_B1_A2_A1_B2_B1_A1
Status: Status.UNKNOWN
Output dim: 0, lower bound: -20.9023718, upper bound: 20.9081548
time: 0.63 seconds

## Relational analysis of NS_B1_A2_A1_B2_B1_A2

### Relational analysis result of NS_B1_A2_A1_B2_B1_A2
Status: Status.UNKNOWN
Output dim: 0, lower bound: -20.9024147, upper bound: 20.9387501
time: 0.75 seconds

## BFS NS instance: NS_B1_A2_A1_B2_B2

### Backsubstitution after applying NS history:
0: -6.0017405, 16.8577003, -6.2724762, 17.4126396, -23.4143772, 23.1301765
1: -15.0798769, 25.8949661, -15.6263056, 26.6745815, -41.7544594, 41.5212708
2: -10.4755583, 23.4678841, -10.9179850, 24.2343845, -34.7099380, 34.3858681
3: -16.1910629, 28.6209011, -16.8763542, 29.5323906, -45.7234535, 45.4972534
4: -14.6383791, 29.0156803, -15.4266500, 29.9108601, -44.5492401, 44.4423294

Time for backsubstitution: 0.75 seconds

### NS candidates at layer 1
type: B, layer: 1, pos: 12
type: A, layer: 1, pos: 12
type: B, layer: 1, pos: 45
type: B, layer: 1, pos: 15
type: A, layer: 1, pos: 15
type: A, layer: 1, pos: 45
type: B, layer: 1, pos: 28
type: A, layer: 1, pos: 28
type: A, layer: 1, pos: 24
type: B, layer: 1, pos: 48
type: B, layer: 1, pos: 9
type: A, layer: 1, pos: 9
type: B, layer: 1, pos: 24
type: B, layer: 1, pos: 39
type: A, layer: 1, pos: 37
type: B, layer: 1, pos: 37
type: B, layer: 1, pos: 8
type: B, layer: 1, pos: 38
type: A, layer: 1, pos: 38
type: A, layer: 1, pos: 6
type: A, layer: 1, pos: 8
type: A, layer: 1, pos: 39
type: B, layer: 1, pos: 26
type: A, layer: 1, pos: 20
type: A, layer: 1, pos: 48
type: A, layer: 1, pos: 26
type: B, layer: 1, pos: 20
type: A, layer: 1, pos: 5
type: B, layer: 1, pos: 41
type: A, layer: 1, pos: 41
type: A, layer: 1, pos: 0
type: B, layer: 1, pos: 31
type: B, layer: 1, pos: 0
type: A, layer: 1, pos: 31
type: B, layer: 1, pos: 5

Time for candidate selection: 0.06 seconds

### Candidate
type: B, layer: 1, pos: 12

### Candidate
type: A, layer: 1, pos: 12

### Candidate
type: B, layer: 1, pos: 45

### Candidate
type: B, layer: 1, pos: 15

### Candidate
type: A, layer: 1, pos: 15

### Candidate
type: A, layer: 1, pos: 45

### Candidate
type: B, layer: 1, pos: 28

### Candidate
type: A, layer: 1, pos: 28

### Candidate
type: A, layer: 1, pos: 24

## Relational analysis of NS_B1_A2_A1_B2_B2_A1
Optimization infeasible because this subproblem isn't reachable.

### Candidate
type: B, layer: 1, pos: 48

## Relational analysis of NS_B1_A2_A1_B2_B2_B1
Optimization infeasible because this subproblem isn't reachable.

### Candidate
type: B, layer: 1, pos: 9

## Relational analysis of NS_B1_A2_A1_B2_B2_B1

### Relational analysis result of NS_B1_A2_A1_B2_B2_B1
Status: Status.UNKNOWN
Output dim: 0, lower bound: -20.9176398, upper bound: 20.9434239
time: 0.68 seconds

## Relational analysis of NS_B1_A2_A1_B2_B2_B2

### Relational analysis result of NS_B1_A2_A1_B2_B2_B2
Status: Status.UNKNOWN
Output dim: 0, lower bound: -20.9476371, upper bound: 20.9513798
time: 0.58 seconds

## BFS NS instance: NS_B2_B1_A1_A2_A1

### Backsubstitution after applying NS history:
0: -5.7576122, 16.1769314, -5.8933477, 16.5183525, -22.2759628, 22.0702763
1: -14.4193316, 24.9086170, -14.6265459, 25.5251884, -39.9445190, 39.5351601
2: -10.0263004, 22.5539265, -10.2202435, 23.0960236, -33.1223221, 32.7741661
3: -15.5176105, 27.5176315, -15.8352146, 28.1530037, -43.6706161, 43.3528442
4: -14.0637922, 27.8690929, -14.5272236, 28.4796982, -42.5434914, 42.3963013

Time for backsubstitution: 0.77 seconds

### NS candidates at layer 1
type: A, layer: 1, pos: 1
type: B, layer: 1, pos: 45
type: A, layer: 1, pos: 45
type: B, layer: 1, pos: 15
type: B, layer: 1, pos: 39
type: B, layer: 1, pos: 47
type: B, layer: 1, pos: 48
type: B, layer: 1, pos: 38
type: A, layer: 1, pos: 38
type: B, layer: 1, pos: 9
type: A, layer: 1, pos: 28
type: A, layer: 1, pos: 9
type: A, layer: 1, pos: 8
type: B, layer: 1, pos: 6
type: A, layer: 1, pos: 39
type: B, layer: 1, pos: 28
type: A, layer: 1, pos: 6
type: A, layer: 1, pos: 37
type: B, layer: 1, pos: 20
type: B, layer: 1, pos: 37
type: A, layer: 1, pos: 20
type: B, layer: 1, pos: 8
type: B, layer: 1, pos: 31
type: A, layer: 1, pos: 31
type: A, layer: 1, pos: 48
type: A, layer: 1, pos: 24
type: B, layer: 1, pos: 24
type: A, layer: 1, pos: 5
type: A, layer: 1, pos: 0
type: B, layer: 1, pos: 26
type: B, layer: 1, pos: 41
type: A, layer: 1, pos: 26
type: A, layer: 1, pos: 41
type: B, layer: 1, pos: 0
type: B, layer: 1, pos: 5

Time for candidate selection: 0.06 seconds

### Candidate
type: A, layer: 1, pos: 1

### Candidate
type: B, layer: 1, pos: 45

## Relational analysis of NS_B2_B1_A1_A2_A1_B1
Optimization infeasible because this subproblem isn't reachable.

### Candidate
type: A, layer: 1, pos: 45

### Candidate
type: B, layer: 1, pos: 15

## Relational analysis of NS_B2_B1_A1_A2_A1_B1
Optimization infeasible because this subproblem isn't reachable.

### Candidate
type: B, layer: 1, pos: 39

## Relational analysis of NS_B2_B1_A1_A2_A1_B1

### Relational analysis result of NS_B2_B1_A1_A2_A1_B1
Status: Status.UNKNOWN
Output dim: 0, lower bound: -20.9608208, upper bound: 20.9611716
time: 0.65 seconds

## Relational analysis of NS_B2_B1_A1_A2_A1_B2

### Relational analysis result of NS_B2_B1_A1_A2_A1_B2
Status: Status.UNKNOWN
Output dim: 0, lower bound: -20.9609915, upper bound: 20.9620561
time: 0.74 seconds

## BFS NS instance: NS_B2_B1_A1_A2_A2

### Backsubstitution after applying NS history:
0: -6.0549908, 16.9044552, -6.0461669, 16.9050064, -22.9599972, 22.9506207
1: -15.0391197, 26.0862732, -15.0026617, 26.0916557, -41.1307755, 41.0889282
2: -10.5142918, 23.6153870, -10.4874372, 23.6187649, -34.1330528, 34.1028252
3: -16.2734241, 28.8037796, -16.2466373, 28.7926636, -45.0660858, 45.0504150
4: -14.9265718, 29.1157589, -14.9184227, 29.1219597, -44.0485229, 44.0341797

Time for backsubstitution: 0.76 seconds

### NS candidates at layer 1
type: A, layer: 1, pos: 1
type: B, layer: 1, pos: 45
type: B, layer: 1, pos: 47
type: B, layer: 1, pos: 15
type: A, layer: 1, pos: 45
type: B, layer: 1, pos: 39
type: A, layer: 1, pos: 39
type: B, layer: 1, pos: 38
type: A, layer: 1, pos: 38
type: A, layer: 1, pos: 8
type: B, layer: 1, pos: 8
type: A, layer: 1, pos: 9
type: B, layer: 1, pos: 9
type: B, layer: 1, pos: 20
type: A, layer: 1, pos: 37
type: B, layer: 1, pos: 37
type: A, layer: 1, pos: 20
type: A, layer: 1, pos: 28
type: A, layer: 1, pos: 6
type: B, layer: 1, pos: 28
type: B, layer: 1, pos: 6
type: B, layer: 1, pos: 48
type: A, layer: 1, pos: 48
type: B, layer: 1, pos: 31
type: A, layer: 1, pos: 31
type: B, layer: 1, pos: 24
type: A, layer: 1, pos: 24
type: B, layer: 1, pos: 26
type: A, layer: 1, pos: 26
type: B, layer: 1, pos: 41
type: A, layer: 1, pos: 41
type: A, layer: 1, pos: 5
type: B, layer: 1, pos: 0
type: B, layer: 1, pos: 5
type: A, layer: 1, pos: 0

Time for candidate selection: 0.06 seconds

### Candidate
type: A, layer: 1, pos: 1

### Candidate
type: B, layer: 1, pos: 45

## Relational analysis of NS_B2_B1_A1_A2_A2_B1
Optimization infeasible because this subproblem isn't reachable.

### Candidate
type: B, layer: 1, pos: 47

## Relational analysis of NS_B2_B1_A1_A2_A2_B1

### Relational analysis result of NS_B2_B1_A1_A2_A2_B1
Status: Status.UNKNOWN
Output dim: 0, lower bound: -20.9614772, upper bound: 20.9614772
time: 1.18 seconds

## Relational analysis of NS_B2_B1_A1_A2_A2_B2

### Relational analysis result of NS_B2_B1_A1_A2_A2_B2
Status: Status.UNKNOWN
Output dim: 0, lower bound: -20.9614772, upper bound: 20.9614772
time: 0.71 seconds

## BFS NS instance: NS_B2_B1_A2_B2_A1

### Backsubstitution after applying NS history:
0: -5.9982681, 16.7135715, -5.7500262, 16.1319809, -22.1302471, 22.4635983
1: -15.0037842, 25.7121830, -14.2483215, 24.9541721, -39.9579506, 39.9605026
2: -10.4593410, 23.2528305, -9.9677210, 22.5709877, -33.0303268, 33.2205505
3: -16.0719185, 28.4230633, -15.4443340, 27.5256233, -43.5975418, 43.8673973
4: -14.4953861, 28.6492424, -14.1886473, 27.8220062, -42.3173904, 42.8378868

Time for backsubstitution: 0.76 seconds

### NS candidates at layer 1
type: A, layer: 1, pos: 1
type: B, layer: 1, pos: 45
type: A, layer: 1, pos: 45
type: A, layer: 1, pos: 15
type: B, layer: 1, pos: 47
type: B, layer: 1, pos: 48
type: B, layer: 1, pos: 39
type: B, layer: 1, pos: 38
type: A, layer: 1, pos: 38
type: B, layer: 1, pos: 9
type: A, layer: 1, pos: 28
type: A, layer: 1, pos: 9
type: A, layer: 1, pos: 8
type: A, layer: 1, pos: 6
type: B, layer: 1, pos: 37
type: A, layer: 1, pos: 37
type: A, layer: 1, pos: 48
type: B, layer: 1, pos: 6
type: B, layer: 1, pos: 20
type: B, layer: 1, pos: 8
type: B, layer: 1, pos: 28
type: A, layer: 1, pos: 20
type: A, layer: 1, pos: 31
type: A, layer: 1, pos: 39
type: B, layer: 1, pos: 31
type: A, layer: 1, pos: 24
type: B, layer: 1, pos: 24
type: A, layer: 1, pos: 5
type: A, layer: 1, pos: 0
type: B, layer: 1, pos: 41
type: B, layer: 1, pos: 26
type: A, layer: 1, pos: 41
type: B, layer: 1, pos: 0
type: A, layer: 1, pos: 26
type: B, layer: 1, pos: 5

Time for candidate selection: 0.06 seconds

### Candidate
type: A, layer: 1, pos: 1

### Candidate
type: B, layer: 1, pos: 45

### Candidate
type: A, layer: 1, pos: 45

### Candidate
type: A, layer: 1, pos: 15

## Relational analysis of NS_B2_B1_A2_B2_A1_A1
Optimization infeasible because this subproblem isn't reachable.

### Candidate
type: B, layer: 1, pos: 47

## Relational analysis of NS_B2_B1_A2_B2_A1_B1

### Relational analysis result of NS_B2_B1_A2_B2_A1_B1
Status: Status.UNKNOWN
Output dim: 0, lower bound: -20.9513168, upper bound: 20.9513119
time: 0.73 seconds

## Relational analysis of NS_B2_B1_A2_B2_A1_B2

### Relational analysis result of NS_B2_B1_A2_B2_A1_B2
Status: Status.UNKNOWN
Output dim: 0, lower bound: -20.9513168, upper bound: 20.9513119
time: 1.05 seconds

## BFS NS instance: NS_B2_B1_A2_B2_A2

### Backsubstitution after applying NS history:
0: -6.1924939, 17.1560497, -5.9021125, 16.5162315, -22.7087212, 23.0581627
1: -15.3266430, 26.4775848, -14.6226358, 25.5178680, -40.8445053, 41.1002159
2: -10.7523117, 23.9293728, -10.2336988, 23.0905800, -33.8428841, 34.1630707
3: -16.5479164, 29.2559509, -15.8537064, 28.1620789, -44.7099876, 45.1096573
4: -15.1311970, 29.4073639, -14.5781689, 28.4602432, -43.5914383, 43.9855309

Time for backsubstitution: 0.77 seconds

### NS candidates at layer 1
type: A, layer: 1, pos: 1
type: B, layer: 1, pos: 45
type: A, layer: 1, pos: 45
type: A, layer: 1, pos: 15
type: B, layer: 1, pos: 47
type: B, layer: 1, pos: 39
type: A, layer: 1, pos: 39
type: A, layer: 1, pos: 6
type: B, layer: 1, pos: 38
type: A, layer: 1, pos: 38
type: A, layer: 1, pos: 9
type: A, layer: 1, pos: 8
type: B, layer: 1, pos: 37
type: B, layer: 1, pos: 9
type: B, layer: 1, pos: 20
type: A, layer: 1, pos: 48
type: A, layer: 1, pos: 28
type: B, layer: 1, pos: 6
type: A, layer: 1, pos: 37
type: B, layer: 1, pos: 48
type: B, layer: 1, pos: 8
type: A, layer: 1, pos: 20
type: A, layer: 1, pos: 31
type: B, layer: 1, pos: 28
type: B, layer: 1, pos: 31
type: A, layer: 1, pos: 24
type: B, layer: 1, pos: 24
type: B, layer: 1, pos: 5
type: A, layer: 1, pos: 0
type: A, layer: 1, pos: 5
type: B, layer: 1, pos: 41
type: A, layer: 1, pos: 41
type: B, layer: 1, pos: 0
type: B, layer: 1, pos: 26
type: A, layer: 1, pos: 26

Time for candidate selection: 0.06 seconds

### Candidate
type: A, layer: 1, pos: 1

### Candidate
type: B, layer: 1, pos: 45

### Candidate
type: A, layer: 1, pos: 45

### Candidate
type: A, layer: 1, pos: 15

## Relational analysis of NS_B2_B1_A2_B2_A2_A1

### Relational analysis result of NS_B2_B1_A2_B2_A2_A1
Status: Status.VERIFIED
Output dim: 0, lower bound: -20.6505694, upper bound: 20.7210061
time: 0.66 seconds

## Relational analysis of NS_B2_B1_A2_B2_A2_A2

### Relational analysis result of NS_B2_B1_A2_B2_A2_A2
Status: Status.UNKNOWN
Output dim: 0, lower bound: -20.6505694, upper bound: 20.9509404
time: 0.71 seconds

## BFS NS instance: NS_B2_B2_A1_B1_A1

### Backsubstitution after applying NS history:
0: -6.0244703, 16.8405571, -5.7815499, 16.1679955, -22.1924667, 22.6221066
1: -15.1832008, 25.7038307, -14.4255676, 24.9067802, -40.0899811, 40.1293983
2: -10.5292540, 23.3857155, -10.0648251, 22.5165768, -33.0458298, 33.4505348
3: -16.2657433, 28.4767113, -15.4787817, 27.5215988, -43.7873421, 43.9554863
4: -14.6604919, 28.9407387, -13.9816208, 27.7338066, -42.3942986, 42.9223442

Time for backsubstitution: 0.76 seconds

### NS candidates at layer 1
type: A, layer: 1, pos: 12
type: A, layer: 1, pos: 28
type: A, layer: 1, pos: 45
type: A, layer: 1, pos: 15
type: B, layer: 1, pos: 28
type: B, layer: 1, pos: 45
type: B, layer: 1, pos: 15
type: B, layer: 1, pos: 6
type: A, layer: 1, pos: 6
type: A, layer: 1, pos: 24
type: A, layer: 1, pos: 48
type: B, layer: 1, pos: 39
type: B, layer: 1, pos: 8
type: B, layer: 1, pos: 9
type: B, layer: 1, pos: 24
type: A, layer: 1, pos: 37
type: A, layer: 1, pos: 38
type: A, layer: 1, pos: 9
type: B, layer: 1, pos: 37
type: A, layer: 1, pos: 20
type: A, layer: 1, pos: 39
type: B, layer: 1, pos: 38
type: B, layer: 1, pos: 26
type: A, layer: 1, pos: 26
type: B, layer: 1, pos: 48
type: A, layer: 1, pos: 8
type: A, layer: 1, pos: 0
type: B, layer: 1, pos: 20
type: A, layer: 1, pos: 41
type: B, layer: 1, pos: 41
type: B, layer: 1, pos: 31
type: B, layer: 1, pos: 5
type: A, layer: 1, pos: 5
type: A, layer: 1, pos: 31
type: B, layer: 1, pos: 0

Time for candidate selection: 0.06 seconds

### Candidate
type: A, layer: 1, pos: 12

### Candidate
type: A, layer: 1, pos: 28

### Candidate
type: A, layer: 1, pos: 45

### Candidate
type: A, layer: 1, pos: 15

### Candidate
type: B, layer: 1, pos: 28

### Candidate
type: B, layer: 1, pos: 45

### Candidate
type: B, layer: 1, pos: 15

### Candidate
type: B, layer: 1, pos: 6

## Relational analysis of NS_B2_B2_A1_B1_A1_B1
Optimization infeasible because this subproblem isn't reachable.

### Candidate
type: A, layer: 1, pos: 6

### Candidate
type: A, layer: 1, pos: 24

## Relational analysis of NS_B2_B2_A1_B1_A1_A1
Optimization infeasible because this subproblem isn't reachable.

### Candidate
type: A, layer: 1, pos: 48

## Relational analysis of NS_B2_B2_A1_B1_A1_A1
Optimization infeasible because this subproblem isn't reachable.

### Candidate
type: B, layer: 1, pos: 39

## Relational analysis of NS_B2_B2_A1_B1_A1_B1
Optimization infeasible because this subproblem isn't reachable.

### Candidate
type: B, layer: 1, pos: 8

## Relational analysis of NS_B2_B2_A1_B1_A1_B1

### Relational analysis result of NS_B2_B2_A1_B1_A1_B1
Status: Status.UNKNOWN
Output dim: 0, lower bound: -20.9511874, upper bound: 20.9463714
time: 0.59 seconds

## Relational analysis of NS_B2_B2_A1_B1_A1_B2

### Relational analysis result of NS_B2_B2_A1_B1_A1_B2
Status: Status.UNKNOWN
Output dim: 0, lower bound: -20.9523051, upper bound: 20.9485296
time: 0.64 seconds

## BFS NS instance: NS_B2_B2_A1_B1_A2

### Backsubstitution after applying NS history:
0: -6.2250795, 17.2942638, -5.7815499, 16.1679955, -22.3930740, 23.0758114
1: -15.5413942, 26.4833488, -14.4255676, 24.9067802, -40.4481697, 40.9089165
2: -10.8425531, 24.0679264, -10.0648251, 22.5165768, -33.3591309, 34.1327515
3: -16.7520714, 29.3229733, -15.4787817, 27.5215988, -44.2736702, 44.8017540
4: -15.2782488, 29.7155247, -13.9816208, 27.7338066, -43.0120544, 43.6971436

Time for backsubstitution: 0.77 seconds

### NS candidates at layer 1
type: A, layer: 1, pos: 12
type: A, layer: 1, pos: 45
type: A, layer: 1, pos: 28
type: B, layer: 1, pos: 45
type: A, layer: 1, pos: 15
type: B, layer: 1, pos: 28
type: A, layer: 1, pos: 48
type: A, layer: 1, pos: 6
type: B, layer: 1, pos: 15
type: B, layer: 1, pos: 6
type: A, layer: 1, pos: 24
type: B, layer: 1, pos: 24
type: B, layer: 1, pos: 8
type: B, layer: 1, pos: 9
type: A, layer: 1, pos: 39
type: A, layer: 1, pos: 9
type: A, layer: 1, pos: 38
type: B, layer: 1, pos: 37
type: A, layer: 1, pos: 37
type: B, layer: 1, pos: 39
type: B, layer: 1, pos: 38
type: A, layer: 1, pos: 20
type: A, layer: 1, pos: 8
type: A, layer: 1, pos: 26
type: B, layer: 1, pos: 26
type: B, layer: 1, pos: 20
type: B, layer: 1, pos: 48
type: A, layer: 1, pos: 0
type: B, layer: 1, pos: 5
type: A, layer: 1, pos: 41
type: B, layer: 1, pos: 31
type: B, layer: 1, pos: 41
type: A, layer: 1, pos: 31
type: B, layer: 1, pos: 0
type: A, layer: 1, pos: 5

Time for candidate selection: 0.06 seconds

### Candidate
type: A, layer: 1, pos: 12

### Candidate
type: A, layer: 1, pos: 45

### Candidate
type: A, layer: 1, pos: 28

### Candidate
type: B, layer: 1, pos: 45

### Candidate
type: A, layer: 1, pos: 15

### Candidate
type: B, layer: 1, pos: 28

### Candidate
type: A, layer: 1, pos: 48

## Relational analysis of NS_B2_B2_A1_B1_A2_A1
Optimization infeasible because this subproblem isn't reachable.

### Candidate
type: A, layer: 1, pos: 6

## Relational analysis of NS_B2_B2_A1_B1_A2_A1

### Relational analysis result of NS_B2_B2_A1_B1_A2_A1
Status: Status.UNKNOWN
Output dim: 0, lower bound: -20.9390731, upper bound: 20.9274296
time: 0.65 seconds

## Relational analysis of NS_B2_B2_A1_B1_A2_A2

### Relational analysis result of NS_B2_B2_A1_B1_A2_A2
Status: Status.UNKNOWN
Output dim: 0, lower bound: -20.9515325, upper bound: 20.9479840
time: 0.63 seconds

## BFS NS instance: NS_B2_B2_A1_B2_A1

### Backsubstitution after applying NS history:
0: -6.0244703, 16.8405571, -5.9625483, 16.5776463, -22.6021137, 22.8031006
1: -15.1832008, 25.7038307, -14.7118168, 25.6270142, -40.8102150, 40.4156494
2: -10.5292540, 23.3857155, -10.3330498, 23.1492825, -33.6785355, 33.7187653
3: -16.2657433, 28.4767113, -15.9134026, 28.2967262, -44.5624695, 44.3901138
4: -14.6604919, 28.9407387, -14.5857401, 28.4363346, -43.0968246, 43.5264702

Time for backsubstitution: 0.77 seconds

### NS candidates at layer 1
type: A, layer: 1, pos: 12
type: A, layer: 1, pos: 45
type: A, layer: 1, pos: 15
type: A, layer: 1, pos: 28
type: B, layer: 1, pos: 15
type: B, layer: 1, pos: 39
type: B, layer: 1, pos: 45
type: B, layer: 1, pos: 6
type: A, layer: 1, pos: 24
type: B, layer: 1, pos: 8
type: B, layer: 1, pos: 9
type: A, layer: 1, pos: 6
type: B, layer: 1, pos: 28
type: A, layer: 1, pos: 38
type: A, layer: 1, pos: 37
type: A, layer: 1, pos: 20
type: A, layer: 1, pos: 9
type: A, layer: 1, pos: 48
type: B, layer: 1, pos: 24
type: B, layer: 1, pos: 48
type: B, layer: 1, pos: 37
type: B, layer: 1, pos: 38
type: A, layer: 1, pos: 8
type: A, layer: 1, pos: 0
type: B, layer: 1, pos: 26
type: A, layer: 1, pos: 26
type: A, layer: 1, pos: 39
type: B, layer: 1, pos: 31
type: B, layer: 1, pos: 41
type: A, layer: 1, pos: 5
type: A, layer: 1, pos: 41
type: B, layer: 1, pos: 0
type: B, layer: 1, pos: 20
type: A, layer: 1, pos: 31
type: B, layer: 1, pos: 5

Time for candidate selection: 0.06 seconds

### Candidate
type: A, layer: 1, pos: 12

### Candidate
type: A, layer: 1, pos: 45

### Candidate
type: A, layer: 1, pos: 15

## Relational analysis of NS_B2_B2_A1_B2_A1_A1
Optimization infeasible because this subproblem isn't reachable.

### Candidate
type: A, layer: 1, pos: 28

## Relational analysis of NS_B2_B2_A1_B2_A1_A1
Optimization infeasible because this subproblem isn't reachable.

### Candidate
type: B, layer: 1, pos: 15

### Candidate
type: B, layer: 1, pos: 39

## Relational analysis of NS_B2_B2_A1_B2_A1_B1
Optimization infeasible because this subproblem isn't reachable.

### Candidate
type: B, layer: 1, pos: 45

### Candidate
type: B, layer: 1, pos: 6

## Relational analysis of NS_B2_B2_A1_B2_A1_B1
Optimization infeasible because this subproblem isn't reachable.

### Candidate
type: A, layer: 1, pos: 24

## Relational analysis of NS_B2_B2_A1_B2_A1_A1
Optimization infeasible because this subproblem isn't reachable.

### Candidate
type: B, layer: 1, pos: 8

## Relational analysis of NS_B2_B2_A1_B2_A1_B1

### Relational analysis result of NS_B2_B2_A1_B2_A1_B1
Status: Status.UNKNOWN
Output dim: 0, lower bound: -20.9326393, upper bound: 20.9340069
time: 1.14 seconds

## Relational analysis of NS_B2_B2_A1_B2_A1_B2

### Relational analysis result of NS_B2_B2_A1_B2_A1_B2
Status: Status.UNKNOWN
Output dim: 0, lower bound: -20.9361618, upper bound: 20.9403090
time: 0.70 seconds

## BFS NS instance: NS_B2_B2_A1_B2_A2

### Backsubstitution after applying NS history:
0: -6.2250795, 17.2942638, -5.9625483, 16.5776463, -22.8027248, 23.2568073
1: -15.5413942, 26.4833488, -14.7118168, 25.6270142, -41.1683960, 41.1951675
2: -10.8425531, 24.0679264, -10.3330498, 23.1492825, -33.9918327, 34.4009781
3: -16.7520714, 29.3229733, -15.9134026, 28.2967262, -45.0487976, 45.2363739
4: -15.2782488, 29.7155247, -14.5857401, 28.4363346, -43.7145805, 44.3012657

Time for backsubstitution: 0.77 seconds

### NS candidates at layer 1
type: A, layer: 1, pos: 12
type: A, layer: 1, pos: 45
type: A, layer: 1, pos: 15
type: B, layer: 1, pos: 45
type: A, layer: 1, pos: 28
type: B, layer: 1, pos: 15
type: B, layer: 1, pos: 6
type: A, layer: 1, pos: 6
type: A, layer: 1, pos: 24
type: A, layer: 1, pos: 48
type: B, layer: 1, pos: 8
type: B, layer: 1, pos: 9
type: B, layer: 1, pos: 28
type: B, layer: 1, pos: 39
type: A, layer: 1, pos: 9
type: B, layer: 1, pos: 24
type: A, layer: 1, pos: 38
type: A, layer: 1, pos: 37
type: A, layer: 1, pos: 20
type: B, layer: 1, pos: 37
type: B, layer: 1, pos: 38
type: A, layer: 1, pos: 8
type: B, layer: 1, pos: 48
type: A, layer: 1, pos: 39
type: B, layer: 1, pos: 26
type: A, layer: 1, pos: 26
type: A, layer: 1, pos: 0
type: B, layer: 1, pos: 20
type: B, layer: 1, pos: 31
type: A, layer: 1, pos: 41
type: B, layer: 1, pos: 41
type: A, layer: 1, pos: 5
type: B, layer: 1, pos: 5
type: A, layer: 1, pos: 31
type: B, layer: 1, pos: 0

Time for candidate selection: 0.06 seconds

### Candidate
type: A, layer: 1, pos: 12

### Candidate
type: A, layer: 1, pos: 45

### Candidate
type: A, layer: 1, pos: 15

## Relational analysis of NS_B2_B2_A1_B2_A2_A1
Optimization infeasible because this subproblem isn't reachable.

### Candidate
type: B, layer: 1, pos: 45

### Candidate
type: A, layer: 1, pos: 28

## Relational analysis of NS_B2_B2_A1_B2_A2_A1
Optimization infeasible because this subproblem isn't reachable.

### Candidate
type: B, layer: 1, pos: 15

### Candidate
type: B, layer: 1, pos: 6

## Relational analysis of NS_B2_B2_A1_B2_A2_B1
Optimization infeasible because this subproblem isn't reachable.

### Candidate
type: A, layer: 1, pos: 6

### Candidate
type: A, layer: 1, pos: 24

## Relational analysis of NS_B2_B2_A1_B2_A2_A1
Optimization infeasible because this subproblem isn't reachable.

### Candidate
type: A, layer: 1, pos: 48

## Relational analysis of NS_B2_B2_A1_B2_A2_A1
Optimization infeasible because this subproblem isn't reachable.

### Candidate
type: B, layer: 1, pos: 8

## Relational analysis of NS_B2_B2_A1_B2_A2_B1

### Relational analysis result of NS_B2_B2_A1_B2_A2_B1
Status: Status.UNKNOWN
Output dim: 0, lower bound: -20.9313875, upper bound: 20.9340069
time: 0.63 seconds

## Relational analysis of NS_B2_B2_A1_B2_A2_B2

### Relational analysis result of NS_B2_B2_A1_B2_A2_B2
Status: Status.UNKNOWN
Output dim: 0, lower bound: -20.9408792, upper bound: 20.9403090
time: 0.64 seconds

## BFS NS instance: NS_B2_B2_A2_B1_A1

### Backsubstitution after applying NS history:
0: -5.8784394, 16.4799786, -5.7815499, 16.1679955, -22.0464344, 22.2615280
1: -14.5903740, 25.4686127, -14.4255676, 24.9067802, -39.4971542, 39.8941803
2: -10.1944265, 23.0433502, -10.0648251, 22.5165768, -32.7109985, 33.1081734
3: -15.7954006, 28.0894241, -15.4787817, 27.5215988, -43.3169975, 43.5682030
4: -14.4894428, 28.4152985, -13.9816208, 27.7338066, -42.2232475, 42.3969193

Time for backsubstitution: 0.78 seconds

### NS candidates at layer 1
type: A, layer: 1, pos: 15
type: A, layer: 1, pos: 45
type: B, layer: 1, pos: 15
type: B, layer: 1, pos: 45
type: A, layer: 1, pos: 47
type: A, layer: 1, pos: 39
type: B, layer: 1, pos: 28
type: A, layer: 1, pos: 48
type: A, layer: 1, pos: 38
type: B, layer: 1, pos: 38
type: A, layer: 1, pos: 9
type: B, layer: 1, pos: 8
type: B, layer: 1, pos: 9
type: A, layer: 1, pos: 28
type: B, layer: 1, pos: 6
type: B, layer: 1, pos: 37
type: A, layer: 1, pos: 20
type: A, layer: 1, pos: 37
type: A, layer: 1, pos: 6
type: B, layer: 1, pos: 20
type: B, layer: 1, pos: 48
type: B, layer: 1, pos: 39
type: A, layer: 1, pos: 8
type: B, layer: 1, pos: 31
type: A, layer: 1, pos: 31
type: B, layer: 1, pos: 24
type: A, layer: 1, pos: 24
type: B, layer: 1, pos: 0
type: B, layer: 1, pos: 5
type: A, layer: 1, pos: 0
type: A, layer: 1, pos: 26
type: B, layer: 1, pos: 26
type: A, layer: 1, pos: 41
type: B, layer: 1, pos: 41
type: A, layer: 1, pos: 5

Time for candidate selection: 0.06 seconds

### Candidate
type: A, layer: 1, pos: 15

## Relational analysis of NS_B2_B2_A2_B1_A1_A1
Optimization infeasible because this subproblem isn't reachable.

### Candidate
type: A, layer: 1, pos: 45

## Relational analysis of NS_B2_B2_A2_B1_A1_A1
Optimization infeasible because this subproblem isn't reachable.

### Candidate
type: B, layer: 1, pos: 15

### Candidate
type: B, layer: 1, pos: 45

### Candidate
type: A, layer: 1, pos: 47

## Relational analysis of NS_B2_B2_A2_B1_A1_A1

### Relational analysis result of NS_B2_B2_A2_B1_A1_A1
Status: Status.UNKNOWN
Output dim: 0, lower bound: -20.9569857, upper bound: 20.9544784
time: 0.69 seconds

## Relational analysis of NS_B2_B2_A2_B1_A1_A2

### Relational analysis result of NS_B2_B2_A2_B1_A1_A2
Status: Status.UNKNOWN
Output dim: 0, lower bound: -20.9569857, upper bound: 20.9544784
time: 0.59 seconds

## BFS NS instance: NS_B2_B2_A2_B1_A2

### Backsubstitution after applying NS history:
0: -5.8447042, 16.2864189, -5.7815499, 16.1679955, -22.0126991, 22.0679684
1: -14.4215288, 25.2035751, -14.4255676, 24.9067802, -39.3283081, 39.6291351
2: -10.1243515, 22.7505417, -10.0648251, 22.5165768, -32.6409302, 32.8153610
3: -15.5963087, 27.8081303, -15.4787817, 27.5215988, -43.1179085, 43.2869072
4: -14.2942791, 27.9535198, -13.9816208, 27.7338066, -42.0280838, 41.9351425

Time for backsubstitution: 0.77 seconds

### NS candidates at layer 1
type: A, layer: 1, pos: 45
type: B, layer: 1, pos: 45
type: B, layer: 1, pos: 28
type: A, layer: 1, pos: 47
type: A, layer: 1, pos: 28
type: B, layer: 1, pos: 20
type: A, layer: 1, pos: 15
type: A, layer: 1, pos: 39
type: B, layer: 1, pos: 15
type: A, layer: 1, pos: 6
type: A, layer: 1, pos: 48
type: A, layer: 1, pos: 20
type: B, layer: 1, pos: 24
type: B, layer: 1, pos: 37
type: A, layer: 1, pos: 9
type: B, layer: 1, pos: 6
type: A, layer: 1, pos: 8
type: A, layer: 1, pos: 24
type: B, layer: 1, pos: 9
type: B, layer: 1, pos: 8
type: A, layer: 1, pos: 37
type: B, layer: 1, pos: 38
type: A, layer: 1, pos: 38
type: B, layer: 1, pos: 39
type: B, layer: 1, pos: 48
type: A, layer: 1, pos: 26
type: B, layer: 1, pos: 26
type: A, layer: 1, pos: 31
type: B, layer: 1, pos: 5
type: B, layer: 1, pos: 31
type: B, layer: 1, pos: 0
type: A, layer: 1, pos: 41
type: B, layer: 1, pos: 41
type: A, layer: 1, pos: 0
type: A, layer: 1, pos: 5

Time for candidate selection: 0.06 seconds

### Candidate
type: A, layer: 1, pos: 45

## Relational analysis of NS_B2_B2_A2_B1_A2_A1
Optimization infeasible because this subproblem isn't reachable.

### Candidate
type: B, layer: 1, pos: 45

### Candidate
type: B, layer: 1, pos: 28

## Relational analysis of NS_B2_B2_A2_B1_A2_B1
Optimization infeasible because this subproblem isn't reachable.

### Candidate
type: A, layer: 1, pos: 47

## Relational analysis of NS_B2_B2_A2_B1_A2_A1

### Relational analysis result of NS_B2_B2_A2_B1_A2_A1
Status: Status.UNKNOWN
Output dim: 0, lower bound: -20.9569857, upper bound: 20.9544784
time: 0.69 seconds

## Relational analysis of NS_B2_B2_A2_B1_A2_A2

### Relational analysis result of NS_B2_B2_A2_B1_A2_A2
Status: Status.UNKNOWN
Output dim: 0, lower bound: -20.9569857, upper bound: 20.9544784
time: 0.64 seconds

## Summary of splitting at layer (split count: 5)
- Time for NS candidates: 2.64 seconds
NS_B1_A1_A1_A1_B1_B1, status: Status.UNKNOWN, split count: 6, time: 2.64
Output dim: 0, lower bound: -20.9235252, upper bound: 20.9362552
NS_B1_A1_A1_A1_B1_B2, status: Status.UNKNOWN, split count: 6, time: 2.64
Output dim: 0, lower bound: -20.9235252, upper bound: 20.9362552
NS_B1_A1_A1_A1_B2_B1, status: Status.UNKNOWN, split count: 6, time: 2.64
Output dim: 0, lower bound: -20.9235252, upper bound: 20.9362552
NS_B1_A1_A1_A1_B2_B2, status: Status.UNKNOWN, split count: 6, time: 2.64
Output dim: 0, lower bound: -20.9235252, upper bound: 20.9362552
NS_B1_A1_A1_A2_B1_B1, status: Status.UNKNOWN, split count: 6, time: 2.64
Output dim: 0, lower bound: -20.9108170, upper bound: 20.9172336
NS_B1_A1_A1_A2_B1_B2, status: Status.UNKNOWN, split count: 6, time: 2.64
Output dim: 0, lower bound: -20.9108170, upper bound: 20.9172336
NS_B1_A1_A1_A2_B2_B1, status: Status.UNKNOWN, split count: 6, time: 2.64
Output dim: 0, lower bound: -20.9108170, upper bound: 20.9172336
NS_B1_A1_A1_A2_B2_B2, status: Status.UNKNOWN, split count: 6, time: 2.64
Output dim: 0, lower bound: -20.9108170, upper bound: 20.9172336
NS_B1_A2_A1_B1_B1_A1, status: Status.UNKNOWN, split count: 6, time: 2.64
Output dim: 0, lower bound: -20.9425656, upper bound: 20.9303604
NS_B1_A2_A1_B1_B1_A2, status: Status.UNKNOWN, split count: 6, time: 2.64
Output dim: 0, lower bound: -20.9601795, upper bound: 20.9570587
NS_B1_A2_A1_B1_B2_A1, status: Status.UNKNOWN, split count: 6, time: 2.64
Output dim: 0, lower bound: -20.9501995, upper bound: 20.9441063
NS_B1_A2_A1_B1_B2_A2, status: Status.UNKNOWN, split count: 6, time: 2.64
Output dim: 0, lower bound: -20.9601795, upper bound: 20.9633596
NS_B1_A2_A1_B2_B1_A1, status: Status.UNKNOWN, split count: 6, time: 2.64
Output dim: 0, lower bound: -20.9023718, upper bound: 20.9081548
NS_B1_A2_A1_B2_B1_A2, status: Status.UNKNOWN, split count: 6, time: 2.64
Output dim: 0, lower bound: -20.9024147, upper bound: 20.9387501
NS_B1_A2_A1_B2_B2_B1, status: Status.UNKNOWN, split count: 6, time: 2.64
Output dim: 0, lower bound: -20.9176398, upper bound: 20.9434239
NS_B1_A2_A1_B2_B2_B2, status: Status.UNKNOWN, split count: 6, time: 2.64
Output dim: 0, lower bound: -20.9476371, upper bound: 20.9513798
NS_B2_B1_A1_A2_A1_B1, status: Status.UNKNOWN, split count: 6, time: 2.64
Output dim: 0, lower bound: -20.9608208, upper bound: 20.9611716
NS_B2_B1_A1_A2_A1_B2, status: Status.UNKNOWN, split count: 6, time: 2.64
Output dim: 0, lower bound: -20.9609915, upper bound: 20.9620561
NS_B2_B1_A1_A2_A2_B1, status: Status.UNKNOWN, split count: 6, time: 2.64
Output dim: 0, lower bound: -20.9614772, upper bound: 20.9614772
NS_B2_B1_A1_A2_A2_B2, status: Status.UNKNOWN, split count: 6, time: 2.64
Output dim: 0, lower bound: -20.9614772, upper bound: 20.9614772
NS_B2_B1_A2_B2_A1_B1, status: Status.UNKNOWN, split count: 6, time: 2.64
Output dim: 0, lower bound: -20.9513168, upper bound: 20.9513119
NS_B2_B1_A2_B2_A1_B2, status: Status.UNKNOWN, split count: 6, time: 2.64
Output dim: 0, lower bound: -20.9513168, upper bound: 20.9513119
NS_B2_B1_A2_B2_A2_A1, status: Status.VERIFIED, split count: 6, time: 2.64
Output dim: 0, lower bound: -20.6505694, upper bound: 20.7210061
NS_B2_B1_A2_B2_A2_A2, status: Status.UNKNOWN, split count: 6, time: 2.64
Output dim: 0, lower bound: -20.6505694, upper bound: 20.9509404
NS_B2_B2_A1_B1_A1_B1, status: Status.UNKNOWN, split count: 6, time: 2.64
Output dim: 0, lower bound: -20.9511874, upper bound: 20.9463714
NS_B2_B2_A1_B1_A1_B2, status: Status.UNKNOWN, split count: 6, time: 2.64
Output dim: 0, lower bound: -20.9523051, upper bound: 20.9485296
NS_B2_B2_A1_B1_A2_A1, status: Status.UNKNOWN, split count: 6, time: 2.64
Output dim: 0, lower bound: -20.9390731, upper bound: 20.9274296
NS_B2_B2_A1_B1_A2_A2, status: Status.UNKNOWN, split count: 6, time: 2.64
Output dim: 0, lower bound: -20.9515325, upper bound: 20.9479840
NS_B2_B2_A1_B2_A1_B1, status: Status.UNKNOWN, split count: 6, time: 2.64
Output dim: 0, lower bound: -20.9326393, upper bound: 20.9340069
NS_B2_B2_A1_B2_A1_B2, status: Status.UNKNOWN, split count: 6, time: 2.64
Output dim: 0, lower bound: -20.9361618, upper bound: 20.9403090
NS_B2_B2_A1_B2_A2_B1, status: Status.UNKNOWN, split count: 6, time: 2.64
Output dim: 0, lower bound: -20.9313875, upper bound: 20.9340069
NS_B2_B2_A1_B2_A2_B2, status: Status.UNKNOWN, split count: 6, time: 2.64
Output dim: 0, lower bound: -20.9408792, upper bound: 20.9403090
NS_B2_B2_A2_B1_A1_A1, status: Status.UNKNOWN, split count: 6, time: 2.64
Output dim: 0, lower bound: -20.9569857, upper bound: 20.9544784
NS_B2_B2_A2_B1_A1_A2, status: Status.UNKNOWN, split count: 6, time: 2.64
Output dim: 0, lower bound: -20.9569857, upper bound: 20.9544784
NS_B2_B2_A2_B1_A2_A1, status: Status.UNKNOWN, split count: 6, time: 2.64
Output dim: 0, lower bound: -20.9569857, upper bound: 20.9544784
NS_B2_B2_A2_B1_A2_A2, status: Status.UNKNOWN, split count: 6, time: 2.64
Output dim: 0, lower bound: -20.9569857, upper bound: 20.9544784

## BFS NS instance: NS_B1_A1_A1_A1_B1_B1

### Backsubstitution after applying NS history:
0: -6.0846200, 16.9946613, -6.0846200, 16.9946613, -23.0792809, 23.0792809
1: -15.3345718, 25.8991718, -15.3345718, 25.8991718, -41.2337341, 41.2337341
2: -10.6436501, 23.5850830, -10.6436501, 23.5850830, -34.2287254, 34.2287254
3: -16.4320755, 28.7094727, -16.4320755, 28.7094727, -45.1415482, 45.1415482
4: -14.8261871, 29.1902122, -14.8261871, 29.1902122, -44.0163918, 44.0163956

Time for backsubstitution: 0.76 seconds

### NS candidates at layer 1
type: A, layer: 1, pos: 12
type: B, layer: 1, pos: 12
type: A, layer: 1, pos: 15
type: B, layer: 1, pos: 15
type: A, layer: 1, pos: 28
type: B, layer: 1, pos: 28
type: B, layer: 1, pos: 6
type: A, layer: 1, pos: 6
type: A, layer: 1, pos: 24
type: B, layer: 1, pos: 24
type: A, layer: 1, pos: 9
type: B, layer: 1, pos: 9
type: B, layer: 1, pos: 37
type: A, layer: 1, pos: 37
type: A, layer: 1, pos: 8
type: B, layer: 1, pos: 8
type: B, layer: 1, pos: 39
type: A, layer: 1, pos: 39
type: A, layer: 1, pos: 38
type: B, layer: 1, pos: 38
type: B, layer: 1, pos: 48
type: A, layer: 1, pos: 48
type: A, layer: 1, pos: 20
type: B, layer: 1, pos: 20
type: B, layer: 1, pos: 26
type: A, layer: 1, pos: 26
type: A, layer: 1, pos: 41
type: B, layer: 1, pos: 41
type: B, layer: 1, pos: 0
type: A, layer: 1, pos: 0
type: A, layer: 1, pos: 5
type: B, layer: 1, pos: 5
type: B, layer: 1, pos: 31
type: A, layer: 1, pos: 31

Time for candidate selection: 0.06 seconds

### Candidate
type: A, layer: 1, pos: 12

### Candidate
type: B, layer: 1, pos: 12

### Candidate
type: A, layer: 1, pos: 15

### Candidate
type: B, layer: 1, pos: 15

### Candidate
type: A, layer: 1, pos: 28

### Candidate
type: B, layer: 1, pos: 28

### Candidate
type: B, layer: 1, pos: 6

## Relational analysis of NS_B1_A1_A1_A1_B1_B1_B1

### Relational analysis result of NS_B1_A1_A1_A1_B1_B1_B1
Status: Status.UNKNOWN
Output dim: 0, lower bound: -20.9580142, upper bound: 20.9548157
time: 0.51 seconds

## Relational analysis of NS_B1_A1_A1_A1_B1_B1_B2
Optimization infeasible because this subproblem isn't reachable.

### Candidate
type: A, layer: 1, pos: 6

### Candidate
type: A, layer: 1, pos: 24

## Relational analysis of NS_B1_A1_A1_A1_B1_B1_A1
Optimization infeasible because this subproblem isn't reachable.

### Candidate
type: B, layer: 1, pos: 24

### Candidate
type: A, layer: 1, pos: 9

## Relational analysis of NS_B1_A1_A1_A1_B1_B1_A1

### Relational analysis result of NS_B1_A1_A1_A1_B1_B1_A1
Status: Status.UNKNOWN
Output dim: 0, lower bound: -20.9355777, upper bound: 20.9196433
time: 0.56 seconds

## Relational analysis of NS_B1_A1_A1_A1_B1_B1_A2

### Relational analysis result of NS_B1_A1_A1_A1_B1_B1_A2
Status: Status.UNKNOWN
Output dim: 0, lower bound: -20.9584755, upper bound: 20.9553408
time: 0.57 seconds

## BFS NS instance: NS_B1_A1_A1_A1_B1_B2

### Backsubstitution after applying NS history:
0: -6.0846200, 16.9946613, -6.7040877, 18.7079010, -24.7925205, 23.6987476
1: -15.3345718, 25.8991718, -16.8718414, 28.4157085, -43.7502747, 42.7710114
2: -10.6436501, 23.5850830, -11.7658329, 25.8589211, -36.5025673, 35.3509140
3: -16.4320755, 28.7094727, -18.1079674, 31.5177822, -47.9498596, 46.8174400
4: -14.8261871, 29.1902122, -16.3970699, 32.0241547, -46.8503418, 45.5872726

Time for backsubstitution: 0.76 seconds

### NS candidates at layer 1
type: B, layer: 1, pos: 12
type: A, layer: 1, pos: 12
type: A, layer: 1, pos: 15
type: B, layer: 1, pos: 15
type: B, layer: 1, pos: 6
type: B, layer: 1, pos: 28
type: A, layer: 1, pos: 28
type: B, layer: 1, pos: 24
type: B, layer: 1, pos: 9
type: B, layer: 1, pos: 39
type: A, layer: 1, pos: 24
type: A, layer: 1, pos: 37
type: A, layer: 1, pos: 9
type: B, layer: 1, pos: 37
type: A, layer: 1, pos: 6
type: A, layer: 1, pos: 8
type: B, layer: 1, pos: 8
type: A, layer: 1, pos: 38
type: A, layer: 1, pos: 48
type: B, layer: 1, pos: 38
type: B, layer: 1, pos: 48
type: A, layer: 1, pos: 39
type: A, layer: 1, pos: 20
type: A, layer: 1, pos: 26
type: B, layer: 1, pos: 20
type: B, layer: 1, pos: 26
type: A, layer: 1, pos: 41
type: B, layer: 1, pos: 41
type: B, layer: 1, pos: 5
type: B, layer: 1, pos: 0
type: A, layer: 1, pos: 0
type: B, layer: 1, pos: 31
type: A, layer: 1, pos: 31
type: A, layer: 1, pos: 5

Time for candidate selection: 0.06 seconds

### Candidate
type: B, layer: 1, pos: 12

### Candidate
type: A, layer: 1, pos: 12

### Candidate
type: A, layer: 1, pos: 15

### Candidate
type: B, layer: 1, pos: 15

### Candidate
type: B, layer: 1, pos: 6

## Relational analysis of NS_B1_A1_A1_A1_B1_B2_B1

### Relational analysis result of NS_B1_A1_A1_A1_B1_B2_B1
Status: Status.UNKNOWN
Output dim: 0, lower bound: -20.9580142, upper bound: 20.9548157
time: 0.56 seconds

## Relational analysis of NS_B1_A1_A1_A1_B1_B2_B2
Optimization infeasible because this subproblem isn't reachable.

### Candidate
type: B, layer: 1, pos: 28

### Candidate
type: A, layer: 1, pos: 28

### Candidate
type: B, layer: 1, pos: 24

## Relational analysis of NS_B1_A1_A1_A1_B1_B2_B1
Optimization infeasible because this subproblem isn't reachable.

### Candidate
type: B, layer: 1, pos: 9

## Relational analysis of NS_B1_A1_A1_A1_B1_B2_B1
Optimization infeasible because this subproblem isn't reachable.

### Candidate
type: B, layer: 1, pos: 39

## Relational analysis of NS_B1_A1_A1_A1_B1_B2_B1
Optimization infeasible because this subproblem isn't reachable.

### Candidate
type: A, layer: 1, pos: 24

### Candidate
type: A, layer: 1, pos: 37

## Relational analysis of NS_B1_A1_A1_A1_B1_B2_A1
Optimization infeasible because this subproblem isn't reachable.

### Candidate
type: A, layer: 1, pos: 9

### Candidate
type: B, layer: 1, pos: 37

### Candidate
type: A, layer: 1, pos: 6

### Candidate
type: A, layer: 1, pos: 8

## Relational analysis of NS_B1_A1_A1_A1_B1_B2_A1

### Relational analysis result of NS_B1_A1_A1_A1_B1_B2_A1
Status: Status.UNKNOWN
Output dim: 0, lower bound: -20.9577583, upper bound: 20.9550095
time: 0.51 seconds

## Relational analysis of NS_B1_A1_A1_A1_B1_B2_A2

### Relational analysis result of NS_B1_A1_A1_A1_B1_B2_A2
Status: Status.UNKNOWN
Output dim: 0, lower bound: -20.9577507, upper bound: 20.9549928
time: 0.61 seconds

## BFS NS instance: NS_B1_A1_A1_A1_B2_B1

### Backsubstitution after applying NS history:
0: -6.0846200, 16.9946613, -6.1871290, 17.1735039, -23.2581234, 23.1817894
1: -15.3345718, 25.8991718, -15.4273548, 26.2602158, -41.5947762, 41.3265266
2: -10.6436501, 23.5850830, -10.7745504, 23.8967419, -34.5403862, 34.3596268
3: -16.4320755, 28.7094727, -16.6525230, 29.0933285, -45.5254059, 45.3619919
4: -14.8261871, 29.1902122, -15.2049751, 29.4977074, -44.3238945, 44.3951836

Time for backsubstitution: 0.76 seconds

### NS candidates at layer 1
type: B, layer: 1, pos: 12
type: A, layer: 1, pos: 12
type: B, layer: 1, pos: 15
type: A, layer: 1, pos: 15
type: A, layer: 1, pos: 28
type: B, layer: 1, pos: 6
type: B, layer: 1, pos: 28
type: A, layer: 1, pos: 24
type: B, layer: 1, pos: 39
type: A, layer: 1, pos: 6
type: B, layer: 1, pos: 9
type: B, layer: 1, pos: 24
type: A, layer: 1, pos: 37
type: A, layer: 1, pos: 9
type: B, layer: 1, pos: 48
type: B, layer: 1, pos: 37
type: B, layer: 1, pos: 8
type: A, layer: 1, pos: 8
type: A, layer: 1, pos: 38
type: B, layer: 1, pos: 38
type: A, layer: 1, pos: 20
type: A, layer: 1, pos: 39
type: A, layer: 1, pos: 48
type: B, layer: 1, pos: 26
type: A, layer: 1, pos: 26
type: B, layer: 1, pos: 20
type: B, layer: 1, pos: 41
type: A, layer: 1, pos: 41
type: A, layer: 1, pos: 0
type: A, layer: 1, pos: 5
type: B, layer: 1, pos: 31
type: B, layer: 1, pos: 0
type: B, layer: 1, pos: 5
type: A, layer: 1, pos: 31

Time for candidate selection: 0.06 seconds

### Candidate
type: B, layer: 1, pos: 12

### Candidate
type: A, layer: 1, pos: 12

### Candidate
type: B, layer: 1, pos: 15

### Candidate
type: A, layer: 1, pos: 15

### Candidate
type: A, layer: 1, pos: 28

### Candidate
type: B, layer: 1, pos: 6

### Candidate
type: B, layer: 1, pos: 28

### Candidate
type: A, layer: 1, pos: 24

## Relational analysis of NS_B1_A1_A1_A1_B2_B1_A1
Optimization infeasible because this subproblem isn't reachable.

### Candidate
type: B, layer: 1, pos: 39

## Relational analysis of NS_B1_A1_A1_A1_B2_B1_B1
Optimization infeasible because this subproblem isn't reachable.

### Candidate
type: A, layer: 1, pos: 6

### Candidate
type: B, layer: 1, pos: 9

## Relational analysis of NS_B1_A1_A1_A1_B2_B1_B1

### Relational analysis result of NS_B1_A1_A1_A1_B2_B1_B1
Status: Status.UNKNOWN
Output dim: 0, lower bound: -20.9106196, upper bound: 20.9286097
time: 0.59 seconds

## Relational analysis of NS_B1_A1_A1_A1_B2_B1_B2

### Relational analysis result of NS_B1_A1_A1_A1_B2_B1_B2
Status: Status.UNKNOWN
Output dim: 0, lower bound: -20.9222198, upper bound: 20.9359225
time: 0.61 seconds

## BFS NS instance: NS_B1_A1_A1_A1_B2_B2

### Backsubstitution after applying NS history:
0: -6.0846200, 16.9946613, -6.9751034, 19.2829285, -25.3675480, 23.9697628
1: -15.3345718, 25.8991718, -17.3758297, 29.4068832, -44.7414474, 43.2749939
2: -10.6436501, 23.5850830, -12.1740484, 26.7304134, -37.3740540, 35.7591324
3: -16.4320755, 28.7094727, -18.7746525, 32.6108894, -49.0429649, 47.4841232
4: -14.8261871, 29.1902122, -17.2096519, 33.0162277, -47.8424149, 46.3998566

Time for backsubstitution: 0.77 seconds

### NS candidates at layer 1
type: B, layer: 1, pos: 12
type: A, layer: 1, pos: 12
type: A, layer: 1, pos: 15
type: B, layer: 1, pos: 15
type: B, layer: 1, pos: 6
type: A, layer: 1, pos: 28
type: B, layer: 1, pos: 28
type: B, layer: 1, pos: 39
type: A, layer: 1, pos: 24
type: B, layer: 1, pos: 24
type: B, layer: 1, pos: 9
type: A, layer: 1, pos: 37
type: B, layer: 1, pos: 48
type: A, layer: 1, pos: 9
type: A, layer: 1, pos: 38
type: B, layer: 1, pos: 8
type: B, layer: 1, pos: 37
type: A, layer: 1, pos: 8
type: A, layer: 1, pos: 6
type: B, layer: 1, pos: 38
type: A, layer: 1, pos: 20
type: A, layer: 1, pos: 48
type: B, layer: 1, pos: 26
type: A, layer: 1, pos: 26
type: A, layer: 1, pos: 39
type: B, layer: 1, pos: 20
type: B, layer: 1, pos: 31
type: A, layer: 1, pos: 41
type: B, layer: 1, pos: 41
type: A, layer: 1, pos: 0
type: A, layer: 1, pos: 5
type: B, layer: 1, pos: 5
type: A, layer: 1, pos: 31
type: B, layer: 1, pos: 0

Time for candidate selection: 0.06 seconds

### Candidate
type: B, layer: 1, pos: 12

### Candidate
type: A, layer: 1, pos: 12

### Candidate
type: A, layer: 1, pos: 15

### Candidate
type: B, layer: 1, pos: 15

### Candidate
type: B, layer: 1, pos: 6

### Candidate
type: A, layer: 1, pos: 28

### Candidate
type: B, layer: 1, pos: 28

### Candidate
type: B, layer: 1, pos: 39

## Relational analysis of NS_B1_A1_A1_A1_B2_B2_B1
Optimization infeasible because this subproblem isn't reachable.

### Candidate
type: A, layer: 1, pos: 24

## Relational analysis of NS_B1_A1_A1_A1_B2_B2_A1
Optimization infeasible because this subproblem isn't reachable.

### Candidate
type: B, layer: 1, pos: 24

### Candidate
type: B, layer: 1, pos: 9

## Relational analysis of NS_B1_A1_A1_A1_B2_B2_B1

### Relational analysis result of NS_B1_A1_A1_A1_B2_B2_B1
Status: Status.UNKNOWN
Output dim: 0, lower bound: -20.9106196, upper bound: 20.9286097
time: 0.53 seconds

## Relational analysis of NS_B1_A1_A1_A1_B2_B2_B2

### Relational analysis result of NS_B1_A1_A1_A1_B2_B2_B2
Status: Status.UNKNOWN
Output dim: 0, lower bound: -20.9222198, upper bound: 20.9359225
time: 0.59 seconds

## BFS NS instance: NS_B1_A1_A1_A2_B1_B1

### Backsubstitution after applying NS history:
0: -6.1871290, 17.1735039, -6.0846200, 16.9946613, -23.1817894, 23.2581234
1: -15.4273548, 26.2602158, -15.3345718, 25.8991718, -41.3265228, 41.5947762
2: -10.7745504, 23.8967419, -10.6436501, 23.5850830, -34.3596306, 34.5403862
3: -16.6525230, 29.0933285, -16.4320755, 28.7094727, -45.3619919, 45.5254059
4: -15.2049751, 29.4977074, -14.8261871, 29.1902122, -44.3951836, 44.3238945

Time for backsubstitution: 0.77 seconds

### NS candidates at layer 1
type: A, layer: 1, pos: 12
type: B, layer: 1, pos: 12
type: A, layer: 1, pos: 15
type: B, layer: 1, pos: 15
type: B, layer: 1, pos: 28
type: A, layer: 1, pos: 6
type: A, layer: 1, pos: 28
type: B, layer: 1, pos: 24
type: A, layer: 1, pos: 39
type: B, layer: 1, pos: 6
type: A, layer: 1, pos: 9
type: A, layer: 1, pos: 24
type: B, layer: 1, pos: 37
type: B, layer: 1, pos: 9
type: A, layer: 1, pos: 48
type: A, layer: 1, pos: 37
type: A, layer: 1, pos: 8
type: B, layer: 1, pos: 8
type: B, layer: 1, pos: 38
type: A, layer: 1, pos: 38
type: B, layer: 1, pos: 20
type: B, layer: 1, pos: 39
type: B, layer: 1, pos: 48
type: A, layer: 1, pos: 26
type: B, layer: 1, pos: 26
type: A, layer: 1, pos: 20
type: A, layer: 1, pos: 41
type: B, layer: 1, pos: 41
type: B, layer: 1, pos: 0
type: B, layer: 1, pos: 5
type: A, layer: 1, pos: 31
type: A, layer: 1, pos: 0
type: A, layer: 1, pos: 5
type: B, layer: 1, pos: 31

Time for candidate selection: 0.06 seconds

### Candidate
type: A, layer: 1, pos: 12

### Candidate
type: B, layer: 1, pos: 12

### Candidate
type: A, layer: 1, pos: 15

### Candidate
type: B, layer: 1, pos: 15

### Candidate
type: B, layer: 1, pos: 28

### Candidate
type: A, layer: 1, pos: 6

### Candidate
type: A, layer: 1, pos: 28

### Candidate
type: B, layer: 1, pos: 24

## Relational analysis of NS_B1_A1_A1_A2_B1_B1_B1
Optimization infeasible because this subproblem isn't reachable.

### Candidate
type: A, layer: 1, pos: 39

## Relational analysis of NS_B1_A1_A1_A2_B1_B1_A1
Optimization infeasible because this subproblem isn't reachable.

### Candidate
type: B, layer: 1, pos: 6

### Candidate
type: A, layer: 1, pos: 9

## Relational analysis of NS_B1_A1_A1_A2_B1_B1_A1

### Relational analysis result of NS_B1_A1_A1_A2_B1_B1_A1
Status: Status.UNKNOWN
Output dim: 0, lower bound: -20.9355777, upper bound: 20.9220791
time: 0.65 seconds

## Relational analysis of NS_B1_A1_A1_A2_B1_B1_A2

### Relational analysis result of NS_B1_A1_A1_A2_B1_B1_A2
Status: Status.UNKNOWN
Output dim: 0, lower bound: -20.9451984, upper bound: 20.9343038
time: 0.61 seconds

## BFS NS instance: NS_B1_A1_A1_A2_B1_B2

### Backsubstitution after applying NS history:
0: -6.1871290, 17.1735039, -6.7040877, 18.7079010, -24.8950272, 23.8775921
1: -15.4273548, 26.2602158, -16.8718414, 28.4157085, -43.8430634, 43.1320496
2: -10.7745504, 23.8967419, -11.7658329, 25.8589211, -36.6334686, 35.6625710
3: -16.6525230, 29.0933285, -18.1079674, 31.5177822, -48.1703033, 47.2012939
4: -15.2049751, 29.4977074, -16.3970699, 32.0241547, -47.2291298, 45.8947754

Time for backsubstitution: 0.78 seconds

### NS candidates at layer 1
type: B, layer: 1, pos: 12
type: A, layer: 1, pos: 12
type: A, layer: 1, pos: 15
type: B, layer: 1, pos: 15
type: B, layer: 1, pos: 28
type: B, layer: 1, pos: 6
type: B, layer: 1, pos: 24
type: A, layer: 1, pos: 28
type: A, layer: 1, pos: 6
type: A, layer: 1, pos: 9
type: B, layer: 1, pos: 9
type: A, layer: 1, pos: 39
type: A, layer: 1, pos: 48
type: B, layer: 1, pos: 37
type: A, layer: 1, pos: 24
type: A, layer: 1, pos: 37
type: A, layer: 1, pos: 8
type: A, layer: 1, pos: 38
type: B, layer: 1, pos: 8
type: B, layer: 1, pos: 38
type: B, layer: 1, pos: 39
type: B, layer: 1, pos: 20
type: A, layer: 1, pos: 26
type: B, layer: 1, pos: 26
type: B, layer: 1, pos: 48
type: A, layer: 1, pos: 20
type: B, layer: 1, pos: 5
type: B, layer: 1, pos: 0
type: A, layer: 1, pos: 41
type: B, layer: 1, pos: 41
type: A, layer: 1, pos: 0
type: A, layer: 1, pos: 31
type: B, layer: 1, pos: 31
type: A, layer: 1, pos: 5

Time for candidate selection: 0.06 seconds

### Candidate
type: B, layer: 1, pos: 12

### Candidate
type: A, layer: 1, pos: 12

### Candidate
type: A, layer: 1, pos: 15

### Candidate
type: B, layer: 1, pos: 15

### Candidate
type: B, layer: 1, pos: 28

### Candidate
type: B, layer: 1, pos: 6

### Candidate
type: B, layer: 1, pos: 24

## Relational analysis of NS_B1_A1_A1_A2_B1_B2_B1
Optimization infeasible because this subproblem isn't reachable.

### Candidate
type: A, layer: 1, pos: 28

### Candidate
type: A, layer: 1, pos: 6

### Candidate
type: A, layer: 1, pos: 9

## Relational analysis of NS_B1_A1_A1_A2_B1_B2_A1

### Relational analysis result of NS_B1_A1_A1_A2_B1_B2_A1
Status: Status.UNKNOWN
Output dim: 0, lower bound: -20.9355777, upper bound: 20.9220791
time: 0.59 seconds

## Relational analysis of NS_B1_A1_A1_A2_B1_B2_A2

### Relational analysis result of NS_B1_A1_A1_A2_B1_B2_A2
Status: Status.UNKNOWN
Output dim: 0, lower bound: -20.9451984, upper bound: 20.9343038
time: 0.57 seconds

## BFS NS instance: NS_B1_A1_A1_A2_B2_B1

### Backsubstitution after applying NS history:
0: -6.1871290, 17.1735039, -6.1871290, 17.1735039, -23.3606339, 23.3606339
1: -15.4273548, 26.2602158, -15.4273548, 26.2602158, -41.6875648, 41.6875648
2: -10.7745504, 23.8967419, -10.7745504, 23.8967419, -34.6712875, 34.6712875
3: -16.6525230, 29.0933285, -16.6525230, 29.0933285, -45.7458496, 45.7458496
4: -15.2049751, 29.4977074, -15.2049751, 29.4977074, -44.7026825, 44.7026825

Time for backsubstitution: 0.78 seconds

### NS candidates at layer 1
type: B, layer: 1, pos: 12
type: A, layer: 1, pos: 12
type: B, layer: 1, pos: 15
type: A, layer: 1, pos: 15
type: A, layer: 1, pos: 28
type: B, layer: 1, pos: 28
type: A, layer: 1, pos: 6
type: B, layer: 1, pos: 6
type: B, layer: 1, pos: 24
type: A, layer: 1, pos: 24
type: B, layer: 1, pos: 9
type: A, layer: 1, pos: 9
type: B, layer: 1, pos: 37
type: A, layer: 1, pos: 37
type: A, layer: 1, pos: 8
type: B, layer: 1, pos: 8
type: A, layer: 1, pos: 38
type: B, layer: 1, pos: 38
type: B, layer: 1, pos: 48
type: A, layer: 1, pos: 48
type: A, layer: 1, pos: 39
type: B, layer: 1, pos: 39
type: B, layer: 1, pos: 20
type: A, layer: 1, pos: 20
type: A, layer: 1, pos: 26
type: B, layer: 1, pos: 26
type: A, layer: 1, pos: 41
type: B, layer: 1, pos: 41
type: A, layer: 1, pos: 5
type: B, layer: 1, pos: 5
type: B, layer: 1, pos: 0
type: A, layer: 1, pos: 0
type: A, layer: 1, pos: 31
type: B, layer: 1, pos: 31

Time for candidate selection: 0.06 seconds

### Candidate
type: B, layer: 1, pos: 12

### Candidate
type: A, layer: 1, pos: 12

### Candidate
type: B, layer: 1, pos: 15

### Candidate
type: A, layer: 1, pos: 15

### Candidate
type: A, layer: 1, pos: 28

## Relational analysis of NS_B1_A1_A1_A2_B2_B1_A1
Optimization infeasible because this subproblem isn't reachable.

### Candidate
type: B, layer: 1, pos: 28

### Candidate
type: A, layer: 1, pos: 6

## Relational analysis of NS_B1_A1_A1_A2_B2_B1_A1

### Relational analysis result of NS_B1_A1_A1_A2_B2_B1_A1
Status: Status.VERIFIED
Output dim: 0, lower bound: -20.7814451, upper bound: 20.7823533
time: 0.57 seconds

## Relational analysis of NS_B1_A1_A1_A2_B2_B1_A2

### Relational analysis result of NS_B1_A1_A1_A2_B2_B1_A2
Status: Status.VERIFIED
Output dim: 0, lower bound: -20.7814451, upper bound: 20.7992035
time: 0.67 seconds

## BFS NS instance: NS_B1_A1_A1_A2_B2_B2

### Backsubstitution after applying NS history:
0: -6.1871290, 17.1735039, -6.9751034, 19.2829285, -25.4700565, 24.1486053
1: -15.4273548, 26.2602158, -17.3758297, 29.4068832, -44.8342361, 43.6360321
2: -10.7745504, 23.8967419, -12.1740484, 26.7304134, -37.5049553, 36.0707893
3: -16.6525230, 29.0933285, -18.7746525, 32.6108894, -49.2634087, 47.8679810
4: -15.2049751, 29.4977074, -17.2096519, 33.0162277, -48.2212029, 46.7073593

Time for backsubstitution: 0.78 seconds

### NS candidates at layer 1
type: B, layer: 1, pos: 12
type: A, layer: 1, pos: 12
type: A, layer: 1, pos: 15
type: B, layer: 1, pos: 15
type: B, layer: 1, pos: 6
type: B, layer: 1, pos: 28
type: A, layer: 1, pos: 28
type: B, layer: 1, pos: 24
type: B, layer: 1, pos: 9
type: A, layer: 1, pos: 24
type: A, layer: 1, pos: 9
type: A, layer: 1, pos: 37
type: A, layer: 1, pos: 6
type: B, layer: 1, pos: 39
type: A, layer: 1, pos: 8
type: B, layer: 1, pos: 37
type: B, layer: 1, pos: 8
type: A, layer: 1, pos: 38
type: B, layer: 1, pos: 38
type: B, layer: 1, pos: 48
type: A, layer: 1, pos: 48
type: A, layer: 1, pos: 20
type: A, layer: 1, pos: 39
type: B, layer: 1, pos: 20
type: A, layer: 1, pos: 26
type: B, layer: 1, pos: 26
type: A, layer: 1, pos: 41
type: B, layer: 1, pos: 41
type: B, layer: 1, pos: 5
type: B, layer: 1, pos: 31
type: B, layer: 1, pos: 0
type: A, layer: 1, pos: 31
type: A, layer: 1, pos: 0
type: A, layer: 1, pos: 5

Time for candidate selection: 0.06 seconds

### Candidate
type: B, layer: 1, pos: 12

### Candidate
type: A, layer: 1, pos: 12

### Candidate
type: A, layer: 1, pos: 15

### Candidate
type: B, layer: 1, pos: 15

### Candidate
type: B, layer: 1, pos: 6

## Relational analysis of NS_B1_A1_A1_A2_B2_B2_B1

### Relational analysis result of NS_B1_A1_A1_A2_B2_B2_B1
Status: Status.UNKNOWN
Output dim: 0, lower bound: -20.9097540, upper bound: 20.9164577
time: 0.58 seconds

## Relational analysis of NS_B1_A1_A1_A2_B2_B2_B2

### Relational analysis result of NS_B1_A1_A1_A2_B2_B2_B2
Status: Status.VERIFIED
Output dim: 0, lower bound: -20.7927946, upper bound: 20.7992037
time: 0.62 seconds

## BFS NS instance: NS_B1_A2_A1_B1_B1_A1

### Backsubstitution after applying NS history:
0: -5.6017060, 15.8043280, -5.4689183, 15.4409122, -21.0426178, 21.2732468
1: -14.0179358, 24.3536930, -13.7117033, 23.7303829, -37.7483177, 38.0653954
2: -9.7453976, 22.0511017, -9.5278883, 21.5665436, -31.3119411, 31.5789909
3: -15.0937595, 26.8936596, -14.7623062, 26.2340851, -41.3278427, 41.6559677
4: -13.6622143, 27.2522964, -13.3521547, 26.6765404, -40.3387527, 40.6044464

Time for backsubstitution: 0.78 seconds

### NS candidates at layer 1
type: A, layer: 1, pos: 12
type: B, layer: 1, pos: 12
type: B, layer: 1, pos: 45
type: A, layer: 1, pos: 45
type: A, layer: 1, pos: 28
type: B, layer: 1, pos: 15
type: A, layer: 1, pos: 15
type: A, layer: 1, pos: 24
type: B, layer: 1, pos: 28
type: B, layer: 1, pos: 8
type: B, layer: 1, pos: 24
type: A, layer: 1, pos: 6
type: A, layer: 1, pos: 39
type: B, layer: 1, pos: 9
type: A, layer: 1, pos: 8
type: B, layer: 1, pos: 20
type: B, layer: 1, pos: 48
type: B, layer: 1, pos: 39
type: A, layer: 1, pos: 37
type: B, layer: 1, pos: 37
type: B, layer: 1, pos: 38
type: A, layer: 1, pos: 48
type: B, layer: 1, pos: 26
type: A, layer: 1, pos: 38
type: A, layer: 1, pos: 26
type: A, layer: 1, pos: 20
type: B, layer: 1, pos: 0
type: A, layer: 1, pos: 0
type: B, layer: 1, pos: 41
type: A, layer: 1, pos: 41
type: A, layer: 1, pos: 5
type: A, layer: 1, pos: 31
type: B, layer: 1, pos: 31
type: B, layer: 1, pos: 5

Time for candidate selection: 0.06 seconds

### Candidate
type: A, layer: 1, pos: 12

### Candidate
type: B, layer: 1, pos: 12

### Candidate
type: B, layer: 1, pos: 45

### Candidate
type: A, layer: 1, pos: 45

### Candidate
type: A, layer: 1, pos: 28

### Candidate
type: B, layer: 1, pos: 15

### Candidate
type: A, layer: 1, pos: 15

### Candidate
type: A, layer: 1, pos: 24

### Candidate
type: B, layer: 1, pos: 28

### Candidate
type: B, layer: 1, pos: 8

## Relational analysis of NS_B1_A2_A1_B1_B1_A1_B1

### Relational analysis result of NS_B1_A2_A1_B1_B1_A1_B1
Status: Status.UNKNOWN
Output dim: 0, lower bound: -20.9425656, upper bound: 20.9297386
time: 0.53 seconds

## Relational analysis of NS_B1_A2_A1_B1_B1_A1_B2

### Relational analysis result of NS_B1_A2_A1_B1_B1_A1_B2
Status: Status.UNKNOWN
Output dim: 0, lower bound: -20.9418692, upper bound: 20.9293078
time: 0.59 seconds

## BFS NS instance: NS_B1_A2_A1_B1_B1_A2

### Backsubstitution after applying NS history:
0: -5.5627847, 15.6472750, -5.4245219, 15.3198185, -20.8826027, 21.0717945
1: -13.9816160, 24.0097771, -13.6016264, 23.5484066, -37.5300064, 37.6113968
2: -9.7048798, 21.8011341, -9.4499693, 21.4018536, -31.1067333, 31.2511024
3: -15.0155830, 26.5744400, -14.6443758, 26.0327320, -41.0483170, 41.2188148
4: -13.5180569, 26.9750786, -13.2421446, 26.4746742, -39.9927254, 40.2172165

Time for backsubstitution: 0.79 seconds

### NS candidates at layer 1
type: A, layer: 1, pos: 12
type: B, layer: 1, pos: 12
type: B, layer: 1, pos: 45
type: A, layer: 1, pos: 28
type: A, layer: 1, pos: 45
type: B, layer: 1, pos: 15
type: A, layer: 1, pos: 15
type: A, layer: 1, pos: 24
type: B, layer: 1, pos: 28
type: B, layer: 1, pos: 8
type: A, layer: 1, pos: 6
type: A, layer: 1, pos: 39
type: B, layer: 1, pos: 24
type: A, layer: 1, pos: 8
type: B, layer: 1, pos: 48
type: B, layer: 1, pos: 20
type: B, layer: 1, pos: 39
type: A, layer: 1, pos: 37
type: B, layer: 1, pos: 37
type: B, layer: 1, pos: 38
type: B, layer: 1, pos: 9
type: A, layer: 1, pos: 48
type: A, layer: 1, pos: 20
type: B, layer: 1, pos: 26
type: A, layer: 1, pos: 26
type: A, layer: 1, pos: 38
type: A, layer: 1, pos: 0
type: B, layer: 1, pos: 0
type: B, layer: 1, pos: 41
type: A, layer: 1, pos: 41
type: A, layer: 1, pos: 5
type: A, layer: 1, pos: 31
type: B, layer: 1, pos: 31
type: B, layer: 1, pos: 5

Time for candidate selection: 0.06 seconds

### Candidate
type: A, layer: 1, pos: 12

### Candidate
type: B, layer: 1, pos: 12

### Candidate
type: B, layer: 1, pos: 45

### Candidate
type: A, layer: 1, pos: 28

### Candidate
type: A, layer: 1, pos: 45

### Candidate
type: B, layer: 1, pos: 15

### Candidate
type: A, layer: 1, pos: 15

### Candidate
type: A, layer: 1, pos: 24

### Candidate
type: B, layer: 1, pos: 28

### Candidate
type: B, layer: 1, pos: 8

## Relational analysis of NS_B1_A2_A1_B1_B1_A2_B1

### Relational analysis result of NS_B1_A2_A1_B1_B1_A2_B1
Status: Status.UNKNOWN
Output dim: 0, lower bound: -20.9601795, upper bound: 20.9570587
time: 0.61 seconds

## Relational analysis of NS_B1_A2_A1_B1_B1_A2_B2

### Relational analysis result of NS_B1_A2_A1_B1_B1_A2_B2
Status: Status.UNKNOWN
Output dim: 0, lower bound: -20.9600750, upper bound: 20.9570448
time: 0.59 seconds

## BFS NS instance: NS_B1_A2_A1_B1_B2_A1

### Backsubstitution after applying NS history:
0: -5.9219766, 16.6530056, -6.1569595, 17.2093010, -23.1312771, 22.8099613
1: -14.8774786, 25.5903778, -15.5148249, 26.2459545, -41.1234322, 41.1052017
2: -10.3299103, 23.1856422, -10.7693729, 23.8826771, -34.2125816, 33.9550171
3: -15.9785748, 28.2802773, -16.6235962, 29.0747719, -45.0533447, 44.9038734
4: -14.4363346, 28.6716270, -15.0020514, 29.5595055, -43.9958344, 43.6736794

Time for backsubstitution: 0.79 seconds

### NS candidates at layer 1
type: A, layer: 1, pos: 12
type: B, layer: 1, pos: 12
type: B, layer: 1, pos: 45
type: B, layer: 1, pos: 15
type: A, layer: 1, pos: 15
type: B, layer: 1, pos: 28
type: A, layer: 1, pos: 45
type: A, layer: 1, pos: 28
type: A, layer: 1, pos: 24
type: B, layer: 1, pos: 24
type: B, layer: 1, pos: 9
type: B, layer: 1, pos: 37
type: B, layer: 1, pos: 48
type: A, layer: 1, pos: 37
type: A, layer: 1, pos: 6
type: A, layer: 1, pos: 39
type: B, layer: 1, pos: 8
type: B, layer: 1, pos: 38
type: A, layer: 1, pos: 8
type: A, layer: 1, pos: 38
type: A, layer: 1, pos: 48
type: B, layer: 1, pos: 39
type: B, layer: 1, pos: 26
type: A, layer: 1, pos: 26
type: B, layer: 1, pos: 20
type: A, layer: 1, pos: 20
type: A, layer: 1, pos: 41
type: B, layer: 1, pos: 41
type: B, layer: 1, pos: 0
type: A, layer: 1, pos: 5
type: B, layer: 1, pos: 5
type: A, layer: 1, pos: 0
type: A, layer: 1, pos: 31
type: B, layer: 1, pos: 31

Time for candidate selection: 0.06 seconds

### Candidate
type: A, layer: 1, pos: 12

### Candidate
type: B, layer: 1, pos: 12

### Candidate
type: B, layer: 1, pos: 45

### Candidate
type: B, layer: 1, pos: 15

### Candidate
type: A, layer: 1, pos: 15

### Candidate
type: B, layer: 1, pos: 28

### Candidate
type: A, layer: 1, pos: 45

### Candidate
type: A, layer: 1, pos: 28

### Candidate
type: A, layer: 1, pos: 24

### Candidate
type: B, layer: 1, pos: 24

### Candidate
type: B, layer: 1, pos: 9

## Relational analysis of NS_B1_A2_A1_B1_B2_A1_B1

### Relational analysis result of NS_B1_A2_A1_B1_B2_A1_B1
Status: Status.UNKNOWN
Output dim: 0, lower bound: -20.9304753, upper bound: 20.9297088
time: 0.57 seconds

## Relational analysis of NS_B1_A2_A1_B1_B2_A1_B2

### Relational analysis result of NS_B1_A2_A1_B1_B2_A1_B2
Status: Status.UNKNOWN
Output dim: 0, lower bound: -20.9304753, upper bound: 20.9441063
time: 0.67 seconds

## BFS NS instance: NS_B1_A2_A1_B1_B2_A2

### Backsubstitution after applying NS history:
0: -5.8575258, 16.4333668, -6.1058140, 17.0732155, -22.9307404, 22.5391808
1: -14.7730999, 25.1493797, -15.3895569, 26.0406761, -40.8137703, 40.5389366
2: -10.2441244, 22.8457184, -10.6804934, 23.6973419, -33.9414673, 33.5262108
3: -15.8305902, 27.8522625, -16.4885902, 28.8462791, -44.6768608, 44.3408508
4: -14.2308779, 28.2878342, -14.8757105, 29.3321552, -43.5630341, 43.1635437

Time for backsubstitution: 0.79 seconds

### NS candidates at layer 1
type: A, layer: 1, pos: 12
type: B, layer: 1, pos: 12
type: B, layer: 1, pos: 45
type: A, layer: 1, pos: 45
type: B, layer: 1, pos: 15
type: B, layer: 1, pos: 28
type: A, layer: 1, pos: 15
type: A, layer: 1, pos: 28
type: A, layer: 1, pos: 24
type: B, layer: 1, pos: 24
type: A, layer: 1, pos: 39
type: A, layer: 1, pos: 6
type: B, layer: 1, pos: 48
type: B, layer: 1, pos: 8
type: B, layer: 1, pos: 37
type: A, layer: 1, pos: 37
type: B, layer: 1, pos: 39
type: B, layer: 1, pos: 38
type: A, layer: 1, pos: 8
type: A, layer: 1, pos: 38
type: B, layer: 1, pos: 9
type: A, layer: 1, pos: 48
type: B, layer: 1, pos: 26
type: A, layer: 1, pos: 26
type: B, layer: 1, pos: 20
type: A, layer: 1, pos: 20
type: A, layer: 1, pos: 41
type: B, layer: 1, pos: 41
type: A, layer: 1, pos: 5
type: B, layer: 1, pos: 0
type: A, layer: 1, pos: 31
type: A, layer: 1, pos: 0
type: B, layer: 1, pos: 31
type: B, layer: 1, pos: 5

Time for candidate selection: 0.06 seconds

### Candidate
type: A, layer: 1, pos: 12

### Candidate
type: B, layer: 1, pos: 12

### Candidate
type: B, layer: 1, pos: 45

### Candidate
type: A, layer: 1, pos: 45

### Candidate
type: B, layer: 1, pos: 15

### Candidate
type: B, layer: 1, pos: 28

### Candidate
type: A, layer: 1, pos: 15

### Candidate
type: A, layer: 1, pos: 28

### Candidate
type: A, layer: 1, pos: 24

### Candidate
type: B, layer: 1, pos: 24

### Candidate
type: A, layer: 1, pos: 39

## Relational analysis of NS_B1_A2_A1_B1_B2_A2_A1
Optimization infeasible because this subproblem isn't reachable.

### Candidate
type: A, layer: 1, pos: 6

## Relational analysis of NS_B1_A2_A1_B1_B2_A2_A1
Optimization infeasible because this subproblem isn't reachable.

### Candidate
type: B, layer: 1, pos: 48

## Relational analysis of NS_B1_A2_A1_B1_B2_A2_B1
Optimization infeasible because this subproblem isn't reachable.

### Candidate
type: B, layer: 1, pos: 8

## Relational analysis of NS_B1_A2_A1_B1_B2_A2_B1

### Relational analysis result of NS_B1_A2_A1_B1_B2_A2_B1
Status: Status.UNKNOWN
Output dim: 0, lower bound: -20.9632714, upper bound: 20.9621042
time: 0.73 seconds

## Relational analysis of NS_B1_A2_A1_B1_B2_A2_B2

### Relational analysis result of NS_B1_A2_A1_B1_B2_A2_B2
Status: Status.UNKNOWN
Output dim: 0, lower bound: -20.9600750, upper bound: 20.9633596
time: 0.74 seconds

## BFS NS instance: NS_B1_A2_A1_B2_B1_A1

### Backsubstitution after applying NS history:
0: -5.6017060, 15.8043280, -5.5659547, 15.6098299, -21.2115364, 21.3702812
1: -14.0179358, 24.3536930, -13.8243294, 24.1012421, -38.1191750, 38.1780243
2: -9.7453976, 22.0511017, -9.6580257, 21.8680553, -31.6134529, 31.7091274
3: -15.0937595, 26.8936596, -14.9626017, 26.6162720, -41.7100296, 41.8562622
4: -13.6622143, 27.2522964, -13.7018490, 26.9766998, -40.6389160, 40.9541473

Time for backsubstitution: 0.80 seconds

### NS candidates at layer 1
type: A, layer: 1, pos: 12
type: B, layer: 1, pos: 12
type: B, layer: 1, pos: 45
type: B, layer: 1, pos: 15
type: A, layer: 1, pos: 15
type: A, layer: 1, pos: 45
type: B, layer: 1, pos: 8
type: B, layer: 1, pos: 48
type: A, layer: 1, pos: 24
type: B, layer: 1, pos: 9
type: A, layer: 1, pos: 20
type: B, layer: 1, pos: 39
type: B, layer: 1, pos: 20
type: A, layer: 1, pos: 6
type: A, layer: 1, pos: 8
type: B, layer: 1, pos: 38
type: A, layer: 1, pos: 37
type: A, layer: 1, pos: 28
type: B, layer: 1, pos: 24
type: B, layer: 1, pos: 37
type: A, layer: 1, pos: 38
type: B, layer: 1, pos: 28
type: A, layer: 1, pos: 39
type: A, layer: 1, pos: 48
type: A, layer: 1, pos: 0
type: B, layer: 1, pos: 0
type: B, layer: 1, pos: 26
type: A, layer: 1, pos: 5
type: B, layer: 1, pos: 31
type: A, layer: 1, pos: 26
type: A, layer: 1, pos: 41
type: B, layer: 1, pos: 41
type: A, layer: 1, pos: 31
type: B, layer: 1, pos: 5

Time for candidate selection: 0.06 seconds

### Candidate
type: A, layer: 1, pos: 12

### Candidate
type: B, layer: 1, pos: 12

### Candidate
type: B, layer: 1, pos: 45

### Candidate
type: B, layer: 1, pos: 15

### Candidate
type: A, layer: 1, pos: 15

### Candidate
type: A, layer: 1, pos: 45

### Candidate
type: B, layer: 1, pos: 8

## Relational analysis of NS_B1_A2_A1_B2_B1_A1_B1

### Relational analysis result of NS_B1_A2_A1_B2_B1_A1_B1
Status: Status.UNKNOWN
Output dim: 0, lower bound: -20.9023541, upper bound: 20.9074667
time: 0.62 seconds

## Relational analysis of NS_B1_A2_A1_B2_B1_A1_B2

### Relational analysis result of NS_B1_A2_A1_B2_B1_A1_B2
Status: Status.UNKNOWN
Output dim: 0, lower bound: -20.9023299, upper bound: 20.9075448
time: 0.58 seconds

## BFS NS instance: NS_B1_A2_A1_B2_B1_A2

### Backsubstitution after applying NS history:
0: -5.5627847, 15.6472750, -5.5195541, 15.4850712, -21.0478554, 21.1668243
1: -13.9816160, 24.0097771, -13.7119064, 23.9133072, -37.8949127, 37.7216759
2: -9.7048798, 21.8011341, -9.5788641, 21.6971607, -31.4020405, 31.3799973
3: -15.0155830, 26.5744400, -14.8412724, 26.4095211, -41.4251022, 41.4157104
4: -13.5180569, 26.9750786, -13.5872326, 26.7685089, -40.2865601, 40.5623093

Time for backsubstitution: 0.79 seconds

### NS candidates at layer 1
type: A, layer: 1, pos: 12
type: B, layer: 1, pos: 12
type: B, layer: 1, pos: 45
type: B, layer: 1, pos: 15
type: A, layer: 1, pos: 45
type: A, layer: 1, pos: 15
type: A, layer: 1, pos: 20
type: B, layer: 1, pos: 8
type: B, layer: 1, pos: 48
type: A, layer: 1, pos: 24
type: B, layer: 1, pos: 20
type: A, layer: 1, pos: 8
type: A, layer: 1, pos: 6
type: B, layer: 1, pos: 9
type: B, layer: 1, pos: 38
type: A, layer: 1, pos: 37
type: B, layer: 1, pos: 39
type: B, layer: 1, pos: 24
type: B, layer: 1, pos: 37
type: A, layer: 1, pos: 28
type: A, layer: 1, pos: 38
type: B, layer: 1, pos: 28
type: A, layer: 1, pos: 39
type: A, layer: 1, pos: 48
type: A, layer: 1, pos: 0
type: A, layer: 1, pos: 5
type: B, layer: 1, pos: 31
type: B, layer: 1, pos: 0
type: A, layer: 1, pos: 31
type: B, layer: 1, pos: 26
type: A, layer: 1, pos: 41
type: B, layer: 1, pos: 41
type: A, layer: 1, pos: 26
type: B, layer: 1, pos: 5

Time for candidate selection: 0.06 seconds

### Candidate
type: A, layer: 1, pos: 12

### Candidate
type: B, layer: 1, pos: 12

### Candidate
type: B, layer: 1, pos: 45

### Candidate
type: B, layer: 1, pos: 15

### Candidate
type: A, layer: 1, pos: 45

### Candidate
type: A, layer: 1, pos: 15

### Candidate
type: A, layer: 1, pos: 20

## Relational analysis of NS_B1_A2_A1_B2_B1_A2_A1

### Relational analysis result of NS_B1_A2_A1_B2_B1_A2_A1
Status: Status.UNKNOWN
Output dim: 0, lower bound: -20.9174305, upper bound: 20.9285825
time: 0.66 seconds

## Relational analysis of NS_B1_A2_A1_B2_B1_A2_A2

### Relational analysis result of NS_B1_A2_A1_B2_B1_A2_A2
Status: Status.UNKNOWN
Output dim: 0, lower bound: -20.9262643, upper bound: 20.9387501
time: 1.07 seconds

## BFS NS instance: NS_B1_A2_A1_B2_B2_B1

### Backsubstitution after applying NS history:
0: -6.0017405, 16.8577003, -6.1855474, 17.1943378, -23.1960773, 23.0432472
1: -15.0798769, 25.8949661, -15.4049635, 26.3551464, -41.4350243, 41.2999306
2: -10.4755583, 23.4678841, -10.7583294, 23.9353752, -34.4109306, 34.2262115
3: -16.1910629, 28.6209011, -16.6418781, 29.1695862, -45.3606491, 45.2627792
4: -14.6383791, 29.0156803, -15.2073355, 29.5463696, -44.1847496, 44.2230148

Time for backsubstitution: 0.80 seconds

### NS candidates at layer 1
type: A, layer: 1, pos: 12
type: B, layer: 1, pos: 12
type: B, layer: 1, pos: 45
type: B, layer: 1, pos: 15
type: A, layer: 1, pos: 15
type: A, layer: 1, pos: 45
type: B, layer: 1, pos: 28
type: A, layer: 1, pos: 28
type: A, layer: 1, pos: 24
type: B, layer: 1, pos: 48
type: B, layer: 1, pos: 24
type: B, layer: 1, pos: 39
type: A, layer: 1, pos: 9
type: A, layer: 1, pos: 37
type: B, layer: 1, pos: 37
type: B, layer: 1, pos: 8
type: B, layer: 1, pos: 38
type: A, layer: 1, pos: 6
type: A, layer: 1, pos: 38
type: A, layer: 1, pos: 8
type: A, layer: 1, pos: 39
type: A, layer: 1, pos: 48
type: B, layer: 1, pos: 26
type: A, layer: 1, pos: 26
type: A, layer: 1, pos: 20
type: B, layer: 1, pos: 20
type: A, layer: 1, pos: 5
type: B, layer: 1, pos: 41
type: A, layer: 1, pos: 41
type: A, layer: 1, pos: 0
type: B, layer: 1, pos: 0
type: B, layer: 1, pos: 31
type: A, layer: 1, pos: 31
type: B, layer: 1, pos: 5

Time for candidate selection: 0.07 seconds

### Candidate
type: A, layer: 1, pos: 12

### Candidate
type: B, layer: 1, pos: 12

### Candidate
type: B, layer: 1, pos: 45

### Candidate
type: B, layer: 1, pos: 15

### Candidate
type: A, layer: 1, pos: 15

### Candidate
type: A, layer: 1, pos: 45

### Candidate
type: B, layer: 1, pos: 28

### Candidate
type: A, layer: 1, pos: 28

### Candidate
type: A, layer: 1, pos: 24

### Candidate
type: B, layer: 1, pos: 48

### Candidate
type: B, layer: 1, pos: 24

### Candidate
type: B, layer: 1, pos: 39

## Relational analysis of NS_B1_A2_A1_B2_B2_B1_B1
Optimization infeasible because this subproblem isn't reachable.

### Candidate
type: A, layer: 1, pos: 9

## Relational analysis of NS_B1_A2_A1_B2_B2_B1_A1

### Relational analysis result of NS_B1_A2_A1_B2_B2_B1_A1
Status: Status.UNKNOWN
Output dim: 0, lower bound: -20.9023832, upper bound: 20.9267171
time: 0.66 seconds

## Relational analysis of NS_B1_A2_A1_B2_B2_B1_A2

### Relational analysis result of NS_B1_A2_A1_B2_B2_B1_A2
Status: Status.UNKNOWN
Output dim: 0, lower bound: -20.9023832, upper bound: 20.9267171
time: 0.68 seconds

## BFS NS instance: NS_B1_A2_A1_B2_B2_B2

### Backsubstitution after applying NS history:
0: -5.9480605, 16.7152119, -6.1548481, 17.0873947, -23.0354519, 22.8700562
1: -14.9457874, 25.6828289, -15.4049559, 26.0736637, -41.0194511, 41.0877800
2: -10.3815269, 23.2739506, -10.7387972, 23.7534828, -34.1350098, 34.0127487
3: -16.0481339, 28.3843365, -16.5831699, 28.8956184, -44.9437523, 44.9675026
4: -14.5056696, 28.7776871, -15.0797491, 29.3512840, -43.8569527, 43.8574219

Time for backsubstitution: 0.79 seconds

### NS candidates at layer 1
type: A, layer: 1, pos: 12
type: B, layer: 1, pos: 12
type: B, layer: 1, pos: 45
type: B, layer: 1, pos: 15
type: A, layer: 1, pos: 15
type: B, layer: 1, pos: 28
type: A, layer: 1, pos: 45
type: A, layer: 1, pos: 28
type: A, layer: 1, pos: 24
type: B, layer: 1, pos: 48
type: B, layer: 1, pos: 24
type: B, layer: 1, pos: 39
type: A, layer: 1, pos: 37
type: B, layer: 1, pos: 37
type: B, layer: 1, pos: 8
type: B, layer: 1, pos: 38
type: A, layer: 1, pos: 6
type: A, layer: 1, pos: 38
type: A, layer: 1, pos: 8
type: A, layer: 1, pos: 9
type: A, layer: 1, pos: 39
type: A, layer: 1, pos: 48
type: B, layer: 1, pos: 26
type: A, layer: 1, pos: 26
type: A, layer: 1, pos: 20
type: B, layer: 1, pos: 20
type: B, layer: 1, pos: 41
type: A, layer: 1, pos: 41
type: A, layer: 1, pos: 5
type: A, layer: 1, pos: 0
type: B, layer: 1, pos: 0
type: A, layer: 1, pos: 31
type: B, layer: 1, pos: 31
type: B, layer: 1, pos: 5

Time for candidate selection: 0.06 seconds

### Candidate
type: A, layer: 1, pos: 12

### Candidate
type: B, layer: 1, pos: 12

### Candidate
type: B, layer: 1, pos: 45

### Candidate
type: B, layer: 1, pos: 15

### Candidate
type: A, layer: 1, pos: 15

### Candidate
type: B, layer: 1, pos: 28

### Candidate
type: A, layer: 1, pos: 45

### Candidate
type: A, layer: 1, pos: 28

### Candidate
type: A, layer: 1, pos: 24

### Candidate
type: B, layer: 1, pos: 48

### Candidate
type: B, layer: 1, pos: 24

### Candidate
type: B, layer: 1, pos: 39

## Relational analysis of NS_B1_A2_A1_B2_B2_B2_B1
Optimization infeasible because this subproblem isn't reachable.

### Candidate
type: A, layer: 1, pos: 37

## Relational analysis of NS_B1_A2_A1_B2_B2_B2_A1

### Relational analysis result of NS_B1_A2_A1_B2_B2_B2_A1
Status: Status.UNKNOWN
Output dim: 0, lower bound: -20.9442696, upper bound: 20.9457414
time: 0.67 seconds

## Relational analysis of NS_B1_A2_A1_B2_B2_B2_A2
Optimization infeasible because this subproblem isn't reachable.

### Candidate
type: B, layer: 1, pos: 37

### Candidate
type: B, layer: 1, pos: 8

## Relational analysis of NS_B1_A2_A1_B2_B2_B2_B1

### Relational analysis result of NS_B1_A2_A1_B2_B2_B2_B1
Status: Status.UNKNOWN
Output dim: 0, lower bound: -20.9422678, upper bound: 20.9484328
time: 0.59 seconds

## Relational analysis of NS_B1_A2_A1_B2_B2_B2_B2

### Relational analysis result of NS_B1_A2_A1_B2_B2_B2_B2
Status: Status.UNKNOWN
Output dim: 0, lower bound: -20.9476371, upper bound: 20.9513798
time: 0.61 seconds

## BFS NS instance: NS_B2_B1_A1_A2_A1_B1

### Backsubstitution after applying NS history:
0: -5.7251363, 16.0905418, -5.6506472, 15.8846350, -21.6097717, 21.7411880
1: -14.3359890, 24.7802849, -14.0206594, 24.5762863, -38.9122772, 38.8009415
2: -9.9683666, 22.4377785, -9.7986345, 22.2425442, -32.2109108, 32.2364082
3: -15.4297438, 27.3755054, -15.1872730, 27.1004562, -42.5301971, 42.5627785
4: -13.9837160, 27.7255211, -13.9226294, 27.4241123, -41.4078293, 41.6481476

Time for backsubstitution: 0.80 seconds

### NS candidates at layer 1
type: A, layer: 1, pos: 1
type: B, layer: 1, pos: 45
type: A, layer: 1, pos: 45
type: B, layer: 1, pos: 15
type: B, layer: 1, pos: 47
type: B, layer: 1, pos: 38
type: B, layer: 1, pos: 48
type: A, layer: 1, pos: 38
type: B, layer: 1, pos: 9
type: A, layer: 1, pos: 9
type: A, layer: 1, pos: 28
type: A, layer: 1, pos: 39
type: B, layer: 1, pos: 28
type: A, layer: 1, pos: 8
type: B, layer: 1, pos: 20
type: A, layer: 1, pos: 6
type: B, layer: 1, pos: 6
type: A, layer: 1, pos: 37
type: B, layer: 1, pos: 8
type: B, layer: 1, pos: 37
type: A, layer: 1, pos: 20
type: B, layer: 1, pos: 31
type: A, layer: 1, pos: 31
type: A, layer: 1, pos: 48
type: B, layer: 1, pos: 24
type: A, layer: 1, pos: 24
type: A, layer: 1, pos: 5
type: B, layer: 1, pos: 26
type: B, layer: 1, pos: 0
type: A, layer: 1, pos: 0
type: B, layer: 1, pos: 41
type: A, layer: 1, pos: 26
type: A, layer: 1, pos: 41
type: B, layer: 1, pos: 5

Time for candidate selection: 0.07 seconds

### Candidate
type: A, layer: 1, pos: 1

### Candidate
type: B, layer: 1, pos: 45

### Candidate
type: A, layer: 1, pos: 45

### Candidate
type: B, layer: 1, pos: 15

### Candidate
type: B, layer: 1, pos: 47

## Relational analysis of NS_B2_B1_A1_A2_A1_B1_B1

### Relational analysis result of NS_B2_B1_A1_A2_A1_B1_B1
Status: Status.UNKNOWN
Output dim: 0, lower bound: -20.9608208, upper bound: 20.9611716
time: 0.93 seconds

## Relational analysis of NS_B2_B1_A1_A2_A1_B1_B2
Optimization infeasible because this subproblem isn't reachable.

### Candidate
type: B, layer: 1, pos: 38

## Relational analysis of NS_B2_B1_A1_A2_A1_B1_B1
Optimization infeasible because this subproblem isn't reachable.

### Candidate
type: B, layer: 1, pos: 48

## Relational analysis of NS_B2_B1_A1_A2_A1_B1_B1
Optimization infeasible because this subproblem isn't reachable.

### Candidate
type: A, layer: 1, pos: 38

### Candidate
type: B, layer: 1, pos: 9

## Relational analysis of NS_B2_B1_A1_A2_A1_B1_B1
Optimization infeasible because this subproblem isn't reachable.

### Candidate
type: A, layer: 1, pos: 9

### Candidate
type: A, layer: 1, pos: 28

## Relational analysis of NS_B2_B1_A1_A2_A1_B1_A1
Optimization infeasible because this subproblem isn't reachable.

### Candidate
type: A, layer: 1, pos: 39

## Relational analysis of NS_B2_B1_A1_A2_A1_B1_A1

### Relational analysis result of NS_B2_B1_A1_A2_A1_B1_A1
Status: Status.UNKNOWN
Output dim: 0, lower bound: -20.9607707, upper bound: 20.9611716
time: 0.59 seconds

## Relational analysis of NS_B2_B1_A1_A2_A1_B1_A2

### Relational analysis result of NS_B2_B1_A1_A2_A1_B1_A2
Status: Status.UNKNOWN
Output dim: 0, lower bound: -20.9607707, upper bound: 20.9611716
time: 0.64 seconds

## BFS NS instance: NS_B2_B1_A1_A2_A1_B2

### Backsubstitution after applying NS history:
0: -5.6542726, 15.9088411, -6.8067040, 19.0791645, -24.7334328, 22.7155457
1: -14.1418114, 24.5143871, -17.1316872, 29.2740116, -43.4158249, 41.6460724
2: -9.8385191, 22.1949253, -11.8529787, 26.5167751, -36.3552933, 34.0479050
3: -15.2351952, 27.0776730, -18.3257275, 32.2822495, -47.5174332, 45.4034004
4: -13.8206072, 27.4217377, -16.5295219, 32.8072777, -46.6278839, 43.9512596

Time for backsubstitution: 0.80 seconds

### NS candidates at layer 1
type: A, layer: 1, pos: 1
type: B, layer: 1, pos: 45
type: A, layer: 1, pos: 39
type: A, layer: 1, pos: 45
type: B, layer: 1, pos: 15
type: A, layer: 1, pos: 28
type: A, layer: 1, pos: 6
type: B, layer: 1, pos: 48
type: B, layer: 1, pos: 6
type: B, layer: 1, pos: 47
type: B, layer: 1, pos: 38
type: B, layer: 1, pos: 28
type: A, layer: 1, pos: 38
type: A, layer: 1, pos: 9
type: B, layer: 1, pos: 9
type: A, layer: 1, pos: 37
type: A, layer: 1, pos: 8
type: B, layer: 1, pos: 20
type: A, layer: 1, pos: 48
type: B, layer: 1, pos: 24
type: B, layer: 1, pos: 37
type: A, layer: 1, pos: 31
type: B, layer: 1, pos: 8
type: B, layer: 1, pos: 31
type: A, layer: 1, pos: 24
type: B, layer: 1, pos: 0
type: A, layer: 1, pos: 20
type: B, layer: 1, pos: 26
type: A, layer: 1, pos: 26
type: A, layer: 1, pos: 0
type: A, layer: 1, pos: 5
type: B, layer: 1, pos: 41
type: A, layer: 1, pos: 41
type: B, layer: 1, pos: 5

Time for candidate selection: 0.07 seconds

### Candidate
type: A, layer: 1, pos: 1

### Candidate
type: B, layer: 1, pos: 45

### Candidate
type: A, layer: 1, pos: 39

## Relational analysis of NS_B2_B1_A1_A2_A1_B2_A1

### Relational analysis result of NS_B2_B1_A1_A2_A1_B2_A1
Status: Status.UNKNOWN
Output dim: 0, lower bound: -20.9609784, upper bound: 20.9620561
time: 0.68 seconds

## Relational analysis of NS_B2_B1_A1_A2_A1_B2_A2

### Relational analysis result of NS_B2_B1_A1_A2_A1_B2_A2
Status: Status.UNKNOWN
Output dim: 0, lower bound: -20.9609784, upper bound: 20.9620561
time: 0.75 seconds

## BFS NS instance: NS_B2_B1_A1_A2_A2_B1

### Backsubstitution after applying NS history:
0: -6.0549908, 16.9044552, -5.7027106, 16.0667610, -22.1217518, 22.6071663
1: -15.0391197, 26.0862732, -14.2820339, 24.7505398, -39.7896576, 40.3683090
2: -10.5142918, 23.6153870, -9.9260283, 22.4121838, -32.9264755, 33.5414162
3: -16.2734241, 28.8037796, -15.3725481, 27.3299770, -43.6034012, 44.1763268
4: -14.9265718, 29.1157589, -13.9278164, 27.6926327, -42.6191978, 43.0435753

Time for backsubstitution: 0.80 seconds

### NS candidates at layer 1
type: A, layer: 1, pos: 1
type: B, layer: 1, pos: 45
type: A, layer: 1, pos: 39
type: A, layer: 1, pos: 45
type: B, layer: 1, pos: 15
type: B, layer: 1, pos: 38
type: A, layer: 1, pos: 38
type: A, layer: 1, pos: 9
type: B, layer: 1, pos: 28
type: A, layer: 1, pos: 48
type: B, layer: 1, pos: 9
type: A, layer: 1, pos: 6
type: B, layer: 1, pos: 39
type: A, layer: 1, pos: 28
type: B, layer: 1, pos: 20
type: B, layer: 1, pos: 37
type: A, layer: 1, pos: 8
type: B, layer: 1, pos: 48
type: A, layer: 1, pos: 37
type: B, layer: 1, pos: 6
type: A, layer: 1, pos: 31
type: B, layer: 1, pos: 8
type: B, layer: 1, pos: 31
type: B, layer: 1, pos: 24
type: A, layer: 1, pos: 20
type: B, layer: 1, pos: 0
type: A, layer: 1, pos: 24
type: A, layer: 1, pos: 26
type: B, layer: 1, pos: 26
type: B, layer: 1, pos: 41
type: A, layer: 1, pos: 41
type: B, layer: 1, pos: 5
type: A, layer: 1, pos: 0
type: A, layer: 1, pos: 5

Time for candidate selection: 0.07 seconds

### Candidate
type: A, layer: 1, pos: 1

### Candidate
type: B, layer: 1, pos: 45

### Candidate
type: A, layer: 1, pos: 39

## Relational analysis of NS_B2_B1_A1_A2_A2_B1_A1

### Relational analysis result of NS_B2_B1_A1_A2_A2_B1_A1
Status: Status.UNKNOWN
Output dim: 0, lower bound: -20.9603972, upper bound: 20.9607285
time: 0.56 seconds

## Relational analysis of NS_B2_B1_A1_A2_A2_B1_A2

### Relational analysis result of NS_B2_B1_A1_A2_A2_B1_A2
Status: Status.UNKNOWN
Output dim: 0, lower bound: -20.9608904, upper bound: 20.9608904
time: 0.90 seconds

## BFS NS instance: NS_B2_B1_A1_A2_A2_B2

### Backsubstitution after applying NS history:
0: -6.0549908, 16.9044552, -5.9576740, 16.6824055, -22.7373962, 22.8621292
1: -15.0391197, 26.0862732, -14.7842646, 25.7619686, -40.8010864, 40.8705292
2: -10.5142918, 23.6153870, -10.3337116, 23.3176594, -33.8319473, 33.9490967
3: -16.2734241, 28.8037796, -16.0070457, 28.4216137, -44.6950378, 44.8108253
4: -14.9265718, 29.1157589, -14.6885023, 28.7500687, -43.6766396, 43.8042603

Time for backsubstitution: 0.81 seconds

### NS candidates at layer 1
type: A, layer: 1, pos: 1
type: B, layer: 1, pos: 45
type: B, layer: 1, pos: 15
type: A, layer: 1, pos: 45
type: A, layer: 1, pos: 39
type: B, layer: 1, pos: 39
type: B, layer: 1, pos: 38
type: A, layer: 1, pos: 8
type: A, layer: 1, pos: 38
type: B, layer: 1, pos: 20
type: B, layer: 1, pos: 8
type: A, layer: 1, pos: 9
type: B, layer: 1, pos: 9
type: B, layer: 1, pos: 37
type: A, layer: 1, pos: 37
type: A, layer: 1, pos: 6
type: A, layer: 1, pos: 20
type: A, layer: 1, pos: 28
type: B, layer: 1, pos: 28
type: B, layer: 1, pos: 48
type: B, layer: 1, pos: 6
type: A, layer: 1, pos: 48
type: A, layer: 1, pos: 31
type: B, layer: 1, pos: 31
type: B, layer: 1, pos: 24
type: A, layer: 1, pos: 24
type: B, layer: 1, pos: 26
type: A, layer: 1, pos: 26
type: B, layer: 1, pos: 0
type: B, layer: 1, pos: 41
type: A, layer: 1, pos: 41
type: A, layer: 1, pos: 5
type: B, layer: 1, pos: 5
type: A, layer: 1, pos: 0

Time for candidate selection: 0.07 seconds

### Candidate
type: A, layer: 1, pos: 1

### Candidate
type: B, layer: 1, pos: 45

### Candidate
type: B, layer: 1, pos: 15

## Relational analysis of NS_B2_B1_A1_A2_A2_B2_B1
Optimization infeasible because this subproblem isn't reachable.

### Candidate
type: A, layer: 1, pos: 45

### Candidate
type: A, layer: 1, pos: 39

## Relational analysis of NS_B2_B1_A1_A2_A2_B2_A1

### Relational analysis result of NS_B2_B1_A1_A2_A2_B2_A1
Status: Status.UNKNOWN
Output dim: 0, lower bound: -20.9603972, upper bound: 20.9607285
time: 1.30 seconds

## Relational analysis of NS_B2_B1_A1_A2_A2_B2_A2

### Relational analysis result of NS_B2_B1_A1_A2_A2_B2_A2
Status: Status.UNKNOWN
Output dim: 0, lower bound: -20.9608904, upper bound: 20.9608904
time: 1.14 seconds

## BFS NS instance: NS_B2_B1_A2_B2_A1_B1

### Backsubstitution after applying NS history:
0: -5.9982681, 16.7135715, -5.5526805, 15.6498661, -21.6481342, 22.2662525
1: -15.0037842, 25.7121830, -13.8711405, 24.1331596, -39.1369438, 39.5833206
2: -10.4593410, 23.2528305, -9.6537743, 21.8439827, -32.3033218, 32.9066048
3: -16.0719185, 28.4230633, -14.9568930, 26.6492329, -42.7211533, 43.3799515
4: -14.4953861, 28.6492424, -13.5780935, 26.9858017, -41.4811859, 42.2273293

Time for backsubstitution: 0.81 seconds

### NS candidates at layer 1
type: A, layer: 1, pos: 1
type: B, layer: 1, pos: 45
type: B, layer: 1, pos: 28
type: A, layer: 1, pos: 45
type: A, layer: 1, pos: 28
type: A, layer: 1, pos: 15
type: A, layer: 1, pos: 39
type: B, layer: 1, pos: 39
type: A, layer: 1, pos: 6
type: B, layer: 1, pos: 38
type: A, layer: 1, pos: 9
type: A, layer: 1, pos: 8
type: A, layer: 1, pos: 38
type: B, layer: 1, pos: 9
type: B, layer: 1, pos: 6
type: A, layer: 1, pos: 48
type: B, layer: 1, pos: 48
type: B, layer: 1, pos: 37
type: B, layer: 1, pos: 8
type: B, layer: 1, pos: 20
type: B, layer: 1, pos: 24
type: A, layer: 1, pos: 37
type: A, layer: 1, pos: 24
type: A, layer: 1, pos: 31
type: B, layer: 1, pos: 26
type: A, layer: 1, pos: 26
type: B, layer: 1, pos: 31
type: A, layer: 1, pos: 20
type: B, layer: 1, pos: 41
type: A, layer: 1, pos: 41
type: B, layer: 1, pos: 5
type: B, layer: 1, pos: 0
type: A, layer: 1, pos: 0
type: A, layer: 1, pos: 5

Time for candidate selection: 0.07 seconds

### Candidate
type: A, layer: 1, pos: 1

### Candidate
type: B, layer: 1, pos: 45

### Candidate
type: B, layer: 1, pos: 28

## Relational analysis of NS_B2_B1_A2_B2_A1_B1_B1
Optimization infeasible because this subproblem isn't reachable.

### Candidate
type: A, layer: 1, pos: 45

### Candidate
type: A, layer: 1, pos: 28

### Candidate
type: A, layer: 1, pos: 15

### Candidate
type: A, layer: 1, pos: 39

## Relational analysis of NS_B2_B1_A2_B2_A1_B1_A1

### Relational analysis result of NS_B2_B1_A2_B2_A1_B1_A1
Status: Status.UNKNOWN
Output dim: 0, lower bound: -20.9575660, upper bound: 20.9601174
time: 0.70 seconds

## Relational analysis of NS_B2_B1_A2_B2_A1_B1_A2

### Relational analysis result of NS_B2_B1_A2_B2_A1_B1_A2
Status: Status.UNKNOWN
Output dim: 0, lower bound: -20.9573158, upper bound: 20.9593464
time: 0.73 seconds

## BFS NS instance: NS_B2_B1_A2_B2_A1_B2

### Backsubstitution after applying NS history:
0: -5.9982681, 16.7135715, -5.8139791, 16.2948456, -22.2931137, 22.5275497
1: -15.0037842, 25.7121830, -14.4053001, 25.1892090, -40.1929893, 40.1174850
2: -10.4593410, 23.2528305, -10.0807743, 22.7908363, -33.2501755, 33.3336029
3: -16.0719185, 28.4230633, -15.6151314, 27.7924881, -43.8644066, 44.0381927
4: -14.4953861, 28.6492424, -14.3490543, 28.0902920, -42.5856781, 42.9982910

Time for backsubstitution: 0.82 seconds

### NS candidates at layer 1
type: A, layer: 1, pos: 1
type: B, layer: 1, pos: 45
type: A, layer: 1, pos: 45
type: A, layer: 1, pos: 15
type: B, layer: 1, pos: 39
type: B, layer: 1, pos: 48
type: B, layer: 1, pos: 38
type: A, layer: 1, pos: 38
type: B, layer: 1, pos: 9
type: A, layer: 1, pos: 28
type: A, layer: 1, pos: 9
type: A, layer: 1, pos: 8
type: A, layer: 1, pos: 6
type: B, layer: 1, pos: 37
type: A, layer: 1, pos: 37
type: B, layer: 1, pos: 6
type: A, layer: 1, pos: 48
type: B, layer: 1, pos: 8
type: B, layer: 1, pos: 20
type: A, layer: 1, pos: 20
type: B, layer: 1, pos: 28
type: A, layer: 1, pos: 31
type: B, layer: 1, pos: 31
type: A, layer: 1, pos: 39
type: A, layer: 1, pos: 24
type: B, layer: 1, pos: 24
type: A, layer: 1, pos: 0
type: A, layer: 1, pos: 5
type: B, layer: 1, pos: 41
type: B, layer: 1, pos: 26
type: A, layer: 1, pos: 41
type: A, layer: 1, pos: 26
type: B, layer: 1, pos: 0
type: B, layer: 1, pos: 5

Time for candidate selection: 0.07 seconds

### Candidate
type: A, layer: 1, pos: 1

### Candidate
type: B, layer: 1, pos: 45

### Candidate
type: A, layer: 1, pos: 45

### Candidate
type: A, layer: 1, pos: 15

### Candidate
type: B, layer: 1, pos: 39

## Relational analysis of NS_B2_B1_A2_B2_A1_B2_B1

### Relational analysis result of NS_B2_B1_A2_B2_A1_B2_B1
Status: Status.UNKNOWN
Output dim: 0, lower bound: -20.9563722, upper bound: 20.9585533
time: 0.65 seconds

## Relational analysis of NS_B2_B1_A2_B2_A1_B2_B2
Optimization infeasible because this subproblem isn't reachable.

### Candidate
type: B, layer: 1, pos: 48

## Relational analysis of NS_B2_B1_A2_B2_A1_B2_B1
Optimization infeasible because this subproblem isn't reachable.

### Candidate
type: B, layer: 1, pos: 38

## Relational analysis of NS_B2_B1_A2_B2_A1_B2_B1
Optimization infeasible because this subproblem isn't reachable.

### Candidate
type: A, layer: 1, pos: 38

### Candidate
type: B, layer: 1, pos: 9

## Relational analysis of NS_B2_B1_A2_B2_A1_B2_B1
Optimization infeasible because this subproblem isn't reachable.

### Candidate
type: A, layer: 1, pos: 28

## Relational analysis of NS_B2_B1_A2_B2_A1_B2_A1
Optimization infeasible because this subproblem isn't reachable.

### Candidate
type: A, layer: 1, pos: 9

### Candidate
type: A, layer: 1, pos: 8

## Relational analysis of NS_B2_B1_A2_B2_A1_B2_A1

### Relational analysis result of NS_B2_B1_A2_B2_A1_B2_A1
Status: Status.UNKNOWN
Output dim: 0, lower bound: -20.9536552, upper bound: 20.9570127
time: 0.59 seconds

## Relational analysis of NS_B2_B1_A2_B2_A1_B2_A2

### Relational analysis result of NS_B2_B1_A2_B2_A1_B2_A2
Status: Status.UNKNOWN
Output dim: 0, lower bound: -20.9579247, upper bound: 20.9604336
time: 0.58 seconds

## BFS NS instance: NS_B2_B1_A2_B2_A2_A2

### Backsubstitution after applying NS history:
0: -6.0384808, 16.7348270, -5.9021125, 16.5162315, -22.5547123, 22.6369400
1: -14.9180260, 25.8566799, -14.6226358, 25.5178680, -40.4358902, 40.4793129
2: -10.4788561, 23.3561134, -10.2336988, 23.0905800, -33.5694351, 33.5898132
3: -16.1243324, 28.5728245, -15.8537064, 28.1620789, -44.2864113, 44.4265251
4: -14.7672710, 28.6892910, -14.5781689, 28.4602432, -43.2275162, 43.2674522

Time for backsubstitution: 0.82 seconds

### NS candidates at layer 1
type: A, layer: 1, pos: 1
type: B, layer: 1, pos: 45
type: A, layer: 1, pos: 45
type: B, layer: 1, pos: 47
type: B, layer: 1, pos: 39
type: A, layer: 1, pos: 39
type: B, layer: 1, pos: 38
type: A, layer: 1, pos: 6
type: A, layer: 1, pos: 38
type: A, layer: 1, pos: 8
type: A, layer: 1, pos: 9
type: B, layer: 1, pos: 20
type: B, layer: 1, pos: 48
type: B, layer: 1, pos: 37
type: B, layer: 1, pos: 9
type: A, layer: 1, pos: 28
type: B, layer: 1, pos: 6
type: A, layer: 1, pos: 37
type: A, layer: 1, pos: 48
type: B, layer: 1, pos: 8
type: A, layer: 1, pos: 20
type: A, layer: 1, pos: 31
type: B, layer: 1, pos: 28
type: B, layer: 1, pos: 31
type: B, layer: 1, pos: 24
type: A, layer: 1, pos: 24
type: B, layer: 1, pos: 5
type: A, layer: 1, pos: 5
type: B, layer: 1, pos: 0
type: A, layer: 1, pos: 0
type: B, layer: 1, pos: 41
type: A, layer: 1, pos: 41
type: B, layer: 1, pos: 26
type: A, layer: 1, pos: 26

Time for candidate selection: 0.07 seconds

### Candidate
type: A, layer: 1, pos: 1

### Candidate
type: B, layer: 1, pos: 45

### Candidate
type: A, layer: 1, pos: 45

### Candidate
type: B, layer: 1, pos: 47

## Relational analysis of NS_B2_B1_A2_B2_A2_A2_B1
Optimization infeasible because this subproblem isn't reachable.

### Candidate
type: B, layer: 1, pos: 39

## Relational analysis of NS_B2_B1_A2_B2_A2_A2_B1

### Relational analysis result of NS_B2_B1_A2_B2_A2_A2_B1
Status: Status.UNKNOWN
Output dim: 0, lower bound: -20.5282947, upper bound: 20.9498387
time: 0.61 seconds

## Relational analysis of NS_B2_B1_A2_B2_A2_A2_B2

### Relational analysis result of NS_B2_B1_A2_B2_A2_A2_B2
Status: Status.UNKNOWN
Output dim: 0, lower bound: -20.4996329, upper bound: 20.9502924
time: 0.65 seconds

## BFS NS instance: NS_B2_B2_A1_B1_A1_B1

### Backsubstitution after applying NS history:
0: -5.6512485, 15.8766308, -5.1090603, 14.2706547, -19.9219036, 20.9856911
1: -14.2241335, 24.3052940, -12.6505194, 22.0396614, -36.2637939, 36.9558105
2: -9.8599586, 22.0852985, -8.8227625, 19.9193935, -29.7793522, 30.9080620
3: -15.2573195, 26.8963223, -13.6273718, 24.3816147, -39.6389351, 40.5236931
4: -13.7445583, 27.3269234, -12.3172474, 24.4959354, -38.2404938, 39.6441689

Time for backsubstitution: 0.82 seconds

### NS candidates at layer 1
type: A, layer: 1, pos: 12
type: B, layer: 1, pos: 28
type: A, layer: 1, pos: 45
type: A, layer: 1, pos: 28
type: B, layer: 1, pos: 6
type: B, layer: 1, pos: 39
type: B, layer: 1, pos: 45
type: B, layer: 1, pos: 24
type: A, layer: 1, pos: 39
type: A, layer: 1, pos: 24
type: A, layer: 1, pos: 6
type: B, layer: 1, pos: 9
type: A, layer: 1, pos: 15
type: A, layer: 1, pos: 9
type: B, layer: 1, pos: 20
type: A, layer: 1, pos: 48
type: A, layer: 1, pos: 20
type: B, layer: 1, pos: 15
type: A, layer: 1, pos: 38
type: A, layer: 1, pos: 37
type: A, layer: 1, pos: 8
type: B, layer: 1, pos: 37
type: A, layer: 1, pos: 26
type: B, layer: 1, pos: 26
type: B, layer: 1, pos: 38
type: A, layer: 1, pos: 0
type: B, layer: 1, pos: 0
type: B, layer: 1, pos: 48
type: B, layer: 1, pos: 41
type: A, layer: 1, pos: 41
type: A, layer: 1, pos: 31
type: A, layer: 1, pos: 5
type: B, layer: 1, pos: 31
type: B, layer: 1, pos: 5

Time for candidate selection: 0.07 seconds

### Candidate
type: A, layer: 1, pos: 12

### Candidate
type: B, layer: 1, pos: 28

### Candidate
type: A, layer: 1, pos: 45

### Candidate
type: A, layer: 1, pos: 28

### Candidate
type: B, layer: 1, pos: 6

### Candidate
type: B, layer: 1, pos: 39

### Candidate
type: B, layer: 1, pos: 45

### Candidate
type: B, layer: 1, pos: 24

### Candidate
type: A, layer: 1, pos: 39

### Candidate
type: A, layer: 1, pos: 24

### Candidate
type: A, layer: 1, pos: 6

### Candidate
type: B, layer: 1, pos: 9

## Relational analysis of NS_B2_B2_A1_B1_A1_B1_B1

### Relational analysis result of NS_B2_B2_A1_B1_A1_B1_B1
Status: Status.UNKNOWN
Output dim: 0, lower bound: -20.9443701, upper bound: 20.9500642
time: 0.75 seconds

## Relational analysis of NS_B2_B2_A1_B1_A1_B1_B2

### Relational analysis result of NS_B2_B2_A1_B1_A1_B1_B2
Status: Status.UNKNOWN
Output dim: 0, lower bound: -20.9629728, upper bound: 20.9625913
time: 0.77 seconds

## BFS NS instance: NS_B2_B2_A1_B1_A1_B2

### Backsubstitution after applying NS history:
0: -6.0244703, 16.8405571, -5.7379360, 16.0502377, -22.0747070, 22.5784931
1: -15.1832008, 25.7038307, -14.3143644, 24.7327061, -39.9159012, 40.0181961
2: -10.5292540, 23.3857155, -9.9866447, 22.3560219, -32.8852768, 33.3723564
3: -16.2657433, 28.4767113, -15.3614969, 27.3282509, -43.5939941, 43.8382072
4: -14.6604919, 28.9407387, -13.8753071, 27.5357323, -42.1962242, 42.8160324

Time for backsubstitution: 0.82 seconds

### NS candidates at layer 1
type: A, layer: 1, pos: 12
type: A, layer: 1, pos: 28
type: A, layer: 1, pos: 45
type: B, layer: 1, pos: 28
type: B, layer: 1, pos: 45
type: A, layer: 1, pos: 15
type: B, layer: 1, pos: 6
type: B, layer: 1, pos: 39
type: A, layer: 1, pos: 24
type: A, layer: 1, pos: 6
type: B, layer: 1, pos: 15
type: A, layer: 1, pos: 48
type: B, layer: 1, pos: 24
type: B, layer: 1, pos: 9
type: A, layer: 1, pos: 39
type: A, layer: 1, pos: 9
type: A, layer: 1, pos: 38
type: A, layer: 1, pos: 37
type: A, layer: 1, pos: 20
type: B, layer: 1, pos: 37
type: B, layer: 1, pos: 26
type: A, layer: 1, pos: 26
type: B, layer: 1, pos: 38
type: A, layer: 1, pos: 8
type: B, layer: 1, pos: 48
type: A, layer: 1, pos: 0
type: B, layer: 1, pos: 20
type: B, layer: 1, pos: 31
type: A, layer: 1, pos: 41
type: B, layer: 1, pos: 41
type: B, layer: 1, pos: 5
type: A, layer: 1, pos: 5
type: A, layer: 1, pos: 31
type: B, layer: 1, pos: 0

Time for candidate selection: 0.07 seconds

### Candidate
type: A, layer: 1, pos: 12

### Candidate
type: A, layer: 1, pos: 28

### Candidate
type: A, layer: 1, pos: 45

### Candidate
type: B, layer: 1, pos: 28

### Candidate
type: B, layer: 1, pos: 45

### Candidate
type: A, layer: 1, pos: 15

### Candidate
type: B, layer: 1, pos: 6

### Candidate
type: B, layer: 1, pos: 39

### Candidate
type: A, layer: 1, pos: 24

### Candidate
type: A, layer: 1, pos: 6

### Candidate
type: B, layer: 1, pos: 15

### Candidate
type: A, layer: 1, pos: 48

### Candidate
type: B, layer: 1, pos: 24

### Candidate
type: B, layer: 1, pos: 9

## Relational analysis of NS_B2_B2_A1_B1_A1_B2_B1

### Relational analysis result of NS_B2_B2_A1_B1_A1_B2_B1
Status: Status.UNKNOWN
Output dim: 0, lower bound: -20.9442458, upper bound: 20.9503509
time: 0.77 seconds

## Relational analysis of NS_B2_B2_A1_B1_A1_B2_B2

### Relational analysis result of NS_B2_B2_A1_B1_A1_B2_B2
Status: Status.UNKNOWN
Output dim: 0, lower bound: -20.9640653, upper bound: 20.9646005
time: 0.72 seconds

## BFS NS instance: NS_B2_B2_A1_B1_A2_A1

### Backsubstitution after applying NS history:
0: -5.4564943, 15.3382692, -5.3909249, 15.1698895, -20.6263847, 20.7291946
1: -13.5537558, 23.7017975, -13.3623333, 23.4799442, -37.0336952, 37.0641289
2: -9.4646673, 21.5025272, -9.3433218, 21.2026176, -30.6672821, 30.8458481
3: -14.6691179, 26.1667538, -14.4016371, 25.9012508, -40.5703697, 40.5683899
4: -13.4205570, 26.5279503, -13.0711050, 26.0887833, -39.5093307, 39.5990562

Time for backsubstitution: 0.83 seconds

### NS candidates at layer 1
type: A, layer: 1, pos: 12
type: A, layer: 1, pos: 45
type: A, layer: 1, pos: 15
type: B, layer: 1, pos: 45
type: B, layer: 1, pos: 15
type: A, layer: 1, pos: 20
type: A, layer: 1, pos: 48
type: B, layer: 1, pos: 8
type: B, layer: 1, pos: 24
type: B, layer: 1, pos: 28
type: B, layer: 1, pos: 9
type: A, layer: 1, pos: 9
type: B, layer: 1, pos: 6
type: A, layer: 1, pos: 24
type: A, layer: 1, pos: 38
type: B, layer: 1, pos: 20
type: A, layer: 1, pos: 8
type: A, layer: 1, pos: 39
type: A, layer: 1, pos: 37
type: A, layer: 1, pos: 28
type: B, layer: 1, pos: 37
type: B, layer: 1, pos: 39
type: B, layer: 1, pos: 38
type: A, layer: 1, pos: 0
type: B, layer: 1, pos: 48
type: B, layer: 1, pos: 0
type: A, layer: 1, pos: 26
type: B, layer: 1, pos: 26
type: B, layer: 1, pos: 31
type: B, layer: 1, pos: 5
type: A, layer: 1, pos: 31
type: A, layer: 1, pos: 41
type: B, layer: 1, pos: 41
type: A, layer: 1, pos: 5

Time for candidate selection: 0.07 seconds

### Candidate
type: A, layer: 1, pos: 12

### Candidate
type: A, layer: 1, pos: 45

### Candidate
type: A, layer: 1, pos: 15

### Candidate
type: B, layer: 1, pos: 45

### Candidate
type: B, layer: 1, pos: 15

### Candidate
type: A, layer: 1, pos: 20

## Relational analysis of NS_B2_B2_A1_B1_A2_A1_A1
Optimization infeasible because this subproblem isn't reachable.

### Candidate
type: A, layer: 1, pos: 48

### Candidate
type: B, layer: 1, pos: 8

## Relational analysis of NS_B2_B2_A1_B1_A2_A1_B1

### Relational analysis result of NS_B2_B2_A1_B1_A2_A1_B1
Status: Status.UNKNOWN
Output dim: 0, lower bound: -20.9384564, upper bound: 20.9264940
time: 0.81 seconds

## Relational analysis of NS_B2_B2_A1_B1_A2_A1_B2

### Relational analysis result of NS_B2_B2_A1_B1_A2_A1_B2
Status: Status.UNKNOWN
Output dim: 0, lower bound: -20.9390731, upper bound: 20.9274296
time: 0.77 seconds

## BFS NS instance: NS_B2_B2_A1_B1_A2_A2

### Backsubstitution after applying NS history:
0: -6.1045394, 16.9758701, -5.7815499, 16.1679955, -22.2725353, 22.7574196
1: -15.1983242, 26.0349178, -14.4255676, 24.9067802, -40.1051025, 40.4604836
2: -10.6195745, 23.6475563, -10.0648251, 22.5165768, -33.1361504, 33.7123795
3: -16.4145107, 28.8137779, -15.4787817, 27.5215988, -43.9361076, 44.2925568
4: -15.0041790, 29.1842613, -13.9816208, 27.7338066, -42.7379837, 43.1658821

Time for backsubstitution: 0.83 seconds

### NS candidates at layer 1
type: A, layer: 1, pos: 12
type: A, layer: 1, pos: 45
type: B, layer: 1, pos: 45
type: A, layer: 1, pos: 28
type: A, layer: 1, pos: 15
type: B, layer: 1, pos: 28
type: A, layer: 1, pos: 48
type: B, layer: 1, pos: 15
type: A, layer: 1, pos: 24
type: B, layer: 1, pos: 24
type: B, layer: 1, pos: 8
type: B, layer: 1, pos: 9
type: A, layer: 1, pos: 39
type: A, layer: 1, pos: 9
type: A, layer: 1, pos: 38
type: B, layer: 1, pos: 37
type: A, layer: 1, pos: 37
type: B, layer: 1, pos: 6
type: B, layer: 1, pos: 39
type: B, layer: 1, pos: 38
type: A, layer: 1, pos: 20
type: A, layer: 1, pos: 8
type: A, layer: 1, pos: 26
type: B, layer: 1, pos: 26
type: B, layer: 1, pos: 20
type: B, layer: 1, pos: 48
type: A, layer: 1, pos: 0
type: B, layer: 1, pos: 5
type: B, layer: 1, pos: 31
type: A, layer: 1, pos: 41
type: B, layer: 1, pos: 41
type: A, layer: 1, pos: 31
type: B, layer: 1, pos: 0
type: A, layer: 1, pos: 5

Time for candidate selection: 0.07 seconds

### Candidate
type: A, layer: 1, pos: 12

### Candidate
type: A, layer: 1, pos: 45

### Candidate
type: B, layer: 1, pos: 45

### Candidate
type: A, layer: 1, pos: 28

### Candidate
type: A, layer: 1, pos: 15

### Candidate
type: B, layer: 1, pos: 28

### Candidate
type: A, layer: 1, pos: 48

### Candidate
type: B, layer: 1, pos: 15

### Candidate
type: A, layer: 1, pos: 24

## Relational analysis of NS_B2_B2_A1_B1_A2_A2_A1
Optimization infeasible because this subproblem isn't reachable.

### Candidate
type: B, layer: 1, pos: 24

### Candidate
type: B, layer: 1, pos: 8

## Relational analysis of NS_B2_B2_A1_B1_A2_A2_B1

### Relational analysis result of NS_B2_B2_A1_B1_A2_A2_B1
Status: Status.UNKNOWN
Output dim: 0, lower bound: -20.9504192, upper bound: 20.9458066
time: 0.65 seconds

## Relational analysis of NS_B2_B2_A1_B1_A2_A2_B2

### Relational analysis result of NS_B2_B2_A1_B1_A2_A2_B2
Status: Status.UNKNOWN
Output dim: 0, lower bound: -20.9390731, upper bound: 20.9479840
time: 0.64 seconds

## BFS NS instance: NS_B2_B2_A1_B2_A1_B1

### Backsubstitution after applying NS history:
0: -5.6512485, 15.8766308, -5.2285590, 14.5446215, -20.1958694, 21.1051846
1: -14.2241335, 24.3052940, -12.7897882, 22.6128254, -36.8369560, 37.0950813
2: -9.8599586, 22.0852985, -8.9804058, 20.3811893, -30.2411480, 31.0657043
3: -15.2573195, 26.8963223, -13.8934088, 24.9570065, -40.2143211, 40.7897301
4: -13.7445583, 27.3269234, -12.7574091, 24.9819221, -38.7264748, 40.0843315

Time for backsubstitution: 0.83 seconds

### NS candidates at layer 1
type: A, layer: 1, pos: 12
type: A, layer: 1, pos: 45
type: B, layer: 1, pos: 39
type: A, layer: 1, pos: 15
type: A, layer: 1, pos: 28
type: B, layer: 1, pos: 28
type: B, layer: 1, pos: 45
type: B, layer: 1, pos: 15
type: B, layer: 1, pos: 6
type: B, layer: 1, pos: 9
type: A, layer: 1, pos: 24
type: A, layer: 1, pos: 20
type: B, layer: 1, pos: 24
type: A, layer: 1, pos: 9
type: A, layer: 1, pos: 37
type: A, layer: 1, pos: 38
type: B, layer: 1, pos: 37
type: A, layer: 1, pos: 6
type: B, layer: 1, pos: 20
type: B, layer: 1, pos: 38
type: A, layer: 1, pos: 48
type: A, layer: 1, pos: 8
type: A, layer: 1, pos: 39
type: B, layer: 1, pos: 26
type: A, layer: 1, pos: 26
type: B, layer: 1, pos: 48
type: A, layer: 1, pos: 0
type: B, layer: 1, pos: 0
type: B, layer: 1, pos: 41
type: A, layer: 1, pos: 5
type: A, layer: 1, pos: 41
type: A, layer: 1, pos: 31
type: B, layer: 1, pos: 31
type: B, layer: 1, pos: 5

Time for candidate selection: 0.07 seconds

### Candidate
type: A, layer: 1, pos: 12

### Candidate
type: A, layer: 1, pos: 45

### Candidate
type: B, layer: 1, pos: 39

### Candidate
type: A, layer: 1, pos: 15

### Candidate
type: A, layer: 1, pos: 28

### Candidate
type: B, layer: 1, pos: 28

### Candidate
type: B, layer: 1, pos: 45

### Candidate
type: B, layer: 1, pos: 15

### Candidate
type: B, layer: 1, pos: 6

### Candidate
type: B, layer: 1, pos: 9

## Relational analysis of NS_B2_B2_A1_B2_A1_B1_B1

### Relational analysis result of NS_B2_B2_A1_B2_A1_B1_B1
Status: Status.UNKNOWN
Output dim: 0, lower bound: -20.9446852, upper bound: 20.9503422
time: 0.78 seconds

## Relational analysis of NS_B2_B2_A1_B2_A1_B1_B2

### Relational analysis result of NS_B2_B2_A1_B2_A1_B1_B2
Status: Status.UNKNOWN
Output dim: 0, lower bound: -20.9460557, upper bound: 20.9511217
time: 0.78 seconds

## BFS NS instance: NS_B2_B2_A1_B2_A1_B2

### Backsubstitution after applying NS history:
0: -6.0244703, 16.8405571, -5.9087162, 16.4332104, -22.4576797, 22.7492733
1: -15.1832008, 25.7038307, -14.5746231, 25.4128418, -40.5960426, 40.2784538
2: -10.5292540, 23.3857155, -10.2367010, 22.9520721, -33.4813232, 33.6224136
3: -16.2657433, 28.4767113, -15.7680588, 28.0584450, -44.3241844, 44.2447701
4: -14.6604919, 28.9407387, -14.4541407, 28.1912174, -42.8517075, 43.3948669

Time for backsubstitution: 0.84 seconds

### NS candidates at layer 1
type: A, layer: 1, pos: 12
type: A, layer: 1, pos: 45
type: A, layer: 1, pos: 15
type: A, layer: 1, pos: 28
type: B, layer: 1, pos: 15
type: B, layer: 1, pos: 39
type: B, layer: 1, pos: 45
type: B, layer: 1, pos: 6
type: A, layer: 1, pos: 24
type: B, layer: 1, pos: 9
type: B, layer: 1, pos: 28
type: A, layer: 1, pos: 6
type: A, layer: 1, pos: 38
type: A, layer: 1, pos: 37
type: A, layer: 1, pos: 20
type: A, layer: 1, pos: 9
type: B, layer: 1, pos: 24
type: A, layer: 1, pos: 48
type: B, layer: 1, pos: 48
type: B, layer: 1, pos: 37
type: B, layer: 1, pos: 38
type: A, layer: 1, pos: 8
type: B, layer: 1, pos: 26
type: A, layer: 1, pos: 0
type: A, layer: 1, pos: 26
type: A, layer: 1, pos: 39
type: B, layer: 1, pos: 31
type: A, layer: 1, pos: 5
type: B, layer: 1, pos: 41
type: A, layer: 1, pos: 41
type: B, layer: 1, pos: 20
type: B, layer: 1, pos: 0
type: A, layer: 1, pos: 31
type: B, layer: 1, pos: 5

Time for candidate selection: 0.07 seconds

### Candidate
type: A, layer: 1, pos: 12

### Candidate
type: A, layer: 1, pos: 45

### Candidate
type: A, layer: 1, pos: 15

### Candidate
type: A, layer: 1, pos: 28

### Candidate
type: B, layer: 1, pos: 15

### Candidate
type: B, layer: 1, pos: 39

### Candidate
type: B, layer: 1, pos: 45

### Candidate
type: B, layer: 1, pos: 6

### Candidate
type: A, layer: 1, pos: 24

### Candidate
type: B, layer: 1, pos: 9

## Relational analysis of NS_B2_B2_A1_B2_A1_B2_B1

### Relational analysis result of NS_B2_B2_A1_B2_A1_B2_B1
Status: Status.UNKNOWN
Output dim: 0, lower bound: -20.9449710, upper bound: 20.9514500
time: 0.68 seconds

## Relational analysis of NS_B2_B2_A1_B2_A1_B2_B2

### Relational analysis result of NS_B2_B2_A1_B2_A1_B2_B2
Status: Status.UNKNOWN
Output dim: 0, lower bound: -20.9523718, upper bound: 20.9564293
time: 0.61 seconds

## BFS NS instance: NS_B2_B2_A1_B2_A2_B1

### Backsubstitution after applying NS history:
0: -5.8963809, 16.4548492, -5.2285590, 14.5446215, -20.4410019, 21.6834049
1: -14.7113638, 25.2500343, -12.7897882, 22.6128254, -37.3241882, 38.0398216
2: -10.2584276, 22.9362125, -8.9804058, 20.3811893, -30.6396179, 31.9166183
3: -15.8670597, 27.9255943, -13.8934088, 24.9570065, -40.8240509, 41.8190041
4: -14.4673929, 28.3199825, -12.7574091, 24.9819221, -39.4493065, 41.0773926

Time for backsubstitution: 0.83 seconds

### NS candidates at layer 1
type: A, layer: 1, pos: 12
type: A, layer: 1, pos: 45
type: B, layer: 1, pos: 45
type: A, layer: 1, pos: 15
type: B, layer: 1, pos: 28
type: B, layer: 1, pos: 15
type: A, layer: 1, pos: 28
type: B, layer: 1, pos: 6
type: B, layer: 1, pos: 39
type: B, layer: 1, pos: 9
type: B, layer: 1, pos: 24
type: A, layer: 1, pos: 24
type: A, layer: 1, pos: 9
type: A, layer: 1, pos: 48
type: A, layer: 1, pos: 37
type: A, layer: 1, pos: 38
type: B, layer: 1, pos: 20
type: A, layer: 1, pos: 6
type: A, layer: 1, pos: 39
type: B, layer: 1, pos: 37
type: A, layer: 1, pos: 20
type: A, layer: 1, pos: 8
type: B, layer: 1, pos: 38
type: B, layer: 1, pos: 26
type: A, layer: 1, pos: 26
type: B, layer: 1, pos: 48
type: A, layer: 1, pos: 0
type: B, layer: 1, pos: 0
type: A, layer: 1, pos: 31
type: B, layer: 1, pos: 41
type: A, layer: 1, pos: 41
type: A, layer: 1, pos: 5
type: B, layer: 1, pos: 31
type: B, layer: 1, pos: 5

Time for candidate selection: 0.07 seconds

### Candidate
type: A, layer: 1, pos: 12

### Candidate
type: A, layer: 1, pos: 45

### Candidate
type: B, layer: 1, pos: 45

### Candidate
type: A, layer: 1, pos: 15

### Candidate
type: B, layer: 1, pos: 28

### Candidate
type: B, layer: 1, pos: 15

### Candidate
type: A, layer: 1, pos: 28

### Candidate
type: B, layer: 1, pos: 6

### Candidate
type: B, layer: 1, pos: 39

## Relational analysis of NS_B2_B2_A1_B2_A2_B1_B1
Optimization infeasible because this subproblem isn't reachable.

### Candidate
type: B, layer: 1, pos: 9

## Relational analysis of NS_B2_B2_A1_B2_A2_B1_B1

### Relational analysis result of NS_B2_B2_A1_B2_A2_B1_B1
Status: Status.UNKNOWN
Output dim: 0, lower bound: -20.9273829, upper bound: 20.9235466
time: 0.80 seconds

## Relational analysis of NS_B2_B2_A1_B2_A2_B1_B2

### Relational analysis result of NS_B2_B2_A1_B2_A2_B1_B2
Status: Status.UNKNOWN
Output dim: 0, lower bound: -20.9313827, upper bound: 20.9330553
time: 0.72 seconds

## BFS NS instance: NS_B2_B2_A1_B2_A2_B2

### Backsubstitution after applying NS history:
0: -6.2250795, 17.2942638, -5.9087162, 16.4332104, -22.6582909, 23.2029781
1: -15.5413942, 26.4833488, -14.5746231, 25.4128418, -40.9542313, 41.0579720
2: -10.8425531, 24.0679264, -10.2367010, 22.9520721, -33.7946205, 34.3046265
3: -16.7520714, 29.3229733, -15.7680588, 28.0584450, -44.8105087, 45.0910339
4: -15.2782488, 29.7155247, -14.4541407, 28.1912174, -43.4694672, 44.1696663

Time for backsubstitution: 0.84 seconds

### NS candidates at layer 1
type: A, layer: 1, pos: 12
type: A, layer: 1, pos: 45
type: B, layer: 1, pos: 45
type: A, layer: 1, pos: 15
type: B, layer: 1, pos: 15
type: A, layer: 1, pos: 28
type: B, layer: 1, pos: 6
type: A, layer: 1, pos: 6
type: A, layer: 1, pos: 24
type: B, layer: 1, pos: 28
type: A, layer: 1, pos: 48
type: B, layer: 1, pos: 9
type: B, layer: 1, pos: 24
type: A, layer: 1, pos: 9
type: B, layer: 1, pos: 39
type: A, layer: 1, pos: 38
type: A, layer: 1, pos: 37
type: B, layer: 1, pos: 37
type: A, layer: 1, pos: 20
type: B, layer: 1, pos: 38
type: A, layer: 1, pos: 8
type: B, layer: 1, pos: 48
type: A, layer: 1, pos: 39
type: B, layer: 1, pos: 26
type: A, layer: 1, pos: 26
type: B, layer: 1, pos: 20
type: A, layer: 1, pos: 0
type: B, layer: 1, pos: 31
type: B, layer: 1, pos: 41
type: A, layer: 1, pos: 41
type: A, layer: 1, pos: 5
type: B, layer: 1, pos: 5
type: A, layer: 1, pos: 31
type: B, layer: 1, pos: 0

Time for candidate selection: 0.07 seconds

### Candidate
type: A, layer: 1, pos: 12

### Candidate
type: A, layer: 1, pos: 45

### Candidate
type: B, layer: 1, pos: 45

### Candidate
type: A, layer: 1, pos: 15

### Candidate
type: B, layer: 1, pos: 15

### Candidate
type: A, layer: 1, pos: 28

### Candidate
type: B, layer: 1, pos: 6

### Candidate
type: A, layer: 1, pos: 6

### Candidate
type: A, layer: 1, pos: 24

### Candidate
type: B, layer: 1, pos: 28

### Candidate
type: A, layer: 1, pos: 48

### Candidate
type: B, layer: 1, pos: 9

## Relational analysis of NS_B2_B2_A1_B2_A2_B2_B1

### Relational analysis result of NS_B2_B2_A1_B2_A2_B2_B1
Status: Status.UNKNOWN
Output dim: 0, lower bound: -20.9272313, upper bound: 20.9235597
time: 0.87 seconds

## Relational analysis of NS_B2_B2_A1_B2_A2_B2_B2

### Relational analysis result of NS_B2_B2_A1_B2_A2_B2_B2
Status: Status.UNKNOWN
Output dim: 0, lower bound: -20.9404709, upper bound: 20.9393947
time: 0.65 seconds

## BFS NS instance: NS_B2_B2_A2_B1_A1_A1

### Backsubstitution after applying NS history:
0: -5.6934490, 16.0427685, -5.7815499, 16.1679955, -21.8614445, 21.8243179
1: -14.2588348, 24.7153702, -14.4255676, 24.9067802, -39.1656151, 39.1409378
2: -9.9096851, 22.3794670, -10.0648251, 22.5165768, -32.4262619, 32.4442902
3: -15.3478031, 27.2905960, -15.4787817, 27.5215988, -42.8694000, 42.7693710
4: -13.9048977, 27.6525459, -13.9816208, 27.7338066, -41.6386986, 41.6341667

Time for backsubstitution: 0.84 seconds

### NS candidates at layer 1
type: B, layer: 1, pos: 28
type: A, layer: 1, pos: 28
type: A, layer: 1, pos: 45
type: B, layer: 1, pos: 45
type: B, layer: 1, pos: 6
type: A, layer: 1, pos: 15
type: B, layer: 1, pos: 15
type: A, layer: 1, pos: 39
type: A, layer: 1, pos: 24
type: A, layer: 1, pos: 6
type: B, layer: 1, pos: 8
type: B, layer: 1, pos: 39
type: B, layer: 1, pos: 24
type: B, layer: 1, pos: 9
type: A, layer: 1, pos: 9
type: A, layer: 1, pos: 37
type: A, layer: 1, pos: 38
type: A, layer: 1, pos: 48
type: A, layer: 1, pos: 8
type: B, layer: 1, pos: 37
type: B, layer: 1, pos: 38
type: B, layer: 1, pos: 48
type: A, layer: 1, pos: 26
type: B, layer: 1, pos: 26
type: A, layer: 1, pos: 20
type: B, layer: 1, pos: 31
type: A, layer: 1, pos: 0
type: B, layer: 1, pos: 20
type: A, layer: 1, pos: 31
type: A, layer: 1, pos: 41
type: B, layer: 1, pos: 41
type: B, layer: 1, pos: 0
type: A, layer: 1, pos: 5
type: B, layer: 1, pos: 5

Time for candidate selection: 0.07 seconds

### Candidate
type: B, layer: 1, pos: 28

## Relational analysis of NS_B2_B2_A2_B1_A1_A1_B1
Optimization infeasible because this subproblem isn't reachable.

### Candidate
type: A, layer: 1, pos: 28

### Candidate
type: A, layer: 1, pos: 45

### Candidate
type: B, layer: 1, pos: 45

### Candidate
type: B, layer: 1, pos: 6

## Relational analysis of NS_B2_B2_A2_B1_A1_A1_B1
Optimization infeasible because this subproblem isn't reachable.

### Candidate
type: A, layer: 1, pos: 15

### Candidate
type: B, layer: 1, pos: 15

### Candidate
type: A, layer: 1, pos: 39

## Relational analysis of NS_B2_B2_A2_B1_A1_A1_A1

### Relational analysis result of NS_B2_B2_A2_B1_A1_A1_A1
Status: Status.UNKNOWN
Output dim: 0, lower bound: -20.9589232, upper bound: 20.9568410
time: 0.63 seconds

## Relational analysis of NS_B2_B2_A2_B1_A1_A1_A2
Optimization infeasible because this subproblem isn't reachable.

### Candidate
type: A, layer: 1, pos: 24

## Relational analysis of NS_B2_B2_A2_B1_A1_A1_A1
Optimization infeasible because this subproblem isn't reachable.

### Candidate
type: A, layer: 1, pos: 6

### Candidate
type: B, layer: 1, pos: 8

## Relational analysis of NS_B2_B2_A2_B1_A1_A1_B1

### Relational analysis result of NS_B2_B2_A2_B1_A1_A1_B1
Status: Status.UNKNOWN
Output dim: 0, lower bound: -20.9574820, upper bound: 20.9539224
time: 0.67 seconds

## Relational analysis of NS_B2_B2_A2_B1_A1_A1_B2

### Relational analysis result of NS_B2_B2_A2_B1_A1_A1_B2
Status: Status.UNKNOWN
Output dim: 0, lower bound: -20.9610848, upper bound: 20.9582666
time: 1.26 seconds

## BFS NS instance: NS_B2_B2_A2_B1_A1_A2

### Backsubstitution after applying NS history:
0: -5.9458299, 16.6518002, -5.7815499, 16.1679955, -22.1138248, 22.4333477
1: -14.7555151, 25.7168369, -14.4255676, 24.9067802, -39.6622963, 40.1424026
2: -10.3130445, 23.2755966, -10.0648251, 22.5165768, -32.8296204, 33.3404121
3: -15.9753675, 28.3708611, -15.4787817, 27.5215988, -43.4969673, 43.8496323
4: -14.6584167, 28.6986294, -13.9816208, 27.7338066, -42.3922234, 42.6802444

Time for backsubstitution: 0.85 seconds

### NS candidates at layer 1
type: A, layer: 1, pos: 15
type: A, layer: 1, pos: 45
type: B, layer: 1, pos: 15
type: B, layer: 1, pos: 45
type: A, layer: 1, pos: 39
type: B, layer: 1, pos: 28
type: A, layer: 1, pos: 48
type: A, layer: 1, pos: 38
type: B, layer: 1, pos: 38
type: A, layer: 1, pos: 9
type: B, layer: 1, pos: 9
type: B, layer: 1, pos: 8
type: A, layer: 1, pos: 28
type: B, layer: 1, pos: 6
type: B, layer: 1, pos: 37
type: A, layer: 1, pos: 37
type: A, layer: 1, pos: 6
type: B, layer: 1, pos: 20
type: A, layer: 1, pos: 20
type: B, layer: 1, pos: 48
type: B, layer: 1, pos: 39
type: A, layer: 1, pos: 8
type: B, layer: 1, pos: 31
type: A, layer: 1, pos: 31
type: B, layer: 1, pos: 24
type: A, layer: 1, pos: 24
type: B, layer: 1, pos: 0
type: B, layer: 1, pos: 5
type: A, layer: 1, pos: 0
type: A, layer: 1, pos: 26
type: B, layer: 1, pos: 26
type: A, layer: 1, pos: 41
type: B, layer: 1, pos: 41
type: A, layer: 1, pos: 5

Time for candidate selection: 0.07 seconds

### Candidate
type: A, layer: 1, pos: 15

### Candidate
type: A, layer: 1, pos: 45

### Candidate
type: B, layer: 1, pos: 15

### Candidate
type: B, layer: 1, pos: 45

### Candidate
type: A, layer: 1, pos: 39

## Relational analysis of NS_B2_B2_A2_B1_A1_A2_A1
Optimization infeasible because this subproblem isn't reachable.

### Candidate
type: B, layer: 1, pos: 28

## Relational analysis of NS_B2_B2_A2_B1_A1_A2_B1
Optimization infeasible because this subproblem isn't reachable.

### Candidate
type: A, layer: 1, pos: 48

## Relational analysis of NS_B2_B2_A2_B1_A1_A2_A1
Optimization infeasible because this subproblem isn't reachable.

### Candidate
type: A, layer: 1, pos: 38

## Relational analysis of NS_B2_B2_A2_B1_A1_A2_A1
Optimization infeasible because this subproblem isn't reachable.

### Candidate
type: B, layer: 1, pos: 38

### Candidate
type: A, layer: 1, pos: 9

## Relational analysis of NS_B2_B2_A2_B1_A1_A2_A1
Optimization infeasible because this subproblem isn't reachable.

### Candidate
type: B, layer: 1, pos: 9

### Candidate
type: B, layer: 1, pos: 8

## Relational analysis of NS_B2_B2_A2_B1_A1_A2_B1

### Relational analysis result of NS_B2_B2_A2_B1_A1_A2_B1
Status: Status.UNKNOWN
Output dim: 0, lower bound: -20.9574820, upper bound: 20.9539224
time: 0.80 seconds

## Relational analysis of NS_B2_B2_A2_B1_A1_A2_B2

### Relational analysis result of NS_B2_B2_A2_B1_A1_A2_B2
Status: Status.UNKNOWN
Output dim: 0, lower bound: -20.9610848, upper bound: 20.9582666
time: 0.72 seconds

## BFS NS instance: NS_B2_B2_A2_B1_A2_A1

### Backsubstitution after applying NS history:
0: -5.7815499, 16.1679955, -5.7815499, 16.1679955, -21.9495449, 21.9495449
1: -14.4255676, 24.9067802, -14.4255676, 24.9067802, -39.3323479, 39.3323479
2: -10.0648251, 22.5165768, -10.0648251, 22.5165768, -32.5814018, 32.5813980
3: -15.4787817, 27.5215988, -15.4787817, 27.5215988, -43.0003738, 43.0003738
4: -13.9816208, 27.7338066, -13.9816208, 27.7338066, -41.7154236, 41.7154236

Time for backsubstitution: 0.84 seconds

### NS candidates at layer 1
type: A, layer: 1, pos: 45
type: B, layer: 1, pos: 45
type: A, layer: 1, pos: 28
type: B, layer: 1, pos: 28
type: A, layer: 1, pos: 15
type: B, layer: 1, pos: 15
type: A, layer: 1, pos: 20
type: B, layer: 1, pos: 20
type: A, layer: 1, pos: 6
type: B, layer: 1, pos: 6
type: B, layer: 1, pos: 39
type: A, layer: 1, pos: 39
type: A, layer: 1, pos: 9
type: B, layer: 1, pos: 9
type: B, layer: 1, pos: 24
type: A, layer: 1, pos: 24
type: A, layer: 1, pos: 37
type: B, layer: 1, pos: 37
type: A, layer: 1, pos: 8
type: B, layer: 1, pos: 8
type: B, layer: 1, pos: 48
type: A, layer: 1, pos: 48
type: A, layer: 1, pos: 38
type: B, layer: 1, pos: 38
type: A, layer: 1, pos: 26
type: B, layer: 1, pos: 26
type: A, layer: 1, pos: 31
type: B, layer: 1, pos: 31
type: B, layer: 1, pos: 0
type: A, layer: 1, pos: 0
type: A, layer: 1, pos: 5
type: B, layer: 1, pos: 5
type: A, layer: 1, pos: 41
type: B, layer: 1, pos: 41

Time for candidate selection: 0.07 seconds

### Candidate
type: A, layer: 1, pos: 45

### Candidate
type: B, layer: 1, pos: 45

### Candidate
type: A, layer: 1, pos: 28

### Candidate
type: B, layer: 1, pos: 28

### Candidate
type: A, layer: 1, pos: 15

## Relational analysis of NS_B2_B2_A2_B1_A2_A1_A1
Optimization infeasible because this subproblem isn't reachable.

### Candidate
type: B, layer: 1, pos: 15

### Candidate
type: A, layer: 1, pos: 20

## Relational analysis of NS_B2_B2_A2_B1_A2_A1_A1

### Relational analysis result of NS_B2_B2_A2_B1_A2_A1_A1
Status: Status.UNKNOWN
Output dim: 0, lower bound: -20.9481945, upper bound: 20.9453032
time: 0.69 seconds

## Relational analysis of NS_B2_B2_A2_B1_A2_A1_A2

### Relational analysis result of NS_B2_B2_A2_B1_A2_A1_A2
Status: Status.UNKNOWN
Output dim: 0, lower bound: -20.9569857, upper bound: 20.9544784
time: 0.71 seconds

## BFS NS instance: NS_B2_B2_A2_B1_A2_A2

### Backsubstitution after applying NS history:
0: -5.9213266, 16.4778671, -5.7815499, 16.1679955, -22.0893211, 22.2594166
1: -14.6102200, 25.4814663, -14.4255676, 24.9067802, -39.5169983, 39.9070320
2: -10.2608318, 23.0094490, -10.0648251, 22.5165768, -32.7774048, 33.0742683
3: -15.8002529, 28.1255474, -15.4787817, 27.5215988, -43.3218536, 43.6043205
4: -14.4836226, 28.2687950, -13.9816208, 27.7338066, -42.2174263, 42.2504158

Time for backsubstitution: 0.85 seconds

### NS candidates at layer 1
type: A, layer: 1, pos: 45
type: B, layer: 1, pos: 45
type: B, layer: 1, pos: 28
type: B, layer: 1, pos: 20
type: A, layer: 1, pos: 39
type: A, layer: 1, pos: 28
type: A, layer: 1, pos: 15
type: B, layer: 1, pos: 15
type: A, layer: 1, pos: 6
type: A, layer: 1, pos: 48
type: B, layer: 1, pos: 24
type: A, layer: 1, pos: 20
type: B, layer: 1, pos: 37
type: A, layer: 1, pos: 9
type: A, layer: 1, pos: 8
type: B, layer: 1, pos: 6
type: A, layer: 1, pos: 24
type: B, layer: 1, pos: 9
type: A, layer: 1, pos: 37
type: B, layer: 1, pos: 8
type: B, layer: 1, pos: 38
type: A, layer: 1, pos: 38
type: B, layer: 1, pos: 39
type: B, layer: 1, pos: 48
type: A, layer: 1, pos: 26
type: B, layer: 1, pos: 26
type: A, layer: 1, pos: 31
type: B, layer: 1, pos: 5
type: B, layer: 1, pos: 0
type: B, layer: 1, pos: 31
type: A, layer: 1, pos: 0
type: A, layer: 1, pos: 41
type: B, layer: 1, pos: 41
type: A, layer: 1, pos: 5

Time for candidate selection: 0.07 seconds

### Candidate
type: A, layer: 1, pos: 45

### Candidate
type: B, layer: 1, pos: 45

### Candidate
type: B, layer: 1, pos: 28

### Candidate
type: B, layer: 1, pos: 20

## Relational analysis of NS_B2_B2_A2_B1_A2_A2_B1

### Relational analysis result of NS_B2_B2_A2_B1_A2_A2_B1
Status: Status.UNKNOWN
Output dim: 0, lower bound: -20.9493107, upper bound: 20.9467545
time: 0.84 seconds

## Relational analysis of NS_B2_B2_A2_B1_A2_A2_B2

### Relational analysis result of NS_B2_B2_A2_B1_A2_A2_B2
Status: Status.UNKNOWN
Output dim: 0, lower bound: -20.9424550, upper bound: 20.9544784
time: 0.76 seconds

## Summary of splitting at layer (split count: 6)
- Time for NS candidates: 2.53 seconds
NS_B1_A1_A1_A1_B1_B1_A1, status: Status.UNKNOWN, split count: 7, time: 2.53
Output dim: 0, lower bound: -20.9355777, upper bound: 20.9196433
NS_B1_A1_A1_A1_B1_B1_A2, status: Status.UNKNOWN, split count: 7, time: 2.53
Output dim: 0, lower bound: -20.9584755, upper bound: 20.9553408
NS_B1_A1_A1_A1_B1_B2_A1, status: Status.UNKNOWN, split count: 7, time: 2.53
Output dim: 0, lower bound: -20.9577583, upper bound: 20.9550095
NS_B1_A1_A1_A1_B1_B2_A2, status: Status.UNKNOWN, split count: 7, time: 2.53
Output dim: 0, lower bound: -20.9577507, upper bound: 20.9549928
NS_B1_A1_A1_A1_B2_B1_B1, status: Status.UNKNOWN, split count: 7, time: 2.53
Output dim: 0, lower bound: -20.9106196, upper bound: 20.9286097
NS_B1_A1_A1_A1_B2_B1_B2, status: Status.UNKNOWN, split count: 7, time: 2.53
Output dim: 0, lower bound: -20.9222198, upper bound: 20.9359225
NS_B1_A1_A1_A1_B2_B2_B1, status: Status.UNKNOWN, split count: 7, time: 2.53
Output dim: 0, lower bound: -20.9106196, upper bound: 20.9286097
NS_B1_A1_A1_A1_B2_B2_B2, status: Status.UNKNOWN, split count: 7, time: 2.53
Output dim: 0, lower bound: -20.9222198, upper bound: 20.9359225
NS_B1_A1_A1_A2_B1_B1_A1, status: Status.UNKNOWN, split count: 7, time: 2.53
Output dim: 0, lower bound: -20.9355777, upper bound: 20.9220791
NS_B1_A1_A1_A2_B1_B1_A2, status: Status.UNKNOWN, split count: 7, time: 2.53
Output dim: 0, lower bound: -20.9451984, upper bound: 20.9343038
NS_B1_A1_A1_A2_B1_B2_A1, status: Status.UNKNOWN, split count: 7, time: 2.53
Output dim: 0, lower bound: -20.9355777, upper bound: 20.9220791
NS_B1_A1_A1_A2_B1_B2_A2, status: Status.UNKNOWN, split count: 7, time: 2.53
Output dim: 0, lower bound: -20.9451984, upper bound: 20.9343038
NS_B1_A1_A1_A2_B2_B1_A1, status: Status.VERIFIED, split count: 7, time: 2.53
Output dim: 0, lower bound: -20.7814451, upper bound: 20.7823533
NS_B1_A1_A1_A2_B2_B1_A2, status: Status.VERIFIED, split count: 7, time: 2.53
Output dim: 0, lower bound: -20.7814451, upper bound: 20.7992035
NS_B1_A1_A1_A2_B2_B2_B1, status: Status.UNKNOWN, split count: 7, time: 2.53
Output dim: 0, lower bound: -20.9097540, upper bound: 20.9164577
NS_B1_A1_A1_A2_B2_B2_B2, status: Status.VERIFIED, split count: 7, time: 2.53
Output dim: 0, lower bound: -20.7927946, upper bound: 20.7992037
NS_B1_A2_A1_B1_B1_A1_B1, status: Status.UNKNOWN, split count: 7, time: 2.53
Output dim: 0, lower bound: -20.9425656, upper bound: 20.9297386
NS_B1_A2_A1_B1_B1_A1_B2, status: Status.UNKNOWN, split count: 7, time: 2.53
Output dim: 0, lower bound: -20.9418692, upper bound: 20.9293078
NS_B1_A2_A1_B1_B1_A2_B1, status: Status.UNKNOWN, split count: 7, time: 2.53
Output dim: 0, lower bound: -20.9601795, upper bound: 20.9570587
NS_B1_A2_A1_B1_B1_A2_B2, status: Status.UNKNOWN, split count: 7, time: 2.53
Output dim: 0, lower bound: -20.9600750, upper bound: 20.9570448
NS_B1_A2_A1_B1_B2_A1_B1, status: Status.UNKNOWN, split count: 7, time: 2.53
Output dim: 0, lower bound: -20.9304753, upper bound: 20.9297088
NS_B1_A2_A1_B1_B2_A1_B2, status: Status.UNKNOWN, split count: 7, time: 2.53
Output dim: 0, lower bound: -20.9304753, upper bound: 20.9441063
NS_B1_A2_A1_B1_B2_A2_B1, status: Status.UNKNOWN, split count: 7, time: 2.53
Output dim: 0, lower bound: -20.9632714, upper bound: 20.9621042
NS_B1_A2_A1_B1_B2_A2_B2, status: Status.UNKNOWN, split count: 7, time: 2.53
Output dim: 0, lower bound: -20.9600750, upper bound: 20.9633596
NS_B1_A2_A1_B2_B1_A1_B1, status: Status.UNKNOWN, split count: 7, time: 2.53
Output dim: 0, lower bound: -20.9023541, upper bound: 20.9074667
NS_B1_A2_A1_B2_B1_A1_B2, status: Status.UNKNOWN, split count: 7, time: 2.53
Output dim: 0, lower bound: -20.9023299, upper bound: 20.9075448
NS_B1_A2_A1_B2_B1_A2_A1, status: Status.UNKNOWN, split count: 7, time: 2.53
Output dim: 0, lower bound: -20.9174305, upper bound: 20.9285825
NS_B1_A2_A1_B2_B1_A2_A2, status: Status.UNKNOWN, split count: 7, time: 2.53
Output dim: 0, lower bound: -20.9262643, upper bound: 20.9387501
NS_B1_A2_A1_B2_B2_B1_A1, status: Status.UNKNOWN, split count: 7, time: 2.53
Output dim: 0, lower bound: -20.9023832, upper bound: 20.9267171
NS_B1_A2_A1_B2_B2_B1_A2, status: Status.UNKNOWN, split count: 7, time: 2.53
Output dim: 0, lower bound: -20.9023832, upper bound: 20.9267171
NS_B1_A2_A1_B2_B2_B2_B1, status: Status.UNKNOWN, split count: 7, time: 2.53
Output dim: 0, lower bound: -20.9422678, upper bound: 20.9484328
NS_B1_A2_A1_B2_B2_B2_B2, status: Status.UNKNOWN, split count: 7, time: 2.53
Output dim: 0, lower bound: -20.9476371, upper bound: 20.9513798
NS_B2_B1_A1_A2_A1_B1_A1, status: Status.UNKNOWN, split count: 7, time: 2.53
Output dim: 0, lower bound: -20.9607707, upper bound: 20.9611716
NS_B2_B1_A1_A2_A1_B1_A2, status: Status.UNKNOWN, split count: 7, time: 2.53
Output dim: 0, lower bound: -20.9607707, upper bound: 20.9611716
NS_B2_B1_A1_A2_A1_B2_A1, status: Status.UNKNOWN, split count: 7, time: 2.53
Output dim: 0, lower bound: -20.9609784, upper bound: 20.9620561
NS_B2_B1_A1_A2_A1_B2_A2, status: Status.UNKNOWN, split count: 7, time: 2.53
Output dim: 0, lower bound: -20.9609784, upper bound: 20.9620561
NS_B2_B1_A1_A2_A2_B1_A1, status: Status.UNKNOWN, split count: 7, time: 2.53
Output dim: 0, lower bound: -20.9603972, upper bound: 20.9607285
NS_B2_B1_A1_A2_A2_B1_A2, status: Status.UNKNOWN, split count: 7, time: 2.53
Output dim: 0, lower bound: -20.9608904, upper bound: 20.9608904
NS_B2_B1_A1_A2_A2_B2_A1, status: Status.UNKNOWN, split count: 7, time: 2.53
Output dim: 0, lower bound: -20.9603972, upper bound: 20.9607285
NS_B2_B1_A1_A2_A2_B2_A2, status: Status.UNKNOWN, split count: 7, time: 2.53
Output dim: 0, lower bound: -20.9608904, upper bound: 20.9608904
NS_B2_B1_A2_B2_A1_B1_A1, status: Status.UNKNOWN, split count: 7, time: 2.53
Output dim: 0, lower bound: -20.9575660, upper bound: 20.9601174
NS_B2_B1_A2_B2_A1_B1_A2, status: Status.UNKNOWN, split count: 7, time: 2.53
Output dim: 0, lower bound: -20.9573158, upper bound: 20.9593464
NS_B2_B1_A2_B2_A1_B2_A1, status: Status.UNKNOWN, split count: 7, time: 2.53
Output dim: 0, lower bound: -20.9536552, upper bound: 20.9570127
NS_B2_B1_A2_B2_A1_B2_A2, status: Status.UNKNOWN, split count: 7, time: 2.53
Output dim: 0, lower bound: -20.9579247, upper bound: 20.9604336
NS_B2_B1_A2_B2_A2_A2_B1, status: Status.UNKNOWN, split count: 7, time: 2.53
Output dim: 0, lower bound: -20.5282947, upper bound: 20.9498387
NS_B2_B1_A2_B2_A2_A2_B2, status: Status.UNKNOWN, split count: 7, time: 2.53
Output dim: 0, lower bound: -20.4996329, upper bound: 20.9502924
NS_B2_B2_A1_B1_A1_B1_B1, status: Status.UNKNOWN, split count: 7, time: 2.53
Output dim: 0, lower bound: -20.9443701, upper bound: 20.9500642
NS_B2_B2_A1_B1_A1_B1_B2, status: Status.UNKNOWN, split count: 7, time: 2.53
Output dim: 0, lower bound: -20.9629728, upper bound: 20.9625913
NS_B2_B2_A1_B1_A1_B2_B1, status: Status.UNKNOWN, split count: 7, time: 2.53
Output dim: 0, lower bound: -20.9442458, upper bound: 20.9503509
NS_B2_B2_A1_B1_A1_B2_B2, status: Status.UNKNOWN, split count: 7, time: 2.53
Output dim: 0, lower bound: -20.9640653, upper bound: 20.9646005
NS_B2_B2_A1_B1_A2_A1_B1, status: Status.UNKNOWN, split count: 7, time: 2.53
Output dim: 0, lower bound: -20.9384564, upper bound: 20.9264940
NS_B2_B2_A1_B1_A2_A1_B2, status: Status.UNKNOWN, split count: 7, time: 2.53
Output dim: 0, lower bound: -20.9390731, upper bound: 20.9274296
NS_B2_B2_A1_B1_A2_A2_B1, status: Status.UNKNOWN, split count: 7, time: 2.53
Output dim: 0, lower bound: -20.9504192, upper bound: 20.9458066
NS_B2_B2_A1_B1_A2_A2_B2, status: Status.UNKNOWN, split count: 7, time: 2.53
Output dim: 0, lower bound: -20.9390731, upper bound: 20.9479840
NS_B2_B2_A1_B2_A1_B1_B1, status: Status.UNKNOWN, split count: 7, time: 2.53
Output dim: 0, lower bound: -20.9446852, upper bound: 20.9503422
NS_B2_B2_A1_B2_A1_B1_B2, status: Status.UNKNOWN, split count: 7, time: 2.53
Output dim: 0, lower bound: -20.9460557, upper bound: 20.9511217
NS_B2_B2_A1_B2_A1_B2_B1, status: Status.UNKNOWN, split count: 7, time: 2.53
Output dim: 0, lower bound: -20.9449710, upper bound: 20.9514500
NS_B2_B2_A1_B2_A1_B2_B2, status: Status.UNKNOWN, split count: 7, time: 2.53
Output dim: 0, lower bound: -20.9523718, upper bound: 20.9564293
NS_B2_B2_A1_B2_A2_B1_B1, status: Status.UNKNOWN, split count: 7, time: 2.53
Output dim: 0, lower bound: -20.9273829, upper bound: 20.9235466
NS_B2_B2_A1_B2_A2_B1_B2, status: Status.UNKNOWN, split count: 7, time: 2.53
Output dim: 0, lower bound: -20.9313827, upper bound: 20.9330553
NS_B2_B2_A1_B2_A2_B2_B1, status: Status.UNKNOWN, split count: 7, time: 2.53
Output dim: 0, lower bound: -20.9272313, upper bound: 20.9235597
NS_B2_B2_A1_B2_A2_B2_B2, status: Status.UNKNOWN, split count: 7, time: 2.53
Output dim: 0, lower bound: -20.9404709, upper bound: 20.9393947
NS_B2_B2_A2_B1_A1_A1_B1, status: Status.UNKNOWN, split count: 7, time: 2.53
Output dim: 0, lower bound: -20.9574820, upper bound: 20.9539224
NS_B2_B2_A2_B1_A1_A1_B2, status: Status.UNKNOWN, split count: 7, time: 2.53
Output dim: 0, lower bound: -20.9610848, upper bound: 20.9582666
NS_B2_B2_A2_B1_A1_A2_B1, status: Status.UNKNOWN, split count: 7, time: 2.53
Output dim: 0, lower bound: -20.9574820, upper bound: 20.9539224
NS_B2_B2_A2_B1_A1_A2_B2, status: Status.UNKNOWN, split count: 7, time: 2.53
Output dim: 0, lower bound: -20.9610848, upper bound: 20.9582666
NS_B2_B2_A2_B1_A2_A1_A1, status: Status.UNKNOWN, split count: 7, time: 2.53
Output dim: 0, lower bound: -20.9481945, upper bound: 20.9453032
NS_B2_B2_A2_B1_A2_A1_A2, status: Status.UNKNOWN, split count: 7, time: 2.53
Output dim: 0, lower bound: -20.9569857, upper bound: 20.9544784
NS_B2_B2_A2_B1_A2_A2_B1, status: Status.UNKNOWN, split count: 7, time: 2.53
Output dim: 0, lower bound: -20.9493107, upper bound: 20.9467545
NS_B2_B2_A2_B1_A2_A2_B2, status: Status.UNKNOWN, split count: 7, time: 2.53
Output dim: 0, lower bound: -20.9424550, upper bound: 20.9544784

## BFS NS instance: NS_B1_A1_A1_A1_B1_B1_A1

### Backsubstitution after applying NS history:
0: -6.0033531, 16.7872543, -6.0846200, 16.9946613, -22.9980145, 22.8718739
1: -15.1282873, 25.5919876, -15.3345718, 25.8991718, -41.0274544, 40.9265518
2: -10.4942646, 23.2998066, -10.6436501, 23.5850830, -34.0793457, 33.9434509
3: -16.2143669, 28.3640823, -16.4320755, 28.7094727, -44.9238319, 44.7961578
4: -14.6194534, 28.8417206, -14.8261871, 29.1902122, -43.8096619, 43.6679039

Time for backsubstitution: 0.82 seconds

### NS candidates at layer 1
type: B, layer: 1, pos: 12
type: A, layer: 1, pos: 12
type: B, layer: 1, pos: 15
type: A, layer: 1, pos: 15
type: A, layer: 1, pos: 28
type: B, layer: 1, pos: 28
type: B, layer: 1, pos: 6
type: A, layer: 1, pos: 6
type: A, layer: 1, pos: 24
type: B, layer: 1, pos: 24
type: B, layer: 1, pos: 9
type: A, layer: 1, pos: 37
type: B, layer: 1, pos: 37
type: B, layer: 1, pos: 8
type: A, layer: 1, pos: 8
type: B, layer: 1, pos: 39
type: B, layer: 1, pos: 38
type: A, layer: 1, pos: 38
type: A, layer: 1, pos: 39
type: B, layer: 1, pos: 48
type: A, layer: 1, pos: 48
type: A, layer: 1, pos: 20
type: B, layer: 1, pos: 20
type: B, layer: 1, pos: 26
type: A, layer: 1, pos: 26
type: B, layer: 1, pos: 41
type: A, layer: 1, pos: 41
type: A, layer: 1, pos: 5
type: A, layer: 1, pos: 0
type: B, layer: 1, pos: 0
type: B, layer: 1, pos: 5
type: B, layer: 1, pos: 31
type: A, layer: 1, pos: 31

Time for candidate selection: 0.07 seconds

### Candidate
type: B, layer: 1, pos: 12

### Candidate
type: A, layer: 1, pos: 12

### Candidate
type: B, layer: 1, pos: 15

### Candidate
type: A, layer: 1, pos: 15

### Candidate
type: A, layer: 1, pos: 28

### Candidate
type: B, layer: 1, pos: 28

### Candidate
type: B, layer: 1, pos: 6

### Candidate
type: A, layer: 1, pos: 6

### Candidate
type: A, layer: 1, pos: 24

### Candidate
type: B, layer: 1, pos: 24

### Candidate
type: B, layer: 1, pos: 9

## Relational analysis of NS_B1_A1_A1_A1_B1_B1_A1_B1

### Relational analysis result of NS_B1_A1_A1_A1_B1_B1_A1_B1
Status: Status.UNKNOWN
Output dim: 0, lower bound: -20.9256445, upper bound: 20.9256445
time: 0.57 seconds

## Relational analysis of NS_B1_A1_A1_A1_B1_B1_A1_B2

### Relational analysis result of NS_B1_A1_A1_A1_B1_B1_A1_B2
Status: Status.UNKNOWN
Output dim: 0, lower bound: -20.9256445, upper bound: 20.9397612
time: 0.60 seconds

## BFS NS instance: NS_B1_A1_A1_A1_B1_B1_A2

### Backsubstitution after applying NS history:
0: -6.0468984, 16.8671551, -6.0347042, 16.8611584, -22.9080544, 22.9018593
1: -15.3067408, 25.6188679, -15.2117434, 25.6982460, -41.0049858, 40.8306046
2: -10.6000986, 23.3775158, -10.5563469, 23.4035225, -34.0036163, 33.9338608
3: -16.3600082, 28.4442596, -16.2995052, 28.4855881, -44.8455963, 44.7437668
4: -14.6819715, 28.9659004, -14.7020760, 28.9674358, -43.6494026, 43.6679764

Time for backsubstitution: 0.81 seconds

### NS candidates at layer 1
type: A, layer: 1, pos: 12
type: B, layer: 1, pos: 12
type: B, layer: 1, pos: 15
type: A, layer: 1, pos: 15
type: A, layer: 1, pos: 28
type: B, layer: 1, pos: 28
type: A, layer: 1, pos: 6
type: B, layer: 1, pos: 6
type: A, layer: 1, pos: 24
type: B, layer: 1, pos: 24
type: A, layer: 1, pos: 37
type: B, layer: 1, pos: 8
type: B, layer: 1, pos: 37
type: A, layer: 1, pos: 8
type: B, layer: 1, pos: 39
type: B, layer: 1, pos: 48
type: A, layer: 1, pos: 39
type: B, layer: 1, pos: 38
type: A, layer: 1, pos: 38
type: B, layer: 1, pos: 9
type: A, layer: 1, pos: 48
type: A, layer: 1, pos: 20
type: B, layer: 1, pos: 20
type: B, layer: 1, pos: 26
type: A, layer: 1, pos: 26
type: B, layer: 1, pos: 41
type: A, layer: 1, pos: 41
type: A, layer: 1, pos: 5
type: A, layer: 1, pos: 0
type: B, layer: 1, pos: 0
type: B, layer: 1, pos: 5
type: A, layer: 1, pos: 31
type: B, layer: 1, pos: 31

Time for candidate selection: 0.07 seconds

### Candidate
type: A, layer: 1, pos: 12

### Candidate
type: B, layer: 1, pos: 12

### Candidate
type: B, layer: 1, pos: 15

### Candidate
type: A, layer: 1, pos: 15

### Candidate
type: A, layer: 1, pos: 28

### Candidate
type: B, layer: 1, pos: 28

### Candidate
type: A, layer: 1, pos: 6

### Candidate
type: B, layer: 1, pos: 6

### Candidate
type: A, layer: 1, pos: 24

### Candidate
type: B, layer: 1, pos: 24

### Candidate
type: A, layer: 1, pos: 37

## Relational analysis of NS_B1_A1_A1_A1_B1_B1_A2_A1
Optimization infeasible because this subproblem isn't reachable.

### Candidate
type: B, layer: 1, pos: 8

## Relational analysis of NS_B1_A1_A1_A1_B1_B1_A2_B1

### Relational analysis result of NS_B1_A1_A1_A1_B1_B1_A2_B1
Status: Status.UNKNOWN
Output dim: 0, lower bound: -20.9446710, upper bound: 20.9619725
time: 0.61 seconds

## Relational analysis of NS_B1_A1_A1_A1_B1_B1_A2_B2

### Relational analysis result of NS_B1_A1_A1_A1_B1_B1_A2_B2
Status: Status.UNKNOWN
Output dim: 0, lower bound: -20.9636160, upper bound: 20.9636160
time: 0.59 seconds

## BFS NS instance: NS_B1_A1_A1_A1_B1_B2_A1

### Backsubstitution after applying NS history:
0: -5.4051991, 15.2213621, -6.3553233, 17.8177166, -23.2229156, 21.5766850
1: -13.5574827, 23.3508892, -15.9844027, 27.1184063, -40.6758881, 39.3352890
2: -9.3942366, 21.1592789, -11.1476793, 24.6566486, -34.0508842, 32.3069534
3: -14.5722885, 25.8379803, -17.1623230, 30.0475502, -44.6198387, 43.0002975
4: -13.1313391, 26.1407356, -15.5392513, 30.5415058, -43.6728439, 41.6799736

Time for backsubstitution: 0.81 seconds

### NS candidates at layer 1
type: B, layer: 1, pos: 12
type: A, layer: 1, pos: 12
type: A, layer: 1, pos: 15
type: B, layer: 1, pos: 15
type: A, layer: 1, pos: 28
type: B, layer: 1, pos: 9
type: B, layer: 1, pos: 28
type: B, layer: 1, pos: 6
type: A, layer: 1, pos: 9
type: B, layer: 1, pos: 39
type: A, layer: 1, pos: 24
type: B, layer: 1, pos: 24
type: A, layer: 1, pos: 37
type: A, layer: 1, pos: 20
type: B, layer: 1, pos: 37
type: B, layer: 1, pos: 8
type: A, layer: 1, pos: 38
type: A, layer: 1, pos: 6
type: A, layer: 1, pos: 48
type: B, layer: 1, pos: 38
type: A, layer: 1, pos: 39
type: B, layer: 1, pos: 20
type: B, layer: 1, pos: 48
type: B, layer: 1, pos: 26
type: A, layer: 1, pos: 26
type: A, layer: 1, pos: 41
type: A, layer: 1, pos: 0
type: B, layer: 1, pos: 41
type: B, layer: 1, pos: 31
type: A, layer: 1, pos: 31
type: B, layer: 1, pos: 0
type: B, layer: 1, pos: 5
type: A, layer: 1, pos: 5

Time for candidate selection: 0.07 seconds

### Candidate
type: B, layer: 1, pos: 12

### Candidate
type: A, layer: 1, pos: 12

### Candidate
type: A, layer: 1, pos: 15

### Candidate
type: B, layer: 1, pos: 15

### Candidate
type: A, layer: 1, pos: 28

### Candidate
type: B, layer: 1, pos: 9

### Candidate
type: B, layer: 1, pos: 28

### Candidate
type: B, layer: 1, pos: 6

### Candidate
type: A, layer: 1, pos: 9

### Candidate
type: B, layer: 1, pos: 39

### Candidate
type: A, layer: 1, pos: 24

### Candidate
type: B, layer: 1, pos: 24

### Candidate
type: A, layer: 1, pos: 37

### Candidate
type: A, layer: 1, pos: 20

## Relational analysis of NS_B1_A1_A1_A1_B1_B2_A1_A1
Optimization infeasible because this subproblem isn't reachable.

### Candidate
type: B, layer: 1, pos: 37

### Candidate
type: B, layer: 1, pos: 8

## Relational analysis of NS_B1_A1_A1_A1_B1_B2_A1_B1

### Relational analysis result of NS_B1_A1_A1_A1_B1_B2_A1_B1
Status: Status.UNKNOWN
Output dim: 0, lower bound: -20.9577507, upper bound: 20.9549928
time: 0.56 seconds

## Relational analysis of NS_B1_A1_A1_A1_B1_B2_A1_B2

### Relational analysis result of NS_B1_A1_A1_A1_B1_B2_A1_B2
Status: Status.UNKNOWN
Output dim: 0, lower bound: -20.9577507, upper bound: 20.9549928
time: 0.62 seconds

## BFS NS instance: NS_B1_A1_A1_A1_B1_B2_A2

### Backsubstitution after applying NS history:
0: -6.0435653, 16.8827038, -6.7040877, 18.7079010, -24.7514610, 23.5867882
1: -15.2276602, 25.7330322, -16.8718414, 28.4157085, -43.6433640, 42.6048737
2: -10.5695868, 23.4322605, -11.7658329, 25.8589211, -36.4285088, 35.1980896
3: -16.3202248, 28.5254173, -18.1079674, 31.5177822, -47.8380051, 46.6333847
4: -14.7273836, 28.9999886, -16.3970699, 32.0241547, -46.7515373, 45.3970490

Time for backsubstitution: 0.81 seconds

### NS candidates at layer 1
type: B, layer: 1, pos: 12
type: A, layer: 1, pos: 12
type: A, layer: 1, pos: 15
type: B, layer: 1, pos: 15
type: B, layer: 1, pos: 6
type: B, layer: 1, pos: 28
type: A, layer: 1, pos: 28
type: B, layer: 1, pos: 24
type: A, layer: 1, pos: 24
type: B, layer: 1, pos: 39
type: B, layer: 1, pos: 9
type: A, layer: 1, pos: 37
type: A, layer: 1, pos: 9
type: A, layer: 1, pos: 6
type: B, layer: 1, pos: 37
type: B, layer: 1, pos: 8
type: A, layer: 1, pos: 38
type: A, layer: 1, pos: 48
type: B, layer: 1, pos: 38
type: B, layer: 1, pos: 48
type: A, layer: 1, pos: 39
type: A, layer: 1, pos: 20
type: B, layer: 1, pos: 26
type: A, layer: 1, pos: 26
type: B, layer: 1, pos: 20
type: A, layer: 1, pos: 41
type: B, layer: 1, pos: 41
type: B, layer: 1, pos: 5
type: A, layer: 1, pos: 0
type: B, layer: 1, pos: 31
type: B, layer: 1, pos: 0
type: A, layer: 1, pos: 31
type: A, layer: 1, pos: 5

Time for candidate selection: 0.07 seconds

### Candidate
type: B, layer: 1, pos: 12

### Candidate
type: A, layer: 1, pos: 12

### Candidate
type: A, layer: 1, pos: 15

### Candidate
type: B, layer: 1, pos: 15

### Candidate
type: B, layer: 1, pos: 6

### Candidate
type: B, layer: 1, pos: 28

### Candidate
type: A, layer: 1, pos: 28

### Candidate
type: B, layer: 1, pos: 24

### Candidate
type: A, layer: 1, pos: 24

### Candidate
type: B, layer: 1, pos: 39

### Candidate
type: B, layer: 1, pos: 9

### Candidate
type: A, layer: 1, pos: 37

### Candidate
type: A, layer: 1, pos: 9

### Candidate
type: A, layer: 1, pos: 6

### Candidate
type: B, layer: 1, pos: 37

### Candidate
type: B, layer: 1, pos: 8

## Relational analysis of NS_B1_A1_A1_A1_B1_B2_A2_B1

### Relational analysis result of NS_B1_A1_A1_A1_B1_B2_A2_B1
Status: Status.UNKNOWN
Output dim: 0, lower bound: -20.9577507, upper bound: 20.9549928
time: 0.55 seconds

## Relational analysis of NS_B1_A1_A1_A1_B1_B2_A2_B2

### Relational analysis result of NS_B1_A1_A1_A1_B1_B2_A2_B2
Status: Status.UNKNOWN
Output dim: 0, lower bound: -20.9577507, upper bound: 20.9549928
time: 0.55 seconds

## BFS NS instance: NS_B1_A1_A1_A1_B2_B1_B1

### Backsubstitution after applying NS history:
0: -6.0846200, 16.9946613, -6.1001744, 16.9544849, -23.0391045, 23.0948334
1: -15.3345718, 25.8991718, -15.2073040, 25.9390106, -41.2735825, 41.1064682
2: -10.6436501, 23.5850830, -10.6146946, 23.5968227, -34.2404594, 34.1997757
3: -16.4320755, 28.7094727, -16.4179573, 28.7281609, -45.1602364, 45.1274147
4: -14.8261871, 29.1902122, -14.9846592, 29.1319141, -43.9580994, 44.1748619

Time for backsubstitution: 0.82 seconds

### NS candidates at layer 1
type: B, layer: 1, pos: 12
type: A, layer: 1, pos: 12
type: B, layer: 1, pos: 15
type: A, layer: 1, pos: 15
type: A, layer: 1, pos: 28
type: B, layer: 1, pos: 28
type: B, layer: 1, pos: 6
type: A, layer: 1, pos: 24
type: A, layer: 1, pos: 6
type: B, layer: 1, pos: 39
type: B, layer: 1, pos: 24
type: A, layer: 1, pos: 37
type: B, layer: 1, pos: 48
type: A, layer: 1, pos: 9
type: B, layer: 1, pos: 37
type: B, layer: 1, pos: 8
type: A, layer: 1, pos: 8
type: A, layer: 1, pos: 38
type: B, layer: 1, pos: 38
type: A, layer: 1, pos: 20
type: A, layer: 1, pos: 39
type: A, layer: 1, pos: 48
type: B, layer: 1, pos: 26
type: A, layer: 1, pos: 26
type: B, layer: 1, pos: 20
type: B, layer: 1, pos: 41
type: A, layer: 1, pos: 0
type: A, layer: 1, pos: 41
type: A, layer: 1, pos: 5
type: B, layer: 1, pos: 31
type: B, layer: 1, pos: 0
type: B, layer: 1, pos: 5
type: A, layer: 1, pos: 31

Time for candidate selection: 0.07 seconds

### Candidate
type: B, layer: 1, pos: 12

### Candidate
type: A, layer: 1, pos: 12

### Candidate
type: B, layer: 1, pos: 15

### Candidate
type: A, layer: 1, pos: 15

### Candidate
type: A, layer: 1, pos: 28

### Candidate
type: B, layer: 1, pos: 28

### Candidate
type: B, layer: 1, pos: 6

### Candidate
type: A, layer: 1, pos: 24

### Candidate
type: A, layer: 1, pos: 6

### Candidate
type: B, layer: 1, pos: 39

### Candidate
type: B, layer: 1, pos: 24

### Candidate
type: A, layer: 1, pos: 37

## Relational analysis of NS_B1_A1_A1_A1_B2_B1_B1_A1
Optimization infeasible because this subproblem isn't reachable.

### Candidate
type: B, layer: 1, pos: 48

## Relational analysis of NS_B1_A1_A1_A1_B2_B1_B1_B1
Optimization infeasible because this subproblem isn't reachable.

### Candidate
type: A, layer: 1, pos: 9

## Relational analysis of NS_B1_A1_A1_A1_B2_B1_B1_A1

### Relational analysis result of NS_B1_A1_A1_A1_B2_B1_B1_A1
Status: Status.UNKNOWN
Output dim: 0, lower bound: -20.9169596, upper bound: 20.9227605
time: 0.56 seconds

## Relational analysis of NS_B1_A1_A1_A1_B2_B1_B1_A2

### Relational analysis result of NS_B1_A1_A1_A1_B2_B1_B1_A2
Status: Status.UNKNOWN
Output dim: 0, lower bound: -20.9169596, upper bound: 20.9430163
time: 1.02 seconds

## BFS NS instance: NS_B1_A1_A1_A1_B2_B1_B2

### Backsubstitution after applying NS history:
0: -6.0347042, 16.8611584, -6.0741062, 16.8618240, -22.8965282, 22.9352646
1: -15.2117434, 25.6982460, -15.2196198, 25.7022247, -40.9139633, 40.9178658
2: -10.5563469, 23.4035225, -10.6018047, 23.4339085, -33.9902573, 34.0053177
3: -16.2995052, 28.4855881, -16.3732090, 28.5004902, -44.7999954, 44.8587952
4: -14.7020760, 28.9674358, -14.8689833, 28.9606113, -43.6626892, 43.8364182

Time for backsubstitution: 0.82 seconds

### NS candidates at layer 1
type: B, layer: 1, pos: 12
type: A, layer: 1, pos: 12
type: B, layer: 1, pos: 15
type: A, layer: 1, pos: 15
type: B, layer: 1, pos: 28
type: A, layer: 1, pos: 28
type: B, layer: 1, pos: 6
type: A, layer: 1, pos: 24
type: A, layer: 1, pos: 6
type: B, layer: 1, pos: 39
type: B, layer: 1, pos: 24
type: A, layer: 1, pos: 37
type: B, layer: 1, pos: 37
type: B, layer: 1, pos: 48
type: A, layer: 1, pos: 8
type: B, layer: 1, pos: 8
type: A, layer: 1, pos: 38
type: B, layer: 1, pos: 38
type: A, layer: 1, pos: 9
type: A, layer: 1, pos: 20
type: A, layer: 1, pos: 39
type: A, layer: 1, pos: 48
type: B, layer: 1, pos: 26
type: A, layer: 1, pos: 26
type: B, layer: 1, pos: 20
type: B, layer: 1, pos: 41
type: A, layer: 1, pos: 41
type: A, layer: 1, pos: 0
type: A, layer: 1, pos: 5
type: B, layer: 1, pos: 0
type: B, layer: 1, pos: 5
type: B, layer: 1, pos: 31
type: A, layer: 1, pos: 31

Time for candidate selection: 0.07 seconds

### Candidate
type: B, layer: 1, pos: 12

### Candidate
type: A, layer: 1, pos: 12

### Candidate
type: B, layer: 1, pos: 15

### Candidate
type: A, layer: 1, pos: 15

### Candidate
type: B, layer: 1, pos: 28

### Candidate
type: A, layer: 1, pos: 28

### Candidate
type: B, layer: 1, pos: 6

### Candidate
type: A, layer: 1, pos: 24

### Candidate
type: A, layer: 1, pos: 6

### Candidate
type: B, layer: 1, pos: 39

### Candidate
type: B, layer: 1, pos: 24

### Candidate
type: A, layer: 1, pos: 37

## Relational analysis of NS_B1_A1_A1_A1_B2_B1_B2_A1
Optimization infeasible because this subproblem isn't reachable.

### Candidate
type: B, layer: 1, pos: 37

### Candidate
type: B, layer: 1, pos: 48

## Relational analysis of NS_B1_A1_A1_A1_B2_B1_B2_B1
Optimization infeasible because this subproblem isn't reachable.

### Candidate
type: A, layer: 1, pos: 8

## Relational analysis of NS_B1_A1_A1_A1_B2_B1_B2_A1

### Relational analysis result of NS_B1_A1_A1_A1_B2_B1_B2_A1
Status: Status.UNKNOWN
Output dim: 0, lower bound: -20.9447177, upper bound: 20.9503799
time: 1.13 seconds

## Relational analysis of NS_B1_A1_A1_A1_B2_B1_B2_A2

### Relational analysis result of NS_B1_A1_A1_A1_B2_B1_B2_A2
Status: Status.UNKNOWN
Output dim: 0, lower bound: -20.9463675, upper bound: 20.9515024
time: 0.63 seconds

## BFS NS instance: NS_B1_A1_A1_A1_B2_B2_B1

### Backsubstitution after applying NS history:
0: -6.0846200, 16.9946613, -6.8861532, 19.0689354, -25.1535549, 23.8808098
1: -15.3345718, 25.8991718, -17.1600342, 29.0932503, -44.4278221, 43.0591888
2: -10.6436501, 23.5850830, -12.0153913, 26.4363899, -37.0800285, 35.6004753
3: -16.4320755, 28.7094727, -18.5391426, 32.2522087, -48.6842842, 47.2486076
4: -14.8261871, 29.1902122, -16.9774246, 32.6610565, -47.4872437, 46.1676292

Time for backsubstitution: 0.83 seconds

### NS candidates at layer 1
type: B, layer: 1, pos: 12
type: A, layer: 1, pos: 12
type: A, layer: 1, pos: 15
type: B, layer: 1, pos: 15
type: B, layer: 1, pos: 6
type: B, layer: 1, pos: 28
type: A, layer: 1, pos: 28
type: B, layer: 1, pos: 39
type: A, layer: 1, pos: 24
type: B, layer: 1, pos: 24
type: A, layer: 1, pos: 37
type: B, layer: 1, pos: 48
type: A, layer: 1, pos: 38
type: A, layer: 1, pos: 9
type: B, layer: 1, pos: 37
type: A, layer: 1, pos: 8
type: B, layer: 1, pos: 8
type: A, layer: 1, pos: 6
type: B, layer: 1, pos: 38
type: A, layer: 1, pos: 20
type: A, layer: 1, pos: 48
type: B, layer: 1, pos: 26
type: A, layer: 1, pos: 26
type: A, layer: 1, pos: 39
type: B, layer: 1, pos: 20
type: A, layer: 1, pos: 41
type: B, layer: 1, pos: 31
type: B, layer: 1, pos: 41
type: A, layer: 1, pos: 0
type: A, layer: 1, pos: 5
type: B, layer: 1, pos: 5
type: A, layer: 1, pos: 31
type: B, layer: 1, pos: 0

Time for candidate selection: 0.07 seconds

### Candidate
type: B, layer: 1, pos: 12

### Candidate
type: A, layer: 1, pos: 12

### Candidate
type: A, layer: 1, pos: 15

### Candidate
type: B, layer: 1, pos: 15

### Candidate
type: B, layer: 1, pos: 6

### Candidate
type: B, layer: 1, pos: 28

### Candidate
type: A, layer: 1, pos: 28

### Candidate
type: B, layer: 1, pos: 39

### Candidate
type: A, layer: 1, pos: 24

### Candidate
type: B, layer: 1, pos: 24

### Candidate
type: A, layer: 1, pos: 37

## Relational analysis of NS_B1_A1_A1_A1_B2_B2_B1_A1
Optimization infeasible because this subproblem isn't reachable.

### Candidate
type: B, layer: 1, pos: 48

## Relational analysis of NS_B1_A1_A1_A1_B2_B2_B1_B1
Optimization infeasible because this subproblem isn't reachable.

### Candidate
type: A, layer: 1, pos: 38

## Relational analysis of NS_B1_A1_A1_A1_B2_B2_B1_A1
Optimization infeasible because this subproblem isn't reachable.

### Candidate
type: A, layer: 1, pos: 9

## Relational analysis of NS_B1_A1_A1_A1_B2_B2_B1_A1

### Relational analysis result of NS_B1_A1_A1_A1_B2_B2_B1_A1
Status: Status.UNKNOWN
Output dim: 0, lower bound: -20.8914581, upper bound: 20.8973311
time: 0.56 seconds

## Relational analysis of NS_B1_A1_A1_A1_B2_B2_B1_A2

### Relational analysis result of NS_B1_A1_A1_A1_B2_B2_B1_A2
Status: Status.UNKNOWN
Output dim: 0, lower bound: -20.8914581, upper bound: 20.9286097
time: 0.61 seconds

## BFS NS instance: NS_B1_A1_A1_A1_B2_B2_B2

### Backsubstitution after applying NS history:
0: -6.0347042, 16.8611584, -6.8079839, 18.8318996, -24.8666039, 23.6691418
1: -15.2117434, 25.6982460, -17.0214043, 28.6349945, -43.8467369, 42.7196503
2: -10.5563469, 23.4035225, -11.9100714, 26.0820293, -36.6383743, 35.3135872
3: -16.2995052, 28.4855881, -18.3529911, 31.7761555, -48.0756607, 46.8385773
4: -14.7020760, 28.9674358, -16.7461910, 32.2454262, -46.9475021, 45.7136230

Time for backsubstitution: 0.83 seconds

### NS candidates at layer 1
type: B, layer: 1, pos: 12
type: A, layer: 1, pos: 12
type: A, layer: 1, pos: 15
type: B, layer: 1, pos: 6
type: B, layer: 1, pos: 15
type: B, layer: 1, pos: 28
type: B, layer: 1, pos: 39
type: A, layer: 1, pos: 28
type: B, layer: 1, pos: 24
type: A, layer: 1, pos: 24
type: A, layer: 1, pos: 37
type: A, layer: 1, pos: 38
type: A, layer: 1, pos: 8
type: B, layer: 1, pos: 48
type: B, layer: 1, pos: 37
type: B, layer: 1, pos: 8
type: A, layer: 1, pos: 6
type: B, layer: 1, pos: 38
type: A, layer: 1, pos: 20
type: A, layer: 1, pos: 9
type: A, layer: 1, pos: 48
type: B, layer: 1, pos: 26
type: A, layer: 1, pos: 26
type: A, layer: 1, pos: 39
type: B, layer: 1, pos: 20
type: A, layer: 1, pos: 41
type: B, layer: 1, pos: 41
type: A, layer: 1, pos: 0
type: B, layer: 1, pos: 31
type: A, layer: 1, pos: 5
type: B, layer: 1, pos: 5
type: B, layer: 1, pos: 0
type: A, layer: 1, pos: 31

Time for candidate selection: 0.07 seconds

### Candidate
type: B, layer: 1, pos: 12

### Candidate
type: A, layer: 1, pos: 12

### Candidate
type: A, layer: 1, pos: 15

### Candidate
type: B, layer: 1, pos: 6

### Candidate
type: B, layer: 1, pos: 15

### Candidate
type: B, layer: 1, pos: 28

### Candidate
type: B, layer: 1, pos: 39

### Candidate
type: A, layer: 1, pos: 28

### Candidate
type: B, layer: 1, pos: 24

### Candidate
type: A, layer: 1, pos: 24

### Candidate
type: A, layer: 1, pos: 37

## Relational analysis of NS_B1_A1_A1_A1_B2_B2_B2_A1
Optimization infeasible because this subproblem isn't reachable.

### Candidate
type: A, layer: 1, pos: 38

## Relational analysis of NS_B1_A1_A1_A1_B2_B2_B2_A1
Optimization infeasible because this subproblem isn't reachable.

### Candidate
type: A, layer: 1, pos: 8

## Relational analysis of NS_B1_A1_A1_A1_B2_B2_B2_A1

### Relational analysis result of NS_B1_A1_A1_A1_B2_B2_B2_A1
Status: Status.UNKNOWN
Output dim: 0, lower bound: -20.9209893, upper bound: 20.9353029
time: 0.58 seconds

## Relational analysis of NS_B1_A1_A1_A1_B2_B2_B2_A2

### Relational analysis result of NS_B1_A1_A1_A1_B2_B2_B2_A2
Status: Status.UNKNOWN
Output dim: 0, lower bound: -20.9212410, upper bound: 20.9354726
time: 0.93 seconds

## BFS NS instance: NS_B1_A1_A1_A2_B1_B1_A1

### Backsubstitution after applying NS history:
0: -6.1001744, 16.9544849, -6.0846200, 16.9946613, -23.0948353, 23.0391045
1: -15.2073040, 25.9390106, -15.3345718, 25.8991718, -41.1064682, 41.2735825
2: -10.6146946, 23.5968227, -10.6436501, 23.5850830, -34.1997757, 34.2404594
3: -16.4179573, 28.7281609, -16.4320755, 28.7094727, -45.1274147, 45.1602364
4: -14.9846592, 29.1319141, -14.8261871, 29.1902122, -44.1748619, 43.9580994

Time for backsubstitution: 0.84 seconds

### NS candidates at layer 1
type: A, layer: 1, pos: 12
type: B, layer: 1, pos: 12
type: A, layer: 1, pos: 15
type: B, layer: 1, pos: 15
type: B, layer: 1, pos: 28
type: A, layer: 1, pos: 28
type: A, layer: 1, pos: 6
type: B, layer: 1, pos: 24
type: B, layer: 1, pos: 6
type: A, layer: 1, pos: 39
type: A, layer: 1, pos: 24
type: B, layer: 1, pos: 37
type: A, layer: 1, pos: 48
type: B, layer: 1, pos: 9
type: A, layer: 1, pos: 37
type: A, layer: 1, pos: 8
type: B, layer: 1, pos: 8
type: B, layer: 1, pos: 38
type: A, layer: 1, pos: 38
type: B, layer: 1, pos: 20
type: B, layer: 1, pos: 39
type: B, layer: 1, pos: 48
type: A, layer: 1, pos: 26
type: B, layer: 1, pos: 26
type: A, layer: 1, pos: 20
type: A, layer: 1, pos: 41
type: B, layer: 1, pos: 0
type: B, layer: 1, pos: 41
type: B, layer: 1, pos: 5
type: A, layer: 1, pos: 31
type: A, layer: 1, pos: 0
type: A, layer: 1, pos: 5
type: B, layer: 1, pos: 31

Time for candidate selection: 0.07 seconds

### Candidate
type: A, layer: 1, pos: 12

### Candidate
type: B, layer: 1, pos: 12

### Candidate
type: A, layer: 1, pos: 15

### Candidate
type: B, layer: 1, pos: 15

### Candidate
type: B, layer: 1, pos: 28

### Candidate
type: A, layer: 1, pos: 28

### Candidate
type: A, layer: 1, pos: 6

### Candidate
type: B, layer: 1, pos: 24

### Candidate
type: B, layer: 1, pos: 6

### Candidate
type: A, layer: 1, pos: 39

### Candidate
type: A, layer: 1, pos: 24

### Candidate
type: B, layer: 1, pos: 37

## Relational analysis of NS_B1_A1_A1_A2_B1_B1_A1_B1
Optimization infeasible because this subproblem isn't reachable.

### Candidate
type: A, layer: 1, pos: 48

## Relational analysis of NS_B1_A1_A1_A2_B1_B1_A1_A1
Optimization infeasible because this subproblem isn't reachable.

### Candidate
type: B, layer: 1, pos: 9

## Relational analysis of NS_B1_A1_A1_A2_B1_B1_A1_B1

### Relational analysis result of NS_B1_A1_A1_A2_B1_B1_A1_B1
Status: Status.UNKNOWN
Output dim: 0, lower bound: -20.9227605, upper bound: 20.9169596
time: 0.95 seconds

## Relational analysis of NS_B1_A1_A1_A2_B1_B1_A1_B2

### Relational analysis result of NS_B1_A1_A1_A2_B1_B1_A1_B2
Status: Status.UNKNOWN
Output dim: 0, lower bound: -20.9227605, upper bound: 20.9328254
time: 0.62 seconds

## BFS NS instance: NS_B1_A1_A1_A2_B1_B1_A2

### Backsubstitution after applying NS history:
0: -6.0741062, 16.8618240, -6.0347042, 16.8611584, -22.9352646, 22.8965282
1: -15.2196198, 25.7022247, -15.2117434, 25.6982460, -40.9178658, 40.9139595
2: -10.6018047, 23.4339085, -10.5563469, 23.4035225, -34.0053177, 33.9902573
3: -16.3732090, 28.5004902, -16.2995052, 28.4855881, -44.8587952, 44.7999954
4: -14.8689833, 28.9606113, -14.7020760, 28.9674358, -43.8364182, 43.6626892

Time for backsubstitution: 0.84 seconds

### NS candidates at layer 1
type: A, layer: 1, pos: 12
type: B, layer: 1, pos: 12
type: A, layer: 1, pos: 15
type: B, layer: 1, pos: 15
type: A, layer: 1, pos: 28
type: B, layer: 1, pos: 28
type: A, layer: 1, pos: 6
type: B, layer: 1, pos: 24
type: B, layer: 1, pos: 6
type: A, layer: 1, pos: 39
type: A, layer: 1, pos: 24
type: B, layer: 1, pos: 37
type: A, layer: 1, pos: 37
type: A, layer: 1, pos: 48
type: B, layer: 1, pos: 8
type: A, layer: 1, pos: 8
type: B, layer: 1, pos: 38
type: A, layer: 1, pos: 38
type: B, layer: 1, pos: 9
type: B, layer: 1, pos: 20
type: B, layer: 1, pos: 39
type: B, layer: 1, pos: 48
type: A, layer: 1, pos: 26
type: B, layer: 1, pos: 26
type: A, layer: 1, pos: 20
type: A, layer: 1, pos: 41
type: B, layer: 1, pos: 41
type: B, layer: 1, pos: 0
type: B, layer: 1, pos: 5
type: A, layer: 1, pos: 0
type: A, layer: 1, pos: 5
type: A, layer: 1, pos: 31
type: B, layer: 1, pos: 31

Time for candidate selection: 0.07 seconds

### Candidate
type: A, layer: 1, pos: 12

### Candidate
type: B, layer: 1, pos: 12

### Candidate
type: A, layer: 1, pos: 15

### Candidate
type: B, layer: 1, pos: 15

### Candidate
type: A, layer: 1, pos: 28

### Candidate
type: B, layer: 1, pos: 28

### Candidate
type: A, layer: 1, pos: 6

### Candidate
type: B, layer: 1, pos: 24

### Candidate
type: B, layer: 1, pos: 6

### Candidate
type: A, layer: 1, pos: 39

### Candidate
type: A, layer: 1, pos: 24

### Candidate
type: B, layer: 1, pos: 37

## Relational analysis of NS_B1_A1_A1_A2_B1_B1_A2_B1
Optimization infeasible because this subproblem isn't reachable.

### Candidate
type: A, layer: 1, pos: 37

### Candidate
type: A, layer: 1, pos: 48

## Relational analysis of NS_B1_A1_A1_A2_B1_B1_A2_A1
Optimization infeasible because this subproblem isn't reachable.

### Candidate
type: B, layer: 1, pos: 8

## Relational analysis of NS_B1_A1_A1_A2_B1_B1_A2_B1

### Relational analysis result of NS_B1_A1_A1_A2_B1_B1_A2_B1
Status: Status.UNKNOWN
Output dim: 0, lower bound: -20.9503799, upper bound: 20.9447177
time: 0.61 seconds

## Relational analysis of NS_B1_A1_A1_A2_B1_B1_A2_B2

### Relational analysis result of NS_B1_A1_A1_A2_B1_B1_A2_B2
Status: Status.UNKNOWN
Output dim: 0, lower bound: -20.9515024, upper bound: 20.9463675
time: 0.62 seconds

## BFS NS instance: NS_B1_A1_A1_A2_B1_B2_A1

### Backsubstitution after applying NS history:
0: -6.1001744, 16.9544849, -6.7040877, 18.7079010, -24.8080750, 23.6585693
1: -15.2073040, 25.9390106, -16.8718414, 28.4157085, -43.6230087, 42.8108521
2: -10.6146946, 23.5968227, -11.7658329, 25.8589211, -36.4736176, 35.3626480
3: -16.4179573, 28.7281609, -18.1079674, 31.5177822, -47.9357300, 46.8361244
4: -14.9846592, 29.1319141, -16.3970699, 32.0241547, -47.0088120, 45.5289764

Time for backsubstitution: 0.84 seconds

### NS candidates at layer 1
type: B, layer: 1, pos: 12
type: A, layer: 1, pos: 12
type: A, layer: 1, pos: 15
type: B, layer: 1, pos: 15
type: B, layer: 1, pos: 28
type: B, layer: 1, pos: 6
type: B, layer: 1, pos: 24
type: A, layer: 1, pos: 28
type: A, layer: 1, pos: 6
type: A, layer: 1, pos: 39
type: A, layer: 1, pos: 48
type: A, layer: 1, pos: 24
type: B, layer: 1, pos: 37
type: A, layer: 1, pos: 37
type: B, layer: 1, pos: 9
type: A, layer: 1, pos: 8
type: A, layer: 1, pos: 38
type: B, layer: 1, pos: 8
type: B, layer: 1, pos: 38
type: B, layer: 1, pos: 39
type: B, layer: 1, pos: 20
type: A, layer: 1, pos: 26
type: B, layer: 1, pos: 48
type: B, layer: 1, pos: 26
type: A, layer: 1, pos: 20
type: B, layer: 1, pos: 5
type: B, layer: 1, pos: 0
type: A, layer: 1, pos: 41
type: B, layer: 1, pos: 41
type: A, layer: 1, pos: 0
type: A, layer: 1, pos: 31
type: B, layer: 1, pos: 31
type: A, layer: 1, pos: 5

Time for candidate selection: 0.07 seconds

### Candidate
type: B, layer: 1, pos: 12

### Candidate
type: A, layer: 1, pos: 12

### Candidate
type: A, layer: 1, pos: 15

### Candidate
type: B, layer: 1, pos: 15

### Candidate
type: B, layer: 1, pos: 28

### Candidate
type: B, layer: 1, pos: 6

### Candidate
type: B, layer: 1, pos: 24

### Candidate
type: A, layer: 1, pos: 28

### Candidate
type: A, layer: 1, pos: 6

### Candidate
type: A, layer: 1, pos: 39

## Relational analysis of NS_B1_A1_A1_A2_B1_B2_A1_A1
Optimization infeasible because this subproblem isn't reachable.

### Candidate
type: A, layer: 1, pos: 48

## Relational analysis of NS_B1_A1_A1_A2_B1_B2_A1_A1
Optimization infeasible because this subproblem isn't reachable.

### Candidate
type: A, layer: 1, pos: 24

### Candidate
type: B, layer: 1, pos: 37

## Relational analysis of NS_B1_A1_A1_A2_B1_B2_A1_B1
Optimization infeasible because this subproblem isn't reachable.

### Candidate
type: A, layer: 1, pos: 37

### Candidate
type: B, layer: 1, pos: 9

## Relational analysis of NS_B1_A1_A1_A2_B1_B2_A1_B1
Optimization infeasible because this subproblem isn't reachable.

### Candidate
type: A, layer: 1, pos: 8

## Relational analysis of NS_B1_A1_A1_A2_B1_B2_A1_A1

### Relational analysis result of NS_B1_A1_A1_A2_B1_B2_A1_A1
Status: Status.UNKNOWN
Output dim: 0, lower bound: -20.9371414, upper bound: 20.9213108
time: 0.54 seconds

## Relational analysis of NS_B1_A1_A1_A2_B1_B2_A1_A2

### Relational analysis result of NS_B1_A1_A1_A2_B1_B2_A1_A2
Status: Status.UNKNOWN
Output dim: 0, lower bound: -20.9371011, upper bound: 20.9209571
time: 0.54 seconds

## BFS NS instance: NS_B1_A1_A1_A2_B1_B2_A2

### Backsubstitution after applying NS history:
0: -6.0741062, 16.8618240, -6.6541419, 18.5776920, -24.6517963, 23.5159626
1: -15.2196198, 25.7022247, -16.7508907, 28.2207680, -43.4403839, 42.4531174
2: -10.6018047, 23.4339085, -11.6802349, 25.6811752, -36.2829742, 35.1141357
3: -16.3732090, 28.5004902, -17.9761639, 31.2994175, -47.6726265, 46.4766541
4: -14.8689833, 28.9606113, -16.2731323, 31.8075790, -46.6765594, 45.2337418

Time for backsubstitution: 0.84 seconds

### NS candidates at layer 1
type: B, layer: 1, pos: 12
type: A, layer: 1, pos: 12
type: A, layer: 1, pos: 15
type: B, layer: 1, pos: 15
type: B, layer: 1, pos: 28
type: B, layer: 1, pos: 6
type: A, layer: 1, pos: 28
type: B, layer: 1, pos: 24
type: A, layer: 1, pos: 6
type: A, layer: 1, pos: 24
type: A, layer: 1, pos: 39
type: B, layer: 1, pos: 37
type: A, layer: 1, pos: 37
type: A, layer: 1, pos: 48
type: A, layer: 1, pos: 8
type: B, layer: 1, pos: 8
type: A, layer: 1, pos: 38
type: B, layer: 1, pos: 39
type: B, layer: 1, pos: 38
type: B, layer: 1, pos: 9
type: B, layer: 1, pos: 20
type: B, layer: 1, pos: 48
type: A, layer: 1, pos: 26
type: B, layer: 1, pos: 26
type: A, layer: 1, pos: 20
type: B, layer: 1, pos: 5
type: A, layer: 1, pos: 41
type: B, layer: 1, pos: 0
type: B, layer: 1, pos: 41
type: A, layer: 1, pos: 0
type: B, layer: 1, pos: 31
type: A, layer: 1, pos: 31
type: A, layer: 1, pos: 5

Time for candidate selection: 0.07 seconds

### Candidate
type: B, layer: 1, pos: 12

### Candidate
type: A, layer: 1, pos: 12

### Candidate
type: A, layer: 1, pos: 15

### Candidate
type: B, layer: 1, pos: 15

### Candidate
type: B, layer: 1, pos: 28

### Candidate
type: B, layer: 1, pos: 6

### Candidate
type: A, layer: 1, pos: 28

### Candidate
type: B, layer: 1, pos: 24

### Candidate
type: A, layer: 1, pos: 6

### Candidate
type: A, layer: 1, pos: 24

### Candidate
type: A, layer: 1, pos: 39

## Relational analysis of NS_B1_A1_A1_A2_B1_B2_A2_A1
Optimization infeasible because this subproblem isn't reachable.

### Candidate
type: B, layer: 1, pos: 37

## Relational analysis of NS_B1_A1_A1_A2_B1_B2_A2_B1
Optimization infeasible because this subproblem isn't reachable.

### Candidate
type: A, layer: 1, pos: 37

### Candidate
type: A, layer: 1, pos: 48

## Relational analysis of NS_B1_A1_A1_A2_B1_B2_A2_A1
Optimization infeasible because this subproblem isn't reachable.

### Candidate
type: A, layer: 1, pos: 8

## Relational analysis of NS_B1_A1_A1_A2_B1_B2_A2_A1

### Relational analysis result of NS_B1_A1_A1_A2_B1_B2_A2_A1
Status: Status.UNKNOWN
Output dim: 0, lower bound: -20.9371414, upper bound: 20.9332980
time: 0.94 seconds

## Relational analysis of NS_B1_A1_A1_A2_B1_B2_A2_A2

### Relational analysis result of NS_B1_A1_A1_A2_B1_B2_A2_A2
Status: Status.UNKNOWN
Output dim: 0, lower bound: -20.9442875, upper bound: 20.9332745
time: 0.63 seconds

## BFS NS instance: NS_B1_A1_A1_A2_B2_B2_B1

### Backsubstitution after applying NS history:
0: -5.9061351, 16.4516106, -5.4940004, 15.3751049, -21.2812405, 21.9456062
1: -14.7086859, 25.2213764, -13.5434618, 23.7361774, -38.4448547, 38.7648392
2: -10.2757845, 22.9412823, -9.5379639, 21.5243587, -31.8001385, 32.4792404
3: -15.8900938, 27.9207497, -14.7336054, 26.2336273, -42.1237221, 42.6543503
4: -14.5270529, 28.3121128, -13.5832644, 26.5258331, -41.0528717, 41.8953743

Time for backsubstitution: 0.85 seconds

### NS candidates at layer 1
type: B, layer: 1, pos: 12
type: A, layer: 1, pos: 12
type: A, layer: 1, pos: 15
type: B, layer: 1, pos: 15
type: B, layer: 1, pos: 20
type: A, layer: 1, pos: 9
type: A, layer: 1, pos: 20
type: B, layer: 1, pos: 9
type: B, layer: 1, pos: 8
type: A, layer: 1, pos: 8
type: A, layer: 1, pos: 48
type: B, layer: 1, pos: 37
type: A, layer: 1, pos: 37
type: A, layer: 1, pos: 38
type: B, layer: 1, pos: 24
type: A, layer: 1, pos: 24
type: B, layer: 1, pos: 38
type: A, layer: 1, pos: 6
type: B, layer: 1, pos: 39
type: A, layer: 1, pos: 39
type: A, layer: 1, pos: 28
type: B, layer: 1, pos: 28
type: B, layer: 1, pos: 48
type: A, layer: 1, pos: 0
type: B, layer: 1, pos: 5
type: B, layer: 1, pos: 31
type: B, layer: 1, pos: 0
type: A, layer: 1, pos: 31
type: A, layer: 1, pos: 5
type: A, layer: 1, pos: 41
type: B, layer: 1, pos: 41
type: B, layer: 1, pos: 26
type: A, layer: 1, pos: 26

Time for candidate selection: 0.07 seconds

### Candidate
type: B, layer: 1, pos: 12

### Candidate
type: A, layer: 1, pos: 12

### Candidate
type: A, layer: 1, pos: 15

### Candidate
type: B, layer: 1, pos: 15

### Candidate
type: B, layer: 1, pos: 20

## Relational analysis of NS_B1_A1_A1_A2_B2_B2_B1_B1
Optimization infeasible because this subproblem isn't reachable.

### Candidate
type: A, layer: 1, pos: 9

## Relational analysis of NS_B1_A1_A1_A2_B2_B2_B1_A1

### Relational analysis result of NS_B1_A1_A1_A2_B2_B2_B1_A1
Status: Status.UNKNOWN
Output dim: 0, lower bound: -20.8997218, upper bound: 20.9000373
time: 0.61 seconds

## Relational analysis of NS_B1_A1_A1_A2_B2_B2_B1_A2

### Relational analysis result of NS_B1_A1_A1_A2_B2_B2_B1_A2
Status: Status.UNKNOWN
Output dim: 0, lower bound: -20.9083546, upper bound: 20.9152738
time: 0.68 seconds

## BFS NS instance: NS_B1_A2_A1_B1_B1_A1_B1

### Backsubstitution after applying NS history:
0: -5.2740431, 14.9444952, -4.3702459, 12.5205765, -17.7946167, 19.3147373
1: -13.1596384, 23.1045856, -10.7683678, 19.5335999, -32.6932373, 33.8729553
2: -9.1495724, 20.8888454, -7.5051541, 17.6466217, -26.7961941, 28.3939991
3: -14.1973553, 25.4861031, -11.7375555, 21.4977207, -35.6950760, 37.2236557
4: -12.8572388, 25.8054695, -10.7301178, 21.7384357, -34.5956726, 36.5355873

Time for backsubstitution: 0.84 seconds

### NS candidates at layer 1
type: A, layer: 1, pos: 12
type: B, layer: 1, pos: 12
type: B, layer: 1, pos: 28
type: B, layer: 1, pos: 45
type: A, layer: 1, pos: 28
type: A, layer: 1, pos: 45
type: A, layer: 1, pos: 24
type: B, layer: 1, pos: 48
type: A, layer: 1, pos: 48
type: B, layer: 1, pos: 24
type: A, layer: 1, pos: 6
type: B, layer: 1, pos: 15
type: A, layer: 1, pos: 39
type: A, layer: 1, pos: 15
type: B, layer: 1, pos: 26
type: A, layer: 1, pos: 26
type: B, layer: 1, pos: 38
type: B, layer: 1, pos: 9
type: A, layer: 1, pos: 8
type: A, layer: 1, pos: 38
type: B, layer: 1, pos: 37
type: B, layer: 1, pos: 39
type: A, layer: 1, pos: 37
type: A, layer: 1, pos: 31
type: A, layer: 1, pos: 5
type: B, layer: 1, pos: 41
type: A, layer: 1, pos: 41
type: B, layer: 1, pos: 0
type: B, layer: 1, pos: 20
type: A, layer: 1, pos: 0
type: B, layer: 1, pos: 31
type: A, layer: 1, pos: 20
type: B, layer: 1, pos: 5

Time for candidate selection: 0.07 seconds

### Candidate
type: A, layer: 1, pos: 12

### Candidate
type: B, layer: 1, pos: 12

### Candidate
type: B, layer: 1, pos: 28

### Candidate
type: B, layer: 1, pos: 45

### Candidate
type: A, layer: 1, pos: 28

### Candidate
type: A, layer: 1, pos: 45

### Candidate
type: A, layer: 1, pos: 24

### Candidate
type: B, layer: 1, pos: 48

## Relational analysis of NS_B1_A2_A1_B1_B1_A1_B1_B1
Optimization infeasible because this subproblem isn't reachable.

### Candidate
type: A, layer: 1, pos: 48

### Candidate
type: B, layer: 1, pos: 24

### Candidate
type: A, layer: 1, pos: 6

## Relational analysis of NS_B1_A2_A1_B1_B1_A1_B1_A1
Optimization infeasible because this subproblem isn't reachable.

### Candidate
type: B, layer: 1, pos: 15

### Candidate
type: A, layer: 1, pos: 39

## Relational analysis of NS_B1_A2_A1_B1_B1_A1_B1_A1
Optimization infeasible because this subproblem isn't reachable.

### Candidate
type: A, layer: 1, pos: 15

### Candidate
type: B, layer: 1, pos: 26

## Relational analysis of NS_B1_A2_A1_B1_B1_A1_B1_B1
Optimization infeasible because this subproblem isn't reachable.

### Candidate
type: A, layer: 1, pos: 26

### Candidate
type: B, layer: 1, pos: 38

## Relational analysis of NS_B1_A2_A1_B1_B1_A1_B1_B1
Optimization infeasible because this subproblem isn't reachable.

### Candidate
type: B, layer: 1, pos: 9

## Relational analysis of NS_B1_A2_A1_B1_B1_A1_B1_B1

### Relational analysis result of NS_B1_A2_A1_B1_B1_A1_B1_B1
Status: Status.UNKNOWN
Output dim: 0, lower bound: -20.9039416, upper bound: 20.9086279
time: 0.56 seconds

## Relational analysis of NS_B1_A2_A1_B1_B1_A1_B1_B2

### Relational analysis result of NS_B1_A2_A1_B1_B1_A1_B1_B2
Status: Status.UNKNOWN
Output dim: 0, lower bound: -20.9039416, upper bound: 20.9297386
time: 0.59 seconds

## BFS NS instance: NS_B1_A2_A1_B1_B1_A1_B2

### Backsubstitution after applying NS history:
0: -5.6017060, 15.8043280, -5.4423366, 15.3682003, -20.9699059, 21.2466640
1: -14.0179358, 24.3536930, -13.6443949, 23.6233311, -37.6412659, 37.9980888
2: -9.7453976, 22.0511017, -9.4798517, 21.4672966, -31.2126942, 31.5309525
3: -15.0937595, 26.8936596, -14.6907730, 26.1148491, -41.2086105, 41.5844307
4: -13.6622143, 27.2522964, -13.2870970, 26.5533371, -40.2155533, 40.5393944

Time for backsubstitution: 0.85 seconds

### NS candidates at layer 1
type: A, layer: 1, pos: 12
type: B, layer: 1, pos: 12
type: B, layer: 1, pos: 45
type: A, layer: 1, pos: 45
type: B, layer: 1, pos: 15
type: A, layer: 1, pos: 15
type: A, layer: 1, pos: 28
type: B, layer: 1, pos: 28
type: A, layer: 1, pos: 24
type: B, layer: 1, pos: 24
type: A, layer: 1, pos: 39
type: B, layer: 1, pos: 20
type: A, layer: 1, pos: 6
type: A, layer: 1, pos: 8
type: B, layer: 1, pos: 9
type: B, layer: 1, pos: 48
type: B, layer: 1, pos: 38
type: B, layer: 1, pos: 37
type: A, layer: 1, pos: 37
type: B, layer: 1, pos: 39
type: A, layer: 1, pos: 48
type: B, layer: 1, pos: 26
type: A, layer: 1, pos: 38
type: A, layer: 1, pos: 26
type: A, layer: 1, pos: 20
type: B, layer: 1, pos: 0
type: B, layer: 1, pos: 41
type: A, layer: 1, pos: 41
type: A, layer: 1, pos: 0
type: A, layer: 1, pos: 5
type: A, layer: 1, pos: 31
type: B, layer: 1, pos: 31
type: B, layer: 1, pos: 5

Time for candidate selection: 0.07 seconds

### Candidate
type: A, layer: 1, pos: 12

### Candidate
type: B, layer: 1, pos: 12

### Candidate
type: B, layer: 1, pos: 45

### Candidate
type: A, layer: 1, pos: 45

### Candidate
type: B, layer: 1, pos: 15

### Candidate
type: A, layer: 1, pos: 15

### Candidate
type: A, layer: 1, pos: 28

### Candidate
type: B, layer: 1, pos: 28

### Candidate
type: A, layer: 1, pos: 24

### Candidate
type: B, layer: 1, pos: 24

### Candidate
type: A, layer: 1, pos: 39

## Relational analysis of NS_B1_A2_A1_B1_B1_A1_B2_A1
Optimization infeasible because this subproblem isn't reachable.

### Candidate
type: B, layer: 1, pos: 20

## Relational analysis of NS_B1_A2_A1_B1_B1_A1_B2_B1
Optimization infeasible because this subproblem isn't reachable.

### Candidate
type: A, layer: 1, pos: 6

## Relational analysis of NS_B1_A2_A1_B1_B1_A1_B2_A1
Optimization infeasible because this subproblem isn't reachable.

### Candidate
type: A, layer: 1, pos: 8

## Relational analysis of NS_B1_A2_A1_B1_B1_A1_B2_A1

### Relational analysis result of NS_B1_A2_A1_B1_B1_A1_B2_A1
Status: Status.UNKNOWN
Output dim: 0, lower bound: -20.9418692, upper bound: 20.9293078
time: 0.55 seconds

## Relational analysis of NS_B1_A2_A1_B1_B1_A1_B2_A2

### Relational analysis result of NS_B1_A2_A1_B1_B1_A1_B2_A2
Status: Status.UNKNOWN
Output dim: 0, lower bound: -20.9418692, upper bound: 20.9293078
time: 0.89 seconds

## BFS NS instance: NS_B1_A2_A1_B1_B1_A2_B1

### Backsubstitution after applying NS history:
0: -5.2297373, 14.7756672, -4.3288293, 12.4058933, -17.6356297, 19.1044941
1: -13.1044369, 22.7413082, -10.6637182, 19.3567238, -32.4611511, 33.4050255
2: -9.0977917, 20.6212330, -7.4333587, 17.4881172, -26.5859089, 28.0545902
3: -14.1042252, 25.1446953, -11.6280355, 21.3023090, -35.4065323, 36.7727242
4: -12.7082987, 25.5085201, -10.6335793, 21.5448170, -34.2531166, 36.1420937

Time for backsubstitution: 0.85 seconds

### NS candidates at layer 1
type: A, layer: 1, pos: 12
type: B, layer: 1, pos: 12
type: B, layer: 1, pos: 45
type: A, layer: 1, pos: 28
type: B, layer: 1, pos: 28
type: A, layer: 1, pos: 45
type: A, layer: 1, pos: 24
type: B, layer: 1, pos: 48
type: A, layer: 1, pos: 6
type: A, layer: 1, pos: 48
type: B, layer: 1, pos: 15
type: B, layer: 1, pos: 24
type: B, layer: 1, pos: 26
type: A, layer: 1, pos: 15
type: A, layer: 1, pos: 26
type: A, layer: 1, pos: 39
type: B, layer: 1, pos: 38
type: A, layer: 1, pos: 38
type: A, layer: 1, pos: 8
type: B, layer: 1, pos: 9
type: B, layer: 1, pos: 39
type: B, layer: 1, pos: 37
type: A, layer: 1, pos: 37
type: A, layer: 1, pos: 5
type: A, layer: 1, pos: 31
type: B, layer: 1, pos: 41
type: A, layer: 1, pos: 41
type: B, layer: 1, pos: 0
type: B, layer: 1, pos: 20
type: A, layer: 1, pos: 0
type: B, layer: 1, pos: 31
type: A, layer: 1, pos: 20
type: B, layer: 1, pos: 5

Time for candidate selection: 0.07 seconds

### Candidate
type: A, layer: 1, pos: 12

### Candidate
type: B, layer: 1, pos: 12

### Candidate
type: B, layer: 1, pos: 45

### Candidate
type: A, layer: 1, pos: 28

### Candidate
type: B, layer: 1, pos: 28

### Candidate
type: A, layer: 1, pos: 45

### Candidate
type: A, layer: 1, pos: 24

### Candidate
type: B, layer: 1, pos: 48

## Relational analysis of NS_B1_A2_A1_B1_B1_A2_B1_B1
Optimization infeasible because this subproblem isn't reachable.

### Candidate
type: A, layer: 1, pos: 6

## Relational analysis of NS_B1_A2_A1_B1_B1_A2_B1_A1
Optimization infeasible because this subproblem isn't reachable.

### Candidate
type: A, layer: 1, pos: 48

### Candidate
type: B, layer: 1, pos: 15

### Candidate
type: B, layer: 1, pos: 24

### Candidate
type: B, layer: 1, pos: 26

## Relational analysis of NS_B1_A2_A1_B1_B1_A2_B1_B1
Optimization infeasible because this subproblem isn't reachable.

### Candidate
type: A, layer: 1, pos: 15

### Candidate
type: A, layer: 1, pos: 26

### Candidate
type: A, layer: 1, pos: 39

## Relational analysis of NS_B1_A2_A1_B1_B1_A2_B1_A1
Optimization infeasible because this subproblem isn't reachable.

### Candidate
type: B, layer: 1, pos: 38

## Relational analysis of NS_B1_A2_A1_B1_B1_A2_B1_B1
Optimization infeasible because this subproblem isn't reachable.

### Candidate
type: A, layer: 1, pos: 38

### Candidate
type: A, layer: 1, pos: 8

## Relational analysis of NS_B1_A2_A1_B1_B1_A2_B1_A1

### Relational analysis result of NS_B1_A2_A1_B1_B1_A2_B1_A1
Status: Status.UNKNOWN
Output dim: 0, lower bound: -20.9592875, upper bound: 20.9564825
time: 0.58 seconds

## Relational analysis of NS_B1_A2_A1_B1_B1_A2_B1_A2

### Relational analysis result of NS_B1_A2_A1_B1_B1_A2_B1_A2
Status: Status.UNKNOWN
Output dim: 0, lower bound: -20.9592875, upper bound: 20.9570448
time: 0.65 seconds

## BFS NS instance: NS_B1_A2_A1_B1_B1_A2_B2

### Backsubstitution after applying NS history:
0: -5.5627847, 15.6472750, -5.3982439, 15.2479134, -20.8106976, 21.0455151
1: -13.9816160, 24.0097771, -13.5351353, 23.4425526, -37.4241638, 37.5449104
2: -9.7048798, 21.8011341, -9.4025078, 21.3037167, -31.0085926, 31.2036419
3: -15.0155830, 26.5744400, -14.5737047, 25.9148712, -40.9304504, 41.1481438
4: -13.5180569, 26.9750786, -13.1778412, 26.3528252, -39.8708649, 40.1529160

Time for backsubstitution: 0.85 seconds

### NS candidates at layer 1
type: A, layer: 1, pos: 12
type: B, layer: 1, pos: 12
type: B, layer: 1, pos: 45
type: B, layer: 1, pos: 15
type: A, layer: 1, pos: 45
type: A, layer: 1, pos: 28
type: A, layer: 1, pos: 15
type: A, layer: 1, pos: 24
type: B, layer: 1, pos: 28
type: A, layer: 1, pos: 39
type: B, layer: 1, pos: 24
type: A, layer: 1, pos: 6
type: B, layer: 1, pos: 20
type: A, layer: 1, pos: 8
type: B, layer: 1, pos: 48
type: B, layer: 1, pos: 38
type: B, layer: 1, pos: 37
type: A, layer: 1, pos: 37
type: B, layer: 1, pos: 39
type: B, layer: 1, pos: 9
type: A, layer: 1, pos: 48
type: B, layer: 1, pos: 26
type: A, layer: 1, pos: 38
type: A, layer: 1, pos: 26
type: A, layer: 1, pos: 20
type: B, layer: 1, pos: 0
type: A, layer: 1, pos: 0
type: B, layer: 1, pos: 41
type: A, layer: 1, pos: 5
type: A, layer: 1, pos: 41
type: A, layer: 1, pos: 31
type: B, layer: 1, pos: 31
type: B, layer: 1, pos: 5

Time for candidate selection: 0.07 seconds

### Candidate
type: A, layer: 1, pos: 12

### Candidate
type: B, layer: 1, pos: 12

### Candidate
type: B, layer: 1, pos: 45

### Candidate
type: B, layer: 1, pos: 15

### Candidate
type: A, layer: 1, pos: 45

### Candidate
type: A, layer: 1, pos: 28

### Candidate
type: A, layer: 1, pos: 15

### Candidate
type: A, layer: 1, pos: 24

### Candidate
type: B, layer: 1, pos: 28

### Candidate
type: A, layer: 1, pos: 39

## Relational analysis of NS_B1_A2_A1_B1_B1_A2_B2_A1
Optimization infeasible because this subproblem isn't reachable.

### Candidate
type: B, layer: 1, pos: 24

### Candidate
type: A, layer: 1, pos: 6

## Relational analysis of NS_B1_A2_A1_B1_B1_A2_B2_A1
Optimization infeasible because this subproblem isn't reachable.

### Candidate
type: B, layer: 1, pos: 20

## Relational analysis of NS_B1_A2_A1_B1_B1_A2_B2_B1
Optimization infeasible because this subproblem isn't reachable.

### Candidate
type: A, layer: 1, pos: 8

## Relational analysis of NS_B1_A2_A1_B1_B1_A2_B2_A1

### Relational analysis result of NS_B1_A2_A1_B1_B1_A2_B2_A1
Status: Status.UNKNOWN
Output dim: 0, lower bound: -20.9592875, upper bound: 20.9564825
time: 0.68 seconds

## Relational analysis of NS_B1_A2_A1_B1_B1_A2_B2_A2

### Relational analysis result of NS_B1_A2_A1_B1_B1_A2_B2_A2
Status: Status.UNKNOWN
Output dim: 0, lower bound: -20.9592875, upper bound: 20.9570448
time: 0.64 seconds

## BFS NS instance: NS_B1_A2_A1_B1_B2_A1_B1

### Backsubstitution after applying NS history:
0: -5.9219766, 16.6530056, -6.0759130, 17.0027866, -22.9247608, 22.7289181
1: -14.8774786, 25.5903778, -15.3089046, 25.9402771, -40.8177567, 40.8992844
2: -10.3299103, 23.1856422, -10.6203365, 23.5985222, -33.9284286, 33.8059769
3: -15.9785748, 28.2802773, -16.4063854, 28.7308846, -44.7094574, 44.6866570
4: -14.4363346, 28.6716270, -14.7959938, 29.2124386, -43.6487732, 43.4676208

Time for backsubstitution: 0.86 seconds

### NS candidates at layer 1
type: A, layer: 1, pos: 12
type: B, layer: 1, pos: 12
type: B, layer: 1, pos: 45
type: B, layer: 1, pos: 15
type: A, layer: 1, pos: 15
type: B, layer: 1, pos: 28
type: A, layer: 1, pos: 45
type: A, layer: 1, pos: 28
type: A, layer: 1, pos: 24
type: B, layer: 1, pos: 24
type: B, layer: 1, pos: 37
type: A, layer: 1, pos: 37
type: A, layer: 1, pos: 6
type: B, layer: 1, pos: 48
type: A, layer: 1, pos: 39
type: B, layer: 1, pos: 38
type: B, layer: 1, pos: 8
type: A, layer: 1, pos: 8
type: A, layer: 1, pos: 38
type: A, layer: 1, pos: 48
type: B, layer: 1, pos: 39
type: B, layer: 1, pos: 26
type: A, layer: 1, pos: 26
type: B, layer: 1, pos: 20
type: A, layer: 1, pos: 20
type: A, layer: 1, pos: 41
type: B, layer: 1, pos: 41
type: B, layer: 1, pos: 0
type: A, layer: 1, pos: 5
type: B, layer: 1, pos: 5
type: A, layer: 1, pos: 0
type: A, layer: 1, pos: 31
type: B, layer: 1, pos: 31

Time for candidate selection: 0.07 seconds

### Candidate
type: A, layer: 1, pos: 12

### Candidate
type: B, layer: 1, pos: 12

### Candidate
type: B, layer: 1, pos: 45

### Candidate
type: B, layer: 1, pos: 15

### Candidate
type: A, layer: 1, pos: 15

### Candidate
type: B, layer: 1, pos: 28

### Candidate
type: A, layer: 1, pos: 45

### Candidate
type: A, layer: 1, pos: 28

### Candidate
type: A, layer: 1, pos: 24

### Candidate
type: B, layer: 1, pos: 24

### Candidate
type: B, layer: 1, pos: 37

## Relational analysis of NS_B1_A2_A1_B1_B2_A1_B1_B1
Optimization infeasible because this subproblem isn't reachable.

### Candidate
type: A, layer: 1, pos: 37

### Candidate
type: A, layer: 1, pos: 6

## Relational analysis of NS_B1_A2_A1_B1_B2_A1_B1_A1
Optimization infeasible because this subproblem isn't reachable.

### Candidate
type: B, layer: 1, pos: 48

## Relational analysis of NS_B1_A2_A1_B1_B2_A1_B1_B1
Optimization infeasible because this subproblem isn't reachable.

### Candidate
type: A, layer: 1, pos: 39

## Relational analysis of NS_B1_A2_A1_B1_B2_A1_B1_A1
Optimization infeasible because this subproblem isn't reachable.

### Candidate
type: B, layer: 1, pos: 38

## Relational analysis of NS_B1_A2_A1_B1_B2_A1_B1_B1
Optimization infeasible because this subproblem isn't reachable.

### Candidate
type: B, layer: 1, pos: 8

## Relational analysis of NS_B1_A2_A1_B1_B2_A1_B1_B1

### Relational analysis result of NS_B1_A2_A1_B1_B2_A1_B1_B1
Status: Status.UNKNOWN
Output dim: 0, lower bound: -20.9304753, upper bound: 20.9293804
time: 1.03 seconds

## Relational analysis of NS_B1_A2_A1_B1_B2_A1_B1_B2

### Relational analysis result of NS_B1_A2_A1_B1_B2_A1_B1_B2
Status: Status.UNKNOWN
Output dim: 0, lower bound: -20.9283053, upper bound: 20.9291819
time: 0.61 seconds

## BFS NS instance: NS_B1_A2_A1_B1_B2_A1_B2

### Backsubstitution after applying NS history:
0: -5.9219766, 16.6530056, -6.1153221, 17.0719090, -22.9938831, 22.7683277
1: -14.8774786, 25.5903778, -15.4780693, 25.9482307, -40.8257065, 41.0684471
2: -10.3299103, 23.1856422, -10.7194395, 23.6610470, -33.9909554, 33.9050827
3: -15.9785748, 28.2802773, -16.5413704, 28.7916222, -44.7701950, 44.8216476
4: -14.4363346, 28.6716270, -14.8485556, 29.3176708, -43.7540054, 43.5201836

Time for backsubstitution: 0.86 seconds

### NS candidates at layer 1
type: A, layer: 1, pos: 12
type: B, layer: 1, pos: 12
type: B, layer: 1, pos: 45
type: B, layer: 1, pos: 15
type: B, layer: 1, pos: 28
type: A, layer: 1, pos: 15
type: A, layer: 1, pos: 45
type: A, layer: 1, pos: 28
type: B, layer: 1, pos: 24
type: A, layer: 1, pos: 24
type: B, layer: 1, pos: 37
type: A, layer: 1, pos: 37
type: A, layer: 1, pos: 39
type: A, layer: 1, pos: 6
type: B, layer: 1, pos: 48
type: A, layer: 1, pos: 8
type: B, layer: 1, pos: 38
type: B, layer: 1, pos: 8
type: A, layer: 1, pos: 38
type: A, layer: 1, pos: 48
type: B, layer: 1, pos: 39
type: B, layer: 1, pos: 20
type: B, layer: 1, pos: 26
type: A, layer: 1, pos: 26
type: A, layer: 1, pos: 20
type: A, layer: 1, pos: 41
type: B, layer: 1, pos: 41
type: B, layer: 1, pos: 0
type: B, layer: 1, pos: 5
type: A, layer: 1, pos: 5
type: A, layer: 1, pos: 31
type: A, layer: 1, pos: 0
type: B, layer: 1, pos: 31

Time for candidate selection: 0.07 seconds

### Candidate
type: A, layer: 1, pos: 12

### Candidate
type: B, layer: 1, pos: 12

### Candidate
type: B, layer: 1, pos: 45

### Candidate
type: B, layer: 1, pos: 15

### Candidate
type: B, layer: 1, pos: 28

### Candidate
type: A, layer: 1, pos: 15

### Candidate
type: A, layer: 1, pos: 45

### Candidate
type: A, layer: 1, pos: 28

### Candidate
type: B, layer: 1, pos: 24

### Candidate
type: A, layer: 1, pos: 24

### Candidate
type: B, layer: 1, pos: 37

## Relational analysis of NS_B1_A2_A1_B1_B2_A1_B2_B1
Optimization infeasible because this subproblem isn't reachable.

### Candidate
type: A, layer: 1, pos: 37

### Candidate
type: A, layer: 1, pos: 39

## Relational analysis of NS_B1_A2_A1_B1_B2_A1_B2_A1
Optimization infeasible because this subproblem isn't reachable.

### Candidate
type: A, layer: 1, pos: 6

## Relational analysis of NS_B1_A2_A1_B1_B2_A1_B2_A1
Optimization infeasible because this subproblem isn't reachable.

### Candidate
type: B, layer: 1, pos: 48

## Relational analysis of NS_B1_A2_A1_B1_B2_A1_B2_B1
Optimization infeasible because this subproblem isn't reachable.

### Candidate
type: A, layer: 1, pos: 8

## Relational analysis of NS_B1_A2_A1_B1_B2_A1_B2_A1

### Relational analysis result of NS_B1_A2_A1_B1_B2_A1_B2_A1
Status: Status.UNKNOWN
Output dim: 0, lower bound: -20.9282818, upper bound: 20.9438823
time: 0.63 seconds

## Relational analysis of NS_B1_A2_A1_B1_B2_A1_B2_A2

### Relational analysis result of NS_B1_A2_A1_B1_B2_A1_B2_A2
Status: Status.UNKNOWN
Output dim: 0, lower bound: -20.9283053, upper bound: 20.9437580
time: 1.06 seconds

## BFS NS instance: NS_B1_A2_A1_B1_B2_A2_B1

### Backsubstitution after applying NS history:
0: -5.5127878, 15.5301762, -5.4734383, 15.4163799, -20.9291649, 21.0036125
1: -13.8671846, 23.8362732, -13.7338095, 23.6569328, -37.5241127, 37.5700798
2: -9.6172028, 21.6250057, -9.5148935, 21.4294472, -31.0466499, 31.1399002
3: -14.8874836, 26.3736210, -14.7569494, 26.1703167, -41.0578003, 41.1305695
4: -13.3854656, 26.7717056, -13.2949619, 26.4767418, -39.8622055, 40.0666656

Time for backsubstitution: 0.86 seconds

### NS candidates at layer 1
type: A, layer: 1, pos: 12
type: B, layer: 1, pos: 12
type: B, layer: 1, pos: 45
type: A, layer: 1, pos: 45
type: B, layer: 1, pos: 28
type: B, layer: 1, pos: 15
type: A, layer: 1, pos: 15
type: A, layer: 1, pos: 39
type: B, layer: 1, pos: 24
type: A, layer: 1, pos: 28
type: B, layer: 1, pos: 48
type: A, layer: 1, pos: 24
type: B, layer: 1, pos: 20
type: A, layer: 1, pos: 8
type: B, layer: 1, pos: 9
type: B, layer: 1, pos: 38
type: B, layer: 1, pos: 37
type: B, layer: 1, pos: 39
type: A, layer: 1, pos: 37
type: A, layer: 1, pos: 38
type: A, layer: 1, pos: 20
type: A, layer: 1, pos: 6
type: A, layer: 1, pos: 48
type: A, layer: 1, pos: 26
type: B, layer: 1, pos: 26
type: B, layer: 1, pos: 41
type: A, layer: 1, pos: 41
type: B, layer: 1, pos: 0
type: B, layer: 1, pos: 31
type: A, layer: 1, pos: 5
type: A, layer: 1, pos: 31
type: A, layer: 1, pos: 0
type: B, layer: 1, pos: 5

Time for candidate selection: 0.07 seconds

### Candidate
type: A, layer: 1, pos: 12

### Candidate
type: B, layer: 1, pos: 12

### Candidate
type: B, layer: 1, pos: 45

### Candidate
type: A, layer: 1, pos: 45

### Candidate
type: B, layer: 1, pos: 28

### Candidate
type: B, layer: 1, pos: 15

### Candidate
type: A, layer: 1, pos: 15

### Candidate
type: A, layer: 1, pos: 39

### Candidate
type: B, layer: 1, pos: 24

### Candidate
type: A, layer: 1, pos: 28

### Candidate
type: B, layer: 1, pos: 48

### Candidate
type: A, layer: 1, pos: 24

### Candidate
type: B, layer: 1, pos: 20

## Relational analysis of NS_B1_A2_A1_B1_B2_A2_B1_B1
Optimization infeasible because this subproblem isn't reachable.

### Candidate
type: A, layer: 1, pos: 8

## Relational analysis of NS_B1_A2_A1_B1_B2_A2_B1_A1

### Relational analysis result of NS_B1_A2_A1_B1_B2_A2_B1_A1
Status: Status.UNKNOWN
Output dim: 0, lower bound: -20.9613564, upper bound: 20.9610719
time: 0.63 seconds

## Relational analysis of NS_B1_A2_A1_B1_B2_A2_B1_A2

### Relational analysis result of NS_B1_A2_A1_B1_B2_A2_B1_A2
Status: Status.UNKNOWN
Output dim: 0, lower bound: -20.9613564, upper bound: 20.9621042
time: 0.61 seconds

## BFS NS instance: NS_B1_A2_A1_B1_B2_A2_B2

### Backsubstitution after applying NS history:
0: -5.8575258, 16.4333668, -6.0655088, 16.9631748, -22.8207016, 22.4988747
1: -14.7730999, 25.1493797, -15.2842464, 25.8777142, -40.6508102, 40.4336243
2: -10.2441244, 22.8457184, -10.6076508, 23.5473042, -33.7914276, 33.4533691
3: -15.8305902, 27.8522625, -16.3785610, 28.6656246, -44.4962006, 44.2308235
4: -14.2308779, 28.2878342, -14.7786846, 29.1453743, -43.3762512, 43.0665207

Time for backsubstitution: 0.86 seconds

### NS candidates at layer 1
type: A, layer: 1, pos: 12
type: B, layer: 1, pos: 12
type: B, layer: 1, pos: 45
type: B, layer: 1, pos: 28
type: A, layer: 1, pos: 45
type: B, layer: 1, pos: 15
type: A, layer: 1, pos: 15
type: A, layer: 1, pos: 28
type: A, layer: 1, pos: 24
type: B, layer: 1, pos: 24
type: A, layer: 1, pos: 39
type: B, layer: 1, pos: 48
type: A, layer: 1, pos: 6
type: B, layer: 1, pos: 38
type: B, layer: 1, pos: 37
type: A, layer: 1, pos: 37
type: A, layer: 1, pos: 8
type: B, layer: 1, pos: 39
type: A, layer: 1, pos: 38
type: B, layer: 1, pos: 9
type: B, layer: 1, pos: 20
type: A, layer: 1, pos: 48
type: B, layer: 1, pos: 26
type: A, layer: 1, pos: 26
type: A, layer: 1, pos: 20
type: B, layer: 1, pos: 0
type: B, layer: 1, pos: 41
type: A, layer: 1, pos: 41
type: A, layer: 1, pos: 5
type: A, layer: 1, pos: 31
type: B, layer: 1, pos: 31
type: A, layer: 1, pos: 0
type: B, layer: 1, pos: 5

Time for candidate selection: 0.07 seconds

### Candidate
type: A, layer: 1, pos: 12

### Candidate
type: B, layer: 1, pos: 12

### Candidate
type: B, layer: 1, pos: 45

### Candidate
type: B, layer: 1, pos: 28

### Candidate
type: A, layer: 1, pos: 45

### Candidate
type: B, layer: 1, pos: 15

### Candidate
type: A, layer: 1, pos: 15

### Candidate
type: A, layer: 1, pos: 28

### Candidate
type: A, layer: 1, pos: 24

### Candidate
type: B, layer: 1, pos: 24

### Candidate
type: A, layer: 1, pos: 39

### Candidate
type: B, layer: 1, pos: 48

### Candidate
type: A, layer: 1, pos: 6

### Candidate
type: B, layer: 1, pos: 38

## Relational analysis of NS_B1_A2_A1_B1_B2_A2_B2_B1
Optimization infeasible because this subproblem isn't reachable.

### Candidate
type: B, layer: 1, pos: 37

## Relational analysis of NS_B1_A2_A1_B1_B2_A2_B2_B1
Optimization infeasible because this subproblem isn't reachable.

### Candidate
type: A, layer: 1, pos: 37

### Candidate
type: A, layer: 1, pos: 8

## Relational analysis of NS_B1_A2_A1_B1_B2_A2_B2_A1

### Relational analysis result of NS_B1_A2_A1_B1_B2_A2_B2_A1
Status: Status.UNKNOWN
Output dim: 0, lower bound: -20.9620715, upper bound: 20.9622880
time: 0.61 seconds

## Relational analysis of NS_B1_A2_A1_B1_B2_A2_B2_A2

### Relational analysis result of NS_B1_A2_A1_B1_B2_A2_B2_A2
Status: Status.UNKNOWN
Output dim: 0, lower bound: -20.9620715, upper bound: 20.9633596
time: 1.04 seconds

## BFS NS instance: NS_B1_A2_A1_B2_B1_A1_B1

### Backsubstitution after applying NS history:
0: -5.2740431, 14.9444952, -4.6193309, 13.1242275, -18.3982658, 19.5638237
1: -13.1596384, 23.1045856, -11.3252010, 20.5041637, -33.6638031, 34.4297791
2: -9.1495724, 20.8888454, -7.9285593, 18.5338135, -27.6833858, 28.8174057
3: -14.1973553, 25.4861031, -12.3707266, 22.5554943, -36.7528458, 37.8568306
4: -12.8572388, 25.8054695, -11.3954086, 22.7912655, -35.6485062, 37.2008781

Time for backsubstitution: 0.87 seconds

### NS candidates at layer 1
type: A, layer: 1, pos: 12
type: B, layer: 1, pos: 12
type: B, layer: 1, pos: 45
type: B, layer: 1, pos: 15
type: A, layer: 1, pos: 15
type: A, layer: 1, pos: 45
type: B, layer: 1, pos: 48
type: A, layer: 1, pos: 24
type: B, layer: 1, pos: 24
type: B, layer: 1, pos: 9
type: B, layer: 1, pos: 39
type: B, layer: 1, pos: 37
type: A, layer: 1, pos: 37
type: B, layer: 1, pos: 38
type: A, layer: 1, pos: 20
type: A, layer: 1, pos: 39
type: B, layer: 1, pos: 28
type: A, layer: 1, pos: 6
type: B, layer: 1, pos: 20
type: A, layer: 1, pos: 8
type: A, layer: 1, pos: 38
type: A, layer: 1, pos: 48
type: B, layer: 1, pos: 0
type: A, layer: 1, pos: 0
type: A, layer: 1, pos: 28
type: A, layer: 1, pos: 31
type: A, layer: 1, pos: 5
type: B, layer: 1, pos: 31
type: B, layer: 1, pos: 26
type: A, layer: 1, pos: 26
type: B, layer: 1, pos: 41
type: A, layer: 1, pos: 41
type: B, layer: 1, pos: 5

Time for candidate selection: 0.07 seconds

### Candidate
type: A, layer: 1, pos: 12

### Candidate
type: B, layer: 1, pos: 12

### Candidate
type: B, layer: 1, pos: 45

### Candidate
type: B, layer: 1, pos: 15

### Candidate
type: A, layer: 1, pos: 15

### Candidate
type: A, layer: 1, pos: 45

### Candidate
type: B, layer: 1, pos: 48

## Relational analysis of NS_B1_A2_A1_B2_B1_A1_B1_B1
Optimization infeasible because this subproblem isn't reachable.

### Candidate
type: A, layer: 1, pos: 24

## Relational analysis of NS_B1_A2_A1_B2_B1_A1_B1_A1
Optimization infeasible because this subproblem isn't reachable.

### Candidate
type: B, layer: 1, pos: 24

### Candidate
type: B, layer: 1, pos: 9

## Relational analysis of NS_B1_A2_A1_B2_B1_A1_B1_B1

### Relational analysis result of NS_B1_A2_A1_B2_B1_A1_B1_B1
Status: Status.UNKNOWN
Output dim: 0, lower bound: -20.9023832, upper bound: 20.9074667
time: 0.96 seconds

## Relational analysis of NS_B1_A2_A1_B2_B1_A1_B1_B2

### Relational analysis result of NS_B1_A2_A1_B2_B1_A1_B1_B2
Status: Status.UNKNOWN
Output dim: 0, lower bound: -20.9023832, upper bound: 20.9074667
time: 0.59 seconds

## BFS NS instance: NS_B1_A2_A1_B2_B1_A1_B2

### Backsubstitution after applying NS history:
0: -5.6017060, 15.8043280, -5.5348625, 15.5223913, -21.1240978, 21.3391914
1: -14.0179358, 24.3536930, -13.7450657, 23.9700623, -37.9879990, 38.0987587
2: -9.7453976, 22.0511017, -9.6019926, 21.7476902, -31.4930878, 31.6530952
3: -15.0937595, 26.8936596, -14.8788052, 26.4719048, -41.5656624, 41.7724609
4: -13.6622143, 27.2522964, -13.6262836, 26.8271027, -40.4893188, 40.8785782

Time for backsubstitution: 0.87 seconds

### NS candidates at layer 1
type: A, layer: 1, pos: 12
type: B, layer: 1, pos: 12
type: B, layer: 1, pos: 45
type: B, layer: 1, pos: 15
type: A, layer: 1, pos: 15
type: A, layer: 1, pos: 45
type: B, layer: 1, pos: 48
type: B, layer: 1, pos: 20
type: B, layer: 1, pos: 9
type: A, layer: 1, pos: 24
type: B, layer: 1, pos: 39
type: A, layer: 1, pos: 20
type: A, layer: 1, pos: 8
type: B, layer: 1, pos: 38
type: A, layer: 1, pos: 6
type: A, layer: 1, pos: 37
type: B, layer: 1, pos: 24
type: A, layer: 1, pos: 28
type: B, layer: 1, pos: 37
type: A, layer: 1, pos: 38
type: A, layer: 1, pos: 39
type: B, layer: 1, pos: 28
type: A, layer: 1, pos: 48
type: A, layer: 1, pos: 0
type: B, layer: 1, pos: 0
type: A, layer: 1, pos: 5
type: B, layer: 1, pos: 26
type: A, layer: 1, pos: 26
type: B, layer: 1, pos: 31
type: B, layer: 1, pos: 41
type: A, layer: 1, pos: 41
type: A, layer: 1, pos: 31
type: B, layer: 1, pos: 5

Time for candidate selection: 0.07 seconds

### Candidate
type: A, layer: 1, pos: 12

### Candidate
type: B, layer: 1, pos: 12

### Candidate
type: B, layer: 1, pos: 45

### Candidate
type: B, layer: 1, pos: 15

### Candidate
type: A, layer: 1, pos: 15

### Candidate
type: A, layer: 1, pos: 45

### Candidate
type: B, layer: 1, pos: 48

## Relational analysis of NS_B1_A2_A1_B2_B1_A1_B2_B1
Optimization infeasible because this subproblem isn't reachable.

### Candidate
type: B, layer: 1, pos: 20

## Relational analysis of NS_B1_A2_A1_B2_B1_A1_B2_B1
Optimization infeasible because this subproblem isn't reachable.

### Candidate
type: B, layer: 1, pos: 9

## Relational analysis of NS_B1_A2_A1_B2_B1_A1_B2_B1

### Relational analysis result of NS_B1_A2_A1_B2_B1_A1_B2_B1
Status: Status.UNKNOWN
Output dim: 0, lower bound: -20.9022185, upper bound: 20.9075448
time: 0.59 seconds

## Relational analysis of NS_B1_A2_A1_B2_B1_A1_B2_B2

### Relational analysis result of NS_B1_A2_A1_B2_B1_A1_B2_B2
Status: Status.UNKNOWN
Output dim: 0, lower bound: -20.9022185, upper bound: 20.9075448
time: 0.65 seconds

## BFS NS instance: NS_B1_A2_A1_B2_B1_A2_A1

### Backsubstitution after applying NS history:
0: -4.4431758, 12.6255226, -5.0153255, 14.1176205, -18.5607948, 17.6408463
1: -10.9563599, 19.6315041, -12.3389311, 21.9800701, -32.9364243, 31.9704304
2: -7.6303101, 17.7165203, -8.6443806, 19.8528118, -27.4831200, 26.3609009
3: -11.8924522, 21.6263542, -13.4369297, 24.2071075, -36.0995598, 35.0632744
4: -10.8212805, 21.7900391, -12.3832541, 24.4330044, -35.2542839, 34.1732941

Time for backsubstitution: 0.87 seconds

### NS candidates at layer 1
type: A, layer: 1, pos: 12
type: B, layer: 1, pos: 12
type: B, layer: 1, pos: 15
type: B, layer: 1, pos: 45
type: A, layer: 1, pos: 15
type: A, layer: 1, pos: 45
type: B, layer: 1, pos: 48
type: B, layer: 1, pos: 39
type: B, layer: 1, pos: 20
type: A, layer: 1, pos: 24
type: B, layer: 1, pos: 8
type: B, layer: 1, pos: 24
type: A, layer: 1, pos: 28
type: A, layer: 1, pos: 8
type: A, layer: 1, pos: 6
type: A, layer: 1, pos: 37
type: B, layer: 1, pos: 38
type: A, layer: 1, pos: 39
type: B, layer: 1, pos: 28
type: B, layer: 1, pos: 37
type: B, layer: 1, pos: 9
type: A, layer: 1, pos: 38
type: A, layer: 1, pos: 0
type: B, layer: 1, pos: 0
type: A, layer: 1, pos: 48
type: B, layer: 1, pos: 26
type: A, layer: 1, pos: 5
type: A, layer: 1, pos: 26
type: B, layer: 1, pos: 31
type: A, layer: 1, pos: 41
type: B, layer: 1, pos: 41
type: A, layer: 1, pos: 31
type: B, layer: 1, pos: 5

Time for candidate selection: 0.07 seconds

### Candidate
type: A, layer: 1, pos: 12

### Candidate
type: B, layer: 1, pos: 12

### Candidate
type: B, layer: 1, pos: 15

### Candidate
type: B, layer: 1, pos: 45

### Candidate
type: A, layer: 1, pos: 15

### Candidate
type: A, layer: 1, pos: 45

### Candidate
type: B, layer: 1, pos: 48

## Relational analysis of NS_B1_A2_A1_B2_B1_A2_A1_B1
Optimization infeasible because this subproblem isn't reachable.

### Candidate
type: B, layer: 1, pos: 39

## Relational analysis of NS_B1_A2_A1_B2_B1_A2_A1_B1
Optimization infeasible because this subproblem isn't reachable.

### Candidate
type: B, layer: 1, pos: 20

## Relational analysis of NS_B1_A2_A1_B2_B1_A2_A1_B1
Optimization infeasible because this subproblem isn't reachable.

### Candidate
type: A, layer: 1, pos: 24

## Relational analysis of NS_B1_A2_A1_B2_B1_A2_A1_A1
Optimization infeasible because this subproblem isn't reachable.

### Candidate
type: B, layer: 1, pos: 8

## Relational analysis of NS_B1_A2_A1_B2_B1_A2_A1_B1
Optimization infeasible because this subproblem isn't reachable.

### Candidate
type: B, layer: 1, pos: 24

### Candidate
type: A, layer: 1, pos: 28

### Candidate
type: A, layer: 1, pos: 8

### Candidate
type: A, layer: 1, pos: 6

## Relational analysis of NS_B1_A2_A1_B2_B1_A2_A1_A1
Optimization infeasible because this subproblem isn't reachable.

### Candidate
type: A, layer: 1, pos: 37

## Relational analysis of NS_B1_A2_A1_B2_B1_A2_A1_A1
Optimization infeasible because this subproblem isn't reachable.

### Candidate
type: B, layer: 1, pos: 38

## Relational analysis of NS_B1_A2_A1_B2_B1_A2_A1_B1
Optimization infeasible because this subproblem isn't reachable.

### Candidate
type: A, layer: 1, pos: 39

### Candidate
type: B, layer: 1, pos: 28

### Candidate
type: B, layer: 1, pos: 37

### Candidate
type: B, layer: 1, pos: 9

## Relational analysis of NS_B1_A2_A1_B2_B1_A2_A1_B1

### Relational analysis result of NS_B1_A2_A1_B2_B1_A2_A1_B1
Status: Status.UNKNOWN
Output dim: 0, lower bound: -20.9023076, upper bound: 20.9179939
time: 0.70 seconds

## Relational analysis of NS_B1_A2_A1_B2_B1_A2_A1_B2
Optimization infeasible because this subproblem isn't reachable.

### Candidate
type: A, layer: 1, pos: 38

### Candidate
type: A, layer: 1, pos: 0

## Relational analysis of NS_B1_A2_A1_B2_B1_A2_A1_A1
Optimization infeasible because this subproblem isn't reachable.

### Candidate
type: B, layer: 1, pos: 0

### Candidate
type: A, layer: 1, pos: 48

### Candidate
type: B, layer: 1, pos: 26

## Relational analysis of NS_B1_A2_A1_B2_B1_A2_A1_B1
Optimization infeasible because this subproblem isn't reachable.

### Candidate
type: A, layer: 1, pos: 5

## Relational analysis of NS_B1_A2_A1_B2_B1_A2_A1_A1

### Relational analysis result of NS_B1_A2_A1_B2_B1_A2_A1_A1
Status: Status.UNKNOWN
Output dim: 0, lower bound: -20.9171211, upper bound: 20.9284197
time: 0.68 seconds

## Relational analysis of NS_B1_A2_A1_B2_B1_A2_A1_A2
Optimization infeasible because this subproblem isn't reachable.

### Candidate
type: A, layer: 1, pos: 26

### Candidate
type: B, layer: 1, pos: 31

## Relational analysis of NS_B1_A2_A1_B2_B1_A2_A1_B1
Optimization infeasible because this subproblem isn't reachable.

### Candidate
type: A, layer: 1, pos: 41

## Relational analysis of NS_B1_A2_A1_B2_B1_A2_A1_A1
Optimization infeasible because this subproblem isn't reachable.

### Candidate
type: B, layer: 1, pos: 41

### Candidate
type: A, layer: 1, pos: 31

### Candidate
type: B, layer: 1, pos: 5

## BFS NS instance: NS_B1_A2_A1_B2_B1_A2_A2

### Backsubstitution after applying NS history:
0: -5.5050721, 15.4902201, -5.5000176, 15.4329062, -20.9379787, 20.9902382
1: -13.8271532, 23.7715569, -13.6598473, 23.8347454, -37.6618958, 37.4313889
2: -9.5997639, 21.5837345, -9.5434170, 21.6267891, -31.2265530, 31.1271515
3: -14.8570557, 26.3102455, -14.7877769, 26.3213177, -41.1783714, 41.0980225
4: -13.3813782, 26.7095051, -13.5414724, 26.6806316, -40.0620041, 40.2509766

Time for backsubstitution: 0.87 seconds

### NS candidates at layer 1
type: A, layer: 1, pos: 12
type: B, layer: 1, pos: 12
type: B, layer: 1, pos: 45
type: B, layer: 1, pos: 15
type: A, layer: 1, pos: 15
type: A, layer: 1, pos: 45
type: B, layer: 1, pos: 20
type: B, layer: 1, pos: 48
type: B, layer: 1, pos: 8
type: A, layer: 1, pos: 24
type: A, layer: 1, pos: 8
type: B, layer: 1, pos: 39
type: A, layer: 1, pos: 6
type: B, layer: 1, pos: 38
type: A, layer: 1, pos: 37
type: A, layer: 1, pos: 28
type: B, layer: 1, pos: 9
type: B, layer: 1, pos: 37
type: B, layer: 1, pos: 24
type: A, layer: 1, pos: 39
type: A, layer: 1, pos: 38
type: B, layer: 1, pos: 28
type: A, layer: 1, pos: 0
type: A, layer: 1, pos: 48
type: B, layer: 1, pos: 0
type: A, layer: 1, pos: 5
type: B, layer: 1, pos: 26
type: B, layer: 1, pos: 31
type: A, layer: 1, pos: 26
type: A, layer: 1, pos: 41
type: B, layer: 1, pos: 41
type: A, layer: 1, pos: 31
type: B, layer: 1, pos: 5

Time for candidate selection: 0.07 seconds

### Candidate
type: A, layer: 1, pos: 12

### Candidate
type: B, layer: 1, pos: 12

### Candidate
type: B, layer: 1, pos: 45

### Candidate
type: B, layer: 1, pos: 15

### Candidate
type: A, layer: 1, pos: 15

### Candidate
type: A, layer: 1, pos: 45

### Candidate
type: B, layer: 1, pos: 20

## Relational analysis of NS_B1_A2_A1_B2_B1_A2_A2_B1
Optimization infeasible because this subproblem isn't reachable.

### Candidate
type: B, layer: 1, pos: 48

## Relational analysis of NS_B1_A2_A1_B2_B1_A2_A2_B1
Optimization infeasible because this subproblem isn't reachable.

### Candidate
type: B, layer: 1, pos: 8

## Relational analysis of NS_B1_A2_A1_B2_B1_A2_A2_B1

### Relational analysis result of NS_B1_A2_A1_B2_B1_A2_A2_B1
Status: Status.UNKNOWN
Output dim: 0, lower bound: -20.9262506, upper bound: 20.9387399
time: 0.63 seconds

## Relational analysis of NS_B1_A2_A1_B2_B1_A2_A2_B2

### Relational analysis result of NS_B1_A2_A1_B2_B1_A2_A2_B2
Status: Status.UNKNOWN
Output dim: 0, lower bound: -20.9262643, upper bound: 20.9387501
time: 0.77 seconds

## BFS NS instance: NS_B1_A2_A1_B2_B2_B1_A1

### Backsubstitution after applying NS history:
0: -5.9219766, 16.6530056, -6.1855474, 17.1943378, -23.1163139, 22.8385525
1: -14.8774786, 25.5903778, -15.4049635, 26.3551464, -41.2326241, 40.9953423
2: -10.3299103, 23.1856422, -10.7583294, 23.9353752, -34.2652855, 33.9439697
3: -15.9785748, 28.2802773, -16.6418781, 29.1695862, -45.1481590, 44.9221573
4: -14.4363346, 28.6716270, -15.2073355, 29.5463696, -43.9827042, 43.8789635

Time for backsubstitution: 0.88 seconds

### NS candidates at layer 1
type: B, layer: 1, pos: 12
type: A, layer: 1, pos: 12
type: B, layer: 1, pos: 45
type: B, layer: 1, pos: 15
type: A, layer: 1, pos: 15
type: A, layer: 1, pos: 45
type: B, layer: 1, pos: 28
type: A, layer: 1, pos: 28
type: A, layer: 1, pos: 24
type: B, layer: 1, pos: 48
type: B, layer: 1, pos: 24
type: B, layer: 1, pos: 39
type: A, layer: 1, pos: 37
type: B, layer: 1, pos: 37
type: B, layer: 1, pos: 8
type: B, layer: 1, pos: 38
type: A, layer: 1, pos: 38
type: A, layer: 1, pos: 6
type: A, layer: 1, pos: 8
type: A, layer: 1, pos: 39
type: B, layer: 1, pos: 26
type: A, layer: 1, pos: 20
type: A, layer: 1, pos: 48
type: A, layer: 1, pos: 26
type: B, layer: 1, pos: 20
type: A, layer: 1, pos: 5
type: B, layer: 1, pos: 41
type: A, layer: 1, pos: 41
type: A, layer: 1, pos: 0
type: B, layer: 1, pos: 31
type: B, layer: 1, pos: 0
type: A, layer: 1, pos: 31
type: B, layer: 1, pos: 5

Time for candidate selection: 0.07 seconds

### Candidate
type: B, layer: 1, pos: 12

### Candidate
type: A, layer: 1, pos: 12

### Candidate
type: B, layer: 1, pos: 45

### Candidate
type: B, layer: 1, pos: 15

### Candidate
type: A, layer: 1, pos: 15

### Candidate
type: A, layer: 1, pos: 45

### Candidate
type: B, layer: 1, pos: 28

### Candidate
type: A, layer: 1, pos: 28

### Candidate
type: A, layer: 1, pos: 24

### Candidate
type: B, layer: 1, pos: 48

### Candidate
type: B, layer: 1, pos: 24

### Candidate
type: B, layer: 1, pos: 39

### Candidate
type: A, layer: 1, pos: 37

## Relational analysis of NS_B1_A2_A1_B2_B2_B1_A1_A1
Optimization infeasible because this subproblem isn't reachable.

### Candidate
type: B, layer: 1, pos: 37

### Candidate
type: B, layer: 1, pos: 8

## Relational analysis of NS_B1_A2_A1_B2_B2_B1_A1_B1

### Relational analysis result of NS_B1_A2_A1_B2_B2_B1_A1_B1
Status: Status.UNKNOWN
Output dim: 0, lower bound: -20.9224177, upper bound: 20.9270178
time: 0.60 seconds

## Relational analysis of NS_B1_A2_A1_B2_B2_B1_A1_B2

### Relational analysis result of NS_B1_A2_A1_B2_B2_B1_A1_B2
Status: Status.UNKNOWN
Output dim: 0, lower bound: -20.9022185, upper bound: 20.9269892
time: 0.65 seconds

## BFS NS instance: NS_B1_A2_A1_B2_B2_B1_A2

### Backsubstitution after applying NS history:
0: -5.8575258, 16.4333668, -6.1855474, 17.1943378, -23.0518627, 22.6189137
1: -14.7730999, 25.1493797, -15.4049635, 26.3551464, -41.1282463, 40.5543442
2: -10.2441244, 22.8457184, -10.7583294, 23.9353752, -34.1795006, 33.6040459
3: -15.8305902, 27.8522625, -16.6418781, 29.1695862, -45.0001717, 44.4941406
4: -14.2308779, 28.2878342, -15.2073355, 29.5463696, -43.7772484, 43.4951706

Time for backsubstitution: 0.88 seconds

### NS candidates at layer 1
type: B, layer: 1, pos: 12
type: A, layer: 1, pos: 12
type: B, layer: 1, pos: 45
type: A, layer: 1, pos: 45
type: B, layer: 1, pos: 15
type: A, layer: 1, pos: 15
type: B, layer: 1, pos: 28
type: A, layer: 1, pos: 28
type: A, layer: 1, pos: 24
type: B, layer: 1, pos: 48
type: B, layer: 1, pos: 39
type: B, layer: 1, pos: 24
type: B, layer: 1, pos: 8
type: A, layer: 1, pos: 37
type: B, layer: 1, pos: 38
type: B, layer: 1, pos: 37
type: A, layer: 1, pos: 38
type: A, layer: 1, pos: 8
type: A, layer: 1, pos: 20
type: A, layer: 1, pos: 6
type: A, layer: 1, pos: 39
type: B, layer: 1, pos: 26
type: A, layer: 1, pos: 26
type: B, layer: 1, pos: 20
type: A, layer: 1, pos: 48
type: A, layer: 1, pos: 5
type: B, layer: 1, pos: 41
type: A, layer: 1, pos: 41
type: A, layer: 1, pos: 0
type: B, layer: 1, pos: 31
type: A, layer: 1, pos: 31
type: B, layer: 1, pos: 0
type: B, layer: 1, pos: 5

Time for candidate selection: 0.07 seconds

### Candidate
type: B, layer: 1, pos: 12

### Candidate
type: A, layer: 1, pos: 12

### Candidate
type: B, layer: 1, pos: 45

### Candidate
type: A, layer: 1, pos: 45

### Candidate
type: B, layer: 1, pos: 15

### Candidate
type: A, layer: 1, pos: 15

### Candidate
type: B, layer: 1, pos: 28

### Candidate
type: A, layer: 1, pos: 28

### Candidate
type: A, layer: 1, pos: 24

### Candidate
type: B, layer: 1, pos: 48

### Candidate
type: B, layer: 1, pos: 39

### Candidate
type: B, layer: 1, pos: 24

### Candidate
type: B, layer: 1, pos: 8

## Relational analysis of NS_B1_A2_A1_B2_B2_B1_A2_B1

### Relational analysis result of NS_B1_A2_A1_B2_B2_B1_A2_B1
Status: Status.UNKNOWN
Output dim: 0, lower bound: -20.9023869, upper bound: 20.9270178
time: 0.72 seconds

## Relational analysis of NS_B1_A2_A1_B2_B2_B1_A2_B2

### Relational analysis result of NS_B1_A2_A1_B2_B2_B1_A2_B2
Status: Status.UNKNOWN
Output dim: 0, lower bound: -20.9216354, upper bound: 20.9433910
time: 0.61 seconds

## BFS NS instance: NS_B1_A2_A1_B2_B2_B2_B1

### Backsubstitution after applying NS history:
0: -5.6015358, 15.8106499, -5.6003308, 15.6679659, -21.2694969, 21.4109764
1: -14.0442076, 24.3673801, -13.9723759, 24.0450115, -38.0892181, 38.3397446
2: -9.7557278, 22.0505695, -9.7245750, 21.8221474, -31.5778751, 31.7751446
3: -15.1043482, 26.9022884, -15.0771303, 26.6096878, -41.7140274, 41.9794197
4: -13.6535530, 27.2579956, -13.6874018, 26.9241676, -40.5777206, 40.9453964

Time for backsubstitution: 0.88 seconds

### NS candidates at layer 1
type: A, layer: 1, pos: 12
type: B, layer: 1, pos: 12
type: B, layer: 1, pos: 45
type: A, layer: 1, pos: 45
type: B, layer: 1, pos: 28
type: B, layer: 1, pos: 15
type: A, layer: 1, pos: 15
type: B, layer: 1, pos: 48
type: B, layer: 1, pos: 39
type: B, layer: 1, pos: 24
type: A, layer: 1, pos: 24
type: A, layer: 1, pos: 28
type: A, layer: 1, pos: 39
type: A, layer: 1, pos: 8
type: A, layer: 1, pos: 9
type: B, layer: 1, pos: 38
type: A, layer: 1, pos: 37
type: B, layer: 1, pos: 37
type: B, layer: 1, pos: 20
type: A, layer: 1, pos: 38
type: A, layer: 1, pos: 20
type: A, layer: 1, pos: 26
type: A, layer: 1, pos: 48
type: B, layer: 1, pos: 26
type: A, layer: 1, pos: 6
type: B, layer: 1, pos: 41
type: A, layer: 1, pos: 41
type: B, layer: 1, pos: 0
type: B, layer: 1, pos: 31
type: A, layer: 1, pos: 5
type: A, layer: 1, pos: 0
type: A, layer: 1, pos: 31
type: B, layer: 1, pos: 5

Time for candidate selection: 0.07 seconds

### Candidate
type: A, layer: 1, pos: 12

### Candidate
type: B, layer: 1, pos: 12

### Candidate
type: B, layer: 1, pos: 45

### Candidate
type: A, layer: 1, pos: 45

### Candidate
type: B, layer: 1, pos: 28

### Candidate
type: B, layer: 1, pos: 15

### Candidate
type: A, layer: 1, pos: 15

### Candidate
type: B, layer: 1, pos: 48

### Candidate
type: B, layer: 1, pos: 39

### Candidate
type: B, layer: 1, pos: 24

### Candidate
type: A, layer: 1, pos: 24

### Candidate
type: A, layer: 1, pos: 28

### Candidate
type: A, layer: 1, pos: 39

### Candidate
type: A, layer: 1, pos: 8

## Relational analysis of NS_B1_A2_A1_B2_B2_B2_B1_A1

### Relational analysis result of NS_B1_A2_A1_B2_B2_B2_B1_A1
Status: Status.UNKNOWN
Output dim: 0, lower bound: -20.9403245, upper bound: 20.9473847
time: 0.68 seconds

## Relational analysis of NS_B1_A2_A1_B2_B2_B2_B1_A2

### Relational analysis result of NS_B1_A2_A1_B2_B2_B2_B1_A2
Status: Status.UNKNOWN
Output dim: 0, lower bound: -20.9403245, upper bound: 20.9484328
time: 0.60 seconds

## BFS NS instance: NS_B1_A2_A1_B2_B2_B2_B2

### Backsubstitution after applying NS history:
0: -5.9480605, 16.7152119, -6.1004305, 16.9416161, -22.8896770, 22.8156357
1: -14.9457874, 25.6828289, -15.2664204, 25.8574886, -40.8032761, 40.9492455
2: -10.3815269, 23.2739506, -10.6415129, 23.5541840, -33.9357071, 33.9154625
3: -16.0481339, 28.3843365, -16.4355812, 28.6546135, -44.7027435, 44.8199081
4: -14.5056696, 28.7776871, -14.9459438, 29.1046467, -43.6103134, 43.7236252

Time for backsubstitution: 0.89 seconds

### NS candidates at layer 1
type: B, layer: 1, pos: 12
type: A, layer: 1, pos: 12
type: B, layer: 1, pos: 45
type: B, layer: 1, pos: 15
type: B, layer: 1, pos: 28
type: A, layer: 1, pos: 15
type: A, layer: 1, pos: 45
type: A, layer: 1, pos: 28
type: B, layer: 1, pos: 48
type: A, layer: 1, pos: 24
type: B, layer: 1, pos: 24
type: A, layer: 1, pos: 37
type: B, layer: 1, pos: 39
type: B, layer: 1, pos: 37
type: B, layer: 1, pos: 38
type: A, layer: 1, pos: 8
type: A, layer: 1, pos: 38
type: A, layer: 1, pos: 9
type: A, layer: 1, pos: 6
type: A, layer: 1, pos: 39
type: A, layer: 1, pos: 48
type: B, layer: 1, pos: 26
type: A, layer: 1, pos: 26
type: B, layer: 1, pos: 20
type: A, layer: 1, pos: 20
type: B, layer: 1, pos: 41
type: A, layer: 1, pos: 41
type: A, layer: 1, pos: 5
type: B, layer: 1, pos: 0
type: A, layer: 1, pos: 0
type: B, layer: 1, pos: 31
type: A, layer: 1, pos: 31
type: B, layer: 1, pos: 5

Time for candidate selection: 0.07 seconds

### Candidate
type: B, layer: 1, pos: 12

### Candidate
type: A, layer: 1, pos: 12

### Candidate
type: B, layer: 1, pos: 45

### Candidate
type: B, layer: 1, pos: 15

### Candidate
type: B, layer: 1, pos: 28

### Candidate
type: A, layer: 1, pos: 15

### Candidate
type: A, layer: 1, pos: 45

### Candidate
type: A, layer: 1, pos: 28

### Candidate
type: B, layer: 1, pos: 48

### Candidate
type: A, layer: 1, pos: 24

### Candidate
type: B, layer: 1, pos: 24

### Candidate
type: A, layer: 1, pos: 37

### Candidate
type: B, layer: 1, pos: 39

### Candidate
type: B, layer: 1, pos: 37

### Candidate
type: B, layer: 1, pos: 38

## Relational analysis of NS_B1_A2_A1_B2_B2_B2_B2_B1
Optimization infeasible because this subproblem isn't reachable.

### Candidate
type: A, layer: 1, pos: 8

## Relational analysis of NS_B1_A2_A1_B2_B2_B2_B2_A1

### Relational analysis result of NS_B1_A2_A1_B2_B2_B2_B2_A1
Status: Status.UNKNOWN
Output dim: 0, lower bound: -20.9323331, upper bound: 20.9502207
time: 0.60 seconds

## Relational analysis of NS_B1_A2_A1_B2_B2_B2_B2_A2

### Relational analysis result of NS_B1_A2_A1_B2_B2_B2_B2_A2
Status: Status.UNKNOWN
Output dim: 0, lower bound: -20.9450275, upper bound: 20.9513798
time: 0.65 seconds

## BFS NS instance: NS_B2_B1_A1_A2_A1_B1_A1

### Backsubstitution after applying NS history:
0: -5.5211277, 15.5494890, -5.6506472, 15.8846350, -21.4057617, 21.2001362
1: -13.8111639, 23.9801216, -14.0206594, 24.5762863, -38.3874435, 38.0007744
2: -9.6053791, 21.7101288, -9.7986345, 22.2425442, -31.8479233, 31.5087585
3: -14.8785429, 26.4883957, -15.1872730, 27.1004562, -41.9790001, 41.6756668
4: -13.4827013, 26.8253231, -13.9226294, 27.4241123, -40.9068146, 40.7479515

Time for backsubstitution: 0.89 seconds

### NS candidates at layer 1
type: A, layer: 1, pos: 1
type: B, layer: 1, pos: 45
type: A, layer: 1, pos: 45
type: B, layer: 1, pos: 15
type: B, layer: 1, pos: 47
type: B, layer: 1, pos: 48
type: B, layer: 1, pos: 38
type: A, layer: 1, pos: 38
type: B, layer: 1, pos: 9
type: A, layer: 1, pos: 28
type: A, layer: 1, pos: 9
type: A, layer: 1, pos: 8
type: B, layer: 1, pos: 6
type: B, layer: 1, pos: 20
type: A, layer: 1, pos: 6
type: A, layer: 1, pos: 37
type: B, layer: 1, pos: 28
type: A, layer: 1, pos: 20
type: B, layer: 1, pos: 8
type: B, layer: 1, pos: 37
type: B, layer: 1, pos: 31
type: A, layer: 1, pos: 31
type: A, layer: 1, pos: 48
type: A, layer: 1, pos: 24
type: B, layer: 1, pos: 24
type: A, layer: 1, pos: 5
type: A, layer: 1, pos: 0
type: B, layer: 1, pos: 26
type: B, layer: 1, pos: 41
type: A, layer: 1, pos: 26
type: A, layer: 1, pos: 41
type: B, layer: 1, pos: 0
type: B, layer: 1, pos: 5

Time for candidate selection: 0.07 seconds

### Candidate
type: A, layer: 1, pos: 1

### Candidate
type: B, layer: 1, pos: 45

### Candidate
type: A, layer: 1, pos: 45

### Candidate
type: B, layer: 1, pos: 15

### Candidate
type: B, layer: 1, pos: 47

### Candidate
type: B, layer: 1, pos: 48

### Candidate
type: B, layer: 1, pos: 38

### Candidate
type: A, layer: 1, pos: 38

### Candidate
type: B, layer: 1, pos: 9

### Candidate
type: A, layer: 1, pos: 28

### Candidate
type: A, layer: 1, pos: 9

### Candidate
type: A, layer: 1, pos: 8

## Relational analysis of NS_B2_B1_A1_A2_A1_B1_A1_A1
Optimization infeasible because this subproblem isn't reachable.

### Candidate
type: B, layer: 1, pos: 6

## Relational analysis of NS_B2_B1_A1_A2_A1_B1_A1_B1
Optimization infeasible because this subproblem isn't reachable.

### Candidate
type: B, layer: 1, pos: 20

## Relational analysis of NS_B2_B1_A1_A2_A1_B1_A1_B1
Optimization infeasible because this subproblem isn't reachable.

### Candidate
type: A, layer: 1, pos: 6

### Candidate
type: A, layer: 1, pos: 37

## Relational analysis of NS_B2_B1_A1_A2_A1_B1_A1_A1

### Relational analysis result of NS_B2_B1_A1_A2_A1_B1_A1_A1
Status: Status.UNKNOWN
Output dim: 0, lower bound: -20.9540109, upper bound: 20.9549650
time: 0.85 seconds

## Relational analysis of NS_B2_B1_A1_A2_A1_B1_A1_A2
Optimization infeasible because this subproblem isn't reachable.

### Candidate
type: B, layer: 1, pos: 28

### Candidate
type: A, layer: 1, pos: 20

### Candidate
type: B, layer: 1, pos: 8

### Candidate
type: B, layer: 1, pos: 37

### Candidate
type: B, layer: 1, pos: 31

## Relational analysis of NS_B2_B1_A1_A2_A1_B1_A1_B1

### Relational analysis result of NS_B2_B1_A1_A2_A1_B1_A1_B1
Status: Status.UNKNOWN
Output dim: 0, lower bound: -20.9602904, upper bound: 20.9605021
time: 0.74 seconds

## Relational analysis of NS_B2_B1_A1_A2_A1_B1_A1_B2

### Relational analysis result of NS_B2_B1_A1_A2_A1_B1_A1_B2
Status: Status.UNKNOWN
Output dim: 0, lower bound: -20.9602828, upper bound: 20.9605971
time: 0.65 seconds

## BFS NS instance: NS_B2_B1_A1_A2_A1_B1_A2

### Backsubstitution after applying NS history:
0: -6.4746389, 18.2255611, -5.6506472, 15.8846350, -22.3592739, 23.8762093
1: -16.3413219, 27.9796524, -14.0206594, 24.5762863, -40.9176102, 42.0003090
2: -11.2886572, 25.3120480, -9.7986345, 22.2425442, -33.5312004, 35.1106796
3: -17.4461594, 30.8622093, -15.1872730, 27.1004562, -44.5466156, 46.0494843
4: -15.6404037, 31.3343525, -13.9226294, 27.4241123, -43.0645142, 45.2569809

Time for backsubstitution: 0.89 seconds

### NS candidates at layer 1
type: A, layer: 1, pos: 1
type: B, layer: 1, pos: 15
type: B, layer: 1, pos: 47
type: B, layer: 1, pos: 45
type: B, layer: 1, pos: 48
type: B, layer: 1, pos: 6
type: A, layer: 1, pos: 45
type: A, layer: 1, pos: 6
type: A, layer: 1, pos: 28
type: A, layer: 1, pos: 38
type: B, layer: 1, pos: 38
type: B, layer: 1, pos: 28
type: B, layer: 1, pos: 9
type: A, layer: 1, pos: 37
type: A, layer: 1, pos: 9
type: B, layer: 1, pos: 37
type: A, layer: 1, pos: 20
type: A, layer: 1, pos: 24
type: A, layer: 1, pos: 8
type: B, layer: 1, pos: 8
type: A, layer: 1, pos: 48
type: B, layer: 1, pos: 31
type: B, layer: 1, pos: 24
type: A, layer: 1, pos: 31
type: A, layer: 1, pos: 0
type: B, layer: 1, pos: 20
type: B, layer: 1, pos: 0
type: B, layer: 1, pos: 26
type: A, layer: 1, pos: 26
type: A, layer: 1, pos: 5
type: B, layer: 1, pos: 41
type: A, layer: 1, pos: 41
type: B, layer: 1, pos: 5

Time for candidate selection: 0.07 seconds

### Candidate
type: A, layer: 1, pos: 1

### Candidate
type: B, layer: 1, pos: 15

### Candidate
type: B, layer: 1, pos: 47

### Candidate
type: B, layer: 1, pos: 45

### Candidate
type: B, layer: 1, pos: 48

### Candidate
type: B, layer: 1, pos: 6

## Relational analysis of NS_B2_B1_A1_A2_A1_B1_A2_B1
Optimization infeasible because this subproblem isn't reachable.

### Candidate
type: A, layer: 1, pos: 45

### Candidate
type: A, layer: 1, pos: 6

### Candidate
type: A, layer: 1, pos: 28

### Candidate
type: A, layer: 1, pos: 38

### Candidate
type: B, layer: 1, pos: 38

### Candidate
type: B, layer: 1, pos: 28

### Candidate
type: B, layer: 1, pos: 9

### Candidate
type: A, layer: 1, pos: 37

## Relational analysis of NS_B2_B1_A1_A2_A1_B1_A2_A1

### Relational analysis result of NS_B2_B1_A1_A2_A1_B1_A2_A1
Status: Status.UNKNOWN
Output dim: 0, lower bound: -20.9540109, upper bound: 20.9549650
time: 0.87 seconds

## Relational analysis of NS_B2_B1_A1_A2_A1_B1_A2_A2

### Relational analysis result of NS_B2_B1_A1_A2_A1_B1_A2_A2
Status: Status.UNKNOWN
Output dim: 0, lower bound: -20.9535335, upper bound: 20.9549650
time: 0.71 seconds

## BFS NS instance: NS_B2_B1_A1_A2_A1_B2_A1

### Backsubstitution after applying NS history:
0: -5.5173144, 15.5399466, -6.8067040, 19.0791645, -24.5964737, 22.3466511
1: -13.8016233, 23.9660931, -17.1316872, 29.2740116, -43.0756302, 41.0977783
2: -9.5986023, 21.6973820, -11.8529787, 26.5167751, -36.1153793, 33.5503616
3: -14.8683949, 26.4723930, -18.3257275, 32.2822495, -47.1506424, 44.7981148
4: -13.4734020, 26.8094215, -16.5295219, 32.8072777, -46.2806778, 43.3389435

Time for backsubstitution: 0.90 seconds

### NS candidates at layer 1
type: A, layer: 1, pos: 1
type: B, layer: 1, pos: 45
type: A, layer: 1, pos: 45
type: A, layer: 1, pos: 28
type: B, layer: 1, pos: 15
type: B, layer: 1, pos: 48
type: A, layer: 1, pos: 6
type: B, layer: 1, pos: 6
type: B, layer: 1, pos: 47
type: B, layer: 1, pos: 38
type: A, layer: 1, pos: 38
type: B, layer: 1, pos: 28
type: A, layer: 1, pos: 9
type: B, layer: 1, pos: 9
type: A, layer: 1, pos: 8
type: A, layer: 1, pos: 37
type: B, layer: 1, pos: 20
type: B, layer: 1, pos: 37
type: B, layer: 1, pos: 24
type: A, layer: 1, pos: 48
type: A, layer: 1, pos: 31
type: B, layer: 1, pos: 8
type: B, layer: 1, pos: 31
type: A, layer: 1, pos: 24
type: B, layer: 1, pos: 0
type: A, layer: 1, pos: 20
type: B, layer: 1, pos: 26
type: A, layer: 1, pos: 0
type: A, layer: 1, pos: 26
type: A, layer: 1, pos: 5
type: B, layer: 1, pos: 41
type: A, layer: 1, pos: 41
type: B, layer: 1, pos: 5

Time for candidate selection: 0.07 seconds

### Candidate
type: A, layer: 1, pos: 1

### Candidate
type: B, layer: 1, pos: 45

### Candidate
type: A, layer: 1, pos: 45

### Candidate
type: A, layer: 1, pos: 28

## Relational analysis of NS_B2_B1_A1_A2_A1_B2_A1_A1
Optimization infeasible because this subproblem isn't reachable.

### Candidate
type: B, layer: 1, pos: 15

### Candidate
type: B, layer: 1, pos: 48

## Relational analysis of NS_B2_B1_A1_A2_A1_B2_A1_B1
Optimization infeasible because this subproblem isn't reachable.

### Candidate
type: A, layer: 1, pos: 6

## Relational analysis of NS_B2_B1_A1_A2_A1_B2_A1_A1
Optimization infeasible because this subproblem isn't reachable.

### Candidate
type: B, layer: 1, pos: 6

### Candidate
type: B, layer: 1, pos: 47

## Relational analysis of NS_B2_B1_A1_A2_A1_B2_A1_B1

### Relational analysis result of NS_B2_B1_A1_A2_A1_B2_A1_B1
Status: Status.UNKNOWN
Output dim: 0, lower bound: -20.9607707, upper bound: 20.9620561
time: 0.62 seconds

## Relational analysis of NS_B2_B1_A1_A2_A1_B2_A1_B2

### Relational analysis result of NS_B2_B1_A1_A2_A1_B2_A1_B2
Status: Status.UNKNOWN
Output dim: 0, lower bound: -20.9607707, upper bound: 20.9620561
time: 0.76 seconds

## BFS NS instance: NS_B2_B1_A1_A2_A1_B2_A2

### Backsubstitution after applying NS history:
0: -6.4746389, 18.2255611, -6.8067040, 19.0791645, -25.5538025, 25.0322647
1: -16.3413219, 27.9796524, -17.1316872, 29.2740116, -45.6153336, 45.1113358
2: -11.2886572, 25.3120480, -11.8529787, 26.5167751, -37.8054314, 37.1650276
3: -17.4461594, 30.8622093, -18.3257275, 32.2822495, -49.7284050, 49.1879349
4: -15.6404037, 31.3343525, -16.5295219, 32.8072777, -48.4476814, 47.8638763

Time for backsubstitution: 0.90 seconds

### NS candidates at layer 1
type: A, layer: 1, pos: 1
type: B, layer: 1, pos: 6
type: B, layer: 1, pos: 47
type: A, layer: 1, pos: 6
type: B, layer: 1, pos: 15
type: A, layer: 1, pos: 28
type: A, layer: 1, pos: 8
type: B, layer: 1, pos: 9
type: A, layer: 1, pos: 9
type: B, layer: 1, pos: 48
type: A, layer: 1, pos: 24
type: A, layer: 1, pos: 37
type: B, layer: 1, pos: 24
type: B, layer: 1, pos: 45
type: B, layer: 1, pos: 38
type: B, layer: 1, pos: 8
type: B, layer: 1, pos: 28
type: A, layer: 1, pos: 45
type: B, layer: 1, pos: 37
type: A, layer: 1, pos: 38
type: A, layer: 1, pos: 20
type: B, layer: 1, pos: 20
type: B, layer: 1, pos: 31
type: A, layer: 1, pos: 31
type: B, layer: 1, pos: 0
type: B, layer: 1, pos: 26
type: A, layer: 1, pos: 26
type: A, layer: 1, pos: 48
type: A, layer: 1, pos: 0
type: A, layer: 1, pos: 5
type: B, layer: 1, pos: 41
type: A, layer: 1, pos: 41
type: B, layer: 1, pos: 5

Time for candidate selection: 0.07 seconds

### Candidate
type: A, layer: 1, pos: 1

### Candidate
type: B, layer: 1, pos: 6

## Relational analysis of NS_B2_B1_A1_A2_A1_B2_A2_B1
Optimization infeasible because this subproblem isn't reachable.

### Candidate
type: B, layer: 1, pos: 47

## Relational analysis of NS_B2_B1_A1_A2_A1_B2_A2_B1

### Relational analysis result of NS_B2_B1_A1_A2_A1_B2_A2_B1
Status: Status.UNKNOWN
Output dim: 0, lower bound: -20.9607707, upper bound: 20.9614392
time: 0.68 seconds

## Relational analysis of NS_B2_B1_A1_A2_A1_B2_A2_B2

### Relational analysis result of NS_B2_B1_A1_A2_A1_B2_A2_B2
Status: Status.UNKNOWN
Output dim: 0, lower bound: -20.9607707, upper bound: 20.9614392
time: 0.77 seconds

## BFS NS instance: NS_B2_B1_A1_A2_A2_B1_A1

### Backsubstitution after applying NS history:
0: -5.8064957, 16.2564983, -5.6711140, 15.9823341, -21.7888260, 21.9276123
1: -14.4214268, 25.1157417, -14.2006493, 24.6264076, -39.0478249, 39.3163872
2: -10.0803680, 22.7406292, -9.8695850, 22.2990055, -32.3793716, 32.6102142
3: -15.6071787, 27.7256927, -15.2870331, 27.1922836, -42.7994614, 43.0127258
4: -14.3034105, 28.0350780, -13.8500452, 27.5519333, -41.8553314, 41.8851242

Time for backsubstitution: 0.90 seconds

### NS candidates at layer 1
type: A, layer: 1, pos: 1
type: B, layer: 1, pos: 45
type: A, layer: 1, pos: 45
type: B, layer: 1, pos: 15
type: B, layer: 1, pos: 38
type: A, layer: 1, pos: 38
type: A, layer: 1, pos: 9
type: B, layer: 1, pos: 28
type: B, layer: 1, pos: 9
type: A, layer: 1, pos: 48
type: B, layer: 1, pos: 39
type: A, layer: 1, pos: 28
type: A, layer: 1, pos: 6
type: A, layer: 1, pos: 8
type: B, layer: 1, pos: 37
type: B, layer: 1, pos: 20
type: B, layer: 1, pos: 48
type: B, layer: 1, pos: 6
type: A, layer: 1, pos: 37
type: A, layer: 1, pos: 31
type: B, layer: 1, pos: 8
type: B, layer: 1, pos: 31
type: B, layer: 1, pos: 24
type: A, layer: 1, pos: 20
type: A, layer: 1, pos: 24
type: B, layer: 1, pos: 0
type: A, layer: 1, pos: 26
type: B, layer: 1, pos: 26
type: B, layer: 1, pos: 41
type: A, layer: 1, pos: 41
type: B, layer: 1, pos: 5
type: A, layer: 1, pos: 0
type: A, layer: 1, pos: 5

Time for candidate selection: 0.08 seconds

### Candidate
type: A, layer: 1, pos: 1

### Candidate
type: B, layer: 1, pos: 45

### Candidate
type: A, layer: 1, pos: 45

### Candidate
type: B, layer: 1, pos: 15

## Relational analysis of NS_B2_B1_A1_A2_A2_B1_A1_B1
Optimization infeasible because this subproblem isn't reachable.

### Candidate
type: B, layer: 1, pos: 38

## Relational analysis of NS_B2_B1_A1_A2_A2_B1_A1_B1
Optimization infeasible because this subproblem isn't reachable.

### Candidate
type: A, layer: 1, pos: 38

### Candidate
type: A, layer: 1, pos: 9

## Relational analysis of NS_B2_B1_A1_A2_A2_B1_A1_A1

### Relational analysis result of NS_B2_B1_A1_A2_A2_B1_A1_A1
Status: Status.UNKNOWN
Output dim: 0, lower bound: -20.9510084, upper bound: 20.9453079
time: 0.72 seconds

## Relational analysis of NS_B2_B1_A1_A2_A2_B1_A1_A2

### Relational analysis result of NS_B2_B1_A1_A2_A2_B1_A1_A2
Status: Status.UNKNOWN
Output dim: 0, lower bound: -20.9610559, upper bound: 20.9607007
time: 0.65 seconds

## BFS NS instance: NS_B2_B1_A1_A2_A2_B1_A2

### Backsubstitution after applying NS history:
0: -6.8216963, 19.0988522, -5.6047597, 15.8112345, -22.6329250, 24.7036114
1: -17.1398964, 29.3009758, -14.0175161, 24.3764324, -41.5163269, 43.3184891
2: -11.8816538, 26.5409317, -9.7476950, 22.0709076, -33.9525566, 36.2886276
3: -18.3513393, 32.3226967, -15.1046944, 26.9131718, -45.2645111, 47.4273872
4: -16.5870399, 32.8193359, -13.6979666, 27.2654343, -43.8524666, 46.5173035

Time for backsubstitution: 0.91 seconds

### NS candidates at layer 1
type: A, layer: 1, pos: 1
type: B, layer: 1, pos: 6
type: A, layer: 1, pos: 6
type: B, layer: 1, pos: 45
type: A, layer: 1, pos: 45
type: B, layer: 1, pos: 28
type: B, layer: 1, pos: 15
type: B, layer: 1, pos: 24
type: B, layer: 1, pos: 39
type: A, layer: 1, pos: 24
type: B, layer: 1, pos: 9
type: A, layer: 1, pos: 9
type: B, layer: 1, pos: 37
type: A, layer: 1, pos: 38
type: B, layer: 1, pos: 38
type: A, layer: 1, pos: 8
type: A, layer: 1, pos: 28
type: A, layer: 1, pos: 37
type: A, layer: 1, pos: 48
type: B, layer: 1, pos: 8
type: B, layer: 1, pos: 48
type: A, layer: 1, pos: 31
type: B, layer: 1, pos: 31
type: B, layer: 1, pos: 0
type: B, layer: 1, pos: 20
type: A, layer: 1, pos: 26
type: B, layer: 1, pos: 26
type: A, layer: 1, pos: 0
type: A, layer: 1, pos: 20
type: B, layer: 1, pos: 5
type: B, layer: 1, pos: 41
type: A, layer: 1, pos: 41
type: A, layer: 1, pos: 5

Time for candidate selection: 0.07 seconds

### Candidate
type: A, layer: 1, pos: 1

### Candidate
type: B, layer: 1, pos: 6

## Relational analysis of NS_B2_B1_A1_A2_A2_B1_A2_B1
Optimization infeasible because this subproblem isn't reachable.

### Candidate
type: A, layer: 1, pos: 6

### Candidate
type: B, layer: 1, pos: 45

### Candidate
type: A, layer: 1, pos: 45

### Candidate
type: B, layer: 1, pos: 28

## Relational analysis of NS_B2_B1_A1_A2_A2_B1_A2_B1

### Relational analysis result of NS_B2_B1_A1_A2_A2_B1_A2_B1
Status: Status.UNKNOWN
Output dim: 0, lower bound: -20.9447425, upper bound: 20.9446478
time: 0.70 seconds

## Relational analysis of NS_B2_B1_A1_A2_A2_B1_A2_B2
Optimization infeasible because this subproblem isn't reachable.

### Candidate
type: B, layer: 1, pos: 15

## Relational analysis of NS_B2_B1_A1_A2_A2_B1_A2_B1
Optimization infeasible because this subproblem isn't reachable.

### Candidate
type: B, layer: 1, pos: 24

## Relational analysis of NS_B2_B1_A1_A2_A2_B1_A2_B1

### Relational analysis result of NS_B2_B1_A1_A2_A2_B1_A2_B1
Status: Status.UNKNOWN
Output dim: 0, lower bound: -20.9613692, upper bound: 20.9602891
time: 0.66 seconds

## Relational analysis of NS_B2_B1_A1_A2_A2_B1_A2_B2

### Relational analysis result of NS_B2_B1_A1_A2_A2_B1_A2_B2
Status: Status.UNKNOWN
Output dim: 0, lower bound: -20.9607348, upper bound: 20.9602994
time: 0.67 seconds

## BFS NS instance: NS_B2_B1_A1_A2_A2_B2_A1

### Backsubstitution after applying NS history:
0: -5.8064957, 16.2564983, -5.9259515, 16.5996265, -22.4061203, 22.1824493
1: -14.4214268, 25.1157417, -14.7052040, 25.6382656, -40.0596924, 39.8209419
2: -10.0803680, 22.7406292, -10.2782459, 23.2060642, -33.2864304, 33.0188751
3: -15.6071787, 27.7256927, -15.9218321, 28.2842026, -43.8913765, 43.6475220
4: -14.3034105, 28.0350780, -14.6089115, 28.6121178, -42.9155159, 42.6439896

Time for backsubstitution: 0.91 seconds

### NS candidates at layer 1
type: A, layer: 1, pos: 1
type: B, layer: 1, pos: 45
type: B, layer: 1, pos: 15
type: A, layer: 1, pos: 45
type: B, layer: 1, pos: 38
type: A, layer: 1, pos: 38
type: B, layer: 1, pos: 9
type: B, layer: 1, pos: 39
type: A, layer: 1, pos: 9
type: A, layer: 1, pos: 8
type: A, layer: 1, pos: 28
type: B, layer: 1, pos: 8
type: B, layer: 1, pos: 20
type: A, layer: 1, pos: 6
type: A, layer: 1, pos: 37
type: B, layer: 1, pos: 28
type: B, layer: 1, pos: 37
type: B, layer: 1, pos: 48
type: B, layer: 1, pos: 6
type: A, layer: 1, pos: 20
type: A, layer: 1, pos: 48
type: B, layer: 1, pos: 31
type: A, layer: 1, pos: 31
type: B, layer: 1, pos: 24
type: A, layer: 1, pos: 24
type: B, layer: 1, pos: 26
type: A, layer: 1, pos: 26
type: A, layer: 1, pos: 5
type: B, layer: 1, pos: 41
type: A, layer: 1, pos: 41
type: A, layer: 1, pos: 0
type: B, layer: 1, pos: 0
type: B, layer: 1, pos: 5

Time for candidate selection: 0.08 seconds

### Candidate
type: A, layer: 1, pos: 1

### Candidate
type: B, layer: 1, pos: 45

### Candidate
type: B, layer: 1, pos: 15

### Candidate
type: A, layer: 1, pos: 45

### Candidate
type: B, layer: 1, pos: 38

## Relational analysis of NS_B2_B1_A1_A2_A2_B2_A1_B1
Optimization infeasible because this subproblem isn't reachable.

### Candidate
type: A, layer: 1, pos: 38

### Candidate
type: B, layer: 1, pos: 9

## Relational analysis of NS_B2_B1_A1_A2_A2_B2_A1_B1

### Relational analysis result of NS_B2_B1_A1_A2_A2_B2_A1_B1
Status: Status.UNKNOWN
Output dim: 0, lower bound: -20.9399659, upper bound: 20.9416250
time: 1.19 seconds

## Relational analysis of NS_B2_B1_A1_A2_A2_B2_A1_B2

### Relational analysis result of NS_B2_B1_A1_A2_A2_B2_A1_B2
Status: Status.UNKNOWN
Output dim: 0, lower bound: -20.9602917, upper bound: 20.9606105
time: 0.74 seconds

## BFS NS instance: NS_B2_B1_A1_A2_A2_B2_A2

### Backsubstitution after applying NS history:
0: -6.8216963, 19.0988522, -5.8502169, 16.4093361, -23.2310295, 24.9490700
1: -17.1398964, 29.3009758, -14.5010986, 25.3585262, -42.4984169, 43.8020706
2: -11.8816538, 26.5409317, -10.1413927, 22.9502277, -34.8318787, 36.6823235
3: -18.3513393, 32.3226967, -15.7145090, 27.9702492, -46.3215866, 48.0372009
4: -16.5870399, 32.8193359, -14.4324970, 28.2923603, -44.8793945, 47.2518311

Time for backsubstitution: 0.91 seconds

### NS candidates at layer 1
type: A, layer: 1, pos: 1
type: B, layer: 1, pos: 15
type: B, layer: 1, pos: 39
type: B, layer: 1, pos: 45
type: B, layer: 1, pos: 6
type: A, layer: 1, pos: 45
type: B, layer: 1, pos: 48
type: A, layer: 1, pos: 6
type: B, layer: 1, pos: 28
type: A, layer: 1, pos: 38
type: B, layer: 1, pos: 38
type: A, layer: 1, pos: 28
type: B, layer: 1, pos: 9
type: A, layer: 1, pos: 9
type: A, layer: 1, pos: 48
type: A, layer: 1, pos: 37
type: B, layer: 1, pos: 37
type: A, layer: 1, pos: 24
type: A, layer: 1, pos: 8
type: A, layer: 1, pos: 20
type: B, layer: 1, pos: 8
type: B, layer: 1, pos: 24
type: B, layer: 1, pos: 31
type: A, layer: 1, pos: 31
type: A, layer: 1, pos: 0
type: B, layer: 1, pos: 20
type: B, layer: 1, pos: 0
type: B, layer: 1, pos: 26
type: A, layer: 1, pos: 26
type: A, layer: 1, pos: 5
type: B, layer: 1, pos: 41
type: A, layer: 1, pos: 41
type: B, layer: 1, pos: 5

Time for candidate selection: 0.07 seconds

### Candidate
type: A, layer: 1, pos: 1

### Candidate
type: B, layer: 1, pos: 15

### Candidate
type: B, layer: 1, pos: 39

## Relational analysis of NS_B2_B1_A1_A2_A2_B2_A2_B1

### Relational analysis result of NS_B2_B1_A1_A2_A2_B2_A2_B1
Status: Status.UNKNOWN
Output dim: 0, lower bound: -20.9607285, upper bound: 20.9603972
time: 0.77 seconds

## Relational analysis of NS_B2_B1_A1_A2_A2_B2_A2_B2

### Relational analysis result of NS_B2_B1_A1_A2_A2_B2_A2_B2
Status: Status.UNKNOWN
Output dim: 0, lower bound: -20.9607285, upper bound: 20.9608904
time: 0.72 seconds

## BFS NS instance: NS_B2_B1_A2_B2_A1_B1_A1

### Backsubstitution after applying NS history:
0: -5.7200837, 15.9770699, -5.5213213, 15.5664530, -21.2865353, 21.4983902
1: -14.2881031, 24.6138248, -13.7904596, 24.0098934, -38.2979965, 38.4042854
2: -9.9634457, 22.2609634, -9.5978346, 21.7318001, -31.6952438, 31.8587990
3: -15.3212786, 27.2044334, -14.8721561, 26.5124702, -41.8337479, 42.0765915
4: -13.8220596, 27.4177246, -13.5008192, 26.8469887, -40.6690483, 40.9185448

Time for backsubstitution: 0.91 seconds

### NS candidates at layer 1
type: A, layer: 1, pos: 1
type: A, layer: 1, pos: 28
type: B, layer: 1, pos: 28
type: B, layer: 1, pos: 45
type: A, layer: 1, pos: 45
type: B, layer: 1, pos: 39
type: A, layer: 1, pos: 15
type: A, layer: 1, pos: 6
type: B, layer: 1, pos: 38
type: A, layer: 1, pos: 38
type: B, layer: 1, pos: 6
type: A, layer: 1, pos: 8
type: A, layer: 1, pos: 9
type: B, layer: 1, pos: 9
type: B, layer: 1, pos: 48
type: A, layer: 1, pos: 48
type: B, layer: 1, pos: 37
type: B, layer: 1, pos: 8
type: A, layer: 1, pos: 24
type: B, layer: 1, pos: 24
type: B, layer: 1, pos: 20
type: A, layer: 1, pos: 37
type: A, layer: 1, pos: 20
type: B, layer: 1, pos: 26
type: A, layer: 1, pos: 26
type: A, layer: 1, pos: 31
type: B, layer: 1, pos: 31
type: A, layer: 1, pos: 0
type: B, layer: 1, pos: 0
type: B, layer: 1, pos: 41
type: A, layer: 1, pos: 41
type: B, layer: 1, pos: 5
type: A, layer: 1, pos: 5

Time for candidate selection: 0.07 seconds

### Candidate
type: A, layer: 1, pos: 1

### Candidate
type: A, layer: 1, pos: 28

### Candidate
type: B, layer: 1, pos: 28

### Candidate
type: B, layer: 1, pos: 45

### Candidate
type: A, layer: 1, pos: 45

### Candidate
type: B, layer: 1, pos: 39

## Relational analysis of NS_B2_B1_A2_B2_A1_B1_A1_B1

### Relational analysis result of NS_B2_B1_A2_B2_A1_B1_A1_B1
Status: Status.UNKNOWN
Output dim: 0, lower bound: -20.9619006, upper bound: 20.9624504
time: 0.62 seconds

## Relational analysis of NS_B2_B1_A2_B2_A1_B1_A1_B2

### Relational analysis result of NS_B2_B1_A2_B2_A1_B1_A1_B2
Status: Status.UNKNOWN
Output dim: 0, lower bound: -20.9619006, upper bound: 20.9624504
time: 0.86 seconds

## BFS NS instance: NS_B2_B1_A2_B2_A1_B1_A2

### Backsubstitution after applying NS history:
0: -6.2399273, 17.5782719, -5.4547906, 15.3952723, -21.6351986, 23.0330620
1: -15.5931826, 27.1239376, -13.6067934, 23.7583561, -39.3515358, 40.7307243
2: -10.8479500, 24.4822655, -9.4758329, 21.5024853, -32.3504333, 33.9580917
3: -16.7372913, 29.8976383, -14.6894608, 26.2318821, -42.9691734, 44.5870934
4: -15.0970030, 30.2145424, -13.3483181, 26.5601501, -41.6571503, 43.5628510

Time for backsubstitution: 0.92 seconds

### NS candidates at layer 1
type: A, layer: 1, pos: 1
type: B, layer: 1, pos: 28
type: A, layer: 1, pos: 28
type: B, layer: 1, pos: 39
type: B, layer: 1, pos: 45
type: A, layer: 1, pos: 45
type: B, layer: 1, pos: 6
type: A, layer: 1, pos: 6
type: A, layer: 1, pos: 15
type: B, layer: 1, pos: 38
type: A, layer: 1, pos: 38
type: B, layer: 1, pos: 48
type: A, layer: 1, pos: 24
type: A, layer: 1, pos: 8
type: B, layer: 1, pos: 9
type: A, layer: 1, pos: 48
type: A, layer: 1, pos: 9
type: B, layer: 1, pos: 24
type: B, layer: 1, pos: 8
type: A, layer: 1, pos: 20
type: B, layer: 1, pos: 37
type: A, layer: 1, pos: 37
type: B, layer: 1, pos: 26
type: A, layer: 1, pos: 26
type: B, layer: 1, pos: 20
type: A, layer: 1, pos: 31
type: A, layer: 1, pos: 0
type: B, layer: 1, pos: 31
type: B, layer: 1, pos: 0
type: B, layer: 1, pos: 41
type: A, layer: 1, pos: 41
type: B, layer: 1, pos: 5
type: A, layer: 1, pos: 5

Time for candidate selection: 0.07 seconds

### Candidate
type: A, layer: 1, pos: 1

### Candidate
type: B, layer: 1, pos: 28

### Candidate
type: A, layer: 1, pos: 28

### Candidate
type: B, layer: 1, pos: 39

## Relational analysis of NS_B2_B1_A2_B2_A1_B1_A2_B1

### Relational analysis result of NS_B2_B1_A2_B2_A1_B1_A2_B1
Status: Status.UNKNOWN
Output dim: 0, lower bound: -20.9619006, upper bound: 20.9624504
time: 0.60 seconds

## Relational analysis of NS_B2_B1_A2_B2_A1_B1_A2_B2

### Relational analysis result of NS_B2_B1_A2_B2_A1_B1_A2_B2
Status: Status.UNKNOWN
Output dim: 0, lower bound: -20.9619006, upper bound: 20.9624504
time: 0.72 seconds

## BFS NS instance: NS_B2_B1_A2_B2_A1_B2_A1

### Backsubstitution after applying NS history:
0: -5.3261676, 14.8327303, -5.5040417, 15.4941473, -20.8203144, 20.3367729
1: -13.2366467, 22.8807583, -13.6172237, 24.0125790, -37.2492104, 36.4979782
2: -9.2212620, 20.6801872, -9.5294647, 21.7030258, -30.9242878, 30.2096519
3: -14.2270927, 25.3174648, -14.7766981, 26.4654503, -40.6925430, 40.0941544
4: -12.8346519, 25.4475536, -13.5740061, 26.7465839, -39.5812378, 39.0215569

Time for backsubstitution: 0.93 seconds

### NS candidates at layer 1
type: A, layer: 1, pos: 1
type: B, layer: 1, pos: 45
type: A, layer: 1, pos: 45
type: B, layer: 1, pos: 39
type: A, layer: 1, pos: 20
type: A, layer: 1, pos: 15
type: A, layer: 1, pos: 28
type: B, layer: 1, pos: 48
type: A, layer: 1, pos: 38
type: B, layer: 1, pos: 38
type: A, layer: 1, pos: 6
type: B, layer: 1, pos: 9
type: A, layer: 1, pos: 9
type: B, layer: 1, pos: 8
type: A, layer: 1, pos: 39
type: B, layer: 1, pos: 6
type: B, layer: 1, pos: 20
type: A, layer: 1, pos: 37
type: A, layer: 1, pos: 48
type: B, layer: 1, pos: 37
type: A, layer: 1, pos: 24
type: B, layer: 1, pos: 28
type: B, layer: 1, pos: 31
type: A, layer: 1, pos: 0
type: A, layer: 1, pos: 31
type: A, layer: 1, pos: 5
type: B, layer: 1, pos: 5
type: B, layer: 1, pos: 26
type: B, layer: 1, pos: 24
type: B, layer: 1, pos: 41
type: A, layer: 1, pos: 26
type: A, layer: 1, pos: 41
type: B, layer: 1, pos: 0

Time for candidate selection: 0.07 seconds

### Candidate
type: A, layer: 1, pos: 1

### Candidate
type: B, layer: 1, pos: 45

### Candidate
type: A, layer: 1, pos: 45

### Candidate
type: B, layer: 1, pos: 39

### Candidate
type: A, layer: 1, pos: 20

## Relational analysis of NS_B2_B1_A2_B2_A1_B2_A1_A1
Optimization infeasible because this subproblem isn't reachable.

### Candidate
type: A, layer: 1, pos: 15

### Candidate
type: A, layer: 1, pos: 28

### Candidate
type: B, layer: 1, pos: 48

### Candidate
type: A, layer: 1, pos: 38

### Candidate
type: B, layer: 1, pos: 38

### Candidate
type: A, layer: 1, pos: 6

## Relational analysis of NS_B2_B1_A2_B2_A1_B2_A1_A1
Optimization infeasible because this subproblem isn't reachable.

### Candidate
type: B, layer: 1, pos: 9

### Candidate
type: A, layer: 1, pos: 9

### Candidate
type: B, layer: 1, pos: 8

## Relational analysis of NS_B2_B1_A2_B2_A1_B2_A1_B1
Optimization infeasible because this subproblem isn't reachable.

### Candidate
type: A, layer: 1, pos: 39

### Candidate
type: B, layer: 1, pos: 6

### Candidate
type: B, layer: 1, pos: 20

### Candidate
type: A, layer: 1, pos: 37

## Relational analysis of NS_B2_B1_A2_B2_A1_B2_A1_A1
Optimization infeasible because this subproblem isn't reachable.

### Candidate
type: A, layer: 1, pos: 48

### Candidate
type: B, layer: 1, pos: 37

### Candidate
type: A, layer: 1, pos: 24

## Relational analysis of NS_B2_B1_A2_B2_A1_B2_A1_A1
Optimization infeasible because this subproblem isn't reachable.

### Candidate
type: B, layer: 1, pos: 28

### Candidate
type: B, layer: 1, pos: 31

## Relational analysis of NS_B2_B1_A2_B2_A1_B2_A1_B1
Optimization infeasible because this subproblem isn't reachable.

### Candidate
type: A, layer: 1, pos: 0

## Relational analysis of NS_B2_B1_A2_B2_A1_B2_A1_A1

### Relational analysis result of NS_B2_B1_A2_B2_A1_B2_A1_A1
Status: Status.UNKNOWN
Output dim: 0, lower bound: -20.9529447, upper bound: 20.9563306
time: 0.62 seconds

## Relational analysis of NS_B2_B1_A2_B2_A1_B2_A1_A2

### Relational analysis result of NS_B2_B1_A2_B2_A1_B2_A1_A2
Status: Status.UNKNOWN
Output dim: 0, lower bound: -20.9513929, upper bound: 20.9539752
time: 0.66 seconds

## BFS NS instance: NS_B2_B1_A2_B2_A1_B2_A2

### Backsubstitution after applying NS history:
0: -5.9545293, 16.5955944, -5.8139791, 16.2948456, -22.2493744, 22.4095726
1: -14.8926563, 25.5386639, -14.4053001, 25.1892090, -40.0818596, 39.9439621
2: -10.3810644, 23.0928288, -10.0807743, 22.7908363, -33.1718941, 33.1735992
3: -15.9541216, 28.2298679, -15.6151314, 27.7924881, -43.7466087, 43.8450012
4: -14.3885117, 28.4506893, -14.3490543, 28.0902920, -42.4788055, 42.7997398

Time for backsubstitution: 0.93 seconds

### NS candidates at layer 1
type: A, layer: 1, pos: 1
type: B, layer: 1, pos: 45
type: A, layer: 1, pos: 45
type: A, layer: 1, pos: 15
type: B, layer: 1, pos: 39
type: B, layer: 1, pos: 48
type: A, layer: 1, pos: 38
type: B, layer: 1, pos: 38
type: B, layer: 1, pos: 9
type: A, layer: 1, pos: 28
type: A, layer: 1, pos: 9
type: A, layer: 1, pos: 6
type: B, layer: 1, pos: 37
type: A, layer: 1, pos: 20
type: A, layer: 1, pos: 48
type: A, layer: 1, pos: 37
type: B, layer: 1, pos: 8
type: B, layer: 1, pos: 6
type: B, layer: 1, pos: 28
type: B, layer: 1, pos: 20
type: A, layer: 1, pos: 31
type: A, layer: 1, pos: 24
type: B, layer: 1, pos: 31
type: A, layer: 1, pos: 39
type: A, layer: 1, pos: 0
type: B, layer: 1, pos: 24
type: A, layer: 1, pos: 5
type: B, layer: 1, pos: 41
type: B, layer: 1, pos: 26
type: A, layer: 1, pos: 41
type: A, layer: 1, pos: 26
type: B, layer: 1, pos: 5
type: B, layer: 1, pos: 0

Time for candidate selection: 0.07 seconds

### Candidate
type: A, layer: 1, pos: 1

### Candidate
type: B, layer: 1, pos: 45

### Candidate
type: A, layer: 1, pos: 45

### Candidate
type: A, layer: 1, pos: 15

### Candidate
type: B, layer: 1, pos: 39

### Candidate
type: B, layer: 1, pos: 48

### Candidate
type: A, layer: 1, pos: 38

### Candidate
type: B, layer: 1, pos: 38

### Candidate
type: B, layer: 1, pos: 9

### Candidate
type: A, layer: 1, pos: 28

### Candidate
type: A, layer: 1, pos: 9

### Candidate
type: A, layer: 1, pos: 6

## Relational analysis of NS_B2_B1_A2_B2_A1_B2_A2_A1
Optimization infeasible because this subproblem isn't reachable.

### Candidate
type: B, layer: 1, pos: 37

## Relational analysis of NS_B2_B1_A2_B2_A1_B2_A2_B1

### Relational analysis result of NS_B2_B1_A2_B2_A1_B2_A2_B1
Status: Status.UNKNOWN
Output dim: 0, lower bound: -20.9579247, upper bound: 20.9604336
time: 0.88 seconds

## Relational analysis of NS_B2_B1_A2_B2_A1_B2_A2_B2

### Relational analysis result of NS_B2_B1_A2_B2_A1_B2_A2_B2
Status: Status.UNKNOWN
Output dim: 0, lower bound: -20.9542086, upper bound: 20.9570735
time: 0.75 seconds

## BFS NS instance: NS_B2_B1_A2_B2_A2_A2_B1

### Backsubstitution after applying NS history:
0: -5.9942079, 16.6205750, -5.6530471, 15.8669186, -21.8611259, 22.2736187
1: -14.8060150, 25.6867104, -14.0018511, 24.5456734, -39.3516884, 39.6885490
2: -10.4000788, 23.2019463, -9.7986202, 22.2141628, -32.6142426, 33.0005646
3: -16.0045681, 28.3820171, -15.1862488, 27.0824966, -43.0870628, 43.5682640
4: -14.6585836, 28.4995098, -13.9546032, 27.3776703, -42.0362549, 42.4541130

Time for backsubstitution: 0.94 seconds

### NS candidates at layer 1
type: A, layer: 1, pos: 1
type: B, layer: 1, pos: 45
type: A, layer: 1, pos: 45
type: B, layer: 1, pos: 47
type: B, layer: 1, pos: 38
type: A, layer: 1, pos: 38
type: A, layer: 1, pos: 9
type: A, layer: 1, pos: 6
type: B, layer: 1, pos: 48
type: A, layer: 1, pos: 39
type: A, layer: 1, pos: 48
type: B, layer: 1, pos: 9
type: A, layer: 1, pos: 8
type: B, layer: 1, pos: 20
type: A, layer: 1, pos: 28
type: B, layer: 1, pos: 37
type: A, layer: 1, pos: 37
type: B, layer: 1, pos: 28
type: B, layer: 1, pos: 6
type: B, layer: 1, pos: 8
type: A, layer: 1, pos: 31
type: A, layer: 1, pos: 20
type: B, layer: 1, pos: 31
type: B, layer: 1, pos: 24
type: A, layer: 1, pos: 24
type: B, layer: 1, pos: 0
type: A, layer: 1, pos: 0
type: B, layer: 1, pos: 5
type: A, layer: 1, pos: 5
type: B, layer: 1, pos: 41
type: A, layer: 1, pos: 41
type: B, layer: 1, pos: 26
type: A, layer: 1, pos: 26

Time for candidate selection: 0.07 seconds

### Candidate
type: A, layer: 1, pos: 1

### Candidate
type: B, layer: 1, pos: 45

### Candidate
type: A, layer: 1, pos: 45

### Candidate
type: B, layer: 1, pos: 47

### Candidate
type: B, layer: 1, pos: 38

## Relational analysis of NS_B2_B1_A2_B2_A2_A2_B1_B1

### Relational analysis result of NS_B2_B1_A2_B2_A2_A2_B1_B1
Status: Status.VERIFIED
Output dim: 0, lower bound: -20.7041063, upper bound: 20.6868226
time: 0.63 seconds

## Relational analysis of NS_B2_B1_A2_B2_A2_A2_B1_B2

### Relational analysis result of NS_B2_B1_A2_B2_A2_A2_B1_B2
Status: Status.VERIFIED
Output dim: 0, lower bound: -20.6625303, upper bound: 20.6605575
time: 0.58 seconds

## BFS NS instance: NS_B2_B1_A2_B2_A2_A2_B2

### Backsubstitution after applying NS history:
0: -5.9174595, 16.4248638, -6.7729082, 18.9741554, -24.8916149, 23.1977711
1: -14.5977459, 25.4009228, -17.0074577, 29.1194172, -43.7171631, 42.4083786
2: -10.2598610, 22.9395370, -11.7895298, 26.3700848, -36.6299400, 34.7290649
3: -15.7931938, 28.0613556, -18.2154408, 32.1135483, -47.9067421, 46.2767944
4: -14.4798212, 28.1716518, -16.4767666, 32.6066933, -47.0865135, 44.6484184

Time for backsubstitution: 0.93 seconds

### NS candidates at layer 1
type: A, layer: 1, pos: 1
type: B, layer: 1, pos: 47
type: B, layer: 1, pos: 45
type: A, layer: 1, pos: 39
type: A, layer: 1, pos: 45
type: A, layer: 1, pos: 6
type: A, layer: 1, pos: 28
type: B, layer: 1, pos: 38
type: B, layer: 1, pos: 48
type: A, layer: 1, pos: 48
type: B, layer: 1, pos: 20
type: B, layer: 1, pos: 6
type: A, layer: 1, pos: 38
type: B, layer: 1, pos: 28
type: A, layer: 1, pos: 9
type: A, layer: 1, pos: 8
type: B, layer: 1, pos: 37
type: B, layer: 1, pos: 24
type: B, layer: 1, pos: 9
type: A, layer: 1, pos: 37
type: B, layer: 1, pos: 0
type: A, layer: 1, pos: 31
type: B, layer: 1, pos: 8
type: A, layer: 1, pos: 24
type: B, layer: 1, pos: 31
type: A, layer: 1, pos: 0
type: A, layer: 1, pos: 20
type: B, layer: 1, pos: 26
type: A, layer: 1, pos: 26
type: B, layer: 1, pos: 41
type: A, layer: 1, pos: 41
type: B, layer: 1, pos: 5
type: A, layer: 1, pos: 5

Time for candidate selection: 0.08 seconds

### Candidate
type: A, layer: 1, pos: 1

### Candidate
type: B, layer: 1, pos: 47

### Candidate
type: B, layer: 1, pos: 45

### Candidate
type: A, layer: 1, pos: 39

## Relational analysis of NS_B2_B1_A2_B2_A2_A2_B2_A1

### Relational analysis result of NS_B2_B1_A2_B2_A2_A2_B2_A1
Status: Status.UNKNOWN
Output dim: 0, lower bound: -20.9503003, upper bound: 20.9499824
time: 0.79 seconds

## Relational analysis of NS_B2_B1_A2_B2_A2_A2_B2_A2

### Relational analysis result of NS_B2_B1_A2_B2_A2_A2_B2_A2
Status: Status.UNKNOWN
Output dim: 0, lower bound: -20.9503003, upper bound: 20.9502923
time: 0.75 seconds

## BFS NS instance: NS_B2_B2_A1_B1_A1_B1_B1

### Backsubstitution after applying NS history:
0: -5.6512485, 15.8766308, -5.0356288, 14.0746908, -19.7259388, 20.9122581
1: -14.2241335, 24.3052940, -12.4586964, 21.7433186, -35.9674454, 36.7639923
2: -9.8599586, 22.0852985, -8.6867218, 19.6489391, -29.5088978, 30.7720203
3: -15.2573195, 26.8963223, -13.4280396, 24.0546970, -39.3120117, 40.3243637
4: -13.7445583, 27.3269234, -12.1308689, 24.1640739, -37.9086304, 39.4577942

Time for backsubstitution: 0.93 seconds

### NS candidates at layer 1
type: A, layer: 1, pos: 12
type: B, layer: 1, pos: 28
type: A, layer: 1, pos: 45
type: A, layer: 1, pos: 28
type: B, layer: 1, pos: 6
type: B, layer: 1, pos: 39
type: B, layer: 1, pos: 45
type: A, layer: 1, pos: 39
type: B, layer: 1, pos: 24
type: A, layer: 1, pos: 24
type: A, layer: 1, pos: 6
type: A, layer: 1, pos: 15
type: A, layer: 1, pos: 48
type: B, layer: 1, pos: 20
type: A, layer: 1, pos: 9
type: A, layer: 1, pos: 20
type: B, layer: 1, pos: 15
type: A, layer: 1, pos: 38
type: A, layer: 1, pos: 37
type: A, layer: 1, pos: 8
type: B, layer: 1, pos: 37
type: A, layer: 1, pos: 26
type: B, layer: 1, pos: 26
type: B, layer: 1, pos: 38
type: A, layer: 1, pos: 0
type: B, layer: 1, pos: 0
type: B, layer: 1, pos: 48
type: B, layer: 1, pos: 41
type: A, layer: 1, pos: 41
type: A, layer: 1, pos: 31
type: A, layer: 1, pos: 5
type: B, layer: 1, pos: 31
type: B, layer: 1, pos: 5

Time for candidate selection: 0.08 seconds

### Candidate
type: A, layer: 1, pos: 12

### Candidate
type: B, layer: 1, pos: 28

### Candidate
type: A, layer: 1, pos: 45

### Candidate
type: A, layer: 1, pos: 28

### Candidate
type: B, layer: 1, pos: 6

### Candidate
type: B, layer: 1, pos: 39

### Candidate
type: B, layer: 1, pos: 45

### Candidate
type: A, layer: 1, pos: 39

### Candidate
type: B, layer: 1, pos: 24

### Candidate
type: A, layer: 1, pos: 24

### Candidate
type: A, layer: 1, pos: 6

### Candidate
type: A, layer: 1, pos: 15

### Candidate
type: A, layer: 1, pos: 48

### Candidate
type: B, layer: 1, pos: 20

## Relational analysis of NS_B2_B2_A1_B1_A1_B1_B1_B1
Optimization infeasible because this subproblem isn't reachable.

### Candidate
type: A, layer: 1, pos: 9

## Relational analysis of NS_B2_B2_A1_B1_A1_B1_B1_A1

### Relational analysis result of NS_B2_B2_A1_B1_A1_B1_B1_A1
Status: Status.UNKNOWN
Output dim: 0, lower bound: -20.9300851, upper bound: 20.9287486
time: 0.90 seconds

## Relational analysis of NS_B2_B2_A1_B1_A1_B1_B1_A2

### Relational analysis result of NS_B2_B2_A1_B1_A1_B1_B1_A2
Status: Status.UNKNOWN
Output dim: 0, lower bound: -20.9300851, upper bound: 20.9500642
time: 0.65 seconds

## BFS NS instance: NS_B2_B2_A1_B1_A1_B1_B2

### Backsubstitution after applying NS history:
0: -5.6015830, 15.7430573, -4.9803114, 13.8781071, -19.4796906, 20.7233677
1: -14.1006050, 24.1030998, -12.3479433, 21.3440819, -35.4446793, 36.4510345
2: -9.7733765, 21.9032230, -8.6038857, 19.3577366, -29.1311131, 30.5071049
3: -15.1259575, 26.6727180, -13.2907629, 23.6612930, -38.7872505, 39.9634743
4: -13.6228781, 27.1033058, -11.9675913, 23.8155308, -37.4384079, 39.0708961

Time for backsubstitution: 0.94 seconds

### NS candidates at layer 1
type: A, layer: 1, pos: 12
type: B, layer: 1, pos: 28
type: A, layer: 1, pos: 45
type: A, layer: 1, pos: 28
type: B, layer: 1, pos: 6
type: B, layer: 1, pos: 39
type: B, layer: 1, pos: 45
type: A, layer: 1, pos: 39
type: B, layer: 1, pos: 24
type: A, layer: 1, pos: 24
type: A, layer: 1, pos: 6
type: A, layer: 1, pos: 15
type: B, layer: 1, pos: 20
type: A, layer: 1, pos: 48
type: A, layer: 1, pos: 20
type: B, layer: 1, pos: 15
type: A, layer: 1, pos: 38
type: A, layer: 1, pos: 37
type: A, layer: 1, pos: 8
type: B, layer: 1, pos: 37
type: A, layer: 1, pos: 26
type: A, layer: 1, pos: 9
type: B, layer: 1, pos: 26
type: B, layer: 1, pos: 38
type: A, layer: 1, pos: 0
type: B, layer: 1, pos: 0
type: A, layer: 1, pos: 41
type: B, layer: 1, pos: 41
type: B, layer: 1, pos: 48
type: A, layer: 1, pos: 31
type: B, layer: 1, pos: 31
type: B, layer: 1, pos: 5
type: A, layer: 1, pos: 5

Time for candidate selection: 0.08 seconds

### Candidate
type: A, layer: 1, pos: 12

### Candidate
type: B, layer: 1, pos: 28

### Candidate
type: A, layer: 1, pos: 45

### Candidate
type: A, layer: 1, pos: 28

### Candidate
type: B, layer: 1, pos: 6

### Candidate
type: B, layer: 1, pos: 39

### Candidate
type: B, layer: 1, pos: 45

### Candidate
type: A, layer: 1, pos: 39

### Candidate
type: B, layer: 1, pos: 24

### Candidate
type: A, layer: 1, pos: 24

### Candidate
type: A, layer: 1, pos: 6

### Candidate
type: A, layer: 1, pos: 15

### Candidate
type: B, layer: 1, pos: 20

## Relational analysis of NS_B2_B2_A1_B1_A1_B1_B2_B1
Optimization infeasible because this subproblem isn't reachable.

### Candidate
type: A, layer: 1, pos: 48

### Candidate
type: A, layer: 1, pos: 20

### Candidate
type: B, layer: 1, pos: 15

### Candidate
type: A, layer: 1, pos: 38

## Relational analysis of NS_B2_B2_A1_B1_A1_B1_B2_A1
Optimization infeasible because this subproblem isn't reachable.

### Candidate
type: A, layer: 1, pos: 37

## Relational analysis of NS_B2_B2_A1_B1_A1_B1_B2_A1
Optimization infeasible because this subproblem isn't reachable.

### Candidate
type: A, layer: 1, pos: 8

## Relational analysis of NS_B2_B2_A1_B1_A1_B1_B2_A1

### Relational analysis result of NS_B2_B2_A1_B1_A1_B1_B2_A1
Status: Status.UNKNOWN
Output dim: 0, lower bound: -20.9615888, upper bound: 20.9616959
time: 0.63 seconds

## Relational analysis of NS_B2_B2_A1_B1_A1_B1_B2_A2

### Relational analysis result of NS_B2_B2_A1_B1_A1_B1_B2_A2
Status: Status.UNKNOWN
Output dim: 0, lower bound: -20.9615888, upper bound: 20.9625913
time: 0.69 seconds

## BFS NS instance: NS_B2_B2_A1_B1_A1_B2_B1

### Backsubstitution after applying NS history:
0: -6.0244703, 16.8405571, -5.6549067, 15.8308163, -21.8552856, 22.4954643
1: -15.1832008, 25.7038307, -14.0980024, 24.4064045, -39.5896072, 39.8018341
2: -10.5292540, 23.3857155, -9.8330898, 22.0536385, -32.5828896, 33.2188034
3: -16.2657433, 28.4767113, -15.1370831, 26.9644737, -43.2302132, 43.6137924
4: -14.6604919, 28.9407387, -13.6658039, 27.1664886, -41.8269806, 42.6065292

Time for backsubstitution: 0.94 seconds

### NS candidates at layer 1
type: A, layer: 1, pos: 12
type: A, layer: 1, pos: 28
type: A, layer: 1, pos: 45
type: B, layer: 1, pos: 28
type: B, layer: 1, pos: 45
type: A, layer: 1, pos: 15
type: B, layer: 1, pos: 6
type: B, layer: 1, pos: 39
type: A, layer: 1, pos: 24
type: A, layer: 1, pos: 6
type: B, layer: 1, pos: 15
type: A, layer: 1, pos: 48
type: B, layer: 1, pos: 24
type: A, layer: 1, pos: 39
type: A, layer: 1, pos: 38
type: A, layer: 1, pos: 37
type: A, layer: 1, pos: 9
type: A, layer: 1, pos: 20
type: B, layer: 1, pos: 37
type: B, layer: 1, pos: 26
type: A, layer: 1, pos: 26
type: B, layer: 1, pos: 38
type: A, layer: 1, pos: 8
type: B, layer: 1, pos: 48
type: A, layer: 1, pos: 0
type: B, layer: 1, pos: 20
type: B, layer: 1, pos: 31
type: A, layer: 1, pos: 41
type: B, layer: 1, pos: 41
type: B, layer: 1, pos: 5
type: A, layer: 1, pos: 5
type: A, layer: 1, pos: 31
type: B, layer: 1, pos: 0

Time for candidate selection: 0.08 seconds

### Candidate
type: A, layer: 1, pos: 12

### Candidate
type: A, layer: 1, pos: 28

### Candidate
type: A, layer: 1, pos: 45

### Candidate
type: B, layer: 1, pos: 28

### Candidate
type: B, layer: 1, pos: 45

### Candidate
type: A, layer: 1, pos: 15

### Candidate
type: B, layer: 1, pos: 6

### Candidate
type: B, layer: 1, pos: 39

### Candidate
type: A, layer: 1, pos: 24

### Candidate
type: A, layer: 1, pos: 6

### Candidate
type: B, layer: 1, pos: 15

### Candidate
type: A, layer: 1, pos: 48

### Candidate
type: B, layer: 1, pos: 24

### Candidate
type: A, layer: 1, pos: 39

### Candidate
type: A, layer: 1, pos: 38

## Relational analysis of NS_B2_B2_A1_B1_A1_B2_B1_A1
Optimization infeasible because this subproblem isn't reachable.

### Candidate
type: A, layer: 1, pos: 37

## Relational analysis of NS_B2_B2_A1_B1_A1_B2_B1_A1
Optimization infeasible because this subproblem isn't reachable.

### Candidate
type: A, layer: 1, pos: 9

## Relational analysis of NS_B2_B2_A1_B1_A1_B2_B1_A1

### Relational analysis result of NS_B2_B2_A1_B1_A1_B2_B1_A1
Status: Status.UNKNOWN
Output dim: 0, lower bound: -20.9298018, upper bound: 20.9287663
time: 0.71 seconds

## Relational analysis of NS_B2_B2_A1_B1_A1_B2_B1_A2

### Relational analysis result of NS_B2_B2_A1_B1_A1_B2_B1_A2
Status: Status.UNKNOWN
Output dim: 0, lower bound: -20.9298018, upper bound: 20.9503509
time: 1.10 seconds

## BFS NS instance: NS_B2_B2_A1_B1_A1_B2_B2

### Backsubstitution after applying NS history:
0: -5.9729180, 16.7020836, -5.6332521, 15.7332449, -21.7061539, 22.3353348
1: -15.0557423, 25.4947567, -14.0975790, 24.1404915, -39.1962357, 39.5923195
2: -10.4393234, 23.1972771, -9.8184195, 21.8792229, -32.3185463, 33.0156975
3: -16.1292458, 28.2445431, -15.1023226, 26.7292881, -42.8585358, 43.3468513
4: -14.5333204, 28.7094154, -13.5720100, 26.9947109, -41.5280266, 42.2814255

Time for backsubstitution: 0.95 seconds

### NS candidates at layer 1
type: A, layer: 1, pos: 12
type: A, layer: 1, pos: 28
type: B, layer: 1, pos: 28
type: B, layer: 1, pos: 6
type: A, layer: 1, pos: 45
type: A, layer: 1, pos: 6
type: B, layer: 1, pos: 39
type: A, layer: 1, pos: 24
type: B, layer: 1, pos: 45
type: A, layer: 1, pos: 15
type: B, layer: 1, pos: 24
type: A, layer: 1, pos: 48
type: A, layer: 1, pos: 39
type: B, layer: 1, pos: 15
type: A, layer: 1, pos: 38
type: A, layer: 1, pos: 37
type: A, layer: 1, pos: 20
type: B, layer: 1, pos: 37
type: A, layer: 1, pos: 26
type: B, layer: 1, pos: 26
type: B, layer: 1, pos: 38
type: A, layer: 1, pos: 0
type: A, layer: 1, pos: 9
type: A, layer: 1, pos: 8
type: B, layer: 1, pos: 48
type: B, layer: 1, pos: 20
type: B, layer: 1, pos: 0
type: B, layer: 1, pos: 31
type: B, layer: 1, pos: 5
type: A, layer: 1, pos: 41
type: B, layer: 1, pos: 41
type: A, layer: 1, pos: 5
type: A, layer: 1, pos: 31

Time for candidate selection: 0.08 seconds

### Candidate
type: A, layer: 1, pos: 12

### Candidate
type: A, layer: 1, pos: 28

### Candidate
type: B, layer: 1, pos: 28

### Candidate
type: B, layer: 1, pos: 6

### Candidate
type: A, layer: 1, pos: 45

### Candidate
type: A, layer: 1, pos: 6

### Candidate
type: B, layer: 1, pos: 39

### Candidate
type: A, layer: 1, pos: 24

### Candidate
type: B, layer: 1, pos: 45

### Candidate
type: A, layer: 1, pos: 15

### Candidate
type: B, layer: 1, pos: 24

### Candidate
type: A, layer: 1, pos: 48

### Candidate
type: A, layer: 1, pos: 39

### Candidate
type: B, layer: 1, pos: 15

### Candidate
type: A, layer: 1, pos: 38

## Relational analysis of NS_B2_B2_A1_B1_A1_B2_B2_A1
Optimization infeasible because this subproblem isn't reachable.

### Candidate
type: A, layer: 1, pos: 37

## Relational analysis of NS_B2_B2_A1_B1_A1_B2_B2_A1
Optimization infeasible because this subproblem isn't reachable.

### Candidate
type: A, layer: 1, pos: 20

## Relational analysis of NS_B2_B2_A1_B1_A1_B2_B2_A1

### Relational analysis result of NS_B2_B2_A1_B1_A1_B2_B2_A1
Status: Status.UNKNOWN
Output dim: 0, lower bound: -20.9548427, upper bound: 20.9551555
time: 1.33 seconds

## Relational analysis of NS_B2_B2_A1_B1_A1_B2_B2_A2

### Relational analysis result of NS_B2_B2_A1_B1_A1_B2_B2_A2
Status: Status.UNKNOWN
Output dim: 0, lower bound: -20.9640653, upper bound: 20.9646005
time: 0.71 seconds

## BFS NS instance: NS_B2_B2_A1_B1_A2_A1_B1

### Backsubstitution after applying NS history:
0: -5.1171942, 14.4601212, -4.7143350, 13.2595787, -18.3767719, 19.1744518
1: -12.6834574, 22.4130745, -11.5646448, 20.6092720, -33.2927246, 33.9777184
2: -8.8560047, 20.3152809, -8.0930729, 18.5866394, -27.4426441, 28.4083538
3: -13.7529573, 24.7136421, -12.5332088, 22.7562466, -36.5092010, 37.2468491
4: -12.5835161, 25.0596924, -11.4073057, 22.8247948, -35.4083099, 36.4669952

Time for backsubstitution: 0.94 seconds

### NS candidates at layer 1
type: A, layer: 1, pos: 12
type: A, layer: 1, pos: 45
type: B, layer: 1, pos: 45
type: A, layer: 1, pos: 15
type: B, layer: 1, pos: 15
type: B, layer: 1, pos: 20
type: A, layer: 1, pos: 48
type: B, layer: 1, pos: 24
type: B, layer: 1, pos: 28
type: A, layer: 1, pos: 39
type: B, layer: 1, pos: 6
type: A, layer: 1, pos: 24
type: B, layer: 1, pos: 9
type: A, layer: 1, pos: 9
type: A, layer: 1, pos: 20
type: B, layer: 1, pos: 39
type: A, layer: 1, pos: 38
type: B, layer: 1, pos: 37
type: A, layer: 1, pos: 37
type: A, layer: 1, pos: 8
type: B, layer: 1, pos: 0
type: A, layer: 1, pos: 28
type: A, layer: 1, pos: 0
type: B, layer: 1, pos: 38
type: B, layer: 1, pos: 48
type: A, layer: 1, pos: 31
type: A, layer: 1, pos: 26
type: B, layer: 1, pos: 26
type: B, layer: 1, pos: 31
type: B, layer: 1, pos: 5
type: B, layer: 1, pos: 41
type: A, layer: 1, pos: 41
type: A, layer: 1, pos: 5

Time for candidate selection: 0.08 seconds

### Candidate
type: A, layer: 1, pos: 12

### Candidate
type: A, layer: 1, pos: 45

### Candidate
type: B, layer: 1, pos: 45

### Candidate
type: A, layer: 1, pos: 15

### Candidate
type: B, layer: 1, pos: 15

### Candidate
type: B, layer: 1, pos: 20

### Candidate
type: A, layer: 1, pos: 48

### Candidate
type: B, layer: 1, pos: 24

## Relational analysis of NS_B2_B2_A1_B1_A2_A1_B1_B1
Optimization infeasible because this subproblem isn't reachable.

### Candidate
type: B, layer: 1, pos: 28

### Candidate
type: A, layer: 1, pos: 39

## Relational analysis of NS_B2_B2_A1_B1_A2_A1_B1_A1
Optimization infeasible because this subproblem isn't reachable.

### Candidate
type: B, layer: 1, pos: 6

## Relational analysis of NS_B2_B2_A1_B1_A2_A1_B1_B1
Optimization infeasible because this subproblem isn't reachable.

### Candidate
type: A, layer: 1, pos: 24

### Candidate
type: B, layer: 1, pos: 9

## Relational analysis of NS_B2_B2_A1_B1_A2_A1_B1_B1

### Relational analysis result of NS_B2_B2_A1_B1_A2_A1_B1_B1
Status: Status.UNKNOWN
Output dim: 0, lower bound: -20.9081548, upper bound: 20.9023583
time: 0.57 seconds

## Relational analysis of NS_B2_B2_A1_B1_A2_A1_B1_B2

### Relational analysis result of NS_B2_B2_A1_B1_A2_A1_B1_B2
Status: Status.UNKNOWN
Output dim: 0, lower bound: -20.9381136, upper bound: 20.9253107
time: 0.72 seconds

## BFS NS instance: NS_B2_B2_A1_B1_A2_A1_B2

### Backsubstitution after applying NS history:
0: -5.4564943, 15.3382692, -5.3488650, 15.0546875, -20.5111790, 20.6871319
1: -13.5537558, 23.7017975, -13.2550144, 23.3103504, -36.8641014, 36.9568062
2: -9.4646673, 21.5025272, -9.2675076, 21.0464725, -30.5111389, 30.7700310
3: -14.6691179, 26.1667538, -14.2884932, 25.7130928, -40.3822098, 40.4552460
4: -13.4205570, 26.5279503, -12.9688034, 25.8948174, -39.3153725, 39.4967537

Time for backsubstitution: 0.95 seconds

### NS candidates at layer 1
type: A, layer: 1, pos: 12
type: A, layer: 1, pos: 45
type: A, layer: 1, pos: 15
type: B, layer: 1, pos: 45
type: B, layer: 1, pos: 15
type: A, layer: 1, pos: 20
type: A, layer: 1, pos: 48
type: B, layer: 1, pos: 24
type: B, layer: 1, pos: 6
type: B, layer: 1, pos: 28
type: B, layer: 1, pos: 9
type: A, layer: 1, pos: 9
type: B, layer: 1, pos: 20
type: A, layer: 1, pos: 8
type: A, layer: 1, pos: 24
type: A, layer: 1, pos: 38
type: A, layer: 1, pos: 39
type: A, layer: 1, pos: 37
type: B, layer: 1, pos: 37
type: A, layer: 1, pos: 28
type: B, layer: 1, pos: 39
type: B, layer: 1, pos: 38
type: A, layer: 1, pos: 0
type: B, layer: 1, pos: 0
type: B, layer: 1, pos: 48
type: A, layer: 1, pos: 26
type: A, layer: 1, pos: 31
type: B, layer: 1, pos: 26
type: B, layer: 1, pos: 31
type: B, layer: 1, pos: 5
type: A, layer: 1, pos: 41
type: B, layer: 1, pos: 41
type: A, layer: 1, pos: 5

Time for candidate selection: 0.08 seconds

### Candidate
type: A, layer: 1, pos: 12

### Candidate
type: A, layer: 1, pos: 45

### Candidate
type: A, layer: 1, pos: 15

### Candidate
type: B, layer: 1, pos: 45

### Candidate
type: B, layer: 1, pos: 15

### Candidate
type: A, layer: 1, pos: 20

### Candidate
type: A, layer: 1, pos: 48

### Candidate
type: B, layer: 1, pos: 24

## Relational analysis of NS_B2_B2_A1_B1_A2_A1_B2_B1
Optimization infeasible because this subproblem isn't reachable.

### Candidate
type: B, layer: 1, pos: 6

## Relational analysis of NS_B2_B2_A1_B1_A2_A1_B2_B1
Optimization infeasible because this subproblem isn't reachable.

### Candidate
type: B, layer: 1, pos: 28

### Candidate
type: B, layer: 1, pos: 9

## Relational analysis of NS_B2_B2_A1_B1_A2_A1_B2_B1

### Relational analysis result of NS_B2_B2_A1_B1_A2_A1_B2_B1
Status: Status.UNKNOWN
Output dim: 0, lower bound: -20.9075448, upper bound: 20.9023529
time: 1.22 seconds

## Relational analysis of NS_B2_B2_A1_B1_A2_A1_B2_B2

### Relational analysis result of NS_B2_B2_A1_B1_A2_A1_B2_B2
Status: Status.UNKNOWN
Output dim: 0, lower bound: -20.9387501, upper bound: 20.9262643
time: 0.80 seconds

## BFS NS instance: NS_B2_B2_A1_B1_A2_A2_B1

### Backsubstitution after applying NS history:
0: -5.7758389, 16.1369019, -5.1090603, 14.2706547, -20.0464916, 21.2459621
1: -14.3686581, 24.8032169, -12.6505194, 22.0396614, -36.4083138, 37.4537277
2: -10.0362244, 22.5169163, -8.8227625, 19.9193935, -29.9556179, 31.3396797
3: -15.5286503, 27.4178371, -13.6273718, 24.3816147, -39.9102631, 41.0452003
4: -14.1942377, 27.7897034, -12.3172474, 24.4959354, -38.6901741, 40.1069489

Time for backsubstitution: 0.95 seconds

### NS candidates at layer 1
type: A, layer: 1, pos: 12
type: A, layer: 1, pos: 45
type: B, layer: 1, pos: 45
type: B, layer: 1, pos: 28
type: A, layer: 1, pos: 39
type: A, layer: 1, pos: 28
type: B, layer: 1, pos: 24
type: A, layer: 1, pos: 48
type: A, layer: 1, pos: 15
type: B, layer: 1, pos: 20
type: A, layer: 1, pos: 24
type: A, layer: 1, pos: 9
type: B, layer: 1, pos: 9
type: B, layer: 1, pos: 6
type: B, layer: 1, pos: 15
type: B, layer: 1, pos: 39
type: A, layer: 1, pos: 38
type: B, layer: 1, pos: 37
type: A, layer: 1, pos: 8
type: A, layer: 1, pos: 37
type: B, layer: 1, pos: 38
type: A, layer: 1, pos: 26
type: B, layer: 1, pos: 26
type: A, layer: 1, pos: 20
type: B, layer: 1, pos: 0
type: A, layer: 1, pos: 0
type: A, layer: 1, pos: 31
type: A, layer: 1, pos: 41
type: B, layer: 1, pos: 41
type: B, layer: 1, pos: 48
type: B, layer: 1, pos: 5
type: B, layer: 1, pos: 31
type: A, layer: 1, pos: 5

Time for candidate selection: 0.08 seconds

### Candidate
type: A, layer: 1, pos: 12

### Candidate
type: A, layer: 1, pos: 45

### Candidate
type: B, layer: 1, pos: 45

### Candidate
type: B, layer: 1, pos: 28

### Candidate
type: A, layer: 1, pos: 39

## Relational analysis of NS_B2_B2_A1_B1_A2_A2_B1_A1
Optimization infeasible because this subproblem isn't reachable.

### Candidate
type: A, layer: 1, pos: 28

### Candidate
type: B, layer: 1, pos: 24

### Candidate
type: A, layer: 1, pos: 48

### Candidate
type: A, layer: 1, pos: 15

### Candidate
type: B, layer: 1, pos: 20

## Relational analysis of NS_B2_B2_A1_B1_A2_A2_B1_B1
Optimization infeasible because this subproblem isn't reachable.

### Candidate
type: A, layer: 1, pos: 24

### Candidate
type: A, layer: 1, pos: 9

## Relational analysis of NS_B2_B2_A1_B1_A2_A2_B1_A1

### Relational analysis result of NS_B2_B2_A1_B1_A2_A2_B1_A1
Status: Status.UNKNOWN
Output dim: 0, lower bound: -20.9430765, upper bound: 20.9323331
time: 0.59 seconds

## Relational analysis of NS_B2_B2_A1_B1_A2_A2_B1_A2

### Relational analysis result of NS_B2_B2_A1_B1_A2_A2_B1_A2
Status: Status.UNKNOWN
Output dim: 0, lower bound: -20.9430765, upper bound: 20.9450275
time: 0.65 seconds

## BFS NS instance: NS_B2_B2_A1_B1_A2_A2_B2

### Backsubstitution after applying NS history:
0: -6.1045394, 16.9758701, -5.7379360, 16.0502377, -22.1547756, 22.7138062
1: -15.1983242, 26.0349178, -14.3143644, 24.7327061, -39.9310188, 40.3492813
2: -10.6195745, 23.6475563, -9.9866447, 22.3560219, -32.9755974, 33.6342010
3: -16.4145107, 28.8137779, -15.3614969, 27.3282509, -43.7427597, 44.1752739
4: -15.0041790, 29.1842613, -13.8753071, 27.5357323, -42.5399094, 43.0595665

Time for backsubstitution: 0.95 seconds

### NS candidates at layer 1
type: A, layer: 1, pos: 12
type: A, layer: 1, pos: 45
type: B, layer: 1, pos: 45
type: A, layer: 1, pos: 28
type: A, layer: 1, pos: 15
type: B, layer: 1, pos: 28
type: A, layer: 1, pos: 48
type: B, layer: 1, pos: 15
type: B, layer: 1, pos: 24
type: A, layer: 1, pos: 24
type: A, layer: 1, pos: 39
type: B, layer: 1, pos: 9
type: A, layer: 1, pos: 9
type: B, layer: 1, pos: 6
type: A, layer: 1, pos: 38
type: B, layer: 1, pos: 39
type: A, layer: 1, pos: 37
type: B, layer: 1, pos: 37
type: B, layer: 1, pos: 38
type: A, layer: 1, pos: 8
type: A, layer: 1, pos: 26
type: B, layer: 1, pos: 26
type: B, layer: 1, pos: 20
type: A, layer: 1, pos: 20
type: B, layer: 1, pos: 48
type: B, layer: 1, pos: 31
type: A, layer: 1, pos: 0
type: A, layer: 1, pos: 31
type: A, layer: 1, pos: 41
type: B, layer: 1, pos: 41
type: B, layer: 1, pos: 5
type: B, layer: 1, pos: 0
type: A, layer: 1, pos: 5

Time for candidate selection: 0.08 seconds

### Candidate
type: A, layer: 1, pos: 12

### Candidate
type: A, layer: 1, pos: 45

### Candidate
type: B, layer: 1, pos: 45

### Candidate
type: A, layer: 1, pos: 28

### Candidate
type: A, layer: 1, pos: 15

### Candidate
type: B, layer: 1, pos: 28

### Candidate
type: A, layer: 1, pos: 48

### Candidate
type: B, layer: 1, pos: 15

### Candidate
type: B, layer: 1, pos: 24

### Candidate
type: A, layer: 1, pos: 24

### Candidate
type: A, layer: 1, pos: 39

## Relational analysis of NS_B2_B2_A1_B1_A2_A2_B2_A1
Optimization infeasible because this subproblem isn't reachable.

### Candidate
type: B, layer: 1, pos: 9

## Relational analysis of NS_B2_B2_A1_B1_A2_A2_B2_B1

### Relational analysis result of NS_B2_B2_A1_B1_A2_A2_B2_B1
Status: Status.UNKNOWN
Output dim: 0, lower bound: -20.9265513, upper bound: 20.9228258
time: 0.91 seconds

## Relational analysis of NS_B2_B2_A1_B1_A2_A2_B2_B2

### Relational analysis result of NS_B2_B2_A1_B1_A2_A2_B2_B2
Status: Status.UNKNOWN
Output dim: 0, lower bound: -20.9513798, upper bound: 20.9472191
time: 0.71 seconds

## BFS NS instance: NS_B2_B2_A1_B2_A1_B1_B1

### Backsubstitution after applying NS history:
0: -5.6512485, 15.8766308, -5.1427388, 14.3199282, -19.9711761, 21.0193653
1: -14.2241335, 24.3052940, -12.5697231, 22.2789974, -36.5031242, 36.8750153
2: -9.8599586, 22.0852985, -8.8216429, 20.0713501, -29.9313087, 30.9069405
3: -15.2573195, 26.8963223, -13.6617050, 24.5833302, -39.8406448, 40.5580292
4: -13.7445583, 27.3269234, -12.5392199, 24.6038437, -38.3484039, 39.8661385

Time for backsubstitution: 0.95 seconds

### NS candidates at layer 1
type: A, layer: 1, pos: 12
type: A, layer: 1, pos: 45
type: B, layer: 1, pos: 39
type: A, layer: 1, pos: 15
type: B, layer: 1, pos: 28
type: A, layer: 1, pos: 28
type: B, layer: 1, pos: 45
type: B, layer: 1, pos: 15
type: B, layer: 1, pos: 6
type: A, layer: 1, pos: 24
type: A, layer: 1, pos: 20
type: B, layer: 1, pos: 24
type: A, layer: 1, pos: 37
type: A, layer: 1, pos: 9
type: A, layer: 1, pos: 38
type: B, layer: 1, pos: 37
type: A, layer: 1, pos: 6
type: B, layer: 1, pos: 20
type: A, layer: 1, pos: 48
type: B, layer: 1, pos: 38
type: A, layer: 1, pos: 8
type: A, layer: 1, pos: 39
type: B, layer: 1, pos: 26
type: A, layer: 1, pos: 26
type: B, layer: 1, pos: 48
type: A, layer: 1, pos: 0
type: B, layer: 1, pos: 0
type: B, layer: 1, pos: 41
type: A, layer: 1, pos: 5
type: A, layer: 1, pos: 41
type: A, layer: 1, pos: 31
type: B, layer: 1, pos: 31
type: B, layer: 1, pos: 5

Time for candidate selection: 0.08 seconds

### Candidate
type: A, layer: 1, pos: 12

### Candidate
type: A, layer: 1, pos: 45

### Candidate
type: B, layer: 1, pos: 39

### Candidate
type: A, layer: 1, pos: 15

### Candidate
type: B, layer: 1, pos: 28

### Candidate
type: A, layer: 1, pos: 28

### Candidate
type: B, layer: 1, pos: 45

### Candidate
type: B, layer: 1, pos: 15

### Candidate
type: B, layer: 1, pos: 6

### Candidate
type: A, layer: 1, pos: 24

### Candidate
type: A, layer: 1, pos: 20

## Relational analysis of NS_B2_B2_A1_B2_A1_B1_B1_A1
Optimization infeasible because this subproblem isn't reachable.

### Candidate
type: B, layer: 1, pos: 24

### Candidate
type: A, layer: 1, pos: 37

## Relational analysis of NS_B2_B2_A1_B2_A1_B1_B1_A1
Optimization infeasible because this subproblem isn't reachable.

### Candidate
type: A, layer: 1, pos: 9

## Relational analysis of NS_B2_B2_A1_B2_A1_B1_B1_A1

### Relational analysis result of NS_B2_B2_A1_B2_A1_B1_B1_A1
Status: Status.UNKNOWN
Output dim: 0, lower bound: -20.9292851, upper bound: 20.9261202
time: 0.61 seconds

## Relational analysis of NS_B2_B2_A1_B2_A1_B1_B1_A2

### Relational analysis result of NS_B2_B2_A1_B2_A1_B1_B1_A2
Status: Status.UNKNOWN
Output dim: 0, lower bound: -20.9292851, upper bound: 20.9503422
time: 0.71 seconds

## BFS NS instance: NS_B2_B2_A1_B2_A1_B1_B2

### Backsubstitution after applying NS history:
0: -5.6015830, 15.7430573, -4.9961505, 13.8887405, -19.4903240, 20.7392082
1: -14.1006050, 24.1030998, -12.2352343, 21.5304585, -35.6310616, 36.3383255
2: -9.7733765, 21.9032230, -8.5844498, 19.4446125, -29.2179890, 30.4876690
3: -15.1259575, 26.6727180, -13.2809668, 23.8006935, -38.9266510, 39.9536743
4: -13.6228781, 27.1033058, -12.1484928, 23.8559036, -37.4787827, 39.2518005

Time for backsubstitution: 0.96 seconds

### NS candidates at layer 1
type: A, layer: 1, pos: 12
type: B, layer: 1, pos: 39
type: B, layer: 1, pos: 28
type: A, layer: 1, pos: 45
type: A, layer: 1, pos: 28
type: B, layer: 1, pos: 6
type: B, layer: 1, pos: 45
type: A, layer: 1, pos: 24
type: B, layer: 1, pos: 24
type: A, layer: 1, pos: 15
type: A, layer: 1, pos: 20
type: A, layer: 1, pos: 6
type: B, layer: 1, pos: 15
type: A, layer: 1, pos: 39
type: B, layer: 1, pos: 20
type: A, layer: 1, pos: 38
type: A, layer: 1, pos: 37
type: A, layer: 1, pos: 48
type: B, layer: 1, pos: 26
type: A, layer: 1, pos: 26
type: B, layer: 1, pos: 37
type: A, layer: 1, pos: 8
type: A, layer: 1, pos: 9
type: B, layer: 1, pos: 38
type: A, layer: 1, pos: 0
type: B, layer: 1, pos: 48
type: B, layer: 1, pos: 0
type: B, layer: 1, pos: 41
type: A, layer: 1, pos: 41
type: A, layer: 1, pos: 5
type: A, layer: 1, pos: 31
type: B, layer: 1, pos: 31
type: B, layer: 1, pos: 5

Time for candidate selection: 0.08 seconds

### Candidate
type: A, layer: 1, pos: 12

### Candidate
type: B, layer: 1, pos: 39

### Candidate
type: B, layer: 1, pos: 28

### Candidate
type: A, layer: 1, pos: 45

### Candidate
type: A, layer: 1, pos: 28

### Candidate
type: B, layer: 1, pos: 6

### Candidate
type: B, layer: 1, pos: 45

### Candidate
type: A, layer: 1, pos: 24

### Candidate
type: B, layer: 1, pos: 24

### Candidate
type: A, layer: 1, pos: 15

### Candidate
type: A, layer: 1, pos: 20

## Relational analysis of NS_B2_B2_A1_B2_A1_B1_B2_A1
Optimization infeasible because this subproblem isn't reachable.

### Candidate
type: A, layer: 1, pos: 6

### Candidate
type: B, layer: 1, pos: 15

### Candidate
type: A, layer: 1, pos: 39

### Candidate
type: B, layer: 1, pos: 20

### Candidate
type: A, layer: 1, pos: 38

## Relational analysis of NS_B2_B2_A1_B2_A1_B1_B2_A1
Optimization infeasible because this subproblem isn't reachable.

### Candidate
type: A, layer: 1, pos: 37

## Relational analysis of NS_B2_B2_A1_B2_A1_B1_B2_A1
Optimization infeasible because this subproblem isn't reachable.

### Candidate
type: A, layer: 1, pos: 48

## Relational analysis of NS_B2_B2_A1_B2_A1_B1_B2_A1
Optimization infeasible because this subproblem isn't reachable.

### Candidate
type: B, layer: 1, pos: 26

## Relational analysis of NS_B2_B2_A1_B2_A1_B1_B2_B1
Optimization infeasible because this subproblem isn't reachable.

### Candidate
type: A, layer: 1, pos: 26

### Candidate
type: B, layer: 1, pos: 37

### Candidate
type: A, layer: 1, pos: 8

## Relational analysis of NS_B2_B2_A1_B2_A1_B1_B2_A1

### Relational analysis result of NS_B2_B2_A1_B2_A1_B1_B2_A1
Status: Status.UNKNOWN
Output dim: 0, lower bound: -20.9407029, upper bound: 20.9511217
time: 0.66 seconds

## Relational analysis of NS_B2_B2_A1_B2_A1_B1_B2_A2

### Relational analysis result of NS_B2_B2_A1_B2_A1_B1_B2_A2
Status: Status.UNKNOWN
Output dim: 0, lower bound: -20.9460557, upper bound: 20.9511217
time: 0.66 seconds

## BFS NS instance: NS_B2_B2_A1_B2_A1_B2_B1

### Backsubstitution after applying NS history:
0: -6.0244703, 16.8405571, -5.8083200, 16.1737309, -22.1982002, 22.6488762
1: -15.1832008, 25.7038307, -14.3167076, 25.0307140, -40.2139130, 40.0205345
2: -10.5292540, 23.3857155, -10.0512648, 22.5959682, -33.1252213, 33.4369812
3: -16.2657433, 28.4767113, -15.4965162, 27.6282158, -43.8939590, 43.9732285
4: -14.6604919, 28.9407387, -14.2014751, 27.7563248, -42.4168167, 43.1422043

Time for backsubstitution: 0.96 seconds

### NS candidates at layer 1
type: A, layer: 1, pos: 12
type: A, layer: 1, pos: 45
type: A, layer: 1, pos: 15
type: A, layer: 1, pos: 28
type: B, layer: 1, pos: 15
type: B, layer: 1, pos: 39
type: B, layer: 1, pos: 45
type: B, layer: 1, pos: 6
type: A, layer: 1, pos: 24
type: B, layer: 1, pos: 28
type: A, layer: 1, pos: 6
type: A, layer: 1, pos: 38
type: A, layer: 1, pos: 37
type: A, layer: 1, pos: 20
type: B, layer: 1, pos: 24
type: A, layer: 1, pos: 48
type: A, layer: 1, pos: 9
type: B, layer: 1, pos: 37
type: B, layer: 1, pos: 48
type: B, layer: 1, pos: 38
type: A, layer: 1, pos: 8
type: B, layer: 1, pos: 26
type: A, layer: 1, pos: 0
type: A, layer: 1, pos: 26
type: A, layer: 1, pos: 39
type: B, layer: 1, pos: 31
type: A, layer: 1, pos: 5
type: B, layer: 1, pos: 41
type: A, layer: 1, pos: 41
type: B, layer: 1, pos: 20
type: B, layer: 1, pos: 0
type: A, layer: 1, pos: 31
type: B, layer: 1, pos: 5

Time for candidate selection: 0.08 seconds

### Candidate
type: A, layer: 1, pos: 12

### Candidate
type: A, layer: 1, pos: 45

### Candidate
type: A, layer: 1, pos: 15

### Candidate
type: A, layer: 1, pos: 28

### Candidate
type: B, layer: 1, pos: 15

### Candidate
type: B, layer: 1, pos: 39

### Candidate
type: B, layer: 1, pos: 45

### Candidate
type: B, layer: 1, pos: 6

### Candidate
type: A, layer: 1, pos: 24

### Candidate
type: B, layer: 1, pos: 28

### Candidate
type: A, layer: 1, pos: 6

### Candidate
type: A, layer: 1, pos: 38

## Relational analysis of NS_B2_B2_A1_B2_A1_B2_B1_A1
Optimization infeasible because this subproblem isn't reachable.

### Candidate
type: A, layer: 1, pos: 37

## Relational analysis of NS_B2_B2_A1_B2_A1_B2_B1_A1
Optimization infeasible because this subproblem isn't reachable.

### Candidate
type: A, layer: 1, pos: 20

## Relational analysis of NS_B2_B2_A1_B2_A1_B2_B1_A1

### Relational analysis result of NS_B2_B2_A1_B2_A1_B2_B1_A1
Status: Status.UNKNOWN
Output dim: 0, lower bound: -20.9227677, upper bound: 20.9353666
time: 0.73 seconds

## Relational analysis of NS_B2_B2_A1_B2_A1_B2_B1_A2

### Relational analysis result of NS_B2_B2_A1_B2_A1_B2_B1_A2
Status: Status.UNKNOWN
Output dim: 0, lower bound: -20.9449710, upper bound: 20.9514500
time: 1.20 seconds

## BFS NS instance: NS_B2_B2_A1_B2_A1_B2_B2

### Backsubstitution after applying NS history:
0: -5.9729180, 16.7020836, -5.6549149, 15.7244186, -21.6973324, 22.3569984
1: -15.0557423, 25.4947567, -13.9724092, 24.2675362, -39.3232803, 39.4671516
2: -10.4393234, 23.1972771, -9.8013439, 21.9487000, -32.3880196, 32.9986191
3: -16.1292458, 28.2445431, -15.0965834, 26.8190613, -42.9483070, 43.3411255
4: -14.5333204, 28.7094154, -13.7853985, 26.9767647, -41.5100861, 42.4948120

Time for backsubstitution: 0.96 seconds

### NS candidates at layer 1
type: A, layer: 1, pos: 12
type: A, layer: 1, pos: 28
type: B, layer: 1, pos: 6
type: B, layer: 1, pos: 39
type: A, layer: 1, pos: 15
type: A, layer: 1, pos: 45
type: A, layer: 1, pos: 6
type: B, layer: 1, pos: 28
type: A, layer: 1, pos: 24
type: B, layer: 1, pos: 15
type: B, layer: 1, pos: 45
type: B, layer: 1, pos: 24
type: A, layer: 1, pos: 48
type: A, layer: 1, pos: 38
type: A, layer: 1, pos: 37
type: A, layer: 1, pos: 20
type: B, layer: 1, pos: 37
type: B, layer: 1, pos: 38
type: B, layer: 1, pos: 48
type: A, layer: 1, pos: 39
type: A, layer: 1, pos: 0
type: A, layer: 1, pos: 9
type: B, layer: 1, pos: 26
type: A, layer: 1, pos: 26
type: A, layer: 1, pos: 8
type: A, layer: 1, pos: 5
type: B, layer: 1, pos: 31
type: B, layer: 1, pos: 0
type: B, layer: 1, pos: 41
type: A, layer: 1, pos: 41
type: B, layer: 1, pos: 20
type: B, layer: 1, pos: 5
type: A, layer: 1, pos: 31

Time for candidate selection: 0.08 seconds

### Candidate
type: A, layer: 1, pos: 12

### Candidate
type: A, layer: 1, pos: 28

### Candidate
type: B, layer: 1, pos: 6

### Candidate
type: B, layer: 1, pos: 39

### Candidate
type: A, layer: 1, pos: 15

### Candidate
type: A, layer: 1, pos: 45

### Candidate
type: A, layer: 1, pos: 6

### Candidate
type: B, layer: 1, pos: 28

### Candidate
type: A, layer: 1, pos: 24

### Candidate
type: B, layer: 1, pos: 15

### Candidate
type: B, layer: 1, pos: 45

### Candidate
type: B, layer: 1, pos: 24

### Candidate
type: A, layer: 1, pos: 48

## Relational analysis of NS_B2_B2_A1_B2_A1_B2_B2_A1
Optimization infeasible because this subproblem isn't reachable.

### Candidate
type: A, layer: 1, pos: 38

## Relational analysis of NS_B2_B2_A1_B2_A1_B2_B2_A1
Optimization infeasible because this subproblem isn't reachable.

### Candidate
type: A, layer: 1, pos: 37

## Relational analysis of NS_B2_B2_A1_B2_A1_B2_B2_A1
Optimization infeasible because this subproblem isn't reachable.

### Candidate
type: A, layer: 1, pos: 20

## Relational analysis of NS_B2_B2_A1_B2_A1_B2_B2_A1

### Relational analysis result of NS_B2_B2_A1_B2_A1_B2_B2_A1
Status: Status.UNKNOWN
Output dim: 0, lower bound: -20.9440242, upper bound: 20.9476770
time: 0.71 seconds

## Relational analysis of NS_B2_B2_A1_B2_A1_B2_B2_A2

### Relational analysis result of NS_B2_B2_A1_B2_A1_B2_B2_A2
Status: Status.UNKNOWN
Output dim: 0, lower bound: -20.9523718, upper bound: 20.9564293
time: 1.73 seconds

## BFS NS instance: NS_B2_B2_A1_B2_A2_B1_B1

### Backsubstitution after applying NS history:
0: -5.8963809, 16.4548492, -5.1427388, 14.3199282, -20.2163086, 21.5975857
1: -14.7113638, 25.2500343, -12.5697231, 22.2789974, -36.9903603, 37.8197556
2: -10.2584276, 22.9362125, -8.8216429, 20.0713501, -30.3297768, 31.7578506
3: -15.8670597, 27.9255943, -13.6617050, 24.5833302, -40.4503746, 41.5872955
4: -14.4673929, 28.3199825, -12.5392199, 24.6038437, -39.0712357, 40.8591995

Time for backsubstitution: 0.97 seconds

### NS candidates at layer 1
type: A, layer: 1, pos: 12
type: A, layer: 1, pos: 45
type: B, layer: 1, pos: 45
type: A, layer: 1, pos: 15
type: B, layer: 1, pos: 28
type: B, layer: 1, pos: 15
type: A, layer: 1, pos: 28
type: B, layer: 1, pos: 6
type: B, layer: 1, pos: 39
type: B, layer: 1, pos: 24
type: A, layer: 1, pos: 24
type: A, layer: 1, pos: 48
type: A, layer: 1, pos: 9
type: B, layer: 1, pos: 20
type: A, layer: 1, pos: 37
type: A, layer: 1, pos: 38
type: A, layer: 1, pos: 6
type: A, layer: 1, pos: 39
type: B, layer: 1, pos: 37
type: A, layer: 1, pos: 20
type: A, layer: 1, pos: 8
type: B, layer: 1, pos: 38
type: B, layer: 1, pos: 26
type: A, layer: 1, pos: 26
type: B, layer: 1, pos: 48
type: A, layer: 1, pos: 0
type: B, layer: 1, pos: 0
type: A, layer: 1, pos: 31
type: B, layer: 1, pos: 41
type: A, layer: 1, pos: 41
type: A, layer: 1, pos: 5
type: B, layer: 1, pos: 31
type: B, layer: 1, pos: 5

Time for candidate selection: 0.08 seconds

### Candidate
type: A, layer: 1, pos: 12

### Candidate
type: A, layer: 1, pos: 45

### Candidate
type: B, layer: 1, pos: 45

### Candidate
type: A, layer: 1, pos: 15

### Candidate
type: B, layer: 1, pos: 28

### Candidate
type: B, layer: 1, pos: 15

### Candidate
type: A, layer: 1, pos: 28

### Candidate
type: B, layer: 1, pos: 6

### Candidate
type: B, layer: 1, pos: 39

### Candidate
type: B, layer: 1, pos: 24

### Candidate
type: A, layer: 1, pos: 24

### Candidate
type: A, layer: 1, pos: 48

### Candidate
type: A, layer: 1, pos: 9

## Relational analysis of NS_B2_B2_A1_B2_A2_B1_B1_A1

### Relational analysis result of NS_B2_B2_A1_B2_A2_B1_B1_A1
Status: Status.UNKNOWN
Output dim: 0, lower bound: -20.9155543, upper bound: 20.9219504
time: 0.70 seconds

## Relational analysis of NS_B2_B2_A1_B2_A2_B1_B1_A2

### Relational analysis result of NS_B2_B2_A1_B2_A2_B1_B1_A2
Status: Status.UNKNOWN
Output dim: 0, lower bound: -20.9273829, upper bound: 20.9235466
time: 0.72 seconds

## BFS NS instance: NS_B2_B2_A1_B2_A2_B1_B2

### Backsubstitution after applying NS history:
0: -5.8457856, 16.3214684, -4.9961505, 13.8887405, -19.7345238, 21.3176193
1: -14.5887270, 25.0508442, -12.2352343, 21.5304585, -36.1191864, 37.2860718
2: -10.1715240, 22.7544518, -8.5844498, 19.4446125, -29.6161366, 31.3388939
3: -15.7338409, 27.7051315, -13.2809668, 23.8006935, -39.5345268, 40.9860878
4: -14.3416634, 28.0981274, -12.1484928, 23.8559036, -38.1975670, 40.2466202

Time for backsubstitution: 0.97 seconds

### NS candidates at layer 1
type: A, layer: 1, pos: 12
type: A, layer: 1, pos: 45
type: B, layer: 1, pos: 45
type: B, layer: 1, pos: 28
type: B, layer: 1, pos: 39
type: B, layer: 1, pos: 6
type: A, layer: 1, pos: 28
type: B, layer: 1, pos: 24
type: A, layer: 1, pos: 24
type: A, layer: 1, pos: 15
type: A, layer: 1, pos: 39
type: A, layer: 1, pos: 6
type: A, layer: 1, pos: 48
type: B, layer: 1, pos: 15
type: B, layer: 1, pos: 20
type: A, layer: 1, pos: 38
type: A, layer: 1, pos: 20
type: A, layer: 1, pos: 37
type: A, layer: 1, pos: 9
type: A, layer: 1, pos: 8
type: B, layer: 1, pos: 37
type: B, layer: 1, pos: 38
type: A, layer: 1, pos: 26
type: B, layer: 1, pos: 26
type: A, layer: 1, pos: 0
type: B, layer: 1, pos: 0
type: B, layer: 1, pos: 48
type: A, layer: 1, pos: 31
type: B, layer: 1, pos: 41
type: A, layer: 1, pos: 41
type: B, layer: 1, pos: 5
type: A, layer: 1, pos: 5
type: B, layer: 1, pos: 31

Time for candidate selection: 0.08 seconds

### Candidate
type: A, layer: 1, pos: 12

### Candidate
type: A, layer: 1, pos: 45

### Candidate
type: B, layer: 1, pos: 45

### Candidate
type: B, layer: 1, pos: 28

### Candidate
type: B, layer: 1, pos: 39

### Candidate
type: B, layer: 1, pos: 6

### Candidate
type: A, layer: 1, pos: 28

### Candidate
type: B, layer: 1, pos: 24

### Candidate
type: A, layer: 1, pos: 24

### Candidate
type: A, layer: 1, pos: 15

### Candidate
type: A, layer: 1, pos: 39

### Candidate
type: A, layer: 1, pos: 6

### Candidate
type: A, layer: 1, pos: 48

### Candidate
type: B, layer: 1, pos: 15

### Candidate
type: B, layer: 1, pos: 20

## Relational analysis of NS_B2_B2_A1_B2_A2_B1_B2_B1
Optimization infeasible because this subproblem isn't reachable.

### Candidate
type: A, layer: 1, pos: 38

## Relational analysis of NS_B2_B2_A1_B2_A2_B1_B2_A1
Optimization infeasible because this subproblem isn't reachable.

### Candidate
type: A, layer: 1, pos: 20

### Candidate
type: A, layer: 1, pos: 37

## Relational analysis of NS_B2_B2_A1_B2_A2_B1_B2_A1

### Relational analysis result of NS_B2_B2_A1_B2_A2_B1_B2_A1
Status: Status.UNKNOWN
Output dim: 0, lower bound: -20.9066594, upper bound: 20.9106975
time: 0.62 seconds

## Relational analysis of NS_B2_B2_A1_B2_A2_B1_B2_A2

### Relational analysis result of NS_B2_B2_A1_B2_A2_B1_B2_A2
Status: Status.UNKNOWN
Output dim: 0, lower bound: -20.9313827, upper bound: 20.9330553
time: 0.84 seconds

## BFS NS instance: NS_B2_B2_A1_B2_A2_B2_B1

### Backsubstitution after applying NS history:
0: -6.2250795, 17.2942638, -5.8083200, 16.1737309, -22.3988075, 23.1025810
1: -15.5413942, 26.4833488, -14.3167076, 25.0307140, -40.5721016, 40.8000526
2: -10.8425531, 24.0679264, -10.0512648, 22.5959682, -33.4385185, 34.1191902
3: -16.7520714, 29.3229733, -15.4965162, 27.6282158, -44.3802872, 44.8194885
4: -15.2782488, 29.7155247, -14.2014751, 27.7563248, -43.0345726, 43.9169998

Time for backsubstitution: 0.97 seconds

### NS candidates at layer 1
type: A, layer: 1, pos: 12
type: A, layer: 1, pos: 45
type: B, layer: 1, pos: 45
type: A, layer: 1, pos: 15
type: B, layer: 1, pos: 15
type: A, layer: 1, pos: 28
type: B, layer: 1, pos: 6
type: A, layer: 1, pos: 6
type: B, layer: 1, pos: 28
type: A, layer: 1, pos: 48
type: A, layer: 1, pos: 24
type: B, layer: 1, pos: 24
type: B, layer: 1, pos: 39
type: A, layer: 1, pos: 9
type: A, layer: 1, pos: 38
type: A, layer: 1, pos: 37
type: B, layer: 1, pos: 37
type: A, layer: 1, pos: 20
type: B, layer: 1, pos: 38
type: A, layer: 1, pos: 8
type: A, layer: 1, pos: 39
type: B, layer: 1, pos: 48
type: B, layer: 1, pos: 26
type: A, layer: 1, pos: 26
type: B, layer: 1, pos: 20
type: A, layer: 1, pos: 0
type: B, layer: 1, pos: 31
type: B, layer: 1, pos: 41
type: A, layer: 1, pos: 41
type: A, layer: 1, pos: 5
type: B, layer: 1, pos: 5
type: A, layer: 1, pos: 31
type: B, layer: 1, pos: 0

Time for candidate selection: 0.08 seconds

### Candidate
type: A, layer: 1, pos: 12

### Candidate
type: A, layer: 1, pos: 45

### Candidate
type: B, layer: 1, pos: 45

### Candidate
type: A, layer: 1, pos: 15

### Candidate
type: B, layer: 1, pos: 15

### Candidate
type: A, layer: 1, pos: 28

### Candidate
type: B, layer: 1, pos: 6

### Candidate
type: A, layer: 1, pos: 6

### Candidate
type: B, layer: 1, pos: 28

### Candidate
type: A, layer: 1, pos: 48

### Candidate
type: A, layer: 1, pos: 24

### Candidate
type: B, layer: 1, pos: 24

### Candidate
type: B, layer: 1, pos: 39

## Relational analysis of NS_B2_B2_A1_B2_A2_B2_B1_B1
Optimization infeasible because this subproblem isn't reachable.

### Candidate
type: A, layer: 1, pos: 9

## Relational analysis of NS_B2_B2_A1_B2_A2_B2_B1_A1

### Relational analysis result of NS_B2_B2_A1_B2_A2_B2_B1_A1
Status: Status.UNKNOWN
Output dim: 0, lower bound: -20.9147679, upper bound: 20.9221320
time: 1.08 seconds

## Relational analysis of NS_B2_B2_A1_B2_A2_B2_B1_A2

### Relational analysis result of NS_B2_B2_A1_B2_A2_B2_B1_A2
Status: Status.UNKNOWN
Output dim: 0, lower bound: -20.9272313, upper bound: 20.9235597
time: 0.65 seconds

## BFS NS instance: NS_B2_B2_A1_B2_A2_B2_B2

### Backsubstitution after applying NS history:
0: -6.1712074, 17.1524982, -5.6549149, 15.7244186, -21.8956242, 22.8074131
1: -15.4109011, 26.2710419, -13.9724092, 24.2675362, -39.6784325, 40.2434502
2: -10.7504940, 23.8740902, -9.8013439, 21.9487000, -32.6991882, 33.6754341
3: -16.6102028, 29.0874271, -15.0965834, 26.8190613, -43.4292564, 44.1840096
4: -15.1453896, 29.4794388, -13.7853985, 26.9767647, -42.1221542, 43.2648392

Time for backsubstitution: 0.98 seconds

### NS candidates at layer 1
type: A, layer: 1, pos: 12
type: A, layer: 1, pos: 45
type: A, layer: 1, pos: 28
type: B, layer: 1, pos: 45
type: B, layer: 1, pos: 6
type: A, layer: 1, pos: 6
type: A, layer: 1, pos: 15
type: B, layer: 1, pos: 28
type: A, layer: 1, pos: 24
type: B, layer: 1, pos: 15
type: A, layer: 1, pos: 48
type: B, layer: 1, pos: 24
type: B, layer: 1, pos: 39
type: A, layer: 1, pos: 38
type: A, layer: 1, pos: 37
type: B, layer: 1, pos: 37
type: A, layer: 1, pos: 39
type: A, layer: 1, pos: 20
type: B, layer: 1, pos: 38
type: A, layer: 1, pos: 9
type: A, layer: 1, pos: 8
type: B, layer: 1, pos: 48
type: B, layer: 1, pos: 26
type: A, layer: 1, pos: 26
type: A, layer: 1, pos: 0
type: B, layer: 1, pos: 20
type: B, layer: 1, pos: 0
type: B, layer: 1, pos: 5
type: B, layer: 1, pos: 31
type: A, layer: 1, pos: 41
type: B, layer: 1, pos: 41
type: A, layer: 1, pos: 5
type: A, layer: 1, pos: 31

Time for candidate selection: 0.08 seconds

### Candidate
type: A, layer: 1, pos: 12

### Candidate
type: A, layer: 1, pos: 45

### Candidate
type: A, layer: 1, pos: 28

### Candidate
type: B, layer: 1, pos: 45

### Candidate
type: B, layer: 1, pos: 6

### Candidate
type: A, layer: 1, pos: 6

### Candidate
type: A, layer: 1, pos: 15

### Candidate
type: B, layer: 1, pos: 28

### Candidate
type: A, layer: 1, pos: 24

### Candidate
type: B, layer: 1, pos: 15

### Candidate
type: A, layer: 1, pos: 48

### Candidate
type: B, layer: 1, pos: 24

### Candidate
type: B, layer: 1, pos: 39

## Relational analysis of NS_B2_B2_A1_B2_A2_B2_B2_B1
Optimization infeasible because this subproblem isn't reachable.

### Candidate
type: A, layer: 1, pos: 38

## Relational analysis of NS_B2_B2_A1_B2_A2_B2_B2_A1
Optimization infeasible because this subproblem isn't reachable.

### Candidate
type: A, layer: 1, pos: 37

## Relational analysis of NS_B2_B2_A1_B2_A2_B2_B2_A1

### Relational analysis result of NS_B2_B2_A1_B2_A2_B2_B2_A1
Status: Status.UNKNOWN
Output dim: 0, lower bound: -20.9290999, upper bound: 20.9170359
time: 1.06 seconds

## Relational analysis of NS_B2_B2_A1_B2_A2_B2_B2_A2

### Relational analysis result of NS_B2_B2_A1_B2_A2_B2_B2_A2
Status: Status.UNKNOWN
Output dim: 0, lower bound: -20.9404709, upper bound: 20.9393947
time: 0.75 seconds

## BFS NS instance: NS_B2_B2_A2_B1_A1_A1_B1

### Backsubstitution after applying NS history:
0: -5.3626714, 15.1720085, -5.1090603, 14.2706547, -19.6333256, 20.2810688
1: -13.3945961, 23.4509029, -12.6505194, 22.0396614, -35.4342575, 36.1014214
2: -9.3099127, 21.2018681, -8.8227625, 19.9193935, -29.2293034, 30.0246277
3: -14.4439678, 25.8672752, -13.6273718, 24.3816147, -38.8255844, 39.4946442
4: -13.0890808, 26.1884937, -12.3172474, 24.4959354, -37.5850143, 38.5057411

Time for backsubstitution: 0.98 seconds

### NS candidates at layer 1
type: B, layer: 1, pos: 28
type: A, layer: 1, pos: 45
type: A, layer: 1, pos: 39
type: A, layer: 1, pos: 28
type: B, layer: 1, pos: 6
type: B, layer: 1, pos: 45
type: B, layer: 1, pos: 24
type: B, layer: 1, pos: 39
type: A, layer: 1, pos: 24
type: B, layer: 1, pos: 9
type: A, layer: 1, pos: 6
type: A, layer: 1, pos: 9
type: B, layer: 1, pos: 20
type: A, layer: 1, pos: 15
type: B, layer: 1, pos: 15
type: A, layer: 1, pos: 8
type: A, layer: 1, pos: 37
type: A, layer: 1, pos: 26
type: B, layer: 1, pos: 37
type: B, layer: 1, pos: 26
type: A, layer: 1, pos: 38
type: B, layer: 1, pos: 38
type: A, layer: 1, pos: 48
type: A, layer: 1, pos: 20
type: B, layer: 1, pos: 48
type: B, layer: 1, pos: 0
type: A, layer: 1, pos: 0
type: A, layer: 1, pos: 31
type: A, layer: 1, pos: 41
type: B, layer: 1, pos: 41
type: A, layer: 1, pos: 5
type: B, layer: 1, pos: 31
type: B, layer: 1, pos: 5

Time for candidate selection: 0.08 seconds

### Candidate
type: B, layer: 1, pos: 28

### Candidate
type: A, layer: 1, pos: 45

### Candidate
type: A, layer: 1, pos: 39

### Candidate
type: A, layer: 1, pos: 28

### Candidate
type: B, layer: 1, pos: 6

### Candidate
type: B, layer: 1, pos: 45

### Candidate
type: B, layer: 1, pos: 24

### Candidate
type: B, layer: 1, pos: 39

### Candidate
type: A, layer: 1, pos: 24

### Candidate
type: B, layer: 1, pos: 9

## Relational analysis of NS_B2_B2_A2_B1_A1_A1_B1_B1
Optimization infeasible because this subproblem isn't reachable.

### Candidate
type: A, layer: 1, pos: 6

### Candidate
type: A, layer: 1, pos: 9

### Candidate
type: B, layer: 1, pos: 20

## Relational analysis of NS_B2_B2_A2_B1_A1_A1_B1_B1
Optimization infeasible because this subproblem isn't reachable.

### Candidate
type: A, layer: 1, pos: 15

### Candidate
type: B, layer: 1, pos: 15

### Candidate
type: A, layer: 1, pos: 8

## Relational analysis of NS_B2_B2_A2_B1_A1_A1_B1_A1
Optimization infeasible because this subproblem isn't reachable.

### Candidate
type: A, layer: 1, pos: 37

## Relational analysis of NS_B2_B2_A2_B1_A1_A1_B1_A1

### Relational analysis result of NS_B2_B2_A2_B1_A1_A1_B1_A1
Status: Status.UNKNOWN
Output dim: 0, lower bound: -20.9610804, upper bound: 20.9591284
time: 0.67 seconds

## Relational analysis of NS_B2_B2_A2_B1_A1_A1_B1_A2

### Relational analysis result of NS_B2_B2_A2_B1_A1_A1_B1_A2
Status: Status.UNKNOWN
Output dim: 0, lower bound: -20.9637382, upper bound: 20.9626796
time: 0.76 seconds

## BFS NS instance: NS_B2_B2_A2_B1_A1_A1_B2

### Backsubstitution after applying NS history:
0: -5.6934490, 16.0427685, -5.7379360, 16.0502377, -21.7436867, 21.7807045
1: -14.2588348, 24.7153702, -14.3143644, 24.7327061, -38.9915352, 39.0297356
2: -9.9096851, 22.3794670, -9.9866447, 22.3560219, -32.2657089, 32.3661118
3: -15.3478031, 27.2905960, -15.3614969, 27.3282509, -42.6760559, 42.6520920
4: -13.9048977, 27.6525459, -13.8753071, 27.5357323, -41.4406281, 41.5278549

Time for backsubstitution: 0.98 seconds

### NS candidates at layer 1
type: B, layer: 1, pos: 28
type: A, layer: 1, pos: 28
type: A, layer: 1, pos: 45
type: B, layer: 1, pos: 45
type: B, layer: 1, pos: 6
type: A, layer: 1, pos: 15
type: A, layer: 1, pos: 39
type: B, layer: 1, pos: 15
type: A, layer: 1, pos: 6
type: B, layer: 1, pos: 24
type: A, layer: 1, pos: 24
type: B, layer: 1, pos: 39
type: B, layer: 1, pos: 9
type: A, layer: 1, pos: 9
type: A, layer: 1, pos: 37
type: A, layer: 1, pos: 38
type: A, layer: 1, pos: 48
type: B, layer: 1, pos: 37
type: B, layer: 1, pos: 38
type: A, layer: 1, pos: 8
type: B, layer: 1, pos: 48
type: A, layer: 1, pos: 26
type: B, layer: 1, pos: 26
type: A, layer: 1, pos: 20
type: B, layer: 1, pos: 20
type: B, layer: 1, pos: 31
type: A, layer: 1, pos: 31
type: A, layer: 1, pos: 0
type: B, layer: 1, pos: 0
type: A, layer: 1, pos: 41
type: B, layer: 1, pos: 41
type: A, layer: 1, pos: 5
type: B, layer: 1, pos: 5

Time for candidate selection: 0.08 seconds

### Candidate
type: B, layer: 1, pos: 28

### Candidate
type: A, layer: 1, pos: 28

### Candidate
type: A, layer: 1, pos: 45

### Candidate
type: B, layer: 1, pos: 45

### Candidate
type: B, layer: 1, pos: 6

### Candidate
type: A, layer: 1, pos: 15

### Candidate
type: A, layer: 1, pos: 39

### Candidate
type: B, layer: 1, pos: 15

### Candidate
type: A, layer: 1, pos: 6

### Candidate
type: B, layer: 1, pos: 24

### Candidate
type: A, layer: 1, pos: 24

### Candidate
type: B, layer: 1, pos: 39

### Candidate
type: B, layer: 1, pos: 9

## Relational analysis of NS_B2_B2_A2_B1_A1_A1_B2_B1

### Relational analysis result of NS_B2_B2_A2_B1_A1_A1_B2_B1
Status: Status.UNKNOWN
Output dim: 0, lower bound: -20.9479032, upper bound: 20.9513797
time: 0.71 seconds

## Relational analysis of NS_B2_B2_A2_B1_A1_A1_B2_B2
Optimization infeasible because this subproblem isn't reachable.

### Candidate
type: A, layer: 1, pos: 9

### Candidate
type: A, layer: 1, pos: 37

## Relational analysis of NS_B2_B2_A2_B1_A1_A1_B2_A1

### Relational analysis result of NS_B2_B2_A2_B1_A1_A1_B2_A1
Status: Status.UNKNOWN
Output dim: 0, lower bound: -20.9635723, upper bound: 20.9630608
time: 0.69 seconds

## Relational analysis of NS_B2_B2_A2_B1_A1_A1_B2_A2

### Relational analysis result of NS_B2_B2_A2_B1_A1_A1_B2_A2
Status: Status.UNKNOWN
Output dim: 0, lower bound: -20.9637382, upper bound: 20.9644216
time: 0.69 seconds

## BFS NS instance: NS_B2_B2_A2_B1_A1_A2_B1

### Backsubstitution after applying NS history:
0: -5.6418986, 15.8690166, -5.1090603, 14.2706547, -19.9125538, 20.9780769
1: -13.9814291, 24.5670586, -12.6505194, 22.0396614, -36.0210915, 37.2175674
2: -9.7727718, 22.2125931, -8.8227625, 19.9193935, -29.6921654, 31.0353546
3: -15.1517401, 27.0732784, -13.6273718, 24.3816147, -39.5333557, 40.7006493
4: -13.8984947, 27.3849049, -12.3172474, 24.4959354, -38.3944321, 39.7021523

Time for backsubstitution: 0.98 seconds

### NS candidates at layer 1
type: A, layer: 1, pos: 45
type: A, layer: 1, pos: 39
type: B, layer: 1, pos: 45
type: B, layer: 1, pos: 28
type: A, layer: 1, pos: 15
type: B, layer: 1, pos: 15
type: B, layer: 1, pos: 20
type: A, layer: 1, pos: 48
type: B, layer: 1, pos: 38
type: A, layer: 1, pos: 38
type: B, layer: 1, pos: 6
type: B, layer: 1, pos: 39
type: A, layer: 1, pos: 9
type: B, layer: 1, pos: 9
type: A, layer: 1, pos: 20
type: B, layer: 1, pos: 24
type: A, layer: 1, pos: 8
type: A, layer: 1, pos: 28
type: A, layer: 1, pos: 6
type: B, layer: 1, pos: 48
type: B, layer: 1, pos: 37
type: A, layer: 1, pos: 37
type: B, layer: 1, pos: 0
type: A, layer: 1, pos: 31
type: A, layer: 1, pos: 24
type: B, layer: 1, pos: 31
type: A, layer: 1, pos: 0
type: B, layer: 1, pos: 5
type: A, layer: 1, pos: 26
type: B, layer: 1, pos: 26
type: A, layer: 1, pos: 5
type: A, layer: 1, pos: 41
type: B, layer: 1, pos: 41

Time for candidate selection: 0.08 seconds

### Candidate
type: A, layer: 1, pos: 45

### Candidate
type: A, layer: 1, pos: 39

### Candidate
type: B, layer: 1, pos: 45

### Candidate
type: B, layer: 1, pos: 28

### Candidate
type: A, layer: 1, pos: 15

### Candidate
type: B, layer: 1, pos: 15

### Candidate
type: B, layer: 1, pos: 20

## Relational analysis of NS_B2_B2_A2_B1_A1_A2_B1_B1
Optimization infeasible because this subproblem isn't reachable.

### Candidate
type: A, layer: 1, pos: 48

### Candidate
type: B, layer: 1, pos: 38

### Candidate
type: A, layer: 1, pos: 38

### Candidate
type: B, layer: 1, pos: 6

## Relational analysis of NS_B2_B2_A2_B1_A1_A2_B1_B1
Optimization infeasible because this subproblem isn't reachable.

### Candidate
type: B, layer: 1, pos: 39

### Candidate
type: A, layer: 1, pos: 9

### Candidate
type: B, layer: 1, pos: 9

### Candidate
type: A, layer: 1, pos: 20

### Candidate
type: B, layer: 1, pos: 24

## Relational analysis of NS_B2_B2_A2_B1_A1_A2_B1_B1
Optimization infeasible because this subproblem isn't reachable.

### Candidate
type: A, layer: 1, pos: 8

## Relational analysis of NS_B2_B2_A2_B1_A1_A2_B1_A1
Optimization infeasible because this subproblem isn't reachable.

### Candidate
type: A, layer: 1, pos: 28

### Candidate
type: A, layer: 1, pos: 6

### Candidate
type: B, layer: 1, pos: 48

### Candidate
type: B, layer: 1, pos: 37

## Relational analysis of NS_B2_B2_A2_B1_A1_A2_B1_B1
Optimization infeasible because this subproblem isn't reachable.

### Candidate
type: A, layer: 1, pos: 37

### Candidate
type: B, layer: 1, pos: 0

## Relational analysis of NS_B2_B2_A2_B1_A1_A2_B1_B1

### Relational analysis result of NS_B2_B2_A2_B1_A1_A2_B1_B1
Status: Status.UNKNOWN
Output dim: 0, lower bound: -20.9568022, upper bound: 20.9532524
time: 0.71 seconds

## Relational analysis of NS_B2_B2_A2_B1_A1_A2_B1_B2
Optimization infeasible because this subproblem isn't reachable.

### Candidate
type: A, layer: 1, pos: 31

## Relational analysis of NS_B2_B2_A2_B1_A1_A2_B1_A1
Optimization infeasible because this subproblem isn't reachable.

### Candidate
type: A, layer: 1, pos: 24

### Candidate
type: B, layer: 1, pos: 31

### Candidate
type: A, layer: 1, pos: 0

### Candidate
type: B, layer: 1, pos: 5

## Relational analysis of NS_B2_B2_A2_B1_A1_A2_B1_B1
Optimization infeasible because this subproblem isn't reachable.

### Candidate
type: A, layer: 1, pos: 26

## Relational analysis of NS_B2_B2_A2_B1_A1_A2_B1_A1
Optimization infeasible because this subproblem isn't reachable.

### Candidate
type: B, layer: 1, pos: 26

### Candidate
type: A, layer: 1, pos: 5

### Candidate
type: A, layer: 1, pos: 41

## Relational analysis of NS_B2_B2_A2_B1_A1_A2_B1_A1
Optimization infeasible because this subproblem isn't reachable.

### Candidate
type: B, layer: 1, pos: 41

## BFS NS instance: NS_B2_B2_A2_B1_A1_A2_B2

### Backsubstitution after applying NS history:
0: -5.9458299, 16.6518002, -5.7379360, 16.0502377, -21.9960670, 22.3897362
1: -14.7555151, 25.7168369, -14.3143644, 24.7327061, -39.4882202, 40.0312004
2: -10.3130445, 23.2755966, -9.9866447, 22.3560219, -32.6690674, 33.2622375
3: -15.9753675, 28.3708611, -15.3614969, 27.3282509, -43.3036194, 43.7323570
4: -14.6584167, 28.6986294, -13.8753071, 27.5357323, -42.1941490, 42.5739365

Time for backsubstitution: 0.99 seconds

### NS candidates at layer 1
type: A, layer: 1, pos: 45
type: B, layer: 1, pos: 45
type: A, layer: 1, pos: 15
type: B, layer: 1, pos: 15
type: A, layer: 1, pos: 39
type: B, layer: 1, pos: 28
type: A, layer: 1, pos: 48
type: A, layer: 1, pos: 38
type: B, layer: 1, pos: 38
type: A, layer: 1, pos: 9
type: B, layer: 1, pos: 9
type: A, layer: 1, pos: 28
type: B, layer: 1, pos: 6
type: B, layer: 1, pos: 20
type: B, layer: 1, pos: 39
type: A, layer: 1, pos: 6
type: B, layer: 1, pos: 37
type: B, layer: 1, pos: 31
type: A, layer: 1, pos: 37
type: A, layer: 1, pos: 31
type: A, layer: 1, pos: 20
type: A, layer: 1, pos: 8
type: B, layer: 1, pos: 48
type: B, layer: 1, pos: 24
type: A, layer: 1, pos: 24
type: B, layer: 1, pos: 0
type: B, layer: 1, pos: 5
type: A, layer: 1, pos: 0
type: A, layer: 1, pos: 26
type: A, layer: 1, pos: 41
type: B, layer: 1, pos: 26
type: B, layer: 1, pos: 41
type: A, layer: 1, pos: 5

Time for candidate selection: 0.08 seconds

### Candidate
type: A, layer: 1, pos: 45

### Candidate
type: B, layer: 1, pos: 45

### Candidate
type: A, layer: 1, pos: 15

### Candidate
type: B, layer: 1, pos: 15

### Candidate
type: A, layer: 1, pos: 39

### Candidate
type: B, layer: 1, pos: 28

### Candidate
type: A, layer: 1, pos: 48

### Candidate
type: A, layer: 1, pos: 38

### Candidate
type: B, layer: 1, pos: 38

### Candidate
type: A, layer: 1, pos: 9

### Candidate
type: B, layer: 1, pos: 9

### Candidate
type: A, layer: 1, pos: 28

### Candidate
type: B, layer: 1, pos: 6

## Relational analysis of NS_B2_B2_A2_B1_A1_A2_B2_B1
Optimization infeasible because this subproblem isn't reachable.

### Candidate
type: B, layer: 1, pos: 20

## Relational analysis of NS_B2_B2_A2_B1_A1_A2_B2_B1

### Relational analysis result of NS_B2_B2_A2_B1_A1_A2_B2_B1
Status: Status.UNKNOWN
Output dim: 0, lower bound: -20.9535715, upper bound: 20.9507958
time: 0.69 seconds

## Relational analysis of NS_B2_B2_A2_B1_A1_A2_B2_B2

### Relational analysis result of NS_B2_B2_A2_B1_A1_A2_B2_B2
Status: Status.UNKNOWN
Output dim: 0, lower bound: -20.9610848, upper bound: 20.9582666
time: 0.80 seconds

## BFS NS instance: NS_B2_B2_A2_B1_A2_A1_A1

### Backsubstitution after applying NS history:
0: -4.8503046, 13.6482563, -5.3933949, 15.1001291, -19.9504299, 19.0416508
1: -11.9245682, 21.2466469, -13.3704920, 23.3481369, -35.2726974, 34.6171379
2: -8.3504200, 19.1107178, -9.3500547, 21.0751381, -29.4255562, 28.4607735
3: -12.8832979, 23.3857803, -14.3921585, 25.7675247, -38.6508217, 37.7779312
4: -11.7305784, 23.4330387, -13.0522251, 25.9072819, -37.6378517, 36.4852638

Time for backsubstitution: 0.99 seconds

### NS candidates at layer 1
type: A, layer: 1, pos: 28
type: B, layer: 1, pos: 28
type: A, layer: 1, pos: 45
type: B, layer: 1, pos: 45
type: B, layer: 1, pos: 6
type: A, layer: 1, pos: 6
type: B, layer: 1, pos: 24
type: A, layer: 1, pos: 24
type: B, layer: 1, pos: 39
type: A, layer: 1, pos: 39
type: B, layer: 1, pos: 15
type: A, layer: 1, pos: 15
type: B, layer: 1, pos: 20
type: B, layer: 1, pos: 9
type: A, layer: 1, pos: 9
type: A, layer: 1, pos: 8
type: B, layer: 1, pos: 8
type: A, layer: 1, pos: 37
type: B, layer: 1, pos: 48
type: B, layer: 1, pos: 38
type: A, layer: 1, pos: 38
type: B, layer: 1, pos: 37
type: A, layer: 1, pos: 26
type: B, layer: 1, pos: 26
type: A, layer: 1, pos: 48
type: A, layer: 1, pos: 31
type: B, layer: 1, pos: 31
type: B, layer: 1, pos: 0
type: A, layer: 1, pos: 0
type: A, layer: 1, pos: 5
type: B, layer: 1, pos: 41
type: A, layer: 1, pos: 41
type: B, layer: 1, pos: 5

Time for candidate selection: 0.08 seconds

### Candidate
type: A, layer: 1, pos: 28

### Candidate
type: B, layer: 1, pos: 28

### Candidate
type: A, layer: 1, pos: 45

### Candidate
type: B, layer: 1, pos: 45

### Candidate
type: B, layer: 1, pos: 6

## Relational analysis of NS_B2_B2_A2_B1_A2_A1_A1_B1
Optimization infeasible because this subproblem isn't reachable.

### Candidate
type: A, layer: 1, pos: 6

### Candidate
type: B, layer: 1, pos: 24

## Relational analysis of NS_B2_B2_A2_B1_A2_A1_A1_B1
Optimization infeasible because this subproblem isn't reachable.

### Candidate
type: A, layer: 1, pos: 24

### Candidate
type: B, layer: 1, pos: 39

## Relational analysis of NS_B2_B2_A2_B1_A2_A1_A1_B1

### Relational analysis result of NS_B2_B2_A2_B1_A2_A1_A1_B1
Status: Status.UNKNOWN
Output dim: 0, lower bound: -20.9535399, upper bound: 20.9524825
time: 1.13 seconds

## Relational analysis of NS_B2_B2_A2_B1_A2_A1_A1_B2
Optimization infeasible because this subproblem isn't reachable.

### Candidate
type: A, layer: 1, pos: 39

### Candidate
type: B, layer: 1, pos: 15

### Candidate
type: A, layer: 1, pos: 15

### Candidate
type: B, layer: 1, pos: 20

## Relational analysis of NS_B2_B2_A2_B1_A2_A1_A1_B1

### Relational analysis result of NS_B2_B2_A2_B1_A2_A1_A1_B1
Status: Status.UNKNOWN
Output dim: 0, lower bound: -20.9543912, upper bound: 20.9543120
time: 0.59 seconds

## Relational analysis of NS_B2_B2_A2_B1_A2_A1_A1_B2

### Relational analysis result of NS_B2_B2_A2_B1_A2_A1_A1_B2
Status: Status.UNKNOWN
Output dim: 0, lower bound: -20.9521551, upper bound: 20.9561758
time: 0.60 seconds

## BFS NS instance: NS_B2_B2_A2_B1_A2_A1_A2

### Backsubstitution after applying NS history:
0: -5.5884404, 15.6797314, -5.7219605, 16.0188351, -21.6072750, 21.4016857
1: -13.9275799, 24.1853638, -14.2732754, 24.6851349, -38.6127167, 38.4586411
2: -9.7165880, 21.8541718, -9.9579859, 22.3140278, -32.0306168, 31.8121567
3: -14.9607763, 26.7048531, -15.3200045, 27.2710991, -42.2318687, 42.0248528
4: -13.5168200, 26.9147892, -13.8380632, 27.4839764, -41.0007973, 40.7528496

Time for backsubstitution: 1.00 seconds

### NS candidates at layer 1
type: B, layer: 1, pos: 20
type: B, layer: 1, pos: 45
type: A, layer: 1, pos: 45
type: B, layer: 1, pos: 28
type: A, layer: 1, pos: 28
type: B, layer: 1, pos: 15
type: A, layer: 1, pos: 15
type: A, layer: 1, pos: 39
type: A, layer: 1, pos: 6
type: B, layer: 1, pos: 39
type: A, layer: 1, pos: 8
type: A, layer: 1, pos: 24
type: B, layer: 1, pos: 24
type: B, layer: 1, pos: 48
type: A, layer: 1, pos: 9
type: B, layer: 1, pos: 9
type: B, layer: 1, pos: 6
type: B, layer: 1, pos: 37
type: B, layer: 1, pos: 8
type: A, layer: 1, pos: 37
type: B, layer: 1, pos: 38
type: A, layer: 1, pos: 48
type: A, layer: 1, pos: 38
type: A, layer: 1, pos: 26
type: B, layer: 1, pos: 26
type: A, layer: 1, pos: 31
type: B, layer: 1, pos: 31
type: B, layer: 1, pos: 0
type: B, layer: 1, pos: 5
type: A, layer: 1, pos: 5
type: A, layer: 1, pos: 0
type: B, layer: 1, pos: 41
type: A, layer: 1, pos: 41

Time for candidate selection: 0.08 seconds

### Candidate
type: B, layer: 1, pos: 20

## Relational analysis of NS_B2_B2_A2_B1_A2_A1_A2_B1

### Relational analysis result of NS_B2_B2_A2_B1_A2_A1_A2_B1
Status: Status.UNKNOWN
Output dim: 0, lower bound: -20.9540550, upper bound: 20.9569058
time: 0.59 seconds

## Relational analysis of NS_B2_B2_A2_B1_A2_A1_A2_B2

### Relational analysis result of NS_B2_B2_A2_B1_A2_A1_A2_B2
Status: Status.UNKNOWN
Output dim: 0, lower bound: -20.9540550, upper bound: 20.9644216
time: 0.72 seconds

## BFS NS instance: NS_B2_B2_A2_B1_A2_A2_B1

### Backsubstitution after applying NS history:
0: -5.4873309, 15.2898788, -4.8503046, 13.6482563, -19.1355839, 20.1401806
1: -13.4300127, 23.7614326, -11.9245682, 21.2466469, -34.6766586, 35.6859932
2: -9.4601784, 21.4030228, -8.3504200, 19.1107178, -28.5708961, 29.7534428
3: -14.5805817, 26.1856422, -12.8832979, 23.3857803, -37.9663582, 39.0689392
4: -13.4461670, 26.2365646, -11.7305784, 23.4330387, -36.8792038, 37.9671402

Time for backsubstitution: 0.99 seconds

### NS candidates at layer 1
type: B, layer: 1, pos: 28
type: A, layer: 1, pos: 45
type: B, layer: 1, pos: 45
type: A, layer: 1, pos: 28
type: A, layer: 1, pos: 6
type: B, layer: 1, pos: 24
type: A, layer: 1, pos: 39
type: B, layer: 1, pos: 6
type: A, layer: 1, pos: 15
type: A, layer: 1, pos: 24
type: B, layer: 1, pos: 15
type: A, layer: 1, pos: 48
type: A, layer: 1, pos: 9
type: A, layer: 1, pos: 20
type: A, layer: 1, pos: 8
type: B, layer: 1, pos: 9
type: B, layer: 1, pos: 8
type: B, layer: 1, pos: 37
type: B, layer: 1, pos: 38
type: B, layer: 1, pos: 39
type: A, layer: 1, pos: 38
type: A, layer: 1, pos: 37
type: A, layer: 1, pos: 26
type: B, layer: 1, pos: 26
type: B, layer: 1, pos: 0
type: A, layer: 1, pos: 31
type: B, layer: 1, pos: 48
type: B, layer: 1, pos: 31
type: B, layer: 1, pos: 5
type: A, layer: 1, pos: 0
type: A, layer: 1, pos: 41
type: B, layer: 1, pos: 41
type: A, layer: 1, pos: 5

Time for candidate selection: 0.08 seconds

### Candidate
type: B, layer: 1, pos: 28

### Candidate
type: A, layer: 1, pos: 45

### Candidate
type: B, layer: 1, pos: 45

### Candidate
type: A, layer: 1, pos: 28

### Candidate
type: A, layer: 1, pos: 6

## Relational analysis of NS_B2_B2_A2_B1_A2_A2_B1_A1
Optimization infeasible because this subproblem isn't reachable.

### Candidate
type: B, layer: 1, pos: 24

## Relational analysis of NS_B2_B2_A2_B1_A2_A2_B1_B1
Optimization infeasible because this subproblem isn't reachable.

### Candidate
type: A, layer: 1, pos: 39

## Relational analysis of NS_B2_B2_A2_B1_A2_A2_B1_A1
Optimization infeasible because this subproblem isn't reachable.

### Candidate
type: B, layer: 1, pos: 6

### Candidate
type: A, layer: 1, pos: 15

## Relational analysis of NS_B2_B2_A2_B1_A2_A2_B1_A1
Optimization infeasible because this subproblem isn't reachable.

### Candidate
type: A, layer: 1, pos: 24

### Candidate
type: B, layer: 1, pos: 15

### Candidate
type: A, layer: 1, pos: 48

## Relational analysis of NS_B2_B2_A2_B1_A2_A2_B1_A1
Optimization infeasible because this subproblem isn't reachable.

### Candidate
type: A, layer: 1, pos: 9

## Relational analysis of NS_B2_B2_A2_B1_A2_A2_B1_A1

### Relational analysis result of NS_B2_B2_A2_B1_A2_A2_B1_A1
Status: Status.UNKNOWN
Output dim: 0, lower bound: -20.9433058, upper bound: 20.9381379
time: 0.75 seconds

## Relational analysis of NS_B2_B2_A2_B1_A2_A2_B1_A2

### Relational analysis result of NS_B2_B2_A2_B1_A2_A2_B1_A2
Status: Status.UNKNOWN
Output dim: 0, lower bound: -20.9491533, upper bound: 20.9464072
time: 0.74 seconds

## BFS NS instance: NS_B2_B2_A2_B1_A2_A2_B2

### Backsubstitution after applying NS history:
0: -5.8704824, 16.3478336, -5.5884404, 15.6797314, -21.5502129, 21.9362736
1: -14.4790936, 25.2903061, -13.9275799, 24.1853638, -38.6644554, 39.2178841
2: -10.1700277, 22.8331394, -9.7165880, 21.8541718, -32.0241966, 32.5497284
3: -15.6631336, 27.9097290, -14.9607763, 26.7048531, -42.3679886, 42.8705025
4: -14.3603430, 28.0491638, -13.5168200, 26.9147892, -41.2751274, 41.5659828

Time for backsubstitution: 1.00 seconds

### NS candidates at layer 1
type: A, layer: 1, pos: 20
type: A, layer: 1, pos: 45
type: B, layer: 1, pos: 45
type: B, layer: 1, pos: 28
type: A, layer: 1, pos: 39
type: A, layer: 1, pos: 28
type: A, layer: 1, pos: 15
type: B, layer: 1, pos: 15
type: A, layer: 1, pos: 48
type: B, layer: 1, pos: 24
type: A, layer: 1, pos: 6
type: B, layer: 1, pos: 6
type: B, layer: 1, pos: 8
type: A, layer: 1, pos: 9
type: A, layer: 1, pos: 8
type: A, layer: 1, pos: 24
type: B, layer: 1, pos: 9
type: B, layer: 1, pos: 39
type: B, layer: 1, pos: 37
type: A, layer: 1, pos: 37
type: A, layer: 1, pos: 38
type: B, layer: 1, pos: 38
type: A, layer: 1, pos: 26
type: B, layer: 1, pos: 26
type: B, layer: 1, pos: 48
type: A, layer: 1, pos: 31
type: B, layer: 1, pos: 31
type: B, layer: 1, pos: 5
type: B, layer: 1, pos: 0
type: A, layer: 1, pos: 41
type: B, layer: 1, pos: 41
type: A, layer: 1, pos: 5
type: A, layer: 1, pos: 0

Time for candidate selection: 0.08 seconds

### Candidate
type: A, layer: 1, pos: 20

## Relational analysis of NS_B2_B2_A2_B1_A2_A2_B2_A1

### Relational analysis result of NS_B2_B2_A2_B1_A2_A2_B2_A1
Status: Status.UNKNOWN
Output dim: 0, lower bound: -20.9481945, upper bound: 20.9453032
time: 0.70 seconds

## Relational analysis of NS_B2_B2_A2_B1_A2_A2_B2_A2

### Relational analysis result of NS_B2_B2_A2_B1_A2_A2_B2_A2
Status: Status.UNKNOWN
Output dim: 0, lower bound: -20.9481945, upper bound: 20.9544784
time: 0.71 seconds

## Summary of splitting at layer (split count: 7)
- Time for NS candidates: 2.50 seconds
NS_B1_A1_A1_A1_B1_B1_A1_B1, status: Status.UNKNOWN, split count: 8, time: 2.50
Output dim: 0, lower bound: -20.9256445, upper bound: 20.9256445
NS_B1_A1_A1_A1_B1_B1_A1_B2, status: Status.UNKNOWN, split count: 8, time: 2.50
Output dim: 0, lower bound: -20.9256445, upper bound: 20.9397612
NS_B1_A1_A1_A1_B1_B1_A2_B1, status: Status.UNKNOWN, split count: 8, time: 2.50
Output dim: 0, lower bound: -20.9446710, upper bound: 20.9619725
NS_B1_A1_A1_A1_B1_B1_A2_B2, status: Status.UNKNOWN, split count: 8, time: 2.50
Output dim: 0, lower bound: -20.9636160, upper bound: 20.9636160
NS_B1_A1_A1_A1_B1_B2_A1_B1, status: Status.UNKNOWN, split count: 8, time: 2.50
Output dim: 0, lower bound: -20.9577507, upper bound: 20.9549928
NS_B1_A1_A1_A1_B1_B2_A1_B2, status: Status.UNKNOWN, split count: 8, time: 2.50
Output dim: 0, lower bound: -20.9577507, upper bound: 20.9549928
NS_B1_A1_A1_A1_B1_B2_A2_B1, status: Status.UNKNOWN, split count: 8, time: 2.50
Output dim: 0, lower bound: -20.9577507, upper bound: 20.9549928
NS_B1_A1_A1_A1_B1_B2_A2_B2, status: Status.UNKNOWN, split count: 8, time: 2.50
Output dim: 0, lower bound: -20.9577507, upper bound: 20.9549928
NS_B1_A1_A1_A1_B2_B1_B1_A1, status: Status.UNKNOWN, split count: 8, time: 2.50
Output dim: 0, lower bound: -20.9169596, upper bound: 20.9227605
NS_B1_A1_A1_A1_B2_B1_B1_A2, status: Status.UNKNOWN, split count: 8, time: 2.50
Output dim: 0, lower bound: -20.9169596, upper bound: 20.9430163
NS_B1_A1_A1_A1_B2_B1_B2_A1, status: Status.UNKNOWN, split count: 8, time: 2.50
Output dim: 0, lower bound: -20.9447177, upper bound: 20.9503799
NS_B1_A1_A1_A1_B2_B1_B2_A2, status: Status.UNKNOWN, split count: 8, time: 2.50
Output dim: 0, lower bound: -20.9463675, upper bound: 20.9515024
NS_B1_A1_A1_A1_B2_B2_B1_A1, status: Status.UNKNOWN, split count: 8, time: 2.50
Output dim: 0, lower bound: -20.8914581, upper bound: 20.8973311
NS_B1_A1_A1_A1_B2_B2_B1_A2, status: Status.UNKNOWN, split count: 8, time: 2.50
Output dim: 0, lower bound: -20.8914581, upper bound: 20.9286097
NS_B1_A1_A1_A1_B2_B2_B2_A1, status: Status.UNKNOWN, split count: 8, time: 2.50
Output dim: 0, lower bound: -20.9209893, upper bound: 20.9353029
NS_B1_A1_A1_A1_B2_B2_B2_A2, status: Status.UNKNOWN, split count: 8, time: 2.50
Output dim: 0, lower bound: -20.9212410, upper bound: 20.9354726
NS_B1_A1_A1_A2_B1_B1_A1_B1, status: Status.UNKNOWN, split count: 8, time: 2.50
Output dim: 0, lower bound: -20.9227605, upper bound: 20.9169596
NS_B1_A1_A1_A2_B1_B1_A1_B2, status: Status.UNKNOWN, split count: 8, time: 2.50
Output dim: 0, lower bound: -20.9227605, upper bound: 20.9328254
NS_B1_A1_A1_A2_B1_B1_A2_B1, status: Status.UNKNOWN, split count: 8, time: 2.50
Output dim: 0, lower bound: -20.9503799, upper bound: 20.9447177
NS_B1_A1_A1_A2_B1_B1_A2_B2, status: Status.UNKNOWN, split count: 8, time: 2.50
Output dim: 0, lower bound: -20.9515024, upper bound: 20.9463675
NS_B1_A1_A1_A2_B1_B2_A1_A1, status: Status.UNKNOWN, split count: 8, time: 2.50
Output dim: 0, lower bound: -20.9371414, upper bound: 20.9213108
NS_B1_A1_A1_A2_B1_B2_A1_A2, status: Status.UNKNOWN, split count: 8, time: 2.50
Output dim: 0, lower bound: -20.9371011, upper bound: 20.9209571
NS_B1_A1_A1_A2_B1_B2_A2_A1, status: Status.UNKNOWN, split count: 8, time: 2.50
Output dim: 0, lower bound: -20.9371414, upper bound: 20.9332980
NS_B1_A1_A1_A2_B1_B2_A2_A2, status: Status.UNKNOWN, split count: 8, time: 2.50
Output dim: 0, lower bound: -20.9442875, upper bound: 20.9332745
NS_B1_A1_A1_A2_B2_B2_B1_A1, status: Status.UNKNOWN, split count: 8, time: 2.50
Output dim: 0, lower bound: -20.8997218, upper bound: 20.9000373
NS_B1_A1_A1_A2_B2_B2_B1_A2, status: Status.UNKNOWN, split count: 8, time: 2.50
Output dim: 0, lower bound: -20.9083546, upper bound: 20.9152738
NS_B1_A2_A1_B1_B1_A1_B1_B1, status: Status.UNKNOWN, split count: 8, time: 2.50
Output dim: 0, lower bound: -20.9039416, upper bound: 20.9086279
NS_B1_A2_A1_B1_B1_A1_B1_B2, status: Status.UNKNOWN, split count: 8, time: 2.50
Output dim: 0, lower bound: -20.9039416, upper bound: 20.9297386
NS_B1_A2_A1_B1_B1_A1_B2_A1, status: Status.UNKNOWN, split count: 8, time: 2.50
Output dim: 0, lower bound: -20.9418692, upper bound: 20.9293078
NS_B1_A2_A1_B1_B1_A1_B2_A2, status: Status.UNKNOWN, split count: 8, time: 2.50
Output dim: 0, lower bound: -20.9418692, upper bound: 20.9293078
NS_B1_A2_A1_B1_B1_A2_B1_A1, status: Status.UNKNOWN, split count: 8, time: 2.50
Output dim: 0, lower bound: -20.9592875, upper bound: 20.9564825
NS_B1_A2_A1_B1_B1_A2_B1_A2, status: Status.UNKNOWN, split count: 8, time: 2.50
Output dim: 0, lower bound: -20.9592875, upper bound: 20.9570448
NS_B1_A2_A1_B1_B1_A2_B2_A1, status: Status.UNKNOWN, split count: 8, time: 2.50
Output dim: 0, lower bound: -20.9592875, upper bound: 20.9564825
NS_B1_A2_A1_B1_B1_A2_B2_A2, status: Status.UNKNOWN, split count: 8, time: 2.50
Output dim: 0, lower bound: -20.9592875, upper bound: 20.9570448
NS_B1_A2_A1_B1_B2_A1_B1_B1, status: Status.UNKNOWN, split count: 8, time: 2.50
Output dim: 0, lower bound: -20.9304753, upper bound: 20.9293804
NS_B1_A2_A1_B1_B2_A1_B1_B2, status: Status.UNKNOWN, split count: 8, time: 2.50
Output dim: 0, lower bound: -20.9283053, upper bound: 20.9291819
NS_B1_A2_A1_B1_B2_A1_B2_A1, status: Status.UNKNOWN, split count: 8, time: 2.50
Output dim: 0, lower bound: -20.9282818, upper bound: 20.9438823
NS_B1_A2_A1_B1_B2_A1_B2_A2, status: Status.UNKNOWN, split count: 8, time: 2.50
Output dim: 0, lower bound: -20.9283053, upper bound: 20.9437580
NS_B1_A2_A1_B1_B2_A2_B1_A1, status: Status.UNKNOWN, split count: 8, time: 2.50
Output dim: 0, lower bound: -20.9613564, upper bound: 20.9610719
NS_B1_A2_A1_B1_B2_A2_B1_A2, status: Status.UNKNOWN, split count: 8, time: 2.50
Output dim: 0, lower bound: -20.9613564, upper bound: 20.9621042
NS_B1_A2_A1_B1_B2_A2_B2_A1, status: Status.UNKNOWN, split count: 8, time: 2.50
Output dim: 0, lower bound: -20.9620715, upper bound: 20.9622880
NS_B1_A2_A1_B1_B2_A2_B2_A2, status: Status.UNKNOWN, split count: 8, time: 2.50
Output dim: 0, lower bound: -20.9620715, upper bound: 20.9633596
NS_B1_A2_A1_B2_B1_A1_B1_B1, status: Status.UNKNOWN, split count: 8, time: 2.50
Output dim: 0, lower bound: -20.9023832, upper bound: 20.9074667
NS_B1_A2_A1_B2_B1_A1_B1_B2, status: Status.UNKNOWN, split count: 8, time: 2.50
Output dim: 0, lower bound: -20.9023832, upper bound: 20.9074667
NS_B1_A2_A1_B2_B1_A1_B2_B1, status: Status.UNKNOWN, split count: 8, time: 2.50
Output dim: 0, lower bound: -20.9022185, upper bound: 20.9075448
NS_B1_A2_A1_B2_B1_A1_B2_B2, status: Status.UNKNOWN, split count: 8, time: 2.50
Output dim: 0, lower bound: -20.9022185, upper bound: 20.9075448
NS_B1_A2_A1_B2_B1_A2_A2_B1, status: Status.UNKNOWN, split count: 8, time: 2.50
Output dim: 0, lower bound: -20.9262506, upper bound: 20.9387399
NS_B1_A2_A1_B2_B1_A2_A2_B2, status: Status.UNKNOWN, split count: 8, time: 2.50
Output dim: 0, lower bound: -20.9262643, upper bound: 20.9387501
NS_B1_A2_A1_B2_B2_B1_A1_B1, status: Status.UNKNOWN, split count: 8, time: 2.50
Output dim: 0, lower bound: -20.9224177, upper bound: 20.9270178
NS_B1_A2_A1_B2_B2_B1_A1_B2, status: Status.UNKNOWN, split count: 8, time: 2.50
Output dim: 0, lower bound: -20.9022185, upper bound: 20.9269892
NS_B1_A2_A1_B2_B2_B1_A2_B1, status: Status.UNKNOWN, split count: 8, time: 2.50
Output dim: 0, lower bound: -20.9023869, upper bound: 20.9270178
NS_B1_A2_A1_B2_B2_B1_A2_B2, status: Status.UNKNOWN, split count: 8, time: 2.50
Output dim: 0, lower bound: -20.9216354, upper bound: 20.9433910
NS_B1_A2_A1_B2_B2_B2_B1_A1, status: Status.UNKNOWN, split count: 8, time: 2.50
Output dim: 0, lower bound: -20.9403245, upper bound: 20.9473847
NS_B1_A2_A1_B2_B2_B2_B1_A2, status: Status.UNKNOWN, split count: 8, time: 2.50
Output dim: 0, lower bound: -20.9403245, upper bound: 20.9484328
NS_B1_A2_A1_B2_B2_B2_B2_A1, status: Status.UNKNOWN, split count: 8, time: 2.50
Output dim: 0, lower bound: -20.9323331, upper bound: 20.9502207
NS_B1_A2_A1_B2_B2_B2_B2_A2, status: Status.UNKNOWN, split count: 8, time: 2.50
Output dim: 0, lower bound: -20.9450275, upper bound: 20.9513798
NS_B2_B1_A1_A2_A1_B1_A1_B1, status: Status.UNKNOWN, split count: 8, time: 2.50
Output dim: 0, lower bound: -20.9602904, upper bound: 20.9605021
NS_B2_B1_A1_A2_A1_B1_A1_B2, status: Status.UNKNOWN, split count: 8, time: 2.50
Output dim: 0, lower bound: -20.9602828, upper bound: 20.9605971
NS_B2_B1_A1_A2_A1_B1_A2_A1, status: Status.UNKNOWN, split count: 8, time: 2.50
Output dim: 0, lower bound: -20.9540109, upper bound: 20.9549650
NS_B2_B1_A1_A2_A1_B1_A2_A2, status: Status.UNKNOWN, split count: 8, time: 2.50
Output dim: 0, lower bound: -20.9535335, upper bound: 20.9549650
NS_B2_B1_A1_A2_A1_B2_A1_B1, status: Status.UNKNOWN, split count: 8, time: 2.50
Output dim: 0, lower bound: -20.9607707, upper bound: 20.9620561
NS_B2_B1_A1_A2_A1_B2_A1_B2, status: Status.UNKNOWN, split count: 8, time: 2.50
Output dim: 0, lower bound: -20.9607707, upper bound: 20.9620561
NS_B2_B1_A1_A2_A1_B2_A2_B1, status: Status.UNKNOWN, split count: 8, time: 2.50
Output dim: 0, lower bound: -20.9607707, upper bound: 20.9614392
NS_B2_B1_A1_A2_A1_B2_A2_B2, status: Status.UNKNOWN, split count: 8, time: 2.50
Output dim: 0, lower bound: -20.9607707, upper bound: 20.9614392
NS_B2_B1_A1_A2_A2_B1_A1_A1, status: Status.UNKNOWN, split count: 8, time: 2.50
Output dim: 0, lower bound: -20.9510084, upper bound: 20.9453079
NS_B2_B1_A1_A2_A2_B1_A1_A2, status: Status.UNKNOWN, split count: 8, time: 2.50
Output dim: 0, lower bound: -20.9610559, upper bound: 20.9607007
NS_B2_B1_A1_A2_A2_B1_A2_B1, status: Status.UNKNOWN, split count: 8, time: 2.50
Output dim: 0, lower bound: -20.9613692, upper bound: 20.9602891
NS_B2_B1_A1_A2_A2_B1_A2_B2, status: Status.UNKNOWN, split count: 8, time: 2.50
Output dim: 0, lower bound: -20.9607348, upper bound: 20.9602994
NS_B2_B1_A1_A2_A2_B2_A1_B1, status: Status.UNKNOWN, split count: 8, time: 2.50
Output dim: 0, lower bound: -20.9399659, upper bound: 20.9416250
NS_B2_B1_A1_A2_A2_B2_A1_B2, status: Status.UNKNOWN, split count: 8, time: 2.50
Output dim: 0, lower bound: -20.9602917, upper bound: 20.9606105
NS_B2_B1_A1_A2_A2_B2_A2_B1, status: Status.UNKNOWN, split count: 8, time: 2.50
Output dim: 0, lower bound: -20.9607285, upper bound: 20.9603972
NS_B2_B1_A1_A2_A2_B2_A2_B2, status: Status.UNKNOWN, split count: 8, time: 2.50
Output dim: 0, lower bound: -20.9607285, upper bound: 20.9608904
NS_B2_B1_A2_B2_A1_B1_A1_B1, status: Status.UNKNOWN, split count: 8, time: 2.50
Output dim: 0, lower bound: -20.9619006, upper bound: 20.9624504
NS_B2_B1_A2_B2_A1_B1_A1_B2, status: Status.UNKNOWN, split count: 8, time: 2.50
Output dim: 0, lower bound: -20.9619006, upper bound: 20.9624504
NS_B2_B1_A2_B2_A1_B1_A2_B1, status: Status.UNKNOWN, split count: 8, time: 2.50
Output dim: 0, lower bound: -20.9619006, upper bound: 20.9624504
NS_B2_B1_A2_B2_A1_B1_A2_B2, status: Status.UNKNOWN, split count: 8, time: 2.50
Output dim: 0, lower bound: -20.9619006, upper bound: 20.9624504
NS_B2_B1_A2_B2_A1_B2_A1_A1, status: Status.UNKNOWN, split count: 8, time: 2.50
Output dim: 0, lower bound: -20.9529447, upper bound: 20.9563306
NS_B2_B1_A2_B2_A1_B2_A1_A2, status: Status.UNKNOWN, split count: 8, time: 2.50
Output dim: 0, lower bound: -20.9513929, upper bound: 20.9539752
NS_B2_B1_A2_B2_A1_B2_A2_B1, status: Status.UNKNOWN, split count: 8, time: 2.50
Output dim: 0, lower bound: -20.9579247, upper bound: 20.9604336
NS_B2_B1_A2_B2_A1_B2_A2_B2, status: Status.UNKNOWN, split count: 8, time: 2.50
Output dim: 0, lower bound: -20.9542086, upper bound: 20.9570735
NS_B2_B1_A2_B2_A2_A2_B1_B1, status: Status.VERIFIED, split count: 8, time: 2.50
Output dim: 0, lower bound: -20.7041063, upper bound: 20.6868226
NS_B2_B1_A2_B2_A2_A2_B1_B2, status: Status.VERIFIED, split count: 8, time: 2.50
Output dim: 0, lower bound: -20.6625303, upper bound: 20.6605575
NS_B2_B1_A2_B2_A2_A2_B2_A1, status: Status.UNKNOWN, split count: 8, time: 2.50
Output dim: 0, lower bound: -20.9503003, upper bound: 20.9499824
NS_B2_B1_A2_B2_A2_A2_B2_A2, status: Status.UNKNOWN, split count: 8, time: 2.50
Output dim: 0, lower bound: -20.9503003, upper bound: 20.9502923
NS_B2_B2_A1_B1_A1_B1_B1_A1, status: Status.UNKNOWN, split count: 8, time: 2.50
Output dim: 0, lower bound: -20.9300851, upper bound: 20.9287486
NS_B2_B2_A1_B1_A1_B1_B1_A2, status: Status.UNKNOWN, split count: 8, time: 2.50
Output dim: 0, lower bound: -20.9300851, upper bound: 20.9500642
NS_B2_B2_A1_B1_A1_B1_B2_A1, status: Status.UNKNOWN, split count: 8, time: 2.50
Output dim: 0, lower bound: -20.9615888, upper bound: 20.9616959
NS_B2_B2_A1_B1_A1_B1_B2_A2, status: Status.UNKNOWN, split count: 8, time: 2.50
Output dim: 0, lower bound: -20.9615888, upper bound: 20.9625913
NS_B2_B2_A1_B1_A1_B2_B1_A1, status: Status.UNKNOWN, split count: 8, time: 2.50
Output dim: 0, lower bound: -20.9298018, upper bound: 20.9287663
NS_B2_B2_A1_B1_A1_B2_B1_A2, status: Status.UNKNOWN, split count: 8, time: 2.50
Output dim: 0, lower bound: -20.9298018, upper bound: 20.9503509
NS_B2_B2_A1_B1_A1_B2_B2_A1, status: Status.UNKNOWN, split count: 8, time: 2.50
Output dim: 0, lower bound: -20.9548427, upper bound: 20.9551555
NS_B2_B2_A1_B1_A1_B2_B2_A2, status: Status.UNKNOWN, split count: 8, time: 2.50
Output dim: 0, lower bound: -20.9640653, upper bound: 20.9646005
NS_B2_B2_A1_B1_A2_A1_B1_B1, status: Status.UNKNOWN, split count: 8, time: 2.50
Output dim: 0, lower bound: -20.9081548, upper bound: 20.9023583
NS_B2_B2_A1_B1_A2_A1_B1_B2, status: Status.UNKNOWN, split count: 8, time: 2.50
Output dim: 0, lower bound: -20.9381136, upper bound: 20.9253107
NS_B2_B2_A1_B1_A2_A1_B2_B1, status: Status.UNKNOWN, split count: 8, time: 2.50
Output dim: 0, lower bound: -20.9075448, upper bound: 20.9023529
NS_B2_B2_A1_B1_A2_A1_B2_B2, status: Status.UNKNOWN, split count: 8, time: 2.50
Output dim: 0, lower bound: -20.9387501, upper bound: 20.9262643
NS_B2_B2_A1_B1_A2_A2_B1_A1, status: Status.UNKNOWN, split count: 8, time: 2.50
Output dim: 0, lower bound: -20.9430765, upper bound: 20.9323331
NS_B2_B2_A1_B1_A2_A2_B1_A2, status: Status.UNKNOWN, split count: 8, time: 2.50
Output dim: 0, lower bound: -20.9430765, upper bound: 20.9450275
NS_B2_B2_A1_B1_A2_A2_B2_B1, status: Status.UNKNOWN, split count: 8, time: 2.50
Output dim: 0, lower bound: -20.9265513, upper bound: 20.9228258
NS_B2_B2_A1_B1_A2_A2_B2_B2, status: Status.UNKNOWN, split count: 8, time: 2.50
Output dim: 0, lower bound: -20.9513798, upper bound: 20.9472191
NS_B2_B2_A1_B2_A1_B1_B1_A1, status: Status.UNKNOWN, split count: 8, time: 2.50
Output dim: 0, lower bound: -20.9292851, upper bound: 20.9261202
NS_B2_B2_A1_B2_A1_B1_B1_A2, status: Status.UNKNOWN, split count: 8, time: 2.50
Output dim: 0, lower bound: -20.9292851, upper bound: 20.9503422
NS_B2_B2_A1_B2_A1_B1_B2_A1, status: Status.UNKNOWN, split count: 8, time: 2.50
Output dim: 0, lower bound: -20.9407029, upper bound: 20.9511217
NS_B2_B2_A1_B2_A1_B1_B2_A2, status: Status.UNKNOWN, split count: 8, time: 2.50
Output dim: 0, lower bound: -20.9460557, upper bound: 20.9511217
NS_B2_B2_A1_B2_A1_B2_B1_A1, status: Status.UNKNOWN, split count: 8, time: 2.50
Output dim: 0, lower bound: -20.9227677, upper bound: 20.9353666
NS_B2_B2_A1_B2_A1_B2_B1_A2, status: Status.UNKNOWN, split count: 8, time: 2.50
Output dim: 0, lower bound: -20.9449710, upper bound: 20.9514500
NS_B2_B2_A1_B2_A1_B2_B2_A1, status: Status.UNKNOWN, split count: 8, time: 2.50
Output dim: 0, lower bound: -20.9440242, upper bound: 20.9476770
NS_B2_B2_A1_B2_A1_B2_B2_A2, status: Status.UNKNOWN, split count: 8, time: 2.50
Output dim: 0, lower bound: -20.9523718, upper bound: 20.9564293
NS_B2_B2_A1_B2_A2_B1_B1_A1, status: Status.UNKNOWN, split count: 8, time: 2.50
Output dim: 0, lower bound: -20.9155543, upper bound: 20.9219504
NS_B2_B2_A1_B2_A2_B1_B1_A2, status: Status.UNKNOWN, split count: 8, time: 2.50
Output dim: 0, lower bound: -20.9273829, upper bound: 20.9235466
NS_B2_B2_A1_B2_A2_B1_B2_A1, status: Status.UNKNOWN, split count: 8, time: 2.50
Output dim: 0, lower bound: -20.9066594, upper bound: 20.9106975
NS_B2_B2_A1_B2_A2_B1_B2_A2, status: Status.UNKNOWN, split count: 8, time: 2.50
Output dim: 0, lower bound: -20.9313827, upper bound: 20.9330553
NS_B2_B2_A1_B2_A2_B2_B1_A1, status: Status.UNKNOWN, split count: 8, time: 2.50
Output dim: 0, lower bound: -20.9147679, upper bound: 20.9221320
NS_B2_B2_A1_B2_A2_B2_B1_A2, status: Status.UNKNOWN, split count: 8, time: 2.50
Output dim: 0, lower bound: -20.9272313, upper bound: 20.9235597
NS_B2_B2_A1_B2_A2_B2_B2_A1, status: Status.UNKNOWN, split count: 8, time: 2.50
Output dim: 0, lower bound: -20.9290999, upper bound: 20.9170359
NS_B2_B2_A1_B2_A2_B2_B2_A2, status: Status.UNKNOWN, split count: 8, time: 2.50
Output dim: 0, lower bound: -20.9404709, upper bound: 20.9393947
NS_B2_B2_A2_B1_A1_A1_B1_A1, status: Status.UNKNOWN, split count: 8, time: 2.50
Output dim: 0, lower bound: -20.9610804, upper bound: 20.9591284
NS_B2_B2_A2_B1_A1_A1_B1_A2, status: Status.UNKNOWN, split count: 8, time: 2.50
Output dim: 0, lower bound: -20.9637382, upper bound: 20.9626796
NS_B2_B2_A2_B1_A1_A1_B2_A1, status: Status.UNKNOWN, split count: 8, time: 2.50
Output dim: 0, lower bound: -20.9635723, upper bound: 20.9630608
NS_B2_B2_A2_B1_A1_A1_B2_A2, status: Status.UNKNOWN, split count: 8, time: 2.50
Output dim: 0, lower bound: -20.9637382, upper bound: 20.9644216
NS_B2_B2_A2_B1_A1_A2_B2_B1, status: Status.UNKNOWN, split count: 8, time: 2.50
Output dim: 0, lower bound: -20.9535715, upper bound: 20.9507958
NS_B2_B2_A2_B1_A1_A2_B2_B2, status: Status.UNKNOWN, split count: 8, time: 2.50
Output dim: 0, lower bound: -20.9610848, upper bound: 20.9582666
NS_B2_B2_A2_B1_A2_A1_A1_B1, status: Status.UNKNOWN, split count: 8, time: 2.50
Output dim: 0, lower bound: -20.9543912, upper bound: 20.9543120
NS_B2_B2_A2_B1_A2_A1_A1_B2, status: Status.UNKNOWN, split count: 8, time: 2.50
Output dim: 0, lower bound: -20.9521551, upper bound: 20.9561758
NS_B2_B2_A2_B1_A2_A1_A2_B1, status: Status.UNKNOWN, split count: 8, time: 2.50
Output dim: 0, lower bound: -20.9540550, upper bound: 20.9569058
NS_B2_B2_A2_B1_A2_A1_A2_B2, status: Status.UNKNOWN, split count: 8, time: 2.50
Output dim: 0, lower bound: -20.9540550, upper bound: 20.9644216
NS_B2_B2_A2_B1_A2_A2_B1_A1, status: Status.UNKNOWN, split count: 8, time: 2.50
Output dim: 0, lower bound: -20.9433058, upper bound: 20.9381379
NS_B2_B2_A2_B1_A2_A2_B1_A2, status: Status.UNKNOWN, split count: 8, time: 2.50
Output dim: 0, lower bound: -20.9491533, upper bound: 20.9464072
NS_B2_B2_A2_B1_A2_A2_B2_A1, status: Status.UNKNOWN, split count: 8, time: 2.50
Output dim: 0, lower bound: -20.9481945, upper bound: 20.9453032
NS_B2_B2_A2_B1_A2_A2_B2_A2, status: Status.UNKNOWN, split count: 8, time: 2.50
Output dim: 0, lower bound: -20.9481945, upper bound: 20.9544784

## BFS NS instance: NS_B1_A1_A1_A1_B1_B1_A1_B1

### Backsubstitution after applying NS history:
0: -6.0033531, 16.7872543, -6.0033531, 16.7872543, -22.7906036, 22.7906036
1: -15.1282873, 25.5919876, -15.1282873, 25.5919876, -40.7202682, 40.7202682
2: -10.4942646, 23.2998066, -10.4942646, 23.2998066, -33.7940712, 33.7940712
3: -16.2143669, 28.3640823, -16.2143669, 28.3640823, -44.5784454, 44.5784454
4: -14.6194534, 28.8417206, -14.6194534, 28.8417206, -43.4611740, 43.4611740

Time for backsubstitution: 0.93 seconds

### NS candidates at layer 1
type: B, layer: 1, pos: 12
type: A, layer: 1, pos: 12
type: A, layer: 1, pos: 15
type: B, layer: 1, pos: 15
type: A, layer: 1, pos: 28
type: B, layer: 1, pos: 28
type: B, layer: 1, pos: 6
type: A, layer: 1, pos: 6
type: B, layer: 1, pos: 24
type: A, layer: 1, pos: 24
type: B, layer: 1, pos: 37
type: A, layer: 1, pos: 37
type: A, layer: 1, pos: 8
type: B, layer: 1, pos: 8
type: B, layer: 1, pos: 39
type: A, layer: 1, pos: 39
type: A, layer: 1, pos: 38
type: B, layer: 1, pos: 38
type: A, layer: 1, pos: 48
type: B, layer: 1, pos: 48
type: B, layer: 1, pos: 20
type: A, layer: 1, pos: 20
type: A, layer: 1, pos: 26
type: B, layer: 1, pos: 26
type: A, layer: 1, pos: 41
type: B, layer: 1, pos: 41
type: A, layer: 1, pos: 0
type: B, layer: 1, pos: 0
type: B, layer: 1, pos: 5
type: A, layer: 1, pos: 5
type: A, layer: 1, pos: 31
type: B, layer: 1, pos: 31

Time for candidate selection: 0.07 seconds

### Candidate
type: B, layer: 1, pos: 12

### Candidate
type: A, layer: 1, pos: 12

### Candidate
type: A, layer: 1, pos: 15

### Candidate
type: B, layer: 1, pos: 15

### Candidate
type: A, layer: 1, pos: 28

### Candidate
type: B, layer: 1, pos: 28

### Candidate
type: B, layer: 1, pos: 6

### Candidate
type: A, layer: 1, pos: 6

### Candidate
type: B, layer: 1, pos: 24

### Candidate
type: A, layer: 1, pos: 24

### Candidate
type: B, layer: 1, pos: 37

## Relational analysis of NS_B1_A1_A1_A1_B1_B1_A1_B1_B1
Optimization infeasible because this subproblem isn't reachable.

### Candidate
type: A, layer: 1, pos: 37

### Candidate
type: A, layer: 1, pos: 8

## Relational analysis of NS_B1_A1_A1_A1_B1_B1_A1_B1_A1

### Relational analysis result of NS_B1_A1_A1_A1_B1_B1_A1_B1_A1
Status: Status.UNKNOWN
Output dim: 0, lower bound: -20.9234888, upper bound: 20.9254870
time: 1.04 seconds

## Relational analysis of NS_B1_A1_A1_A1_B1_B1_A1_B1_A2

### Relational analysis result of NS_B1_A1_A1_A1_B1_B1_A1_B1_A2
Status: Status.UNKNOWN
Output dim: 0, lower bound: -20.9234560, upper bound: 20.9234560
time: 0.54 seconds

## BFS NS instance: NS_B1_A1_A1_A1_B1_B1_A1_B2

### Backsubstitution after applying NS history:
0: -6.0033531, 16.7872543, -6.0468984, 16.8671551, -22.8705082, 22.8341465
1: -15.1282873, 25.5919876, -15.3067408, 25.6188679, -40.7471542, 40.8987274
2: -10.4942646, 23.2998066, -10.6000986, 23.3775158, -33.8717804, 33.8999062
3: -16.2143669, 28.3640823, -16.3600082, 28.4442596, -44.6586266, 44.7240829
4: -14.6194534, 28.8417206, -14.6819715, 28.9659004, -43.5853539, 43.5236816

Time for backsubstitution: 0.91 seconds

### NS candidates at layer 1
type: A, layer: 1, pos: 12
type: B, layer: 1, pos: 12
type: A, layer: 1, pos: 15
type: B, layer: 1, pos: 15
type: B, layer: 1, pos: 28
type: A, layer: 1, pos: 28
type: B, layer: 1, pos: 6
type: A, layer: 1, pos: 6
type: B, layer: 1, pos: 24
type: A, layer: 1, pos: 24
type: B, layer: 1, pos: 37
type: A, layer: 1, pos: 8
type: A, layer: 1, pos: 37
type: B, layer: 1, pos: 8
type: A, layer: 1, pos: 39
type: B, layer: 1, pos: 39
type: B, layer: 1, pos: 38
type: A, layer: 1, pos: 38
type: A, layer: 1, pos: 48
type: B, layer: 1, pos: 48
type: B, layer: 1, pos: 20
type: A, layer: 1, pos: 20
type: A, layer: 1, pos: 26
type: B, layer: 1, pos: 26
type: A, layer: 1, pos: 41
type: B, layer: 1, pos: 41
type: B, layer: 1, pos: 5
type: B, layer: 1, pos: 0
type: A, layer: 1, pos: 0
type: A, layer: 1, pos: 5
type: A, layer: 1, pos: 31
type: B, layer: 1, pos: 31

Time for candidate selection: 0.07 seconds

### Candidate
type: A, layer: 1, pos: 12

### Candidate
type: B, layer: 1, pos: 12

### Candidate
type: A, layer: 1, pos: 15

### Candidate
type: B, layer: 1, pos: 15

### Candidate
type: B, layer: 1, pos: 28

### Candidate
type: A, layer: 1, pos: 28

### Candidate
type: B, layer: 1, pos: 6

### Candidate
type: A, layer: 1, pos: 6

### Candidate
type: B, layer: 1, pos: 24

### Candidate
type: A, layer: 1, pos: 24

### Candidate
type: B, layer: 1, pos: 37

## Relational analysis of NS_B1_A1_A1_A1_B1_B1_A1_B2_B1
Optimization infeasible because this subproblem isn't reachable.

### Candidate
type: A, layer: 1, pos: 8

## Relational analysis of NS_B1_A1_A1_A1_B1_B1_A1_B2_A1

### Relational analysis result of NS_B1_A1_A1_A1_B1_B1_A1_B2_A1
Status: Status.UNKNOWN
Output dim: 0, lower bound: -20.9234888, upper bound: 20.9395926
time: 0.55 seconds

## Relational analysis of NS_B1_A1_A1_A1_B1_B1_A1_B2_A2

### Relational analysis result of NS_B1_A1_A1_A1_B1_B1_A1_B2_A2
Status: Status.UNKNOWN
Output dim: 0, lower bound: -20.9234560, upper bound: 20.9375250
time: 0.62 seconds

## BFS NS instance: NS_B1_A1_A1_A1_B1_B1_A2_B1

### Backsubstitution after applying NS history:
0: -5.6577873, 15.8578768, -5.3533392, 15.0835075, -20.7412949, 21.2112160
1: -14.3030930, 24.1545410, -13.4276791, 23.1441307, -37.4472160, 37.5822182
2: -9.9010229, 22.0157089, -9.3034277, 20.9727650, -30.8737869, 31.3191376
3: -15.3068848, 26.7919388, -14.4344606, 25.6085052, -40.9153862, 41.2263985
4: -13.7272148, 27.2735519, -13.0051641, 25.9115906, -39.6387978, 40.2787132

Time for backsubstitution: 0.91 seconds

### NS candidates at layer 1
type: A, layer: 1, pos: 12
type: B, layer: 1, pos: 12
type: B, layer: 1, pos: 28
type: B, layer: 1, pos: 15
type: A, layer: 1, pos: 15
type: A, layer: 1, pos: 28
type: B, layer: 1, pos: 24
type: B, layer: 1, pos: 6
type: A, layer: 1, pos: 39
type: B, layer: 1, pos: 39
type: A, layer: 1, pos: 24
type: B, layer: 1, pos: 20
type: A, layer: 1, pos: 8
type: B, layer: 1, pos: 9
type: B, layer: 1, pos: 37
type: A, layer: 1, pos: 37
type: A, layer: 1, pos: 20
type: A, layer: 1, pos: 6
type: B, layer: 1, pos: 48
type: B, layer: 1, pos: 38
type: A, layer: 1, pos: 38
type: A, layer: 1, pos: 48
type: A, layer: 1, pos: 26
type: B, layer: 1, pos: 26
type: B, layer: 1, pos: 41
type: A, layer: 1, pos: 41
type: B, layer: 1, pos: 31
type: B, layer: 1, pos: 0
type: A, layer: 1, pos: 31
type: A, layer: 1, pos: 5
type: A, layer: 1, pos: 0
type: B, layer: 1, pos: 5

Time for candidate selection: 0.07 seconds

### Candidate
type: A, layer: 1, pos: 12

### Candidate
type: B, layer: 1, pos: 12

### Candidate
type: B, layer: 1, pos: 28

### Candidate
type: B, layer: 1, pos: 15

### Candidate
type: A, layer: 1, pos: 15

### Candidate
type: A, layer: 1, pos: 28

### Candidate
type: B, layer: 1, pos: 24

### Candidate
type: B, layer: 1, pos: 6

### Candidate
type: A, layer: 1, pos: 39

## Relational analysis of NS_B1_A1_A1_A1_B1_B1_A2_B1_A1
Optimization infeasible because this subproblem isn't reachable.

### Candidate
type: B, layer: 1, pos: 39

### Candidate
type: A, layer: 1, pos: 24

### Candidate
type: B, layer: 1, pos: 20

## Relational analysis of NS_B1_A1_A1_A1_B1_B1_A2_B1_B1
Optimization infeasible because this subproblem isn't reachable.

### Candidate
type: A, layer: 1, pos: 8

## Relational analysis of NS_B1_A1_A1_A1_B1_B1_A2_B1_A1

### Relational analysis result of NS_B1_A1_A1_A1_B1_B1_A2_B1_A1
Status: Status.UNKNOWN
Output dim: 0, lower bound: -20.9608807, upper bound: 20.9608807
time: 0.58 seconds

## Relational analysis of NS_B1_A1_A1_A1_B1_B1_A2_B1_A2

### Relational analysis result of NS_B1_A1_A1_A1_B1_B1_A2_B1_A2
Status: Status.UNKNOWN
Output dim: 0, lower bound: -20.9445812, upper bound: 20.9619725
time: 0.66 seconds

## BFS NS instance: NS_B1_A1_A1_A1_B1_B1_A2_B2

### Backsubstitution after applying NS history:
0: -6.0468984, 16.8671551, -5.9941044, 16.7504959, -22.7973900, 22.8612595
1: -15.3067408, 25.6188679, -15.1059580, 25.5340748, -40.8408127, 40.7248268
2: -10.6000986, 23.3775158, -10.4830847, 23.2524834, -33.8525810, 33.8605957
3: -16.3600082, 28.4442596, -16.1889801, 28.3036919, -44.6637001, 44.6332397
4: -14.6819715, 28.9659004, -14.6044674, 28.7794151, -43.4613800, 43.5703659

Time for backsubstitution: 0.91 seconds

### NS candidates at layer 1
type: B, layer: 1, pos: 12
type: A, layer: 1, pos: 12
type: B, layer: 1, pos: 15
type: A, layer: 1, pos: 15
type: B, layer: 1, pos: 28
type: A, layer: 1, pos: 28
type: A, layer: 1, pos: 6
type: B, layer: 1, pos: 6
type: B, layer: 1, pos: 24
type: A, layer: 1, pos: 24
type: B, layer: 1, pos: 37
type: A, layer: 1, pos: 37
type: A, layer: 1, pos: 8
type: A, layer: 1, pos: 39
type: B, layer: 1, pos: 38
type: B, layer: 1, pos: 48
type: A, layer: 1, pos: 38
type: B, layer: 1, pos: 39
type: B, layer: 1, pos: 9
type: A, layer: 1, pos: 48
type: B, layer: 1, pos: 20
type: A, layer: 1, pos: 20
type: A, layer: 1, pos: 26
type: B, layer: 1, pos: 26
type: B, layer: 1, pos: 41
type: A, layer: 1, pos: 41
type: B, layer: 1, pos: 0
type: A, layer: 1, pos: 5
type: A, layer: 1, pos: 0
type: B, layer: 1, pos: 5
type: A, layer: 1, pos: 31
type: B, layer: 1, pos: 31

Time for candidate selection: 0.07 seconds

### Candidate
type: B, layer: 1, pos: 12

### Candidate
type: A, layer: 1, pos: 12

### Candidate
type: B, layer: 1, pos: 15

### Candidate
type: A, layer: 1, pos: 15

### Candidate
type: B, layer: 1, pos: 28

### Candidate
type: A, layer: 1, pos: 28

### Candidate
type: A, layer: 1, pos: 6

### Candidate
type: B, layer: 1, pos: 6

### Candidate
type: B, layer: 1, pos: 24

### Candidate
type: A, layer: 1, pos: 24

### Candidate
type: B, layer: 1, pos: 37

### Candidate
type: A, layer: 1, pos: 37

### Candidate
type: A, layer: 1, pos: 8

## Relational analysis of NS_B1_A1_A1_A1_B1_B1_A2_B2_A1

### Relational analysis result of NS_B1_A1_A1_A1_B1_B1_A2_B2_A1
Status: Status.UNKNOWN
Output dim: 0, lower bound: -20.9619725, upper bound: 20.9624935
time: 0.60 seconds

## Relational analysis of NS_B1_A1_A1_A1_B1_B1_A2_B2_A2

### Relational analysis result of NS_B1_A1_A1_A1_B1_B1_A2_B2_A2
Status: Status.UNKNOWN
Output dim: 0, lower bound: -20.9619725, upper bound: 20.9636160
time: 0.55 seconds

## BFS NS instance: NS_B1_A1_A1_A1_B1_B2_A1_B1

### Backsubstitution after applying NS history:
0: -5.4051991, 15.2213621, -6.1071935, 17.1215572, -22.5267563, 21.3285561
1: -13.5574827, 23.3508892, -15.2758675, 26.1317062, -39.6891899, 38.6267548
2: -9.3942366, 21.1592789, -10.6721554, 23.6951408, -33.0893784, 31.8314323
3: -14.5722885, 25.8379803, -16.4547119, 28.9661388, -43.5384216, 42.2926865
4: -13.1313391, 26.1407356, -14.9335165, 29.2816982, -42.4130325, 41.0742493

Time for backsubstitution: 0.93 seconds

### NS candidates at layer 1
type: B, layer: 1, pos: 12
type: A, layer: 1, pos: 12
type: A, layer: 1, pos: 15
type: B, layer: 1, pos: 15
type: B, layer: 1, pos: 28
type: B, layer: 1, pos: 6
type: B, layer: 1, pos: 9
type: A, layer: 1, pos: 28
type: A, layer: 1, pos: 9
type: B, layer: 1, pos: 24
type: B, layer: 1, pos: 39
type: A, layer: 1, pos: 37
type: A, layer: 1, pos: 24
type: B, layer: 1, pos: 37
type: A, layer: 1, pos: 20
type: A, layer: 1, pos: 38
type: B, layer: 1, pos: 20
type: B, layer: 1, pos: 38
type: A, layer: 1, pos: 48
type: A, layer: 1, pos: 39
type: B, layer: 1, pos: 48
type: A, layer: 1, pos: 6
type: A, layer: 1, pos: 26
type: B, layer: 1, pos: 26
type: B, layer: 1, pos: 31
type: A, layer: 1, pos: 41
type: B, layer: 1, pos: 41
type: A, layer: 1, pos: 31
type: B, layer: 1, pos: 0
type: A, layer: 1, pos: 0
type: B, layer: 1, pos: 5
type: A, layer: 1, pos: 5

Time for candidate selection: 0.07 seconds

### Candidate
type: B, layer: 1, pos: 12

### Candidate
type: A, layer: 1, pos: 12

### Candidate
type: A, layer: 1, pos: 15

### Candidate
type: B, layer: 1, pos: 15

### Candidate
type: B, layer: 1, pos: 28

### Candidate
type: B, layer: 1, pos: 6

### Candidate
type: B, layer: 1, pos: 9

### Candidate
type: A, layer: 1, pos: 28

### Candidate
type: A, layer: 1, pos: 9

### Candidate
type: B, layer: 1, pos: 24

### Candidate
type: B, layer: 1, pos: 39

### Candidate
type: A, layer: 1, pos: 37

### Candidate
type: A, layer: 1, pos: 24

### Candidate
type: B, layer: 1, pos: 37

### Candidate
type: A, layer: 1, pos: 20

### Candidate
type: A, layer: 1, pos: 38

## Relational analysis of NS_B1_A1_A1_A1_B1_B2_A1_B1_A1
Optimization infeasible because this subproblem isn't reachable.

### Candidate
type: B, layer: 1, pos: 20

### Candidate
type: B, layer: 1, pos: 38

### Candidate
type: A, layer: 1, pos: 48

## Relational analysis of NS_B1_A1_A1_A1_B1_B2_A1_B1_A1
Optimization infeasible because this subproblem isn't reachable.

### Candidate
type: A, layer: 1, pos: 39

### Candidate
type: B, layer: 1, pos: 48

### Candidate
type: A, layer: 1, pos: 6

### Candidate
type: A, layer: 1, pos: 26

## Relational analysis of NS_B1_A1_A1_A1_B1_B2_A1_B1_A1
Optimization infeasible because this subproblem isn't reachable.

### Candidate
type: B, layer: 1, pos: 26

### Candidate
type: B, layer: 1, pos: 31

## Relational analysis of NS_B1_A1_A1_A1_B1_B2_A1_B1_B1
Optimization infeasible because this subproblem isn't reachable.

### Candidate
type: A, layer: 1, pos: 41

## Relational analysis of NS_B1_A1_A1_A1_B1_B2_A1_B1_A1
Optimization infeasible because this subproblem isn't reachable.

### Candidate
type: B, layer: 1, pos: 41

### Candidate
type: A, layer: 1, pos: 31

### Candidate
type: B, layer: 1, pos: 0

## Relational analysis of NS_B1_A1_A1_A1_B1_B2_A1_B1_B1
Optimization infeasible because this subproblem isn't reachable.

### Candidate
type: A, layer: 1, pos: 0

### Candidate
type: B, layer: 1, pos: 5

## Relational analysis of NS_B1_A1_A1_A1_B1_B2_A1_B1_B1

### Relational analysis result of NS_B1_A1_A1_A1_B1_B2_A1_B1_B1
Status: Status.UNKNOWN
Output dim: 0, lower bound: -20.9556989, upper bound: 20.9539958
time: 0.56 seconds

## Relational analysis of NS_B1_A1_A1_A1_B1_B2_A1_B1_B2

### Relational analysis result of NS_B1_A1_A1_A1_B1_B2_A1_B1_B2
Status: Status.UNKNOWN
Output dim: 0, lower bound: -20.9556989, upper bound: 20.9546921
time: 0.55 seconds

## NS Result
status: Status.UNKNOWN
execution time: (base) + (ns) = 2.48 + 417.81 = 420.30 seconds
