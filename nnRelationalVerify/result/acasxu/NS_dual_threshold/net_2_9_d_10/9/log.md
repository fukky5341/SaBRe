## Execution arguments:
Dataset: Dataset.ACAS
Network: ds/onnx/acasxu_op11/ACASXU_2_9.onnx
Relational property: GLOBAL_ROBUSTNESS
LP Analysis: True
Epsilon: None
Delta epsilon: 0.1
execution index: (10, None, 9)
Time budget: 420 seconds
Split limit: 100
Threshold: 912.9840697103999


## IAR start

### BASE IAR bounds
Layer (inp1_lb, inp1_ub, inp2_lb, inp2_ub, d_lb, d_ub)
0: (-631.4873047, 411.7176819, -631.4873047, 411.7176819, -1043.2049561, 1043.2049561)
1: (-47.2466049, 36.8136635, -47.2466049, 36.8136635, -84.0602646, 84.0602646)
2: (-38.6318474, 55.3112831, -38.6318474, 55.3112831, -93.9431152, 93.9431152)
3: (-44.2114601, 87.6876907, -44.2114601, 87.6876907, -131.8991547, 131.8991547)
4: (-33.6581497, 55.1203232, -33.6581497, 55.1203232, -88.7784729, 88.7784729)

## BASE Relational Analysis

## BASE Result
execution time: IAR + RelationalAnalysis = 3.03 + 1.53 = 4.56 seconds
status: Status.UNKNOWN
relational distance
Output dim: 0, lower bound: -916.6506724, upper bound: 916.6506724

# Neuron Split (NS) starts

## BFS NS instance: NS

Time for backsubstitution: 0.01 seconds

### NS candidates at layer 1
type: A, layer: 1, pos: 0
type: B, layer: 1, pos: 0
type: A, layer: 1, pos: 8
type: B, layer: 1, pos: 8
type: A, layer: 1, pos: 5
type: B, layer: 1, pos: 5
type: A, layer: 1, pos: 35
type: B, layer: 1, pos: 35
type: A, layer: 1, pos: 38
type: B, layer: 1, pos: 38
type: A, layer: 1, pos: 18
type: B, layer: 1, pos: 18
type: A, layer: 1, pos: 47
type: B, layer: 1, pos: 47
type: A, layer: 1, pos: 10
type: B, layer: 1, pos: 10
type: A, layer: 1, pos: 3
type: B, layer: 1, pos: 3
type: A, layer: 1, pos: 21
type: B, layer: 1, pos: 21
type: B, layer: 1, pos: 30
type: A, layer: 1, pos: 30
type: A, layer: 1, pos: 24
type: B, layer: 1, pos: 24
type: A, layer: 1, pos: 15
type: B, layer: 1, pos: 15
type: A, layer: 1, pos: 1
type: B, layer: 1, pos: 1
type: A, layer: 1, pos: 14
type: B, layer: 1, pos: 14
type: A, layer: 1, pos: 40
type: B, layer: 1, pos: 40

Time for candidate selection: 0.23 seconds

### Candidate
type: A, layer: 1, pos: 0

## Relational analysis of NS_A1

### Relational analysis result of NS_A1
Status: Status.UNKNOWN
Output dim: 0, lower bound: -916.3384520, upper bound: 914.9857326
time: 0.46 seconds

## Relational analysis of NS_A2

### Relational analysis result of NS_A2
Status: Status.UNKNOWN
Output dim: 0, lower bound: -916.6168118, upper bound: 916.6168114
time: 0.46 seconds

## Summary of splitting at layer (split count: 0)
- Time for NS candidates: 1.17 seconds
NS_A1, status: Status.UNKNOWN, split count: 1, time: 1.17
Output dim: 0, lower bound: -916.3384520, upper bound: 914.9857326
NS_A2, status: Status.UNKNOWN, split count: 1, time: 1.17
Output dim: 0, lower bound: -916.6168118, upper bound: 916.6168114

## BFS NS instance: NS_A1

### Backsubstitution after applying NS history:
0: -511.7307129, 333.6546936, -588.4508057, 386.1644287, -897.8951416, 922.1054688
1: -38.9493980, 29.6902771, -44.5147667, 34.2707787, -73.2201767, 74.2050247
2: -32.5045624, 44.4840584, -36.5841789, 51.7268448, -84.2314072, 81.0682373
3: -36.7355003, 70.9758224, -41.7016068, 82.0571289, -118.7926025, 112.6774216
4: -27.8541012, 44.5239906, -31.7192993, 51.6075249, -79.4616241, 76.2432861

Time for backsubstitution: 2.79 seconds

### NS candidates at layer 1
type: B, layer: 1, pos: 0
type: B, layer: 1, pos: 8
type: A, layer: 1, pos: 8
type: B, layer: 1, pos: 18
type: A, layer: 1, pos: 18
type: B, layer: 1, pos: 38
type: A, layer: 1, pos: 21
type: B, layer: 1, pos: 21
type: A, layer: 1, pos: 38
type: B, layer: 1, pos: 35
type: B, layer: 1, pos: 5
type: A, layer: 1, pos: 35
type: B, layer: 1, pos: 30
type: A, layer: 1, pos: 5
type: A, layer: 1, pos: 30
type: B, layer: 1, pos: 3
type: A, layer: 1, pos: 24
type: B, layer: 1, pos: 47
type: A, layer: 1, pos: 15
type: A, layer: 1, pos: 3
type: B, layer: 1, pos: 10
type: A, layer: 1, pos: 47
type: B, layer: 1, pos: 1
type: B, layer: 1, pos: 15
type: B, layer: 1, pos: 24
type: A, layer: 1, pos: 10
type: A, layer: 1, pos: 1
type: A, layer: 1, pos: 14
type: A, layer: 1, pos: 40
type: B, layer: 1, pos: 40
type: B, layer: 1, pos: 14

Time for candidate selection: 0.23 seconds

### Candidate
type: B, layer: 1, pos: 0

## Relational analysis of NS_A1_B1

### Relational analysis result of NS_A1_B1
Status: Status.UNKNOWN
Output dim: 0, lower bound: -914.9857326, upper bound: 914.9857326
time: 0.44 seconds

## Relational analysis of NS_A1_B2

### Relational analysis result of NS_A1_B2
Status: Status.UNKNOWN
Output dim: 0, lower bound: -914.9857326, upper bound: 914.9857326
time: 0.45 seconds

## BFS NS instance: NS_A2

### Backsubstitution after applying NS history:
0: -534.0440063, 349.2824707, -608.1965942, 397.9551392, -931.9991455, 957.4790649
1: -40.7368507, 31.0746841, -45.7593231, 35.4486732, -76.1855240, 76.8339844
2: -33.6063690, 46.8075294, -37.5224533, 53.4315834, -87.0379486, 84.3299866
3: -38.0319176, 74.4518585, -42.8564987, 84.6857834, -122.7176971, 117.3083420
4: -28.9225883, 46.8524361, -32.6301079, 53.2674866, -82.1900635, 79.4825439

Time for backsubstitution: 2.81 seconds

### NS candidates at layer 1
type: B, layer: 1, pos: 0
type: B, layer: 1, pos: 8
type: A, layer: 1, pos: 8
type: B, layer: 1, pos: 5
type: B, layer: 1, pos: 35
type: B, layer: 1, pos: 38
type: A, layer: 1, pos: 5
type: A, layer: 1, pos: 38
type: B, layer: 1, pos: 3
type: A, layer: 1, pos: 35
type: B, layer: 1, pos: 10
type: B, layer: 1, pos: 18
type: B, layer: 1, pos: 47
type: A, layer: 1, pos: 18
type: A, layer: 1, pos: 3
type: A, layer: 1, pos: 47
type: A, layer: 1, pos: 10
type: A, layer: 1, pos: 21
type: B, layer: 1, pos: 21
type: A, layer: 1, pos: 30
type: B, layer: 1, pos: 30
type: B, layer: 1, pos: 1
type: B, layer: 1, pos: 24
type: A, layer: 1, pos: 24
type: B, layer: 1, pos: 15
type: A, layer: 1, pos: 15
type: A, layer: 1, pos: 1
type: B, layer: 1, pos: 14
type: A, layer: 1, pos: 14
type: B, layer: 1, pos: 40
type: A, layer: 1, pos: 40

Time for candidate selection: 0.23 seconds

### Candidate
type: B, layer: 1, pos: 0

## Relational analysis of NS_A2_B1

### Relational analysis result of NS_A2_B1
Status: Status.UNKNOWN
Output dim: 0, lower bound: -914.9857326, upper bound: 916.3384520
time: 0.44 seconds

## Relational analysis of NS_A2_B2

### Relational analysis result of NS_A2_B2
Status: Status.UNKNOWN
Output dim: 0, lower bound: -914.9857326, upper bound: 916.6168118
time: 0.45 seconds

## Summary of splitting at layer (split count: 1)
- Time for NS candidates: 3.95 seconds
NS_A1_B1, status: Status.UNKNOWN, split count: 2, time: 3.95
Output dim: 0, lower bound: -914.9857326, upper bound: 914.9857326
NS_A1_B2, status: Status.UNKNOWN, split count: 2, time: 3.95
Output dim: 0, lower bound: -914.9857326, upper bound: 914.9857326
NS_A2_B1, status: Status.UNKNOWN, split count: 2, time: 3.95
Output dim: 0, lower bound: -914.9857326, upper bound: 916.3384520
NS_A2_B2, status: Status.UNKNOWN, split count: 2, time: 3.95
Output dim: 0, lower bound: -914.9857326, upper bound: 916.6168118

## BFS NS instance: NS_A1_B1

### Backsubstitution after applying NS history:
0: -511.7307129, 333.6546936, -511.7307129, 333.6546936, -845.3853760, 845.3853760
1: -38.9493980, 29.6902771, -38.9493980, 29.6902771, -68.6396790, 68.6396790
2: -32.5045624, 44.4840584, -32.5045624, 44.4840584, -76.9886169, 76.9886169
3: -36.7355003, 70.9758224, -36.7355003, 70.9758224, -107.7112961, 107.7112961
4: -27.8541012, 44.5239906, -27.8541012, 44.5239906, -72.3780899, 72.3780899

Time for backsubstitution: 2.80 seconds

### NS candidates at layer 1
type: A, layer: 1, pos: 8
type: B, layer: 1, pos: 8
type: A, layer: 1, pos: 18
type: B, layer: 1, pos: 18
type: A, layer: 1, pos: 38
type: B, layer: 1, pos: 38
type: A, layer: 1, pos: 21
type: B, layer: 1, pos: 21
type: B, layer: 1, pos: 35
type: A, layer: 1, pos: 35
type: A, layer: 1, pos: 5
type: B, layer: 1, pos: 5
type: A, layer: 1, pos: 30
type: B, layer: 1, pos: 30
type: A, layer: 1, pos: 3
type: B, layer: 1, pos: 3
type: A, layer: 1, pos: 47
type: B, layer: 1, pos: 47
type: A, layer: 1, pos: 24
type: B, layer: 1, pos: 24
type: A, layer: 1, pos: 15
type: B, layer: 1, pos: 15
type: B, layer: 1, pos: 10
type: A, layer: 1, pos: 10
type: B, layer: 1, pos: 1
type: A, layer: 1, pos: 1
type: A, layer: 1, pos: 40
type: B, layer: 1, pos: 40
type: A, layer: 1, pos: 14
type: B, layer: 1, pos: 14

Time for candidate selection: 0.23 seconds

### Candidate
type: A, layer: 1, pos: 8

## Relational analysis of NS_A1_B1_A1
Optimization infeasible because this subproblem isn't reachable.

### Candidate
type: B, layer: 1, pos: 8

### Candidate
type: A, layer: 1, pos: 18

## Relational analysis of NS_A1_B1_A1

### Relational analysis result of NS_A1_B1_A1
Status: Status.UNKNOWN
Output dim: 0, lower bound: -914.9312023, upper bound: 914.9408957
time: 0.44 seconds

## Relational analysis of NS_A1_B1_A2

### Relational analysis result of NS_A1_B1_A2
Status: Status.UNKNOWN
Output dim: 0, lower bound: -914.9312023, upper bound: 914.9312023
time: 0.45 seconds

## BFS NS instance: NS_A1_B2

### Backsubstitution after applying NS history:
0: -511.7307129, 333.6546936, -532.9789429, 348.5797729, -860.3104248, 866.6336670
1: -38.9493980, 29.6902771, -40.6601448, 31.0109177, -69.9603119, 70.3504181
2: -32.5045624, 44.4840584, -33.5497093, 46.7177048, -79.2222672, 78.0337677
3: -36.7355003, 70.9758224, -37.9622993, 74.3035583, -111.0390320, 108.9381180
4: -27.8541012, 44.5239906, -28.8685322, 46.7647743, -74.6188736, 73.3925247

Time for backsubstitution: 2.81 seconds

### NS candidates at layer 1
type: B, layer: 1, pos: 8
type: A, layer: 1, pos: 8
type: B, layer: 1, pos: 18
type: A, layer: 1, pos: 18
type: B, layer: 1, pos: 38
type: A, layer: 1, pos: 38
type: A, layer: 1, pos: 21
type: B, layer: 1, pos: 5
type: B, layer: 1, pos: 21
type: B, layer: 1, pos: 35
type: A, layer: 1, pos: 35
type: A, layer: 1, pos: 5
type: B, layer: 1, pos: 30
type: A, layer: 1, pos: 30
type: B, layer: 1, pos: 3
type: A, layer: 1, pos: 24
type: A, layer: 1, pos: 3
type: B, layer: 1, pos: 47
type: A, layer: 1, pos: 47
type: B, layer: 1, pos: 10
type: A, layer: 1, pos: 1
type: A, layer: 1, pos: 10
type: B, layer: 1, pos: 1
type: A, layer: 1, pos: 15
type: B, layer: 1, pos: 24
type: B, layer: 1, pos: 15
type: A, layer: 1, pos: 14
type: B, layer: 1, pos: 14
type: B, layer: 1, pos: 40
type: A, layer: 1, pos: 40

Time for candidate selection: 0.23 seconds

### Candidate
type: B, layer: 1, pos: 8

## Relational analysis of NS_A1_B2_B1
Optimization infeasible because this subproblem isn't reachable.

### Candidate
type: A, layer: 1, pos: 8

### Candidate
type: B, layer: 1, pos: 18

## Relational analysis of NS_A1_B2_B1

### Relational analysis result of NS_A1_B2_B1
Status: Status.UNKNOWN
Output dim: 0, lower bound: -914.9408957, upper bound: 914.9312023
time: 0.42 seconds

## Relational analysis of NS_A1_B2_B2

### Relational analysis result of NS_A1_B2_B2
Status: Status.UNKNOWN
Output dim: 0, lower bound: -914.9312023, upper bound: 914.9312023
time: 0.45 seconds

## BFS NS instance: NS_A2_B1

### Backsubstitution after applying NS history:
0: -534.0440063, 349.2824707, -511.7307129, 333.6546936, -867.6986694, 861.0131836
1: -40.7368507, 31.0746841, -38.9493980, 29.6902771, -70.4271240, 70.0240784
2: -33.6063690, 46.8075294, -32.5045624, 44.4840584, -78.0904236, 79.3120880
3: -38.0319176, 74.4518585, -36.7355003, 70.9758224, -109.0077286, 111.1873398
4: -28.9225883, 46.8524361, -27.8541012, 44.5239906, -73.4465714, 74.7065353

Time for backsubstitution: 2.80 seconds

### NS candidates at layer 1
type: A, layer: 1, pos: 8
type: B, layer: 1, pos: 8
type: A, layer: 1, pos: 18
type: B, layer: 1, pos: 18
type: A, layer: 1, pos: 38
type: B, layer: 1, pos: 38
type: B, layer: 1, pos: 21
type: A, layer: 1, pos: 5
type: A, layer: 1, pos: 21
type: A, layer: 1, pos: 35
type: B, layer: 1, pos: 35
type: B, layer: 1, pos: 5
type: A, layer: 1, pos: 30
type: B, layer: 1, pos: 30
type: A, layer: 1, pos: 3
type: B, layer: 1, pos: 24
type: B, layer: 1, pos: 3
type: A, layer: 1, pos: 47
type: B, layer: 1, pos: 47
type: A, layer: 1, pos: 10
type: B, layer: 1, pos: 1
type: B, layer: 1, pos: 10
type: A, layer: 1, pos: 1
type: B, layer: 1, pos: 15
type: A, layer: 1, pos: 24
type: A, layer: 1, pos: 15
type: B, layer: 1, pos: 14
type: A, layer: 1, pos: 14
type: A, layer: 1, pos: 40
type: B, layer: 1, pos: 40

Time for candidate selection: 0.23 seconds

### Candidate
type: A, layer: 1, pos: 8

## Relational analysis of NS_A2_B1_A1
Optimization infeasible because this subproblem isn't reachable.

### Candidate
type: B, layer: 1, pos: 8

### Candidate
type: A, layer: 1, pos: 18

## Relational analysis of NS_A2_B1_A1

### Relational analysis result of NS_A2_B1_A1
Status: Status.UNKNOWN
Output dim: 0, lower bound: -914.9312023, upper bound: 916.2828792
time: 0.44 seconds

## Relational analysis of NS_A2_B1_A2

### Relational analysis result of NS_A2_B1_A2
Status: Status.UNKNOWN
Output dim: 0, lower bound: -914.9312023, upper bound: 916.3063148
time: 0.44 seconds

## BFS NS instance: NS_A2_B2

### Backsubstitution after applying NS history:
0: -534.0440063, 349.2824707, -534.0440063, 349.2824707, -883.3264771, 883.3264771
1: -40.7368507, 31.0746841, -40.7368507, 31.0746841, -71.8115387, 71.8115387
2: -33.6063690, 46.8075294, -33.6063690, 46.8075294, -80.4138947, 80.4138947
3: -38.0319176, 74.4518585, -38.0319176, 74.4518585, -112.4837799, 112.4837799
4: -28.9225883, 46.8524361, -28.9225883, 46.8524361, -75.7750168, 75.7750168

Time for backsubstitution: 2.82 seconds

### NS candidates at layer 1
type: A, layer: 1, pos: 35
type: B, layer: 1, pos: 35
type: B, layer: 1, pos: 18
type: A, layer: 1, pos: 18
type: A, layer: 1, pos: 5
type: B, layer: 1, pos: 5
type: A, layer: 1, pos: 38
type: B, layer: 1, pos: 38
type: A, layer: 1, pos: 8
type: B, layer: 1, pos: 8
type: A, layer: 1, pos: 3
type: B, layer: 1, pos: 3
type: A, layer: 1, pos: 21
type: B, layer: 1, pos: 21
type: B, layer: 1, pos: 10
type: A, layer: 1, pos: 10
type: A, layer: 1, pos: 47
type: B, layer: 1, pos: 47
type: A, layer: 1, pos: 30
type: B, layer: 1, pos: 30
type: B, layer: 1, pos: 1
type: A, layer: 1, pos: 1
type: B, layer: 1, pos: 15
type: A, layer: 1, pos: 15
type: B, layer: 1, pos: 24
type: A, layer: 1, pos: 24
type: B, layer: 1, pos: 14
type: A, layer: 1, pos: 14
type: B, layer: 1, pos: 40
type: A, layer: 1, pos: 40

Time for candidate selection: 0.23 seconds

### Candidate
type: A, layer: 1, pos: 35

## Relational analysis of NS_A2_B2_A1

### Relational analysis result of NS_A2_B2_A1
Status: Status.UNKNOWN
Output dim: 0, lower bound: -914.3119385, upper bound: 915.1714112
time: 0.57 seconds

## Relational analysis of NS_A2_B2_A2

### Relational analysis result of NS_A2_B2_A2
Status: Status.UNKNOWN
Output dim: 0, lower bound: -914.9739698, upper bound: 916.6136649
time: 0.45 seconds

## Summary of splitting at layer (split count: 2)
- Time for NS candidates: 4.08 seconds
NS_A1_B1_A1, status: Status.UNKNOWN, split count: 3, time: 4.08
Output dim: 0, lower bound: -914.9312023, upper bound: 914.9408957
NS_A1_B1_A2, status: Status.UNKNOWN, split count: 3, time: 4.08
Output dim: 0, lower bound: -914.9312023, upper bound: 914.9312023
NS_A1_B2_B1, status: Status.UNKNOWN, split count: 3, time: 4.08
Output dim: 0, lower bound: -914.9408957, upper bound: 914.9312023
NS_A1_B2_B2, status: Status.UNKNOWN, split count: 3, time: 4.08
Output dim: 0, lower bound: -914.9312023, upper bound: 914.9312023
NS_A2_B1_A1, status: Status.UNKNOWN, split count: 3, time: 4.08
Output dim: 0, lower bound: -914.9312023, upper bound: 916.2828792
NS_A2_B1_A2, status: Status.UNKNOWN, split count: 3, time: 4.08
Output dim: 0, lower bound: -914.9312023, upper bound: 916.3063148
NS_A2_B2_A1, status: Status.UNKNOWN, split count: 3, time: 4.08
Output dim: 0, lower bound: -914.3119385, upper bound: 915.1714112
NS_A2_B2_A2, status: Status.UNKNOWN, split count: 3, time: 4.08
Output dim: 0, lower bound: -914.9739698, upper bound: 916.6136649

## BFS NS instance: NS_A1_B1_A1

### Backsubstitution after applying NS history:
0: -462.3272095, 293.1559448, -497.5832825, 321.8188782, -784.1460571, 790.7392578
1: -34.5358887, 26.5900002, -37.6684494, 28.8035984, -63.3394852, 64.2584457
2: -29.3298416, 38.9291420, -31.4760418, 42.9133759, -72.2432098, 70.4051819
3: -32.8668022, 62.9823685, -35.5307846, 68.7048569, -101.5716553, 98.5131531
4: -25.1946354, 39.0688972, -26.9676399, 42.9941216, -68.1887589, 66.0365219

Time for backsubstitution: 2.82 seconds

### NS candidates at layer 1
type: B, layer: 1, pos: 8
type: A, layer: 1, pos: 8
type: B, layer: 1, pos: 18
type: B, layer: 1, pos: 38
type: A, layer: 1, pos: 38
type: A, layer: 1, pos: 21
type: B, layer: 1, pos: 21
type: B, layer: 1, pos: 35
type: A, layer: 1, pos: 35
type: B, layer: 1, pos: 5
type: A, layer: 1, pos: 5
type: B, layer: 1, pos: 30
type: A, layer: 1, pos: 15
type: A, layer: 1, pos: 30
type: A, layer: 1, pos: 24
type: B, layer: 1, pos: 15
type: A, layer: 1, pos: 1
type: B, layer: 1, pos: 3
type: B, layer: 1, pos: 47
type: B, layer: 1, pos: 1
type: A, layer: 1, pos: 3
type: B, layer: 1, pos: 10
type: A, layer: 1, pos: 47
type: B, layer: 1, pos: 24
type: A, layer: 1, pos: 10
type: A, layer: 1, pos: 40
type: B, layer: 1, pos: 40
type: A, layer: 1, pos: 14
type: B, layer: 1, pos: 14

Time for candidate selection: 0.24 seconds

### Candidate
type: B, layer: 1, pos: 8

### Candidate
type: A, layer: 1, pos: 8

### Candidate
type: B, layer: 1, pos: 18

## Relational analysis of NS_A1_B1_A1_B1

### Relational analysis result of NS_A1_B1_A1_B1
Status: Status.UNKNOWN
Output dim: 0, lower bound: -914.9312023, upper bound: 914.9312023
time: 0.45 seconds

## Relational analysis of NS_A1_B1_A1_B2

### Relational analysis result of NS_A1_B1_A1_B2
Status: Status.UNKNOWN
Output dim: 0, lower bound: -914.9312023, upper bound: 914.9312023
time: 0.42 seconds

## BFS NS instance: NS_A1_B1_A2

### Backsubstitution after applying NS history:
0: -502.7959900, 326.4364929, -509.0769043, 331.5209656, -834.3168945, 835.5133667
1: -38.1636848, 29.1213188, -38.7159309, 29.5196915, -67.6833725, 67.8372498
2: -31.9113941, 43.5109329, -32.3307037, 44.1949081, -76.1063004, 75.8416061
3: -36.0328712, 69.5438232, -36.5294762, 70.5489502, -106.5818176, 106.0732880
4: -27.3170071, 43.5654984, -27.6952934, 44.2384186, -71.5554276, 71.2607880

Time for backsubstitution: 2.81 seconds

### NS candidates at layer 1
type: B, layer: 1, pos: 18
type: A, layer: 1, pos: 8
type: B, layer: 1, pos: 8
type: A, layer: 1, pos: 5
type: B, layer: 1, pos: 5
type: A, layer: 1, pos: 21
type: B, layer: 1, pos: 21
type: B, layer: 1, pos: 38
type: A, layer: 1, pos: 38
type: A, layer: 1, pos: 47
type: B, layer: 1, pos: 47
type: A, layer: 1, pos: 15
type: B, layer: 1, pos: 35
type: A, layer: 1, pos: 35
type: B, layer: 1, pos: 10
type: A, layer: 1, pos: 10
type: B, layer: 1, pos: 30
type: B, layer: 1, pos: 15
type: A, layer: 1, pos: 30
type: A, layer: 1, pos: 3
type: B, layer: 1, pos: 3
type: A, layer: 1, pos: 24
type: B, layer: 1, pos: 24
type: B, layer: 1, pos: 1
type: A, layer: 1, pos: 1
type: A, layer: 1, pos: 14
type: B, layer: 1, pos: 14
type: B, layer: 1, pos: 40
type: A, layer: 1, pos: 40

Time for candidate selection: 0.23 seconds

### Candidate
type: B, layer: 1, pos: 18

## Relational analysis of NS_A1_B1_A2_B1

### Relational analysis result of NS_A1_B1_A2_B1
Status: Status.UNKNOWN
Output dim: 0, lower bound: -914.9312023, upper bound: 914.9312023
time: 0.42 seconds

## Relational analysis of NS_A1_B1_A2_B2

### Relational analysis result of NS_A1_B1_A2_B2
Status: Status.UNKNOWN
Output dim: 0, lower bound: -914.9312023, upper bound: 914.9312023
time: 0.45 seconds

## BFS NS instance: NS_A1_B2_B1

### Backsubstitution after applying NS history:
0: -497.5832825, 321.8188782, -488.3984070, 313.5505981, -811.1339111, 810.2171631
1: -37.6684494, 28.8035984, -36.8722458, 28.1667194, -65.8351669, 65.6758423
2: -31.4760418, 42.9133759, -30.8656483, 41.8754425, -73.3514786, 73.7790146
3: -35.5307846, 68.7048569, -34.6121788, 67.0116425, -102.5424271, 103.3170242
4: -26.9676399, 42.9941216, -26.5074844, 42.0064163, -68.9740448, 69.5016022

Time for backsubstitution: 2.81 seconds

### NS candidates at layer 1
type: B, layer: 1, pos: 8
type: A, layer: 1, pos: 8
type: A, layer: 1, pos: 18
type: A, layer: 1, pos: 38
type: B, layer: 1, pos: 38
type: B, layer: 1, pos: 21
type: A, layer: 1, pos: 21
type: B, layer: 1, pos: 5
type: A, layer: 1, pos: 35
type: B, layer: 1, pos: 35
type: A, layer: 1, pos: 5
type: A, layer: 1, pos: 30
type: B, layer: 1, pos: 30
type: B, layer: 1, pos: 3
type: A, layer: 1, pos: 10
type: A, layer: 1, pos: 3
type: B, layer: 1, pos: 15
type: B, layer: 1, pos: 10
type: A, layer: 1, pos: 47
type: A, layer: 1, pos: 24
type: B, layer: 1, pos: 1
type: B, layer: 1, pos: 47
type: A, layer: 1, pos: 15
type: A, layer: 1, pos: 1
type: B, layer: 1, pos: 24
type: A, layer: 1, pos: 14
type: B, layer: 1, pos: 14
type: B, layer: 1, pos: 40
type: A, layer: 1, pos: 40

Time for candidate selection: 0.23 seconds

### Candidate
type: B, layer: 1, pos: 8

### Candidate
type: A, layer: 1, pos: 8

### Candidate
type: A, layer: 1, pos: 18

## Relational analysis of NS_A1_B2_B1_A1

### Relational analysis result of NS_A1_B2_B1_A1
Status: Status.UNKNOWN
Output dim: 0, lower bound: -916.2828792, upper bound: 914.9312023
time: 0.42 seconds

## Relational analysis of NS_A1_B2_B1_A2

### Relational analysis result of NS_A1_B2_B1_A2
Status: Status.UNKNOWN
Output dim: 0, lower bound: -916.2828798, upper bound: 914.9312023
time: 0.47 seconds

## BFS NS instance: NS_A1_B2_B2

### Backsubstitution after applying NS history:
0: -509.0769043, 331.5209656, -520.1079102, 338.3668213, -847.4437256, 851.6287842
1: -38.7159309, 29.5196915, -39.5576324, 30.1827927, -68.8987274, 69.0773239
2: -32.3307037, 44.1949081, -32.7032204, 45.3268547, -77.6575546, 76.8981323
3: -36.5294762, 70.5489502, -36.9639473, 72.1907349, -108.7201996, 107.5128937
4: -27.6952934, 44.2384186, -28.1038265, 45.3841934, -73.0794754, 72.3422470

Time for backsubstitution: 2.82 seconds

### NS candidates at layer 1
type: B, layer: 1, pos: 8
type: A, layer: 1, pos: 8
type: A, layer: 1, pos: 18
type: B, layer: 1, pos: 38
type: A, layer: 1, pos: 38
type: A, layer: 1, pos: 21
type: B, layer: 1, pos: 21
type: B, layer: 1, pos: 5
type: A, layer: 1, pos: 35
type: B, layer: 1, pos: 35
type: A, layer: 1, pos: 5
type: B, layer: 1, pos: 30
type: A, layer: 1, pos: 30
type: B, layer: 1, pos: 3
type: A, layer: 1, pos: 24
type: A, layer: 1, pos: 3
type: B, layer: 1, pos: 47
type: A, layer: 1, pos: 1
type: B, layer: 1, pos: 10
type: A, layer: 1, pos: 47
type: A, layer: 1, pos: 10
type: B, layer: 1, pos: 1
type: B, layer: 1, pos: 24
type: A, layer: 1, pos: 15
type: B, layer: 1, pos: 15
type: A, layer: 1, pos: 14
type: B, layer: 1, pos: 14
type: B, layer: 1, pos: 40
type: A, layer: 1, pos: 40

Time for candidate selection: 0.23 seconds

### Candidate
type: B, layer: 1, pos: 8

### Candidate
type: A, layer: 1, pos: 8

### Candidate
type: A, layer: 1, pos: 18

## Relational analysis of NS_A1_B2_B2_A1

### Relational analysis result of NS_A1_B2_B2_A1
Status: Status.UNKNOWN
Output dim: 0, lower bound: -916.3063148, upper bound: 914.9312023
time: 0.42 seconds

## Relational analysis of NS_A1_B2_B2_A2

### Relational analysis result of NS_A1_B2_B2_A2
Status: Status.UNKNOWN
Output dim: 0, lower bound: -916.3063148, upper bound: 914.9312023
time: 0.45 seconds

## BFS NS instance: NS_A2_B1_A1

### Backsubstitution after applying NS history:
0: -489.2505188, 314.2459106, -497.5832825, 321.8188782, -811.0693359, 811.8292236
1: -36.9483871, 28.2214146, -37.6684494, 28.8035984, -65.7519836, 65.8898621
2: -30.9117508, 41.9644508, -31.4760418, 42.9133759, -73.8251266, 73.4404907
3: -34.6661301, 67.1468887, -35.5307846, 68.7048569, -103.3709717, 102.6776733
4: -26.5482178, 42.0926247, -26.9676399, 42.9941216, -69.5423431, 69.0602570

Time for backsubstitution: 2.83 seconds

### NS candidates at layer 1
type: A, layer: 1, pos: 8
type: B, layer: 1, pos: 8
type: B, layer: 1, pos: 18
type: B, layer: 1, pos: 38
type: A, layer: 1, pos: 38
type: A, layer: 1, pos: 21
type: B, layer: 1, pos: 21
type: A, layer: 1, pos: 5
type: B, layer: 1, pos: 35
type: A, layer: 1, pos: 35
type: B, layer: 1, pos: 5
type: B, layer: 1, pos: 30
type: A, layer: 1, pos: 30
type: A, layer: 1, pos: 3
type: B, layer: 1, pos: 10
type: B, layer: 1, pos: 3
type: A, layer: 1, pos: 15
type: A, layer: 1, pos: 10
type: B, layer: 1, pos: 47
type: B, layer: 1, pos: 24
type: A, layer: 1, pos: 1
type: A, layer: 1, pos: 47
type: B, layer: 1, pos: 15
type: B, layer: 1, pos: 1
type: A, layer: 1, pos: 24
type: B, layer: 1, pos: 14
type: A, layer: 1, pos: 14
type: A, layer: 1, pos: 40
type: B, layer: 1, pos: 40

Time for candidate selection: 0.23 seconds

### Candidate
type: A, layer: 1, pos: 8

### Candidate
type: B, layer: 1, pos: 8

### Candidate
type: B, layer: 1, pos: 18

## Relational analysis of NS_A2_B1_A1_B1

### Relational analysis result of NS_A2_B1_A1_B1
Status: Status.UNKNOWN
Output dim: 0, lower bound: -914.9312023, upper bound: 916.2828792
time: 0.45 seconds

## Relational analysis of NS_A2_B1_A1_B2

### Relational analysis result of NS_A2_B1_A1_B2
Status: Status.UNKNOWN
Output dim: 0, lower bound: -914.9312023, upper bound: 916.2828792
time: 0.42 seconds

## BFS NS instance: NS_A2_B1_A2

### Backsubstitution after applying NS history:
0: -520.7755737, 338.8567810, -509.0769043, 331.5209656, -852.2965088, 847.9337158
1: -39.6113853, 30.2250767, -38.7159309, 29.5196915, -69.1310730, 68.9410095
2: -32.7410431, 45.3893623, -32.3307037, 44.1949081, -76.9359512, 77.7200623
3: -37.0099106, 72.2946548, -36.5294762, 70.5489502, -107.5588608, 108.8241196
4: -28.1395817, 45.4455528, -27.6952934, 44.2384186, -72.3779907, 73.1408310

Time for backsubstitution: 2.83 seconds

### NS candidates at layer 1
type: A, layer: 1, pos: 8
type: B, layer: 1, pos: 8
type: B, layer: 1, pos: 18
type: A, layer: 1, pos: 38
type: B, layer: 1, pos: 38
type: B, layer: 1, pos: 21
type: A, layer: 1, pos: 21
type: A, layer: 1, pos: 5
type: B, layer: 1, pos: 35
type: A, layer: 1, pos: 35
type: B, layer: 1, pos: 5
type: A, layer: 1, pos: 30
type: B, layer: 1, pos: 30
type: A, layer: 1, pos: 3
type: B, layer: 1, pos: 24
type: B, layer: 1, pos: 3
type: A, layer: 1, pos: 47
type: B, layer: 1, pos: 1
type: A, layer: 1, pos: 10
type: B, layer: 1, pos: 47
type: B, layer: 1, pos: 10
type: A, layer: 1, pos: 1
type: A, layer: 1, pos: 24
type: B, layer: 1, pos: 15
type: A, layer: 1, pos: 15
type: B, layer: 1, pos: 14
type: A, layer: 1, pos: 14
type: A, layer: 1, pos: 40
type: B, layer: 1, pos: 40

Time for candidate selection: 0.23 seconds

### Candidate
type: A, layer: 1, pos: 8

### Candidate
type: B, layer: 1, pos: 8

### Candidate
type: B, layer: 1, pos: 18

## Relational analysis of NS_A2_B1_A2_B1

### Relational analysis result of NS_A2_B1_A2_B1
Status: Status.UNKNOWN
Output dim: 0, lower bound: -914.9312023, upper bound: 916.3063148
time: 0.43 seconds

## Relational analysis of NS_A2_B1_A2_B2

### Relational analysis result of NS_A2_B1_A2_B2
Status: Status.UNKNOWN
Output dim: 0, lower bound: -914.9312023, upper bound: 916.3063148
time: 0.43 seconds

## BFS NS instance: NS_A2_B2_A1

### Backsubstitution after applying NS history:
0: -507.4825134, 328.1139526, -526.9636230, 343.5709229, -851.0534668, 855.0775757
1: -38.4932861, 29.3632183, -40.1324005, 30.6154480, -69.1087341, 69.4956131
2: -31.9028378, 43.9977188, -33.1421700, 46.0421486, -77.9449844, 77.1398697
3: -36.0221329, 70.2285461, -37.4819374, 73.2999573, -109.3220901, 107.7104797
4: -27.4177380, 44.0472527, -28.5068474, 46.0919724, -73.5096970, 72.5540924

Time for backsubstitution: 2.84 seconds

### NS candidates at layer 1
type: B, layer: 1, pos: 18
type: B, layer: 1, pos: 5
type: A, layer: 1, pos: 5
type: A, layer: 1, pos: 18
type: A, layer: 1, pos: 8
type: B, layer: 1, pos: 8
type: B, layer: 1, pos: 35
type: B, layer: 1, pos: 38
type: A, layer: 1, pos: 38
type: B, layer: 1, pos: 3
type: A, layer: 1, pos: 3
type: A, layer: 1, pos: 21
type: B, layer: 1, pos: 21
type: A, layer: 1, pos: 47
type: A, layer: 1, pos: 10
type: B, layer: 1, pos: 47
type: B, layer: 1, pos: 10
type: A, layer: 1, pos: 30
type: B, layer: 1, pos: 30
type: B, layer: 1, pos: 15
type: A, layer: 1, pos: 15
type: B, layer: 1, pos: 1
type: B, layer: 1, pos: 24
type: A, layer: 1, pos: 1
type: A, layer: 1, pos: 24
type: B, layer: 1, pos: 14
type: A, layer: 1, pos: 14
type: B, layer: 1, pos: 40
type: A, layer: 1, pos: 40

Time for candidate selection: 0.24 seconds

### Candidate
type: B, layer: 1, pos: 18

## Relational analysis of NS_A2_B2_A1_B1

### Relational analysis result of NS_A2_B2_A1_B1
Status: Status.UNKNOWN
Output dim: 0, lower bound: -914.9731457, upper bound: 914.7501866
time: 0.49 seconds

## Relational analysis of NS_A2_B2_A1_B2

### Relational analysis result of NS_A2_B2_A1_B2
Status: Status.UNKNOWN
Output dim: 0, lower bound: -915.6483968, upper bound: 915.1457901
time: 0.43 seconds

## BFS NS instance: NS_A2_B2_A2

### Backsubstitution after applying NS history:
0: -521.0087280, 339.0867920, -533.4074707, 348.8117981, -869.8205566, 872.4941406
1: -39.6358490, 30.2453690, -40.6860390, 31.0354023, -70.6712494, 70.9314117
2: -32.7683792, 45.4239540, -33.5669289, 46.7436142, -79.5119781, 78.9908829
3: -37.0397453, 72.3502197, -37.9849548, 74.3548126, -111.3945541, 110.3351746
4: -28.1704330, 45.4808884, -28.8870239, 46.7890816, -74.9595184, 74.3679123

Time for backsubstitution: 2.85 seconds

### NS candidates at layer 1
type: B, layer: 1, pos: 18
type: A, layer: 1, pos: 18
type: B, layer: 1, pos: 35
type: A, layer: 1, pos: 5
type: B, layer: 1, pos: 5
type: B, layer: 1, pos: 38
type: A, layer: 1, pos: 38
type: A, layer: 1, pos: 8
type: B, layer: 1, pos: 8
type: B, layer: 1, pos: 3
type: A, layer: 1, pos: 3
type: B, layer: 1, pos: 21
type: A, layer: 1, pos: 21
type: A, layer: 1, pos: 10
type: B, layer: 1, pos: 10
type: A, layer: 1, pos: 47
type: B, layer: 1, pos: 47
type: A, layer: 1, pos: 30
type: B, layer: 1, pos: 30
type: B, layer: 1, pos: 15
type: B, layer: 1, pos: 1
type: A, layer: 1, pos: 1
type: A, layer: 1, pos: 15
type: A, layer: 1, pos: 24
type: B, layer: 1, pos: 24
type: B, layer: 1, pos: 14
type: A, layer: 1, pos: 14
type: B, layer: 1, pos: 40
type: A, layer: 1, pos: 40

Time for candidate selection: 0.24 seconds

### Candidate
type: B, layer: 1, pos: 18

## Relational analysis of NS_A2_B2_A2_B1

### Relational analysis result of NS_A2_B2_A2_B1
Status: Status.UNKNOWN
Output dim: 0, lower bound: -916.5482259, upper bound: 916.5976782
time: 0.42 seconds

## Relational analysis of NS_A2_B2_A2_B2

### Relational analysis result of NS_A2_B2_A2_B2
Status: Status.UNKNOWN
Output dim: 0, lower bound: -916.6077273, upper bound: 916.6077273
time: 0.50 seconds

## Summary of splitting at layer (split count: 3)
- Time for NS candidates: 4.02 seconds
NS_A1_B1_A1_B1, status: Status.UNKNOWN, split count: 4, time: 4.02
Output dim: 0, lower bound: -914.9312023, upper bound: 914.9312023
NS_A1_B1_A1_B2, status: Status.UNKNOWN, split count: 4, time: 4.02
Output dim: 0, lower bound: -914.9312023, upper bound: 914.9312023
NS_A1_B1_A2_B1, status: Status.UNKNOWN, split count: 4, time: 4.02
Output dim: 0, lower bound: -914.9312023, upper bound: 914.9312023
NS_A1_B1_A2_B2, status: Status.UNKNOWN, split count: 4, time: 4.02
Output dim: 0, lower bound: -914.9312023, upper bound: 914.9312023
NS_A1_B2_B1_A1, status: Status.UNKNOWN, split count: 4, time: 4.02
Output dim: 0, lower bound: -916.2828792, upper bound: 914.9312023
NS_A1_B2_B1_A2, status: Status.UNKNOWN, split count: 4, time: 4.02
Output dim: 0, lower bound: -916.2828798, upper bound: 914.9312023
NS_A1_B2_B2_A1, status: Status.UNKNOWN, split count: 4, time: 4.02
Output dim: 0, lower bound: -916.3063148, upper bound: 914.9312023
NS_A1_B2_B2_A2, status: Status.UNKNOWN, split count: 4, time: 4.02
Output dim: 0, lower bound: -916.3063148, upper bound: 914.9312023
NS_A2_B1_A1_B1, status: Status.UNKNOWN, split count: 4, time: 4.02
Output dim: 0, lower bound: -914.9312023, upper bound: 916.2828792
NS_A2_B1_A1_B2, status: Status.UNKNOWN, split count: 4, time: 4.02
Output dim: 0, lower bound: -914.9312023, upper bound: 916.2828792
NS_A2_B1_A2_B1, status: Status.UNKNOWN, split count: 4, time: 4.02
Output dim: 0, lower bound: -914.9312023, upper bound: 916.3063148
NS_A2_B1_A2_B2, status: Status.UNKNOWN, split count: 4, time: 4.02
Output dim: 0, lower bound: -914.9312023, upper bound: 916.3063148
NS_A2_B2_A1_B1, status: Status.UNKNOWN, split count: 4, time: 4.02
Output dim: 0, lower bound: -914.9731457, upper bound: 914.7501866
NS_A2_B2_A1_B2, status: Status.UNKNOWN, split count: 4, time: 4.02
Output dim: 0, lower bound: -915.6483968, upper bound: 915.1457901
NS_A2_B2_A2_B1, status: Status.UNKNOWN, split count: 4, time: 4.02
Output dim: 0, lower bound: -916.5482259, upper bound: 916.5976782
NS_A2_B2_A2_B2, status: Status.UNKNOWN, split count: 4, time: 4.02
Output dim: 0, lower bound: -916.6077273, upper bound: 916.6077273

## BFS NS instance: NS_A1_B1_A1_B1

### Backsubstitution after applying NS history:
0: -462.3272095, 293.1559448, -462.3272095, 293.1559448, -755.4831543, 755.4831543
1: -34.5358887, 26.5900002, -34.5358887, 26.5900002, -61.1258888, 61.1258888
2: -29.3298416, 38.9291420, -29.3298416, 38.9291420, -68.2589874, 68.2589874
3: -32.8668022, 62.9823685, -32.8668022, 62.9823685, -95.8491669, 95.8491669
4: -25.1946354, 39.0688972, -25.1946354, 39.0688972, -64.2635193, 64.2635193

Time for backsubstitution: 2.82 seconds

### NS candidates at layer 1
type: A, layer: 1, pos: 8
type: B, layer: 1, pos: 8
type: B, layer: 1, pos: 38
type: A, layer: 1, pos: 38
type: A, layer: 1, pos: 30
type: B, layer: 1, pos: 30
type: A, layer: 1, pos: 21
type: B, layer: 1, pos: 21
type: A, layer: 1, pos: 47
type: B, layer: 1, pos: 47
type: A, layer: 1, pos: 35
type: B, layer: 1, pos: 35
type: A, layer: 1, pos: 1
type: B, layer: 1, pos: 1
type: A, layer: 1, pos: 5
type: B, layer: 1, pos: 5
type: A, layer: 1, pos: 15
type: B, layer: 1, pos: 15
type: A, layer: 1, pos: 24
type: B, layer: 1, pos: 24
type: A, layer: 1, pos: 3
type: B, layer: 1, pos: 3
type: A, layer: 1, pos: 10
type: B, layer: 1, pos: 10
type: A, layer: 1, pos: 14
type: B, layer: 1, pos: 14
type: B, layer: 1, pos: 40
type: A, layer: 1, pos: 40

Time for candidate selection: 0.24 seconds

### Candidate
type: A, layer: 1, pos: 8

### Candidate
type: B, layer: 1, pos: 8

### Candidate
type: B, layer: 1, pos: 38

## Relational analysis of NS_A1_B1_A1_B1_B1
Optimization infeasible because this subproblem isn't reachable.

### Candidate
type: A, layer: 1, pos: 38

### Candidate
type: A, layer: 1, pos: 30

## Relational analysis of NS_A1_B1_A1_B1_A1
Optimization infeasible because this subproblem isn't reachable.

### Candidate
type: B, layer: 1, pos: 30

### Candidate
type: A, layer: 1, pos: 21

## Relational analysis of NS_A1_B1_A1_B1_A1
Optimization infeasible because this subproblem isn't reachable.

### Candidate
type: B, layer: 1, pos: 21

### Candidate
type: A, layer: 1, pos: 47

## Relational analysis of NS_A1_B1_A1_B1_A1

### Relational analysis result of NS_A1_B1_A1_B1_A1
Status: Status.UNKNOWN
Output dim: 0, lower bound: -912.7683715, upper bound: 914.3765485
time: 0.48 seconds

## Relational analysis of NS_A1_B1_A1_B1_A2

### Relational analysis result of NS_A1_B1_A1_B1_A2
Status: Status.VERIFIED
Output dim: 0, lower bound: -912.7683715, upper bound: 912.7683715
time: 0.48 seconds

## BFS NS instance: NS_A1_B1_A1_B2

### Backsubstitution after applying NS history:
0: -462.3272095, 293.1559448, -502.7959900, 326.4364929, -788.7636719, 795.9517212
1: -34.5358887, 26.5900002, -38.1636848, 29.1213188, -63.6572075, 64.7536850
2: -29.3298416, 38.9291420, -31.9113941, 43.5109329, -72.8407669, 70.8405380
3: -32.8668022, 62.9823685, -36.0328712, 69.5438232, -102.4106293, 99.0152283
4: -25.1946354, 39.0688972, -27.3170071, 43.5654984, -68.7601318, 66.3859024

Time for backsubstitution: 2.82 seconds

### NS candidates at layer 1
type: B, layer: 1, pos: 8
type: A, layer: 1, pos: 8
type: B, layer: 1, pos: 5
type: A, layer: 1, pos: 5
type: A, layer: 1, pos: 21
type: B, layer: 1, pos: 21
type: B, layer: 1, pos: 38
type: A, layer: 1, pos: 38
type: A, layer: 1, pos: 15
type: B, layer: 1, pos: 47
type: B, layer: 1, pos: 35
type: A, layer: 1, pos: 47
type: B, layer: 1, pos: 10
type: A, layer: 1, pos: 35
type: B, layer: 1, pos: 15
type: B, layer: 1, pos: 30
type: A, layer: 1, pos: 10
type: B, layer: 1, pos: 3
type: A, layer: 1, pos: 3
type: A, layer: 1, pos: 30
type: A, layer: 1, pos: 24
type: B, layer: 1, pos: 24
type: B, layer: 1, pos: 1
type: A, layer: 1, pos: 1
type: A, layer: 1, pos: 14
type: B, layer: 1, pos: 14
type: A, layer: 1, pos: 40
type: B, layer: 1, pos: 40

Time for candidate selection: 0.24 seconds

### Candidate
type: B, layer: 1, pos: 8

### Candidate
type: A, layer: 1, pos: 8

### Candidate
type: B, layer: 1, pos: 5

## Relational analysis of NS_A1_B1_A1_B2_B1

### Relational analysis result of NS_A1_B1_A1_B2_B1
Status: Status.UNKNOWN
Output dim: 0, lower bound: -911.0559777, upper bound: 914.4599584
time: 0.49 seconds

## Relational analysis of NS_A1_B1_A1_B2_B2

### Relational analysis result of NS_A1_B1_A1_B2_B2
Status: Status.UNKNOWN
Output dim: 0, lower bound: -914.8548589, upper bound: 914.8657075
time: 0.41 seconds

## BFS NS instance: NS_A1_B1_A2_B1

### Backsubstitution after applying NS history:
0: -502.7959900, 326.4364929, -462.3272095, 293.1559448, -795.9517212, 788.7636719
1: -38.1636848, 29.1213188, -34.5358887, 26.5900002, -64.7536774, 63.6572075
2: -31.9113941, 43.5109329, -29.3298416, 38.9291420, -70.8405380, 72.8407669
3: -36.0328712, 69.5438232, -32.8668022, 62.9823685, -99.0152283, 102.4106293
4: -27.3170071, 43.5654984, -25.1946354, 39.0688972, -66.3859024, 68.7601318

Time for backsubstitution: 2.83 seconds

### NS candidates at layer 1
type: A, layer: 1, pos: 8
type: B, layer: 1, pos: 8
type: A, layer: 1, pos: 5
type: B, layer: 1, pos: 5
type: B, layer: 1, pos: 21
type: A, layer: 1, pos: 21
type: A, layer: 1, pos: 38
type: B, layer: 1, pos: 38
type: B, layer: 1, pos: 15
type: A, layer: 1, pos: 47
type: A, layer: 1, pos: 35
type: B, layer: 1, pos: 47
type: A, layer: 1, pos: 10
type: B, layer: 1, pos: 35
type: A, layer: 1, pos: 15
type: A, layer: 1, pos: 30
type: B, layer: 1, pos: 10
type: A, layer: 1, pos: 3
type: B, layer: 1, pos: 3
type: B, layer: 1, pos: 30
type: B, layer: 1, pos: 24
type: A, layer: 1, pos: 24
type: A, layer: 1, pos: 1
type: B, layer: 1, pos: 1
type: B, layer: 1, pos: 14
type: A, layer: 1, pos: 14
type: B, layer: 1, pos: 40
type: A, layer: 1, pos: 40

Time for candidate selection: 0.24 seconds

### Candidate
type: A, layer: 1, pos: 8

### Candidate
type: B, layer: 1, pos: 8

### Candidate
type: A, layer: 1, pos: 5

## Relational analysis of NS_A1_B1_A2_B1_A1

### Relational analysis result of NS_A1_B1_A2_B1_A1
Status: Status.UNKNOWN
Output dim: 0, lower bound: -914.4599584, upper bound: 912.3608761
time: 0.49 seconds

## Relational analysis of NS_A1_B1_A2_B1_A2

### Relational analysis result of NS_A1_B1_A2_B1_A2
Status: Status.UNKNOWN
Output dim: 0, lower bound: -914.8548591, upper bound: 914.8548589
time: 0.44 seconds

## BFS NS instance: NS_A1_B1_A2_B2

### Backsubstitution after applying NS history:
0: -502.7959900, 326.4364929, -502.7959900, 326.4364929, -829.2322388, 829.2322388
1: -38.1636848, 29.1213188, -38.1636848, 29.1213188, -67.2849960, 67.2849960
2: -31.9113941, 43.5109329, -31.9113941, 43.5109329, -75.4223099, 75.4223099
3: -36.0328712, 69.5438232, -36.0328712, 69.5438232, -105.5766907, 105.5766907
4: -27.3170071, 43.5654984, -27.3170071, 43.5654984, -70.8825073, 70.8825073

Time for backsubstitution: 2.83 seconds

### NS candidates at layer 1
type: A, layer: 1, pos: 8
type: B, layer: 1, pos: 8
type: A, layer: 1, pos: 5
type: B, layer: 1, pos: 5
type: A, layer: 1, pos: 21
type: B, layer: 1, pos: 21
type: A, layer: 1, pos: 38
type: B, layer: 1, pos: 38
type: A, layer: 1, pos: 47
type: B, layer: 1, pos: 47
type: A, layer: 1, pos: 35
type: B, layer: 1, pos: 35
type: A, layer: 1, pos: 15
type: B, layer: 1, pos: 15
type: A, layer: 1, pos: 10
type: B, layer: 1, pos: 10
type: A, layer: 1, pos: 30
type: B, layer: 1, pos: 30
type: A, layer: 1, pos: 3
type: B, layer: 1, pos: 3
type: B, layer: 1, pos: 24
type: A, layer: 1, pos: 24
type: A, layer: 1, pos: 1
type: B, layer: 1, pos: 1
type: A, layer: 1, pos: 14
type: B, layer: 1, pos: 14
type: A, layer: 1, pos: 40
type: B, layer: 1, pos: 40

Time for candidate selection: 0.24 seconds

### Candidate
type: A, layer: 1, pos: 8

### Candidate
type: B, layer: 1, pos: 8

### Candidate
type: A, layer: 1, pos: 5

## Relational analysis of NS_A1_B1_A2_B2_A1

### Relational analysis result of NS_A1_B1_A2_B2_A1
Status: Status.UNKNOWN
Output dim: 0, lower bound: -914.4599584, upper bound: 913.5295566
time: 0.50 seconds

## Relational analysis of NS_A1_B1_A2_B2_A2

### Relational analysis result of NS_A1_B1_A2_B2_A2
Status: Status.UNKNOWN
Output dim: 0, lower bound: -914.8548591, upper bound: 914.8548589
time: 0.45 seconds

## BFS NS instance: NS_A1_B2_B1_A1

### Backsubstitution after applying NS history:
0: -462.3272095, 293.1559448, -488.3984070, 313.5505981, -775.8778076, 781.5543213
1: -34.5358887, 26.5900002, -36.8722458, 28.1667194, -62.7026062, 63.4622421
2: -29.3298416, 38.9291420, -30.8656483, 41.8754425, -71.2052765, 69.7947922
3: -32.8668022, 62.9823685, -34.6121788, 67.0116425, -99.8784485, 97.5945435
4: -25.1946354, 39.0688972, -26.5074844, 42.0064163, -67.2010422, 65.5763855

Time for backsubstitution: 2.83 seconds

### NS candidates at layer 1
type: B, layer: 1, pos: 8
type: A, layer: 1, pos: 8
type: B, layer: 1, pos: 30
type: B, layer: 1, pos: 38
type: A, layer: 1, pos: 38
type: A, layer: 1, pos: 30
type: A, layer: 1, pos: 21
type: B, layer: 1, pos: 21
type: B, layer: 1, pos: 47
type: A, layer: 1, pos: 47
type: B, layer: 1, pos: 35
type: B, layer: 1, pos: 5
type: A, layer: 1, pos: 35
type: A, layer: 1, pos: 5
type: A, layer: 1, pos: 1
type: B, layer: 1, pos: 1
type: A, layer: 1, pos: 24
type: A, layer: 1, pos: 15
type: B, layer: 1, pos: 3
type: B, layer: 1, pos: 15
type: A, layer: 1, pos: 3
type: B, layer: 1, pos: 10
type: A, layer: 1, pos: 10
type: B, layer: 1, pos: 24
type: A, layer: 1, pos: 14
type: B, layer: 1, pos: 14
type: B, layer: 1, pos: 40
type: A, layer: 1, pos: 40

Time for candidate selection: 0.24 seconds

### Candidate
type: B, layer: 1, pos: 8

### Candidate
type: A, layer: 1, pos: 8

### Candidate
type: B, layer: 1, pos: 30

## Relational analysis of NS_A1_B2_B1_A1_B1

### Relational analysis result of NS_A1_B2_B1_A1_B1
Status: Status.UNKNOWN
Output dim: 0, lower bound: -915.9973822, upper bound: 914.5537350
time: 0.45 seconds

## Relational analysis of NS_A1_B2_B1_A1_B2

### Relational analysis result of NS_A1_B2_B1_A1_B2
Status: Status.UNKNOWN
Output dim: 0, lower bound: -916.2819361, upper bound: 914.9312023
time: 0.44 seconds

## BFS NS instance: NS_A1_B2_B1_A2

### Backsubstitution after applying NS history:
0: -502.7959900, 326.4364929, -488.3984070, 313.5505981, -816.3465576, 814.8348389
1: -38.1636848, 29.1213188, -36.8722458, 28.1667194, -66.3304062, 65.9935608
2: -31.9113941, 43.5109329, -30.8656483, 41.8754425, -73.7868347, 74.3765640
3: -36.0328712, 69.5438232, -34.6121788, 67.0116425, -103.0445099, 104.1560059
4: -27.3170071, 43.5654984, -26.5074844, 42.0064163, -69.3234253, 70.0729828

Time for backsubstitution: 2.85 seconds

### NS candidates at layer 1
type: B, layer: 1, pos: 8
type: A, layer: 1, pos: 8
type: B, layer: 1, pos: 5
type: A, layer: 1, pos: 5
type: B, layer: 1, pos: 21
type: A, layer: 1, pos: 21
type: A, layer: 1, pos: 38
type: B, layer: 1, pos: 38
type: A, layer: 1, pos: 47
type: A, layer: 1, pos: 10
type: A, layer: 1, pos: 35
type: B, layer: 1, pos: 47
type: B, layer: 1, pos: 35
type: B, layer: 1, pos: 15
type: B, layer: 1, pos: 10
type: B, layer: 1, pos: 3
type: A, layer: 1, pos: 3
type: A, layer: 1, pos: 15
type: A, layer: 1, pos: 30
type: B, layer: 1, pos: 30
type: A, layer: 1, pos: 1
type: A, layer: 1, pos: 24
type: B, layer: 1, pos: 1
type: B, layer: 1, pos: 24
type: A, layer: 1, pos: 14
type: B, layer: 1, pos: 14
type: B, layer: 1, pos: 40
type: A, layer: 1, pos: 40

Time for candidate selection: 0.24 seconds

### Candidate
type: B, layer: 1, pos: 8

### Candidate
type: A, layer: 1, pos: 8

### Candidate
type: B, layer: 1, pos: 5

## Relational analysis of NS_A1_B2_B1_A2_B1
Optimization infeasible because this subproblem isn't reachable.

### Candidate
type: A, layer: 1, pos: 5

### Candidate
type: B, layer: 1, pos: 21

## Relational analysis of NS_A1_B2_B1_A2_B1
Optimization infeasible because this subproblem isn't reachable.

### Candidate
type: A, layer: 1, pos: 21

### Candidate
type: A, layer: 1, pos: 38

## Relational analysis of NS_A1_B2_B1_A2_A1
Optimization infeasible because this subproblem isn't reachable.

### Candidate
type: B, layer: 1, pos: 38

### Candidate
type: A, layer: 1, pos: 47

## Relational analysis of NS_A1_B2_B1_A2_A1

### Relational analysis result of NS_A1_B2_B1_A2_A1
Status: Status.UNKNOWN
Output dim: 0, lower bound: -912.7943344, upper bound: 913.7259685
time: 0.50 seconds

## Relational analysis of NS_A1_B2_B1_A2_A2

### Relational analysis result of NS_A1_B2_B1_A2_A2
Status: Status.VERIFIED
Output dim: 0, lower bound: -912.7943344, upper bound: 912.8298473
time: 0.50 seconds

## BFS NS instance: NS_A1_B2_B2_A1

### Backsubstitution after applying NS history:
0: -462.3272095, 293.1559448, -520.1079102, 338.3668213, -800.6940308, 813.2636719
1: -34.5358887, 26.5900002, -39.5576324, 30.1827927, -64.7186661, 66.1476212
2: -29.3298416, 38.9291420, -32.7032204, 45.3268547, -74.6566925, 71.6323624
3: -32.8668022, 62.9823685, -36.9639473, 72.1907349, -105.0575409, 99.9463196
4: -25.1946354, 39.0688972, -28.1038265, 45.3841934, -70.5788116, 67.1727142

Time for backsubstitution: 2.84 seconds

### NS candidates at layer 1
type: B, layer: 1, pos: 8
type: A, layer: 1, pos: 8
type: B, layer: 1, pos: 30
type: B, layer: 1, pos: 38
type: A, layer: 1, pos: 38
type: A, layer: 1, pos: 21
type: A, layer: 1, pos: 30
type: B, layer: 1, pos: 21
type: B, layer: 1, pos: 47
type: B, layer: 1, pos: 35
type: A, layer: 1, pos: 47
type: B, layer: 1, pos: 5
type: A, layer: 1, pos: 35
type: A, layer: 1, pos: 5
type: A, layer: 1, pos: 1
type: B, layer: 1, pos: 1
type: A, layer: 1, pos: 24
type: A, layer: 1, pos: 15
type: B, layer: 1, pos: 3
type: A, layer: 1, pos: 3
type: B, layer: 1, pos: 10
type: B, layer: 1, pos: 24
type: A, layer: 1, pos: 10
type: B, layer: 1, pos: 15
type: A, layer: 1, pos: 14
type: B, layer: 1, pos: 14
type: B, layer: 1, pos: 40
type: A, layer: 1, pos: 40

Time for candidate selection: 0.24 seconds

### Candidate
type: B, layer: 1, pos: 8

### Candidate
type: A, layer: 1, pos: 8

### Candidate
type: B, layer: 1, pos: 30

## Relational analysis of NS_A1_B2_B2_A1_B1

### Relational analysis result of NS_A1_B2_B2_A1_B1
Status: Status.UNKNOWN
Output dim: 0, lower bound: -915.9973822, upper bound: 914.7131008
time: 0.46 seconds

## Relational analysis of NS_A1_B2_B2_A1_B2

### Relational analysis result of NS_A1_B2_B2_A1_B2
Status: Status.UNKNOWN
Output dim: 0, lower bound: -916.2819355, upper bound: 914.9312023
time: 0.46 seconds

## BFS NS instance: NS_A1_B2_B2_A2

### Backsubstitution after applying NS history:
0: -502.7959900, 326.4364929, -520.1079102, 338.3668213, -841.1626587, 846.5441895
1: -38.1636848, 29.1213188, -39.5576324, 30.1827927, -68.3464737, 68.6789474
2: -31.9113941, 43.5109329, -32.7032204, 45.3268547, -77.2382507, 76.2141571
3: -36.0328712, 69.5438232, -36.9639473, 72.1907349, -108.2236023, 106.5077667
4: -27.3170071, 43.5654984, -28.1038265, 45.3841934, -72.7011948, 71.6693268

Time for backsubstitution: 2.85 seconds

### NS candidates at layer 1
type: B, layer: 1, pos: 8
type: A, layer: 1, pos: 8
type: B, layer: 1, pos: 5
type: A, layer: 1, pos: 5
type: A, layer: 1, pos: 21
type: B, layer: 1, pos: 21
type: B, layer: 1, pos: 38
type: A, layer: 1, pos: 38
type: B, layer: 1, pos: 47
type: A, layer: 1, pos: 35
type: A, layer: 1, pos: 47
type: B, layer: 1, pos: 35
type: B, layer: 1, pos: 10
type: A, layer: 1, pos: 10
type: B, layer: 1, pos: 30
type: B, layer: 1, pos: 3
type: A, layer: 1, pos: 15
type: A, layer: 1, pos: 3
type: A, layer: 1, pos: 30
type: B, layer: 1, pos: 15
type: A, layer: 1, pos: 1
type: A, layer: 1, pos: 24
type: B, layer: 1, pos: 1
type: B, layer: 1, pos: 24
type: A, layer: 1, pos: 14
type: B, layer: 1, pos: 14
type: B, layer: 1, pos: 40
type: A, layer: 1, pos: 40

Time for candidate selection: 0.24 seconds

### Candidate
type: B, layer: 1, pos: 8

### Candidate
type: A, layer: 1, pos: 8

### Candidate
type: B, layer: 1, pos: 5

## Relational analysis of NS_A1_B2_B2_A2_B1
Optimization infeasible because this subproblem isn't reachable.

### Candidate
type: A, layer: 1, pos: 5

### Candidate
type: A, layer: 1, pos: 21

## Relational analysis of NS_A1_B2_B2_A2_A1
Optimization infeasible because this subproblem isn't reachable.

### Candidate
type: B, layer: 1, pos: 21

### Candidate
type: B, layer: 1, pos: 38

## Relational analysis of NS_A1_B2_B2_A2_B1
Optimization infeasible because this subproblem isn't reachable.

### Candidate
type: A, layer: 1, pos: 38

### Candidate
type: B, layer: 1, pos: 47

## Relational analysis of NS_A1_B2_B2_A2_B1

### Relational analysis result of NS_A1_B2_B2_A2_B1
Status: Status.UNKNOWN
Output dim: 0, lower bound: -915.1771303, upper bound: 912.9773291
time: 0.50 seconds

## Relational analysis of NS_A1_B2_B2_A2_B2

### Relational analysis result of NS_A1_B2_B2_A2_B2
Status: Status.VERIFIED
Output dim: 0, lower bound: -912.7943344, upper bound: 912.9773291
time: 0.50 seconds

## BFS NS instance: NS_A2_B1_A1_B1

### Backsubstitution after applying NS history:
0: -489.2505188, 314.2459106, -462.3272095, 293.1559448, -782.4064941, 776.5731201
1: -36.9483871, 28.2214146, -34.5358887, 26.5900002, -63.5383873, 62.7573013
2: -30.9117508, 41.9644508, -29.3298416, 38.9291420, -69.8408966, 71.2942963
3: -34.6661301, 67.1468887, -32.8668022, 62.9823685, -97.6484985, 100.0136871
4: -26.5482178, 42.0926247, -25.1946354, 39.0688972, -65.6171112, 67.2872543

Time for backsubstitution: 2.86 seconds

### NS candidates at layer 1
type: A, layer: 1, pos: 8
type: B, layer: 1, pos: 8
type: A, layer: 1, pos: 30
type: A, layer: 1, pos: 38
type: B, layer: 1, pos: 38
type: B, layer: 1, pos: 30
type: B, layer: 1, pos: 21
type: A, layer: 1, pos: 21
type: A, layer: 1, pos: 47
type: B, layer: 1, pos: 47
type: A, layer: 1, pos: 35
type: A, layer: 1, pos: 5
type: B, layer: 1, pos: 35
type: B, layer: 1, pos: 5
type: B, layer: 1, pos: 1
type: A, layer: 1, pos: 1
type: B, layer: 1, pos: 24
type: B, layer: 1, pos: 15
type: A, layer: 1, pos: 3
type: A, layer: 1, pos: 15
type: B, layer: 1, pos: 3
type: A, layer: 1, pos: 10
type: B, layer: 1, pos: 10
type: A, layer: 1, pos: 24
type: B, layer: 1, pos: 14
type: A, layer: 1, pos: 14
type: A, layer: 1, pos: 40
type: B, layer: 1, pos: 40

Time for candidate selection: 0.24 seconds

### Candidate
type: A, layer: 1, pos: 8

### Candidate
type: B, layer: 1, pos: 8

### Candidate
type: A, layer: 1, pos: 30

## Relational analysis of NS_A2_B1_A1_B1_A1

### Relational analysis result of NS_A2_B1_A1_B1_A1
Status: Status.UNKNOWN
Output dim: 0, lower bound: -914.5537350, upper bound: 915.9973822
time: 0.47 seconds

## Relational analysis of NS_A2_B1_A1_B1_A2

### Relational analysis result of NS_A2_B1_A1_B1_A2
Status: Status.UNKNOWN
Output dim: 0, lower bound: -914.9312023, upper bound: 916.2819355
time: 0.43 seconds

## BFS NS instance: NS_A2_B1_A1_B2

### Backsubstitution after applying NS history:
0: -489.2505188, 314.2459106, -502.7959900, 326.4364929, -815.6870117, 817.0418091
1: -36.9483871, 28.2214146, -38.1636848, 29.1213188, -66.0697021, 66.3851013
2: -30.9117508, 41.9644508, -31.9113941, 43.5109329, -74.4226837, 73.8758469
3: -34.6661301, 67.1468887, -36.0328712, 69.5438232, -104.2099533, 103.1797638
4: -26.5482178, 42.0926247, -27.3170071, 43.5654984, -70.1137161, 69.4096298

Time for backsubstitution: 2.88 seconds

### NS candidates at layer 1
type: A, layer: 1, pos: 8
type: B, layer: 1, pos: 8
type: A, layer: 1, pos: 5
type: B, layer: 1, pos: 5
type: A, layer: 1, pos: 21
type: B, layer: 1, pos: 21
type: B, layer: 1, pos: 38
type: A, layer: 1, pos: 38
type: B, layer: 1, pos: 47
type: B, layer: 1, pos: 10
type: B, layer: 1, pos: 35
type: A, layer: 1, pos: 47
type: A, layer: 1, pos: 35
type: A, layer: 1, pos: 15
type: A, layer: 1, pos: 10
type: A, layer: 1, pos: 3
type: B, layer: 1, pos: 3
type: B, layer: 1, pos: 15
type: B, layer: 1, pos: 30
type: A, layer: 1, pos: 30
type: B, layer: 1, pos: 1
type: B, layer: 1, pos: 24
type: A, layer: 1, pos: 1
type: A, layer: 1, pos: 24
type: B, layer: 1, pos: 14
type: A, layer: 1, pos: 14
type: A, layer: 1, pos: 40
type: B, layer: 1, pos: 40

Time for candidate selection: 0.24 seconds

### Candidate
type: A, layer: 1, pos: 8

### Candidate
type: B, layer: 1, pos: 8

### Candidate
type: A, layer: 1, pos: 5

## Relational analysis of NS_A2_B1_A1_B2_A1
Optimization infeasible because this subproblem isn't reachable.

### Candidate
type: B, layer: 1, pos: 5

### Candidate
type: A, layer: 1, pos: 21

## Relational analysis of NS_A2_B1_A1_B2_A1
Optimization infeasible because this subproblem isn't reachable.

### Candidate
type: B, layer: 1, pos: 21

### Candidate
type: B, layer: 1, pos: 38

## Relational analysis of NS_A2_B1_A1_B2_B1
Optimization infeasible because this subproblem isn't reachable.

### Candidate
type: A, layer: 1, pos: 38

### Candidate
type: B, layer: 1, pos: 47

## Relational analysis of NS_A2_B1_A1_B2_B1

### Relational analysis result of NS_A2_B1_A1_B2_B1
Status: Status.UNKNOWN
Output dim: 0, lower bound: -913.7259685, upper bound: 912.7943344
time: 0.49 seconds

## Relational analysis of NS_A2_B1_A1_B2_B2

### Relational analysis result of NS_A2_B1_A1_B2_B2
Status: Status.VERIFIED
Output dim: 0, lower bound: -912.7683715, upper bound: 912.7943344
time: 0.46 seconds

## BFS NS instance: NS_A2_B1_A2_B1

### Backsubstitution after applying NS history:
0: -520.7755737, 338.8567810, -462.3272095, 293.1559448, -813.9315186, 801.1839600
1: -39.6113853, 30.2250767, -34.5358887, 26.5900002, -66.2013702, 64.7609482
2: -32.7410431, 45.3893623, -29.3298416, 38.9291420, -71.6701813, 74.7192078
3: -37.0099106, 72.2946548, -32.8668022, 62.9823685, -99.9922791, 105.1614532
4: -28.1395817, 45.4455528, -25.1946354, 39.0688972, -67.2084579, 70.6401672

Time for backsubstitution: 2.88 seconds

### NS candidates at layer 1
type: A, layer: 1, pos: 8
type: B, layer: 1, pos: 8
type: A, layer: 1, pos: 30
type: A, layer: 1, pos: 38
type: B, layer: 1, pos: 38
type: B, layer: 1, pos: 21
type: B, layer: 1, pos: 30
type: A, layer: 1, pos: 21
type: A, layer: 1, pos: 47
type: A, layer: 1, pos: 35
type: B, layer: 1, pos: 47
type: A, layer: 1, pos: 5
type: B, layer: 1, pos: 35
type: B, layer: 1, pos: 5
type: B, layer: 1, pos: 1
type: A, layer: 1, pos: 1
type: B, layer: 1, pos: 24
type: B, layer: 1, pos: 15
type: A, layer: 1, pos: 3
type: B, layer: 1, pos: 3
type: A, layer: 1, pos: 10
type: A, layer: 1, pos: 24
type: B, layer: 1, pos: 10
type: A, layer: 1, pos: 15
type: B, layer: 1, pos: 14
type: A, layer: 1, pos: 14
type: A, layer: 1, pos: 40
type: B, layer: 1, pos: 40

Time for candidate selection: 0.24 seconds

### Candidate
type: A, layer: 1, pos: 8

### Candidate
type: B, layer: 1, pos: 8

### Candidate
type: A, layer: 1, pos: 30

## Relational analysis of NS_A2_B1_A2_B1_A1

### Relational analysis result of NS_A2_B1_A2_B1_A1
Status: Status.UNKNOWN
Output dim: 0, lower bound: -914.7131008, upper bound: 916.1800588
time: 0.47 seconds

## Relational analysis of NS_A2_B1_A2_B1_A2

### Relational analysis result of NS_A2_B1_A2_B1_A2
Status: Status.UNKNOWN
Output dim: 0, lower bound: -914.9312023, upper bound: 916.2992972
time: 0.47 seconds

## BFS NS instance: NS_A2_B1_A2_B2

### Backsubstitution after applying NS history:
0: -520.7755737, 338.8567810, -502.7959900, 326.4364929, -847.2119751, 841.6526489
1: -39.6113853, 30.2250767, -38.1636848, 29.1213188, -68.7327042, 68.3887482
2: -32.7410431, 45.3893623, -31.9113941, 43.5109329, -76.2519455, 77.3007584
3: -37.0099106, 72.2946548, -36.0328712, 69.5438232, -106.5537338, 108.3275223
4: -28.1395817, 45.4455528, -27.3170071, 43.5654984, -71.7050705, 72.7625580

Time for backsubstitution: 2.88 seconds

### NS candidates at layer 1
type: A, layer: 1, pos: 8
type: B, layer: 1, pos: 8
type: A, layer: 1, pos: 5
type: B, layer: 1, pos: 5
type: B, layer: 1, pos: 21
type: A, layer: 1, pos: 21
type: A, layer: 1, pos: 38
type: B, layer: 1, pos: 38
type: A, layer: 1, pos: 47
type: B, layer: 1, pos: 35
type: B, layer: 1, pos: 47
type: A, layer: 1, pos: 35
type: A, layer: 1, pos: 10
type: B, layer: 1, pos: 10
type: A, layer: 1, pos: 30
type: A, layer: 1, pos: 3
type: B, layer: 1, pos: 15
type: B, layer: 1, pos: 3
type: B, layer: 1, pos: 30
type: A, layer: 1, pos: 15
type: B, layer: 1, pos: 1
type: B, layer: 1, pos: 24
type: A, layer: 1, pos: 1
type: A, layer: 1, pos: 24
type: B, layer: 1, pos: 14
type: A, layer: 1, pos: 14
type: A, layer: 1, pos: 40
type: B, layer: 1, pos: 40

Time for candidate selection: 0.24 seconds

### Candidate
type: A, layer: 1, pos: 8

### Candidate
type: B, layer: 1, pos: 8

### Candidate
type: A, layer: 1, pos: 5

## Relational analysis of NS_A2_B1_A2_B2_A1
Optimization infeasible because this subproblem isn't reachable.

### Candidate
type: B, layer: 1, pos: 5

### Candidate
type: B, layer: 1, pos: 21

## Relational analysis of NS_A2_B1_A2_B2_B1
Optimization infeasible because this subproblem isn't reachable.

### Candidate
type: A, layer: 1, pos: 21

### Candidate
type: A, layer: 1, pos: 38

## Relational analysis of NS_A2_B1_A2_B2_A1
Optimization infeasible because this subproblem isn't reachable.

### Candidate
type: B, layer: 1, pos: 38

### Candidate
type: A, layer: 1, pos: 47

## Relational analysis of NS_A2_B1_A2_B2_A1

### Relational analysis result of NS_A2_B1_A2_B2_A1
Status: Status.UNKNOWN
Output dim: 0, lower bound: -912.7683715, upper bound: 915.2814492
time: 0.47 seconds

## Relational analysis of NS_A2_B1_A2_B2_A2

### Relational analysis result of NS_A2_B1_A2_B2_A2
Status: Status.UNKNOWN
Output dim: 0, lower bound: -912.7683715, upper bound: 913.1631148
time: 0.52 seconds

## BFS NS instance: NS_A2_B2_A1_B1

### Backsubstitution after applying NS history:
0: -495.0452271, 318.1841431, -483.1653442, 308.8505249, -803.8957520, 801.3494263
1: -37.4236259, 28.5858040, -36.3776741, 27.8167915, -65.2404175, 64.9634705
2: -31.0599270, 42.6855927, -30.5187321, 41.2554703, -72.3153839, 73.2043152
3: -34.9849892, 68.2513962, -34.2021103, 66.1254959, -101.1104889, 102.4534988
4: -26.6837177, 42.7664680, -26.2118130, 41.3911629, -68.0748749, 68.9782791

Time for backsubstitution: 2.88 seconds

### NS candidates at layer 1
type: B, layer: 1, pos: 5
type: A, layer: 1, pos: 5
type: A, layer: 1, pos: 8
type: B, layer: 1, pos: 8
type: A, layer: 1, pos: 18
type: A, layer: 1, pos: 38
type: B, layer: 1, pos: 35
type: B, layer: 1, pos: 38
type: B, layer: 1, pos: 3
type: A, layer: 1, pos: 3
type: A, layer: 1, pos: 10
type: B, layer: 1, pos: 21
type: A, layer: 1, pos: 21
type: B, layer: 1, pos: 10
type: A, layer: 1, pos: 47
type: A, layer: 1, pos: 30
type: B, layer: 1, pos: 47
type: B, layer: 1, pos: 30
type: B, layer: 1, pos: 15
type: B, layer: 1, pos: 1
type: A, layer: 1, pos: 15
type: B, layer: 1, pos: 24
type: A, layer: 1, pos: 1
type: A, layer: 1, pos: 24
type: B, layer: 1, pos: 14
type: A, layer: 1, pos: 14
type: B, layer: 1, pos: 40
type: A, layer: 1, pos: 40

Time for candidate selection: 0.24 seconds

### Candidate
type: B, layer: 1, pos: 5

## Relational analysis of NS_A2_B2_A1_B1_B1
Optimization infeasible because this subproblem isn't reachable.

### Candidate
type: A, layer: 1, pos: 5

### Candidate
type: A, layer: 1, pos: 8

## Relational analysis of NS_A2_B2_A1_B1_A1
Optimization infeasible because this subproblem isn't reachable.

### Candidate
type: B, layer: 1, pos: 8

### Candidate
type: A, layer: 1, pos: 18

## Relational analysis of NS_A2_B2_A1_B1_A1

### Relational analysis result of NS_A2_B2_A1_B1_A1
Status: Status.UNKNOWN
Output dim: 0, lower bound: -914.7940733, upper bound: 914.1047483
time: 0.49 seconds

## Relational analysis of NS_A2_B2_A1_B1_A2

### Relational analysis result of NS_A2_B2_A1_B1_A2
Status: Status.UNKNOWN
Output dim: 0, lower bound: -914.7940757, upper bound: 914.7501866
time: 0.47 seconds

## BFS NS instance: NS_A2_B2_A1_B2

### Backsubstitution after applying NS history:
0: -504.6235962, 325.8970947, -513.9433594, 333.2450562, -837.8686523, 839.8404541
1: -38.2529411, 29.1801548, -39.0157394, 29.7766666, -68.0295944, 68.1958923
2: -31.7173271, 43.6959000, -32.2850876, 44.6390572, -76.3563843, 75.9809875
3: -35.8032913, 69.7666550, -36.4700165, 71.1604691, -106.9637604, 106.2366714
4: -27.2554989, 43.7480659, -27.7318687, 44.6966934, -71.9521637, 71.4799271

Time for backsubstitution: 2.88 seconds

### NS candidates at layer 1
type: A, layer: 1, pos: 18
type: B, layer: 1, pos: 5
type: B, layer: 1, pos: 8
type: A, layer: 1, pos: 5
type: A, layer: 1, pos: 8
type: A, layer: 1, pos: 38
type: B, layer: 1, pos: 38
type: B, layer: 1, pos: 35
type: B, layer: 1, pos: 10
type: B, layer: 1, pos: 3
type: A, layer: 1, pos: 3
type: A, layer: 1, pos: 21
type: B, layer: 1, pos: 21
type: A, layer: 1, pos: 47
type: A, layer: 1, pos: 10
type: A, layer: 1, pos: 30
type: B, layer: 1, pos: 47
type: B, layer: 1, pos: 30
type: B, layer: 1, pos: 15
type: A, layer: 1, pos: 15
type: B, layer: 1, pos: 1
type: A, layer: 1, pos: 1
type: B, layer: 1, pos: 24
type: A, layer: 1, pos: 24
type: B, layer: 1, pos: 14
type: A, layer: 1, pos: 14
type: B, layer: 1, pos: 40
type: A, layer: 1, pos: 40

Time for candidate selection: 0.24 seconds

### Candidate
type: A, layer: 1, pos: 18

## Relational analysis of NS_A2_B2_A1_B2_A1

### Relational analysis result of NS_A2_B2_A1_B2_A1
Status: Status.UNKNOWN
Output dim: 0, lower bound: -914.7940757, upper bound: 914.1047483
time: 0.45 seconds

## Relational analysis of NS_A2_B2_A1_B2_A2

### Relational analysis result of NS_A2_B2_A1_B2_A2
Status: Status.UNKNOWN
Output dim: 0, lower bound: -914.7940757, upper bound: 915.1457901
time: 0.50 seconds

## BFS NS instance: NS_A2_B2_A2_B1

### Backsubstitution after applying NS history:
0: -508.0582275, 328.6637573, -488.7510376, 313.8247070, -821.8829346, 817.4144897
1: -38.5196991, 29.4332695, -36.9031830, 28.1892624, -66.7089539, 66.3364563
2: -31.8580914, 44.0472260, -30.8801479, 41.9077339, -73.7658234, 74.9273529
3: -35.9649582, 70.2625427, -34.6289291, 67.0647278, -103.0296631, 104.8914719
4: -27.3524818, 44.1296921, -26.5209370, 42.0369606, -69.3894424, 70.6506271

Time for backsubstitution: 2.90 seconds

### NS candidates at layer 1
type: A, layer: 1, pos: 5
type: B, layer: 1, pos: 5
type: B, layer: 1, pos: 35
type: A, layer: 1, pos: 8
type: B, layer: 1, pos: 8
type: A, layer: 1, pos: 18
type: A, layer: 1, pos: 38
type: B, layer: 1, pos: 38
type: A, layer: 1, pos: 3
type: A, layer: 1, pos: 10
type: B, layer: 1, pos: 10
type: B, layer: 1, pos: 3
type: B, layer: 1, pos: 21
type: A, layer: 1, pos: 21
type: A, layer: 1, pos: 47
type: B, layer: 1, pos: 47
type: A, layer: 1, pos: 30
type: B, layer: 1, pos: 30
type: B, layer: 1, pos: 15
type: B, layer: 1, pos: 1
type: A, layer: 1, pos: 15
type: B, layer: 1, pos: 24
type: A, layer: 1, pos: 1
type: A, layer: 1, pos: 24
type: B, layer: 1, pos: 14
type: A, layer: 1, pos: 14
type: B, layer: 1, pos: 40
type: A, layer: 1, pos: 40

Time for candidate selection: 0.24 seconds

### Candidate
type: A, layer: 1, pos: 5

## Relational analysis of NS_A2_B2_A2_B1_A1
Optimization infeasible because this subproblem isn't reachable.

### Candidate
type: B, layer: 1, pos: 5

### Candidate
type: B, layer: 1, pos: 35

## Relational analysis of NS_A2_B2_A2_B1_B1

### Relational analysis result of NS_A2_B2_A2_B1_B1
Status: Status.UNKNOWN
Output dim: 0, lower bound: -914.1047483, upper bound: 915.9322522
time: 0.50 seconds

## Relational analysis of NS_A2_B2_A2_B1_B2

### Relational analysis result of NS_A2_B2_A2_B1_B2
Status: Status.UNKNOWN
Output dim: 0, lower bound: -914.1047483, upper bound: 916.5976784
time: 0.48 seconds

## BFS NS instance: NS_A2_B2_A2_B2

### Backsubstitution after applying NS history:
0: -517.9037476, 336.6044922, -520.1938477, 338.3971863, -856.3009033, 856.7982178
1: -39.3679237, 30.0452213, -39.5616646, 30.1879539, -69.5558777, 69.6068878
2: -32.5624428, 45.0869827, -32.7032356, 45.3272285, -77.8896637, 77.7902222
3: -36.7962952, 71.8380203, -36.9652023, 72.2003250, -108.9966202, 108.8032227
4: -27.9848843, 45.1464348, -28.1057796, 45.3838654, -73.3687515, 73.2522125

Time for backsubstitution: 2.91 seconds

### NS candidates at layer 1
type: A, layer: 1, pos: 18
type: B, layer: 1, pos: 5
type: A, layer: 1, pos: 5
type: B, layer: 1, pos: 8
type: B, layer: 1, pos: 35
type: A, layer: 1, pos: 8
type: A, layer: 1, pos: 38
type: B, layer: 1, pos: 38
type: B, layer: 1, pos: 10
type: B, layer: 1, pos: 3
type: A, layer: 1, pos: 3
type: B, layer: 1, pos: 21
type: A, layer: 1, pos: 21
type: A, layer: 1, pos: 10
type: A, layer: 1, pos: 47
type: B, layer: 1, pos: 47
type: A, layer: 1, pos: 30
type: B, layer: 1, pos: 30
type: A, layer: 1, pos: 1
type: B, layer: 1, pos: 15
type: B, layer: 1, pos: 1
type: A, layer: 1, pos: 15
type: B, layer: 1, pos: 24
type: A, layer: 1, pos: 24
type: A, layer: 1, pos: 14
type: B, layer: 1, pos: 14
type: B, layer: 1, pos: 40
type: A, layer: 1, pos: 40

Time for candidate selection: 0.24 seconds

### Candidate
type: A, layer: 1, pos: 18

## Relational analysis of NS_A2_B2_A2_B2_A1

### Relational analysis result of NS_A2_B2_A2_B2_A1
Status: Status.UNKNOWN
Output dim: 0, lower bound: -916.5978581, upper bound: 916.5482256
time: 0.44 seconds

## Relational analysis of NS_A2_B2_A2_B2_A2

### Relational analysis result of NS_A2_B2_A2_B2_A2
Status: Status.UNKNOWN
Output dim: 0, lower bound: -916.5978581, upper bound: 916.6077273
time: 0.44 seconds

## Summary of splitting at layer (split count: 4)
- Time for NS candidates: 4.04 seconds
NS_A1_B1_A1_B1_A1, status: Status.UNKNOWN, split count: 5, time: 4.04
Output dim: 0, lower bound: -912.7683715, upper bound: 914.3765485
NS_A1_B1_A1_B1_A2, status: Status.VERIFIED, split count: 5, time: 4.04
Output dim: 0, lower bound: -912.7683715, upper bound: 912.7683715
NS_A1_B1_A1_B2_B1, status: Status.UNKNOWN, split count: 5, time: 4.04
Output dim: 0, lower bound: -911.0559777, upper bound: 914.4599584
NS_A1_B1_A1_B2_B2, status: Status.UNKNOWN, split count: 5, time: 4.04
Output dim: 0, lower bound: -914.8548589, upper bound: 914.8657075
NS_A1_B1_A2_B1_A1, status: Status.UNKNOWN, split count: 5, time: 4.04
Output dim: 0, lower bound: -914.4599584, upper bound: 912.3608761
NS_A1_B1_A2_B1_A2, status: Status.UNKNOWN, split count: 5, time: 4.04
Output dim: 0, lower bound: -914.8548591, upper bound: 914.8548589
NS_A1_B1_A2_B2_A1, status: Status.UNKNOWN, split count: 5, time: 4.04
Output dim: 0, lower bound: -914.4599584, upper bound: 913.5295566
NS_A1_B1_A2_B2_A2, status: Status.UNKNOWN, split count: 5, time: 4.04
Output dim: 0, lower bound: -914.8548591, upper bound: 914.8548589
NS_A1_B2_B1_A1_B1, status: Status.UNKNOWN, split count: 5, time: 4.04
Output dim: 0, lower bound: -915.9973822, upper bound: 914.5537350
NS_A1_B2_B1_A1_B2, status: Status.UNKNOWN, split count: 5, time: 4.04
Output dim: 0, lower bound: -916.2819361, upper bound: 914.9312023
NS_A1_B2_B1_A2_A1, status: Status.UNKNOWN, split count: 5, time: 4.04
Output dim: 0, lower bound: -912.7943344, upper bound: 913.7259685
NS_A1_B2_B1_A2_A2, status: Status.VERIFIED, split count: 5, time: 4.04
Output dim: 0, lower bound: -912.7943344, upper bound: 912.8298473
NS_A1_B2_B2_A1_B1, status: Status.UNKNOWN, split count: 5, time: 4.04
Output dim: 0, lower bound: -915.9973822, upper bound: 914.7131008
NS_A1_B2_B2_A1_B2, status: Status.UNKNOWN, split count: 5, time: 4.04
Output dim: 0, lower bound: -916.2819355, upper bound: 914.9312023
NS_A1_B2_B2_A2_B1, status: Status.UNKNOWN, split count: 5, time: 4.04
Output dim: 0, lower bound: -915.1771303, upper bound: 912.9773291
NS_A1_B2_B2_A2_B2, status: Status.VERIFIED, split count: 5, time: 4.04
Output dim: 0, lower bound: -912.7943344, upper bound: 912.9773291
NS_A2_B1_A1_B1_A1, status: Status.UNKNOWN, split count: 5, time: 4.04
Output dim: 0, lower bound: -914.5537350, upper bound: 915.9973822
NS_A2_B1_A1_B1_A2, status: Status.UNKNOWN, split count: 5, time: 4.04
Output dim: 0, lower bound: -914.9312023, upper bound: 916.2819355
NS_A2_B1_A1_B2_B1, status: Status.UNKNOWN, split count: 5, time: 4.04
Output dim: 0, lower bound: -913.7259685, upper bound: 912.7943344
NS_A2_B1_A1_B2_B2, status: Status.VERIFIED, split count: 5, time: 4.04
Output dim: 0, lower bound: -912.7683715, upper bound: 912.7943344
NS_A2_B1_A2_B1_A1, status: Status.UNKNOWN, split count: 5, time: 4.04
Output dim: 0, lower bound: -914.7131008, upper bound: 916.1800588
NS_A2_B1_A2_B1_A2, status: Status.UNKNOWN, split count: 5, time: 4.04
Output dim: 0, lower bound: -914.9312023, upper bound: 916.2992972
NS_A2_B1_A2_B2_A1, status: Status.UNKNOWN, split count: 5, time: 4.04
Output dim: 0, lower bound: -912.7683715, upper bound: 915.2814492
NS_A2_B1_A2_B2_A2, status: Status.UNKNOWN, split count: 5, time: 4.04
Output dim: 0, lower bound: -912.7683715, upper bound: 913.1631148
NS_A2_B2_A1_B1_A1, status: Status.UNKNOWN, split count: 5, time: 4.04
Output dim: 0, lower bound: -914.7940733, upper bound: 914.1047483
NS_A2_B2_A1_B1_A2, status: Status.UNKNOWN, split count: 5, time: 4.04
Output dim: 0, lower bound: -914.7940757, upper bound: 914.7501866
NS_A2_B2_A1_B2_A1, status: Status.UNKNOWN, split count: 5, time: 4.04
Output dim: 0, lower bound: -914.7940757, upper bound: 914.1047483
NS_A2_B2_A1_B2_A2, status: Status.UNKNOWN, split count: 5, time: 4.04
Output dim: 0, lower bound: -914.7940757, upper bound: 915.1457901
NS_A2_B2_A2_B1_B1, status: Status.UNKNOWN, split count: 5, time: 4.04
Output dim: 0, lower bound: -914.1047483, upper bound: 915.9322522
NS_A2_B2_A2_B1_B2, status: Status.UNKNOWN, split count: 5, time: 4.04
Output dim: 0, lower bound: -914.1047483, upper bound: 916.5976784
NS_A2_B2_A2_B2_A1, status: Status.UNKNOWN, split count: 5, time: 4.04
Output dim: 0, lower bound: -916.5978581, upper bound: 916.5482256
NS_A2_B2_A2_B2_A2, status: Status.UNKNOWN, split count: 5, time: 4.04
Output dim: 0, lower bound: -916.5978581, upper bound: 916.6077273

## BFS NS instance: NS_A1_B1_A1_B1_A1

### Backsubstitution after applying NS history:
0: -439.2742004, 273.3491821, -462.3272095, 293.1559448, -732.4300537, 735.6763916
1: -32.4280014, 25.1546001, -34.5358887, 26.5900002, -59.0180016, 59.6904831
2: -27.9652081, 36.3489151, -29.3298416, 38.9291420, -66.8943481, 65.6787567
3: -31.2662277, 59.5446854, -32.8668022, 62.9823685, -94.2485962, 92.4114838
4: -24.0656223, 36.5126572, -25.1946354, 39.0688972, -63.1345215, 61.7072868

Time for backsubstitution: 2.87 seconds

### NS candidates at layer 1
type: B, layer: 1, pos: 8
type: A, layer: 1, pos: 8
type: B, layer: 1, pos: 38
type: A, layer: 1, pos: 38
type: A, layer: 1, pos: 21
type: B, layer: 1, pos: 21
type: B, layer: 1, pos: 35
type: A, layer: 1, pos: 35
type: A, layer: 1, pos: 24
type: B, layer: 1, pos: 30
type: B, layer: 1, pos: 10
type: A, layer: 1, pos: 15
type: A, layer: 1, pos: 10
type: B, layer: 1, pos: 5
type: A, layer: 1, pos: 5
type: B, layer: 1, pos: 3
type: B, layer: 1, pos: 47
type: A, layer: 1, pos: 3
type: B, layer: 1, pos: 24
type: A, layer: 1, pos: 1
type: A, layer: 1, pos: 30
type: B, layer: 1, pos: 15
type: B, layer: 1, pos: 1
type: A, layer: 1, pos: 14
type: B, layer: 1, pos: 14
type: A, layer: 1, pos: 40
type: B, layer: 1, pos: 40

Time for candidate selection: 0.24 seconds

### Candidate
type: B, layer: 1, pos: 8

### Candidate
type: A, layer: 1, pos: 8

### Candidate
type: B, layer: 1, pos: 38

### Candidate
type: A, layer: 1, pos: 38

### Candidate
type: A, layer: 1, pos: 21

### Candidate
type: B, layer: 1, pos: 21

### Candidate
type: B, layer: 1, pos: 35

## Relational analysis of NS_A1_B1_A1_B1_A1_B1

### Relational analysis result of NS_A1_B1_A1_B1_A1_B1
Status: Status.UNKNOWN
Output dim: 0, lower bound: -912.7004530, upper bound: 914.4268375
time: 0.49 seconds

## Relational analysis of NS_A1_B1_A1_B1_A1_B2

### Relational analysis result of NS_A1_B1_A1_B1_A1_B2
Status: Status.UNKNOWN
Output dim: 0, lower bound: -912.3628316, upper bound: 914.4189475
time: 0.46 seconds

## BFS NS instance: NS_A1_B1_A1_B2_B1

### Backsubstitution after applying NS history:
0: -452.7433777, 283.5047302, -462.6254578, 289.2396240, -741.9830322, 746.1301880
1: -33.5141373, 25.9835148, -34.1428070, 26.6093521, -60.1234894, 60.1263123
2: -28.6552238, 37.6543694, -29.0894375, 38.4800377, -67.1352463, 66.7438049
3: -32.0763931, 61.4653053, -32.6629601, 62.8527527, -94.9291153, 94.1282654
4: -24.6221142, 37.8069191, -24.9707031, 38.5878029, -63.2099113, 62.7776184

Time for backsubstitution: 2.88 seconds

### NS candidates at layer 1
type: B, layer: 1, pos: 8
type: A, layer: 1, pos: 8
type: B, layer: 1, pos: 38
type: B, layer: 1, pos: 30
type: A, layer: 1, pos: 38
type: A, layer: 1, pos: 21
type: B, layer: 1, pos: 21
type: B, layer: 1, pos: 35
type: A, layer: 1, pos: 35
type: A, layer: 1, pos: 5
type: A, layer: 1, pos: 24
type: A, layer: 1, pos: 15
type: B, layer: 1, pos: 15
type: B, layer: 1, pos: 10
type: A, layer: 1, pos: 1
type: A, layer: 1, pos: 30
type: A, layer: 1, pos: 3
type: B, layer: 1, pos: 47
type: B, layer: 1, pos: 1
type: B, layer: 1, pos: 3
type: A, layer: 1, pos: 47
type: A, layer: 1, pos: 10
type: B, layer: 1, pos: 24
type: B, layer: 1, pos: 40
type: A, layer: 1, pos: 40
type: A, layer: 1, pos: 14
type: B, layer: 1, pos: 14

Time for candidate selection: 0.24 seconds

### Candidate
type: B, layer: 1, pos: 8

### Candidate
type: A, layer: 1, pos: 8

### Candidate
type: B, layer: 1, pos: 38

## Relational analysis of NS_A1_B1_A1_B2_B1_B1
Optimization infeasible because this subproblem isn't reachable.

### Candidate
type: B, layer: 1, pos: 30

## Relational analysis of NS_A1_B1_A1_B2_B1_B1
Optimization infeasible because this subproblem isn't reachable.

### Candidate
type: A, layer: 1, pos: 38

### Candidate
type: A, layer: 1, pos: 21

## Relational analysis of NS_A1_B1_A1_B2_B1_A1
Optimization infeasible because this subproblem isn't reachable.

### Candidate
type: B, layer: 1, pos: 21

### Candidate
type: B, layer: 1, pos: 35

## Relational analysis of NS_A1_B1_A1_B2_B1_B1

### Relational analysis result of NS_A1_B1_A1_B2_B1_B1
Status: Status.UNKNOWN
Output dim: 0, lower bound: -911.7179774, upper bound: 914.0186715
time: 0.47 seconds

## Relational analysis of NS_A1_B1_A1_B2_B1_B2

### Relational analysis result of NS_A1_B1_A1_B2_B1_B2
Status: Status.UNKNOWN
Output dim: 0, lower bound: -912.2883269, upper bound: 914.4298630
time: 0.48 seconds

## BFS NS instance: NS_A1_B1_A1_B2_B2

### Backsubstitution after applying NS history:
0: -462.3272095, 293.1559448, -487.9168091, 314.3346558, -776.6618652, 781.0727539
1: -34.5358887, 26.5900002, -36.8756714, 28.1965370, -62.7324257, 63.4656715
2: -29.3298416, 38.9291420, -30.9391079, 41.8894653, -71.2193069, 69.8682480
3: -32.8668022, 62.9823685, -34.8926201, 67.1765060, -100.0433044, 97.8749847
4: -25.1946354, 39.0688972, -26.5249214, 41.9749985, -67.1696243, 65.5938187

Time for backsubstitution: 2.87 seconds

### NS candidates at layer 1
type: B, layer: 1, pos: 8
type: A, layer: 1, pos: 8
type: A, layer: 1, pos: 5
type: A, layer: 1, pos: 21
type: B, layer: 1, pos: 21
type: B, layer: 1, pos: 38
type: A, layer: 1, pos: 15
type: A, layer: 1, pos: 38
type: B, layer: 1, pos: 47
type: B, layer: 1, pos: 10
type: B, layer: 1, pos: 35
type: A, layer: 1, pos: 47
type: A, layer: 1, pos: 35
type: B, layer: 1, pos: 30
type: B, layer: 1, pos: 15
type: A, layer: 1, pos: 10
type: B, layer: 1, pos: 3
type: A, layer: 1, pos: 3
type: A, layer: 1, pos: 30
type: A, layer: 1, pos: 24
type: B, layer: 1, pos: 24
type: B, layer: 1, pos: 1
type: A, layer: 1, pos: 1
type: A, layer: 1, pos: 14
type: B, layer: 1, pos: 14
type: B, layer: 1, pos: 40
type: A, layer: 1, pos: 40

Time for candidate selection: 0.24 seconds

### Candidate
type: B, layer: 1, pos: 8

### Candidate
type: A, layer: 1, pos: 8

### Candidate
type: A, layer: 1, pos: 5

## Relational analysis of NS_A1_B1_A1_B2_B2_A1

### Relational analysis result of NS_A1_B1_A1_B2_B2_A1
Status: Status.UNKNOWN
Output dim: 0, lower bound: -914.1038146, upper bound: 911.0559777
time: 0.48 seconds

## Relational analysis of NS_A1_B1_A1_B2_B2_A2

### Relational analysis result of NS_A1_B1_A1_B2_B2_A2
Status: Status.UNKNOWN
Output dim: 0, lower bound: -914.1038146, upper bound: 914.8657075
time: 0.48 seconds

## BFS NS instance: NS_A1_B1_A2_B1_A1

### Backsubstitution after applying NS history:
0: -462.6254578, 289.2396240, -452.7433777, 283.5047302, -746.1301880, 741.9830322
1: -34.1428070, 26.6093521, -33.5141373, 25.9835148, -60.1263123, 60.1234894
2: -29.0894375, 38.4800377, -28.6552238, 37.6543694, -66.7438049, 67.1352463
3: -32.6629601, 62.8527527, -32.0763931, 61.4653053, -94.1282654, 94.9291153
4: -24.9707031, 38.5878029, -24.6221142, 37.8069191, -62.7776146, 63.2099113

Time for backsubstitution: 2.88 seconds

### NS candidates at layer 1
type: A, layer: 1, pos: 8
type: B, layer: 1, pos: 8
type: A, layer: 1, pos: 38
type: A, layer: 1, pos: 30
type: B, layer: 1, pos: 38
type: B, layer: 1, pos: 21
type: A, layer: 1, pos: 21
type: A, layer: 1, pos: 35
type: B, layer: 1, pos: 35
type: B, layer: 1, pos: 5
type: B, layer: 1, pos: 24
type: B, layer: 1, pos: 15
type: A, layer: 1, pos: 15
type: A, layer: 1, pos: 10
type: B, layer: 1, pos: 1
type: B, layer: 1, pos: 30
type: B, layer: 1, pos: 3
type: A, layer: 1, pos: 47
type: A, layer: 1, pos: 1
type: A, layer: 1, pos: 3
type: B, layer: 1, pos: 47
type: B, layer: 1, pos: 10
type: A, layer: 1, pos: 24
type: A, layer: 1, pos: 40
type: B, layer: 1, pos: 40
type: B, layer: 1, pos: 14
type: A, layer: 1, pos: 14

Time for candidate selection: 0.24 seconds

### Candidate
type: A, layer: 1, pos: 8

### Candidate
type: B, layer: 1, pos: 8

### Candidate
type: A, layer: 1, pos: 38

## Relational analysis of NS_A1_B1_A2_B1_A1_A1
Optimization infeasible because this subproblem isn't reachable.

### Candidate
type: A, layer: 1, pos: 30

## Relational analysis of NS_A1_B1_A2_B1_A1_A1
Optimization infeasible because this subproblem isn't reachable.

### Candidate
type: B, layer: 1, pos: 38

### Candidate
type: B, layer: 1, pos: 21

## Relational analysis of NS_A1_B1_A2_B1_A1_B1
Optimization infeasible because this subproblem isn't reachable.

### Candidate
type: A, layer: 1, pos: 21

### Candidate
type: A, layer: 1, pos: 35

## Relational analysis of NS_A1_B1_A2_B1_A1_A1

### Relational analysis result of NS_A1_B1_A2_B1_A1_A1
Status: Status.UNKNOWN
Output dim: 0, lower bound: -914.0186715, upper bound: 911.7179774
time: 0.51 seconds

## Relational analysis of NS_A1_B1_A2_B1_A1_A2

### Relational analysis result of NS_A1_B1_A2_B1_A1_A2
Status: Status.UNKNOWN
Output dim: 0, lower bound: -914.4298630, upper bound: 912.2883266
time: 0.46 seconds

## BFS NS instance: NS_A1_B1_A2_B1_A2

### Backsubstitution after applying NS history:
0: -487.9168091, 314.3346558, -462.3272095, 293.1559448, -781.0727539, 776.6618652
1: -36.8756714, 28.1965370, -34.5358887, 26.5900002, -63.4656715, 62.7324257
2: -30.9391079, 41.8894653, -29.3298416, 38.9291420, -69.8682480, 71.2193069
3: -34.8926201, 67.1765060, -32.8668022, 62.9823685, -97.8749847, 100.0433044
4: -26.5249214, 41.9749985, -25.1946354, 39.0688972, -65.5938187, 67.1696243

Time for backsubstitution: 2.89 seconds

### NS candidates at layer 1
type: A, layer: 1, pos: 8
type: B, layer: 1, pos: 8
type: B, layer: 1, pos: 5
type: B, layer: 1, pos: 21
type: A, layer: 1, pos: 21
type: A, layer: 1, pos: 38
type: B, layer: 1, pos: 15
type: B, layer: 1, pos: 38
type: A, layer: 1, pos: 47
type: A, layer: 1, pos: 10
type: A, layer: 1, pos: 35
type: B, layer: 1, pos: 47
type: B, layer: 1, pos: 35
type: A, layer: 1, pos: 30
type: A, layer: 1, pos: 15
type: B, layer: 1, pos: 10
type: A, layer: 1, pos: 3
type: B, layer: 1, pos: 3
type: B, layer: 1, pos: 30
type: B, layer: 1, pos: 24
type: A, layer: 1, pos: 24
type: A, layer: 1, pos: 1
type: B, layer: 1, pos: 1
type: B, layer: 1, pos: 14
type: A, layer: 1, pos: 14
type: A, layer: 1, pos: 40
type: B, layer: 1, pos: 40

Time for candidate selection: 0.24 seconds

### Candidate
type: A, layer: 1, pos: 8

### Candidate
type: B, layer: 1, pos: 8

### Candidate
type: B, layer: 1, pos: 5

## Relational analysis of NS_A1_B1_A2_B1_A2_B1

### Relational analysis result of NS_A1_B1_A2_B1_A2_B1
Status: Status.UNKNOWN
Output dim: 0, lower bound: -911.0559777, upper bound: 914.1038146
time: 0.48 seconds

## Relational analysis of NS_A1_B1_A2_B1_A2_B2

### Relational analysis result of NS_A1_B1_A2_B1_A2_B2
Status: Status.UNKNOWN
Output dim: 0, lower bound: -911.0559777, upper bound: 914.8548591
time: 0.48 seconds

## BFS NS instance: NS_A1_B1_A2_B2_A1

### Backsubstitution after applying NS history:
0: -462.6254578, 289.2396240, -491.0391846, 316.1770935, -778.8025513, 780.2788086
1: -34.1428070, 26.6093521, -37.0626755, 28.3693600, -62.5121651, 63.6720276
2: -29.0894375, 38.4800377, -31.0702171, 42.1192245, -71.2086563, 69.5502548
3: -32.6629601, 62.8527527, -35.0394287, 67.5606308, -100.2235870, 97.8921432
4: -24.9707031, 38.5878029, -26.6566658, 42.2071419, -67.1778412, 65.2444687

Time for backsubstitution: 2.90 seconds

### NS candidates at layer 1
type: A, layer: 1, pos: 8
type: B, layer: 1, pos: 8
type: B, layer: 1, pos: 5
type: B, layer: 1, pos: 21
type: A, layer: 1, pos: 21
type: A, layer: 1, pos: 38
type: B, layer: 1, pos: 38
type: A, layer: 1, pos: 47
type: B, layer: 1, pos: 15
type: B, layer: 1, pos: 35
type: B, layer: 1, pos: 47
type: A, layer: 1, pos: 30
type: A, layer: 1, pos: 10
type: A, layer: 1, pos: 35
type: A, layer: 1, pos: 15
type: B, layer: 1, pos: 10
type: B, layer: 1, pos: 3
type: B, layer: 1, pos: 30
type: A, layer: 1, pos: 3
type: B, layer: 1, pos: 24
type: B, layer: 1, pos: 1
type: A, layer: 1, pos: 1
type: A, layer: 1, pos: 24
type: B, layer: 1, pos: 14
type: A, layer: 1, pos: 14
type: A, layer: 1, pos: 40
type: B, layer: 1, pos: 40

Time for candidate selection: 0.24 seconds

### Candidate
type: A, layer: 1, pos: 8

### Candidate
type: B, layer: 1, pos: 8

### Candidate
type: B, layer: 1, pos: 5

## Relational analysis of NS_A1_B1_A2_B2_A1_B1

### Relational analysis result of NS_A1_B1_A2_B2_A1_B1
Status: Status.UNKNOWN
Output dim: 0, lower bound: -913.5295563, upper bound: 913.5295563
time: 0.49 seconds

## Relational analysis of NS_A1_B1_A2_B2_A1_B2

### Relational analysis result of NS_A1_B1_A2_B2_A1_B2
Status: Status.UNKNOWN
Output dim: 0, lower bound: -913.5295563, upper bound: 913.5295563
time: 0.51 seconds

## BFS NS instance: NS_A1_B1_A2_B2_A2

### Backsubstitution after applying NS history:
0: -487.9168091, 314.3346558, -502.7959900, 326.4364929, -814.3532715, 817.1304932
1: -36.8756714, 28.1965370, -38.1636848, 29.1213188, -65.9969940, 66.3602142
2: -30.9391079, 41.8894653, -31.9113941, 43.5109329, -74.4500351, 73.8008575
3: -34.8926201, 67.1765060, -36.0328712, 69.5438232, -104.4364471, 103.2093658
4: -26.5249214, 41.9749985, -27.3170071, 43.5654984, -70.0904236, 69.2920074

Time for backsubstitution: 2.90 seconds

### NS candidates at layer 1
type: B, layer: 1, pos: 8
type: A, layer: 1, pos: 8
type: B, layer: 1, pos: 5
type: B, layer: 1, pos: 21
type: A, layer: 1, pos: 21
type: A, layer: 1, pos: 38
type: B, layer: 1, pos: 38
type: A, layer: 1, pos: 47
type: B, layer: 1, pos: 47
type: B, layer: 1, pos: 35
type: A, layer: 1, pos: 35
type: B, layer: 1, pos: 15
type: A, layer: 1, pos: 10
type: A, layer: 1, pos: 15
type: A, layer: 1, pos: 30
type: B, layer: 1, pos: 10
type: B, layer: 1, pos: 3
type: B, layer: 1, pos: 30
type: A, layer: 1, pos: 3
type: B, layer: 1, pos: 24
type: A, layer: 1, pos: 24
type: A, layer: 1, pos: 1
type: B, layer: 1, pos: 1
type: B, layer: 1, pos: 14
type: A, layer: 1, pos: 14
type: A, layer: 1, pos: 40
type: B, layer: 1, pos: 40

Time for candidate selection: 0.24 seconds

### Candidate
type: B, layer: 1, pos: 8

### Candidate
type: A, layer: 1, pos: 8

### Candidate
type: B, layer: 1, pos: 5

## Relational analysis of NS_A1_B1_A2_B2_A2_B1

### Relational analysis result of NS_A1_B1_A2_B2_A2_B1
Status: Status.UNKNOWN
Output dim: 0, lower bound: -913.5295563, upper bound: 914.6337209
time: 0.47 seconds

## Relational analysis of NS_A1_B1_A2_B2_A2_B2

### Relational analysis result of NS_A1_B1_A2_B2_A2_B2
Status: Status.UNKNOWN
Output dim: 0, lower bound: -913.5295563, upper bound: 914.8548591
time: 0.49 seconds

## BFS NS instance: NS_A1_B2_B1_A1_B1

### Backsubstitution after applying NS history:
0: -430.6995239, 265.6437378, -411.8660278, 232.6477051, -663.3471069, 677.5097656
1: -31.6333313, 24.5905685, -28.6539593, 23.1143284, -54.7476463, 53.2445259
2: -27.3756618, 35.3720207, -25.7596779, 31.3545437, -58.7302055, 61.1316948
3: -30.5782852, 58.2092552, -28.4703903, 54.1781807, -84.7564697, 86.6796341
4: -23.5798454, 35.5405579, -22.2013893, 31.6630421, -55.2428703, 57.7419357

Time for backsubstitution: 2.90 seconds

### NS candidates at layer 1
type: A, layer: 1, pos: 8
type: B, layer: 1, pos: 8
type: A, layer: 1, pos: 38
type: A, layer: 1, pos: 5
type: B, layer: 1, pos: 5
type: A, layer: 1, pos: 10
type: B, layer: 1, pos: 38
type: B, layer: 1, pos: 10
type: A, layer: 1, pos: 35
type: B, layer: 1, pos: 35
type: A, layer: 1, pos: 47
type: A, layer: 1, pos: 3
type: B, layer: 1, pos: 15
type: A, layer: 1, pos: 1
type: B, layer: 1, pos: 21
type: A, layer: 1, pos: 21
type: A, layer: 1, pos: 15
type: B, layer: 1, pos: 1
type: B, layer: 1, pos: 3
type: A, layer: 1, pos: 30
type: B, layer: 1, pos: 47
type: A, layer: 1, pos: 24
type: B, layer: 1, pos: 24
type: A, layer: 1, pos: 14
type: A, layer: 1, pos: 40
type: B, layer: 1, pos: 14
type: B, layer: 1, pos: 40

Time for candidate selection: 0.24 seconds

### Candidate
type: A, layer: 1, pos: 8

### Candidate
type: B, layer: 1, pos: 8

### Candidate
type: A, layer: 1, pos: 38

## Relational analysis of NS_A1_B2_B1_A1_B1_A1
Optimization infeasible because this subproblem isn't reachable.

### Candidate
type: A, layer: 1, pos: 5

## Relational analysis of NS_A1_B2_B1_A1_B1_A1
Optimization infeasible because this subproblem isn't reachable.

### Candidate
type: B, layer: 1, pos: 5

### Candidate
type: A, layer: 1, pos: 10

## Relational analysis of NS_A1_B2_B1_A1_B1_A1

### Relational analysis result of NS_A1_B2_B1_A1_B1_A1
Status: Status.UNKNOWN
Output dim: 0, lower bound: -915.8446039, upper bound: 914.2045524
time: 0.46 seconds

## Relational analysis of NS_A1_B2_B1_A1_B1_A2

### Relational analysis result of NS_A1_B2_B1_A1_B1_A2
Status: Status.UNKNOWN
Output dim: 0, lower bound: -915.8598203, upper bound: 914.2533971
time: 0.48 seconds

## BFS NS instance: NS_A1_B2_B1_A1_B2

### Backsubstitution after applying NS history:
0: -457.2351990, 288.3154297, -452.7079468, 280.5736389, -737.8087769, 741.0233154
1: -34.0262680, 26.2646275, -33.3736992, 25.8884563, -59.9147263, 59.6383209
2: -28.9976425, 38.3013458, -28.5619106, 37.4851913, -66.4828339, 66.8632584
3: -32.4748497, 62.1999435, -31.8986073, 61.1683121, -93.6431580, 94.0985489
4: -24.9160557, 38.4459381, -24.5331535, 37.6732979, -62.5893517, 62.9790916

Time for backsubstitution: 2.90 seconds

### NS candidates at layer 1
type: A, layer: 1, pos: 8
type: B, layer: 1, pos: 8
type: A, layer: 1, pos: 38
type: B, layer: 1, pos: 38
type: B, layer: 1, pos: 21
type: A, layer: 1, pos: 21
type: A, layer: 1, pos: 47
type: A, layer: 1, pos: 30
type: A, layer: 1, pos: 35
type: B, layer: 1, pos: 5
type: B, layer: 1, pos: 47
type: B, layer: 1, pos: 35
type: A, layer: 1, pos: 5
type: B, layer: 1, pos: 10
type: B, layer: 1, pos: 15
type: A, layer: 1, pos: 10
type: B, layer: 1, pos: 24
type: A, layer: 1, pos: 3
type: B, layer: 1, pos: 3
type: B, layer: 1, pos: 1
type: A, layer: 1, pos: 24
type: A, layer: 1, pos: 15
type: A, layer: 1, pos: 1
type: B, layer: 1, pos: 14
type: A, layer: 1, pos: 14
type: A, layer: 1, pos: 40
type: B, layer: 1, pos: 40

Time for candidate selection: 0.24 seconds

### Candidate
type: A, layer: 1, pos: 8

### Candidate
type: B, layer: 1, pos: 8

### Candidate
type: A, layer: 1, pos: 38

## Relational analysis of NS_A1_B2_B1_A1_B2_A1
Optimization infeasible because this subproblem isn't reachable.

### Candidate
type: B, layer: 1, pos: 38

### Candidate
type: B, layer: 1, pos: 21

## Relational analysis of NS_A1_B2_B1_A1_B2_B1
Optimization infeasible because this subproblem isn't reachable.

### Candidate
type: A, layer: 1, pos: 21

### Candidate
type: A, layer: 1, pos: 47

## Relational analysis of NS_A1_B2_B1_A1_B2_A1

### Relational analysis result of NS_A1_B2_B1_A1_B2_A1
Status: Status.UNKNOWN
Output dim: 0, lower bound: -912.7943344, upper bound: 913.7291939
time: 0.51 seconds

## Relational analysis of NS_A1_B2_B1_A1_B2_A2

### Relational analysis result of NS_A1_B2_B1_A1_B2_A2
Status: Status.VERIFIED
Output dim: 0, lower bound: -912.7943344, upper bound: 912.7683715
time: 0.48 seconds

## BFS NS instance: NS_A1_B2_B1_A2_A1

### Backsubstitution after applying NS history:
0: -473.8457336, 304.6529846, -488.3984070, 313.5505981, -787.3963623, 793.0513916
1: -35.7674484, 27.3159313, -36.8722458, 28.1667194, -63.9341660, 64.1881638
2: -30.2721481, 40.5057869, -30.8656483, 41.8754425, -72.1475754, 71.3714218
3: -34.0434380, 65.0209045, -34.6121788, 67.0116425, -101.0550842, 99.6330795
4: -26.0014992, 40.6273994, -26.5074844, 42.0064163, -68.0079117, 67.1348801

Time for backsubstitution: 2.91 seconds

### NS candidates at layer 1
type: B, layer: 1, pos: 8
type: A, layer: 1, pos: 8
type: B, layer: 1, pos: 5
type: A, layer: 1, pos: 5
type: B, layer: 1, pos: 21
type: A, layer: 1, pos: 21
type: B, layer: 1, pos: 38
type: A, layer: 1, pos: 38
type: A, layer: 1, pos: 15
type: A, layer: 1, pos: 35
type: B, layer: 1, pos: 10
type: B, layer: 1, pos: 35
type: A, layer: 1, pos: 10
type: B, layer: 1, pos: 15
type: B, layer: 1, pos: 3
type: B, layer: 1, pos: 30
type: A, layer: 1, pos: 3
type: B, layer: 1, pos: 47
type: A, layer: 1, pos: 1
type: A, layer: 1, pos: 30
type: A, layer: 1, pos: 24
type: B, layer: 1, pos: 1
type: B, layer: 1, pos: 24
type: A, layer: 1, pos: 14
type: B, layer: 1, pos: 14
type: A, layer: 1, pos: 40
type: B, layer: 1, pos: 40

Time for candidate selection: 0.24 seconds

### Candidate
type: B, layer: 1, pos: 8

### Candidate
type: A, layer: 1, pos: 8

### Candidate
type: B, layer: 1, pos: 5

### Candidate
type: A, layer: 1, pos: 5

### Candidate
type: B, layer: 1, pos: 21

### Candidate
type: A, layer: 1, pos: 21

### Candidate
type: B, layer: 1, pos: 38

### Candidate
type: A, layer: 1, pos: 38

### Candidate
type: A, layer: 1, pos: 15

## Relational analysis of NS_A1_B2_B1_A2_A1_A1
Optimization infeasible because this subproblem isn't reachable.

### Candidate
type: A, layer: 1, pos: 35

## Relational analysis of NS_A1_B2_B1_A2_A1_A1
Optimization infeasible because this subproblem isn't reachable.

### Candidate
type: B, layer: 1, pos: 10

## Relational analysis of NS_A1_B2_B1_A2_A1_B1
Optimization infeasible because this subproblem isn't reachable.

### Candidate
type: B, layer: 1, pos: 35

### Candidate
type: A, layer: 1, pos: 10

### Candidate
type: B, layer: 1, pos: 15

### Candidate
type: B, layer: 1, pos: 3

## Relational analysis of NS_A1_B2_B1_A2_A1_B1

### Relational analysis result of NS_A1_B2_B1_A2_A1_B1
Status: Status.UNKNOWN
Output dim: 0, lower bound: -912.6991240, upper bound: 913.2513766
time: 0.51 seconds

## Relational analysis of NS_A1_B2_B1_A2_A1_B2
Optimization infeasible because this subproblem isn't reachable.

### Candidate
type: B, layer: 1, pos: 30

## Relational analysis of NS_A1_B2_B1_A2_A1_B1
Optimization infeasible because this subproblem isn't reachable.

### Candidate
type: A, layer: 1, pos: 3

### Candidate
type: B, layer: 1, pos: 47

## Relational analysis of NS_A1_B2_B1_A2_A1_B1

### Relational analysis result of NS_A1_B2_B1_A2_A1_B1
Status: Status.VERIFIED
Output dim: 0, lower bound: -912.7943344, upper bound: 912.8298472
time: 0.47 seconds

## Relational analysis of NS_A1_B2_B1_A2_A1_B2

### Relational analysis result of NS_A1_B2_B1_A2_A1_B2
Status: Status.VERIFIED
Output dim: 0, lower bound: -912.7943344, upper bound: 912.8298472
time: 0.50 seconds

## BFS NS instance: NS_A1_B2_B2_A1_B1

### Backsubstitution after applying NS history:
0: -430.6995239, 265.6437378, -444.9456482, 264.8309326, -695.5304565, 710.5893555
1: -31.6333313, 24.5905685, -31.9622993, 25.3064785, -56.9398117, 56.5528679
2: -27.3756618, 35.3720207, -27.7928543, 35.6828957, -63.0585556, 63.1648750
3: -30.5782852, 58.2092552, -30.9051781, 59.6276207, -90.2059021, 89.1144104
4: -23.5798454, 35.5405579, -23.8390045, 35.9607735, -59.5406189, 59.3795624

Time for backsubstitution: 2.93 seconds

### NS candidates at layer 1
type: A, layer: 1, pos: 21
type: B, layer: 1, pos: 21
type: B, layer: 1, pos: 10
type: A, layer: 1, pos: 38
type: A, layer: 1, pos: 10
type: B, layer: 1, pos: 38
type: A, layer: 1, pos: 8
type: B, layer: 1, pos: 8
type: B, layer: 1, pos: 5
type: A, layer: 1, pos: 5
type: A, layer: 1, pos: 35
type: B, layer: 1, pos: 35
type: A, layer: 1, pos: 47
type: A, layer: 1, pos: 1
type: B, layer: 1, pos: 47
type: B, layer: 1, pos: 1
type: A, layer: 1, pos: 15
type: A, layer: 1, pos: 3
type: B, layer: 1, pos: 15
type: B, layer: 1, pos: 3
type: B, layer: 1, pos: 24
type: A, layer: 1, pos: 30
type: A, layer: 1, pos: 24
type: A, layer: 1, pos: 14
type: B, layer: 1, pos: 14
type: A, layer: 1, pos: 40
type: B, layer: 1, pos: 40

Time for candidate selection: 0.24 seconds

### Candidate
type: A, layer: 1, pos: 21

## Relational analysis of NS_A1_B2_B2_A1_B1_A1
Optimization infeasible because this subproblem isn't reachable.

### Candidate
type: B, layer: 1, pos: 21

### Candidate
type: B, layer: 1, pos: 10

## Relational analysis of NS_A1_B2_B2_A1_B1_B1

### Relational analysis result of NS_A1_B2_B2_A1_B1_B1
Status: Status.UNKNOWN
Output dim: 0, lower bound: -916.1789315, upper bound: 914.7131008
time: 0.45 seconds

## Relational analysis of NS_A1_B2_B2_A1_B1_B2
Optimization infeasible because this subproblem isn't reachable.

### Candidate
type: A, layer: 1, pos: 38

## Relational analysis of NS_A1_B2_B2_A1_B1_A1

### Relational analysis result of NS_A1_B2_B2_A1_B1_A1
Status: Status.UNKNOWN
Output dim: 0, lower bound: -916.0325072, upper bound: 914.6195073
time: 0.50 seconds

## Relational analysis of NS_A1_B2_B2_A1_B1_A2
Optimization infeasible because this subproblem isn't reachable.

### Candidate
type: A, layer: 1, pos: 10

### Candidate
type: B, layer: 1, pos: 38

### Candidate
type: A, layer: 1, pos: 8

### Candidate
type: B, layer: 1, pos: 8

### Candidate
type: B, layer: 1, pos: 5

## Relational analysis of NS_A1_B2_B2_A1_B1_B1
Optimization infeasible because this subproblem isn't reachable.

### Candidate
type: A, layer: 1, pos: 5

### Candidate
type: A, layer: 1, pos: 35

## Relational analysis of NS_A1_B2_B2_A1_B1_A1

### Relational analysis result of NS_A1_B2_B2_A1_B1_A1
Status: Status.UNKNOWN
Output dim: 0, lower bound: -915.5498258, upper bound: 913.4911499
time: 0.43 seconds

## Relational analysis of NS_A1_B2_B2_A1_B1_A2

### Relational analysis result of NS_A1_B2_B2_A1_B1_A2
Status: Status.UNKNOWN
Output dim: 0, lower bound: -916.1746993, upper bound: 914.6891401
time: 0.47 seconds

## BFS NS instance: NS_A1_B2_B2_A1_B2

### Backsubstitution after applying NS history:
0: -457.2351990, 288.3154297, -480.0439453, 306.1353455, -763.3704834, 768.3591309
1: -34.0262680, 26.2646275, -36.0746346, 27.5946579, -61.6209259, 62.3392563
2: -28.9976425, 38.3013458, -30.3002110, 40.8993454, -69.8969879, 68.6015549
3: -32.4748497, 62.1999435, -33.9682083, 65.5940628, -98.0689011, 96.1681366
4: -24.9160557, 38.4459381, -25.9955521, 41.0402107, -65.9562607, 64.4414902

Time for backsubstitution: 2.94 seconds

### NS candidates at layer 1
type: A, layer: 1, pos: 8
type: B, layer: 1, pos: 8
type: B, layer: 1, pos: 38
type: A, layer: 1, pos: 38
type: A, layer: 1, pos: 21
type: B, layer: 1, pos: 21
type: B, layer: 1, pos: 35
type: B, layer: 1, pos: 5
type: A, layer: 1, pos: 35
type: A, layer: 1, pos: 5
type: A, layer: 1, pos: 47
type: B, layer: 1, pos: 3
type: A, layer: 1, pos: 30
type: B, layer: 1, pos: 24
type: B, layer: 1, pos: 10
type: A, layer: 1, pos: 3
type: A, layer: 1, pos: 15
type: A, layer: 1, pos: 24
type: B, layer: 1, pos: 1
type: A, layer: 1, pos: 10
type: B, layer: 1, pos: 47
type: B, layer: 1, pos: 15
type: A, layer: 1, pos: 1
type: B, layer: 1, pos: 14
type: A, layer: 1, pos: 14
type: A, layer: 1, pos: 40
type: B, layer: 1, pos: 40

Time for candidate selection: 0.25 seconds

### Candidate
type: A, layer: 1, pos: 8

### Candidate
type: B, layer: 1, pos: 8

### Candidate
type: B, layer: 1, pos: 38

## Relational analysis of NS_A1_B2_B2_A1_B2_B1
Optimization infeasible because this subproblem isn't reachable.

### Candidate
type: A, layer: 1, pos: 38

### Candidate
type: A, layer: 1, pos: 21

## Relational analysis of NS_A1_B2_B2_A1_B2_A1
Optimization infeasible because this subproblem isn't reachable.

### Candidate
type: B, layer: 1, pos: 21

### Candidate
type: B, layer: 1, pos: 35

## Relational analysis of NS_A1_B2_B2_A1_B2_B1

### Relational analysis result of NS_A1_B2_B2_A1_B2_B1
Status: Status.UNKNOWN
Output dim: 0, lower bound: -914.3409043, upper bound: 914.1697744
time: 0.51 seconds

## Relational analysis of NS_A1_B2_B2_A1_B2_B2

### Relational analysis result of NS_A1_B2_B2_A1_B2_B2
Status: Status.UNKNOWN
Output dim: 0, lower bound: -916.3030290, upper bound: 914.9279862
time: 0.48 seconds

## BFS NS instance: NS_A1_B2_B2_A2_B1

### Backsubstitution after applying NS history:
0: -502.7959900, 326.4364929, -489.5349121, 314.8911743, -817.6870117, 815.9712524
1: -38.1636848, 29.1213188, -37.0346642, 28.2443161, -66.4079895, 66.1559830
2: -31.9113941, 43.5109329, -31.0930405, 42.1184120, -74.0298080, 74.6039581
3: -36.0328712, 69.5438232, -34.8812561, 67.3007965, -103.3336563, 104.4250793
4: -27.3170071, 43.5654984, -26.6835747, 42.2701988, -69.5872040, 70.2490692

Time for backsubstitution: 2.93 seconds

### NS candidates at layer 1
type: A, layer: 1, pos: 8
type: B, layer: 1, pos: 8
type: B, layer: 1, pos: 5
type: A, layer: 1, pos: 5
type: A, layer: 1, pos: 21
type: B, layer: 1, pos: 21
type: A, layer: 1, pos: 38
type: B, layer: 1, pos: 38
type: A, layer: 1, pos: 35
type: B, layer: 1, pos: 35
type: A, layer: 1, pos: 10
type: B, layer: 1, pos: 10
type: B, layer: 1, pos: 3
type: A, layer: 1, pos: 3
type: B, layer: 1, pos: 30
type: A, layer: 1, pos: 30
type: B, layer: 1, pos: 15
type: A, layer: 1, pos: 15
type: A, layer: 1, pos: 47
type: B, layer: 1, pos: 1
type: B, layer: 1, pos: 24
type: A, layer: 1, pos: 1
type: A, layer: 1, pos: 24
type: B, layer: 1, pos: 14
type: A, layer: 1, pos: 14
type: B, layer: 1, pos: 40
type: A, layer: 1, pos: 40

Time for candidate selection: 0.25 seconds

### Candidate
type: A, layer: 1, pos: 8

### Candidate
type: B, layer: 1, pos: 8

### Candidate
type: B, layer: 1, pos: 5

### Candidate
type: A, layer: 1, pos: 5

### Candidate
type: A, layer: 1, pos: 21

### Candidate
type: B, layer: 1, pos: 21

### Candidate
type: A, layer: 1, pos: 38

### Candidate
type: B, layer: 1, pos: 38

### Candidate
type: A, layer: 1, pos: 35

## Relational analysis of NS_A1_B2_B2_A2_B1_A1

### Relational analysis result of NS_A1_B2_B2_A2_B1_A1
Status: Status.UNKNOWN
Output dim: 0, lower bound: -915.2242985, upper bound: 912.9268752
time: 0.50 seconds

## Relational analysis of NS_A1_B2_B2_A2_B1_A2
Optimization infeasible because this subproblem isn't reachable.

### Candidate
type: B, layer: 1, pos: 35

### Candidate
type: A, layer: 1, pos: 10

## Relational analysis of NS_A1_B2_B2_A2_B1_A1

### Relational analysis result of NS_A1_B2_B2_A2_B1_A1
Status: Status.UNKNOWN
Output dim: 0, lower bound: -914.4943917, upper bound: 912.7972417
time: 0.52 seconds

## Relational analysis of NS_A1_B2_B2_A2_B1_A2

### Relational analysis result of NS_A1_B2_B2_A2_B1_A2
Status: Status.UNKNOWN
Output dim: 0, lower bound: -914.4782008, upper bound: 912.5881746
time: 0.51 seconds

## BFS NS instance: NS_A2_B1_A1_B1_A1

### Backsubstitution after applying NS history:
0: -411.8660278, 232.6477051, -430.6995239, 265.6437378, -677.5097656, 663.3471069
1: -28.6539593, 23.1143284, -31.6333313, 24.5905685, -53.2445221, 54.7476463
2: -25.7596779, 31.3545437, -27.3756618, 35.3720207, -61.1316948, 58.7302055
3: -28.4703903, 54.1781807, -30.5782852, 58.2092552, -86.6796341, 84.7564697
4: -22.2013893, 31.6630421, -23.5798454, 35.5405579, -57.7419357, 55.2428741

Time for backsubstitution: 2.95 seconds

### NS candidates at layer 1
type: B, layer: 1, pos: 8
type: A, layer: 1, pos: 8
type: B, layer: 1, pos: 38
type: B, layer: 1, pos: 5
type: A, layer: 1, pos: 5
type: B, layer: 1, pos: 10
type: A, layer: 1, pos: 38
type: A, layer: 1, pos: 10
type: B, layer: 1, pos: 35
type: A, layer: 1, pos: 35
type: B, layer: 1, pos: 47
type: B, layer: 1, pos: 3
type: A, layer: 1, pos: 15
type: B, layer: 1, pos: 1
type: A, layer: 1, pos: 21
type: B, layer: 1, pos: 21
type: B, layer: 1, pos: 15
type: A, layer: 1, pos: 1
type: A, layer: 1, pos: 3
type: B, layer: 1, pos: 30
type: A, layer: 1, pos: 47
type: B, layer: 1, pos: 24
type: A, layer: 1, pos: 24
type: B, layer: 1, pos: 14
type: B, layer: 1, pos: 40
type: A, layer: 1, pos: 14
type: A, layer: 1, pos: 40

Time for candidate selection: 0.24 seconds

### Candidate
type: B, layer: 1, pos: 8

### Candidate
type: A, layer: 1, pos: 8

### Candidate
type: B, layer: 1, pos: 38

## Relational analysis of NS_A2_B1_A1_B1_A1_B1
Optimization infeasible because this subproblem isn't reachable.

### Candidate
type: B, layer: 1, pos: 5

## Relational analysis of NS_A2_B1_A1_B1_A1_B1
Optimization infeasible because this subproblem isn't reachable.

### Candidate
type: A, layer: 1, pos: 5

### Candidate
type: B, layer: 1, pos: 10

## Relational analysis of NS_A2_B1_A1_B1_A1_B1

### Relational analysis result of NS_A2_B1_A1_B1_A1_B1
Status: Status.UNKNOWN
Output dim: 0, lower bound: -914.2045524, upper bound: 915.8446039
time: 0.48 seconds

## Relational analysis of NS_A2_B1_A1_B1_A1_B2

### Relational analysis result of NS_A2_B1_A1_B1_A1_B2
Status: Status.UNKNOWN
Output dim: 0, lower bound: -914.2533971, upper bound: 915.8598194
time: 0.44 seconds

## BFS NS instance: NS_A2_B1_A1_B1_A2

### Backsubstitution after applying NS history:
0: -453.1636047, 281.0346069, -457.2351990, 288.3154297, -741.4789429, 738.2697144
1: -33.4238472, 25.9172192, -34.0262680, 26.2646275, -59.6884689, 59.9434891
2: -28.5894508, 37.5433464, -28.9976425, 38.3013458, -66.8907928, 66.5409851
3: -31.9307480, 61.2325745, -32.4748497, 62.1999435, -94.1306686, 93.7074280
4: -24.5563545, 37.7311325, -24.9160557, 38.4459381, -63.0022926, 62.6471863

Time for backsubstitution: 2.95 seconds

### NS candidates at layer 1
type: B, layer: 1, pos: 8
type: A, layer: 1, pos: 8
type: B, layer: 1, pos: 38
type: A, layer: 1, pos: 38
type: A, layer: 1, pos: 21
type: B, layer: 1, pos: 21
type: B, layer: 1, pos: 47
type: B, layer: 1, pos: 30
type: B, layer: 1, pos: 35
type: A, layer: 1, pos: 5
type: A, layer: 1, pos: 47
type: A, layer: 1, pos: 35
type: B, layer: 1, pos: 5
type: A, layer: 1, pos: 10
type: A, layer: 1, pos: 15
type: B, layer: 1, pos: 10
type: A, layer: 1, pos: 24
type: B, layer: 1, pos: 3
type: A, layer: 1, pos: 3
type: A, layer: 1, pos: 1
type: B, layer: 1, pos: 24
type: B, layer: 1, pos: 15
type: B, layer: 1, pos: 1
type: A, layer: 1, pos: 14
type: B, layer: 1, pos: 14
type: B, layer: 1, pos: 40
type: A, layer: 1, pos: 40

Time for candidate selection: 0.25 seconds

### Candidate
type: B, layer: 1, pos: 8

### Candidate
type: A, layer: 1, pos: 8

### Candidate
type: B, layer: 1, pos: 38

## Relational analysis of NS_A2_B1_A1_B1_A2_B1
Optimization infeasible because this subproblem isn't reachable.

### Candidate
type: A, layer: 1, pos: 38

### Candidate
type: A, layer: 1, pos: 21

## Relational analysis of NS_A2_B1_A1_B1_A2_A1
Optimization infeasible because this subproblem isn't reachable.

### Candidate
type: B, layer: 1, pos: 21

### Candidate
type: B, layer: 1, pos: 47

## Relational analysis of NS_A2_B1_A1_B1_A2_B1

### Relational analysis result of NS_A2_B1_A1_B1_A2_B1
Status: Status.UNKNOWN
Output dim: 0, lower bound: -913.7291939, upper bound: 912.7943344
time: 0.48 seconds

## Relational analysis of NS_A2_B1_A1_B1_A2_B2

### Relational analysis result of NS_A2_B1_A1_B1_A2_B2
Status: Status.VERIFIED
Output dim: 0, lower bound: -912.7683715, upper bound: 912.7943344
time: 0.48 seconds

## BFS NS instance: NS_A2_B1_A1_B2_B1

### Backsubstitution after applying NS history:
0: -489.2505188, 314.2459106, -473.8457336, 304.6529846, -793.9035034, 788.0916138
1: -36.9483871, 28.2214146, -35.7674484, 27.3159313, -64.2643127, 63.9888611
2: -30.9117508, 41.9644508, -30.2721481, 40.5057869, -71.4175415, 72.2365952
3: -34.6661301, 67.1468887, -34.0434380, 65.0209045, -99.6870346, 101.1903229
4: -26.5482178, 42.0926247, -26.0014992, 40.6273994, -67.1756134, 68.0941238

Time for backsubstitution: 2.95 seconds

### NS candidates at layer 1
type: A, layer: 1, pos: 8
type: B, layer: 1, pos: 8
type: A, layer: 1, pos: 5
type: B, layer: 1, pos: 5
type: A, layer: 1, pos: 21
type: B, layer: 1, pos: 21
type: A, layer: 1, pos: 38
type: B, layer: 1, pos: 38
type: B, layer: 1, pos: 15
type: B, layer: 1, pos: 35
type: A, layer: 1, pos: 10
type: A, layer: 1, pos: 35
type: B, layer: 1, pos: 10
type: A, layer: 1, pos: 15
type: A, layer: 1, pos: 3
type: A, layer: 1, pos: 30
type: B, layer: 1, pos: 3
type: A, layer: 1, pos: 47
type: B, layer: 1, pos: 1
type: B, layer: 1, pos: 30
type: B, layer: 1, pos: 24
type: A, layer: 1, pos: 1
type: A, layer: 1, pos: 24
type: B, layer: 1, pos: 14
type: A, layer: 1, pos: 14
type: B, layer: 1, pos: 40
type: A, layer: 1, pos: 40

Time for candidate selection: 0.24 seconds

### Candidate
type: A, layer: 1, pos: 8

### Candidate
type: B, layer: 1, pos: 8

### Candidate
type: A, layer: 1, pos: 5

### Candidate
type: B, layer: 1, pos: 5

### Candidate
type: A, layer: 1, pos: 21

### Candidate
type: B, layer: 1, pos: 21

### Candidate
type: A, layer: 1, pos: 38

### Candidate
type: B, layer: 1, pos: 38

### Candidate
type: B, layer: 1, pos: 15

## Relational analysis of NS_A2_B1_A1_B2_B1_B1
Optimization infeasible because this subproblem isn't reachable.

### Candidate
type: B, layer: 1, pos: 35

## Relational analysis of NS_A2_B1_A1_B2_B1_B1
Optimization infeasible because this subproblem isn't reachable.

### Candidate
type: A, layer: 1, pos: 10

## Relational analysis of NS_A2_B1_A1_B2_B1_A1
Optimization infeasible because this subproblem isn't reachable.

### Candidate
type: A, layer: 1, pos: 35

### Candidate
type: B, layer: 1, pos: 10

### Candidate
type: A, layer: 1, pos: 15

### Candidate
type: A, layer: 1, pos: 3

## Relational analysis of NS_A2_B1_A1_B2_B1_A1

### Relational analysis result of NS_A2_B1_A1_B2_B1_A1
Status: Status.UNKNOWN
Output dim: 0, lower bound: -913.2513766, upper bound: 912.6991240
time: 0.51 seconds

## Relational analysis of NS_A2_B1_A1_B2_B1_A2
Optimization infeasible because this subproblem isn't reachable.

### Candidate
type: A, layer: 1, pos: 30

## Relational analysis of NS_A2_B1_A1_B2_B1_A1
Optimization infeasible because this subproblem isn't reachable.

### Candidate
type: B, layer: 1, pos: 3

### Candidate
type: A, layer: 1, pos: 47

## Relational analysis of NS_A2_B1_A1_B2_B1_A1

### Relational analysis result of NS_A2_B1_A1_B2_B1_A1
Status: Status.VERIFIED
Output dim: 0, lower bound: -912.8298472, upper bound: 912.7943344
time: 0.50 seconds

## Relational analysis of NS_A2_B1_A1_B2_B1_A2

### Relational analysis result of NS_A2_B1_A1_B2_B1_A2
Status: Status.VERIFIED
Output dim: 0, lower bound: -912.8298472, upper bound: 912.7943344
time: 0.46 seconds

## BFS NS instance: NS_A2_B1_A2_B1_A1

### Backsubstitution after applying NS history:
0: -445.0460510, 264.9409180, -430.6995239, 265.6437378, -710.6898193, 695.6404419
1: -31.9736042, 25.3131657, -31.6333313, 24.5905685, -56.5641708, 56.9464951
2: -27.7984943, 35.6963768, -27.3756618, 35.3720207, -63.1705170, 63.0720367
3: -30.9119720, 59.6424522, -30.5782852, 58.2092552, -89.1212311, 90.2207336
4: -23.8436604, 35.9742317, -23.5798454, 35.5405579, -59.3842163, 59.5540771

Time for backsubstitution: 2.95 seconds

### NS candidates at layer 1
type: B, layer: 1, pos: 21
type: A, layer: 1, pos: 21
type: A, layer: 1, pos: 10
type: B, layer: 1, pos: 38
type: B, layer: 1, pos: 10
type: A, layer: 1, pos: 38
type: B, layer: 1, pos: 8
type: A, layer: 1, pos: 8
type: A, layer: 1, pos: 5
type: B, layer: 1, pos: 5
type: B, layer: 1, pos: 35
type: A, layer: 1, pos: 35
type: B, layer: 1, pos: 47
type: B, layer: 1, pos: 1
type: A, layer: 1, pos: 47
type: A, layer: 1, pos: 1
type: B, layer: 1, pos: 15
type: B, layer: 1, pos: 3
type: A, layer: 1, pos: 15
type: A, layer: 1, pos: 3
type: A, layer: 1, pos: 24
type: B, layer: 1, pos: 30
type: B, layer: 1, pos: 24
type: B, layer: 1, pos: 14
type: A, layer: 1, pos: 14
type: B, layer: 1, pos: 40
type: A, layer: 1, pos: 40

Time for candidate selection: 0.24 seconds

### Candidate
type: B, layer: 1, pos: 21

## Relational analysis of NS_A2_B1_A2_B1_A1_B1
Optimization infeasible because this subproblem isn't reachable.

### Candidate
type: A, layer: 1, pos: 21

### Candidate
type: A, layer: 1, pos: 10

## Relational analysis of NS_A2_B1_A2_B1_A1_A1

### Relational analysis result of NS_A2_B1_A2_B1_A1_A1
Status: Status.UNKNOWN
Output dim: 0, lower bound: -914.7131008, upper bound: 916.1789315
time: 0.47 seconds

## Relational analysis of NS_A2_B1_A2_B1_A1_A2
Optimization infeasible because this subproblem isn't reachable.

### Candidate
type: B, layer: 1, pos: 38

## Relational analysis of NS_A2_B1_A2_B1_A1_B1

### Relational analysis result of NS_A2_B1_A2_B1_A1_B1
Status: Status.UNKNOWN
Output dim: 0, lower bound: -914.6195073, upper bound: 916.0325072
time: 0.46 seconds

## Relational analysis of NS_A2_B1_A2_B1_A1_B2
Optimization infeasible because this subproblem isn't reachable.

### Candidate
type: B, layer: 1, pos: 10

### Candidate
type: A, layer: 1, pos: 38

### Candidate
type: B, layer: 1, pos: 8

### Candidate
type: A, layer: 1, pos: 8

### Candidate
type: A, layer: 1, pos: 5

## Relational analysis of NS_A2_B1_A2_B1_A1_A1
Optimization infeasible because this subproblem isn't reachable.

### Candidate
type: B, layer: 1, pos: 5

### Candidate
type: B, layer: 1, pos: 35

## Relational analysis of NS_A2_B1_A2_B1_A1_B1

### Relational analysis result of NS_A2_B1_A2_B1_A1_B1
Status: Status.UNKNOWN
Output dim: 0, lower bound: -913.4911499, upper bound: 915.5498258
time: 0.49 seconds

## Relational analysis of NS_A2_B1_A2_B1_A1_B2

### Relational analysis result of NS_A2_B1_A2_B1_A1_B2
Status: Status.UNKNOWN
Output dim: 0, lower bound: -914.6891401, upper bound: 916.1746992
time: 0.47 seconds

## BFS NS instance: NS_A2_B1_A2_B1_A2

### Backsubstitution after applying NS history:
0: -480.2791138, 306.3418274, -457.2351990, 288.3154297, -768.5944214, 763.5770264
1: -36.0972176, 27.6105824, -34.0262680, 26.2646275, -62.3618393, 61.6368484
2: -30.3139515, 40.9253883, -28.9976425, 38.3013458, -68.6152878, 69.9230347
3: -33.9842377, 65.6338654, -32.4748497, 62.1999435, -96.1841660, 98.1087189
4: -26.0074596, 41.0660439, -24.9160557, 38.4459381, -64.4533997, 65.9821014

Time for backsubstitution: 3.00 seconds

### NS candidates at layer 1
type: B, layer: 1, pos: 8
type: A, layer: 1, pos: 8
type: A, layer: 1, pos: 38
type: B, layer: 1, pos: 38
type: B, layer: 1, pos: 21
type: A, layer: 1, pos: 21
type: A, layer: 1, pos: 35
type: A, layer: 1, pos: 5
type: B, layer: 1, pos: 35
type: B, layer: 1, pos: 5
type: B, layer: 1, pos: 47
type: A, layer: 1, pos: 3
type: B, layer: 1, pos: 30
type: A, layer: 1, pos: 24
type: A, layer: 1, pos: 10
type: B, layer: 1, pos: 3
type: B, layer: 1, pos: 15
type: B, layer: 1, pos: 24
type: A, layer: 1, pos: 1
type: B, layer: 1, pos: 10
type: A, layer: 1, pos: 47
type: A, layer: 1, pos: 15
type: B, layer: 1, pos: 1
type: A, layer: 1, pos: 14
type: B, layer: 1, pos: 14
type: B, layer: 1, pos: 40
type: A, layer: 1, pos: 40

Time for candidate selection: 0.25 seconds

### Candidate
type: B, layer: 1, pos: 8

### Candidate
type: A, layer: 1, pos: 8

### Candidate
type: A, layer: 1, pos: 38

## Relational analysis of NS_A2_B1_A2_B1_A2_A1
Optimization infeasible because this subproblem isn't reachable.

### Candidate
type: B, layer: 1, pos: 38

### Candidate
type: B, layer: 1, pos: 21

## Relational analysis of NS_A2_B1_A2_B1_A2_B1
Optimization infeasible because this subproblem isn't reachable.

### Candidate
type: A, layer: 1, pos: 21

### Candidate
type: A, layer: 1, pos: 35

## Relational analysis of NS_A2_B1_A2_B1_A2_A1

### Relational analysis result of NS_A2_B1_A2_B1_A2_A1
Status: Status.UNKNOWN
Output dim: 0, lower bound: -914.1697744, upper bound: 914.3409043
time: 0.49 seconds

## Relational analysis of NS_A2_B1_A2_B1_A2_A2

### Relational analysis result of NS_A2_B1_A2_B1_A2_A2
Status: Status.UNKNOWN
Output dim: 0, lower bound: -914.9279862, upper bound: 916.3030283
time: 0.48 seconds

## BFS NS instance: NS_A2_B1_A2_B2_A1

### Backsubstitution after applying NS history:
0: -489.9631958, 315.2864380, -502.7959900, 326.4364929, -816.3996582, 818.0822754
1: -37.0779495, 28.2730389, -38.1636848, 29.1213188, -66.1992645, 66.4367065
2: -31.1182842, 42.1677780, -31.9113941, 43.5109329, -74.6291885, 74.0791702
3: -34.9102936, 67.3761978, -36.0328712, 69.5438232, -104.4541168, 103.4090576
4: -26.7058048, 42.3191261, -27.3170071, 43.5654984, -70.2712936, 69.6361313

Time for backsubstitution: 3.00 seconds

### NS candidates at layer 1
type: B, layer: 1, pos: 8
type: A, layer: 1, pos: 8
type: A, layer: 1, pos: 5
type: B, layer: 1, pos: 5
type: B, layer: 1, pos: 21
type: A, layer: 1, pos: 21
type: B, layer: 1, pos: 38
type: A, layer: 1, pos: 38
type: B, layer: 1, pos: 35
type: A, layer: 1, pos: 35
type: B, layer: 1, pos: 10
type: A, layer: 1, pos: 10
type: A, layer: 1, pos: 3
type: B, layer: 1, pos: 3
type: A, layer: 1, pos: 30
type: B, layer: 1, pos: 30
type: A, layer: 1, pos: 15
type: B, layer: 1, pos: 15
type: B, layer: 1, pos: 47
type: A, layer: 1, pos: 1
type: A, layer: 1, pos: 24
type: B, layer: 1, pos: 1
type: B, layer: 1, pos: 24
type: A, layer: 1, pos: 14
type: B, layer: 1, pos: 14
type: A, layer: 1, pos: 40
type: B, layer: 1, pos: 40

Time for candidate selection: 0.25 seconds

### Candidate
type: B, layer: 1, pos: 8

### Candidate
type: A, layer: 1, pos: 8

### Candidate
type: A, layer: 1, pos: 5

### Candidate
type: B, layer: 1, pos: 5

### Candidate
type: B, layer: 1, pos: 21

### Candidate
type: A, layer: 1, pos: 21

### Candidate
type: B, layer: 1, pos: 38

### Candidate
type: A, layer: 1, pos: 38

### Candidate
type: B, layer: 1, pos: 35

## Relational analysis of NS_A2_B1_A2_B2_A1_B1

### Relational analysis result of NS_A2_B1_A2_B2_A1_B1
Status: Status.UNKNOWN
Output dim: 0, lower bound: -912.9268752, upper bound: 915.2535207
time: 0.47 seconds

## Relational analysis of NS_A2_B1_A2_B2_A1_B2
Optimization infeasible because this subproblem isn't reachable.

### Candidate
type: A, layer: 1, pos: 35

### Candidate
type: B, layer: 1, pos: 10

## Relational analysis of NS_A2_B1_A2_B2_A1_B1

### Relational analysis result of NS_A2_B1_A2_B2_A1_B1
Status: Status.UNKNOWN
Output dim: 0, lower bound: -912.6393855, upper bound: 914.4915282
time: 0.51 seconds

## Relational analysis of NS_A2_B1_A2_B2_A1_B2

### Relational analysis result of NS_A2_B1_A2_B2_A1_B2
Status: Status.UNKNOWN
Output dim: 0, lower bound: -912.5728632, upper bound: 914.4782008
time: 0.52 seconds

## BFS NS instance: NS_A2_B1_A2_B2_A2

### Backsubstitution after applying NS history:
0: -535.7368774, 356.5688782, -497.2687073, 321.9904175, -857.7272949, 853.8375854
1: -41.4338570, 31.1200237, -37.6844101, 28.7705345, -70.2043762, 68.8044205
2: -34.5892220, 47.4951859, -31.5697098, 42.9011574, -77.4903641, 79.0648727
3: -39.1576920, 74.9632263, -35.6226730, 68.6508255, -107.8085175, 110.5858917
4: -29.6620693, 47.4986610, -27.0508842, 42.9773788, -72.6394348, 74.5495453

Time for backsubstitution: 3.00 seconds

### NS candidates at layer 1
type: B, layer: 1, pos: 8
type: A, layer: 1, pos: 8
type: A, layer: 1, pos: 5
type: B, layer: 1, pos: 5
type: B, layer: 1, pos: 38
type: A, layer: 1, pos: 15
type: B, layer: 1, pos: 21
type: A, layer: 1, pos: 21
type: A, layer: 1, pos: 38
type: B, layer: 1, pos: 30
type: B, layer: 1, pos: 10
type: A, layer: 1, pos: 35
type: B, layer: 1, pos: 35
type: A, layer: 1, pos: 24
type: A, layer: 1, pos: 1
type: B, layer: 1, pos: 47
type: A, layer: 1, pos: 3
type: B, layer: 1, pos: 3
type: A, layer: 1, pos: 10
type: B, layer: 1, pos: 15
type: B, layer: 1, pos: 1
type: A, layer: 1, pos: 30
type: A, layer: 1, pos: 14
type: B, layer: 1, pos: 24
type: B, layer: 1, pos: 14
type: A, layer: 1, pos: 40
type: B, layer: 1, pos: 40

Time for candidate selection: 0.25 seconds

### Candidate
type: B, layer: 1, pos: 8

### Candidate
type: A, layer: 1, pos: 8

### Candidate
type: A, layer: 1, pos: 5

### Candidate
type: B, layer: 1, pos: 5

### Candidate
type: B, layer: 1, pos: 38

### Candidate
type: A, layer: 1, pos: 15

## Relational analysis of NS_A2_B1_A2_B2_A2_A1
Optimization infeasible because this subproblem isn't reachable.

### Candidate
type: B, layer: 1, pos: 21

### Candidate
type: A, layer: 1, pos: 21

### Candidate
type: A, layer: 1, pos: 38

### Candidate
type: B, layer: 1, pos: 30

## Relational analysis of NS_A2_B1_A2_B2_A2_B1

### Relational analysis result of NS_A2_B1_A2_B2_A2_B1
Status: Status.UNKNOWN
Output dim: 0, lower bound: -912.9698978, upper bound: 912.9914879
time: 0.48 seconds

## Relational analysis of NS_A2_B1_A2_B2_A2_B2

### Relational analysis result of NS_A2_B1_A2_B2_A2_B2
Status: Status.UNKNOWN
Output dim: 0, lower bound: -912.9698978, upper bound: 913.0607464
time: 0.51 seconds

## BFS NS instance: NS_A2_B2_A1_B1_A1

### Backsubstitution after applying NS history:
0: -465.6074829, 294.3510742, -483.1653442, 308.8505249, -774.4580078, 777.5163574
1: -34.8219070, 26.7153473, -36.3776741, 27.8167915, -62.6386986, 63.0930214
2: -29.4445629, 39.3454018, -30.5187321, 41.2554703, -70.7000198, 69.8641205
3: -32.9577789, 63.3820267, -34.2021103, 66.1254959, -99.0832748, 97.5841370
4: -25.3062611, 39.4749718, -26.2118130, 41.3911629, -66.6974258, 65.6867752

Time for backsubstitution: 3.01 seconds

### NS candidates at layer 1
type: A, layer: 1, pos: 8
type: B, layer: 1, pos: 8
type: B, layer: 1, pos: 5
type: A, layer: 1, pos: 5
type: A, layer: 1, pos: 10
type: B, layer: 1, pos: 38
type: B, layer: 1, pos: 10
type: B, layer: 1, pos: 35
type: A, layer: 1, pos: 38
type: A, layer: 1, pos: 47
type: A, layer: 1, pos: 21
type: B, layer: 1, pos: 21
type: B, layer: 1, pos: 47
type: B, layer: 1, pos: 3
type: A, layer: 1, pos: 3
type: B, layer: 1, pos: 15
type: A, layer: 1, pos: 15
type: A, layer: 1, pos: 30
type: B, layer: 1, pos: 30
type: B, layer: 1, pos: 24
type: B, layer: 1, pos: 1
type: A, layer: 1, pos: 24
type: A, layer: 1, pos: 1
type: B, layer: 1, pos: 14
type: A, layer: 1, pos: 14
type: B, layer: 1, pos: 40
type: A, layer: 1, pos: 40

Time for candidate selection: 0.25 seconds

### Candidate
type: A, layer: 1, pos: 8

### Candidate
type: B, layer: 1, pos: 8

### Candidate
type: B, layer: 1, pos: 5

### Candidate
type: A, layer: 1, pos: 5

### Candidate
type: A, layer: 1, pos: 10

## Relational analysis of NS_A2_B2_A1_B1_A1_A1

### Relational analysis result of NS_A2_B2_A1_B1_A1_A1
Status: Status.UNKNOWN
Output dim: 0, lower bound: -914.3228451, upper bound: 913.8158196
time: 0.50 seconds

## Relational analysis of NS_A2_B2_A1_B1_A1_A2
Optimization infeasible because this subproblem isn't reachable.

### Candidate
type: B, layer: 1, pos: 38

## Relational analysis of NS_A2_B2_A1_B1_A1_B1
Optimization infeasible because this subproblem isn't reachable.

### Candidate
type: B, layer: 1, pos: 10

### Candidate
type: B, layer: 1, pos: 35

## Relational analysis of NS_A2_B2_A1_B1_A1_B1

### Relational analysis result of NS_A2_B2_A1_B1_A1_B1
Status: Status.UNKNOWN
Output dim: 0, lower bound: -914.1047483, upper bound: 914.1047483
time: 0.50 seconds

## Relational analysis of NS_A2_B2_A1_B1_A1_B2

### Relational analysis result of NS_A2_B2_A1_B1_A1_B2
Status: Status.UNKNOWN
Output dim: 0, lower bound: -914.1047483, upper bound: 914.1047483
time: 0.58 seconds

## BFS NS instance: NS_A2_B2_A1_B1_A2

### Backsubstitution after applying NS history:
0: -495.7195740, 318.8988037, -483.1653442, 308.8505249, -804.5700684, 802.0641479
1: -37.4901772, 28.6070480, -36.3776741, 27.8167915, -65.3069687, 64.9847031
2: -31.1777763, 42.7417831, -30.5187321, 41.2554703, -72.4332428, 73.2605057
3: -35.1153107, 68.3260193, -34.2021103, 66.1254959, -101.2408066, 102.5281296
4: -26.7884121, 42.8062248, -26.2118130, 41.3911629, -68.1795731, 69.0180359

Time for backsubstitution: 3.03 seconds

### NS candidates at layer 1
type: A, layer: 1, pos: 5
type: B, layer: 1, pos: 5
type: A, layer: 1, pos: 8
type: B, layer: 1, pos: 8
type: A, layer: 1, pos: 38
type: B, layer: 1, pos: 35
type: B, layer: 1, pos: 38
type: A, layer: 1, pos: 10
type: B, layer: 1, pos: 3
type: A, layer: 1, pos: 3
type: B, layer: 1, pos: 10
type: B, layer: 1, pos: 21
type: A, layer: 1, pos: 21
type: A, layer: 1, pos: 47
type: A, layer: 1, pos: 30
type: B, layer: 1, pos: 47
type: B, layer: 1, pos: 30
type: B, layer: 1, pos: 15
type: A, layer: 1, pos: 15
type: B, layer: 1, pos: 1
type: B, layer: 1, pos: 24
type: A, layer: 1, pos: 1
type: A, layer: 1, pos: 24
type: B, layer: 1, pos: 14
type: A, layer: 1, pos: 14
type: B, layer: 1, pos: 40
type: A, layer: 1, pos: 40

Time for candidate selection: 0.25 seconds

### Candidate
type: A, layer: 1, pos: 5

### Candidate
type: B, layer: 1, pos: 5

### Candidate
type: A, layer: 1, pos: 8

### Candidate
type: B, layer: 1, pos: 8

### Candidate
type: A, layer: 1, pos: 38

## Relational analysis of NS_A2_B2_A1_B1_A2_A1
Optimization infeasible because this subproblem isn't reachable.

### Candidate
type: B, layer: 1, pos: 35

## Relational analysis of NS_A2_B2_A1_B1_A2_B1

### Relational analysis result of NS_A2_B2_A1_B1_A2_B1
Status: Status.UNKNOWN
Output dim: 0, lower bound: -914.1047483, upper bound: 914.6940041
time: 0.50 seconds

## Relational analysis of NS_A2_B2_A1_B1_A2_B2

### Relational analysis result of NS_A2_B2_A1_B1_A2_B2
Status: Status.UNKNOWN
Output dim: 0, lower bound: -914.1047483, upper bound: 914.7501862
time: 0.49 seconds

## BFS NS instance: NS_A2_B2_A1_B2_A1

### Backsubstitution after applying NS history:
0: -465.6074829, 294.3510742, -513.9433594, 333.2450562, -798.8525391, 808.2944336
1: -34.8219070, 26.7153473, -39.0157394, 29.7766666, -64.5985565, 65.7310715
2: -29.4445629, 39.3454018, -32.2850876, 44.6390572, -74.0836182, 71.6304855
3: -32.9577789, 63.3820267, -36.4700165, 71.1604691, -104.1182404, 99.8520432
4: -25.3062611, 39.4749718, -27.7318687, 44.6966934, -70.0029373, 67.2068253

Time for backsubstitution: 3.04 seconds

### NS candidates at layer 1
type: A, layer: 1, pos: 8
type: B, layer: 1, pos: 8
type: B, layer: 1, pos: 5
type: A, layer: 1, pos: 5
type: B, layer: 1, pos: 38
type: A, layer: 1, pos: 10
type: B, layer: 1, pos: 10
type: B, layer: 1, pos: 35
type: A, layer: 1, pos: 38
type: B, layer: 1, pos: 47
type: B, layer: 1, pos: 3
type: A, layer: 1, pos: 47
type: A, layer: 1, pos: 21
type: B, layer: 1, pos: 21
type: A, layer: 1, pos: 3
type: B, layer: 1, pos: 15
type: A, layer: 1, pos: 15
type: A, layer: 1, pos: 30
type: B, layer: 1, pos: 30
type: B, layer: 1, pos: 1
type: B, layer: 1, pos: 24
type: A, layer: 1, pos: 24
type: A, layer: 1, pos: 1
type: B, layer: 1, pos: 14
type: A, layer: 1, pos: 14
type: B, layer: 1, pos: 40
type: A, layer: 1, pos: 40

Time for candidate selection: 0.25 seconds

### Candidate
type: A, layer: 1, pos: 8

## Relational analysis of NS_A2_B2_A1_B2_A1_A1
Optimization infeasible because this subproblem isn't reachable.

### Candidate
type: B, layer: 1, pos: 8

### Candidate
type: B, layer: 1, pos: 5

## Relational analysis of NS_A2_B2_A1_B2_A1_B1
Optimization infeasible because this subproblem isn't reachable.

### Candidate
type: A, layer: 1, pos: 5

### Candidate
type: B, layer: 1, pos: 38

## Relational analysis of NS_A2_B2_A1_B2_A1_B1
Optimization infeasible because this subproblem isn't reachable.

### Candidate
type: A, layer: 1, pos: 10

## Relational analysis of NS_A2_B2_A1_B2_A1_A1

### Relational analysis result of NS_A2_B2_A1_B2_A1_A1
Status: Status.UNKNOWN
Output dim: 0, lower bound: -914.3176703, upper bound: 913.6386180
time: 0.50 seconds

## Relational analysis of NS_A2_B2_A1_B2_A1_A2
Optimization infeasible because this subproblem isn't reachable.

### Candidate
type: B, layer: 1, pos: 10

### Candidate
type: B, layer: 1, pos: 35

## Relational analysis of NS_A2_B2_A1_B2_A1_B1

### Relational analysis result of NS_A2_B2_A1_B2_A1_B1
Status: Status.UNKNOWN
Output dim: 0, lower bound: -914.1047483, upper bound: 914.1047482
time: 0.48 seconds

## Relational analysis of NS_A2_B2_A1_B2_A1_B2

### Relational analysis result of NS_A2_B2_A1_B2_A1_B2
Status: Status.UNKNOWN
Output dim: 0, lower bound: -914.1047483, upper bound: 914.1047483
time: 0.47 seconds

## BFS NS instance: NS_A2_B2_A1_B2_A2

### Backsubstitution after applying NS history:
0: -495.7195740, 318.8988037, -513.9433594, 333.2450562, -828.9645996, 832.8421631
1: -37.4901772, 28.6070480, -39.0157394, 29.7766666, -67.2668381, 67.6227722
2: -31.1777763, 42.7417831, -32.2850876, 44.6390572, -75.8168335, 75.0268707
3: -35.1153107, 68.3260193, -36.4700165, 71.1604691, -106.2757797, 104.7960358
4: -26.7884121, 42.8062248, -27.7318687, 44.6966934, -71.4850998, 70.5380936

Time for backsubstitution: 3.03 seconds

### NS candidates at layer 1
type: B, layer: 1, pos: 5
type: A, layer: 1, pos: 5
type: A, layer: 1, pos: 8
type: B, layer: 1, pos: 8
type: B, layer: 1, pos: 35
type: B, layer: 1, pos: 38
type: A, layer: 1, pos: 38
type: A, layer: 1, pos: 10
type: B, layer: 1, pos: 3
type: B, layer: 1, pos: 10
type: A, layer: 1, pos: 3
type: A, layer: 1, pos: 21
type: B, layer: 1, pos: 21
type: A, layer: 1, pos: 47
type: B, layer: 1, pos: 47
type: A, layer: 1, pos: 30
type: B, layer: 1, pos: 30
type: B, layer: 1, pos: 15
type: A, layer: 1, pos: 15
type: B, layer: 1, pos: 1
type: B, layer: 1, pos: 24
type: A, layer: 1, pos: 1
type: A, layer: 1, pos: 24
type: B, layer: 1, pos: 14
type: A, layer: 1, pos: 14
type: B, layer: 1, pos: 40
type: A, layer: 1, pos: 40

Time for candidate selection: 0.25 seconds

### Candidate
type: B, layer: 1, pos: 5

## Relational analysis of NS_A2_B2_A1_B2_A2_B1
Optimization infeasible because this subproblem isn't reachable.

### Candidate
type: A, layer: 1, pos: 5

### Candidate
type: A, layer: 1, pos: 8

## Relational analysis of NS_A2_B2_A1_B2_A2_A1
Optimization infeasible because this subproblem isn't reachable.

### Candidate
type: B, layer: 1, pos: 8

### Candidate
type: B, layer: 1, pos: 35

## Relational analysis of NS_A2_B2_A1_B2_A2_B1

### Relational analysis result of NS_A2_B2_A1_B2_A2_B1
Status: Status.UNKNOWN
Output dim: 0, lower bound: -914.1047483, upper bound: 915.1457901
time: 0.49 seconds

## Relational analysis of NS_A2_B2_A1_B2_A2_B2

### Relational analysis result of NS_A2_B2_A1_B2_A2_B2
Status: Status.UNKNOWN
Output dim: 0, lower bound: -914.1047483, upper bound: 915.1457901
time: 0.49 seconds

## BFS NS instance: NS_A2_B2_A2_B1_B1

### Backsubstitution after applying NS history:
0: -508.0582275, 328.6637573, -465.6074829, 294.3510742, -802.4093018, 794.2711182
1: -38.5196991, 29.4332695, -34.8219070, 26.7153473, -65.2350464, 64.2551727
2: -31.8580914, 44.0472260, -29.4445629, 39.3454018, -71.2034912, 73.4917755
3: -35.9649582, 70.2625427, -32.9577789, 63.3820267, -99.3469772, 103.2203217
4: -27.3524818, 44.1296921, -25.3062611, 39.4749718, -66.8274536, 69.4359512

Time for backsubstitution: 3.02 seconds

### NS candidates at layer 1
type: A, layer: 1, pos: 8
type: B, layer: 1, pos: 8
type: A, layer: 1, pos: 5
type: B, layer: 1, pos: 5
type: A, layer: 1, pos: 38
type: A, layer: 1, pos: 10
type: B, layer: 1, pos: 10
type: A, layer: 1, pos: 18
type: B, layer: 1, pos: 38
type: A, layer: 1, pos: 47
type: A, layer: 1, pos: 3
type: B, layer: 1, pos: 21
type: B, layer: 1, pos: 47
type: A, layer: 1, pos: 21
type: B, layer: 1, pos: 3
type: A, layer: 1, pos: 15
type: B, layer: 1, pos: 15
type: B, layer: 1, pos: 30
type: A, layer: 1, pos: 30
type: B, layer: 1, pos: 24
type: A, layer: 1, pos: 24
type: A, layer: 1, pos: 1
type: B, layer: 1, pos: 1
type: A, layer: 1, pos: 14
type: B, layer: 1, pos: 14
type: A, layer: 1, pos: 40
type: B, layer: 1, pos: 40

Time for candidate selection: 0.25 seconds

### Candidate
type: A, layer: 1, pos: 8

## Relational analysis of NS_A2_B2_A2_B1_B1_A1
Optimization infeasible because this subproblem isn't reachable.

### Candidate
type: B, layer: 1, pos: 8

### Candidate
type: A, layer: 1, pos: 5

### Candidate
type: B, layer: 1, pos: 5

### Candidate
type: A, layer: 1, pos: 38

## Relational analysis of NS_A2_B2_A2_B1_B1_A1
Optimization infeasible because this subproblem isn't reachable.

### Candidate
type: A, layer: 1, pos: 10

## Relational analysis of NS_A2_B2_A2_B1_B1_A1
Optimization infeasible because this subproblem isn't reachable.

### Candidate
type: B, layer: 1, pos: 10

### Candidate
type: A, layer: 1, pos: 18

## Relational analysis of NS_A2_B2_A2_B1_B1_A1

### Relational analysis result of NS_A2_B2_A2_B1_B1_A1
Status: Status.UNKNOWN
Output dim: 0, lower bound: -914.1047483, upper bound: 915.8888488
time: 0.48 seconds

## Relational analysis of NS_A2_B2_A2_B1_B1_A2

### Relational analysis result of NS_A2_B2_A2_B1_B1_A2
Status: Status.UNKNOWN
Output dim: 0, lower bound: -914.1047483, upper bound: 915.9322522
time: 0.48 seconds

## BFS NS instance: NS_A2_B2_A2_B1_B2

### Backsubstitution after applying NS history:
0: -508.0582275, 328.6637573, -478.8077698, 305.1115723, -813.1697998, 807.4712524
1: -38.5196991, 29.4332695, -35.9700279, 27.5468063, -66.0665054, 65.4032898
2: -31.8580914, 44.0472260, -30.2316093, 40.7472878, -72.6053772, 74.2788239
3: -35.9649582, 70.2625427, -33.8646011, 65.4005203, -101.3654709, 104.1271362
4: -27.3524818, 44.1296921, -25.9642067, 40.8872604, -68.2397461, 70.0939026

Time for backsubstitution: 3.03 seconds

### NS candidates at layer 1
type: B, layer: 1, pos: 8
type: A, layer: 1, pos: 8
type: B, layer: 1, pos: 5
type: A, layer: 1, pos: 5
type: A, layer: 1, pos: 38
type: B, layer: 1, pos: 10
type: A, layer: 1, pos: 10
type: B, layer: 1, pos: 38
type: A, layer: 1, pos: 18
type: A, layer: 1, pos: 47
type: B, layer: 1, pos: 21
type: A, layer: 1, pos: 21
type: B, layer: 1, pos: 47
type: A, layer: 1, pos: 3
type: B, layer: 1, pos: 3
type: A, layer: 1, pos: 15
type: B, layer: 1, pos: 15
type: A, layer: 1, pos: 30
type: B, layer: 1, pos: 30
type: B, layer: 1, pos: 24
type: B, layer: 1, pos: 1
type: A, layer: 1, pos: 24
type: A, layer: 1, pos: 1
type: B, layer: 1, pos: 14
type: A, layer: 1, pos: 14
type: B, layer: 1, pos: 40
type: A, layer: 1, pos: 40

Time for candidate selection: 0.25 seconds

### Candidate
type: B, layer: 1, pos: 8

## Relational analysis of NS_A2_B2_A2_B1_B2_B1
Optimization infeasible because this subproblem isn't reachable.

### Candidate
type: A, layer: 1, pos: 8

### Candidate
type: B, layer: 1, pos: 5

### Candidate
type: A, layer: 1, pos: 5

### Candidate
type: A, layer: 1, pos: 38

## Relational analysis of NS_A2_B2_A2_B1_B2_A1
Optimization infeasible because this subproblem isn't reachable.

### Candidate
type: B, layer: 1, pos: 10

## Relational analysis of NS_A2_B2_A2_B1_B2_B1

### Relational analysis result of NS_A2_B2_A2_B1_B2_B1
Status: Status.UNKNOWN
Output dim: 0, lower bound: -913.6301185, upper bound: 914.7463779
time: 0.54 seconds

## Relational analysis of NS_A2_B2_A2_B1_B2_B2
Optimization infeasible because this subproblem isn't reachable.

### Candidate
type: A, layer: 1, pos: 10

### Candidate
type: B, layer: 1, pos: 38

### Candidate
type: A, layer: 1, pos: 18

## Relational analysis of NS_A2_B2_A2_B1_B2_A1

### Relational analysis result of NS_A2_B2_A2_B1_B2_A1
Status: Status.UNKNOWN
Output dim: 0, lower bound: -914.1047483, upper bound: 916.5482256
time: 0.49 seconds

## Relational analysis of NS_A2_B2_A2_B1_B2_A2

### Relational analysis result of NS_A2_B2_A2_B1_B2_A2
Status: Status.UNKNOWN
Output dim: 0, lower bound: -914.1047483, upper bound: 916.5976782
time: 0.51 seconds

## BFS NS instance: NS_A2_B2_A2_B2_A1

### Backsubstitution after applying NS history:
0: -478.8077698, 305.1115723, -520.1938477, 338.3971863, -817.2048340, 825.3054199
1: -35.9700279, 27.5468063, -39.5616646, 30.1879539, -66.1579742, 67.1084747
2: -30.2316093, 40.7472878, -32.7032356, 45.3272285, -75.5588226, 73.4505081
3: -33.8646011, 65.4005203, -36.9652023, 72.2003250, -106.0649261, 102.3657150
4: -25.9642067, 40.8872604, -28.1057796, 45.3838654, -71.3480682, 68.9930420

Time for backsubstitution: 3.04 seconds

### NS candidates at layer 1
type: A, layer: 1, pos: 8
type: B, layer: 1, pos: 8
type: B, layer: 1, pos: 35
type: B, layer: 1, pos: 5
type: A, layer: 1, pos: 5
type: B, layer: 1, pos: 38
type: A, layer: 1, pos: 10
type: B, layer: 1, pos: 10
type: A, layer: 1, pos: 38
type: B, layer: 1, pos: 47
type: A, layer: 1, pos: 21
type: B, layer: 1, pos: 21
type: A, layer: 1, pos: 47
type: B, layer: 1, pos: 3
type: A, layer: 1, pos: 3
type: B, layer: 1, pos: 15
type: A, layer: 1, pos: 15
type: B, layer: 1, pos: 30
type: A, layer: 1, pos: 30
type: A, layer: 1, pos: 24
type: B, layer: 1, pos: 24
type: B, layer: 1, pos: 1
type: A, layer: 1, pos: 1
type: B, layer: 1, pos: 14
type: A, layer: 1, pos: 14
type: A, layer: 1, pos: 40
type: B, layer: 1, pos: 40

Time for candidate selection: 0.25 seconds

### Candidate
type: A, layer: 1, pos: 8

## Relational analysis of NS_A2_B2_A2_B2_A1_A1
Optimization infeasible because this subproblem isn't reachable.

### Candidate
type: B, layer: 1, pos: 8

### Candidate
type: B, layer: 1, pos: 35

## Relational analysis of NS_A2_B2_A2_B2_A1_B1

### Relational analysis result of NS_A2_B2_A2_B2_A1_B1
Status: Status.UNKNOWN
Output dim: 0, lower bound: -914.1047483, upper bound: 916.2293075
time: 0.52 seconds

## Relational analysis of NS_A2_B2_A2_B2_A1_B2

### Relational analysis result of NS_A2_B2_A2_B2_A1_B2
Status: Status.UNKNOWN
Output dim: 0, lower bound: -914.1047483, upper bound: 916.5482262
time: 0.50 seconds

## BFS NS instance: NS_A2_B2_A2_B2_A2

### Backsubstitution after applying NS history:
0: -511.0637207, 331.1865845, -520.1938477, 338.3971863, -849.4608154, 851.3803101
1: -38.7831726, 29.6075211, -39.5616646, 30.1879539, -68.9711304, 69.1691895
2: -32.1094704, 44.3574219, -32.7032356, 45.3272285, -77.4366913, 77.0606537
3: -36.2618828, 70.7250671, -36.9652023, 72.2003250, -108.4622040, 107.6902618
4: -27.5750580, 44.4196701, -28.1057796, 45.3838654, -72.9589233, 72.5254517

Time for backsubstitution: 3.04 seconds

### NS candidates at layer 1
type: A, layer: 1, pos: 8
type: B, layer: 1, pos: 8
type: A, layer: 1, pos: 5
type: B, layer: 1, pos: 5
type: B, layer: 1, pos: 35
type: B, layer: 1, pos: 38
type: A, layer: 1, pos: 38
type: A, layer: 1, pos: 10
type: B, layer: 1, pos: 10
type: B, layer: 1, pos: 3
type: A, layer: 1, pos: 3
type: B, layer: 1, pos: 21
type: A, layer: 1, pos: 21
type: B, layer: 1, pos: 47
type: A, layer: 1, pos: 47
type: A, layer: 1, pos: 30
type: B, layer: 1, pos: 30
type: B, layer: 1, pos: 15
type: A, layer: 1, pos: 15
type: B, layer: 1, pos: 1
type: A, layer: 1, pos: 1
type: B, layer: 1, pos: 24
type: A, layer: 1, pos: 24
type: A, layer: 1, pos: 14
type: B, layer: 1, pos: 14
type: B, layer: 1, pos: 40
type: A, layer: 1, pos: 40

Time for candidate selection: 0.25 seconds

### Candidate
type: A, layer: 1, pos: 8

## Relational analysis of NS_A2_B2_A2_B2_A2_A1
Optimization infeasible because this subproblem isn't reachable.

### Candidate
type: B, layer: 1, pos: 8

### Candidate
type: A, layer: 1, pos: 5

## Relational analysis of NS_A2_B2_A2_B2_A2_A1
Optimization infeasible because this subproblem isn't reachable.

### Candidate
type: B, layer: 1, pos: 5

### Candidate
type: B, layer: 1, pos: 35

## Relational analysis of NS_A2_B2_A2_B2_A2_B1

### Relational analysis result of NS_A2_B2_A2_B2_A2_B1
Status: Status.UNKNOWN
Output dim: 0, lower bound: -914.1047483, upper bound: 916.4518879
time: 0.51 seconds

## Relational analysis of NS_A2_B2_A2_B2_A2_B2

### Relational analysis result of NS_A2_B2_A2_B2_A2_B2
Status: Status.UNKNOWN
Output dim: 0, lower bound: -914.1047483, upper bound: 916.6077273
time: 0.50 seconds

## Summary of splitting at layer (split count: 5)
- Time for NS candidates: 5.54 seconds
NS_A1_B1_A1_B1_A1_B1, status: Status.UNKNOWN, split count: 6, time: 5.54
Output dim: 0, lower bound: -912.7004530, upper bound: 914.4268375
NS_A1_B1_A1_B1_A1_B2, status: Status.UNKNOWN, split count: 6, time: 5.54
Output dim: 0, lower bound: -912.3628316, upper bound: 914.4189475
NS_A1_B1_A1_B2_B1_B1, status: Status.UNKNOWN, split count: 6, time: 5.54
Output dim: 0, lower bound: -911.7179774, upper bound: 914.0186715
NS_A1_B1_A1_B2_B1_B2, status: Status.UNKNOWN, split count: 6, time: 5.54
Output dim: 0, lower bound: -912.2883269, upper bound: 914.4298630
NS_A1_B1_A1_B2_B2_A1, status: Status.UNKNOWN, split count: 6, time: 5.54
Output dim: 0, lower bound: -914.1038146, upper bound: 911.0559777
NS_A1_B1_A1_B2_B2_A2, status: Status.UNKNOWN, split count: 6, time: 5.54
Output dim: 0, lower bound: -914.1038146, upper bound: 914.8657075
NS_A1_B1_A2_B1_A1_A1, status: Status.UNKNOWN, split count: 6, time: 5.54
Output dim: 0, lower bound: -914.0186715, upper bound: 911.7179774
NS_A1_B1_A2_B1_A1_A2, status: Status.UNKNOWN, split count: 6, time: 5.54
Output dim: 0, lower bound: -914.4298630, upper bound: 912.2883266
NS_A1_B1_A2_B1_A2_B1, status: Status.UNKNOWN, split count: 6, time: 5.54
Output dim: 0, lower bound: -911.0559777, upper bound: 914.1038146
NS_A1_B1_A2_B1_A2_B2, status: Status.UNKNOWN, split count: 6, time: 5.54
Output dim: 0, lower bound: -911.0559777, upper bound: 914.8548591
NS_A1_B1_A2_B2_A1_B1, status: Status.UNKNOWN, split count: 6, time: 5.54
Output dim: 0, lower bound: -913.5295563, upper bound: 913.5295563
NS_A1_B1_A2_B2_A1_B2, status: Status.UNKNOWN, split count: 6, time: 5.54
Output dim: 0, lower bound: -913.5295563, upper bound: 913.5295563
NS_A1_B1_A2_B2_A2_B1, status: Status.UNKNOWN, split count: 6, time: 5.54
Output dim: 0, lower bound: -913.5295563, upper bound: 914.6337209
NS_A1_B1_A2_B2_A2_B2, status: Status.UNKNOWN, split count: 6, time: 5.54
Output dim: 0, lower bound: -913.5295563, upper bound: 914.8548591
NS_A1_B2_B1_A1_B1_A1, status: Status.UNKNOWN, split count: 6, time: 5.54
Output dim: 0, lower bound: -915.8446039, upper bound: 914.2045524
NS_A1_B2_B1_A1_B1_A2, status: Status.UNKNOWN, split count: 6, time: 5.54
Output dim: 0, lower bound: -915.8598203, upper bound: 914.2533971
NS_A1_B2_B1_A1_B2_A1, status: Status.UNKNOWN, split count: 6, time: 5.54
Output dim: 0, lower bound: -912.7943344, upper bound: 913.7291939
NS_A1_B2_B1_A1_B2_A2, status: Status.VERIFIED, split count: 6, time: 5.54
Output dim: 0, lower bound: -912.7943344, upper bound: 912.7683715
NS_A1_B2_B1_A2_A1_B1, status: Status.VERIFIED, split count: 6, time: 5.54
Output dim: 0, lower bound: -912.7943344, upper bound: 912.8298472
NS_A1_B2_B1_A2_A1_B2, status: Status.VERIFIED, split count: 6, time: 5.54
Output dim: 0, lower bound: -912.7943344, upper bound: 912.8298472
NS_A1_B2_B2_A1_B1_A1, status: Status.UNKNOWN, split count: 6, time: 5.54
Output dim: 0, lower bound: -915.5498258, upper bound: 913.4911499
NS_A1_B2_B2_A1_B1_A2, status: Status.UNKNOWN, split count: 6, time: 5.54
Output dim: 0, lower bound: -916.1746993, upper bound: 914.6891401
NS_A1_B2_B2_A1_B2_B1, status: Status.UNKNOWN, split count: 6, time: 5.54
Output dim: 0, lower bound: -914.3409043, upper bound: 914.1697744
NS_A1_B2_B2_A1_B2_B2, status: Status.UNKNOWN, split count: 6, time: 5.54
Output dim: 0, lower bound: -916.3030290, upper bound: 914.9279862
NS_A1_B2_B2_A2_B1_A1, status: Status.UNKNOWN, split count: 6, time: 5.54
Output dim: 0, lower bound: -914.4943917, upper bound: 912.7972417
NS_A1_B2_B2_A2_B1_A2, status: Status.UNKNOWN, split count: 6, time: 5.54
Output dim: 0, lower bound: -914.4782008, upper bound: 912.5881746
NS_A2_B1_A1_B1_A1_B1, status: Status.UNKNOWN, split count: 6, time: 5.54
Output dim: 0, lower bound: -914.2045524, upper bound: 915.8446039
NS_A2_B1_A1_B1_A1_B2, status: Status.UNKNOWN, split count: 6, time: 5.54
Output dim: 0, lower bound: -914.2533971, upper bound: 915.8598194
NS_A2_B1_A1_B1_A2_B1, status: Status.UNKNOWN, split count: 6, time: 5.54
Output dim: 0, lower bound: -913.7291939, upper bound: 912.7943344
NS_A2_B1_A1_B1_A2_B2, status: Status.VERIFIED, split count: 6, time: 5.54
Output dim: 0, lower bound: -912.7683715, upper bound: 912.7943344
NS_A2_B1_A1_B2_B1_A1, status: Status.VERIFIED, split count: 6, time: 5.54
Output dim: 0, lower bound: -912.8298472, upper bound: 912.7943344
NS_A2_B1_A1_B2_B1_A2, status: Status.VERIFIED, split count: 6, time: 5.54
Output dim: 0, lower bound: -912.8298472, upper bound: 912.7943344
NS_A2_B1_A2_B1_A1_B1, status: Status.UNKNOWN, split count: 6, time: 5.54
Output dim: 0, lower bound: -913.4911499, upper bound: 915.5498258
NS_A2_B1_A2_B1_A1_B2, status: Status.UNKNOWN, split count: 6, time: 5.54
Output dim: 0, lower bound: -914.6891401, upper bound: 916.1746992
NS_A2_B1_A2_B1_A2_A1, status: Status.UNKNOWN, split count: 6, time: 5.54
Output dim: 0, lower bound: -914.1697744, upper bound: 914.3409043
NS_A2_B1_A2_B1_A2_A2, status: Status.UNKNOWN, split count: 6, time: 5.54
Output dim: 0, lower bound: -914.9279862, upper bound: 916.3030283
NS_A2_B1_A2_B2_A1_B1, status: Status.UNKNOWN, split count: 6, time: 5.54
Output dim: 0, lower bound: -912.6393855, upper bound: 914.4915282
NS_A2_B1_A2_B2_A1_B2, status: Status.UNKNOWN, split count: 6, time: 5.54
Output dim: 0, lower bound: -912.5728632, upper bound: 914.4782008
NS_A2_B1_A2_B2_A2_B1, status: Status.UNKNOWN, split count: 6, time: 5.54
Output dim: 0, lower bound: -912.9698978, upper bound: 912.9914879
NS_A2_B1_A2_B2_A2_B2, status: Status.UNKNOWN, split count: 6, time: 5.54
Output dim: 0, lower bound: -912.9698978, upper bound: 913.0607464
NS_A2_B2_A1_B1_A1_B1, status: Status.UNKNOWN, split count: 6, time: 5.54
Output dim: 0, lower bound: -914.1047483, upper bound: 914.1047483
NS_A2_B2_A1_B1_A1_B2, status: Status.UNKNOWN, split count: 6, time: 5.54
Output dim: 0, lower bound: -914.1047483, upper bound: 914.1047483
NS_A2_B2_A1_B1_A2_B1, status: Status.UNKNOWN, split count: 6, time: 5.54
Output dim: 0, lower bound: -914.1047483, upper bound: 914.6940041
NS_A2_B2_A1_B1_A2_B2, status: Status.UNKNOWN, split count: 6, time: 5.54
Output dim: 0, lower bound: -914.1047483, upper bound: 914.7501862
NS_A2_B2_A1_B2_A1_B1, status: Status.UNKNOWN, split count: 6, time: 5.54
Output dim: 0, lower bound: -914.1047483, upper bound: 914.1047482
NS_A2_B2_A1_B2_A1_B2, status: Status.UNKNOWN, split count: 6, time: 5.54
Output dim: 0, lower bound: -914.1047483, upper bound: 914.1047483
NS_A2_B2_A1_B2_A2_B1, status: Status.UNKNOWN, split count: 6, time: 5.54
Output dim: 0, lower bound: -914.1047483, upper bound: 915.1457901
NS_A2_B2_A1_B2_A2_B2, status: Status.UNKNOWN, split count: 6, time: 5.54
Output dim: 0, lower bound: -914.1047483, upper bound: 915.1457901
NS_A2_B2_A2_B1_B1_A1, status: Status.UNKNOWN, split count: 6, time: 5.54
Output dim: 0, lower bound: -914.1047483, upper bound: 915.8888488
NS_A2_B2_A2_B1_B1_A2, status: Status.UNKNOWN, split count: 6, time: 5.54
Output dim: 0, lower bound: -914.1047483, upper bound: 915.9322522
NS_A2_B2_A2_B1_B2_A1, status: Status.UNKNOWN, split count: 6, time: 5.54
Output dim: 0, lower bound: -914.1047483, upper bound: 916.5482256
NS_A2_B2_A2_B1_B2_A2, status: Status.UNKNOWN, split count: 6, time: 5.54
Output dim: 0, lower bound: -914.1047483, upper bound: 916.5976782
NS_A2_B2_A2_B2_A1_B1, status: Status.UNKNOWN, split count: 6, time: 5.54
Output dim: 0, lower bound: -914.1047483, upper bound: 916.2293075
NS_A2_B2_A2_B2_A1_B2, status: Status.UNKNOWN, split count: 6, time: 5.54
Output dim: 0, lower bound: -914.1047483, upper bound: 916.5482262
NS_A2_B2_A2_B2_A2_B1, status: Status.UNKNOWN, split count: 6, time: 5.54
Output dim: 0, lower bound: -914.1047483, upper bound: 916.4518879
NS_A2_B2_A2_B2_A2_B2, status: Status.UNKNOWN, split count: 6, time: 5.54
Output dim: 0, lower bound: -914.1047483, upper bound: 916.6077273

## BFS NS instance: NS_A1_B1_A1_B1_A1_B1

### Backsubstitution after applying NS history:
0: -434.5860291, 268.4107971, -437.5225220, 271.0540771, -705.6401367, 705.9332275
1: -31.9206696, 24.8383789, -32.1982918, 24.9978085, -56.9184685, 57.0366669
2: -27.6220016, 35.7143478, -27.7263260, 36.0323639, -63.6543617, 63.4406738
3: -30.8580379, 58.7825966, -30.9891148, 59.2409935, -90.0990295, 89.7716751
4: -23.7789459, 35.8797493, -23.8569965, 36.1721725, -59.9511147, 59.7367325

Time for backsubstitution: 3.00 seconds

### NS candidates at layer 1
type: B, layer: 1, pos: 38
type: A, layer: 1, pos: 15
type: A, layer: 1, pos: 38
type: B, layer: 1, pos: 21
type: A, layer: 1, pos: 21
type: B, layer: 1, pos: 8
type: B, layer: 1, pos: 10
type: A, layer: 1, pos: 5
type: A, layer: 1, pos: 8
type: B, layer: 1, pos: 47
type: B, layer: 1, pos: 5
type: A, layer: 1, pos: 10
type: B, layer: 1, pos: 15
type: A, layer: 1, pos: 24
type: A, layer: 1, pos: 35
type: B, layer: 1, pos: 30
type: A, layer: 1, pos: 3
type: B, layer: 1, pos: 3
type: A, layer: 1, pos: 1
type: B, layer: 1, pos: 24
type: B, layer: 1, pos: 1
type: A, layer: 1, pos: 30
type: A, layer: 1, pos: 14
type: B, layer: 1, pos: 14
type: A, layer: 1, pos: 40
type: B, layer: 1, pos: 40

Time for candidate selection: 0.25 seconds

### Candidate
type: B, layer: 1, pos: 38

### Candidate
type: A, layer: 1, pos: 15

## Relational analysis of NS_A1_B1_A1_B1_A1_B1_A1
Optimization infeasible because this subproblem isn't reachable.

### Candidate
type: A, layer: 1, pos: 38

### Candidate
type: B, layer: 1, pos: 21

### Candidate
type: A, layer: 1, pos: 21

### Candidate
type: B, layer: 1, pos: 8

### Candidate
type: B, layer: 1, pos: 10

## Relational analysis of NS_A1_B1_A1_B1_A1_B1_B1

### Relational analysis result of NS_A1_B1_A1_B1_A1_B1_B1
Status: Status.UNKNOWN
Output dim: 0, lower bound: -912.6176020, upper bound: 914.3728162
time: 0.51 seconds

## Relational analysis of NS_A1_B1_A1_B1_A1_B1_B2

### Relational analysis result of NS_A1_B1_A1_B1_A1_B1_B2
Status: Status.UNKNOWN
Output dim: 0, lower bound: -912.4419034, upper bound: 914.4268375
time: 0.51 seconds

## BFS NS instance: NS_A1_B1_A1_B1_A1_B2

### Backsubstitution after applying NS history:
0: -438.8737488, 272.9502563, -455.5523071, 286.7955322, -725.6692505, 728.5025024
1: -32.3863754, 25.1283436, -33.8574944, 26.1630859, -58.5494614, 58.9858360
2: -27.9357700, 36.2967033, -28.8621521, 38.0937462, -66.0295181, 65.1588440
3: -31.2315922, 59.4788399, -32.3181038, 61.9380798, -93.1696625, 91.7969208
4: -24.0411091, 36.4608269, -24.8068333, 38.2403984, -62.2815094, 61.2676544

Time for backsubstitution: 2.99 seconds

### NS candidates at layer 1
type: A, layer: 1, pos: 21
type: B, layer: 1, pos: 21
type: B, layer: 1, pos: 38
type: A, layer: 1, pos: 38
type: B, layer: 1, pos: 8
type: A, layer: 1, pos: 8
type: B, layer: 1, pos: 5
type: A, layer: 1, pos: 5
type: A, layer: 1, pos: 15
type: B, layer: 1, pos: 10
type: A, layer: 1, pos: 10
type: B, layer: 1, pos: 47
type: A, layer: 1, pos: 35
type: B, layer: 1, pos: 30
type: B, layer: 1, pos: 15
type: B, layer: 1, pos: 3
type: A, layer: 1, pos: 3
type: A, layer: 1, pos: 24
type: A, layer: 1, pos: 1
type: A, layer: 1, pos: 30
type: B, layer: 1, pos: 1
type: B, layer: 1, pos: 24
type: A, layer: 1, pos: 14
type: B, layer: 1, pos: 14
type: A, layer: 1, pos: 40
type: B, layer: 1, pos: 40

Time for candidate selection: 0.24 seconds

### Candidate
type: A, layer: 1, pos: 21

### Candidate
type: B, layer: 1, pos: 21

### Candidate
type: B, layer: 1, pos: 38

### Candidate
type: A, layer: 1, pos: 38

### Candidate
type: B, layer: 1, pos: 8

### Candidate
type: A, layer: 1, pos: 8

### Candidate
type: B, layer: 1, pos: 5

## Relational analysis of NS_A1_B1_A1_B1_A1_B2_B1
Optimization infeasible because this subproblem isn't reachable.

### Candidate
type: A, layer: 1, pos: 5

### Candidate
type: A, layer: 1, pos: 15

## Relational analysis of NS_A1_B1_A1_B1_A1_B2_A1
Optimization infeasible because this subproblem isn't reachable.

### Candidate
type: B, layer: 1, pos: 10

## Relational analysis of NS_A1_B1_A1_B1_A1_B2_B1

### Relational analysis result of NS_A1_B1_A1_B1_A1_B2_B1
Status: Status.UNKNOWN
Output dim: 0, lower bound: -912.3628316, upper bound: 914.2749616
time: 0.49 seconds

## Relational analysis of NS_A1_B1_A1_B1_A1_B2_B2

### Relational analysis result of NS_A1_B1_A1_B1_A1_B2_B2
Status: Status.UNKNOWN
Output dim: 0, lower bound: -912.2688969, upper bound: 914.4189475
time: 0.48 seconds

## BFS NS instance: NS_A1_B1_A1_B2_B1_B1

### Backsubstitution after applying NS history:
0: -447.2741394, 278.1007385, -440.3682861, 269.2666626, -716.5407715, 718.4689941
1: -32.9486160, 25.6265907, -32.0260201, 25.1562080, -58.1048203, 57.6526108
2: -28.2855682, 36.9536285, -27.6412544, 35.8669624, -64.1525269, 64.5948792
3: -31.6387711, 60.6078339, -30.9664783, 59.4783897, -91.1171570, 91.5743103
4: -24.3149261, 37.1056862, -23.7714996, 35.9690704, -60.2839966, 60.8771782

Time for backsubstitution: 2.93 seconds

### NS candidates at layer 1
type: B, layer: 1, pos: 8
type: A, layer: 1, pos: 8
type: B, layer: 1, pos: 47
type: B, layer: 1, pos: 38
type: A, layer: 1, pos: 15
type: A, layer: 1, pos: 24
type: A, layer: 1, pos: 38
type: A, layer: 1, pos: 47
type: A, layer: 1, pos: 1
type: B, layer: 1, pos: 15
type: B, layer: 1, pos: 1
type: A, layer: 1, pos: 21
type: B, layer: 1, pos: 10
type: B, layer: 1, pos: 30
type: B, layer: 1, pos: 21
type: A, layer: 1, pos: 10
type: A, layer: 1, pos: 35
type: A, layer: 1, pos: 5
type: B, layer: 1, pos: 24
type: A, layer: 1, pos: 3
type: B, layer: 1, pos: 3
type: A, layer: 1, pos: 30
type: A, layer: 1, pos: 14
type: B, layer: 1, pos: 40
type: A, layer: 1, pos: 40
type: B, layer: 1, pos: 14

Time for candidate selection: 0.24 seconds

### Candidate
type: B, layer: 1, pos: 8

### Candidate
type: A, layer: 1, pos: 8

### Candidate
type: B, layer: 1, pos: 47

## Relational analysis of NS_A1_B1_A1_B2_B1_B1_B1

### Relational analysis result of NS_A1_B1_A1_B2_B1_B1_B1
Status: Status.VERIFIED
Output dim: 0, lower bound: -910.0144011, upper bound: 912.2040577
time: 0.53 seconds

## Relational analysis of NS_A1_B1_A1_B2_B1_B1_B2

### Relational analysis result of NS_A1_B1_A1_B2_B1_B1_B2
Status: Status.VERIFIED
Output dim: 0, lower bound: -909.7541680, upper bound: 912.2040577
time: 0.46 seconds

## BFS NS instance: NS_A1_B1_A1_B2_B1_B2

### Backsubstitution after applying NS history:
0: -452.2647705, 283.0553589, -455.9036865, 283.0632324, -735.3278198, 738.9590454
1: -33.4665756, 25.9531231, -33.4761047, 26.1830158, -59.6495895, 59.4292297
2: -28.6231174, 37.5950356, -28.6366215, 37.6613274, -66.2844467, 66.2316437
3: -32.0387459, 61.3902779, -32.1303596, 61.8211899, -93.8599396, 93.5206375
4: -24.5955162, 37.7479630, -24.5983639, 37.7725945, -62.3681107, 62.3463287

Time for backsubstitution: 2.93 seconds

### NS candidates at layer 1
type: B, layer: 1, pos: 8
type: A, layer: 1, pos: 8
type: B, layer: 1, pos: 30
type: B, layer: 1, pos: 38
type: A, layer: 1, pos: 38
type: A, layer: 1, pos: 21
type: B, layer: 1, pos: 21
type: A, layer: 1, pos: 24
type: A, layer: 1, pos: 1
type: A, layer: 1, pos: 15
type: A, layer: 1, pos: 35
type: B, layer: 1, pos: 1
type: A, layer: 1, pos: 5
type: B, layer: 1, pos: 10
type: B, layer: 1, pos: 15
type: B, layer: 1, pos: 47
type: A, layer: 1, pos: 30
type: A, layer: 1, pos: 47
type: A, layer: 1, pos: 10
type: A, layer: 1, pos: 3
type: B, layer: 1, pos: 3
type: B, layer: 1, pos: 24
type: A, layer: 1, pos: 40
type: B, layer: 1, pos: 40
type: A, layer: 1, pos: 14
type: B, layer: 1, pos: 14

Time for candidate selection: 0.24 seconds

### Candidate
type: B, layer: 1, pos: 8

### Candidate
type: A, layer: 1, pos: 8

### Candidate
type: B, layer: 1, pos: 30

### Candidate
type: B, layer: 1, pos: 38

### Candidate
type: A, layer: 1, pos: 38

### Candidate
type: A, layer: 1, pos: 21

### Candidate
type: B, layer: 1, pos: 21

### Candidate
type: A, layer: 1, pos: 24

## Relational analysis of NS_A1_B1_A1_B2_B1_B2_A1
Optimization infeasible because this subproblem isn't reachable.

### Candidate
type: A, layer: 1, pos: 1

## Relational analysis of NS_A1_B1_A1_B2_B1_B2_A1
Optimization infeasible because this subproblem isn't reachable.

### Candidate
type: A, layer: 1, pos: 15

## Relational analysis of NS_A1_B1_A1_B2_B1_B2_A1

### Relational analysis result of NS_A1_B1_A1_B2_B1_B2_A1
Status: Status.UNKNOWN
Output dim: 0, lower bound: -912.1290806, upper bound: 914.3781155
time: 0.48 seconds

## Relational analysis of NS_A1_B1_A1_B2_B1_B2_A2

### Relational analysis result of NS_A1_B1_A1_B2_B1_B2_A2
Status: Status.UNKNOWN
Output dim: 0, lower bound: -912.2835885, upper bound: 914.3795664
time: 0.49 seconds

## BFS NS instance: NS_A1_B1_A1_B2_B2_A1

### Backsubstitution after applying NS history:
0: -443.6393127, 273.8376770, -487.9168091, 314.3346558, -757.9739990, 761.7543945
1: -32.4688263, 25.4613533, -36.8756714, 28.1965370, -60.6653633, 62.3370247
2: -27.9574223, 36.3912735, -30.9391079, 41.8894653, -69.8468857, 67.3303833
3: -31.3120842, 60.0954323, -34.8926201, 67.1765060, -98.4885864, 94.9880524
4: -24.0433693, 36.5230064, -26.5249214, 41.9749985, -66.0183716, 63.0479279

Time for backsubstitution: 2.95 seconds

### NS candidates at layer 1
type: A, layer: 1, pos: 21
type: B, layer: 1, pos: 8
type: B, layer: 1, pos: 21
type: B, layer: 1, pos: 38
type: A, layer: 1, pos: 8
type: B, layer: 1, pos: 10
type: A, layer: 1, pos: 38
type: A, layer: 1, pos: 10
type: B, layer: 1, pos: 35
type: B, layer: 1, pos: 47
type: A, layer: 1, pos: 47
type: A, layer: 1, pos: 35
type: B, layer: 1, pos: 3
type: A, layer: 1, pos: 15
type: B, layer: 1, pos: 15
type: A, layer: 1, pos: 3
type: B, layer: 1, pos: 24
type: A, layer: 1, pos: 30
type: B, layer: 1, pos: 30
type: A, layer: 1, pos: 24
type: B, layer: 1, pos: 1
type: A, layer: 1, pos: 1
type: B, layer: 1, pos: 14
type: A, layer: 1, pos: 14
type: B, layer: 1, pos: 40
type: A, layer: 1, pos: 40

Time for candidate selection: 0.24 seconds

### Candidate
type: A, layer: 1, pos: 21

## Relational analysis of NS_A1_B1_A1_B2_B2_A1_A1
Optimization infeasible because this subproblem isn't reachable.

### Candidate
type: B, layer: 1, pos: 8

### Candidate
type: B, layer: 1, pos: 21

### Candidate
type: B, layer: 1, pos: 38

## Relational analysis of NS_A1_B1_A1_B2_B2_A1_B1
Optimization infeasible because this subproblem isn't reachable.

### Candidate
type: A, layer: 1, pos: 8

### Candidate
type: B, layer: 1, pos: 10

## Relational analysis of NS_A1_B1_A1_B2_B2_A1_B1

### Relational analysis result of NS_A1_B1_A1_B2_B2_A1_B1
Status: Status.VERIFIED
Output dim: 0, lower bound: -912.0034651, upper bound: 910.6107966
time: 0.49 seconds

## Relational analysis of NS_A1_B1_A1_B2_B2_A1_B2

### Relational analysis result of NS_A1_B1_A1_B2_B2_A1_B2
Status: Status.VERIFIED
Output dim: 0, lower bound: -911.0867913, upper bound: 910.6107962
time: 0.49 seconds

## BFS NS instance: NS_A1_B1_A1_B2_B2_A2

### Backsubstitution after applying NS history:
0: -448.4552002, 281.0086365, -487.9168091, 314.3346558, -762.7897949, 768.9254150
1: -33.2473564, 25.7387543, -36.8756714, 28.1965370, -61.4438934, 62.6144142
2: -28.4261436, 37.3166351, -30.9391079, 41.8894653, -70.3156128, 68.2557449
3: -31.8153515, 60.8366051, -34.8926201, 67.1765060, -98.9918442, 95.7292252
4: -24.4282951, 37.4797440, -26.5249214, 41.9749985, -66.4032898, 64.0046692

Time for backsubstitution: 2.95 seconds

### NS candidates at layer 1
type: B, layer: 1, pos: 8
type: A, layer: 1, pos: 8
type: A, layer: 1, pos: 21
type: B, layer: 1, pos: 21
type: B, layer: 1, pos: 38
type: A, layer: 1, pos: 38
type: A, layer: 1, pos: 15
type: B, layer: 1, pos: 47
type: B, layer: 1, pos: 35
type: B, layer: 1, pos: 10
type: A, layer: 1, pos: 47
type: A, layer: 1, pos: 35
type: B, layer: 1, pos: 15
type: B, layer: 1, pos: 30
type: A, layer: 1, pos: 10
type: B, layer: 1, pos: 3
type: A, layer: 1, pos: 3
type: A, layer: 1, pos: 30
type: A, layer: 1, pos: 24
type: B, layer: 1, pos: 24
type: B, layer: 1, pos: 1
type: A, layer: 1, pos: 1
type: A, layer: 1, pos: 14
type: B, layer: 1, pos: 14
type: A, layer: 1, pos: 40
type: B, layer: 1, pos: 40

Time for candidate selection: 0.25 seconds

### Candidate
type: B, layer: 1, pos: 8

### Candidate
type: A, layer: 1, pos: 8

### Candidate
type: A, layer: 1, pos: 21

## Relational analysis of NS_A1_B1_A1_B2_B2_A2_A1
Optimization infeasible because this subproblem isn't reachable.

### Candidate
type: B, layer: 1, pos: 21

### Candidate
type: B, layer: 1, pos: 38

## Relational analysis of NS_A1_B1_A1_B2_B2_A2_B1
Optimization infeasible because this subproblem isn't reachable.

### Candidate
type: A, layer: 1, pos: 38

### Candidate
type: A, layer: 1, pos: 15

## Relational analysis of NS_A1_B1_A1_B2_B2_A2_A1
Optimization infeasible because this subproblem isn't reachable.

### Candidate
type: B, layer: 1, pos: 47

## Relational analysis of NS_A1_B1_A1_B2_B2_A2_B1

### Relational analysis result of NS_A1_B1_A1_B2_B2_A2_B1
Status: Status.VERIFIED
Output dim: 0, lower bound: -911.7636763, upper bound: 912.6127237
time: 0.56 seconds

## Relational analysis of NS_A1_B1_A1_B2_B2_A2_B2

### Relational analysis result of NS_A1_B1_A1_B2_B2_A2_B2
Status: Status.VERIFIED
Output dim: 0, lower bound: -910.7130677, upper bound: 912.6127237
time: 0.55 seconds

## BFS NS instance: NS_A1_B1_A2_B1_A1_A1

### Backsubstitution after applying NS history:
0: -440.3682861, 269.2666626, -447.2741394, 278.1007385, -718.4689941, 716.5407715
1: -32.0260201, 25.1562080, -32.9486160, 25.6265907, -57.6526108, 58.1048203
2: -27.6412544, 35.8669624, -28.2855682, 36.9536285, -64.5948792, 64.1525269
3: -30.9664783, 59.4783897, -31.6387711, 60.6078339, -91.5743103, 91.1171570
4: -23.7714996, 35.9690704, -24.3149261, 37.1056862, -60.8771782, 60.2839966

Time for backsubstitution: 2.98 seconds

### NS candidates at layer 1
type: A, layer: 1, pos: 8
type: B, layer: 1, pos: 8
type: A, layer: 1, pos: 47
type: A, layer: 1, pos: 38
type: B, layer: 1, pos: 15
type: B, layer: 1, pos: 24
type: B, layer: 1, pos: 38
type: B, layer: 1, pos: 47
type: B, layer: 1, pos: 1
type: A, layer: 1, pos: 15
type: A, layer: 1, pos: 1
type: B, layer: 1, pos: 21
type: A, layer: 1, pos: 10
type: A, layer: 1, pos: 30
type: A, layer: 1, pos: 21
type: B, layer: 1, pos: 10
type: B, layer: 1, pos: 35
type: B, layer: 1, pos: 5
type: A, layer: 1, pos: 24
type: B, layer: 1, pos: 3
type: A, layer: 1, pos: 3
type: B, layer: 1, pos: 30
type: B, layer: 1, pos: 14
type: A, layer: 1, pos: 40
type: B, layer: 1, pos: 40
type: A, layer: 1, pos: 14

Time for candidate selection: 0.25 seconds

### Candidate
type: A, layer: 1, pos: 8

### Candidate
type: B, layer: 1, pos: 8

### Candidate
type: A, layer: 1, pos: 47

## Relational analysis of NS_A1_B1_A2_B1_A1_A1_A1

### Relational analysis result of NS_A1_B1_A2_B1_A1_A1_A1
Status: Status.VERIFIED
Output dim: 0, lower bound: -912.2040577, upper bound: 910.0144011
time: 0.49 seconds

## Relational analysis of NS_A1_B1_A2_B1_A1_A1_A2

### Relational analysis result of NS_A1_B1_A2_B1_A1_A1_A2
Status: Status.VERIFIED
Output dim: 0, lower bound: -912.2040577, upper bound: 909.7541680
time: 0.50 seconds

## BFS NS instance: NS_A1_B1_A2_B1_A1_A2

### Backsubstitution after applying NS history:
0: -455.9036865, 283.0632324, -452.2647705, 283.0553589, -738.9590454, 735.3278198
1: -33.4761047, 26.1830158, -33.4665756, 25.9531231, -59.4292297, 59.6495895
2: -28.6366215, 37.6613274, -28.6231174, 37.5950356, -66.2316513, 66.2844391
3: -32.1303596, 61.8211899, -32.0387459, 61.3902779, -93.5206375, 93.8599396
4: -24.5983639, 37.7725945, -24.5955162, 37.7479630, -62.3463287, 62.3681107

Time for backsubstitution: 2.98 seconds

### NS candidates at layer 1
type: A, layer: 1, pos: 8
type: B, layer: 1, pos: 8
type: A, layer: 1, pos: 30
type: A, layer: 1, pos: 38
type: B, layer: 1, pos: 38
type: B, layer: 1, pos: 21
type: A, layer: 1, pos: 21
type: B, layer: 1, pos: 24
type: B, layer: 1, pos: 1
type: B, layer: 1, pos: 15
type: B, layer: 1, pos: 35
type: A, layer: 1, pos: 1
type: B, layer: 1, pos: 5
type: A, layer: 1, pos: 10
type: A, layer: 1, pos: 15
type: A, layer: 1, pos: 47
type: B, layer: 1, pos: 30
type: B, layer: 1, pos: 47
type: B, layer: 1, pos: 10
type: B, layer: 1, pos: 3
type: A, layer: 1, pos: 3
type: A, layer: 1, pos: 24
type: B, layer: 1, pos: 40
type: A, layer: 1, pos: 40
type: B, layer: 1, pos: 14
type: A, layer: 1, pos: 14

Time for candidate selection: 0.25 seconds

### Candidate
type: A, layer: 1, pos: 8

### Candidate
type: B, layer: 1, pos: 8

### Candidate
type: A, layer: 1, pos: 30

### Candidate
type: A, layer: 1, pos: 38

### Candidate
type: B, layer: 1, pos: 38

### Candidate
type: B, layer: 1, pos: 21

### Candidate
type: A, layer: 1, pos: 21

### Candidate
type: B, layer: 1, pos: 24

## Relational analysis of NS_A1_B1_A2_B1_A1_A2_B1
Optimization infeasible because this subproblem isn't reachable.

### Candidate
type: B, layer: 1, pos: 1

## Relational analysis of NS_A1_B1_A2_B1_A1_A2_B1
Optimization infeasible because this subproblem isn't reachable.

### Candidate
type: B, layer: 1, pos: 15

## Relational analysis of NS_A1_B1_A2_B1_A1_A2_B1

### Relational analysis result of NS_A1_B1_A2_B1_A1_A2_B1
Status: Status.UNKNOWN
Output dim: 0, lower bound: -914.3781155, upper bound: 912.1290806
time: 0.49 seconds

## Relational analysis of NS_A1_B1_A2_B1_A1_A2_B2

### Relational analysis result of NS_A1_B1_A2_B1_A1_A2_B2
Status: Status.UNKNOWN
Output dim: 0, lower bound: -914.3795664, upper bound: 912.2835881
time: 0.49 seconds

## BFS NS instance: NS_A1_B1_A2_B1_A2_B1

### Backsubstitution after applying NS history:
0: -487.9168091, 314.3346558, -443.6393127, 273.8376770, -761.7543945, 757.9739990
1: -36.8756714, 28.1965370, -32.4688263, 25.4613533, -62.3370247, 60.6653633
2: -30.9391079, 41.8894653, -27.9574223, 36.3912735, -67.3303757, 69.8468857
3: -34.8926201, 67.1765060, -31.3120842, 60.0954323, -94.9880524, 98.4885864
4: -26.5249214, 41.9749985, -24.0433693, 36.5230064, -63.0479279, 66.0183716

Time for backsubstitution: 2.98 seconds

### NS candidates at layer 1
type: B, layer: 1, pos: 21
type: A, layer: 1, pos: 8
type: A, layer: 1, pos: 21
type: A, layer: 1, pos: 38
type: B, layer: 1, pos: 8
type: A, layer: 1, pos: 10
type: B, layer: 1, pos: 38
type: B, layer: 1, pos: 10
type: A, layer: 1, pos: 35
type: A, layer: 1, pos: 47
type: B, layer: 1, pos: 47
type: B, layer: 1, pos: 35
type: A, layer: 1, pos: 3
type: B, layer: 1, pos: 15
type: A, layer: 1, pos: 15
type: B, layer: 1, pos: 3
type: A, layer: 1, pos: 24
type: B, layer: 1, pos: 30
type: A, layer: 1, pos: 30
type: B, layer: 1, pos: 24
type: A, layer: 1, pos: 1
type: B, layer: 1, pos: 1
type: A, layer: 1, pos: 14
type: B, layer: 1, pos: 14
type: A, layer: 1, pos: 40
type: B, layer: 1, pos: 40

Time for candidate selection: 0.25 seconds

### Candidate
type: B, layer: 1, pos: 21

## Relational analysis of NS_A1_B1_A2_B1_A2_B1_B1
Optimization infeasible because this subproblem isn't reachable.

### Candidate
type: A, layer: 1, pos: 8

### Candidate
type: A, layer: 1, pos: 21

### Candidate
type: A, layer: 1, pos: 38

## Relational analysis of NS_A1_B1_A2_B1_A2_B1_A1
Optimization infeasible because this subproblem isn't reachable.

### Candidate
type: B, layer: 1, pos: 8

### Candidate
type: A, layer: 1, pos: 10

## Relational analysis of NS_A1_B1_A2_B1_A2_B1_A1

### Relational analysis result of NS_A1_B1_A2_B1_A2_B1_A1
Status: Status.UNKNOWN
Output dim: 0, lower bound: -910.6107966, upper bound: 913.8721940
time: 0.46 seconds

## Relational analysis of NS_A1_B1_A2_B1_A2_B1_A2

### Relational analysis result of NS_A1_B1_A2_B1_A2_B1_A2
Status: Status.UNKNOWN
Output dim: 0, lower bound: -910.6107966, upper bound: 913.7902280
time: 0.46 seconds

## BFS NS instance: NS_A1_B1_A2_B1_A2_B2

### Backsubstitution after applying NS history:
0: -487.9168091, 314.3346558, -448.4552002, 281.0086365, -768.9254150, 762.7897949
1: -36.8756714, 28.1965370, -33.2473564, 25.7387543, -62.6144104, 61.4438934
2: -30.9391079, 41.8894653, -28.4261436, 37.3166351, -68.2557449, 70.3156052
3: -34.8926201, 67.1765060, -31.8153515, 60.8366051, -95.7292252, 98.9918518
4: -26.5249214, 41.9749985, -24.4282951, 37.4797440, -64.0046692, 66.4032898

Time for backsubstitution: 3.00 seconds

### NS candidates at layer 1
type: A, layer: 1, pos: 8
type: B, layer: 1, pos: 8
type: B, layer: 1, pos: 21
type: A, layer: 1, pos: 21
type: A, layer: 1, pos: 38
type: B, layer: 1, pos: 38
type: B, layer: 1, pos: 15
type: A, layer: 1, pos: 47
type: A, layer: 1, pos: 35
type: A, layer: 1, pos: 10
type: B, layer: 1, pos: 47
type: B, layer: 1, pos: 35
type: A, layer: 1, pos: 15
type: A, layer: 1, pos: 30
type: B, layer: 1, pos: 10
type: A, layer: 1, pos: 3
type: B, layer: 1, pos: 3
type: B, layer: 1, pos: 30
type: B, layer: 1, pos: 24
type: A, layer: 1, pos: 24
type: A, layer: 1, pos: 1
type: B, layer: 1, pos: 1
type: B, layer: 1, pos: 14
type: A, layer: 1, pos: 14
type: B, layer: 1, pos: 40
type: A, layer: 1, pos: 40

Time for candidate selection: 0.25 seconds

### Candidate
type: A, layer: 1, pos: 8

### Candidate
type: B, layer: 1, pos: 8

### Candidate
type: B, layer: 1, pos: 21

## Relational analysis of NS_A1_B1_A2_B1_A2_B2_B1
Optimization infeasible because this subproblem isn't reachable.

### Candidate
type: A, layer: 1, pos: 21

### Candidate
type: A, layer: 1, pos: 38

## Relational analysis of NS_A1_B1_A2_B1_A2_B2_A1
Optimization infeasible because this subproblem isn't reachable.

### Candidate
type: B, layer: 1, pos: 38

### Candidate
type: B, layer: 1, pos: 15

## Relational analysis of NS_A1_B1_A2_B1_A2_B2_B1
Optimization infeasible because this subproblem isn't reachable.

### Candidate
type: A, layer: 1, pos: 47

## Relational analysis of NS_A1_B1_A2_B1_A2_B2_A1

### Relational analysis result of NS_A1_B1_A2_B1_A2_B2_A1
Status: Status.UNKNOWN
Output dim: 0, lower bound: -909.9689814, upper bound: 914.2650981
time: 0.52 seconds

## Relational analysis of NS_A1_B1_A2_B1_A2_B2_A2

### Relational analysis result of NS_A1_B1_A2_B1_A2_B2_A2
Status: Status.VERIFIED
Output dim: 0, lower bound: -909.9689814, upper bound: 912.6981000
time: 0.51 seconds

## BFS NS instance: NS_A1_B1_A2_B2_A1_B1

### Backsubstitution after applying NS history:
0: -462.6254578, 289.2396240, -462.6254578, 289.2396240, -751.8651123, 751.8651123
1: -34.1428070, 26.6093521, -34.1428070, 26.6093521, -60.7521553, 60.7521553
2: -29.0894375, 38.4800377, -29.0894375, 38.4800377, -67.5694733, 67.5694656
3: -32.6629601, 62.8527527, -32.6629601, 62.8527527, -95.5157013, 95.5157013
4: -24.9707031, 38.5878029, -24.9707031, 38.5878029, -63.5585060, 63.5584946

Time for backsubstitution: 2.99 seconds

### NS candidates at layer 1
type: B, layer: 1, pos: 8
type: A, layer: 1, pos: 8
type: A, layer: 1, pos: 38
type: B, layer: 1, pos: 38
type: A, layer: 1, pos: 30
type: B, layer: 1, pos: 30
type: A, layer: 1, pos: 21
type: B, layer: 1, pos: 21
type: B, layer: 1, pos: 35
type: A, layer: 1, pos: 35
type: B, layer: 1, pos: 15
type: A, layer: 1, pos: 15
type: A, layer: 1, pos: 10
type: B, layer: 1, pos: 10
type: A, layer: 1, pos: 3
type: B, layer: 1, pos: 3
type: A, layer: 1, pos: 47
type: B, layer: 1, pos: 47
type: B, layer: 1, pos: 1
type: A, layer: 1, pos: 1
type: A, layer: 1, pos: 24
type: B, layer: 1, pos: 24
type: A, layer: 1, pos: 40
type: B, layer: 1, pos: 40
type: A, layer: 1, pos: 14
type: B, layer: 1, pos: 14

Time for candidate selection: 0.25 seconds

### Candidate
type: B, layer: 1, pos: 8

### Candidate
type: A, layer: 1, pos: 8

### Candidate
type: A, layer: 1, pos: 38

## Relational analysis of NS_A1_B1_A2_B2_A1_B1_A1

### Relational analysis result of NS_A1_B1_A2_B2_A1_B1_A1
Status: Status.UNKNOWN
Output dim: 0, lower bound: -913.3449900, upper bound: 913.3894010
time: 0.48 seconds

## Relational analysis of NS_A1_B1_A2_B2_A1_B1_A2

### Relational analysis result of NS_A1_B1_A2_B2_A1_B1_A2
Status: Status.UNKNOWN
Output dim: 0, lower bound: -913.3449900, upper bound: 913.3449900
time: 0.48 seconds

## BFS NS instance: NS_A1_B1_A2_B2_A1_B2

### Backsubstitution after applying NS history:
0: -462.6254578, 289.2396240, -487.9168091, 314.3346558, -776.9600830, 777.1564331
1: -34.1428070, 26.6093521, -36.8756714, 28.1965370, -62.3393440, 63.4850235
2: -29.0894375, 38.4800377, -30.9391079, 41.8894653, -70.9789047, 69.4191437
3: -32.6629601, 62.8527527, -34.8926201, 67.1765060, -99.8394623, 97.7453613
4: -24.9707031, 38.5878029, -26.5249214, 41.9749985, -66.9456940, 65.1127243

Time for backsubstitution: 2.98 seconds

### NS candidates at layer 1
type: A, layer: 1, pos: 8
type: B, layer: 1, pos: 8
type: B, layer: 1, pos: 21
type: A, layer: 1, pos: 21
type: A, layer: 1, pos: 38
type: B, layer: 1, pos: 38
type: B, layer: 1, pos: 15
type: A, layer: 1, pos: 47
type: A, layer: 1, pos: 30
type: B, layer: 1, pos: 35
type: B, layer: 1, pos: 47
type: A, layer: 1, pos: 10
type: A, layer: 1, pos: 35
type: A, layer: 1, pos: 15
type: B, layer: 1, pos: 10
type: B, layer: 1, pos: 3
type: A, layer: 1, pos: 3
type: B, layer: 1, pos: 30
type: B, layer: 1, pos: 24
type: B, layer: 1, pos: 1
type: A, layer: 1, pos: 1
type: A, layer: 1, pos: 24
type: B, layer: 1, pos: 14
type: A, layer: 1, pos: 14
type: A, layer: 1, pos: 40
type: B, layer: 1, pos: 40

Time for candidate selection: 0.25 seconds

### Candidate
type: A, layer: 1, pos: 8

### Candidate
type: B, layer: 1, pos: 8

### Candidate
type: B, layer: 1, pos: 21

## Relational analysis of NS_A1_B1_A2_B2_A1_B2_B1
Optimization infeasible because this subproblem isn't reachable.

### Candidate
type: A, layer: 1, pos: 21

### Candidate
type: A, layer: 1, pos: 38

## Relational analysis of NS_A1_B1_A2_B2_A1_B2_A1
Optimization infeasible because this subproblem isn't reachable.

### Candidate
type: B, layer: 1, pos: 38

### Candidate
type: B, layer: 1, pos: 15

## Relational analysis of NS_A1_B1_A2_B2_A1_B2_B1

### Relational analysis result of NS_A1_B1_A2_B2_A1_B2_B1
Status: Status.UNKNOWN
Output dim: 0, lower bound: -913.4179838, upper bound: 913.5295563
time: 0.48 seconds

## Relational analysis of NS_A1_B1_A2_B2_A1_B2_B2

### Relational analysis result of NS_A1_B1_A2_B2_A1_B2_B2
Status: Status.UNKNOWN
Output dim: 0, lower bound: -913.5295563, upper bound: 913.5295563
time: 0.47 seconds

## BFS NS instance: NS_A1_B1_A2_B2_A2_B1

### Backsubstitution after applying NS history:
0: -487.9168091, 314.3346558, -462.6254578, 289.2396240, -777.1564331, 776.9600830
1: -36.8756714, 28.1965370, -34.1428070, 26.6093521, -63.4850197, 62.3393440
2: -30.9391079, 41.8894653, -29.0894375, 38.4800377, -69.4191437, 70.9789047
3: -34.8926201, 67.1765060, -32.6629601, 62.8527527, -97.7453613, 99.8394623
4: -26.5249214, 41.9749985, -24.9707031, 38.5878029, -65.1127243, 66.9456940

Time for backsubstitution: 3.00 seconds

### NS candidates at layer 1
type: B, layer: 1, pos: 8
type: A, layer: 1, pos: 8
type: A, layer: 1, pos: 21
type: B, layer: 1, pos: 21
type: B, layer: 1, pos: 38
type: A, layer: 1, pos: 38
type: A, layer: 1, pos: 15
type: B, layer: 1, pos: 47
type: B, layer: 1, pos: 30
type: A, layer: 1, pos: 35
type: A, layer: 1, pos: 47
type: B, layer: 1, pos: 10
type: B, layer: 1, pos: 35
type: B, layer: 1, pos: 15
type: A, layer: 1, pos: 10
type: A, layer: 1, pos: 3
type: B, layer: 1, pos: 3
type: A, layer: 1, pos: 30
type: A, layer: 1, pos: 24
type: A, layer: 1, pos: 1
type: B, layer: 1, pos: 1
type: B, layer: 1, pos: 24
type: A, layer: 1, pos: 14
type: B, layer: 1, pos: 14
type: B, layer: 1, pos: 40
type: A, layer: 1, pos: 40

Time for candidate selection: 0.25 seconds

### Candidate
type: B, layer: 1, pos: 8

### Candidate
type: A, layer: 1, pos: 8

### Candidate
type: A, layer: 1, pos: 21

## Relational analysis of NS_A1_B1_A2_B2_A2_B1_A1

### Relational analysis result of NS_A1_B1_A2_B2_A2_B1_A1
Status: Status.UNKNOWN
Output dim: 0, lower bound: -913.4799742, upper bound: 913.8292779
time: 0.49 seconds

## Relational analysis of NS_A1_B1_A2_B2_A2_B1_A2

### Relational analysis result of NS_A1_B1_A2_B2_A2_B1_A2
Status: Status.UNKNOWN
Output dim: 0, lower bound: -913.5192845, upper bound: 914.1487040
time: 0.52 seconds

## BFS NS instance: NS_A1_B1_A2_B2_A2_B2

### Backsubstitution after applying NS history:
0: -487.9168091, 314.3346558, -487.9168091, 314.3346558, -802.2514648, 802.2514648
1: -36.8756714, 28.1965370, -36.8756714, 28.1965370, -65.0722046, 65.0722046
2: -30.9391079, 41.8894653, -30.9391079, 41.8894653, -72.8285751, 72.8285751
3: -34.8926201, 67.1765060, -34.8926201, 67.1765060, -102.0691223, 102.0691223
4: -26.5249214, 41.9749985, -26.5249214, 41.9749985, -68.4999237, 68.4999237

Time for backsubstitution: 3.01 seconds

### NS candidates at layer 1
type: A, layer: 1, pos: 8
type: B, layer: 1, pos: 8
type: A, layer: 1, pos: 21
type: B, layer: 1, pos: 21
type: A, layer: 1, pos: 38
type: B, layer: 1, pos: 38
type: A, layer: 1, pos: 47
type: B, layer: 1, pos: 47
type: B, layer: 1, pos: 35
type: A, layer: 1, pos: 35
type: B, layer: 1, pos: 15
type: A, layer: 1, pos: 15
type: B, layer: 1, pos: 10
type: A, layer: 1, pos: 10
type: A, layer: 1, pos: 30
type: B, layer: 1, pos: 30
type: A, layer: 1, pos: 3
type: B, layer: 1, pos: 3
type: B, layer: 1, pos: 24
type: A, layer: 1, pos: 24
type: B, layer: 1, pos: 1
type: A, layer: 1, pos: 1
type: B, layer: 1, pos: 14
type: A, layer: 1, pos: 14
type: A, layer: 1, pos: 40
type: B, layer: 1, pos: 40

Time for candidate selection: 0.25 seconds

### Candidate
type: A, layer: 1, pos: 8

### Candidate
type: B, layer: 1, pos: 8

### Candidate
type: A, layer: 1, pos: 21

## Relational analysis of NS_A1_B1_A2_B2_A2_B2_A1

### Relational analysis result of NS_A1_B1_A2_B2_A2_B2_A1
Status: Status.UNKNOWN
Output dim: 0, lower bound: -913.4799742, upper bound: 914.1796411
time: 0.50 seconds

## Relational analysis of NS_A1_B1_A2_B2_A2_B2_A2

### Relational analysis result of NS_A1_B1_A2_B2_A2_B2_A2
Status: Status.UNKNOWN
Output dim: 0, lower bound: -913.5192845, upper bound: 914.1796410
time: 0.51 seconds

## BFS NS instance: NS_A1_B2_B1_A1_B1_A1

### Backsubstitution after applying NS history:
0: -420.7253113, 255.4279938, -411.8660278, 232.6477051, -653.3729248, 667.2939453
1: -30.5903091, 23.9250946, -28.6539593, 23.1143284, -53.7046280, 52.5790367
2: -26.6393108, 34.0855064, -25.7596779, 31.3545437, -57.9938507, 59.8451843
3: -29.7036209, 56.6203690, -28.4703903, 54.1781807, -83.8818054, 85.0907516
4: -22.9709263, 34.2622719, -22.2013893, 31.6630421, -54.6339569, 56.4636497

Time for backsubstitution: 3.01 seconds

### NS candidates at layer 1
type: A, layer: 1, pos: 8
type: B, layer: 1, pos: 8
type: B, layer: 1, pos: 10
type: A, layer: 1, pos: 38
type: B, layer: 1, pos: 5
type: A, layer: 1, pos: 5
type: B, layer: 1, pos: 1
type: A, layer: 1, pos: 35
type: B, layer: 1, pos: 35
type: A, layer: 1, pos: 1
type: A, layer: 1, pos: 15
type: B, layer: 1, pos: 38
type: A, layer: 1, pos: 30
type: B, layer: 1, pos: 15
type: A, layer: 1, pos: 3
type: A, layer: 1, pos: 24
type: B, layer: 1, pos: 21
type: A, layer: 1, pos: 21
type: B, layer: 1, pos: 3
type: A, layer: 1, pos: 47
type: B, layer: 1, pos: 24
type: B, layer: 1, pos: 47
type: A, layer: 1, pos: 14
type: A, layer: 1, pos: 40
type: B, layer: 1, pos: 14
type: B, layer: 1, pos: 40

Time for candidate selection: 0.25 seconds

### Candidate
type: A, layer: 1, pos: 8

### Candidate
type: B, layer: 1, pos: 8

### Candidate
type: B, layer: 1, pos: 10

## Relational analysis of NS_A1_B2_B1_A1_B1_A1_B1

### Relational analysis result of NS_A1_B2_B1_A1_B1_A1_B1
Status: Status.UNKNOWN
Output dim: 0, lower bound: -915.8446039, upper bound: 914.2045524
time: 0.48 seconds

## Relational analysis of NS_A1_B2_B1_A1_B1_A1_B2

### Relational analysis result of NS_A1_B2_B1_A1_B1_A1_B2
Status: Status.UNKNOWN
Output dim: 0, lower bound: -915.8446039, upper bound: 914.2045524
time: 0.47 seconds

## BFS NS instance: NS_A1_B2_B1_A1_B1_A2

### Backsubstitution after applying NS history:
0: -456.1683655, 283.4699402, -402.9411011, 222.8238831, -678.9921875, 686.4110107
1: -33.7150154, 25.9370136, -27.6453457, 22.5332718, -56.2482872, 53.5823555
2: -29.3280602, 37.6919594, -25.1795826, 30.0961266, -59.4241829, 62.8715439
3: -32.7249680, 61.7395744, -27.7737122, 52.7536621, -85.4786301, 89.5132904
4: -25.2619076, 37.8055687, -21.7345295, 30.4252205, -55.6871262, 59.5401001

Time for backsubstitution: 3.02 seconds

### NS candidates at layer 1
type: A, layer: 1, pos: 8
type: B, layer: 1, pos: 8
type: A, layer: 1, pos: 15
type: A, layer: 1, pos: 5
type: B, layer: 1, pos: 1
type: B, layer: 1, pos: 5
type: A, layer: 1, pos: 38
type: B, layer: 1, pos: 35
type: A, layer: 1, pos: 35
type: B, layer: 1, pos: 38
type: A, layer: 1, pos: 1
type: A, layer: 1, pos: 30
type: B, layer: 1, pos: 10
type: B, layer: 1, pos: 15
type: A, layer: 1, pos: 24
type: A, layer: 1, pos: 21
type: B, layer: 1, pos: 21
type: A, layer: 1, pos: 3
type: B, layer: 1, pos: 3
type: A, layer: 1, pos: 47
type: B, layer: 1, pos: 24
type: B, layer: 1, pos: 47
type: A, layer: 1, pos: 14
type: A, layer: 1, pos: 40
type: B, layer: 1, pos: 14
type: B, layer: 1, pos: 40

Time for candidate selection: 0.25 seconds

### Candidate
type: A, layer: 1, pos: 8

### Candidate
type: B, layer: 1, pos: 8

### Candidate
type: A, layer: 1, pos: 15

## Relational analysis of NS_A1_B2_B1_A1_B1_A2_A1
Optimization infeasible because this subproblem isn't reachable.

### Candidate
type: A, layer: 1, pos: 5

### Candidate
type: B, layer: 1, pos: 1

## Relational analysis of NS_A1_B2_B1_A1_B1_A2_B1
Optimization infeasible because this subproblem isn't reachable.

### Candidate
type: B, layer: 1, pos: 5

### Candidate
type: A, layer: 1, pos: 38

### Candidate
type: B, layer: 1, pos: 35

## Relational analysis of NS_A1_B2_B1_A1_B1_A2_B1
Optimization infeasible because this subproblem isn't reachable.

### Candidate
type: A, layer: 1, pos: 35

### Candidate
type: B, layer: 1, pos: 38

### Candidate
type: A, layer: 1, pos: 1

### Candidate
type: A, layer: 1, pos: 30

## Relational analysis of NS_A1_B2_B1_A1_B1_A2_A1
Optimization infeasible because this subproblem isn't reachable.

### Candidate
type: B, layer: 1, pos: 10

## Relational analysis of NS_A1_B2_B1_A1_B1_A2_B1

### Relational analysis result of NS_A1_B2_B1_A1_B1_A2_B1
Status: Status.UNKNOWN
Output dim: 0, lower bound: -915.8598194, upper bound: 914.2533971
time: 0.46 seconds

## Relational analysis of NS_A1_B2_B1_A1_B1_A2_B2

### Relational analysis result of NS_A1_B2_B1_A1_B1_A2_B2
Status: Status.UNKNOWN
Output dim: 0, lower bound: -915.8598194, upper bound: 914.2533971
time: 0.47 seconds

## BFS NS instance: NS_A1_B2_B1_A1_B2_A1

### Backsubstitution after applying NS history:
0: -435.6751099, 269.6290588, -452.7079468, 280.5736389, -716.2487183, 722.3370361
1: -32.0471001, 24.9120903, -33.3736992, 25.8884563, -57.9355545, 58.2857857
2: -27.7036819, 35.8731728, -28.5619106, 37.4851913, -65.1888657, 64.4350739
3: -30.9559784, 58.9597969, -31.8986073, 61.1683121, -92.1242905, 90.8584061
4: -23.8459702, 36.0397148, -24.5331535, 37.6732979, -61.5192604, 60.5728683

Time for backsubstitution: 3.02 seconds

### NS candidates at layer 1
type: B, layer: 1, pos: 8
type: A, layer: 1, pos: 8
type: B, layer: 1, pos: 10
type: B, layer: 1, pos: 38
type: A, layer: 1, pos: 38
type: A, layer: 1, pos: 35
type: B, layer: 1, pos: 21
type: A, layer: 1, pos: 21
type: B, layer: 1, pos: 35
type: A, layer: 1, pos: 10
type: A, layer: 1, pos: 24
type: B, layer: 1, pos: 5
type: B, layer: 1, pos: 15
type: A, layer: 1, pos: 15
type: B, layer: 1, pos: 24
type: A, layer: 1, pos: 5
type: A, layer: 1, pos: 1
type: B, layer: 1, pos: 1
type: B, layer: 1, pos: 3
type: A, layer: 1, pos: 3
type: B, layer: 1, pos: 47
type: A, layer: 1, pos: 30
type: A, layer: 1, pos: 14
type: B, layer: 1, pos: 14
type: A, layer: 1, pos: 40
type: B, layer: 1, pos: 40

Time for candidate selection: 0.25 seconds

### Candidate
type: B, layer: 1, pos: 8

### Candidate
type: A, layer: 1, pos: 8

### Candidate
type: B, layer: 1, pos: 10

## Relational analysis of NS_A1_B2_B1_A1_B2_A1_B1
Optimization infeasible because this subproblem isn't reachable.

### Candidate
type: B, layer: 1, pos: 38

### Candidate
type: A, layer: 1, pos: 38

### Candidate
type: A, layer: 1, pos: 35

## Relational analysis of NS_A1_B2_B1_A1_B2_A1_A1
Optimization infeasible because this subproblem isn't reachable.

### Candidate
type: B, layer: 1, pos: 21

### Candidate
type: A, layer: 1, pos: 21

### Candidate
type: B, layer: 1, pos: 35

### Candidate
type: A, layer: 1, pos: 10

### Candidate
type: A, layer: 1, pos: 24

## Relational analysis of NS_A1_B2_B1_A1_B2_A1_A1
Optimization infeasible because this subproblem isn't reachable.

### Candidate
type: B, layer: 1, pos: 5

## Relational analysis of NS_A1_B2_B1_A1_B2_A1_B1
Optimization infeasible because this subproblem isn't reachable.

### Candidate
type: B, layer: 1, pos: 15

## Relational analysis of NS_A1_B2_B1_A1_B2_A1_B1
Optimization infeasible because this subproblem isn't reachable.

### Candidate
type: A, layer: 1, pos: 15

### Candidate
type: B, layer: 1, pos: 24

### Candidate
type: A, layer: 1, pos: 5

### Candidate
type: A, layer: 1, pos: 1

## Relational analysis of NS_A1_B2_B1_A1_B2_A1_A1
Optimization infeasible because this subproblem isn't reachable.

### Candidate
type: B, layer: 1, pos: 1

### Candidate
type: B, layer: 1, pos: 3

## Relational analysis of NS_A1_B2_B1_A1_B2_A1_B1

### Relational analysis result of NS_A1_B2_B1_A1_B2_A1_B1
Status: Status.VERIFIED
Output dim: 0, lower bound: -912.6991240, upper bound: 912.1303756
time: 0.48 seconds

## Relational analysis of NS_A1_B2_B1_A1_B2_A1_B2
Optimization infeasible because this subproblem isn't reachable.

### Candidate
type: A, layer: 1, pos: 3

### Candidate
type: B, layer: 1, pos: 47

## Relational analysis of NS_A1_B2_B1_A1_B2_A1_B1

### Relational analysis result of NS_A1_B2_B1_A1_B2_A1_B1
Status: Status.VERIFIED
Output dim: 0, lower bound: -912.7943344, upper bound: 912.7683715
time: 0.49 seconds

## Relational analysis of NS_A1_B2_B1_A1_B2_A1_B2

### Relational analysis result of NS_A1_B2_B1_A1_B2_A1_B2
Status: Status.VERIFIED
Output dim: 0, lower bound: -912.7943344, upper bound: 912.7683715
time: 0.46 seconds

## BFS NS instance: NS_A1_B2_B2_A1_B1_A1

### Backsubstitution after applying NS history:
0: -410.9015503, 247.1380310, -441.0475159, 260.9769287, -671.8784790, 688.1854858
1: -29.7045422, 23.2836685, -31.5716000, 25.0454521, -54.7499924, 54.8552666
2: -26.0190964, 32.9667931, -27.5723667, 35.1714249, -61.1905212, 60.5391617
3: -28.9830647, 55.1883430, -30.6380043, 59.0280304, -88.0110855, 85.8263474
4: -22.4508877, 33.1186676, -23.6614170, 35.4425697, -57.8934517, 56.7800827

Time for backsubstitution: 3.04 seconds

### NS candidates at layer 1
type: B, layer: 1, pos: 10
type: B, layer: 1, pos: 38
type: A, layer: 1, pos: 38
type: A, layer: 1, pos: 47
type: A, layer: 1, pos: 21
type: B, layer: 1, pos: 21
type: A, layer: 1, pos: 8
type: A, layer: 1, pos: 10
type: B, layer: 1, pos: 47
type: B, layer: 1, pos: 8
type: B, layer: 1, pos: 5
type: A, layer: 1, pos: 5
type: B, layer: 1, pos: 15
type: A, layer: 1, pos: 15
type: B, layer: 1, pos: 35
type: B, layer: 1, pos: 24
type: A, layer: 1, pos: 3
type: B, layer: 1, pos: 3
type: A, layer: 1, pos: 30
type: A, layer: 1, pos: 24
type: B, layer: 1, pos: 1
type: A, layer: 1, pos: 1
type: A, layer: 1, pos: 14
type: B, layer: 1, pos: 14
type: A, layer: 1, pos: 40
type: B, layer: 1, pos: 40

Time for candidate selection: 0.25 seconds

### Candidate
type: B, layer: 1, pos: 10

### Candidate
type: B, layer: 1, pos: 38

### Candidate
type: A, layer: 1, pos: 38

### Candidate
type: A, layer: 1, pos: 47

## Relational analysis of NS_A1_B2_B2_A1_B1_A1_A1
Optimization infeasible because this subproblem isn't reachable.

### Candidate
type: A, layer: 1, pos: 21

### Candidate
type: B, layer: 1, pos: 21

### Candidate
type: A, layer: 1, pos: 8

### Candidate
type: A, layer: 1, pos: 10

### Candidate
type: B, layer: 1, pos: 47

### Candidate
type: B, layer: 1, pos: 8

### Candidate
type: B, layer: 1, pos: 5

### Candidate
type: A, layer: 1, pos: 5

### Candidate
type: B, layer: 1, pos: 15

## Relational analysis of NS_A1_B2_B2_A1_B1_A1_B1

### Relational analysis result of NS_A1_B2_B2_A1_B1_A1_B1
Status: Status.UNKNOWN
Output dim: 0, lower bound: -915.5469533, upper bound: 913.4679795
time: 0.47 seconds

## Relational analysis of NS_A1_B2_B2_A1_B1_A1_B2
Optimization infeasible because this subproblem isn't reachable.

### Candidate
type: A, layer: 1, pos: 15

### Candidate
type: B, layer: 1, pos: 35

## Relational analysis of NS_A1_B2_B2_A1_B1_A1_B1

### Relational analysis result of NS_A1_B2_B2_A1_B1_A1_B1
Status: Status.UNKNOWN
Output dim: 0, lower bound: -914.3656260, upper bound: 913.2249840
time: 0.53 seconds

## Relational analysis of NS_A1_B2_B2_A1_B1_A1_B2

### Relational analysis result of NS_A1_B2_B2_A1_B1_A1_B2
Status: Status.UNKNOWN
Output dim: 0, lower bound: -914.3656260, upper bound: 913.4911499
time: 0.49 seconds

## BFS NS instance: NS_A1_B2_B2_A1_B1_A2

### Backsubstitution after applying NS history:
0: -425.3090820, 260.1288147, -444.6026306, 264.5179443, -689.8270264, 704.7314453
1: -31.0589294, 24.2380390, -31.9292221, 25.2847805, -56.3437042, 56.1672554
2: -26.9664078, 34.6580811, -27.7712402, 35.6411972, -62.6076012, 62.4293213
3: -30.0966187, 57.3346519, -30.8796558, 59.5743904, -89.6710052, 88.2142944
4: -23.2446899, 34.8299408, -23.8213329, 35.9191093, -59.1637993, 58.6512756

Time for backsubstitution: 3.03 seconds

### NS candidates at layer 1
type: B, layer: 1, pos: 10
type: A, layer: 1, pos: 21
type: B, layer: 1, pos: 21
type: A, layer: 1, pos: 38
type: B, layer: 1, pos: 38
type: A, layer: 1, pos: 10
type: A, layer: 1, pos: 8
type: B, layer: 1, pos: 8
type: B, layer: 1, pos: 5
type: A, layer: 1, pos: 5
type: B, layer: 1, pos: 35
type: A, layer: 1, pos: 47
type: A, layer: 1, pos: 1
type: B, layer: 1, pos: 1
type: B, layer: 1, pos: 47
type: A, layer: 1, pos: 15
type: A, layer: 1, pos: 3
type: B, layer: 1, pos: 15
type: B, layer: 1, pos: 3
type: A, layer: 1, pos: 30
type: B, layer: 1, pos: 24
type: A, layer: 1, pos: 24
type: A, layer: 1, pos: 14
type: B, layer: 1, pos: 14
type: A, layer: 1, pos: 40
type: B, layer: 1, pos: 40

Time for candidate selection: 0.26 seconds

### Candidate
type: B, layer: 1, pos: 10

### Candidate
type: A, layer: 1, pos: 21

### Candidate
type: B, layer: 1, pos: 21

### Candidate
type: A, layer: 1, pos: 38

### Candidate
type: B, layer: 1, pos: 38

### Candidate
type: A, layer: 1, pos: 10

### Candidate
type: A, layer: 1, pos: 8

### Candidate
type: B, layer: 1, pos: 8

### Candidate
type: B, layer: 1, pos: 5

### Candidate
type: A, layer: 1, pos: 5

### Candidate
type: B, layer: 1, pos: 35

## Relational analysis of NS_A1_B2_B2_A1_B1_A2_B1

### Relational analysis result of NS_A1_B2_B2_A1_B1_A2_B1
Status: Status.UNKNOWN
Output dim: 0, lower bound: -914.3656260, upper bound: 913.2249840
time: 0.51 seconds

## Relational analysis of NS_A1_B2_B2_A1_B1_A2_B2

### Relational analysis result of NS_A1_B2_B2_A1_B1_A2_B2
Status: Status.UNKNOWN
Output dim: 0, lower bound: -914.3656260, upper bound: 914.6891405
time: 0.50 seconds

## BFS NS instance: NS_A1_B2_B2_A1_B2_B1

### Backsubstitution after applying NS history:
0: -451.6561279, 282.8571472, -460.9415283, 289.6008911, -741.2570190, 743.7986450
1: -33.4562607, 25.9021244, -34.3236656, 26.3693466, -59.8256073, 60.2257843
2: -28.6227245, 37.5928574, -29.0562210, 38.7226753, -67.3453827, 66.6490784
3: -32.0312691, 61.3333549, -32.5205078, 62.5040169, -94.5352554, 93.8538589
4: -24.6041679, 37.7395096, -24.9555225, 38.8636246, -63.4677925, 62.6950302

Time for backsubstitution: 3.04 seconds

### NS candidates at layer 1
type: A, layer: 1, pos: 8
type: B, layer: 1, pos: 8
type: A, layer: 1, pos: 38
type: B, layer: 1, pos: 38
type: A, layer: 1, pos: 21
type: B, layer: 1, pos: 21
type: B, layer: 1, pos: 5
type: A, layer: 1, pos: 5
type: A, layer: 1, pos: 35
type: A, layer: 1, pos: 3
type: B, layer: 1, pos: 3
type: A, layer: 1, pos: 15
type: A, layer: 1, pos: 30
type: A, layer: 1, pos: 24
type: A, layer: 1, pos: 47
type: B, layer: 1, pos: 10
type: B, layer: 1, pos: 24
type: A, layer: 1, pos: 10
type: B, layer: 1, pos: 1
type: B, layer: 1, pos: 47
type: A, layer: 1, pos: 1
type: B, layer: 1, pos: 15
type: A, layer: 1, pos: 14
type: B, layer: 1, pos: 14
type: A, layer: 1, pos: 40
type: B, layer: 1, pos: 40

Time for candidate selection: 0.25 seconds

### Candidate
type: A, layer: 1, pos: 8

### Candidate
type: B, layer: 1, pos: 8

### Candidate
type: A, layer: 1, pos: 38

### Candidate
type: B, layer: 1, pos: 38

### Candidate
type: A, layer: 1, pos: 21

### Candidate
type: B, layer: 1, pos: 21

### Candidate
type: B, layer: 1, pos: 5

## Relational analysis of NS_A1_B2_B2_A1_B2_B1_B1
Optimization infeasible because this subproblem isn't reachable.

### Candidate
type: A, layer: 1, pos: 5

### Candidate
type: A, layer: 1, pos: 35

## Relational analysis of NS_A1_B2_B2_A1_B2_B1_A1

### Relational analysis result of NS_A1_B2_B2_A1_B2_B1_A1
Status: Status.UNKNOWN
Output dim: 0, lower bound: -914.3409048, upper bound: 913.4911499
time: 0.50 seconds

## Relational analysis of NS_A1_B2_B2_A1_B2_B1_A2

### Relational analysis result of NS_A1_B2_B2_A1_B2_B1_A2
Status: Status.UNKNOWN
Output dim: 0, lower bound: -914.3409043, upper bound: 914.1697744
time: 0.50 seconds

## BFS NS instance: NS_A1_B2_B2_A1_B2_B2

### Backsubstitution after applying NS history:
0: -456.7611694, 287.8726196, -469.4467468, 296.2907104, -753.0518799, 757.3192139
1: -33.9791489, 26.2344818, -35.0258026, 26.9110928, -60.8902435, 61.2602768
2: -28.9651814, 38.2430115, -29.5866070, 39.5891266, -68.5543060, 67.8296051
3: -32.4367981, 62.1259613, -33.1275291, 63.7577286, -96.1945190, 95.2534866
4: -24.8890457, 38.3879318, -25.3947392, 39.7422485, -64.6312790, 63.7826691

Time for backsubstitution: 3.05 seconds

### NS candidates at layer 1
type: A, layer: 1, pos: 8
type: B, layer: 1, pos: 8
type: B, layer: 1, pos: 5
type: A, layer: 1, pos: 5
type: A, layer: 1, pos: 21
type: B, layer: 1, pos: 21
type: A, layer: 1, pos: 38
type: B, layer: 1, pos: 38
type: A, layer: 1, pos: 47
type: B, layer: 1, pos: 47
type: A, layer: 1, pos: 15
type: B, layer: 1, pos: 10
type: A, layer: 1, pos: 10
type: A, layer: 1, pos: 35
type: B, layer: 1, pos: 3
type: A, layer: 1, pos: 3
type: B, layer: 1, pos: 15
type: A, layer: 1, pos: 30
type: B, layer: 1, pos: 24
type: B, layer: 1, pos: 1
type: A, layer: 1, pos: 1
type: A, layer: 1, pos: 24
type: A, layer: 1, pos: 14
type: B, layer: 1, pos: 14
type: A, layer: 1, pos: 40
type: B, layer: 1, pos: 40

Time for candidate selection: 0.25 seconds

### Candidate
type: A, layer: 1, pos: 8

### Candidate
type: B, layer: 1, pos: 8

### Candidate
type: B, layer: 1, pos: 5

## Relational analysis of NS_A1_B2_B2_A1_B2_B2_B1
Optimization infeasible because this subproblem isn't reachable.

### Candidate
type: A, layer: 1, pos: 5

### Candidate
type: A, layer: 1, pos: 21

### Candidate
type: B, layer: 1, pos: 21

### Candidate
type: A, layer: 1, pos: 38

### Candidate
type: B, layer: 1, pos: 38

### Candidate
type: A, layer: 1, pos: 47

## Relational analysis of NS_A1_B2_B2_A1_B2_B2_A1
Optimization infeasible because this subproblem isn't reachable.

### Candidate
type: B, layer: 1, pos: 47

### Candidate
type: A, layer: 1, pos: 15

## Relational analysis of NS_A1_B2_B2_A1_B2_B2_A1

### Relational analysis result of NS_A1_B2_B2_A1_B2_B2_A1
Status: Status.UNKNOWN
Output dim: 0, lower bound: -914.6692551, upper bound: 914.8899845
time: 0.48 seconds

## Relational analysis of NS_A1_B2_B2_A1_B2_B2_A2

### Relational analysis result of NS_A1_B2_B2_A1_B2_B2_A2
Status: Status.UNKNOWN
Output dim: 0, lower bound: -914.6692551, upper bound: 914.7524119
time: 0.48 seconds

## BFS NS instance: NS_A1_B2_B2_A2_B1_A1

### Backsubstitution after applying NS history:
0: -482.4231567, 309.3693542, -489.5349121, 314.8911743, -797.3143311, 798.9041748
1: -36.3289642, 27.8114147, -37.0346642, 28.2443161, -64.5732803, 64.8460770
2: -30.5974331, 41.2250862, -31.0930405, 42.1184120, -72.7158356, 72.3181152
3: -34.4228249, 66.2290421, -34.8812561, 67.3007965, -101.7236099, 101.1102982
4: -26.2674236, 41.3273392, -26.6835747, 42.2701988, -68.5376053, 68.0109100

Time for backsubstitution: 3.05 seconds

### NS candidates at layer 1
type: B, layer: 1, pos: 8
type: A, layer: 1, pos: 8
type: B, layer: 1, pos: 5
type: A, layer: 1, pos: 5
type: B, layer: 1, pos: 10
type: A, layer: 1, pos: 21
type: B, layer: 1, pos: 21
type: B, layer: 1, pos: 38
type: A, layer: 1, pos: 38
type: B, layer: 1, pos: 35
type: A, layer: 1, pos: 35
type: B, layer: 1, pos: 3
type: A, layer: 1, pos: 15
type: A, layer: 1, pos: 3
type: B, layer: 1, pos: 30
type: A, layer: 1, pos: 30
type: B, layer: 1, pos: 15
type: A, layer: 1, pos: 47
type: A, layer: 1, pos: 24
type: B, layer: 1, pos: 1
type: A, layer: 1, pos: 1
type: B, layer: 1, pos: 24
type: A, layer: 1, pos: 14
type: B, layer: 1, pos: 14
type: B, layer: 1, pos: 40
type: A, layer: 1, pos: 40

Time for candidate selection: 0.26 seconds

### Candidate
type: B, layer: 1, pos: 8

### Candidate
type: A, layer: 1, pos: 8

### Candidate
type: B, layer: 1, pos: 5

### Candidate
type: A, layer: 1, pos: 5

### Candidate
type: B, layer: 1, pos: 10

## Relational analysis of NS_A1_B2_B2_A2_B1_A1_B1

### Relational analysis result of NS_A1_B2_B2_A2_B1_A1_B1
Status: Status.UNKNOWN
Output dim: 0, lower bound: -914.4782008, upper bound: 912.5881746
time: 0.55 seconds

## Relational analysis of NS_A1_B2_B2_A2_B1_A1_B2

### Relational analysis result of NS_A1_B2_B2_A2_B1_A1_B2
Status: Status.UNKNOWN
Output dim: 0, lower bound: -914.4782008, upper bound: 912.5881746
time: 0.55 seconds

## BFS NS instance: NS_A1_B2_B2_A2_B1_A2

### Backsubstitution after applying NS history:
0: -528.9248047, 347.8817139, -476.9782104, 302.7768555, -831.7015991, 824.8599243
1: -40.5204468, 30.6128063, -35.7544556, 27.4163132, -67.9367447, 66.3672638
2: -33.8557434, 46.2785683, -30.2382431, 40.5267372, -74.3824768, 76.5167999
3: -38.2463913, 73.5360947, -33.8758926, 65.0682602, -103.3146515, 107.4119644
4: -29.0111713, 46.3163719, -25.9578133, 40.6919212, -69.7030945, 72.2741776

Time for backsubstitution: 3.05 seconds

### NS candidates at layer 1
type: B, layer: 1, pos: 8
type: A, layer: 1, pos: 8
type: B, layer: 1, pos: 5
type: A, layer: 1, pos: 5
type: B, layer: 1, pos: 21
type: A, layer: 1, pos: 21
type: A, layer: 1, pos: 35
type: A, layer: 1, pos: 38
type: B, layer: 1, pos: 38
type: B, layer: 1, pos: 10
type: B, layer: 1, pos: 35
type: A, layer: 1, pos: 15
type: A, layer: 1, pos: 30
type: B, layer: 1, pos: 30
type: B, layer: 1, pos: 3
type: B, layer: 1, pos: 15
type: A, layer: 1, pos: 3
type: B, layer: 1, pos: 1
type: A, layer: 1, pos: 47
type: A, layer: 1, pos: 1
type: A, layer: 1, pos: 24
type: A, layer: 1, pos: 14
type: B, layer: 1, pos: 24
type: B, layer: 1, pos: 14
type: A, layer: 1, pos: 40
type: B, layer: 1, pos: 40

Time for candidate selection: 0.25 seconds

### Candidate
type: B, layer: 1, pos: 8

### Candidate
type: A, layer: 1, pos: 8

### Candidate
type: B, layer: 1, pos: 5

### Candidate
type: A, layer: 1, pos: 5

### Candidate
type: B, layer: 1, pos: 21

### Candidate
type: A, layer: 1, pos: 21

### Candidate
type: A, layer: 1, pos: 35

### Candidate
type: A, layer: 1, pos: 38

### Candidate
type: B, layer: 1, pos: 38

### Candidate
type: B, layer: 1, pos: 10

## Relational analysis of NS_A1_B2_B2_A2_B1_A2_B1

### Relational analysis result of NS_A1_B2_B2_A2_B1_A2_B1
Status: Status.UNKNOWN
Output dim: 0, lower bound: -914.4782008, upper bound: 912.5881746
time: 0.54 seconds

## Relational analysis of NS_A1_B2_B2_A2_B1_A2_B2

### Relational analysis result of NS_A1_B2_B2_A2_B1_A2_B2
Status: Status.UNKNOWN
Output dim: 0, lower bound: -914.4782008, upper bound: 912.5881746
time: 0.52 seconds

## BFS NS instance: NS_A2_B1_A1_B1_A1_B1

### Backsubstitution after applying NS history:
0: -411.8660278, 232.6477051, -420.7253113, 255.4279938, -667.2939453, 653.3729248
1: -28.6539593, 23.1143284, -30.5903091, 23.9250946, -52.5790367, 53.7046280
2: -25.7596779, 31.3545437, -26.6393108, 34.0855064, -59.8451843, 57.9938507
3: -28.4703903, 54.1781807, -29.7036209, 56.6203690, -85.0907516, 83.8818054
4: -22.2013893, 31.6630421, -22.9709263, 34.2622719, -56.4636497, 54.6339607

Time for backsubstitution: 3.07 seconds

### NS candidates at layer 1
type: B, layer: 1, pos: 8
type: A, layer: 1, pos: 8
type: A, layer: 1, pos: 10
type: B, layer: 1, pos: 38
type: A, layer: 1, pos: 5
type: B, layer: 1, pos: 5
type: A, layer: 1, pos: 1
type: B, layer: 1, pos: 35
type: A, layer: 1, pos: 35
type: B, layer: 1, pos: 1
type: B, layer: 1, pos: 15
type: A, layer: 1, pos: 38
type: B, layer: 1, pos: 30
type: A, layer: 1, pos: 15
type: B, layer: 1, pos: 3
type: B, layer: 1, pos: 24
type: A, layer: 1, pos: 21
type: B, layer: 1, pos: 21
type: A, layer: 1, pos: 3
type: B, layer: 1, pos: 47
type: A, layer: 1, pos: 24
type: A, layer: 1, pos: 47
type: B, layer: 1, pos: 14
type: B, layer: 1, pos: 40
type: A, layer: 1, pos: 14
type: A, layer: 1, pos: 40

Time for candidate selection: 0.26 seconds

### Candidate
type: B, layer: 1, pos: 8

### Candidate
type: A, layer: 1, pos: 8

### Candidate
type: A, layer: 1, pos: 10

## Relational analysis of NS_A2_B1_A1_B1_A1_B1_A1

### Relational analysis result of NS_A2_B1_A1_B1_A1_B1_A1
Status: Status.UNKNOWN
Output dim: 0, lower bound: -914.2045524, upper bound: 915.8446039
time: 0.47 seconds

## Relational analysis of NS_A2_B1_A1_B1_A1_B1_A2

### Relational analysis result of NS_A2_B1_A1_B1_A1_B1_A2
Status: Status.UNKNOWN
Output dim: 0, lower bound: -914.2045524, upper bound: 915.8446039
time: 0.49 seconds

## NS Result
status: Status.UNKNOWN
execution time: (base) + (ns) = 4.56 + 415.53 = 420.10 seconds
