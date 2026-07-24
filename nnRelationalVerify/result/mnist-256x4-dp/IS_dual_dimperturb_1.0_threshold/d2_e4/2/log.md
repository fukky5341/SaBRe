## Execution arguments:
Dataset: Dataset.MNIST
Network: onnx/mnist-net_256x4.onnx
Relational property: GLOBAL_ROBUSTNESS
LP Analysis: True
Epsilon: 0.03125
Delta epsilon: 0.0078125
execution index: (2, 4, 2)
Time budget: 600 seconds
Split limit: 100
Threshold: 14.544726514199999


## IAR start

### Unstable ReLU Count (Linear/Conv2D Layers)
- layer_idx=0, type=LayerType.Linear, total=256, inp1_unstable=34, inp2_unstable=34, delta_unstable=256
- layer_idx=2, type=LayerType.Linear, total=256, inp1_unstable=71, inp2_unstable=71, delta_unstable=256
- layer_idx=4, type=LayerType.Linear, total=256, inp1_unstable=52, inp2_unstable=52, delta_unstable=256
- layer_idx=6, type=LayerType.Linear, total=256, inp1_unstable=256, inp2_unstable=256, delta_unstable=256
- layer_idx=8, type=LayerType.Linear, total=10, inp1_unstable=10, inp2_unstable=10, delta_unstable=10

### BASE IAR bounds
Layer (inp1_lb, inp1_ub, inp2_lb, inp2_ub, d_lb, d_ub)
0: (-9.7792997, 7.9942503, -9.7792997, 7.9942503, -17.7735500, 17.7735500)
1: (-8.0821228, 7.1762996, -8.0821228, 7.1762996, -15.2584229, 15.2584229)
2: (-10.1421604, 6.3774090, -10.1421604, 6.3774090, -16.5195694, 16.5195675)
3: (-11.8299160, 5.6739855, -11.8299160, 5.6739855, -17.5039024, 17.5039024)
4: (-10.7803135, 8.5861006, -10.7803135, 8.5861006, -19.3664131, 19.3664131)
5: (-9.2919331, 7.2536936, -9.2919331, 7.2536936, -16.5456238, 16.5456238)
6: (-8.7421551, 9.3571310, -8.7421551, 9.3571310, -18.0992813, 18.0992851)
7: (-10.7748833, 6.7838645, -10.7748833, 6.7838645, -17.5587482, 17.5587482)
8: (-11.0645905, 7.8223886, -11.0645905, 7.8223886, -18.8869781, 18.8869781)
9: (-8.9881773, 9.0959854, -8.9881773, 9.0959854, -18.0841618, 18.0841618)

## BASE Relational Analysis

## BASE Result
execution time: IAR + RelationalAnalysis = 1.33 + 8.78 = 10.11 seconds
status: Status.UNKNOWN
relational distance
Output dim: 7, lower bound: -14.5592858, upper bound: 14.5592858

# Indivdual Split (IS) starts

## BFS IS instance: IS

Time for backsubstitution: 0.01 seconds

### IS candidates at layer 1
type: A, layer: 1, pos: 140
type: B, layer: 1, pos: 140
type: A, layer: 1, pos: 181
type: B, layer: 1, pos: 181
type: A, layer: 1, pos: 187
type: B, layer: 1, pos: 187
type: A, layer: 1, pos: 144
type: B, layer: 1, pos: 144
type: A, layer: 1, pos: 196
type: B, layer: 1, pos: 196
type: B, layer: 1, pos: 195
type: A, layer: 1, pos: 195
type: A, layer: 1, pos: 155
type: B, layer: 1, pos: 155
type: A, layer: 1, pos: 232
type: B, layer: 1, pos: 232
type: B, layer: 1, pos: 71
type: A, layer: 1, pos: 71
type: B, layer: 1, pos: 25
type: A, layer: 1, pos: 25
type: A, layer: 1, pos: 132
type: B, layer: 1, pos: 132
type: B, layer: 1, pos: 77
type: A, layer: 1, pos: 77
type: B, layer: 1, pos: 197
type: A, layer: 1, pos: 197
type: A, layer: 1, pos: 185
type: B, layer: 1, pos: 185
type: A, layer: 1, pos: 184
type: B, layer: 1, pos: 184
type: A, layer: 1, pos: 1
type: B, layer: 1, pos: 1
type: A, layer: 1, pos: 89
type: B, layer: 1, pos: 89
type: B, layer: 1, pos: 35
type: A, layer: 1, pos: 35
type: A, layer: 1, pos: 15
type: B, layer: 1, pos: 15
type: A, layer: 1, pos: 179
type: B, layer: 1, pos: 179
type: B, layer: 1, pos: 85
type: A, layer: 1, pos: 85
type: A, layer: 1, pos: 171
type: B, layer: 1, pos: 171
type: A, layer: 1, pos: 128
type: B, layer: 1, pos: 128
type: A, layer: 1, pos: 219
type: B, layer: 1, pos: 219
type: A, layer: 1, pos: 204
type: B, layer: 1, pos: 204
type: A, layer: 1, pos: 146
type: B, layer: 1, pos: 146
type: A, layer: 1, pos: 97
type: B, layer: 1, pos: 97
type: B, layer: 1, pos: 107
type: A, layer: 1, pos: 107
type: A, layer: 1, pos: 138
type: B, layer: 1, pos: 138
type: A, layer: 1, pos: 251
type: B, layer: 1, pos: 251
type: A, layer: 1, pos: 123
type: B, layer: 1, pos: 123
type: A, layer: 1, pos: 104
type: B, layer: 1, pos: 104
type: A, layer: 1, pos: 57
type: B, layer: 1, pos: 57
type: A, layer: 1, pos: 233
type: B, layer: 1, pos: 233

Time for candidate selection: 0.13 seconds

### Candidate
type: A, layer: 1, pos: 140

## Relational analysis of IS_A1

### Relational analysis result of IS_A1
Status: Status.UNKNOWN
Output dim: 7, lower bound: -14.5484615, upper bound: 14.5494866
time: 5.60 seconds

## Relational analysis of IS_A2

### Relational analysis result of IS_A2
Status: Status.UNKNOWN
Output dim: 7, lower bound: -14.5472453, upper bound: 14.5472453
time: 6.26 seconds

## Summary of splitting at layer (split count: 0)
- Time for IS candidates: 12.00 seconds
IS_A1, status: Status.UNKNOWN, split count: 1, time: 12.00
Output dim: 7, lower bound: -14.5484615, upper bound: 14.5494866
IS_A2, status: Status.UNKNOWN, split count: 1, time: 12.00
Output dim: 7, lower bound: -14.5472453, upper bound: 14.5472453

## BFS IS instance: IS_A1

### Backsubstitution after applying IS history:
0: -9.6612396, 7.8967943, -9.7792997, 7.9942503, -17.6554909, 17.6760941
1: -7.9856696, 7.0928993, -8.0821228, 7.1762996, -15.1619692, 15.1750221
2: -10.0136776, 6.2966886, -10.1421604, 6.3774090, -16.3910847, 16.4388466
3: -11.6924953, 5.6022739, -11.8299160, 5.6739855, -17.3664799, 17.4321899
4: -10.6546726, 8.4849911, -10.7803135, 8.5861006, -19.2407722, 19.2653027
5: -9.1768894, 7.1605692, -9.2919331, 7.2536936, -16.4305801, 16.4525032
6: -8.6340132, 9.2505045, -8.7421551, 9.3571310, -17.9911442, 17.9926605
7: -10.6555367, 6.6775041, -10.7748833, 6.7838645, -17.4394016, 17.4523869
8: -10.9247494, 7.7314601, -11.0645905, 7.8223886, -18.7471390, 18.7960510
9: -8.8780117, 8.9821739, -8.9881773, 9.0959854, -17.9739933, 17.9703484

### Unstable ReLU Count (Linear/Conv2D Layers)
- layer_idx=0, type=LayerType.Linear, total=256, inp1_unstable=33, inp2_unstable=34, delta_unstable=256
- layer_idx=2, type=LayerType.Linear, total=256, inp1_unstable=70, inp2_unstable=71, delta_unstable=256
- layer_idx=4, type=LayerType.Linear, total=256, inp1_unstable=50, inp2_unstable=52, delta_unstable=256
- layer_idx=6, type=LayerType.Linear, total=256, inp1_unstable=255, inp2_unstable=256, delta_unstable=256
- layer_idx=8, type=LayerType.Linear, total=10, inp1_unstable=10, inp2_unstable=10, delta_unstable=10

Time for backsubstitution: 1.22 seconds

### IS candidates at layer 1
type: B, layer: 1, pos: 181
type: A, layer: 1, pos: 181
type: B, layer: 1, pos: 187
type: A, layer: 1, pos: 187
type: B, layer: 1, pos: 144
type: A, layer: 1, pos: 144
type: B, layer: 1, pos: 196
type: A, layer: 1, pos: 196
type: B, layer: 1, pos: 25
type: B, layer: 1, pos: 195
type: A, layer: 1, pos: 232
type: A, layer: 1, pos: 195
type: B, layer: 1, pos: 232
type: B, layer: 1, pos: 71
type: A, layer: 1, pos: 155
type: A, layer: 1, pos: 25
type: B, layer: 1, pos: 132
type: A, layer: 1, pos: 71
type: B, layer: 1, pos: 77
type: B, layer: 1, pos: 155
type: A, layer: 1, pos: 132
type: B, layer: 1, pos: 184
type: A, layer: 1, pos: 77
type: B, layer: 1, pos: 197
type: A, layer: 1, pos: 197
type: A, layer: 1, pos: 185
type: B, layer: 1, pos: 1
type: B, layer: 1, pos: 185
type: B, layer: 1, pos: 89
type: A, layer: 1, pos: 184
type: A, layer: 1, pos: 1
type: A, layer: 1, pos: 89
type: A, layer: 1, pos: 35
type: B, layer: 1, pos: 35
type: B, layer: 1, pos: 179
type: A, layer: 1, pos: 15
type: B, layer: 1, pos: 140
type: A, layer: 1, pos: 128
type: B, layer: 1, pos: 219
type: A, layer: 1, pos: 85
type: B, layer: 1, pos: 15
type: A, layer: 1, pos: 204
type: B, layer: 1, pos: 85
type: A, layer: 1, pos: 179
type: B, layer: 1, pos: 128
type: A, layer: 1, pos: 171
type: B, layer: 1, pos: 171
type: B, layer: 1, pos: 204
type: B, layer: 1, pos: 107
type: A, layer: 1, pos: 107
type: A, layer: 1, pos: 219
type: A, layer: 1, pos: 146
type: A, layer: 1, pos: 97
type: B, layer: 1, pos: 138
type: B, layer: 1, pos: 146
type: B, layer: 1, pos: 97
type: A, layer: 1, pos: 138
type: B, layer: 1, pos: 251
type: A, layer: 1, pos: 251
type: B, layer: 1, pos: 123
type: A, layer: 1, pos: 123
type: B, layer: 1, pos: 104
type: A, layer: 1, pos: 233
type: A, layer: 1, pos: 104
type: B, layer: 1, pos: 233
type: B, layer: 1, pos: 57
type: A, layer: 1, pos: 57

Time for candidate selection: 0.12 seconds

### Candidate
type: B, layer: 1, pos: 181

## Relational analysis of IS_A1_B1
Optimization infeasible because this subproblem isn't reachable.

### Candidate
type: A, layer: 1, pos: 181

## Relational analysis of IS_A1_A1
Optimization infeasible because this subproblem isn't reachable.

### Candidate
type: B, layer: 1, pos: 187

## Relational analysis of IS_A1_B1

### Relational analysis result of IS_A1_B1
Status: Status.UNKNOWN
Output dim: 7, lower bound: -14.5432908, upper bound: 14.5452286
time: 7.11 seconds

## Relational analysis of IS_A1_B2

### Relational analysis result of IS_A1_B2
Status: Status.UNKNOWN
Output dim: 7, lower bound: -14.5469192, upper bound: 14.5480980
time: 7.01 seconds

## BFS IS instance: IS_A2

### Backsubstitution after applying IS history:
0: -10.6505451, 8.6375523, -9.4677448, 7.7387609, -18.3893051, 18.1052933
1: -8.7718887, 7.7356014, -7.8263350, 6.9552321, -15.7271204, 15.5619364
2: -10.9133415, 6.7153158, -9.8002796, 6.1623058, -17.0756416, 16.5155945
3: -12.9688931, 6.0202065, -11.4663076, 5.4834790, -18.4523716, 17.4865150
4: -11.7545815, 9.2443399, -10.4488115, 8.3168840, -20.0714645, 19.6931515
5: -10.0350838, 7.7576222, -8.9882717, 7.0084348, -17.0435143, 16.7458935
6: -9.4005165, 10.1860647, -8.4536304, 9.0757656, -18.4762821, 18.6396942
7: -11.7271471, 6.9218836, -10.4573307, 6.4959698, -18.2231178, 17.3792114
8: -11.9139671, 8.4321308, -10.6918297, 7.5813270, -19.4952946, 19.1239605
9: -9.6156349, 9.7219467, -8.6959972, 8.7923346, -18.4079704, 18.4179440

### Unstable ReLU Count (Linear/Conv2D Layers)
- layer_idx=0, type=LayerType.Linear, total=256, inp1_unstable=33, inp2_unstable=34, delta_unstable=256
- layer_idx=2, type=LayerType.Linear, total=256, inp1_unstable=71, inp2_unstable=70, delta_unstable=256
- layer_idx=4, type=LayerType.Linear, total=256, inp1_unstable=30, inp2_unstable=47, delta_unstable=256
- layer_idx=6, type=LayerType.Linear, total=256, inp1_unstable=250, inp2_unstable=255, delta_unstable=256
- layer_idx=8, type=LayerType.Linear, total=10, inp1_unstable=10, inp2_unstable=10, delta_unstable=10

Time for backsubstitution: 1.22 seconds

### IS candidates at layer 1
type: B, layer: 1, pos: 187
type: B, layer: 1, pos: 181
type: A, layer: 1, pos: 181
type: B, layer: 1, pos: 196
type: B, layer: 1, pos: 144
type: B, layer: 1, pos: 195
type: A, layer: 1, pos: 232
type: A, layer: 1, pos: 155
type: A, layer: 1, pos: 187
type: A, layer: 1, pos: 195
type: B, layer: 1, pos: 25
type: A, layer: 1, pos: 144
type: B, layer: 1, pos: 71
type: B, layer: 1, pos: 132
type: A, layer: 1, pos: 196
type: A, layer: 1, pos: 71
type: B, layer: 1, pos: 232
type: B, layer: 1, pos: 89
type: B, layer: 1, pos: 184
type: B, layer: 1, pos: 1
type: B, layer: 1, pos: 185
type: B, layer: 1, pos: 197
type: B, layer: 1, pos: 155
type: A, layer: 1, pos: 185
type: B, layer: 1, pos: 77
type: A, layer: 1, pos: 85
type: A, layer: 1, pos: 197
type: B, layer: 1, pos: 219
type: A, layer: 1, pos: 204
type: B, layer: 1, pos: 85
type: A, layer: 1, pos: 25
type: A, layer: 1, pos: 1
type: A, layer: 1, pos: 35
type: B, layer: 1, pos: 138
type: A, layer: 1, pos: 132
type: A, layer: 1, pos: 15
type: A, layer: 1, pos: 128
type: A, layer: 1, pos: 89
type: A, layer: 1, pos: 77
type: A, layer: 1, pos: 146
type: B, layer: 1, pos: 179
type: A, layer: 1, pos: 138
type: B, layer: 1, pos: 140
type: A, layer: 1, pos: 171
type: B, layer: 1, pos: 15
type: B, layer: 1, pos: 204
type: B, layer: 1, pos: 35
type: B, layer: 1, pos: 128
type: A, layer: 1, pos: 184
type: A, layer: 1, pos: 219
type: B, layer: 1, pos: 171
type: A, layer: 1, pos: 179
type: B, layer: 1, pos: 251
type: B, layer: 1, pos: 123
type: A, layer: 1, pos: 97
type: A, layer: 1, pos: 123
type: B, layer: 1, pos: 146
type: B, layer: 1, pos: 104
type: A, layer: 1, pos: 233
type: A, layer: 1, pos: 107
type: B, layer: 1, pos: 233
type: A, layer: 1, pos: 251
type: B, layer: 1, pos: 97
type: A, layer: 1, pos: 104
type: B, layer: 1, pos: 57
type: B, layer: 1, pos: 107
type: A, layer: 1, pos: 57

Time for candidate selection: 0.13 seconds

### Candidate
type: B, layer: 1, pos: 187

## Relational analysis of IS_A2_B1

### Relational analysis result of IS_A2_B1
Status: Status.VERIFIED
Output dim: 7, lower bound: -14.5421468, upper bound: 14.5427956
time: 3.99 seconds

## Relational analysis of IS_A2_B2

### Relational analysis result of IS_A2_B2
Status: Status.UNKNOWN
Output dim: 7, lower bound: -14.5456956, upper bound: 14.5456956
time: 3.87 seconds

## Summary of splitting at layer (split count: 1)
- Time for IS candidates: 9.22 seconds
IS_A1_B1, status: Status.UNKNOWN, split count: 2, time: 9.22
Output dim: 7, lower bound: -14.5432908, upper bound: 14.5452286
IS_A1_B2, status: Status.UNKNOWN, split count: 2, time: 9.22
Output dim: 7, lower bound: -14.5469192, upper bound: 14.5480980
IS_A2_B1, status: Status.VERIFIED, split count: 2, time: 9.22
Output dim: 7, lower bound: -14.5421468, upper bound: 14.5427956
IS_A2_B2, status: Status.UNKNOWN, split count: 2, time: 9.22
Output dim: 7, lower bound: -14.5456956, upper bound: 14.5456956

## BFS IS instance: IS_A1_B1

### Backsubstitution after applying IS history:
0: -8.8672314, 7.2438998, -8.3950663, 6.8535013, -15.7207308, 15.6389656
1: -7.3019176, 6.5198593, -6.8837366, 6.1762300, -13.4781475, 13.4035959
2: -9.1508274, 5.7149415, -8.6284618, 5.3296161, -14.4804440, 14.3434029
3: -10.7236061, 5.1127486, -10.1435251, 4.8167267, -15.5403328, 15.2562733
4: -9.7993164, 7.7975168, -9.2854586, 7.3931799, -17.1924973, 17.0829754
5: -8.3928757, 6.5447617, -7.9133244, 6.1707859, -14.5636616, 14.4580841
6: -7.8969975, 8.5366650, -7.4555187, 8.1266079, -16.0236034, 15.9921827
7: -9.8314705, 5.9441910, -9.3513021, 5.4657993, -15.2972679, 15.2954931
8: -9.9776325, 7.0994377, -9.4206514, 6.7142367, -16.6918697, 16.5200882
9: -8.1013041, 8.2121372, -7.6150713, 7.7320771, -15.8333817, 15.8272085

### Unstable ReLU Count (Linear/Conv2D Layers)
- layer_idx=0, type=LayerType.Linear, total=256, inp1_unstable=33, inp2_unstable=33, delta_unstable=256
- layer_idx=2, type=LayerType.Linear, total=256, inp1_unstable=69, inp2_unstable=70, delta_unstable=256
- layer_idx=4, type=LayerType.Linear, total=256, inp1_unstable=31, inp2_unstable=25, delta_unstable=256
- layer_idx=6, type=LayerType.Linear, total=256, inp1_unstable=249, inp2_unstable=245, delta_unstable=256
- layer_idx=8, type=LayerType.Linear, total=10, inp1_unstable=10, inp2_unstable=10, delta_unstable=10

Time for backsubstitution: 1.21 seconds

### IS candidates at layer 1
type: A, layer: 1, pos: 187
type: A, layer: 1, pos: 181
type: B, layer: 1, pos: 181
type: A, layer: 1, pos: 196
type: A, layer: 1, pos: 144
type: A, layer: 1, pos: 25
type: B, layer: 1, pos: 232
type: B, layer: 1, pos: 144
type: A, layer: 1, pos: 132
type: A, layer: 1, pos: 195
type: A, layer: 1, pos: 184
type: B, layer: 1, pos: 195
type: A, layer: 1, pos: 71
type: A, layer: 1, pos: 1
type: A, layer: 1, pos: 89
type: B, layer: 1, pos: 25
type: B, layer: 1, pos: 71
type: B, layer: 1, pos: 185
type: A, layer: 1, pos: 77
type: B, layer: 1, pos: 155
type: A, layer: 1, pos: 232
type: A, layer: 1, pos: 219
type: B, layer: 1, pos: 204
type: B, layer: 1, pos: 196
type: A, layer: 1, pos: 185
type: B, layer: 1, pos: 146
type: B, layer: 1, pos: 77
type: A, layer: 1, pos: 197
type: A, layer: 1, pos: 85
type: B, layer: 1, pos: 128
type: B, layer: 1, pos: 132
type: B, layer: 1, pos: 35
type: A, layer: 1, pos: 179
type: B, layer: 1, pos: 85
type: B, layer: 1, pos: 184
type: B, layer: 1, pos: 89
type: B, layer: 1, pos: 197
type: B, layer: 1, pos: 1
type: B, layer: 1, pos: 15
type: B, layer: 1, pos: 140
type: A, layer: 1, pos: 128
type: A, layer: 1, pos: 204
type: A, layer: 1, pos: 35
type: B, layer: 1, pos: 171
type: B, layer: 1, pos: 138
type: A, layer: 1, pos: 171
type: A, layer: 1, pos: 138
type: A, layer: 1, pos: 107
type: A, layer: 1, pos: 15
type: B, layer: 1, pos: 219
type: A, layer: 1, pos: 146
type: B, layer: 1, pos: 107
type: A, layer: 1, pos: 155
type: A, layer: 1, pos: 104
type: B, layer: 1, pos: 179
type: A, layer: 1, pos: 123
type: B, layer: 1, pos: 123
type: A, layer: 1, pos: 251
type: B, layer: 1, pos: 97
type: B, layer: 1, pos: 104
type: B, layer: 1, pos: 233
type: A, layer: 1, pos: 97
type: B, layer: 1, pos: 251
type: A, layer: 1, pos: 57
type: B, layer: 1, pos: 57
type: A, layer: 1, pos: 233

Time for candidate selection: 0.12 seconds

### Candidate
type: A, layer: 1, pos: 187

## Relational analysis of IS_A1_B1_A1

### Relational analysis result of IS_A1_B1_A1
Status: Status.VERIFIED
Output dim: 7, lower bound: -14.5430381, upper bound: 14.5445664
time: 6.04 seconds

## Relational analysis of IS_A1_B1_A2

### Relational analysis result of IS_A1_B1_A2
Status: Status.UNKNOWN
Output dim: 7, lower bound: -14.5430381, upper bound: 14.5452286
time: 4.02 seconds

## BFS IS instance: IS_A1_B2

### Backsubstitution after applying IS history:
0: -9.4409580, 7.7158575, -9.0029488, 7.3502402, -16.7911968, 16.7188034
1: -7.7994318, 6.9319983, -7.4139910, 6.6096821, -14.4091120, 14.3459883
2: -9.7709312, 6.1344261, -9.2876854, 5.8046761, -15.5756073, 15.4221096
3: -11.4306526, 5.4629250, -10.8875570, 5.1826987, -16.6133518, 16.3504829
4: -10.4172726, 8.2936974, -9.9391861, 7.9098296, -18.3270988, 18.2328835
5: -8.9632883, 6.9839573, -8.5249977, 6.6367931, -15.6000814, 15.5089550
6: -8.4268446, 9.0517540, -8.0167885, 8.6539660, -17.0808067, 17.0685425
7: -10.4258718, 6.4670920, -9.9620924, 6.0461173, -16.4719887, 16.4291840
8: -10.6582546, 7.5552454, -10.1262531, 7.2002540, -17.8585091, 17.6814995
9: -8.6621666, 8.7642899, -8.2195110, 8.3277893, -16.9899559, 16.9838009

### Unstable ReLU Count (Linear/Conv2D Layers)
- layer_idx=0, type=LayerType.Linear, total=256, inp1_unstable=33, inp2_unstable=33, delta_unstable=256
- layer_idx=2, type=LayerType.Linear, total=256, inp1_unstable=70, inp2_unstable=70, delta_unstable=256
- layer_idx=4, type=LayerType.Linear, total=256, inp1_unstable=47, inp2_unstable=35, delta_unstable=256
- layer_idx=6, type=LayerType.Linear, total=256, inp1_unstable=255, inp2_unstable=249, delta_unstable=256
- layer_idx=8, type=LayerType.Linear, total=10, inp1_unstable=10, inp2_unstable=10, delta_unstable=10

Time for backsubstitution: 1.21 seconds

### IS candidates at layer 1
type: A, layer: 1, pos: 187
type: A, layer: 1, pos: 181
type: B, layer: 1, pos: 181
type: A, layer: 1, pos: 196
type: A, layer: 1, pos: 144
type: B, layer: 1, pos: 232
type: A, layer: 1, pos: 195
type: B, layer: 1, pos: 144
type: B, layer: 1, pos: 195
type: B, layer: 1, pos: 155
type: A, layer: 1, pos: 25
type: A, layer: 1, pos: 71
type: A, layer: 1, pos: 132
type: A, layer: 1, pos: 232
type: A, layer: 1, pos: 1
type: B, layer: 1, pos: 71
type: B, layer: 1, pos: 196
type: A, layer: 1, pos: 89
type: A, layer: 1, pos: 184
type: B, layer: 1, pos: 185
type: A, layer: 1, pos: 197
type: B, layer: 1, pos: 25
type: A, layer: 1, pos: 77
type: A, layer: 1, pos: 155
type: A, layer: 1, pos: 185
type: A, layer: 1, pos: 219
type: B, layer: 1, pos: 197
type: B, layer: 1, pos: 77
type: B, layer: 1, pos: 35
type: B, layer: 1, pos: 85
type: B, layer: 1, pos: 132
type: B, layer: 1, pos: 204
type: B, layer: 1, pos: 1
type: B, layer: 1, pos: 15
type: A, layer: 1, pos: 85
type: B, layer: 1, pos: 146
type: B, layer: 1, pos: 89
type: B, layer: 1, pos: 128
type: B, layer: 1, pos: 171
type: A, layer: 1, pos: 35
type: B, layer: 1, pos: 140
type: A, layer: 1, pos: 15
type: A, layer: 1, pos: 128
type: A, layer: 1, pos: 171
type: A, layer: 1, pos: 179
type: B, layer: 1, pos: 184
type: B, layer: 1, pos: 138
type: A, layer: 1, pos: 204
type: A, layer: 1, pos: 138
type: B, layer: 1, pos: 219
type: A, layer: 1, pos: 146
type: B, layer: 1, pos: 179
type: B, layer: 1, pos: 97
type: A, layer: 1, pos: 123
type: B, layer: 1, pos: 123
type: B, layer: 1, pos: 233
type: B, layer: 1, pos: 107
type: A, layer: 1, pos: 104
type: A, layer: 1, pos: 107
type: A, layer: 1, pos: 97
type: A, layer: 1, pos: 251
type: B, layer: 1, pos: 104
type: A, layer: 1, pos: 233
type: B, layer: 1, pos: 251
type: A, layer: 1, pos: 57
type: B, layer: 1, pos: 57

Time for candidate selection: 0.12 seconds

### Candidate
type: A, layer: 1, pos: 187

## Relational analysis of IS_A1_B2_A1

### Relational analysis result of IS_A1_B2_A1
Status: Status.UNKNOWN
Output dim: 7, lower bound: -14.5447624, upper bound: 14.5453999
time: 4.82 seconds

## Relational analysis of IS_A1_B2_A2

### Relational analysis result of IS_A1_B2_A2
Status: Status.UNKNOWN
Output dim: 7, lower bound: -14.5447624, upper bound: 14.5480980
time: 9.29 seconds

## BFS IS instance: IS_A2_B2

### Backsubstitution after applying IS history:
0: -10.5248947, 8.5329390, -8.8123608, 7.1903181, -17.7152119, 17.3452988
1: -8.6617489, 7.6438951, -7.2486658, 6.4731855, -15.1349344, 14.8925610
2: -10.7750034, 6.6219125, -9.0770378, 5.6740808, -16.4490852, 15.6989489
3: -12.8121605, 5.9414787, -10.6465759, 5.0700808, -17.8822403, 16.5880547
4: -11.6155968, 9.1381779, -9.7285595, 7.7443118, -19.3599072, 18.8667374
5: -9.9095097, 7.6590281, -8.3303404, 6.4930820, -16.4025917, 15.9893646
6: -9.2859306, 10.0740509, -7.8410296, 8.4758492, -17.7617798, 17.9150810
7: -11.5928020, 6.8066483, -9.7599030, 5.8847671, -17.4775696, 16.5665474
8: -11.7644615, 8.3305283, -9.9020290, 7.0500040, -18.8144646, 18.2325554
9: -9.4881697, 9.5968952, -8.0356579, 8.1402626, -17.6284332, 17.6325531

### Unstable ReLU Count (Linear/Conv2D Layers)
- layer_idx=0, type=LayerType.Linear, total=256, inp1_unstable=33, inp2_unstable=33, delta_unstable=256
- layer_idx=2, type=LayerType.Linear, total=256, inp1_unstable=71, inp2_unstable=69, delta_unstable=256
- layer_idx=4, type=LayerType.Linear, total=256, inp1_unstable=29, inp2_unstable=30, delta_unstable=256
- layer_idx=6, type=LayerType.Linear, total=256, inp1_unstable=249, inp2_unstable=248, delta_unstable=256
- layer_idx=8, type=LayerType.Linear, total=10, inp1_unstable=10, inp2_unstable=10, delta_unstable=10

Time for backsubstitution: 1.22 seconds

### IS candidates at layer 1
type: A, layer: 1, pos: 187
type: B, layer: 1, pos: 181
type: A, layer: 1, pos: 181
type: B, layer: 1, pos: 144
type: B, layer: 1, pos: 25
type: B, layer: 1, pos: 195
type: A, layer: 1, pos: 196
type: B, layer: 1, pos: 196
type: A, layer: 1, pos: 144
type: A, layer: 1, pos: 232
type: A, layer: 1, pos: 195
type: B, layer: 1, pos: 71
type: A, layer: 1, pos: 71
type: B, layer: 1, pos: 232
type: A, layer: 1, pos: 25
type: B, layer: 1, pos: 185
type: B, layer: 1, pos: 132
type: B, layer: 1, pos: 184
type: A, layer: 1, pos: 1
type: B, layer: 1, pos: 89
type: A, layer: 1, pos: 132
type: B, layer: 1, pos: 77
type: A, layer: 1, pos: 184
type: A, layer: 1, pos: 185
type: B, layer: 1, pos: 155
type: B, layer: 1, pos: 1
type: A, layer: 1, pos: 89
type: A, layer: 1, pos: 85
type: A, layer: 1, pos: 155
type: A, layer: 1, pos: 77
type: A, layer: 1, pos: 197
type: B, layer: 1, pos: 197
type: B, layer: 1, pos: 219
type: B, layer: 1, pos: 85
type: A, layer: 1, pos: 219
type: A, layer: 1, pos: 204
type: A, layer: 1, pos: 128
type: B, layer: 1, pos: 128
type: B, layer: 1, pos: 204
type: B, layer: 1, pos: 15
type: B, layer: 1, pos: 179
type: B, layer: 1, pos: 138
type: A, layer: 1, pos: 35
type: B, layer: 1, pos: 140
type: B, layer: 1, pos: 35
type: A, layer: 1, pos: 146
type: A, layer: 1, pos: 15
type: B, layer: 1, pos: 146
type: A, layer: 1, pos: 179
type: A, layer: 1, pos: 138
type: A, layer: 1, pos: 171
type: A, layer: 1, pos: 107
type: B, layer: 1, pos: 171
type: B, layer: 1, pos: 107
type: A, layer: 1, pos: 251
type: A, layer: 1, pos: 123
type: B, layer: 1, pos: 251
type: B, layer: 1, pos: 123
type: B, layer: 1, pos: 104
type: A, layer: 1, pos: 104
type: A, layer: 1, pos: 97
type: B, layer: 1, pos: 233
type: B, layer: 1, pos: 97
type: A, layer: 1, pos: 57
type: B, layer: 1, pos: 57
type: A, layer: 1, pos: 233

Time for candidate selection: 0.13 seconds

### Candidate
type: A, layer: 1, pos: 187

## Relational analysis of IS_A2_B2_A1

### Relational analysis result of IS_A2_B2_A1
Status: Status.VERIFIED
Output dim: 7, lower bound: -14.5427956, upper bound: 14.5421467
time: 2.98 seconds

## Relational analysis of IS_A2_B2_A2

### Relational analysis result of IS_A2_B2_A2
Status: Status.UNKNOWN
Output dim: 7, lower bound: -14.5427956, upper bound: 14.5456956
time: 3.62 seconds

## Summary of splitting at layer (split count: 2)
- Time for IS candidates: 7.97 seconds
IS_A1_B1_A1, status: Status.VERIFIED, split count: 3, time: 7.97
Output dim: 7, lower bound: -14.5430381, upper bound: 14.5445664
IS_A1_B1_A2, status: Status.UNKNOWN, split count: 3, time: 7.97
Output dim: 7, lower bound: -14.5430381, upper bound: 14.5452286
IS_A1_B2_A1, status: Status.UNKNOWN, split count: 3, time: 7.97
Output dim: 7, lower bound: -14.5447624, upper bound: 14.5453999
IS_A1_B2_A2, status: Status.UNKNOWN, split count: 3, time: 7.97
Output dim: 7, lower bound: -14.5447624, upper bound: 14.5480980
IS_A2_B2_A1, status: Status.VERIFIED, split count: 3, time: 7.97
Output dim: 7, lower bound: -14.5427956, upper bound: 14.5421467
IS_A2_B2_A2, status: Status.UNKNOWN, split count: 3, time: 7.97
Output dim: 7, lower bound: -14.5427956, upper bound: 14.5456956

## BFS IS instance: IS_A1_B1_A2

### Backsubstitution after applying IS history:
0: -8.9278440, 7.2875338, -8.3950663, 6.8535013, -15.7813425, 15.6826000
1: -7.3487864, 6.5564070, -6.8837366, 6.1762300, -13.5250168, 13.4401417
2: -9.2051926, 5.7542429, -8.6284618, 5.3296161, -14.5348091, 14.3827047
3: -10.7932682, 5.1391420, -10.1435251, 4.8167267, -15.6099949, 15.2826672
4: -9.8566465, 7.8447084, -9.2854586, 7.3931799, -17.2498245, 17.1301651
5: -8.4488010, 6.5810938, -7.9133244, 6.1707859, -14.6195869, 14.4944181
6: -7.9477148, 8.5837374, -7.4555187, 8.1266079, -16.0743217, 16.0392570
7: -9.8826313, 5.9839482, -9.3513021, 5.4657993, -15.3484306, 15.3352509
8: -10.0378313, 7.1416960, -9.4206514, 6.7142367, -16.7520657, 16.5623455
9: -8.1479759, 8.2557297, -7.6150713, 7.7320771, -15.8800526, 15.8708010

### Unstable ReLU Count (Linear/Conv2D Layers)
- layer_idx=0, type=LayerType.Linear, total=256, inp1_unstable=32, inp2_unstable=33, delta_unstable=255
- layer_idx=2, type=LayerType.Linear, total=256, inp1_unstable=69, inp2_unstable=70, delta_unstable=256
- layer_idx=4, type=LayerType.Linear, total=256, inp1_unstable=32, inp2_unstable=25, delta_unstable=256
- layer_idx=6, type=LayerType.Linear, total=256, inp1_unstable=249, inp2_unstable=245, delta_unstable=256
- layer_idx=8, type=LayerType.Linear, total=10, inp1_unstable=10, inp2_unstable=10, delta_unstable=10

Time for backsubstitution: 1.21 seconds

### IS candidates at layer 1
type: A, layer: 1, pos: 181
type: B, layer: 1, pos: 181
type: A, layer: 1, pos: 196
type: A, layer: 1, pos: 144
type: A, layer: 1, pos: 25
type: B, layer: 1, pos: 232
type: B, layer: 1, pos: 144
type: A, layer: 1, pos: 195
type: A, layer: 1, pos: 132
type: A, layer: 1, pos: 184
type: B, layer: 1, pos: 195
type: A, layer: 1, pos: 71
type: A, layer: 1, pos: 1
type: B, layer: 1, pos: 25
type: A, layer: 1, pos: 89
type: B, layer: 1, pos: 71
type: B, layer: 1, pos: 185
type: A, layer: 1, pos: 77
type: A, layer: 1, pos: 232
type: B, layer: 1, pos: 155
type: A, layer: 1, pos: 219
type: B, layer: 1, pos: 196
type: B, layer: 1, pos: 204
type: A, layer: 1, pos: 185
type: B, layer: 1, pos: 146
type: B, layer: 1, pos: 77
type: A, layer: 1, pos: 197
type: A, layer: 1, pos: 85
type: B, layer: 1, pos: 132
type: B, layer: 1, pos: 128
type: B, layer: 1, pos: 35
type: B, layer: 1, pos: 85
type: A, layer: 1, pos: 179
type: B, layer: 1, pos: 184
type: B, layer: 1, pos: 89
type: B, layer: 1, pos: 1
type: B, layer: 1, pos: 197
type: B, layer: 1, pos: 15
type: B, layer: 1, pos: 140
type: A, layer: 1, pos: 128
type: A, layer: 1, pos: 204
type: A, layer: 1, pos: 35
type: B, layer: 1, pos: 171
type: B, layer: 1, pos: 138
type: A, layer: 1, pos: 138
type: A, layer: 1, pos: 15
type: B, layer: 1, pos: 219
type: A, layer: 1, pos: 171
type: A, layer: 1, pos: 146
type: A, layer: 1, pos: 107
type: B, layer: 1, pos: 107
type: A, layer: 1, pos: 155
type: B, layer: 1, pos: 179
type: A, layer: 1, pos: 104
type: A, layer: 1, pos: 123
type: B, layer: 1, pos: 123
type: B, layer: 1, pos: 97
type: A, layer: 1, pos: 251
type: B, layer: 1, pos: 104
type: B, layer: 1, pos: 233
type: A, layer: 1, pos: 97
type: B, layer: 1, pos: 251
type: A, layer: 1, pos: 57
type: B, layer: 1, pos: 57
type: A, layer: 1, pos: 233

Time for candidate selection: 0.12 seconds

### Candidate
type: A, layer: 1, pos: 181

## Relational analysis of IS_A1_B1_A2_A1
Optimization infeasible because this subproblem isn't reachable.

### Candidate
type: B, layer: 1, pos: 181

## Relational analysis of IS_A1_B1_A2_B1
Optimization infeasible because this subproblem isn't reachable.

### Candidate
type: A, layer: 1, pos: 196

## Relational analysis of IS_A1_B1_A2_A1

### Relational analysis result of IS_A1_B1_A2_A1
Status: Status.VERIFIED
Output dim: 7, lower bound: -14.5372705, upper bound: 14.5387503
time: 4.81 seconds

## Relational analysis of IS_A1_B1_A2_A2

### Relational analysis result of IS_A1_B1_A2_A2
Status: Status.VERIFIED
Output dim: 7, lower bound: -14.5414957, upper bound: 14.5437617
time: 6.86 seconds

## BFS IS instance: IS_A1_B2_A1

### Backsubstitution after applying IS history:
0: -8.3296471, 6.8011222, -9.0029488, 7.3502402, -15.6798878, 15.8040686
1: -6.8286066, 6.1309958, -7.4139910, 6.6096821, -13.4382887, 13.5449858
2: -8.5590858, 5.2883511, -9.2876854, 5.8046761, -14.3637619, 14.5760365
3: -10.0618210, 4.7805834, -10.8875570, 5.1826987, -15.2445202, 15.6681404
4: -9.2138901, 7.3378448, -9.9391861, 7.9098296, -17.1237202, 17.2770309
5: -7.8493633, 6.1237741, -8.5249977, 6.6367931, -14.4861565, 14.6487713
6: -7.3976803, 8.0669155, -8.0167885, 8.6539660, -16.0516472, 16.0837021
7: -9.2825670, 5.4177780, -9.9620924, 6.0461173, -15.3286819, 15.3798666
8: -9.3480167, 6.6629009, -10.1262531, 7.2002540, -16.5482712, 16.7891541
9: -7.5548682, 7.6706157, -8.2195110, 8.3277893, -15.8826580, 15.8901272

### Unstable ReLU Count (Linear/Conv2D Layers)
- layer_idx=0, type=LayerType.Linear, total=256, inp1_unstable=32, inp2_unstable=33, delta_unstable=255
- layer_idx=2, type=LayerType.Linear, total=256, inp1_unstable=70, inp2_unstable=70, delta_unstable=256
- layer_idx=4, type=LayerType.Linear, total=256, inp1_unstable=23, inp2_unstable=35, delta_unstable=256
- layer_idx=6, type=LayerType.Linear, total=256, inp1_unstable=244, inp2_unstable=249, delta_unstable=256
- layer_idx=8, type=LayerType.Linear, total=10, inp1_unstable=10, inp2_unstable=10, delta_unstable=10

Time for backsubstitution: 1.24 seconds

### IS candidates at layer 1
type: B, layer: 1, pos: 181
type: A, layer: 1, pos: 181
type: B, layer: 1, pos: 196
type: B, layer: 1, pos: 144
type: B, layer: 1, pos: 71
type: B, layer: 1, pos: 132
type: B, layer: 1, pos: 184
type: B, layer: 1, pos: 89
type: A, layer: 1, pos: 232
type: A, layer: 1, pos: 144
type: B, layer: 1, pos: 25
type: A, layer: 1, pos: 71
type: B, layer: 1, pos: 195
type: A, layer: 1, pos: 195
type: B, layer: 1, pos: 77
type: B, layer: 1, pos: 1
type: B, layer: 1, pos: 219
type: B, layer: 1, pos: 197
type: A, layer: 1, pos: 146
type: B, layer: 1, pos: 232
type: A, layer: 1, pos: 25
type: A, layer: 1, pos: 197
type: A, layer: 1, pos: 155
type: A, layer: 1, pos: 77
type: B, layer: 1, pos: 85
type: A, layer: 1, pos: 185
type: A, layer: 1, pos: 196
type: A, layer: 1, pos: 85
type: B, layer: 1, pos: 179
type: A, layer: 1, pos: 89
type: A, layer: 1, pos: 184
type: A, layer: 1, pos: 132
type: A, layer: 1, pos: 204
type: B, layer: 1, pos: 185
type: B, layer: 1, pos: 140
type: B, layer: 1, pos: 123
type: A, layer: 1, pos: 1
type: A, layer: 1, pos: 123
type: B, layer: 1, pos: 146
type: B, layer: 1, pos: 107
type: B, layer: 1, pos: 251
type: A, layer: 1, pos: 219
type: A, layer: 1, pos: 35
type: A, layer: 1, pos: 15
type: A, layer: 1, pos: 171
type: A, layer: 1, pos: 128
type: A, layer: 1, pos: 179
type: B, layer: 1, pos: 204
type: A, layer: 1, pos: 138
type: B, layer: 1, pos: 138
type: B, layer: 1, pos: 171
type: B, layer: 1, pos: 35
type: A, layer: 1, pos: 97
type: A, layer: 1, pos: 107
type: B, layer: 1, pos: 128
type: B, layer: 1, pos: 15
type: B, layer: 1, pos: 104
type: B, layer: 1, pos: 155
type: A, layer: 1, pos: 233
type: A, layer: 1, pos: 251
type: B, layer: 1, pos: 97
type: A, layer: 1, pos: 104
type: B, layer: 1, pos: 233
type: B, layer: 1, pos: 57
type: A, layer: 1, pos: 57

Time for candidate selection: 0.12 seconds

### Candidate
type: B, layer: 1, pos: 181

## Relational analysis of IS_A1_B2_A1_B1
Optimization infeasible because this subproblem isn't reachable.

### Candidate
type: A, layer: 1, pos: 181

## Relational analysis of IS_A1_B2_A1_A1
Optimization infeasible because this subproblem isn't reachable.

### Candidate
type: B, layer: 1, pos: 196

## Relational analysis of IS_A1_B2_A1_B1

### Relational analysis result of IS_A1_B2_A1_B1
Status: Status.VERIFIED
Output dim: 7, lower bound: -14.5306006, upper bound: 14.5380102
time: 6.25 seconds

## Relational analysis of IS_A1_B2_A1_B2

### Relational analysis result of IS_A1_B2_A1_B2
Status: Status.VERIFIED
Output dim: 7, lower bound: -14.5414957, upper bound: 14.5439251
time: 5.39 seconds

## BFS IS instance: IS_A1_B2_A2

### Backsubstitution after applying IS history:
0: -8.9292421, 7.2887859, -9.0029488, 7.3502402, -16.2794819, 16.2917328
1: -7.3501306, 6.5573106, -7.4139910, 6.6096821, -13.9598122, 13.9713011
2: -9.2069159, 5.7553377, -9.2876854, 5.8046761, -15.0115919, 15.0430222
3: -10.7948494, 5.1398354, -10.8875570, 5.1826987, -15.9775486, 16.0273933
4: -9.8581409, 7.8458586, -9.9391861, 7.9098296, -17.7679710, 17.7850456
5: -8.4503450, 6.5820780, -8.5249977, 6.6367931, -15.0871353, 15.1070757
6: -7.9490976, 8.5851669, -8.0167885, 8.6539660, -16.6030617, 16.6019554
7: -9.8840208, 5.9855270, -9.9620924, 6.0461173, -15.9301376, 15.9476194
8: -10.0395546, 7.1427965, -10.1262531, 7.2002540, -17.2398090, 17.2690506
9: -8.1493378, 8.2569475, -8.2195110, 8.3277893, -16.4771271, 16.4764576

### Unstable ReLU Count (Linear/Conv2D Layers)
- layer_idx=0, type=LayerType.Linear, total=256, inp1_unstable=32, inp2_unstable=33, delta_unstable=256
- layer_idx=2, type=LayerType.Linear, total=256, inp1_unstable=69, inp2_unstable=70, delta_unstable=256
- layer_idx=4, type=LayerType.Linear, total=256, inp1_unstable=32, inp2_unstable=35, delta_unstable=256
- layer_idx=6, type=LayerType.Linear, total=256, inp1_unstable=249, inp2_unstable=249, delta_unstable=256
- layer_idx=8, type=LayerType.Linear, total=10, inp1_unstable=10, inp2_unstable=10, delta_unstable=10

Time for backsubstitution: 1.22 seconds

### IS candidates at layer 1
type: B, layer: 1, pos: 181
type: A, layer: 1, pos: 181
type: B, layer: 1, pos: 144
type: A, layer: 1, pos: 144
type: B, layer: 1, pos: 25
type: A, layer: 1, pos: 25
type: B, layer: 1, pos: 196
type: A, layer: 1, pos: 196
type: A, layer: 1, pos: 232
type: B, layer: 1, pos: 195
type: A, layer: 1, pos: 195
type: B, layer: 1, pos: 232
type: B, layer: 1, pos: 71
type: A, layer: 1, pos: 71
type: B, layer: 1, pos: 132
type: A, layer: 1, pos: 132
type: B, layer: 1, pos: 184
type: B, layer: 1, pos: 1
type: A, layer: 1, pos: 185
type: B, layer: 1, pos: 77
type: A, layer: 1, pos: 184
type: A, layer: 1, pos: 1
type: B, layer: 1, pos: 185
type: B, layer: 1, pos: 89
type: A, layer: 1, pos: 77
type: A, layer: 1, pos: 89
type: A, layer: 1, pos: 155
type: B, layer: 1, pos: 219
type: B, layer: 1, pos: 155
type: B, layer: 1, pos: 197
type: A, layer: 1, pos: 197
type: B, layer: 1, pos: 85
type: A, layer: 1, pos: 85
type: A, layer: 1, pos: 35
type: A, layer: 1, pos: 219
type: A, layer: 1, pos: 128
type: B, layer: 1, pos: 35
type: B, layer: 1, pos: 128
type: A, layer: 1, pos: 146
type: B, layer: 1, pos: 140
type: B, layer: 1, pos: 146
type: B, layer: 1, pos: 179
type: A, layer: 1, pos: 204
type: A, layer: 1, pos: 15
type: B, layer: 1, pos: 15
type: A, layer: 1, pos: 179
type: B, layer: 1, pos: 204
type: A, layer: 1, pos: 171
type: B, layer: 1, pos: 171
type: B, layer: 1, pos: 107
type: A, layer: 1, pos: 107
type: B, layer: 1, pos: 138
type: A, layer: 1, pos: 138
type: B, layer: 1, pos: 123
type: A, layer: 1, pos: 123
type: B, layer: 1, pos: 104
type: A, layer: 1, pos: 104
type: A, layer: 1, pos: 97
type: B, layer: 1, pos: 97
type: B, layer: 1, pos: 251
type: A, layer: 1, pos: 251
type: A, layer: 1, pos: 233
type: B, layer: 1, pos: 233
type: B, layer: 1, pos: 57
type: A, layer: 1, pos: 57

Time for candidate selection: 0.13 seconds

### Candidate
type: B, layer: 1, pos: 181

## Relational analysis of IS_A1_B2_A2_B1
Optimization infeasible because this subproblem isn't reachable.

### Candidate
type: A, layer: 1, pos: 181

## Relational analysis of IS_A1_B2_A2_A1
Optimization infeasible because this subproblem isn't reachable.

### Candidate
type: B, layer: 1, pos: 144

## Relational analysis of IS_A1_B2_A2_B1
Optimization infeasible because this subproblem isn't reachable.

### Candidate
type: A, layer: 1, pos: 144

## Relational analysis of IS_A1_B2_A2_A1
Optimization infeasible because this subproblem isn't reachable.

### Candidate
type: B, layer: 1, pos: 25

## Relational analysis of IS_A1_B2_A2_B1

### Relational analysis result of IS_A1_B2_A2_B1
Status: Status.VERIFIED
Output dim: 7, lower bound: -14.5415402, upper bound: 14.5430997
time: 4.77 seconds

## Relational analysis of IS_A1_B2_A2_B2

### Relational analysis result of IS_A1_B2_A2_B2
Status: Status.UNKNOWN
Output dim: 7, lower bound: -14.5417921, upper bound: 14.5469509
time: 7.99 seconds

## BFS IS instance: IS_A2_B2_A2

### Backsubstitution after applying IS history:
0: -10.1931286, 8.2650700, -8.8123608, 7.1903181, -17.3834438, 17.0774307
1: -8.3694534, 7.4049911, -7.2486658, 6.4731855, -14.8426390, 14.6536570
2: -10.4125700, 6.3744378, -9.0770378, 5.6740808, -16.0866508, 15.4514751
3: -12.3892517, 5.7361546, -10.6465759, 5.0700808, -17.4593315, 16.3827286
4: -11.2463703, 8.8527966, -9.7285595, 7.7443118, -18.9906826, 18.5813560
5: -9.5800495, 7.4024010, -8.3303404, 6.4930820, -16.0731316, 15.7327366
6: -8.9838810, 9.7767248, -7.8410296, 8.4758492, -17.4597301, 17.6177540
7: -11.2387772, 6.5067730, -9.7599030, 5.8847671, -17.1235447, 16.2666759
8: -11.3806877, 8.0593033, -9.9020290, 7.0500040, -18.4306908, 17.9613304
9: -9.1534939, 9.2629604, -8.0356579, 8.1402626, -17.2937565, 17.2986183

### Unstable ReLU Count (Linear/Conv2D Layers)
- layer_idx=0, type=LayerType.Linear, total=256, inp1_unstable=32, inp2_unstable=33, delta_unstable=256
- layer_idx=2, type=LayerType.Linear, total=256, inp1_unstable=71, inp2_unstable=69, delta_unstable=256
- layer_idx=4, type=LayerType.Linear, total=256, inp1_unstable=26, inp2_unstable=30, delta_unstable=256
- layer_idx=6, type=LayerType.Linear, total=256, inp1_unstable=246, inp2_unstable=248, delta_unstable=256
- layer_idx=8, type=LayerType.Linear, total=10, inp1_unstable=10, inp2_unstable=10, delta_unstable=10

Time for backsubstitution: 1.24 seconds

### IS candidates at layer 1
type: B, layer: 1, pos: 181
type: A, layer: 1, pos: 181
type: B, layer: 1, pos: 144
type: B, layer: 1, pos: 196
type: B, layer: 1, pos: 25
type: B, layer: 1, pos: 195
type: A, layer: 1, pos: 232
type: B, layer: 1, pos: 71
type: A, layer: 1, pos: 144
type: B, layer: 1, pos: 132
type: A, layer: 1, pos: 195
type: B, layer: 1, pos: 184
type: A, layer: 1, pos: 71
type: B, layer: 1, pos: 89
type: A, layer: 1, pos: 196
type: B, layer: 1, pos: 232
type: B, layer: 1, pos: 1
type: B, layer: 1, pos: 185
type: A, layer: 1, pos: 25
type: B, layer: 1, pos: 77
type: A, layer: 1, pos: 155
type: A, layer: 1, pos: 185
type: B, layer: 1, pos: 219
type: A, layer: 1, pos: 85
type: B, layer: 1, pos: 197
type: A, layer: 1, pos: 1
type: A, layer: 1, pos: 132
type: A, layer: 1, pos: 204
type: A, layer: 1, pos: 77
type: B, layer: 1, pos: 85
type: A, layer: 1, pos: 128
type: A, layer: 1, pos: 197
type: B, layer: 1, pos: 179
type: A, layer: 1, pos: 184
type: A, layer: 1, pos: 146
type: B, layer: 1, pos: 155
type: A, layer: 1, pos: 89
type: A, layer: 1, pos: 35
type: B, layer: 1, pos: 138
type: B, layer: 1, pos: 140
type: A, layer: 1, pos: 15
type: B, layer: 1, pos: 128
type: B, layer: 1, pos: 15
type: B, layer: 1, pos: 204
type: B, layer: 1, pos: 35
type: A, layer: 1, pos: 138
type: A, layer: 1, pos: 219
type: A, layer: 1, pos: 171
type: A, layer: 1, pos: 107
type: B, layer: 1, pos: 146
type: A, layer: 1, pos: 179
type: B, layer: 1, pos: 171
type: B, layer: 1, pos: 251
type: B, layer: 1, pos: 104
type: B, layer: 1, pos: 107
type: B, layer: 1, pos: 123
type: A, layer: 1, pos: 123
type: A, layer: 1, pos: 97
type: A, layer: 1, pos: 251
type: A, layer: 1, pos: 104
type: A, layer: 1, pos: 233
type: B, layer: 1, pos: 233
type: B, layer: 1, pos: 57
type: A, layer: 1, pos: 57
type: B, layer: 1, pos: 97

Time for candidate selection: 0.12 seconds

### Candidate
type: B, layer: 1, pos: 181

## Relational analysis of IS_A2_B2_A2_B1
Optimization infeasible because this subproblem isn't reachable.

### Candidate
type: A, layer: 1, pos: 181

## Relational analysis of IS_A2_B2_A2_A1
Optimization infeasible because this subproblem isn't reachable.

### Candidate
type: B, layer: 1, pos: 144

## Relational analysis of IS_A2_B2_A2_B1
Optimization infeasible because this subproblem isn't reachable.

### Candidate
type: B, layer: 1, pos: 196

## Relational analysis of IS_A2_B2_A2_B1

### Relational analysis result of IS_A2_B2_A2_B1
Status: Status.VERIFIED
Output dim: 7, lower bound: -14.5289902, upper bound: 14.5397491
time: 4.62 seconds

## Relational analysis of IS_A2_B2_A2_B2

### Relational analysis result of IS_A2_B2_A2_B2
Status: Status.VERIFIED
Output dim: 7, lower bound: -14.5402292, upper bound: 14.5440281
time: 3.99 seconds

## Summary of splitting at layer (split count: 3)
- Time for IS candidates: 15.83 seconds
IS_A1_B1_A2_A1, status: Status.VERIFIED, split count: 4, time: 15.83
Output dim: 7, lower bound: -14.5372705, upper bound: 14.5387503
IS_A1_B1_A2_A2, status: Status.VERIFIED, split count: 4, time: 15.83
Output dim: 7, lower bound: -14.5414957, upper bound: 14.5437617
IS_A1_B2_A1_B1, status: Status.VERIFIED, split count: 4, time: 15.83
Output dim: 7, lower bound: -14.5306006, upper bound: 14.5380102
IS_A1_B2_A1_B2, status: Status.VERIFIED, split count: 4, time: 15.83
Output dim: 7, lower bound: -14.5414957, upper bound: 14.5439251
IS_A1_B2_A2_B1, status: Status.VERIFIED, split count: 4, time: 15.83
Output dim: 7, lower bound: -14.5415402, upper bound: 14.5430997
IS_A1_B2_A2_B2, status: Status.UNKNOWN, split count: 4, time: 15.83
Output dim: 7, lower bound: -14.5417921, upper bound: 14.5469509
IS_A2_B2_A2_B1, status: Status.VERIFIED, split count: 4, time: 15.83
Output dim: 7, lower bound: -14.5289902, upper bound: 14.5397491
IS_A2_B2_A2_B2, status: Status.VERIFIED, split count: 4, time: 15.83
Output dim: 7, lower bound: -14.5402292, upper bound: 14.5440281

## BFS IS instance: IS_A1_B2_A2_B2

### Backsubstitution after applying IS history:
0: -8.5626488, 6.9908175, -9.3786602, 7.6348906, -16.1975365, 16.3694782
1: -7.0343914, 6.2946072, -7.7030253, 6.8475742, -13.8819656, 13.9976320
2: -8.8179798, 5.5001774, -9.6300602, 5.9268150, -14.7447948, 15.1302376
3: -10.3419838, 4.9312382, -11.3960857, 5.3616848, -15.7036648, 16.3273239
4: -9.4572716, 7.5395823, -10.3492012, 8.2091465, -17.6664181, 17.8887806
5: -8.0881920, 6.3087826, -8.8516703, 6.8622532, -14.9504452, 15.1604528
6: -7.6172175, 8.2509708, -8.3003416, 9.0193529, -16.6365700, 16.5513115
7: -9.4979448, 5.6975961, -10.3701286, 6.1380644, -15.6360092, 16.0677242
8: -9.6167459, 6.8624101, -10.4976187, 7.4749055, -17.0916481, 17.3600292
9: -7.8084731, 7.9141841, -8.5060654, 8.6230488, -16.4315224, 16.4202499

### Unstable ReLU Count (Linear/Conv2D Layers)
- layer_idx=0, type=LayerType.Linear, total=256, inp1_unstable=32, inp2_unstable=32, delta_unstable=256
- layer_idx=2, type=LayerType.Linear, total=256, inp1_unstable=69, inp2_unstable=71, delta_unstable=256
- layer_idx=4, type=LayerType.Linear, total=256, inp1_unstable=27, inp2_unstable=29, delta_unstable=256
- layer_idx=6, type=LayerType.Linear, total=256, inp1_unstable=246, inp2_unstable=248, delta_unstable=256
- layer_idx=8, type=LayerType.Linear, total=10, inp1_unstable=10, inp2_unstable=10, delta_unstable=10

Time for backsubstitution: 1.23 seconds

### IS candidates at layer 1
type: B, layer: 1, pos: 181
type: A, layer: 1, pos: 181
type: A, layer: 1, pos: 25
type: A, layer: 1, pos: 144
type: B, layer: 1, pos: 196
type: B, layer: 1, pos: 144
type: A, layer: 1, pos: 195
type: A, layer: 1, pos: 196
type: A, layer: 1, pos: 232
type: B, layer: 1, pos: 232
type: B, layer: 1, pos: 195
type: B, layer: 1, pos: 71
type: A, layer: 1, pos: 71
type: B, layer: 1, pos: 132
type: A, layer: 1, pos: 132
type: A, layer: 1, pos: 185
type: B, layer: 1, pos: 1
type: A, layer: 1, pos: 184
type: A, layer: 1, pos: 77
type: B, layer: 1, pos: 89
type: B, layer: 1, pos: 184
type: B, layer: 1, pos: 77
type: A, layer: 1, pos: 89
type: B, layer: 1, pos: 185
type: A, layer: 1, pos: 155
type: A, layer: 1, pos: 1
type: B, layer: 1, pos: 85
type: B, layer: 1, pos: 197
type: B, layer: 1, pos: 219
type: A, layer: 1, pos: 197
type: A, layer: 1, pos: 35
type: A, layer: 1, pos: 146
type: A, layer: 1, pos: 219
type: B, layer: 1, pos: 35
type: A, layer: 1, pos: 85
type: A, layer: 1, pos: 204
type: A, layer: 1, pos: 15
type: B, layer: 1, pos: 128
type: A, layer: 1, pos: 128
type: B, layer: 1, pos: 155
type: B, layer: 1, pos: 204
type: B, layer: 1, pos: 140
type: B, layer: 1, pos: 146
type: A, layer: 1, pos: 179
type: B, layer: 1, pos: 179
type: A, layer: 1, pos: 171
type: B, layer: 1, pos: 171
type: B, layer: 1, pos: 15
type: B, layer: 1, pos: 107
type: A, layer: 1, pos: 138
type: B, layer: 1, pos: 138
type: A, layer: 1, pos: 107
type: B, layer: 1, pos: 123
type: A, layer: 1, pos: 123
type: A, layer: 1, pos: 104
type: B, layer: 1, pos: 251
type: B, layer: 1, pos: 104
type: A, layer: 1, pos: 233
type: A, layer: 1, pos: 97
type: B, layer: 1, pos: 97
type: A, layer: 1, pos: 251
type: B, layer: 1, pos: 233
type: B, layer: 1, pos: 57
type: A, layer: 1, pos: 57

Time for candidate selection: 0.12 seconds

### Candidate
type: B, layer: 1, pos: 181

## Relational analysis of IS_A1_B2_A2_B2_B1
Optimization infeasible because this subproblem isn't reachable.

### Candidate
type: A, layer: 1, pos: 181

## Relational analysis of IS_A1_B2_A2_B2_A1
Optimization infeasible because this subproblem isn't reachable.

### Candidate
type: A, layer: 1, pos: 25

## Relational analysis of IS_A1_B2_A2_B2_A1

### Relational analysis result of IS_A1_B2_A2_B2_A1
Status: Status.UNKNOWN
Output dim: 7, lower bound: -14.5456949, upper bound: 14.5469509
time: 5.59 seconds

## Relational analysis of IS_A1_B2_A2_B2_A2

### Relational analysis result of IS_A1_B2_A2_B2_A2
Status: Status.UNKNOWN
Output dim: 7, lower bound: -14.5456949, upper bound: 14.5469509
time: 8.58 seconds

## Summary of splitting at layer (split count: 4)
- Time for IS candidates: 19.99 seconds
IS_A1_B2_A2_B2_A1, status: Status.UNKNOWN, split count: 5, time: 19.99
Output dim: 7, lower bound: -14.5456949, upper bound: 14.5469509
IS_A1_B2_A2_B2_A2, status: Status.UNKNOWN, split count: 5, time: 19.99
Output dim: 7, lower bound: -14.5456949, upper bound: 14.5469509

## BFS IS instance: IS_A1_B2_A2_B2_A1

### Backsubstitution after applying IS history:
0: -8.3119307, 6.7919483, -9.3786602, 7.6348906, -15.9468164, 16.1706085
1: -6.8238068, 6.1178541, -7.7030253, 6.8475742, -13.6713791, 13.8208790
2: -8.5593643, 5.3439221, -9.6300602, 5.9268150, -14.4861794, 14.9739819
3: -10.0325327, 4.7968550, -11.3960857, 5.3616848, -15.3942165, 16.1929398
4: -9.1838837, 7.3321261, -10.3492012, 8.2091465, -17.3930302, 17.6813278
5: -7.8502474, 6.1286697, -8.8516703, 6.8622532, -14.7125006, 14.9803400
6: -7.3981333, 8.0209808, -8.3003416, 9.0193529, -16.4174862, 16.3213234
7: -9.2337646, 5.5301852, -10.3701286, 6.1380644, -15.3718290, 15.9003143
8: -9.3351984, 6.6732249, -10.4976187, 7.4749055, -16.8101006, 17.1708431
9: -7.5848250, 7.6905842, -8.5060654, 8.6230488, -16.2078743, 16.1966496

### Unstable ReLU Count (Linear/Conv2D Layers)
- layer_idx=0, type=LayerType.Linear, total=256, inp1_unstable=31, inp2_unstable=32, delta_unstable=255
- layer_idx=2, type=LayerType.Linear, total=256, inp1_unstable=69, inp2_unstable=71, delta_unstable=256
- layer_idx=4, type=LayerType.Linear, total=256, inp1_unstable=27, inp2_unstable=29, delta_unstable=256
- layer_idx=6, type=LayerType.Linear, total=256, inp1_unstable=246, inp2_unstable=248, delta_unstable=256
- layer_idx=8, type=LayerType.Linear, total=10, inp1_unstable=10, inp2_unstable=10, delta_unstable=10

Time for backsubstitution: 1.24 seconds

### IS candidates at layer 1
type: B, layer: 1, pos: 181
type: A, layer: 1, pos: 181
type: A, layer: 1, pos: 144
type: B, layer: 1, pos: 196
type: B, layer: 1, pos: 144
type: A, layer: 1, pos: 195
type: A, layer: 1, pos: 232
type: A, layer: 1, pos: 196
type: B, layer: 1, pos: 232
type: B, layer: 1, pos: 195
type: B, layer: 1, pos: 71
type: A, layer: 1, pos: 71
type: B, layer: 1, pos: 132
type: B, layer: 1, pos: 1
type: A, layer: 1, pos: 185
type: A, layer: 1, pos: 132
type: B, layer: 1, pos: 89
type: B, layer: 1, pos: 184
type: A, layer: 1, pos: 77
type: A, layer: 1, pos: 184
type: B, layer: 1, pos: 77
type: A, layer: 1, pos: 155
type: B, layer: 1, pos: 185
type: A, layer: 1, pos: 89
type: A, layer: 1, pos: 1
type: B, layer: 1, pos: 85
type: B, layer: 1, pos: 219
type: B, layer: 1, pos: 197
type: A, layer: 1, pos: 197
type: A, layer: 1, pos: 146
type: A, layer: 1, pos: 35
type: A, layer: 1, pos: 204
type: A, layer: 1, pos: 128
type: B, layer: 1, pos: 35
type: A, layer: 1, pos: 15
type: A, layer: 1, pos: 85
type: B, layer: 1, pos: 128
type: A, layer: 1, pos: 219
type: B, layer: 1, pos: 140
type: B, layer: 1, pos: 155
type: B, layer: 1, pos: 204
type: B, layer: 1, pos: 179
type: B, layer: 1, pos: 146
type: A, layer: 1, pos: 179
type: A, layer: 1, pos: 171
type: B, layer: 1, pos: 171
type: B, layer: 1, pos: 15
type: B, layer: 1, pos: 107
type: A, layer: 1, pos: 138
type: B, layer: 1, pos: 138
type: A, layer: 1, pos: 107
type: B, layer: 1, pos: 123
type: A, layer: 1, pos: 123
type: A, layer: 1, pos: 104
type: B, layer: 1, pos: 251
type: B, layer: 1, pos: 104
type: A, layer: 1, pos: 233
type: A, layer: 1, pos: 97
type: B, layer: 1, pos: 97
type: A, layer: 1, pos: 251
type: B, layer: 1, pos: 233
type: B, layer: 1, pos: 57
type: A, layer: 1, pos: 57

Time for candidate selection: 0.12 seconds

### Candidate
type: B, layer: 1, pos: 181

## Relational analysis of IS_A1_B2_A2_B2_A1_B1
Optimization infeasible because this subproblem isn't reachable.

### Candidate
type: A, layer: 1, pos: 181

## Relational analysis of IS_A1_B2_A2_B2_A1_A1
Optimization infeasible because this subproblem isn't reachable.

### Candidate
type: A, layer: 1, pos: 144

## Relational analysis of IS_A1_B2_A2_B2_A1_A1

### Relational analysis result of IS_A1_B2_A2_B2_A1_A1
Status: Status.VERIFIED
Output dim: 7, lower bound: -14.5391646, upper bound: 14.5405298
time: 6.81 seconds

## Relational analysis of IS_A1_B2_A2_B2_A1_A2

### Relational analysis result of IS_A1_B2_A2_B2_A1_A2
Status: Status.UNKNOWN
Output dim: 7, lower bound: -14.5456504, upper bound: 14.5469509
time: 13.43 seconds

## BFS IS instance: IS_A1_B2_A2_B2_A2

### Backsubstitution after applying IS history:
0: -9.3012133, 7.5714560, -9.3786602, 7.6348906, -16.9361038, 16.9501152
1: -7.6304379, 6.7853346, -7.7030253, 6.8475742, -14.4780121, 14.4883595
2: -9.5468712, 5.8699808, -9.6300602, 5.9268150, -15.4736862, 15.5000410
3: -11.2761478, 5.2942743, -11.3960857, 5.3616848, -16.6378307, 16.6903610
4: -10.2641258, 8.1395044, -10.3492012, 8.2091465, -18.4732723, 18.4887047
5: -8.7704115, 6.8003316, -8.8516703, 6.8622532, -15.6326637, 15.6520023
6: -8.2286777, 8.9423656, -8.3003416, 9.0193529, -17.2480278, 17.2427063
7: -10.2768955, 6.0641828, -10.3701286, 6.1380644, -16.4149590, 16.4343109
8: -10.4090796, 7.4093018, -10.4976187, 7.4749055, -17.8839836, 17.9069195
9: -8.4186478, 8.5407791, -8.5060654, 8.6230488, -17.0416965, 17.0468445

### Unstable ReLU Count (Linear/Conv2D Layers)
- layer_idx=0, type=LayerType.Linear, total=256, inp1_unstable=31, inp2_unstable=32, delta_unstable=256
- layer_idx=2, type=LayerType.Linear, total=256, inp1_unstable=71, inp2_unstable=71, delta_unstable=256
- layer_idx=4, type=LayerType.Linear, total=256, inp1_unstable=29, inp2_unstable=29, delta_unstable=256
- layer_idx=6, type=LayerType.Linear, total=256, inp1_unstable=247, inp2_unstable=248, delta_unstable=256
- layer_idx=8, type=LayerType.Linear, total=10, inp1_unstable=10, inp2_unstable=10, delta_unstable=10

Time for backsubstitution: 1.26 seconds

### IS candidates at layer 1
type: B, layer: 1, pos: 181
type: A, layer: 1, pos: 181
type: B, layer: 1, pos: 144
type: A, layer: 1, pos: 144
type: B, layer: 1, pos: 195
type: B, layer: 1, pos: 71
type: A, layer: 1, pos: 195
type: B, layer: 1, pos: 196
type: A, layer: 1, pos: 71
type: A, layer: 1, pos: 232
type: A, layer: 1, pos: 155
type: A, layer: 1, pos: 196
type: B, layer: 1, pos: 232
type: B, layer: 1, pos: 132
type: B, layer: 1, pos: 155
type: B, layer: 1, pos: 1
type: A, layer: 1, pos: 185
type: B, layer: 1, pos: 89
type: B, layer: 1, pos: 185
type: B, layer: 1, pos: 77
type: B, layer: 1, pos: 184
type: A, layer: 1, pos: 77
type: A, layer: 1, pos: 132
type: A, layer: 1, pos: 1
type: A, layer: 1, pos: 89
type: A, layer: 1, pos: 85
type: B, layer: 1, pos: 85
type: A, layer: 1, pos: 35
type: A, layer: 1, pos: 15
type: B, layer: 1, pos: 35
type: A, layer: 1, pos: 171
type: B, layer: 1, pos: 197
type: A, layer: 1, pos: 184
type: B, layer: 1, pos: 219
type: B, layer: 1, pos: 171
type: A, layer: 1, pos: 197
type: B, layer: 1, pos: 15
type: B, layer: 1, pos: 140
type: B, layer: 1, pos: 138
type: A, layer: 1, pos: 128
type: A, layer: 1, pos: 138
type: B, layer: 1, pos: 128
type: B, layer: 1, pos: 179
type: A, layer: 1, pos: 204
type: A, layer: 1, pos: 219
type: B, layer: 1, pos: 204
type: A, layer: 1, pos: 179
type: A, layer: 1, pos: 146
type: B, layer: 1, pos: 104
type: A, layer: 1, pos: 104
type: B, layer: 1, pos: 146
type: A, layer: 1, pos: 97
type: A, layer: 1, pos: 107
type: B, layer: 1, pos: 107
type: B, layer: 1, pos: 97
type: B, layer: 1, pos: 57
type: A, layer: 1, pos: 57
type: A, layer: 1, pos: 233
type: B, layer: 1, pos: 233
type: B, layer: 1, pos: 251
type: B, layer: 1, pos: 123
type: A, layer: 1, pos: 123
type: A, layer: 1, pos: 251

Time for candidate selection: 0.12 seconds

### Candidate
type: B, layer: 1, pos: 181

## Relational analysis of IS_A1_B2_A2_B2_A2_B1
Optimization infeasible because this subproblem isn't reachable.

### Candidate
type: A, layer: 1, pos: 181

## Relational analysis of IS_A1_B2_A2_B2_A2_A1
Optimization infeasible because this subproblem isn't reachable.

### Candidate
type: B, layer: 1, pos: 144

## Relational analysis of IS_A1_B2_A2_B2_A2_B1
Optimization infeasible because this subproblem isn't reachable.

### Candidate
type: A, layer: 1, pos: 144

## Relational analysis of IS_A1_B2_A2_B2_A2_A1

### Relational analysis result of IS_A1_B2_A2_B2_A2_A1
Status: Status.VERIFIED
Output dim: 7, lower bound: -14.5391646, upper bound: 14.5401733
time: 5.70 seconds

## Relational analysis of IS_A1_B2_A2_B2_A2_A2

### Relational analysis result of IS_A1_B2_A2_B2_A2_A2
Status: Status.UNKNOWN
Output dim: 7, lower bound: -14.5456504, upper bound: 14.5468797
time: 5.18 seconds

## Summary of splitting at layer (split count: 5)
- Time for IS candidates: 18.57 seconds
IS_A1_B2_A2_B2_A1_A1, status: Status.VERIFIED, split count: 6, time: 18.57
Output dim: 7, lower bound: -14.5391646, upper bound: 14.5405298
IS_A1_B2_A2_B2_A1_A2, status: Status.UNKNOWN, split count: 6, time: 18.57
Output dim: 7, lower bound: -14.5456504, upper bound: 14.5469509
IS_A1_B2_A2_B2_A2_A1, status: Status.VERIFIED, split count: 6, time: 18.57
Output dim: 7, lower bound: -14.5391646, upper bound: 14.5401733
IS_A1_B2_A2_B2_A2_A2, status: Status.UNKNOWN, split count: 6, time: 18.57
Output dim: 7, lower bound: -14.5456504, upper bound: 14.5468797

## BFS IS instance: IS_A1_B2_A2_B2_A1_A2

### Backsubstitution after applying IS history:
0: -8.1557198, 6.6697416, -9.3786602, 7.6348906, -15.7906084, 16.0484009
1: -6.6940680, 6.0112209, -7.7030253, 6.8475742, -13.5416422, 13.7142458
2: -8.3988991, 5.2438917, -9.6300602, 5.9268150, -14.3257122, 14.8739519
3: -9.8433475, 4.7122660, -11.3960857, 5.3616848, -15.2050304, 16.1083527
4: -9.0125179, 7.2032733, -10.3492012, 8.2091465, -17.2216644, 17.5524750
5: -7.7026291, 6.0173702, -8.8516703, 6.8622532, -14.5648823, 14.8690395
6: -7.2623854, 7.8781481, -8.3003416, 9.0193529, -16.2817383, 16.1784897
7: -9.0685425, 5.4237080, -10.3701286, 6.1380644, -15.2066040, 15.7938366
8: -9.1626472, 6.5543847, -10.4976187, 7.4749055, -16.6375523, 17.0520020
9: -7.4460583, 7.5507956, -8.5060654, 8.6230488, -16.0691071, 16.0568619

### Unstable ReLU Count (Linear/Conv2D Layers)
- layer_idx=0, type=LayerType.Linear, total=256, inp1_unstable=30, inp2_unstable=32, delta_unstable=255
- layer_idx=2, type=LayerType.Linear, total=256, inp1_unstable=69, inp2_unstable=71, delta_unstable=256
- layer_idx=4, type=LayerType.Linear, total=256, inp1_unstable=27, inp2_unstable=29, delta_unstable=256
- layer_idx=6, type=LayerType.Linear, total=256, inp1_unstable=245, inp2_unstable=248, delta_unstable=256
- layer_idx=8, type=LayerType.Linear, total=10, inp1_unstable=10, inp2_unstable=10, delta_unstable=10

Time for backsubstitution: 1.21 seconds

### IS candidates at layer 1
type: B, layer: 1, pos: 144
type: B, layer: 1, pos: 181
type: A, layer: 1, pos: 181
type: B, layer: 1, pos: 196
type: A, layer: 1, pos: 232
type: A, layer: 1, pos: 195
type: B, layer: 1, pos: 195
type: B, layer: 1, pos: 132
type: B, layer: 1, pos: 232
type: B, layer: 1, pos: 71
type: A, layer: 1, pos: 196
type: A, layer: 1, pos: 71
type: B, layer: 1, pos: 184
type: B, layer: 1, pos: 1
type: B, layer: 1, pos: 89
type: A, layer: 1, pos: 185
type: B, layer: 1, pos: 77
type: A, layer: 1, pos: 77
type: A, layer: 1, pos: 132
type: B, layer: 1, pos: 185
type: A, layer: 1, pos: 155
type: A, layer: 1, pos: 184
type: B, layer: 1, pos: 219
type: A, layer: 1, pos: 1
type: A, layer: 1, pos: 89
type: B, layer: 1, pos: 85
type: B, layer: 1, pos: 197
type: A, layer: 1, pos: 146
type: A, layer: 1, pos: 204
type: A, layer: 1, pos: 35
type: A, layer: 1, pos: 197
type: A, layer: 1, pos: 85
type: A, layer: 1, pos: 128
type: B, layer: 1, pos: 179
type: B, layer: 1, pos: 35
type: A, layer: 1, pos: 15
type: B, layer: 1, pos: 140
type: B, layer: 1, pos: 128
type: A, layer: 1, pos: 219
type: B, layer: 1, pos: 155
type: B, layer: 1, pos: 146
type: B, layer: 1, pos: 204
type: B, layer: 1, pos: 171
type: A, layer: 1, pos: 179
type: B, layer: 1, pos: 15
type: A, layer: 1, pos: 171
type: B, layer: 1, pos: 107
type: A, layer: 1, pos: 138
type: B, layer: 1, pos: 138
type: A, layer: 1, pos: 107
type: B, layer: 1, pos: 123
type: A, layer: 1, pos: 123
type: B, layer: 1, pos: 104
type: A, layer: 1, pos: 104
type: B, layer: 1, pos: 251
type: A, layer: 1, pos: 233
type: A, layer: 1, pos: 97
type: B, layer: 1, pos: 97
type: A, layer: 1, pos: 251
type: B, layer: 1, pos: 233
type: B, layer: 1, pos: 57
type: A, layer: 1, pos: 57

Time for candidate selection: 0.12 seconds

### Candidate
type: B, layer: 1, pos: 144

## Relational analysis of IS_A1_B2_A2_B2_A1_A2_B1
Optimization infeasible because this subproblem isn't reachable.

### Candidate
type: B, layer: 1, pos: 181

## Relational analysis of IS_A1_B2_A2_B2_A1_A2_B1
Optimization infeasible because this subproblem isn't reachable.

### Candidate
type: A, layer: 1, pos: 181

## Relational analysis of IS_A1_B2_A2_B2_A1_A2_A1
Optimization infeasible because this subproblem isn't reachable.

### Candidate
type: B, layer: 1, pos: 196

## Relational analysis of IS_A1_B2_A2_B2_A1_A2_B1

### Relational analysis result of IS_A1_B2_A2_B2_A1_A2_B1
Status: Status.VERIFIED
Output dim: 7, lower bound: -14.5389360, upper bound: 14.5412031
time: 4.96 seconds

## Relational analysis of IS_A1_B2_A2_B2_A1_A2_B2

### Relational analysis result of IS_A1_B2_A2_B2_A1_A2_B2
Status: Status.UNKNOWN
Output dim: 7, lower bound: -14.5441769, upper bound: 14.5457765
time: 4.90 seconds

## BFS IS instance: IS_A1_B2_A2_B2_A2_A2

### Backsubstitution after applying IS history:
0: -9.1397943, 7.4451542, -9.3786602, 7.6348906, -16.7746811, 16.8238144
1: -7.4962096, 6.6749787, -7.7030253, 6.8475742, -14.3437843, 14.3780041
2: -9.3808746, 5.7669182, -9.6300602, 5.9268150, -15.3076897, 15.3969784
3: -11.0799637, 5.2070255, -11.3960857, 5.3616848, -16.4416447, 16.6031113
4: -10.0868626, 8.0059509, -10.3492012, 8.2091465, -18.2960091, 18.3551521
5: -8.6178236, 6.6854610, -8.8516703, 6.8622532, -15.4800768, 15.5371313
6: -8.0884151, 8.7942982, -8.3003416, 9.0193529, -17.1077652, 17.0946388
7: -10.1058741, 5.9546185, -10.3701286, 6.1380644, -16.2439384, 16.3247471
8: -10.2306833, 7.2862978, -10.4976187, 7.4749055, -17.7055893, 17.7839165
9: -8.2752724, 8.3960285, -8.5060654, 8.6230488, -16.8983212, 16.9020939

### Unstable ReLU Count (Linear/Conv2D Layers)
- layer_idx=0, type=LayerType.Linear, total=256, inp1_unstable=30, inp2_unstable=32, delta_unstable=256
- layer_idx=2, type=LayerType.Linear, total=256, inp1_unstable=71, inp2_unstable=71, delta_unstable=256
- layer_idx=4, type=LayerType.Linear, total=256, inp1_unstable=28, inp2_unstable=29, delta_unstable=256
- layer_idx=6, type=LayerType.Linear, total=256, inp1_unstable=245, inp2_unstable=248, delta_unstable=256
- layer_idx=8, type=LayerType.Linear, total=10, inp1_unstable=10, inp2_unstable=10, delta_unstable=10

Time for backsubstitution: 1.23 seconds

### IS candidates at layer 1
type: B, layer: 1, pos: 144
type: B, layer: 1, pos: 181
type: A, layer: 1, pos: 181
type: B, layer: 1, pos: 195
type: B, layer: 1, pos: 71
type: B, layer: 1, pos: 196
type: A, layer: 1, pos: 71
type: A, layer: 1, pos: 195
type: A, layer: 1, pos: 232
type: A, layer: 1, pos: 155
type: A, layer: 1, pos: 196
type: B, layer: 1, pos: 132
type: B, layer: 1, pos: 232
type: B, layer: 1, pos: 1
type: B, layer: 1, pos: 89
type: B, layer: 1, pos: 77
type: B, layer: 1, pos: 184
type: B, layer: 1, pos: 185
type: B, layer: 1, pos: 155
type: A, layer: 1, pos: 185
type: A, layer: 1, pos: 1
type: A, layer: 1, pos: 77
type: A, layer: 1, pos: 85
type: A, layer: 1, pos: 132
type: A, layer: 1, pos: 35
type: A, layer: 1, pos: 89
type: B, layer: 1, pos: 85
type: B, layer: 1, pos: 219
type: B, layer: 1, pos: 35
type: B, layer: 1, pos: 197
type: B, layer: 1, pos: 171
type: A, layer: 1, pos: 15
type: A, layer: 1, pos: 171
type: B, layer: 1, pos: 138
type: B, layer: 1, pos: 179
type: A, layer: 1, pos: 197
type: B, layer: 1, pos: 140
type: A, layer: 1, pos: 128
type: B, layer: 1, pos: 15
type: A, layer: 1, pos: 184
type: A, layer: 1, pos: 138
type: A, layer: 1, pos: 204
type: B, layer: 1, pos: 128
type: A, layer: 1, pos: 219
type: A, layer: 1, pos: 146
type: B, layer: 1, pos: 204
type: A, layer: 1, pos: 179
type: B, layer: 1, pos: 104
type: A, layer: 1, pos: 107
type: B, layer: 1, pos: 146
type: A, layer: 1, pos: 97
type: A, layer: 1, pos: 104
type: B, layer: 1, pos: 107
type: B, layer: 1, pos: 57
type: A, layer: 1, pos: 57
type: B, layer: 1, pos: 97
type: B, layer: 1, pos: 233
type: A, layer: 1, pos: 233
type: A, layer: 1, pos: 123
type: B, layer: 1, pos: 123
type: B, layer: 1, pos: 251
type: A, layer: 1, pos: 251

Time for candidate selection: 0.13 seconds

### Candidate
type: B, layer: 1, pos: 144

## Relational analysis of IS_A1_B2_A2_B2_A2_A2_B1
Optimization infeasible because this subproblem isn't reachable.

### Candidate
type: B, layer: 1, pos: 181

## Relational analysis of IS_A1_B2_A2_B2_A2_A2_B1
Optimization infeasible because this subproblem isn't reachable.

### Candidate
type: A, layer: 1, pos: 181

## Relational analysis of IS_A1_B2_A2_B2_A2_A2_A1
Optimization infeasible because this subproblem isn't reachable.

### Candidate
type: B, layer: 1, pos: 195

## Relational analysis of IS_A1_B2_A2_B2_A2_A2_B1

### Relational analysis result of IS_A1_B2_A2_B2_A2_A2_B1
Status: Status.VERIFIED
Output dim: 7, lower bound: -14.5414250, upper bound: 14.5417868
time: 7.96 seconds

## Relational analysis of IS_A1_B2_A2_B2_A2_A2_B2
Optimization infeasible because this subproblem isn't reachable.

### Candidate
type: B, layer: 1, pos: 71

## Relational analysis of IS_A1_B2_A2_B2_A2_A2_B1
Optimization infeasible because this subproblem isn't reachable.

### Candidate
type: B, layer: 1, pos: 196

## Relational analysis of IS_A1_B2_A2_B2_A2_A2_B1

### Relational analysis result of IS_A1_B2_A2_B2_A2_A2_B1
Status: Status.VERIFIED
Output dim: 7, lower bound: -14.5388269, upper bound: 14.5409368
time: 9.42 seconds

## Relational analysis of IS_A1_B2_A2_B2_A2_A2_B2

### Relational analysis result of IS_A1_B2_A2_B2_A2_A2_B2
Status: Status.UNKNOWN
Output dim: 7, lower bound: -14.5440658, upper bound: 14.5455654
time: 4.70 seconds

## Summary of splitting at layer (split count: 6)
- Time for IS candidates: 36.19 seconds
IS_A1_B2_A2_B2_A1_A2_B1, status: Status.VERIFIED, split count: 7, time: 36.19
Output dim: 7, lower bound: -14.5389360, upper bound: 14.5412031
IS_A1_B2_A2_B2_A1_A2_B2, status: Status.UNKNOWN, split count: 7, time: 36.19
Output dim: 7, lower bound: -14.5441769, upper bound: 14.5457765
IS_A1_B2_A2_B2_A2_A2_B1, status: Status.VERIFIED, split count: 7, time: 36.19
Output dim: 7, lower bound: -14.5388269, upper bound: 14.5409368
IS_A1_B2_A2_B2_A2_A2_B2, status: Status.UNKNOWN, split count: 7, time: 36.19
Output dim: 7, lower bound: -14.5440658, upper bound: 14.5455654

## BFS IS instance: IS_A1_B2_A2_B2_A1_A2_B2

### Backsubstitution after applying IS history:
0: -8.0245161, 6.5639658, -8.6709995, 7.0688548, -15.0933704, 15.2349653
1: -6.5819645, 5.9169636, -7.0915613, 6.3387699, -12.9207344, 13.0085239
2: -8.2591877, 5.1555128, -8.8749981, 5.4518685, -13.7110558, 14.0305109
3: -9.6799831, 4.6360025, -10.4889832, 4.9400735, -14.6200562, 15.1249857
4: -8.8679523, 7.0921578, -9.5657997, 7.6046371, -16.4725876, 16.6579552
5: -7.5747232, 5.9183369, -8.1566887, 6.3330913, -13.9078140, 14.0750237
6: -7.1460199, 7.7585864, -7.6730251, 8.3708897, -15.5169096, 15.4316120
7: -8.9306164, 5.3184347, -9.6130085, 5.5768275, -14.5074434, 14.9314423
8: -9.0122156, 6.4509997, -9.6916828, 6.9120235, -15.9242353, 16.1426830
9: -7.3205867, 7.4240370, -7.8260403, 7.9338923, -15.2544785, 15.2500744

### Unstable ReLU Count (Linear/Conv2D Layers)
- layer_idx=0, type=LayerType.Linear, total=256, inp1_unstable=30, inp2_unstable=31, delta_unstable=255
- layer_idx=2, type=LayerType.Linear, total=256, inp1_unstable=69, inp2_unstable=68, delta_unstable=256
- layer_idx=4, type=LayerType.Linear, total=256, inp1_unstable=26, inp2_unstable=24, delta_unstable=256
- layer_idx=6, type=LayerType.Linear, total=256, inp1_unstable=245, inp2_unstable=244, delta_unstable=256
- layer_idx=8, type=LayerType.Linear, total=10, inp1_unstable=10, inp2_unstable=10, delta_unstable=10

Time for backsubstitution: 1.24 seconds

### IS candidates at layer 1
type: B, layer: 1, pos: 144
type: A, layer: 1, pos: 196
type: A, layer: 1, pos: 181
type: B, layer: 1, pos: 181
type: B, layer: 1, pos: 232
type: A, layer: 1, pos: 195
type: B, layer: 1, pos: 195
type: A, layer: 1, pos: 132
type: A, layer: 1, pos: 71
type: A, layer: 1, pos: 184
type: B, layer: 1, pos: 71
type: A, layer: 1, pos: 89
type: A, layer: 1, pos: 1
type: A, layer: 1, pos: 232
type: B, layer: 1, pos: 185
type: B, layer: 1, pos: 155
type: A, layer: 1, pos: 77
type: A, layer: 1, pos: 219
type: A, layer: 1, pos: 185
type: B, layer: 1, pos: 132
type: B, layer: 1, pos: 204
type: A, layer: 1, pos: 197
type: B, layer: 1, pos: 77
type: B, layer: 1, pos: 146
type: A, layer: 1, pos: 85
type: B, layer: 1, pos: 89
type: B, layer: 1, pos: 184
type: B, layer: 1, pos: 1
type: B, layer: 1, pos: 197
type: B, layer: 1, pos: 35
type: B, layer: 1, pos: 128
type: A, layer: 1, pos: 179
type: B, layer: 1, pos: 85
type: A, layer: 1, pos: 155
type: B, layer: 1, pos: 15
type: B, layer: 1, pos: 140
type: A, layer: 1, pos: 35
type: A, layer: 1, pos: 128
type: A, layer: 1, pos: 204
type: B, layer: 1, pos: 219
type: B, layer: 1, pos: 138
type: A, layer: 1, pos: 15
type: B, layer: 1, pos: 171
type: A, layer: 1, pos: 171
type: A, layer: 1, pos: 138
type: A, layer: 1, pos: 146
type: B, layer: 1, pos: 179
type: A, layer: 1, pos: 107
type: B, layer: 1, pos: 107
type: A, layer: 1, pos: 123
type: A, layer: 1, pos: 104
type: B, layer: 1, pos: 123
type: A, layer: 1, pos: 251
type: B, layer: 1, pos: 97
type: B, layer: 1, pos: 233
type: B, layer: 1, pos: 104
type: B, layer: 1, pos: 251
type: A, layer: 1, pos: 97
type: A, layer: 1, pos: 57
type: A, layer: 1, pos: 233
type: B, layer: 1, pos: 57

Time for candidate selection: 0.13 seconds

### Candidate
type: B, layer: 1, pos: 144

## Relational analysis of IS_A1_B2_A2_B2_A1_A2_B2_B1
Optimization infeasible because this subproblem isn't reachable.

### Candidate
type: A, layer: 1, pos: 196

## Relational analysis of IS_A1_B2_A2_B2_A1_A2_B2_A1

### Relational analysis result of IS_A1_B2_A2_B2_A1_A2_B2_A1
Status: Status.VERIFIED
Output dim: 7, lower bound: -14.5414995, upper bound: 14.5422842
time: 3.62 seconds

## Relational analysis of IS_A1_B2_A2_B2_A1_A2_B2_A2

### Relational analysis result of IS_A1_B2_A2_B2_A1_A2_B2_A2
Status: Status.UNKNOWN
Output dim: 7, lower bound: -14.5414995, upper bound: 14.5457765
time: 5.26 seconds

## BFS IS instance: IS_A1_B2_A2_B2_A2_A2_B2

### Backsubstitution after applying IS history:
0: -9.0164318, 7.3470612, -8.6709995, 7.0688548, -16.0852833, 16.0180607
1: -7.3909788, 6.5874958, -7.0915613, 6.3387699, -13.7297487, 13.6790571
2: -9.2495985, 5.6851921, -8.8749981, 5.4518685, -14.7014675, 14.5601902
3: -10.9254665, 5.1370215, -10.4889832, 4.9400735, -15.8655396, 15.6260052
4: -9.9506397, 7.9013019, -9.5657997, 7.6046371, -17.5552750, 17.4671021
5: -8.4979954, 6.5937324, -8.1566887, 6.3330913, -14.8310871, 14.7504215
6: -7.9797735, 8.6823578, -7.6730251, 8.3708897, -16.3506622, 16.3553829
7: -9.9760618, 5.8591294, -9.6130085, 5.5768275, -15.5528889, 15.4721375
8: -10.0906734, 7.1890063, -9.6916828, 6.9120235, -17.0026932, 16.8806896
9: -8.1588945, 8.2773352, -7.8260403, 7.9338923, -16.0927868, 16.1033726

### Unstable ReLU Count (Linear/Conv2D Layers)
- layer_idx=0, type=LayerType.Linear, total=256, inp1_unstable=30, inp2_unstable=31, delta_unstable=256
- layer_idx=2, type=LayerType.Linear, total=256, inp1_unstable=70, inp2_unstable=68, delta_unstable=256
- layer_idx=4, type=LayerType.Linear, total=256, inp1_unstable=27, inp2_unstable=24, delta_unstable=256
- layer_idx=6, type=LayerType.Linear, total=256, inp1_unstable=245, inp2_unstable=244, delta_unstable=256
- layer_idx=8, type=LayerType.Linear, total=10, inp1_unstable=10, inp2_unstable=10, delta_unstable=10

Time for backsubstitution: 1.25 seconds

### IS candidates at layer 1
type: B, layer: 1, pos: 144
type: A, layer: 1, pos: 196
type: A, layer: 1, pos: 181
type: B, layer: 1, pos: 181
type: B, layer: 1, pos: 195
type: A, layer: 1, pos: 195
type: B, layer: 1, pos: 232
type: B, layer: 1, pos: 155
type: A, layer: 1, pos: 132
type: A, layer: 1, pos: 232
type: A, layer: 1, pos: 71
type: B, layer: 1, pos: 185
type: A, layer: 1, pos: 1
type: B, layer: 1, pos: 71
type: A, layer: 1, pos: 89
type: A, layer: 1, pos: 197
type: A, layer: 1, pos: 184
type: A, layer: 1, pos: 155
type: A, layer: 1, pos: 85
type: A, layer: 1, pos: 219
type: B, layer: 1, pos: 197
type: B, layer: 1, pos: 204
type: A, layer: 1, pos: 185
type: B, layer: 1, pos: 132
type: B, layer: 1, pos: 77
type: A, layer: 1, pos: 77
type: B, layer: 1, pos: 89
type: B, layer: 1, pos: 35
type: B, layer: 1, pos: 146
type: B, layer: 1, pos: 15
type: B, layer: 1, pos: 1
type: B, layer: 1, pos: 85
type: B, layer: 1, pos: 184
type: B, layer: 1, pos: 138
type: A, layer: 1, pos: 35
type: A, layer: 1, pos: 204
type: B, layer: 1, pos: 128
type: B, layer: 1, pos: 140
type: B, layer: 1, pos: 171
type: A, layer: 1, pos: 128
type: A, layer: 1, pos: 179
type: A, layer: 1, pos: 138
type: A, layer: 1, pos: 171
type: B, layer: 1, pos: 219
type: A, layer: 1, pos: 15
type: B, layer: 1, pos: 179
type: A, layer: 1, pos: 123
type: A, layer: 1, pos: 146
type: B, layer: 1, pos: 123
type: B, layer: 1, pos: 233
type: A, layer: 1, pos: 107
type: A, layer: 1, pos: 251
type: B, layer: 1, pos: 97
type: B, layer: 1, pos: 104
type: A, layer: 1, pos: 104
type: A, layer: 1, pos: 97
type: A, layer: 1, pos: 233
type: B, layer: 1, pos: 107
type: B, layer: 1, pos: 251
type: A, layer: 1, pos: 57
type: B, layer: 1, pos: 57

Time for candidate selection: 0.13 seconds

### Candidate
type: B, layer: 1, pos: 144

## Relational analysis of IS_A1_B2_A2_B2_A2_A2_B2_B1
Optimization infeasible because this subproblem isn't reachable.

### Candidate
type: A, layer: 1, pos: 196

## Relational analysis of IS_A1_B2_A2_B2_A2_A2_B2_A1

### Relational analysis result of IS_A1_B2_A2_B2_A2_A2_B2_A1
Status: Status.VERIFIED
Output dim: 7, lower bound: -14.5412440, upper bound: 14.5417596
time: 5.90 seconds

## Relational analysis of IS_A1_B2_A2_B2_A2_A2_B2_A2

### Relational analysis result of IS_A1_B2_A2_B2_A2_A2_B2_A2
Status: Status.UNKNOWN
Output dim: 7, lower bound: -14.5412440, upper bound: 14.5455654
time: 4.39 seconds

## Summary of splitting at layer (split count: 7)
- Time for IS candidates: 13.65 seconds
IS_A1_B2_A2_B2_A1_A2_B2_A1, status: Status.VERIFIED, split count: 8, time: 13.65
Output dim: 7, lower bound: -14.5414995, upper bound: 14.5422842
IS_A1_B2_A2_B2_A1_A2_B2_A2, status: Status.UNKNOWN, split count: 8, time: 13.65
Output dim: 7, lower bound: -14.5414995, upper bound: 14.5457765
IS_A1_B2_A2_B2_A2_A2_B2_A1, status: Status.VERIFIED, split count: 8, time: 13.65
Output dim: 7, lower bound: -14.5412440, upper bound: 14.5417596
IS_A1_B2_A2_B2_A2_A2_B2_A2, status: Status.UNKNOWN, split count: 8, time: 13.65
Output dim: 7, lower bound: -14.5412440, upper bound: 14.5455654

## BFS IS instance: IS_A1_B2_A2_B2_A1_A2_B2_A2

### Backsubstitution after applying IS history:
0: -7.4651093, 6.1137004, -8.6709995, 7.0688548, -14.5339642, 14.7847004
1: -6.0979452, 5.5158386, -7.0915613, 6.3387699, -12.4367151, 12.6073990
2: -7.6572151, 4.7801495, -8.8749981, 5.4518685, -13.1090832, 13.6551476
3: -8.9715481, 4.3136401, -10.4889832, 4.9400735, -13.9116211, 14.8026237
4: -8.2448711, 6.6112976, -9.5657997, 7.6046371, -15.8495064, 16.1770973
5: -7.0224705, 5.5037708, -8.1566887, 6.3330913, -13.3555622, 13.6604595
6: -6.6476517, 7.2454195, -7.6730251, 8.3708897, -15.0185404, 14.9184446
7: -8.3347683, 4.8763151, -9.6130085, 5.5768275, -13.9115934, 14.4893236
8: -8.3732653, 6.0050087, -9.6916828, 6.9120235, -15.2852888, 15.6966915
9: -6.7888875, 6.8764267, -7.8260403, 7.9338923, -14.7227783, 14.7024651

### Unstable ReLU Count (Linear/Conv2D Layers)
- layer_idx=0, type=LayerType.Linear, total=256, inp1_unstable=29, inp2_unstable=31, delta_unstable=255
- layer_idx=2, type=LayerType.Linear, total=256, inp1_unstable=68, inp2_unstable=68, delta_unstable=256
- layer_idx=4, type=LayerType.Linear, total=256, inp1_unstable=23, inp2_unstable=24, delta_unstable=256
- layer_idx=6, type=LayerType.Linear, total=256, inp1_unstable=242, inp2_unstable=244, delta_unstable=256
- layer_idx=8, type=LayerType.Linear, total=10, inp1_unstable=10, inp2_unstable=10, delta_unstable=10

Time for backsubstitution: 1.24 seconds

### IS candidates at layer 1
type: B, layer: 1, pos: 144
type: B, layer: 1, pos: 181
type: A, layer: 1, pos: 181
type: A, layer: 1, pos: 195
type: B, layer: 1, pos: 195
type: A, layer: 1, pos: 232
type: B, layer: 1, pos: 232
type: B, layer: 1, pos: 71
type: A, layer: 1, pos: 71
type: B, layer: 1, pos: 132
type: B, layer: 1, pos: 184
type: A, layer: 1, pos: 185
type: B, layer: 1, pos: 89
type: B, layer: 1, pos: 1
type: A, layer: 1, pos: 132
type: B, layer: 1, pos: 185
type: A, layer: 1, pos: 184
type: B, layer: 1, pos: 77
type: A, layer: 1, pos: 77
type: A, layer: 1, pos: 89
type: A, layer: 1, pos: 1
type: A, layer: 1, pos: 155
type: B, layer: 1, pos: 219
type: B, layer: 1, pos: 85
type: B, layer: 1, pos: 197
type: A, layer: 1, pos: 204
type: A, layer: 1, pos: 197
type: A, layer: 1, pos: 85
type: A, layer: 1, pos: 146
type: A, layer: 1, pos: 35
type: A, layer: 1, pos: 219
type: A, layer: 1, pos: 128
type: B, layer: 1, pos: 204
type: B, layer: 1, pos: 35
type: B, layer: 1, pos: 155
type: B, layer: 1, pos: 140
type: B, layer: 1, pos: 128
type: B, layer: 1, pos: 179
type: B, layer: 1, pos: 146
type: A, layer: 1, pos: 15
type: A, layer: 1, pos: 179
type: B, layer: 1, pos: 15
type: B, layer: 1, pos: 171
type: A, layer: 1, pos: 138
type: B, layer: 1, pos: 138
type: A, layer: 1, pos: 171
type: B, layer: 1, pos: 107
type: A, layer: 1, pos: 107
type: B, layer: 1, pos: 123
type: A, layer: 1, pos: 123
type: B, layer: 1, pos: 104
type: A, layer: 1, pos: 104
type: B, layer: 1, pos: 251
type: A, layer: 1, pos: 251
type: A, layer: 1, pos: 233
type: A, layer: 1, pos: 97
type: B, layer: 1, pos: 97
type: B, layer: 1, pos: 233
type: B, layer: 1, pos: 57
type: A, layer: 1, pos: 57

Time for candidate selection: 0.13 seconds

### Candidate
type: B, layer: 1, pos: 144

## Relational analysis of IS_A1_B2_A2_B2_A1_A2_B2_A2_B1
Optimization infeasible because this subproblem isn't reachable.

### Candidate
type: B, layer: 1, pos: 181

## Relational analysis of IS_A1_B2_A2_B2_A1_A2_B2_A2_B1
Optimization infeasible because this subproblem isn't reachable.

### Candidate
type: A, layer: 1, pos: 181

## Relational analysis of IS_A1_B2_A2_B2_A1_A2_B2_A2_A1
Optimization infeasible because this subproblem isn't reachable.

### Candidate
type: A, layer: 1, pos: 195

## Relational analysis of IS_A1_B2_A2_B2_A1_A2_B2_A2_A1
Optimization infeasible because this subproblem isn't reachable.

### Candidate
type: B, layer: 1, pos: 195

## Relational analysis of IS_A1_B2_A2_B2_A1_A2_B2_A2_B1
Optimization infeasible because this subproblem isn't reachable.

### Candidate
type: A, layer: 1, pos: 232

## Relational analysis of IS_A1_B2_A2_B2_A1_A2_B2_A2_A1

### Relational analysis result of IS_A1_B2_A2_B2_A1_A2_B2_A2_A1
Status: Status.UNKNOWN
Output dim: 7, lower bound: -14.5386979, upper bound: 14.5457363
time: 4.19 seconds

## Relational analysis of IS_A1_B2_A2_B2_A1_A2_B2_A2_A2

### Relational analysis result of IS_A1_B2_A2_B2_A1_A2_B2_A2_A2
Status: Status.UNKNOWN
Output dim: 7, lower bound: -14.5386568, upper bound: 14.5456522
time: 8.14 seconds

## BFS IS instance: IS_A1_B2_A2_B2_A2_A2_B2_A2

### Backsubstitution after applying IS history:
0: -8.4535503, 6.8957567, -8.6709995, 7.0688548, -15.5224037, 15.5667562
1: -6.9075065, 6.1881065, -7.0915613, 6.3387699, -13.2462769, 13.2796679
2: -8.6481619, 5.3129158, -8.8749981, 5.4518685, -14.1000290, 14.1879139
3: -10.2177944, 4.8166976, -10.4889832, 4.9400735, -15.1578674, 15.3056812
4: -9.3262653, 7.4224491, -9.5657997, 7.6046371, -16.9308968, 16.9882488
5: -7.9465761, 6.1778445, -8.1566887, 6.3330913, -14.2796650, 14.3345337
6: -7.4821081, 8.1698608, -7.6730251, 8.3708897, -15.8529978, 15.8428860
7: -9.3793526, 5.4228888, -9.6130085, 5.5768275, -14.9561796, 15.0358973
8: -9.4503431, 6.7436471, -9.6916828, 6.9120235, -16.3623657, 16.4353294
9: -7.6284404, 7.7333913, -7.8260403, 7.9338923, -15.5623302, 15.5594311

### Unstable ReLU Count (Linear/Conv2D Layers)
- layer_idx=0, type=LayerType.Linear, total=256, inp1_unstable=29, inp2_unstable=31, delta_unstable=256
- layer_idx=2, type=LayerType.Linear, total=256, inp1_unstable=68, inp2_unstable=68, delta_unstable=256
- layer_idx=4, type=LayerType.Linear, total=256, inp1_unstable=23, inp2_unstable=24, delta_unstable=256
- layer_idx=6, type=LayerType.Linear, total=256, inp1_unstable=242, inp2_unstable=244, delta_unstable=256
- layer_idx=8, type=LayerType.Linear, total=10, inp1_unstable=10, inp2_unstable=10, delta_unstable=10

Time for backsubstitution: 1.24 seconds

### IS candidates at layer 1
type: B, layer: 1, pos: 144
type: B, layer: 1, pos: 181
type: A, layer: 1, pos: 181
type: B, layer: 1, pos: 195
type: A, layer: 1, pos: 195
type: B, layer: 1, pos: 71
type: A, layer: 1, pos: 232
type: A, layer: 1, pos: 71
type: B, layer: 1, pos: 232
type: B, layer: 1, pos: 132
type: B, layer: 1, pos: 89
type: B, layer: 1, pos: 197
type: A, layer: 1, pos: 197
type: B, layer: 1, pos: 184
type: A, layer: 1, pos: 155
type: A, layer: 1, pos: 89
type: B, layer: 1, pos: 77
type: A, layer: 1, pos: 132
type: B, layer: 1, pos: 155
type: A, layer: 1, pos: 85
type: B, layer: 1, pos: 1
type: B, layer: 1, pos: 185
type: A, layer: 1, pos: 77
type: B, layer: 1, pos: 85
type: A, layer: 1, pos: 1
type: B, layer: 1, pos: 219
type: A, layer: 1, pos: 184
type: A, layer: 1, pos: 185
type: A, layer: 1, pos: 146
type: A, layer: 1, pos: 219
type: B, layer: 1, pos: 140
type: A, layer: 1, pos: 204
type: B, layer: 1, pos: 138
type: A, layer: 1, pos: 35
type: B, layer: 1, pos: 146
type: B, layer: 1, pos: 179
type: B, layer: 1, pos: 35
type: B, layer: 1, pos: 171
type: A, layer: 1, pos: 138
type: A, layer: 1, pos: 171
type: B, layer: 1, pos: 123
type: A, layer: 1, pos: 123
type: B, layer: 1, pos: 204
type: B, layer: 1, pos: 15
type: A, layer: 1, pos: 15
type: A, layer: 1, pos: 179
type: A, layer: 1, pos: 128
type: B, layer: 1, pos: 128
type: B, layer: 1, pos: 233
type: B, layer: 1, pos: 104
type: A, layer: 1, pos: 233
type: B, layer: 1, pos: 251
type: A, layer: 1, pos: 251
type: A, layer: 1, pos: 104
type: A, layer: 1, pos: 107
type: B, layer: 1, pos: 107
type: A, layer: 1, pos: 97
type: B, layer: 1, pos: 97
type: B, layer: 1, pos: 57
type: A, layer: 1, pos: 57

Time for candidate selection: 0.13 seconds

### Candidate
type: B, layer: 1, pos: 144

## Relational analysis of IS_A1_B2_A2_B2_A2_A2_B2_A2_B1
Optimization infeasible because this subproblem isn't reachable.

### Candidate
type: B, layer: 1, pos: 181

## Relational analysis of IS_A1_B2_A2_B2_A2_A2_B2_A2_B1
Optimization infeasible because this subproblem isn't reachable.

### Candidate
type: A, layer: 1, pos: 181

## Relational analysis of IS_A1_B2_A2_B2_A2_A2_B2_A2_A1
Optimization infeasible because this subproblem isn't reachable.

### Candidate
type: B, layer: 1, pos: 195

## Relational analysis of IS_A1_B2_A2_B2_A2_A2_B2_A2_B1
Optimization infeasible because this subproblem isn't reachable.

### Candidate
type: A, layer: 1, pos: 195

## Relational analysis of IS_A1_B2_A2_B2_A2_A2_B2_A2_A1
Optimization infeasible because this subproblem isn't reachable.

### Candidate
type: B, layer: 1, pos: 71

## Relational analysis of IS_A1_B2_A2_B2_A2_A2_B2_A2_B1
Optimization infeasible because this subproblem isn't reachable.

### Candidate
type: A, layer: 1, pos: 232

## Relational analysis of IS_A1_B2_A2_B2_A2_A2_B2_A2_A1

### Relational analysis result of IS_A1_B2_A2_B2_A2_A2_B2_A2_A1
Status: Status.VERIFIED
Output dim: 7, lower bound: -14.5385127, upper bound: 14.5401438
time: 4.86 seconds

## Relational analysis of IS_A1_B2_A2_B2_A2_A2_B2_A2_A2

### Relational analysis result of IS_A1_B2_A2_B2_A2_A2_B2_A2_A2
Status: Status.UNKNOWN
Output dim: 7, lower bound: -14.5384971, upper bound: 14.5454915
time: 6.96 seconds

## Summary of splitting at layer (split count: 8)
- Time for IS candidates: 27.21 seconds
IS_A1_B2_A2_B2_A1_A2_B2_A2_A1, status: Status.UNKNOWN, split count: 9, time: 27.21
Output dim: 7, lower bound: -14.5386979, upper bound: 14.5457363
IS_A1_B2_A2_B2_A1_A2_B2_A2_A2, status: Status.UNKNOWN, split count: 9, time: 27.21
Output dim: 7, lower bound: -14.5386568, upper bound: 14.5456522
IS_A1_B2_A2_B2_A2_A2_B2_A2_A1, status: Status.VERIFIED, split count: 9, time: 27.21
Output dim: 7, lower bound: -14.5385127, upper bound: 14.5401438
IS_A1_B2_A2_B2_A2_A2_B2_A2_A2, status: Status.UNKNOWN, split count: 9, time: 27.21
Output dim: 7, lower bound: -14.5384971, upper bound: 14.5454915

## BFS IS instance: IS_A1_B2_A2_B2_A1_A2_B2_A2_A1

### Backsubstitution after applying IS history:
0: -6.8659167, 5.6347075, -8.4509430, 6.8939638, -13.7598801, 14.0856504
1: -5.5897865, 5.0935063, -6.9073596, 6.1868591, -11.7766457, 12.0008659
2: -7.0250888, 4.4181824, -8.6496515, 5.3241568, -12.3492451, 13.0678339
3: -8.2239513, 3.9801202, -10.2123680, 4.8232064, -13.0471573, 14.1924877
4: -7.5833907, 6.1094637, -9.3257732, 7.4215822, -15.0049725, 15.4352360
5: -6.4368515, 5.0749936, -7.9474607, 6.1798878, -12.6167393, 13.0224543
6: -6.1235962, 6.6986585, -7.4831796, 8.1670208, -14.2906151, 14.1818380
7: -7.7066422, 4.4494753, -9.3798714, 5.4392133, -13.1458549, 13.8293467
8: -7.7020302, 5.5454140, -9.4497890, 6.7448263, -14.4468546, 14.9952030
9: -6.2474928, 6.3235593, -7.6332426, 7.7391896, -13.9866829, 13.9568024

### Unstable ReLU Count (Linear/Conv2D Layers)
- layer_idx=0, type=LayerType.Linear, total=256, inp1_unstable=28, inp2_unstable=31, delta_unstable=255
- layer_idx=2, type=LayerType.Linear, total=256, inp1_unstable=68, inp2_unstable=68, delta_unstable=256
- layer_idx=4, type=LayerType.Linear, total=256, inp1_unstable=21, inp2_unstable=24, delta_unstable=256
- layer_idx=6, type=LayerType.Linear, total=256, inp1_unstable=236, inp2_unstable=244, delta_unstable=256
- layer_idx=8, type=LayerType.Linear, total=10, inp1_unstable=10, inp2_unstable=10, delta_unstable=10

Time for backsubstitution: 1.24 seconds

### IS candidates at layer 1
type: B, layer: 1, pos: 144
type: B, layer: 1, pos: 181
type: A, layer: 1, pos: 181
type: B, layer: 1, pos: 132
type: B, layer: 1, pos: 184
type: B, layer: 1, pos: 71
type: B, layer: 1, pos: 89
type: A, layer: 1, pos: 71
type: A, layer: 1, pos: 195
type: B, layer: 1, pos: 195
type: B, layer: 1, pos: 1
type: B, layer: 1, pos: 77
type: B, layer: 1, pos: 219
type: B, layer: 1, pos: 197
type: A, layer: 1, pos: 146
type: A, layer: 1, pos: 77
type: B, layer: 1, pos: 85
type: A, layer: 1, pos: 184
type: A, layer: 1, pos: 185
type: A, layer: 1, pos: 197
type: A, layer: 1, pos: 89
type: A, layer: 1, pos: 132
type: A, layer: 1, pos: 155
type: B, layer: 1, pos: 179
type: A, layer: 1, pos: 204
type: A, layer: 1, pos: 85
type: B, layer: 1, pos: 232
type: B, layer: 1, pos: 140
type: B, layer: 1, pos: 185
type: B, layer: 1, pos: 107
type: B, layer: 1, pos: 146
type: A, layer: 1, pos: 1
type: A, layer: 1, pos: 219
type: A, layer: 1, pos: 179
type: B, layer: 1, pos: 123
type: A, layer: 1, pos: 35
type: A, layer: 1, pos: 123
type: A, layer: 1, pos: 128
type: B, layer: 1, pos: 171
type: A, layer: 1, pos: 15
type: A, layer: 1, pos: 138
type: B, layer: 1, pos: 35
type: B, layer: 1, pos: 251
type: B, layer: 1, pos: 128
type: B, layer: 1, pos: 204
type: A, layer: 1, pos: 171
type: B, layer: 1, pos: 138
type: A, layer: 1, pos: 107
type: A, layer: 1, pos: 97
type: B, layer: 1, pos: 104
type: A, layer: 1, pos: 233
type: B, layer: 1, pos: 15
type: A, layer: 1, pos: 104
type: B, layer: 1, pos: 97
type: A, layer: 1, pos: 251
type: B, layer: 1, pos: 155
type: B, layer: 1, pos: 233
type: B, layer: 1, pos: 57
type: A, layer: 1, pos: 57

Time for candidate selection: 0.12 seconds

### Candidate
type: B, layer: 1, pos: 144

## Relational analysis of IS_A1_B2_A2_B2_A1_A2_B2_A2_A1_B1
Optimization infeasible because this subproblem isn't reachable.

### Candidate
type: B, layer: 1, pos: 181

## Relational analysis of IS_A1_B2_A2_B2_A1_A2_B2_A2_A1_B1
Optimization infeasible because this subproblem isn't reachable.

### Candidate
type: A, layer: 1, pos: 181

## Relational analysis of IS_A1_B2_A2_B2_A1_A2_B2_A2_A1_A1
Optimization infeasible because this subproblem isn't reachable.

### Candidate
type: B, layer: 1, pos: 132

## Relational analysis of IS_A1_B2_A2_B2_A1_A2_B2_A2_A1_B1
Optimization infeasible because this subproblem isn't reachable.

### Candidate
type: B, layer: 1, pos: 184

## Relational analysis of IS_A1_B2_A2_B2_A1_A2_B2_A2_A1_B1

### Relational analysis result of IS_A1_B2_A2_B2_A1_A2_B2_A2_A1_B1
Status: Status.VERIFIED
Output dim: 7, lower bound: -14.5409450, upper bound: 14.5428131
time: 18.67 seconds

## Relational analysis of IS_A1_B2_A2_B2_A1_A2_B2_A2_A1_B2

### Relational analysis result of IS_A1_B2_A2_B2_A1_A2_B2_A2_A1_B2
Status: Status.UNKNOWN
Output dim: 7, lower bound: -14.5433436, upper bound: 14.5450581
time: 11.96 seconds

## BFS IS instance: IS_A1_B2_A2_B2_A1_A2_B2_A2_A2

### Backsubstitution after applying IS history:
0: -7.1317897, 5.8505726, -8.5700703, 6.9890771, -14.1208649, 14.4206429
1: -5.8223839, 5.2889948, -7.0078249, 6.2698736, -12.0922575, 12.2968187
2: -7.3215103, 4.5938978, -8.7733355, 5.3949366, -12.7164459, 13.3672333
3: -8.5553446, 4.1435890, -10.3622055, 4.8880501, -13.4433937, 14.5057945
4: -7.8822889, 6.3361988, -9.4558372, 7.5213122, -15.4036007, 15.7920361
5: -6.7089047, 5.2774601, -8.0617666, 6.2641191, -12.9730234, 13.3392267
6: -6.3632131, 6.9376812, -7.5866823, 8.2774925, -14.6407051, 14.5243626
7: -7.9842520, 4.6805553, -9.5063562, 5.5173182, -13.5015697, 14.1869116
8: -8.0107136, 5.7554693, -9.5818253, 6.8360929, -14.8468037, 15.3372946
9: -6.5047040, 6.5890784, -7.7394524, 7.8466926, -14.3513966, 14.3285313

### Unstable ReLU Count (Linear/Conv2D Layers)
- layer_idx=0, type=LayerType.Linear, total=256, inp1_unstable=28, inp2_unstable=31, delta_unstable=255
- layer_idx=2, type=LayerType.Linear, total=256, inp1_unstable=68, inp2_unstable=68, delta_unstable=256
- layer_idx=4, type=LayerType.Linear, total=256, inp1_unstable=23, inp2_unstable=24, delta_unstable=256
- layer_idx=6, type=LayerType.Linear, total=256, inp1_unstable=241, inp2_unstable=244, delta_unstable=256
- layer_idx=8, type=LayerType.Linear, total=10, inp1_unstable=10, inp2_unstable=10, delta_unstable=10

Time for backsubstitution: 1.26 seconds

### IS candidates at layer 1
type: B, layer: 1, pos: 144
type: B, layer: 1, pos: 181
type: A, layer: 1, pos: 181
type: A, layer: 1, pos: 195
type: B, layer: 1, pos: 195
type: B, layer: 1, pos: 71
type: A, layer: 1, pos: 71
type: B, layer: 1, pos: 132
type: B, layer: 1, pos: 184
type: A, layer: 1, pos: 185
type: B, layer: 1, pos: 89
type: B, layer: 1, pos: 232
type: B, layer: 1, pos: 1
type: A, layer: 1, pos: 132
type: B, layer: 1, pos: 185
type: A, layer: 1, pos: 184
type: B, layer: 1, pos: 77
type: A, layer: 1, pos: 89
type: A, layer: 1, pos: 77
type: A, layer: 1, pos: 1
type: A, layer: 1, pos: 155
type: B, layer: 1, pos: 219
type: B, layer: 1, pos: 85
type: B, layer: 1, pos: 197
type: A, layer: 1, pos: 204
type: A, layer: 1, pos: 197
type: A, layer: 1, pos: 85
type: A, layer: 1, pos: 146
type: A, layer: 1, pos: 35
type: A, layer: 1, pos: 219
type: A, layer: 1, pos: 128
type: B, layer: 1, pos: 155
type: B, layer: 1, pos: 35
type: B, layer: 1, pos: 204
type: B, layer: 1, pos: 128
type: B, layer: 1, pos: 140
type: B, layer: 1, pos: 146
type: B, layer: 1, pos: 179
type: A, layer: 1, pos: 15
type: A, layer: 1, pos: 179
type: B, layer: 1, pos: 171
type: B, layer: 1, pos: 15
type: A, layer: 1, pos: 138
type: B, layer: 1, pos: 138
type: B, layer: 1, pos: 107
type: A, layer: 1, pos: 171
type: A, layer: 1, pos: 107
type: B, layer: 1, pos: 123
type: A, layer: 1, pos: 123
type: B, layer: 1, pos: 104
type: A, layer: 1, pos: 104
type: B, layer: 1, pos: 251
type: A, layer: 1, pos: 251
type: A, layer: 1, pos: 233
type: B, layer: 1, pos: 97
type: A, layer: 1, pos: 97
type: B, layer: 1, pos: 233
type: B, layer: 1, pos: 57
type: A, layer: 1, pos: 57

Time for candidate selection: 0.13 seconds

### Candidate
type: B, layer: 1, pos: 144

## Relational analysis of IS_A1_B2_A2_B2_A1_A2_B2_A2_A2_B1
Optimization infeasible because this subproblem isn't reachable.

### Candidate
type: B, layer: 1, pos: 181

## Relational analysis of IS_A1_B2_A2_B2_A1_A2_B2_A2_A2_B1
Optimization infeasible because this subproblem isn't reachable.

### Candidate
type: A, layer: 1, pos: 181

## Relational analysis of IS_A1_B2_A2_B2_A1_A2_B2_A2_A2_A1
Optimization infeasible because this subproblem isn't reachable.

### Candidate
type: A, layer: 1, pos: 195

## Relational analysis of IS_A1_B2_A2_B2_A1_A2_B2_A2_A2_A1
Optimization infeasible because this subproblem isn't reachable.

### Candidate
type: B, layer: 1, pos: 195

## Relational analysis of IS_A1_B2_A2_B2_A1_A2_B2_A2_A2_B1

### Relational analysis result of IS_A1_B2_A2_B2_A1_A2_B2_A2_A2_B1
Status: Status.VERIFIED
Output dim: 7, lower bound: -14.5401165, upper bound: 14.5406666
time: 7.20 seconds

## Relational analysis of IS_A1_B2_A2_B2_A1_A2_B2_A2_A2_B2
Optimization infeasible because this subproblem isn't reachable.

### Candidate
type: B, layer: 1, pos: 71

## Relational analysis of IS_A1_B2_A2_B2_A1_A2_B2_A2_A2_B1
Optimization infeasible because this subproblem isn't reachable.

### Candidate
type: A, layer: 1, pos: 71

## Relational analysis of IS_A1_B2_A2_B2_A1_A2_B2_A2_A2_A1
Optimization infeasible because this subproblem isn't reachable.

### Candidate
type: B, layer: 1, pos: 132

## Relational analysis of IS_A1_B2_A2_B2_A1_A2_B2_A2_A2_B1
Optimization infeasible because this subproblem isn't reachable.

### Candidate
type: B, layer: 1, pos: 184

## Relational analysis of IS_A1_B2_A2_B2_A1_A2_B2_A2_A2_B1

### Relational analysis result of IS_A1_B2_A2_B2_A1_A2_B2_A2_A2_B1
Status: Status.VERIFIED
Output dim: 7, lower bound: -14.5329942, upper bound: 14.5427548
time: 7.36 seconds

## Relational analysis of IS_A1_B2_A2_B2_A1_A2_B2_A2_A2_B2

### Relational analysis result of IS_A1_B2_A2_B2_A1_A2_B2_A2_A2_B2
Status: Status.UNKNOWN
Output dim: 7, lower bound: -14.5433136, upper bound: 14.5449770
time: 4.46 seconds

## BFS IS instance: IS_A1_B2_A2_B2_A2_A2_B2_A2_A2

### Backsubstitution after applying IS history:
0: -8.1196823, 6.6319427, -8.5700703, 6.9890771, -15.1087570, 15.2020130
1: -6.6310201, 5.9606194, -7.0078249, 6.2698736, -12.9008942, 12.9684439
2: -8.3120365, 5.1268282, -8.7733355, 5.3949366, -13.7069702, 13.9001637
3: -9.8001118, 4.6463852, -10.3622055, 4.8880501, -14.6881618, 15.0085907
4: -8.9625206, 7.1468525, -9.4558372, 7.5213122, -16.4838333, 16.6026897
5: -7.6323085, 5.9508572, -8.0617666, 6.2641191, -13.8964272, 14.0126238
6: -7.1968904, 7.8611221, -7.5866823, 8.2774925, -15.4743786, 15.4478045
7: -9.0280600, 5.2278304, -9.5063562, 5.5173182, -14.5453777, 14.7341862
8: -9.0867748, 6.4937897, -9.5818253, 6.8360929, -15.9228678, 16.0756149
9: -7.3437471, 7.4460368, -7.7394524, 7.8466926, -15.1904392, 15.1854897

### Unstable ReLU Count (Linear/Conv2D Layers)
- layer_idx=0, type=LayerType.Linear, total=256, inp1_unstable=28, inp2_unstable=31, delta_unstable=256
- layer_idx=2, type=LayerType.Linear, total=256, inp1_unstable=68, inp2_unstable=68, delta_unstable=256
- layer_idx=4, type=LayerType.Linear, total=256, inp1_unstable=23, inp2_unstable=24, delta_unstable=256
- layer_idx=6, type=LayerType.Linear, total=256, inp1_unstable=242, inp2_unstable=244, delta_unstable=256
- layer_idx=8, type=LayerType.Linear, total=10, inp1_unstable=10, inp2_unstable=10, delta_unstable=10

Time for backsubstitution: 1.25 seconds

### IS candidates at layer 1
type: B, layer: 1, pos: 144
type: B, layer: 1, pos: 181
type: A, layer: 1, pos: 181
type: B, layer: 1, pos: 195
type: A, layer: 1, pos: 195
type: B, layer: 1, pos: 71
type: A, layer: 1, pos: 71
type: B, layer: 1, pos: 132
type: B, layer: 1, pos: 89
type: B, layer: 1, pos: 197
type: A, layer: 1, pos: 197
type: B, layer: 1, pos: 184
type: A, layer: 1, pos: 155
type: B, layer: 1, pos: 77
type: A, layer: 1, pos: 89
type: A, layer: 1, pos: 132
type: B, layer: 1, pos: 155
type: A, layer: 1, pos: 85
type: B, layer: 1, pos: 232
type: B, layer: 1, pos: 1
type: B, layer: 1, pos: 185
type: A, layer: 1, pos: 77
type: B, layer: 1, pos: 85
type: B, layer: 1, pos: 219
type: A, layer: 1, pos: 1
type: A, layer: 1, pos: 184
type: A, layer: 1, pos: 185
type: A, layer: 1, pos: 146
type: A, layer: 1, pos: 219
type: B, layer: 1, pos: 140
type: A, layer: 1, pos: 204
type: B, layer: 1, pos: 171
type: B, layer: 1, pos: 138
type: A, layer: 1, pos: 35
type: B, layer: 1, pos: 146
type: B, layer: 1, pos: 35
type: B, layer: 1, pos: 179
type: A, layer: 1, pos: 138
type: B, layer: 1, pos: 123
type: A, layer: 1, pos: 123
type: B, layer: 1, pos: 204
type: B, layer: 1, pos: 15
type: A, layer: 1, pos: 171
type: A, layer: 1, pos: 179
type: A, layer: 1, pos: 15
type: A, layer: 1, pos: 128
type: B, layer: 1, pos: 128
type: B, layer: 1, pos: 104
type: B, layer: 1, pos: 233
type: A, layer: 1, pos: 233
type: A, layer: 1, pos: 251
type: B, layer: 1, pos: 251
type: A, layer: 1, pos: 104
type: A, layer: 1, pos: 107
type: B, layer: 1, pos: 107
type: A, layer: 1, pos: 97
type: B, layer: 1, pos: 97
type: B, layer: 1, pos: 57
type: A, layer: 1, pos: 57

Time for candidate selection: 0.13 seconds

### Candidate
type: B, layer: 1, pos: 144

## Relational analysis of IS_A1_B2_A2_B2_A2_A2_B2_A2_A2_B1
Optimization infeasible because this subproblem isn't reachable.

### Candidate
type: B, layer: 1, pos: 181

## Relational analysis of IS_A1_B2_A2_B2_A2_A2_B2_A2_A2_B1
Optimization infeasible because this subproblem isn't reachable.

### Candidate
type: A, layer: 1, pos: 181

## Relational analysis of IS_A1_B2_A2_B2_A2_A2_B2_A2_A2_A1
Optimization infeasible because this subproblem isn't reachable.

### Candidate
type: B, layer: 1, pos: 195

## Relational analysis of IS_A1_B2_A2_B2_A2_A2_B2_A2_A2_B1

### Relational analysis result of IS_A1_B2_A2_B2_A2_A2_B2_A2_A2_B1
Status: Status.VERIFIED
Output dim: 7, lower bound: -14.5398899, upper bound: 14.5403373
time: 4.09 seconds

## Relational analysis of IS_A1_B2_A2_B2_A2_A2_B2_A2_A2_B2
Optimization infeasible because this subproblem isn't reachable.

### Candidate
type: A, layer: 1, pos: 195

## Relational analysis of IS_A1_B2_A2_B2_A2_A2_B2_A2_A2_A1
Optimization infeasible because this subproblem isn't reachable.

### Candidate
type: B, layer: 1, pos: 71

## Relational analysis of IS_A1_B2_A2_B2_A2_A2_B2_A2_A2_B1
Optimization infeasible because this subproblem isn't reachable.

### Candidate
type: A, layer: 1, pos: 71

## Relational analysis of IS_A1_B2_A2_B2_A2_A2_B2_A2_A2_A1
Optimization infeasible because this subproblem isn't reachable.

### Candidate
type: B, layer: 1, pos: 132

## Relational analysis of IS_A1_B2_A2_B2_A2_A2_B2_A2_A2_B1
Optimization infeasible because this subproblem isn't reachable.

### Candidate
type: B, layer: 1, pos: 89

## Relational analysis of IS_A1_B2_A2_B2_A2_A2_B2_A2_A2_B1

### Relational analysis result of IS_A1_B2_A2_B2_A2_A2_B2_A2_A2_B1
Status: Status.VERIFIED
Output dim: 7, lower bound: -14.5430605, upper bound: 14.5447189
time: 7.97 seconds

## Relational analysis of IS_A1_B2_A2_B2_A2_A2_B2_A2_A2_B2

### Relational analysis result of IS_A1_B2_A2_B2_A2_A2_B2_A2_A2_B2
Status: Status.UNKNOWN
Output dim: 7, lower bound: -14.5430921, upper bound: 14.5447826
time: 4.86 seconds

## Summary of splitting at layer (split count: 9)
- Time for IS candidates: 38.29 seconds
IS_A1_B2_A2_B2_A1_A2_B2_A2_A1_B1, status: Status.VERIFIED, split count: 10, time: 38.29
Output dim: 7, lower bound: -14.5409450, upper bound: 14.5428131
IS_A1_B2_A2_B2_A1_A2_B2_A2_A1_B2, status: Status.UNKNOWN, split count: 10, time: 38.29
Output dim: 7, lower bound: -14.5433436, upper bound: 14.5450581
IS_A1_B2_A2_B2_A1_A2_B2_A2_A2_B1, status: Status.VERIFIED, split count: 10, time: 38.29
Output dim: 7, lower bound: -14.5329942, upper bound: 14.5427548
IS_A1_B2_A2_B2_A1_A2_B2_A2_A2_B2, status: Status.UNKNOWN, split count: 10, time: 38.29
Output dim: 7, lower bound: -14.5433136, upper bound: 14.5449770
IS_A1_B2_A2_B2_A2_A2_B2_A2_A2_B1, status: Status.VERIFIED, split count: 10, time: 38.29
Output dim: 7, lower bound: -14.5430605, upper bound: 14.5447189
IS_A1_B2_A2_B2_A2_A2_B2_A2_A2_B2, status: Status.UNKNOWN, split count: 10, time: 38.29
Output dim: 7, lower bound: -14.5430921, upper bound: 14.5447826

## BFS IS instance: IS_A1_B2_A2_B2_A1_A2_B2_A2_A1_B2

### Backsubstitution after applying IS history:
0: -6.6526551, 5.4628062, -7.8960209, 6.4446712, -13.0973263, 13.3588276
1: -5.4030862, 4.9388170, -6.4203687, 5.7895098, -11.1925964, 11.3591862
2: -6.7976270, 4.2804070, -8.0586700, 4.9631824, -11.7608089, 12.3390751
3: -7.9554315, 3.8622003, -9.5109596, 4.5125647, -12.4679947, 13.3731594
4: -7.3466687, 5.9251680, -8.7133532, 6.9418459, -14.2885151, 14.6385183
5: -6.2267241, 4.9202089, -7.4040031, 5.7764273, -12.0031509, 12.3242121
6: -5.9307203, 6.5027657, -6.9859581, 7.6583843, -13.5891047, 13.4887238
7: -7.4799123, 4.2920694, -8.7906656, 5.0263891, -12.5063019, 13.0827351
8: -7.4585133, 5.3778181, -8.8181734, 6.3088531, -13.7673664, 14.1959906
9: -6.0475664, 6.1226707, -7.1115284, 7.2143545, -13.2619209, 13.2341995

### Unstable ReLU Count (Linear/Conv2D Layers)
- layer_idx=0, type=LayerType.Linear, total=256, inp1_unstable=28, inp2_unstable=30, delta_unstable=255
- layer_idx=2, type=LayerType.Linear, total=256, inp1_unstable=68, inp2_unstable=68, delta_unstable=256
- layer_idx=4, type=LayerType.Linear, total=256, inp1_unstable=20, inp2_unstable=22, delta_unstable=256
- layer_idx=6, type=LayerType.Linear, total=256, inp1_unstable=232, inp2_unstable=242, delta_unstable=256
- layer_idx=8, type=LayerType.Linear, total=10, inp1_unstable=10, inp2_unstable=10, delta_unstable=10

Time for backsubstitution: 1.25 seconds

### IS candidates at layer 1
type: B, layer: 1, pos: 144
type: B, layer: 1, pos: 181
type: A, layer: 1, pos: 181
type: A, layer: 1, pos: 184
type: B, layer: 1, pos: 71
type: A, layer: 1, pos: 71
type: B, layer: 1, pos: 132
type: A, layer: 1, pos: 195
type: B, layer: 1, pos: 89
type: B, layer: 1, pos: 195
type: B, layer: 1, pos: 1
type: B, layer: 1, pos: 77
type: B, layer: 1, pos: 197
type: A, layer: 1, pos: 132
type: A, layer: 1, pos: 77
type: A, layer: 1, pos: 89
type: B, layer: 1, pos: 219
type: A, layer: 1, pos: 197
type: A, layer: 1, pos: 146
type: B, layer: 1, pos: 85
type: A, layer: 1, pos: 185
type: B, layer: 1, pos: 232
type: A, layer: 1, pos: 85
type: B, layer: 1, pos: 146
type: A, layer: 1, pos: 1
type: A, layer: 1, pos: 219
type: A, layer: 1, pos: 155
type: B, layer: 1, pos: 140
type: B, layer: 1, pos: 185
type: A, layer: 1, pos: 204
type: B, layer: 1, pos: 179
type: B, layer: 1, pos: 107
type: B, layer: 1, pos: 123
type: A, layer: 1, pos: 179
type: A, layer: 1, pos: 123
type: A, layer: 1, pos: 35
type: B, layer: 1, pos: 204
type: A, layer: 1, pos: 15
type: A, layer: 1, pos: 128
type: B, layer: 1, pos: 35
type: A, layer: 1, pos: 171
type: B, layer: 1, pos: 128
type: A, layer: 1, pos: 138
type: B, layer: 1, pos: 251
type: B, layer: 1, pos: 171
type: B, layer: 1, pos: 138
type: A, layer: 1, pos: 107
type: A, layer: 1, pos: 97
type: A, layer: 1, pos: 104
type: B, layer: 1, pos: 97
type: A, layer: 1, pos: 233
type: B, layer: 1, pos: 15
type: A, layer: 1, pos: 251
type: B, layer: 1, pos: 104
type: B, layer: 1, pos: 155
type: B, layer: 1, pos: 233
type: B, layer: 1, pos: 57
type: A, layer: 1, pos: 57

Time for candidate selection: 0.13 seconds

### Candidate
type: B, layer: 1, pos: 144

## Relational analysis of IS_A1_B2_A2_B2_A1_A2_B2_A2_A1_B2_B1
Optimization infeasible because this subproblem isn't reachable.

### Candidate
type: B, layer: 1, pos: 181

## Relational analysis of IS_A1_B2_A2_B2_A1_A2_B2_A2_A1_B2_B1
Optimization infeasible because this subproblem isn't reachable.

### Candidate
type: A, layer: 1, pos: 181

## Relational analysis of IS_A1_B2_A2_B2_A1_A2_B2_A2_A1_B2_A1
Optimization infeasible because this subproblem isn't reachable.

### Candidate
type: A, layer: 1, pos: 184

## Relational analysis of IS_A1_B2_A2_B2_A1_A2_B2_A2_A1_B2_A1

### Relational analysis result of IS_A1_B2_A2_B2_A1_A2_B2_A2_A1_B2_A1
Status: Status.VERIFIED
Output dim: 7, lower bound: -14.5427067, upper bound: 14.5442975
time: 3.73 seconds

## Relational analysis of IS_A1_B2_A2_B2_A1_A2_B2_A2_A1_B2_A2

### Relational analysis result of IS_A1_B2_A2_B2_A1_A2_B2_A2_A1_B2_A2
Status: Status.UNKNOWN
Output dim: 7, lower bound: -14.5427067, upper bound: 14.5450581
time: 7.68 seconds

## BFS IS instance: IS_A1_B2_A2_B2_A1_A2_B2_A2_A2_B2

### Backsubstitution after applying IS history:
0: -6.9080863, 5.6699915, -8.0131979, 6.5383091, -13.4463959, 13.6831894
1: -5.6269679, 5.1291962, -6.5193958, 5.8717031, -11.4986706, 11.6485920
2: -7.0818996, 4.4462881, -8.1806698, 5.0318294, -12.1137295, 12.6269579
3: -8.2758093, 4.0190673, -9.6585207, 4.5766225, -12.8524323, 13.6775875
4: -7.6363468, 6.1437159, -8.8414698, 7.0402012, -14.6765461, 14.9851837
5: -6.4903955, 5.1147499, -7.5165310, 5.8594084, -12.3498039, 12.6312809
6: -6.1634111, 6.7340908, -7.0877342, 7.7673492, -13.9307594, 13.8218241
7: -7.7474074, 4.5120502, -8.9156342, 5.1025314, -12.8499393, 13.4276848
8: -7.7560544, 5.5805402, -8.9481907, 6.3982639, -14.1543179, 14.5287304
9: -6.2955155, 6.3768950, -7.2161489, 7.3201971, -13.6157131, 13.5930433

### Unstable ReLU Count (Linear/Conv2D Layers)
- layer_idx=0, type=LayerType.Linear, total=256, inp1_unstable=28, inp2_unstable=30, delta_unstable=255
- layer_idx=2, type=LayerType.Linear, total=256, inp1_unstable=68, inp2_unstable=68, delta_unstable=256
- layer_idx=4, type=LayerType.Linear, total=256, inp1_unstable=22, inp2_unstable=22, delta_unstable=256
- layer_idx=6, type=LayerType.Linear, total=256, inp1_unstable=238, inp2_unstable=242, delta_unstable=256
- layer_idx=8, type=LayerType.Linear, total=10, inp1_unstable=10, inp2_unstable=10, delta_unstable=10

Time for backsubstitution: 1.26 seconds

### IS candidates at layer 1
type: B, layer: 1, pos: 144
type: A, layer: 1, pos: 181
type: B, layer: 1, pos: 181
type: A, layer: 1, pos: 184
type: A, layer: 1, pos: 71
type: B, layer: 1, pos: 71
type: A, layer: 1, pos: 195
type: B, layer: 1, pos: 195
type: A, layer: 1, pos: 132
type: B, layer: 1, pos: 132
type: A, layer: 1, pos: 89
type: B, layer: 1, pos: 89
type: A, layer: 1, pos: 77
type: B, layer: 1, pos: 232
type: B, layer: 1, pos: 77
type: A, layer: 1, pos: 197
type: B, layer: 1, pos: 197
type: B, layer: 1, pos: 1
type: A, layer: 1, pos: 219
type: A, layer: 1, pos: 1
type: B, layer: 1, pos: 85
type: B, layer: 1, pos: 146
type: B, layer: 1, pos: 219
type: A, layer: 1, pos: 146
type: A, layer: 1, pos: 85
type: B, layer: 1, pos: 185
type: A, layer: 1, pos: 185
type: B, layer: 1, pos: 140
type: A, layer: 1, pos: 179
type: B, layer: 1, pos: 179
type: B, layer: 1, pos: 204
type: A, layer: 1, pos: 204
type: B, layer: 1, pos: 107
type: A, layer: 1, pos: 123
type: B, layer: 1, pos: 123
type: B, layer: 1, pos: 35
type: A, layer: 1, pos: 155
type: A, layer: 1, pos: 35
type: B, layer: 1, pos: 128
type: A, layer: 1, pos: 107
type: A, layer: 1, pos: 128
type: A, layer: 1, pos: 171
type: B, layer: 1, pos: 171
type: A, layer: 1, pos: 15
type: A, layer: 1, pos: 138
type: B, layer: 1, pos: 138
type: B, layer: 1, pos: 155
type: B, layer: 1, pos: 15
type: B, layer: 1, pos: 97
type: A, layer: 1, pos: 251
type: B, layer: 1, pos: 251
type: A, layer: 1, pos: 104
type: A, layer: 1, pos: 97
type: B, layer: 1, pos: 104
type: B, layer: 1, pos: 233
type: A, layer: 1, pos: 233
type: B, layer: 1, pos: 57
type: A, layer: 1, pos: 57

Time for candidate selection: 0.12 seconds

### Candidate
type: B, layer: 1, pos: 144

## Relational analysis of IS_A1_B2_A2_B2_A1_A2_B2_A2_A2_B2_B1
Optimization infeasible because this subproblem isn't reachable.

### Candidate
type: A, layer: 1, pos: 181

## Relational analysis of IS_A1_B2_A2_B2_A1_A2_B2_A2_A2_B2_A1
Optimization infeasible because this subproblem isn't reachable.

### Candidate
type: B, layer: 1, pos: 181

## Relational analysis of IS_A1_B2_A2_B2_A1_A2_B2_A2_A2_B2_B1
Optimization infeasible because this subproblem isn't reachable.

### Candidate
type: A, layer: 1, pos: 184

## Relational analysis of IS_A1_B2_A2_B2_A1_A2_B2_A2_A2_B2_A1

### Relational analysis result of IS_A1_B2_A2_B2_A1_A2_B2_A2_A2_B2_A1
Status: Status.VERIFIED
Output dim: 7, lower bound: -14.5426790, upper bound: 14.5441800
time: 4.58 seconds

## Relational analysis of IS_A1_B2_A2_B2_A1_A2_B2_A2_A2_B2_A2

### Relational analysis result of IS_A1_B2_A2_B2_A1_A2_B2_A2_A2_B2_A2
Status: Status.UNKNOWN
Output dim: 7, lower bound: -14.5426790, upper bound: 14.5449770
time: 5.00 seconds

## BFS IS instance: IS_A1_B2_A2_B2_A2_A2_B2_A2_A2_B2

### Backsubstitution after applying IS history:
0: -7.8359609, 6.4016929, -8.4182262, 6.8561401, -14.6921005, 14.8199196
1: -6.3856044, 5.7612543, -6.8625107, 6.1539650, -12.5395699, 12.6237640
2: -8.0061283, 4.9374433, -8.5784225, 5.2430000, -13.2491264, 13.5158634
3: -9.4479027, 4.4813070, -10.1725426, 4.7672505, -14.2151527, 14.6538496
4: -8.6449108, 6.9004855, -9.2705469, 7.3704705, -16.0153790, 16.1710300
5: -7.3496099, 5.7409248, -7.8867402, 6.1248312, -13.4744415, 13.6276646
6: -6.9440899, 7.6001506, -7.4344378, 8.1390123, -15.0830975, 15.0345860
7: -8.7233028, 4.9993510, -9.3286066, 5.2954950, -14.0187969, 14.3279572
8: -8.7660456, 6.2669983, -9.3940983, 6.6904812, -15.4565258, 15.6610966
9: -7.0694971, 7.1663208, -7.5392866, 7.6443014, -14.7137976, 14.7056074

### Unstable ReLU Count (Linear/Conv2D Layers)
- layer_idx=0, type=LayerType.Linear, total=256, inp1_unstable=28, inp2_unstable=30, delta_unstable=256
- layer_idx=2, type=LayerType.Linear, total=256, inp1_unstable=68, inp2_unstable=70, delta_unstable=256
- layer_idx=4, type=LayerType.Linear, total=256, inp1_unstable=21, inp2_unstable=23, delta_unstable=256
- layer_idx=6, type=LayerType.Linear, total=256, inp1_unstable=242, inp2_unstable=242, delta_unstable=256
- layer_idx=8, type=LayerType.Linear, total=10, inp1_unstable=10, inp2_unstable=10, delta_unstable=10

Time for backsubstitution: 1.26 seconds

### IS candidates at layer 1
type: B, layer: 1, pos: 144
type: A, layer: 1, pos: 181
type: B, layer: 1, pos: 181
type: A, layer: 1, pos: 89
type: A, layer: 1, pos: 71
type: B, layer: 1, pos: 71
type: A, layer: 1, pos: 132
type: B, layer: 1, pos: 195
type: A, layer: 1, pos: 195
type: A, layer: 1, pos: 184
type: B, layer: 1, pos: 184
type: B, layer: 1, pos: 232
type: B, layer: 1, pos: 77
type: A, layer: 1, pos: 77
type: A, layer: 1, pos: 1
type: A, layer: 1, pos: 197
type: B, layer: 1, pos: 132
type: A, layer: 1, pos: 219
type: A, layer: 1, pos: 85
type: B, layer: 1, pos: 146
type: B, layer: 1, pos: 197
type: B, layer: 1, pos: 185
type: B, layer: 1, pos: 1
type: A, layer: 1, pos: 146
type: A, layer: 1, pos: 179
type: B, layer: 1, pos: 140
type: B, layer: 1, pos: 219
type: B, layer: 1, pos: 85
type: A, layer: 1, pos: 185
type: B, layer: 1, pos: 155
type: B, layer: 1, pos: 204
type: A, layer: 1, pos: 107
type: B, layer: 1, pos: 179
type: A, layer: 1, pos: 123
type: B, layer: 1, pos: 123
type: A, layer: 1, pos: 204
type: A, layer: 1, pos: 35
type: B, layer: 1, pos: 35
type: B, layer: 1, pos: 15
type: B, layer: 1, pos: 128
type: A, layer: 1, pos: 128
type: A, layer: 1, pos: 251
type: B, layer: 1, pos: 171
type: B, layer: 1, pos: 138
type: A, layer: 1, pos: 138
type: B, layer: 1, pos: 107
type: A, layer: 1, pos: 171
type: B, layer: 1, pos: 97
type: A, layer: 1, pos: 15
type: B, layer: 1, pos: 233
type: A, layer: 1, pos: 155
type: B, layer: 1, pos: 104
type: A, layer: 1, pos: 97
type: A, layer: 1, pos: 104
type: B, layer: 1, pos: 251
type: A, layer: 1, pos: 233
type: A, layer: 1, pos: 57
type: B, layer: 1, pos: 57

Time for candidate selection: 0.12 seconds

### Candidate
type: B, layer: 1, pos: 144

## Relational analysis of IS_A1_B2_A2_B2_A2_A2_B2_A2_A2_B2_B1
Optimization infeasible because this subproblem isn't reachable.

### Candidate
type: A, layer: 1, pos: 181

## Relational analysis of IS_A1_B2_A2_B2_A2_A2_B2_A2_A2_B2_A1
Optimization infeasible because this subproblem isn't reachable.

### Candidate
type: B, layer: 1, pos: 181

## Relational analysis of IS_A1_B2_A2_B2_A2_A2_B2_A2_A2_B2_B1
Optimization infeasible because this subproblem isn't reachable.

### Candidate
type: A, layer: 1, pos: 89

## Relational analysis of IS_A1_B2_A2_B2_A2_A2_B2_A2_A2_B2_A1

### Relational analysis result of IS_A1_B2_A2_B2_A2_A2_B2_A2_A2_B2_A1
Status: Status.UNKNOWN
Output dim: 7, lower bound: -14.5430867, upper bound: 14.5447810
time: 10.33 seconds

## Relational analysis of IS_A1_B2_A2_B2_A2_A2_B2_A2_A2_B2_A2

### Relational analysis result of IS_A1_B2_A2_B2_A2_A2_B2_A2_A2_B2_A2
Status: Status.UNKNOWN
Output dim: 7, lower bound: -14.5430867, upper bound: 14.5447826
time: 5.97 seconds

## Summary of splitting at layer (split count: 10)
- Time for IS candidates: 25.24 seconds
IS_A1_B2_A2_B2_A1_A2_B2_A2_A1_B2_A1, status: Status.VERIFIED, split count: 11, time: 25.24
Output dim: 7, lower bound: -14.5427067, upper bound: 14.5442975
IS_A1_B2_A2_B2_A1_A2_B2_A2_A1_B2_A2, status: Status.UNKNOWN, split count: 11, time: 25.24
Output dim: 7, lower bound: -14.5427067, upper bound: 14.5450581
IS_A1_B2_A2_B2_A1_A2_B2_A2_A2_B2_A1, status: Status.VERIFIED, split count: 11, time: 25.24
Output dim: 7, lower bound: -14.5426790, upper bound: 14.5441800
IS_A1_B2_A2_B2_A1_A2_B2_A2_A2_B2_A2, status: Status.UNKNOWN, split count: 11, time: 25.24
Output dim: 7, lower bound: -14.5426790, upper bound: 14.5449770
IS_A1_B2_A2_B2_A2_A2_B2_A2_A2_B2_A1, status: Status.UNKNOWN, split count: 11, time: 25.24
Output dim: 7, lower bound: -14.5430867, upper bound: 14.5447810
IS_A1_B2_A2_B2_A2_A2_B2_A2_A2_B2_A2, status: Status.UNKNOWN, split count: 11, time: 25.24
Output dim: 7, lower bound: -14.5430867, upper bound: 14.5447826

## BFS IS instance: IS_A1_B2_A2_B2_A1_A2_B2_A2_A1_B2_A2

### Backsubstitution after applying IS history:
0: -6.3102894, 5.1874495, -7.8960209, 6.4446712, -12.7549610, 13.0834703
1: -5.1038389, 4.6920466, -6.4203687, 5.7895098, -10.8933487, 11.1124153
2: -6.4331913, 4.0593624, -8.0586700, 4.9631824, -11.3963737, 12.1180325
3: -7.5258441, 3.6737804, -9.5109596, 4.5125647, -12.0384083, 13.1847401
4: -6.9661217, 5.6293097, -8.7133532, 6.9418459, -13.9079676, 14.3426619
5: -5.8903370, 4.6729784, -7.4040031, 5.7764273, -11.6667643, 12.0769806
6: -5.6210299, 6.1881032, -6.9859581, 7.6583843, -13.2794142, 13.1740608
7: -7.1146784, 4.0424051, -8.7906656, 5.0263891, -12.1410675, 12.8330708
8: -7.0677104, 5.1102281, -8.8181734, 6.3088531, -13.3765640, 13.9283991
9: -5.7280288, 5.8010807, -7.1115284, 7.2143545, -12.9423819, 12.9126091

### Unstable ReLU Count (Linear/Conv2D Layers)
- layer_idx=0, type=LayerType.Linear, total=256, inp1_unstable=27, inp2_unstable=30, delta_unstable=255
- layer_idx=2, type=LayerType.Linear, total=256, inp1_unstable=68, inp2_unstable=68, delta_unstable=256
- layer_idx=4, type=LayerType.Linear, total=256, inp1_unstable=18, inp2_unstable=22, delta_unstable=256
- layer_idx=6, type=LayerType.Linear, total=256, inp1_unstable=230, inp2_unstable=242, delta_unstable=256
- layer_idx=8, type=LayerType.Linear, total=10, inp1_unstable=10, inp2_unstable=10, delta_unstable=10

Time for backsubstitution: 1.28 seconds

### IS candidates at layer 1
type: B, layer: 1, pos: 144
type: B, layer: 1, pos: 181
type: A, layer: 1, pos: 181
type: B, layer: 1, pos: 132
type: B, layer: 1, pos: 71
type: B, layer: 1, pos: 89
type: A, layer: 1, pos: 71
type: A, layer: 1, pos: 195
type: B, layer: 1, pos: 195
type: B, layer: 1, pos: 1
type: B, layer: 1, pos: 219
type: B, layer: 1, pos: 77
type: A, layer: 1, pos: 146
type: B, layer: 1, pos: 197
type: A, layer: 1, pos: 77
type: A, layer: 1, pos: 89
type: B, layer: 1, pos: 85
type: A, layer: 1, pos: 197
type: A, layer: 1, pos: 132
type: A, layer: 1, pos: 185
type: A, layer: 1, pos: 204
type: B, layer: 1, pos: 232
type: B, layer: 1, pos: 179
type: A, layer: 1, pos: 85
type: B, layer: 1, pos: 107
type: A, layer: 1, pos: 155
type: B, layer: 1, pos: 146
type: B, layer: 1, pos: 140
type: A, layer: 1, pos: 1
type: A, layer: 1, pos: 219
type: B, layer: 1, pos: 185
type: A, layer: 1, pos: 179
type: B, layer: 1, pos: 123
type: B, layer: 1, pos: 251
type: A, layer: 1, pos: 123
type: A, layer: 1, pos: 35
type: A, layer: 1, pos: 128
type: B, layer: 1, pos: 204
type: B, layer: 1, pos: 171
type: A, layer: 1, pos: 15
type: A, layer: 1, pos: 138
type: B, layer: 1, pos: 35
type: A, layer: 1, pos: 107
type: A, layer: 1, pos: 97
type: B, layer: 1, pos: 128
type: B, layer: 1, pos: 138
type: A, layer: 1, pos: 233
type: A, layer: 1, pos: 171
type: B, layer: 1, pos: 104
type: B, layer: 1, pos: 97
type: A, layer: 1, pos: 251
type: B, layer: 1, pos: 15
type: A, layer: 1, pos: 104
type: B, layer: 1, pos: 155
type: B, layer: 1, pos: 233
type: B, layer: 1, pos: 57
type: A, layer: 1, pos: 57

Time for candidate selection: 0.14 seconds

### Candidate
type: B, layer: 1, pos: 144

## Relational analysis of IS_A1_B2_A2_B2_A1_A2_B2_A2_A1_B2_A2_B1
Optimization infeasible because this subproblem isn't reachable.

### Candidate
type: B, layer: 1, pos: 181

## Relational analysis of IS_A1_B2_A2_B2_A1_A2_B2_A2_A1_B2_A2_B1
Optimization infeasible because this subproblem isn't reachable.

### Candidate
type: A, layer: 1, pos: 181

## Relational analysis of IS_A1_B2_A2_B2_A1_A2_B2_A2_A1_B2_A2_A1
Optimization infeasible because this subproblem isn't reachable.

### Candidate
type: B, layer: 1, pos: 132

## Relational analysis of IS_A1_B2_A2_B2_A1_A2_B2_A2_A1_B2_A2_B1
Optimization infeasible because this subproblem isn't reachable.

### Candidate
type: B, layer: 1, pos: 71

## Relational analysis of IS_A1_B2_A2_B2_A1_A2_B2_A2_A1_B2_A2_B1
Optimization infeasible because this subproblem isn't reachable.

### Candidate
type: B, layer: 1, pos: 89

## Relational analysis of IS_A1_B2_A2_B2_A1_A2_B2_A2_A1_B2_A2_B1

### Relational analysis result of IS_A1_B2_A2_B2_A1_A2_B2_A2_A1_B2_A2_B1
Status: Status.VERIFIED
Output dim: 7, lower bound: -14.5398835, upper bound: 14.5442169
time: 29.61 seconds

## Relational analysis of IS_A1_B2_A2_B2_A1_A2_B2_A2_A1_B2_A2_B2

### Relational analysis result of IS_A1_B2_A2_B2_A1_A2_B2_A2_A1_B2_A2_B2
Status: Status.VERIFIED
Output dim: 7, lower bound: -14.5401128, upper bound: 14.5420575
time: 3.21 seconds

## BFS IS instance: IS_A1_B2_A2_B2_A1_A2_B2_A2_A2_B2_A2

### Backsubstitution after applying IS history:
0: -6.5552301, 5.3844128, -8.0131979, 6.5383091, -13.0935392, 13.3976107
1: -5.3177767, 4.8735175, -6.5193958, 5.8717031, -11.1894798, 11.3929119
2: -6.7048883, 4.2179413, -8.1806698, 5.0318294, -11.7367172, 12.3986111
3: -7.8311448, 3.8232615, -9.6585207, 4.5766225, -12.4077673, 13.4817810
4: -7.2455878, 5.8382220, -8.8414698, 7.0402012, -14.2857876, 14.6796875
5: -6.1436458, 4.8573656, -7.5165310, 5.8594084, -12.0030537, 12.3738956
6: -5.8453541, 6.4107943, -7.0877342, 7.7673492, -13.6127033, 13.4985275
7: -7.3721647, 4.2512555, -8.9156342, 5.1025314, -12.4746962, 13.1668892
8: -7.3532114, 5.3029232, -8.9481907, 6.3982639, -13.7514734, 14.2511139
9: -5.9647408, 6.0439863, -7.2161489, 7.3201971, -13.2849379, 13.2601328

### Unstable ReLU Count (Linear/Conv2D Layers)
- layer_idx=0, type=LayerType.Linear, total=256, inp1_unstable=27, inp2_unstable=30, delta_unstable=255
- layer_idx=2, type=LayerType.Linear, total=256, inp1_unstable=68, inp2_unstable=68, delta_unstable=256
- layer_idx=4, type=LayerType.Linear, total=256, inp1_unstable=21, inp2_unstable=22, delta_unstable=256
- layer_idx=6, type=LayerType.Linear, total=256, inp1_unstable=232, inp2_unstable=242, delta_unstable=256
- layer_idx=8, type=LayerType.Linear, total=10, inp1_unstable=10, inp2_unstable=10, delta_unstable=10

Time for backsubstitution: 1.25 seconds

### IS candidates at layer 1
type: B, layer: 1, pos: 144
type: B, layer: 1, pos: 181
type: A, layer: 1, pos: 181
type: B, layer: 1, pos: 71
type: A, layer: 1, pos: 71
type: A, layer: 1, pos: 195
type: B, layer: 1, pos: 132
type: B, layer: 1, pos: 195
type: B, layer: 1, pos: 89
type: A, layer: 1, pos: 132
type: B, layer: 1, pos: 77
type: A, layer: 1, pos: 89
type: A, layer: 1, pos: 77
type: B, layer: 1, pos: 232
type: B, layer: 1, pos: 197
type: B, layer: 1, pos: 1
type: A, layer: 1, pos: 197
type: B, layer: 1, pos: 219
type: A, layer: 1, pos: 146
type: B, layer: 1, pos: 85
type: A, layer: 1, pos: 185
type: A, layer: 1, pos: 1
type: A, layer: 1, pos: 85
type: A, layer: 1, pos: 219
type: B, layer: 1, pos: 146
type: B, layer: 1, pos: 185
type: B, layer: 1, pos: 140
type: A, layer: 1, pos: 204
type: B, layer: 1, pos: 179
type: A, layer: 1, pos: 155
type: A, layer: 1, pos: 179
type: B, layer: 1, pos: 107
type: B, layer: 1, pos: 123
type: A, layer: 1, pos: 123
type: B, layer: 1, pos: 171
type: B, layer: 1, pos: 204
type: A, layer: 1, pos: 35
type: B, layer: 1, pos: 35
type: A, layer: 1, pos: 128
type: B, layer: 1, pos: 128
type: A, layer: 1, pos: 138
type: A, layer: 1, pos: 107
type: A, layer: 1, pos: 15
type: B, layer: 1, pos: 138
type: B, layer: 1, pos: 15
type: A, layer: 1, pos: 171
type: B, layer: 1, pos: 155
type: B, layer: 1, pos: 251
type: A, layer: 1, pos: 251
type: B, layer: 1, pos: 97
type: A, layer: 1, pos: 97
type: B, layer: 1, pos: 104
type: A, layer: 1, pos: 233
type: A, layer: 1, pos: 104
type: B, layer: 1, pos: 233
type: B, layer: 1, pos: 57
type: A, layer: 1, pos: 57

Time for candidate selection: 0.13 seconds

### Candidate
type: B, layer: 1, pos: 144

## Relational analysis of IS_A1_B2_A2_B2_A1_A2_B2_A2_A2_B2_A2_B1
Optimization infeasible because this subproblem isn't reachable.

### Candidate
type: B, layer: 1, pos: 181

## Relational analysis of IS_A1_B2_A2_B2_A1_A2_B2_A2_A2_B2_A2_B1
Optimization infeasible because this subproblem isn't reachable.

### Candidate
type: A, layer: 1, pos: 181

## Relational analysis of IS_A1_B2_A2_B2_A1_A2_B2_A2_A2_B2_A2_A1
Optimization infeasible because this subproblem isn't reachable.

### Candidate
type: B, layer: 1, pos: 71

## Relational analysis of IS_A1_B2_A2_B2_A1_A2_B2_A2_A2_B2_A2_B1
Optimization infeasible because this subproblem isn't reachable.

### Candidate
type: A, layer: 1, pos: 71

## Relational analysis of IS_A1_B2_A2_B2_A1_A2_B2_A2_A2_B2_A2_A1
Optimization infeasible because this subproblem isn't reachable.

### Candidate
type: A, layer: 1, pos: 195

## Relational analysis of IS_A1_B2_A2_B2_A1_A2_B2_A2_A2_B2_A2_A1
Optimization infeasible because this subproblem isn't reachable.

### Candidate
type: B, layer: 1, pos: 132

## Relational analysis of IS_A1_B2_A2_B2_A1_A2_B2_A2_A2_B2_A2_B1
Optimization infeasible because this subproblem isn't reachable.

### Candidate
type: B, layer: 1, pos: 195

## Relational analysis of IS_A1_B2_A2_B2_A1_A2_B2_A2_A2_B2_A2_B1

### Relational analysis result of IS_A1_B2_A2_B2_A1_A2_B2_A2_A2_B2_A2_B1
Status: Status.VERIFIED
Output dim: 7, lower bound: -14.5328176, upper bound: 14.5398874
time: 5.29 seconds

## Relational analysis of IS_A1_B2_A2_B2_A1_A2_B2_A2_A2_B2_A2_B2
Optimization infeasible because this subproblem isn't reachable.

## IS Result
status: Status.UNKNOWN
execution time: (base) + (is) = 10.11 + 594.66 = 604.78 seconds
