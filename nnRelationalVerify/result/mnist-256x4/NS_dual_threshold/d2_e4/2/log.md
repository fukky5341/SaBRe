## Execution arguments:
Dataset: Dataset.MNIST
Network: ds/onnx/mnist-net_256x4.onnx
Relational property: GLOBAL_ROBUSTNESS
LP Analysis: True
Epsilon: 0.03125
Delta epsilon: 0.0078125
execution index: (2, 4, 2)
Time budget: 600 seconds
Split limit: 100
Threshold: 14.544726514199999


## IAR start

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
execution time: IAR + RelationalAnalysis = 2.09 + 8.74 = 10.83 seconds
status: Status.UNKNOWN
relational distance
Output dim: 7, lower bound: -14.5592858, upper bound: 14.5592858

# Neuron Split (NS) starts

## BFS NS instance: NS

Time for backsubstitution: 0.00 seconds

### NS candidates at layer 1
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

Time for candidate selection: 0.21 seconds

### Candidate
type: A, layer: 1, pos: 140

## Relational analysis of NS_A1

### Relational analysis result of NS_A1
Status: Status.UNKNOWN
Output dim: 7, lower bound: -14.5484615, upper bound: 14.5494866
time: 5.51 seconds

## Relational analysis of NS_A2

### Relational analysis result of NS_A2
Status: Status.UNKNOWN
Output dim: 7, lower bound: -14.5472453, upper bound: 14.5472453
time: 6.20 seconds

## Summary of splitting at layer (split count: 0)
- Time for NS candidates: 11.93 seconds
NS_A1, status: Status.UNKNOWN, split count: 1, time: 11.93
Output dim: 7, lower bound: -14.5484615, upper bound: 14.5494866
NS_A2, status: Status.UNKNOWN, split count: 1, time: 11.93
Output dim: 7, lower bound: -14.5472453, upper bound: 14.5472453

## BFS NS instance: NS_A1

### Backsubstitution after applying NS history:
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

Time for backsubstitution: 1.88 seconds

### NS candidates at layer 1
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

Time for candidate selection: 0.19 seconds

### Candidate
type: B, layer: 1, pos: 181

## Relational analysis of NS_A1_B1
Optimization infeasible because this subproblem isn't reachable.

### Candidate
type: A, layer: 1, pos: 181

### Candidate
type: B, layer: 1, pos: 187

## Relational analysis of NS_A1_B1

### Relational analysis result of NS_A1_B1
Status: Status.UNKNOWN
Output dim: 7, lower bound: -14.5432908, upper bound: 14.5452286
time: 7.05 seconds

## Relational analysis of NS_A1_B2

### Relational analysis result of NS_A1_B2
Status: Status.UNKNOWN
Output dim: 7, lower bound: -14.5469192, upper bound: 14.5480980
time: 6.93 seconds

## BFS NS instance: NS_A2

### Backsubstitution after applying NS history:
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

Time for backsubstitution: 2.11 seconds

### NS candidates at layer 1
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

Time for candidate selection: 0.25 seconds

### Candidate
type: B, layer: 1, pos: 187

## Relational analysis of NS_A2_B1

### Relational analysis result of NS_A2_B1
Status: Status.VERIFIED
Output dim: 7, lower bound: -14.5421468, upper bound: 14.5427956
time: 4.03 seconds

## Relational analysis of NS_A2_B2

### Relational analysis result of NS_A2_B2
Status: Status.UNKNOWN
Output dim: 7, lower bound: -14.5456956, upper bound: 14.5456956
time: 3.83 seconds

## Summary of splitting at layer (split count: 1)
- Time for NS candidates: 10.22 seconds
NS_A1_B1, status: Status.UNKNOWN, split count: 2, time: 10.22
Output dim: 7, lower bound: -14.5432908, upper bound: 14.5452286
NS_A1_B2, status: Status.UNKNOWN, split count: 2, time: 10.22
Output dim: 7, lower bound: -14.5469192, upper bound: 14.5480980
NS_A2_B1, status: Status.VERIFIED, split count: 2, time: 10.22
Output dim: 7, lower bound: -14.5421468, upper bound: 14.5427956
NS_A2_B2, status: Status.UNKNOWN, split count: 2, time: 10.22
Output dim: 7, lower bound: -14.5456956, upper bound: 14.5456956

## BFS NS instance: NS_A1_B1

### Backsubstitution after applying NS history:
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

Time for backsubstitution: 1.94 seconds

### NS candidates at layer 1
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

Time for candidate selection: 0.20 seconds

### Candidate
type: A, layer: 1, pos: 187

## Relational analysis of NS_A1_B1_A1

### Relational analysis result of NS_A1_B1_A1
Status: Status.VERIFIED
Output dim: 7, lower bound: -14.5430381, upper bound: 14.5445664
time: 6.01 seconds

## Relational analysis of NS_A1_B1_A2

### Relational analysis result of NS_A1_B1_A2
Status: Status.UNKNOWN
Output dim: 7, lower bound: -14.5430381, upper bound: 14.5452286
time: 4.02 seconds

## BFS NS instance: NS_A1_B2

### Backsubstitution after applying NS history:
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

Time for backsubstitution: 1.88 seconds

### NS candidates at layer 1
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

Time for candidate selection: 0.19 seconds

### Candidate
type: A, layer: 1, pos: 187

## Relational analysis of NS_A1_B2_A1

### Relational analysis result of NS_A1_B2_A1
Status: Status.UNKNOWN
Output dim: 7, lower bound: -14.5447624, upper bound: 14.5453999
time: 4.78 seconds

## Relational analysis of NS_A1_B2_A2

### Relational analysis result of NS_A1_B2_A2
Status: Status.UNKNOWN
Output dim: 7, lower bound: -14.5447624, upper bound: 14.5480980
time: 9.09 seconds

## BFS NS instance: NS_A2_B2

### Backsubstitution after applying NS history:
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

Time for backsubstitution: 1.87 seconds

### NS candidates at layer 1
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

Time for candidate selection: 0.19 seconds

### Candidate
type: A, layer: 1, pos: 187

## Relational analysis of NS_A2_B2_A1

### Relational analysis result of NS_A2_B2_A1
Status: Status.VERIFIED
Output dim: 7, lower bound: -14.5427956, upper bound: 14.5421467
time: 2.94 seconds

## Relational analysis of NS_A2_B2_A2

### Relational analysis result of NS_A2_B2_A2
Status: Status.UNKNOWN
Output dim: 7, lower bound: -14.5427956, upper bound: 14.5456956
time: 3.58 seconds

## Summary of splitting at layer (split count: 2)
- Time for NS candidates: 8.59 seconds
NS_A1_B1_A1, status: Status.VERIFIED, split count: 3, time: 8.59
Output dim: 7, lower bound: -14.5430381, upper bound: 14.5445664
NS_A1_B1_A2, status: Status.UNKNOWN, split count: 3, time: 8.59
Output dim: 7, lower bound: -14.5430381, upper bound: 14.5452286
NS_A1_B2_A1, status: Status.UNKNOWN, split count: 3, time: 8.59
Output dim: 7, lower bound: -14.5447624, upper bound: 14.5453999
NS_A1_B2_A2, status: Status.UNKNOWN, split count: 3, time: 8.59
Output dim: 7, lower bound: -14.5447624, upper bound: 14.5480980
NS_A2_B2_A1, status: Status.VERIFIED, split count: 3, time: 8.59
Output dim: 7, lower bound: -14.5427956, upper bound: 14.5421467
NS_A2_B2_A2, status: Status.UNKNOWN, split count: 3, time: 8.59
Output dim: 7, lower bound: -14.5427956, upper bound: 14.5456956

## BFS NS instance: NS_A1_B1_A2

### Backsubstitution after applying NS history:
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

Time for backsubstitution: 2.09 seconds

### NS candidates at layer 1
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

Time for candidate selection: 0.24 seconds

### Candidate
type: A, layer: 1, pos: 181

### Candidate
type: B, layer: 1, pos: 181

### Candidate
type: A, layer: 1, pos: 196

## Relational analysis of NS_A1_B1_A2_A1

### Relational analysis result of NS_A1_B1_A2_A1
Status: Status.VERIFIED
Output dim: 7, lower bound: -14.5372705, upper bound: 14.5387503
time: 4.85 seconds

## Relational analysis of NS_A1_B1_A2_A2

### Relational analysis result of NS_A1_B1_A2_A2
Status: Status.VERIFIED
Output dim: 7, lower bound: -14.5414957, upper bound: 14.5437617
time: 6.84 seconds

## BFS NS instance: NS_A1_B2_A1

### Backsubstitution after applying NS history:
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

Time for backsubstitution: 2.03 seconds

### NS candidates at layer 1
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

Time for candidate selection: 0.19 seconds

### Candidate
type: B, layer: 1, pos: 181

### Candidate
type: A, layer: 1, pos: 181

### Candidate
type: B, layer: 1, pos: 196

## Relational analysis of NS_A1_B2_A1_B1

### Relational analysis result of NS_A1_B2_A1_B1
Status: Status.VERIFIED
Output dim: 7, lower bound: -14.5306006, upper bound: 14.5380102
time: 6.22 seconds

## Relational analysis of NS_A1_B2_A1_B2

### Relational analysis result of NS_A1_B2_A1_B2
Status: Status.VERIFIED
Output dim: 7, lower bound: -14.5414957, upper bound: 14.5439251
time: 5.37 seconds

## BFS NS instance: NS_A1_B2_A2

### Backsubstitution after applying NS history:
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

Time for backsubstitution: 1.87 seconds

### NS candidates at layer 1
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

Time for candidate selection: 0.20 seconds

### Candidate
type: B, layer: 1, pos: 181

### Candidate
type: A, layer: 1, pos: 181

### Candidate
type: B, layer: 1, pos: 144

## Relational analysis of NS_A1_B2_A2_B1
Optimization infeasible because this subproblem isn't reachable.

### Candidate
type: A, layer: 1, pos: 144

### Candidate
type: B, layer: 1, pos: 25

## Relational analysis of NS_A1_B2_A2_B1

### Relational analysis result of NS_A1_B2_A2_B1
Status: Status.VERIFIED
Output dim: 7, lower bound: -14.5415402, upper bound: 14.5430997
time: 4.74 seconds

## Relational analysis of NS_A1_B2_A2_B2

### Relational analysis result of NS_A1_B2_A2_B2
Status: Status.UNKNOWN
Output dim: 7, lower bound: -14.5417921, upper bound: 14.5469509
time: 7.75 seconds

## BFS NS instance: NS_A2_B2_A2

### Backsubstitution after applying NS history:
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

Time for backsubstitution: 1.92 seconds

### NS candidates at layer 1
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

Time for candidate selection: 0.19 seconds

### Candidate
type: B, layer: 1, pos: 181

## Relational analysis of NS_A2_B2_A2_B1
Optimization infeasible because this subproblem isn't reachable.

### Candidate
type: A, layer: 1, pos: 181

### Candidate
type: B, layer: 1, pos: 144

## Relational analysis of NS_A2_B2_A2_B1
Optimization infeasible because this subproblem isn't reachable.

### Candidate
type: B, layer: 1, pos: 196

## Relational analysis of NS_A2_B2_A2_B1

### Relational analysis result of NS_A2_B2_A2_B1
Status: Status.VERIFIED
Output dim: 7, lower bound: -14.5289902, upper bound: 14.5397491
time: 4.55 seconds

## Relational analysis of NS_A2_B2_A2_B2

### Relational analysis result of NS_A2_B2_A2_B2
Status: Status.VERIFIED
Output dim: 7, lower bound: -14.5402292, upper bound: 14.5440281
time: 3.93 seconds

## Summary of splitting at layer (split count: 3)
- Time for NS candidates: 14.78 seconds
NS_A1_B1_A2_A1, status: Status.VERIFIED, split count: 4, time: 14.78
Output dim: 7, lower bound: -14.5372705, upper bound: 14.5387503
NS_A1_B1_A2_A2, status: Status.VERIFIED, split count: 4, time: 14.78
Output dim: 7, lower bound: -14.5414957, upper bound: 14.5437617
NS_A1_B2_A1_B1, status: Status.VERIFIED, split count: 4, time: 14.78
Output dim: 7, lower bound: -14.5306006, upper bound: 14.5380102
NS_A1_B2_A1_B2, status: Status.VERIFIED, split count: 4, time: 14.78
Output dim: 7, lower bound: -14.5414957, upper bound: 14.5439251
NS_A1_B2_A2_B1, status: Status.VERIFIED, split count: 4, time: 14.78
Output dim: 7, lower bound: -14.5415402, upper bound: 14.5430997
NS_A1_B2_A2_B2, status: Status.UNKNOWN, split count: 4, time: 14.78
Output dim: 7, lower bound: -14.5417921, upper bound: 14.5469509
NS_A2_B2_A2_B1, status: Status.VERIFIED, split count: 4, time: 14.78
Output dim: 7, lower bound: -14.5289902, upper bound: 14.5397491
NS_A2_B2_A2_B2, status: Status.VERIFIED, split count: 4, time: 14.78
Output dim: 7, lower bound: -14.5402292, upper bound: 14.5440281

## BFS NS instance: NS_A1_B2_A2_B2

### Backsubstitution after applying NS history:
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

Time for backsubstitution: 1.96 seconds

### NS candidates at layer 1
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

Time for candidate selection: 0.21 seconds

### Candidate
type: B, layer: 1, pos: 181

### Candidate
type: A, layer: 1, pos: 181

### Candidate
type: A, layer: 1, pos: 25

## Relational analysis of NS_A1_B2_A2_B2_A1

### Relational analysis result of NS_A1_B2_A2_B2_A1
Status: Status.UNKNOWN
Output dim: 7, lower bound: -14.5456949, upper bound: 14.5469509
time: 5.46 seconds

## Relational analysis of NS_A1_B2_A2_B2_A2

### Relational analysis result of NS_A1_B2_A2_B2_A2
Status: Status.UNKNOWN
Output dim: 7, lower bound: -14.5456949, upper bound: 14.5469509
time: 8.36 seconds

## Summary of splitting at layer (split count: 4)
- Time for NS candidates: 16.00 seconds
NS_A1_B2_A2_B2_A1, status: Status.UNKNOWN, split count: 5, time: 16.00
Output dim: 7, lower bound: -14.5456949, upper bound: 14.5469509
NS_A1_B2_A2_B2_A2, status: Status.UNKNOWN, split count: 5, time: 16.00
Output dim: 7, lower bound: -14.5456949, upper bound: 14.5469509

## BFS NS instance: NS_A1_B2_A2_B2_A1

### Backsubstitution after applying NS history:
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

Time for backsubstitution: 2.02 seconds

### NS candidates at layer 1
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

Time for candidate selection: 0.19 seconds

### Candidate
type: B, layer: 1, pos: 181

### Candidate
type: A, layer: 1, pos: 181

### Candidate
type: A, layer: 1, pos: 144

### Candidate
type: B, layer: 1, pos: 196

## Relational analysis of NS_A1_B2_A2_B2_A1_B1

### Relational analysis result of NS_A1_B2_A2_B2_A1_B1
Status: Status.VERIFIED
Output dim: 7, lower bound: -14.5388269, upper bound: 14.5409368
time: 4.92 seconds

## Relational analysis of NS_A1_B2_A2_B2_A1_B2

### Relational analysis result of NS_A1_B2_A2_B2_A1_B2
Status: Status.UNKNOWN
Output dim: 7, lower bound: -14.5440484, upper bound: 14.5456485
time: 5.65 seconds

## BFS NS instance: NS_A1_B2_A2_B2_A2

### Backsubstitution after applying NS history:
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

Time for backsubstitution: 1.87 seconds

### NS candidates at layer 1
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

Time for candidate selection: 0.20 seconds

### Candidate
type: B, layer: 1, pos: 181

### Candidate
type: A, layer: 1, pos: 181

### Candidate
type: B, layer: 1, pos: 144

### Candidate
type: A, layer: 1, pos: 144

### Candidate
type: B, layer: 1, pos: 195

## Relational analysis of NS_A1_B2_A2_B2_A2_B1

### Relational analysis result of NS_A1_B2_A2_B2_A2_B1
Status: Status.VERIFIED
Output dim: 7, lower bound: -14.5414207, upper bound: 14.5417868
time: 14.08 seconds

## Relational analysis of NS_A1_B2_A2_B2_A2_B2
Optimization infeasible because this subproblem isn't reachable.

### Candidate
type: B, layer: 1, pos: 71

## Relational analysis of NS_A1_B2_A2_B2_A2_B1
Optimization infeasible because this subproblem isn't reachable.

### Candidate
type: A, layer: 1, pos: 195

### Candidate
type: B, layer: 1, pos: 196

## Relational analysis of NS_A1_B2_A2_B2_A2_B1

### Relational analysis result of NS_A1_B2_A2_B2_A2_B1
Status: Status.VERIFIED
Output dim: 7, lower bound: -14.5388269, upper bound: 14.5409368
time: 4.93 seconds

## Relational analysis of NS_A1_B2_A2_B2_A2_B2

### Relational analysis result of NS_A1_B2_A2_B2_A2_B2
Status: Status.UNKNOWN
Output dim: 7, lower bound: -14.5440484, upper bound: 14.5455655
time: 5.95 seconds

## Summary of splitting at layer (split count: 5)
- Time for NS candidates: 34.15 seconds
NS_A1_B2_A2_B2_A1_B1, status: Status.VERIFIED, split count: 6, time: 34.15
Output dim: 7, lower bound: -14.5388269, upper bound: 14.5409368
NS_A1_B2_A2_B2_A1_B2, status: Status.UNKNOWN, split count: 6, time: 34.15
Output dim: 7, lower bound: -14.5440484, upper bound: 14.5456485
NS_A1_B2_A2_B2_A2_B1, status: Status.VERIFIED, split count: 6, time: 34.15
Output dim: 7, lower bound: -14.5388269, upper bound: 14.5409368
NS_A1_B2_A2_B2_A2_B2, status: Status.UNKNOWN, split count: 6, time: 34.15
Output dim: 7, lower bound: -14.5440484, upper bound: 14.5455655

## BFS NS instance: NS_A1_B2_A2_B2_A1_B2

### Backsubstitution after applying NS history:
0: -8.1808271, 6.6862020, -8.6709995, 7.0688548, -15.2496815, 15.3572016
1: -6.7117448, 6.0236254, -7.0915613, 6.3387699, -13.0505142, 13.1151848
2: -8.4196920, 5.2555780, -8.8749981, 5.4518685, -13.8715591, 14.1305761
3: -9.8692665, 4.7206626, -10.4889832, 4.9400735, -14.8093395, 15.2096462
4: -9.0393677, 7.2210746, -9.5657997, 7.6046371, -16.6439991, 16.7868748
5: -7.7224636, 6.0296807, -8.1566887, 6.3330913, -14.0555515, 14.1863689
6: -7.2818480, 7.9014616, -7.6730251, 8.3708897, -15.6527338, 15.5744839
7: -9.0958672, 5.4250364, -9.6130085, 5.5768275, -14.6726952, 15.0380449
8: -9.1848364, 6.5699186, -9.6916828, 6.9120235, -16.0968590, 16.2616005
9: -7.4594560, 7.5638666, -7.8260403, 7.9338923, -15.3933468, 15.3899059

Time for backsubstitution: 2.41 seconds

### NS candidates at layer 1
type: A, layer: 1, pos: 196
type: A, layer: 1, pos: 181
type: B, layer: 1, pos: 181
type: A, layer: 1, pos: 144
type: A, layer: 1, pos: 195
type: B, layer: 1, pos: 232
type: A, layer: 1, pos: 132
type: B, layer: 1, pos: 144
type: B, layer: 1, pos: 195
type: A, layer: 1, pos: 184
type: A, layer: 1, pos: 71
type: A, layer: 1, pos: 89
type: B, layer: 1, pos: 71
type: A, layer: 1, pos: 1
type: B, layer: 1, pos: 185
type: A, layer: 1, pos: 232
type: A, layer: 1, pos: 77
type: B, layer: 1, pos: 155
type: A, layer: 1, pos: 219
type: A, layer: 1, pos: 185
type: B, layer: 1, pos: 204
type: B, layer: 1, pos: 146
type: A, layer: 1, pos: 197
type: B, layer: 1, pos: 132
type: A, layer: 1, pos: 85
type: B, layer: 1, pos: 77
type: A, layer: 1, pos: 179
type: B, layer: 1, pos: 1
type: B, layer: 1, pos: 89
type: B, layer: 1, pos: 85
type: B, layer: 1, pos: 128
type: B, layer: 1, pos: 197
type: B, layer: 1, pos: 35
type: B, layer: 1, pos: 184
type: B, layer: 1, pos: 140
type: A, layer: 1, pos: 155
type: B, layer: 1, pos: 15
type: A, layer: 1, pos: 35
type: A, layer: 1, pos: 128
type: A, layer: 1, pos: 204
type: A, layer: 1, pos: 15
type: A, layer: 1, pos: 171
type: A, layer: 1, pos: 138
type: B, layer: 1, pos: 138
type: B, layer: 1, pos: 219
type: B, layer: 1, pos: 171
type: A, layer: 1, pos: 146
type: B, layer: 1, pos: 107
type: A, layer: 1, pos: 107
type: B, layer: 1, pos: 179
type: A, layer: 1, pos: 104
type: A, layer: 1, pos: 123
type: B, layer: 1, pos: 123
type: A, layer: 1, pos: 251
type: B, layer: 1, pos: 97
type: B, layer: 1, pos: 233
type: B, layer: 1, pos: 251
type: B, layer: 1, pos: 104
type: A, layer: 1, pos: 97
type: A, layer: 1, pos: 233
type: A, layer: 1, pos: 57
type: B, layer: 1, pos: 57

Time for candidate selection: 0.27 seconds

### Candidate
type: A, layer: 1, pos: 196

## Relational analysis of NS_A1_B2_A2_B2_A1_B2_A1

### Relational analysis result of NS_A1_B2_A2_B2_A1_B2_A1
Status: Status.VERIFIED
Output dim: 7, lower bound: -14.5414995, upper bound: 14.5422842
time: 59.45 seconds

## Relational analysis of NS_A1_B2_A2_B2_A1_B2_A2

### Relational analysis result of NS_A1_B2_A2_B2_A1_B2_A2
Status: Status.UNKNOWN
Output dim: 7, lower bound: -14.5414995, upper bound: 14.5457765
time: 21.08 seconds

## BFS NS instance: NS_A1_B2_A2_B2_A2_B2

### Backsubstitution after applying NS history:
0: -9.1770811, 7.4726753, -8.6709995, 7.0688548, -16.2459354, 16.1436749
1: -7.5243440, 6.6969638, -7.0915613, 6.3387699, -13.8631134, 13.7885246
2: -9.4146576, 5.7873363, -8.8749981, 5.4518685, -14.8665257, 14.6623344
3: -11.1200943, 5.2234764, -10.4889832, 4.9400735, -16.0601673, 15.7124596
4: -10.1267776, 8.0336666, -9.5657997, 7.6046371, -17.7314148, 17.5994663
5: -8.6496792, 6.7078190, -8.1566887, 6.3330913, -14.9827709, 14.8645077
6: -8.1190376, 8.8292694, -7.6730251, 8.3708897, -16.4899254, 16.5022945
7: -10.1461468, 5.9675188, -9.6130085, 5.5768275, -15.7229738, 15.5805273
8: -10.2680130, 7.3111315, -9.6916828, 6.9120235, -17.1800346, 17.0028114
9: -8.3011436, 8.4209061, -7.8260403, 7.9338923, -16.2350330, 16.2469463

Time for backsubstitution: 1.77 seconds

### NS candidates at layer 1
type: A, layer: 1, pos: 196
type: A, layer: 1, pos: 181
type: B, layer: 1, pos: 181
type: A, layer: 1, pos: 144
type: B, layer: 1, pos: 195
type: A, layer: 1, pos: 195
type: B, layer: 1, pos: 144
type: B, layer: 1, pos: 232
type: B, layer: 1, pos: 155
type: A, layer: 1, pos: 132
type: A, layer: 1, pos: 71
type: A, layer: 1, pos: 232
type: A, layer: 1, pos: 1
type: B, layer: 1, pos: 185
type: B, layer: 1, pos: 71
type: A, layer: 1, pos: 89
type: A, layer: 1, pos: 184
type: A, layer: 1, pos: 197
type: A, layer: 1, pos: 155
type: B, layer: 1, pos: 204
type: A, layer: 1, pos: 219
type: A, layer: 1, pos: 85
type: A, layer: 1, pos: 185
type: B, layer: 1, pos: 197
type: A, layer: 1, pos: 77
type: B, layer: 1, pos: 35
type: B, layer: 1, pos: 77
type: B, layer: 1, pos: 132
type: B, layer: 1, pos: 146
type: B, layer: 1, pos: 89
type: B, layer: 1, pos: 85
type: B, layer: 1, pos: 1
type: B, layer: 1, pos: 15
type: A, layer: 1, pos: 35
type: B, layer: 1, pos: 128
type: B, layer: 1, pos: 138
type: A, layer: 1, pos: 179
type: A, layer: 1, pos: 204
type: A, layer: 1, pos: 138
type: B, layer: 1, pos: 140
type: B, layer: 1, pos: 184
type: B, layer: 1, pos: 171
type: A, layer: 1, pos: 128
type: A, layer: 1, pos: 171
type: A, layer: 1, pos: 15
type: B, layer: 1, pos: 219
type: B, layer: 1, pos: 179
type: A, layer: 1, pos: 123
type: B, layer: 1, pos: 123
type: A, layer: 1, pos: 146
type: B, layer: 1, pos: 233
type: B, layer: 1, pos: 97
type: A, layer: 1, pos: 251
type: A, layer: 1, pos: 104
type: A, layer: 1, pos: 107
type: B, layer: 1, pos: 104
type: A, layer: 1, pos: 97
type: B, layer: 1, pos: 107
type: A, layer: 1, pos: 233
type: B, layer: 1, pos: 251
type: A, layer: 1, pos: 57
type: B, layer: 1, pos: 57

Time for candidate selection: 0.17 seconds

### Candidate
type: A, layer: 1, pos: 196

## Relational analysis of NS_A1_B2_A2_B2_A2_B2_A1

### Relational analysis result of NS_A1_B2_A2_B2_A2_B2_A1
Status: Status.VERIFIED
Output dim: 7, lower bound: -14.5412440, upper bound: 14.5417596
time: 4.84 seconds

## Relational analysis of NS_A1_B2_A2_B2_A2_B2_A2

### Relational analysis result of NS_A1_B2_A2_B2_A2_B2_A2
Status: Status.VERIFIED
Output dim: 7, lower bound: -14.5412440, upper bound: 14.5417596
time: 10.76 seconds

## Summary of splitting at layer (split count: 6)
- Time for NS candidates: 17.55 seconds
NS_A1_B2_A2_B2_A1_B2_A1, status: Status.VERIFIED, split count: 7, time: 17.55
Output dim: 7, lower bound: -14.5414995, upper bound: 14.5422842
NS_A1_B2_A2_B2_A1_B2_A2, status: Status.UNKNOWN, split count: 7, time: 17.55
Output dim: 7, lower bound: -14.5414995, upper bound: 14.5457765
NS_A1_B2_A2_B2_A2_B2_A1, status: Status.VERIFIED, split count: 7, time: 17.55
Output dim: 7, lower bound: -14.5412440, upper bound: 14.5417596
NS_A1_B2_A2_B2_A2_B2_A2, status: Status.VERIFIED, split count: 7, time: 17.55
Output dim: 7, lower bound: -14.5412440, upper bound: 14.5417596

## BFS NS instance: NS_A1_B2_A2_B2_A1_B2_A2

### Backsubstitution after applying NS history:
0: -7.6168938, 6.2334061, -8.6709995, 7.0688548, -14.6857491, 14.9044056
1: -6.2251019, 5.6192727, -7.0915613, 6.3387699, -12.5638704, 12.7108345
2: -7.8140869, 4.8768549, -8.8749981, 5.4518685, -13.2659550, 13.7518530
3: -9.1560440, 4.3962660, -10.4889832, 4.9400735, -14.0961170, 14.8852491
4: -8.4120827, 6.7368822, -9.5657997, 7.6046371, -16.0167141, 16.3026810
5: -7.1674604, 5.6106596, -8.1566887, 6.3330913, -13.5005512, 13.7673483
6: -6.7798491, 7.3847647, -7.6730251, 8.3708897, -15.1507349, 15.0577869
7: -8.4964600, 4.9797282, -9.6130085, 5.5768275, -14.0732851, 14.5927362
8: -8.5413551, 6.1205754, -9.6916828, 6.9120235, -15.4533777, 15.8122578
9: -6.9230633, 7.0132627, -7.8260403, 7.9338923, -14.8569555, 14.8392992

Time for backsubstitution: 1.78 seconds

### NS candidates at layer 1
type: B, layer: 1, pos: 181
type: A, layer: 1, pos: 181
type: A, layer: 1, pos: 144
type: B, layer: 1, pos: 144
type: A, layer: 1, pos: 195
type: B, layer: 1, pos: 195
type: B, layer: 1, pos: 232
type: A, layer: 1, pos: 232
type: A, layer: 1, pos: 71
type: B, layer: 1, pos: 71
type: B, layer: 1, pos: 132
type: A, layer: 1, pos: 185
type: A, layer: 1, pos: 132
type: A, layer: 1, pos: 184
type: B, layer: 1, pos: 1
type: B, layer: 1, pos: 89
type: B, layer: 1, pos: 184
type: B, layer: 1, pos: 185
type: A, layer: 1, pos: 89
type: A, layer: 1, pos: 77
type: A, layer: 1, pos: 155
type: A, layer: 1, pos: 1
type: B, layer: 1, pos: 77
type: B, layer: 1, pos: 85
type: B, layer: 1, pos: 219
type: B, layer: 1, pos: 197
type: A, layer: 1, pos: 197
type: A, layer: 1, pos: 204
type: A, layer: 1, pos: 219
type: B, layer: 1, pos: 155
type: B, layer: 1, pos: 204
type: A, layer: 1, pos: 85
type: A, layer: 1, pos: 146
type: A, layer: 1, pos: 35
type: B, layer: 1, pos: 128
type: B, layer: 1, pos: 35
type: A, layer: 1, pos: 128
type: B, layer: 1, pos: 140
type: A, layer: 1, pos: 15
type: B, layer: 1, pos: 146
type: A, layer: 1, pos: 179
type: A, layer: 1, pos: 138
type: B, layer: 1, pos: 179
type: A, layer: 1, pos: 171
type: B, layer: 1, pos: 15
type: B, layer: 1, pos: 171
type: B, layer: 1, pos: 138
type: B, layer: 1, pos: 107
type: B, layer: 1, pos: 123
type: A, layer: 1, pos: 123
type: A, layer: 1, pos: 107
type: A, layer: 1, pos: 104
type: B, layer: 1, pos: 251
type: B, layer: 1, pos: 104
type: A, layer: 1, pos: 251
type: A, layer: 1, pos: 233
type: B, layer: 1, pos: 97
type: A, layer: 1, pos: 97
type: B, layer: 1, pos: 233
type: B, layer: 1, pos: 57
type: A, layer: 1, pos: 57

Time for candidate selection: 0.18 seconds

### Candidate
type: B, layer: 1, pos: 181

### Candidate
type: A, layer: 1, pos: 181

### Candidate
type: A, layer: 1, pos: 144

### Candidate
type: B, layer: 1, pos: 144

### Candidate
type: A, layer: 1, pos: 195

## Relational analysis of NS_A1_B2_A2_B2_A1_B2_A2_A1
Optimization infeasible because this subproblem isn't reachable.

### Candidate
type: B, layer: 1, pos: 195

### Candidate
type: B, layer: 1, pos: 232

## Relational analysis of NS_A1_B2_A2_B2_A1_B2_A2_B1

### Relational analysis result of NS_A1_B2_A2_B2_A1_B2_A2_B1
Status: Status.UNKNOWN
Output dim: 7, lower bound: -14.5386514, upper bound: 14.5456518
time: 5.55 seconds

## Relational analysis of NS_A1_B2_A2_B2_A1_B2_A2_B2

### Relational analysis result of NS_A1_B2_A2_B2_A1_B2_A2_B2
Status: Status.UNKNOWN
Output dim: 7, lower bound: -14.5386568, upper bound: 14.5456522
time: 4.18 seconds

## Summary of splitting at layer (split count: 7)
- Time for NS candidates: 13.77 seconds
NS_A1_B2_A2_B2_A1_B2_A2_B1, status: Status.UNKNOWN, split count: 8, time: 13.77
Output dim: 7, lower bound: -14.5386514, upper bound: 14.5456518
NS_A1_B2_A2_B2_A1_B2_A2_B2, status: Status.UNKNOWN, split count: 8, time: 13.77
Output dim: 7, lower bound: -14.5386568, upper bound: 14.5456522

## BFS NS instance: NS_A1_B2_A2_B2_A1_B2_A2_B1

### Backsubstitution after applying NS history:
0: -7.3933349, 6.0560455, -8.0865593, 6.6010985, -13.9944334, 14.1426048
1: -6.0386696, 5.4656672, -6.5943241, 5.9270654, -11.9657345, 12.0599909
2: -7.5853133, 4.7484560, -8.2625294, 5.1019878, -12.6873016, 13.0109854
3: -8.8769531, 4.2794790, -9.7521420, 4.6118021, -13.4887552, 14.0316210
4: -8.1684942, 6.5511532, -8.9202118, 7.1153765, -15.2838688, 15.4713650
5: -6.9551401, 5.4561720, -7.5853829, 5.9156828, -12.8708229, 13.0415554
6: -6.5874872, 7.1781383, -7.1622496, 7.8347039, -14.4221897, 14.3403873
7: -8.2608738, 4.8412457, -8.9990358, 5.1637077, -13.4245815, 13.8402815
8: -8.2956867, 5.9517488, -9.0389519, 6.4620404, -14.7577267, 14.9906998
9: -6.7286973, 6.8164682, -7.2960072, 7.3946457, -14.1233425, 14.1124744

Time for backsubstitution: 1.78 seconds

### NS candidates at layer 1
type: A, layer: 1, pos: 181
type: B, layer: 1, pos: 181
type: A, layer: 1, pos: 144
type: A, layer: 1, pos: 71
type: A, layer: 1, pos: 195
type: B, layer: 1, pos: 144
type: B, layer: 1, pos: 71
type: A, layer: 1, pos: 132
type: A, layer: 1, pos: 184
type: A, layer: 1, pos: 89
type: B, layer: 1, pos: 195
type: A, layer: 1, pos: 77
type: A, layer: 1, pos: 197
type: A, layer: 1, pos: 1
type: B, layer: 1, pos: 89
type: A, layer: 1, pos: 219
type: B, layer: 1, pos: 132
type: B, layer: 1, pos: 197
type: B, layer: 1, pos: 77
type: B, layer: 1, pos: 184
type: B, layer: 1, pos: 146
type: B, layer: 1, pos: 85
type: A, layer: 1, pos: 185
type: B, layer: 1, pos: 185
type: A, layer: 1, pos: 85
type: B, layer: 1, pos: 1
type: A, layer: 1, pos: 232
type: B, layer: 1, pos: 155
type: A, layer: 1, pos: 179
type: B, layer: 1, pos: 219
type: B, layer: 1, pos: 140
type: A, layer: 1, pos: 146
type: B, layer: 1, pos: 204
type: A, layer: 1, pos: 155
type: A, layer: 1, pos: 171
type: B, layer: 1, pos: 35
type: A, layer: 1, pos: 123
type: B, layer: 1, pos: 123
type: B, layer: 1, pos: 179
type: A, layer: 1, pos: 35
type: B, layer: 1, pos: 128
type: B, layer: 1, pos: 171
type: A, layer: 1, pos: 138
type: A, layer: 1, pos: 15
type: B, layer: 1, pos: 138
type: B, layer: 1, pos: 107
type: B, layer: 1, pos: 15
type: A, layer: 1, pos: 107
type: A, layer: 1, pos: 128
type: A, layer: 1, pos: 204
type: A, layer: 1, pos: 251
type: A, layer: 1, pos: 104
type: B, layer: 1, pos: 97
type: B, layer: 1, pos: 251
type: B, layer: 1, pos: 233
type: B, layer: 1, pos: 104
type: A, layer: 1, pos: 233
type: A, layer: 1, pos: 97
type: A, layer: 1, pos: 57
type: B, layer: 1, pos: 57

Time for candidate selection: 0.17 seconds

### Candidate
type: A, layer: 1, pos: 181

### Candidate
type: B, layer: 1, pos: 181

### Candidate
type: A, layer: 1, pos: 144

### Candidate
type: A, layer: 1, pos: 71

## Relational analysis of NS_A1_B2_A2_B2_A1_B2_A2_B1_A1
Optimization infeasible because this subproblem isn't reachable.

### Candidate
type: A, layer: 1, pos: 195

### Candidate
type: B, layer: 1, pos: 144

### Candidate
type: B, layer: 1, pos: 71

### Candidate
type: A, layer: 1, pos: 132

## Relational analysis of NS_A1_B2_A2_B2_A1_B2_A2_B1_A1
Optimization infeasible because this subproblem isn't reachable.

### Candidate
type: A, layer: 1, pos: 184

## Relational analysis of NS_A1_B2_A2_B2_A1_B2_A2_B1_A1

### Relational analysis result of NS_A1_B2_A2_B2_A1_B2_A2_B1_A1
Status: Status.VERIFIED
Output dim: 7, lower bound: -14.5426815, upper bound: 14.5441793
time: 5.33 seconds

## Relational analysis of NS_A1_B2_A2_B2_A1_B2_A2_B1_A2

### Relational analysis result of NS_A1_B2_A2_B2_A1_B2_A2_B1_A2
Status: Status.UNKNOWN
Output dim: 7, lower bound: -14.5433294, upper bound: 14.5449758
time: 5.75 seconds

## BFS NS instance: NS_A1_B2_A2_B2_A1_B2_A2_B2

### Backsubstitution after applying NS history:
0: -7.5161257, 6.1538391, -8.3376369, 6.8053031, -14.3214283, 14.4914761
1: -6.1416707, 5.5507178, -6.8148966, 6.1111684, -12.2528391, 12.3656130
2: -7.7126637, 4.8203402, -8.5392437, 5.2637272, -12.9763889, 13.3595810
3: -9.0300617, 4.3448906, -10.0701342, 4.7681961, -13.7982578, 14.4150248
4: -8.3024902, 6.6536827, -9.2026052, 7.3293796, -15.6318703, 15.8562880
5: -7.0727482, 5.5420198, -7.8430371, 6.1052866, -13.1780348, 13.3850555
6: -6.6937594, 7.2917418, -7.3878446, 8.0624189, -14.7561779, 14.6795864
7: -8.3903971, 4.9206195, -9.2607651, 5.3799806, -13.7703781, 14.1813831
8: -8.4317741, 6.0448618, -9.3288422, 6.6611662, -15.0929403, 15.3737030
9: -6.8369913, 6.9264221, -7.5399542, 7.6457505, -14.4827423, 14.4663763

Time for backsubstitution: 1.93 seconds

### NS candidates at layer 1
type: B, layer: 1, pos: 181
type: A, layer: 1, pos: 181
type: A, layer: 1, pos: 144
type: B, layer: 1, pos: 144
type: A, layer: 1, pos: 195
type: B, layer: 1, pos: 195
type: A, layer: 1, pos: 71
type: B, layer: 1, pos: 71
type: A, layer: 1, pos: 185
type: B, layer: 1, pos: 132
type: A, layer: 1, pos: 132
type: A, layer: 1, pos: 232
type: A, layer: 1, pos: 184
type: B, layer: 1, pos: 1
type: B, layer: 1, pos: 89
type: A, layer: 1, pos: 77
type: A, layer: 1, pos: 89
type: B, layer: 1, pos: 185
type: B, layer: 1, pos: 184
type: A, layer: 1, pos: 155
type: A, layer: 1, pos: 1
type: B, layer: 1, pos: 85
type: B, layer: 1, pos: 77
type: B, layer: 1, pos: 219
type: B, layer: 1, pos: 197
type: A, layer: 1, pos: 197
type: A, layer: 1, pos: 204
type: B, layer: 1, pos: 204
type: A, layer: 1, pos: 219
type: B, layer: 1, pos: 155
type: B, layer: 1, pos: 35
type: A, layer: 1, pos: 85
type: A, layer: 1, pos: 35
type: B, layer: 1, pos: 128
type: A, layer: 1, pos: 146
type: A, layer: 1, pos: 128
type: A, layer: 1, pos: 15
type: B, layer: 1, pos: 140
type: B, layer: 1, pos: 146
type: A, layer: 1, pos: 179
type: A, layer: 1, pos: 171
type: B, layer: 1, pos: 179
type: A, layer: 1, pos: 138
type: B, layer: 1, pos: 15
type: B, layer: 1, pos: 138
type: B, layer: 1, pos: 107
type: B, layer: 1, pos: 171
type: B, layer: 1, pos: 123
type: A, layer: 1, pos: 107
type: A, layer: 1, pos: 123
type: A, layer: 1, pos: 104
type: B, layer: 1, pos: 251
type: B, layer: 1, pos: 104
type: A, layer: 1, pos: 251
type: A, layer: 1, pos: 233
type: A, layer: 1, pos: 97
type: B, layer: 1, pos: 97
type: B, layer: 1, pos: 233
type: B, layer: 1, pos: 57
type: A, layer: 1, pos: 57

Time for candidate selection: 0.19 seconds

### Candidate
type: B, layer: 1, pos: 181

### Candidate
type: A, layer: 1, pos: 181

### Candidate
type: A, layer: 1, pos: 144

### Candidate
type: B, layer: 1, pos: 144

### Candidate
type: A, layer: 1, pos: 195

### Candidate
type: B, layer: 1, pos: 195

### Candidate
type: A, layer: 1, pos: 71

## Relational analysis of NS_A1_B2_A2_B2_A1_B2_A2_B2_A1
Optimization infeasible because this subproblem isn't reachable.

### Candidate
type: B, layer: 1, pos: 71

### Candidate
type: A, layer: 1, pos: 185

## Relational analysis of NS_A1_B2_A2_B2_A1_B2_A2_B2_A1

### Relational analysis result of NS_A1_B2_A2_B2_A1_B2_A2_B2_A1
Status: Status.UNKNOWN
Output dim: 7, lower bound: -14.5433005, upper bound: 14.5450839
time: 7.11 seconds

## Relational analysis of NS_A1_B2_A2_B2_A1_B2_A2_B2_A2

### Relational analysis result of NS_A1_B2_A2_B2_A1_B2_A2_B2_A2
Status: Status.UNKNOWN
Output dim: 7, lower bound: -14.5431785, upper bound: 14.5447682
time: 5.55 seconds

## Summary of splitting at layer (split count: 8)
- Time for NS candidates: 16.85 seconds
NS_A1_B2_A2_B2_A1_B2_A2_B1_A1, status: Status.VERIFIED, split count: 9, time: 16.85
Output dim: 7, lower bound: -14.5426815, upper bound: 14.5441793
NS_A1_B2_A2_B2_A1_B2_A2_B1_A2, status: Status.UNKNOWN, split count: 9, time: 16.85
Output dim: 7, lower bound: -14.5433294, upper bound: 14.5449758
NS_A1_B2_A2_B2_A1_B2_A2_B2_A1, status: Status.UNKNOWN, split count: 9, time: 16.85
Output dim: 7, lower bound: -14.5433005, upper bound: 14.5450839
NS_A1_B2_A2_B2_A1_B2_A2_B2_A2, status: Status.UNKNOWN, split count: 9, time: 16.85
Output dim: 7, lower bound: -14.5431785, upper bound: 14.5447682

## BFS NS instance: NS_A1_B2_A2_B2_A1_B2_A2_B1_A2

### Backsubstitution after applying NS history:
0: -6.8201776, 5.5926523, -7.8754120, 6.4301152, -13.2502928, 13.4680643
1: -5.5370245, 5.0543809, -6.4090166, 5.7756820, -11.3127060, 11.4633970
2: -6.9714651, 4.3725748, -8.0372725, 4.9657760, -11.9372406, 12.4098473
3: -8.1580238, 3.9605215, -9.4860249, 4.4944248, -12.6524487, 13.4465466
4: -7.5363359, 6.0564542, -8.6873035, 6.9323244, -14.4686604, 14.7437572
5: -6.3943176, 5.0387316, -7.3786564, 5.7619600, -12.1562777, 12.4173880
6: -6.0738630, 6.6551256, -6.9730177, 7.6410503, -13.7149134, 13.6281433
7: -7.6526074, 4.4116292, -8.7745228, 5.0080729, -12.6606808, 13.1861515
8: -7.6423826, 5.5024552, -8.7987051, 6.2961431, -13.9385252, 14.3011589
9: -6.1917210, 6.2731462, -7.0979218, 7.1954298, -13.3871508, 13.3710670

Time for backsubstitution: 1.81 seconds

### NS candidates at layer 1
type: A, layer: 1, pos: 181
type: B, layer: 1, pos: 181
type: A, layer: 1, pos: 144
type: B, layer: 1, pos: 184
type: B, layer: 1, pos: 144
type: A, layer: 1, pos: 71
type: A, layer: 1, pos: 195
type: B, layer: 1, pos: 71
type: A, layer: 1, pos: 132
type: B, layer: 1, pos: 195
type: A, layer: 1, pos: 89
type: B, layer: 1, pos: 132
type: A, layer: 1, pos: 77
type: B, layer: 1, pos: 89
type: B, layer: 1, pos: 77
type: A, layer: 1, pos: 197
type: B, layer: 1, pos: 197
type: A, layer: 1, pos: 1
type: B, layer: 1, pos: 85
type: A, layer: 1, pos: 219
type: A, layer: 1, pos: 185
type: B, layer: 1, pos: 146
type: B, layer: 1, pos: 1
type: B, layer: 1, pos: 219
type: A, layer: 1, pos: 232
type: A, layer: 1, pos: 146
type: A, layer: 1, pos: 85
type: B, layer: 1, pos: 185
type: A, layer: 1, pos: 179
type: B, layer: 1, pos: 140
type: B, layer: 1, pos: 179
type: B, layer: 1, pos: 204
type: A, layer: 1, pos: 204
type: B, layer: 1, pos: 107
type: A, layer: 1, pos: 123
type: B, layer: 1, pos: 123
type: B, layer: 1, pos: 35
type: A, layer: 1, pos: 35
type: B, layer: 1, pos: 171
type: A, layer: 1, pos: 155
type: B, layer: 1, pos: 128
type: A, layer: 1, pos: 107
type: A, layer: 1, pos: 138
type: A, layer: 1, pos: 128
type: B, layer: 1, pos: 155
type: A, layer: 1, pos: 171
type: B, layer: 1, pos: 138
type: B, layer: 1, pos: 15
type: A, layer: 1, pos: 15
type: B, layer: 1, pos: 97
type: A, layer: 1, pos: 251
type: B, layer: 1, pos: 251
type: A, layer: 1, pos: 97
type: B, layer: 1, pos: 104
type: A, layer: 1, pos: 104
type: A, layer: 1, pos: 233
type: B, layer: 1, pos: 233
type: A, layer: 1, pos: 57
type: B, layer: 1, pos: 57

Time for candidate selection: 0.19 seconds

### Candidate
type: A, layer: 1, pos: 181

### Candidate
type: B, layer: 1, pos: 181

### Candidate
type: A, layer: 1, pos: 144

### Candidate
type: B, layer: 1, pos: 184

## Relational analysis of NS_A1_B2_A2_B2_A1_B2_A2_B1_A2_B1

### Relational analysis result of NS_A1_B2_A2_B2_A1_B2_A2_B1_A2_B1
Status: Status.VERIFIED
Output dim: 7, lower bound: -14.5409205, upper bound: 14.5427337
time: 46.58 seconds

## Relational analysis of NS_A1_B2_A2_B2_A1_B2_A2_B1_A2_B2

### Relational analysis result of NS_A1_B2_A2_B2_A1_B2_A2_B1_A2_B2
Status: Status.UNKNOWN
Output dim: 7, lower bound: -14.5409205, upper bound: 14.5449758
time: 4.10 seconds

## BFS NS instance: NS_A1_B2_A2_B2_A1_B2_A2_B2_A1

### Backsubstitution after applying NS history:
0: -5.7695556, 4.7530775, -8.2114544, 6.7041006, -12.4736557, 12.9645319
1: -4.6437531, 4.3097949, -6.7071452, 6.0221405, -10.6658936, 11.0169392
2: -5.8834624, 3.7545567, -8.4055843, 5.1853704, -11.0688324, 12.1601410
3: -6.8312330, 3.3972685, -9.9124241, 4.6985664, -11.5297985, 13.3096924
4: -6.3601527, 5.1737609, -9.0627699, 7.2222419, -13.5823946, 14.2365303
5: -5.3845978, 4.2954493, -7.7205219, 6.0145950, -11.3991928, 12.0159712
6: -5.1582503, 5.6730232, -7.2768517, 7.9431767, -13.1014271, 12.9498749
7: -6.5093341, 3.7315526, -9.1203146, 5.2926307, -11.8019638, 12.8518677
8: -6.4662566, 4.6969242, -9.1874018, 6.5633106, -13.0295677, 13.8843250
9: -5.2590218, 5.3054347, -7.4260163, 7.5277238, -12.7867451, 12.7314510

Time for backsubstitution: 1.86 seconds

### NS candidates at layer 1
type: B, layer: 1, pos: 181
type: A, layer: 1, pos: 181
type: B, layer: 1, pos: 132
type: B, layer: 1, pos: 184
type: A, layer: 1, pos: 144
type: B, layer: 1, pos: 89
type: B, layer: 1, pos: 144
type: B, layer: 1, pos: 71
type: A, layer: 1, pos: 195
type: A, layer: 1, pos: 232
type: A, layer: 1, pos: 71
type: B, layer: 1, pos: 1
type: B, layer: 1, pos: 219
type: A, layer: 1, pos: 146
type: B, layer: 1, pos: 195
type: A, layer: 1, pos: 155
type: B, layer: 1, pos: 197
type: B, layer: 1, pos: 85
type: B, layer: 1, pos: 77
type: A, layer: 1, pos: 77
type: B, layer: 1, pos: 179
type: A, layer: 1, pos: 204
type: A, layer: 1, pos: 197
type: A, layer: 1, pos: 184
type: B, layer: 1, pos: 107
type: A, layer: 1, pos: 89
type: A, layer: 1, pos: 132
type: B, layer: 1, pos: 140
type: A, layer: 1, pos: 179
type: B, layer: 1, pos: 146
type: B, layer: 1, pos: 251
type: A, layer: 1, pos: 233
type: A, layer: 1, pos: 85
type: B, layer: 1, pos: 185
type: B, layer: 1, pos: 123
type: A, layer: 1, pos: 128
type: A, layer: 1, pos: 97
type: A, layer: 1, pos: 138
type: A, layer: 1, pos: 15
type: A, layer: 1, pos: 123
type: A, layer: 1, pos: 35
type: A, layer: 1, pos: 1
type: A, layer: 1, pos: 219
type: A, layer: 1, pos: 171
type: B, layer: 1, pos: 35
type: B, layer: 1, pos: 138
type: B, layer: 1, pos: 204
type: B, layer: 1, pos: 128
type: B, layer: 1, pos: 104
type: B, layer: 1, pos: 171
type: B, layer: 1, pos: 97
type: A, layer: 1, pos: 107
type: A, layer: 1, pos: 104
type: A, layer: 1, pos: 251
type: B, layer: 1, pos: 15
type: B, layer: 1, pos: 57
type: B, layer: 1, pos: 155
type: A, layer: 1, pos: 57
type: B, layer: 1, pos: 233

Time for candidate selection: 0.18 seconds

### Candidate
type: B, layer: 1, pos: 181

### Candidate
type: A, layer: 1, pos: 181

### Candidate
type: B, layer: 1, pos: 132

## Relational analysis of NS_A1_B2_A2_B2_A1_B2_A2_B2_A1_B1
Optimization infeasible because this subproblem isn't reachable.

### Candidate
type: B, layer: 1, pos: 184

## Relational analysis of NS_A1_B2_A2_B2_A1_B2_A2_B2_A1_B1

### Relational analysis result of NS_A1_B2_A2_B2_A1_B2_A2_B2_A1_B1
Status: Status.VERIFIED
Output dim: 7, lower bound: -14.5404722, upper bound: 14.5424454
time: 9.01 seconds

## Relational analysis of NS_A1_B2_A2_B2_A1_B2_A2_B2_A1_B2

### Relational analysis result of NS_A1_B2_A2_B2_A1_B2_A2_B2_A1_B2
Status: Status.VERIFIED
Output dim: 7, lower bound: -14.5424968, upper bound: 14.5443701
time: 6.70 seconds

## BFS NS instance: NS_A1_B2_A2_B2_A1_B2_A2_B2_A2

### Backsubstitution after applying NS history:
0: -7.3024654, 5.9836607, -8.2516356, 6.7366514, -14.0391169, 14.2352962
1: -5.9618216, 5.4028044, -6.7421703, 6.0512505, -12.0130711, 12.1449747
2: -7.4904370, 4.6915913, -8.4496517, 5.2116194, -12.7020550, 13.1412430
3: -8.7652473, 4.2300968, -9.9627457, 4.7212191, -13.4864664, 14.1928425
4: -8.0673447, 6.4750042, -9.1078720, 7.2573767, -15.3247204, 15.5828762
5: -6.8670893, 5.3911390, -7.7601061, 6.0442982, -12.9113865, 13.1512451
6: -6.5082827, 7.0926065, -7.3130507, 7.9819789, -14.4902611, 14.4056549
7: -8.1584578, 4.7795110, -9.1668968, 5.3226795, -13.4811373, 13.9464073
8: -8.1952553, 5.8814588, -9.2335320, 6.5951767, -14.7904263, 15.1149902
9: -6.6483479, 6.7318063, -7.4634914, 7.5670400, -14.2153845, 14.1952953

Time for backsubstitution: 1.85 seconds

### NS candidates at layer 1
type: B, layer: 1, pos: 181
type: A, layer: 1, pos: 181
type: A, layer: 1, pos: 144
type: B, layer: 1, pos: 144
type: A, layer: 1, pos: 195
type: B, layer: 1, pos: 195
type: A, layer: 1, pos: 71
type: B, layer: 1, pos: 71
type: B, layer: 1, pos: 132
type: A, layer: 1, pos: 132
type: A, layer: 1, pos: 232
type: A, layer: 1, pos: 184
type: B, layer: 1, pos: 89
type: B, layer: 1, pos: 1
type: B, layer: 1, pos: 184
type: A, layer: 1, pos: 77
type: A, layer: 1, pos: 89
type: B, layer: 1, pos: 185
type: A, layer: 1, pos: 1
type: B, layer: 1, pos: 77
type: A, layer: 1, pos: 155
type: B, layer: 1, pos: 85
type: B, layer: 1, pos: 197
type: B, layer: 1, pos: 219
type: A, layer: 1, pos: 197
type: A, layer: 1, pos: 219
type: A, layer: 1, pos: 204
type: B, layer: 1, pos: 204
type: A, layer: 1, pos: 85
type: B, layer: 1, pos: 155
type: A, layer: 1, pos: 35
type: B, layer: 1, pos: 35
type: B, layer: 1, pos: 146
type: B, layer: 1, pos: 128
type: A, layer: 1, pos: 146
type: A, layer: 1, pos: 128
type: B, layer: 1, pos: 140
type: A, layer: 1, pos: 15
type: A, layer: 1, pos: 179
type: B, layer: 1, pos: 179
type: A, layer: 1, pos: 171
type: A, layer: 1, pos: 138
type: B, layer: 1, pos: 138
type: B, layer: 1, pos: 15
type: B, layer: 1, pos: 171
type: B, layer: 1, pos: 107
type: A, layer: 1, pos: 107
type: B, layer: 1, pos: 123
type: A, layer: 1, pos: 123
type: A, layer: 1, pos: 104
type: B, layer: 1, pos: 251
type: B, layer: 1, pos: 104
type: A, layer: 1, pos: 251
type: A, layer: 1, pos: 97
type: B, layer: 1, pos: 97
type: A, layer: 1, pos: 233
type: B, layer: 1, pos: 233
type: B, layer: 1, pos: 57
type: A, layer: 1, pos: 57

Time for candidate selection: 0.18 seconds

### Candidate
type: B, layer: 1, pos: 181

### Candidate
type: A, layer: 1, pos: 181

### Candidate
type: A, layer: 1, pos: 144

### Candidate
type: B, layer: 1, pos: 144

### Candidate
type: A, layer: 1, pos: 195

### Candidate
type: B, layer: 1, pos: 195

### Candidate
type: A, layer: 1, pos: 71

### Candidate
type: B, layer: 1, pos: 71

### Candidate
type: B, layer: 1, pos: 132

## Relational analysis of NS_A1_B2_A2_B2_A1_B2_A2_B2_A2_B1
Optimization infeasible because this subproblem isn't reachable.

### Candidate
type: A, layer: 1, pos: 132

### Candidate
type: A, layer: 1, pos: 232

## Relational analysis of NS_A1_B2_A2_B2_A1_B2_A2_B2_A2_A1

### Relational analysis result of NS_A1_B2_A2_B2_A1_B2_A2_B2_A2_A1
Status: Status.UNKNOWN
Output dim: 7, lower bound: -14.5431775, upper bound: 14.5447682
time: 7.50 seconds

## Relational analysis of NS_A1_B2_A2_B2_A1_B2_A2_B2_A2_A2

### Relational analysis result of NS_A1_B2_A2_B2_A1_B2_A2_B2_A2_A2
Status: Status.UNKNOWN
Output dim: 7, lower bound: -14.5431775, upper bound: 14.5447433
time: 5.37 seconds

## Summary of splitting at layer (split count: 9)
- Time for NS candidates: 16.99 seconds
NS_A1_B2_A2_B2_A1_B2_A2_B1_A2_B1, status: Status.VERIFIED, split count: 10, time: 16.99
Output dim: 7, lower bound: -14.5409205, upper bound: 14.5427337
NS_A1_B2_A2_B2_A1_B2_A2_B1_A2_B2, status: Status.UNKNOWN, split count: 10, time: 16.99
Output dim: 7, lower bound: -14.5409205, upper bound: 14.5449758
NS_A1_B2_A2_B2_A1_B2_A2_B2_A1_B1, status: Status.VERIFIED, split count: 10, time: 16.99
Output dim: 7, lower bound: -14.5404722, upper bound: 14.5424454
NS_A1_B2_A2_B2_A1_B2_A2_B2_A1_B2, status: Status.VERIFIED, split count: 10, time: 16.99
Output dim: 7, lower bound: -14.5424968, upper bound: 14.5443701
NS_A1_B2_A2_B2_A1_B2_A2_B2_A2_A1, status: Status.UNKNOWN, split count: 10, time: 16.99
Output dim: 7, lower bound: -14.5431775, upper bound: 14.5447682
NS_A1_B2_A2_B2_A1_B2_A2_B2_A2_A2, status: Status.UNKNOWN, split count: 10, time: 16.99
Output dim: 7, lower bound: -14.5431775, upper bound: 14.5447433

## BFS NS instance: NS_A1_B2_A2_B2_A1_B2_A2_B1_A2_B2

### Backsubstitution after applying NS history:
0: -6.8201776, 5.5926523, -7.5390844, 6.1595316, -12.9797096, 13.1317368
1: -5.5370245, 5.0543809, -6.1144781, 5.5336204, -11.0706444, 11.1688585
2: -6.9714651, 4.3725748, -7.6769876, 4.7494025, -11.7208672, 12.0495625
3: -8.1580238, 3.9605215, -9.0635147, 4.3070116, -12.4650354, 13.0240364
4: -7.5363359, 6.0564542, -8.3159275, 6.6415720, -14.1779079, 14.3723803
5: -6.3943176, 5.0387316, -7.0493894, 5.5171475, -11.9114628, 12.0881214
6: -6.0738630, 6.6551256, -6.6711655, 7.3322697, -13.4061327, 13.3262901
7: -7.6526074, 4.4116292, -8.4154663, 4.7613411, -12.4139462, 12.8270950
8: -7.6423826, 5.5024552, -8.4159641, 6.0316687, -13.6740484, 13.9184189
9: -6.1917210, 6.2731462, -6.7818327, 6.8787951, -13.0705166, 13.0549793

Time for backsubstitution: 1.79 seconds

### NS candidates at layer 1
type: A, layer: 1, pos: 181
type: B, layer: 1, pos: 181
type: A, layer: 1, pos: 144
type: A, layer: 1, pos: 71
type: B, layer: 1, pos: 144
type: A, layer: 1, pos: 195
type: B, layer: 1, pos: 71
type: A, layer: 1, pos: 132
type: A, layer: 1, pos: 89
type: B, layer: 1, pos: 195
type: A, layer: 1, pos: 77
type: B, layer: 1, pos: 132
type: B, layer: 1, pos: 89
type: A, layer: 1, pos: 197
type: A, layer: 1, pos: 1
type: A, layer: 1, pos: 219
type: B, layer: 1, pos: 77
type: B, layer: 1, pos: 146
type: B, layer: 1, pos: 197
type: B, layer: 1, pos: 85
type: B, layer: 1, pos: 1
type: A, layer: 1, pos: 85
type: A, layer: 1, pos: 185
type: A, layer: 1, pos: 232
type: B, layer: 1, pos: 185
type: B, layer: 1, pos: 219
type: A, layer: 1, pos: 146
type: A, layer: 1, pos: 179
type: B, layer: 1, pos: 140
type: B, layer: 1, pos: 204
type: A, layer: 1, pos: 123
type: B, layer: 1, pos: 123
type: B, layer: 1, pos: 107
type: B, layer: 1, pos: 179
type: B, layer: 1, pos: 35
type: A, layer: 1, pos: 107
type: A, layer: 1, pos: 204
type: B, layer: 1, pos: 155
type: B, layer: 1, pos: 128
type: A, layer: 1, pos: 35
type: A, layer: 1, pos: 171
type: A, layer: 1, pos: 138
type: A, layer: 1, pos: 128
type: B, layer: 1, pos: 171
type: B, layer: 1, pos: 138
type: A, layer: 1, pos: 15
type: A, layer: 1, pos: 155
type: A, layer: 1, pos: 251
type: B, layer: 1, pos: 15
type: B, layer: 1, pos: 97
type: A, layer: 1, pos: 104
type: B, layer: 1, pos: 251
type: A, layer: 1, pos: 97
type: B, layer: 1, pos: 233
type: B, layer: 1, pos: 104
type: A, layer: 1, pos: 233
type: A, layer: 1, pos: 57
type: B, layer: 1, pos: 57

Time for candidate selection: 0.17 seconds

### Candidate
type: A, layer: 1, pos: 181

### Candidate
type: B, layer: 1, pos: 181

### Candidate
type: A, layer: 1, pos: 144

### Candidate
type: A, layer: 1, pos: 71

### Candidate
type: B, layer: 1, pos: 144

### Candidate
type: A, layer: 1, pos: 195

### Candidate
type: B, layer: 1, pos: 71

### Candidate
type: A, layer: 1, pos: 132

### Candidate
type: A, layer: 1, pos: 89

## Relational analysis of NS_A1_B2_A2_B2_A1_B2_A2_B1_A2_B2_A1

### Relational analysis result of NS_A1_B2_A2_B2_A1_B2_A2_B1_A2_B2_A1
Status: Status.VERIFIED
Output dim: 7, lower bound: -14.5400846, upper bound: 14.5442748
time: 5.41 seconds

## Relational analysis of NS_A1_B2_A2_B2_A1_B2_A2_B1_A2_B2_A2

### Relational analysis result of NS_A1_B2_A2_B2_A1_B2_A2_B1_A2_B2_A2
Status: Status.VERIFIED
Output dim: 7, lower bound: -14.5400952, upper bound: 14.5420008
time: 45.74 seconds

## BFS NS instance: NS_A1_B2_A2_B2_A1_B2_A2_B2_A2_A1

### Backsubstitution after applying NS history:
0: -6.8094187, 5.5887132, -8.2516356, 6.7366514, -13.5460701, 13.8403482
1: -5.5419664, 5.0528164, -6.7421703, 6.0512505, -11.5932169, 11.7949858
2: -6.9652848, 4.3887849, -8.4496517, 5.2116194, -12.1769047, 12.8384352
3: -8.1506023, 3.9510241, -9.9627457, 4.7212191, -12.8718204, 13.9137697
4: -7.5210023, 6.0612640, -9.1078720, 7.2573767, -14.7783794, 15.1691351
5: -6.3815475, 5.0346541, -7.7601061, 6.0442982, -12.4258461, 12.7947598
6: -6.0751133, 6.6441212, -7.3130507, 7.9819789, -14.0570927, 13.9571724
7: -7.6424408, 4.4145126, -9.1668968, 5.3226795, -12.9651203, 13.5814095
8: -7.6393218, 5.5016627, -9.2335320, 6.5951767, -14.2344933, 14.7351933
9: -6.1977501, 6.2700930, -7.4634914, 7.5670400, -13.7647896, 13.7335844

Time for backsubstitution: 1.84 seconds

### NS candidates at layer 1
type: B, layer: 1, pos: 181
type: A, layer: 1, pos: 181
type: A, layer: 1, pos: 144
type: B, layer: 1, pos: 144
type: B, layer: 1, pos: 71
type: B, layer: 1, pos: 132
type: B, layer: 1, pos: 184
type: A, layer: 1, pos: 195
type: A, layer: 1, pos: 71
type: B, layer: 1, pos: 89
type: B, layer: 1, pos: 195
type: B, layer: 1, pos: 1
type: B, layer: 1, pos: 77
type: B, layer: 1, pos: 219
type: B, layer: 1, pos: 197
type: A, layer: 1, pos: 77
type: A, layer: 1, pos: 184
type: A, layer: 1, pos: 146
type: A, layer: 1, pos: 89
type: A, layer: 1, pos: 132
type: B, layer: 1, pos: 85
type: A, layer: 1, pos: 197
type: B, layer: 1, pos: 185
type: A, layer: 1, pos: 155
type: A, layer: 1, pos: 85
type: A, layer: 1, pos: 204
type: B, layer: 1, pos: 179
type: B, layer: 1, pos: 140
type: B, layer: 1, pos: 146
type: A, layer: 1, pos: 219
type: B, layer: 1, pos: 107
type: A, layer: 1, pos: 1
type: A, layer: 1, pos: 179
type: B, layer: 1, pos: 123
type: A, layer: 1, pos: 35
type: A, layer: 1, pos: 123
type: A, layer: 1, pos: 128
type: B, layer: 1, pos: 171
type: A, layer: 1, pos: 15
type: B, layer: 1, pos: 35
type: A, layer: 1, pos: 138
type: B, layer: 1, pos: 204
type: B, layer: 1, pos: 128
type: B, layer: 1, pos: 251
type: A, layer: 1, pos: 171
type: B, layer: 1, pos: 138
type: A, layer: 1, pos: 107
type: A, layer: 1, pos: 97
type: A, layer: 1, pos: 233
type: B, layer: 1, pos: 104
type: B, layer: 1, pos: 15
type: A, layer: 1, pos: 104
type: B, layer: 1, pos: 97
type: A, layer: 1, pos: 251
type: B, layer: 1, pos: 155
type: B, layer: 1, pos: 233
type: B, layer: 1, pos: 57
type: A, layer: 1, pos: 57

Time for candidate selection: 0.18 seconds

### Candidate
type: B, layer: 1, pos: 181

### Candidate
type: A, layer: 1, pos: 181

### Candidate
type: A, layer: 1, pos: 144

### Candidate
type: B, layer: 1, pos: 144

### Candidate
type: B, layer: 1, pos: 71

### Candidate
type: B, layer: 1, pos: 132

### Candidate
type: B, layer: 1, pos: 184

## Relational analysis of NS_A1_B2_A2_B2_A1_B2_A2_B2_A2_A1_B1

### Relational analysis result of NS_A1_B2_A2_B2_A1_B2_A2_B2_A2_A1_B1
Status: Status.VERIFIED
Output dim: 7, lower bound: -14.5403380, upper bound: 14.5420682
time: 5.55 seconds

## Relational analysis of NS_A1_B2_A2_B2_A1_B2_A2_B2_A2_A1_B2

### Relational analysis result of NS_A1_B2_A2_B2_A1_B2_A2_B2_A2_A1_B2
Status: Status.VERIFIED
Output dim: 7, lower bound: -14.5423667, upper bound: 14.5439724
time: 34.86 seconds

## BFS NS instance: NS_A1_B2_A2_B2_A1_B2_A2_B2_A2_A2

### Backsubstitution after applying NS history:
0: -7.0717020, 5.8015718, -8.2516356, 6.7366514, -13.8083515, 14.0532074
1: -5.7712903, 5.2457819, -6.7421703, 6.0512505, -11.8225403, 11.9879522
2: -7.2581401, 4.5627689, -8.4496517, 5.2116194, -12.4697590, 13.0124207
3: -8.4772091, 4.1124449, -9.9627457, 4.7212191, -13.1984282, 14.0751905
4: -7.8163276, 6.2846122, -9.1078720, 7.2573767, -15.0737038, 15.3924818
5: -6.6500502, 5.2344933, -7.7601061, 6.0442982, -12.6943483, 12.9945993
6: -6.3115015, 6.8795595, -7.3130507, 7.9819789, -14.2934799, 14.1926098
7: -7.9157972, 4.6440873, -9.1668968, 5.3226795, -13.2384768, 13.8109837
8: -7.9444141, 5.7087288, -9.2335320, 6.5951767, -14.5395889, 14.9422588
9: -6.4518485, 6.5328712, -7.4634914, 7.5670400, -14.0188875, 13.9963617

Time for backsubstitution: 1.81 seconds

### NS candidates at layer 1
type: B, layer: 1, pos: 181
type: A, layer: 1, pos: 181
type: A, layer: 1, pos: 144
type: B, layer: 1, pos: 144
type: A, layer: 1, pos: 195
type: B, layer: 1, pos: 195
type: B, layer: 1, pos: 71
type: A, layer: 1, pos: 71
type: B, layer: 1, pos: 132
type: A, layer: 1, pos: 132
type: A, layer: 1, pos: 184
type: B, layer: 1, pos: 1
type: B, layer: 1, pos: 89
type: B, layer: 1, pos: 184
type: A, layer: 1, pos: 89
type: A, layer: 1, pos: 77
type: B, layer: 1, pos: 185
type: A, layer: 1, pos: 1
type: B, layer: 1, pos: 77
type: A, layer: 1, pos: 155
type: B, layer: 1, pos: 85
type: B, layer: 1, pos: 197
type: B, layer: 1, pos: 219
type: A, layer: 1, pos: 197
type: A, layer: 1, pos: 204
type: A, layer: 1, pos: 219
type: B, layer: 1, pos: 204
type: A, layer: 1, pos: 85
type: B, layer: 1, pos: 155
type: A, layer: 1, pos: 35
type: B, layer: 1, pos: 35
type: B, layer: 1, pos: 128
type: A, layer: 1, pos: 146
type: B, layer: 1, pos: 146
type: A, layer: 1, pos: 128
type: B, layer: 1, pos: 140
type: A, layer: 1, pos: 179
type: A, layer: 1, pos: 15
type: B, layer: 1, pos: 179
type: B, layer: 1, pos: 15
type: A, layer: 1, pos: 138
type: B, layer: 1, pos: 171
type: B, layer: 1, pos: 138
type: A, layer: 1, pos: 171
type: B, layer: 1, pos: 107
type: A, layer: 1, pos: 107
type: B, layer: 1, pos: 123
type: A, layer: 1, pos: 123
type: A, layer: 1, pos: 104
type: B, layer: 1, pos: 104
type: B, layer: 1, pos: 251
type: A, layer: 1, pos: 251
type: A, layer: 1, pos: 97
type: B, layer: 1, pos: 97
type: A, layer: 1, pos: 233
type: B, layer: 1, pos: 233
type: B, layer: 1, pos: 57
type: A, layer: 1, pos: 57

Time for candidate selection: 0.18 seconds

### Candidate
type: B, layer: 1, pos: 181

### Candidate
type: A, layer: 1, pos: 181

### Candidate
type: A, layer: 1, pos: 144

### Candidate
type: B, layer: 1, pos: 144

### Candidate
type: A, layer: 1, pos: 195

### Candidate
type: B, layer: 1, pos: 195

### Candidate
type: B, layer: 1, pos: 71

### Candidate
type: A, layer: 1, pos: 71

### Candidate
type: B, layer: 1, pos: 132

### Candidate
type: A, layer: 1, pos: 132

### Candidate
type: A, layer: 1, pos: 184

## Relational analysis of NS_A1_B2_A2_B2_A1_B2_A2_B2_A2_A2_A1

### Relational analysis result of NS_A1_B2_A2_B2_A1_B2_A2_B2_A2_A2_A1
Status: Status.VERIFIED
Output dim: 7, lower bound: -14.5418517, upper bound: 14.5432902
time: 5.89 seconds

## Relational analysis of NS_A1_B2_A2_B2_A1_B2_A2_B2_A2_A2_A2

### Relational analysis result of NS_A1_B2_A2_B2_A1_B2_A2_B2_A2_A2_A2
Status: Status.VERIFIED
Output dim: 7, lower bound: -14.5423667, upper bound: 14.5439724
time: 5.78 seconds

## Summary of splitting at layer (split count: 10)
- Time for NS candidates: 13.68 seconds
NS_A1_B2_A2_B2_A1_B2_A2_B1_A2_B2_A1, status: Status.VERIFIED, split count: 11, time: 13.68
Output dim: 7, lower bound: -14.5400846, upper bound: 14.5442748
NS_A1_B2_A2_B2_A1_B2_A2_B1_A2_B2_A2, status: Status.VERIFIED, split count: 11, time: 13.68
Output dim: 7, lower bound: -14.5400952, upper bound: 14.5420008
NS_A1_B2_A2_B2_A1_B2_A2_B2_A2_A1_B1, status: Status.VERIFIED, split count: 11, time: 13.68
Output dim: 7, lower bound: -14.5403380, upper bound: 14.5420682
NS_A1_B2_A2_B2_A1_B2_A2_B2_A2_A1_B2, status: Status.VERIFIED, split count: 11, time: 13.68
Output dim: 7, lower bound: -14.5423667, upper bound: 14.5439724
NS_A1_B2_A2_B2_A1_B2_A2_B2_A2_A2_A1, status: Status.VERIFIED, split count: 11, time: 13.68
Output dim: 7, lower bound: -14.5418517, upper bound: 14.5432902
NS_A1_B2_A2_B2_A1_B2_A2_B2_A2_A2_A2, status: Status.VERIFIED, split count: 11, time: 13.68
Output dim: 7, lower bound: -14.5423667, upper bound: 14.5439724

## NS Result
status: Status.VERIFIED
execution time: (base) + (ns) = 10.83 + 548.83 = 559.66 seconds
