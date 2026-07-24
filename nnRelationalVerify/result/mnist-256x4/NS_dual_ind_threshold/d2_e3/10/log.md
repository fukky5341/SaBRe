## Execution arguments:
Dataset: Dataset.MNIST
Network: ds/onnx/mnist-net_256x4.onnx
Relational property: GLOBAL_ROBUSTNESS
LP Analysis: True
Epsilon: 0.0234375
Delta epsilon: 0.0078125
execution index: (2, 3, 10)
Time budget: 600 seconds
Split limit: 100
Threshold: 0.320591111


## IAR start

### BASE IAR bounds
Layer (inp1_lb, inp1_ub, inp2_lb, inp2_ub, d_lb, d_ub)
0: (-0.5426022, 0.4665188, -0.5426022, 0.4665188, -1.0091211, 1.0091211)
1: (-0.3630026, 0.3911184, -0.3630026, 0.3911184, -0.7541210, 0.7541210)
2: (-0.5269889, 0.4806983, -0.5269889, 0.4806983, -1.0076871, 1.0076871)
3: (0.7543352, 1.1590887, 0.7543352, 1.1590887, -0.4047535, 0.4047535)
4: (-0.4614211, 0.4625955, -0.4614211, 0.4625955, -0.9240167, 0.9240167)
5: (-0.4214845, 0.4672619, -0.4214845, 0.4672619, -0.8887464, 0.8887464)
6: (-0.5138452, 0.5371552, -0.5138452, 0.5371552, -1.0510004, 1.0510004)
7: (-0.4409947, 0.4028524, -0.4409947, 0.4028524, -0.8438470, 0.8438470)
8: (-0.4306176, 0.5533380, -0.4306176, 0.5533380, -0.9839556, 0.9839556)
9: (-0.4872051, 0.3854194, -0.4872051, 0.3854194, -0.8726246, 0.8726246)

## BASE Relational Analysis

## BASE Result
execution time: IAR + RelationalAnalysis = 2.34 + 2.49 = 4.83 seconds
status: Status.UNKNOWN
relational distance
Output dim: 3, lower bound: -0.3305063, upper bound: 0.3305063

# Neuron Split (NS) starts

## BFS NS instance: NS

Time for backsubstitution: 0.00 seconds

### NS candidates at layer 1
type: A, layer: 1, pos: 83
type: A, layer: 1, pos: 29
type: A, layer: 1, pos: 220
type: A, layer: 1, pos: 15
type: A, layer: 1, pos: 30
type: A, layer: 1, pos: 190
type: A, layer: 1, pos: 175
type: A, layer: 1, pos: 221
type: A, layer: 1, pos: 161
type: A, layer: 1, pos: 62
type: A, layer: 1, pos: 178
type: A, layer: 1, pos: 233
type: A, layer: 1, pos: 103
type: A, layer: 1, pos: 188
type: A, layer: 1, pos: 136
type: A, layer: 1, pos: 176
type: A, layer: 1, pos: 251
type: A, layer: 1, pos: 66
type: A, layer: 1, pos: 201
type: A, layer: 1, pos: 93
type: A, layer: 1, pos: 71
type: A, layer: 1, pos: 235
type: A, layer: 1, pos: 198
type: A, layer: 1, pos: 172
type: A, layer: 1, pos: 219
type: A, layer: 1, pos: 177
type: A, layer: 1, pos: 68
type: A, layer: 1, pos: 1
type: A, layer: 1, pos: 120
type: A, layer: 1, pos: 0
type: A, layer: 1, pos: 182
type: A, layer: 1, pos: 13
type: A, layer: 1, pos: 245
type: A, layer: 1, pos: 37
type: A, layer: 1, pos: 183
type: A, layer: 1, pos: 166
type: A, layer: 1, pos: 240

Time for candidate selection: 0.23 seconds

### Candidate
type: A, layer: 1, pos: 83

## Relational analysis of NS_A1

### Relational analysis result of NS_A1
Status: Status.UNKNOWN
Output dim: 3, lower bound: -0.3305063, upper bound: 0.3303112
time: 1.27 seconds

## Relational analysis of NS_A2

### Relational analysis result of NS_A2
Status: Status.UNKNOWN
Output dim: 3, lower bound: -0.3303112, upper bound: 0.3303112
time: 1.68 seconds

## Summary of splitting at layer (split count: 0)
- Time for NS candidates: 3.19 seconds
NS_A1, status: Status.UNKNOWN, split count: 1, time: 3.19
Output dim: 3, lower bound: -0.3305063, upper bound: 0.3303112
NS_A2, status: Status.UNKNOWN, split count: 1, time: 3.19
Output dim: 3, lower bound: -0.3303112, upper bound: 0.3303112

## BFS NS instance: NS_A1

### Backsubstitution after applying NS history:
0: -0.4989560, 0.4343209, -0.5311284, 0.4579680, -0.9569240, 0.9654493
1: -0.3301082, 0.3608218, -0.3541678, 0.3831487, -0.7132568, 0.7149895
2: -0.4872135, 0.4428630, -0.5165144, 0.4707302, -0.9579437, 0.9593774
3: 0.7801661, 1.1486214, 0.7612032, 1.1563219, -0.3761558, 0.3874183
4: -0.4189468, 0.4281434, -0.4502595, 0.4534048, -0.8723516, 0.8784029
5: -0.3833302, 0.4400102, -0.4113419, 0.4600840, -0.8434142, 0.8513521
6: -0.4809147, 0.4975284, -0.5050698, 0.5267280, -1.0076427, 1.0025982
7: -0.4066606, 0.3595893, -0.4319701, 0.3913552, -0.7980158, 0.7915594
8: -0.3936677, 0.5123415, -0.4208334, 0.5425459, -0.9362135, 0.9331748
9: -0.4450092, 0.3498356, -0.4758850, 0.3760591, -0.8210683, 0.8257207

Time for backsubstitution: 1.66 seconds

### NS candidates at layer 1
type: B, layer: 1, pos: 83
type: B, layer: 1, pos: 29
type: B, layer: 1, pos: 220
type: B, layer: 1, pos: 15
type: B, layer: 1, pos: 30
type: B, layer: 1, pos: 190
type: B, layer: 1, pos: 175
type: B, layer: 1, pos: 221
type: B, layer: 1, pos: 161
type: B, layer: 1, pos: 62
type: B, layer: 1, pos: 178
type: B, layer: 1, pos: 233
type: B, layer: 1, pos: 103
type: B, layer: 1, pos: 188
type: B, layer: 1, pos: 136
type: B, layer: 1, pos: 176
type: B, layer: 1, pos: 251
type: B, layer: 1, pos: 66
type: B, layer: 1, pos: 201
type: B, layer: 1, pos: 93
type: B, layer: 1, pos: 71
type: B, layer: 1, pos: 235
type: B, layer: 1, pos: 198
type: B, layer: 1, pos: 172
type: B, layer: 1, pos: 219
type: B, layer: 1, pos: 1
type: B, layer: 1, pos: 177
type: B, layer: 1, pos: 68
type: B, layer: 1, pos: 120
type: B, layer: 1, pos: 0
type: B, layer: 1, pos: 182
type: B, layer: 1, pos: 13
type: B, layer: 1, pos: 245
type: B, layer: 1, pos: 37
type: B, layer: 1, pos: 166
type: B, layer: 1, pos: 183
type: B, layer: 1, pos: 240

Time for candidate selection: 0.15 seconds

### Candidate
type: B, layer: 1, pos: 83

## Relational analysis of NS_A1_B1

### Relational analysis result of NS_A1_B1
Status: Status.UNKNOWN
Output dim: 3, lower bound: -0.3303112, upper bound: 0.3303112
time: 1.11 seconds

## Relational analysis of NS_A1_B2

### Relational analysis result of NS_A1_B2
Status: Status.UNKNOWN
Output dim: 3, lower bound: -0.3303112, upper bound: 0.3303112
time: 1.22 seconds

## BFS NS instance: NS_A2

### Backsubstitution after applying NS history:
0: -0.5779052, 0.4932721, -0.5315576, 0.4582958, -1.0362010, 1.0248296
1: -0.3902119, 0.4162342, -0.3544971, 0.3834571, -0.7736689, 0.7707313
2: -0.5609355, 0.5101271, -0.5169256, 0.4710829, -1.0320184, 1.0270526
3: 0.7364339, 1.1685661, 0.7609799, 1.1564380, -0.4200041, 0.4075862
4: -0.4951320, 0.4911126, -0.4506671, 0.4537513, -0.9488833, 0.9417797
5: -0.4530188, 0.4892210, -0.4117240, 0.4603533, -0.9133722, 0.9009449
6: -0.5419909, 0.5697627, -0.5054137, 0.5271270, -1.0691180, 1.0751765
7: -0.4689539, 0.4369200, -0.4323127, 0.3917649, -0.8607188, 0.8692327
8: -0.4612342, 0.5861606, -0.4212043, 0.5429480, -1.0041822, 1.0073649
9: -0.5213108, 0.4136213, -0.4763000, 0.3764027, -0.8977135, 0.8899213

Time for backsubstitution: 1.70 seconds

### NS candidates at layer 1
type: B, layer: 1, pos: 83
type: B, layer: 1, pos: 29
type: B, layer: 1, pos: 220
type: B, layer: 1, pos: 15
type: B, layer: 1, pos: 30
type: B, layer: 1, pos: 190
type: B, layer: 1, pos: 175
type: B, layer: 1, pos: 221
type: B, layer: 1, pos: 161
type: B, layer: 1, pos: 62
type: B, layer: 1, pos: 178
type: B, layer: 1, pos: 233
type: B, layer: 1, pos: 103
type: B, layer: 1, pos: 188
type: B, layer: 1, pos: 136
type: B, layer: 1, pos: 176
type: B, layer: 1, pos: 251
type: B, layer: 1, pos: 66
type: B, layer: 1, pos: 201
type: B, layer: 1, pos: 93
type: B, layer: 1, pos: 71
type: B, layer: 1, pos: 235
type: B, layer: 1, pos: 198
type: B, layer: 1, pos: 172
type: B, layer: 1, pos: 219
type: B, layer: 1, pos: 1
type: B, layer: 1, pos: 177
type: B, layer: 1, pos: 68
type: B, layer: 1, pos: 120
type: B, layer: 1, pos: 0
type: B, layer: 1, pos: 182
type: B, layer: 1, pos: 13
type: B, layer: 1, pos: 245
type: B, layer: 1, pos: 37
type: B, layer: 1, pos: 166
type: B, layer: 1, pos: 183
type: B, layer: 1, pos: 240

Time for candidate selection: 0.16 seconds

### Candidate
type: B, layer: 1, pos: 83

## Relational analysis of NS_A2_B1

### Relational analysis result of NS_A2_B1
Status: Status.UNKNOWN
Output dim: 3, lower bound: -0.3303112, upper bound: 0.3303112
time: 1.20 seconds

## Relational analysis of NS_A2_B2

### Relational analysis result of NS_A2_B2
Status: Status.UNKNOWN
Output dim: 3, lower bound: -0.3303112, upper bound: 0.3303112
time: 1.30 seconds

## Summary of splitting at layer (split count: 1)
- Time for NS candidates: 4.36 seconds
NS_A1_B1, status: Status.UNKNOWN, split count: 2, time: 4.36
Output dim: 3, lower bound: -0.3303112, upper bound: 0.3303112
NS_A1_B2, status: Status.UNKNOWN, split count: 2, time: 4.36
Output dim: 3, lower bound: -0.3303112, upper bound: 0.3303112
NS_A2_B1, status: Status.UNKNOWN, split count: 2, time: 4.36
Output dim: 3, lower bound: -0.3303112, upper bound: 0.3303112
NS_A2_B2, status: Status.UNKNOWN, split count: 2, time: 4.36
Output dim: 3, lower bound: -0.3303112, upper bound: 0.3303112

## BFS NS instance: NS_A1_B1

### Backsubstitution after applying NS history:
0: -0.4989560, 0.4343209, -0.4989560, 0.4343209, -0.9332769, 0.9332769
1: -0.3301082, 0.3608218, -0.3301082, 0.3608218, -0.6909299, 0.6909299
2: -0.4872135, 0.4428630, -0.4872135, 0.4428630, -0.9300765, 0.9300765
3: 0.7801661, 1.1486214, 0.7801661, 1.1486214, -0.3684554, 0.3684554
4: -0.4189468, 0.4281434, -0.4189468, 0.4281434, -0.8470902, 0.8470902
5: -0.3833302, 0.4400102, -0.3833302, 0.4400102, -0.8233404, 0.8233404
6: -0.4809147, 0.4975284, -0.4809147, 0.4975284, -0.9784431, 0.9784431
7: -0.4066606, 0.3595893, -0.4066606, 0.3595893, -0.7662499, 0.7662499
8: -0.3936677, 0.5123415, -0.3936677, 0.5123415, -0.9060092, 0.9060092
9: -0.4450092, 0.3498356, -0.4450092, 0.3498356, -0.7948449, 0.7948449

Time for backsubstitution: 1.95 seconds

### NS candidates at layer 1
type: A, layer: 1, pos: 29
type: A, layer: 1, pos: 220
type: A, layer: 1, pos: 15
type: A, layer: 1, pos: 30
type: A, layer: 1, pos: 190
type: A, layer: 1, pos: 175
type: A, layer: 1, pos: 221
type: A, layer: 1, pos: 161
type: A, layer: 1, pos: 62
type: A, layer: 1, pos: 178
type: A, layer: 1, pos: 233
type: A, layer: 1, pos: 103
type: A, layer: 1, pos: 188
type: A, layer: 1, pos: 136
type: A, layer: 1, pos: 176
type: A, layer: 1, pos: 251
type: A, layer: 1, pos: 66
type: A, layer: 1, pos: 201
type: A, layer: 1, pos: 93
type: A, layer: 1, pos: 71
type: A, layer: 1, pos: 198
type: A, layer: 1, pos: 235
type: A, layer: 1, pos: 1
type: A, layer: 1, pos: 172
type: A, layer: 1, pos: 177
type: A, layer: 1, pos: 219
type: A, layer: 1, pos: 120
type: A, layer: 1, pos: 68
type: A, layer: 1, pos: 0
type: A, layer: 1, pos: 182
type: A, layer: 1, pos: 13
type: A, layer: 1, pos: 245
type: A, layer: 1, pos: 166
type: A, layer: 1, pos: 37
type: A, layer: 1, pos: 183
type: A, layer: 1, pos: 240

Time for candidate selection: 0.19 seconds

### Candidate
type: A, layer: 1, pos: 29

## Relational analysis of NS_A1_B1_A1

### Relational analysis result of NS_A1_B1_A1
Status: Status.UNKNOWN
Output dim: 3, lower bound: -0.3273759, upper bound: 0.3267132
time: 1.33 seconds

## Relational analysis of NS_A1_B1_A2

### Relational analysis result of NS_A1_B1_A2
Status: Status.UNKNOWN
Output dim: 3, lower bound: -0.3268922, upper bound: 0.3267132
time: 1.24 seconds

## BFS NS instance: NS_A1_B2

### Backsubstitution after applying NS history:
0: -0.4989560, 0.4343209, -0.5779052, 0.4932721, -0.9922282, 1.0122261
1: -0.3301082, 0.3608218, -0.3902119, 0.4162342, -0.7463424, 0.7510337
2: -0.4872135, 0.4428630, -0.5609355, 0.5101271, -0.9973406, 1.0037985
3: 0.7801661, 1.1486214, 0.7364339, 1.1685661, -0.3884000, 0.4121876
4: -0.4189468, 0.4281434, -0.4951320, 0.4911126, -0.9100595, 0.9232754
5: -0.3833302, 0.4400102, -0.4530188, 0.4892210, -0.8725511, 0.8930290
6: -0.4809147, 0.4975284, -0.5419909, 0.5697627, -1.0506774, 1.0395193
7: -0.4066606, 0.3595893, -0.4689539, 0.4369200, -0.8435806, 0.8285432
8: -0.3936677, 0.5123415, -0.4612342, 0.5861606, -0.9798282, 0.9735757
9: -0.4450092, 0.3498356, -0.5213108, 0.4136213, -0.8586305, 0.8711464

Time for backsubstitution: 1.66 seconds

### NS candidates at layer 1
type: A, layer: 1, pos: 29
type: A, layer: 1, pos: 220
type: A, layer: 1, pos: 15
type: A, layer: 1, pos: 30
type: A, layer: 1, pos: 190
type: A, layer: 1, pos: 175
type: A, layer: 1, pos: 221
type: A, layer: 1, pos: 161
type: A, layer: 1, pos: 62
type: A, layer: 1, pos: 178
type: A, layer: 1, pos: 233
type: A, layer: 1, pos: 103
type: A, layer: 1, pos: 188
type: A, layer: 1, pos: 136
type: A, layer: 1, pos: 176
type: A, layer: 1, pos: 251
type: A, layer: 1, pos: 66
type: A, layer: 1, pos: 201
type: A, layer: 1, pos: 93
type: A, layer: 1, pos: 71
type: A, layer: 1, pos: 198
type: A, layer: 1, pos: 235
type: A, layer: 1, pos: 1
type: A, layer: 1, pos: 172
type: A, layer: 1, pos: 177
type: A, layer: 1, pos: 219
type: A, layer: 1, pos: 120
type: A, layer: 1, pos: 68
type: A, layer: 1, pos: 0
type: A, layer: 1, pos: 182
type: A, layer: 1, pos: 13
type: A, layer: 1, pos: 245
type: A, layer: 1, pos: 166
type: A, layer: 1, pos: 37
type: A, layer: 1, pos: 183
type: A, layer: 1, pos: 240

Time for candidate selection: 0.15 seconds

### Candidate
type: A, layer: 1, pos: 29

## Relational analysis of NS_A1_B2_A1

### Relational analysis result of NS_A1_B2_A1
Status: Status.UNKNOWN
Output dim: 3, lower bound: -0.3273759, upper bound: 0.3267132
time: 1.27 seconds

## Relational analysis of NS_A1_B2_A2

### Relational analysis result of NS_A1_B2_A2
Status: Status.UNKNOWN
Output dim: 3, lower bound: -0.3268922, upper bound: 0.3267132
time: 1.15 seconds

## BFS NS instance: NS_A2_B1

### Backsubstitution after applying NS history:
0: -0.5779052, 0.4932721, -0.4989560, 0.4343209, -1.0122261, 0.9922282
1: -0.3902119, 0.4162342, -0.3301082, 0.3608218, -0.7510337, 0.7463424
2: -0.5609355, 0.5101271, -0.4872135, 0.4428630, -1.0037985, 0.9973406
3: 0.7364339, 1.1685661, 0.7801661, 1.1486214, -0.4121876, 0.3884000
4: -0.4951320, 0.4911126, -0.4189468, 0.4281434, -0.9232754, 0.9100595
5: -0.4530188, 0.4892210, -0.3833302, 0.4400102, -0.8930290, 0.8725511
6: -0.5419909, 0.5697627, -0.4809147, 0.4975284, -1.0395193, 1.0506774
7: -0.4689539, 0.4369200, -0.4066606, 0.3595893, -0.8285432, 0.8435806
8: -0.4612342, 0.5861606, -0.3936677, 0.5123415, -0.9735757, 0.9798282
9: -0.5213108, 0.4136213, -0.4450092, 0.3498356, -0.8711464, 0.8586305

Time for backsubstitution: 1.68 seconds

### NS candidates at layer 1
type: A, layer: 1, pos: 29
type: A, layer: 1, pos: 15
type: A, layer: 1, pos: 220
type: A, layer: 1, pos: 190
type: A, layer: 1, pos: 30
type: A, layer: 1, pos: 175
type: A, layer: 1, pos: 221
type: A, layer: 1, pos: 161
type: A, layer: 1, pos: 62
type: A, layer: 1, pos: 178
type: A, layer: 1, pos: 233
type: A, layer: 1, pos: 103
type: A, layer: 1, pos: 188
type: A, layer: 1, pos: 136
type: A, layer: 1, pos: 176
type: A, layer: 1, pos: 251
type: A, layer: 1, pos: 201
type: A, layer: 1, pos: 66
type: A, layer: 1, pos: 93
type: A, layer: 1, pos: 71
type: A, layer: 1, pos: 235
type: A, layer: 1, pos: 198
type: A, layer: 1, pos: 172
type: A, layer: 1, pos: 1
type: A, layer: 1, pos: 219
type: A, layer: 1, pos: 177
type: A, layer: 1, pos: 68
type: A, layer: 1, pos: 120
type: A, layer: 1, pos: 0
type: A, layer: 1, pos: 182
type: A, layer: 1, pos: 13
type: A, layer: 1, pos: 245
type: A, layer: 1, pos: 37
type: A, layer: 1, pos: 166
type: A, layer: 1, pos: 183
type: A, layer: 1, pos: 240

Time for candidate selection: 0.14 seconds

### Candidate
type: A, layer: 1, pos: 29

## Relational analysis of NS_A2_B1_A1

### Relational analysis result of NS_A2_B1_A1
Status: Status.UNKNOWN
Output dim: 3, lower bound: -0.3271649, upper bound: 0.3267132
time: 1.10 seconds

## Relational analysis of NS_A2_B1_A2

### Relational analysis result of NS_A2_B1_A2
Status: Status.UNKNOWN
Output dim: 3, lower bound: -0.3267132, upper bound: 0.3267132
time: 1.22 seconds

## BFS NS instance: NS_A2_B2

### Backsubstitution after applying NS history:
0: -0.5779052, 0.4932721, -0.5779052, 0.4932721, -1.0711772, 1.0711772
1: -0.3902119, 0.4162342, -0.3902119, 0.4162342, -0.8064461, 0.8064461
2: -0.5609355, 0.5101271, -0.5609355, 0.5101271, -1.0710626, 1.0710626
3: 0.7364339, 1.1685661, 0.7364339, 1.1685661, -0.4321322, 0.4321322
4: -0.4951320, 0.4911126, -0.4951320, 0.4911126, -0.9862446, 0.9862446
5: -0.4530188, 0.4892210, -0.4530188, 0.4892210, -0.9422398, 0.9422398
6: -0.5419909, 0.5697627, -0.5419909, 0.5697627, -1.1117537, 1.1117537
7: -0.4689539, 0.4369200, -0.4689539, 0.4369200, -0.9058739, 0.9058739
8: -0.4612342, 0.5861606, -0.4612342, 0.5861606, -1.0473948, 1.0473948
9: -0.5213108, 0.4136213, -0.5213108, 0.4136213, -0.9349321, 0.9349321

Time for backsubstitution: 1.92 seconds

### NS candidates at layer 1
type: A, layer: 1, pos: 29
type: A, layer: 1, pos: 15
type: A, layer: 1, pos: 220
type: A, layer: 1, pos: 190
type: A, layer: 1, pos: 30
type: A, layer: 1, pos: 175
type: A, layer: 1, pos: 221
type: A, layer: 1, pos: 161
type: A, layer: 1, pos: 62
type: A, layer: 1, pos: 178
type: A, layer: 1, pos: 233
type: A, layer: 1, pos: 103
type: A, layer: 1, pos: 188
type: A, layer: 1, pos: 136
type: A, layer: 1, pos: 176
type: A, layer: 1, pos: 251
type: A, layer: 1, pos: 201
type: A, layer: 1, pos: 66
type: A, layer: 1, pos: 93
type: A, layer: 1, pos: 71
type: A, layer: 1, pos: 235
type: A, layer: 1, pos: 198
type: A, layer: 1, pos: 172
type: A, layer: 1, pos: 1
type: A, layer: 1, pos: 219
type: A, layer: 1, pos: 177
type: A, layer: 1, pos: 68
type: A, layer: 1, pos: 120
type: A, layer: 1, pos: 0
type: A, layer: 1, pos: 182
type: A, layer: 1, pos: 13
type: A, layer: 1, pos: 245
type: A, layer: 1, pos: 37
type: A, layer: 1, pos: 166
type: A, layer: 1, pos: 183
type: A, layer: 1, pos: 240

Time for candidate selection: 0.17 seconds

### Candidate
type: A, layer: 1, pos: 29

## Relational analysis of NS_A2_B2_A1

### Relational analysis result of NS_A2_B2_A1
Status: Status.UNKNOWN
Output dim: 3, lower bound: -0.3271649, upper bound: 0.3267132
time: 1.29 seconds

## Relational analysis of NS_A2_B2_A2

### Relational analysis result of NS_A2_B2_A2
Status: Status.UNKNOWN
Output dim: 3, lower bound: -0.3267132, upper bound: 0.3267132
time: 1.11 seconds

## Summary of splitting at layer (split count: 2)
- Time for NS candidates: 4.50 seconds
NS_A1_B1_A1, status: Status.UNKNOWN, split count: 3, time: 4.50
Output dim: 3, lower bound: -0.3273759, upper bound: 0.3267132
NS_A1_B1_A2, status: Status.UNKNOWN, split count: 3, time: 4.50
Output dim: 3, lower bound: -0.3268922, upper bound: 0.3267132
NS_A1_B2_A1, status: Status.UNKNOWN, split count: 3, time: 4.50
Output dim: 3, lower bound: -0.3273759, upper bound: 0.3267132
NS_A1_B2_A2, status: Status.UNKNOWN, split count: 3, time: 4.50
Output dim: 3, lower bound: -0.3268922, upper bound: 0.3267132
NS_A2_B1_A1, status: Status.UNKNOWN, split count: 3, time: 4.50
Output dim: 3, lower bound: -0.3271649, upper bound: 0.3267132
NS_A2_B1_A2, status: Status.UNKNOWN, split count: 3, time: 4.50
Output dim: 3, lower bound: -0.3267132, upper bound: 0.3267132
NS_A2_B2_A1, status: Status.UNKNOWN, split count: 3, time: 4.50
Output dim: 3, lower bound: -0.3271649, upper bound: 0.3267132
NS_A2_B2_A2, status: Status.UNKNOWN, split count: 3, time: 4.50
Output dim: 3, lower bound: -0.3267132, upper bound: 0.3267132

## BFS NS instance: NS_A1_B1_A1

### Backsubstitution after applying NS history:
0: -0.2969490, 0.2755476, -0.4402137, 0.3899135, -0.6868625, 0.7157613
1: -0.1876454, 0.2096218, -0.2883425, 0.3197150, -0.5073604, 0.4979643
2: -0.3089357, 0.2545137, -0.4335209, 0.3908970, -0.6998327, 0.6880347
3: 0.8904049, 1.0996947, 0.8107187, 1.1336141, -0.2432091, 0.2889761
4: -0.2094393, 0.2789370, -0.3608077, 0.3848245, -0.5942637, 0.6397446
5: -0.2208162, 0.3047529, -0.3352770, 0.4033831, -0.6241993, 0.6400300
6: -0.3187222, 0.2987844, -0.4367090, 0.4405210, -0.7592431, 0.7354934
7: -0.2428993, 0.1723002, -0.3600132, 0.3049544, -0.5478538, 0.5323135
8: -0.2096012, 0.3157430, -0.3424704, 0.4560282, -0.6656293, 0.6582134
9: -0.2501553, 0.2001792, -0.3907135, 0.3061140, -0.5562693, 0.5908927

Time for backsubstitution: 1.78 seconds

### NS candidates at layer 1
type: B, layer: 1, pos: 220
type: B, layer: 1, pos: 15
type: B, layer: 1, pos: 30
type: B, layer: 1, pos: 175
type: B, layer: 1, pos: 190
type: B, layer: 1, pos: 221
type: B, layer: 1, pos: 29
type: B, layer: 1, pos: 161
type: B, layer: 1, pos: 62
type: B, layer: 1, pos: 178
type: B, layer: 1, pos: 233
type: B, layer: 1, pos: 103
type: B, layer: 1, pos: 136
type: B, layer: 1, pos: 188
type: B, layer: 1, pos: 176
type: B, layer: 1, pos: 251
type: B, layer: 1, pos: 66
type: B, layer: 1, pos: 93
type: B, layer: 1, pos: 201
type: B, layer: 1, pos: 71
type: B, layer: 1, pos: 198
type: B, layer: 1, pos: 235
type: B, layer: 1, pos: 1
type: B, layer: 1, pos: 177
type: B, layer: 1, pos: 219
type: B, layer: 1, pos: 172
type: B, layer: 1, pos: 120
type: B, layer: 1, pos: 68
type: B, layer: 1, pos: 0
type: B, layer: 1, pos: 182
type: B, layer: 1, pos: 13
type: B, layer: 1, pos: 245
type: B, layer: 1, pos: 166
type: B, layer: 1, pos: 240
type: B, layer: 1, pos: 37
type: B, layer: 1, pos: 183

Time for candidate selection: 0.16 seconds

### Candidate
type: B, layer: 1, pos: 220

## Relational analysis of NS_A1_B1_A1_B1

### Relational analysis result of NS_A1_B1_A1_B1
Status: Status.UNKNOWN
Output dim: 3, lower bound: -0.3148098, upper bound: 0.3215004
time: 1.53 seconds

## Relational analysis of NS_A1_B1_A1_B2

### Relational analysis result of NS_A1_B1_A1_B2
Status: Status.VERIFIED
Output dim: 3, lower bound: -0.3145860, upper bound: 0.3157829
time: 1.38 seconds

## BFS NS instance: NS_A1_B1_A2

### Backsubstitution after applying NS history:
0: -0.3718126, 0.3375633, -0.4500996, 0.3976302, -0.7694428, 0.7876629
1: -0.2396725, 0.2706760, -0.2953461, 0.3270362, -0.5667087, 0.5660222
2: -0.3747793, 0.3248672, -0.4430532, 0.3991996, -0.7739789, 0.7679204
3: 0.8483785, 1.1185884, 0.8061924, 1.1364830, -0.2881044, 0.3123960
4: -0.2902533, 0.3331347, -0.3701621, 0.3922749, -0.6825283, 0.7032968
5: -0.2803392, 0.3589308, -0.3434704, 0.4095869, -0.6899261, 0.7024012
6: -0.3845040, 0.3726339, -0.4447160, 0.4505739, -0.8350779, 0.8173499
7: -0.3064096, 0.2381508, -0.3677219, 0.3137425, -0.6201521, 0.6058728
8: -0.2810340, 0.3885365, -0.3513905, 0.4656196, -0.7466536, 0.7399271
9: -0.3228636, 0.2575039, -0.3997307, 0.3130678, -0.6359314, 0.6572346

Time for backsubstitution: 2.05 seconds

### NS candidates at layer 1
type: B, layer: 1, pos: 220
type: B, layer: 1, pos: 15
type: B, layer: 1, pos: 30
type: B, layer: 1, pos: 190
type: B, layer: 1, pos: 175
type: B, layer: 1, pos: 221
type: B, layer: 1, pos: 29
type: B, layer: 1, pos: 161
type: B, layer: 1, pos: 62
type: B, layer: 1, pos: 178
type: B, layer: 1, pos: 233
type: B, layer: 1, pos: 103
type: B, layer: 1, pos: 136
type: B, layer: 1, pos: 188
type: B, layer: 1, pos: 176
type: B, layer: 1, pos: 251
type: B, layer: 1, pos: 66
type: B, layer: 1, pos: 201
type: B, layer: 1, pos: 93
type: B, layer: 1, pos: 71
type: B, layer: 1, pos: 198
type: B, layer: 1, pos: 235
type: B, layer: 1, pos: 1
type: B, layer: 1, pos: 177
type: B, layer: 1, pos: 172
type: B, layer: 1, pos: 219
type: B, layer: 1, pos: 120
type: B, layer: 1, pos: 68
type: B, layer: 1, pos: 0
type: B, layer: 1, pos: 182
type: B, layer: 1, pos: 13
type: B, layer: 1, pos: 245
type: B, layer: 1, pos: 166
type: B, layer: 1, pos: 37
type: B, layer: 1, pos: 240
type: B, layer: 1, pos: 183

Time for candidate selection: 0.23 seconds

### Candidate
type: B, layer: 1, pos: 220

## Relational analysis of NS_A1_B1_A2_B1

### Relational analysis result of NS_A1_B1_A2_B1
Status: Status.UNKNOWN
Output dim: 3, lower bound: -0.3164355, upper bound: 0.3223389
time: 1.25 seconds

## Relational analysis of NS_A1_B1_A2_B2

### Relational analysis result of NS_A1_B1_A2_B2
Status: Status.VERIFIED
Output dim: 3, lower bound: -0.3163202, upper bound: 0.3163202
time: 1.25 seconds

## BFS NS instance: NS_A1_B2_A1

### Backsubstitution after applying NS history:
0: -0.2969490, 0.2755476, -0.5152947, 0.4461309, -0.7430799, 0.7908423
1: -0.1876454, 0.2096218, -0.3419876, 0.3720972, -0.5597426, 0.5516093
2: -0.3089357, 0.2545137, -0.5020123, 0.4570911, -0.7660269, 0.7565261
3: 0.8904049, 1.0996947, 0.7702841, 1.1525100, -0.2621051, 0.3294106
4: -0.2094393, 0.2789370, -0.4349009, 0.4407123, -0.6501516, 0.7138379
5: -0.2208162, 0.3047529, -0.3973655, 0.4501685, -0.6709847, 0.7021184
6: -0.3187222, 0.2987844, -0.4928842, 0.5123028, -0.8310249, 0.7916687
7: -0.2428993, 0.1723002, -0.4193979, 0.3756782, -0.6185775, 0.5916981
8: -0.2096012, 0.3157430, -0.4073055, 0.5277054, -0.7373066, 0.7230484
9: -0.2501553, 0.2001792, -0.4603017, 0.3632245, -0.6133798, 0.6604809

Time for backsubstitution: 1.98 seconds

### NS candidates at layer 1
type: B, layer: 1, pos: 220
type: B, layer: 1, pos: 15
type: B, layer: 1, pos: 30
type: B, layer: 1, pos: 190
type: B, layer: 1, pos: 175
type: B, layer: 1, pos: 29
type: B, layer: 1, pos: 221
type: B, layer: 1, pos: 161
type: B, layer: 1, pos: 62
type: B, layer: 1, pos: 178
type: B, layer: 1, pos: 233
type: B, layer: 1, pos: 103
type: B, layer: 1, pos: 188
type: B, layer: 1, pos: 136
type: B, layer: 1, pos: 176
type: B, layer: 1, pos: 251
type: B, layer: 1, pos: 201
type: B, layer: 1, pos: 66
type: B, layer: 1, pos: 93
type: B, layer: 1, pos: 71
type: B, layer: 1, pos: 235
type: B, layer: 1, pos: 198
type: B, layer: 1, pos: 1
type: B, layer: 1, pos: 177
type: B, layer: 1, pos: 172
type: B, layer: 1, pos: 219
type: B, layer: 1, pos: 120
type: B, layer: 1, pos: 68
type: B, layer: 1, pos: 0
type: B, layer: 1, pos: 182
type: B, layer: 1, pos: 13
type: B, layer: 1, pos: 245
type: B, layer: 1, pos: 37
type: B, layer: 1, pos: 166
type: B, layer: 1, pos: 183
type: B, layer: 1, pos: 240

Time for candidate selection: 0.23 seconds

### Candidate
type: B, layer: 1, pos: 220

## Relational analysis of NS_A1_B2_A1_B1

### Relational analysis result of NS_A1_B2_A1_B1
Status: Status.UNKNOWN
Output dim: 3, lower bound: -0.3148088, upper bound: 0.3213939
time: 1.78 seconds

## Relational analysis of NS_A1_B2_A1_B2

### Relational analysis result of NS_A1_B2_A1_B2
Status: Status.VERIFIED
Output dim: 3, lower bound: -0.3145823, upper bound: 0.3157761
time: 1.56 seconds

## BFS NS instance: NS_A1_B2_A2

### Backsubstitution after applying NS history:
0: -0.3718126, 0.3375633, -0.5256237, 0.4541863, -0.8259990, 0.8631870
1: -0.2396725, 0.2706760, -0.3499980, 0.3796693, -0.6193419, 0.6206740
2: -0.3747793, 0.3248672, -0.5124507, 0.4653453, -0.8401247, 0.8373179
3: 0.8483785, 1.1185884, 0.7651343, 1.1555773, -0.3071988, 0.3534541
4: -0.2902533, 0.3331347, -0.4445953, 0.4492094, -0.7394627, 0.7777300
5: -0.2803392, 0.3589308, -0.4067204, 0.4567230, -0.7370623, 0.7656512
6: -0.3845040, 0.3726339, -0.5014983, 0.5221134, -0.9066173, 0.8741322
7: -0.3064096, 0.2381508, -0.4277053, 0.3854311, -0.6918406, 0.6658561
8: -0.2810340, 0.3885365, -0.4164434, 0.5373636, -0.8183976, 0.8049799
9: -0.3228636, 0.2575039, -0.4700886, 0.3715566, -0.6944202, 0.7275925

Time for backsubstitution: 2.16 seconds

### NS candidates at layer 1
type: B, layer: 1, pos: 220
type: B, layer: 1, pos: 15
type: B, layer: 1, pos: 30
type: B, layer: 1, pos: 190
type: B, layer: 1, pos: 175
type: B, layer: 1, pos: 221
type: B, layer: 1, pos: 29
type: B, layer: 1, pos: 161
type: B, layer: 1, pos: 62
type: B, layer: 1, pos: 178
type: B, layer: 1, pos: 233
type: B, layer: 1, pos: 103
type: B, layer: 1, pos: 188
type: B, layer: 1, pos: 136
type: B, layer: 1, pos: 176
type: B, layer: 1, pos: 251
type: B, layer: 1, pos: 201
type: B, layer: 1, pos: 66
type: B, layer: 1, pos: 93
type: B, layer: 1, pos: 71
type: B, layer: 1, pos: 235
type: B, layer: 1, pos: 198
type: B, layer: 1, pos: 1
type: B, layer: 1, pos: 172
type: B, layer: 1, pos: 177
type: B, layer: 1, pos: 219
type: B, layer: 1, pos: 120
type: B, layer: 1, pos: 68
type: B, layer: 1, pos: 0
type: B, layer: 1, pos: 182
type: B, layer: 1, pos: 13
type: B, layer: 1, pos: 245
type: B, layer: 1, pos: 37
type: B, layer: 1, pos: 166
type: B, layer: 1, pos: 183
type: B, layer: 1, pos: 240

Time for candidate selection: 0.23 seconds

### Candidate
type: B, layer: 1, pos: 220

## Relational analysis of NS_A1_B2_A2_B1

### Relational analysis result of NS_A1_B2_A2_B1
Status: Status.UNKNOWN
Output dim: 3, lower bound: -0.3164355, upper bound: 0.3222042
time: 1.60 seconds

## Relational analysis of NS_A1_B2_A2_B2

### Relational analysis result of NS_A1_B2_A2_B2
Status: Status.VERIFIED
Output dim: 3, lower bound: -0.3163152, upper bound: 0.3163053
time: 1.23 seconds

## BFS NS instance: NS_A2_B1_A1

### Backsubstitution after applying NS history:
0: -0.3549846, 0.3238278, -0.4402137, 0.3899135, -0.7448981, 0.7640415
1: -0.2278231, 0.2568449, -0.2883425, 0.3197150, -0.5475382, 0.5451875
2: -0.3584748, 0.3102523, -0.4335209, 0.3908970, -0.7493718, 0.7437732
3: 0.8560573, 1.1135459, 0.8107187, 1.1336141, -0.2775568, 0.3028272
4: -0.2741081, 0.3194192, -0.3608077, 0.3848245, -0.6589325, 0.6802269
5: -0.2658185, 0.3474402, -0.3352770, 0.4033831, -0.6692016, 0.6827172
6: -0.3694423, 0.3551175, -0.4367090, 0.4405210, -0.8099633, 0.7918265
7: -0.2924380, 0.2233330, -0.3600132, 0.3049544, -0.5973924, 0.5833462
8: -0.2645506, 0.3727035, -0.3424704, 0.4560282, -0.7205788, 0.7151740
9: -0.3065494, 0.2454274, -0.3907135, 0.3061140, -0.6126634, 0.6361409

Time for backsubstitution: 1.66 seconds

### NS candidates at layer 1
type: B, layer: 1, pos: 220
type: B, layer: 1, pos: 15
type: B, layer: 1, pos: 30
type: B, layer: 1, pos: 175
type: B, layer: 1, pos: 190
type: B, layer: 1, pos: 221
type: B, layer: 1, pos: 29
type: B, layer: 1, pos: 161
type: B, layer: 1, pos: 62
type: B, layer: 1, pos: 178
type: B, layer: 1, pos: 233
type: B, layer: 1, pos: 103
type: B, layer: 1, pos: 136
type: B, layer: 1, pos: 188
type: B, layer: 1, pos: 176
type: B, layer: 1, pos: 251
type: B, layer: 1, pos: 66
type: B, layer: 1, pos: 93
type: B, layer: 1, pos: 201
type: B, layer: 1, pos: 71
type: B, layer: 1, pos: 198
type: B, layer: 1, pos: 235
type: B, layer: 1, pos: 1
type: B, layer: 1, pos: 177
type: B, layer: 1, pos: 219
type: B, layer: 1, pos: 172
type: B, layer: 1, pos: 120
type: B, layer: 1, pos: 68
type: B, layer: 1, pos: 0
type: B, layer: 1, pos: 182
type: B, layer: 1, pos: 13
type: B, layer: 1, pos: 245
type: B, layer: 1, pos: 166
type: B, layer: 1, pos: 240
type: B, layer: 1, pos: 37
type: B, layer: 1, pos: 183

Time for candidate selection: 0.16 seconds

### Candidate
type: B, layer: 1, pos: 220

## Relational analysis of NS_A2_B1_A1_B1

### Relational analysis result of NS_A2_B1_A1_B1
Status: Status.UNKNOWN
Output dim: 3, lower bound: -0.3148023, upper bound: 0.3215003
time: 2.78 seconds

## Relational analysis of NS_A2_B1_A1_B2

### Relational analysis result of NS_A2_B1_A1_B2
Status: Status.VERIFIED
Output dim: 3, lower bound: -0.3145788, upper bound: 0.3157809
time: 1.51 seconds

## BFS NS instance: NS_A2_B1_A2

### Backsubstitution after applying NS history:
0: -0.4399982, 0.3903420, -0.4500996, 0.3976302, -0.8376284, 0.8404416
1: -0.2882589, 0.3201866, -0.2953461, 0.3270362, -0.6152951, 0.6155328
2: -0.4345021, 0.3898732, -0.4430532, 0.3991996, -0.8337017, 0.8329264
3: 0.8101259, 1.1342641, 0.8061924, 1.1364830, -0.3263570, 0.3280717
4: -0.3601486, 0.3850370, -0.3701621, 0.3922749, -0.7524235, 0.7551991
5: -0.3355809, 0.4035141, -0.3434704, 0.4095869, -0.7451677, 0.7469845
6: -0.4374112, 0.4410149, -0.4447160, 0.4505739, -0.8879851, 0.8857310
7: -0.3599446, 0.3045519, -0.3677219, 0.3137425, -0.6736871, 0.6722739
8: -0.3427046, 0.4560596, -0.3513905, 0.4656196, -0.8083242, 0.8074501
9: -0.3901408, 0.3065187, -0.3997307, 0.3130678, -0.7032086, 0.7062495

Time for backsubstitution: 1.71 seconds

### NS candidates at layer 1
type: B, layer: 1, pos: 220
type: B, layer: 1, pos: 15
type: B, layer: 1, pos: 30
type: B, layer: 1, pos: 190
type: B, layer: 1, pos: 175
type: B, layer: 1, pos: 221
type: B, layer: 1, pos: 29
type: B, layer: 1, pos: 161
type: B, layer: 1, pos: 62
type: B, layer: 1, pos: 178
type: B, layer: 1, pos: 233
type: B, layer: 1, pos: 103
type: B, layer: 1, pos: 136
type: B, layer: 1, pos: 188
type: B, layer: 1, pos: 176
type: B, layer: 1, pos: 251
type: B, layer: 1, pos: 66
type: B, layer: 1, pos: 201
type: B, layer: 1, pos: 93
type: B, layer: 1, pos: 71
type: B, layer: 1, pos: 198
type: B, layer: 1, pos: 235
type: B, layer: 1, pos: 1
type: B, layer: 1, pos: 177
type: B, layer: 1, pos: 172
type: B, layer: 1, pos: 219
type: B, layer: 1, pos: 120
type: B, layer: 1, pos: 68
type: B, layer: 1, pos: 0
type: B, layer: 1, pos: 182
type: B, layer: 1, pos: 13
type: B, layer: 1, pos: 245
type: B, layer: 1, pos: 166
type: B, layer: 1, pos: 37
type: B, layer: 1, pos: 240
type: B, layer: 1, pos: 183

Time for candidate selection: 0.15 seconds

### Candidate
type: B, layer: 1, pos: 220

## Relational analysis of NS_A2_B1_A2_B1

### Relational analysis result of NS_A2_B1_A2_B1
Status: Status.UNKNOWN
Output dim: 3, lower bound: -0.3164216, upper bound: 0.3223389
time: 1.27 seconds

## Relational analysis of NS_A2_B1_A2_B2

### Relational analysis result of NS_A2_B1_A2_B2
Status: Status.VERIFIED
Output dim: 3, lower bound: -0.3163053, upper bound: 0.3163152
time: 1.23 seconds

## BFS NS instance: NS_A2_B2_A1

### Backsubstitution after applying NS history:
0: -0.3549846, 0.3238278, -0.5152947, 0.4461309, -0.8011155, 0.8391225
1: -0.2278231, 0.2568449, -0.3419876, 0.3720972, -0.5999203, 0.5988325
2: -0.3584748, 0.3102523, -0.5020123, 0.4570911, -0.8155659, 0.8122647
3: 0.8560573, 1.1135459, 0.7702841, 1.1525100, -0.2964528, 0.3432618
4: -0.2741081, 0.3194192, -0.4349009, 0.4407123, -0.7148204, 0.7543201
5: -0.2658185, 0.3474402, -0.3973655, 0.4501685, -0.7159870, 0.7448056
6: -0.3694423, 0.3551175, -0.4928842, 0.5123028, -0.8817451, 0.8480017
7: -0.2924380, 0.2233330, -0.4193979, 0.3756782, -0.6681162, 0.6427310
8: -0.2645506, 0.3727035, -0.4073055, 0.5277054, -0.7922560, 0.7800090
9: -0.3065494, 0.2454274, -0.4603017, 0.3632245, -0.6697739, 0.7057290

Time for backsubstitution: 2.15 seconds

### NS candidates at layer 1
type: B, layer: 1, pos: 220
type: B, layer: 1, pos: 15
type: B, layer: 1, pos: 30
type: B, layer: 1, pos: 190
type: B, layer: 1, pos: 175
type: B, layer: 1, pos: 29
type: B, layer: 1, pos: 221
type: B, layer: 1, pos: 161
type: B, layer: 1, pos: 62
type: B, layer: 1, pos: 178
type: B, layer: 1, pos: 233
type: B, layer: 1, pos: 103
type: B, layer: 1, pos: 188
type: B, layer: 1, pos: 136
type: B, layer: 1, pos: 176
type: B, layer: 1, pos: 251
type: B, layer: 1, pos: 201
type: B, layer: 1, pos: 66
type: B, layer: 1, pos: 93
type: B, layer: 1, pos: 71
type: B, layer: 1, pos: 235
type: B, layer: 1, pos: 198
type: B, layer: 1, pos: 1
type: B, layer: 1, pos: 177
type: B, layer: 1, pos: 172
type: B, layer: 1, pos: 219
type: B, layer: 1, pos: 120
type: B, layer: 1, pos: 68
type: B, layer: 1, pos: 0
type: B, layer: 1, pos: 182
type: B, layer: 1, pos: 13
type: B, layer: 1, pos: 245
type: B, layer: 1, pos: 37
type: B, layer: 1, pos: 166
type: B, layer: 1, pos: 183
type: B, layer: 1, pos: 240

Time for candidate selection: 0.23 seconds

### Candidate
type: B, layer: 1, pos: 220

## Relational analysis of NS_A2_B2_A1_B1

### Relational analysis result of NS_A2_B2_A1_B1
Status: Status.UNKNOWN
Output dim: 3, lower bound: -0.3148019, upper bound: 0.3214073
time: 1.62 seconds

## Relational analysis of NS_A2_B2_A1_B2

### Relational analysis result of NS_A2_B2_A1_B2
Status: Status.VERIFIED
Output dim: 3, lower bound: -0.3145784, upper bound: 0.3157757
time: 1.56 seconds

## BFS NS instance: NS_A2_B2_A2

### Backsubstitution after applying NS history:
0: -0.4399982, 0.3903420, -0.5256237, 0.4541863, -0.8941846, 0.9159657
1: -0.2882589, 0.3201866, -0.3499980, 0.3796693, -0.6679282, 0.6701846
2: -0.4345021, 0.3898732, -0.5124507, 0.4653453, -0.8998474, 0.9023239
3: 0.8101259, 1.1342641, 0.7651343, 1.1555773, -0.3454514, 0.3691298
4: -0.3601486, 0.3850370, -0.4445953, 0.4492094, -0.8093580, 0.8296323
5: -0.3355809, 0.4035141, -0.4067204, 0.4567230, -0.7923039, 0.8102345
6: -0.4374112, 0.4410149, -0.5014983, 0.5221134, -0.9595246, 0.9425133
7: -0.3599446, 0.3045519, -0.4277053, 0.3854311, -0.7453756, 0.7322572
8: -0.3427046, 0.4560596, -0.4164434, 0.5373636, -0.8800682, 0.8725029
9: -0.3901408, 0.3065187, -0.4700886, 0.3715566, -0.7616974, 0.7766073

Time for backsubstitution: 1.84 seconds

### NS candidates at layer 1
type: B, layer: 1, pos: 220
type: B, layer: 1, pos: 15
type: B, layer: 1, pos: 30
type: B, layer: 1, pos: 190
type: B, layer: 1, pos: 175
type: B, layer: 1, pos: 221
type: B, layer: 1, pos: 29
type: B, layer: 1, pos: 161
type: B, layer: 1, pos: 62
type: B, layer: 1, pos: 178
type: B, layer: 1, pos: 233
type: B, layer: 1, pos: 103
type: B, layer: 1, pos: 188
type: B, layer: 1, pos: 136
type: B, layer: 1, pos: 176
type: B, layer: 1, pos: 251
type: B, layer: 1, pos: 201
type: B, layer: 1, pos: 66
type: B, layer: 1, pos: 93
type: B, layer: 1, pos: 71
type: B, layer: 1, pos: 235
type: B, layer: 1, pos: 198
type: B, layer: 1, pos: 1
type: B, layer: 1, pos: 172
type: B, layer: 1, pos: 177
type: B, layer: 1, pos: 219
type: B, layer: 1, pos: 120
type: B, layer: 1, pos: 68
type: B, layer: 1, pos: 0
type: B, layer: 1, pos: 182
type: B, layer: 1, pos: 13
type: B, layer: 1, pos: 245
type: B, layer: 1, pos: 37
type: B, layer: 1, pos: 166
type: B, layer: 1, pos: 183
type: B, layer: 1, pos: 240

Time for candidate selection: 0.16 seconds

### Candidate
type: B, layer: 1, pos: 220

## Relational analysis of NS_A2_B2_A2_B1

### Relational analysis result of NS_A2_B2_A2_B1
Status: Status.UNKNOWN
Output dim: 3, lower bound: -0.3164216, upper bound: 0.3222081
time: 1.40 seconds

## Relational analysis of NS_A2_B2_A2_B2

### Relational analysis result of NS_A2_B2_A2_B2
Status: Status.VERIFIED
Output dim: 3, lower bound: -0.3163047, upper bound: 0.3163047
time: 1.41 seconds

## Summary of splitting at layer (split count: 3)
- Time for NS candidates: 4.83 seconds
NS_A1_B1_A1_B1, status: Status.UNKNOWN, split count: 4, time: 4.83
Output dim: 3, lower bound: -0.3148098, upper bound: 0.3215004
NS_A1_B1_A1_B2, status: Status.VERIFIED, split count: 4, time: 4.83
Output dim: 3, lower bound: -0.3145860, upper bound: 0.3157829
NS_A1_B1_A2_B1, status: Status.UNKNOWN, split count: 4, time: 4.83
Output dim: 3, lower bound: -0.3164355, upper bound: 0.3223389
NS_A1_B1_A2_B2, status: Status.VERIFIED, split count: 4, time: 4.83
Output dim: 3, lower bound: -0.3163202, upper bound: 0.3163202
NS_A1_B2_A1_B1, status: Status.UNKNOWN, split count: 4, time: 4.83
Output dim: 3, lower bound: -0.3148088, upper bound: 0.3213939
NS_A1_B2_A1_B2, status: Status.VERIFIED, split count: 4, time: 4.83
Output dim: 3, lower bound: -0.3145823, upper bound: 0.3157761
NS_A1_B2_A2_B1, status: Status.UNKNOWN, split count: 4, time: 4.83
Output dim: 3, lower bound: -0.3164355, upper bound: 0.3222042
NS_A1_B2_A2_B2, status: Status.VERIFIED, split count: 4, time: 4.83
Output dim: 3, lower bound: -0.3163152, upper bound: 0.3163053
NS_A2_B1_A1_B1, status: Status.UNKNOWN, split count: 4, time: 4.83
Output dim: 3, lower bound: -0.3148023, upper bound: 0.3215003
NS_A2_B1_A1_B2, status: Status.VERIFIED, split count: 4, time: 4.83
Output dim: 3, lower bound: -0.3145788, upper bound: 0.3157809
NS_A2_B1_A2_B1, status: Status.UNKNOWN, split count: 4, time: 4.83
Output dim: 3, lower bound: -0.3164216, upper bound: 0.3223389
NS_A2_B1_A2_B2, status: Status.VERIFIED, split count: 4, time: 4.83
Output dim: 3, lower bound: -0.3163053, upper bound: 0.3163152
NS_A2_B2_A1_B1, status: Status.UNKNOWN, split count: 4, time: 4.83
Output dim: 3, lower bound: -0.3148019, upper bound: 0.3214073
NS_A2_B2_A1_B2, status: Status.VERIFIED, split count: 4, time: 4.83
Output dim: 3, lower bound: -0.3145784, upper bound: 0.3157757
NS_A2_B2_A2_B1, status: Status.UNKNOWN, split count: 4, time: 4.83
Output dim: 3, lower bound: -0.3164216, upper bound: 0.3222081
NS_A2_B2_A2_B2, status: Status.VERIFIED, split count: 4, time: 4.83
Output dim: 3, lower bound: -0.3163047, upper bound: 0.3163047

## BFS NS instance: NS_A1_B1_A1_B1

### Backsubstitution after applying NS history:
0: -0.2956372, 0.2742563, -0.3920762, 0.3538485, -0.6494857, 0.6663325
1: -0.1867329, 0.2083831, -0.2540990, 0.2864872, -0.4732201, 0.4624820
2: -0.3077389, 0.2531745, -0.3934389, 0.3441919, -0.6519308, 0.6466134
3: 0.8913908, 1.0993620, 0.8400056, 1.1235851, -0.2321943, 0.2593564
4: -0.2078079, 0.2780033, -0.3111942, 0.3491313, -0.5569392, 0.5891976
5: -0.2197553, 0.3037720, -0.2971888, 0.3725173, -0.5922726, 0.6009608
6: -0.3174028, 0.2974916, -0.4011486, 0.3937304, -0.7111331, 0.6986402
7: -0.2417928, 0.1709570, -0.3231278, 0.2570808, -0.4988736, 0.4940848
8: -0.2082485, 0.3143705, -0.3003026, 0.4082607, -0.6165092, 0.6146731
9: -0.2489048, 0.1989274, -0.3433677, 0.2715535, -0.5204583, 0.5422950

Time for backsubstitution: 2.04 seconds

### NS candidates at layer 1
type: A, layer: 1, pos: 220
type: A, layer: 1, pos: 15
type: A, layer: 1, pos: 175
type: A, layer: 1, pos: 30
type: A, layer: 1, pos: 221
type: A, layer: 1, pos: 62
type: A, layer: 1, pos: 178
type: A, layer: 1, pos: 233
type: A, layer: 1, pos: 161
type: A, layer: 1, pos: 136
type: A, layer: 1, pos: 103
type: A, layer: 1, pos: 190
type: A, layer: 1, pos: 251
type: A, layer: 1, pos: 93
type: A, layer: 1, pos: 71
type: A, layer: 1, pos: 176
type: A, layer: 1, pos: 1
type: A, layer: 1, pos: 188
type: A, layer: 1, pos: 66
type: A, layer: 1, pos: 201
type: A, layer: 1, pos: 177
type: A, layer: 1, pos: 198
type: A, layer: 1, pos: 13
type: A, layer: 1, pos: 120
type: A, layer: 1, pos: 219
type: A, layer: 1, pos: 235
type: A, layer: 1, pos: 182
type: A, layer: 1, pos: 0
type: A, layer: 1, pos: 172
type: A, layer: 1, pos: 68
type: A, layer: 1, pos: 245
type: A, layer: 1, pos: 37
type: A, layer: 1, pos: 183
type: A, layer: 1, pos: 240
type: A, layer: 1, pos: 166

Time for candidate selection: 0.23 seconds

### Candidate
type: A, layer: 1, pos: 220

## Relational analysis of NS_A1_B1_A1_B1_A1

### Relational analysis result of NS_A1_B1_A1_B1_A1
Status: Status.VERIFIED
Output dim: 3, lower bound: -0.3145860, upper bound: 0.3157829
time: 1.40 seconds

## Relational analysis of NS_A1_B1_A1_B1_A2

### Relational analysis result of NS_A1_B1_A1_B1_A2
Status: Status.VERIFIED
Output dim: 3, lower bound: -0.3145860, upper bound: 0.3157829
time: 1.48 seconds

## BFS NS instance: NS_A1_B1_A2_B1

### Backsubstitution after applying NS history:
0: -0.3698886, 0.3360421, -0.4005199, 0.3607367, -0.7306253, 0.7365620
1: -0.2382730, 0.2692519, -0.2600246, 0.2931717, -0.5314447, 0.5292765
2: -0.3731204, 0.3228899, -0.4014205, 0.3518343, -0.7249547, 0.7243104
3: 0.8496231, 1.1181831, 0.8358310, 1.1259507, -0.2763276, 0.2823521
4: -0.2881854, 0.3316317, -0.3196350, 0.3558160, -0.6440014, 0.6512668
5: -0.2787703, 0.3576140, -0.3042822, 0.3782081, -0.6569784, 0.6618962
6: -0.3830273, 0.3706621, -0.4082978, 0.4026270, -0.7856542, 0.7789599
7: -0.3048434, 0.2361929, -0.3299913, 0.2649111, -0.5697545, 0.5661841
8: -0.2792376, 0.3866141, -0.3083292, 0.4166488, -0.6958864, 0.6949433
9: -0.3208500, 0.2560733, -0.3516247, 0.2775913, -0.5984412, 0.6076980

Time for backsubstitution: 2.07 seconds

### NS candidates at layer 1
type: A, layer: 1, pos: 220
type: A, layer: 1, pos: 15
type: A, layer: 1, pos: 30
type: A, layer: 1, pos: 175
type: A, layer: 1, pos: 221
type: A, layer: 1, pos: 190
type: A, layer: 1, pos: 62
type: A, layer: 1, pos: 161
type: A, layer: 1, pos: 178
type: A, layer: 1, pos: 233
type: A, layer: 1, pos: 103
type: A, layer: 1, pos: 136
type: A, layer: 1, pos: 188
type: A, layer: 1, pos: 251
type: A, layer: 1, pos: 176
type: A, layer: 1, pos: 93
type: A, layer: 1, pos: 71
type: A, layer: 1, pos: 66
type: A, layer: 1, pos: 201
type: A, layer: 1, pos: 1
type: A, layer: 1, pos: 177
type: A, layer: 1, pos: 198
type: A, layer: 1, pos: 235
type: A, layer: 1, pos: 219
type: A, layer: 1, pos: 120
type: A, layer: 1, pos: 13
type: A, layer: 1, pos: 172
type: A, layer: 1, pos: 68
type: A, layer: 1, pos: 0
type: A, layer: 1, pos: 182
type: A, layer: 1, pos: 245
type: A, layer: 1, pos: 166
type: A, layer: 1, pos: 240
type: A, layer: 1, pos: 37
type: A, layer: 1, pos: 183

Time for candidate selection: 0.22 seconds

### Candidate
type: A, layer: 1, pos: 220

## Relational analysis of NS_A1_B1_A2_B1_A1

### Relational analysis result of NS_A1_B1_A2_B1_A1
Status: Status.VERIFIED
Output dim: 3, lower bound: -0.3163202, upper bound: 0.3163202
time: 1.28 seconds

## Relational analysis of NS_A1_B1_A2_B1_A2

### Relational analysis result of NS_A1_B1_A2_B1_A2
Status: Status.VERIFIED
Output dim: 3, lower bound: -0.3163202, upper bound: 0.3163202
time: 1.21 seconds

## BFS NS instance: NS_A1_B2_A1_B1

### Backsubstitution after applying NS history:
0: -0.2956372, 0.2742563, -0.4633013, 0.4079013, -0.7035385, 0.7375576
1: -0.1867329, 0.2083831, -0.3046660, 0.3368634, -0.5235963, 0.5130491
2: -0.3077389, 0.2531745, -0.4558421, 0.4101121, -0.7178509, 0.7090166
3: 0.8913908, 1.0993620, 0.8013147, 1.1402977, -0.2489069, 0.2980473
4: -0.2078079, 0.2780033, -0.3825811, 0.4022008, -0.6100087, 0.6605844
5: -0.2197553, 0.3037720, -0.3543988, 0.4177817, -0.6375370, 0.6581708
6: -0.3174028, 0.2974916, -0.4554920, 0.4639602, -0.7813630, 0.7529836
7: -0.2417928, 0.1709570, -0.3780777, 0.3251879, -0.5669807, 0.5490347
8: -0.2082485, 0.3143705, -0.3633014, 0.4782517, -0.6865002, 0.6776719
9: -0.2489048, 0.1989274, -0.4117097, 0.3220733, -0.5709780, 0.6106371

Time for backsubstitution: 2.08 seconds

### NS candidates at layer 1
type: A, layer: 1, pos: 220
type: A, layer: 1, pos: 15
type: A, layer: 1, pos: 175
type: A, layer: 1, pos: 30
type: A, layer: 1, pos: 221
type: A, layer: 1, pos: 62
type: A, layer: 1, pos: 178
type: A, layer: 1, pos: 233
type: A, layer: 1, pos: 161
type: A, layer: 1, pos: 136
type: A, layer: 1, pos: 103
type: A, layer: 1, pos: 190
type: A, layer: 1, pos: 251
type: A, layer: 1, pos: 93
type: A, layer: 1, pos: 71
type: A, layer: 1, pos: 176
type: A, layer: 1, pos: 1
type: A, layer: 1, pos: 188
type: A, layer: 1, pos: 66
type: A, layer: 1, pos: 201
type: A, layer: 1, pos: 177
type: A, layer: 1, pos: 198
type: A, layer: 1, pos: 13
type: A, layer: 1, pos: 120
type: A, layer: 1, pos: 219
type: A, layer: 1, pos: 235
type: A, layer: 1, pos: 182
type: A, layer: 1, pos: 0
type: A, layer: 1, pos: 172
type: A, layer: 1, pos: 68
type: A, layer: 1, pos: 245
type: A, layer: 1, pos: 37
type: A, layer: 1, pos: 183
type: A, layer: 1, pos: 240
type: A, layer: 1, pos: 166

Time for candidate selection: 0.22 seconds

### Candidate
type: A, layer: 1, pos: 220

## Relational analysis of NS_A1_B2_A1_B1_A1

### Relational analysis result of NS_A1_B2_A1_B1_A1
Status: Status.VERIFIED
Output dim: 3, lower bound: -0.3145823, upper bound: 0.3157761
time: 1.33 seconds

## Relational analysis of NS_A1_B2_A1_B1_A2

### Relational analysis result of NS_A1_B2_A1_B1_A2
Status: Status.VERIFIED
Output dim: 3, lower bound: -0.3145823, upper bound: 0.3157761
time: 1.46 seconds

## BFS NS instance: NS_A1_B2_A2_B1

### Backsubstitution after applying NS history:
0: -0.3698886, 0.3360421, -0.4720473, 0.4147801, -0.7846687, 0.8080894
1: -0.2382730, 0.2692519, -0.3109462, 0.3432207, -0.5814937, 0.5801982
2: -0.3731204, 0.3228899, -0.4644855, 0.4174038, -0.7905242, 0.7873755
3: 0.8496231, 1.1181831, 0.7971720, 1.1429830, -0.2933598, 0.3210112
4: -0.2881854, 0.3316317, -0.3910520, 0.4087661, -0.6969515, 0.7226838
5: -0.2787703, 0.3576140, -0.3616869, 0.4233303, -0.7021006, 0.7193009
6: -0.3830273, 0.3706621, -0.4625316, 0.4727616, -0.8557889, 0.8331937
7: -0.3048434, 0.2361929, -0.3852404, 0.3329762, -0.6378196, 0.6214333
8: -0.2792376, 0.3866141, -0.3711649, 0.4866079, -0.7658455, 0.7577790
9: -0.3208500, 0.2560733, -0.4195342, 0.3286684, -0.6495184, 0.6756075

Time for backsubstitution: 2.06 seconds

### NS candidates at layer 1
type: A, layer: 1, pos: 220
type: A, layer: 1, pos: 15
type: A, layer: 1, pos: 30
type: A, layer: 1, pos: 175
type: A, layer: 1, pos: 221
type: A, layer: 1, pos: 190
type: A, layer: 1, pos: 62
type: A, layer: 1, pos: 161
type: A, layer: 1, pos: 178
type: A, layer: 1, pos: 233
type: A, layer: 1, pos: 103
type: A, layer: 1, pos: 136
type: A, layer: 1, pos: 188
type: A, layer: 1, pos: 251
type: A, layer: 1, pos: 176
type: A, layer: 1, pos: 93
type: A, layer: 1, pos: 71
type: A, layer: 1, pos: 66
type: A, layer: 1, pos: 201
type: A, layer: 1, pos: 1
type: A, layer: 1, pos: 177
type: A, layer: 1, pos: 198
type: A, layer: 1, pos: 235
type: A, layer: 1, pos: 219
type: A, layer: 1, pos: 120
type: A, layer: 1, pos: 13
type: A, layer: 1, pos: 172
type: A, layer: 1, pos: 68
type: A, layer: 1, pos: 0
type: A, layer: 1, pos: 182
type: A, layer: 1, pos: 245
type: A, layer: 1, pos: 166
type: A, layer: 1, pos: 240
type: A, layer: 1, pos: 37
type: A, layer: 1, pos: 183

Time for candidate selection: 0.22 seconds

### Candidate
type: A, layer: 1, pos: 220

## Relational analysis of NS_A1_B2_A2_B1_A1

### Relational analysis result of NS_A1_B2_A2_B1_A1
Status: Status.VERIFIED
Output dim: 3, lower bound: -0.3163152, upper bound: 0.3163053
time: 1.15 seconds

## Relational analysis of NS_A1_B2_A2_B1_A2

### Relational analysis result of NS_A1_B2_A2_B1_A2
Status: Status.VERIFIED
Output dim: 3, lower bound: -0.3163152, upper bound: 0.3163053
time: 1.30 seconds

## BFS NS instance: NS_A2_B1_A1_B1

### Backsubstitution after applying NS history:
0: -0.3532027, 0.3224772, -0.3920762, 0.3538485, -0.7070513, 0.7145534
1: -0.2265246, 0.2555524, -0.2540990, 0.2864872, -0.5130118, 0.5096514
2: -0.3570169, 0.3084190, -0.3934389, 0.3441919, -0.7012088, 0.7018580
3: 0.8572156, 1.1131964, 0.8400056, 1.1235851, -0.2663695, 0.2731907
4: -0.2722330, 0.3180251, -0.3111942, 0.3491313, -0.6213644, 0.6292193
5: -0.2643655, 0.3462325, -0.2971888, 0.3725173, -0.6368828, 0.6434213
6: -0.3680750, 0.3533955, -0.4011486, 0.3937304, -0.7618054, 0.7545441
7: -0.2909860, 0.2215497, -0.3231278, 0.2570808, -0.5480669, 0.5446774
8: -0.2628871, 0.3710253, -0.3003026, 0.4082607, -0.6711478, 0.6713278
9: -0.3046828, 0.2441054, -0.3433677, 0.2715535, -0.5762363, 0.5874731

Time for backsubstitution: 2.06 seconds

### NS candidates at layer 1
type: A, layer: 1, pos: 220
type: A, layer: 1, pos: 15
type: A, layer: 1, pos: 30
type: A, layer: 1, pos: 175
type: A, layer: 1, pos: 221
type: A, layer: 1, pos: 62
type: A, layer: 1, pos: 161
type: A, layer: 1, pos: 178
type: A, layer: 1, pos: 190
type: A, layer: 1, pos: 233
type: A, layer: 1, pos: 103
type: A, layer: 1, pos: 136
type: A, layer: 1, pos: 251
type: A, layer: 1, pos: 188
type: A, layer: 1, pos: 176
type: A, layer: 1, pos: 93
type: A, layer: 1, pos: 71
type: A, layer: 1, pos: 201
type: A, layer: 1, pos: 66
type: A, layer: 1, pos: 1
type: A, layer: 1, pos: 177
type: A, layer: 1, pos: 198
type: A, layer: 1, pos: 235
type: A, layer: 1, pos: 120
type: A, layer: 1, pos: 219
type: A, layer: 1, pos: 13
type: A, layer: 1, pos: 172
type: A, layer: 1, pos: 0
type: A, layer: 1, pos: 68
type: A, layer: 1, pos: 182
type: A, layer: 1, pos: 245
type: A, layer: 1, pos: 183
type: A, layer: 1, pos: 37
type: A, layer: 1, pos: 240
type: A, layer: 1, pos: 166

Time for candidate selection: 0.22 seconds

### Candidate
type: A, layer: 1, pos: 220

## Relational analysis of NS_A2_B1_A1_B1_A1

### Relational analysis result of NS_A2_B1_A1_B1_A1
Status: Status.VERIFIED
Output dim: 3, lower bound: -0.3145788, upper bound: 0.3157809
time: 1.34 seconds

## Relational analysis of NS_A2_B1_A1_B1_A2

### Relational analysis result of NS_A2_B1_A1_B1_A2
Status: Status.VERIFIED
Output dim: 3, lower bound: -0.3145788, upper bound: 0.3157809
time: 1.36 seconds

## BFS NS instance: NS_A2_B1_A2_B1

### Backsubstitution after applying NS history:
0: -0.4379394, 0.3887961, -0.4005199, 0.3607367, -0.7986761, 0.7893159
1: -0.2867818, 0.3187693, -0.2600246, 0.2931717, -0.5799534, 0.5787939
2: -0.4327600, 0.3879135, -0.4014205, 0.3518343, -0.7845942, 0.7893341
3: 0.8113470, 1.1338257, 0.8358310, 1.1259507, -0.3146037, 0.2979947
4: -0.3580533, 0.3835085, -0.3196350, 0.3558160, -0.7138692, 0.7031435
5: -0.3339469, 0.4022052, -0.3042822, 0.3782081, -0.7121550, 0.7064874
6: -0.4358879, 0.4390062, -0.4082978, 0.4026270, -0.8385149, 0.8473040
7: -0.3583753, 0.3025244, -0.3299913, 0.2649111, -0.6232864, 0.6325157
8: -0.3409136, 0.4540221, -0.3083292, 0.4166488, -0.7575624, 0.7623513
9: -0.3881409, 0.3050356, -0.3516247, 0.2775913, -0.6657323, 0.6566603

Time for backsubstitution: 1.94 seconds

### NS candidates at layer 1
type: A, layer: 1, pos: 220
type: A, layer: 1, pos: 15
type: A, layer: 1, pos: 30
type: A, layer: 1, pos: 175
type: A, layer: 1, pos: 190
type: A, layer: 1, pos: 221
type: A, layer: 1, pos: 161
type: A, layer: 1, pos: 62
type: A, layer: 1, pos: 178
type: A, layer: 1, pos: 233
type: A, layer: 1, pos: 103
type: A, layer: 1, pos: 136
type: A, layer: 1, pos: 188
type: A, layer: 1, pos: 251
type: A, layer: 1, pos: 176
type: A, layer: 1, pos: 93
type: A, layer: 1, pos: 201
type: A, layer: 1, pos: 71
type: A, layer: 1, pos: 66
type: A, layer: 1, pos: 198
type: A, layer: 1, pos: 235
type: A, layer: 1, pos: 1
type: A, layer: 1, pos: 177
type: A, layer: 1, pos: 219
type: A, layer: 1, pos: 120
type: A, layer: 1, pos: 172
type: A, layer: 1, pos: 68
type: A, layer: 1, pos: 13
type: A, layer: 1, pos: 0
type: A, layer: 1, pos: 182
type: A, layer: 1, pos: 245
type: A, layer: 1, pos: 166
type: A, layer: 1, pos: 240
type: A, layer: 1, pos: 37
type: A, layer: 1, pos: 183

Time for candidate selection: 0.19 seconds

### Candidate
type: A, layer: 1, pos: 220

## Relational analysis of NS_A2_B1_A2_B1_A1

### Relational analysis result of NS_A2_B1_A2_B1_A1
Status: Status.VERIFIED
Output dim: 3, lower bound: -0.3163053, upper bound: 0.3163152
time: 2.08 seconds

## Relational analysis of NS_A2_B1_A2_B1_A2

### Relational analysis result of NS_A2_B1_A2_B1_A2
Status: Status.VERIFIED
Output dim: 3, lower bound: -0.3163053, upper bound: 0.3163152
time: 1.40 seconds

## BFS NS instance: NS_A2_B2_A1_B1

### Backsubstitution after applying NS history:
0: -0.3532027, 0.3224772, -0.4633013, 0.4079013, -0.7611040, 0.7857785
1: -0.2265246, 0.2555524, -0.3046660, 0.3368634, -0.5633880, 0.5602183
2: -0.3570169, 0.3084190, -0.4558421, 0.4101121, -0.7671289, 0.7642612
3: 0.8572156, 1.1131964, 0.8013147, 1.1402977, -0.2830821, 0.3118817
4: -0.2722330, 0.3180251, -0.3825811, 0.4022008, -0.6744338, 0.7006062
5: -0.2643655, 0.3462325, -0.3543988, 0.4177817, -0.6821473, 0.7006313
6: -0.3680750, 0.3533955, -0.4554920, 0.4639602, -0.8320352, 0.8088875
7: -0.2909860, 0.2215497, -0.3780777, 0.3251879, -0.6161740, 0.5996274
8: -0.2628871, 0.3710253, -0.3633014, 0.4782517, -0.7411388, 0.7343267
9: -0.3046828, 0.2441054, -0.4117097, 0.3220733, -0.6267561, 0.6558151

Time for backsubstitution: 2.09 seconds

### NS candidates at layer 1
type: A, layer: 1, pos: 220
type: A, layer: 1, pos: 15
type: A, layer: 1, pos: 30
type: A, layer: 1, pos: 175
type: A, layer: 1, pos: 221
type: A, layer: 1, pos: 62
type: A, layer: 1, pos: 161
type: A, layer: 1, pos: 178
type: A, layer: 1, pos: 190
type: A, layer: 1, pos: 233
type: A, layer: 1, pos: 103
type: A, layer: 1, pos: 136
type: A, layer: 1, pos: 251
type: A, layer: 1, pos: 188
type: A, layer: 1, pos: 176
type: A, layer: 1, pos: 93
type: A, layer: 1, pos: 71
type: A, layer: 1, pos: 201
type: A, layer: 1, pos: 66
type: A, layer: 1, pos: 1
type: A, layer: 1, pos: 177
type: A, layer: 1, pos: 198
type: A, layer: 1, pos: 235
type: A, layer: 1, pos: 120
type: A, layer: 1, pos: 219
type: A, layer: 1, pos: 13
type: A, layer: 1, pos: 172
type: A, layer: 1, pos: 0
type: A, layer: 1, pos: 68
type: A, layer: 1, pos: 182
type: A, layer: 1, pos: 245
type: A, layer: 1, pos: 183
type: A, layer: 1, pos: 37
type: A, layer: 1, pos: 240
type: A, layer: 1, pos: 166

Time for candidate selection: 0.22 seconds

### Candidate
type: A, layer: 1, pos: 220

## Relational analysis of NS_A2_B2_A1_B1_A1

### Relational analysis result of NS_A2_B2_A1_B1_A1
Status: Status.VERIFIED
Output dim: 3, lower bound: -0.3145784, upper bound: 0.3157757
time: 1.33 seconds

## Relational analysis of NS_A2_B2_A1_B1_A2

### Relational analysis result of NS_A2_B2_A1_B1_A2
Status: Status.VERIFIED
Output dim: 3, lower bound: -0.3145784, upper bound: 0.3157757
time: 1.27 seconds

## BFS NS instance: NS_A2_B2_A2_B1

### Backsubstitution after applying NS history:
0: -0.4379394, 0.3887961, -0.4720473, 0.4147801, -0.8527195, 0.8608434
1: -0.2867818, 0.3187693, -0.3109462, 0.3432207, -0.6300025, 0.6297156
2: -0.4327600, 0.3879135, -0.4644855, 0.4174038, -0.8501638, 0.8523991
3: 0.8113470, 1.1338257, 0.7971720, 1.1429830, -0.3316360, 0.3366537
4: -0.3580533, 0.3835085, -0.3910520, 0.4087661, -0.7668194, 0.7745606
5: -0.3339469, 0.4022052, -0.3616869, 0.4233303, -0.7572772, 0.7638921
6: -0.4358879, 0.4390062, -0.4625316, 0.4727616, -0.9086496, 0.9015378
7: -0.3583753, 0.3025244, -0.3852404, 0.3329762, -0.6913515, 0.6877648
8: -0.3409136, 0.4540221, -0.3711649, 0.4866079, -0.8275214, 0.8251870
9: -0.3881409, 0.3050356, -0.4195342, 0.3286684, -0.7168094, 0.7245698

Time for backsubstitution: 2.06 seconds

### NS candidates at layer 1
type: A, layer: 1, pos: 220
type: A, layer: 1, pos: 15
type: A, layer: 1, pos: 30
type: A, layer: 1, pos: 175
type: A, layer: 1, pos: 190
type: A, layer: 1, pos: 221
type: A, layer: 1, pos: 161
type: A, layer: 1, pos: 62
type: A, layer: 1, pos: 178
type: A, layer: 1, pos: 233
type: A, layer: 1, pos: 103
type: A, layer: 1, pos: 136
type: A, layer: 1, pos: 188
type: A, layer: 1, pos: 251
type: A, layer: 1, pos: 176
type: A, layer: 1, pos: 93
type: A, layer: 1, pos: 201
type: A, layer: 1, pos: 71
type: A, layer: 1, pos: 66
type: A, layer: 1, pos: 198
type: A, layer: 1, pos: 235
type: A, layer: 1, pos: 1
type: A, layer: 1, pos: 177
type: A, layer: 1, pos: 219
type: A, layer: 1, pos: 120
type: A, layer: 1, pos: 172
type: A, layer: 1, pos: 68
type: A, layer: 1, pos: 13
type: A, layer: 1, pos: 0
type: A, layer: 1, pos: 182
type: A, layer: 1, pos: 245
type: A, layer: 1, pos: 166
type: A, layer: 1, pos: 240
type: A, layer: 1, pos: 37
type: A, layer: 1, pos: 183

Time for candidate selection: 0.23 seconds

### Candidate
type: A, layer: 1, pos: 220

## Relational analysis of NS_A2_B2_A2_B1_A1

### Relational analysis result of NS_A2_B2_A2_B1_A1
Status: Status.VERIFIED
Output dim: 3, lower bound: -0.3163047, upper bound: 0.3163047
time: 1.23 seconds

## Relational analysis of NS_A2_B2_A2_B1_A2

### Relational analysis result of NS_A2_B2_A2_B1_A2
Status: Status.VERIFIED
Output dim: 3, lower bound: -0.3163047, upper bound: 0.3163047
time: 1.18 seconds

## Summary of splitting at layer (split count: 4)
- Time for NS candidates: 4.71 seconds
NS_A1_B1_A1_B1_A1, status: Status.VERIFIED, split count: 5, time: 4.71
Output dim: 3, lower bound: -0.3145860, upper bound: 0.3157829
NS_A1_B1_A1_B1_A2, status: Status.VERIFIED, split count: 5, time: 4.71
Output dim: 3, lower bound: -0.3145860, upper bound: 0.3157829
NS_A1_B1_A2_B1_A1, status: Status.VERIFIED, split count: 5, time: 4.71
Output dim: 3, lower bound: -0.3163202, upper bound: 0.3163202
NS_A1_B1_A2_B1_A2, status: Status.VERIFIED, split count: 5, time: 4.71
Output dim: 3, lower bound: -0.3163202, upper bound: 0.3163202
NS_A1_B2_A1_B1_A1, status: Status.VERIFIED, split count: 5, time: 4.71
Output dim: 3, lower bound: -0.3145823, upper bound: 0.3157761
NS_A1_B2_A1_B1_A2, status: Status.VERIFIED, split count: 5, time: 4.71
Output dim: 3, lower bound: -0.3145823, upper bound: 0.3157761
NS_A1_B2_A2_B1_A1, status: Status.VERIFIED, split count: 5, time: 4.71
Output dim: 3, lower bound: -0.3163152, upper bound: 0.3163053
NS_A1_B2_A2_B1_A2, status: Status.VERIFIED, split count: 5, time: 4.71
Output dim: 3, lower bound: -0.3163152, upper bound: 0.3163053
NS_A2_B1_A1_B1_A1, status: Status.VERIFIED, split count: 5, time: 4.71
Output dim: 3, lower bound: -0.3145788, upper bound: 0.3157809
NS_A2_B1_A1_B1_A2, status: Status.VERIFIED, split count: 5, time: 4.71
Output dim: 3, lower bound: -0.3145788, upper bound: 0.3157809
NS_A2_B1_A2_B1_A1, status: Status.VERIFIED, split count: 5, time: 4.71
Output dim: 3, lower bound: -0.3163053, upper bound: 0.3163152
NS_A2_B1_A2_B1_A2, status: Status.VERIFIED, split count: 5, time: 4.71
Output dim: 3, lower bound: -0.3163053, upper bound: 0.3163152
NS_A2_B2_A1_B1_A1, status: Status.VERIFIED, split count: 5, time: 4.71
Output dim: 3, lower bound: -0.3145784, upper bound: 0.3157757
NS_A2_B2_A1_B1_A2, status: Status.VERIFIED, split count: 5, time: 4.71
Output dim: 3, lower bound: -0.3145784, upper bound: 0.3157757
NS_A2_B2_A2_B1_A1, status: Status.VERIFIED, split count: 5, time: 4.71
Output dim: 3, lower bound: -0.3163047, upper bound: 0.3163047
NS_A2_B2_A2_B1_A2, status: Status.VERIFIED, split count: 5, time: 4.71
Output dim: 3, lower bound: -0.3163047, upper bound: 0.3163047

## NS Result
status: Status.VERIFIED
execution time: (base) + (ns) = 4.83 + 110.66 = 115.49 seconds
