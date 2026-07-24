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
execution time: IAR + RelationalAnalysis = 2.03 + 2.32 = 4.35 seconds
status: Status.UNKNOWN
relational distance
Output dim: 3, lower bound: -0.3305063, upper bound: 0.3305063

# Neuron Split (NS) starts

## BFS NS instance: NS

Time for backsubstitution: 0.00 seconds

### NS candidates at layer 1
type: A, layer: 1, pos: 83
type: B, layer: 1, pos: 83
type: A, layer: 1, pos: 161
type: B, layer: 1, pos: 161
type: B, layer: 1, pos: 29
type: A, layer: 1, pos: 29
type: B, layer: 1, pos: 220
type: A, layer: 1, pos: 220
type: A, layer: 1, pos: 15
type: B, layer: 1, pos: 15
type: A, layer: 1, pos: 30
type: B, layer: 1, pos: 30
type: B, layer: 1, pos: 233
type: A, layer: 1, pos: 233
type: B, layer: 1, pos: 175
type: A, layer: 1, pos: 175
type: A, layer: 1, pos: 62
type: B, layer: 1, pos: 62
type: B, layer: 1, pos: 251
type: A, layer: 1, pos: 251
type: B, layer: 1, pos: 103
type: A, layer: 1, pos: 103
type: A, layer: 1, pos: 221
type: B, layer: 1, pos: 221
type: A, layer: 1, pos: 190
type: B, layer: 1, pos: 190
type: A, layer: 1, pos: 176
type: B, layer: 1, pos: 176
type: A, layer: 1, pos: 178
type: B, layer: 1, pos: 178
type: A, layer: 1, pos: 172
type: B, layer: 1, pos: 172
type: A, layer: 1, pos: 136
type: B, layer: 1, pos: 136
type: B, layer: 1, pos: 68
type: A, layer: 1, pos: 68
type: B, layer: 1, pos: 66
type: A, layer: 1, pos: 66
type: A, layer: 1, pos: 183
type: B, layer: 1, pos: 183
type: B, layer: 1, pos: 188
type: A, layer: 1, pos: 188
type: A, layer: 1, pos: 0
type: B, layer: 1, pos: 0
type: A, layer: 1, pos: 201
type: B, layer: 1, pos: 201
type: A, layer: 1, pos: 177
type: B, layer: 1, pos: 177
type: A, layer: 1, pos: 120
type: B, layer: 1, pos: 120
type: B, layer: 1, pos: 235
type: A, layer: 1, pos: 235
type: B, layer: 1, pos: 71
type: A, layer: 1, pos: 71
type: B, layer: 1, pos: 198
type: A, layer: 1, pos: 198
type: A, layer: 1, pos: 1
type: B, layer: 1, pos: 1
type: A, layer: 1, pos: 219
type: B, layer: 1, pos: 219
type: A, layer: 1, pos: 93
type: B, layer: 1, pos: 93
type: A, layer: 1, pos: 37
type: B, layer: 1, pos: 37
type: A, layer: 1, pos: 13
type: B, layer: 1, pos: 13
type: B, layer: 1, pos: 245
type: A, layer: 1, pos: 245
type: A, layer: 1, pos: 182
type: B, layer: 1, pos: 182
type: A, layer: 1, pos: 166
type: B, layer: 1, pos: 166
type: B, layer: 1, pos: 240
type: A, layer: 1, pos: 240

Time for candidate selection: 0.18 seconds

### Candidate
type: A, layer: 1, pos: 83

## Relational analysis of NS_A1

### Relational analysis result of NS_A1
Status: Status.UNKNOWN
Output dim: 3, lower bound: -0.3305063, upper bound: 0.3303112
time: 1.15 seconds

## Relational analysis of NS_A2

### Relational analysis result of NS_A2
Status: Status.UNKNOWN
Output dim: 3, lower bound: -0.3303112, upper bound: 0.3303112
time: 1.60 seconds

## Summary of splitting at layer (split count: 0)
- Time for NS candidates: 2.94 seconds
NS_A1, status: Status.UNKNOWN, split count: 1, time: 2.94
Output dim: 3, lower bound: -0.3305063, upper bound: 0.3303112
NS_A2, status: Status.UNKNOWN, split count: 1, time: 2.94
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

Time for backsubstitution: 1.73 seconds

### NS candidates at layer 1
type: B, layer: 1, pos: 161
type: A, layer: 1, pos: 161
type: B, layer: 1, pos: 83
type: A, layer: 1, pos: 29
type: B, layer: 1, pos: 29
type: B, layer: 1, pos: 220
type: A, layer: 1, pos: 220
type: B, layer: 1, pos: 30
type: B, layer: 1, pos: 15
type: A, layer: 1, pos: 30
type: A, layer: 1, pos: 15
type: B, layer: 1, pos: 233
type: A, layer: 1, pos: 233
type: A, layer: 1, pos: 175
type: B, layer: 1, pos: 175
type: A, layer: 1, pos: 251
type: B, layer: 1, pos: 251
type: B, layer: 1, pos: 62
type: A, layer: 1, pos: 62
type: B, layer: 1, pos: 103
type: B, layer: 1, pos: 221
type: A, layer: 1, pos: 103
type: A, layer: 1, pos: 221
type: B, layer: 1, pos: 190
type: A, layer: 1, pos: 190
type: B, layer: 1, pos: 176
type: A, layer: 1, pos: 176
type: B, layer: 1, pos: 178
type: A, layer: 1, pos: 178
type: B, layer: 1, pos: 172
type: A, layer: 1, pos: 172
type: B, layer: 1, pos: 136
type: A, layer: 1, pos: 136
type: B, layer: 1, pos: 68
type: A, layer: 1, pos: 68
type: B, layer: 1, pos: 0
type: A, layer: 1, pos: 0
type: B, layer: 1, pos: 66
type: A, layer: 1, pos: 66
type: A, layer: 1, pos: 188
type: B, layer: 1, pos: 183
type: A, layer: 1, pos: 183
type: B, layer: 1, pos: 188
type: B, layer: 1, pos: 201
type: A, layer: 1, pos: 201
type: A, layer: 1, pos: 177
type: B, layer: 1, pos: 177
type: B, layer: 1, pos: 120
type: A, layer: 1, pos: 120
type: B, layer: 1, pos: 198
type: A, layer: 1, pos: 198
type: B, layer: 1, pos: 71
type: A, layer: 1, pos: 235
type: A, layer: 1, pos: 71
type: B, layer: 1, pos: 235
type: B, layer: 1, pos: 219
type: A, layer: 1, pos: 219
type: A, layer: 1, pos: 1
type: B, layer: 1, pos: 1
type: B, layer: 1, pos: 93
type: A, layer: 1, pos: 93
type: B, layer: 1, pos: 37
type: A, layer: 1, pos: 37
type: B, layer: 1, pos: 13
type: A, layer: 1, pos: 13
type: B, layer: 1, pos: 245
type: A, layer: 1, pos: 245
type: A, layer: 1, pos: 182
type: B, layer: 1, pos: 182
type: A, layer: 1, pos: 240
type: B, layer: 1, pos: 240
type: B, layer: 1, pos: 166
type: A, layer: 1, pos: 166

Time for candidate selection: 0.18 seconds

### Candidate
type: B, layer: 1, pos: 161

## Relational analysis of NS_A1_B1

### Relational analysis result of NS_A1_B1
Status: Status.VERIFIED
Output dim: 3, lower bound: -0.2918461, upper bound: 0.3026376
time: 2.25 seconds

## Relational analysis of NS_A1_B2

### Relational analysis result of NS_A1_B2
Status: Status.VERIFIED
Output dim: 3, lower bound: -0.2899924, upper bound: 0.2880382
time: 1.44 seconds

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

Time for backsubstitution: 1.78 seconds

### NS candidates at layer 1
type: B, layer: 1, pos: 161
type: A, layer: 1, pos: 161
type: B, layer: 1, pos: 83
type: A, layer: 1, pos: 29
type: B, layer: 1, pos: 29
type: B, layer: 1, pos: 15
type: A, layer: 1, pos: 30
type: A, layer: 1, pos: 220
type: B, layer: 1, pos: 220
type: B, layer: 1, pos: 30
type: A, layer: 1, pos: 15
type: A, layer: 1, pos: 233
type: B, layer: 1, pos: 233
type: A, layer: 1, pos: 175
type: B, layer: 1, pos: 62
type: A, layer: 1, pos: 251
type: B, layer: 1, pos: 175
type: B, layer: 1, pos: 251
type: B, layer: 1, pos: 190
type: A, layer: 1, pos: 62
type: B, layer: 1, pos: 221
type: B, layer: 1, pos: 103
type: A, layer: 1, pos: 103
type: B, layer: 1, pos: 176
type: A, layer: 1, pos: 190
type: A, layer: 1, pos: 221
type: A, layer: 1, pos: 176
type: B, layer: 1, pos: 178
type: A, layer: 1, pos: 178
type: B, layer: 1, pos: 172
type: A, layer: 1, pos: 172
type: A, layer: 1, pos: 68
type: B, layer: 1, pos: 68
type: B, layer: 1, pos: 136
type: A, layer: 1, pos: 136
type: A, layer: 1, pos: 188
type: A, layer: 1, pos: 0
type: B, layer: 1, pos: 0
type: B, layer: 1, pos: 66
type: A, layer: 1, pos: 66
type: B, layer: 1, pos: 183
type: A, layer: 1, pos: 183
type: B, layer: 1, pos: 188
type: B, layer: 1, pos: 201
type: A, layer: 1, pos: 201
type: A, layer: 1, pos: 177
type: B, layer: 1, pos: 177
type: B, layer: 1, pos: 120
type: A, layer: 1, pos: 120
type: A, layer: 1, pos: 198
type: A, layer: 1, pos: 235
type: B, layer: 1, pos: 198
type: B, layer: 1, pos: 71
type: B, layer: 1, pos: 235
type: B, layer: 1, pos: 219
type: A, layer: 1, pos: 71
type: A, layer: 1, pos: 219
type: A, layer: 1, pos: 1
type: B, layer: 1, pos: 93
type: B, layer: 1, pos: 1
type: A, layer: 1, pos: 93
type: B, layer: 1, pos: 37
type: A, layer: 1, pos: 37
type: B, layer: 1, pos: 13
type: A, layer: 1, pos: 245
type: B, layer: 1, pos: 245
type: A, layer: 1, pos: 13
type: B, layer: 1, pos: 182
type: A, layer: 1, pos: 182
type: A, layer: 1, pos: 240
type: B, layer: 1, pos: 240
type: B, layer: 1, pos: 166
type: A, layer: 1, pos: 166

Time for candidate selection: 0.21 seconds

### Candidate
type: B, layer: 1, pos: 161

## Relational analysis of NS_A2_B1

### Relational analysis result of NS_A2_B1
Status: Status.VERIFIED
Output dim: 3, lower bound: -0.2891638, upper bound: 0.3017278
time: 1.98 seconds

## Relational analysis of NS_A2_B2

### Relational analysis result of NS_A2_B2
Status: Status.VERIFIED
Output dim: 3, lower bound: -0.2879503, upper bound: 0.2879503
time: 1.26 seconds

## Summary of splitting at layer (split count: 1)
- Time for NS candidates: 5.24 seconds
NS_A1_B1, status: Status.VERIFIED, split count: 2, time: 5.24
Output dim: 3, lower bound: -0.2918461, upper bound: 0.3026376
NS_A1_B2, status: Status.VERIFIED, split count: 2, time: 5.24
Output dim: 3, lower bound: -0.2899924, upper bound: 0.2880382
NS_A2_B1, status: Status.VERIFIED, split count: 2, time: 5.24
Output dim: 3, lower bound: -0.2891638, upper bound: 0.3017278
NS_A2_B2, status: Status.VERIFIED, split count: 2, time: 5.24
Output dim: 3, lower bound: -0.2879503, upper bound: 0.2879503

## NS Result
status: Status.VERIFIED
execution time: (base) + (ns) = 4.35 + 13.79 = 18.14 seconds
