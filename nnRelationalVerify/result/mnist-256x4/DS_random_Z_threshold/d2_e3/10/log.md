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
execution time: IAR + RelationalAnalysis = 0.84 + 2.22 = 3.06 seconds
status: Status.UNKNOWN
relational distance
Output dim: 3, lower bound: -0.3305063, upper bound: 0.3305063

# Delta Split (DS) starts

## BFS DS instance: DS

Time for backsubstitution: 0.00 seconds

### DS candidates at layer 1
type: DSZ, layer: 1, pos: 190
type: DSZ, layer: 1, pos: 220
type: DSZ, layer: 1, pos: 0
type: DSZ, layer: 1, pos: 120
type: DSZ, layer: 1, pos: 30
type: DSZ, layer: 1, pos: 233
type: DSZ, layer: 1, pos: 15
type: DSZ, layer: 1, pos: 251
type: DSZ, layer: 1, pos: 62
type: DSZ, layer: 1, pos: 183
type: DSZ, layer: 1, pos: 37
type: DSZ, layer: 1, pos: 177
type: DSZ, layer: 1, pos: 235
type: DSZ, layer: 1, pos: 136
type: DSZ, layer: 1, pos: 166
type: DSZ, layer: 1, pos: 219
type: DSZ, layer: 1, pos: 68
type: DSZ, layer: 1, pos: 172
type: DSZ, layer: 1, pos: 198
type: DSZ, layer: 1, pos: 175
type: DSZ, layer: 1, pos: 245
type: DSZ, layer: 1, pos: 13
type: DSZ, layer: 1, pos: 71
type: DSZ, layer: 1, pos: 161
type: DSZ, layer: 1, pos: 182
type: DSZ, layer: 1, pos: 188
type: DSZ, layer: 1, pos: 103
type: DSZ, layer: 1, pos: 176
type: DSZ, layer: 1, pos: 93
type: DSZ, layer: 1, pos: 240
type: DSZ, layer: 1, pos: 221
type: DSZ, layer: 1, pos: 1
type: DSZ, layer: 1, pos: 178
type: DSZ, layer: 1, pos: 201
type: DSZ, layer: 1, pos: 66
type: DSZ, layer: 1, pos: 83
type: DSZ, layer: 1, pos: 29

Time for candidate selection: 0.00 seconds

### Candidate
type: DSZ, layer: 1, pos: 190

### Relational analysis ABCD of DS_DSZ1

#### Relational analysis ABCD result of DS_DSZ1
Status: Status.UNKNOWN
Output dim: 3, lower bound: -0.3282285, upper bound: 0.3282285
time: 1.19 seconds

### Relational analysis ABCD of DS_DSZ2

#### Relational analysis ABCD result of DS_DSZ2
Status: Status.UNKNOWN
Output dim: 3, lower bound: -0.3282285, upper bound: 0.3282285
time: 1.18 seconds

## Summary of splitting (split count: 0)
- Time for DS candidates: 2.38 seconds
DS_DSZ1, status: Status.UNKNOWN, split count: 1, time: 2.38
Output dim: 3, lower bound: -0.3282285, upper bound: 0.3282285
DS_DSZ2, status: Status.UNKNOWN, split count: 1, time: 2.38
Output dim: 3, lower bound: -0.3282285, upper bound: 0.3282285

## BFS DS instance: DS_DSZ1

### Backsubstitution after applying DS history:
0: -0.5426022, 0.4665188, -0.5426022, 0.4665188, -1.0091211, 1.0091211
1: -0.3630026, 0.3911184, -0.3630026, 0.3911184, -0.7541210, 0.7541210
2: -0.5269889, 0.4806983, -0.5269889, 0.4806983, -1.0076871, 1.0076871
3: 0.7543352, 1.1590887, 0.7543352, 1.1590887, -0.4047535, 0.4047535
4: -0.4614211, 0.4625955, -0.4614211, 0.4625955, -0.9240167, 0.9240167
5: -0.4214845, 0.4672619, -0.4214845, 0.4672619, -0.8887464, 0.8887464
6: -0.5138452, 0.5371552, -0.5138452, 0.5371552, -1.0510004, 1.0510004
7: -0.4409947, 0.4028524, -0.4409947, 0.4028524, -0.8438470, 0.8438470
8: -0.4306176, 0.5533380, -0.4306176, 0.5533380, -0.9839556, 0.9839556
9: -0.4872051, 0.3854194, -0.4872051, 0.3854194, -0.8726246, 0.8726246

Time for backsubstitution: 0.77 seconds

### DS candidates at layer 1
type: DSZ, layer: 1, pos: 1
type: DSZ, layer: 1, pos: 83
type: DSZ, layer: 1, pos: 37
type: DSZ, layer: 1, pos: 176
type: DSZ, layer: 1, pos: 201
type: DSZ, layer: 1, pos: 188
type: DSZ, layer: 1, pos: 29
type: DSZ, layer: 1, pos: 103
type: DSZ, layer: 1, pos: 166
type: DSZ, layer: 1, pos: 221
type: DSZ, layer: 1, pos: 161
type: DSZ, layer: 1, pos: 30
type: DSZ, layer: 1, pos: 178
type: DSZ, layer: 1, pos: 172
type: DSZ, layer: 1, pos: 175
type: DSZ, layer: 1, pos: 177
type: DSZ, layer: 1, pos: 183
type: DSZ, layer: 1, pos: 182
type: DSZ, layer: 1, pos: 0
type: DSZ, layer: 1, pos: 220
type: DSZ, layer: 1, pos: 68
type: DSZ, layer: 1, pos: 251
type: DSZ, layer: 1, pos: 71
type: DSZ, layer: 1, pos: 245
type: DSZ, layer: 1, pos: 13
type: DSZ, layer: 1, pos: 240
type: DSZ, layer: 1, pos: 93
type: DSZ, layer: 1, pos: 219
type: DSZ, layer: 1, pos: 120
type: DSZ, layer: 1, pos: 198
type: DSZ, layer: 1, pos: 233
type: DSZ, layer: 1, pos: 15
type: DSZ, layer: 1, pos: 66
type: DSZ, layer: 1, pos: 62
type: DSZ, layer: 1, pos: 136
type: DSZ, layer: 1, pos: 235

Time for candidate selection: 0.00 seconds

### Candidate
type: DSZ, layer: 1, pos: 1

### Relational analysis ABCD of DS_DSZ1_DSZ1

#### Relational analysis ABCD result of DS_DSZ1_DSZ1
Status: Status.UNKNOWN
Output dim: 3, lower bound: -0.3282285, upper bound: 0.3282285
time: 0.99 seconds

### Relational analysis ABCD of DS_DSZ1_DSZ2

#### Relational analysis ABCD result of DS_DSZ1_DSZ2
Status: Status.UNKNOWN
Output dim: 3, lower bound: -0.3282285, upper bound: 0.3282285
time: 1.00 seconds

## BFS DS instance: DS_DSZ2

### Backsubstitution after applying DS history:
0: -0.5426022, 0.4665188, -0.5426022, 0.4665188, -1.0091211, 1.0091211
1: -0.3630026, 0.3911184, -0.3630026, 0.3911184, -0.7541210, 0.7541210
2: -0.5269889, 0.4806983, -0.5269889, 0.4806983, -1.0076871, 1.0076871
3: 0.7543352, 1.1590887, 0.7543352, 1.1590887, -0.4047535, 0.4047535
4: -0.4614211, 0.4625955, -0.4614211, 0.4625955, -0.9240167, 0.9240167
5: -0.4214845, 0.4672619, -0.4214845, 0.4672619, -0.8887464, 0.8887464
6: -0.5138452, 0.5371552, -0.5138452, 0.5371552, -1.0510004, 1.0510004
7: -0.4409947, 0.4028524, -0.4409947, 0.4028524, -0.8438470, 0.8438470
8: -0.4306176, 0.5533380, -0.4306176, 0.5533380, -0.9839556, 0.9839556
9: -0.4872051, 0.3854194, -0.4872051, 0.3854194, -0.8726246, 0.8726246

Time for backsubstitution: 0.77 seconds

### DS candidates at layer 1
type: DSZ, layer: 1, pos: 201
type: DSZ, layer: 1, pos: 30
type: DSZ, layer: 1, pos: 13
type: DSZ, layer: 1, pos: 1
type: DSZ, layer: 1, pos: 29
type: DSZ, layer: 1, pos: 172
type: DSZ, layer: 1, pos: 66
type: DSZ, layer: 1, pos: 178
type: DSZ, layer: 1, pos: 188
type: DSZ, layer: 1, pos: 251
type: DSZ, layer: 1, pos: 161
type: DSZ, layer: 1, pos: 0
type: DSZ, layer: 1, pos: 120
type: DSZ, layer: 1, pos: 182
type: DSZ, layer: 1, pos: 93
type: DSZ, layer: 1, pos: 235
type: DSZ, layer: 1, pos: 233
type: DSZ, layer: 1, pos: 219
type: DSZ, layer: 1, pos: 183
type: DSZ, layer: 1, pos: 198
type: DSZ, layer: 1, pos: 220
type: DSZ, layer: 1, pos: 62
type: DSZ, layer: 1, pos: 15
type: DSZ, layer: 1, pos: 177
type: DSZ, layer: 1, pos: 166
type: DSZ, layer: 1, pos: 136
type: DSZ, layer: 1, pos: 240
type: DSZ, layer: 1, pos: 175
type: DSZ, layer: 1, pos: 37
type: DSZ, layer: 1, pos: 221
type: DSZ, layer: 1, pos: 245
type: DSZ, layer: 1, pos: 103
type: DSZ, layer: 1, pos: 176
type: DSZ, layer: 1, pos: 71
type: DSZ, layer: 1, pos: 68
type: DSZ, layer: 1, pos: 83

Time for candidate selection: 0.00 seconds

### Candidate
type: DSZ, layer: 1, pos: 201

### Relational analysis ABCD of DS_DSZ2_DSZ1

#### Relational analysis ABCD result of DS_DSZ2_DSZ1
Status: Status.UNKNOWN
Output dim: 3, lower bound: -0.3282285, upper bound: 0.3282285
time: 1.10 seconds

### Relational analysis ABCD of DS_DSZ2_DSZ2

#### Relational analysis ABCD result of DS_DSZ2_DSZ2
Status: Status.UNKNOWN
Output dim: 3, lower bound: -0.3282285, upper bound: 0.3282285
time: 1.18 seconds

## Summary of splitting (split count: 1)
- Time for DS candidates: 3.07 seconds
DS_DSZ1_DSZ1, status: Status.UNKNOWN, split count: 2, time: 3.07
Output dim: 3, lower bound: -0.3282285, upper bound: 0.3282285
DS_DSZ1_DSZ2, status: Status.UNKNOWN, split count: 2, time: 3.07
Output dim: 3, lower bound: -0.3282285, upper bound: 0.3282285
DS_DSZ2_DSZ1, status: Status.UNKNOWN, split count: 2, time: 3.07
Output dim: 3, lower bound: -0.3282285, upper bound: 0.3282285
DS_DSZ2_DSZ2, status: Status.UNKNOWN, split count: 2, time: 3.07
Output dim: 3, lower bound: -0.3282285, upper bound: 0.3282285

## BFS DS instance: DS_DSZ1_DSZ1

### Backsubstitution after applying DS history:
0: -0.5426022, 0.4665188, -0.5426022, 0.4665188, -1.0091211, 1.0091211
1: -0.3630026, 0.3911184, -0.3630026, 0.3911184, -0.7541210, 0.7541210
2: -0.5269889, 0.4806983, -0.5269889, 0.4806983, -1.0076871, 1.0076871
3: 0.7543352, 1.1590887, 0.7543352, 1.1590887, -0.4047535, 0.4047535
4: -0.4614211, 0.4625955, -0.4614211, 0.4625955, -0.9240167, 0.9240167
5: -0.4214845, 0.4672619, -0.4214845, 0.4672619, -0.8887464, 0.8887464
6: -0.5138452, 0.5371552, -0.5138452, 0.5371552, -1.0510004, 1.0510004
7: -0.4409947, 0.4028524, -0.4409947, 0.4028524, -0.8438470, 0.8438470
8: -0.4306176, 0.5533380, -0.4306176, 0.5533380, -0.9839556, 0.9839556
9: -0.4872051, 0.3854194, -0.4872051, 0.3854194, -0.8726246, 0.8726246

Time for backsubstitution: 0.77 seconds

### DS candidates at layer 1
type: DSZ, layer: 1, pos: 93
type: DSZ, layer: 1, pos: 0
type: DSZ, layer: 1, pos: 240
type: DSZ, layer: 1, pos: 29
type: DSZ, layer: 1, pos: 220
type: DSZ, layer: 1, pos: 71
type: DSZ, layer: 1, pos: 68
type: DSZ, layer: 1, pos: 251
type: DSZ, layer: 1, pos: 30
type: DSZ, layer: 1, pos: 201
type: DSZ, layer: 1, pos: 37
type: DSZ, layer: 1, pos: 177
type: DSZ, layer: 1, pos: 183
type: DSZ, layer: 1, pos: 15
type: DSZ, layer: 1, pos: 172
type: DSZ, layer: 1, pos: 178
type: DSZ, layer: 1, pos: 175
type: DSZ, layer: 1, pos: 233
type: DSZ, layer: 1, pos: 136
type: DSZ, layer: 1, pos: 188
type: DSZ, layer: 1, pos: 198
type: DSZ, layer: 1, pos: 221
type: DSZ, layer: 1, pos: 66
type: DSZ, layer: 1, pos: 219
type: DSZ, layer: 1, pos: 62
type: DSZ, layer: 1, pos: 166
type: DSZ, layer: 1, pos: 83
type: DSZ, layer: 1, pos: 103
type: DSZ, layer: 1, pos: 161
type: DSZ, layer: 1, pos: 235
type: DSZ, layer: 1, pos: 182
type: DSZ, layer: 1, pos: 13
type: DSZ, layer: 1, pos: 176
type: DSZ, layer: 1, pos: 120
type: DSZ, layer: 1, pos: 245

Time for candidate selection: 0.00 seconds

### Candidate
type: DSZ, layer: 1, pos: 93

### Relational analysis ABCD of DS_DSZ1_DSZ1_DSZ1

#### Relational analysis ABCD result of DS_DSZ1_DSZ1_DSZ1
Status: Status.UNKNOWN
Output dim: 3, lower bound: -0.3282285, upper bound: 0.3282285
time: 1.32 seconds

### Relational analysis ABCD of DS_DSZ1_DSZ1_DSZ2

#### Relational analysis ABCD result of DS_DSZ1_DSZ1_DSZ2
Status: Status.UNKNOWN
Output dim: 3, lower bound: -0.3282285, upper bound: 0.3282285
time: 1.12 seconds

## BFS DS instance: DS_DSZ1_DSZ2

### Backsubstitution after applying DS history:
0: -0.5426022, 0.4665188, -0.5426022, 0.4665188, -1.0091211, 1.0091211
1: -0.3630026, 0.3911184, -0.3630026, 0.3911184, -0.7541210, 0.7541210
2: -0.5269889, 0.4806983, -0.5269889, 0.4806983, -1.0076871, 1.0076871
3: 0.7543352, 1.1590887, 0.7543352, 1.1590887, -0.4047535, 0.4047535
4: -0.4614211, 0.4625955, -0.4614211, 0.4625955, -0.9240167, 0.9240167
5: -0.4214845, 0.4672619, -0.4214845, 0.4672619, -0.8887464, 0.8887464
6: -0.5138452, 0.5371552, -0.5138452, 0.5371552, -1.0510004, 1.0510004
7: -0.4409947, 0.4028524, -0.4409947, 0.4028524, -0.8438470, 0.8438470
8: -0.4306176, 0.5533380, -0.4306176, 0.5533380, -0.9839556, 0.9839556
9: -0.4872051, 0.3854194, -0.4872051, 0.3854194, -0.8726246, 0.8726246

Time for backsubstitution: 0.77 seconds

### DS candidates at layer 1
type: DSZ, layer: 1, pos: 182
type: DSZ, layer: 1, pos: 245
type: DSZ, layer: 1, pos: 66
type: DSZ, layer: 1, pos: 0
type: DSZ, layer: 1, pos: 183
type: DSZ, layer: 1, pos: 172
type: DSZ, layer: 1, pos: 120
type: DSZ, layer: 1, pos: 83
type: DSZ, layer: 1, pos: 235
type: DSZ, layer: 1, pos: 221
type: DSZ, layer: 1, pos: 15
type: DSZ, layer: 1, pos: 198
type: DSZ, layer: 1, pos: 166
type: DSZ, layer: 1, pos: 71
type: DSZ, layer: 1, pos: 219
type: DSZ, layer: 1, pos: 201
type: DSZ, layer: 1, pos: 177
type: DSZ, layer: 1, pos: 220
type: DSZ, layer: 1, pos: 93
type: DSZ, layer: 1, pos: 240
type: DSZ, layer: 1, pos: 176
type: DSZ, layer: 1, pos: 233
type: DSZ, layer: 1, pos: 37
type: DSZ, layer: 1, pos: 175
type: DSZ, layer: 1, pos: 29
type: DSZ, layer: 1, pos: 178
type: DSZ, layer: 1, pos: 188
type: DSZ, layer: 1, pos: 136
type: DSZ, layer: 1, pos: 68
type: DSZ, layer: 1, pos: 161
type: DSZ, layer: 1, pos: 62
type: DSZ, layer: 1, pos: 13
type: DSZ, layer: 1, pos: 103
type: DSZ, layer: 1, pos: 251
type: DSZ, layer: 1, pos: 30

Time for candidate selection: 0.00 seconds

### Candidate
type: DSZ, layer: 1, pos: 182

### Relational analysis ABCD of DS_DSZ1_DSZ2_DSZ1

#### Relational analysis ABCD result of DS_DSZ1_DSZ2_DSZ1
Status: Status.VERIFIED
Output dim: 3, lower bound: -0.2811652, upper bound: 0.2811652
time: 0.98 seconds

### Relational analysis ABCD of DS_DSZ1_DSZ2_DSZ2

#### Relational analysis ABCD result of DS_DSZ1_DSZ2_DSZ2
Status: Status.VERIFIED
Output dim: 3, lower bound: -0.2811652, upper bound: 0.2811652
time: 0.97 seconds

## BFS DS instance: DS_DSZ2_DSZ1

### Backsubstitution after applying DS history:
0: -0.5426022, 0.4665188, -0.5426022, 0.4665188, -1.0091211, 1.0091211
1: -0.3630026, 0.3911184, -0.3630026, 0.3911184, -0.7541210, 0.7541210
2: -0.5269889, 0.4806983, -0.5269889, 0.4806983, -1.0076871, 1.0076871
3: 0.7543352, 1.1590887, 0.7543352, 1.1590887, -0.4047535, 0.4047535
4: -0.4614211, 0.4625955, -0.4614211, 0.4625955, -0.9240167, 0.9240167
5: -0.4214845, 0.4672619, -0.4214845, 0.4672619, -0.8887464, 0.8887464
6: -0.5138452, 0.5371552, -0.5138452, 0.5371552, -1.0510004, 1.0510004
7: -0.4409947, 0.4028524, -0.4409947, 0.4028524, -0.8438470, 0.8438470
8: -0.4306176, 0.5533380, -0.4306176, 0.5533380, -0.9839556, 0.9839556
9: -0.4872051, 0.3854194, -0.4872051, 0.3854194, -0.8726246, 0.8726246

Time for backsubstitution: 0.79 seconds

### DS candidates at layer 1
type: DSZ, layer: 1, pos: 120
type: DSZ, layer: 1, pos: 235
type: DSZ, layer: 1, pos: 62
type: DSZ, layer: 1, pos: 188
type: DSZ, layer: 1, pos: 66
type: DSZ, layer: 1, pos: 176
type: DSZ, layer: 1, pos: 15
type: DSZ, layer: 1, pos: 245
type: DSZ, layer: 1, pos: 172
type: DSZ, layer: 1, pos: 103
type: DSZ, layer: 1, pos: 219
type: DSZ, layer: 1, pos: 183
type: DSZ, layer: 1, pos: 233
type: DSZ, layer: 1, pos: 175
type: DSZ, layer: 1, pos: 220
type: DSZ, layer: 1, pos: 0
type: DSZ, layer: 1, pos: 198
type: DSZ, layer: 1, pos: 182
type: DSZ, layer: 1, pos: 93
type: DSZ, layer: 1, pos: 161
type: DSZ, layer: 1, pos: 71
type: DSZ, layer: 1, pos: 30
type: DSZ, layer: 1, pos: 13
type: DSZ, layer: 1, pos: 68
type: DSZ, layer: 1, pos: 136
type: DSZ, layer: 1, pos: 178
type: DSZ, layer: 1, pos: 83
type: DSZ, layer: 1, pos: 251
type: DSZ, layer: 1, pos: 240
type: DSZ, layer: 1, pos: 1
type: DSZ, layer: 1, pos: 29
type: DSZ, layer: 1, pos: 177
type: DSZ, layer: 1, pos: 166
type: DSZ, layer: 1, pos: 221
type: DSZ, layer: 1, pos: 37

Time for candidate selection: 0.00 seconds

### Candidate
type: DSZ, layer: 1, pos: 120

### Relational analysis ABCD of DS_DSZ2_DSZ1_DSZ1

#### Relational analysis ABCD result of DS_DSZ2_DSZ1_DSZ1
Status: Status.VERIFIED
Output dim: 3, lower bound: -0.2931644, upper bound: 0.2931644
time: 0.94 seconds

### Relational analysis ABCD of DS_DSZ2_DSZ1_DSZ2

#### Relational analysis ABCD result of DS_DSZ2_DSZ1_DSZ2
Status: Status.VERIFIED
Output dim: 3, lower bound: -0.2931644, upper bound: 0.2931644
time: 0.96 seconds

## BFS DS instance: DS_DSZ2_DSZ2

### Backsubstitution after applying DS history:
0: -0.5426022, 0.4665188, -0.5426022, 0.4665188, -1.0091211, 1.0091211
1: -0.3630026, 0.3911184, -0.3630026, 0.3911184, -0.7541210, 0.7541210
2: -0.5269889, 0.4806983, -0.5269889, 0.4806983, -1.0076871, 1.0076871
3: 0.7543352, 1.1590887, 0.7543352, 1.1590887, -0.4047535, 0.4047535
4: -0.4614211, 0.4625955, -0.4614211, 0.4625955, -0.9240167, 0.9240167
5: -0.4214845, 0.4672619, -0.4214845, 0.4672619, -0.8887464, 0.8887464
6: -0.5138452, 0.5371552, -0.5138452, 0.5371552, -1.0510004, 1.0510004
7: -0.4409947, 0.4028524, -0.4409947, 0.4028524, -0.8438470, 0.8438470
8: -0.4306176, 0.5533380, -0.4306176, 0.5533380, -0.9839556, 0.9839556
9: -0.4872051, 0.3854194, -0.4872051, 0.3854194, -0.8726246, 0.8726246

Time for backsubstitution: 0.80 seconds

### DS candidates at layer 1
type: DSZ, layer: 1, pos: 15
type: DSZ, layer: 1, pos: 221
type: DSZ, layer: 1, pos: 245
type: DSZ, layer: 1, pos: 120
type: DSZ, layer: 1, pos: 0
type: DSZ, layer: 1, pos: 176
type: DSZ, layer: 1, pos: 251
type: DSZ, layer: 1, pos: 175
type: DSZ, layer: 1, pos: 93
type: DSZ, layer: 1, pos: 183
type: DSZ, layer: 1, pos: 37
type: DSZ, layer: 1, pos: 219
type: DSZ, layer: 1, pos: 233
type: DSZ, layer: 1, pos: 1
type: DSZ, layer: 1, pos: 30
type: DSZ, layer: 1, pos: 66
type: DSZ, layer: 1, pos: 220
type: DSZ, layer: 1, pos: 166
type: DSZ, layer: 1, pos: 13
type: DSZ, layer: 1, pos: 83
type: DSZ, layer: 1, pos: 136
type: DSZ, layer: 1, pos: 177
type: DSZ, layer: 1, pos: 178
type: DSZ, layer: 1, pos: 62
type: DSZ, layer: 1, pos: 71
type: DSZ, layer: 1, pos: 188
type: DSZ, layer: 1, pos: 198
type: DSZ, layer: 1, pos: 172
type: DSZ, layer: 1, pos: 240
type: DSZ, layer: 1, pos: 29
type: DSZ, layer: 1, pos: 235
type: DSZ, layer: 1, pos: 68
type: DSZ, layer: 1, pos: 103
type: DSZ, layer: 1, pos: 161
type: DSZ, layer: 1, pos: 182

Time for candidate selection: 0.00 seconds

### Candidate
type: DSZ, layer: 1, pos: 15

### Relational analysis ABCD of DS_DSZ2_DSZ2_DSZ1

#### Relational analysis ABCD result of DS_DSZ2_DSZ2_DSZ1
Status: Status.UNKNOWN
Output dim: 3, lower bound: -0.3281160, upper bound: 0.3281160
time: 1.24 seconds

### Relational analysis ABCD of DS_DSZ2_DSZ2_DSZ2

#### Relational analysis ABCD result of DS_DSZ2_DSZ2_DSZ2
Status: Status.UNKNOWN
Output dim: 3, lower bound: -0.3281160, upper bound: 0.3281160
time: 1.22 seconds

## Summary of splitting (split count: 2)
- Time for DS candidates: 3.26 seconds
DS_DSZ1_DSZ1_DSZ1, status: Status.UNKNOWN, split count: 3, time: 3.26
Output dim: 3, lower bound: -0.3282285, upper bound: 0.3282285
DS_DSZ1_DSZ1_DSZ2, status: Status.UNKNOWN, split count: 3, time: 3.26
Output dim: 3, lower bound: -0.3282285, upper bound: 0.3282285
DS_DSZ1_DSZ2_DSZ1, status: Status.VERIFIED, split count: 3, time: 3.26
Output dim: 3, lower bound: -0.2811652, upper bound: 0.2811652
DS_DSZ1_DSZ2_DSZ2, status: Status.VERIFIED, split count: 3, time: 3.26
Output dim: 3, lower bound: -0.2811652, upper bound: 0.2811652
DS_DSZ2_DSZ1_DSZ1, status: Status.VERIFIED, split count: 3, time: 3.26
Output dim: 3, lower bound: -0.2931644, upper bound: 0.2931644
DS_DSZ2_DSZ1_DSZ2, status: Status.VERIFIED, split count: 3, time: 3.26
Output dim: 3, lower bound: -0.2931644, upper bound: 0.2931644
DS_DSZ2_DSZ2_DSZ1, status: Status.UNKNOWN, split count: 3, time: 3.26
Output dim: 3, lower bound: -0.3281160, upper bound: 0.3281160
DS_DSZ2_DSZ2_DSZ2, status: Status.UNKNOWN, split count: 3, time: 3.26
Output dim: 3, lower bound: -0.3281160, upper bound: 0.3281160

## BFS DS instance: DS_DSZ1_DSZ1_DSZ1

### Backsubstitution after applying DS history:
0: -0.5426022, 0.4665188, -0.5426022, 0.4665188, -1.0091211, 1.0091211
1: -0.3630026, 0.3911184, -0.3630026, 0.3911184, -0.7541210, 0.7541210
2: -0.5269889, 0.4806983, -0.5269889, 0.4806983, -1.0076871, 1.0076871
3: 0.7543352, 1.1590887, 0.7543352, 1.1590887, -0.4047535, 0.4047535
4: -0.4614211, 0.4625955, -0.4614211, 0.4625955, -0.9240167, 0.9240167
5: -0.4214845, 0.4672619, -0.4214845, 0.4672619, -0.8887464, 0.8887464
6: -0.5138452, 0.5371552, -0.5138452, 0.5371552, -1.0510004, 1.0510004
7: -0.4409947, 0.4028524, -0.4409947, 0.4028524, -0.8438470, 0.8438470
8: -0.4306176, 0.5533380, -0.4306176, 0.5533380, -0.9839556, 0.9839556
9: -0.4872051, 0.3854194, -0.4872051, 0.3854194, -0.8726246, 0.8726246

Time for backsubstitution: 0.77 seconds

### DS candidates at layer 1
type: DSZ, layer: 1, pos: 245
type: DSZ, layer: 1, pos: 29
type: DSZ, layer: 1, pos: 68
type: DSZ, layer: 1, pos: 15
type: DSZ, layer: 1, pos: 13
type: DSZ, layer: 1, pos: 30
type: DSZ, layer: 1, pos: 161
type: DSZ, layer: 1, pos: 83
type: DSZ, layer: 1, pos: 177
type: DSZ, layer: 1, pos: 178
type: DSZ, layer: 1, pos: 175
type: DSZ, layer: 1, pos: 62
type: DSZ, layer: 1, pos: 220
type: DSZ, layer: 1, pos: 37
type: DSZ, layer: 1, pos: 0
type: DSZ, layer: 1, pos: 136
type: DSZ, layer: 1, pos: 233
type: DSZ, layer: 1, pos: 71
type: DSZ, layer: 1, pos: 201
type: DSZ, layer: 1, pos: 182
type: DSZ, layer: 1, pos: 240
type: DSZ, layer: 1, pos: 176
type: DSZ, layer: 1, pos: 235
type: DSZ, layer: 1, pos: 251
type: DSZ, layer: 1, pos: 221
type: DSZ, layer: 1, pos: 103
type: DSZ, layer: 1, pos: 172
type: DSZ, layer: 1, pos: 219
type: DSZ, layer: 1, pos: 183
type: DSZ, layer: 1, pos: 120
type: DSZ, layer: 1, pos: 198
type: DSZ, layer: 1, pos: 166
type: DSZ, layer: 1, pos: 188
type: DSZ, layer: 1, pos: 66

Time for candidate selection: 0.00 seconds

### Candidate
type: DSZ, layer: 1, pos: 245

### Relational analysis ABCD of DS_DSZ1_DSZ1_DSZ1_DSZ1

#### Relational analysis ABCD result of DS_DSZ1_DSZ1_DSZ1_DSZ1
Status: Status.VERIFIED
Output dim: 3, lower bound: -0.2950990, upper bound: 0.2950990
time: 0.92 seconds

### Relational analysis ABCD of DS_DSZ1_DSZ1_DSZ1_DSZ2

#### Relational analysis ABCD result of DS_DSZ1_DSZ1_DSZ1_DSZ2
Status: Status.VERIFIED
Output dim: 3, lower bound: -0.2950990, upper bound: 0.2950990
time: 0.92 seconds

## BFS DS instance: DS_DSZ1_DSZ1_DSZ2

### Backsubstitution after applying DS history:
0: -0.5426022, 0.4665188, -0.5426022, 0.4665188, -1.0091211, 1.0091211
1: -0.3630026, 0.3911184, -0.3630026, 0.3911184, -0.7541210, 0.7541210
2: -0.5269889, 0.4806983, -0.5269889, 0.4806983, -1.0076871, 1.0076871
3: 0.7543352, 1.1590887, 0.7543352, 1.1590887, -0.4047535, 0.4047535
4: -0.4614211, 0.4625955, -0.4614211, 0.4625955, -0.9240167, 0.9240167
5: -0.4214845, 0.4672619, -0.4214845, 0.4672619, -0.8887464, 0.8887464
6: -0.5138452, 0.5371552, -0.5138452, 0.5371552, -1.0510004, 1.0510004
7: -0.4409947, 0.4028524, -0.4409947, 0.4028524, -0.8438470, 0.8438470
8: -0.4306176, 0.5533380, -0.4306176, 0.5533380, -0.9839556, 0.9839556
9: -0.4872051, 0.3854194, -0.4872051, 0.3854194, -0.8726246, 0.8726246

Time for backsubstitution: 0.77 seconds

### DS candidates at layer 1
type: DSZ, layer: 1, pos: 172
type: DSZ, layer: 1, pos: 15
type: DSZ, layer: 1, pos: 62
type: DSZ, layer: 1, pos: 37
type: DSZ, layer: 1, pos: 30
type: DSZ, layer: 1, pos: 201
type: DSZ, layer: 1, pos: 183
type: DSZ, layer: 1, pos: 103
type: DSZ, layer: 1, pos: 68
type: DSZ, layer: 1, pos: 166
type: DSZ, layer: 1, pos: 198
type: DSZ, layer: 1, pos: 219
type: DSZ, layer: 1, pos: 182
type: DSZ, layer: 1, pos: 245
type: DSZ, layer: 1, pos: 221
type: DSZ, layer: 1, pos: 178
type: DSZ, layer: 1, pos: 83
type: DSZ, layer: 1, pos: 177
type: DSZ, layer: 1, pos: 0
type: DSZ, layer: 1, pos: 188
type: DSZ, layer: 1, pos: 176
type: DSZ, layer: 1, pos: 251
type: DSZ, layer: 1, pos: 233
type: DSZ, layer: 1, pos: 120
type: DSZ, layer: 1, pos: 240
type: DSZ, layer: 1, pos: 235
type: DSZ, layer: 1, pos: 71
type: DSZ, layer: 1, pos: 220
type: DSZ, layer: 1, pos: 66
type: DSZ, layer: 1, pos: 175
type: DSZ, layer: 1, pos: 136
type: DSZ, layer: 1, pos: 29
type: DSZ, layer: 1, pos: 13
type: DSZ, layer: 1, pos: 161

Time for candidate selection: 0.00 seconds

### Candidate
type: DSZ, layer: 1, pos: 172

### Relational analysis ABCD of DS_DSZ1_DSZ1_DSZ2_DSZ1

#### Relational analysis ABCD result of DS_DSZ1_DSZ1_DSZ2_DSZ1
Status: Status.VERIFIED
Output dim: 3, lower bound: -0.2950689, upper bound: 0.2950689
time: 0.94 seconds

### Relational analysis ABCD of DS_DSZ1_DSZ1_DSZ2_DSZ2

#### Relational analysis ABCD result of DS_DSZ1_DSZ1_DSZ2_DSZ2
Status: Status.VERIFIED
Output dim: 3, lower bound: -0.2950689, upper bound: 0.2950689
time: 0.96 seconds

## BFS DS instance: DS_DSZ2_DSZ2_DSZ1

### Backsubstitution after applying DS history:
0: -0.5426022, 0.4665188, -0.5426022, 0.4665188, -1.0091211, 1.0091211
1: -0.3630026, 0.3911184, -0.3630026, 0.3911184, -0.7541210, 0.7541210
2: -0.5269889, 0.4806983, -0.5269889, 0.4806983, -1.0076871, 1.0076871
3: 0.7543352, 1.1590887, 0.7543352, 1.1590887, -0.4047535, 0.4047535
4: -0.4614211, 0.4625955, -0.4614211, 0.4625955, -0.9240167, 0.9240167
5: -0.4214845, 0.4672619, -0.4214845, 0.4672619, -0.8887464, 0.8887464
6: -0.5138452, 0.5371552, -0.5138452, 0.5371552, -1.0510004, 1.0510004
7: -0.4409947, 0.4028524, -0.4409947, 0.4028524, -0.8438470, 0.8438470
8: -0.4306176, 0.5533380, -0.4306176, 0.5533380, -0.9839556, 0.9839556
9: -0.4872051, 0.3854194, -0.4872051, 0.3854194, -0.8726246, 0.8726246

Time for backsubstitution: 0.78 seconds

### DS candidates at layer 1
type: DSZ, layer: 1, pos: 93
type: DSZ, layer: 1, pos: 29
type: DSZ, layer: 1, pos: 103
type: DSZ, layer: 1, pos: 182
type: DSZ, layer: 1, pos: 188
type: DSZ, layer: 1, pos: 183
type: DSZ, layer: 1, pos: 220
type: DSZ, layer: 1, pos: 175
type: DSZ, layer: 1, pos: 136
type: DSZ, layer: 1, pos: 0
type: DSZ, layer: 1, pos: 245
type: DSZ, layer: 1, pos: 233
type: DSZ, layer: 1, pos: 161
type: DSZ, layer: 1, pos: 221
type: DSZ, layer: 1, pos: 178
type: DSZ, layer: 1, pos: 251
type: DSZ, layer: 1, pos: 176
type: DSZ, layer: 1, pos: 240
type: DSZ, layer: 1, pos: 71
type: DSZ, layer: 1, pos: 120
type: DSZ, layer: 1, pos: 13
type: DSZ, layer: 1, pos: 30
type: DSZ, layer: 1, pos: 198
type: DSZ, layer: 1, pos: 172
type: DSZ, layer: 1, pos: 66
type: DSZ, layer: 1, pos: 62
type: DSZ, layer: 1, pos: 166
type: DSZ, layer: 1, pos: 235
type: DSZ, layer: 1, pos: 37
type: DSZ, layer: 1, pos: 219
type: DSZ, layer: 1, pos: 1
type: DSZ, layer: 1, pos: 177
type: DSZ, layer: 1, pos: 68
type: DSZ, layer: 1, pos: 83

Time for candidate selection: 0.00 seconds

### Candidate
type: DSZ, layer: 1, pos: 93

### Relational analysis ABCD of DS_DSZ2_DSZ2_DSZ1_DSZ1

#### Relational analysis ABCD result of DS_DSZ2_DSZ2_DSZ1_DSZ1
Status: Status.UNKNOWN
Output dim: 3, lower bound: -0.3281160, upper bound: 0.3281160
time: 1.06 seconds

### Relational analysis ABCD of DS_DSZ2_DSZ2_DSZ1_DSZ2

#### Relational analysis ABCD result of DS_DSZ2_DSZ2_DSZ1_DSZ2
Status: Status.UNKNOWN
Output dim: 3, lower bound: -0.3281160, upper bound: 0.3281160
time: 1.15 seconds

## BFS DS instance: DS_DSZ2_DSZ2_DSZ2

### Backsubstitution after applying DS history:
0: -0.5426022, 0.4665188, -0.5426022, 0.4665188, -1.0091211, 1.0091211
1: -0.3630026, 0.3911184, -0.3630026, 0.3911184, -0.7541210, 0.7541210
2: -0.5269889, 0.4806983, -0.5269889, 0.4806983, -1.0076871, 1.0076871
3: 0.7543352, 1.1590887, 0.7543352, 1.1590887, -0.4047535, 0.4047535
4: -0.4614211, 0.4625955, -0.4614211, 0.4625955, -0.9240167, 0.9240167
5: -0.4214845, 0.4672619, -0.4214845, 0.4672619, -0.8887464, 0.8887464
6: -0.5138452, 0.5371552, -0.5138452, 0.5371552, -1.0510004, 1.0510004
7: -0.4409947, 0.4028524, -0.4409947, 0.4028524, -0.8438470, 0.8438470
8: -0.4306176, 0.5533380, -0.4306176, 0.5533380, -0.9839556, 0.9839556
9: -0.4872051, 0.3854194, -0.4872051, 0.3854194, -0.8726246, 0.8726246

Time for backsubstitution: 0.77 seconds

### DS candidates at layer 1
type: DSZ, layer: 1, pos: 177
type: DSZ, layer: 1, pos: 13
type: DSZ, layer: 1, pos: 71
type: DSZ, layer: 1, pos: 136
type: DSZ, layer: 1, pos: 235
type: DSZ, layer: 1, pos: 0
type: DSZ, layer: 1, pos: 29
type: DSZ, layer: 1, pos: 166
type: DSZ, layer: 1, pos: 245
type: DSZ, layer: 1, pos: 251
type: DSZ, layer: 1, pos: 188
type: DSZ, layer: 1, pos: 1
type: DSZ, layer: 1, pos: 221
type: DSZ, layer: 1, pos: 172
type: DSZ, layer: 1, pos: 233
type: DSZ, layer: 1, pos: 198
type: DSZ, layer: 1, pos: 178
type: DSZ, layer: 1, pos: 175
type: DSZ, layer: 1, pos: 219
type: DSZ, layer: 1, pos: 240
type: DSZ, layer: 1, pos: 176
type: DSZ, layer: 1, pos: 93
type: DSZ, layer: 1, pos: 103
type: DSZ, layer: 1, pos: 83
type: DSZ, layer: 1, pos: 66
type: DSZ, layer: 1, pos: 30
type: DSZ, layer: 1, pos: 220
type: DSZ, layer: 1, pos: 37
type: DSZ, layer: 1, pos: 120
type: DSZ, layer: 1, pos: 183
type: DSZ, layer: 1, pos: 182
type: DSZ, layer: 1, pos: 161
type: DSZ, layer: 1, pos: 62
type: DSZ, layer: 1, pos: 68

Time for candidate selection: 0.00 seconds

### Candidate
type: DSZ, layer: 1, pos: 177

### Relational analysis ABCD of DS_DSZ2_DSZ2_DSZ2_DSZ1

#### Relational analysis ABCD result of DS_DSZ2_DSZ2_DSZ2_DSZ1
Status: Status.UNKNOWN
Output dim: 3, lower bound: -0.3281093, upper bound: 0.3281093
time: 1.03 seconds

### Relational analysis ABCD of DS_DSZ2_DSZ2_DSZ2_DSZ2

#### Relational analysis ABCD result of DS_DSZ2_DSZ2_DSZ2_DSZ2
Status: Status.UNKNOWN
Output dim: 3, lower bound: -0.3281093, upper bound: 0.3281093
time: 1.04 seconds

## Summary of splitting (split count: 3)
- Time for DS candidates: 2.84 seconds
DS_DSZ1_DSZ1_DSZ1_DSZ1, status: Status.VERIFIED, split count: 4, time: 2.84
Output dim: 3, lower bound: -0.2950990, upper bound: 0.2950990
DS_DSZ1_DSZ1_DSZ1_DSZ2, status: Status.VERIFIED, split count: 4, time: 2.84
Output dim: 3, lower bound: -0.2950990, upper bound: 0.2950990
DS_DSZ1_DSZ1_DSZ2_DSZ1, status: Status.VERIFIED, split count: 4, time: 2.84
Output dim: 3, lower bound: -0.2950689, upper bound: 0.2950689
DS_DSZ1_DSZ1_DSZ2_DSZ2, status: Status.VERIFIED, split count: 4, time: 2.84
Output dim: 3, lower bound: -0.2950689, upper bound: 0.2950689
DS_DSZ2_DSZ2_DSZ1_DSZ1, status: Status.UNKNOWN, split count: 4, time: 2.84
Output dim: 3, lower bound: -0.3281160, upper bound: 0.3281160
DS_DSZ2_DSZ2_DSZ1_DSZ2, status: Status.UNKNOWN, split count: 4, time: 2.84
Output dim: 3, lower bound: -0.3281160, upper bound: 0.3281160
DS_DSZ2_DSZ2_DSZ2_DSZ1, status: Status.UNKNOWN, split count: 4, time: 2.84
Output dim: 3, lower bound: -0.3281093, upper bound: 0.3281093
DS_DSZ2_DSZ2_DSZ2_DSZ2, status: Status.UNKNOWN, split count: 4, time: 2.84
Output dim: 3, lower bound: -0.3281093, upper bound: 0.3281093

## BFS DS instance: DS_DSZ2_DSZ2_DSZ1_DSZ1

### Backsubstitution after applying DS history:
0: -0.5426022, 0.4665188, -0.5426022, 0.4665188, -1.0091211, 1.0091211
1: -0.3630026, 0.3911184, -0.3630026, 0.3911184, -0.7541210, 0.7541210
2: -0.5269889, 0.4806983, -0.5269889, 0.4806983, -1.0076871, 1.0076871
3: 0.7543352, 1.1590887, 0.7543352, 1.1590887, -0.4047535, 0.4047535
4: -0.4614211, 0.4625955, -0.4614211, 0.4625955, -0.9240167, 0.9240167
5: -0.4214845, 0.4672619, -0.4214845, 0.4672619, -0.8887464, 0.8887464
6: -0.5138452, 0.5371552, -0.5138452, 0.5371552, -1.0510004, 1.0510004
7: -0.4409947, 0.4028524, -0.4409947, 0.4028524, -0.8438470, 0.8438470
8: -0.4306176, 0.5533380, -0.4306176, 0.5533380, -0.9839556, 0.9839556
9: -0.4872051, 0.3854194, -0.4872051, 0.3854194, -0.8726246, 0.8726246

Time for backsubstitution: 0.77 seconds

### DS candidates at layer 1
type: DSZ, layer: 1, pos: 219
type: DSZ, layer: 1, pos: 37
type: DSZ, layer: 1, pos: 176
type: DSZ, layer: 1, pos: 13
type: DSZ, layer: 1, pos: 220
type: DSZ, layer: 1, pos: 71
type: DSZ, layer: 1, pos: 245
type: DSZ, layer: 1, pos: 1
type: DSZ, layer: 1, pos: 172
type: DSZ, layer: 1, pos: 103
type: DSZ, layer: 1, pos: 120
type: DSZ, layer: 1, pos: 175
type: DSZ, layer: 1, pos: 233
type: DSZ, layer: 1, pos: 0
type: DSZ, layer: 1, pos: 66
type: DSZ, layer: 1, pos: 161
type: DSZ, layer: 1, pos: 29
type: DSZ, layer: 1, pos: 221
type: DSZ, layer: 1, pos: 177
type: DSZ, layer: 1, pos: 166
type: DSZ, layer: 1, pos: 240
type: DSZ, layer: 1, pos: 62
type: DSZ, layer: 1, pos: 83
type: DSZ, layer: 1, pos: 198
type: DSZ, layer: 1, pos: 68
type: DSZ, layer: 1, pos: 251
type: DSZ, layer: 1, pos: 183
type: DSZ, layer: 1, pos: 178
type: DSZ, layer: 1, pos: 30
type: DSZ, layer: 1, pos: 182
type: DSZ, layer: 1, pos: 136
type: DSZ, layer: 1, pos: 188
type: DSZ, layer: 1, pos: 235

Time for candidate selection: 0.00 seconds

### Candidate
type: DSZ, layer: 1, pos: 219

### Relational analysis ABCD of DS_DSZ2_DSZ2_DSZ1_DSZ1_DSZ1
Optimization infeasible because this subproblem isn't reachable.

### Candidate
type: DSZ, layer: 1, pos: 37

### Relational analysis ABCD of DS_DSZ2_DSZ2_DSZ1_DSZ1_DSZ1

#### Relational analysis ABCD result of DS_DSZ2_DSZ2_DSZ1_DSZ1_DSZ1
Status: Status.VERIFIED
Output dim: 3, lower bound: -0.2950242, upper bound: 0.2950242
time: 1.02 seconds

### Relational analysis ABCD of DS_DSZ2_DSZ2_DSZ1_DSZ1_DSZ2

#### Relational analysis ABCD result of DS_DSZ2_DSZ2_DSZ1_DSZ1_DSZ2
Status: Status.VERIFIED
Output dim: 3, lower bound: -0.2950242, upper bound: 0.2950242
time: 1.03 seconds

## BFS DS instance: DS_DSZ2_DSZ2_DSZ1_DSZ2

### Backsubstitution after applying DS history:
0: -0.5426022, 0.4665188, -0.5426022, 0.4665188, -1.0091211, 1.0091211
1: -0.3630026, 0.3911184, -0.3630026, 0.3911184, -0.7541210, 0.7541210
2: -0.5269889, 0.4806983, -0.5269889, 0.4806983, -1.0076871, 1.0076871
3: 0.7543352, 1.1590887, 0.7543352, 1.1590887, -0.4047535, 0.4047535
4: -0.4614211, 0.4625955, -0.4614211, 0.4625955, -0.9240167, 0.9240167
5: -0.4214845, 0.4672619, -0.4214845, 0.4672619, -0.8887464, 0.8887464
6: -0.5138452, 0.5371552, -0.5138452, 0.5371552, -1.0510004, 1.0510004
7: -0.4409947, 0.4028524, -0.4409947, 0.4028524, -0.8438470, 0.8438470
8: -0.4306176, 0.5533380, -0.4306176, 0.5533380, -0.9839556, 0.9839556
9: -0.4872051, 0.3854194, -0.4872051, 0.3854194, -0.8726246, 0.8726246

Time for backsubstitution: 0.79 seconds

### DS candidates at layer 1
type: DSZ, layer: 1, pos: 188
type: DSZ, layer: 1, pos: 1
type: DSZ, layer: 1, pos: 30
type: DSZ, layer: 1, pos: 233
type: DSZ, layer: 1, pos: 178
type: DSZ, layer: 1, pos: 71
type: DSZ, layer: 1, pos: 62
type: DSZ, layer: 1, pos: 219
type: DSZ, layer: 1, pos: 37
type: DSZ, layer: 1, pos: 235
type: DSZ, layer: 1, pos: 183
type: DSZ, layer: 1, pos: 103
type: DSZ, layer: 1, pos: 176
type: DSZ, layer: 1, pos: 136
type: DSZ, layer: 1, pos: 120
type: DSZ, layer: 1, pos: 13
type: DSZ, layer: 1, pos: 68
type: DSZ, layer: 1, pos: 83
type: DSZ, layer: 1, pos: 66
type: DSZ, layer: 1, pos: 245
type: DSZ, layer: 1, pos: 221
type: DSZ, layer: 1, pos: 175
type: DSZ, layer: 1, pos: 177
type: DSZ, layer: 1, pos: 182
type: DSZ, layer: 1, pos: 0
type: DSZ, layer: 1, pos: 172
type: DSZ, layer: 1, pos: 198
type: DSZ, layer: 1, pos: 220
type: DSZ, layer: 1, pos: 166
type: DSZ, layer: 1, pos: 251
type: DSZ, layer: 1, pos: 161
type: DSZ, layer: 1, pos: 29
type: DSZ, layer: 1, pos: 240

Time for candidate selection: 0.00 seconds

### Candidate
type: DSZ, layer: 1, pos: 188

### Relational analysis ABCD of DS_DSZ2_DSZ2_DSZ1_DSZ2_DSZ1

#### Relational analysis ABCD result of DS_DSZ2_DSZ2_DSZ1_DSZ2_DSZ1
Status: Status.UNKNOWN
Output dim: 3, lower bound: -0.3264099, upper bound: 0.3264099
time: 1.19 seconds

### Relational analysis ABCD of DS_DSZ2_DSZ2_DSZ1_DSZ2_DSZ2

#### Relational analysis ABCD result of DS_DSZ2_DSZ2_DSZ1_DSZ2_DSZ2
Status: Status.UNKNOWN
Output dim: 3, lower bound: -0.3264099, upper bound: 0.3264099
time: 1.17 seconds

## BFS DS instance: DS_DSZ2_DSZ2_DSZ2_DSZ1

### Backsubstitution after applying DS history:
0: -0.5426022, 0.4665188, -0.5426022, 0.4665188, -1.0091211, 1.0091211
1: -0.3630026, 0.3911184, -0.3630026, 0.3911184, -0.7541210, 0.7541210
2: -0.5269889, 0.4806983, -0.5269889, 0.4806983, -1.0076871, 1.0076871
3: 0.7543352, 1.1590887, 0.7543352, 1.1590887, -0.4047535, 0.4047535
4: -0.4614211, 0.4625955, -0.4614211, 0.4625955, -0.9240167, 0.9240167
5: -0.4214845, 0.4672619, -0.4214845, 0.4672619, -0.8887464, 0.8887464
6: -0.5138452, 0.5371552, -0.5138452, 0.5371552, -1.0510004, 1.0510004
7: -0.4409947, 0.4028524, -0.4409947, 0.4028524, -0.8438470, 0.8438470
8: -0.4306176, 0.5533380, -0.4306176, 0.5533380, -0.9839556, 0.9839556
9: -0.4872051, 0.3854194, -0.4872051, 0.3854194, -0.8726246, 0.8726246

Time for backsubstitution: 0.81 seconds

### DS candidates at layer 1
type: DSZ, layer: 1, pos: 93
type: DSZ, layer: 1, pos: 136
type: DSZ, layer: 1, pos: 172
type: DSZ, layer: 1, pos: 29
type: DSZ, layer: 1, pos: 251
type: DSZ, layer: 1, pos: 178
type: DSZ, layer: 1, pos: 1
type: DSZ, layer: 1, pos: 198
type: DSZ, layer: 1, pos: 235
type: DSZ, layer: 1, pos: 240
type: DSZ, layer: 1, pos: 188
type: DSZ, layer: 1, pos: 183
type: DSZ, layer: 1, pos: 245
type: DSZ, layer: 1, pos: 161
type: DSZ, layer: 1, pos: 233
type: DSZ, layer: 1, pos: 120
type: DSZ, layer: 1, pos: 62
type: DSZ, layer: 1, pos: 71
type: DSZ, layer: 1, pos: 13
type: DSZ, layer: 1, pos: 219
type: DSZ, layer: 1, pos: 221
type: DSZ, layer: 1, pos: 68
type: DSZ, layer: 1, pos: 175
type: DSZ, layer: 1, pos: 103
type: DSZ, layer: 1, pos: 0
type: DSZ, layer: 1, pos: 83
type: DSZ, layer: 1, pos: 220
type: DSZ, layer: 1, pos: 66
type: DSZ, layer: 1, pos: 30
type: DSZ, layer: 1, pos: 37
type: DSZ, layer: 1, pos: 166
type: DSZ, layer: 1, pos: 176
type: DSZ, layer: 1, pos: 182

Time for candidate selection: 0.00 seconds

### Candidate
type: DSZ, layer: 1, pos: 93

### Relational analysis ABCD of DS_DSZ2_DSZ2_DSZ2_DSZ1_DSZ1

#### Relational analysis ABCD result of DS_DSZ2_DSZ2_DSZ2_DSZ1_DSZ1
Status: Status.UNKNOWN
Output dim: 3, lower bound: -0.3281093, upper bound: 0.3281093
time: 1.13 seconds

### Relational analysis ABCD of DS_DSZ2_DSZ2_DSZ2_DSZ1_DSZ2

#### Relational analysis ABCD result of DS_DSZ2_DSZ2_DSZ2_DSZ1_DSZ2
Status: Status.UNKNOWN
Output dim: 3, lower bound: -0.3281093, upper bound: 0.3281093
time: 1.10 seconds

## BFS DS instance: DS_DSZ2_DSZ2_DSZ2_DSZ2

### Backsubstitution after applying DS history:
0: -0.5426022, 0.4665188, -0.5426022, 0.4665188, -1.0091211, 1.0091211
1: -0.3630026, 0.3911184, -0.3630026, 0.3911184, -0.7541210, 0.7541210
2: -0.5269889, 0.4806983, -0.5269889, 0.4806983, -1.0076871, 1.0076871
3: 0.7543352, 1.1590887, 0.7543352, 1.1590887, -0.4047535, 0.4047535
4: -0.4614211, 0.4625955, -0.4614211, 0.4625955, -0.9240167, 0.9240167
5: -0.4214845, 0.4672619, -0.4214845, 0.4672619, -0.8887464, 0.8887464
6: -0.5138452, 0.5371552, -0.5138452, 0.5371552, -1.0510004, 1.0510004
7: -0.4409947, 0.4028524, -0.4409947, 0.4028524, -0.8438470, 0.8438470
8: -0.4306176, 0.5533380, -0.4306176, 0.5533380, -0.9839556, 0.9839556
9: -0.4872051, 0.3854194, -0.4872051, 0.3854194, -0.8726246, 0.8726246

Time for backsubstitution: 0.79 seconds

### DS candidates at layer 1
type: DSZ, layer: 1, pos: 136
type: DSZ, layer: 1, pos: 166
type: DSZ, layer: 1, pos: 198
type: DSZ, layer: 1, pos: 120
type: DSZ, layer: 1, pos: 93
type: DSZ, layer: 1, pos: 103
type: DSZ, layer: 1, pos: 220
type: DSZ, layer: 1, pos: 1
type: DSZ, layer: 1, pos: 178
type: DSZ, layer: 1, pos: 219
type: DSZ, layer: 1, pos: 233
type: DSZ, layer: 1, pos: 37
type: DSZ, layer: 1, pos: 62
type: DSZ, layer: 1, pos: 245
type: DSZ, layer: 1, pos: 161
type: DSZ, layer: 1, pos: 71
type: DSZ, layer: 1, pos: 221
type: DSZ, layer: 1, pos: 251
type: DSZ, layer: 1, pos: 188
type: DSZ, layer: 1, pos: 175
type: DSZ, layer: 1, pos: 66
type: DSZ, layer: 1, pos: 172
type: DSZ, layer: 1, pos: 235
type: DSZ, layer: 1, pos: 176
type: DSZ, layer: 1, pos: 0
type: DSZ, layer: 1, pos: 182
type: DSZ, layer: 1, pos: 68
type: DSZ, layer: 1, pos: 29
type: DSZ, layer: 1, pos: 240
type: DSZ, layer: 1, pos: 183
type: DSZ, layer: 1, pos: 83
type: DSZ, layer: 1, pos: 30
type: DSZ, layer: 1, pos: 13

Time for candidate selection: 0.00 seconds

### Candidate
type: DSZ, layer: 1, pos: 136

### Relational analysis ABCD of DS_DSZ2_DSZ2_DSZ2_DSZ2_DSZ1

#### Relational analysis ABCD result of DS_DSZ2_DSZ2_DSZ2_DSZ2_DSZ1
Status: Status.UNKNOWN
Output dim: 3, lower bound: -0.3277898, upper bound: 0.3277898
time: 1.19 seconds

### Relational analysis ABCD of DS_DSZ2_DSZ2_DSZ2_DSZ2_DSZ2

#### Relational analysis ABCD result of DS_DSZ2_DSZ2_DSZ2_DSZ2_DSZ2
Status: Status.UNKNOWN
Output dim: 3, lower bound: -0.3277898, upper bound: 0.3277898
time: 1.15 seconds

## Summary of splitting (split count: 4)
- Time for DS candidates: 3.14 seconds
DS_DSZ2_DSZ2_DSZ1_DSZ1_DSZ1, status: Status.VERIFIED, split count: 5, time: 3.14
Output dim: 3, lower bound: -0.2950242, upper bound: 0.2950242
DS_DSZ2_DSZ2_DSZ1_DSZ1_DSZ2, status: Status.VERIFIED, split count: 5, time: 3.14
Output dim: 3, lower bound: -0.2950242, upper bound: 0.2950242
DS_DSZ2_DSZ2_DSZ1_DSZ2_DSZ1, status: Status.UNKNOWN, split count: 5, time: 3.14
Output dim: 3, lower bound: -0.3264099, upper bound: 0.3264099
DS_DSZ2_DSZ2_DSZ1_DSZ2_DSZ2, status: Status.UNKNOWN, split count: 5, time: 3.14
Output dim: 3, lower bound: -0.3264099, upper bound: 0.3264099
DS_DSZ2_DSZ2_DSZ2_DSZ1_DSZ1, status: Status.UNKNOWN, split count: 5, time: 3.14
Output dim: 3, lower bound: -0.3281093, upper bound: 0.3281093
DS_DSZ2_DSZ2_DSZ2_DSZ1_DSZ2, status: Status.UNKNOWN, split count: 5, time: 3.14
Output dim: 3, lower bound: -0.3281093, upper bound: 0.3281093
DS_DSZ2_DSZ2_DSZ2_DSZ2_DSZ1, status: Status.UNKNOWN, split count: 5, time: 3.14
Output dim: 3, lower bound: -0.3277898, upper bound: 0.3277898
DS_DSZ2_DSZ2_DSZ2_DSZ2_DSZ2, status: Status.UNKNOWN, split count: 5, time: 3.14
Output dim: 3, lower bound: -0.3277898, upper bound: 0.3277898

## BFS DS instance: DS_DSZ2_DSZ2_DSZ1_DSZ2_DSZ1

### Backsubstitution after applying DS history:
0: -0.5426022, 0.4665188, -0.5426022, 0.4665188, -1.0091211, 1.0091211
1: -0.3630026, 0.3911184, -0.3630026, 0.3911184, -0.7541210, 0.7541210
2: -0.5269889, 0.4806983, -0.5269889, 0.4806983, -1.0076871, 1.0076871
3: 0.7543352, 1.1590887, 0.7543352, 1.1590887, -0.4047535, 0.4047535
4: -0.4614211, 0.4625955, -0.4614211, 0.4625955, -0.9240167, 0.9240167
5: -0.4214845, 0.4672619, -0.4214845, 0.4672619, -0.8887464, 0.8887464
6: -0.5138452, 0.5371552, -0.5138452, 0.5371552, -1.0510004, 1.0510004
7: -0.4409947, 0.4028524, -0.4409947, 0.4028524, -0.8438470, 0.8438470
8: -0.4306176, 0.5533380, -0.4306176, 0.5533380, -0.9839556, 0.9839556
9: -0.4872051, 0.3854194, -0.4872051, 0.3854194, -0.8726246, 0.8726246

Time for backsubstitution: 0.80 seconds

### DS candidates at layer 1
type: DSZ, layer: 1, pos: 176
type: DSZ, layer: 1, pos: 219
type: DSZ, layer: 1, pos: 177
type: DSZ, layer: 1, pos: 221
type: DSZ, layer: 1, pos: 71
type: DSZ, layer: 1, pos: 66
type: DSZ, layer: 1, pos: 220
type: DSZ, layer: 1, pos: 178
type: DSZ, layer: 1, pos: 62
type: DSZ, layer: 1, pos: 183
type: DSZ, layer: 1, pos: 68
type: DSZ, layer: 1, pos: 0
type: DSZ, layer: 1, pos: 182
type: DSZ, layer: 1, pos: 240
type: DSZ, layer: 1, pos: 29
type: DSZ, layer: 1, pos: 251
type: DSZ, layer: 1, pos: 37
type: DSZ, layer: 1, pos: 1
type: DSZ, layer: 1, pos: 30
type: DSZ, layer: 1, pos: 120
type: DSZ, layer: 1, pos: 136
type: DSZ, layer: 1, pos: 83
type: DSZ, layer: 1, pos: 235
type: DSZ, layer: 1, pos: 245
type: DSZ, layer: 1, pos: 103
type: DSZ, layer: 1, pos: 172
type: DSZ, layer: 1, pos: 13
type: DSZ, layer: 1, pos: 175
type: DSZ, layer: 1, pos: 233
type: DSZ, layer: 1, pos: 198
type: DSZ, layer: 1, pos: 161
type: DSZ, layer: 1, pos: 166

Time for candidate selection: 0.00 seconds

### Candidate
type: DSZ, layer: 1, pos: 176

### Relational analysis ABCD of DS_DSZ2_DSZ2_DSZ1_DSZ2_DSZ1_DSZ1

#### Relational analysis ABCD result of DS_DSZ2_DSZ2_DSZ1_DSZ2_DSZ1_DSZ1
Status: Status.VERIFIED
Output dim: 3, lower bound: -0.3059422, upper bound: 0.3059422
time: 1.43 seconds

### Relational analysis ABCD of DS_DSZ2_DSZ2_DSZ1_DSZ2_DSZ1_DSZ2

#### Relational analysis ABCD result of DS_DSZ2_DSZ2_DSZ1_DSZ2_DSZ1_DSZ2
Status: Status.VERIFIED
Output dim: 3, lower bound: -0.3059422, upper bound: 0.3059422
time: 1.42 seconds

## BFS DS instance: DS_DSZ2_DSZ2_DSZ1_DSZ2_DSZ2

### Backsubstitution after applying DS history:
0: -0.5426022, 0.4665188, -0.5426022, 0.4665188, -1.0091211, 1.0091211
1: -0.3630026, 0.3911184, -0.3630026, 0.3911184, -0.7541210, 0.7541210
2: -0.5269889, 0.4806983, -0.5269889, 0.4806983, -1.0076871, 1.0076871
3: 0.7543352, 1.1590887, 0.7543352, 1.1590887, -0.4047535, 0.4047535
4: -0.4614211, 0.4625955, -0.4614211, 0.4625955, -0.9240167, 0.9240167
5: -0.4214845, 0.4672619, -0.4214845, 0.4672619, -0.8887464, 0.8887464
6: -0.5138452, 0.5371552, -0.5138452, 0.5371552, -1.0510004, 1.0510004
7: -0.4409947, 0.4028524, -0.4409947, 0.4028524, -0.8438470, 0.8438470
8: -0.4306176, 0.5533380, -0.4306176, 0.5533380, -0.9839556, 0.9839556
9: -0.4872051, 0.3854194, -0.4872051, 0.3854194, -0.8726246, 0.8726246

Time for backsubstitution: 0.79 seconds

### DS candidates at layer 1
type: DSZ, layer: 1, pos: 175
type: DSZ, layer: 1, pos: 71
type: DSZ, layer: 1, pos: 182
type: DSZ, layer: 1, pos: 166
type: DSZ, layer: 1, pos: 0
type: DSZ, layer: 1, pos: 177
type: DSZ, layer: 1, pos: 29
type: DSZ, layer: 1, pos: 198
type: DSZ, layer: 1, pos: 161
type: DSZ, layer: 1, pos: 251
type: DSZ, layer: 1, pos: 103
type: DSZ, layer: 1, pos: 120
type: DSZ, layer: 1, pos: 245
type: DSZ, layer: 1, pos: 62
type: DSZ, layer: 1, pos: 68
type: DSZ, layer: 1, pos: 172
type: DSZ, layer: 1, pos: 240
type: DSZ, layer: 1, pos: 136
type: DSZ, layer: 1, pos: 83
type: DSZ, layer: 1, pos: 221
type: DSZ, layer: 1, pos: 176
type: DSZ, layer: 1, pos: 30
type: DSZ, layer: 1, pos: 1
type: DSZ, layer: 1, pos: 233
type: DSZ, layer: 1, pos: 13
type: DSZ, layer: 1, pos: 37
type: DSZ, layer: 1, pos: 66
type: DSZ, layer: 1, pos: 220
type: DSZ, layer: 1, pos: 235
type: DSZ, layer: 1, pos: 183
type: DSZ, layer: 1, pos: 219
type: DSZ, layer: 1, pos: 178

Time for candidate selection: 0.00 seconds

### Candidate
type: DSZ, layer: 1, pos: 175

### Relational analysis ABCD of DS_DSZ2_DSZ2_DSZ1_DSZ2_DSZ2_DSZ1

#### Relational analysis ABCD result of DS_DSZ2_DSZ2_DSZ1_DSZ2_DSZ2_DSZ1
Status: Status.UNKNOWN
Output dim: 3, lower bound: -0.3247424, upper bound: 0.3247424
time: 1.23 seconds

### Relational analysis ABCD of DS_DSZ2_DSZ2_DSZ1_DSZ2_DSZ2_DSZ2

#### Relational analysis ABCD result of DS_DSZ2_DSZ2_DSZ1_DSZ2_DSZ2_DSZ2
Status: Status.UNKNOWN
Output dim: 3, lower bound: -0.3247424, upper bound: 0.3247424
time: 1.19 seconds

## BFS DS instance: DS_DSZ2_DSZ2_DSZ2_DSZ1_DSZ1

### Backsubstitution after applying DS history:
0: -0.5426022, 0.4665188, -0.5426022, 0.4665188, -1.0091211, 1.0091211
1: -0.3630026, 0.3911184, -0.3630026, 0.3911184, -0.7541210, 0.7541210
2: -0.5269889, 0.4806983, -0.5269889, 0.4806983, -1.0076871, 1.0076871
3: 0.7543352, 1.1590887, 0.7543352, 1.1590887, -0.4047535, 0.4047535
4: -0.4614211, 0.4625955, -0.4614211, 0.4625955, -0.9240167, 0.9240167
5: -0.4214845, 0.4672619, -0.4214845, 0.4672619, -0.8887464, 0.8887464
6: -0.5138452, 0.5371552, -0.5138452, 0.5371552, -1.0510004, 1.0510004
7: -0.4409947, 0.4028524, -0.4409947, 0.4028524, -0.8438470, 0.8438470
8: -0.4306176, 0.5533380, -0.4306176, 0.5533380, -0.9839556, 0.9839556
9: -0.4872051, 0.3854194, -0.4872051, 0.3854194, -0.8726246, 0.8726246

Time for backsubstitution: 0.80 seconds

### DS candidates at layer 1
type: DSZ, layer: 1, pos: 188
type: DSZ, layer: 1, pos: 30
type: DSZ, layer: 1, pos: 233
type: DSZ, layer: 1, pos: 0
type: DSZ, layer: 1, pos: 68
type: DSZ, layer: 1, pos: 176
type: DSZ, layer: 1, pos: 245
type: DSZ, layer: 1, pos: 240
type: DSZ, layer: 1, pos: 251
type: DSZ, layer: 1, pos: 172
type: DSZ, layer: 1, pos: 161
type: DSZ, layer: 1, pos: 13
type: DSZ, layer: 1, pos: 1
type: DSZ, layer: 1, pos: 182
type: DSZ, layer: 1, pos: 175
type: DSZ, layer: 1, pos: 183
type: DSZ, layer: 1, pos: 198
type: DSZ, layer: 1, pos: 37
type: DSZ, layer: 1, pos: 120
type: DSZ, layer: 1, pos: 83
type: DSZ, layer: 1, pos: 136
type: DSZ, layer: 1, pos: 103
type: DSZ, layer: 1, pos: 29
type: DSZ, layer: 1, pos: 62
type: DSZ, layer: 1, pos: 235
type: DSZ, layer: 1, pos: 221
type: DSZ, layer: 1, pos: 219
type: DSZ, layer: 1, pos: 178
type: DSZ, layer: 1, pos: 66
type: DSZ, layer: 1, pos: 166
type: DSZ, layer: 1, pos: 71
type: DSZ, layer: 1, pos: 220

Time for candidate selection: 0.00 seconds

### Candidate
type: DSZ, layer: 1, pos: 188

### Relational analysis ABCD of DS_DSZ2_DSZ2_DSZ2_DSZ1_DSZ1_DSZ1

#### Relational analysis ABCD result of DS_DSZ2_DSZ2_DSZ2_DSZ1_DSZ1_DSZ1
Status: Status.UNKNOWN
Output dim: 3, lower bound: -0.3264099, upper bound: 0.3264099
time: 1.09 seconds

### Relational analysis ABCD of DS_DSZ2_DSZ2_DSZ2_DSZ1_DSZ1_DSZ2

#### Relational analysis ABCD result of DS_DSZ2_DSZ2_DSZ2_DSZ1_DSZ1_DSZ2
Status: Status.UNKNOWN
Output dim: 3, lower bound: -0.3264099, upper bound: 0.3264099
time: 1.29 seconds

## BFS DS instance: DS_DSZ2_DSZ2_DSZ2_DSZ1_DSZ2

### Backsubstitution after applying DS history:
0: -0.5426022, 0.4665188, -0.5426022, 0.4665188, -1.0091211, 1.0091211
1: -0.3630026, 0.3911184, -0.3630026, 0.3911184, -0.7541210, 0.7541210
2: -0.5269889, 0.4806983, -0.5269889, 0.4806983, -1.0076871, 1.0076871
3: 0.7543352, 1.1590887, 0.7543352, 1.1590887, -0.4047535, 0.4047535
4: -0.4614211, 0.4625955, -0.4614211, 0.4625955, -0.9240167, 0.9240167
5: -0.4214845, 0.4672619, -0.4214845, 0.4672619, -0.8887464, 0.8887464
6: -0.5138452, 0.5371552, -0.5138452, 0.5371552, -1.0510004, 1.0510004
7: -0.4409947, 0.4028524, -0.4409947, 0.4028524, -0.8438470, 0.8438470
8: -0.4306176, 0.5533380, -0.4306176, 0.5533380, -0.9839556, 0.9839556
9: -0.4872051, 0.3854194, -0.4872051, 0.3854194, -0.8726246, 0.8726246

Time for backsubstitution: 0.79 seconds

### DS candidates at layer 1
type: DSZ, layer: 1, pos: 161
type: DSZ, layer: 1, pos: 166
type: DSZ, layer: 1, pos: 175
type: DSZ, layer: 1, pos: 68
type: DSZ, layer: 1, pos: 176
type: DSZ, layer: 1, pos: 182
type: DSZ, layer: 1, pos: 188
type: DSZ, layer: 1, pos: 30
type: DSZ, layer: 1, pos: 120
type: DSZ, layer: 1, pos: 37
type: DSZ, layer: 1, pos: 220
type: DSZ, layer: 1, pos: 83
type: DSZ, layer: 1, pos: 183
type: DSZ, layer: 1, pos: 66
type: DSZ, layer: 1, pos: 235
type: DSZ, layer: 1, pos: 172
type: DSZ, layer: 1, pos: 251
type: DSZ, layer: 1, pos: 103
type: DSZ, layer: 1, pos: 1
type: DSZ, layer: 1, pos: 219
type: DSZ, layer: 1, pos: 62
type: DSZ, layer: 1, pos: 221
type: DSZ, layer: 1, pos: 13
type: DSZ, layer: 1, pos: 245
type: DSZ, layer: 1, pos: 198
type: DSZ, layer: 1, pos: 240
type: DSZ, layer: 1, pos: 233
type: DSZ, layer: 1, pos: 71
type: DSZ, layer: 1, pos: 0
type: DSZ, layer: 1, pos: 136
type: DSZ, layer: 1, pos: 178
type: DSZ, layer: 1, pos: 29

Time for candidate selection: 0.00 seconds

### Candidate
type: DSZ, layer: 1, pos: 161

### Relational analysis ABCD of DS_DSZ2_DSZ2_DSZ2_DSZ1_DSZ2_DSZ1

#### Relational analysis ABCD result of DS_DSZ2_DSZ2_DSZ2_DSZ1_DSZ2_DSZ1
Status: Status.VERIFIED
Output dim: 3, lower bound: -0.2857863, upper bound: 0.2857863
time: 1.03 seconds

### Relational analysis ABCD of DS_DSZ2_DSZ2_DSZ2_DSZ1_DSZ2_DSZ2

#### Relational analysis ABCD result of DS_DSZ2_DSZ2_DSZ2_DSZ1_DSZ2_DSZ2
Status: Status.VERIFIED
Output dim: 3, lower bound: -0.2857863, upper bound: 0.2857863
time: 1.02 seconds

## BFS DS instance: DS_DSZ2_DSZ2_DSZ2_DSZ2_DSZ1

### Backsubstitution after applying DS history:
0: -0.5426022, 0.4665188, -0.5426022, 0.4665188, -1.0091211, 1.0091211
1: -0.3630026, 0.3911184, -0.3630026, 0.3911184, -0.7541210, 0.7541210
2: -0.5269889, 0.4806983, -0.5269889, 0.4806983, -1.0076871, 1.0076871
3: 0.7543352, 1.1590887, 0.7543352, 1.1590887, -0.4047535, 0.4047535
4: -0.4614211, 0.4625955, -0.4614211, 0.4625955, -0.9240167, 0.9240167
5: -0.4214845, 0.4672619, -0.4214845, 0.4672619, -0.8887464, 0.8887464
6: -0.5138452, 0.5371552, -0.5138452, 0.5371552, -1.0510004, 1.0510004
7: -0.4409947, 0.4028524, -0.4409947, 0.4028524, -0.8438470, 0.8438470
8: -0.4306176, 0.5533380, -0.4306176, 0.5533380, -0.9839556, 0.9839556
9: -0.4872051, 0.3854194, -0.4872051, 0.3854194, -0.8726246, 0.8726246

Time for backsubstitution: 0.78 seconds

### DS candidates at layer 1
type: DSZ, layer: 1, pos: 198
type: DSZ, layer: 1, pos: 182
type: DSZ, layer: 1, pos: 68
type: DSZ, layer: 1, pos: 245
type: DSZ, layer: 1, pos: 120
type: DSZ, layer: 1, pos: 220
type: DSZ, layer: 1, pos: 175
type: DSZ, layer: 1, pos: 172
type: DSZ, layer: 1, pos: 103
type: DSZ, layer: 1, pos: 251
type: DSZ, layer: 1, pos: 183
type: DSZ, layer: 1, pos: 93
type: DSZ, layer: 1, pos: 66
type: DSZ, layer: 1, pos: 221
type: DSZ, layer: 1, pos: 233
type: DSZ, layer: 1, pos: 219
type: DSZ, layer: 1, pos: 161
type: DSZ, layer: 1, pos: 1
type: DSZ, layer: 1, pos: 166
type: DSZ, layer: 1, pos: 62
type: DSZ, layer: 1, pos: 37
type: DSZ, layer: 1, pos: 176
type: DSZ, layer: 1, pos: 188
type: DSZ, layer: 1, pos: 29
type: DSZ, layer: 1, pos: 0
type: DSZ, layer: 1, pos: 13
type: DSZ, layer: 1, pos: 83
type: DSZ, layer: 1, pos: 235
type: DSZ, layer: 1, pos: 240
type: DSZ, layer: 1, pos: 71
type: DSZ, layer: 1, pos: 178
type: DSZ, layer: 1, pos: 30

Time for candidate selection: 0.00 seconds

### Candidate
type: DSZ, layer: 1, pos: 198

### Relational analysis ABCD of DS_DSZ2_DSZ2_DSZ2_DSZ2_DSZ1_DSZ1

#### Relational analysis ABCD result of DS_DSZ2_DSZ2_DSZ2_DSZ2_DSZ1_DSZ1
Status: Status.VERIFIED
Output dim: 3, lower bound: -0.3009578, upper bound: 0.3009578
time: 1.03 seconds

### Relational analysis ABCD of DS_DSZ2_DSZ2_DSZ2_DSZ2_DSZ1_DSZ2

#### Relational analysis ABCD result of DS_DSZ2_DSZ2_DSZ2_DSZ2_DSZ1_DSZ2
Status: Status.VERIFIED
Output dim: 3, lower bound: -0.3009578, upper bound: 0.3009578
time: 1.04 seconds

## BFS DS instance: DS_DSZ2_DSZ2_DSZ2_DSZ2_DSZ2

### Backsubstitution after applying DS history:
0: -0.5426022, 0.4665188, -0.5426022, 0.4665188, -1.0091211, 1.0091211
1: -0.3630026, 0.3911184, -0.3630026, 0.3911184, -0.7541210, 0.7541210
2: -0.5269889, 0.4806983, -0.5269889, 0.4806983, -1.0076871, 1.0076871
3: 0.7543352, 1.1590887, 0.7543352, 1.1590887, -0.4047535, 0.4047535
4: -0.4614211, 0.4625955, -0.4614211, 0.4625955, -0.9240167, 0.9240167
5: -0.4214845, 0.4672619, -0.4214845, 0.4672619, -0.8887464, 0.8887464
6: -0.5138452, 0.5371552, -0.5138452, 0.5371552, -1.0510004, 1.0510004
7: -0.4409947, 0.4028524, -0.4409947, 0.4028524, -0.8438470, 0.8438470
8: -0.4306176, 0.5533380, -0.4306176, 0.5533380, -0.9839556, 0.9839556
9: -0.4872051, 0.3854194, -0.4872051, 0.3854194, -0.8726246, 0.8726246

Time for backsubstitution: 0.80 seconds

### DS candidates at layer 1
type: DSZ, layer: 1, pos: 183
type: DSZ, layer: 1, pos: 120
type: DSZ, layer: 1, pos: 178
type: DSZ, layer: 1, pos: 0
type: DSZ, layer: 1, pos: 240
type: DSZ, layer: 1, pos: 29
type: DSZ, layer: 1, pos: 37
type: DSZ, layer: 1, pos: 71
type: DSZ, layer: 1, pos: 176
type: DSZ, layer: 1, pos: 251
type: DSZ, layer: 1, pos: 220
type: DSZ, layer: 1, pos: 30
type: DSZ, layer: 1, pos: 66
type: DSZ, layer: 1, pos: 198
type: DSZ, layer: 1, pos: 103
type: DSZ, layer: 1, pos: 93
type: DSZ, layer: 1, pos: 166
type: DSZ, layer: 1, pos: 221
type: DSZ, layer: 1, pos: 62
type: DSZ, layer: 1, pos: 188
type: DSZ, layer: 1, pos: 182
type: DSZ, layer: 1, pos: 13
type: DSZ, layer: 1, pos: 233
type: DSZ, layer: 1, pos: 83
type: DSZ, layer: 1, pos: 1
type: DSZ, layer: 1, pos: 161
type: DSZ, layer: 1, pos: 219
type: DSZ, layer: 1, pos: 68
type: DSZ, layer: 1, pos: 245
type: DSZ, layer: 1, pos: 235
type: DSZ, layer: 1, pos: 175
type: DSZ, layer: 1, pos: 172

Time for candidate selection: 0.00 seconds

### Candidate
type: DSZ, layer: 1, pos: 183

### Relational analysis ABCD of DS_DSZ2_DSZ2_DSZ2_DSZ2_DSZ2_DSZ1

#### Relational analysis ABCD result of DS_DSZ2_DSZ2_DSZ2_DSZ2_DSZ2_DSZ1
Status: Status.VERIFIED
Output dim: 3, lower bound: -0.2645701, upper bound: 0.2645701
time: 0.87 seconds

### Relational analysis ABCD of DS_DSZ2_DSZ2_DSZ2_DSZ2_DSZ2_DSZ2

#### Relational analysis ABCD result of DS_DSZ2_DSZ2_DSZ2_DSZ2_DSZ2_DSZ2
Status: Status.VERIFIED
Output dim: 3, lower bound: -0.2645701, upper bound: 0.2645701
time: 0.86 seconds

## Summary of splitting (split count: 5)
- Time for DS candidates: 2.54 seconds
DS_DSZ2_DSZ2_DSZ1_DSZ2_DSZ1_DSZ1, status: Status.VERIFIED, split count: 6, time: 2.54
Output dim: 3, lower bound: -0.3059422, upper bound: 0.3059422
DS_DSZ2_DSZ2_DSZ1_DSZ2_DSZ1_DSZ2, status: Status.VERIFIED, split count: 6, time: 2.54
Output dim: 3, lower bound: -0.3059422, upper bound: 0.3059422
DS_DSZ2_DSZ2_DSZ1_DSZ2_DSZ2_DSZ1, status: Status.UNKNOWN, split count: 6, time: 2.54
Output dim: 3, lower bound: -0.3247424, upper bound: 0.3247424
DS_DSZ2_DSZ2_DSZ1_DSZ2_DSZ2_DSZ2, status: Status.UNKNOWN, split count: 6, time: 2.54
Output dim: 3, lower bound: -0.3247424, upper bound: 0.3247424
DS_DSZ2_DSZ2_DSZ2_DSZ1_DSZ1_DSZ1, status: Status.UNKNOWN, split count: 6, time: 2.54
Output dim: 3, lower bound: -0.3264099, upper bound: 0.3264099
DS_DSZ2_DSZ2_DSZ2_DSZ1_DSZ1_DSZ2, status: Status.UNKNOWN, split count: 6, time: 2.54
Output dim: 3, lower bound: -0.3264099, upper bound: 0.3264099
DS_DSZ2_DSZ2_DSZ2_DSZ1_DSZ2_DSZ1, status: Status.VERIFIED, split count: 6, time: 2.54
Output dim: 3, lower bound: -0.2857863, upper bound: 0.2857863
DS_DSZ2_DSZ2_DSZ2_DSZ1_DSZ2_DSZ2, status: Status.VERIFIED, split count: 6, time: 2.54
Output dim: 3, lower bound: -0.2857863, upper bound: 0.2857863
DS_DSZ2_DSZ2_DSZ2_DSZ2_DSZ1_DSZ1, status: Status.VERIFIED, split count: 6, time: 2.54
Output dim: 3, lower bound: -0.3009578, upper bound: 0.3009578
DS_DSZ2_DSZ2_DSZ2_DSZ2_DSZ1_DSZ2, status: Status.VERIFIED, split count: 6, time: 2.54
Output dim: 3, lower bound: -0.3009578, upper bound: 0.3009578
DS_DSZ2_DSZ2_DSZ2_DSZ2_DSZ2_DSZ1, status: Status.VERIFIED, split count: 6, time: 2.54
Output dim: 3, lower bound: -0.2645701, upper bound: 0.2645701
DS_DSZ2_DSZ2_DSZ2_DSZ2_DSZ2_DSZ2, status: Status.VERIFIED, split count: 6, time: 2.54
Output dim: 3, lower bound: -0.2645701, upper bound: 0.2645701

## BFS DS instance: DS_DSZ2_DSZ2_DSZ1_DSZ2_DSZ2_DSZ1

### Backsubstitution after applying DS history:
0: -0.5426022, 0.4665188, -0.5426022, 0.4665188, -1.0091211, 1.0091211
1: -0.3630026, 0.3911184, -0.3630026, 0.3911184, -0.7541210, 0.7541210
2: -0.5269889, 0.4806983, -0.5269889, 0.4806983, -1.0076871, 1.0076871
3: 0.7543352, 1.1590887, 0.7543352, 1.1590887, -0.4047535, 0.4047535
4: -0.4614211, 0.4625955, -0.4614211, 0.4625955, -0.9240167, 0.9240167
5: -0.4214845, 0.4672619, -0.4214845, 0.4672619, -0.8887464, 0.8887464
6: -0.5138452, 0.5371552, -0.5138452, 0.5371552, -1.0510004, 1.0510004
7: -0.4409947, 0.4028524, -0.4409947, 0.4028524, -0.8438470, 0.8438470
8: -0.4306176, 0.5533380, -0.4306176, 0.5533380, -0.9839556, 0.9839556
9: -0.4872051, 0.3854194, -0.4872051, 0.3854194, -0.8726246, 0.8726246

Time for backsubstitution: 0.80 seconds

### DS candidates at layer 1
type: DSZ, layer: 1, pos: 182
type: DSZ, layer: 1, pos: 13
type: DSZ, layer: 1, pos: 183
type: DSZ, layer: 1, pos: 245
type: DSZ, layer: 1, pos: 83
type: DSZ, layer: 1, pos: 66
type: DSZ, layer: 1, pos: 120
type: DSZ, layer: 1, pos: 166
type: DSZ, layer: 1, pos: 177
type: DSZ, layer: 1, pos: 37
type: DSZ, layer: 1, pos: 136
type: DSZ, layer: 1, pos: 68
type: DSZ, layer: 1, pos: 29
type: DSZ, layer: 1, pos: 30
type: DSZ, layer: 1, pos: 235
type: DSZ, layer: 1, pos: 1
type: DSZ, layer: 1, pos: 233
type: DSZ, layer: 1, pos: 161
type: DSZ, layer: 1, pos: 172
type: DSZ, layer: 1, pos: 178
type: DSZ, layer: 1, pos: 240
type: DSZ, layer: 1, pos: 0
type: DSZ, layer: 1, pos: 220
type: DSZ, layer: 1, pos: 103
type: DSZ, layer: 1, pos: 71
type: DSZ, layer: 1, pos: 219
type: DSZ, layer: 1, pos: 62
type: DSZ, layer: 1, pos: 198
type: DSZ, layer: 1, pos: 176
type: DSZ, layer: 1, pos: 251
type: DSZ, layer: 1, pos: 221

Time for candidate selection: 0.00 seconds

### Candidate
type: DSZ, layer: 1, pos: 182

### Relational analysis ABCD of DS_DSZ2_DSZ2_DSZ1_DSZ2_DSZ2_DSZ1_DSZ1
Optimization infeasible because this subproblem isn't reachable.

### Candidate
type: DSZ, layer: 1, pos: 13

### Relational analysis ABCD of DS_DSZ2_DSZ2_DSZ1_DSZ2_DSZ2_DSZ1_DSZ1

#### Relational analysis ABCD result of DS_DSZ2_DSZ2_DSZ1_DSZ2_DSZ2_DSZ1_DSZ1
Status: Status.UNKNOWN
Output dim: 3, lower bound: -0.3235577, upper bound: 0.3235577
time: 1.12 seconds

### Relational analysis ABCD of DS_DSZ2_DSZ2_DSZ1_DSZ2_DSZ2_DSZ1_DSZ2

#### Relational analysis ABCD result of DS_DSZ2_DSZ2_DSZ1_DSZ2_DSZ2_DSZ1_DSZ2
Status: Status.UNKNOWN
Output dim: 3, lower bound: -0.3235577, upper bound: 0.3235577
time: 1.06 seconds

## BFS DS instance: DS_DSZ2_DSZ2_DSZ1_DSZ2_DSZ2_DSZ2

### Backsubstitution after applying DS history:
0: -0.5426022, 0.4665188, -0.5426022, 0.4665188, -1.0091211, 1.0091211
1: -0.3630026, 0.3911184, -0.3630026, 0.3911184, -0.7541210, 0.7541210
2: -0.5269889, 0.4806983, -0.5269889, 0.4806983, -1.0076871, 1.0076871
3: 0.7543352, 1.1590887, 0.7543352, 1.1590887, -0.4047535, 0.4047535
4: -0.4614211, 0.4625955, -0.4614211, 0.4625955, -0.9240167, 0.9240167
5: -0.4214845, 0.4672619, -0.4214845, 0.4672619, -0.8887464, 0.8887464
6: -0.5138452, 0.5371552, -0.5138452, 0.5371552, -1.0510004, 1.0510004
7: -0.4409947, 0.4028524, -0.4409947, 0.4028524, -0.8438470, 0.8438470
8: -0.4306176, 0.5533380, -0.4306176, 0.5533380, -0.9839556, 0.9839556
9: -0.4872051, 0.3854194, -0.4872051, 0.3854194, -0.8726246, 0.8726246

Time for backsubstitution: 0.76 seconds

### DS candidates at layer 1
type: DSZ, layer: 1, pos: 176
type: DSZ, layer: 1, pos: 71
type: DSZ, layer: 1, pos: 178
type: DSZ, layer: 1, pos: 172
type: DSZ, layer: 1, pos: 136
type: DSZ, layer: 1, pos: 219
type: DSZ, layer: 1, pos: 29
type: DSZ, layer: 1, pos: 183
type: DSZ, layer: 1, pos: 103
type: DSZ, layer: 1, pos: 1
type: DSZ, layer: 1, pos: 182
type: DSZ, layer: 1, pos: 62
type: DSZ, layer: 1, pos: 68
type: DSZ, layer: 1, pos: 221
type: DSZ, layer: 1, pos: 0
type: DSZ, layer: 1, pos: 177
type: DSZ, layer: 1, pos: 198
type: DSZ, layer: 1, pos: 240
type: DSZ, layer: 1, pos: 120
type: DSZ, layer: 1, pos: 166
type: DSZ, layer: 1, pos: 66
type: DSZ, layer: 1, pos: 233
type: DSZ, layer: 1, pos: 220
type: DSZ, layer: 1, pos: 245
type: DSZ, layer: 1, pos: 161
type: DSZ, layer: 1, pos: 83
type: DSZ, layer: 1, pos: 235
type: DSZ, layer: 1, pos: 37
type: DSZ, layer: 1, pos: 13
type: DSZ, layer: 1, pos: 30
type: DSZ, layer: 1, pos: 251

Time for candidate selection: 0.00 seconds

### Candidate
type: DSZ, layer: 1, pos: 176

### Relational analysis ABCD of DS_DSZ2_DSZ2_DSZ1_DSZ2_DSZ2_DSZ2_DSZ1

#### Relational analysis ABCD result of DS_DSZ2_DSZ2_DSZ1_DSZ2_DSZ2_DSZ2_DSZ1
Status: Status.VERIFIED
Output dim: 3, lower bound: -0.3040789, upper bound: 0.3040789
time: 1.04 seconds

### Relational analysis ABCD of DS_DSZ2_DSZ2_DSZ1_DSZ2_DSZ2_DSZ2_DSZ2

#### Relational analysis ABCD result of DS_DSZ2_DSZ2_DSZ1_DSZ2_DSZ2_DSZ2_DSZ2
Status: Status.VERIFIED
Output dim: 3, lower bound: -0.3040789, upper bound: 0.3040789
time: 1.03 seconds

## BFS DS instance: DS_DSZ2_DSZ2_DSZ2_DSZ1_DSZ1_DSZ1

### Backsubstitution after applying DS history:
0: -0.5426022, 0.4665188, -0.5426022, 0.4665188, -1.0091211, 1.0091211
1: -0.3630026, 0.3911184, -0.3630026, 0.3911184, -0.7541210, 0.7541210
2: -0.5269889, 0.4806983, -0.5269889, 0.4806983, -1.0076871, 1.0076871
3: 0.7543352, 1.1590887, 0.7543352, 1.1590887, -0.4047535, 0.4047535
4: -0.4614211, 0.4625955, -0.4614211, 0.4625955, -0.9240167, 0.9240167
5: -0.4214845, 0.4672619, -0.4214845, 0.4672619, -0.8887464, 0.8887464
6: -0.5138452, 0.5371552, -0.5138452, 0.5371552, -1.0510004, 1.0510004
7: -0.4409947, 0.4028524, -0.4409947, 0.4028524, -0.8438470, 0.8438470
8: -0.4306176, 0.5533380, -0.4306176, 0.5533380, -0.9839556, 0.9839556
9: -0.4872051, 0.3854194, -0.4872051, 0.3854194, -0.8726246, 0.8726246

Time for backsubstitution: 0.75 seconds

### DS candidates at layer 1
type: DSZ, layer: 1, pos: 30
type: DSZ, layer: 1, pos: 178
type: DSZ, layer: 1, pos: 233
type: DSZ, layer: 1, pos: 71
type: DSZ, layer: 1, pos: 198
type: DSZ, layer: 1, pos: 1
type: DSZ, layer: 1, pos: 120
type: DSZ, layer: 1, pos: 166
type: DSZ, layer: 1, pos: 235
type: DSZ, layer: 1, pos: 0
type: DSZ, layer: 1, pos: 175
type: DSZ, layer: 1, pos: 37
type: DSZ, layer: 1, pos: 62
type: DSZ, layer: 1, pos: 29
type: DSZ, layer: 1, pos: 68
type: DSZ, layer: 1, pos: 136
type: DSZ, layer: 1, pos: 83
type: DSZ, layer: 1, pos: 220
type: DSZ, layer: 1, pos: 13
type: DSZ, layer: 1, pos: 221
type: DSZ, layer: 1, pos: 240
type: DSZ, layer: 1, pos: 251
type: DSZ, layer: 1, pos: 66
type: DSZ, layer: 1, pos: 182
type: DSZ, layer: 1, pos: 172
type: DSZ, layer: 1, pos: 103
type: DSZ, layer: 1, pos: 219
type: DSZ, layer: 1, pos: 183
type: DSZ, layer: 1, pos: 176
type: DSZ, layer: 1, pos: 161
type: DSZ, layer: 1, pos: 245

Time for candidate selection: 0.00 seconds

### Candidate
type: DSZ, layer: 1, pos: 30

### Relational analysis ABCD of DS_DSZ2_DSZ2_DSZ2_DSZ1_DSZ1_DSZ1_DSZ1

#### Relational analysis ABCD result of DS_DSZ2_DSZ2_DSZ2_DSZ1_DSZ1_DSZ1_DSZ1
Status: Status.UNKNOWN
Output dim: 3, lower bound: -0.3242944, upper bound: 0.3242944
time: 1.13 seconds

### Relational analysis ABCD of DS_DSZ2_DSZ2_DSZ2_DSZ1_DSZ1_DSZ1_DSZ2

#### Relational analysis ABCD result of DS_DSZ2_DSZ2_DSZ2_DSZ1_DSZ1_DSZ1_DSZ2
Status: Status.UNKNOWN
Output dim: 3, lower bound: -0.3242944, upper bound: 0.3242944
time: 1.11 seconds

## BFS DS instance: DS_DSZ2_DSZ2_DSZ2_DSZ1_DSZ1_DSZ2

### Backsubstitution after applying DS history:
0: -0.5426022, 0.4665188, -0.5426022, 0.4665188, -1.0091211, 1.0091211
1: -0.3630026, 0.3911184, -0.3630026, 0.3911184, -0.7541210, 0.7541210
2: -0.5269889, 0.4806983, -0.5269889, 0.4806983, -1.0076871, 1.0076871
3: 0.7543352, 1.1590887, 0.7543352, 1.1590887, -0.4047535, 0.4047535
4: -0.4614211, 0.4625955, -0.4614211, 0.4625955, -0.9240167, 0.9240167
5: -0.4214845, 0.4672619, -0.4214845, 0.4672619, -0.8887464, 0.8887464
6: -0.5138452, 0.5371552, -0.5138452, 0.5371552, -1.0510004, 1.0510004
7: -0.4409947, 0.4028524, -0.4409947, 0.4028524, -0.8438470, 0.8438470
8: -0.4306176, 0.5533380, -0.4306176, 0.5533380, -0.9839556, 0.9839556
9: -0.4872051, 0.3854194, -0.4872051, 0.3854194, -0.8726246, 0.8726246

Time for backsubstitution: 0.75 seconds

### DS candidates at layer 1
type: DSZ, layer: 1, pos: 178
type: DSZ, layer: 1, pos: 233
type: DSZ, layer: 1, pos: 13
type: DSZ, layer: 1, pos: 136
type: DSZ, layer: 1, pos: 37
type: DSZ, layer: 1, pos: 245
type: DSZ, layer: 1, pos: 175
type: DSZ, layer: 1, pos: 30
type: DSZ, layer: 1, pos: 71
type: DSZ, layer: 1, pos: 183
type: DSZ, layer: 1, pos: 0
type: DSZ, layer: 1, pos: 68
type: DSZ, layer: 1, pos: 172
type: DSZ, layer: 1, pos: 198
type: DSZ, layer: 1, pos: 120
type: DSZ, layer: 1, pos: 176
type: DSZ, layer: 1, pos: 161
type: DSZ, layer: 1, pos: 182
type: DSZ, layer: 1, pos: 166
type: DSZ, layer: 1, pos: 29
type: DSZ, layer: 1, pos: 251
type: DSZ, layer: 1, pos: 66
type: DSZ, layer: 1, pos: 240
type: DSZ, layer: 1, pos: 219
type: DSZ, layer: 1, pos: 103
type: DSZ, layer: 1, pos: 221
type: DSZ, layer: 1, pos: 62
type: DSZ, layer: 1, pos: 83
type: DSZ, layer: 1, pos: 220
type: DSZ, layer: 1, pos: 1
type: DSZ, layer: 1, pos: 235

Time for candidate selection: 0.00 seconds

### Candidate
type: DSZ, layer: 1, pos: 178

### Relational analysis ABCD of DS_DSZ2_DSZ2_DSZ2_DSZ1_DSZ1_DSZ2_DSZ1

#### Relational analysis ABCD result of DS_DSZ2_DSZ2_DSZ2_DSZ1_DSZ1_DSZ2_DSZ1
Status: Status.UNKNOWN
Output dim: 3, lower bound: -0.3264099, upper bound: 0.3264099
time: 1.35 seconds

### Relational analysis ABCD of DS_DSZ2_DSZ2_DSZ2_DSZ1_DSZ1_DSZ2_DSZ2

#### Relational analysis ABCD result of DS_DSZ2_DSZ2_DSZ2_DSZ1_DSZ1_DSZ2_DSZ2
Status: Status.UNKNOWN
Output dim: 3, lower bound: -0.3264099, upper bound: 0.3264099
time: 1.18 seconds

## Summary of splitting (split count: 6)
- Time for DS candidates: 3.29 seconds
DS_DSZ2_DSZ2_DSZ1_DSZ2_DSZ2_DSZ1_DSZ1, status: Status.UNKNOWN, split count: 7, time: 3.29
Output dim: 3, lower bound: -0.3235577, upper bound: 0.3235577
DS_DSZ2_DSZ2_DSZ1_DSZ2_DSZ2_DSZ1_DSZ2, status: Status.UNKNOWN, split count: 7, time: 3.29
Output dim: 3, lower bound: -0.3235577, upper bound: 0.3235577
DS_DSZ2_DSZ2_DSZ1_DSZ2_DSZ2_DSZ2_DSZ1, status: Status.VERIFIED, split count: 7, time: 3.29
Output dim: 3, lower bound: -0.3040789, upper bound: 0.3040789
DS_DSZ2_DSZ2_DSZ1_DSZ2_DSZ2_DSZ2_DSZ2, status: Status.VERIFIED, split count: 7, time: 3.29
Output dim: 3, lower bound: -0.3040789, upper bound: 0.3040789
DS_DSZ2_DSZ2_DSZ2_DSZ1_DSZ1_DSZ1_DSZ1, status: Status.UNKNOWN, split count: 7, time: 3.29
Output dim: 3, lower bound: -0.3242944, upper bound: 0.3242944
DS_DSZ2_DSZ2_DSZ2_DSZ1_DSZ1_DSZ1_DSZ2, status: Status.UNKNOWN, split count: 7, time: 3.29
Output dim: 3, lower bound: -0.3242944, upper bound: 0.3242944
DS_DSZ2_DSZ2_DSZ2_DSZ1_DSZ1_DSZ2_DSZ1, status: Status.UNKNOWN, split count: 7, time: 3.29
Output dim: 3, lower bound: -0.3264099, upper bound: 0.3264099
DS_DSZ2_DSZ2_DSZ2_DSZ1_DSZ1_DSZ2_DSZ2, status: Status.UNKNOWN, split count: 7, time: 3.29
Output dim: 3, lower bound: -0.3264099, upper bound: 0.3264099

## BFS DS instance: DS_DSZ2_DSZ2_DSZ1_DSZ2_DSZ2_DSZ1_DSZ1

### Backsubstitution after applying DS history:
0: -0.5426022, 0.4665188, -0.5426022, 0.4665188, -1.0091211, 1.0091211
1: -0.3630026, 0.3911184, -0.3630026, 0.3911184, -0.7541210, 0.7541210
2: -0.5269889, 0.4806983, -0.5269889, 0.4806983, -1.0076871, 1.0076871
3: 0.7543352, 1.1590887, 0.7543352, 1.1590887, -0.4047535, 0.4047535
4: -0.4614211, 0.4625955, -0.4614211, 0.4625955, -0.9240167, 0.9240167
5: -0.4214845, 0.4672619, -0.4214845, 0.4672619, -0.8887464, 0.8887464
6: -0.5138452, 0.5371552, -0.5138452, 0.5371552, -1.0510004, 1.0510004
7: -0.4409947, 0.4028524, -0.4409947, 0.4028524, -0.8438470, 0.8438470
8: -0.4306176, 0.5533380, -0.4306176, 0.5533380, -0.9839556, 0.9839556
9: -0.4872051, 0.3854194, -0.4872051, 0.3854194, -0.8726246, 0.8726246

Time for backsubstitution: 0.74 seconds

### DS candidates at layer 1
type: DSZ, layer: 1, pos: 120
type: DSZ, layer: 1, pos: 176
type: DSZ, layer: 1, pos: 172
type: DSZ, layer: 1, pos: 221
type: DSZ, layer: 1, pos: 0
type: DSZ, layer: 1, pos: 68
type: DSZ, layer: 1, pos: 1
type: DSZ, layer: 1, pos: 37
type: DSZ, layer: 1, pos: 220
type: DSZ, layer: 1, pos: 219
type: DSZ, layer: 1, pos: 178
type: DSZ, layer: 1, pos: 166
type: DSZ, layer: 1, pos: 83
type: DSZ, layer: 1, pos: 182
type: DSZ, layer: 1, pos: 161
type: DSZ, layer: 1, pos: 177
type: DSZ, layer: 1, pos: 30
type: DSZ, layer: 1, pos: 62
type: DSZ, layer: 1, pos: 66
type: DSZ, layer: 1, pos: 240
type: DSZ, layer: 1, pos: 103
type: DSZ, layer: 1, pos: 29
type: DSZ, layer: 1, pos: 233
type: DSZ, layer: 1, pos: 251
type: DSZ, layer: 1, pos: 245
type: DSZ, layer: 1, pos: 183
type: DSZ, layer: 1, pos: 136
type: DSZ, layer: 1, pos: 198
type: DSZ, layer: 1, pos: 235
type: DSZ, layer: 1, pos: 71

Time for candidate selection: 0.00 seconds

### Candidate
type: DSZ, layer: 1, pos: 120

### Relational analysis ABCD of DS_DSZ2_DSZ2_DSZ1_DSZ2_DSZ2_DSZ1_DSZ1_DSZ1

#### Relational analysis ABCD result of DS_DSZ2_DSZ2_DSZ1_DSZ2_DSZ2_DSZ1_DSZ1_DSZ1
Status: Status.VERIFIED
Output dim: 3, lower bound: -0.2849105, upper bound: 0.2849105
time: 1.06 seconds

### Relational analysis ABCD of DS_DSZ2_DSZ2_DSZ1_DSZ2_DSZ2_DSZ1_DSZ1_DSZ2

#### Relational analysis ABCD result of DS_DSZ2_DSZ2_DSZ1_DSZ2_DSZ2_DSZ1_DSZ1_DSZ2
Status: Status.VERIFIED
Output dim: 3, lower bound: -0.2849105, upper bound: 0.2849105
time: 1.06 seconds

## BFS DS instance: DS_DSZ2_DSZ2_DSZ1_DSZ2_DSZ2_DSZ1_DSZ2

### Backsubstitution after applying DS history:
0: -0.5426022, 0.4665188, -0.5426022, 0.4665188, -1.0091211, 1.0091211
1: -0.3630026, 0.3911184, -0.3630026, 0.3911184, -0.7541210, 0.7541210
2: -0.5269889, 0.4806983, -0.5269889, 0.4806983, -1.0076871, 1.0076871
3: 0.7543352, 1.1590887, 0.7543352, 1.1590887, -0.4047535, 0.4047535
4: -0.4614211, 0.4625955, -0.4614211, 0.4625955, -0.9240167, 0.9240167
5: -0.4214845, 0.4672619, -0.4214845, 0.4672619, -0.8887464, 0.8887464
6: -0.5138452, 0.5371552, -0.5138452, 0.5371552, -1.0510004, 1.0510004
7: -0.4409947, 0.4028524, -0.4409947, 0.4028524, -0.8438470, 0.8438470
8: -0.4306176, 0.5533380, -0.4306176, 0.5533380, -0.9839556, 0.9839556
9: -0.4872051, 0.3854194, -0.4872051, 0.3854194, -0.8726246, 0.8726246

Time for backsubstitution: 0.77 seconds

### DS candidates at layer 1
type: DSZ, layer: 1, pos: 235
type: DSZ, layer: 1, pos: 136
type: DSZ, layer: 1, pos: 221
type: DSZ, layer: 1, pos: 178
type: DSZ, layer: 1, pos: 220
type: DSZ, layer: 1, pos: 161
type: DSZ, layer: 1, pos: 71
type: DSZ, layer: 1, pos: 245
type: DSZ, layer: 1, pos: 182
type: DSZ, layer: 1, pos: 103
type: DSZ, layer: 1, pos: 68
type: DSZ, layer: 1, pos: 166
type: DSZ, layer: 1, pos: 183
type: DSZ, layer: 1, pos: 30
type: DSZ, layer: 1, pos: 240
type: DSZ, layer: 1, pos: 0
type: DSZ, layer: 1, pos: 120
type: DSZ, layer: 1, pos: 176
type: DSZ, layer: 1, pos: 219
type: DSZ, layer: 1, pos: 62
type: DSZ, layer: 1, pos: 37
type: DSZ, layer: 1, pos: 198
type: DSZ, layer: 1, pos: 83
type: DSZ, layer: 1, pos: 66
type: DSZ, layer: 1, pos: 172
type: DSZ, layer: 1, pos: 233
type: DSZ, layer: 1, pos: 251
type: DSZ, layer: 1, pos: 177
type: DSZ, layer: 1, pos: 29
type: DSZ, layer: 1, pos: 1

Time for candidate selection: 0.00 seconds

### Candidate
type: DSZ, layer: 1, pos: 235

### Relational analysis ABCD of DS_DSZ2_DSZ2_DSZ1_DSZ2_DSZ2_DSZ1_DSZ2_DSZ1

#### Relational analysis ABCD result of DS_DSZ2_DSZ2_DSZ1_DSZ2_DSZ2_DSZ1_DSZ2_DSZ1
Status: Status.VERIFIED
Output dim: 3, lower bound: -0.3147222, upper bound: 0.3147222
time: 0.99 seconds

### Relational analysis ABCD of DS_DSZ2_DSZ2_DSZ1_DSZ2_DSZ2_DSZ1_DSZ2_DSZ2

#### Relational analysis ABCD result of DS_DSZ2_DSZ2_DSZ1_DSZ2_DSZ2_DSZ1_DSZ2_DSZ2
Status: Status.VERIFIED
Output dim: 3, lower bound: -0.3147222, upper bound: 0.3147222
time: 1.02 seconds

## BFS DS instance: DS_DSZ2_DSZ2_DSZ2_DSZ1_DSZ1_DSZ1_DSZ1

### Backsubstitution after applying DS history:
0: -0.5426022, 0.4665188, -0.5426022, 0.4665188, -1.0091211, 1.0091211
1: -0.3630026, 0.3911184, -0.3630026, 0.3911184, -0.7541210, 0.7541210
2: -0.5269889, 0.4806983, -0.5269889, 0.4806983, -1.0076871, 1.0076871
3: 0.7543352, 1.1590887, 0.7543352, 1.1590887, -0.4047535, 0.4047535
4: -0.4614211, 0.4625955, -0.4614211, 0.4625955, -0.9240167, 0.9240167
5: -0.4214845, 0.4672619, -0.4214845, 0.4672619, -0.8887464, 0.8887464
6: -0.5138452, 0.5371552, -0.5138452, 0.5371552, -1.0510004, 1.0510004
7: -0.4409947, 0.4028524, -0.4409947, 0.4028524, -0.8438470, 0.8438470
8: -0.4306176, 0.5533380, -0.4306176, 0.5533380, -0.9839556, 0.9839556
9: -0.4872051, 0.3854194, -0.4872051, 0.3854194, -0.8726246, 0.8726246

Time for backsubstitution: 0.78 seconds

### DS candidates at layer 1
type: DSZ, layer: 1, pos: 176
type: DSZ, layer: 1, pos: 0
type: DSZ, layer: 1, pos: 183
type: DSZ, layer: 1, pos: 1
type: DSZ, layer: 1, pos: 68
type: DSZ, layer: 1, pos: 13
type: DSZ, layer: 1, pos: 161
type: DSZ, layer: 1, pos: 83
type: DSZ, layer: 1, pos: 120
type: DSZ, layer: 1, pos: 233
type: DSZ, layer: 1, pos: 29
type: DSZ, layer: 1, pos: 221
type: DSZ, layer: 1, pos: 245
type: DSZ, layer: 1, pos: 175
type: DSZ, layer: 1, pos: 172
type: DSZ, layer: 1, pos: 251
type: DSZ, layer: 1, pos: 198
type: DSZ, layer: 1, pos: 71
type: DSZ, layer: 1, pos: 235
type: DSZ, layer: 1, pos: 219
type: DSZ, layer: 1, pos: 62
type: DSZ, layer: 1, pos: 220
type: DSZ, layer: 1, pos: 66
type: DSZ, layer: 1, pos: 166
type: DSZ, layer: 1, pos: 103
type: DSZ, layer: 1, pos: 136
type: DSZ, layer: 1, pos: 182
type: DSZ, layer: 1, pos: 240
type: DSZ, layer: 1, pos: 178
type: DSZ, layer: 1, pos: 37

Time for candidate selection: 0.00 seconds

### Candidate
type: DSZ, layer: 1, pos: 176

### Relational analysis ABCD of DS_DSZ2_DSZ2_DSZ2_DSZ1_DSZ1_DSZ1_DSZ1_DSZ1

#### Relational analysis ABCD result of DS_DSZ2_DSZ2_DSZ2_DSZ1_DSZ1_DSZ1_DSZ1_DSZ1
Status: Status.VERIFIED
Output dim: 3, lower bound: -0.3037453, upper bound: 0.3037453
time: 1.09 seconds

### Relational analysis ABCD of DS_DSZ2_DSZ2_DSZ2_DSZ1_DSZ1_DSZ1_DSZ1_DSZ2

#### Relational analysis ABCD result of DS_DSZ2_DSZ2_DSZ2_DSZ1_DSZ1_DSZ1_DSZ1_DSZ2
Status: Status.VERIFIED
Output dim: 3, lower bound: -0.3037453, upper bound: 0.3037453
time: 1.07 seconds

## BFS DS instance: DS_DSZ2_DSZ2_DSZ2_DSZ1_DSZ1_DSZ1_DSZ2

### Backsubstitution after applying DS history:
0: -0.5426022, 0.4665188, -0.5426022, 0.4665188, -1.0091211, 1.0091211
1: -0.3630026, 0.3911184, -0.3630026, 0.3911184, -0.7541210, 0.7541210
2: -0.5269889, 0.4806983, -0.5269889, 0.4806983, -1.0076871, 1.0076871
3: 0.7543352, 1.1590887, 0.7543352, 1.1590887, -0.4047535, 0.4047535
4: -0.4614211, 0.4625955, -0.4614211, 0.4625955, -0.9240167, 0.9240167
5: -0.4214845, 0.4672619, -0.4214845, 0.4672619, -0.8887464, 0.8887464
6: -0.5138452, 0.5371552, -0.5138452, 0.5371552, -1.0510004, 1.0510004
7: -0.4409947, 0.4028524, -0.4409947, 0.4028524, -0.8438470, 0.8438470
8: -0.4306176, 0.5533380, -0.4306176, 0.5533380, -0.9839556, 0.9839556
9: -0.4872051, 0.3854194, -0.4872051, 0.3854194, -0.8726246, 0.8726246

Time for backsubstitution: 0.78 seconds

### DS candidates at layer 1
type: DSZ, layer: 1, pos: 62
type: DSZ, layer: 1, pos: 136
type: DSZ, layer: 1, pos: 83
type: DSZ, layer: 1, pos: 178
type: DSZ, layer: 1, pos: 0
type: DSZ, layer: 1, pos: 103
type: DSZ, layer: 1, pos: 219
type: DSZ, layer: 1, pos: 245
type: DSZ, layer: 1, pos: 166
type: DSZ, layer: 1, pos: 13
type: DSZ, layer: 1, pos: 221
type: DSZ, layer: 1, pos: 66
type: DSZ, layer: 1, pos: 198
type: DSZ, layer: 1, pos: 251
type: DSZ, layer: 1, pos: 176
type: DSZ, layer: 1, pos: 183
type: DSZ, layer: 1, pos: 182
type: DSZ, layer: 1, pos: 120
type: DSZ, layer: 1, pos: 68
type: DSZ, layer: 1, pos: 240
type: DSZ, layer: 1, pos: 175
type: DSZ, layer: 1, pos: 172
type: DSZ, layer: 1, pos: 235
type: DSZ, layer: 1, pos: 37
type: DSZ, layer: 1, pos: 1
type: DSZ, layer: 1, pos: 71
type: DSZ, layer: 1, pos: 220
type: DSZ, layer: 1, pos: 29
type: DSZ, layer: 1, pos: 233
type: DSZ, layer: 1, pos: 161

Time for candidate selection: 0.00 seconds

### Candidate
type: DSZ, layer: 1, pos: 62

### Relational analysis ABCD of DS_DSZ2_DSZ2_DSZ2_DSZ1_DSZ1_DSZ1_DSZ2_DSZ1

#### Relational analysis ABCD result of DS_DSZ2_DSZ2_DSZ2_DSZ1_DSZ1_DSZ1_DSZ2_DSZ1
Status: Status.UNKNOWN
Output dim: 3, lower bound: -0.3242944, upper bound: 0.3242944
time: 1.21 seconds

### Relational analysis ABCD of DS_DSZ2_DSZ2_DSZ2_DSZ1_DSZ1_DSZ1_DSZ2_DSZ2

#### Relational analysis ABCD result of DS_DSZ2_DSZ2_DSZ2_DSZ1_DSZ1_DSZ1_DSZ2_DSZ2
Status: Status.UNKNOWN
Output dim: 3, lower bound: -0.3242944, upper bound: 0.3242944
time: 1.25 seconds

## BFS DS instance: DS_DSZ2_DSZ2_DSZ2_DSZ1_DSZ1_DSZ2_DSZ1

### Backsubstitution after applying DS history:
0: -0.5426022, 0.4665188, -0.5426022, 0.4665188, -1.0091211, 1.0091211
1: -0.3630026, 0.3911184, -0.3630026, 0.3911184, -0.7541210, 0.7541210
2: -0.5269889, 0.4806983, -0.5269889, 0.4806983, -1.0076871, 1.0076871
3: 0.7543352, 1.1590887, 0.7543352, 1.1590887, -0.4047535, 0.4047535
4: -0.4614211, 0.4625955, -0.4614211, 0.4625955, -0.9240167, 0.9240167
5: -0.4214845, 0.4672619, -0.4214845, 0.4672619, -0.8887464, 0.8887464
6: -0.5138452, 0.5371552, -0.5138452, 0.5371552, -1.0510004, 1.0510004
7: -0.4409947, 0.4028524, -0.4409947, 0.4028524, -0.8438470, 0.8438470
8: -0.4306176, 0.5533380, -0.4306176, 0.5533380, -0.9839556, 0.9839556
9: -0.4872051, 0.3854194, -0.4872051, 0.3854194, -0.8726246, 0.8726246

Time for backsubstitution: 0.80 seconds

### DS candidates at layer 1
type: DSZ, layer: 1, pos: 233
type: DSZ, layer: 1, pos: 37
type: DSZ, layer: 1, pos: 166
type: DSZ, layer: 1, pos: 240
type: DSZ, layer: 1, pos: 83
type: DSZ, layer: 1, pos: 13
type: DSZ, layer: 1, pos: 1
type: DSZ, layer: 1, pos: 30
type: DSZ, layer: 1, pos: 176
type: DSZ, layer: 1, pos: 182
type: DSZ, layer: 1, pos: 235
type: DSZ, layer: 1, pos: 103
type: DSZ, layer: 1, pos: 120
type: DSZ, layer: 1, pos: 29
type: DSZ, layer: 1, pos: 245
type: DSZ, layer: 1, pos: 136
type: DSZ, layer: 1, pos: 161
type: DSZ, layer: 1, pos: 183
type: DSZ, layer: 1, pos: 219
type: DSZ, layer: 1, pos: 175
type: DSZ, layer: 1, pos: 66
type: DSZ, layer: 1, pos: 0
type: DSZ, layer: 1, pos: 251
type: DSZ, layer: 1, pos: 62
type: DSZ, layer: 1, pos: 221
type: DSZ, layer: 1, pos: 71
type: DSZ, layer: 1, pos: 220
type: DSZ, layer: 1, pos: 172
type: DSZ, layer: 1, pos: 68
type: DSZ, layer: 1, pos: 198

Time for candidate selection: 0.00 seconds

### Candidate
type: DSZ, layer: 1, pos: 233

### Relational analysis ABCD of DS_DSZ2_DSZ2_DSZ2_DSZ1_DSZ1_DSZ2_DSZ1_DSZ1

#### Relational analysis ABCD result of DS_DSZ2_DSZ2_DSZ2_DSZ1_DSZ1_DSZ2_DSZ1_DSZ1
Status: Status.UNKNOWN
Output dim: 3, lower bound: -0.3250834, upper bound: 0.3250834
time: 1.17 seconds

### Relational analysis ABCD of DS_DSZ2_DSZ2_DSZ2_DSZ1_DSZ1_DSZ2_DSZ1_DSZ2

#### Relational analysis ABCD result of DS_DSZ2_DSZ2_DSZ2_DSZ1_DSZ1_DSZ2_DSZ1_DSZ2
Status: Status.UNKNOWN
Output dim: 3, lower bound: -0.3250834, upper bound: 0.3250834
time: 1.16 seconds

## BFS DS instance: DS_DSZ2_DSZ2_DSZ2_DSZ1_DSZ1_DSZ2_DSZ2

### Backsubstitution after applying DS history:
0: -0.5426022, 0.4665188, -0.5426022, 0.4665188, -1.0091211, 1.0091211
1: -0.3630026, 0.3911184, -0.3630026, 0.3911184, -0.7541210, 0.7541210
2: -0.5269889, 0.4806983, -0.5269889, 0.4806983, -1.0076871, 1.0076871
3: 0.7543352, 1.1590887, 0.7543352, 1.1590887, -0.4047535, 0.4047535
4: -0.4614211, 0.4625955, -0.4614211, 0.4625955, -0.9240167, 0.9240167
5: -0.4214845, 0.4672619, -0.4214845, 0.4672619, -0.8887464, 0.8887464
6: -0.5138452, 0.5371552, -0.5138452, 0.5371552, -1.0510004, 1.0510004
7: -0.4409947, 0.4028524, -0.4409947, 0.4028524, -0.8438470, 0.8438470
8: -0.4306176, 0.5533380, -0.4306176, 0.5533380, -0.9839556, 0.9839556
9: -0.4872051, 0.3854194, -0.4872051, 0.3854194, -0.8726246, 0.8726246

Time for backsubstitution: 0.78 seconds

### DS candidates at layer 1
type: DSZ, layer: 1, pos: 68
type: DSZ, layer: 1, pos: 251
type: DSZ, layer: 1, pos: 245
type: DSZ, layer: 1, pos: 183
type: DSZ, layer: 1, pos: 166
type: DSZ, layer: 1, pos: 233
type: DSZ, layer: 1, pos: 120
type: DSZ, layer: 1, pos: 1
type: DSZ, layer: 1, pos: 0
type: DSZ, layer: 1, pos: 220
type: DSZ, layer: 1, pos: 172
type: DSZ, layer: 1, pos: 37
type: DSZ, layer: 1, pos: 161
type: DSZ, layer: 1, pos: 176
type: DSZ, layer: 1, pos: 136
type: DSZ, layer: 1, pos: 66
type: DSZ, layer: 1, pos: 30
type: DSZ, layer: 1, pos: 29
type: DSZ, layer: 1, pos: 175
type: DSZ, layer: 1, pos: 240
type: DSZ, layer: 1, pos: 235
type: DSZ, layer: 1, pos: 198
type: DSZ, layer: 1, pos: 71
type: DSZ, layer: 1, pos: 62
type: DSZ, layer: 1, pos: 83
type: DSZ, layer: 1, pos: 13
type: DSZ, layer: 1, pos: 182
type: DSZ, layer: 1, pos: 221
type: DSZ, layer: 1, pos: 103
type: DSZ, layer: 1, pos: 219

Time for candidate selection: 0.00 seconds

### Candidate
type: DSZ, layer: 1, pos: 68

### Relational analysis ABCD of DS_DSZ2_DSZ2_DSZ2_DSZ1_DSZ1_DSZ2_DSZ2_DSZ1

#### Relational analysis ABCD result of DS_DSZ2_DSZ2_DSZ2_DSZ1_DSZ1_DSZ2_DSZ2_DSZ1
Status: Status.VERIFIED
Output dim: 3, lower bound: -0.3027268, upper bound: 0.3027268
time: 1.26 seconds

### Relational analysis ABCD of DS_DSZ2_DSZ2_DSZ2_DSZ1_DSZ1_DSZ2_DSZ2_DSZ2

#### Relational analysis ABCD result of DS_DSZ2_DSZ2_DSZ2_DSZ1_DSZ1_DSZ2_DSZ2_DSZ2
Status: Status.VERIFIED
Output dim: 3, lower bound: -0.3027268, upper bound: 0.3027268
time: 1.27 seconds

## Summary of splitting (split count: 7)
- Time for DS candidates: 3.32 seconds
DS_DSZ2_DSZ2_DSZ1_DSZ2_DSZ2_DSZ1_DSZ1_DSZ1, status: Status.VERIFIED, split count: 8, time: 3.32
Output dim: 3, lower bound: -0.2849105, upper bound: 0.2849105
DS_DSZ2_DSZ2_DSZ1_DSZ2_DSZ2_DSZ1_DSZ1_DSZ2, status: Status.VERIFIED, split count: 8, time: 3.32
Output dim: 3, lower bound: -0.2849105, upper bound: 0.2849105
DS_DSZ2_DSZ2_DSZ1_DSZ2_DSZ2_DSZ1_DSZ2_DSZ1, status: Status.VERIFIED, split count: 8, time: 3.32
Output dim: 3, lower bound: -0.3147222, upper bound: 0.3147222
DS_DSZ2_DSZ2_DSZ1_DSZ2_DSZ2_DSZ1_DSZ2_DSZ2, status: Status.VERIFIED, split count: 8, time: 3.32
Output dim: 3, lower bound: -0.3147222, upper bound: 0.3147222
DS_DSZ2_DSZ2_DSZ2_DSZ1_DSZ1_DSZ1_DSZ1_DSZ1, status: Status.VERIFIED, split count: 8, time: 3.32
Output dim: 3, lower bound: -0.3037453, upper bound: 0.3037453
DS_DSZ2_DSZ2_DSZ2_DSZ1_DSZ1_DSZ1_DSZ1_DSZ2, status: Status.VERIFIED, split count: 8, time: 3.32
Output dim: 3, lower bound: -0.3037453, upper bound: 0.3037453
DS_DSZ2_DSZ2_DSZ2_DSZ1_DSZ1_DSZ1_DSZ2_DSZ1, status: Status.UNKNOWN, split count: 8, time: 3.32
Output dim: 3, lower bound: -0.3242944, upper bound: 0.3242944
DS_DSZ2_DSZ2_DSZ2_DSZ1_DSZ1_DSZ1_DSZ2_DSZ2, status: Status.UNKNOWN, split count: 8, time: 3.32
Output dim: 3, lower bound: -0.3242944, upper bound: 0.3242944
DS_DSZ2_DSZ2_DSZ2_DSZ1_DSZ1_DSZ2_DSZ1_DSZ1, status: Status.UNKNOWN, split count: 8, time: 3.32
Output dim: 3, lower bound: -0.3250834, upper bound: 0.3250834
DS_DSZ2_DSZ2_DSZ2_DSZ1_DSZ1_DSZ2_DSZ1_DSZ2, status: Status.UNKNOWN, split count: 8, time: 3.32
Output dim: 3, lower bound: -0.3250834, upper bound: 0.3250834
DS_DSZ2_DSZ2_DSZ2_DSZ1_DSZ1_DSZ2_DSZ2_DSZ1, status: Status.VERIFIED, split count: 8, time: 3.32
Output dim: 3, lower bound: -0.3027268, upper bound: 0.3027268
DS_DSZ2_DSZ2_DSZ2_DSZ1_DSZ1_DSZ2_DSZ2_DSZ2, status: Status.VERIFIED, split count: 8, time: 3.32
Output dim: 3, lower bound: -0.3027268, upper bound: 0.3027268

## BFS DS instance: DS_DSZ2_DSZ2_DSZ2_DSZ1_DSZ1_DSZ1_DSZ2_DSZ1

### Backsubstitution after applying DS history:
0: -0.5426022, 0.4665188, -0.5426022, 0.4665188, -1.0091211, 1.0091211
1: -0.3630026, 0.3911184, -0.3630026, 0.3911184, -0.7541210, 0.7541210
2: -0.5269889, 0.4806983, -0.5269889, 0.4806983, -1.0076871, 1.0076871
3: 0.7543352, 1.1590887, 0.7543352, 1.1590887, -0.4047535, 0.4047535
4: -0.4614211, 0.4625955, -0.4614211, 0.4625955, -0.9240167, 0.9240167
5: -0.4214845, 0.4672619, -0.4214845, 0.4672619, -0.8887464, 0.8887464
6: -0.5138452, 0.5371552, -0.5138452, 0.5371552, -1.0510004, 1.0510004
7: -0.4409947, 0.4028524, -0.4409947, 0.4028524, -0.8438470, 0.8438470
8: -0.4306176, 0.5533380, -0.4306176, 0.5533380, -0.9839556, 0.9839556
9: -0.4872051, 0.3854194, -0.4872051, 0.3854194, -0.8726246, 0.8726246

Time for backsubstitution: 0.74 seconds

### DS candidates at layer 1
type: DSZ, layer: 1, pos: 161
type: DSZ, layer: 1, pos: 1
type: DSZ, layer: 1, pos: 166
type: DSZ, layer: 1, pos: 176
type: DSZ, layer: 1, pos: 220
type: DSZ, layer: 1, pos: 233
type: DSZ, layer: 1, pos: 235
type: DSZ, layer: 1, pos: 240
type: DSZ, layer: 1, pos: 178
type: DSZ, layer: 1, pos: 251
type: DSZ, layer: 1, pos: 221
type: DSZ, layer: 1, pos: 83
type: DSZ, layer: 1, pos: 219
type: DSZ, layer: 1, pos: 29
type: DSZ, layer: 1, pos: 68
type: DSZ, layer: 1, pos: 245
type: DSZ, layer: 1, pos: 103
type: DSZ, layer: 1, pos: 71
type: DSZ, layer: 1, pos: 0
type: DSZ, layer: 1, pos: 198
type: DSZ, layer: 1, pos: 183
type: DSZ, layer: 1, pos: 175
type: DSZ, layer: 1, pos: 13
type: DSZ, layer: 1, pos: 66
type: DSZ, layer: 1, pos: 37
type: DSZ, layer: 1, pos: 172
type: DSZ, layer: 1, pos: 136
type: DSZ, layer: 1, pos: 182
type: DSZ, layer: 1, pos: 120

Time for candidate selection: 0.00 seconds

### Candidate
type: DSZ, layer: 1, pos: 161

### Relational analysis ABCD of DS_DSZ2_DSZ2_DSZ2_DSZ1_DSZ1_DSZ1_DSZ2_DSZ1_DSZ1

#### Relational analysis ABCD result of DS_DSZ2_DSZ2_DSZ2_DSZ1_DSZ1_DSZ1_DSZ2_DSZ1_DSZ1
Status: Status.VERIFIED
Output dim: 3, lower bound: -0.2811331, upper bound: 0.2811331
time: 0.97 seconds

### Relational analysis ABCD of DS_DSZ2_DSZ2_DSZ2_DSZ1_DSZ1_DSZ1_DSZ2_DSZ1_DSZ2

#### Relational analysis ABCD result of DS_DSZ2_DSZ2_DSZ2_DSZ1_DSZ1_DSZ1_DSZ2_DSZ1_DSZ2
Status: Status.VERIFIED
Output dim: 3, lower bound: -0.2811331, upper bound: 0.2811331
time: 0.96 seconds

## BFS DS instance: DS_DSZ2_DSZ2_DSZ2_DSZ1_DSZ1_DSZ1_DSZ2_DSZ2

### Backsubstitution after applying DS history:
0: -0.5426022, 0.4665188, -0.5426022, 0.4665188, -1.0091211, 1.0091211
1: -0.3630026, 0.3911184, -0.3630026, 0.3911184, -0.7541210, 0.7541210
2: -0.5269889, 0.4806983, -0.5269889, 0.4806983, -1.0076871, 1.0076871
3: 0.7543352, 1.1590887, 0.7543352, 1.1590887, -0.4047535, 0.4047535
4: -0.4614211, 0.4625955, -0.4614211, 0.4625955, -0.9240167, 0.9240167
5: -0.4214845, 0.4672619, -0.4214845, 0.4672619, -0.8887464, 0.8887464
6: -0.5138452, 0.5371552, -0.5138452, 0.5371552, -1.0510004, 1.0510004
7: -0.4409947, 0.4028524, -0.4409947, 0.4028524, -0.8438470, 0.8438470
8: -0.4306176, 0.5533380, -0.4306176, 0.5533380, -0.9839556, 0.9839556
9: -0.4872051, 0.3854194, -0.4872051, 0.3854194, -0.8726246, 0.8726246

Time for backsubstitution: 0.74 seconds

### DS candidates at layer 1
type: DSZ, layer: 1, pos: 37
type: DSZ, layer: 1, pos: 166
type: DSZ, layer: 1, pos: 71
type: DSZ, layer: 1, pos: 68
type: DSZ, layer: 1, pos: 245
type: DSZ, layer: 1, pos: 221
type: DSZ, layer: 1, pos: 233
type: DSZ, layer: 1, pos: 175
type: DSZ, layer: 1, pos: 240
type: DSZ, layer: 1, pos: 1
type: DSZ, layer: 1, pos: 103
type: DSZ, layer: 1, pos: 29
type: DSZ, layer: 1, pos: 13
type: DSZ, layer: 1, pos: 251
type: DSZ, layer: 1, pos: 219
type: DSZ, layer: 1, pos: 161
type: DSZ, layer: 1, pos: 172
type: DSZ, layer: 1, pos: 182
type: DSZ, layer: 1, pos: 220
type: DSZ, layer: 1, pos: 183
type: DSZ, layer: 1, pos: 120
type: DSZ, layer: 1, pos: 66
type: DSZ, layer: 1, pos: 83
type: DSZ, layer: 1, pos: 0
type: DSZ, layer: 1, pos: 235
type: DSZ, layer: 1, pos: 176
type: DSZ, layer: 1, pos: 178
type: DSZ, layer: 1, pos: 198
type: DSZ, layer: 1, pos: 136

Time for candidate selection: 0.00 seconds

### Candidate
type: DSZ, layer: 1, pos: 37

### Relational analysis ABCD of DS_DSZ2_DSZ2_DSZ2_DSZ1_DSZ1_DSZ1_DSZ2_DSZ2_DSZ1

#### Relational analysis ABCD result of DS_DSZ2_DSZ2_DSZ2_DSZ1_DSZ1_DSZ1_DSZ2_DSZ2_DSZ1
Status: Status.VERIFIED
Output dim: 3, lower bound: -0.2909268, upper bound: 0.2909268
time: 0.96 seconds

### Relational analysis ABCD of DS_DSZ2_DSZ2_DSZ2_DSZ1_DSZ1_DSZ1_DSZ2_DSZ2_DSZ2

#### Relational analysis ABCD result of DS_DSZ2_DSZ2_DSZ2_DSZ1_DSZ1_DSZ1_DSZ2_DSZ2_DSZ2
Status: Status.VERIFIED
Output dim: 3, lower bound: -0.2909268, upper bound: 0.2909268
time: 0.96 seconds

## BFS DS instance: DS_DSZ2_DSZ2_DSZ2_DSZ1_DSZ1_DSZ2_DSZ1_DSZ1

### Backsubstitution after applying DS history:
0: -0.5426022, 0.4665188, -0.5426022, 0.4665188, -1.0091211, 1.0091211
1: -0.3630026, 0.3911184, -0.3630026, 0.3911184, -0.7541210, 0.7541210
2: -0.5269889, 0.4806983, -0.5269889, 0.4806983, -1.0076871, 1.0076871
3: 0.7543352, 1.1590887, 0.7543352, 1.1590887, -0.4047535, 0.4047535
4: -0.4614211, 0.4625955, -0.4614211, 0.4625955, -0.9240167, 0.9240167
5: -0.4214845, 0.4672619, -0.4214845, 0.4672619, -0.8887464, 0.8887464
6: -0.5138452, 0.5371552, -0.5138452, 0.5371552, -1.0510004, 1.0510004
7: -0.4409947, 0.4028524, -0.4409947, 0.4028524, -0.8438470, 0.8438470
8: -0.4306176, 0.5533380, -0.4306176, 0.5533380, -0.9839556, 0.9839556
9: -0.4872051, 0.3854194, -0.4872051, 0.3854194, -0.8726246, 0.8726246

Time for backsubstitution: 0.81 seconds

### DS candidates at layer 1
type: DSZ, layer: 1, pos: 13
type: DSZ, layer: 1, pos: 235
type: DSZ, layer: 1, pos: 166
type: DSZ, layer: 1, pos: 198
type: DSZ, layer: 1, pos: 62
type: DSZ, layer: 1, pos: 175
type: DSZ, layer: 1, pos: 245
type: DSZ, layer: 1, pos: 71
type: DSZ, layer: 1, pos: 66
type: DSZ, layer: 1, pos: 83
type: DSZ, layer: 1, pos: 219
type: DSZ, layer: 1, pos: 176
type: DSZ, layer: 1, pos: 37
type: DSZ, layer: 1, pos: 172
type: DSZ, layer: 1, pos: 30
type: DSZ, layer: 1, pos: 182
type: DSZ, layer: 1, pos: 103
type: DSZ, layer: 1, pos: 120
type: DSZ, layer: 1, pos: 136
type: DSZ, layer: 1, pos: 1
type: DSZ, layer: 1, pos: 68
type: DSZ, layer: 1, pos: 251
type: DSZ, layer: 1, pos: 29
type: DSZ, layer: 1, pos: 0
type: DSZ, layer: 1, pos: 220
type: DSZ, layer: 1, pos: 161
type: DSZ, layer: 1, pos: 221
type: DSZ, layer: 1, pos: 183
type: DSZ, layer: 1, pos: 240

Time for candidate selection: 0.00 seconds

### Candidate
type: DSZ, layer: 1, pos: 13

### Relational analysis ABCD of DS_DSZ2_DSZ2_DSZ2_DSZ1_DSZ1_DSZ2_DSZ1_DSZ1_DSZ1

#### Relational analysis ABCD result of DS_DSZ2_DSZ2_DSZ2_DSZ1_DSZ1_DSZ2_DSZ1_DSZ1_DSZ1
Status: Status.UNKNOWN
Output dim: 3, lower bound: -0.3239216, upper bound: 0.3239216
time: 1.17 seconds

### Relational analysis ABCD of DS_DSZ2_DSZ2_DSZ2_DSZ1_DSZ1_DSZ2_DSZ1_DSZ1_DSZ2

#### Relational analysis ABCD result of DS_DSZ2_DSZ2_DSZ2_DSZ1_DSZ1_DSZ2_DSZ1_DSZ1_DSZ2
Status: Status.UNKNOWN
Output dim: 3, lower bound: -0.3239216, upper bound: 0.3239216
time: 1.24 seconds

## BFS DS instance: DS_DSZ2_DSZ2_DSZ2_DSZ1_DSZ1_DSZ2_DSZ1_DSZ2

### Backsubstitution after applying DS history:
0: -0.5426022, 0.4665188, -0.5426022, 0.4665188, -1.0091211, 1.0091211
1: -0.3630026, 0.3911184, -0.3630026, 0.3911184, -0.7541210, 0.7541210
2: -0.5269889, 0.4806983, -0.5269889, 0.4806983, -1.0076871, 1.0076871
3: 0.7543352, 1.1590887, 0.7543352, 1.1590887, -0.4047535, 0.4047535
4: -0.4614211, 0.4625955, -0.4614211, 0.4625955, -0.9240167, 0.9240167
5: -0.4214845, 0.4672619, -0.4214845, 0.4672619, -0.8887464, 0.8887464
6: -0.5138452, 0.5371552, -0.5138452, 0.5371552, -1.0510004, 1.0510004
7: -0.4409947, 0.4028524, -0.4409947, 0.4028524, -0.8438470, 0.8438470
8: -0.4306176, 0.5533380, -0.4306176, 0.5533380, -0.9839556, 0.9839556
9: -0.4872051, 0.3854194, -0.4872051, 0.3854194, -0.8726246, 0.8726246

Time for backsubstitution: 0.78 seconds

### DS candidates at layer 1
type: DSZ, layer: 1, pos: 198
type: DSZ, layer: 1, pos: 83
type: DSZ, layer: 1, pos: 161
type: DSZ, layer: 1, pos: 182
type: DSZ, layer: 1, pos: 235
type: DSZ, layer: 1, pos: 183
type: DSZ, layer: 1, pos: 136
type: DSZ, layer: 1, pos: 175
type: DSZ, layer: 1, pos: 219
type: DSZ, layer: 1, pos: 1
type: DSZ, layer: 1, pos: 0
type: DSZ, layer: 1, pos: 221
type: DSZ, layer: 1, pos: 240
type: DSZ, layer: 1, pos: 13
type: DSZ, layer: 1, pos: 62
type: DSZ, layer: 1, pos: 30
type: DSZ, layer: 1, pos: 103
type: DSZ, layer: 1, pos: 71
type: DSZ, layer: 1, pos: 37
type: DSZ, layer: 1, pos: 29
type: DSZ, layer: 1, pos: 251
type: DSZ, layer: 1, pos: 176
type: DSZ, layer: 1, pos: 68
type: DSZ, layer: 1, pos: 245
type: DSZ, layer: 1, pos: 120
type: DSZ, layer: 1, pos: 220
type: DSZ, layer: 1, pos: 166
type: DSZ, layer: 1, pos: 66
type: DSZ, layer: 1, pos: 172

Time for candidate selection: 0.00 seconds

### Candidate
type: DSZ, layer: 1, pos: 198

### Relational analysis ABCD of DS_DSZ2_DSZ2_DSZ2_DSZ1_DSZ1_DSZ2_DSZ1_DSZ2_DSZ1

#### Relational analysis ABCD result of DS_DSZ2_DSZ2_DSZ2_DSZ1_DSZ1_DSZ2_DSZ1_DSZ2_DSZ1
Status: Status.VERIFIED
Output dim: 3, lower bound: -0.2981158, upper bound: 0.2981158
time: 1.15 seconds

### Relational analysis ABCD of DS_DSZ2_DSZ2_DSZ2_DSZ1_DSZ1_DSZ2_DSZ1_DSZ2_DSZ2

#### Relational analysis ABCD result of DS_DSZ2_DSZ2_DSZ2_DSZ1_DSZ1_DSZ2_DSZ1_DSZ2_DSZ2
Status: Status.VERIFIED
Output dim: 3, lower bound: -0.2981158, upper bound: 0.2981158
time: 1.14 seconds

## Summary of splitting (split count: 8)
- Time for DS candidates: 3.08 seconds
DS_DSZ2_DSZ2_DSZ2_DSZ1_DSZ1_DSZ1_DSZ2_DSZ1_DSZ1, status: Status.VERIFIED, split count: 9, time: 3.08
Output dim: 3, lower bound: -0.2811331, upper bound: 0.2811331
DS_DSZ2_DSZ2_DSZ2_DSZ1_DSZ1_DSZ1_DSZ2_DSZ1_DSZ2, status: Status.VERIFIED, split count: 9, time: 3.08
Output dim: 3, lower bound: -0.2811331, upper bound: 0.2811331
DS_DSZ2_DSZ2_DSZ2_DSZ1_DSZ1_DSZ1_DSZ2_DSZ2_DSZ1, status: Status.VERIFIED, split count: 9, time: 3.08
Output dim: 3, lower bound: -0.2909268, upper bound: 0.2909268
DS_DSZ2_DSZ2_DSZ2_DSZ1_DSZ1_DSZ1_DSZ2_DSZ2_DSZ2, status: Status.VERIFIED, split count: 9, time: 3.08
Output dim: 3, lower bound: -0.2909268, upper bound: 0.2909268
DS_DSZ2_DSZ2_DSZ2_DSZ1_DSZ1_DSZ2_DSZ1_DSZ1_DSZ1, status: Status.UNKNOWN, split count: 9, time: 3.08
Output dim: 3, lower bound: -0.3239216, upper bound: 0.3239216
DS_DSZ2_DSZ2_DSZ2_DSZ1_DSZ1_DSZ2_DSZ1_DSZ1_DSZ2, status: Status.UNKNOWN, split count: 9, time: 3.08
Output dim: 3, lower bound: -0.3239216, upper bound: 0.3239216
DS_DSZ2_DSZ2_DSZ2_DSZ1_DSZ1_DSZ2_DSZ1_DSZ2_DSZ1, status: Status.VERIFIED, split count: 9, time: 3.08
Output dim: 3, lower bound: -0.2981158, upper bound: 0.2981158
DS_DSZ2_DSZ2_DSZ2_DSZ1_DSZ1_DSZ2_DSZ1_DSZ2_DSZ2, status: Status.VERIFIED, split count: 9, time: 3.08
Output dim: 3, lower bound: -0.2981158, upper bound: 0.2981158

## BFS DS instance: DS_DSZ2_DSZ2_DSZ2_DSZ1_DSZ1_DSZ2_DSZ1_DSZ1_DSZ1

### Backsubstitution after applying DS history:
0: -0.5426022, 0.4665188, -0.5426022, 0.4665188, -1.0091211, 1.0091211
1: -0.3630026, 0.3911184, -0.3630026, 0.3911184, -0.7541210, 0.7541210
2: -0.5269889, 0.4806983, -0.5269889, 0.4806983, -1.0076871, 1.0076871
3: 0.7543352, 1.1590887, 0.7543352, 1.1590887, -0.4047535, 0.4047535
4: -0.4614211, 0.4625955, -0.4614211, 0.4625955, -0.9240167, 0.9240167
5: -0.4214845, 0.4672619, -0.4214845, 0.4672619, -0.8887464, 0.8887464
6: -0.5138452, 0.5371552, -0.5138452, 0.5371552, -1.0510004, 1.0510004
7: -0.4409947, 0.4028524, -0.4409947, 0.4028524, -0.8438470, 0.8438470
8: -0.4306176, 0.5533380, -0.4306176, 0.5533380, -0.9839556, 0.9839556
9: -0.4872051, 0.3854194, -0.4872051, 0.3854194, -0.8726246, 0.8726246

Time for backsubstitution: 0.77 seconds

### DS candidates at layer 1
type: DSZ, layer: 1, pos: 161
type: DSZ, layer: 1, pos: 182
type: DSZ, layer: 1, pos: 1
type: DSZ, layer: 1, pos: 220
type: DSZ, layer: 1, pos: 71
type: DSZ, layer: 1, pos: 29
type: DSZ, layer: 1, pos: 66
type: DSZ, layer: 1, pos: 176
type: DSZ, layer: 1, pos: 235
type: DSZ, layer: 1, pos: 37
type: DSZ, layer: 1, pos: 83
type: DSZ, layer: 1, pos: 0
type: DSZ, layer: 1, pos: 175
type: DSZ, layer: 1, pos: 68
type: DSZ, layer: 1, pos: 240
type: DSZ, layer: 1, pos: 183
type: DSZ, layer: 1, pos: 221
type: DSZ, layer: 1, pos: 245
type: DSZ, layer: 1, pos: 251
type: DSZ, layer: 1, pos: 172
type: DSZ, layer: 1, pos: 166
type: DSZ, layer: 1, pos: 136
type: DSZ, layer: 1, pos: 219
type: DSZ, layer: 1, pos: 103
type: DSZ, layer: 1, pos: 30
type: DSZ, layer: 1, pos: 62
type: DSZ, layer: 1, pos: 120
type: DSZ, layer: 1, pos: 198

Time for candidate selection: 0.00 seconds

### Candidate
type: DSZ, layer: 1, pos: 161

### Relational analysis ABCD of DS_DSZ2_DSZ2_DSZ2_DSZ1_DSZ1_DSZ2_DSZ1_DSZ1_DSZ1_DSZ1

#### Relational analysis ABCD result of DS_DSZ2_DSZ2_DSZ2_DSZ1_DSZ1_DSZ2_DSZ1_DSZ1_DSZ1_DSZ1
Status: Status.VERIFIED
Output dim: 3, lower bound: -0.2814206, upper bound: 0.2814206
time: 0.98 seconds

### Relational analysis ABCD of DS_DSZ2_DSZ2_DSZ2_DSZ1_DSZ1_DSZ2_DSZ1_DSZ1_DSZ1_DSZ2

#### Relational analysis ABCD result of DS_DSZ2_DSZ2_DSZ2_DSZ1_DSZ1_DSZ2_DSZ1_DSZ1_DSZ1_DSZ2
Status: Status.VERIFIED
Output dim: 3, lower bound: -0.2814206, upper bound: 0.2814206
time: 0.99 seconds

## BFS DS instance: DS_DSZ2_DSZ2_DSZ2_DSZ1_DSZ1_DSZ2_DSZ1_DSZ1_DSZ2

### Backsubstitution after applying DS history:
0: -0.5426022, 0.4665188, -0.5426022, 0.4665188, -1.0091211, 1.0091211
1: -0.3630026, 0.3911184, -0.3630026, 0.3911184, -0.7541210, 0.7541210
2: -0.5269889, 0.4806983, -0.5269889, 0.4806983, -1.0076871, 1.0076871
3: 0.7543352, 1.1590887, 0.7543352, 1.1590887, -0.4047535, 0.4047535
4: -0.4614211, 0.4625955, -0.4614211, 0.4625955, -0.9240167, 0.9240167
5: -0.4214845, 0.4672619, -0.4214845, 0.4672619, -0.8887464, 0.8887464
6: -0.5138452, 0.5371552, -0.5138452, 0.5371552, -1.0510004, 1.0510004
7: -0.4409947, 0.4028524, -0.4409947, 0.4028524, -0.8438470, 0.8438470
8: -0.4306176, 0.5533380, -0.4306176, 0.5533380, -0.9839556, 0.9839556
9: -0.4872051, 0.3854194, -0.4872051, 0.3854194, -0.8726246, 0.8726246

Time for backsubstitution: 0.78 seconds

### DS candidates at layer 1
type: DSZ, layer: 1, pos: 245
type: DSZ, layer: 1, pos: 1
type: DSZ, layer: 1, pos: 30
type: DSZ, layer: 1, pos: 136
type: DSZ, layer: 1, pos: 221
type: DSZ, layer: 1, pos: 176
type: DSZ, layer: 1, pos: 182
type: DSZ, layer: 1, pos: 235
type: DSZ, layer: 1, pos: 83
type: DSZ, layer: 1, pos: 66
type: DSZ, layer: 1, pos: 161
type: DSZ, layer: 1, pos: 71
type: DSZ, layer: 1, pos: 37
type: DSZ, layer: 1, pos: 240
type: DSZ, layer: 1, pos: 175
type: DSZ, layer: 1, pos: 29
type: DSZ, layer: 1, pos: 0
type: DSZ, layer: 1, pos: 103
type: DSZ, layer: 1, pos: 198
type: DSZ, layer: 1, pos: 62
type: DSZ, layer: 1, pos: 219
type: DSZ, layer: 1, pos: 183
type: DSZ, layer: 1, pos: 220
type: DSZ, layer: 1, pos: 166
type: DSZ, layer: 1, pos: 172
type: DSZ, layer: 1, pos: 120
type: DSZ, layer: 1, pos: 68
type: DSZ, layer: 1, pos: 251

Time for candidate selection: 0.00 seconds

### Candidate
type: DSZ, layer: 1, pos: 245

### Relational analysis ABCD of DS_DSZ2_DSZ2_DSZ2_DSZ1_DSZ1_DSZ2_DSZ1_DSZ1_DSZ2_DSZ1

#### Relational analysis ABCD result of DS_DSZ2_DSZ2_DSZ2_DSZ1_DSZ1_DSZ2_DSZ1_DSZ1_DSZ2_DSZ1
Status: Status.VERIFIED
Output dim: 3, lower bound: -0.2911361, upper bound: 0.2911361
time: 1.11 seconds

### Relational analysis ABCD of DS_DSZ2_DSZ2_DSZ2_DSZ1_DSZ1_DSZ2_DSZ1_DSZ1_DSZ2_DSZ2

#### Relational analysis ABCD result of DS_DSZ2_DSZ2_DSZ2_DSZ1_DSZ1_DSZ2_DSZ1_DSZ1_DSZ2_DSZ2
Status: Status.VERIFIED
Output dim: 3, lower bound: -0.2911361, upper bound: 0.2911361
time: 1.10 seconds

## Summary of splitting (split count: 9)
- Time for DS candidates: 3.00 seconds
DS_DSZ2_DSZ2_DSZ2_DSZ1_DSZ1_DSZ2_DSZ1_DSZ1_DSZ1_DSZ1, status: Status.VERIFIED, split count: 10, time: 3.00
Output dim: 3, lower bound: -0.2814206, upper bound: 0.2814206
DS_DSZ2_DSZ2_DSZ2_DSZ1_DSZ1_DSZ2_DSZ1_DSZ1_DSZ1_DSZ2, status: Status.VERIFIED, split count: 10, time: 3.00
Output dim: 3, lower bound: -0.2814206, upper bound: 0.2814206
DS_DSZ2_DSZ2_DSZ2_DSZ1_DSZ1_DSZ2_DSZ1_DSZ1_DSZ2_DSZ1, status: Status.VERIFIED, split count: 10, time: 3.00
Output dim: 3, lower bound: -0.2911361, upper bound: 0.2911361
DS_DSZ2_DSZ2_DSZ2_DSZ1_DSZ1_DSZ2_DSZ1_DSZ1_DSZ2_DSZ2, status: Status.VERIFIED, split count: 10, time: 3.00
Output dim: 3, lower bound: -0.2911361, upper bound: 0.2911361

## DS Result
status: Status.VERIFIED
execution time: (base) + (ds) = 3.06 + 113.44 = 116.51 seconds
