## Execution arguments:
Dataset: Dataset.MNIST
Network: onnx/mnist-net_256x4.onnx
Relational property: GLOBAL_ROBUSTNESS
LP Analysis: True
Epsilon: 0.01171875
Delta epsilon: 0.00390625
execution index: (1, 3, 2)
Time budget: 600 seconds
Split limit: 100
Threshold: 0.0055854


## IAR start

### Unstable ReLU Count (Linear/Conv2D Layers)
- layer_idx=0, type=LayerType.Linear, total=256, inp1_unstable=15, inp2_unstable=15, delta_unstable=256
- layer_idx=2, type=LayerType.Linear, total=256, inp1_unstable=19, inp2_unstable=19, delta_unstable=256
- layer_idx=4, type=LayerType.Linear, total=256, inp1_unstable=0, inp2_unstable=0, delta_unstable=256
- layer_idx=6, type=LayerType.Linear, total=256, inp1_unstable=0, inp2_unstable=0, delta_unstable=256
- layer_idx=8, type=LayerType.Linear, total=10, inp1_unstable=1, inp2_unstable=1, delta_unstable=10

### BASE IAR bounds
Layer (inp1_lb, inp1_ub, inp2_lb, inp2_ub, d_lb, d_ub)
0: (-0.0044784, -0.0038492, -0.0044784, -0.0038492, -0.0005995, 0.0005995)
1: (0.0010607, 0.0045450, 0.0010607, 0.0045450, -0.0033195, 0.0033195)
2: (0.0048122, 0.0125964, 0.0048122, 0.0125964, -0.0074161, 0.0074161)
3: (0.0020262, 0.0053065, 0.0020262, 0.0053065, -0.0031252, 0.0031252)
4: (1.0046113, 1.0173376, 1.0046113, 1.0173376, -0.0121244, 0.0121244)
5: (0.0031385, 0.0056142, 0.0031385, 0.0056142, -0.0023587, 0.0023587)
6: (-0.0130490, -0.0098272, -0.0130490, -0.0098272, -0.0030695, 0.0030695)
7: (-0.0104679, -0.0100569, -0.0104679, -0.0100569, -0.0003915, 0.0003915)
8: (-0.0040643, -0.0018383, -0.0040643, -0.0018383, -0.0021208, 0.0021208)
9: (-0.0089681, 0.0021759, -0.0089681, 0.0021759, -0.0106170, 0.0106170)

## BASE Relational Analysis

## BASE Result
execution time: IAR + RelationalAnalysis = 1.18 + 2.10 = 3.28 seconds
status: Status.UNKNOWN
relational distance
Output dim: 4, lower bound: -0.0062060, upper bound: 0.0062060

# Relational Split (RS) starts

## BFS RS instance: RS

Time for backsubstitution: 0.00 seconds

### RS candidates at layer 1
type: RSZ, layer: 1, pos: 136
type: RSZ, layer: 1, pos: 104
type: RSZ, layer: 1, pos: 179
type: RSZ, layer: 1, pos: 238
type: RSZ, layer: 1, pos: 155
type: RSZ, layer: 1, pos: 76
type: RSZ, layer: 1, pos: 182
type: RSZ, layer: 1, pos: 15
type: RSZ, layer: 1, pos: 132
type: RSZ, layer: 1, pos: 250
type: RSZ, layer: 1, pos: 30
type: RSZ, layer: 1, pos: 204
type: RSZ, layer: 1, pos: 197
type: RSZ, layer: 1, pos: 85
type: RSZ, layer: 1, pos: 216

Time for candidate selection: 0.00 seconds

### Candidate
type: RSZ, layer: 1, pos: 136

### Relational analysis RSZ of RS_RSZ1

#### Relational analysis RSZ result of RS_RSZ1
Status: Status.UNKNOWN
Output dim: 4, lower bound: -0.0059466, upper bound: 0.0059466
time: 1.16 seconds

### Relational analysis RSZ of RS_RSZ2

#### Relational analysis RSZ result of RS_RSZ2
Status: Status.UNKNOWN
Output dim: 4, lower bound: -0.0059466, upper bound: 0.0059466
time: 1.17 seconds

## Summary of splitting (split count: 0)
- Time for RS candidates: 2.34 seconds
RS_RSZ1, status: Status.UNKNOWN, split count: 1, time: 2.34
Output dim: 4, lower bound: -0.0059466, upper bound: 0.0059466
RS_RSZ2, status: Status.UNKNOWN, split count: 1, time: 2.34
Output dim: 4, lower bound: -0.0059466, upper bound: 0.0059466

## BFS RS instance: RS_RSZ1

### Backsubstitution after applying RS history:
0: -0.0044784, -0.0038492, -0.0044784, -0.0038492, -0.0005948, 0.0005958
1: 0.0010607, 0.0045450, 0.0010607, 0.0045450, -0.0032990, 0.0032936
2: 0.0048122, 0.0125964, 0.0048122, 0.0125964, -0.0073582, 0.0073703
3: 0.0020262, 0.0053065, 0.0020262, 0.0053065, -0.0031059, 0.0031008
4: 1.0046113, 1.0173376, 1.0046113, 1.0173376, -0.0120496, 0.0120298
5: 0.0031385, 0.0056142, 0.0031385, 0.0056142, -0.0023441, 0.0023403
6: -0.0130490, -0.0098272, -0.0130490, -0.0098272, -0.0030455, 0.0030505
7: -0.0104679, -0.0100569, -0.0104679, -0.0100569, -0.0003885, 0.0003891
8: -0.0040643, -0.0018383, -0.0040643, -0.0018383, -0.0021077, 0.0021042
9: -0.0089681, 0.0021759, -0.0089681, 0.0021759, -0.0105342, 0.0105516

### Unstable ReLU Count (Linear/Conv2D Layers)
- layer_idx=0, type=LayerType.Linear, total=256, inp1_unstable=15, inp2_unstable=15, delta_unstable=255
- layer_idx=2, type=LayerType.Linear, total=256, inp1_unstable=19, inp2_unstable=19, delta_unstable=256
- layer_idx=4, type=LayerType.Linear, total=256, inp1_unstable=0, inp2_unstable=0, delta_unstable=256
- layer_idx=6, type=LayerType.Linear, total=256, inp1_unstable=0, inp2_unstable=0, delta_unstable=256
- layer_idx=8, type=LayerType.Linear, total=10, inp1_unstable=1, inp2_unstable=1, delta_unstable=10

Time for backsubstitution: 1.09 seconds

### RS candidates at layer 1
type: RSZ, layer: 1, pos: 30
type: RSZ, layer: 1, pos: 15
type: RSZ, layer: 1, pos: 76
type: RSZ, layer: 1, pos: 250
type: RSZ, layer: 1, pos: 104
type: RSZ, layer: 1, pos: 204
type: RSZ, layer: 1, pos: 238
type: RSZ, layer: 1, pos: 182
type: RSZ, layer: 1, pos: 155
type: RSZ, layer: 1, pos: 179
type: RSZ, layer: 1, pos: 132
type: RSZ, layer: 1, pos: 85
type: RSZ, layer: 1, pos: 216
type: RSZ, layer: 1, pos: 197

Time for candidate selection: 0.00 seconds

### Candidate
type: RSZ, layer: 1, pos: 30

### Relational analysis RSZ of RS_RSZ1_RSZ1
Optimization infeasible because this subproblem isn't reachable.

### Candidate
type: RSZ, layer: 1, pos: 15

### Relational analysis RSZ of RS_RSZ1_RSZ1

#### Relational analysis RSZ result of RS_RSZ1_RSZ1
Status: Status.UNKNOWN
Output dim: 4, lower bound: -0.0057675, upper bound: 0.0056977
time: 1.17 seconds

### Relational analysis RSZ of RS_RSZ1_RSZ2

#### Relational analysis RSZ result of RS_RSZ1_RSZ2
Status: Status.UNKNOWN
Output dim: 4, lower bound: -0.0056977, upper bound: 0.0057675
time: 1.14 seconds

## BFS RS instance: RS_RSZ2

### Backsubstitution after applying RS history:
0: -0.0044784, -0.0038492, -0.0044784, -0.0038492, -0.0005958, 0.0005995
1: 0.0010607, 0.0045450, 0.0010607, 0.0045450, -0.0033195, 0.0032990
2: 0.0048122, 0.0125964, 0.0048122, 0.0125964, -0.0073703, 0.0074161
3: 0.0020262, 0.0053065, 0.0020262, 0.0053065, -0.0031252, 0.0031059
4: 1.0046113, 1.0173376, 1.0046113, 1.0173376, -0.0121244, 0.0120496
5: 0.0031385, 0.0056142, 0.0031385, 0.0056142, -0.0023587, 0.0023441
6: -0.0130490, -0.0098272, -0.0130490, -0.0098272, -0.0030505, 0.0030695
7: -0.0104679, -0.0100569, -0.0104679, -0.0100569, -0.0003891, 0.0003915
8: -0.0040643, -0.0018383, -0.0040643, -0.0018383, -0.0021208, 0.0021077
9: -0.0089681, 0.0021759, -0.0089681, 0.0021759, -0.0105516, 0.0106170

### Unstable ReLU Count (Linear/Conv2D Layers)
- layer_idx=0, type=LayerType.Linear, total=256, inp1_unstable=15, inp2_unstable=15, delta_unstable=255
- layer_idx=2, type=LayerType.Linear, total=256, inp1_unstable=19, inp2_unstable=19, delta_unstable=256
- layer_idx=4, type=LayerType.Linear, total=256, inp1_unstable=0, inp2_unstable=0, delta_unstable=256
- layer_idx=6, type=LayerType.Linear, total=256, inp1_unstable=0, inp2_unstable=0, delta_unstable=256
- layer_idx=8, type=LayerType.Linear, total=10, inp1_unstable=1, inp2_unstable=1, delta_unstable=10

Time for backsubstitution: 1.08 seconds

### RS candidates at layer 1
type: RSZ, layer: 1, pos: 204
type: RSZ, layer: 1, pos: 250
type: RSZ, layer: 1, pos: 30
type: RSZ, layer: 1, pos: 216
type: RSZ, layer: 1, pos: 132
type: RSZ, layer: 1, pos: 76
type: RSZ, layer: 1, pos: 238
type: RSZ, layer: 1, pos: 104
type: RSZ, layer: 1, pos: 182
type: RSZ, layer: 1, pos: 155
type: RSZ, layer: 1, pos: 85
type: RSZ, layer: 1, pos: 179
type: RSZ, layer: 1, pos: 15
type: RSZ, layer: 1, pos: 197

Time for candidate selection: 0.00 seconds

### Candidate
type: RSZ, layer: 1, pos: 204

### Relational analysis RSZ of RS_RSZ2_RSZ1

#### Relational analysis RSZ result of RS_RSZ2_RSZ1
Status: Status.VERIFIED
Output dim: 4, lower bound: -0.0042904, upper bound: 0.0042904
time: 0.82 seconds

### Relational analysis RSZ of RS_RSZ2_RSZ2

#### Relational analysis RSZ result of RS_RSZ2_RSZ2
Status: Status.VERIFIED
Output dim: 4, lower bound: -0.0042904, upper bound: 0.0042904
time: 0.82 seconds

## Summary of splitting (split count: 1)
- Time for RS candidates: 2.73 seconds
RS_RSZ1_RSZ1, status: Status.UNKNOWN, split count: 2, time: 2.73
Output dim: 4, lower bound: -0.0057675, upper bound: 0.0056977
RS_RSZ1_RSZ2, status: Status.UNKNOWN, split count: 2, time: 2.73
Output dim: 4, lower bound: -0.0056977, upper bound: 0.0057675
RS_RSZ2_RSZ1, status: Status.VERIFIED, split count: 2, time: 2.73
Output dim: 4, lower bound: -0.0042904, upper bound: 0.0042904
RS_RSZ2_RSZ2, status: Status.VERIFIED, split count: 2, time: 2.73
Output dim: 4, lower bound: -0.0042904, upper bound: 0.0042904

## BFS RS instance: RS_RSZ1_RSZ1

### Backsubstitution after applying RS history:
0: -0.0044784, -0.0038492, -0.0044784, -0.0038492, -0.0005664, 0.0005698
1: 0.0010607, 0.0045450, 0.0010607, 0.0045450, -0.0031550, 0.0031364
2: 0.0048122, 0.0125964, 0.0048122, 0.0125964, -0.0070071, 0.0070486
3: 0.0020262, 0.0053065, 0.0020262, 0.0053065, -0.0029703, 0.0029528
4: 1.0046113, 1.0173376, 1.0046113, 1.0173376, -0.0115236, 0.0114557
5: 0.0031385, 0.0056142, 0.0031385, 0.0056142, -0.0022418, 0.0022286
6: -0.0130490, -0.0098272, -0.0130490, -0.0098272, -0.0029002, 0.0029174
7: -0.0104679, -0.0100569, -0.0104679, -0.0100569, -0.0003699, 0.0003721
8: -0.0040643, -0.0018383, -0.0040643, -0.0018383, -0.0020157, 0.0020038
9: -0.0089681, 0.0021759, -0.0089681, 0.0021759, -0.0100315, 0.0100910

### Unstable ReLU Count (Linear/Conv2D Layers)
- layer_idx=0, type=LayerType.Linear, total=256, inp1_unstable=15, inp2_unstable=15, delta_unstable=254
- layer_idx=2, type=LayerType.Linear, total=256, inp1_unstable=19, inp2_unstable=19, delta_unstable=256
- layer_idx=4, type=LayerType.Linear, total=256, inp1_unstable=0, inp2_unstable=0, delta_unstable=256
- layer_idx=6, type=LayerType.Linear, total=256, inp1_unstable=0, inp2_unstable=0, delta_unstable=256
- layer_idx=8, type=LayerType.Linear, total=10, inp1_unstable=1, inp2_unstable=1, delta_unstable=10

Time for backsubstitution: 1.09 seconds

### RS candidates at layer 1
type: RSZ, layer: 1, pos: 204
type: RSZ, layer: 1, pos: 197
type: RSZ, layer: 1, pos: 132
type: RSZ, layer: 1, pos: 85
type: RSZ, layer: 1, pos: 104
type: RSZ, layer: 1, pos: 179
type: RSZ, layer: 1, pos: 155
type: RSZ, layer: 1, pos: 182
type: RSZ, layer: 1, pos: 250
type: RSZ, layer: 1, pos: 238
type: RSZ, layer: 1, pos: 76
type: RSZ, layer: 1, pos: 216
type: RSZ, layer: 1, pos: 30

Time for candidate selection: 0.00 seconds

### Candidate
type: RSZ, layer: 1, pos: 204

### Relational analysis RSZ of RS_RSZ1_RSZ1_RSZ1

#### Relational analysis RSZ result of RS_RSZ1_RSZ1_RSZ1
Status: Status.VERIFIED
Output dim: 4, lower bound: -0.0040784, upper bound: 0.0040564
time: 0.84 seconds

### Relational analysis RSZ of RS_RSZ1_RSZ1_RSZ2

#### Relational analysis RSZ result of RS_RSZ1_RSZ1_RSZ2
Status: Status.VERIFIED
Output dim: 4, lower bound: -0.0040784, upper bound: 0.0040564
time: 0.84 seconds

## BFS RS instance: RS_RSZ1_RSZ2

### Backsubstitution after applying RS history:
0: -0.0044784, -0.0038492, -0.0044784, -0.0038492, -0.0005688, 0.0005674
1: 0.0010607, 0.0045450, 0.0010607, 0.0045450, -0.0031418, 0.0031494
2: 0.0048122, 0.0125964, 0.0048122, 0.0125964, -0.0070361, 0.0070192
3: 0.0020262, 0.0053065, 0.0020262, 0.0053065, -0.0029579, 0.0029650
4: 1.0046113, 1.0173376, 1.0046113, 1.0173376, -0.0114755, 0.0115031
5: 0.0031385, 0.0056142, 0.0031385, 0.0056142, -0.0022324, 0.0022378
6: -0.0130490, -0.0098272, -0.0130490, -0.0098272, -0.0029122, 0.0029052
7: -0.0104679, -0.0100569, -0.0104679, -0.0100569, -0.0003715, 0.0003706
8: -0.0040643, -0.0018383, -0.0040643, -0.0018383, -0.0020073, 0.0020121
9: -0.0089681, 0.0021759, -0.0089681, 0.0021759, -0.0100730, 0.0100488

### Unstable ReLU Count (Linear/Conv2D Layers)
- layer_idx=0, type=LayerType.Linear, total=256, inp1_unstable=15, inp2_unstable=15, delta_unstable=254
- layer_idx=2, type=LayerType.Linear, total=256, inp1_unstable=19, inp2_unstable=19, delta_unstable=256
- layer_idx=4, type=LayerType.Linear, total=256, inp1_unstable=0, inp2_unstable=0, delta_unstable=256
- layer_idx=6, type=LayerType.Linear, total=256, inp1_unstable=0, inp2_unstable=0, delta_unstable=256
- layer_idx=8, type=LayerType.Linear, total=10, inp1_unstable=1, inp2_unstable=1, delta_unstable=10

Time for backsubstitution: 1.08 seconds

### RS candidates at layer 1
type: RSZ, layer: 1, pos: 104
type: RSZ, layer: 1, pos: 76
type: RSZ, layer: 1, pos: 238
type: RSZ, layer: 1, pos: 132
type: RSZ, layer: 1, pos: 182
type: RSZ, layer: 1, pos: 216
type: RSZ, layer: 1, pos: 30
type: RSZ, layer: 1, pos: 85
type: RSZ, layer: 1, pos: 197
type: RSZ, layer: 1, pos: 179
type: RSZ, layer: 1, pos: 204
type: RSZ, layer: 1, pos: 250
type: RSZ, layer: 1, pos: 155

Time for candidate selection: 0.00 seconds

### Candidate
type: RSZ, layer: 1, pos: 104

### Relational analysis RSZ of RS_RSZ1_RSZ2_RSZ1

#### Relational analysis RSZ result of RS_RSZ1_RSZ2_RSZ1
Status: Status.UNKNOWN
Output dim: 4, lower bound: -0.0056836, upper bound: 0.0054002
time: 1.23 seconds

### Relational analysis RSZ of RS_RSZ1_RSZ2_RSZ2

#### Relational analysis RSZ result of RS_RSZ1_RSZ2_RSZ2
Status: Status.UNKNOWN
Output dim: 4, lower bound: -0.0053792, upper bound: 0.0057536
time: 1.32 seconds

## Summary of splitting (split count: 2)
- Time for RS candidates: 3.63 seconds
RS_RSZ1_RSZ1_RSZ1, status: Status.VERIFIED, split count: 3, time: 3.63
Output dim: 4, lower bound: -0.0040784, upper bound: 0.0040564
RS_RSZ1_RSZ1_RSZ2, status: Status.VERIFIED, split count: 3, time: 3.63
Output dim: 4, lower bound: -0.0040784, upper bound: 0.0040564
RS_RSZ1_RSZ2_RSZ1, status: Status.UNKNOWN, split count: 3, time: 3.63
Output dim: 4, lower bound: -0.0056836, upper bound: 0.0054002
RS_RSZ1_RSZ2_RSZ2, status: Status.UNKNOWN, split count: 3, time: 3.63
Output dim: 4, lower bound: -0.0053792, upper bound: 0.0057536

## BFS RS instance: RS_RSZ1_RSZ2_RSZ1

### Backsubstitution after applying RS history:
0: -0.0044784, -0.0038492, -0.0044784, -0.0038492, -0.0005775, 0.0005844
1: 0.0010607, 0.0045450, 0.0010607, 0.0045450, -0.0032359, 0.0031976
2: 0.0048122, 0.0125964, 0.0048122, 0.0125964, -0.0071437, 0.0072294
3: 0.0020262, 0.0053065, 0.0020262, 0.0053065, -0.0030465, 0.0030104
4: 1.0046113, 1.0173376, 1.0046113, 1.0173376, -0.0118192, 0.0116791
5: 0.0031385, 0.0056142, 0.0031385, 0.0056142, -0.0022993, 0.0022720
6: -0.0130490, -0.0098272, -0.0130490, -0.0098272, -0.0029567, 0.0029922
7: -0.0104679, -0.0100569, -0.0104679, -0.0100569, -0.0003772, 0.0003817
8: -0.0040643, -0.0018383, -0.0040643, -0.0018383, -0.0020674, 0.0020429
9: -0.0089681, 0.0021759, -0.0089681, 0.0021759, -0.0102271, 0.0103498

### Unstable ReLU Count (Linear/Conv2D Layers)
- layer_idx=0, type=LayerType.Linear, total=256, inp1_unstable=15, inp2_unstable=15, delta_unstable=253
- layer_idx=2, type=LayerType.Linear, total=256, inp1_unstable=19, inp2_unstable=19, delta_unstable=256
- layer_idx=4, type=LayerType.Linear, total=256, inp1_unstable=0, inp2_unstable=0, delta_unstable=256
- layer_idx=6, type=LayerType.Linear, total=256, inp1_unstable=0, inp2_unstable=0, delta_unstable=256
- layer_idx=8, type=LayerType.Linear, total=10, inp1_unstable=1, inp2_unstable=1, delta_unstable=10

Time for backsubstitution: 1.10 seconds

### RS candidates at layer 1
type: RSZ, layer: 1, pos: 238
type: RSZ, layer: 1, pos: 179
type: RSZ, layer: 1, pos: 182
type: RSZ, layer: 1, pos: 204
type: RSZ, layer: 1, pos: 197
type: RSZ, layer: 1, pos: 76
type: RSZ, layer: 1, pos: 250
type: RSZ, layer: 1, pos: 85
type: RSZ, layer: 1, pos: 155
type: RSZ, layer: 1, pos: 216
type: RSZ, layer: 1, pos: 30
type: RSZ, layer: 1, pos: 132

Time for candidate selection: 0.00 seconds

### Candidate
type: RSZ, layer: 1, pos: 238

### Relational analysis RSZ of RS_RSZ1_RSZ2_RSZ1_RSZ1

#### Relational analysis RSZ result of RS_RSZ1_RSZ2_RSZ1_RSZ1
Status: Status.UNKNOWN
Output dim: 4, lower bound: -0.0055890, upper bound: 0.0053166
time: 1.58 seconds

### Relational analysis RSZ of RS_RSZ1_RSZ2_RSZ1_RSZ2

#### Relational analysis RSZ result of RS_RSZ1_RSZ2_RSZ1_RSZ2
Status: Status.VERIFIED
Output dim: 4, lower bound: -0.0055847, upper bound: 0.0053172
time: 1.55 seconds

## BFS RS instance: RS_RSZ1_RSZ2_RSZ2

### Backsubstitution after applying RS history:
0: -0.0044784, -0.0038492, -0.0044784, -0.0038492, -0.0005860, 0.0005761
1: 0.0010607, 0.0045450, 0.0010607, 0.0045450, -0.0031900, 0.0032446
2: 0.0048122, 0.0125964, 0.0048122, 0.0125964, -0.0072488, 0.0071268
3: 0.0020262, 0.0053065, 0.0020262, 0.0053065, -0.0030033, 0.0030547
4: 1.0046113, 1.0173376, 1.0046113, 1.0173376, -0.0116515, 0.0118509
5: 0.0031385, 0.0056142, 0.0031385, 0.0056142, -0.0022667, 0.0023055
6: -0.0130490, -0.0098272, -0.0130490, -0.0098272, -0.0030002, 0.0029498
7: -0.0104679, -0.0100569, -0.0104679, -0.0100569, -0.0003827, 0.0003763
8: -0.0040643, -0.0018383, -0.0040643, -0.0018383, -0.0020380, 0.0020729
9: -0.0089681, 0.0021759, -0.0089681, 0.0021759, -0.0103775, 0.0102030

### Unstable ReLU Count (Linear/Conv2D Layers)
- layer_idx=0, type=LayerType.Linear, total=256, inp1_unstable=15, inp2_unstable=15, delta_unstable=253
- layer_idx=2, type=LayerType.Linear, total=256, inp1_unstable=19, inp2_unstable=19, delta_unstable=256
- layer_idx=4, type=LayerType.Linear, total=256, inp1_unstable=0, inp2_unstable=0, delta_unstable=256
- layer_idx=6, type=LayerType.Linear, total=256, inp1_unstable=0, inp2_unstable=0, delta_unstable=256
- layer_idx=8, type=LayerType.Linear, total=10, inp1_unstable=1, inp2_unstable=1, delta_unstable=10

Time for backsubstitution: 1.08 seconds

### RS candidates at layer 1
type: RSZ, layer: 1, pos: 216
type: RSZ, layer: 1, pos: 155
type: RSZ, layer: 1, pos: 76
type: RSZ, layer: 1, pos: 204
type: RSZ, layer: 1, pos: 179
type: RSZ, layer: 1, pos: 182
type: RSZ, layer: 1, pos: 238
type: RSZ, layer: 1, pos: 30
type: RSZ, layer: 1, pos: 132
type: RSZ, layer: 1, pos: 197
type: RSZ, layer: 1, pos: 85
type: RSZ, layer: 1, pos: 250

Time for candidate selection: 0.00 seconds

### Candidate
type: RSZ, layer: 1, pos: 216

### Relational analysis RSZ of RS_RSZ1_RSZ2_RSZ2_RSZ1

#### Relational analysis RSZ result of RS_RSZ1_RSZ2_RSZ2_RSZ1
Status: Status.UNKNOWN
Output dim: 4, lower bound: -0.0052611, upper bound: 0.0056383
time: 1.56 seconds

### Relational analysis RSZ of RS_RSZ1_RSZ2_RSZ2_RSZ2

#### Relational analysis RSZ result of RS_RSZ1_RSZ2_RSZ2_RSZ2
Status: Status.UNKNOWN
Output dim: 4, lower bound: -0.0052664, upper bound: 0.0056376
time: 1.50 seconds

## Summary of splitting (split count: 3)
- Time for RS candidates: 4.15 seconds
RS_RSZ1_RSZ2_RSZ1_RSZ1, status: Status.UNKNOWN, split count: 4, time: 4.15
Output dim: 4, lower bound: -0.0055890, upper bound: 0.0053166
RS_RSZ1_RSZ2_RSZ1_RSZ2, status: Status.VERIFIED, split count: 4, time: 4.15
Output dim: 4, lower bound: -0.0055847, upper bound: 0.0053172
RS_RSZ1_RSZ2_RSZ2_RSZ1, status: Status.UNKNOWN, split count: 4, time: 4.15
Output dim: 4, lower bound: -0.0052611, upper bound: 0.0056383
RS_RSZ1_RSZ2_RSZ2_RSZ2, status: Status.UNKNOWN, split count: 4, time: 4.15
Output dim: 4, lower bound: -0.0052664, upper bound: 0.0056376

## BFS RS instance: RS_RSZ1_RSZ2_RSZ1_RSZ1

### Backsubstitution after applying RS history:
0: -0.0044784, -0.0038492, -0.0044784, -0.0038492, -0.0005740, 0.0005816
1: 0.0010607, 0.0045450, 0.0010607, 0.0045450, -0.0032205, 0.0031783
2: 0.0048122, 0.0125964, 0.0048122, 0.0125964, -0.0071006, 0.0071949
3: 0.0020262, 0.0053065, 0.0020262, 0.0053065, -0.0030319, 0.0029922
4: 1.0046113, 1.0173376, 1.0046113, 1.0173376, -0.0117628, 0.0116087
5: 0.0031385, 0.0056142, 0.0031385, 0.0056142, -0.0022883, 0.0022583
6: -0.0130490, -0.0098272, -0.0130490, -0.0098272, -0.0029389, 0.0029779
7: -0.0104679, -0.0100569, -0.0104679, -0.0100569, -0.0003749, 0.0003799
8: -0.0040643, -0.0018383, -0.0040643, -0.0018383, -0.0020575, 0.0020305
9: -0.0089681, 0.0021759, -0.0089681, 0.0021759, -0.0101654, 0.0103004

### Unstable ReLU Count (Linear/Conv2D Layers)
- layer_idx=0, type=LayerType.Linear, total=256, inp1_unstable=15, inp2_unstable=15, delta_unstable=252
- layer_idx=2, type=LayerType.Linear, total=256, inp1_unstable=19, inp2_unstable=19, delta_unstable=256
- layer_idx=4, type=LayerType.Linear, total=256, inp1_unstable=0, inp2_unstable=0, delta_unstable=256
- layer_idx=6, type=LayerType.Linear, total=256, inp1_unstable=0, inp2_unstable=0, delta_unstable=256
- layer_idx=8, type=LayerType.Linear, total=10, inp1_unstable=1, inp2_unstable=1, delta_unstable=10

Time for backsubstitution: 1.07 seconds

### RS candidates at layer 1
type: RSZ, layer: 1, pos: 250
type: RSZ, layer: 1, pos: 30
type: RSZ, layer: 1, pos: 216
type: RSZ, layer: 1, pos: 182
type: RSZ, layer: 1, pos: 85
type: RSZ, layer: 1, pos: 179
type: RSZ, layer: 1, pos: 197
type: RSZ, layer: 1, pos: 132
type: RSZ, layer: 1, pos: 155
type: RSZ, layer: 1, pos: 76
type: RSZ, layer: 1, pos: 204

Time for candidate selection: 0.00 seconds

### Candidate
type: RSZ, layer: 1, pos: 250

### Relational analysis RSZ of RS_RSZ1_RSZ2_RSZ1_RSZ1_RSZ1

#### Relational analysis RSZ result of RS_RSZ1_RSZ2_RSZ1_RSZ1_RSZ1
Status: Status.VERIFIED
Output dim: 4, lower bound: -0.0055112, upper bound: 0.0052701
time: 1.25 seconds

### Relational analysis RSZ of RS_RSZ1_RSZ2_RSZ1_RSZ1_RSZ2

#### Relational analysis RSZ result of RS_RSZ1_RSZ2_RSZ1_RSZ1_RSZ2
Status: Status.VERIFIED
Output dim: 4, lower bound: -0.0055418, upper bound: 0.0052373
time: 1.18 seconds

## BFS RS instance: RS_RSZ1_RSZ2_RSZ2_RSZ1

### Backsubstitution after applying RS history:
0: -0.0044784, -0.0038492, -0.0044784, -0.0038492, -0.0005807, 0.0005714
1: 0.0010607, 0.0045450, 0.0010607, 0.0045450, -0.0031637, 0.0032156
2: 0.0048122, 0.0125964, 0.0048122, 0.0125964, -0.0071840, 0.0070681
3: 0.0020262, 0.0053065, 0.0020262, 0.0053065, -0.0029785, 0.0030273
4: 1.0046113, 1.0173376, 1.0046113, 1.0173376, -0.0115555, 0.0117449
5: 0.0031385, 0.0056142, 0.0031385, 0.0056142, -0.0022480, 0.0022849
6: -0.0130490, -0.0098272, -0.0130490, -0.0098272, -0.0029734, 0.0029255
7: -0.0104679, -0.0100569, -0.0104679, -0.0100569, -0.0003793, 0.0003732
8: -0.0040643, -0.0018383, -0.0040643, -0.0018383, -0.0020213, 0.0020544
9: -0.0089681, 0.0021759, -0.0089681, 0.0021759, -0.0102847, 0.0101189

### Unstable ReLU Count (Linear/Conv2D Layers)
- layer_idx=0, type=LayerType.Linear, total=256, inp1_unstable=15, inp2_unstable=15, delta_unstable=252
- layer_idx=2, type=LayerType.Linear, total=256, inp1_unstable=19, inp2_unstable=19, delta_unstable=256
- layer_idx=4, type=LayerType.Linear, total=256, inp1_unstable=0, inp2_unstable=0, delta_unstable=256
- layer_idx=6, type=LayerType.Linear, total=256, inp1_unstable=0, inp2_unstable=0, delta_unstable=256
- layer_idx=8, type=LayerType.Linear, total=10, inp1_unstable=1, inp2_unstable=1, delta_unstable=10

Time for backsubstitution: 1.08 seconds

### RS candidates at layer 1
type: RSZ, layer: 1, pos: 132
type: RSZ, layer: 1, pos: 30
type: RSZ, layer: 1, pos: 250
type: RSZ, layer: 1, pos: 85
type: RSZ, layer: 1, pos: 238
type: RSZ, layer: 1, pos: 179
type: RSZ, layer: 1, pos: 197
type: RSZ, layer: 1, pos: 155
type: RSZ, layer: 1, pos: 76
type: RSZ, layer: 1, pos: 182
type: RSZ, layer: 1, pos: 204

Time for candidate selection: 0.00 seconds

### Candidate
type: RSZ, layer: 1, pos: 132

### Relational analysis RSZ of RS_RSZ1_RSZ2_RSZ2_RSZ1_RSZ1

#### Relational analysis RSZ result of RS_RSZ1_RSZ2_RSZ2_RSZ1_RSZ1
Status: Status.VERIFIED
Output dim: 4, lower bound: -0.0052604, upper bound: 0.0055660
time: 1.43 seconds

### Relational analysis RSZ of RS_RSZ1_RSZ2_RSZ2_RSZ1_RSZ2

#### Relational analysis RSZ result of RS_RSZ1_RSZ2_RSZ2_RSZ1_RSZ2
Status: Status.UNKNOWN
Output dim: 4, lower bound: -0.0052535, upper bound: 0.0056376
time: 1.22 seconds

## BFS RS instance: RS_RSZ1_RSZ2_RSZ2_RSZ2

### Backsubstitution after applying RS history:
0: -0.0044784, -0.0038492, -0.0044784, -0.0038492, -0.0005812, 0.0005710
1: 0.0010607, 0.0045450, 0.0010607, 0.0045450, -0.0031617, 0.0032183
2: 0.0048122, 0.0125964, 0.0048122, 0.0125964, -0.0071901, 0.0070636
3: 0.0020262, 0.0053065, 0.0020262, 0.0053065, -0.0029766, 0.0030299
4: 1.0046113, 1.0173376, 1.0046113, 1.0173376, -0.0115482, 0.0117549
5: 0.0031385, 0.0056142, 0.0031385, 0.0056142, -0.0022466, 0.0022868
6: -0.0130490, -0.0098272, -0.0130490, -0.0098272, -0.0029759, 0.0029236
7: -0.0104679, -0.0100569, -0.0104679, -0.0100569, -0.0003796, 0.0003729
8: -0.0040643, -0.0018383, -0.0040643, -0.0018383, -0.0020200, 0.0020561
9: -0.0089681, 0.0021759, -0.0089681, 0.0021759, -0.0102934, 0.0101125

### Unstable ReLU Count (Linear/Conv2D Layers)
- layer_idx=0, type=LayerType.Linear, total=256, inp1_unstable=15, inp2_unstable=15, delta_unstable=252
- layer_idx=2, type=LayerType.Linear, total=256, inp1_unstable=19, inp2_unstable=19, delta_unstable=256
- layer_idx=4, type=LayerType.Linear, total=256, inp1_unstable=0, inp2_unstable=0, delta_unstable=256
- layer_idx=6, type=LayerType.Linear, total=256, inp1_unstable=0, inp2_unstable=0, delta_unstable=256
- layer_idx=8, type=LayerType.Linear, total=10, inp1_unstable=1, inp2_unstable=1, delta_unstable=10

Time for backsubstitution: 1.08 seconds

### RS candidates at layer 1
type: RSZ, layer: 1, pos: 30
type: RSZ, layer: 1, pos: 197
type: RSZ, layer: 1, pos: 76
type: RSZ, layer: 1, pos: 155
type: RSZ, layer: 1, pos: 182
type: RSZ, layer: 1, pos: 179
type: RSZ, layer: 1, pos: 238
type: RSZ, layer: 1, pos: 132
type: RSZ, layer: 1, pos: 85
type: RSZ, layer: 1, pos: 204
type: RSZ, layer: 1, pos: 250

Time for candidate selection: 0.00 seconds

### Candidate
type: RSZ, layer: 1, pos: 30

### Relational analysis RSZ of RS_RSZ1_RSZ2_RSZ2_RSZ2_RSZ1
Optimization infeasible because this subproblem isn't reachable.

### Candidate
type: RSZ, layer: 1, pos: 197

### Relational analysis RSZ of RS_RSZ1_RSZ2_RSZ2_RSZ2_RSZ1

#### Relational analysis RSZ result of RS_RSZ1_RSZ2_RSZ2_RSZ2_RSZ1
Status: Status.UNKNOWN
Output dim: 4, lower bound: -0.0052372, upper bound: 0.0056076
time: 1.22 seconds

### Relational analysis RSZ of RS_RSZ1_RSZ2_RSZ2_RSZ2_RSZ2

#### Relational analysis RSZ result of RS_RSZ1_RSZ2_RSZ2_RSZ2_RSZ2
Status: Status.UNKNOWN
Output dim: 4, lower bound: -0.0052356, upper bound: 0.0056080
time: 1.49 seconds

## Summary of splitting (split count: 4)
- Time for RS candidates: 5.68 seconds
RS_RSZ1_RSZ2_RSZ1_RSZ1_RSZ1, status: Status.VERIFIED, split count: 5, time: 5.68
Output dim: 4, lower bound: -0.0055112, upper bound: 0.0052701
RS_RSZ1_RSZ2_RSZ1_RSZ1_RSZ2, status: Status.VERIFIED, split count: 5, time: 5.68
Output dim: 4, lower bound: -0.0055418, upper bound: 0.0052373
RS_RSZ1_RSZ2_RSZ2_RSZ1_RSZ1, status: Status.VERIFIED, split count: 5, time: 5.68
Output dim: 4, lower bound: -0.0052604, upper bound: 0.0055660
RS_RSZ1_RSZ2_RSZ2_RSZ1_RSZ2, status: Status.UNKNOWN, split count: 5, time: 5.68
Output dim: 4, lower bound: -0.0052535, upper bound: 0.0056376
RS_RSZ1_RSZ2_RSZ2_RSZ2_RSZ1, status: Status.UNKNOWN, split count: 5, time: 5.68
Output dim: 4, lower bound: -0.0052372, upper bound: 0.0056076
RS_RSZ1_RSZ2_RSZ2_RSZ2_RSZ2, status: Status.UNKNOWN, split count: 5, time: 5.68
Output dim: 4, lower bound: -0.0052356, upper bound: 0.0056080

## BFS RS instance: RS_RSZ1_RSZ2_RSZ2_RSZ1_RSZ2

### Backsubstitution after applying RS history:
0: -0.0044784, -0.0038492, -0.0044784, -0.0038492, -0.0005846, 0.0005725
1: 0.0010607, 0.0045450, 0.0010607, 0.0045450, -0.0031700, 0.0032370
2: 0.0048122, 0.0125964, 0.0048122, 0.0125964, -0.0072319, 0.0070821
3: 0.0020262, 0.0053065, 0.0020262, 0.0053065, -0.0029844, 0.0030475
4: 1.0046113, 1.0173376, 1.0046113, 1.0173376, -0.0115783, 0.0118232
5: 0.0031385, 0.0056142, 0.0031385, 0.0056142, -0.0022524, 0.0023001
6: -0.0130490, -0.0098272, -0.0130490, -0.0098272, -0.0029932, 0.0029312
7: -0.0104679, -0.0100569, -0.0104679, -0.0100569, -0.0003818, 0.0003739
8: -0.0040643, -0.0018383, -0.0040643, -0.0018383, -0.0020252, 0.0020681
9: -0.0089681, 0.0021759, -0.0089681, 0.0021759, -0.0103533, 0.0101389

### Unstable ReLU Count (Linear/Conv2D Layers)
- layer_idx=0, type=LayerType.Linear, total=256, inp1_unstable=15, inp2_unstable=15, delta_unstable=251
- layer_idx=2, type=LayerType.Linear, total=256, inp1_unstable=19, inp2_unstable=19, delta_unstable=256
- layer_idx=4, type=LayerType.Linear, total=256, inp1_unstable=0, inp2_unstable=0, delta_unstable=256
- layer_idx=6, type=LayerType.Linear, total=256, inp1_unstable=0, inp2_unstable=0, delta_unstable=256
- layer_idx=8, type=LayerType.Linear, total=10, inp1_unstable=1, inp2_unstable=1, delta_unstable=10

Time for backsubstitution: 1.06 seconds

### RS candidates at layer 1
type: RSZ, layer: 1, pos: 197
type: RSZ, layer: 1, pos: 76
type: RSZ, layer: 1, pos: 85
type: RSZ, layer: 1, pos: 204
type: RSZ, layer: 1, pos: 250
type: RSZ, layer: 1, pos: 238
type: RSZ, layer: 1, pos: 155
type: RSZ, layer: 1, pos: 182
type: RSZ, layer: 1, pos: 30
type: RSZ, layer: 1, pos: 179

Time for candidate selection: 0.00 seconds

### Candidate
type: RSZ, layer: 1, pos: 197

### Relational analysis RSZ of RS_RSZ1_RSZ2_RSZ2_RSZ1_RSZ2_RSZ1

#### Relational analysis RSZ result of RS_RSZ1_RSZ2_RSZ2_RSZ1_RSZ2_RSZ1
Status: Status.UNKNOWN
Output dim: 4, lower bound: -0.0052244, upper bound: 0.0056077
time: 1.32 seconds

### Relational analysis RSZ of RS_RSZ1_RSZ2_RSZ2_RSZ1_RSZ2_RSZ2

#### Relational analysis RSZ result of RS_RSZ1_RSZ2_RSZ2_RSZ1_RSZ2_RSZ2
Status: Status.UNKNOWN
Output dim: 4, lower bound: -0.0052220, upper bound: 0.0056079
time: 1.24 seconds

## BFS RS instance: RS_RSZ1_RSZ2_RSZ2_RSZ2_RSZ1

### Backsubstitution after applying RS history:
0: -0.0044784, -0.0038492, -0.0044784, -0.0038492, -0.0005918, 0.0005801
1: 0.0010607, 0.0045450, 0.0010607, 0.0045450, -0.0032122, 0.0032767
2: 0.0048122, 0.0125964, 0.0048122, 0.0125964, -0.0073204, 0.0071764
3: 0.0020262, 0.0053065, 0.0020262, 0.0053065, -0.0030242, 0.0030848
4: 1.0046113, 1.0173376, 1.0046113, 1.0173376, -0.0117326, 0.0119680
5: 0.0031385, 0.0056142, 0.0031385, 0.0056142, -0.0022824, 0.0023282
6: -0.0130490, -0.0098272, -0.0130490, -0.0098272, -0.0030299, 0.0029703
7: -0.0104679, -0.0100569, -0.0104679, -0.0100569, -0.0003865, 0.0003789
8: -0.0040643, -0.0018383, -0.0040643, -0.0018383, -0.0020522, 0.0020934
9: -0.0089681, 0.0021759, -0.0089681, 0.0021759, -0.0104801, 0.0102739

### Unstable ReLU Count (Linear/Conv2D Layers)
- layer_idx=0, type=LayerType.Linear, total=256, inp1_unstable=15, inp2_unstable=15, delta_unstable=251
- layer_idx=2, type=LayerType.Linear, total=256, inp1_unstable=19, inp2_unstable=19, delta_unstable=256
- layer_idx=4, type=LayerType.Linear, total=256, inp1_unstable=0, inp2_unstable=0, delta_unstable=256
- layer_idx=6, type=LayerType.Linear, total=256, inp1_unstable=0, inp2_unstable=0, delta_unstable=256
- layer_idx=8, type=LayerType.Linear, total=10, inp1_unstable=1, inp2_unstable=1, delta_unstable=10

Time for backsubstitution: 1.09 seconds

### RS candidates at layer 1
type: RSZ, layer: 1, pos: 179
type: RSZ, layer: 1, pos: 155
type: RSZ, layer: 1, pos: 250
type: RSZ, layer: 1, pos: 238
type: RSZ, layer: 1, pos: 85
type: RSZ, layer: 1, pos: 76
type: RSZ, layer: 1, pos: 132
type: RSZ, layer: 1, pos: 182
type: RSZ, layer: 1, pos: 30
type: RSZ, layer: 1, pos: 204

Time for candidate selection: 0.00 seconds

### Candidate
type: RSZ, layer: 1, pos: 179

### Relational analysis RSZ of RS_RSZ1_RSZ2_RSZ2_RSZ2_RSZ1_RSZ1

#### Relational analysis RSZ result of RS_RSZ1_RSZ2_RSZ2_RSZ2_RSZ1_RSZ1
Status: Status.VERIFIED
Output dim: 4, lower bound: -0.0052372, upper bound: 0.0055355
time: 1.15 seconds

### Relational analysis RSZ of RS_RSZ1_RSZ2_RSZ2_RSZ2_RSZ1_RSZ2

#### Relational analysis RSZ result of RS_RSZ1_RSZ2_RSZ2_RSZ2_RSZ1_RSZ2
Status: Status.UNKNOWN
Output dim: 4, lower bound: -0.0052021, upper bound: 0.0056076
time: 1.25 seconds

## BFS RS instance: RS_RSZ1_RSZ2_RSZ2_RSZ2_RSZ2

### Backsubstitution after applying RS history:
0: -0.0044784, -0.0038492, -0.0044784, -0.0038492, -0.0005904, 0.0005821
1: 0.0010607, 0.0045450, 0.0010607, 0.0045450, -0.0032233, 0.0032688
2: 0.0048122, 0.0125964, 0.0048122, 0.0125964, -0.0073028, 0.0072013
3: 0.0020262, 0.0053065, 0.0020262, 0.0053065, -0.0030346, 0.0030774
4: 1.0046113, 1.0173376, 1.0046113, 1.0173376, -0.0117733, 0.0119392
5: 0.0031385, 0.0056142, 0.0031385, 0.0056142, -0.0022904, 0.0023227
6: -0.0130490, -0.0098272, -0.0130490, -0.0098272, -0.0030226, 0.0029806
7: -0.0104679, -0.0100569, -0.0104679, -0.0100569, -0.0003856, 0.0003802
8: -0.0040643, -0.0018383, -0.0040643, -0.0018383, -0.0020593, 0.0020884
9: -0.0089681, 0.0021759, -0.0089681, 0.0021759, -0.0104549, 0.0103096

### Unstable ReLU Count (Linear/Conv2D Layers)
- layer_idx=0, type=LayerType.Linear, total=256, inp1_unstable=15, inp2_unstable=15, delta_unstable=251
- layer_idx=2, type=LayerType.Linear, total=256, inp1_unstable=19, inp2_unstable=19, delta_unstable=256
- layer_idx=4, type=LayerType.Linear, total=256, inp1_unstable=0, inp2_unstable=0, delta_unstable=256
- layer_idx=6, type=LayerType.Linear, total=256, inp1_unstable=0, inp2_unstable=0, delta_unstable=256
- layer_idx=8, type=LayerType.Linear, total=10, inp1_unstable=1, inp2_unstable=1, delta_unstable=10

Time for backsubstitution: 1.06 seconds

### RS candidates at layer 1
type: RSZ, layer: 1, pos: 238
type: RSZ, layer: 1, pos: 132
type: RSZ, layer: 1, pos: 250
type: RSZ, layer: 1, pos: 179
type: RSZ, layer: 1, pos: 155
type: RSZ, layer: 1, pos: 85
type: RSZ, layer: 1, pos: 76
type: RSZ, layer: 1, pos: 182
type: RSZ, layer: 1, pos: 30
type: RSZ, layer: 1, pos: 204

Time for candidate selection: 0.00 seconds

### Candidate
type: RSZ, layer: 1, pos: 238

### Relational analysis RSZ of RS_RSZ1_RSZ2_RSZ2_RSZ2_RSZ2_RSZ1

#### Relational analysis RSZ result of RS_RSZ1_RSZ2_RSZ2_RSZ2_RSZ2_RSZ1
Status: Status.VERIFIED
Output dim: 4, lower bound: -0.0051529, upper bound: 0.0055157
time: 1.18 seconds

### Relational analysis RSZ of RS_RSZ1_RSZ2_RSZ2_RSZ2_RSZ2_RSZ2

#### Relational analysis RSZ result of RS_RSZ1_RSZ2_RSZ2_RSZ2_RSZ2_RSZ2
Status: Status.VERIFIED
Output dim: 4, lower bound: -0.0051520, upper bound: 0.0055159
time: 1.61 seconds

## Summary of splitting (split count: 5)
- Time for RS candidates: 3.86 seconds
RS_RSZ1_RSZ2_RSZ2_RSZ1_RSZ2_RSZ1, status: Status.UNKNOWN, split count: 6, time: 3.86
Output dim: 4, lower bound: -0.0052244, upper bound: 0.0056077
RS_RSZ1_RSZ2_RSZ2_RSZ1_RSZ2_RSZ2, status: Status.UNKNOWN, split count: 6, time: 3.86
Output dim: 4, lower bound: -0.0052220, upper bound: 0.0056079
RS_RSZ1_RSZ2_RSZ2_RSZ2_RSZ1_RSZ1, status: Status.VERIFIED, split count: 6, time: 3.86
Output dim: 4, lower bound: -0.0052372, upper bound: 0.0055355
RS_RSZ1_RSZ2_RSZ2_RSZ2_RSZ1_RSZ2, status: Status.UNKNOWN, split count: 6, time: 3.86
Output dim: 4, lower bound: -0.0052021, upper bound: 0.0056076
RS_RSZ1_RSZ2_RSZ2_RSZ2_RSZ2_RSZ1, status: Status.VERIFIED, split count: 6, time: 3.86
Output dim: 4, lower bound: -0.0051529, upper bound: 0.0055157
RS_RSZ1_RSZ2_RSZ2_RSZ2_RSZ2_RSZ2, status: Status.VERIFIED, split count: 6, time: 3.86
Output dim: 4, lower bound: -0.0051520, upper bound: 0.0055159

## BFS RS instance: RS_RSZ1_RSZ2_RSZ2_RSZ1_RSZ2_RSZ1

### Backsubstitution after applying RS history:
0: -0.0044784, -0.0038492, -0.0044784, -0.0038492, -0.0005948, 0.0005811
1: 0.0010607, 0.0045450, 0.0010607, 0.0045450, -0.0032175, 0.0032937
2: 0.0048122, 0.0125964, 0.0048122, 0.0125964, -0.0073584, 0.0071882
3: 0.0020262, 0.0053065, 0.0020262, 0.0053065, -0.0030291, 0.0031008
4: 1.0046113, 1.0173376, 1.0046113, 1.0173376, -0.0117519, 0.0120301
5: 0.0031385, 0.0056142, 0.0031385, 0.0056142, -0.0022862, 0.0023403
6: -0.0130490, -0.0098272, -0.0130490, -0.0098272, -0.0030456, 0.0029752
7: -0.0104679, -0.0100569, -0.0104679, -0.0100569, -0.0003885, 0.0003795
8: -0.0040643, -0.0018383, -0.0040643, -0.0018383, -0.0020556, 0.0021043
9: -0.0089681, 0.0021759, -0.0089681, 0.0021759, -0.0105345, 0.0102909

### Unstable ReLU Count (Linear/Conv2D Layers)
- layer_idx=0, type=LayerType.Linear, total=256, inp1_unstable=15, inp2_unstable=15, delta_unstable=250
- layer_idx=2, type=LayerType.Linear, total=256, inp1_unstable=19, inp2_unstable=19, delta_unstable=256
- layer_idx=4, type=LayerType.Linear, total=256, inp1_unstable=0, inp2_unstable=0, delta_unstable=256
- layer_idx=6, type=LayerType.Linear, total=256, inp1_unstable=0, inp2_unstable=0, delta_unstable=256
- layer_idx=8, type=LayerType.Linear, total=10, inp1_unstable=1, inp2_unstable=1, delta_unstable=10

Time for backsubstitution: 1.06 seconds

### RS candidates at layer 1
type: RSZ, layer: 1, pos: 155
type: RSZ, layer: 1, pos: 76
type: RSZ, layer: 1, pos: 238
type: RSZ, layer: 1, pos: 179
type: RSZ, layer: 1, pos: 182
type: RSZ, layer: 1, pos: 30
type: RSZ, layer: 1, pos: 204
type: RSZ, layer: 1, pos: 250
type: RSZ, layer: 1, pos: 85

Time for candidate selection: 0.00 seconds

### Candidate
type: RSZ, layer: 1, pos: 155

### Relational analysis RSZ of RS_RSZ1_RSZ2_RSZ2_RSZ1_RSZ2_RSZ1_RSZ1

#### Relational analysis RSZ result of RS_RSZ1_RSZ2_RSZ2_RSZ1_RSZ2_RSZ1_RSZ1
Status: Status.VERIFIED
Output dim: 4, lower bound: -0.0050033, upper bound: 0.0053360
time: 1.51 seconds

### Relational analysis RSZ of RS_RSZ1_RSZ2_RSZ2_RSZ1_RSZ2_RSZ1_RSZ2

#### Relational analysis RSZ result of RS_RSZ1_RSZ2_RSZ2_RSZ1_RSZ2_RSZ1_RSZ2
Status: Status.VERIFIED
Output dim: 4, lower bound: -0.0050033, upper bound: 0.0053360
time: 1.25 seconds

## BFS RS instance: RS_RSZ1_RSZ2_RSZ2_RSZ1_RSZ2_RSZ2

### Backsubstitution after applying RS history:
0: -0.0044784, -0.0038492, -0.0044784, -0.0038492, -0.0005932, 0.0005825
1: 0.0010607, 0.0045450, 0.0010607, 0.0045450, -0.0032255, 0.0032845
2: 0.0048122, 0.0125964, 0.0048122, 0.0125964, -0.0073380, 0.0072061
3: 0.0020262, 0.0053065, 0.0020262, 0.0053065, -0.0030367, 0.0030923
4: 1.0046113, 1.0173376, 1.0046113, 1.0173376, -0.0117811, 0.0119968
5: 0.0031385, 0.0056142, 0.0031385, 0.0056142, -0.0022919, 0.0023339
6: -0.0130490, -0.0098272, -0.0130490, -0.0098272, -0.0030372, 0.0029826
7: -0.0104679, -0.0100569, -0.0104679, -0.0100569, -0.0003874, 0.0003805
8: -0.0040643, -0.0018383, -0.0040643, -0.0018383, -0.0020607, 0.0020984
9: -0.0089681, 0.0021759, -0.0089681, 0.0021759, -0.0105053, 0.0103164

### Unstable ReLU Count (Linear/Conv2D Layers)
- layer_idx=0, type=LayerType.Linear, total=256, inp1_unstable=15, inp2_unstable=15, delta_unstable=250
- layer_idx=2, type=LayerType.Linear, total=256, inp1_unstable=19, inp2_unstable=19, delta_unstable=256
- layer_idx=4, type=LayerType.Linear, total=256, inp1_unstable=0, inp2_unstable=0, delta_unstable=256
- layer_idx=6, type=LayerType.Linear, total=256, inp1_unstable=0, inp2_unstable=0, delta_unstable=256
- layer_idx=8, type=LayerType.Linear, total=10, inp1_unstable=1, inp2_unstable=1, delta_unstable=10

Time for backsubstitution: 1.07 seconds

### RS candidates at layer 1
type: RSZ, layer: 1, pos: 155
type: RSZ, layer: 1, pos: 30
type: RSZ, layer: 1, pos: 85
type: RSZ, layer: 1, pos: 204
type: RSZ, layer: 1, pos: 179
type: RSZ, layer: 1, pos: 238
type: RSZ, layer: 1, pos: 76
type: RSZ, layer: 1, pos: 182
type: RSZ, layer: 1, pos: 250

Time for candidate selection: 0.00 seconds

### Candidate
type: RSZ, layer: 1, pos: 155

### Relational analysis RSZ of RS_RSZ1_RSZ2_RSZ2_RSZ1_RSZ2_RSZ2_RSZ1

#### Relational analysis RSZ result of RS_RSZ1_RSZ2_RSZ2_RSZ1_RSZ2_RSZ2_RSZ1
Status: Status.VERIFIED
Output dim: 4, lower bound: -0.0049986, upper bound: 0.0053360
time: 1.57 seconds

### Relational analysis RSZ of RS_RSZ1_RSZ2_RSZ2_RSZ1_RSZ2_RSZ2_RSZ2

#### Relational analysis RSZ result of RS_RSZ1_RSZ2_RSZ2_RSZ1_RSZ2_RSZ2_RSZ2
Status: Status.VERIFIED
Output dim: 4, lower bound: -0.0049986, upper bound: 0.0053360
time: 1.58 seconds

## BFS RS instance: RS_RSZ1_RSZ2_RSZ2_RSZ2_RSZ1_RSZ2

### Backsubstitution after applying RS history:
0: -0.0044784, -0.0038492, -0.0044784, -0.0038492, -0.0005924, 0.0005804
1: 0.0010607, 0.0045450, 0.0010607, 0.0045450, -0.0032134, 0.0032803
2: 0.0048122, 0.0125964, 0.0048122, 0.0125964, -0.0073286, 0.0071791
3: 0.0020262, 0.0053065, 0.0020262, 0.0053065, -0.0030253, 0.0030883
4: 1.0046113, 1.0173376, 1.0046113, 1.0173376, -0.0117370, 0.0119814
5: 0.0031385, 0.0056142, 0.0031385, 0.0056142, -0.0022833, 0.0023308
6: -0.0130490, -0.0098272, -0.0130490, -0.0098272, -0.0030333, 0.0029714
7: -0.0104679, -0.0100569, -0.0104679, -0.0100569, -0.0003869, 0.0003790
8: -0.0040643, -0.0018383, -0.0040643, -0.0018383, -0.0020530, 0.0020957
9: -0.0089681, 0.0021759, -0.0089681, 0.0021759, -0.0104918, 0.0102778

### Unstable ReLU Count (Linear/Conv2D Layers)
- layer_idx=0, type=LayerType.Linear, total=256, inp1_unstable=15, inp2_unstable=15, delta_unstable=250
- layer_idx=2, type=LayerType.Linear, total=256, inp1_unstable=19, inp2_unstable=19, delta_unstable=256
- layer_idx=4, type=LayerType.Linear, total=256, inp1_unstable=0, inp2_unstable=0, delta_unstable=256
- layer_idx=6, type=LayerType.Linear, total=256, inp1_unstable=0, inp2_unstable=0, delta_unstable=256
- layer_idx=8, type=LayerType.Linear, total=10, inp1_unstable=1, inp2_unstable=1, delta_unstable=10

Time for backsubstitution: 1.04 seconds

### RS candidates at layer 1
type: RSZ, layer: 1, pos: 238
type: RSZ, layer: 1, pos: 30
type: RSZ, layer: 1, pos: 250
type: RSZ, layer: 1, pos: 76
type: RSZ, layer: 1, pos: 132
type: RSZ, layer: 1, pos: 204
type: RSZ, layer: 1, pos: 182
type: RSZ, layer: 1, pos: 155
type: RSZ, layer: 1, pos: 85

Time for candidate selection: 0.00 seconds

### Candidate
type: RSZ, layer: 1, pos: 238

### Relational analysis RSZ of RS_RSZ1_RSZ2_RSZ2_RSZ2_RSZ1_RSZ2_RSZ1

#### Relational analysis RSZ result of RS_RSZ1_RSZ2_RSZ2_RSZ2_RSZ1_RSZ2_RSZ1
Status: Status.VERIFIED
Output dim: 4, lower bound: -0.0051185, upper bound: 0.0055149
time: 1.63 seconds

### Relational analysis RSZ of RS_RSZ1_RSZ2_RSZ2_RSZ2_RSZ1_RSZ2_RSZ2

#### Relational analysis RSZ result of RS_RSZ1_RSZ2_RSZ2_RSZ2_RSZ1_RSZ2_RSZ2
Status: Status.VERIFIED
Output dim: 4, lower bound: -0.0051177, upper bound: 0.0055151
time: 1.66 seconds

## Summary of splitting (split count: 6)
- Time for RS candidates: 4.35 seconds
RS_RSZ1_RSZ2_RSZ2_RSZ1_RSZ2_RSZ1_RSZ1, status: Status.VERIFIED, split count: 7, time: 4.35
Output dim: 4, lower bound: -0.0050033, upper bound: 0.0053360
RS_RSZ1_RSZ2_RSZ2_RSZ1_RSZ2_RSZ1_RSZ2, status: Status.VERIFIED, split count: 7, time: 4.35
Output dim: 4, lower bound: -0.0050033, upper bound: 0.0053360
RS_RSZ1_RSZ2_RSZ2_RSZ1_RSZ2_RSZ2_RSZ1, status: Status.VERIFIED, split count: 7, time: 4.35
Output dim: 4, lower bound: -0.0049986, upper bound: 0.0053360
RS_RSZ1_RSZ2_RSZ2_RSZ1_RSZ2_RSZ2_RSZ2, status: Status.VERIFIED, split count: 7, time: 4.35
Output dim: 4, lower bound: -0.0049986, upper bound: 0.0053360
RS_RSZ1_RSZ2_RSZ2_RSZ2_RSZ1_RSZ2_RSZ1, status: Status.VERIFIED, split count: 7, time: 4.35
Output dim: 4, lower bound: -0.0051185, upper bound: 0.0055149
RS_RSZ1_RSZ2_RSZ2_RSZ2_RSZ1_RSZ2_RSZ2, status: Status.VERIFIED, split count: 7, time: 4.35
Output dim: 4, lower bound: -0.0051177, upper bound: 0.0055151

## RS Result
status: Status.VERIFIED
execution time: (base) + (rs) = 3.28 + 61.46 = 64.74 seconds
