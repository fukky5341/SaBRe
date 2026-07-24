## Execution arguments:
Dataset: Dataset.MNIST
Network: onnx/mnist-net_256x4.onnx
Relational property: GLOBAL_ROBUSTNESS
LP Analysis: True
Epsilon: 0.03515625
Delta epsilon: 0.01171875
execution index: (3, 3, 1)
Time budget: 600 seconds
Split limit: 100
Threshold: 0.00073336


## IAR start

### Unstable ReLU Count (Linear/Conv2D Layers)
- layer_idx=0, type=LayerType.Linear, total=256, inp1_unstable=6, inp2_unstable=6, delta_unstable=256
- layer_idx=2, type=LayerType.Linear, total=256, inp1_unstable=7, inp2_unstable=7, delta_unstable=256
- layer_idx=4, type=LayerType.Linear, total=256, inp1_unstable=0, inp2_unstable=0, delta_unstable=256
- layer_idx=6, type=LayerType.Linear, total=256, inp1_unstable=0, inp2_unstable=0, delta_unstable=256
- layer_idx=8, type=LayerType.Linear, total=10, inp1_unstable=0, inp2_unstable=0, delta_unstable=10

### BASE IAR bounds
Layer (inp1_lb, inp1_ub, inp2_lb, inp2_ub, d_lb, d_ub)
0: (0.0070497, 0.0081861, 0.0070497, 0.0081861, -0.0006016, 0.0006016)
1: (0.0023408, 0.0025050, 0.0023408, 0.0025050, -0.0000869, 0.0000869)
2: (0.0098340, 0.0104623, 0.0098340, 0.0104623, -0.0003326, 0.0003326)
3: (-0.0045097, -0.0038599, -0.0045097, -0.0038599, -0.0003440, 0.0003440)
4: (0.0001416, 0.0008451, 0.0001416, 0.0008451, -0.0003724, 0.0003724)
5: (0.0033136, 0.0039793, 0.0033136, 0.0039793, -0.0003524, 0.0003524)
6: (-0.0091528, -0.0065115, -0.0091528, -0.0065115, -0.0013982, 0.0013982)
7: (0.0063114, 0.0099086, 0.0063114, 0.0099086, -0.0019043, 0.0019043)
8: (0.9936597, 0.9961938, 0.9936597, 0.9961938, -0.0013414, 0.0013414)
9: (-0.0124322, -0.0101320, -0.0124322, -0.0101320, -0.0012177, 0.0012177)

## BASE Relational Analysis

## BASE Result
execution time: IAR + RelationalAnalysis = 1.76 + 1.35 = 3.11 seconds
status: Status.UNKNOWN
relational distance
Output dim: 8, lower bound: -0.0009772, upper bound: 0.0009773

# Relational Split (RS) starts

## BFS RS instance: RS

Time for backsubstitution: 0.01 seconds

### RS candidates at layer 1
type: RSZ, layer: 1, pos: 176
type: RSZ, layer: 1, pos: 77
type: RSZ, layer: 1, pos: 126
type: RSZ, layer: 1, pos: 44
type: RSZ, layer: 1, pos: 247
type: RSZ, layer: 1, pos: 219

Time for candidate selection: 0.00 seconds

### Candidate
type: RSZ, layer: 1, pos: 176

### Relational analysis RSZ of RS_RSZ1

#### Relational analysis RSZ result of RS_RSZ1
Status: Status.UNKNOWN
Output dim: 8, lower bound: -0.0009484, upper bound: 0.0009200
time: 0.45 seconds

### Relational analysis RSZ of RS_RSZ2

#### Relational analysis RSZ result of RS_RSZ2
Status: Status.UNKNOWN
Output dim: 8, lower bound: -0.0009200, upper bound: 0.0009484
time: 0.46 seconds

## Summary of splitting (split count: 0)
- Time for RS candidates: 0.92 seconds
RS_RSZ1, status: Status.UNKNOWN, split count: 1, time: 0.92
Output dim: 8, lower bound: -0.0009484, upper bound: 0.0009200
RS_RSZ2, status: Status.UNKNOWN, split count: 1, time: 0.92
Output dim: 8, lower bound: -0.0009200, upper bound: 0.0009484

## BFS RS instance: RS_RSZ1

### Backsubstitution after applying RS history:
0: 0.0070497, 0.0081861, 0.0070497, 0.0081861, -0.0005556, 0.0005523
1: 0.0023408, 0.0025050, 0.0023408, 0.0025050, -0.0000803, 0.0000798
2: 0.0098340, 0.0104623, 0.0098340, 0.0104623, -0.0003054, 0.0003072
3: -0.0045097, -0.0038599, -0.0045097, -0.0038599, -0.0003158, 0.0003177
4: 0.0001416, 0.0008451, 0.0001416, 0.0008451, -0.0003439, 0.0003419
5: 0.0033136, 0.0039793, 0.0033136, 0.0039793, -0.0003235, 0.0003255
6: -0.0091528, -0.0065115, -0.0091528, -0.0065115, -0.0012837, 0.0012913
7: 0.0063114, 0.0099086, 0.0063114, 0.0099086, -0.0017587, 0.0017483
8: 0.9936597, 0.9961938, 0.9936597, 0.9961938, -0.0012389, 0.0012316
9: -0.0124322, -0.0101320, -0.0124322, -0.0101320, -0.0011179, 0.0011246

### Unstable ReLU Count (Linear/Conv2D Layers)
- layer_idx=0, type=LayerType.Linear, total=256, inp1_unstable=6, inp2_unstable=6, delta_unstable=255
- layer_idx=2, type=LayerType.Linear, total=256, inp1_unstable=7, inp2_unstable=7, delta_unstable=256
- layer_idx=4, type=LayerType.Linear, total=256, inp1_unstable=0, inp2_unstable=0, delta_unstable=256
- layer_idx=6, type=LayerType.Linear, total=256, inp1_unstable=0, inp2_unstable=0, delta_unstable=256
- layer_idx=8, type=LayerType.Linear, total=10, inp1_unstable=0, inp2_unstable=0, delta_unstable=10

Time for backsubstitution: 1.63 seconds

### RS candidates at layer 1
type: RSZ, layer: 1, pos: 77
type: RSZ, layer: 1, pos: 219
type: RSZ, layer: 1, pos: 44
type: RSZ, layer: 1, pos: 247
type: RSZ, layer: 1, pos: 126

Time for candidate selection: 0.00 seconds

### Candidate
type: RSZ, layer: 1, pos: 77

### Relational analysis RSZ of RS_RSZ1_RSZ1

#### Relational analysis RSZ result of RS_RSZ1_RSZ1
Status: Status.UNKNOWN
Output dim: 8, lower bound: -0.0008149, upper bound: 0.0007971
time: 0.48 seconds

### Relational analysis RSZ of RS_RSZ1_RSZ2

#### Relational analysis RSZ result of RS_RSZ1_RSZ2
Status: Status.UNKNOWN
Output dim: 8, lower bound: -0.0008149, upper bound: 0.0007971
time: 0.49 seconds

## BFS RS instance: RS_RSZ2

### Backsubstitution after applying RS history:
0: 0.0070497, 0.0081861, 0.0070497, 0.0081861, -0.0005523, 0.0005556
1: 0.0023408, 0.0025050, 0.0023408, 0.0025050, -0.0000798, 0.0000803
2: 0.0098340, 0.0104623, 0.0098340, 0.0104623, -0.0003072, 0.0003054
3: -0.0045097, -0.0038599, -0.0045097, -0.0038599, -0.0003177, 0.0003158
4: 0.0001416, 0.0008451, 0.0001416, 0.0008451, -0.0003419, 0.0003439
5: 0.0033136, 0.0039793, 0.0033136, 0.0039793, -0.0003255, 0.0003235
6: -0.0091528, -0.0065115, -0.0091528, -0.0065115, -0.0012913, 0.0012837
7: 0.0063114, 0.0099086, 0.0063114, 0.0099086, -0.0017483, 0.0017587
8: 0.9936597, 0.9961938, 0.9936597, 0.9961938, -0.0012316, 0.0012389
9: -0.0124322, -0.0101320, -0.0124322, -0.0101320, -0.0011246, 0.0011179

### Unstable ReLU Count (Linear/Conv2D Layers)
- layer_idx=0, type=LayerType.Linear, total=256, inp1_unstable=6, inp2_unstable=6, delta_unstable=255
- layer_idx=2, type=LayerType.Linear, total=256, inp1_unstable=7, inp2_unstable=7, delta_unstable=256
- layer_idx=4, type=LayerType.Linear, total=256, inp1_unstable=0, inp2_unstable=0, delta_unstable=256
- layer_idx=6, type=LayerType.Linear, total=256, inp1_unstable=0, inp2_unstable=0, delta_unstable=256
- layer_idx=8, type=LayerType.Linear, total=10, inp1_unstable=0, inp2_unstable=0, delta_unstable=10

Time for backsubstitution: 1.65 seconds

### RS candidates at layer 1
type: RSZ, layer: 1, pos: 126
type: RSZ, layer: 1, pos: 44
type: RSZ, layer: 1, pos: 247
type: RSZ, layer: 1, pos: 219
type: RSZ, layer: 1, pos: 77

Time for candidate selection: 0.00 seconds

### Candidate
type: RSZ, layer: 1, pos: 126

### Relational analysis RSZ of RS_RSZ2_RSZ1

#### Relational analysis RSZ result of RS_RSZ2_RSZ1
Status: Status.UNKNOWN
Output dim: 8, lower bound: -0.0008814, upper bound: 0.0009091
time: 0.49 seconds

### Relational analysis RSZ of RS_RSZ2_RSZ2

#### Relational analysis RSZ result of RS_RSZ2_RSZ2
Status: Status.UNKNOWN
Output dim: 8, lower bound: -0.0008815, upper bound: 0.0009091
time: 0.49 seconds

## Summary of splitting (split count: 1)
- Time for RS candidates: 2.64 seconds
RS_RSZ1_RSZ1, status: Status.UNKNOWN, split count: 2, time: 2.64
Output dim: 8, lower bound: -0.0008149, upper bound: 0.0007971
RS_RSZ1_RSZ2, status: Status.UNKNOWN, split count: 2, time: 2.64
Output dim: 8, lower bound: -0.0008149, upper bound: 0.0007971
RS_RSZ2_RSZ1, status: Status.UNKNOWN, split count: 2, time: 2.64
Output dim: 8, lower bound: -0.0008814, upper bound: 0.0009091
RS_RSZ2_RSZ2, status: Status.UNKNOWN, split count: 2, time: 2.64
Output dim: 8, lower bound: -0.0008815, upper bound: 0.0009091

## BFS RS instance: RS_RSZ1_RSZ1

### Backsubstitution after applying RS history:
0: 0.0070497, 0.0081861, 0.0070497, 0.0081861, -0.0005535, 0.0005407
1: 0.0023408, 0.0025050, 0.0023408, 0.0025050, -0.0000800, 0.0000781
2: 0.0098340, 0.0104623, 0.0098340, 0.0104623, -0.0002990, 0.0003060
3: -0.0045097, -0.0038599, -0.0045097, -0.0038599, -0.0003092, 0.0003165
4: 0.0001416, 0.0008451, 0.0001416, 0.0008451, -0.0003426, 0.0003347
5: 0.0033136, 0.0039793, 0.0033136, 0.0039793, -0.0003168, 0.0003243
6: -0.0091528, -0.0065115, -0.0091528, -0.0065115, -0.0012568, 0.0012866
7: 0.0063114, 0.0099086, 0.0063114, 0.0099086, -0.0017522, 0.0017116
8: 0.9936597, 0.9961938, 0.9936597, 0.9961938, -0.0012343, 0.0012057
9: -0.0124322, -0.0101320, -0.0124322, -0.0101320, -0.0010945, 0.0011204

### Unstable ReLU Count (Linear/Conv2D Layers)
- layer_idx=0, type=LayerType.Linear, total=256, inp1_unstable=6, inp2_unstable=6, delta_unstable=254
- layer_idx=2, type=LayerType.Linear, total=256, inp1_unstable=7, inp2_unstable=7, delta_unstable=256
- layer_idx=4, type=LayerType.Linear, total=256, inp1_unstable=0, inp2_unstable=0, delta_unstable=256
- layer_idx=6, type=LayerType.Linear, total=256, inp1_unstable=0, inp2_unstable=0, delta_unstable=256
- layer_idx=8, type=LayerType.Linear, total=10, inp1_unstable=0, inp2_unstable=0, delta_unstable=10

Time for backsubstitution: 1.57 seconds

### RS candidates at layer 1
type: RSZ, layer: 1, pos: 44
type: RSZ, layer: 1, pos: 219
type: RSZ, layer: 1, pos: 126
type: RSZ, layer: 1, pos: 247

Time for candidate selection: 0.00 seconds

### Candidate
type: RSZ, layer: 1, pos: 44

### Relational analysis RSZ of RS_RSZ1_RSZ1_RSZ1
Optimization infeasible because this subproblem isn't reachable.

### Candidate
type: RSZ, layer: 1, pos: 219

### Relational analysis RSZ of RS_RSZ1_RSZ1_RSZ1

#### Relational analysis RSZ result of RS_RSZ1_RSZ1_RSZ1
Status: Status.UNKNOWN
Output dim: 8, lower bound: -0.0008146, upper bound: 0.0007907
time: 0.49 seconds

### Relational analysis RSZ of RS_RSZ1_RSZ1_RSZ2

#### Relational analysis RSZ result of RS_RSZ1_RSZ1_RSZ2
Status: Status.UNKNOWN
Output dim: 8, lower bound: -0.0008053, upper bound: 0.0007967
time: 0.50 seconds

## BFS RS instance: RS_RSZ1_RSZ2

### Backsubstitution after applying RS history:
0: 0.0070497, 0.0081861, 0.0070497, 0.0081861, -0.0005440, 0.0005523
1: 0.0023408, 0.0025050, 0.0023408, 0.0025050, -0.0000786, 0.0000798
2: 0.0098340, 0.0104623, 0.0098340, 0.0104623, -0.0003054, 0.0003008
3: -0.0045097, -0.0038599, -0.0045097, -0.0038599, -0.0003158, 0.0003111
4: 0.0001416, 0.0008451, 0.0001416, 0.0008451, -0.0003367, 0.0003419
5: 0.0033136, 0.0039793, 0.0033136, 0.0039793, -0.0003235, 0.0003187
6: -0.0091528, -0.0065115, -0.0091528, -0.0065115, -0.0012837, 0.0012644
7: 0.0063114, 0.0099086, 0.0063114, 0.0099086, -0.0017220, 0.0017483
8: 0.9936597, 0.9961938, 0.9936597, 0.9961938, -0.0012130, 0.0012316
9: -0.0124322, -0.0101320, -0.0124322, -0.0101320, -0.0011179, 0.0011011

### Unstable ReLU Count (Linear/Conv2D Layers)
- layer_idx=0, type=LayerType.Linear, total=256, inp1_unstable=6, inp2_unstable=6, delta_unstable=254
- layer_idx=2, type=LayerType.Linear, total=256, inp1_unstable=7, inp2_unstable=7, delta_unstable=256
- layer_idx=4, type=LayerType.Linear, total=256, inp1_unstable=0, inp2_unstable=0, delta_unstable=256
- layer_idx=6, type=LayerType.Linear, total=256, inp1_unstable=0, inp2_unstable=0, delta_unstable=256
- layer_idx=8, type=LayerType.Linear, total=10, inp1_unstable=0, inp2_unstable=0, delta_unstable=10

Time for backsubstitution: 1.62 seconds

### RS candidates at layer 1
type: RSZ, layer: 1, pos: 247
type: RSZ, layer: 1, pos: 44
type: RSZ, layer: 1, pos: 126
type: RSZ, layer: 1, pos: 219

Time for candidate selection: 0.00 seconds

### Candidate
type: RSZ, layer: 1, pos: 247

### Relational analysis RSZ of RS_RSZ1_RSZ2_RSZ1
Optimization infeasible because this subproblem isn't reachable.

### Candidate
type: RSZ, layer: 1, pos: 44

### Relational analysis RSZ of RS_RSZ1_RSZ2_RSZ1
Optimization infeasible because this subproblem isn't reachable.

### Candidate
type: RSZ, layer: 1, pos: 126

### Relational analysis RSZ of RS_RSZ1_RSZ2_RSZ1

#### Relational analysis RSZ result of RS_RSZ1_RSZ2_RSZ1
Status: Status.VERIFIED
Output dim: 8, lower bound: -0.0006374, upper bound: 0.0006244
time: 0.47 seconds

### Relational analysis RSZ of RS_RSZ1_RSZ2_RSZ2

#### Relational analysis RSZ result of RS_RSZ1_RSZ2_RSZ2
Status: Status.VERIFIED
Output dim: 8, lower bound: -0.0006374, upper bound: 0.0006244
time: 0.49 seconds

## BFS RS instance: RS_RSZ2_RSZ1

### Backsubstitution after applying RS history:
0: 0.0070497, 0.0081861, 0.0070497, 0.0081861, -0.0005465, 0.0005497
1: 0.0023408, 0.0025050, 0.0023408, 0.0025050, -0.0000790, 0.0000794
2: 0.0098340, 0.0104623, 0.0098340, 0.0104623, -0.0003039, 0.0003022
3: -0.0045097, -0.0038599, -0.0045097, -0.0038599, -0.0003143, 0.0003125
4: 0.0001416, 0.0008451, 0.0001416, 0.0008451, -0.0003383, 0.0003403
5: 0.0033136, 0.0039793, 0.0033136, 0.0039793, -0.0003220, 0.0003202
6: -0.0091528, -0.0065115, -0.0091528, -0.0065115, -0.0012776, 0.0012703
7: 0.0063114, 0.0099086, 0.0063114, 0.0099086, -0.0017300, 0.0017400
8: 0.9936597, 0.9961938, 0.9936597, 0.9961938, -0.0012187, 0.0012257
9: -0.0124322, -0.0101320, -0.0124322, -0.0101320, -0.0011126, 0.0011062

### Unstable ReLU Count (Linear/Conv2D Layers)
- layer_idx=0, type=LayerType.Linear, total=256, inp1_unstable=6, inp2_unstable=6, delta_unstable=254
- layer_idx=2, type=LayerType.Linear, total=256, inp1_unstable=7, inp2_unstable=7, delta_unstable=256
- layer_idx=4, type=LayerType.Linear, total=256, inp1_unstable=0, inp2_unstable=0, delta_unstable=256
- layer_idx=6, type=LayerType.Linear, total=256, inp1_unstable=0, inp2_unstable=0, delta_unstable=256
- layer_idx=8, type=LayerType.Linear, total=10, inp1_unstable=0, inp2_unstable=0, delta_unstable=10

Time for backsubstitution: 1.63 seconds

### RS candidates at layer 1
type: RSZ, layer: 1, pos: 44
type: RSZ, layer: 1, pos: 247
type: RSZ, layer: 1, pos: 219
type: RSZ, layer: 1, pos: 77

Time for candidate selection: 0.00 seconds

### Candidate
type: RSZ, layer: 1, pos: 44

### Relational analysis RSZ of RS_RSZ2_RSZ1_RSZ1

#### Relational analysis RSZ result of RS_RSZ2_RSZ1_RSZ1
Status: Status.VERIFIED
Output dim: 8, lower bound: -0.0005630, upper bound: 0.0005850
time: 0.47 seconds

### Relational analysis RSZ of RS_RSZ2_RSZ1_RSZ2

#### Relational analysis RSZ result of RS_RSZ2_RSZ1_RSZ2
Status: Status.VERIFIED
Output dim: 8, lower bound: -0.0005630, upper bound: 0.0005850
time: 0.47 seconds

## BFS RS instance: RS_RSZ2_RSZ2

### Backsubstitution after applying RS history:
0: 0.0070497, 0.0081861, 0.0070497, 0.0081861, -0.0005523, 0.0005498
1: 0.0023408, 0.0025050, 0.0023408, 0.0025050, -0.0000798, 0.0000794
2: 0.0098340, 0.0104623, 0.0098340, 0.0104623, -0.0003040, 0.0003054
3: -0.0045097, -0.0038599, -0.0045097, -0.0038599, -0.0003144, 0.0003158
4: 0.0001416, 0.0008451, 0.0001416, 0.0008451, -0.0003419, 0.0003403
5: 0.0033136, 0.0039793, 0.0033136, 0.0039793, -0.0003221, 0.0003235
6: -0.0091528, -0.0065115, -0.0091528, -0.0065115, -0.0012779, 0.0012837
7: 0.0063114, 0.0099086, 0.0063114, 0.0099086, -0.0017483, 0.0017404
8: 0.9936597, 0.9961938, 0.9936597, 0.9961938, -0.0012316, 0.0012260
9: -0.0124322, -0.0101320, -0.0124322, -0.0101320, -0.0011129, 0.0011179

### Unstable ReLU Count (Linear/Conv2D Layers)
- layer_idx=0, type=LayerType.Linear, total=256, inp1_unstable=6, inp2_unstable=6, delta_unstable=254
- layer_idx=2, type=LayerType.Linear, total=256, inp1_unstable=7, inp2_unstable=7, delta_unstable=256
- layer_idx=4, type=LayerType.Linear, total=256, inp1_unstable=0, inp2_unstable=0, delta_unstable=256
- layer_idx=6, type=LayerType.Linear, total=256, inp1_unstable=0, inp2_unstable=0, delta_unstable=256
- layer_idx=8, type=LayerType.Linear, total=10, inp1_unstable=0, inp2_unstable=0, delta_unstable=10

Time for backsubstitution: 1.69 seconds

### RS candidates at layer 1
type: RSZ, layer: 1, pos: 44
type: RSZ, layer: 1, pos: 247
type: RSZ, layer: 1, pos: 219
type: RSZ, layer: 1, pos: 77

Time for candidate selection: 0.00 seconds

### Candidate
type: RSZ, layer: 1, pos: 44

### Relational analysis RSZ of RS_RSZ2_RSZ2_RSZ1

#### Relational analysis RSZ result of RS_RSZ2_RSZ2_RSZ1
Status: Status.VERIFIED
Output dim: 8, lower bound: -0.0005630, upper bound: 0.0005850
time: 0.47 seconds

### Relational analysis RSZ of RS_RSZ2_RSZ2_RSZ2

#### Relational analysis RSZ result of RS_RSZ2_RSZ2_RSZ2
Status: Status.VERIFIED
Output dim: 8, lower bound: -0.0005630, upper bound: 0.0005850
time: 0.47 seconds

## Summary of splitting (split count: 2)
- Time for RS candidates: 2.64 seconds
RS_RSZ1_RSZ1_RSZ1, status: Status.UNKNOWN, split count: 3, time: 2.64
Output dim: 8, lower bound: -0.0008146, upper bound: 0.0007907
RS_RSZ1_RSZ1_RSZ2, status: Status.UNKNOWN, split count: 3, time: 2.64
Output dim: 8, lower bound: -0.0008053, upper bound: 0.0007967
RS_RSZ1_RSZ2_RSZ1, status: Status.VERIFIED, split count: 3, time: 2.64
Output dim: 8, lower bound: -0.0006374, upper bound: 0.0006244
RS_RSZ1_RSZ2_RSZ2, status: Status.VERIFIED, split count: 3, time: 2.64
Output dim: 8, lower bound: -0.0006374, upper bound: 0.0006244
RS_RSZ2_RSZ1_RSZ1, status: Status.VERIFIED, split count: 3, time: 2.64
Output dim: 8, lower bound: -0.0005630, upper bound: 0.0005850
RS_RSZ2_RSZ1_RSZ2, status: Status.VERIFIED, split count: 3, time: 2.64
Output dim: 8, lower bound: -0.0005630, upper bound: 0.0005850
RS_RSZ2_RSZ2_RSZ1, status: Status.VERIFIED, split count: 3, time: 2.64
Output dim: 8, lower bound: -0.0005630, upper bound: 0.0005850
RS_RSZ2_RSZ2_RSZ2, status: Status.VERIFIED, split count: 3, time: 2.64
Output dim: 8, lower bound: -0.0005630, upper bound: 0.0005850

## BFS RS instance: RS_RSZ1_RSZ1_RSZ1

### Backsubstitution after applying RS history:
0: 0.0070497, 0.0081861, 0.0070497, 0.0081861, -0.0005507, 0.0005371
1: 0.0023408, 0.0025050, 0.0023408, 0.0025050, -0.0000796, 0.0000776
2: 0.0098340, 0.0104623, 0.0098340, 0.0104623, -0.0002969, 0.0003045
3: -0.0045097, -0.0038599, -0.0045097, -0.0038599, -0.0003071, 0.0003149
4: 0.0001416, 0.0008451, 0.0001416, 0.0008451, -0.0003409, 0.0003324
5: 0.0033136, 0.0039793, 0.0033136, 0.0039793, -0.0003146, 0.0003226
6: -0.0091528, -0.0065115, -0.0091528, -0.0065115, -0.0012483, 0.0012801
7: 0.0063114, 0.0099086, 0.0063114, 0.0099086, -0.0017434, 0.0017000
8: 0.9936597, 0.9961938, 0.9936597, 0.9961938, -0.0012281, 0.0011975
9: -0.0124322, -0.0101320, -0.0124322, -0.0101320, -0.0010870, 0.0011148

### Unstable ReLU Count (Linear/Conv2D Layers)
- layer_idx=0, type=LayerType.Linear, total=256, inp1_unstable=6, inp2_unstable=6, delta_unstable=253
- layer_idx=2, type=LayerType.Linear, total=256, inp1_unstable=7, inp2_unstable=7, delta_unstable=256
- layer_idx=4, type=LayerType.Linear, total=256, inp1_unstable=0, inp2_unstable=0, delta_unstable=256
- layer_idx=6, type=LayerType.Linear, total=256, inp1_unstable=0, inp2_unstable=0, delta_unstable=256
- layer_idx=8, type=LayerType.Linear, total=10, inp1_unstable=0, inp2_unstable=0, delta_unstable=10

Time for backsubstitution: 1.73 seconds

### RS candidates at layer 1
type: RSZ, layer: 1, pos: 44
type: RSZ, layer: 1, pos: 126
type: RSZ, layer: 1, pos: 247

Time for candidate selection: 0.00 seconds

### Candidate
type: RSZ, layer: 1, pos: 44

### Relational analysis RSZ of RS_RSZ1_RSZ1_RSZ1_RSZ1
Optimization infeasible because this subproblem isn't reachable.

### Candidate
type: RSZ, layer: 1, pos: 126

### Relational analysis RSZ of RS_RSZ1_RSZ1_RSZ1_RSZ1

#### Relational analysis RSZ result of RS_RSZ1_RSZ1_RSZ1_RSZ1
Status: Status.VERIFIED
Output dim: 8, lower bound: -0.0006370, upper bound: 0.0006213
time: 0.48 seconds

### Relational analysis RSZ of RS_RSZ1_RSZ1_RSZ1_RSZ2

#### Relational analysis RSZ result of RS_RSZ1_RSZ1_RSZ1_RSZ2
Status: Status.VERIFIED
Output dim: 8, lower bound: -0.0006370, upper bound: 0.0006213
time: 0.49 seconds

## BFS RS instance: RS_RSZ1_RSZ1_RSZ2

### Backsubstitution after applying RS history:
0: 0.0070497, 0.0081861, 0.0070497, 0.0081861, -0.0005499, 0.0005380
1: 0.0023408, 0.0025050, 0.0023408, 0.0025050, -0.0000794, 0.0000777
2: 0.0098340, 0.0104623, 0.0098340, 0.0104623, -0.0002974, 0.0003040
3: -0.0045097, -0.0038599, -0.0045097, -0.0038599, -0.0003076, 0.0003144
4: 0.0001416, 0.0008451, 0.0001416, 0.0008451, -0.0003404, 0.0003330
5: 0.0033136, 0.0039793, 0.0033136, 0.0039793, -0.0003151, 0.0003221
6: -0.0091528, -0.0065115, -0.0091528, -0.0065115, -0.0012504, 0.0012781
7: 0.0063114, 0.0099086, 0.0063114, 0.0099086, -0.0017406, 0.0017029
8: 0.9936597, 0.9961938, 0.9936597, 0.9961938, -0.0012261, 0.0011996
9: -0.0124322, -0.0101320, -0.0124322, -0.0101320, -0.0010889, 0.0011130

### Unstable ReLU Count (Linear/Conv2D Layers)
- layer_idx=0, type=LayerType.Linear, total=256, inp1_unstable=6, inp2_unstable=6, delta_unstable=253
- layer_idx=2, type=LayerType.Linear, total=256, inp1_unstable=7, inp2_unstable=7, delta_unstable=256
- layer_idx=4, type=LayerType.Linear, total=256, inp1_unstable=0, inp2_unstable=0, delta_unstable=256
- layer_idx=6, type=LayerType.Linear, total=256, inp1_unstable=0, inp2_unstable=0, delta_unstable=256
- layer_idx=8, type=LayerType.Linear, total=10, inp1_unstable=0, inp2_unstable=0, delta_unstable=10

Time for backsubstitution: 1.57 seconds

### RS candidates at layer 1
type: RSZ, layer: 1, pos: 247
type: RSZ, layer: 1, pos: 126
type: RSZ, layer: 1, pos: 44

Time for candidate selection: 0.00 seconds

### Candidate
type: RSZ, layer: 1, pos: 247

### Relational analysis RSZ of RS_RSZ1_RSZ1_RSZ2_RSZ1
Optimization infeasible because this subproblem isn't reachable.

### Candidate
type: RSZ, layer: 1, pos: 126

### Relational analysis RSZ of RS_RSZ1_RSZ1_RSZ2_RSZ1

#### Relational analysis RSZ result of RS_RSZ1_RSZ1_RSZ2_RSZ1
Status: Status.VERIFIED
Output dim: 8, lower bound: -0.0006369, upper bound: 0.0006240
time: 0.47 seconds

### Relational analysis RSZ of RS_RSZ1_RSZ1_RSZ2_RSZ2

#### Relational analysis RSZ result of RS_RSZ1_RSZ1_RSZ2_RSZ2
Status: Status.VERIFIED
Output dim: 8, lower bound: -0.0006369, upper bound: 0.0006240
time: 0.47 seconds

## Summary of splitting (split count: 3)
- Time for RS candidates: 4.41 seconds
RS_RSZ1_RSZ1_RSZ1_RSZ1, status: Status.VERIFIED, split count: 4, time: 4.41
Output dim: 8, lower bound: -0.0006370, upper bound: 0.0006213
RS_RSZ1_RSZ1_RSZ1_RSZ2, status: Status.VERIFIED, split count: 4, time: 4.41
Output dim: 8, lower bound: -0.0006370, upper bound: 0.0006213
RS_RSZ1_RSZ1_RSZ2_RSZ1, status: Status.VERIFIED, split count: 4, time: 4.41
Output dim: 8, lower bound: -0.0006369, upper bound: 0.0006240
RS_RSZ1_RSZ1_RSZ2_RSZ2, status: Status.VERIFIED, split count: 4, time: 4.41
Output dim: 8, lower bound: -0.0006369, upper bound: 0.0006240

## RS Result
status: Status.VERIFIED
execution time: (base) + (rs) = 3.11 + 31.19 = 34.29 seconds
