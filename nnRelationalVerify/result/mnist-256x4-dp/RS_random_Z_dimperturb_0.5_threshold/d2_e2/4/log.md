## Execution arguments:
Dataset: Dataset.MNIST
Network: onnx/mnist-net_256x4.onnx
Relational property: GLOBAL_ROBUSTNESS
LP Analysis: True
Epsilon: 0.015625
Delta epsilon: 0.0078125
execution index: (2, 2, 4)
Time budget: 600 seconds
Split limit: 100
Threshold: 0.0061821


## IAR start

### Unstable ReLU Count (Linear/Conv2D Layers)
- layer_idx=0, type=LayerType.Linear, total=256, inp1_unstable=10, inp2_unstable=10, delta_unstable=256
- layer_idx=2, type=LayerType.Linear, total=256, inp1_unstable=6, inp2_unstable=6, delta_unstable=256
- layer_idx=4, type=LayerType.Linear, total=256, inp1_unstable=0, inp2_unstable=0, delta_unstable=256
- layer_idx=6, type=LayerType.Linear, total=256, inp1_unstable=1, inp2_unstable=1, delta_unstable=256
- layer_idx=8, type=LayerType.Linear, total=10, inp1_unstable=0, inp2_unstable=0, delta_unstable=10

### BASE IAR bounds
Layer (inp1_lb, inp1_ub, inp2_lb, inp2_ub, d_lb, d_ub)
0: (0.9815361, 0.9898027, 0.9815361, 0.9898027, -0.0082666, 0.0082666)
1: (-0.0045208, -0.0038048, -0.0045208, -0.0038048, -0.0007160, 0.0007160)
2: (0.0101096, 0.0139039, 0.0101096, 0.0139039, -0.0037943, 0.0037943)
3: (-0.0076910, -0.0058746, -0.0076910, -0.0058746, -0.0018164, 0.0018164)
4: (0.0024846, 0.0036056, 0.0024846, 0.0036056, -0.0011210, 0.0011210)
5: (0.0116747, 0.0204531, 0.0116747, 0.0204531, -0.0087784, 0.0087784)
6: (-0.0026336, -0.0014223, -0.0026336, -0.0014223, -0.0012112, 0.0012112)
7: (-0.0099515, -0.0068176, -0.0099515, -0.0068176, -0.0031339, 0.0031339)
8: (-0.0047975, -0.0027613, -0.0047975, -0.0027613, -0.0020363, 0.0020363)
9: (0.0017881, 0.0036991, 0.0017881, 0.0036991, -0.0019110, 0.0019110)

## BASE Relational Analysis

## BASE Result
execution time: IAR + RelationalAnalysis = 1.22 + 1.96 = 3.19 seconds
status: Status.UNKNOWN
relational distance
Output dim: 0, lower bound: -0.0067132, upper bound: 0.0067132

# Relational Split (RS) starts

## BFS RS instance: RS

Time for backsubstitution: 0.01 seconds

### RS candidates at layer 1
type: RSZ, layer: 1, pos: 85
type: RSZ, layer: 1, pos: 201
type: RSZ, layer: 1, pos: 2
type: RSZ, layer: 1, pos: 165
type: RSZ, layer: 1, pos: 128
type: RSZ, layer: 1, pos: 29
type: RSZ, layer: 1, pos: 69
type: RSZ, layer: 1, pos: 155
type: RSZ, layer: 1, pos: 235
type: RSZ, layer: 1, pos: 120

Time for candidate selection: 0.00 seconds

### Candidate
type: RSZ, layer: 1, pos: 85

### Relational analysis RSZ of RS_RSZ1

#### Relational analysis RSZ result of RS_RSZ1
Status: Status.UNKNOWN
Output dim: 0, lower bound: -0.0065990, upper bound: 0.0065990
time: 0.94 seconds

### Relational analysis RSZ of RS_RSZ2

#### Relational analysis RSZ result of RS_RSZ2
Status: Status.UNKNOWN
Output dim: 0, lower bound: -0.0065990, upper bound: 0.0065990
time: 0.89 seconds

## Summary of splitting (split count: 0)
- Time for RS candidates: 1.84 seconds
RS_RSZ1, status: Status.UNKNOWN, split count: 1, time: 1.84
Output dim: 0, lower bound: -0.0065990, upper bound: 0.0065990
RS_RSZ2, status: Status.UNKNOWN, split count: 1, time: 1.84
Output dim: 0, lower bound: -0.0065990, upper bound: 0.0065990

## BFS RS instance: RS_RSZ1

### Backsubstitution after applying RS history:
0: 0.9815361, 0.9898027, 0.9815361, 0.9898027, -0.0082666, 0.0082666
1: -0.0045208, -0.0038048, -0.0045208, -0.0038048, -0.0007160, 0.0007160
2: 0.0101096, 0.0139039, 0.0101096, 0.0139039, -0.0037943, 0.0037943
3: -0.0076910, -0.0058746, -0.0076910, -0.0058746, -0.0018164, 0.0018164
4: 0.0024846, 0.0036056, 0.0024846, 0.0036056, -0.0011210, 0.0011210
5: 0.0116747, 0.0204531, 0.0116747, 0.0204531, -0.0087784, 0.0087784
6: -0.0026336, -0.0014223, -0.0026336, -0.0014223, -0.0012112, 0.0012112
7: -0.0099515, -0.0068176, -0.0099515, -0.0068176, -0.0031339, 0.0031339
8: -0.0047975, -0.0027613, -0.0047975, -0.0027613, -0.0020363, 0.0020363
9: 0.0017881, 0.0036991, 0.0017881, 0.0036991, -0.0019110, 0.0019110

### Unstable ReLU Count (Linear/Conv2D Layers)
- layer_idx=0, type=LayerType.Linear, total=256, inp1_unstable=10, inp2_unstable=10, delta_unstable=255
- layer_idx=2, type=LayerType.Linear, total=256, inp1_unstable=6, inp2_unstable=6, delta_unstable=256
- layer_idx=4, type=LayerType.Linear, total=256, inp1_unstable=0, inp2_unstable=0, delta_unstable=256
- layer_idx=6, type=LayerType.Linear, total=256, inp1_unstable=1, inp2_unstable=1, delta_unstable=256
- layer_idx=8, type=LayerType.Linear, total=10, inp1_unstable=0, inp2_unstable=0, delta_unstable=10

Time for backsubstitution: 1.06 seconds

### RS candidates at layer 1
type: RSZ, layer: 1, pos: 155
type: RSZ, layer: 1, pos: 128
type: RSZ, layer: 1, pos: 29
type: RSZ, layer: 1, pos: 120
type: RSZ, layer: 1, pos: 2
type: RSZ, layer: 1, pos: 235
type: RSZ, layer: 1, pos: 201
type: RSZ, layer: 1, pos: 165
type: RSZ, layer: 1, pos: 69

Time for candidate selection: 0.00 seconds

### Candidate
type: RSZ, layer: 1, pos: 155

### Relational analysis RSZ of RS_RSZ1_RSZ1

#### Relational analysis RSZ result of RS_RSZ1_RSZ1
Status: Status.UNKNOWN
Output dim: 0, lower bound: -0.0063552, upper bound: 0.0063538
time: 1.03 seconds

### Relational analysis RSZ of RS_RSZ1_RSZ2

#### Relational analysis RSZ result of RS_RSZ1_RSZ2
Status: Status.UNKNOWN
Output dim: 0, lower bound: -0.0063538, upper bound: 0.0063552
time: 0.82 seconds

## BFS RS instance: RS_RSZ2

### Backsubstitution after applying RS history:
0: 0.9815361, 0.9898027, 0.9815361, 0.9898027, -0.0082666, 0.0082666
1: -0.0045208, -0.0038048, -0.0045208, -0.0038048, -0.0007160, 0.0007160
2: 0.0101096, 0.0139039, 0.0101096, 0.0139039, -0.0037943, 0.0037943
3: -0.0076910, -0.0058746, -0.0076910, -0.0058746, -0.0018164, 0.0018164
4: 0.0024846, 0.0036056, 0.0024846, 0.0036056, -0.0011210, 0.0011210
5: 0.0116747, 0.0204531, 0.0116747, 0.0204531, -0.0087784, 0.0087784
6: -0.0026336, -0.0014223, -0.0026336, -0.0014223, -0.0012112, 0.0012112
7: -0.0099515, -0.0068176, -0.0099515, -0.0068176, -0.0031339, 0.0031339
8: -0.0047975, -0.0027613, -0.0047975, -0.0027613, -0.0020363, 0.0020363
9: 0.0017881, 0.0036991, 0.0017881, 0.0036991, -0.0019110, 0.0019110

### Unstable ReLU Count (Linear/Conv2D Layers)
- layer_idx=0, type=LayerType.Linear, total=256, inp1_unstable=10, inp2_unstable=10, delta_unstable=255
- layer_idx=2, type=LayerType.Linear, total=256, inp1_unstable=6, inp2_unstable=6, delta_unstable=256
- layer_idx=4, type=LayerType.Linear, total=256, inp1_unstable=0, inp2_unstable=0, delta_unstable=256
- layer_idx=6, type=LayerType.Linear, total=256, inp1_unstable=1, inp2_unstable=1, delta_unstable=256
- layer_idx=8, type=LayerType.Linear, total=10, inp1_unstable=0, inp2_unstable=0, delta_unstable=10

Time for backsubstitution: 1.04 seconds

### RS candidates at layer 1
type: RSZ, layer: 1, pos: 29
type: RSZ, layer: 1, pos: 165
type: RSZ, layer: 1, pos: 155
type: RSZ, layer: 1, pos: 201
type: RSZ, layer: 1, pos: 235
type: RSZ, layer: 1, pos: 128
type: RSZ, layer: 1, pos: 120
type: RSZ, layer: 1, pos: 69
type: RSZ, layer: 1, pos: 2

Time for candidate selection: 0.00 seconds

### Candidate
type: RSZ, layer: 1, pos: 29

### Relational analysis RSZ of RS_RSZ2_RSZ1

#### Relational analysis RSZ result of RS_RSZ2_RSZ1
Status: Status.UNKNOWN
Output dim: 0, lower bound: -0.0065194, upper bound: 0.0064140
time: 0.96 seconds

### Relational analysis RSZ of RS_RSZ2_RSZ2

#### Relational analysis RSZ result of RS_RSZ2_RSZ2
Status: Status.UNKNOWN
Output dim: 0, lower bound: -0.0064140, upper bound: 0.0065194
time: 1.07 seconds

## Summary of splitting (split count: 1)
- Time for RS candidates: 3.09 seconds
RS_RSZ1_RSZ1, status: Status.UNKNOWN, split count: 2, time: 3.09
Output dim: 0, lower bound: -0.0063552, upper bound: 0.0063538
RS_RSZ1_RSZ2, status: Status.UNKNOWN, split count: 2, time: 3.09
Output dim: 0, lower bound: -0.0063538, upper bound: 0.0063552
RS_RSZ2_RSZ1, status: Status.UNKNOWN, split count: 2, time: 3.09
Output dim: 0, lower bound: -0.0065194, upper bound: 0.0064140
RS_RSZ2_RSZ2, status: Status.UNKNOWN, split count: 2, time: 3.09
Output dim: 0, lower bound: -0.0064140, upper bound: 0.0065194

## BFS RS instance: RS_RSZ1_RSZ1

### Backsubstitution after applying RS history:
0: 0.9815361, 0.9898027, 0.9815361, 0.9898027, -0.0082666, 0.0082666
1: -0.0045208, -0.0038048, -0.0045208, -0.0038048, -0.0007160, 0.0007160
2: 0.0101096, 0.0139039, 0.0101096, 0.0139039, -0.0037943, 0.0037943
3: -0.0076910, -0.0058746, -0.0076910, -0.0058746, -0.0018164, 0.0018164
4: 0.0024846, 0.0036056, 0.0024846, 0.0036056, -0.0011210, 0.0011210
5: 0.0116747, 0.0204531, 0.0116747, 0.0204531, -0.0087784, 0.0087784
6: -0.0026336, -0.0014223, -0.0026336, -0.0014223, -0.0012112, 0.0012112
7: -0.0099515, -0.0068176, -0.0099515, -0.0068176, -0.0031339, 0.0031339
8: -0.0047975, -0.0027613, -0.0047975, -0.0027613, -0.0020363, 0.0020363
9: 0.0017881, 0.0036991, 0.0017881, 0.0036991, -0.0019110, 0.0019110

### Unstable ReLU Count (Linear/Conv2D Layers)
- layer_idx=0, type=LayerType.Linear, total=256, inp1_unstable=10, inp2_unstable=10, delta_unstable=254
- layer_idx=2, type=LayerType.Linear, total=256, inp1_unstable=6, inp2_unstable=6, delta_unstable=256
- layer_idx=4, type=LayerType.Linear, total=256, inp1_unstable=0, inp2_unstable=0, delta_unstable=256
- layer_idx=6, type=LayerType.Linear, total=256, inp1_unstable=1, inp2_unstable=1, delta_unstable=256
- layer_idx=8, type=LayerType.Linear, total=10, inp1_unstable=0, inp2_unstable=0, delta_unstable=10

Time for backsubstitution: 1.04 seconds

### RS candidates at layer 1
type: RSZ, layer: 1, pos: 2
type: RSZ, layer: 1, pos: 165
type: RSZ, layer: 1, pos: 235
type: RSZ, layer: 1, pos: 29
type: RSZ, layer: 1, pos: 120
type: RSZ, layer: 1, pos: 201
type: RSZ, layer: 1, pos: 69
type: RSZ, layer: 1, pos: 128

Time for candidate selection: 0.00 seconds

### Candidate
type: RSZ, layer: 1, pos: 2

### Relational analysis RSZ of RS_RSZ1_RSZ1_RSZ1

#### Relational analysis RSZ result of RS_RSZ1_RSZ1_RSZ1
Status: Status.UNKNOWN
Output dim: 0, lower bound: -0.0063524, upper bound: 0.0063509
time: 1.03 seconds

### Relational analysis RSZ of RS_RSZ1_RSZ1_RSZ2

#### Relational analysis RSZ result of RS_RSZ1_RSZ1_RSZ2
Status: Status.UNKNOWN
Output dim: 0, lower bound: -0.0063522, upper bound: 0.0063510
time: 1.02 seconds

## BFS RS instance: RS_RSZ1_RSZ2

### Backsubstitution after applying RS history:
0: 0.9815361, 0.9898027, 0.9815361, 0.9898027, -0.0082666, 0.0082666
1: -0.0045208, -0.0038048, -0.0045208, -0.0038048, -0.0007160, 0.0007160
2: 0.0101096, 0.0139039, 0.0101096, 0.0139039, -0.0037943, 0.0037943
3: -0.0076910, -0.0058746, -0.0076910, -0.0058746, -0.0018164, 0.0018164
4: 0.0024846, 0.0036056, 0.0024846, 0.0036056, -0.0011210, 0.0011210
5: 0.0116747, 0.0204531, 0.0116747, 0.0204531, -0.0087784, 0.0087784
6: -0.0026336, -0.0014223, -0.0026336, -0.0014223, -0.0012112, 0.0012112
7: -0.0099515, -0.0068176, -0.0099515, -0.0068176, -0.0031339, 0.0031339
8: -0.0047975, -0.0027613, -0.0047975, -0.0027613, -0.0020363, 0.0020363
9: 0.0017881, 0.0036991, 0.0017881, 0.0036991, -0.0019110, 0.0019110

### Unstable ReLU Count (Linear/Conv2D Layers)
- layer_idx=0, type=LayerType.Linear, total=256, inp1_unstable=10, inp2_unstable=10, delta_unstable=254
- layer_idx=2, type=LayerType.Linear, total=256, inp1_unstable=6, inp2_unstable=6, delta_unstable=256
- layer_idx=4, type=LayerType.Linear, total=256, inp1_unstable=0, inp2_unstable=0, delta_unstable=256
- layer_idx=6, type=LayerType.Linear, total=256, inp1_unstable=1, inp2_unstable=1, delta_unstable=256
- layer_idx=8, type=LayerType.Linear, total=10, inp1_unstable=0, inp2_unstable=0, delta_unstable=10

Time for backsubstitution: 1.05 seconds

### RS candidates at layer 1
type: RSZ, layer: 1, pos: 201
type: RSZ, layer: 1, pos: 29
type: RSZ, layer: 1, pos: 128
type: RSZ, layer: 1, pos: 69
type: RSZ, layer: 1, pos: 120
type: RSZ, layer: 1, pos: 165
type: RSZ, layer: 1, pos: 2
type: RSZ, layer: 1, pos: 235

Time for candidate selection: 0.00 seconds

### Candidate
type: RSZ, layer: 1, pos: 201

### Relational analysis RSZ of RS_RSZ1_RSZ2_RSZ1

#### Relational analysis RSZ result of RS_RSZ1_RSZ2_RSZ1
Status: Status.VERIFIED
Output dim: 0, lower bound: -0.0050466, upper bound: 0.0050468
time: 0.66 seconds

### Relational analysis RSZ of RS_RSZ1_RSZ2_RSZ2

#### Relational analysis RSZ result of RS_RSZ1_RSZ2_RSZ2
Status: Status.VERIFIED
Output dim: 0, lower bound: -0.0050466, upper bound: 0.0050468
time: 0.66 seconds

## BFS RS instance: RS_RSZ2_RSZ1

### Backsubstitution after applying RS history:
0: 0.9815361, 0.9898027, 0.9815361, 0.9898027, -0.0082666, 0.0082666
1: -0.0045208, -0.0038048, -0.0045208, -0.0038048, -0.0007160, 0.0007160
2: 0.0101096, 0.0139039, 0.0101096, 0.0139039, -0.0037943, 0.0037943
3: -0.0076910, -0.0058746, -0.0076910, -0.0058746, -0.0018164, 0.0018164
4: 0.0024846, 0.0036056, 0.0024846, 0.0036056, -0.0011210, 0.0011210
5: 0.0116747, 0.0204531, 0.0116747, 0.0204531, -0.0087784, 0.0087784
6: -0.0026336, -0.0014223, -0.0026336, -0.0014223, -0.0012112, 0.0012112
7: -0.0099515, -0.0068176, -0.0099515, -0.0068176, -0.0031339, 0.0031339
8: -0.0047975, -0.0027613, -0.0047975, -0.0027613, -0.0020363, 0.0020363
9: 0.0017881, 0.0036991, 0.0017881, 0.0036991, -0.0019110, 0.0019110

### Unstable ReLU Count (Linear/Conv2D Layers)
- layer_idx=0, type=LayerType.Linear, total=256, inp1_unstable=10, inp2_unstable=10, delta_unstable=254
- layer_idx=2, type=LayerType.Linear, total=256, inp1_unstable=6, inp2_unstable=6, delta_unstable=256
- layer_idx=4, type=LayerType.Linear, total=256, inp1_unstable=0, inp2_unstable=0, delta_unstable=256
- layer_idx=6, type=LayerType.Linear, total=256, inp1_unstable=1, inp2_unstable=1, delta_unstable=256
- layer_idx=8, type=LayerType.Linear, total=10, inp1_unstable=0, inp2_unstable=0, delta_unstable=10

Time for backsubstitution: 1.10 seconds

### RS candidates at layer 1
type: RSZ, layer: 1, pos: 155
type: RSZ, layer: 1, pos: 69
type: RSZ, layer: 1, pos: 128
type: RSZ, layer: 1, pos: 235
type: RSZ, layer: 1, pos: 165
type: RSZ, layer: 1, pos: 120
type: RSZ, layer: 1, pos: 201
type: RSZ, layer: 1, pos: 2

Time for candidate selection: 0.00 seconds

### Candidate
type: RSZ, layer: 1, pos: 155

### Relational analysis RSZ of RS_RSZ2_RSZ1_RSZ1

#### Relational analysis RSZ result of RS_RSZ2_RSZ1_RSZ1
Status: Status.UNKNOWN
Output dim: 0, lower bound: -0.0062754, upper bound: 0.0061745
time: 1.13 seconds

### Relational analysis RSZ of RS_RSZ2_RSZ1_RSZ2

#### Relational analysis RSZ result of RS_RSZ2_RSZ1_RSZ2
Status: Status.UNKNOWN
Output dim: 0, lower bound: -0.0062742, upper bound: 0.0061746
time: 1.03 seconds

## BFS RS instance: RS_RSZ2_RSZ2

### Backsubstitution after applying RS history:
0: 0.9815361, 0.9898027, 0.9815361, 0.9898027, -0.0082666, 0.0082666
1: -0.0045208, -0.0038048, -0.0045208, -0.0038048, -0.0007160, 0.0007160
2: 0.0101096, 0.0139039, 0.0101096, 0.0139039, -0.0037943, 0.0037943
3: -0.0076910, -0.0058746, -0.0076910, -0.0058746, -0.0018164, 0.0018164
4: 0.0024846, 0.0036056, 0.0024846, 0.0036056, -0.0011210, 0.0011210
5: 0.0116747, 0.0204531, 0.0116747, 0.0204531, -0.0087784, 0.0087784
6: -0.0026336, -0.0014223, -0.0026336, -0.0014223, -0.0012112, 0.0012112
7: -0.0099515, -0.0068176, -0.0099515, -0.0068176, -0.0031339, 0.0031339
8: -0.0047975, -0.0027613, -0.0047975, -0.0027613, -0.0020363, 0.0020363
9: 0.0017881, 0.0036991, 0.0017881, 0.0036991, -0.0019110, 0.0019110

### Unstable ReLU Count (Linear/Conv2D Layers)
- layer_idx=0, type=LayerType.Linear, total=256, inp1_unstable=10, inp2_unstable=10, delta_unstable=254
- layer_idx=2, type=LayerType.Linear, total=256, inp1_unstable=6, inp2_unstable=6, delta_unstable=256
- layer_idx=4, type=LayerType.Linear, total=256, inp1_unstable=0, inp2_unstable=0, delta_unstable=256
- layer_idx=6, type=LayerType.Linear, total=256, inp1_unstable=1, inp2_unstable=1, delta_unstable=256
- layer_idx=8, type=LayerType.Linear, total=10, inp1_unstable=0, inp2_unstable=0, delta_unstable=10

Time for backsubstitution: 1.07 seconds

### RS candidates at layer 1
type: RSZ, layer: 1, pos: 128
type: RSZ, layer: 1, pos: 155
type: RSZ, layer: 1, pos: 201
type: RSZ, layer: 1, pos: 69
type: RSZ, layer: 1, pos: 165
type: RSZ, layer: 1, pos: 120
type: RSZ, layer: 1, pos: 2
type: RSZ, layer: 1, pos: 235

Time for candidate selection: 0.00 seconds

### Candidate
type: RSZ, layer: 1, pos: 128

### Relational analysis RSZ of RS_RSZ2_RSZ2_RSZ1

#### Relational analysis RSZ result of RS_RSZ2_RSZ2_RSZ1
Status: Status.UNKNOWN
Output dim: 0, lower bound: -0.0062084, upper bound: 0.0063084
time: 0.79 seconds

### Relational analysis RSZ of RS_RSZ2_RSZ2_RSZ2

#### Relational analysis RSZ result of RS_RSZ2_RSZ2_RSZ2
Status: Status.UNKNOWN
Output dim: 0, lower bound: -0.0062041, upper bound: 0.0063113
time: 0.96 seconds

## Summary of splitting (split count: 2)
- Time for RS candidates: 2.83 seconds
RS_RSZ1_RSZ1_RSZ1, status: Status.UNKNOWN, split count: 3, time: 2.83
Output dim: 0, lower bound: -0.0063524, upper bound: 0.0063509
RS_RSZ1_RSZ1_RSZ2, status: Status.UNKNOWN, split count: 3, time: 2.83
Output dim: 0, lower bound: -0.0063522, upper bound: 0.0063510
RS_RSZ1_RSZ2_RSZ1, status: Status.VERIFIED, split count: 3, time: 2.83
Output dim: 0, lower bound: -0.0050466, upper bound: 0.0050468
RS_RSZ1_RSZ2_RSZ2, status: Status.VERIFIED, split count: 3, time: 2.83
Output dim: 0, lower bound: -0.0050466, upper bound: 0.0050468
RS_RSZ2_RSZ1_RSZ1, status: Status.UNKNOWN, split count: 3, time: 2.83
Output dim: 0, lower bound: -0.0062754, upper bound: 0.0061745
RS_RSZ2_RSZ1_RSZ2, status: Status.UNKNOWN, split count: 3, time: 2.83
Output dim: 0, lower bound: -0.0062742, upper bound: 0.0061746
RS_RSZ2_RSZ2_RSZ1, status: Status.UNKNOWN, split count: 3, time: 2.83
Output dim: 0, lower bound: -0.0062084, upper bound: 0.0063084
RS_RSZ2_RSZ2_RSZ2, status: Status.UNKNOWN, split count: 3, time: 2.83
Output dim: 0, lower bound: -0.0062041, upper bound: 0.0063113

## BFS RS instance: RS_RSZ1_RSZ1_RSZ1

### Backsubstitution after applying RS history:
0: 0.9815361, 0.9898027, 0.9815361, 0.9898027, -0.0082666, 0.0082666
1: -0.0045208, -0.0038048, -0.0045208, -0.0038048, -0.0007160, 0.0007160
2: 0.0101096, 0.0139039, 0.0101096, 0.0139039, -0.0037943, 0.0037943
3: -0.0076910, -0.0058746, -0.0076910, -0.0058746, -0.0018164, 0.0018164
4: 0.0024846, 0.0036056, 0.0024846, 0.0036056, -0.0011210, 0.0011210
5: 0.0116747, 0.0204531, 0.0116747, 0.0204531, -0.0087784, 0.0087784
6: -0.0026336, -0.0014223, -0.0026336, -0.0014223, -0.0012112, 0.0012112
7: -0.0099515, -0.0068176, -0.0099515, -0.0068176, -0.0031339, 0.0031339
8: -0.0047975, -0.0027613, -0.0047975, -0.0027613, -0.0020363, 0.0020363
9: 0.0017881, 0.0036991, 0.0017881, 0.0036991, -0.0019110, 0.0019110

### Unstable ReLU Count (Linear/Conv2D Layers)
- layer_idx=0, type=LayerType.Linear, total=256, inp1_unstable=10, inp2_unstable=10, delta_unstable=253
- layer_idx=2, type=LayerType.Linear, total=256, inp1_unstable=6, inp2_unstable=6, delta_unstable=256
- layer_idx=4, type=LayerType.Linear, total=256, inp1_unstable=0, inp2_unstable=0, delta_unstable=256
- layer_idx=6, type=LayerType.Linear, total=256, inp1_unstable=1, inp2_unstable=1, delta_unstable=256
- layer_idx=8, type=LayerType.Linear, total=10, inp1_unstable=0, inp2_unstable=0, delta_unstable=10

Time for backsubstitution: 1.06 seconds

### RS candidates at layer 1
type: RSZ, layer: 1, pos: 165
type: RSZ, layer: 1, pos: 29
type: RSZ, layer: 1, pos: 69
type: RSZ, layer: 1, pos: 201
type: RSZ, layer: 1, pos: 128
type: RSZ, layer: 1, pos: 235
type: RSZ, layer: 1, pos: 120

Time for candidate selection: 0.00 seconds

### Candidate
type: RSZ, layer: 1, pos: 165

### Relational analysis RSZ of RS_RSZ1_RSZ1_RSZ1_RSZ1

#### Relational analysis RSZ result of RS_RSZ1_RSZ1_RSZ1_RSZ1
Status: Status.VERIFIED
Output dim: 0, lower bound: -0.0057085, upper bound: 0.0057058
time: 0.89 seconds

### Relational analysis RSZ of RS_RSZ1_RSZ1_RSZ1_RSZ2

#### Relational analysis RSZ result of RS_RSZ1_RSZ1_RSZ1_RSZ2
Status: Status.VERIFIED
Output dim: 0, lower bound: -0.0057085, upper bound: 0.0057058
time: 0.91 seconds

## BFS RS instance: RS_RSZ1_RSZ1_RSZ2

### Backsubstitution after applying RS history:
0: 0.9815361, 0.9898027, 0.9815361, 0.9898027, -0.0082666, 0.0082666
1: -0.0045208, -0.0038048, -0.0045208, -0.0038048, -0.0007160, 0.0007160
2: 0.0101096, 0.0139039, 0.0101096, 0.0139039, -0.0037943, 0.0037943
3: -0.0076910, -0.0058746, -0.0076910, -0.0058746, -0.0018164, 0.0018164
4: 0.0024846, 0.0036056, 0.0024846, 0.0036056, -0.0011210, 0.0011210
5: 0.0116747, 0.0204531, 0.0116747, 0.0204531, -0.0087784, 0.0087784
6: -0.0026336, -0.0014223, -0.0026336, -0.0014223, -0.0012112, 0.0012112
7: -0.0099515, -0.0068176, -0.0099515, -0.0068176, -0.0031339, 0.0031339
8: -0.0047975, -0.0027613, -0.0047975, -0.0027613, -0.0020363, 0.0020363
9: 0.0017881, 0.0036991, 0.0017881, 0.0036991, -0.0019110, 0.0019110

### Unstable ReLU Count (Linear/Conv2D Layers)
- layer_idx=0, type=LayerType.Linear, total=256, inp1_unstable=10, inp2_unstable=10, delta_unstable=253
- layer_idx=2, type=LayerType.Linear, total=256, inp1_unstable=6, inp2_unstable=6, delta_unstable=256
- layer_idx=4, type=LayerType.Linear, total=256, inp1_unstable=0, inp2_unstable=0, delta_unstable=256
- layer_idx=6, type=LayerType.Linear, total=256, inp1_unstable=1, inp2_unstable=1, delta_unstable=256
- layer_idx=8, type=LayerType.Linear, total=10, inp1_unstable=0, inp2_unstable=0, delta_unstable=10

Time for backsubstitution: 1.07 seconds

### RS candidates at layer 1
type: RSZ, layer: 1, pos: 201
type: RSZ, layer: 1, pos: 29
type: RSZ, layer: 1, pos: 235
type: RSZ, layer: 1, pos: 165
type: RSZ, layer: 1, pos: 69
type: RSZ, layer: 1, pos: 120
type: RSZ, layer: 1, pos: 128

Time for candidate selection: 0.00 seconds

### Candidate
type: RSZ, layer: 1, pos: 201

### Relational analysis RSZ of RS_RSZ1_RSZ1_RSZ2_RSZ1

#### Relational analysis RSZ result of RS_RSZ1_RSZ1_RSZ2_RSZ1
Status: Status.VERIFIED
Output dim: 0, lower bound: -0.0050396, upper bound: 0.0050439
time: 0.68 seconds

### Relational analysis RSZ of RS_RSZ1_RSZ1_RSZ2_RSZ2

#### Relational analysis RSZ result of RS_RSZ1_RSZ1_RSZ2_RSZ2
Status: Status.VERIFIED
Output dim: 0, lower bound: -0.0050396, upper bound: 0.0050439
time: 0.68 seconds

## BFS RS instance: RS_RSZ2_RSZ1_RSZ1

### Backsubstitution after applying RS history:
0: 0.9815361, 0.9898027, 0.9815361, 0.9898027, -0.0082666, 0.0082666
1: -0.0045208, -0.0038048, -0.0045208, -0.0038048, -0.0007160, 0.0007160
2: 0.0101096, 0.0139039, 0.0101096, 0.0139039, -0.0037943, 0.0037943
3: -0.0076910, -0.0058746, -0.0076910, -0.0058746, -0.0018164, 0.0018164
4: 0.0024846, 0.0036056, 0.0024846, 0.0036056, -0.0011210, 0.0011210
5: 0.0116747, 0.0204531, 0.0116747, 0.0204531, -0.0087784, 0.0087784
6: -0.0026336, -0.0014223, -0.0026336, -0.0014223, -0.0012112, 0.0012112
7: -0.0099515, -0.0068176, -0.0099515, -0.0068176, -0.0031339, 0.0031339
8: -0.0047975, -0.0027613, -0.0047975, -0.0027613, -0.0020363, 0.0020363
9: 0.0017881, 0.0036991, 0.0017881, 0.0036991, -0.0019110, 0.0019110

### Unstable ReLU Count (Linear/Conv2D Layers)
- layer_idx=0, type=LayerType.Linear, total=256, inp1_unstable=10, inp2_unstable=10, delta_unstable=253
- layer_idx=2, type=LayerType.Linear, total=256, inp1_unstable=6, inp2_unstable=6, delta_unstable=256
- layer_idx=4, type=LayerType.Linear, total=256, inp1_unstable=0, inp2_unstable=0, delta_unstable=256
- layer_idx=6, type=LayerType.Linear, total=256, inp1_unstable=1, inp2_unstable=1, delta_unstable=256
- layer_idx=8, type=LayerType.Linear, total=10, inp1_unstable=0, inp2_unstable=0, delta_unstable=10

Time for backsubstitution: 1.04 seconds

### RS candidates at layer 1
type: RSZ, layer: 1, pos: 235
type: RSZ, layer: 1, pos: 201
type: RSZ, layer: 1, pos: 120
type: RSZ, layer: 1, pos: 69
type: RSZ, layer: 1, pos: 165
type: RSZ, layer: 1, pos: 2
type: RSZ, layer: 1, pos: 128

Time for candidate selection: 0.00 seconds

### Candidate
type: RSZ, layer: 1, pos: 235

### Relational analysis RSZ of RS_RSZ2_RSZ1_RSZ1_RSZ1

#### Relational analysis RSZ result of RS_RSZ2_RSZ1_RSZ1_RSZ1
Status: Status.VERIFIED
Output dim: 0, lower bound: -0.0057157, upper bound: 0.0056107
time: 0.83 seconds

### Relational analysis RSZ of RS_RSZ2_RSZ1_RSZ1_RSZ2

#### Relational analysis RSZ result of RS_RSZ2_RSZ1_RSZ1_RSZ2
Status: Status.VERIFIED
Output dim: 0, lower bound: -0.0057157, upper bound: 0.0056107
time: 0.85 seconds

## BFS RS instance: RS_RSZ2_RSZ1_RSZ2

### Backsubstitution after applying RS history:
0: 0.9815361, 0.9898027, 0.9815361, 0.9898027, -0.0082666, 0.0082666
1: -0.0045208, -0.0038048, -0.0045208, -0.0038048, -0.0007160, 0.0007160
2: 0.0101096, 0.0139039, 0.0101096, 0.0139039, -0.0037943, 0.0037943
3: -0.0076910, -0.0058746, -0.0076910, -0.0058746, -0.0018164, 0.0018164
4: 0.0024846, 0.0036056, 0.0024846, 0.0036056, -0.0011210, 0.0011210
5: 0.0116747, 0.0204531, 0.0116747, 0.0204531, -0.0087784, 0.0087784
6: -0.0026336, -0.0014223, -0.0026336, -0.0014223, -0.0012112, 0.0012112
7: -0.0099515, -0.0068176, -0.0099515, -0.0068176, -0.0031339, 0.0031339
8: -0.0047975, -0.0027613, -0.0047975, -0.0027613, -0.0020363, 0.0020363
9: 0.0017881, 0.0036991, 0.0017881, 0.0036991, -0.0019110, 0.0019110

### Unstable ReLU Count (Linear/Conv2D Layers)
- layer_idx=0, type=LayerType.Linear, total=256, inp1_unstable=10, inp2_unstable=10, delta_unstable=253
- layer_idx=2, type=LayerType.Linear, total=256, inp1_unstable=6, inp2_unstable=6, delta_unstable=256
- layer_idx=4, type=LayerType.Linear, total=256, inp1_unstable=0, inp2_unstable=0, delta_unstable=256
- layer_idx=6, type=LayerType.Linear, total=256, inp1_unstable=1, inp2_unstable=1, delta_unstable=256
- layer_idx=8, type=LayerType.Linear, total=10, inp1_unstable=0, inp2_unstable=0, delta_unstable=10

Time for backsubstitution: 1.15 seconds

### RS candidates at layer 1
type: RSZ, layer: 1, pos: 2
type: RSZ, layer: 1, pos: 201
type: RSZ, layer: 1, pos: 235
type: RSZ, layer: 1, pos: 128
type: RSZ, layer: 1, pos: 120
type: RSZ, layer: 1, pos: 165
type: RSZ, layer: 1, pos: 69

Time for candidate selection: 0.00 seconds

### Candidate
type: RSZ, layer: 1, pos: 2

### Relational analysis RSZ of RS_RSZ2_RSZ1_RSZ2_RSZ1

#### Relational analysis RSZ result of RS_RSZ2_RSZ1_RSZ2_RSZ1
Status: Status.UNKNOWN
Output dim: 0, lower bound: -0.0062714, upper bound: 0.0061703
time: 0.92 seconds

### Relational analysis RSZ of RS_RSZ2_RSZ1_RSZ2_RSZ2

#### Relational analysis RSZ result of RS_RSZ2_RSZ1_RSZ2_RSZ2
Status: Status.UNKNOWN
Output dim: 0, lower bound: -0.0062713, upper bound: 0.0061718
time: 0.90 seconds

## BFS RS instance: RS_RSZ2_RSZ2_RSZ1

### Backsubstitution after applying RS history:
0: 0.9815361, 0.9898027, 0.9815361, 0.9898027, -0.0082666, 0.0082666
1: -0.0045208, -0.0038048, -0.0045208, -0.0038048, -0.0007160, 0.0007160
2: 0.0101096, 0.0139039, 0.0101096, 0.0139039, -0.0037943, 0.0037943
3: -0.0076910, -0.0058746, -0.0076910, -0.0058746, -0.0018164, 0.0018164
4: 0.0024846, 0.0036056, 0.0024846, 0.0036056, -0.0011210, 0.0011210
5: 0.0116747, 0.0204531, 0.0116747, 0.0204531, -0.0087784, 0.0087784
6: -0.0026336, -0.0014223, -0.0026336, -0.0014223, -0.0012112, 0.0012112
7: -0.0099515, -0.0068176, -0.0099515, -0.0068176, -0.0031339, 0.0031339
8: -0.0047975, -0.0027613, -0.0047975, -0.0027613, -0.0020363, 0.0020363
9: 0.0017881, 0.0036991, 0.0017881, 0.0036991, -0.0019110, 0.0019110

### Unstable ReLU Count (Linear/Conv2D Layers)
- layer_idx=0, type=LayerType.Linear, total=256, inp1_unstable=10, inp2_unstable=10, delta_unstable=253
- layer_idx=2, type=LayerType.Linear, total=256, inp1_unstable=6, inp2_unstable=6, delta_unstable=256
- layer_idx=4, type=LayerType.Linear, total=256, inp1_unstable=0, inp2_unstable=0, delta_unstable=256
- layer_idx=6, type=LayerType.Linear, total=256, inp1_unstable=1, inp2_unstable=1, delta_unstable=256
- layer_idx=8, type=LayerType.Linear, total=10, inp1_unstable=0, inp2_unstable=0, delta_unstable=10

Time for backsubstitution: 1.05 seconds

### RS candidates at layer 1
type: RSZ, layer: 1, pos: 201
type: RSZ, layer: 1, pos: 69
type: RSZ, layer: 1, pos: 165
type: RSZ, layer: 1, pos: 235
type: RSZ, layer: 1, pos: 155
type: RSZ, layer: 1, pos: 2
type: RSZ, layer: 1, pos: 120

Time for candidate selection: 0.00 seconds

### Candidate
type: RSZ, layer: 1, pos: 201

### Relational analysis RSZ of RS_RSZ2_RSZ2_RSZ1_RSZ1

#### Relational analysis RSZ result of RS_RSZ2_RSZ2_RSZ1_RSZ1
Status: Status.VERIFIED
Output dim: 0, lower bound: -0.0041876, upper bound: 0.0042244
time: 0.67 seconds

### Relational analysis RSZ of RS_RSZ2_RSZ2_RSZ1_RSZ2

#### Relational analysis RSZ result of RS_RSZ2_RSZ2_RSZ1_RSZ2
Status: Status.VERIFIED
Output dim: 0, lower bound: -0.0041876, upper bound: 0.0042244
time: 0.66 seconds

## BFS RS instance: RS_RSZ2_RSZ2_RSZ2

### Backsubstitution after applying RS history:
0: 0.9815361, 0.9898027, 0.9815361, 0.9898027, -0.0082666, 0.0082666
1: -0.0045208, -0.0038048, -0.0045208, -0.0038048, -0.0007160, 0.0007160
2: 0.0101096, 0.0139039, 0.0101096, 0.0139039, -0.0037943, 0.0037943
3: -0.0076910, -0.0058746, -0.0076910, -0.0058746, -0.0018164, 0.0018164
4: 0.0024846, 0.0036056, 0.0024846, 0.0036056, -0.0011210, 0.0011210
5: 0.0116747, 0.0204531, 0.0116747, 0.0204531, -0.0087784, 0.0087784
6: -0.0026336, -0.0014223, -0.0026336, -0.0014223, -0.0012112, 0.0012112
7: -0.0099515, -0.0068176, -0.0099515, -0.0068176, -0.0031339, 0.0031339
8: -0.0047975, -0.0027613, -0.0047975, -0.0027613, -0.0020363, 0.0020363
9: 0.0017881, 0.0036991, 0.0017881, 0.0036991, -0.0019110, 0.0019110

### Unstable ReLU Count (Linear/Conv2D Layers)
- layer_idx=0, type=LayerType.Linear, total=256, inp1_unstable=10, inp2_unstable=10, delta_unstable=253
- layer_idx=2, type=LayerType.Linear, total=256, inp1_unstable=6, inp2_unstable=6, delta_unstable=256
- layer_idx=4, type=LayerType.Linear, total=256, inp1_unstable=0, inp2_unstable=0, delta_unstable=256
- layer_idx=6, type=LayerType.Linear, total=256, inp1_unstable=1, inp2_unstable=1, delta_unstable=256
- layer_idx=8, type=LayerType.Linear, total=10, inp1_unstable=0, inp2_unstable=0, delta_unstable=10

Time for backsubstitution: 1.16 seconds

### RS candidates at layer 1
type: RSZ, layer: 1, pos: 235
type: RSZ, layer: 1, pos: 155
type: RSZ, layer: 1, pos: 2
type: RSZ, layer: 1, pos: 201
type: RSZ, layer: 1, pos: 120
type: RSZ, layer: 1, pos: 69
type: RSZ, layer: 1, pos: 165

Time for candidate selection: 0.00 seconds

### Candidate
type: RSZ, layer: 1, pos: 235

### Relational analysis RSZ of RS_RSZ2_RSZ2_RSZ2_RSZ1

#### Relational analysis RSZ result of RS_RSZ2_RSZ2_RSZ2_RSZ1
Status: Status.VERIFIED
Output dim: 0, lower bound: -0.0055525, upper bound: 0.0056502
time: 1.04 seconds

### Relational analysis RSZ of RS_RSZ2_RSZ2_RSZ2_RSZ2

#### Relational analysis RSZ result of RS_RSZ2_RSZ2_RSZ2_RSZ2
Status: Status.VERIFIED
Output dim: 0, lower bound: -0.0055525, upper bound: 0.0056502
time: 1.05 seconds

## Summary of splitting (split count: 3)
- Time for RS candidates: 3.26 seconds
RS_RSZ1_RSZ1_RSZ1_RSZ1, status: Status.VERIFIED, split count: 4, time: 3.26
Output dim: 0, lower bound: -0.0057085, upper bound: 0.0057058
RS_RSZ1_RSZ1_RSZ1_RSZ2, status: Status.VERIFIED, split count: 4, time: 3.26
Output dim: 0, lower bound: -0.0057085, upper bound: 0.0057058
RS_RSZ1_RSZ1_RSZ2_RSZ1, status: Status.VERIFIED, split count: 4, time: 3.26
Output dim: 0, lower bound: -0.0050396, upper bound: 0.0050439
RS_RSZ1_RSZ1_RSZ2_RSZ2, status: Status.VERIFIED, split count: 4, time: 3.26
Output dim: 0, lower bound: -0.0050396, upper bound: 0.0050439
RS_RSZ2_RSZ1_RSZ1_RSZ1, status: Status.VERIFIED, split count: 4, time: 3.26
Output dim: 0, lower bound: -0.0057157, upper bound: 0.0056107
RS_RSZ2_RSZ1_RSZ1_RSZ2, status: Status.VERIFIED, split count: 4, time: 3.26
Output dim: 0, lower bound: -0.0057157, upper bound: 0.0056107
RS_RSZ2_RSZ1_RSZ2_RSZ1, status: Status.UNKNOWN, split count: 4, time: 3.26
Output dim: 0, lower bound: -0.0062714, upper bound: 0.0061703
RS_RSZ2_RSZ1_RSZ2_RSZ2, status: Status.UNKNOWN, split count: 4, time: 3.26
Output dim: 0, lower bound: -0.0062713, upper bound: 0.0061718
RS_RSZ2_RSZ2_RSZ1_RSZ1, status: Status.VERIFIED, split count: 4, time: 3.26
Output dim: 0, lower bound: -0.0041876, upper bound: 0.0042244
RS_RSZ2_RSZ2_RSZ1_RSZ2, status: Status.VERIFIED, split count: 4, time: 3.26
Output dim: 0, lower bound: -0.0041876, upper bound: 0.0042244
RS_RSZ2_RSZ2_RSZ2_RSZ1, status: Status.VERIFIED, split count: 4, time: 3.26
Output dim: 0, lower bound: -0.0055525, upper bound: 0.0056502
RS_RSZ2_RSZ2_RSZ2_RSZ2, status: Status.VERIFIED, split count: 4, time: 3.26
Output dim: 0, lower bound: -0.0055525, upper bound: 0.0056502

## BFS RS instance: RS_RSZ2_RSZ1_RSZ2_RSZ1

### Backsubstitution after applying RS history:
0: 0.9815361, 0.9898027, 0.9815361, 0.9898027, -0.0082666, 0.0082666
1: -0.0045208, -0.0038048, -0.0045208, -0.0038048, -0.0007160, 0.0007160
2: 0.0101096, 0.0139039, 0.0101096, 0.0139039, -0.0037943, 0.0037943
3: -0.0076910, -0.0058746, -0.0076910, -0.0058746, -0.0018164, 0.0018164
4: 0.0024846, 0.0036056, 0.0024846, 0.0036056, -0.0011210, 0.0011210
5: 0.0116747, 0.0204531, 0.0116747, 0.0204531, -0.0087784, 0.0087784
6: -0.0026336, -0.0014223, -0.0026336, -0.0014223, -0.0012112, 0.0012112
7: -0.0099515, -0.0068176, -0.0099515, -0.0068176, -0.0031339, 0.0031339
8: -0.0047975, -0.0027613, -0.0047975, -0.0027613, -0.0020363, 0.0020363
9: 0.0017881, 0.0036991, 0.0017881, 0.0036991, -0.0019110, 0.0019110

### Unstable ReLU Count (Linear/Conv2D Layers)
- layer_idx=0, type=LayerType.Linear, total=256, inp1_unstable=10, inp2_unstable=10, delta_unstable=252
- layer_idx=2, type=LayerType.Linear, total=256, inp1_unstable=6, inp2_unstable=6, delta_unstable=256
- layer_idx=4, type=LayerType.Linear, total=256, inp1_unstable=0, inp2_unstable=0, delta_unstable=256
- layer_idx=6, type=LayerType.Linear, total=256, inp1_unstable=1, inp2_unstable=1, delta_unstable=256
- layer_idx=8, type=LayerType.Linear, total=10, inp1_unstable=0, inp2_unstable=0, delta_unstable=10

Time for backsubstitution: 1.12 seconds

### RS candidates at layer 1
type: RSZ, layer: 1, pos: 201
type: RSZ, layer: 1, pos: 120
type: RSZ, layer: 1, pos: 69
type: RSZ, layer: 1, pos: 165
type: RSZ, layer: 1, pos: 235
type: RSZ, layer: 1, pos: 128

Time for candidate selection: 0.00 seconds

### Candidate
type: RSZ, layer: 1, pos: 201

### Relational analysis RSZ of RS_RSZ2_RSZ1_RSZ2_RSZ1_RSZ1

#### Relational analysis RSZ result of RS_RSZ2_RSZ1_RSZ2_RSZ1_RSZ1
Status: Status.VERIFIED
Output dim: 0, lower bound: -0.0049518, upper bound: 0.0048578
time: 0.73 seconds

### Relational analysis RSZ of RS_RSZ2_RSZ1_RSZ2_RSZ1_RSZ2

#### Relational analysis RSZ result of RS_RSZ2_RSZ1_RSZ2_RSZ1_RSZ2
Status: Status.VERIFIED
Output dim: 0, lower bound: -0.0049518, upper bound: 0.0048578
time: 0.72 seconds

## BFS RS instance: RS_RSZ2_RSZ1_RSZ2_RSZ2

### Backsubstitution after applying RS history:
0: 0.9815361, 0.9898027, 0.9815361, 0.9898027, -0.0082666, 0.0082666
1: -0.0045208, -0.0038048, -0.0045208, -0.0038048, -0.0007160, 0.0007160
2: 0.0101096, 0.0139039, 0.0101096, 0.0139039, -0.0037943, 0.0037943
3: -0.0076910, -0.0058746, -0.0076910, -0.0058746, -0.0018164, 0.0018164
4: 0.0024846, 0.0036056, 0.0024846, 0.0036056, -0.0011210, 0.0011210
5: 0.0116747, 0.0204531, 0.0116747, 0.0204531, -0.0087784, 0.0087784
6: -0.0026336, -0.0014223, -0.0026336, -0.0014223, -0.0012112, 0.0012112
7: -0.0099515, -0.0068176, -0.0099515, -0.0068176, -0.0031339, 0.0031339
8: -0.0047975, -0.0027613, -0.0047975, -0.0027613, -0.0020363, 0.0020363
9: 0.0017881, 0.0036991, 0.0017881, 0.0036991, -0.0019110, 0.0019110

### Unstable ReLU Count (Linear/Conv2D Layers)
- layer_idx=0, type=LayerType.Linear, total=256, inp1_unstable=10, inp2_unstable=10, delta_unstable=252
- layer_idx=2, type=LayerType.Linear, total=256, inp1_unstable=6, inp2_unstable=6, delta_unstable=256
- layer_idx=4, type=LayerType.Linear, total=256, inp1_unstable=0, inp2_unstable=0, delta_unstable=256
- layer_idx=6, type=LayerType.Linear, total=256, inp1_unstable=1, inp2_unstable=1, delta_unstable=256
- layer_idx=8, type=LayerType.Linear, total=10, inp1_unstable=0, inp2_unstable=0, delta_unstable=10

Time for backsubstitution: 1.08 seconds

### RS candidates at layer 1
type: RSZ, layer: 1, pos: 120
type: RSZ, layer: 1, pos: 69
type: RSZ, layer: 1, pos: 235
type: RSZ, layer: 1, pos: 128
type: RSZ, layer: 1, pos: 201
type: RSZ, layer: 1, pos: 165

Time for candidate selection: 0.00 seconds

### Candidate
type: RSZ, layer: 1, pos: 120

### Relational analysis RSZ of RS_RSZ2_RSZ1_RSZ2_RSZ2_RSZ1

#### Relational analysis RSZ result of RS_RSZ2_RSZ1_RSZ2_RSZ2_RSZ1
Status: Status.VERIFIED
Output dim: 0, lower bound: -0.0061557, upper bound: 0.0060114
time: 0.96 seconds

### Relational analysis RSZ of RS_RSZ2_RSZ1_RSZ2_RSZ2_RSZ2

#### Relational analysis RSZ result of RS_RSZ2_RSZ1_RSZ2_RSZ2_RSZ2
Status: Status.VERIFIED
Output dim: 0, lower bound: -0.0061125, upper bound: 0.0060524
time: 0.91 seconds

## Summary of splitting (split count: 4)
- Time for RS candidates: 2.96 seconds
RS_RSZ2_RSZ1_RSZ2_RSZ1_RSZ1, status: Status.VERIFIED, split count: 5, time: 2.96
Output dim: 0, lower bound: -0.0049518, upper bound: 0.0048578
RS_RSZ2_RSZ1_RSZ2_RSZ1_RSZ2, status: Status.VERIFIED, split count: 5, time: 2.96
Output dim: 0, lower bound: -0.0049518, upper bound: 0.0048578
RS_RSZ2_RSZ1_RSZ2_RSZ2_RSZ1, status: Status.VERIFIED, split count: 5, time: 2.96
Output dim: 0, lower bound: -0.0061557, upper bound: 0.0060114
RS_RSZ2_RSZ1_RSZ2_RSZ2_RSZ2, status: Status.VERIFIED, split count: 5, time: 2.96
Output dim: 0, lower bound: -0.0061125, upper bound: 0.0060524

## RS Result
status: Status.VERIFIED
execution time: (base) + (rs) = 3.19 + 41.63 = 44.82 seconds
