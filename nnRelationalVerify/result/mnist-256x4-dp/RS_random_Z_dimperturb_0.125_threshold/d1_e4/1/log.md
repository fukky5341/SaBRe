## Execution arguments:
Dataset: Dataset.MNIST
Network: onnx/mnist-net_256x4.onnx
Relational property: GLOBAL_ROBUSTNESS
LP Analysis: True
Epsilon: 0.015625
Delta epsilon: 0.00390625
execution index: (1, 4, 1)
Time budget: 600 seconds
Split limit: 100
Threshold: 0.00021588


## IAR start

### Unstable ReLU Count (Linear/Conv2D Layers)
- layer_idx=0, type=LayerType.Linear, total=256, inp1_unstable=4, inp2_unstable=4, delta_unstable=256
- layer_idx=2, type=LayerType.Linear, total=256, inp1_unstable=4, inp2_unstable=4, delta_unstable=256
- layer_idx=4, type=LayerType.Linear, total=256, inp1_unstable=0, inp2_unstable=0, delta_unstable=256
- layer_idx=6, type=LayerType.Linear, total=256, inp1_unstable=0, inp2_unstable=0, delta_unstable=256
- layer_idx=8, type=LayerType.Linear, total=10, inp1_unstable=0, inp2_unstable=0, delta_unstable=10

### BASE IAR bounds
Layer (inp1_lb, inp1_ub, inp2_lb, inp2_ub, d_lb, d_ub)
0: (0.0035972, 0.0041150, 0.0035972, 0.0041150, -0.0002705, 0.0002705)
1: (0.0018420, 0.0019168, 0.0018420, 0.0019168, -0.0000391, 0.0000391)
2: (0.0120848, 0.0123711, 0.0120848, 0.0123711, -0.0001496, 0.0001496)
3: (-0.0021818, -0.0018857, -0.0021818, -0.0018857, -0.0001547, 0.0001547)
4: (-0.0019956, -0.0016750, -0.0019956, -0.0016750, -0.0001674, 0.0001674)
5: (0.0056985, 0.0060018, 0.0056985, 0.0060018, -0.0001585, 0.0001585)
6: (0.0003095, 0.0015131, 0.0003095, 0.0015131, -0.0006287, 0.0006287)
7: (-0.0046174, -0.0029782, -0.0046174, -0.0029782, -0.0008562, 0.0008562)
8: (0.9859612, 0.9871160, 0.9859612, 0.9871160, -0.0006032, 0.0006032)
9: (-0.0041920, -0.0031438, -0.0041920, -0.0031438, -0.0005475, 0.0005475)

## BASE Relational Analysis

## BASE Result
execution time: IAR + RelationalAnalysis = 1.59 + 1.26 = 2.86 seconds
status: Status.UNKNOWN
relational distance
Output dim: 8, lower bound: -0.0003301, upper bound: 0.0003301

# Relational Split (RS) starts

## BFS RS instance: RS

Time for backsubstitution: 0.01 seconds

### RS candidates at layer 1
type: RSZ, layer: 1, pos: 89
type: RSZ, layer: 1, pos: 24
type: RSZ, layer: 1, pos: 146
type: RSZ, layer: 1, pos: 191

Time for candidate selection: 0.00 seconds

### Candidate
type: RSZ, layer: 1, pos: 89

### Relational analysis RSZ of RS_RSZ1

#### Relational analysis RSZ result of RS_RSZ1
Status: Status.UNKNOWN
Output dim: 8, lower bound: -0.0003082, upper bound: 0.0003080
time: 0.46 seconds

### Relational analysis RSZ of RS_RSZ2

#### Relational analysis RSZ result of RS_RSZ2
Status: Status.UNKNOWN
Output dim: 8, lower bound: -0.0003080, upper bound: 0.0003082
time: 0.45 seconds

## Summary of splitting (split count: 0)
- Time for RS candidates: 0.92 seconds
RS_RSZ1, status: Status.UNKNOWN, split count: 1, time: 0.92
Output dim: 8, lower bound: -0.0003082, upper bound: 0.0003080
RS_RSZ2, status: Status.UNKNOWN, split count: 1, time: 0.92
Output dim: 8, lower bound: -0.0003080, upper bound: 0.0003082

## BFS RS instance: RS_RSZ1

### Backsubstitution after applying RS history:
0: 0.0035972, 0.0041150, 0.0035972, 0.0041150, -0.0002544, 0.0002568
1: 0.0018420, 0.0019168, 0.0018420, 0.0019168, -0.0000368, 0.0000371
2: 0.0120848, 0.0123711, 0.0120848, 0.0123711, -0.0001420, 0.0001407
3: -0.0021818, -0.0018857, -0.0021818, -0.0018857, -0.0001468, 0.0001455
4: -0.0019956, -0.0016750, -0.0019956, -0.0016750, -0.0001575, 0.0001590
5: 0.0056985, 0.0060018, 0.0056985, 0.0060018, -0.0001504, 0.0001491
6: 0.0003095, 0.0015131, 0.0003095, 0.0015131, -0.0005969, 0.0005914
7: -0.0046174, -0.0029782, -0.0046174, -0.0029782, -0.0008054, 0.0008129
8: 0.9859612, 0.9871160, 0.9859612, 0.9871160, -0.0005674, 0.0005726
9: -0.0041920, -0.0031438, -0.0041920, -0.0031438, -0.0005198, 0.0005150

### Unstable ReLU Count (Linear/Conv2D Layers)
- layer_idx=0, type=LayerType.Linear, total=256, inp1_unstable=4, inp2_unstable=4, delta_unstable=255
- layer_idx=2, type=LayerType.Linear, total=256, inp1_unstable=4, inp2_unstable=4, delta_unstable=256
- layer_idx=4, type=LayerType.Linear, total=256, inp1_unstable=0, inp2_unstable=0, delta_unstable=256
- layer_idx=6, type=LayerType.Linear, total=256, inp1_unstable=0, inp2_unstable=0, delta_unstable=256
- layer_idx=8, type=LayerType.Linear, total=10, inp1_unstable=0, inp2_unstable=0, delta_unstable=10

Time for backsubstitution: 1.42 seconds

### RS candidates at layer 1
type: RSZ, layer: 1, pos: 191
type: RSZ, layer: 1, pos: 24
type: RSZ, layer: 1, pos: 146

Time for candidate selection: 0.00 seconds

### Candidate
type: RSZ, layer: 1, pos: 191

### Relational analysis RSZ of RS_RSZ1_RSZ1

#### Relational analysis RSZ result of RS_RSZ1_RSZ1
Status: Status.UNKNOWN
Output dim: 8, lower bound: -0.0003009, upper bound: 0.0003006
time: 0.46 seconds

### Relational analysis RSZ of RS_RSZ1_RSZ2

#### Relational analysis RSZ result of RS_RSZ1_RSZ2
Status: Status.UNKNOWN
Output dim: 8, lower bound: -0.0003007, upper bound: 0.0003009
time: 0.46 seconds

## BFS RS instance: RS_RSZ2

### Backsubstitution after applying RS history:
0: 0.0035972, 0.0041150, 0.0035972, 0.0041150, -0.0002568, 0.0002544
1: 0.0018420, 0.0019168, 0.0018420, 0.0019168, -0.0000371, 0.0000368
2: 0.0120848, 0.0123711, 0.0120848, 0.0123711, -0.0001407, 0.0001420
3: -0.0021818, -0.0018857, -0.0021818, -0.0018857, -0.0001455, 0.0001468
4: -0.0019956, -0.0016750, -0.0019956, -0.0016750, -0.0001590, 0.0001575
5: 0.0056985, 0.0060018, 0.0056985, 0.0060018, -0.0001491, 0.0001504
6: 0.0003095, 0.0015131, 0.0003095, 0.0015131, -0.0005914, 0.0005969
7: -0.0046174, -0.0029782, -0.0046174, -0.0029782, -0.0008129, 0.0008054
8: 0.9859612, 0.9871160, 0.9859612, 0.9871160, -0.0005726, 0.0005674
9: -0.0041920, -0.0031438, -0.0041920, -0.0031438, -0.0005150, 0.0005198

### Unstable ReLU Count (Linear/Conv2D Layers)
- layer_idx=0, type=LayerType.Linear, total=256, inp1_unstable=4, inp2_unstable=4, delta_unstable=255
- layer_idx=2, type=LayerType.Linear, total=256, inp1_unstable=4, inp2_unstable=4, delta_unstable=256
- layer_idx=4, type=LayerType.Linear, total=256, inp1_unstable=0, inp2_unstable=0, delta_unstable=256
- layer_idx=6, type=LayerType.Linear, total=256, inp1_unstable=0, inp2_unstable=0, delta_unstable=256
- layer_idx=8, type=LayerType.Linear, total=10, inp1_unstable=0, inp2_unstable=0, delta_unstable=10

Time for backsubstitution: 1.42 seconds

### RS candidates at layer 1
type: RSZ, layer: 1, pos: 191
type: RSZ, layer: 1, pos: 146
type: RSZ, layer: 1, pos: 24

Time for candidate selection: 0.00 seconds

### Candidate
type: RSZ, layer: 1, pos: 191

### Relational analysis RSZ of RS_RSZ2_RSZ1

#### Relational analysis RSZ result of RS_RSZ2_RSZ1
Status: Status.UNKNOWN
Output dim: 8, lower bound: -0.0003009, upper bound: 0.0003007
time: 0.47 seconds

### Relational analysis RSZ of RS_RSZ2_RSZ2

#### Relational analysis RSZ result of RS_RSZ2_RSZ2
Status: Status.UNKNOWN
Output dim: 8, lower bound: -0.0003006, upper bound: 0.0003009
time: 0.46 seconds

## Summary of splitting (split count: 1)
- Time for RS candidates: 2.36 seconds
RS_RSZ1_RSZ1, status: Status.UNKNOWN, split count: 2, time: 2.36
Output dim: 8, lower bound: -0.0003009, upper bound: 0.0003006
RS_RSZ1_RSZ2, status: Status.UNKNOWN, split count: 2, time: 2.36
Output dim: 8, lower bound: -0.0003007, upper bound: 0.0003009
RS_RSZ2_RSZ1, status: Status.UNKNOWN, split count: 2, time: 2.36
Output dim: 8, lower bound: -0.0003009, upper bound: 0.0003007
RS_RSZ2_RSZ2, status: Status.UNKNOWN, split count: 2, time: 2.36
Output dim: 8, lower bound: -0.0003006, upper bound: 0.0003009

## BFS RS instance: RS_RSZ1_RSZ1

### Backsubstitution after applying RS history:
0: 0.0035972, 0.0041150, 0.0035972, 0.0041150, -0.0002472, 0.0002482
1: 0.0018420, 0.0019168, 0.0018420, 0.0019168, -0.0000357, 0.0000359
2: 0.0120848, 0.0123711, 0.0120848, 0.0123711, -0.0001372, 0.0001367
3: -0.0021818, -0.0018857, -0.0021818, -0.0018857, -0.0001419, 0.0001414
4: -0.0019956, -0.0016750, -0.0019956, -0.0016750, -0.0001530, 0.0001536
5: 0.0056985, 0.0060018, 0.0056985, 0.0060018, -0.0001454, 0.0001448
6: 0.0003095, 0.0015131, 0.0003095, 0.0015131, -0.0005769, 0.0005746
7: -0.0046174, -0.0029782, -0.0046174, -0.0029782, -0.0007826, 0.0007856
8: 0.9859612, 0.9871160, 0.9859612, 0.9871160, -0.0005513, 0.0005534
9: -0.0041920, -0.0031438, -0.0041920, -0.0031438, -0.0005024, 0.0005004

### Unstable ReLU Count (Linear/Conv2D Layers)
- layer_idx=0, type=LayerType.Linear, total=256, inp1_unstable=4, inp2_unstable=4, delta_unstable=254
- layer_idx=2, type=LayerType.Linear, total=256, inp1_unstable=4, inp2_unstable=4, delta_unstable=256
- layer_idx=4, type=LayerType.Linear, total=256, inp1_unstable=0, inp2_unstable=0, delta_unstable=256
- layer_idx=6, type=LayerType.Linear, total=256, inp1_unstable=0, inp2_unstable=0, delta_unstable=256
- layer_idx=8, type=LayerType.Linear, total=10, inp1_unstable=0, inp2_unstable=0, delta_unstable=10

Time for backsubstitution: 1.38 seconds

### RS candidates at layer 1
type: RSZ, layer: 1, pos: 146
type: RSZ, layer: 1, pos: 24

Time for candidate selection: 0.00 seconds

### Candidate
type: RSZ, layer: 1, pos: 146

### Relational analysis RSZ of RS_RSZ1_RSZ1_RSZ1

#### Relational analysis RSZ result of RS_RSZ1_RSZ1_RSZ1
Status: Status.UNKNOWN
Output dim: 8, lower bound: -0.0002768, upper bound: 0.0002818
time: 0.45 seconds

### Relational analysis RSZ of RS_RSZ1_RSZ1_RSZ2

#### Relational analysis RSZ result of RS_RSZ1_RSZ1_RSZ2
Status: Status.UNKNOWN
Output dim: 8, lower bound: -0.0002818, upper bound: 0.0002762
time: 0.46 seconds

## BFS RS instance: RS_RSZ1_RSZ2

### Backsubstitution after applying RS history:
0: 0.0035972, 0.0041150, 0.0035972, 0.0041150, -0.0002458, 0.0002498
1: 0.0018420, 0.0019168, 0.0018420, 0.0019168, -0.0000355, 0.0000361
2: 0.0120848, 0.0123711, 0.0120848, 0.0123711, -0.0001381, 0.0001359
3: -0.0021818, -0.0018857, -0.0021818, -0.0018857, -0.0001429, 0.0001406
4: -0.0019956, -0.0016750, -0.0019956, -0.0016750, -0.0001522, 0.0001547
5: 0.0056985, 0.0060018, 0.0056985, 0.0060018, -0.0001464, 0.0001440
6: 0.0003095, 0.0015131, 0.0003095, 0.0015131, -0.0005807, 0.0005714
7: -0.0046174, -0.0029782, -0.0046174, -0.0029782, -0.0007782, 0.0007909
8: 0.9859612, 0.9871160, 0.9859612, 0.9871160, -0.0005482, 0.0005571
9: -0.0041920, -0.0031438, -0.0041920, -0.0031438, -0.0005057, 0.0004976

### Unstable ReLU Count (Linear/Conv2D Layers)
- layer_idx=0, type=LayerType.Linear, total=256, inp1_unstable=4, inp2_unstable=4, delta_unstable=254
- layer_idx=2, type=LayerType.Linear, total=256, inp1_unstable=4, inp2_unstable=4, delta_unstable=256
- layer_idx=4, type=LayerType.Linear, total=256, inp1_unstable=0, inp2_unstable=0, delta_unstable=256
- layer_idx=6, type=LayerType.Linear, total=256, inp1_unstable=0, inp2_unstable=0, delta_unstable=256
- layer_idx=8, type=LayerType.Linear, total=10, inp1_unstable=0, inp2_unstable=0, delta_unstable=10

Time for backsubstitution: 1.38 seconds

### RS candidates at layer 1
type: RSZ, layer: 1, pos: 24
type: RSZ, layer: 1, pos: 146

Time for candidate selection: 0.00 seconds

### Candidate
type: RSZ, layer: 1, pos: 24

### Relational analysis RSZ of RS_RSZ1_RSZ2_RSZ1

#### Relational analysis RSZ result of RS_RSZ1_RSZ2_RSZ1
Status: Status.UNKNOWN
Output dim: 8, lower bound: -0.0002671, upper bound: 0.0002743
time: 0.48 seconds

### Relational analysis RSZ of RS_RSZ1_RSZ2_RSZ2

#### Relational analysis RSZ result of RS_RSZ1_RSZ2_RSZ2
Status: Status.UNKNOWN
Output dim: 8, lower bound: -0.0002745, upper bound: 0.0002672
time: 0.47 seconds

## BFS RS instance: RS_RSZ2_RSZ1

### Backsubstitution after applying RS history:
0: 0.0035972, 0.0041150, 0.0035972, 0.0041150, -0.0002498, 0.0002458
1: 0.0018420, 0.0019168, 0.0018420, 0.0019168, -0.0000361, 0.0000355
2: 0.0120848, 0.0123711, 0.0120848, 0.0123711, -0.0001359, 0.0001381
3: -0.0021818, -0.0018857, -0.0021818, -0.0018857, -0.0001406, 0.0001429
4: -0.0019956, -0.0016750, -0.0019956, -0.0016750, -0.0001547, 0.0001522
5: 0.0056985, 0.0060018, 0.0056985, 0.0060018, -0.0001440, 0.0001464
6: 0.0003095, 0.0015131, 0.0003095, 0.0015131, -0.0005714, 0.0005807
7: -0.0046174, -0.0029782, -0.0046174, -0.0029782, -0.0007909, 0.0007782
8: 0.9859612, 0.9871160, 0.9859612, 0.9871160, -0.0005571, 0.0005482
9: -0.0041920, -0.0031438, -0.0041920, -0.0031438, -0.0004976, 0.0005057

### Unstable ReLU Count (Linear/Conv2D Layers)
- layer_idx=0, type=LayerType.Linear, total=256, inp1_unstable=4, inp2_unstable=4, delta_unstable=254
- layer_idx=2, type=LayerType.Linear, total=256, inp1_unstable=4, inp2_unstable=4, delta_unstable=256
- layer_idx=4, type=LayerType.Linear, total=256, inp1_unstable=0, inp2_unstable=0, delta_unstable=256
- layer_idx=6, type=LayerType.Linear, total=256, inp1_unstable=0, inp2_unstable=0, delta_unstable=256
- layer_idx=8, type=LayerType.Linear, total=10, inp1_unstable=0, inp2_unstable=0, delta_unstable=10

Time for backsubstitution: 1.43 seconds

### RS candidates at layer 1
type: RSZ, layer: 1, pos: 24
type: RSZ, layer: 1, pos: 146

Time for candidate selection: 0.00 seconds

### Candidate
type: RSZ, layer: 1, pos: 24

### Relational analysis RSZ of RS_RSZ2_RSZ1_RSZ1

#### Relational analysis RSZ result of RS_RSZ2_RSZ1_RSZ1
Status: Status.UNKNOWN
Output dim: 8, lower bound: -0.0002672, upper bound: 0.0002745
time: 0.47 seconds

### Relational analysis RSZ of RS_RSZ2_RSZ1_RSZ2

#### Relational analysis RSZ result of RS_RSZ2_RSZ1_RSZ2
Status: Status.UNKNOWN
Output dim: 8, lower bound: -0.0002743, upper bound: 0.0002671
time: 0.46 seconds

## BFS RS instance: RS_RSZ2_RSZ2

### Backsubstitution after applying RS history:
0: 0.0035972, 0.0041150, 0.0035972, 0.0041150, -0.0002482, 0.0002472
1: 0.0018420, 0.0019168, 0.0018420, 0.0019168, -0.0000359, 0.0000357
2: 0.0120848, 0.0123711, 0.0120848, 0.0123711, -0.0001367, 0.0001372
3: -0.0021818, -0.0018857, -0.0021818, -0.0018857, -0.0001414, 0.0001419
4: -0.0019956, -0.0016750, -0.0019956, -0.0016750, -0.0001536, 0.0001530
5: 0.0056985, 0.0060018, 0.0056985, 0.0060018, -0.0001448, 0.0001454
6: 0.0003095, 0.0015131, 0.0003095, 0.0015131, -0.0005746, 0.0005769
7: -0.0046174, -0.0029782, -0.0046174, -0.0029782, -0.0007856, 0.0007826
8: 0.9859612, 0.9871160, 0.9859612, 0.9871160, -0.0005534, 0.0005513
9: -0.0041920, -0.0031438, -0.0041920, -0.0031438, -0.0005004, 0.0005024

### Unstable ReLU Count (Linear/Conv2D Layers)
- layer_idx=0, type=LayerType.Linear, total=256, inp1_unstable=4, inp2_unstable=4, delta_unstable=254
- layer_idx=2, type=LayerType.Linear, total=256, inp1_unstable=4, inp2_unstable=4, delta_unstable=256
- layer_idx=4, type=LayerType.Linear, total=256, inp1_unstable=0, inp2_unstable=0, delta_unstable=256
- layer_idx=6, type=LayerType.Linear, total=256, inp1_unstable=0, inp2_unstable=0, delta_unstable=256
- layer_idx=8, type=LayerType.Linear, total=10, inp1_unstable=0, inp2_unstable=0, delta_unstable=10

Time for backsubstitution: 1.37 seconds

### RS candidates at layer 1
type: RSZ, layer: 1, pos: 146
type: RSZ, layer: 1, pos: 24

Time for candidate selection: 0.00 seconds

### Candidate
type: RSZ, layer: 1, pos: 146

### Relational analysis RSZ of RS_RSZ2_RSZ2_RSZ1

#### Relational analysis RSZ result of RS_RSZ2_RSZ2_RSZ1
Status: Status.UNKNOWN
Output dim: 8, lower bound: -0.0002761, upper bound: 0.0002818
time: 0.48 seconds

### Relational analysis RSZ of RS_RSZ2_RSZ2_RSZ2

#### Relational analysis RSZ result of RS_RSZ2_RSZ2_RSZ2
Status: Status.UNKNOWN
Output dim: 8, lower bound: -0.0002818, upper bound: 0.0002768
time: 0.48 seconds

## Summary of splitting (split count: 2)
- Time for RS candidates: 2.34 seconds
RS_RSZ1_RSZ1_RSZ1, status: Status.UNKNOWN, split count: 3, time: 2.34
Output dim: 8, lower bound: -0.0002768, upper bound: 0.0002818
RS_RSZ1_RSZ1_RSZ2, status: Status.UNKNOWN, split count: 3, time: 2.34
Output dim: 8, lower bound: -0.0002818, upper bound: 0.0002762
RS_RSZ1_RSZ2_RSZ1, status: Status.UNKNOWN, split count: 3, time: 2.34
Output dim: 8, lower bound: -0.0002671, upper bound: 0.0002743
RS_RSZ1_RSZ2_RSZ2, status: Status.UNKNOWN, split count: 3, time: 2.34
Output dim: 8, lower bound: -0.0002745, upper bound: 0.0002672
RS_RSZ2_RSZ1_RSZ1, status: Status.UNKNOWN, split count: 3, time: 2.34
Output dim: 8, lower bound: -0.0002672, upper bound: 0.0002745
RS_RSZ2_RSZ1_RSZ2, status: Status.UNKNOWN, split count: 3, time: 2.34
Output dim: 8, lower bound: -0.0002743, upper bound: 0.0002671
RS_RSZ2_RSZ2_RSZ1, status: Status.UNKNOWN, split count: 3, time: 2.34
Output dim: 8, lower bound: -0.0002761, upper bound: 0.0002818
RS_RSZ2_RSZ2_RSZ2, status: Status.UNKNOWN, split count: 3, time: 2.34
Output dim: 8, lower bound: -0.0002818, upper bound: 0.0002768

## BFS RS instance: RS_RSZ1_RSZ1_RSZ1

### Backsubstitution after applying RS history:
0: 0.0035972, 0.0041150, 0.0035972, 0.0041150, -0.0001628, 0.0001643
1: 0.0018420, 0.0019168, 0.0018420, 0.0019168, -0.0000235, 0.0000237
2: 0.0120848, 0.0123711, 0.0120848, 0.0123711, -0.0000908, 0.0000900
3: -0.0021818, -0.0018857, -0.0021818, -0.0018857, -0.0000940, 0.0000931
4: -0.0019956, -0.0016750, -0.0019956, -0.0016750, -0.0001008, 0.0001017
5: 0.0056985, 0.0060018, 0.0056985, 0.0060018, -0.0000963, 0.0000954
6: 0.0003095, 0.0015131, 0.0003095, 0.0015131, -0.0003819, 0.0003785
7: -0.0046174, -0.0029782, -0.0046174, -0.0029782, -0.0005154, 0.0005201
8: 0.9859612, 0.9871160, 0.9859612, 0.9871160, -0.0003631, 0.0003664
9: -0.0041920, -0.0031438, -0.0041920, -0.0031438, -0.0003326, 0.0003296

### Unstable ReLU Count (Linear/Conv2D Layers)
- layer_idx=0, type=LayerType.Linear, total=256, inp1_unstable=4, inp2_unstable=4, delta_unstable=253
- layer_idx=2, type=LayerType.Linear, total=256, inp1_unstable=4, inp2_unstable=4, delta_unstable=256
- layer_idx=4, type=LayerType.Linear, total=256, inp1_unstable=0, inp2_unstable=0, delta_unstable=256
- layer_idx=6, type=LayerType.Linear, total=256, inp1_unstable=0, inp2_unstable=0, delta_unstable=256
- layer_idx=8, type=LayerType.Linear, total=10, inp1_unstable=0, inp2_unstable=0, delta_unstable=10

Time for backsubstitution: 1.37 seconds

### RS candidates at layer 1
type: RSZ, layer: 1, pos: 24

Time for candidate selection: 0.00 seconds

### Candidate
type: RSZ, layer: 1, pos: 24

### Relational analysis RSZ of RS_RSZ1_RSZ1_RSZ1_RSZ1

#### Relational analysis RSZ result of RS_RSZ1_RSZ1_RSZ1_RSZ1
Status: Status.UNKNOWN
Output dim: 8, lower bound: -0.0002465, upper bound: 0.0002570
time: 0.47 seconds

### Relational analysis RSZ of RS_RSZ1_RSZ1_RSZ1_RSZ2

#### Relational analysis RSZ result of RS_RSZ1_RSZ1_RSZ1_RSZ2
Status: Status.UNKNOWN
Output dim: 8, lower bound: -0.0002534, upper bound: 0.0002494
time: 0.47 seconds

## BFS RS instance: RS_RSZ1_RSZ1_RSZ2

### Backsubstitution after applying RS history:
0: 0.0035972, 0.0041150, 0.0035972, 0.0041150, -0.0001624, 0.0001638
1: 0.0018420, 0.0019168, 0.0018420, 0.0019168, -0.0000235, 0.0000237
2: 0.0120848, 0.0123711, 0.0120848, 0.0123711, -0.0000906, 0.0000898
3: -0.0021818, -0.0018857, -0.0021818, -0.0018857, -0.0000937, 0.0000928
4: -0.0019956, -0.0016750, -0.0019956, -0.0016750, -0.0001005, 0.0001014
5: 0.0056985, 0.0060018, 0.0056985, 0.0060018, -0.0000959, 0.0000951
6: 0.0003095, 0.0015131, 0.0003095, 0.0015131, -0.0003807, 0.0003774
7: -0.0046174, -0.0029782, -0.0046174, -0.0029782, -0.0005140, 0.0005185
8: 0.9859612, 0.9871160, 0.9859612, 0.9871160, -0.0003621, 0.0003652
9: -0.0041920, -0.0031438, -0.0041920, -0.0031438, -0.0003315, 0.0003287

### Unstable ReLU Count (Linear/Conv2D Layers)
- layer_idx=0, type=LayerType.Linear, total=256, inp1_unstable=4, inp2_unstable=4, delta_unstable=253
- layer_idx=2, type=LayerType.Linear, total=256, inp1_unstable=4, inp2_unstable=4, delta_unstable=256
- layer_idx=4, type=LayerType.Linear, total=256, inp1_unstable=0, inp2_unstable=0, delta_unstable=256
- layer_idx=6, type=LayerType.Linear, total=256, inp1_unstable=0, inp2_unstable=0, delta_unstable=256
- layer_idx=8, type=LayerType.Linear, total=10, inp1_unstable=0, inp2_unstable=0, delta_unstable=10

Time for backsubstitution: 1.50 seconds

### RS candidates at layer 1
type: RSZ, layer: 1, pos: 24

Time for candidate selection: 0.00 seconds

### Candidate
type: RSZ, layer: 1, pos: 24

### Relational analysis RSZ of RS_RSZ1_RSZ1_RSZ2_RSZ1

#### Relational analysis RSZ result of RS_RSZ1_RSZ1_RSZ2_RSZ1
Status: Status.UNKNOWN
Output dim: 8, lower bound: -0.0002495, upper bound: 0.0002532
time: 0.51 seconds

### Relational analysis RSZ of RS_RSZ1_RSZ1_RSZ2_RSZ2

#### Relational analysis RSZ result of RS_RSZ1_RSZ1_RSZ2_RSZ2
Status: Status.UNKNOWN
Output dim: 8, lower bound: -0.0002573, upper bound: 0.0002452
time: 0.51 seconds

## BFS RS instance: RS_RSZ1_RSZ2_RSZ1

### Backsubstitution after applying RS history:
0: 0.0035972, 0.0041150, 0.0035972, 0.0041150, -0.0002174, 0.0002214
1: 0.0018420, 0.0019168, 0.0018420, 0.0019168, -0.0000314, 0.0000320
2: 0.0120848, 0.0123711, 0.0120848, 0.0123711, -0.0001224, 0.0001202
3: -0.0021818, -0.0018857, -0.0021818, -0.0018857, -0.0001266, 0.0001243
4: -0.0019956, -0.0016750, -0.0019956, -0.0016750, -0.0001345, 0.0001370
5: 0.0056985, 0.0060018, 0.0056985, 0.0060018, -0.0001297, 0.0001273
6: 0.0003095, 0.0015131, 0.0003095, 0.0015131, -0.0005146, 0.0005052
7: -0.0046174, -0.0029782, -0.0046174, -0.0029782, -0.0006880, 0.0007008
8: 0.9859612, 0.9871160, 0.9859612, 0.9871160, -0.0004847, 0.0004936
9: -0.0041920, -0.0031438, -0.0041920, -0.0031438, -0.0004481, 0.0004399

### Unstable ReLU Count (Linear/Conv2D Layers)
- layer_idx=0, type=LayerType.Linear, total=256, inp1_unstable=4, inp2_unstable=4, delta_unstable=253
- layer_idx=2, type=LayerType.Linear, total=256, inp1_unstable=4, inp2_unstable=4, delta_unstable=256
- layer_idx=4, type=LayerType.Linear, total=256, inp1_unstable=0, inp2_unstable=0, delta_unstable=256
- layer_idx=6, type=LayerType.Linear, total=256, inp1_unstable=0, inp2_unstable=0, delta_unstable=256
- layer_idx=8, type=LayerType.Linear, total=10, inp1_unstable=0, inp2_unstable=0, delta_unstable=10

Time for backsubstitution: 1.39 seconds

### RS candidates at layer 1
type: RSZ, layer: 1, pos: 146

Time for candidate selection: 0.00 seconds

### Candidate
type: RSZ, layer: 1, pos: 146

### Relational analysis RSZ of RS_RSZ1_RSZ2_RSZ1_RSZ1

#### Relational analysis RSZ result of RS_RSZ1_RSZ2_RSZ1_RSZ1
Status: Status.UNKNOWN
Output dim: 8, lower bound: -0.0002452, upper bound: 0.0002570
time: 0.47 seconds

### Relational analysis RSZ of RS_RSZ1_RSZ2_RSZ1_RSZ2

#### Relational analysis RSZ result of RS_RSZ1_RSZ2_RSZ1_RSZ2
Status: Status.UNKNOWN
Output dim: 8, lower bound: -0.0002494, upper bound: 0.0002532
time: 0.47 seconds

## BFS RS instance: RS_RSZ1_RSZ2_RSZ2

### Backsubstitution after applying RS history:
0: 0.0035972, 0.0041150, 0.0035972, 0.0041150, -0.0002207, 0.0002214
1: 0.0018420, 0.0019168, 0.0018420, 0.0019168, -0.0000319, 0.0000320
2: 0.0120848, 0.0123711, 0.0120848, 0.0123711, -0.0001224, 0.0001220
3: -0.0021818, -0.0018857, -0.0021818, -0.0018857, -0.0001266, 0.0001262
4: -0.0019956, -0.0016750, -0.0019956, -0.0016750, -0.0001366, 0.0001370
5: 0.0056985, 0.0060018, 0.0056985, 0.0060018, -0.0001297, 0.0001293
6: 0.0003095, 0.0015131, 0.0003095, 0.0015131, -0.0005145, 0.0005129
7: -0.0046174, -0.0029782, -0.0046174, -0.0029782, -0.0006985, 0.0007007
8: 0.9859612, 0.9871160, 0.9859612, 0.9871160, -0.0004921, 0.0004936
9: -0.0041920, -0.0031438, -0.0041920, -0.0031438, -0.0004481, 0.0004467

### Unstable ReLU Count (Linear/Conv2D Layers)
- layer_idx=0, type=LayerType.Linear, total=256, inp1_unstable=4, inp2_unstable=4, delta_unstable=253
- layer_idx=2, type=LayerType.Linear, total=256, inp1_unstable=4, inp2_unstable=4, delta_unstable=256
- layer_idx=4, type=LayerType.Linear, total=256, inp1_unstable=0, inp2_unstable=0, delta_unstable=256
- layer_idx=6, type=LayerType.Linear, total=256, inp1_unstable=0, inp2_unstable=0, delta_unstable=256
- layer_idx=8, type=LayerType.Linear, total=10, inp1_unstable=0, inp2_unstable=0, delta_unstable=10

Time for backsubstitution: 1.37 seconds

### RS candidates at layer 1
type: RSZ, layer: 1, pos: 146

Time for candidate selection: 0.00 seconds

### Candidate
type: RSZ, layer: 1, pos: 146

### Relational analysis RSZ of RS_RSZ1_RSZ2_RSZ2_RSZ1

#### Relational analysis RSZ result of RS_RSZ1_RSZ2_RSZ2_RSZ1
Status: Status.UNKNOWN
Output dim: 8, lower bound: -0.0002534, upper bound: 0.0002495
time: 0.49 seconds

### Relational analysis RSZ of RS_RSZ1_RSZ2_RSZ2_RSZ2

#### Relational analysis RSZ result of RS_RSZ1_RSZ2_RSZ2_RSZ2
Status: Status.UNKNOWN
Output dim: 8, lower bound: -0.0002573, upper bound: 0.0002464
time: 0.50 seconds

## BFS RS instance: RS_RSZ2_RSZ1_RSZ1

### Backsubstitution after applying RS history:
0: 0.0035972, 0.0041150, 0.0035972, 0.0041150, -0.0002214, 0.0002207
1: 0.0018420, 0.0019168, 0.0018420, 0.0019168, -0.0000320, 0.0000319
2: 0.0120848, 0.0123711, 0.0120848, 0.0123711, -0.0001220, 0.0001224
3: -0.0021818, -0.0018857, -0.0021818, -0.0018857, -0.0001262, 0.0001266
4: -0.0019956, -0.0016750, -0.0019956, -0.0016750, -0.0001370, 0.0001366
5: 0.0056985, 0.0060018, 0.0056985, 0.0060018, -0.0001293, 0.0001297
6: 0.0003095, 0.0015131, 0.0003095, 0.0015131, -0.0005129, 0.0005145
7: -0.0046174, -0.0029782, -0.0046174, -0.0029782, -0.0007007, 0.0006985
8: 0.9859612, 0.9871160, 0.9859612, 0.9871160, -0.0004936, 0.0004921
9: -0.0041920, -0.0031438, -0.0041920, -0.0031438, -0.0004467, 0.0004481

### Unstable ReLU Count (Linear/Conv2D Layers)
- layer_idx=0, type=LayerType.Linear, total=256, inp1_unstable=4, inp2_unstable=4, delta_unstable=253
- layer_idx=2, type=LayerType.Linear, total=256, inp1_unstable=4, inp2_unstable=4, delta_unstable=256
- layer_idx=4, type=LayerType.Linear, total=256, inp1_unstable=0, inp2_unstable=0, delta_unstable=256
- layer_idx=6, type=LayerType.Linear, total=256, inp1_unstable=0, inp2_unstable=0, delta_unstable=256
- layer_idx=8, type=LayerType.Linear, total=10, inp1_unstable=0, inp2_unstable=0, delta_unstable=10

Time for backsubstitution: 1.38 seconds

### RS candidates at layer 1
type: RSZ, layer: 1, pos: 146

Time for candidate selection: 0.00 seconds

### Candidate
type: RSZ, layer: 1, pos: 146

### Relational analysis RSZ of RS_RSZ2_RSZ1_RSZ1_RSZ1

#### Relational analysis RSZ result of RS_RSZ2_RSZ1_RSZ1_RSZ1
Status: Status.UNKNOWN
Output dim: 8, lower bound: -0.0002464, upper bound: 0.0002573
time: 0.47 seconds

### Relational analysis RSZ of RS_RSZ2_RSZ1_RSZ1_RSZ2

#### Relational analysis RSZ result of RS_RSZ2_RSZ1_RSZ1_RSZ2
Status: Status.UNKNOWN
Output dim: 8, lower bound: -0.0002495, upper bound: 0.0002534
time: 0.50 seconds

## BFS RS instance: RS_RSZ2_RSZ1_RSZ2

### Backsubstitution after applying RS history:
0: 0.0035972, 0.0041150, 0.0035972, 0.0041150, -0.0002214, 0.0002174
1: 0.0018420, 0.0019168, 0.0018420, 0.0019168, -0.0000320, 0.0000314
2: 0.0120848, 0.0123711, 0.0120848, 0.0123711, -0.0001202, 0.0001224
3: -0.0021818, -0.0018857, -0.0021818, -0.0018857, -0.0001243, 0.0001266
4: -0.0019956, -0.0016750, -0.0019956, -0.0016750, -0.0001370, 0.0001345
5: 0.0056985, 0.0060018, 0.0056985, 0.0060018, -0.0001273, 0.0001297
6: 0.0003095, 0.0015131, 0.0003095, 0.0015131, -0.0005052, 0.0005146
7: -0.0046174, -0.0029782, -0.0046174, -0.0029782, -0.0007008, 0.0006880
8: 0.9859612, 0.9871160, 0.9859612, 0.9871160, -0.0004936, 0.0004847
9: -0.0041920, -0.0031438, -0.0041920, -0.0031438, -0.0004399, 0.0004481

### Unstable ReLU Count (Linear/Conv2D Layers)
- layer_idx=0, type=LayerType.Linear, total=256, inp1_unstable=4, inp2_unstable=4, delta_unstable=253
- layer_idx=2, type=LayerType.Linear, total=256, inp1_unstable=4, inp2_unstable=4, delta_unstable=256
- layer_idx=4, type=LayerType.Linear, total=256, inp1_unstable=0, inp2_unstable=0, delta_unstable=256
- layer_idx=6, type=LayerType.Linear, total=256, inp1_unstable=0, inp2_unstable=0, delta_unstable=256
- layer_idx=8, type=LayerType.Linear, total=10, inp1_unstable=0, inp2_unstable=0, delta_unstable=10

Time for backsubstitution: 1.36 seconds

### RS candidates at layer 1
type: RSZ, layer: 1, pos: 146

Time for candidate selection: 0.00 seconds

### Candidate
type: RSZ, layer: 1, pos: 146

### Relational analysis RSZ of RS_RSZ2_RSZ1_RSZ2_RSZ1

#### Relational analysis RSZ result of RS_RSZ2_RSZ1_RSZ2_RSZ1
Status: Status.UNKNOWN
Output dim: 8, lower bound: -0.0002532, upper bound: 0.0002494
time: 0.47 seconds

### Relational analysis RSZ of RS_RSZ2_RSZ1_RSZ2_RSZ2

#### Relational analysis RSZ result of RS_RSZ2_RSZ1_RSZ2_RSZ2
Status: Status.UNKNOWN
Output dim: 8, lower bound: -0.0002570, upper bound: 0.0002452
time: 0.52 seconds

## BFS RS instance: RS_RSZ2_RSZ2_RSZ1

### Backsubstitution after applying RS history:
0: 0.0035972, 0.0041150, 0.0035972, 0.0041150, -0.0001638, 0.0001624
1: 0.0018420, 0.0019168, 0.0018420, 0.0019168, -0.0000237, 0.0000235
2: 0.0120848, 0.0123711, 0.0120848, 0.0123711, -0.0000898, 0.0000906
3: -0.0021818, -0.0018857, -0.0021818, -0.0018857, -0.0000928, 0.0000937
4: -0.0019956, -0.0016750, -0.0019956, -0.0016750, -0.0001014, 0.0001005
5: 0.0056985, 0.0060018, 0.0056985, 0.0060018, -0.0000951, 0.0000959
6: 0.0003095, 0.0015131, 0.0003095, 0.0015131, -0.0003774, 0.0003807
7: -0.0046174, -0.0029782, -0.0046174, -0.0029782, -0.0005185, 0.0005140
8: 0.9859612, 0.9871160, 0.9859612, 0.9871160, -0.0003652, 0.0003621
9: -0.0041920, -0.0031438, -0.0041920, -0.0031438, -0.0003287, 0.0003315

### Unstable ReLU Count (Linear/Conv2D Layers)
- layer_idx=0, type=LayerType.Linear, total=256, inp1_unstable=4, inp2_unstable=4, delta_unstable=253
- layer_idx=2, type=LayerType.Linear, total=256, inp1_unstable=4, inp2_unstable=4, delta_unstable=256
- layer_idx=4, type=LayerType.Linear, total=256, inp1_unstable=0, inp2_unstable=0, delta_unstable=256
- layer_idx=6, type=LayerType.Linear, total=256, inp1_unstable=0, inp2_unstable=0, delta_unstable=256
- layer_idx=8, type=LayerType.Linear, total=10, inp1_unstable=0, inp2_unstable=0, delta_unstable=10

Time for backsubstitution: 1.37 seconds

### RS candidates at layer 1
type: RSZ, layer: 1, pos: 24

Time for candidate selection: 0.00 seconds

### Candidate
type: RSZ, layer: 1, pos: 24

### Relational analysis RSZ of RS_RSZ2_RSZ2_RSZ1_RSZ1

#### Relational analysis RSZ result of RS_RSZ2_RSZ2_RSZ1_RSZ1
Status: Status.UNKNOWN
Output dim: 8, lower bound: -0.0002452, upper bound: 0.0002573
time: 0.47 seconds

### Relational analysis RSZ of RS_RSZ2_RSZ2_RSZ1_RSZ2

#### Relational analysis RSZ result of RS_RSZ2_RSZ2_RSZ1_RSZ2
Status: Status.UNKNOWN
Output dim: 8, lower bound: -0.0002532, upper bound: 0.0002495
time: 0.47 seconds

## BFS RS instance: RS_RSZ2_RSZ2_RSZ2

### Backsubstitution after applying RS history:
0: 0.0035972, 0.0041150, 0.0035972, 0.0041150, -0.0001643, 0.0001628
1: 0.0018420, 0.0019168, 0.0018420, 0.0019168, -0.0000237, 0.0000235
2: 0.0120848, 0.0123711, 0.0120848, 0.0123711, -0.0000900, 0.0000908
3: -0.0021818, -0.0018857, -0.0021818, -0.0018857, -0.0000931, 0.0000940
4: -0.0019956, -0.0016750, -0.0019956, -0.0016750, -0.0001017, 0.0001008
5: 0.0056985, 0.0060018, 0.0056985, 0.0060018, -0.0000954, 0.0000963
6: 0.0003095, 0.0015131, 0.0003095, 0.0015131, -0.0003785, 0.0003819
7: -0.0046174, -0.0029782, -0.0046174, -0.0029782, -0.0005201, 0.0005154
8: 0.9859612, 0.9871160, 0.9859612, 0.9871160, -0.0003664, 0.0003631
9: -0.0041920, -0.0031438, -0.0041920, -0.0031438, -0.0003296, 0.0003326

### Unstable ReLU Count (Linear/Conv2D Layers)
- layer_idx=0, type=LayerType.Linear, total=256, inp1_unstable=4, inp2_unstable=4, delta_unstable=253
- layer_idx=2, type=LayerType.Linear, total=256, inp1_unstable=4, inp2_unstable=4, delta_unstable=256
- layer_idx=4, type=LayerType.Linear, total=256, inp1_unstable=0, inp2_unstable=0, delta_unstable=256
- layer_idx=6, type=LayerType.Linear, total=256, inp1_unstable=0, inp2_unstable=0, delta_unstable=256
- layer_idx=8, type=LayerType.Linear, total=10, inp1_unstable=0, inp2_unstable=0, delta_unstable=10

Time for backsubstitution: 1.40 seconds

### RS candidates at layer 1
type: RSZ, layer: 1, pos: 24

Time for candidate selection: 0.00 seconds

### Candidate
type: RSZ, layer: 1, pos: 24

### Relational analysis RSZ of RS_RSZ2_RSZ2_RSZ2_RSZ1

#### Relational analysis RSZ result of RS_RSZ2_RSZ2_RSZ2_RSZ1
Status: Status.UNKNOWN
Output dim: 8, lower bound: -0.0002494, upper bound: 0.0002534
time: 0.48 seconds

### Relational analysis RSZ of RS_RSZ2_RSZ2_RSZ2_RSZ2

#### Relational analysis RSZ result of RS_RSZ2_RSZ2_RSZ2_RSZ2
Status: Status.UNKNOWN
Output dim: 8, lower bound: -0.0002570, upper bound: 0.0002465
time: 0.48 seconds

## Summary of splitting (split count: 3)
- Time for RS candidates: 2.37 seconds
RS_RSZ1_RSZ1_RSZ1_RSZ1, status: Status.UNKNOWN, split count: 4, time: 2.37
Output dim: 8, lower bound: -0.0002465, upper bound: 0.0002570
RS_RSZ1_RSZ1_RSZ1_RSZ2, status: Status.UNKNOWN, split count: 4, time: 2.37
Output dim: 8, lower bound: -0.0002534, upper bound: 0.0002494
RS_RSZ1_RSZ1_RSZ2_RSZ1, status: Status.UNKNOWN, split count: 4, time: 2.37
Output dim: 8, lower bound: -0.0002495, upper bound: 0.0002532
RS_RSZ1_RSZ1_RSZ2_RSZ2, status: Status.UNKNOWN, split count: 4, time: 2.37
Output dim: 8, lower bound: -0.0002573, upper bound: 0.0002452
RS_RSZ1_RSZ2_RSZ1_RSZ1, status: Status.UNKNOWN, split count: 4, time: 2.37
Output dim: 8, lower bound: -0.0002452, upper bound: 0.0002570
RS_RSZ1_RSZ2_RSZ1_RSZ2, status: Status.UNKNOWN, split count: 4, time: 2.37
Output dim: 8, lower bound: -0.0002494, upper bound: 0.0002532
RS_RSZ1_RSZ2_RSZ2_RSZ1, status: Status.UNKNOWN, split count: 4, time: 2.37
Output dim: 8, lower bound: -0.0002534, upper bound: 0.0002495
RS_RSZ1_RSZ2_RSZ2_RSZ2, status: Status.UNKNOWN, split count: 4, time: 2.37
Output dim: 8, lower bound: -0.0002573, upper bound: 0.0002464
RS_RSZ2_RSZ1_RSZ1_RSZ1, status: Status.UNKNOWN, split count: 4, time: 2.37
Output dim: 8, lower bound: -0.0002464, upper bound: 0.0002573
RS_RSZ2_RSZ1_RSZ1_RSZ2, status: Status.UNKNOWN, split count: 4, time: 2.37
Output dim: 8, lower bound: -0.0002495, upper bound: 0.0002534
RS_RSZ2_RSZ1_RSZ2_RSZ1, status: Status.UNKNOWN, split count: 4, time: 2.37
Output dim: 8, lower bound: -0.0002532, upper bound: 0.0002494
RS_RSZ2_RSZ1_RSZ2_RSZ2, status: Status.UNKNOWN, split count: 4, time: 2.37
Output dim: 8, lower bound: -0.0002570, upper bound: 0.0002452
RS_RSZ2_RSZ2_RSZ1_RSZ1, status: Status.UNKNOWN, split count: 4, time: 2.37
Output dim: 8, lower bound: -0.0002452, upper bound: 0.0002573
RS_RSZ2_RSZ2_RSZ1_RSZ2, status: Status.UNKNOWN, split count: 4, time: 2.37
Output dim: 8, lower bound: -0.0002532, upper bound: 0.0002495
RS_RSZ2_RSZ2_RSZ2_RSZ1, status: Status.UNKNOWN, split count: 4, time: 2.37
Output dim: 8, lower bound: -0.0002494, upper bound: 0.0002534
RS_RSZ2_RSZ2_RSZ2_RSZ2, status: Status.UNKNOWN, split count: 4, time: 2.37
Output dim: 8, lower bound: -0.0002570, upper bound: 0.0002465

## BFS RS instance: RS_RSZ1_RSZ1_RSZ1_RSZ1

### Backsubstitution after applying RS history:
0: 0.0035972, 0.0041150, 0.0035972, 0.0041150, -0.0001325, 0.0001373
1: 0.0018420, 0.0019168, 0.0018420, 0.0019168, -0.0000191, 0.0000198
2: 0.0120848, 0.0123711, 0.0120848, 0.0123711, -0.0000759, 0.0000733
3: -0.0021818, -0.0018857, -0.0021818, -0.0018857, -0.0000785, 0.0000758
4: -0.0019956, -0.0016750, -0.0019956, -0.0016750, -0.0000820, 0.0000850
5: 0.0056985, 0.0060018, 0.0056985, 0.0060018, -0.0000804, 0.0000776
6: 0.0003095, 0.0015131, 0.0003095, 0.0015131, -0.0003192, 0.0003080
7: -0.0046174, -0.0029782, -0.0046174, -0.0029782, -0.0004195, 0.0004347
8: 0.9859612, 0.9871160, 0.9859612, 0.9871160, -0.0002955, 0.0003062
9: -0.0041920, -0.0031438, -0.0041920, -0.0031438, -0.0002779, 0.0002683

### Unstable ReLU Count (Linear/Conv2D Layers)
- layer_idx=0, type=LayerType.Linear, total=256, inp1_unstable=4, inp2_unstable=4, delta_unstable=252
- layer_idx=2, type=LayerType.Linear, total=256, inp1_unstable=4, inp2_unstable=4, delta_unstable=256
- layer_idx=4, type=LayerType.Linear, total=256, inp1_unstable=0, inp2_unstable=0, delta_unstable=256
- layer_idx=6, type=LayerType.Linear, total=256, inp1_unstable=0, inp2_unstable=0, delta_unstable=256
- layer_idx=8, type=LayerType.Linear, total=10, inp1_unstable=0, inp2_unstable=0, delta_unstable=10

Time for backsubstitution: 1.42 seconds

### RS candidates at layer 3
type: RSZ, layer: 3, pos: 34
type: RSZ, layer: 3, pos: 240
type: RSZ, layer: 3, pos: 83
type: RSZ, layer: 3, pos: 49

Time for candidate selection: 0.00 seconds

### Candidate
type: RSZ, layer: 3, pos: 34

### Relational analysis RSZ of RS_RSZ1_RSZ1_RSZ1_RSZ1_RSZ1

#### Relational analysis RSZ result of RS_RSZ1_RSZ1_RSZ1_RSZ1_RSZ1
Status: Status.UNKNOWN
Output dim: 8, lower bound: -0.0002338, upper bound: 0.0002422
time: 0.49 seconds

### Relational analysis RSZ of RS_RSZ1_RSZ1_RSZ1_RSZ1_RSZ2

#### Relational analysis RSZ result of RS_RSZ1_RSZ1_RSZ1_RSZ1_RSZ2
Status: Status.UNKNOWN
Output dim: 8, lower bound: -0.0002296, upper bound: 0.0002446
time: 0.47 seconds

## BFS RS instance: RS_RSZ1_RSZ1_RSZ1_RSZ2

### Backsubstitution after applying RS history:
0: 0.0035972, 0.0041150, 0.0035972, 0.0041150, -0.0001345, 0.0001340
1: 0.0018420, 0.0019168, 0.0018420, 0.0019168, -0.0000194, 0.0000194
2: 0.0120848, 0.0123711, 0.0120848, 0.0123711, -0.0000741, 0.0000743
3: -0.0021818, -0.0018857, -0.0021818, -0.0018857, -0.0000766, 0.0000769
4: -0.0019956, -0.0016750, -0.0019956, -0.0016750, -0.0000832, 0.0000830
5: 0.0056985, 0.0060018, 0.0056985, 0.0060018, -0.0000785, 0.0000788
6: 0.0003095, 0.0015131, 0.0003095, 0.0015131, -0.0003115, 0.0003125
7: -0.0046174, -0.0029782, -0.0046174, -0.0029782, -0.0004256, 0.0004242
8: 0.9859612, 0.9871160, 0.9859612, 0.9871160, -0.0002998, 0.0002988
9: -0.0041920, -0.0031438, -0.0041920, -0.0031438, -0.0002712, 0.0002722

### Unstable ReLU Count (Linear/Conv2D Layers)
- layer_idx=0, type=LayerType.Linear, total=256, inp1_unstable=4, inp2_unstable=4, delta_unstable=252
- layer_idx=2, type=LayerType.Linear, total=256, inp1_unstable=4, inp2_unstable=4, delta_unstable=256
- layer_idx=4, type=LayerType.Linear, total=256, inp1_unstable=0, inp2_unstable=0, delta_unstable=256
- layer_idx=6, type=LayerType.Linear, total=256, inp1_unstable=0, inp2_unstable=0, delta_unstable=256
- layer_idx=8, type=LayerType.Linear, total=10, inp1_unstable=0, inp2_unstable=0, delta_unstable=10

Time for backsubstitution: 1.36 seconds

### RS candidates at layer 3
type: RSZ, layer: 3, pos: 34
type: RSZ, layer: 3, pos: 49
type: RSZ, layer: 3, pos: 83
type: RSZ, layer: 3, pos: 240

Time for candidate selection: 0.00 seconds

### Candidate
type: RSZ, layer: 3, pos: 34

### Relational analysis RSZ of RS_RSZ1_RSZ1_RSZ1_RSZ2_RSZ1

#### Relational analysis RSZ result of RS_RSZ1_RSZ1_RSZ1_RSZ2_RSZ1
Status: Status.UNKNOWN
Output dim: 8, lower bound: -0.0002411, upper bound: 0.0002335
time: 0.49 seconds

### Relational analysis RSZ of RS_RSZ1_RSZ1_RSZ1_RSZ2_RSZ2

#### Relational analysis RSZ result of RS_RSZ1_RSZ1_RSZ1_RSZ2_RSZ2
Status: Status.UNKNOWN
Output dim: 8, lower bound: -0.0002388, upper bound: 0.0002369
time: 0.48 seconds

## BFS RS instance: RS_RSZ1_RSZ1_RSZ2_RSZ1

### Backsubstitution after applying RS history:
0: 0.0035972, 0.0041150, 0.0035972, 0.0041150, -0.0001321, 0.0001351
1: 0.0018420, 0.0019168, 0.0018420, 0.0019168, -0.0000191, 0.0000195
2: 0.0120848, 0.0123711, 0.0120848, 0.0123711, -0.0000747, 0.0000730
3: -0.0021818, -0.0018857, -0.0021818, -0.0018857, -0.0000772, 0.0000755
4: -0.0019956, -0.0016750, -0.0019956, -0.0016750, -0.0000818, 0.0000836
5: 0.0056985, 0.0060018, 0.0056985, 0.0060018, -0.0000791, 0.0000774
6: 0.0003095, 0.0015131, 0.0003095, 0.0015131, -0.0003139, 0.0003070
7: -0.0046174, -0.0029782, -0.0046174, -0.0029782, -0.0004181, 0.0004275
8: 0.9859612, 0.9871160, 0.9859612, 0.9871160, -0.0002945, 0.0003012
9: -0.0041920, -0.0031438, -0.0041920, -0.0031438, -0.0002734, 0.0002673

### Unstable ReLU Count (Linear/Conv2D Layers)
- layer_idx=0, type=LayerType.Linear, total=256, inp1_unstable=4, inp2_unstable=4, delta_unstable=252
- layer_idx=2, type=LayerType.Linear, total=256, inp1_unstable=4, inp2_unstable=4, delta_unstable=256
- layer_idx=4, type=LayerType.Linear, total=256, inp1_unstable=0, inp2_unstable=0, delta_unstable=256
- layer_idx=6, type=LayerType.Linear, total=256, inp1_unstable=0, inp2_unstable=0, delta_unstable=256
- layer_idx=8, type=LayerType.Linear, total=10, inp1_unstable=0, inp2_unstable=0, delta_unstable=10

Time for backsubstitution: 1.36 seconds

### RS candidates at layer 3
type: RSZ, layer: 3, pos: 49
type: RSZ, layer: 3, pos: 240
type: RSZ, layer: 3, pos: 34
type: RSZ, layer: 3, pos: 83

Time for candidate selection: 0.00 seconds

### Candidate
type: RSZ, layer: 3, pos: 49

### Relational analysis RSZ of RS_RSZ1_RSZ1_RSZ2_RSZ1_RSZ1

#### Relational analysis RSZ result of RS_RSZ1_RSZ1_RSZ2_RSZ1_RSZ1
Status: Status.UNKNOWN
Output dim: 8, lower bound: -0.0001693, upper bound: 0.0002284
time: 0.50 seconds

### Relational analysis RSZ of RS_RSZ1_RSZ1_RSZ2_RSZ1_RSZ2

#### Relational analysis RSZ result of RS_RSZ1_RSZ1_RSZ2_RSZ1_RSZ2
Status: Status.UNKNOWN
Output dim: 8, lower bound: -0.0002277, upper bound: 0.0001689
time: 0.48 seconds

## BFS RS instance: RS_RSZ1_RSZ1_RSZ2_RSZ2

### Backsubstitution after applying RS history:
0: 0.0035972, 0.0041150, 0.0035972, 0.0041150, -0.0001357, 0.0001335
1: 0.0018420, 0.0019168, 0.0018420, 0.0019168, -0.0000196, 0.0000193
2: 0.0120848, 0.0123711, 0.0120848, 0.0123711, -0.0000738, 0.0000750
3: -0.0021818, -0.0018857, -0.0021818, -0.0018857, -0.0000763, 0.0000776
4: -0.0019956, -0.0016750, -0.0019956, -0.0016750, -0.0000840, 0.0000826
5: 0.0056985, 0.0060018, 0.0056985, 0.0060018, -0.0000782, 0.0000795
6: 0.0003095, 0.0015131, 0.0003095, 0.0015131, -0.0003103, 0.0003153
7: -0.0046174, -0.0029782, -0.0046174, -0.0029782, -0.0004294, 0.0004226
8: 0.9859612, 0.9871160, 0.9859612, 0.9871160, -0.0003025, 0.0002977
9: -0.0041920, -0.0031438, -0.0041920, -0.0031438, -0.0002702, 0.0002746

### Unstable ReLU Count (Linear/Conv2D Layers)
- layer_idx=0, type=LayerType.Linear, total=256, inp1_unstable=4, inp2_unstable=4, delta_unstable=252
- layer_idx=2, type=LayerType.Linear, total=256, inp1_unstable=4, inp2_unstable=4, delta_unstable=256
- layer_idx=4, type=LayerType.Linear, total=256, inp1_unstable=0, inp2_unstable=0, delta_unstable=256
- layer_idx=6, type=LayerType.Linear, total=256, inp1_unstable=0, inp2_unstable=0, delta_unstable=256
- layer_idx=8, type=LayerType.Linear, total=10, inp1_unstable=0, inp2_unstable=0, delta_unstable=10

Time for backsubstitution: 1.42 seconds

### RS candidates at layer 3
type: RSZ, layer: 3, pos: 240
type: RSZ, layer: 3, pos: 49
type: RSZ, layer: 3, pos: 83
type: RSZ, layer: 3, pos: 34

Time for candidate selection: 0.00 seconds

### Candidate
type: RSZ, layer: 3, pos: 240

### Relational analysis RSZ of RS_RSZ1_RSZ1_RSZ2_RSZ2_RSZ1

#### Relational analysis RSZ result of RS_RSZ1_RSZ1_RSZ2_RSZ2_RSZ1
Status: Status.UNKNOWN
Output dim: 8, lower bound: -0.0002567, upper bound: 0.0002450
time: 0.53 seconds

### Relational analysis RSZ of RS_RSZ1_RSZ1_RSZ2_RSZ2_RSZ2

#### Relational analysis RSZ result of RS_RSZ1_RSZ1_RSZ2_RSZ2_RSZ2
Status: Status.UNKNOWN
Output dim: 8, lower bound: -0.0002571, upper bound: 0.0002406
time: 0.49 seconds

## BFS RS instance: RS_RSZ1_RSZ2_RSZ1_RSZ1

### Backsubstitution after applying RS history:
0: 0.0035972, 0.0041150, 0.0035972, 0.0041150, -0.0001311, 0.0001375
1: 0.0018420, 0.0019168, 0.0018420, 0.0019168, -0.0000189, 0.0000199
2: 0.0120848, 0.0123711, 0.0120848, 0.0123711, -0.0000760, 0.0000725
3: -0.0021818, -0.0018857, -0.0021818, -0.0018857, -0.0000786, 0.0000750
4: -0.0019956, -0.0016750, -0.0019956, -0.0016750, -0.0000812, 0.0000851
5: 0.0056985, 0.0060018, 0.0056985, 0.0060018, -0.0000805, 0.0000768
6: 0.0003095, 0.0015131, 0.0003095, 0.0015131, -0.0003196, 0.0003048
7: -0.0046174, -0.0029782, -0.0046174, -0.0029782, -0.0004151, 0.0004352
8: 0.9859612, 0.9871160, 0.9859612, 0.9871160, -0.0002924, 0.0003066
9: -0.0041920, -0.0031438, -0.0041920, -0.0031438, -0.0002783, 0.0002654

### Unstable ReLU Count (Linear/Conv2D Layers)
- layer_idx=0, type=LayerType.Linear, total=256, inp1_unstable=4, inp2_unstable=4, delta_unstable=252
- layer_idx=2, type=LayerType.Linear, total=256, inp1_unstable=4, inp2_unstable=4, delta_unstable=256
- layer_idx=4, type=LayerType.Linear, total=256, inp1_unstable=0, inp2_unstable=0, delta_unstable=256
- layer_idx=6, type=LayerType.Linear, total=256, inp1_unstable=0, inp2_unstable=0, delta_unstable=256
- layer_idx=8, type=LayerType.Linear, total=10, inp1_unstable=0, inp2_unstable=0, delta_unstable=10

Time for backsubstitution: 1.46 seconds

### RS candidates at layer 3
type: RSZ, layer: 3, pos: 83
type: RSZ, layer: 3, pos: 49
type: RSZ, layer: 3, pos: 34
type: RSZ, layer: 3, pos: 240

Time for candidate selection: 0.00 seconds

### Candidate
type: RSZ, layer: 3, pos: 83

### Relational analysis RSZ of RS_RSZ1_RSZ2_RSZ1_RSZ1_RSZ1

#### Relational analysis RSZ result of RS_RSZ1_RSZ2_RSZ1_RSZ1_RSZ1
Status: Status.UNKNOWN
Output dim: 8, lower bound: -0.0001860, upper bound: 0.0002381
time: 0.49 seconds

### Relational analysis RSZ of RS_RSZ1_RSZ2_RSZ1_RSZ1_RSZ2

#### Relational analysis RSZ result of RS_RSZ1_RSZ2_RSZ1_RSZ1_RSZ2
Status: Status.UNKNOWN
Output dim: 8, lower bound: -0.0002296, upper bound: 0.0002136
time: 0.48 seconds

## BFS RS instance: RS_RSZ1_RSZ2_RSZ1_RSZ2

### Backsubstitution after applying RS history:
0: 0.0035972, 0.0041150, 0.0035972, 0.0041150, -0.0001319, 0.0001352
1: 0.0018420, 0.0019168, 0.0018420, 0.0019168, -0.0000191, 0.0000195
2: 0.0120848, 0.0123711, 0.0120848, 0.0123711, -0.0000747, 0.0000729
3: -0.0021818, -0.0018857, -0.0021818, -0.0018857, -0.0000773, 0.0000754
4: -0.0019956, -0.0016750, -0.0019956, -0.0016750, -0.0000816, 0.0000837
5: 0.0056985, 0.0060018, 0.0056985, 0.0060018, -0.0000792, 0.0000773
6: 0.0003095, 0.0015131, 0.0003095, 0.0015131, -0.0003142, 0.0003065
7: -0.0046174, -0.0029782, -0.0046174, -0.0029782, -0.0004175, 0.0004278
8: 0.9859612, 0.9871160, 0.9859612, 0.9871160, -0.0002941, 0.0003014
9: -0.0041920, -0.0031438, -0.0041920, -0.0031438, -0.0002736, 0.0002669

### Unstable ReLU Count (Linear/Conv2D Layers)
- layer_idx=0, type=LayerType.Linear, total=256, inp1_unstable=4, inp2_unstable=4, delta_unstable=252
- layer_idx=2, type=LayerType.Linear, total=256, inp1_unstable=4, inp2_unstable=4, delta_unstable=256
- layer_idx=4, type=LayerType.Linear, total=256, inp1_unstable=0, inp2_unstable=0, delta_unstable=256
- layer_idx=6, type=LayerType.Linear, total=256, inp1_unstable=0, inp2_unstable=0, delta_unstable=256
- layer_idx=8, type=LayerType.Linear, total=10, inp1_unstable=0, inp2_unstable=0, delta_unstable=10

Time for backsubstitution: 1.38 seconds

### RS candidates at layer 3
type: RSZ, layer: 3, pos: 83
type: RSZ, layer: 3, pos: 34
type: RSZ, layer: 3, pos: 49
type: RSZ, layer: 3, pos: 240

Time for candidate selection: 0.00 seconds

### Candidate
type: RSZ, layer: 3, pos: 83

### Relational analysis RSZ of RS_RSZ1_RSZ2_RSZ1_RSZ2_RSZ1

#### Relational analysis RSZ result of RS_RSZ1_RSZ2_RSZ1_RSZ2_RSZ1
Status: Status.UNKNOWN
Output dim: 8, lower bound: -0.0001948, upper bound: 0.0002327
time: 0.49 seconds

### Relational analysis RSZ of RS_RSZ1_RSZ2_RSZ1_RSZ2_RSZ2

#### Relational analysis RSZ result of RS_RSZ1_RSZ2_RSZ1_RSZ2_RSZ2
Status: Status.UNKNOWN
Output dim: 8, lower bound: -0.0002335, upper bound: 0.0002057
time: 0.48 seconds

## BFS RS instance: RS_RSZ1_RSZ2_RSZ2_RSZ1

### Backsubstitution after applying RS history:
0: 0.0035972, 0.0041150, 0.0035972, 0.0041150, -0.0001345, 0.0001346
1: 0.0018420, 0.0019168, 0.0018420, 0.0019168, -0.0000194, 0.0000195
2: 0.0120848, 0.0123711, 0.0120848, 0.0123711, -0.0000744, 0.0000743
3: -0.0021818, -0.0018857, -0.0021818, -0.0018857, -0.0000770, 0.0000769
4: -0.0019956, -0.0016750, -0.0019956, -0.0016750, -0.0000832, 0.0000833
5: 0.0056985, 0.0060018, 0.0056985, 0.0060018, -0.0000789, 0.0000788
6: 0.0003095, 0.0015131, 0.0003095, 0.0015131, -0.0003129, 0.0003125
7: -0.0046174, -0.0029782, -0.0046174, -0.0029782, -0.0004256, 0.0004262
8: 0.9859612, 0.9871160, 0.9859612, 0.9871160, -0.0002998, 0.0003002
9: -0.0041920, -0.0031438, -0.0041920, -0.0031438, -0.0002725, 0.0002722

### Unstable ReLU Count (Linear/Conv2D Layers)
- layer_idx=0, type=LayerType.Linear, total=256, inp1_unstable=4, inp2_unstable=4, delta_unstable=252
- layer_idx=2, type=LayerType.Linear, total=256, inp1_unstable=4, inp2_unstable=4, delta_unstable=256
- layer_idx=4, type=LayerType.Linear, total=256, inp1_unstable=0, inp2_unstable=0, delta_unstable=256
- layer_idx=6, type=LayerType.Linear, total=256, inp1_unstable=0, inp2_unstable=0, delta_unstable=256
- layer_idx=8, type=LayerType.Linear, total=10, inp1_unstable=0, inp2_unstable=0, delta_unstable=10

Time for backsubstitution: 1.57 seconds

### RS candidates at layer 3
type: RSZ, layer: 3, pos: 240
type: RSZ, layer: 3, pos: 49
type: RSZ, layer: 3, pos: 34
type: RSZ, layer: 3, pos: 83

Time for candidate selection: 0.00 seconds

### Candidate
type: RSZ, layer: 3, pos: 240

### Relational analysis RSZ of RS_RSZ1_RSZ2_RSZ2_RSZ1_RSZ1

#### Relational analysis RSZ result of RS_RSZ1_RSZ2_RSZ2_RSZ1_RSZ1
Status: Status.UNKNOWN
Output dim: 8, lower bound: -0.0002532, upper bound: 0.0002493
time: 0.53 seconds

### Relational analysis RSZ of RS_RSZ1_RSZ2_RSZ2_RSZ1_RSZ2

#### Relational analysis RSZ result of RS_RSZ1_RSZ2_RSZ2_RSZ1_RSZ2
Status: Status.UNKNOWN
Output dim: 8, lower bound: -0.0002532, upper bound: 0.0002471
time: 0.48 seconds

## BFS RS instance: RS_RSZ1_RSZ2_RSZ2_RSZ2

### Backsubstitution after applying RS history:
0: 0.0035972, 0.0041150, 0.0035972, 0.0041150, -0.0001361, 0.0001351
1: 0.0018420, 0.0019168, 0.0018420, 0.0019168, -0.0000197, 0.0000195
2: 0.0120848, 0.0123711, 0.0120848, 0.0123711, -0.0000747, 0.0000752
3: -0.0021818, -0.0018857, -0.0021818, -0.0018857, -0.0000773, 0.0000778
4: -0.0019956, -0.0016750, -0.0019956, -0.0016750, -0.0000842, 0.0000837
5: 0.0056985, 0.0060018, 0.0056985, 0.0060018, -0.0000792, 0.0000797
6: 0.0003095, 0.0015131, 0.0003095, 0.0015131, -0.0003141, 0.0003163
7: -0.0046174, -0.0029782, -0.0046174, -0.0029782, -0.0004307, 0.0004278
8: 0.9859612, 0.9871160, 0.9859612, 0.9871160, -0.0003034, 0.0003013
9: -0.0041920, -0.0031438, -0.0041920, -0.0031438, -0.0002735, 0.0002754

### Unstable ReLU Count (Linear/Conv2D Layers)
- layer_idx=0, type=LayerType.Linear, total=256, inp1_unstable=4, inp2_unstable=4, delta_unstable=252
- layer_idx=2, type=LayerType.Linear, total=256, inp1_unstable=4, inp2_unstable=4, delta_unstable=256
- layer_idx=4, type=LayerType.Linear, total=256, inp1_unstable=0, inp2_unstable=0, delta_unstable=256
- layer_idx=6, type=LayerType.Linear, total=256, inp1_unstable=0, inp2_unstable=0, delta_unstable=256
- layer_idx=8, type=LayerType.Linear, total=10, inp1_unstable=0, inp2_unstable=0, delta_unstable=10

Time for backsubstitution: 1.39 seconds

### RS candidates at layer 3
type: RSZ, layer: 3, pos: 49
type: RSZ, layer: 3, pos: 34
type: RSZ, layer: 3, pos: 240
type: RSZ, layer: 3, pos: 83

Time for candidate selection: 0.00 seconds

### Candidate
type: RSZ, layer: 3, pos: 49

### Relational analysis RSZ of RS_RSZ1_RSZ2_RSZ2_RSZ2_RSZ1

#### Relational analysis RSZ result of RS_RSZ1_RSZ2_RSZ2_RSZ2_RSZ1
Status: Status.UNKNOWN
Output dim: 8, lower bound: -0.0001798, upper bound: 0.0002223
time: 0.48 seconds

### Relational analysis RSZ of RS_RSZ1_RSZ2_RSZ2_RSZ2_RSZ2

#### Relational analysis RSZ result of RS_RSZ1_RSZ2_RSZ2_RSZ2_RSZ2
Status: Status.UNKNOWN
Output dim: 8, lower bound: -0.0002344, upper bound: 0.0001603
time: 0.49 seconds

## BFS RS instance: RS_RSZ2_RSZ1_RSZ1_RSZ1

### Backsubstitution after applying RS history:
0: 0.0035972, 0.0041150, 0.0035972, 0.0041150, -0.0001351, 0.0001361
1: 0.0018420, 0.0019168, 0.0018420, 0.0019168, -0.0000195, 0.0000197
2: 0.0120848, 0.0123711, 0.0120848, 0.0123711, -0.0000752, 0.0000747
3: -0.0021818, -0.0018857, -0.0021818, -0.0018857, -0.0000778, 0.0000773
4: -0.0019956, -0.0016750, -0.0019956, -0.0016750, -0.0000837, 0.0000842
5: 0.0056985, 0.0060018, 0.0056985, 0.0060018, -0.0000797, 0.0000792
6: 0.0003095, 0.0015131, 0.0003095, 0.0015131, -0.0003163, 0.0003141
7: -0.0046174, -0.0029782, -0.0046174, -0.0029782, -0.0004278, 0.0004307
8: 0.9859612, 0.9871160, 0.9859612, 0.9871160, -0.0003013, 0.0003034
9: -0.0041920, -0.0031438, -0.0041920, -0.0031438, -0.0002754, 0.0002735

### Unstable ReLU Count (Linear/Conv2D Layers)
- layer_idx=0, type=LayerType.Linear, total=256, inp1_unstable=4, inp2_unstable=4, delta_unstable=252
- layer_idx=2, type=LayerType.Linear, total=256, inp1_unstable=4, inp2_unstable=4, delta_unstable=256
- layer_idx=4, type=LayerType.Linear, total=256, inp1_unstable=0, inp2_unstable=0, delta_unstable=256
- layer_idx=6, type=LayerType.Linear, total=256, inp1_unstable=0, inp2_unstable=0, delta_unstable=256
- layer_idx=8, type=LayerType.Linear, total=10, inp1_unstable=0, inp2_unstable=0, delta_unstable=10

Time for backsubstitution: 1.49 seconds

### RS candidates at layer 3
type: RSZ, layer: 3, pos: 83
type: RSZ, layer: 3, pos: 34
type: RSZ, layer: 3, pos: 240
type: RSZ, layer: 3, pos: 49

Time for candidate selection: 0.00 seconds

### Candidate
type: RSZ, layer: 3, pos: 83

### Relational analysis RSZ of RS_RSZ2_RSZ1_RSZ1_RSZ1_RSZ1

#### Relational analysis RSZ result of RS_RSZ2_RSZ1_RSZ1_RSZ1_RSZ1
Status: Status.UNKNOWN
Output dim: 8, lower bound: -0.0001873, upper bound: 0.0002381
time: 0.54 seconds

### Relational analysis RSZ of RS_RSZ2_RSZ1_RSZ1_RSZ1_RSZ2

#### Relational analysis RSZ result of RS_RSZ2_RSZ1_RSZ1_RSZ1_RSZ2
Status: Status.UNKNOWN
Output dim: 8, lower bound: -0.0002302, upper bound: 0.0002147
time: 0.55 seconds

## BFS RS instance: RS_RSZ2_RSZ1_RSZ1_RSZ2

### Backsubstitution after applying RS history:
0: 0.0035972, 0.0041150, 0.0035972, 0.0041150, -0.0001346, 0.0001345
1: 0.0018420, 0.0019168, 0.0018420, 0.0019168, -0.0000195, 0.0000194
2: 0.0120848, 0.0123711, 0.0120848, 0.0123711, -0.0000743, 0.0000744
3: -0.0021818, -0.0018857, -0.0021818, -0.0018857, -0.0000769, 0.0000770
4: -0.0019956, -0.0016750, -0.0019956, -0.0016750, -0.0000833, 0.0000832
5: 0.0056985, 0.0060018, 0.0056985, 0.0060018, -0.0000788, 0.0000789
6: 0.0003095, 0.0015131, 0.0003095, 0.0015131, -0.0003125, 0.0003129
7: -0.0046174, -0.0029782, -0.0046174, -0.0029782, -0.0004262, 0.0004256
8: 0.9859612, 0.9871160, 0.9859612, 0.9871160, -0.0003002, 0.0002998
9: -0.0041920, -0.0031438, -0.0041920, -0.0031438, -0.0002722, 0.0002725

### Unstable ReLU Count (Linear/Conv2D Layers)
- layer_idx=0, type=LayerType.Linear, total=256, inp1_unstable=4, inp2_unstable=4, delta_unstable=252
- layer_idx=2, type=LayerType.Linear, total=256, inp1_unstable=4, inp2_unstable=4, delta_unstable=256
- layer_idx=4, type=LayerType.Linear, total=256, inp1_unstable=0, inp2_unstable=0, delta_unstable=256
- layer_idx=6, type=LayerType.Linear, total=256, inp1_unstable=0, inp2_unstable=0, delta_unstable=256
- layer_idx=8, type=LayerType.Linear, total=10, inp1_unstable=0, inp2_unstable=0, delta_unstable=10

Time for backsubstitution: 1.49 seconds

### RS candidates at layer 3
type: RSZ, layer: 3, pos: 83
type: RSZ, layer: 3, pos: 49
type: RSZ, layer: 3, pos: 34
type: RSZ, layer: 3, pos: 240

Time for candidate selection: 0.00 seconds

### Candidate
type: RSZ, layer: 3, pos: 83

### Relational analysis RSZ of RS_RSZ2_RSZ1_RSZ1_RSZ2_RSZ1

#### Relational analysis RSZ result of RS_RSZ2_RSZ1_RSZ1_RSZ2_RSZ1
Status: Status.UNKNOWN
Output dim: 8, lower bound: -0.0001974, upper bound: 0.0002327
time: 0.51 seconds

### Relational analysis RSZ of RS_RSZ2_RSZ1_RSZ1_RSZ2_RSZ2

#### Relational analysis RSZ result of RS_RSZ2_RSZ1_RSZ1_RSZ2_RSZ2
Status: Status.UNKNOWN
Output dim: 8, lower bound: -0.0002336, upper bound: 0.0002057
time: 0.49 seconds

## BFS RS instance: RS_RSZ2_RSZ1_RSZ2_RSZ1

### Backsubstitution after applying RS history:
0: 0.0035972, 0.0041150, 0.0035972, 0.0041150, -0.0001352, 0.0001319
1: 0.0018420, 0.0019168, 0.0018420, 0.0019168, -0.0000195, 0.0000191
2: 0.0120848, 0.0123711, 0.0120848, 0.0123711, -0.0000729, 0.0000747
3: -0.0021818, -0.0018857, -0.0021818, -0.0018857, -0.0000754, 0.0000773
4: -0.0019956, -0.0016750, -0.0019956, -0.0016750, -0.0000837, 0.0000816
5: 0.0056985, 0.0060018, 0.0056985, 0.0060018, -0.0000773, 0.0000792
6: 0.0003095, 0.0015131, 0.0003095, 0.0015131, -0.0003065, 0.0003142
7: -0.0046174, -0.0029782, -0.0046174, -0.0029782, -0.0004278, 0.0004175
8: 0.9859612, 0.9871160, 0.9859612, 0.9871160, -0.0003014, 0.0002941
9: -0.0041920, -0.0031438, -0.0041920, -0.0031438, -0.0002669, 0.0002736

### Unstable ReLU Count (Linear/Conv2D Layers)
- layer_idx=0, type=LayerType.Linear, total=256, inp1_unstable=4, inp2_unstable=4, delta_unstable=252
- layer_idx=2, type=LayerType.Linear, total=256, inp1_unstable=4, inp2_unstable=4, delta_unstable=256
- layer_idx=4, type=LayerType.Linear, total=256, inp1_unstable=0, inp2_unstable=0, delta_unstable=256
- layer_idx=6, type=LayerType.Linear, total=256, inp1_unstable=0, inp2_unstable=0, delta_unstable=256
- layer_idx=8, type=LayerType.Linear, total=10, inp1_unstable=0, inp2_unstable=0, delta_unstable=10

Time for backsubstitution: 1.38 seconds

### RS candidates at layer 3
type: RSZ, layer: 3, pos: 83
type: RSZ, layer: 3, pos: 240
type: RSZ, layer: 3, pos: 49
type: RSZ, layer: 3, pos: 34

Time for candidate selection: 0.00 seconds

### Candidate
type: RSZ, layer: 3, pos: 83

### Relational analysis RSZ of RS_RSZ2_RSZ1_RSZ2_RSZ1_RSZ1

#### Relational analysis RSZ result of RS_RSZ2_RSZ1_RSZ2_RSZ1_RSZ1
Status: Status.UNKNOWN
Output dim: 8, lower bound: -0.0002057, upper bound: 0.0002335
time: 0.53 seconds

### Relational analysis RSZ of RS_RSZ2_RSZ1_RSZ2_RSZ1_RSZ2

#### Relational analysis RSZ result of RS_RSZ2_RSZ1_RSZ2_RSZ1_RSZ2
Status: Status.UNKNOWN
Output dim: 8, lower bound: -0.0002327, upper bound: 0.0001948
time: 0.54 seconds

## BFS RS instance: RS_RSZ2_RSZ1_RSZ2_RSZ2

### Backsubstitution after applying RS history:
0: 0.0035972, 0.0041150, 0.0035972, 0.0041150, -0.0001375, 0.0001311
1: 0.0018420, 0.0019168, 0.0018420, 0.0019168, -0.0000199, 0.0000189
2: 0.0120848, 0.0123711, 0.0120848, 0.0123711, -0.0000725, 0.0000760
3: -0.0021818, -0.0018857, -0.0021818, -0.0018857, -0.0000750, 0.0000786
4: -0.0019956, -0.0016750, -0.0019956, -0.0016750, -0.0000851, 0.0000812
5: 0.0056985, 0.0060018, 0.0056985, 0.0060018, -0.0000768, 0.0000805
6: 0.0003095, 0.0015131, 0.0003095, 0.0015131, -0.0003048, 0.0003196
7: -0.0046174, -0.0029782, -0.0046174, -0.0029782, -0.0004352, 0.0004151
8: 0.9859612, 0.9871160, 0.9859612, 0.9871160, -0.0003066, 0.0002924
9: -0.0041920, -0.0031438, -0.0041920, -0.0031438, -0.0002654, 0.0002783

### Unstable ReLU Count (Linear/Conv2D Layers)
- layer_idx=0, type=LayerType.Linear, total=256, inp1_unstable=4, inp2_unstable=4, delta_unstable=252
- layer_idx=2, type=LayerType.Linear, total=256, inp1_unstable=4, inp2_unstable=4, delta_unstable=256
- layer_idx=4, type=LayerType.Linear, total=256, inp1_unstable=0, inp2_unstable=0, delta_unstable=256
- layer_idx=6, type=LayerType.Linear, total=256, inp1_unstable=0, inp2_unstable=0, delta_unstable=256
- layer_idx=8, type=LayerType.Linear, total=10, inp1_unstable=0, inp2_unstable=0, delta_unstable=10

Time for backsubstitution: 1.39 seconds

### RS candidates at layer 3
type: RSZ, layer: 3, pos: 34
type: RSZ, layer: 3, pos: 240
type: RSZ, layer: 3, pos: 83
type: RSZ, layer: 3, pos: 49

Time for candidate selection: 0.00 seconds

### Candidate
type: RSZ, layer: 3, pos: 34

### Relational analysis RSZ of RS_RSZ2_RSZ1_RSZ2_RSZ2_RSZ1

#### Relational analysis RSZ result of RS_RSZ2_RSZ1_RSZ2_RSZ2_RSZ1
Status: Status.UNKNOWN
Output dim: 8, lower bound: -0.0002447, upper bound: 0.0002295
time: 0.48 seconds

### Relational analysis RSZ of RS_RSZ2_RSZ1_RSZ2_RSZ2_RSZ2

#### Relational analysis RSZ result of RS_RSZ2_RSZ1_RSZ2_RSZ2_RSZ2
Status: Status.UNKNOWN
Output dim: 8, lower bound: -0.0002416, upper bound: 0.0002325
time: 0.48 seconds

## BFS RS instance: RS_RSZ2_RSZ2_RSZ1_RSZ1

### Backsubstitution after applying RS history:
0: 0.0035972, 0.0041150, 0.0035972, 0.0041150, -0.0001335, 0.0001357
1: 0.0018420, 0.0019168, 0.0018420, 0.0019168, -0.0000193, 0.0000196
2: 0.0120848, 0.0123711, 0.0120848, 0.0123711, -0.0000750, 0.0000738
3: -0.0021818, -0.0018857, -0.0021818, -0.0018857, -0.0000776, 0.0000763
4: -0.0019956, -0.0016750, -0.0019956, -0.0016750, -0.0000826, 0.0000840
5: 0.0056985, 0.0060018, 0.0056985, 0.0060018, -0.0000795, 0.0000782
6: 0.0003095, 0.0015131, 0.0003095, 0.0015131, -0.0003153, 0.0003103
7: -0.0046174, -0.0029782, -0.0046174, -0.0029782, -0.0004226, 0.0004294
8: 0.9859612, 0.9871160, 0.9859612, 0.9871160, -0.0002977, 0.0003025
9: -0.0041920, -0.0031438, -0.0041920, -0.0031438, -0.0002746, 0.0002702

### Unstable ReLU Count (Linear/Conv2D Layers)
- layer_idx=0, type=LayerType.Linear, total=256, inp1_unstable=4, inp2_unstable=4, delta_unstable=252
- layer_idx=2, type=LayerType.Linear, total=256, inp1_unstable=4, inp2_unstable=4, delta_unstable=256
- layer_idx=4, type=LayerType.Linear, total=256, inp1_unstable=0, inp2_unstable=0, delta_unstable=256
- layer_idx=6, type=LayerType.Linear, total=256, inp1_unstable=0, inp2_unstable=0, delta_unstable=256
- layer_idx=8, type=LayerType.Linear, total=10, inp1_unstable=0, inp2_unstable=0, delta_unstable=10

Time for backsubstitution: 1.42 seconds

### RS candidates at layer 3
type: RSZ, layer: 3, pos: 83
type: RSZ, layer: 3, pos: 240
type: RSZ, layer: 3, pos: 49
type: RSZ, layer: 3, pos: 34

Time for candidate selection: 0.00 seconds

### Candidate
type: RSZ, layer: 3, pos: 83

### Relational analysis RSZ of RS_RSZ2_RSZ2_RSZ1_RSZ1_RSZ1

#### Relational analysis RSZ result of RS_RSZ2_RSZ2_RSZ1_RSZ1_RSZ1
Status: Status.UNKNOWN
Output dim: 8, lower bound: -0.0001865, upper bound: 0.0002381
time: 0.50 seconds

### Relational analysis RSZ of RS_RSZ2_RSZ2_RSZ1_RSZ1_RSZ2

#### Relational analysis RSZ result of RS_RSZ2_RSZ2_RSZ1_RSZ1_RSZ2
Status: Status.UNKNOWN
Output dim: 8, lower bound: -0.0002296, upper bound: 0.0002136
time: 0.48 seconds

## BFS RS instance: RS_RSZ2_RSZ2_RSZ1_RSZ2

### Backsubstitution after applying RS history:
0: 0.0035972, 0.0041150, 0.0035972, 0.0041150, -0.0001351, 0.0001321
1: 0.0018420, 0.0019168, 0.0018420, 0.0019168, -0.0000195, 0.0000191
2: 0.0120848, 0.0123711, 0.0120848, 0.0123711, -0.0000730, 0.0000747
3: -0.0021818, -0.0018857, -0.0021818, -0.0018857, -0.0000755, 0.0000772
4: -0.0019956, -0.0016750, -0.0019956, -0.0016750, -0.0000836, 0.0000818
5: 0.0056985, 0.0060018, 0.0056985, 0.0060018, -0.0000774, 0.0000791
6: 0.0003095, 0.0015131, 0.0003095, 0.0015131, -0.0003070, 0.0003139
7: -0.0046174, -0.0029782, -0.0046174, -0.0029782, -0.0004275, 0.0004181
8: 0.9859612, 0.9871160, 0.9859612, 0.9871160, -0.0003012, 0.0002945
9: -0.0041920, -0.0031438, -0.0041920, -0.0031438, -0.0002673, 0.0002734

### Unstable ReLU Count (Linear/Conv2D Layers)
- layer_idx=0, type=LayerType.Linear, total=256, inp1_unstable=4, inp2_unstable=4, delta_unstable=252
- layer_idx=2, type=LayerType.Linear, total=256, inp1_unstable=4, inp2_unstable=4, delta_unstable=256
- layer_idx=4, type=LayerType.Linear, total=256, inp1_unstable=0, inp2_unstable=0, delta_unstable=256
- layer_idx=6, type=LayerType.Linear, total=256, inp1_unstable=0, inp2_unstable=0, delta_unstable=256
- layer_idx=8, type=LayerType.Linear, total=10, inp1_unstable=0, inp2_unstable=0, delta_unstable=10

Time for backsubstitution: 1.44 seconds

### RS candidates at layer 3
type: RSZ, layer: 3, pos: 240
type: RSZ, layer: 3, pos: 49
type: RSZ, layer: 3, pos: 83
type: RSZ, layer: 3, pos: 34

Time for candidate selection: 0.00 seconds

### Candidate
type: RSZ, layer: 3, pos: 240

### Relational analysis RSZ of RS_RSZ2_RSZ2_RSZ1_RSZ2_RSZ1

#### Relational analysis RSZ result of RS_RSZ2_RSZ2_RSZ1_RSZ2_RSZ1
Status: Status.UNKNOWN
Output dim: 8, lower bound: -0.0002530, upper bound: 0.0002493
time: 0.48 seconds

### Relational analysis RSZ of RS_RSZ2_RSZ2_RSZ1_RSZ2_RSZ2

#### Relational analysis RSZ result of RS_RSZ2_RSZ2_RSZ1_RSZ2_RSZ2
Status: Status.UNKNOWN
Output dim: 8, lower bound: -0.0002530, upper bound: 0.0002471
time: 0.49 seconds

## BFS RS instance: RS_RSZ2_RSZ2_RSZ2_RSZ1

### Backsubstitution after applying RS history:
0: 0.0035972, 0.0041150, 0.0035972, 0.0041150, -0.0001340, 0.0001345
1: 0.0018420, 0.0019168, 0.0018420, 0.0019168, -0.0000194, 0.0000194
2: 0.0120848, 0.0123711, 0.0120848, 0.0123711, -0.0000743, 0.0000741
3: -0.0021818, -0.0018857, -0.0021818, -0.0018857, -0.0000769, 0.0000766
4: -0.0019956, -0.0016750, -0.0019956, -0.0016750, -0.0000830, 0.0000832
5: 0.0056985, 0.0060018, 0.0056985, 0.0060018, -0.0000788, 0.0000785
6: 0.0003095, 0.0015131, 0.0003095, 0.0015131, -0.0003125, 0.0003115
7: -0.0046174, -0.0029782, -0.0046174, -0.0029782, -0.0004242, 0.0004256
8: 0.9859612, 0.9871160, 0.9859612, 0.9871160, -0.0002988, 0.0002998
9: -0.0041920, -0.0031438, -0.0041920, -0.0031438, -0.0002722, 0.0002712

### Unstable ReLU Count (Linear/Conv2D Layers)
- layer_idx=0, type=LayerType.Linear, total=256, inp1_unstable=4, inp2_unstable=4, delta_unstable=252
- layer_idx=2, type=LayerType.Linear, total=256, inp1_unstable=4, inp2_unstable=4, delta_unstable=256
- layer_idx=4, type=LayerType.Linear, total=256, inp1_unstable=0, inp2_unstable=0, delta_unstable=256
- layer_idx=6, type=LayerType.Linear, total=256, inp1_unstable=0, inp2_unstable=0, delta_unstable=256
- layer_idx=8, type=LayerType.Linear, total=10, inp1_unstable=0, inp2_unstable=0, delta_unstable=10

Time for backsubstitution: 1.50 seconds

### RS candidates at layer 3
type: RSZ, layer: 3, pos: 240
type: RSZ, layer: 3, pos: 34
type: RSZ, layer: 3, pos: 83
type: RSZ, layer: 3, pos: 49

Time for candidate selection: 0.00 seconds

### Candidate
type: RSZ, layer: 3, pos: 240

### Relational analysis RSZ of RS_RSZ2_RSZ2_RSZ2_RSZ1_RSZ1

#### Relational analysis RSZ result of RS_RSZ2_RSZ2_RSZ2_RSZ1_RSZ1
Status: Status.UNKNOWN
Output dim: 8, lower bound: -0.0002472, upper bound: 0.0002532
time: 0.48 seconds

### Relational analysis RSZ of RS_RSZ2_RSZ2_RSZ2_RSZ1_RSZ2

#### Relational analysis RSZ result of RS_RSZ2_RSZ2_RSZ2_RSZ1_RSZ2
Status: Status.UNKNOWN
Output dim: 8, lower bound: -0.0002493, upper bound: 0.0002532
time: 0.47 seconds

## BFS RS instance: RS_RSZ2_RSZ2_RSZ2_RSZ2

### Backsubstitution after applying RS history:
0: 0.0035972, 0.0041150, 0.0035972, 0.0041150, -0.0001373, 0.0001325
1: 0.0018420, 0.0019168, 0.0018420, 0.0019168, -0.0000198, 0.0000191
2: 0.0120848, 0.0123711, 0.0120848, 0.0123711, -0.0000733, 0.0000759
3: -0.0021818, -0.0018857, -0.0021818, -0.0018857, -0.0000758, 0.0000785
4: -0.0019956, -0.0016750, -0.0019956, -0.0016750, -0.0000850, 0.0000820
5: 0.0056985, 0.0060018, 0.0056985, 0.0060018, -0.0000776, 0.0000804
6: 0.0003095, 0.0015131, 0.0003095, 0.0015131, -0.0003080, 0.0003192
7: -0.0046174, -0.0029782, -0.0046174, -0.0029782, -0.0004347, 0.0004195
8: 0.9859612, 0.9871160, 0.9859612, 0.9871160, -0.0003062, 0.0002955
9: -0.0041920, -0.0031438, -0.0041920, -0.0031438, -0.0002683, 0.0002779

### Unstable ReLU Count (Linear/Conv2D Layers)
- layer_idx=0, type=LayerType.Linear, total=256, inp1_unstable=4, inp2_unstable=4, delta_unstable=252
- layer_idx=2, type=LayerType.Linear, total=256, inp1_unstable=4, inp2_unstable=4, delta_unstable=256
- layer_idx=4, type=LayerType.Linear, total=256, inp1_unstable=0, inp2_unstable=0, delta_unstable=256
- layer_idx=6, type=LayerType.Linear, total=256, inp1_unstable=0, inp2_unstable=0, delta_unstable=256
- layer_idx=8, type=LayerType.Linear, total=10, inp1_unstable=0, inp2_unstable=0, delta_unstable=10

Time for backsubstitution: 1.40 seconds

### RS candidates at layer 3
type: RSZ, layer: 3, pos: 240
type: RSZ, layer: 3, pos: 83
type: RSZ, layer: 3, pos: 49
type: RSZ, layer: 3, pos: 34

Time for candidate selection: 0.00 seconds

### Candidate
type: RSZ, layer: 3, pos: 240

### Relational analysis RSZ of RS_RSZ2_RSZ2_RSZ2_RSZ2_RSZ1

#### Relational analysis RSZ result of RS_RSZ2_RSZ2_RSZ2_RSZ2_RSZ1
Status: Status.UNKNOWN
Output dim: 8, lower bound: -0.0002565, upper bound: 0.0002463
time: 0.49 seconds

### Relational analysis RSZ of RS_RSZ2_RSZ2_RSZ2_RSZ2_RSZ2

#### Relational analysis RSZ result of RS_RSZ2_RSZ2_RSZ2_RSZ2_RSZ2
Status: Status.UNKNOWN
Output dim: 8, lower bound: -0.0002568, upper bound: 0.0002405
time: 0.48 seconds

## Summary of splitting (split count: 4)
- Time for RS candidates: 2.38 seconds
RS_RSZ1_RSZ1_RSZ1_RSZ1_RSZ1, status: Status.UNKNOWN, split count: 5, time: 2.38
Output dim: 8, lower bound: -0.0002338, upper bound: 0.0002422
RS_RSZ1_RSZ1_RSZ1_RSZ1_RSZ2, status: Status.UNKNOWN, split count: 5, time: 2.38
Output dim: 8, lower bound: -0.0002296, upper bound: 0.0002446
RS_RSZ1_RSZ1_RSZ1_RSZ2_RSZ1, status: Status.UNKNOWN, split count: 5, time: 2.38
Output dim: 8, lower bound: -0.0002411, upper bound: 0.0002335
RS_RSZ1_RSZ1_RSZ1_RSZ2_RSZ2, status: Status.UNKNOWN, split count: 5, time: 2.38
Output dim: 8, lower bound: -0.0002388, upper bound: 0.0002369
RS_RSZ1_RSZ1_RSZ2_RSZ1_RSZ1, status: Status.UNKNOWN, split count: 5, time: 2.38
Output dim: 8, lower bound: -0.0001693, upper bound: 0.0002284
RS_RSZ1_RSZ1_RSZ2_RSZ1_RSZ2, status: Status.UNKNOWN, split count: 5, time: 2.38
Output dim: 8, lower bound: -0.0002277, upper bound: 0.0001689
RS_RSZ1_RSZ1_RSZ2_RSZ2_RSZ1, status: Status.UNKNOWN, split count: 5, time: 2.38
Output dim: 8, lower bound: -0.0002567, upper bound: 0.0002450
RS_RSZ1_RSZ1_RSZ2_RSZ2_RSZ2, status: Status.UNKNOWN, split count: 5, time: 2.38
Output dim: 8, lower bound: -0.0002571, upper bound: 0.0002406
RS_RSZ1_RSZ2_RSZ1_RSZ1_RSZ1, status: Status.UNKNOWN, split count: 5, time: 2.38
Output dim: 8, lower bound: -0.0001860, upper bound: 0.0002381
RS_RSZ1_RSZ2_RSZ1_RSZ1_RSZ2, status: Status.UNKNOWN, split count: 5, time: 2.38
Output dim: 8, lower bound: -0.0002296, upper bound: 0.0002136
RS_RSZ1_RSZ2_RSZ1_RSZ2_RSZ1, status: Status.UNKNOWN, split count: 5, time: 2.38
Output dim: 8, lower bound: -0.0001948, upper bound: 0.0002327
RS_RSZ1_RSZ2_RSZ1_RSZ2_RSZ2, status: Status.UNKNOWN, split count: 5, time: 2.38
Output dim: 8, lower bound: -0.0002335, upper bound: 0.0002057
RS_RSZ1_RSZ2_RSZ2_RSZ1_RSZ1, status: Status.UNKNOWN, split count: 5, time: 2.38
Output dim: 8, lower bound: -0.0002532, upper bound: 0.0002493
RS_RSZ1_RSZ2_RSZ2_RSZ1_RSZ2, status: Status.UNKNOWN, split count: 5, time: 2.38
Output dim: 8, lower bound: -0.0002532, upper bound: 0.0002471
RS_RSZ1_RSZ2_RSZ2_RSZ2_RSZ1, status: Status.UNKNOWN, split count: 5, time: 2.38
Output dim: 8, lower bound: -0.0001798, upper bound: 0.0002223
RS_RSZ1_RSZ2_RSZ2_RSZ2_RSZ2, status: Status.UNKNOWN, split count: 5, time: 2.38
Output dim: 8, lower bound: -0.0002344, upper bound: 0.0001603
RS_RSZ2_RSZ1_RSZ1_RSZ1_RSZ1, status: Status.UNKNOWN, split count: 5, time: 2.38
Output dim: 8, lower bound: -0.0001873, upper bound: 0.0002381
RS_RSZ2_RSZ1_RSZ1_RSZ1_RSZ2, status: Status.UNKNOWN, split count: 5, time: 2.38
Output dim: 8, lower bound: -0.0002302, upper bound: 0.0002147
RS_RSZ2_RSZ1_RSZ1_RSZ2_RSZ1, status: Status.UNKNOWN, split count: 5, time: 2.38
Output dim: 8, lower bound: -0.0001974, upper bound: 0.0002327
RS_RSZ2_RSZ1_RSZ1_RSZ2_RSZ2, status: Status.UNKNOWN, split count: 5, time: 2.38
Output dim: 8, lower bound: -0.0002336, upper bound: 0.0002057
RS_RSZ2_RSZ1_RSZ2_RSZ1_RSZ1, status: Status.UNKNOWN, split count: 5, time: 2.38
Output dim: 8, lower bound: -0.0002057, upper bound: 0.0002335
RS_RSZ2_RSZ1_RSZ2_RSZ1_RSZ2, status: Status.UNKNOWN, split count: 5, time: 2.38
Output dim: 8, lower bound: -0.0002327, upper bound: 0.0001948
RS_RSZ2_RSZ1_RSZ2_RSZ2_RSZ1, status: Status.UNKNOWN, split count: 5, time: 2.38
Output dim: 8, lower bound: -0.0002447, upper bound: 0.0002295
RS_RSZ2_RSZ1_RSZ2_RSZ2_RSZ2, status: Status.UNKNOWN, split count: 5, time: 2.38
Output dim: 8, lower bound: -0.0002416, upper bound: 0.0002325
RS_RSZ2_RSZ2_RSZ1_RSZ1_RSZ1, status: Status.UNKNOWN, split count: 5, time: 2.38
Output dim: 8, lower bound: -0.0001865, upper bound: 0.0002381
RS_RSZ2_RSZ2_RSZ1_RSZ1_RSZ2, status: Status.UNKNOWN, split count: 5, time: 2.38
Output dim: 8, lower bound: -0.0002296, upper bound: 0.0002136
RS_RSZ2_RSZ2_RSZ1_RSZ2_RSZ1, status: Status.UNKNOWN, split count: 5, time: 2.38
Output dim: 8, lower bound: -0.0002530, upper bound: 0.0002493
RS_RSZ2_RSZ2_RSZ1_RSZ2_RSZ2, status: Status.UNKNOWN, split count: 5, time: 2.38
Output dim: 8, lower bound: -0.0002530, upper bound: 0.0002471
RS_RSZ2_RSZ2_RSZ2_RSZ1_RSZ1, status: Status.UNKNOWN, split count: 5, time: 2.38
Output dim: 8, lower bound: -0.0002472, upper bound: 0.0002532
RS_RSZ2_RSZ2_RSZ2_RSZ1_RSZ2, status: Status.UNKNOWN, split count: 5, time: 2.38
Output dim: 8, lower bound: -0.0002493, upper bound: 0.0002532
RS_RSZ2_RSZ2_RSZ2_RSZ2_RSZ1, status: Status.UNKNOWN, split count: 5, time: 2.38
Output dim: 8, lower bound: -0.0002565, upper bound: 0.0002463
RS_RSZ2_RSZ2_RSZ2_RSZ2_RSZ2, status: Status.UNKNOWN, split count: 5, time: 2.38
Output dim: 8, lower bound: -0.0002568, upper bound: 0.0002405

## BFS RS instance: RS_RSZ1_RSZ1_RSZ1_RSZ1_RSZ1

### Backsubstitution after applying RS history:
0: 0.0035972, 0.0041150, 0.0035972, 0.0041150, -0.0001166, 0.0001203
1: 0.0018420, 0.0019168, 0.0018420, 0.0019168, -0.0000168, 0.0000174
2: 0.0120848, 0.0123711, 0.0120848, 0.0123711, -0.0000665, 0.0000644
3: -0.0021818, -0.0018857, -0.0021818, -0.0018857, -0.0000688, 0.0000666
4: -0.0019956, -0.0016750, -0.0019956, -0.0016750, -0.0000721, 0.0000745
5: 0.0056985, 0.0060018, 0.0056985, 0.0060018, -0.0000705, 0.0000683
6: 0.0003095, 0.0015131, 0.0003095, 0.0015131, -0.0002796, 0.0002709
7: -0.0046174, -0.0029782, -0.0046174, -0.0029782, -0.0003689, 0.0003808
8: 0.9859612, 0.9871160, 0.9859612, 0.9871160, -0.0002599, 0.0002683
9: -0.0041920, -0.0031438, -0.0041920, -0.0031438, -0.0002435, 0.0002359

### Unstable ReLU Count (Linear/Conv2D Layers)
- layer_idx=0, type=LayerType.Linear, total=256, inp1_unstable=4, inp2_unstable=4, delta_unstable=252
- layer_idx=2, type=LayerType.Linear, total=256, inp1_unstable=4, inp2_unstable=4, delta_unstable=255
- layer_idx=4, type=LayerType.Linear, total=256, inp1_unstable=0, inp2_unstable=0, delta_unstable=256
- layer_idx=6, type=LayerType.Linear, total=256, inp1_unstable=0, inp2_unstable=0, delta_unstable=256
- layer_idx=8, type=LayerType.Linear, total=10, inp1_unstable=0, inp2_unstable=0, delta_unstable=10

Time for backsubstitution: 1.43 seconds

### RS candidates at layer 3
type: RSZ, layer: 3, pos: 83
type: RSZ, layer: 3, pos: 240
type: RSZ, layer: 3, pos: 49

Time for candidate selection: 0.00 seconds

### Candidate
type: RSZ, layer: 3, pos: 83

### Relational analysis RSZ of RS_RSZ1_RSZ1_RSZ1_RSZ1_RSZ1_RSZ1

#### Relational analysis RSZ result of RS_RSZ1_RSZ1_RSZ1_RSZ1_RSZ1_RSZ1
Status: Status.UNKNOWN
Output dim: 8, lower bound: -0.0001741, upper bound: 0.0002229
time: 0.53 seconds

### Relational analysis RSZ of RS_RSZ1_RSZ1_RSZ1_RSZ1_RSZ1_RSZ2

#### Relational analysis RSZ result of RS_RSZ1_RSZ1_RSZ1_RSZ1_RSZ1_RSZ2
Status: Status.UNKNOWN
Output dim: 8, lower bound: -0.0002179, upper bound: 0.0002029
time: 0.53 seconds

## BFS RS instance: RS_RSZ1_RSZ1_RSZ1_RSZ1_RSZ2

### Backsubstitution after applying RS history:
0: 0.0035972, 0.0041150, 0.0035972, 0.0041150, -0.0001148, 0.0001213
1: 0.0018420, 0.0019168, 0.0018420, 0.0019168, -0.0000166, 0.0000175
2: 0.0120848, 0.0123711, 0.0120848, 0.0123711, -0.0000671, 0.0000635
3: -0.0021818, -0.0018857, -0.0021818, -0.0018857, -0.0000694, 0.0000657
4: -0.0019956, -0.0016750, -0.0019956, -0.0016750, -0.0000711, 0.0000751
5: 0.0056985, 0.0060018, 0.0056985, 0.0060018, -0.0000711, 0.0000673
6: 0.0003095, 0.0015131, 0.0003095, 0.0015131, -0.0002820, 0.0002669
7: -0.0046174, -0.0029782, -0.0046174, -0.0029782, -0.0003635, 0.0003841
8: 0.9859612, 0.9871160, 0.9859612, 0.9871160, -0.0002560, 0.0002705
9: -0.0041920, -0.0031438, -0.0041920, -0.0031438, -0.0002456, 0.0002324

### Unstable ReLU Count (Linear/Conv2D Layers)
- layer_idx=0, type=LayerType.Linear, total=256, inp1_unstable=4, inp2_unstable=4, delta_unstable=252
- layer_idx=2, type=LayerType.Linear, total=256, inp1_unstable=4, inp2_unstable=4, delta_unstable=255
- layer_idx=4, type=LayerType.Linear, total=256, inp1_unstable=0, inp2_unstable=0, delta_unstable=256
- layer_idx=6, type=LayerType.Linear, total=256, inp1_unstable=0, inp2_unstable=0, delta_unstable=256
- layer_idx=8, type=LayerType.Linear, total=10, inp1_unstable=0, inp2_unstable=0, delta_unstable=10

Time for backsubstitution: 1.38 seconds

### RS candidates at layer 3
type: RSZ, layer: 3, pos: 49
type: RSZ, layer: 3, pos: 240
type: RSZ, layer: 3, pos: 83

Time for candidate selection: 0.00 seconds

### Candidate
type: RSZ, layer: 3, pos: 49

### Relational analysis RSZ of RS_RSZ1_RSZ1_RSZ1_RSZ1_RSZ2_RSZ1

#### Relational analysis RSZ result of RS_RSZ1_RSZ1_RSZ1_RSZ1_RSZ2_RSZ1
Status: Status.UNKNOWN
Output dim: 8, lower bound: -0.0001577, upper bound: 0.0002224
time: 0.51 seconds

### Relational analysis RSZ of RS_RSZ1_RSZ1_RSZ1_RSZ1_RSZ2_RSZ2

#### Relational analysis RSZ result of RS_RSZ1_RSZ1_RSZ1_RSZ1_RSZ2_RSZ2
Status: Status.VERIFIED
Output dim: 8, lower bound: -0.0002025, upper bound: 0.0001562
time: 0.51 seconds

## BFS RS instance: RS_RSZ1_RSZ1_RSZ1_RSZ2_RSZ1

### Backsubstitution after applying RS history:
0: 0.0035972, 0.0041150, 0.0035972, 0.0041150, -0.0001185, 0.0001170
1: 0.0018420, 0.0019168, 0.0018420, 0.0019168, -0.0000171, 0.0000169
2: 0.0120848, 0.0123711, 0.0120848, 0.0123711, -0.0000647, 0.0000655
3: -0.0021818, -0.0018857, -0.0021818, -0.0018857, -0.0000669, 0.0000677
4: -0.0019956, -0.0016750, -0.0019956, -0.0016750, -0.0000733, 0.0000725
5: 0.0056985, 0.0060018, 0.0056985, 0.0060018, -0.0000686, 0.0000694
6: 0.0003095, 0.0015131, 0.0003095, 0.0015131, -0.0002721, 0.0002754
7: -0.0046174, -0.0029782, -0.0046174, -0.0029782, -0.0003750, 0.0003705
8: 0.9859612, 0.9871160, 0.9859612, 0.9871160, -0.0002642, 0.0002610
9: -0.0041920, -0.0031438, -0.0041920, -0.0031438, -0.0002369, 0.0002398

### Unstable ReLU Count (Linear/Conv2D Layers)
- layer_idx=0, type=LayerType.Linear, total=256, inp1_unstable=4, inp2_unstable=4, delta_unstable=252
- layer_idx=2, type=LayerType.Linear, total=256, inp1_unstable=4, inp2_unstable=4, delta_unstable=255
- layer_idx=4, type=LayerType.Linear, total=256, inp1_unstable=0, inp2_unstable=0, delta_unstable=256
- layer_idx=6, type=LayerType.Linear, total=256, inp1_unstable=0, inp2_unstable=0, delta_unstable=256
- layer_idx=8, type=LayerType.Linear, total=10, inp1_unstable=0, inp2_unstable=0, delta_unstable=10

Time for backsubstitution: 1.37 seconds

### RS candidates at layer 3
type: RSZ, layer: 3, pos: 240
type: RSZ, layer: 3, pos: 49
type: RSZ, layer: 3, pos: 83

Time for candidate selection: 0.00 seconds

### Candidate
type: RSZ, layer: 3, pos: 240

### Relational analysis RSZ of RS_RSZ1_RSZ1_RSZ1_RSZ2_RSZ1_RSZ1

#### Relational analysis RSZ result of RS_RSZ1_RSZ1_RSZ1_RSZ2_RSZ1_RSZ1
Status: Status.UNKNOWN
Output dim: 8, lower bound: -0.0002409, upper bound: 0.0002333
time: 0.51 seconds

### Relational analysis RSZ of RS_RSZ1_RSZ1_RSZ1_RSZ2_RSZ1_RSZ2

#### Relational analysis RSZ result of RS_RSZ1_RSZ1_RSZ1_RSZ2_RSZ1_RSZ2
Status: Status.UNKNOWN
Output dim: 8, lower bound: -0.0002409, upper bound: 0.0002333
time: 0.48 seconds

## BFS RS instance: RS_RSZ1_RSZ1_RSZ1_RSZ2_RSZ2

### Backsubstitution after applying RS history:
0: 0.0035972, 0.0041150, 0.0035972, 0.0041150, -0.0001176, 0.0001180
1: 0.0018420, 0.0019168, 0.0018420, 0.0019168, -0.0000170, 0.0000171
2: 0.0120848, 0.0123711, 0.0120848, 0.0123711, -0.0000653, 0.0000650
3: -0.0021818, -0.0018857, -0.0021818, -0.0018857, -0.0000675, 0.0000672
4: -0.0019956, -0.0016750, -0.0019956, -0.0016750, -0.0000728, 0.0000731
5: 0.0056985, 0.0060018, 0.0056985, 0.0060018, -0.0000691, 0.0000689
6: 0.0003095, 0.0015131, 0.0003095, 0.0015131, -0.0002743, 0.0002733
7: -0.0046174, -0.0029782, -0.0046174, -0.0029782, -0.0003722, 0.0003736
8: 0.9859612, 0.9871160, 0.9859612, 0.9871160, -0.0002622, 0.0002632
9: -0.0041920, -0.0031438, -0.0041920, -0.0031438, -0.0002389, 0.0002380

### Unstable ReLU Count (Linear/Conv2D Layers)
- layer_idx=0, type=LayerType.Linear, total=256, inp1_unstable=4, inp2_unstable=4, delta_unstable=252
- layer_idx=2, type=LayerType.Linear, total=256, inp1_unstable=4, inp2_unstable=4, delta_unstable=255
- layer_idx=4, type=LayerType.Linear, total=256, inp1_unstable=0, inp2_unstable=0, delta_unstable=256
- layer_idx=6, type=LayerType.Linear, total=256, inp1_unstable=0, inp2_unstable=0, delta_unstable=256
- layer_idx=8, type=LayerType.Linear, total=10, inp1_unstable=0, inp2_unstable=0, delta_unstable=10

Time for backsubstitution: 1.48 seconds

### RS candidates at layer 3
type: RSZ, layer: 3, pos: 49
type: RSZ, layer: 3, pos: 83
type: RSZ, layer: 3, pos: 240

Time for candidate selection: 0.00 seconds

### Candidate
type: RSZ, layer: 3, pos: 49

### Relational analysis RSZ of RS_RSZ1_RSZ1_RSZ1_RSZ2_RSZ2_RSZ1

#### Relational analysis RSZ result of RS_RSZ1_RSZ1_RSZ1_RSZ2_RSZ2_RSZ1
Status: Status.VERIFIED
Output dim: 8, lower bound: -0.0001633, upper bound: 0.0002153
time: 0.48 seconds

### Relational analysis RSZ of RS_RSZ1_RSZ1_RSZ1_RSZ2_RSZ2_RSZ2

#### Relational analysis RSZ result of RS_RSZ1_RSZ1_RSZ1_RSZ2_RSZ2_RSZ2
Status: Status.VERIFIED
Output dim: 8, lower bound: -0.0002126, upper bound: 0.0001501
time: 0.48 seconds

## BFS RS instance: RS_RSZ1_RSZ1_RSZ2_RSZ1_RSZ1

### Backsubstitution after applying RS history:
0: 0.0035972, 0.0041150, 0.0035972, 0.0041150, -0.0001153, 0.0001280
1: 0.0018420, 0.0019168, 0.0018420, 0.0019168, -0.0000167, 0.0000185
2: 0.0120848, 0.0123711, 0.0120848, 0.0123711, -0.0000708, 0.0000637
3: -0.0021818, -0.0018857, -0.0021818, -0.0018857, -0.0000732, 0.0000659
4: -0.0019956, -0.0016750, -0.0019956, -0.0016750, -0.0000714, 0.0000793
5: 0.0056985, 0.0060018, 0.0056985, 0.0060018, -0.0000750, 0.0000675
6: 0.0003095, 0.0015131, 0.0003095, 0.0015131, -0.0002976, 0.0002680
7: -0.0046174, -0.0029782, -0.0046174, -0.0029782, -0.0003649, 0.0004053
8: 0.9859612, 0.9871160, 0.9859612, 0.9871160, -0.0002571, 0.0002855
9: -0.0041920, -0.0031438, -0.0041920, -0.0031438, -0.0002592, 0.0002334

### Unstable ReLU Count (Linear/Conv2D Layers)
- layer_idx=0, type=LayerType.Linear, total=256, inp1_unstable=4, inp2_unstable=4, delta_unstable=252
- layer_idx=2, type=LayerType.Linear, total=256, inp1_unstable=4, inp2_unstable=4, delta_unstable=255
- layer_idx=4, type=LayerType.Linear, total=256, inp1_unstable=0, inp2_unstable=0, delta_unstable=256
- layer_idx=6, type=LayerType.Linear, total=256, inp1_unstable=0, inp2_unstable=0, delta_unstable=256
- layer_idx=8, type=LayerType.Linear, total=10, inp1_unstable=0, inp2_unstable=0, delta_unstable=10

Time for backsubstitution: 1.37 seconds

### RS candidates at layer 3
type: RSZ, layer: 3, pos: 240
type: RSZ, layer: 3, pos: 34
type: RSZ, layer: 3, pos: 83

Time for candidate selection: 0.00 seconds

### Candidate
type: RSZ, layer: 3, pos: 240

### Relational analysis RSZ of RS_RSZ1_RSZ1_RSZ2_RSZ1_RSZ1_RSZ1

#### Relational analysis RSZ result of RS_RSZ1_RSZ1_RSZ2_RSZ1_RSZ1_RSZ1
Status: Status.UNKNOWN
Output dim: 8, lower bound: -0.0001691, upper bound: 0.0002282
time: 0.54 seconds

### Relational analysis RSZ of RS_RSZ1_RSZ1_RSZ2_RSZ1_RSZ1_RSZ2

#### Relational analysis RSZ result of RS_RSZ1_RSZ1_RSZ2_RSZ1_RSZ1_RSZ2
Status: Status.UNKNOWN
Output dim: 8, lower bound: -0.0001691, upper bound: 0.0002281
time: 0.52 seconds

## BFS RS instance: RS_RSZ1_RSZ1_RSZ2_RSZ1_RSZ2

### Backsubstitution after applying RS history:
0: 0.0035972, 0.0041150, 0.0035972, 0.0041150, -0.0001265, 0.0001183
1: 0.0018420, 0.0019168, 0.0018420, 0.0019168, -0.0000183, 0.0000171
2: 0.0120848, 0.0123711, 0.0120848, 0.0123711, -0.0000654, 0.0000700
3: -0.0021818, -0.0018857, -0.0021818, -0.0018857, -0.0000676, 0.0000724
4: -0.0019956, -0.0016750, -0.0019956, -0.0016750, -0.0000783, 0.0000732
5: 0.0056985, 0.0060018, 0.0056985, 0.0060018, -0.0000693, 0.0000741
6: 0.0003095, 0.0015131, 0.0003095, 0.0015131, -0.0002749, 0.0002941
7: -0.0046174, -0.0029782, -0.0046174, -0.0029782, -0.0004005, 0.0003744
8: 0.9859612, 0.9871160, 0.9859612, 0.9871160, -0.0002822, 0.0002637
9: -0.0041920, -0.0031438, -0.0041920, -0.0031438, -0.0002394, 0.0002561

### Unstable ReLU Count (Linear/Conv2D Layers)
- layer_idx=0, type=LayerType.Linear, total=256, inp1_unstable=4, inp2_unstable=4, delta_unstable=252
- layer_idx=2, type=LayerType.Linear, total=256, inp1_unstable=4, inp2_unstable=4, delta_unstable=255
- layer_idx=4, type=LayerType.Linear, total=256, inp1_unstable=0, inp2_unstable=0, delta_unstable=256
- layer_idx=6, type=LayerType.Linear, total=256, inp1_unstable=0, inp2_unstable=0, delta_unstable=256
- layer_idx=8, type=LayerType.Linear, total=10, inp1_unstable=0, inp2_unstable=0, delta_unstable=10

Time for backsubstitution: 1.42 seconds

### RS candidates at layer 3
type: RSZ, layer: 3, pos: 83
type: RSZ, layer: 3, pos: 34
type: RSZ, layer: 3, pos: 240

Time for candidate selection: 0.00 seconds

### Candidate
type: RSZ, layer: 3, pos: 83

### Relational analysis RSZ of RS_RSZ1_RSZ1_RSZ2_RSZ1_RSZ2_RSZ1

#### Relational analysis RSZ result of RS_RSZ1_RSZ1_RSZ2_RSZ1_RSZ2_RSZ1
Status: Status.VERIFIED
Output dim: 8, lower bound: -0.0001655, upper bound: 0.0001497
time: 0.53 seconds

### Relational analysis RSZ of RS_RSZ1_RSZ1_RSZ2_RSZ1_RSZ2_RSZ2

#### Relational analysis RSZ result of RS_RSZ1_RSZ1_RSZ2_RSZ1_RSZ2_RSZ2
Status: Status.VERIFIED
Output dim: 8, lower bound: -0.0002112, upper bound: 0.0001490
time: 0.53 seconds

## BFS RS instance: RS_RSZ1_RSZ1_RSZ2_RSZ2_RSZ1

### Backsubstitution after applying RS history:
0: 0.0035972, 0.0041150, 0.0035972, 0.0041150, -0.0001309, 0.0001289
1: 0.0018420, 0.0019168, 0.0018420, 0.0019168, -0.0000189, 0.0000186
2: 0.0120848, 0.0123711, 0.0120848, 0.0123711, -0.0000713, 0.0000724
3: -0.0021818, -0.0018857, -0.0021818, -0.0018857, -0.0000737, 0.0000749
4: -0.0019956, -0.0016750, -0.0019956, -0.0016750, -0.0000810, 0.0000798
5: 0.0056985, 0.0060018, 0.0056985, 0.0060018, -0.0000755, 0.0000767
6: 0.0003095, 0.0015131, 0.0003095, 0.0015131, -0.0002997, 0.0003043
7: -0.0046174, -0.0029782, -0.0046174, -0.0029782, -0.0004144, 0.0004081
8: 0.9859612, 0.9871160, 0.9859612, 0.9871160, -0.0002919, 0.0002875
9: -0.0041920, -0.0031438, -0.0041920, -0.0031438, -0.0002610, 0.0002650

### Unstable ReLU Count (Linear/Conv2D Layers)
- layer_idx=0, type=LayerType.Linear, total=256, inp1_unstable=4, inp2_unstable=4, delta_unstable=252
- layer_idx=2, type=LayerType.Linear, total=256, inp1_unstable=4, inp2_unstable=4, delta_unstable=255
- layer_idx=4, type=LayerType.Linear, total=256, inp1_unstable=0, inp2_unstable=0, delta_unstable=256
- layer_idx=6, type=LayerType.Linear, total=256, inp1_unstable=0, inp2_unstable=0, delta_unstable=256
- layer_idx=8, type=LayerType.Linear, total=10, inp1_unstable=0, inp2_unstable=0, delta_unstable=10

Time for backsubstitution: 1.40 seconds

### RS candidates at layer 3
type: RSZ, layer: 3, pos: 49
type: RSZ, layer: 3, pos: 83
type: RSZ, layer: 3, pos: 34

Time for candidate selection: 0.00 seconds

### Candidate
type: RSZ, layer: 3, pos: 49

### Relational analysis RSZ of RS_RSZ1_RSZ1_RSZ2_RSZ2_RSZ1_RSZ1

#### Relational analysis RSZ result of RS_RSZ1_RSZ1_RSZ2_RSZ2_RSZ1_RSZ1
Status: Status.UNKNOWN
Output dim: 8, lower bound: -0.0001748, upper bound: 0.0002200
time: 0.55 seconds

### Relational analysis RSZ of RS_RSZ1_RSZ1_RSZ2_RSZ2_RSZ1_RSZ2

#### Relational analysis RSZ result of RS_RSZ1_RSZ1_RSZ2_RSZ2_RSZ1_RSZ2
Status: Status.UNKNOWN
Output dim: 8, lower bound: -0.0002338, upper bound: 0.0001614
time: 0.54 seconds

## BFS RS instance: RS_RSZ1_RSZ1_RSZ2_RSZ2_RSZ2

### Backsubstitution after applying RS history:
0: 0.0035972, 0.0041150, 0.0035972, 0.0041150, -0.0001311, 0.0001286
1: 0.0018420, 0.0019168, 0.0018420, 0.0019168, -0.0000189, 0.0000186
2: 0.0120848, 0.0123711, 0.0120848, 0.0123711, -0.0000711, 0.0000725
3: -0.0021818, -0.0018857, -0.0021818, -0.0018857, -0.0000735, 0.0000750
4: -0.0019956, -0.0016750, -0.0019956, -0.0016750, -0.0000811, 0.0000796
5: 0.0056985, 0.0060018, 0.0056985, 0.0060018, -0.0000753, 0.0000768
6: 0.0003095, 0.0015131, 0.0003095, 0.0015131, -0.0002989, 0.0003047
7: -0.0046174, -0.0029782, -0.0046174, -0.0029782, -0.0004150, 0.0004071
8: 0.9859612, 0.9871160, 0.9859612, 0.9871160, -0.0002923, 0.0002868
9: -0.0041920, -0.0031438, -0.0041920, -0.0031438, -0.0002603, 0.0002653

### Unstable ReLU Count (Linear/Conv2D Layers)
- layer_idx=0, type=LayerType.Linear, total=256, inp1_unstable=4, inp2_unstable=4, delta_unstable=252
- layer_idx=2, type=LayerType.Linear, total=256, inp1_unstable=4, inp2_unstable=4, delta_unstable=255
- layer_idx=4, type=LayerType.Linear, total=256, inp1_unstable=0, inp2_unstable=0, delta_unstable=256
- layer_idx=6, type=LayerType.Linear, total=256, inp1_unstable=0, inp2_unstable=0, delta_unstable=256
- layer_idx=8, type=LayerType.Linear, total=10, inp1_unstable=0, inp2_unstable=0, delta_unstable=10

Time for backsubstitution: 1.47 seconds

### RS candidates at layer 3
type: RSZ, layer: 3, pos: 49
type: RSZ, layer: 3, pos: 83
type: RSZ, layer: 3, pos: 34

Time for candidate selection: 0.00 seconds

### Candidate
type: RSZ, layer: 3, pos: 49

### Relational analysis RSZ of RS_RSZ1_RSZ1_RSZ2_RSZ2_RSZ2_RSZ1

#### Relational analysis RSZ result of RS_RSZ1_RSZ1_RSZ2_RSZ2_RSZ2_RSZ1
Status: Status.VERIFIED
Output dim: 8, lower bound: -0.0001737, upper bound: 0.0002147
time: 0.53 seconds

### Relational analysis RSZ of RS_RSZ1_RSZ1_RSZ2_RSZ2_RSZ2_RSZ2

#### Relational analysis RSZ result of RS_RSZ1_RSZ1_RSZ2_RSZ2_RSZ2_RSZ2
Status: Status.UNKNOWN
Output dim: 8, lower bound: -0.0002343, upper bound: 0.0001628
time: 0.51 seconds

## BFS RS instance: RS_RSZ1_RSZ2_RSZ1_RSZ1_RSZ1

### Backsubstitution after applying RS history:
0: 0.0035972, 0.0041150, 0.0035972, 0.0041150, -0.0001232, 0.0001334
1: 0.0018420, 0.0019168, 0.0018420, 0.0019168, -0.0000178, 0.0000193
2: 0.0120848, 0.0123711, 0.0120848, 0.0123711, -0.0000738, 0.0000681
3: -0.0021818, -0.0018857, -0.0021818, -0.0018857, -0.0000763, 0.0000704
4: -0.0019956, -0.0016750, -0.0019956, -0.0016750, -0.0000762, 0.0000826
5: 0.0056985, 0.0060018, 0.0056985, 0.0060018, -0.0000782, 0.0000722
6: 0.0003095, 0.0015131, 0.0003095, 0.0015131, -0.0003101, 0.0002863
7: -0.0046174, -0.0029782, -0.0046174, -0.0029782, -0.0003899, 0.0004223
8: 0.9859612, 0.9871160, 0.9859612, 0.9871160, -0.0002747, 0.0002975
9: -0.0041920, -0.0031438, -0.0041920, -0.0031438, -0.0002700, 0.0002493

### Unstable ReLU Count (Linear/Conv2D Layers)
- layer_idx=0, type=LayerType.Linear, total=256, inp1_unstable=4, inp2_unstable=4, delta_unstable=252
- layer_idx=2, type=LayerType.Linear, total=256, inp1_unstable=4, inp2_unstable=4, delta_unstable=255
- layer_idx=4, type=LayerType.Linear, total=256, inp1_unstable=0, inp2_unstable=0, delta_unstable=256
- layer_idx=6, type=LayerType.Linear, total=256, inp1_unstable=0, inp2_unstable=0, delta_unstable=256
- layer_idx=8, type=LayerType.Linear, total=10, inp1_unstable=0, inp2_unstable=0, delta_unstable=10

Time for backsubstitution: 1.41 seconds

### RS candidates at layer 3
type: RSZ, layer: 3, pos: 240
type: RSZ, layer: 3, pos: 49
type: RSZ, layer: 3, pos: 34

Time for candidate selection: 0.00 seconds

### Candidate
type: RSZ, layer: 3, pos: 240

### Relational analysis RSZ of RS_RSZ1_RSZ2_RSZ1_RSZ1_RSZ1_RSZ1

#### Relational analysis RSZ result of RS_RSZ1_RSZ2_RSZ1_RSZ1_RSZ1_RSZ1
Status: Status.UNKNOWN
Output dim: 8, lower bound: -0.0001858, upper bound: 0.0002380
time: 0.50 seconds

### Relational analysis RSZ of RS_RSZ1_RSZ2_RSZ1_RSZ1_RSZ1_RSZ2

#### Relational analysis RSZ result of RS_RSZ1_RSZ2_RSZ1_RSZ1_RSZ1_RSZ2
Status: Status.UNKNOWN
Output dim: 8, lower bound: -0.0001829, upper bound: 0.0002375
time: 0.52 seconds

## BFS RS instance: RS_RSZ1_RSZ2_RSZ1_RSZ1_RSZ2

### Backsubstitution after applying RS history:
0: 0.0035972, 0.0041150, 0.0035972, 0.0041150, -0.0001273, 0.0001295
1: 0.0018420, 0.0019168, 0.0018420, 0.0019168, -0.0000184, 0.0000187
2: 0.0120848, 0.0123711, 0.0120848, 0.0123711, -0.0000716, 0.0000704
3: -0.0021818, -0.0018857, -0.0021818, -0.0018857, -0.0000741, 0.0000728
4: -0.0019956, -0.0016750, -0.0019956, -0.0016750, -0.0000788, 0.0000802
5: 0.0056985, 0.0060018, 0.0056985, 0.0060018, -0.0000759, 0.0000746
6: 0.0003095, 0.0015131, 0.0003095, 0.0015131, -0.0003011, 0.0002960
7: -0.0046174, -0.0029782, -0.0046174, -0.0029782, -0.0004031, 0.0004100
8: 0.9859612, 0.9871160, 0.9859612, 0.9871160, -0.0002839, 0.0002888
9: -0.0041920, -0.0031438, -0.0041920, -0.0031438, -0.0002622, 0.0002577

### Unstable ReLU Count (Linear/Conv2D Layers)
- layer_idx=0, type=LayerType.Linear, total=256, inp1_unstable=4, inp2_unstable=4, delta_unstable=252
- layer_idx=2, type=LayerType.Linear, total=256, inp1_unstable=4, inp2_unstable=4, delta_unstable=255
- layer_idx=4, type=LayerType.Linear, total=256, inp1_unstable=0, inp2_unstable=0, delta_unstable=256
- layer_idx=6, type=LayerType.Linear, total=256, inp1_unstable=0, inp2_unstable=0, delta_unstable=256
- layer_idx=8, type=LayerType.Linear, total=10, inp1_unstable=0, inp2_unstable=0, delta_unstable=10

Time for backsubstitution: 1.39 seconds

### RS candidates at layer 3
type: RSZ, layer: 3, pos: 34
type: RSZ, layer: 3, pos: 49
type: RSZ, layer: 3, pos: 240

Time for candidate selection: 0.00 seconds

### Candidate
type: RSZ, layer: 3, pos: 34

### Relational analysis RSZ of RS_RSZ1_RSZ2_RSZ1_RSZ1_RSZ2_RSZ1

#### Relational analysis RSZ result of RS_RSZ1_RSZ2_RSZ1_RSZ1_RSZ2_RSZ1
Status: Status.UNKNOWN
Output dim: 8, lower bound: -0.0002170, upper bound: 0.0002016
time: 0.52 seconds

### Relational analysis RSZ of RS_RSZ1_RSZ2_RSZ1_RSZ1_RSZ2_RSZ2

#### Relational analysis RSZ result of RS_RSZ1_RSZ2_RSZ1_RSZ1_RSZ2_RSZ2
Status: Status.VERIFIED
Output dim: 8, lower bound: -0.0002134, upper bound: 0.0002015
time: 0.52 seconds

## BFS RS instance: RS_RSZ1_RSZ2_RSZ1_RSZ2_RSZ1

### Backsubstitution after applying RS history:
0: 0.0035972, 0.0041150, 0.0035972, 0.0041150, -0.0001239, 0.0001310
1: 0.0018420, 0.0019168, 0.0018420, 0.0019168, -0.0000179, 0.0000189
2: 0.0120848, 0.0123711, 0.0120848, 0.0123711, -0.0000724, 0.0000685
3: -0.0021818, -0.0018857, -0.0021818, -0.0018857, -0.0000749, 0.0000709
4: -0.0019956, -0.0016750, -0.0019956, -0.0016750, -0.0000767, 0.0000811
5: 0.0056985, 0.0060018, 0.0056985, 0.0060018, -0.0000768, 0.0000726
6: 0.0003095, 0.0015131, 0.0003095, 0.0015131, -0.0003045, 0.0002880
7: -0.0046174, -0.0029782, -0.0046174, -0.0029782, -0.0003923, 0.0004148
8: 0.9859612, 0.9871160, 0.9859612, 0.9871160, -0.0002763, 0.0002922
9: -0.0041920, -0.0031438, -0.0041920, -0.0031438, -0.0002652, 0.0002508

### Unstable ReLU Count (Linear/Conv2D Layers)
- layer_idx=0, type=LayerType.Linear, total=256, inp1_unstable=4, inp2_unstable=4, delta_unstable=252
- layer_idx=2, type=LayerType.Linear, total=256, inp1_unstable=4, inp2_unstable=4, delta_unstable=255
- layer_idx=4, type=LayerType.Linear, total=256, inp1_unstable=0, inp2_unstable=0, delta_unstable=256
- layer_idx=6, type=LayerType.Linear, total=256, inp1_unstable=0, inp2_unstable=0, delta_unstable=256
- layer_idx=8, type=LayerType.Linear, total=10, inp1_unstable=0, inp2_unstable=0, delta_unstable=10

Time for backsubstitution: 1.41 seconds

### RS candidates at layer 3
type: RSZ, layer: 3, pos: 49
type: RSZ, layer: 3, pos: 240
type: RSZ, layer: 3, pos: 34

Time for candidate selection: 0.00 seconds

### Candidate
type: RSZ, layer: 3, pos: 49

### Relational analysis RSZ of RS_RSZ1_RSZ2_RSZ1_RSZ2_RSZ1_RSZ1

#### Relational analysis RSZ result of RS_RSZ1_RSZ2_RSZ1_RSZ2_RSZ1_RSZ1
Status: Status.VERIFIED
Output dim: 8, lower bound: -0.0001421, upper bound: 0.0002075
time: 0.53 seconds

### Relational analysis RSZ of RS_RSZ1_RSZ2_RSZ1_RSZ2_RSZ1_RSZ2

#### Relational analysis RSZ result of RS_RSZ1_RSZ2_RSZ1_RSZ2_RSZ1_RSZ2
Status: Status.VERIFIED
Output dim: 8, lower bound: -0.0001652, upper bound: 0.0001448
time: 0.52 seconds

## BFS RS instance: RS_RSZ1_RSZ2_RSZ1_RSZ2_RSZ2

### Backsubstitution after applying RS history:
0: 0.0035972, 0.0041150, 0.0035972, 0.0041150, -0.0001284, 0.0001272
1: 0.0018420, 0.0019168, 0.0018420, 0.0019168, -0.0000186, 0.0000184
2: 0.0120848, 0.0123711, 0.0120848, 0.0123711, -0.0000703, 0.0000710
3: -0.0021818, -0.0018857, -0.0021818, -0.0018857, -0.0000727, 0.0000734
4: -0.0019956, -0.0016750, -0.0019956, -0.0016750, -0.0000795, 0.0000787
5: 0.0056985, 0.0060018, 0.0056985, 0.0060018, -0.0000745, 0.0000752
6: 0.0003095, 0.0015131, 0.0003095, 0.0015131, -0.0002957, 0.0002986
7: -0.0046174, -0.0029782, -0.0046174, -0.0029782, -0.0004066, 0.0004027
8: 0.9859612, 0.9871160, 0.9859612, 0.9871160, -0.0002864, 0.0002836
9: -0.0041920, -0.0031438, -0.0041920, -0.0031438, -0.0002575, 0.0002600

### Unstable ReLU Count (Linear/Conv2D Layers)
- layer_idx=0, type=LayerType.Linear, total=256, inp1_unstable=4, inp2_unstable=4, delta_unstable=252
- layer_idx=2, type=LayerType.Linear, total=256, inp1_unstable=4, inp2_unstable=4, delta_unstable=255
- layer_idx=4, type=LayerType.Linear, total=256, inp1_unstable=0, inp2_unstable=0, delta_unstable=256
- layer_idx=6, type=LayerType.Linear, total=256, inp1_unstable=0, inp2_unstable=0, delta_unstable=256
- layer_idx=8, type=LayerType.Linear, total=10, inp1_unstable=0, inp2_unstable=0, delta_unstable=10

Time for backsubstitution: 1.39 seconds

### RS candidates at layer 3
type: RSZ, layer: 3, pos: 34
type: RSZ, layer: 3, pos: 49
type: RSZ, layer: 3, pos: 240

Time for candidate selection: 0.00 seconds

### Candidate
type: RSZ, layer: 3, pos: 34

### Relational analysis RSZ of RS_RSZ1_RSZ2_RSZ1_RSZ2_RSZ2_RSZ1

#### Relational analysis RSZ result of RS_RSZ1_RSZ2_RSZ1_RSZ2_RSZ2_RSZ1
Status: Status.UNKNOWN
Output dim: 8, lower bound: -0.0002212, upper bound: 0.0001937
time: 0.49 seconds

### Relational analysis RSZ of RS_RSZ1_RSZ2_RSZ1_RSZ2_RSZ2_RSZ2

#### Relational analysis RSZ result of RS_RSZ1_RSZ2_RSZ1_RSZ2_RSZ2_RSZ2
Status: Status.UNKNOWN
Output dim: 8, lower bound: -0.0002187, upper bound: 0.0001935
time: 0.52 seconds

## BFS RS instance: RS_RSZ1_RSZ2_RSZ2_RSZ1_RSZ1

### Backsubstitution after applying RS history:
0: 0.0035972, 0.0041150, 0.0035972, 0.0041150, -0.0001299, 0.0001301
1: 0.0018420, 0.0019168, 0.0018420, 0.0019168, -0.0000188, 0.0000188
2: 0.0120848, 0.0123711, 0.0120848, 0.0123711, -0.0000719, 0.0000718
3: -0.0021818, -0.0018857, -0.0021818, -0.0018857, -0.0000744, 0.0000743
4: -0.0019956, -0.0016750, -0.0019956, -0.0016750, -0.0000804, 0.0000805
5: 0.0056985, 0.0060018, 0.0056985, 0.0060018, -0.0000762, 0.0000761
6: 0.0003095, 0.0015131, 0.0003095, 0.0015131, -0.0003023, 0.0003019
7: -0.0046174, -0.0029782, -0.0046174, -0.0029782, -0.0004112, 0.0004117
8: 0.9859612, 0.9871160, 0.9859612, 0.9871160, -0.0002896, 0.0002900
9: -0.0041920, -0.0031438, -0.0041920, -0.0031438, -0.0002633, 0.0002629

### Unstable ReLU Count (Linear/Conv2D Layers)
- layer_idx=0, type=LayerType.Linear, total=256, inp1_unstable=4, inp2_unstable=4, delta_unstable=252
- layer_idx=2, type=LayerType.Linear, total=256, inp1_unstable=4, inp2_unstable=4, delta_unstable=255
- layer_idx=4, type=LayerType.Linear, total=256, inp1_unstable=0, inp2_unstable=0, delta_unstable=256
- layer_idx=6, type=LayerType.Linear, total=256, inp1_unstable=0, inp2_unstable=0, delta_unstable=256
- layer_idx=8, type=LayerType.Linear, total=10, inp1_unstable=0, inp2_unstable=0, delta_unstable=10

Time for backsubstitution: 1.42 seconds

### RS candidates at layer 3
type: RSZ, layer: 3, pos: 49
type: RSZ, layer: 3, pos: 83
type: RSZ, layer: 3, pos: 34

Time for candidate selection: 0.00 seconds

### Candidate
type: RSZ, layer: 3, pos: 49

### Relational analysis RSZ of RS_RSZ1_RSZ2_RSZ2_RSZ1_RSZ1_RSZ1

#### Relational analysis RSZ result of RS_RSZ1_RSZ2_RSZ2_RSZ1_RSZ1_RSZ1
Status: Status.UNKNOWN
Output dim: 8, lower bound: -0.0001795, upper bound: 0.0002276
time: 0.53 seconds

### Relational analysis RSZ of RS_RSZ1_RSZ2_RSZ2_RSZ1_RSZ1_RSZ2

#### Relational analysis RSZ result of RS_RSZ1_RSZ2_RSZ2_RSZ1_RSZ1_RSZ2
Status: Status.UNKNOWN
Output dim: 8, lower bound: -0.0002281, upper bound: 0.0001598
time: 0.51 seconds

## BFS RS instance: RS_RSZ1_RSZ2_RSZ2_RSZ1_RSZ2

### Backsubstitution after applying RS history:
0: 0.0035972, 0.0041150, 0.0035972, 0.0041150, -0.0001299, 0.0001297
1: 0.0018420, 0.0019168, 0.0018420, 0.0019168, -0.0000188, 0.0000187
2: 0.0120848, 0.0123711, 0.0120848, 0.0123711, -0.0000717, 0.0000718
3: -0.0021818, -0.0018857, -0.0021818, -0.0018857, -0.0000742, 0.0000743
4: -0.0019956, -0.0016750, -0.0019956, -0.0016750, -0.0000804, 0.0000803
5: 0.0056985, 0.0060018, 0.0056985, 0.0060018, -0.0000760, 0.0000761
6: 0.0003095, 0.0015131, 0.0003095, 0.0015131, -0.0003015, 0.0003019
7: -0.0046174, -0.0029782, -0.0046174, -0.0029782, -0.0004112, 0.0004106
8: 0.9859612, 0.9871160, 0.9859612, 0.9871160, -0.0002896, 0.0002893
9: -0.0041920, -0.0031438, -0.0041920, -0.0031438, -0.0002626, 0.0002629

### Unstable ReLU Count (Linear/Conv2D Layers)
- layer_idx=0, type=LayerType.Linear, total=256, inp1_unstable=4, inp2_unstable=4, delta_unstable=252
- layer_idx=2, type=LayerType.Linear, total=256, inp1_unstable=4, inp2_unstable=4, delta_unstable=255
- layer_idx=4, type=LayerType.Linear, total=256, inp1_unstable=0, inp2_unstable=0, delta_unstable=256
- layer_idx=6, type=LayerType.Linear, total=256, inp1_unstable=0, inp2_unstable=0, delta_unstable=256
- layer_idx=8, type=LayerType.Linear, total=10, inp1_unstable=0, inp2_unstable=0, delta_unstable=10

Time for backsubstitution: 1.42 seconds

### RS candidates at layer 3
type: RSZ, layer: 3, pos: 49
type: RSZ, layer: 3, pos: 83
type: RSZ, layer: 3, pos: 34

Time for candidate selection: 0.00 seconds

### Candidate
type: RSZ, layer: 3, pos: 49

### Relational analysis RSZ of RS_RSZ1_RSZ2_RSZ2_RSZ1_RSZ2_RSZ1

#### Relational analysis RSZ result of RS_RSZ1_RSZ2_RSZ2_RSZ1_RSZ2_RSZ1
Status: Status.UNKNOWN
Output dim: 8, lower bound: -0.0001757, upper bound: 0.0002245
time: 0.52 seconds

### Relational analysis RSZ of RS_RSZ1_RSZ2_RSZ2_RSZ1_RSZ2_RSZ2

#### Relational analysis RSZ result of RS_RSZ1_RSZ2_RSZ2_RSZ1_RSZ2_RSZ2
Status: Status.UNKNOWN
Output dim: 8, lower bound: -0.0002282, upper bound: 0.0001600
time: 0.52 seconds

## BFS RS instance: RS_RSZ1_RSZ2_RSZ2_RSZ2_RSZ1

### Backsubstitution after applying RS history:
0: 0.0035972, 0.0041150, 0.0035972, 0.0041150, -0.0001193, 0.0001282
1: 0.0018420, 0.0019168, 0.0018420, 0.0019168, -0.0000172, 0.0000185
2: 0.0120848, 0.0123711, 0.0120848, 0.0123711, -0.0000709, 0.0000659
3: -0.0021818, -0.0018857, -0.0021818, -0.0018857, -0.0000733, 0.0000682
4: -0.0019956, -0.0016750, -0.0019956, -0.0016750, -0.0000738, 0.0000794
5: 0.0056985, 0.0060018, 0.0056985, 0.0060018, -0.0000751, 0.0000699
6: 0.0003095, 0.0015131, 0.0003095, 0.0015131, -0.0002980, 0.0002772
7: -0.0046174, -0.0029782, -0.0046174, -0.0029782, -0.0003776, 0.0004059
8: 0.9859612, 0.9871160, 0.9859612, 0.9871160, -0.0002660, 0.0002859
9: -0.0041920, -0.0031438, -0.0041920, -0.0031438, -0.0002595, 0.0002414

### Unstable ReLU Count (Linear/Conv2D Layers)
- layer_idx=0, type=LayerType.Linear, total=256, inp1_unstable=4, inp2_unstable=4, delta_unstable=252
- layer_idx=2, type=LayerType.Linear, total=256, inp1_unstable=4, inp2_unstable=4, delta_unstable=255
- layer_idx=4, type=LayerType.Linear, total=256, inp1_unstable=0, inp2_unstable=0, delta_unstable=256
- layer_idx=6, type=LayerType.Linear, total=256, inp1_unstable=0, inp2_unstable=0, delta_unstable=256
- layer_idx=8, type=LayerType.Linear, total=10, inp1_unstable=0, inp2_unstable=0, delta_unstable=10

Time for backsubstitution: 1.44 seconds

### RS candidates at layer 3
type: RSZ, layer: 3, pos: 34
type: RSZ, layer: 3, pos: 240
type: RSZ, layer: 3, pos: 83

Time for candidate selection: 0.00 seconds

### Candidate
type: RSZ, layer: 3, pos: 34

### Relational analysis RSZ of RS_RSZ1_RSZ2_RSZ2_RSZ2_RSZ1_RSZ1

#### Relational analysis RSZ result of RS_RSZ1_RSZ2_RSZ2_RSZ2_RSZ1_RSZ1
Status: Status.VERIFIED
Output dim: 8, lower bound: -0.0001643, upper bound: 0.0002013
time: 0.57 seconds

### Relational analysis RSZ of RS_RSZ1_RSZ2_RSZ2_RSZ2_RSZ1_RSZ2

#### Relational analysis RSZ result of RS_RSZ1_RSZ2_RSZ2_RSZ2_RSZ1_RSZ2
Status: Status.VERIFIED
Output dim: 8, lower bound: -0.0001683, upper bound: 0.0002097
time: 0.53 seconds

## BFS RS instance: RS_RSZ1_RSZ2_RSZ2_RSZ2_RSZ2

### Backsubstitution after applying RS history:
0: 0.0035972, 0.0041150, 0.0035972, 0.0041150, -0.0001301, 0.0001184
1: 0.0018420, 0.0019168, 0.0018420, 0.0019168, -0.0000188, 0.0000171
2: 0.0120848, 0.0123711, 0.0120848, 0.0123711, -0.0000654, 0.0000719
3: -0.0021818, -0.0018857, -0.0021818, -0.0018857, -0.0000677, 0.0000744
4: -0.0019956, -0.0016750, -0.0019956, -0.0016750, -0.0000805, 0.0000733
5: 0.0056985, 0.0060018, 0.0056985, 0.0060018, -0.0000693, 0.0000762
6: 0.0003095, 0.0015131, 0.0003095, 0.0015131, -0.0002751, 0.0003024
7: -0.0046174, -0.0029782, -0.0046174, -0.0029782, -0.0004118, 0.0003746
8: 0.9859612, 0.9871160, 0.9859612, 0.9871160, -0.0002901, 0.0002639
9: -0.0041920, -0.0031438, -0.0041920, -0.0031438, -0.0002396, 0.0002633

### Unstable ReLU Count (Linear/Conv2D Layers)
- layer_idx=0, type=LayerType.Linear, total=256, inp1_unstable=4, inp2_unstable=4, delta_unstable=252
- layer_idx=2, type=LayerType.Linear, total=256, inp1_unstable=4, inp2_unstable=4, delta_unstable=255
- layer_idx=4, type=LayerType.Linear, total=256, inp1_unstable=0, inp2_unstable=0, delta_unstable=256
- layer_idx=6, type=LayerType.Linear, total=256, inp1_unstable=0, inp2_unstable=0, delta_unstable=256
- layer_idx=8, type=LayerType.Linear, total=10, inp1_unstable=0, inp2_unstable=0, delta_unstable=10

Time for backsubstitution: 1.60 seconds

### RS candidates at layer 3
type: RSZ, layer: 3, pos: 240
type: RSZ, layer: 3, pos: 83
type: RSZ, layer: 3, pos: 34

Time for candidate selection: 0.00 seconds

### Candidate
type: RSZ, layer: 3, pos: 240

### Relational analysis RSZ of RS_RSZ1_RSZ2_RSZ2_RSZ2_RSZ2_RSZ1

#### Relational analysis RSZ result of RS_RSZ1_RSZ2_RSZ2_RSZ2_RSZ2_RSZ1
Status: Status.UNKNOWN
Output dim: 8, lower bound: -0.0002339, upper bound: 0.0001600
time: 0.53 seconds

### Relational analysis RSZ of RS_RSZ1_RSZ2_RSZ2_RSZ2_RSZ2_RSZ2

#### Relational analysis RSZ result of RS_RSZ1_RSZ2_RSZ2_RSZ2_RSZ2_RSZ2
Status: Status.UNKNOWN
Output dim: 8, lower bound: -0.0002343, upper bound: 0.0001602
time: 0.55 seconds

## BFS RS instance: RS_RSZ2_RSZ1_RSZ1_RSZ1_RSZ1

### Backsubstitution after applying RS history:
0: 0.0035972, 0.0041150, 0.0035972, 0.0041150, -0.0001272, 0.0001315
1: 0.0018420, 0.0019168, 0.0018420, 0.0019168, -0.0000184, 0.0000190
2: 0.0120848, 0.0123711, 0.0120848, 0.0123711, -0.0000727, 0.0000703
3: -0.0021818, -0.0018857, -0.0021818, -0.0018857, -0.0000752, 0.0000727
4: -0.0019956, -0.0016750, -0.0019956, -0.0016750, -0.0000787, 0.0000814
5: 0.0056985, 0.0060018, 0.0056985, 0.0060018, -0.0000770, 0.0000745
6: 0.0003095, 0.0015131, 0.0003095, 0.0015131, -0.0003056, 0.0002956
7: -0.0046174, -0.0029782, -0.0046174, -0.0029782, -0.0004026, 0.0004162
8: 0.9859612, 0.9871160, 0.9859612, 0.9871160, -0.0002836, 0.0002932
9: -0.0041920, -0.0031438, -0.0041920, -0.0031438, -0.0002661, 0.0002574

### Unstable ReLU Count (Linear/Conv2D Layers)
- layer_idx=0, type=LayerType.Linear, total=256, inp1_unstable=4, inp2_unstable=4, delta_unstable=252
- layer_idx=2, type=LayerType.Linear, total=256, inp1_unstable=4, inp2_unstable=4, delta_unstable=255
- layer_idx=4, type=LayerType.Linear, total=256, inp1_unstable=0, inp2_unstable=0, delta_unstable=256
- layer_idx=6, type=LayerType.Linear, total=256, inp1_unstable=0, inp2_unstable=0, delta_unstable=256
- layer_idx=8, type=LayerType.Linear, total=10, inp1_unstable=0, inp2_unstable=0, delta_unstable=10

Time for backsubstitution: 1.45 seconds

### RS candidates at layer 3
type: RSZ, layer: 3, pos: 240
type: RSZ, layer: 3, pos: 49
type: RSZ, layer: 3, pos: 34

Time for candidate selection: 0.00 seconds

### Candidate
type: RSZ, layer: 3, pos: 240

### Relational analysis RSZ of RS_RSZ2_RSZ1_RSZ1_RSZ1_RSZ1_RSZ1

#### Relational analysis RSZ result of RS_RSZ2_RSZ1_RSZ1_RSZ1_RSZ1_RSZ1
Status: Status.UNKNOWN
Output dim: 8, lower bound: -0.0001871, upper bound: 0.0002379
time: 0.50 seconds

### Relational analysis RSZ of RS_RSZ2_RSZ1_RSZ1_RSZ1_RSZ1_RSZ2

#### Relational analysis RSZ result of RS_RSZ2_RSZ1_RSZ1_RSZ1_RSZ1_RSZ2
Status: Status.UNKNOWN
Output dim: 8, lower bound: -0.0001869, upper bound: 0.0002376
time: 0.50 seconds

## BFS RS instance: RS_RSZ2_RSZ1_RSZ1_RSZ1_RSZ2

### Backsubstitution after applying RS history:
0: 0.0035972, 0.0041150, 0.0035972, 0.0041150, -0.0001311, 0.0001281
1: 0.0018420, 0.0019168, 0.0018420, 0.0019168, -0.0000189, 0.0000185
2: 0.0120848, 0.0123711, 0.0120848, 0.0123711, -0.0000708, 0.0000725
3: -0.0021818, -0.0018857, -0.0021818, -0.0018857, -0.0000733, 0.0000750
4: -0.0019956, -0.0016750, -0.0019956, -0.0016750, -0.0000812, 0.0000793
5: 0.0056985, 0.0060018, 0.0056985, 0.0060018, -0.0000751, 0.0000768
6: 0.0003095, 0.0015131, 0.0003095, 0.0015131, -0.0002978, 0.0003048
7: -0.0046174, -0.0029782, -0.0046174, -0.0029782, -0.0004151, 0.0004056
8: 0.9859612, 0.9871160, 0.9859612, 0.9871160, -0.0002924, 0.0002857
9: -0.0041920, -0.0031438, -0.0041920, -0.0031438, -0.0002593, 0.0002654

### Unstable ReLU Count (Linear/Conv2D Layers)
- layer_idx=0, type=LayerType.Linear, total=256, inp1_unstable=4, inp2_unstable=4, delta_unstable=252
- layer_idx=2, type=LayerType.Linear, total=256, inp1_unstable=4, inp2_unstable=4, delta_unstable=255
- layer_idx=4, type=LayerType.Linear, total=256, inp1_unstable=0, inp2_unstable=0, delta_unstable=256
- layer_idx=6, type=LayerType.Linear, total=256, inp1_unstable=0, inp2_unstable=0, delta_unstable=256
- layer_idx=8, type=LayerType.Linear, total=10, inp1_unstable=0, inp2_unstable=0, delta_unstable=10

Time for backsubstitution: 1.43 seconds

### RS candidates at layer 3
type: RSZ, layer: 3, pos: 49
type: RSZ, layer: 3, pos: 34
type: RSZ, layer: 3, pos: 240

Time for candidate selection: 0.00 seconds

### Candidate
type: RSZ, layer: 3, pos: 49

### Relational analysis RSZ of RS_RSZ2_RSZ1_RSZ1_RSZ1_RSZ2_RSZ1

#### Relational analysis RSZ result of RS_RSZ2_RSZ1_RSZ1_RSZ1_RSZ2_RSZ1
Status: Status.VERIFIED
Output dim: 8, lower bound: -0.0001424, upper bound: 0.0001859
time: 0.53 seconds

### Relational analysis RSZ of RS_RSZ2_RSZ1_RSZ1_RSZ1_RSZ2_RSZ2

#### Relational analysis RSZ result of RS_RSZ2_RSZ1_RSZ1_RSZ1_RSZ2_RSZ2
Status: Status.VERIFIED
Output dim: 8, lower bound: -0.0002056, upper bound: 0.0001545
time: 0.53 seconds

## BFS RS instance: RS_RSZ2_RSZ1_RSZ1_RSZ2_RSZ1

### Backsubstitution after applying RS history:
0: 0.0035972, 0.0041150, 0.0035972, 0.0041150, -0.0001267, 0.0001299
1: 0.0018420, 0.0019168, 0.0018420, 0.0019168, -0.0000183, 0.0000188
2: 0.0120848, 0.0123711, 0.0120848, 0.0123711, -0.0000718, 0.0000700
3: -0.0021818, -0.0018857, -0.0021818, -0.0018857, -0.0000743, 0.0000724
4: -0.0019956, -0.0016750, -0.0019956, -0.0016750, -0.0000784, 0.0000804
5: 0.0056985, 0.0060018, 0.0056985, 0.0060018, -0.0000761, 0.0000742
6: 0.0003095, 0.0015131, 0.0003095, 0.0015131, -0.0003019, 0.0002944
7: -0.0046174, -0.0029782, -0.0046174, -0.0029782, -0.0004010, 0.0004112
8: 0.9859612, 0.9871160, 0.9859612, 0.9871160, -0.0002825, 0.0002897
9: -0.0041920, -0.0031438, -0.0041920, -0.0031438, -0.0002629, 0.0002564

### Unstable ReLU Count (Linear/Conv2D Layers)
- layer_idx=0, type=LayerType.Linear, total=256, inp1_unstable=4, inp2_unstable=4, delta_unstable=252
- layer_idx=2, type=LayerType.Linear, total=256, inp1_unstable=4, inp2_unstable=4, delta_unstable=255
- layer_idx=4, type=LayerType.Linear, total=256, inp1_unstable=0, inp2_unstable=0, delta_unstable=256
- layer_idx=6, type=LayerType.Linear, total=256, inp1_unstable=0, inp2_unstable=0, delta_unstable=256
- layer_idx=8, type=LayerType.Linear, total=10, inp1_unstable=0, inp2_unstable=0, delta_unstable=10

Time for backsubstitution: 1.43 seconds

### RS candidates at layer 3
type: RSZ, layer: 3, pos: 34
type: RSZ, layer: 3, pos: 49
type: RSZ, layer: 3, pos: 240

Time for candidate selection: 0.00 seconds

### Candidate
type: RSZ, layer: 3, pos: 34

### Relational analysis RSZ of RS_RSZ2_RSZ1_RSZ1_RSZ2_RSZ1_RSZ1

#### Relational analysis RSZ result of RS_RSZ2_RSZ1_RSZ1_RSZ2_RSZ1_RSZ1
Status: Status.UNKNOWN
Output dim: 8, lower bound: -0.0001848, upper bound: 0.0002183
time: 0.48 seconds

### Relational analysis RSZ of RS_RSZ2_RSZ1_RSZ1_RSZ2_RSZ1_RSZ2

#### Relational analysis RSZ result of RS_RSZ2_RSZ1_RSZ1_RSZ2_RSZ1_RSZ2
Status: Status.UNKNOWN
Output dim: 8, lower bound: -0.0001846, upper bound: 0.0002204
time: 0.50 seconds

## BFS RS instance: RS_RSZ2_RSZ1_RSZ1_RSZ2_RSZ2

### Backsubstitution after applying RS history:
0: 0.0035972, 0.0041150, 0.0035972, 0.0041150, -0.0001307, 0.0001265
1: 0.0018420, 0.0019168, 0.0018420, 0.0019168, -0.0000189, 0.0000183
2: 0.0120848, 0.0123711, 0.0120848, 0.0123711, -0.0000699, 0.0000723
3: -0.0021818, -0.0018857, -0.0021818, -0.0018857, -0.0000723, 0.0000748
4: -0.0019956, -0.0016750, -0.0019956, -0.0016750, -0.0000809, 0.0000783
5: 0.0056985, 0.0060018, 0.0056985, 0.0060018, -0.0000741, 0.0000766
6: 0.0003095, 0.0015131, 0.0003095, 0.0015131, -0.0002940, 0.0003039
7: -0.0046174, -0.0029782, -0.0046174, -0.0029782, -0.0004138, 0.0004004
8: 0.9859612, 0.9871160, 0.9859612, 0.9871160, -0.0002915, 0.0002821
9: -0.0041920, -0.0031438, -0.0041920, -0.0031438, -0.0002560, 0.0002646

### Unstable ReLU Count (Linear/Conv2D Layers)
- layer_idx=0, type=LayerType.Linear, total=256, inp1_unstable=4, inp2_unstable=4, delta_unstable=252
- layer_idx=2, type=LayerType.Linear, total=256, inp1_unstable=4, inp2_unstable=4, delta_unstable=255
- layer_idx=4, type=LayerType.Linear, total=256, inp1_unstable=0, inp2_unstable=0, delta_unstable=256
- layer_idx=6, type=LayerType.Linear, total=256, inp1_unstable=0, inp2_unstable=0, delta_unstable=256
- layer_idx=8, type=LayerType.Linear, total=10, inp1_unstable=0, inp2_unstable=0, delta_unstable=10

Time for backsubstitution: 1.47 seconds

### RS candidates at layer 3
type: RSZ, layer: 3, pos: 240
type: RSZ, layer: 3, pos: 34
type: RSZ, layer: 3, pos: 49

Time for candidate selection: 0.00 seconds

### Candidate
type: RSZ, layer: 3, pos: 240

### Relational analysis RSZ of RS_RSZ2_RSZ1_RSZ1_RSZ2_RSZ2_RSZ1

#### Relational analysis RSZ result of RS_RSZ2_RSZ1_RSZ1_RSZ2_RSZ2_RSZ1
Status: Status.UNKNOWN
Output dim: 8, lower bound: -0.0002318, upper bound: 0.0001955
time: 0.51 seconds

### Relational analysis RSZ of RS_RSZ2_RSZ1_RSZ1_RSZ2_RSZ2_RSZ2

#### Relational analysis RSZ result of RS_RSZ2_RSZ1_RSZ1_RSZ2_RSZ2_RSZ2
Status: Status.UNKNOWN
Output dim: 8, lower bound: -0.0002334, upper bound: 0.0002055
time: 0.50 seconds

## BFS RS instance: RS_RSZ2_RSZ1_RSZ2_RSZ1_RSZ1

### Backsubstitution after applying RS history:
0: 0.0035972, 0.0041150, 0.0035972, 0.0041150, -0.0001272, 0.0001284
1: 0.0018420, 0.0019168, 0.0018420, 0.0019168, -0.0000184, 0.0000186
2: 0.0120848, 0.0123711, 0.0120848, 0.0123711, -0.0000710, 0.0000703
3: -0.0021818, -0.0018857, -0.0021818, -0.0018857, -0.0000734, 0.0000727
4: -0.0019956, -0.0016750, -0.0019956, -0.0016750, -0.0000787, 0.0000795
5: 0.0056985, 0.0060018, 0.0056985, 0.0060018, -0.0000752, 0.0000745
6: 0.0003095, 0.0015131, 0.0003095, 0.0015131, -0.0002986, 0.0002957
7: -0.0046174, -0.0029782, -0.0046174, -0.0029782, -0.0004027, 0.0004066
8: 0.9859612, 0.9871160, 0.9859612, 0.9871160, -0.0002836, 0.0002864
9: -0.0041920, -0.0031438, -0.0041920, -0.0031438, -0.0002600, 0.0002575

### Unstable ReLU Count (Linear/Conv2D Layers)
- layer_idx=0, type=LayerType.Linear, total=256, inp1_unstable=4, inp2_unstable=4, delta_unstable=252
- layer_idx=2, type=LayerType.Linear, total=256, inp1_unstable=4, inp2_unstable=4, delta_unstable=255
- layer_idx=4, type=LayerType.Linear, total=256, inp1_unstable=0, inp2_unstable=0, delta_unstable=256
- layer_idx=6, type=LayerType.Linear, total=256, inp1_unstable=0, inp2_unstable=0, delta_unstable=256
- layer_idx=8, type=LayerType.Linear, total=10, inp1_unstable=0, inp2_unstable=0, delta_unstable=10

Time for backsubstitution: 1.41 seconds

### RS candidates at layer 3
type: RSZ, layer: 3, pos: 34
type: RSZ, layer: 3, pos: 49
type: RSZ, layer: 3, pos: 240

Time for candidate selection: 0.00 seconds

### Candidate
type: RSZ, layer: 3, pos: 34

### Relational analysis RSZ of RS_RSZ2_RSZ1_RSZ2_RSZ1_RSZ1_RSZ1

#### Relational analysis RSZ result of RS_RSZ2_RSZ1_RSZ2_RSZ1_RSZ1_RSZ1
Status: Status.UNKNOWN
Output dim: 8, lower bound: -0.0001935, upper bound: 0.0002187
time: 0.52 seconds

### Relational analysis RSZ of RS_RSZ2_RSZ1_RSZ2_RSZ1_RSZ1_RSZ2

#### Relational analysis RSZ result of RS_RSZ2_RSZ1_RSZ2_RSZ1_RSZ1_RSZ2
Status: Status.UNKNOWN
Output dim: 8, lower bound: -0.0001937, upper bound: 0.0002212
time: 0.49 seconds

## BFS RS instance: RS_RSZ2_RSZ1_RSZ2_RSZ1_RSZ2

### Backsubstitution after applying RS history:
0: 0.0035972, 0.0041150, 0.0035972, 0.0041150, -0.0001310, 0.0001239
1: 0.0018420, 0.0019168, 0.0018420, 0.0019168, -0.0000189, 0.0000179
2: 0.0120848, 0.0123711, 0.0120848, 0.0123711, -0.0000685, 0.0000724
3: -0.0021818, -0.0018857, -0.0021818, -0.0018857, -0.0000709, 0.0000749
4: -0.0019956, -0.0016750, -0.0019956, -0.0016750, -0.0000811, 0.0000767
5: 0.0056985, 0.0060018, 0.0056985, 0.0060018, -0.0000726, 0.0000768
6: 0.0003095, 0.0015131, 0.0003095, 0.0015131, -0.0002880, 0.0003045
7: -0.0046174, -0.0029782, -0.0046174, -0.0029782, -0.0004148, 0.0003923
8: 0.9859612, 0.9871160, 0.9859612, 0.9871160, -0.0002922, 0.0002763
9: -0.0041920, -0.0031438, -0.0041920, -0.0031438, -0.0002508, 0.0002652

### Unstable ReLU Count (Linear/Conv2D Layers)
- layer_idx=0, type=LayerType.Linear, total=256, inp1_unstable=4, inp2_unstable=4, delta_unstable=252
- layer_idx=2, type=LayerType.Linear, total=256, inp1_unstable=4, inp2_unstable=4, delta_unstable=255
- layer_idx=4, type=LayerType.Linear, total=256, inp1_unstable=0, inp2_unstable=0, delta_unstable=256
- layer_idx=6, type=LayerType.Linear, total=256, inp1_unstable=0, inp2_unstable=0, delta_unstable=256
- layer_idx=8, type=LayerType.Linear, total=10, inp1_unstable=0, inp2_unstable=0, delta_unstable=10

Time for backsubstitution: 1.45 seconds

### RS candidates at layer 3
type: RSZ, layer: 3, pos: 34
type: RSZ, layer: 3, pos: 240
type: RSZ, layer: 3, pos: 49

Time for candidate selection: 0.00 seconds

### Candidate
type: RSZ, layer: 3, pos: 34

### Relational analysis RSZ of RS_RSZ2_RSZ1_RSZ2_RSZ1_RSZ2_RSZ1

#### Relational analysis RSZ result of RS_RSZ2_RSZ1_RSZ2_RSZ1_RSZ2_RSZ1
Status: Status.UNKNOWN
Output dim: 8, lower bound: -0.0002204, upper bound: 0.0001828
time: 0.51 seconds

### Relational analysis RSZ of RS_RSZ2_RSZ1_RSZ2_RSZ1_RSZ2_RSZ2

#### Relational analysis RSZ result of RS_RSZ2_RSZ1_RSZ2_RSZ1_RSZ2_RSZ2
Status: Status.UNKNOWN
Output dim: 8, lower bound: -0.0002177, upper bound: 0.0001822
time: 0.48 seconds

## BFS RS instance: RS_RSZ2_RSZ1_RSZ2_RSZ2_RSZ1

### Backsubstitution after applying RS history:
0: 0.0035972, 0.0041150, 0.0035972, 0.0041150, -0.0001215, 0.0001139
1: 0.0018420, 0.0019168, 0.0018420, 0.0019168, -0.0000176, 0.0000165
2: 0.0120848, 0.0123711, 0.0120848, 0.0123711, -0.0000630, 0.0000672
3: -0.0021818, -0.0018857, -0.0021818, -0.0018857, -0.0000651, 0.0000695
4: -0.0019956, -0.0016750, -0.0019956, -0.0016750, -0.0000752, 0.0000705
5: 0.0056985, 0.0060018, 0.0056985, 0.0060018, -0.0000667, 0.0000712
6: 0.0003095, 0.0015131, 0.0003095, 0.0015131, -0.0002647, 0.0002824
7: -0.0046174, -0.0029782, -0.0046174, -0.0029782, -0.0003846, 0.0003604
8: 0.9859612, 0.9871160, 0.9859612, 0.9871160, -0.0002709, 0.0002539
9: -0.0041920, -0.0031438, -0.0041920, -0.0031438, -0.0002305, 0.0002459

### Unstable ReLU Count (Linear/Conv2D Layers)
- layer_idx=0, type=LayerType.Linear, total=256, inp1_unstable=4, inp2_unstable=4, delta_unstable=252
- layer_idx=2, type=LayerType.Linear, total=256, inp1_unstable=4, inp2_unstable=4, delta_unstable=255
- layer_idx=4, type=LayerType.Linear, total=256, inp1_unstable=0, inp2_unstable=0, delta_unstable=256
- layer_idx=6, type=LayerType.Linear, total=256, inp1_unstable=0, inp2_unstable=0, delta_unstable=256
- layer_idx=8, type=LayerType.Linear, total=10, inp1_unstable=0, inp2_unstable=0, delta_unstable=10

Time for backsubstitution: 1.45 seconds

### RS candidates at layer 3
type: RSZ, layer: 3, pos: 83
type: RSZ, layer: 3, pos: 240
type: RSZ, layer: 3, pos: 49

Time for candidate selection: 0.00 seconds

### Candidate
type: RSZ, layer: 3, pos: 83

### Relational analysis RSZ of RS_RSZ2_RSZ1_RSZ2_RSZ2_RSZ1_RSZ1

#### Relational analysis RSZ result of RS_RSZ2_RSZ1_RSZ2_RSZ2_RSZ1_RSZ1
Status: Status.VERIFIED
Output dim: 8, lower bound: -0.0002015, upper bound: 0.0002134
time: 0.49 seconds

### Relational analysis RSZ of RS_RSZ2_RSZ1_RSZ2_RSZ2_RSZ1_RSZ2

#### Relational analysis RSZ result of RS_RSZ2_RSZ1_RSZ2_RSZ2_RSZ1_RSZ2
Status: Status.UNKNOWN
Output dim: 8, lower bound: -0.0002257, upper bound: 0.0001739
time: 0.52 seconds

## BFS RS instance: RS_RSZ2_RSZ1_RSZ2_RSZ2_RSZ2

### Backsubstitution after applying RS history:
0: 0.0035972, 0.0041150, 0.0035972, 0.0041150, -0.0001203, 0.0001151
1: 0.0018420, 0.0019168, 0.0018420, 0.0019168, -0.0000174, 0.0000166
2: 0.0120848, 0.0123711, 0.0120848, 0.0123711, -0.0000637, 0.0000665
3: -0.0021818, -0.0018857, -0.0021818, -0.0018857, -0.0000658, 0.0000688
4: -0.0019956, -0.0016750, -0.0019956, -0.0016750, -0.0000745, 0.0000713
5: 0.0056985, 0.0060018, 0.0056985, 0.0060018, -0.0000675, 0.0000705
6: 0.0003095, 0.0015131, 0.0003095, 0.0015131, -0.0002676, 0.0002797
7: -0.0046174, -0.0029782, -0.0046174, -0.0029782, -0.0003810, 0.0003645
8: 0.9859612, 0.9871160, 0.9859612, 0.9871160, -0.0002684, 0.0002568
9: -0.0041920, -0.0031438, -0.0041920, -0.0031438, -0.0002331, 0.0002436

### Unstable ReLU Count (Linear/Conv2D Layers)
- layer_idx=0, type=LayerType.Linear, total=256, inp1_unstable=4, inp2_unstable=4, delta_unstable=252
- layer_idx=2, type=LayerType.Linear, total=256, inp1_unstable=4, inp2_unstable=4, delta_unstable=255
- layer_idx=4, type=LayerType.Linear, total=256, inp1_unstable=0, inp2_unstable=0, delta_unstable=256
- layer_idx=6, type=LayerType.Linear, total=256, inp1_unstable=0, inp2_unstable=0, delta_unstable=256
- layer_idx=8, type=LayerType.Linear, total=10, inp1_unstable=0, inp2_unstable=0, delta_unstable=10

Time for backsubstitution: 1.44 seconds

### RS candidates at layer 3
type: RSZ, layer: 3, pos: 240
type: RSZ, layer: 3, pos: 83
type: RSZ, layer: 3, pos: 49

Time for candidate selection: 0.00 seconds

### Candidate
type: RSZ, layer: 3, pos: 240

### Relational analysis RSZ of RS_RSZ2_RSZ1_RSZ2_RSZ2_RSZ2_RSZ1

#### Relational analysis RSZ result of RS_RSZ2_RSZ1_RSZ2_RSZ2_RSZ2_RSZ1
Status: Status.UNKNOWN
Output dim: 8, lower bound: -0.0002415, upper bound: 0.0002323
time: 0.50 seconds

### Relational analysis RSZ of RS_RSZ2_RSZ1_RSZ2_RSZ2_RSZ2_RSZ2

#### Relational analysis RSZ result of RS_RSZ2_RSZ1_RSZ2_RSZ2_RSZ2_RSZ2
Status: Status.UNKNOWN
Output dim: 8, lower bound: -0.0002409, upper bound: 0.0002281
time: 0.56 seconds

## BFS RS instance: RS_RSZ2_RSZ2_RSZ1_RSZ1_RSZ1

### Backsubstitution after applying RS history:
0: 0.0035972, 0.0041150, 0.0035972, 0.0041150, -0.0001255, 0.0001312
1: 0.0018420, 0.0019168, 0.0018420, 0.0019168, -0.0000181, 0.0000189
2: 0.0120848, 0.0123711, 0.0120848, 0.0123711, -0.0000725, 0.0000694
3: -0.0021818, -0.0018857, -0.0021818, -0.0018857, -0.0000750, 0.0000718
4: -0.0019956, -0.0016750, -0.0019956, -0.0016750, -0.0000777, 0.0000812
5: 0.0056985, 0.0060018, 0.0056985, 0.0060018, -0.0000768, 0.0000735
6: 0.0003095, 0.0015131, 0.0003095, 0.0015131, -0.0003049, 0.0002918
7: -0.0046174, -0.0029782, -0.0046174, -0.0029782, -0.0003974, 0.0004152
8: 0.9859612, 0.9871160, 0.9859612, 0.9871160, -0.0002799, 0.0002925
9: -0.0041920, -0.0031438, -0.0041920, -0.0031438, -0.0002655, 0.0002541

### Unstable ReLU Count (Linear/Conv2D Layers)
- layer_idx=0, type=LayerType.Linear, total=256, inp1_unstable=4, inp2_unstable=4, delta_unstable=252
- layer_idx=2, type=LayerType.Linear, total=256, inp1_unstable=4, inp2_unstable=4, delta_unstable=255
- layer_idx=4, type=LayerType.Linear, total=256, inp1_unstable=0, inp2_unstable=0, delta_unstable=256
- layer_idx=6, type=LayerType.Linear, total=256, inp1_unstable=0, inp2_unstable=0, delta_unstable=256
- layer_idx=8, type=LayerType.Linear, total=10, inp1_unstable=0, inp2_unstable=0, delta_unstable=10

Time for backsubstitution: 1.45 seconds

### RS candidates at layer 3
type: RSZ, layer: 3, pos: 34
type: RSZ, layer: 3, pos: 240
type: RSZ, layer: 3, pos: 49

Time for candidate selection: 0.00 seconds

### Candidate
type: RSZ, layer: 3, pos: 34

### Relational analysis RSZ of RS_RSZ2_RSZ2_RSZ1_RSZ1_RSZ1_RSZ1

#### Relational analysis RSZ result of RS_RSZ2_RSZ2_RSZ1_RSZ1_RSZ1_RSZ1
Status: Status.UNKNOWN
Output dim: 8, lower bound: -0.0001742, upper bound: 0.0002230
time: 0.53 seconds

### Relational analysis RSZ of RS_RSZ2_RSZ2_RSZ1_RSZ1_RSZ1_RSZ2

#### Relational analysis RSZ result of RS_RSZ2_RSZ2_RSZ1_RSZ1_RSZ1_RSZ2
Status: Status.UNKNOWN
Output dim: 8, lower bound: -0.0001741, upper bound: 0.0002257
time: 0.54 seconds

## BFS RS instance: RS_RSZ2_RSZ2_RSZ1_RSZ1_RSZ2

### Backsubstitution after applying RS history:
0: 0.0035972, 0.0041150, 0.0035972, 0.0041150, -0.0001298, 0.0001277
1: 0.0018420, 0.0019168, 0.0018420, 0.0019168, -0.0000187, 0.0000184
2: 0.0120848, 0.0123711, 0.0120848, 0.0123711, -0.0000706, 0.0000717
3: -0.0021818, -0.0018857, -0.0021818, -0.0018857, -0.0000730, 0.0000742
4: -0.0019956, -0.0016750, -0.0019956, -0.0016750, -0.0000803, 0.0000790
5: 0.0056985, 0.0060018, 0.0056985, 0.0060018, -0.0000748, 0.0000760
6: 0.0003095, 0.0015131, 0.0003095, 0.0015131, -0.0002968, 0.0003016
7: -0.0046174, -0.0029782, -0.0046174, -0.0029782, -0.0004107, 0.0004042
8: 0.9859612, 0.9871160, 0.9859612, 0.9871160, -0.0002893, 0.0002847
9: -0.0041920, -0.0031438, -0.0041920, -0.0031438, -0.0002585, 0.0002626

### Unstable ReLU Count (Linear/Conv2D Layers)
- layer_idx=0, type=LayerType.Linear, total=256, inp1_unstable=4, inp2_unstable=4, delta_unstable=252
- layer_idx=2, type=LayerType.Linear, total=256, inp1_unstable=4, inp2_unstable=4, delta_unstable=255
- layer_idx=4, type=LayerType.Linear, total=256, inp1_unstable=0, inp2_unstable=0, delta_unstable=256
- layer_idx=6, type=LayerType.Linear, total=256, inp1_unstable=0, inp2_unstable=0, delta_unstable=256
- layer_idx=8, type=LayerType.Linear, total=10, inp1_unstable=0, inp2_unstable=0, delta_unstable=10

Time for backsubstitution: 1.50 seconds

### RS candidates at layer 3
type: RSZ, layer: 3, pos: 240
type: RSZ, layer: 3, pos: 49
type: RSZ, layer: 3, pos: 34

Time for candidate selection: 0.00 seconds

### Candidate
type: RSZ, layer: 3, pos: 240

### Relational analysis RSZ of RS_RSZ2_RSZ2_RSZ1_RSZ1_RSZ2_RSZ1

#### Relational analysis RSZ result of RS_RSZ2_RSZ2_RSZ1_RSZ1_RSZ2_RSZ1
Status: Status.UNKNOWN
Output dim: 8, lower bound: -0.0002258, upper bound: 0.0002054
time: 0.48 seconds

### Relational analysis RSZ of RS_RSZ2_RSZ2_RSZ1_RSZ1_RSZ2_RSZ2

#### Relational analysis RSZ result of RS_RSZ2_RSZ2_RSZ1_RSZ1_RSZ2_RSZ2
Status: Status.UNKNOWN
Output dim: 8, lower bound: -0.0002294, upper bound: 0.0002135
time: 0.50 seconds

## BFS RS instance: RS_RSZ2_RSZ2_RSZ1_RSZ2_RSZ1

### Backsubstitution after applying RS history:
0: 0.0035972, 0.0041150, 0.0035972, 0.0041150, -0.0001306, 0.0001275
1: 0.0018420, 0.0019168, 0.0018420, 0.0019168, -0.0000189, 0.0000184
2: 0.0120848, 0.0123711, 0.0120848, 0.0123711, -0.0000705, 0.0000722
3: -0.0021818, -0.0018857, -0.0021818, -0.0018857, -0.0000729, 0.0000747
4: -0.0019956, -0.0016750, -0.0019956, -0.0016750, -0.0000809, 0.0000789
5: 0.0056985, 0.0060018, 0.0056985, 0.0060018, -0.0000747, 0.0000765
6: 0.0003095, 0.0015131, 0.0003095, 0.0015131, -0.0002964, 0.0003036
7: -0.0046174, -0.0029782, -0.0046174, -0.0029782, -0.0004135, 0.0004037
8: 0.9859612, 0.9871160, 0.9859612, 0.9871160, -0.0002912, 0.0002843
9: -0.0041920, -0.0031438, -0.0041920, -0.0031438, -0.0002581, 0.0002644

### Unstable ReLU Count (Linear/Conv2D Layers)
- layer_idx=0, type=LayerType.Linear, total=256, inp1_unstable=4, inp2_unstable=4, delta_unstable=252
- layer_idx=2, type=LayerType.Linear, total=256, inp1_unstable=4, inp2_unstable=4, delta_unstable=255
- layer_idx=4, type=LayerType.Linear, total=256, inp1_unstable=0, inp2_unstable=0, delta_unstable=256
- layer_idx=6, type=LayerType.Linear, total=256, inp1_unstable=0, inp2_unstable=0, delta_unstable=256
- layer_idx=8, type=LayerType.Linear, total=10, inp1_unstable=0, inp2_unstable=0, delta_unstable=10

Time for backsubstitution: 1.43 seconds

### RS candidates at layer 3
type: RSZ, layer: 3, pos: 83
type: RSZ, layer: 3, pos: 34
type: RSZ, layer: 3, pos: 49

Time for candidate selection: 0.00 seconds

### Candidate
type: RSZ, layer: 3, pos: 83

### Relational analysis RSZ of RS_RSZ2_RSZ2_RSZ1_RSZ2_RSZ1_RSZ1

#### Relational analysis RSZ result of RS_RSZ2_RSZ2_RSZ1_RSZ2_RSZ1_RSZ1
Status: Status.UNKNOWN
Output dim: 8, lower bound: -0.0002055, upper bound: 0.0002334
time: 0.49 seconds

### Relational analysis RSZ of RS_RSZ2_RSZ2_RSZ1_RSZ2_RSZ1_RSZ2

#### Relational analysis RSZ result of RS_RSZ2_RSZ2_RSZ1_RSZ2_RSZ1_RSZ2
Status: Status.UNKNOWN
Output dim: 8, lower bound: -0.0002323, upper bound: 0.0001946
time: 0.48 seconds

## BFS RS instance: RS_RSZ2_RSZ2_RSZ1_RSZ2_RSZ2

### Backsubstitution after applying RS history:
0: 0.0035972, 0.0041150, 0.0035972, 0.0041150, -0.0001305, 0.0001271
1: 0.0018420, 0.0019168, 0.0018420, 0.0019168, -0.0000189, 0.0000184
2: 0.0120848, 0.0123711, 0.0120848, 0.0123711, -0.0000703, 0.0000722
3: -0.0021818, -0.0018857, -0.0021818, -0.0018857, -0.0000727, 0.0000746
4: -0.0019956, -0.0016750, -0.0019956, -0.0016750, -0.0000808, 0.0000787
5: 0.0056985, 0.0060018, 0.0056985, 0.0060018, -0.0000744, 0.0000764
6: 0.0003095, 0.0015131, 0.0003095, 0.0015131, -0.0002953, 0.0003033
7: -0.0046174, -0.0029782, -0.0046174, -0.0029782, -0.0004131, 0.0004022
8: 0.9859612, 0.9871160, 0.9859612, 0.9871160, -0.0002910, 0.0002833
9: -0.0041920, -0.0031438, -0.0041920, -0.0031438, -0.0002572, 0.0002641

### Unstable ReLU Count (Linear/Conv2D Layers)
- layer_idx=0, type=LayerType.Linear, total=256, inp1_unstable=4, inp2_unstable=4, delta_unstable=252
- layer_idx=2, type=LayerType.Linear, total=256, inp1_unstable=4, inp2_unstable=4, delta_unstable=255
- layer_idx=4, type=LayerType.Linear, total=256, inp1_unstable=0, inp2_unstable=0, delta_unstable=256
- layer_idx=6, type=LayerType.Linear, total=256, inp1_unstable=0, inp2_unstable=0, delta_unstable=256
- layer_idx=8, type=LayerType.Linear, total=10, inp1_unstable=0, inp2_unstable=0, delta_unstable=10

Time for backsubstitution: 1.44 seconds

### RS candidates at layer 3
type: RSZ, layer: 3, pos: 83
type: RSZ, layer: 3, pos: 34
type: RSZ, layer: 3, pos: 49

Time for candidate selection: 0.00 seconds

### Candidate
type: RSZ, layer: 3, pos: 83

### Relational analysis RSZ of RS_RSZ2_RSZ2_RSZ1_RSZ2_RSZ2_RSZ1

#### Relational analysis RSZ result of RS_RSZ2_RSZ2_RSZ1_RSZ2_RSZ2_RSZ1
Status: Status.UNKNOWN
Output dim: 8, lower bound: -0.0001955, upper bound: 0.0002318
time: 0.51 seconds

### Relational analysis RSZ of RS_RSZ2_RSZ2_RSZ1_RSZ2_RSZ2_RSZ2

#### Relational analysis RSZ result of RS_RSZ2_RSZ2_RSZ1_RSZ2_RSZ2_RSZ2
Status: Status.UNKNOWN
Output dim: 8, lower bound: -0.0002325, upper bound: 0.0001948
time: 0.49 seconds

## BFS RS instance: RS_RSZ2_RSZ2_RSZ2_RSZ1_RSZ1

### Backsubstitution after applying RS history:
0: 0.0035972, 0.0041150, 0.0035972, 0.0041150, -0.0001293, 0.0001299
1: 0.0018420, 0.0019168, 0.0018420, 0.0019168, -0.0000187, 0.0000188
2: 0.0120848, 0.0123711, 0.0120848, 0.0123711, -0.0000718, 0.0000715
3: -0.0021818, -0.0018857, -0.0021818, -0.0018857, -0.0000743, 0.0000739
4: -0.0019956, -0.0016750, -0.0019956, -0.0016750, -0.0000800, 0.0000804
5: 0.0056985, 0.0060018, 0.0056985, 0.0060018, -0.0000761, 0.0000757
6: 0.0003095, 0.0015131, 0.0003095, 0.0015131, -0.0003019, 0.0003005
7: -0.0046174, -0.0029782, -0.0046174, -0.0029782, -0.0004092, 0.0004112
8: 0.9859612, 0.9871160, 0.9859612, 0.9871160, -0.0002883, 0.0002897
9: -0.0041920, -0.0031438, -0.0041920, -0.0031438, -0.0002629, 0.0002617

### Unstable ReLU Count (Linear/Conv2D Layers)
- layer_idx=0, type=LayerType.Linear, total=256, inp1_unstable=4, inp2_unstable=4, delta_unstable=252
- layer_idx=2, type=LayerType.Linear, total=256, inp1_unstable=4, inp2_unstable=4, delta_unstable=255
- layer_idx=4, type=LayerType.Linear, total=256, inp1_unstable=0, inp2_unstable=0, delta_unstable=256
- layer_idx=6, type=LayerType.Linear, total=256, inp1_unstable=0, inp2_unstable=0, delta_unstable=256
- layer_idx=8, type=LayerType.Linear, total=10, inp1_unstable=0, inp2_unstable=0, delta_unstable=10

Time for backsubstitution: 1.48 seconds

### RS candidates at layer 3
type: RSZ, layer: 3, pos: 83
type: RSZ, layer: 3, pos: 34
type: RSZ, layer: 3, pos: 49

Time for candidate selection: 0.00 seconds

### Candidate
type: RSZ, layer: 3, pos: 83

### Relational analysis RSZ of RS_RSZ2_RSZ2_RSZ2_RSZ1_RSZ1_RSZ1

#### Relational analysis RSZ result of RS_RSZ2_RSZ2_RSZ2_RSZ1_RSZ1_RSZ1
Status: Status.UNKNOWN
Output dim: 8, lower bound: -0.0001963, upper bound: 0.0002326
time: 0.51 seconds

### Relational analysis RSZ of RS_RSZ2_RSZ2_RSZ2_RSZ1_RSZ1_RSZ2

#### Relational analysis RSZ result of RS_RSZ2_RSZ2_RSZ2_RSZ1_RSZ1_RSZ2
Status: Status.UNKNOWN
Output dim: 8, lower bound: -0.0002321, upper bound: 0.0001981
time: 0.52 seconds

## BFS RS instance: RS_RSZ2_RSZ2_RSZ2_RSZ1_RSZ2

### Backsubstitution after applying RS history:
0: 0.0035972, 0.0041150, 0.0035972, 0.0041150, -0.0001294, 0.0001298
1: 0.0018420, 0.0019168, 0.0018420, 0.0019168, -0.0000187, 0.0000188
2: 0.0120848, 0.0123711, 0.0120848, 0.0123711, -0.0000718, 0.0000716
3: -0.0021818, -0.0018857, -0.0021818, -0.0018857, -0.0000742, 0.0000740
4: -0.0019956, -0.0016750, -0.0019956, -0.0016750, -0.0000801, 0.0000804
5: 0.0056985, 0.0060018, 0.0056985, 0.0060018, -0.0000760, 0.0000758
6: 0.0003095, 0.0015131, 0.0003095, 0.0015131, -0.0003017, 0.0003009
7: -0.0046174, -0.0029782, -0.0046174, -0.0029782, -0.0004098, 0.0004109
8: 0.9859612, 0.9871160, 0.9859612, 0.9871160, -0.0002886, 0.0002895
9: -0.0041920, -0.0031438, -0.0041920, -0.0031438, -0.0002628, 0.0002620

### Unstable ReLU Count (Linear/Conv2D Layers)
- layer_idx=0, type=LayerType.Linear, total=256, inp1_unstable=4, inp2_unstable=4, delta_unstable=252
- layer_idx=2, type=LayerType.Linear, total=256, inp1_unstable=4, inp2_unstable=4, delta_unstable=255
- layer_idx=4, type=LayerType.Linear, total=256, inp1_unstable=0, inp2_unstable=0, delta_unstable=256
- layer_idx=6, type=LayerType.Linear, total=256, inp1_unstable=0, inp2_unstable=0, delta_unstable=256
- layer_idx=8, type=LayerType.Linear, total=10, inp1_unstable=0, inp2_unstable=0, delta_unstable=10

Time for backsubstitution: 1.38 seconds

### RS candidates at layer 3
type: RSZ, layer: 3, pos: 49
type: RSZ, layer: 3, pos: 83
type: RSZ, layer: 3, pos: 34

Time for candidate selection: 0.00 seconds

### Candidate
type: RSZ, layer: 3, pos: 49

### Relational analysis RSZ of RS_RSZ2_RSZ2_RSZ2_RSZ1_RSZ2_RSZ1

#### Relational analysis RSZ result of RS_RSZ2_RSZ2_RSZ2_RSZ1_RSZ2_RSZ1
Status: Status.UNKNOWN
Output dim: 8, lower bound: -0.0001613, upper bound: 0.0002281
time: 0.52 seconds

### Relational analysis RSZ of RS_RSZ2_RSZ2_RSZ2_RSZ1_RSZ2_RSZ2

#### Relational analysis RSZ result of RS_RSZ2_RSZ2_RSZ2_RSZ1_RSZ2_RSZ2
Status: Status.UNKNOWN
Output dim: 8, lower bound: -0.0002274, upper bound: 0.0001750
time: 0.52 seconds

## BFS RS instance: RS_RSZ2_RSZ2_RSZ2_RSZ2_RSZ1

### Backsubstitution after applying RS history:
0: 0.0035972, 0.0041150, 0.0035972, 0.0041150, -0.0001327, 0.0001280
1: 0.0018420, 0.0019168, 0.0018420, 0.0019168, -0.0000192, 0.0000185
2: 0.0120848, 0.0123711, 0.0120848, 0.0123711, -0.0000708, 0.0000734
3: -0.0021818, -0.0018857, -0.0021818, -0.0018857, -0.0000732, 0.0000759
4: -0.0019956, -0.0016750, -0.0019956, -0.0016750, -0.0000822, 0.0000792
5: 0.0056985, 0.0060018, 0.0056985, 0.0060018, -0.0000750, 0.0000777
6: 0.0003095, 0.0015131, 0.0003095, 0.0015131, -0.0002974, 0.0003085
7: -0.0046174, -0.0029782, -0.0046174, -0.0029782, -0.0004201, 0.0004051
8: 0.9859612, 0.9871160, 0.9859612, 0.9871160, -0.0002959, 0.0002854
9: -0.0041920, -0.0031438, -0.0041920, -0.0031438, -0.0002590, 0.0002686

### Unstable ReLU Count (Linear/Conv2D Layers)
- layer_idx=0, type=LayerType.Linear, total=256, inp1_unstable=4, inp2_unstable=4, delta_unstable=252
- layer_idx=2, type=LayerType.Linear, total=256, inp1_unstable=4, inp2_unstable=4, delta_unstable=255
- layer_idx=4, type=LayerType.Linear, total=256, inp1_unstable=0, inp2_unstable=0, delta_unstable=256
- layer_idx=6, type=LayerType.Linear, total=256, inp1_unstable=0, inp2_unstable=0, delta_unstable=256
- layer_idx=8, type=LayerType.Linear, total=10, inp1_unstable=0, inp2_unstable=0, delta_unstable=10

Time for backsubstitution: 1.39 seconds

### RS candidates at layer 3
type: RSZ, layer: 3, pos: 83
type: RSZ, layer: 3, pos: 49
type: RSZ, layer: 3, pos: 34

Time for candidate selection: 0.00 seconds

### Candidate
type: RSZ, layer: 3, pos: 83

### Relational analysis RSZ of RS_RSZ2_RSZ2_RSZ2_RSZ2_RSZ1_RSZ1

#### Relational analysis RSZ result of RS_RSZ2_RSZ2_RSZ2_RSZ2_RSZ1_RSZ1
Status: Status.UNKNOWN
Output dim: 8, lower bound: -0.0002145, upper bound: 0.0002301
time: 0.49 seconds

### Relational analysis RSZ of RS_RSZ2_RSZ2_RSZ2_RSZ2_RSZ1_RSZ2

#### Relational analysis RSZ result of RS_RSZ2_RSZ2_RSZ2_RSZ2_RSZ1_RSZ2
Status: Status.UNKNOWN
Output dim: 8, lower bound: -0.0002376, upper bound: 0.0001863
time: 0.51 seconds

## BFS RS instance: RS_RSZ2_RSZ2_RSZ2_RSZ2_RSZ2

### Backsubstitution after applying RS history:
0: 0.0035972, 0.0041150, 0.0035972, 0.0041150, -0.0001328, 0.0001272
1: 0.0018420, 0.0019168, 0.0018420, 0.0019168, -0.0000192, 0.0000184
2: 0.0120848, 0.0123711, 0.0120848, 0.0123711, -0.0000703, 0.0000734
3: -0.0021818, -0.0018857, -0.0021818, -0.0018857, -0.0000727, 0.0000759
4: -0.0019956, -0.0016750, -0.0019956, -0.0016750, -0.0000822, 0.0000787
5: 0.0056985, 0.0060018, 0.0056985, 0.0060018, -0.0000745, 0.0000778
6: 0.0003095, 0.0015131, 0.0003095, 0.0015131, -0.0002955, 0.0003086
7: -0.0046174, -0.0029782, -0.0046174, -0.0029782, -0.0004202, 0.0004025
8: 0.9859612, 0.9871160, 0.9859612, 0.9871160, -0.0002960, 0.0002835
9: -0.0041920, -0.0031438, -0.0041920, -0.0031438, -0.0002574, 0.0002687

### Unstable ReLU Count (Linear/Conv2D Layers)
- layer_idx=0, type=LayerType.Linear, total=256, inp1_unstable=4, inp2_unstable=4, delta_unstable=252
- layer_idx=2, type=LayerType.Linear, total=256, inp1_unstable=4, inp2_unstable=4, delta_unstable=255
- layer_idx=4, type=LayerType.Linear, total=256, inp1_unstable=0, inp2_unstable=0, delta_unstable=256
- layer_idx=6, type=LayerType.Linear, total=256, inp1_unstable=0, inp2_unstable=0, delta_unstable=256
- layer_idx=8, type=LayerType.Linear, total=10, inp1_unstable=0, inp2_unstable=0, delta_unstable=10

Time for backsubstitution: 1.52 seconds

### RS candidates at layer 3
type: RSZ, layer: 3, pos: 34
type: RSZ, layer: 3, pos: 49
type: RSZ, layer: 3, pos: 83

Time for candidate selection: 0.00 seconds

### Candidate
type: RSZ, layer: 3, pos: 34

### Relational analysis RSZ of RS_RSZ2_RSZ2_RSZ2_RSZ2_RSZ2_RSZ1

#### Relational analysis RSZ result of RS_RSZ2_RSZ2_RSZ2_RSZ2_RSZ2_RSZ1
Status: Status.UNKNOWN
Output dim: 8, lower bound: -0.0002444, upper bound: 0.0002270
time: 0.49 seconds

### Relational analysis RSZ of RS_RSZ2_RSZ2_RSZ2_RSZ2_RSZ2_RSZ2

#### Relational analysis RSZ result of RS_RSZ2_RSZ2_RSZ2_RSZ2_RSZ2_RSZ2
Status: Status.UNKNOWN
Output dim: 8, lower bound: -0.0002409, upper bound: 0.0002280
time: 0.48 seconds

## Summary of splitting (split count: 5)
- Time for RS candidates: 2.50 seconds
RS_RSZ1_RSZ1_RSZ1_RSZ1_RSZ1_RSZ1, status: Status.UNKNOWN, split count: 6, time: 2.50
Output dim: 8, lower bound: -0.0001741, upper bound: 0.0002229
RS_RSZ1_RSZ1_RSZ1_RSZ1_RSZ1_RSZ2, status: Status.UNKNOWN, split count: 6, time: 2.50
Output dim: 8, lower bound: -0.0002179, upper bound: 0.0002029
RS_RSZ1_RSZ1_RSZ1_RSZ1_RSZ2_RSZ1, status: Status.UNKNOWN, split count: 6, time: 2.50
Output dim: 8, lower bound: -0.0001577, upper bound: 0.0002224
RS_RSZ1_RSZ1_RSZ1_RSZ1_RSZ2_RSZ2, status: Status.VERIFIED, split count: 6, time: 2.50
Output dim: 8, lower bound: -0.0002025, upper bound: 0.0001562
RS_RSZ1_RSZ1_RSZ1_RSZ2_RSZ1_RSZ1, status: Status.UNKNOWN, split count: 6, time: 2.50
Output dim: 8, lower bound: -0.0002409, upper bound: 0.0002333
RS_RSZ1_RSZ1_RSZ1_RSZ2_RSZ1_RSZ2, status: Status.UNKNOWN, split count: 6, time: 2.50
Output dim: 8, lower bound: -0.0002409, upper bound: 0.0002333
RS_RSZ1_RSZ1_RSZ1_RSZ2_RSZ2_RSZ1, status: Status.VERIFIED, split count: 6, time: 2.50
Output dim: 8, lower bound: -0.0001633, upper bound: 0.0002153
RS_RSZ1_RSZ1_RSZ1_RSZ2_RSZ2_RSZ2, status: Status.VERIFIED, split count: 6, time: 2.50
Output dim: 8, lower bound: -0.0002126, upper bound: 0.0001501
RS_RSZ1_RSZ1_RSZ2_RSZ1_RSZ1_RSZ1, status: Status.UNKNOWN, split count: 6, time: 2.50
Output dim: 8, lower bound: -0.0001691, upper bound: 0.0002282
RS_RSZ1_RSZ1_RSZ2_RSZ1_RSZ1_RSZ2, status: Status.UNKNOWN, split count: 6, time: 2.50
Output dim: 8, lower bound: -0.0001691, upper bound: 0.0002281
RS_RSZ1_RSZ1_RSZ2_RSZ1_RSZ2_RSZ1, status: Status.VERIFIED, split count: 6, time: 2.50
Output dim: 8, lower bound: -0.0001655, upper bound: 0.0001497
RS_RSZ1_RSZ1_RSZ2_RSZ1_RSZ2_RSZ2, status: Status.VERIFIED, split count: 6, time: 2.50
Output dim: 8, lower bound: -0.0002112, upper bound: 0.0001490
RS_RSZ1_RSZ1_RSZ2_RSZ2_RSZ1_RSZ1, status: Status.UNKNOWN, split count: 6, time: 2.50
Output dim: 8, lower bound: -0.0001748, upper bound: 0.0002200
RS_RSZ1_RSZ1_RSZ2_RSZ2_RSZ1_RSZ2, status: Status.UNKNOWN, split count: 6, time: 2.50
Output dim: 8, lower bound: -0.0002338, upper bound: 0.0001614
RS_RSZ1_RSZ1_RSZ2_RSZ2_RSZ2_RSZ1, status: Status.VERIFIED, split count: 6, time: 2.50
Output dim: 8, lower bound: -0.0001737, upper bound: 0.0002147
RS_RSZ1_RSZ1_RSZ2_RSZ2_RSZ2_RSZ2, status: Status.UNKNOWN, split count: 6, time: 2.50
Output dim: 8, lower bound: -0.0002343, upper bound: 0.0001628
RS_RSZ1_RSZ2_RSZ1_RSZ1_RSZ1_RSZ1, status: Status.UNKNOWN, split count: 6, time: 2.50
Output dim: 8, lower bound: -0.0001858, upper bound: 0.0002380
RS_RSZ1_RSZ2_RSZ1_RSZ1_RSZ1_RSZ2, status: Status.UNKNOWN, split count: 6, time: 2.50
Output dim: 8, lower bound: -0.0001829, upper bound: 0.0002375
RS_RSZ1_RSZ2_RSZ1_RSZ1_RSZ2_RSZ1, status: Status.UNKNOWN, split count: 6, time: 2.50
Output dim: 8, lower bound: -0.0002170, upper bound: 0.0002016
RS_RSZ1_RSZ2_RSZ1_RSZ1_RSZ2_RSZ2, status: Status.VERIFIED, split count: 6, time: 2.50
Output dim: 8, lower bound: -0.0002134, upper bound: 0.0002015
RS_RSZ1_RSZ2_RSZ1_RSZ2_RSZ1_RSZ1, status: Status.VERIFIED, split count: 6, time: 2.50
Output dim: 8, lower bound: -0.0001421, upper bound: 0.0002075
RS_RSZ1_RSZ2_RSZ1_RSZ2_RSZ1_RSZ2, status: Status.VERIFIED, split count: 6, time: 2.50
Output dim: 8, lower bound: -0.0001652, upper bound: 0.0001448
RS_RSZ1_RSZ2_RSZ1_RSZ2_RSZ2_RSZ1, status: Status.UNKNOWN, split count: 6, time: 2.50
Output dim: 8, lower bound: -0.0002212, upper bound: 0.0001937
RS_RSZ1_RSZ2_RSZ1_RSZ2_RSZ2_RSZ2, status: Status.UNKNOWN, split count: 6, time: 2.50
Output dim: 8, lower bound: -0.0002187, upper bound: 0.0001935
RS_RSZ1_RSZ2_RSZ2_RSZ1_RSZ1_RSZ1, status: Status.UNKNOWN, split count: 6, time: 2.50
Output dim: 8, lower bound: -0.0001795, upper bound: 0.0002276
RS_RSZ1_RSZ2_RSZ2_RSZ1_RSZ1_RSZ2, status: Status.UNKNOWN, split count: 6, time: 2.50
Output dim: 8, lower bound: -0.0002281, upper bound: 0.0001598
RS_RSZ1_RSZ2_RSZ2_RSZ1_RSZ2_RSZ1, status: Status.UNKNOWN, split count: 6, time: 2.50
Output dim: 8, lower bound: -0.0001757, upper bound: 0.0002245
RS_RSZ1_RSZ2_RSZ2_RSZ1_RSZ2_RSZ2, status: Status.UNKNOWN, split count: 6, time: 2.50
Output dim: 8, lower bound: -0.0002282, upper bound: 0.0001600
RS_RSZ1_RSZ2_RSZ2_RSZ2_RSZ1_RSZ1, status: Status.VERIFIED, split count: 6, time: 2.50
Output dim: 8, lower bound: -0.0001643, upper bound: 0.0002013
RS_RSZ1_RSZ2_RSZ2_RSZ2_RSZ1_RSZ2, status: Status.VERIFIED, split count: 6, time: 2.50
Output dim: 8, lower bound: -0.0001683, upper bound: 0.0002097
RS_RSZ1_RSZ2_RSZ2_RSZ2_RSZ2_RSZ1, status: Status.UNKNOWN, split count: 6, time: 2.50
Output dim: 8, lower bound: -0.0002339, upper bound: 0.0001600
RS_RSZ1_RSZ2_RSZ2_RSZ2_RSZ2_RSZ2, status: Status.UNKNOWN, split count: 6, time: 2.50
Output dim: 8, lower bound: -0.0002343, upper bound: 0.0001602
RS_RSZ2_RSZ1_RSZ1_RSZ1_RSZ1_RSZ1, status: Status.UNKNOWN, split count: 6, time: 2.50
Output dim: 8, lower bound: -0.0001871, upper bound: 0.0002379
RS_RSZ2_RSZ1_RSZ1_RSZ1_RSZ1_RSZ2, status: Status.UNKNOWN, split count: 6, time: 2.50
Output dim: 8, lower bound: -0.0001869, upper bound: 0.0002376
RS_RSZ2_RSZ1_RSZ1_RSZ1_RSZ2_RSZ1, status: Status.VERIFIED, split count: 6, time: 2.50
Output dim: 8, lower bound: -0.0001424, upper bound: 0.0001859
RS_RSZ2_RSZ1_RSZ1_RSZ1_RSZ2_RSZ2, status: Status.VERIFIED, split count: 6, time: 2.50
Output dim: 8, lower bound: -0.0002056, upper bound: 0.0001545
RS_RSZ2_RSZ1_RSZ1_RSZ2_RSZ1_RSZ1, status: Status.UNKNOWN, split count: 6, time: 2.50
Output dim: 8, lower bound: -0.0001848, upper bound: 0.0002183
RS_RSZ2_RSZ1_RSZ1_RSZ2_RSZ1_RSZ2, status: Status.UNKNOWN, split count: 6, time: 2.50
Output dim: 8, lower bound: -0.0001846, upper bound: 0.0002204
RS_RSZ2_RSZ1_RSZ1_RSZ2_RSZ2_RSZ1, status: Status.UNKNOWN, split count: 6, time: 2.50
Output dim: 8, lower bound: -0.0002318, upper bound: 0.0001955
RS_RSZ2_RSZ1_RSZ1_RSZ2_RSZ2_RSZ2, status: Status.UNKNOWN, split count: 6, time: 2.50
Output dim: 8, lower bound: -0.0002334, upper bound: 0.0002055
RS_RSZ2_RSZ1_RSZ2_RSZ1_RSZ1_RSZ1, status: Status.UNKNOWN, split count: 6, time: 2.50
Output dim: 8, lower bound: -0.0001935, upper bound: 0.0002187
RS_RSZ2_RSZ1_RSZ2_RSZ1_RSZ1_RSZ2, status: Status.UNKNOWN, split count: 6, time: 2.50
Output dim: 8, lower bound: -0.0001937, upper bound: 0.0002212
RS_RSZ2_RSZ1_RSZ2_RSZ1_RSZ2_RSZ1, status: Status.UNKNOWN, split count: 6, time: 2.50
Output dim: 8, lower bound: -0.0002204, upper bound: 0.0001828
RS_RSZ2_RSZ1_RSZ2_RSZ1_RSZ2_RSZ2, status: Status.UNKNOWN, split count: 6, time: 2.50
Output dim: 8, lower bound: -0.0002177, upper bound: 0.0001822
RS_RSZ2_RSZ1_RSZ2_RSZ2_RSZ1_RSZ1, status: Status.VERIFIED, split count: 6, time: 2.50
Output dim: 8, lower bound: -0.0002015, upper bound: 0.0002134
RS_RSZ2_RSZ1_RSZ2_RSZ2_RSZ1_RSZ2, status: Status.UNKNOWN, split count: 6, time: 2.50
Output dim: 8, lower bound: -0.0002257, upper bound: 0.0001739
RS_RSZ2_RSZ1_RSZ2_RSZ2_RSZ2_RSZ1, status: Status.UNKNOWN, split count: 6, time: 2.50
Output dim: 8, lower bound: -0.0002415, upper bound: 0.0002323
RS_RSZ2_RSZ1_RSZ2_RSZ2_RSZ2_RSZ2, status: Status.UNKNOWN, split count: 6, time: 2.50
Output dim: 8, lower bound: -0.0002409, upper bound: 0.0002281
RS_RSZ2_RSZ2_RSZ1_RSZ1_RSZ1_RSZ1, status: Status.UNKNOWN, split count: 6, time: 2.50
Output dim: 8, lower bound: -0.0001742, upper bound: 0.0002230
RS_RSZ2_RSZ2_RSZ1_RSZ1_RSZ1_RSZ2, status: Status.UNKNOWN, split count: 6, time: 2.50
Output dim: 8, lower bound: -0.0001741, upper bound: 0.0002257
RS_RSZ2_RSZ2_RSZ1_RSZ1_RSZ2_RSZ1, status: Status.UNKNOWN, split count: 6, time: 2.50
Output dim: 8, lower bound: -0.0002258, upper bound: 0.0002054
RS_RSZ2_RSZ2_RSZ1_RSZ1_RSZ2_RSZ2, status: Status.UNKNOWN, split count: 6, time: 2.50
Output dim: 8, lower bound: -0.0002294, upper bound: 0.0002135
RS_RSZ2_RSZ2_RSZ1_RSZ2_RSZ1_RSZ1, status: Status.UNKNOWN, split count: 6, time: 2.50
Output dim: 8, lower bound: -0.0002055, upper bound: 0.0002334
RS_RSZ2_RSZ2_RSZ1_RSZ2_RSZ1_RSZ2, status: Status.UNKNOWN, split count: 6, time: 2.50
Output dim: 8, lower bound: -0.0002323, upper bound: 0.0001946
RS_RSZ2_RSZ2_RSZ1_RSZ2_RSZ2_RSZ1, status: Status.UNKNOWN, split count: 6, time: 2.50
Output dim: 8, lower bound: -0.0001955, upper bound: 0.0002318
RS_RSZ2_RSZ2_RSZ1_RSZ2_RSZ2_RSZ2, status: Status.UNKNOWN, split count: 6, time: 2.50
Output dim: 8, lower bound: -0.0002325, upper bound: 0.0001948
RS_RSZ2_RSZ2_RSZ2_RSZ1_RSZ1_RSZ1, status: Status.UNKNOWN, split count: 6, time: 2.50
Output dim: 8, lower bound: -0.0001963, upper bound: 0.0002326
RS_RSZ2_RSZ2_RSZ2_RSZ1_RSZ1_RSZ2, status: Status.UNKNOWN, split count: 6, time: 2.50
Output dim: 8, lower bound: -0.0002321, upper bound: 0.0001981
RS_RSZ2_RSZ2_RSZ2_RSZ1_RSZ2_RSZ1, status: Status.UNKNOWN, split count: 6, time: 2.50
Output dim: 8, lower bound: -0.0001613, upper bound: 0.0002281
RS_RSZ2_RSZ2_RSZ2_RSZ1_RSZ2_RSZ2, status: Status.UNKNOWN, split count: 6, time: 2.50
Output dim: 8, lower bound: -0.0002274, upper bound: 0.0001750
RS_RSZ2_RSZ2_RSZ2_RSZ2_RSZ1_RSZ1, status: Status.UNKNOWN, split count: 6, time: 2.50
Output dim: 8, lower bound: -0.0002145, upper bound: 0.0002301
RS_RSZ2_RSZ2_RSZ2_RSZ2_RSZ1_RSZ2, status: Status.UNKNOWN, split count: 6, time: 2.50
Output dim: 8, lower bound: -0.0002376, upper bound: 0.0001863
RS_RSZ2_RSZ2_RSZ2_RSZ2_RSZ2_RSZ1, status: Status.UNKNOWN, split count: 6, time: 2.50
Output dim: 8, lower bound: -0.0002444, upper bound: 0.0002270
RS_RSZ2_RSZ2_RSZ2_RSZ2_RSZ2_RSZ2, status: Status.UNKNOWN, split count: 6, time: 2.50
Output dim: 8, lower bound: -0.0002409, upper bound: 0.0002280

## BFS RS instance: RS_RSZ1_RSZ1_RSZ1_RSZ1_RSZ1_RSZ1

### Backsubstitution after applying RS history:
0: 0.0035972, 0.0041150, 0.0035972, 0.0041150, -0.0001246, 0.0001333
1: 0.0018420, 0.0019168, 0.0018420, 0.0019168, -0.0000180, 0.0000193
2: 0.0120848, 0.0123711, 0.0120848, 0.0123711, -0.0000737, 0.0000689
3: -0.0021818, -0.0018857, -0.0021818, -0.0018857, -0.0000762, 0.0000712
4: -0.0019956, -0.0016750, -0.0019956, -0.0016750, -0.0000771, 0.0000825
5: 0.0056985, 0.0060018, 0.0056985, 0.0060018, -0.0000781, 0.0000730
6: 0.0003095, 0.0015131, 0.0003095, 0.0015131, -0.0003097, 0.0002896
7: -0.0046174, -0.0029782, -0.0046174, -0.0029782, -0.0003944, 0.0004218
8: 0.9859612, 0.9871160, 0.9859612, 0.9871160, -0.0002778, 0.0002972
9: -0.0041920, -0.0031438, -0.0041920, -0.0031438, -0.0002697, 0.0002522

### Unstable ReLU Count (Linear/Conv2D Layers)
- layer_idx=0, type=LayerType.Linear, total=256, inp1_unstable=4, inp2_unstable=4, delta_unstable=252
- layer_idx=2, type=LayerType.Linear, total=256, inp1_unstable=4, inp2_unstable=4, delta_unstable=255
- layer_idx=4, type=LayerType.Linear, total=256, inp1_unstable=0, inp2_unstable=0, delta_unstable=256
- layer_idx=6, type=LayerType.Linear, total=256, inp1_unstable=0, inp2_unstable=0, delta_unstable=256
- layer_idx=8, type=LayerType.Linear, total=10, inp1_unstable=0, inp2_unstable=0, delta_unstable=10

Time for backsubstitution: 1.37 seconds

### RS candidates at layer 3
type: RSZ, layer: 3, pos: 49
type: RSZ, layer: 3, pos: 240
type: RSZ, layer: 3, pos: 34

Time for candidate selection: 0.00 seconds

### Candidate
type: RSZ, layer: 3, pos: 49

### Relational analysis RSZ of RS_RSZ1_RSZ1_RSZ1_RSZ1_RSZ1_RSZ1_RSZ1

#### Relational analysis RSZ result of RS_RSZ1_RSZ1_RSZ1_RSZ1_RSZ1_RSZ1_RSZ1
Status: Status.VERIFIED
Output dim: 8, lower bound: -0.0001260, upper bound: 0.0001992
time: 0.54 seconds

### Relational analysis RSZ of RS_RSZ1_RSZ1_RSZ1_RSZ1_RSZ1_RSZ1_RSZ2

#### Relational analysis RSZ result of RS_RSZ1_RSZ1_RSZ1_RSZ1_RSZ1_RSZ1_RSZ2
Status: Status.VERIFIED
Output dim: 8, lower bound: -0.0001423, upper bound: 0.0001379
time: 0.53 seconds

## BFS RS instance: RS_RSZ1_RSZ1_RSZ1_RSZ1_RSZ1_RSZ2

### Backsubstitution after applying RS history:
0: 0.0035972, 0.0041150, 0.0035972, 0.0041150, -0.0001284, 0.0001294
1: 0.0018420, 0.0019168, 0.0018420, 0.0019168, -0.0000186, 0.0000187
2: 0.0120848, 0.0123711, 0.0120848, 0.0123711, -0.0000715, 0.0000710
3: -0.0021818, -0.0018857, -0.0021818, -0.0018857, -0.0000740, 0.0000734
4: -0.0019956, -0.0016750, -0.0019956, -0.0016750, -0.0000795, 0.0000801
5: 0.0056985, 0.0060018, 0.0056985, 0.0060018, -0.0000758, 0.0000752
6: 0.0003095, 0.0015131, 0.0003095, 0.0015131, -0.0003007, 0.0002985
7: -0.0046174, -0.0029782, -0.0046174, -0.0029782, -0.0004065, 0.0004095
8: 0.9859612, 0.9871160, 0.9859612, 0.9871160, -0.0002863, 0.0002884
9: -0.0041920, -0.0031438, -0.0041920, -0.0031438, -0.0002618, 0.0002599

### Unstable ReLU Count (Linear/Conv2D Layers)
- layer_idx=0, type=LayerType.Linear, total=256, inp1_unstable=4, inp2_unstable=4, delta_unstable=252
- layer_idx=2, type=LayerType.Linear, total=256, inp1_unstable=4, inp2_unstable=4, delta_unstable=255
- layer_idx=4, type=LayerType.Linear, total=256, inp1_unstable=0, inp2_unstable=0, delta_unstable=256
- layer_idx=6, type=LayerType.Linear, total=256, inp1_unstable=0, inp2_unstable=0, delta_unstable=256
- layer_idx=8, type=LayerType.Linear, total=10, inp1_unstable=0, inp2_unstable=0, delta_unstable=10

Time for backsubstitution: 1.37 seconds

### RS candidates at layer 3
type: RSZ, layer: 3, pos: 240
type: RSZ, layer: 3, pos: 34
type: RSZ, layer: 3, pos: 49

Time for candidate selection: 0.00 seconds

### Candidate
type: RSZ, layer: 3, pos: 240

### Relational analysis RSZ of RS_RSZ1_RSZ1_RSZ1_RSZ1_RSZ1_RSZ2_RSZ1

#### Relational analysis RSZ result of RS_RSZ1_RSZ1_RSZ1_RSZ1_RSZ1_RSZ2_RSZ1
Status: Status.VERIFIED
Output dim: 8, lower bound: -0.0002134, upper bound: 0.0001889
time: 0.49 seconds

### Relational analysis RSZ of RS_RSZ1_RSZ1_RSZ1_RSZ1_RSZ1_RSZ2_RSZ2

#### Relational analysis RSZ result of RS_RSZ1_RSZ1_RSZ1_RSZ1_RSZ1_RSZ2_RSZ2
Status: Status.UNKNOWN
Output dim: 8, lower bound: -0.0002177, upper bound: 0.0002028
time: 0.49 seconds

## BFS RS instance: RS_RSZ1_RSZ1_RSZ1_RSZ1_RSZ2_RSZ1

### Backsubstitution after applying RS history:
0: 0.0035972, 0.0041150, 0.0035972, 0.0041150, -0.0001157, 0.0001321
1: 0.0018420, 0.0019168, 0.0018420, 0.0019168, -0.0000167, 0.0000191
2: 0.0120848, 0.0123711, 0.0120848, 0.0123711, -0.0000730, 0.0000640
3: -0.0021818, -0.0018857, -0.0021818, -0.0018857, -0.0000755, 0.0000662
4: -0.0019956, -0.0016750, -0.0019956, -0.0016750, -0.0000716, 0.0000817
5: 0.0056985, 0.0060018, 0.0056985, 0.0060018, -0.0000774, 0.0000678
6: 0.0003095, 0.0015131, 0.0003095, 0.0015131, -0.0003069, 0.0002690
7: -0.0046174, -0.0029782, -0.0046174, -0.0029782, -0.0003664, 0.0004180
8: 0.9859612, 0.9871160, 0.9859612, 0.9871160, -0.0002581, 0.0002945
9: -0.0041920, -0.0031438, -0.0041920, -0.0031438, -0.0002673, 0.0002343

### Unstable ReLU Count (Linear/Conv2D Layers)
- layer_idx=0, type=LayerType.Linear, total=256, inp1_unstable=4, inp2_unstable=4, delta_unstable=252
- layer_idx=2, type=LayerType.Linear, total=256, inp1_unstable=4, inp2_unstable=4, delta_unstable=255
- layer_idx=4, type=LayerType.Linear, total=256, inp1_unstable=0, inp2_unstable=0, delta_unstable=256
- layer_idx=6, type=LayerType.Linear, total=256, inp1_unstable=0, inp2_unstable=0, delta_unstable=256
- layer_idx=8, type=LayerType.Linear, total=10, inp1_unstable=0, inp2_unstable=0, delta_unstable=10

Time for backsubstitution: 1.37 seconds

### RS candidates at layer 3
type: RSZ, layer: 3, pos: 34
type: RSZ, layer: 3, pos: 83
type: RSZ, layer: 3, pos: 240

Time for candidate selection: 0.00 seconds

### Candidate
type: RSZ, layer: 3, pos: 34

### Candidate
type: RSZ, layer: 3, pos: 83

### Relational analysis RSZ of RS_RSZ1_RSZ1_RSZ1_RSZ1_RSZ2_RSZ1_RSZ1

#### Relational analysis RSZ result of RS_RSZ1_RSZ1_RSZ1_RSZ1_RSZ2_RSZ1_RSZ1
Status: Status.VERIFIED
Output dim: 8, lower bound: -0.0001271, upper bound: 0.0002028
time: 0.52 seconds

### Relational analysis RSZ of RS_RSZ1_RSZ1_RSZ1_RSZ1_RSZ2_RSZ1_RSZ2

#### Relational analysis RSZ result of RS_RSZ1_RSZ1_RSZ1_RSZ1_RSZ2_RSZ1_RSZ2
Status: Status.VERIFIED
Output dim: 8, lower bound: -0.0001388, upper bound: 0.0001735
time: 0.53 seconds

## BFS RS instance: RS_RSZ1_RSZ1_RSZ1_RSZ2_RSZ1_RSZ1

### Backsubstitution after applying RS history:
0: 0.0035972, 0.0041150, 0.0035972, 0.0041150, -0.0001298, 0.0001294
1: 0.0018420, 0.0019168, 0.0018420, 0.0019168, -0.0000188, 0.0000187
2: 0.0120848, 0.0123711, 0.0120848, 0.0123711, -0.0000716, 0.0000718
3: -0.0021818, -0.0018857, -0.0021818, -0.0018857, -0.0000740, 0.0000742
4: -0.0019956, -0.0016750, -0.0019956, -0.0016750, -0.0000804, 0.0000801
5: 0.0056985, 0.0060018, 0.0056985, 0.0060018, -0.0000758, 0.0000760
6: 0.0003095, 0.0015131, 0.0003095, 0.0015131, -0.0003009, 0.0003017
7: -0.0046174, -0.0029782, -0.0046174, -0.0029782, -0.0004109, 0.0004098
8: 0.9859612, 0.9871160, 0.9859612, 0.9871160, -0.0002895, 0.0002886
9: -0.0041920, -0.0031438, -0.0041920, -0.0031438, -0.0002620, 0.0002628

### Unstable ReLU Count (Linear/Conv2D Layers)
- layer_idx=0, type=LayerType.Linear, total=256, inp1_unstable=4, inp2_unstable=4, delta_unstable=252
- layer_idx=2, type=LayerType.Linear, total=256, inp1_unstable=4, inp2_unstable=4, delta_unstable=255
- layer_idx=4, type=LayerType.Linear, total=256, inp1_unstable=0, inp2_unstable=0, delta_unstable=256
- layer_idx=6, type=LayerType.Linear, total=256, inp1_unstable=0, inp2_unstable=0, delta_unstable=256
- layer_idx=8, type=LayerType.Linear, total=10, inp1_unstable=0, inp2_unstable=0, delta_unstable=10

Time for backsubstitution: 1.48 seconds

### RS candidates at layer 3
type: RSZ, layer: 3, pos: 83
type: RSZ, layer: 3, pos: 34
type: RSZ, layer: 3, pos: 49

Time for candidate selection: 0.00 seconds

### Candidate
type: RSZ, layer: 3, pos: 83

### Relational analysis RSZ of RS_RSZ1_RSZ1_RSZ1_RSZ2_RSZ1_RSZ1_RSZ1

#### Relational analysis RSZ result of RS_RSZ1_RSZ1_RSZ1_RSZ2_RSZ1_RSZ1_RSZ1
Status: Status.UNKNOWN
Output dim: 8, lower bound: -0.0001933, upper bound: 0.0002175
time: 0.50 seconds

### Relational analysis RSZ of RS_RSZ1_RSZ1_RSZ1_RSZ2_RSZ1_RSZ1_RSZ2

#### Relational analysis RSZ result of RS_RSZ1_RSZ1_RSZ1_RSZ2_RSZ1_RSZ1_RSZ2
Status: Status.UNKNOWN
Output dim: 8, lower bound: -0.0002200, upper bound: 0.0001805
time: 0.49 seconds

## BFS RS instance: RS_RSZ1_RSZ1_RSZ1_RSZ2_RSZ1_RSZ2

### Backsubstitution after applying RS history:
0: 0.0035972, 0.0041150, 0.0035972, 0.0041150, -0.0001299, 0.0001293
1: 0.0018420, 0.0019168, 0.0018420, 0.0019168, -0.0000188, 0.0000187
2: 0.0120848, 0.0123711, 0.0120848, 0.0123711, -0.0000715, 0.0000718
3: -0.0021818, -0.0018857, -0.0021818, -0.0018857, -0.0000739, 0.0000743
4: -0.0019956, -0.0016750, -0.0019956, -0.0016750, -0.0000804, 0.0000800
5: 0.0056985, 0.0060018, 0.0056985, 0.0060018, -0.0000757, 0.0000761
6: 0.0003095, 0.0015131, 0.0003095, 0.0015131, -0.0003005, 0.0003019
7: -0.0046174, -0.0029782, -0.0046174, -0.0029782, -0.0004112, 0.0004092
8: 0.9859612, 0.9871160, 0.9859612, 0.9871160, -0.0002897, 0.0002883
9: -0.0041920, -0.0031438, -0.0041920, -0.0031438, -0.0002617, 0.0002629

### Unstable ReLU Count (Linear/Conv2D Layers)
- layer_idx=0, type=LayerType.Linear, total=256, inp1_unstable=4, inp2_unstable=4, delta_unstable=252
- layer_idx=2, type=LayerType.Linear, total=256, inp1_unstable=4, inp2_unstable=4, delta_unstable=255
- layer_idx=4, type=LayerType.Linear, total=256, inp1_unstable=0, inp2_unstable=0, delta_unstable=256
- layer_idx=6, type=LayerType.Linear, total=256, inp1_unstable=0, inp2_unstable=0, delta_unstable=256
- layer_idx=8, type=LayerType.Linear, total=10, inp1_unstable=0, inp2_unstable=0, delta_unstable=10

Time for backsubstitution: 1.37 seconds

### RS candidates at layer 3
type: RSZ, layer: 3, pos: 49
type: RSZ, layer: 3, pos: 83
type: RSZ, layer: 3, pos: 34

Time for candidate selection: 0.00 seconds

### Candidate
type: RSZ, layer: 3, pos: 49

### Relational analysis RSZ of RS_RSZ1_RSZ1_RSZ1_RSZ2_RSZ1_RSZ2_RSZ1

#### Relational analysis RSZ result of RS_RSZ1_RSZ1_RSZ1_RSZ2_RSZ1_RSZ2_RSZ1
Status: Status.VERIFIED
Output dim: 8, lower bound: -0.0001616, upper bound: 0.0002100
time: 0.49 seconds

### Relational analysis RSZ of RS_RSZ1_RSZ1_RSZ1_RSZ2_RSZ1_RSZ2_RSZ2

#### Relational analysis RSZ result of RS_RSZ1_RSZ1_RSZ1_RSZ2_RSZ1_RSZ2_RSZ2
Status: Status.UNKNOWN
Output dim: 8, lower bound: -0.0002161, upper bound: 0.0001499
time: 0.49 seconds

## BFS RS instance: RS_RSZ1_RSZ1_RSZ2_RSZ1_RSZ1_RSZ1

### Backsubstitution after applying RS history:
0: 0.0035972, 0.0041150, 0.0035972, 0.0041150, -0.0001271, 0.0001305
1: 0.0018420, 0.0019168, 0.0018420, 0.0019168, -0.0000184, 0.0000189
2: 0.0120848, 0.0123711, 0.0120848, 0.0123711, -0.0000722, 0.0000703
3: -0.0021818, -0.0018857, -0.0021818, -0.0018857, -0.0000746, 0.0000727
4: -0.0019956, -0.0016750, -0.0019956, -0.0016750, -0.0000787, 0.0000808
5: 0.0056985, 0.0060018, 0.0056985, 0.0060018, -0.0000764, 0.0000744
6: 0.0003095, 0.0015131, 0.0003095, 0.0015131, -0.0003033, 0.0002953
7: -0.0046174, -0.0029782, -0.0046174, -0.0029782, -0.0004022, 0.0004131
8: 0.9859612, 0.9871160, 0.9859612, 0.9871160, -0.0002833, 0.0002910
9: -0.0041920, -0.0031438, -0.0041920, -0.0031438, -0.0002641, 0.0002572

### Unstable ReLU Count (Linear/Conv2D Layers)
- layer_idx=0, type=LayerType.Linear, total=256, inp1_unstable=4, inp2_unstable=4, delta_unstable=252
- layer_idx=2, type=LayerType.Linear, total=256, inp1_unstable=4, inp2_unstable=4, delta_unstable=255
- layer_idx=4, type=LayerType.Linear, total=256, inp1_unstable=0, inp2_unstable=0, delta_unstable=256
- layer_idx=6, type=LayerType.Linear, total=256, inp1_unstable=0, inp2_unstable=0, delta_unstable=256
- layer_idx=8, type=LayerType.Linear, total=10, inp1_unstable=0, inp2_unstable=0, delta_unstable=10

Time for backsubstitution: 1.45 seconds

### RS candidates at layer 3
type: RSZ, layer: 3, pos: 83
type: RSZ, layer: 3, pos: 34
type: RSZ, layer: 3, pos: 49

Time for candidate selection: 0.00 seconds

### Candidate
type: RSZ, layer: 3, pos: 83

### Relational analysis RSZ of RS_RSZ1_RSZ1_RSZ2_RSZ1_RSZ1_RSZ1_RSZ1

#### Relational analysis RSZ result of RS_RSZ1_RSZ1_RSZ2_RSZ1_RSZ1_RSZ1_RSZ1
Status: Status.VERIFIED
Output dim: 8, lower bound: -0.0001418, upper bound: 0.0002063
time: 0.53 seconds

### Relational analysis RSZ of RS_RSZ1_RSZ1_RSZ2_RSZ1_RSZ1_RSZ1_RSZ2

#### Relational analysis RSZ result of RS_RSZ1_RSZ1_RSZ2_RSZ1_RSZ1_RSZ1_RSZ2
Status: Status.VERIFIED
Output dim: 8, lower bound: -0.0001505, upper bound: 0.0001662
time: 0.53 seconds

## BFS RS instance: RS_RSZ1_RSZ1_RSZ2_RSZ1_RSZ1_RSZ2

### Backsubstitution after applying RS history:
0: 0.0035972, 0.0041150, 0.0035972, 0.0041150, -0.0001275, 0.0001306
1: 0.0018420, 0.0019168, 0.0018420, 0.0019168, -0.0000184, 0.0000189
2: 0.0120848, 0.0123711, 0.0120848, 0.0123711, -0.0000722, 0.0000705
3: -0.0021818, -0.0018857, -0.0021818, -0.0018857, -0.0000747, 0.0000729
4: -0.0019956, -0.0016750, -0.0019956, -0.0016750, -0.0000789, 0.0000809
5: 0.0056985, 0.0060018, 0.0056985, 0.0060018, -0.0000765, 0.0000747
6: 0.0003095, 0.0015131, 0.0003095, 0.0015131, -0.0003036, 0.0002964
7: -0.0046174, -0.0029782, -0.0046174, -0.0029782, -0.0004037, 0.0004135
8: 0.9859612, 0.9871160, 0.9859612, 0.9871160, -0.0002843, 0.0002912
9: -0.0041920, -0.0031438, -0.0041920, -0.0031438, -0.0002644, 0.0002581

### Unstable ReLU Count (Linear/Conv2D Layers)
- layer_idx=0, type=LayerType.Linear, total=256, inp1_unstable=4, inp2_unstable=4, delta_unstable=252
- layer_idx=2, type=LayerType.Linear, total=256, inp1_unstable=4, inp2_unstable=4, delta_unstable=255
- layer_idx=4, type=LayerType.Linear, total=256, inp1_unstable=0, inp2_unstable=0, delta_unstable=256
- layer_idx=6, type=LayerType.Linear, total=256, inp1_unstable=0, inp2_unstable=0, delta_unstable=256
- layer_idx=8, type=LayerType.Linear, total=10, inp1_unstable=0, inp2_unstable=0, delta_unstable=10

Time for backsubstitution: 1.37 seconds

### RS candidates at layer 3
type: RSZ, layer: 3, pos: 83
type: RSZ, layer: 3, pos: 34
type: RSZ, layer: 3, pos: 49

Time for candidate selection: 0.00 seconds

### Candidate
type: RSZ, layer: 3, pos: 83

### Relational analysis RSZ of RS_RSZ1_RSZ1_RSZ2_RSZ1_RSZ1_RSZ2_RSZ1

#### Relational analysis RSZ result of RS_RSZ1_RSZ1_RSZ2_RSZ1_RSZ1_RSZ2_RSZ1
Status: Status.VERIFIED
Output dim: 8, lower bound: -0.0001380, upper bound: 0.0002060
time: 0.51 seconds

### Relational analysis RSZ of RS_RSZ1_RSZ1_RSZ2_RSZ1_RSZ1_RSZ2_RSZ2

#### Relational analysis RSZ result of RS_RSZ1_RSZ1_RSZ2_RSZ1_RSZ1_RSZ2_RSZ2
Status: Status.VERIFIED
Output dim: 8, lower bound: -0.0001504, upper bound: 0.0001741
time: 0.52 seconds

## BFS RS instance: RS_RSZ1_RSZ1_RSZ2_RSZ2_RSZ1_RSZ1

### Backsubstitution after applying RS history:
0: 0.0035972, 0.0041150, 0.0035972, 0.0041150, -0.0001189, 0.0001261
1: 0.0018420, 0.0019168, 0.0018420, 0.0019168, -0.0000172, 0.0000182
2: 0.0120848, 0.0123711, 0.0120848, 0.0123711, -0.0000697, 0.0000657
3: -0.0021818, -0.0018857, -0.0021818, -0.0018857, -0.0000721, 0.0000680
4: -0.0019956, -0.0016750, -0.0019956, -0.0016750, -0.0000736, 0.0000781
5: 0.0056985, 0.0060018, 0.0056985, 0.0060018, -0.0000739, 0.0000696
6: 0.0003095, 0.0015131, 0.0003095, 0.0015131, -0.0002931, 0.0002763
7: -0.0046174, -0.0029782, -0.0046174, -0.0029782, -0.0003762, 0.0003992
8: 0.9859612, 0.9871160, 0.9859612, 0.9871160, -0.0002650, 0.0002812
9: -0.0041920, -0.0031438, -0.0041920, -0.0031438, -0.0002553, 0.0002406

### Unstable ReLU Count (Linear/Conv2D Layers)
- layer_idx=0, type=LayerType.Linear, total=256, inp1_unstable=4, inp2_unstable=4, delta_unstable=252
- layer_idx=2, type=LayerType.Linear, total=256, inp1_unstable=4, inp2_unstable=4, delta_unstable=255
- layer_idx=4, type=LayerType.Linear, total=256, inp1_unstable=0, inp2_unstable=0, delta_unstable=256
- layer_idx=6, type=LayerType.Linear, total=256, inp1_unstable=0, inp2_unstable=0, delta_unstable=256
- layer_idx=8, type=LayerType.Linear, total=10, inp1_unstable=0, inp2_unstable=0, delta_unstable=10

Time for backsubstitution: 1.54 seconds

### RS candidates at layer 3
type: RSZ, layer: 3, pos: 34
type: RSZ, layer: 3, pos: 240
type: RSZ, layer: 3, pos: 83

Time for candidate selection: 0.00 seconds

### Candidate
type: RSZ, layer: 3, pos: 34

### Relational analysis RSZ of RS_RSZ1_RSZ1_RSZ2_RSZ2_RSZ1_RSZ1_RSZ1

#### Relational analysis RSZ result of RS_RSZ1_RSZ1_RSZ2_RSZ2_RSZ1_RSZ1_RSZ1
Status: Status.VERIFIED
Output dim: 8, lower bound: -0.0001611, upper bound: 0.0002005
time: 0.53 seconds

### Relational analysis RSZ of RS_RSZ1_RSZ1_RSZ2_RSZ2_RSZ1_RSZ1_RSZ2

#### Relational analysis RSZ result of RS_RSZ1_RSZ1_RSZ2_RSZ2_RSZ1_RSZ1_RSZ2
Status: Status.VERIFIED
Output dim: 8, lower bound: -0.0001630, upper bound: 0.0002074
time: 0.52 seconds

## BFS RS instance: RS_RSZ1_RSZ1_RSZ2_RSZ2_RSZ1_RSZ2

### Backsubstitution after applying RS history:
0: 0.0035972, 0.0041150, 0.0035972, 0.0041150, -0.0001300, 0.0001167
1: 0.0018420, 0.0019168, 0.0018420, 0.0019168, -0.0000188, 0.0000169
2: 0.0120848, 0.0123711, 0.0120848, 0.0123711, -0.0000645, 0.0000719
3: -0.0021818, -0.0018857, -0.0021818, -0.0018857, -0.0000667, 0.0000743
4: -0.0019956, -0.0016750, -0.0019956, -0.0016750, -0.0000805, 0.0000722
5: 0.0056985, 0.0060018, 0.0056985, 0.0060018, -0.0000684, 0.0000761
6: 0.0003095, 0.0015131, 0.0003095, 0.0015131, -0.0002712, 0.0003021
7: -0.0046174, -0.0029782, -0.0046174, -0.0029782, -0.0004115, 0.0003694
8: 0.9859612, 0.9871160, 0.9859612, 0.9871160, -0.0002898, 0.0002602
9: -0.0041920, -0.0031438, -0.0041920, -0.0031438, -0.0002362, 0.0002631

### Unstable ReLU Count (Linear/Conv2D Layers)
- layer_idx=0, type=LayerType.Linear, total=256, inp1_unstable=4, inp2_unstable=4, delta_unstable=252
- layer_idx=2, type=LayerType.Linear, total=256, inp1_unstable=4, inp2_unstable=4, delta_unstable=255
- layer_idx=4, type=LayerType.Linear, total=256, inp1_unstable=0, inp2_unstable=0, delta_unstable=256
- layer_idx=6, type=LayerType.Linear, total=256, inp1_unstable=0, inp2_unstable=0, delta_unstable=256
- layer_idx=8, type=LayerType.Linear, total=10, inp1_unstable=0, inp2_unstable=0, delta_unstable=10

Time for backsubstitution: 1.41 seconds

### RS candidates at layer 3
type: RSZ, layer: 3, pos: 83
type: RSZ, layer: 3, pos: 34
type: RSZ, layer: 3, pos: 240

Time for candidate selection: 0.00 seconds

### Candidate
type: RSZ, layer: 3, pos: 83

### Relational analysis RSZ of RS_RSZ1_RSZ1_RSZ2_RSZ2_RSZ1_RSZ2_RSZ1

#### Relational analysis RSZ result of RS_RSZ1_RSZ1_RSZ2_RSZ2_RSZ1_RSZ2_RSZ1
Status: Status.VERIFIED
Output dim: 8, lower bound: -0.0001853, upper bound: 0.0001441
time: 0.50 seconds

### Relational analysis RSZ of RS_RSZ1_RSZ1_RSZ2_RSZ2_RSZ1_RSZ2_RSZ2

#### Relational analysis RSZ result of RS_RSZ1_RSZ1_RSZ2_RSZ2_RSZ1_RSZ2_RSZ2
Status: Status.VERIFIED
Output dim: 8, lower bound: -0.0002144, upper bound: 0.0001337
time: 0.51 seconds

## BFS RS instance: RS_RSZ1_RSZ1_RSZ2_RSZ2_RSZ2_RSZ2

### Backsubstitution after applying RS history:
0: 0.0035972, 0.0041150, 0.0035972, 0.0041150, -0.0001300, 0.0001167
1: 0.0018420, 0.0019168, 0.0018420, 0.0019168, -0.0000188, 0.0000169
2: 0.0120848, 0.0123711, 0.0120848, 0.0123711, -0.0000645, 0.0000719
3: -0.0021818, -0.0018857, -0.0021818, -0.0018857, -0.0000667, 0.0000743
4: -0.0019956, -0.0016750, -0.0019956, -0.0016750, -0.0000805, 0.0000722
5: 0.0056985, 0.0060018, 0.0056985, 0.0060018, -0.0000684, 0.0000761
6: 0.0003095, 0.0015131, 0.0003095, 0.0015131, -0.0002712, 0.0003021
7: -0.0046174, -0.0029782, -0.0046174, -0.0029782, -0.0004115, 0.0003694
8: 0.9859612, 0.9871160, 0.9859612, 0.9871160, -0.0002898, 0.0002602
9: -0.0041920, -0.0031438, -0.0041920, -0.0031438, -0.0002362, 0.0002631

### Unstable ReLU Count (Linear/Conv2D Layers)
- layer_idx=0, type=LayerType.Linear, total=256, inp1_unstable=4, inp2_unstable=4, delta_unstable=252
- layer_idx=2, type=LayerType.Linear, total=256, inp1_unstable=4, inp2_unstable=4, delta_unstable=255
- layer_idx=4, type=LayerType.Linear, total=256, inp1_unstable=0, inp2_unstable=0, delta_unstable=256
- layer_idx=6, type=LayerType.Linear, total=256, inp1_unstable=0, inp2_unstable=0, delta_unstable=256
- layer_idx=8, type=LayerType.Linear, total=10, inp1_unstable=0, inp2_unstable=0, delta_unstable=10

Time for backsubstitution: 1.51 seconds

### RS candidates at layer 3
type: RSZ, layer: 3, pos: 240
type: RSZ, layer: 3, pos: 34
type: RSZ, layer: 3, pos: 83

Time for candidate selection: 0.00 seconds

### Candidate
type: RSZ, layer: 3, pos: 240

### Candidate
type: RSZ, layer: 3, pos: 34

### Relational analysis RSZ of RS_RSZ1_RSZ1_RSZ2_RSZ2_RSZ2_RSZ2_RSZ1

#### Relational analysis RSZ result of RS_RSZ1_RSZ1_RSZ2_RSZ2_RSZ2_RSZ2_RSZ1
Status: Status.UNKNOWN
Output dim: 8, lower bound: -0.0002222, upper bound: 0.0001502
time: 0.53 seconds

### Relational analysis RSZ of RS_RSZ1_RSZ1_RSZ2_RSZ2_RSZ2_RSZ2_RSZ2

#### Relational analysis RSZ result of RS_RSZ1_RSZ1_RSZ2_RSZ2_RSZ2_RSZ2_RSZ2
Status: Status.UNKNOWN
Output dim: 8, lower bound: -0.0002193, upper bound: 0.0001504
time: 0.54 seconds

## BFS RS instance: RS_RSZ1_RSZ2_RSZ1_RSZ1_RSZ1_RSZ1

### Backsubstitution after applying RS history:
0: 0.0035972, 0.0041150, 0.0035972, 0.0041150, -0.0001260, 0.0001329
1: 0.0018420, 0.0019168, 0.0018420, 0.0019168, -0.0000182, 0.0000192
2: 0.0120848, 0.0123711, 0.0120848, 0.0123711, -0.0000735, 0.0000697
3: -0.0021818, -0.0018857, -0.0021818, -0.0018857, -0.0000760, 0.0000721
4: -0.0019956, -0.0016750, -0.0019956, -0.0016750, -0.0000780, 0.0000823
5: 0.0056985, 0.0060018, 0.0056985, 0.0060018, -0.0000779, 0.0000738
6: 0.0003095, 0.0015131, 0.0003095, 0.0015131, -0.0003090, 0.0002929
7: -0.0046174, -0.0029782, -0.0046174, -0.0029782, -0.0003989, 0.0004208
8: 0.9859612, 0.9871160, 0.9859612, 0.9871160, -0.0002810, 0.0002964
9: -0.0041920, -0.0031438, -0.0041920, -0.0031438, -0.0002691, 0.0002551

### Unstable ReLU Count (Linear/Conv2D Layers)
- layer_idx=0, type=LayerType.Linear, total=256, inp1_unstable=4, inp2_unstable=4, delta_unstable=252
- layer_idx=2, type=LayerType.Linear, total=256, inp1_unstable=4, inp2_unstable=4, delta_unstable=255
- layer_idx=4, type=LayerType.Linear, total=256, inp1_unstable=0, inp2_unstable=0, delta_unstable=256
- layer_idx=6, type=LayerType.Linear, total=256, inp1_unstable=0, inp2_unstable=0, delta_unstable=256
- layer_idx=8, type=LayerType.Linear, total=10, inp1_unstable=0, inp2_unstable=0, delta_unstable=10

Time for backsubstitution: 1.43 seconds

### RS candidates at layer 3
type: RSZ, layer: 3, pos: 34
type: RSZ, layer: 3, pos: 83
type: RSZ, layer: 3, pos: 49

Time for candidate selection: 0.00 seconds

### Candidate
type: RSZ, layer: 3, pos: 34

### Relational analysis RSZ of RS_RSZ1_RSZ2_RSZ1_RSZ1_RSZ1_RSZ1_RSZ1

#### Relational analysis RSZ result of RS_RSZ1_RSZ2_RSZ1_RSZ1_RSZ1_RSZ1_RSZ1
Status: Status.UNKNOWN
Output dim: 8, lower bound: -0.0001728, upper bound: 0.0002221
time: 0.53 seconds

### Relational analysis RSZ of RS_RSZ1_RSZ2_RSZ1_RSZ1_RSZ1_RSZ1_RSZ2

#### Relational analysis RSZ result of RS_RSZ1_RSZ2_RSZ1_RSZ1_RSZ1_RSZ1_RSZ2
Status: Status.UNKNOWN
Output dim: 8, lower bound: -0.0001737, upper bound: 0.0002256
time: 0.53 seconds

## BFS RS instance: RS_RSZ1_RSZ2_RSZ1_RSZ1_RSZ1_RSZ2

### Backsubstitution after applying RS history:
0: 0.0035972, 0.0041150, 0.0035972, 0.0041150, -0.0001266, 0.0001328
1: 0.0018420, 0.0019168, 0.0018420, 0.0019168, -0.0000183, 0.0000192
2: 0.0120848, 0.0123711, 0.0120848, 0.0123711, -0.0000734, 0.0000700
3: -0.0021818, -0.0018857, -0.0021818, -0.0018857, -0.0000760, 0.0000724
4: -0.0019956, -0.0016750, -0.0019956, -0.0016750, -0.0000783, 0.0000822
5: 0.0056985, 0.0060018, 0.0056985, 0.0060018, -0.0000778, 0.0000741
6: 0.0003095, 0.0015131, 0.0003095, 0.0015131, -0.0003087, 0.0002942
7: -0.0046174, -0.0029782, -0.0046174, -0.0029782, -0.0004007, 0.0004205
8: 0.9859612, 0.9871160, 0.9859612, 0.9871160, -0.0002822, 0.0002962
9: -0.0041920, -0.0031438, -0.0041920, -0.0031438, -0.0002689, 0.0002562

### Unstable ReLU Count (Linear/Conv2D Layers)
- layer_idx=0, type=LayerType.Linear, total=256, inp1_unstable=4, inp2_unstable=4, delta_unstable=252
- layer_idx=2, type=LayerType.Linear, total=256, inp1_unstable=4, inp2_unstable=4, delta_unstable=255
- layer_idx=4, type=LayerType.Linear, total=256, inp1_unstable=0, inp2_unstable=0, delta_unstable=256
- layer_idx=6, type=LayerType.Linear, total=256, inp1_unstable=0, inp2_unstable=0, delta_unstable=256
- layer_idx=8, type=LayerType.Linear, total=10, inp1_unstable=0, inp2_unstable=0, delta_unstable=10

Time for backsubstitution: 1.44 seconds

### RS candidates at layer 3
type: RSZ, layer: 3, pos: 83
type: RSZ, layer: 3, pos: 34
type: RSZ, layer: 3, pos: 49

Time for candidate selection: 0.00 seconds

### Candidate
type: RSZ, layer: 3, pos: 83

### Candidate
type: RSZ, layer: 3, pos: 34

### Relational analysis RSZ of RS_RSZ1_RSZ2_RSZ1_RSZ1_RSZ1_RSZ2_RSZ1

#### Relational analysis RSZ result of RS_RSZ1_RSZ2_RSZ1_RSZ1_RSZ1_RSZ2_RSZ1
Status: Status.UNKNOWN
Output dim: 8, lower bound: -0.0001706, upper bound: 0.0002222
time: 0.53 seconds

### Relational analysis RSZ of RS_RSZ1_RSZ2_RSZ1_RSZ1_RSZ1_RSZ2_RSZ2

#### Relational analysis RSZ result of RS_RSZ1_RSZ2_RSZ1_RSZ1_RSZ1_RSZ2_RSZ2
Status: Status.UNKNOWN
Output dim: 8, lower bound: -0.0001706, upper bound: 0.0002252
time: 0.54 seconds

## BFS RS instance: RS_RSZ1_RSZ2_RSZ1_RSZ1_RSZ2_RSZ1

### Backsubstitution after applying RS history:
0: 0.0035972, 0.0041150, 0.0035972, 0.0041150, -0.0001151, 0.0001203
1: 0.0018420, 0.0019168, 0.0018420, 0.0019168, -0.0000166, 0.0000174
2: 0.0120848, 0.0123711, 0.0120848, 0.0123711, -0.0000665, 0.0000637
3: -0.0021818, -0.0018857, -0.0021818, -0.0018857, -0.0000688, 0.0000658
4: -0.0019956, -0.0016750, -0.0019956, -0.0016750, -0.0000713, 0.0000745
5: 0.0056985, 0.0060018, 0.0056985, 0.0060018, -0.0000705, 0.0000675
6: 0.0003095, 0.0015131, 0.0003095, 0.0015131, -0.0002797, 0.0002676
7: -0.0046174, -0.0029782, -0.0046174, -0.0029782, -0.0003645, 0.0003810
8: 0.9859612, 0.9871160, 0.9859612, 0.9871160, -0.0002568, 0.0002684
9: -0.0041920, -0.0031438, -0.0041920, -0.0031438, -0.0002436, 0.0002331

### Unstable ReLU Count (Linear/Conv2D Layers)
- layer_idx=0, type=LayerType.Linear, total=256, inp1_unstable=4, inp2_unstable=4, delta_unstable=252
- layer_idx=2, type=LayerType.Linear, total=256, inp1_unstable=4, inp2_unstable=4, delta_unstable=255
- layer_idx=4, type=LayerType.Linear, total=256, inp1_unstable=0, inp2_unstable=0, delta_unstable=256
- layer_idx=6, type=LayerType.Linear, total=256, inp1_unstable=0, inp2_unstable=0, delta_unstable=256
- layer_idx=8, type=LayerType.Linear, total=10, inp1_unstable=0, inp2_unstable=0, delta_unstable=10

Time for backsubstitution: 1.47 seconds

### RS candidates at layer 3
type: RSZ, layer: 3, pos: 49
type: RSZ, layer: 3, pos: 240
type: RSZ, layer: 3, pos: 83

Time for candidate selection: 0.00 seconds

### Candidate
type: RSZ, layer: 3, pos: 49

### Relational analysis RSZ of RS_RSZ1_RSZ2_RSZ1_RSZ1_RSZ2_RSZ1_RSZ1

#### Relational analysis RSZ result of RS_RSZ1_RSZ2_RSZ1_RSZ1_RSZ2_RSZ1_RSZ1
Status: Status.VERIFIED
Output dim: 8, lower bound: -0.0001401, upper bound: 0.0001736
time: 0.53 seconds

### Relational analysis RSZ of RS_RSZ1_RSZ2_RSZ1_RSZ1_RSZ2_RSZ1_RSZ2

#### Relational analysis RSZ result of RS_RSZ1_RSZ2_RSZ1_RSZ1_RSZ2_RSZ1_RSZ2
Status: Status.VERIFIED
Output dim: 8, lower bound: -0.0001908, upper bound: 0.0001317
time: 0.52 seconds

## BFS RS instance: RS_RSZ1_RSZ2_RSZ1_RSZ2_RSZ2_RSZ1

### Backsubstitution after applying RS history:
0: 0.0035972, 0.0041150, 0.0035972, 0.0041150, -0.0001159, 0.0001180
1: 0.0018420, 0.0019168, 0.0018420, 0.0019168, -0.0000167, 0.0000170
2: 0.0120848, 0.0123711, 0.0120848, 0.0123711, -0.0000652, 0.0000641
3: -0.0021818, -0.0018857, -0.0021818, -0.0018857, -0.0000675, 0.0000663
4: -0.0019956, -0.0016750, -0.0019956, -0.0016750, -0.0000717, 0.0000730
5: 0.0056985, 0.0060018, 0.0056985, 0.0060018, -0.0000691, 0.0000679
6: 0.0003095, 0.0015131, 0.0003095, 0.0015131, -0.0002743, 0.0002694
7: -0.0046174, -0.0029782, -0.0046174, -0.0029782, -0.0003669, 0.0003735
8: 0.9859612, 0.9871160, 0.9859612, 0.9871160, -0.0002584, 0.0002631
9: -0.0041920, -0.0031438, -0.0041920, -0.0031438, -0.0002388, 0.0002346

### Unstable ReLU Count (Linear/Conv2D Layers)
- layer_idx=0, type=LayerType.Linear, total=256, inp1_unstable=4, inp2_unstable=4, delta_unstable=252
- layer_idx=2, type=LayerType.Linear, total=256, inp1_unstable=4, inp2_unstable=4, delta_unstable=255
- layer_idx=4, type=LayerType.Linear, total=256, inp1_unstable=0, inp2_unstable=0, delta_unstable=256
- layer_idx=6, type=LayerType.Linear, total=256, inp1_unstable=0, inp2_unstable=0, delta_unstable=256
- layer_idx=8, type=LayerType.Linear, total=10, inp1_unstable=0, inp2_unstable=0, delta_unstable=10

Time for backsubstitution: 1.45 seconds

### RS candidates at layer 3
type: RSZ, layer: 3, pos: 240
type: RSZ, layer: 3, pos: 83
type: RSZ, layer: 3, pos: 49

Time for candidate selection: 0.00 seconds

### Candidate
type: RSZ, layer: 3, pos: 240

### Relational analysis RSZ of RS_RSZ1_RSZ2_RSZ1_RSZ2_RSZ2_RSZ1_RSZ1

#### Relational analysis RSZ result of RS_RSZ1_RSZ2_RSZ1_RSZ2_RSZ2_RSZ1_RSZ1
Status: Status.UNKNOWN
Output dim: 8, lower bound: -0.0002200, upper bound: 0.0001850
time: 0.50 seconds

### Relational analysis RSZ of RS_RSZ1_RSZ2_RSZ1_RSZ2_RSZ2_RSZ1_RSZ2

#### Relational analysis RSZ result of RS_RSZ1_RSZ2_RSZ1_RSZ2_RSZ2_RSZ1_RSZ2
Status: Status.UNKNOWN
Output dim: 8, lower bound: -0.0002211, upper bound: 0.0001935
time: 0.50 seconds

## BFS RS instance: RS_RSZ1_RSZ2_RSZ1_RSZ2_RSZ2_RSZ2

### Backsubstitution after applying RS history:
0: 0.0035972, 0.0041150, 0.0035972, 0.0041150, -0.0001152, 0.0001192
1: 0.0018420, 0.0019168, 0.0018420, 0.0019168, -0.0000166, 0.0000172
2: 0.0120848, 0.0123711, 0.0120848, 0.0123711, -0.0000659, 0.0000637
3: -0.0021818, -0.0018857, -0.0021818, -0.0018857, -0.0000681, 0.0000659
4: -0.0019956, -0.0016750, -0.0019956, -0.0016750, -0.0000713, 0.0000738
5: 0.0056985, 0.0060018, 0.0056985, 0.0060018, -0.0000698, 0.0000675
6: 0.0003095, 0.0015131, 0.0003095, 0.0015131, -0.0002770, 0.0002677
7: -0.0046174, -0.0029782, -0.0046174, -0.0029782, -0.0003646, 0.0003773
8: 0.9859612, 0.9871160, 0.9859612, 0.9871160, -0.0002568, 0.0002657
9: -0.0041920, -0.0031438, -0.0041920, -0.0031438, -0.0002412, 0.0002331

### Unstable ReLU Count (Linear/Conv2D Layers)
- layer_idx=0, type=LayerType.Linear, total=256, inp1_unstable=4, inp2_unstable=4, delta_unstable=252
- layer_idx=2, type=LayerType.Linear, total=256, inp1_unstable=4, inp2_unstable=4, delta_unstable=255
- layer_idx=4, type=LayerType.Linear, total=256, inp1_unstable=0, inp2_unstable=0, delta_unstable=256
- layer_idx=6, type=LayerType.Linear, total=256, inp1_unstable=0, inp2_unstable=0, delta_unstable=256
- layer_idx=8, type=LayerType.Linear, total=10, inp1_unstable=0, inp2_unstable=0, delta_unstable=10

Time for backsubstitution: 1.44 seconds

### RS candidates at layer 3
type: RSZ, layer: 3, pos: 49
type: RSZ, layer: 3, pos: 83
type: RSZ, layer: 3, pos: 240

Time for candidate selection: 0.00 seconds

### Candidate
type: RSZ, layer: 3, pos: 49

### Relational analysis RSZ of RS_RSZ1_RSZ2_RSZ1_RSZ2_RSZ2_RSZ2_RSZ1

#### Relational analysis RSZ result of RS_RSZ1_RSZ2_RSZ1_RSZ2_RSZ2_RSZ2_RSZ1
Status: Status.VERIFIED
Output dim: 8, lower bound: -0.0001411, upper bound: 0.0001623
time: 0.51 seconds

### Relational analysis RSZ of RS_RSZ1_RSZ2_RSZ1_RSZ2_RSZ2_RSZ2_RSZ2

#### Relational analysis RSZ result of RS_RSZ1_RSZ2_RSZ1_RSZ2_RSZ2_RSZ2_RSZ2
Status: Status.VERIFIED
Output dim: 8, lower bound: -0.0001940, upper bound: 0.0001308
time: 0.55 seconds

## BFS RS instance: RS_RSZ1_RSZ2_RSZ2_RSZ1_RSZ1_RSZ1

### Backsubstitution after applying RS history:
0: 0.0035972, 0.0041150, 0.0035972, 0.0041150, -0.0001177, 0.0001298
1: 0.0018420, 0.0019168, 0.0018420, 0.0019168, -0.0000170, 0.0000188
2: 0.0120848, 0.0123711, 0.0120848, 0.0123711, -0.0000718, 0.0000651
3: -0.0021818, -0.0018857, -0.0021818, -0.0018857, -0.0000742, 0.0000673
4: -0.0019956, -0.0016750, -0.0019956, -0.0016750, -0.0000728, 0.0000804
5: 0.0056985, 0.0060018, 0.0056985, 0.0060018, -0.0000760, 0.0000689
6: 0.0003095, 0.0015131, 0.0003095, 0.0015131, -0.0003017, 0.0002735
7: -0.0046174, -0.0029782, -0.0046174, -0.0029782, -0.0003725, 0.0004109
8: 0.9859612, 0.9871160, 0.9859612, 0.9871160, -0.0002624, 0.0002894
9: -0.0041920, -0.0031438, -0.0041920, -0.0031438, -0.0002627, 0.0002382

### Unstable ReLU Count (Linear/Conv2D Layers)
- layer_idx=0, type=LayerType.Linear, total=256, inp1_unstable=4, inp2_unstable=4, delta_unstable=252
- layer_idx=2, type=LayerType.Linear, total=256, inp1_unstable=4, inp2_unstable=4, delta_unstable=255
- layer_idx=4, type=LayerType.Linear, total=256, inp1_unstable=0, inp2_unstable=0, delta_unstable=256
- layer_idx=6, type=LayerType.Linear, total=256, inp1_unstable=0, inp2_unstable=0, delta_unstable=256
- layer_idx=8, type=LayerType.Linear, total=10, inp1_unstable=0, inp2_unstable=0, delta_unstable=10

Time for backsubstitution: 1.52 seconds

### RS candidates at layer 3
type: RSZ, layer: 3, pos: 83
type: RSZ, layer: 3, pos: 240
type: RSZ, layer: 3, pos: 34

Time for candidate selection: 0.00 seconds

### Candidate
type: RSZ, layer: 3, pos: 83

### Relational analysis RSZ of RS_RSZ1_RSZ2_RSZ2_RSZ1_RSZ1_RSZ1_RSZ1

#### Relational analysis RSZ result of RS_RSZ1_RSZ2_RSZ2_RSZ1_RSZ1_RSZ1_RSZ1
Status: Status.VERIFIED
Output dim: 8, lower bound: -0.0001526, upper bound: 0.0002110
time: 0.52 seconds

### Relational analysis RSZ of RS_RSZ1_RSZ2_RSZ2_RSZ1_RSZ1_RSZ1_RSZ2

#### Relational analysis RSZ result of RS_RSZ1_RSZ2_RSZ2_RSZ1_RSZ1_RSZ1_RSZ2
Status: Status.VERIFIED
Output dim: 8, lower bound: -0.0001577, upper bound: 0.0001704
time: 0.54 seconds

## BFS RS instance: RS_RSZ1_RSZ2_RSZ2_RSZ1_RSZ1_RSZ2

### Backsubstitution after applying RS history:
0: 0.0035972, 0.0041150, 0.0035972, 0.0041150, -0.0001264, 0.0001178
1: 0.0018420, 0.0019168, 0.0018420, 0.0019168, -0.0000183, 0.0000170
2: 0.0120848, 0.0123711, 0.0120848, 0.0123711, -0.0000652, 0.0000699
3: -0.0021818, -0.0018857, -0.0021818, -0.0018857, -0.0000674, 0.0000723
4: -0.0019956, -0.0016750, -0.0019956, -0.0016750, -0.0000782, 0.0000729
5: 0.0056985, 0.0060018, 0.0056985, 0.0060018, -0.0000690, 0.0000740
6: 0.0003095, 0.0015131, 0.0003095, 0.0015131, -0.0002739, 0.0002938
7: -0.0046174, -0.0029782, -0.0046174, -0.0029782, -0.0004001, 0.0003730
8: 0.9859612, 0.9871160, 0.9859612, 0.9871160, -0.0002818, 0.0002628
9: -0.0041920, -0.0031438, -0.0041920, -0.0031438, -0.0002385, 0.0002558

### Unstable ReLU Count (Linear/Conv2D Layers)
- layer_idx=0, type=LayerType.Linear, total=256, inp1_unstable=4, inp2_unstable=4, delta_unstable=252
- layer_idx=2, type=LayerType.Linear, total=256, inp1_unstable=4, inp2_unstable=4, delta_unstable=255
- layer_idx=4, type=LayerType.Linear, total=256, inp1_unstable=0, inp2_unstable=0, delta_unstable=256
- layer_idx=6, type=LayerType.Linear, total=256, inp1_unstable=0, inp2_unstable=0, delta_unstable=256
- layer_idx=8, type=LayerType.Linear, total=10, inp1_unstable=0, inp2_unstable=0, delta_unstable=10

Time for backsubstitution: 1.51 seconds

### RS candidates at layer 3
type: RSZ, layer: 3, pos: 34
type: RSZ, layer: 3, pos: 240
type: RSZ, layer: 3, pos: 83

Time for candidate selection: 0.00 seconds

### Candidate
type: RSZ, layer: 3, pos: 34

### Relational analysis RSZ of RS_RSZ1_RSZ2_RSZ2_RSZ1_RSZ1_RSZ2_RSZ1

#### Relational analysis RSZ result of RS_RSZ1_RSZ2_RSZ2_RSZ1_RSZ1_RSZ2_RSZ1
Status: Status.UNKNOWN
Output dim: 8, lower bound: -0.0002160, upper bound: 0.0001457
time: 0.50 seconds

### Relational analysis RSZ of RS_RSZ1_RSZ2_RSZ2_RSZ1_RSZ1_RSZ2_RSZ2

#### Relational analysis RSZ result of RS_RSZ1_RSZ2_RSZ2_RSZ1_RSZ1_RSZ2_RSZ2
Status: Status.VERIFIED
Output dim: 8, lower bound: -0.0002127, upper bound: 0.0001472
time: 0.53 seconds

## BFS RS instance: RS_RSZ1_RSZ2_RSZ2_RSZ1_RSZ2_RSZ1

### Backsubstitution after applying RS history:
0: 0.0035972, 0.0041150, 0.0035972, 0.0041150, -0.0001177, 0.0001298
1: 0.0018420, 0.0019168, 0.0018420, 0.0019168, -0.0000170, 0.0000188
2: 0.0120848, 0.0123711, 0.0120848, 0.0123711, -0.0000718, 0.0000651
3: -0.0021818, -0.0018857, -0.0021818, -0.0018857, -0.0000742, 0.0000673
4: -0.0019956, -0.0016750, -0.0019956, -0.0016750, -0.0000728, 0.0000804
5: 0.0056985, 0.0060018, 0.0056985, 0.0060018, -0.0000760, 0.0000689
6: 0.0003095, 0.0015131, 0.0003095, 0.0015131, -0.0003017, 0.0002735
7: -0.0046174, -0.0029782, -0.0046174, -0.0029782, -0.0003725, 0.0004109
8: 0.9859612, 0.9871160, 0.9859612, 0.9871160, -0.0002624, 0.0002894
9: -0.0041920, -0.0031438, -0.0041920, -0.0031438, -0.0002627, 0.0002382

### Unstable ReLU Count (Linear/Conv2D Layers)
- layer_idx=0, type=LayerType.Linear, total=256, inp1_unstable=4, inp2_unstable=4, delta_unstable=252
- layer_idx=2, type=LayerType.Linear, total=256, inp1_unstable=4, inp2_unstable=4, delta_unstable=255
- layer_idx=4, type=LayerType.Linear, total=256, inp1_unstable=0, inp2_unstable=0, delta_unstable=256
- layer_idx=6, type=LayerType.Linear, total=256, inp1_unstable=0, inp2_unstable=0, delta_unstable=256
- layer_idx=8, type=LayerType.Linear, total=10, inp1_unstable=0, inp2_unstable=0, delta_unstable=10

Time for backsubstitution: 1.48 seconds

### RS candidates at layer 3
type: RSZ, layer: 3, pos: 34
type: RSZ, layer: 3, pos: 83
type: RSZ, layer: 3, pos: 240

Time for candidate selection: 0.00 seconds

### Candidate
type: RSZ, layer: 3, pos: 34

### Relational analysis RSZ of RS_RSZ1_RSZ2_RSZ2_RSZ1_RSZ2_RSZ1_RSZ1

#### Relational analysis RSZ result of RS_RSZ1_RSZ2_RSZ2_RSZ1_RSZ2_RSZ1_RSZ1
Status: Status.VERIFIED
Output dim: 8, lower bound: -0.0001630, upper bound: 0.0002095
time: 0.51 seconds

### Relational analysis RSZ of RS_RSZ1_RSZ2_RSZ2_RSZ1_RSZ2_RSZ1_RSZ2

#### Relational analysis RSZ result of RS_RSZ1_RSZ2_RSZ2_RSZ1_RSZ2_RSZ1_RSZ2
Status: Status.VERIFIED
Output dim: 8, lower bound: -0.0001638, upper bound: 0.0002122
time: 0.52 seconds

## BFS RS instance: RS_RSZ1_RSZ2_RSZ2_RSZ1_RSZ2_RSZ2

### Backsubstitution after applying RS history:
0: 0.0035972, 0.0041150, 0.0035972, 0.0041150, -0.0001264, 0.0001178
1: 0.0018420, 0.0019168, 0.0018420, 0.0019168, -0.0000183, 0.0000170
2: 0.0120848, 0.0123711, 0.0120848, 0.0123711, -0.0000652, 0.0000699
3: -0.0021818, -0.0018857, -0.0021818, -0.0018857, -0.0000674, 0.0000723
4: -0.0019956, -0.0016750, -0.0019956, -0.0016750, -0.0000782, 0.0000729
5: 0.0056985, 0.0060018, 0.0056985, 0.0060018, -0.0000690, 0.0000740
6: 0.0003095, 0.0015131, 0.0003095, 0.0015131, -0.0002739, 0.0002938
7: -0.0046174, -0.0029782, -0.0046174, -0.0029782, -0.0004001, 0.0003730
8: 0.9859612, 0.9871160, 0.9859612, 0.9871160, -0.0002818, 0.0002628
9: -0.0041920, -0.0031438, -0.0041920, -0.0031438, -0.0002385, 0.0002558

### Unstable ReLU Count (Linear/Conv2D Layers)
- layer_idx=0, type=LayerType.Linear, total=256, inp1_unstable=4, inp2_unstable=4, delta_unstable=252
- layer_idx=2, type=LayerType.Linear, total=256, inp1_unstable=4, inp2_unstable=4, delta_unstable=255
- layer_idx=4, type=LayerType.Linear, total=256, inp1_unstable=0, inp2_unstable=0, delta_unstable=256
- layer_idx=6, type=LayerType.Linear, total=256, inp1_unstable=0, inp2_unstable=0, delta_unstable=256
- layer_idx=8, type=LayerType.Linear, total=10, inp1_unstable=0, inp2_unstable=0, delta_unstable=10

Time for backsubstitution: 1.42 seconds

### RS candidates at layer 3
type: RSZ, layer: 3, pos: 240
type: RSZ, layer: 3, pos: 34
type: RSZ, layer: 3, pos: 83

Time for candidate selection: 0.00 seconds

### Candidate
type: RSZ, layer: 3, pos: 240

### Candidate
type: RSZ, layer: 3, pos: 34

### Relational analysis RSZ of RS_RSZ1_RSZ2_RSZ2_RSZ1_RSZ2_RSZ2_RSZ1

#### Relational analysis RSZ result of RS_RSZ1_RSZ2_RSZ2_RSZ1_RSZ2_RSZ2_RSZ1
Status: Status.UNKNOWN
Output dim: 8, lower bound: -0.0002160, upper bound: 0.0001475
time: 0.49 seconds

### Relational analysis RSZ of RS_RSZ1_RSZ2_RSZ2_RSZ1_RSZ2_RSZ2_RSZ2

#### Relational analysis RSZ result of RS_RSZ1_RSZ2_RSZ2_RSZ1_RSZ2_RSZ2_RSZ2
Status: Status.VERIFIED
Output dim: 8, lower bound: -0.0002119, upper bound: 0.0001476
time: 0.51 seconds

## BFS RS instance: RS_RSZ1_RSZ2_RSZ2_RSZ2_RSZ2_RSZ1

### Backsubstitution after applying RS history:
0: 0.0035972, 0.0041150, 0.0035972, 0.0041150, -0.0001315, 0.0001306
1: 0.0018420, 0.0019168, 0.0018420, 0.0019168, -0.0000190, 0.0000189
2: 0.0120848, 0.0123711, 0.0120848, 0.0123711, -0.0000722, 0.0000727
3: -0.0021818, -0.0018857, -0.0021818, -0.0018857, -0.0000747, 0.0000752
4: -0.0019956, -0.0016750, -0.0019956, -0.0016750, -0.0000814, 0.0000808
5: 0.0056985, 0.0060018, 0.0056985, 0.0060018, -0.0000765, 0.0000770
6: 0.0003095, 0.0015131, 0.0003095, 0.0015131, -0.0003035, 0.0003057
7: -0.0046174, -0.0029782, -0.0046174, -0.0029782, -0.0004163, 0.0004134
8: 0.9859612, 0.9871160, 0.9859612, 0.9871160, -0.0002932, 0.0002912
9: -0.0041920, -0.0031438, -0.0041920, -0.0031438, -0.0002643, 0.0002662

### Unstable ReLU Count (Linear/Conv2D Layers)
- layer_idx=0, type=LayerType.Linear, total=256, inp1_unstable=4, inp2_unstable=4, delta_unstable=252
- layer_idx=2, type=LayerType.Linear, total=256, inp1_unstable=4, inp2_unstable=4, delta_unstable=255
- layer_idx=4, type=LayerType.Linear, total=256, inp1_unstable=0, inp2_unstable=0, delta_unstable=256
- layer_idx=6, type=LayerType.Linear, total=256, inp1_unstable=0, inp2_unstable=0, delta_unstable=256
- layer_idx=8, type=LayerType.Linear, total=10, inp1_unstable=0, inp2_unstable=0, delta_unstable=10

Time for backsubstitution: 1.44 seconds

### RS candidates at layer 3
type: RSZ, layer: 3, pos: 34
type: RSZ, layer: 3, pos: 83
type: RSZ, layer: 3, pos: 49

Time for candidate selection: 0.00 seconds

### Candidate
type: RSZ, layer: 3, pos: 34

### Relational analysis RSZ of RS_RSZ1_RSZ2_RSZ2_RSZ2_RSZ2_RSZ1_RSZ1

#### Relational analysis RSZ result of RS_RSZ1_RSZ2_RSZ2_RSZ2_RSZ2_RSZ1_RSZ1
Status: Status.UNKNOWN
Output dim: 8, lower bound: -0.0002221, upper bound: 0.0001457
time: 0.49 seconds

### Relational analysis RSZ of RS_RSZ1_RSZ2_RSZ2_RSZ2_RSZ2_RSZ1_RSZ2

#### Relational analysis RSZ result of RS_RSZ1_RSZ2_RSZ2_RSZ2_RSZ2_RSZ1_RSZ2
Status: Status.UNKNOWN
Output dim: 8, lower bound: -0.0002200, upper bound: 0.0001474
time: 0.52 seconds

## BFS RS instance: RS_RSZ1_RSZ2_RSZ2_RSZ2_RSZ2_RSZ2

### Backsubstitution after applying RS history:
0: 0.0035972, 0.0041150, 0.0035972, 0.0041150, -0.0001315, 0.0001300
1: 0.0018420, 0.0019168, 0.0018420, 0.0019168, -0.0000190, 0.0000188
2: 0.0120848, 0.0123711, 0.0120848, 0.0123711, -0.0000719, 0.0000727
3: -0.0021818, -0.0018857, -0.0021818, -0.0018857, -0.0000744, 0.0000752
4: -0.0019956, -0.0016750, -0.0019956, -0.0016750, -0.0000814, 0.0000805
5: 0.0056985, 0.0060018, 0.0056985, 0.0060018, -0.0000762, 0.0000770
6: 0.0003095, 0.0015131, 0.0003095, 0.0015131, -0.0003022, 0.0003057
7: -0.0046174, -0.0029782, -0.0046174, -0.0029782, -0.0004163, 0.0004116
8: 0.9859612, 0.9871160, 0.9859612, 0.9871160, -0.0002933, 0.0002900
9: -0.0041920, -0.0031438, -0.0041920, -0.0031438, -0.0002632, 0.0002662

### Unstable ReLU Count (Linear/Conv2D Layers)
- layer_idx=0, type=LayerType.Linear, total=256, inp1_unstable=4, inp2_unstable=4, delta_unstable=252
- layer_idx=2, type=LayerType.Linear, total=256, inp1_unstable=4, inp2_unstable=4, delta_unstable=255
- layer_idx=4, type=LayerType.Linear, total=256, inp1_unstable=0, inp2_unstable=0, delta_unstable=256
- layer_idx=6, type=LayerType.Linear, total=256, inp1_unstable=0, inp2_unstable=0, delta_unstable=256
- layer_idx=8, type=LayerType.Linear, total=10, inp1_unstable=0, inp2_unstable=0, delta_unstable=10

Time for backsubstitution: 1.38 seconds

### RS candidates at layer 3
type: RSZ, layer: 3, pos: 49
type: RSZ, layer: 3, pos: 34
type: RSZ, layer: 3, pos: 83

Time for candidate selection: 0.00 seconds

### Candidate
type: RSZ, layer: 3, pos: 49

### Candidate
type: RSZ, layer: 3, pos: 34

### Relational analysis RSZ of RS_RSZ1_RSZ2_RSZ2_RSZ2_RSZ2_RSZ2_RSZ1

#### Relational analysis RSZ result of RS_RSZ1_RSZ2_RSZ2_RSZ2_RSZ2_RSZ2_RSZ1
Status: Status.UNKNOWN
Output dim: 8, lower bound: -0.0002222, upper bound: 0.0001475
time: 0.51 seconds

### Relational analysis RSZ of RS_RSZ1_RSZ2_RSZ2_RSZ2_RSZ2_RSZ2_RSZ2

#### Relational analysis RSZ result of RS_RSZ1_RSZ2_RSZ2_RSZ2_RSZ2_RSZ2_RSZ2
Status: Status.UNKNOWN
Output dim: 8, lower bound: -0.0002198, upper bound: 0.0001477
time: 0.49 seconds

## BFS RS instance: RS_RSZ2_RSZ1_RSZ1_RSZ1_RSZ1_RSZ1

### Backsubstitution after applying RS history:
0: 0.0035972, 0.0041150, 0.0035972, 0.0041150, -0.0001300, 0.0001315
1: 0.0018420, 0.0019168, 0.0018420, 0.0019168, -0.0000188, 0.0000190
2: 0.0120848, 0.0123711, 0.0120848, 0.0123711, -0.0000727, 0.0000719
3: -0.0021818, -0.0018857, -0.0021818, -0.0018857, -0.0000752, 0.0000744
4: -0.0019956, -0.0016750, -0.0019956, -0.0016750, -0.0000805, 0.0000814
5: 0.0056985, 0.0060018, 0.0056985, 0.0060018, -0.0000770, 0.0000762
6: 0.0003095, 0.0015131, 0.0003095, 0.0015131, -0.0003057, 0.0003022
7: -0.0046174, -0.0029782, -0.0046174, -0.0029782, -0.0004116, 0.0004163
8: 0.9859612, 0.9871160, 0.9859612, 0.9871160, -0.0002900, 0.0002933
9: -0.0041920, -0.0031438, -0.0041920, -0.0031438, -0.0002662, 0.0002632

### Unstable ReLU Count (Linear/Conv2D Layers)
- layer_idx=0, type=LayerType.Linear, total=256, inp1_unstable=4, inp2_unstable=4, delta_unstable=252
- layer_idx=2, type=LayerType.Linear, total=256, inp1_unstable=4, inp2_unstable=4, delta_unstable=255
- layer_idx=4, type=LayerType.Linear, total=256, inp1_unstable=0, inp2_unstable=0, delta_unstable=256
- layer_idx=6, type=LayerType.Linear, total=256, inp1_unstable=0, inp2_unstable=0, delta_unstable=256
- layer_idx=8, type=LayerType.Linear, total=10, inp1_unstable=0, inp2_unstable=0, delta_unstable=10

Time for backsubstitution: 1.51 seconds

### RS candidates at layer 3
type: RSZ, layer: 3, pos: 83
type: RSZ, layer: 3, pos: 34
type: RSZ, layer: 3, pos: 49

Time for candidate selection: 0.00 seconds

### Candidate
type: RSZ, layer: 3, pos: 83

### Candidate
type: RSZ, layer: 3, pos: 34

### Relational analysis RSZ of RS_RSZ2_RSZ1_RSZ1_RSZ1_RSZ1_RSZ1_RSZ1

#### Relational analysis RSZ result of RS_RSZ2_RSZ1_RSZ1_RSZ1_RSZ1_RSZ1_RSZ1
Status: Status.UNKNOWN
Output dim: 8, lower bound: -0.0001749, upper bound: 0.0002232
time: 0.51 seconds

### Relational analysis RSZ of RS_RSZ2_RSZ1_RSZ1_RSZ1_RSZ1_RSZ1_RSZ2

#### Relational analysis RSZ result of RS_RSZ2_RSZ1_RSZ1_RSZ1_RSZ1_RSZ1_RSZ2
Status: Status.UNKNOWN
Output dim: 8, lower bound: -0.0001750, upper bound: 0.0002255
time: 0.49 seconds

## BFS RS instance: RS_RSZ2_RSZ1_RSZ1_RSZ1_RSZ1_RSZ2

### Backsubstitution after applying RS history:
0: 0.0035972, 0.0041150, 0.0035972, 0.0041150, -0.0001306, 0.0001315
1: 0.0018420, 0.0019168, 0.0018420, 0.0019168, -0.0000189, 0.0000190
2: 0.0120848, 0.0123711, 0.0120848, 0.0123711, -0.0000727, 0.0000722
3: -0.0021818, -0.0018857, -0.0021818, -0.0018857, -0.0000752, 0.0000747
4: -0.0019956, -0.0016750, -0.0019956, -0.0016750, -0.0000808, 0.0000814
5: 0.0056985, 0.0060018, 0.0056985, 0.0060018, -0.0000770, 0.0000765
6: 0.0003095, 0.0015131, 0.0003095, 0.0015131, -0.0003057, 0.0003035
7: -0.0046174, -0.0029782, -0.0046174, -0.0029782, -0.0004134, 0.0004163
8: 0.9859612, 0.9871160, 0.9859612, 0.9871160, -0.0002912, 0.0002932
9: -0.0041920, -0.0031438, -0.0041920, -0.0031438, -0.0002662, 0.0002643

### Unstable ReLU Count (Linear/Conv2D Layers)
- layer_idx=0, type=LayerType.Linear, total=256, inp1_unstable=4, inp2_unstable=4, delta_unstable=252
- layer_idx=2, type=LayerType.Linear, total=256, inp1_unstable=4, inp2_unstable=4, delta_unstable=255
- layer_idx=4, type=LayerType.Linear, total=256, inp1_unstable=0, inp2_unstable=0, delta_unstable=256
- layer_idx=6, type=LayerType.Linear, total=256, inp1_unstable=0, inp2_unstable=0, delta_unstable=256
- layer_idx=8, type=LayerType.Linear, total=10, inp1_unstable=0, inp2_unstable=0, delta_unstable=10

Time for backsubstitution: 1.40 seconds

### RS candidates at layer 3
type: RSZ, layer: 3, pos: 83
type: RSZ, layer: 3, pos: 49
type: RSZ, layer: 3, pos: 34

Time for candidate selection: 0.00 seconds

### Candidate
type: RSZ, layer: 3, pos: 83

### Candidate
type: RSZ, layer: 3, pos: 49

### Relational analysis RSZ of RS_RSZ2_RSZ1_RSZ1_RSZ1_RSZ1_RSZ2_RSZ1

#### Relational analysis RSZ result of RS_RSZ2_RSZ1_RSZ1_RSZ1_RSZ1_RSZ2_RSZ1
Status: Status.VERIFIED
Output dim: 8, lower bound: -0.0001342, upper bound: 0.0002146
time: 0.53 seconds

### Relational analysis RSZ of RS_RSZ2_RSZ1_RSZ1_RSZ1_RSZ1_RSZ2_RSZ2

#### Relational analysis RSZ result of RS_RSZ2_RSZ1_RSZ1_RSZ1_RSZ1_RSZ2_RSZ2
Status: Status.VERIFIED
Output dim: 8, lower bound: -0.0001576, upper bound: 0.0001578
time: 0.53 seconds

## BFS RS instance: RS_RSZ2_RSZ1_RSZ1_RSZ2_RSZ1_RSZ1

### Backsubstitution after applying RS history:
0: 0.0035972, 0.0041150, 0.0035972, 0.0041150, -0.0001186, 0.0001176
1: 0.0018420, 0.0019168, 0.0018420, 0.0019168, -0.0000171, 0.0000170
2: 0.0120848, 0.0123711, 0.0120848, 0.0123711, -0.0000650, 0.0000656
3: -0.0021818, -0.0018857, -0.0021818, -0.0018857, -0.0000673, 0.0000678
4: -0.0019956, -0.0016750, -0.0019956, -0.0016750, -0.0000734, 0.0000728
5: 0.0056985, 0.0060018, 0.0056985, 0.0060018, -0.0000689, 0.0000695
6: 0.0003095, 0.0015131, 0.0003095, 0.0015131, -0.0002734, 0.0002758
7: -0.0046174, -0.0029782, -0.0046174, -0.0029782, -0.0003756, 0.0003724
8: 0.9859612, 0.9871160, 0.9859612, 0.9871160, -0.0002646, 0.0002623
9: -0.0041920, -0.0031438, -0.0041920, -0.0031438, -0.0002381, 0.0002402

### Unstable ReLU Count (Linear/Conv2D Layers)
- layer_idx=0, type=LayerType.Linear, total=256, inp1_unstable=4, inp2_unstable=4, delta_unstable=252
- layer_idx=2, type=LayerType.Linear, total=256, inp1_unstable=4, inp2_unstable=4, delta_unstable=255
- layer_idx=4, type=LayerType.Linear, total=256, inp1_unstable=0, inp2_unstable=0, delta_unstable=256
- layer_idx=6, type=LayerType.Linear, total=256, inp1_unstable=0, inp2_unstable=0, delta_unstable=256
- layer_idx=8, type=LayerType.Linear, total=10, inp1_unstable=0, inp2_unstable=0, delta_unstable=10

Time for backsubstitution: 1.45 seconds

### RS candidates at layer 3
type: RSZ, layer: 3, pos: 240
type: RSZ, layer: 3, pos: 83
type: RSZ, layer: 3, pos: 49

Time for candidate selection: 0.00 seconds

### Candidate
type: RSZ, layer: 3, pos: 240

### Relational analysis RSZ of RS_RSZ2_RSZ1_RSZ1_RSZ2_RSZ1_RSZ1_RSZ1

#### Relational analysis RSZ result of RS_RSZ2_RSZ1_RSZ1_RSZ2_RSZ1_RSZ1_RSZ1
Status: Status.UNKNOWN
Output dim: 8, lower bound: -0.0001846, upper bound: 0.0002177
time: 0.52 seconds

### Relational analysis RSZ of RS_RSZ2_RSZ1_RSZ1_RSZ2_RSZ1_RSZ1_RSZ2

#### Relational analysis RSZ result of RS_RSZ2_RSZ1_RSZ1_RSZ2_RSZ1_RSZ1_RSZ2
Status: Status.UNKNOWN
Output dim: 8, lower bound: -0.0001845, upper bound: 0.0002182
time: 0.50 seconds

## BFS RS instance: RS_RSZ2_RSZ1_RSZ1_RSZ2_RSZ1_RSZ2

### Backsubstitution after applying RS history:
0: 0.0035972, 0.0041150, 0.0035972, 0.0041150, -0.0001173, 0.0001185
1: 0.0018420, 0.0019168, 0.0018420, 0.0019168, -0.0000170, 0.0000171
2: 0.0120848, 0.0123711, 0.0120848, 0.0123711, -0.0000655, 0.0000649
3: -0.0021818, -0.0018857, -0.0021818, -0.0018857, -0.0000677, 0.0000671
4: -0.0019956, -0.0016750, -0.0019956, -0.0016750, -0.0000726, 0.0000733
5: 0.0056985, 0.0060018, 0.0056985, 0.0060018, -0.0000694, 0.0000687
6: 0.0003095, 0.0015131, 0.0003095, 0.0015131, -0.0002754, 0.0002727
7: -0.0046174, -0.0029782, -0.0046174, -0.0029782, -0.0003715, 0.0003750
8: 0.9859612, 0.9871160, 0.9859612, 0.9871160, -0.0002617, 0.0002642
9: -0.0041920, -0.0031438, -0.0041920, -0.0031438, -0.0002398, 0.0002375

### Unstable ReLU Count (Linear/Conv2D Layers)
- layer_idx=0, type=LayerType.Linear, total=256, inp1_unstable=4, inp2_unstable=4, delta_unstable=252
- layer_idx=2, type=LayerType.Linear, total=256, inp1_unstable=4, inp2_unstable=4, delta_unstable=255
- layer_idx=4, type=LayerType.Linear, total=256, inp1_unstable=0, inp2_unstable=0, delta_unstable=256
- layer_idx=6, type=LayerType.Linear, total=256, inp1_unstable=0, inp2_unstable=0, delta_unstable=256
- layer_idx=8, type=LayerType.Linear, total=10, inp1_unstable=0, inp2_unstable=0, delta_unstable=10

Time for backsubstitution: 1.40 seconds

### RS candidates at layer 3
type: RSZ, layer: 3, pos: 240
type: RSZ, layer: 3, pos: 49
type: RSZ, layer: 3, pos: 83

Time for candidate selection: 0.00 seconds

### Candidate
type: RSZ, layer: 3, pos: 240

### Relational analysis RSZ of RS_RSZ2_RSZ1_RSZ1_RSZ2_RSZ1_RSZ2_RSZ1

#### Relational analysis RSZ result of RS_RSZ2_RSZ1_RSZ1_RSZ2_RSZ1_RSZ2_RSZ1
Status: Status.UNKNOWN
Output dim: 8, lower bound: -0.0001844, upper bound: 0.0002202
time: 0.49 seconds

### Relational analysis RSZ of RS_RSZ2_RSZ1_RSZ1_RSZ2_RSZ1_RSZ2_RSZ2

#### Relational analysis RSZ result of RS_RSZ2_RSZ1_RSZ1_RSZ2_RSZ1_RSZ2_RSZ2
Status: Status.UNKNOWN
Output dim: 8, lower bound: -0.0001833, upper bound: 0.0002200
time: 0.53 seconds

## BFS RS instance: RS_RSZ2_RSZ1_RSZ1_RSZ2_RSZ2_RSZ1

### Backsubstitution after applying RS history:
0: 0.0035972, 0.0041150, 0.0035972, 0.0041150, -0.0001297, 0.0001299
1: 0.0018420, 0.0019168, 0.0018420, 0.0019168, -0.0000187, 0.0000188
2: 0.0120848, 0.0123711, 0.0120848, 0.0123711, -0.0000718, 0.0000717
3: -0.0021818, -0.0018857, -0.0021818, -0.0018857, -0.0000743, 0.0000742
4: -0.0019956, -0.0016750, -0.0019956, -0.0016750, -0.0000803, 0.0000804
5: 0.0056985, 0.0060018, 0.0056985, 0.0060018, -0.0000761, 0.0000760
6: 0.0003095, 0.0015131, 0.0003095, 0.0015131, -0.0003019, 0.0003015
7: -0.0046174, -0.0029782, -0.0046174, -0.0029782, -0.0004106, 0.0004112
8: 0.9859612, 0.9871160, 0.9859612, 0.9871160, -0.0002893, 0.0002896
9: -0.0041920, -0.0031438, -0.0041920, -0.0031438, -0.0002629, 0.0002626

### Unstable ReLU Count (Linear/Conv2D Layers)
- layer_idx=0, type=LayerType.Linear, total=256, inp1_unstable=4, inp2_unstable=4, delta_unstable=252
- layer_idx=2, type=LayerType.Linear, total=256, inp1_unstable=4, inp2_unstable=4, delta_unstable=255
- layer_idx=4, type=LayerType.Linear, total=256, inp1_unstable=0, inp2_unstable=0, delta_unstable=256
- layer_idx=6, type=LayerType.Linear, total=256, inp1_unstable=0, inp2_unstable=0, delta_unstable=256
- layer_idx=8, type=LayerType.Linear, total=10, inp1_unstable=0, inp2_unstable=0, delta_unstable=10

Time for backsubstitution: 1.42 seconds

### RS candidates at layer 3
type: RSZ, layer: 3, pos: 49
type: RSZ, layer: 3, pos: 34
type: RSZ, layer: 3, pos: 83

Time for candidate selection: 0.00 seconds

### Candidate
type: RSZ, layer: 3, pos: 49

### Relational analysis RSZ of RS_RSZ2_RSZ1_RSZ1_RSZ2_RSZ2_RSZ1_RSZ1

#### Relational analysis RSZ result of RS_RSZ2_RSZ1_RSZ1_RSZ2_RSZ2_RSZ1_RSZ1
Status: Status.VERIFIED
Output dim: 8, lower bound: -0.0001421, upper bound: 0.0001655
time: 0.52 seconds

### Relational analysis RSZ of RS_RSZ2_RSZ1_RSZ1_RSZ2_RSZ2_RSZ1_RSZ2

#### Relational analysis RSZ result of RS_RSZ2_RSZ1_RSZ1_RSZ2_RSZ2_RSZ1_RSZ2
Status: Status.VERIFIED
Output dim: 8, lower bound: -0.0002086, upper bound: 0.0001407
time: 0.50 seconds

## BFS RS instance: RS_RSZ2_RSZ1_RSZ1_RSZ2_RSZ2_RSZ2

### Backsubstitution after applying RS history:
0: 0.0035972, 0.0041150, 0.0035972, 0.0041150, -0.0001301, 0.0001299
1: 0.0018420, 0.0019168, 0.0018420, 0.0019168, -0.0000188, 0.0000188
2: 0.0120848, 0.0123711, 0.0120848, 0.0123711, -0.0000718, 0.0000719
3: -0.0021818, -0.0018857, -0.0021818, -0.0018857, -0.0000743, 0.0000744
4: -0.0019956, -0.0016750, -0.0019956, -0.0016750, -0.0000805, 0.0000804
5: 0.0056985, 0.0060018, 0.0056985, 0.0060018, -0.0000761, 0.0000762
6: 0.0003095, 0.0015131, 0.0003095, 0.0015131, -0.0003019, 0.0003023
7: -0.0046174, -0.0029782, -0.0046174, -0.0029782, -0.0004117, 0.0004112
8: 0.9859612, 0.9871160, 0.9859612, 0.9871160, -0.0002900, 0.0002896
9: -0.0041920, -0.0031438, -0.0041920, -0.0031438, -0.0002629, 0.0002633

### Unstable ReLU Count (Linear/Conv2D Layers)
- layer_idx=0, type=LayerType.Linear, total=256, inp1_unstable=4, inp2_unstable=4, delta_unstable=252
- layer_idx=2, type=LayerType.Linear, total=256, inp1_unstable=4, inp2_unstable=4, delta_unstable=255
- layer_idx=4, type=LayerType.Linear, total=256, inp1_unstable=0, inp2_unstable=0, delta_unstable=256
- layer_idx=6, type=LayerType.Linear, total=256, inp1_unstable=0, inp2_unstable=0, delta_unstable=256
- layer_idx=8, type=LayerType.Linear, total=10, inp1_unstable=0, inp2_unstable=0, delta_unstable=10

Time for backsubstitution: 1.42 seconds

### RS candidates at layer 3
type: RSZ, layer: 3, pos: 83
type: RSZ, layer: 3, pos: 34
type: RSZ, layer: 3, pos: 49

Time for candidate selection: 0.00 seconds

### Candidate
type: RSZ, layer: 3, pos: 83

### Candidate
type: RSZ, layer: 3, pos: 34

### Relational analysis RSZ of RS_RSZ2_RSZ1_RSZ1_RSZ2_RSZ2_RSZ2_RSZ1

#### Relational analysis RSZ result of RS_RSZ2_RSZ1_RSZ1_RSZ2_RSZ2_RSZ2_RSZ1
Status: Status.UNKNOWN
Output dim: 8, lower bound: -0.0002212, upper bound: 0.0001935
time: 0.52 seconds

### Relational analysis RSZ of RS_RSZ2_RSZ1_RSZ1_RSZ2_RSZ2_RSZ2_RSZ2

#### Relational analysis RSZ result of RS_RSZ2_RSZ1_RSZ1_RSZ2_RSZ2_RSZ2_RSZ2
Status: Status.UNKNOWN
Output dim: 8, lower bound: -0.0002169, upper bound: 0.0001934
time: 0.51 seconds

## BFS RS instance: RS_RSZ2_RSZ1_RSZ2_RSZ1_RSZ1_RSZ1

### Backsubstitution after applying RS history:
0: 0.0035972, 0.0041150, 0.0035972, 0.0041150, -0.0001192, 0.0001152
1: 0.0018420, 0.0019168, 0.0018420, 0.0019168, -0.0000172, 0.0000166
2: 0.0120848, 0.0123711, 0.0120848, 0.0123711, -0.0000637, 0.0000659
3: -0.0021818, -0.0018857, -0.0021818, -0.0018857, -0.0000659, 0.0000681
4: -0.0019956, -0.0016750, -0.0019956, -0.0016750, -0.0000738, 0.0000713
5: 0.0056985, 0.0060018, 0.0056985, 0.0060018, -0.0000675, 0.0000698
6: 0.0003095, 0.0015131, 0.0003095, 0.0015131, -0.0002677, 0.0002770
7: -0.0046174, -0.0029782, -0.0046174, -0.0029782, -0.0003773, 0.0003646
8: 0.9859612, 0.9871160, 0.9859612, 0.9871160, -0.0002657, 0.0002568
9: -0.0041920, -0.0031438, -0.0041920, -0.0031438, -0.0002331, 0.0002412

### Unstable ReLU Count (Linear/Conv2D Layers)
- layer_idx=0, type=LayerType.Linear, total=256, inp1_unstable=4, inp2_unstable=4, delta_unstable=252
- layer_idx=2, type=LayerType.Linear, total=256, inp1_unstable=4, inp2_unstable=4, delta_unstable=255
- layer_idx=4, type=LayerType.Linear, total=256, inp1_unstable=0, inp2_unstable=0, delta_unstable=256
- layer_idx=6, type=LayerType.Linear, total=256, inp1_unstable=0, inp2_unstable=0, delta_unstable=256
- layer_idx=8, type=LayerType.Linear, total=10, inp1_unstable=0, inp2_unstable=0, delta_unstable=10

Time for backsubstitution: 1.46 seconds

### RS candidates at layer 3
type: RSZ, layer: 3, pos: 49
type: RSZ, layer: 3, pos: 83
type: RSZ, layer: 3, pos: 240

Time for candidate selection: 0.00 seconds

### Candidate
type: RSZ, layer: 3, pos: 49

### Relational analysis RSZ of RS_RSZ2_RSZ1_RSZ2_RSZ1_RSZ1_RSZ1_RSZ1

#### Relational analysis RSZ result of RS_RSZ2_RSZ1_RSZ2_RSZ1_RSZ1_RSZ1_RSZ1
Status: Status.VERIFIED
Output dim: 8, lower bound: -0.0001308, upper bound: 0.0001940
time: 0.53 seconds

### Relational analysis RSZ of RS_RSZ2_RSZ1_RSZ2_RSZ1_RSZ1_RSZ1_RSZ2

#### Relational analysis RSZ result of RS_RSZ2_RSZ1_RSZ2_RSZ1_RSZ1_RSZ1_RSZ2
Status: Status.VERIFIED
Output dim: 8, lower bound: -0.0001623, upper bound: 0.0001411
time: 0.55 seconds

## BFS RS instance: RS_RSZ2_RSZ1_RSZ2_RSZ1_RSZ1_RSZ2

### Backsubstitution after applying RS history:
0: 0.0035972, 0.0041150, 0.0035972, 0.0041150, -0.0001180, 0.0001159
1: 0.0018420, 0.0019168, 0.0018420, 0.0019168, -0.0000170, 0.0000167
2: 0.0120848, 0.0123711, 0.0120848, 0.0123711, -0.0000641, 0.0000652
3: -0.0021818, -0.0018857, -0.0021818, -0.0018857, -0.0000663, 0.0000675
4: -0.0019956, -0.0016750, -0.0019956, -0.0016750, -0.0000730, 0.0000717
5: 0.0056985, 0.0060018, 0.0056985, 0.0060018, -0.0000679, 0.0000691
6: 0.0003095, 0.0015131, 0.0003095, 0.0015131, -0.0002694, 0.0002743
7: -0.0046174, -0.0029782, -0.0046174, -0.0029782, -0.0003735, 0.0003669
8: 0.9859612, 0.9871160, 0.9859612, 0.9871160, -0.0002631, 0.0002584
9: -0.0041920, -0.0031438, -0.0041920, -0.0031438, -0.0002346, 0.0002388

### Unstable ReLU Count (Linear/Conv2D Layers)
- layer_idx=0, type=LayerType.Linear, total=256, inp1_unstable=4, inp2_unstable=4, delta_unstable=252
- layer_idx=2, type=LayerType.Linear, total=256, inp1_unstable=4, inp2_unstable=4, delta_unstable=255
- layer_idx=4, type=LayerType.Linear, total=256, inp1_unstable=0, inp2_unstable=0, delta_unstable=256
- layer_idx=6, type=LayerType.Linear, total=256, inp1_unstable=0, inp2_unstable=0, delta_unstable=256
- layer_idx=8, type=LayerType.Linear, total=10, inp1_unstable=0, inp2_unstable=0, delta_unstable=10

Time for backsubstitution: 1.43 seconds

### RS candidates at layer 3
type: RSZ, layer: 3, pos: 83
type: RSZ, layer: 3, pos: 240
type: RSZ, layer: 3, pos: 49

Time for candidate selection: 0.00 seconds

### Candidate
type: RSZ, layer: 3, pos: 83

### Candidate
type: RSZ, layer: 3, pos: 240

### Relational analysis RSZ of RS_RSZ2_RSZ1_RSZ2_RSZ1_RSZ1_RSZ2_RSZ1

#### Relational analysis RSZ result of RS_RSZ2_RSZ1_RSZ2_RSZ1_RSZ1_RSZ2_RSZ1
Status: Status.UNKNOWN
Output dim: 8, lower bound: -0.0001935, upper bound: 0.0002211
time: 0.54 seconds

### Relational analysis RSZ of RS_RSZ2_RSZ1_RSZ2_RSZ1_RSZ1_RSZ2_RSZ2

#### Relational analysis RSZ result of RS_RSZ2_RSZ1_RSZ2_RSZ1_RSZ1_RSZ2_RSZ2
Status: Status.UNKNOWN
Output dim: 8, lower bound: -0.0001850, upper bound: 0.0002200
time: 0.49 seconds

## BFS RS instance: RS_RSZ2_RSZ1_RSZ2_RSZ1_RSZ2_RSZ1

### Backsubstitution after applying RS history:
0: 0.0035972, 0.0041150, 0.0035972, 0.0041150, -0.0001192, 0.0001152
1: 0.0018420, 0.0019168, 0.0018420, 0.0019168, -0.0000172, 0.0000166
2: 0.0120848, 0.0123711, 0.0120848, 0.0123711, -0.0000637, 0.0000659
3: -0.0021818, -0.0018857, -0.0021818, -0.0018857, -0.0000659, 0.0000681
4: -0.0019956, -0.0016750, -0.0019956, -0.0016750, -0.0000738, 0.0000713
5: 0.0056985, 0.0060018, 0.0056985, 0.0060018, -0.0000675, 0.0000698
6: 0.0003095, 0.0015131, 0.0003095, 0.0015131, -0.0002677, 0.0002770
7: -0.0046174, -0.0029782, -0.0046174, -0.0029782, -0.0003773, 0.0003646
8: 0.9859612, 0.9871160, 0.9859612, 0.9871160, -0.0002657, 0.0002568
9: -0.0041920, -0.0031438, -0.0041920, -0.0031438, -0.0002331, 0.0002412

### Unstable ReLU Count (Linear/Conv2D Layers)
- layer_idx=0, type=LayerType.Linear, total=256, inp1_unstable=4, inp2_unstable=4, delta_unstable=252
- layer_idx=2, type=LayerType.Linear, total=256, inp1_unstable=4, inp2_unstable=4, delta_unstable=255
- layer_idx=4, type=LayerType.Linear, total=256, inp1_unstable=0, inp2_unstable=0, delta_unstable=256
- layer_idx=6, type=LayerType.Linear, total=256, inp1_unstable=0, inp2_unstable=0, delta_unstable=256
- layer_idx=8, type=LayerType.Linear, total=10, inp1_unstable=0, inp2_unstable=0, delta_unstable=10

Time for backsubstitution: 1.42 seconds

### RS candidates at layer 3
type: RSZ, layer: 3, pos: 49
type: RSZ, layer: 3, pos: 240
type: RSZ, layer: 3, pos: 83

Time for candidate selection: 0.00 seconds

### Candidate
type: RSZ, layer: 3, pos: 49

### Relational analysis RSZ of RS_RSZ2_RSZ1_RSZ2_RSZ1_RSZ2_RSZ1_RSZ1

#### Relational analysis RSZ result of RS_RSZ2_RSZ1_RSZ2_RSZ1_RSZ2_RSZ1_RSZ1
Status: Status.VERIFIED
Output dim: 8, lower bound: -0.0001325, upper bound: 0.0001531
time: 0.54 seconds

### Relational analysis RSZ of RS_RSZ2_RSZ1_RSZ2_RSZ1_RSZ2_RSZ1_RSZ2

#### Relational analysis RSZ result of RS_RSZ2_RSZ1_RSZ2_RSZ1_RSZ2_RSZ1_RSZ2
Status: Status.VERIFIED
Output dim: 8, lower bound: -0.0001954, upper bound: 0.0001305
time: 0.53 seconds

## BFS RS instance: RS_RSZ2_RSZ1_RSZ2_RSZ1_RSZ2_RSZ2

### Backsubstitution after applying RS history:
0: 0.0035972, 0.0041150, 0.0035972, 0.0041150, -0.0001180, 0.0001159
1: 0.0018420, 0.0019168, 0.0018420, 0.0019168, -0.0000170, 0.0000167
2: 0.0120848, 0.0123711, 0.0120848, 0.0123711, -0.0000641, 0.0000652
3: -0.0021818, -0.0018857, -0.0021818, -0.0018857, -0.0000663, 0.0000675
4: -0.0019956, -0.0016750, -0.0019956, -0.0016750, -0.0000730, 0.0000717
5: 0.0056985, 0.0060018, 0.0056985, 0.0060018, -0.0000679, 0.0000691
6: 0.0003095, 0.0015131, 0.0003095, 0.0015131, -0.0002694, 0.0002743
7: -0.0046174, -0.0029782, -0.0046174, -0.0029782, -0.0003735, 0.0003669
8: 0.9859612, 0.9871160, 0.9859612, 0.9871160, -0.0002631, 0.0002584
9: -0.0041920, -0.0031438, -0.0041920, -0.0031438, -0.0002346, 0.0002388

### Unstable ReLU Count (Linear/Conv2D Layers)
- layer_idx=0, type=LayerType.Linear, total=256, inp1_unstable=4, inp2_unstable=4, delta_unstable=252
- layer_idx=2, type=LayerType.Linear, total=256, inp1_unstable=4, inp2_unstable=4, delta_unstable=255
- layer_idx=4, type=LayerType.Linear, total=256, inp1_unstable=0, inp2_unstable=0, delta_unstable=256
- layer_idx=6, type=LayerType.Linear, total=256, inp1_unstable=0, inp2_unstable=0, delta_unstable=256
- layer_idx=8, type=LayerType.Linear, total=10, inp1_unstable=0, inp2_unstable=0, delta_unstable=10

Time for backsubstitution: 1.41 seconds

### RS candidates at layer 3
type: RSZ, layer: 3, pos: 240
type: RSZ, layer: 3, pos: 49
type: RSZ, layer: 3, pos: 83

Time for candidate selection: 0.00 seconds

### Candidate
type: RSZ, layer: 3, pos: 240

### Relational analysis RSZ of RS_RSZ2_RSZ1_RSZ2_RSZ1_RSZ2_RSZ2_RSZ1

#### Relational analysis RSZ result of RS_RSZ2_RSZ1_RSZ2_RSZ1_RSZ2_RSZ2_RSZ1
Status: Status.UNKNOWN
Output dim: 8, lower bound: -0.0002176, upper bound: 0.0001799
time: 0.52 seconds

### Relational analysis RSZ of RS_RSZ2_RSZ1_RSZ2_RSZ1_RSZ2_RSZ2_RSZ2

#### Relational analysis RSZ result of RS_RSZ2_RSZ1_RSZ2_RSZ1_RSZ2_RSZ2_RSZ2
Status: Status.UNKNOWN
Output dim: 8, lower bound: -0.0002160, upper bound: 0.0001820
time: 0.51 seconds

## BFS RS instance: RS_RSZ2_RSZ1_RSZ2_RSZ2_RSZ1_RSZ2

### Backsubstitution after applying RS history:
0: 0.0035972, 0.0041150, 0.0035972, 0.0041150, -0.0001334, 0.0001232
1: 0.0018420, 0.0019168, 0.0018420, 0.0019168, -0.0000193, 0.0000178
2: 0.0120848, 0.0123711, 0.0120848, 0.0123711, -0.0000681, 0.0000738
3: -0.0021818, -0.0018857, -0.0021818, -0.0018857, -0.0000704, 0.0000763
4: -0.0019956, -0.0016750, -0.0019956, -0.0016750, -0.0000826, 0.0000762
5: 0.0056985, 0.0060018, 0.0056985, 0.0060018, -0.0000722, 0.0000782
6: 0.0003095, 0.0015131, 0.0003095, 0.0015131, -0.0002863, 0.0003101
7: -0.0046174, -0.0029782, -0.0046174, -0.0029782, -0.0004223, 0.0003899
8: 0.9859612, 0.9871160, 0.9859612, 0.9871160, -0.0002975, 0.0002747
9: -0.0041920, -0.0031438, -0.0041920, -0.0031438, -0.0002493, 0.0002700

### Unstable ReLU Count (Linear/Conv2D Layers)
- layer_idx=0, type=LayerType.Linear, total=256, inp1_unstable=4, inp2_unstable=4, delta_unstable=252
- layer_idx=2, type=LayerType.Linear, total=256, inp1_unstable=4, inp2_unstable=4, delta_unstable=255
- layer_idx=4, type=LayerType.Linear, total=256, inp1_unstable=0, inp2_unstable=0, delta_unstable=256
- layer_idx=6, type=LayerType.Linear, total=256, inp1_unstable=0, inp2_unstable=0, delta_unstable=256
- layer_idx=8, type=LayerType.Linear, total=10, inp1_unstable=0, inp2_unstable=0, delta_unstable=10

Time for backsubstitution: 1.60 seconds

### RS candidates at layer 3
type: RSZ, layer: 3, pos: 49
type: RSZ, layer: 3, pos: 34
type: RSZ, layer: 3, pos: 240

Time for candidate selection: 0.00 seconds

### Candidate
type: RSZ, layer: 3, pos: 49

### Relational analysis RSZ of RS_RSZ2_RSZ1_RSZ2_RSZ2_RSZ1_RSZ2_RSZ1

#### Relational analysis RSZ result of RS_RSZ2_RSZ1_RSZ2_RSZ2_RSZ1_RSZ2_RSZ1
Status: Status.VERIFIED
Output dim: 8, lower bound: -0.0001323, upper bound: 0.0001414
time: 0.54 seconds

### Relational analysis RSZ of RS_RSZ2_RSZ1_RSZ2_RSZ2_RSZ1_RSZ2_RSZ2

#### Relational analysis RSZ result of RS_RSZ2_RSZ1_RSZ2_RSZ2_RSZ1_RSZ2_RSZ2
Status: Status.VERIFIED
Output dim: 8, lower bound: -0.0002028, upper bound: 0.0001271
time: 0.50 seconds

## BFS RS instance: RS_RSZ2_RSZ1_RSZ2_RSZ2_RSZ2_RSZ1

### Backsubstitution after applying RS history:
0: 0.0035972, 0.0041150, 0.0035972, 0.0041150, -0.0001328, 0.0001266
1: 0.0018420, 0.0019168, 0.0018420, 0.0019168, -0.0000192, 0.0000183
2: 0.0120848, 0.0123711, 0.0120848, 0.0123711, -0.0000700, 0.0000734
3: -0.0021818, -0.0018857, -0.0021818, -0.0018857, -0.0000724, 0.0000760
4: -0.0019956, -0.0016750, -0.0019956, -0.0016750, -0.0000822, 0.0000783
5: 0.0056985, 0.0060018, 0.0056985, 0.0060018, -0.0000741, 0.0000778
6: 0.0003095, 0.0015131, 0.0003095, 0.0015131, -0.0002942, 0.0003087
7: -0.0046174, -0.0029782, -0.0046174, -0.0029782, -0.0004205, 0.0004007
8: 0.9859612, 0.9871160, 0.9859612, 0.9871160, -0.0002962, 0.0002822
9: -0.0041920, -0.0031438, -0.0041920, -0.0031438, -0.0002562, 0.0002689

### Unstable ReLU Count (Linear/Conv2D Layers)
- layer_idx=0, type=LayerType.Linear, total=256, inp1_unstable=4, inp2_unstable=4, delta_unstable=252
- layer_idx=2, type=LayerType.Linear, total=256, inp1_unstable=4, inp2_unstable=4, delta_unstable=255
- layer_idx=4, type=LayerType.Linear, total=256, inp1_unstable=0, inp2_unstable=0, delta_unstable=256
- layer_idx=6, type=LayerType.Linear, total=256, inp1_unstable=0, inp2_unstable=0, delta_unstable=256
- layer_idx=8, type=LayerType.Linear, total=10, inp1_unstable=0, inp2_unstable=0, delta_unstable=10

Time for backsubstitution: 1.45 seconds

### RS candidates at layer 3
type: RSZ, layer: 3, pos: 34
type: RSZ, layer: 3, pos: 49
type: RSZ, layer: 3, pos: 83

Time for candidate selection: 0.00 seconds

### Candidate
type: RSZ, layer: 3, pos: 34

### Candidate
type: RSZ, layer: 3, pos: 49

### Relational analysis RSZ of RS_RSZ2_RSZ1_RSZ2_RSZ2_RSZ2_RSZ1_RSZ1

#### Relational analysis RSZ result of RS_RSZ2_RSZ1_RSZ2_RSZ2_RSZ2_RSZ1_RSZ1
Status: Status.VERIFIED
Output dim: 8, lower bound: -0.0001515, upper bound: 0.0002073
time: 0.52 seconds

### Relational analysis RSZ of RS_RSZ2_RSZ1_RSZ2_RSZ2_RSZ2_RSZ1_RSZ2

#### Relational analysis RSZ result of RS_RSZ2_RSZ1_RSZ2_RSZ2_RSZ2_RSZ1_RSZ2
Status: Status.UNKNOWN
Output dim: 8, lower bound: -0.0002179, upper bound: 0.0001576
time: 0.53 seconds

## BFS RS instance: RS_RSZ2_RSZ1_RSZ2_RSZ2_RSZ2_RSZ2

### Backsubstitution after applying RS history:
0: 0.0035972, 0.0041150, 0.0035972, 0.0041150, -0.0001329, 0.0001260
1: 0.0018420, 0.0019168, 0.0018420, 0.0019168, -0.0000192, 0.0000182
2: 0.0120848, 0.0123711, 0.0120848, 0.0123711, -0.0000697, 0.0000735
3: -0.0021818, -0.0018857, -0.0021818, -0.0018857, -0.0000721, 0.0000760
4: -0.0019956, -0.0016750, -0.0019956, -0.0016750, -0.0000823, 0.0000780
5: 0.0056985, 0.0060018, 0.0056985, 0.0060018, -0.0000738, 0.0000779
6: 0.0003095, 0.0015131, 0.0003095, 0.0015131, -0.0002929, 0.0003090
7: -0.0046174, -0.0029782, -0.0046174, -0.0029782, -0.0004208, 0.0003989
8: 0.9859612, 0.9871160, 0.9859612, 0.9871160, -0.0002964, 0.0002810
9: -0.0041920, -0.0031438, -0.0041920, -0.0031438, -0.0002551, 0.0002691

### Unstable ReLU Count (Linear/Conv2D Layers)
- layer_idx=0, type=LayerType.Linear, total=256, inp1_unstable=4, inp2_unstable=4, delta_unstable=252
- layer_idx=2, type=LayerType.Linear, total=256, inp1_unstable=4, inp2_unstable=4, delta_unstable=255
- layer_idx=4, type=LayerType.Linear, total=256, inp1_unstable=0, inp2_unstable=0, delta_unstable=256
- layer_idx=6, type=LayerType.Linear, total=256, inp1_unstable=0, inp2_unstable=0, delta_unstable=256
- layer_idx=8, type=LayerType.Linear, total=10, inp1_unstable=0, inp2_unstable=0, delta_unstable=10

Time for backsubstitution: 1.46 seconds

### RS candidates at layer 3
type: RSZ, layer: 3, pos: 49
type: RSZ, layer: 3, pos: 83
type: RSZ, layer: 3, pos: 34

Time for candidate selection: 0.00 seconds

### Candidate
type: RSZ, layer: 3, pos: 49

### Relational analysis RSZ of RS_RSZ2_RSZ1_RSZ2_RSZ2_RSZ2_RSZ2_RSZ1

#### Relational analysis RSZ result of RS_RSZ2_RSZ1_RSZ2_RSZ2_RSZ2_RSZ2_RSZ1
Status: Status.VERIFIED
Output dim: 8, lower bound: -0.0001463, upper bound: 0.0002011
time: 0.51 seconds

### Relational analysis RSZ of RS_RSZ2_RSZ1_RSZ2_RSZ2_RSZ2_RSZ2_RSZ2

#### Relational analysis RSZ result of RS_RSZ2_RSZ1_RSZ2_RSZ2_RSZ2_RSZ2_RSZ2
Status: Status.UNKNOWN
Output dim: 8, lower bound: -0.0002178, upper bound: 0.0001576
time: 0.53 seconds

## BFS RS instance: RS_RSZ2_RSZ2_RSZ1_RSZ1_RSZ1_RSZ1

### Backsubstitution after applying RS history:
0: 0.0035972, 0.0041150, 0.0035972, 0.0041150, -0.0001175, 0.0001189
1: 0.0018420, 0.0019168, 0.0018420, 0.0019168, -0.0000170, 0.0000172
2: 0.0120848, 0.0123711, 0.0120848, 0.0123711, -0.0000657, 0.0000650
3: -0.0021818, -0.0018857, -0.0021818, -0.0018857, -0.0000680, 0.0000672
4: -0.0019956, -0.0016750, -0.0019956, -0.0016750, -0.0000727, 0.0000736
5: 0.0056985, 0.0060018, 0.0056985, 0.0060018, -0.0000697, 0.0000688
6: 0.0003095, 0.0015131, 0.0003095, 0.0015131, -0.0002764, 0.0002731
7: -0.0046174, -0.0029782, -0.0046174, -0.0029782, -0.0003720, 0.0003764
8: 0.9859612, 0.9871160, 0.9859612, 0.9871160, -0.0002620, 0.0002652
9: -0.0041920, -0.0031438, -0.0041920, -0.0031438, -0.0002407, 0.0002378

### Unstable ReLU Count (Linear/Conv2D Layers)
- layer_idx=0, type=LayerType.Linear, total=256, inp1_unstable=4, inp2_unstable=4, delta_unstable=252
- layer_idx=2, type=LayerType.Linear, total=256, inp1_unstable=4, inp2_unstable=4, delta_unstable=255
- layer_idx=4, type=LayerType.Linear, total=256, inp1_unstable=0, inp2_unstable=0, delta_unstable=256
- layer_idx=6, type=LayerType.Linear, total=256, inp1_unstable=0, inp2_unstable=0, delta_unstable=256
- layer_idx=8, type=LayerType.Linear, total=10, inp1_unstable=0, inp2_unstable=0, delta_unstable=10

Time for backsubstitution: 1.51 seconds

### RS candidates at layer 3
type: RSZ, layer: 3, pos: 240
type: RSZ, layer: 3, pos: 49
type: RSZ, layer: 3, pos: 83

Time for candidate selection: 0.00 seconds

### Candidate
type: RSZ, layer: 3, pos: 240

### Relational analysis RSZ of RS_RSZ2_RSZ2_RSZ1_RSZ1_RSZ1_RSZ1_RSZ1

#### Relational analysis RSZ result of RS_RSZ2_RSZ2_RSZ1_RSZ1_RSZ1_RSZ1_RSZ1
Status: Status.UNKNOWN
Output dim: 8, lower bound: -0.0001740, upper bound: 0.0002228
time: 0.57 seconds

### Relational analysis RSZ of RS_RSZ2_RSZ2_RSZ1_RSZ1_RSZ1_RSZ1_RSZ2

#### Relational analysis RSZ result of RS_RSZ2_RSZ2_RSZ1_RSZ1_RSZ1_RSZ1_RSZ2
Status: Status.UNKNOWN
Output dim: 8, lower bound: -0.0001707, upper bound: 0.0002228
time: 0.53 seconds

## BFS RS instance: RS_RSZ2_RSZ2_RSZ1_RSZ1_RSZ1_RSZ2

### Backsubstitution after applying RS history:
0: 0.0035972, 0.0041150, 0.0035972, 0.0041150, -0.0001164, 0.0001197
1: 0.0018420, 0.0019168, 0.0018420, 0.0019168, -0.0000168, 0.0000173
2: 0.0120848, 0.0123711, 0.0120848, 0.0123711, -0.0000662, 0.0000644
3: -0.0021818, -0.0018857, -0.0021818, -0.0018857, -0.0000684, 0.0000666
4: -0.0019956, -0.0016750, -0.0019956, -0.0016750, -0.0000721, 0.0000741
5: 0.0056985, 0.0060018, 0.0056985, 0.0060018, -0.0000701, 0.0000682
6: 0.0003095, 0.0015131, 0.0003095, 0.0015131, -0.0002781, 0.0002706
7: -0.0046174, -0.0029782, -0.0046174, -0.0029782, -0.0003685, 0.0003788
8: 0.9859612, 0.9871160, 0.9859612, 0.9871160, -0.0002596, 0.0002668
9: -0.0041920, -0.0031438, -0.0041920, -0.0031438, -0.0002422, 0.0002356

### Unstable ReLU Count (Linear/Conv2D Layers)
- layer_idx=0, type=LayerType.Linear, total=256, inp1_unstable=4, inp2_unstable=4, delta_unstable=252
- layer_idx=2, type=LayerType.Linear, total=256, inp1_unstable=4, inp2_unstable=4, delta_unstable=255
- layer_idx=4, type=LayerType.Linear, total=256, inp1_unstable=0, inp2_unstable=0, delta_unstable=256
- layer_idx=6, type=LayerType.Linear, total=256, inp1_unstable=0, inp2_unstable=0, delta_unstable=256
- layer_idx=8, type=LayerType.Linear, total=10, inp1_unstable=0, inp2_unstable=0, delta_unstable=10

Time for backsubstitution: 1.44 seconds

### RS candidates at layer 3
type: RSZ, layer: 3, pos: 240
type: RSZ, layer: 3, pos: 49
type: RSZ, layer: 3, pos: 83

Time for candidate selection: 0.00 seconds

### Candidate
type: RSZ, layer: 3, pos: 240

### Relational analysis RSZ of RS_RSZ2_RSZ2_RSZ1_RSZ1_RSZ1_RSZ2_RSZ1

#### Relational analysis RSZ result of RS_RSZ2_RSZ2_RSZ1_RSZ1_RSZ1_RSZ2_RSZ1
Status: Status.UNKNOWN
Output dim: 8, lower bound: -0.0001739, upper bound: 0.0002256
time: 0.53 seconds

### Relational analysis RSZ of RS_RSZ2_RSZ2_RSZ1_RSZ1_RSZ1_RSZ2_RSZ2

#### Relational analysis RSZ result of RS_RSZ2_RSZ2_RSZ1_RSZ1_RSZ1_RSZ2_RSZ2
Status: Status.UNKNOWN
Output dim: 8, lower bound: -0.0001705, upper bound: 0.0002252
time: 0.54 seconds

## BFS RS instance: RS_RSZ2_RSZ2_RSZ1_RSZ1_RSZ2_RSZ1

### Backsubstitution after applying RS history:
0: 0.0035972, 0.0041150, 0.0035972, 0.0041150, -0.0001286, 0.0001311
1: 0.0018420, 0.0019168, 0.0018420, 0.0019168, -0.0000186, 0.0000189
2: 0.0120848, 0.0123711, 0.0120848, 0.0123711, -0.0000725, 0.0000711
3: -0.0021818, -0.0018857, -0.0021818, -0.0018857, -0.0000750, 0.0000735
4: -0.0019956, -0.0016750, -0.0019956, -0.0016750, -0.0000796, 0.0000811
5: 0.0056985, 0.0060018, 0.0056985, 0.0060018, -0.0000768, 0.0000753
6: 0.0003095, 0.0015131, 0.0003095, 0.0015131, -0.0003047, 0.0002989
7: -0.0046174, -0.0029782, -0.0046174, -0.0029782, -0.0004071, 0.0004150
8: 0.9859612, 0.9871160, 0.9859612, 0.9871160, -0.0002868, 0.0002923
9: -0.0041920, -0.0031438, -0.0041920, -0.0031438, -0.0002653, 0.0002603

### Unstable ReLU Count (Linear/Conv2D Layers)
- layer_idx=0, type=LayerType.Linear, total=256, inp1_unstable=4, inp2_unstable=4, delta_unstable=252
- layer_idx=2, type=LayerType.Linear, total=256, inp1_unstable=4, inp2_unstable=4, delta_unstable=255
- layer_idx=4, type=LayerType.Linear, total=256, inp1_unstable=0, inp2_unstable=0, delta_unstable=256
- layer_idx=6, type=LayerType.Linear, total=256, inp1_unstable=0, inp2_unstable=0, delta_unstable=256
- layer_idx=8, type=LayerType.Linear, total=10, inp1_unstable=0, inp2_unstable=0, delta_unstable=10

Time for backsubstitution: 1.50 seconds

### RS candidates at layer 3
type: RSZ, layer: 3, pos: 83
type: RSZ, layer: 3, pos: 34
type: RSZ, layer: 3, pos: 49

Time for candidate selection: 0.00 seconds

### Candidate
type: RSZ, layer: 3, pos: 83

### Candidate
type: RSZ, layer: 3, pos: 34

### Relational analysis RSZ of RS_RSZ2_RSZ2_RSZ1_RSZ1_RSZ2_RSZ1_RSZ1

#### Relational analysis RSZ result of RS_RSZ2_RSZ2_RSZ1_RSZ1_RSZ2_RSZ1_RSZ1
Status: Status.VERIFIED
Output dim: 8, lower bound: -0.0002134, upper bound: 0.0001931
time: 0.53 seconds

### Relational analysis RSZ of RS_RSZ2_RSZ2_RSZ1_RSZ1_RSZ2_RSZ1_RSZ2

#### Relational analysis RSZ result of RS_RSZ2_RSZ2_RSZ1_RSZ1_RSZ2_RSZ1_RSZ2
Status: Status.VERIFIED
Output dim: 8, lower bound: -0.0002117, upper bound: 0.0001931
time: 0.52 seconds

## BFS RS instance: RS_RSZ2_RSZ2_RSZ1_RSZ1_RSZ2_RSZ2

### Backsubstitution after applying RS history:
0: 0.0035972, 0.0041150, 0.0035972, 0.0041150, -0.0001289, 0.0001309
1: 0.0018420, 0.0019168, 0.0018420, 0.0019168, -0.0000186, 0.0000189
2: 0.0120848, 0.0123711, 0.0120848, 0.0123711, -0.0000724, 0.0000713
3: -0.0021818, -0.0018857, -0.0021818, -0.0018857, -0.0000749, 0.0000737
4: -0.0019956, -0.0016750, -0.0019956, -0.0016750, -0.0000798, 0.0000810
5: 0.0056985, 0.0060018, 0.0056985, 0.0060018, -0.0000767, 0.0000755
6: 0.0003095, 0.0015131, 0.0003095, 0.0015131, -0.0003043, 0.0002997
7: -0.0046174, -0.0029782, -0.0046174, -0.0029782, -0.0004081, 0.0004144
8: 0.9859612, 0.9871160, 0.9859612, 0.9871160, -0.0002875, 0.0002919
9: -0.0041920, -0.0031438, -0.0041920, -0.0031438, -0.0002650, 0.0002610

### Unstable ReLU Count (Linear/Conv2D Layers)
- layer_idx=0, type=LayerType.Linear, total=256, inp1_unstable=4, inp2_unstable=4, delta_unstable=252
- layer_idx=2, type=LayerType.Linear, total=256, inp1_unstable=4, inp2_unstable=4, delta_unstable=255
- layer_idx=4, type=LayerType.Linear, total=256, inp1_unstable=0, inp2_unstable=0, delta_unstable=256
- layer_idx=6, type=LayerType.Linear, total=256, inp1_unstable=0, inp2_unstable=0, delta_unstable=256
- layer_idx=8, type=LayerType.Linear, total=10, inp1_unstable=0, inp2_unstable=0, delta_unstable=10

Time for backsubstitution: 1.40 seconds

### RS candidates at layer 3
type: RSZ, layer: 3, pos: 49
type: RSZ, layer: 3, pos: 83
type: RSZ, layer: 3, pos: 34

Time for candidate selection: 0.00 seconds

### Candidate
type: RSZ, layer: 3, pos: 49

### Relational analysis RSZ of RS_RSZ2_RSZ2_RSZ1_RSZ1_RSZ2_RSZ2_RSZ1

#### Relational analysis RSZ result of RS_RSZ2_RSZ2_RSZ1_RSZ1_RSZ2_RSZ2_RSZ1
Status: Status.VERIFIED
Output dim: 8, lower bound: -0.0001441, upper bound: 0.0001853
time: 0.49 seconds

### Relational analysis RSZ of RS_RSZ2_RSZ2_RSZ1_RSZ1_RSZ2_RSZ2_RSZ2

#### Relational analysis RSZ result of RS_RSZ2_RSZ2_RSZ1_RSZ1_RSZ2_RSZ2_RSZ2
Status: Status.VERIFIED
Output dim: 8, lower bound: -0.0002033, upper bound: 0.0001505
time: 0.50 seconds

## BFS RS instance: RS_RSZ2_RSZ2_RSZ1_RSZ2_RSZ1_RSZ1

### Backsubstitution after applying RS history:
0: 0.0035972, 0.0041150, 0.0035972, 0.0041150, -0.0001271, 0.0001282
1: 0.0018420, 0.0019168, 0.0018420, 0.0019168, -0.0000184, 0.0000185
2: 0.0120848, 0.0123711, 0.0120848, 0.0123711, -0.0000709, 0.0000703
3: -0.0021818, -0.0018857, -0.0021818, -0.0018857, -0.0000733, 0.0000727
4: -0.0019956, -0.0016750, -0.0019956, -0.0016750, -0.0000787, 0.0000793
5: 0.0056985, 0.0060018, 0.0056985, 0.0060018, -0.0000751, 0.0000745
6: 0.0003095, 0.0015131, 0.0003095, 0.0015131, -0.0002979, 0.0002954
7: -0.0046174, -0.0029782, -0.0046174, -0.0029782, -0.0004024, 0.0004057
8: 0.9859612, 0.9871160, 0.9859612, 0.9871160, -0.0002834, 0.0002858
9: -0.0041920, -0.0031438, -0.0041920, -0.0031438, -0.0002594, 0.0002573

### Unstable ReLU Count (Linear/Conv2D Layers)
- layer_idx=0, type=LayerType.Linear, total=256, inp1_unstable=4, inp2_unstable=4, delta_unstable=252
- layer_idx=2, type=LayerType.Linear, total=256, inp1_unstable=4, inp2_unstable=4, delta_unstable=255
- layer_idx=4, type=LayerType.Linear, total=256, inp1_unstable=0, inp2_unstable=0, delta_unstable=256
- layer_idx=6, type=LayerType.Linear, total=256, inp1_unstable=0, inp2_unstable=0, delta_unstable=256
- layer_idx=8, type=LayerType.Linear, total=10, inp1_unstable=0, inp2_unstable=0, delta_unstable=10

Time for backsubstitution: 1.44 seconds

### RS candidates at layer 3
type: RSZ, layer: 3, pos: 49
type: RSZ, layer: 3, pos: 34
type: RSZ, layer: 3, pos: 240

Time for candidate selection: 0.00 seconds

### Candidate
type: RSZ, layer: 3, pos: 49

### Relational analysis RSZ of RS_RSZ2_RSZ2_RSZ1_RSZ2_RSZ1_RSZ1_RSZ1

#### Relational analysis RSZ result of RS_RSZ2_RSZ2_RSZ1_RSZ2_RSZ1_RSZ1_RSZ1
Status: Status.VERIFIED
Output dim: 8, lower bound: -0.0001488, upper bound: 0.0002110
time: 0.49 seconds

### Relational analysis RSZ of RS_RSZ2_RSZ2_RSZ1_RSZ2_RSZ1_RSZ1_RSZ2

#### Relational analysis RSZ result of RS_RSZ2_RSZ2_RSZ1_RSZ2_RSZ1_RSZ1_RSZ2
Status: Status.VERIFIED
Output dim: 8, lower bound: -0.0001741, upper bound: 0.0001504
time: 0.50 seconds

## BFS RS instance: RS_RSZ2_RSZ2_RSZ1_RSZ2_RSZ1_RSZ2

### Backsubstitution after applying RS history:
0: 0.0035972, 0.0041150, 0.0035972, 0.0041150, -0.0001311, 0.0001241
1: 0.0018420, 0.0019168, 0.0018420, 0.0019168, -0.0000189, 0.0000179
2: 0.0120848, 0.0123711, 0.0120848, 0.0123711, -0.0000686, 0.0000725
3: -0.0021818, -0.0018857, -0.0021818, -0.0018857, -0.0000710, 0.0000749
4: -0.0019956, -0.0016750, -0.0019956, -0.0016750, -0.0000811, 0.0000768
5: 0.0056985, 0.0060018, 0.0056985, 0.0060018, -0.0000727, 0.0000768
6: 0.0003095, 0.0015131, 0.0003095, 0.0015131, -0.0002885, 0.0003046
7: -0.0046174, -0.0029782, -0.0046174, -0.0029782, -0.0004149, 0.0003929
8: 0.9859612, 0.9871160, 0.9859612, 0.9871160, -0.0002923, 0.0002768
9: -0.0041920, -0.0031438, -0.0041920, -0.0031438, -0.0002512, 0.0002653

### Unstable ReLU Count (Linear/Conv2D Layers)
- layer_idx=0, type=LayerType.Linear, total=256, inp1_unstable=4, inp2_unstable=4, delta_unstable=252
- layer_idx=2, type=LayerType.Linear, total=256, inp1_unstable=4, inp2_unstable=4, delta_unstable=255
- layer_idx=4, type=LayerType.Linear, total=256, inp1_unstable=0, inp2_unstable=0, delta_unstable=256
- layer_idx=6, type=LayerType.Linear, total=256, inp1_unstable=0, inp2_unstable=0, delta_unstable=256
- layer_idx=8, type=LayerType.Linear, total=10, inp1_unstable=0, inp2_unstable=0, delta_unstable=10

Time for backsubstitution: 1.47 seconds

### RS candidates at layer 3
type: RSZ, layer: 3, pos: 240
type: RSZ, layer: 3, pos: 34
type: RSZ, layer: 3, pos: 49

Time for candidate selection: 0.00 seconds

### Candidate
type: RSZ, layer: 3, pos: 240

### Candidate
type: RSZ, layer: 3, pos: 34

### Relational analysis RSZ of RS_RSZ2_RSZ2_RSZ1_RSZ2_RSZ1_RSZ2_RSZ1

#### Relational analysis RSZ result of RS_RSZ2_RSZ2_RSZ1_RSZ2_RSZ1_RSZ2_RSZ1
Status: Status.UNKNOWN
Output dim: 8, lower bound: -0.0002200, upper bound: 0.0001821
time: 0.52 seconds

### Relational analysis RSZ of RS_RSZ2_RSZ2_RSZ1_RSZ2_RSZ1_RSZ2_RSZ2

#### Relational analysis RSZ result of RS_RSZ2_RSZ2_RSZ1_RSZ2_RSZ1_RSZ2_RSZ2
Status: Status.UNKNOWN
Output dim: 8, lower bound: -0.0002177, upper bound: 0.0001823
time: 0.53 seconds

## BFS RS instance: RS_RSZ2_RSZ2_RSZ1_RSZ2_RSZ2_RSZ1

### Backsubstitution after applying RS history:
0: 0.0035972, 0.0041150, 0.0035972, 0.0041150, -0.0001271, 0.0001282
1: 0.0018420, 0.0019168, 0.0018420, 0.0019168, -0.0000184, 0.0000185
2: 0.0120848, 0.0123711, 0.0120848, 0.0123711, -0.0000709, 0.0000703
3: -0.0021818, -0.0018857, -0.0021818, -0.0018857, -0.0000733, 0.0000727
4: -0.0019956, -0.0016750, -0.0019956, -0.0016750, -0.0000787, 0.0000793
5: 0.0056985, 0.0060018, 0.0056985, 0.0060018, -0.0000751, 0.0000745
6: 0.0003095, 0.0015131, 0.0003095, 0.0015131, -0.0002979, 0.0002954
7: -0.0046174, -0.0029782, -0.0046174, -0.0029782, -0.0004024, 0.0004057
8: 0.9859612, 0.9871160, 0.9859612, 0.9871160, -0.0002834, 0.0002858
9: -0.0041920, -0.0031438, -0.0041920, -0.0031438, -0.0002594, 0.0002573

### Unstable ReLU Count (Linear/Conv2D Layers)
- layer_idx=0, type=LayerType.Linear, total=256, inp1_unstable=4, inp2_unstable=4, delta_unstable=252
- layer_idx=2, type=LayerType.Linear, total=256, inp1_unstable=4, inp2_unstable=4, delta_unstable=255
- layer_idx=4, type=LayerType.Linear, total=256, inp1_unstable=0, inp2_unstable=0, delta_unstable=256
- layer_idx=6, type=LayerType.Linear, total=256, inp1_unstable=0, inp2_unstable=0, delta_unstable=256
- layer_idx=8, type=LayerType.Linear, total=10, inp1_unstable=0, inp2_unstable=0, delta_unstable=10

Time for backsubstitution: 1.48 seconds

### RS candidates at layer 3
type: RSZ, layer: 3, pos: 34
type: RSZ, layer: 3, pos: 240
type: RSZ, layer: 3, pos: 49

Time for candidate selection: 0.00 seconds

### Candidate
type: RSZ, layer: 3, pos: 34

### Relational analysis RSZ of RS_RSZ2_RSZ2_RSZ1_RSZ2_RSZ2_RSZ1_RSZ1

#### Relational analysis RSZ result of RS_RSZ2_RSZ2_RSZ1_RSZ2_RSZ2_RSZ1_RSZ1
Status: Status.UNKNOWN
Output dim: 8, lower bound: -0.0001833, upper bound: 0.0002177
time: 0.52 seconds

### Relational analysis RSZ of RS_RSZ2_RSZ2_RSZ1_RSZ2_RSZ2_RSZ1_RSZ2

#### Relational analysis RSZ result of RS_RSZ2_RSZ2_RSZ1_RSZ2_RSZ2_RSZ1_RSZ2
Status: Status.UNKNOWN
Output dim: 8, lower bound: -0.0001832, upper bound: 0.0002198
time: 0.50 seconds

## BFS RS instance: RS_RSZ2_RSZ2_RSZ1_RSZ2_RSZ2_RSZ2

### Backsubstitution after applying RS history:
0: 0.0035972, 0.0041150, 0.0035972, 0.0041150, -0.0001311, 0.0001241
1: 0.0018420, 0.0019168, 0.0018420, 0.0019168, -0.0000189, 0.0000179
2: 0.0120848, 0.0123711, 0.0120848, 0.0123711, -0.0000686, 0.0000725
3: -0.0021818, -0.0018857, -0.0021818, -0.0018857, -0.0000710, 0.0000749
4: -0.0019956, -0.0016750, -0.0019956, -0.0016750, -0.0000811, 0.0000768
5: 0.0056985, 0.0060018, 0.0056985, 0.0060018, -0.0000727, 0.0000768
6: 0.0003095, 0.0015131, 0.0003095, 0.0015131, -0.0002885, 0.0003046
7: -0.0046174, -0.0029782, -0.0046174, -0.0029782, -0.0004149, 0.0003929
8: 0.9859612, 0.9871160, 0.9859612, 0.9871160, -0.0002923, 0.0002768
9: -0.0041920, -0.0031438, -0.0041920, -0.0031438, -0.0002512, 0.0002653

### Unstable ReLU Count (Linear/Conv2D Layers)
- layer_idx=0, type=LayerType.Linear, total=256, inp1_unstable=4, inp2_unstable=4, delta_unstable=252
- layer_idx=2, type=LayerType.Linear, total=256, inp1_unstable=4, inp2_unstable=4, delta_unstable=255
- layer_idx=4, type=LayerType.Linear, total=256, inp1_unstable=0, inp2_unstable=0, delta_unstable=256
- layer_idx=6, type=LayerType.Linear, total=256, inp1_unstable=0, inp2_unstable=0, delta_unstable=256
- layer_idx=8, type=LayerType.Linear, total=10, inp1_unstable=0, inp2_unstable=0, delta_unstable=10

Time for backsubstitution: 1.46 seconds

### RS candidates at layer 3
type: RSZ, layer: 3, pos: 240
type: RSZ, layer: 3, pos: 34
type: RSZ, layer: 3, pos: 49

Time for candidate selection: 0.00 seconds

### Candidate
type: RSZ, layer: 3, pos: 240

### Candidate
type: RSZ, layer: 3, pos: 34

### Relational analysis RSZ of RS_RSZ2_RSZ2_RSZ1_RSZ2_RSZ2_RSZ2_RSZ1

#### Relational analysis RSZ result of RS_RSZ2_RSZ2_RSZ1_RSZ2_RSZ2_RSZ2_RSZ1
Status: Status.UNKNOWN
Output dim: 8, lower bound: -0.0002202, upper bound: 0.0001827
time: 0.51 seconds

### Relational analysis RSZ of RS_RSZ2_RSZ2_RSZ1_RSZ2_RSZ2_RSZ2_RSZ2

#### Relational analysis RSZ result of RS_RSZ2_RSZ2_RSZ1_RSZ2_RSZ2_RSZ2_RSZ2
Status: Status.UNKNOWN
Output dim: 8, lower bound: -0.0002160, upper bound: 0.0001825
time: 0.49 seconds

## BFS RS instance: RS_RSZ2_RSZ2_RSZ2_RSZ1_RSZ1_RSZ1

### Backsubstitution after applying RS history:
0: 0.0035972, 0.0041150, 0.0035972, 0.0041150, -0.0001261, 0.0001300
1: 0.0018420, 0.0019168, 0.0018420, 0.0019168, -0.0000182, 0.0000188
2: 0.0120848, 0.0123711, 0.0120848, 0.0123711, -0.0000719, 0.0000697
3: -0.0021818, -0.0018857, -0.0021818, -0.0018857, -0.0000743, 0.0000721
4: -0.0019956, -0.0016750, -0.0019956, -0.0016750, -0.0000780, 0.0000805
5: 0.0056985, 0.0060018, 0.0056985, 0.0060018, -0.0000761, 0.0000738
6: 0.0003095, 0.0015131, 0.0003095, 0.0015131, -0.0003021, 0.0002930
7: -0.0046174, -0.0029782, -0.0046174, -0.0029782, -0.0003990, 0.0004114
8: 0.9859612, 0.9871160, 0.9859612, 0.9871160, -0.0002811, 0.0002898
9: -0.0041920, -0.0031438, -0.0041920, -0.0031438, -0.0002631, 0.0002551

### Unstable ReLU Count (Linear/Conv2D Layers)
- layer_idx=0, type=LayerType.Linear, total=256, inp1_unstable=4, inp2_unstable=4, delta_unstable=252
- layer_idx=2, type=LayerType.Linear, total=256, inp1_unstable=4, inp2_unstable=4, delta_unstable=255
- layer_idx=4, type=LayerType.Linear, total=256, inp1_unstable=0, inp2_unstable=0, delta_unstable=256
- layer_idx=6, type=LayerType.Linear, total=256, inp1_unstable=0, inp2_unstable=0, delta_unstable=256
- layer_idx=8, type=LayerType.Linear, total=10, inp1_unstable=0, inp2_unstable=0, delta_unstable=10

Time for backsubstitution: 1.45 seconds

### RS candidates at layer 3
type: RSZ, layer: 3, pos: 49
type: RSZ, layer: 3, pos: 240
type: RSZ, layer: 3, pos: 34

Time for candidate selection: 0.00 seconds

### Candidate
type: RSZ, layer: 3, pos: 49

### Relational analysis RSZ of RS_RSZ2_RSZ2_RSZ2_RSZ1_RSZ1_RSZ1_RSZ1

#### Relational analysis RSZ result of RS_RSZ2_RSZ2_RSZ2_RSZ1_RSZ1_RSZ1_RSZ1
Status: Status.VERIFIED
Output dim: 8, lower bound: -0.0001378, upper bound: 0.0002074
time: 0.53 seconds

### Relational analysis RSZ of RS_RSZ2_RSZ2_RSZ2_RSZ1_RSZ1_RSZ1_RSZ2

#### Relational analysis RSZ result of RS_RSZ2_RSZ2_RSZ2_RSZ1_RSZ1_RSZ1_RSZ2
Status: Status.VERIFIED
Output dim: 8, lower bound: -0.0001681, upper bound: 0.0001537
time: 0.52 seconds

## BFS RS instance: RS_RSZ2_RSZ2_RSZ2_RSZ1_RSZ1_RSZ2

### Backsubstitution after applying RS history:
0: 0.0035972, 0.0041150, 0.0035972, 0.0041150, -0.0001302, 0.0001265
1: 0.0018420, 0.0019168, 0.0018420, 0.0019168, -0.0000188, 0.0000183
2: 0.0120848, 0.0123711, 0.0120848, 0.0123711, -0.0000699, 0.0000720
3: -0.0021818, -0.0018857, -0.0021818, -0.0018857, -0.0000723, 0.0000745
4: -0.0019956, -0.0016750, -0.0019956, -0.0016750, -0.0000806, 0.0000783
5: 0.0056985, 0.0060018, 0.0056985, 0.0060018, -0.0000741, 0.0000763
6: 0.0003095, 0.0015131, 0.0003095, 0.0015131, -0.0002940, 0.0003027
7: -0.0046174, -0.0029782, -0.0046174, -0.0029782, -0.0004123, 0.0004005
8: 0.9859612, 0.9871160, 0.9859612, 0.9871160, -0.0002904, 0.0002821
9: -0.0041920, -0.0031438, -0.0041920, -0.0031438, -0.0002561, 0.0002636

### Unstable ReLU Count (Linear/Conv2D Layers)
- layer_idx=0, type=LayerType.Linear, total=256, inp1_unstable=4, inp2_unstable=4, delta_unstable=252
- layer_idx=2, type=LayerType.Linear, total=256, inp1_unstable=4, inp2_unstable=4, delta_unstable=255
- layer_idx=4, type=LayerType.Linear, total=256, inp1_unstable=0, inp2_unstable=0, delta_unstable=256
- layer_idx=6, type=LayerType.Linear, total=256, inp1_unstable=0, inp2_unstable=0, delta_unstable=256
- layer_idx=8, type=LayerType.Linear, total=10, inp1_unstable=0, inp2_unstable=0, delta_unstable=10

Time for backsubstitution: 1.45 seconds

### RS candidates at layer 3
type: RSZ, layer: 3, pos: 49
type: RSZ, layer: 3, pos: 240
type: RSZ, layer: 3, pos: 34

Time for candidate selection: 0.00 seconds

### Candidate
type: RSZ, layer: 3, pos: 49

### Relational analysis RSZ of RS_RSZ2_RSZ2_RSZ2_RSZ1_RSZ1_RSZ2_RSZ1

#### Relational analysis RSZ result of RS_RSZ2_RSZ2_RSZ2_RSZ1_RSZ1_RSZ2_RSZ1
Status: Status.VERIFIED
Output dim: 8, lower bound: -0.0001450, upper bound: 0.0001691
time: 0.52 seconds

### Relational analysis RSZ of RS_RSZ2_RSZ2_RSZ2_RSZ1_RSZ1_RSZ2_RSZ2

#### Relational analysis RSZ result of RS_RSZ2_RSZ2_RSZ2_RSZ1_RSZ1_RSZ2_RSZ2
Status: Status.VERIFIED
Output dim: 8, lower bound: -0.0002087, upper bound: 0.0001407
time: 0.52 seconds

## BFS RS instance: RS_RSZ2_RSZ2_RSZ2_RSZ1_RSZ2_RSZ1

### Backsubstitution after applying RS history:
0: 0.0035972, 0.0041150, 0.0035972, 0.0041150, -0.0001172, 0.0001264
1: 0.0018420, 0.0019168, 0.0018420, 0.0019168, -0.0000169, 0.0000183
2: 0.0120848, 0.0123711, 0.0120848, 0.0123711, -0.0000699, 0.0000648
3: -0.0021818, -0.0018857, -0.0021818, -0.0018857, -0.0000723, 0.0000670
4: -0.0019956, -0.0016750, -0.0019956, -0.0016750, -0.0000726, 0.0000782
5: 0.0056985, 0.0060018, 0.0056985, 0.0060018, -0.0000740, 0.0000687
6: 0.0003095, 0.0015131, 0.0003095, 0.0015131, -0.0002938, 0.0002724
7: -0.0046174, -0.0029782, -0.0046174, -0.0029782, -0.0003710, 0.0004001
8: 0.9859612, 0.9871160, 0.9859612, 0.9871160, -0.0002614, 0.0002818
9: -0.0041920, -0.0031438, -0.0041920, -0.0031438, -0.0002558, 0.0002373

### Unstable ReLU Count (Linear/Conv2D Layers)
- layer_idx=0, type=LayerType.Linear, total=256, inp1_unstable=4, inp2_unstable=4, delta_unstable=252
- layer_idx=2, type=LayerType.Linear, total=256, inp1_unstable=4, inp2_unstable=4, delta_unstable=255
- layer_idx=4, type=LayerType.Linear, total=256, inp1_unstable=0, inp2_unstable=0, delta_unstable=256
- layer_idx=6, type=LayerType.Linear, total=256, inp1_unstable=0, inp2_unstable=0, delta_unstable=256
- layer_idx=8, type=LayerType.Linear, total=10, inp1_unstable=0, inp2_unstable=0, delta_unstable=10

Time for backsubstitution: 1.44 seconds

### RS candidates at layer 3
type: RSZ, layer: 3, pos: 83
type: RSZ, layer: 3, pos: 240
type: RSZ, layer: 3, pos: 34

Time for candidate selection: 0.00 seconds

### Candidate
type: RSZ, layer: 3, pos: 83

### Relational analysis RSZ of RS_RSZ2_RSZ2_RSZ2_RSZ1_RSZ2_RSZ1_RSZ1

#### Relational analysis RSZ result of RS_RSZ2_RSZ2_RSZ2_RSZ1_RSZ2_RSZ1_RSZ1
Status: Status.VERIFIED
Output dim: 8, lower bound: -0.0001342, upper bound: 0.0002062
time: 0.55 seconds

### Relational analysis RSZ of RS_RSZ2_RSZ2_RSZ2_RSZ1_RSZ2_RSZ1_RSZ2

#### Relational analysis RSZ result of RS_RSZ2_RSZ2_RSZ2_RSZ1_RSZ2_RSZ1_RSZ2
Status: Status.VERIFIED
Output dim: 8, lower bound: -0.0001437, upper bound: 0.0001742
time: 0.54 seconds

## BFS RS instance: RS_RSZ2_RSZ2_RSZ2_RSZ1_RSZ2_RSZ2

### Backsubstitution after applying RS history:
0: 0.0035972, 0.0041150, 0.0035972, 0.0041150, -0.0001286, 0.0001177
1: 0.0018420, 0.0019168, 0.0018420, 0.0019168, -0.0000186, 0.0000170
2: 0.0120848, 0.0123711, 0.0120848, 0.0123711, -0.0000651, 0.0000711
3: -0.0021818, -0.0018857, -0.0021818, -0.0018857, -0.0000673, 0.0000735
4: -0.0019956, -0.0016750, -0.0019956, -0.0016750, -0.0000796, 0.0000728
5: 0.0056985, 0.0060018, 0.0056985, 0.0060018, -0.0000689, 0.0000753
6: 0.0003095, 0.0015131, 0.0003095, 0.0015131, -0.0002735, 0.0002988
7: -0.0046174, -0.0029782, -0.0046174, -0.0029782, -0.0004070, 0.0003725
8: 0.9859612, 0.9871160, 0.9859612, 0.9871160, -0.0002867, 0.0002624
9: -0.0041920, -0.0031438, -0.0041920, -0.0031438, -0.0002382, 0.0002602

### Unstable ReLU Count (Linear/Conv2D Layers)
- layer_idx=0, type=LayerType.Linear, total=256, inp1_unstable=4, inp2_unstable=4, delta_unstable=252
- layer_idx=2, type=LayerType.Linear, total=256, inp1_unstable=4, inp2_unstable=4, delta_unstable=255
- layer_idx=4, type=LayerType.Linear, total=256, inp1_unstable=0, inp2_unstable=0, delta_unstable=256
- layer_idx=6, type=LayerType.Linear, total=256, inp1_unstable=0, inp2_unstable=0, delta_unstable=256
- layer_idx=8, type=LayerType.Linear, total=10, inp1_unstable=0, inp2_unstable=0, delta_unstable=10

Time for backsubstitution: 1.47 seconds

### RS candidates at layer 3
type: RSZ, layer: 3, pos: 83
type: RSZ, layer: 3, pos: 240
type: RSZ, layer: 3, pos: 34

Time for candidate selection: 0.00 seconds

### Candidate
type: RSZ, layer: 3, pos: 83

### Relational analysis RSZ of RS_RSZ2_RSZ2_RSZ2_RSZ1_RSZ2_RSZ2_RSZ1

#### Relational analysis RSZ result of RS_RSZ2_RSZ2_RSZ2_RSZ1_RSZ2_RSZ2_RSZ1
Status: Status.VERIFIED
Output dim: 8, lower bound: -0.0001654, upper bound: 0.0001543
time: 0.52 seconds

### Relational analysis RSZ of RS_RSZ2_RSZ2_RSZ2_RSZ1_RSZ2_RSZ2_RSZ2

#### Relational analysis RSZ result of RS_RSZ2_RSZ2_RSZ2_RSZ1_RSZ2_RSZ2_RSZ2
Status: Status.VERIFIED
Output dim: 8, lower bound: -0.0002107, upper bound: 0.0001498
time: 0.54 seconds

## BFS RS instance: RS_RSZ2_RSZ2_RSZ2_RSZ2_RSZ1_RSZ1

### Backsubstitution after applying RS history:
0: 0.0035972, 0.0041150, 0.0035972, 0.0041150, -0.0001294, 0.0001284
1: 0.0018420, 0.0019168, 0.0018420, 0.0019168, -0.0000187, 0.0000186
2: 0.0120848, 0.0123711, 0.0120848, 0.0123711, -0.0000710, 0.0000715
3: -0.0021818, -0.0018857, -0.0021818, -0.0018857, -0.0000734, 0.0000740
4: -0.0019956, -0.0016750, -0.0019956, -0.0016750, -0.0000801, 0.0000795
5: 0.0056985, 0.0060018, 0.0056985, 0.0060018, -0.0000752, 0.0000758
6: 0.0003095, 0.0015131, 0.0003095, 0.0015131, -0.0002985, 0.0003007
7: -0.0046174, -0.0029782, -0.0046174, -0.0029782, -0.0004095, 0.0004065
8: 0.9859612, 0.9871160, 0.9859612, 0.9871160, -0.0002884, 0.0002863
9: -0.0041920, -0.0031438, -0.0041920, -0.0031438, -0.0002599, 0.0002618

### Unstable ReLU Count (Linear/Conv2D Layers)
- layer_idx=0, type=LayerType.Linear, total=256, inp1_unstable=4, inp2_unstable=4, delta_unstable=252
- layer_idx=2, type=LayerType.Linear, total=256, inp1_unstable=4, inp2_unstable=4, delta_unstable=255
- layer_idx=4, type=LayerType.Linear, total=256, inp1_unstable=0, inp2_unstable=0, delta_unstable=256
- layer_idx=6, type=LayerType.Linear, total=256, inp1_unstable=0, inp2_unstable=0, delta_unstable=256
- layer_idx=8, type=LayerType.Linear, total=10, inp1_unstable=0, inp2_unstable=0, delta_unstable=10

Time for backsubstitution: 1.52 seconds

### RS candidates at layer 3
type: RSZ, layer: 3, pos: 240
type: RSZ, layer: 3, pos: 49
type: RSZ, layer: 3, pos: 34

Time for candidate selection: 0.00 seconds

### Candidate
type: RSZ, layer: 3, pos: 240

### Candidate
type: RSZ, layer: 3, pos: 49

### Relational analysis RSZ of RS_RSZ2_RSZ2_RSZ2_RSZ2_RSZ1_RSZ1_RSZ1

#### Relational analysis RSZ result of RS_RSZ2_RSZ2_RSZ2_RSZ2_RSZ1_RSZ1_RSZ1
Status: Status.VERIFIED
Output dim: 8, lower bound: -0.0001492, upper bound: 0.0002055
time: 0.53 seconds

### Relational analysis RSZ of RS_RSZ2_RSZ2_RSZ2_RSZ2_RSZ1_RSZ1_RSZ2

#### Relational analysis RSZ result of RS_RSZ2_RSZ2_RSZ2_RSZ2_RSZ1_RSZ1_RSZ2
Status: Status.VERIFIED
Output dim: 8, lower bound: -0.0001858, upper bound: 0.0001505
time: 0.55 seconds

## BFS RS instance: RS_RSZ2_RSZ2_RSZ2_RSZ2_RSZ1_RSZ2

### Backsubstitution after applying RS history:
0: 0.0035972, 0.0041150, 0.0035972, 0.0041150, -0.0001333, 0.0001246
1: 0.0018420, 0.0019168, 0.0018420, 0.0019168, -0.0000193, 0.0000180
2: 0.0120848, 0.0123711, 0.0120848, 0.0123711, -0.0000689, 0.0000737
3: -0.0021818, -0.0018857, -0.0021818, -0.0018857, -0.0000712, 0.0000762
4: -0.0019956, -0.0016750, -0.0019956, -0.0016750, -0.0000825, 0.0000771
5: 0.0056985, 0.0060018, 0.0056985, 0.0060018, -0.0000730, 0.0000781
6: 0.0003095, 0.0015131, 0.0003095, 0.0015131, -0.0002896, 0.0003097
7: -0.0046174, -0.0029782, -0.0046174, -0.0029782, -0.0004218, 0.0003944
8: 0.9859612, 0.9871160, 0.9859612, 0.9871160, -0.0002972, 0.0002778
9: -0.0041920, -0.0031438, -0.0041920, -0.0031438, -0.0002522, 0.0002697

### Unstable ReLU Count (Linear/Conv2D Layers)
- layer_idx=0, type=LayerType.Linear, total=256, inp1_unstable=4, inp2_unstable=4, delta_unstable=252
- layer_idx=2, type=LayerType.Linear, total=256, inp1_unstable=4, inp2_unstable=4, delta_unstable=255
- layer_idx=4, type=LayerType.Linear, total=256, inp1_unstable=0, inp2_unstable=0, delta_unstable=256
- layer_idx=6, type=LayerType.Linear, total=256, inp1_unstable=0, inp2_unstable=0, delta_unstable=256
- layer_idx=8, type=LayerType.Linear, total=10, inp1_unstable=0, inp2_unstable=0, delta_unstable=10

Time for backsubstitution: 1.53 seconds

### RS candidates at layer 3
type: RSZ, layer: 3, pos: 240
type: RSZ, layer: 3, pos: 49
type: RSZ, layer: 3, pos: 34

Time for candidate selection: 0.00 seconds

### Candidate
type: RSZ, layer: 3, pos: 240

### Candidate
type: RSZ, layer: 3, pos: 49

### Relational analysis RSZ of RS_RSZ2_RSZ2_RSZ2_RSZ2_RSZ1_RSZ2_RSZ1

#### Relational analysis RSZ result of RS_RSZ2_RSZ2_RSZ2_RSZ2_RSZ1_RSZ2_RSZ1
Status: Status.VERIFIED
Output dim: 8, lower bound: -0.0001496, upper bound: 0.0001549
time: 0.53 seconds

### Relational analysis RSZ of RS_RSZ2_RSZ2_RSZ2_RSZ2_RSZ1_RSZ2_RSZ2

#### Relational analysis RSZ result of RS_RSZ2_RSZ2_RSZ2_RSZ2_RSZ1_RSZ2_RSZ2
Status: Status.VERIFIED
Output dim: 8, lower bound: -0.0002146, upper bound: 0.0001369
time: 0.53 seconds

## BFS RS instance: RS_RSZ2_RSZ2_RSZ2_RSZ2_RSZ2_RSZ1

### Backsubstitution after applying RS history:
0: 0.0035972, 0.0041150, 0.0035972, 0.0041150, -0.0001213, 0.0001148
1: 0.0018420, 0.0019168, 0.0018420, 0.0019168, -0.0000175, 0.0000166
2: 0.0120848, 0.0123711, 0.0120848, 0.0123711, -0.0000635, 0.0000671
3: -0.0021818, -0.0018857, -0.0021818, -0.0018857, -0.0000657, 0.0000694
4: -0.0019956, -0.0016750, -0.0019956, -0.0016750, -0.0000751, 0.0000711
5: 0.0056985, 0.0060018, 0.0056985, 0.0060018, -0.0000673, 0.0000711
6: 0.0003095, 0.0015131, 0.0003095, 0.0015131, -0.0002669, 0.0002820
7: -0.0046174, -0.0029782, -0.0046174, -0.0029782, -0.0003841, 0.0003635
8: 0.9859612, 0.9871160, 0.9859612, 0.9871160, -0.0002705, 0.0002560
9: -0.0041920, -0.0031438, -0.0041920, -0.0031438, -0.0002324, 0.0002456

### Unstable ReLU Count (Linear/Conv2D Layers)
- layer_idx=0, type=LayerType.Linear, total=256, inp1_unstable=4, inp2_unstable=4, delta_unstable=252
- layer_idx=2, type=LayerType.Linear, total=256, inp1_unstable=4, inp2_unstable=4, delta_unstable=255
- layer_idx=4, type=LayerType.Linear, total=256, inp1_unstable=0, inp2_unstable=0, delta_unstable=256
- layer_idx=6, type=LayerType.Linear, total=256, inp1_unstable=0, inp2_unstable=0, delta_unstable=256
- layer_idx=8, type=LayerType.Linear, total=10, inp1_unstable=0, inp2_unstable=0, delta_unstable=10

Time for backsubstitution: 1.51 seconds

### RS candidates at layer 3
type: RSZ, layer: 3, pos: 240
type: RSZ, layer: 3, pos: 49
type: RSZ, layer: 3, pos: 83

Time for candidate selection: 0.00 seconds

### Candidate
type: RSZ, layer: 3, pos: 240

### Candidate
type: RSZ, layer: 3, pos: 49

### Relational analysis RSZ of RS_RSZ2_RSZ2_RSZ2_RSZ2_RSZ2_RSZ1_RSZ1

#### Relational analysis RSZ result of RS_RSZ2_RSZ2_RSZ2_RSZ2_RSZ2_RSZ1_RSZ1
Status: Status.VERIFIED
Output dim: 8, lower bound: -0.0001495, upper bound: 0.0001997
time: 0.54 seconds

### Relational analysis RSZ of RS_RSZ2_RSZ2_RSZ2_RSZ2_RSZ2_RSZ1_RSZ2

#### Relational analysis RSZ result of RS_RSZ2_RSZ2_RSZ2_RSZ2_RSZ2_RSZ1_RSZ2
Status: Status.UNKNOWN
Output dim: 8, lower bound: -0.0002222, upper bound: 0.0001574
time: 0.54 seconds

## BFS RS instance: RS_RSZ2_RSZ2_RSZ2_RSZ2_RSZ2_RSZ2

### Backsubstitution after applying RS history:
0: 0.0035972, 0.0041150, 0.0035972, 0.0041150, -0.0001203, 0.0001166
1: 0.0018420, 0.0019168, 0.0018420, 0.0019168, -0.0000174, 0.0000168
2: 0.0120848, 0.0123711, 0.0120848, 0.0123711, -0.0000644, 0.0000665
3: -0.0021818, -0.0018857, -0.0021818, -0.0018857, -0.0000666, 0.0000688
4: -0.0019956, -0.0016750, -0.0019956, -0.0016750, -0.0000745, 0.0000721
5: 0.0056985, 0.0060018, 0.0056985, 0.0060018, -0.0000683, 0.0000705
6: 0.0003095, 0.0015131, 0.0003095, 0.0015131, -0.0002709, 0.0002796
7: -0.0046174, -0.0029782, -0.0046174, -0.0029782, -0.0003808, 0.0003689
8: 0.9859612, 0.9871160, 0.9859612, 0.9871160, -0.0002683, 0.0002599
9: -0.0041920, -0.0031438, -0.0041920, -0.0031438, -0.0002359, 0.0002435

### Unstable ReLU Count (Linear/Conv2D Layers)
- layer_idx=0, type=LayerType.Linear, total=256, inp1_unstable=4, inp2_unstable=4, delta_unstable=252
- layer_idx=2, type=LayerType.Linear, total=256, inp1_unstable=4, inp2_unstable=4, delta_unstable=255
- layer_idx=4, type=LayerType.Linear, total=256, inp1_unstable=0, inp2_unstable=0, delta_unstable=256
- layer_idx=6, type=LayerType.Linear, total=256, inp1_unstable=0, inp2_unstable=0, delta_unstable=256
- layer_idx=8, type=LayerType.Linear, total=10, inp1_unstable=0, inp2_unstable=0, delta_unstable=10

Time for backsubstitution: 1.51 seconds

### RS candidates at layer 3
type: RSZ, layer: 3, pos: 83
type: RSZ, layer: 3, pos: 49
type: RSZ, layer: 3, pos: 240

Time for candidate selection: 0.00 seconds

### Candidate
type: RSZ, layer: 3, pos: 83

### Relational analysis RSZ of RS_RSZ2_RSZ2_RSZ2_RSZ2_RSZ2_RSZ2_RSZ1

#### Relational analysis RSZ result of RS_RSZ2_RSZ2_RSZ2_RSZ2_RSZ2_RSZ2_RSZ1
Status: Status.VERIFIED
Output dim: 8, lower bound: -0.0001889, upper bound: 0.0002134
time: 0.54 seconds

### Relational analysis RSZ of RS_RSZ2_RSZ2_RSZ2_RSZ2_RSZ2_RSZ2_RSZ2

#### Relational analysis RSZ result of RS_RSZ2_RSZ2_RSZ2_RSZ2_RSZ2_RSZ2_RSZ2
Status: Status.UNKNOWN
Output dim: 8, lower bound: -0.0002224, upper bound: 0.0001739
time: 0.53 seconds

## Summary of splitting (split count: 6)
- Time for RS candidates: 2.58 seconds
RS_RSZ1_RSZ1_RSZ1_RSZ1_RSZ1_RSZ1_RSZ1, status: Status.VERIFIED, split count: 7, time: 2.58
Output dim: 8, lower bound: -0.0001260, upper bound: 0.0001992
RS_RSZ1_RSZ1_RSZ1_RSZ1_RSZ1_RSZ1_RSZ2, status: Status.VERIFIED, split count: 7, time: 2.58
Output dim: 8, lower bound: -0.0001423, upper bound: 0.0001379
RS_RSZ1_RSZ1_RSZ1_RSZ1_RSZ1_RSZ2_RSZ1, status: Status.VERIFIED, split count: 7, time: 2.58
Output dim: 8, lower bound: -0.0002134, upper bound: 0.0001889
RS_RSZ1_RSZ1_RSZ1_RSZ1_RSZ1_RSZ2_RSZ2, status: Status.UNKNOWN, split count: 7, time: 2.58
Output dim: 8, lower bound: -0.0002177, upper bound: 0.0002028
RS_RSZ1_RSZ1_RSZ1_RSZ1_RSZ2_RSZ1_RSZ1, status: Status.VERIFIED, split count: 7, time: 2.58
Output dim: 8, lower bound: -0.0001271, upper bound: 0.0002028
RS_RSZ1_RSZ1_RSZ1_RSZ1_RSZ2_RSZ1_RSZ2, status: Status.VERIFIED, split count: 7, time: 2.58
Output dim: 8, lower bound: -0.0001388, upper bound: 0.0001735
RS_RSZ1_RSZ1_RSZ1_RSZ2_RSZ1_RSZ1_RSZ1, status: Status.UNKNOWN, split count: 7, time: 2.58
Output dim: 8, lower bound: -0.0001933, upper bound: 0.0002175
RS_RSZ1_RSZ1_RSZ1_RSZ2_RSZ1_RSZ1_RSZ2, status: Status.UNKNOWN, split count: 7, time: 2.58
Output dim: 8, lower bound: -0.0002200, upper bound: 0.0001805
RS_RSZ1_RSZ1_RSZ1_RSZ2_RSZ1_RSZ2_RSZ1, status: Status.VERIFIED, split count: 7, time: 2.58
Output dim: 8, lower bound: -0.0001616, upper bound: 0.0002100
RS_RSZ1_RSZ1_RSZ1_RSZ2_RSZ1_RSZ2_RSZ2, status: Status.UNKNOWN, split count: 7, time: 2.58
Output dim: 8, lower bound: -0.0002161, upper bound: 0.0001499
RS_RSZ1_RSZ1_RSZ2_RSZ1_RSZ1_RSZ1_RSZ1, status: Status.VERIFIED, split count: 7, time: 2.58
Output dim: 8, lower bound: -0.0001418, upper bound: 0.0002063
RS_RSZ1_RSZ1_RSZ2_RSZ1_RSZ1_RSZ1_RSZ2, status: Status.VERIFIED, split count: 7, time: 2.58
Output dim: 8, lower bound: -0.0001505, upper bound: 0.0001662
RS_RSZ1_RSZ1_RSZ2_RSZ1_RSZ1_RSZ2_RSZ1, status: Status.VERIFIED, split count: 7, time: 2.58
Output dim: 8, lower bound: -0.0001380, upper bound: 0.0002060
RS_RSZ1_RSZ1_RSZ2_RSZ1_RSZ1_RSZ2_RSZ2, status: Status.VERIFIED, split count: 7, time: 2.58
Output dim: 8, lower bound: -0.0001504, upper bound: 0.0001741
RS_RSZ1_RSZ1_RSZ2_RSZ2_RSZ1_RSZ1_RSZ1, status: Status.VERIFIED, split count: 7, time: 2.58
Output dim: 8, lower bound: -0.0001611, upper bound: 0.0002005
RS_RSZ1_RSZ1_RSZ2_RSZ2_RSZ1_RSZ1_RSZ2, status: Status.VERIFIED, split count: 7, time: 2.58
Output dim: 8, lower bound: -0.0001630, upper bound: 0.0002074
RS_RSZ1_RSZ1_RSZ2_RSZ2_RSZ1_RSZ2_RSZ1, status: Status.VERIFIED, split count: 7, time: 2.58
Output dim: 8, lower bound: -0.0001853, upper bound: 0.0001441
RS_RSZ1_RSZ1_RSZ2_RSZ2_RSZ1_RSZ2_RSZ2, status: Status.VERIFIED, split count: 7, time: 2.58
Output dim: 8, lower bound: -0.0002144, upper bound: 0.0001337
RS_RSZ1_RSZ1_RSZ2_RSZ2_RSZ2_RSZ2_RSZ1, status: Status.UNKNOWN, split count: 7, time: 2.58
Output dim: 8, lower bound: -0.0002222, upper bound: 0.0001502
RS_RSZ1_RSZ1_RSZ2_RSZ2_RSZ2_RSZ2_RSZ2, status: Status.UNKNOWN, split count: 7, time: 2.58
Output dim: 8, lower bound: -0.0002193, upper bound: 0.0001504
RS_RSZ1_RSZ2_RSZ1_RSZ1_RSZ1_RSZ1_RSZ1, status: Status.UNKNOWN, split count: 7, time: 2.58
Output dim: 8, lower bound: -0.0001728, upper bound: 0.0002221
RS_RSZ1_RSZ2_RSZ1_RSZ1_RSZ1_RSZ1_RSZ2, status: Status.UNKNOWN, split count: 7, time: 2.58
Output dim: 8, lower bound: -0.0001737, upper bound: 0.0002256
RS_RSZ1_RSZ2_RSZ1_RSZ1_RSZ1_RSZ2_RSZ1, status: Status.UNKNOWN, split count: 7, time: 2.58
Output dim: 8, lower bound: -0.0001706, upper bound: 0.0002222
RS_RSZ1_RSZ2_RSZ1_RSZ1_RSZ1_RSZ2_RSZ2, status: Status.UNKNOWN, split count: 7, time: 2.58
Output dim: 8, lower bound: -0.0001706, upper bound: 0.0002252
RS_RSZ1_RSZ2_RSZ1_RSZ1_RSZ2_RSZ1_RSZ1, status: Status.VERIFIED, split count: 7, time: 2.58
Output dim: 8, lower bound: -0.0001401, upper bound: 0.0001736
RS_RSZ1_RSZ2_RSZ1_RSZ1_RSZ2_RSZ1_RSZ2, status: Status.VERIFIED, split count: 7, time: 2.58
Output dim: 8, lower bound: -0.0001908, upper bound: 0.0001317
RS_RSZ1_RSZ2_RSZ1_RSZ2_RSZ2_RSZ1_RSZ1, status: Status.UNKNOWN, split count: 7, time: 2.58
Output dim: 8, lower bound: -0.0002200, upper bound: 0.0001850
RS_RSZ1_RSZ2_RSZ1_RSZ2_RSZ2_RSZ1_RSZ2, status: Status.UNKNOWN, split count: 7, time: 2.58
Output dim: 8, lower bound: -0.0002211, upper bound: 0.0001935
RS_RSZ1_RSZ2_RSZ1_RSZ2_RSZ2_RSZ2_RSZ1, status: Status.VERIFIED, split count: 7, time: 2.58
Output dim: 8, lower bound: -0.0001411, upper bound: 0.0001623
RS_RSZ1_RSZ2_RSZ1_RSZ2_RSZ2_RSZ2_RSZ2, status: Status.VERIFIED, split count: 7, time: 2.58
Output dim: 8, lower bound: -0.0001940, upper bound: 0.0001308
RS_RSZ1_RSZ2_RSZ2_RSZ1_RSZ1_RSZ1_RSZ1, status: Status.VERIFIED, split count: 7, time: 2.58
Output dim: 8, lower bound: -0.0001526, upper bound: 0.0002110
RS_RSZ1_RSZ2_RSZ2_RSZ1_RSZ1_RSZ1_RSZ2, status: Status.VERIFIED, split count: 7, time: 2.58
Output dim: 8, lower bound: -0.0001577, upper bound: 0.0001704
RS_RSZ1_RSZ2_RSZ2_RSZ1_RSZ1_RSZ2_RSZ1, status: Status.UNKNOWN, split count: 7, time: 2.58
Output dim: 8, lower bound: -0.0002160, upper bound: 0.0001457
RS_RSZ1_RSZ2_RSZ2_RSZ1_RSZ1_RSZ2_RSZ2, status: Status.VERIFIED, split count: 7, time: 2.58
Output dim: 8, lower bound: -0.0002127, upper bound: 0.0001472
RS_RSZ1_RSZ2_RSZ2_RSZ1_RSZ2_RSZ1_RSZ1, status: Status.VERIFIED, split count: 7, time: 2.58
Output dim: 8, lower bound: -0.0001630, upper bound: 0.0002095
RS_RSZ1_RSZ2_RSZ2_RSZ1_RSZ2_RSZ1_RSZ2, status: Status.VERIFIED, split count: 7, time: 2.58
Output dim: 8, lower bound: -0.0001638, upper bound: 0.0002122
RS_RSZ1_RSZ2_RSZ2_RSZ1_RSZ2_RSZ2_RSZ1, status: Status.UNKNOWN, split count: 7, time: 2.58
Output dim: 8, lower bound: -0.0002160, upper bound: 0.0001475
RS_RSZ1_RSZ2_RSZ2_RSZ1_RSZ2_RSZ2_RSZ2, status: Status.VERIFIED, split count: 7, time: 2.58
Output dim: 8, lower bound: -0.0002119, upper bound: 0.0001476
RS_RSZ1_RSZ2_RSZ2_RSZ2_RSZ2_RSZ1_RSZ1, status: Status.UNKNOWN, split count: 7, time: 2.58
Output dim: 8, lower bound: -0.0002221, upper bound: 0.0001457
RS_RSZ1_RSZ2_RSZ2_RSZ2_RSZ2_RSZ1_RSZ2, status: Status.UNKNOWN, split count: 7, time: 2.58
Output dim: 8, lower bound: -0.0002200, upper bound: 0.0001474
RS_RSZ1_RSZ2_RSZ2_RSZ2_RSZ2_RSZ2_RSZ1, status: Status.UNKNOWN, split count: 7, time: 2.58
Output dim: 8, lower bound: -0.0002222, upper bound: 0.0001475
RS_RSZ1_RSZ2_RSZ2_RSZ2_RSZ2_RSZ2_RSZ2, status: Status.UNKNOWN, split count: 7, time: 2.58
Output dim: 8, lower bound: -0.0002198, upper bound: 0.0001477
RS_RSZ2_RSZ1_RSZ1_RSZ1_RSZ1_RSZ1_RSZ1, status: Status.UNKNOWN, split count: 7, time: 2.58
Output dim: 8, lower bound: -0.0001749, upper bound: 0.0002232
RS_RSZ2_RSZ1_RSZ1_RSZ1_RSZ1_RSZ1_RSZ2, status: Status.UNKNOWN, split count: 7, time: 2.58
Output dim: 8, lower bound: -0.0001750, upper bound: 0.0002255
RS_RSZ2_RSZ1_RSZ1_RSZ1_RSZ1_RSZ2_RSZ1, status: Status.VERIFIED, split count: 7, time: 2.58
Output dim: 8, lower bound: -0.0001342, upper bound: 0.0002146
RS_RSZ2_RSZ1_RSZ1_RSZ1_RSZ1_RSZ2_RSZ2, status: Status.VERIFIED, split count: 7, time: 2.58
Output dim: 8, lower bound: -0.0001576, upper bound: 0.0001578
RS_RSZ2_RSZ1_RSZ1_RSZ2_RSZ1_RSZ1_RSZ1, status: Status.UNKNOWN, split count: 7, time: 2.58
Output dim: 8, lower bound: -0.0001846, upper bound: 0.0002177
RS_RSZ2_RSZ1_RSZ1_RSZ2_RSZ1_RSZ1_RSZ2, status: Status.UNKNOWN, split count: 7, time: 2.58
Output dim: 8, lower bound: -0.0001845, upper bound: 0.0002182
RS_RSZ2_RSZ1_RSZ1_RSZ2_RSZ1_RSZ2_RSZ1, status: Status.UNKNOWN, split count: 7, time: 2.58
Output dim: 8, lower bound: -0.0001844, upper bound: 0.0002202
RS_RSZ2_RSZ1_RSZ1_RSZ2_RSZ1_RSZ2_RSZ2, status: Status.UNKNOWN, split count: 7, time: 2.58
Output dim: 8, lower bound: -0.0001833, upper bound: 0.0002200
RS_RSZ2_RSZ1_RSZ1_RSZ2_RSZ2_RSZ1_RSZ1, status: Status.VERIFIED, split count: 7, time: 2.58
Output dim: 8, lower bound: -0.0001421, upper bound: 0.0001655
RS_RSZ2_RSZ1_RSZ1_RSZ2_RSZ2_RSZ1_RSZ2, status: Status.VERIFIED, split count: 7, time: 2.58
Output dim: 8, lower bound: -0.0002086, upper bound: 0.0001407
RS_RSZ2_RSZ1_RSZ1_RSZ2_RSZ2_RSZ2_RSZ1, status: Status.UNKNOWN, split count: 7, time: 2.58
Output dim: 8, lower bound: -0.0002212, upper bound: 0.0001935
RS_RSZ2_RSZ1_RSZ1_RSZ2_RSZ2_RSZ2_RSZ2, status: Status.UNKNOWN, split count: 7, time: 2.58
Output dim: 8, lower bound: -0.0002169, upper bound: 0.0001934
RS_RSZ2_RSZ1_RSZ2_RSZ1_RSZ1_RSZ1_RSZ1, status: Status.VERIFIED, split count: 7, time: 2.58
Output dim: 8, lower bound: -0.0001308, upper bound: 0.0001940
RS_RSZ2_RSZ1_RSZ2_RSZ1_RSZ1_RSZ1_RSZ2, status: Status.VERIFIED, split count: 7, time: 2.58
Output dim: 8, lower bound: -0.0001623, upper bound: 0.0001411
RS_RSZ2_RSZ1_RSZ2_RSZ1_RSZ1_RSZ2_RSZ1, status: Status.UNKNOWN, split count: 7, time: 2.58
Output dim: 8, lower bound: -0.0001935, upper bound: 0.0002211
RS_RSZ2_RSZ1_RSZ2_RSZ1_RSZ1_RSZ2_RSZ2, status: Status.UNKNOWN, split count: 7, time: 2.58
Output dim: 8, lower bound: -0.0001850, upper bound: 0.0002200
RS_RSZ2_RSZ1_RSZ2_RSZ1_RSZ2_RSZ1_RSZ1, status: Status.VERIFIED, split count: 7, time: 2.58
Output dim: 8, lower bound: -0.0001325, upper bound: 0.0001531
RS_RSZ2_RSZ1_RSZ2_RSZ1_RSZ2_RSZ1_RSZ2, status: Status.VERIFIED, split count: 7, time: 2.58
Output dim: 8, lower bound: -0.0001954, upper bound: 0.0001305
RS_RSZ2_RSZ1_RSZ2_RSZ1_RSZ2_RSZ2_RSZ1, status: Status.UNKNOWN, split count: 7, time: 2.58
Output dim: 8, lower bound: -0.0002176, upper bound: 0.0001799
RS_RSZ2_RSZ1_RSZ2_RSZ1_RSZ2_RSZ2_RSZ2, status: Status.UNKNOWN, split count: 7, time: 2.58
Output dim: 8, lower bound: -0.0002160, upper bound: 0.0001820
RS_RSZ2_RSZ1_RSZ2_RSZ2_RSZ1_RSZ2_RSZ1, status: Status.VERIFIED, split count: 7, time: 2.58
Output dim: 8, lower bound: -0.0001323, upper bound: 0.0001414
RS_RSZ2_RSZ1_RSZ2_RSZ2_RSZ1_RSZ2_RSZ2, status: Status.VERIFIED, split count: 7, time: 2.58
Output dim: 8, lower bound: -0.0002028, upper bound: 0.0001271
RS_RSZ2_RSZ1_RSZ2_RSZ2_RSZ2_RSZ1_RSZ1, status: Status.VERIFIED, split count: 7, time: 2.58
Output dim: 8, lower bound: -0.0001515, upper bound: 0.0002073
RS_RSZ2_RSZ1_RSZ2_RSZ2_RSZ2_RSZ1_RSZ2, status: Status.UNKNOWN, split count: 7, time: 2.58
Output dim: 8, lower bound: -0.0002179, upper bound: 0.0001576
RS_RSZ2_RSZ1_RSZ2_RSZ2_RSZ2_RSZ2_RSZ1, status: Status.VERIFIED, split count: 7, time: 2.58
Output dim: 8, lower bound: -0.0001463, upper bound: 0.0002011
RS_RSZ2_RSZ1_RSZ2_RSZ2_RSZ2_RSZ2_RSZ2, status: Status.UNKNOWN, split count: 7, time: 2.58
Output dim: 8, lower bound: -0.0002178, upper bound: 0.0001576
RS_RSZ2_RSZ2_RSZ1_RSZ1_RSZ1_RSZ1_RSZ1, status: Status.UNKNOWN, split count: 7, time: 2.58
Output dim: 8, lower bound: -0.0001740, upper bound: 0.0002228
RS_RSZ2_RSZ2_RSZ1_RSZ1_RSZ1_RSZ1_RSZ2, status: Status.UNKNOWN, split count: 7, time: 2.58
Output dim: 8, lower bound: -0.0001707, upper bound: 0.0002228
RS_RSZ2_RSZ2_RSZ1_RSZ1_RSZ1_RSZ2_RSZ1, status: Status.UNKNOWN, split count: 7, time: 2.58
Output dim: 8, lower bound: -0.0001739, upper bound: 0.0002256
RS_RSZ2_RSZ2_RSZ1_RSZ1_RSZ1_RSZ2_RSZ2, status: Status.UNKNOWN, split count: 7, time: 2.58
Output dim: 8, lower bound: -0.0001705, upper bound: 0.0002252
RS_RSZ2_RSZ2_RSZ1_RSZ1_RSZ2_RSZ1_RSZ1, status: Status.VERIFIED, split count: 7, time: 2.58
Output dim: 8, lower bound: -0.0002134, upper bound: 0.0001931
RS_RSZ2_RSZ2_RSZ1_RSZ1_RSZ2_RSZ1_RSZ2, status: Status.VERIFIED, split count: 7, time: 2.58
Output dim: 8, lower bound: -0.0002117, upper bound: 0.0001931
RS_RSZ2_RSZ2_RSZ1_RSZ1_RSZ2_RSZ2_RSZ1, status: Status.VERIFIED, split count: 7, time: 2.58
Output dim: 8, lower bound: -0.0001441, upper bound: 0.0001853
RS_RSZ2_RSZ2_RSZ1_RSZ1_RSZ2_RSZ2_RSZ2, status: Status.VERIFIED, split count: 7, time: 2.58
Output dim: 8, lower bound: -0.0002033, upper bound: 0.0001505
RS_RSZ2_RSZ2_RSZ1_RSZ2_RSZ1_RSZ1_RSZ1, status: Status.VERIFIED, split count: 7, time: 2.58
Output dim: 8, lower bound: -0.0001488, upper bound: 0.0002110
RS_RSZ2_RSZ2_RSZ1_RSZ2_RSZ1_RSZ1_RSZ2, status: Status.VERIFIED, split count: 7, time: 2.58
Output dim: 8, lower bound: -0.0001741, upper bound: 0.0001504
RS_RSZ2_RSZ2_RSZ1_RSZ2_RSZ1_RSZ2_RSZ1, status: Status.UNKNOWN, split count: 7, time: 2.58
Output dim: 8, lower bound: -0.0002200, upper bound: 0.0001821
RS_RSZ2_RSZ2_RSZ1_RSZ2_RSZ1_RSZ2_RSZ2, status: Status.UNKNOWN, split count: 7, time: 2.58
Output dim: 8, lower bound: -0.0002177, upper bound: 0.0001823
RS_RSZ2_RSZ2_RSZ1_RSZ2_RSZ2_RSZ1_RSZ1, status: Status.UNKNOWN, split count: 7, time: 2.58
Output dim: 8, lower bound: -0.0001833, upper bound: 0.0002177
RS_RSZ2_RSZ2_RSZ1_RSZ2_RSZ2_RSZ1_RSZ2, status: Status.UNKNOWN, split count: 7, time: 2.58
Output dim: 8, lower bound: -0.0001832, upper bound: 0.0002198
RS_RSZ2_RSZ2_RSZ1_RSZ2_RSZ2_RSZ2_RSZ1, status: Status.UNKNOWN, split count: 7, time: 2.58
Output dim: 8, lower bound: -0.0002202, upper bound: 0.0001827
RS_RSZ2_RSZ2_RSZ1_RSZ2_RSZ2_RSZ2_RSZ2, status: Status.UNKNOWN, split count: 7, time: 2.58
Output dim: 8, lower bound: -0.0002160, upper bound: 0.0001825
RS_RSZ2_RSZ2_RSZ2_RSZ1_RSZ1_RSZ1_RSZ1, status: Status.VERIFIED, split count: 7, time: 2.58
Output dim: 8, lower bound: -0.0001378, upper bound: 0.0002074
RS_RSZ2_RSZ2_RSZ2_RSZ1_RSZ1_RSZ1_RSZ2, status: Status.VERIFIED, split count: 7, time: 2.58
Output dim: 8, lower bound: -0.0001681, upper bound: 0.0001537
RS_RSZ2_RSZ2_RSZ2_RSZ1_RSZ1_RSZ2_RSZ1, status: Status.VERIFIED, split count: 7, time: 2.58
Output dim: 8, lower bound: -0.0001450, upper bound: 0.0001691
RS_RSZ2_RSZ2_RSZ2_RSZ1_RSZ1_RSZ2_RSZ2, status: Status.VERIFIED, split count: 7, time: 2.58
Output dim: 8, lower bound: -0.0002087, upper bound: 0.0001407
RS_RSZ2_RSZ2_RSZ2_RSZ1_RSZ2_RSZ1_RSZ1, status: Status.VERIFIED, split count: 7, time: 2.58
Output dim: 8, lower bound: -0.0001342, upper bound: 0.0002062
RS_RSZ2_RSZ2_RSZ2_RSZ1_RSZ2_RSZ1_RSZ2, status: Status.VERIFIED, split count: 7, time: 2.58
Output dim: 8, lower bound: -0.0001437, upper bound: 0.0001742
RS_RSZ2_RSZ2_RSZ2_RSZ1_RSZ2_RSZ2_RSZ1, status: Status.VERIFIED, split count: 7, time: 2.58
Output dim: 8, lower bound: -0.0001654, upper bound: 0.0001543
RS_RSZ2_RSZ2_RSZ2_RSZ1_RSZ2_RSZ2_RSZ2, status: Status.VERIFIED, split count: 7, time: 2.58
Output dim: 8, lower bound: -0.0002107, upper bound: 0.0001498
RS_RSZ2_RSZ2_RSZ2_RSZ2_RSZ1_RSZ1_RSZ1, status: Status.VERIFIED, split count: 7, time: 2.58
Output dim: 8, lower bound: -0.0001492, upper bound: 0.0002055
RS_RSZ2_RSZ2_RSZ2_RSZ2_RSZ1_RSZ1_RSZ2, status: Status.VERIFIED, split count: 7, time: 2.58
Output dim: 8, lower bound: -0.0001858, upper bound: 0.0001505
RS_RSZ2_RSZ2_RSZ2_RSZ2_RSZ1_RSZ2_RSZ1, status: Status.VERIFIED, split count: 7, time: 2.58
Output dim: 8, lower bound: -0.0001496, upper bound: 0.0001549
RS_RSZ2_RSZ2_RSZ2_RSZ2_RSZ1_RSZ2_RSZ2, status: Status.VERIFIED, split count: 7, time: 2.58
Output dim: 8, lower bound: -0.0002146, upper bound: 0.0001369
RS_RSZ2_RSZ2_RSZ2_RSZ2_RSZ2_RSZ1_RSZ1, status: Status.VERIFIED, split count: 7, time: 2.58
Output dim: 8, lower bound: -0.0001495, upper bound: 0.0001997
RS_RSZ2_RSZ2_RSZ2_RSZ2_RSZ2_RSZ1_RSZ2, status: Status.UNKNOWN, split count: 7, time: 2.58
Output dim: 8, lower bound: -0.0002222, upper bound: 0.0001574
RS_RSZ2_RSZ2_RSZ2_RSZ2_RSZ2_RSZ2_RSZ1, status: Status.VERIFIED, split count: 7, time: 2.58
Output dim: 8, lower bound: -0.0001889, upper bound: 0.0002134
RS_RSZ2_RSZ2_RSZ2_RSZ2_RSZ2_RSZ2_RSZ2, status: Status.UNKNOWN, split count: 7, time: 2.58
Output dim: 8, lower bound: -0.0002224, upper bound: 0.0001739

## BFS RS instance: RS_RSZ1_RSZ1_RSZ1_RSZ1_RSZ1_RSZ2_RSZ2

### Backsubstitution after applying RS history:
0: 0.0035972, 0.0041150, 0.0035972, 0.0041150, -0.0001280, 0.0001327
1: 0.0018420, 0.0019168, 0.0018420, 0.0019168, -0.0000185, 0.0000192
2: 0.0120848, 0.0123711, 0.0120848, 0.0123711, -0.0000734, 0.0000708
3: -0.0021818, -0.0018857, -0.0021818, -0.0018857, -0.0000759, 0.0000732
4: -0.0019956, -0.0016750, -0.0019956, -0.0016750, -0.0000792, 0.0000822
5: 0.0056985, 0.0060018, 0.0056985, 0.0060018, -0.0000777, 0.0000750
6: 0.0003095, 0.0015131, 0.0003095, 0.0015131, -0.0003085, 0.0002974
7: -0.0046174, -0.0029782, -0.0046174, -0.0029782, -0.0004051, 0.0004201
8: 0.9859612, 0.9871160, 0.9859612, 0.9871160, -0.0002854, 0.0002959
9: -0.0041920, -0.0031438, -0.0041920, -0.0031438, -0.0002686, 0.0002590

### Unstable ReLU Count (Linear/Conv2D Layers)
- layer_idx=0, type=LayerType.Linear, total=256, inp1_unstable=4, inp2_unstable=4, delta_unstable=252
- layer_idx=2, type=LayerType.Linear, total=256, inp1_unstable=4, inp2_unstable=4, delta_unstable=255
- layer_idx=4, type=LayerType.Linear, total=256, inp1_unstable=0, inp2_unstable=0, delta_unstable=256
- layer_idx=6, type=LayerType.Linear, total=256, inp1_unstable=0, inp2_unstable=0, delta_unstable=256
- layer_idx=8, type=LayerType.Linear, total=10, inp1_unstable=0, inp2_unstable=0, delta_unstable=10

Time for backsubstitution: 1.46 seconds

### RS candidates at layer 3
type: RSZ, layer: 3, pos: 83
type: RSZ, layer: 3, pos: 49
type: RSZ, layer: 3, pos: 34

Time for candidate selection: 0.00 seconds

### Candidate
type: RSZ, layer: 3, pos: 83

### Candidate
type: RSZ, layer: 3, pos: 49

### Relational analysis RSZ of RS_RSZ1_RSZ1_RSZ1_RSZ1_RSZ1_RSZ2_RSZ2_RSZ1

#### Relational analysis RSZ result of RS_RSZ1_RSZ1_RSZ1_RSZ1_RSZ1_RSZ2_RSZ2_RSZ1
Status: Status.VERIFIED
Output dim: 8, lower bound: -0.0001380, upper bound: 0.0001741
time: 0.56 seconds

### Relational analysis RSZ of RS_RSZ1_RSZ1_RSZ1_RSZ1_RSZ1_RSZ2_RSZ2_RSZ2

#### Relational analysis RSZ result of RS_RSZ1_RSZ1_RSZ1_RSZ1_RSZ1_RSZ2_RSZ2_RSZ2
Status: Status.VERIFIED
Output dim: 8, lower bound: -0.0001932, upper bound: 0.0001374
time: 0.52 seconds

## BFS RS instance: RS_RSZ1_RSZ1_RSZ1_RSZ2_RSZ1_RSZ1_RSZ1

### Backsubstitution after applying RS history:
0: 0.0035972, 0.0041150, 0.0035972, 0.0041150, -0.0001265, 0.0001302
1: 0.0018420, 0.0019168, 0.0018420, 0.0019168, -0.0000183, 0.0000188
2: 0.0120848, 0.0123711, 0.0120848, 0.0123711, -0.0000720, 0.0000699
3: -0.0021818, -0.0018857, -0.0021818, -0.0018857, -0.0000745, 0.0000723
4: -0.0019956, -0.0016750, -0.0019956, -0.0016750, -0.0000783, 0.0000806
5: 0.0056985, 0.0060018, 0.0056985, 0.0060018, -0.0000763, 0.0000741
6: 0.0003095, 0.0015131, 0.0003095, 0.0015131, -0.0003027, 0.0002940
7: -0.0046174, -0.0029782, -0.0046174, -0.0029782, -0.0004005, 0.0004123
8: 0.9859612, 0.9871160, 0.9859612, 0.9871160, -0.0002821, 0.0002904
9: -0.0041920, -0.0031438, -0.0041920, -0.0031438, -0.0002636, 0.0002561

### Unstable ReLU Count (Linear/Conv2D Layers)
- layer_idx=0, type=LayerType.Linear, total=256, inp1_unstable=4, inp2_unstable=4, delta_unstable=252
- layer_idx=2, type=LayerType.Linear, total=256, inp1_unstable=4, inp2_unstable=4, delta_unstable=255
- layer_idx=4, type=LayerType.Linear, total=256, inp1_unstable=0, inp2_unstable=0, delta_unstable=256
- layer_idx=6, type=LayerType.Linear, total=256, inp1_unstable=0, inp2_unstable=0, delta_unstable=256
- layer_idx=8, type=LayerType.Linear, total=10, inp1_unstable=0, inp2_unstable=0, delta_unstable=10

Time for backsubstitution: 1.48 seconds

### RS candidates at layer 3
type: RSZ, layer: 3, pos: 240
type: RSZ, layer: 3, pos: 49
type: RSZ, layer: 3, pos: 34

Time for candidate selection: 0.00 seconds

### Candidate
type: RSZ, layer: 3, pos: 240

### Candidate
type: RSZ, layer: 3, pos: 49

### Relational analysis RSZ of RS_RSZ1_RSZ1_RSZ1_RSZ2_RSZ1_RSZ1_RSZ1_RSZ1

#### Relational analysis RSZ result of RS_RSZ1_RSZ1_RSZ1_RSZ2_RSZ1_RSZ1_RSZ1_RSZ1
Status: Status.VERIFIED
Output dim: 8, lower bound: -0.0001352, upper bound: 0.0001936
time: 0.56 seconds

### Relational analysis RSZ of RS_RSZ1_RSZ1_RSZ1_RSZ2_RSZ1_RSZ1_RSZ1_RSZ2

#### Relational analysis RSZ result of RS_RSZ1_RSZ1_RSZ1_RSZ2_RSZ1_RSZ1_RSZ1_RSZ2
Status: Status.VERIFIED
Output dim: 8, lower bound: -0.0001621, upper bound: 0.0001285
time: 0.55 seconds

## BFS RS instance: RS_RSZ1_RSZ1_RSZ1_RSZ2_RSZ1_RSZ1_RSZ2

### Backsubstitution after applying RS history:
0: 0.0035972, 0.0041150, 0.0035972, 0.0041150, -0.0001300, 0.0001261
1: 0.0018420, 0.0019168, 0.0018420, 0.0019168, -0.0000188, 0.0000182
2: 0.0120848, 0.0123711, 0.0120848, 0.0123711, -0.0000697, 0.0000719
3: -0.0021818, -0.0018857, -0.0021818, -0.0018857, -0.0000721, 0.0000743
4: -0.0019956, -0.0016750, -0.0019956, -0.0016750, -0.0000805, 0.0000780
5: 0.0056985, 0.0060018, 0.0056985, 0.0060018, -0.0000738, 0.0000761
6: 0.0003095, 0.0015131, 0.0003095, 0.0015131, -0.0002930, 0.0003021
7: -0.0046174, -0.0029782, -0.0046174, -0.0029782, -0.0004114, 0.0003990
8: 0.9859612, 0.9871160, 0.9859612, 0.9871160, -0.0002898, 0.0002811
9: -0.0041920, -0.0031438, -0.0041920, -0.0031438, -0.0002551, 0.0002631

### Unstable ReLU Count (Linear/Conv2D Layers)
- layer_idx=0, type=LayerType.Linear, total=256, inp1_unstable=4, inp2_unstable=4, delta_unstable=252
- layer_idx=2, type=LayerType.Linear, total=256, inp1_unstable=4, inp2_unstable=4, delta_unstable=255
- layer_idx=4, type=LayerType.Linear, total=256, inp1_unstable=0, inp2_unstable=0, delta_unstable=256
- layer_idx=6, type=LayerType.Linear, total=256, inp1_unstable=0, inp2_unstable=0, delta_unstable=256
- layer_idx=8, type=LayerType.Linear, total=10, inp1_unstable=0, inp2_unstable=0, delta_unstable=10

Time for backsubstitution: 1.54 seconds

### RS candidates at layer 3
type: RSZ, layer: 3, pos: 240
type: RSZ, layer: 3, pos: 34
type: RSZ, layer: 3, pos: 49

Time for candidate selection: 0.00 seconds

### Candidate
type: RSZ, layer: 3, pos: 240

### Candidate
type: RSZ, layer: 3, pos: 34

### Candidate
type: RSZ, layer: 3, pos: 49

### Relational analysis RSZ of RS_RSZ1_RSZ1_RSZ1_RSZ2_RSZ1_RSZ1_RSZ2_RSZ1

#### Relational analysis RSZ result of RS_RSZ1_RSZ1_RSZ1_RSZ2_RSZ1_RSZ1_RSZ2_RSZ1
Status: Status.VERIFIED
Output dim: 8, lower bound: -0.0001409, upper bound: 0.0001532
time: 0.56 seconds

### Relational analysis RSZ of RS_RSZ1_RSZ1_RSZ1_RSZ2_RSZ1_RSZ1_RSZ2_RSZ2

#### Relational analysis RSZ result of RS_RSZ1_RSZ1_RSZ1_RSZ2_RSZ1_RSZ1_RSZ2_RSZ2
Status: Status.VERIFIED
Output dim: 8, lower bound: -0.0001942, upper bound: 0.0001213
time: 0.55 seconds

## BFS RS instance: RS_RSZ1_RSZ1_RSZ1_RSZ2_RSZ1_RSZ2_RSZ2

### Backsubstitution after applying RS history:
0: 0.0035972, 0.0041150, 0.0035972, 0.0041150, -0.0001264, 0.0001172
1: 0.0018420, 0.0019168, 0.0018420, 0.0019168, -0.0000183, 0.0000169
2: 0.0120848, 0.0123711, 0.0120848, 0.0123711, -0.0000648, 0.0000699
3: -0.0021818, -0.0018857, -0.0021818, -0.0018857, -0.0000670, 0.0000723
4: -0.0019956, -0.0016750, -0.0019956, -0.0016750, -0.0000782, 0.0000726
5: 0.0056985, 0.0060018, 0.0056985, 0.0060018, -0.0000687, 0.0000740
6: 0.0003095, 0.0015131, 0.0003095, 0.0015131, -0.0002724, 0.0002938
7: -0.0046174, -0.0029782, -0.0046174, -0.0029782, -0.0004001, 0.0003710
8: 0.9859612, 0.9871160, 0.9859612, 0.9871160, -0.0002818, 0.0002614
9: -0.0041920, -0.0031438, -0.0041920, -0.0031438, -0.0002373, 0.0002558

### Unstable ReLU Count (Linear/Conv2D Layers)
- layer_idx=0, type=LayerType.Linear, total=256, inp1_unstable=4, inp2_unstable=4, delta_unstable=252
- layer_idx=2, type=LayerType.Linear, total=256, inp1_unstable=4, inp2_unstable=4, delta_unstable=255
- layer_idx=4, type=LayerType.Linear, total=256, inp1_unstable=0, inp2_unstable=0, delta_unstable=256
- layer_idx=6, type=LayerType.Linear, total=256, inp1_unstable=0, inp2_unstable=0, delta_unstable=256
- layer_idx=8, type=LayerType.Linear, total=10, inp1_unstable=0, inp2_unstable=0, delta_unstable=10

Time for backsubstitution: 1.52 seconds

### RS candidates at layer 3
type: RSZ, layer: 3, pos: 83
type: RSZ, layer: 3, pos: 34
type: RSZ, layer: 3, pos: 240

Time for candidate selection: 0.00 seconds

### Candidate
type: RSZ, layer: 3, pos: 83

### Relational analysis RSZ of RS_RSZ1_RSZ1_RSZ1_RSZ2_RSZ1_RSZ2_RSZ2_RSZ1

#### Relational analysis RSZ result of RS_RSZ1_RSZ1_RSZ1_RSZ2_RSZ1_RSZ2_RSZ2_RSZ1
Status: Status.VERIFIED
Output dim: 8, lower bound: -0.0001570, upper bound: 0.0001322
time: 0.51 seconds

### Relational analysis RSZ of RS_RSZ1_RSZ1_RSZ1_RSZ2_RSZ1_RSZ2_RSZ2_RSZ2

#### Relational analysis RSZ result of RS_RSZ1_RSZ1_RSZ1_RSZ2_RSZ1_RSZ2_RSZ2_RSZ2
Status: Status.VERIFIED
Output dim: 8, lower bound: -0.0001952, upper bound: 0.0001259
time: 0.50 seconds

## BFS RS instance: RS_RSZ1_RSZ1_RSZ2_RSZ2_RSZ2_RSZ2_RSZ1

### Backsubstitution after applying RS history:
0: 0.0035972, 0.0041150, 0.0035972, 0.0041150, -0.0001197, 0.0001164
1: 0.0018420, 0.0019168, 0.0018420, 0.0019168, -0.0000173, 0.0000168
2: 0.0120848, 0.0123711, 0.0120848, 0.0123711, -0.0000644, 0.0000662
3: -0.0021818, -0.0018857, -0.0021818, -0.0018857, -0.0000666, 0.0000684
4: -0.0019956, -0.0016750, -0.0019956, -0.0016750, -0.0000741, 0.0000721
5: 0.0056985, 0.0060018, 0.0056985, 0.0060018, -0.0000682, 0.0000701
6: 0.0003095, 0.0015131, 0.0003095, 0.0015131, -0.0002706, 0.0002781
7: -0.0046174, -0.0029782, -0.0046174, -0.0029782, -0.0003788, 0.0003685
8: 0.9859612, 0.9871160, 0.9859612, 0.9871160, -0.0002668, 0.0002596
9: -0.0041920, -0.0031438, -0.0041920, -0.0031438, -0.0002356, 0.0002422

### Unstable ReLU Count (Linear/Conv2D Layers)
- layer_idx=0, type=LayerType.Linear, total=256, inp1_unstable=4, inp2_unstable=4, delta_unstable=252
- layer_idx=2, type=LayerType.Linear, total=256, inp1_unstable=4, inp2_unstable=4, delta_unstable=255
- layer_idx=4, type=LayerType.Linear, total=256, inp1_unstable=0, inp2_unstable=0, delta_unstable=256
- layer_idx=6, type=LayerType.Linear, total=256, inp1_unstable=0, inp2_unstable=0, delta_unstable=256
- layer_idx=8, type=LayerType.Linear, total=10, inp1_unstable=0, inp2_unstable=0, delta_unstable=10

Time for backsubstitution: 1.47 seconds

### RS candidates at layer 3
type: RSZ, layer: 3, pos: 83
type: RSZ, layer: 3, pos: 49
type: RSZ, layer: 3, pos: 240

Time for candidate selection: 0.00 seconds

### Candidate
type: RSZ, layer: 3, pos: 83

### Relational analysis RSZ of RS_RSZ1_RSZ1_RSZ2_RSZ2_RSZ2_RSZ2_RSZ1_RSZ1

#### Relational analysis RSZ result of RS_RSZ1_RSZ1_RSZ2_RSZ2_RSZ2_RSZ2_RSZ1_RSZ1
Status: Status.VERIFIED
Output dim: 8, lower bound: -0.0001670, upper bound: 0.0001326
time: 0.54 seconds

### Relational analysis RSZ of RS_RSZ1_RSZ1_RSZ2_RSZ2_RSZ2_RSZ2_RSZ1_RSZ2

#### Relational analysis RSZ result of RS_RSZ1_RSZ1_RSZ2_RSZ2_RSZ2_RSZ2_RSZ1_RSZ2
Status: Status.VERIFIED
Output dim: 8, lower bound: -0.0002027, upper bound: 0.0001240
time: 0.58 seconds

## BFS RS instance: RS_RSZ1_RSZ1_RSZ2_RSZ2_RSZ2_RSZ2_RSZ2

### Backsubstitution after applying RS history:
0: 0.0035972, 0.0041150, 0.0035972, 0.0041150, -0.0001189, 0.0001175
1: 0.0018420, 0.0019168, 0.0018420, 0.0019168, -0.0000172, 0.0000170
2: 0.0120848, 0.0123711, 0.0120848, 0.0123711, -0.0000650, 0.0000657
3: -0.0021818, -0.0018857, -0.0021818, -0.0018857, -0.0000672, 0.0000680
4: -0.0019956, -0.0016750, -0.0019956, -0.0016750, -0.0000736, 0.0000727
5: 0.0056985, 0.0060018, 0.0056985, 0.0060018, -0.0000688, 0.0000697
6: 0.0003095, 0.0015131, 0.0003095, 0.0015131, -0.0002731, 0.0002764
7: -0.0046174, -0.0029782, -0.0046174, -0.0029782, -0.0003764, 0.0003720
8: 0.9859612, 0.9871160, 0.9859612, 0.9871160, -0.0002652, 0.0002620
9: -0.0041920, -0.0031438, -0.0041920, -0.0031438, -0.0002378, 0.0002407

### Unstable ReLU Count (Linear/Conv2D Layers)
- layer_idx=0, type=LayerType.Linear, total=256, inp1_unstable=4, inp2_unstable=4, delta_unstable=252
- layer_idx=2, type=LayerType.Linear, total=256, inp1_unstable=4, inp2_unstable=4, delta_unstable=255
- layer_idx=4, type=LayerType.Linear, total=256, inp1_unstable=0, inp2_unstable=0, delta_unstable=256
- layer_idx=6, type=LayerType.Linear, total=256, inp1_unstable=0, inp2_unstable=0, delta_unstable=256
- layer_idx=8, type=LayerType.Linear, total=10, inp1_unstable=0, inp2_unstable=0, delta_unstable=10

Time for backsubstitution: 1.50 seconds

### RS candidates at layer 3
type: RSZ, layer: 3, pos: 83
type: RSZ, layer: 3, pos: 240
type: RSZ, layer: 3, pos: 49

Time for candidate selection: 0.00 seconds

### Candidate
type: RSZ, layer: 3, pos: 83

### Relational analysis RSZ of RS_RSZ1_RSZ1_RSZ2_RSZ2_RSZ2_RSZ2_RSZ2_RSZ1

#### Relational analysis RSZ result of RS_RSZ1_RSZ1_RSZ2_RSZ2_RSZ2_RSZ2_RSZ2_RSZ1
Status: Status.VERIFIED
Output dim: 8, lower bound: -0.0001667, upper bound: 0.0001331
time: 0.55 seconds

### Relational analysis RSZ of RS_RSZ1_RSZ1_RSZ2_RSZ2_RSZ2_RSZ2_RSZ2_RSZ2

#### Relational analysis RSZ result of RS_RSZ1_RSZ1_RSZ2_RSZ2_RSZ2_RSZ2_RSZ2_RSZ2
Status: Status.VERIFIED
Output dim: 8, lower bound: -0.0001982, upper bound: 0.0001240
time: 0.50 seconds

## BFS RS instance: RS_RSZ1_RSZ2_RSZ1_RSZ1_RSZ1_RSZ1_RSZ1

### Backsubstitution after applying RS history:
0: 0.0035972, 0.0041150, 0.0035972, 0.0041150, -0.0001151, 0.0001203
1: 0.0018420, 0.0019168, 0.0018420, 0.0019168, -0.0000166, 0.0000174
2: 0.0120848, 0.0123711, 0.0120848, 0.0123711, -0.0000665, 0.0000637
3: -0.0021818, -0.0018857, -0.0021818, -0.0018857, -0.0000688, 0.0000658
4: -0.0019956, -0.0016750, -0.0019956, -0.0016750, -0.0000713, 0.0000745
5: 0.0056985, 0.0060018, 0.0056985, 0.0060018, -0.0000705, 0.0000675
6: 0.0003095, 0.0015131, 0.0003095, 0.0015131, -0.0002797, 0.0002676
7: -0.0046174, -0.0029782, -0.0046174, -0.0029782, -0.0003645, 0.0003810
8: 0.9859612, 0.9871160, 0.9859612, 0.9871160, -0.0002568, 0.0002684
9: -0.0041920, -0.0031438, -0.0041920, -0.0031438, -0.0002436, 0.0002331

### Unstable ReLU Count (Linear/Conv2D Layers)
- layer_idx=0, type=LayerType.Linear, total=256, inp1_unstable=4, inp2_unstable=4, delta_unstable=252
- layer_idx=2, type=LayerType.Linear, total=256, inp1_unstable=4, inp2_unstable=4, delta_unstable=255
- layer_idx=4, type=LayerType.Linear, total=256, inp1_unstable=0, inp2_unstable=0, delta_unstable=256
- layer_idx=6, type=LayerType.Linear, total=256, inp1_unstable=0, inp2_unstable=0, delta_unstable=256
- layer_idx=8, type=LayerType.Linear, total=10, inp1_unstable=0, inp2_unstable=0, delta_unstable=10

Time for backsubstitution: 1.45 seconds

### RS candidates at layer 3
type: RSZ, layer: 3, pos: 240
type: RSZ, layer: 3, pos: 83
type: RSZ, layer: 3, pos: 49

Time for candidate selection: 0.00 seconds

### Candidate
type: RSZ, layer: 3, pos: 240

### Candidate
type: RSZ, layer: 3, pos: 83

### Candidate
type: RSZ, layer: 3, pos: 49

### Relational analysis RSZ of RS_RSZ1_RSZ2_RSZ1_RSZ1_RSZ1_RSZ1_RSZ1_RSZ1

#### Relational analysis RSZ result of RS_RSZ1_RSZ2_RSZ1_RSZ1_RSZ1_RSZ1_RSZ1_RSZ1
Status: Status.VERIFIED
Output dim: 8, lower bound: -0.0001257, upper bound: 0.0001980
time: 0.54 seconds

### Relational analysis RSZ of RS_RSZ1_RSZ2_RSZ1_RSZ1_RSZ1_RSZ1_RSZ1_RSZ2

#### Relational analysis RSZ result of RS_RSZ1_RSZ2_RSZ1_RSZ1_RSZ1_RSZ1_RSZ1_RSZ2
Status: Status.VERIFIED
Output dim: 8, lower bound: -0.0001400, upper bound: 0.0001276
time: 0.51 seconds

## BFS RS instance: RS_RSZ1_RSZ2_RSZ1_RSZ1_RSZ1_RSZ1_RSZ2

### Backsubstitution after applying RS history:
0: 0.0035972, 0.0041150, 0.0035972, 0.0041150, -0.0001139, 0.0001215
1: 0.0018420, 0.0019168, 0.0018420, 0.0019168, -0.0000165, 0.0000176
2: 0.0120848, 0.0123711, 0.0120848, 0.0123711, -0.0000672, 0.0000630
3: -0.0021818, -0.0018857, -0.0021818, -0.0018857, -0.0000695, 0.0000651
4: -0.0019956, -0.0016750, -0.0019956, -0.0016750, -0.0000705, 0.0000752
5: 0.0056985, 0.0060018, 0.0056985, 0.0060018, -0.0000712, 0.0000667
6: 0.0003095, 0.0015131, 0.0003095, 0.0015131, -0.0002824, 0.0002647
7: -0.0046174, -0.0029782, -0.0046174, -0.0029782, -0.0003604, 0.0003846
8: 0.9859612, 0.9871160, 0.9859612, 0.9871160, -0.0002539, 0.0002709
9: -0.0041920, -0.0031438, -0.0041920, -0.0031438, -0.0002459, 0.0002305

### Unstable ReLU Count (Linear/Conv2D Layers)
- layer_idx=0, type=LayerType.Linear, total=256, inp1_unstable=4, inp2_unstable=4, delta_unstable=252
- layer_idx=2, type=LayerType.Linear, total=256, inp1_unstable=4, inp2_unstable=4, delta_unstable=255
- layer_idx=4, type=LayerType.Linear, total=256, inp1_unstable=0, inp2_unstable=0, delta_unstable=256
- layer_idx=6, type=LayerType.Linear, total=256, inp1_unstable=0, inp2_unstable=0, delta_unstable=256
- layer_idx=8, type=LayerType.Linear, total=10, inp1_unstable=0, inp2_unstable=0, delta_unstable=10

Time for backsubstitution: 1.49 seconds

### RS candidates at layer 3
type: RSZ, layer: 3, pos: 240
type: RSZ, layer: 3, pos: 83
type: RSZ, layer: 3, pos: 49

Time for candidate selection: 0.00 seconds

### Candidate
type: RSZ, layer: 3, pos: 240

### Candidate
type: RSZ, layer: 3, pos: 83

### Candidate
type: RSZ, layer: 3, pos: 49

### Relational analysis RSZ of RS_RSZ1_RSZ2_RSZ1_RSZ1_RSZ1_RSZ1_RSZ2_RSZ1

#### Relational analysis RSZ result of RS_RSZ1_RSZ2_RSZ1_RSZ1_RSZ1_RSZ1_RSZ2_RSZ1
Status: Status.VERIFIED
Output dim: 8, lower bound: -0.0001269, upper bound: 0.0002027
time: 0.53 seconds

### Relational analysis RSZ of RS_RSZ1_RSZ2_RSZ1_RSZ1_RSZ1_RSZ1_RSZ2_RSZ2

#### Relational analysis RSZ result of RS_RSZ1_RSZ2_RSZ1_RSZ1_RSZ1_RSZ1_RSZ2_RSZ2
Status: Status.VERIFIED
Output dim: 8, lower bound: -0.0001412, upper bound: 0.0001295
time: 0.51 seconds

## BFS RS instance: RS_RSZ1_RSZ2_RSZ1_RSZ1_RSZ1_RSZ2_RSZ1

### Backsubstitution after applying RS history:
0: 0.0035972, 0.0041150, 0.0035972, 0.0041150, -0.0001151, 0.0001203
1: 0.0018420, 0.0019168, 0.0018420, 0.0019168, -0.0000166, 0.0000174
2: 0.0120848, 0.0123711, 0.0120848, 0.0123711, -0.0000665, 0.0000637
3: -0.0021818, -0.0018857, -0.0021818, -0.0018857, -0.0000688, 0.0000658
4: -0.0019956, -0.0016750, -0.0019956, -0.0016750, -0.0000713, 0.0000745
5: 0.0056985, 0.0060018, 0.0056985, 0.0060018, -0.0000705, 0.0000675
6: 0.0003095, 0.0015131, 0.0003095, 0.0015131, -0.0002797, 0.0002676
7: -0.0046174, -0.0029782, -0.0046174, -0.0029782, -0.0003645, 0.0003810
8: 0.9859612, 0.9871160, 0.9859612, 0.9871160, -0.0002568, 0.0002684
9: -0.0041920, -0.0031438, -0.0041920, -0.0031438, -0.0002436, 0.0002331

### Unstable ReLU Count (Linear/Conv2D Layers)
- layer_idx=0, type=LayerType.Linear, total=256, inp1_unstable=4, inp2_unstable=4, delta_unstable=252
- layer_idx=2, type=LayerType.Linear, total=256, inp1_unstable=4, inp2_unstable=4, delta_unstable=255
- layer_idx=4, type=LayerType.Linear, total=256, inp1_unstable=0, inp2_unstable=0, delta_unstable=256
- layer_idx=6, type=LayerType.Linear, total=256, inp1_unstable=0, inp2_unstable=0, delta_unstable=256
- layer_idx=8, type=LayerType.Linear, total=10, inp1_unstable=0, inp2_unstable=0, delta_unstable=10

Time for backsubstitution: 1.52 seconds

### RS candidates at layer 3
type: RSZ, layer: 3, pos: 83
type: RSZ, layer: 3, pos: 240
type: RSZ, layer: 3, pos: 49

Time for candidate selection: 0.00 seconds

### Candidate
type: RSZ, layer: 3, pos: 83

### Candidate
type: RSZ, layer: 3, pos: 240

### Candidate
type: RSZ, layer: 3, pos: 49

### Relational analysis RSZ of RS_RSZ1_RSZ2_RSZ1_RSZ1_RSZ1_RSZ2_RSZ1_RSZ1

#### Relational analysis RSZ result of RS_RSZ1_RSZ2_RSZ1_RSZ1_RSZ1_RSZ2_RSZ1_RSZ1
Status: Status.VERIFIED
Output dim: 8, lower bound: -0.0001234, upper bound: 0.0001981
time: 0.55 seconds

### Relational analysis RSZ of RS_RSZ1_RSZ2_RSZ1_RSZ1_RSZ1_RSZ2_RSZ1_RSZ2

#### Relational analysis RSZ result of RS_RSZ1_RSZ2_RSZ1_RSZ1_RSZ1_RSZ2_RSZ1_RSZ2
Status: Status.VERIFIED
Output dim: 8, lower bound: -0.0001379, upper bound: 0.0001321
time: 0.55 seconds

## BFS RS instance: RS_RSZ1_RSZ2_RSZ1_RSZ1_RSZ1_RSZ2_RSZ2

### Backsubstitution after applying RS history:
0: 0.0035972, 0.0041150, 0.0035972, 0.0041150, -0.0001139, 0.0001215
1: 0.0018420, 0.0019168, 0.0018420, 0.0019168, -0.0000165, 0.0000176
2: 0.0120848, 0.0123711, 0.0120848, 0.0123711, -0.0000672, 0.0000630
3: -0.0021818, -0.0018857, -0.0021818, -0.0018857, -0.0000695, 0.0000651
4: -0.0019956, -0.0016750, -0.0019956, -0.0016750, -0.0000705, 0.0000752
5: 0.0056985, 0.0060018, 0.0056985, 0.0060018, -0.0000712, 0.0000667
6: 0.0003095, 0.0015131, 0.0003095, 0.0015131, -0.0002824, 0.0002647
7: -0.0046174, -0.0029782, -0.0046174, -0.0029782, -0.0003604, 0.0003846
8: 0.9859612, 0.9871160, 0.9859612, 0.9871160, -0.0002539, 0.0002709
9: -0.0041920, -0.0031438, -0.0041920, -0.0031438, -0.0002459, 0.0002305

### Unstable ReLU Count (Linear/Conv2D Layers)
- layer_idx=0, type=LayerType.Linear, total=256, inp1_unstable=4, inp2_unstable=4, delta_unstable=252
- layer_idx=2, type=LayerType.Linear, total=256, inp1_unstable=4, inp2_unstable=4, delta_unstable=255
- layer_idx=4, type=LayerType.Linear, total=256, inp1_unstable=0, inp2_unstable=0, delta_unstable=256
- layer_idx=6, type=LayerType.Linear, total=256, inp1_unstable=0, inp2_unstable=0, delta_unstable=256
- layer_idx=8, type=LayerType.Linear, total=10, inp1_unstable=0, inp2_unstable=0, delta_unstable=10

Time for backsubstitution: 1.44 seconds

### RS candidates at layer 3
type: RSZ, layer: 3, pos: 240
type: RSZ, layer: 3, pos: 83
type: RSZ, layer: 3, pos: 49

Time for candidate selection: 0.00 seconds

### Candidate
type: RSZ, layer: 3, pos: 240

### Candidate
type: RSZ, layer: 3, pos: 83

### Candidate
type: RSZ, layer: 3, pos: 49

### Relational analysis RSZ of RS_RSZ1_RSZ2_RSZ1_RSZ1_RSZ1_RSZ2_RSZ2_RSZ1

#### Relational analysis RSZ result of RS_RSZ1_RSZ2_RSZ1_RSZ1_RSZ1_RSZ2_RSZ2_RSZ1
Status: Status.VERIFIED
Output dim: 8, lower bound: -0.0001234, upper bound: 0.0002023
time: 0.55 seconds

### Relational analysis RSZ of RS_RSZ1_RSZ2_RSZ1_RSZ1_RSZ1_RSZ2_RSZ2_RSZ2

#### Relational analysis RSZ result of RS_RSZ1_RSZ2_RSZ1_RSZ1_RSZ1_RSZ2_RSZ2_RSZ2
Status: Status.VERIFIED
Output dim: 8, lower bound: -0.0001379, upper bound: 0.0001321
time: 0.54 seconds

## BFS RS instance: RS_RSZ1_RSZ2_RSZ1_RSZ2_RSZ2_RSZ1_RSZ1

### Backsubstitution after applying RS history:
0: 0.0035972, 0.0041150, 0.0035972, 0.0041150, -0.0001270, 0.0001306
1: 0.0018420, 0.0019168, 0.0018420, 0.0019168, -0.0000183, 0.0000189
2: 0.0120848, 0.0123711, 0.0120848, 0.0123711, -0.0000722, 0.0000702
3: -0.0021818, -0.0018857, -0.0021818, -0.0018857, -0.0000747, 0.0000726
4: -0.0019956, -0.0016750, -0.0019956, -0.0016750, -0.0000786, 0.0000808
5: 0.0056985, 0.0060018, 0.0056985, 0.0060018, -0.0000765, 0.0000744
6: 0.0003095, 0.0015131, 0.0003095, 0.0015131, -0.0003036, 0.0002951
7: -0.0046174, -0.0029782, -0.0046174, -0.0029782, -0.0004020, 0.0004134
8: 0.9859612, 0.9871160, 0.9859612, 0.9871160, -0.0002831, 0.0002912
9: -0.0041920, -0.0031438, -0.0041920, -0.0031438, -0.0002643, 0.0002570

### Unstable ReLU Count (Linear/Conv2D Layers)
- layer_idx=0, type=LayerType.Linear, total=256, inp1_unstable=4, inp2_unstable=4, delta_unstable=252
- layer_idx=2, type=LayerType.Linear, total=256, inp1_unstable=4, inp2_unstable=4, delta_unstable=255
- layer_idx=4, type=LayerType.Linear, total=256, inp1_unstable=0, inp2_unstable=0, delta_unstable=256
- layer_idx=6, type=LayerType.Linear, total=256, inp1_unstable=0, inp2_unstable=0, delta_unstable=256
- layer_idx=8, type=LayerType.Linear, total=10, inp1_unstable=0, inp2_unstable=0, delta_unstable=10

Time for backsubstitution: 1.47 seconds

### RS candidates at layer 3
type: RSZ, layer: 3, pos: 49
type: RSZ, layer: 3, pos: 83
type: RSZ, layer: 3, pos: 34

Time for candidate selection: 0.00 seconds

### Candidate
type: RSZ, layer: 3, pos: 49

### Relational analysis RSZ of RS_RSZ1_RSZ2_RSZ1_RSZ2_RSZ2_RSZ1_RSZ1_RSZ1

#### Relational analysis RSZ result of RS_RSZ1_RSZ2_RSZ1_RSZ2_RSZ2_RSZ1_RSZ1_RSZ1
Status: Status.VERIFIED
Output dim: 8, lower bound: -0.0001400, upper bound: 0.0001563
time: 0.52 seconds

### Relational analysis RSZ of RS_RSZ1_RSZ2_RSZ1_RSZ2_RSZ2_RSZ1_RSZ1_RSZ2

#### Relational analysis RSZ result of RS_RSZ1_RSZ2_RSZ1_RSZ2_RSZ2_RSZ1_RSZ1_RSZ2
Status: Status.VERIFIED
Output dim: 8, lower bound: -0.0001967, upper bound: 0.0001231
time: 0.50 seconds

## BFS RS instance: RS_RSZ1_RSZ2_RSZ1_RSZ2_RSZ2_RSZ1_RSZ2

### Backsubstitution after applying RS history:
0: 0.0035972, 0.0041150, 0.0035972, 0.0041150, -0.0001273, 0.0001305
1: 0.0018420, 0.0019168, 0.0018420, 0.0019168, -0.0000184, 0.0000189
2: 0.0120848, 0.0123711, 0.0120848, 0.0123711, -0.0000721, 0.0000704
3: -0.0021818, -0.0018857, -0.0021818, -0.0018857, -0.0000746, 0.0000728
4: -0.0019956, -0.0016750, -0.0019956, -0.0016750, -0.0000788, 0.0000808
5: 0.0056985, 0.0060018, 0.0056985, 0.0060018, -0.0000764, 0.0000746
6: 0.0003095, 0.0015131, 0.0003095, 0.0015131, -0.0003033, 0.0002959
7: -0.0046174, -0.0029782, -0.0046174, -0.0029782, -0.0004030, 0.0004131
8: 0.9859612, 0.9871160, 0.9859612, 0.9871160, -0.0002839, 0.0002910
9: -0.0041920, -0.0031438, -0.0041920, -0.0031438, -0.0002641, 0.0002577

### Unstable ReLU Count (Linear/Conv2D Layers)
- layer_idx=0, type=LayerType.Linear, total=256, inp1_unstable=4, inp2_unstable=4, delta_unstable=252
- layer_idx=2, type=LayerType.Linear, total=256, inp1_unstable=4, inp2_unstable=4, delta_unstable=255
- layer_idx=4, type=LayerType.Linear, total=256, inp1_unstable=0, inp2_unstable=0, delta_unstable=256
- layer_idx=6, type=LayerType.Linear, total=256, inp1_unstable=0, inp2_unstable=0, delta_unstable=256
- layer_idx=8, type=LayerType.Linear, total=10, inp1_unstable=0, inp2_unstable=0, delta_unstable=10

Time for backsubstitution: 1.42 seconds

### RS candidates at layer 3
type: RSZ, layer: 3, pos: 49
type: RSZ, layer: 3, pos: 83
type: RSZ, layer: 3, pos: 34

Time for candidate selection: 0.00 seconds

### Candidate
type: RSZ, layer: 3, pos: 49

### Relational analysis RSZ of RS_RSZ1_RSZ2_RSZ1_RSZ2_RSZ2_RSZ1_RSZ2_RSZ1

#### Relational analysis RSZ result of RS_RSZ1_RSZ2_RSZ1_RSZ2_RSZ2_RSZ1_RSZ2_RSZ1
Status: Status.VERIFIED
Output dim: 8, lower bound: -0.0001396, upper bound: 0.0001621
time: 0.51 seconds

### Relational analysis RSZ of RS_RSZ1_RSZ2_RSZ1_RSZ2_RSZ2_RSZ1_RSZ2_RSZ2

#### Relational analysis RSZ result of RS_RSZ1_RSZ2_RSZ1_RSZ2_RSZ2_RSZ1_RSZ2_RSZ2
Status: Status.VERIFIED
Output dim: 8, lower bound: -0.0001985, upper bound: 0.0001315
time: 0.49 seconds

## BFS RS instance: RS_RSZ1_RSZ2_RSZ2_RSZ1_RSZ1_RSZ2_RSZ1

### Backsubstitution after applying RS history:
0: 0.0035972, 0.0041150, 0.0035972, 0.0041150, -0.0001185, 0.0001173
1: 0.0018420, 0.0019168, 0.0018420, 0.0019168, -0.0000171, 0.0000170
2: 0.0120848, 0.0123711, 0.0120848, 0.0123711, -0.0000649, 0.0000655
3: -0.0021818, -0.0018857, -0.0021818, -0.0018857, -0.0000671, 0.0000677
4: -0.0019956, -0.0016750, -0.0019956, -0.0016750, -0.0000733, 0.0000726
5: 0.0056985, 0.0060018, 0.0056985, 0.0060018, -0.0000687, 0.0000694
6: 0.0003095, 0.0015131, 0.0003095, 0.0015131, -0.0002727, 0.0002754
7: -0.0046174, -0.0029782, -0.0046174, -0.0029782, -0.0003750, 0.0003715
8: 0.9859612, 0.9871160, 0.9859612, 0.9871160, -0.0002642, 0.0002617
9: -0.0041920, -0.0031438, -0.0041920, -0.0031438, -0.0002375, 0.0002398

### Unstable ReLU Count (Linear/Conv2D Layers)
- layer_idx=0, type=LayerType.Linear, total=256, inp1_unstable=4, inp2_unstable=4, delta_unstable=252
- layer_idx=2, type=LayerType.Linear, total=256, inp1_unstable=4, inp2_unstable=4, delta_unstable=255
- layer_idx=4, type=LayerType.Linear, total=256, inp1_unstable=0, inp2_unstable=0, delta_unstable=256
- layer_idx=6, type=LayerType.Linear, total=256, inp1_unstable=0, inp2_unstable=0, delta_unstable=256
- layer_idx=8, type=LayerType.Linear, total=10, inp1_unstable=0, inp2_unstable=0, delta_unstable=10

Time for backsubstitution: 1.46 seconds

### RS candidates at layer 3
type: RSZ, layer: 3, pos: 83
type: RSZ, layer: 3, pos: 49
type: RSZ, layer: 3, pos: 240

Time for candidate selection: 0.00 seconds

### Candidate
type: RSZ, layer: 3, pos: 83

### Relational analysis RSZ of RS_RSZ1_RSZ2_RSZ2_RSZ1_RSZ1_RSZ2_RSZ1_RSZ1

#### Relational analysis RSZ result of RS_RSZ1_RSZ2_RSZ2_RSZ1_RSZ1_RSZ2_RSZ1_RSZ1
Status: Status.VERIFIED
Output dim: 8, lower bound: -0.0001620, upper bound: 0.0001274
time: 0.52 seconds

### Relational analysis RSZ of RS_RSZ1_RSZ2_RSZ2_RSZ1_RSZ1_RSZ2_RSZ1_RSZ2

#### Relational analysis RSZ result of RS_RSZ1_RSZ2_RSZ2_RSZ1_RSZ1_RSZ2_RSZ1_RSZ2
Status: Status.VERIFIED
Output dim: 8, lower bound: -0.0001939, upper bound: 0.0001213
time: 0.53 seconds

## BFS RS instance: RS_RSZ1_RSZ2_RSZ2_RSZ1_RSZ2_RSZ2_RSZ1

### Backsubstitution after applying RS history:
0: 0.0035972, 0.0041150, 0.0035972, 0.0041150, -0.0001185, 0.0001173
1: 0.0018420, 0.0019168, 0.0018420, 0.0019168, -0.0000171, 0.0000170
2: 0.0120848, 0.0123711, 0.0120848, 0.0123711, -0.0000649, 0.0000655
3: -0.0021818, -0.0018857, -0.0021818, -0.0018857, -0.0000671, 0.0000677
4: -0.0019956, -0.0016750, -0.0019956, -0.0016750, -0.0000733, 0.0000726
5: 0.0056985, 0.0060018, 0.0056985, 0.0060018, -0.0000687, 0.0000694
6: 0.0003095, 0.0015131, 0.0003095, 0.0015131, -0.0002727, 0.0002754
7: -0.0046174, -0.0029782, -0.0046174, -0.0029782, -0.0003750, 0.0003715
8: 0.9859612, 0.9871160, 0.9859612, 0.9871160, -0.0002642, 0.0002617
9: -0.0041920, -0.0031438, -0.0041920, -0.0031438, -0.0002375, 0.0002398

### Unstable ReLU Count (Linear/Conv2D Layers)
- layer_idx=0, type=LayerType.Linear, total=256, inp1_unstable=4, inp2_unstable=4, delta_unstable=252
- layer_idx=2, type=LayerType.Linear, total=256, inp1_unstable=4, inp2_unstable=4, delta_unstable=255
- layer_idx=4, type=LayerType.Linear, total=256, inp1_unstable=0, inp2_unstable=0, delta_unstable=256
- layer_idx=6, type=LayerType.Linear, total=256, inp1_unstable=0, inp2_unstable=0, delta_unstable=256
- layer_idx=8, type=LayerType.Linear, total=10, inp1_unstable=0, inp2_unstable=0, delta_unstable=10

Time for backsubstitution: 1.52 seconds

### RS candidates at layer 3
type: RSZ, layer: 3, pos: 240
type: RSZ, layer: 3, pos: 83
type: RSZ, layer: 3, pos: 49

Time for candidate selection: 0.00 seconds

### Candidate
type: RSZ, layer: 3, pos: 240

### Candidate
type: RSZ, layer: 3, pos: 83

### Relational analysis RSZ of RS_RSZ1_RSZ2_RSZ2_RSZ1_RSZ2_RSZ2_RSZ1_RSZ1

#### Relational analysis RSZ result of RS_RSZ1_RSZ2_RSZ2_RSZ1_RSZ2_RSZ2_RSZ1_RSZ1
Status: Status.VERIFIED
Output dim: 8, lower bound: -0.0001533, upper bound: 0.0001291
time: 0.54 seconds

### Relational analysis RSZ of RS_RSZ1_RSZ2_RSZ2_RSZ1_RSZ2_RSZ2_RSZ1_RSZ2

#### Relational analysis RSZ result of RS_RSZ1_RSZ2_RSZ2_RSZ1_RSZ2_RSZ2_RSZ1_RSZ2
Status: Status.VERIFIED
Output dim: 8, lower bound: -0.0001941, upper bound: 0.0001243
time: 0.51 seconds

## BFS RS instance: RS_RSZ1_RSZ2_RSZ2_RSZ2_RSZ2_RSZ1_RSZ1

### Backsubstitution after applying RS history:
0: 0.0035972, 0.0041150, 0.0035972, 0.0041150, -0.0001201, 0.0001177
1: 0.0018420, 0.0019168, 0.0018420, 0.0019168, -0.0000173, 0.0000170
2: 0.0120848, 0.0123711, 0.0120848, 0.0123711, -0.0000651, 0.0000664
3: -0.0021818, -0.0018857, -0.0021818, -0.0018857, -0.0000673, 0.0000687
4: -0.0019956, -0.0016750, -0.0019956, -0.0016750, -0.0000743, 0.0000728
5: 0.0056985, 0.0060018, 0.0056985, 0.0060018, -0.0000689, 0.0000703
6: 0.0003095, 0.0015131, 0.0003095, 0.0015131, -0.0002735, 0.0002791
7: -0.0046174, -0.0029782, -0.0046174, -0.0029782, -0.0003801, 0.0003725
8: 0.9859612, 0.9871160, 0.9859612, 0.9871160, -0.0002678, 0.0002624
9: -0.0041920, -0.0031438, -0.0041920, -0.0031438, -0.0002382, 0.0002431

### Unstable ReLU Count (Linear/Conv2D Layers)
- layer_idx=0, type=LayerType.Linear, total=256, inp1_unstable=4, inp2_unstable=4, delta_unstable=252
- layer_idx=2, type=LayerType.Linear, total=256, inp1_unstable=4, inp2_unstable=4, delta_unstable=255
- layer_idx=4, type=LayerType.Linear, total=256, inp1_unstable=0, inp2_unstable=0, delta_unstable=256
- layer_idx=6, type=LayerType.Linear, total=256, inp1_unstable=0, inp2_unstable=0, delta_unstable=256
- layer_idx=8, type=LayerType.Linear, total=10, inp1_unstable=0, inp2_unstable=0, delta_unstable=10

Time for backsubstitution: 1.45 seconds

### RS candidates at layer 3
type: RSZ, layer: 3, pos: 49
type: RSZ, layer: 3, pos: 240
type: RSZ, layer: 3, pos: 83

Time for candidate selection: 0.00 seconds

### Candidate
type: RSZ, layer: 3, pos: 49

### Candidate
type: RSZ, layer: 3, pos: 240

### Candidate
type: RSZ, layer: 3, pos: 83

### Relational analysis RSZ of RS_RSZ1_RSZ2_RSZ2_RSZ2_RSZ2_RSZ1_RSZ1_RSZ1

#### Relational analysis RSZ result of RS_RSZ1_RSZ2_RSZ2_RSZ2_RSZ2_RSZ1_RSZ1_RSZ1
Status: Status.VERIFIED
Output dim: 8, lower bound: -0.0001733, upper bound: 0.0001277
time: 0.52 seconds

### Relational analysis RSZ of RS_RSZ1_RSZ2_RSZ2_RSZ2_RSZ2_RSZ1_RSZ1_RSZ2

#### Relational analysis RSZ result of RS_RSZ1_RSZ2_RSZ2_RSZ2_RSZ2_RSZ1_RSZ1_RSZ2
Status: Status.VERIFIED
Output dim: 8, lower bound: -0.0002026, upper bound: 0.0001204
time: 0.52 seconds

## BFS RS instance: RS_RSZ1_RSZ2_RSZ2_RSZ2_RSZ2_RSZ1_RSZ2

### Backsubstitution after applying RS history:
0: 0.0035972, 0.0041150, 0.0035972, 0.0041150, -0.0001196, 0.0001192
1: 0.0018420, 0.0019168, 0.0018420, 0.0019168, -0.0000173, 0.0000172
2: 0.0120848, 0.0123711, 0.0120848, 0.0123711, -0.0000659, 0.0000661
3: -0.0021818, -0.0018857, -0.0021818, -0.0018857, -0.0000681, 0.0000684
4: -0.0019956, -0.0016750, -0.0019956, -0.0016750, -0.0000740, 0.0000738
5: 0.0056985, 0.0060018, 0.0056985, 0.0060018, -0.0000698, 0.0000701
6: 0.0003095, 0.0015131, 0.0003095, 0.0015131, -0.0002770, 0.0002780
7: -0.0046174, -0.0029782, -0.0046174, -0.0029782, -0.0003786, 0.0003772
8: 0.9859612, 0.9871160, 0.9859612, 0.9871160, -0.0002667, 0.0002657
9: -0.0041920, -0.0031438, -0.0041920, -0.0031438, -0.0002412, 0.0002421

### Unstable ReLU Count (Linear/Conv2D Layers)
- layer_idx=0, type=LayerType.Linear, total=256, inp1_unstable=4, inp2_unstable=4, delta_unstable=252
- layer_idx=2, type=LayerType.Linear, total=256, inp1_unstable=4, inp2_unstable=4, delta_unstable=255
- layer_idx=4, type=LayerType.Linear, total=256, inp1_unstable=0, inp2_unstable=0, delta_unstable=256
- layer_idx=6, type=LayerType.Linear, total=256, inp1_unstable=0, inp2_unstable=0, delta_unstable=256
- layer_idx=8, type=LayerType.Linear, total=10, inp1_unstable=0, inp2_unstable=0, delta_unstable=10

Time for backsubstitution: 1.46 seconds

### RS candidates at layer 3
type: RSZ, layer: 3, pos: 83
type: RSZ, layer: 3, pos: 49
type: RSZ, layer: 3, pos: 240

Time for candidate selection: 0.00 seconds

### Candidate
type: RSZ, layer: 3, pos: 83

### Relational analysis RSZ of RS_RSZ1_RSZ2_RSZ2_RSZ2_RSZ2_RSZ1_RSZ2_RSZ1

#### Relational analysis RSZ result of RS_RSZ1_RSZ2_RSZ2_RSZ2_RSZ2_RSZ1_RSZ2_RSZ1
Status: Status.VERIFIED
Output dim: 8, lower bound: -0.0001741, upper bound: 0.0001295
time: 0.51 seconds

### Relational analysis RSZ of RS_RSZ1_RSZ2_RSZ2_RSZ2_RSZ2_RSZ1_RSZ2_RSZ2

#### Relational analysis RSZ result of RS_RSZ1_RSZ2_RSZ2_RSZ2_RSZ2_RSZ1_RSZ2_RSZ2
Status: Status.VERIFIED
Output dim: 8, lower bound: -0.0001991, upper bound: 0.0001216
time: 0.52 seconds

## BFS RS instance: RS_RSZ1_RSZ2_RSZ2_RSZ2_RSZ2_RSZ2_RSZ1

### Backsubstitution after applying RS history:
0: 0.0035972, 0.0041150, 0.0035972, 0.0041150, -0.0001201, 0.0001177
1: 0.0018420, 0.0019168, 0.0018420, 0.0019168, -0.0000173, 0.0000170
2: 0.0120848, 0.0123711, 0.0120848, 0.0123711, -0.0000651, 0.0000664
3: -0.0021818, -0.0018857, -0.0021818, -0.0018857, -0.0000673, 0.0000687
4: -0.0019956, -0.0016750, -0.0019956, -0.0016750, -0.0000743, 0.0000728
5: 0.0056985, 0.0060018, 0.0056985, 0.0060018, -0.0000689, 0.0000703
6: 0.0003095, 0.0015131, 0.0003095, 0.0015131, -0.0002735, 0.0002791
7: -0.0046174, -0.0029782, -0.0046174, -0.0029782, -0.0003801, 0.0003725
8: 0.9859612, 0.9871160, 0.9859612, 0.9871160, -0.0002678, 0.0002624
9: -0.0041920, -0.0031438, -0.0041920, -0.0031438, -0.0002382, 0.0002431

### Unstable ReLU Count (Linear/Conv2D Layers)
- layer_idx=0, type=LayerType.Linear, total=256, inp1_unstable=4, inp2_unstable=4, delta_unstable=252
- layer_idx=2, type=LayerType.Linear, total=256, inp1_unstable=4, inp2_unstable=4, delta_unstable=255
- layer_idx=4, type=LayerType.Linear, total=256, inp1_unstable=0, inp2_unstable=0, delta_unstable=256
- layer_idx=6, type=LayerType.Linear, total=256, inp1_unstable=0, inp2_unstable=0, delta_unstable=256
- layer_idx=8, type=LayerType.Linear, total=10, inp1_unstable=0, inp2_unstable=0, delta_unstable=10

Time for backsubstitution: 1.50 seconds

### RS candidates at layer 3
type: RSZ, layer: 3, pos: 240
type: RSZ, layer: 3, pos: 49
type: RSZ, layer: 3, pos: 83

Time for candidate selection: 0.00 seconds

### Candidate
type: RSZ, layer: 3, pos: 240

### Candidate
type: RSZ, layer: 3, pos: 49

### Candidate
type: RSZ, layer: 3, pos: 83

### Relational analysis RSZ of RS_RSZ1_RSZ2_RSZ2_RSZ2_RSZ2_RSZ2_RSZ1_RSZ1

#### Relational analysis RSZ result of RS_RSZ1_RSZ2_RSZ2_RSZ2_RSZ2_RSZ2_RSZ1_RSZ1
Status: Status.VERIFIED
Output dim: 8, lower bound: -0.0001614, upper bound: 0.0001291
time: 0.52 seconds

### Relational analysis RSZ of RS_RSZ1_RSZ2_RSZ2_RSZ2_RSZ2_RSZ2_RSZ1_RSZ2

#### Relational analysis RSZ result of RS_RSZ1_RSZ2_RSZ2_RSZ2_RSZ2_RSZ2_RSZ1_RSZ2
Status: Status.VERIFIED
Output dim: 8, lower bound: -0.0002027, upper bound: 0.0001237
time: 0.50 seconds

## BFS RS instance: RS_RSZ1_RSZ2_RSZ2_RSZ2_RSZ2_RSZ2_RSZ2

### Backsubstitution after applying RS history:
0: 0.0035972, 0.0041150, 0.0035972, 0.0041150, -0.0001196, 0.0001192
1: 0.0018420, 0.0019168, 0.0018420, 0.0019168, -0.0000173, 0.0000172
2: 0.0120848, 0.0123711, 0.0120848, 0.0123711, -0.0000659, 0.0000661
3: -0.0021818, -0.0018857, -0.0021818, -0.0018857, -0.0000681, 0.0000684
4: -0.0019956, -0.0016750, -0.0019956, -0.0016750, -0.0000740, 0.0000738
5: 0.0056985, 0.0060018, 0.0056985, 0.0060018, -0.0000698, 0.0000701
6: 0.0003095, 0.0015131, 0.0003095, 0.0015131, -0.0002770, 0.0002780
7: -0.0046174, -0.0029782, -0.0046174, -0.0029782, -0.0003786, 0.0003772
8: 0.9859612, 0.9871160, 0.9859612, 0.9871160, -0.0002667, 0.0002657
9: -0.0041920, -0.0031438, -0.0041920, -0.0031438, -0.0002412, 0.0002421

### Unstable ReLU Count (Linear/Conv2D Layers)
- layer_idx=0, type=LayerType.Linear, total=256, inp1_unstable=4, inp2_unstable=4, delta_unstable=252
- layer_idx=2, type=LayerType.Linear, total=256, inp1_unstable=4, inp2_unstable=4, delta_unstable=255
- layer_idx=4, type=LayerType.Linear, total=256, inp1_unstable=0, inp2_unstable=0, delta_unstable=256
- layer_idx=6, type=LayerType.Linear, total=256, inp1_unstable=0, inp2_unstable=0, delta_unstable=256
- layer_idx=8, type=LayerType.Linear, total=10, inp1_unstable=0, inp2_unstable=0, delta_unstable=10

Time for backsubstitution: 1.44 seconds

### RS candidates at layer 3
type: RSZ, layer: 3, pos: 240
type: RSZ, layer: 3, pos: 83
type: RSZ, layer: 3, pos: 49

Time for candidate selection: 0.00 seconds

### Candidate
type: RSZ, layer: 3, pos: 240

### Candidate
type: RSZ, layer: 3, pos: 83

### Relational analysis RSZ of RS_RSZ1_RSZ2_RSZ2_RSZ2_RSZ2_RSZ2_RSZ2_RSZ1

#### Relational analysis RSZ result of RS_RSZ1_RSZ2_RSZ2_RSZ2_RSZ2_RSZ2_RSZ2_RSZ1
Status: Status.VERIFIED
Output dim: 8, lower bound: -0.0001621, upper bound: 0.0001297
time: 0.53 seconds

### Relational analysis RSZ of RS_RSZ1_RSZ2_RSZ2_RSZ2_RSZ2_RSZ2_RSZ2_RSZ2

#### Relational analysis RSZ result of RS_RSZ1_RSZ2_RSZ2_RSZ2_RSZ2_RSZ2_RSZ2_RSZ2
Status: Status.VERIFIED
Output dim: 8, lower bound: -0.0001989, upper bound: 0.0001237
time: 0.51 seconds

## BFS RS instance: RS_RSZ2_RSZ1_RSZ1_RSZ1_RSZ1_RSZ1_RSZ1

### Backsubstitution after applying RS history:
0: 0.0035972, 0.0041150, 0.0035972, 0.0041150, -0.0001192, 0.0001196
1: 0.0018420, 0.0019168, 0.0018420, 0.0019168, -0.0000172, 0.0000173
2: 0.0120848, 0.0123711, 0.0120848, 0.0123711, -0.0000661, 0.0000659
3: -0.0021818, -0.0018857, -0.0021818, -0.0018857, -0.0000684, 0.0000681
4: -0.0019956, -0.0016750, -0.0019956, -0.0016750, -0.0000738, 0.0000740
5: 0.0056985, 0.0060018, 0.0056985, 0.0060018, -0.0000701, 0.0000698
6: 0.0003095, 0.0015131, 0.0003095, 0.0015131, -0.0002780, 0.0002770
7: -0.0046174, -0.0029782, -0.0046174, -0.0029782, -0.0003772, 0.0003786
8: 0.9859612, 0.9871160, 0.9859612, 0.9871160, -0.0002657, 0.0002667
9: -0.0041920, -0.0031438, -0.0041920, -0.0031438, -0.0002421, 0.0002412

### Unstable ReLU Count (Linear/Conv2D Layers)
- layer_idx=0, type=LayerType.Linear, total=256, inp1_unstable=4, inp2_unstable=4, delta_unstable=252
- layer_idx=2, type=LayerType.Linear, total=256, inp1_unstable=4, inp2_unstable=4, delta_unstable=255
- layer_idx=4, type=LayerType.Linear, total=256, inp1_unstable=0, inp2_unstable=0, delta_unstable=256
- layer_idx=6, type=LayerType.Linear, total=256, inp1_unstable=0, inp2_unstable=0, delta_unstable=256
- layer_idx=8, type=LayerType.Linear, total=10, inp1_unstable=0, inp2_unstable=0, delta_unstable=10

Time for backsubstitution: 1.43 seconds

### RS candidates at layer 3
type: RSZ, layer: 3, pos: 83
type: RSZ, layer: 3, pos: 49
type: RSZ, layer: 3, pos: 240

Time for candidate selection: 0.00 seconds

### Candidate
type: RSZ, layer: 3, pos: 83

### Candidate
type: RSZ, layer: 3, pos: 49

### Relational analysis RSZ of RS_RSZ2_RSZ1_RSZ1_RSZ1_RSZ1_RSZ1_RSZ1_RSZ1

#### Relational analysis RSZ result of RS_RSZ2_RSZ1_RSZ1_RSZ1_RSZ1_RSZ1_RSZ1_RSZ1
Status: Status.VERIFIED
Output dim: 8, lower bound: -0.0001237, upper bound: 0.0001989
time: 0.57 seconds

### Relational analysis RSZ of RS_RSZ2_RSZ1_RSZ1_RSZ1_RSZ1_RSZ1_RSZ1_RSZ2

#### Relational analysis RSZ result of RS_RSZ2_RSZ1_RSZ1_RSZ1_RSZ1_RSZ1_RSZ1_RSZ2
Status: Status.VERIFIED
Output dim: 8, lower bound: -0.0001450, upper bound: 0.0001421
time: 0.55 seconds

## BFS RS instance: RS_RSZ2_RSZ1_RSZ1_RSZ1_RSZ1_RSZ1_RSZ2

### Backsubstitution after applying RS history:
0: 0.0035972, 0.0041150, 0.0035972, 0.0041150, -0.0001177, 0.0001201
1: 0.0018420, 0.0019168, 0.0018420, 0.0019168, -0.0000170, 0.0000173
2: 0.0120848, 0.0123711, 0.0120848, 0.0123711, -0.0000664, 0.0000651
3: -0.0021818, -0.0018857, -0.0021818, -0.0018857, -0.0000687, 0.0000673
4: -0.0019956, -0.0016750, -0.0019956, -0.0016750, -0.0000728, 0.0000743
5: 0.0056985, 0.0060018, 0.0056985, 0.0060018, -0.0000703, 0.0000689
6: 0.0003095, 0.0015131, 0.0003095, 0.0015131, -0.0002791, 0.0002735
7: -0.0046174, -0.0029782, -0.0046174, -0.0029782, -0.0003725, 0.0003801
8: 0.9859612, 0.9871160, 0.9859612, 0.9871160, -0.0002624, 0.0002678
9: -0.0041920, -0.0031438, -0.0041920, -0.0031438, -0.0002431, 0.0002382

### Unstable ReLU Count (Linear/Conv2D Layers)
- layer_idx=0, type=LayerType.Linear, total=256, inp1_unstable=4, inp2_unstable=4, delta_unstable=252
- layer_idx=2, type=LayerType.Linear, total=256, inp1_unstable=4, inp2_unstable=4, delta_unstable=255
- layer_idx=4, type=LayerType.Linear, total=256, inp1_unstable=0, inp2_unstable=0, delta_unstable=256
- layer_idx=6, type=LayerType.Linear, total=256, inp1_unstable=0, inp2_unstable=0, delta_unstable=256
- layer_idx=8, type=LayerType.Linear, total=10, inp1_unstable=0, inp2_unstable=0, delta_unstable=10

Time for backsubstitution: 1.42 seconds

### RS candidates at layer 3
type: RSZ, layer: 3, pos: 49
type: RSZ, layer: 3, pos: 83
type: RSZ, layer: 3, pos: 240

Time for candidate selection: 0.00 seconds

### Candidate
type: RSZ, layer: 3, pos: 49

### Relational analysis RSZ of RS_RSZ2_RSZ1_RSZ1_RSZ1_RSZ1_RSZ1_RSZ2_RSZ1

#### Relational analysis RSZ result of RS_RSZ2_RSZ1_RSZ1_RSZ1_RSZ1_RSZ1_RSZ2_RSZ1
Status: Status.VERIFIED
Output dim: 8, lower bound: -0.0001237, upper bound: 0.0002027
time: 0.56 seconds

### Relational analysis RSZ of RS_RSZ2_RSZ1_RSZ1_RSZ1_RSZ1_RSZ1_RSZ2_RSZ2

#### Relational analysis RSZ result of RS_RSZ2_RSZ1_RSZ1_RSZ1_RSZ1_RSZ1_RSZ2_RSZ2
Status: Status.VERIFIED
Output dim: 8, lower bound: -0.0001444, upper bound: 0.0001417
time: 0.56 seconds

## BFS RS instance: RS_RSZ2_RSZ1_RSZ1_RSZ2_RSZ1_RSZ1_RSZ1

### Backsubstitution after applying RS history:
0: 0.0035972, 0.0041150, 0.0035972, 0.0041150, -0.0001297, 0.0001299
1: 0.0018420, 0.0019168, 0.0018420, 0.0019168, -0.0000187, 0.0000188
2: 0.0120848, 0.0123711, 0.0120848, 0.0123711, -0.0000718, 0.0000717
3: -0.0021818, -0.0018857, -0.0021818, -0.0018857, -0.0000743, 0.0000742
4: -0.0019956, -0.0016750, -0.0019956, -0.0016750, -0.0000803, 0.0000804
5: 0.0056985, 0.0060018, 0.0056985, 0.0060018, -0.0000761, 0.0000760
6: 0.0003095, 0.0015131, 0.0003095, 0.0015131, -0.0003019, 0.0003015
7: -0.0046174, -0.0029782, -0.0046174, -0.0029782, -0.0004106, 0.0004112
8: 0.9859612, 0.9871160, 0.9859612, 0.9871160, -0.0002893, 0.0002896
9: -0.0041920, -0.0031438, -0.0041920, -0.0031438, -0.0002629, 0.0002626

### Unstable ReLU Count (Linear/Conv2D Layers)
- layer_idx=0, type=LayerType.Linear, total=256, inp1_unstable=4, inp2_unstable=4, delta_unstable=252
- layer_idx=2, type=LayerType.Linear, total=256, inp1_unstable=4, inp2_unstable=4, delta_unstable=255
- layer_idx=4, type=LayerType.Linear, total=256, inp1_unstable=0, inp2_unstable=0, delta_unstable=256
- layer_idx=6, type=LayerType.Linear, total=256, inp1_unstable=0, inp2_unstable=0, delta_unstable=256
- layer_idx=8, type=LayerType.Linear, total=10, inp1_unstable=0, inp2_unstable=0, delta_unstable=10

Time for backsubstitution: 1.43 seconds

### RS candidates at layer 3
type: RSZ, layer: 3, pos: 83
type: RSZ, layer: 3, pos: 34
type: RSZ, layer: 3, pos: 49

Time for candidate selection: 0.00 seconds

### Candidate
type: RSZ, layer: 3, pos: 83

### Candidate
type: RSZ, layer: 3, pos: 34

### Candidate
type: RSZ, layer: 3, pos: 49

### Relational analysis RSZ of RS_RSZ2_RSZ1_RSZ1_RSZ2_RSZ1_RSZ1_RSZ1_RSZ1

#### Relational analysis RSZ result of RS_RSZ2_RSZ1_RSZ1_RSZ2_RSZ1_RSZ1_RSZ1_RSZ1
Status: Status.VERIFIED
Output dim: 8, lower bound: -0.0001243, upper bound: 0.0001909
time: 0.61 seconds

### Relational analysis RSZ of RS_RSZ2_RSZ1_RSZ1_RSZ2_RSZ1_RSZ1_RSZ1_RSZ2

#### Relational analysis RSZ result of RS_RSZ2_RSZ1_RSZ1_RSZ2_RSZ1_RSZ1_RSZ1_RSZ2
Status: Status.VERIFIED
Output dim: 8, lower bound: -0.0001581, upper bound: 0.0001428
time: 0.57 seconds

## BFS RS instance: RS_RSZ2_RSZ1_RSZ1_RSZ2_RSZ1_RSZ1_RSZ2

### Backsubstitution after applying RS history:
0: 0.0035972, 0.0041150, 0.0035972, 0.0041150, -0.0001301, 0.0001299
1: 0.0018420, 0.0019168, 0.0018420, 0.0019168, -0.0000188, 0.0000188
2: 0.0120848, 0.0123711, 0.0120848, 0.0123711, -0.0000718, 0.0000719
3: -0.0021818, -0.0018857, -0.0021818, -0.0018857, -0.0000743, 0.0000744
4: -0.0019956, -0.0016750, -0.0019956, -0.0016750, -0.0000805, 0.0000804
5: 0.0056985, 0.0060018, 0.0056985, 0.0060018, -0.0000761, 0.0000762
6: 0.0003095, 0.0015131, 0.0003095, 0.0015131, -0.0003019, 0.0003023
7: -0.0046174, -0.0029782, -0.0046174, -0.0029782, -0.0004117, 0.0004112
8: 0.9859612, 0.9871160, 0.9859612, 0.9871160, -0.0002900, 0.0002896
9: -0.0041920, -0.0031438, -0.0041920, -0.0031438, -0.0002629, 0.0002633

### Unstable ReLU Count (Linear/Conv2D Layers)
- layer_idx=0, type=LayerType.Linear, total=256, inp1_unstable=4, inp2_unstable=4, delta_unstable=252
- layer_idx=2, type=LayerType.Linear, total=256, inp1_unstable=4, inp2_unstable=4, delta_unstable=255
- layer_idx=4, type=LayerType.Linear, total=256, inp1_unstable=0, inp2_unstable=0, delta_unstable=256
- layer_idx=6, type=LayerType.Linear, total=256, inp1_unstable=0, inp2_unstable=0, delta_unstable=256
- layer_idx=8, type=LayerType.Linear, total=10, inp1_unstable=0, inp2_unstable=0, delta_unstable=10

Time for backsubstitution: 1.46 seconds

### RS candidates at layer 3
type: RSZ, layer: 3, pos: 83
type: RSZ, layer: 3, pos: 49
type: RSZ, layer: 3, pos: 34

Time for candidate selection: 0.00 seconds

### Candidate
type: RSZ, layer: 3, pos: 83

### Candidate
type: RSZ, layer: 3, pos: 49

### Relational analysis RSZ of RS_RSZ2_RSZ1_RSZ1_RSZ2_RSZ1_RSZ1_RSZ2_RSZ1

#### Relational analysis RSZ result of RS_RSZ2_RSZ1_RSZ1_RSZ2_RSZ1_RSZ1_RSZ2_RSZ1
Status: Status.VERIFIED
Output dim: 8, lower bound: -0.0001222, upper bound: 0.0001912
time: 0.54 seconds

### Relational analysis RSZ of RS_RSZ2_RSZ1_RSZ1_RSZ2_RSZ1_RSZ1_RSZ2_RSZ2

#### Relational analysis RSZ result of RS_RSZ2_RSZ1_RSZ1_RSZ2_RSZ1_RSZ1_RSZ2_RSZ2
Status: Status.VERIFIED
Output dim: 8, lower bound: -0.0001580, upper bound: 0.0001459
time: 0.51 seconds

## BFS RS instance: RS_RSZ2_RSZ1_RSZ1_RSZ2_RSZ1_RSZ2_RSZ1

### Backsubstitution after applying RS history:
0: 0.0035972, 0.0041150, 0.0035972, 0.0041150, -0.0001297, 0.0001299
1: 0.0018420, 0.0019168, 0.0018420, 0.0019168, -0.0000187, 0.0000188
2: 0.0120848, 0.0123711, 0.0120848, 0.0123711, -0.0000718, 0.0000717
3: -0.0021818, -0.0018857, -0.0021818, -0.0018857, -0.0000743, 0.0000742
4: -0.0019956, -0.0016750, -0.0019956, -0.0016750, -0.0000803, 0.0000804
5: 0.0056985, 0.0060018, 0.0056985, 0.0060018, -0.0000761, 0.0000760
6: 0.0003095, 0.0015131, 0.0003095, 0.0015131, -0.0003019, 0.0003015
7: -0.0046174, -0.0029782, -0.0046174, -0.0029782, -0.0004106, 0.0004112
8: 0.9859612, 0.9871160, 0.9859612, 0.9871160, -0.0002893, 0.0002896
9: -0.0041920, -0.0031438, -0.0041920, -0.0031438, -0.0002629, 0.0002626

### Unstable ReLU Count (Linear/Conv2D Layers)
- layer_idx=0, type=LayerType.Linear, total=256, inp1_unstable=4, inp2_unstable=4, delta_unstable=252
- layer_idx=2, type=LayerType.Linear, total=256, inp1_unstable=4, inp2_unstable=4, delta_unstable=255
- layer_idx=4, type=LayerType.Linear, total=256, inp1_unstable=0, inp2_unstable=0, delta_unstable=256
- layer_idx=6, type=LayerType.Linear, total=256, inp1_unstable=0, inp2_unstable=0, delta_unstable=256
- layer_idx=8, type=LayerType.Linear, total=10, inp1_unstable=0, inp2_unstable=0, delta_unstable=10

Time for backsubstitution: 1.45 seconds

### RS candidates at layer 3
type: RSZ, layer: 3, pos: 34
type: RSZ, layer: 3, pos: 49
type: RSZ, layer: 3, pos: 83

Time for candidate selection: 0.00 seconds

### Candidate
type: RSZ, layer: 3, pos: 34

### Candidate
type: RSZ, layer: 3, pos: 49

### Relational analysis RSZ of RS_RSZ2_RSZ1_RSZ1_RSZ2_RSZ1_RSZ2_RSZ1_RSZ1

#### Relational analysis RSZ result of RS_RSZ2_RSZ1_RSZ1_RSZ2_RSZ1_RSZ2_RSZ1_RSZ1
Status: Status.VERIFIED
Output dim: 8, lower bound: -0.0001243, upper bound: 0.0001941
time: 0.51 seconds

### Relational analysis RSZ of RS_RSZ2_RSZ1_RSZ1_RSZ2_RSZ1_RSZ2_RSZ1_RSZ2

#### Relational analysis RSZ result of RS_RSZ2_RSZ1_RSZ1_RSZ2_RSZ1_RSZ2_RSZ1_RSZ2
Status: Status.VERIFIED
Output dim: 8, lower bound: -0.0001568, upper bound: 0.0001420
time: 0.55 seconds

## BFS RS instance: RS_RSZ2_RSZ1_RSZ1_RSZ2_RSZ1_RSZ2_RSZ2

### Backsubstitution after applying RS history:
0: 0.0035972, 0.0041150, 0.0035972, 0.0041150, -0.0001301, 0.0001299
1: 0.0018420, 0.0019168, 0.0018420, 0.0019168, -0.0000188, 0.0000188
2: 0.0120848, 0.0123711, 0.0120848, 0.0123711, -0.0000718, 0.0000719
3: -0.0021818, -0.0018857, -0.0021818, -0.0018857, -0.0000743, 0.0000744
4: -0.0019956, -0.0016750, -0.0019956, -0.0016750, -0.0000805, 0.0000804
5: 0.0056985, 0.0060018, 0.0056985, 0.0060018, -0.0000761, 0.0000762
6: 0.0003095, 0.0015131, 0.0003095, 0.0015131, -0.0003019, 0.0003023
7: -0.0046174, -0.0029782, -0.0046174, -0.0029782, -0.0004117, 0.0004112
8: 0.9859612, 0.9871160, 0.9859612, 0.9871160, -0.0002900, 0.0002896
9: -0.0041920, -0.0031438, -0.0041920, -0.0031438, -0.0002629, 0.0002633

### Unstable ReLU Count (Linear/Conv2D Layers)
- layer_idx=0, type=LayerType.Linear, total=256, inp1_unstable=4, inp2_unstable=4, delta_unstable=252
- layer_idx=2, type=LayerType.Linear, total=256, inp1_unstable=4, inp2_unstable=4, delta_unstable=255
- layer_idx=4, type=LayerType.Linear, total=256, inp1_unstable=0, inp2_unstable=0, delta_unstable=256
- layer_idx=6, type=LayerType.Linear, total=256, inp1_unstable=0, inp2_unstable=0, delta_unstable=256
- layer_idx=8, type=LayerType.Linear, total=10, inp1_unstable=0, inp2_unstable=0, delta_unstable=10

Time for backsubstitution: 1.43 seconds

### RS candidates at layer 3
type: RSZ, layer: 3, pos: 83
type: RSZ, layer: 3, pos: 34
type: RSZ, layer: 3, pos: 49

Time for candidate selection: 0.00 seconds

### Candidate
type: RSZ, layer: 3, pos: 83

### Candidate
type: RSZ, layer: 3, pos: 34

### Candidate
type: RSZ, layer: 3, pos: 49

### Relational analysis RSZ of RS_RSZ2_RSZ1_RSZ1_RSZ2_RSZ1_RSZ2_RSZ2_RSZ1

#### Relational analysis RSZ result of RS_RSZ2_RSZ1_RSZ1_RSZ2_RSZ1_RSZ2_RSZ2_RSZ1
Status: Status.VERIFIED
Output dim: 8, lower bound: -0.0001213, upper bound: 0.0001939
time: 0.55 seconds

### Relational analysis RSZ of RS_RSZ2_RSZ1_RSZ1_RSZ2_RSZ1_RSZ2_RSZ2_RSZ2

#### Relational analysis RSZ result of RS_RSZ2_RSZ1_RSZ1_RSZ2_RSZ1_RSZ2_RSZ2_RSZ2
Status: Status.VERIFIED
Output dim: 8, lower bound: -0.0001564, upper bound: 0.0001439
time: 0.54 seconds

## BFS RS instance: RS_RSZ2_RSZ1_RSZ1_RSZ2_RSZ2_RSZ2_RSZ1

### Backsubstitution after applying RS history:
0: 0.0035972, 0.0041150, 0.0035972, 0.0041150, -0.0001186, 0.0001176
1: 0.0018420, 0.0019168, 0.0018420, 0.0019168, -0.0000171, 0.0000170
2: 0.0120848, 0.0123711, 0.0120848, 0.0123711, -0.0000650, 0.0000656
3: -0.0021818, -0.0018857, -0.0021818, -0.0018857, -0.0000673, 0.0000678
4: -0.0019956, -0.0016750, -0.0019956, -0.0016750, -0.0000734, 0.0000728
5: 0.0056985, 0.0060018, 0.0056985, 0.0060018, -0.0000689, 0.0000695
6: 0.0003095, 0.0015131, 0.0003095, 0.0015131, -0.0002734, 0.0002758
7: -0.0046174, -0.0029782, -0.0046174, -0.0029782, -0.0003756, 0.0003724
8: 0.9859612, 0.9871160, 0.9859612, 0.9871160, -0.0002646, 0.0002623
9: -0.0041920, -0.0031438, -0.0041920, -0.0031438, -0.0002381, 0.0002402

### Unstable ReLU Count (Linear/Conv2D Layers)
- layer_idx=0, type=LayerType.Linear, total=256, inp1_unstable=4, inp2_unstable=4, delta_unstable=252
- layer_idx=2, type=LayerType.Linear, total=256, inp1_unstable=4, inp2_unstable=4, delta_unstable=255
- layer_idx=4, type=LayerType.Linear, total=256, inp1_unstable=0, inp2_unstable=0, delta_unstable=256
- layer_idx=6, type=LayerType.Linear, total=256, inp1_unstable=0, inp2_unstable=0, delta_unstable=256
- layer_idx=8, type=LayerType.Linear, total=10, inp1_unstable=0, inp2_unstable=0, delta_unstable=10

Time for backsubstitution: 1.44 seconds

### RS candidates at layer 3
type: RSZ, layer: 3, pos: 240
type: RSZ, layer: 3, pos: 49
type: RSZ, layer: 3, pos: 83

Time for candidate selection: 0.00 seconds

### Candidate
type: RSZ, layer: 3, pos: 240

### Candidate
type: RSZ, layer: 3, pos: 49

### Relational analysis RSZ of RS_RSZ2_RSZ1_RSZ1_RSZ2_RSZ2_RSZ2_RSZ1_RSZ1

#### Relational analysis RSZ result of RS_RSZ2_RSZ1_RSZ1_RSZ2_RSZ2_RSZ2_RSZ1_RSZ1
Status: Status.VERIFIED
Output dim: 8, lower bound: -0.0001291, upper bound: 0.0001620
time: 0.55 seconds

### Relational analysis RSZ of RS_RSZ2_RSZ1_RSZ1_RSZ2_RSZ2_RSZ2_RSZ1_RSZ2

#### Relational analysis RSZ result of RS_RSZ2_RSZ1_RSZ1_RSZ2_RSZ2_RSZ2_RSZ1_RSZ2
Status: Status.VERIFIED
Output dim: 8, lower bound: -0.0001988, upper bound: 0.0001412
time: 0.56 seconds

## BFS RS instance: RS_RSZ2_RSZ1_RSZ1_RSZ2_RSZ2_RSZ2_RSZ2

### Backsubstitution after applying RS history:
0: 0.0035972, 0.0041150, 0.0035972, 0.0041150, -0.0001173, 0.0001185
1: 0.0018420, 0.0019168, 0.0018420, 0.0019168, -0.0000170, 0.0000171
2: 0.0120848, 0.0123711, 0.0120848, 0.0123711, -0.0000655, 0.0000649
3: -0.0021818, -0.0018857, -0.0021818, -0.0018857, -0.0000677, 0.0000671
4: -0.0019956, -0.0016750, -0.0019956, -0.0016750, -0.0000726, 0.0000733
5: 0.0056985, 0.0060018, 0.0056985, 0.0060018, -0.0000694, 0.0000687
6: 0.0003095, 0.0015131, 0.0003095, 0.0015131, -0.0002754, 0.0002727
7: -0.0046174, -0.0029782, -0.0046174, -0.0029782, -0.0003715, 0.0003750
8: 0.9859612, 0.9871160, 0.9859612, 0.9871160, -0.0002617, 0.0002642
9: -0.0041920, -0.0031438, -0.0041920, -0.0031438, -0.0002398, 0.0002375

### Unstable ReLU Count (Linear/Conv2D Layers)
- layer_idx=0, type=LayerType.Linear, total=256, inp1_unstable=4, inp2_unstable=4, delta_unstable=252
- layer_idx=2, type=LayerType.Linear, total=256, inp1_unstable=4, inp2_unstable=4, delta_unstable=255
- layer_idx=4, type=LayerType.Linear, total=256, inp1_unstable=0, inp2_unstable=0, delta_unstable=256
- layer_idx=6, type=LayerType.Linear, total=256, inp1_unstable=0, inp2_unstable=0, delta_unstable=256
- layer_idx=8, type=LayerType.Linear, total=10, inp1_unstable=0, inp2_unstable=0, delta_unstable=10

Time for backsubstitution: 1.44 seconds

### RS candidates at layer 3
type: RSZ, layer: 3, pos: 49
type: RSZ, layer: 3, pos: 83
type: RSZ, layer: 3, pos: 240

Time for candidate selection: 0.00 seconds

### Candidate
type: RSZ, layer: 3, pos: 49

### Relational analysis RSZ of RS_RSZ2_RSZ1_RSZ1_RSZ2_RSZ2_RSZ2_RSZ2_RSZ1

#### Relational analysis RSZ result of RS_RSZ2_RSZ1_RSZ1_RSZ2_RSZ2_RSZ2_RSZ2_RSZ1
Status: Status.VERIFIED
Output dim: 8, lower bound: -0.0001274, upper bound: 0.0001620
time: 0.51 seconds

### Relational analysis RSZ of RS_RSZ2_RSZ1_RSZ1_RSZ2_RSZ2_RSZ2_RSZ2_RSZ2

#### Relational analysis RSZ result of RS_RSZ2_RSZ1_RSZ1_RSZ2_RSZ2_RSZ2_RSZ2_RSZ2
Status: Status.VERIFIED
Output dim: 8, lower bound: -0.0001933, upper bound: 0.0001376
time: 0.50 seconds

## BFS RS instance: RS_RSZ2_RSZ1_RSZ2_RSZ1_RSZ1_RSZ2_RSZ1

### Backsubstitution after applying RS history:
0: 0.0035972, 0.0041150, 0.0035972, 0.0041150, -0.0001305, 0.0001273
1: 0.0018420, 0.0019168, 0.0018420, 0.0019168, -0.0000189, 0.0000184
2: 0.0120848, 0.0123711, 0.0120848, 0.0123711, -0.0000704, 0.0000721
3: -0.0021818, -0.0018857, -0.0021818, -0.0018857, -0.0000728, 0.0000746
4: -0.0019956, -0.0016750, -0.0019956, -0.0016750, -0.0000808, 0.0000788
5: 0.0056985, 0.0060018, 0.0056985, 0.0060018, -0.0000746, 0.0000764
6: 0.0003095, 0.0015131, 0.0003095, 0.0015131, -0.0002959, 0.0003033
7: -0.0046174, -0.0029782, -0.0046174, -0.0029782, -0.0004131, 0.0004030
8: 0.9859612, 0.9871160, 0.9859612, 0.9871160, -0.0002910, 0.0002839
9: -0.0041920, -0.0031438, -0.0041920, -0.0031438, -0.0002577, 0.0002641

### Unstable ReLU Count (Linear/Conv2D Layers)
- layer_idx=0, type=LayerType.Linear, total=256, inp1_unstable=4, inp2_unstable=4, delta_unstable=252
- layer_idx=2, type=LayerType.Linear, total=256, inp1_unstable=4, inp2_unstable=4, delta_unstable=255
- layer_idx=4, type=LayerType.Linear, total=256, inp1_unstable=0, inp2_unstable=0, delta_unstable=256
- layer_idx=6, type=LayerType.Linear, total=256, inp1_unstable=0, inp2_unstable=0, delta_unstable=256
- layer_idx=8, type=LayerType.Linear, total=10, inp1_unstable=0, inp2_unstable=0, delta_unstable=10

Time for backsubstitution: 1.43 seconds

### RS candidates at layer 3
type: RSZ, layer: 3, pos: 34
type: RSZ, layer: 3, pos: 83
type: RSZ, layer: 3, pos: 49

Time for candidate selection: 0.00 seconds

### Candidate
type: RSZ, layer: 3, pos: 34

### Candidate
type: RSZ, layer: 3, pos: 83

### Candidate
type: RSZ, layer: 3, pos: 49

### Relational analysis RSZ of RS_RSZ2_RSZ1_RSZ2_RSZ1_RSZ1_RSZ2_RSZ1_RSZ1

#### Relational analysis RSZ result of RS_RSZ2_RSZ1_RSZ2_RSZ1_RSZ1_RSZ2_RSZ1_RSZ1
Status: Status.VERIFIED
Output dim: 8, lower bound: -0.0001315, upper bound: 0.0001985
time: 0.55 seconds

### Relational analysis RSZ of RS_RSZ2_RSZ1_RSZ2_RSZ1_RSZ1_RSZ2_RSZ1_RSZ2

#### Relational analysis RSZ result of RS_RSZ2_RSZ1_RSZ2_RSZ1_RSZ1_RSZ2_RSZ1_RSZ2
Status: Status.VERIFIED
Output dim: 8, lower bound: -0.0001621, upper bound: 0.0001396
time: 0.55 seconds

## BFS RS instance: RS_RSZ2_RSZ1_RSZ2_RSZ1_RSZ1_RSZ2_RSZ2

### Backsubstitution after applying RS history:
0: 0.0035972, 0.0041150, 0.0035972, 0.0041150, -0.0001306, 0.0001270
1: 0.0018420, 0.0019168, 0.0018420, 0.0019168, -0.0000189, 0.0000183
2: 0.0120848, 0.0123711, 0.0120848, 0.0123711, -0.0000702, 0.0000722
3: -0.0021818, -0.0018857, -0.0021818, -0.0018857, -0.0000726, 0.0000747
4: -0.0019956, -0.0016750, -0.0019956, -0.0016750, -0.0000808, 0.0000786
5: 0.0056985, 0.0060018, 0.0056985, 0.0060018, -0.0000744, 0.0000765
6: 0.0003095, 0.0015131, 0.0003095, 0.0015131, -0.0002951, 0.0003036
7: -0.0046174, -0.0029782, -0.0046174, -0.0029782, -0.0004134, 0.0004020
8: 0.9859612, 0.9871160, 0.9859612, 0.9871160, -0.0002912, 0.0002831
9: -0.0041920, -0.0031438, -0.0041920, -0.0031438, -0.0002570, 0.0002643

### Unstable ReLU Count (Linear/Conv2D Layers)
- layer_idx=0, type=LayerType.Linear, total=256, inp1_unstable=4, inp2_unstable=4, delta_unstable=252
- layer_idx=2, type=LayerType.Linear, total=256, inp1_unstable=4, inp2_unstable=4, delta_unstable=255
- layer_idx=4, type=LayerType.Linear, total=256, inp1_unstable=0, inp2_unstable=0, delta_unstable=256
- layer_idx=6, type=LayerType.Linear, total=256, inp1_unstable=0, inp2_unstable=0, delta_unstable=256
- layer_idx=8, type=LayerType.Linear, total=10, inp1_unstable=0, inp2_unstable=0, delta_unstable=10

Time for backsubstitution: 1.47 seconds

### RS candidates at layer 3
type: RSZ, layer: 3, pos: 83
type: RSZ, layer: 3, pos: 34
type: RSZ, layer: 3, pos: 49

Time for candidate selection: 0.00 seconds

### Candidate
type: RSZ, layer: 3, pos: 83

### Candidate
type: RSZ, layer: 3, pos: 34

### Candidate
type: RSZ, layer: 3, pos: 49

### Relational analysis RSZ of RS_RSZ2_RSZ1_RSZ2_RSZ1_RSZ1_RSZ2_RSZ2_RSZ1

#### Relational analysis RSZ result of RS_RSZ2_RSZ1_RSZ2_RSZ1_RSZ1_RSZ2_RSZ2_RSZ1
Status: Status.VERIFIED
Output dim: 8, lower bound: -0.0001231, upper bound: 0.0001967
time: 0.52 seconds

### Relational analysis RSZ of RS_RSZ2_RSZ1_RSZ2_RSZ1_RSZ1_RSZ2_RSZ2_RSZ2

#### Relational analysis RSZ result of RS_RSZ2_RSZ1_RSZ2_RSZ1_RSZ1_RSZ2_RSZ2_RSZ2
Status: Status.VERIFIED
Output dim: 8, lower bound: -0.0001563, upper bound: 0.0001400
time: 0.51 seconds

## BFS RS instance: RS_RSZ2_RSZ1_RSZ2_RSZ1_RSZ2_RSZ2_RSZ1

### Backsubstitution after applying RS history:
0: 0.0035972, 0.0041150, 0.0035972, 0.0041150, -0.0001305, 0.0001273
1: 0.0018420, 0.0019168, 0.0018420, 0.0019168, -0.0000189, 0.0000184
2: 0.0120848, 0.0123711, 0.0120848, 0.0123711, -0.0000704, 0.0000721
3: -0.0021818, -0.0018857, -0.0021818, -0.0018857, -0.0000728, 0.0000746
4: -0.0019956, -0.0016750, -0.0019956, -0.0016750, -0.0000808, 0.0000788
5: 0.0056985, 0.0060018, 0.0056985, 0.0060018, -0.0000746, 0.0000764
6: 0.0003095, 0.0015131, 0.0003095, 0.0015131, -0.0002959, 0.0003033
7: -0.0046174, -0.0029782, -0.0046174, -0.0029782, -0.0004131, 0.0004030
8: 0.9859612, 0.9871160, 0.9859612, 0.9871160, -0.0002910, 0.0002839
9: -0.0041920, -0.0031438, -0.0041920, -0.0031438, -0.0002577, 0.0002641

### Unstable ReLU Count (Linear/Conv2D Layers)
- layer_idx=0, type=LayerType.Linear, total=256, inp1_unstable=4, inp2_unstable=4, delta_unstable=252
- layer_idx=2, type=LayerType.Linear, total=256, inp1_unstable=4, inp2_unstable=4, delta_unstable=255
- layer_idx=4, type=LayerType.Linear, total=256, inp1_unstable=0, inp2_unstable=0, delta_unstable=256
- layer_idx=6, type=LayerType.Linear, total=256, inp1_unstable=0, inp2_unstable=0, delta_unstable=256
- layer_idx=8, type=LayerType.Linear, total=10, inp1_unstable=0, inp2_unstable=0, delta_unstable=10

Time for backsubstitution: 1.59 seconds

### RS candidates at layer 3
type: RSZ, layer: 3, pos: 83
type: RSZ, layer: 3, pos: 49
type: RSZ, layer: 3, pos: 34

Time for candidate selection: 0.00 seconds

### Candidate
type: RSZ, layer: 3, pos: 83

### Candidate
type: RSZ, layer: 3, pos: 49

### Relational analysis RSZ of RS_RSZ2_RSZ1_RSZ2_RSZ1_RSZ2_RSZ2_RSZ1_RSZ1

#### Relational analysis RSZ result of RS_RSZ2_RSZ1_RSZ2_RSZ1_RSZ2_RSZ2_RSZ1_RSZ1
Status: Status.VERIFIED
Output dim: 8, lower bound: -0.0001324, upper bound: 0.0001501
time: 0.58 seconds

### Relational analysis RSZ of RS_RSZ2_RSZ1_RSZ2_RSZ1_RSZ2_RSZ2_RSZ1_RSZ2

#### Relational analysis RSZ result of RS_RSZ2_RSZ1_RSZ2_RSZ1_RSZ2_RSZ2_RSZ1_RSZ2
Status: Status.VERIFIED
Output dim: 8, lower bound: -0.0001907, upper bound: 0.0001252
time: 0.55 seconds

## BFS RS instance: RS_RSZ2_RSZ1_RSZ2_RSZ1_RSZ2_RSZ2_RSZ2

### Backsubstitution after applying RS history:
0: 0.0035972, 0.0041150, 0.0035972, 0.0041150, -0.0001306, 0.0001270
1: 0.0018420, 0.0019168, 0.0018420, 0.0019168, -0.0000189, 0.0000183
2: 0.0120848, 0.0123711, 0.0120848, 0.0123711, -0.0000702, 0.0000722
3: -0.0021818, -0.0018857, -0.0021818, -0.0018857, -0.0000726, 0.0000747
4: -0.0019956, -0.0016750, -0.0019956, -0.0016750, -0.0000808, 0.0000786
5: 0.0056985, 0.0060018, 0.0056985, 0.0060018, -0.0000744, 0.0000765
6: 0.0003095, 0.0015131, 0.0003095, 0.0015131, -0.0002951, 0.0003036
7: -0.0046174, -0.0029782, -0.0046174, -0.0029782, -0.0004134, 0.0004020
8: 0.9859612, 0.9871160, 0.9859612, 0.9871160, -0.0002912, 0.0002831
9: -0.0041920, -0.0031438, -0.0041920, -0.0031438, -0.0002570, 0.0002643

### Unstable ReLU Count (Linear/Conv2D Layers)
- layer_idx=0, type=LayerType.Linear, total=256, inp1_unstable=4, inp2_unstable=4, delta_unstable=252
- layer_idx=2, type=LayerType.Linear, total=256, inp1_unstable=4, inp2_unstable=4, delta_unstable=255
- layer_idx=4, type=LayerType.Linear, total=256, inp1_unstable=0, inp2_unstable=0, delta_unstable=256
- layer_idx=6, type=LayerType.Linear, total=256, inp1_unstable=0, inp2_unstable=0, delta_unstable=256
- layer_idx=8, type=LayerType.Linear, total=10, inp1_unstable=0, inp2_unstable=0, delta_unstable=10

Time for backsubstitution: 1.47 seconds

### RS candidates at layer 3
type: RSZ, layer: 3, pos: 83
type: RSZ, layer: 3, pos: 34
type: RSZ, layer: 3, pos: 49

Time for candidate selection: 0.00 seconds

### Candidate
type: RSZ, layer: 3, pos: 83

### Candidate
type: RSZ, layer: 3, pos: 34

### Candidate
type: RSZ, layer: 3, pos: 49

### Relational analysis RSZ of RS_RSZ2_RSZ1_RSZ2_RSZ1_RSZ2_RSZ2_RSZ2_RSZ1

#### Relational analysis RSZ result of RS_RSZ2_RSZ1_RSZ2_RSZ1_RSZ2_RSZ2_RSZ2_RSZ1
Status: Status.VERIFIED
Output dim: 8, lower bound: -0.0001286, upper bound: 0.0001524
time: 0.52 seconds

### Relational analysis RSZ of RS_RSZ2_RSZ1_RSZ2_RSZ1_RSZ2_RSZ2_RSZ2_RSZ2

#### Relational analysis RSZ result of RS_RSZ2_RSZ1_RSZ2_RSZ1_RSZ2_RSZ2_RSZ2_RSZ2
Status: Status.VERIFIED
Output dim: 8, lower bound: -0.0001900, upper bound: 0.0001279
time: 0.53 seconds

## BFS RS instance: RS_RSZ2_RSZ1_RSZ2_RSZ2_RSZ2_RSZ1_RSZ2

### Backsubstitution after applying RS history:
0: 0.0035972, 0.0041150, 0.0035972, 0.0041150, -0.0001322, 0.0001143
1: 0.0018420, 0.0019168, 0.0018420, 0.0019168, -0.0000191, 0.0000165
2: 0.0120848, 0.0123711, 0.0120848, 0.0123711, -0.0000632, 0.0000731
3: -0.0021818, -0.0018857, -0.0021818, -0.0018857, -0.0000654, 0.0000756
4: -0.0019956, -0.0016750, -0.0019956, -0.0016750, -0.0000818, 0.0000708
5: 0.0056985, 0.0060018, 0.0056985, 0.0060018, -0.0000670, 0.0000774
6: 0.0003095, 0.0015131, 0.0003095, 0.0015131, -0.0002658, 0.0003072
7: -0.0046174, -0.0029782, -0.0046174, -0.0029782, -0.0004184, 0.0003619
8: 0.9859612, 0.9871160, 0.9859612, 0.9871160, -0.0002947, 0.0002550
9: -0.0041920, -0.0031438, -0.0041920, -0.0031438, -0.0002314, 0.0002675

### Unstable ReLU Count (Linear/Conv2D Layers)
- layer_idx=0, type=LayerType.Linear, total=256, inp1_unstable=4, inp2_unstable=4, delta_unstable=252
- layer_idx=2, type=LayerType.Linear, total=256, inp1_unstable=4, inp2_unstable=4, delta_unstable=255
- layer_idx=4, type=LayerType.Linear, total=256, inp1_unstable=0, inp2_unstable=0, delta_unstable=256
- layer_idx=6, type=LayerType.Linear, total=256, inp1_unstable=0, inp2_unstable=0, delta_unstable=256
- layer_idx=8, type=LayerType.Linear, total=10, inp1_unstable=0, inp2_unstable=0, delta_unstable=10

Time for backsubstitution: 1.55 seconds

### RS candidates at layer 3
type: RSZ, layer: 3, pos: 34
type: RSZ, layer: 3, pos: 240
type: RSZ, layer: 3, pos: 83

Time for candidate selection: 0.00 seconds

### Candidate
type: RSZ, layer: 3, pos: 34

### Candidate
type: RSZ, layer: 3, pos: 240

### Candidate
type: RSZ, layer: 3, pos: 83

### Relational analysis RSZ of RS_RSZ2_RSZ1_RSZ2_RSZ2_RSZ2_RSZ1_RSZ2_RSZ1

#### Relational analysis RSZ result of RS_RSZ2_RSZ1_RSZ2_RSZ2_RSZ2_RSZ1_RSZ2_RSZ1
Status: Status.VERIFIED
Output dim: 8, lower bound: -0.0001735, upper bound: 0.0001398
time: 0.56 seconds

### Relational analysis RSZ of RS_RSZ2_RSZ1_RSZ2_RSZ2_RSZ2_RSZ1_RSZ2_RSZ2

#### Relational analysis RSZ result of RS_RSZ2_RSZ1_RSZ2_RSZ2_RSZ2_RSZ1_RSZ2_RSZ2
Status: Status.VERIFIED
Output dim: 8, lower bound: -0.0001981, upper bound: 0.0001234
time: 0.53 seconds

## BFS RS instance: RS_RSZ2_RSZ1_RSZ2_RSZ2_RSZ2_RSZ2_RSZ2

### Backsubstitution after applying RS history:
0: 0.0035972, 0.0041150, 0.0035972, 0.0041150, -0.0001322, 0.0001143
1: 0.0018420, 0.0019168, 0.0018420, 0.0019168, -0.0000191, 0.0000165
2: 0.0120848, 0.0123711, 0.0120848, 0.0123711, -0.0000632, 0.0000731
3: -0.0021818, -0.0018857, -0.0021818, -0.0018857, -0.0000654, 0.0000756
4: -0.0019956, -0.0016750, -0.0019956, -0.0016750, -0.0000818, 0.0000708
5: 0.0056985, 0.0060018, 0.0056985, 0.0060018, -0.0000670, 0.0000774
6: 0.0003095, 0.0015131, 0.0003095, 0.0015131, -0.0002658, 0.0003072
7: -0.0046174, -0.0029782, -0.0046174, -0.0029782, -0.0004184, 0.0003619
8: 0.9859612, 0.9871160, 0.9859612, 0.9871160, -0.0002947, 0.0002550
9: -0.0041920, -0.0031438, -0.0041920, -0.0031438, -0.0002314, 0.0002675

### Unstable ReLU Count (Linear/Conv2D Layers)
- layer_idx=0, type=LayerType.Linear, total=256, inp1_unstable=4, inp2_unstable=4, delta_unstable=252
- layer_idx=2, type=LayerType.Linear, total=256, inp1_unstable=4, inp2_unstable=4, delta_unstable=255
- layer_idx=4, type=LayerType.Linear, total=256, inp1_unstable=0, inp2_unstable=0, delta_unstable=256
- layer_idx=6, type=LayerType.Linear, total=256, inp1_unstable=0, inp2_unstable=0, delta_unstable=256
- layer_idx=8, type=LayerType.Linear, total=10, inp1_unstable=0, inp2_unstable=0, delta_unstable=10

Time for backsubstitution: 1.58 seconds

### RS candidates at layer 3
type: RSZ, layer: 3, pos: 34
type: RSZ, layer: 3, pos: 83
type: RSZ, layer: 3, pos: 240

Time for candidate selection: 0.00 seconds

### Candidate
type: RSZ, layer: 3, pos: 34

### Candidate
type: RSZ, layer: 3, pos: 83

### Relational analysis RSZ of RS_RSZ2_RSZ1_RSZ2_RSZ2_RSZ2_RSZ2_RSZ2_RSZ1

#### Relational analysis RSZ result of RS_RSZ2_RSZ1_RSZ2_RSZ2_RSZ2_RSZ2_RSZ2_RSZ1
Status: Status.VERIFIED
Output dim: 8, lower bound: -0.0001667, upper bound: 0.0001400
time: 0.56 seconds

### Relational analysis RSZ of RS_RSZ2_RSZ1_RSZ2_RSZ2_RSZ2_RSZ2_RSZ2_RSZ2

#### Relational analysis RSZ result of RS_RSZ2_RSZ1_RSZ2_RSZ2_RSZ2_RSZ2_RSZ2_RSZ2
Status: Status.VERIFIED
Output dim: 8, lower bound: -0.0001980, upper bound: 0.0001257
time: 0.52 seconds

## BFS RS instance: RS_RSZ2_RSZ2_RSZ1_RSZ1_RSZ1_RSZ1_RSZ1

### Backsubstitution after applying RS history:
0: 0.0035972, 0.0041150, 0.0035972, 0.0041150, -0.0001286, 0.0001311
1: 0.0018420, 0.0019168, 0.0018420, 0.0019168, -0.0000186, 0.0000189
2: 0.0120848, 0.0123711, 0.0120848, 0.0123711, -0.0000725, 0.0000711
3: -0.0021818, -0.0018857, -0.0021818, -0.0018857, -0.0000750, 0.0000735
4: -0.0019956, -0.0016750, -0.0019956, -0.0016750, -0.0000796, 0.0000811
5: 0.0056985, 0.0060018, 0.0056985, 0.0060018, -0.0000768, 0.0000753
6: 0.0003095, 0.0015131, 0.0003095, 0.0015131, -0.0003047, 0.0002989
7: -0.0046174, -0.0029782, -0.0046174, -0.0029782, -0.0004071, 0.0004150
8: 0.9859612, 0.9871160, 0.9859612, 0.9871160, -0.0002868, 0.0002923
9: -0.0041920, -0.0031438, -0.0041920, -0.0031438, -0.0002653, 0.0002603

### Unstable ReLU Count (Linear/Conv2D Layers)
- layer_idx=0, type=LayerType.Linear, total=256, inp1_unstable=4, inp2_unstable=4, delta_unstable=252
- layer_idx=2, type=LayerType.Linear, total=256, inp1_unstable=4, inp2_unstable=4, delta_unstable=255
- layer_idx=4, type=LayerType.Linear, total=256, inp1_unstable=0, inp2_unstable=0, delta_unstable=256
- layer_idx=6, type=LayerType.Linear, total=256, inp1_unstable=0, inp2_unstable=0, delta_unstable=256
- layer_idx=8, type=LayerType.Linear, total=10, inp1_unstable=0, inp2_unstable=0, delta_unstable=10

Time for backsubstitution: 1.56 seconds

### RS candidates at layer 3
type: RSZ, layer: 3, pos: 34
type: RSZ, layer: 3, pos: 83
type: RSZ, layer: 3, pos: 49

Time for candidate selection: 0.00 seconds

### Candidate
type: RSZ, layer: 3, pos: 34

### Candidate
type: RSZ, layer: 3, pos: 83

### Candidate
type: RSZ, layer: 3, pos: 49

### Relational analysis RSZ of RS_RSZ2_RSZ2_RSZ1_RSZ1_RSZ1_RSZ1_RSZ1_RSZ1

#### Relational analysis RSZ result of RS_RSZ2_RSZ2_RSZ1_RSZ1_RSZ1_RSZ1_RSZ1_RSZ1
Status: Status.VERIFIED
Output dim: 8, lower bound: -0.0001240, upper bound: 0.0001982
time: 0.53 seconds

### Relational analysis RSZ of RS_RSZ2_RSZ2_RSZ1_RSZ1_RSZ1_RSZ1_RSZ1_RSZ2

#### Relational analysis RSZ result of RS_RSZ2_RSZ2_RSZ1_RSZ1_RSZ1_RSZ1_RSZ1_RSZ2
Status: Status.VERIFIED
Output dim: 8, lower bound: -0.0001416, upper bound: 0.0001411
time: 0.55 seconds

## BFS RS instance: RS_RSZ2_RSZ2_RSZ1_RSZ1_RSZ1_RSZ1_RSZ2

### Backsubstitution after applying RS history:
0: 0.0035972, 0.0041150, 0.0035972, 0.0041150, -0.0001289, 0.0001309
1: 0.0018420, 0.0019168, 0.0018420, 0.0019168, -0.0000186, 0.0000189
2: 0.0120848, 0.0123711, 0.0120848, 0.0123711, -0.0000724, 0.0000713
3: -0.0021818, -0.0018857, -0.0021818, -0.0018857, -0.0000749, 0.0000737
4: -0.0019956, -0.0016750, -0.0019956, -0.0016750, -0.0000798, 0.0000810
5: 0.0056985, 0.0060018, 0.0056985, 0.0060018, -0.0000767, 0.0000755
6: 0.0003095, 0.0015131, 0.0003095, 0.0015131, -0.0003043, 0.0002997
7: -0.0046174, -0.0029782, -0.0046174, -0.0029782, -0.0004081, 0.0004144
8: 0.9859612, 0.9871160, 0.9859612, 0.9871160, -0.0002875, 0.0002919
9: -0.0041920, -0.0031438, -0.0041920, -0.0031438, -0.0002650, 0.0002610

### Unstable ReLU Count (Linear/Conv2D Layers)
- layer_idx=0, type=LayerType.Linear, total=256, inp1_unstable=4, inp2_unstable=4, delta_unstable=252
- layer_idx=2, type=LayerType.Linear, total=256, inp1_unstable=4, inp2_unstable=4, delta_unstable=255
- layer_idx=4, type=LayerType.Linear, total=256, inp1_unstable=0, inp2_unstable=0, delta_unstable=256
- layer_idx=6, type=LayerType.Linear, total=256, inp1_unstable=0, inp2_unstable=0, delta_unstable=256
- layer_idx=8, type=LayerType.Linear, total=10, inp1_unstable=0, inp2_unstable=0, delta_unstable=10

Time for backsubstitution: 1.57 seconds

### RS candidates at layer 3
type: RSZ, layer: 3, pos: 34
type: RSZ, layer: 3, pos: 49
type: RSZ, layer: 3, pos: 83

Time for candidate selection: 0.00 seconds

### Candidate
type: RSZ, layer: 3, pos: 34

### Candidate
type: RSZ, layer: 3, pos: 49

### Relational analysis RSZ of RS_RSZ2_RSZ2_RSZ1_RSZ1_RSZ1_RSZ1_RSZ2_RSZ1

#### Relational analysis RSZ result of RS_RSZ2_RSZ2_RSZ1_RSZ1_RSZ1_RSZ1_RSZ2_RSZ1
Status: Status.VERIFIED
Output dim: 8, lower bound: -0.0001212, upper bound: 0.0001982
time: 0.57 seconds

### Relational analysis RSZ of RS_RSZ2_RSZ2_RSZ1_RSZ1_RSZ1_RSZ1_RSZ2_RSZ2

#### Relational analysis RSZ result of RS_RSZ2_RSZ2_RSZ1_RSZ1_RSZ1_RSZ1_RSZ2_RSZ2
Status: Status.VERIFIED
Output dim: 8, lower bound: -0.0001384, upper bound: 0.0001424
time: 0.56 seconds

## BFS RS instance: RS_RSZ2_RSZ2_RSZ1_RSZ1_RSZ1_RSZ2_RSZ1

### Backsubstitution after applying RS history:
0: 0.0035972, 0.0041150, 0.0035972, 0.0041150, -0.0001286, 0.0001311
1: 0.0018420, 0.0019168, 0.0018420, 0.0019168, -0.0000186, 0.0000189
2: 0.0120848, 0.0123711, 0.0120848, 0.0123711, -0.0000725, 0.0000711
3: -0.0021818, -0.0018857, -0.0021818, -0.0018857, -0.0000750, 0.0000735
4: -0.0019956, -0.0016750, -0.0019956, -0.0016750, -0.0000796, 0.0000811
5: 0.0056985, 0.0060018, 0.0056985, 0.0060018, -0.0000768, 0.0000753
6: 0.0003095, 0.0015131, 0.0003095, 0.0015131, -0.0003047, 0.0002989
7: -0.0046174, -0.0029782, -0.0046174, -0.0029782, -0.0004071, 0.0004150
8: 0.9859612, 0.9871160, 0.9859612, 0.9871160, -0.0002868, 0.0002923
9: -0.0041920, -0.0031438, -0.0041920, -0.0031438, -0.0002653, 0.0002603

### Unstable ReLU Count (Linear/Conv2D Layers)
- layer_idx=0, type=LayerType.Linear, total=256, inp1_unstable=4, inp2_unstable=4, delta_unstable=252
- layer_idx=2, type=LayerType.Linear, total=256, inp1_unstable=4, inp2_unstable=4, delta_unstable=255
- layer_idx=4, type=LayerType.Linear, total=256, inp1_unstable=0, inp2_unstable=0, delta_unstable=256
- layer_idx=6, type=LayerType.Linear, total=256, inp1_unstable=0, inp2_unstable=0, delta_unstable=256
- layer_idx=8, type=LayerType.Linear, total=10, inp1_unstable=0, inp2_unstable=0, delta_unstable=10

Time for backsubstitution: 1.57 seconds

### RS candidates at layer 3
type: RSZ, layer: 3, pos: 34
type: RSZ, layer: 3, pos: 83
type: RSZ, layer: 3, pos: 49

Time for candidate selection: 0.00 seconds

### Candidate
type: RSZ, layer: 3, pos: 34

### Candidate
type: RSZ, layer: 3, pos: 83

### Candidate
type: RSZ, layer: 3, pos: 49

### Relational analysis RSZ of RS_RSZ2_RSZ2_RSZ1_RSZ1_RSZ1_RSZ2_RSZ1_RSZ1

#### Relational analysis RSZ result of RS_RSZ2_RSZ2_RSZ1_RSZ1_RSZ1_RSZ2_RSZ1_RSZ1
Status: Status.VERIFIED
Output dim: 8, lower bound: -0.0001240, upper bound: 0.0002027
time: 0.53 seconds

### Relational analysis RSZ of RS_RSZ2_RSZ2_RSZ1_RSZ1_RSZ1_RSZ2_RSZ1_RSZ2

#### Relational analysis RSZ result of RS_RSZ2_RSZ2_RSZ1_RSZ1_RSZ1_RSZ2_RSZ1_RSZ2
Status: Status.VERIFIED
Output dim: 8, lower bound: -0.0001416, upper bound: 0.0001406
time: 0.54 seconds

## BFS RS instance: RS_RSZ2_RSZ2_RSZ1_RSZ1_RSZ1_RSZ2_RSZ2

### Backsubstitution after applying RS history:
0: 0.0035972, 0.0041150, 0.0035972, 0.0041150, -0.0001289, 0.0001309
1: 0.0018420, 0.0019168, 0.0018420, 0.0019168, -0.0000186, 0.0000189
2: 0.0120848, 0.0123711, 0.0120848, 0.0123711, -0.0000724, 0.0000713
3: -0.0021818, -0.0018857, -0.0021818, -0.0018857, -0.0000749, 0.0000737
4: -0.0019956, -0.0016750, -0.0019956, -0.0016750, -0.0000798, 0.0000810
5: 0.0056985, 0.0060018, 0.0056985, 0.0060018, -0.0000767, 0.0000755
6: 0.0003095, 0.0015131, 0.0003095, 0.0015131, -0.0003043, 0.0002997
7: -0.0046174, -0.0029782, -0.0046174, -0.0029782, -0.0004081, 0.0004144
8: 0.9859612, 0.9871160, 0.9859612, 0.9871160, -0.0002875, 0.0002919
9: -0.0041920, -0.0031438, -0.0041920, -0.0031438, -0.0002650, 0.0002610

### Unstable ReLU Count (Linear/Conv2D Layers)
- layer_idx=0, type=LayerType.Linear, total=256, inp1_unstable=4, inp2_unstable=4, delta_unstable=252
- layer_idx=2, type=LayerType.Linear, total=256, inp1_unstable=4, inp2_unstable=4, delta_unstable=255
- layer_idx=4, type=LayerType.Linear, total=256, inp1_unstable=0, inp2_unstable=0, delta_unstable=256
- layer_idx=6, type=LayerType.Linear, total=256, inp1_unstable=0, inp2_unstable=0, delta_unstable=256
- layer_idx=8, type=LayerType.Linear, total=10, inp1_unstable=0, inp2_unstable=0, delta_unstable=10

Time for backsubstitution: 1.50 seconds

### RS candidates at layer 3
type: RSZ, layer: 3, pos: 83
type: RSZ, layer: 3, pos: 49
type: RSZ, layer: 3, pos: 34

Time for candidate selection: 0.00 seconds

### Candidate
type: RSZ, layer: 3, pos: 83

### Candidate
type: RSZ, layer: 3, pos: 49

### Relational analysis RSZ of RS_RSZ2_RSZ2_RSZ1_RSZ1_RSZ1_RSZ2_RSZ2_RSZ1

#### Relational analysis RSZ result of RS_RSZ2_RSZ2_RSZ1_RSZ1_RSZ1_RSZ2_RSZ2_RSZ1
Status: Status.VERIFIED
Output dim: 8, lower bound: -0.0001201, upper bound: 0.0002023
time: 0.56 seconds

### Relational analysis RSZ of RS_RSZ2_RSZ2_RSZ1_RSZ1_RSZ1_RSZ2_RSZ2_RSZ2

#### Relational analysis RSZ result of RS_RSZ2_RSZ2_RSZ1_RSZ1_RSZ1_RSZ2_RSZ2_RSZ2
Status: Status.VERIFIED
Output dim: 8, lower bound: -0.0001383, upper bound: 0.0001408
time: 0.56 seconds

## BFS RS instance: RS_RSZ2_RSZ2_RSZ1_RSZ2_RSZ1_RSZ2_RSZ1

### Backsubstitution after applying RS history:
0: 0.0035972, 0.0041150, 0.0035972, 0.0041150, -0.0001191, 0.0001150
1: 0.0018420, 0.0019168, 0.0018420, 0.0019168, -0.0000172, 0.0000166
2: 0.0120848, 0.0123711, 0.0120848, 0.0123711, -0.0000636, 0.0000658
3: -0.0021818, -0.0018857, -0.0021818, -0.0018857, -0.0000658, 0.0000681
4: -0.0019956, -0.0016750, -0.0019956, -0.0016750, -0.0000737, 0.0000712
5: 0.0056985, 0.0060018, 0.0056985, 0.0060018, -0.0000674, 0.0000698
6: 0.0003095, 0.0015131, 0.0003095, 0.0015131, -0.0002673, 0.0002768
7: -0.0046174, -0.0029782, -0.0046174, -0.0029782, -0.0003769, 0.0003640
8: 0.9859612, 0.9871160, 0.9859612, 0.9871160, -0.0002655, 0.0002564
9: -0.0041920, -0.0031438, -0.0041920, -0.0031438, -0.0002328, 0.0002410

### Unstable ReLU Count (Linear/Conv2D Layers)
- layer_idx=0, type=LayerType.Linear, total=256, inp1_unstable=4, inp2_unstable=4, delta_unstable=252
- layer_idx=2, type=LayerType.Linear, total=256, inp1_unstable=4, inp2_unstable=4, delta_unstable=255
- layer_idx=4, type=LayerType.Linear, total=256, inp1_unstable=0, inp2_unstable=0, delta_unstable=256
- layer_idx=6, type=LayerType.Linear, total=256, inp1_unstable=0, inp2_unstable=0, delta_unstable=256
- layer_idx=8, type=LayerType.Linear, total=10, inp1_unstable=0, inp2_unstable=0, delta_unstable=10

Time for backsubstitution: 1.51 seconds

### RS candidates at layer 3
type: RSZ, layer: 3, pos: 240
type: RSZ, layer: 3, pos: 49
type: RSZ, layer: 3, pos: 83

Time for candidate selection: 0.00 seconds

### Candidate
type: RSZ, layer: 3, pos: 240

### Candidate
type: RSZ, layer: 3, pos: 49

### Relational analysis RSZ of RS_RSZ2_RSZ2_RSZ1_RSZ2_RSZ1_RSZ2_RSZ1_RSZ1

#### Relational analysis RSZ result of RS_RSZ2_RSZ2_RSZ1_RSZ2_RSZ1_RSZ2_RSZ1_RSZ1
Status: Status.VERIFIED
Output dim: 8, lower bound: -0.0001372, upper bound: 0.0001531
time: 0.51 seconds

### Relational analysis RSZ of RS_RSZ2_RSZ2_RSZ1_RSZ2_RSZ1_RSZ2_RSZ1_RSZ2

#### Relational analysis RSZ result of RS_RSZ2_RSZ2_RSZ1_RSZ2_RSZ1_RSZ2_RSZ1_RSZ2
Status: Status.VERIFIED
Output dim: 8, lower bound: -0.0001939, upper bound: 0.0001261
time: 0.53 seconds

## BFS RS instance: RS_RSZ2_RSZ2_RSZ1_RSZ2_RSZ1_RSZ2_RSZ2

### Backsubstitution after applying RS history:
0: 0.0035972, 0.0041150, 0.0035972, 0.0041150, -0.0001182, 0.0001161
1: 0.0018420, 0.0019168, 0.0018420, 0.0019168, -0.0000171, 0.0000168
2: 0.0120848, 0.0123711, 0.0120848, 0.0123711, -0.0000642, 0.0000654
3: -0.0021818, -0.0018857, -0.0021818, -0.0018857, -0.0000664, 0.0000676
4: -0.0019956, -0.0016750, -0.0019956, -0.0016750, -0.0000732, 0.0000719
5: 0.0056985, 0.0060018, 0.0056985, 0.0060018, -0.0000680, 0.0000693
6: 0.0003095, 0.0015131, 0.0003095, 0.0015131, -0.0002698, 0.0002748
7: -0.0046174, -0.0029782, -0.0046174, -0.0029782, -0.0003742, 0.0003675
8: 0.9859612, 0.9871160, 0.9859612, 0.9871160, -0.0002636, 0.0002589
9: -0.0041920, -0.0031438, -0.0041920, -0.0031438, -0.0002350, 0.0002393

### Unstable ReLU Count (Linear/Conv2D Layers)
- layer_idx=0, type=LayerType.Linear, total=256, inp1_unstable=4, inp2_unstable=4, delta_unstable=252
- layer_idx=2, type=LayerType.Linear, total=256, inp1_unstable=4, inp2_unstable=4, delta_unstable=255
- layer_idx=4, type=LayerType.Linear, total=256, inp1_unstable=0, inp2_unstable=0, delta_unstable=256
- layer_idx=6, type=LayerType.Linear, total=256, inp1_unstable=0, inp2_unstable=0, delta_unstable=256
- layer_idx=8, type=LayerType.Linear, total=10, inp1_unstable=0, inp2_unstable=0, delta_unstable=10

Time for backsubstitution: 1.48 seconds

### RS candidates at layer 3
type: RSZ, layer: 3, pos: 49
type: RSZ, layer: 3, pos: 240
type: RSZ, layer: 3, pos: 83

Time for candidate selection: 0.00 seconds

### Candidate
type: RSZ, layer: 3, pos: 49

### Relational analysis RSZ of RS_RSZ2_RSZ2_RSZ1_RSZ2_RSZ1_RSZ2_RSZ2_RSZ1

#### Relational analysis RSZ result of RS_RSZ2_RSZ2_RSZ1_RSZ2_RSZ1_RSZ2_RSZ2_RSZ1
Status: Status.VERIFIED
Output dim: 8, lower bound: -0.0001377, upper bound: 0.0001530
time: 0.50 seconds

### Relational analysis RSZ of RS_RSZ2_RSZ2_RSZ1_RSZ2_RSZ1_RSZ2_RSZ2_RSZ2

#### Relational analysis RSZ result of RS_RSZ2_RSZ2_RSZ1_RSZ2_RSZ1_RSZ2_RSZ2_RSZ2
Status: Status.VERIFIED
Output dim: 8, lower bound: -0.0001907, upper bound: 0.0001254
time: 0.51 seconds

## BFS RS instance: RS_RSZ2_RSZ2_RSZ1_RSZ2_RSZ2_RSZ1_RSZ1

### Backsubstitution after applying RS history:
0: 0.0035972, 0.0041150, 0.0035972, 0.0041150, -0.0001191, 0.0001150
1: 0.0018420, 0.0019168, 0.0018420, 0.0019168, -0.0000172, 0.0000166
2: 0.0120848, 0.0123711, 0.0120848, 0.0123711, -0.0000636, 0.0000658
3: -0.0021818, -0.0018857, -0.0021818, -0.0018857, -0.0000658, 0.0000681
4: -0.0019956, -0.0016750, -0.0019956, -0.0016750, -0.0000737, 0.0000712
5: 0.0056985, 0.0060018, 0.0056985, 0.0060018, -0.0000674, 0.0000698
6: 0.0003095, 0.0015131, 0.0003095, 0.0015131, -0.0002673, 0.0002768
7: -0.0046174, -0.0029782, -0.0046174, -0.0029782, -0.0003769, 0.0003640
8: 0.9859612, 0.9871160, 0.9859612, 0.9871160, -0.0002655, 0.0002564
9: -0.0041920, -0.0031438, -0.0041920, -0.0031438, -0.0002328, 0.0002410

### Unstable ReLU Count (Linear/Conv2D Layers)
- layer_idx=0, type=LayerType.Linear, total=256, inp1_unstable=4, inp2_unstable=4, delta_unstable=252
- layer_idx=2, type=LayerType.Linear, total=256, inp1_unstable=4, inp2_unstable=4, delta_unstable=255
- layer_idx=4, type=LayerType.Linear, total=256, inp1_unstable=0, inp2_unstable=0, delta_unstable=256
- layer_idx=6, type=LayerType.Linear, total=256, inp1_unstable=0, inp2_unstable=0, delta_unstable=256
- layer_idx=8, type=LayerType.Linear, total=10, inp1_unstable=0, inp2_unstable=0, delta_unstable=10

Time for backsubstitution: 1.51 seconds

### RS candidates at layer 3
type: RSZ, layer: 3, pos: 49
type: RSZ, layer: 3, pos: 83
type: RSZ, layer: 3, pos: 240

Time for candidate selection: 0.00 seconds

### Candidate
type: RSZ, layer: 3, pos: 49

### Relational analysis RSZ of RS_RSZ2_RSZ2_RSZ1_RSZ2_RSZ2_RSZ1_RSZ1_RSZ1

#### Relational analysis RSZ result of RS_RSZ2_RSZ2_RSZ1_RSZ2_RSZ2_RSZ1_RSZ1_RSZ1
Status: Status.VERIFIED
Output dim: 8, lower bound: -0.0001244, upper bound: 0.0001933
time: 0.55 seconds

### Relational analysis RSZ of RS_RSZ2_RSZ2_RSZ1_RSZ2_RSZ2_RSZ1_RSZ1_RSZ2

#### Relational analysis RSZ result of RS_RSZ2_RSZ2_RSZ1_RSZ2_RSZ2_RSZ1_RSZ1_RSZ2
Status: Status.VERIFIED
Output dim: 8, lower bound: -0.0001541, upper bound: 0.0001386
time: 0.53 seconds

## BFS RS instance: RS_RSZ2_RSZ2_RSZ1_RSZ2_RSZ2_RSZ1_RSZ2

### Backsubstitution after applying RS history:
0: 0.0035972, 0.0041150, 0.0035972, 0.0041150, -0.0001182, 0.0001161
1: 0.0018420, 0.0019168, 0.0018420, 0.0019168, -0.0000171, 0.0000168
2: 0.0120848, 0.0123711, 0.0120848, 0.0123711, -0.0000642, 0.0000654
3: -0.0021818, -0.0018857, -0.0021818, -0.0018857, -0.0000664, 0.0000676
4: -0.0019956, -0.0016750, -0.0019956, -0.0016750, -0.0000732, 0.0000719
5: 0.0056985, 0.0060018, 0.0056985, 0.0060018, -0.0000680, 0.0000693
6: 0.0003095, 0.0015131, 0.0003095, 0.0015131, -0.0002698, 0.0002748
7: -0.0046174, -0.0029782, -0.0046174, -0.0029782, -0.0003742, 0.0003675
8: 0.9859612, 0.9871160, 0.9859612, 0.9871160, -0.0002636, 0.0002589
9: -0.0041920, -0.0031438, -0.0041920, -0.0031438, -0.0002350, 0.0002393

### Unstable ReLU Count (Linear/Conv2D Layers)
- layer_idx=0, type=LayerType.Linear, total=256, inp1_unstable=4, inp2_unstable=4, delta_unstable=252
- layer_idx=2, type=LayerType.Linear, total=256, inp1_unstable=4, inp2_unstable=4, delta_unstable=255
- layer_idx=4, type=LayerType.Linear, total=256, inp1_unstable=0, inp2_unstable=0, delta_unstable=256
- layer_idx=6, type=LayerType.Linear, total=256, inp1_unstable=0, inp2_unstable=0, delta_unstable=256
- layer_idx=8, type=LayerType.Linear, total=10, inp1_unstable=0, inp2_unstable=0, delta_unstable=10

Time for backsubstitution: 1.45 seconds

### RS candidates at layer 3
type: RSZ, layer: 3, pos: 49
type: RSZ, layer: 3, pos: 240
type: RSZ, layer: 3, pos: 83

Time for candidate selection: 0.00 seconds

### Candidate
type: RSZ, layer: 3, pos: 49

### Relational analysis RSZ of RS_RSZ2_RSZ2_RSZ1_RSZ2_RSZ2_RSZ1_RSZ2_RSZ1

#### Relational analysis RSZ result of RS_RSZ2_RSZ2_RSZ1_RSZ2_RSZ2_RSZ1_RSZ2_RSZ1
Status: Status.VERIFIED
Output dim: 8, lower bound: -0.0001230, upper bound: 0.0001967
time: 0.55 seconds

### Relational analysis RSZ of RS_RSZ2_RSZ2_RSZ1_RSZ2_RSZ2_RSZ1_RSZ2_RSZ2

#### Relational analysis RSZ result of RS_RSZ2_RSZ2_RSZ1_RSZ2_RSZ2_RSZ1_RSZ2_RSZ2
Status: Status.VERIFIED
Output dim: 8, lower bound: -0.0001539, upper bound: 0.0001375
time: 0.55 seconds

## BFS RS instance: RS_RSZ2_RSZ2_RSZ1_RSZ2_RSZ2_RSZ2_RSZ1

### Backsubstitution after applying RS history:
0: 0.0035972, 0.0041150, 0.0035972, 0.0041150, -0.0001191, 0.0001150
1: 0.0018420, 0.0019168, 0.0018420, 0.0019168, -0.0000172, 0.0000166
2: 0.0120848, 0.0123711, 0.0120848, 0.0123711, -0.0000636, 0.0000658
3: -0.0021818, -0.0018857, -0.0021818, -0.0018857, -0.0000658, 0.0000681
4: -0.0019956, -0.0016750, -0.0019956, -0.0016750, -0.0000737, 0.0000712
5: 0.0056985, 0.0060018, 0.0056985, 0.0060018, -0.0000674, 0.0000698
6: 0.0003095, 0.0015131, 0.0003095, 0.0015131, -0.0002673, 0.0002768
7: -0.0046174, -0.0029782, -0.0046174, -0.0029782, -0.0003769, 0.0003640
8: 0.9859612, 0.9871160, 0.9859612, 0.9871160, -0.0002655, 0.0002564
9: -0.0041920, -0.0031438, -0.0041920, -0.0031438, -0.0002328, 0.0002410

### Unstable ReLU Count (Linear/Conv2D Layers)
- layer_idx=0, type=LayerType.Linear, total=256, inp1_unstable=4, inp2_unstable=4, delta_unstable=252
- layer_idx=2, type=LayerType.Linear, total=256, inp1_unstable=4, inp2_unstable=4, delta_unstable=255
- layer_idx=4, type=LayerType.Linear, total=256, inp1_unstable=0, inp2_unstable=0, delta_unstable=256
- layer_idx=6, type=LayerType.Linear, total=256, inp1_unstable=0, inp2_unstable=0, delta_unstable=256
- layer_idx=8, type=LayerType.Linear, total=10, inp1_unstable=0, inp2_unstable=0, delta_unstable=10

Time for backsubstitution: 1.48 seconds

### RS candidates at layer 3
type: RSZ, layer: 3, pos: 49
type: RSZ, layer: 3, pos: 83
type: RSZ, layer: 3, pos: 240

Time for candidate selection: 0.00 seconds

### Candidate
type: RSZ, layer: 3, pos: 49

### Relational analysis RSZ of RS_RSZ2_RSZ2_RSZ1_RSZ2_RSZ2_RSZ2_RSZ1_RSZ1

#### Relational analysis RSZ result of RS_RSZ2_RSZ2_RSZ1_RSZ2_RSZ2_RSZ2_RSZ1_RSZ1
Status: Status.VERIFIED
Output dim: 8, lower bound: -0.0001323, upper bound: 0.0001532
time: 0.54 seconds

### Relational analysis RSZ of RS_RSZ2_RSZ2_RSZ1_RSZ2_RSZ2_RSZ2_RSZ1_RSZ2

#### Relational analysis RSZ result of RS_RSZ2_RSZ2_RSZ1_RSZ2_RSZ2_RSZ2_RSZ1_RSZ2
Status: Status.VERIFIED
Output dim: 8, lower bound: -0.0001941, upper bound: 0.0001301
time: 0.51 seconds

## BFS RS instance: RS_RSZ2_RSZ2_RSZ1_RSZ2_RSZ2_RSZ2_RSZ2

### Backsubstitution after applying RS history:
0: 0.0035972, 0.0041150, 0.0035972, 0.0041150, -0.0001182, 0.0001161
1: 0.0018420, 0.0019168, 0.0018420, 0.0019168, -0.0000171, 0.0000168
2: 0.0120848, 0.0123711, 0.0120848, 0.0123711, -0.0000642, 0.0000654
3: -0.0021818, -0.0018857, -0.0021818, -0.0018857, -0.0000664, 0.0000676
4: -0.0019956, -0.0016750, -0.0019956, -0.0016750, -0.0000732, 0.0000719
5: 0.0056985, 0.0060018, 0.0056985, 0.0060018, -0.0000680, 0.0000693
6: 0.0003095, 0.0015131, 0.0003095, 0.0015131, -0.0002698, 0.0002748
7: -0.0046174, -0.0029782, -0.0046174, -0.0029782, -0.0003742, 0.0003675
8: 0.9859612, 0.9871160, 0.9859612, 0.9871160, -0.0002636, 0.0002589
9: -0.0041920, -0.0031438, -0.0041920, -0.0031438, -0.0002350, 0.0002393

### Unstable ReLU Count (Linear/Conv2D Layers)
- layer_idx=0, type=LayerType.Linear, total=256, inp1_unstable=4, inp2_unstable=4, delta_unstable=252
- layer_idx=2, type=LayerType.Linear, total=256, inp1_unstable=4, inp2_unstable=4, delta_unstable=255
- layer_idx=4, type=LayerType.Linear, total=256, inp1_unstable=0, inp2_unstable=0, delta_unstable=256
- layer_idx=6, type=LayerType.Linear, total=256, inp1_unstable=0, inp2_unstable=0, delta_unstable=256
- layer_idx=8, type=LayerType.Linear, total=10, inp1_unstable=0, inp2_unstable=0, delta_unstable=10

Time for backsubstitution: 1.46 seconds

### RS candidates at layer 3
type: RSZ, layer: 3, pos: 49
type: RSZ, layer: 3, pos: 240
type: RSZ, layer: 3, pos: 83

Time for candidate selection: 0.00 seconds

### Candidate
type: RSZ, layer: 3, pos: 49

### Relational analysis RSZ of RS_RSZ2_RSZ2_RSZ1_RSZ2_RSZ2_RSZ2_RSZ2_RSZ1

#### Relational analysis RSZ result of RS_RSZ2_RSZ2_RSZ1_RSZ2_RSZ2_RSZ2_RSZ2_RSZ1
Status: Status.VERIFIED
Output dim: 8, lower bound: -0.0001292, upper bound: 0.0001531
time: 0.51 seconds

### Relational analysis RSZ of RS_RSZ2_RSZ2_RSZ1_RSZ2_RSZ2_RSZ2_RSZ2_RSZ2

#### Relational analysis RSZ result of RS_RSZ2_RSZ2_RSZ1_RSZ2_RSZ2_RSZ2_RSZ2_RSZ2
Status: Status.VERIFIED
Output dim: 8, lower bound: -0.0001896, upper bound: 0.0001276
time: 0.53 seconds

## BFS RS instance: RS_RSZ2_RSZ2_RSZ2_RSZ2_RSZ2_RSZ1_RSZ2

### Backsubstitution after applying RS history:
0: 0.0035972, 0.0041150, 0.0035972, 0.0041150, -0.0001321, 0.0001157
1: 0.0018420, 0.0019168, 0.0018420, 0.0019168, -0.0000191, 0.0000167
2: 0.0120848, 0.0123711, 0.0120848, 0.0123711, -0.0000640, 0.0000730
3: -0.0021818, -0.0018857, -0.0021818, -0.0018857, -0.0000662, 0.0000755
4: -0.0019956, -0.0016750, -0.0019956, -0.0016750, -0.0000817, 0.0000716
5: 0.0056985, 0.0060018, 0.0056985, 0.0060018, -0.0000678, 0.0000774
6: 0.0003095, 0.0015131, 0.0003095, 0.0015131, -0.0002690, 0.0003069
7: -0.0046174, -0.0029782, -0.0046174, -0.0029782, -0.0004180, 0.0003664
8: 0.9859612, 0.9871160, 0.9859612, 0.9871160, -0.0002945, 0.0002581
9: -0.0041920, -0.0031438, -0.0041920, -0.0031438, -0.0002343, 0.0002673

### Unstable ReLU Count (Linear/Conv2D Layers)
- layer_idx=0, type=LayerType.Linear, total=256, inp1_unstable=4, inp2_unstable=4, delta_unstable=252
- layer_idx=2, type=LayerType.Linear, total=256, inp1_unstable=4, inp2_unstable=4, delta_unstable=255
- layer_idx=4, type=LayerType.Linear, total=256, inp1_unstable=0, inp2_unstable=0, delta_unstable=256
- layer_idx=6, type=LayerType.Linear, total=256, inp1_unstable=0, inp2_unstable=0, delta_unstable=256
- layer_idx=8, type=LayerType.Linear, total=10, inp1_unstable=0, inp2_unstable=0, delta_unstable=10

Time for backsubstitution: 1.46 seconds

### RS candidates at layer 3
type: RSZ, layer: 3, pos: 34
type: RSZ, layer: 3, pos: 240
type: RSZ, layer: 3, pos: 83

Time for candidate selection: 0.00 seconds

### Candidate
type: RSZ, layer: 3, pos: 34

### Candidate
type: RSZ, layer: 3, pos: 240

### Candidate
type: RSZ, layer: 3, pos: 83

### Relational analysis RSZ of RS_RSZ2_RSZ2_RSZ2_RSZ2_RSZ2_RSZ1_RSZ2_RSZ1

#### Relational analysis RSZ result of RS_RSZ2_RSZ2_RSZ2_RSZ2_RSZ2_RSZ1_RSZ2_RSZ1
Status: Status.VERIFIED
Output dim: 8, lower bound: -0.0001624, upper bound: 0.0001387
time: 0.52 seconds

### Relational analysis RSZ of RS_RSZ2_RSZ2_RSZ2_RSZ2_RSZ2_RSZ1_RSZ2_RSZ2

#### Relational analysis RSZ result of RS_RSZ2_RSZ2_RSZ2_RSZ2_RSZ2_RSZ1_RSZ2_RSZ2
Status: Status.VERIFIED
Output dim: 8, lower bound: -0.0002027, upper bound: 0.0001269
time: 0.50 seconds

## BFS RS instance: RS_RSZ2_RSZ2_RSZ2_RSZ2_RSZ2_RSZ2_RSZ2

### Backsubstitution after applying RS history:
0: 0.0035972, 0.0041150, 0.0035972, 0.0041150, -0.0001333, 0.0001246
1: 0.0018420, 0.0019168, 0.0018420, 0.0019168, -0.0000193, 0.0000180
2: 0.0120848, 0.0123711, 0.0120848, 0.0123711, -0.0000689, 0.0000737
3: -0.0021818, -0.0018857, -0.0021818, -0.0018857, -0.0000712, 0.0000762
4: -0.0019956, -0.0016750, -0.0019956, -0.0016750, -0.0000825, 0.0000771
5: 0.0056985, 0.0060018, 0.0056985, 0.0060018, -0.0000730, 0.0000781
6: 0.0003095, 0.0015131, 0.0003095, 0.0015131, -0.0002896, 0.0003097
7: -0.0046174, -0.0029782, -0.0046174, -0.0029782, -0.0004218, 0.0003944
8: 0.9859612, 0.9871160, 0.9859612, 0.9871160, -0.0002972, 0.0002778
9: -0.0041920, -0.0031438, -0.0041920, -0.0031438, -0.0002522, 0.0002697

### Unstable ReLU Count (Linear/Conv2D Layers)
- layer_idx=0, type=LayerType.Linear, total=256, inp1_unstable=4, inp2_unstable=4, delta_unstable=252
- layer_idx=2, type=LayerType.Linear, total=256, inp1_unstable=4, inp2_unstable=4, delta_unstable=255
- layer_idx=4, type=LayerType.Linear, total=256, inp1_unstable=0, inp2_unstable=0, delta_unstable=256
- layer_idx=6, type=LayerType.Linear, total=256, inp1_unstable=0, inp2_unstable=0, delta_unstable=256
- layer_idx=8, type=LayerType.Linear, total=10, inp1_unstable=0, inp2_unstable=0, delta_unstable=10

Time for backsubstitution: 1.45 seconds

### RS candidates at layer 3
type: RSZ, layer: 3, pos: 34
type: RSZ, layer: 3, pos: 240
type: RSZ, layer: 3, pos: 49

Time for candidate selection: 0.00 seconds

### Candidate
type: RSZ, layer: 3, pos: 34

### Candidate
type: RSZ, layer: 3, pos: 240

### Candidate
type: RSZ, layer: 3, pos: 49

### Relational analysis RSZ of RS_RSZ2_RSZ2_RSZ2_RSZ2_RSZ2_RSZ2_RSZ2_RSZ1

#### Relational analysis RSZ result of RS_RSZ2_RSZ2_RSZ2_RSZ2_RSZ2_RSZ2_RSZ2_RSZ1
Status: Status.VERIFIED
Output dim: 8, lower bound: -0.0001289, upper bound: 0.0001420
time: 0.51 seconds

### Relational analysis RSZ of RS_RSZ2_RSZ2_RSZ2_RSZ2_RSZ2_RSZ2_RSZ2_RSZ2

#### Relational analysis RSZ result of RS_RSZ2_RSZ2_RSZ2_RSZ2_RSZ2_RSZ2_RSZ2_RSZ2
Status: Status.VERIFIED
Output dim: 8, lower bound: -0.0001985, upper bound: 0.0001258
time: 0.52 seconds

## Summary of splitting (split count: 7)
- Time for RS candidates: 2.48 seconds
RS_RSZ1_RSZ1_RSZ1_RSZ1_RSZ1_RSZ2_RSZ2_RSZ1, status: Status.VERIFIED, split count: 8, time: 2.48
Output dim: 8, lower bound: -0.0001380, upper bound: 0.0001741
RS_RSZ1_RSZ1_RSZ1_RSZ1_RSZ1_RSZ2_RSZ2_RSZ2, status: Status.VERIFIED, split count: 8, time: 2.48
Output dim: 8, lower bound: -0.0001932, upper bound: 0.0001374
RS_RSZ1_RSZ1_RSZ1_RSZ2_RSZ1_RSZ1_RSZ1_RSZ1, status: Status.VERIFIED, split count: 8, time: 2.48
Output dim: 8, lower bound: -0.0001352, upper bound: 0.0001936
RS_RSZ1_RSZ1_RSZ1_RSZ2_RSZ1_RSZ1_RSZ1_RSZ2, status: Status.VERIFIED, split count: 8, time: 2.48
Output dim: 8, lower bound: -0.0001621, upper bound: 0.0001285
RS_RSZ1_RSZ1_RSZ1_RSZ2_RSZ1_RSZ1_RSZ2_RSZ1, status: Status.VERIFIED, split count: 8, time: 2.48
Output dim: 8, lower bound: -0.0001409, upper bound: 0.0001532
RS_RSZ1_RSZ1_RSZ1_RSZ2_RSZ1_RSZ1_RSZ2_RSZ2, status: Status.VERIFIED, split count: 8, time: 2.48
Output dim: 8, lower bound: -0.0001942, upper bound: 0.0001213
RS_RSZ1_RSZ1_RSZ1_RSZ2_RSZ1_RSZ2_RSZ2_RSZ1, status: Status.VERIFIED, split count: 8, time: 2.48
Output dim: 8, lower bound: -0.0001570, upper bound: 0.0001322
RS_RSZ1_RSZ1_RSZ1_RSZ2_RSZ1_RSZ2_RSZ2_RSZ2, status: Status.VERIFIED, split count: 8, time: 2.48
Output dim: 8, lower bound: -0.0001952, upper bound: 0.0001259
RS_RSZ1_RSZ1_RSZ2_RSZ2_RSZ2_RSZ2_RSZ1_RSZ1, status: Status.VERIFIED, split count: 8, time: 2.48
Output dim: 8, lower bound: -0.0001670, upper bound: 0.0001326
RS_RSZ1_RSZ1_RSZ2_RSZ2_RSZ2_RSZ2_RSZ1_RSZ2, status: Status.VERIFIED, split count: 8, time: 2.48
Output dim: 8, lower bound: -0.0002027, upper bound: 0.0001240
RS_RSZ1_RSZ1_RSZ2_RSZ2_RSZ2_RSZ2_RSZ2_RSZ1, status: Status.VERIFIED, split count: 8, time: 2.48
Output dim: 8, lower bound: -0.0001667, upper bound: 0.0001331
RS_RSZ1_RSZ1_RSZ2_RSZ2_RSZ2_RSZ2_RSZ2_RSZ2, status: Status.VERIFIED, split count: 8, time: 2.48
Output dim: 8, lower bound: -0.0001982, upper bound: 0.0001240
RS_RSZ1_RSZ2_RSZ1_RSZ1_RSZ1_RSZ1_RSZ1_RSZ1, status: Status.VERIFIED, split count: 8, time: 2.48
Output dim: 8, lower bound: -0.0001257, upper bound: 0.0001980
RS_RSZ1_RSZ2_RSZ1_RSZ1_RSZ1_RSZ1_RSZ1_RSZ2, status: Status.VERIFIED, split count: 8, time: 2.48
Output dim: 8, lower bound: -0.0001400, upper bound: 0.0001276
RS_RSZ1_RSZ2_RSZ1_RSZ1_RSZ1_RSZ1_RSZ2_RSZ1, status: Status.VERIFIED, split count: 8, time: 2.48
Output dim: 8, lower bound: -0.0001269, upper bound: 0.0002027
RS_RSZ1_RSZ2_RSZ1_RSZ1_RSZ1_RSZ1_RSZ2_RSZ2, status: Status.VERIFIED, split count: 8, time: 2.48
Output dim: 8, lower bound: -0.0001412, upper bound: 0.0001295
RS_RSZ1_RSZ2_RSZ1_RSZ1_RSZ1_RSZ2_RSZ1_RSZ1, status: Status.VERIFIED, split count: 8, time: 2.48
Output dim: 8, lower bound: -0.0001234, upper bound: 0.0001981
RS_RSZ1_RSZ2_RSZ1_RSZ1_RSZ1_RSZ2_RSZ1_RSZ2, status: Status.VERIFIED, split count: 8, time: 2.48
Output dim: 8, lower bound: -0.0001379, upper bound: 0.0001321
RS_RSZ1_RSZ2_RSZ1_RSZ1_RSZ1_RSZ2_RSZ2_RSZ1, status: Status.VERIFIED, split count: 8, time: 2.48
Output dim: 8, lower bound: -0.0001234, upper bound: 0.0002023
RS_RSZ1_RSZ2_RSZ1_RSZ1_RSZ1_RSZ2_RSZ2_RSZ2, status: Status.VERIFIED, split count: 8, time: 2.48
Output dim: 8, lower bound: -0.0001379, upper bound: 0.0001321
RS_RSZ1_RSZ2_RSZ1_RSZ2_RSZ2_RSZ1_RSZ1_RSZ1, status: Status.VERIFIED, split count: 8, time: 2.48
Output dim: 8, lower bound: -0.0001400, upper bound: 0.0001563
RS_RSZ1_RSZ2_RSZ1_RSZ2_RSZ2_RSZ1_RSZ1_RSZ2, status: Status.VERIFIED, split count: 8, time: 2.48
Output dim: 8, lower bound: -0.0001967, upper bound: 0.0001231
RS_RSZ1_RSZ2_RSZ1_RSZ2_RSZ2_RSZ1_RSZ2_RSZ1, status: Status.VERIFIED, split count: 8, time: 2.48
Output dim: 8, lower bound: -0.0001396, upper bound: 0.0001621
RS_RSZ1_RSZ2_RSZ1_RSZ2_RSZ2_RSZ1_RSZ2_RSZ2, status: Status.VERIFIED, split count: 8, time: 2.48
Output dim: 8, lower bound: -0.0001985, upper bound: 0.0001315
RS_RSZ1_RSZ2_RSZ2_RSZ1_RSZ1_RSZ2_RSZ1_RSZ1, status: Status.VERIFIED, split count: 8, time: 2.48
Output dim: 8, lower bound: -0.0001620, upper bound: 0.0001274
RS_RSZ1_RSZ2_RSZ2_RSZ1_RSZ1_RSZ2_RSZ1_RSZ2, status: Status.VERIFIED, split count: 8, time: 2.48
Output dim: 8, lower bound: -0.0001939, upper bound: 0.0001213
RS_RSZ1_RSZ2_RSZ2_RSZ1_RSZ2_RSZ2_RSZ1_RSZ1, status: Status.VERIFIED, split count: 8, time: 2.48
Output dim: 8, lower bound: -0.0001533, upper bound: 0.0001291
RS_RSZ1_RSZ2_RSZ2_RSZ1_RSZ2_RSZ2_RSZ1_RSZ2, status: Status.VERIFIED, split count: 8, time: 2.48
Output dim: 8, lower bound: -0.0001941, upper bound: 0.0001243
RS_RSZ1_RSZ2_RSZ2_RSZ2_RSZ2_RSZ1_RSZ1_RSZ1, status: Status.VERIFIED, split count: 8, time: 2.48
Output dim: 8, lower bound: -0.0001733, upper bound: 0.0001277
RS_RSZ1_RSZ2_RSZ2_RSZ2_RSZ2_RSZ1_RSZ1_RSZ2, status: Status.VERIFIED, split count: 8, time: 2.48
Output dim: 8, lower bound: -0.0002026, upper bound: 0.0001204
RS_RSZ1_RSZ2_RSZ2_RSZ2_RSZ2_RSZ1_RSZ2_RSZ1, status: Status.VERIFIED, split count: 8, time: 2.48
Output dim: 8, lower bound: -0.0001741, upper bound: 0.0001295
RS_RSZ1_RSZ2_RSZ2_RSZ2_RSZ2_RSZ1_RSZ2_RSZ2, status: Status.VERIFIED, split count: 8, time: 2.48
Output dim: 8, lower bound: -0.0001991, upper bound: 0.0001216
RS_RSZ1_RSZ2_RSZ2_RSZ2_RSZ2_RSZ2_RSZ1_RSZ1, status: Status.VERIFIED, split count: 8, time: 2.48
Output dim: 8, lower bound: -0.0001614, upper bound: 0.0001291
RS_RSZ1_RSZ2_RSZ2_RSZ2_RSZ2_RSZ2_RSZ1_RSZ2, status: Status.VERIFIED, split count: 8, time: 2.48
Output dim: 8, lower bound: -0.0002027, upper bound: 0.0001237
RS_RSZ1_RSZ2_RSZ2_RSZ2_RSZ2_RSZ2_RSZ2_RSZ1, status: Status.VERIFIED, split count: 8, time: 2.48
Output dim: 8, lower bound: -0.0001621, upper bound: 0.0001297
RS_RSZ1_RSZ2_RSZ2_RSZ2_RSZ2_RSZ2_RSZ2_RSZ2, status: Status.VERIFIED, split count: 8, time: 2.48
Output dim: 8, lower bound: -0.0001989, upper bound: 0.0001237
RS_RSZ2_RSZ1_RSZ1_RSZ1_RSZ1_RSZ1_RSZ1_RSZ1, status: Status.VERIFIED, split count: 8, time: 2.48
Output dim: 8, lower bound: -0.0001237, upper bound: 0.0001989
RS_RSZ2_RSZ1_RSZ1_RSZ1_RSZ1_RSZ1_RSZ1_RSZ2, status: Status.VERIFIED, split count: 8, time: 2.48
Output dim: 8, lower bound: -0.0001450, upper bound: 0.0001421
RS_RSZ2_RSZ1_RSZ1_RSZ1_RSZ1_RSZ1_RSZ2_RSZ1, status: Status.VERIFIED, split count: 8, time: 2.48
Output dim: 8, lower bound: -0.0001237, upper bound: 0.0002027
RS_RSZ2_RSZ1_RSZ1_RSZ1_RSZ1_RSZ1_RSZ2_RSZ2, status: Status.VERIFIED, split count: 8, time: 2.48
Output dim: 8, lower bound: -0.0001444, upper bound: 0.0001417
RS_RSZ2_RSZ1_RSZ1_RSZ2_RSZ1_RSZ1_RSZ1_RSZ1, status: Status.VERIFIED, split count: 8, time: 2.48
Output dim: 8, lower bound: -0.0001243, upper bound: 0.0001909
RS_RSZ2_RSZ1_RSZ1_RSZ2_RSZ1_RSZ1_RSZ1_RSZ2, status: Status.VERIFIED, split count: 8, time: 2.48
Output dim: 8, lower bound: -0.0001581, upper bound: 0.0001428
RS_RSZ2_RSZ1_RSZ1_RSZ2_RSZ1_RSZ1_RSZ2_RSZ1, status: Status.VERIFIED, split count: 8, time: 2.48
Output dim: 8, lower bound: -0.0001222, upper bound: 0.0001912
RS_RSZ2_RSZ1_RSZ1_RSZ2_RSZ1_RSZ1_RSZ2_RSZ2, status: Status.VERIFIED, split count: 8, time: 2.48
Output dim: 8, lower bound: -0.0001580, upper bound: 0.0001459
RS_RSZ2_RSZ1_RSZ1_RSZ2_RSZ1_RSZ2_RSZ1_RSZ1, status: Status.VERIFIED, split count: 8, time: 2.48
Output dim: 8, lower bound: -0.0001243, upper bound: 0.0001941
RS_RSZ2_RSZ1_RSZ1_RSZ2_RSZ1_RSZ2_RSZ1_RSZ2, status: Status.VERIFIED, split count: 8, time: 2.48
Output dim: 8, lower bound: -0.0001568, upper bound: 0.0001420
RS_RSZ2_RSZ1_RSZ1_RSZ2_RSZ1_RSZ2_RSZ2_RSZ1, status: Status.VERIFIED, split count: 8, time: 2.48
Output dim: 8, lower bound: -0.0001213, upper bound: 0.0001939
RS_RSZ2_RSZ1_RSZ1_RSZ2_RSZ1_RSZ2_RSZ2_RSZ2, status: Status.VERIFIED, split count: 8, time: 2.48
Output dim: 8, lower bound: -0.0001564, upper bound: 0.0001439
RS_RSZ2_RSZ1_RSZ1_RSZ2_RSZ2_RSZ2_RSZ1_RSZ1, status: Status.VERIFIED, split count: 8, time: 2.48
Output dim: 8, lower bound: -0.0001291, upper bound: 0.0001620
RS_RSZ2_RSZ1_RSZ1_RSZ2_RSZ2_RSZ2_RSZ1_RSZ2, status: Status.VERIFIED, split count: 8, time: 2.48
Output dim: 8, lower bound: -0.0001988, upper bound: 0.0001412
RS_RSZ2_RSZ1_RSZ1_RSZ2_RSZ2_RSZ2_RSZ2_RSZ1, status: Status.VERIFIED, split count: 8, time: 2.48
Output dim: 8, lower bound: -0.0001274, upper bound: 0.0001620
RS_RSZ2_RSZ1_RSZ1_RSZ2_RSZ2_RSZ2_RSZ2_RSZ2, status: Status.VERIFIED, split count: 8, time: 2.48
Output dim: 8, lower bound: -0.0001933, upper bound: 0.0001376
RS_RSZ2_RSZ1_RSZ2_RSZ1_RSZ1_RSZ2_RSZ1_RSZ1, status: Status.VERIFIED, split count: 8, time: 2.48
Output dim: 8, lower bound: -0.0001315, upper bound: 0.0001985
RS_RSZ2_RSZ1_RSZ2_RSZ1_RSZ1_RSZ2_RSZ1_RSZ2, status: Status.VERIFIED, split count: 8, time: 2.48
Output dim: 8, lower bound: -0.0001621, upper bound: 0.0001396
RS_RSZ2_RSZ1_RSZ2_RSZ1_RSZ1_RSZ2_RSZ2_RSZ1, status: Status.VERIFIED, split count: 8, time: 2.48
Output dim: 8, lower bound: -0.0001231, upper bound: 0.0001967
RS_RSZ2_RSZ1_RSZ2_RSZ1_RSZ1_RSZ2_RSZ2_RSZ2, status: Status.VERIFIED, split count: 8, time: 2.48
Output dim: 8, lower bound: -0.0001563, upper bound: 0.0001400
RS_RSZ2_RSZ1_RSZ2_RSZ1_RSZ2_RSZ2_RSZ1_RSZ1, status: Status.VERIFIED, split count: 8, time: 2.48
Output dim: 8, lower bound: -0.0001324, upper bound: 0.0001501
RS_RSZ2_RSZ1_RSZ2_RSZ1_RSZ2_RSZ2_RSZ1_RSZ2, status: Status.VERIFIED, split count: 8, time: 2.48
Output dim: 8, lower bound: -0.0001907, upper bound: 0.0001252
RS_RSZ2_RSZ1_RSZ2_RSZ1_RSZ2_RSZ2_RSZ2_RSZ1, status: Status.VERIFIED, split count: 8, time: 2.48
Output dim: 8, lower bound: -0.0001286, upper bound: 0.0001524
RS_RSZ2_RSZ1_RSZ2_RSZ1_RSZ2_RSZ2_RSZ2_RSZ2, status: Status.VERIFIED, split count: 8, time: 2.48
Output dim: 8, lower bound: -0.0001900, upper bound: 0.0001279
RS_RSZ2_RSZ1_RSZ2_RSZ2_RSZ2_RSZ1_RSZ2_RSZ1, status: Status.VERIFIED, split count: 8, time: 2.48
Output dim: 8, lower bound: -0.0001735, upper bound: 0.0001398
RS_RSZ2_RSZ1_RSZ2_RSZ2_RSZ2_RSZ1_RSZ2_RSZ2, status: Status.VERIFIED, split count: 8, time: 2.48
Output dim: 8, lower bound: -0.0001981, upper bound: 0.0001234
RS_RSZ2_RSZ1_RSZ2_RSZ2_RSZ2_RSZ2_RSZ2_RSZ1, status: Status.VERIFIED, split count: 8, time: 2.48
Output dim: 8, lower bound: -0.0001667, upper bound: 0.0001400
RS_RSZ2_RSZ1_RSZ2_RSZ2_RSZ2_RSZ2_RSZ2_RSZ2, status: Status.VERIFIED, split count: 8, time: 2.48
Output dim: 8, lower bound: -0.0001980, upper bound: 0.0001257
RS_RSZ2_RSZ2_RSZ1_RSZ1_RSZ1_RSZ1_RSZ1_RSZ1, status: Status.VERIFIED, split count: 8, time: 2.48
Output dim: 8, lower bound: -0.0001240, upper bound: 0.0001982
RS_RSZ2_RSZ2_RSZ1_RSZ1_RSZ1_RSZ1_RSZ1_RSZ2, status: Status.VERIFIED, split count: 8, time: 2.48
Output dim: 8, lower bound: -0.0001416, upper bound: 0.0001411
RS_RSZ2_RSZ2_RSZ1_RSZ1_RSZ1_RSZ1_RSZ2_RSZ1, status: Status.VERIFIED, split count: 8, time: 2.48
Output dim: 8, lower bound: -0.0001212, upper bound: 0.0001982
RS_RSZ2_RSZ2_RSZ1_RSZ1_RSZ1_RSZ1_RSZ2_RSZ2, status: Status.VERIFIED, split count: 8, time: 2.48
Output dim: 8, lower bound: -0.0001384, upper bound: 0.0001424
RS_RSZ2_RSZ2_RSZ1_RSZ1_RSZ1_RSZ2_RSZ1_RSZ1, status: Status.VERIFIED, split count: 8, time: 2.48
Output dim: 8, lower bound: -0.0001240, upper bound: 0.0002027
RS_RSZ2_RSZ2_RSZ1_RSZ1_RSZ1_RSZ2_RSZ1_RSZ2, status: Status.VERIFIED, split count: 8, time: 2.48
Output dim: 8, lower bound: -0.0001416, upper bound: 0.0001406
RS_RSZ2_RSZ2_RSZ1_RSZ1_RSZ1_RSZ2_RSZ2_RSZ1, status: Status.VERIFIED, split count: 8, time: 2.48
Output dim: 8, lower bound: -0.0001201, upper bound: 0.0002023
RS_RSZ2_RSZ2_RSZ1_RSZ1_RSZ1_RSZ2_RSZ2_RSZ2, status: Status.VERIFIED, split count: 8, time: 2.48
Output dim: 8, lower bound: -0.0001383, upper bound: 0.0001408
RS_RSZ2_RSZ2_RSZ1_RSZ2_RSZ1_RSZ2_RSZ1_RSZ1, status: Status.VERIFIED, split count: 8, time: 2.48
Output dim: 8, lower bound: -0.0001372, upper bound: 0.0001531
RS_RSZ2_RSZ2_RSZ1_RSZ2_RSZ1_RSZ2_RSZ1_RSZ2, status: Status.VERIFIED, split count: 8, time: 2.48
Output dim: 8, lower bound: -0.0001939, upper bound: 0.0001261
RS_RSZ2_RSZ2_RSZ1_RSZ2_RSZ1_RSZ2_RSZ2_RSZ1, status: Status.VERIFIED, split count: 8, time: 2.48
Output dim: 8, lower bound: -0.0001377, upper bound: 0.0001530
RS_RSZ2_RSZ2_RSZ1_RSZ2_RSZ1_RSZ2_RSZ2_RSZ2, status: Status.VERIFIED, split count: 8, time: 2.48
Output dim: 8, lower bound: -0.0001907, upper bound: 0.0001254
RS_RSZ2_RSZ2_RSZ1_RSZ2_RSZ2_RSZ1_RSZ1_RSZ1, status: Status.VERIFIED, split count: 8, time: 2.48
Output dim: 8, lower bound: -0.0001244, upper bound: 0.0001933
RS_RSZ2_RSZ2_RSZ1_RSZ2_RSZ2_RSZ1_RSZ1_RSZ2, status: Status.VERIFIED, split count: 8, time: 2.48
Output dim: 8, lower bound: -0.0001541, upper bound: 0.0001386
RS_RSZ2_RSZ2_RSZ1_RSZ2_RSZ2_RSZ1_RSZ2_RSZ1, status: Status.VERIFIED, split count: 8, time: 2.48
Output dim: 8, lower bound: -0.0001230, upper bound: 0.0001967
RS_RSZ2_RSZ2_RSZ1_RSZ2_RSZ2_RSZ1_RSZ2_RSZ2, status: Status.VERIFIED, split count: 8, time: 2.48
Output dim: 8, lower bound: -0.0001539, upper bound: 0.0001375
RS_RSZ2_RSZ2_RSZ1_RSZ2_RSZ2_RSZ2_RSZ1_RSZ1, status: Status.VERIFIED, split count: 8, time: 2.48
Output dim: 8, lower bound: -0.0001323, upper bound: 0.0001532
RS_RSZ2_RSZ2_RSZ1_RSZ2_RSZ2_RSZ2_RSZ1_RSZ2, status: Status.VERIFIED, split count: 8, time: 2.48
Output dim: 8, lower bound: -0.0001941, upper bound: 0.0001301
RS_RSZ2_RSZ2_RSZ1_RSZ2_RSZ2_RSZ2_RSZ2_RSZ1, status: Status.VERIFIED, split count: 8, time: 2.48
Output dim: 8, lower bound: -0.0001292, upper bound: 0.0001531
RS_RSZ2_RSZ2_RSZ1_RSZ2_RSZ2_RSZ2_RSZ2_RSZ2, status: Status.VERIFIED, split count: 8, time: 2.48
Output dim: 8, lower bound: -0.0001896, upper bound: 0.0001276
RS_RSZ2_RSZ2_RSZ2_RSZ2_RSZ2_RSZ1_RSZ2_RSZ1, status: Status.VERIFIED, split count: 8, time: 2.48
Output dim: 8, lower bound: -0.0001624, upper bound: 0.0001387
RS_RSZ2_RSZ2_RSZ2_RSZ2_RSZ2_RSZ1_RSZ2_RSZ2, status: Status.VERIFIED, split count: 8, time: 2.48
Output dim: 8, lower bound: -0.0002027, upper bound: 0.0001269
RS_RSZ2_RSZ2_RSZ2_RSZ2_RSZ2_RSZ2_RSZ2_RSZ1, status: Status.VERIFIED, split count: 8, time: 2.48
Output dim: 8, lower bound: -0.0001289, upper bound: 0.0001420
RS_RSZ2_RSZ2_RSZ2_RSZ2_RSZ2_RSZ2_RSZ2_RSZ2, status: Status.VERIFIED, split count: 8, time: 2.48
Output dim: 8, lower bound: -0.0001985, upper bound: 0.0001258

## RS Result
status: Status.VERIFIED
execution time: (base) + (rs) = 2.86 + 389.48 = 392.34 seconds
