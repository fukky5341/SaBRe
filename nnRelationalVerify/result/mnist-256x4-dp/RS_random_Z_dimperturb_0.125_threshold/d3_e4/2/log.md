## Execution arguments:
Dataset: Dataset.MNIST
Network: onnx/mnist-net_256x4.onnx
Relational property: GLOBAL_ROBUSTNESS
LP Analysis: True
Epsilon: 0.046875
Delta epsilon: 0.01171875
execution index: (3, 4, 2)
Time budget: 600 seconds
Split limit: 100
Threshold: 0.00061831


## IAR start

### Unstable ReLU Count (Linear/Conv2D Layers)
- layer_idx=0, type=LayerType.Linear, total=256, inp1_unstable=4, inp2_unstable=4, delta_unstable=256
- layer_idx=2, type=LayerType.Linear, total=256, inp1_unstable=8, inp2_unstable=8, delta_unstable=256
- layer_idx=4, type=LayerType.Linear, total=256, inp1_unstable=0, inp2_unstable=0, delta_unstable=256
- layer_idx=6, type=LayerType.Linear, total=256, inp1_unstable=0, inp2_unstable=0, delta_unstable=256
- layer_idx=8, type=LayerType.Linear, total=10, inp1_unstable=0, inp2_unstable=0, delta_unstable=10

### BASE IAR bounds
Layer (inp1_lb, inp1_ub, inp2_lb, inp2_ub, d_lb, d_ub)
0: (-0.0039650, -0.0038291, -0.0039650, -0.0038291, -0.0000723, 0.0000723)
1: (0.0009495, 0.0017023, 0.0009495, 0.0017023, -0.0004003, 0.0004003)
2: (0.0111630, 0.0128448, 0.0111630, 0.0128448, -0.0008943, 0.0008943)
3: (0.0019215, 0.0026303, 0.0019215, 0.0026303, -0.0003769, 0.0003769)
4: (1.0042051, 1.0069547, 1.0042051, 1.0069547, -0.0014621, 0.0014621)
5: (0.0030595, 0.0035944, 0.0030595, 0.0035944, -0.0002844, 0.0002844)
6: (-0.0104205, -0.0097244, -0.0104205, -0.0097244, -0.0003702, 0.0003702)
7: (-0.0101326, -0.0100438, -0.0101326, -0.0100438, -0.0000472, 0.0000472)
8: (-0.0041353, -0.0036544, -0.0041353, -0.0036544, -0.0002558, 0.0002558)
9: (0.0001238, 0.0025315, 0.0001238, 0.0025315, -0.0012804, 0.0012804)

## BASE Relational Analysis

## BASE Result
execution time: IAR + RelationalAnalysis = 1.98 + 1.37 = 3.35 seconds
status: Status.UNKNOWN
relational distance
Output dim: 4, lower bound: -0.0008288, upper bound: 0.0008288

# Relational Split (RS) starts

## BFS RS instance: RS

Time for backsubstitution: 0.01 seconds

### RS candidates at layer 1
type: RSZ, layer: 1, pos: 251
type: RSZ, layer: 1, pos: 155
type: RSZ, layer: 1, pos: 196
type: RSZ, layer: 1, pos: 86

Time for candidate selection: 0.00 seconds

### Candidate
type: RSZ, layer: 1, pos: 251

### Relational analysis RSZ of RS_RSZ1

#### Relational analysis RSZ result of RS_RSZ1
Status: Status.UNKNOWN
Output dim: 4, lower bound: -0.0008191, upper bound: 0.0008194
time: 0.50 seconds

### Relational analysis RSZ of RS_RSZ2

#### Relational analysis RSZ result of RS_RSZ2
Status: Status.UNKNOWN
Output dim: 4, lower bound: -0.0008194, upper bound: 0.0008191
time: 0.49 seconds

## Summary of splitting (split count: 0)
- Time for RS candidates: 1.00 seconds
RS_RSZ1, status: Status.UNKNOWN, split count: 1, time: 1.00
Output dim: 4, lower bound: -0.0008191, upper bound: 0.0008194
RS_RSZ2, status: Status.UNKNOWN, split count: 1, time: 1.00
Output dim: 4, lower bound: -0.0008194, upper bound: 0.0008191

## BFS RS instance: RS_RSZ1

### Backsubstitution after applying RS history:
0: -0.0039650, -0.0038291, -0.0039650, -0.0038291, -0.0000699, 0.0000699
1: 0.0009495, 0.0017023, 0.0009495, 0.0017023, -0.0003870, 0.0003872
2: 0.0111630, 0.0128448, 0.0111630, 0.0128448, -0.0008651, 0.0008647
3: 0.0019215, 0.0026303, 0.0019215, 0.0026303, -0.0003644, 0.0003646
4: 1.0042051, 1.0069547, 1.0042051, 1.0069547, -0.0014136, 0.0014144
5: 0.0030595, 0.0035944, 0.0030595, 0.0035944, -0.0002750, 0.0002751
6: -0.0104205, -0.0097244, -0.0104205, -0.0097244, -0.0003581, 0.0003579
7: -0.0101326, -0.0100438, -0.0101326, -0.0100438, -0.0000457, 0.0000457
8: -0.0041353, -0.0036544, -0.0041353, -0.0036544, -0.0002473, 0.0002474
9: 0.0001238, 0.0025315, 0.0001238, 0.0025315, -0.0012385, 0.0012379

### Unstable ReLU Count (Linear/Conv2D Layers)
- layer_idx=0, type=LayerType.Linear, total=256, inp1_unstable=4, inp2_unstable=4, delta_unstable=255
- layer_idx=2, type=LayerType.Linear, total=256, inp1_unstable=8, inp2_unstable=8, delta_unstable=256
- layer_idx=4, type=LayerType.Linear, total=256, inp1_unstable=0, inp2_unstable=0, delta_unstable=256
- layer_idx=6, type=LayerType.Linear, total=256, inp1_unstable=0, inp2_unstable=0, delta_unstable=256
- layer_idx=8, type=LayerType.Linear, total=10, inp1_unstable=0, inp2_unstable=0, delta_unstable=10

Time for backsubstitution: 1.66 seconds

### RS candidates at layer 1
type: RSZ, layer: 1, pos: 196
type: RSZ, layer: 1, pos: 155
type: RSZ, layer: 1, pos: 86

Time for candidate selection: 0.00 seconds

### Candidate
type: RSZ, layer: 1, pos: 196

### Relational analysis RSZ of RS_RSZ1_RSZ1

#### Relational analysis RSZ result of RS_RSZ1_RSZ1
Status: Status.UNKNOWN
Output dim: 4, lower bound: -0.0007913, upper bound: 0.0008127
time: 0.50 seconds

### Relational analysis RSZ of RS_RSZ1_RSZ2

#### Relational analysis RSZ result of RS_RSZ1_RSZ2
Status: Status.UNKNOWN
Output dim: 4, lower bound: -0.0008120, upper bound: 0.0007969
time: 0.48 seconds

## BFS RS instance: RS_RSZ2

### Backsubstitution after applying RS history:
0: -0.0039650, -0.0038291, -0.0039650, -0.0038291, -0.0000699, 0.0000699
1: 0.0009495, 0.0017023, 0.0009495, 0.0017023, -0.0003872, 0.0003870
2: 0.0111630, 0.0128448, 0.0111630, 0.0128448, -0.0008647, 0.0008651
3: 0.0019215, 0.0026303, 0.0019215, 0.0026303, -0.0003646, 0.0003644
4: 1.0042051, 1.0069547, 1.0042051, 1.0069547, -0.0014144, 0.0014136
5: 0.0030595, 0.0035944, 0.0030595, 0.0035944, -0.0002751, 0.0002750
6: -0.0104205, -0.0097244, -0.0104205, -0.0097244, -0.0003579, 0.0003581
7: -0.0101326, -0.0100438, -0.0101326, -0.0100438, -0.0000457, 0.0000457
8: -0.0041353, -0.0036544, -0.0041353, -0.0036544, -0.0002474, 0.0002473
9: 0.0001238, 0.0025315, 0.0001238, 0.0025315, -0.0012379, 0.0012385

### Unstable ReLU Count (Linear/Conv2D Layers)
- layer_idx=0, type=LayerType.Linear, total=256, inp1_unstable=4, inp2_unstable=4, delta_unstable=255
- layer_idx=2, type=LayerType.Linear, total=256, inp1_unstable=8, inp2_unstable=8, delta_unstable=256
- layer_idx=4, type=LayerType.Linear, total=256, inp1_unstable=0, inp2_unstable=0, delta_unstable=256
- layer_idx=6, type=LayerType.Linear, total=256, inp1_unstable=0, inp2_unstable=0, delta_unstable=256
- layer_idx=8, type=LayerType.Linear, total=10, inp1_unstable=0, inp2_unstable=0, delta_unstable=10

Time for backsubstitution: 1.67 seconds

### RS candidates at layer 1
type: RSZ, layer: 1, pos: 155
type: RSZ, layer: 1, pos: 196
type: RSZ, layer: 1, pos: 86

Time for candidate selection: 0.00 seconds

### Candidate
type: RSZ, layer: 1, pos: 155

### Relational analysis RSZ of RS_RSZ2_RSZ1

#### Relational analysis RSZ result of RS_RSZ2_RSZ1
Status: Status.UNKNOWN
Output dim: 4, lower bound: -0.0007877, upper bound: 0.0007826
time: 0.51 seconds

### Relational analysis RSZ of RS_RSZ2_RSZ2

#### Relational analysis RSZ result of RS_RSZ2_RSZ2
Status: Status.UNKNOWN
Output dim: 4, lower bound: -0.0007827, upper bound: 0.0007867
time: 0.51 seconds

## Summary of splitting (split count: 1)
- Time for RS candidates: 2.70 seconds
RS_RSZ1_RSZ1, status: Status.UNKNOWN, split count: 2, time: 2.70
Output dim: 4, lower bound: -0.0007913, upper bound: 0.0008127
RS_RSZ1_RSZ2, status: Status.UNKNOWN, split count: 2, time: 2.70
Output dim: 4, lower bound: -0.0008120, upper bound: 0.0007969
RS_RSZ2_RSZ1, status: Status.UNKNOWN, split count: 2, time: 2.70
Output dim: 4, lower bound: -0.0007877, upper bound: 0.0007826
RS_RSZ2_RSZ2, status: Status.UNKNOWN, split count: 2, time: 2.70
Output dim: 4, lower bound: -0.0007827, upper bound: 0.0007867

## BFS RS instance: RS_RSZ1_RSZ1

### Backsubstitution after applying RS history:
0: -0.0039650, -0.0038291, -0.0039650, -0.0038291, -0.0000688, 0.0000683
1: 0.0009495, 0.0017023, 0.0009495, 0.0017023, -0.0003780, 0.0003811
2: 0.0111630, 0.0128448, 0.0111630, 0.0128448, -0.0008514, 0.0008446
3: 0.0019215, 0.0026303, 0.0019215, 0.0026303, -0.0003559, 0.0003588
4: 1.0042051, 1.0069547, 1.0042051, 1.0069547, -0.0013808, 0.0013920
5: 0.0030595, 0.0035944, 0.0030595, 0.0035944, -0.0002686, 0.0002708
6: -0.0104205, -0.0097244, -0.0104205, -0.0097244, -0.0003524, 0.0003496
7: -0.0101326, -0.0100438, -0.0101326, -0.0100438, -0.0000450, 0.0000446
8: -0.0041353, -0.0036544, -0.0041353, -0.0036544, -0.0002415, 0.0002435
9: 0.0001238, 0.0025315, 0.0001238, 0.0025315, -0.0012189, 0.0012091

### Unstable ReLU Count (Linear/Conv2D Layers)
- layer_idx=0, type=LayerType.Linear, total=256, inp1_unstable=4, inp2_unstable=4, delta_unstable=254
- layer_idx=2, type=LayerType.Linear, total=256, inp1_unstable=8, inp2_unstable=8, delta_unstable=256
- layer_idx=4, type=LayerType.Linear, total=256, inp1_unstable=0, inp2_unstable=0, delta_unstable=256
- layer_idx=6, type=LayerType.Linear, total=256, inp1_unstable=0, inp2_unstable=0, delta_unstable=256
- layer_idx=8, type=LayerType.Linear, total=10, inp1_unstable=0, inp2_unstable=0, delta_unstable=10

Time for backsubstitution: 1.62 seconds

### RS candidates at layer 1
type: RSZ, layer: 1, pos: 155
type: RSZ, layer: 1, pos: 86

Time for candidate selection: 0.00 seconds

### Candidate
type: RSZ, layer: 1, pos: 155

### Relational analysis RSZ of RS_RSZ1_RSZ1_RSZ1

#### Relational analysis RSZ result of RS_RSZ1_RSZ1_RSZ1
Status: Status.UNKNOWN
Output dim: 4, lower bound: -0.0007626, upper bound: 0.0007767
time: 0.49 seconds

### Relational analysis RSZ of RS_RSZ1_RSZ1_RSZ2

#### Relational analysis RSZ result of RS_RSZ1_RSZ1_RSZ2
Status: Status.UNKNOWN
Output dim: 4, lower bound: -0.0007339, upper bound: 0.0007831
time: 0.51 seconds

## BFS RS instance: RS_RSZ1_RSZ2

### Backsubstitution after applying RS history:
0: -0.0039650, -0.0038291, -0.0039650, -0.0038291, -0.0000683, 0.0000688
1: 0.0009495, 0.0017023, 0.0009495, 0.0017023, -0.0003809, 0.0003783
2: 0.0111630, 0.0128448, 0.0111630, 0.0128448, -0.0008451, 0.0008510
3: 0.0019215, 0.0026303, 0.0019215, 0.0026303, -0.0003586, 0.0003561
4: 1.0042051, 1.0069547, 1.0042051, 1.0069547, -0.0013913, 0.0013817
5: 0.0030595, 0.0035944, 0.0030595, 0.0035944, -0.0002707, 0.0002688
6: -0.0104205, -0.0097244, -0.0104205, -0.0097244, -0.0003498, 0.0003522
7: -0.0101326, -0.0100438, -0.0101326, -0.0100438, -0.0000446, 0.0000449
8: -0.0041353, -0.0036544, -0.0041353, -0.0036544, -0.0002434, 0.0002417
9: 0.0001238, 0.0025315, 0.0001238, 0.0025315, -0.0012099, 0.0012183

### Unstable ReLU Count (Linear/Conv2D Layers)
- layer_idx=0, type=LayerType.Linear, total=256, inp1_unstable=4, inp2_unstable=4, delta_unstable=254
- layer_idx=2, type=LayerType.Linear, total=256, inp1_unstable=8, inp2_unstable=8, delta_unstable=256
- layer_idx=4, type=LayerType.Linear, total=256, inp1_unstable=0, inp2_unstable=0, delta_unstable=256
- layer_idx=6, type=LayerType.Linear, total=256, inp1_unstable=0, inp2_unstable=0, delta_unstable=256
- layer_idx=8, type=LayerType.Linear, total=10, inp1_unstable=0, inp2_unstable=0, delta_unstable=10

Time for backsubstitution: 1.65 seconds

### RS candidates at layer 1
type: RSZ, layer: 1, pos: 86
type: RSZ, layer: 1, pos: 155

Time for candidate selection: 0.00 seconds

### Candidate
type: RSZ, layer: 1, pos: 86

### Relational analysis RSZ of RS_RSZ1_RSZ2_RSZ1

#### Relational analysis RSZ result of RS_RSZ1_RSZ2_RSZ1
Status: Status.UNKNOWN
Output dim: 4, lower bound: -0.0006673, upper bound: 0.0006874
time: 0.48 seconds

### Relational analysis RSZ of RS_RSZ1_RSZ2_RSZ2

#### Relational analysis RSZ result of RS_RSZ1_RSZ2_RSZ2
Status: Status.UNKNOWN
Output dim: 4, lower bound: -0.0006947, upper bound: 0.0006614
time: 0.49 seconds

## BFS RS instance: RS_RSZ2_RSZ1

### Backsubstitution after applying RS history:
0: -0.0039650, -0.0038291, -0.0039650, -0.0038291, -0.0000692, 0.0000694
1: 0.0009495, 0.0017023, 0.0009495, 0.0017023, -0.0003843, 0.0003831
2: 0.0111630, 0.0128448, 0.0111630, 0.0128448, -0.0008559, 0.0008585
3: 0.0019215, 0.0026303, 0.0019215, 0.0026303, -0.0003618, 0.0003607
4: 1.0042051, 1.0069547, 1.0042051, 1.0069547, -0.0014035, 0.0013993
5: 0.0030595, 0.0035944, 0.0030595, 0.0035944, -0.0002730, 0.0002722
6: -0.0104205, -0.0097244, -0.0104205, -0.0097244, -0.0003543, 0.0003553
7: -0.0101326, -0.0100438, -0.0101326, -0.0100438, -0.0000452, 0.0000453
8: -0.0041353, -0.0036544, -0.0041353, -0.0036544, -0.0002455, 0.0002448
9: 0.0001238, 0.0025315, 0.0001238, 0.0025315, -0.0012254, 0.0012290

### Unstable ReLU Count (Linear/Conv2D Layers)
- layer_idx=0, type=LayerType.Linear, total=256, inp1_unstable=4, inp2_unstable=4, delta_unstable=254
- layer_idx=2, type=LayerType.Linear, total=256, inp1_unstable=8, inp2_unstable=8, delta_unstable=256
- layer_idx=4, type=LayerType.Linear, total=256, inp1_unstable=0, inp2_unstable=0, delta_unstable=256
- layer_idx=6, type=LayerType.Linear, total=256, inp1_unstable=0, inp2_unstable=0, delta_unstable=256
- layer_idx=8, type=LayerType.Linear, total=10, inp1_unstable=0, inp2_unstable=0, delta_unstable=10

Time for backsubstitution: 1.67 seconds

### RS candidates at layer 1
type: RSZ, layer: 1, pos: 196
type: RSZ, layer: 1, pos: 86

Time for candidate selection: 0.00 seconds

### Candidate
type: RSZ, layer: 1, pos: 196

### Relational analysis RSZ of RS_RSZ2_RSZ1_RSZ1

#### Relational analysis RSZ result of RS_RSZ2_RSZ1_RSZ1
Status: Status.UNKNOWN
Output dim: 4, lower bound: -0.0007667, upper bound: 0.0007763
time: 0.50 seconds

### Relational analysis RSZ of RS_RSZ2_RSZ1_RSZ2

#### Relational analysis RSZ result of RS_RSZ2_RSZ1_RSZ2
Status: Status.UNKNOWN
Output dim: 4, lower bound: -0.0007831, upper bound: 0.0007340
time: 0.48 seconds

## BFS RS instance: RS_RSZ2_RSZ2

### Backsubstitution after applying RS history:
0: -0.0039650, -0.0038291, -0.0039650, -0.0038291, -0.0000699, 0.0000692
1: 0.0009495, 0.0017023, 0.0009495, 0.0017023, -0.0003833, 0.0003870
2: 0.0111630, 0.0128448, 0.0111630, 0.0128448, -0.0008647, 0.0008564
3: 0.0019215, 0.0026303, 0.0019215, 0.0026303, -0.0003609, 0.0003644
4: 1.0042051, 1.0069547, 1.0042051, 1.0069547, -0.0014001, 0.0014136
5: 0.0030595, 0.0035944, 0.0030595, 0.0035944, -0.0002724, 0.0002750
6: -0.0104205, -0.0097244, -0.0104205, -0.0097244, -0.0003579, 0.0003544
7: -0.0101326, -0.0100438, -0.0101326, -0.0100438, -0.0000457, 0.0000452
8: -0.0041353, -0.0036544, -0.0041353, -0.0036544, -0.0002449, 0.0002473
9: 0.0001238, 0.0025315, 0.0001238, 0.0025315, -0.0012379, 0.0012260

### Unstable ReLU Count (Linear/Conv2D Layers)
- layer_idx=0, type=LayerType.Linear, total=256, inp1_unstable=4, inp2_unstable=4, delta_unstable=254
- layer_idx=2, type=LayerType.Linear, total=256, inp1_unstable=8, inp2_unstable=8, delta_unstable=256
- layer_idx=4, type=LayerType.Linear, total=256, inp1_unstable=0, inp2_unstable=0, delta_unstable=256
- layer_idx=6, type=LayerType.Linear, total=256, inp1_unstable=0, inp2_unstable=0, delta_unstable=256
- layer_idx=8, type=LayerType.Linear, total=10, inp1_unstable=0, inp2_unstable=0, delta_unstable=10

Time for backsubstitution: 1.67 seconds

### RS candidates at layer 1
type: RSZ, layer: 1, pos: 196
type: RSZ, layer: 1, pos: 86

Time for candidate selection: 0.00 seconds

### Candidate
type: RSZ, layer: 1, pos: 196

### Relational analysis RSZ of RS_RSZ2_RSZ2_RSZ1

#### Relational analysis RSZ result of RS_RSZ2_RSZ2_RSZ1
Status: Status.UNKNOWN
Output dim: 4, lower bound: -0.0007339, upper bound: 0.0007821
time: 0.49 seconds

### Relational analysis RSZ of RS_RSZ2_RSZ2_RSZ2

#### Relational analysis RSZ result of RS_RSZ2_RSZ2_RSZ2
Status: Status.UNKNOWN
Output dim: 4, lower bound: -0.0007766, upper bound: 0.0007626
time: 0.50 seconds

## Summary of splitting (split count: 2)
- Time for RS candidates: 2.67 seconds
RS_RSZ1_RSZ1_RSZ1, status: Status.UNKNOWN, split count: 3, time: 2.67
Output dim: 4, lower bound: -0.0007626, upper bound: 0.0007767
RS_RSZ1_RSZ1_RSZ2, status: Status.UNKNOWN, split count: 3, time: 2.67
Output dim: 4, lower bound: -0.0007339, upper bound: 0.0007831
RS_RSZ1_RSZ2_RSZ1, status: Status.UNKNOWN, split count: 3, time: 2.67
Output dim: 4, lower bound: -0.0006673, upper bound: 0.0006874
RS_RSZ1_RSZ2_RSZ2, status: Status.UNKNOWN, split count: 3, time: 2.67
Output dim: 4, lower bound: -0.0006947, upper bound: 0.0006614
RS_RSZ2_RSZ1_RSZ1, status: Status.UNKNOWN, split count: 3, time: 2.67
Output dim: 4, lower bound: -0.0007667, upper bound: 0.0007763
RS_RSZ2_RSZ1_RSZ2, status: Status.UNKNOWN, split count: 3, time: 2.67
Output dim: 4, lower bound: -0.0007831, upper bound: 0.0007340
RS_RSZ2_RSZ2_RSZ1, status: Status.UNKNOWN, split count: 3, time: 2.67
Output dim: 4, lower bound: -0.0007339, upper bound: 0.0007821
RS_RSZ2_RSZ2_RSZ2, status: Status.UNKNOWN, split count: 3, time: 2.67
Output dim: 4, lower bound: -0.0007766, upper bound: 0.0007626

## BFS RS instance: RS_RSZ1_RSZ1_RSZ1

### Backsubstitution after applying RS history:
0: -0.0039650, -0.0038291, -0.0039650, -0.0038291, -0.0000682, 0.0000679
1: 0.0009495, 0.0017023, 0.0009495, 0.0017023, -0.0003758, 0.0003774
2: 0.0111630, 0.0128448, 0.0111630, 0.0128448, -0.0008431, 0.0008397
3: 0.0019215, 0.0026303, 0.0019215, 0.0026303, -0.0003538, 0.0003553
4: 1.0042051, 1.0069547, 1.0042051, 1.0069547, -0.0013728, 0.0013783
5: 0.0030595, 0.0035944, 0.0030595, 0.0035944, -0.0002671, 0.0002681
6: -0.0104205, -0.0097244, -0.0104205, -0.0097244, -0.0003489, 0.0003475
7: -0.0101326, -0.0100438, -0.0101326, -0.0100438, -0.0000445, 0.0000443
8: -0.0041353, -0.0036544, -0.0041353, -0.0036544, -0.0002401, 0.0002411
9: 0.0001238, 0.0025315, 0.0001238, 0.0025315, -0.0012070, 0.0012021

### Unstable ReLU Count (Linear/Conv2D Layers)
- layer_idx=0, type=LayerType.Linear, total=256, inp1_unstable=4, inp2_unstable=4, delta_unstable=253
- layer_idx=2, type=LayerType.Linear, total=256, inp1_unstable=8, inp2_unstable=8, delta_unstable=256
- layer_idx=4, type=LayerType.Linear, total=256, inp1_unstable=0, inp2_unstable=0, delta_unstable=256
- layer_idx=6, type=LayerType.Linear, total=256, inp1_unstable=0, inp2_unstable=0, delta_unstable=256
- layer_idx=8, type=LayerType.Linear, total=10, inp1_unstable=0, inp2_unstable=0, delta_unstable=10

Time for backsubstitution: 1.62 seconds

### RS candidates at layer 1
type: RSZ, layer: 1, pos: 86

Time for candidate selection: 0.00 seconds

### Candidate
type: RSZ, layer: 1, pos: 86

### Relational analysis RSZ of RS_RSZ1_RSZ1_RSZ1_RSZ1

#### Relational analysis RSZ result of RS_RSZ1_RSZ1_RSZ1_RSZ1
Status: Status.VERIFIED
Output dim: 4, lower bound: -0.0005664, upper bound: 0.0005880
time: 0.50 seconds

### Relational analysis RSZ of RS_RSZ1_RSZ1_RSZ1_RSZ2

#### Relational analysis RSZ result of RS_RSZ1_RSZ1_RSZ1_RSZ2
Status: Status.VERIFIED
Output dim: 4, lower bound: -0.0005727, upper bound: 0.0005818
time: 0.50 seconds

## BFS RS instance: RS_RSZ1_RSZ1_RSZ2

### Backsubstitution after applying RS history:
0: -0.0039650, -0.0038291, -0.0039650, -0.0038291, -0.0000688, 0.0000676
1: 0.0009495, 0.0017023, 0.0009495, 0.0017023, -0.0003743, 0.0003811
2: 0.0111630, 0.0128448, 0.0111630, 0.0128448, -0.0008514, 0.0008362
3: 0.0019215, 0.0026303, 0.0019215, 0.0026303, -0.0003524, 0.0003588
4: 1.0042051, 1.0069547, 1.0042051, 1.0069547, -0.0013671, 0.0013920
5: 0.0030595, 0.0035944, 0.0030595, 0.0035944, -0.0002660, 0.0002708
6: -0.0104205, -0.0097244, -0.0104205, -0.0097244, -0.0003524, 0.0003461
7: -0.0101326, -0.0100438, -0.0101326, -0.0100438, -0.0000450, 0.0000441
8: -0.0041353, -0.0036544, -0.0041353, -0.0036544, -0.0002391, 0.0002435
9: 0.0001238, 0.0025315, 0.0001238, 0.0025315, -0.0012189, 0.0011972

### Unstable ReLU Count (Linear/Conv2D Layers)
- layer_idx=0, type=LayerType.Linear, total=256, inp1_unstable=4, inp2_unstable=4, delta_unstable=253
- layer_idx=2, type=LayerType.Linear, total=256, inp1_unstable=8, inp2_unstable=8, delta_unstable=256
- layer_idx=4, type=LayerType.Linear, total=256, inp1_unstable=0, inp2_unstable=0, delta_unstable=256
- layer_idx=6, type=LayerType.Linear, total=256, inp1_unstable=0, inp2_unstable=0, delta_unstable=256
- layer_idx=8, type=LayerType.Linear, total=10, inp1_unstable=0, inp2_unstable=0, delta_unstable=10

Time for backsubstitution: 1.63 seconds

### RS candidates at layer 1
type: RSZ, layer: 1, pos: 86

Time for candidate selection: 0.00 seconds

### Candidate
type: RSZ, layer: 1, pos: 86

### Relational analysis RSZ of RS_RSZ1_RSZ1_RSZ2_RSZ1

#### Relational analysis RSZ result of RS_RSZ1_RSZ1_RSZ2_RSZ1
Status: Status.VERIFIED
Output dim: 4, lower bound: -0.0005659, upper bound: 0.0005880
time: 0.51 seconds

### Relational analysis RSZ of RS_RSZ1_RSZ1_RSZ2_RSZ2

#### Relational analysis RSZ result of RS_RSZ1_RSZ1_RSZ2_RSZ2
Status: Status.VERIFIED
Output dim: 4, lower bound: -0.0005727, upper bound: 0.0005818
time: 0.51 seconds

## BFS RS instance: RS_RSZ1_RSZ2_RSZ1

### Backsubstitution after applying RS history:
0: -0.0039650, -0.0038291, -0.0039650, -0.0038291, -0.0000708, 0.0000652
1: 0.0009495, 0.0017023, 0.0009495, 0.0017023, -0.0003610, 0.0003922
2: 0.0111630, 0.0128448, 0.0111630, 0.0128448, -0.0008762, 0.0008065
3: 0.0019215, 0.0026303, 0.0019215, 0.0026303, -0.0003398, 0.0003692
4: 1.0042051, 1.0069547, 1.0042051, 1.0069547, -0.0013185, 0.0014325
5: 0.0030595, 0.0035944, 0.0030595, 0.0035944, -0.0002565, 0.0002787
6: -0.0104205, -0.0097244, -0.0104205, -0.0097244, -0.0003627, 0.0003338
7: -0.0101326, -0.0100438, -0.0101326, -0.0100438, -0.0000463, 0.0000426
8: -0.0041353, -0.0036544, -0.0041353, -0.0036544, -0.0002306, 0.0002506
9: 0.0001238, 0.0025315, 0.0001238, 0.0025315, -0.0012544, 0.0011545

### Unstable ReLU Count (Linear/Conv2D Layers)
- layer_idx=0, type=LayerType.Linear, total=256, inp1_unstable=4, inp2_unstable=4, delta_unstable=253
- layer_idx=2, type=LayerType.Linear, total=256, inp1_unstable=8, inp2_unstable=8, delta_unstable=256
- layer_idx=4, type=LayerType.Linear, total=256, inp1_unstable=0, inp2_unstable=0, delta_unstable=256
- layer_idx=6, type=LayerType.Linear, total=256, inp1_unstable=0, inp2_unstable=0, delta_unstable=256
- layer_idx=8, type=LayerType.Linear, total=10, inp1_unstable=0, inp2_unstable=0, delta_unstable=10

Time for backsubstitution: 1.73 seconds

### RS candidates at layer 1
type: RSZ, layer: 1, pos: 155

Time for candidate selection: 0.00 seconds

### Candidate
type: RSZ, layer: 1, pos: 155

### Relational analysis RSZ of RS_RSZ1_RSZ2_RSZ1_RSZ1

#### Relational analysis RSZ result of RS_RSZ1_RSZ2_RSZ1_RSZ1
Status: Status.VERIFIED
Output dim: 4, lower bound: -0.0005820, upper bound: 0.0005714
time: 0.50 seconds

### Relational analysis RSZ of RS_RSZ1_RSZ2_RSZ1_RSZ2

#### Relational analysis RSZ result of RS_RSZ1_RSZ2_RSZ1_RSZ2
Status: Status.VERIFIED
Output dim: 4, lower bound: -0.0005820, upper bound: 0.0005715
time: 0.49 seconds

## BFS RS instance: RS_RSZ1_RSZ2_RSZ2

### Backsubstitution after applying RS history:
0: -0.0039650, -0.0038291, -0.0039650, -0.0038291, -0.0000647, 0.0000688
1: 0.0009495, 0.0017023, 0.0009495, 0.0017023, -0.0003809, 0.0003583
2: 0.0111630, 0.0128448, 0.0111630, 0.0128448, -0.0008006, 0.0008510
3: 0.0019215, 0.0026303, 0.0019215, 0.0026303, -0.0003586, 0.0003374
4: 1.0042051, 1.0069547, 1.0042051, 1.0069547, -0.0013913, 0.0013089
5: 0.0030595, 0.0035944, 0.0030595, 0.0035944, -0.0002707, 0.0002546
6: -0.0104205, -0.0097244, -0.0104205, -0.0097244, -0.0003314, 0.0003522
7: -0.0101326, -0.0100438, -0.0101326, -0.0100438, -0.0000423, 0.0000449
8: -0.0041353, -0.0036544, -0.0041353, -0.0036544, -0.0002434, 0.0002289
9: 0.0001238, 0.0025315, 0.0001238, 0.0025315, -0.0011461, 0.0012183

### Unstable ReLU Count (Linear/Conv2D Layers)
- layer_idx=0, type=LayerType.Linear, total=256, inp1_unstable=4, inp2_unstable=4, delta_unstable=253
- layer_idx=2, type=LayerType.Linear, total=256, inp1_unstable=8, inp2_unstable=8, delta_unstable=256
- layer_idx=4, type=LayerType.Linear, total=256, inp1_unstable=0, inp2_unstable=0, delta_unstable=256
- layer_idx=6, type=LayerType.Linear, total=256, inp1_unstable=0, inp2_unstable=0, delta_unstable=256
- layer_idx=8, type=LayerType.Linear, total=10, inp1_unstable=0, inp2_unstable=0, delta_unstable=10

Time for backsubstitution: 1.69 seconds

### RS candidates at layer 1
type: RSZ, layer: 1, pos: 155

Time for candidate selection: 0.00 seconds

### Candidate
type: RSZ, layer: 1, pos: 155

### Relational analysis RSZ of RS_RSZ1_RSZ2_RSZ2_RSZ1

#### Relational analysis RSZ result of RS_RSZ1_RSZ2_RSZ2_RSZ1
Status: Status.VERIFIED
Output dim: 4, lower bound: -0.0005883, upper bound: 0.0005659
time: 0.49 seconds

### Relational analysis RSZ of RS_RSZ1_RSZ2_RSZ2_RSZ2

#### Relational analysis RSZ result of RS_RSZ1_RSZ2_RSZ2_RSZ2
Status: Status.VERIFIED
Output dim: 4, lower bound: -0.0005883, upper bound: 0.0005664
time: 0.49 seconds

## BFS RS instance: RS_RSZ2_RSZ1_RSZ1

### Backsubstitution after applying RS history:
0: -0.0039650, -0.0038291, -0.0039650, -0.0038291, -0.0000681, 0.0000680
1: 0.0009495, 0.0017023, 0.0009495, 0.0017023, -0.0003764, 0.0003772
2: 0.0111630, 0.0128448, 0.0111630, 0.0128448, -0.0008426, 0.0008410
3: 0.0019215, 0.0026303, 0.0019215, 0.0026303, -0.0003544, 0.0003551
4: 1.0042051, 1.0069547, 1.0042051, 1.0069547, -0.0013749, 0.0013776
5: 0.0030595, 0.0035944, 0.0030595, 0.0035944, -0.0002675, 0.0002680
6: -0.0104205, -0.0097244, -0.0104205, -0.0097244, -0.0003488, 0.0003481
7: -0.0101326, -0.0100438, -0.0101326, -0.0100438, -0.0000445, 0.0000444
8: -0.0041353, -0.0036544, -0.0041353, -0.0036544, -0.0002405, 0.0002410
9: 0.0001238, 0.0025315, 0.0001238, 0.0025315, -0.0012063, 0.0012039

### Unstable ReLU Count (Linear/Conv2D Layers)
- layer_idx=0, type=LayerType.Linear, total=256, inp1_unstable=4, inp2_unstable=4, delta_unstable=253
- layer_idx=2, type=LayerType.Linear, total=256, inp1_unstable=8, inp2_unstable=8, delta_unstable=256
- layer_idx=4, type=LayerType.Linear, total=256, inp1_unstable=0, inp2_unstable=0, delta_unstable=256
- layer_idx=6, type=LayerType.Linear, total=256, inp1_unstable=0, inp2_unstable=0, delta_unstable=256
- layer_idx=8, type=LayerType.Linear, total=10, inp1_unstable=0, inp2_unstable=0, delta_unstable=10

Time for backsubstitution: 1.72 seconds

### RS candidates at layer 1
type: RSZ, layer: 1, pos: 86

Time for candidate selection: 0.00 seconds

### Candidate
type: RSZ, layer: 1, pos: 86

### Relational analysis RSZ of RS_RSZ2_RSZ1_RSZ1_RSZ1

#### Relational analysis RSZ result of RS_RSZ2_RSZ1_RSZ1_RSZ1
Status: Status.VERIFIED
Output dim: 4, lower bound: -0.0005664, upper bound: 0.0005883
time: 0.48 seconds

### Relational analysis RSZ of RS_RSZ2_RSZ1_RSZ1_RSZ2

#### Relational analysis RSZ result of RS_RSZ2_RSZ1_RSZ1_RSZ2
Status: Status.VERIFIED
Output dim: 4, lower bound: -0.0005715, upper bound: 0.0005820
time: 0.49 seconds

## BFS RS instance: RS_RSZ2_RSZ1_RSZ2

### Backsubstitution after applying RS history:
0: -0.0039650, -0.0038291, -0.0039650, -0.0038291, -0.0000676, 0.0000683
1: 0.0009495, 0.0017023, 0.0009495, 0.0017023, -0.0003783, 0.0003743
2: 0.0111630, 0.0128448, 0.0111630, 0.0128448, -0.0008362, 0.0008452
3: 0.0019215, 0.0026303, 0.0019215, 0.0026303, -0.0003562, 0.0003524
4: 1.0042051, 1.0069547, 1.0042051, 1.0069547, -0.0013817, 0.0013671
5: 0.0030595, 0.0035944, 0.0030595, 0.0035944, -0.0002688, 0.0002660
6: -0.0104205, -0.0097244, -0.0104205, -0.0097244, -0.0003461, 0.0003498
7: -0.0101326, -0.0100438, -0.0101326, -0.0100438, -0.0000441, 0.0000446
8: -0.0041353, -0.0036544, -0.0041353, -0.0036544, -0.0002417, 0.0002391
9: 0.0001238, 0.0025315, 0.0001238, 0.0025315, -0.0011972, 0.0012100

### Unstable ReLU Count (Linear/Conv2D Layers)
- layer_idx=0, type=LayerType.Linear, total=256, inp1_unstable=4, inp2_unstable=4, delta_unstable=253
- layer_idx=2, type=LayerType.Linear, total=256, inp1_unstable=8, inp2_unstable=8, delta_unstable=256
- layer_idx=4, type=LayerType.Linear, total=256, inp1_unstable=0, inp2_unstable=0, delta_unstable=256
- layer_idx=6, type=LayerType.Linear, total=256, inp1_unstable=0, inp2_unstable=0, delta_unstable=256
- layer_idx=8, type=LayerType.Linear, total=10, inp1_unstable=0, inp2_unstable=0, delta_unstable=10

Time for backsubstitution: 1.67 seconds

### RS candidates at layer 1
type: RSZ, layer: 1, pos: 86

Time for candidate selection: 0.00 seconds

### Candidate
type: RSZ, layer: 1, pos: 86

### Relational analysis RSZ of RS_RSZ2_RSZ1_RSZ2_RSZ1

#### Relational analysis RSZ result of RS_RSZ2_RSZ1_RSZ2_RSZ1
Status: Status.VERIFIED
Output dim: 4, lower bound: -0.0005818, upper bound: 0.0005727
time: 0.50 seconds

### Relational analysis RSZ of RS_RSZ2_RSZ1_RSZ2_RSZ2

#### Relational analysis RSZ result of RS_RSZ2_RSZ1_RSZ2_RSZ2
Status: Status.VERIFIED
Output dim: 4, lower bound: -0.0005880, upper bound: 0.0005659
time: 0.49 seconds

## BFS RS instance: RS_RSZ2_RSZ2_RSZ1

### Backsubstitution after applying RS history:
0: -0.0039650, -0.0038291, -0.0039650, -0.0038291, -0.0000688, 0.0000676
1: 0.0009495, 0.0017023, 0.0009495, 0.0017023, -0.0003745, 0.0003809
2: 0.0111630, 0.0128448, 0.0111630, 0.0128448, -0.0008510, 0.0008368
3: 0.0019215, 0.0026303, 0.0019215, 0.0026303, -0.0003526, 0.0003586
4: 1.0042051, 1.0069547, 1.0042051, 1.0069547, -0.0013680, 0.0013913
5: 0.0030595, 0.0035944, 0.0030595, 0.0035944, -0.0002661, 0.0002707
6: -0.0104205, -0.0097244, -0.0104205, -0.0097244, -0.0003522, 0.0003463
7: -0.0101326, -0.0100438, -0.0101326, -0.0100438, -0.0000449, 0.0000442
8: -0.0041353, -0.0036544, -0.0041353, -0.0036544, -0.0002393, 0.0002434
9: 0.0001238, 0.0025315, 0.0001238, 0.0025315, -0.0012183, 0.0011979

### Unstable ReLU Count (Linear/Conv2D Layers)
- layer_idx=0, type=LayerType.Linear, total=256, inp1_unstable=4, inp2_unstable=4, delta_unstable=253
- layer_idx=2, type=LayerType.Linear, total=256, inp1_unstable=8, inp2_unstable=8, delta_unstable=256
- layer_idx=4, type=LayerType.Linear, total=256, inp1_unstable=0, inp2_unstable=0, delta_unstable=256
- layer_idx=6, type=LayerType.Linear, total=256, inp1_unstable=0, inp2_unstable=0, delta_unstable=256
- layer_idx=8, type=LayerType.Linear, total=10, inp1_unstable=0, inp2_unstable=0, delta_unstable=10

Time for backsubstitution: 1.65 seconds

### RS candidates at layer 1
type: RSZ, layer: 1, pos: 86

Time for candidate selection: 0.00 seconds

### Candidate
type: RSZ, layer: 1, pos: 86

### Relational analysis RSZ of RS_RSZ2_RSZ2_RSZ1_RSZ1

#### Relational analysis RSZ result of RS_RSZ2_RSZ2_RSZ1_RSZ1
Status: Status.VERIFIED
Output dim: 4, lower bound: -0.0005659, upper bound: 0.0005883
time: 0.49 seconds

### Relational analysis RSZ of RS_RSZ2_RSZ2_RSZ1_RSZ2

#### Relational analysis RSZ result of RS_RSZ2_RSZ2_RSZ1_RSZ2
Status: Status.VERIFIED
Output dim: 4, lower bound: -0.0005714, upper bound: 0.0005820
time: 0.48 seconds

## BFS RS instance: RS_RSZ2_RSZ2_RSZ2

### Backsubstitution after applying RS history:
0: -0.0039650, -0.0038291, -0.0039650, -0.0038291, -0.0000683, 0.0000682
1: 0.0009495, 0.0017023, 0.0009495, 0.0017023, -0.0003774, 0.0003780
2: 0.0111630, 0.0128448, 0.0111630, 0.0128448, -0.0008446, 0.0008431
3: 0.0019215, 0.0026303, 0.0019215, 0.0026303, -0.0003553, 0.0003559
4: 1.0042051, 1.0069547, 1.0042051, 1.0069547, -0.0013783, 0.0013808
5: 0.0030595, 0.0035944, 0.0030595, 0.0035944, -0.0002681, 0.0002686
6: -0.0104205, -0.0097244, -0.0104205, -0.0097244, -0.0003496, 0.0003489
7: -0.0101326, -0.0100438, -0.0101326, -0.0100438, -0.0000446, 0.0000445
8: -0.0041353, -0.0036544, -0.0041353, -0.0036544, -0.0002411, 0.0002415
9: 0.0001238, 0.0025315, 0.0001238, 0.0025315, -0.0012091, 0.0012070

### Unstable ReLU Count (Linear/Conv2D Layers)
- layer_idx=0, type=LayerType.Linear, total=256, inp1_unstable=4, inp2_unstable=4, delta_unstable=253
- layer_idx=2, type=LayerType.Linear, total=256, inp1_unstable=8, inp2_unstable=8, delta_unstable=256
- layer_idx=4, type=LayerType.Linear, total=256, inp1_unstable=0, inp2_unstable=0, delta_unstable=256
- layer_idx=6, type=LayerType.Linear, total=256, inp1_unstable=0, inp2_unstable=0, delta_unstable=256
- layer_idx=8, type=LayerType.Linear, total=10, inp1_unstable=0, inp2_unstable=0, delta_unstable=10

Time for backsubstitution: 1.66 seconds

### RS candidates at layer 1
type: RSZ, layer: 1, pos: 86

Time for candidate selection: 0.00 seconds

### Candidate
type: RSZ, layer: 1, pos: 86

### Relational analysis RSZ of RS_RSZ2_RSZ2_RSZ2_RSZ1

#### Relational analysis RSZ result of RS_RSZ2_RSZ2_RSZ2_RSZ1
Status: Status.VERIFIED
Output dim: 4, lower bound: -0.0005818, upper bound: 0.0005727
time: 0.54 seconds

### Relational analysis RSZ of RS_RSZ2_RSZ2_RSZ2_RSZ2

#### Relational analysis RSZ result of RS_RSZ2_RSZ2_RSZ2_RSZ2
Status: Status.VERIFIED
Output dim: 4, lower bound: -0.0005880, upper bound: 0.0005664
time: 0.54 seconds

## Summary of splitting (split count: 3)
- Time for RS candidates: 2.76 seconds
RS_RSZ1_RSZ1_RSZ1_RSZ1, status: Status.VERIFIED, split count: 4, time: 2.76
Output dim: 4, lower bound: -0.0005664, upper bound: 0.0005880
RS_RSZ1_RSZ1_RSZ1_RSZ2, status: Status.VERIFIED, split count: 4, time: 2.76
Output dim: 4, lower bound: -0.0005727, upper bound: 0.0005818
RS_RSZ1_RSZ1_RSZ2_RSZ1, status: Status.VERIFIED, split count: 4, time: 2.76
Output dim: 4, lower bound: -0.0005659, upper bound: 0.0005880
RS_RSZ1_RSZ1_RSZ2_RSZ2, status: Status.VERIFIED, split count: 4, time: 2.76
Output dim: 4, lower bound: -0.0005727, upper bound: 0.0005818
RS_RSZ1_RSZ2_RSZ1_RSZ1, status: Status.VERIFIED, split count: 4, time: 2.76
Output dim: 4, lower bound: -0.0005820, upper bound: 0.0005714
RS_RSZ1_RSZ2_RSZ1_RSZ2, status: Status.VERIFIED, split count: 4, time: 2.76
Output dim: 4, lower bound: -0.0005820, upper bound: 0.0005715
RS_RSZ1_RSZ2_RSZ2_RSZ1, status: Status.VERIFIED, split count: 4, time: 2.76
Output dim: 4, lower bound: -0.0005883, upper bound: 0.0005659
RS_RSZ1_RSZ2_RSZ2_RSZ2, status: Status.VERIFIED, split count: 4, time: 2.76
Output dim: 4, lower bound: -0.0005883, upper bound: 0.0005664
RS_RSZ2_RSZ1_RSZ1_RSZ1, status: Status.VERIFIED, split count: 4, time: 2.76
Output dim: 4, lower bound: -0.0005664, upper bound: 0.0005883
RS_RSZ2_RSZ1_RSZ1_RSZ2, status: Status.VERIFIED, split count: 4, time: 2.76
Output dim: 4, lower bound: -0.0005715, upper bound: 0.0005820
RS_RSZ2_RSZ1_RSZ2_RSZ1, status: Status.VERIFIED, split count: 4, time: 2.76
Output dim: 4, lower bound: -0.0005818, upper bound: 0.0005727
RS_RSZ2_RSZ1_RSZ2_RSZ2, status: Status.VERIFIED, split count: 4, time: 2.76
Output dim: 4, lower bound: -0.0005880, upper bound: 0.0005659
RS_RSZ2_RSZ2_RSZ1_RSZ1, status: Status.VERIFIED, split count: 4, time: 2.76
Output dim: 4, lower bound: -0.0005659, upper bound: 0.0005883
RS_RSZ2_RSZ2_RSZ1_RSZ2, status: Status.VERIFIED, split count: 4, time: 2.76
Output dim: 4, lower bound: -0.0005714, upper bound: 0.0005820
RS_RSZ2_RSZ2_RSZ2_RSZ1, status: Status.VERIFIED, split count: 4, time: 2.76
Output dim: 4, lower bound: -0.0005818, upper bound: 0.0005727
RS_RSZ2_RSZ2_RSZ2_RSZ2, status: Status.VERIFIED, split count: 4, time: 2.76
Output dim: 4, lower bound: -0.0005880, upper bound: 0.0005664

## RS Result
status: Status.VERIFIED
execution time: (base) + (rs) = 3.35 + 38.42 = 41.77 seconds
