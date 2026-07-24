## Execution arguments:
Dataset: Dataset.MNIST
Network: onnx/mnist-net_256x4.onnx
Relational property: GLOBAL_ROBUSTNESS
LP Analysis: True
Epsilon: 0.01171875
Delta epsilon: 0.00390625
execution index: (1, 3, 0)
Time budget: 600 seconds
Split limit: 100
Threshold: 0.0005928


## IAR start

### Unstable ReLU Count (Linear/Conv2D Layers)
- layer_idx=0, type=LayerType.Linear, total=256, inp1_unstable=2, inp2_unstable=2, delta_unstable=256
- layer_idx=2, type=LayerType.Linear, total=256, inp1_unstable=5, inp2_unstable=5, delta_unstable=256
- layer_idx=4, type=LayerType.Linear, total=256, inp1_unstable=0, inp2_unstable=0, delta_unstable=256
- layer_idx=6, type=LayerType.Linear, total=256, inp1_unstable=0, inp2_unstable=0, delta_unstable=256
- layer_idx=8, type=LayerType.Linear, total=10, inp1_unstable=0, inp2_unstable=0, delta_unstable=10

### BASE IAR bounds
Layer (inp1_lb, inp1_ub, inp2_lb, inp2_ub, d_lb, d_ub)
0: (1.0026824, 1.0047039, 1.0026824, 1.0047039, -0.0012827, 0.0012827)
1: (-0.0005956, -0.0000919, -0.0005956, -0.0000919, -0.0003196, 0.0003196)
2: (-0.0095672, -0.0068978, -0.0095672, -0.0068978, -0.0016938, 0.0016938)
3: (0.0018665, 0.0030814, 0.0018665, 0.0030814, -0.0007710, 0.0007710)
4: (-0.0013238, -0.0008072, -0.0013238, -0.0008072, -0.0003278, 0.0003278)
5: (-0.0130736, -0.0097162, -0.0130736, -0.0097162, -0.0021304, 0.0021304)
6: (0.0040069, 0.0048590, 0.0040069, 0.0048590, -0.0005407, 0.0005407)
7: (0.0072295, 0.0094342, 0.0072295, 0.0094342, -0.0013990, 0.0013990)
8: (0.0042378, 0.0053972, 0.0042378, 0.0053972, -0.0007357, 0.0007357)
9: (-0.0081222, -0.0067777, -0.0081222, -0.0067777, -0.0008531, 0.0008531)

## BASE Relational Analysis

## BASE Result
execution time: IAR + RelationalAnalysis = 1.28 + 1.56 = 2.83 seconds
status: Status.UNKNOWN
relational distance
Output dim: 0, lower bound: -0.0006736, upper bound: 0.0006737

# Relational Split (RS) starts

## BFS RS instance: RS

Time for backsubstitution: 0.01 seconds

### RS candidates at layer 1
type: RSZ, layer: 1, pos: 75
type: RSZ, layer: 1, pos: 83

Time for candidate selection: 0.13 seconds

### Candidate
type: RSZ, layer: 1, pos: 75

### Relational analysis RSZ of RS_RSZ1

#### Relational analysis RSZ result of RS_RSZ1
Status: Status.UNKNOWN
Output dim: 0, lower bound: -0.0006320, upper bound: 0.0005614
time: 0.72 seconds

### Relational analysis RSZ of RS_RSZ2

#### Relational analysis RSZ result of RS_RSZ2
Status: Status.UNKNOWN
Output dim: 0, lower bound: -0.0005614, upper bound: 0.0006320
time: 0.72 seconds

## Summary of splitting (split count: 0)
- Time for RS candidates: 1.58 seconds
RS_RSZ1, status: Status.UNKNOWN, split count: 1, time: 1.58
Output dim: 0, lower bound: -0.0006320, upper bound: 0.0005614
RS_RSZ2, status: Status.UNKNOWN, split count: 1, time: 1.58
Output dim: 0, lower bound: -0.0005614, upper bound: 0.0006320

## BFS RS instance: RS_RSZ1

### Backsubstitution after applying RS history:
0: 1.0026824, 1.0047039, 1.0026824, 1.0047039, -0.0009909, 0.0009398
1: -0.0005956, -0.0000919, -0.0005956, -0.0000919, -0.0002469, 0.0002342
2: -0.0095672, -0.0068978, -0.0095672, -0.0068978, -0.0012410, 0.0013085
3: 0.0018665, 0.0030814, 0.0018665, 0.0030814, -0.0005956, 0.0005649
4: -0.0013238, -0.0008072, -0.0013238, -0.0008072, -0.0002402, 0.0002533
5: -0.0130736, -0.0097162, -0.0130736, -0.0097162, -0.0015609, 0.0016457
6: 0.0040069, 0.0048590, 0.0040069, 0.0048590, -0.0004177, 0.0003962
7: 0.0072295, 0.0094342, 0.0072295, 0.0094342, -0.0010807, 0.0010250
8: 0.0042378, 0.0053972, 0.0042378, 0.0053972, -0.0005683, 0.0005390
9: -0.0081222, -0.0067777, -0.0081222, -0.0067777, -0.0006251, 0.0006590

### Unstable ReLU Count (Linear/Conv2D Layers)
- layer_idx=0, type=LayerType.Linear, total=256, inp1_unstable=2, inp2_unstable=2, delta_unstable=255
- layer_idx=2, type=LayerType.Linear, total=256, inp1_unstable=5, inp2_unstable=5, delta_unstable=256
- layer_idx=4, type=LayerType.Linear, total=256, inp1_unstable=0, inp2_unstable=0, delta_unstable=256
- layer_idx=6, type=LayerType.Linear, total=256, inp1_unstable=0, inp2_unstable=0, delta_unstable=256
- layer_idx=8, type=LayerType.Linear, total=10, inp1_unstable=0, inp2_unstable=0, delta_unstable=10

Time for backsubstitution: 1.17 seconds

### RS candidates at layer 1
type: RSZ, layer: 1, pos: 83

Time for candidate selection: 0.12 seconds

### Candidate
type: RSZ, layer: 1, pos: 83

### Relational analysis RSZ of RS_RSZ1_RSZ1

#### Relational analysis RSZ result of RS_RSZ1_RSZ1
Status: Status.UNKNOWN
Output dim: 0, lower bound: -0.0006127, upper bound: 0.0005478
time: 0.76 seconds

### Relational analysis RSZ of RS_RSZ1_RSZ2

#### Relational analysis RSZ result of RS_RSZ1_RSZ2
Status: Status.UNKNOWN
Output dim: 0, lower bound: -0.0006185, upper bound: 0.0005460
time: 0.76 seconds

## BFS RS instance: RS_RSZ2

### Backsubstitution after applying RS history:
0: 1.0026824, 1.0047039, 1.0026824, 1.0047039, -0.0009398, 0.0009909
1: -0.0005956, -0.0000919, -0.0005956, -0.0000919, -0.0002342, 0.0002469
2: -0.0095672, -0.0068978, -0.0095672, -0.0068978, -0.0013085, 0.0012410
3: 0.0018665, 0.0030814, 0.0018665, 0.0030814, -0.0005649, 0.0005956
4: -0.0013238, -0.0008072, -0.0013238, -0.0008072, -0.0002533, 0.0002402
5: -0.0130736, -0.0097162, -0.0130736, -0.0097162, -0.0016457, 0.0015609
6: 0.0040069, 0.0048590, 0.0040069, 0.0048590, -0.0003962, 0.0004177
7: 0.0072295, 0.0094342, 0.0072295, 0.0094342, -0.0010250, 0.0010807
8: 0.0042378, 0.0053972, 0.0042378, 0.0053972, -0.0005390, 0.0005683
9: -0.0081222, -0.0067777, -0.0081222, -0.0067777, -0.0006590, 0.0006251

### Unstable ReLU Count (Linear/Conv2D Layers)
- layer_idx=0, type=LayerType.Linear, total=256, inp1_unstable=2, inp2_unstable=2, delta_unstable=255
- layer_idx=2, type=LayerType.Linear, total=256, inp1_unstable=5, inp2_unstable=5, delta_unstable=256
- layer_idx=4, type=LayerType.Linear, total=256, inp1_unstable=0, inp2_unstable=0, delta_unstable=256
- layer_idx=6, type=LayerType.Linear, total=256, inp1_unstable=0, inp2_unstable=0, delta_unstable=256
- layer_idx=8, type=LayerType.Linear, total=10, inp1_unstable=0, inp2_unstable=0, delta_unstable=10

Time for backsubstitution: 1.15 seconds

### RS candidates at layer 1
type: RSZ, layer: 1, pos: 83

Time for candidate selection: 0.12 seconds

### Candidate
type: RSZ, layer: 1, pos: 83

### Relational analysis RSZ of RS_RSZ2_RSZ1

#### Relational analysis RSZ result of RS_RSZ2_RSZ1
Status: Status.UNKNOWN
Output dim: 0, lower bound: -0.0005460, upper bound: 0.0006185
time: 0.72 seconds

### Relational analysis RSZ of RS_RSZ2_RSZ2

#### Relational analysis RSZ result of RS_RSZ2_RSZ2
Status: Status.UNKNOWN
Output dim: 0, lower bound: -0.0005478, upper bound: 0.0006127
time: 0.77 seconds

## Summary of splitting (split count: 1)
- Time for RS candidates: 2.77 seconds
RS_RSZ1_RSZ1, status: Status.UNKNOWN, split count: 2, time: 2.77
Output dim: 0, lower bound: -0.0006127, upper bound: 0.0005478
RS_RSZ1_RSZ2, status: Status.UNKNOWN, split count: 2, time: 2.77
Output dim: 0, lower bound: -0.0006185, upper bound: 0.0005460
RS_RSZ2_RSZ1, status: Status.UNKNOWN, split count: 2, time: 2.77
Output dim: 0, lower bound: -0.0005460, upper bound: 0.0006185
RS_RSZ2_RSZ2, status: Status.UNKNOWN, split count: 2, time: 2.77
Output dim: 0, lower bound: -0.0005478, upper bound: 0.0006127

## BFS RS instance: RS_RSZ1_RSZ1

### Backsubstitution after applying RS history:
0: 1.0026824, 1.0047039, 1.0026824, 1.0047039, -0.0008342, 0.0007900
1: -0.0005956, -0.0000919, -0.0005956, -0.0000919, -0.0002079, 0.0001968
2: -0.0095672, -0.0068978, -0.0095672, -0.0068978, -0.0010431, 0.0011016
3: 0.0018665, 0.0030814, 0.0018665, 0.0030814, -0.0005014, 0.0004748
4: -0.0013238, -0.0008072, -0.0013238, -0.0008072, -0.0002019, 0.0002132
5: -0.0130736, -0.0097162, -0.0130736, -0.0097162, -0.0013120, 0.0013855
6: 0.0040069, 0.0048590, 0.0040069, 0.0048590, -0.0003517, 0.0003330
7: 0.0072295, 0.0094342, 0.0072295, 0.0094342, -0.0009099, 0.0008616
8: 0.0042378, 0.0053972, 0.0042378, 0.0053972, -0.0004785, 0.0004531
9: -0.0081222, -0.0067777, -0.0081222, -0.0067777, -0.0005254, 0.0005548

### Unstable ReLU Count (Linear/Conv2D Layers)
- layer_idx=0, type=LayerType.Linear, total=256, inp1_unstable=2, inp2_unstable=2, delta_unstable=254
- layer_idx=2, type=LayerType.Linear, total=256, inp1_unstable=5, inp2_unstable=5, delta_unstable=256
- layer_idx=4, type=LayerType.Linear, total=256, inp1_unstable=0, inp2_unstable=0, delta_unstable=256
- layer_idx=6, type=LayerType.Linear, total=256, inp1_unstable=0, inp2_unstable=0, delta_unstable=256
- layer_idx=8, type=LayerType.Linear, total=10, inp1_unstable=0, inp2_unstable=0, delta_unstable=10

Time for backsubstitution: 1.17 seconds

### RS candidates at layer 3
type: RSZ, layer: 3, pos: 66
type: RSZ, layer: 3, pos: 129
type: RSZ, layer: 3, pos: 144
type: RSZ, layer: 3, pos: 145
type: RSZ, layer: 3, pos: 242

Time for candidate selection: 0.23 seconds

### Candidate
type: RSZ, layer: 3, pos: 66

### Relational analysis RSZ of RS_RSZ1_RSZ1_RSZ1

#### Relational analysis RSZ result of RS_RSZ1_RSZ1_RSZ1
Status: Status.UNKNOWN
Output dim: 0, lower bound: -0.0006061, upper bound: 0.0005358
time: 0.72 seconds

### Relational analysis RSZ of RS_RSZ1_RSZ1_RSZ2

#### Relational analysis RSZ result of RS_RSZ1_RSZ1_RSZ2
Status: Status.UNKNOWN
Output dim: 0, lower bound: -0.0006040, upper bound: 0.0005411
time: 0.72 seconds

## BFS RS instance: RS_RSZ1_RSZ2

### Backsubstitution after applying RS history:
0: 1.0026824, 1.0047039, 1.0026824, 1.0047039, -0.0008420, 0.0007832
1: -0.0005956, -0.0000919, -0.0005956, -0.0000919, -0.0002098, 0.0001951
2: -0.0095672, -0.0068978, -0.0095672, -0.0068978, -0.0010342, 0.0011119
3: 0.0018665, 0.0030814, 0.0018665, 0.0030814, -0.0005061, 0.0004707
4: -0.0013238, -0.0008072, -0.0013238, -0.0008072, -0.0002002, 0.0002152
5: -0.0130736, -0.0097162, -0.0130736, -0.0097162, -0.0013007, 0.0013985
6: 0.0040069, 0.0048590, 0.0040069, 0.0048590, -0.0003549, 0.0003301
7: 0.0072295, 0.0094342, 0.0072295, 0.0094342, -0.0009183, 0.0008541
8: 0.0042378, 0.0053972, 0.0042378, 0.0053972, -0.0004829, 0.0004492
9: -0.0081222, -0.0067777, -0.0081222, -0.0067777, -0.0005209, 0.0005600

### Unstable ReLU Count (Linear/Conv2D Layers)
- layer_idx=0, type=LayerType.Linear, total=256, inp1_unstable=2, inp2_unstable=2, delta_unstable=254
- layer_idx=2, type=LayerType.Linear, total=256, inp1_unstable=5, inp2_unstable=5, delta_unstable=256
- layer_idx=4, type=LayerType.Linear, total=256, inp1_unstable=0, inp2_unstable=0, delta_unstable=256
- layer_idx=6, type=LayerType.Linear, total=256, inp1_unstable=0, inp2_unstable=0, delta_unstable=256
- layer_idx=8, type=LayerType.Linear, total=10, inp1_unstable=0, inp2_unstable=0, delta_unstable=10

Time for backsubstitution: 1.18 seconds

### RS candidates at layer 3
type: RSZ, layer: 3, pos: 66
type: RSZ, layer: 3, pos: 129
type: RSZ, layer: 3, pos: 144
type: RSZ, layer: 3, pos: 145
type: RSZ, layer: 3, pos: 242

Time for candidate selection: 0.22 seconds

### Candidate
type: RSZ, layer: 3, pos: 66

### Relational analysis RSZ of RS_RSZ1_RSZ2_RSZ1

#### Relational analysis RSZ result of RS_RSZ1_RSZ2_RSZ1
Status: Status.UNKNOWN
Output dim: 0, lower bound: -0.0006119, upper bound: 0.0005332
time: 0.77 seconds

### Relational analysis RSZ of RS_RSZ1_RSZ2_RSZ2

#### Relational analysis RSZ result of RS_RSZ1_RSZ2_RSZ2
Status: Status.UNKNOWN
Output dim: 0, lower bound: -0.0006084, upper bound: 0.0005393
time: 0.74 seconds

## BFS RS instance: RS_RSZ2_RSZ1

### Backsubstitution after applying RS history:
0: 1.0026824, 1.0047039, 1.0026824, 1.0047039, -0.0007832, 0.0008420
1: -0.0005956, -0.0000919, -0.0005956, -0.0000919, -0.0001951, 0.0002098
2: -0.0095672, -0.0068978, -0.0095672, -0.0068978, -0.0011119, 0.0010342
3: 0.0018665, 0.0030814, 0.0018665, 0.0030814, -0.0004707, 0.0005061
4: -0.0013238, -0.0008072, -0.0013238, -0.0008072, -0.0002152, 0.0002002
5: -0.0130736, -0.0097162, -0.0130736, -0.0097162, -0.0013985, 0.0013007
6: 0.0040069, 0.0048590, 0.0040069, 0.0048590, -0.0003301, 0.0003549
7: 0.0072295, 0.0094342, 0.0072295, 0.0094342, -0.0008541, 0.0009183
8: 0.0042378, 0.0053972, 0.0042378, 0.0053972, -0.0004492, 0.0004829
9: -0.0081222, -0.0067777, -0.0081222, -0.0067777, -0.0005600, 0.0005209

### Unstable ReLU Count (Linear/Conv2D Layers)
- layer_idx=0, type=LayerType.Linear, total=256, inp1_unstable=2, inp2_unstable=2, delta_unstable=254
- layer_idx=2, type=LayerType.Linear, total=256, inp1_unstable=5, inp2_unstable=5, delta_unstable=256
- layer_idx=4, type=LayerType.Linear, total=256, inp1_unstable=0, inp2_unstable=0, delta_unstable=256
- layer_idx=6, type=LayerType.Linear, total=256, inp1_unstable=0, inp2_unstable=0, delta_unstable=256
- layer_idx=8, type=LayerType.Linear, total=10, inp1_unstable=0, inp2_unstable=0, delta_unstable=10

Time for backsubstitution: 1.17 seconds

### RS candidates at layer 3
type: RSZ, layer: 3, pos: 66
type: RSZ, layer: 3, pos: 129
type: RSZ, layer: 3, pos: 144
type: RSZ, layer: 3, pos: 145
type: RSZ, layer: 3, pos: 242

Time for candidate selection: 0.23 seconds

### Candidate
type: RSZ, layer: 3, pos: 66

### Relational analysis RSZ of RS_RSZ2_RSZ1_RSZ1

#### Relational analysis RSZ result of RS_RSZ2_RSZ1_RSZ1
Status: Status.UNKNOWN
Output dim: 0, lower bound: -0.0005393, upper bound: 0.0006084
time: 0.72 seconds

### Relational analysis RSZ of RS_RSZ2_RSZ1_RSZ2

#### Relational analysis RSZ result of RS_RSZ2_RSZ1_RSZ2
Status: Status.UNKNOWN
Output dim: 0, lower bound: -0.0005332, upper bound: 0.0006119
time: 0.73 seconds

## BFS RS instance: RS_RSZ2_RSZ2

### Backsubstitution after applying RS history:
0: 1.0026824, 1.0047039, 1.0026824, 1.0047039, -0.0007900, 0.0008342
1: -0.0005956, -0.0000919, -0.0005956, -0.0000919, -0.0001968, 0.0002079
2: -0.0095672, -0.0068978, -0.0095672, -0.0068978, -0.0011016, 0.0010431
3: 0.0018665, 0.0030814, 0.0018665, 0.0030814, -0.0004748, 0.0005014
4: -0.0013238, -0.0008072, -0.0013238, -0.0008072, -0.0002132, 0.0002019
5: -0.0130736, -0.0097162, -0.0130736, -0.0097162, -0.0013855, 0.0013120
6: 0.0040069, 0.0048590, 0.0040069, 0.0048590, -0.0003330, 0.0003517
7: 0.0072295, 0.0094342, 0.0072295, 0.0094342, -0.0008616, 0.0009099
8: 0.0042378, 0.0053972, 0.0042378, 0.0053972, -0.0004531, 0.0004785
9: -0.0081222, -0.0067777, -0.0081222, -0.0067777, -0.0005548, 0.0005254

### Unstable ReLU Count (Linear/Conv2D Layers)
- layer_idx=0, type=LayerType.Linear, total=256, inp1_unstable=2, inp2_unstable=2, delta_unstable=254
- layer_idx=2, type=LayerType.Linear, total=256, inp1_unstable=5, inp2_unstable=5, delta_unstable=256
- layer_idx=4, type=LayerType.Linear, total=256, inp1_unstable=0, inp2_unstable=0, delta_unstable=256
- layer_idx=6, type=LayerType.Linear, total=256, inp1_unstable=0, inp2_unstable=0, delta_unstable=256
- layer_idx=8, type=LayerType.Linear, total=10, inp1_unstable=0, inp2_unstable=0, delta_unstable=10

Time for backsubstitution: 1.16 seconds

### RS candidates at layer 3
type: RSZ, layer: 3, pos: 66
type: RSZ, layer: 3, pos: 129
type: RSZ, layer: 3, pos: 144
type: RSZ, layer: 3, pos: 145
type: RSZ, layer: 3, pos: 242

Time for candidate selection: 0.23 seconds

### Candidate
type: RSZ, layer: 3, pos: 66

### Relational analysis RSZ of RS_RSZ2_RSZ2_RSZ1

#### Relational analysis RSZ result of RS_RSZ2_RSZ2_RSZ1
Status: Status.UNKNOWN
Output dim: 0, lower bound: -0.0005410, upper bound: 0.0006040
time: 0.74 seconds

### Relational analysis RSZ of RS_RSZ2_RSZ2_RSZ2

#### Relational analysis RSZ result of RS_RSZ2_RSZ2_RSZ2
Status: Status.UNKNOWN
Output dim: 0, lower bound: -0.0005357, upper bound: 0.0006061
time: 0.76 seconds

## Summary of splitting (split count: 2)
- Time for RS candidates: 2.90 seconds
RS_RSZ1_RSZ1_RSZ1, status: Status.UNKNOWN, split count: 3, time: 2.90
Output dim: 0, lower bound: -0.0006061, upper bound: 0.0005358
RS_RSZ1_RSZ1_RSZ2, status: Status.UNKNOWN, split count: 3, time: 2.90
Output dim: 0, lower bound: -0.0006040, upper bound: 0.0005411
RS_RSZ1_RSZ2_RSZ1, status: Status.UNKNOWN, split count: 3, time: 2.90
Output dim: 0, lower bound: -0.0006119, upper bound: 0.0005332
RS_RSZ1_RSZ2_RSZ2, status: Status.UNKNOWN, split count: 3, time: 2.90
Output dim: 0, lower bound: -0.0006084, upper bound: 0.0005393
RS_RSZ2_RSZ1_RSZ1, status: Status.UNKNOWN, split count: 3, time: 2.90
Output dim: 0, lower bound: -0.0005393, upper bound: 0.0006084
RS_RSZ2_RSZ1_RSZ2, status: Status.UNKNOWN, split count: 3, time: 2.90
Output dim: 0, lower bound: -0.0005332, upper bound: 0.0006119
RS_RSZ2_RSZ2_RSZ1, status: Status.UNKNOWN, split count: 3, time: 2.90
Output dim: 0, lower bound: -0.0005410, upper bound: 0.0006040
RS_RSZ2_RSZ2_RSZ2, status: Status.UNKNOWN, split count: 3, time: 2.90
Output dim: 0, lower bound: -0.0005357, upper bound: 0.0006061

## BFS RS instance: RS_RSZ1_RSZ1_RSZ1

### Backsubstitution after applying RS history:
0: 1.0026824, 1.0047039, 1.0026824, 1.0047039, -0.0008050, 0.0007534
1: -0.0005956, -0.0000919, -0.0005956, -0.0000919, -0.0002006, 0.0001877
2: -0.0095672, -0.0068978, -0.0095672, -0.0068978, -0.0009948, 0.0010629
3: 0.0018665, 0.0030814, 0.0018665, 0.0030814, -0.0004838, 0.0004528
4: -0.0013238, -0.0008072, -0.0013238, -0.0008072, -0.0001925, 0.0002057
5: -0.0130736, -0.0097162, -0.0130736, -0.0097162, -0.0012512, 0.0013369
6: 0.0040069, 0.0048590, 0.0040069, 0.0048590, -0.0003393, 0.0003176
7: 0.0072295, 0.0094342, 0.0072295, 0.0094342, -0.0008779, 0.0008216
8: 0.0042378, 0.0053972, 0.0042378, 0.0053972, -0.0004617, 0.0004321
9: -0.0081222, -0.0067777, -0.0081222, -0.0067777, -0.0005010, 0.0005354

### Unstable ReLU Count (Linear/Conv2D Layers)
- layer_idx=0, type=LayerType.Linear, total=256, inp1_unstable=2, inp2_unstable=2, delta_unstable=254
- layer_idx=2, type=LayerType.Linear, total=256, inp1_unstable=5, inp2_unstable=5, delta_unstable=255
- layer_idx=4, type=LayerType.Linear, total=256, inp1_unstable=0, inp2_unstable=0, delta_unstable=256
- layer_idx=6, type=LayerType.Linear, total=256, inp1_unstable=0, inp2_unstable=0, delta_unstable=256
- layer_idx=8, type=LayerType.Linear, total=10, inp1_unstable=0, inp2_unstable=0, delta_unstable=10

Time for backsubstitution: 1.16 seconds

### RS candidates at layer 3
type: RSZ, layer: 3, pos: 129
type: RSZ, layer: 3, pos: 144
type: RSZ, layer: 3, pos: 145
type: RSZ, layer: 3, pos: 242

Time for candidate selection: 0.12 seconds

### Candidate
type: RSZ, layer: 3, pos: 129

### Relational analysis RSZ of RS_RSZ1_RSZ1_RSZ1_RSZ1

#### Relational analysis RSZ result of RS_RSZ1_RSZ1_RSZ1_RSZ1
Status: Status.VERIFIED
Output dim: 0, lower bound: -0.0005528, upper bound: 0.0004791
time: 0.76 seconds

### Relational analysis RSZ of RS_RSZ1_RSZ1_RSZ1_RSZ2

#### Relational analysis RSZ result of RS_RSZ1_RSZ1_RSZ1_RSZ2
Status: Status.VERIFIED
Output dim: 0, lower bound: -0.0005493, upper bound: 0.0004788
time: 0.80 seconds

## BFS RS instance: RS_RSZ1_RSZ1_RSZ2

### Backsubstitution after applying RS history:
0: 1.0026824, 1.0047039, 1.0026824, 1.0047039, -0.0008003, 0.0007607
1: -0.0005956, -0.0000919, -0.0005956, -0.0000919, -0.0001994, 0.0001895
2: -0.0095672, -0.0068978, -0.0095672, -0.0068978, -0.0010045, 0.0010568
3: 0.0018665, 0.0030814, 0.0018665, 0.0030814, -0.0004810, 0.0004572
4: -0.0013238, -0.0008072, -0.0013238, -0.0008072, -0.0001944, 0.0002045
5: -0.0130736, -0.0097162, -0.0130736, -0.0097162, -0.0012634, 0.0013291
6: 0.0040069, 0.0048590, 0.0040069, 0.0048590, -0.0003373, 0.0003207
7: 0.0072295, 0.0094342, 0.0072295, 0.0094342, -0.0008728, 0.0008296
8: 0.0042378, 0.0053972, 0.0042378, 0.0053972, -0.0004590, 0.0004363
9: -0.0081222, -0.0067777, -0.0081222, -0.0067777, -0.0005059, 0.0005322

### Unstable ReLU Count (Linear/Conv2D Layers)
- layer_idx=0, type=LayerType.Linear, total=256, inp1_unstable=2, inp2_unstable=2, delta_unstable=254
- layer_idx=2, type=LayerType.Linear, total=256, inp1_unstable=5, inp2_unstable=5, delta_unstable=255
- layer_idx=4, type=LayerType.Linear, total=256, inp1_unstable=0, inp2_unstable=0, delta_unstable=256
- layer_idx=6, type=LayerType.Linear, total=256, inp1_unstable=0, inp2_unstable=0, delta_unstable=256
- layer_idx=8, type=LayerType.Linear, total=10, inp1_unstable=0, inp2_unstable=0, delta_unstable=10

Time for backsubstitution: 1.20 seconds

### RS candidates at layer 3
type: RSZ, layer: 3, pos: 129
type: RSZ, layer: 3, pos: 144
type: RSZ, layer: 3, pos: 145
type: RSZ, layer: 3, pos: 242

Time for candidate selection: 0.13 seconds

### Candidate
type: RSZ, layer: 3, pos: 129

### Relational analysis RSZ of RS_RSZ1_RSZ1_RSZ2_RSZ1

#### Relational analysis RSZ result of RS_RSZ1_RSZ1_RSZ2_RSZ1
Status: Status.VERIFIED
Output dim: 0, lower bound: -0.0005499, upper bound: 0.0004849
time: 0.78 seconds

### Relational analysis RSZ of RS_RSZ1_RSZ1_RSZ2_RSZ2

#### Relational analysis RSZ result of RS_RSZ1_RSZ1_RSZ2_RSZ2
Status: Status.VERIFIED
Output dim: 0, lower bound: -0.0005474, upper bound: 0.0004849
time: 0.76 seconds

## BFS RS instance: RS_RSZ1_RSZ2_RSZ1

### Backsubstitution after applying RS history:
0: 1.0026824, 1.0047039, 1.0026824, 1.0047039, -0.0008127, 0.0007473
1: -0.0005956, -0.0000919, -0.0005956, -0.0000919, -0.0002025, 0.0001862
2: -0.0095672, -0.0068978, -0.0095672, -0.0068978, -0.0009868, 0.0010732
3: 0.0018665, 0.0030814, 0.0018665, 0.0030814, -0.0004885, 0.0004491
4: -0.0013238, -0.0008072, -0.0013238, -0.0008072, -0.0001910, 0.0002077
5: -0.0130736, -0.0097162, -0.0130736, -0.0097162, -0.0012411, 0.0013498
6: 0.0040069, 0.0048590, 0.0040069, 0.0048590, -0.0003426, 0.0003150
7: 0.0072295, 0.0094342, 0.0072295, 0.0094342, -0.0008864, 0.0008150
8: 0.0042378, 0.0053972, 0.0042378, 0.0053972, -0.0004662, 0.0004286
9: -0.0081222, -0.0067777, -0.0081222, -0.0067777, -0.0004970, 0.0005405

### Unstable ReLU Count (Linear/Conv2D Layers)
- layer_idx=0, type=LayerType.Linear, total=256, inp1_unstable=2, inp2_unstable=2, delta_unstable=254
- layer_idx=2, type=LayerType.Linear, total=256, inp1_unstable=5, inp2_unstable=5, delta_unstable=255
- layer_idx=4, type=LayerType.Linear, total=256, inp1_unstable=0, inp2_unstable=0, delta_unstable=256
- layer_idx=6, type=LayerType.Linear, total=256, inp1_unstable=0, inp2_unstable=0, delta_unstable=256
- layer_idx=8, type=LayerType.Linear, total=10, inp1_unstable=0, inp2_unstable=0, delta_unstable=10

Time for backsubstitution: 1.17 seconds

### RS candidates at layer 3
type: RSZ, layer: 3, pos: 129
type: RSZ, layer: 3, pos: 144
type: RSZ, layer: 3, pos: 145
type: RSZ, layer: 3, pos: 242

Time for candidate selection: 0.13 seconds

### Candidate
type: RSZ, layer: 3, pos: 129

### Relational analysis RSZ of RS_RSZ1_RSZ2_RSZ1_RSZ1

#### Relational analysis RSZ result of RS_RSZ1_RSZ2_RSZ1_RSZ1
Status: Status.VERIFIED
Output dim: 0, lower bound: -0.0005578, upper bound: 0.0004768
time: 0.81 seconds

### Relational analysis RSZ of RS_RSZ1_RSZ2_RSZ1_RSZ2

#### Relational analysis RSZ result of RS_RSZ1_RSZ2_RSZ1_RSZ2
Status: Status.VERIFIED
Output dim: 0, lower bound: -0.0005564, upper bound: 0.0004767
time: 0.77 seconds

## BFS RS instance: RS_RSZ1_RSZ2_RSZ2

### Backsubstitution after applying RS history:
0: 1.0026824, 1.0047039, 1.0026824, 1.0047039, -0.0008081, 0.0007539
1: -0.0005956, -0.0000919, -0.0005956, -0.0000919, -0.0002014, 0.0001878
2: -0.0095672, -0.0068978, -0.0095672, -0.0068978, -0.0009955, 0.0010671
3: 0.0018665, 0.0030814, 0.0018665, 0.0030814, -0.0004857, 0.0004531
4: -0.0013238, -0.0008072, -0.0013238, -0.0008072, -0.0001927, 0.0002065
5: -0.0130736, -0.0097162, -0.0130736, -0.0097162, -0.0012521, 0.0013421
6: 0.0040069, 0.0048590, 0.0040069, 0.0048590, -0.0003406, 0.0003178
7: 0.0072295, 0.0094342, 0.0072295, 0.0094342, -0.0008813, 0.0008222
8: 0.0042378, 0.0053972, 0.0042378, 0.0053972, -0.0004635, 0.0004324
9: -0.0081222, -0.0067777, -0.0081222, -0.0067777, -0.0005014, 0.0005374

### Unstable ReLU Count (Linear/Conv2D Layers)
- layer_idx=0, type=LayerType.Linear, total=256, inp1_unstable=2, inp2_unstable=2, delta_unstable=254
- layer_idx=2, type=LayerType.Linear, total=256, inp1_unstable=5, inp2_unstable=5, delta_unstable=255
- layer_idx=4, type=LayerType.Linear, total=256, inp1_unstable=0, inp2_unstable=0, delta_unstable=256
- layer_idx=6, type=LayerType.Linear, total=256, inp1_unstable=0, inp2_unstable=0, delta_unstable=256
- layer_idx=8, type=LayerType.Linear, total=10, inp1_unstable=0, inp2_unstable=0, delta_unstable=10

Time for backsubstitution: 1.22 seconds

### RS candidates at layer 3
type: RSZ, layer: 3, pos: 129
type: RSZ, layer: 3, pos: 144
type: RSZ, layer: 3, pos: 145
type: RSZ, layer: 3, pos: 242

Time for candidate selection: 0.15 seconds

### Candidate
type: RSZ, layer: 3, pos: 129

### Relational analysis RSZ of RS_RSZ1_RSZ2_RSZ2_RSZ1

#### Relational analysis RSZ result of RS_RSZ1_RSZ2_RSZ2_RSZ1
Status: Status.VERIFIED
Output dim: 0, lower bound: -0.0005536, upper bound: 0.0004832
time: 0.82 seconds

### Relational analysis RSZ of RS_RSZ1_RSZ2_RSZ2_RSZ2

#### Relational analysis RSZ result of RS_RSZ1_RSZ2_RSZ2_RSZ2
Status: Status.VERIFIED
Output dim: 0, lower bound: -0.0005533, upper bound: 0.0004831
time: 0.80 seconds

## BFS RS instance: RS_RSZ2_RSZ1_RSZ1

### Backsubstitution after applying RS history:
0: 1.0026824, 1.0047039, 1.0026824, 1.0047039, -0.0007539, 0.0008081
1: -0.0005956, -0.0000919, -0.0005956, -0.0000919, -0.0001878, 0.0002014
2: -0.0095672, -0.0068978, -0.0095672, -0.0068978, -0.0010671, 0.0009955
3: 0.0018665, 0.0030814, 0.0018665, 0.0030814, -0.0004531, 0.0004857
4: -0.0013238, -0.0008072, -0.0013238, -0.0008072, -0.0002065, 0.0001927
5: -0.0130736, -0.0097162, -0.0130736, -0.0097162, -0.0013421, 0.0012521
6: 0.0040069, 0.0048590, 0.0040069, 0.0048590, -0.0003178, 0.0003406
7: 0.0072295, 0.0094342, 0.0072295, 0.0094342, -0.0008222, 0.0008813
8: 0.0042378, 0.0053972, 0.0042378, 0.0053972, -0.0004324, 0.0004635
9: -0.0081222, -0.0067777, -0.0081222, -0.0067777, -0.0005374, 0.0005014

### Unstable ReLU Count (Linear/Conv2D Layers)
- layer_idx=0, type=LayerType.Linear, total=256, inp1_unstable=2, inp2_unstable=2, delta_unstable=254
- layer_idx=2, type=LayerType.Linear, total=256, inp1_unstable=5, inp2_unstable=5, delta_unstable=255
- layer_idx=4, type=LayerType.Linear, total=256, inp1_unstable=0, inp2_unstable=0, delta_unstable=256
- layer_idx=6, type=LayerType.Linear, total=256, inp1_unstable=0, inp2_unstable=0, delta_unstable=256
- layer_idx=8, type=LayerType.Linear, total=10, inp1_unstable=0, inp2_unstable=0, delta_unstable=10

Time for backsubstitution: 1.18 seconds

### RS candidates at layer 3
type: RSZ, layer: 3, pos: 129
type: RSZ, layer: 3, pos: 144
type: RSZ, layer: 3, pos: 145
type: RSZ, layer: 3, pos: 242

Time for candidate selection: 0.12 seconds

### Candidate
type: RSZ, layer: 3, pos: 129

### Relational analysis RSZ of RS_RSZ2_RSZ1_RSZ1_RSZ1

#### Relational analysis RSZ result of RS_RSZ2_RSZ1_RSZ1_RSZ1
Status: Status.VERIFIED
Output dim: 0, lower bound: -0.0004831, upper bound: 0.0005533
time: 0.76 seconds

### Relational analysis RSZ of RS_RSZ2_RSZ1_RSZ1_RSZ2

#### Relational analysis RSZ result of RS_RSZ2_RSZ1_RSZ1_RSZ2
Status: Status.VERIFIED
Output dim: 0, lower bound: -0.0004831, upper bound: 0.0005536
time: 0.77 seconds

## BFS RS instance: RS_RSZ2_RSZ1_RSZ2

### Backsubstitution after applying RS history:
0: 1.0026824, 1.0047039, 1.0026824, 1.0047039, -0.0007473, 0.0008127
1: -0.0005956, -0.0000919, -0.0005956, -0.0000919, -0.0001862, 0.0002025
2: -0.0095672, -0.0068978, -0.0095672, -0.0068978, -0.0010732, 0.0009868
3: 0.0018665, 0.0030814, 0.0018665, 0.0030814, -0.0004491, 0.0004885
4: -0.0013238, -0.0008072, -0.0013238, -0.0008072, -0.0002077, 0.0001910
5: -0.0130736, -0.0097162, -0.0130736, -0.0097162, -0.0013498, 0.0012411
6: 0.0040069, 0.0048590, 0.0040069, 0.0048590, -0.0003150, 0.0003426
7: 0.0072295, 0.0094342, 0.0072295, 0.0094342, -0.0008150, 0.0008864
8: 0.0042378, 0.0053972, 0.0042378, 0.0053972, -0.0004286, 0.0004662
9: -0.0081222, -0.0067777, -0.0081222, -0.0067777, -0.0005405, 0.0004970

### Unstable ReLU Count (Linear/Conv2D Layers)
- layer_idx=0, type=LayerType.Linear, total=256, inp1_unstable=2, inp2_unstable=2, delta_unstable=254
- layer_idx=2, type=LayerType.Linear, total=256, inp1_unstable=5, inp2_unstable=5, delta_unstable=255
- layer_idx=4, type=LayerType.Linear, total=256, inp1_unstable=0, inp2_unstable=0, delta_unstable=256
- layer_idx=6, type=LayerType.Linear, total=256, inp1_unstable=0, inp2_unstable=0, delta_unstable=256
- layer_idx=8, type=LayerType.Linear, total=10, inp1_unstable=0, inp2_unstable=0, delta_unstable=10

Time for backsubstitution: 1.22 seconds

### RS candidates at layer 3
type: RSZ, layer: 3, pos: 129
type: RSZ, layer: 3, pos: 144
type: RSZ, layer: 3, pos: 145
type: RSZ, layer: 3, pos: 242

Time for candidate selection: 0.13 seconds

### Candidate
type: RSZ, layer: 3, pos: 129

### Relational analysis RSZ of RS_RSZ2_RSZ1_RSZ2_RSZ1

#### Relational analysis RSZ result of RS_RSZ2_RSZ1_RSZ2_RSZ1
Status: Status.VERIFIED
Output dim: 0, lower bound: -0.0004767, upper bound: 0.0005564
time: 0.82 seconds

### Relational analysis RSZ of RS_RSZ2_RSZ1_RSZ2_RSZ2

#### Relational analysis RSZ result of RS_RSZ2_RSZ1_RSZ2_RSZ2
Status: Status.VERIFIED
Output dim: 0, lower bound: -0.0004768, upper bound: 0.0005578
time: 0.81 seconds

## BFS RS instance: RS_RSZ2_RSZ2_RSZ1

### Backsubstitution after applying RS history:
0: 1.0026824, 1.0047039, 1.0026824, 1.0047039, -0.0007607, 0.0008003
1: -0.0005956, -0.0000919, -0.0005956, -0.0000919, -0.0001895, 0.0001994
2: -0.0095672, -0.0068978, -0.0095672, -0.0068978, -0.0010568, 0.0010045
3: 0.0018665, 0.0030814, 0.0018665, 0.0030814, -0.0004572, 0.0004810
4: -0.0013238, -0.0008072, -0.0013238, -0.0008072, -0.0002045, 0.0001944
5: -0.0130736, -0.0097162, -0.0130736, -0.0097162, -0.0013291, 0.0012634
6: 0.0040069, 0.0048590, 0.0040069, 0.0048590, -0.0003207, 0.0003373
7: 0.0072295, 0.0094342, 0.0072295, 0.0094342, -0.0008296, 0.0008728
8: 0.0042378, 0.0053972, 0.0042378, 0.0053972, -0.0004363, 0.0004590
9: -0.0081222, -0.0067777, -0.0081222, -0.0067777, -0.0005322, 0.0005059

### Unstable ReLU Count (Linear/Conv2D Layers)
- layer_idx=0, type=LayerType.Linear, total=256, inp1_unstable=2, inp2_unstable=2, delta_unstable=254
- layer_idx=2, type=LayerType.Linear, total=256, inp1_unstable=5, inp2_unstable=5, delta_unstable=255
- layer_idx=4, type=LayerType.Linear, total=256, inp1_unstable=0, inp2_unstable=0, delta_unstable=256
- layer_idx=6, type=LayerType.Linear, total=256, inp1_unstable=0, inp2_unstable=0, delta_unstable=256
- layer_idx=8, type=LayerType.Linear, total=10, inp1_unstable=0, inp2_unstable=0, delta_unstable=10

Time for backsubstitution: 1.21 seconds

### RS candidates at layer 3
type: RSZ, layer: 3, pos: 129
type: RSZ, layer: 3, pos: 144
type: RSZ, layer: 3, pos: 145
type: RSZ, layer: 3, pos: 242

Time for candidate selection: 0.12 seconds

### Candidate
type: RSZ, layer: 3, pos: 129

### Relational analysis RSZ of RS_RSZ2_RSZ2_RSZ1_RSZ1

#### Relational analysis RSZ result of RS_RSZ2_RSZ2_RSZ1_RSZ1
Status: Status.VERIFIED
Output dim: 0, lower bound: -0.0004849, upper bound: 0.0005474
time: 0.79 seconds

### Relational analysis RSZ of RS_RSZ2_RSZ2_RSZ1_RSZ2

#### Relational analysis RSZ result of RS_RSZ2_RSZ2_RSZ1_RSZ2
Status: Status.VERIFIED
Output dim: 0, lower bound: -0.0004849, upper bound: 0.0005499
time: 0.78 seconds

## BFS RS instance: RS_RSZ2_RSZ2_RSZ2

### Backsubstitution after applying RS history:
0: 1.0026824, 1.0047039, 1.0026824, 1.0047039, -0.0007534, 0.0008050
1: -0.0005956, -0.0000919, -0.0005956, -0.0000919, -0.0001877, 0.0002006
2: -0.0095672, -0.0068978, -0.0095672, -0.0068978, -0.0010629, 0.0009948
3: 0.0018665, 0.0030814, 0.0018665, 0.0030814, -0.0004528, 0.0004838
4: -0.0013238, -0.0008072, -0.0013238, -0.0008072, -0.0002057, 0.0001925
5: -0.0130736, -0.0097162, -0.0130736, -0.0097162, -0.0013369, 0.0012512
6: 0.0040069, 0.0048590, 0.0040069, 0.0048590, -0.0003176, 0.0003393
7: 0.0072295, 0.0094342, 0.0072295, 0.0094342, -0.0008216, 0.0008779
8: 0.0042378, 0.0053972, 0.0042378, 0.0053972, -0.0004321, 0.0004617
9: -0.0081222, -0.0067777, -0.0081222, -0.0067777, -0.0005354, 0.0005010

### Unstable ReLU Count (Linear/Conv2D Layers)
- layer_idx=0, type=LayerType.Linear, total=256, inp1_unstable=2, inp2_unstable=2, delta_unstable=254
- layer_idx=2, type=LayerType.Linear, total=256, inp1_unstable=5, inp2_unstable=5, delta_unstable=255
- layer_idx=4, type=LayerType.Linear, total=256, inp1_unstable=0, inp2_unstable=0, delta_unstable=256
- layer_idx=6, type=LayerType.Linear, total=256, inp1_unstable=0, inp2_unstable=0, delta_unstable=256
- layer_idx=8, type=LayerType.Linear, total=10, inp1_unstable=0, inp2_unstable=0, delta_unstable=10

Time for backsubstitution: 1.21 seconds

### RS candidates at layer 3
type: RSZ, layer: 3, pos: 129
type: RSZ, layer: 3, pos: 144
type: RSZ, layer: 3, pos: 145
type: RSZ, layer: 3, pos: 242

Time for candidate selection: 0.12 seconds

### Candidate
type: RSZ, layer: 3, pos: 129

### Relational analysis RSZ of RS_RSZ2_RSZ2_RSZ2_RSZ1

#### Relational analysis RSZ result of RS_RSZ2_RSZ2_RSZ2_RSZ1
Status: Status.VERIFIED
Output dim: 0, lower bound: -0.0004787, upper bound: 0.0005493
time: 0.82 seconds

### Relational analysis RSZ of RS_RSZ2_RSZ2_RSZ2_RSZ2

#### Relational analysis RSZ result of RS_RSZ2_RSZ2_RSZ2_RSZ2
Status: Status.VERIFIED
Output dim: 0, lower bound: -0.0004791, upper bound: 0.0005528
time: 0.76 seconds

## Summary of splitting (split count: 3)
- Time for RS candidates: 2.92 seconds
RS_RSZ1_RSZ1_RSZ1_RSZ1, status: Status.VERIFIED, split count: 4, time: 2.92
Output dim: 0, lower bound: -0.0005528, upper bound: 0.0004791
RS_RSZ1_RSZ1_RSZ1_RSZ2, status: Status.VERIFIED, split count: 4, time: 2.92
Output dim: 0, lower bound: -0.0005493, upper bound: 0.0004788
RS_RSZ1_RSZ1_RSZ2_RSZ1, status: Status.VERIFIED, split count: 4, time: 2.92
Output dim: 0, lower bound: -0.0005499, upper bound: 0.0004849
RS_RSZ1_RSZ1_RSZ2_RSZ2, status: Status.VERIFIED, split count: 4, time: 2.92
Output dim: 0, lower bound: -0.0005474, upper bound: 0.0004849
RS_RSZ1_RSZ2_RSZ1_RSZ1, status: Status.VERIFIED, split count: 4, time: 2.92
Output dim: 0, lower bound: -0.0005578, upper bound: 0.0004768
RS_RSZ1_RSZ2_RSZ1_RSZ2, status: Status.VERIFIED, split count: 4, time: 2.92
Output dim: 0, lower bound: -0.0005564, upper bound: 0.0004767
RS_RSZ1_RSZ2_RSZ2_RSZ1, status: Status.VERIFIED, split count: 4, time: 2.92
Output dim: 0, lower bound: -0.0005536, upper bound: 0.0004832
RS_RSZ1_RSZ2_RSZ2_RSZ2, status: Status.VERIFIED, split count: 4, time: 2.92
Output dim: 0, lower bound: -0.0005533, upper bound: 0.0004831
RS_RSZ2_RSZ1_RSZ1_RSZ1, status: Status.VERIFIED, split count: 4, time: 2.92
Output dim: 0, lower bound: -0.0004831, upper bound: 0.0005533
RS_RSZ2_RSZ1_RSZ1_RSZ2, status: Status.VERIFIED, split count: 4, time: 2.92
Output dim: 0, lower bound: -0.0004831, upper bound: 0.0005536
RS_RSZ2_RSZ1_RSZ2_RSZ1, status: Status.VERIFIED, split count: 4, time: 2.92
Output dim: 0, lower bound: -0.0004767, upper bound: 0.0005564
RS_RSZ2_RSZ1_RSZ2_RSZ2, status: Status.VERIFIED, split count: 4, time: 2.92
Output dim: 0, lower bound: -0.0004768, upper bound: 0.0005578
RS_RSZ2_RSZ2_RSZ1_RSZ1, status: Status.VERIFIED, split count: 4, time: 2.92
Output dim: 0, lower bound: -0.0004849, upper bound: 0.0005474
RS_RSZ2_RSZ2_RSZ1_RSZ2, status: Status.VERIFIED, split count: 4, time: 2.92
Output dim: 0, lower bound: -0.0004849, upper bound: 0.0005499
RS_RSZ2_RSZ2_RSZ2_RSZ1, status: Status.VERIFIED, split count: 4, time: 2.92
Output dim: 0, lower bound: -0.0004787, upper bound: 0.0005493
RS_RSZ2_RSZ2_RSZ2_RSZ2, status: Status.VERIFIED, split count: 4, time: 2.92
Output dim: 0, lower bound: -0.0004791, upper bound: 0.0005528

## RS Result
status: Status.VERIFIED
execution time: (base) + (rs) = 2.83 + 41.97 = 44.81 seconds
