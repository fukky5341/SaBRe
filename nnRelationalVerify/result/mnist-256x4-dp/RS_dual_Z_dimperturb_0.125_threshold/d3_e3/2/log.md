## Execution arguments:
Dataset: Dataset.MNIST
Network: onnx/mnist-net_256x4.onnx
Relational property: GLOBAL_ROBUSTNESS
LP Analysis: True
Epsilon: 0.03515625
Delta epsilon: 0.01171875
execution index: (3, 3, 2)
Time budget: 600 seconds
Split limit: 100
Threshold: 0.00058113


## IAR start

### Unstable ReLU Count (Linear/Conv2D Layers)
- layer_idx=0, type=LayerType.Linear, total=256, inp1_unstable=10, inp2_unstable=10, delta_unstable=256
- layer_idx=2, type=LayerType.Linear, total=256, inp1_unstable=8, inp2_unstable=8, delta_unstable=256
- layer_idx=4, type=LayerType.Linear, total=256, inp1_unstable=0, inp2_unstable=0, delta_unstable=256
- layer_idx=6, type=LayerType.Linear, total=256, inp1_unstable=2, inp2_unstable=2, delta_unstable=256
- layer_idx=8, type=LayerType.Linear, total=10, inp1_unstable=0, inp2_unstable=0, delta_unstable=10

### BASE IAR bounds
Layer (inp1_lb, inp1_ub, inp2_lb, inp2_ub, d_lb, d_ub)
0: (0.0006996, 0.0011177, 0.0006996, 0.0011177, -0.0004181, 0.0004181)
1: (0.9934013, 0.9943976, 0.9934013, 0.9943976, -0.0009964, 0.0009964)
2: (-0.0088595, -0.0053283, -0.0088595, -0.0053283, -0.0032878, 0.0032878)
3: (0.0035633, 0.0041662, 0.0035633, 0.0041662, -0.0006029, 0.0006029)
4: (0.0026282, 0.0054191, 0.0026282, 0.0054191, -0.0027909, 0.0027909)
5: (0.0050201, 0.0064446, 0.0050201, 0.0064446, -0.0014245, 0.0014245)
6: (-0.0021963, -0.0008150, -0.0021963, -0.0008150, -0.0013813, 0.0013813)
7: (-0.0082640, -0.0074494, -0.0082640, -0.0074494, -0.0008146, 0.0008146)
8: (0.0052524, 0.0098919, 0.0052524, 0.0098919, -0.0045745, 0.0045745)
9: (-0.0036845, -0.0030730, -0.0036845, -0.0030730, -0.0006115, 0.0006115)

## BASE Relational Analysis

## BASE Result
execution time: IAR + RelationalAnalysis = 1.75 + 1.48 = 3.22 seconds
status: Status.UNKNOWN
relational distance
Output dim: 1, lower bound: -0.0008551, upper bound: 0.0008551

# Relational Split (RS) starts

## BFS RS instance: RS

Time for backsubstitution: 0.00 seconds

### RS candidates at layer 1
type: RSZ, layer: 1, pos: 52
type: RSZ, layer: 1, pos: 68
type: RSZ, layer: 1, pos: 78
type: RSZ, layer: 1, pos: 83
type: RSZ, layer: 1, pos: 136
type: RSZ, layer: 1, pos: 176
type: RSZ, layer: 1, pos: 198
type: RSZ, layer: 1, pos: 199
type: RSZ, layer: 1, pos: 204
type: RSZ, layer: 1, pos: 250

Time for candidate selection: 0.16 seconds

### Candidate
type: RSZ, layer: 1, pos: 52

### Relational analysis RSZ of RS_RSZ1

#### Relational analysis RSZ result of RS_RSZ1
Status: Status.UNKNOWN
Output dim: 1, lower bound: -0.0008551, upper bound: 0.0008551
time: 0.57 seconds

### Relational analysis RSZ of RS_RSZ2

#### Relational analysis RSZ result of RS_RSZ2
Status: Status.UNKNOWN
Output dim: 1, lower bound: -0.0008551, upper bound: 0.0008551
time: 0.58 seconds

## Summary of splitting (split count: 0)
- Time for RS candidates: 1.32 seconds
RS_RSZ1, status: Status.UNKNOWN, split count: 1, time: 1.32
Output dim: 1, lower bound: -0.0008551, upper bound: 0.0008551
RS_RSZ2, status: Status.UNKNOWN, split count: 1, time: 1.32
Output dim: 1, lower bound: -0.0008551, upper bound: 0.0008551

## BFS RS instance: RS_RSZ1

### Backsubstitution after applying RS history:
0: 0.0006996, 0.0011177, 0.0006996, 0.0011177, -0.0004181, 0.0004181
1: 0.9934013, 0.9943976, 0.9934013, 0.9943976, -0.0009964, 0.0009964
2: -0.0088595, -0.0053283, -0.0088595, -0.0053283, -0.0032847, 0.0032847
3: 0.0035633, 0.0041662, 0.0035633, 0.0041662, -0.0006029, 0.0006029
4: 0.0026282, 0.0054191, 0.0026282, 0.0054191, -0.0027909, 0.0027909
5: 0.0050201, 0.0064446, 0.0050201, 0.0064446, -0.0014245, 0.0014245
6: -0.0021963, -0.0008150, -0.0021963, -0.0008150, -0.0013813, 0.0013813
7: -0.0082640, -0.0074494, -0.0082640, -0.0074494, -0.0008146, 0.0008146
8: 0.0052524, 0.0098919, 0.0052524, 0.0098919, -0.0045737, 0.0045737
9: -0.0036845, -0.0030730, -0.0036845, -0.0030730, -0.0006115, 0.0006115

### Unstable ReLU Count (Linear/Conv2D Layers)
- layer_idx=0, type=LayerType.Linear, total=256, inp1_unstable=10, inp2_unstable=10, delta_unstable=255
- layer_idx=2, type=LayerType.Linear, total=256, inp1_unstable=8, inp2_unstable=8, delta_unstable=256
- layer_idx=4, type=LayerType.Linear, total=256, inp1_unstable=0, inp2_unstable=0, delta_unstable=256
- layer_idx=6, type=LayerType.Linear, total=256, inp1_unstable=2, inp2_unstable=2, delta_unstable=256
- layer_idx=8, type=LayerType.Linear, total=10, inp1_unstable=0, inp2_unstable=0, delta_unstable=10

Time for backsubstitution: 1.65 seconds

### RS candidates at layer 1
type: RSZ, layer: 1, pos: 68
type: RSZ, layer: 1, pos: 78
type: RSZ, layer: 1, pos: 83
type: RSZ, layer: 1, pos: 136
type: RSZ, layer: 1, pos: 176
type: RSZ, layer: 1, pos: 198
type: RSZ, layer: 1, pos: 199
type: RSZ, layer: 1, pos: 204
type: RSZ, layer: 1, pos: 250

Time for candidate selection: 0.19 seconds

### Candidate
type: RSZ, layer: 1, pos: 68

### Relational analysis RSZ of RS_RSZ1_RSZ1

#### Relational analysis RSZ result of RS_RSZ1_RSZ1
Status: Status.UNKNOWN
Output dim: 1, lower bound: -0.0008524, upper bound: 0.0008524
time: 0.64 seconds

### Relational analysis RSZ of RS_RSZ1_RSZ2

#### Relational analysis RSZ result of RS_RSZ1_RSZ2
Status: Status.UNKNOWN
Output dim: 1, lower bound: -0.0008524, upper bound: 0.0008524
time: 0.58 seconds

## BFS RS instance: RS_RSZ2

### Backsubstitution after applying RS history:
0: 0.0006996, 0.0011177, 0.0006996, 0.0011177, -0.0004181, 0.0004181
1: 0.9934013, 0.9943976, 0.9934013, 0.9943976, -0.0009964, 0.0009964
2: -0.0088595, -0.0053283, -0.0088595, -0.0053283, -0.0032847, 0.0032847
3: 0.0035633, 0.0041662, 0.0035633, 0.0041662, -0.0006029, 0.0006029
4: 0.0026282, 0.0054191, 0.0026282, 0.0054191, -0.0027909, 0.0027909
5: 0.0050201, 0.0064446, 0.0050201, 0.0064446, -0.0014245, 0.0014245
6: -0.0021963, -0.0008150, -0.0021963, -0.0008150, -0.0013813, 0.0013813
7: -0.0082640, -0.0074494, -0.0082640, -0.0074494, -0.0008146, 0.0008146
8: 0.0052524, 0.0098919, 0.0052524, 0.0098919, -0.0045737, 0.0045737
9: -0.0036845, -0.0030730, -0.0036845, -0.0030730, -0.0006115, 0.0006115

### Unstable ReLU Count (Linear/Conv2D Layers)
- layer_idx=0, type=LayerType.Linear, total=256, inp1_unstable=10, inp2_unstable=10, delta_unstable=255
- layer_idx=2, type=LayerType.Linear, total=256, inp1_unstable=8, inp2_unstable=8, delta_unstable=256
- layer_idx=4, type=LayerType.Linear, total=256, inp1_unstable=0, inp2_unstable=0, delta_unstable=256
- layer_idx=6, type=LayerType.Linear, total=256, inp1_unstable=2, inp2_unstable=2, delta_unstable=256
- layer_idx=8, type=LayerType.Linear, total=10, inp1_unstable=0, inp2_unstable=0, delta_unstable=10

Time for backsubstitution: 1.63 seconds

### RS candidates at layer 1
type: RSZ, layer: 1, pos: 68
type: RSZ, layer: 1, pos: 78
type: RSZ, layer: 1, pos: 83
type: RSZ, layer: 1, pos: 136
type: RSZ, layer: 1, pos: 176
type: RSZ, layer: 1, pos: 198
type: RSZ, layer: 1, pos: 199
type: RSZ, layer: 1, pos: 204
type: RSZ, layer: 1, pos: 250

Time for candidate selection: 0.17 seconds

### Candidate
type: RSZ, layer: 1, pos: 68

### Relational analysis RSZ of RS_RSZ2_RSZ1

#### Relational analysis RSZ result of RS_RSZ2_RSZ1
Status: Status.UNKNOWN
Output dim: 1, lower bound: -0.0008524, upper bound: 0.0008524
time: 0.66 seconds

### Relational analysis RSZ of RS_RSZ2_RSZ2

#### Relational analysis RSZ result of RS_RSZ2_RSZ2
Status: Status.UNKNOWN
Output dim: 1, lower bound: -0.0008524, upper bound: 0.0008524
time: 0.62 seconds

## Summary of splitting (split count: 1)
- Time for RS candidates: 3.09 seconds
RS_RSZ1_RSZ1, status: Status.UNKNOWN, split count: 2, time: 3.09
Output dim: 1, lower bound: -0.0008524, upper bound: 0.0008524
RS_RSZ1_RSZ2, status: Status.UNKNOWN, split count: 2, time: 3.09
Output dim: 1, lower bound: -0.0008524, upper bound: 0.0008524
RS_RSZ2_RSZ1, status: Status.UNKNOWN, split count: 2, time: 3.09
Output dim: 1, lower bound: -0.0008524, upper bound: 0.0008524
RS_RSZ2_RSZ2, status: Status.UNKNOWN, split count: 2, time: 3.09
Output dim: 1, lower bound: -0.0008524, upper bound: 0.0008524

## BFS RS instance: RS_RSZ1_RSZ1

### Backsubstitution after applying RS history:
0: 0.0006996, 0.0011177, 0.0006996, 0.0011177, -0.0004181, 0.0004181
1: 0.9934013, 0.9943976, 0.9934013, 0.9943976, -0.0009964, 0.0009964
2: -0.0088595, -0.0053283, -0.0088595, -0.0053283, -0.0032716, 0.0032692
3: 0.0035633, 0.0041662, 0.0035633, 0.0041662, -0.0006029, 0.0006029
4: 0.0026282, 0.0054191, 0.0026282, 0.0054191, -0.0027909, 0.0027909
5: 0.0050201, 0.0064446, 0.0050201, 0.0064446, -0.0014245, 0.0014245
6: -0.0021963, -0.0008150, -0.0021963, -0.0008150, -0.0013813, 0.0013813
7: -0.0082640, -0.0074494, -0.0082640, -0.0074494, -0.0008146, 0.0008146
8: 0.0052524, 0.0098919, 0.0052524, 0.0098919, -0.0045694, 0.0045701
9: -0.0036845, -0.0030730, -0.0036845, -0.0030730, -0.0006115, 0.0006115

### Unstable ReLU Count (Linear/Conv2D Layers)
- layer_idx=0, type=LayerType.Linear, total=256, inp1_unstable=10, inp2_unstable=10, delta_unstable=254
- layer_idx=2, type=LayerType.Linear, total=256, inp1_unstable=8, inp2_unstable=8, delta_unstable=256
- layer_idx=4, type=LayerType.Linear, total=256, inp1_unstable=0, inp2_unstable=0, delta_unstable=256
- layer_idx=6, type=LayerType.Linear, total=256, inp1_unstable=2, inp2_unstable=2, delta_unstable=256
- layer_idx=8, type=LayerType.Linear, total=10, inp1_unstable=0, inp2_unstable=0, delta_unstable=10

Time for backsubstitution: 1.63 seconds

### RS candidates at layer 1
type: RSZ, layer: 1, pos: 78
type: RSZ, layer: 1, pos: 83
type: RSZ, layer: 1, pos: 136
type: RSZ, layer: 1, pos: 176
type: RSZ, layer: 1, pos: 198
type: RSZ, layer: 1, pos: 199
type: RSZ, layer: 1, pos: 204
type: RSZ, layer: 1, pos: 250

Time for candidate selection: 0.17 seconds

### Candidate
type: RSZ, layer: 1, pos: 78

### Relational analysis RSZ of RS_RSZ1_RSZ1_RSZ1

#### Relational analysis RSZ result of RS_RSZ1_RSZ1_RSZ1
Status: Status.UNKNOWN
Output dim: 1, lower bound: -0.0008505, upper bound: 0.0008505
time: 0.65 seconds

### Relational analysis RSZ of RS_RSZ1_RSZ1_RSZ2

#### Relational analysis RSZ result of RS_RSZ1_RSZ1_RSZ2
Status: Status.UNKNOWN
Output dim: 1, lower bound: -0.0008505, upper bound: 0.0008505
time: 0.69 seconds

## BFS RS instance: RS_RSZ1_RSZ2

### Backsubstitution after applying RS history:
0: 0.0006996, 0.0011177, 0.0006996, 0.0011177, -0.0004181, 0.0004181
1: 0.9934013, 0.9943976, 0.9934013, 0.9943976, -0.0009964, 0.0009964
2: -0.0088595, -0.0053283, -0.0088595, -0.0053283, -0.0032691, 0.0032721
3: 0.0035633, 0.0041662, 0.0035633, 0.0041662, -0.0006029, 0.0006029
4: 0.0026282, 0.0054191, 0.0026282, 0.0054191, -0.0027909, 0.0027909
5: 0.0050201, 0.0064446, 0.0050201, 0.0064446, -0.0014245, 0.0014245
6: -0.0021963, -0.0008150, -0.0021963, -0.0008150, -0.0013813, 0.0013813
7: -0.0082640, -0.0074494, -0.0082640, -0.0074494, -0.0008146, 0.0008146
8: 0.0052524, 0.0098919, 0.0052524, 0.0098919, -0.0045702, 0.0045694
9: -0.0036845, -0.0030730, -0.0036845, -0.0030730, -0.0006115, 0.0006115

### Unstable ReLU Count (Linear/Conv2D Layers)
- layer_idx=0, type=LayerType.Linear, total=256, inp1_unstable=10, inp2_unstable=10, delta_unstable=254
- layer_idx=2, type=LayerType.Linear, total=256, inp1_unstable=8, inp2_unstable=8, delta_unstable=256
- layer_idx=4, type=LayerType.Linear, total=256, inp1_unstable=0, inp2_unstable=0, delta_unstable=256
- layer_idx=6, type=LayerType.Linear, total=256, inp1_unstable=2, inp2_unstable=2, delta_unstable=256
- layer_idx=8, type=LayerType.Linear, total=10, inp1_unstable=0, inp2_unstable=0, delta_unstable=10

Time for backsubstitution: 1.69 seconds

### RS candidates at layer 1
type: RSZ, layer: 1, pos: 78
type: RSZ, layer: 1, pos: 83
type: RSZ, layer: 1, pos: 136
type: RSZ, layer: 1, pos: 176
type: RSZ, layer: 1, pos: 198
type: RSZ, layer: 1, pos: 199
type: RSZ, layer: 1, pos: 204
type: RSZ, layer: 1, pos: 250

Time for candidate selection: 0.16 seconds

### Candidate
type: RSZ, layer: 1, pos: 78

### Relational analysis RSZ of RS_RSZ1_RSZ2_RSZ1

#### Relational analysis RSZ result of RS_RSZ1_RSZ2_RSZ1
Status: Status.UNKNOWN
Output dim: 1, lower bound: -0.0008505, upper bound: 0.0008505
time: 0.67 seconds

### Relational analysis RSZ of RS_RSZ1_RSZ2_RSZ2

#### Relational analysis RSZ result of RS_RSZ1_RSZ2_RSZ2
Status: Status.UNKNOWN
Output dim: 1, lower bound: -0.0008505, upper bound: 0.0008505
time: 0.65 seconds

## BFS RS instance: RS_RSZ2_RSZ1

### Backsubstitution after applying RS history:
0: 0.0006996, 0.0011177, 0.0006996, 0.0011177, -0.0004181, 0.0004181
1: 0.9934013, 0.9943976, 0.9934013, 0.9943976, -0.0009964, 0.0009964
2: -0.0088595, -0.0053283, -0.0088595, -0.0053283, -0.0032721, 0.0032691
3: 0.0035633, 0.0041662, 0.0035633, 0.0041662, -0.0006029, 0.0006029
4: 0.0026282, 0.0054191, 0.0026282, 0.0054191, -0.0027909, 0.0027909
5: 0.0050201, 0.0064446, 0.0050201, 0.0064446, -0.0014245, 0.0014245
6: -0.0021963, -0.0008150, -0.0021963, -0.0008150, -0.0013813, 0.0013813
7: -0.0082640, -0.0074494, -0.0082640, -0.0074494, -0.0008146, 0.0008146
8: 0.0052524, 0.0098919, 0.0052524, 0.0098919, -0.0045694, 0.0045702
9: -0.0036845, -0.0030730, -0.0036845, -0.0030730, -0.0006115, 0.0006115

### Unstable ReLU Count (Linear/Conv2D Layers)
- layer_idx=0, type=LayerType.Linear, total=256, inp1_unstable=10, inp2_unstable=10, delta_unstable=254
- layer_idx=2, type=LayerType.Linear, total=256, inp1_unstable=8, inp2_unstable=8, delta_unstable=256
- layer_idx=4, type=LayerType.Linear, total=256, inp1_unstable=0, inp2_unstable=0, delta_unstable=256
- layer_idx=6, type=LayerType.Linear, total=256, inp1_unstable=2, inp2_unstable=2, delta_unstable=256
- layer_idx=8, type=LayerType.Linear, total=10, inp1_unstable=0, inp2_unstable=0, delta_unstable=10

Time for backsubstitution: 1.63 seconds

### RS candidates at layer 1
type: RSZ, layer: 1, pos: 78
type: RSZ, layer: 1, pos: 83
type: RSZ, layer: 1, pos: 136
type: RSZ, layer: 1, pos: 176
type: RSZ, layer: 1, pos: 198
type: RSZ, layer: 1, pos: 199
type: RSZ, layer: 1, pos: 204
type: RSZ, layer: 1, pos: 250

Time for candidate selection: 0.21 seconds

### Candidate
type: RSZ, layer: 1, pos: 78

### Relational analysis RSZ of RS_RSZ2_RSZ1_RSZ1

#### Relational analysis RSZ result of RS_RSZ2_RSZ1_RSZ1
Status: Status.UNKNOWN
Output dim: 1, lower bound: -0.0008505, upper bound: 0.0008505
time: 0.66 seconds

### Relational analysis RSZ of RS_RSZ2_RSZ1_RSZ2

#### Relational analysis RSZ result of RS_RSZ2_RSZ1_RSZ2
Status: Status.UNKNOWN
Output dim: 1, lower bound: -0.0008505, upper bound: 0.0008505
time: 0.67 seconds

## BFS RS instance: RS_RSZ2_RSZ2

### Backsubstitution after applying RS history:
0: 0.0006996, 0.0011177, 0.0006996, 0.0011177, -0.0004181, 0.0004181
1: 0.9934013, 0.9943976, 0.9934013, 0.9943976, -0.0009964, 0.0009964
2: -0.0088595, -0.0053283, -0.0088595, -0.0053283, -0.0032692, 0.0032716
3: 0.0035633, 0.0041662, 0.0035633, 0.0041662, -0.0006029, 0.0006029
4: 0.0026282, 0.0054191, 0.0026282, 0.0054191, -0.0027909, 0.0027909
5: 0.0050201, 0.0064446, 0.0050201, 0.0064446, -0.0014245, 0.0014245
6: -0.0021963, -0.0008150, -0.0021963, -0.0008150, -0.0013813, 0.0013813
7: -0.0082640, -0.0074494, -0.0082640, -0.0074494, -0.0008146, 0.0008146
8: 0.0052524, 0.0098919, 0.0052524, 0.0098919, -0.0045701, 0.0045695
9: -0.0036845, -0.0030730, -0.0036845, -0.0030730, -0.0006115, 0.0006115

### Unstable ReLU Count (Linear/Conv2D Layers)
- layer_idx=0, type=LayerType.Linear, total=256, inp1_unstable=10, inp2_unstable=10, delta_unstable=254
- layer_idx=2, type=LayerType.Linear, total=256, inp1_unstable=8, inp2_unstable=8, delta_unstable=256
- layer_idx=4, type=LayerType.Linear, total=256, inp1_unstable=0, inp2_unstable=0, delta_unstable=256
- layer_idx=6, type=LayerType.Linear, total=256, inp1_unstable=2, inp2_unstable=2, delta_unstable=256
- layer_idx=8, type=LayerType.Linear, total=10, inp1_unstable=0, inp2_unstable=0, delta_unstable=10

Time for backsubstitution: 1.62 seconds

### RS candidates at layer 1
type: RSZ, layer: 1, pos: 78
type: RSZ, layer: 1, pos: 83
type: RSZ, layer: 1, pos: 136
type: RSZ, layer: 1, pos: 176
type: RSZ, layer: 1, pos: 198
type: RSZ, layer: 1, pos: 199
type: RSZ, layer: 1, pos: 204
type: RSZ, layer: 1, pos: 250

Time for candidate selection: 0.17 seconds

### Candidate
type: RSZ, layer: 1, pos: 78

### Relational analysis RSZ of RS_RSZ2_RSZ2_RSZ1

#### Relational analysis RSZ result of RS_RSZ2_RSZ2_RSZ1
Status: Status.UNKNOWN
Output dim: 1, lower bound: -0.0008505, upper bound: 0.0008505
time: 0.71 seconds

### Relational analysis RSZ of RS_RSZ2_RSZ2_RSZ2

#### Relational analysis RSZ result of RS_RSZ2_RSZ2_RSZ2
Status: Status.UNKNOWN
Output dim: 1, lower bound: -0.0008505, upper bound: 0.0008505
time: 0.64 seconds

## Summary of splitting (split count: 2)
- Time for RS candidates: 3.16 seconds
RS_RSZ1_RSZ1_RSZ1, status: Status.UNKNOWN, split count: 3, time: 3.16
Output dim: 1, lower bound: -0.0008505, upper bound: 0.0008505
RS_RSZ1_RSZ1_RSZ2, status: Status.UNKNOWN, split count: 3, time: 3.16
Output dim: 1, lower bound: -0.0008505, upper bound: 0.0008505
RS_RSZ1_RSZ2_RSZ1, status: Status.UNKNOWN, split count: 3, time: 3.16
Output dim: 1, lower bound: -0.0008505, upper bound: 0.0008505
RS_RSZ1_RSZ2_RSZ2, status: Status.UNKNOWN, split count: 3, time: 3.16
Output dim: 1, lower bound: -0.0008505, upper bound: 0.0008505
RS_RSZ2_RSZ1_RSZ1, status: Status.UNKNOWN, split count: 3, time: 3.16
Output dim: 1, lower bound: -0.0008505, upper bound: 0.0008505
RS_RSZ2_RSZ1_RSZ2, status: Status.UNKNOWN, split count: 3, time: 3.16
Output dim: 1, lower bound: -0.0008505, upper bound: 0.0008505
RS_RSZ2_RSZ2_RSZ1, status: Status.UNKNOWN, split count: 3, time: 3.16
Output dim: 1, lower bound: -0.0008505, upper bound: 0.0008505
RS_RSZ2_RSZ2_RSZ2, status: Status.UNKNOWN, split count: 3, time: 3.16
Output dim: 1, lower bound: -0.0008505, upper bound: 0.0008505

## BFS RS instance: RS_RSZ1_RSZ1_RSZ1

### Backsubstitution after applying RS history:
0: 0.0006996, 0.0011177, 0.0006996, 0.0011177, -0.0004181, 0.0004181
1: 0.9934013, 0.9943976, 0.9934013, 0.9943976, -0.0009964, 0.0009964
2: -0.0088595, -0.0053283, -0.0088595, -0.0053283, -0.0032710, 0.0032689
3: 0.0035633, 0.0041662, 0.0035633, 0.0041662, -0.0006029, 0.0006029
4: 0.0026282, 0.0054191, 0.0026282, 0.0054191, -0.0027909, 0.0027909
5: 0.0050201, 0.0064446, 0.0050201, 0.0064446, -0.0014245, 0.0014245
6: -0.0021963, -0.0008150, -0.0021963, -0.0008150, -0.0013813, 0.0013813
7: -0.0082640, -0.0074494, -0.0082640, -0.0074494, -0.0008146, 0.0008146
8: 0.0052524, 0.0098919, 0.0052524, 0.0098919, -0.0045693, 0.0045699
9: -0.0036845, -0.0030730, -0.0036845, -0.0030730, -0.0006115, 0.0006115

### Unstable ReLU Count (Linear/Conv2D Layers)
- layer_idx=0, type=LayerType.Linear, total=256, inp1_unstable=10, inp2_unstable=10, delta_unstable=253
- layer_idx=2, type=LayerType.Linear, total=256, inp1_unstable=8, inp2_unstable=8, delta_unstable=256
- layer_idx=4, type=LayerType.Linear, total=256, inp1_unstable=0, inp2_unstable=0, delta_unstable=256
- layer_idx=6, type=LayerType.Linear, total=256, inp1_unstable=2, inp2_unstable=2, delta_unstable=256
- layer_idx=8, type=LayerType.Linear, total=10, inp1_unstable=0, inp2_unstable=0, delta_unstable=10

Time for backsubstitution: 1.61 seconds

### RS candidates at layer 1
type: RSZ, layer: 1, pos: 83
type: RSZ, layer: 1, pos: 136
type: RSZ, layer: 1, pos: 176
type: RSZ, layer: 1, pos: 198
type: RSZ, layer: 1, pos: 199
type: RSZ, layer: 1, pos: 204
type: RSZ, layer: 1, pos: 250

Time for candidate selection: 0.16 seconds

### Candidate
type: RSZ, layer: 1, pos: 83

### Relational analysis RSZ of RS_RSZ1_RSZ1_RSZ1_RSZ1

#### Relational analysis RSZ result of RS_RSZ1_RSZ1_RSZ1_RSZ1
Status: Status.UNKNOWN
Output dim: 1, lower bound: -0.0008426, upper bound: 0.0008426
time: 0.65 seconds

### Relational analysis RSZ of RS_RSZ1_RSZ1_RSZ1_RSZ2

#### Relational analysis RSZ result of RS_RSZ1_RSZ1_RSZ1_RSZ2
Status: Status.UNKNOWN
Output dim: 1, lower bound: -0.0008426, upper bound: 0.0008426
time: 0.63 seconds

## BFS RS instance: RS_RSZ1_RSZ1_RSZ2

### Backsubstitution after applying RS history:
0: 0.0006996, 0.0011177, 0.0006996, 0.0011177, -0.0004181, 0.0004181
1: 0.9934013, 0.9943976, 0.9934013, 0.9943976, -0.0009964, 0.0009964
2: -0.0088595, -0.0053283, -0.0088595, -0.0053283, -0.0032708, 0.0032685
3: 0.0035633, 0.0041662, 0.0035633, 0.0041662, -0.0006029, 0.0006029
4: 0.0026282, 0.0054191, 0.0026282, 0.0054191, -0.0027909, 0.0027909
5: 0.0050201, 0.0064446, 0.0050201, 0.0064446, -0.0014245, 0.0014245
6: -0.0021963, -0.0008150, -0.0021963, -0.0008150, -0.0013813, 0.0013813
7: -0.0082640, -0.0074494, -0.0082640, -0.0074494, -0.0008146, 0.0008146
8: 0.0052524, 0.0098919, 0.0052524, 0.0098919, -0.0045692, 0.0045699
9: -0.0036845, -0.0030730, -0.0036845, -0.0030730, -0.0006115, 0.0006115

### Unstable ReLU Count (Linear/Conv2D Layers)
- layer_idx=0, type=LayerType.Linear, total=256, inp1_unstable=10, inp2_unstable=10, delta_unstable=253
- layer_idx=2, type=LayerType.Linear, total=256, inp1_unstable=8, inp2_unstable=8, delta_unstable=256
- layer_idx=4, type=LayerType.Linear, total=256, inp1_unstable=0, inp2_unstable=0, delta_unstable=256
- layer_idx=6, type=LayerType.Linear, total=256, inp1_unstable=2, inp2_unstable=2, delta_unstable=256
- layer_idx=8, type=LayerType.Linear, total=10, inp1_unstable=0, inp2_unstable=0, delta_unstable=10

Time for backsubstitution: 1.60 seconds

### RS candidates at layer 1
type: RSZ, layer: 1, pos: 83
type: RSZ, layer: 1, pos: 136
type: RSZ, layer: 1, pos: 176
type: RSZ, layer: 1, pos: 198
type: RSZ, layer: 1, pos: 199
type: RSZ, layer: 1, pos: 204
type: RSZ, layer: 1, pos: 250

Time for candidate selection: 0.17 seconds

### Candidate
type: RSZ, layer: 1, pos: 83

### Relational analysis RSZ of RS_RSZ1_RSZ1_RSZ2_RSZ1

#### Relational analysis RSZ result of RS_RSZ1_RSZ1_RSZ2_RSZ1
Status: Status.UNKNOWN
Output dim: 1, lower bound: -0.0008426, upper bound: 0.0008426
time: 0.67 seconds

### Relational analysis RSZ of RS_RSZ1_RSZ1_RSZ2_RSZ2

#### Relational analysis RSZ result of RS_RSZ1_RSZ1_RSZ2_RSZ2
Status: Status.UNKNOWN
Output dim: 1, lower bound: -0.0008426, upper bound: 0.0008426
time: 0.66 seconds

## BFS RS instance: RS_RSZ1_RSZ2_RSZ1

### Backsubstitution after applying RS history:
0: 0.0006996, 0.0011177, 0.0006996, 0.0011177, -0.0004181, 0.0004181
1: 0.9934013, 0.9943976, 0.9934013, 0.9943976, -0.0009964, 0.0009964
2: -0.0088595, -0.0053283, -0.0088595, -0.0053283, -0.0032685, 0.0032714
3: 0.0035633, 0.0041662, 0.0035633, 0.0041662, -0.0006029, 0.0006029
4: 0.0026282, 0.0054191, 0.0026282, 0.0054191, -0.0027909, 0.0027909
5: 0.0050201, 0.0064446, 0.0050201, 0.0064446, -0.0014245, 0.0014245
6: -0.0021963, -0.0008150, -0.0021963, -0.0008150, -0.0013813, 0.0013813
7: -0.0082640, -0.0074494, -0.0082640, -0.0074494, -0.0008146, 0.0008146
8: 0.0052524, 0.0098919, 0.0052524, 0.0098919, -0.0045700, 0.0045692
9: -0.0036845, -0.0030730, -0.0036845, -0.0030730, -0.0006115, 0.0006115

### Unstable ReLU Count (Linear/Conv2D Layers)
- layer_idx=0, type=LayerType.Linear, total=256, inp1_unstable=10, inp2_unstable=10, delta_unstable=253
- layer_idx=2, type=LayerType.Linear, total=256, inp1_unstable=8, inp2_unstable=8, delta_unstable=256
- layer_idx=4, type=LayerType.Linear, total=256, inp1_unstable=0, inp2_unstable=0, delta_unstable=256
- layer_idx=6, type=LayerType.Linear, total=256, inp1_unstable=2, inp2_unstable=2, delta_unstable=256
- layer_idx=8, type=LayerType.Linear, total=10, inp1_unstable=0, inp2_unstable=0, delta_unstable=10

Time for backsubstitution: 1.58 seconds

### RS candidates at layer 1
type: RSZ, layer: 1, pos: 83
type: RSZ, layer: 1, pos: 136
type: RSZ, layer: 1, pos: 176
type: RSZ, layer: 1, pos: 198
type: RSZ, layer: 1, pos: 199
type: RSZ, layer: 1, pos: 204
type: RSZ, layer: 1, pos: 250

Time for candidate selection: 0.17 seconds

### Candidate
type: RSZ, layer: 1, pos: 83

### Relational analysis RSZ of RS_RSZ1_RSZ2_RSZ1_RSZ1

#### Relational analysis RSZ result of RS_RSZ1_RSZ2_RSZ1_RSZ1
Status: Status.UNKNOWN
Output dim: 1, lower bound: -0.0008426, upper bound: 0.0008426
time: 0.66 seconds

### Relational analysis RSZ of RS_RSZ1_RSZ2_RSZ1_RSZ2

#### Relational analysis RSZ result of RS_RSZ1_RSZ2_RSZ1_RSZ2
Status: Status.UNKNOWN
Output dim: 1, lower bound: -0.0008426, upper bound: 0.0008426
time: 0.69 seconds

## BFS RS instance: RS_RSZ1_RSZ2_RSZ2

### Backsubstitution after applying RS history:
0: 0.0006996, 0.0011177, 0.0006996, 0.0011177, -0.0004181, 0.0004181
1: 0.9934013, 0.9943976, 0.9934013, 0.9943976, -0.0009964, 0.0009964
2: -0.0088595, -0.0053283, -0.0088595, -0.0053283, -0.0032684, 0.0032714
3: 0.0035633, 0.0041662, 0.0035633, 0.0041662, -0.0006029, 0.0006029
4: 0.0026282, 0.0054191, 0.0026282, 0.0054191, -0.0027909, 0.0027909
5: 0.0050201, 0.0064446, 0.0050201, 0.0064446, -0.0014245, 0.0014245
6: -0.0021963, -0.0008150, -0.0021963, -0.0008150, -0.0013813, 0.0013813
7: -0.0082640, -0.0074494, -0.0082640, -0.0074494, -0.0008146, 0.0008146
8: 0.0052524, 0.0098919, 0.0052524, 0.0098919, -0.0045700, 0.0045692
9: -0.0036845, -0.0030730, -0.0036845, -0.0030730, -0.0006115, 0.0006115

### Unstable ReLU Count (Linear/Conv2D Layers)
- layer_idx=0, type=LayerType.Linear, total=256, inp1_unstable=10, inp2_unstable=10, delta_unstable=253
- layer_idx=2, type=LayerType.Linear, total=256, inp1_unstable=8, inp2_unstable=8, delta_unstable=256
- layer_idx=4, type=LayerType.Linear, total=256, inp1_unstable=0, inp2_unstable=0, delta_unstable=256
- layer_idx=6, type=LayerType.Linear, total=256, inp1_unstable=2, inp2_unstable=2, delta_unstable=256
- layer_idx=8, type=LayerType.Linear, total=10, inp1_unstable=0, inp2_unstable=0, delta_unstable=10

Time for backsubstitution: 1.83 seconds

### RS candidates at layer 1
type: RSZ, layer: 1, pos: 83
type: RSZ, layer: 1, pos: 136
type: RSZ, layer: 1, pos: 176
type: RSZ, layer: 1, pos: 198
type: RSZ, layer: 1, pos: 199
type: RSZ, layer: 1, pos: 204
type: RSZ, layer: 1, pos: 250

Time for candidate selection: 0.22 seconds

### Candidate
type: RSZ, layer: 1, pos: 83

### Relational analysis RSZ of RS_RSZ1_RSZ2_RSZ2_RSZ1

#### Relational analysis RSZ result of RS_RSZ1_RSZ2_RSZ2_RSZ1
Status: Status.UNKNOWN
Output dim: 1, lower bound: -0.0008426, upper bound: 0.0008426
time: 0.62 seconds

### Relational analysis RSZ of RS_RSZ1_RSZ2_RSZ2_RSZ2

#### Relational analysis RSZ result of RS_RSZ1_RSZ2_RSZ2_RSZ2
Status: Status.UNKNOWN
Output dim: 1, lower bound: -0.0008426, upper bound: 0.0008426
time: 0.61 seconds

## BFS RS instance: RS_RSZ2_RSZ1_RSZ1

### Backsubstitution after applying RS history:
0: 0.0006996, 0.0011177, 0.0006996, 0.0011177, -0.0004181, 0.0004181
1: 0.9934013, 0.9943976, 0.9934013, 0.9943976, -0.0009964, 0.0009964
2: -0.0088595, -0.0053283, -0.0088595, -0.0053283, -0.0032714, 0.0032684
3: 0.0035633, 0.0041662, 0.0035633, 0.0041662, -0.0006029, 0.0006029
4: 0.0026282, 0.0054191, 0.0026282, 0.0054191, -0.0027909, 0.0027909
5: 0.0050201, 0.0064446, 0.0050201, 0.0064446, -0.0014245, 0.0014245
6: -0.0021963, -0.0008150, -0.0021963, -0.0008150, -0.0013813, 0.0013813
7: -0.0082640, -0.0074494, -0.0082640, -0.0074494, -0.0008146, 0.0008146
8: 0.0052524, 0.0098919, 0.0052524, 0.0098919, -0.0045692, 0.0045700
9: -0.0036845, -0.0030730, -0.0036845, -0.0030730, -0.0006115, 0.0006115

### Unstable ReLU Count (Linear/Conv2D Layers)
- layer_idx=0, type=LayerType.Linear, total=256, inp1_unstable=10, inp2_unstable=10, delta_unstable=253
- layer_idx=2, type=LayerType.Linear, total=256, inp1_unstable=8, inp2_unstable=8, delta_unstable=256
- layer_idx=4, type=LayerType.Linear, total=256, inp1_unstable=0, inp2_unstable=0, delta_unstable=256
- layer_idx=6, type=LayerType.Linear, total=256, inp1_unstable=2, inp2_unstable=2, delta_unstable=256
- layer_idx=8, type=LayerType.Linear, total=10, inp1_unstable=0, inp2_unstable=0, delta_unstable=10

Time for backsubstitution: 1.63 seconds

### RS candidates at layer 1
type: RSZ, layer: 1, pos: 83
type: RSZ, layer: 1, pos: 136
type: RSZ, layer: 1, pos: 176
type: RSZ, layer: 1, pos: 198
type: RSZ, layer: 1, pos: 199
type: RSZ, layer: 1, pos: 204
type: RSZ, layer: 1, pos: 250

Time for candidate selection: 0.18 seconds

### Candidate
type: RSZ, layer: 1, pos: 83

### Relational analysis RSZ of RS_RSZ2_RSZ1_RSZ1_RSZ1

#### Relational analysis RSZ result of RS_RSZ2_RSZ1_RSZ1_RSZ1
Status: Status.UNKNOWN
Output dim: 1, lower bound: -0.0008426, upper bound: 0.0008426
time: 0.64 seconds

### Relational analysis RSZ of RS_RSZ2_RSZ1_RSZ1_RSZ2

#### Relational analysis RSZ result of RS_RSZ2_RSZ1_RSZ1_RSZ2
Status: Status.UNKNOWN
Output dim: 1, lower bound: -0.0008426, upper bound: 0.0008426
time: 0.66 seconds

## BFS RS instance: RS_RSZ2_RSZ1_RSZ2

### Backsubstitution after applying RS history:
0: 0.0006996, 0.0011177, 0.0006996, 0.0011177, -0.0004181, 0.0004181
1: 0.9934013, 0.9943976, 0.9934013, 0.9943976, -0.0009964, 0.0009964
2: -0.0088595, -0.0053283, -0.0088595, -0.0053283, -0.0032714, 0.0032685
3: 0.0035633, 0.0041662, 0.0035633, 0.0041662, -0.0006029, 0.0006029
4: 0.0026282, 0.0054191, 0.0026282, 0.0054191, -0.0027909, 0.0027909
5: 0.0050201, 0.0064446, 0.0050201, 0.0064446, -0.0014245, 0.0014245
6: -0.0021963, -0.0008150, -0.0021963, -0.0008150, -0.0013813, 0.0013813
7: -0.0082640, -0.0074494, -0.0082640, -0.0074494, -0.0008146, 0.0008146
8: 0.0052524, 0.0098919, 0.0052524, 0.0098919, -0.0045692, 0.0045700
9: -0.0036845, -0.0030730, -0.0036845, -0.0030730, -0.0006115, 0.0006115

### Unstable ReLU Count (Linear/Conv2D Layers)
- layer_idx=0, type=LayerType.Linear, total=256, inp1_unstable=10, inp2_unstable=10, delta_unstable=253
- layer_idx=2, type=LayerType.Linear, total=256, inp1_unstable=8, inp2_unstable=8, delta_unstable=256
- layer_idx=4, type=LayerType.Linear, total=256, inp1_unstable=0, inp2_unstable=0, delta_unstable=256
- layer_idx=6, type=LayerType.Linear, total=256, inp1_unstable=2, inp2_unstable=2, delta_unstable=256
- layer_idx=8, type=LayerType.Linear, total=10, inp1_unstable=0, inp2_unstable=0, delta_unstable=10

Time for backsubstitution: 1.64 seconds

### RS candidates at layer 1
type: RSZ, layer: 1, pos: 83
type: RSZ, layer: 1, pos: 136
type: RSZ, layer: 1, pos: 176
type: RSZ, layer: 1, pos: 198
type: RSZ, layer: 1, pos: 199
type: RSZ, layer: 1, pos: 204
type: RSZ, layer: 1, pos: 250

Time for candidate selection: 0.18 seconds

### Candidate
type: RSZ, layer: 1, pos: 83

### Relational analysis RSZ of RS_RSZ2_RSZ1_RSZ2_RSZ1

#### Relational analysis RSZ result of RS_RSZ2_RSZ1_RSZ2_RSZ1
Status: Status.UNKNOWN
Output dim: 1, lower bound: -0.0008426, upper bound: 0.0008426
time: 0.63 seconds

### Relational analysis RSZ of RS_RSZ2_RSZ1_RSZ2_RSZ2

#### Relational analysis RSZ result of RS_RSZ2_RSZ1_RSZ2_RSZ2
Status: Status.UNKNOWN
Output dim: 1, lower bound: -0.0008426, upper bound: 0.0008426
time: 0.63 seconds

## BFS RS instance: RS_RSZ2_RSZ2_RSZ1

### Backsubstitution after applying RS history:
0: 0.0006996, 0.0011177, 0.0006996, 0.0011177, -0.0004181, 0.0004181
1: 0.9934013, 0.9943976, 0.9934013, 0.9943976, -0.0009964, 0.0009964
2: -0.0088595, -0.0053283, -0.0088595, -0.0053283, -0.0032685, 0.0032708
3: 0.0035633, 0.0041662, 0.0035633, 0.0041662, -0.0006029, 0.0006029
4: 0.0026282, 0.0054191, 0.0026282, 0.0054191, -0.0027909, 0.0027909
5: 0.0050201, 0.0064446, 0.0050201, 0.0064446, -0.0014245, 0.0014245
6: -0.0021963, -0.0008150, -0.0021963, -0.0008150, -0.0013813, 0.0013813
7: -0.0082640, -0.0074494, -0.0082640, -0.0074494, -0.0008146, 0.0008146
8: 0.0052524, 0.0098919, 0.0052524, 0.0098919, -0.0045699, 0.0045692
9: -0.0036845, -0.0030730, -0.0036845, -0.0030730, -0.0006115, 0.0006115

### Unstable ReLU Count (Linear/Conv2D Layers)
- layer_idx=0, type=LayerType.Linear, total=256, inp1_unstable=10, inp2_unstable=10, delta_unstable=253
- layer_idx=2, type=LayerType.Linear, total=256, inp1_unstable=8, inp2_unstable=8, delta_unstable=256
- layer_idx=4, type=LayerType.Linear, total=256, inp1_unstable=0, inp2_unstable=0, delta_unstable=256
- layer_idx=6, type=LayerType.Linear, total=256, inp1_unstable=2, inp2_unstable=2, delta_unstable=256
- layer_idx=8, type=LayerType.Linear, total=10, inp1_unstable=0, inp2_unstable=0, delta_unstable=10

Time for backsubstitution: 1.62 seconds

### RS candidates at layer 1
type: RSZ, layer: 1, pos: 83
type: RSZ, layer: 1, pos: 136
type: RSZ, layer: 1, pos: 176
type: RSZ, layer: 1, pos: 198
type: RSZ, layer: 1, pos: 199
type: RSZ, layer: 1, pos: 204
type: RSZ, layer: 1, pos: 250

Time for candidate selection: 0.17 seconds

### Candidate
type: RSZ, layer: 1, pos: 83

### Relational analysis RSZ of RS_RSZ2_RSZ2_RSZ1_RSZ1

#### Relational analysis RSZ result of RS_RSZ2_RSZ2_RSZ1_RSZ1
Status: Status.UNKNOWN
Output dim: 1, lower bound: -0.0008426, upper bound: 0.0008426
time: 0.63 seconds

### Relational analysis RSZ of RS_RSZ2_RSZ2_RSZ1_RSZ2

#### Relational analysis RSZ result of RS_RSZ2_RSZ2_RSZ1_RSZ2
Status: Status.UNKNOWN
Output dim: 1, lower bound: -0.0008426, upper bound: 0.0008426
time: 0.66 seconds

## BFS RS instance: RS_RSZ2_RSZ2_RSZ2

### Backsubstitution after applying RS history:
0: 0.0006996, 0.0011177, 0.0006996, 0.0011177, -0.0004181, 0.0004181
1: 0.9934013, 0.9943976, 0.9934013, 0.9943976, -0.0009964, 0.0009964
2: -0.0088595, -0.0053283, -0.0088595, -0.0053283, -0.0032689, 0.0032710
3: 0.0035633, 0.0041662, 0.0035633, 0.0041662, -0.0006029, 0.0006029
4: 0.0026282, 0.0054191, 0.0026282, 0.0054191, -0.0027909, 0.0027909
5: 0.0050201, 0.0064446, 0.0050201, 0.0064446, -0.0014245, 0.0014245
6: -0.0021963, -0.0008150, -0.0021963, -0.0008150, -0.0013813, 0.0013813
7: -0.0082640, -0.0074494, -0.0082640, -0.0074494, -0.0008146, 0.0008146
8: 0.0052524, 0.0098919, 0.0052524, 0.0098919, -0.0045699, 0.0045693
9: -0.0036845, -0.0030730, -0.0036845, -0.0030730, -0.0006115, 0.0006115

### Unstable ReLU Count (Linear/Conv2D Layers)
- layer_idx=0, type=LayerType.Linear, total=256, inp1_unstable=10, inp2_unstable=10, delta_unstable=253
- layer_idx=2, type=LayerType.Linear, total=256, inp1_unstable=8, inp2_unstable=8, delta_unstable=256
- layer_idx=4, type=LayerType.Linear, total=256, inp1_unstable=0, inp2_unstable=0, delta_unstable=256
- layer_idx=6, type=LayerType.Linear, total=256, inp1_unstable=2, inp2_unstable=2, delta_unstable=256
- layer_idx=8, type=LayerType.Linear, total=10, inp1_unstable=0, inp2_unstable=0, delta_unstable=10

Time for backsubstitution: 1.67 seconds

### RS candidates at layer 1
type: RSZ, layer: 1, pos: 83
type: RSZ, layer: 1, pos: 136
type: RSZ, layer: 1, pos: 176
type: RSZ, layer: 1, pos: 198
type: RSZ, layer: 1, pos: 199
type: RSZ, layer: 1, pos: 204
type: RSZ, layer: 1, pos: 250

Time for candidate selection: 0.17 seconds

### Candidate
type: RSZ, layer: 1, pos: 83

### Relational analysis RSZ of RS_RSZ2_RSZ2_RSZ2_RSZ1

#### Relational analysis RSZ result of RS_RSZ2_RSZ2_RSZ2_RSZ1
Status: Status.UNKNOWN
Output dim: 1, lower bound: -0.0008426, upper bound: 0.0008426
time: 0.63 seconds

### Relational analysis RSZ of RS_RSZ2_RSZ2_RSZ2_RSZ2

#### Relational analysis RSZ result of RS_RSZ2_RSZ2_RSZ2_RSZ2
Status: Status.UNKNOWN
Output dim: 1, lower bound: -0.0008426, upper bound: 0.0008426
time: 0.62 seconds

## Summary of splitting (split count: 3)
- Time for RS candidates: 3.10 seconds
RS_RSZ1_RSZ1_RSZ1_RSZ1, status: Status.UNKNOWN, split count: 4, time: 3.10
Output dim: 1, lower bound: -0.0008426, upper bound: 0.0008426
RS_RSZ1_RSZ1_RSZ1_RSZ2, status: Status.UNKNOWN, split count: 4, time: 3.10
Output dim: 1, lower bound: -0.0008426, upper bound: 0.0008426
RS_RSZ1_RSZ1_RSZ2_RSZ1, status: Status.UNKNOWN, split count: 4, time: 3.10
Output dim: 1, lower bound: -0.0008426, upper bound: 0.0008426
RS_RSZ1_RSZ1_RSZ2_RSZ2, status: Status.UNKNOWN, split count: 4, time: 3.10
Output dim: 1, lower bound: -0.0008426, upper bound: 0.0008426
RS_RSZ1_RSZ2_RSZ1_RSZ1, status: Status.UNKNOWN, split count: 4, time: 3.10
Output dim: 1, lower bound: -0.0008426, upper bound: 0.0008426
RS_RSZ1_RSZ2_RSZ1_RSZ2, status: Status.UNKNOWN, split count: 4, time: 3.10
Output dim: 1, lower bound: -0.0008426, upper bound: 0.0008426
RS_RSZ1_RSZ2_RSZ2_RSZ1, status: Status.UNKNOWN, split count: 4, time: 3.10
Output dim: 1, lower bound: -0.0008426, upper bound: 0.0008426
RS_RSZ1_RSZ2_RSZ2_RSZ2, status: Status.UNKNOWN, split count: 4, time: 3.10
Output dim: 1, lower bound: -0.0008426, upper bound: 0.0008426
RS_RSZ2_RSZ1_RSZ1_RSZ1, status: Status.UNKNOWN, split count: 4, time: 3.10
Output dim: 1, lower bound: -0.0008426, upper bound: 0.0008426
RS_RSZ2_RSZ1_RSZ1_RSZ2, status: Status.UNKNOWN, split count: 4, time: 3.10
Output dim: 1, lower bound: -0.0008426, upper bound: 0.0008426
RS_RSZ2_RSZ1_RSZ2_RSZ1, status: Status.UNKNOWN, split count: 4, time: 3.10
Output dim: 1, lower bound: -0.0008426, upper bound: 0.0008426
RS_RSZ2_RSZ1_RSZ2_RSZ2, status: Status.UNKNOWN, split count: 4, time: 3.10
Output dim: 1, lower bound: -0.0008426, upper bound: 0.0008426
RS_RSZ2_RSZ2_RSZ1_RSZ1, status: Status.UNKNOWN, split count: 4, time: 3.10
Output dim: 1, lower bound: -0.0008426, upper bound: 0.0008426
RS_RSZ2_RSZ2_RSZ1_RSZ2, status: Status.UNKNOWN, split count: 4, time: 3.10
Output dim: 1, lower bound: -0.0008426, upper bound: 0.0008426
RS_RSZ2_RSZ2_RSZ2_RSZ1, status: Status.UNKNOWN, split count: 4, time: 3.10
Output dim: 1, lower bound: -0.0008426, upper bound: 0.0008426
RS_RSZ2_RSZ2_RSZ2_RSZ2, status: Status.UNKNOWN, split count: 4, time: 3.10
Output dim: 1, lower bound: -0.0008426, upper bound: 0.0008426

## BFS RS instance: RS_RSZ1_RSZ1_RSZ1_RSZ1

### Backsubstitution after applying RS history:
0: 0.0006996, 0.0011177, 0.0006996, 0.0011177, -0.0004181, 0.0004181
1: 0.9934013, 0.9943976, 0.9934013, 0.9943976, -0.0009964, 0.0009964
2: -0.0088595, -0.0053283, -0.0088595, -0.0053283, -0.0032659, 0.0032625
3: 0.0035633, 0.0041662, 0.0035633, 0.0041662, -0.0006029, 0.0006029
4: 0.0026282, 0.0054191, 0.0026282, 0.0054191, -0.0027909, 0.0027909
5: 0.0050201, 0.0064446, 0.0050201, 0.0064446, -0.0014245, 0.0014245
6: -0.0021963, -0.0008150, -0.0021963, -0.0008150, -0.0013813, 0.0013813
7: -0.0082640, -0.0074494, -0.0082640, -0.0074494, -0.0008146, 0.0008146
8: 0.0052524, 0.0098919, 0.0052524, 0.0098919, -0.0045676, 0.0045685
9: -0.0036845, -0.0030730, -0.0036845, -0.0030730, -0.0006115, 0.0006115

### Unstable ReLU Count (Linear/Conv2D Layers)
- layer_idx=0, type=LayerType.Linear, total=256, inp1_unstable=10, inp2_unstable=10, delta_unstable=252
- layer_idx=2, type=LayerType.Linear, total=256, inp1_unstable=8, inp2_unstable=8, delta_unstable=256
- layer_idx=4, type=LayerType.Linear, total=256, inp1_unstable=0, inp2_unstable=0, delta_unstable=256
- layer_idx=6, type=LayerType.Linear, total=256, inp1_unstable=2, inp2_unstable=2, delta_unstable=256
- layer_idx=8, type=LayerType.Linear, total=10, inp1_unstable=0, inp2_unstable=0, delta_unstable=10

Time for backsubstitution: 1.59 seconds

### RS candidates at layer 1
type: RSZ, layer: 1, pos: 136
type: RSZ, layer: 1, pos: 176
type: RSZ, layer: 1, pos: 198
type: RSZ, layer: 1, pos: 199
type: RSZ, layer: 1, pos: 204
type: RSZ, layer: 1, pos: 250

Time for candidate selection: 0.17 seconds

### Candidate
type: RSZ, layer: 1, pos: 136

### Relational analysis RSZ of RS_RSZ1_RSZ1_RSZ1_RSZ1_RSZ1

#### Relational analysis RSZ result of RS_RSZ1_RSZ1_RSZ1_RSZ1_RSZ1
Status: Status.UNKNOWN
Output dim: 1, lower bound: -0.0008426, upper bound: 0.0008426
time: 0.60 seconds

### Relational analysis RSZ of RS_RSZ1_RSZ1_RSZ1_RSZ1_RSZ2

#### Relational analysis RSZ result of RS_RSZ1_RSZ1_RSZ1_RSZ1_RSZ2
Status: Status.UNKNOWN
Output dim: 1, lower bound: -0.0008426, upper bound: 0.0008426
time: 0.77 seconds

## BFS RS instance: RS_RSZ1_RSZ1_RSZ1_RSZ2

### Backsubstitution after applying RS history:
0: 0.0006996, 0.0011177, 0.0006996, 0.0011177, -0.0004181, 0.0004181
1: 0.9934013, 0.9943976, 0.9934013, 0.9943976, -0.0009964, 0.0009964
2: -0.0088595, -0.0053283, -0.0088595, -0.0053283, -0.0032650, 0.0032638
3: 0.0035633, 0.0041662, 0.0035633, 0.0041662, -0.0006029, 0.0006029
4: 0.0026282, 0.0054191, 0.0026282, 0.0054191, -0.0027909, 0.0027909
5: 0.0050201, 0.0064446, 0.0050201, 0.0064446, -0.0014245, 0.0014245
6: -0.0021963, -0.0008150, -0.0021963, -0.0008150, -0.0013813, 0.0013813
7: -0.0082640, -0.0074494, -0.0082640, -0.0074494, -0.0008146, 0.0008146
8: 0.0052524, 0.0098919, 0.0052524, 0.0098919, -0.0045680, 0.0045683
9: -0.0036845, -0.0030730, -0.0036845, -0.0030730, -0.0006115, 0.0006115

### Unstable ReLU Count (Linear/Conv2D Layers)
- layer_idx=0, type=LayerType.Linear, total=256, inp1_unstable=10, inp2_unstable=10, delta_unstable=252
- layer_idx=2, type=LayerType.Linear, total=256, inp1_unstable=8, inp2_unstable=8, delta_unstable=256
- layer_idx=4, type=LayerType.Linear, total=256, inp1_unstable=0, inp2_unstable=0, delta_unstable=256
- layer_idx=6, type=LayerType.Linear, total=256, inp1_unstable=2, inp2_unstable=2, delta_unstable=256
- layer_idx=8, type=LayerType.Linear, total=10, inp1_unstable=0, inp2_unstable=0, delta_unstable=10

Time for backsubstitution: 1.61 seconds

### RS candidates at layer 1
type: RSZ, layer: 1, pos: 136
type: RSZ, layer: 1, pos: 176
type: RSZ, layer: 1, pos: 198
type: RSZ, layer: 1, pos: 199
type: RSZ, layer: 1, pos: 204
type: RSZ, layer: 1, pos: 250

Time for candidate selection: 0.17 seconds

### Candidate
type: RSZ, layer: 1, pos: 136

### Relational analysis RSZ of RS_RSZ1_RSZ1_RSZ1_RSZ2_RSZ1

#### Relational analysis RSZ result of RS_RSZ1_RSZ1_RSZ1_RSZ2_RSZ1
Status: Status.UNKNOWN
Output dim: 1, lower bound: -0.0008426, upper bound: 0.0008426
time: 0.64 seconds

### Relational analysis RSZ of RS_RSZ1_RSZ1_RSZ1_RSZ2_RSZ2

#### Relational analysis RSZ result of RS_RSZ1_RSZ1_RSZ1_RSZ2_RSZ2
Status: Status.UNKNOWN
Output dim: 1, lower bound: -0.0008426, upper bound: 0.0008426
time: 0.75 seconds

## BFS RS instance: RS_RSZ1_RSZ1_RSZ2_RSZ1

### Backsubstitution after applying RS history:
0: 0.0006996, 0.0011177, 0.0006996, 0.0011177, -0.0004181, 0.0004181
1: 0.9934013, 0.9943976, 0.9934013, 0.9943976, -0.0009964, 0.0009964
2: -0.0088595, -0.0053283, -0.0088595, -0.0053283, -0.0032657, 0.0032624
3: 0.0035633, 0.0041662, 0.0035633, 0.0041662, -0.0006029, 0.0006029
4: 0.0026282, 0.0054191, 0.0026282, 0.0054191, -0.0027909, 0.0027909
5: 0.0050201, 0.0064446, 0.0050201, 0.0064446, -0.0014245, 0.0014245
6: -0.0021963, -0.0008150, -0.0021963, -0.0008150, -0.0013813, 0.0013813
7: -0.0082640, -0.0074494, -0.0082640, -0.0074494, -0.0008146, 0.0008146
8: 0.0052524, 0.0098919, 0.0052524, 0.0098919, -0.0045676, 0.0045685
9: -0.0036845, -0.0030730, -0.0036845, -0.0030730, -0.0006115, 0.0006115

### Unstable ReLU Count (Linear/Conv2D Layers)
- layer_idx=0, type=LayerType.Linear, total=256, inp1_unstable=10, inp2_unstable=10, delta_unstable=252
- layer_idx=2, type=LayerType.Linear, total=256, inp1_unstable=8, inp2_unstable=8, delta_unstable=256
- layer_idx=4, type=LayerType.Linear, total=256, inp1_unstable=0, inp2_unstable=0, delta_unstable=256
- layer_idx=6, type=LayerType.Linear, total=256, inp1_unstable=2, inp2_unstable=2, delta_unstable=256
- layer_idx=8, type=LayerType.Linear, total=10, inp1_unstable=0, inp2_unstable=0, delta_unstable=10

Time for backsubstitution: 1.59 seconds

### RS candidates at layer 1
type: RSZ, layer: 1, pos: 136
type: RSZ, layer: 1, pos: 176
type: RSZ, layer: 1, pos: 198
type: RSZ, layer: 1, pos: 199
type: RSZ, layer: 1, pos: 204
type: RSZ, layer: 1, pos: 250

Time for candidate selection: 0.17 seconds

### Candidate
type: RSZ, layer: 1, pos: 136

### Relational analysis RSZ of RS_RSZ1_RSZ1_RSZ2_RSZ1_RSZ1

#### Relational analysis RSZ result of RS_RSZ1_RSZ1_RSZ2_RSZ1_RSZ1
Status: Status.UNKNOWN
Output dim: 1, lower bound: -0.0008426, upper bound: 0.0008426
time: 0.72 seconds

### Relational analysis RSZ of RS_RSZ1_RSZ1_RSZ2_RSZ1_RSZ2

#### Relational analysis RSZ result of RS_RSZ1_RSZ1_RSZ2_RSZ1_RSZ2
Status: Status.UNKNOWN
Output dim: 1, lower bound: -0.0008426, upper bound: 0.0008426
time: 0.63 seconds

## BFS RS instance: RS_RSZ1_RSZ1_RSZ2_RSZ2

### Backsubstitution after applying RS history:
0: 0.0006996, 0.0011177, 0.0006996, 0.0011177, -0.0004181, 0.0004181
1: 0.9934013, 0.9943976, 0.9934013, 0.9943976, -0.0009964, 0.0009964
2: -0.0088595, -0.0053283, -0.0088595, -0.0053283, -0.0032645, 0.0032634
3: 0.0035633, 0.0041662, 0.0035633, 0.0041662, -0.0006029, 0.0006029
4: 0.0026282, 0.0054191, 0.0026282, 0.0054191, -0.0027909, 0.0027909
5: 0.0050201, 0.0064446, 0.0050201, 0.0064446, -0.0014245, 0.0014245
6: -0.0021963, -0.0008150, -0.0021963, -0.0008150, -0.0013813, 0.0013813
7: -0.0082640, -0.0074494, -0.0082640, -0.0074494, -0.0008146, 0.0008146
8: 0.0052524, 0.0098919, 0.0052524, 0.0098919, -0.0045679, 0.0045682
9: -0.0036845, -0.0030730, -0.0036845, -0.0030730, -0.0006115, 0.0006115

### Unstable ReLU Count (Linear/Conv2D Layers)
- layer_idx=0, type=LayerType.Linear, total=256, inp1_unstable=10, inp2_unstable=10, delta_unstable=252
- layer_idx=2, type=LayerType.Linear, total=256, inp1_unstable=8, inp2_unstable=8, delta_unstable=256
- layer_idx=4, type=LayerType.Linear, total=256, inp1_unstable=0, inp2_unstable=0, delta_unstable=256
- layer_idx=6, type=LayerType.Linear, total=256, inp1_unstable=2, inp2_unstable=2, delta_unstable=256
- layer_idx=8, type=LayerType.Linear, total=10, inp1_unstable=0, inp2_unstable=0, delta_unstable=10

Time for backsubstitution: 1.68 seconds

### RS candidates at layer 1
type: RSZ, layer: 1, pos: 136
type: RSZ, layer: 1, pos: 176
type: RSZ, layer: 1, pos: 198
type: RSZ, layer: 1, pos: 199
type: RSZ, layer: 1, pos: 204
type: RSZ, layer: 1, pos: 250

Time for candidate selection: 0.17 seconds

### Candidate
type: RSZ, layer: 1, pos: 136

### Relational analysis RSZ of RS_RSZ1_RSZ1_RSZ2_RSZ2_RSZ1

#### Relational analysis RSZ result of RS_RSZ1_RSZ1_RSZ2_RSZ2_RSZ1
Status: Status.UNKNOWN
Output dim: 1, lower bound: -0.0008426, upper bound: 0.0008426
time: 0.67 seconds

### Relational analysis RSZ of RS_RSZ1_RSZ1_RSZ2_RSZ2_RSZ2

#### Relational analysis RSZ result of RS_RSZ1_RSZ1_RSZ2_RSZ2_RSZ2
Status: Status.UNKNOWN
Output dim: 1, lower bound: -0.0008426, upper bound: 0.0008426
time: 0.67 seconds

## BFS RS instance: RS_RSZ1_RSZ2_RSZ1_RSZ1

### Backsubstitution after applying RS history:
0: 0.0006996, 0.0011177, 0.0006996, 0.0011177, -0.0004181, 0.0004181
1: 0.9934013, 0.9943976, 0.9934013, 0.9943976, -0.0009964, 0.0009964
2: -0.0088595, -0.0053283, -0.0088595, -0.0053283, -0.0032634, 0.0032649
3: 0.0035633, 0.0041662, 0.0035633, 0.0041662, -0.0006029, 0.0006029
4: 0.0026282, 0.0054191, 0.0026282, 0.0054191, -0.0027909, 0.0027909
5: 0.0050201, 0.0064446, 0.0050201, 0.0064446, -0.0014245, 0.0014245
6: -0.0021963, -0.0008150, -0.0021963, -0.0008150, -0.0013813, 0.0013813
7: -0.0082640, -0.0074494, -0.0082640, -0.0074494, -0.0008146, 0.0008146
8: 0.0052524, 0.0098919, 0.0052524, 0.0098919, -0.0045683, 0.0045679
9: -0.0036845, -0.0030730, -0.0036845, -0.0030730, -0.0006115, 0.0006115

### Unstable ReLU Count (Linear/Conv2D Layers)
- layer_idx=0, type=LayerType.Linear, total=256, inp1_unstable=10, inp2_unstable=10, delta_unstable=252
- layer_idx=2, type=LayerType.Linear, total=256, inp1_unstable=8, inp2_unstable=8, delta_unstable=256
- layer_idx=4, type=LayerType.Linear, total=256, inp1_unstable=0, inp2_unstable=0, delta_unstable=256
- layer_idx=6, type=LayerType.Linear, total=256, inp1_unstable=2, inp2_unstable=2, delta_unstable=256
- layer_idx=8, type=LayerType.Linear, total=10, inp1_unstable=0, inp2_unstable=0, delta_unstable=10

Time for backsubstitution: 1.59 seconds

### RS candidates at layer 1
type: RSZ, layer: 1, pos: 136
type: RSZ, layer: 1, pos: 176
type: RSZ, layer: 1, pos: 198
type: RSZ, layer: 1, pos: 199
type: RSZ, layer: 1, pos: 204
type: RSZ, layer: 1, pos: 250

Time for candidate selection: 0.17 seconds

### Candidate
type: RSZ, layer: 1, pos: 136

### Relational analysis RSZ of RS_RSZ1_RSZ2_RSZ1_RSZ1_RSZ1

#### Relational analysis RSZ result of RS_RSZ1_RSZ2_RSZ1_RSZ1_RSZ1
Status: Status.UNKNOWN
Output dim: 1, lower bound: -0.0008426, upper bound: 0.0008426
time: 0.67 seconds

### Relational analysis RSZ of RS_RSZ1_RSZ2_RSZ1_RSZ1_RSZ2

#### Relational analysis RSZ result of RS_RSZ1_RSZ2_RSZ1_RSZ1_RSZ2
Status: Status.UNKNOWN
Output dim: 1, lower bound: -0.0008426, upper bound: 0.0008426
time: 0.64 seconds

## BFS RS instance: RS_RSZ1_RSZ2_RSZ1_RSZ2

### Backsubstitution after applying RS history:
0: 0.0006996, 0.0011177, 0.0006996, 0.0011177, -0.0004181, 0.0004181
1: 0.9934013, 0.9943976, 0.9934013, 0.9943976, -0.0009964, 0.0009964
2: -0.0088595, -0.0053283, -0.0088595, -0.0053283, -0.0032620, 0.0032663
3: 0.0035633, 0.0041662, 0.0035633, 0.0041662, -0.0006029, 0.0006029
4: 0.0026282, 0.0054191, 0.0026282, 0.0054191, -0.0027909, 0.0027909
5: 0.0050201, 0.0064446, 0.0050201, 0.0064446, -0.0014245, 0.0014245
6: -0.0021963, -0.0008150, -0.0021963, -0.0008150, -0.0013813, 0.0013813
7: -0.0082640, -0.0074494, -0.0082640, -0.0074494, -0.0008146, 0.0008146
8: 0.0052524, 0.0098919, 0.0052524, 0.0098919, -0.0045687, 0.0045675
9: -0.0036845, -0.0030730, -0.0036845, -0.0030730, -0.0006115, 0.0006115

### Unstable ReLU Count (Linear/Conv2D Layers)
- layer_idx=0, type=LayerType.Linear, total=256, inp1_unstable=10, inp2_unstable=10, delta_unstable=252
- layer_idx=2, type=LayerType.Linear, total=256, inp1_unstable=8, inp2_unstable=8, delta_unstable=256
- layer_idx=4, type=LayerType.Linear, total=256, inp1_unstable=0, inp2_unstable=0, delta_unstable=256
- layer_idx=6, type=LayerType.Linear, total=256, inp1_unstable=2, inp2_unstable=2, delta_unstable=256
- layer_idx=8, type=LayerType.Linear, total=10, inp1_unstable=0, inp2_unstable=0, delta_unstable=10

Time for backsubstitution: 1.62 seconds

### RS candidates at layer 1
type: RSZ, layer: 1, pos: 136
type: RSZ, layer: 1, pos: 176
type: RSZ, layer: 1, pos: 198
type: RSZ, layer: 1, pos: 199
type: RSZ, layer: 1, pos: 204
type: RSZ, layer: 1, pos: 250

Time for candidate selection: 0.16 seconds

### Candidate
type: RSZ, layer: 1, pos: 136

### Relational analysis RSZ of RS_RSZ1_RSZ2_RSZ1_RSZ2_RSZ1

#### Relational analysis RSZ result of RS_RSZ1_RSZ2_RSZ1_RSZ2_RSZ1
Status: Status.UNKNOWN
Output dim: 1, lower bound: -0.0008426, upper bound: 0.0008426
time: 0.65 seconds

### Relational analysis RSZ of RS_RSZ1_RSZ2_RSZ1_RSZ2_RSZ2

#### Relational analysis RSZ result of RS_RSZ1_RSZ2_RSZ1_RSZ2_RSZ2
Status: Status.UNKNOWN
Output dim: 1, lower bound: -0.0008426, upper bound: 0.0008426
time: 0.66 seconds

## BFS RS instance: RS_RSZ1_RSZ2_RSZ2_RSZ1

### Backsubstitution after applying RS history:
0: 0.0006996, 0.0011177, 0.0006996, 0.0011177, -0.0004181, 0.0004181
1: 0.9934013, 0.9943976, 0.9934013, 0.9943976, -0.0009964, 0.0009964
2: -0.0088595, -0.0053283, -0.0088595, -0.0053283, -0.0032633, 0.0032654
3: 0.0035633, 0.0041662, 0.0035633, 0.0041662, -0.0006029, 0.0006029
4: 0.0026282, 0.0054191, 0.0026282, 0.0054191, -0.0027909, 0.0027909
5: 0.0050201, 0.0064446, 0.0050201, 0.0064446, -0.0014245, 0.0014245
6: -0.0021963, -0.0008150, -0.0021963, -0.0008150, -0.0013813, 0.0013813
7: -0.0082640, -0.0074494, -0.0082640, -0.0074494, -0.0008146, 0.0008146
8: 0.0052524, 0.0098919, 0.0052524, 0.0098919, -0.0045684, 0.0045679
9: -0.0036845, -0.0030730, -0.0036845, -0.0030730, -0.0006115, 0.0006115

### Unstable ReLU Count (Linear/Conv2D Layers)
- layer_idx=0, type=LayerType.Linear, total=256, inp1_unstable=10, inp2_unstable=10, delta_unstable=252
- layer_idx=2, type=LayerType.Linear, total=256, inp1_unstable=8, inp2_unstable=8, delta_unstable=256
- layer_idx=4, type=LayerType.Linear, total=256, inp1_unstable=0, inp2_unstable=0, delta_unstable=256
- layer_idx=6, type=LayerType.Linear, total=256, inp1_unstable=2, inp2_unstable=2, delta_unstable=256
- layer_idx=8, type=LayerType.Linear, total=10, inp1_unstable=0, inp2_unstable=0, delta_unstable=10

Time for backsubstitution: 1.64 seconds

### RS candidates at layer 1
type: RSZ, layer: 1, pos: 136
type: RSZ, layer: 1, pos: 176
type: RSZ, layer: 1, pos: 198
type: RSZ, layer: 1, pos: 199
type: RSZ, layer: 1, pos: 204
type: RSZ, layer: 1, pos: 250

Time for candidate selection: 0.19 seconds

### Candidate
type: RSZ, layer: 1, pos: 136

### Relational analysis RSZ of RS_RSZ1_RSZ2_RSZ2_RSZ1_RSZ1

#### Relational analysis RSZ result of RS_RSZ1_RSZ2_RSZ2_RSZ1_RSZ1
Status: Status.UNKNOWN
Output dim: 1, lower bound: -0.0008426, upper bound: 0.0008426
time: 0.67 seconds

### Relational analysis RSZ of RS_RSZ1_RSZ2_RSZ2_RSZ1_RSZ2

#### Relational analysis RSZ result of RS_RSZ1_RSZ2_RSZ2_RSZ1_RSZ2
Status: Status.UNKNOWN
Output dim: 1, lower bound: -0.0008426, upper bound: 0.0008426
time: 0.67 seconds

## BFS RS instance: RS_RSZ1_RSZ2_RSZ2_RSZ2

### Backsubstitution after applying RS history:
0: 0.0006996, 0.0011177, 0.0006996, 0.0011177, -0.0004181, 0.0004181
1: 0.9934013, 0.9943976, 0.9934013, 0.9943976, -0.0009964, 0.0009964
2: -0.0088595, -0.0053283, -0.0088595, -0.0053283, -0.0032620, 0.0032663
3: 0.0035633, 0.0041662, 0.0035633, 0.0041662, -0.0006029, 0.0006029
4: 0.0026282, 0.0054191, 0.0026282, 0.0054191, -0.0027909, 0.0027909
5: 0.0050201, 0.0064446, 0.0050201, 0.0064446, -0.0014245, 0.0014245
6: -0.0021963, -0.0008150, -0.0021963, -0.0008150, -0.0013813, 0.0013813
7: -0.0082640, -0.0074494, -0.0082640, -0.0074494, -0.0008146, 0.0008146
8: 0.0052524, 0.0098919, 0.0052524, 0.0098919, -0.0045687, 0.0045675
9: -0.0036845, -0.0030730, -0.0036845, -0.0030730, -0.0006115, 0.0006115

### Unstable ReLU Count (Linear/Conv2D Layers)
- layer_idx=0, type=LayerType.Linear, total=256, inp1_unstable=10, inp2_unstable=10, delta_unstable=252
- layer_idx=2, type=LayerType.Linear, total=256, inp1_unstable=8, inp2_unstable=8, delta_unstable=256
- layer_idx=4, type=LayerType.Linear, total=256, inp1_unstable=0, inp2_unstable=0, delta_unstable=256
- layer_idx=6, type=LayerType.Linear, total=256, inp1_unstable=2, inp2_unstable=2, delta_unstable=256
- layer_idx=8, type=LayerType.Linear, total=10, inp1_unstable=0, inp2_unstable=0, delta_unstable=10

Time for backsubstitution: 1.61 seconds

### RS candidates at layer 1
type: RSZ, layer: 1, pos: 136
type: RSZ, layer: 1, pos: 176
type: RSZ, layer: 1, pos: 198
type: RSZ, layer: 1, pos: 199
type: RSZ, layer: 1, pos: 204
type: RSZ, layer: 1, pos: 250

Time for candidate selection: 0.18 seconds

### Candidate
type: RSZ, layer: 1, pos: 136

### Relational analysis RSZ of RS_RSZ1_RSZ2_RSZ2_RSZ2_RSZ1

#### Relational analysis RSZ result of RS_RSZ1_RSZ2_RSZ2_RSZ2_RSZ1
Status: Status.UNKNOWN
Output dim: 1, lower bound: -0.0008426, upper bound: 0.0008426
time: 0.64 seconds

### Relational analysis RSZ of RS_RSZ1_RSZ2_RSZ2_RSZ2_RSZ2

#### Relational analysis RSZ result of RS_RSZ1_RSZ2_RSZ2_RSZ2_RSZ2
Status: Status.UNKNOWN
Output dim: 1, lower bound: -0.0008426, upper bound: 0.0008426
time: 0.70 seconds

## BFS RS instance: RS_RSZ2_RSZ1_RSZ1_RSZ1

### Backsubstitution after applying RS history:
0: 0.0006996, 0.0011177, 0.0006996, 0.0011177, -0.0004181, 0.0004181
1: 0.9934013, 0.9943976, 0.9934013, 0.9943976, -0.0009964, 0.0009964
2: -0.0088595, -0.0053283, -0.0088595, -0.0053283, -0.0032663, 0.0032620
3: 0.0035633, 0.0041662, 0.0035633, 0.0041662, -0.0006029, 0.0006029
4: 0.0026282, 0.0054191, 0.0026282, 0.0054191, -0.0027909, 0.0027909
5: 0.0050201, 0.0064446, 0.0050201, 0.0064446, -0.0014245, 0.0014245
6: -0.0021963, -0.0008150, -0.0021963, -0.0008150, -0.0013813, 0.0013813
7: -0.0082640, -0.0074494, -0.0082640, -0.0074494, -0.0008146, 0.0008146
8: 0.0052524, 0.0098919, 0.0052524, 0.0098919, -0.0045675, 0.0045687
9: -0.0036845, -0.0030730, -0.0036845, -0.0030730, -0.0006115, 0.0006115

### Unstable ReLU Count (Linear/Conv2D Layers)
- layer_idx=0, type=LayerType.Linear, total=256, inp1_unstable=10, inp2_unstable=10, delta_unstable=252
- layer_idx=2, type=LayerType.Linear, total=256, inp1_unstable=8, inp2_unstable=8, delta_unstable=256
- layer_idx=4, type=LayerType.Linear, total=256, inp1_unstable=0, inp2_unstable=0, delta_unstable=256
- layer_idx=6, type=LayerType.Linear, total=256, inp1_unstable=2, inp2_unstable=2, delta_unstable=256
- layer_idx=8, type=LayerType.Linear, total=10, inp1_unstable=0, inp2_unstable=0, delta_unstable=10

Time for backsubstitution: 1.73 seconds

### RS candidates at layer 1
type: RSZ, layer: 1, pos: 136
type: RSZ, layer: 1, pos: 176
type: RSZ, layer: 1, pos: 198
type: RSZ, layer: 1, pos: 199
type: RSZ, layer: 1, pos: 204
type: RSZ, layer: 1, pos: 250

Time for candidate selection: 0.18 seconds

### Candidate
type: RSZ, layer: 1, pos: 136

### Relational analysis RSZ of RS_RSZ2_RSZ1_RSZ1_RSZ1_RSZ1

#### Relational analysis RSZ result of RS_RSZ2_RSZ1_RSZ1_RSZ1_RSZ1
Status: Status.UNKNOWN
Output dim: 1, lower bound: -0.0008426, upper bound: 0.0008426
time: 0.65 seconds

### Relational analysis RSZ of RS_RSZ2_RSZ1_RSZ1_RSZ1_RSZ2

#### Relational analysis RSZ result of RS_RSZ2_RSZ1_RSZ1_RSZ1_RSZ2
Status: Status.UNKNOWN
Output dim: 1, lower bound: -0.0008426, upper bound: 0.0008426
time: 0.69 seconds

## BFS RS instance: RS_RSZ2_RSZ1_RSZ1_RSZ2

### Backsubstitution after applying RS history:
0: 0.0006996, 0.0011177, 0.0006996, 0.0011177, -0.0004181, 0.0004181
1: 0.9934013, 0.9943976, 0.9934013, 0.9943976, -0.0009964, 0.0009964
2: -0.0088595, -0.0053283, -0.0088595, -0.0053283, -0.0032654, 0.0032633
3: 0.0035633, 0.0041662, 0.0035633, 0.0041662, -0.0006029, 0.0006029
4: 0.0026282, 0.0054191, 0.0026282, 0.0054191, -0.0027909, 0.0027909
5: 0.0050201, 0.0064446, 0.0050201, 0.0064446, -0.0014245, 0.0014245
6: -0.0021963, -0.0008150, -0.0021963, -0.0008150, -0.0013813, 0.0013813
7: -0.0082640, -0.0074494, -0.0082640, -0.0074494, -0.0008146, 0.0008146
8: 0.0052524, 0.0098919, 0.0052524, 0.0098919, -0.0045679, 0.0045684
9: -0.0036845, -0.0030730, -0.0036845, -0.0030730, -0.0006115, 0.0006115

### Unstable ReLU Count (Linear/Conv2D Layers)
- layer_idx=0, type=LayerType.Linear, total=256, inp1_unstable=10, inp2_unstable=10, delta_unstable=252
- layer_idx=2, type=LayerType.Linear, total=256, inp1_unstable=8, inp2_unstable=8, delta_unstable=256
- layer_idx=4, type=LayerType.Linear, total=256, inp1_unstable=0, inp2_unstable=0, delta_unstable=256
- layer_idx=6, type=LayerType.Linear, total=256, inp1_unstable=2, inp2_unstable=2, delta_unstable=256
- layer_idx=8, type=LayerType.Linear, total=10, inp1_unstable=0, inp2_unstable=0, delta_unstable=10

Time for backsubstitution: 1.67 seconds

### RS candidates at layer 1
type: RSZ, layer: 1, pos: 136
type: RSZ, layer: 1, pos: 176
type: RSZ, layer: 1, pos: 198
type: RSZ, layer: 1, pos: 199
type: RSZ, layer: 1, pos: 204
type: RSZ, layer: 1, pos: 250

Time for candidate selection: 0.17 seconds

### Candidate
type: RSZ, layer: 1, pos: 136

### Relational analysis RSZ of RS_RSZ2_RSZ1_RSZ1_RSZ2_RSZ1

#### Relational analysis RSZ result of RS_RSZ2_RSZ1_RSZ1_RSZ2_RSZ1
Status: Status.UNKNOWN
Output dim: 1, lower bound: -0.0008426, upper bound: 0.0008426
time: 0.81 seconds

### Relational analysis RSZ of RS_RSZ2_RSZ1_RSZ1_RSZ2_RSZ2

#### Relational analysis RSZ result of RS_RSZ2_RSZ1_RSZ1_RSZ2_RSZ2
Status: Status.UNKNOWN
Output dim: 1, lower bound: -0.0008426, upper bound: 0.0008426
time: 0.69 seconds

## BFS RS instance: RS_RSZ2_RSZ1_RSZ2_RSZ1

### Backsubstitution after applying RS history:
0: 0.0006996, 0.0011177, 0.0006996, 0.0011177, -0.0004181, 0.0004181
1: 0.9934013, 0.9943976, 0.9934013, 0.9943976, -0.0009964, 0.0009964
2: -0.0088595, -0.0053283, -0.0088595, -0.0053283, -0.0032663, 0.0032620
3: 0.0035633, 0.0041662, 0.0035633, 0.0041662, -0.0006029, 0.0006029
4: 0.0026282, 0.0054191, 0.0026282, 0.0054191, -0.0027909, 0.0027909
5: 0.0050201, 0.0064446, 0.0050201, 0.0064446, -0.0014245, 0.0014245
6: -0.0021963, -0.0008150, -0.0021963, -0.0008150, -0.0013813, 0.0013813
7: -0.0082640, -0.0074494, -0.0082640, -0.0074494, -0.0008146, 0.0008146
8: 0.0052524, 0.0098919, 0.0052524, 0.0098919, -0.0045675, 0.0045687
9: -0.0036845, -0.0030730, -0.0036845, -0.0030730, -0.0006115, 0.0006115

### Unstable ReLU Count (Linear/Conv2D Layers)
- layer_idx=0, type=LayerType.Linear, total=256, inp1_unstable=10, inp2_unstable=10, delta_unstable=252
- layer_idx=2, type=LayerType.Linear, total=256, inp1_unstable=8, inp2_unstable=8, delta_unstable=256
- layer_idx=4, type=LayerType.Linear, total=256, inp1_unstable=0, inp2_unstable=0, delta_unstable=256
- layer_idx=6, type=LayerType.Linear, total=256, inp1_unstable=2, inp2_unstable=2, delta_unstable=256
- layer_idx=8, type=LayerType.Linear, total=10, inp1_unstable=0, inp2_unstable=0, delta_unstable=10

Time for backsubstitution: 1.58 seconds

### RS candidates at layer 1
type: RSZ, layer: 1, pos: 136
type: RSZ, layer: 1, pos: 176
type: RSZ, layer: 1, pos: 198
type: RSZ, layer: 1, pos: 199
type: RSZ, layer: 1, pos: 204
type: RSZ, layer: 1, pos: 250

Time for candidate selection: 0.17 seconds

### Candidate
type: RSZ, layer: 1, pos: 136

### Relational analysis RSZ of RS_RSZ2_RSZ1_RSZ2_RSZ1_RSZ1

#### Relational analysis RSZ result of RS_RSZ2_RSZ1_RSZ2_RSZ1_RSZ1
Status: Status.UNKNOWN
Output dim: 1, lower bound: -0.0008426, upper bound: 0.0008426
time: 0.89 seconds

### Relational analysis RSZ of RS_RSZ2_RSZ1_RSZ2_RSZ1_RSZ2

#### Relational analysis RSZ result of RS_RSZ2_RSZ1_RSZ2_RSZ1_RSZ2
Status: Status.UNKNOWN
Output dim: 1, lower bound: -0.0008426, upper bound: 0.0008426
time: 0.63 seconds

## BFS RS instance: RS_RSZ2_RSZ1_RSZ2_RSZ2

### Backsubstitution after applying RS history:
0: 0.0006996, 0.0011177, 0.0006996, 0.0011177, -0.0004181, 0.0004181
1: 0.9934013, 0.9943976, 0.9934013, 0.9943976, -0.0009964, 0.0009964
2: -0.0088595, -0.0053283, -0.0088595, -0.0053283, -0.0032649, 0.0032634
3: 0.0035633, 0.0041662, 0.0035633, 0.0041662, -0.0006029, 0.0006029
4: 0.0026282, 0.0054191, 0.0026282, 0.0054191, -0.0027909, 0.0027909
5: 0.0050201, 0.0064446, 0.0050201, 0.0064446, -0.0014245, 0.0014245
6: -0.0021963, -0.0008150, -0.0021963, -0.0008150, -0.0013813, 0.0013813
7: -0.0082640, -0.0074494, -0.0082640, -0.0074494, -0.0008146, 0.0008146
8: 0.0052524, 0.0098919, 0.0052524, 0.0098919, -0.0045679, 0.0045683
9: -0.0036845, -0.0030730, -0.0036845, -0.0030730, -0.0006115, 0.0006115

### Unstable ReLU Count (Linear/Conv2D Layers)
- layer_idx=0, type=LayerType.Linear, total=256, inp1_unstable=10, inp2_unstable=10, delta_unstable=252
- layer_idx=2, type=LayerType.Linear, total=256, inp1_unstable=8, inp2_unstable=8, delta_unstable=256
- layer_idx=4, type=LayerType.Linear, total=256, inp1_unstable=0, inp2_unstable=0, delta_unstable=256
- layer_idx=6, type=LayerType.Linear, total=256, inp1_unstable=2, inp2_unstable=2, delta_unstable=256
- layer_idx=8, type=LayerType.Linear, total=10, inp1_unstable=0, inp2_unstable=0, delta_unstable=10

Time for backsubstitution: 1.62 seconds

### RS candidates at layer 1
type: RSZ, layer: 1, pos: 136
type: RSZ, layer: 1, pos: 176
type: RSZ, layer: 1, pos: 198
type: RSZ, layer: 1, pos: 199
type: RSZ, layer: 1, pos: 204
type: RSZ, layer: 1, pos: 250

Time for candidate selection: 0.18 seconds

### Candidate
type: RSZ, layer: 1, pos: 136

### Relational analysis RSZ of RS_RSZ2_RSZ1_RSZ2_RSZ2_RSZ1

#### Relational analysis RSZ result of RS_RSZ2_RSZ1_RSZ2_RSZ2_RSZ1
Status: Status.UNKNOWN
Output dim: 1, lower bound: -0.0008426, upper bound: 0.0008426
time: 0.64 seconds

### Relational analysis RSZ of RS_RSZ2_RSZ1_RSZ2_RSZ2_RSZ2

#### Relational analysis RSZ result of RS_RSZ2_RSZ1_RSZ2_RSZ2_RSZ2
Status: Status.UNKNOWN
Output dim: 1, lower bound: -0.0008426, upper bound: 0.0008426
time: 0.71 seconds

## BFS RS instance: RS_RSZ2_RSZ2_RSZ1_RSZ1

### Backsubstitution after applying RS history:
0: 0.0006996, 0.0011177, 0.0006996, 0.0011177, -0.0004181, 0.0004181
1: 0.9934013, 0.9943976, 0.9934013, 0.9943976, -0.0009964, 0.0009964
2: -0.0088595, -0.0053283, -0.0088595, -0.0053283, -0.0032634, 0.0032645
3: 0.0035633, 0.0041662, 0.0035633, 0.0041662, -0.0006029, 0.0006029
4: 0.0026282, 0.0054191, 0.0026282, 0.0054191, -0.0027909, 0.0027909
5: 0.0050201, 0.0064446, 0.0050201, 0.0064446, -0.0014245, 0.0014245
6: -0.0021963, -0.0008150, -0.0021963, -0.0008150, -0.0013813, 0.0013813
7: -0.0082640, -0.0074494, -0.0082640, -0.0074494, -0.0008146, 0.0008146
8: 0.0052524, 0.0098919, 0.0052524, 0.0098919, -0.0045682, 0.0045679
9: -0.0036845, -0.0030730, -0.0036845, -0.0030730, -0.0006115, 0.0006115

### Unstable ReLU Count (Linear/Conv2D Layers)
- layer_idx=0, type=LayerType.Linear, total=256, inp1_unstable=10, inp2_unstable=10, delta_unstable=252
- layer_idx=2, type=LayerType.Linear, total=256, inp1_unstable=8, inp2_unstable=8, delta_unstable=256
- layer_idx=4, type=LayerType.Linear, total=256, inp1_unstable=0, inp2_unstable=0, delta_unstable=256
- layer_idx=6, type=LayerType.Linear, total=256, inp1_unstable=2, inp2_unstable=2, delta_unstable=256
- layer_idx=8, type=LayerType.Linear, total=10, inp1_unstable=0, inp2_unstable=0, delta_unstable=10

Time for backsubstitution: 1.63 seconds

### RS candidates at layer 1
type: RSZ, layer: 1, pos: 136
type: RSZ, layer: 1, pos: 176
type: RSZ, layer: 1, pos: 198
type: RSZ, layer: 1, pos: 199
type: RSZ, layer: 1, pos: 204
type: RSZ, layer: 1, pos: 250

Time for candidate selection: 0.18 seconds

### Candidate
type: RSZ, layer: 1, pos: 136

### Relational analysis RSZ of RS_RSZ2_RSZ2_RSZ1_RSZ1_RSZ1

#### Relational analysis RSZ result of RS_RSZ2_RSZ2_RSZ1_RSZ1_RSZ1
Status: Status.UNKNOWN
Output dim: 1, lower bound: -0.0008426, upper bound: 0.0008426
time: 0.76 seconds

### Relational analysis RSZ of RS_RSZ2_RSZ2_RSZ1_RSZ1_RSZ2

#### Relational analysis RSZ result of RS_RSZ2_RSZ2_RSZ1_RSZ1_RSZ2
Status: Status.UNKNOWN
Output dim: 1, lower bound: -0.0008426, upper bound: 0.0008426
time: 0.68 seconds

## BFS RS instance: RS_RSZ2_RSZ2_RSZ1_RSZ2

### Backsubstitution after applying RS history:
0: 0.0006996, 0.0011177, 0.0006996, 0.0011177, -0.0004181, 0.0004181
1: 0.9934013, 0.9943976, 0.9934013, 0.9943976, -0.0009964, 0.0009964
2: -0.0088595, -0.0053283, -0.0088595, -0.0053283, -0.0032624, 0.0032657
3: 0.0035633, 0.0041662, 0.0035633, 0.0041662, -0.0006029, 0.0006029
4: 0.0026282, 0.0054191, 0.0026282, 0.0054191, -0.0027909, 0.0027909
5: 0.0050201, 0.0064446, 0.0050201, 0.0064446, -0.0014245, 0.0014245
6: -0.0021963, -0.0008150, -0.0021963, -0.0008150, -0.0013813, 0.0013813
7: -0.0082640, -0.0074494, -0.0082640, -0.0074494, -0.0008146, 0.0008146
8: 0.0052524, 0.0098919, 0.0052524, 0.0098919, -0.0045685, 0.0045676
9: -0.0036845, -0.0030730, -0.0036845, -0.0030730, -0.0006115, 0.0006115

### Unstable ReLU Count (Linear/Conv2D Layers)
- layer_idx=0, type=LayerType.Linear, total=256, inp1_unstable=10, inp2_unstable=10, delta_unstable=252
- layer_idx=2, type=LayerType.Linear, total=256, inp1_unstable=8, inp2_unstable=8, delta_unstable=256
- layer_idx=4, type=LayerType.Linear, total=256, inp1_unstable=0, inp2_unstable=0, delta_unstable=256
- layer_idx=6, type=LayerType.Linear, total=256, inp1_unstable=2, inp2_unstable=2, delta_unstable=256
- layer_idx=8, type=LayerType.Linear, total=10, inp1_unstable=0, inp2_unstable=0, delta_unstable=10

Time for backsubstitution: 1.60 seconds

### RS candidates at layer 1
type: RSZ, layer: 1, pos: 136
type: RSZ, layer: 1, pos: 176
type: RSZ, layer: 1, pos: 198
type: RSZ, layer: 1, pos: 199
type: RSZ, layer: 1, pos: 204
type: RSZ, layer: 1, pos: 250

Time for candidate selection: 0.16 seconds

### Candidate
type: RSZ, layer: 1, pos: 136

### Relational analysis RSZ of RS_RSZ2_RSZ2_RSZ1_RSZ2_RSZ1

#### Relational analysis RSZ result of RS_RSZ2_RSZ2_RSZ1_RSZ2_RSZ1
Status: Status.UNKNOWN
Output dim: 1, lower bound: -0.0008426, upper bound: 0.0008426
time: 0.70 seconds

### Relational analysis RSZ of RS_RSZ2_RSZ2_RSZ1_RSZ2_RSZ2

#### Relational analysis RSZ result of RS_RSZ2_RSZ2_RSZ1_RSZ2_RSZ2
Status: Status.UNKNOWN
Output dim: 1, lower bound: -0.0008426, upper bound: 0.0008426
time: 0.67 seconds

## BFS RS instance: RS_RSZ2_RSZ2_RSZ2_RSZ1

### Backsubstitution after applying RS history:
0: 0.0006996, 0.0011177, 0.0006996, 0.0011177, -0.0004181, 0.0004181
1: 0.9934013, 0.9943976, 0.9934013, 0.9943976, -0.0009964, 0.0009964
2: -0.0088595, -0.0053283, -0.0088595, -0.0053283, -0.0032638, 0.0032650
3: 0.0035633, 0.0041662, 0.0035633, 0.0041662, -0.0006029, 0.0006029
4: 0.0026282, 0.0054191, 0.0026282, 0.0054191, -0.0027909, 0.0027909
5: 0.0050201, 0.0064446, 0.0050201, 0.0064446, -0.0014245, 0.0014245
6: -0.0021963, -0.0008150, -0.0021963, -0.0008150, -0.0013813, 0.0013813
7: -0.0082640, -0.0074494, -0.0082640, -0.0074494, -0.0008146, 0.0008146
8: 0.0052524, 0.0098919, 0.0052524, 0.0098919, -0.0045683, 0.0045680
9: -0.0036845, -0.0030730, -0.0036845, -0.0030730, -0.0006115, 0.0006115

### Unstable ReLU Count (Linear/Conv2D Layers)
- layer_idx=0, type=LayerType.Linear, total=256, inp1_unstable=10, inp2_unstable=10, delta_unstable=252
- layer_idx=2, type=LayerType.Linear, total=256, inp1_unstable=8, inp2_unstable=8, delta_unstable=256
- layer_idx=4, type=LayerType.Linear, total=256, inp1_unstable=0, inp2_unstable=0, delta_unstable=256
- layer_idx=6, type=LayerType.Linear, total=256, inp1_unstable=2, inp2_unstable=2, delta_unstable=256
- layer_idx=8, type=LayerType.Linear, total=10, inp1_unstable=0, inp2_unstable=0, delta_unstable=10

Time for backsubstitution: 1.82 seconds

### RS candidates at layer 1
type: RSZ, layer: 1, pos: 136
type: RSZ, layer: 1, pos: 176
type: RSZ, layer: 1, pos: 198
type: RSZ, layer: 1, pos: 199
type: RSZ, layer: 1, pos: 204
type: RSZ, layer: 1, pos: 250

Time for candidate selection: 0.18 seconds

### Candidate
type: RSZ, layer: 1, pos: 136

### Relational analysis RSZ of RS_RSZ2_RSZ2_RSZ2_RSZ1_RSZ1

#### Relational analysis RSZ result of RS_RSZ2_RSZ2_RSZ2_RSZ1_RSZ1
Status: Status.UNKNOWN
Output dim: 1, lower bound: -0.0008426, upper bound: 0.0008426
time: 0.64 seconds

### Relational analysis RSZ of RS_RSZ2_RSZ2_RSZ2_RSZ1_RSZ2

#### Relational analysis RSZ result of RS_RSZ2_RSZ2_RSZ2_RSZ1_RSZ2
Status: Status.UNKNOWN
Output dim: 1, lower bound: -0.0008426, upper bound: 0.0008426
time: 0.68 seconds

## BFS RS instance: RS_RSZ2_RSZ2_RSZ2_RSZ2

### Backsubstitution after applying RS history:
0: 0.0006996, 0.0011177, 0.0006996, 0.0011177, -0.0004181, 0.0004181
1: 0.9934013, 0.9943976, 0.9934013, 0.9943976, -0.0009964, 0.0009964
2: -0.0088595, -0.0053283, -0.0088595, -0.0053283, -0.0032625, 0.0032659
3: 0.0035633, 0.0041662, 0.0035633, 0.0041662, -0.0006029, 0.0006029
4: 0.0026282, 0.0054191, 0.0026282, 0.0054191, -0.0027909, 0.0027909
5: 0.0050201, 0.0064446, 0.0050201, 0.0064446, -0.0014245, 0.0014245
6: -0.0021963, -0.0008150, -0.0021963, -0.0008150, -0.0013813, 0.0013813
7: -0.0082640, -0.0074494, -0.0082640, -0.0074494, -0.0008146, 0.0008146
8: 0.0052524, 0.0098919, 0.0052524, 0.0098919, -0.0045685, 0.0045676
9: -0.0036845, -0.0030730, -0.0036845, -0.0030730, -0.0006115, 0.0006115

### Unstable ReLU Count (Linear/Conv2D Layers)
- layer_idx=0, type=LayerType.Linear, total=256, inp1_unstable=10, inp2_unstable=10, delta_unstable=252
- layer_idx=2, type=LayerType.Linear, total=256, inp1_unstable=8, inp2_unstable=8, delta_unstable=256
- layer_idx=4, type=LayerType.Linear, total=256, inp1_unstable=0, inp2_unstable=0, delta_unstable=256
- layer_idx=6, type=LayerType.Linear, total=256, inp1_unstable=2, inp2_unstable=2, delta_unstable=256
- layer_idx=8, type=LayerType.Linear, total=10, inp1_unstable=0, inp2_unstable=0, delta_unstable=10

Time for backsubstitution: 1.64 seconds

### RS candidates at layer 1
type: RSZ, layer: 1, pos: 136
type: RSZ, layer: 1, pos: 176
type: RSZ, layer: 1, pos: 198
type: RSZ, layer: 1, pos: 199
type: RSZ, layer: 1, pos: 204
type: RSZ, layer: 1, pos: 250

Time for candidate selection: 0.17 seconds

### Candidate
type: RSZ, layer: 1, pos: 136

### Relational analysis RSZ of RS_RSZ2_RSZ2_RSZ2_RSZ2_RSZ1

#### Relational analysis RSZ result of RS_RSZ2_RSZ2_RSZ2_RSZ2_RSZ1
Status: Status.UNKNOWN
Output dim: 1, lower bound: -0.0008426, upper bound: 0.0008426
time: 0.73 seconds

### Relational analysis RSZ of RS_RSZ2_RSZ2_RSZ2_RSZ2_RSZ2

#### Relational analysis RSZ result of RS_RSZ2_RSZ2_RSZ2_RSZ2_RSZ2
Status: Status.UNKNOWN
Output dim: 1, lower bound: -0.0008426, upper bound: 0.0008426
time: 0.65 seconds

## Summary of splitting (split count: 4)
- Time for RS candidates: 3.20 seconds
RS_RSZ1_RSZ1_RSZ1_RSZ1_RSZ1, status: Status.UNKNOWN, split count: 5, time: 3.20
Output dim: 1, lower bound: -0.0008426, upper bound: 0.0008426
RS_RSZ1_RSZ1_RSZ1_RSZ1_RSZ2, status: Status.UNKNOWN, split count: 5, time: 3.20
Output dim: 1, lower bound: -0.0008426, upper bound: 0.0008426
RS_RSZ1_RSZ1_RSZ1_RSZ2_RSZ1, status: Status.UNKNOWN, split count: 5, time: 3.20
Output dim: 1, lower bound: -0.0008426, upper bound: 0.0008426
RS_RSZ1_RSZ1_RSZ1_RSZ2_RSZ2, status: Status.UNKNOWN, split count: 5, time: 3.20
Output dim: 1, lower bound: -0.0008426, upper bound: 0.0008426
RS_RSZ1_RSZ1_RSZ2_RSZ1_RSZ1, status: Status.UNKNOWN, split count: 5, time: 3.20
Output dim: 1, lower bound: -0.0008426, upper bound: 0.0008426
RS_RSZ1_RSZ1_RSZ2_RSZ1_RSZ2, status: Status.UNKNOWN, split count: 5, time: 3.20
Output dim: 1, lower bound: -0.0008426, upper bound: 0.0008426
RS_RSZ1_RSZ1_RSZ2_RSZ2_RSZ1, status: Status.UNKNOWN, split count: 5, time: 3.20
Output dim: 1, lower bound: -0.0008426, upper bound: 0.0008426
RS_RSZ1_RSZ1_RSZ2_RSZ2_RSZ2, status: Status.UNKNOWN, split count: 5, time: 3.20
Output dim: 1, lower bound: -0.0008426, upper bound: 0.0008426
RS_RSZ1_RSZ2_RSZ1_RSZ1_RSZ1, status: Status.UNKNOWN, split count: 5, time: 3.20
Output dim: 1, lower bound: -0.0008426, upper bound: 0.0008426
RS_RSZ1_RSZ2_RSZ1_RSZ1_RSZ2, status: Status.UNKNOWN, split count: 5, time: 3.20
Output dim: 1, lower bound: -0.0008426, upper bound: 0.0008426
RS_RSZ1_RSZ2_RSZ1_RSZ2_RSZ1, status: Status.UNKNOWN, split count: 5, time: 3.20
Output dim: 1, lower bound: -0.0008426, upper bound: 0.0008426
RS_RSZ1_RSZ2_RSZ1_RSZ2_RSZ2, status: Status.UNKNOWN, split count: 5, time: 3.20
Output dim: 1, lower bound: -0.0008426, upper bound: 0.0008426
RS_RSZ1_RSZ2_RSZ2_RSZ1_RSZ1, status: Status.UNKNOWN, split count: 5, time: 3.20
Output dim: 1, lower bound: -0.0008426, upper bound: 0.0008426
RS_RSZ1_RSZ2_RSZ2_RSZ1_RSZ2, status: Status.UNKNOWN, split count: 5, time: 3.20
Output dim: 1, lower bound: -0.0008426, upper bound: 0.0008426
RS_RSZ1_RSZ2_RSZ2_RSZ2_RSZ1, status: Status.UNKNOWN, split count: 5, time: 3.20
Output dim: 1, lower bound: -0.0008426, upper bound: 0.0008426
RS_RSZ1_RSZ2_RSZ2_RSZ2_RSZ2, status: Status.UNKNOWN, split count: 5, time: 3.20
Output dim: 1, lower bound: -0.0008426, upper bound: 0.0008426
RS_RSZ2_RSZ1_RSZ1_RSZ1_RSZ1, status: Status.UNKNOWN, split count: 5, time: 3.20
Output dim: 1, lower bound: -0.0008426, upper bound: 0.0008426
RS_RSZ2_RSZ1_RSZ1_RSZ1_RSZ2, status: Status.UNKNOWN, split count: 5, time: 3.20
Output dim: 1, lower bound: -0.0008426, upper bound: 0.0008426
RS_RSZ2_RSZ1_RSZ1_RSZ2_RSZ1, status: Status.UNKNOWN, split count: 5, time: 3.20
Output dim: 1, lower bound: -0.0008426, upper bound: 0.0008426
RS_RSZ2_RSZ1_RSZ1_RSZ2_RSZ2, status: Status.UNKNOWN, split count: 5, time: 3.20
Output dim: 1, lower bound: -0.0008426, upper bound: 0.0008426
RS_RSZ2_RSZ1_RSZ2_RSZ1_RSZ1, status: Status.UNKNOWN, split count: 5, time: 3.20
Output dim: 1, lower bound: -0.0008426, upper bound: 0.0008426
RS_RSZ2_RSZ1_RSZ2_RSZ1_RSZ2, status: Status.UNKNOWN, split count: 5, time: 3.20
Output dim: 1, lower bound: -0.0008426, upper bound: 0.0008426
RS_RSZ2_RSZ1_RSZ2_RSZ2_RSZ1, status: Status.UNKNOWN, split count: 5, time: 3.20
Output dim: 1, lower bound: -0.0008426, upper bound: 0.0008426
RS_RSZ2_RSZ1_RSZ2_RSZ2_RSZ2, status: Status.UNKNOWN, split count: 5, time: 3.20
Output dim: 1, lower bound: -0.0008426, upper bound: 0.0008426
RS_RSZ2_RSZ2_RSZ1_RSZ1_RSZ1, status: Status.UNKNOWN, split count: 5, time: 3.20
Output dim: 1, lower bound: -0.0008426, upper bound: 0.0008426
RS_RSZ2_RSZ2_RSZ1_RSZ1_RSZ2, status: Status.UNKNOWN, split count: 5, time: 3.20
Output dim: 1, lower bound: -0.0008426, upper bound: 0.0008426
RS_RSZ2_RSZ2_RSZ1_RSZ2_RSZ1, status: Status.UNKNOWN, split count: 5, time: 3.20
Output dim: 1, lower bound: -0.0008426, upper bound: 0.0008426
RS_RSZ2_RSZ2_RSZ1_RSZ2_RSZ2, status: Status.UNKNOWN, split count: 5, time: 3.20
Output dim: 1, lower bound: -0.0008426, upper bound: 0.0008426
RS_RSZ2_RSZ2_RSZ2_RSZ1_RSZ1, status: Status.UNKNOWN, split count: 5, time: 3.20
Output dim: 1, lower bound: -0.0008426, upper bound: 0.0008426
RS_RSZ2_RSZ2_RSZ2_RSZ1_RSZ2, status: Status.UNKNOWN, split count: 5, time: 3.20
Output dim: 1, lower bound: -0.0008426, upper bound: 0.0008426
RS_RSZ2_RSZ2_RSZ2_RSZ2_RSZ1, status: Status.UNKNOWN, split count: 5, time: 3.20
Output dim: 1, lower bound: -0.0008426, upper bound: 0.0008426
RS_RSZ2_RSZ2_RSZ2_RSZ2_RSZ2, status: Status.UNKNOWN, split count: 5, time: 3.20
Output dim: 1, lower bound: -0.0008426, upper bound: 0.0008426

## BFS RS instance: RS_RSZ1_RSZ1_RSZ1_RSZ1_RSZ1

### Backsubstitution after applying RS history:
0: 0.0006996, 0.0011177, 0.0006996, 0.0011177, -0.0004181, 0.0004181
1: 0.9934013, 0.9943976, 0.9934013, 0.9943976, -0.0009964, 0.0009964
2: -0.0088595, -0.0053283, -0.0088595, -0.0053283, -0.0032543, 0.0032491
3: 0.0035633, 0.0041662, 0.0035633, 0.0041662, -0.0006029, 0.0006029
4: 0.0026282, 0.0054191, 0.0026282, 0.0054191, -0.0027909, 0.0027909
5: 0.0050201, 0.0064446, 0.0050201, 0.0064446, -0.0014245, 0.0014245
6: -0.0021963, -0.0008150, -0.0021963, -0.0008150, -0.0013813, 0.0013813
7: -0.0082640, -0.0074494, -0.0082640, -0.0074494, -0.0008146, 0.0008146
8: 0.0052524, 0.0098919, 0.0052524, 0.0098919, -0.0045642, 0.0045656
9: -0.0036845, -0.0030730, -0.0036845, -0.0030730, -0.0006115, 0.0006115

### Unstable ReLU Count (Linear/Conv2D Layers)
- layer_idx=0, type=LayerType.Linear, total=256, inp1_unstable=10, inp2_unstable=10, delta_unstable=251
- layer_idx=2, type=LayerType.Linear, total=256, inp1_unstable=8, inp2_unstable=8, delta_unstable=256
- layer_idx=4, type=LayerType.Linear, total=256, inp1_unstable=0, inp2_unstable=0, delta_unstable=256
- layer_idx=6, type=LayerType.Linear, total=256, inp1_unstable=2, inp2_unstable=2, delta_unstable=256
- layer_idx=8, type=LayerType.Linear, total=10, inp1_unstable=0, inp2_unstable=0, delta_unstable=10

Time for backsubstitution: 1.62 seconds

### RS candidates at layer 1
type: RSZ, layer: 1, pos: 176
type: RSZ, layer: 1, pos: 198
type: RSZ, layer: 1, pos: 199
type: RSZ, layer: 1, pos: 204
type: RSZ, layer: 1, pos: 250

Time for candidate selection: 0.17 seconds

### Candidate
type: RSZ, layer: 1, pos: 176

### Relational analysis RSZ of RS_RSZ1_RSZ1_RSZ1_RSZ1_RSZ1_RSZ1

#### Relational analysis RSZ result of RS_RSZ1_RSZ1_RSZ1_RSZ1_RSZ1_RSZ1
Status: Status.UNKNOWN
Output dim: 1, lower bound: -0.0008374, upper bound: 0.0008374
time: 0.63 seconds

### Relational analysis RSZ of RS_RSZ1_RSZ1_RSZ1_RSZ1_RSZ1_RSZ2

#### Relational analysis RSZ result of RS_RSZ1_RSZ1_RSZ1_RSZ1_RSZ1_RSZ2
Status: Status.UNKNOWN
Output dim: 1, lower bound: -0.0008374, upper bound: 0.0008374
time: 0.68 seconds

## BFS RS instance: RS_RSZ1_RSZ1_RSZ1_RSZ1_RSZ2

### Backsubstitution after applying RS history:
0: 0.0006996, 0.0011177, 0.0006996, 0.0011177, -0.0004181, 0.0004181
1: 0.9934013, 0.9943976, 0.9934013, 0.9943976, -0.0009964, 0.0009964
2: -0.0088595, -0.0053283, -0.0088595, -0.0053283, -0.0032524, 0.0032508
3: 0.0035633, 0.0041662, 0.0035633, 0.0041662, -0.0006029, 0.0006029
4: 0.0026282, 0.0054191, 0.0026282, 0.0054191, -0.0027909, 0.0027909
5: 0.0050201, 0.0064446, 0.0050201, 0.0064446, -0.0014245, 0.0014245
6: -0.0021963, -0.0008150, -0.0021963, -0.0008150, -0.0013813, 0.0013813
7: -0.0082640, -0.0074494, -0.0082640, -0.0074494, -0.0008146, 0.0008146
8: 0.0052524, 0.0098919, 0.0052524, 0.0098919, -0.0045646, 0.0045651
9: -0.0036845, -0.0030730, -0.0036845, -0.0030730, -0.0006115, 0.0006115

### Unstable ReLU Count (Linear/Conv2D Layers)
- layer_idx=0, type=LayerType.Linear, total=256, inp1_unstable=10, inp2_unstable=10, delta_unstable=251
- layer_idx=2, type=LayerType.Linear, total=256, inp1_unstable=8, inp2_unstable=8, delta_unstable=256
- layer_idx=4, type=LayerType.Linear, total=256, inp1_unstable=0, inp2_unstable=0, delta_unstable=256
- layer_idx=6, type=LayerType.Linear, total=256, inp1_unstable=2, inp2_unstable=2, delta_unstable=256
- layer_idx=8, type=LayerType.Linear, total=10, inp1_unstable=0, inp2_unstable=0, delta_unstable=10

Time for backsubstitution: 1.62 seconds

### RS candidates at layer 1
type: RSZ, layer: 1, pos: 176
type: RSZ, layer: 1, pos: 198
type: RSZ, layer: 1, pos: 199
type: RSZ, layer: 1, pos: 204
type: RSZ, layer: 1, pos: 250

Time for candidate selection: 0.18 seconds

### Candidate
type: RSZ, layer: 1, pos: 176

### Relational analysis RSZ of RS_RSZ1_RSZ1_RSZ1_RSZ1_RSZ2_RSZ1

#### Relational analysis RSZ result of RS_RSZ1_RSZ1_RSZ1_RSZ1_RSZ2_RSZ1
Status: Status.UNKNOWN
Output dim: 1, lower bound: -0.0008374, upper bound: 0.0008374
time: 0.65 seconds

### Relational analysis RSZ of RS_RSZ1_RSZ1_RSZ1_RSZ1_RSZ2_RSZ2

#### Relational analysis RSZ result of RS_RSZ1_RSZ1_RSZ1_RSZ1_RSZ2_RSZ2
Status: Status.UNKNOWN
Output dim: 1, lower bound: -0.0008374, upper bound: 0.0008374
time: 0.66 seconds

## BFS RS instance: RS_RSZ1_RSZ1_RSZ1_RSZ2_RSZ1

### Backsubstitution after applying RS history:
0: 0.0006996, 0.0011177, 0.0006996, 0.0011177, -0.0004181, 0.0004181
1: 0.9934013, 0.9943976, 0.9934013, 0.9943976, -0.0009964, 0.0009964
2: -0.0088595, -0.0053283, -0.0088595, -0.0053283, -0.0032535, 0.0032503
3: 0.0035633, 0.0041662, 0.0035633, 0.0041662, -0.0006029, 0.0006029
4: 0.0026282, 0.0054191, 0.0026282, 0.0054191, -0.0027909, 0.0027909
5: 0.0050201, 0.0064446, 0.0050201, 0.0064446, -0.0014245, 0.0014245
6: -0.0021963, -0.0008150, -0.0021963, -0.0008150, -0.0013813, 0.0013813
7: -0.0082640, -0.0074494, -0.0082640, -0.0074494, -0.0008146, 0.0008146
8: 0.0052524, 0.0098919, 0.0052524, 0.0098919, -0.0045645, 0.0045654
9: -0.0036845, -0.0030730, -0.0036845, -0.0030730, -0.0006115, 0.0006115

### Unstable ReLU Count (Linear/Conv2D Layers)
- layer_idx=0, type=LayerType.Linear, total=256, inp1_unstable=10, inp2_unstable=10, delta_unstable=251
- layer_idx=2, type=LayerType.Linear, total=256, inp1_unstable=8, inp2_unstable=8, delta_unstable=256
- layer_idx=4, type=LayerType.Linear, total=256, inp1_unstable=0, inp2_unstable=0, delta_unstable=256
- layer_idx=6, type=LayerType.Linear, total=256, inp1_unstable=2, inp2_unstable=2, delta_unstable=256
- layer_idx=8, type=LayerType.Linear, total=10, inp1_unstable=0, inp2_unstable=0, delta_unstable=10

Time for backsubstitution: 1.57 seconds

### RS candidates at layer 1
type: RSZ, layer: 1, pos: 176
type: RSZ, layer: 1, pos: 198
type: RSZ, layer: 1, pos: 199
type: RSZ, layer: 1, pos: 204
type: RSZ, layer: 1, pos: 250

Time for candidate selection: 0.17 seconds

### Candidate
type: RSZ, layer: 1, pos: 176

### Relational analysis RSZ of RS_RSZ1_RSZ1_RSZ1_RSZ2_RSZ1_RSZ1

#### Relational analysis RSZ result of RS_RSZ1_RSZ1_RSZ1_RSZ2_RSZ1_RSZ1
Status: Status.UNKNOWN
Output dim: 1, lower bound: -0.0008374, upper bound: 0.0008374
time: 0.65 seconds

### Relational analysis RSZ of RS_RSZ1_RSZ1_RSZ1_RSZ2_RSZ1_RSZ2

#### Relational analysis RSZ result of RS_RSZ1_RSZ1_RSZ1_RSZ2_RSZ1_RSZ2
Status: Status.UNKNOWN
Output dim: 1, lower bound: -0.0008374, upper bound: 0.0008374
time: 0.69 seconds

## BFS RS instance: RS_RSZ1_RSZ1_RSZ1_RSZ2_RSZ2

### Backsubstitution after applying RS history:
0: 0.0006996, 0.0011177, 0.0006996, 0.0011177, -0.0004181, 0.0004181
1: 0.9934013, 0.9943976, 0.9934013, 0.9943976, -0.0009964, 0.0009964
2: -0.0088595, -0.0053283, -0.0088595, -0.0053283, -0.0032515, 0.0032517
3: 0.0035633, 0.0041662, 0.0035633, 0.0041662, -0.0006029, 0.0006029
4: 0.0026282, 0.0054191, 0.0026282, 0.0054191, -0.0027909, 0.0027909
5: 0.0050201, 0.0064446, 0.0050201, 0.0064446, -0.0014245, 0.0014245
6: -0.0021963, -0.0008150, -0.0021963, -0.0008150, -0.0013813, 0.0013813
7: -0.0082640, -0.0074494, -0.0082640, -0.0074494, -0.0008146, 0.0008146
8: 0.0052524, 0.0098919, 0.0052524, 0.0098919, -0.0045649, 0.0045648
9: -0.0036845, -0.0030730, -0.0036845, -0.0030730, -0.0006115, 0.0006115

### Unstable ReLU Count (Linear/Conv2D Layers)
- layer_idx=0, type=LayerType.Linear, total=256, inp1_unstable=10, inp2_unstable=10, delta_unstable=251
- layer_idx=2, type=LayerType.Linear, total=256, inp1_unstable=8, inp2_unstable=8, delta_unstable=256
- layer_idx=4, type=LayerType.Linear, total=256, inp1_unstable=0, inp2_unstable=0, delta_unstable=256
- layer_idx=6, type=LayerType.Linear, total=256, inp1_unstable=2, inp2_unstable=2, delta_unstable=256
- layer_idx=8, type=LayerType.Linear, total=10, inp1_unstable=0, inp2_unstable=0, delta_unstable=10

Time for backsubstitution: 1.60 seconds

### RS candidates at layer 1
type: RSZ, layer: 1, pos: 176
type: RSZ, layer: 1, pos: 198
type: RSZ, layer: 1, pos: 199
type: RSZ, layer: 1, pos: 204
type: RSZ, layer: 1, pos: 250

Time for candidate selection: 0.17 seconds

### Candidate
type: RSZ, layer: 1, pos: 176

### Relational analysis RSZ of RS_RSZ1_RSZ1_RSZ1_RSZ2_RSZ2_RSZ1

#### Relational analysis RSZ result of RS_RSZ1_RSZ1_RSZ1_RSZ2_RSZ2_RSZ1
Status: Status.UNKNOWN
Output dim: 1, lower bound: -0.0008374, upper bound: 0.0008374
time: 0.64 seconds

### Relational analysis RSZ of RS_RSZ1_RSZ1_RSZ1_RSZ2_RSZ2_RSZ2

#### Relational analysis RSZ result of RS_RSZ1_RSZ1_RSZ1_RSZ2_RSZ2_RSZ2
Status: Status.UNKNOWN
Output dim: 1, lower bound: -0.0008374, upper bound: 0.0008374
time: 0.74 seconds

## BFS RS instance: RS_RSZ1_RSZ1_RSZ2_RSZ1_RSZ1

### Backsubstitution after applying RS history:
0: 0.0006996, 0.0011177, 0.0006996, 0.0011177, -0.0004181, 0.0004181
1: 0.9934013, 0.9943976, 0.9934013, 0.9943976, -0.0009964, 0.0009964
2: -0.0088595, -0.0053283, -0.0088595, -0.0053283, -0.0032539, 0.0032490
3: 0.0035633, 0.0041662, 0.0035633, 0.0041662, -0.0006029, 0.0006029
4: 0.0026282, 0.0054191, 0.0026282, 0.0054191, -0.0027909, 0.0027909
5: 0.0050201, 0.0064446, 0.0050201, 0.0064446, -0.0014245, 0.0014245
6: -0.0021963, -0.0008150, -0.0021963, -0.0008150, -0.0013813, 0.0013813
7: -0.0082640, -0.0074494, -0.0082640, -0.0074494, -0.0008146, 0.0008146
8: 0.0052524, 0.0098919, 0.0052524, 0.0098919, -0.0045642, 0.0045655
9: -0.0036845, -0.0030730, -0.0036845, -0.0030730, -0.0006115, 0.0006115

### Unstable ReLU Count (Linear/Conv2D Layers)
- layer_idx=0, type=LayerType.Linear, total=256, inp1_unstable=10, inp2_unstable=10, delta_unstable=251
- layer_idx=2, type=LayerType.Linear, total=256, inp1_unstable=8, inp2_unstable=8, delta_unstable=256
- layer_idx=4, type=LayerType.Linear, total=256, inp1_unstable=0, inp2_unstable=0, delta_unstable=256
- layer_idx=6, type=LayerType.Linear, total=256, inp1_unstable=2, inp2_unstable=2, delta_unstable=256
- layer_idx=8, type=LayerType.Linear, total=10, inp1_unstable=0, inp2_unstable=0, delta_unstable=10

Time for backsubstitution: 1.70 seconds

### RS candidates at layer 1
type: RSZ, layer: 1, pos: 176
type: RSZ, layer: 1, pos: 198
type: RSZ, layer: 1, pos: 199
type: RSZ, layer: 1, pos: 204
type: RSZ, layer: 1, pos: 250

Time for candidate selection: 0.20 seconds

### Candidate
type: RSZ, layer: 1, pos: 176

### Relational analysis RSZ of RS_RSZ1_RSZ1_RSZ2_RSZ1_RSZ1_RSZ1

#### Relational analysis RSZ result of RS_RSZ1_RSZ1_RSZ2_RSZ1_RSZ1_RSZ1
Status: Status.UNKNOWN
Output dim: 1, lower bound: -0.0008374, upper bound: 0.0008374
time: 0.69 seconds

### Relational analysis RSZ of RS_RSZ1_RSZ1_RSZ2_RSZ1_RSZ1_RSZ2

#### Relational analysis RSZ result of RS_RSZ1_RSZ1_RSZ2_RSZ1_RSZ1_RSZ2
Status: Status.UNKNOWN
Output dim: 1, lower bound: -0.0008374, upper bound: 0.0008374
time: 0.70 seconds

## BFS RS instance: RS_RSZ1_RSZ1_RSZ2_RSZ1_RSZ2

### Backsubstitution after applying RS history:
0: 0.0006996, 0.0011177, 0.0006996, 0.0011177, -0.0004181, 0.0004181
1: 0.9934013, 0.9943976, 0.9934013, 0.9943976, -0.0009964, 0.0009964
2: -0.0088595, -0.0053283, -0.0088595, -0.0053283, -0.0032523, 0.0032508
3: 0.0035633, 0.0041662, 0.0035633, 0.0041662, -0.0006029, 0.0006029
4: 0.0026282, 0.0054191, 0.0026282, 0.0054191, -0.0027909, 0.0027909
5: 0.0050201, 0.0064446, 0.0050201, 0.0064446, -0.0014245, 0.0014245
6: -0.0021963, -0.0008150, -0.0021963, -0.0008150, -0.0013813, 0.0013813
7: -0.0082640, -0.0074494, -0.0082640, -0.0074494, -0.0008146, 0.0008146
8: 0.0052524, 0.0098919, 0.0052524, 0.0098919, -0.0045646, 0.0045650
9: -0.0036845, -0.0030730, -0.0036845, -0.0030730, -0.0006115, 0.0006115

### Unstable ReLU Count (Linear/Conv2D Layers)
- layer_idx=0, type=LayerType.Linear, total=256, inp1_unstable=10, inp2_unstable=10, delta_unstable=251
- layer_idx=2, type=LayerType.Linear, total=256, inp1_unstable=8, inp2_unstable=8, delta_unstable=256
- layer_idx=4, type=LayerType.Linear, total=256, inp1_unstable=0, inp2_unstable=0, delta_unstable=256
- layer_idx=6, type=LayerType.Linear, total=256, inp1_unstable=2, inp2_unstable=2, delta_unstable=256
- layer_idx=8, type=LayerType.Linear, total=10, inp1_unstable=0, inp2_unstable=0, delta_unstable=10

Time for backsubstitution: 1.70 seconds

### RS candidates at layer 1
type: RSZ, layer: 1, pos: 176
type: RSZ, layer: 1, pos: 198
type: RSZ, layer: 1, pos: 199
type: RSZ, layer: 1, pos: 204
type: RSZ, layer: 1, pos: 250

Time for candidate selection: 0.19 seconds

### Candidate
type: RSZ, layer: 1, pos: 176

### Relational analysis RSZ of RS_RSZ1_RSZ1_RSZ2_RSZ1_RSZ2_RSZ1

#### Relational analysis RSZ result of RS_RSZ1_RSZ1_RSZ2_RSZ1_RSZ2_RSZ1
Status: Status.UNKNOWN
Output dim: 1, lower bound: -0.0008374, upper bound: 0.0008374
time: 0.88 seconds

### Relational analysis RSZ of RS_RSZ1_RSZ1_RSZ2_RSZ1_RSZ2_RSZ2

#### Relational analysis RSZ result of RS_RSZ1_RSZ1_RSZ2_RSZ1_RSZ2_RSZ2
Status: Status.UNKNOWN
Output dim: 1, lower bound: -0.0008374, upper bound: 0.0008374
time: 0.70 seconds

## BFS RS instance: RS_RSZ1_RSZ1_RSZ2_RSZ2_RSZ1

### Backsubstitution after applying RS history:
0: 0.0006996, 0.0011177, 0.0006996, 0.0011177, -0.0004181, 0.0004181
1: 0.9934013, 0.9943976, 0.9934013, 0.9943976, -0.0009964, 0.0009964
2: -0.0088595, -0.0053283, -0.0088595, -0.0053283, -0.0032531, 0.0032499
3: 0.0035633, 0.0041662, 0.0035633, 0.0041662, -0.0006029, 0.0006029
4: 0.0026282, 0.0054191, 0.0026282, 0.0054191, -0.0027909, 0.0027909
5: 0.0050201, 0.0064446, 0.0050201, 0.0064446, -0.0014245, 0.0014245
6: -0.0021963, -0.0008150, -0.0021963, -0.0008150, -0.0013813, 0.0013813
7: -0.0082640, -0.0074494, -0.0082640, -0.0074494, -0.0008146, 0.0008146
8: 0.0052524, 0.0098919, 0.0052524, 0.0098919, -0.0045644, 0.0045652
9: -0.0036845, -0.0030730, -0.0036845, -0.0030730, -0.0006115, 0.0006115

### Unstable ReLU Count (Linear/Conv2D Layers)
- layer_idx=0, type=LayerType.Linear, total=256, inp1_unstable=10, inp2_unstable=10, delta_unstable=251
- layer_idx=2, type=LayerType.Linear, total=256, inp1_unstable=8, inp2_unstable=8, delta_unstable=256
- layer_idx=4, type=LayerType.Linear, total=256, inp1_unstable=0, inp2_unstable=0, delta_unstable=256
- layer_idx=6, type=LayerType.Linear, total=256, inp1_unstable=2, inp2_unstable=2, delta_unstable=256
- layer_idx=8, type=LayerType.Linear, total=10, inp1_unstable=0, inp2_unstable=0, delta_unstable=10

Time for backsubstitution: 1.71 seconds

### RS candidates at layer 1
type: RSZ, layer: 1, pos: 176
type: RSZ, layer: 1, pos: 198
type: RSZ, layer: 1, pos: 199
type: RSZ, layer: 1, pos: 204
type: RSZ, layer: 1, pos: 250

Time for candidate selection: 0.17 seconds

### Candidate
type: RSZ, layer: 1, pos: 176

### Relational analysis RSZ of RS_RSZ1_RSZ1_RSZ2_RSZ2_RSZ1_RSZ1

#### Relational analysis RSZ result of RS_RSZ1_RSZ1_RSZ2_RSZ2_RSZ1_RSZ1
Status: Status.UNKNOWN
Output dim: 1, lower bound: -0.0008374, upper bound: 0.0008374
time: 0.71 seconds

### Relational analysis RSZ of RS_RSZ1_RSZ1_RSZ2_RSZ2_RSZ1_RSZ2

#### Relational analysis RSZ result of RS_RSZ1_RSZ1_RSZ2_RSZ2_RSZ1_RSZ2
Status: Status.UNKNOWN
Output dim: 1, lower bound: -0.0008374, upper bound: 0.0008374
time: 0.67 seconds

## BFS RS instance: RS_RSZ1_RSZ1_RSZ2_RSZ2_RSZ2

### Backsubstitution after applying RS history:
0: 0.0006996, 0.0011177, 0.0006996, 0.0011177, -0.0004181, 0.0004181
1: 0.9934013, 0.9943976, 0.9934013, 0.9943976, -0.0009964, 0.0009964
2: -0.0088595, -0.0053283, -0.0088595, -0.0053283, -0.0032510, 0.0032518
3: 0.0035633, 0.0041662, 0.0035633, 0.0041662, -0.0006029, 0.0006029
4: 0.0026282, 0.0054191, 0.0026282, 0.0054191, -0.0027909, 0.0027909
5: 0.0050201, 0.0064446, 0.0050201, 0.0064446, -0.0014245, 0.0014245
6: -0.0021963, -0.0008150, -0.0021963, -0.0008150, -0.0013813, 0.0013813
7: -0.0082640, -0.0074494, -0.0082640, -0.0074494, -0.0008146, 0.0008146
8: 0.0052524, 0.0098919, 0.0052524, 0.0098919, -0.0045649, 0.0045647
9: -0.0036845, -0.0030730, -0.0036845, -0.0030730, -0.0006115, 0.0006115

### Unstable ReLU Count (Linear/Conv2D Layers)
- layer_idx=0, type=LayerType.Linear, total=256, inp1_unstable=10, inp2_unstable=10, delta_unstable=251
- layer_idx=2, type=LayerType.Linear, total=256, inp1_unstable=8, inp2_unstable=8, delta_unstable=256
- layer_idx=4, type=LayerType.Linear, total=256, inp1_unstable=0, inp2_unstable=0, delta_unstable=256
- layer_idx=6, type=LayerType.Linear, total=256, inp1_unstable=2, inp2_unstable=2, delta_unstable=256
- layer_idx=8, type=LayerType.Linear, total=10, inp1_unstable=0, inp2_unstable=0, delta_unstable=10

Time for backsubstitution: 1.59 seconds

### RS candidates at layer 1
type: RSZ, layer: 1, pos: 176
type: RSZ, layer: 1, pos: 198
type: RSZ, layer: 1, pos: 199
type: RSZ, layer: 1, pos: 204
type: RSZ, layer: 1, pos: 250

Time for candidate selection: 0.17 seconds

### Candidate
type: RSZ, layer: 1, pos: 176

### Relational analysis RSZ of RS_RSZ1_RSZ1_RSZ2_RSZ2_RSZ2_RSZ1

#### Relational analysis RSZ result of RS_RSZ1_RSZ1_RSZ2_RSZ2_RSZ2_RSZ1
Status: Status.UNKNOWN
Output dim: 1, lower bound: -0.0008374, upper bound: 0.0008374
time: 0.71 seconds

### Relational analysis RSZ of RS_RSZ1_RSZ1_RSZ2_RSZ2_RSZ2_RSZ2

#### Relational analysis RSZ result of RS_RSZ1_RSZ1_RSZ2_RSZ2_RSZ2_RSZ2
Status: Status.UNKNOWN
Output dim: 1, lower bound: -0.0008374, upper bound: 0.0008374
time: 0.68 seconds

## BFS RS instance: RS_RSZ1_RSZ2_RSZ1_RSZ1_RSZ1

### Backsubstitution after applying RS history:
0: 0.0006996, 0.0011177, 0.0006996, 0.0011177, -0.0004181, 0.0004181
1: 0.9934013, 0.9943976, 0.9934013, 0.9943976, -0.0009964, 0.0009964
2: -0.0088595, -0.0053283, -0.0088595, -0.0053283, -0.0032520, 0.0032515
3: 0.0035633, 0.0041662, 0.0035633, 0.0041662, -0.0006029, 0.0006029
4: 0.0026282, 0.0054191, 0.0026282, 0.0054191, -0.0027909, 0.0027909
5: 0.0050201, 0.0064446, 0.0050201, 0.0064446, -0.0014245, 0.0014245
6: -0.0021963, -0.0008150, -0.0021963, -0.0008150, -0.0013813, 0.0013813
7: -0.0082640, -0.0074494, -0.0082640, -0.0074494, -0.0008146, 0.0008146
8: 0.0052524, 0.0098919, 0.0052524, 0.0098919, -0.0045648, 0.0045650
9: -0.0036845, -0.0030730, -0.0036845, -0.0030730, -0.0006115, 0.0006115

### Unstable ReLU Count (Linear/Conv2D Layers)
- layer_idx=0, type=LayerType.Linear, total=256, inp1_unstable=10, inp2_unstable=10, delta_unstable=251
- layer_idx=2, type=LayerType.Linear, total=256, inp1_unstable=8, inp2_unstable=8, delta_unstable=256
- layer_idx=4, type=LayerType.Linear, total=256, inp1_unstable=0, inp2_unstable=0, delta_unstable=256
- layer_idx=6, type=LayerType.Linear, total=256, inp1_unstable=2, inp2_unstable=2, delta_unstable=256
- layer_idx=8, type=LayerType.Linear, total=10, inp1_unstable=0, inp2_unstable=0, delta_unstable=10

Time for backsubstitution: 1.68 seconds

### RS candidates at layer 1
type: RSZ, layer: 1, pos: 176
type: RSZ, layer: 1, pos: 198
type: RSZ, layer: 1, pos: 199
type: RSZ, layer: 1, pos: 204
type: RSZ, layer: 1, pos: 250

Time for candidate selection: 0.18 seconds

### Candidate
type: RSZ, layer: 1, pos: 176

### Relational analysis RSZ of RS_RSZ1_RSZ2_RSZ1_RSZ1_RSZ1_RSZ1

#### Relational analysis RSZ result of RS_RSZ1_RSZ2_RSZ1_RSZ1_RSZ1_RSZ1
Status: Status.UNKNOWN
Output dim: 1, lower bound: -0.0008374, upper bound: 0.0008374
time: 0.62 seconds

### Relational analysis RSZ of RS_RSZ1_RSZ2_RSZ1_RSZ1_RSZ1_RSZ2

#### Relational analysis RSZ result of RS_RSZ1_RSZ2_RSZ1_RSZ1_RSZ1_RSZ2
Status: Status.UNKNOWN
Output dim: 1, lower bound: -0.0008374, upper bound: 0.0008374
time: 0.69 seconds

## BFS RS instance: RS_RSZ1_RSZ2_RSZ1_RSZ1_RSZ2

### Backsubstitution after applying RS history:
0: 0.0006996, 0.0011177, 0.0006996, 0.0011177, -0.0004181, 0.0004181
1: 0.9934013, 0.9943976, 0.9934013, 0.9943976, -0.0009964, 0.0009964
2: -0.0088595, -0.0053283, -0.0088595, -0.0053283, -0.0032499, 0.0032535
3: 0.0035633, 0.0041662, 0.0035633, 0.0041662, -0.0006029, 0.0006029
4: 0.0026282, 0.0054191, 0.0026282, 0.0054191, -0.0027909, 0.0027909
5: 0.0050201, 0.0064446, 0.0050201, 0.0064446, -0.0014245, 0.0014245
6: -0.0021963, -0.0008150, -0.0021963, -0.0008150, -0.0013813, 0.0013813
7: -0.0082640, -0.0074494, -0.0082640, -0.0074494, -0.0008146, 0.0008146
8: 0.0052524, 0.0098919, 0.0052524, 0.0098919, -0.0045654, 0.0045644
9: -0.0036845, -0.0030730, -0.0036845, -0.0030730, -0.0006115, 0.0006115

### Unstable ReLU Count (Linear/Conv2D Layers)
- layer_idx=0, type=LayerType.Linear, total=256, inp1_unstable=10, inp2_unstable=10, delta_unstable=251
- layer_idx=2, type=LayerType.Linear, total=256, inp1_unstable=8, inp2_unstable=8, delta_unstable=256
- layer_idx=4, type=LayerType.Linear, total=256, inp1_unstable=0, inp2_unstable=0, delta_unstable=256
- layer_idx=6, type=LayerType.Linear, total=256, inp1_unstable=2, inp2_unstable=2, delta_unstable=256
- layer_idx=8, type=LayerType.Linear, total=10, inp1_unstable=0, inp2_unstable=0, delta_unstable=10

Time for backsubstitution: 1.65 seconds

### RS candidates at layer 1
type: RSZ, layer: 1, pos: 176
type: RSZ, layer: 1, pos: 198
type: RSZ, layer: 1, pos: 199
type: RSZ, layer: 1, pos: 204
type: RSZ, layer: 1, pos: 250

Time for candidate selection: 0.16 seconds

### Candidate
type: RSZ, layer: 1, pos: 176

### Relational analysis RSZ of RS_RSZ1_RSZ2_RSZ1_RSZ1_RSZ2_RSZ1

#### Relational analysis RSZ result of RS_RSZ1_RSZ2_RSZ1_RSZ1_RSZ2_RSZ1
Status: Status.UNKNOWN
Output dim: 1, lower bound: -0.0008374, upper bound: 0.0008374
time: 0.64 seconds

### Relational analysis RSZ of RS_RSZ1_RSZ2_RSZ1_RSZ1_RSZ2_RSZ2

#### Relational analysis RSZ result of RS_RSZ1_RSZ2_RSZ1_RSZ1_RSZ2_RSZ2
Status: Status.UNKNOWN
Output dim: 1, lower bound: -0.0008374, upper bound: 0.0008374
time: 0.67 seconds

## BFS RS instance: RS_RSZ1_RSZ2_RSZ1_RSZ2_RSZ1

### Backsubstitution after applying RS history:
0: 0.0006996, 0.0011177, 0.0006996, 0.0011177, -0.0004181, 0.0004181
1: 0.9934013, 0.9943976, 0.9934013, 0.9943976, -0.0009964, 0.0009964
2: -0.0088595, -0.0053283, -0.0088595, -0.0053283, -0.0032507, 0.0032529
3: 0.0035633, 0.0041662, 0.0035633, 0.0041662, -0.0006029, 0.0006029
4: 0.0026282, 0.0054191, 0.0026282, 0.0054191, -0.0027909, 0.0027909
5: 0.0050201, 0.0064446, 0.0050201, 0.0064446, -0.0014245, 0.0014245
6: -0.0021963, -0.0008150, -0.0021963, -0.0008150, -0.0013813, 0.0013813
7: -0.0082640, -0.0074494, -0.0082640, -0.0074494, -0.0008146, 0.0008146
8: 0.0052524, 0.0098919, 0.0052524, 0.0098919, -0.0045652, 0.0045646
9: -0.0036845, -0.0030730, -0.0036845, -0.0030730, -0.0006115, 0.0006115

### Unstable ReLU Count (Linear/Conv2D Layers)
- layer_idx=0, type=LayerType.Linear, total=256, inp1_unstable=10, inp2_unstable=10, delta_unstable=251
- layer_idx=2, type=LayerType.Linear, total=256, inp1_unstable=8, inp2_unstable=8, delta_unstable=256
- layer_idx=4, type=LayerType.Linear, total=256, inp1_unstable=0, inp2_unstable=0, delta_unstable=256
- layer_idx=6, type=LayerType.Linear, total=256, inp1_unstable=2, inp2_unstable=2, delta_unstable=256
- layer_idx=8, type=LayerType.Linear, total=10, inp1_unstable=0, inp2_unstable=0, delta_unstable=10

Time for backsubstitution: 1.64 seconds

### RS candidates at layer 1
type: RSZ, layer: 1, pos: 176
type: RSZ, layer: 1, pos: 198
type: RSZ, layer: 1, pos: 199
type: RSZ, layer: 1, pos: 204
type: RSZ, layer: 1, pos: 250

Time for candidate selection: 0.17 seconds

### Candidate
type: RSZ, layer: 1, pos: 176

### Relational analysis RSZ of RS_RSZ1_RSZ2_RSZ1_RSZ2_RSZ1_RSZ1

#### Relational analysis RSZ result of RS_RSZ1_RSZ2_RSZ1_RSZ2_RSZ1_RSZ1
Status: Status.UNKNOWN
Output dim: 1, lower bound: -0.0008374, upper bound: 0.0008374
time: 0.71 seconds

### Relational analysis RSZ of RS_RSZ1_RSZ2_RSZ1_RSZ2_RSZ1_RSZ2

#### Relational analysis RSZ result of RS_RSZ1_RSZ2_RSZ1_RSZ2_RSZ1_RSZ2
Status: Status.UNKNOWN
Output dim: 1, lower bound: -0.0008374, upper bound: 0.0008374
time: 0.69 seconds

## BFS RS instance: RS_RSZ1_RSZ2_RSZ1_RSZ2_RSZ2

### Backsubstitution after applying RS history:
0: 0.0006996, 0.0011177, 0.0006996, 0.0011177, -0.0004181, 0.0004181
1: 0.9934013, 0.9943976, 0.9934013, 0.9943976, -0.0009964, 0.0009964
2: -0.0088595, -0.0053283, -0.0088595, -0.0053283, -0.0032486, 0.0032542
3: 0.0035633, 0.0041662, 0.0035633, 0.0041662, -0.0006029, 0.0006029
4: 0.0026282, 0.0054191, 0.0026282, 0.0054191, -0.0027909, 0.0027909
5: 0.0050201, 0.0064446, 0.0050201, 0.0064446, -0.0014245, 0.0014245
6: -0.0021963, -0.0008150, -0.0021963, -0.0008150, -0.0013813, 0.0013813
7: -0.0082640, -0.0074494, -0.0082640, -0.0074494, -0.0008146, 0.0008146
8: 0.0052524, 0.0098919, 0.0052524, 0.0098919, -0.0045655, 0.0045640
9: -0.0036845, -0.0030730, -0.0036845, -0.0030730, -0.0006115, 0.0006115

### Unstable ReLU Count (Linear/Conv2D Layers)
- layer_idx=0, type=LayerType.Linear, total=256, inp1_unstable=10, inp2_unstable=10, delta_unstable=251
- layer_idx=2, type=LayerType.Linear, total=256, inp1_unstable=8, inp2_unstable=8, delta_unstable=256
- layer_idx=4, type=LayerType.Linear, total=256, inp1_unstable=0, inp2_unstable=0, delta_unstable=256
- layer_idx=6, type=LayerType.Linear, total=256, inp1_unstable=2, inp2_unstable=2, delta_unstable=256
- layer_idx=8, type=LayerType.Linear, total=10, inp1_unstable=0, inp2_unstable=0, delta_unstable=10

Time for backsubstitution: 1.65 seconds

### RS candidates at layer 1
type: RSZ, layer: 1, pos: 176
type: RSZ, layer: 1, pos: 198
type: RSZ, layer: 1, pos: 199
type: RSZ, layer: 1, pos: 204
type: RSZ, layer: 1, pos: 250

Time for candidate selection: 0.17 seconds

### Candidate
type: RSZ, layer: 1, pos: 176

### Relational analysis RSZ of RS_RSZ1_RSZ2_RSZ1_RSZ2_RSZ2_RSZ1

#### Relational analysis RSZ result of RS_RSZ1_RSZ2_RSZ1_RSZ2_RSZ2_RSZ1
Status: Status.UNKNOWN
Output dim: 1, lower bound: -0.0008374, upper bound: 0.0008374
time: 0.64 seconds

### Relational analysis RSZ of RS_RSZ1_RSZ2_RSZ1_RSZ2_RSZ2_RSZ2

#### Relational analysis RSZ result of RS_RSZ1_RSZ2_RSZ1_RSZ2_RSZ2_RSZ2
Status: Status.UNKNOWN
Output dim: 1, lower bound: -0.0008374, upper bound: 0.0008374
time: 0.68 seconds

## BFS RS instance: RS_RSZ1_RSZ2_RSZ2_RSZ1_RSZ1

### Backsubstitution after applying RS history:
0: 0.0006996, 0.0011177, 0.0006996, 0.0011177, -0.0004181, 0.0004181
1: 0.9934013, 0.9943976, 0.9934013, 0.9943976, -0.0009964, 0.0009964
2: -0.0088595, -0.0053283, -0.0088595, -0.0053283, -0.0032519, 0.0032519
3: 0.0035633, 0.0041662, 0.0035633, 0.0041662, -0.0006029, 0.0006029
4: 0.0026282, 0.0054191, 0.0026282, 0.0054191, -0.0027909, 0.0027909
5: 0.0050201, 0.0064446, 0.0050201, 0.0064446, -0.0014245, 0.0014245
6: -0.0021963, -0.0008150, -0.0021963, -0.0008150, -0.0013813, 0.0013813
7: -0.0082640, -0.0074494, -0.0082640, -0.0074494, -0.0008146, 0.0008146
8: 0.0052524, 0.0098919, 0.0052524, 0.0098919, -0.0045649, 0.0045649
9: -0.0036845, -0.0030730, -0.0036845, -0.0030730, -0.0006115, 0.0006115

### Unstable ReLU Count (Linear/Conv2D Layers)
- layer_idx=0, type=LayerType.Linear, total=256, inp1_unstable=10, inp2_unstable=10, delta_unstable=251
- layer_idx=2, type=LayerType.Linear, total=256, inp1_unstable=8, inp2_unstable=8, delta_unstable=256
- layer_idx=4, type=LayerType.Linear, total=256, inp1_unstable=0, inp2_unstable=0, delta_unstable=256
- layer_idx=6, type=LayerType.Linear, total=256, inp1_unstable=2, inp2_unstable=2, delta_unstable=256
- layer_idx=8, type=LayerType.Linear, total=10, inp1_unstable=0, inp2_unstable=0, delta_unstable=10

Time for backsubstitution: 1.68 seconds

### RS candidates at layer 1
type: RSZ, layer: 1, pos: 176
type: RSZ, layer: 1, pos: 198
type: RSZ, layer: 1, pos: 199
type: RSZ, layer: 1, pos: 204
type: RSZ, layer: 1, pos: 250

Time for candidate selection: 0.24 seconds

### Candidate
type: RSZ, layer: 1, pos: 176

### Relational analysis RSZ of RS_RSZ1_RSZ2_RSZ2_RSZ1_RSZ1_RSZ1

#### Relational analysis RSZ result of RS_RSZ1_RSZ2_RSZ2_RSZ1_RSZ1_RSZ1
Status: Status.UNKNOWN
Output dim: 1, lower bound: -0.0008374, upper bound: 0.0008374
time: 0.69 seconds

### Relational analysis RSZ of RS_RSZ1_RSZ2_RSZ2_RSZ1_RSZ1_RSZ2

#### Relational analysis RSZ result of RS_RSZ1_RSZ2_RSZ2_RSZ1_RSZ1_RSZ2
Status: Status.UNKNOWN
Output dim: 1, lower bound: -0.0008374, upper bound: 0.0008374
time: 0.66 seconds

## BFS RS instance: RS_RSZ1_RSZ2_RSZ2_RSZ1_RSZ2

### Backsubstitution after applying RS history:
0: 0.0006996, 0.0011177, 0.0006996, 0.0011177, -0.0004181, 0.0004181
1: 0.9934013, 0.9943976, 0.9934013, 0.9943976, -0.0009964, 0.0009964
2: -0.0088595, -0.0053283, -0.0088595, -0.0053283, -0.0032499, 0.0032537
3: 0.0035633, 0.0041662, 0.0035633, 0.0041662, -0.0006029, 0.0006029
4: 0.0026282, 0.0054191, 0.0026282, 0.0054191, -0.0027909, 0.0027909
5: 0.0050201, 0.0064446, 0.0050201, 0.0064446, -0.0014245, 0.0014245
6: -0.0021963, -0.0008150, -0.0021963, -0.0008150, -0.0013813, 0.0013813
7: -0.0082640, -0.0074494, -0.0082640, -0.0074494, -0.0008146, 0.0008146
8: 0.0052524, 0.0098919, 0.0052524, 0.0098919, -0.0045654, 0.0045644
9: -0.0036845, -0.0030730, -0.0036845, -0.0030730, -0.0006115, 0.0006115

### Unstable ReLU Count (Linear/Conv2D Layers)
- layer_idx=0, type=LayerType.Linear, total=256, inp1_unstable=10, inp2_unstable=10, delta_unstable=251
- layer_idx=2, type=LayerType.Linear, total=256, inp1_unstable=8, inp2_unstable=8, delta_unstable=256
- layer_idx=4, type=LayerType.Linear, total=256, inp1_unstable=0, inp2_unstable=0, delta_unstable=256
- layer_idx=6, type=LayerType.Linear, total=256, inp1_unstable=2, inp2_unstable=2, delta_unstable=256
- layer_idx=8, type=LayerType.Linear, total=10, inp1_unstable=0, inp2_unstable=0, delta_unstable=10

Time for backsubstitution: 1.60 seconds

### RS candidates at layer 1
type: RSZ, layer: 1, pos: 176
type: RSZ, layer: 1, pos: 198
type: RSZ, layer: 1, pos: 199
type: RSZ, layer: 1, pos: 204
type: RSZ, layer: 1, pos: 250

Time for candidate selection: 0.17 seconds

### Candidate
type: RSZ, layer: 1, pos: 176

### Relational analysis RSZ of RS_RSZ1_RSZ2_RSZ2_RSZ1_RSZ2_RSZ1

#### Relational analysis RSZ result of RS_RSZ1_RSZ2_RSZ2_RSZ1_RSZ2_RSZ1
Status: Status.UNKNOWN
Output dim: 1, lower bound: -0.0008374, upper bound: 0.0008374
time: 0.66 seconds

### Relational analysis RSZ of RS_RSZ1_RSZ2_RSZ2_RSZ1_RSZ2_RSZ2

#### Relational analysis RSZ result of RS_RSZ1_RSZ2_RSZ2_RSZ1_RSZ2_RSZ2
Status: Status.UNKNOWN
Output dim: 1, lower bound: -0.0008374, upper bound: 0.0008374
time: 0.66 seconds

## BFS RS instance: RS_RSZ1_RSZ2_RSZ2_RSZ2_RSZ1

### Backsubstitution after applying RS history:
0: 0.0006996, 0.0011177, 0.0006996, 0.0011177, -0.0004181, 0.0004181
1: 0.9934013, 0.9943976, 0.9934013, 0.9943976, -0.0009964, 0.0009964
2: -0.0088595, -0.0053283, -0.0088595, -0.0053283, -0.0032505, 0.0032529
3: 0.0035633, 0.0041662, 0.0035633, 0.0041662, -0.0006029, 0.0006029
4: 0.0026282, 0.0054191, 0.0026282, 0.0054191, -0.0027909, 0.0027909
5: 0.0050201, 0.0064446, 0.0050201, 0.0064446, -0.0014245, 0.0014245
6: -0.0021963, -0.0008150, -0.0021963, -0.0008150, -0.0013813, 0.0013813
7: -0.0082640, -0.0074494, -0.0082640, -0.0074494, -0.0008146, 0.0008146
8: 0.0052524, 0.0098919, 0.0052524, 0.0098919, -0.0045652, 0.0045646
9: -0.0036845, -0.0030730, -0.0036845, -0.0030730, -0.0006115, 0.0006115

### Unstable ReLU Count (Linear/Conv2D Layers)
- layer_idx=0, type=LayerType.Linear, total=256, inp1_unstable=10, inp2_unstable=10, delta_unstable=251
- layer_idx=2, type=LayerType.Linear, total=256, inp1_unstable=8, inp2_unstable=8, delta_unstable=256
- layer_idx=4, type=LayerType.Linear, total=256, inp1_unstable=0, inp2_unstable=0, delta_unstable=256
- layer_idx=6, type=LayerType.Linear, total=256, inp1_unstable=2, inp2_unstable=2, delta_unstable=256
- layer_idx=8, type=LayerType.Linear, total=10, inp1_unstable=0, inp2_unstable=0, delta_unstable=10

Time for backsubstitution: 1.61 seconds

### RS candidates at layer 1
type: RSZ, layer: 1, pos: 176
type: RSZ, layer: 1, pos: 198
type: RSZ, layer: 1, pos: 199
type: RSZ, layer: 1, pos: 204
type: RSZ, layer: 1, pos: 250

Time for candidate selection: 0.17 seconds

### Candidate
type: RSZ, layer: 1, pos: 176

### Relational analysis RSZ of RS_RSZ1_RSZ2_RSZ2_RSZ2_RSZ1_RSZ1

#### Relational analysis RSZ result of RS_RSZ1_RSZ2_RSZ2_RSZ2_RSZ1_RSZ1
Status: Status.UNKNOWN
Output dim: 1, lower bound: -0.0008374, upper bound: 0.0008374
time: 0.65 seconds

### Relational analysis RSZ of RS_RSZ1_RSZ2_RSZ2_RSZ2_RSZ1_RSZ2

#### Relational analysis RSZ result of RS_RSZ1_RSZ2_RSZ2_RSZ2_RSZ1_RSZ2
Status: Status.UNKNOWN
Output dim: 1, lower bound: -0.0008374, upper bound: 0.0008374
time: 0.70 seconds

## BFS RS instance: RS_RSZ1_RSZ2_RSZ2_RSZ2_RSZ2

### Backsubstitution after applying RS history:
0: 0.0006996, 0.0011177, 0.0006996, 0.0011177, -0.0004181, 0.0004181
1: 0.9934013, 0.9943976, 0.9934013, 0.9943976, -0.0009964, 0.0009964
2: -0.0088595, -0.0053283, -0.0088595, -0.0053283, -0.0032485, 0.0032545
3: 0.0035633, 0.0041662, 0.0035633, 0.0041662, -0.0006029, 0.0006029
4: 0.0026282, 0.0054191, 0.0026282, 0.0054191, -0.0027909, 0.0027909
5: 0.0050201, 0.0064446, 0.0050201, 0.0064446, -0.0014245, 0.0014245
6: -0.0021963, -0.0008150, -0.0021963, -0.0008150, -0.0013813, 0.0013813
7: -0.0082640, -0.0074494, -0.0082640, -0.0074494, -0.0008146, 0.0008146
8: 0.0052524, 0.0098919, 0.0052524, 0.0098919, -0.0045656, 0.0045640
9: -0.0036845, -0.0030730, -0.0036845, -0.0030730, -0.0006115, 0.0006115

### Unstable ReLU Count (Linear/Conv2D Layers)
- layer_idx=0, type=LayerType.Linear, total=256, inp1_unstable=10, inp2_unstable=10, delta_unstable=251
- layer_idx=2, type=LayerType.Linear, total=256, inp1_unstable=8, inp2_unstable=8, delta_unstable=256
- layer_idx=4, type=LayerType.Linear, total=256, inp1_unstable=0, inp2_unstable=0, delta_unstable=256
- layer_idx=6, type=LayerType.Linear, total=256, inp1_unstable=2, inp2_unstable=2, delta_unstable=256
- layer_idx=8, type=LayerType.Linear, total=10, inp1_unstable=0, inp2_unstable=0, delta_unstable=10

Time for backsubstitution: 1.75 seconds

### RS candidates at layer 1
type: RSZ, layer: 1, pos: 176
type: RSZ, layer: 1, pos: 198
type: RSZ, layer: 1, pos: 199
type: RSZ, layer: 1, pos: 204
type: RSZ, layer: 1, pos: 250

Time for candidate selection: 0.18 seconds

### Candidate
type: RSZ, layer: 1, pos: 176

### Relational analysis RSZ of RS_RSZ1_RSZ2_RSZ2_RSZ2_RSZ2_RSZ1

#### Relational analysis RSZ result of RS_RSZ1_RSZ2_RSZ2_RSZ2_RSZ2_RSZ1
Status: Status.UNKNOWN
Output dim: 1, lower bound: -0.0008374, upper bound: 0.0008374
time: 0.67 seconds

### Relational analysis RSZ of RS_RSZ1_RSZ2_RSZ2_RSZ2_RSZ2_RSZ2

#### Relational analysis RSZ result of RS_RSZ1_RSZ2_RSZ2_RSZ2_RSZ2_RSZ2
Status: Status.UNKNOWN
Output dim: 1, lower bound: -0.0008374, upper bound: 0.0008374
time: 0.65 seconds

## BFS RS instance: RS_RSZ2_RSZ1_RSZ1_RSZ1_RSZ1

### Backsubstitution after applying RS history:
0: 0.0006996, 0.0011177, 0.0006996, 0.0011177, -0.0004181, 0.0004181
1: 0.9934013, 0.9943976, 0.9934013, 0.9943976, -0.0009964, 0.0009964
2: -0.0088595, -0.0053283, -0.0088595, -0.0053283, -0.0032545, 0.0032485
3: 0.0035633, 0.0041662, 0.0035633, 0.0041662, -0.0006029, 0.0006029
4: 0.0026282, 0.0054191, 0.0026282, 0.0054191, -0.0027909, 0.0027909
5: 0.0050201, 0.0064446, 0.0050201, 0.0064446, -0.0014245, 0.0014245
6: -0.0021963, -0.0008150, -0.0021963, -0.0008150, -0.0013813, 0.0013813
7: -0.0082640, -0.0074494, -0.0082640, -0.0074494, -0.0008146, 0.0008146
8: 0.0052524, 0.0098919, 0.0052524, 0.0098919, -0.0045640, 0.0045656
9: -0.0036845, -0.0030730, -0.0036845, -0.0030730, -0.0006115, 0.0006115

### Unstable ReLU Count (Linear/Conv2D Layers)
- layer_idx=0, type=LayerType.Linear, total=256, inp1_unstable=10, inp2_unstable=10, delta_unstable=251
- layer_idx=2, type=LayerType.Linear, total=256, inp1_unstable=8, inp2_unstable=8, delta_unstable=256
- layer_idx=4, type=LayerType.Linear, total=256, inp1_unstable=0, inp2_unstable=0, delta_unstable=256
- layer_idx=6, type=LayerType.Linear, total=256, inp1_unstable=2, inp2_unstable=2, delta_unstable=256
- layer_idx=8, type=LayerType.Linear, total=10, inp1_unstable=0, inp2_unstable=0, delta_unstable=10

Time for backsubstitution: 1.64 seconds

### RS candidates at layer 1
type: RSZ, layer: 1, pos: 176
type: RSZ, layer: 1, pos: 198
type: RSZ, layer: 1, pos: 199
type: RSZ, layer: 1, pos: 204
type: RSZ, layer: 1, pos: 250

Time for candidate selection: 0.19 seconds

### Candidate
type: RSZ, layer: 1, pos: 176

### Relational analysis RSZ of RS_RSZ2_RSZ1_RSZ1_RSZ1_RSZ1_RSZ1

#### Relational analysis RSZ result of RS_RSZ2_RSZ1_RSZ1_RSZ1_RSZ1_RSZ1
Status: Status.UNKNOWN
Output dim: 1, lower bound: -0.0008374, upper bound: 0.0008374
time: 0.71 seconds

### Relational analysis RSZ of RS_RSZ2_RSZ1_RSZ1_RSZ1_RSZ1_RSZ2

#### Relational analysis RSZ result of RS_RSZ2_RSZ1_RSZ1_RSZ1_RSZ1_RSZ2
Status: Status.UNKNOWN
Output dim: 1, lower bound: -0.0008374, upper bound: 0.0008374
time: 0.74 seconds

## BFS RS instance: RS_RSZ2_RSZ1_RSZ1_RSZ1_RSZ2

### Backsubstitution after applying RS history:
0: 0.0006996, 0.0011177, 0.0006996, 0.0011177, -0.0004181, 0.0004181
1: 0.9934013, 0.9943976, 0.9934013, 0.9943976, -0.0009964, 0.0009964
2: -0.0088595, -0.0053283, -0.0088595, -0.0053283, -0.0032529, 0.0032505
3: 0.0035633, 0.0041662, 0.0035633, 0.0041662, -0.0006029, 0.0006029
4: 0.0026282, 0.0054191, 0.0026282, 0.0054191, -0.0027909, 0.0027909
5: 0.0050201, 0.0064446, 0.0050201, 0.0064446, -0.0014245, 0.0014245
6: -0.0021963, -0.0008150, -0.0021963, -0.0008150, -0.0013813, 0.0013813
7: -0.0082640, -0.0074494, -0.0082640, -0.0074494, -0.0008146, 0.0008146
8: 0.0052524, 0.0098919, 0.0052524, 0.0098919, -0.0045646, 0.0045652
9: -0.0036845, -0.0030730, -0.0036845, -0.0030730, -0.0006115, 0.0006115

### Unstable ReLU Count (Linear/Conv2D Layers)
- layer_idx=0, type=LayerType.Linear, total=256, inp1_unstable=10, inp2_unstable=10, delta_unstable=251
- layer_idx=2, type=LayerType.Linear, total=256, inp1_unstable=8, inp2_unstable=8, delta_unstable=256
- layer_idx=4, type=LayerType.Linear, total=256, inp1_unstable=0, inp2_unstable=0, delta_unstable=256
- layer_idx=6, type=LayerType.Linear, total=256, inp1_unstable=2, inp2_unstable=2, delta_unstable=256
- layer_idx=8, type=LayerType.Linear, total=10, inp1_unstable=0, inp2_unstable=0, delta_unstable=10

Time for backsubstitution: 1.79 seconds

### RS candidates at layer 1
type: RSZ, layer: 1, pos: 176
type: RSZ, layer: 1, pos: 198
type: RSZ, layer: 1, pos: 199
type: RSZ, layer: 1, pos: 204
type: RSZ, layer: 1, pos: 250

Time for candidate selection: 0.18 seconds

### Candidate
type: RSZ, layer: 1, pos: 176

### Relational analysis RSZ of RS_RSZ2_RSZ1_RSZ1_RSZ1_RSZ2_RSZ1

#### Relational analysis RSZ result of RS_RSZ2_RSZ1_RSZ1_RSZ1_RSZ2_RSZ1
Status: Status.UNKNOWN
Output dim: 1, lower bound: -0.0008374, upper bound: 0.0008374
time: 0.73 seconds

### Relational analysis RSZ of RS_RSZ2_RSZ1_RSZ1_RSZ1_RSZ2_RSZ2

#### Relational analysis RSZ result of RS_RSZ2_RSZ1_RSZ1_RSZ1_RSZ2_RSZ2
Status: Status.UNKNOWN
Output dim: 1, lower bound: -0.0008374, upper bound: 0.0008374
time: 0.68 seconds

## BFS RS instance: RS_RSZ2_RSZ1_RSZ1_RSZ2_RSZ1

### Backsubstitution after applying RS history:
0: 0.0006996, 0.0011177, 0.0006996, 0.0011177, -0.0004181, 0.0004181
1: 0.9934013, 0.9943976, 0.9934013, 0.9943976, -0.0009964, 0.0009964
2: -0.0088595, -0.0053283, -0.0088595, -0.0053283, -0.0032537, 0.0032499
3: 0.0035633, 0.0041662, 0.0035633, 0.0041662, -0.0006029, 0.0006029
4: 0.0026282, 0.0054191, 0.0026282, 0.0054191, -0.0027909, 0.0027909
5: 0.0050201, 0.0064446, 0.0050201, 0.0064446, -0.0014245, 0.0014245
6: -0.0021963, -0.0008150, -0.0021963, -0.0008150, -0.0013813, 0.0013813
7: -0.0082640, -0.0074494, -0.0082640, -0.0074494, -0.0008146, 0.0008146
8: 0.0052524, 0.0098919, 0.0052524, 0.0098919, -0.0045644, 0.0045654
9: -0.0036845, -0.0030730, -0.0036845, -0.0030730, -0.0006115, 0.0006115

### Unstable ReLU Count (Linear/Conv2D Layers)
- layer_idx=0, type=LayerType.Linear, total=256, inp1_unstable=10, inp2_unstable=10, delta_unstable=251
- layer_idx=2, type=LayerType.Linear, total=256, inp1_unstable=8, inp2_unstable=8, delta_unstable=256
- layer_idx=4, type=LayerType.Linear, total=256, inp1_unstable=0, inp2_unstable=0, delta_unstable=256
- layer_idx=6, type=LayerType.Linear, total=256, inp1_unstable=2, inp2_unstable=2, delta_unstable=256
- layer_idx=8, type=LayerType.Linear, total=10, inp1_unstable=0, inp2_unstable=0, delta_unstable=10

Time for backsubstitution: 1.71 seconds

### RS candidates at layer 1
type: RSZ, layer: 1, pos: 176
type: RSZ, layer: 1, pos: 198
type: RSZ, layer: 1, pos: 199
type: RSZ, layer: 1, pos: 204
type: RSZ, layer: 1, pos: 250

Time for candidate selection: 0.18 seconds

### Candidate
type: RSZ, layer: 1, pos: 176

### Relational analysis RSZ of RS_RSZ2_RSZ1_RSZ1_RSZ2_RSZ1_RSZ1

#### Relational analysis RSZ result of RS_RSZ2_RSZ1_RSZ1_RSZ2_RSZ1_RSZ1
Status: Status.UNKNOWN
Output dim: 1, lower bound: -0.0008374, upper bound: 0.0008374
time: 0.64 seconds

### Relational analysis RSZ of RS_RSZ2_RSZ1_RSZ1_RSZ2_RSZ1_RSZ2

#### Relational analysis RSZ result of RS_RSZ2_RSZ1_RSZ1_RSZ2_RSZ1_RSZ2
Status: Status.UNKNOWN
Output dim: 1, lower bound: -0.0008374, upper bound: 0.0008374
time: 0.73 seconds

## BFS RS instance: RS_RSZ2_RSZ1_RSZ1_RSZ2_RSZ2

### Backsubstitution after applying RS history:
0: 0.0006996, 0.0011177, 0.0006996, 0.0011177, -0.0004181, 0.0004181
1: 0.9934013, 0.9943976, 0.9934013, 0.9943976, -0.0009964, 0.0009964
2: -0.0088595, -0.0053283, -0.0088595, -0.0053283, -0.0032519, 0.0032519
3: 0.0035633, 0.0041662, 0.0035633, 0.0041662, -0.0006029, 0.0006029
4: 0.0026282, 0.0054191, 0.0026282, 0.0054191, -0.0027909, 0.0027909
5: 0.0050201, 0.0064446, 0.0050201, 0.0064446, -0.0014245, 0.0014245
6: -0.0021963, -0.0008150, -0.0021963, -0.0008150, -0.0013813, 0.0013813
7: -0.0082640, -0.0074494, -0.0082640, -0.0074494, -0.0008146, 0.0008146
8: 0.0052524, 0.0098919, 0.0052524, 0.0098919, -0.0045649, 0.0045649
9: -0.0036845, -0.0030730, -0.0036845, -0.0030730, -0.0006115, 0.0006115

### Unstable ReLU Count (Linear/Conv2D Layers)
- layer_idx=0, type=LayerType.Linear, total=256, inp1_unstable=10, inp2_unstable=10, delta_unstable=251
- layer_idx=2, type=LayerType.Linear, total=256, inp1_unstable=8, inp2_unstable=8, delta_unstable=256
- layer_idx=4, type=LayerType.Linear, total=256, inp1_unstable=0, inp2_unstable=0, delta_unstable=256
- layer_idx=6, type=LayerType.Linear, total=256, inp1_unstable=2, inp2_unstable=2, delta_unstable=256
- layer_idx=8, type=LayerType.Linear, total=10, inp1_unstable=0, inp2_unstable=0, delta_unstable=10

Time for backsubstitution: 1.65 seconds

### RS candidates at layer 1
type: RSZ, layer: 1, pos: 176
type: RSZ, layer: 1, pos: 198
type: RSZ, layer: 1, pos: 199
type: RSZ, layer: 1, pos: 204
type: RSZ, layer: 1, pos: 250

Time for candidate selection: 0.18 seconds

### Candidate
type: RSZ, layer: 1, pos: 176

### Relational analysis RSZ of RS_RSZ2_RSZ1_RSZ1_RSZ2_RSZ2_RSZ1

#### Relational analysis RSZ result of RS_RSZ2_RSZ1_RSZ1_RSZ2_RSZ2_RSZ1
Status: Status.UNKNOWN
Output dim: 1, lower bound: -0.0008374, upper bound: 0.0008374
time: 0.69 seconds

### Relational analysis RSZ of RS_RSZ2_RSZ1_RSZ1_RSZ2_RSZ2_RSZ2

#### Relational analysis RSZ result of RS_RSZ2_RSZ1_RSZ1_RSZ2_RSZ2_RSZ2
Status: Status.UNKNOWN
Output dim: 1, lower bound: -0.0008374, upper bound: 0.0008374
time: 0.66 seconds

## BFS RS instance: RS_RSZ2_RSZ1_RSZ2_RSZ1_RSZ1

### Backsubstitution after applying RS history:
0: 0.0006996, 0.0011177, 0.0006996, 0.0011177, -0.0004181, 0.0004181
1: 0.9934013, 0.9943976, 0.9934013, 0.9943976, -0.0009964, 0.0009964
2: -0.0088595, -0.0053283, -0.0088595, -0.0053283, -0.0032542, 0.0032486
3: 0.0035633, 0.0041662, 0.0035633, 0.0041662, -0.0006029, 0.0006029
4: 0.0026282, 0.0054191, 0.0026282, 0.0054191, -0.0027909, 0.0027909
5: 0.0050201, 0.0064446, 0.0050201, 0.0064446, -0.0014245, 0.0014245
6: -0.0021963, -0.0008150, -0.0021963, -0.0008150, -0.0013813, 0.0013813
7: -0.0082640, -0.0074494, -0.0082640, -0.0074494, -0.0008146, 0.0008146
8: 0.0052524, 0.0098919, 0.0052524, 0.0098919, -0.0045640, 0.0045655
9: -0.0036845, -0.0030730, -0.0036845, -0.0030730, -0.0006115, 0.0006115

### Unstable ReLU Count (Linear/Conv2D Layers)
- layer_idx=0, type=LayerType.Linear, total=256, inp1_unstable=10, inp2_unstable=10, delta_unstable=251
- layer_idx=2, type=LayerType.Linear, total=256, inp1_unstable=8, inp2_unstable=8, delta_unstable=256
- layer_idx=4, type=LayerType.Linear, total=256, inp1_unstable=0, inp2_unstable=0, delta_unstable=256
- layer_idx=6, type=LayerType.Linear, total=256, inp1_unstable=2, inp2_unstable=2, delta_unstable=256
- layer_idx=8, type=LayerType.Linear, total=10, inp1_unstable=0, inp2_unstable=0, delta_unstable=10

Time for backsubstitution: 1.62 seconds

### RS candidates at layer 1
type: RSZ, layer: 1, pos: 176
type: RSZ, layer: 1, pos: 198
type: RSZ, layer: 1, pos: 199
type: RSZ, layer: 1, pos: 204
type: RSZ, layer: 1, pos: 250

Time for candidate selection: 0.17 seconds

### Candidate
type: RSZ, layer: 1, pos: 176

### Relational analysis RSZ of RS_RSZ2_RSZ1_RSZ2_RSZ1_RSZ1_RSZ1

#### Relational analysis RSZ result of RS_RSZ2_RSZ1_RSZ2_RSZ1_RSZ1_RSZ1
Status: Status.UNKNOWN
Output dim: 1, lower bound: -0.0008374, upper bound: 0.0008374
time: 0.72 seconds

### Relational analysis RSZ of RS_RSZ2_RSZ1_RSZ2_RSZ1_RSZ1_RSZ2

#### Relational analysis RSZ result of RS_RSZ2_RSZ1_RSZ2_RSZ1_RSZ1_RSZ2
Status: Status.UNKNOWN
Output dim: 1, lower bound: -0.0008374, upper bound: 0.0008374
time: 0.69 seconds

## BFS RS instance: RS_RSZ2_RSZ1_RSZ2_RSZ1_RSZ2

### Backsubstitution after applying RS history:
0: 0.0006996, 0.0011177, 0.0006996, 0.0011177, -0.0004181, 0.0004181
1: 0.9934013, 0.9943976, 0.9934013, 0.9943976, -0.0009964, 0.0009964
2: -0.0088595, -0.0053283, -0.0088595, -0.0053283, -0.0032528, 0.0032507
3: 0.0035633, 0.0041662, 0.0035633, 0.0041662, -0.0006029, 0.0006029
4: 0.0026282, 0.0054191, 0.0026282, 0.0054191, -0.0027909, 0.0027909
5: 0.0050201, 0.0064446, 0.0050201, 0.0064446, -0.0014245, 0.0014245
6: -0.0021963, -0.0008150, -0.0021963, -0.0008150, -0.0013813, 0.0013813
7: -0.0082640, -0.0074494, -0.0082640, -0.0074494, -0.0008146, 0.0008146
8: 0.0052524, 0.0098919, 0.0052524, 0.0098919, -0.0045646, 0.0045652
9: -0.0036845, -0.0030730, -0.0036845, -0.0030730, -0.0006115, 0.0006115

### Unstable ReLU Count (Linear/Conv2D Layers)
- layer_idx=0, type=LayerType.Linear, total=256, inp1_unstable=10, inp2_unstable=10, delta_unstable=251
- layer_idx=2, type=LayerType.Linear, total=256, inp1_unstable=8, inp2_unstable=8, delta_unstable=256
- layer_idx=4, type=LayerType.Linear, total=256, inp1_unstable=0, inp2_unstable=0, delta_unstable=256
- layer_idx=6, type=LayerType.Linear, total=256, inp1_unstable=2, inp2_unstable=2, delta_unstable=256
- layer_idx=8, type=LayerType.Linear, total=10, inp1_unstable=0, inp2_unstable=0, delta_unstable=10

Time for backsubstitution: 1.69 seconds

### RS candidates at layer 1
type: RSZ, layer: 1, pos: 176
type: RSZ, layer: 1, pos: 198
type: RSZ, layer: 1, pos: 199
type: RSZ, layer: 1, pos: 204
type: RSZ, layer: 1, pos: 250

Time for candidate selection: 0.19 seconds

### Candidate
type: RSZ, layer: 1, pos: 176

### Relational analysis RSZ of RS_RSZ2_RSZ1_RSZ2_RSZ1_RSZ2_RSZ1

#### Relational analysis RSZ result of RS_RSZ2_RSZ1_RSZ2_RSZ1_RSZ2_RSZ1
Status: Status.UNKNOWN
Output dim: 1, lower bound: -0.0008374, upper bound: 0.0008374
time: 0.64 seconds

### Relational analysis RSZ of RS_RSZ2_RSZ1_RSZ2_RSZ1_RSZ2_RSZ2

#### Relational analysis RSZ result of RS_RSZ2_RSZ1_RSZ2_RSZ1_RSZ2_RSZ2
Status: Status.UNKNOWN
Output dim: 1, lower bound: -0.0008374, upper bound: 0.0008374
time: 0.69 seconds

## BFS RS instance: RS_RSZ2_RSZ1_RSZ2_RSZ2_RSZ1

### Backsubstitution after applying RS history:
0: 0.0006996, 0.0011177, 0.0006996, 0.0011177, -0.0004181, 0.0004181
1: 0.9934013, 0.9943976, 0.9934013, 0.9943976, -0.0009964, 0.0009964
2: -0.0088595, -0.0053283, -0.0088595, -0.0053283, -0.0032535, 0.0032499
3: 0.0035633, 0.0041662, 0.0035633, 0.0041662, -0.0006029, 0.0006029
4: 0.0026282, 0.0054191, 0.0026282, 0.0054191, -0.0027909, 0.0027909
5: 0.0050201, 0.0064446, 0.0050201, 0.0064446, -0.0014245, 0.0014245
6: -0.0021963, -0.0008150, -0.0021963, -0.0008150, -0.0013813, 0.0013813
7: -0.0082640, -0.0074494, -0.0082640, -0.0074494, -0.0008146, 0.0008146
8: 0.0052524, 0.0098919, 0.0052524, 0.0098919, -0.0045644, 0.0045654
9: -0.0036845, -0.0030730, -0.0036845, -0.0030730, -0.0006115, 0.0006115

### Unstable ReLU Count (Linear/Conv2D Layers)
- layer_idx=0, type=LayerType.Linear, total=256, inp1_unstable=10, inp2_unstable=10, delta_unstable=251
- layer_idx=2, type=LayerType.Linear, total=256, inp1_unstable=8, inp2_unstable=8, delta_unstable=256
- layer_idx=4, type=LayerType.Linear, total=256, inp1_unstable=0, inp2_unstable=0, delta_unstable=256
- layer_idx=6, type=LayerType.Linear, total=256, inp1_unstable=2, inp2_unstable=2, delta_unstable=256
- layer_idx=8, type=LayerType.Linear, total=10, inp1_unstable=0, inp2_unstable=0, delta_unstable=10

Time for backsubstitution: 1.80 seconds

### RS candidates at layer 1
type: RSZ, layer: 1, pos: 176
type: RSZ, layer: 1, pos: 198
type: RSZ, layer: 1, pos: 199
type: RSZ, layer: 1, pos: 204
type: RSZ, layer: 1, pos: 250

Time for candidate selection: 0.19 seconds

### Candidate
type: RSZ, layer: 1, pos: 176

### Relational analysis RSZ of RS_RSZ2_RSZ1_RSZ2_RSZ2_RSZ1_RSZ1

#### Relational analysis RSZ result of RS_RSZ2_RSZ1_RSZ2_RSZ2_RSZ1_RSZ1
Status: Status.UNKNOWN
Output dim: 1, lower bound: -0.0008374, upper bound: 0.0008374
time: 0.70 seconds

### Relational analysis RSZ of RS_RSZ2_RSZ1_RSZ2_RSZ2_RSZ1_RSZ2

#### Relational analysis RSZ result of RS_RSZ2_RSZ1_RSZ2_RSZ2_RSZ1_RSZ2
Status: Status.UNKNOWN
Output dim: 1, lower bound: -0.0008374, upper bound: 0.0008374
time: 0.72 seconds

## BFS RS instance: RS_RSZ2_RSZ1_RSZ2_RSZ2_RSZ2

### Backsubstitution after applying RS history:
0: 0.0006996, 0.0011177, 0.0006996, 0.0011177, -0.0004181, 0.0004181
1: 0.9934013, 0.9943976, 0.9934013, 0.9943976, -0.0009964, 0.0009964
2: -0.0088595, -0.0053283, -0.0088595, -0.0053283, -0.0032515, 0.0032520
3: 0.0035633, 0.0041662, 0.0035633, 0.0041662, -0.0006029, 0.0006029
4: 0.0026282, 0.0054191, 0.0026282, 0.0054191, -0.0027909, 0.0027909
5: 0.0050201, 0.0064446, 0.0050201, 0.0064446, -0.0014245, 0.0014245
6: -0.0021963, -0.0008150, -0.0021963, -0.0008150, -0.0013813, 0.0013813
7: -0.0082640, -0.0074494, -0.0082640, -0.0074494, -0.0008146, 0.0008146
8: 0.0052524, 0.0098919, 0.0052524, 0.0098919, -0.0045650, 0.0045648
9: -0.0036845, -0.0030730, -0.0036845, -0.0030730, -0.0006115, 0.0006115

### Unstable ReLU Count (Linear/Conv2D Layers)
- layer_idx=0, type=LayerType.Linear, total=256, inp1_unstable=10, inp2_unstable=10, delta_unstable=251
- layer_idx=2, type=LayerType.Linear, total=256, inp1_unstable=8, inp2_unstable=8, delta_unstable=256
- layer_idx=4, type=LayerType.Linear, total=256, inp1_unstable=0, inp2_unstable=0, delta_unstable=256
- layer_idx=6, type=LayerType.Linear, total=256, inp1_unstable=2, inp2_unstable=2, delta_unstable=256
- layer_idx=8, type=LayerType.Linear, total=10, inp1_unstable=0, inp2_unstable=0, delta_unstable=10

Time for backsubstitution: 1.65 seconds

### RS candidates at layer 1
type: RSZ, layer: 1, pos: 176
type: RSZ, layer: 1, pos: 198
type: RSZ, layer: 1, pos: 199
type: RSZ, layer: 1, pos: 204
type: RSZ, layer: 1, pos: 250

Time for candidate selection: 0.18 seconds

### Candidate
type: RSZ, layer: 1, pos: 176

### Relational analysis RSZ of RS_RSZ2_RSZ1_RSZ2_RSZ2_RSZ2_RSZ1

#### Relational analysis RSZ result of RS_RSZ2_RSZ1_RSZ2_RSZ2_RSZ2_RSZ1
Status: Status.UNKNOWN
Output dim: 1, lower bound: -0.0008374, upper bound: 0.0008374
time: 0.63 seconds

### Relational analysis RSZ of RS_RSZ2_RSZ1_RSZ2_RSZ2_RSZ2_RSZ2

#### Relational analysis RSZ result of RS_RSZ2_RSZ1_RSZ2_RSZ2_RSZ2_RSZ2
Status: Status.UNKNOWN
Output dim: 1, lower bound: -0.0008374, upper bound: 0.0008374
time: 0.69 seconds

## BFS RS instance: RS_RSZ2_RSZ2_RSZ1_RSZ1_RSZ1

### Backsubstitution after applying RS history:
0: 0.0006996, 0.0011177, 0.0006996, 0.0011177, -0.0004181, 0.0004181
1: 0.9934013, 0.9943976, 0.9934013, 0.9943976, -0.0009964, 0.0009964
2: -0.0088595, -0.0053283, -0.0088595, -0.0053283, -0.0032518, 0.0032510
3: 0.0035633, 0.0041662, 0.0035633, 0.0041662, -0.0006029, 0.0006029
4: 0.0026282, 0.0054191, 0.0026282, 0.0054191, -0.0027909, 0.0027909
5: 0.0050201, 0.0064446, 0.0050201, 0.0064446, -0.0014245, 0.0014245
6: -0.0021963, -0.0008150, -0.0021963, -0.0008150, -0.0013813, 0.0013813
7: -0.0082640, -0.0074494, -0.0082640, -0.0074494, -0.0008146, 0.0008146
8: 0.0052524, 0.0098919, 0.0052524, 0.0098919, -0.0045647, 0.0045649
9: -0.0036845, -0.0030730, -0.0036845, -0.0030730, -0.0006115, 0.0006115

### Unstable ReLU Count (Linear/Conv2D Layers)
- layer_idx=0, type=LayerType.Linear, total=256, inp1_unstable=10, inp2_unstable=10, delta_unstable=251
- layer_idx=2, type=LayerType.Linear, total=256, inp1_unstable=8, inp2_unstable=8, delta_unstable=256
- layer_idx=4, type=LayerType.Linear, total=256, inp1_unstable=0, inp2_unstable=0, delta_unstable=256
- layer_idx=6, type=LayerType.Linear, total=256, inp1_unstable=2, inp2_unstable=2, delta_unstable=256
- layer_idx=8, type=LayerType.Linear, total=10, inp1_unstable=0, inp2_unstable=0, delta_unstable=10

Time for backsubstitution: 1.71 seconds

### RS candidates at layer 1
type: RSZ, layer: 1, pos: 176
type: RSZ, layer: 1, pos: 198
type: RSZ, layer: 1, pos: 199
type: RSZ, layer: 1, pos: 204
type: RSZ, layer: 1, pos: 250

Time for candidate selection: 0.18 seconds

### Candidate
type: RSZ, layer: 1, pos: 176

### Relational analysis RSZ of RS_RSZ2_RSZ2_RSZ1_RSZ1_RSZ1_RSZ1

#### Relational analysis RSZ result of RS_RSZ2_RSZ2_RSZ1_RSZ1_RSZ1_RSZ1
Status: Status.UNKNOWN
Output dim: 1, lower bound: -0.0008374, upper bound: 0.0008374
time: 0.67 seconds

### Relational analysis RSZ of RS_RSZ2_RSZ2_RSZ1_RSZ1_RSZ1_RSZ2

#### Relational analysis RSZ result of RS_RSZ2_RSZ2_RSZ1_RSZ1_RSZ1_RSZ2
Status: Status.UNKNOWN
Output dim: 1, lower bound: -0.0008374, upper bound: 0.0008374
time: 0.92 seconds

## BFS RS instance: RS_RSZ2_RSZ2_RSZ1_RSZ1_RSZ2

### Backsubstitution after applying RS history:
0: 0.0006996, 0.0011177, 0.0006996, 0.0011177, -0.0004181, 0.0004181
1: 0.9934013, 0.9943976, 0.9934013, 0.9943976, -0.0009964, 0.0009964
2: -0.0088595, -0.0053283, -0.0088595, -0.0053283, -0.0032499, 0.0032531
3: 0.0035633, 0.0041662, 0.0035633, 0.0041662, -0.0006029, 0.0006029
4: 0.0026282, 0.0054191, 0.0026282, 0.0054191, -0.0027909, 0.0027909
5: 0.0050201, 0.0064446, 0.0050201, 0.0064446, -0.0014245, 0.0014245
6: -0.0021963, -0.0008150, -0.0021963, -0.0008150, -0.0013813, 0.0013813
7: -0.0082640, -0.0074494, -0.0082640, -0.0074494, -0.0008146, 0.0008146
8: 0.0052524, 0.0098919, 0.0052524, 0.0098919, -0.0045652, 0.0045644
9: -0.0036845, -0.0030730, -0.0036845, -0.0030730, -0.0006115, 0.0006115

### Unstable ReLU Count (Linear/Conv2D Layers)
- layer_idx=0, type=LayerType.Linear, total=256, inp1_unstable=10, inp2_unstable=10, delta_unstable=251
- layer_idx=2, type=LayerType.Linear, total=256, inp1_unstable=8, inp2_unstable=8, delta_unstable=256
- layer_idx=4, type=LayerType.Linear, total=256, inp1_unstable=0, inp2_unstable=0, delta_unstable=256
- layer_idx=6, type=LayerType.Linear, total=256, inp1_unstable=2, inp2_unstable=2, delta_unstable=256
- layer_idx=8, type=LayerType.Linear, total=10, inp1_unstable=0, inp2_unstable=0, delta_unstable=10

Time for backsubstitution: 1.76 seconds

### RS candidates at layer 1
type: RSZ, layer: 1, pos: 176
type: RSZ, layer: 1, pos: 198
type: RSZ, layer: 1, pos: 199
type: RSZ, layer: 1, pos: 204
type: RSZ, layer: 1, pos: 250

Time for candidate selection: 0.19 seconds

### Candidate
type: RSZ, layer: 1, pos: 176

### Relational analysis RSZ of RS_RSZ2_RSZ2_RSZ1_RSZ1_RSZ2_RSZ1

#### Relational analysis RSZ result of RS_RSZ2_RSZ2_RSZ1_RSZ1_RSZ2_RSZ1
Status: Status.UNKNOWN
Output dim: 1, lower bound: -0.0008374, upper bound: 0.0008374
time: 0.67 seconds

### Relational analysis RSZ of RS_RSZ2_RSZ2_RSZ1_RSZ1_RSZ2_RSZ2

#### Relational analysis RSZ result of RS_RSZ2_RSZ2_RSZ1_RSZ1_RSZ2_RSZ2
Status: Status.UNKNOWN
Output dim: 1, lower bound: -0.0008374, upper bound: 0.0008374
time: 0.68 seconds

## BFS RS instance: RS_RSZ2_RSZ2_RSZ1_RSZ2_RSZ1

### Backsubstitution after applying RS history:
0: 0.0006996, 0.0011177, 0.0006996, 0.0011177, -0.0004181, 0.0004181
1: 0.9934013, 0.9943976, 0.9934013, 0.9943976, -0.0009964, 0.0009964
2: -0.0088595, -0.0053283, -0.0088595, -0.0053283, -0.0032508, 0.0032523
3: 0.0035633, 0.0041662, 0.0035633, 0.0041662, -0.0006029, 0.0006029
4: 0.0026282, 0.0054191, 0.0026282, 0.0054191, -0.0027909, 0.0027909
5: 0.0050201, 0.0064446, 0.0050201, 0.0064446, -0.0014245, 0.0014245
6: -0.0021963, -0.0008150, -0.0021963, -0.0008150, -0.0013813, 0.0013813
7: -0.0082640, -0.0074494, -0.0082640, -0.0074494, -0.0008146, 0.0008146
8: 0.0052524, 0.0098919, 0.0052524, 0.0098919, -0.0045650, 0.0045646
9: -0.0036845, -0.0030730, -0.0036845, -0.0030730, -0.0006115, 0.0006115

### Unstable ReLU Count (Linear/Conv2D Layers)
- layer_idx=0, type=LayerType.Linear, total=256, inp1_unstable=10, inp2_unstable=10, delta_unstable=251
- layer_idx=2, type=LayerType.Linear, total=256, inp1_unstable=8, inp2_unstable=8, delta_unstable=256
- layer_idx=4, type=LayerType.Linear, total=256, inp1_unstable=0, inp2_unstable=0, delta_unstable=256
- layer_idx=6, type=LayerType.Linear, total=256, inp1_unstable=2, inp2_unstable=2, delta_unstable=256
- layer_idx=8, type=LayerType.Linear, total=10, inp1_unstable=0, inp2_unstable=0, delta_unstable=10

Time for backsubstitution: 1.77 seconds

### RS candidates at layer 1
type: RSZ, layer: 1, pos: 176
type: RSZ, layer: 1, pos: 198
type: RSZ, layer: 1, pos: 199
type: RSZ, layer: 1, pos: 204
type: RSZ, layer: 1, pos: 250

Time for candidate selection: 0.19 seconds

### Candidate
type: RSZ, layer: 1, pos: 176

### Relational analysis RSZ of RS_RSZ2_RSZ2_RSZ1_RSZ2_RSZ1_RSZ1

#### Relational analysis RSZ result of RS_RSZ2_RSZ2_RSZ1_RSZ2_RSZ1_RSZ1
Status: Status.UNKNOWN
Output dim: 1, lower bound: -0.0008374, upper bound: 0.0008374
time: 0.67 seconds

### Relational analysis RSZ of RS_RSZ2_RSZ2_RSZ1_RSZ2_RSZ1_RSZ2

#### Relational analysis RSZ result of RS_RSZ2_RSZ2_RSZ1_RSZ2_RSZ1_RSZ2
Status: Status.UNKNOWN
Output dim: 1, lower bound: -0.0008374, upper bound: 0.0008374
time: 0.69 seconds

## BFS RS instance: RS_RSZ2_RSZ2_RSZ1_RSZ2_RSZ2

### Backsubstitution after applying RS history:
0: 0.0006996, 0.0011177, 0.0006996, 0.0011177, -0.0004181, 0.0004181
1: 0.9934013, 0.9943976, 0.9934013, 0.9943976, -0.0009964, 0.0009964
2: -0.0088595, -0.0053283, -0.0088595, -0.0053283, -0.0032490, 0.0032539
3: 0.0035633, 0.0041662, 0.0035633, 0.0041662, -0.0006029, 0.0006029
4: 0.0026282, 0.0054191, 0.0026282, 0.0054191, -0.0027909, 0.0027909
5: 0.0050201, 0.0064446, 0.0050201, 0.0064446, -0.0014245, 0.0014245
6: -0.0021963, -0.0008150, -0.0021963, -0.0008150, -0.0013813, 0.0013813
7: -0.0082640, -0.0074494, -0.0082640, -0.0074494, -0.0008146, 0.0008146
8: 0.0052524, 0.0098919, 0.0052524, 0.0098919, -0.0045655, 0.0045642
9: -0.0036845, -0.0030730, -0.0036845, -0.0030730, -0.0006115, 0.0006115

### Unstable ReLU Count (Linear/Conv2D Layers)
- layer_idx=0, type=LayerType.Linear, total=256, inp1_unstable=10, inp2_unstable=10, delta_unstable=251
- layer_idx=2, type=LayerType.Linear, total=256, inp1_unstable=8, inp2_unstable=8, delta_unstable=256
- layer_idx=4, type=LayerType.Linear, total=256, inp1_unstable=0, inp2_unstable=0, delta_unstable=256
- layer_idx=6, type=LayerType.Linear, total=256, inp1_unstable=2, inp2_unstable=2, delta_unstable=256
- layer_idx=8, type=LayerType.Linear, total=10, inp1_unstable=0, inp2_unstable=0, delta_unstable=10

Time for backsubstitution: 1.76 seconds

### RS candidates at layer 1
type: RSZ, layer: 1, pos: 176
type: RSZ, layer: 1, pos: 198
type: RSZ, layer: 1, pos: 199
type: RSZ, layer: 1, pos: 204
type: RSZ, layer: 1, pos: 250

Time for candidate selection: 0.22 seconds

### Candidate
type: RSZ, layer: 1, pos: 176

### Relational analysis RSZ of RS_RSZ2_RSZ2_RSZ1_RSZ2_RSZ2_RSZ1

#### Relational analysis RSZ result of RS_RSZ2_RSZ2_RSZ1_RSZ2_RSZ2_RSZ1
Status: Status.UNKNOWN
Output dim: 1, lower bound: -0.0008374, upper bound: 0.0008374
time: 0.70 seconds

### Relational analysis RSZ of RS_RSZ2_RSZ2_RSZ1_RSZ2_RSZ2_RSZ2

#### Relational analysis RSZ result of RS_RSZ2_RSZ2_RSZ1_RSZ2_RSZ2_RSZ2
Status: Status.UNKNOWN
Output dim: 1, lower bound: -0.0008374, upper bound: 0.0008374
time: 0.66 seconds

## BFS RS instance: RS_RSZ2_RSZ2_RSZ2_RSZ1_RSZ1

### Backsubstitution after applying RS history:
0: 0.0006996, 0.0011177, 0.0006996, 0.0011177, -0.0004181, 0.0004181
1: 0.9934013, 0.9943976, 0.9934013, 0.9943976, -0.0009964, 0.0009964
2: -0.0088595, -0.0053283, -0.0088595, -0.0053283, -0.0032517, 0.0032515
3: 0.0035633, 0.0041662, 0.0035633, 0.0041662, -0.0006029, 0.0006029
4: 0.0026282, 0.0054191, 0.0026282, 0.0054191, -0.0027909, 0.0027909
5: 0.0050201, 0.0064446, 0.0050201, 0.0064446, -0.0014245, 0.0014245
6: -0.0021963, -0.0008150, -0.0021963, -0.0008150, -0.0013813, 0.0013813
7: -0.0082640, -0.0074494, -0.0082640, -0.0074494, -0.0008146, 0.0008146
8: 0.0052524, 0.0098919, 0.0052524, 0.0098919, -0.0045648, 0.0045649
9: -0.0036845, -0.0030730, -0.0036845, -0.0030730, -0.0006115, 0.0006115

### Unstable ReLU Count (Linear/Conv2D Layers)
- layer_idx=0, type=LayerType.Linear, total=256, inp1_unstable=10, inp2_unstable=10, delta_unstable=251
- layer_idx=2, type=LayerType.Linear, total=256, inp1_unstable=8, inp2_unstable=8, delta_unstable=256
- layer_idx=4, type=LayerType.Linear, total=256, inp1_unstable=0, inp2_unstable=0, delta_unstable=256
- layer_idx=6, type=LayerType.Linear, total=256, inp1_unstable=2, inp2_unstable=2, delta_unstable=256
- layer_idx=8, type=LayerType.Linear, total=10, inp1_unstable=0, inp2_unstable=0, delta_unstable=10

Time for backsubstitution: 1.69 seconds

### RS candidates at layer 1
type: RSZ, layer: 1, pos: 176
type: RSZ, layer: 1, pos: 198
type: RSZ, layer: 1, pos: 199
type: RSZ, layer: 1, pos: 204
type: RSZ, layer: 1, pos: 250

Time for candidate selection: 0.18 seconds

### Candidate
type: RSZ, layer: 1, pos: 176

### Relational analysis RSZ of RS_RSZ2_RSZ2_RSZ2_RSZ1_RSZ1_RSZ1

#### Relational analysis RSZ result of RS_RSZ2_RSZ2_RSZ2_RSZ1_RSZ1_RSZ1
Status: Status.UNKNOWN
Output dim: 1, lower bound: -0.0008374, upper bound: 0.0008374
time: 0.64 seconds

### Relational analysis RSZ of RS_RSZ2_RSZ2_RSZ2_RSZ1_RSZ1_RSZ2

#### Relational analysis RSZ result of RS_RSZ2_RSZ2_RSZ2_RSZ1_RSZ1_RSZ2
Status: Status.UNKNOWN
Output dim: 1, lower bound: -0.0008374, upper bound: 0.0008374
time: 0.65 seconds

## BFS RS instance: RS_RSZ2_RSZ2_RSZ2_RSZ1_RSZ2

### Backsubstitution after applying RS history:
0: 0.0006996, 0.0011177, 0.0006996, 0.0011177, -0.0004181, 0.0004181
1: 0.9934013, 0.9943976, 0.9934013, 0.9943976, -0.0009964, 0.0009964
2: -0.0088595, -0.0053283, -0.0088595, -0.0053283, -0.0032503, 0.0032535
3: 0.0035633, 0.0041662, 0.0035633, 0.0041662, -0.0006029, 0.0006029
4: 0.0026282, 0.0054191, 0.0026282, 0.0054191, -0.0027909, 0.0027909
5: 0.0050201, 0.0064446, 0.0050201, 0.0064446, -0.0014245, 0.0014245
6: -0.0021963, -0.0008150, -0.0021963, -0.0008150, -0.0013813, 0.0013813
7: -0.0082640, -0.0074494, -0.0082640, -0.0074494, -0.0008146, 0.0008146
8: 0.0052524, 0.0098919, 0.0052524, 0.0098919, -0.0045654, 0.0045645
9: -0.0036845, -0.0030730, -0.0036845, -0.0030730, -0.0006115, 0.0006115

### Unstable ReLU Count (Linear/Conv2D Layers)
- layer_idx=0, type=LayerType.Linear, total=256, inp1_unstable=10, inp2_unstable=10, delta_unstable=251
- layer_idx=2, type=LayerType.Linear, total=256, inp1_unstable=8, inp2_unstable=8, delta_unstable=256
- layer_idx=4, type=LayerType.Linear, total=256, inp1_unstable=0, inp2_unstable=0, delta_unstable=256
- layer_idx=6, type=LayerType.Linear, total=256, inp1_unstable=2, inp2_unstable=2, delta_unstable=256
- layer_idx=8, type=LayerType.Linear, total=10, inp1_unstable=0, inp2_unstable=0, delta_unstable=10

Time for backsubstitution: 1.72 seconds

### RS candidates at layer 1
type: RSZ, layer: 1, pos: 176
type: RSZ, layer: 1, pos: 198
type: RSZ, layer: 1, pos: 199
type: RSZ, layer: 1, pos: 204
type: RSZ, layer: 1, pos: 250

Time for candidate selection: 0.19 seconds

### Candidate
type: RSZ, layer: 1, pos: 176

### Relational analysis RSZ of RS_RSZ2_RSZ2_RSZ2_RSZ1_RSZ2_RSZ1

#### Relational analysis RSZ result of RS_RSZ2_RSZ2_RSZ2_RSZ1_RSZ2_RSZ1
Status: Status.UNKNOWN
Output dim: 1, lower bound: -0.0008374, upper bound: 0.0008374
time: 0.65 seconds

### Relational analysis RSZ of RS_RSZ2_RSZ2_RSZ2_RSZ1_RSZ2_RSZ2

#### Relational analysis RSZ result of RS_RSZ2_RSZ2_RSZ2_RSZ1_RSZ2_RSZ2
Status: Status.UNKNOWN
Output dim: 1, lower bound: -0.0008374, upper bound: 0.0008374
time: 0.66 seconds

## BFS RS instance: RS_RSZ2_RSZ2_RSZ2_RSZ2_RSZ1

### Backsubstitution after applying RS history:
0: 0.0006996, 0.0011177, 0.0006996, 0.0011177, -0.0004181, 0.0004181
1: 0.9934013, 0.9943976, 0.9934013, 0.9943976, -0.0009964, 0.0009964
2: -0.0088595, -0.0053283, -0.0088595, -0.0053283, -0.0032508, 0.0032524
3: 0.0035633, 0.0041662, 0.0035633, 0.0041662, -0.0006029, 0.0006029
4: 0.0026282, 0.0054191, 0.0026282, 0.0054191, -0.0027909, 0.0027909
5: 0.0050201, 0.0064446, 0.0050201, 0.0064446, -0.0014245, 0.0014245
6: -0.0021963, -0.0008150, -0.0021963, -0.0008150, -0.0013813, 0.0013813
7: -0.0082640, -0.0074494, -0.0082640, -0.0074494, -0.0008146, 0.0008146
8: 0.0052524, 0.0098919, 0.0052524, 0.0098919, -0.0045651, 0.0045646
9: -0.0036845, -0.0030730, -0.0036845, -0.0030730, -0.0006115, 0.0006115

### Unstable ReLU Count (Linear/Conv2D Layers)
- layer_idx=0, type=LayerType.Linear, total=256, inp1_unstable=10, inp2_unstable=10, delta_unstable=251
- layer_idx=2, type=LayerType.Linear, total=256, inp1_unstable=8, inp2_unstable=8, delta_unstable=256
- layer_idx=4, type=LayerType.Linear, total=256, inp1_unstable=0, inp2_unstable=0, delta_unstable=256
- layer_idx=6, type=LayerType.Linear, total=256, inp1_unstable=2, inp2_unstable=2, delta_unstable=256
- layer_idx=8, type=LayerType.Linear, total=10, inp1_unstable=0, inp2_unstable=0, delta_unstable=10

Time for backsubstitution: 1.59 seconds

### RS candidates at layer 1
type: RSZ, layer: 1, pos: 176
type: RSZ, layer: 1, pos: 198
type: RSZ, layer: 1, pos: 199
type: RSZ, layer: 1, pos: 204
type: RSZ, layer: 1, pos: 250

Time for candidate selection: 0.18 seconds

### Candidate
type: RSZ, layer: 1, pos: 176

### Relational analysis RSZ of RS_RSZ2_RSZ2_RSZ2_RSZ2_RSZ1_RSZ1

#### Relational analysis RSZ result of RS_RSZ2_RSZ2_RSZ2_RSZ2_RSZ1_RSZ1
Status: Status.UNKNOWN
Output dim: 1, lower bound: -0.0008374, upper bound: 0.0008374
time: 0.68 seconds

### Relational analysis RSZ of RS_RSZ2_RSZ2_RSZ2_RSZ2_RSZ1_RSZ2

#### Relational analysis RSZ result of RS_RSZ2_RSZ2_RSZ2_RSZ2_RSZ1_RSZ2
Status: Status.UNKNOWN
Output dim: 1, lower bound: -0.0008374, upper bound: 0.0008374
time: 0.66 seconds

## BFS RS instance: RS_RSZ2_RSZ2_RSZ2_RSZ2_RSZ2

### Backsubstitution after applying RS history:
0: 0.0006996, 0.0011177, 0.0006996, 0.0011177, -0.0004181, 0.0004181
1: 0.9934013, 0.9943976, 0.9934013, 0.9943976, -0.0009964, 0.0009964
2: -0.0088595, -0.0053283, -0.0088595, -0.0053283, -0.0032490, 0.0032543
3: 0.0035633, 0.0041662, 0.0035633, 0.0041662, -0.0006029, 0.0006029
4: 0.0026282, 0.0054191, 0.0026282, 0.0054191, -0.0027909, 0.0027909
5: 0.0050201, 0.0064446, 0.0050201, 0.0064446, -0.0014245, 0.0014245
6: -0.0021963, -0.0008150, -0.0021963, -0.0008150, -0.0013813, 0.0013813
7: -0.0082640, -0.0074494, -0.0082640, -0.0074494, -0.0008146, 0.0008146
8: 0.0052524, 0.0098919, 0.0052524, 0.0098919, -0.0045656, 0.0045642
9: -0.0036845, -0.0030730, -0.0036845, -0.0030730, -0.0006115, 0.0006115

### Unstable ReLU Count (Linear/Conv2D Layers)
- layer_idx=0, type=LayerType.Linear, total=256, inp1_unstable=10, inp2_unstable=10, delta_unstable=251
- layer_idx=2, type=LayerType.Linear, total=256, inp1_unstable=8, inp2_unstable=8, delta_unstable=256
- layer_idx=4, type=LayerType.Linear, total=256, inp1_unstable=0, inp2_unstable=0, delta_unstable=256
- layer_idx=6, type=LayerType.Linear, total=256, inp1_unstable=2, inp2_unstable=2, delta_unstable=256
- layer_idx=8, type=LayerType.Linear, total=10, inp1_unstable=0, inp2_unstable=0, delta_unstable=10

Time for backsubstitution: 1.62 seconds

### RS candidates at layer 1
type: RSZ, layer: 1, pos: 176
type: RSZ, layer: 1, pos: 198
type: RSZ, layer: 1, pos: 199
type: RSZ, layer: 1, pos: 204
type: RSZ, layer: 1, pos: 250

Time for candidate selection: 0.22 seconds

### Candidate
type: RSZ, layer: 1, pos: 176

### Relational analysis RSZ of RS_RSZ2_RSZ2_RSZ2_RSZ2_RSZ2_RSZ1

#### Relational analysis RSZ result of RS_RSZ2_RSZ2_RSZ2_RSZ2_RSZ2_RSZ1
Status: Status.UNKNOWN
Output dim: 1, lower bound: -0.0008374, upper bound: 0.0008374
time: 0.65 seconds

### Relational analysis RSZ of RS_RSZ2_RSZ2_RSZ2_RSZ2_RSZ2_RSZ2

#### Relational analysis RSZ result of RS_RSZ2_RSZ2_RSZ2_RSZ2_RSZ2_RSZ2
Status: Status.UNKNOWN
Output dim: 1, lower bound: -0.0008374, upper bound: 0.0008374
time: 0.69 seconds

## Summary of splitting (split count: 5)
- Time for RS candidates: 3.20 seconds
RS_RSZ1_RSZ1_RSZ1_RSZ1_RSZ1_RSZ1, status: Status.UNKNOWN, split count: 6, time: 3.20
Output dim: 1, lower bound: -0.0008374, upper bound: 0.0008374
RS_RSZ1_RSZ1_RSZ1_RSZ1_RSZ1_RSZ2, status: Status.UNKNOWN, split count: 6, time: 3.20
Output dim: 1, lower bound: -0.0008374, upper bound: 0.0008374
RS_RSZ1_RSZ1_RSZ1_RSZ1_RSZ2_RSZ1, status: Status.UNKNOWN, split count: 6, time: 3.20
Output dim: 1, lower bound: -0.0008374, upper bound: 0.0008374
RS_RSZ1_RSZ1_RSZ1_RSZ1_RSZ2_RSZ2, status: Status.UNKNOWN, split count: 6, time: 3.20
Output dim: 1, lower bound: -0.0008374, upper bound: 0.0008374
RS_RSZ1_RSZ1_RSZ1_RSZ2_RSZ1_RSZ1, status: Status.UNKNOWN, split count: 6, time: 3.20
Output dim: 1, lower bound: -0.0008374, upper bound: 0.0008374
RS_RSZ1_RSZ1_RSZ1_RSZ2_RSZ1_RSZ2, status: Status.UNKNOWN, split count: 6, time: 3.20
Output dim: 1, lower bound: -0.0008374, upper bound: 0.0008374
RS_RSZ1_RSZ1_RSZ1_RSZ2_RSZ2_RSZ1, status: Status.UNKNOWN, split count: 6, time: 3.20
Output dim: 1, lower bound: -0.0008374, upper bound: 0.0008374
RS_RSZ1_RSZ1_RSZ1_RSZ2_RSZ2_RSZ2, status: Status.UNKNOWN, split count: 6, time: 3.20
Output dim: 1, lower bound: -0.0008374, upper bound: 0.0008374
RS_RSZ1_RSZ1_RSZ2_RSZ1_RSZ1_RSZ1, status: Status.UNKNOWN, split count: 6, time: 3.20
Output dim: 1, lower bound: -0.0008374, upper bound: 0.0008374
RS_RSZ1_RSZ1_RSZ2_RSZ1_RSZ1_RSZ2, status: Status.UNKNOWN, split count: 6, time: 3.20
Output dim: 1, lower bound: -0.0008374, upper bound: 0.0008374
RS_RSZ1_RSZ1_RSZ2_RSZ1_RSZ2_RSZ1, status: Status.UNKNOWN, split count: 6, time: 3.20
Output dim: 1, lower bound: -0.0008374, upper bound: 0.0008374
RS_RSZ1_RSZ1_RSZ2_RSZ1_RSZ2_RSZ2, status: Status.UNKNOWN, split count: 6, time: 3.20
Output dim: 1, lower bound: -0.0008374, upper bound: 0.0008374
RS_RSZ1_RSZ1_RSZ2_RSZ2_RSZ1_RSZ1, status: Status.UNKNOWN, split count: 6, time: 3.20
Output dim: 1, lower bound: -0.0008374, upper bound: 0.0008374
RS_RSZ1_RSZ1_RSZ2_RSZ2_RSZ1_RSZ2, status: Status.UNKNOWN, split count: 6, time: 3.20
Output dim: 1, lower bound: -0.0008374, upper bound: 0.0008374
RS_RSZ1_RSZ1_RSZ2_RSZ2_RSZ2_RSZ1, status: Status.UNKNOWN, split count: 6, time: 3.20
Output dim: 1, lower bound: -0.0008374, upper bound: 0.0008374
RS_RSZ1_RSZ1_RSZ2_RSZ2_RSZ2_RSZ2, status: Status.UNKNOWN, split count: 6, time: 3.20
Output dim: 1, lower bound: -0.0008374, upper bound: 0.0008374
RS_RSZ1_RSZ2_RSZ1_RSZ1_RSZ1_RSZ1, status: Status.UNKNOWN, split count: 6, time: 3.20
Output dim: 1, lower bound: -0.0008374, upper bound: 0.0008374
RS_RSZ1_RSZ2_RSZ1_RSZ1_RSZ1_RSZ2, status: Status.UNKNOWN, split count: 6, time: 3.20
Output dim: 1, lower bound: -0.0008374, upper bound: 0.0008374
RS_RSZ1_RSZ2_RSZ1_RSZ1_RSZ2_RSZ1, status: Status.UNKNOWN, split count: 6, time: 3.20
Output dim: 1, lower bound: -0.0008374, upper bound: 0.0008374
RS_RSZ1_RSZ2_RSZ1_RSZ1_RSZ2_RSZ2, status: Status.UNKNOWN, split count: 6, time: 3.20
Output dim: 1, lower bound: -0.0008374, upper bound: 0.0008374
RS_RSZ1_RSZ2_RSZ1_RSZ2_RSZ1_RSZ1, status: Status.UNKNOWN, split count: 6, time: 3.20
Output dim: 1, lower bound: -0.0008374, upper bound: 0.0008374
RS_RSZ1_RSZ2_RSZ1_RSZ2_RSZ1_RSZ2, status: Status.UNKNOWN, split count: 6, time: 3.20
Output dim: 1, lower bound: -0.0008374, upper bound: 0.0008374
RS_RSZ1_RSZ2_RSZ1_RSZ2_RSZ2_RSZ1, status: Status.UNKNOWN, split count: 6, time: 3.20
Output dim: 1, lower bound: -0.0008374, upper bound: 0.0008374
RS_RSZ1_RSZ2_RSZ1_RSZ2_RSZ2_RSZ2, status: Status.UNKNOWN, split count: 6, time: 3.20
Output dim: 1, lower bound: -0.0008374, upper bound: 0.0008374
RS_RSZ1_RSZ2_RSZ2_RSZ1_RSZ1_RSZ1, status: Status.UNKNOWN, split count: 6, time: 3.20
Output dim: 1, lower bound: -0.0008374, upper bound: 0.0008374
RS_RSZ1_RSZ2_RSZ2_RSZ1_RSZ1_RSZ2, status: Status.UNKNOWN, split count: 6, time: 3.20
Output dim: 1, lower bound: -0.0008374, upper bound: 0.0008374
RS_RSZ1_RSZ2_RSZ2_RSZ1_RSZ2_RSZ1, status: Status.UNKNOWN, split count: 6, time: 3.20
Output dim: 1, lower bound: -0.0008374, upper bound: 0.0008374
RS_RSZ1_RSZ2_RSZ2_RSZ1_RSZ2_RSZ2, status: Status.UNKNOWN, split count: 6, time: 3.20
Output dim: 1, lower bound: -0.0008374, upper bound: 0.0008374
RS_RSZ1_RSZ2_RSZ2_RSZ2_RSZ1_RSZ1, status: Status.UNKNOWN, split count: 6, time: 3.20
Output dim: 1, lower bound: -0.0008374, upper bound: 0.0008374
RS_RSZ1_RSZ2_RSZ2_RSZ2_RSZ1_RSZ2, status: Status.UNKNOWN, split count: 6, time: 3.20
Output dim: 1, lower bound: -0.0008374, upper bound: 0.0008374
RS_RSZ1_RSZ2_RSZ2_RSZ2_RSZ2_RSZ1, status: Status.UNKNOWN, split count: 6, time: 3.20
Output dim: 1, lower bound: -0.0008374, upper bound: 0.0008374
RS_RSZ1_RSZ2_RSZ2_RSZ2_RSZ2_RSZ2, status: Status.UNKNOWN, split count: 6, time: 3.20
Output dim: 1, lower bound: -0.0008374, upper bound: 0.0008374
RS_RSZ2_RSZ1_RSZ1_RSZ1_RSZ1_RSZ1, status: Status.UNKNOWN, split count: 6, time: 3.20
Output dim: 1, lower bound: -0.0008374, upper bound: 0.0008374
RS_RSZ2_RSZ1_RSZ1_RSZ1_RSZ1_RSZ2, status: Status.UNKNOWN, split count: 6, time: 3.20
Output dim: 1, lower bound: -0.0008374, upper bound: 0.0008374
RS_RSZ2_RSZ1_RSZ1_RSZ1_RSZ2_RSZ1, status: Status.UNKNOWN, split count: 6, time: 3.20
Output dim: 1, lower bound: -0.0008374, upper bound: 0.0008374
RS_RSZ2_RSZ1_RSZ1_RSZ1_RSZ2_RSZ2, status: Status.UNKNOWN, split count: 6, time: 3.20
Output dim: 1, lower bound: -0.0008374, upper bound: 0.0008374
RS_RSZ2_RSZ1_RSZ1_RSZ2_RSZ1_RSZ1, status: Status.UNKNOWN, split count: 6, time: 3.20
Output dim: 1, lower bound: -0.0008374, upper bound: 0.0008374
RS_RSZ2_RSZ1_RSZ1_RSZ2_RSZ1_RSZ2, status: Status.UNKNOWN, split count: 6, time: 3.20
Output dim: 1, lower bound: -0.0008374, upper bound: 0.0008374
RS_RSZ2_RSZ1_RSZ1_RSZ2_RSZ2_RSZ1, status: Status.UNKNOWN, split count: 6, time: 3.20
Output dim: 1, lower bound: -0.0008374, upper bound: 0.0008374
RS_RSZ2_RSZ1_RSZ1_RSZ2_RSZ2_RSZ2, status: Status.UNKNOWN, split count: 6, time: 3.20
Output dim: 1, lower bound: -0.0008374, upper bound: 0.0008374
RS_RSZ2_RSZ1_RSZ2_RSZ1_RSZ1_RSZ1, status: Status.UNKNOWN, split count: 6, time: 3.20
Output dim: 1, lower bound: -0.0008374, upper bound: 0.0008374
RS_RSZ2_RSZ1_RSZ2_RSZ1_RSZ1_RSZ2, status: Status.UNKNOWN, split count: 6, time: 3.20
Output dim: 1, lower bound: -0.0008374, upper bound: 0.0008374
RS_RSZ2_RSZ1_RSZ2_RSZ1_RSZ2_RSZ1, status: Status.UNKNOWN, split count: 6, time: 3.20
Output dim: 1, lower bound: -0.0008374, upper bound: 0.0008374
RS_RSZ2_RSZ1_RSZ2_RSZ1_RSZ2_RSZ2, status: Status.UNKNOWN, split count: 6, time: 3.20
Output dim: 1, lower bound: -0.0008374, upper bound: 0.0008374
RS_RSZ2_RSZ1_RSZ2_RSZ2_RSZ1_RSZ1, status: Status.UNKNOWN, split count: 6, time: 3.20
Output dim: 1, lower bound: -0.0008374, upper bound: 0.0008374
RS_RSZ2_RSZ1_RSZ2_RSZ2_RSZ1_RSZ2, status: Status.UNKNOWN, split count: 6, time: 3.20
Output dim: 1, lower bound: -0.0008374, upper bound: 0.0008374
RS_RSZ2_RSZ1_RSZ2_RSZ2_RSZ2_RSZ1, status: Status.UNKNOWN, split count: 6, time: 3.20
Output dim: 1, lower bound: -0.0008374, upper bound: 0.0008374
RS_RSZ2_RSZ1_RSZ2_RSZ2_RSZ2_RSZ2, status: Status.UNKNOWN, split count: 6, time: 3.20
Output dim: 1, lower bound: -0.0008374, upper bound: 0.0008374
RS_RSZ2_RSZ2_RSZ1_RSZ1_RSZ1_RSZ1, status: Status.UNKNOWN, split count: 6, time: 3.20
Output dim: 1, lower bound: -0.0008374, upper bound: 0.0008374
RS_RSZ2_RSZ2_RSZ1_RSZ1_RSZ1_RSZ2, status: Status.UNKNOWN, split count: 6, time: 3.20
Output dim: 1, lower bound: -0.0008374, upper bound: 0.0008374
RS_RSZ2_RSZ2_RSZ1_RSZ1_RSZ2_RSZ1, status: Status.UNKNOWN, split count: 6, time: 3.20
Output dim: 1, lower bound: -0.0008374, upper bound: 0.0008374
RS_RSZ2_RSZ2_RSZ1_RSZ1_RSZ2_RSZ2, status: Status.UNKNOWN, split count: 6, time: 3.20
Output dim: 1, lower bound: -0.0008374, upper bound: 0.0008374
RS_RSZ2_RSZ2_RSZ1_RSZ2_RSZ1_RSZ1, status: Status.UNKNOWN, split count: 6, time: 3.20
Output dim: 1, lower bound: -0.0008374, upper bound: 0.0008374
RS_RSZ2_RSZ2_RSZ1_RSZ2_RSZ1_RSZ2, status: Status.UNKNOWN, split count: 6, time: 3.20
Output dim: 1, lower bound: -0.0008374, upper bound: 0.0008374
RS_RSZ2_RSZ2_RSZ1_RSZ2_RSZ2_RSZ1, status: Status.UNKNOWN, split count: 6, time: 3.20
Output dim: 1, lower bound: -0.0008374, upper bound: 0.0008374
RS_RSZ2_RSZ2_RSZ1_RSZ2_RSZ2_RSZ2, status: Status.UNKNOWN, split count: 6, time: 3.20
Output dim: 1, lower bound: -0.0008374, upper bound: 0.0008374
RS_RSZ2_RSZ2_RSZ2_RSZ1_RSZ1_RSZ1, status: Status.UNKNOWN, split count: 6, time: 3.20
Output dim: 1, lower bound: -0.0008374, upper bound: 0.0008374
RS_RSZ2_RSZ2_RSZ2_RSZ1_RSZ1_RSZ2, status: Status.UNKNOWN, split count: 6, time: 3.20
Output dim: 1, lower bound: -0.0008374, upper bound: 0.0008374
RS_RSZ2_RSZ2_RSZ2_RSZ1_RSZ2_RSZ1, status: Status.UNKNOWN, split count: 6, time: 3.20
Output dim: 1, lower bound: -0.0008374, upper bound: 0.0008374
RS_RSZ2_RSZ2_RSZ2_RSZ1_RSZ2_RSZ2, status: Status.UNKNOWN, split count: 6, time: 3.20
Output dim: 1, lower bound: -0.0008374, upper bound: 0.0008374
RS_RSZ2_RSZ2_RSZ2_RSZ2_RSZ1_RSZ1, status: Status.UNKNOWN, split count: 6, time: 3.20
Output dim: 1, lower bound: -0.0008374, upper bound: 0.0008374
RS_RSZ2_RSZ2_RSZ2_RSZ2_RSZ1_RSZ2, status: Status.UNKNOWN, split count: 6, time: 3.20
Output dim: 1, lower bound: -0.0008374, upper bound: 0.0008374
RS_RSZ2_RSZ2_RSZ2_RSZ2_RSZ2_RSZ1, status: Status.UNKNOWN, split count: 6, time: 3.20
Output dim: 1, lower bound: -0.0008374, upper bound: 0.0008374
RS_RSZ2_RSZ2_RSZ2_RSZ2_RSZ2_RSZ2, status: Status.UNKNOWN, split count: 6, time: 3.20
Output dim: 1, lower bound: -0.0008374, upper bound: 0.0008374

## BFS RS instance: RS_RSZ1_RSZ1_RSZ1_RSZ1_RSZ1_RSZ1

### Backsubstitution after applying RS history:
0: 0.0006996, 0.0011177, 0.0006996, 0.0011177, -0.0004181, 0.0004181
1: 0.9934013, 0.9943976, 0.9934013, 0.9943976, -0.0009964, 0.0009964
2: -0.0088595, -0.0053283, -0.0088595, -0.0053283, -0.0032443, 0.0032400
3: 0.0035633, 0.0041662, 0.0035633, 0.0041662, -0.0006029, 0.0006029
4: 0.0026282, 0.0054191, 0.0026282, 0.0054191, -0.0027909, 0.0027909
5: 0.0050201, 0.0064446, 0.0050201, 0.0064446, -0.0014245, 0.0014245
6: -0.0021963, -0.0008150, -0.0021963, -0.0008150, -0.0013813, 0.0013813
7: -0.0082640, -0.0074494, -0.0082640, -0.0074494, -0.0008146, 0.0008146
8: 0.0052524, 0.0098919, 0.0052524, 0.0098919, -0.0045618, 0.0045629
9: -0.0036845, -0.0030730, -0.0036845, -0.0030730, -0.0006115, 0.0006115

### Unstable ReLU Count (Linear/Conv2D Layers)
- layer_idx=0, type=LayerType.Linear, total=256, inp1_unstable=10, inp2_unstable=10, delta_unstable=250
- layer_idx=2, type=LayerType.Linear, total=256, inp1_unstable=8, inp2_unstable=8, delta_unstable=256
- layer_idx=4, type=LayerType.Linear, total=256, inp1_unstable=0, inp2_unstable=0, delta_unstable=256
- layer_idx=6, type=LayerType.Linear, total=256, inp1_unstable=2, inp2_unstable=2, delta_unstable=256
- layer_idx=8, type=LayerType.Linear, total=10, inp1_unstable=0, inp2_unstable=0, delta_unstable=10

Time for backsubstitution: 1.66 seconds

### RS candidates at layer 1
type: RSZ, layer: 1, pos: 198
type: RSZ, layer: 1, pos: 199
type: RSZ, layer: 1, pos: 204
type: RSZ, layer: 1, pos: 250

Time for candidate selection: 0.18 seconds

### Candidate
type: RSZ, layer: 1, pos: 198

### Relational analysis RSZ of RS_RSZ1_RSZ1_RSZ1_RSZ1_RSZ1_RSZ1_RSZ1

#### Relational analysis RSZ result of RS_RSZ1_RSZ1_RSZ1_RSZ1_RSZ1_RSZ1_RSZ1
Status: Status.UNKNOWN
Output dim: 1, lower bound: -0.0007828, upper bound: 0.0007828
time: 0.56 seconds

### Relational analysis RSZ of RS_RSZ1_RSZ1_RSZ1_RSZ1_RSZ1_RSZ1_RSZ2

#### Relational analysis RSZ result of RS_RSZ1_RSZ1_RSZ1_RSZ1_RSZ1_RSZ1_RSZ2
Status: Status.UNKNOWN
Output dim: 1, lower bound: -0.0007828, upper bound: 0.0007828
time: 0.57 seconds

## BFS RS instance: RS_RSZ1_RSZ1_RSZ1_RSZ1_RSZ1_RSZ2

### Backsubstitution after applying RS history:
0: 0.0006996, 0.0011177, 0.0006996, 0.0011177, -0.0004181, 0.0004181
1: 0.9934013, 0.9943976, 0.9934013, 0.9943976, -0.0009964, 0.0009964
2: -0.0088595, -0.0053283, -0.0088595, -0.0053283, -0.0032456, 0.0032390
3: 0.0035633, 0.0041662, 0.0035633, 0.0041662, -0.0006029, 0.0006029
4: 0.0026282, 0.0054191, 0.0026282, 0.0054191, -0.0027909, 0.0027909
5: 0.0050201, 0.0064446, 0.0050201, 0.0064446, -0.0014245, 0.0014245
6: -0.0021963, -0.0008150, -0.0021963, -0.0008150, -0.0013813, 0.0013813
7: -0.0082640, -0.0074494, -0.0082640, -0.0074494, -0.0008146, 0.0008146
8: 0.0052524, 0.0098919, 0.0052524, 0.0098919, -0.0045615, 0.0045633
9: -0.0036845, -0.0030730, -0.0036845, -0.0030730, -0.0006115, 0.0006115

### Unstable ReLU Count (Linear/Conv2D Layers)
- layer_idx=0, type=LayerType.Linear, total=256, inp1_unstable=10, inp2_unstable=10, delta_unstable=250
- layer_idx=2, type=LayerType.Linear, total=256, inp1_unstable=8, inp2_unstable=8, delta_unstable=256
- layer_idx=4, type=LayerType.Linear, total=256, inp1_unstable=0, inp2_unstable=0, delta_unstable=256
- layer_idx=6, type=LayerType.Linear, total=256, inp1_unstable=2, inp2_unstable=2, delta_unstable=256
- layer_idx=8, type=LayerType.Linear, total=10, inp1_unstable=0, inp2_unstable=0, delta_unstable=10

Time for backsubstitution: 1.66 seconds

### RS candidates at layer 1
type: RSZ, layer: 1, pos: 198
type: RSZ, layer: 1, pos: 199
type: RSZ, layer: 1, pos: 204
type: RSZ, layer: 1, pos: 250

Time for candidate selection: 0.17 seconds

### Candidate
type: RSZ, layer: 1, pos: 198

### Relational analysis RSZ of RS_RSZ1_RSZ1_RSZ1_RSZ1_RSZ1_RSZ2_RSZ1

#### Relational analysis RSZ result of RS_RSZ1_RSZ1_RSZ1_RSZ1_RSZ1_RSZ2_RSZ1
Status: Status.UNKNOWN
Output dim: 1, lower bound: -0.0007828, upper bound: 0.0007828
time: 0.60 seconds

### Relational analysis RSZ of RS_RSZ1_RSZ1_RSZ1_RSZ1_RSZ1_RSZ2_RSZ2

#### Relational analysis RSZ result of RS_RSZ1_RSZ1_RSZ1_RSZ1_RSZ1_RSZ2_RSZ2
Status: Status.UNKNOWN
Output dim: 1, lower bound: -0.0007828, upper bound: 0.0007828
time: 0.59 seconds

## BFS RS instance: RS_RSZ1_RSZ1_RSZ1_RSZ1_RSZ2_RSZ1

### Backsubstitution after applying RS history:
0: 0.0006996, 0.0011177, 0.0006996, 0.0011177, -0.0004181, 0.0004181
1: 0.9934013, 0.9943976, 0.9934013, 0.9943976, -0.0009964, 0.0009964
2: -0.0088595, -0.0053283, -0.0088595, -0.0053283, -0.0032424, 0.0032419
3: 0.0035633, 0.0041662, 0.0035633, 0.0041662, -0.0006029, 0.0006029
4: 0.0026282, 0.0054191, 0.0026282, 0.0054191, -0.0027909, 0.0027909
5: 0.0050201, 0.0064446, 0.0050201, 0.0064446, -0.0014245, 0.0014245
6: -0.0021963, -0.0008150, -0.0021963, -0.0008150, -0.0013813, 0.0013813
7: -0.0082640, -0.0074494, -0.0082640, -0.0074494, -0.0008146, 0.0008146
8: 0.0052524, 0.0098919, 0.0052524, 0.0098919, -0.0045623, 0.0045624
9: -0.0036845, -0.0030730, -0.0036845, -0.0030730, -0.0006115, 0.0006115

### Unstable ReLU Count (Linear/Conv2D Layers)
- layer_idx=0, type=LayerType.Linear, total=256, inp1_unstable=10, inp2_unstable=10, delta_unstable=250
- layer_idx=2, type=LayerType.Linear, total=256, inp1_unstable=8, inp2_unstable=8, delta_unstable=256
- layer_idx=4, type=LayerType.Linear, total=256, inp1_unstable=0, inp2_unstable=0, delta_unstable=256
- layer_idx=6, type=LayerType.Linear, total=256, inp1_unstable=2, inp2_unstable=2, delta_unstable=256
- layer_idx=8, type=LayerType.Linear, total=10, inp1_unstable=0, inp2_unstable=0, delta_unstable=10

Time for backsubstitution: 1.67 seconds

### RS candidates at layer 1
type: RSZ, layer: 1, pos: 198
type: RSZ, layer: 1, pos: 199
type: RSZ, layer: 1, pos: 204
type: RSZ, layer: 1, pos: 250

Time for candidate selection: 0.18 seconds

### Candidate
type: RSZ, layer: 1, pos: 198

### Relational analysis RSZ of RS_RSZ1_RSZ1_RSZ1_RSZ1_RSZ2_RSZ1_RSZ1

#### Relational analysis RSZ result of RS_RSZ1_RSZ1_RSZ1_RSZ1_RSZ2_RSZ1_RSZ1
Status: Status.UNKNOWN
Output dim: 1, lower bound: -0.0007828, upper bound: 0.0007828
time: 0.60 seconds

### Relational analysis RSZ of RS_RSZ1_RSZ1_RSZ1_RSZ1_RSZ2_RSZ1_RSZ2

#### Relational analysis RSZ result of RS_RSZ1_RSZ1_RSZ1_RSZ1_RSZ2_RSZ1_RSZ2
Status: Status.UNKNOWN
Output dim: 1, lower bound: -0.0007828, upper bound: 0.0007828
time: 0.61 seconds

## BFS RS instance: RS_RSZ1_RSZ1_RSZ1_RSZ1_RSZ2_RSZ2

### Backsubstitution after applying RS history:
0: 0.0006996, 0.0011177, 0.0006996, 0.0011177, -0.0004181, 0.0004181
1: 0.9934013, 0.9943976, 0.9934013, 0.9943976, -0.0009964, 0.0009964
2: -0.0088595, -0.0053283, -0.0088595, -0.0053283, -0.0032433, 0.0032408
3: 0.0035633, 0.0041662, 0.0035633, 0.0041662, -0.0006029, 0.0006029
4: 0.0026282, 0.0054191, 0.0026282, 0.0054191, -0.0027909, 0.0027909
5: 0.0050201, 0.0064446, 0.0050201, 0.0064446, -0.0014245, 0.0014245
6: -0.0021963, -0.0008150, -0.0021963, -0.0008150, -0.0013813, 0.0013813
7: -0.0082640, -0.0074494, -0.0082640, -0.0074494, -0.0008146, 0.0008146
8: 0.0052524, 0.0098919, 0.0052524, 0.0098919, -0.0045620, 0.0045627
9: -0.0036845, -0.0030730, -0.0036845, -0.0030730, -0.0006115, 0.0006115

### Unstable ReLU Count (Linear/Conv2D Layers)
- layer_idx=0, type=LayerType.Linear, total=256, inp1_unstable=10, inp2_unstable=10, delta_unstable=250
- layer_idx=2, type=LayerType.Linear, total=256, inp1_unstable=8, inp2_unstable=8, delta_unstable=256
- layer_idx=4, type=LayerType.Linear, total=256, inp1_unstable=0, inp2_unstable=0, delta_unstable=256
- layer_idx=6, type=LayerType.Linear, total=256, inp1_unstable=2, inp2_unstable=2, delta_unstable=256
- layer_idx=8, type=LayerType.Linear, total=10, inp1_unstable=0, inp2_unstable=0, delta_unstable=10

Time for backsubstitution: 1.72 seconds

### RS candidates at layer 1
type: RSZ, layer: 1, pos: 198
type: RSZ, layer: 1, pos: 199
type: RSZ, layer: 1, pos: 204
type: RSZ, layer: 1, pos: 250

Time for candidate selection: 0.22 seconds

### Candidate
type: RSZ, layer: 1, pos: 198

### Relational analysis RSZ of RS_RSZ1_RSZ1_RSZ1_RSZ1_RSZ2_RSZ2_RSZ1

#### Relational analysis RSZ result of RS_RSZ1_RSZ1_RSZ1_RSZ1_RSZ2_RSZ2_RSZ1
Status: Status.UNKNOWN
Output dim: 1, lower bound: -0.0007828, upper bound: 0.0007828
time: 0.65 seconds

### Relational analysis RSZ of RS_RSZ1_RSZ1_RSZ1_RSZ1_RSZ2_RSZ2_RSZ2

#### Relational analysis RSZ result of RS_RSZ1_RSZ1_RSZ1_RSZ1_RSZ2_RSZ2_RSZ2
Status: Status.UNKNOWN
Output dim: 1, lower bound: -0.0007828, upper bound: 0.0007828
time: 0.62 seconds

## BFS RS instance: RS_RSZ1_RSZ1_RSZ1_RSZ2_RSZ1_RSZ1

### Backsubstitution after applying RS history:
0: 0.0006996, 0.0011177, 0.0006996, 0.0011177, -0.0004181, 0.0004181
1: 0.9934013, 0.9943976, 0.9934013, 0.9943976, -0.0009964, 0.0009964
2: -0.0088595, -0.0053283, -0.0088595, -0.0053283, -0.0032435, 0.0032415
3: 0.0035633, 0.0041662, 0.0035633, 0.0041662, -0.0006029, 0.0006029
4: 0.0026282, 0.0054191, 0.0026282, 0.0054191, -0.0027909, 0.0027909
5: 0.0050201, 0.0064446, 0.0050201, 0.0064446, -0.0014245, 0.0014245
6: -0.0021963, -0.0008150, -0.0021963, -0.0008150, -0.0013813, 0.0013813
7: -0.0082640, -0.0074494, -0.0082640, -0.0074494, -0.0008146, 0.0008146
8: 0.0052524, 0.0098919, 0.0052524, 0.0098919, -0.0045622, 0.0045627
9: -0.0036845, -0.0030730, -0.0036845, -0.0030730, -0.0006115, 0.0006115

### Unstable ReLU Count (Linear/Conv2D Layers)
- layer_idx=0, type=LayerType.Linear, total=256, inp1_unstable=10, inp2_unstable=10, delta_unstable=250
- layer_idx=2, type=LayerType.Linear, total=256, inp1_unstable=8, inp2_unstable=8, delta_unstable=256
- layer_idx=4, type=LayerType.Linear, total=256, inp1_unstable=0, inp2_unstable=0, delta_unstable=256
- layer_idx=6, type=LayerType.Linear, total=256, inp1_unstable=2, inp2_unstable=2, delta_unstable=256
- layer_idx=8, type=LayerType.Linear, total=10, inp1_unstable=0, inp2_unstable=0, delta_unstable=10

Time for backsubstitution: 1.61 seconds

### RS candidates at layer 1
type: RSZ, layer: 1, pos: 198
type: RSZ, layer: 1, pos: 199
type: RSZ, layer: 1, pos: 204
type: RSZ, layer: 1, pos: 250

Time for candidate selection: 0.17 seconds

### Candidate
type: RSZ, layer: 1, pos: 198

### Relational analysis RSZ of RS_RSZ1_RSZ1_RSZ1_RSZ2_RSZ1_RSZ1_RSZ1

#### Relational analysis RSZ result of RS_RSZ1_RSZ1_RSZ1_RSZ2_RSZ1_RSZ1_RSZ1
Status: Status.UNKNOWN
Output dim: 1, lower bound: -0.0007828, upper bound: 0.0007828
time: 0.56 seconds

### Relational analysis RSZ of RS_RSZ1_RSZ1_RSZ1_RSZ2_RSZ1_RSZ1_RSZ2

#### Relational analysis RSZ result of RS_RSZ1_RSZ1_RSZ1_RSZ2_RSZ1_RSZ1_RSZ2
Status: Status.UNKNOWN
Output dim: 1, lower bound: -0.0007828, upper bound: 0.0007828
time: 0.56 seconds

## BFS RS instance: RS_RSZ1_RSZ1_RSZ1_RSZ2_RSZ1_RSZ2

### Backsubstitution after applying RS history:
0: 0.0006996, 0.0011177, 0.0006996, 0.0011177, -0.0004181, 0.0004181
1: 0.9934013, 0.9943976, 0.9934013, 0.9943976, -0.0009964, 0.0009964
2: -0.0088595, -0.0053283, -0.0088595, -0.0053283, -0.0032446, 0.0032403
3: 0.0035633, 0.0041662, 0.0035633, 0.0041662, -0.0006029, 0.0006029
4: 0.0026282, 0.0054191, 0.0026282, 0.0054191, -0.0027909, 0.0027909
5: 0.0050201, 0.0064446, 0.0050201, 0.0064446, -0.0014245, 0.0014245
6: -0.0021963, -0.0008150, -0.0021963, -0.0008150, -0.0013813, 0.0013813
7: -0.0082640, -0.0074494, -0.0082640, -0.0074494, -0.0008146, 0.0008146
8: 0.0052524, 0.0098919, 0.0052524, 0.0098919, -0.0045619, 0.0045630
9: -0.0036845, -0.0030730, -0.0036845, -0.0030730, -0.0006115, 0.0006115

### Unstable ReLU Count (Linear/Conv2D Layers)
- layer_idx=0, type=LayerType.Linear, total=256, inp1_unstable=10, inp2_unstable=10, delta_unstable=250
- layer_idx=2, type=LayerType.Linear, total=256, inp1_unstable=8, inp2_unstable=8, delta_unstable=256
- layer_idx=4, type=LayerType.Linear, total=256, inp1_unstable=0, inp2_unstable=0, delta_unstable=256
- layer_idx=6, type=LayerType.Linear, total=256, inp1_unstable=2, inp2_unstable=2, delta_unstable=256
- layer_idx=8, type=LayerType.Linear, total=10, inp1_unstable=0, inp2_unstable=0, delta_unstable=10

Time for backsubstitution: 1.69 seconds

### RS candidates at layer 1
type: RSZ, layer: 1, pos: 198
type: RSZ, layer: 1, pos: 199
type: RSZ, layer: 1, pos: 204
type: RSZ, layer: 1, pos: 250

Time for candidate selection: 0.17 seconds

### Candidate
type: RSZ, layer: 1, pos: 198

### Relational analysis RSZ of RS_RSZ1_RSZ1_RSZ1_RSZ2_RSZ1_RSZ2_RSZ1

#### Relational analysis RSZ result of RS_RSZ1_RSZ1_RSZ1_RSZ2_RSZ1_RSZ2_RSZ1
Status: Status.UNKNOWN
Output dim: 1, lower bound: -0.0007828, upper bound: 0.0007828
time: 0.60 seconds

### Relational analysis RSZ of RS_RSZ1_RSZ1_RSZ1_RSZ2_RSZ1_RSZ2_RSZ2

#### Relational analysis RSZ result of RS_RSZ1_RSZ1_RSZ1_RSZ2_RSZ1_RSZ2_RSZ2
Status: Status.UNKNOWN
Output dim: 1, lower bound: -0.0007828, upper bound: 0.0007828
time: 0.60 seconds

## BFS RS instance: RS_RSZ1_RSZ1_RSZ1_RSZ2_RSZ2_RSZ1

### Backsubstitution after applying RS history:
0: 0.0006996, 0.0011177, 0.0006996, 0.0011177, -0.0004181, 0.0004181
1: 0.9934013, 0.9943976, 0.9934013, 0.9943976, -0.0009964, 0.0009964
2: -0.0088595, -0.0053283, -0.0088595, -0.0053283, -0.0032415, 0.0032431
3: 0.0035633, 0.0041662, 0.0035633, 0.0041662, -0.0006029, 0.0006029
4: 0.0026282, 0.0054191, 0.0026282, 0.0054191, -0.0027909, 0.0027909
5: 0.0050201, 0.0064446, 0.0050201, 0.0064446, -0.0014245, 0.0014245
6: -0.0021963, -0.0008150, -0.0021963, -0.0008150, -0.0013813, 0.0013813
7: -0.0082640, -0.0074494, -0.0082640, -0.0074494, -0.0008146, 0.0008146
8: 0.0052524, 0.0098919, 0.0052524, 0.0098919, -0.0045626, 0.0045622
9: -0.0036845, -0.0030730, -0.0036845, -0.0030730, -0.0006115, 0.0006115

### Unstable ReLU Count (Linear/Conv2D Layers)
- layer_idx=0, type=LayerType.Linear, total=256, inp1_unstable=10, inp2_unstable=10, delta_unstable=250
- layer_idx=2, type=LayerType.Linear, total=256, inp1_unstable=8, inp2_unstable=8, delta_unstable=256
- layer_idx=4, type=LayerType.Linear, total=256, inp1_unstable=0, inp2_unstable=0, delta_unstable=256
- layer_idx=6, type=LayerType.Linear, total=256, inp1_unstable=2, inp2_unstable=2, delta_unstable=256
- layer_idx=8, type=LayerType.Linear, total=10, inp1_unstable=0, inp2_unstable=0, delta_unstable=10

Time for backsubstitution: 1.62 seconds

### RS candidates at layer 1
type: RSZ, layer: 1, pos: 198
type: RSZ, layer: 1, pos: 199
type: RSZ, layer: 1, pos: 204
type: RSZ, layer: 1, pos: 250

Time for candidate selection: 0.18 seconds

### Candidate
type: RSZ, layer: 1, pos: 198

### Relational analysis RSZ of RS_RSZ1_RSZ1_RSZ1_RSZ2_RSZ2_RSZ1_RSZ1

#### Relational analysis RSZ result of RS_RSZ1_RSZ1_RSZ1_RSZ2_RSZ2_RSZ1_RSZ1
Status: Status.UNKNOWN
Output dim: 1, lower bound: -0.0007828, upper bound: 0.0007828
time: 0.60 seconds

### Relational analysis RSZ of RS_RSZ1_RSZ1_RSZ1_RSZ2_RSZ2_RSZ1_RSZ2

#### Relational analysis RSZ result of RS_RSZ1_RSZ1_RSZ1_RSZ2_RSZ2_RSZ1_RSZ2
Status: Status.UNKNOWN
Output dim: 1, lower bound: -0.0007828, upper bound: 0.0007828
time: 0.61 seconds

## BFS RS instance: RS_RSZ1_RSZ1_RSZ1_RSZ2_RSZ2_RSZ2

### Backsubstitution after applying RS history:
0: 0.0006996, 0.0011177, 0.0006996, 0.0011177, -0.0004181, 0.0004181
1: 0.9934013, 0.9943976, 0.9934013, 0.9943976, -0.0009964, 0.0009964
2: -0.0088595, -0.0053283, -0.0088595, -0.0053283, -0.0032424, 0.0032417
3: 0.0035633, 0.0041662, 0.0035633, 0.0041662, -0.0006029, 0.0006029
4: 0.0026282, 0.0054191, 0.0026282, 0.0054191, -0.0027909, 0.0027909
5: 0.0050201, 0.0064446, 0.0050201, 0.0064446, -0.0014245, 0.0014245
6: -0.0021963, -0.0008150, -0.0021963, -0.0008150, -0.0013813, 0.0013813
7: -0.0082640, -0.0074494, -0.0082640, -0.0074494, -0.0008146, 0.0008146
8: 0.0052524, 0.0098919, 0.0052524, 0.0098919, -0.0045622, 0.0045624
9: -0.0036845, -0.0030730, -0.0036845, -0.0030730, -0.0006115, 0.0006115

### Unstable ReLU Count (Linear/Conv2D Layers)
- layer_idx=0, type=LayerType.Linear, total=256, inp1_unstable=10, inp2_unstable=10, delta_unstable=250
- layer_idx=2, type=LayerType.Linear, total=256, inp1_unstable=8, inp2_unstable=8, delta_unstable=256
- layer_idx=4, type=LayerType.Linear, total=256, inp1_unstable=0, inp2_unstable=0, delta_unstable=256
- layer_idx=6, type=LayerType.Linear, total=256, inp1_unstable=2, inp2_unstable=2, delta_unstable=256
- layer_idx=8, type=LayerType.Linear, total=10, inp1_unstable=0, inp2_unstable=0, delta_unstable=10

Time for backsubstitution: 1.64 seconds

### RS candidates at layer 1
type: RSZ, layer: 1, pos: 198
type: RSZ, layer: 1, pos: 199
type: RSZ, layer: 1, pos: 204
type: RSZ, layer: 1, pos: 250

Time for candidate selection: 0.18 seconds

### Candidate
type: RSZ, layer: 1, pos: 198

### Relational analysis RSZ of RS_RSZ1_RSZ1_RSZ1_RSZ2_RSZ2_RSZ2_RSZ1

#### Relational analysis RSZ result of RS_RSZ1_RSZ1_RSZ1_RSZ2_RSZ2_RSZ2_RSZ1
Status: Status.UNKNOWN
Output dim: 1, lower bound: -0.0007828, upper bound: 0.0007828
time: 0.62 seconds

### Relational analysis RSZ of RS_RSZ1_RSZ1_RSZ1_RSZ2_RSZ2_RSZ2_RSZ2

#### Relational analysis RSZ result of RS_RSZ1_RSZ1_RSZ1_RSZ2_RSZ2_RSZ2_RSZ2
Status: Status.UNKNOWN
Output dim: 1, lower bound: -0.0007828, upper bound: 0.0007828
time: 0.63 seconds

## BFS RS instance: RS_RSZ1_RSZ1_RSZ2_RSZ1_RSZ1_RSZ1

### Backsubstitution after applying RS history:
0: 0.0006996, 0.0011177, 0.0006996, 0.0011177, -0.0004181, 0.0004181
1: 0.9934013, 0.9943976, 0.9934013, 0.9943976, -0.0009964, 0.0009964
2: -0.0088595, -0.0053283, -0.0088595, -0.0053283, -0.0032439, 0.0032400
3: 0.0035633, 0.0041662, 0.0035633, 0.0041662, -0.0006029, 0.0006029
4: 0.0026282, 0.0054191, 0.0026282, 0.0054191, -0.0027909, 0.0027909
5: 0.0050201, 0.0064446, 0.0050201, 0.0064446, -0.0014245, 0.0014245
6: -0.0021963, -0.0008150, -0.0021963, -0.0008150, -0.0013813, 0.0013813
7: -0.0082640, -0.0074494, -0.0082640, -0.0074494, -0.0008146, 0.0008146
8: 0.0052524, 0.0098919, 0.0052524, 0.0098919, -0.0045618, 0.0045628
9: -0.0036845, -0.0030730, -0.0036845, -0.0030730, -0.0006115, 0.0006115

### Unstable ReLU Count (Linear/Conv2D Layers)
- layer_idx=0, type=LayerType.Linear, total=256, inp1_unstable=10, inp2_unstable=10, delta_unstable=250
- layer_idx=2, type=LayerType.Linear, total=256, inp1_unstable=8, inp2_unstable=8, delta_unstable=256
- layer_idx=4, type=LayerType.Linear, total=256, inp1_unstable=0, inp2_unstable=0, delta_unstable=256
- layer_idx=6, type=LayerType.Linear, total=256, inp1_unstable=2, inp2_unstable=2, delta_unstable=256
- layer_idx=8, type=LayerType.Linear, total=10, inp1_unstable=0, inp2_unstable=0, delta_unstable=10

Time for backsubstitution: 1.62 seconds

### RS candidates at layer 1
type: RSZ, layer: 1, pos: 198
type: RSZ, layer: 1, pos: 199
type: RSZ, layer: 1, pos: 204
type: RSZ, layer: 1, pos: 250

Time for candidate selection: 0.17 seconds

### Candidate
type: RSZ, layer: 1, pos: 198

### Relational analysis RSZ of RS_RSZ1_RSZ1_RSZ2_RSZ1_RSZ1_RSZ1_RSZ1

#### Relational analysis RSZ result of RS_RSZ1_RSZ1_RSZ2_RSZ1_RSZ1_RSZ1_RSZ1
Status: Status.UNKNOWN
Output dim: 1, lower bound: -0.0007828, upper bound: 0.0007828
time: 0.55 seconds

### Relational analysis RSZ of RS_RSZ1_RSZ1_RSZ2_RSZ1_RSZ1_RSZ1_RSZ2

#### Relational analysis RSZ result of RS_RSZ1_RSZ1_RSZ2_RSZ1_RSZ1_RSZ1_RSZ2
Status: Status.UNKNOWN
Output dim: 1, lower bound: -0.0007828, upper bound: 0.0007828
time: 0.56 seconds

## BFS RS instance: RS_RSZ1_RSZ1_RSZ2_RSZ1_RSZ1_RSZ2

### Backsubstitution after applying RS history:
0: 0.0006996, 0.0011177, 0.0006996, 0.0011177, -0.0004181, 0.0004181
1: 0.9934013, 0.9943976, 0.9934013, 0.9943976, -0.0009964, 0.0009964
2: -0.0088595, -0.0053283, -0.0088595, -0.0053283, -0.0032451, 0.0032390
3: 0.0035633, 0.0041662, 0.0035633, 0.0041662, -0.0006029, 0.0006029
4: 0.0026282, 0.0054191, 0.0026282, 0.0054191, -0.0027909, 0.0027909
5: 0.0050201, 0.0064446, 0.0050201, 0.0064446, -0.0014245, 0.0014245
6: -0.0021963, -0.0008150, -0.0021963, -0.0008150, -0.0013813, 0.0013813
7: -0.0082640, -0.0074494, -0.0082640, -0.0074494, -0.0008146, 0.0008146
8: 0.0052524, 0.0098919, 0.0052524, 0.0098919, -0.0045615, 0.0045632
9: -0.0036845, -0.0030730, -0.0036845, -0.0030730, -0.0006115, 0.0006115

### Unstable ReLU Count (Linear/Conv2D Layers)
- layer_idx=0, type=LayerType.Linear, total=256, inp1_unstable=10, inp2_unstable=10, delta_unstable=250
- layer_idx=2, type=LayerType.Linear, total=256, inp1_unstable=8, inp2_unstable=8, delta_unstable=256
- layer_idx=4, type=LayerType.Linear, total=256, inp1_unstable=0, inp2_unstable=0, delta_unstable=256
- layer_idx=6, type=LayerType.Linear, total=256, inp1_unstable=2, inp2_unstable=2, delta_unstable=256
- layer_idx=8, type=LayerType.Linear, total=10, inp1_unstable=0, inp2_unstable=0, delta_unstable=10

Time for backsubstitution: 1.64 seconds

### RS candidates at layer 1
type: RSZ, layer: 1, pos: 198
type: RSZ, layer: 1, pos: 199
type: RSZ, layer: 1, pos: 204
type: RSZ, layer: 1, pos: 250

Time for candidate selection: 0.17 seconds

### Candidate
type: RSZ, layer: 1, pos: 198

### Relational analysis RSZ of RS_RSZ1_RSZ1_RSZ2_RSZ1_RSZ1_RSZ2_RSZ1

#### Relational analysis RSZ result of RS_RSZ1_RSZ1_RSZ2_RSZ1_RSZ1_RSZ2_RSZ1
Status: Status.UNKNOWN
Output dim: 1, lower bound: -0.0007828, upper bound: 0.0007828
time: 0.56 seconds

### Relational analysis RSZ of RS_RSZ1_RSZ1_RSZ2_RSZ1_RSZ1_RSZ2_RSZ2

#### Relational analysis RSZ result of RS_RSZ1_RSZ1_RSZ2_RSZ1_RSZ1_RSZ2_RSZ2
Status: Status.UNKNOWN
Output dim: 1, lower bound: -0.0007828, upper bound: 0.0007828
time: 0.59 seconds

## BFS RS instance: RS_RSZ1_RSZ1_RSZ2_RSZ1_RSZ2_RSZ1

### Backsubstitution after applying RS history:
0: 0.0006996, 0.0011177, 0.0006996, 0.0011177, -0.0004181, 0.0004181
1: 0.9934013, 0.9943976, 0.9934013, 0.9943976, -0.0009964, 0.0009964
2: -0.0088595, -0.0053283, -0.0088595, -0.0053283, -0.0032423, 0.0032420
3: 0.0035633, 0.0041662, 0.0035633, 0.0041662, -0.0006029, 0.0006029
4: 0.0026282, 0.0054191, 0.0026282, 0.0054191, -0.0027909, 0.0027909
5: 0.0050201, 0.0064446, 0.0050201, 0.0064446, -0.0014245, 0.0014245
6: -0.0021963, -0.0008150, -0.0021963, -0.0008150, -0.0013813, 0.0013813
7: -0.0082640, -0.0074494, -0.0082640, -0.0074494, -0.0008146, 0.0008146
8: 0.0052524, 0.0098919, 0.0052524, 0.0098919, -0.0045623, 0.0045624
9: -0.0036845, -0.0030730, -0.0036845, -0.0030730, -0.0006115, 0.0006115

### Unstable ReLU Count (Linear/Conv2D Layers)
- layer_idx=0, type=LayerType.Linear, total=256, inp1_unstable=10, inp2_unstable=10, delta_unstable=250
- layer_idx=2, type=LayerType.Linear, total=256, inp1_unstable=8, inp2_unstable=8, delta_unstable=256
- layer_idx=4, type=LayerType.Linear, total=256, inp1_unstable=0, inp2_unstable=0, delta_unstable=256
- layer_idx=6, type=LayerType.Linear, total=256, inp1_unstable=2, inp2_unstable=2, delta_unstable=256
- layer_idx=8, type=LayerType.Linear, total=10, inp1_unstable=0, inp2_unstable=0, delta_unstable=10

Time for backsubstitution: 1.67 seconds

### RS candidates at layer 1
type: RSZ, layer: 1, pos: 198
type: RSZ, layer: 1, pos: 199
type: RSZ, layer: 1, pos: 204
type: RSZ, layer: 1, pos: 250

Time for candidate selection: 0.19 seconds

### Candidate
type: RSZ, layer: 1, pos: 198

### Relational analysis RSZ of RS_RSZ1_RSZ1_RSZ2_RSZ1_RSZ2_RSZ1_RSZ1

#### Relational analysis RSZ result of RS_RSZ1_RSZ1_RSZ2_RSZ1_RSZ2_RSZ1_RSZ1
Status: Status.UNKNOWN
Output dim: 1, lower bound: -0.0007828, upper bound: 0.0007828
time: 0.61 seconds

### Relational analysis RSZ of RS_RSZ1_RSZ1_RSZ2_RSZ1_RSZ2_RSZ1_RSZ2

#### Relational analysis RSZ result of RS_RSZ1_RSZ1_RSZ2_RSZ1_RSZ2_RSZ1_RSZ2
Status: Status.UNKNOWN
Output dim: 1, lower bound: -0.0007828, upper bound: 0.0007828
time: 0.62 seconds

## BFS RS instance: RS_RSZ1_RSZ1_RSZ2_RSZ1_RSZ2_RSZ2

### Backsubstitution after applying RS history:
0: 0.0006996, 0.0011177, 0.0006996, 0.0011177, -0.0004181, 0.0004181
1: 0.9934013, 0.9943976, 0.9934013, 0.9943976, -0.0009964, 0.0009964
2: -0.0088595, -0.0053283, -0.0088595, -0.0053283, -0.0032432, 0.0032408
3: 0.0035633, 0.0041662, 0.0035633, 0.0041662, -0.0006029, 0.0006029
4: 0.0026282, 0.0054191, 0.0026282, 0.0054191, -0.0027909, 0.0027909
5: 0.0050201, 0.0064446, 0.0050201, 0.0064446, -0.0014245, 0.0014245
6: -0.0021963, -0.0008150, -0.0021963, -0.0008150, -0.0013813, 0.0013813
7: -0.0082640, -0.0074494, -0.0082640, -0.0074494, -0.0008146, 0.0008146
8: 0.0052524, 0.0098919, 0.0052524, 0.0098919, -0.0045620, 0.0045626
9: -0.0036845, -0.0030730, -0.0036845, -0.0030730, -0.0006115, 0.0006115

### Unstable ReLU Count (Linear/Conv2D Layers)
- layer_idx=0, type=LayerType.Linear, total=256, inp1_unstable=10, inp2_unstable=10, delta_unstable=250
- layer_idx=2, type=LayerType.Linear, total=256, inp1_unstable=8, inp2_unstable=8, delta_unstable=256
- layer_idx=4, type=LayerType.Linear, total=256, inp1_unstable=0, inp2_unstable=0, delta_unstable=256
- layer_idx=6, type=LayerType.Linear, total=256, inp1_unstable=2, inp2_unstable=2, delta_unstable=256
- layer_idx=8, type=LayerType.Linear, total=10, inp1_unstable=0, inp2_unstable=0, delta_unstable=10

Time for backsubstitution: 1.83 seconds

### RS candidates at layer 1
type: RSZ, layer: 1, pos: 198
type: RSZ, layer: 1, pos: 199
type: RSZ, layer: 1, pos: 204
type: RSZ, layer: 1, pos: 250

Time for candidate selection: 0.20 seconds

### Candidate
type: RSZ, layer: 1, pos: 198

### Relational analysis RSZ of RS_RSZ1_RSZ1_RSZ2_RSZ1_RSZ2_RSZ2_RSZ1

#### Relational analysis RSZ result of RS_RSZ1_RSZ1_RSZ2_RSZ1_RSZ2_RSZ2_RSZ1
Status: Status.UNKNOWN
Output dim: 1, lower bound: -0.0007828, upper bound: 0.0007828
time: 0.63 seconds

### Relational analysis RSZ of RS_RSZ1_RSZ1_RSZ2_RSZ1_RSZ2_RSZ2_RSZ2

#### Relational analysis RSZ result of RS_RSZ1_RSZ1_RSZ2_RSZ1_RSZ2_RSZ2_RSZ2
Status: Status.UNKNOWN
Output dim: 1, lower bound: -0.0007828, upper bound: 0.0007828
time: 0.65 seconds

## BFS RS instance: RS_RSZ1_RSZ1_RSZ2_RSZ2_RSZ1_RSZ1

### Backsubstitution after applying RS history:
0: 0.0006996, 0.0011177, 0.0006996, 0.0011177, -0.0004181, 0.0004181
1: 0.9934013, 0.9943976, 0.9934013, 0.9943976, -0.0009964, 0.0009964
2: -0.0088595, -0.0053283, -0.0088595, -0.0053283, -0.0032431, 0.0032413
3: 0.0035633, 0.0041662, 0.0035633, 0.0041662, -0.0006029, 0.0006029
4: 0.0026282, 0.0054191, 0.0026282, 0.0054191, -0.0027909, 0.0027909
5: 0.0050201, 0.0064446, 0.0050201, 0.0064446, -0.0014245, 0.0014245
6: -0.0021963, -0.0008150, -0.0021963, -0.0008150, -0.0013813, 0.0013813
7: -0.0082640, -0.0074494, -0.0082640, -0.0074494, -0.0008146, 0.0008146
8: 0.0052524, 0.0098919, 0.0052524, 0.0098919, -0.0045621, 0.0045626
9: -0.0036845, -0.0030730, -0.0036845, -0.0030730, -0.0006115, 0.0006115

### Unstable ReLU Count (Linear/Conv2D Layers)
- layer_idx=0, type=LayerType.Linear, total=256, inp1_unstable=10, inp2_unstable=10, delta_unstable=250
- layer_idx=2, type=LayerType.Linear, total=256, inp1_unstable=8, inp2_unstable=8, delta_unstable=256
- layer_idx=4, type=LayerType.Linear, total=256, inp1_unstable=0, inp2_unstable=0, delta_unstable=256
- layer_idx=6, type=LayerType.Linear, total=256, inp1_unstable=2, inp2_unstable=2, delta_unstable=256
- layer_idx=8, type=LayerType.Linear, total=10, inp1_unstable=0, inp2_unstable=0, delta_unstable=10

Time for backsubstitution: 1.74 seconds

### RS candidates at layer 1
type: RSZ, layer: 1, pos: 198
type: RSZ, layer: 1, pos: 199
type: RSZ, layer: 1, pos: 204
type: RSZ, layer: 1, pos: 250

Time for candidate selection: 0.20 seconds

### Candidate
type: RSZ, layer: 1, pos: 198

### Relational analysis RSZ of RS_RSZ1_RSZ1_RSZ2_RSZ2_RSZ1_RSZ1_RSZ1

#### Relational analysis RSZ result of RS_RSZ1_RSZ1_RSZ2_RSZ2_RSZ1_RSZ1_RSZ1
Status: Status.UNKNOWN
Output dim: 1, lower bound: -0.0007828, upper bound: 0.0007828
time: 0.56 seconds

### Relational analysis RSZ of RS_RSZ1_RSZ1_RSZ2_RSZ2_RSZ1_RSZ1_RSZ2

#### Relational analysis RSZ result of RS_RSZ1_RSZ1_RSZ2_RSZ2_RSZ1_RSZ1_RSZ2
Status: Status.UNKNOWN
Output dim: 1, lower bound: -0.0007828, upper bound: 0.0007828
time: 0.56 seconds

## BFS RS instance: RS_RSZ1_RSZ1_RSZ2_RSZ2_RSZ1_RSZ2

### Backsubstitution after applying RS history:
0: 0.0006996, 0.0011177, 0.0006996, 0.0011177, -0.0004181, 0.0004181
1: 0.9934013, 0.9943976, 0.9934013, 0.9943976, -0.0009964, 0.0009964
2: -0.0088595, -0.0053283, -0.0088595, -0.0053283, -0.0032443, 0.0032399
3: 0.0035633, 0.0041662, 0.0035633, 0.0041662, -0.0006029, 0.0006029
4: 0.0026282, 0.0054191, 0.0026282, 0.0054191, -0.0027909, 0.0027909
5: 0.0050201, 0.0064446, 0.0050201, 0.0064446, -0.0014245, 0.0014245
6: -0.0021963, -0.0008150, -0.0021963, -0.0008150, -0.0013813, 0.0013813
7: -0.0082640, -0.0074494, -0.0082640, -0.0074494, -0.0008146, 0.0008146
8: 0.0052524, 0.0098919, 0.0052524, 0.0098919, -0.0045618, 0.0045629
9: -0.0036845, -0.0030730, -0.0036845, -0.0030730, -0.0006115, 0.0006115

### Unstable ReLU Count (Linear/Conv2D Layers)
- layer_idx=0, type=LayerType.Linear, total=256, inp1_unstable=10, inp2_unstable=10, delta_unstable=250
- layer_idx=2, type=LayerType.Linear, total=256, inp1_unstable=8, inp2_unstable=8, delta_unstable=256
- layer_idx=4, type=LayerType.Linear, total=256, inp1_unstable=0, inp2_unstable=0, delta_unstable=256
- layer_idx=6, type=LayerType.Linear, total=256, inp1_unstable=2, inp2_unstable=2, delta_unstable=256
- layer_idx=8, type=LayerType.Linear, total=10, inp1_unstable=0, inp2_unstable=0, delta_unstable=10

Time for backsubstitution: 1.72 seconds

### RS candidates at layer 1
type: RSZ, layer: 1, pos: 198
type: RSZ, layer: 1, pos: 199
type: RSZ, layer: 1, pos: 204
type: RSZ, layer: 1, pos: 250

Time for candidate selection: 0.20 seconds

### Candidate
type: RSZ, layer: 1, pos: 198

### Relational analysis RSZ of RS_RSZ1_RSZ1_RSZ2_RSZ2_RSZ1_RSZ2_RSZ1

#### Relational analysis RSZ result of RS_RSZ1_RSZ1_RSZ2_RSZ2_RSZ1_RSZ2_RSZ1
Status: Status.UNKNOWN
Output dim: 1, lower bound: -0.0007828, upper bound: 0.0007828
time: 0.57 seconds

### Relational analysis RSZ of RS_RSZ1_RSZ1_RSZ2_RSZ2_RSZ1_RSZ2_RSZ2

#### Relational analysis RSZ result of RS_RSZ1_RSZ1_RSZ2_RSZ2_RSZ1_RSZ2_RSZ2
Status: Status.UNKNOWN
Output dim: 1, lower bound: -0.0007828, upper bound: 0.0007828
time: 0.56 seconds

## BFS RS instance: RS_RSZ1_RSZ1_RSZ2_RSZ2_RSZ2_RSZ1

### Backsubstitution after applying RS history:
0: 0.0006996, 0.0011177, 0.0006996, 0.0011177, -0.0004181, 0.0004181
1: 0.9934013, 0.9943976, 0.9934013, 0.9943976, -0.0009964, 0.0009964
2: -0.0088595, -0.0053283, -0.0088595, -0.0053283, -0.0032410, 0.0032434
3: 0.0035633, 0.0041662, 0.0035633, 0.0041662, -0.0006029, 0.0006029
4: 0.0026282, 0.0054191, 0.0026282, 0.0054191, -0.0027909, 0.0027909
5: 0.0050201, 0.0064446, 0.0050201, 0.0064446, -0.0014245, 0.0014245
6: -0.0021963, -0.0008150, -0.0021963, -0.0008150, -0.0013813, 0.0013813
7: -0.0082640, -0.0074494, -0.0082640, -0.0074494, -0.0008146, 0.0008146
8: 0.0052524, 0.0098919, 0.0052524, 0.0098919, -0.0045627, 0.0045620
9: -0.0036845, -0.0030730, -0.0036845, -0.0030730, -0.0006115, 0.0006115

### Unstable ReLU Count (Linear/Conv2D Layers)
- layer_idx=0, type=LayerType.Linear, total=256, inp1_unstable=10, inp2_unstable=10, delta_unstable=250
- layer_idx=2, type=LayerType.Linear, total=256, inp1_unstable=8, inp2_unstable=8, delta_unstable=256
- layer_idx=4, type=LayerType.Linear, total=256, inp1_unstable=0, inp2_unstable=0, delta_unstable=256
- layer_idx=6, type=LayerType.Linear, total=256, inp1_unstable=2, inp2_unstable=2, delta_unstable=256
- layer_idx=8, type=LayerType.Linear, total=10, inp1_unstable=0, inp2_unstable=0, delta_unstable=10

Time for backsubstitution: 1.62 seconds

### RS candidates at layer 1
type: RSZ, layer: 1, pos: 198
type: RSZ, layer: 1, pos: 199
type: RSZ, layer: 1, pos: 204
type: RSZ, layer: 1, pos: 250

Time for candidate selection: 0.18 seconds

### Candidate
type: RSZ, layer: 1, pos: 198

### Relational analysis RSZ of RS_RSZ1_RSZ1_RSZ2_RSZ2_RSZ2_RSZ1_RSZ1

#### Relational analysis RSZ result of RS_RSZ1_RSZ1_RSZ2_RSZ2_RSZ2_RSZ1_RSZ1
Status: Status.UNKNOWN
Output dim: 1, lower bound: -0.0007828, upper bound: 0.0007828
time: 0.63 seconds

### Relational analysis RSZ of RS_RSZ1_RSZ1_RSZ2_RSZ2_RSZ2_RSZ1_RSZ2

#### Relational analysis RSZ result of RS_RSZ1_RSZ1_RSZ2_RSZ2_RSZ2_RSZ1_RSZ2
Status: Status.UNKNOWN
Output dim: 1, lower bound: -0.0007828, upper bound: 0.0007828
time: 0.64 seconds

## BFS RS instance: RS_RSZ1_RSZ1_RSZ2_RSZ2_RSZ2_RSZ2

### Backsubstitution after applying RS history:
0: 0.0006996, 0.0011177, 0.0006996, 0.0011177, -0.0004181, 0.0004181
1: 0.9934013, 0.9943976, 0.9934013, 0.9943976, -0.0009964, 0.0009964
2: -0.0088595, -0.0053283, -0.0088595, -0.0053283, -0.0032422, 0.0032418
3: 0.0035633, 0.0041662, 0.0035633, 0.0041662, -0.0006029, 0.0006029
4: 0.0026282, 0.0054191, 0.0026282, 0.0054191, -0.0027909, 0.0027909
5: 0.0050201, 0.0064446, 0.0050201, 0.0064446, -0.0014245, 0.0014245
6: -0.0021963, -0.0008150, -0.0021963, -0.0008150, -0.0013813, 0.0013813
7: -0.0082640, -0.0074494, -0.0082640, -0.0074494, -0.0008146, 0.0008146
8: 0.0052524, 0.0098919, 0.0052524, 0.0098919, -0.0045623, 0.0045624
9: -0.0036845, -0.0030730, -0.0036845, -0.0030730, -0.0006115, 0.0006115

### Unstable ReLU Count (Linear/Conv2D Layers)
- layer_idx=0, type=LayerType.Linear, total=256, inp1_unstable=10, inp2_unstable=10, delta_unstable=250
- layer_idx=2, type=LayerType.Linear, total=256, inp1_unstable=8, inp2_unstable=8, delta_unstable=256
- layer_idx=4, type=LayerType.Linear, total=256, inp1_unstable=0, inp2_unstable=0, delta_unstable=256
- layer_idx=6, type=LayerType.Linear, total=256, inp1_unstable=2, inp2_unstable=2, delta_unstable=256
- layer_idx=8, type=LayerType.Linear, total=10, inp1_unstable=0, inp2_unstable=0, delta_unstable=10

Time for backsubstitution: 1.63 seconds

### RS candidates at layer 1
type: RSZ, layer: 1, pos: 198
type: RSZ, layer: 1, pos: 199
type: RSZ, layer: 1, pos: 204
type: RSZ, layer: 1, pos: 250

Time for candidate selection: 0.17 seconds

### Candidate
type: RSZ, layer: 1, pos: 198

### Relational analysis RSZ of RS_RSZ1_RSZ1_RSZ2_RSZ2_RSZ2_RSZ2_RSZ1

#### Relational analysis RSZ result of RS_RSZ1_RSZ1_RSZ2_RSZ2_RSZ2_RSZ2_RSZ1
Status: Status.UNKNOWN
Output dim: 1, lower bound: -0.0007828, upper bound: 0.0007828
time: 0.62 seconds

### Relational analysis RSZ of RS_RSZ1_RSZ1_RSZ2_RSZ2_RSZ2_RSZ2_RSZ2

#### Relational analysis RSZ result of RS_RSZ1_RSZ1_RSZ2_RSZ2_RSZ2_RSZ2_RSZ2
Status: Status.UNKNOWN
Output dim: 1, lower bound: -0.0007828, upper bound: 0.0007828
time: 0.62 seconds

## BFS RS instance: RS_RSZ1_RSZ2_RSZ1_RSZ1_RSZ1_RSZ1

### Backsubstitution after applying RS history:
0: 0.0006996, 0.0011177, 0.0006996, 0.0011177, -0.0004181, 0.0004181
1: 0.9934013, 0.9943976, 0.9934013, 0.9943976, -0.0009964, 0.0009964
2: -0.0088595, -0.0053283, -0.0088595, -0.0053283, -0.0032420, 0.0032423
3: 0.0035633, 0.0041662, 0.0035633, 0.0041662, -0.0006029, 0.0006029
4: 0.0026282, 0.0054191, 0.0026282, 0.0054191, -0.0027909, 0.0027909
5: 0.0050201, 0.0064446, 0.0050201, 0.0064446, -0.0014245, 0.0014245
6: -0.0021963, -0.0008150, -0.0021963, -0.0008150, -0.0013813, 0.0013813
7: -0.0082640, -0.0074494, -0.0082640, -0.0074494, -0.0008146, 0.0008146
8: 0.0052524, 0.0098919, 0.0052524, 0.0098919, -0.0045624, 0.0045623
9: -0.0036845, -0.0030730, -0.0036845, -0.0030730, -0.0006115, 0.0006115

### Unstable ReLU Count (Linear/Conv2D Layers)
- layer_idx=0, type=LayerType.Linear, total=256, inp1_unstable=10, inp2_unstable=10, delta_unstable=250
- layer_idx=2, type=LayerType.Linear, total=256, inp1_unstable=8, inp2_unstable=8, delta_unstable=256
- layer_idx=4, type=LayerType.Linear, total=256, inp1_unstable=0, inp2_unstable=0, delta_unstable=256
- layer_idx=6, type=LayerType.Linear, total=256, inp1_unstable=2, inp2_unstable=2, delta_unstable=256
- layer_idx=8, type=LayerType.Linear, total=10, inp1_unstable=0, inp2_unstable=0, delta_unstable=10

Time for backsubstitution: 1.67 seconds

### RS candidates at layer 1
type: RSZ, layer: 1, pos: 198
type: RSZ, layer: 1, pos: 199
type: RSZ, layer: 1, pos: 204
type: RSZ, layer: 1, pos: 250

Time for candidate selection: 0.17 seconds

### Candidate
type: RSZ, layer: 1, pos: 198

### Relational analysis RSZ of RS_RSZ1_RSZ2_RSZ1_RSZ1_RSZ1_RSZ1_RSZ1

#### Relational analysis RSZ result of RS_RSZ1_RSZ2_RSZ1_RSZ1_RSZ1_RSZ1_RSZ1
Status: Status.UNKNOWN
Output dim: 1, lower bound: -0.0007828, upper bound: 0.0007828
time: 0.55 seconds

### Relational analysis RSZ of RS_RSZ1_RSZ2_RSZ1_RSZ1_RSZ1_RSZ1_RSZ2

#### Relational analysis RSZ result of RS_RSZ1_RSZ2_RSZ1_RSZ1_RSZ1_RSZ1_RSZ2
Status: Status.UNKNOWN
Output dim: 1, lower bound: -0.0007828, upper bound: 0.0007828
time: 0.60 seconds

## BFS RS instance: RS_RSZ1_RSZ2_RSZ1_RSZ1_RSZ1_RSZ2

### Backsubstitution after applying RS history:
0: 0.0006996, 0.0011177, 0.0006996, 0.0011177, -0.0004181, 0.0004181
1: 0.9934013, 0.9943976, 0.9934013, 0.9943976, -0.0009964, 0.0009964
2: -0.0088595, -0.0053283, -0.0088595, -0.0053283, -0.0032436, 0.0032415
3: 0.0035633, 0.0041662, 0.0035633, 0.0041662, -0.0006029, 0.0006029
4: 0.0026282, 0.0054191, 0.0026282, 0.0054191, -0.0027909, 0.0027909
5: 0.0050201, 0.0064446, 0.0050201, 0.0064446, -0.0014245, 0.0014245
6: -0.0021963, -0.0008150, -0.0021963, -0.0008150, -0.0013813, 0.0013813
7: -0.0082640, -0.0074494, -0.0082640, -0.0074494, -0.0008146, 0.0008146
8: 0.0052524, 0.0098919, 0.0052524, 0.0098919, -0.0045622, 0.0045627
9: -0.0036845, -0.0030730, -0.0036845, -0.0030730, -0.0006115, 0.0006115

### Unstable ReLU Count (Linear/Conv2D Layers)
- layer_idx=0, type=LayerType.Linear, total=256, inp1_unstable=10, inp2_unstable=10, delta_unstable=250
- layer_idx=2, type=LayerType.Linear, total=256, inp1_unstable=8, inp2_unstable=8, delta_unstable=256
- layer_idx=4, type=LayerType.Linear, total=256, inp1_unstable=0, inp2_unstable=0, delta_unstable=256
- layer_idx=6, type=LayerType.Linear, total=256, inp1_unstable=2, inp2_unstable=2, delta_unstable=256
- layer_idx=8, type=LayerType.Linear, total=10, inp1_unstable=0, inp2_unstable=0, delta_unstable=10

Time for backsubstitution: 1.67 seconds

### RS candidates at layer 1
type: RSZ, layer: 1, pos: 198
type: RSZ, layer: 1, pos: 199
type: RSZ, layer: 1, pos: 204
type: RSZ, layer: 1, pos: 250

Time for candidate selection: 0.17 seconds

### Candidate
type: RSZ, layer: 1, pos: 198

### Relational analysis RSZ of RS_RSZ1_RSZ2_RSZ1_RSZ1_RSZ1_RSZ2_RSZ1

#### Relational analysis RSZ result of RS_RSZ1_RSZ2_RSZ1_RSZ1_RSZ1_RSZ2_RSZ1
Status: Status.UNKNOWN
Output dim: 1, lower bound: -0.0007828, upper bound: 0.0007828
time: 0.57 seconds

### Relational analysis RSZ of RS_RSZ1_RSZ2_RSZ1_RSZ1_RSZ1_RSZ2_RSZ2

#### Relational analysis RSZ result of RS_RSZ1_RSZ2_RSZ1_RSZ1_RSZ1_RSZ2_RSZ2
Status: Status.UNKNOWN
Output dim: 1, lower bound: -0.0007828, upper bound: 0.0007828
time: 0.56 seconds

## BFS RS instance: RS_RSZ1_RSZ2_RSZ1_RSZ1_RSZ2_RSZ1

### Backsubstitution after applying RS history:
0: 0.0006996, 0.0011177, 0.0006996, 0.0011177, -0.0004181, 0.0004181
1: 0.9934013, 0.9943976, 0.9934013, 0.9943976, -0.0009964, 0.0009964
2: -0.0088595, -0.0053283, -0.0088595, -0.0053283, -0.0032399, 0.0032444
3: 0.0035633, 0.0041662, 0.0035633, 0.0041662, -0.0006029, 0.0006029
4: 0.0026282, 0.0054191, 0.0026282, 0.0054191, -0.0027909, 0.0027909
5: 0.0050201, 0.0064446, 0.0050201, 0.0064446, -0.0014245, 0.0014245
6: -0.0021963, -0.0008150, -0.0021963, -0.0008150, -0.0013813, 0.0013813
7: -0.0082640, -0.0074494, -0.0082640, -0.0074494, -0.0008146, 0.0008146
8: 0.0052524, 0.0098919, 0.0052524, 0.0098919, -0.0045629, 0.0045618
9: -0.0036845, -0.0030730, -0.0036845, -0.0030730, -0.0006115, 0.0006115

### Unstable ReLU Count (Linear/Conv2D Layers)
- layer_idx=0, type=LayerType.Linear, total=256, inp1_unstable=10, inp2_unstable=10, delta_unstable=250
- layer_idx=2, type=LayerType.Linear, total=256, inp1_unstable=8, inp2_unstable=8, delta_unstable=256
- layer_idx=4, type=LayerType.Linear, total=256, inp1_unstable=0, inp2_unstable=0, delta_unstable=256
- layer_idx=6, type=LayerType.Linear, total=256, inp1_unstable=2, inp2_unstable=2, delta_unstable=256
- layer_idx=8, type=LayerType.Linear, total=10, inp1_unstable=0, inp2_unstable=0, delta_unstable=10

Time for backsubstitution: 1.65 seconds

### RS candidates at layer 1
type: RSZ, layer: 1, pos: 198
type: RSZ, layer: 1, pos: 199
type: RSZ, layer: 1, pos: 204
type: RSZ, layer: 1, pos: 250

Time for candidate selection: 0.18 seconds

### Candidate
type: RSZ, layer: 1, pos: 198

### Relational analysis RSZ of RS_RSZ1_RSZ2_RSZ1_RSZ1_RSZ2_RSZ1_RSZ1

#### Relational analysis RSZ result of RS_RSZ1_RSZ2_RSZ1_RSZ1_RSZ2_RSZ1_RSZ1
Status: Status.UNKNOWN
Output dim: 1, lower bound: -0.0007828, upper bound: 0.0007828
time: 0.60 seconds

### Relational analysis RSZ of RS_RSZ1_RSZ2_RSZ1_RSZ1_RSZ2_RSZ1_RSZ2

#### Relational analysis RSZ result of RS_RSZ1_RSZ2_RSZ1_RSZ1_RSZ2_RSZ1_RSZ2
Status: Status.UNKNOWN
Output dim: 1, lower bound: -0.0007828, upper bound: 0.0007828
time: 0.62 seconds

## BFS RS instance: RS_RSZ1_RSZ2_RSZ1_RSZ1_RSZ2_RSZ2

### Backsubstitution after applying RS history:
0: 0.0006996, 0.0011177, 0.0006996, 0.0011177, -0.0004181, 0.0004181
1: 0.9934013, 0.9943976, 0.9934013, 0.9943976, -0.0009964, 0.0009964
2: -0.0088595, -0.0053283, -0.0088595, -0.0053283, -0.0032413, 0.0032435
3: 0.0035633, 0.0041662, 0.0035633, 0.0041662, -0.0006029, 0.0006029
4: 0.0026282, 0.0054191, 0.0026282, 0.0054191, -0.0027909, 0.0027909
5: 0.0050201, 0.0064446, 0.0050201, 0.0064446, -0.0014245, 0.0014245
6: -0.0021963, -0.0008150, -0.0021963, -0.0008150, -0.0013813, 0.0013813
7: -0.0082640, -0.0074494, -0.0082640, -0.0074494, -0.0008146, 0.0008146
8: 0.0052524, 0.0098919, 0.0052524, 0.0098919, -0.0045627, 0.0045621
9: -0.0036845, -0.0030730, -0.0036845, -0.0030730, -0.0006115, 0.0006115

### Unstable ReLU Count (Linear/Conv2D Layers)
- layer_idx=0, type=LayerType.Linear, total=256, inp1_unstable=10, inp2_unstable=10, delta_unstable=250
- layer_idx=2, type=LayerType.Linear, total=256, inp1_unstable=8, inp2_unstable=8, delta_unstable=256
- layer_idx=4, type=LayerType.Linear, total=256, inp1_unstable=0, inp2_unstable=0, delta_unstable=256
- layer_idx=6, type=LayerType.Linear, total=256, inp1_unstable=2, inp2_unstable=2, delta_unstable=256
- layer_idx=8, type=LayerType.Linear, total=10, inp1_unstable=0, inp2_unstable=0, delta_unstable=10

Time for backsubstitution: 1.71 seconds

### RS candidates at layer 1
type: RSZ, layer: 1, pos: 198
type: RSZ, layer: 1, pos: 199
type: RSZ, layer: 1, pos: 204
type: RSZ, layer: 1, pos: 250

Time for candidate selection: 0.19 seconds

### Candidate
type: RSZ, layer: 1, pos: 198

### Relational analysis RSZ of RS_RSZ1_RSZ2_RSZ1_RSZ1_RSZ2_RSZ2_RSZ1

#### Relational analysis RSZ result of RS_RSZ1_RSZ2_RSZ1_RSZ1_RSZ2_RSZ2_RSZ1
Status: Status.UNKNOWN
Output dim: 1, lower bound: -0.0007828, upper bound: 0.0007828
time: 0.59 seconds

### Relational analysis RSZ of RS_RSZ1_RSZ2_RSZ1_RSZ1_RSZ2_RSZ2_RSZ2

#### Relational analysis RSZ result of RS_RSZ1_RSZ2_RSZ1_RSZ1_RSZ2_RSZ2_RSZ2
Status: Status.UNKNOWN
Output dim: 1, lower bound: -0.0007828, upper bound: 0.0007828
time: 0.59 seconds

## BFS RS instance: RS_RSZ1_RSZ2_RSZ1_RSZ2_RSZ1_RSZ1

### Backsubstitution after applying RS history:
0: 0.0006996, 0.0011177, 0.0006996, 0.0011177, -0.0004181, 0.0004181
1: 0.9934013, 0.9943976, 0.9934013, 0.9943976, -0.0009964, 0.0009964
2: -0.0088595, -0.0053283, -0.0088595, -0.0053283, -0.0032407, 0.0032435
3: 0.0035633, 0.0041662, 0.0035633, 0.0041662, -0.0006029, 0.0006029
4: 0.0026282, 0.0054191, 0.0026282, 0.0054191, -0.0027909, 0.0027909
5: 0.0050201, 0.0064446, 0.0050201, 0.0064446, -0.0014245, 0.0014245
6: -0.0021963, -0.0008150, -0.0021963, -0.0008150, -0.0013813, 0.0013813
7: -0.0082640, -0.0074494, -0.0082640, -0.0074494, -0.0008146, 0.0008146
8: 0.0052524, 0.0098919, 0.0052524, 0.0098919, -0.0045627, 0.0045620
9: -0.0036845, -0.0030730, -0.0036845, -0.0030730, -0.0006115, 0.0006115

### Unstable ReLU Count (Linear/Conv2D Layers)
- layer_idx=0, type=LayerType.Linear, total=256, inp1_unstable=10, inp2_unstable=10, delta_unstable=250
- layer_idx=2, type=LayerType.Linear, total=256, inp1_unstable=8, inp2_unstable=8, delta_unstable=256
- layer_idx=4, type=LayerType.Linear, total=256, inp1_unstable=0, inp2_unstable=0, delta_unstable=256
- layer_idx=6, type=LayerType.Linear, total=256, inp1_unstable=2, inp2_unstable=2, delta_unstable=256
- layer_idx=8, type=LayerType.Linear, total=10, inp1_unstable=0, inp2_unstable=0, delta_unstable=10

Time for backsubstitution: 1.74 seconds

### RS candidates at layer 1
type: RSZ, layer: 1, pos: 198
type: RSZ, layer: 1, pos: 199
type: RSZ, layer: 1, pos: 204
type: RSZ, layer: 1, pos: 250

Time for candidate selection: 0.19 seconds

### Candidate
type: RSZ, layer: 1, pos: 198

### Relational analysis RSZ of RS_RSZ1_RSZ2_RSZ1_RSZ2_RSZ1_RSZ1_RSZ1

#### Relational analysis RSZ result of RS_RSZ1_RSZ2_RSZ1_RSZ2_RSZ1_RSZ1_RSZ1
Status: Status.UNKNOWN
Output dim: 1, lower bound: -0.0007828, upper bound: 0.0007828
time: 0.59 seconds

### Relational analysis RSZ of RS_RSZ1_RSZ2_RSZ1_RSZ2_RSZ1_RSZ1_RSZ2

#### Relational analysis RSZ result of RS_RSZ1_RSZ2_RSZ1_RSZ2_RSZ1_RSZ1_RSZ2
Status: Status.UNKNOWN
Output dim: 1, lower bound: -0.0007828, upper bound: 0.0007828
time: 0.58 seconds

## BFS RS instance: RS_RSZ1_RSZ2_RSZ1_RSZ2_RSZ1_RSZ2

### Backsubstitution after applying RS history:
0: 0.0006996, 0.0011177, 0.0006996, 0.0011177, -0.0004181, 0.0004181
1: 0.9934013, 0.9943976, 0.9934013, 0.9943976, -0.0009964, 0.0009964
2: -0.0088595, -0.0053283, -0.0088595, -0.0053283, -0.0032422, 0.0032428
3: 0.0035633, 0.0041662, 0.0035633, 0.0041662, -0.0006029, 0.0006029
4: 0.0026282, 0.0054191, 0.0026282, 0.0054191, -0.0027909, 0.0027909
5: 0.0050201, 0.0064446, 0.0050201, 0.0064446, -0.0014245, 0.0014245
6: -0.0021963, -0.0008150, -0.0021963, -0.0008150, -0.0013813, 0.0013813
7: -0.0082640, -0.0074494, -0.0082640, -0.0074494, -0.0008146, 0.0008146
8: 0.0052524, 0.0098919, 0.0052524, 0.0098919, -0.0045625, 0.0045624
9: -0.0036845, -0.0030730, -0.0036845, -0.0030730, -0.0006115, 0.0006115

### Unstable ReLU Count (Linear/Conv2D Layers)
- layer_idx=0, type=LayerType.Linear, total=256, inp1_unstable=10, inp2_unstable=10, delta_unstable=250
- layer_idx=2, type=LayerType.Linear, total=256, inp1_unstable=8, inp2_unstable=8, delta_unstable=256
- layer_idx=4, type=LayerType.Linear, total=256, inp1_unstable=0, inp2_unstable=0, delta_unstable=256
- layer_idx=6, type=LayerType.Linear, total=256, inp1_unstable=2, inp2_unstable=2, delta_unstable=256
- layer_idx=8, type=LayerType.Linear, total=10, inp1_unstable=0, inp2_unstable=0, delta_unstable=10

Time for backsubstitution: 1.75 seconds

### RS candidates at layer 1
type: RSZ, layer: 1, pos: 198
type: RSZ, layer: 1, pos: 199
type: RSZ, layer: 1, pos: 204
type: RSZ, layer: 1, pos: 250

Time for candidate selection: 0.21 seconds

### Candidate
type: RSZ, layer: 1, pos: 198

### Relational analysis RSZ of RS_RSZ1_RSZ2_RSZ1_RSZ2_RSZ1_RSZ2_RSZ1

#### Relational analysis RSZ result of RS_RSZ1_RSZ2_RSZ1_RSZ2_RSZ1_RSZ2_RSZ1
Status: Status.UNKNOWN
Output dim: 1, lower bound: -0.0007828, upper bound: 0.0007828
time: 0.58 seconds

### Relational analysis RSZ of RS_RSZ1_RSZ2_RSZ1_RSZ2_RSZ1_RSZ2_RSZ2

#### Relational analysis RSZ result of RS_RSZ1_RSZ2_RSZ1_RSZ2_RSZ1_RSZ2_RSZ2
Status: Status.UNKNOWN
Output dim: 1, lower bound: -0.0007828, upper bound: 0.0007828
time: 0.58 seconds

## BFS RS instance: RS_RSZ1_RSZ2_RSZ1_RSZ2_RSZ2_RSZ1

### Backsubstitution after applying RS history:
0: 0.0006996, 0.0011177, 0.0006996, 0.0011177, -0.0004181, 0.0004181
1: 0.9934013, 0.9943976, 0.9934013, 0.9943976, -0.0009964, 0.0009964
2: -0.0088595, -0.0053283, -0.0088595, -0.0053283, -0.0032386, 0.0032451
3: 0.0035633, 0.0041662, 0.0035633, 0.0041662, -0.0006029, 0.0006029
4: 0.0026282, 0.0054191, 0.0026282, 0.0054191, -0.0027909, 0.0027909
5: 0.0050201, 0.0064446, 0.0050201, 0.0064446, -0.0014245, 0.0014245
6: -0.0021963, -0.0008150, -0.0021963, -0.0008150, -0.0013813, 0.0013813
7: -0.0082640, -0.0074494, -0.0082640, -0.0074494, -0.0008146, 0.0008146
8: 0.0052524, 0.0098919, 0.0052524, 0.0098919, -0.0045631, 0.0045614
9: -0.0036845, -0.0030730, -0.0036845, -0.0030730, -0.0006115, 0.0006115

### Unstable ReLU Count (Linear/Conv2D Layers)
- layer_idx=0, type=LayerType.Linear, total=256, inp1_unstable=10, inp2_unstable=10, delta_unstable=250
- layer_idx=2, type=LayerType.Linear, total=256, inp1_unstable=8, inp2_unstable=8, delta_unstable=256
- layer_idx=4, type=LayerType.Linear, total=256, inp1_unstable=0, inp2_unstable=0, delta_unstable=256
- layer_idx=6, type=LayerType.Linear, total=256, inp1_unstable=2, inp2_unstable=2, delta_unstable=256
- layer_idx=8, type=LayerType.Linear, total=10, inp1_unstable=0, inp2_unstable=0, delta_unstable=10

Time for backsubstitution: 1.65 seconds

### RS candidates at layer 1
type: RSZ, layer: 1, pos: 198
type: RSZ, layer: 1, pos: 199
type: RSZ, layer: 1, pos: 204
type: RSZ, layer: 1, pos: 250

Time for candidate selection: 0.18 seconds

### Candidate
type: RSZ, layer: 1, pos: 198

### Relational analysis RSZ of RS_RSZ1_RSZ2_RSZ1_RSZ2_RSZ2_RSZ1_RSZ1

#### Relational analysis RSZ result of RS_RSZ1_RSZ2_RSZ1_RSZ2_RSZ2_RSZ1_RSZ1
Status: Status.UNKNOWN
Output dim: 1, lower bound: -0.0007828, upper bound: 0.0007828
time: 0.65 seconds

### Relational analysis RSZ of RS_RSZ1_RSZ2_RSZ1_RSZ2_RSZ2_RSZ1_RSZ2

#### Relational analysis RSZ result of RS_RSZ1_RSZ2_RSZ1_RSZ2_RSZ2_RSZ1_RSZ2
Status: Status.UNKNOWN
Output dim: 1, lower bound: -0.0007828, upper bound: 0.0007828
time: 0.66 seconds

## BFS RS instance: RS_RSZ1_RSZ2_RSZ1_RSZ2_RSZ2_RSZ2

### Backsubstitution after applying RS history:
0: 0.0006996, 0.0011177, 0.0006996, 0.0011177, -0.0004181, 0.0004181
1: 0.9934013, 0.9943976, 0.9934013, 0.9943976, -0.0009964, 0.0009964
2: -0.0088595, -0.0053283, -0.0088595, -0.0053283, -0.0032399, 0.0032442
3: 0.0035633, 0.0041662, 0.0035633, 0.0041662, -0.0006029, 0.0006029
4: 0.0026282, 0.0054191, 0.0026282, 0.0054191, -0.0027909, 0.0027909
5: 0.0050201, 0.0064446, 0.0050201, 0.0064446, -0.0014245, 0.0014245
6: -0.0021963, -0.0008150, -0.0021963, -0.0008150, -0.0013813, 0.0013813
7: -0.0082640, -0.0074494, -0.0082640, -0.0074494, -0.0008146, 0.0008146
8: 0.0052524, 0.0098919, 0.0052524, 0.0098919, -0.0045629, 0.0045618
9: -0.0036845, -0.0030730, -0.0036845, -0.0030730, -0.0006115, 0.0006115

### Unstable ReLU Count (Linear/Conv2D Layers)
- layer_idx=0, type=LayerType.Linear, total=256, inp1_unstable=10, inp2_unstable=10, delta_unstable=250
- layer_idx=2, type=LayerType.Linear, total=256, inp1_unstable=8, inp2_unstable=8, delta_unstable=256
- layer_idx=4, type=LayerType.Linear, total=256, inp1_unstable=0, inp2_unstable=0, delta_unstable=256
- layer_idx=6, type=LayerType.Linear, total=256, inp1_unstable=2, inp2_unstable=2, delta_unstable=256
- layer_idx=8, type=LayerType.Linear, total=10, inp1_unstable=0, inp2_unstable=0, delta_unstable=10

Time for backsubstitution: 1.78 seconds

### RS candidates at layer 1
type: RSZ, layer: 1, pos: 198
type: RSZ, layer: 1, pos: 199
type: RSZ, layer: 1, pos: 204
type: RSZ, layer: 1, pos: 250

Time for candidate selection: 0.19 seconds

### Candidate
type: RSZ, layer: 1, pos: 198

### Relational analysis RSZ of RS_RSZ1_RSZ2_RSZ1_RSZ2_RSZ2_RSZ2_RSZ1

#### Relational analysis RSZ result of RS_RSZ1_RSZ2_RSZ1_RSZ2_RSZ2_RSZ2_RSZ1
Status: Status.UNKNOWN
Output dim: 1, lower bound: -0.0007828, upper bound: 0.0007828
time: 0.58 seconds

### Relational analysis RSZ of RS_RSZ1_RSZ2_RSZ1_RSZ2_RSZ2_RSZ2_RSZ2

#### Relational analysis RSZ result of RS_RSZ1_RSZ2_RSZ1_RSZ2_RSZ2_RSZ2_RSZ2
Status: Status.UNKNOWN
Output dim: 1, lower bound: -0.0007828, upper bound: 0.0007828
time: 0.59 seconds

## BFS RS instance: RS_RSZ1_RSZ2_RSZ2_RSZ1_RSZ1_RSZ1

### Backsubstitution after applying RS history:
0: 0.0006996, 0.0011177, 0.0006996, 0.0011177, -0.0004181, 0.0004181
1: 0.9934013, 0.9943976, 0.9934013, 0.9943976, -0.0009964, 0.0009964
2: -0.0088595, -0.0053283, -0.0088595, -0.0053283, -0.0032419, 0.0032426
3: 0.0035633, 0.0041662, 0.0035633, 0.0041662, -0.0006029, 0.0006029
4: 0.0026282, 0.0054191, 0.0026282, 0.0054191, -0.0027909, 0.0027909
5: 0.0050201, 0.0064446, 0.0050201, 0.0064446, -0.0014245, 0.0014245
6: -0.0021963, -0.0008150, -0.0021963, -0.0008150, -0.0013813, 0.0013813
7: -0.0082640, -0.0074494, -0.0082640, -0.0074494, -0.0008146, 0.0008146
8: 0.0052524, 0.0098919, 0.0052524, 0.0098919, -0.0045625, 0.0045623
9: -0.0036845, -0.0030730, -0.0036845, -0.0030730, -0.0006115, 0.0006115

### Unstable ReLU Count (Linear/Conv2D Layers)
- layer_idx=0, type=LayerType.Linear, total=256, inp1_unstable=10, inp2_unstable=10, delta_unstable=250
- layer_idx=2, type=LayerType.Linear, total=256, inp1_unstable=8, inp2_unstable=8, delta_unstable=256
- layer_idx=4, type=LayerType.Linear, total=256, inp1_unstable=0, inp2_unstable=0, delta_unstable=256
- layer_idx=6, type=LayerType.Linear, total=256, inp1_unstable=2, inp2_unstable=2, delta_unstable=256
- layer_idx=8, type=LayerType.Linear, total=10, inp1_unstable=0, inp2_unstable=0, delta_unstable=10

Time for backsubstitution: 1.72 seconds

### RS candidates at layer 1
type: RSZ, layer: 1, pos: 198
type: RSZ, layer: 1, pos: 199
type: RSZ, layer: 1, pos: 204
type: RSZ, layer: 1, pos: 250

Time for candidate selection: 0.20 seconds

### Candidate
type: RSZ, layer: 1, pos: 198

### Relational analysis RSZ of RS_RSZ1_RSZ2_RSZ2_RSZ1_RSZ1_RSZ1_RSZ1

#### Relational analysis RSZ result of RS_RSZ1_RSZ2_RSZ2_RSZ1_RSZ1_RSZ1_RSZ1
Status: Status.UNKNOWN
Output dim: 1, lower bound: -0.0007828, upper bound: 0.0007828
time: 0.62 seconds

### Relational analysis RSZ of RS_RSZ1_RSZ2_RSZ2_RSZ1_RSZ1_RSZ1_RSZ2

#### Relational analysis RSZ result of RS_RSZ1_RSZ2_RSZ2_RSZ1_RSZ1_RSZ1_RSZ2
Status: Status.UNKNOWN
Output dim: 1, lower bound: -0.0007828, upper bound: 0.0007828
time: 0.62 seconds

## BFS RS instance: RS_RSZ1_RSZ2_RSZ2_RSZ1_RSZ1_RSZ2

### Backsubstitution after applying RS history:
0: 0.0006996, 0.0011177, 0.0006996, 0.0011177, -0.0004181, 0.0004181
1: 0.9934013, 0.9943976, 0.9934013, 0.9943976, -0.0009964, 0.0009964
2: -0.0088595, -0.0053283, -0.0088595, -0.0053283, -0.0032433, 0.0032419
3: 0.0035633, 0.0041662, 0.0035633, 0.0041662, -0.0006029, 0.0006029
4: 0.0026282, 0.0054191, 0.0026282, 0.0054191, -0.0027909, 0.0027909
5: 0.0050201, 0.0064446, 0.0050201, 0.0064446, -0.0014245, 0.0014245
6: -0.0021963, -0.0008150, -0.0021963, -0.0008150, -0.0013813, 0.0013813
7: -0.0082640, -0.0074494, -0.0082640, -0.0074494, -0.0008146, 0.0008146
8: 0.0052524, 0.0098919, 0.0052524, 0.0098919, -0.0045623, 0.0045627
9: -0.0036845, -0.0030730, -0.0036845, -0.0030730, -0.0006115, 0.0006115

### Unstable ReLU Count (Linear/Conv2D Layers)
- layer_idx=0, type=LayerType.Linear, total=256, inp1_unstable=10, inp2_unstable=10, delta_unstable=250
- layer_idx=2, type=LayerType.Linear, total=256, inp1_unstable=8, inp2_unstable=8, delta_unstable=256
- layer_idx=4, type=LayerType.Linear, total=256, inp1_unstable=0, inp2_unstable=0, delta_unstable=256
- layer_idx=6, type=LayerType.Linear, total=256, inp1_unstable=2, inp2_unstable=2, delta_unstable=256
- layer_idx=8, type=LayerType.Linear, total=10, inp1_unstable=0, inp2_unstable=0, delta_unstable=10

Time for backsubstitution: 1.70 seconds

### RS candidates at layer 1
type: RSZ, layer: 1, pos: 198
type: RSZ, layer: 1, pos: 199
type: RSZ, layer: 1, pos: 204
type: RSZ, layer: 1, pos: 250

Time for candidate selection: 0.19 seconds

### Candidate
type: RSZ, layer: 1, pos: 198

### Relational analysis RSZ of RS_RSZ1_RSZ2_RSZ2_RSZ1_RSZ1_RSZ2_RSZ1

#### Relational analysis RSZ result of RS_RSZ1_RSZ2_RSZ2_RSZ1_RSZ1_RSZ2_RSZ1
Status: Status.UNKNOWN
Output dim: 1, lower bound: -0.0007828, upper bound: 0.0007828
time: 0.61 seconds

### Relational analysis RSZ of RS_RSZ1_RSZ2_RSZ2_RSZ1_RSZ1_RSZ2_RSZ2

#### Relational analysis RSZ result of RS_RSZ1_RSZ2_RSZ2_RSZ1_RSZ1_RSZ2_RSZ2
Status: Status.UNKNOWN
Output dim: 1, lower bound: -0.0007828, upper bound: 0.0007828
time: 0.61 seconds

## BFS RS instance: RS_RSZ1_RSZ2_RSZ2_RSZ1_RSZ2_RSZ1

### Backsubstitution after applying RS history:
0: 0.0006996, 0.0011177, 0.0006996, 0.0011177, -0.0004181, 0.0004181
1: 0.9934013, 0.9943976, 0.9934013, 0.9943976, -0.0009964, 0.0009964
2: -0.0088595, -0.0053283, -0.0088595, -0.0053283, -0.0032398, 0.0032446
3: 0.0035633, 0.0041662, 0.0035633, 0.0041662, -0.0006029, 0.0006029
4: 0.0026282, 0.0054191, 0.0026282, 0.0054191, -0.0027909, 0.0027909
5: 0.0050201, 0.0064446, 0.0050201, 0.0064446, -0.0014245, 0.0014245
6: -0.0021963, -0.0008150, -0.0021963, -0.0008150, -0.0013813, 0.0013813
7: -0.0082640, -0.0074494, -0.0082640, -0.0074494, -0.0008146, 0.0008146
8: 0.0052524, 0.0098919, 0.0052524, 0.0098919, -0.0045630, 0.0045617
9: -0.0036845, -0.0030730, -0.0036845, -0.0030730, -0.0006115, 0.0006115

### Unstable ReLU Count (Linear/Conv2D Layers)
- layer_idx=0, type=LayerType.Linear, total=256, inp1_unstable=10, inp2_unstable=10, delta_unstable=250
- layer_idx=2, type=LayerType.Linear, total=256, inp1_unstable=8, inp2_unstable=8, delta_unstable=256
- layer_idx=4, type=LayerType.Linear, total=256, inp1_unstable=0, inp2_unstable=0, delta_unstable=256
- layer_idx=6, type=LayerType.Linear, total=256, inp1_unstable=2, inp2_unstable=2, delta_unstable=256
- layer_idx=8, type=LayerType.Linear, total=10, inp1_unstable=0, inp2_unstable=0, delta_unstable=10

Time for backsubstitution: 1.70 seconds

### RS candidates at layer 1
type: RSZ, layer: 1, pos: 198
type: RSZ, layer: 1, pos: 199
type: RSZ, layer: 1, pos: 204
type: RSZ, layer: 1, pos: 250

Time for candidate selection: 0.18 seconds

### Candidate
type: RSZ, layer: 1, pos: 198

### Relational analysis RSZ of RS_RSZ1_RSZ2_RSZ2_RSZ1_RSZ2_RSZ1_RSZ1

#### Relational analysis RSZ result of RS_RSZ1_RSZ2_RSZ2_RSZ1_RSZ2_RSZ1_RSZ1
Status: Status.UNKNOWN
Output dim: 1, lower bound: -0.0007828, upper bound: 0.0007828
time: 0.56 seconds

### Relational analysis RSZ of RS_RSZ1_RSZ2_RSZ2_RSZ1_RSZ2_RSZ1_RSZ2

#### Relational analysis RSZ result of RS_RSZ1_RSZ2_RSZ2_RSZ1_RSZ2_RSZ1_RSZ2
Status: Status.UNKNOWN
Output dim: 1, lower bound: -0.0007828, upper bound: 0.0007828
time: 0.55 seconds

## BFS RS instance: RS_RSZ1_RSZ2_RSZ2_RSZ1_RSZ2_RSZ2

### Backsubstitution after applying RS history:
0: 0.0006996, 0.0011177, 0.0006996, 0.0011177, -0.0004181, 0.0004181
1: 0.9934013, 0.9943976, 0.9934013, 0.9943976, -0.0009964, 0.0009964
2: -0.0088595, -0.0053283, -0.0088595, -0.0053283, -0.0032412, 0.0032437
3: 0.0035633, 0.0041662, 0.0035633, 0.0041662, -0.0006029, 0.0006029
4: 0.0026282, 0.0054191, 0.0026282, 0.0054191, -0.0027909, 0.0027909
5: 0.0050201, 0.0064446, 0.0050201, 0.0064446, -0.0014245, 0.0014245
6: -0.0021963, -0.0008150, -0.0021963, -0.0008150, -0.0013813, 0.0013813
7: -0.0082640, -0.0074494, -0.0082640, -0.0074494, -0.0008146, 0.0008146
8: 0.0052524, 0.0098919, 0.0052524, 0.0098919, -0.0045628, 0.0045621
9: -0.0036845, -0.0030730, -0.0036845, -0.0030730, -0.0006115, 0.0006115

### Unstable ReLU Count (Linear/Conv2D Layers)
- layer_idx=0, type=LayerType.Linear, total=256, inp1_unstable=10, inp2_unstable=10, delta_unstable=250
- layer_idx=2, type=LayerType.Linear, total=256, inp1_unstable=8, inp2_unstable=8, delta_unstable=256
- layer_idx=4, type=LayerType.Linear, total=256, inp1_unstable=0, inp2_unstable=0, delta_unstable=256
- layer_idx=6, type=LayerType.Linear, total=256, inp1_unstable=2, inp2_unstable=2, delta_unstable=256
- layer_idx=8, type=LayerType.Linear, total=10, inp1_unstable=0, inp2_unstable=0, delta_unstable=10

Time for backsubstitution: 1.71 seconds

### RS candidates at layer 1
type: RSZ, layer: 1, pos: 198
type: RSZ, layer: 1, pos: 199
type: RSZ, layer: 1, pos: 204
type: RSZ, layer: 1, pos: 250

Time for candidate selection: 0.22 seconds

### Candidate
type: RSZ, layer: 1, pos: 198

### Relational analysis RSZ of RS_RSZ1_RSZ2_RSZ2_RSZ1_RSZ2_RSZ2_RSZ1

#### Relational analysis RSZ result of RS_RSZ1_RSZ2_RSZ2_RSZ1_RSZ2_RSZ2_RSZ1
Status: Status.UNKNOWN
Output dim: 1, lower bound: -0.0007828, upper bound: 0.0007828
time: 0.58 seconds

### Relational analysis RSZ of RS_RSZ1_RSZ2_RSZ2_RSZ1_RSZ2_RSZ2_RSZ2

#### Relational analysis RSZ result of RS_RSZ1_RSZ2_RSZ2_RSZ1_RSZ2_RSZ2_RSZ2
Status: Status.UNKNOWN
Output dim: 1, lower bound: -0.0007828, upper bound: 0.0007828
time: 0.56 seconds

## BFS RS instance: RS_RSZ1_RSZ2_RSZ2_RSZ2_RSZ1_RSZ1

### Backsubstitution after applying RS history:
0: 0.0006996, 0.0011177, 0.0006996, 0.0011177, -0.0004181, 0.0004181
1: 0.9934013, 0.9943976, 0.9934013, 0.9943976, -0.0009964, 0.0009964
2: -0.0088595, -0.0053283, -0.0088595, -0.0053283, -0.0032405, 0.0032434
3: 0.0035633, 0.0041662, 0.0035633, 0.0041662, -0.0006029, 0.0006029
4: 0.0026282, 0.0054191, 0.0026282, 0.0054191, -0.0027909, 0.0027909
5: 0.0050201, 0.0064446, 0.0050201, 0.0064446, -0.0014245, 0.0014245
6: -0.0021963, -0.0008150, -0.0021963, -0.0008150, -0.0013813, 0.0013813
7: -0.0082640, -0.0074494, -0.0082640, -0.0074494, -0.0008146, 0.0008146
8: 0.0052524, 0.0098919, 0.0052524, 0.0098919, -0.0045627, 0.0045619
9: -0.0036845, -0.0030730, -0.0036845, -0.0030730, -0.0006115, 0.0006115

### Unstable ReLU Count (Linear/Conv2D Layers)
- layer_idx=0, type=LayerType.Linear, total=256, inp1_unstable=10, inp2_unstable=10, delta_unstable=250
- layer_idx=2, type=LayerType.Linear, total=256, inp1_unstable=8, inp2_unstable=8, delta_unstable=256
- layer_idx=4, type=LayerType.Linear, total=256, inp1_unstable=0, inp2_unstable=0, delta_unstable=256
- layer_idx=6, type=LayerType.Linear, total=256, inp1_unstable=2, inp2_unstable=2, delta_unstable=256
- layer_idx=8, type=LayerType.Linear, total=10, inp1_unstable=0, inp2_unstable=0, delta_unstable=10

Time for backsubstitution: 1.66 seconds

### RS candidates at layer 1
type: RSZ, layer: 1, pos: 198
type: RSZ, layer: 1, pos: 199
type: RSZ, layer: 1, pos: 204
type: RSZ, layer: 1, pos: 250

Time for candidate selection: 0.18 seconds

### Candidate
type: RSZ, layer: 1, pos: 198

### Relational analysis RSZ of RS_RSZ1_RSZ2_RSZ2_RSZ2_RSZ1_RSZ1_RSZ1

#### Relational analysis RSZ result of RS_RSZ1_RSZ2_RSZ2_RSZ2_RSZ1_RSZ1_RSZ1
Status: Status.UNKNOWN
Output dim: 1, lower bound: -0.0007828, upper bound: 0.0007828
time: 0.61 seconds

### Relational analysis RSZ of RS_RSZ1_RSZ2_RSZ2_RSZ2_RSZ1_RSZ1_RSZ2

#### Relational analysis RSZ result of RS_RSZ1_RSZ2_RSZ2_RSZ2_RSZ1_RSZ1_RSZ2
Status: Status.UNKNOWN
Output dim: 1, lower bound: -0.0007828, upper bound: 0.0007828
time: 0.64 seconds

## BFS RS instance: RS_RSZ1_RSZ2_RSZ2_RSZ2_RSZ1_RSZ2

### Backsubstitution after applying RS history:
0: 0.0006996, 0.0011177, 0.0006996, 0.0011177, -0.0004181, 0.0004181
1: 0.9934013, 0.9943976, 0.9934013, 0.9943976, -0.0009964, 0.0009964
2: -0.0088595, -0.0053283, -0.0088595, -0.0053283, -0.0032418, 0.0032429
3: 0.0035633, 0.0041662, 0.0035633, 0.0041662, -0.0006029, 0.0006029
4: 0.0026282, 0.0054191, 0.0026282, 0.0054191, -0.0027909, 0.0027909
5: 0.0050201, 0.0064446, 0.0050201, 0.0064446, -0.0014245, 0.0014245
6: -0.0021963, -0.0008150, -0.0021963, -0.0008150, -0.0013813, 0.0013813
7: -0.0082640, -0.0074494, -0.0082640, -0.0074494, -0.0008146, 0.0008146
8: 0.0052524, 0.0098919, 0.0052524, 0.0098919, -0.0045625, 0.0045623
9: -0.0036845, -0.0030730, -0.0036845, -0.0030730, -0.0006115, 0.0006115

### Unstable ReLU Count (Linear/Conv2D Layers)
- layer_idx=0, type=LayerType.Linear, total=256, inp1_unstable=10, inp2_unstable=10, delta_unstable=250
- layer_idx=2, type=LayerType.Linear, total=256, inp1_unstable=8, inp2_unstable=8, delta_unstable=256
- layer_idx=4, type=LayerType.Linear, total=256, inp1_unstable=0, inp2_unstable=0, delta_unstable=256
- layer_idx=6, type=LayerType.Linear, total=256, inp1_unstable=2, inp2_unstable=2, delta_unstable=256
- layer_idx=8, type=LayerType.Linear, total=10, inp1_unstable=0, inp2_unstable=0, delta_unstable=10

Time for backsubstitution: 1.71 seconds

### RS candidates at layer 1
type: RSZ, layer: 1, pos: 198
type: RSZ, layer: 1, pos: 199
type: RSZ, layer: 1, pos: 204
type: RSZ, layer: 1, pos: 250

Time for candidate selection: 0.19 seconds

### Candidate
type: RSZ, layer: 1, pos: 198

### Relational analysis RSZ of RS_RSZ1_RSZ2_RSZ2_RSZ2_RSZ1_RSZ2_RSZ1

#### Relational analysis RSZ result of RS_RSZ1_RSZ2_RSZ2_RSZ2_RSZ1_RSZ2_RSZ1
Status: Status.UNKNOWN
Output dim: 1, lower bound: -0.0007828, upper bound: 0.0007828
time: 0.76 seconds

### Relational analysis RSZ of RS_RSZ1_RSZ2_RSZ2_RSZ2_RSZ1_RSZ2_RSZ2

#### Relational analysis RSZ result of RS_RSZ1_RSZ2_RSZ2_RSZ2_RSZ1_RSZ2_RSZ2
Status: Status.UNKNOWN
Output dim: 1, lower bound: -0.0007828, upper bound: 0.0007828
time: 0.66 seconds

## BFS RS instance: RS_RSZ1_RSZ2_RSZ2_RSZ2_RSZ2_RSZ1

### Backsubstitution after applying RS history:
0: 0.0006996, 0.0011177, 0.0006996, 0.0011177, -0.0004181, 0.0004181
1: 0.9934013, 0.9943976, 0.9934013, 0.9943976, -0.0009964, 0.0009964
2: -0.0088595, -0.0053283, -0.0088595, -0.0053283, -0.0032385, 0.0032455
3: 0.0035633, 0.0041662, 0.0035633, 0.0041662, -0.0006029, 0.0006029
4: 0.0026282, 0.0054191, 0.0026282, 0.0054191, -0.0027909, 0.0027909
5: 0.0050201, 0.0064446, 0.0050201, 0.0064446, -0.0014245, 0.0014245
6: -0.0021963, -0.0008150, -0.0021963, -0.0008150, -0.0013813, 0.0013813
7: -0.0082640, -0.0074494, -0.0082640, -0.0074494, -0.0008146, 0.0008146
8: 0.0052524, 0.0098919, 0.0052524, 0.0098919, -0.0045632, 0.0045614
9: -0.0036845, -0.0030730, -0.0036845, -0.0030730, -0.0006115, 0.0006115

### Unstable ReLU Count (Linear/Conv2D Layers)
- layer_idx=0, type=LayerType.Linear, total=256, inp1_unstable=10, inp2_unstable=10, delta_unstable=250
- layer_idx=2, type=LayerType.Linear, total=256, inp1_unstable=8, inp2_unstable=8, delta_unstable=256
- layer_idx=4, type=LayerType.Linear, total=256, inp1_unstable=0, inp2_unstable=0, delta_unstable=256
- layer_idx=6, type=LayerType.Linear, total=256, inp1_unstable=2, inp2_unstable=2, delta_unstable=256
- layer_idx=8, type=LayerType.Linear, total=10, inp1_unstable=0, inp2_unstable=0, delta_unstable=10

Time for backsubstitution: 1.68 seconds

### RS candidates at layer 1
type: RSZ, layer: 1, pos: 198
type: RSZ, layer: 1, pos: 199
type: RSZ, layer: 1, pos: 204
type: RSZ, layer: 1, pos: 250

Time for candidate selection: 0.18 seconds

### Candidate
type: RSZ, layer: 1, pos: 198

### Relational analysis RSZ of RS_RSZ1_RSZ2_RSZ2_RSZ2_RSZ2_RSZ1_RSZ1

#### Relational analysis RSZ result of RS_RSZ1_RSZ2_RSZ2_RSZ2_RSZ2_RSZ1_RSZ1
Status: Status.UNKNOWN
Output dim: 1, lower bound: -0.0007828, upper bound: 0.0007828
time: 0.56 seconds

### Relational analysis RSZ of RS_RSZ1_RSZ2_RSZ2_RSZ2_RSZ2_RSZ1_RSZ2

#### Relational analysis RSZ result of RS_RSZ1_RSZ2_RSZ2_RSZ2_RSZ2_RSZ1_RSZ2
Status: Status.UNKNOWN
Output dim: 1, lower bound: -0.0007828, upper bound: 0.0007828
time: 0.57 seconds

## BFS RS instance: RS_RSZ1_RSZ2_RSZ2_RSZ2_RSZ2_RSZ2

### Backsubstitution after applying RS history:
0: 0.0006996, 0.0011177, 0.0006996, 0.0011177, -0.0004181, 0.0004181
1: 0.9934013, 0.9943976, 0.9934013, 0.9943976, -0.0009964, 0.0009964
2: -0.0088595, -0.0053283, -0.0088595, -0.0053283, -0.0032397, 0.0032445
3: 0.0035633, 0.0041662, 0.0035633, 0.0041662, -0.0006029, 0.0006029
4: 0.0026282, 0.0054191, 0.0026282, 0.0054191, -0.0027909, 0.0027909
5: 0.0050201, 0.0064446, 0.0050201, 0.0064446, -0.0014245, 0.0014245
6: -0.0021963, -0.0008150, -0.0021963, -0.0008150, -0.0013813, 0.0013813
7: -0.0082640, -0.0074494, -0.0082640, -0.0074494, -0.0008146, 0.0008146
8: 0.0052524, 0.0098919, 0.0052524, 0.0098919, -0.0045630, 0.0045617
9: -0.0036845, -0.0030730, -0.0036845, -0.0030730, -0.0006115, 0.0006115

### Unstable ReLU Count (Linear/Conv2D Layers)
- layer_idx=0, type=LayerType.Linear, total=256, inp1_unstable=10, inp2_unstable=10, delta_unstable=250
- layer_idx=2, type=LayerType.Linear, total=256, inp1_unstable=8, inp2_unstable=8, delta_unstable=256
- layer_idx=4, type=LayerType.Linear, total=256, inp1_unstable=0, inp2_unstable=0, delta_unstable=256
- layer_idx=6, type=LayerType.Linear, total=256, inp1_unstable=2, inp2_unstable=2, delta_unstable=256
- layer_idx=8, type=LayerType.Linear, total=10, inp1_unstable=0, inp2_unstable=0, delta_unstable=10

Time for backsubstitution: 1.77 seconds

### RS candidates at layer 1
type: RSZ, layer: 1, pos: 198
type: RSZ, layer: 1, pos: 199
type: RSZ, layer: 1, pos: 204
type: RSZ, layer: 1, pos: 250

Time for candidate selection: 0.19 seconds

### Candidate
type: RSZ, layer: 1, pos: 198

### Relational analysis RSZ of RS_RSZ1_RSZ2_RSZ2_RSZ2_RSZ2_RSZ2_RSZ1

#### Relational analysis RSZ result of RS_RSZ1_RSZ2_RSZ2_RSZ2_RSZ2_RSZ2_RSZ1
Status: Status.UNKNOWN
Output dim: 1, lower bound: -0.0007828, upper bound: 0.0007828
time: 0.56 seconds

### Relational analysis RSZ of RS_RSZ1_RSZ2_RSZ2_RSZ2_RSZ2_RSZ2_RSZ2

#### Relational analysis RSZ result of RS_RSZ1_RSZ2_RSZ2_RSZ2_RSZ2_RSZ2_RSZ2
Status: Status.UNKNOWN
Output dim: 1, lower bound: -0.0007828, upper bound: 0.0007828
time: 0.57 seconds

## BFS RS instance: RS_RSZ2_RSZ1_RSZ1_RSZ1_RSZ1_RSZ1

### Backsubstitution after applying RS history:
0: 0.0006996, 0.0011177, 0.0006996, 0.0011177, -0.0004181, 0.0004181
1: 0.9934013, 0.9943976, 0.9934013, 0.9943976, -0.0009964, 0.0009964
2: -0.0088595, -0.0053283, -0.0088595, -0.0053283, -0.0032445, 0.0032397
3: 0.0035633, 0.0041662, 0.0035633, 0.0041662, -0.0006029, 0.0006029
4: 0.0026282, 0.0054191, 0.0026282, 0.0054191, -0.0027909, 0.0027909
5: 0.0050201, 0.0064446, 0.0050201, 0.0064446, -0.0014245, 0.0014245
6: -0.0021963, -0.0008150, -0.0021963, -0.0008150, -0.0013813, 0.0013813
7: -0.0082640, -0.0074494, -0.0082640, -0.0074494, -0.0008146, 0.0008146
8: 0.0052524, 0.0098919, 0.0052524, 0.0098919, -0.0045617, 0.0045630
9: -0.0036845, -0.0030730, -0.0036845, -0.0030730, -0.0006115, 0.0006115

### Unstable ReLU Count (Linear/Conv2D Layers)
- layer_idx=0, type=LayerType.Linear, total=256, inp1_unstable=10, inp2_unstable=10, delta_unstable=250
- layer_idx=2, type=LayerType.Linear, total=256, inp1_unstable=8, inp2_unstable=8, delta_unstable=256
- layer_idx=4, type=LayerType.Linear, total=256, inp1_unstable=0, inp2_unstable=0, delta_unstable=256
- layer_idx=6, type=LayerType.Linear, total=256, inp1_unstable=2, inp2_unstable=2, delta_unstable=256
- layer_idx=8, type=LayerType.Linear, total=10, inp1_unstable=0, inp2_unstable=0, delta_unstable=10

Time for backsubstitution: 1.73 seconds

### RS candidates at layer 1
type: RSZ, layer: 1, pos: 198
type: RSZ, layer: 1, pos: 199
type: RSZ, layer: 1, pos: 204
type: RSZ, layer: 1, pos: 250

Time for candidate selection: 0.18 seconds

### Candidate
type: RSZ, layer: 1, pos: 198

### Relational analysis RSZ of RS_RSZ2_RSZ1_RSZ1_RSZ1_RSZ1_RSZ1_RSZ1

#### Relational analysis RSZ result of RS_RSZ2_RSZ1_RSZ1_RSZ1_RSZ1_RSZ1_RSZ1
Status: Status.UNKNOWN
Output dim: 1, lower bound: -0.0007828, upper bound: 0.0007828
time: 0.57 seconds

### Relational analysis RSZ of RS_RSZ2_RSZ1_RSZ1_RSZ1_RSZ1_RSZ1_RSZ2

#### Relational analysis RSZ result of RS_RSZ2_RSZ1_RSZ1_RSZ1_RSZ1_RSZ1_RSZ2
Status: Status.UNKNOWN
Output dim: 1, lower bound: -0.0007828, upper bound: 0.0007828
time: 0.58 seconds

## BFS RS instance: RS_RSZ2_RSZ1_RSZ1_RSZ1_RSZ1_RSZ2

### Backsubstitution after applying RS history:
0: 0.0006996, 0.0011177, 0.0006996, 0.0011177, -0.0004181, 0.0004181
1: 0.9934013, 0.9943976, 0.9934013, 0.9943976, -0.0009964, 0.0009964
2: -0.0088595, -0.0053283, -0.0088595, -0.0053283, -0.0032455, 0.0032385
3: 0.0035633, 0.0041662, 0.0035633, 0.0041662, -0.0006029, 0.0006029
4: 0.0026282, 0.0054191, 0.0026282, 0.0054191, -0.0027909, 0.0027909
5: 0.0050201, 0.0064446, 0.0050201, 0.0064446, -0.0014245, 0.0014245
6: -0.0021963, -0.0008150, -0.0021963, -0.0008150, -0.0013813, 0.0013813
7: -0.0082640, -0.0074494, -0.0082640, -0.0074494, -0.0008146, 0.0008146
8: 0.0052524, 0.0098919, 0.0052524, 0.0098919, -0.0045614, 0.0045632
9: -0.0036845, -0.0030730, -0.0036845, -0.0030730, -0.0006115, 0.0006115

### Unstable ReLU Count (Linear/Conv2D Layers)
- layer_idx=0, type=LayerType.Linear, total=256, inp1_unstable=10, inp2_unstable=10, delta_unstable=250
- layer_idx=2, type=LayerType.Linear, total=256, inp1_unstable=8, inp2_unstable=8, delta_unstable=256
- layer_idx=4, type=LayerType.Linear, total=256, inp1_unstable=0, inp2_unstable=0, delta_unstable=256
- layer_idx=6, type=LayerType.Linear, total=256, inp1_unstable=2, inp2_unstable=2, delta_unstable=256
- layer_idx=8, type=LayerType.Linear, total=10, inp1_unstable=0, inp2_unstable=0, delta_unstable=10

Time for backsubstitution: 1.66 seconds

### RS candidates at layer 1
type: RSZ, layer: 1, pos: 198
type: RSZ, layer: 1, pos: 199
type: RSZ, layer: 1, pos: 204
type: RSZ, layer: 1, pos: 250

Time for candidate selection: 0.19 seconds

### Candidate
type: RSZ, layer: 1, pos: 198

### Relational analysis RSZ of RS_RSZ2_RSZ1_RSZ1_RSZ1_RSZ1_RSZ2_RSZ1

#### Relational analysis RSZ result of RS_RSZ2_RSZ1_RSZ1_RSZ1_RSZ1_RSZ2_RSZ1
Status: Status.UNKNOWN
Output dim: 1, lower bound: -0.0007828, upper bound: 0.0007828
time: 0.57 seconds

### Relational analysis RSZ of RS_RSZ2_RSZ1_RSZ1_RSZ1_RSZ1_RSZ2_RSZ2

#### Relational analysis RSZ result of RS_RSZ2_RSZ1_RSZ1_RSZ1_RSZ1_RSZ2_RSZ2
Status: Status.UNKNOWN
Output dim: 1, lower bound: -0.0007828, upper bound: 0.0007828
time: 0.61 seconds

## BFS RS instance: RS_RSZ2_RSZ1_RSZ1_RSZ1_RSZ2_RSZ1

### Backsubstitution after applying RS history:
0: 0.0006996, 0.0011177, 0.0006996, 0.0011177, -0.0004181, 0.0004181
1: 0.9934013, 0.9943976, 0.9934013, 0.9943976, -0.0009964, 0.0009964
2: -0.0088595, -0.0053283, -0.0088595, -0.0053283, -0.0032429, 0.0032418
3: 0.0035633, 0.0041662, 0.0035633, 0.0041662, -0.0006029, 0.0006029
4: 0.0026282, 0.0054191, 0.0026282, 0.0054191, -0.0027909, 0.0027909
5: 0.0050201, 0.0064446, 0.0050201, 0.0064446, -0.0014245, 0.0014245
6: -0.0021963, -0.0008150, -0.0021963, -0.0008150, -0.0013813, 0.0013813
7: -0.0082640, -0.0074494, -0.0082640, -0.0074494, -0.0008146, 0.0008146
8: 0.0052524, 0.0098919, 0.0052524, 0.0098919, -0.0045623, 0.0045625
9: -0.0036845, -0.0030730, -0.0036845, -0.0030730, -0.0006115, 0.0006115

### Unstable ReLU Count (Linear/Conv2D Layers)
- layer_idx=0, type=LayerType.Linear, total=256, inp1_unstable=10, inp2_unstable=10, delta_unstable=250
- layer_idx=2, type=LayerType.Linear, total=256, inp1_unstable=8, inp2_unstable=8, delta_unstable=256
- layer_idx=4, type=LayerType.Linear, total=256, inp1_unstable=0, inp2_unstable=0, delta_unstable=256
- layer_idx=6, type=LayerType.Linear, total=256, inp1_unstable=2, inp2_unstable=2, delta_unstable=256
- layer_idx=8, type=LayerType.Linear, total=10, inp1_unstable=0, inp2_unstable=0, delta_unstable=10

Time for backsubstitution: 1.71 seconds

### RS candidates at layer 1
type: RSZ, layer: 1, pos: 198
type: RSZ, layer: 1, pos: 199
type: RSZ, layer: 1, pos: 204
type: RSZ, layer: 1, pos: 250

Time for candidate selection: 0.18 seconds

### Candidate
type: RSZ, layer: 1, pos: 198

### Relational analysis RSZ of RS_RSZ2_RSZ1_RSZ1_RSZ1_RSZ2_RSZ1_RSZ1

#### Relational analysis RSZ result of RS_RSZ2_RSZ1_RSZ1_RSZ1_RSZ2_RSZ1_RSZ1
Status: Status.UNKNOWN
Output dim: 1, lower bound: -0.0007828, upper bound: 0.0007828
time: 0.61 seconds

### Relational analysis RSZ of RS_RSZ2_RSZ1_RSZ1_RSZ1_RSZ2_RSZ1_RSZ2

#### Relational analysis RSZ result of RS_RSZ2_RSZ1_RSZ1_RSZ1_RSZ2_RSZ1_RSZ2
Status: Status.UNKNOWN
Output dim: 1, lower bound: -0.0007828, upper bound: 0.0007828
time: 0.63 seconds

## BFS RS instance: RS_RSZ2_RSZ1_RSZ1_RSZ1_RSZ2_RSZ2

### Backsubstitution after applying RS history:
0: 0.0006996, 0.0011177, 0.0006996, 0.0011177, -0.0004181, 0.0004181
1: 0.9934013, 0.9943976, 0.9934013, 0.9943976, -0.0009964, 0.0009964
2: -0.0088595, -0.0053283, -0.0088595, -0.0053283, -0.0032434, 0.0032405
3: 0.0035633, 0.0041662, 0.0035633, 0.0041662, -0.0006029, 0.0006029
4: 0.0026282, 0.0054191, 0.0026282, 0.0054191, -0.0027909, 0.0027909
5: 0.0050201, 0.0064446, 0.0050201, 0.0064446, -0.0014245, 0.0014245
6: -0.0021963, -0.0008150, -0.0021963, -0.0008150, -0.0013813, 0.0013813
7: -0.0082640, -0.0074494, -0.0082640, -0.0074494, -0.0008146, 0.0008146
8: 0.0052524, 0.0098919, 0.0052524, 0.0098919, -0.0045619, 0.0045627
9: -0.0036845, -0.0030730, -0.0036845, -0.0030730, -0.0006115, 0.0006115

### Unstable ReLU Count (Linear/Conv2D Layers)
- layer_idx=0, type=LayerType.Linear, total=256, inp1_unstable=10, inp2_unstable=10, delta_unstable=250
- layer_idx=2, type=LayerType.Linear, total=256, inp1_unstable=8, inp2_unstable=8, delta_unstable=256
- layer_idx=4, type=LayerType.Linear, total=256, inp1_unstable=0, inp2_unstable=0, delta_unstable=256
- layer_idx=6, type=LayerType.Linear, total=256, inp1_unstable=2, inp2_unstable=2, delta_unstable=256
- layer_idx=8, type=LayerType.Linear, total=10, inp1_unstable=0, inp2_unstable=0, delta_unstable=10

Time for backsubstitution: 1.67 seconds

### RS candidates at layer 1
type: RSZ, layer: 1, pos: 198
type: RSZ, layer: 1, pos: 199
type: RSZ, layer: 1, pos: 204
type: RSZ, layer: 1, pos: 250

Time for candidate selection: 0.18 seconds

### Candidate
type: RSZ, layer: 1, pos: 198

### Relational analysis RSZ of RS_RSZ2_RSZ1_RSZ1_RSZ1_RSZ2_RSZ2_RSZ1

#### Relational analysis RSZ result of RS_RSZ2_RSZ1_RSZ1_RSZ1_RSZ2_RSZ2_RSZ1
Status: Status.UNKNOWN
Output dim: 1, lower bound: -0.0007828, upper bound: 0.0007828
time: 0.61 seconds

### Relational analysis RSZ of RS_RSZ2_RSZ1_RSZ1_RSZ1_RSZ2_RSZ2_RSZ2

#### Relational analysis RSZ result of RS_RSZ2_RSZ1_RSZ1_RSZ1_RSZ2_RSZ2_RSZ2
Status: Status.UNKNOWN
Output dim: 1, lower bound: -0.0007828, upper bound: 0.0007828
time: 0.61 seconds

## BFS RS instance: RS_RSZ2_RSZ1_RSZ1_RSZ2_RSZ1_RSZ1

### Backsubstitution after applying RS history:
0: 0.0006996, 0.0011177, 0.0006996, 0.0011177, -0.0004181, 0.0004181
1: 0.9934013, 0.9943976, 0.9934013, 0.9943976, -0.0009964, 0.0009964
2: -0.0088595, -0.0053283, -0.0088595, -0.0053283, -0.0032437, 0.0032412
3: 0.0035633, 0.0041662, 0.0035633, 0.0041662, -0.0006029, 0.0006029
4: 0.0026282, 0.0054191, 0.0026282, 0.0054191, -0.0027909, 0.0027909
5: 0.0050201, 0.0064446, 0.0050201, 0.0064446, -0.0014245, 0.0014245
6: -0.0021963, -0.0008150, -0.0021963, -0.0008150, -0.0013813, 0.0013813
7: -0.0082640, -0.0074494, -0.0082640, -0.0074494, -0.0008146, 0.0008146
8: 0.0052524, 0.0098919, 0.0052524, 0.0098919, -0.0045621, 0.0045628
9: -0.0036845, -0.0030730, -0.0036845, -0.0030730, -0.0006115, 0.0006115

### Unstable ReLU Count (Linear/Conv2D Layers)
- layer_idx=0, type=LayerType.Linear, total=256, inp1_unstable=10, inp2_unstable=10, delta_unstable=250
- layer_idx=2, type=LayerType.Linear, total=256, inp1_unstable=8, inp2_unstable=8, delta_unstable=256
- layer_idx=4, type=LayerType.Linear, total=256, inp1_unstable=0, inp2_unstable=0, delta_unstable=256
- layer_idx=6, type=LayerType.Linear, total=256, inp1_unstable=2, inp2_unstable=2, delta_unstable=256
- layer_idx=8, type=LayerType.Linear, total=10, inp1_unstable=0, inp2_unstable=0, delta_unstable=10

Time for backsubstitution: 1.77 seconds

### RS candidates at layer 1
type: RSZ, layer: 1, pos: 198
type: RSZ, layer: 1, pos: 199
type: RSZ, layer: 1, pos: 204
type: RSZ, layer: 1, pos: 250

Time for candidate selection: 0.18 seconds

### Candidate
type: RSZ, layer: 1, pos: 198

### Relational analysis RSZ of RS_RSZ2_RSZ1_RSZ1_RSZ2_RSZ1_RSZ1_RSZ1

#### Relational analysis RSZ result of RS_RSZ2_RSZ1_RSZ1_RSZ2_RSZ1_RSZ1_RSZ1
Status: Status.UNKNOWN
Output dim: 1, lower bound: -0.0007828, upper bound: 0.0007828
time: 0.58 seconds

### Relational analysis RSZ of RS_RSZ2_RSZ1_RSZ1_RSZ2_RSZ1_RSZ1_RSZ2

#### Relational analysis RSZ result of RS_RSZ2_RSZ1_RSZ1_RSZ2_RSZ1_RSZ1_RSZ2
Status: Status.UNKNOWN
Output dim: 1, lower bound: -0.0007828, upper bound: 0.0007828
time: 0.57 seconds

## BFS RS instance: RS_RSZ2_RSZ1_RSZ1_RSZ2_RSZ1_RSZ2

### Backsubstitution after applying RS history:
0: 0.0006996, 0.0011177, 0.0006996, 0.0011177, -0.0004181, 0.0004181
1: 0.9934013, 0.9943976, 0.9934013, 0.9943976, -0.0009964, 0.0009964
2: -0.0088595, -0.0053283, -0.0088595, -0.0053283, -0.0032446, 0.0032398
3: 0.0035633, 0.0041662, 0.0035633, 0.0041662, -0.0006029, 0.0006029
4: 0.0026282, 0.0054191, 0.0026282, 0.0054191, -0.0027909, 0.0027909
5: 0.0050201, 0.0064446, 0.0050201, 0.0064446, -0.0014245, 0.0014245
6: -0.0021963, -0.0008150, -0.0021963, -0.0008150, -0.0013813, 0.0013813
7: -0.0082640, -0.0074494, -0.0082640, -0.0074494, -0.0008146, 0.0008146
8: 0.0052524, 0.0098919, 0.0052524, 0.0098919, -0.0045617, 0.0045630
9: -0.0036845, -0.0030730, -0.0036845, -0.0030730, -0.0006115, 0.0006115

### Unstable ReLU Count (Linear/Conv2D Layers)
- layer_idx=0, type=LayerType.Linear, total=256, inp1_unstable=10, inp2_unstable=10, delta_unstable=250
- layer_idx=2, type=LayerType.Linear, total=256, inp1_unstable=8, inp2_unstable=8, delta_unstable=256
- layer_idx=4, type=LayerType.Linear, total=256, inp1_unstable=0, inp2_unstable=0, delta_unstable=256
- layer_idx=6, type=LayerType.Linear, total=256, inp1_unstable=2, inp2_unstable=2, delta_unstable=256
- layer_idx=8, type=LayerType.Linear, total=10, inp1_unstable=0, inp2_unstable=0, delta_unstable=10

Time for backsubstitution: 1.68 seconds

### RS candidates at layer 1
type: RSZ, layer: 1, pos: 198
type: RSZ, layer: 1, pos: 199
type: RSZ, layer: 1, pos: 204
type: RSZ, layer: 1, pos: 250

Time for candidate selection: 0.19 seconds

### Candidate
type: RSZ, layer: 1, pos: 198

### Relational analysis RSZ of RS_RSZ2_RSZ1_RSZ1_RSZ2_RSZ1_RSZ2_RSZ1

#### Relational analysis RSZ result of RS_RSZ2_RSZ1_RSZ1_RSZ2_RSZ1_RSZ2_RSZ1
Status: Status.UNKNOWN
Output dim: 1, lower bound: -0.0007828, upper bound: 0.0007828
time: 0.59 seconds

### Relational analysis RSZ of RS_RSZ2_RSZ1_RSZ1_RSZ2_RSZ1_RSZ2_RSZ2

#### Relational analysis RSZ result of RS_RSZ2_RSZ1_RSZ1_RSZ2_RSZ1_RSZ2_RSZ2
Status: Status.UNKNOWN
Output dim: 1, lower bound: -0.0007828, upper bound: 0.0007828
time: 0.62 seconds

## BFS RS instance: RS_RSZ2_RSZ1_RSZ1_RSZ2_RSZ2_RSZ1

### Backsubstitution after applying RS history:
0: 0.0006996, 0.0011177, 0.0006996, 0.0011177, -0.0004181, 0.0004181
1: 0.9934013, 0.9943976, 0.9934013, 0.9943976, -0.0009964, 0.0009964
2: -0.0088595, -0.0053283, -0.0088595, -0.0053283, -0.0032419, 0.0032433
3: 0.0035633, 0.0041662, 0.0035633, 0.0041662, -0.0006029, 0.0006029
4: 0.0026282, 0.0054191, 0.0026282, 0.0054191, -0.0027909, 0.0027909
5: 0.0050201, 0.0064446, 0.0050201, 0.0064446, -0.0014245, 0.0014245
6: -0.0021963, -0.0008150, -0.0021963, -0.0008150, -0.0013813, 0.0013813
7: -0.0082640, -0.0074494, -0.0082640, -0.0074494, -0.0008146, 0.0008146
8: 0.0052524, 0.0098919, 0.0052524, 0.0098919, -0.0045627, 0.0045623
9: -0.0036845, -0.0030730, -0.0036845, -0.0030730, -0.0006115, 0.0006115

### Unstable ReLU Count (Linear/Conv2D Layers)
- layer_idx=0, type=LayerType.Linear, total=256, inp1_unstable=10, inp2_unstable=10, delta_unstable=250
- layer_idx=2, type=LayerType.Linear, total=256, inp1_unstable=8, inp2_unstable=8, delta_unstable=256
- layer_idx=4, type=LayerType.Linear, total=256, inp1_unstable=0, inp2_unstable=0, delta_unstable=256
- layer_idx=6, type=LayerType.Linear, total=256, inp1_unstable=2, inp2_unstable=2, delta_unstable=256
- layer_idx=8, type=LayerType.Linear, total=10, inp1_unstable=0, inp2_unstable=0, delta_unstable=10

Time for backsubstitution: 1.73 seconds

### RS candidates at layer 1
type: RSZ, layer: 1, pos: 198
type: RSZ, layer: 1, pos: 199
type: RSZ, layer: 1, pos: 204
type: RSZ, layer: 1, pos: 250

Time for candidate selection: 0.19 seconds

### Candidate
type: RSZ, layer: 1, pos: 198

### Relational analysis RSZ of RS_RSZ2_RSZ1_RSZ1_RSZ2_RSZ2_RSZ1_RSZ1

#### Relational analysis RSZ result of RS_RSZ2_RSZ1_RSZ1_RSZ2_RSZ2_RSZ1_RSZ1
Status: Status.UNKNOWN
Output dim: 1, lower bound: -0.0007828, upper bound: 0.0007828
time: 0.63 seconds

### Relational analysis RSZ of RS_RSZ2_RSZ1_RSZ1_RSZ2_RSZ2_RSZ1_RSZ2

#### Relational analysis RSZ result of RS_RSZ2_RSZ1_RSZ1_RSZ2_RSZ2_RSZ1_RSZ2
Status: Status.UNKNOWN
Output dim: 1, lower bound: -0.0007828, upper bound: 0.0007828
time: 0.63 seconds

## BFS RS instance: RS_RSZ2_RSZ1_RSZ1_RSZ2_RSZ2_RSZ2

### Backsubstitution after applying RS history:
0: 0.0006996, 0.0011177, 0.0006996, 0.0011177, -0.0004181, 0.0004181
1: 0.9934013, 0.9943976, 0.9934013, 0.9943976, -0.0009964, 0.0009964
2: -0.0088595, -0.0053283, -0.0088595, -0.0053283, -0.0032426, 0.0032419
3: 0.0035633, 0.0041662, 0.0035633, 0.0041662, -0.0006029, 0.0006029
4: 0.0026282, 0.0054191, 0.0026282, 0.0054191, -0.0027909, 0.0027909
5: 0.0050201, 0.0064446, 0.0050201, 0.0064446, -0.0014245, 0.0014245
6: -0.0021963, -0.0008150, -0.0021963, -0.0008150, -0.0013813, 0.0013813
7: -0.0082640, -0.0074494, -0.0082640, -0.0074494, -0.0008146, 0.0008146
8: 0.0052524, 0.0098919, 0.0052524, 0.0098919, -0.0045623, 0.0045625
9: -0.0036845, -0.0030730, -0.0036845, -0.0030730, -0.0006115, 0.0006115

### Unstable ReLU Count (Linear/Conv2D Layers)
- layer_idx=0, type=LayerType.Linear, total=256, inp1_unstable=10, inp2_unstable=10, delta_unstable=250
- layer_idx=2, type=LayerType.Linear, total=256, inp1_unstable=8, inp2_unstable=8, delta_unstable=256
- layer_idx=4, type=LayerType.Linear, total=256, inp1_unstable=0, inp2_unstable=0, delta_unstable=256
- layer_idx=6, type=LayerType.Linear, total=256, inp1_unstable=2, inp2_unstable=2, delta_unstable=256
- layer_idx=8, type=LayerType.Linear, total=10, inp1_unstable=0, inp2_unstable=0, delta_unstable=10

Time for backsubstitution: 1.79 seconds

### RS candidates at layer 1
type: RSZ, layer: 1, pos: 198
type: RSZ, layer: 1, pos: 199
type: RSZ, layer: 1, pos: 204
type: RSZ, layer: 1, pos: 250

Time for candidate selection: 0.20 seconds

### Candidate
type: RSZ, layer: 1, pos: 198

### Relational analysis RSZ of RS_RSZ2_RSZ1_RSZ1_RSZ2_RSZ2_RSZ2_RSZ1

#### Relational analysis RSZ result of RS_RSZ2_RSZ1_RSZ1_RSZ2_RSZ2_RSZ2_RSZ1
Status: Status.UNKNOWN
Output dim: 1, lower bound: -0.0007828, upper bound: 0.0007828
time: 0.67 seconds

### Relational analysis RSZ of RS_RSZ2_RSZ1_RSZ1_RSZ2_RSZ2_RSZ2_RSZ2

#### Relational analysis RSZ result of RS_RSZ2_RSZ1_RSZ1_RSZ2_RSZ2_RSZ2_RSZ2
Status: Status.UNKNOWN
Output dim: 1, lower bound: -0.0007828, upper bound: 0.0007828
time: 0.63 seconds

## BFS RS instance: RS_RSZ2_RSZ1_RSZ2_RSZ1_RSZ1_RSZ1

### Backsubstitution after applying RS history:
0: 0.0006996, 0.0011177, 0.0006996, 0.0011177, -0.0004181, 0.0004181
1: 0.9934013, 0.9943976, 0.9934013, 0.9943976, -0.0009964, 0.0009964
2: -0.0088595, -0.0053283, -0.0088595, -0.0053283, -0.0032442, 0.0032399
3: 0.0035633, 0.0041662, 0.0035633, 0.0041662, -0.0006029, 0.0006029
4: 0.0026282, 0.0054191, 0.0026282, 0.0054191, -0.0027909, 0.0027909
5: 0.0050201, 0.0064446, 0.0050201, 0.0064446, -0.0014245, 0.0014245
6: -0.0021963, -0.0008150, -0.0021963, -0.0008150, -0.0013813, 0.0013813
7: -0.0082640, -0.0074494, -0.0082640, -0.0074494, -0.0008146, 0.0008146
8: 0.0052524, 0.0098919, 0.0052524, 0.0098919, -0.0045618, 0.0045629
9: -0.0036845, -0.0030730, -0.0036845, -0.0030730, -0.0006115, 0.0006115

### Unstable ReLU Count (Linear/Conv2D Layers)
- layer_idx=0, type=LayerType.Linear, total=256, inp1_unstable=10, inp2_unstable=10, delta_unstable=250
- layer_idx=2, type=LayerType.Linear, total=256, inp1_unstable=8, inp2_unstable=8, delta_unstable=256
- layer_idx=4, type=LayerType.Linear, total=256, inp1_unstable=0, inp2_unstable=0, delta_unstable=256
- layer_idx=6, type=LayerType.Linear, total=256, inp1_unstable=2, inp2_unstable=2, delta_unstable=256
- layer_idx=8, type=LayerType.Linear, total=10, inp1_unstable=0, inp2_unstable=0, delta_unstable=10

Time for backsubstitution: 1.78 seconds

### RS candidates at layer 1
type: RSZ, layer: 1, pos: 198
type: RSZ, layer: 1, pos: 199
type: RSZ, layer: 1, pos: 204
type: RSZ, layer: 1, pos: 250

Time for candidate selection: 0.20 seconds

### Candidate
type: RSZ, layer: 1, pos: 198

### Relational analysis RSZ of RS_RSZ2_RSZ1_RSZ2_RSZ1_RSZ1_RSZ1_RSZ1

#### Relational analysis RSZ result of RS_RSZ2_RSZ1_RSZ2_RSZ1_RSZ1_RSZ1_RSZ1
Status: Status.UNKNOWN
Output dim: 1, lower bound: -0.0007828, upper bound: 0.0007828
time: 0.66 seconds

### Relational analysis RSZ of RS_RSZ2_RSZ1_RSZ2_RSZ1_RSZ1_RSZ1_RSZ2

#### Relational analysis RSZ result of RS_RSZ2_RSZ1_RSZ2_RSZ1_RSZ1_RSZ1_RSZ2
Status: Status.UNKNOWN
Output dim: 1, lower bound: -0.0007828, upper bound: 0.0007828
time: 0.67 seconds

## BFS RS instance: RS_RSZ2_RSZ1_RSZ2_RSZ1_RSZ1_RSZ2

### Backsubstitution after applying RS history:
0: 0.0006996, 0.0011177, 0.0006996, 0.0011177, -0.0004181, 0.0004181
1: 0.9934013, 0.9943976, 0.9934013, 0.9943976, -0.0009964, 0.0009964
2: -0.0088595, -0.0053283, -0.0088595, -0.0053283, -0.0032451, 0.0032386
3: 0.0035633, 0.0041662, 0.0035633, 0.0041662, -0.0006029, 0.0006029
4: 0.0026282, 0.0054191, 0.0026282, 0.0054191, -0.0027909, 0.0027909
5: 0.0050201, 0.0064446, 0.0050201, 0.0064446, -0.0014245, 0.0014245
6: -0.0021963, -0.0008150, -0.0021963, -0.0008150, -0.0013813, 0.0013813
7: -0.0082640, -0.0074494, -0.0082640, -0.0074494, -0.0008146, 0.0008146
8: 0.0052524, 0.0098919, 0.0052524, 0.0098919, -0.0045614, 0.0045631
9: -0.0036845, -0.0030730, -0.0036845, -0.0030730, -0.0006115, 0.0006115

### Unstable ReLU Count (Linear/Conv2D Layers)
- layer_idx=0, type=LayerType.Linear, total=256, inp1_unstable=10, inp2_unstable=10, delta_unstable=250
- layer_idx=2, type=LayerType.Linear, total=256, inp1_unstable=8, inp2_unstable=8, delta_unstable=256
- layer_idx=4, type=LayerType.Linear, total=256, inp1_unstable=0, inp2_unstable=0, delta_unstable=256
- layer_idx=6, type=LayerType.Linear, total=256, inp1_unstable=2, inp2_unstable=2, delta_unstable=256
- layer_idx=8, type=LayerType.Linear, total=10, inp1_unstable=0, inp2_unstable=0, delta_unstable=10

Time for backsubstitution: 1.80 seconds

### RS candidates at layer 1
type: RSZ, layer: 1, pos: 198
type: RSZ, layer: 1, pos: 199
type: RSZ, layer: 1, pos: 204
type: RSZ, layer: 1, pos: 250

Time for candidate selection: 0.20 seconds

### Candidate
type: RSZ, layer: 1, pos: 198

### Relational analysis RSZ of RS_RSZ2_RSZ1_RSZ2_RSZ1_RSZ1_RSZ2_RSZ1

#### Relational analysis RSZ result of RS_RSZ2_RSZ1_RSZ2_RSZ1_RSZ1_RSZ2_RSZ1
Status: Status.UNKNOWN
Output dim: 1, lower bound: -0.0007828, upper bound: 0.0007828
time: 0.65 seconds

### Relational analysis RSZ of RS_RSZ2_RSZ1_RSZ2_RSZ1_RSZ1_RSZ2_RSZ2

#### Relational analysis RSZ result of RS_RSZ2_RSZ1_RSZ2_RSZ1_RSZ1_RSZ2_RSZ2
Status: Status.UNKNOWN
Output dim: 1, lower bound: -0.0007828, upper bound: 0.0007828
time: 0.69 seconds

## BFS RS instance: RS_RSZ2_RSZ1_RSZ2_RSZ1_RSZ2_RSZ1

### Backsubstitution after applying RS history:
0: 0.0006996, 0.0011177, 0.0006996, 0.0011177, -0.0004181, 0.0004181
1: 0.9934013, 0.9943976, 0.9934013, 0.9943976, -0.0009964, 0.0009964
2: -0.0088595, -0.0053283, -0.0088595, -0.0053283, -0.0032428, 0.0032422
3: 0.0035633, 0.0041662, 0.0035633, 0.0041662, -0.0006029, 0.0006029
4: 0.0026282, 0.0054191, 0.0026282, 0.0054191, -0.0027909, 0.0027909
5: 0.0050201, 0.0064446, 0.0050201, 0.0064446, -0.0014245, 0.0014245
6: -0.0021963, -0.0008150, -0.0021963, -0.0008150, -0.0013813, 0.0013813
7: -0.0082640, -0.0074494, -0.0082640, -0.0074494, -0.0008146, 0.0008146
8: 0.0052524, 0.0098919, 0.0052524, 0.0098919, -0.0045624, 0.0045625
9: -0.0036845, -0.0030730, -0.0036845, -0.0030730, -0.0006115, 0.0006115

### Unstable ReLU Count (Linear/Conv2D Layers)
- layer_idx=0, type=LayerType.Linear, total=256, inp1_unstable=10, inp2_unstable=10, delta_unstable=250
- layer_idx=2, type=LayerType.Linear, total=256, inp1_unstable=8, inp2_unstable=8, delta_unstable=256
- layer_idx=4, type=LayerType.Linear, total=256, inp1_unstable=0, inp2_unstable=0, delta_unstable=256
- layer_idx=6, type=LayerType.Linear, total=256, inp1_unstable=2, inp2_unstable=2, delta_unstable=256
- layer_idx=8, type=LayerType.Linear, total=10, inp1_unstable=0, inp2_unstable=0, delta_unstable=10

Time for backsubstitution: 1.95 seconds

### RS candidates at layer 1
type: RSZ, layer: 1, pos: 198
type: RSZ, layer: 1, pos: 199
type: RSZ, layer: 1, pos: 204
type: RSZ, layer: 1, pos: 250

Time for candidate selection: 0.20 seconds

### Candidate
type: RSZ, layer: 1, pos: 198

### Relational analysis RSZ of RS_RSZ2_RSZ1_RSZ2_RSZ1_RSZ2_RSZ1_RSZ1

#### Relational analysis RSZ result of RS_RSZ2_RSZ1_RSZ2_RSZ1_RSZ2_RSZ1_RSZ1
Status: Status.UNKNOWN
Output dim: 1, lower bound: -0.0007828, upper bound: 0.0007828
time: 0.60 seconds

### Relational analysis RSZ of RS_RSZ2_RSZ1_RSZ2_RSZ1_RSZ2_RSZ1_RSZ2

#### Relational analysis RSZ result of RS_RSZ2_RSZ1_RSZ2_RSZ1_RSZ2_RSZ1_RSZ2
Status: Status.UNKNOWN
Output dim: 1, lower bound: -0.0007828, upper bound: 0.0007828
time: 0.60 seconds

## BFS RS instance: RS_RSZ2_RSZ1_RSZ2_RSZ1_RSZ2_RSZ2

### Backsubstitution after applying RS history:
0: 0.0006996, 0.0011177, 0.0006996, 0.0011177, -0.0004181, 0.0004181
1: 0.9934013, 0.9943976, 0.9934013, 0.9943976, -0.0009964, 0.0009964
2: -0.0088595, -0.0053283, -0.0088595, -0.0053283, -0.0032435, 0.0032407
3: 0.0035633, 0.0041662, 0.0035633, 0.0041662, -0.0006029, 0.0006029
4: 0.0026282, 0.0054191, 0.0026282, 0.0054191, -0.0027909, 0.0027909
5: 0.0050201, 0.0064446, 0.0050201, 0.0064446, -0.0014245, 0.0014245
6: -0.0021963, -0.0008150, -0.0021963, -0.0008150, -0.0013813, 0.0013813
7: -0.0082640, -0.0074494, -0.0082640, -0.0074494, -0.0008146, 0.0008146
8: 0.0052524, 0.0098919, 0.0052524, 0.0098919, -0.0045620, 0.0045627
9: -0.0036845, -0.0030730, -0.0036845, -0.0030730, -0.0006115, 0.0006115

### Unstable ReLU Count (Linear/Conv2D Layers)
- layer_idx=0, type=LayerType.Linear, total=256, inp1_unstable=10, inp2_unstable=10, delta_unstable=250
- layer_idx=2, type=LayerType.Linear, total=256, inp1_unstable=8, inp2_unstable=8, delta_unstable=256
- layer_idx=4, type=LayerType.Linear, total=256, inp1_unstable=0, inp2_unstable=0, delta_unstable=256
- layer_idx=6, type=LayerType.Linear, total=256, inp1_unstable=2, inp2_unstable=2, delta_unstable=256
- layer_idx=8, type=LayerType.Linear, total=10, inp1_unstable=0, inp2_unstable=0, delta_unstable=10

Time for backsubstitution: 1.76 seconds

### RS candidates at layer 1
type: RSZ, layer: 1, pos: 198
type: RSZ, layer: 1, pos: 199
type: RSZ, layer: 1, pos: 204
type: RSZ, layer: 1, pos: 250

Time for candidate selection: 0.20 seconds

### Candidate
type: RSZ, layer: 1, pos: 198

### Relational analysis RSZ of RS_RSZ2_RSZ1_RSZ2_RSZ1_RSZ2_RSZ2_RSZ1

#### Relational analysis RSZ result of RS_RSZ2_RSZ1_RSZ2_RSZ1_RSZ2_RSZ2_RSZ1
Status: Status.UNKNOWN
Output dim: 1, lower bound: -0.0007828, upper bound: 0.0007828
time: 0.60 seconds

### Relational analysis RSZ of RS_RSZ2_RSZ1_RSZ2_RSZ1_RSZ2_RSZ2_RSZ2

#### Relational analysis RSZ result of RS_RSZ2_RSZ1_RSZ2_RSZ1_RSZ2_RSZ2_RSZ2
Status: Status.UNKNOWN
Output dim: 1, lower bound: -0.0007828, upper bound: 0.0007828
time: 0.58 seconds

## BFS RS instance: RS_RSZ2_RSZ1_RSZ2_RSZ2_RSZ1_RSZ1

### Backsubstitution after applying RS history:
0: 0.0006996, 0.0011177, 0.0006996, 0.0011177, -0.0004181, 0.0004181
1: 0.9934013, 0.9943976, 0.9934013, 0.9943976, -0.0009964, 0.0009964
2: -0.0088595, -0.0053283, -0.0088595, -0.0053283, -0.0032435, 0.0032413
3: 0.0035633, 0.0041662, 0.0035633, 0.0041662, -0.0006029, 0.0006029
4: 0.0026282, 0.0054191, 0.0026282, 0.0054191, -0.0027909, 0.0027909
5: 0.0050201, 0.0064446, 0.0050201, 0.0064446, -0.0014245, 0.0014245
6: -0.0021963, -0.0008150, -0.0021963, -0.0008150, -0.0013813, 0.0013813
7: -0.0082640, -0.0074494, -0.0082640, -0.0074494, -0.0008146, 0.0008146
8: 0.0052524, 0.0098919, 0.0052524, 0.0098919, -0.0045621, 0.0045627
9: -0.0036845, -0.0030730, -0.0036845, -0.0030730, -0.0006115, 0.0006115

### Unstable ReLU Count (Linear/Conv2D Layers)
- layer_idx=0, type=LayerType.Linear, total=256, inp1_unstable=10, inp2_unstable=10, delta_unstable=250
- layer_idx=2, type=LayerType.Linear, total=256, inp1_unstable=8, inp2_unstable=8, delta_unstable=256
- layer_idx=4, type=LayerType.Linear, total=256, inp1_unstable=0, inp2_unstable=0, delta_unstable=256
- layer_idx=6, type=LayerType.Linear, total=256, inp1_unstable=2, inp2_unstable=2, delta_unstable=256
- layer_idx=8, type=LayerType.Linear, total=10, inp1_unstable=0, inp2_unstable=0, delta_unstable=10

Time for backsubstitution: 1.74 seconds

### RS candidates at layer 1
type: RSZ, layer: 1, pos: 198
type: RSZ, layer: 1, pos: 199
type: RSZ, layer: 1, pos: 204
type: RSZ, layer: 1, pos: 250

Time for candidate selection: 0.18 seconds

### Candidate
type: RSZ, layer: 1, pos: 198

### Relational analysis RSZ of RS_RSZ2_RSZ1_RSZ2_RSZ2_RSZ1_RSZ1_RSZ1

#### Relational analysis RSZ result of RS_RSZ2_RSZ1_RSZ2_RSZ2_RSZ1_RSZ1_RSZ1
Status: Status.UNKNOWN
Output dim: 1, lower bound: -0.0007828, upper bound: 0.0007828
time: 0.65 seconds

### Relational analysis RSZ of RS_RSZ2_RSZ1_RSZ2_RSZ2_RSZ1_RSZ1_RSZ2

#### Relational analysis RSZ result of RS_RSZ2_RSZ1_RSZ2_RSZ2_RSZ1_RSZ1_RSZ2
Status: Status.UNKNOWN
Output dim: 1, lower bound: -0.0007828, upper bound: 0.0007828
time: 0.65 seconds

## BFS RS instance: RS_RSZ2_RSZ1_RSZ2_RSZ2_RSZ1_RSZ2

### Backsubstitution after applying RS history:
0: 0.0006996, 0.0011177, 0.0006996, 0.0011177, -0.0004181, 0.0004181
1: 0.9934013, 0.9943976, 0.9934013, 0.9943976, -0.0009964, 0.0009964
2: -0.0088595, -0.0053283, -0.0088595, -0.0053283, -0.0032444, 0.0032399
3: 0.0035633, 0.0041662, 0.0035633, 0.0041662, -0.0006029, 0.0006029
4: 0.0026282, 0.0054191, 0.0026282, 0.0054191, -0.0027909, 0.0027909
5: 0.0050201, 0.0064446, 0.0050201, 0.0064446, -0.0014245, 0.0014245
6: -0.0021963, -0.0008150, -0.0021963, -0.0008150, -0.0013813, 0.0013813
7: -0.0082640, -0.0074494, -0.0082640, -0.0074494, -0.0008146, 0.0008146
8: 0.0052524, 0.0098919, 0.0052524, 0.0098919, -0.0045618, 0.0045629
9: -0.0036845, -0.0030730, -0.0036845, -0.0030730, -0.0006115, 0.0006115

### Unstable ReLU Count (Linear/Conv2D Layers)
- layer_idx=0, type=LayerType.Linear, total=256, inp1_unstable=10, inp2_unstable=10, delta_unstable=250
- layer_idx=2, type=LayerType.Linear, total=256, inp1_unstable=8, inp2_unstable=8, delta_unstable=256
- layer_idx=4, type=LayerType.Linear, total=256, inp1_unstable=0, inp2_unstable=0, delta_unstable=256
- layer_idx=6, type=LayerType.Linear, total=256, inp1_unstable=2, inp2_unstable=2, delta_unstable=256
- layer_idx=8, type=LayerType.Linear, total=10, inp1_unstable=0, inp2_unstable=0, delta_unstable=10

Time for backsubstitution: 1.71 seconds

### RS candidates at layer 1
type: RSZ, layer: 1, pos: 198
type: RSZ, layer: 1, pos: 199
type: RSZ, layer: 1, pos: 204
type: RSZ, layer: 1, pos: 250

Time for candidate selection: 0.18 seconds

### Candidate
type: RSZ, layer: 1, pos: 198

### Relational analysis RSZ of RS_RSZ2_RSZ1_RSZ2_RSZ2_RSZ1_RSZ2_RSZ1

#### Relational analysis RSZ result of RS_RSZ2_RSZ1_RSZ2_RSZ2_RSZ1_RSZ2_RSZ1
Status: Status.UNKNOWN
Output dim: 1, lower bound: -0.0007828, upper bound: 0.0007828
time: 0.61 seconds

### Relational analysis RSZ of RS_RSZ2_RSZ1_RSZ2_RSZ2_RSZ1_RSZ2_RSZ2

#### Relational analysis RSZ result of RS_RSZ2_RSZ1_RSZ2_RSZ2_RSZ1_RSZ2_RSZ2
Status: Status.UNKNOWN
Output dim: 1, lower bound: -0.0007828, upper bound: 0.0007828
time: 0.61 seconds

## BFS RS instance: RS_RSZ2_RSZ1_RSZ2_RSZ2_RSZ2_RSZ1

### Backsubstitution after applying RS history:
0: 0.0006996, 0.0011177, 0.0006996, 0.0011177, -0.0004181, 0.0004181
1: 0.9934013, 0.9943976, 0.9934013, 0.9943976, -0.0009964, 0.0009964
2: -0.0088595, -0.0053283, -0.0088595, -0.0053283, -0.0032415, 0.0032436
3: 0.0035633, 0.0041662, 0.0035633, 0.0041662, -0.0006029, 0.0006029
4: 0.0026282, 0.0054191, 0.0026282, 0.0054191, -0.0027909, 0.0027909
5: 0.0050201, 0.0064446, 0.0050201, 0.0064446, -0.0014245, 0.0014245
6: -0.0021963, -0.0008150, -0.0021963, -0.0008150, -0.0013813, 0.0013813
7: -0.0082640, -0.0074494, -0.0082640, -0.0074494, -0.0008146, 0.0008146
8: 0.0052524, 0.0098919, 0.0052524, 0.0098919, -0.0045627, 0.0045622
9: -0.0036845, -0.0030730, -0.0036845, -0.0030730, -0.0006115, 0.0006115

### Unstable ReLU Count (Linear/Conv2D Layers)
- layer_idx=0, type=LayerType.Linear, total=256, inp1_unstable=10, inp2_unstable=10, delta_unstable=250
- layer_idx=2, type=LayerType.Linear, total=256, inp1_unstable=8, inp2_unstable=8, delta_unstable=256
- layer_idx=4, type=LayerType.Linear, total=256, inp1_unstable=0, inp2_unstable=0, delta_unstable=256
- layer_idx=6, type=LayerType.Linear, total=256, inp1_unstable=2, inp2_unstable=2, delta_unstable=256
- layer_idx=8, type=LayerType.Linear, total=10, inp1_unstable=0, inp2_unstable=0, delta_unstable=10

Time for backsubstitution: 1.70 seconds

### RS candidates at layer 1
type: RSZ, layer: 1, pos: 198
type: RSZ, layer: 1, pos: 199
type: RSZ, layer: 1, pos: 204
type: RSZ, layer: 1, pos: 250

Time for candidate selection: 0.19 seconds

### Candidate
type: RSZ, layer: 1, pos: 198

### Relational analysis RSZ of RS_RSZ2_RSZ1_RSZ2_RSZ2_RSZ2_RSZ1_RSZ1

#### Relational analysis RSZ result of RS_RSZ2_RSZ1_RSZ2_RSZ2_RSZ2_RSZ1_RSZ1
Status: Status.UNKNOWN
Output dim: 1, lower bound: -0.0007828, upper bound: 0.0007828
time: 0.70 seconds

### Relational analysis RSZ of RS_RSZ2_RSZ1_RSZ2_RSZ2_RSZ2_RSZ1_RSZ2

#### Relational analysis RSZ result of RS_RSZ2_RSZ1_RSZ2_RSZ2_RSZ2_RSZ1_RSZ2
Status: Status.UNKNOWN
Output dim: 1, lower bound: -0.0007828, upper bound: 0.0007828
time: 0.70 seconds

## BFS RS instance: RS_RSZ2_RSZ1_RSZ2_RSZ2_RSZ2_RSZ2

### Backsubstitution after applying RS history:
0: 0.0006996, 0.0011177, 0.0006996, 0.0011177, -0.0004181, 0.0004181
1: 0.9934013, 0.9943976, 0.9934013, 0.9943976, -0.0009964, 0.0009964
2: -0.0088595, -0.0053283, -0.0088595, -0.0053283, -0.0032423, 0.0032420
3: 0.0035633, 0.0041662, 0.0035633, 0.0041662, -0.0006029, 0.0006029
4: 0.0026282, 0.0054191, 0.0026282, 0.0054191, -0.0027909, 0.0027909
5: 0.0050201, 0.0064446, 0.0050201, 0.0064446, -0.0014245, 0.0014245
6: -0.0021963, -0.0008150, -0.0021963, -0.0008150, -0.0013813, 0.0013813
7: -0.0082640, -0.0074494, -0.0082640, -0.0074494, -0.0008146, 0.0008146
8: 0.0052524, 0.0098919, 0.0052524, 0.0098919, -0.0045623, 0.0045624
9: -0.0036845, -0.0030730, -0.0036845, -0.0030730, -0.0006115, 0.0006115

### Unstable ReLU Count (Linear/Conv2D Layers)
- layer_idx=0, type=LayerType.Linear, total=256, inp1_unstable=10, inp2_unstable=10, delta_unstable=250
- layer_idx=2, type=LayerType.Linear, total=256, inp1_unstable=8, inp2_unstable=8, delta_unstable=256
- layer_idx=4, type=LayerType.Linear, total=256, inp1_unstable=0, inp2_unstable=0, delta_unstable=256
- layer_idx=6, type=LayerType.Linear, total=256, inp1_unstable=2, inp2_unstable=2, delta_unstable=256
- layer_idx=8, type=LayerType.Linear, total=10, inp1_unstable=0, inp2_unstable=0, delta_unstable=10

Time for backsubstitution: 1.70 seconds

### RS candidates at layer 1
type: RSZ, layer: 1, pos: 198
type: RSZ, layer: 1, pos: 199
type: RSZ, layer: 1, pos: 204
type: RSZ, layer: 1, pos: 250

Time for candidate selection: 0.20 seconds

### Candidate
type: RSZ, layer: 1, pos: 198

### Relational analysis RSZ of RS_RSZ2_RSZ1_RSZ2_RSZ2_RSZ2_RSZ2_RSZ1

#### Relational analysis RSZ result of RS_RSZ2_RSZ1_RSZ2_RSZ2_RSZ2_RSZ2_RSZ1
Status: Status.UNKNOWN
Output dim: 1, lower bound: -0.0007828, upper bound: 0.0007828
time: 0.57 seconds

### Relational analysis RSZ of RS_RSZ2_RSZ1_RSZ2_RSZ2_RSZ2_RSZ2_RSZ2

#### Relational analysis RSZ result of RS_RSZ2_RSZ1_RSZ2_RSZ2_RSZ2_RSZ2_RSZ2
Status: Status.UNKNOWN
Output dim: 1, lower bound: -0.0007828, upper bound: 0.0007828
time: 0.58 seconds

## BFS RS instance: RS_RSZ2_RSZ2_RSZ1_RSZ1_RSZ1_RSZ1

### Backsubstitution after applying RS history:
0: 0.0006996, 0.0011177, 0.0006996, 0.0011177, -0.0004181, 0.0004181
1: 0.9934013, 0.9943976, 0.9934013, 0.9943976, -0.0009964, 0.0009964
2: -0.0088595, -0.0053283, -0.0088595, -0.0053283, -0.0032418, 0.0032422
3: 0.0035633, 0.0041662, 0.0035633, 0.0041662, -0.0006029, 0.0006029
4: 0.0026282, 0.0054191, 0.0026282, 0.0054191, -0.0027909, 0.0027909
5: 0.0050201, 0.0064446, 0.0050201, 0.0064446, -0.0014245, 0.0014245
6: -0.0021963, -0.0008150, -0.0021963, -0.0008150, -0.0013813, 0.0013813
7: -0.0082640, -0.0074494, -0.0082640, -0.0074494, -0.0008146, 0.0008146
8: 0.0052524, 0.0098919, 0.0052524, 0.0098919, -0.0045624, 0.0045623
9: -0.0036845, -0.0030730, -0.0036845, -0.0030730, -0.0006115, 0.0006115

### Unstable ReLU Count (Linear/Conv2D Layers)
- layer_idx=0, type=LayerType.Linear, total=256, inp1_unstable=10, inp2_unstable=10, delta_unstable=250
- layer_idx=2, type=LayerType.Linear, total=256, inp1_unstable=8, inp2_unstable=8, delta_unstable=256
- layer_idx=4, type=LayerType.Linear, total=256, inp1_unstable=0, inp2_unstable=0, delta_unstable=256
- layer_idx=6, type=LayerType.Linear, total=256, inp1_unstable=2, inp2_unstable=2, delta_unstable=256
- layer_idx=8, type=LayerType.Linear, total=10, inp1_unstable=0, inp2_unstable=0, delta_unstable=10

Time for backsubstitution: 1.73 seconds

### RS candidates at layer 1
type: RSZ, layer: 1, pos: 198
type: RSZ, layer: 1, pos: 199
type: RSZ, layer: 1, pos: 204
type: RSZ, layer: 1, pos: 250

Time for candidate selection: 0.20 seconds

### Candidate
type: RSZ, layer: 1, pos: 198

### Relational analysis RSZ of RS_RSZ2_RSZ2_RSZ1_RSZ1_RSZ1_RSZ1_RSZ1

#### Relational analysis RSZ result of RS_RSZ2_RSZ2_RSZ1_RSZ1_RSZ1_RSZ1_RSZ1
Status: Status.UNKNOWN
Output dim: 1, lower bound: -0.0007828, upper bound: 0.0007828
time: 0.62 seconds

### Relational analysis RSZ of RS_RSZ2_RSZ2_RSZ1_RSZ1_RSZ1_RSZ1_RSZ2

#### Relational analysis RSZ result of RS_RSZ2_RSZ2_RSZ1_RSZ1_RSZ1_RSZ1_RSZ2
Status: Status.UNKNOWN
Output dim: 1, lower bound: -0.0007828, upper bound: 0.0007828
time: 0.59 seconds

## BFS RS instance: RS_RSZ2_RSZ2_RSZ1_RSZ1_RSZ1_RSZ2

### Backsubstitution after applying RS history:
0: 0.0006996, 0.0011177, 0.0006996, 0.0011177, -0.0004181, 0.0004181
1: 0.9934013, 0.9943976, 0.9934013, 0.9943976, -0.0009964, 0.0009964
2: -0.0088595, -0.0053283, -0.0088595, -0.0053283, -0.0032434, 0.0032410
3: 0.0035633, 0.0041662, 0.0035633, 0.0041662, -0.0006029, 0.0006029
4: 0.0026282, 0.0054191, 0.0026282, 0.0054191, -0.0027909, 0.0027909
5: 0.0050201, 0.0064446, 0.0050201, 0.0064446, -0.0014245, 0.0014245
6: -0.0021963, -0.0008150, -0.0021963, -0.0008150, -0.0013813, 0.0013813
7: -0.0082640, -0.0074494, -0.0082640, -0.0074494, -0.0008146, 0.0008146
8: 0.0052524, 0.0098919, 0.0052524, 0.0098919, -0.0045620, 0.0045627
9: -0.0036845, -0.0030730, -0.0036845, -0.0030730, -0.0006115, 0.0006115

### Unstable ReLU Count (Linear/Conv2D Layers)
- layer_idx=0, type=LayerType.Linear, total=256, inp1_unstable=10, inp2_unstable=10, delta_unstable=250
- layer_idx=2, type=LayerType.Linear, total=256, inp1_unstable=8, inp2_unstable=8, delta_unstable=256
- layer_idx=4, type=LayerType.Linear, total=256, inp1_unstable=0, inp2_unstable=0, delta_unstable=256
- layer_idx=6, type=LayerType.Linear, total=256, inp1_unstable=2, inp2_unstable=2, delta_unstable=256
- layer_idx=8, type=LayerType.Linear, total=10, inp1_unstable=0, inp2_unstable=0, delta_unstable=10

Time for backsubstitution: 1.82 seconds

### RS candidates at layer 1
type: RSZ, layer: 1, pos: 198
type: RSZ, layer: 1, pos: 199
type: RSZ, layer: 1, pos: 204
type: RSZ, layer: 1, pos: 250

Time for candidate selection: 0.19 seconds

### Candidate
type: RSZ, layer: 1, pos: 198

### Relational analysis RSZ of RS_RSZ2_RSZ2_RSZ1_RSZ1_RSZ1_RSZ2_RSZ1

#### Relational analysis RSZ result of RS_RSZ2_RSZ2_RSZ1_RSZ1_RSZ1_RSZ2_RSZ1
Status: Status.UNKNOWN
Output dim: 1, lower bound: -0.0007828, upper bound: 0.0007828
time: 0.56 seconds

### Relational analysis RSZ of RS_RSZ2_RSZ2_RSZ1_RSZ1_RSZ1_RSZ2_RSZ2

#### Relational analysis RSZ result of RS_RSZ2_RSZ2_RSZ1_RSZ1_RSZ1_RSZ2_RSZ2
Status: Status.UNKNOWN
Output dim: 1, lower bound: -0.0007828, upper bound: 0.0007828
time: 0.57 seconds

## BFS RS instance: RS_RSZ2_RSZ2_RSZ1_RSZ1_RSZ2_RSZ1

### Backsubstitution after applying RS history:
0: 0.0006996, 0.0011177, 0.0006996, 0.0011177, -0.0004181, 0.0004181
1: 0.9934013, 0.9943976, 0.9934013, 0.9943976, -0.0009964, 0.0009964
2: -0.0088595, -0.0053283, -0.0088595, -0.0053283, -0.0032399, 0.0032443
3: 0.0035633, 0.0041662, 0.0035633, 0.0041662, -0.0006029, 0.0006029
4: 0.0026282, 0.0054191, 0.0026282, 0.0054191, -0.0027909, 0.0027909
5: 0.0050201, 0.0064446, 0.0050201, 0.0064446, -0.0014245, 0.0014245
6: -0.0021963, -0.0008150, -0.0021963, -0.0008150, -0.0013813, 0.0013813
7: -0.0082640, -0.0074494, -0.0082640, -0.0074494, -0.0008146, 0.0008146
8: 0.0052524, 0.0098919, 0.0052524, 0.0098919, -0.0045629, 0.0045618
9: -0.0036845, -0.0030730, -0.0036845, -0.0030730, -0.0006115, 0.0006115

### Unstable ReLU Count (Linear/Conv2D Layers)
- layer_idx=0, type=LayerType.Linear, total=256, inp1_unstable=10, inp2_unstable=10, delta_unstable=250
- layer_idx=2, type=LayerType.Linear, total=256, inp1_unstable=8, inp2_unstable=8, delta_unstable=256
- layer_idx=4, type=LayerType.Linear, total=256, inp1_unstable=0, inp2_unstable=0, delta_unstable=256
- layer_idx=6, type=LayerType.Linear, total=256, inp1_unstable=2, inp2_unstable=2, delta_unstable=256
- layer_idx=8, type=LayerType.Linear, total=10, inp1_unstable=0, inp2_unstable=0, delta_unstable=10

Time for backsubstitution: 1.74 seconds

### RS candidates at layer 1
type: RSZ, layer: 1, pos: 198
type: RSZ, layer: 1, pos: 199
type: RSZ, layer: 1, pos: 204
type: RSZ, layer: 1, pos: 250

Time for candidate selection: 0.21 seconds

### Candidate
type: RSZ, layer: 1, pos: 198

### Relational analysis RSZ of RS_RSZ2_RSZ2_RSZ1_RSZ1_RSZ2_RSZ1_RSZ1

#### Relational analysis RSZ result of RS_RSZ2_RSZ2_RSZ1_RSZ1_RSZ2_RSZ1_RSZ1
Status: Status.UNKNOWN
Output dim: 1, lower bound: -0.0007828, upper bound: 0.0007828
time: 0.65 seconds

### Relational analysis RSZ of RS_RSZ2_RSZ2_RSZ1_RSZ1_RSZ2_RSZ1_RSZ2

#### Relational analysis RSZ result of RS_RSZ2_RSZ2_RSZ1_RSZ1_RSZ2_RSZ1_RSZ2
Status: Status.UNKNOWN
Output dim: 1, lower bound: -0.0007828, upper bound: 0.0007828
time: 0.64 seconds

## BFS RS instance: RS_RSZ2_RSZ2_RSZ1_RSZ1_RSZ2_RSZ2

### Backsubstitution after applying RS history:
0: 0.0006996, 0.0011177, 0.0006996, 0.0011177, -0.0004181, 0.0004181
1: 0.9934013, 0.9943976, 0.9934013, 0.9943976, -0.0009964, 0.0009964
2: -0.0088595, -0.0053283, -0.0088595, -0.0053283, -0.0032413, 0.0032431
3: 0.0035633, 0.0041662, 0.0035633, 0.0041662, -0.0006029, 0.0006029
4: 0.0026282, 0.0054191, 0.0026282, 0.0054191, -0.0027909, 0.0027909
5: 0.0050201, 0.0064446, 0.0050201, 0.0064446, -0.0014245, 0.0014245
6: -0.0021963, -0.0008150, -0.0021963, -0.0008150, -0.0013813, 0.0013813
7: -0.0082640, -0.0074494, -0.0082640, -0.0074494, -0.0008146, 0.0008146
8: 0.0052524, 0.0098919, 0.0052524, 0.0098919, -0.0045626, 0.0045621
9: -0.0036845, -0.0030730, -0.0036845, -0.0030730, -0.0006115, 0.0006115

### Unstable ReLU Count (Linear/Conv2D Layers)
- layer_idx=0, type=LayerType.Linear, total=256, inp1_unstable=10, inp2_unstable=10, delta_unstable=250
- layer_idx=2, type=LayerType.Linear, total=256, inp1_unstable=8, inp2_unstable=8, delta_unstable=256
- layer_idx=4, type=LayerType.Linear, total=256, inp1_unstable=0, inp2_unstable=0, delta_unstable=256
- layer_idx=6, type=LayerType.Linear, total=256, inp1_unstable=2, inp2_unstable=2, delta_unstable=256
- layer_idx=8, type=LayerType.Linear, total=10, inp1_unstable=0, inp2_unstable=0, delta_unstable=10

Time for backsubstitution: 1.85 seconds

### RS candidates at layer 1
type: RSZ, layer: 1, pos: 198
type: RSZ, layer: 1, pos: 199
type: RSZ, layer: 1, pos: 204
type: RSZ, layer: 1, pos: 250

Time for candidate selection: 0.21 seconds

### Candidate
type: RSZ, layer: 1, pos: 198

### Relational analysis RSZ of RS_RSZ2_RSZ2_RSZ1_RSZ1_RSZ2_RSZ2_RSZ1

#### Relational analysis RSZ result of RS_RSZ2_RSZ2_RSZ1_RSZ1_RSZ2_RSZ2_RSZ1
Status: Status.UNKNOWN
Output dim: 1, lower bound: -0.0007828, upper bound: 0.0007828
time: 0.62 seconds

### Relational analysis RSZ of RS_RSZ2_RSZ2_RSZ1_RSZ1_RSZ2_RSZ2_RSZ2

#### Relational analysis RSZ result of RS_RSZ2_RSZ2_RSZ1_RSZ1_RSZ2_RSZ2_RSZ2
Status: Status.UNKNOWN
Output dim: 1, lower bound: -0.0007828, upper bound: 0.0007828
time: 0.61 seconds

## BFS RS instance: RS_RSZ2_RSZ2_RSZ1_RSZ2_RSZ1_RSZ1

### Backsubstitution after applying RS history:
0: 0.0006996, 0.0011177, 0.0006996, 0.0011177, -0.0004181, 0.0004181
1: 0.9934013, 0.9943976, 0.9934013, 0.9943976, -0.0009964, 0.0009964
2: -0.0088595, -0.0053283, -0.0088595, -0.0053283, -0.0032408, 0.0032432
3: 0.0035633, 0.0041662, 0.0035633, 0.0041662, -0.0006029, 0.0006029
4: 0.0026282, 0.0054191, 0.0026282, 0.0054191, -0.0027909, 0.0027909
5: 0.0050201, 0.0064446, 0.0050201, 0.0064446, -0.0014245, 0.0014245
6: -0.0021963, -0.0008150, -0.0021963, -0.0008150, -0.0013813, 0.0013813
7: -0.0082640, -0.0074494, -0.0082640, -0.0074494, -0.0008146, 0.0008146
8: 0.0052524, 0.0098919, 0.0052524, 0.0098919, -0.0045626, 0.0045620
9: -0.0036845, -0.0030730, -0.0036845, -0.0030730, -0.0006115, 0.0006115

### Unstable ReLU Count (Linear/Conv2D Layers)
- layer_idx=0, type=LayerType.Linear, total=256, inp1_unstable=10, inp2_unstable=10, delta_unstable=250
- layer_idx=2, type=LayerType.Linear, total=256, inp1_unstable=8, inp2_unstable=8, delta_unstable=256
- layer_idx=4, type=LayerType.Linear, total=256, inp1_unstable=0, inp2_unstable=0, delta_unstable=256
- layer_idx=6, type=LayerType.Linear, total=256, inp1_unstable=2, inp2_unstable=2, delta_unstable=256
- layer_idx=8, type=LayerType.Linear, total=10, inp1_unstable=0, inp2_unstable=0, delta_unstable=10

Time for backsubstitution: 1.68 seconds

### RS candidates at layer 1
type: RSZ, layer: 1, pos: 198
type: RSZ, layer: 1, pos: 199
type: RSZ, layer: 1, pos: 204
type: RSZ, layer: 1, pos: 250

Time for candidate selection: 0.18 seconds

### Candidate
type: RSZ, layer: 1, pos: 198

### Relational analysis RSZ of RS_RSZ2_RSZ2_RSZ1_RSZ2_RSZ1_RSZ1_RSZ1

#### Relational analysis RSZ result of RS_RSZ2_RSZ2_RSZ1_RSZ2_RSZ1_RSZ1_RSZ1
Status: Status.UNKNOWN
Output dim: 1, lower bound: -0.0007828, upper bound: 0.0007828
time: 0.61 seconds

### Relational analysis RSZ of RS_RSZ2_RSZ2_RSZ1_RSZ2_RSZ1_RSZ1_RSZ2

#### Relational analysis RSZ result of RS_RSZ2_RSZ2_RSZ1_RSZ2_RSZ1_RSZ1_RSZ2
Status: Status.UNKNOWN
Output dim: 1, lower bound: -0.0007828, upper bound: 0.0007828
time: 0.60 seconds

## BFS RS instance: RS_RSZ2_RSZ2_RSZ1_RSZ2_RSZ1_RSZ2

### Backsubstitution after applying RS history:
0: 0.0006996, 0.0011177, 0.0006996, 0.0011177, -0.0004181, 0.0004181
1: 0.9934013, 0.9943976, 0.9934013, 0.9943976, -0.0009964, 0.0009964
2: -0.0088595, -0.0053283, -0.0088595, -0.0053283, -0.0032420, 0.0032423
3: 0.0035633, 0.0041662, 0.0035633, 0.0041662, -0.0006029, 0.0006029
4: 0.0026282, 0.0054191, 0.0026282, 0.0054191, -0.0027909, 0.0027909
5: 0.0050201, 0.0064446, 0.0050201, 0.0064446, -0.0014245, 0.0014245
6: -0.0021963, -0.0008150, -0.0021963, -0.0008150, -0.0013813, 0.0013813
7: -0.0082640, -0.0074494, -0.0082640, -0.0074494, -0.0008146, 0.0008146
8: 0.0052524, 0.0098919, 0.0052524, 0.0098919, -0.0045624, 0.0045623
9: -0.0036845, -0.0030730, -0.0036845, -0.0030730, -0.0006115, 0.0006115

### Unstable ReLU Count (Linear/Conv2D Layers)
- layer_idx=0, type=LayerType.Linear, total=256, inp1_unstable=10, inp2_unstable=10, delta_unstable=250
- layer_idx=2, type=LayerType.Linear, total=256, inp1_unstable=8, inp2_unstable=8, delta_unstable=256
- layer_idx=4, type=LayerType.Linear, total=256, inp1_unstable=0, inp2_unstable=0, delta_unstable=256
- layer_idx=6, type=LayerType.Linear, total=256, inp1_unstable=2, inp2_unstable=2, delta_unstable=256
- layer_idx=8, type=LayerType.Linear, total=10, inp1_unstable=0, inp2_unstable=0, delta_unstable=10

Time for backsubstitution: 1.79 seconds

### RS candidates at layer 1
type: RSZ, layer: 1, pos: 198
type: RSZ, layer: 1, pos: 199
type: RSZ, layer: 1, pos: 204
type: RSZ, layer: 1, pos: 250

Time for candidate selection: 0.18 seconds

### Candidate
type: RSZ, layer: 1, pos: 198

### Relational analysis RSZ of RS_RSZ2_RSZ2_RSZ1_RSZ2_RSZ1_RSZ2_RSZ1

#### Relational analysis RSZ result of RS_RSZ2_RSZ2_RSZ1_RSZ2_RSZ1_RSZ2_RSZ1
Status: Status.UNKNOWN
Output dim: 1, lower bound: -0.0007828, upper bound: 0.0007828
time: 0.56 seconds

### Relational analysis RSZ of RS_RSZ2_RSZ2_RSZ1_RSZ2_RSZ1_RSZ2_RSZ2

#### Relational analysis RSZ result of RS_RSZ2_RSZ2_RSZ1_RSZ2_RSZ1_RSZ2_RSZ2
Status: Status.UNKNOWN
Output dim: 1, lower bound: -0.0007828, upper bound: 0.0007828
time: 0.56 seconds

## BFS RS instance: RS_RSZ2_RSZ2_RSZ1_RSZ2_RSZ2_RSZ1

### Backsubstitution after applying RS history:
0: 0.0006996, 0.0011177, 0.0006996, 0.0011177, -0.0004181, 0.0004181
1: 0.9934013, 0.9943976, 0.9934013, 0.9943976, -0.0009964, 0.0009964
2: -0.0088595, -0.0053283, -0.0088595, -0.0053283, -0.0032390, 0.0032451
3: 0.0035633, 0.0041662, 0.0035633, 0.0041662, -0.0006029, 0.0006029
4: 0.0026282, 0.0054191, 0.0026282, 0.0054191, -0.0027909, 0.0027909
5: 0.0050201, 0.0064446, 0.0050201, 0.0064446, -0.0014245, 0.0014245
6: -0.0021963, -0.0008150, -0.0021963, -0.0008150, -0.0013813, 0.0013813
7: -0.0082640, -0.0074494, -0.0082640, -0.0074494, -0.0008146, 0.0008146
8: 0.0052524, 0.0098919, 0.0052524, 0.0098919, -0.0045632, 0.0045615
9: -0.0036845, -0.0030730, -0.0036845, -0.0030730, -0.0006115, 0.0006115

### Unstable ReLU Count (Linear/Conv2D Layers)
- layer_idx=0, type=LayerType.Linear, total=256, inp1_unstable=10, inp2_unstable=10, delta_unstable=250
- layer_idx=2, type=LayerType.Linear, total=256, inp1_unstable=8, inp2_unstable=8, delta_unstable=256
- layer_idx=4, type=LayerType.Linear, total=256, inp1_unstable=0, inp2_unstable=0, delta_unstable=256
- layer_idx=6, type=LayerType.Linear, total=256, inp1_unstable=2, inp2_unstable=2, delta_unstable=256
- layer_idx=8, type=LayerType.Linear, total=10, inp1_unstable=0, inp2_unstable=0, delta_unstable=10

Time for backsubstitution: 1.70 seconds

### RS candidates at layer 1
type: RSZ, layer: 1, pos: 198
type: RSZ, layer: 1, pos: 199
type: RSZ, layer: 1, pos: 204
type: RSZ, layer: 1, pos: 250

Time for candidate selection: 0.19 seconds

### Candidate
type: RSZ, layer: 1, pos: 198

### Relational analysis RSZ of RS_RSZ2_RSZ2_RSZ1_RSZ2_RSZ2_RSZ1_RSZ1

#### Relational analysis RSZ result of RS_RSZ2_RSZ2_RSZ1_RSZ2_RSZ2_RSZ1_RSZ1
Status: Status.UNKNOWN
Output dim: 1, lower bound: -0.0007828, upper bound: 0.0007828
time: 0.61 seconds

### Relational analysis RSZ of RS_RSZ2_RSZ2_RSZ1_RSZ2_RSZ2_RSZ1_RSZ2

#### Relational analysis RSZ result of RS_RSZ2_RSZ2_RSZ1_RSZ2_RSZ2_RSZ1_RSZ2
Status: Status.UNKNOWN
Output dim: 1, lower bound: -0.0007828, upper bound: 0.0007828
time: 0.64 seconds

## BFS RS instance: RS_RSZ2_RSZ2_RSZ1_RSZ2_RSZ2_RSZ2

### Backsubstitution after applying RS history:
0: 0.0006996, 0.0011177, 0.0006996, 0.0011177, -0.0004181, 0.0004181
1: 0.9934013, 0.9943976, 0.9934013, 0.9943976, -0.0009964, 0.0009964
2: -0.0088595, -0.0053283, -0.0088595, -0.0053283, -0.0032400, 0.0032439
3: 0.0035633, 0.0041662, 0.0035633, 0.0041662, -0.0006029, 0.0006029
4: 0.0026282, 0.0054191, 0.0026282, 0.0054191, -0.0027909, 0.0027909
5: 0.0050201, 0.0064446, 0.0050201, 0.0064446, -0.0014245, 0.0014245
6: -0.0021963, -0.0008150, -0.0021963, -0.0008150, -0.0013813, 0.0013813
7: -0.0082640, -0.0074494, -0.0082640, -0.0074494, -0.0008146, 0.0008146
8: 0.0052524, 0.0098919, 0.0052524, 0.0098919, -0.0045628, 0.0045618
9: -0.0036845, -0.0030730, -0.0036845, -0.0030730, -0.0006115, 0.0006115

### Unstable ReLU Count (Linear/Conv2D Layers)
- layer_idx=0, type=LayerType.Linear, total=256, inp1_unstable=10, inp2_unstable=10, delta_unstable=250
- layer_idx=2, type=LayerType.Linear, total=256, inp1_unstable=8, inp2_unstable=8, delta_unstable=256
- layer_idx=4, type=LayerType.Linear, total=256, inp1_unstable=0, inp2_unstable=0, delta_unstable=256
- layer_idx=6, type=LayerType.Linear, total=256, inp1_unstable=2, inp2_unstable=2, delta_unstable=256
- layer_idx=8, type=LayerType.Linear, total=10, inp1_unstable=0, inp2_unstable=0, delta_unstable=10

Time for backsubstitution: 1.73 seconds

### RS candidates at layer 1
type: RSZ, layer: 1, pos: 198
type: RSZ, layer: 1, pos: 199
type: RSZ, layer: 1, pos: 204
type: RSZ, layer: 1, pos: 250

Time for candidate selection: 0.18 seconds

### Candidate
type: RSZ, layer: 1, pos: 198

### Relational analysis RSZ of RS_RSZ2_RSZ2_RSZ1_RSZ2_RSZ2_RSZ2_RSZ1

#### Relational analysis RSZ result of RS_RSZ2_RSZ2_RSZ1_RSZ2_RSZ2_RSZ2_RSZ1
Status: Status.UNKNOWN
Output dim: 1, lower bound: -0.0007828, upper bound: 0.0007828
time: 0.61 seconds

### Relational analysis RSZ of RS_RSZ2_RSZ2_RSZ1_RSZ2_RSZ2_RSZ2_RSZ2

#### Relational analysis RSZ result of RS_RSZ2_RSZ2_RSZ1_RSZ2_RSZ2_RSZ2_RSZ2
Status: Status.UNKNOWN
Output dim: 1, lower bound: -0.0007828, upper bound: 0.0007828
time: 0.61 seconds

## BFS RS instance: RS_RSZ2_RSZ2_RSZ2_RSZ1_RSZ1_RSZ1

### Backsubstitution after applying RS history:
0: 0.0006996, 0.0011177, 0.0006996, 0.0011177, -0.0004181, 0.0004181
1: 0.9934013, 0.9943976, 0.9934013, 0.9943976, -0.0009964, 0.0009964
2: -0.0088595, -0.0053283, -0.0088595, -0.0053283, -0.0032417, 0.0032424
3: 0.0035633, 0.0041662, 0.0035633, 0.0041662, -0.0006029, 0.0006029
4: 0.0026282, 0.0054191, 0.0026282, 0.0054191, -0.0027909, 0.0027909
5: 0.0050201, 0.0064446, 0.0050201, 0.0064446, -0.0014245, 0.0014245
6: -0.0021963, -0.0008150, -0.0021963, -0.0008150, -0.0013813, 0.0013813
7: -0.0082640, -0.0074494, -0.0082640, -0.0074494, -0.0008146, 0.0008146
8: 0.0052524, 0.0098919, 0.0052524, 0.0098919, -0.0045624, 0.0045622
9: -0.0036845, -0.0030730, -0.0036845, -0.0030730, -0.0006115, 0.0006115

### Unstable ReLU Count (Linear/Conv2D Layers)
- layer_idx=0, type=LayerType.Linear, total=256, inp1_unstable=10, inp2_unstable=10, delta_unstable=250
- layer_idx=2, type=LayerType.Linear, total=256, inp1_unstable=8, inp2_unstable=8, delta_unstable=256
- layer_idx=4, type=LayerType.Linear, total=256, inp1_unstable=0, inp2_unstable=0, delta_unstable=256
- layer_idx=6, type=LayerType.Linear, total=256, inp1_unstable=2, inp2_unstable=2, delta_unstable=256
- layer_idx=8, type=LayerType.Linear, total=10, inp1_unstable=0, inp2_unstable=0, delta_unstable=10

Time for backsubstitution: 1.72 seconds

### RS candidates at layer 1
type: RSZ, layer: 1, pos: 198
type: RSZ, layer: 1, pos: 199
type: RSZ, layer: 1, pos: 204
type: RSZ, layer: 1, pos: 250

Time for candidate selection: 0.18 seconds

### Candidate
type: RSZ, layer: 1, pos: 198

### Relational analysis RSZ of RS_RSZ2_RSZ2_RSZ2_RSZ1_RSZ1_RSZ1_RSZ1

#### Relational analysis RSZ result of RS_RSZ2_RSZ2_RSZ2_RSZ1_RSZ1_RSZ1_RSZ1
Status: Status.UNKNOWN
Output dim: 1, lower bound: -0.0007828, upper bound: 0.0007828
time: 0.62 seconds

### Relational analysis RSZ of RS_RSZ2_RSZ2_RSZ2_RSZ1_RSZ1_RSZ1_RSZ2

#### Relational analysis RSZ result of RS_RSZ2_RSZ2_RSZ2_RSZ1_RSZ1_RSZ1_RSZ2
Status: Status.UNKNOWN
Output dim: 1, lower bound: -0.0007828, upper bound: 0.0007828
time: 0.60 seconds

## BFS RS instance: RS_RSZ2_RSZ2_RSZ2_RSZ1_RSZ1_RSZ2

### Backsubstitution after applying RS history:
0: 0.0006996, 0.0011177, 0.0006996, 0.0011177, -0.0004181, 0.0004181
1: 0.9934013, 0.9943976, 0.9934013, 0.9943976, -0.0009964, 0.0009964
2: -0.0088595, -0.0053283, -0.0088595, -0.0053283, -0.0032431, 0.0032415
3: 0.0035633, 0.0041662, 0.0035633, 0.0041662, -0.0006029, 0.0006029
4: 0.0026282, 0.0054191, 0.0026282, 0.0054191, -0.0027909, 0.0027909
5: 0.0050201, 0.0064446, 0.0050201, 0.0064446, -0.0014245, 0.0014245
6: -0.0021963, -0.0008150, -0.0021963, -0.0008150, -0.0013813, 0.0013813
7: -0.0082640, -0.0074494, -0.0082640, -0.0074494, -0.0008146, 0.0008146
8: 0.0052524, 0.0098919, 0.0052524, 0.0098919, -0.0045622, 0.0045626
9: -0.0036845, -0.0030730, -0.0036845, -0.0030730, -0.0006115, 0.0006115

### Unstable ReLU Count (Linear/Conv2D Layers)
- layer_idx=0, type=LayerType.Linear, total=256, inp1_unstable=10, inp2_unstable=10, delta_unstable=250
- layer_idx=2, type=LayerType.Linear, total=256, inp1_unstable=8, inp2_unstable=8, delta_unstable=256
- layer_idx=4, type=LayerType.Linear, total=256, inp1_unstable=0, inp2_unstable=0, delta_unstable=256
- layer_idx=6, type=LayerType.Linear, total=256, inp1_unstable=2, inp2_unstable=2, delta_unstable=256
- layer_idx=8, type=LayerType.Linear, total=10, inp1_unstable=0, inp2_unstable=0, delta_unstable=10

Time for backsubstitution: 1.73 seconds

### RS candidates at layer 1
type: RSZ, layer: 1, pos: 198
type: RSZ, layer: 1, pos: 199
type: RSZ, layer: 1, pos: 204
type: RSZ, layer: 1, pos: 250

Time for candidate selection: 0.19 seconds

### Candidate
type: RSZ, layer: 1, pos: 198

### Relational analysis RSZ of RS_RSZ2_RSZ2_RSZ2_RSZ1_RSZ1_RSZ2_RSZ1

#### Relational analysis RSZ result of RS_RSZ2_RSZ2_RSZ2_RSZ1_RSZ1_RSZ2_RSZ1
Status: Status.UNKNOWN
Output dim: 1, lower bound: -0.0007828, upper bound: 0.0007828
time: 0.58 seconds

### Relational analysis RSZ of RS_RSZ2_RSZ2_RSZ2_RSZ1_RSZ1_RSZ2_RSZ2

#### Relational analysis RSZ result of RS_RSZ2_RSZ2_RSZ2_RSZ1_RSZ1_RSZ2_RSZ2
Status: Status.UNKNOWN
Output dim: 1, lower bound: -0.0007828, upper bound: 0.0007828
time: 0.58 seconds

## BFS RS instance: RS_RSZ2_RSZ2_RSZ2_RSZ1_RSZ2_RSZ1

### Backsubstitution after applying RS history:
0: 0.0006996, 0.0011177, 0.0006996, 0.0011177, -0.0004181, 0.0004181
1: 0.9934013, 0.9943976, 0.9934013, 0.9943976, -0.0009964, 0.0009964
2: -0.0088595, -0.0053283, -0.0088595, -0.0053283, -0.0032403, 0.0032446
3: 0.0035633, 0.0041662, 0.0035633, 0.0041662, -0.0006029, 0.0006029
4: 0.0026282, 0.0054191, 0.0026282, 0.0054191, -0.0027909, 0.0027909
5: 0.0050201, 0.0064446, 0.0050201, 0.0064446, -0.0014245, 0.0014245
6: -0.0021963, -0.0008150, -0.0021963, -0.0008150, -0.0013813, 0.0013813
7: -0.0082640, -0.0074494, -0.0082640, -0.0074494, -0.0008146, 0.0008146
8: 0.0052524, 0.0098919, 0.0052524, 0.0098919, -0.0045630, 0.0045619
9: -0.0036845, -0.0030730, -0.0036845, -0.0030730, -0.0006115, 0.0006115

### Unstable ReLU Count (Linear/Conv2D Layers)
- layer_idx=0, type=LayerType.Linear, total=256, inp1_unstable=10, inp2_unstable=10, delta_unstable=250
- layer_idx=2, type=LayerType.Linear, total=256, inp1_unstable=8, inp2_unstable=8, delta_unstable=256
- layer_idx=4, type=LayerType.Linear, total=256, inp1_unstable=0, inp2_unstable=0, delta_unstable=256
- layer_idx=6, type=LayerType.Linear, total=256, inp1_unstable=2, inp2_unstable=2, delta_unstable=256
- layer_idx=8, type=LayerType.Linear, total=10, inp1_unstable=0, inp2_unstable=0, delta_unstable=10

Time for backsubstitution: 1.70 seconds

### RS candidates at layer 1
type: RSZ, layer: 1, pos: 198
type: RSZ, layer: 1, pos: 199
type: RSZ, layer: 1, pos: 204
type: RSZ, layer: 1, pos: 250

Time for candidate selection: 0.18 seconds

### Candidate
type: RSZ, layer: 1, pos: 198

### Relational analysis RSZ of RS_RSZ2_RSZ2_RSZ2_RSZ1_RSZ2_RSZ1_RSZ1

#### Relational analysis RSZ result of RS_RSZ2_RSZ2_RSZ2_RSZ1_RSZ2_RSZ1_RSZ1
Status: Status.UNKNOWN
Output dim: 1, lower bound: -0.0007828, upper bound: 0.0007828
time: 0.57 seconds

### Relational analysis RSZ of RS_RSZ2_RSZ2_RSZ2_RSZ1_RSZ2_RSZ1_RSZ2

#### Relational analysis RSZ result of RS_RSZ2_RSZ2_RSZ2_RSZ1_RSZ2_RSZ1_RSZ2
Status: Status.UNKNOWN
Output dim: 1, lower bound: -0.0007828, upper bound: 0.0007828
time: 0.56 seconds

## BFS RS instance: RS_RSZ2_RSZ2_RSZ2_RSZ1_RSZ2_RSZ2

### Backsubstitution after applying RS history:
0: 0.0006996, 0.0011177, 0.0006996, 0.0011177, -0.0004181, 0.0004181
1: 0.9934013, 0.9943976, 0.9934013, 0.9943976, -0.0009964, 0.0009964
2: -0.0088595, -0.0053283, -0.0088595, -0.0053283, -0.0032415, 0.0032435
3: 0.0035633, 0.0041662, 0.0035633, 0.0041662, -0.0006029, 0.0006029
4: 0.0026282, 0.0054191, 0.0026282, 0.0054191, -0.0027909, 0.0027909
5: 0.0050201, 0.0064446, 0.0050201, 0.0064446, -0.0014245, 0.0014245
6: -0.0021963, -0.0008150, -0.0021963, -0.0008150, -0.0013813, 0.0013813
7: -0.0082640, -0.0074494, -0.0082640, -0.0074494, -0.0008146, 0.0008146
8: 0.0052524, 0.0098919, 0.0052524, 0.0098919, -0.0045627, 0.0045622
9: -0.0036845, -0.0030730, -0.0036845, -0.0030730, -0.0006115, 0.0006115

### Unstable ReLU Count (Linear/Conv2D Layers)
- layer_idx=0, type=LayerType.Linear, total=256, inp1_unstable=10, inp2_unstable=10, delta_unstable=250
- layer_idx=2, type=LayerType.Linear, total=256, inp1_unstable=8, inp2_unstable=8, delta_unstable=256
- layer_idx=4, type=LayerType.Linear, total=256, inp1_unstable=0, inp2_unstable=0, delta_unstable=256
- layer_idx=6, type=LayerType.Linear, total=256, inp1_unstable=2, inp2_unstable=2, delta_unstable=256
- layer_idx=8, type=LayerType.Linear, total=10, inp1_unstable=0, inp2_unstable=0, delta_unstable=10

Time for backsubstitution: 1.79 seconds

### RS candidates at layer 1
type: RSZ, layer: 1, pos: 198
type: RSZ, layer: 1, pos: 199
type: RSZ, layer: 1, pos: 204
type: RSZ, layer: 1, pos: 250

Time for candidate selection: 0.18 seconds

### Candidate
type: RSZ, layer: 1, pos: 198

### Relational analysis RSZ of RS_RSZ2_RSZ2_RSZ2_RSZ1_RSZ2_RSZ2_RSZ1

#### Relational analysis RSZ result of RS_RSZ2_RSZ2_RSZ2_RSZ1_RSZ2_RSZ2_RSZ1
Status: Status.UNKNOWN
Output dim: 1, lower bound: -0.0007828, upper bound: 0.0007828
time: 0.55 seconds

### Relational analysis RSZ of RS_RSZ2_RSZ2_RSZ2_RSZ1_RSZ2_RSZ2_RSZ2

#### Relational analysis RSZ result of RS_RSZ2_RSZ2_RSZ2_RSZ1_RSZ2_RSZ2_RSZ2
Status: Status.UNKNOWN
Output dim: 1, lower bound: -0.0007828, upper bound: 0.0007828
time: 0.56 seconds

## BFS RS instance: RS_RSZ2_RSZ2_RSZ2_RSZ2_RSZ1_RSZ1

### Backsubstitution after applying RS history:
0: 0.0006996, 0.0011177, 0.0006996, 0.0011177, -0.0004181, 0.0004181
1: 0.9934013, 0.9943976, 0.9934013, 0.9943976, -0.0009964, 0.0009964
2: -0.0088595, -0.0053283, -0.0088595, -0.0053283, -0.0032408, 0.0032433
3: 0.0035633, 0.0041662, 0.0035633, 0.0041662, -0.0006029, 0.0006029
4: 0.0026282, 0.0054191, 0.0026282, 0.0054191, -0.0027909, 0.0027909
5: 0.0050201, 0.0064446, 0.0050201, 0.0064446, -0.0014245, 0.0014245
6: -0.0021963, -0.0008150, -0.0021963, -0.0008150, -0.0013813, 0.0013813
7: -0.0082640, -0.0074494, -0.0082640, -0.0074494, -0.0008146, 0.0008146
8: 0.0052524, 0.0098919, 0.0052524, 0.0098919, -0.0045627, 0.0045620
9: -0.0036845, -0.0030730, -0.0036845, -0.0030730, -0.0006115, 0.0006115

### Unstable ReLU Count (Linear/Conv2D Layers)
- layer_idx=0, type=LayerType.Linear, total=256, inp1_unstable=10, inp2_unstable=10, delta_unstable=250
- layer_idx=2, type=LayerType.Linear, total=256, inp1_unstable=8, inp2_unstable=8, delta_unstable=256
- layer_idx=4, type=LayerType.Linear, total=256, inp1_unstable=0, inp2_unstable=0, delta_unstable=256
- layer_idx=6, type=LayerType.Linear, total=256, inp1_unstable=2, inp2_unstable=2, delta_unstable=256
- layer_idx=8, type=LayerType.Linear, total=10, inp1_unstable=0, inp2_unstable=0, delta_unstable=10

Time for backsubstitution: 1.70 seconds

### RS candidates at layer 1
type: RSZ, layer: 1, pos: 198
type: RSZ, layer: 1, pos: 199
type: RSZ, layer: 1, pos: 204
type: RSZ, layer: 1, pos: 250

Time for candidate selection: 0.18 seconds

### Candidate
type: RSZ, layer: 1, pos: 198

### Relational analysis RSZ of RS_RSZ2_RSZ2_RSZ2_RSZ2_RSZ1_RSZ1_RSZ1

#### Relational analysis RSZ result of RS_RSZ2_RSZ2_RSZ2_RSZ2_RSZ1_RSZ1_RSZ1
Status: Status.UNKNOWN
Output dim: 1, lower bound: -0.0007828, upper bound: 0.0007828
time: 0.63 seconds

### Relational analysis RSZ of RS_RSZ2_RSZ2_RSZ2_RSZ2_RSZ1_RSZ1_RSZ2

#### Relational analysis RSZ result of RS_RSZ2_RSZ2_RSZ2_RSZ2_RSZ1_RSZ1_RSZ2
Status: Status.UNKNOWN
Output dim: 1, lower bound: -0.0007828, upper bound: 0.0007828
time: 0.66 seconds

## BFS RS instance: RS_RSZ2_RSZ2_RSZ2_RSZ2_RSZ1_RSZ2

### Backsubstitution after applying RS history:
0: 0.0006996, 0.0011177, 0.0006996, 0.0011177, -0.0004181, 0.0004181
1: 0.9934013, 0.9943976, 0.9934013, 0.9943976, -0.0009964, 0.0009964
2: -0.0088595, -0.0053283, -0.0088595, -0.0053283, -0.0032419, 0.0032424
3: 0.0035633, 0.0041662, 0.0035633, 0.0041662, -0.0006029, 0.0006029
4: 0.0026282, 0.0054191, 0.0026282, 0.0054191, -0.0027909, 0.0027909
5: 0.0050201, 0.0064446, 0.0050201, 0.0064446, -0.0014245, 0.0014245
6: -0.0021963, -0.0008150, -0.0021963, -0.0008150, -0.0013813, 0.0013813
7: -0.0082640, -0.0074494, -0.0082640, -0.0074494, -0.0008146, 0.0008146
8: 0.0052524, 0.0098919, 0.0052524, 0.0098919, -0.0045624, 0.0045623
9: -0.0036845, -0.0030730, -0.0036845, -0.0030730, -0.0006115, 0.0006115

### Unstable ReLU Count (Linear/Conv2D Layers)
- layer_idx=0, type=LayerType.Linear, total=256, inp1_unstable=10, inp2_unstable=10, delta_unstable=250
- layer_idx=2, type=LayerType.Linear, total=256, inp1_unstable=8, inp2_unstable=8, delta_unstable=256
- layer_idx=4, type=LayerType.Linear, total=256, inp1_unstable=0, inp2_unstable=0, delta_unstable=256
- layer_idx=6, type=LayerType.Linear, total=256, inp1_unstable=2, inp2_unstable=2, delta_unstable=256
- layer_idx=8, type=LayerType.Linear, total=10, inp1_unstable=0, inp2_unstable=0, delta_unstable=10

Time for backsubstitution: 1.97 seconds

### RS candidates at layer 1
type: RSZ, layer: 1, pos: 198
type: RSZ, layer: 1, pos: 199
type: RSZ, layer: 1, pos: 204
type: RSZ, layer: 1, pos: 250

Time for candidate selection: 0.20 seconds

### Candidate
type: RSZ, layer: 1, pos: 198

### Relational analysis RSZ of RS_RSZ2_RSZ2_RSZ2_RSZ2_RSZ1_RSZ2_RSZ1

#### Relational analysis RSZ result of RS_RSZ2_RSZ2_RSZ2_RSZ2_RSZ1_RSZ2_RSZ1
Status: Status.UNKNOWN
Output dim: 1, lower bound: -0.0007828, upper bound: 0.0007828
time: 0.59 seconds

### Relational analysis RSZ of RS_RSZ2_RSZ2_RSZ2_RSZ2_RSZ1_RSZ2_RSZ2

#### Relational analysis RSZ result of RS_RSZ2_RSZ2_RSZ2_RSZ2_RSZ1_RSZ2_RSZ2
Status: Status.UNKNOWN
Output dim: 1, lower bound: -0.0007828, upper bound: 0.0007828
time: 0.58 seconds

## BFS RS instance: RS_RSZ2_RSZ2_RSZ2_RSZ2_RSZ2_RSZ1

### Backsubstitution after applying RS history:
0: 0.0006996, 0.0011177, 0.0006996, 0.0011177, -0.0004181, 0.0004181
1: 0.9934013, 0.9943976, 0.9934013, 0.9943976, -0.0009964, 0.0009964
2: -0.0088595, -0.0053283, -0.0088595, -0.0053283, -0.0032390, 0.0032456
3: 0.0035633, 0.0041662, 0.0035633, 0.0041662, -0.0006029, 0.0006029
4: 0.0026282, 0.0054191, 0.0026282, 0.0054191, -0.0027909, 0.0027909
5: 0.0050201, 0.0064446, 0.0050201, 0.0064446, -0.0014245, 0.0014245
6: -0.0021963, -0.0008150, -0.0021963, -0.0008150, -0.0013813, 0.0013813
7: -0.0082640, -0.0074494, -0.0082640, -0.0074494, -0.0008146, 0.0008146
8: 0.0052524, 0.0098919, 0.0052524, 0.0098919, -0.0045633, 0.0045615
9: -0.0036845, -0.0030730, -0.0036845, -0.0030730, -0.0006115, 0.0006115

### Unstable ReLU Count (Linear/Conv2D Layers)
- layer_idx=0, type=LayerType.Linear, total=256, inp1_unstable=10, inp2_unstable=10, delta_unstable=250
- layer_idx=2, type=LayerType.Linear, total=256, inp1_unstable=8, inp2_unstable=8, delta_unstable=256
- layer_idx=4, type=LayerType.Linear, total=256, inp1_unstable=0, inp2_unstable=0, delta_unstable=256
- layer_idx=6, type=LayerType.Linear, total=256, inp1_unstable=2, inp2_unstable=2, delta_unstable=256
- layer_idx=8, type=LayerType.Linear, total=10, inp1_unstable=0, inp2_unstable=0, delta_unstable=10

Time for backsubstitution: 1.85 seconds

### RS candidates at layer 1
type: RSZ, layer: 1, pos: 198
type: RSZ, layer: 1, pos: 199
type: RSZ, layer: 1, pos: 204
type: RSZ, layer: 1, pos: 250

Time for candidate selection: 0.22 seconds

### Candidate
type: RSZ, layer: 1, pos: 198

### Relational analysis RSZ of RS_RSZ2_RSZ2_RSZ2_RSZ2_RSZ2_RSZ1_RSZ1

#### Relational analysis RSZ result of RS_RSZ2_RSZ2_RSZ2_RSZ2_RSZ2_RSZ1_RSZ1
Status: Status.UNKNOWN
Output dim: 1, lower bound: -0.0007828, upper bound: 0.0007828
time: 0.59 seconds

### Relational analysis RSZ of RS_RSZ2_RSZ2_RSZ2_RSZ2_RSZ2_RSZ1_RSZ2

#### Relational analysis RSZ result of RS_RSZ2_RSZ2_RSZ2_RSZ2_RSZ2_RSZ1_RSZ2
Status: Status.UNKNOWN
Output dim: 1, lower bound: -0.0007828, upper bound: 0.0007828
time: 0.58 seconds

## BFS RS instance: RS_RSZ2_RSZ2_RSZ2_RSZ2_RSZ2_RSZ2

### Backsubstitution after applying RS history:
0: 0.0006996, 0.0011177, 0.0006996, 0.0011177, -0.0004181, 0.0004181
1: 0.9934013, 0.9943976, 0.9934013, 0.9943976, -0.0009964, 0.0009964
2: -0.0088595, -0.0053283, -0.0088595, -0.0053283, -0.0032400, 0.0032443
3: 0.0035633, 0.0041662, 0.0035633, 0.0041662, -0.0006029, 0.0006029
4: 0.0026282, 0.0054191, 0.0026282, 0.0054191, -0.0027909, 0.0027909
5: 0.0050201, 0.0064446, 0.0050201, 0.0064446, -0.0014245, 0.0014245
6: -0.0021963, -0.0008150, -0.0021963, -0.0008150, -0.0013813, 0.0013813
7: -0.0082640, -0.0074494, -0.0082640, -0.0074494, -0.0008146, 0.0008146
8: 0.0052524, 0.0098919, 0.0052524, 0.0098919, -0.0045629, 0.0045618
9: -0.0036845, -0.0030730, -0.0036845, -0.0030730, -0.0006115, 0.0006115

### Unstable ReLU Count (Linear/Conv2D Layers)
- layer_idx=0, type=LayerType.Linear, total=256, inp1_unstable=10, inp2_unstable=10, delta_unstable=250
- layer_idx=2, type=LayerType.Linear, total=256, inp1_unstable=8, inp2_unstable=8, delta_unstable=256
- layer_idx=4, type=LayerType.Linear, total=256, inp1_unstable=0, inp2_unstable=0, delta_unstable=256
- layer_idx=6, type=LayerType.Linear, total=256, inp1_unstable=2, inp2_unstable=2, delta_unstable=256
- layer_idx=8, type=LayerType.Linear, total=10, inp1_unstable=0, inp2_unstable=0, delta_unstable=10

Time for backsubstitution: 1.87 seconds

### RS candidates at layer 1
type: RSZ, layer: 1, pos: 198
type: RSZ, layer: 1, pos: 199
type: RSZ, layer: 1, pos: 204
type: RSZ, layer: 1, pos: 250

Time for candidate selection: 0.20 seconds

### Candidate
type: RSZ, layer: 1, pos: 198

### Relational analysis RSZ of RS_RSZ2_RSZ2_RSZ2_RSZ2_RSZ2_RSZ2_RSZ1

#### Relational analysis RSZ result of RS_RSZ2_RSZ2_RSZ2_RSZ2_RSZ2_RSZ2_RSZ1
Status: Status.UNKNOWN
Output dim: 1, lower bound: -0.0007828, upper bound: 0.0007828
time: 0.58 seconds

### Relational analysis RSZ of RS_RSZ2_RSZ2_RSZ2_RSZ2_RSZ2_RSZ2_RSZ2

#### Relational analysis RSZ result of RS_RSZ2_RSZ2_RSZ2_RSZ2_RSZ2_RSZ2_RSZ2
Status: Status.UNKNOWN
Output dim: 1, lower bound: -0.0007828, upper bound: 0.0007828
time: 0.58 seconds

## Summary of splitting (split count: 6)
- Time for RS candidates: 3.23 seconds
RS_RSZ1_RSZ1_RSZ1_RSZ1_RSZ1_RSZ1_RSZ1, status: Status.UNKNOWN, split count: 7, time: 3.23
Output dim: 1, lower bound: -0.0007828, upper bound: 0.0007828
RS_RSZ1_RSZ1_RSZ1_RSZ1_RSZ1_RSZ1_RSZ2, status: Status.UNKNOWN, split count: 7, time: 3.23
Output dim: 1, lower bound: -0.0007828, upper bound: 0.0007828
RS_RSZ1_RSZ1_RSZ1_RSZ1_RSZ1_RSZ2_RSZ1, status: Status.UNKNOWN, split count: 7, time: 3.23
Output dim: 1, lower bound: -0.0007828, upper bound: 0.0007828
RS_RSZ1_RSZ1_RSZ1_RSZ1_RSZ1_RSZ2_RSZ2, status: Status.UNKNOWN, split count: 7, time: 3.23
Output dim: 1, lower bound: -0.0007828, upper bound: 0.0007828
RS_RSZ1_RSZ1_RSZ1_RSZ1_RSZ2_RSZ1_RSZ1, status: Status.UNKNOWN, split count: 7, time: 3.23
Output dim: 1, lower bound: -0.0007828, upper bound: 0.0007828
RS_RSZ1_RSZ1_RSZ1_RSZ1_RSZ2_RSZ1_RSZ2, status: Status.UNKNOWN, split count: 7, time: 3.23
Output dim: 1, lower bound: -0.0007828, upper bound: 0.0007828
RS_RSZ1_RSZ1_RSZ1_RSZ1_RSZ2_RSZ2_RSZ1, status: Status.UNKNOWN, split count: 7, time: 3.23
Output dim: 1, lower bound: -0.0007828, upper bound: 0.0007828
RS_RSZ1_RSZ1_RSZ1_RSZ1_RSZ2_RSZ2_RSZ2, status: Status.UNKNOWN, split count: 7, time: 3.23
Output dim: 1, lower bound: -0.0007828, upper bound: 0.0007828
RS_RSZ1_RSZ1_RSZ1_RSZ2_RSZ1_RSZ1_RSZ1, status: Status.UNKNOWN, split count: 7, time: 3.23
Output dim: 1, lower bound: -0.0007828, upper bound: 0.0007828
RS_RSZ1_RSZ1_RSZ1_RSZ2_RSZ1_RSZ1_RSZ2, status: Status.UNKNOWN, split count: 7, time: 3.23
Output dim: 1, lower bound: -0.0007828, upper bound: 0.0007828
RS_RSZ1_RSZ1_RSZ1_RSZ2_RSZ1_RSZ2_RSZ1, status: Status.UNKNOWN, split count: 7, time: 3.23
Output dim: 1, lower bound: -0.0007828, upper bound: 0.0007828
RS_RSZ1_RSZ1_RSZ1_RSZ2_RSZ1_RSZ2_RSZ2, status: Status.UNKNOWN, split count: 7, time: 3.23
Output dim: 1, lower bound: -0.0007828, upper bound: 0.0007828
RS_RSZ1_RSZ1_RSZ1_RSZ2_RSZ2_RSZ1_RSZ1, status: Status.UNKNOWN, split count: 7, time: 3.23
Output dim: 1, lower bound: -0.0007828, upper bound: 0.0007828
RS_RSZ1_RSZ1_RSZ1_RSZ2_RSZ2_RSZ1_RSZ2, status: Status.UNKNOWN, split count: 7, time: 3.23
Output dim: 1, lower bound: -0.0007828, upper bound: 0.0007828
RS_RSZ1_RSZ1_RSZ1_RSZ2_RSZ2_RSZ2_RSZ1, status: Status.UNKNOWN, split count: 7, time: 3.23
Output dim: 1, lower bound: -0.0007828, upper bound: 0.0007828
RS_RSZ1_RSZ1_RSZ1_RSZ2_RSZ2_RSZ2_RSZ2, status: Status.UNKNOWN, split count: 7, time: 3.23
Output dim: 1, lower bound: -0.0007828, upper bound: 0.0007828
RS_RSZ1_RSZ1_RSZ2_RSZ1_RSZ1_RSZ1_RSZ1, status: Status.UNKNOWN, split count: 7, time: 3.23
Output dim: 1, lower bound: -0.0007828, upper bound: 0.0007828
RS_RSZ1_RSZ1_RSZ2_RSZ1_RSZ1_RSZ1_RSZ2, status: Status.UNKNOWN, split count: 7, time: 3.23
Output dim: 1, lower bound: -0.0007828, upper bound: 0.0007828
RS_RSZ1_RSZ1_RSZ2_RSZ1_RSZ1_RSZ2_RSZ1, status: Status.UNKNOWN, split count: 7, time: 3.23
Output dim: 1, lower bound: -0.0007828, upper bound: 0.0007828
RS_RSZ1_RSZ1_RSZ2_RSZ1_RSZ1_RSZ2_RSZ2, status: Status.UNKNOWN, split count: 7, time: 3.23
Output dim: 1, lower bound: -0.0007828, upper bound: 0.0007828
RS_RSZ1_RSZ1_RSZ2_RSZ1_RSZ2_RSZ1_RSZ1, status: Status.UNKNOWN, split count: 7, time: 3.23
Output dim: 1, lower bound: -0.0007828, upper bound: 0.0007828
RS_RSZ1_RSZ1_RSZ2_RSZ1_RSZ2_RSZ1_RSZ2, status: Status.UNKNOWN, split count: 7, time: 3.23
Output dim: 1, lower bound: -0.0007828, upper bound: 0.0007828
RS_RSZ1_RSZ1_RSZ2_RSZ1_RSZ2_RSZ2_RSZ1, status: Status.UNKNOWN, split count: 7, time: 3.23
Output dim: 1, lower bound: -0.0007828, upper bound: 0.0007828
RS_RSZ1_RSZ1_RSZ2_RSZ1_RSZ2_RSZ2_RSZ2, status: Status.UNKNOWN, split count: 7, time: 3.23
Output dim: 1, lower bound: -0.0007828, upper bound: 0.0007828
RS_RSZ1_RSZ1_RSZ2_RSZ2_RSZ1_RSZ1_RSZ1, status: Status.UNKNOWN, split count: 7, time: 3.23
Output dim: 1, lower bound: -0.0007828, upper bound: 0.0007828
RS_RSZ1_RSZ1_RSZ2_RSZ2_RSZ1_RSZ1_RSZ2, status: Status.UNKNOWN, split count: 7, time: 3.23
Output dim: 1, lower bound: -0.0007828, upper bound: 0.0007828
RS_RSZ1_RSZ1_RSZ2_RSZ2_RSZ1_RSZ2_RSZ1, status: Status.UNKNOWN, split count: 7, time: 3.23
Output dim: 1, lower bound: -0.0007828, upper bound: 0.0007828
RS_RSZ1_RSZ1_RSZ2_RSZ2_RSZ1_RSZ2_RSZ2, status: Status.UNKNOWN, split count: 7, time: 3.23
Output dim: 1, lower bound: -0.0007828, upper bound: 0.0007828
RS_RSZ1_RSZ1_RSZ2_RSZ2_RSZ2_RSZ1_RSZ1, status: Status.UNKNOWN, split count: 7, time: 3.23
Output dim: 1, lower bound: -0.0007828, upper bound: 0.0007828
RS_RSZ1_RSZ1_RSZ2_RSZ2_RSZ2_RSZ1_RSZ2, status: Status.UNKNOWN, split count: 7, time: 3.23
Output dim: 1, lower bound: -0.0007828, upper bound: 0.0007828
RS_RSZ1_RSZ1_RSZ2_RSZ2_RSZ2_RSZ2_RSZ1, status: Status.UNKNOWN, split count: 7, time: 3.23
Output dim: 1, lower bound: -0.0007828, upper bound: 0.0007828
RS_RSZ1_RSZ1_RSZ2_RSZ2_RSZ2_RSZ2_RSZ2, status: Status.UNKNOWN, split count: 7, time: 3.23
Output dim: 1, lower bound: -0.0007828, upper bound: 0.0007828
RS_RSZ1_RSZ2_RSZ1_RSZ1_RSZ1_RSZ1_RSZ1, status: Status.UNKNOWN, split count: 7, time: 3.23
Output dim: 1, lower bound: -0.0007828, upper bound: 0.0007828
RS_RSZ1_RSZ2_RSZ1_RSZ1_RSZ1_RSZ1_RSZ2, status: Status.UNKNOWN, split count: 7, time: 3.23
Output dim: 1, lower bound: -0.0007828, upper bound: 0.0007828
RS_RSZ1_RSZ2_RSZ1_RSZ1_RSZ1_RSZ2_RSZ1, status: Status.UNKNOWN, split count: 7, time: 3.23
Output dim: 1, lower bound: -0.0007828, upper bound: 0.0007828
RS_RSZ1_RSZ2_RSZ1_RSZ1_RSZ1_RSZ2_RSZ2, status: Status.UNKNOWN, split count: 7, time: 3.23
Output dim: 1, lower bound: -0.0007828, upper bound: 0.0007828
RS_RSZ1_RSZ2_RSZ1_RSZ1_RSZ2_RSZ1_RSZ1, status: Status.UNKNOWN, split count: 7, time: 3.23
Output dim: 1, lower bound: -0.0007828, upper bound: 0.0007828
RS_RSZ1_RSZ2_RSZ1_RSZ1_RSZ2_RSZ1_RSZ2, status: Status.UNKNOWN, split count: 7, time: 3.23
Output dim: 1, lower bound: -0.0007828, upper bound: 0.0007828
RS_RSZ1_RSZ2_RSZ1_RSZ1_RSZ2_RSZ2_RSZ1, status: Status.UNKNOWN, split count: 7, time: 3.23
Output dim: 1, lower bound: -0.0007828, upper bound: 0.0007828
RS_RSZ1_RSZ2_RSZ1_RSZ1_RSZ2_RSZ2_RSZ2, status: Status.UNKNOWN, split count: 7, time: 3.23
Output dim: 1, lower bound: -0.0007828, upper bound: 0.0007828
RS_RSZ1_RSZ2_RSZ1_RSZ2_RSZ1_RSZ1_RSZ1, status: Status.UNKNOWN, split count: 7, time: 3.23
Output dim: 1, lower bound: -0.0007828, upper bound: 0.0007828
RS_RSZ1_RSZ2_RSZ1_RSZ2_RSZ1_RSZ1_RSZ2, status: Status.UNKNOWN, split count: 7, time: 3.23
Output dim: 1, lower bound: -0.0007828, upper bound: 0.0007828
RS_RSZ1_RSZ2_RSZ1_RSZ2_RSZ1_RSZ2_RSZ1, status: Status.UNKNOWN, split count: 7, time: 3.23
Output dim: 1, lower bound: -0.0007828, upper bound: 0.0007828
RS_RSZ1_RSZ2_RSZ1_RSZ2_RSZ1_RSZ2_RSZ2, status: Status.UNKNOWN, split count: 7, time: 3.23
Output dim: 1, lower bound: -0.0007828, upper bound: 0.0007828
RS_RSZ1_RSZ2_RSZ1_RSZ2_RSZ2_RSZ1_RSZ1, status: Status.UNKNOWN, split count: 7, time: 3.23
Output dim: 1, lower bound: -0.0007828, upper bound: 0.0007828
RS_RSZ1_RSZ2_RSZ1_RSZ2_RSZ2_RSZ1_RSZ2, status: Status.UNKNOWN, split count: 7, time: 3.23
Output dim: 1, lower bound: -0.0007828, upper bound: 0.0007828
RS_RSZ1_RSZ2_RSZ1_RSZ2_RSZ2_RSZ2_RSZ1, status: Status.UNKNOWN, split count: 7, time: 3.23
Output dim: 1, lower bound: -0.0007828, upper bound: 0.0007828
RS_RSZ1_RSZ2_RSZ1_RSZ2_RSZ2_RSZ2_RSZ2, status: Status.UNKNOWN, split count: 7, time: 3.23
Output dim: 1, lower bound: -0.0007828, upper bound: 0.0007828
RS_RSZ1_RSZ2_RSZ2_RSZ1_RSZ1_RSZ1_RSZ1, status: Status.UNKNOWN, split count: 7, time: 3.23
Output dim: 1, lower bound: -0.0007828, upper bound: 0.0007828
RS_RSZ1_RSZ2_RSZ2_RSZ1_RSZ1_RSZ1_RSZ2, status: Status.UNKNOWN, split count: 7, time: 3.23
Output dim: 1, lower bound: -0.0007828, upper bound: 0.0007828
RS_RSZ1_RSZ2_RSZ2_RSZ1_RSZ1_RSZ2_RSZ1, status: Status.UNKNOWN, split count: 7, time: 3.23
Output dim: 1, lower bound: -0.0007828, upper bound: 0.0007828
RS_RSZ1_RSZ2_RSZ2_RSZ1_RSZ1_RSZ2_RSZ2, status: Status.UNKNOWN, split count: 7, time: 3.23
Output dim: 1, lower bound: -0.0007828, upper bound: 0.0007828
RS_RSZ1_RSZ2_RSZ2_RSZ1_RSZ2_RSZ1_RSZ1, status: Status.UNKNOWN, split count: 7, time: 3.23
Output dim: 1, lower bound: -0.0007828, upper bound: 0.0007828
RS_RSZ1_RSZ2_RSZ2_RSZ1_RSZ2_RSZ1_RSZ2, status: Status.UNKNOWN, split count: 7, time: 3.23
Output dim: 1, lower bound: -0.0007828, upper bound: 0.0007828
RS_RSZ1_RSZ2_RSZ2_RSZ1_RSZ2_RSZ2_RSZ1, status: Status.UNKNOWN, split count: 7, time: 3.23
Output dim: 1, lower bound: -0.0007828, upper bound: 0.0007828
RS_RSZ1_RSZ2_RSZ2_RSZ1_RSZ2_RSZ2_RSZ2, status: Status.UNKNOWN, split count: 7, time: 3.23
Output dim: 1, lower bound: -0.0007828, upper bound: 0.0007828
RS_RSZ1_RSZ2_RSZ2_RSZ2_RSZ1_RSZ1_RSZ1, status: Status.UNKNOWN, split count: 7, time: 3.23
Output dim: 1, lower bound: -0.0007828, upper bound: 0.0007828
RS_RSZ1_RSZ2_RSZ2_RSZ2_RSZ1_RSZ1_RSZ2, status: Status.UNKNOWN, split count: 7, time: 3.23
Output dim: 1, lower bound: -0.0007828, upper bound: 0.0007828
RS_RSZ1_RSZ2_RSZ2_RSZ2_RSZ1_RSZ2_RSZ1, status: Status.UNKNOWN, split count: 7, time: 3.23
Output dim: 1, lower bound: -0.0007828, upper bound: 0.0007828
RS_RSZ1_RSZ2_RSZ2_RSZ2_RSZ1_RSZ2_RSZ2, status: Status.UNKNOWN, split count: 7, time: 3.23
Output dim: 1, lower bound: -0.0007828, upper bound: 0.0007828
RS_RSZ1_RSZ2_RSZ2_RSZ2_RSZ2_RSZ1_RSZ1, status: Status.UNKNOWN, split count: 7, time: 3.23
Output dim: 1, lower bound: -0.0007828, upper bound: 0.0007828
RS_RSZ1_RSZ2_RSZ2_RSZ2_RSZ2_RSZ1_RSZ2, status: Status.UNKNOWN, split count: 7, time: 3.23
Output dim: 1, lower bound: -0.0007828, upper bound: 0.0007828
RS_RSZ1_RSZ2_RSZ2_RSZ2_RSZ2_RSZ2_RSZ1, status: Status.UNKNOWN, split count: 7, time: 3.23
Output dim: 1, lower bound: -0.0007828, upper bound: 0.0007828
RS_RSZ1_RSZ2_RSZ2_RSZ2_RSZ2_RSZ2_RSZ2, status: Status.UNKNOWN, split count: 7, time: 3.23
Output dim: 1, lower bound: -0.0007828, upper bound: 0.0007828
RS_RSZ2_RSZ1_RSZ1_RSZ1_RSZ1_RSZ1_RSZ1, status: Status.UNKNOWN, split count: 7, time: 3.23
Output dim: 1, lower bound: -0.0007828, upper bound: 0.0007828
RS_RSZ2_RSZ1_RSZ1_RSZ1_RSZ1_RSZ1_RSZ2, status: Status.UNKNOWN, split count: 7, time: 3.23
Output dim: 1, lower bound: -0.0007828, upper bound: 0.0007828
RS_RSZ2_RSZ1_RSZ1_RSZ1_RSZ1_RSZ2_RSZ1, status: Status.UNKNOWN, split count: 7, time: 3.23
Output dim: 1, lower bound: -0.0007828, upper bound: 0.0007828
RS_RSZ2_RSZ1_RSZ1_RSZ1_RSZ1_RSZ2_RSZ2, status: Status.UNKNOWN, split count: 7, time: 3.23
Output dim: 1, lower bound: -0.0007828, upper bound: 0.0007828
RS_RSZ2_RSZ1_RSZ1_RSZ1_RSZ2_RSZ1_RSZ1, status: Status.UNKNOWN, split count: 7, time: 3.23
Output dim: 1, lower bound: -0.0007828, upper bound: 0.0007828
RS_RSZ2_RSZ1_RSZ1_RSZ1_RSZ2_RSZ1_RSZ2, status: Status.UNKNOWN, split count: 7, time: 3.23
Output dim: 1, lower bound: -0.0007828, upper bound: 0.0007828
RS_RSZ2_RSZ1_RSZ1_RSZ1_RSZ2_RSZ2_RSZ1, status: Status.UNKNOWN, split count: 7, time: 3.23
Output dim: 1, lower bound: -0.0007828, upper bound: 0.0007828
RS_RSZ2_RSZ1_RSZ1_RSZ1_RSZ2_RSZ2_RSZ2, status: Status.UNKNOWN, split count: 7, time: 3.23
Output dim: 1, lower bound: -0.0007828, upper bound: 0.0007828
RS_RSZ2_RSZ1_RSZ1_RSZ2_RSZ1_RSZ1_RSZ1, status: Status.UNKNOWN, split count: 7, time: 3.23
Output dim: 1, lower bound: -0.0007828, upper bound: 0.0007828
RS_RSZ2_RSZ1_RSZ1_RSZ2_RSZ1_RSZ1_RSZ2, status: Status.UNKNOWN, split count: 7, time: 3.23
Output dim: 1, lower bound: -0.0007828, upper bound: 0.0007828
RS_RSZ2_RSZ1_RSZ1_RSZ2_RSZ1_RSZ2_RSZ1, status: Status.UNKNOWN, split count: 7, time: 3.23
Output dim: 1, lower bound: -0.0007828, upper bound: 0.0007828
RS_RSZ2_RSZ1_RSZ1_RSZ2_RSZ1_RSZ2_RSZ2, status: Status.UNKNOWN, split count: 7, time: 3.23
Output dim: 1, lower bound: -0.0007828, upper bound: 0.0007828
RS_RSZ2_RSZ1_RSZ1_RSZ2_RSZ2_RSZ1_RSZ1, status: Status.UNKNOWN, split count: 7, time: 3.23
Output dim: 1, lower bound: -0.0007828, upper bound: 0.0007828
RS_RSZ2_RSZ1_RSZ1_RSZ2_RSZ2_RSZ1_RSZ2, status: Status.UNKNOWN, split count: 7, time: 3.23
Output dim: 1, lower bound: -0.0007828, upper bound: 0.0007828
RS_RSZ2_RSZ1_RSZ1_RSZ2_RSZ2_RSZ2_RSZ1, status: Status.UNKNOWN, split count: 7, time: 3.23
Output dim: 1, lower bound: -0.0007828, upper bound: 0.0007828
RS_RSZ2_RSZ1_RSZ1_RSZ2_RSZ2_RSZ2_RSZ2, status: Status.UNKNOWN, split count: 7, time: 3.23
Output dim: 1, lower bound: -0.0007828, upper bound: 0.0007828
RS_RSZ2_RSZ1_RSZ2_RSZ1_RSZ1_RSZ1_RSZ1, status: Status.UNKNOWN, split count: 7, time: 3.23
Output dim: 1, lower bound: -0.0007828, upper bound: 0.0007828
RS_RSZ2_RSZ1_RSZ2_RSZ1_RSZ1_RSZ1_RSZ2, status: Status.UNKNOWN, split count: 7, time: 3.23
Output dim: 1, lower bound: -0.0007828, upper bound: 0.0007828
RS_RSZ2_RSZ1_RSZ2_RSZ1_RSZ1_RSZ2_RSZ1, status: Status.UNKNOWN, split count: 7, time: 3.23
Output dim: 1, lower bound: -0.0007828, upper bound: 0.0007828
RS_RSZ2_RSZ1_RSZ2_RSZ1_RSZ1_RSZ2_RSZ2, status: Status.UNKNOWN, split count: 7, time: 3.23
Output dim: 1, lower bound: -0.0007828, upper bound: 0.0007828
RS_RSZ2_RSZ1_RSZ2_RSZ1_RSZ2_RSZ1_RSZ1, status: Status.UNKNOWN, split count: 7, time: 3.23
Output dim: 1, lower bound: -0.0007828, upper bound: 0.0007828
RS_RSZ2_RSZ1_RSZ2_RSZ1_RSZ2_RSZ1_RSZ2, status: Status.UNKNOWN, split count: 7, time: 3.23
Output dim: 1, lower bound: -0.0007828, upper bound: 0.0007828
RS_RSZ2_RSZ1_RSZ2_RSZ1_RSZ2_RSZ2_RSZ1, status: Status.UNKNOWN, split count: 7, time: 3.23
Output dim: 1, lower bound: -0.0007828, upper bound: 0.0007828
RS_RSZ2_RSZ1_RSZ2_RSZ1_RSZ2_RSZ2_RSZ2, status: Status.UNKNOWN, split count: 7, time: 3.23
Output dim: 1, lower bound: -0.0007828, upper bound: 0.0007828
RS_RSZ2_RSZ1_RSZ2_RSZ2_RSZ1_RSZ1_RSZ1, status: Status.UNKNOWN, split count: 7, time: 3.23
Output dim: 1, lower bound: -0.0007828, upper bound: 0.0007828
RS_RSZ2_RSZ1_RSZ2_RSZ2_RSZ1_RSZ1_RSZ2, status: Status.UNKNOWN, split count: 7, time: 3.23
Output dim: 1, lower bound: -0.0007828, upper bound: 0.0007828
RS_RSZ2_RSZ1_RSZ2_RSZ2_RSZ1_RSZ2_RSZ1, status: Status.UNKNOWN, split count: 7, time: 3.23
Output dim: 1, lower bound: -0.0007828, upper bound: 0.0007828
RS_RSZ2_RSZ1_RSZ2_RSZ2_RSZ1_RSZ2_RSZ2, status: Status.UNKNOWN, split count: 7, time: 3.23
Output dim: 1, lower bound: -0.0007828, upper bound: 0.0007828
RS_RSZ2_RSZ1_RSZ2_RSZ2_RSZ2_RSZ1_RSZ1, status: Status.UNKNOWN, split count: 7, time: 3.23
Output dim: 1, lower bound: -0.0007828, upper bound: 0.0007828
RS_RSZ2_RSZ1_RSZ2_RSZ2_RSZ2_RSZ1_RSZ2, status: Status.UNKNOWN, split count: 7, time: 3.23
Output dim: 1, lower bound: -0.0007828, upper bound: 0.0007828
RS_RSZ2_RSZ1_RSZ2_RSZ2_RSZ2_RSZ2_RSZ1, status: Status.UNKNOWN, split count: 7, time: 3.23
Output dim: 1, lower bound: -0.0007828, upper bound: 0.0007828
RS_RSZ2_RSZ1_RSZ2_RSZ2_RSZ2_RSZ2_RSZ2, status: Status.UNKNOWN, split count: 7, time: 3.23
Output dim: 1, lower bound: -0.0007828, upper bound: 0.0007828
RS_RSZ2_RSZ2_RSZ1_RSZ1_RSZ1_RSZ1_RSZ1, status: Status.UNKNOWN, split count: 7, time: 3.23
Output dim: 1, lower bound: -0.0007828, upper bound: 0.0007828
RS_RSZ2_RSZ2_RSZ1_RSZ1_RSZ1_RSZ1_RSZ2, status: Status.UNKNOWN, split count: 7, time: 3.23
Output dim: 1, lower bound: -0.0007828, upper bound: 0.0007828
RS_RSZ2_RSZ2_RSZ1_RSZ1_RSZ1_RSZ2_RSZ1, status: Status.UNKNOWN, split count: 7, time: 3.23
Output dim: 1, lower bound: -0.0007828, upper bound: 0.0007828
RS_RSZ2_RSZ2_RSZ1_RSZ1_RSZ1_RSZ2_RSZ2, status: Status.UNKNOWN, split count: 7, time: 3.23
Output dim: 1, lower bound: -0.0007828, upper bound: 0.0007828
RS_RSZ2_RSZ2_RSZ1_RSZ1_RSZ2_RSZ1_RSZ1, status: Status.UNKNOWN, split count: 7, time: 3.23
Output dim: 1, lower bound: -0.0007828, upper bound: 0.0007828
RS_RSZ2_RSZ2_RSZ1_RSZ1_RSZ2_RSZ1_RSZ2, status: Status.UNKNOWN, split count: 7, time: 3.23
Output dim: 1, lower bound: -0.0007828, upper bound: 0.0007828
RS_RSZ2_RSZ2_RSZ1_RSZ1_RSZ2_RSZ2_RSZ1, status: Status.UNKNOWN, split count: 7, time: 3.23
Output dim: 1, lower bound: -0.0007828, upper bound: 0.0007828
RS_RSZ2_RSZ2_RSZ1_RSZ1_RSZ2_RSZ2_RSZ2, status: Status.UNKNOWN, split count: 7, time: 3.23
Output dim: 1, lower bound: -0.0007828, upper bound: 0.0007828
RS_RSZ2_RSZ2_RSZ1_RSZ2_RSZ1_RSZ1_RSZ1, status: Status.UNKNOWN, split count: 7, time: 3.23
Output dim: 1, lower bound: -0.0007828, upper bound: 0.0007828
RS_RSZ2_RSZ2_RSZ1_RSZ2_RSZ1_RSZ1_RSZ2, status: Status.UNKNOWN, split count: 7, time: 3.23
Output dim: 1, lower bound: -0.0007828, upper bound: 0.0007828
RS_RSZ2_RSZ2_RSZ1_RSZ2_RSZ1_RSZ2_RSZ1, status: Status.UNKNOWN, split count: 7, time: 3.23
Output dim: 1, lower bound: -0.0007828, upper bound: 0.0007828
RS_RSZ2_RSZ2_RSZ1_RSZ2_RSZ1_RSZ2_RSZ2, status: Status.UNKNOWN, split count: 7, time: 3.23
Output dim: 1, lower bound: -0.0007828, upper bound: 0.0007828
RS_RSZ2_RSZ2_RSZ1_RSZ2_RSZ2_RSZ1_RSZ1, status: Status.UNKNOWN, split count: 7, time: 3.23
Output dim: 1, lower bound: -0.0007828, upper bound: 0.0007828
RS_RSZ2_RSZ2_RSZ1_RSZ2_RSZ2_RSZ1_RSZ2, status: Status.UNKNOWN, split count: 7, time: 3.23
Output dim: 1, lower bound: -0.0007828, upper bound: 0.0007828
RS_RSZ2_RSZ2_RSZ1_RSZ2_RSZ2_RSZ2_RSZ1, status: Status.UNKNOWN, split count: 7, time: 3.23
Output dim: 1, lower bound: -0.0007828, upper bound: 0.0007828
RS_RSZ2_RSZ2_RSZ1_RSZ2_RSZ2_RSZ2_RSZ2, status: Status.UNKNOWN, split count: 7, time: 3.23
Output dim: 1, lower bound: -0.0007828, upper bound: 0.0007828
RS_RSZ2_RSZ2_RSZ2_RSZ1_RSZ1_RSZ1_RSZ1, status: Status.UNKNOWN, split count: 7, time: 3.23
Output dim: 1, lower bound: -0.0007828, upper bound: 0.0007828
RS_RSZ2_RSZ2_RSZ2_RSZ1_RSZ1_RSZ1_RSZ2, status: Status.UNKNOWN, split count: 7, time: 3.23
Output dim: 1, lower bound: -0.0007828, upper bound: 0.0007828
RS_RSZ2_RSZ2_RSZ2_RSZ1_RSZ1_RSZ2_RSZ1, status: Status.UNKNOWN, split count: 7, time: 3.23
Output dim: 1, lower bound: -0.0007828, upper bound: 0.0007828
RS_RSZ2_RSZ2_RSZ2_RSZ1_RSZ1_RSZ2_RSZ2, status: Status.UNKNOWN, split count: 7, time: 3.23
Output dim: 1, lower bound: -0.0007828, upper bound: 0.0007828
RS_RSZ2_RSZ2_RSZ2_RSZ1_RSZ2_RSZ1_RSZ1, status: Status.UNKNOWN, split count: 7, time: 3.23
Output dim: 1, lower bound: -0.0007828, upper bound: 0.0007828
RS_RSZ2_RSZ2_RSZ2_RSZ1_RSZ2_RSZ1_RSZ2, status: Status.UNKNOWN, split count: 7, time: 3.23
Output dim: 1, lower bound: -0.0007828, upper bound: 0.0007828
RS_RSZ2_RSZ2_RSZ2_RSZ1_RSZ2_RSZ2_RSZ1, status: Status.UNKNOWN, split count: 7, time: 3.23
Output dim: 1, lower bound: -0.0007828, upper bound: 0.0007828
RS_RSZ2_RSZ2_RSZ2_RSZ1_RSZ2_RSZ2_RSZ2, status: Status.UNKNOWN, split count: 7, time: 3.23
Output dim: 1, lower bound: -0.0007828, upper bound: 0.0007828
RS_RSZ2_RSZ2_RSZ2_RSZ2_RSZ1_RSZ1_RSZ1, status: Status.UNKNOWN, split count: 7, time: 3.23
Output dim: 1, lower bound: -0.0007828, upper bound: 0.0007828
RS_RSZ2_RSZ2_RSZ2_RSZ2_RSZ1_RSZ1_RSZ2, status: Status.UNKNOWN, split count: 7, time: 3.23
Output dim: 1, lower bound: -0.0007828, upper bound: 0.0007828
RS_RSZ2_RSZ2_RSZ2_RSZ2_RSZ1_RSZ2_RSZ1, status: Status.UNKNOWN, split count: 7, time: 3.23
Output dim: 1, lower bound: -0.0007828, upper bound: 0.0007828
RS_RSZ2_RSZ2_RSZ2_RSZ2_RSZ1_RSZ2_RSZ2, status: Status.UNKNOWN, split count: 7, time: 3.23
Output dim: 1, lower bound: -0.0007828, upper bound: 0.0007828
RS_RSZ2_RSZ2_RSZ2_RSZ2_RSZ2_RSZ1_RSZ1, status: Status.UNKNOWN, split count: 7, time: 3.23
Output dim: 1, lower bound: -0.0007828, upper bound: 0.0007828
RS_RSZ2_RSZ2_RSZ2_RSZ2_RSZ2_RSZ1_RSZ2, status: Status.UNKNOWN, split count: 7, time: 3.23
Output dim: 1, lower bound: -0.0007828, upper bound: 0.0007828
RS_RSZ2_RSZ2_RSZ2_RSZ2_RSZ2_RSZ2_RSZ1, status: Status.UNKNOWN, split count: 7, time: 3.23
Output dim: 1, lower bound: -0.0007828, upper bound: 0.0007828
RS_RSZ2_RSZ2_RSZ2_RSZ2_RSZ2_RSZ2_RSZ2, status: Status.UNKNOWN, split count: 7, time: 3.23
Output dim: 1, lower bound: -0.0007828, upper bound: 0.0007828

## BFS RS instance: RS_RSZ1_RSZ1_RSZ1_RSZ1_RSZ1_RSZ1_RSZ1

### Backsubstitution after applying RS history:
0: 0.0006996, 0.0011177, 0.0006996, 0.0011177, -0.0004181, 0.0004181
1: 0.9934013, 0.9943976, 0.9934013, 0.9943976, -0.0009964, 0.0009964
2: -0.0088595, -0.0053283, -0.0088595, -0.0053283, -0.0032409, 0.0032380
3: 0.0035633, 0.0041662, 0.0035633, 0.0041662, -0.0006029, 0.0006029
4: 0.0026282, 0.0054191, 0.0026282, 0.0054191, -0.0027909, 0.0027909
5: 0.0050201, 0.0064446, 0.0050201, 0.0064446, -0.0014245, 0.0014245
6: -0.0021963, -0.0008150, -0.0021963, -0.0008150, -0.0013813, 0.0013813
7: -0.0082640, -0.0074494, -0.0082640, -0.0074494, -0.0008146, 0.0008146
8: 0.0052524, 0.0098919, 0.0052524, 0.0098919, -0.0045613, 0.0045620
9: -0.0036845, -0.0030730, -0.0036845, -0.0030730, -0.0006115, 0.0006115

### Unstable ReLU Count (Linear/Conv2D Layers)
- layer_idx=0, type=LayerType.Linear, total=256, inp1_unstable=10, inp2_unstable=10, delta_unstable=249
- layer_idx=2, type=LayerType.Linear, total=256, inp1_unstable=8, inp2_unstable=8, delta_unstable=256
- layer_idx=4, type=LayerType.Linear, total=256, inp1_unstable=0, inp2_unstable=0, delta_unstable=256
- layer_idx=6, type=LayerType.Linear, total=256, inp1_unstable=2, inp2_unstable=2, delta_unstable=256
- layer_idx=8, type=LayerType.Linear, total=10, inp1_unstable=0, inp2_unstable=0, delta_unstable=10

Time for backsubstitution: 1.74 seconds

### RS candidates at layer 1
type: RSZ, layer: 1, pos: 199
type: RSZ, layer: 1, pos: 204
type: RSZ, layer: 1, pos: 250

Time for candidate selection: 0.20 seconds

### Candidate
type: RSZ, layer: 1, pos: 199

### Relational analysis RSZ of RS_RSZ1_RSZ1_RSZ1_RSZ1_RSZ1_RSZ1_RSZ1_RSZ1

#### Relational analysis RSZ result of RS_RSZ1_RSZ1_RSZ1_RSZ1_RSZ1_RSZ1_RSZ1_RSZ1
Status: Status.UNKNOWN
Output dim: 1, lower bound: -0.0007785, upper bound: 0.0007785
time: 0.62 seconds

### Relational analysis RSZ of RS_RSZ1_RSZ1_RSZ1_RSZ1_RSZ1_RSZ1_RSZ1_RSZ2

#### Relational analysis RSZ result of RS_RSZ1_RSZ1_RSZ1_RSZ1_RSZ1_RSZ1_RSZ1_RSZ2
Status: Status.UNKNOWN
Output dim: 1, lower bound: -0.0007785, upper bound: 0.0007785
time: 0.61 seconds

## BFS RS instance: RS_RSZ1_RSZ1_RSZ1_RSZ1_RSZ1_RSZ1_RSZ2

### Backsubstitution after applying RS history:
0: 0.0006996, 0.0011177, 0.0006996, 0.0011177, -0.0004181, 0.0004181
1: 0.9934013, 0.9943976, 0.9934013, 0.9943976, -0.0009964, 0.0009964
2: -0.0088595, -0.0053283, -0.0088595, -0.0053283, -0.0032443, 0.0032367
3: 0.0035633, 0.0041662, 0.0035633, 0.0041662, -0.0006029, 0.0006029
4: 0.0026282, 0.0054191, 0.0026282, 0.0054191, -0.0027909, 0.0027909
5: 0.0050201, 0.0064446, 0.0050201, 0.0064446, -0.0014245, 0.0014245
6: -0.0021963, -0.0008150, -0.0021963, -0.0008150, -0.0013813, 0.0013813
7: -0.0082640, -0.0074494, -0.0082640, -0.0074494, -0.0008146, 0.0008146
8: 0.0052524, 0.0098919, 0.0052524, 0.0098919, -0.0045609, 0.0045629
9: -0.0036845, -0.0030730, -0.0036845, -0.0030730, -0.0006115, 0.0006115

### Unstable ReLU Count (Linear/Conv2D Layers)
- layer_idx=0, type=LayerType.Linear, total=256, inp1_unstable=10, inp2_unstable=10, delta_unstable=249
- layer_idx=2, type=LayerType.Linear, total=256, inp1_unstable=8, inp2_unstable=8, delta_unstable=256
- layer_idx=4, type=LayerType.Linear, total=256, inp1_unstable=0, inp2_unstable=0, delta_unstable=256
- layer_idx=6, type=LayerType.Linear, total=256, inp1_unstable=2, inp2_unstable=2, delta_unstable=256
- layer_idx=8, type=LayerType.Linear, total=10, inp1_unstable=0, inp2_unstable=0, delta_unstable=10

Time for backsubstitution: 1.69 seconds

### RS candidates at layer 1
type: RSZ, layer: 1, pos: 199
type: RSZ, layer: 1, pos: 204
type: RSZ, layer: 1, pos: 250

Time for candidate selection: 0.19 seconds

### Candidate
type: RSZ, layer: 1, pos: 199

### Relational analysis RSZ of RS_RSZ1_RSZ1_RSZ1_RSZ1_RSZ1_RSZ1_RSZ2_RSZ1

#### Relational analysis RSZ result of RS_RSZ1_RSZ1_RSZ1_RSZ1_RSZ1_RSZ1_RSZ2_RSZ1
Status: Status.UNKNOWN
Output dim: 1, lower bound: -0.0007785, upper bound: 0.0007785
time: 0.62 seconds

### Relational analysis RSZ of RS_RSZ1_RSZ1_RSZ1_RSZ1_RSZ1_RSZ1_RSZ2_RSZ2

#### Relational analysis RSZ result of RS_RSZ1_RSZ1_RSZ1_RSZ1_RSZ1_RSZ1_RSZ2_RSZ2
Status: Status.UNKNOWN
Output dim: 1, lower bound: -0.0007785, upper bound: 0.0007785
time: 0.61 seconds

## BFS RS instance: RS_RSZ1_RSZ1_RSZ1_RSZ1_RSZ1_RSZ2_RSZ1

### Backsubstitution after applying RS history:
0: 0.0006996, 0.0011177, 0.0006996, 0.0011177, -0.0004181, 0.0004181
1: 0.9934013, 0.9943976, 0.9934013, 0.9943976, -0.0009964, 0.0009964
2: -0.0088595, -0.0053283, -0.0088595, -0.0053283, -0.0032422, 0.0032370
3: 0.0035633, 0.0041662, 0.0035633, 0.0041662, -0.0006029, 0.0006029
4: 0.0026282, 0.0054191, 0.0026282, 0.0054191, -0.0027909, 0.0027909
5: 0.0050201, 0.0064446, 0.0050201, 0.0064446, -0.0014245, 0.0014245
6: -0.0021963, -0.0008150, -0.0021963, -0.0008150, -0.0013813, 0.0013813
7: -0.0082640, -0.0074494, -0.0082640, -0.0074494, -0.0008146, 0.0008146
8: 0.0052524, 0.0098919, 0.0052524, 0.0098919, -0.0045610, 0.0045624
9: -0.0036845, -0.0030730, -0.0036845, -0.0030730, -0.0006115, 0.0006115

### Unstable ReLU Count (Linear/Conv2D Layers)
- layer_idx=0, type=LayerType.Linear, total=256, inp1_unstable=10, inp2_unstable=10, delta_unstable=249
- layer_idx=2, type=LayerType.Linear, total=256, inp1_unstable=8, inp2_unstable=8, delta_unstable=256
- layer_idx=4, type=LayerType.Linear, total=256, inp1_unstable=0, inp2_unstable=0, delta_unstable=256
- layer_idx=6, type=LayerType.Linear, total=256, inp1_unstable=2, inp2_unstable=2, delta_unstable=256
- layer_idx=8, type=LayerType.Linear, total=10, inp1_unstable=0, inp2_unstable=0, delta_unstable=10

Time for backsubstitution: 1.67 seconds

### RS candidates at layer 1
type: RSZ, layer: 1, pos: 199
type: RSZ, layer: 1, pos: 204
type: RSZ, layer: 1, pos: 250

Time for candidate selection: 0.18 seconds

### Candidate
type: RSZ, layer: 1, pos: 199

### Relational analysis RSZ of RS_RSZ1_RSZ1_RSZ1_RSZ1_RSZ1_RSZ2_RSZ1_RSZ1

#### Relational analysis RSZ result of RS_RSZ1_RSZ1_RSZ1_RSZ1_RSZ1_RSZ2_RSZ1_RSZ1
Status: Status.UNKNOWN
Output dim: 1, lower bound: -0.0007785, upper bound: 0.0007785
time: 0.61 seconds

### Relational analysis RSZ of RS_RSZ1_RSZ1_RSZ1_RSZ1_RSZ1_RSZ2_RSZ1_RSZ2

#### Relational analysis RSZ result of RS_RSZ1_RSZ1_RSZ1_RSZ1_RSZ1_RSZ2_RSZ1_RSZ2
Status: Status.UNKNOWN
Output dim: 1, lower bound: -0.0007785, upper bound: 0.0007785
time: 0.61 seconds

## BFS RS instance: RS_RSZ1_RSZ1_RSZ1_RSZ1_RSZ1_RSZ2_RSZ2

### Backsubstitution after applying RS history:
0: 0.0006996, 0.0011177, 0.0006996, 0.0011177, -0.0004181, 0.0004181
1: 0.9934013, 0.9943976, 0.9934013, 0.9943976, -0.0009964, 0.0009964
2: -0.0088595, -0.0053283, -0.0088595, -0.0053283, -0.0032456, 0.0032357
3: 0.0035633, 0.0041662, 0.0035633, 0.0041662, -0.0006029, 0.0006029
4: 0.0026282, 0.0054191, 0.0026282, 0.0054191, -0.0027909, 0.0027909
5: 0.0050201, 0.0064446, 0.0050201, 0.0064446, -0.0014245, 0.0014245
6: -0.0021963, -0.0008150, -0.0021963, -0.0008150, -0.0013813, 0.0013813
7: -0.0082640, -0.0074494, -0.0082640, -0.0074494, -0.0008146, 0.0008146
8: 0.0052524, 0.0098919, 0.0052524, 0.0098919, -0.0045606, 0.0045633
9: -0.0036845, -0.0030730, -0.0036845, -0.0030730, -0.0006115, 0.0006115

### Unstable ReLU Count (Linear/Conv2D Layers)
- layer_idx=0, type=LayerType.Linear, total=256, inp1_unstable=10, inp2_unstable=10, delta_unstable=249
- layer_idx=2, type=LayerType.Linear, total=256, inp1_unstable=8, inp2_unstable=8, delta_unstable=256
- layer_idx=4, type=LayerType.Linear, total=256, inp1_unstable=0, inp2_unstable=0, delta_unstable=256
- layer_idx=6, type=LayerType.Linear, total=256, inp1_unstable=2, inp2_unstable=2, delta_unstable=256
- layer_idx=8, type=LayerType.Linear, total=10, inp1_unstable=0, inp2_unstable=0, delta_unstable=10

Time for backsubstitution: 1.73 seconds

### RS candidates at layer 1
type: RSZ, layer: 1, pos: 199
type: RSZ, layer: 1, pos: 204
type: RSZ, layer: 1, pos: 250

Time for candidate selection: 0.19 seconds

### Candidate
type: RSZ, layer: 1, pos: 199

### Relational analysis RSZ of RS_RSZ1_RSZ1_RSZ1_RSZ1_RSZ1_RSZ2_RSZ2_RSZ1

#### Relational analysis RSZ result of RS_RSZ1_RSZ1_RSZ1_RSZ1_RSZ1_RSZ2_RSZ2_RSZ1
Status: Status.UNKNOWN
Output dim: 1, lower bound: -0.0007785, upper bound: 0.0007785
time: 0.62 seconds

### Relational analysis RSZ of RS_RSZ1_RSZ1_RSZ1_RSZ1_RSZ1_RSZ2_RSZ2_RSZ2

#### Relational analysis RSZ result of RS_RSZ1_RSZ1_RSZ1_RSZ1_RSZ1_RSZ2_RSZ2_RSZ2
Status: Status.UNKNOWN
Output dim: 1, lower bound: -0.0007785, upper bound: 0.0007785
time: 0.62 seconds

## BFS RS instance: RS_RSZ1_RSZ1_RSZ1_RSZ1_RSZ2_RSZ1_RSZ1

### Backsubstitution after applying RS history:
0: 0.0006996, 0.0011177, 0.0006996, 0.0011177, -0.0004181, 0.0004181
1: 0.9934013, 0.9943976, 0.9934013, 0.9943976, -0.0009964, 0.0009964
2: -0.0088595, -0.0053283, -0.0088595, -0.0053283, -0.0032391, 0.0032391
3: 0.0035633, 0.0041662, 0.0035633, 0.0041662, -0.0006029, 0.0006029
4: 0.0026282, 0.0054191, 0.0026282, 0.0054191, -0.0027909, 0.0027909
5: 0.0050201, 0.0064446, 0.0050201, 0.0064446, -0.0014245, 0.0014245
6: -0.0021963, -0.0008150, -0.0021963, -0.0008150, -0.0013813, 0.0013813
7: -0.0082640, -0.0074494, -0.0082640, -0.0074494, -0.0008146, 0.0008146
8: 0.0052524, 0.0098919, 0.0052524, 0.0098919, -0.0045615, 0.0045615
9: -0.0036845, -0.0030730, -0.0036845, -0.0030730, -0.0006115, 0.0006115

### Unstable ReLU Count (Linear/Conv2D Layers)
- layer_idx=0, type=LayerType.Linear, total=256, inp1_unstable=10, inp2_unstable=10, delta_unstable=249
- layer_idx=2, type=LayerType.Linear, total=256, inp1_unstable=8, inp2_unstable=8, delta_unstable=256
- layer_idx=4, type=LayerType.Linear, total=256, inp1_unstable=0, inp2_unstable=0, delta_unstable=256
- layer_idx=6, type=LayerType.Linear, total=256, inp1_unstable=2, inp2_unstable=2, delta_unstable=256
- layer_idx=8, type=LayerType.Linear, total=10, inp1_unstable=0, inp2_unstable=0, delta_unstable=10

Time for backsubstitution: 1.68 seconds

### RS candidates at layer 1
type: RSZ, layer: 1, pos: 199
type: RSZ, layer: 1, pos: 204
type: RSZ, layer: 1, pos: 250

Time for candidate selection: 0.19 seconds

### Candidate
type: RSZ, layer: 1, pos: 199

### Relational analysis RSZ of RS_RSZ1_RSZ1_RSZ1_RSZ1_RSZ2_RSZ1_RSZ1_RSZ1

#### Relational analysis RSZ result of RS_RSZ1_RSZ1_RSZ1_RSZ1_RSZ2_RSZ1_RSZ1_RSZ1
Status: Status.UNKNOWN
Output dim: 1, lower bound: -0.0007785, upper bound: 0.0007785
time: 0.60 seconds

### Relational analysis RSZ of RS_RSZ1_RSZ1_RSZ1_RSZ1_RSZ2_RSZ1_RSZ1_RSZ2

#### Relational analysis RSZ result of RS_RSZ1_RSZ1_RSZ1_RSZ1_RSZ2_RSZ1_RSZ1_RSZ2
Status: Status.UNKNOWN
Output dim: 1, lower bound: -0.0007785, upper bound: 0.0007785
time: 0.64 seconds

## BFS RS instance: RS_RSZ1_RSZ1_RSZ1_RSZ1_RSZ2_RSZ1_RSZ2

### Backsubstitution after applying RS history:
0: 0.0006996, 0.0011177, 0.0006996, 0.0011177, -0.0004181, 0.0004181
1: 0.9934013, 0.9943976, 0.9934013, 0.9943976, -0.0009964, 0.0009964
2: -0.0088595, -0.0053283, -0.0088595, -0.0053283, -0.0032424, 0.0032385
3: 0.0035633, 0.0041662, 0.0035633, 0.0041662, -0.0006029, 0.0006029
4: 0.0026282, 0.0054191, 0.0026282, 0.0054191, -0.0027909, 0.0027909
5: 0.0050201, 0.0064446, 0.0050201, 0.0064446, -0.0014245, 0.0014245
6: -0.0021963, -0.0008150, -0.0021963, -0.0008150, -0.0013813, 0.0013813
7: -0.0082640, -0.0074494, -0.0082640, -0.0074494, -0.0008146, 0.0008146
8: 0.0052524, 0.0098919, 0.0052524, 0.0098919, -0.0045614, 0.0045624
9: -0.0036845, -0.0030730, -0.0036845, -0.0030730, -0.0006115, 0.0006115

### Unstable ReLU Count (Linear/Conv2D Layers)
- layer_idx=0, type=LayerType.Linear, total=256, inp1_unstable=10, inp2_unstable=10, delta_unstable=249
- layer_idx=2, type=LayerType.Linear, total=256, inp1_unstable=8, inp2_unstable=8, delta_unstable=256
- layer_idx=4, type=LayerType.Linear, total=256, inp1_unstable=0, inp2_unstable=0, delta_unstable=256
- layer_idx=6, type=LayerType.Linear, total=256, inp1_unstable=2, inp2_unstable=2, delta_unstable=256
- layer_idx=8, type=LayerType.Linear, total=10, inp1_unstable=0, inp2_unstable=0, delta_unstable=10

Time for backsubstitution: 1.76 seconds

### RS candidates at layer 1
type: RSZ, layer: 1, pos: 199
type: RSZ, layer: 1, pos: 204
type: RSZ, layer: 1, pos: 250

Time for candidate selection: 0.18 seconds

### Candidate
type: RSZ, layer: 1, pos: 199

### Relational analysis RSZ of RS_RSZ1_RSZ1_RSZ1_RSZ1_RSZ2_RSZ1_RSZ2_RSZ1

#### Relational analysis RSZ result of RS_RSZ1_RSZ1_RSZ1_RSZ1_RSZ2_RSZ1_RSZ2_RSZ1
Status: Status.UNKNOWN
Output dim: 1, lower bound: -0.0007785, upper bound: 0.0007785
time: 0.60 seconds

### Relational analysis RSZ of RS_RSZ1_RSZ1_RSZ1_RSZ1_RSZ2_RSZ1_RSZ2_RSZ2

#### Relational analysis RSZ result of RS_RSZ1_RSZ1_RSZ1_RSZ1_RSZ2_RSZ1_RSZ2_RSZ2
Status: Status.UNKNOWN
Output dim: 1, lower bound: -0.0007785, upper bound: 0.0007785
time: 0.59 seconds

## BFS RS instance: RS_RSZ1_RSZ1_RSZ1_RSZ1_RSZ2_RSZ2_RSZ1

### Backsubstitution after applying RS history:
0: 0.0006996, 0.0011177, 0.0006996, 0.0011177, -0.0004181, 0.0004181
1: 0.9934013, 0.9943976, 0.9934013, 0.9943976, -0.0009964, 0.0009964
2: -0.0088595, -0.0053283, -0.0088595, -0.0053283, -0.0032400, 0.0032378
3: 0.0035633, 0.0041662, 0.0035633, 0.0041662, -0.0006029, 0.0006029
4: 0.0026282, 0.0054191, 0.0026282, 0.0054191, -0.0027909, 0.0027909
5: 0.0050201, 0.0064446, 0.0050201, 0.0064446, -0.0014245, 0.0014245
6: -0.0021963, -0.0008150, -0.0021963, -0.0008150, -0.0013813, 0.0013813
7: -0.0082640, -0.0074494, -0.0082640, -0.0074494, -0.0008146, 0.0008146
8: 0.0052524, 0.0098919, 0.0052524, 0.0098919, -0.0045612, 0.0045618
9: -0.0036845, -0.0030730, -0.0036845, -0.0030730, -0.0006115, 0.0006115

### Unstable ReLU Count (Linear/Conv2D Layers)
- layer_idx=0, type=LayerType.Linear, total=256, inp1_unstable=10, inp2_unstable=10, delta_unstable=249
- layer_idx=2, type=LayerType.Linear, total=256, inp1_unstable=8, inp2_unstable=8, delta_unstable=256
- layer_idx=4, type=LayerType.Linear, total=256, inp1_unstable=0, inp2_unstable=0, delta_unstable=256
- layer_idx=6, type=LayerType.Linear, total=256, inp1_unstable=2, inp2_unstable=2, delta_unstable=256
- layer_idx=8, type=LayerType.Linear, total=10, inp1_unstable=0, inp2_unstable=0, delta_unstable=10

Time for backsubstitution: 1.74 seconds

### RS candidates at layer 1
type: RSZ, layer: 1, pos: 199
type: RSZ, layer: 1, pos: 204
type: RSZ, layer: 1, pos: 250

Time for candidate selection: 0.19 seconds

### Candidate
type: RSZ, layer: 1, pos: 199

### Relational analysis RSZ of RS_RSZ1_RSZ1_RSZ1_RSZ1_RSZ2_RSZ2_RSZ1_RSZ1

#### Relational analysis RSZ result of RS_RSZ1_RSZ1_RSZ1_RSZ1_RSZ2_RSZ2_RSZ1_RSZ1
Status: Status.UNKNOWN
Output dim: 1, lower bound: -0.0007785, upper bound: 0.0007785
time: 0.64 seconds

### Relational analysis RSZ of RS_RSZ1_RSZ1_RSZ1_RSZ1_RSZ2_RSZ2_RSZ1_RSZ2

#### Relational analysis RSZ result of RS_RSZ1_RSZ1_RSZ1_RSZ1_RSZ2_RSZ2_RSZ1_RSZ2
Status: Status.UNKNOWN
Output dim: 1, lower bound: -0.0007785, upper bound: 0.0007785
time: 0.62 seconds

## BFS RS instance: RS_RSZ1_RSZ1_RSZ1_RSZ1_RSZ2_RSZ2_RSZ2

### Backsubstitution after applying RS history:
0: 0.0006996, 0.0011177, 0.0006996, 0.0011177, -0.0004181, 0.0004181
1: 0.9934013, 0.9943976, 0.9934013, 0.9943976, -0.0009964, 0.0009964
2: -0.0088595, -0.0053283, -0.0088595, -0.0053283, -0.0032433, 0.0032375
3: 0.0035633, 0.0041662, 0.0035633, 0.0041662, -0.0006029, 0.0006029
4: 0.0026282, 0.0054191, 0.0026282, 0.0054191, -0.0027909, 0.0027909
5: 0.0050201, 0.0064446, 0.0050201, 0.0064446, -0.0014245, 0.0014245
6: -0.0021963, -0.0008150, -0.0021963, -0.0008150, -0.0013813, 0.0013813
7: -0.0082640, -0.0074494, -0.0082640, -0.0074494, -0.0008146, 0.0008146
8: 0.0052524, 0.0098919, 0.0052524, 0.0098919, -0.0045611, 0.0045627
9: -0.0036845, -0.0030730, -0.0036845, -0.0030730, -0.0006115, 0.0006115

### Unstable ReLU Count (Linear/Conv2D Layers)
- layer_idx=0, type=LayerType.Linear, total=256, inp1_unstable=10, inp2_unstable=10, delta_unstable=249
- layer_idx=2, type=LayerType.Linear, total=256, inp1_unstable=8, inp2_unstable=8, delta_unstable=256
- layer_idx=4, type=LayerType.Linear, total=256, inp1_unstable=0, inp2_unstable=0, delta_unstable=256
- layer_idx=6, type=LayerType.Linear, total=256, inp1_unstable=2, inp2_unstable=2, delta_unstable=256
- layer_idx=8, type=LayerType.Linear, total=10, inp1_unstable=0, inp2_unstable=0, delta_unstable=10

Time for backsubstitution: 1.70 seconds

### RS candidates at layer 1
type: RSZ, layer: 1, pos: 199
type: RSZ, layer: 1, pos: 204
type: RSZ, layer: 1, pos: 250

Time for candidate selection: 0.18 seconds

### Candidate
type: RSZ, layer: 1, pos: 199

### Relational analysis RSZ of RS_RSZ1_RSZ1_RSZ1_RSZ1_RSZ2_RSZ2_RSZ2_RSZ1

#### Relational analysis RSZ result of RS_RSZ1_RSZ1_RSZ1_RSZ1_RSZ2_RSZ2_RSZ2_RSZ1
Status: Status.UNKNOWN
Output dim: 1, lower bound: -0.0007785, upper bound: 0.0007785
time: 0.62 seconds

### Relational analysis RSZ of RS_RSZ1_RSZ1_RSZ1_RSZ1_RSZ2_RSZ2_RSZ2_RSZ2

#### Relational analysis RSZ result of RS_RSZ1_RSZ1_RSZ1_RSZ1_RSZ2_RSZ2_RSZ2_RSZ2
Status: Status.UNKNOWN
Output dim: 1, lower bound: -0.0007785, upper bound: 0.0007785
time: 0.59 seconds

## BFS RS instance: RS_RSZ1_RSZ1_RSZ1_RSZ2_RSZ1_RSZ1_RSZ1

### Backsubstitution after applying RS history:
0: 0.0006996, 0.0011177, 0.0006996, 0.0011177, -0.0004181, 0.0004181
1: 0.9934013, 0.9943976, 0.9934013, 0.9943976, -0.0009964, 0.0009964
2: -0.0088595, -0.0053283, -0.0088595, -0.0053283, -0.0032402, 0.0032391
3: 0.0035633, 0.0041662, 0.0035633, 0.0041662, -0.0006029, 0.0006029
4: 0.0026282, 0.0054191, 0.0026282, 0.0054191, -0.0027909, 0.0027909
5: 0.0050201, 0.0064446, 0.0050201, 0.0064446, -0.0014245, 0.0014245
6: -0.0021963, -0.0008150, -0.0021963, -0.0008150, -0.0013813, 0.0013813
7: -0.0082640, -0.0074494, -0.0082640, -0.0074494, -0.0008146, 0.0008146
8: 0.0052524, 0.0098919, 0.0052524, 0.0098919, -0.0045616, 0.0045618
9: -0.0036845, -0.0030730, -0.0036845, -0.0030730, -0.0006115, 0.0006115

### Unstable ReLU Count (Linear/Conv2D Layers)
- layer_idx=0, type=LayerType.Linear, total=256, inp1_unstable=10, inp2_unstable=10, delta_unstable=249
- layer_idx=2, type=LayerType.Linear, total=256, inp1_unstable=8, inp2_unstable=8, delta_unstable=256
- layer_idx=4, type=LayerType.Linear, total=256, inp1_unstable=0, inp2_unstable=0, delta_unstable=256
- layer_idx=6, type=LayerType.Linear, total=256, inp1_unstable=2, inp2_unstable=2, delta_unstable=256
- layer_idx=8, type=LayerType.Linear, total=10, inp1_unstable=0, inp2_unstable=0, delta_unstable=10

Time for backsubstitution: 1.73 seconds

### RS candidates at layer 1
type: RSZ, layer: 1, pos: 199
type: RSZ, layer: 1, pos: 204
type: RSZ, layer: 1, pos: 250

Time for candidate selection: 0.20 seconds

### Candidate
type: RSZ, layer: 1, pos: 199

### Relational analysis RSZ of RS_RSZ1_RSZ1_RSZ1_RSZ2_RSZ1_RSZ1_RSZ1_RSZ1

#### Relational analysis RSZ result of RS_RSZ1_RSZ1_RSZ1_RSZ2_RSZ1_RSZ1_RSZ1_RSZ1
Status: Status.UNKNOWN
Output dim: 1, lower bound: -0.0007785, upper bound: 0.0007785
time: 0.62 seconds

### Relational analysis RSZ of RS_RSZ1_RSZ1_RSZ1_RSZ2_RSZ1_RSZ1_RSZ1_RSZ2

#### Relational analysis RSZ result of RS_RSZ1_RSZ1_RSZ1_RSZ2_RSZ1_RSZ1_RSZ1_RSZ2
Status: Status.UNKNOWN
Output dim: 1, lower bound: -0.0007785, upper bound: 0.0007785
time: 0.61 seconds

## BFS RS instance: RS_RSZ1_RSZ1_RSZ1_RSZ2_RSZ1_RSZ1_RSZ2

### Backsubstitution after applying RS history:
0: 0.0006996, 0.0011177, 0.0006996, 0.0011177, -0.0004181, 0.0004181
1: 0.9934013, 0.9943976, 0.9934013, 0.9943976, -0.0009964, 0.0009964
2: -0.0088595, -0.0053283, -0.0088595, -0.0053283, -0.0032435, 0.0032381
3: 0.0035633, 0.0041662, 0.0035633, 0.0041662, -0.0006029, 0.0006029
4: 0.0026282, 0.0054191, 0.0026282, 0.0054191, -0.0027909, 0.0027909
5: 0.0050201, 0.0064446, 0.0050201, 0.0064446, -0.0014245, 0.0014245
6: -0.0021963, -0.0008150, -0.0021963, -0.0008150, -0.0013813, 0.0013813
7: -0.0082640, -0.0074494, -0.0082640, -0.0074494, -0.0008146, 0.0008146
8: 0.0052524, 0.0098919, 0.0052524, 0.0098919, -0.0045613, 0.0045627
9: -0.0036845, -0.0030730, -0.0036845, -0.0030730, -0.0006115, 0.0006115

### Unstable ReLU Count (Linear/Conv2D Layers)
- layer_idx=0, type=LayerType.Linear, total=256, inp1_unstable=10, inp2_unstable=10, delta_unstable=249
- layer_idx=2, type=LayerType.Linear, total=256, inp1_unstable=8, inp2_unstable=8, delta_unstable=256
- layer_idx=4, type=LayerType.Linear, total=256, inp1_unstable=0, inp2_unstable=0, delta_unstable=256
- layer_idx=6, type=LayerType.Linear, total=256, inp1_unstable=2, inp2_unstable=2, delta_unstable=256
- layer_idx=8, type=LayerType.Linear, total=10, inp1_unstable=0, inp2_unstable=0, delta_unstable=10

Time for backsubstitution: 1.69 seconds

### RS candidates at layer 1
type: RSZ, layer: 1, pos: 199
type: RSZ, layer: 1, pos: 204
type: RSZ, layer: 1, pos: 250

Time for candidate selection: 0.18 seconds

### Candidate
type: RSZ, layer: 1, pos: 199

### Relational analysis RSZ of RS_RSZ1_RSZ1_RSZ1_RSZ2_RSZ1_RSZ1_RSZ2_RSZ1

#### Relational analysis RSZ result of RS_RSZ1_RSZ1_RSZ1_RSZ2_RSZ1_RSZ1_RSZ2_RSZ1
Status: Status.UNKNOWN
Output dim: 1, lower bound: -0.0007785, upper bound: 0.0007785
time: 0.61 seconds

### Relational analysis RSZ of RS_RSZ1_RSZ1_RSZ1_RSZ2_RSZ1_RSZ1_RSZ2_RSZ2

#### Relational analysis RSZ result of RS_RSZ1_RSZ1_RSZ1_RSZ2_RSZ1_RSZ1_RSZ2_RSZ2
Status: Status.UNKNOWN
Output dim: 1, lower bound: -0.0007785, upper bound: 0.0007785
time: 0.62 seconds

## BFS RS instance: RS_RSZ1_RSZ1_RSZ1_RSZ2_RSZ1_RSZ2_RSZ1

### Backsubstitution after applying RS history:
0: 0.0006996, 0.0011177, 0.0006996, 0.0011177, -0.0004181, 0.0004181
1: 0.9934013, 0.9943976, 0.9934013, 0.9943976, -0.0009964, 0.0009964
2: -0.0088595, -0.0053283, -0.0088595, -0.0053283, -0.0032413, 0.0032382
3: 0.0035633, 0.0041662, 0.0035633, 0.0041662, -0.0006029, 0.0006029
4: 0.0026282, 0.0054191, 0.0026282, 0.0054191, -0.0027909, 0.0027909
5: 0.0050201, 0.0064446, 0.0050201, 0.0064446, -0.0014245, 0.0014245
6: -0.0021963, -0.0008150, -0.0021963, -0.0008150, -0.0013813, 0.0013813
7: -0.0082640, -0.0074494, -0.0082640, -0.0074494, -0.0008146, 0.0008146
8: 0.0052524, 0.0098919, 0.0052524, 0.0098919, -0.0045613, 0.0045621
9: -0.0036845, -0.0030730, -0.0036845, -0.0030730, -0.0006115, 0.0006115

### Unstable ReLU Count (Linear/Conv2D Layers)
- layer_idx=0, type=LayerType.Linear, total=256, inp1_unstable=10, inp2_unstable=10, delta_unstable=249
- layer_idx=2, type=LayerType.Linear, total=256, inp1_unstable=8, inp2_unstable=8, delta_unstable=256
- layer_idx=4, type=LayerType.Linear, total=256, inp1_unstable=0, inp2_unstable=0, delta_unstable=256
- layer_idx=6, type=LayerType.Linear, total=256, inp1_unstable=2, inp2_unstable=2, delta_unstable=256
- layer_idx=8, type=LayerType.Linear, total=10, inp1_unstable=0, inp2_unstable=0, delta_unstable=10

Time for backsubstitution: 1.76 seconds

### RS candidates at layer 1
type: RSZ, layer: 1, pos: 199
type: RSZ, layer: 1, pos: 204
type: RSZ, layer: 1, pos: 250

Time for candidate selection: 0.19 seconds

### Candidate
type: RSZ, layer: 1, pos: 199

### Relational analysis RSZ of RS_RSZ1_RSZ1_RSZ1_RSZ2_RSZ1_RSZ2_RSZ1_RSZ1

#### Relational analysis RSZ result of RS_RSZ1_RSZ1_RSZ1_RSZ2_RSZ1_RSZ2_RSZ1_RSZ1
Status: Status.UNKNOWN
Output dim: 1, lower bound: -0.0007785, upper bound: 0.0007785
time: 0.61 seconds

### Relational analysis RSZ of RS_RSZ1_RSZ1_RSZ1_RSZ2_RSZ1_RSZ2_RSZ1_RSZ2

#### Relational analysis RSZ result of RS_RSZ1_RSZ1_RSZ1_RSZ2_RSZ1_RSZ2_RSZ1_RSZ2
Status: Status.UNKNOWN
Output dim: 1, lower bound: -0.0007784, upper bound: 0.0007785
time: 0.72 seconds

## BFS RS instance: RS_RSZ1_RSZ1_RSZ1_RSZ2_RSZ1_RSZ2_RSZ2

### Backsubstitution after applying RS history:
0: 0.0006996, 0.0011177, 0.0006996, 0.0011177, -0.0004181, 0.0004181
1: 0.9934013, 0.9943976, 0.9934013, 0.9943976, -0.0009964, 0.0009964
2: -0.0088595, -0.0053283, -0.0088595, -0.0053283, -0.0032446, 0.0032369
3: 0.0035633, 0.0041662, 0.0035633, 0.0041662, -0.0006029, 0.0006029
4: 0.0026282, 0.0054191, 0.0026282, 0.0054191, -0.0027909, 0.0027909
5: 0.0050201, 0.0064446, 0.0050201, 0.0064446, -0.0014245, 0.0014245
6: -0.0021963, -0.0008150, -0.0021963, -0.0008150, -0.0013813, 0.0013813
7: -0.0082640, -0.0074494, -0.0082640, -0.0074494, -0.0008146, 0.0008146
8: 0.0052524, 0.0098919, 0.0052524, 0.0098919, -0.0045610, 0.0045630
9: -0.0036845, -0.0030730, -0.0036845, -0.0030730, -0.0006115, 0.0006115

### Unstable ReLU Count (Linear/Conv2D Layers)
- layer_idx=0, type=LayerType.Linear, total=256, inp1_unstable=10, inp2_unstable=10, delta_unstable=249
- layer_idx=2, type=LayerType.Linear, total=256, inp1_unstable=8, inp2_unstable=8, delta_unstable=256
- layer_idx=4, type=LayerType.Linear, total=256, inp1_unstable=0, inp2_unstable=0, delta_unstable=256
- layer_idx=6, type=LayerType.Linear, total=256, inp1_unstable=2, inp2_unstable=2, delta_unstable=256
- layer_idx=8, type=LayerType.Linear, total=10, inp1_unstable=0, inp2_unstable=0, delta_unstable=10

Time for backsubstitution: 1.69 seconds

### RS candidates at layer 1
type: RSZ, layer: 1, pos: 199
type: RSZ, layer: 1, pos: 204
type: RSZ, layer: 1, pos: 250

Time for candidate selection: 0.18 seconds

### Candidate
type: RSZ, layer: 1, pos: 199

### Relational analysis RSZ of RS_RSZ1_RSZ1_RSZ1_RSZ2_RSZ1_RSZ2_RSZ2_RSZ1

#### Relational analysis RSZ result of RS_RSZ1_RSZ1_RSZ1_RSZ2_RSZ1_RSZ2_RSZ2_RSZ1
Status: Status.UNKNOWN
Output dim: 1, lower bound: -0.0007785, upper bound: 0.0007785
time: 0.61 seconds

### Relational analysis RSZ of RS_RSZ1_RSZ1_RSZ1_RSZ2_RSZ1_RSZ2_RSZ2_RSZ2

#### Relational analysis RSZ result of RS_RSZ1_RSZ1_RSZ1_RSZ2_RSZ1_RSZ2_RSZ2_RSZ2
Status: Status.UNKNOWN
Output dim: 1, lower bound: -0.0007784, upper bound: 0.0007785
time: 0.72 seconds

## BFS RS instance: RS_RSZ1_RSZ1_RSZ1_RSZ2_RSZ2_RSZ1_RSZ1

### Backsubstitution after applying RS history:
0: 0.0006996, 0.0011177, 0.0006996, 0.0011177, -0.0004181, 0.0004181
1: 0.9934013, 0.9943976, 0.9934013, 0.9943976, -0.0009964, 0.0009964
2: -0.0088595, -0.0053283, -0.0088595, -0.0053283, -0.0032381, 0.0032396
3: 0.0035633, 0.0041662, 0.0035633, 0.0041662, -0.0006029, 0.0006029
4: 0.0026282, 0.0054191, 0.0026282, 0.0054191, -0.0027909, 0.0027909
5: 0.0050201, 0.0064446, 0.0050201, 0.0064446, -0.0014245, 0.0014245
6: -0.0021963, -0.0008150, -0.0021963, -0.0008150, -0.0013813, 0.0013813
7: -0.0082640, -0.0074494, -0.0082640, -0.0074494, -0.0008146, 0.0008146
8: 0.0052524, 0.0098919, 0.0052524, 0.0098919, -0.0045617, 0.0045613
9: -0.0036845, -0.0030730, -0.0036845, -0.0030730, -0.0006115, 0.0006115

### Unstable ReLU Count (Linear/Conv2D Layers)
- layer_idx=0, type=LayerType.Linear, total=256, inp1_unstable=10, inp2_unstable=10, delta_unstable=249
- layer_idx=2, type=LayerType.Linear, total=256, inp1_unstable=8, inp2_unstable=8, delta_unstable=256
- layer_idx=4, type=LayerType.Linear, total=256, inp1_unstable=0, inp2_unstable=0, delta_unstable=256
- layer_idx=6, type=LayerType.Linear, total=256, inp1_unstable=2, inp2_unstable=2, delta_unstable=256
- layer_idx=8, type=LayerType.Linear, total=10, inp1_unstable=0, inp2_unstable=0, delta_unstable=10

Time for backsubstitution: 1.68 seconds

### RS candidates at layer 1
type: RSZ, layer: 1, pos: 199
type: RSZ, layer: 1, pos: 204
type: RSZ, layer: 1, pos: 250

Time for candidate selection: 0.18 seconds

### Candidate
type: RSZ, layer: 1, pos: 199

### Relational analysis RSZ of RS_RSZ1_RSZ1_RSZ1_RSZ2_RSZ2_RSZ1_RSZ1_RSZ1

#### Relational analysis RSZ result of RS_RSZ1_RSZ1_RSZ1_RSZ2_RSZ2_RSZ1_RSZ1_RSZ1
Status: Status.UNKNOWN
Output dim: 1, lower bound: -0.0007785, upper bound: 0.0007785
time: 0.60 seconds

### Relational analysis RSZ of RS_RSZ1_RSZ1_RSZ1_RSZ2_RSZ2_RSZ1_RSZ1_RSZ2

#### Relational analysis RSZ result of RS_RSZ1_RSZ1_RSZ1_RSZ2_RSZ2_RSZ1_RSZ1_RSZ2
Status: Status.UNKNOWN
Output dim: 1, lower bound: -0.0007785, upper bound: 0.0007785
time: 0.59 seconds

## BFS RS instance: RS_RSZ1_RSZ1_RSZ1_RSZ2_RSZ2_RSZ1_RSZ2

### Backsubstitution after applying RS history:
0: 0.0006996, 0.0011177, 0.0006996, 0.0011177, -0.0004181, 0.0004181
1: 0.9934013, 0.9943976, 0.9934013, 0.9943976, -0.0009964, 0.0009964
2: -0.0088595, -0.0053283, -0.0088595, -0.0053283, -0.0032415, 0.0032398
3: 0.0035633, 0.0041662, 0.0035633, 0.0041662, -0.0006029, 0.0006029
4: 0.0026282, 0.0054191, 0.0026282, 0.0054191, -0.0027909, 0.0027909
5: 0.0050201, 0.0064446, 0.0050201, 0.0064446, -0.0014245, 0.0014245
6: -0.0021963, -0.0008150, -0.0021963, -0.0008150, -0.0013813, 0.0013813
7: -0.0082640, -0.0074494, -0.0082640, -0.0074494, -0.0008146, 0.0008146
8: 0.0052524, 0.0098919, 0.0052524, 0.0098919, -0.0045617, 0.0045622
9: -0.0036845, -0.0030730, -0.0036845, -0.0030730, -0.0006115, 0.0006115

### Unstable ReLU Count (Linear/Conv2D Layers)
- layer_idx=0, type=LayerType.Linear, total=256, inp1_unstable=10, inp2_unstable=10, delta_unstable=249
- layer_idx=2, type=LayerType.Linear, total=256, inp1_unstable=8, inp2_unstable=8, delta_unstable=256
- layer_idx=4, type=LayerType.Linear, total=256, inp1_unstable=0, inp2_unstable=0, delta_unstable=256
- layer_idx=6, type=LayerType.Linear, total=256, inp1_unstable=2, inp2_unstable=2, delta_unstable=256
- layer_idx=8, type=LayerType.Linear, total=10, inp1_unstable=0, inp2_unstable=0, delta_unstable=10

Time for backsubstitution: 1.67 seconds

### RS candidates at layer 1
type: RSZ, layer: 1, pos: 199
type: RSZ, layer: 1, pos: 204
type: RSZ, layer: 1, pos: 250

Time for candidate selection: 0.18 seconds

### Candidate
type: RSZ, layer: 1, pos: 199

### Relational analysis RSZ of RS_RSZ1_RSZ1_RSZ1_RSZ2_RSZ2_RSZ1_RSZ2_RSZ1

#### Relational analysis RSZ result of RS_RSZ1_RSZ1_RSZ1_RSZ2_RSZ2_RSZ1_RSZ2_RSZ1
Status: Status.UNKNOWN
Output dim: 1, lower bound: -0.0007785, upper bound: 0.0007785
time: 0.60 seconds

### Relational analysis RSZ of RS_RSZ1_RSZ1_RSZ1_RSZ2_RSZ2_RSZ1_RSZ2_RSZ2

#### Relational analysis RSZ result of RS_RSZ1_RSZ1_RSZ1_RSZ2_RSZ2_RSZ1_RSZ2_RSZ2
Status: Status.UNKNOWN
Output dim: 1, lower bound: -0.0007785, upper bound: 0.0007785
time: 0.59 seconds

## BFS RS instance: RS_RSZ1_RSZ1_RSZ1_RSZ2_RSZ2_RSZ2_RSZ1

### Backsubstitution after applying RS history:
0: 0.0006996, 0.0011177, 0.0006996, 0.0011177, -0.0004181, 0.0004181
1: 0.9934013, 0.9943976, 0.9934013, 0.9943976, -0.0009964, 0.0009964
2: -0.0088595, -0.0053283, -0.0088595, -0.0053283, -0.0032391, 0.0032385
3: 0.0035633, 0.0041662, 0.0035633, 0.0041662, -0.0006029, 0.0006029
4: 0.0026282, 0.0054191, 0.0026282, 0.0054191, -0.0027909, 0.0027909
5: 0.0050201, 0.0064446, 0.0050201, 0.0064446, -0.0014245, 0.0014245
6: -0.0021963, -0.0008150, -0.0021963, -0.0008150, -0.0013813, 0.0013813
7: -0.0082640, -0.0074494, -0.0082640, -0.0074494, -0.0008146, 0.0008146
8: 0.0052524, 0.0098919, 0.0052524, 0.0098919, -0.0045614, 0.0045615
9: -0.0036845, -0.0030730, -0.0036845, -0.0030730, -0.0006115, 0.0006115

### Unstable ReLU Count (Linear/Conv2D Layers)
- layer_idx=0, type=LayerType.Linear, total=256, inp1_unstable=10, inp2_unstable=10, delta_unstable=249
- layer_idx=2, type=LayerType.Linear, total=256, inp1_unstable=8, inp2_unstable=8, delta_unstable=256
- layer_idx=4, type=LayerType.Linear, total=256, inp1_unstable=0, inp2_unstable=0, delta_unstable=256
- layer_idx=6, type=LayerType.Linear, total=256, inp1_unstable=2, inp2_unstable=2, delta_unstable=256
- layer_idx=8, type=LayerType.Linear, total=10, inp1_unstable=0, inp2_unstable=0, delta_unstable=10

Time for backsubstitution: 1.72 seconds

### RS candidates at layer 1
type: RSZ, layer: 1, pos: 199
type: RSZ, layer: 1, pos: 204
type: RSZ, layer: 1, pos: 250

Time for candidate selection: 0.18 seconds

### Candidate
type: RSZ, layer: 1, pos: 199

### Relational analysis RSZ of RS_RSZ1_RSZ1_RSZ1_RSZ2_RSZ2_RSZ2_RSZ1_RSZ1

#### Relational analysis RSZ result of RS_RSZ1_RSZ1_RSZ1_RSZ2_RSZ2_RSZ2_RSZ1_RSZ1
Status: Status.UNKNOWN
Output dim: 1, lower bound: -0.0007785, upper bound: 0.0007785
time: 0.61 seconds

### Relational analysis RSZ of RS_RSZ1_RSZ1_RSZ1_RSZ2_RSZ2_RSZ2_RSZ1_RSZ2

#### Relational analysis RSZ result of RS_RSZ1_RSZ1_RSZ1_RSZ2_RSZ2_RSZ2_RSZ1_RSZ2
Status: Status.UNKNOWN
Output dim: 1, lower bound: -0.0007785, upper bound: 0.0007785
time: 0.61 seconds

## BFS RS instance: RS_RSZ1_RSZ1_RSZ1_RSZ2_RSZ2_RSZ2_RSZ2

### Backsubstitution after applying RS history:
0: 0.0006996, 0.0011177, 0.0006996, 0.0011177, -0.0004181, 0.0004181
1: 0.9934013, 0.9943976, 0.9934013, 0.9943976, -0.0009964, 0.0009964
2: -0.0088595, -0.0053283, -0.0088595, -0.0053283, -0.0032424, 0.0032384
3: 0.0035633, 0.0041662, 0.0035633, 0.0041662, -0.0006029, 0.0006029
4: 0.0026282, 0.0054191, 0.0026282, 0.0054191, -0.0027909, 0.0027909
5: 0.0050201, 0.0064446, 0.0050201, 0.0064446, -0.0014245, 0.0014245
6: -0.0021963, -0.0008150, -0.0021963, -0.0008150, -0.0013813, 0.0013813
7: -0.0082640, -0.0074494, -0.0082640, -0.0074494, -0.0008146, 0.0008146
8: 0.0052524, 0.0098919, 0.0052524, 0.0098919, -0.0045613, 0.0045624
9: -0.0036845, -0.0030730, -0.0036845, -0.0030730, -0.0006115, 0.0006115

### Unstable ReLU Count (Linear/Conv2D Layers)
- layer_idx=0, type=LayerType.Linear, total=256, inp1_unstable=10, inp2_unstable=10, delta_unstable=249
- layer_idx=2, type=LayerType.Linear, total=256, inp1_unstable=8, inp2_unstable=8, delta_unstable=256
- layer_idx=4, type=LayerType.Linear, total=256, inp1_unstable=0, inp2_unstable=0, delta_unstable=256
- layer_idx=6, type=LayerType.Linear, total=256, inp1_unstable=2, inp2_unstable=2, delta_unstable=256
- layer_idx=8, type=LayerType.Linear, total=10, inp1_unstable=0, inp2_unstable=0, delta_unstable=10

Time for backsubstitution: 1.75 seconds

### RS candidates at layer 1
type: RSZ, layer: 1, pos: 199
type: RSZ, layer: 1, pos: 204
type: RSZ, layer: 1, pos: 250

Time for candidate selection: 0.18 seconds

### Candidate
type: RSZ, layer: 1, pos: 199

### Relational analysis RSZ of RS_RSZ1_RSZ1_RSZ1_RSZ2_RSZ2_RSZ2_RSZ2_RSZ1

#### Relational analysis RSZ result of RS_RSZ1_RSZ1_RSZ1_RSZ2_RSZ2_RSZ2_RSZ2_RSZ1
Status: Status.UNKNOWN
Output dim: 1, lower bound: -0.0007785, upper bound: 0.0007785
time: 0.59 seconds

### Relational analysis RSZ of RS_RSZ1_RSZ1_RSZ1_RSZ2_RSZ2_RSZ2_RSZ2_RSZ2

#### Relational analysis RSZ result of RS_RSZ1_RSZ1_RSZ1_RSZ2_RSZ2_RSZ2_RSZ2_RSZ2
Status: Status.UNKNOWN
Output dim: 1, lower bound: -0.0007785, upper bound: 0.0007785
time: 0.66 seconds

## BFS RS instance: RS_RSZ1_RSZ1_RSZ2_RSZ1_RSZ1_RSZ1_RSZ1

### Backsubstitution after applying RS history:
0: 0.0006996, 0.0011177, 0.0006996, 0.0011177, -0.0004181, 0.0004181
1: 0.9934013, 0.9943976, 0.9934013, 0.9943976, -0.0009964, 0.0009964
2: -0.0088595, -0.0053283, -0.0088595, -0.0053283, -0.0032406, 0.0032375
3: 0.0035633, 0.0041662, 0.0035633, 0.0041662, -0.0006029, 0.0006029
4: 0.0026282, 0.0054191, 0.0026282, 0.0054191, -0.0027909, 0.0027909
5: 0.0050201, 0.0064446, 0.0050201, 0.0064446, -0.0014245, 0.0014245
6: -0.0021963, -0.0008150, -0.0021963, -0.0008150, -0.0013813, 0.0013813
7: -0.0082640, -0.0074494, -0.0082640, -0.0074494, -0.0008146, 0.0008146
8: 0.0052524, 0.0098919, 0.0052524, 0.0098919, -0.0045611, 0.0045619
9: -0.0036845, -0.0030730, -0.0036845, -0.0030730, -0.0006115, 0.0006115

### Unstable ReLU Count (Linear/Conv2D Layers)
- layer_idx=0, type=LayerType.Linear, total=256, inp1_unstable=10, inp2_unstable=10, delta_unstable=249
- layer_idx=2, type=LayerType.Linear, total=256, inp1_unstable=8, inp2_unstable=8, delta_unstable=256
- layer_idx=4, type=LayerType.Linear, total=256, inp1_unstable=0, inp2_unstable=0, delta_unstable=256
- layer_idx=6, type=LayerType.Linear, total=256, inp1_unstable=2, inp2_unstable=2, delta_unstable=256
- layer_idx=8, type=LayerType.Linear, total=10, inp1_unstable=0, inp2_unstable=0, delta_unstable=10

Time for backsubstitution: 1.88 seconds

### RS candidates at layer 1
type: RSZ, layer: 1, pos: 199
type: RSZ, layer: 1, pos: 204
type: RSZ, layer: 1, pos: 250

Time for candidate selection: 0.19 seconds

### Candidate
type: RSZ, layer: 1, pos: 199

### Relational analysis RSZ of RS_RSZ1_RSZ1_RSZ2_RSZ1_RSZ1_RSZ1_RSZ1_RSZ1

#### Relational analysis RSZ result of RS_RSZ1_RSZ1_RSZ2_RSZ1_RSZ1_RSZ1_RSZ1_RSZ1
Status: Status.UNKNOWN
Output dim: 1, lower bound: -0.0007785, upper bound: 0.0007785
time: 0.64 seconds

### Relational analysis RSZ of RS_RSZ1_RSZ1_RSZ2_RSZ1_RSZ1_RSZ1_RSZ1_RSZ2

#### Relational analysis RSZ result of RS_RSZ1_RSZ1_RSZ2_RSZ1_RSZ1_RSZ1_RSZ1_RSZ2
Status: Status.UNKNOWN
Output dim: 1, lower bound: -0.0007785, upper bound: 0.0007785
time: 0.65 seconds

## BFS RS instance: RS_RSZ1_RSZ1_RSZ2_RSZ1_RSZ1_RSZ1_RSZ2

### Backsubstitution after applying RS history:
0: 0.0006996, 0.0011177, 0.0006996, 0.0011177, -0.0004181, 0.0004181
1: 0.9934013, 0.9943976, 0.9934013, 0.9943976, -0.0009964, 0.0009964
2: -0.0088595, -0.0053283, -0.0088595, -0.0053283, -0.0032439, 0.0032366
3: 0.0035633, 0.0041662, 0.0035633, 0.0041662, -0.0006029, 0.0006029
4: 0.0026282, 0.0054191, 0.0026282, 0.0054191, -0.0027909, 0.0027909
5: 0.0050201, 0.0064446, 0.0050201, 0.0064446, -0.0014245, 0.0014245
6: -0.0021963, -0.0008150, -0.0021963, -0.0008150, -0.0013813, 0.0013813
7: -0.0082640, -0.0074494, -0.0082640, -0.0074494, -0.0008146, 0.0008146
8: 0.0052524, 0.0098919, 0.0052524, 0.0098919, -0.0045609, 0.0045628
9: -0.0036845, -0.0030730, -0.0036845, -0.0030730, -0.0006115, 0.0006115

### Unstable ReLU Count (Linear/Conv2D Layers)
- layer_idx=0, type=LayerType.Linear, total=256, inp1_unstable=10, inp2_unstable=10, delta_unstable=249
- layer_idx=2, type=LayerType.Linear, total=256, inp1_unstable=8, inp2_unstable=8, delta_unstable=256
- layer_idx=4, type=LayerType.Linear, total=256, inp1_unstable=0, inp2_unstable=0, delta_unstable=256
- layer_idx=6, type=LayerType.Linear, total=256, inp1_unstable=2, inp2_unstable=2, delta_unstable=256
- layer_idx=8, type=LayerType.Linear, total=10, inp1_unstable=0, inp2_unstable=0, delta_unstable=10

Time for backsubstitution: 1.74 seconds

### RS candidates at layer 1
type: RSZ, layer: 1, pos: 199
type: RSZ, layer: 1, pos: 204
type: RSZ, layer: 1, pos: 250

Time for candidate selection: 0.22 seconds

### Candidate
type: RSZ, layer: 1, pos: 199

### Relational analysis RSZ of RS_RSZ1_RSZ1_RSZ2_RSZ1_RSZ1_RSZ1_RSZ2_RSZ1

#### Relational analysis RSZ result of RS_RSZ1_RSZ1_RSZ2_RSZ1_RSZ1_RSZ1_RSZ2_RSZ1
Status: Status.UNKNOWN
Output dim: 1, lower bound: -0.0007785, upper bound: 0.0007785
time: 0.65 seconds

### Relational analysis RSZ of RS_RSZ1_RSZ1_RSZ2_RSZ1_RSZ1_RSZ1_RSZ2_RSZ2

#### Relational analysis RSZ result of RS_RSZ1_RSZ1_RSZ2_RSZ1_RSZ1_RSZ1_RSZ2_RSZ2
Status: Status.UNKNOWN
Output dim: 1, lower bound: -0.0007785, upper bound: 0.0007785
time: 0.65 seconds

## BFS RS instance: RS_RSZ1_RSZ1_RSZ2_RSZ1_RSZ1_RSZ2_RSZ1

### Backsubstitution after applying RS history:
0: 0.0006996, 0.0011177, 0.0006996, 0.0011177, -0.0004181, 0.0004181
1: 0.9934013, 0.9943976, 0.9934013, 0.9943976, -0.0009964, 0.0009964
2: -0.0088595, -0.0053283, -0.0088595, -0.0053283, -0.0032418, 0.0032365
3: 0.0035633, 0.0041662, 0.0035633, 0.0041662, -0.0006029, 0.0006029
4: 0.0026282, 0.0054191, 0.0026282, 0.0054191, -0.0027909, 0.0027909
5: 0.0050201, 0.0064446, 0.0050201, 0.0064446, -0.0014245, 0.0014245
6: -0.0021963, -0.0008150, -0.0021963, -0.0008150, -0.0013813, 0.0013813
7: -0.0082640, -0.0074494, -0.0082640, -0.0074494, -0.0008146, 0.0008146
8: 0.0052524, 0.0098919, 0.0052524, 0.0098919, -0.0045608, 0.0045623
9: -0.0036845, -0.0030730, -0.0036845, -0.0030730, -0.0006115, 0.0006115

### Unstable ReLU Count (Linear/Conv2D Layers)
- layer_idx=0, type=LayerType.Linear, total=256, inp1_unstable=10, inp2_unstable=10, delta_unstable=249
- layer_idx=2, type=LayerType.Linear, total=256, inp1_unstable=8, inp2_unstable=8, delta_unstable=256
- layer_idx=4, type=LayerType.Linear, total=256, inp1_unstable=0, inp2_unstable=0, delta_unstable=256
- layer_idx=6, type=LayerType.Linear, total=256, inp1_unstable=2, inp2_unstable=2, delta_unstable=256
- layer_idx=8, type=LayerType.Linear, total=10, inp1_unstable=0, inp2_unstable=0, delta_unstable=10

Time for backsubstitution: 1.77 seconds

### RS candidates at layer 1
type: RSZ, layer: 1, pos: 199
type: RSZ, layer: 1, pos: 204
type: RSZ, layer: 1, pos: 250

Time for candidate selection: 0.19 seconds

### Candidate
type: RSZ, layer: 1, pos: 199

### Relational analysis RSZ of RS_RSZ1_RSZ1_RSZ2_RSZ1_RSZ1_RSZ2_RSZ1_RSZ1

#### Relational analysis RSZ result of RS_RSZ1_RSZ1_RSZ2_RSZ1_RSZ1_RSZ2_RSZ1_RSZ1
Status: Status.UNKNOWN
Output dim: 1, lower bound: -0.0007785, upper bound: 0.0007785
time: 0.63 seconds

### Relational analysis RSZ of RS_RSZ1_RSZ1_RSZ2_RSZ1_RSZ1_RSZ2_RSZ1_RSZ2

#### Relational analysis RSZ result of RS_RSZ1_RSZ1_RSZ2_RSZ1_RSZ1_RSZ2_RSZ1_RSZ2
Status: Status.UNKNOWN
Output dim: 1, lower bound: -0.0007785, upper bound: 0.0007785
time: 0.62 seconds

## BFS RS instance: RS_RSZ1_RSZ1_RSZ2_RSZ1_RSZ1_RSZ2_RSZ2

### Backsubstitution after applying RS history:
0: 0.0006996, 0.0011177, 0.0006996, 0.0011177, -0.0004181, 0.0004181
1: 0.9934013, 0.9943976, 0.9934013, 0.9943976, -0.0009964, 0.0009964
2: -0.0088595, -0.0053283, -0.0088595, -0.0053283, -0.0032451, 0.0032356
3: 0.0035633, 0.0041662, 0.0035633, 0.0041662, -0.0006029, 0.0006029
4: 0.0026282, 0.0054191, 0.0026282, 0.0054191, -0.0027909, 0.0027909
5: 0.0050201, 0.0064446, 0.0050201, 0.0064446, -0.0014245, 0.0014245
6: -0.0021963, -0.0008150, -0.0021963, -0.0008150, -0.0013813, 0.0013813
7: -0.0082640, -0.0074494, -0.0082640, -0.0074494, -0.0008146, 0.0008146
8: 0.0052524, 0.0098919, 0.0052524, 0.0098919, -0.0045606, 0.0045632
9: -0.0036845, -0.0030730, -0.0036845, -0.0030730, -0.0006115, 0.0006115

### Unstable ReLU Count (Linear/Conv2D Layers)
- layer_idx=0, type=LayerType.Linear, total=256, inp1_unstable=10, inp2_unstable=10, delta_unstable=249
- layer_idx=2, type=LayerType.Linear, total=256, inp1_unstable=8, inp2_unstable=8, delta_unstable=256
- layer_idx=4, type=LayerType.Linear, total=256, inp1_unstable=0, inp2_unstable=0, delta_unstable=256
- layer_idx=6, type=LayerType.Linear, total=256, inp1_unstable=2, inp2_unstable=2, delta_unstable=256
- layer_idx=8, type=LayerType.Linear, total=10, inp1_unstable=0, inp2_unstable=0, delta_unstable=10

Time for backsubstitution: 1.73 seconds

### RS candidates at layer 1
type: RSZ, layer: 1, pos: 199
type: RSZ, layer: 1, pos: 204
type: RSZ, layer: 1, pos: 250

Time for candidate selection: 0.19 seconds

### Candidate
type: RSZ, layer: 1, pos: 199

### Relational analysis RSZ of RS_RSZ1_RSZ1_RSZ2_RSZ1_RSZ1_RSZ2_RSZ2_RSZ1

#### Relational analysis RSZ result of RS_RSZ1_RSZ1_RSZ2_RSZ1_RSZ1_RSZ2_RSZ2_RSZ1
Status: Status.UNKNOWN
Output dim: 1, lower bound: -0.0007785, upper bound: 0.0007785
time: 0.62 seconds

### Relational analysis RSZ of RS_RSZ1_RSZ1_RSZ2_RSZ1_RSZ1_RSZ2_RSZ2_RSZ2

#### Relational analysis RSZ result of RS_RSZ1_RSZ1_RSZ2_RSZ1_RSZ1_RSZ2_RSZ2_RSZ2
Status: Status.UNKNOWN
Output dim: 1, lower bound: -0.0007785, upper bound: 0.0007785
time: 0.61 seconds

## BFS RS instance: RS_RSZ1_RSZ1_RSZ2_RSZ1_RSZ2_RSZ1_RSZ1

### Backsubstitution after applying RS history:
0: 0.0006996, 0.0011177, 0.0006996, 0.0011177, -0.0004181, 0.0004181
1: 0.9934013, 0.9943976, 0.9934013, 0.9943976, -0.0009964, 0.0009964
2: -0.0088595, -0.0053283, -0.0088595, -0.0053283, -0.0032389, 0.0032387
3: 0.0035633, 0.0041662, 0.0035633, 0.0041662, -0.0006029, 0.0006029
4: 0.0026282, 0.0054191, 0.0026282, 0.0054191, -0.0027909, 0.0027909
5: 0.0050201, 0.0064446, 0.0050201, 0.0064446, -0.0014245, 0.0014245
6: -0.0021963, -0.0008150, -0.0021963, -0.0008150, -0.0013813, 0.0013813
7: -0.0082640, -0.0074494, -0.0082640, -0.0074494, -0.0008146, 0.0008146
8: 0.0052524, 0.0098919, 0.0052524, 0.0098919, -0.0045614, 0.0045615
9: -0.0036845, -0.0030730, -0.0036845, -0.0030730, -0.0006115, 0.0006115

### Unstable ReLU Count (Linear/Conv2D Layers)
- layer_idx=0, type=LayerType.Linear, total=256, inp1_unstable=10, inp2_unstable=10, delta_unstable=249
- layer_idx=2, type=LayerType.Linear, total=256, inp1_unstable=8, inp2_unstable=8, delta_unstable=256
- layer_idx=4, type=LayerType.Linear, total=256, inp1_unstable=0, inp2_unstable=0, delta_unstable=256
- layer_idx=6, type=LayerType.Linear, total=256, inp1_unstable=2, inp2_unstable=2, delta_unstable=256
- layer_idx=8, type=LayerType.Linear, total=10, inp1_unstable=0, inp2_unstable=0, delta_unstable=10

Time for backsubstitution: 1.75 seconds

### RS candidates at layer 1
type: RSZ, layer: 1, pos: 199
type: RSZ, layer: 1, pos: 204
type: RSZ, layer: 1, pos: 250

Time for candidate selection: 0.21 seconds

### Candidate
type: RSZ, layer: 1, pos: 199

### Relational analysis RSZ of RS_RSZ1_RSZ1_RSZ2_RSZ1_RSZ2_RSZ1_RSZ1_RSZ1

#### Relational analysis RSZ result of RS_RSZ1_RSZ1_RSZ2_RSZ1_RSZ2_RSZ1_RSZ1_RSZ1
Status: Status.UNKNOWN
Output dim: 1, lower bound: -0.0007785, upper bound: 0.0007785
time: 0.61 seconds

### Relational analysis RSZ of RS_RSZ1_RSZ1_RSZ2_RSZ1_RSZ2_RSZ1_RSZ1_RSZ2

#### Relational analysis RSZ result of RS_RSZ1_RSZ1_RSZ2_RSZ1_RSZ2_RSZ1_RSZ1_RSZ2
Status: Status.UNKNOWN
Output dim: 1, lower bound: -0.0007785, upper bound: 0.0007785
time: 0.60 seconds

## BFS RS instance: RS_RSZ1_RSZ1_RSZ2_RSZ1_RSZ2_RSZ1_RSZ2

### Backsubstitution after applying RS history:
0: 0.0006996, 0.0011177, 0.0006996, 0.0011177, -0.0004181, 0.0004181
1: 0.9934013, 0.9943976, 0.9934013, 0.9943976, -0.0009964, 0.0009964
2: -0.0088595, -0.0053283, -0.0088595, -0.0053283, -0.0032423, 0.0032386
3: 0.0035633, 0.0041662, 0.0035633, 0.0041662, -0.0006029, 0.0006029
4: 0.0026282, 0.0054191, 0.0026282, 0.0054191, -0.0027909, 0.0027909
5: 0.0050201, 0.0064446, 0.0050201, 0.0064446, -0.0014245, 0.0014245
6: -0.0021963, -0.0008150, -0.0021963, -0.0008150, -0.0013813, 0.0013813
7: -0.0082640, -0.0074494, -0.0082640, -0.0074494, -0.0008146, 0.0008146
8: 0.0052524, 0.0098919, 0.0052524, 0.0098919, -0.0045614, 0.0045624
9: -0.0036845, -0.0030730, -0.0036845, -0.0030730, -0.0006115, 0.0006115

### Unstable ReLU Count (Linear/Conv2D Layers)
- layer_idx=0, type=LayerType.Linear, total=256, inp1_unstable=10, inp2_unstable=10, delta_unstable=249
- layer_idx=2, type=LayerType.Linear, total=256, inp1_unstable=8, inp2_unstable=8, delta_unstable=256
- layer_idx=4, type=LayerType.Linear, total=256, inp1_unstable=0, inp2_unstable=0, delta_unstable=256
- layer_idx=6, type=LayerType.Linear, total=256, inp1_unstable=2, inp2_unstable=2, delta_unstable=256
- layer_idx=8, type=LayerType.Linear, total=10, inp1_unstable=0, inp2_unstable=0, delta_unstable=10

Time for backsubstitution: 1.75 seconds

### RS candidates at layer 1
type: RSZ, layer: 1, pos: 199
type: RSZ, layer: 1, pos: 204
type: RSZ, layer: 1, pos: 250

Time for candidate selection: 0.19 seconds

### Candidate
type: RSZ, layer: 1, pos: 199

### Relational analysis RSZ of RS_RSZ1_RSZ1_RSZ2_RSZ1_RSZ2_RSZ1_RSZ2_RSZ1

#### Relational analysis RSZ result of RS_RSZ1_RSZ1_RSZ2_RSZ1_RSZ2_RSZ1_RSZ2_RSZ1
Status: Status.UNKNOWN
Output dim: 1, lower bound: -0.0007785, upper bound: 0.0007785
time: 0.58 seconds

### Relational analysis RSZ of RS_RSZ1_RSZ1_RSZ2_RSZ1_RSZ2_RSZ1_RSZ2_RSZ2

#### Relational analysis RSZ result of RS_RSZ1_RSZ1_RSZ2_RSZ1_RSZ2_RSZ1_RSZ2_RSZ2
Status: Status.UNKNOWN
Output dim: 1, lower bound: -0.0007785, upper bound: 0.0007785
time: 0.59 seconds

## BFS RS instance: RS_RSZ1_RSZ1_RSZ2_RSZ1_RSZ2_RSZ2_RSZ1

### Backsubstitution after applying RS history:
0: 0.0006996, 0.0011177, 0.0006996, 0.0011177, -0.0004181, 0.0004181
1: 0.9934013, 0.9943976, 0.9934013, 0.9943976, -0.0009964, 0.0009964
2: -0.0088595, -0.0053283, -0.0088595, -0.0053283, -0.0032399, 0.0032372
3: 0.0035633, 0.0041662, 0.0035633, 0.0041662, -0.0006029, 0.0006029
4: 0.0026282, 0.0054191, 0.0026282, 0.0054191, -0.0027909, 0.0027909
5: 0.0050201, 0.0064446, 0.0050201, 0.0064446, -0.0014245, 0.0014245
6: -0.0021963, -0.0008150, -0.0021963, -0.0008150, -0.0013813, 0.0013813
7: -0.0082640, -0.0074494, -0.0082640, -0.0074494, -0.0008146, 0.0008146
8: 0.0052524, 0.0098919, 0.0052524, 0.0098919, -0.0045610, 0.0045617
9: -0.0036845, -0.0030730, -0.0036845, -0.0030730, -0.0006115, 0.0006115

### Unstable ReLU Count (Linear/Conv2D Layers)
- layer_idx=0, type=LayerType.Linear, total=256, inp1_unstable=10, inp2_unstable=10, delta_unstable=249
- layer_idx=2, type=LayerType.Linear, total=256, inp1_unstable=8, inp2_unstable=8, delta_unstable=256
- layer_idx=4, type=LayerType.Linear, total=256, inp1_unstable=0, inp2_unstable=0, delta_unstable=256
- layer_idx=6, type=LayerType.Linear, total=256, inp1_unstable=2, inp2_unstable=2, delta_unstable=256
- layer_idx=8, type=LayerType.Linear, total=10, inp1_unstable=0, inp2_unstable=0, delta_unstable=10

Time for backsubstitution: 1.73 seconds

### RS candidates at layer 1
type: RSZ, layer: 1, pos: 199
type: RSZ, layer: 1, pos: 204
type: RSZ, layer: 1, pos: 250

Time for candidate selection: 0.20 seconds

### Candidate
type: RSZ, layer: 1, pos: 199

### Relational analysis RSZ of RS_RSZ1_RSZ1_RSZ2_RSZ1_RSZ2_RSZ2_RSZ1_RSZ1

#### Relational analysis RSZ result of RS_RSZ1_RSZ1_RSZ2_RSZ1_RSZ2_RSZ2_RSZ1_RSZ1
Status: Status.UNKNOWN
Output dim: 1, lower bound: -0.0007785, upper bound: 0.0007785
time: 0.64 seconds

### Relational analysis RSZ of RS_RSZ1_RSZ1_RSZ2_RSZ1_RSZ2_RSZ2_RSZ1_RSZ2

#### Relational analysis RSZ result of RS_RSZ1_RSZ1_RSZ2_RSZ1_RSZ2_RSZ2_RSZ1_RSZ2
Status: Status.UNKNOWN
Output dim: 1, lower bound: -0.0007785, upper bound: 0.0007785
time: 0.66 seconds

## BFS RS instance: RS_RSZ1_RSZ1_RSZ2_RSZ1_RSZ2_RSZ2_RSZ2

### Backsubstitution after applying RS history:
0: 0.0006996, 0.0011177, 0.0006996, 0.0011177, -0.0004181, 0.0004181
1: 0.9934013, 0.9943976, 0.9934013, 0.9943976, -0.0009964, 0.0009964
2: -0.0088595, -0.0053283, -0.0088595, -0.0053283, -0.0032432, 0.0032375
3: 0.0035633, 0.0041662, 0.0035633, 0.0041662, -0.0006029, 0.0006029
4: 0.0026282, 0.0054191, 0.0026282, 0.0054191, -0.0027909, 0.0027909
5: 0.0050201, 0.0064446, 0.0050201, 0.0064446, -0.0014245, 0.0014245
6: -0.0021963, -0.0008150, -0.0021963, -0.0008150, -0.0013813, 0.0013813
7: -0.0082640, -0.0074494, -0.0082640, -0.0074494, -0.0008146, 0.0008146
8: 0.0052524, 0.0098919, 0.0052524, 0.0098919, -0.0045611, 0.0045626
9: -0.0036845, -0.0030730, -0.0036845, -0.0030730, -0.0006115, 0.0006115

### Unstable ReLU Count (Linear/Conv2D Layers)
- layer_idx=0, type=LayerType.Linear, total=256, inp1_unstable=10, inp2_unstable=10, delta_unstable=249
- layer_idx=2, type=LayerType.Linear, total=256, inp1_unstable=8, inp2_unstable=8, delta_unstable=256
- layer_idx=4, type=LayerType.Linear, total=256, inp1_unstable=0, inp2_unstable=0, delta_unstable=256
- layer_idx=6, type=LayerType.Linear, total=256, inp1_unstable=2, inp2_unstable=2, delta_unstable=256
- layer_idx=8, type=LayerType.Linear, total=10, inp1_unstable=0, inp2_unstable=0, delta_unstable=10

Time for backsubstitution: 1.68 seconds

### RS candidates at layer 1
type: RSZ, layer: 1, pos: 199
type: RSZ, layer: 1, pos: 204
type: RSZ, layer: 1, pos: 250

Time for candidate selection: 0.18 seconds

### Candidate
type: RSZ, layer: 1, pos: 199

### Relational analysis RSZ of RS_RSZ1_RSZ1_RSZ2_RSZ1_RSZ2_RSZ2_RSZ2_RSZ1

#### Relational analysis RSZ result of RS_RSZ1_RSZ1_RSZ2_RSZ1_RSZ2_RSZ2_RSZ2_RSZ1
Status: Status.UNKNOWN
Output dim: 1, lower bound: -0.0007785, upper bound: 0.0007785
time: 0.64 seconds

### Relational analysis RSZ of RS_RSZ1_RSZ1_RSZ2_RSZ1_RSZ2_RSZ2_RSZ2_RSZ2

#### Relational analysis RSZ result of RS_RSZ1_RSZ1_RSZ2_RSZ1_RSZ2_RSZ2_RSZ2_RSZ2
Status: Status.UNKNOWN
Output dim: 1, lower bound: -0.0007785, upper bound: 0.0007785
time: 0.66 seconds

## BFS RS instance: RS_RSZ1_RSZ1_RSZ2_RSZ2_RSZ1_RSZ1_RSZ1

### Backsubstitution after applying RS history:
0: 0.0006996, 0.0011177, 0.0006996, 0.0011177, -0.0004181, 0.0004181
1: 0.9934013, 0.9943976, 0.9934013, 0.9943976, -0.0009964, 0.0009964
2: -0.0088595, -0.0053283, -0.0088595, -0.0053283, -0.0032397, 0.0032384
3: 0.0035633, 0.0041662, 0.0035633, 0.0041662, -0.0006029, 0.0006029
4: 0.0026282, 0.0054191, 0.0026282, 0.0054191, -0.0027909, 0.0027909
5: 0.0050201, 0.0064446, 0.0050201, 0.0064446, -0.0014245, 0.0014245
6: -0.0021963, -0.0008150, -0.0021963, -0.0008150, -0.0013813, 0.0013813
7: -0.0082640, -0.0074494, -0.0082640, -0.0074494, -0.0008146, 0.0008146
8: 0.0052524, 0.0098919, 0.0052524, 0.0098919, -0.0045614, 0.0045617
9: -0.0036845, -0.0030730, -0.0036845, -0.0030730, -0.0006115, 0.0006115

### Unstable ReLU Count (Linear/Conv2D Layers)
- layer_idx=0, type=LayerType.Linear, total=256, inp1_unstable=10, inp2_unstable=10, delta_unstable=249
- layer_idx=2, type=LayerType.Linear, total=256, inp1_unstable=8, inp2_unstable=8, delta_unstable=256
- layer_idx=4, type=LayerType.Linear, total=256, inp1_unstable=0, inp2_unstable=0, delta_unstable=256
- layer_idx=6, type=LayerType.Linear, total=256, inp1_unstable=2, inp2_unstable=2, delta_unstable=256
- layer_idx=8, type=LayerType.Linear, total=10, inp1_unstable=0, inp2_unstable=0, delta_unstable=10

Time for backsubstitution: 1.74 seconds

### RS candidates at layer 1
type: RSZ, layer: 1, pos: 199
type: RSZ, layer: 1, pos: 204
type: RSZ, layer: 1, pos: 250

Time for candidate selection: 0.24 seconds

### Candidate
type: RSZ, layer: 1, pos: 199

### Relational analysis RSZ of RS_RSZ1_RSZ1_RSZ2_RSZ2_RSZ1_RSZ1_RSZ1_RSZ1

#### Relational analysis RSZ result of RS_RSZ1_RSZ1_RSZ2_RSZ2_RSZ1_RSZ1_RSZ1_RSZ1
Status: Status.UNKNOWN
Output dim: 1, lower bound: -0.0007785, upper bound: 0.0007785
time: 0.65 seconds

### Relational analysis RSZ of RS_RSZ1_RSZ1_RSZ2_RSZ2_RSZ1_RSZ1_RSZ1_RSZ2

#### Relational analysis RSZ result of RS_RSZ1_RSZ1_RSZ2_RSZ2_RSZ1_RSZ1_RSZ1_RSZ2
Status: Status.UNKNOWN
Output dim: 1, lower bound: -0.0007785, upper bound: 0.0007785
time: 0.63 seconds

## BFS RS instance: RS_RSZ1_RSZ1_RSZ2_RSZ2_RSZ1_RSZ1_RSZ2

### Backsubstitution after applying RS history:
0: 0.0006996, 0.0011177, 0.0006996, 0.0011177, -0.0004181, 0.0004181
1: 0.9934013, 0.9943976, 0.9934013, 0.9943976, -0.0009964, 0.0009964
2: -0.0088595, -0.0053283, -0.0088595, -0.0053283, -0.0032431, 0.0032380
3: 0.0035633, 0.0041662, 0.0035633, 0.0041662, -0.0006029, 0.0006029
4: 0.0026282, 0.0054191, 0.0026282, 0.0054191, -0.0027909, 0.0027909
5: 0.0050201, 0.0064446, 0.0050201, 0.0064446, -0.0014245, 0.0014245
6: -0.0021963, -0.0008150, -0.0021963, -0.0008150, -0.0013813, 0.0013813
7: -0.0082640, -0.0074494, -0.0082640, -0.0074494, -0.0008146, 0.0008146
8: 0.0052524, 0.0098919, 0.0052524, 0.0098919, -0.0045612, 0.0045626
9: -0.0036845, -0.0030730, -0.0036845, -0.0030730, -0.0006115, 0.0006115

### Unstable ReLU Count (Linear/Conv2D Layers)
- layer_idx=0, type=LayerType.Linear, total=256, inp1_unstable=10, inp2_unstable=10, delta_unstable=249
- layer_idx=2, type=LayerType.Linear, total=256, inp1_unstable=8, inp2_unstable=8, delta_unstable=256
- layer_idx=4, type=LayerType.Linear, total=256, inp1_unstable=0, inp2_unstable=0, delta_unstable=256
- layer_idx=6, type=LayerType.Linear, total=256, inp1_unstable=2, inp2_unstable=2, delta_unstable=256
- layer_idx=8, type=LayerType.Linear, total=10, inp1_unstable=0, inp2_unstable=0, delta_unstable=10

Time for backsubstitution: 1.74 seconds

### RS candidates at layer 1
type: RSZ, layer: 1, pos: 199
type: RSZ, layer: 1, pos: 204
type: RSZ, layer: 1, pos: 250

Time for candidate selection: 0.21 seconds

### Candidate
type: RSZ, layer: 1, pos: 199

### Relational analysis RSZ of RS_RSZ1_RSZ1_RSZ2_RSZ2_RSZ1_RSZ1_RSZ2_RSZ1

#### Relational analysis RSZ result of RS_RSZ1_RSZ1_RSZ2_RSZ2_RSZ1_RSZ1_RSZ2_RSZ1
Status: Status.UNKNOWN
Output dim: 1, lower bound: -0.0007785, upper bound: 0.0007785
time: 0.64 seconds

### Relational analysis RSZ of RS_RSZ1_RSZ1_RSZ2_RSZ2_RSZ1_RSZ1_RSZ2_RSZ2

#### Relational analysis RSZ result of RS_RSZ1_RSZ1_RSZ2_RSZ2_RSZ1_RSZ1_RSZ2_RSZ2
Status: Status.UNKNOWN
Output dim: 1, lower bound: -0.0007785, upper bound: 0.0007785
time: 0.63 seconds

## BFS RS instance: RS_RSZ1_RSZ1_RSZ2_RSZ2_RSZ1_RSZ2_RSZ1

### Backsubstitution after applying RS history:
0: 0.0006996, 0.0011177, 0.0006996, 0.0011177, -0.0004181, 0.0004181
1: 0.9934013, 0.9943976, 0.9934013, 0.9943976, -0.0009964, 0.0009964
2: -0.0088595, -0.0053283, -0.0088595, -0.0053283, -0.0032410, 0.0032375
3: 0.0035633, 0.0041662, 0.0035633, 0.0041662, -0.0006029, 0.0006029
4: 0.0026282, 0.0054191, 0.0026282, 0.0054191, -0.0027909, 0.0027909
5: 0.0050201, 0.0064446, 0.0050201, 0.0064446, -0.0014245, 0.0014245
6: -0.0021963, -0.0008150, -0.0021963, -0.0008150, -0.0013813, 0.0013813
7: -0.0082640, -0.0074494, -0.0082640, -0.0074494, -0.0008146, 0.0008146
8: 0.0052524, 0.0098919, 0.0052524, 0.0098919, -0.0045611, 0.0045620
9: -0.0036845, -0.0030730, -0.0036845, -0.0030730, -0.0006115, 0.0006115

### Unstable ReLU Count (Linear/Conv2D Layers)
- layer_idx=0, type=LayerType.Linear, total=256, inp1_unstable=10, inp2_unstable=10, delta_unstable=249
- layer_idx=2, type=LayerType.Linear, total=256, inp1_unstable=8, inp2_unstable=8, delta_unstable=256
- layer_idx=4, type=LayerType.Linear, total=256, inp1_unstable=0, inp2_unstable=0, delta_unstable=256
- layer_idx=6, type=LayerType.Linear, total=256, inp1_unstable=2, inp2_unstable=2, delta_unstable=256
- layer_idx=8, type=LayerType.Linear, total=10, inp1_unstable=0, inp2_unstable=0, delta_unstable=10

Time for backsubstitution: 1.71 seconds

### RS candidates at layer 1
type: RSZ, layer: 1, pos: 199
type: RSZ, layer: 1, pos: 204
type: RSZ, layer: 1, pos: 250

Time for candidate selection: 0.18 seconds

### Candidate
type: RSZ, layer: 1, pos: 199

### Relational analysis RSZ of RS_RSZ1_RSZ1_RSZ2_RSZ2_RSZ1_RSZ2_RSZ1_RSZ1

#### Relational analysis RSZ result of RS_RSZ1_RSZ1_RSZ2_RSZ2_RSZ1_RSZ2_RSZ1_RSZ1
Status: Status.UNKNOWN
Output dim: 1, lower bound: -0.0007785, upper bound: 0.0007785
time: 0.61 seconds

### Relational analysis RSZ of RS_RSZ1_RSZ1_RSZ2_RSZ2_RSZ1_RSZ2_RSZ1_RSZ2

#### Relational analysis RSZ result of RS_RSZ1_RSZ1_RSZ2_RSZ2_RSZ1_RSZ2_RSZ1_RSZ2
Status: Status.UNKNOWN
Output dim: 1, lower bound: -0.0007785, upper bound: 0.0007785
time: 0.70 seconds

## BFS RS instance: RS_RSZ1_RSZ1_RSZ2_RSZ2_RSZ1_RSZ2_RSZ2

### Backsubstitution after applying RS history:
0: 0.0006996, 0.0011177, 0.0006996, 0.0011177, -0.0004181, 0.0004181
1: 0.9934013, 0.9943976, 0.9934013, 0.9943976, -0.0009964, 0.0009964
2: -0.0088595, -0.0053283, -0.0088595, -0.0053283, -0.0032443, 0.0032366
3: 0.0035633, 0.0041662, 0.0035633, 0.0041662, -0.0006029, 0.0006029
4: 0.0026282, 0.0054191, 0.0026282, 0.0054191, -0.0027909, 0.0027909
5: 0.0050201, 0.0064446, 0.0050201, 0.0064446, -0.0014245, 0.0014245
6: -0.0021963, -0.0008150, -0.0021963, -0.0008150, -0.0013813, 0.0013813
7: -0.0082640, -0.0074494, -0.0082640, -0.0074494, -0.0008146, 0.0008146
8: 0.0052524, 0.0098919, 0.0052524, 0.0098919, -0.0045609, 0.0045629
9: -0.0036845, -0.0030730, -0.0036845, -0.0030730, -0.0006115, 0.0006115

### Unstable ReLU Count (Linear/Conv2D Layers)
- layer_idx=0, type=LayerType.Linear, total=256, inp1_unstable=10, inp2_unstable=10, delta_unstable=249
- layer_idx=2, type=LayerType.Linear, total=256, inp1_unstable=8, inp2_unstable=8, delta_unstable=256
- layer_idx=4, type=LayerType.Linear, total=256, inp1_unstable=0, inp2_unstable=0, delta_unstable=256
- layer_idx=6, type=LayerType.Linear, total=256, inp1_unstable=2, inp2_unstable=2, delta_unstable=256
- layer_idx=8, type=LayerType.Linear, total=10, inp1_unstable=0, inp2_unstable=0, delta_unstable=10

Time for backsubstitution: 1.74 seconds

### RS candidates at layer 1
type: RSZ, layer: 1, pos: 199
type: RSZ, layer: 1, pos: 204
type: RSZ, layer: 1, pos: 250

Time for candidate selection: 0.20 seconds

### Candidate
type: RSZ, layer: 1, pos: 199

### Relational analysis RSZ of RS_RSZ1_RSZ1_RSZ2_RSZ2_RSZ1_RSZ2_RSZ2_RSZ1

#### Relational analysis RSZ result of RS_RSZ1_RSZ1_RSZ2_RSZ2_RSZ1_RSZ2_RSZ2_RSZ1
Status: Status.UNKNOWN
Output dim: 1, lower bound: -0.0007785, upper bound: 0.0007785
time: 0.62 seconds

### Relational analysis RSZ of RS_RSZ1_RSZ1_RSZ2_RSZ2_RSZ1_RSZ2_RSZ2_RSZ2

#### Relational analysis RSZ result of RS_RSZ1_RSZ1_RSZ2_RSZ2_RSZ1_RSZ2_RSZ2_RSZ2
Status: Status.UNKNOWN
Output dim: 1, lower bound: -0.0007785, upper bound: 0.0007785
time: 0.67 seconds

## BFS RS instance: RS_RSZ1_RSZ1_RSZ2_RSZ2_RSZ2_RSZ1_RSZ1

### Backsubstitution after applying RS history:
0: 0.0006996, 0.0011177, 0.0006996, 0.0011177, -0.0004181, 0.0004181
1: 0.9934013, 0.9943976, 0.9934013, 0.9943976, -0.0009964, 0.0009964
2: -0.0088595, -0.0053283, -0.0088595, -0.0053283, -0.0032377, 0.0032396
3: 0.0035633, 0.0041662, 0.0035633, 0.0041662, -0.0006029, 0.0006029
4: 0.0026282, 0.0054191, 0.0026282, 0.0054191, -0.0027909, 0.0027909
5: 0.0050201, 0.0064446, 0.0050201, 0.0064446, -0.0014245, 0.0014245
6: -0.0021963, -0.0008150, -0.0021963, -0.0008150, -0.0013813, 0.0013813
7: -0.0082640, -0.0074494, -0.0082640, -0.0074494, -0.0008146, 0.0008146
8: 0.0052524, 0.0098919, 0.0052524, 0.0098919, -0.0045617, 0.0045612
9: -0.0036845, -0.0030730, -0.0036845, -0.0030730, -0.0006115, 0.0006115

### Unstable ReLU Count (Linear/Conv2D Layers)
- layer_idx=0, type=LayerType.Linear, total=256, inp1_unstable=10, inp2_unstable=10, delta_unstable=249
- layer_idx=2, type=LayerType.Linear, total=256, inp1_unstable=8, inp2_unstable=8, delta_unstable=256
- layer_idx=4, type=LayerType.Linear, total=256, inp1_unstable=0, inp2_unstable=0, delta_unstable=256
- layer_idx=6, type=LayerType.Linear, total=256, inp1_unstable=2, inp2_unstable=2, delta_unstable=256
- layer_idx=8, type=LayerType.Linear, total=10, inp1_unstable=0, inp2_unstable=0, delta_unstable=10

Time for backsubstitution: 1.75 seconds

### RS candidates at layer 1
type: RSZ, layer: 1, pos: 199
type: RSZ, layer: 1, pos: 204
type: RSZ, layer: 1, pos: 250

Time for candidate selection: 0.19 seconds

### Candidate
type: RSZ, layer: 1, pos: 199

### Relational analysis RSZ of RS_RSZ1_RSZ1_RSZ2_RSZ2_RSZ2_RSZ1_RSZ1_RSZ1

#### Relational analysis RSZ result of RS_RSZ1_RSZ1_RSZ2_RSZ2_RSZ2_RSZ1_RSZ1_RSZ1
Status: Status.UNKNOWN
Output dim: 1, lower bound: -0.0007785, upper bound: 0.0007785
time: 0.60 seconds

### Relational analysis RSZ of RS_RSZ1_RSZ1_RSZ2_RSZ2_RSZ2_RSZ1_RSZ1_RSZ2

#### Relational analysis RSZ result of RS_RSZ1_RSZ1_RSZ2_RSZ2_RSZ2_RSZ1_RSZ1_RSZ2
Status: Status.UNKNOWN
Output dim: 1, lower bound: -0.0007785, upper bound: 0.0007785
time: 0.61 seconds

## BFS RS instance: RS_RSZ1_RSZ1_RSZ2_RSZ2_RSZ2_RSZ1_RSZ2

### Backsubstitution after applying RS history:
0: 0.0006996, 0.0011177, 0.0006996, 0.0011177, -0.0004181, 0.0004181
1: 0.9934013, 0.9943976, 0.9934013, 0.9943976, -0.0009964, 0.0009964
2: -0.0088595, -0.0053283, -0.0088595, -0.0053283, -0.0032410, 0.0032401
3: 0.0035633, 0.0041662, 0.0035633, 0.0041662, -0.0006029, 0.0006029
4: 0.0026282, 0.0054191, 0.0026282, 0.0054191, -0.0027909, 0.0027909
5: 0.0050201, 0.0064446, 0.0050201, 0.0064446, -0.0014245, 0.0014245
6: -0.0021963, -0.0008150, -0.0021963, -0.0008150, -0.0013813, 0.0013813
7: -0.0082640, -0.0074494, -0.0082640, -0.0074494, -0.0008146, 0.0008146
8: 0.0052524, 0.0098919, 0.0052524, 0.0098919, -0.0045618, 0.0045620
9: -0.0036845, -0.0030730, -0.0036845, -0.0030730, -0.0006115, 0.0006115

### Unstable ReLU Count (Linear/Conv2D Layers)
- layer_idx=0, type=LayerType.Linear, total=256, inp1_unstable=10, inp2_unstable=10, delta_unstable=249
- layer_idx=2, type=LayerType.Linear, total=256, inp1_unstable=8, inp2_unstable=8, delta_unstable=256
- layer_idx=4, type=LayerType.Linear, total=256, inp1_unstable=0, inp2_unstable=0, delta_unstable=256
- layer_idx=6, type=LayerType.Linear, total=256, inp1_unstable=2, inp2_unstable=2, delta_unstable=256
- layer_idx=8, type=LayerType.Linear, total=10, inp1_unstable=0, inp2_unstable=0, delta_unstable=10

Time for backsubstitution: 1.80 seconds

### RS candidates at layer 1
type: RSZ, layer: 1, pos: 199
type: RSZ, layer: 1, pos: 204
type: RSZ, layer: 1, pos: 250

Time for candidate selection: 0.24 seconds

### Candidate
type: RSZ, layer: 1, pos: 199

### Relational analysis RSZ of RS_RSZ1_RSZ1_RSZ2_RSZ2_RSZ2_RSZ1_RSZ2_RSZ1

#### Relational analysis RSZ result of RS_RSZ1_RSZ1_RSZ2_RSZ2_RSZ2_RSZ1_RSZ2_RSZ1
Status: Status.UNKNOWN
Output dim: 1, lower bound: -0.0007785, upper bound: 0.0007785
time: 0.61 seconds

### Relational analysis RSZ of RS_RSZ1_RSZ1_RSZ2_RSZ2_RSZ2_RSZ1_RSZ2_RSZ2

#### Relational analysis RSZ result of RS_RSZ1_RSZ1_RSZ2_RSZ2_RSZ2_RSZ1_RSZ2_RSZ2
Status: Status.UNKNOWN
Output dim: 1, lower bound: -0.0007785, upper bound: 0.0007785
time: 0.58 seconds

## BFS RS instance: RS_RSZ1_RSZ1_RSZ2_RSZ2_RSZ2_RSZ2_RSZ1

### Backsubstitution after applying RS history:
0: 0.0006996, 0.0011177, 0.0006996, 0.0011177, -0.0004181, 0.0004181
1: 0.9934013, 0.9943976, 0.9934013, 0.9943976, -0.0009964, 0.0009964
2: -0.0088595, -0.0053283, -0.0088595, -0.0053283, -0.0032388, 0.0032382
3: 0.0035633, 0.0041662, 0.0035633, 0.0041662, -0.0006029, 0.0006029
4: 0.0026282, 0.0054191, 0.0026282, 0.0054191, -0.0027909, 0.0027909
5: 0.0050201, 0.0064446, 0.0050201, 0.0064446, -0.0014245, 0.0014245
6: -0.0021963, -0.0008150, -0.0021963, -0.0008150, -0.0013813, 0.0013813
7: -0.0082640, -0.0074494, -0.0082640, -0.0074494, -0.0008146, 0.0008146
8: 0.0052524, 0.0098919, 0.0052524, 0.0098919, -0.0045613, 0.0045615
9: -0.0036845, -0.0030730, -0.0036845, -0.0030730, -0.0006115, 0.0006115

### Unstable ReLU Count (Linear/Conv2D Layers)
- layer_idx=0, type=LayerType.Linear, total=256, inp1_unstable=10, inp2_unstable=10, delta_unstable=249
- layer_idx=2, type=LayerType.Linear, total=256, inp1_unstable=8, inp2_unstable=8, delta_unstable=256
- layer_idx=4, type=LayerType.Linear, total=256, inp1_unstable=0, inp2_unstable=0, delta_unstable=256
- layer_idx=6, type=LayerType.Linear, total=256, inp1_unstable=2, inp2_unstable=2, delta_unstable=256
- layer_idx=8, type=LayerType.Linear, total=10, inp1_unstable=0, inp2_unstable=0, delta_unstable=10

Time for backsubstitution: 1.70 seconds

### RS candidates at layer 1
type: RSZ, layer: 1, pos: 199
type: RSZ, layer: 1, pos: 204
type: RSZ, layer: 1, pos: 250

Time for candidate selection: 0.19 seconds

### Candidate
type: RSZ, layer: 1, pos: 199

### Relational analysis RSZ of RS_RSZ1_RSZ1_RSZ2_RSZ2_RSZ2_RSZ2_RSZ1_RSZ1

#### Relational analysis RSZ result of RS_RSZ1_RSZ1_RSZ2_RSZ2_RSZ2_RSZ2_RSZ1_RSZ1
Status: Status.UNKNOWN
Output dim: 1, lower bound: -0.0007785, upper bound: 0.0007785
time: 0.59 seconds

### Relational analysis RSZ of RS_RSZ1_RSZ1_RSZ2_RSZ2_RSZ2_RSZ2_RSZ1_RSZ2

#### Relational analysis RSZ result of RS_RSZ1_RSZ1_RSZ2_RSZ2_RSZ2_RSZ2_RSZ1_RSZ2
Status: Status.UNKNOWN
Output dim: 1, lower bound: -0.0007785, upper bound: 0.0007785
time: 0.57 seconds

## BFS RS instance: RS_RSZ1_RSZ1_RSZ2_RSZ2_RSZ2_RSZ2_RSZ2

### Backsubstitution after applying RS history:
0: 0.0006996, 0.0011177, 0.0006996, 0.0011177, -0.0004181, 0.0004181
1: 0.9934013, 0.9943976, 0.9934013, 0.9943976, -0.0009964, 0.0009964
2: -0.0088595, -0.0053283, -0.0088595, -0.0053283, -0.0032422, 0.0032385
3: 0.0035633, 0.0041662, 0.0035633, 0.0041662, -0.0006029, 0.0006029
4: 0.0026282, 0.0054191, 0.0026282, 0.0054191, -0.0027909, 0.0027909
5: 0.0050201, 0.0064446, 0.0050201, 0.0064446, -0.0014245, 0.0014245
6: -0.0021963, -0.0008150, -0.0021963, -0.0008150, -0.0013813, 0.0013813
7: -0.0082640, -0.0074494, -0.0082640, -0.0074494, -0.0008146, 0.0008146
8: 0.0052524, 0.0098919, 0.0052524, 0.0098919, -0.0045614, 0.0045624
9: -0.0036845, -0.0030730, -0.0036845, -0.0030730, -0.0006115, 0.0006115

### Unstable ReLU Count (Linear/Conv2D Layers)
- layer_idx=0, type=LayerType.Linear, total=256, inp1_unstable=10, inp2_unstable=10, delta_unstable=249
- layer_idx=2, type=LayerType.Linear, total=256, inp1_unstable=8, inp2_unstable=8, delta_unstable=256
- layer_idx=4, type=LayerType.Linear, total=256, inp1_unstable=0, inp2_unstable=0, delta_unstable=256
- layer_idx=6, type=LayerType.Linear, total=256, inp1_unstable=2, inp2_unstable=2, delta_unstable=256
- layer_idx=8, type=LayerType.Linear, total=10, inp1_unstable=0, inp2_unstable=0, delta_unstable=10

Time for backsubstitution: 1.72 seconds

### RS candidates at layer 1
type: RSZ, layer: 1, pos: 199
type: RSZ, layer: 1, pos: 204
type: RSZ, layer: 1, pos: 250

Time for candidate selection: 0.18 seconds

### Candidate
type: RSZ, layer: 1, pos: 199

### Relational analysis RSZ of RS_RSZ1_RSZ1_RSZ2_RSZ2_RSZ2_RSZ2_RSZ2_RSZ1

#### Relational analysis RSZ result of RS_RSZ1_RSZ1_RSZ2_RSZ2_RSZ2_RSZ2_RSZ2_RSZ1
Status: Status.UNKNOWN
Output dim: 1, lower bound: -0.0007785, upper bound: 0.0007785
time: 0.58 seconds

### Relational analysis RSZ of RS_RSZ1_RSZ1_RSZ2_RSZ2_RSZ2_RSZ2_RSZ2_RSZ2

#### Relational analysis RSZ result of RS_RSZ1_RSZ1_RSZ2_RSZ2_RSZ2_RSZ2_RSZ2_RSZ2
Status: Status.UNKNOWN
Output dim: 1, lower bound: -0.0007785, upper bound: 0.0007785
time: 0.58 seconds

## BFS RS instance: RS_RSZ1_RSZ2_RSZ1_RSZ1_RSZ1_RSZ1_RSZ1

### Backsubstitution after applying RS history:
0: 0.0006996, 0.0011177, 0.0006996, 0.0011177, -0.0004181, 0.0004181
1: 0.9934013, 0.9943976, 0.9934013, 0.9943976, -0.0009964, 0.0009964
2: -0.0088595, -0.0053283, -0.0088595, -0.0053283, -0.0032387, 0.0032386
3: 0.0035633, 0.0041662, 0.0035633, 0.0041662, -0.0006029, 0.0006029
4: 0.0026282, 0.0054191, 0.0026282, 0.0054191, -0.0027909, 0.0027909
5: 0.0050201, 0.0064446, 0.0050201, 0.0064446, -0.0014245, 0.0014245
6: -0.0021963, -0.0008150, -0.0021963, -0.0008150, -0.0013813, 0.0013813
7: -0.0082640, -0.0074494, -0.0082640, -0.0074494, -0.0008146, 0.0008146
8: 0.0052524, 0.0098919, 0.0052524, 0.0098919, -0.0045614, 0.0045614
9: -0.0036845, -0.0030730, -0.0036845, -0.0030730, -0.0006115, 0.0006115

### Unstable ReLU Count (Linear/Conv2D Layers)
- layer_idx=0, type=LayerType.Linear, total=256, inp1_unstable=10, inp2_unstable=10, delta_unstable=249
- layer_idx=2, type=LayerType.Linear, total=256, inp1_unstable=8, inp2_unstable=8, delta_unstable=256
- layer_idx=4, type=LayerType.Linear, total=256, inp1_unstable=0, inp2_unstable=0, delta_unstable=256
- layer_idx=6, type=LayerType.Linear, total=256, inp1_unstable=2, inp2_unstable=2, delta_unstable=256
- layer_idx=8, type=LayerType.Linear, total=10, inp1_unstable=0, inp2_unstable=0, delta_unstable=10

Time for backsubstitution: 1.72 seconds

### RS candidates at layer 1
type: RSZ, layer: 1, pos: 199
type: RSZ, layer: 1, pos: 204
type: RSZ, layer: 1, pos: 250

Time for candidate selection: 0.19 seconds

### Candidate
type: RSZ, layer: 1, pos: 199

### Relational analysis RSZ of RS_RSZ1_RSZ2_RSZ1_RSZ1_RSZ1_RSZ1_RSZ1_RSZ1

#### Relational analysis RSZ result of RS_RSZ1_RSZ2_RSZ1_RSZ1_RSZ1_RSZ1_RSZ1_RSZ1
Status: Status.UNKNOWN
Output dim: 1, lower bound: -0.0007785, upper bound: 0.0007785
time: 0.64 seconds

### Relational analysis RSZ of RS_RSZ1_RSZ2_RSZ1_RSZ1_RSZ1_RSZ1_RSZ1_RSZ2

#### Relational analysis RSZ result of RS_RSZ1_RSZ2_RSZ1_RSZ1_RSZ1_RSZ1_RSZ1_RSZ2
Status: Status.UNKNOWN
Output dim: 1, lower bound: -0.0007785, upper bound: 0.0007785
time: 0.63 seconds

## BFS RS instance: RS_RSZ1_RSZ2_RSZ1_RSZ1_RSZ1_RSZ1_RSZ2

### Backsubstitution after applying RS history:
0: 0.0006996, 0.0011177, 0.0006996, 0.0011177, -0.0004181, 0.0004181
1: 0.9934013, 0.9943976, 0.9934013, 0.9943976, -0.0009964, 0.0009964
2: -0.0088595, -0.0053283, -0.0088595, -0.0053283, -0.0032420, 0.0032389
3: 0.0035633, 0.0041662, 0.0035633, 0.0041662, -0.0006029, 0.0006029
4: 0.0026282, 0.0054191, 0.0026282, 0.0054191, -0.0027909, 0.0027909
5: 0.0050201, 0.0064446, 0.0050201, 0.0064446, -0.0014245, 0.0014245
6: -0.0021963, -0.0008150, -0.0021963, -0.0008150, -0.0013813, 0.0013813
7: -0.0082640, -0.0074494, -0.0082640, -0.0074494, -0.0008146, 0.0008146
8: 0.0052524, 0.0098919, 0.0052524, 0.0098919, -0.0045615, 0.0045623
9: -0.0036845, -0.0030730, -0.0036845, -0.0030730, -0.0006115, 0.0006115

### Unstable ReLU Count (Linear/Conv2D Layers)
- layer_idx=0, type=LayerType.Linear, total=256, inp1_unstable=10, inp2_unstable=10, delta_unstable=249
- layer_idx=2, type=LayerType.Linear, total=256, inp1_unstable=8, inp2_unstable=8, delta_unstable=256
- layer_idx=4, type=LayerType.Linear, total=256, inp1_unstable=0, inp2_unstable=0, delta_unstable=256
- layer_idx=6, type=LayerType.Linear, total=256, inp1_unstable=2, inp2_unstable=2, delta_unstable=256
- layer_idx=8, type=LayerType.Linear, total=10, inp1_unstable=0, inp2_unstable=0, delta_unstable=10

Time for backsubstitution: 1.83 seconds

### RS candidates at layer 1
type: RSZ, layer: 1, pos: 199
type: RSZ, layer: 1, pos: 204
type: RSZ, layer: 1, pos: 250

Time for candidate selection: 0.20 seconds

### Candidate
type: RSZ, layer: 1, pos: 199

### Relational analysis RSZ of RS_RSZ1_RSZ2_RSZ1_RSZ1_RSZ1_RSZ1_RSZ2_RSZ1

#### Relational analysis RSZ result of RS_RSZ1_RSZ2_RSZ1_RSZ1_RSZ1_RSZ1_RSZ2_RSZ1
Status: Status.UNKNOWN
Output dim: 1, lower bound: -0.0007785, upper bound: 0.0007785
time: 0.62 seconds

### Relational analysis RSZ of RS_RSZ1_RSZ2_RSZ1_RSZ1_RSZ1_RSZ1_RSZ2_RSZ2

#### Relational analysis RSZ result of RS_RSZ1_RSZ2_RSZ1_RSZ1_RSZ1_RSZ1_RSZ2_RSZ2
Status: Status.UNKNOWN
Output dim: 1, lower bound: -0.0007785, upper bound: 0.0007785
time: 0.63 seconds

## BFS RS instance: RS_RSZ1_RSZ2_RSZ1_RSZ1_RSZ1_RSZ2_RSZ1

### Backsubstitution after applying RS history:
0: 0.0006996, 0.0011177, 0.0006996, 0.0011177, -0.0004181, 0.0004181
1: 0.9934013, 0.9943976, 0.9934013, 0.9943976, -0.0009964, 0.0009964
2: -0.0088595, -0.0053283, -0.0088595, -0.0053283, -0.0032402, 0.0032382
3: 0.0035633, 0.0041662, 0.0035633, 0.0041662, -0.0006029, 0.0006029
4: 0.0026282, 0.0054191, 0.0026282, 0.0054191, -0.0027909, 0.0027909
5: 0.0050201, 0.0064446, 0.0050201, 0.0064446, -0.0014245, 0.0014245
6: -0.0021963, -0.0008150, -0.0021963, -0.0008150, -0.0013813, 0.0013813
7: -0.0082640, -0.0074494, -0.0082640, -0.0074494, -0.0008146, 0.0008146
8: 0.0052524, 0.0098919, 0.0052524, 0.0098919, -0.0045613, 0.0045618
9: -0.0036845, -0.0030730, -0.0036845, -0.0030730, -0.0006115, 0.0006115

### Unstable ReLU Count (Linear/Conv2D Layers)
- layer_idx=0, type=LayerType.Linear, total=256, inp1_unstable=10, inp2_unstable=10, delta_unstable=249
- layer_idx=2, type=LayerType.Linear, total=256, inp1_unstable=8, inp2_unstable=8, delta_unstable=256
- layer_idx=4, type=LayerType.Linear, total=256, inp1_unstable=0, inp2_unstable=0, delta_unstable=256
- layer_idx=6, type=LayerType.Linear, total=256, inp1_unstable=2, inp2_unstable=2, delta_unstable=256
- layer_idx=8, type=LayerType.Linear, total=10, inp1_unstable=0, inp2_unstable=0, delta_unstable=10

Time for backsubstitution: 1.95 seconds

### RS candidates at layer 1
type: RSZ, layer: 1, pos: 199
type: RSZ, layer: 1, pos: 204
type: RSZ, layer: 1, pos: 250

Time for candidate selection: 0.24 seconds

### Candidate
type: RSZ, layer: 1, pos: 199

### Relational analysis RSZ of RS_RSZ1_RSZ2_RSZ1_RSZ1_RSZ1_RSZ2_RSZ1_RSZ1

#### Relational analysis RSZ result of RS_RSZ1_RSZ2_RSZ1_RSZ1_RSZ1_RSZ2_RSZ1_RSZ1
Status: Status.UNKNOWN
Output dim: 1, lower bound: -0.0007785, upper bound: 0.0007785
time: 0.71 seconds

### Relational analysis RSZ of RS_RSZ1_RSZ2_RSZ1_RSZ1_RSZ1_RSZ2_RSZ1_RSZ2

#### Relational analysis RSZ result of RS_RSZ1_RSZ2_RSZ1_RSZ1_RSZ1_RSZ2_RSZ1_RSZ2
Status: Status.UNKNOWN
Output dim: 1, lower bound: -0.0007785, upper bound: 0.0007785
time: 0.71 seconds

## BFS RS instance: RS_RSZ1_RSZ2_RSZ1_RSZ1_RSZ1_RSZ2_RSZ2

### Backsubstitution after applying RS history:
0: 0.0006996, 0.0011177, 0.0006996, 0.0011177, -0.0004181, 0.0004181
1: 0.9934013, 0.9943976, 0.9934013, 0.9943976, -0.0009964, 0.0009964
2: -0.0088595, -0.0053283, -0.0088595, -0.0053283, -0.0032436, 0.0032381
3: 0.0035633, 0.0041662, 0.0035633, 0.0041662, -0.0006029, 0.0006029
4: 0.0026282, 0.0054191, 0.0026282, 0.0054191, -0.0027909, 0.0027909
5: 0.0050201, 0.0064446, 0.0050201, 0.0064446, -0.0014245, 0.0014245
6: -0.0021963, -0.0008150, -0.0021963, -0.0008150, -0.0013813, 0.0013813
7: -0.0082640, -0.0074494, -0.0082640, -0.0074494, -0.0008146, 0.0008146
8: 0.0052524, 0.0098919, 0.0052524, 0.0098919, -0.0045613, 0.0045627
9: -0.0036845, -0.0030730, -0.0036845, -0.0030730, -0.0006115, 0.0006115

### Unstable ReLU Count (Linear/Conv2D Layers)
- layer_idx=0, type=LayerType.Linear, total=256, inp1_unstable=10, inp2_unstable=10, delta_unstable=249
- layer_idx=2, type=LayerType.Linear, total=256, inp1_unstable=8, inp2_unstable=8, delta_unstable=256
- layer_idx=4, type=LayerType.Linear, total=256, inp1_unstable=0, inp2_unstable=0, delta_unstable=256
- layer_idx=6, type=LayerType.Linear, total=256, inp1_unstable=2, inp2_unstable=2, delta_unstable=256
- layer_idx=8, type=LayerType.Linear, total=10, inp1_unstable=0, inp2_unstable=0, delta_unstable=10

Time for backsubstitution: 2.08 seconds

### RS candidates at layer 1
type: RSZ, layer: 1, pos: 199
type: RSZ, layer: 1, pos: 204
type: RSZ, layer: 1, pos: 250

Time for candidate selection: 0.23 seconds

### Candidate
type: RSZ, layer: 1, pos: 199

### Relational analysis RSZ of RS_RSZ1_RSZ2_RSZ1_RSZ1_RSZ1_RSZ2_RSZ2_RSZ1

#### Relational analysis RSZ result of RS_RSZ1_RSZ2_RSZ1_RSZ1_RSZ1_RSZ2_RSZ2_RSZ1
Status: Status.UNKNOWN
Output dim: 1, lower bound: -0.0007785, upper bound: 0.0007785
time: 0.68 seconds

### Relational analysis RSZ of RS_RSZ1_RSZ2_RSZ1_RSZ1_RSZ1_RSZ2_RSZ2_RSZ2

#### Relational analysis RSZ result of RS_RSZ1_RSZ2_RSZ1_RSZ1_RSZ1_RSZ2_RSZ2_RSZ2
Status: Status.UNKNOWN
Output dim: 1, lower bound: -0.0007785, upper bound: 0.0007785
time: 0.66 seconds

## BFS RS instance: RS_RSZ1_RSZ2_RSZ1_RSZ1_RSZ2_RSZ1_RSZ1

### Backsubstitution after applying RS history:
0: 0.0006996, 0.0011177, 0.0006996, 0.0011177, -0.0004181, 0.0004181
1: 0.9934013, 0.9943976, 0.9934013, 0.9943976, -0.0009964, 0.0009964
2: -0.0088595, -0.0053283, -0.0088595, -0.0053283, -0.0032365, 0.0032405
3: 0.0035633, 0.0041662, 0.0035633, 0.0041662, -0.0006029, 0.0006029
4: 0.0026282, 0.0054191, 0.0026282, 0.0054191, -0.0027909, 0.0027909
5: 0.0050201, 0.0064446, 0.0050201, 0.0064446, -0.0014245, 0.0014245
6: -0.0021963, -0.0008150, -0.0021963, -0.0008150, -0.0013813, 0.0013813
7: -0.0082640, -0.0074494, -0.0082640, -0.0074494, -0.0008146, 0.0008146
8: 0.0052524, 0.0098919, 0.0052524, 0.0098919, -0.0045619, 0.0045609
9: -0.0036845, -0.0030730, -0.0036845, -0.0030730, -0.0006115, 0.0006115

### Unstable ReLU Count (Linear/Conv2D Layers)
- layer_idx=0, type=LayerType.Linear, total=256, inp1_unstable=10, inp2_unstable=10, delta_unstable=249
- layer_idx=2, type=LayerType.Linear, total=256, inp1_unstable=8, inp2_unstable=8, delta_unstable=256
- layer_idx=4, type=LayerType.Linear, total=256, inp1_unstable=0, inp2_unstable=0, delta_unstable=256
- layer_idx=6, type=LayerType.Linear, total=256, inp1_unstable=2, inp2_unstable=2, delta_unstable=256
- layer_idx=8, type=LayerType.Linear, total=10, inp1_unstable=0, inp2_unstable=0, delta_unstable=10

Time for backsubstitution: 1.82 seconds

### RS candidates at layer 1
type: RSZ, layer: 1, pos: 199
type: RSZ, layer: 1, pos: 204
type: RSZ, layer: 1, pos: 250

Time for candidate selection: 0.19 seconds

### Candidate
type: RSZ, layer: 1, pos: 199

### Relational analysis RSZ of RS_RSZ1_RSZ2_RSZ1_RSZ1_RSZ2_RSZ1_RSZ1_RSZ1

#### Relational analysis RSZ result of RS_RSZ1_RSZ2_RSZ1_RSZ1_RSZ2_RSZ1_RSZ1_RSZ1
Status: Status.UNKNOWN
Output dim: 1, lower bound: -0.0007785, upper bound: 0.0007785
time: 0.64 seconds

### Relational analysis RSZ of RS_RSZ1_RSZ2_RSZ1_RSZ1_RSZ2_RSZ1_RSZ1_RSZ2

#### Relational analysis RSZ result of RS_RSZ1_RSZ2_RSZ1_RSZ1_RSZ2_RSZ1_RSZ1_RSZ2
Status: Status.UNKNOWN
Output dim: 1, lower bound: -0.0007785, upper bound: 0.0007785
time: 0.65 seconds

## BFS RS instance: RS_RSZ1_RSZ2_RSZ1_RSZ1_RSZ2_RSZ1_RSZ2

### Backsubstitution after applying RS history:
0: 0.0006996, 0.0011177, 0.0006996, 0.0011177, -0.0004181, 0.0004181
1: 0.9934013, 0.9943976, 0.9934013, 0.9943976, -0.0009964, 0.0009964
2: -0.0088595, -0.0053283, -0.0088595, -0.0053283, -0.0032399, 0.0032410
3: 0.0035633, 0.0041662, 0.0035633, 0.0041662, -0.0006029, 0.0006029
4: 0.0026282, 0.0054191, 0.0026282, 0.0054191, -0.0027909, 0.0027909
5: 0.0050201, 0.0064446, 0.0050201, 0.0064446, -0.0014245, 0.0014245
6: -0.0021963, -0.0008150, -0.0021963, -0.0008150, -0.0013813, 0.0013813
7: -0.0082640, -0.0074494, -0.0082640, -0.0074494, -0.0008146, 0.0008146
8: 0.0052524, 0.0098919, 0.0052524, 0.0098919, -0.0045621, 0.0045618
9: -0.0036845, -0.0030730, -0.0036845, -0.0030730, -0.0006115, 0.0006115

### Unstable ReLU Count (Linear/Conv2D Layers)
- layer_idx=0, type=LayerType.Linear, total=256, inp1_unstable=10, inp2_unstable=10, delta_unstable=249
- layer_idx=2, type=LayerType.Linear, total=256, inp1_unstable=8, inp2_unstable=8, delta_unstable=256
- layer_idx=4, type=LayerType.Linear, total=256, inp1_unstable=0, inp2_unstable=0, delta_unstable=256
- layer_idx=6, type=LayerType.Linear, total=256, inp1_unstable=2, inp2_unstable=2, delta_unstable=256
- layer_idx=8, type=LayerType.Linear, total=10, inp1_unstable=0, inp2_unstable=0, delta_unstable=10

Time for backsubstitution: 1.81 seconds

### RS candidates at layer 1
type: RSZ, layer: 1, pos: 199
type: RSZ, layer: 1, pos: 204
type: RSZ, layer: 1, pos: 250

Time for candidate selection: 0.19 seconds

### Candidate
type: RSZ, layer: 1, pos: 199

### Relational analysis RSZ of RS_RSZ1_RSZ2_RSZ1_RSZ1_RSZ2_RSZ1_RSZ2_RSZ1

#### Relational analysis RSZ result of RS_RSZ1_RSZ2_RSZ1_RSZ1_RSZ2_RSZ1_RSZ2_RSZ1
Status: Status.UNKNOWN
Output dim: 1, lower bound: -0.0007785, upper bound: 0.0007785
time: 0.64 seconds

### Relational analysis RSZ of RS_RSZ1_RSZ2_RSZ1_RSZ1_RSZ2_RSZ1_RSZ2_RSZ2

#### Relational analysis RSZ result of RS_RSZ1_RSZ2_RSZ1_RSZ1_RSZ2_RSZ1_RSZ2_RSZ2
Status: Status.UNKNOWN
Output dim: 1, lower bound: -0.0007785, upper bound: 0.0007785
time: 0.63 seconds

## BFS RS instance: RS_RSZ1_RSZ2_RSZ1_RSZ1_RSZ2_RSZ2_RSZ1

### Backsubstitution after applying RS history:
0: 0.0006996, 0.0011177, 0.0006996, 0.0011177, -0.0004181, 0.0004181
1: 0.9934013, 0.9943976, 0.9934013, 0.9943976, -0.0009964, 0.0009964
2: -0.0088595, -0.0053283, -0.0088595, -0.0053283, -0.0032379, 0.0032394
3: 0.0035633, 0.0041662, 0.0035633, 0.0041662, -0.0006029, 0.0006029
4: 0.0026282, 0.0054191, 0.0026282, 0.0054191, -0.0027909, 0.0027909
5: 0.0050201, 0.0064446, 0.0050201, 0.0064446, -0.0014245, 0.0014245
6: -0.0021963, -0.0008150, -0.0021963, -0.0008150, -0.0013813, 0.0013813
7: -0.0082640, -0.0074494, -0.0082640, -0.0074494, -0.0008146, 0.0008146
8: 0.0052524, 0.0098919, 0.0052524, 0.0098919, -0.0045616, 0.0045612
9: -0.0036845, -0.0030730, -0.0036845, -0.0030730, -0.0006115, 0.0006115

### Unstable ReLU Count (Linear/Conv2D Layers)
- layer_idx=0, type=LayerType.Linear, total=256, inp1_unstable=10, inp2_unstable=10, delta_unstable=249
- layer_idx=2, type=LayerType.Linear, total=256, inp1_unstable=8, inp2_unstable=8, delta_unstable=256
- layer_idx=4, type=LayerType.Linear, total=256, inp1_unstable=0, inp2_unstable=0, delta_unstable=256
- layer_idx=6, type=LayerType.Linear, total=256, inp1_unstable=2, inp2_unstable=2, delta_unstable=256
- layer_idx=8, type=LayerType.Linear, total=10, inp1_unstable=0, inp2_unstable=0, delta_unstable=10

Time for backsubstitution: 1.76 seconds

### RS candidates at layer 1
type: RSZ, layer: 1, pos: 199
type: RSZ, layer: 1, pos: 204
type: RSZ, layer: 1, pos: 250

Time for candidate selection: 0.22 seconds

### Candidate
type: RSZ, layer: 1, pos: 199

### Relational analysis RSZ of RS_RSZ1_RSZ2_RSZ1_RSZ1_RSZ2_RSZ2_RSZ1_RSZ1

#### Relational analysis RSZ result of RS_RSZ1_RSZ2_RSZ1_RSZ1_RSZ2_RSZ2_RSZ1_RSZ1
Status: Status.UNKNOWN
Output dim: 1, lower bound: -0.0007785, upper bound: 0.0007785
time: 0.64 seconds

### Relational analysis RSZ of RS_RSZ1_RSZ2_RSZ1_RSZ1_RSZ2_RSZ2_RSZ1_RSZ2

#### Relational analysis RSZ result of RS_RSZ1_RSZ2_RSZ1_RSZ1_RSZ2_RSZ2_RSZ1_RSZ2
Status: Status.UNKNOWN
Output dim: 1, lower bound: -0.0007785, upper bound: 0.0007785
time: 0.62 seconds

## BFS RS instance: RS_RSZ1_RSZ2_RSZ1_RSZ1_RSZ2_RSZ2_RSZ2

### Backsubstitution after applying RS history:
0: 0.0006996, 0.0011177, 0.0006996, 0.0011177, -0.0004181, 0.0004181
1: 0.9934013, 0.9943976, 0.9934013, 0.9943976, -0.0009964, 0.0009964
2: -0.0088595, -0.0053283, -0.0088595, -0.0053283, -0.0032413, 0.0032401
3: 0.0035633, 0.0041662, 0.0035633, 0.0041662, -0.0006029, 0.0006029
4: 0.0026282, 0.0054191, 0.0026282, 0.0054191, -0.0027909, 0.0027909
5: 0.0050201, 0.0064446, 0.0050201, 0.0064446, -0.0014245, 0.0014245
6: -0.0021963, -0.0008150, -0.0021963, -0.0008150, -0.0013813, 0.0013813
7: -0.0082640, -0.0074494, -0.0082640, -0.0074494, -0.0008146, 0.0008146
8: 0.0052524, 0.0098919, 0.0052524, 0.0098919, -0.0045618, 0.0045621
9: -0.0036845, -0.0030730, -0.0036845, -0.0030730, -0.0006115, 0.0006115

### Unstable ReLU Count (Linear/Conv2D Layers)
- layer_idx=0, type=LayerType.Linear, total=256, inp1_unstable=10, inp2_unstable=10, delta_unstable=249
- layer_idx=2, type=LayerType.Linear, total=256, inp1_unstable=8, inp2_unstable=8, delta_unstable=256
- layer_idx=4, type=LayerType.Linear, total=256, inp1_unstable=0, inp2_unstable=0, delta_unstable=256
- layer_idx=6, type=LayerType.Linear, total=256, inp1_unstable=2, inp2_unstable=2, delta_unstable=256
- layer_idx=8, type=LayerType.Linear, total=10, inp1_unstable=0, inp2_unstable=0, delta_unstable=10

Time for backsubstitution: 1.78 seconds

### RS candidates at layer 1
type: RSZ, layer: 1, pos: 199
type: RSZ, layer: 1, pos: 204
type: RSZ, layer: 1, pos: 250

Time for candidate selection: 0.21 seconds

### Candidate
type: RSZ, layer: 1, pos: 199

### Relational analysis RSZ of RS_RSZ1_RSZ2_RSZ1_RSZ1_RSZ2_RSZ2_RSZ2_RSZ1

#### Relational analysis RSZ result of RS_RSZ1_RSZ2_RSZ1_RSZ1_RSZ2_RSZ2_RSZ2_RSZ1
Status: Status.UNKNOWN
Output dim: 1, lower bound: -0.0007785, upper bound: 0.0007785
time: 0.62 seconds

### Relational analysis RSZ of RS_RSZ1_RSZ2_RSZ1_RSZ1_RSZ2_RSZ2_RSZ2_RSZ2

#### Relational analysis RSZ result of RS_RSZ1_RSZ2_RSZ1_RSZ1_RSZ2_RSZ2_RSZ2_RSZ2
Status: Status.UNKNOWN
Output dim: 1, lower bound: -0.0007785, upper bound: 0.0007785
time: 0.62 seconds

## BFS RS instance: RS_RSZ1_RSZ2_RSZ1_RSZ2_RSZ1_RSZ1_RSZ1

### Backsubstitution after applying RS history:
0: 0.0006996, 0.0011177, 0.0006996, 0.0011177, -0.0004181, 0.0004181
1: 0.9934013, 0.9943976, 0.9934013, 0.9943976, -0.0009964, 0.0009964
2: -0.0088595, -0.0053283, -0.0088595, -0.0053283, -0.0032373, 0.0032401
3: 0.0035633, 0.0041662, 0.0035633, 0.0041662, -0.0006029, 0.0006029
4: 0.0026282, 0.0054191, 0.0026282, 0.0054191, -0.0027909, 0.0027909
5: 0.0050201, 0.0064446, 0.0050201, 0.0064446, -0.0014245, 0.0014245
6: -0.0021963, -0.0008150, -0.0021963, -0.0008150, -0.0013813, 0.0013813
7: -0.0082640, -0.0074494, -0.0082640, -0.0074494, -0.0008146, 0.0008146
8: 0.0052524, 0.0098919, 0.0052524, 0.0098919, -0.0045618, 0.0045611
9: -0.0036845, -0.0030730, -0.0036845, -0.0030730, -0.0006115, 0.0006115

### Unstable ReLU Count (Linear/Conv2D Layers)
- layer_idx=0, type=LayerType.Linear, total=256, inp1_unstable=10, inp2_unstable=10, delta_unstable=249
- layer_idx=2, type=LayerType.Linear, total=256, inp1_unstable=8, inp2_unstable=8, delta_unstable=256
- layer_idx=4, type=LayerType.Linear, total=256, inp1_unstable=0, inp2_unstable=0, delta_unstable=256
- layer_idx=6, type=LayerType.Linear, total=256, inp1_unstable=2, inp2_unstable=2, delta_unstable=256
- layer_idx=8, type=LayerType.Linear, total=10, inp1_unstable=0, inp2_unstable=0, delta_unstable=10

Time for backsubstitution: 1.75 seconds

### RS candidates at layer 1
type: RSZ, layer: 1, pos: 199
type: RSZ, layer: 1, pos: 204
type: RSZ, layer: 1, pos: 250

Time for candidate selection: 0.19 seconds

### Candidate
type: RSZ, layer: 1, pos: 199

### Relational analysis RSZ of RS_RSZ1_RSZ2_RSZ1_RSZ2_RSZ1_RSZ1_RSZ1_RSZ1

#### Relational analysis RSZ result of RS_RSZ1_RSZ2_RSZ1_RSZ2_RSZ1_RSZ1_RSZ1_RSZ1
Status: Status.UNKNOWN
Output dim: 1, lower bound: -0.0007785, upper bound: 0.0007785
time: 0.63 seconds

### Relational analysis RSZ of RS_RSZ1_RSZ2_RSZ1_RSZ2_RSZ1_RSZ1_RSZ1_RSZ2

#### Relational analysis RSZ result of RS_RSZ1_RSZ2_RSZ1_RSZ2_RSZ1_RSZ1_RSZ1_RSZ2
Status: Status.UNKNOWN
Output dim: 1, lower bound: -0.0007785, upper bound: 0.0007785
time: 0.62 seconds

## BFS RS instance: RS_RSZ1_RSZ2_RSZ1_RSZ2_RSZ1_RSZ1_RSZ2

### Backsubstitution after applying RS history:
0: 0.0006996, 0.0011177, 0.0006996, 0.0011177, -0.0004181, 0.0004181
1: 0.9934013, 0.9943976, 0.9934013, 0.9943976, -0.0009964, 0.0009964
2: -0.0088595, -0.0053283, -0.0088595, -0.0053283, -0.0032407, 0.0032401
3: 0.0035633, 0.0041662, 0.0035633, 0.0041662, -0.0006029, 0.0006029
4: 0.0026282, 0.0054191, 0.0026282, 0.0054191, -0.0027909, 0.0027909
5: 0.0050201, 0.0064446, 0.0050201, 0.0064446, -0.0014245, 0.0014245
6: -0.0021963, -0.0008150, -0.0021963, -0.0008150, -0.0013813, 0.0013813
7: -0.0082640, -0.0074494, -0.0082640, -0.0074494, -0.0008146, 0.0008146
8: 0.0052524, 0.0098919, 0.0052524, 0.0098919, -0.0045618, 0.0045620
9: -0.0036845, -0.0030730, -0.0036845, -0.0030730, -0.0006115, 0.0006115

### Unstable ReLU Count (Linear/Conv2D Layers)
- layer_idx=0, type=LayerType.Linear, total=256, inp1_unstable=10, inp2_unstable=10, delta_unstable=249
- layer_idx=2, type=LayerType.Linear, total=256, inp1_unstable=8, inp2_unstable=8, delta_unstable=256
- layer_idx=4, type=LayerType.Linear, total=256, inp1_unstable=0, inp2_unstable=0, delta_unstable=256
- layer_idx=6, type=LayerType.Linear, total=256, inp1_unstable=2, inp2_unstable=2, delta_unstable=256
- layer_idx=8, type=LayerType.Linear, total=10, inp1_unstable=0, inp2_unstable=0, delta_unstable=10

Time for backsubstitution: 1.71 seconds

### RS candidates at layer 1
type: RSZ, layer: 1, pos: 199
type: RSZ, layer: 1, pos: 204
type: RSZ, layer: 1, pos: 250

Time for candidate selection: 0.19 seconds

### Candidate
type: RSZ, layer: 1, pos: 199

### Relational analysis RSZ of RS_RSZ1_RSZ2_RSZ1_RSZ2_RSZ1_RSZ1_RSZ2_RSZ1

#### Relational analysis RSZ result of RS_RSZ1_RSZ2_RSZ1_RSZ2_RSZ1_RSZ1_RSZ2_RSZ1
Status: Status.UNKNOWN
Output dim: 1, lower bound: -0.0007785, upper bound: 0.0007785
time: 0.64 seconds

### Relational analysis RSZ of RS_RSZ1_RSZ2_RSZ1_RSZ2_RSZ1_RSZ1_RSZ2_RSZ2

#### Relational analysis RSZ result of RS_RSZ1_RSZ2_RSZ1_RSZ2_RSZ1_RSZ1_RSZ2_RSZ2
Status: Status.UNKNOWN
Output dim: 1, lower bound: -0.0007785, upper bound: 0.0007785
time: 0.66 seconds

## BFS RS instance: RS_RSZ1_RSZ2_RSZ1_RSZ2_RSZ1_RSZ2_RSZ1

### Backsubstitution after applying RS history:
0: 0.0006996, 0.0011177, 0.0006996, 0.0011177, -0.0004181, 0.0004181
1: 0.9934013, 0.9943976, 0.9934013, 0.9943976, -0.0009964, 0.0009964
2: -0.0088595, -0.0053283, -0.0088595, -0.0053283, -0.0032388, 0.0032401
3: 0.0035633, 0.0041662, 0.0035633, 0.0041662, -0.0006029, 0.0006029
4: 0.0026282, 0.0054191, 0.0026282, 0.0054191, -0.0027909, 0.0027909
5: 0.0050201, 0.0064446, 0.0050201, 0.0064446, -0.0014245, 0.0014245
6: -0.0021963, -0.0008150, -0.0021963, -0.0008150, -0.0013813, 0.0013813
7: -0.0082640, -0.0074494, -0.0082640, -0.0074494, -0.0008146, 0.0008146
8: 0.0052524, 0.0098919, 0.0052524, 0.0098919, -0.0045618, 0.0045615
9: -0.0036845, -0.0030730, -0.0036845, -0.0030730, -0.0006115, 0.0006115

### Unstable ReLU Count (Linear/Conv2D Layers)
- layer_idx=0, type=LayerType.Linear, total=256, inp1_unstable=10, inp2_unstable=10, delta_unstable=249
- layer_idx=2, type=LayerType.Linear, total=256, inp1_unstable=8, inp2_unstable=8, delta_unstable=256
- layer_idx=4, type=LayerType.Linear, total=256, inp1_unstable=0, inp2_unstable=0, delta_unstable=256
- layer_idx=6, type=LayerType.Linear, total=256, inp1_unstable=2, inp2_unstable=2, delta_unstable=256
- layer_idx=8, type=LayerType.Linear, total=10, inp1_unstable=0, inp2_unstable=0, delta_unstable=10

Time for backsubstitution: 1.76 seconds

### RS candidates at layer 1
type: RSZ, layer: 1, pos: 199
type: RSZ, layer: 1, pos: 204
type: RSZ, layer: 1, pos: 250

Time for candidate selection: 0.18 seconds

### Candidate
type: RSZ, layer: 1, pos: 199

### Relational analysis RSZ of RS_RSZ1_RSZ2_RSZ1_RSZ2_RSZ1_RSZ2_RSZ1_RSZ1

#### Relational analysis RSZ result of RS_RSZ1_RSZ2_RSZ1_RSZ2_RSZ1_RSZ2_RSZ1_RSZ1
Status: Status.UNKNOWN
Output dim: 1, lower bound: -0.0007785, upper bound: 0.0007785
time: 0.62 seconds

### Relational analysis RSZ of RS_RSZ1_RSZ2_RSZ1_RSZ2_RSZ1_RSZ2_RSZ1_RSZ2

#### Relational analysis RSZ result of RS_RSZ1_RSZ2_RSZ1_RSZ2_RSZ1_RSZ2_RSZ1_RSZ2
Status: Status.UNKNOWN
Output dim: 1, lower bound: -0.0007785, upper bound: 0.0007785
time: 0.62 seconds

## BFS RS instance: RS_RSZ1_RSZ2_RSZ1_RSZ2_RSZ1_RSZ2_RSZ2

### Backsubstitution after applying RS history:
0: 0.0006996, 0.0011177, 0.0006996, 0.0011177, -0.0004181, 0.0004181
1: 0.9934013, 0.9943976, 0.9934013, 0.9943976, -0.0009964, 0.0009964
2: -0.0088595, -0.0053283, -0.0088595, -0.0053283, -0.0032422, 0.0032395
3: 0.0035633, 0.0041662, 0.0035633, 0.0041662, -0.0006029, 0.0006029
4: 0.0026282, 0.0054191, 0.0026282, 0.0054191, -0.0027909, 0.0027909
5: 0.0050201, 0.0064446, 0.0050201, 0.0064446, -0.0014245, 0.0014245
6: -0.0021963, -0.0008150, -0.0021963, -0.0008150, -0.0013813, 0.0013813
7: -0.0082640, -0.0074494, -0.0082640, -0.0074494, -0.0008146, 0.0008146
8: 0.0052524, 0.0098919, 0.0052524, 0.0098919, -0.0045616, 0.0045624
9: -0.0036845, -0.0030730, -0.0036845, -0.0030730, -0.0006115, 0.0006115

### Unstable ReLU Count (Linear/Conv2D Layers)
- layer_idx=0, type=LayerType.Linear, total=256, inp1_unstable=10, inp2_unstable=10, delta_unstable=249
- layer_idx=2, type=LayerType.Linear, total=256, inp1_unstable=8, inp2_unstable=8, delta_unstable=256
- layer_idx=4, type=LayerType.Linear, total=256, inp1_unstable=0, inp2_unstable=0, delta_unstable=256
- layer_idx=6, type=LayerType.Linear, total=256, inp1_unstable=2, inp2_unstable=2, delta_unstable=256
- layer_idx=8, type=LayerType.Linear, total=10, inp1_unstable=0, inp2_unstable=0, delta_unstable=10

Time for backsubstitution: 1.76 seconds

### RS candidates at layer 1
type: RSZ, layer: 1, pos: 199
type: RSZ, layer: 1, pos: 204
type: RSZ, layer: 1, pos: 250

Time for candidate selection: 0.20 seconds

### Candidate
type: RSZ, layer: 1, pos: 199

### Relational analysis RSZ of RS_RSZ1_RSZ2_RSZ1_RSZ2_RSZ1_RSZ2_RSZ2_RSZ1

#### Relational analysis RSZ result of RS_RSZ1_RSZ2_RSZ1_RSZ2_RSZ1_RSZ2_RSZ2_RSZ1
Status: Status.UNKNOWN
Output dim: 1, lower bound: -0.0007785, upper bound: 0.0007785
time: 0.67 seconds

### Relational analysis RSZ of RS_RSZ1_RSZ2_RSZ1_RSZ2_RSZ1_RSZ2_RSZ2_RSZ2

#### Relational analysis RSZ result of RS_RSZ1_RSZ2_RSZ1_RSZ2_RSZ1_RSZ2_RSZ2_RSZ2
Status: Status.UNKNOWN
Output dim: 1, lower bound: -0.0007785, upper bound: 0.0007785
time: 0.64 seconds

## BFS RS instance: RS_RSZ1_RSZ2_RSZ1_RSZ2_RSZ2_RSZ1_RSZ1

### Backsubstitution after applying RS history:
0: 0.0006996, 0.0011177, 0.0006996, 0.0011177, -0.0004181, 0.0004181
1: 0.9934013, 0.9943976, 0.9934013, 0.9943976, -0.0009964, 0.0009964
2: -0.0088595, -0.0053283, -0.0088595, -0.0053283, -0.0032352, 0.0032413
3: 0.0035633, 0.0041662, 0.0035633, 0.0041662, -0.0006029, 0.0006029
4: 0.0026282, 0.0054191, 0.0026282, 0.0054191, -0.0027909, 0.0027909
5: 0.0050201, 0.0064446, 0.0050201, 0.0064446, -0.0014245, 0.0014245
6: -0.0021963, -0.0008150, -0.0021963, -0.0008150, -0.0013813, 0.0013813
7: -0.0082640, -0.0074494, -0.0082640, -0.0074494, -0.0008146, 0.0008146
8: 0.0052524, 0.0098919, 0.0052524, 0.0098919, -0.0045621, 0.0045605
9: -0.0036845, -0.0030730, -0.0036845, -0.0030730, -0.0006115, 0.0006115

### Unstable ReLU Count (Linear/Conv2D Layers)
- layer_idx=0, type=LayerType.Linear, total=256, inp1_unstable=10, inp2_unstable=10, delta_unstable=249
- layer_idx=2, type=LayerType.Linear, total=256, inp1_unstable=8, inp2_unstable=8, delta_unstable=256
- layer_idx=4, type=LayerType.Linear, total=256, inp1_unstable=0, inp2_unstable=0, delta_unstable=256
- layer_idx=6, type=LayerType.Linear, total=256, inp1_unstable=2, inp2_unstable=2, delta_unstable=256
- layer_idx=8, type=LayerType.Linear, total=10, inp1_unstable=0, inp2_unstable=0, delta_unstable=10

Time for backsubstitution: 1.77 seconds

### RS candidates at layer 1
type: RSZ, layer: 1, pos: 199
type: RSZ, layer: 1, pos: 204
type: RSZ, layer: 1, pos: 250

Time for candidate selection: 0.20 seconds

### Candidate
type: RSZ, layer: 1, pos: 199

### Relational analysis RSZ of RS_RSZ1_RSZ2_RSZ1_RSZ2_RSZ2_RSZ1_RSZ1_RSZ1

#### Relational analysis RSZ result of RS_RSZ1_RSZ2_RSZ1_RSZ2_RSZ2_RSZ1_RSZ1_RSZ1
Status: Status.UNKNOWN
Output dim: 1, lower bound: -0.0007785, upper bound: 0.0007785
time: 0.62 seconds

### Relational analysis RSZ of RS_RSZ1_RSZ2_RSZ1_RSZ2_RSZ2_RSZ1_RSZ1_RSZ2

#### Relational analysis RSZ result of RS_RSZ1_RSZ2_RSZ1_RSZ2_RSZ2_RSZ1_RSZ1_RSZ2
Status: Status.UNKNOWN
Output dim: 1, lower bound: -0.0007785, upper bound: 0.0007785
time: 0.63 seconds

## BFS RS instance: RS_RSZ1_RSZ2_RSZ1_RSZ2_RSZ2_RSZ1_RSZ2

### Backsubstitution after applying RS history:
0: 0.0006996, 0.0011177, 0.0006996, 0.0011177, -0.0004181, 0.0004181
1: 0.9934013, 0.9943976, 0.9934013, 0.9943976, -0.0009964, 0.0009964
2: -0.0088595, -0.0053283, -0.0088595, -0.0053283, -0.0032386, 0.0032417
3: 0.0035633, 0.0041662, 0.0035633, 0.0041662, -0.0006029, 0.0006029
4: 0.0026282, 0.0054191, 0.0026282, 0.0054191, -0.0027909, 0.0027909
5: 0.0050201, 0.0064446, 0.0050201, 0.0064446, -0.0014245, 0.0014245
6: -0.0021963, -0.0008150, -0.0021963, -0.0008150, -0.0013813, 0.0013813
7: -0.0082640, -0.0074494, -0.0082640, -0.0074494, -0.0008146, 0.0008146
8: 0.0052524, 0.0098919, 0.0052524, 0.0098919, -0.0045623, 0.0045614
9: -0.0036845, -0.0030730, -0.0036845, -0.0030730, -0.0006115, 0.0006115

### Unstable ReLU Count (Linear/Conv2D Layers)
- layer_idx=0, type=LayerType.Linear, total=256, inp1_unstable=10, inp2_unstable=10, delta_unstable=249
- layer_idx=2, type=LayerType.Linear, total=256, inp1_unstable=8, inp2_unstable=8, delta_unstable=256
- layer_idx=4, type=LayerType.Linear, total=256, inp1_unstable=0, inp2_unstable=0, delta_unstable=256
- layer_idx=6, type=LayerType.Linear, total=256, inp1_unstable=2, inp2_unstable=2, delta_unstable=256
- layer_idx=8, type=LayerType.Linear, total=10, inp1_unstable=0, inp2_unstable=0, delta_unstable=10

Time for backsubstitution: 1.79 seconds

### RS candidates at layer 1
type: RSZ, layer: 1, pos: 199
type: RSZ, layer: 1, pos: 204
type: RSZ, layer: 1, pos: 250

Time for candidate selection: 0.19 seconds

### Candidate
type: RSZ, layer: 1, pos: 199

### Relational analysis RSZ of RS_RSZ1_RSZ2_RSZ1_RSZ2_RSZ2_RSZ1_RSZ2_RSZ1

#### Relational analysis RSZ result of RS_RSZ1_RSZ2_RSZ1_RSZ2_RSZ2_RSZ1_RSZ2_RSZ1
Status: Status.UNKNOWN
Output dim: 1, lower bound: -0.0007785, upper bound: 0.0007785
time: 0.62 seconds

### Relational analysis RSZ of RS_RSZ1_RSZ2_RSZ1_RSZ2_RSZ2_RSZ1_RSZ2_RSZ2

#### Relational analysis RSZ result of RS_RSZ1_RSZ2_RSZ1_RSZ2_RSZ2_RSZ1_RSZ2_RSZ2
Status: Status.UNKNOWN
Output dim: 1, lower bound: -0.0007785, upper bound: 0.0007785
time: 0.62 seconds

## BFS RS instance: RS_RSZ1_RSZ2_RSZ1_RSZ2_RSZ2_RSZ2_RSZ1

### Backsubstitution after applying RS history:
0: 0.0006996, 0.0011177, 0.0006996, 0.0011177, -0.0004181, 0.0004181
1: 0.9934013, 0.9943976, 0.9934013, 0.9943976, -0.0009964, 0.0009964
2: -0.0088595, -0.0053283, -0.0088595, -0.0053283, -0.0032365, 0.0032403
3: 0.0035633, 0.0041662, 0.0035633, 0.0041662, -0.0006029, 0.0006029
4: 0.0026282, 0.0054191, 0.0026282, 0.0054191, -0.0027909, 0.0027909
5: 0.0050201, 0.0064446, 0.0050201, 0.0064446, -0.0014245, 0.0014245
6: -0.0021963, -0.0008150, -0.0021963, -0.0008150, -0.0013813, 0.0013813
7: -0.0082640, -0.0074494, -0.0082640, -0.0074494, -0.0008146, 0.0008146
8: 0.0052524, 0.0098919, 0.0052524, 0.0098919, -0.0045619, 0.0045609
9: -0.0036845, -0.0030730, -0.0036845, -0.0030730, -0.0006115, 0.0006115

### Unstable ReLU Count (Linear/Conv2D Layers)
- layer_idx=0, type=LayerType.Linear, total=256, inp1_unstable=10, inp2_unstable=10, delta_unstable=249
- layer_idx=2, type=LayerType.Linear, total=256, inp1_unstable=8, inp2_unstable=8, delta_unstable=256
- layer_idx=4, type=LayerType.Linear, total=256, inp1_unstable=0, inp2_unstable=0, delta_unstable=256
- layer_idx=6, type=LayerType.Linear, total=256, inp1_unstable=2, inp2_unstable=2, delta_unstable=256
- layer_idx=8, type=LayerType.Linear, total=10, inp1_unstable=0, inp2_unstable=0, delta_unstable=10

Time for backsubstitution: 1.75 seconds

### RS candidates at layer 1
type: RSZ, layer: 1, pos: 199
type: RSZ, layer: 1, pos: 204
type: RSZ, layer: 1, pos: 250

Time for candidate selection: 0.20 seconds

### Candidate
type: RSZ, layer: 1, pos: 199

### Relational analysis RSZ of RS_RSZ1_RSZ2_RSZ1_RSZ2_RSZ2_RSZ2_RSZ1_RSZ1

#### Relational analysis RSZ result of RS_RSZ1_RSZ2_RSZ1_RSZ2_RSZ2_RSZ2_RSZ1_RSZ1
Status: Status.UNKNOWN
Output dim: 1, lower bound: -0.0007785, upper bound: 0.0007785
time: 0.62 seconds

### Relational analysis RSZ of RS_RSZ1_RSZ2_RSZ1_RSZ2_RSZ2_RSZ2_RSZ1_RSZ2

#### Relational analysis RSZ result of RS_RSZ1_RSZ2_RSZ1_RSZ2_RSZ2_RSZ2_RSZ1_RSZ2
Status: Status.UNKNOWN
Output dim: 1, lower bound: -0.0007785, upper bound: 0.0007785
time: 0.61 seconds

## BFS RS instance: RS_RSZ1_RSZ2_RSZ1_RSZ2_RSZ2_RSZ2_RSZ2

### Backsubstitution after applying RS history:
0: 0.0006996, 0.0011177, 0.0006996, 0.0011177, -0.0004181, 0.0004181
1: 0.9934013, 0.9943976, 0.9934013, 0.9943976, -0.0009964, 0.0009964
2: -0.0088595, -0.0053283, -0.0088595, -0.0053283, -0.0032399, 0.0032408
3: 0.0035633, 0.0041662, 0.0035633, 0.0041662, -0.0006029, 0.0006029
4: 0.0026282, 0.0054191, 0.0026282, 0.0054191, -0.0027909, 0.0027909
5: 0.0050201, 0.0064446, 0.0050201, 0.0064446, -0.0014245, 0.0014245
6: -0.0021963, -0.0008150, -0.0021963, -0.0008150, -0.0013813, 0.0013813
7: -0.0082640, -0.0074494, -0.0082640, -0.0074494, -0.0008146, 0.0008146
8: 0.0052524, 0.0098919, 0.0052524, 0.0098919, -0.0045620, 0.0045618
9: -0.0036845, -0.0030730, -0.0036845, -0.0030730, -0.0006115, 0.0006115

### Unstable ReLU Count (Linear/Conv2D Layers)
- layer_idx=0, type=LayerType.Linear, total=256, inp1_unstable=10, inp2_unstable=10, delta_unstable=249
- layer_idx=2, type=LayerType.Linear, total=256, inp1_unstable=8, inp2_unstable=8, delta_unstable=256
- layer_idx=4, type=LayerType.Linear, total=256, inp1_unstable=0, inp2_unstable=0, delta_unstable=256
- layer_idx=6, type=LayerType.Linear, total=256, inp1_unstable=2, inp2_unstable=2, delta_unstable=256
- layer_idx=8, type=LayerType.Linear, total=10, inp1_unstable=0, inp2_unstable=0, delta_unstable=10

Time for backsubstitution: 1.79 seconds

### RS candidates at layer 1
type: RSZ, layer: 1, pos: 199
type: RSZ, layer: 1, pos: 204
type: RSZ, layer: 1, pos: 250

Time for candidate selection: 0.20 seconds

### Candidate
type: RSZ, layer: 1, pos: 199

### Relational analysis RSZ of RS_RSZ1_RSZ2_RSZ1_RSZ2_RSZ2_RSZ2_RSZ2_RSZ1

#### Relational analysis RSZ result of RS_RSZ1_RSZ2_RSZ1_RSZ2_RSZ2_RSZ2_RSZ2_RSZ1
Status: Status.UNKNOWN
Output dim: 1, lower bound: -0.0007785, upper bound: 0.0007785
time: 0.64 seconds

### Relational analysis RSZ of RS_RSZ1_RSZ2_RSZ1_RSZ2_RSZ2_RSZ2_RSZ2_RSZ2

#### Relational analysis RSZ result of RS_RSZ1_RSZ2_RSZ1_RSZ2_RSZ2_RSZ2_RSZ2_RSZ2
Status: Status.UNKNOWN
Output dim: 1, lower bound: -0.0007785, upper bound: 0.0007785
time: 0.65 seconds

## BFS RS instance: RS_RSZ1_RSZ2_RSZ2_RSZ1_RSZ1_RSZ1_RSZ1

### Backsubstitution after applying RS history:
0: 0.0006996, 0.0011177, 0.0006996, 0.0011177, -0.0004181, 0.0004181
1: 0.9934013, 0.9943976, 0.9934013, 0.9943976, -0.0009964, 0.0009964
2: -0.0088595, -0.0053283, -0.0088595, -0.0053283, -0.0032385, 0.0032385
3: 0.0035633, 0.0041662, 0.0035633, 0.0041662, -0.0006029, 0.0006029
4: 0.0026282, 0.0054191, 0.0026282, 0.0054191, -0.0027909, 0.0027909
5: 0.0050201, 0.0064446, 0.0050201, 0.0064446, -0.0014245, 0.0014245
6: -0.0021963, -0.0008150, -0.0021963, -0.0008150, -0.0013813, 0.0013813
7: -0.0082640, -0.0074494, -0.0082640, -0.0074494, -0.0008146, 0.0008146
8: 0.0052524, 0.0098919, 0.0052524, 0.0098919, -0.0045614, 0.0045614
9: -0.0036845, -0.0030730, -0.0036845, -0.0030730, -0.0006115, 0.0006115

### Unstable ReLU Count (Linear/Conv2D Layers)
- layer_idx=0, type=LayerType.Linear, total=256, inp1_unstable=10, inp2_unstable=10, delta_unstable=249
- layer_idx=2, type=LayerType.Linear, total=256, inp1_unstable=8, inp2_unstable=8, delta_unstable=256
- layer_idx=4, type=LayerType.Linear, total=256, inp1_unstable=0, inp2_unstable=0, delta_unstable=256
- layer_idx=6, type=LayerType.Linear, total=256, inp1_unstable=2, inp2_unstable=2, delta_unstable=256
- layer_idx=8, type=LayerType.Linear, total=10, inp1_unstable=0, inp2_unstable=0, delta_unstable=10

Time for backsubstitution: 1.91 seconds

### RS candidates at layer 1
type: RSZ, layer: 1, pos: 199
type: RSZ, layer: 1, pos: 204
type: RSZ, layer: 1, pos: 250

Time for candidate selection: 0.21 seconds

### Candidate
type: RSZ, layer: 1, pos: 199

### Relational analysis RSZ of RS_RSZ1_RSZ2_RSZ2_RSZ1_RSZ1_RSZ1_RSZ1_RSZ1

#### Relational analysis RSZ result of RS_RSZ1_RSZ2_RSZ2_RSZ1_RSZ1_RSZ1_RSZ1_RSZ1
Status: Status.UNKNOWN
Output dim: 1, lower bound: -0.0007785, upper bound: 0.0007785
time: 0.59 seconds

### Relational analysis RSZ of RS_RSZ1_RSZ2_RSZ2_RSZ1_RSZ1_RSZ1_RSZ1_RSZ2

#### Relational analysis RSZ result of RS_RSZ1_RSZ2_RSZ2_RSZ1_RSZ1_RSZ1_RSZ1_RSZ2
Status: Status.UNKNOWN
Output dim: 1, lower bound: -0.0007785, upper bound: 0.0007785
time: 0.64 seconds

## BFS RS instance: RS_RSZ1_RSZ2_RSZ2_RSZ1_RSZ1_RSZ1_RSZ2

### Backsubstitution after applying RS history:
0: 0.0006996, 0.0011177, 0.0006996, 0.0011177, -0.0004181, 0.0004181
1: 0.9934013, 0.9943976, 0.9934013, 0.9943976, -0.0009964, 0.0009964
2: -0.0088595, -0.0053283, -0.0088595, -0.0053283, -0.0032419, 0.0032392
3: 0.0035633, 0.0041662, 0.0035633, 0.0041662, -0.0006029, 0.0006029
4: 0.0026282, 0.0054191, 0.0026282, 0.0054191, -0.0027909, 0.0027909
5: 0.0050201, 0.0064446, 0.0050201, 0.0064446, -0.0014245, 0.0014245
6: -0.0021963, -0.0008150, -0.0021963, -0.0008150, -0.0013813, 0.0013813
7: -0.0082640, -0.0074494, -0.0082640, -0.0074494, -0.0008146, 0.0008146
8: 0.0052524, 0.0098919, 0.0052524, 0.0098919, -0.0045616, 0.0045623
9: -0.0036845, -0.0030730, -0.0036845, -0.0030730, -0.0006115, 0.0006115

### Unstable ReLU Count (Linear/Conv2D Layers)
- layer_idx=0, type=LayerType.Linear, total=256, inp1_unstable=10, inp2_unstable=10, delta_unstable=249
- layer_idx=2, type=LayerType.Linear, total=256, inp1_unstable=8, inp2_unstable=8, delta_unstable=256
- layer_idx=4, type=LayerType.Linear, total=256, inp1_unstable=0, inp2_unstable=0, delta_unstable=256
- layer_idx=6, type=LayerType.Linear, total=256, inp1_unstable=2, inp2_unstable=2, delta_unstable=256
- layer_idx=8, type=LayerType.Linear, total=10, inp1_unstable=0, inp2_unstable=0, delta_unstable=10

Time for backsubstitution: 1.84 seconds

### RS candidates at layer 1
type: RSZ, layer: 1, pos: 199
type: RSZ, layer: 1, pos: 204
type: RSZ, layer: 1, pos: 250

Time for candidate selection: 0.23 seconds

### Candidate
type: RSZ, layer: 1, pos: 199

### Relational analysis RSZ of RS_RSZ1_RSZ2_RSZ2_RSZ1_RSZ1_RSZ1_RSZ2_RSZ1

#### Relational analysis RSZ result of RS_RSZ1_RSZ2_RSZ2_RSZ1_RSZ1_RSZ1_RSZ2_RSZ1
Status: Status.UNKNOWN
Output dim: 1, lower bound: -0.0007785, upper bound: 0.0007785
time: 0.64 seconds

### Relational analysis RSZ of RS_RSZ1_RSZ2_RSZ2_RSZ1_RSZ1_RSZ1_RSZ2_RSZ2

#### Relational analysis RSZ result of RS_RSZ1_RSZ2_RSZ2_RSZ1_RSZ1_RSZ1_RSZ2_RSZ2
Status: Status.UNKNOWN
Output dim: 1, lower bound: -0.0007785, upper bound: 0.0007785
time: 0.66 seconds

## BFS RS instance: RS_RSZ1_RSZ2_RSZ2_RSZ1_RSZ1_RSZ2_RSZ1

### Backsubstitution after applying RS history:
0: 0.0006996, 0.0011177, 0.0006996, 0.0011177, -0.0004181, 0.0004181
1: 0.9934013, 0.9943976, 0.9934013, 0.9943976, -0.0009964, 0.0009964
2: -0.0088595, -0.0053283, -0.0088595, -0.0053283, -0.0032399, 0.0032381
3: 0.0035633, 0.0041662, 0.0035633, 0.0041662, -0.0006029, 0.0006029
4: 0.0026282, 0.0054191, 0.0026282, 0.0054191, -0.0027909, 0.0027909
5: 0.0050201, 0.0064446, 0.0050201, 0.0064446, -0.0014245, 0.0014245
6: -0.0021963, -0.0008150, -0.0021963, -0.0008150, -0.0013813, 0.0013813
7: -0.0082640, -0.0074494, -0.0082640, -0.0074494, -0.0008146, 0.0008146
8: 0.0052524, 0.0098919, 0.0052524, 0.0098919, -0.0045613, 0.0045618
9: -0.0036845, -0.0030730, -0.0036845, -0.0030730, -0.0006115, 0.0006115

### Unstable ReLU Count (Linear/Conv2D Layers)
- layer_idx=0, type=LayerType.Linear, total=256, inp1_unstable=10, inp2_unstable=10, delta_unstable=249
- layer_idx=2, type=LayerType.Linear, total=256, inp1_unstable=8, inp2_unstable=8, delta_unstable=256
- layer_idx=4, type=LayerType.Linear, total=256, inp1_unstable=0, inp2_unstable=0, delta_unstable=256
- layer_idx=6, type=LayerType.Linear, total=256, inp1_unstable=2, inp2_unstable=2, delta_unstable=256
- layer_idx=8, type=LayerType.Linear, total=10, inp1_unstable=0, inp2_unstable=0, delta_unstable=10

Time for backsubstitution: 1.83 seconds

### RS candidates at layer 1
type: RSZ, layer: 1, pos: 199
type: RSZ, layer: 1, pos: 204
type: RSZ, layer: 1, pos: 250

Time for candidate selection: 0.20 seconds

### Candidate
type: RSZ, layer: 1, pos: 199

### Relational analysis RSZ of RS_RSZ1_RSZ2_RSZ2_RSZ1_RSZ1_RSZ2_RSZ1_RSZ1

#### Relational analysis RSZ result of RS_RSZ1_RSZ2_RSZ2_RSZ1_RSZ1_RSZ2_RSZ1_RSZ1
Status: Status.UNKNOWN
Output dim: 1, lower bound: -0.0007785, upper bound: 0.0007785
time: 0.64 seconds

### Relational analysis RSZ of RS_RSZ1_RSZ2_RSZ2_RSZ1_RSZ1_RSZ2_RSZ1_RSZ2

#### Relational analysis RSZ result of RS_RSZ1_RSZ2_RSZ2_RSZ1_RSZ1_RSZ2_RSZ1_RSZ2
Status: Status.UNKNOWN
Output dim: 1, lower bound: -0.0007785, upper bound: 0.0007785
time: 0.66 seconds

## BFS RS instance: RS_RSZ1_RSZ2_RSZ2_RSZ1_RSZ1_RSZ2_RSZ2

### Backsubstitution after applying RS history:
0: 0.0006996, 0.0011177, 0.0006996, 0.0011177, -0.0004181, 0.0004181
1: 0.9934013, 0.9943976, 0.9934013, 0.9943976, -0.0009964, 0.0009964
2: -0.0088595, -0.0053283, -0.0088595, -0.0053283, -0.0032433, 0.0032385
3: 0.0035633, 0.0041662, 0.0035633, 0.0041662, -0.0006029, 0.0006029
4: 0.0026282, 0.0054191, 0.0026282, 0.0054191, -0.0027909, 0.0027909
5: 0.0050201, 0.0064446, 0.0050201, 0.0064446, -0.0014245, 0.0014245
6: -0.0021963, -0.0008150, -0.0021963, -0.0008150, -0.0013813, 0.0013813
7: -0.0082640, -0.0074494, -0.0082640, -0.0074494, -0.0008146, 0.0008146
8: 0.0052524, 0.0098919, 0.0052524, 0.0098919, -0.0045614, 0.0045627
9: -0.0036845, -0.0030730, -0.0036845, -0.0030730, -0.0006115, 0.0006115

### Unstable ReLU Count (Linear/Conv2D Layers)
- layer_idx=0, type=LayerType.Linear, total=256, inp1_unstable=10, inp2_unstable=10, delta_unstable=249
- layer_idx=2, type=LayerType.Linear, total=256, inp1_unstable=8, inp2_unstable=8, delta_unstable=256
- layer_idx=4, type=LayerType.Linear, total=256, inp1_unstable=0, inp2_unstable=0, delta_unstable=256
- layer_idx=6, type=LayerType.Linear, total=256, inp1_unstable=2, inp2_unstable=2, delta_unstable=256
- layer_idx=8, type=LayerType.Linear, total=10, inp1_unstable=0, inp2_unstable=0, delta_unstable=10

Time for backsubstitution: 1.87 seconds

### RS candidates at layer 1
type: RSZ, layer: 1, pos: 199
type: RSZ, layer: 1, pos: 204
type: RSZ, layer: 1, pos: 250

Time for candidate selection: 0.19 seconds

### Candidate
type: RSZ, layer: 1, pos: 199

### Relational analysis RSZ of RS_RSZ1_RSZ2_RSZ2_RSZ1_RSZ1_RSZ2_RSZ2_RSZ1

#### Relational analysis RSZ result of RS_RSZ1_RSZ2_RSZ2_RSZ1_RSZ1_RSZ2_RSZ2_RSZ1
Status: Status.UNKNOWN
Output dim: 1, lower bound: -0.0007785, upper bound: 0.0007785
time: 0.61 seconds

### Relational analysis RSZ of RS_RSZ1_RSZ2_RSZ2_RSZ1_RSZ1_RSZ2_RSZ2_RSZ2

#### Relational analysis RSZ result of RS_RSZ1_RSZ2_RSZ2_RSZ1_RSZ1_RSZ2_RSZ2_RSZ2
Status: Status.UNKNOWN
Output dim: 1, lower bound: -0.0007785, upper bound: 0.0007785
time: 0.61 seconds

## BFS RS instance: RS_RSZ1_RSZ2_RSZ2_RSZ1_RSZ2_RSZ1_RSZ1

### Backsubstitution after applying RS history:
0: 0.0006996, 0.0011177, 0.0006996, 0.0011177, -0.0004181, 0.0004181
1: 0.9934013, 0.9943976, 0.9934013, 0.9943976, -0.0009964, 0.0009964
2: -0.0088595, -0.0053283, -0.0088595, -0.0053283, -0.0032365, 0.0032407
3: 0.0035633, 0.0041662, 0.0035633, 0.0041662, -0.0006029, 0.0006029
4: 0.0026282, 0.0054191, 0.0026282, 0.0054191, -0.0027909, 0.0027909
5: 0.0050201, 0.0064446, 0.0050201, 0.0064446, -0.0014245, 0.0014245
6: -0.0021963, -0.0008150, -0.0021963, -0.0008150, -0.0013813, 0.0013813
7: -0.0082640, -0.0074494, -0.0082640, -0.0074494, -0.0008146, 0.0008146
8: 0.0052524, 0.0098919, 0.0052524, 0.0098919, -0.0045620, 0.0045608
9: -0.0036845, -0.0030730, -0.0036845, -0.0030730, -0.0006115, 0.0006115

### Unstable ReLU Count (Linear/Conv2D Layers)
- layer_idx=0, type=LayerType.Linear, total=256, inp1_unstable=10, inp2_unstable=10, delta_unstable=249
- layer_idx=2, type=LayerType.Linear, total=256, inp1_unstable=8, inp2_unstable=8, delta_unstable=256
- layer_idx=4, type=LayerType.Linear, total=256, inp1_unstable=0, inp2_unstable=0, delta_unstable=256
- layer_idx=6, type=LayerType.Linear, total=256, inp1_unstable=2, inp2_unstable=2, delta_unstable=256
- layer_idx=8, type=LayerType.Linear, total=10, inp1_unstable=0, inp2_unstable=0, delta_unstable=10

Time for backsubstitution: 1.71 seconds

### RS candidates at layer 1
type: RSZ, layer: 1, pos: 199
type: RSZ, layer: 1, pos: 204
type: RSZ, layer: 1, pos: 250

Time for candidate selection: 0.19 seconds

### Candidate
type: RSZ, layer: 1, pos: 199

### Relational analysis RSZ of RS_RSZ1_RSZ2_RSZ2_RSZ1_RSZ2_RSZ1_RSZ1_RSZ1

#### Relational analysis RSZ result of RS_RSZ1_RSZ2_RSZ2_RSZ1_RSZ2_RSZ1_RSZ1_RSZ1
Status: Status.UNKNOWN
Output dim: 1, lower bound: -0.0007785, upper bound: 0.0007785
time: 0.68 seconds

### Relational analysis RSZ of RS_RSZ1_RSZ2_RSZ2_RSZ1_RSZ2_RSZ1_RSZ1_RSZ2

#### Relational analysis RSZ result of RS_RSZ1_RSZ2_RSZ2_RSZ1_RSZ2_RSZ1_RSZ1_RSZ2
Status: Status.UNKNOWN
Output dim: 1, lower bound: -0.0007785, upper bound: 0.0007785
time: 0.68 seconds

## BFS RS instance: RS_RSZ1_RSZ2_RSZ2_RSZ1_RSZ2_RSZ1_RSZ2

### Backsubstitution after applying RS history:
0: 0.0006996, 0.0011177, 0.0006996, 0.0011177, -0.0004181, 0.0004181
1: 0.9934013, 0.9943976, 0.9934013, 0.9943976, -0.0009964, 0.0009964
2: -0.0088595, -0.0053283, -0.0088595, -0.0053283, -0.0032398, 0.0032413
3: 0.0035633, 0.0041662, 0.0035633, 0.0041662, -0.0006029, 0.0006029
4: 0.0026282, 0.0054191, 0.0026282, 0.0054191, -0.0027909, 0.0027909
5: 0.0050201, 0.0064446, 0.0050201, 0.0064446, -0.0014245, 0.0014245
6: -0.0021963, -0.0008150, -0.0021963, -0.0008150, -0.0013813, 0.0013813
7: -0.0082640, -0.0074494, -0.0082640, -0.0074494, -0.0008146, 0.0008146
8: 0.0052524, 0.0098919, 0.0052524, 0.0098919, -0.0045621, 0.0045617
9: -0.0036845, -0.0030730, -0.0036845, -0.0030730, -0.0006115, 0.0006115

### Unstable ReLU Count (Linear/Conv2D Layers)
- layer_idx=0, type=LayerType.Linear, total=256, inp1_unstable=10, inp2_unstable=10, delta_unstable=249
- layer_idx=2, type=LayerType.Linear, total=256, inp1_unstable=8, inp2_unstable=8, delta_unstable=256
- layer_idx=4, type=LayerType.Linear, total=256, inp1_unstable=0, inp2_unstable=0, delta_unstable=256
- layer_idx=6, type=LayerType.Linear, total=256, inp1_unstable=2, inp2_unstable=2, delta_unstable=256
- layer_idx=8, type=LayerType.Linear, total=10, inp1_unstable=0, inp2_unstable=0, delta_unstable=10

Time for backsubstitution: 2.02 seconds

### RS candidates at layer 1
type: RSZ, layer: 1, pos: 199
type: RSZ, layer: 1, pos: 204
type: RSZ, layer: 1, pos: 250

Time for candidate selection: 0.22 seconds

### Candidate
type: RSZ, layer: 1, pos: 199

### Relational analysis RSZ of RS_RSZ1_RSZ2_RSZ2_RSZ1_RSZ2_RSZ1_RSZ2_RSZ1

#### Relational analysis RSZ result of RS_RSZ1_RSZ2_RSZ2_RSZ1_RSZ2_RSZ1_RSZ2_RSZ1
Status: Status.UNKNOWN
Output dim: 1, lower bound: -0.0007785, upper bound: 0.0007785
time: 0.64 seconds

### Relational analysis RSZ of RS_RSZ1_RSZ2_RSZ2_RSZ1_RSZ2_RSZ1_RSZ2_RSZ2

#### Relational analysis RSZ result of RS_RSZ1_RSZ2_RSZ2_RSZ1_RSZ2_RSZ1_RSZ2_RSZ2
Status: Status.UNKNOWN
Output dim: 1, lower bound: -0.0007785, upper bound: 0.0007785
time: 0.66 seconds

## BFS RS instance: RS_RSZ1_RSZ2_RSZ2_RSZ1_RSZ2_RSZ2_RSZ1

### Backsubstitution after applying RS history:
0: 0.0006996, 0.0011177, 0.0006996, 0.0011177, -0.0004181, 0.0004181
1: 0.9934013, 0.9943976, 0.9934013, 0.9943976, -0.0009964, 0.0009964
2: -0.0088595, -0.0053283, -0.0088595, -0.0053283, -0.0032379, 0.0032395
3: 0.0035633, 0.0041662, 0.0035633, 0.0041662, -0.0006029, 0.0006029
4: 0.0026282, 0.0054191, 0.0026282, 0.0054191, -0.0027909, 0.0027909
5: 0.0050201, 0.0064446, 0.0050201, 0.0064446, -0.0014245, 0.0014245
6: -0.0021963, -0.0008150, -0.0021963, -0.0008150, -0.0013813, 0.0013813
7: -0.0082640, -0.0074494, -0.0082640, -0.0074494, -0.0008146, 0.0008146
8: 0.0052524, 0.0098919, 0.0052524, 0.0098919, -0.0045617, 0.0045612
9: -0.0036845, -0.0030730, -0.0036845, -0.0030730, -0.0006115, 0.0006115

### Unstable ReLU Count (Linear/Conv2D Layers)
- layer_idx=0, type=LayerType.Linear, total=256, inp1_unstable=10, inp2_unstable=10, delta_unstable=249
- layer_idx=2, type=LayerType.Linear, total=256, inp1_unstable=8, inp2_unstable=8, delta_unstable=256
- layer_idx=4, type=LayerType.Linear, total=256, inp1_unstable=0, inp2_unstable=0, delta_unstable=256
- layer_idx=6, type=LayerType.Linear, total=256, inp1_unstable=2, inp2_unstable=2, delta_unstable=256
- layer_idx=8, type=LayerType.Linear, total=10, inp1_unstable=0, inp2_unstable=0, delta_unstable=10

Time for backsubstitution: 1.99 seconds

### RS candidates at layer 1
type: RSZ, layer: 1, pos: 199
type: RSZ, layer: 1, pos: 204
type: RSZ, layer: 1, pos: 250

Time for candidate selection: 0.23 seconds

### Candidate
type: RSZ, layer: 1, pos: 199

### Relational analysis RSZ of RS_RSZ1_RSZ2_RSZ2_RSZ1_RSZ2_RSZ2_RSZ1_RSZ1

#### Relational analysis RSZ result of RS_RSZ1_RSZ2_RSZ2_RSZ1_RSZ2_RSZ2_RSZ1_RSZ1
Status: Status.UNKNOWN
Output dim: 1, lower bound: -0.0007785, upper bound: 0.0007785
time: 0.66 seconds

### Relational analysis RSZ of RS_RSZ1_RSZ2_RSZ2_RSZ1_RSZ2_RSZ2_RSZ1_RSZ2

#### Relational analysis RSZ result of RS_RSZ1_RSZ2_RSZ2_RSZ1_RSZ2_RSZ2_RSZ1_RSZ2
Status: Status.UNKNOWN
Output dim: 1, lower bound: -0.0007785, upper bound: 0.0007785
time: 0.67 seconds

## BFS RS instance: RS_RSZ1_RSZ2_RSZ2_RSZ1_RSZ2_RSZ2_RSZ2

### Backsubstitution after applying RS history:
0: 0.0006996, 0.0011177, 0.0006996, 0.0011177, -0.0004181, 0.0004181
1: 0.9934013, 0.9943976, 0.9934013, 0.9943976, -0.0009964, 0.0009964
2: -0.0088595, -0.0053283, -0.0088595, -0.0053283, -0.0032412, 0.0032404
3: 0.0035633, 0.0041662, 0.0035633, 0.0041662, -0.0006029, 0.0006029
4: 0.0026282, 0.0054191, 0.0026282, 0.0054191, -0.0027909, 0.0027909
5: 0.0050201, 0.0064446, 0.0050201, 0.0064446, -0.0014245, 0.0014245
6: -0.0021963, -0.0008150, -0.0021963, -0.0008150, -0.0013813, 0.0013813
7: -0.0082640, -0.0074494, -0.0082640, -0.0074494, -0.0008146, 0.0008146
8: 0.0052524, 0.0098919, 0.0052524, 0.0098919, -0.0045619, 0.0045621
9: -0.0036845, -0.0030730, -0.0036845, -0.0030730, -0.0006115, 0.0006115

### Unstable ReLU Count (Linear/Conv2D Layers)
- layer_idx=0, type=LayerType.Linear, total=256, inp1_unstable=10, inp2_unstable=10, delta_unstable=249
- layer_idx=2, type=LayerType.Linear, total=256, inp1_unstable=8, inp2_unstable=8, delta_unstable=256
- layer_idx=4, type=LayerType.Linear, total=256, inp1_unstable=0, inp2_unstable=0, delta_unstable=256
- layer_idx=6, type=LayerType.Linear, total=256, inp1_unstable=2, inp2_unstable=2, delta_unstable=256
- layer_idx=8, type=LayerType.Linear, total=10, inp1_unstable=0, inp2_unstable=0, delta_unstable=10

Time for backsubstitution: 1.95 seconds

### RS candidates at layer 1
type: RSZ, layer: 1, pos: 199
type: RSZ, layer: 1, pos: 204
type: RSZ, layer: 1, pos: 250

Time for candidate selection: 0.22 seconds

### Candidate
type: RSZ, layer: 1, pos: 199

### Relational analysis RSZ of RS_RSZ1_RSZ2_RSZ2_RSZ1_RSZ2_RSZ2_RSZ2_RSZ1

#### Relational analysis RSZ result of RS_RSZ1_RSZ2_RSZ2_RSZ1_RSZ2_RSZ2_RSZ2_RSZ1
Status: Status.UNKNOWN
Output dim: 1, lower bound: -0.0007785, upper bound: 0.0007785
time: 0.65 seconds

### Relational analysis RSZ of RS_RSZ1_RSZ2_RSZ2_RSZ1_RSZ2_RSZ2_RSZ2_RSZ2

#### Relational analysis RSZ result of RS_RSZ1_RSZ2_RSZ2_RSZ1_RSZ2_RSZ2_RSZ2_RSZ2
Status: Status.UNKNOWN
Output dim: 1, lower bound: -0.0007785, upper bound: 0.0007785
time: 0.68 seconds

## BFS RS instance: RS_RSZ1_RSZ2_RSZ2_RSZ2_RSZ1_RSZ1_RSZ1

### Backsubstitution after applying RS history:
0: 0.0006996, 0.0011177, 0.0006996, 0.0011177, -0.0004181, 0.0004181
1: 0.9934013, 0.9943976, 0.9934013, 0.9943976, -0.0009964, 0.0009964
2: -0.0088595, -0.0053283, -0.0088595, -0.0053283, -0.0032371, 0.0032398
3: 0.0035633, 0.0041662, 0.0035633, 0.0041662, -0.0006029, 0.0006029
4: 0.0026282, 0.0054191, 0.0026282, 0.0054191, -0.0027909, 0.0027909
5: 0.0050201, 0.0064446, 0.0050201, 0.0064446, -0.0014245, 0.0014245
6: -0.0021963, -0.0008150, -0.0021963, -0.0008150, -0.0013813, 0.0013813
7: -0.0082640, -0.0074494, -0.0082640, -0.0074494, -0.0008146, 0.0008146
8: 0.0052524, 0.0098919, 0.0052524, 0.0098919, -0.0045617, 0.0045610
9: -0.0036845, -0.0030730, -0.0036845, -0.0030730, -0.0006115, 0.0006115

### Unstable ReLU Count (Linear/Conv2D Layers)
- layer_idx=0, type=LayerType.Linear, total=256, inp1_unstable=10, inp2_unstable=10, delta_unstable=249
- layer_idx=2, type=LayerType.Linear, total=256, inp1_unstable=8, inp2_unstable=8, delta_unstable=256
- layer_idx=4, type=LayerType.Linear, total=256, inp1_unstable=0, inp2_unstable=0, delta_unstable=256
- layer_idx=6, type=LayerType.Linear, total=256, inp1_unstable=2, inp2_unstable=2, delta_unstable=256
- layer_idx=8, type=LayerType.Linear, total=10, inp1_unstable=0, inp2_unstable=0, delta_unstable=10

Time for backsubstitution: 1.95 seconds

### RS candidates at layer 1
type: RSZ, layer: 1, pos: 199
type: RSZ, layer: 1, pos: 204
type: RSZ, layer: 1, pos: 250

Time for candidate selection: 0.22 seconds

### Candidate
type: RSZ, layer: 1, pos: 199

### Relational analysis RSZ of RS_RSZ1_RSZ2_RSZ2_RSZ2_RSZ1_RSZ1_RSZ1_RSZ1

#### Relational analysis RSZ result of RS_RSZ1_RSZ2_RSZ2_RSZ2_RSZ1_RSZ1_RSZ1_RSZ1
Status: Status.UNKNOWN
Output dim: 1, lower bound: -0.0007785, upper bound: 0.0007785
time: 0.65 seconds

### Relational analysis RSZ of RS_RSZ1_RSZ2_RSZ2_RSZ2_RSZ1_RSZ1_RSZ1_RSZ2

#### Relational analysis RSZ result of RS_RSZ1_RSZ2_RSZ2_RSZ2_RSZ1_RSZ1_RSZ1_RSZ2
Status: Status.UNKNOWN
Output dim: 1, lower bound: -0.0007785, upper bound: 0.0007785
time: 0.65 seconds

## BFS RS instance: RS_RSZ1_RSZ2_RSZ2_RSZ2_RSZ1_RSZ1_RSZ2

### Backsubstitution after applying RS history:
0: 0.0006996, 0.0011177, 0.0006996, 0.0011177, -0.0004181, 0.0004181
1: 0.9934013, 0.9943976, 0.9934013, 0.9943976, -0.0009964, 0.0009964
2: -0.0088595, -0.0053283, -0.0088595, -0.0053283, -0.0032405, 0.0032401
3: 0.0035633, 0.0041662, 0.0035633, 0.0041662, -0.0006029, 0.0006029
4: 0.0026282, 0.0054191, 0.0026282, 0.0054191, -0.0027909, 0.0027909
5: 0.0050201, 0.0064446, 0.0050201, 0.0064446, -0.0014245, 0.0014245
6: -0.0021963, -0.0008150, -0.0021963, -0.0008150, -0.0013813, 0.0013813
7: -0.0082640, -0.0074494, -0.0082640, -0.0074494, -0.0008146, 0.0008146
8: 0.0052524, 0.0098919, 0.0052524, 0.0098919, -0.0045618, 0.0045619
9: -0.0036845, -0.0030730, -0.0036845, -0.0030730, -0.0006115, 0.0006115

### Unstable ReLU Count (Linear/Conv2D Layers)
- layer_idx=0, type=LayerType.Linear, total=256, inp1_unstable=10, inp2_unstable=10, delta_unstable=249
- layer_idx=2, type=LayerType.Linear, total=256, inp1_unstable=8, inp2_unstable=8, delta_unstable=256
- layer_idx=4, type=LayerType.Linear, total=256, inp1_unstable=0, inp2_unstable=0, delta_unstable=256
- layer_idx=6, type=LayerType.Linear, total=256, inp1_unstable=2, inp2_unstable=2, delta_unstable=256
- layer_idx=8, type=LayerType.Linear, total=10, inp1_unstable=0, inp2_unstable=0, delta_unstable=10

Time for backsubstitution: 1.85 seconds

### RS candidates at layer 1
type: RSZ, layer: 1, pos: 199
type: RSZ, layer: 1, pos: 204
type: RSZ, layer: 1, pos: 250

Time for candidate selection: 0.20 seconds

### Candidate
type: RSZ, layer: 1, pos: 199

### Relational analysis RSZ of RS_RSZ1_RSZ2_RSZ2_RSZ2_RSZ1_RSZ1_RSZ2_RSZ1

#### Relational analysis RSZ result of RS_RSZ1_RSZ2_RSZ2_RSZ2_RSZ1_RSZ1_RSZ2_RSZ1
Status: Status.UNKNOWN
Output dim: 1, lower bound: -0.0007785, upper bound: 0.0007785
time: 0.63 seconds

### Relational analysis RSZ of RS_RSZ1_RSZ2_RSZ2_RSZ2_RSZ1_RSZ1_RSZ2_RSZ2

#### Relational analysis RSZ result of RS_RSZ1_RSZ2_RSZ2_RSZ2_RSZ1_RSZ1_RSZ2_RSZ2
Status: Status.UNKNOWN
Output dim: 1, lower bound: -0.0007785, upper bound: 0.0007785
time: 0.65 seconds

## BFS RS instance: RS_RSZ1_RSZ2_RSZ2_RSZ2_RSZ1_RSZ2_RSZ1

### Backsubstitution after applying RS history:
0: 0.0006996, 0.0011177, 0.0006996, 0.0011177, -0.0004181, 0.0004181
1: 0.9934013, 0.9943976, 0.9934013, 0.9943976, -0.0009964, 0.0009964
2: -0.0088595, -0.0053283, -0.0088595, -0.0053283, -0.0032385, 0.0032396
3: 0.0035633, 0.0041662, 0.0035633, 0.0041662, -0.0006029, 0.0006029
4: 0.0026282, 0.0054191, 0.0026282, 0.0054191, -0.0027909, 0.0027909
5: 0.0050201, 0.0064446, 0.0050201, 0.0064446, -0.0014245, 0.0014245
6: -0.0021963, -0.0008150, -0.0021963, -0.0008150, -0.0013813, 0.0013813
7: -0.0082640, -0.0074494, -0.0082640, -0.0074494, -0.0008146, 0.0008146
8: 0.0052524, 0.0098919, 0.0052524, 0.0098919, -0.0045617, 0.0045614
9: -0.0036845, -0.0030730, -0.0036845, -0.0030730, -0.0006115, 0.0006115

### Unstable ReLU Count (Linear/Conv2D Layers)
- layer_idx=0, type=LayerType.Linear, total=256, inp1_unstable=10, inp2_unstable=10, delta_unstable=249
- layer_idx=2, type=LayerType.Linear, total=256, inp1_unstable=8, inp2_unstable=8, delta_unstable=256
- layer_idx=4, type=LayerType.Linear, total=256, inp1_unstable=0, inp2_unstable=0, delta_unstable=256
- layer_idx=6, type=LayerType.Linear, total=256, inp1_unstable=2, inp2_unstable=2, delta_unstable=256
- layer_idx=8, type=LayerType.Linear, total=10, inp1_unstable=0, inp2_unstable=0, delta_unstable=10

Time for backsubstitution: 1.79 seconds

### RS candidates at layer 1
type: RSZ, layer: 1, pos: 199
type: RSZ, layer: 1, pos: 204
type: RSZ, layer: 1, pos: 250

Time for candidate selection: 0.20 seconds

### Candidate
type: RSZ, layer: 1, pos: 199

### Relational analysis RSZ of RS_RSZ1_RSZ2_RSZ2_RSZ2_RSZ1_RSZ2_RSZ1_RSZ1

#### Relational analysis RSZ result of RS_RSZ1_RSZ2_RSZ2_RSZ2_RSZ1_RSZ2_RSZ1_RSZ1
Status: Status.UNKNOWN
Output dim: 1, lower bound: -0.0007785, upper bound: 0.0007785
time: 0.67 seconds

### Relational analysis RSZ of RS_RSZ1_RSZ2_RSZ2_RSZ2_RSZ1_RSZ2_RSZ1_RSZ2

#### Relational analysis RSZ result of RS_RSZ1_RSZ2_RSZ2_RSZ2_RSZ1_RSZ2_RSZ1_RSZ2
Status: Status.UNKNOWN
Output dim: 1, lower bound: -0.0007785, upper bound: 0.0007785
time: 0.63 seconds

## BFS RS instance: RS_RSZ1_RSZ2_RSZ2_RSZ2_RSZ1_RSZ2_RSZ2

### Backsubstitution after applying RS history:
0: 0.0006996, 0.0011177, 0.0006996, 0.0011177, -0.0004181, 0.0004181
1: 0.9934013, 0.9943976, 0.9934013, 0.9943976, -0.0009964, 0.0009964
2: -0.0088595, -0.0053283, -0.0088595, -0.0053283, -0.0032418, 0.0032395
3: 0.0035633, 0.0041662, 0.0035633, 0.0041662, -0.0006029, 0.0006029
4: 0.0026282, 0.0054191, 0.0026282, 0.0054191, -0.0027909, 0.0027909
5: 0.0050201, 0.0064446, 0.0050201, 0.0064446, -0.0014245, 0.0014245
6: -0.0021963, -0.0008150, -0.0021963, -0.0008150, -0.0013813, 0.0013813
7: -0.0082640, -0.0074494, -0.0082640, -0.0074494, -0.0008146, 0.0008146
8: 0.0052524, 0.0098919, 0.0052524, 0.0098919, -0.0045617, 0.0045623
9: -0.0036845, -0.0030730, -0.0036845, -0.0030730, -0.0006115, 0.0006115

### Unstable ReLU Count (Linear/Conv2D Layers)
- layer_idx=0, type=LayerType.Linear, total=256, inp1_unstable=10, inp2_unstable=10, delta_unstable=249
- layer_idx=2, type=LayerType.Linear, total=256, inp1_unstable=8, inp2_unstable=8, delta_unstable=256
- layer_idx=4, type=LayerType.Linear, total=256, inp1_unstable=0, inp2_unstable=0, delta_unstable=256
- layer_idx=6, type=LayerType.Linear, total=256, inp1_unstable=2, inp2_unstable=2, delta_unstable=256
- layer_idx=8, type=LayerType.Linear, total=10, inp1_unstable=0, inp2_unstable=0, delta_unstable=10

Time for backsubstitution: 1.84 seconds

### RS candidates at layer 1
type: RSZ, layer: 1, pos: 199
type: RSZ, layer: 1, pos: 204
type: RSZ, layer: 1, pos: 250

Time for candidate selection: 0.21 seconds

### Candidate
type: RSZ, layer: 1, pos: 199

### Relational analysis RSZ of RS_RSZ1_RSZ2_RSZ2_RSZ2_RSZ1_RSZ2_RSZ2_RSZ1

#### Relational analysis RSZ result of RS_RSZ1_RSZ2_RSZ2_RSZ2_RSZ1_RSZ2_RSZ2_RSZ1
Status: Status.UNKNOWN
Output dim: 1, lower bound: -0.0007785, upper bound: 0.0007785
time: 0.64 seconds

### Relational analysis RSZ of RS_RSZ1_RSZ2_RSZ2_RSZ2_RSZ1_RSZ2_RSZ2_RSZ2

#### Relational analysis RSZ result of RS_RSZ1_RSZ2_RSZ2_RSZ2_RSZ1_RSZ2_RSZ2_RSZ2
Status: Status.UNKNOWN
Output dim: 1, lower bound: -0.0007785, upper bound: 0.0007785
time: 0.64 seconds

## BFS RS instance: RS_RSZ1_RSZ2_RSZ2_RSZ2_RSZ2_RSZ1_RSZ1

### Backsubstitution after applying RS history:
0: 0.0006996, 0.0011177, 0.0006996, 0.0011177, -0.0004181, 0.0004181
1: 0.9934013, 0.9943976, 0.9934013, 0.9943976, -0.0009964, 0.0009964
2: -0.0088595, -0.0053283, -0.0088595, -0.0053283, -0.0032352, 0.0032411
3: 0.0035633, 0.0041662, 0.0035633, 0.0041662, -0.0006029, 0.0006029
4: 0.0026282, 0.0054191, 0.0026282, 0.0054191, -0.0027909, 0.0027909
5: 0.0050201, 0.0064446, 0.0050201, 0.0064446, -0.0014245, 0.0014245
6: -0.0021963, -0.0008150, -0.0021963, -0.0008150, -0.0013813, 0.0013813
7: -0.0082640, -0.0074494, -0.0082640, -0.0074494, -0.0008146, 0.0008146
8: 0.0052524, 0.0098919, 0.0052524, 0.0098919, -0.0045621, 0.0045605
9: -0.0036845, -0.0030730, -0.0036845, -0.0030730, -0.0006115, 0.0006115

### Unstable ReLU Count (Linear/Conv2D Layers)
- layer_idx=0, type=LayerType.Linear, total=256, inp1_unstable=10, inp2_unstable=10, delta_unstable=249
- layer_idx=2, type=LayerType.Linear, total=256, inp1_unstable=8, inp2_unstable=8, delta_unstable=256
- layer_idx=4, type=LayerType.Linear, total=256, inp1_unstable=0, inp2_unstable=0, delta_unstable=256
- layer_idx=6, type=LayerType.Linear, total=256, inp1_unstable=2, inp2_unstable=2, delta_unstable=256
- layer_idx=8, type=LayerType.Linear, total=10, inp1_unstable=0, inp2_unstable=0, delta_unstable=10

Time for backsubstitution: 1.85 seconds

### RS candidates at layer 1
type: RSZ, layer: 1, pos: 199
type: RSZ, layer: 1, pos: 204
type: RSZ, layer: 1, pos: 250

Time for candidate selection: 0.22 seconds

## RS Result
status: Status.UNKNOWN
execution time: (base) + (rs) = 3.22 + 596.90 = 600.12 seconds
