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
- layer_idx=0, type=LayerType.Linear, total=256, inp1_unstable=9, inp2_unstable=9, delta_unstable=256
- layer_idx=2, type=LayerType.Linear, total=256, inp1_unstable=6, inp2_unstable=6, delta_unstable=256
- layer_idx=4, type=LayerType.Linear, total=256, inp1_unstable=0, inp2_unstable=0, delta_unstable=256
- layer_idx=6, type=LayerType.Linear, total=256, inp1_unstable=2, inp2_unstable=2, delta_unstable=256
- layer_idx=8, type=LayerType.Linear, total=10, inp1_unstable=0, inp2_unstable=0, delta_unstable=10

### BASE IAR bounds
Layer (inp1_lb, inp1_ub, inp2_lb, inp2_ub, d_lb, d_ub)
0: (0.0007184, 0.0011172, 0.0007184, 0.0011172, -0.0003988, 0.0003988)
1: (0.9934075, 0.9943579, 0.9934075, 0.9943579, -0.0009505, 0.0009505)
2: (-0.0087946, -0.0055215, -0.0087946, -0.0055215, -0.0030070, 0.0030070)
3: (0.0035868, 0.0041619, 0.0035868, 0.0041619, -0.0005751, 0.0005751)
4: (0.0027809, 0.0053678, 0.0027809, 0.0053678, -0.0025869, 0.0025869)
5: (0.0050646, 0.0064029, 0.0050646, 0.0064029, -0.0013382, 0.0013382)
6: (-0.0021738, -0.0008562, -0.0021738, -0.0008562, -0.0013176, 0.0013176)
7: (-0.0082275, -0.0074685, -0.0082275, -0.0074685, -0.0007590, 0.0007590)
8: (0.0055062, 0.0098066, 0.0055062, 0.0098066, -0.0042280, 0.0042280)
9: (-0.0036825, -0.0031002, -0.0036825, -0.0031002, -0.0005823, 0.0005823)

## BASE Relational Analysis

## BASE Result
execution time: IAR + RelationalAnalysis = 1.74 + 1.48 = 3.22 seconds
status: Status.UNKNOWN
relational distance
Output dim: 1, lower bound: -0.0008230, upper bound: 0.0008230

# Relational Split (RS) starts

## BFS RS instance: RS

Time for backsubstitution: 0.01 seconds

### RS candidates at layer 1
type: RSZ, layer: 1, pos: 78
type: RSZ, layer: 1, pos: 83
type: RSZ, layer: 1, pos: 204
type: RSZ, layer: 1, pos: 68
type: RSZ, layer: 1, pos: 52
type: RSZ, layer: 1, pos: 198
type: RSZ, layer: 1, pos: 136
type: RSZ, layer: 1, pos: 199
type: RSZ, layer: 1, pos: 176

Time for candidate selection: 0.00 seconds

### Candidate
type: RSZ, layer: 1, pos: 78

### Relational analysis RSZ of RS_RSZ1

#### Relational analysis RSZ result of RS_RSZ1
Status: Status.UNKNOWN
Output dim: 1, lower bound: -0.0008217, upper bound: 0.0008217
time: 0.58 seconds

### Relational analysis RSZ of RS_RSZ2

#### Relational analysis RSZ result of RS_RSZ2
Status: Status.UNKNOWN
Output dim: 1, lower bound: -0.0008217, upper bound: 0.0008218
time: 0.61 seconds

## Summary of splitting (split count: 0)
- Time for RS candidates: 1.20 seconds
RS_RSZ1, status: Status.UNKNOWN, split count: 1, time: 1.20
Output dim: 1, lower bound: -0.0008217, upper bound: 0.0008217
RS_RSZ2, status: Status.UNKNOWN, split count: 1, time: 1.20
Output dim: 1, lower bound: -0.0008217, upper bound: 0.0008218

## BFS RS instance: RS_RSZ1

### Backsubstitution after applying RS history:
0: 0.0007184, 0.0011172, 0.0007184, 0.0011172, -0.0003988, 0.0003988
1: 0.9934075, 0.9943579, 0.9934075, 0.9943579, -0.0009505, 0.0009505
2: -0.0087946, -0.0055215, -0.0087946, -0.0055215, -0.0030044, 0.0030045
3: 0.0035868, 0.0041619, 0.0035868, 0.0041619, -0.0005751, 0.0005751
4: 0.0027809, 0.0053678, 0.0027809, 0.0053678, -0.0025869, 0.0025869
5: 0.0050646, 0.0064029, 0.0050646, 0.0064029, -0.0013382, 0.0013382
6: -0.0021738, -0.0008562, -0.0021738, -0.0008562, -0.0013176, 0.0013176
7: -0.0082275, -0.0074685, -0.0082275, -0.0074685, -0.0007590, 0.0007590
8: 0.0055062, 0.0098066, 0.0055062, 0.0098066, -0.0042275, 0.0042275
9: -0.0036825, -0.0031002, -0.0036825, -0.0031002, -0.0005823, 0.0005823

### Unstable ReLU Count (Linear/Conv2D Layers)
- layer_idx=0, type=LayerType.Linear, total=256, inp1_unstable=9, inp2_unstable=9, delta_unstable=255
- layer_idx=2, type=LayerType.Linear, total=256, inp1_unstable=6, inp2_unstable=6, delta_unstable=256
- layer_idx=4, type=LayerType.Linear, total=256, inp1_unstable=0, inp2_unstable=0, delta_unstable=256
- layer_idx=6, type=LayerType.Linear, total=256, inp1_unstable=2, inp2_unstable=2, delta_unstable=256
- layer_idx=8, type=LayerType.Linear, total=10, inp1_unstable=0, inp2_unstable=0, delta_unstable=10

Time for backsubstitution: 1.83 seconds

### RS candidates at layer 1
type: RSZ, layer: 1, pos: 176
type: RSZ, layer: 1, pos: 68
type: RSZ, layer: 1, pos: 204
type: RSZ, layer: 1, pos: 198
type: RSZ, layer: 1, pos: 83
type: RSZ, layer: 1, pos: 52
type: RSZ, layer: 1, pos: 199
type: RSZ, layer: 1, pos: 136

Time for candidate selection: 0.00 seconds

### Candidate
type: RSZ, layer: 1, pos: 176

### Relational analysis RSZ of RS_RSZ1_RSZ1

#### Relational analysis RSZ result of RS_RSZ1_RSZ1
Status: Status.UNKNOWN
Output dim: 1, lower bound: -0.0008137, upper bound: 0.0008137
time: 0.62 seconds

### Relational analysis RSZ of RS_RSZ1_RSZ2

#### Relational analysis RSZ result of RS_RSZ1_RSZ2
Status: Status.UNKNOWN
Output dim: 1, lower bound: -0.0008137, upper bound: 0.0008137
time: 0.63 seconds

## BFS RS instance: RS_RSZ2

### Backsubstitution after applying RS history:
0: 0.0007184, 0.0011172, 0.0007184, 0.0011172, -0.0003988, 0.0003988
1: 0.9934075, 0.9943579, 0.9934075, 0.9943579, -0.0009505, 0.0009505
2: -0.0087946, -0.0055215, -0.0087946, -0.0055215, -0.0030045, 0.0030044
3: 0.0035868, 0.0041619, 0.0035868, 0.0041619, -0.0005751, 0.0005751
4: 0.0027809, 0.0053678, 0.0027809, 0.0053678, -0.0025869, 0.0025869
5: 0.0050646, 0.0064029, 0.0050646, 0.0064029, -0.0013382, 0.0013382
6: -0.0021738, -0.0008562, -0.0021738, -0.0008562, -0.0013176, 0.0013176
7: -0.0082275, -0.0074685, -0.0082275, -0.0074685, -0.0007590, 0.0007590
8: 0.0055062, 0.0098066, 0.0055062, 0.0098066, -0.0042275, 0.0042275
9: -0.0036825, -0.0031002, -0.0036825, -0.0031002, -0.0005823, 0.0005823

### Unstable ReLU Count (Linear/Conv2D Layers)
- layer_idx=0, type=LayerType.Linear, total=256, inp1_unstable=9, inp2_unstable=9, delta_unstable=255
- layer_idx=2, type=LayerType.Linear, total=256, inp1_unstable=6, inp2_unstable=6, delta_unstable=256
- layer_idx=4, type=LayerType.Linear, total=256, inp1_unstable=0, inp2_unstable=0, delta_unstable=256
- layer_idx=6, type=LayerType.Linear, total=256, inp1_unstable=2, inp2_unstable=2, delta_unstable=256
- layer_idx=8, type=LayerType.Linear, total=10, inp1_unstable=0, inp2_unstable=0, delta_unstable=10

Time for backsubstitution: 1.61 seconds

### RS candidates at layer 1
type: RSZ, layer: 1, pos: 68
type: RSZ, layer: 1, pos: 52
type: RSZ, layer: 1, pos: 176
type: RSZ, layer: 1, pos: 83
type: RSZ, layer: 1, pos: 198
type: RSZ, layer: 1, pos: 199
type: RSZ, layer: 1, pos: 136
type: RSZ, layer: 1, pos: 204

Time for candidate selection: 0.00 seconds

### Candidate
type: RSZ, layer: 1, pos: 68

### Relational analysis RSZ of RS_RSZ2_RSZ1

#### Relational analysis RSZ result of RS_RSZ2_RSZ1
Status: Status.UNKNOWN
Output dim: 1, lower bound: -0.0008175, upper bound: 0.0008202
time: 0.61 seconds

### Relational analysis RSZ of RS_RSZ2_RSZ2

#### Relational analysis RSZ result of RS_RSZ2_RSZ2
Status: Status.UNKNOWN
Output dim: 1, lower bound: -0.0008201, upper bound: 0.0008178
time: 0.62 seconds

## Summary of splitting (split count: 1)
- Time for RS candidates: 2.85 seconds
RS_RSZ1_RSZ1, status: Status.UNKNOWN, split count: 2, time: 2.85
Output dim: 1, lower bound: -0.0008137, upper bound: 0.0008137
RS_RSZ1_RSZ2, status: Status.UNKNOWN, split count: 2, time: 2.85
Output dim: 1, lower bound: -0.0008137, upper bound: 0.0008137
RS_RSZ2_RSZ1, status: Status.UNKNOWN, split count: 2, time: 2.85
Output dim: 1, lower bound: -0.0008175, upper bound: 0.0008202
RS_RSZ2_RSZ2, status: Status.UNKNOWN, split count: 2, time: 2.85
Output dim: 1, lower bound: -0.0008201, upper bound: 0.0008178

## BFS RS instance: RS_RSZ1_RSZ1

### Backsubstitution after applying RS history:
0: 0.0007184, 0.0011172, 0.0007184, 0.0011172, -0.0003988, 0.0003988
1: 0.9934075, 0.9943579, 0.9934075, 0.9943579, -0.0009505, 0.0009505
2: -0.0087946, -0.0055215, -0.0087946, -0.0055215, -0.0030016, 0.0030011
3: 0.0035868, 0.0041619, 0.0035868, 0.0041619, -0.0005751, 0.0005751
4: 0.0027809, 0.0053678, 0.0027809, 0.0053678, -0.0025869, 0.0025869
5: 0.0050646, 0.0064029, 0.0050646, 0.0064029, -0.0013382, 0.0013382
6: -0.0021738, -0.0008562, -0.0021738, -0.0008562, -0.0013176, 0.0013176
7: -0.0082275, -0.0074685, -0.0082275, -0.0074685, -0.0007590, 0.0007590
8: 0.0055062, 0.0098066, 0.0055062, 0.0098066, -0.0042266, 0.0042268
9: -0.0036825, -0.0031002, -0.0036825, -0.0031002, -0.0005823, 0.0005823

### Unstable ReLU Count (Linear/Conv2D Layers)
- layer_idx=0, type=LayerType.Linear, total=256, inp1_unstable=9, inp2_unstable=9, delta_unstable=254
- layer_idx=2, type=LayerType.Linear, total=256, inp1_unstable=6, inp2_unstable=6, delta_unstable=256
- layer_idx=4, type=LayerType.Linear, total=256, inp1_unstable=0, inp2_unstable=0, delta_unstable=256
- layer_idx=6, type=LayerType.Linear, total=256, inp1_unstable=2, inp2_unstable=2, delta_unstable=256
- layer_idx=8, type=LayerType.Linear, total=10, inp1_unstable=0, inp2_unstable=0, delta_unstable=10

Time for backsubstitution: 1.64 seconds

### RS candidates at layer 1
type: RSZ, layer: 1, pos: 204
type: RSZ, layer: 1, pos: 199
type: RSZ, layer: 1, pos: 52
type: RSZ, layer: 1, pos: 83
type: RSZ, layer: 1, pos: 198
type: RSZ, layer: 1, pos: 68
type: RSZ, layer: 1, pos: 136

Time for candidate selection: 0.00 seconds

### Candidate
type: RSZ, layer: 1, pos: 204

### Relational analysis RSZ of RS_RSZ1_RSZ1_RSZ1

#### Relational analysis RSZ result of RS_RSZ1_RSZ1_RSZ1
Status: Status.UNKNOWN
Output dim: 1, lower bound: -0.0008103, upper bound: 0.0008103
time: 0.74 seconds

### Relational analysis RSZ of RS_RSZ1_RSZ1_RSZ2

#### Relational analysis RSZ result of RS_RSZ1_RSZ1_RSZ2
Status: Status.UNKNOWN
Output dim: 1, lower bound: -0.0008103, upper bound: 0.0008103
time: 0.70 seconds

## BFS RS instance: RS_RSZ1_RSZ2

### Backsubstitution after applying RS history:
0: 0.0007184, 0.0011172, 0.0007184, 0.0011172, -0.0003988, 0.0003988
1: 0.9934075, 0.9943579, 0.9934075, 0.9943579, -0.0009505, 0.0009505
2: -0.0087946, -0.0055215, -0.0087946, -0.0055215, -0.0030010, 0.0030017
3: 0.0035868, 0.0041619, 0.0035868, 0.0041619, -0.0005751, 0.0005751
4: 0.0027809, 0.0053678, 0.0027809, 0.0053678, -0.0025869, 0.0025869
5: 0.0050646, 0.0064029, 0.0050646, 0.0064029, -0.0013382, 0.0013382
6: -0.0021738, -0.0008562, -0.0021738, -0.0008562, -0.0013176, 0.0013176
7: -0.0082275, -0.0074685, -0.0082275, -0.0074685, -0.0007590, 0.0007590
8: 0.0055062, 0.0098066, 0.0055062, 0.0098066, -0.0042268, 0.0042266
9: -0.0036825, -0.0031002, -0.0036825, -0.0031002, -0.0005823, 0.0005823

### Unstable ReLU Count (Linear/Conv2D Layers)
- layer_idx=0, type=LayerType.Linear, total=256, inp1_unstable=9, inp2_unstable=9, delta_unstable=254
- layer_idx=2, type=LayerType.Linear, total=256, inp1_unstable=6, inp2_unstable=6, delta_unstable=256
- layer_idx=4, type=LayerType.Linear, total=256, inp1_unstable=0, inp2_unstable=0, delta_unstable=256
- layer_idx=6, type=LayerType.Linear, total=256, inp1_unstable=2, inp2_unstable=2, delta_unstable=256
- layer_idx=8, type=LayerType.Linear, total=10, inp1_unstable=0, inp2_unstable=0, delta_unstable=10

Time for backsubstitution: 1.66 seconds

### RS candidates at layer 1
type: RSZ, layer: 1, pos: 136
type: RSZ, layer: 1, pos: 199
type: RSZ, layer: 1, pos: 204
type: RSZ, layer: 1, pos: 83
type: RSZ, layer: 1, pos: 68
type: RSZ, layer: 1, pos: 198
type: RSZ, layer: 1, pos: 52

Time for candidate selection: 0.00 seconds

### Candidate
type: RSZ, layer: 1, pos: 136

### Relational analysis RSZ of RS_RSZ1_RSZ2_RSZ1

#### Relational analysis RSZ result of RS_RSZ1_RSZ2_RSZ1
Status: Status.UNKNOWN
Output dim: 1, lower bound: -0.0008079, upper bound: 0.0008071
time: 0.63 seconds

### Relational analysis RSZ of RS_RSZ1_RSZ2_RSZ2

#### Relational analysis RSZ result of RS_RSZ1_RSZ2_RSZ2
Status: Status.UNKNOWN
Output dim: 1, lower bound: -0.0008078, upper bound: 0.0008072
time: 0.67 seconds

## BFS RS instance: RS_RSZ2_RSZ1

### Backsubstitution after applying RS history:
0: 0.0007184, 0.0011172, 0.0007184, 0.0011172, -0.0003988, 0.0003988
1: 0.9934075, 0.9943579, 0.9934075, 0.9943579, -0.0009505, 0.0009505
2: -0.0087946, -0.0055215, -0.0087946, -0.0055215, -0.0029973, 0.0029953
3: 0.0035868, 0.0041619, 0.0035868, 0.0041619, -0.0005751, 0.0005751
4: 0.0027809, 0.0053678, 0.0027809, 0.0053678, -0.0025869, 0.0025869
5: 0.0050646, 0.0064029, 0.0050646, 0.0064029, -0.0013382, 0.0013382
6: -0.0021738, -0.0008562, -0.0021738, -0.0008562, -0.0013176, 0.0013176
7: -0.0082275, -0.0074685, -0.0082275, -0.0074685, -0.0007590, 0.0007590
8: 0.0055062, 0.0098066, 0.0055062, 0.0098066, -0.0042246, 0.0042251
9: -0.0036825, -0.0031002, -0.0036825, -0.0031002, -0.0005823, 0.0005823

### Unstable ReLU Count (Linear/Conv2D Layers)
- layer_idx=0, type=LayerType.Linear, total=256, inp1_unstable=9, inp2_unstable=9, delta_unstable=254
- layer_idx=2, type=LayerType.Linear, total=256, inp1_unstable=6, inp2_unstable=6, delta_unstable=256
- layer_idx=4, type=LayerType.Linear, total=256, inp1_unstable=0, inp2_unstable=0, delta_unstable=256
- layer_idx=6, type=LayerType.Linear, total=256, inp1_unstable=2, inp2_unstable=2, delta_unstable=256
- layer_idx=8, type=LayerType.Linear, total=10, inp1_unstable=0, inp2_unstable=0, delta_unstable=10

Time for backsubstitution: 1.69 seconds

### RS candidates at layer 1
type: RSZ, layer: 1, pos: 204
type: RSZ, layer: 1, pos: 52
type: RSZ, layer: 1, pos: 199
type: RSZ, layer: 1, pos: 198
type: RSZ, layer: 1, pos: 83
type: RSZ, layer: 1, pos: 136
type: RSZ, layer: 1, pos: 176

Time for candidate selection: 0.00 seconds

### Candidate
type: RSZ, layer: 1, pos: 204

### Relational analysis RSZ of RS_RSZ2_RSZ1_RSZ1

#### Relational analysis RSZ result of RS_RSZ2_RSZ1_RSZ1
Status: Status.UNKNOWN
Output dim: 1, lower bound: -0.0008096, upper bound: 0.0008156
time: 0.65 seconds

### Relational analysis RSZ of RS_RSZ2_RSZ1_RSZ2

#### Relational analysis RSZ result of RS_RSZ2_RSZ1_RSZ2
Status: Status.UNKNOWN
Output dim: 1, lower bound: -0.0008129, upper bound: 0.0008147
time: 0.77 seconds

## BFS RS instance: RS_RSZ2_RSZ2

### Backsubstitution after applying RS history:
0: 0.0007184, 0.0011172, 0.0007184, 0.0011172, -0.0003988, 0.0003988
1: 0.9934075, 0.9943579, 0.9934075, 0.9943579, -0.0009505, 0.0009505
2: -0.0087946, -0.0055215, -0.0087946, -0.0055215, -0.0029954, 0.0029970
3: 0.0035868, 0.0041619, 0.0035868, 0.0041619, -0.0005751, 0.0005751
4: 0.0027809, 0.0053678, 0.0027809, 0.0053678, -0.0025869, 0.0025869
5: 0.0050646, 0.0064029, 0.0050646, 0.0064029, -0.0013382, 0.0013382
6: -0.0021738, -0.0008562, -0.0021738, -0.0008562, -0.0013176, 0.0013176
7: -0.0082275, -0.0074685, -0.0082275, -0.0074685, -0.0007590, 0.0007590
8: 0.0055062, 0.0098066, 0.0055062, 0.0098066, -0.0042251, 0.0042247
9: -0.0036825, -0.0031002, -0.0036825, -0.0031002, -0.0005823, 0.0005823

### Unstable ReLU Count (Linear/Conv2D Layers)
- layer_idx=0, type=LayerType.Linear, total=256, inp1_unstable=9, inp2_unstable=9, delta_unstable=254
- layer_idx=2, type=LayerType.Linear, total=256, inp1_unstable=6, inp2_unstable=6, delta_unstable=256
- layer_idx=4, type=LayerType.Linear, total=256, inp1_unstable=0, inp2_unstable=0, delta_unstable=256
- layer_idx=6, type=LayerType.Linear, total=256, inp1_unstable=2, inp2_unstable=2, delta_unstable=256
- layer_idx=8, type=LayerType.Linear, total=10, inp1_unstable=0, inp2_unstable=0, delta_unstable=10

Time for backsubstitution: 1.73 seconds

### RS candidates at layer 1
type: RSZ, layer: 1, pos: 52
type: RSZ, layer: 1, pos: 136
type: RSZ, layer: 1, pos: 83
type: RSZ, layer: 1, pos: 199
type: RSZ, layer: 1, pos: 198
type: RSZ, layer: 1, pos: 176
type: RSZ, layer: 1, pos: 204

Time for candidate selection: 0.00 seconds

### Candidate
type: RSZ, layer: 1, pos: 52

### Relational analysis RSZ of RS_RSZ2_RSZ2_RSZ1

#### Relational analysis RSZ result of RS_RSZ2_RSZ2_RSZ1
Status: Status.UNKNOWN
Output dim: 1, lower bound: -0.0008164, upper bound: 0.0008111
time: 0.58 seconds

### Relational analysis RSZ of RS_RSZ2_RSZ2_RSZ2

#### Relational analysis RSZ result of RS_RSZ2_RSZ2_RSZ2
Status: Status.UNKNOWN
Output dim: 1, lower bound: -0.0008129, upper bound: 0.0008140
time: 0.60 seconds

## Summary of splitting (split count: 2)
- Time for RS candidates: 2.93 seconds
RS_RSZ1_RSZ1_RSZ1, status: Status.UNKNOWN, split count: 3, time: 2.93
Output dim: 1, lower bound: -0.0008103, upper bound: 0.0008103
RS_RSZ1_RSZ1_RSZ2, status: Status.UNKNOWN, split count: 3, time: 2.93
Output dim: 1, lower bound: -0.0008103, upper bound: 0.0008103
RS_RSZ1_RSZ2_RSZ1, status: Status.UNKNOWN, split count: 3, time: 2.93
Output dim: 1, lower bound: -0.0008079, upper bound: 0.0008071
RS_RSZ1_RSZ2_RSZ2, status: Status.UNKNOWN, split count: 3, time: 2.93
Output dim: 1, lower bound: -0.0008078, upper bound: 0.0008072
RS_RSZ2_RSZ1_RSZ1, status: Status.UNKNOWN, split count: 3, time: 2.93
Output dim: 1, lower bound: -0.0008096, upper bound: 0.0008156
RS_RSZ2_RSZ1_RSZ2, status: Status.UNKNOWN, split count: 3, time: 2.93
Output dim: 1, lower bound: -0.0008129, upper bound: 0.0008147
RS_RSZ2_RSZ2_RSZ1, status: Status.UNKNOWN, split count: 3, time: 2.93
Output dim: 1, lower bound: -0.0008164, upper bound: 0.0008111
RS_RSZ2_RSZ2_RSZ2, status: Status.UNKNOWN, split count: 3, time: 2.93
Output dim: 1, lower bound: -0.0008129, upper bound: 0.0008140

## BFS RS instance: RS_RSZ1_RSZ1_RSZ1

### Backsubstitution after applying RS history:
0: 0.0007184, 0.0011172, 0.0007184, 0.0011172, -0.0003988, 0.0003988
1: 0.9934075, 0.9943579, 0.9934075, 0.9943579, -0.0009505, 0.0009505
2: -0.0087946, -0.0055215, -0.0087946, -0.0055215, -0.0029962, 0.0029950
3: 0.0035868, 0.0041619, 0.0035868, 0.0041619, -0.0005751, 0.0005751
4: 0.0027809, 0.0053678, 0.0027809, 0.0053678, -0.0025869, 0.0025869
5: 0.0050646, 0.0064029, 0.0050646, 0.0064029, -0.0013382, 0.0013382
6: -0.0021738, -0.0008562, -0.0021738, -0.0008562, -0.0013176, 0.0013176
7: -0.0082275, -0.0074685, -0.0082275, -0.0074685, -0.0007590, 0.0007590
8: 0.0055062, 0.0098066, 0.0055062, 0.0098066, -0.0042251, 0.0042254
9: -0.0036825, -0.0031002, -0.0036825, -0.0031002, -0.0005823, 0.0005823

### Unstable ReLU Count (Linear/Conv2D Layers)
- layer_idx=0, type=LayerType.Linear, total=256, inp1_unstable=9, inp2_unstable=9, delta_unstable=253
- layer_idx=2, type=LayerType.Linear, total=256, inp1_unstable=6, inp2_unstable=6, delta_unstable=256
- layer_idx=4, type=LayerType.Linear, total=256, inp1_unstable=0, inp2_unstable=0, delta_unstable=256
- layer_idx=6, type=LayerType.Linear, total=256, inp1_unstable=2, inp2_unstable=2, delta_unstable=256
- layer_idx=8, type=LayerType.Linear, total=10, inp1_unstable=0, inp2_unstable=0, delta_unstable=10

Time for backsubstitution: 1.67 seconds

### RS candidates at layer 1
type: RSZ, layer: 1, pos: 136
type: RSZ, layer: 1, pos: 83
type: RSZ, layer: 1, pos: 198
type: RSZ, layer: 1, pos: 52
type: RSZ, layer: 1, pos: 68
type: RSZ, layer: 1, pos: 199

Time for candidate selection: 0.00 seconds

### Candidate
type: RSZ, layer: 1, pos: 136

### Relational analysis RSZ of RS_RSZ1_RSZ1_RSZ1_RSZ1

#### Relational analysis RSZ result of RS_RSZ1_RSZ1_RSZ1_RSZ1
Status: Status.UNKNOWN
Output dim: 1, lower bound: -0.0008023, upper bound: 0.0008027
time: 0.64 seconds

### Relational analysis RSZ of RS_RSZ1_RSZ1_RSZ1_RSZ2

#### Relational analysis RSZ result of RS_RSZ1_RSZ1_RSZ1_RSZ2
Status: Status.UNKNOWN
Output dim: 1, lower bound: -0.0008021, upper bound: 0.0008029
time: 0.67 seconds

## BFS RS instance: RS_RSZ1_RSZ1_RSZ2

### Backsubstitution after applying RS history:
0: 0.0007184, 0.0011172, 0.0007184, 0.0011172, -0.0003988, 0.0003988
1: 0.9934075, 0.9943579, 0.9934075, 0.9943579, -0.0009505, 0.0009505
2: -0.0087946, -0.0055215, -0.0087946, -0.0055215, -0.0029955, 0.0029958
3: 0.0035868, 0.0041619, 0.0035868, 0.0041619, -0.0005751, 0.0005751
4: 0.0027809, 0.0053678, 0.0027809, 0.0053678, -0.0025869, 0.0025869
5: 0.0050646, 0.0064029, 0.0050646, 0.0064029, -0.0013382, 0.0013382
6: -0.0021738, -0.0008562, -0.0021738, -0.0008562, -0.0013176, 0.0013176
7: -0.0082275, -0.0074685, -0.0082275, -0.0074685, -0.0007590, 0.0007590
8: 0.0055062, 0.0098066, 0.0055062, 0.0098066, -0.0042253, 0.0042252
9: -0.0036825, -0.0031002, -0.0036825, -0.0031002, -0.0005823, 0.0005823

### Unstable ReLU Count (Linear/Conv2D Layers)
- layer_idx=0, type=LayerType.Linear, total=256, inp1_unstable=9, inp2_unstable=9, delta_unstable=253
- layer_idx=2, type=LayerType.Linear, total=256, inp1_unstable=6, inp2_unstable=6, delta_unstable=256
- layer_idx=4, type=LayerType.Linear, total=256, inp1_unstable=0, inp2_unstable=0, delta_unstable=256
- layer_idx=6, type=LayerType.Linear, total=256, inp1_unstable=2, inp2_unstable=2, delta_unstable=256
- layer_idx=8, type=LayerType.Linear, total=10, inp1_unstable=0, inp2_unstable=0, delta_unstable=10

Time for backsubstitution: 1.69 seconds

### RS candidates at layer 1
type: RSZ, layer: 1, pos: 198
type: RSZ, layer: 1, pos: 136
type: RSZ, layer: 1, pos: 68
type: RSZ, layer: 1, pos: 83
type: RSZ, layer: 1, pos: 52
type: RSZ, layer: 1, pos: 199

Time for candidate selection: 0.00 seconds

### Candidate
type: RSZ, layer: 1, pos: 198

### Relational analysis RSZ of RS_RSZ1_RSZ1_RSZ2_RSZ1

#### Relational analysis RSZ result of RS_RSZ1_RSZ1_RSZ2_RSZ1
Status: Status.UNKNOWN
Output dim: 1, lower bound: -0.0007767, upper bound: 0.0007767
time: 0.81 seconds

### Relational analysis RSZ of RS_RSZ1_RSZ1_RSZ2_RSZ2

#### Relational analysis RSZ result of RS_RSZ1_RSZ1_RSZ2_RSZ2
Status: Status.UNKNOWN
Output dim: 1, lower bound: -0.0007767, upper bound: 0.0007767
time: 0.70 seconds

## BFS RS instance: RS_RSZ1_RSZ2_RSZ1

### Backsubstitution after applying RS history:
0: 0.0007184, 0.0011172, 0.0007184, 0.0011172, -0.0003988, 0.0003988
1: 0.9934075, 0.9943579, 0.9934075, 0.9943579, -0.0009505, 0.0009505
2: -0.0087946, -0.0055215, -0.0087946, -0.0055215, -0.0029943, 0.0029972
3: 0.0035868, 0.0041619, 0.0035868, 0.0041619, -0.0005751, 0.0005751
4: 0.0027809, 0.0053678, 0.0027809, 0.0053678, -0.0025869, 0.0025869
5: 0.0050646, 0.0064029, 0.0050646, 0.0064029, -0.0013382, 0.0013382
6: -0.0021738, -0.0008562, -0.0021738, -0.0008562, -0.0013176, 0.0013176
7: -0.0082275, -0.0074685, -0.0082275, -0.0074685, -0.0007590, 0.0007590
8: 0.0055062, 0.0098066, 0.0055062, 0.0098066, -0.0042260, 0.0042253
9: -0.0036825, -0.0031002, -0.0036825, -0.0031002, -0.0005823, 0.0005823

### Unstable ReLU Count (Linear/Conv2D Layers)
- layer_idx=0, type=LayerType.Linear, total=256, inp1_unstable=9, inp2_unstable=9, delta_unstable=253
- layer_idx=2, type=LayerType.Linear, total=256, inp1_unstable=6, inp2_unstable=6, delta_unstable=256
- layer_idx=4, type=LayerType.Linear, total=256, inp1_unstable=0, inp2_unstable=0, delta_unstable=256
- layer_idx=6, type=LayerType.Linear, total=256, inp1_unstable=2, inp2_unstable=2, delta_unstable=256
- layer_idx=8, type=LayerType.Linear, total=10, inp1_unstable=0, inp2_unstable=0, delta_unstable=10

Time for backsubstitution: 1.71 seconds

### RS candidates at layer 1
type: RSZ, layer: 1, pos: 198
type: RSZ, layer: 1, pos: 68
type: RSZ, layer: 1, pos: 204
type: RSZ, layer: 1, pos: 52
type: RSZ, layer: 1, pos: 199
type: RSZ, layer: 1, pos: 83

Time for candidate selection: 0.00 seconds

### Candidate
type: RSZ, layer: 1, pos: 198

### Relational analysis RSZ of RS_RSZ1_RSZ2_RSZ1_RSZ1

#### Relational analysis RSZ result of RS_RSZ1_RSZ2_RSZ1_RSZ1
Status: Status.UNKNOWN
Output dim: 1, lower bound: -0.0007812, upper bound: 0.0007811
time: 0.63 seconds

### Relational analysis RSZ of RS_RSZ1_RSZ2_RSZ1_RSZ2

#### Relational analysis RSZ result of RS_RSZ1_RSZ2_RSZ1_RSZ2
Status: Status.UNKNOWN
Output dim: 1, lower bound: -0.0007812, upper bound: 0.0007811
time: 0.61 seconds

## BFS RS instance: RS_RSZ1_RSZ2_RSZ2

### Backsubstitution after applying RS history:
0: 0.0007184, 0.0011172, 0.0007184, 0.0011172, -0.0003988, 0.0003988
1: 0.9934075, 0.9943579, 0.9934075, 0.9943579, -0.0009505, 0.0009505
2: -0.0087946, -0.0055215, -0.0087946, -0.0055215, -0.0029965, 0.0029951
3: 0.0035868, 0.0041619, 0.0035868, 0.0041619, -0.0005751, 0.0005751
4: 0.0027809, 0.0053678, 0.0027809, 0.0053678, -0.0025869, 0.0025869
5: 0.0050646, 0.0064029, 0.0050646, 0.0064029, -0.0013382, 0.0013382
6: -0.0021738, -0.0008562, -0.0021738, -0.0008562, -0.0013176, 0.0013176
7: -0.0082275, -0.0074685, -0.0082275, -0.0074685, -0.0007590, 0.0007590
8: 0.0055062, 0.0098066, 0.0055062, 0.0098066, -0.0042255, 0.0042258
9: -0.0036825, -0.0031002, -0.0036825, -0.0031002, -0.0005823, 0.0005823

### Unstable ReLU Count (Linear/Conv2D Layers)
- layer_idx=0, type=LayerType.Linear, total=256, inp1_unstable=9, inp2_unstable=9, delta_unstable=253
- layer_idx=2, type=LayerType.Linear, total=256, inp1_unstable=6, inp2_unstable=6, delta_unstable=256
- layer_idx=4, type=LayerType.Linear, total=256, inp1_unstable=0, inp2_unstable=0, delta_unstable=256
- layer_idx=6, type=LayerType.Linear, total=256, inp1_unstable=2, inp2_unstable=2, delta_unstable=256
- layer_idx=8, type=LayerType.Linear, total=10, inp1_unstable=0, inp2_unstable=0, delta_unstable=10

Time for backsubstitution: 1.67 seconds

### RS candidates at layer 1
type: RSZ, layer: 1, pos: 199
type: RSZ, layer: 1, pos: 198
type: RSZ, layer: 1, pos: 68
type: RSZ, layer: 1, pos: 83
type: RSZ, layer: 1, pos: 52
type: RSZ, layer: 1, pos: 204

Time for candidate selection: 0.00 seconds

### Candidate
type: RSZ, layer: 1, pos: 199

### Relational analysis RSZ of RS_RSZ1_RSZ2_RSZ2_RSZ1

#### Relational analysis RSZ result of RS_RSZ1_RSZ2_RSZ2_RSZ1
Status: Status.UNKNOWN
Output dim: 1, lower bound: -0.0007601, upper bound: 0.0007603
time: 0.67 seconds

### Relational analysis RSZ of RS_RSZ1_RSZ2_RSZ2_RSZ2

#### Relational analysis RSZ result of RS_RSZ1_RSZ2_RSZ2_RSZ2
Status: Status.UNKNOWN
Output dim: 1, lower bound: -0.0007601, upper bound: 0.0007603
time: 0.67 seconds

## BFS RS instance: RS_RSZ2_RSZ1_RSZ1

### Backsubstitution after applying RS history:
0: 0.0007184, 0.0011172, 0.0007184, 0.0011172, -0.0003988, 0.0003988
1: 0.9934075, 0.9943579, 0.9934075, 0.9943579, -0.0009505, 0.0009505
2: -0.0087946, -0.0055215, -0.0087946, -0.0055215, -0.0029917, 0.0029892
3: 0.0035868, 0.0041619, 0.0035868, 0.0041619, -0.0005751, 0.0005751
4: 0.0027809, 0.0053678, 0.0027809, 0.0053678, -0.0025869, 0.0025869
5: 0.0050646, 0.0064029, 0.0050646, 0.0064029, -0.0013382, 0.0013382
6: -0.0021738, -0.0008562, -0.0021738, -0.0008562, -0.0013176, 0.0013176
7: -0.0082275, -0.0074685, -0.0082275, -0.0074685, -0.0007590, 0.0007590
8: 0.0055062, 0.0098066, 0.0055062, 0.0098066, -0.0042231, 0.0042237
9: -0.0036825, -0.0031002, -0.0036825, -0.0031002, -0.0005823, 0.0005823

### Unstable ReLU Count (Linear/Conv2D Layers)
- layer_idx=0, type=LayerType.Linear, total=256, inp1_unstable=9, inp2_unstable=9, delta_unstable=253
- layer_idx=2, type=LayerType.Linear, total=256, inp1_unstable=6, inp2_unstable=6, delta_unstable=256
- layer_idx=4, type=LayerType.Linear, total=256, inp1_unstable=0, inp2_unstable=0, delta_unstable=256
- layer_idx=6, type=LayerType.Linear, total=256, inp1_unstable=2, inp2_unstable=2, delta_unstable=256
- layer_idx=8, type=LayerType.Linear, total=10, inp1_unstable=0, inp2_unstable=0, delta_unstable=10

Time for backsubstitution: 1.61 seconds

### RS candidates at layer 1
type: RSZ, layer: 1, pos: 83
type: RSZ, layer: 1, pos: 198
type: RSZ, layer: 1, pos: 52
type: RSZ, layer: 1, pos: 136
type: RSZ, layer: 1, pos: 199
type: RSZ, layer: 1, pos: 176

Time for candidate selection: 0.00 seconds

### Candidate
type: RSZ, layer: 1, pos: 83

### Relational analysis RSZ of RS_RSZ2_RSZ1_RSZ1_RSZ1

#### Relational analysis RSZ result of RS_RSZ2_RSZ1_RSZ1_RSZ1
Status: Status.UNKNOWN
Output dim: 1, lower bound: -0.0008067, upper bound: 0.0008125
time: 0.75 seconds

### Relational analysis RSZ of RS_RSZ2_RSZ1_RSZ1_RSZ2

#### Relational analysis RSZ result of RS_RSZ2_RSZ1_RSZ1_RSZ2
Status: Status.UNKNOWN
Output dim: 1, lower bound: -0.0008067, upper bound: 0.0008126
time: 0.71 seconds

## BFS RS instance: RS_RSZ2_RSZ1_RSZ2

### Backsubstitution after applying RS history:
0: 0.0007184, 0.0011172, 0.0007184, 0.0011172, -0.0003988, 0.0003988
1: 0.9934075, 0.9943579, 0.9934075, 0.9943579, -0.0009505, 0.0009505
2: -0.0087946, -0.0055215, -0.0087946, -0.0055215, -0.0029911, 0.0029900
3: 0.0035868, 0.0041619, 0.0035868, 0.0041619, -0.0005751, 0.0005751
4: 0.0027809, 0.0053678, 0.0027809, 0.0053678, -0.0025869, 0.0025869
5: 0.0050646, 0.0064029, 0.0050646, 0.0064029, -0.0013382, 0.0013382
6: -0.0021738, -0.0008562, -0.0021738, -0.0008562, -0.0013176, 0.0013176
7: -0.0082275, -0.0074685, -0.0082275, -0.0074685, -0.0007590, 0.0007590
8: 0.0055062, 0.0098066, 0.0055062, 0.0098066, -0.0042233, 0.0042236
9: -0.0036825, -0.0031002, -0.0036825, -0.0031002, -0.0005823, 0.0005823

### Unstable ReLU Count (Linear/Conv2D Layers)
- layer_idx=0, type=LayerType.Linear, total=256, inp1_unstable=9, inp2_unstable=9, delta_unstable=253
- layer_idx=2, type=LayerType.Linear, total=256, inp1_unstable=6, inp2_unstable=6, delta_unstable=256
- layer_idx=4, type=LayerType.Linear, total=256, inp1_unstable=0, inp2_unstable=0, delta_unstable=256
- layer_idx=6, type=LayerType.Linear, total=256, inp1_unstable=2, inp2_unstable=2, delta_unstable=256
- layer_idx=8, type=LayerType.Linear, total=10, inp1_unstable=0, inp2_unstable=0, delta_unstable=10

Time for backsubstitution: 1.62 seconds

### RS candidates at layer 1
type: RSZ, layer: 1, pos: 176
type: RSZ, layer: 1, pos: 136
type: RSZ, layer: 1, pos: 52
type: RSZ, layer: 1, pos: 83
type: RSZ, layer: 1, pos: 199
type: RSZ, layer: 1, pos: 198

Time for candidate selection: 0.00 seconds

### Candidate
type: RSZ, layer: 1, pos: 176

### Relational analysis RSZ of RS_RSZ2_RSZ1_RSZ2_RSZ1

#### Relational analysis RSZ result of RS_RSZ2_RSZ1_RSZ2_RSZ1
Status: Status.UNKNOWN
Output dim: 1, lower bound: -0.0008061, upper bound: 0.0008083
time: 0.78 seconds

### Relational analysis RSZ of RS_RSZ2_RSZ1_RSZ2_RSZ2

#### Relational analysis RSZ result of RS_RSZ2_RSZ1_RSZ2_RSZ2
Status: Status.UNKNOWN
Output dim: 1, lower bound: -0.0008061, upper bound: 0.0008082
time: 0.78 seconds

## BFS RS instance: RS_RSZ2_RSZ2_RSZ1

### Backsubstitution after applying RS history:
0: 0.0007184, 0.0011172, 0.0007184, 0.0011172, -0.0003988, 0.0003988
1: 0.9934075, 0.9943579, 0.9934075, 0.9943579, -0.0009505, 0.0009505
2: -0.0087946, -0.0055215, -0.0087946, -0.0055215, -0.0029919, 0.0029945
3: 0.0035868, 0.0041619, 0.0035868, 0.0041619, -0.0005751, 0.0005751
4: 0.0027809, 0.0053678, 0.0027809, 0.0053678, -0.0025869, 0.0025869
5: 0.0050646, 0.0064029, 0.0050646, 0.0064029, -0.0013382, 0.0013382
6: -0.0021738, -0.0008562, -0.0021738, -0.0008562, -0.0013176, 0.0013176
7: -0.0082275, -0.0074685, -0.0082275, -0.0074685, -0.0007590, 0.0007590
8: 0.0055062, 0.0098066, 0.0055062, 0.0098066, -0.0042244, 0.0042237
9: -0.0036825, -0.0031002, -0.0036825, -0.0031002, -0.0005823, 0.0005823

### Unstable ReLU Count (Linear/Conv2D Layers)
- layer_idx=0, type=LayerType.Linear, total=256, inp1_unstable=9, inp2_unstable=9, delta_unstable=253
- layer_idx=2, type=LayerType.Linear, total=256, inp1_unstable=6, inp2_unstable=6, delta_unstable=256
- layer_idx=4, type=LayerType.Linear, total=256, inp1_unstable=0, inp2_unstable=0, delta_unstable=256
- layer_idx=6, type=LayerType.Linear, total=256, inp1_unstable=2, inp2_unstable=2, delta_unstable=256
- layer_idx=8, type=LayerType.Linear, total=10, inp1_unstable=0, inp2_unstable=0, delta_unstable=10

Time for backsubstitution: 1.66 seconds

### RS candidates at layer 1
type: RSZ, layer: 1, pos: 204
type: RSZ, layer: 1, pos: 198
type: RSZ, layer: 1, pos: 199
type: RSZ, layer: 1, pos: 83
type: RSZ, layer: 1, pos: 176
type: RSZ, layer: 1, pos: 136

Time for candidate selection: 0.00 seconds

### Candidate
type: RSZ, layer: 1, pos: 204

### Relational analysis RSZ of RS_RSZ2_RSZ2_RSZ1_RSZ1

#### Relational analysis RSZ result of RS_RSZ2_RSZ2_RSZ1_RSZ1
Status: Status.UNKNOWN
Output dim: 1, lower bound: -0.0008104, upper bound: 0.0008067
time: 0.66 seconds

### Relational analysis RSZ of RS_RSZ2_RSZ2_RSZ1_RSZ2

#### Relational analysis RSZ result of RS_RSZ2_RSZ2_RSZ1_RSZ2
Status: Status.UNKNOWN
Output dim: 1, lower bound: -0.0008118, upper bound: 0.0008053
time: 0.67 seconds

## BFS RS instance: RS_RSZ2_RSZ2_RSZ2

### Backsubstitution after applying RS history:
0: 0.0007184, 0.0011172, 0.0007184, 0.0011172, -0.0003988, 0.0003988
1: 0.9934075, 0.9943579, 0.9934075, 0.9943579, -0.0009505, 0.0009505
2: -0.0087946, -0.0055215, -0.0087946, -0.0055215, -0.0029929, 0.0029935
3: 0.0035868, 0.0041619, 0.0035868, 0.0041619, -0.0005751, 0.0005751
4: 0.0027809, 0.0053678, 0.0027809, 0.0053678, -0.0025869, 0.0025869
5: 0.0050646, 0.0064029, 0.0050646, 0.0064029, -0.0013382, 0.0013382
6: -0.0021738, -0.0008562, -0.0021738, -0.0008562, -0.0013176, 0.0013176
7: -0.0082275, -0.0074685, -0.0082275, -0.0074685, -0.0007590, 0.0007590
8: 0.0055062, 0.0098066, 0.0055062, 0.0098066, -0.0042241, 0.0042239
9: -0.0036825, -0.0031002, -0.0036825, -0.0031002, -0.0005823, 0.0005823

### Unstable ReLU Count (Linear/Conv2D Layers)
- layer_idx=0, type=LayerType.Linear, total=256, inp1_unstable=9, inp2_unstable=9, delta_unstable=253
- layer_idx=2, type=LayerType.Linear, total=256, inp1_unstable=6, inp2_unstable=6, delta_unstable=256
- layer_idx=4, type=LayerType.Linear, total=256, inp1_unstable=0, inp2_unstable=0, delta_unstable=256
- layer_idx=6, type=LayerType.Linear, total=256, inp1_unstable=2, inp2_unstable=2, delta_unstable=256
- layer_idx=8, type=LayerType.Linear, total=10, inp1_unstable=0, inp2_unstable=0, delta_unstable=10

Time for backsubstitution: 1.63 seconds

### RS candidates at layer 1
type: RSZ, layer: 1, pos: 198
type: RSZ, layer: 1, pos: 204
type: RSZ, layer: 1, pos: 83
type: RSZ, layer: 1, pos: 176
type: RSZ, layer: 1, pos: 199
type: RSZ, layer: 1, pos: 136

Time for candidate selection: 0.00 seconds

### Candidate
type: RSZ, layer: 1, pos: 198

### Relational analysis RSZ of RS_RSZ2_RSZ2_RSZ2_RSZ1

#### Relational analysis RSZ result of RS_RSZ2_RSZ2_RSZ2_RSZ1
Status: Status.UNKNOWN
Output dim: 1, lower bound: -0.0007887, upper bound: 0.0007883
time: 0.71 seconds

### Relational analysis RSZ of RS_RSZ2_RSZ2_RSZ2_RSZ2

#### Relational analysis RSZ result of RS_RSZ2_RSZ2_RSZ2_RSZ2
Status: Status.UNKNOWN
Output dim: 1, lower bound: -0.0007887, upper bound: 0.0007883
time: 0.71 seconds

## Summary of splitting (split count: 3)
- Time for RS candidates: 3.06 seconds
RS_RSZ1_RSZ1_RSZ1_RSZ1, status: Status.UNKNOWN, split count: 4, time: 3.06
Output dim: 1, lower bound: -0.0008023, upper bound: 0.0008027
RS_RSZ1_RSZ1_RSZ1_RSZ2, status: Status.UNKNOWN, split count: 4, time: 3.06
Output dim: 1, lower bound: -0.0008021, upper bound: 0.0008029
RS_RSZ1_RSZ1_RSZ2_RSZ1, status: Status.UNKNOWN, split count: 4, time: 3.06
Output dim: 1, lower bound: -0.0007767, upper bound: 0.0007767
RS_RSZ1_RSZ1_RSZ2_RSZ2, status: Status.UNKNOWN, split count: 4, time: 3.06
Output dim: 1, lower bound: -0.0007767, upper bound: 0.0007767
RS_RSZ1_RSZ2_RSZ1_RSZ1, status: Status.UNKNOWN, split count: 4, time: 3.06
Output dim: 1, lower bound: -0.0007812, upper bound: 0.0007811
RS_RSZ1_RSZ2_RSZ1_RSZ2, status: Status.UNKNOWN, split count: 4, time: 3.06
Output dim: 1, lower bound: -0.0007812, upper bound: 0.0007811
RS_RSZ1_RSZ2_RSZ2_RSZ1, status: Status.UNKNOWN, split count: 4, time: 3.06
Output dim: 1, lower bound: -0.0007601, upper bound: 0.0007603
RS_RSZ1_RSZ2_RSZ2_RSZ2, status: Status.UNKNOWN, split count: 4, time: 3.06
Output dim: 1, lower bound: -0.0007601, upper bound: 0.0007603
RS_RSZ2_RSZ1_RSZ1_RSZ1, status: Status.UNKNOWN, split count: 4, time: 3.06
Output dim: 1, lower bound: -0.0008067, upper bound: 0.0008125
RS_RSZ2_RSZ1_RSZ1_RSZ2, status: Status.UNKNOWN, split count: 4, time: 3.06
Output dim: 1, lower bound: -0.0008067, upper bound: 0.0008126
RS_RSZ2_RSZ1_RSZ2_RSZ1, status: Status.UNKNOWN, split count: 4, time: 3.06
Output dim: 1, lower bound: -0.0008061, upper bound: 0.0008083
RS_RSZ2_RSZ1_RSZ2_RSZ2, status: Status.UNKNOWN, split count: 4, time: 3.06
Output dim: 1, lower bound: -0.0008061, upper bound: 0.0008082
RS_RSZ2_RSZ2_RSZ1_RSZ1, status: Status.UNKNOWN, split count: 4, time: 3.06
Output dim: 1, lower bound: -0.0008104, upper bound: 0.0008067
RS_RSZ2_RSZ2_RSZ1_RSZ2, status: Status.UNKNOWN, split count: 4, time: 3.06
Output dim: 1, lower bound: -0.0008118, upper bound: 0.0008053
RS_RSZ2_RSZ2_RSZ2_RSZ1, status: Status.UNKNOWN, split count: 4, time: 3.06
Output dim: 1, lower bound: -0.0007887, upper bound: 0.0007883
RS_RSZ2_RSZ2_RSZ2_RSZ2, status: Status.UNKNOWN, split count: 4, time: 3.06
Output dim: 1, lower bound: -0.0007887, upper bound: 0.0007883

## BFS RS instance: RS_RSZ1_RSZ1_RSZ1_RSZ1

### Backsubstitution after applying RS history:
0: 0.0007184, 0.0011172, 0.0007184, 0.0011172, -0.0003988, 0.0003988
1: 0.9934075, 0.9943579, 0.9934075, 0.9943579, -0.0009505, 0.0009505
2: -0.0087946, -0.0055215, -0.0087946, -0.0055215, -0.0029897, 0.0029905
3: 0.0035868, 0.0041619, 0.0035868, 0.0041619, -0.0005751, 0.0005751
4: 0.0027809, 0.0053678, 0.0027809, 0.0053678, -0.0025869, 0.0025869
5: 0.0050646, 0.0064029, 0.0050646, 0.0064029, -0.0013382, 0.0013382
6: -0.0021738, -0.0008562, -0.0021738, -0.0008562, -0.0013176, 0.0013176
7: -0.0082275, -0.0074685, -0.0082275, -0.0074685, -0.0007590, 0.0007590
8: 0.0055062, 0.0098066, 0.0055062, 0.0098066, -0.0042243, 0.0042241
9: -0.0036825, -0.0031002, -0.0036825, -0.0031002, -0.0005823, 0.0005823

### Unstable ReLU Count (Linear/Conv2D Layers)
- layer_idx=0, type=LayerType.Linear, total=256, inp1_unstable=9, inp2_unstable=9, delta_unstable=252
- layer_idx=2, type=LayerType.Linear, total=256, inp1_unstable=6, inp2_unstable=6, delta_unstable=256
- layer_idx=4, type=LayerType.Linear, total=256, inp1_unstable=0, inp2_unstable=0, delta_unstable=256
- layer_idx=6, type=LayerType.Linear, total=256, inp1_unstable=2, inp2_unstable=2, delta_unstable=256
- layer_idx=8, type=LayerType.Linear, total=10, inp1_unstable=0, inp2_unstable=0, delta_unstable=10

Time for backsubstitution: 1.66 seconds

### RS candidates at layer 1
type: RSZ, layer: 1, pos: 52
type: RSZ, layer: 1, pos: 199
type: RSZ, layer: 1, pos: 83
type: RSZ, layer: 1, pos: 68
type: RSZ, layer: 1, pos: 198

Time for candidate selection: 0.00 seconds

### Candidate
type: RSZ, layer: 1, pos: 52

### Relational analysis RSZ of RS_RSZ1_RSZ1_RSZ1_RSZ1_RSZ1

#### Relational analysis RSZ result of RS_RSZ1_RSZ1_RSZ1_RSZ1_RSZ1
Status: Status.UNKNOWN
Output dim: 1, lower bound: -0.0007983, upper bound: 0.0007961
time: 0.63 seconds

### Relational analysis RSZ of RS_RSZ1_RSZ1_RSZ1_RSZ1_RSZ2

#### Relational analysis RSZ result of RS_RSZ1_RSZ1_RSZ1_RSZ1_RSZ2
Status: Status.UNKNOWN
Output dim: 1, lower bound: -0.0007959, upper bound: 0.0007987
time: 0.70 seconds

## BFS RS instance: RS_RSZ1_RSZ1_RSZ1_RSZ2

### Backsubstitution after applying RS history:
0: 0.0007184, 0.0011172, 0.0007184, 0.0011172, -0.0003988, 0.0003988
1: 0.9934075, 0.9943579, 0.9934075, 0.9943579, -0.0009505, 0.0009505
2: -0.0087946, -0.0055215, -0.0087946, -0.0055215, -0.0029917, 0.0029884
3: 0.0035868, 0.0041619, 0.0035868, 0.0041619, -0.0005751, 0.0005751
4: 0.0027809, 0.0053678, 0.0027809, 0.0053678, -0.0025869, 0.0025869
5: 0.0050646, 0.0064029, 0.0050646, 0.0064029, -0.0013382, 0.0013382
6: -0.0021738, -0.0008562, -0.0021738, -0.0008562, -0.0013176, 0.0013176
7: -0.0082275, -0.0074685, -0.0082275, -0.0074685, -0.0007590, 0.0007590
8: 0.0055062, 0.0098066, 0.0055062, 0.0098066, -0.0042238, 0.0042246
9: -0.0036825, -0.0031002, -0.0036825, -0.0031002, -0.0005823, 0.0005823

### Unstable ReLU Count (Linear/Conv2D Layers)
- layer_idx=0, type=LayerType.Linear, total=256, inp1_unstable=9, inp2_unstable=9, delta_unstable=252
- layer_idx=2, type=LayerType.Linear, total=256, inp1_unstable=6, inp2_unstable=6, delta_unstable=256
- layer_idx=4, type=LayerType.Linear, total=256, inp1_unstable=0, inp2_unstable=0, delta_unstable=256
- layer_idx=6, type=LayerType.Linear, total=256, inp1_unstable=2, inp2_unstable=2, delta_unstable=256
- layer_idx=8, type=LayerType.Linear, total=10, inp1_unstable=0, inp2_unstable=0, delta_unstable=10

Time for backsubstitution: 1.70 seconds

### RS candidates at layer 1
type: RSZ, layer: 1, pos: 83
type: RSZ, layer: 1, pos: 68
type: RSZ, layer: 1, pos: 198
type: RSZ, layer: 1, pos: 199
type: RSZ, layer: 1, pos: 52

Time for candidate selection: 0.00 seconds

### Candidate
type: RSZ, layer: 1, pos: 83

### Relational analysis RSZ of RS_RSZ1_RSZ1_RSZ1_RSZ2_RSZ1

#### Relational analysis RSZ result of RS_RSZ1_RSZ1_RSZ1_RSZ2_RSZ1
Status: Status.UNKNOWN
Output dim: 1, lower bound: -0.0007992, upper bound: 0.0007998
time: 0.72 seconds

### Relational analysis RSZ of RS_RSZ1_RSZ1_RSZ1_RSZ2_RSZ2

#### Relational analysis RSZ result of RS_RSZ1_RSZ1_RSZ1_RSZ2_RSZ2
Status: Status.UNKNOWN
Output dim: 1, lower bound: -0.0007991, upper bound: 0.0007999
time: 0.64 seconds

## BFS RS instance: RS_RSZ1_RSZ1_RSZ2_RSZ1

### Backsubstitution after applying RS history:
0: 0.0007184, 0.0011172, 0.0007184, 0.0011172, -0.0003988, 0.0003988
1: 0.9934075, 0.9943579, 0.9934075, 0.9943579, -0.0009505, 0.0009505
2: -0.0087946, -0.0055215, -0.0087946, -0.0055215, -0.0029849, 0.0029857
3: 0.0035868, 0.0041619, 0.0035868, 0.0041619, -0.0005751, 0.0005751
4: 0.0027809, 0.0053678, 0.0027809, 0.0053678, -0.0025869, 0.0025869
5: 0.0050646, 0.0064029, 0.0050646, 0.0064029, -0.0013382, 0.0013382
6: -0.0021738, -0.0008562, -0.0021738, -0.0008562, -0.0013176, 0.0013176
7: -0.0082275, -0.0074685, -0.0082275, -0.0074685, -0.0007590, 0.0007590
8: 0.0055062, 0.0098066, 0.0055062, 0.0098066, -0.0042229, 0.0042227
9: -0.0036825, -0.0031002, -0.0036825, -0.0031002, -0.0005823, 0.0005823

### Unstable ReLU Count (Linear/Conv2D Layers)
- layer_idx=0, type=LayerType.Linear, total=256, inp1_unstable=9, inp2_unstable=9, delta_unstable=252
- layer_idx=2, type=LayerType.Linear, total=256, inp1_unstable=6, inp2_unstable=6, delta_unstable=256
- layer_idx=4, type=LayerType.Linear, total=256, inp1_unstable=0, inp2_unstable=0, delta_unstable=256
- layer_idx=6, type=LayerType.Linear, total=256, inp1_unstable=2, inp2_unstable=2, delta_unstable=256
- layer_idx=8, type=LayerType.Linear, total=10, inp1_unstable=0, inp2_unstable=0, delta_unstable=10

Time for backsubstitution: 1.62 seconds

### RS candidates at layer 1
type: RSZ, layer: 1, pos: 52
type: RSZ, layer: 1, pos: 68
type: RSZ, layer: 1, pos: 136
type: RSZ, layer: 1, pos: 199
type: RSZ, layer: 1, pos: 83

Time for candidate selection: 0.00 seconds

### Candidate
type: RSZ, layer: 1, pos: 52

### Relational analysis RSZ of RS_RSZ1_RSZ1_RSZ2_RSZ1_RSZ1

#### Relational analysis RSZ result of RS_RSZ1_RSZ1_RSZ2_RSZ1_RSZ1
Status: Status.UNKNOWN
Output dim: 1, lower bound: -0.0007766, upper bound: 0.0007766
time: 0.68 seconds

### Relational analysis RSZ of RS_RSZ1_RSZ1_RSZ2_RSZ1_RSZ2

#### Relational analysis RSZ result of RS_RSZ1_RSZ1_RSZ2_RSZ1_RSZ2
Status: Status.UNKNOWN
Output dim: 1, lower bound: -0.0007766, upper bound: 0.0007766
time: 0.63 seconds

## BFS RS instance: RS_RSZ1_RSZ1_RSZ2_RSZ2

### Backsubstitution after applying RS history:
0: 0.0007184, 0.0011172, 0.0007184, 0.0011172, -0.0003988, 0.0003988
1: 0.9934075, 0.9943579, 0.9934075, 0.9943579, -0.0009505, 0.0009505
2: -0.0087946, -0.0055215, -0.0087946, -0.0055215, -0.0029955, 0.0029852
3: 0.0035868, 0.0041619, 0.0035868, 0.0041619, -0.0005751, 0.0005751
4: 0.0027809, 0.0053678, 0.0027809, 0.0053678, -0.0025869, 0.0025869
5: 0.0050646, 0.0064029, 0.0050646, 0.0064029, -0.0013382, 0.0013382
6: -0.0021738, -0.0008562, -0.0021738, -0.0008562, -0.0013176, 0.0013176
7: -0.0082275, -0.0074685, -0.0082275, -0.0074685, -0.0007590, 0.0007590
8: 0.0055062, 0.0098066, 0.0055062, 0.0098066, -0.0042227, 0.0042252
9: -0.0036825, -0.0031002, -0.0036825, -0.0031002, -0.0005823, 0.0005823

### Unstable ReLU Count (Linear/Conv2D Layers)
- layer_idx=0, type=LayerType.Linear, total=256, inp1_unstable=9, inp2_unstable=9, delta_unstable=252
- layer_idx=2, type=LayerType.Linear, total=256, inp1_unstable=6, inp2_unstable=6, delta_unstable=256
- layer_idx=4, type=LayerType.Linear, total=256, inp1_unstable=0, inp2_unstable=0, delta_unstable=256
- layer_idx=6, type=LayerType.Linear, total=256, inp1_unstable=2, inp2_unstable=2, delta_unstable=256
- layer_idx=8, type=LayerType.Linear, total=10, inp1_unstable=0, inp2_unstable=0, delta_unstable=10

Time for backsubstitution: 1.62 seconds

### RS candidates at layer 1
type: RSZ, layer: 1, pos: 68
type: RSZ, layer: 1, pos: 199
type: RSZ, layer: 1, pos: 52
type: RSZ, layer: 1, pos: 136
type: RSZ, layer: 1, pos: 83

Time for candidate selection: 0.00 seconds

### Candidate
type: RSZ, layer: 1, pos: 68

### Relational analysis RSZ of RS_RSZ1_RSZ1_RSZ2_RSZ2_RSZ1

#### Relational analysis RSZ result of RS_RSZ1_RSZ1_RSZ2_RSZ2_RSZ1
Status: Status.UNKNOWN
Output dim: 1, lower bound: -0.0007766, upper bound: 0.0007766
time: 0.73 seconds

### Relational analysis RSZ of RS_RSZ1_RSZ1_RSZ2_RSZ2_RSZ2

#### Relational analysis RSZ result of RS_RSZ1_RSZ1_RSZ2_RSZ2_RSZ2
Status: Status.UNKNOWN
Output dim: 1, lower bound: -0.0007766, upper bound: 0.0007766
time: 0.70 seconds

## BFS RS instance: RS_RSZ1_RSZ2_RSZ1_RSZ1

### Backsubstitution after applying RS history:
0: 0.0007184, 0.0011172, 0.0007184, 0.0011172, -0.0003988, 0.0003988
1: 0.9934075, 0.9943579, 0.9934075, 0.9943579, -0.0009505, 0.0009505
2: -0.0087946, -0.0055215, -0.0087946, -0.0055215, -0.0029838, 0.0029874
3: 0.0035868, 0.0041619, 0.0035868, 0.0041619, -0.0005751, 0.0005751
4: 0.0027809, 0.0053678, 0.0027809, 0.0053678, -0.0025869, 0.0025869
5: 0.0050646, 0.0064029, 0.0050646, 0.0064029, -0.0013382, 0.0013382
6: -0.0021738, -0.0008562, -0.0021738, -0.0008562, -0.0013176, 0.0013176
7: -0.0082275, -0.0074685, -0.0082275, -0.0074685, -0.0007590, 0.0007590
8: 0.0055062, 0.0098066, 0.0055062, 0.0098066, -0.0042237, 0.0042227
9: -0.0036825, -0.0031002, -0.0036825, -0.0031002, -0.0005823, 0.0005823

### Unstable ReLU Count (Linear/Conv2D Layers)
- layer_idx=0, type=LayerType.Linear, total=256, inp1_unstable=9, inp2_unstable=9, delta_unstable=252
- layer_idx=2, type=LayerType.Linear, total=256, inp1_unstable=6, inp2_unstable=6, delta_unstable=256
- layer_idx=4, type=LayerType.Linear, total=256, inp1_unstable=0, inp2_unstable=0, delta_unstable=256
- layer_idx=6, type=LayerType.Linear, total=256, inp1_unstable=2, inp2_unstable=2, delta_unstable=256
- layer_idx=8, type=LayerType.Linear, total=10, inp1_unstable=0, inp2_unstable=0, delta_unstable=10

Time for backsubstitution: 1.62 seconds

### RS candidates at layer 1
type: RSZ, layer: 1, pos: 52
type: RSZ, layer: 1, pos: 83
type: RSZ, layer: 1, pos: 68
type: RSZ, layer: 1, pos: 199
type: RSZ, layer: 1, pos: 204

Time for candidate selection: 0.00 seconds

### Candidate
type: RSZ, layer: 1, pos: 52

### Relational analysis RSZ of RS_RSZ1_RSZ2_RSZ1_RSZ1_RSZ1

#### Relational analysis RSZ result of RS_RSZ1_RSZ2_RSZ1_RSZ1_RSZ1
Status: Status.UNKNOWN
Output dim: 1, lower bound: -0.0007800, upper bound: 0.0007799
time: 0.69 seconds

### Relational analysis RSZ of RS_RSZ1_RSZ2_RSZ1_RSZ1_RSZ2

#### Relational analysis RSZ result of RS_RSZ1_RSZ2_RSZ1_RSZ1_RSZ2
Status: Status.UNKNOWN
Output dim: 1, lower bound: -0.0007800, upper bound: 0.0007799
time: 0.75 seconds

## BFS RS instance: RS_RSZ1_RSZ2_RSZ1_RSZ2

### Backsubstitution after applying RS history:
0: 0.0007184, 0.0011172, 0.0007184, 0.0011172, -0.0003988, 0.0003988
1: 0.9934075, 0.9943579, 0.9934075, 0.9943579, -0.0009505, 0.0009505
2: -0.0087946, -0.0055215, -0.0087946, -0.0055215, -0.0029943, 0.0029867
3: 0.0035868, 0.0041619, 0.0035868, 0.0041619, -0.0005751, 0.0005751
4: 0.0027809, 0.0053678, 0.0027809, 0.0053678, -0.0025869, 0.0025869
5: 0.0050646, 0.0064029, 0.0050646, 0.0064029, -0.0013382, 0.0013382
6: -0.0021738, -0.0008562, -0.0021738, -0.0008562, -0.0013176, 0.0013176
7: -0.0082275, -0.0074685, -0.0082275, -0.0074685, -0.0007590, 0.0007590
8: 0.0055062, 0.0098066, 0.0055062, 0.0098066, -0.0042235, 0.0042253
9: -0.0036825, -0.0031002, -0.0036825, -0.0031002, -0.0005823, 0.0005823

### Unstable ReLU Count (Linear/Conv2D Layers)
- layer_idx=0, type=LayerType.Linear, total=256, inp1_unstable=9, inp2_unstable=9, delta_unstable=252
- layer_idx=2, type=LayerType.Linear, total=256, inp1_unstable=6, inp2_unstable=6, delta_unstable=256
- layer_idx=4, type=LayerType.Linear, total=256, inp1_unstable=0, inp2_unstable=0, delta_unstable=256
- layer_idx=6, type=LayerType.Linear, total=256, inp1_unstable=2, inp2_unstable=2, delta_unstable=256
- layer_idx=8, type=LayerType.Linear, total=10, inp1_unstable=0, inp2_unstable=0, delta_unstable=10

Time for backsubstitution: 1.65 seconds

### RS candidates at layer 1
type: RSZ, layer: 1, pos: 204
type: RSZ, layer: 1, pos: 199
type: RSZ, layer: 1, pos: 52
type: RSZ, layer: 1, pos: 83
type: RSZ, layer: 1, pos: 68

Time for candidate selection: 0.00 seconds

### Candidate
type: RSZ, layer: 1, pos: 204

### Relational analysis RSZ of RS_RSZ1_RSZ2_RSZ1_RSZ2_RSZ1

#### Relational analysis RSZ result of RS_RSZ1_RSZ2_RSZ1_RSZ2_RSZ1
Status: Status.UNKNOWN
Output dim: 1, lower bound: -0.0007762, upper bound: 0.0007763
time: 0.67 seconds

### Relational analysis RSZ of RS_RSZ1_RSZ2_RSZ1_RSZ2_RSZ2

#### Relational analysis RSZ result of RS_RSZ1_RSZ2_RSZ1_RSZ2_RSZ2
Status: Status.UNKNOWN
Output dim: 1, lower bound: -0.0007763, upper bound: 0.0007762
time: 0.72 seconds

## BFS RS instance: RS_RSZ1_RSZ2_RSZ2_RSZ1

### Backsubstitution after applying RS history:
0: 0.0007184, 0.0011172, 0.0007184, 0.0011172, -0.0003988, 0.0003988
1: 0.9934075, 0.9943579, 0.9934075, 0.9943579, -0.0009505, 0.0009505
2: -0.0087946, -0.0055215, -0.0087946, -0.0055215, -0.0029937, 0.0029930
3: 0.0035868, 0.0041619, 0.0035868, 0.0041619, -0.0005751, 0.0005751
4: 0.0027809, 0.0053678, 0.0027809, 0.0053678, -0.0025869, 0.0025869
5: 0.0050646, 0.0064029, 0.0050646, 0.0064029, -0.0013382, 0.0013382
6: -0.0021738, -0.0008562, -0.0021738, -0.0008562, -0.0013176, 0.0013176
7: -0.0082275, -0.0074685, -0.0082275, -0.0074685, -0.0007590, 0.0007590
8: 0.0055062, 0.0098066, 0.0055062, 0.0098066, -0.0042250, 0.0042251
9: -0.0036825, -0.0031002, -0.0036825, -0.0031002, -0.0005823, 0.0005823

### Unstable ReLU Count (Linear/Conv2D Layers)
- layer_idx=0, type=LayerType.Linear, total=256, inp1_unstable=9, inp2_unstable=9, delta_unstable=252
- layer_idx=2, type=LayerType.Linear, total=256, inp1_unstable=6, inp2_unstable=6, delta_unstable=256
- layer_idx=4, type=LayerType.Linear, total=256, inp1_unstable=0, inp2_unstable=0, delta_unstable=256
- layer_idx=6, type=LayerType.Linear, total=256, inp1_unstable=2, inp2_unstable=2, delta_unstable=256
- layer_idx=8, type=LayerType.Linear, total=10, inp1_unstable=0, inp2_unstable=0, delta_unstable=10

Time for backsubstitution: 1.92 seconds

### RS candidates at layer 1
type: RSZ, layer: 1, pos: 83
type: RSZ, layer: 1, pos: 52
type: RSZ, layer: 1, pos: 198
type: RSZ, layer: 1, pos: 204
type: RSZ, layer: 1, pos: 68

Time for candidate selection: 0.00 seconds

### Candidate
type: RSZ, layer: 1, pos: 83

### Relational analysis RSZ of RS_RSZ1_RSZ2_RSZ2_RSZ1_RSZ1

#### Relational analysis RSZ result of RS_RSZ1_RSZ2_RSZ2_RSZ1_RSZ1
Status: Status.UNKNOWN
Output dim: 1, lower bound: -0.0007567, upper bound: 0.0007569
time: 0.60 seconds

### Relational analysis RSZ of RS_RSZ1_RSZ2_RSZ2_RSZ1_RSZ2

#### Relational analysis RSZ result of RS_RSZ1_RSZ2_RSZ2_RSZ1_RSZ2
Status: Status.UNKNOWN
Output dim: 1, lower bound: -0.0007567, upper bound: 0.0007569
time: 0.68 seconds

## BFS RS instance: RS_RSZ1_RSZ2_RSZ2_RSZ2

### Backsubstitution after applying RS history:
0: 0.0007184, 0.0011172, 0.0007184, 0.0011172, -0.0003988, 0.0003988
1: 0.9934075, 0.9943579, 0.9934075, 0.9943579, -0.0009505, 0.0009505
2: -0.0087946, -0.0055215, -0.0087946, -0.0055215, -0.0029944, 0.0029951
3: 0.0035868, 0.0041619, 0.0035868, 0.0041619, -0.0005751, 0.0005751
4: 0.0027809, 0.0053678, 0.0027809, 0.0053678, -0.0025869, 0.0025869
5: 0.0050646, 0.0064029, 0.0050646, 0.0064029, -0.0013382, 0.0013382
6: -0.0021738, -0.0008562, -0.0021738, -0.0008562, -0.0013176, 0.0013176
7: -0.0082275, -0.0074685, -0.0082275, -0.0074685, -0.0007590, 0.0007590
8: 0.0055062, 0.0098066, 0.0055062, 0.0098066, -0.0042255, 0.0042253
9: -0.0036825, -0.0031002, -0.0036825, -0.0031002, -0.0005823, 0.0005823

### Unstable ReLU Count (Linear/Conv2D Layers)
- layer_idx=0, type=LayerType.Linear, total=256, inp1_unstable=9, inp2_unstable=9, delta_unstable=252
- layer_idx=2, type=LayerType.Linear, total=256, inp1_unstable=6, inp2_unstable=6, delta_unstable=256
- layer_idx=4, type=LayerType.Linear, total=256, inp1_unstable=0, inp2_unstable=0, delta_unstable=256
- layer_idx=6, type=LayerType.Linear, total=256, inp1_unstable=2, inp2_unstable=2, delta_unstable=256
- layer_idx=8, type=LayerType.Linear, total=10, inp1_unstable=0, inp2_unstable=0, delta_unstable=10

Time for backsubstitution: 1.73 seconds

### RS candidates at layer 1
type: RSZ, layer: 1, pos: 198
type: RSZ, layer: 1, pos: 204
type: RSZ, layer: 1, pos: 68
type: RSZ, layer: 1, pos: 52
type: RSZ, layer: 1, pos: 83

Time for candidate selection: 0.00 seconds

### Candidate
type: RSZ, layer: 1, pos: 198

### Relational analysis RSZ of RS_RSZ1_RSZ2_RSZ2_RSZ2_RSZ1

#### Relational analysis RSZ result of RS_RSZ1_RSZ2_RSZ2_RSZ2_RSZ1
Status: Status.UNKNOWN
Output dim: 1, lower bound: -0.0007131, upper bound: 0.0007131
time: 0.66 seconds

### Relational analysis RSZ of RS_RSZ1_RSZ2_RSZ2_RSZ2_RSZ2

#### Relational analysis RSZ result of RS_RSZ1_RSZ2_RSZ2_RSZ2_RSZ2
Status: Status.UNKNOWN
Output dim: 1, lower bound: -0.0007131, upper bound: 0.0007131
time: 0.61 seconds

## BFS RS instance: RS_RSZ2_RSZ1_RSZ1_RSZ1

### Backsubstitution after applying RS history:
0: 0.0007184, 0.0011172, 0.0007184, 0.0011172, -0.0003988, 0.0003988
1: 0.9934075, 0.9943579, 0.9934075, 0.9943579, -0.0009505, 0.0009505
2: -0.0087946, -0.0055215, -0.0087946, -0.0055215, -0.0029888, 0.0029860
3: 0.0035868, 0.0041619, 0.0035868, 0.0041619, -0.0005751, 0.0005751
4: 0.0027809, 0.0053678, 0.0027809, 0.0053678, -0.0025869, 0.0025869
5: 0.0050646, 0.0064029, 0.0050646, 0.0064029, -0.0013382, 0.0013382
6: -0.0021738, -0.0008562, -0.0021738, -0.0008562, -0.0013176, 0.0013176
7: -0.0082275, -0.0074685, -0.0082275, -0.0074685, -0.0007590, 0.0007590
8: 0.0055062, 0.0098066, 0.0055062, 0.0098066, -0.0042223, 0.0042231
9: -0.0036825, -0.0031002, -0.0036825, -0.0031002, -0.0005823, 0.0005823

### Unstable ReLU Count (Linear/Conv2D Layers)
- layer_idx=0, type=LayerType.Linear, total=256, inp1_unstable=9, inp2_unstable=9, delta_unstable=252
- layer_idx=2, type=LayerType.Linear, total=256, inp1_unstable=6, inp2_unstable=6, delta_unstable=256
- layer_idx=4, type=LayerType.Linear, total=256, inp1_unstable=0, inp2_unstable=0, delta_unstable=256
- layer_idx=6, type=LayerType.Linear, total=256, inp1_unstable=2, inp2_unstable=2, delta_unstable=256
- layer_idx=8, type=LayerType.Linear, total=10, inp1_unstable=0, inp2_unstable=0, delta_unstable=10

Time for backsubstitution: 1.73 seconds

### RS candidates at layer 1
type: RSZ, layer: 1, pos: 52
type: RSZ, layer: 1, pos: 136
type: RSZ, layer: 1, pos: 199
type: RSZ, layer: 1, pos: 198
type: RSZ, layer: 1, pos: 176

Time for candidate selection: 0.00 seconds

### Candidate
type: RSZ, layer: 1, pos: 52

### Relational analysis RSZ of RS_RSZ2_RSZ1_RSZ1_RSZ1_RSZ1

#### Relational analysis RSZ result of RS_RSZ2_RSZ1_RSZ1_RSZ1_RSZ1
Status: Status.UNKNOWN
Output dim: 1, lower bound: -0.0008028, upper bound: 0.0008052
time: 0.63 seconds

### Relational analysis RSZ of RS_RSZ2_RSZ1_RSZ1_RSZ1_RSZ2

#### Relational analysis RSZ result of RS_RSZ2_RSZ1_RSZ1_RSZ1_RSZ2
Status: Status.UNKNOWN
Output dim: 1, lower bound: -0.0008018, upper bound: 0.0008089
time: 0.65 seconds

## BFS RS instance: RS_RSZ2_RSZ1_RSZ1_RSZ2

### Backsubstitution after applying RS history:
0: 0.0007184, 0.0011172, 0.0007184, 0.0011172, -0.0003988, 0.0003988
1: 0.9934075, 0.9943579, 0.9934075, 0.9943579, -0.0009505, 0.0009505
2: -0.0087946, -0.0055215, -0.0087946, -0.0055215, -0.0029885, 0.0029863
3: 0.0035868, 0.0041619, 0.0035868, 0.0041619, -0.0005751, 0.0005751
4: 0.0027809, 0.0053678, 0.0027809, 0.0053678, -0.0025869, 0.0025869
5: 0.0050646, 0.0064029, 0.0050646, 0.0064029, -0.0013382, 0.0013382
6: -0.0021738, -0.0008562, -0.0021738, -0.0008562, -0.0013176, 0.0013176
7: -0.0082275, -0.0074685, -0.0082275, -0.0074685, -0.0007590, 0.0007590
8: 0.0055062, 0.0098066, 0.0055062, 0.0098066, -0.0042224, 0.0042230
9: -0.0036825, -0.0031002, -0.0036825, -0.0031002, -0.0005823, 0.0005823

### Unstable ReLU Count (Linear/Conv2D Layers)
- layer_idx=0, type=LayerType.Linear, total=256, inp1_unstable=9, inp2_unstable=9, delta_unstable=252
- layer_idx=2, type=LayerType.Linear, total=256, inp1_unstable=6, inp2_unstable=6, delta_unstable=256
- layer_idx=4, type=LayerType.Linear, total=256, inp1_unstable=0, inp2_unstable=0, delta_unstable=256
- layer_idx=6, type=LayerType.Linear, total=256, inp1_unstable=2, inp2_unstable=2, delta_unstable=256
- layer_idx=8, type=LayerType.Linear, total=10, inp1_unstable=0, inp2_unstable=0, delta_unstable=10

Time for backsubstitution: 1.71 seconds

### RS candidates at layer 1
type: RSZ, layer: 1, pos: 198
type: RSZ, layer: 1, pos: 136
type: RSZ, layer: 1, pos: 176
type: RSZ, layer: 1, pos: 52
type: RSZ, layer: 1, pos: 199

Time for candidate selection: 0.00 seconds

### Candidate
type: RSZ, layer: 1, pos: 198

### Relational analysis RSZ of RS_RSZ2_RSZ1_RSZ1_RSZ2_RSZ1

#### Relational analysis RSZ result of RS_RSZ2_RSZ1_RSZ1_RSZ2_RSZ1
Status: Status.UNKNOWN
Output dim: 1, lower bound: -0.0007810, upper bound: 0.0007812
time: 0.72 seconds

### Relational analysis RSZ of RS_RSZ2_RSZ1_RSZ1_RSZ2_RSZ2

#### Relational analysis RSZ result of RS_RSZ2_RSZ1_RSZ1_RSZ2_RSZ2
Status: Status.UNKNOWN
Output dim: 1, lower bound: -0.0007810, upper bound: 0.0007812
time: 0.73 seconds

## BFS RS instance: RS_RSZ2_RSZ1_RSZ2_RSZ1

### Backsubstitution after applying RS history:
0: 0.0007184, 0.0011172, 0.0007184, 0.0011172, -0.0003988, 0.0003988
1: 0.9934075, 0.9943579, 0.9934075, 0.9943579, -0.0009505, 0.0009505
2: -0.0087946, -0.0055215, -0.0087946, -0.0055215, -0.0029883, 0.0029867
3: 0.0035868, 0.0041619, 0.0035868, 0.0041619, -0.0005751, 0.0005751
4: 0.0027809, 0.0053678, 0.0027809, 0.0053678, -0.0025869, 0.0025869
5: 0.0050646, 0.0064029, 0.0050646, 0.0064029, -0.0013382, 0.0013382
6: -0.0021738, -0.0008562, -0.0021738, -0.0008562, -0.0013176, 0.0013176
7: -0.0082275, -0.0074685, -0.0082275, -0.0074685, -0.0007590, 0.0007590
8: 0.0055062, 0.0098066, 0.0055062, 0.0098066, -0.0042224, 0.0042229
9: -0.0036825, -0.0031002, -0.0036825, -0.0031002, -0.0005823, 0.0005823

### Unstable ReLU Count (Linear/Conv2D Layers)
- layer_idx=0, type=LayerType.Linear, total=256, inp1_unstable=9, inp2_unstable=9, delta_unstable=252
- layer_idx=2, type=LayerType.Linear, total=256, inp1_unstable=6, inp2_unstable=6, delta_unstable=256
- layer_idx=4, type=LayerType.Linear, total=256, inp1_unstable=0, inp2_unstable=0, delta_unstable=256
- layer_idx=6, type=LayerType.Linear, total=256, inp1_unstable=2, inp2_unstable=2, delta_unstable=256
- layer_idx=8, type=LayerType.Linear, total=10, inp1_unstable=0, inp2_unstable=0, delta_unstable=10

Time for backsubstitution: 1.76 seconds

### RS candidates at layer 1
type: RSZ, layer: 1, pos: 199
type: RSZ, layer: 1, pos: 83
type: RSZ, layer: 1, pos: 198
type: RSZ, layer: 1, pos: 136
type: RSZ, layer: 1, pos: 52

Time for candidate selection: 0.00 seconds

### Candidate
type: RSZ, layer: 1, pos: 199

### Relational analysis RSZ of RS_RSZ2_RSZ1_RSZ2_RSZ1_RSZ1

#### Relational analysis RSZ result of RS_RSZ2_RSZ1_RSZ2_RSZ1_RSZ1
Status: Status.UNKNOWN
Output dim: 1, lower bound: -0.0007600, upper bound: 0.0007602
time: 0.70 seconds

### Relational analysis RSZ of RS_RSZ2_RSZ1_RSZ2_RSZ1_RSZ2

#### Relational analysis RSZ result of RS_RSZ2_RSZ1_RSZ2_RSZ1_RSZ2
Status: Status.UNKNOWN
Output dim: 1, lower bound: -0.0007600, upper bound: 0.0007602
time: 0.68 seconds

## BFS RS instance: RS_RSZ2_RSZ1_RSZ2_RSZ2

### Backsubstitution after applying RS history:
0: 0.0007184, 0.0011172, 0.0007184, 0.0011172, -0.0003988, 0.0003988
1: 0.9934075, 0.9943579, 0.9934075, 0.9943579, -0.0009505, 0.0009505
2: -0.0087946, -0.0055215, -0.0087946, -0.0055215, -0.0029877, 0.0029871
3: 0.0035868, 0.0041619, 0.0035868, 0.0041619, -0.0005751, 0.0005751
4: 0.0027809, 0.0053678, 0.0027809, 0.0053678, -0.0025869, 0.0025869
5: 0.0050646, 0.0064029, 0.0050646, 0.0064029, -0.0013382, 0.0013382
6: -0.0021738, -0.0008562, -0.0021738, -0.0008562, -0.0013176, 0.0013176
7: -0.0082275, -0.0074685, -0.0082275, -0.0074685, -0.0007590, 0.0007590
8: 0.0055062, 0.0098066, 0.0055062, 0.0098066, -0.0042226, 0.0042227
9: -0.0036825, -0.0031002, -0.0036825, -0.0031002, -0.0005823, 0.0005823

### Unstable ReLU Count (Linear/Conv2D Layers)
- layer_idx=0, type=LayerType.Linear, total=256, inp1_unstable=9, inp2_unstable=9, delta_unstable=252
- layer_idx=2, type=LayerType.Linear, total=256, inp1_unstable=6, inp2_unstable=6, delta_unstable=256
- layer_idx=4, type=LayerType.Linear, total=256, inp1_unstable=0, inp2_unstable=0, delta_unstable=256
- layer_idx=6, type=LayerType.Linear, total=256, inp1_unstable=2, inp2_unstable=2, delta_unstable=256
- layer_idx=8, type=LayerType.Linear, total=10, inp1_unstable=0, inp2_unstable=0, delta_unstable=10

Time for backsubstitution: 1.59 seconds

### RS candidates at layer 1
type: RSZ, layer: 1, pos: 83
type: RSZ, layer: 1, pos: 136
type: RSZ, layer: 1, pos: 52
type: RSZ, layer: 1, pos: 199
type: RSZ, layer: 1, pos: 198

Time for candidate selection: 0.00 seconds

### Candidate
type: RSZ, layer: 1, pos: 83

### Relational analysis RSZ of RS_RSZ2_RSZ1_RSZ2_RSZ2_RSZ1

#### Relational analysis RSZ result of RS_RSZ2_RSZ1_RSZ2_RSZ2_RSZ1
Status: Status.UNKNOWN
Output dim: 1, lower bound: -0.0008031, upper bound: 0.0008052
time: 0.69 seconds

### Relational analysis RSZ of RS_RSZ2_RSZ1_RSZ2_RSZ2_RSZ2

#### Relational analysis RSZ result of RS_RSZ2_RSZ1_RSZ2_RSZ2_RSZ2
Status: Status.UNKNOWN
Output dim: 1, lower bound: -0.0008030, upper bound: 0.0008051
time: 0.69 seconds

## BFS RS instance: RS_RSZ2_RSZ2_RSZ1_RSZ1

### Backsubstitution after applying RS history:
0: 0.0007184, 0.0011172, 0.0007184, 0.0011172, -0.0003988, 0.0003988
1: 0.9934075, 0.9943579, 0.9934075, 0.9943579, -0.0009505, 0.0009505
2: -0.0087946, -0.0055215, -0.0087946, -0.0055215, -0.0029864, 0.0029884
3: 0.0035868, 0.0041619, 0.0035868, 0.0041619, -0.0005751, 0.0005751
4: 0.0027809, 0.0053678, 0.0027809, 0.0053678, -0.0025869, 0.0025869
5: 0.0050646, 0.0064029, 0.0050646, 0.0064029, -0.0013382, 0.0013382
6: -0.0021738, -0.0008562, -0.0021738, -0.0008562, -0.0013176, 0.0013176
7: -0.0082275, -0.0074685, -0.0082275, -0.0074685, -0.0007590, 0.0007590
8: 0.0055062, 0.0098066, 0.0055062, 0.0098066, -0.0042228, 0.0042223
9: -0.0036825, -0.0031002, -0.0036825, -0.0031002, -0.0005823, 0.0005823

### Unstable ReLU Count (Linear/Conv2D Layers)
- layer_idx=0, type=LayerType.Linear, total=256, inp1_unstable=9, inp2_unstable=9, delta_unstable=252
- layer_idx=2, type=LayerType.Linear, total=256, inp1_unstable=6, inp2_unstable=6, delta_unstable=256
- layer_idx=4, type=LayerType.Linear, total=256, inp1_unstable=0, inp2_unstable=0, delta_unstable=256
- layer_idx=6, type=LayerType.Linear, total=256, inp1_unstable=2, inp2_unstable=2, delta_unstable=256
- layer_idx=8, type=LayerType.Linear, total=10, inp1_unstable=0, inp2_unstable=0, delta_unstable=10

Time for backsubstitution: 1.76 seconds

### RS candidates at layer 1
type: RSZ, layer: 1, pos: 136
type: RSZ, layer: 1, pos: 83
type: RSZ, layer: 1, pos: 198
type: RSZ, layer: 1, pos: 176
type: RSZ, layer: 1, pos: 199

Time for candidate selection: 0.00 seconds

### Candidate
type: RSZ, layer: 1, pos: 136

### Relational analysis RSZ of RS_RSZ2_RSZ2_RSZ1_RSZ1_RSZ1

#### Relational analysis RSZ result of RS_RSZ2_RSZ2_RSZ1_RSZ1_RSZ1
Status: Status.UNKNOWN
Output dim: 1, lower bound: -0.0008005, upper bound: 0.0007968
time: 0.64 seconds

### Relational analysis RSZ of RS_RSZ2_RSZ2_RSZ1_RSZ1_RSZ2

#### Relational analysis RSZ result of RS_RSZ2_RSZ2_RSZ1_RSZ1_RSZ2
Status: Status.UNKNOWN
Output dim: 1, lower bound: -0.0008005, upper bound: 0.0007968
time: 0.65 seconds

## BFS RS instance: RS_RSZ2_RSZ2_RSZ1_RSZ2

### Backsubstitution after applying RS history:
0: 0.0007184, 0.0011172, 0.0007184, 0.0011172, -0.0003988, 0.0003988
1: 0.9934075, 0.9943579, 0.9934075, 0.9943579, -0.0009505, 0.0009505
2: -0.0087946, -0.0055215, -0.0087946, -0.0055215, -0.0029858, 0.0029890
3: 0.0035868, 0.0041619, 0.0035868, 0.0041619, -0.0005751, 0.0005751
4: 0.0027809, 0.0053678, 0.0027809, 0.0053678, -0.0025869, 0.0025869
5: 0.0050646, 0.0064029, 0.0050646, 0.0064029, -0.0013382, 0.0013382
6: -0.0021738, -0.0008562, -0.0021738, -0.0008562, -0.0013176, 0.0013176
7: -0.0082275, -0.0074685, -0.0082275, -0.0074685, -0.0007590, 0.0007590
8: 0.0055062, 0.0098066, 0.0055062, 0.0098066, -0.0042230, 0.0042221
9: -0.0036825, -0.0031002, -0.0036825, -0.0031002, -0.0005823, 0.0005823

### Unstable ReLU Count (Linear/Conv2D Layers)
- layer_idx=0, type=LayerType.Linear, total=256, inp1_unstable=9, inp2_unstable=9, delta_unstable=252
- layer_idx=2, type=LayerType.Linear, total=256, inp1_unstable=6, inp2_unstable=6, delta_unstable=256
- layer_idx=4, type=LayerType.Linear, total=256, inp1_unstable=0, inp2_unstable=0, delta_unstable=256
- layer_idx=6, type=LayerType.Linear, total=256, inp1_unstable=2, inp2_unstable=2, delta_unstable=256
- layer_idx=8, type=LayerType.Linear, total=10, inp1_unstable=0, inp2_unstable=0, delta_unstable=10

Time for backsubstitution: 1.69 seconds

### RS candidates at layer 1
type: RSZ, layer: 1, pos: 136
type: RSZ, layer: 1, pos: 83
type: RSZ, layer: 1, pos: 199
type: RSZ, layer: 1, pos: 176
type: RSZ, layer: 1, pos: 198

Time for candidate selection: 0.00 seconds

### Candidate
type: RSZ, layer: 1, pos: 136

### Relational analysis RSZ of RS_RSZ2_RSZ2_RSZ1_RSZ2_RSZ1

#### Relational analysis RSZ result of RS_RSZ2_RSZ2_RSZ1_RSZ2_RSZ1
Status: Status.UNKNOWN
Output dim: 1, lower bound: -0.0008015, upper bound: 0.0007956
time: 0.67 seconds

### Relational analysis RSZ of RS_RSZ2_RSZ2_RSZ1_RSZ2_RSZ2

#### Relational analysis RSZ result of RS_RSZ2_RSZ2_RSZ1_RSZ2_RSZ2
Status: Status.UNKNOWN
Output dim: 1, lower bound: -0.0008013, upper bound: 0.0007956
time: 0.65 seconds

## BFS RS instance: RS_RSZ2_RSZ2_RSZ2_RSZ1

### Backsubstitution after applying RS history:
0: 0.0007184, 0.0011172, 0.0007184, 0.0011172, -0.0003988, 0.0003988
1: 0.9934075, 0.9943579, 0.9934075, 0.9943579, -0.0009505, 0.0009505
2: -0.0087946, -0.0055215, -0.0087946, -0.0055215, -0.0029824, 0.0029838
3: 0.0035868, 0.0041619, 0.0035868, 0.0041619, -0.0005751, 0.0005751
4: 0.0027809, 0.0053678, 0.0027809, 0.0053678, -0.0025869, 0.0025869
5: 0.0050646, 0.0064029, 0.0050646, 0.0064029, -0.0013382, 0.0013382
6: -0.0021738, -0.0008562, -0.0021738, -0.0008562, -0.0013176, 0.0013176
7: -0.0082275, -0.0074685, -0.0082275, -0.0074685, -0.0007590, 0.0007590
8: 0.0055062, 0.0098066, 0.0055062, 0.0098066, -0.0042218, 0.0042214
9: -0.0036825, -0.0031002, -0.0036825, -0.0031002, -0.0005823, 0.0005823

### Unstable ReLU Count (Linear/Conv2D Layers)
- layer_idx=0, type=LayerType.Linear, total=256, inp1_unstable=9, inp2_unstable=9, delta_unstable=252
- layer_idx=2, type=LayerType.Linear, total=256, inp1_unstable=6, inp2_unstable=6, delta_unstable=256
- layer_idx=4, type=LayerType.Linear, total=256, inp1_unstable=0, inp2_unstable=0, delta_unstable=256
- layer_idx=6, type=LayerType.Linear, total=256, inp1_unstable=2, inp2_unstable=2, delta_unstable=256
- layer_idx=8, type=LayerType.Linear, total=10, inp1_unstable=0, inp2_unstable=0, delta_unstable=10

Time for backsubstitution: 1.67 seconds

### RS candidates at layer 1
type: RSZ, layer: 1, pos: 176
type: RSZ, layer: 1, pos: 83
type: RSZ, layer: 1, pos: 199
type: RSZ, layer: 1, pos: 204
type: RSZ, layer: 1, pos: 136

Time for candidate selection: 0.00 seconds

### Candidate
type: RSZ, layer: 1, pos: 176

### Relational analysis RSZ of RS_RSZ2_RSZ2_RSZ2_RSZ1_RSZ1

#### Relational analysis RSZ result of RS_RSZ2_RSZ2_RSZ2_RSZ1_RSZ1
Status: Status.UNKNOWN
Output dim: 1, lower bound: -0.0007814, upper bound: 0.0007811
time: 0.65 seconds

### Relational analysis RSZ of RS_RSZ2_RSZ2_RSZ2_RSZ1_RSZ2

#### Relational analysis RSZ result of RS_RSZ2_RSZ2_RSZ2_RSZ1_RSZ2
Status: Status.UNKNOWN
Output dim: 1, lower bound: -0.0007814, upper bound: 0.0007811
time: 0.72 seconds

## BFS RS instance: RS_RSZ2_RSZ2_RSZ2_RSZ2

### Backsubstitution after applying RS history:
0: 0.0007184, 0.0011172, 0.0007184, 0.0011172, -0.0003988, 0.0003988
1: 0.9934075, 0.9943579, 0.9934075, 0.9943579, -0.0009505, 0.0009505
2: -0.0087946, -0.0055215, -0.0087946, -0.0055215, -0.0029929, 0.0029830
3: 0.0035868, 0.0041619, 0.0035868, 0.0041619, -0.0005751, 0.0005751
4: 0.0027809, 0.0053678, 0.0027809, 0.0053678, -0.0025869, 0.0025869
5: 0.0050646, 0.0064029, 0.0050646, 0.0064029, -0.0013382, 0.0013382
6: -0.0021738, -0.0008562, -0.0021738, -0.0008562, -0.0013176, 0.0013176
7: -0.0082275, -0.0074685, -0.0082275, -0.0074685, -0.0007590, 0.0007590
8: 0.0055062, 0.0098066, 0.0055062, 0.0098066, -0.0042216, 0.0042239
9: -0.0036825, -0.0031002, -0.0036825, -0.0031002, -0.0005823, 0.0005823

### Unstable ReLU Count (Linear/Conv2D Layers)
- layer_idx=0, type=LayerType.Linear, total=256, inp1_unstable=9, inp2_unstable=9, delta_unstable=252
- layer_idx=2, type=LayerType.Linear, total=256, inp1_unstable=6, inp2_unstable=6, delta_unstable=256
- layer_idx=4, type=LayerType.Linear, total=256, inp1_unstable=0, inp2_unstable=0, delta_unstable=256
- layer_idx=6, type=LayerType.Linear, total=256, inp1_unstable=2, inp2_unstable=2, delta_unstable=256
- layer_idx=8, type=LayerType.Linear, total=10, inp1_unstable=0, inp2_unstable=0, delta_unstable=10

Time for backsubstitution: 1.64 seconds

### RS candidates at layer 1
type: RSZ, layer: 1, pos: 136
type: RSZ, layer: 1, pos: 204
type: RSZ, layer: 1, pos: 199
type: RSZ, layer: 1, pos: 176
type: RSZ, layer: 1, pos: 83

Time for candidate selection: 0.00 seconds

### Candidate
type: RSZ, layer: 1, pos: 136

### Relational analysis RSZ of RS_RSZ2_RSZ2_RSZ2_RSZ2_RSZ1

#### Relational analysis RSZ result of RS_RSZ2_RSZ2_RSZ2_RSZ2_RSZ1
Status: Status.UNKNOWN
Output dim: 1, lower bound: -0.0007853, upper bound: 0.0007835
time: 0.68 seconds

### Relational analysis RSZ of RS_RSZ2_RSZ2_RSZ2_RSZ2_RSZ2

#### Relational analysis RSZ result of RS_RSZ2_RSZ2_RSZ2_RSZ2_RSZ2
Status: Status.UNKNOWN
Output dim: 1, lower bound: -0.0007853, upper bound: 0.0007836
time: 0.68 seconds

## Summary of splitting (split count: 4)
- Time for RS candidates: 3.02 seconds
RS_RSZ1_RSZ1_RSZ1_RSZ1_RSZ1, status: Status.UNKNOWN, split count: 5, time: 3.02
Output dim: 1, lower bound: -0.0007983, upper bound: 0.0007961
RS_RSZ1_RSZ1_RSZ1_RSZ1_RSZ2, status: Status.UNKNOWN, split count: 5, time: 3.02
Output dim: 1, lower bound: -0.0007959, upper bound: 0.0007987
RS_RSZ1_RSZ1_RSZ1_RSZ2_RSZ1, status: Status.UNKNOWN, split count: 5, time: 3.02
Output dim: 1, lower bound: -0.0007992, upper bound: 0.0007998
RS_RSZ1_RSZ1_RSZ1_RSZ2_RSZ2, status: Status.UNKNOWN, split count: 5, time: 3.02
Output dim: 1, lower bound: -0.0007991, upper bound: 0.0007999
RS_RSZ1_RSZ1_RSZ2_RSZ1_RSZ1, status: Status.UNKNOWN, split count: 5, time: 3.02
Output dim: 1, lower bound: -0.0007766, upper bound: 0.0007766
RS_RSZ1_RSZ1_RSZ2_RSZ1_RSZ2, status: Status.UNKNOWN, split count: 5, time: 3.02
Output dim: 1, lower bound: -0.0007766, upper bound: 0.0007766
RS_RSZ1_RSZ1_RSZ2_RSZ2_RSZ1, status: Status.UNKNOWN, split count: 5, time: 3.02
Output dim: 1, lower bound: -0.0007766, upper bound: 0.0007766
RS_RSZ1_RSZ1_RSZ2_RSZ2_RSZ2, status: Status.UNKNOWN, split count: 5, time: 3.02
Output dim: 1, lower bound: -0.0007766, upper bound: 0.0007766
RS_RSZ1_RSZ2_RSZ1_RSZ1_RSZ1, status: Status.UNKNOWN, split count: 5, time: 3.02
Output dim: 1, lower bound: -0.0007800, upper bound: 0.0007799
RS_RSZ1_RSZ2_RSZ1_RSZ1_RSZ2, status: Status.UNKNOWN, split count: 5, time: 3.02
Output dim: 1, lower bound: -0.0007800, upper bound: 0.0007799
RS_RSZ1_RSZ2_RSZ1_RSZ2_RSZ1, status: Status.UNKNOWN, split count: 5, time: 3.02
Output dim: 1, lower bound: -0.0007762, upper bound: 0.0007763
RS_RSZ1_RSZ2_RSZ1_RSZ2_RSZ2, status: Status.UNKNOWN, split count: 5, time: 3.02
Output dim: 1, lower bound: -0.0007763, upper bound: 0.0007762
RS_RSZ1_RSZ2_RSZ2_RSZ1_RSZ1, status: Status.UNKNOWN, split count: 5, time: 3.02
Output dim: 1, lower bound: -0.0007567, upper bound: 0.0007569
RS_RSZ1_RSZ2_RSZ2_RSZ1_RSZ2, status: Status.UNKNOWN, split count: 5, time: 3.02
Output dim: 1, lower bound: -0.0007567, upper bound: 0.0007569
RS_RSZ1_RSZ2_RSZ2_RSZ2_RSZ1, status: Status.UNKNOWN, split count: 5, time: 3.02
Output dim: 1, lower bound: -0.0007131, upper bound: 0.0007131
RS_RSZ1_RSZ2_RSZ2_RSZ2_RSZ2, status: Status.UNKNOWN, split count: 5, time: 3.02
Output dim: 1, lower bound: -0.0007131, upper bound: 0.0007131
RS_RSZ2_RSZ1_RSZ1_RSZ1_RSZ1, status: Status.UNKNOWN, split count: 5, time: 3.02
Output dim: 1, lower bound: -0.0008028, upper bound: 0.0008052
RS_RSZ2_RSZ1_RSZ1_RSZ1_RSZ2, status: Status.UNKNOWN, split count: 5, time: 3.02
Output dim: 1, lower bound: -0.0008018, upper bound: 0.0008089
RS_RSZ2_RSZ1_RSZ1_RSZ2_RSZ1, status: Status.UNKNOWN, split count: 5, time: 3.02
Output dim: 1, lower bound: -0.0007810, upper bound: 0.0007812
RS_RSZ2_RSZ1_RSZ1_RSZ2_RSZ2, status: Status.UNKNOWN, split count: 5, time: 3.02
Output dim: 1, lower bound: -0.0007810, upper bound: 0.0007812
RS_RSZ2_RSZ1_RSZ2_RSZ1_RSZ1, status: Status.UNKNOWN, split count: 5, time: 3.02
Output dim: 1, lower bound: -0.0007600, upper bound: 0.0007602
RS_RSZ2_RSZ1_RSZ2_RSZ1_RSZ2, status: Status.UNKNOWN, split count: 5, time: 3.02
Output dim: 1, lower bound: -0.0007600, upper bound: 0.0007602
RS_RSZ2_RSZ1_RSZ2_RSZ2_RSZ1, status: Status.UNKNOWN, split count: 5, time: 3.02
Output dim: 1, lower bound: -0.0008031, upper bound: 0.0008052
RS_RSZ2_RSZ1_RSZ2_RSZ2_RSZ2, status: Status.UNKNOWN, split count: 5, time: 3.02
Output dim: 1, lower bound: -0.0008030, upper bound: 0.0008051
RS_RSZ2_RSZ2_RSZ1_RSZ1_RSZ1, status: Status.UNKNOWN, split count: 5, time: 3.02
Output dim: 1, lower bound: -0.0008005, upper bound: 0.0007968
RS_RSZ2_RSZ2_RSZ1_RSZ1_RSZ2, status: Status.UNKNOWN, split count: 5, time: 3.02
Output dim: 1, lower bound: -0.0008005, upper bound: 0.0007968
RS_RSZ2_RSZ2_RSZ1_RSZ2_RSZ1, status: Status.UNKNOWN, split count: 5, time: 3.02
Output dim: 1, lower bound: -0.0008015, upper bound: 0.0007956
RS_RSZ2_RSZ2_RSZ1_RSZ2_RSZ2, status: Status.UNKNOWN, split count: 5, time: 3.02
Output dim: 1, lower bound: -0.0008013, upper bound: 0.0007956
RS_RSZ2_RSZ2_RSZ2_RSZ1_RSZ1, status: Status.UNKNOWN, split count: 5, time: 3.02
Output dim: 1, lower bound: -0.0007814, upper bound: 0.0007811
RS_RSZ2_RSZ2_RSZ2_RSZ1_RSZ2, status: Status.UNKNOWN, split count: 5, time: 3.02
Output dim: 1, lower bound: -0.0007814, upper bound: 0.0007811
RS_RSZ2_RSZ2_RSZ2_RSZ2_RSZ1, status: Status.UNKNOWN, split count: 5, time: 3.02
Output dim: 1, lower bound: -0.0007853, upper bound: 0.0007835
RS_RSZ2_RSZ2_RSZ2_RSZ2_RSZ2, status: Status.UNKNOWN, split count: 5, time: 3.02
Output dim: 1, lower bound: -0.0007853, upper bound: 0.0007836

## BFS RS instance: RS_RSZ1_RSZ1_RSZ1_RSZ1_RSZ1

### Backsubstitution after applying RS history:
0: 0.0007184, 0.0011172, 0.0007184, 0.0011172, -0.0003988, 0.0003988
1: 0.9934075, 0.9943579, 0.9934075, 0.9943579, -0.0009505, 0.0009505
2: -0.0087946, -0.0055215, -0.0087946, -0.0055215, -0.0029863, 0.0029880
3: 0.0035868, 0.0041619, 0.0035868, 0.0041619, -0.0005751, 0.0005751
4: 0.0027809, 0.0053678, 0.0027809, 0.0053678, -0.0025869, 0.0025869
5: 0.0050646, 0.0064029, 0.0050646, 0.0064029, -0.0013382, 0.0013382
6: -0.0021738, -0.0008562, -0.0021738, -0.0008562, -0.0013176, 0.0013176
7: -0.0082275, -0.0074685, -0.0082275, -0.0074685, -0.0007590, 0.0007590
8: 0.0055062, 0.0098066, 0.0055062, 0.0098066, -0.0042236, 0.0042231
9: -0.0036825, -0.0031002, -0.0036825, -0.0031002, -0.0005823, 0.0005823

### Unstable ReLU Count (Linear/Conv2D Layers)
- layer_idx=0, type=LayerType.Linear, total=256, inp1_unstable=9, inp2_unstable=9, delta_unstable=251
- layer_idx=2, type=LayerType.Linear, total=256, inp1_unstable=6, inp2_unstable=6, delta_unstable=256
- layer_idx=4, type=LayerType.Linear, total=256, inp1_unstable=0, inp2_unstable=0, delta_unstable=256
- layer_idx=6, type=LayerType.Linear, total=256, inp1_unstable=2, inp2_unstable=2, delta_unstable=256
- layer_idx=8, type=LayerType.Linear, total=10, inp1_unstable=0, inp2_unstable=0, delta_unstable=10

Time for backsubstitution: 1.76 seconds

### RS candidates at layer 1
type: RSZ, layer: 1, pos: 199
type: RSZ, layer: 1, pos: 198
type: RSZ, layer: 1, pos: 68
type: RSZ, layer: 1, pos: 83

Time for candidate selection: 0.00 seconds

### Candidate
type: RSZ, layer: 1, pos: 199

### Relational analysis RSZ of RS_RSZ1_RSZ1_RSZ1_RSZ1_RSZ1_RSZ1

#### Relational analysis RSZ result of RS_RSZ1_RSZ1_RSZ1_RSZ1_RSZ1_RSZ1
Status: Status.UNKNOWN
Output dim: 1, lower bound: -0.0007589, upper bound: 0.0007582
time: 0.65 seconds

### Relational analysis RSZ of RS_RSZ1_RSZ1_RSZ1_RSZ1_RSZ1_RSZ2

#### Relational analysis RSZ result of RS_RSZ1_RSZ1_RSZ1_RSZ1_RSZ1_RSZ2
Status: Status.UNKNOWN
Output dim: 1, lower bound: -0.0007589, upper bound: 0.0007582
time: 0.68 seconds

## BFS RS instance: RS_RSZ1_RSZ1_RSZ1_RSZ1_RSZ2

### Backsubstitution after applying RS history:
0: 0.0007184, 0.0011172, 0.0007184, 0.0011172, -0.0003988, 0.0003988
1: 0.9934075, 0.9943579, 0.9934075, 0.9943579, -0.0009505, 0.0009505
2: -0.0087946, -0.0055215, -0.0087946, -0.0055215, -0.0029871, 0.0029871
3: 0.0035868, 0.0041619, 0.0035868, 0.0041619, -0.0005751, 0.0005751
4: 0.0027809, 0.0053678, 0.0027809, 0.0053678, -0.0025869, 0.0025869
5: 0.0050646, 0.0064029, 0.0050646, 0.0064029, -0.0013382, 0.0013382
6: -0.0021738, -0.0008562, -0.0021738, -0.0008562, -0.0013176, 0.0013176
7: -0.0082275, -0.0074685, -0.0082275, -0.0074685, -0.0007590, 0.0007590
8: 0.0055062, 0.0098066, 0.0055062, 0.0098066, -0.0042234, 0.0042234
9: -0.0036825, -0.0031002, -0.0036825, -0.0031002, -0.0005823, 0.0005823

### Unstable ReLU Count (Linear/Conv2D Layers)
- layer_idx=0, type=LayerType.Linear, total=256, inp1_unstable=9, inp2_unstable=9, delta_unstable=251
- layer_idx=2, type=LayerType.Linear, total=256, inp1_unstable=6, inp2_unstable=6, delta_unstable=256
- layer_idx=4, type=LayerType.Linear, total=256, inp1_unstable=0, inp2_unstable=0, delta_unstable=256
- layer_idx=6, type=LayerType.Linear, total=256, inp1_unstable=2, inp2_unstable=2, delta_unstable=256
- layer_idx=8, type=LayerType.Linear, total=10, inp1_unstable=0, inp2_unstable=0, delta_unstable=10

Time for backsubstitution: 1.74 seconds

### RS candidates at layer 1
type: RSZ, layer: 1, pos: 198
type: RSZ, layer: 1, pos: 199
type: RSZ, layer: 1, pos: 83
type: RSZ, layer: 1, pos: 68

Time for candidate selection: 0.00 seconds

### Candidate
type: RSZ, layer: 1, pos: 198

### Relational analysis RSZ of RS_RSZ1_RSZ1_RSZ1_RSZ1_RSZ2_RSZ1

#### Relational analysis RSZ result of RS_RSZ1_RSZ1_RSZ1_RSZ1_RSZ2_RSZ1
Status: Status.UNKNOWN
Output dim: 1, lower bound: -0.0007749, upper bound: 0.0007751
time: 0.71 seconds

### Relational analysis RSZ of RS_RSZ1_RSZ1_RSZ1_RSZ1_RSZ2_RSZ2

#### Relational analysis RSZ result of RS_RSZ1_RSZ1_RSZ1_RSZ1_RSZ2_RSZ2
Status: Status.UNKNOWN
Output dim: 1, lower bound: -0.0007749, upper bound: 0.0007751
time: 0.68 seconds

## BFS RS instance: RS_RSZ1_RSZ1_RSZ1_RSZ2_RSZ1

### Backsubstitution after applying RS history:
0: 0.0007184, 0.0011172, 0.0007184, 0.0011172, -0.0003988, 0.0003988
1: 0.9934075, 0.9943579, 0.9934075, 0.9943579, -0.0009505, 0.0009505
2: -0.0087946, -0.0055215, -0.0087946, -0.0055215, -0.0029889, 0.0029853
3: 0.0035868, 0.0041619, 0.0035868, 0.0041619, -0.0005751, 0.0005751
4: 0.0027809, 0.0053678, 0.0027809, 0.0053678, -0.0025869, 0.0025869
5: 0.0050646, 0.0064029, 0.0050646, 0.0064029, -0.0013382, 0.0013382
6: -0.0021738, -0.0008562, -0.0021738, -0.0008562, -0.0013176, 0.0013176
7: -0.0082275, -0.0074685, -0.0082275, -0.0074685, -0.0007590, 0.0007590
8: 0.0055062, 0.0098066, 0.0055062, 0.0098066, -0.0042230, 0.0042240
9: -0.0036825, -0.0031002, -0.0036825, -0.0031002, -0.0005823, 0.0005823

### Unstable ReLU Count (Linear/Conv2D Layers)
- layer_idx=0, type=LayerType.Linear, total=256, inp1_unstable=9, inp2_unstable=9, delta_unstable=251
- layer_idx=2, type=LayerType.Linear, total=256, inp1_unstable=6, inp2_unstable=6, delta_unstable=256
- layer_idx=4, type=LayerType.Linear, total=256, inp1_unstable=0, inp2_unstable=0, delta_unstable=256
- layer_idx=6, type=LayerType.Linear, total=256, inp1_unstable=2, inp2_unstable=2, delta_unstable=256
- layer_idx=8, type=LayerType.Linear, total=10, inp1_unstable=0, inp2_unstable=0, delta_unstable=10

Time for backsubstitution: 1.72 seconds

### RS candidates at layer 1
type: RSZ, layer: 1, pos: 68
type: RSZ, layer: 1, pos: 199
type: RSZ, layer: 1, pos: 52
type: RSZ, layer: 1, pos: 198

Time for candidate selection: 0.00 seconds

### Candidate
type: RSZ, layer: 1, pos: 68

### Relational analysis RSZ of RS_RSZ1_RSZ1_RSZ1_RSZ2_RSZ1_RSZ1

#### Relational analysis RSZ result of RS_RSZ1_RSZ1_RSZ1_RSZ2_RSZ1_RSZ1
Status: Status.UNKNOWN
Output dim: 1, lower bound: -0.0007916, upper bound: 0.0007958
time: 0.64 seconds

### Relational analysis RSZ of RS_RSZ1_RSZ1_RSZ1_RSZ2_RSZ1_RSZ2

#### Relational analysis RSZ result of RS_RSZ1_RSZ1_RSZ1_RSZ2_RSZ1_RSZ2
Status: Status.UNKNOWN
Output dim: 1, lower bound: -0.0007953, upper bound: 0.0007937
time: 0.78 seconds

## BFS RS instance: RS_RSZ1_RSZ1_RSZ1_RSZ2_RSZ2

### Backsubstitution after applying RS history:
0: 0.0007184, 0.0011172, 0.0007184, 0.0011172, -0.0003988, 0.0003988
1: 0.9934075, 0.9943579, 0.9934075, 0.9943579, -0.0009505, 0.0009505
2: -0.0087946, -0.0055215, -0.0087946, -0.0055215, -0.0029886, 0.0029856
3: 0.0035868, 0.0041619, 0.0035868, 0.0041619, -0.0005751, 0.0005751
4: 0.0027809, 0.0053678, 0.0027809, 0.0053678, -0.0025869, 0.0025869
5: 0.0050646, 0.0064029, 0.0050646, 0.0064029, -0.0013382, 0.0013382
6: -0.0021738, -0.0008562, -0.0021738, -0.0008562, -0.0013176, 0.0013176
7: -0.0082275, -0.0074685, -0.0082275, -0.0074685, -0.0007590, 0.0007590
8: 0.0055062, 0.0098066, 0.0055062, 0.0098066, -0.0042231, 0.0042239
9: -0.0036825, -0.0031002, -0.0036825, -0.0031002, -0.0005823, 0.0005823

### Unstable ReLU Count (Linear/Conv2D Layers)
- layer_idx=0, type=LayerType.Linear, total=256, inp1_unstable=9, inp2_unstable=9, delta_unstable=251
- layer_idx=2, type=LayerType.Linear, total=256, inp1_unstable=6, inp2_unstable=6, delta_unstable=256
- layer_idx=4, type=LayerType.Linear, total=256, inp1_unstable=0, inp2_unstable=0, delta_unstable=256
- layer_idx=6, type=LayerType.Linear, total=256, inp1_unstable=2, inp2_unstable=2, delta_unstable=256
- layer_idx=8, type=LayerType.Linear, total=10, inp1_unstable=0, inp2_unstable=0, delta_unstable=10

Time for backsubstitution: 1.68 seconds

### RS candidates at layer 1
type: RSZ, layer: 1, pos: 198
type: RSZ, layer: 1, pos: 52
type: RSZ, layer: 1, pos: 68
type: RSZ, layer: 1, pos: 199

Time for candidate selection: 0.00 seconds

### Candidate
type: RSZ, layer: 1, pos: 198

### Relational analysis RSZ of RS_RSZ1_RSZ1_RSZ1_RSZ2_RSZ2_RSZ1

#### Relational analysis RSZ result of RS_RSZ1_RSZ1_RSZ1_RSZ2_RSZ2_RSZ1
Status: Status.UNKNOWN
Output dim: 1, lower bound: -0.0007734, upper bound: 0.0007735
time: 0.71 seconds

### Relational analysis RSZ of RS_RSZ1_RSZ1_RSZ1_RSZ2_RSZ2_RSZ2

#### Relational analysis RSZ result of RS_RSZ1_RSZ1_RSZ1_RSZ2_RSZ2_RSZ2
Status: Status.UNKNOWN
Output dim: 1, lower bound: -0.0007734, upper bound: 0.0007735
time: 0.71 seconds

## BFS RS instance: RS_RSZ1_RSZ1_RSZ2_RSZ1_RSZ1

### Backsubstitution after applying RS history:
0: 0.0007184, 0.0011172, 0.0007184, 0.0011172, -0.0003988, 0.0003988
1: 0.9934075, 0.9943579, 0.9934075, 0.9943579, -0.0009505, 0.0009505
2: -0.0087946, -0.0055215, -0.0087946, -0.0055215, -0.0029814, 0.0029832
3: 0.0035868, 0.0041619, 0.0035868, 0.0041619, -0.0005751, 0.0005751
4: 0.0027809, 0.0053678, 0.0027809, 0.0053678, -0.0025869, 0.0025869
5: 0.0050646, 0.0064029, 0.0050646, 0.0064029, -0.0013382, 0.0013382
6: -0.0021738, -0.0008562, -0.0021738, -0.0008562, -0.0013176, 0.0013176
7: -0.0082275, -0.0074685, -0.0082275, -0.0074685, -0.0007590, 0.0007590
8: 0.0055062, 0.0098066, 0.0055062, 0.0098066, -0.0042221, 0.0042216
9: -0.0036825, -0.0031002, -0.0036825, -0.0031002, -0.0005823, 0.0005823

### Unstable ReLU Count (Linear/Conv2D Layers)
- layer_idx=0, type=LayerType.Linear, total=256, inp1_unstable=9, inp2_unstable=9, delta_unstable=251
- layer_idx=2, type=LayerType.Linear, total=256, inp1_unstable=6, inp2_unstable=6, delta_unstable=256
- layer_idx=4, type=LayerType.Linear, total=256, inp1_unstable=0, inp2_unstable=0, delta_unstable=256
- layer_idx=6, type=LayerType.Linear, total=256, inp1_unstable=2, inp2_unstable=2, delta_unstable=256
- layer_idx=8, type=LayerType.Linear, total=10, inp1_unstable=0, inp2_unstable=0, delta_unstable=10

Time for backsubstitution: 1.67 seconds

### RS candidates at layer 1
type: RSZ, layer: 1, pos: 83
type: RSZ, layer: 1, pos: 199
type: RSZ, layer: 1, pos: 68
type: RSZ, layer: 1, pos: 136

Time for candidate selection: 0.00 seconds

### Candidate
type: RSZ, layer: 1, pos: 83

### Relational analysis RSZ of RS_RSZ1_RSZ1_RSZ2_RSZ1_RSZ1_RSZ1

#### Relational analysis RSZ result of RS_RSZ1_RSZ1_RSZ2_RSZ1_RSZ1_RSZ1
Status: Status.UNKNOWN
Output dim: 1, lower bound: -0.0007740, upper bound: 0.0007739
time: 0.80 seconds

### Relational analysis RSZ of RS_RSZ1_RSZ1_RSZ2_RSZ1_RSZ1_RSZ2

#### Relational analysis RSZ result of RS_RSZ1_RSZ1_RSZ2_RSZ1_RSZ1_RSZ2
Status: Status.UNKNOWN
Output dim: 1, lower bound: -0.0007740, upper bound: 0.0007739
time: 0.63 seconds

## BFS RS instance: RS_RSZ1_RSZ1_RSZ2_RSZ1_RSZ2

### Backsubstitution after applying RS history:
0: 0.0007184, 0.0011172, 0.0007184, 0.0011172, -0.0003988, 0.0003988
1: 0.9934075, 0.9943579, 0.9934075, 0.9943579, -0.0009505, 0.0009505
2: -0.0087946, -0.0055215, -0.0087946, -0.0055215, -0.0029824, 0.0029823
3: 0.0035868, 0.0041619, 0.0035868, 0.0041619, -0.0005751, 0.0005751
4: 0.0027809, 0.0053678, 0.0027809, 0.0053678, -0.0025869, 0.0025869
5: 0.0050646, 0.0064029, 0.0050646, 0.0064029, -0.0013382, 0.0013382
6: -0.0021738, -0.0008562, -0.0021738, -0.0008562, -0.0013176, 0.0013176
7: -0.0082275, -0.0074685, -0.0082275, -0.0074685, -0.0007590, 0.0007590
8: 0.0055062, 0.0098066, 0.0055062, 0.0098066, -0.0042219, 0.0042219
9: -0.0036825, -0.0031002, -0.0036825, -0.0031002, -0.0005823, 0.0005823

### Unstable ReLU Count (Linear/Conv2D Layers)
- layer_idx=0, type=LayerType.Linear, total=256, inp1_unstable=9, inp2_unstable=9, delta_unstable=251
- layer_idx=2, type=LayerType.Linear, total=256, inp1_unstable=6, inp2_unstable=6, delta_unstable=256
- layer_idx=4, type=LayerType.Linear, total=256, inp1_unstable=0, inp2_unstable=0, delta_unstable=256
- layer_idx=6, type=LayerType.Linear, total=256, inp1_unstable=2, inp2_unstable=2, delta_unstable=256
- layer_idx=8, type=LayerType.Linear, total=10, inp1_unstable=0, inp2_unstable=0, delta_unstable=10

Time for backsubstitution: 1.68 seconds

### RS candidates at layer 1
type: RSZ, layer: 1, pos: 199
type: RSZ, layer: 1, pos: 136
type: RSZ, layer: 1, pos: 68
type: RSZ, layer: 1, pos: 83

Time for candidate selection: 0.00 seconds

### Candidate
type: RSZ, layer: 1, pos: 199

### Relational analysis RSZ of RS_RSZ1_RSZ1_RSZ2_RSZ1_RSZ2_RSZ1

#### Relational analysis RSZ result of RS_RSZ1_RSZ1_RSZ2_RSZ1_RSZ2_RSZ1
Status: Status.UNKNOWN
Output dim: 1, lower bound: -0.0007131, upper bound: 0.0007131
time: 0.62 seconds

### Relational analysis RSZ of RS_RSZ1_RSZ1_RSZ2_RSZ1_RSZ2_RSZ2

#### Relational analysis RSZ result of RS_RSZ1_RSZ1_RSZ2_RSZ1_RSZ2_RSZ2
Status: Status.UNKNOWN
Output dim: 1, lower bound: -0.0007131, upper bound: 0.0007131
time: 0.63 seconds

## BFS RS instance: RS_RSZ1_RSZ1_RSZ2_RSZ2_RSZ1

### Backsubstitution after applying RS history:
0: 0.0007184, 0.0011172, 0.0007184, 0.0011172, -0.0003988, 0.0003988
1: 0.9934075, 0.9943579, 0.9934075, 0.9943579, -0.0009505, 0.0009505
2: -0.0087946, -0.0055215, -0.0087946, -0.0055215, -0.0029881, 0.0029761
3: 0.0035868, 0.0041619, 0.0035868, 0.0041619, -0.0005751, 0.0005751
4: 0.0027809, 0.0053678, 0.0027809, 0.0053678, -0.0025869, 0.0025869
5: 0.0050646, 0.0064029, 0.0050646, 0.0064029, -0.0013382, 0.0013382
6: -0.0021738, -0.0008562, -0.0021738, -0.0008562, -0.0013176, 0.0013176
7: -0.0082275, -0.0074685, -0.0082275, -0.0074685, -0.0007590, 0.0007590
8: 0.0055062, 0.0098066, 0.0055062, 0.0098066, -0.0042199, 0.0042228
9: -0.0036825, -0.0031002, -0.0036825, -0.0031002, -0.0005823, 0.0005823

### Unstable ReLU Count (Linear/Conv2D Layers)
- layer_idx=0, type=LayerType.Linear, total=256, inp1_unstable=9, inp2_unstable=9, delta_unstable=251
- layer_idx=2, type=LayerType.Linear, total=256, inp1_unstable=6, inp2_unstable=6, delta_unstable=256
- layer_idx=4, type=LayerType.Linear, total=256, inp1_unstable=0, inp2_unstable=0, delta_unstable=256
- layer_idx=6, type=LayerType.Linear, total=256, inp1_unstable=2, inp2_unstable=2, delta_unstable=256
- layer_idx=8, type=LayerType.Linear, total=10, inp1_unstable=0, inp2_unstable=0, delta_unstable=10

Time for backsubstitution: 1.69 seconds

### RS candidates at layer 1
type: RSZ, layer: 1, pos: 136
type: RSZ, layer: 1, pos: 199
type: RSZ, layer: 1, pos: 83
type: RSZ, layer: 1, pos: 52

Time for candidate selection: 0.00 seconds

### Candidate
type: RSZ, layer: 1, pos: 136

### Relational analysis RSZ of RS_RSZ1_RSZ1_RSZ2_RSZ2_RSZ1_RSZ1

#### Relational analysis RSZ result of RS_RSZ1_RSZ1_RSZ2_RSZ2_RSZ1_RSZ1
Status: Status.UNKNOWN
Output dim: 1, lower bound: -0.0007739, upper bound: 0.0007748
time: 0.73 seconds

### Relational analysis RSZ of RS_RSZ1_RSZ1_RSZ2_RSZ2_RSZ1_RSZ2

#### Relational analysis RSZ result of RS_RSZ1_RSZ1_RSZ2_RSZ2_RSZ1_RSZ2
Status: Status.UNKNOWN
Output dim: 1, lower bound: -0.0007739, upper bound: 0.0007748
time: 0.67 seconds

## BFS RS instance: RS_RSZ1_RSZ1_RSZ2_RSZ2_RSZ2

### Backsubstitution after applying RS history:
0: 0.0007184, 0.0011172, 0.0007184, 0.0011172, -0.0003988, 0.0003988
1: 0.9934075, 0.9943579, 0.9934075, 0.9943579, -0.0009505, 0.0009505
2: -0.0087946, -0.0055215, -0.0087946, -0.0055215, -0.0029864, 0.0029778
3: 0.0035868, 0.0041619, 0.0035868, 0.0041619, -0.0005751, 0.0005751
4: 0.0027809, 0.0053678, 0.0027809, 0.0053678, -0.0025869, 0.0025869
5: 0.0050646, 0.0064029, 0.0050646, 0.0064029, -0.0013382, 0.0013382
6: -0.0021738, -0.0008562, -0.0021738, -0.0008562, -0.0013176, 0.0013176
7: -0.0082275, -0.0074685, -0.0082275, -0.0074685, -0.0007590, 0.0007590
8: 0.0055062, 0.0098066, 0.0055062, 0.0098066, -0.0042203, 0.0042224
9: -0.0036825, -0.0031002, -0.0036825, -0.0031002, -0.0005823, 0.0005823

### Unstable ReLU Count (Linear/Conv2D Layers)
- layer_idx=0, type=LayerType.Linear, total=256, inp1_unstable=9, inp2_unstable=9, delta_unstable=251
- layer_idx=2, type=LayerType.Linear, total=256, inp1_unstable=6, inp2_unstable=6, delta_unstable=256
- layer_idx=4, type=LayerType.Linear, total=256, inp1_unstable=0, inp2_unstable=0, delta_unstable=256
- layer_idx=6, type=LayerType.Linear, total=256, inp1_unstable=2, inp2_unstable=2, delta_unstable=256
- layer_idx=8, type=LayerType.Linear, total=10, inp1_unstable=0, inp2_unstable=0, delta_unstable=10

Time for backsubstitution: 1.60 seconds

### RS candidates at layer 1
type: RSZ, layer: 1, pos: 52
type: RSZ, layer: 1, pos: 136
type: RSZ, layer: 1, pos: 83
type: RSZ, layer: 1, pos: 199

Time for candidate selection: 0.00 seconds

### Candidate
type: RSZ, layer: 1, pos: 52

### Relational analysis RSZ of RS_RSZ1_RSZ1_RSZ2_RSZ2_RSZ2_RSZ1

#### Relational analysis RSZ result of RS_RSZ1_RSZ1_RSZ2_RSZ2_RSZ2_RSZ1
Status: Status.UNKNOWN
Output dim: 1, lower bound: -0.0007765, upper bound: 0.0007759
time: 0.66 seconds

### Relational analysis RSZ of RS_RSZ1_RSZ1_RSZ2_RSZ2_RSZ2_RSZ2

#### Relational analysis RSZ result of RS_RSZ1_RSZ1_RSZ2_RSZ2_RSZ2_RSZ2
Status: Status.UNKNOWN
Output dim: 1, lower bound: -0.0007765, upper bound: 0.0007760
time: 0.61 seconds

## BFS RS instance: RS_RSZ1_RSZ2_RSZ1_RSZ1_RSZ1

### Backsubstitution after applying RS history:
0: 0.0007184, 0.0011172, 0.0007184, 0.0011172, -0.0003988, 0.0003988
1: 0.9934075, 0.9943579, 0.9934075, 0.9943579, -0.0009505, 0.0009505
2: -0.0087946, -0.0055215, -0.0087946, -0.0055215, -0.0029805, 0.0029849
3: 0.0035868, 0.0041619, 0.0035868, 0.0041619, -0.0005751, 0.0005751
4: 0.0027809, 0.0053678, 0.0027809, 0.0053678, -0.0025869, 0.0025869
5: 0.0050646, 0.0064029, 0.0050646, 0.0064029, -0.0013382, 0.0013382
6: -0.0021738, -0.0008562, -0.0021738, -0.0008562, -0.0013176, 0.0013176
7: -0.0082275, -0.0074685, -0.0082275, -0.0074685, -0.0007590, 0.0007590
8: 0.0055062, 0.0098066, 0.0055062, 0.0098066, -0.0042230, 0.0042218
9: -0.0036825, -0.0031002, -0.0036825, -0.0031002, -0.0005823, 0.0005823

### Unstable ReLU Count (Linear/Conv2D Layers)
- layer_idx=0, type=LayerType.Linear, total=256, inp1_unstable=9, inp2_unstable=9, delta_unstable=251
- layer_idx=2, type=LayerType.Linear, total=256, inp1_unstable=6, inp2_unstable=6, delta_unstable=256
- layer_idx=4, type=LayerType.Linear, total=256, inp1_unstable=0, inp2_unstable=0, delta_unstable=256
- layer_idx=6, type=LayerType.Linear, total=256, inp1_unstable=2, inp2_unstable=2, delta_unstable=256
- layer_idx=8, type=LayerType.Linear, total=10, inp1_unstable=0, inp2_unstable=0, delta_unstable=10

Time for backsubstitution: 1.59 seconds

### RS candidates at layer 1
type: RSZ, layer: 1, pos: 83
type: RSZ, layer: 1, pos: 68
type: RSZ, layer: 1, pos: 199
type: RSZ, layer: 1, pos: 204

Time for candidate selection: 0.00 seconds

### Candidate
type: RSZ, layer: 1, pos: 83

### Relational analysis RSZ of RS_RSZ1_RSZ2_RSZ1_RSZ1_RSZ1_RSZ1

#### Relational analysis RSZ result of RS_RSZ1_RSZ2_RSZ1_RSZ1_RSZ1_RSZ1
Status: Status.UNKNOWN
Output dim: 1, lower bound: -0.0007773, upper bound: 0.0007771
time: 0.66 seconds

### Relational analysis RSZ of RS_RSZ1_RSZ2_RSZ1_RSZ1_RSZ1_RSZ2

#### Relational analysis RSZ result of RS_RSZ1_RSZ2_RSZ1_RSZ1_RSZ1_RSZ2
Status: Status.UNKNOWN
Output dim: 1, lower bound: -0.0007772, upper bound: 0.0007772
time: 0.61 seconds

## BFS RS instance: RS_RSZ1_RSZ2_RSZ1_RSZ1_RSZ2

### Backsubstitution after applying RS history:
0: 0.0007184, 0.0011172, 0.0007184, 0.0011172, -0.0003988, 0.0003988
1: 0.9934075, 0.9943579, 0.9934075, 0.9943579, -0.0009505, 0.0009505
2: -0.0087946, -0.0055215, -0.0087946, -0.0055215, -0.0029813, 0.0029838
3: 0.0035868, 0.0041619, 0.0035868, 0.0041619, -0.0005751, 0.0005751
4: 0.0027809, 0.0053678, 0.0027809, 0.0053678, -0.0025869, 0.0025869
5: 0.0050646, 0.0064029, 0.0050646, 0.0064029, -0.0013382, 0.0013382
6: -0.0021738, -0.0008562, -0.0021738, -0.0008562, -0.0013176, 0.0013176
7: -0.0082275, -0.0074685, -0.0082275, -0.0074685, -0.0007590, 0.0007590
8: 0.0055062, 0.0098066, 0.0055062, 0.0098066, -0.0042227, 0.0042220
9: -0.0036825, -0.0031002, -0.0036825, -0.0031002, -0.0005823, 0.0005823

### Unstable ReLU Count (Linear/Conv2D Layers)
- layer_idx=0, type=LayerType.Linear, total=256, inp1_unstable=9, inp2_unstable=9, delta_unstable=251
- layer_idx=2, type=LayerType.Linear, total=256, inp1_unstable=6, inp2_unstable=6, delta_unstable=256
- layer_idx=4, type=LayerType.Linear, total=256, inp1_unstable=0, inp2_unstable=0, delta_unstable=256
- layer_idx=6, type=LayerType.Linear, total=256, inp1_unstable=2, inp2_unstable=2, delta_unstable=256
- layer_idx=8, type=LayerType.Linear, total=10, inp1_unstable=0, inp2_unstable=0, delta_unstable=10

Time for backsubstitution: 1.78 seconds

### RS candidates at layer 1
type: RSZ, layer: 1, pos: 204
type: RSZ, layer: 1, pos: 199
type: RSZ, layer: 1, pos: 83
type: RSZ, layer: 1, pos: 68

Time for candidate selection: 0.00 seconds

### Candidate
type: RSZ, layer: 1, pos: 204

### Relational analysis RSZ of RS_RSZ1_RSZ2_RSZ1_RSZ1_RSZ2_RSZ1

#### Relational analysis RSZ result of RS_RSZ1_RSZ2_RSZ1_RSZ1_RSZ2_RSZ1
Status: Status.UNKNOWN
Output dim: 1, lower bound: -0.0007749, upper bound: 0.0007750
time: 0.73 seconds

### Relational analysis RSZ of RS_RSZ1_RSZ2_RSZ1_RSZ1_RSZ2_RSZ2

#### Relational analysis RSZ result of RS_RSZ1_RSZ2_RSZ1_RSZ1_RSZ2_RSZ2
Status: Status.UNKNOWN
Output dim: 1, lower bound: -0.0007750, upper bound: 0.0007750
time: 0.74 seconds

## BFS RS instance: RS_RSZ1_RSZ2_RSZ1_RSZ2_RSZ1

### Backsubstitution after applying RS history:
0: 0.0007184, 0.0011172, 0.0007184, 0.0011172, -0.0003988, 0.0003988
1: 0.9934075, 0.9943579, 0.9934075, 0.9943579, -0.0009505, 0.0009505
2: -0.0087946, -0.0055215, -0.0087946, -0.0055215, -0.0029893, 0.0029805
3: 0.0035868, 0.0041619, 0.0035868, 0.0041619, -0.0005751, 0.0005751
4: 0.0027809, 0.0053678, 0.0027809, 0.0053678, -0.0025869, 0.0025869
5: 0.0050646, 0.0064029, 0.0050646, 0.0064029, -0.0013382, 0.0013382
6: -0.0021738, -0.0008562, -0.0021738, -0.0008562, -0.0013176, 0.0013176
7: -0.0082275, -0.0074685, -0.0082275, -0.0074685, -0.0007590, 0.0007590
8: 0.0055062, 0.0098066, 0.0055062, 0.0098066, -0.0042219, 0.0042240
9: -0.0036825, -0.0031002, -0.0036825, -0.0031002, -0.0005823, 0.0005823

### Unstable ReLU Count (Linear/Conv2D Layers)
- layer_idx=0, type=LayerType.Linear, total=256, inp1_unstable=9, inp2_unstable=9, delta_unstable=251
- layer_idx=2, type=LayerType.Linear, total=256, inp1_unstable=6, inp2_unstable=6, delta_unstable=256
- layer_idx=4, type=LayerType.Linear, total=256, inp1_unstable=0, inp2_unstable=0, delta_unstable=256
- layer_idx=6, type=LayerType.Linear, total=256, inp1_unstable=2, inp2_unstable=2, delta_unstable=256
- layer_idx=8, type=LayerType.Linear, total=10, inp1_unstable=0, inp2_unstable=0, delta_unstable=10

Time for backsubstitution: 1.62 seconds

### RS candidates at layer 1
type: RSZ, layer: 1, pos: 68
type: RSZ, layer: 1, pos: 83
type: RSZ, layer: 1, pos: 199
type: RSZ, layer: 1, pos: 52

Time for candidate selection: 0.00 seconds

### Candidate
type: RSZ, layer: 1, pos: 68

### Relational analysis RSZ of RS_RSZ1_RSZ2_RSZ1_RSZ2_RSZ1_RSZ1

#### Relational analysis RSZ result of RS_RSZ1_RSZ2_RSZ1_RSZ2_RSZ1_RSZ1
Status: Status.UNKNOWN
Output dim: 1, lower bound: -0.0007731, upper bound: 0.0007748
time: 0.64 seconds

### Relational analysis RSZ of RS_RSZ1_RSZ2_RSZ1_RSZ2_RSZ1_RSZ2

#### Relational analysis RSZ result of RS_RSZ1_RSZ2_RSZ1_RSZ2_RSZ1_RSZ2
Status: Status.UNKNOWN
Output dim: 1, lower bound: -0.0007748, upper bound: 0.0007738
time: 0.69 seconds

## BFS RS instance: RS_RSZ1_RSZ2_RSZ1_RSZ2_RSZ2

### Backsubstitution after applying RS history:
0: 0.0007184, 0.0011172, 0.0007184, 0.0011172, -0.0003988, 0.0003988
1: 0.9934075, 0.9943579, 0.9934075, 0.9943579, -0.0009505, 0.0009505
2: -0.0087946, -0.0055215, -0.0087946, -0.0055215, -0.0029882, 0.0029812
3: 0.0035868, 0.0041619, 0.0035868, 0.0041619, -0.0005751, 0.0005751
4: 0.0027809, 0.0053678, 0.0027809, 0.0053678, -0.0025869, 0.0025869
5: 0.0050646, 0.0064029, 0.0050646, 0.0064029, -0.0013382, 0.0013382
6: -0.0021738, -0.0008562, -0.0021738, -0.0008562, -0.0013176, 0.0013176
7: -0.0082275, -0.0074685, -0.0082275, -0.0074685, -0.0007590, 0.0007590
8: 0.0055062, 0.0098066, 0.0055062, 0.0098066, -0.0042221, 0.0042237
9: -0.0036825, -0.0031002, -0.0036825, -0.0031002, -0.0005823, 0.0005823

### Unstable ReLU Count (Linear/Conv2D Layers)
- layer_idx=0, type=LayerType.Linear, total=256, inp1_unstable=9, inp2_unstable=9, delta_unstable=251
- layer_idx=2, type=LayerType.Linear, total=256, inp1_unstable=6, inp2_unstable=6, delta_unstable=256
- layer_idx=4, type=LayerType.Linear, total=256, inp1_unstable=0, inp2_unstable=0, delta_unstable=256
- layer_idx=6, type=LayerType.Linear, total=256, inp1_unstable=2, inp2_unstable=2, delta_unstable=256
- layer_idx=8, type=LayerType.Linear, total=10, inp1_unstable=0, inp2_unstable=0, delta_unstable=10

Time for backsubstitution: 1.70 seconds

### RS candidates at layer 1
type: RSZ, layer: 1, pos: 52
type: RSZ, layer: 1, pos: 83
type: RSZ, layer: 1, pos: 68
type: RSZ, layer: 1, pos: 199

Time for candidate selection: 0.00 seconds

### Candidate
type: RSZ, layer: 1, pos: 52

### Relational analysis RSZ of RS_RSZ1_RSZ2_RSZ1_RSZ2_RSZ2_RSZ1

#### Relational analysis RSZ result of RS_RSZ1_RSZ2_RSZ1_RSZ2_RSZ2_RSZ1
Status: Status.UNKNOWN
Output dim: 1, lower bound: -0.0007752, upper bound: 0.0007749
time: 0.69 seconds

### Relational analysis RSZ of RS_RSZ1_RSZ2_RSZ1_RSZ2_RSZ2_RSZ2

#### Relational analysis RSZ result of RS_RSZ1_RSZ2_RSZ1_RSZ2_RSZ2_RSZ2
Status: Status.UNKNOWN
Output dim: 1, lower bound: -0.0007750, upper bound: 0.0007749
time: 0.69 seconds

## BFS RS instance: RS_RSZ1_RSZ2_RSZ2_RSZ1_RSZ1

### Backsubstitution after applying RS history:
0: 0.0007184, 0.0011172, 0.0007184, 0.0011172, -0.0003988, 0.0003988
1: 0.9934075, 0.9943579, 0.9934075, 0.9943579, -0.0009505, 0.0009505
2: -0.0087946, -0.0055215, -0.0087946, -0.0055215, -0.0029908, 0.0029897
3: 0.0035868, 0.0041619, 0.0035868, 0.0041619, -0.0005751, 0.0005751
4: 0.0027809, 0.0053678, 0.0027809, 0.0053678, -0.0025869, 0.0025869
5: 0.0050646, 0.0064029, 0.0050646, 0.0064029, -0.0013382, 0.0013382
6: -0.0021738, -0.0008562, -0.0021738, -0.0008562, -0.0013176, 0.0013176
7: -0.0082275, -0.0074685, -0.0082275, -0.0074685, -0.0007590, 0.0007590
8: 0.0055062, 0.0098066, 0.0055062, 0.0098066, -0.0042242, 0.0042245
9: -0.0036825, -0.0031002, -0.0036825, -0.0031002, -0.0005823, 0.0005823

### Unstable ReLU Count (Linear/Conv2D Layers)
- layer_idx=0, type=LayerType.Linear, total=256, inp1_unstable=9, inp2_unstable=9, delta_unstable=251
- layer_idx=2, type=LayerType.Linear, total=256, inp1_unstable=6, inp2_unstable=6, delta_unstable=256
- layer_idx=4, type=LayerType.Linear, total=256, inp1_unstable=0, inp2_unstable=0, delta_unstable=256
- layer_idx=6, type=LayerType.Linear, total=256, inp1_unstable=2, inp2_unstable=2, delta_unstable=256
- layer_idx=8, type=LayerType.Linear, total=10, inp1_unstable=0, inp2_unstable=0, delta_unstable=10

Time for backsubstitution: 1.63 seconds

### RS candidates at layer 1
type: RSZ, layer: 1, pos: 68
type: RSZ, layer: 1, pos: 204
type: RSZ, layer: 1, pos: 198
type: RSZ, layer: 1, pos: 52

Time for candidate selection: 0.00 seconds

### Candidate
type: RSZ, layer: 1, pos: 68

### Relational analysis RSZ of RS_RSZ1_RSZ2_RSZ2_RSZ1_RSZ1_RSZ1

#### Relational analysis RSZ result of RS_RSZ1_RSZ2_RSZ2_RSZ1_RSZ1_RSZ1
Status: Status.UNKNOWN
Output dim: 1, lower bound: -0.0007558, upper bound: 0.0007564
time: 0.63 seconds

### Relational analysis RSZ of RS_RSZ1_RSZ2_RSZ2_RSZ1_RSZ1_RSZ2

#### Relational analysis RSZ result of RS_RSZ1_RSZ2_RSZ2_RSZ1_RSZ1_RSZ2
Status: Status.UNKNOWN
Output dim: 1, lower bound: -0.0007562, upper bound: 0.0007560
time: 0.63 seconds

## BFS RS instance: RS_RSZ1_RSZ2_RSZ2_RSZ1_RSZ2

### Backsubstitution after applying RS history:
0: 0.0007184, 0.0011172, 0.0007184, 0.0011172, -0.0003988, 0.0003988
1: 0.9934075, 0.9943579, 0.9934075, 0.9943579, -0.0009505, 0.0009505
2: -0.0087946, -0.0055215, -0.0087946, -0.0055215, -0.0029906, 0.0029902
3: 0.0035868, 0.0041619, 0.0035868, 0.0041619, -0.0005751, 0.0005751
4: 0.0027809, 0.0053678, 0.0027809, 0.0053678, -0.0025869, 0.0025869
5: 0.0050646, 0.0064029, 0.0050646, 0.0064029, -0.0013382, 0.0013382
6: -0.0021738, -0.0008562, -0.0021738, -0.0008562, -0.0013176, 0.0013176
7: -0.0082275, -0.0074685, -0.0082275, -0.0074685, -0.0007590, 0.0007590
8: 0.0055062, 0.0098066, 0.0055062, 0.0098066, -0.0042243, 0.0042244
9: -0.0036825, -0.0031002, -0.0036825, -0.0031002, -0.0005823, 0.0005823

### Unstable ReLU Count (Linear/Conv2D Layers)
- layer_idx=0, type=LayerType.Linear, total=256, inp1_unstable=9, inp2_unstable=9, delta_unstable=251
- layer_idx=2, type=LayerType.Linear, total=256, inp1_unstable=6, inp2_unstable=6, delta_unstable=256
- layer_idx=4, type=LayerType.Linear, total=256, inp1_unstable=0, inp2_unstable=0, delta_unstable=256
- layer_idx=6, type=LayerType.Linear, total=256, inp1_unstable=2, inp2_unstable=2, delta_unstable=256
- layer_idx=8, type=LayerType.Linear, total=10, inp1_unstable=0, inp2_unstable=0, delta_unstable=10

Time for backsubstitution: 1.61 seconds

### RS candidates at layer 1
type: RSZ, layer: 1, pos: 68
type: RSZ, layer: 1, pos: 52
type: RSZ, layer: 1, pos: 204
type: RSZ, layer: 1, pos: 198

Time for candidate selection: 0.00 seconds

### Candidate
type: RSZ, layer: 1, pos: 68

### Relational analysis RSZ of RS_RSZ1_RSZ2_RSZ2_RSZ1_RSZ2_RSZ1

#### Relational analysis RSZ result of RS_RSZ1_RSZ2_RSZ2_RSZ1_RSZ2_RSZ1
Status: Status.UNKNOWN
Output dim: 1, lower bound: -0.0007557, upper bound: 0.0007564
time: 0.65 seconds

### Relational analysis RSZ of RS_RSZ1_RSZ2_RSZ2_RSZ1_RSZ2_RSZ2

#### Relational analysis RSZ result of RS_RSZ1_RSZ2_RSZ2_RSZ1_RSZ2_RSZ2
Status: Status.UNKNOWN
Output dim: 1, lower bound: -0.0007562, upper bound: 0.0007561
time: 0.57 seconds

## BFS RS instance: RS_RSZ1_RSZ2_RSZ2_RSZ2_RSZ1

### Backsubstitution after applying RS history:
0: 0.0007184, 0.0011172, 0.0007184, 0.0011172, -0.0003988, 0.0003988
1: 0.9934075, 0.9943579, 0.9934075, 0.9943579, -0.0009505, 0.0009505
2: -0.0087946, -0.0055215, -0.0087946, -0.0055215, -0.0029839, 0.0029853
3: 0.0035868, 0.0041619, 0.0035868, 0.0041619, -0.0005751, 0.0005751
4: 0.0027809, 0.0053678, 0.0027809, 0.0053678, -0.0025869, 0.0025869
5: 0.0050646, 0.0064029, 0.0050646, 0.0064029, -0.0013382, 0.0013382
6: -0.0021738, -0.0008562, -0.0021738, -0.0008562, -0.0013176, 0.0013176
7: -0.0082275, -0.0074685, -0.0082275, -0.0074685, -0.0007590, 0.0007590
8: 0.0055062, 0.0098066, 0.0055062, 0.0098066, -0.0042231, 0.0042228
9: -0.0036825, -0.0031002, -0.0036825, -0.0031002, -0.0005823, 0.0005823

### Unstable ReLU Count (Linear/Conv2D Layers)
- layer_idx=0, type=LayerType.Linear, total=256, inp1_unstable=9, inp2_unstable=9, delta_unstable=251
- layer_idx=2, type=LayerType.Linear, total=256, inp1_unstable=6, inp2_unstable=6, delta_unstable=256
- layer_idx=4, type=LayerType.Linear, total=256, inp1_unstable=0, inp2_unstable=0, delta_unstable=256
- layer_idx=6, type=LayerType.Linear, total=256, inp1_unstable=2, inp2_unstable=2, delta_unstable=256
- layer_idx=8, type=LayerType.Linear, total=10, inp1_unstable=0, inp2_unstable=0, delta_unstable=10

Time for backsubstitution: 1.63 seconds

### RS candidates at layer 1
type: RSZ, layer: 1, pos: 68
type: RSZ, layer: 1, pos: 83
type: RSZ, layer: 1, pos: 204
type: RSZ, layer: 1, pos: 52

Time for candidate selection: 0.00 seconds

### Candidate
type: RSZ, layer: 1, pos: 68

### Relational analysis RSZ of RS_RSZ1_RSZ2_RSZ2_RSZ2_RSZ1_RSZ1

#### Relational analysis RSZ result of RS_RSZ1_RSZ2_RSZ2_RSZ2_RSZ1_RSZ1
Status: Status.UNKNOWN
Output dim: 1, lower bound: -0.0007131, upper bound: 0.0007131
time: 0.62 seconds

### Relational analysis RSZ of RS_RSZ1_RSZ2_RSZ2_RSZ2_RSZ1_RSZ2

#### Relational analysis RSZ result of RS_RSZ1_RSZ2_RSZ2_RSZ2_RSZ1_RSZ2
Status: Status.UNKNOWN
Output dim: 1, lower bound: -0.0007131, upper bound: 0.0007131
time: 0.58 seconds

## BFS RS instance: RS_RSZ1_RSZ2_RSZ2_RSZ2_RSZ2

### Backsubstitution after applying RS history:
0: 0.0007184, 0.0011172, 0.0007184, 0.0011172, -0.0003988, 0.0003988
1: 0.9934075, 0.9943579, 0.9934075, 0.9943579, -0.0009505, 0.0009505
2: -0.0087946, -0.0055215, -0.0087946, -0.0055215, -0.0029944, 0.0029845
3: 0.0035868, 0.0041619, 0.0035868, 0.0041619, -0.0005751, 0.0005751
4: 0.0027809, 0.0053678, 0.0027809, 0.0053678, -0.0025869, 0.0025869
5: 0.0050646, 0.0064029, 0.0050646, 0.0064029, -0.0013382, 0.0013382
6: -0.0021738, -0.0008562, -0.0021738, -0.0008562, -0.0013176, 0.0013176
7: -0.0082275, -0.0074685, -0.0082275, -0.0074685, -0.0007590, 0.0007590
8: 0.0055062, 0.0098066, 0.0055062, 0.0098066, -0.0042229, 0.0042253
9: -0.0036825, -0.0031002, -0.0036825, -0.0031002, -0.0005823, 0.0005823

### Unstable ReLU Count (Linear/Conv2D Layers)
- layer_idx=0, type=LayerType.Linear, total=256, inp1_unstable=9, inp2_unstable=9, delta_unstable=251
- layer_idx=2, type=LayerType.Linear, total=256, inp1_unstable=6, inp2_unstable=6, delta_unstable=256
- layer_idx=4, type=LayerType.Linear, total=256, inp1_unstable=0, inp2_unstable=0, delta_unstable=256
- layer_idx=6, type=LayerType.Linear, total=256, inp1_unstable=2, inp2_unstable=2, delta_unstable=256
- layer_idx=8, type=LayerType.Linear, total=10, inp1_unstable=0, inp2_unstable=0, delta_unstable=10

Time for backsubstitution: 1.59 seconds

### RS candidates at layer 1
type: RSZ, layer: 1, pos: 52
type: RSZ, layer: 1, pos: 204
type: RSZ, layer: 1, pos: 83
type: RSZ, layer: 1, pos: 68

Time for candidate selection: 0.00 seconds

### Candidate
type: RSZ, layer: 1, pos: 52

### Relational analysis RSZ of RS_RSZ1_RSZ2_RSZ2_RSZ2_RSZ2_RSZ1

#### Relational analysis RSZ result of RS_RSZ1_RSZ2_RSZ2_RSZ2_RSZ2_RSZ1
Status: Status.UNKNOWN
Output dim: 1, lower bound: -0.0007131, upper bound: 0.0007131
time: 0.68 seconds

### Relational analysis RSZ of RS_RSZ1_RSZ2_RSZ2_RSZ2_RSZ2_RSZ2

#### Relational analysis RSZ result of RS_RSZ1_RSZ2_RSZ2_RSZ2_RSZ2_RSZ2
Status: Status.UNKNOWN
Output dim: 1, lower bound: -0.0007131, upper bound: 0.0007131
time: 0.61 seconds

## BFS RS instance: RS_RSZ2_RSZ1_RSZ1_RSZ1_RSZ1

### Backsubstitution after applying RS history:
0: 0.0007184, 0.0011172, 0.0007184, 0.0011172, -0.0003988, 0.0003988
1: 0.9934075, 0.9943579, 0.9934075, 0.9943579, -0.0009505, 0.0009505
2: -0.0087946, -0.0055215, -0.0087946, -0.0055215, -0.0029853, 0.0029835
3: 0.0035868, 0.0041619, 0.0035868, 0.0041619, -0.0005751, 0.0005751
4: 0.0027809, 0.0053678, 0.0027809, 0.0053678, -0.0025869, 0.0025869
5: 0.0050646, 0.0064029, 0.0050646, 0.0064029, -0.0013382, 0.0013382
6: -0.0021738, -0.0008562, -0.0021738, -0.0008562, -0.0013176, 0.0013176
7: -0.0082275, -0.0074685, -0.0082275, -0.0074685, -0.0007590, 0.0007590
8: 0.0055062, 0.0098066, 0.0055062, 0.0098066, -0.0042216, 0.0042221
9: -0.0036825, -0.0031002, -0.0036825, -0.0031002, -0.0005823, 0.0005823

### Unstable ReLU Count (Linear/Conv2D Layers)
- layer_idx=0, type=LayerType.Linear, total=256, inp1_unstable=9, inp2_unstable=9, delta_unstable=251
- layer_idx=2, type=LayerType.Linear, total=256, inp1_unstable=6, inp2_unstable=6, delta_unstable=256
- layer_idx=4, type=LayerType.Linear, total=256, inp1_unstable=0, inp2_unstable=0, delta_unstable=256
- layer_idx=6, type=LayerType.Linear, total=256, inp1_unstable=2, inp2_unstable=2, delta_unstable=256
- layer_idx=8, type=LayerType.Linear, total=10, inp1_unstable=0, inp2_unstable=0, delta_unstable=10

Time for backsubstitution: 1.61 seconds

### RS candidates at layer 1
type: RSZ, layer: 1, pos: 176
type: RSZ, layer: 1, pos: 198
type: RSZ, layer: 1, pos: 136
type: RSZ, layer: 1, pos: 199

Time for candidate selection: 0.00 seconds

### Candidate
type: RSZ, layer: 1, pos: 176

### Relational analysis RSZ of RS_RSZ2_RSZ1_RSZ1_RSZ1_RSZ1_RSZ1

#### Relational analysis RSZ result of RS_RSZ2_RSZ1_RSZ1_RSZ1_RSZ1_RSZ1
Status: Status.UNKNOWN
Output dim: 1, lower bound: -0.0007966, upper bound: 0.0007988
time: 0.68 seconds

### Relational analysis RSZ of RS_RSZ2_RSZ1_RSZ1_RSZ1_RSZ1_RSZ2

#### Relational analysis RSZ result of RS_RSZ2_RSZ1_RSZ1_RSZ1_RSZ1_RSZ2
Status: Status.UNKNOWN
Output dim: 1, lower bound: -0.0007966, upper bound: 0.0007988
time: 0.65 seconds

## BFS RS instance: RS_RSZ2_RSZ1_RSZ1_RSZ1_RSZ2

### Backsubstitution after applying RS history:
0: 0.0007184, 0.0011172, 0.0007184, 0.0011172, -0.0003988, 0.0003988
1: 0.9934075, 0.9943579, 0.9934075, 0.9943579, -0.0009505, 0.0009505
2: -0.0087946, -0.0055215, -0.0087946, -0.0055215, -0.0029863, 0.0029825
3: 0.0035868, 0.0041619, 0.0035868, 0.0041619, -0.0005751, 0.0005751
4: 0.0027809, 0.0053678, 0.0027809, 0.0053678, -0.0025869, 0.0025869
5: 0.0050646, 0.0064029, 0.0050646, 0.0064029, -0.0013382, 0.0013382
6: -0.0021738, -0.0008562, -0.0021738, -0.0008562, -0.0013176, 0.0013176
7: -0.0082275, -0.0074685, -0.0082275, -0.0074685, -0.0007590, 0.0007590
8: 0.0055062, 0.0098066, 0.0055062, 0.0098066, -0.0042213, 0.0042223
9: -0.0036825, -0.0031002, -0.0036825, -0.0031002, -0.0005823, 0.0005823

### Unstable ReLU Count (Linear/Conv2D Layers)
- layer_idx=0, type=LayerType.Linear, total=256, inp1_unstable=9, inp2_unstable=9, delta_unstable=251
- layer_idx=2, type=LayerType.Linear, total=256, inp1_unstable=6, inp2_unstable=6, delta_unstable=256
- layer_idx=4, type=LayerType.Linear, total=256, inp1_unstable=0, inp2_unstable=0, delta_unstable=256
- layer_idx=6, type=LayerType.Linear, total=256, inp1_unstable=2, inp2_unstable=2, delta_unstable=256
- layer_idx=8, type=LayerType.Linear, total=10, inp1_unstable=0, inp2_unstable=0, delta_unstable=10

Time for backsubstitution: 1.59 seconds

### RS candidates at layer 1
type: RSZ, layer: 1, pos: 176
type: RSZ, layer: 1, pos: 136
type: RSZ, layer: 1, pos: 199
type: RSZ, layer: 1, pos: 198

Time for candidate selection: 0.00 seconds

### Candidate
type: RSZ, layer: 1, pos: 176

### Relational analysis RSZ of RS_RSZ2_RSZ1_RSZ1_RSZ1_RSZ2_RSZ1

#### Relational analysis RSZ result of RS_RSZ2_RSZ1_RSZ1_RSZ1_RSZ2_RSZ1
Status: Status.UNKNOWN
Output dim: 1, lower bound: -0.0007954, upper bound: 0.0008023
time: 0.66 seconds

### Relational analysis RSZ of RS_RSZ2_RSZ1_RSZ1_RSZ1_RSZ2_RSZ2

#### Relational analysis RSZ result of RS_RSZ2_RSZ1_RSZ1_RSZ1_RSZ2_RSZ2
Status: Status.UNKNOWN
Output dim: 1, lower bound: -0.0007955, upper bound: 0.0008020
time: 0.66 seconds

## BFS RS instance: RS_RSZ2_RSZ1_RSZ1_RSZ2_RSZ1

### Backsubstitution after applying RS history:
0: 0.0007184, 0.0011172, 0.0007184, 0.0011172, -0.0003988, 0.0003988
1: 0.9934075, 0.9943579, 0.9934075, 0.9943579, -0.0009505, 0.0009505
2: -0.0087946, -0.0055215, -0.0087946, -0.0055215, -0.0029779, 0.0029766
3: 0.0035868, 0.0041619, 0.0035868, 0.0041619, -0.0005751, 0.0005751
4: 0.0027809, 0.0053678, 0.0027809, 0.0053678, -0.0025869, 0.0025869
5: 0.0050646, 0.0064029, 0.0050646, 0.0064029, -0.0013382, 0.0013382
6: -0.0021738, -0.0008562, -0.0021738, -0.0008562, -0.0013176, 0.0013176
7: -0.0082275, -0.0074685, -0.0082275, -0.0074685, -0.0007590, 0.0007590
8: 0.0055062, 0.0098066, 0.0055062, 0.0098066, -0.0042201, 0.0042205
9: -0.0036825, -0.0031002, -0.0036825, -0.0031002, -0.0005823, 0.0005823

### Unstable ReLU Count (Linear/Conv2D Layers)
- layer_idx=0, type=LayerType.Linear, total=256, inp1_unstable=9, inp2_unstable=9, delta_unstable=251
- layer_idx=2, type=LayerType.Linear, total=256, inp1_unstable=6, inp2_unstable=6, delta_unstable=256
- layer_idx=4, type=LayerType.Linear, total=256, inp1_unstable=0, inp2_unstable=0, delta_unstable=256
- layer_idx=6, type=LayerType.Linear, total=256, inp1_unstable=2, inp2_unstable=2, delta_unstable=256
- layer_idx=8, type=LayerType.Linear, total=10, inp1_unstable=0, inp2_unstable=0, delta_unstable=10

Time for backsubstitution: 1.63 seconds

### RS candidates at layer 1
type: RSZ, layer: 1, pos: 176
type: RSZ, layer: 1, pos: 136
type: RSZ, layer: 1, pos: 52
type: RSZ, layer: 1, pos: 199

Time for candidate selection: 0.00 seconds

### Candidate
type: RSZ, layer: 1, pos: 176

### Relational analysis RSZ of RS_RSZ2_RSZ1_RSZ1_RSZ2_RSZ1_RSZ1

#### Relational analysis RSZ result of RS_RSZ2_RSZ1_RSZ1_RSZ2_RSZ1_RSZ1
Status: Status.UNKNOWN
Output dim: 1, lower bound: -0.0007738, upper bound: 0.0007740
time: 0.66 seconds

### Relational analysis RSZ of RS_RSZ2_RSZ1_RSZ1_RSZ2_RSZ1_RSZ2

#### Relational analysis RSZ result of RS_RSZ2_RSZ1_RSZ1_RSZ2_RSZ1_RSZ2
Status: Status.UNKNOWN
Output dim: 1, lower bound: -0.0007738, upper bound: 0.0007740
time: 0.68 seconds

## BFS RS instance: RS_RSZ2_RSZ1_RSZ1_RSZ2_RSZ2

### Backsubstitution after applying RS history:
0: 0.0007184, 0.0011172, 0.0007184, 0.0011172, -0.0003988, 0.0003988
1: 0.9934075, 0.9943579, 0.9934075, 0.9943579, -0.0009505, 0.0009505
2: -0.0087946, -0.0055215, -0.0087946, -0.0055215, -0.0029885, 0.0029758
3: 0.0035868, 0.0041619, 0.0035868, 0.0041619, -0.0005751, 0.0005751
4: 0.0027809, 0.0053678, 0.0027809, 0.0053678, -0.0025869, 0.0025869
5: 0.0050646, 0.0064029, 0.0050646, 0.0064029, -0.0013382, 0.0013382
6: -0.0021738, -0.0008562, -0.0021738, -0.0008562, -0.0013176, 0.0013176
7: -0.0082275, -0.0074685, -0.0082275, -0.0074685, -0.0007590, 0.0007590
8: 0.0055062, 0.0098066, 0.0055062, 0.0098066, -0.0042199, 0.0042230
9: -0.0036825, -0.0031002, -0.0036825, -0.0031002, -0.0005823, 0.0005823

### Unstable ReLU Count (Linear/Conv2D Layers)
- layer_idx=0, type=LayerType.Linear, total=256, inp1_unstable=9, inp2_unstable=9, delta_unstable=251
- layer_idx=2, type=LayerType.Linear, total=256, inp1_unstable=6, inp2_unstable=6, delta_unstable=256
- layer_idx=4, type=LayerType.Linear, total=256, inp1_unstable=0, inp2_unstable=0, delta_unstable=256
- layer_idx=6, type=LayerType.Linear, total=256, inp1_unstable=2, inp2_unstable=2, delta_unstable=256
- layer_idx=8, type=LayerType.Linear, total=10, inp1_unstable=0, inp2_unstable=0, delta_unstable=10

Time for backsubstitution: 1.61 seconds

### RS candidates at layer 1
type: RSZ, layer: 1, pos: 52
type: RSZ, layer: 1, pos: 176
type: RSZ, layer: 1, pos: 199
type: RSZ, layer: 1, pos: 136

Time for candidate selection: 0.00 seconds

### Candidate
type: RSZ, layer: 1, pos: 52

### Relational analysis RSZ of RS_RSZ2_RSZ1_RSZ1_RSZ2_RSZ2_RSZ1

#### Relational analysis RSZ result of RS_RSZ2_RSZ1_RSZ1_RSZ2_RSZ2_RSZ1
Status: Status.UNKNOWN
Output dim: 1, lower bound: -0.0007801, upper bound: 0.0007810
time: 0.68 seconds

### Relational analysis RSZ of RS_RSZ2_RSZ1_RSZ1_RSZ2_RSZ2_RSZ2

#### Relational analysis RSZ result of RS_RSZ2_RSZ1_RSZ1_RSZ2_RSZ2_RSZ2
Status: Status.UNKNOWN
Output dim: 1, lower bound: -0.0007800, upper bound: 0.0007810
time: 0.68 seconds

## BFS RS instance: RS_RSZ2_RSZ1_RSZ2_RSZ1_RSZ1

### Backsubstitution after applying RS history:
0: 0.0007184, 0.0011172, 0.0007184, 0.0011172, -0.0003988, 0.0003988
1: 0.9934075, 0.9943579, 0.9934075, 0.9943579, -0.0009505, 0.0009505
2: -0.0087946, -0.0055215, -0.0087946, -0.0055215, -0.0029857, 0.0029846
3: 0.0035868, 0.0041619, 0.0035868, 0.0041619, -0.0005751, 0.0005751
4: 0.0027809, 0.0053678, 0.0027809, 0.0053678, -0.0025869, 0.0025869
5: 0.0050646, 0.0064029, 0.0050646, 0.0064029, -0.0013382, 0.0013382
6: -0.0021738, -0.0008562, -0.0021738, -0.0008562, -0.0013176, 0.0013176
7: -0.0082275, -0.0074685, -0.0082275, -0.0074685, -0.0007590, 0.0007590
8: 0.0055062, 0.0098066, 0.0055062, 0.0098066, -0.0042219, 0.0042222
9: -0.0036825, -0.0031002, -0.0036825, -0.0031002, -0.0005823, 0.0005823

### Unstable ReLU Count (Linear/Conv2D Layers)
- layer_idx=0, type=LayerType.Linear, total=256, inp1_unstable=9, inp2_unstable=9, delta_unstable=251
- layer_idx=2, type=LayerType.Linear, total=256, inp1_unstable=6, inp2_unstable=6, delta_unstable=256
- layer_idx=4, type=LayerType.Linear, total=256, inp1_unstable=0, inp2_unstable=0, delta_unstable=256
- layer_idx=6, type=LayerType.Linear, total=256, inp1_unstable=2, inp2_unstable=2, delta_unstable=256
- layer_idx=8, type=LayerType.Linear, total=10, inp1_unstable=0, inp2_unstable=0, delta_unstable=10

Time for backsubstitution: 1.66 seconds

### RS candidates at layer 1
type: RSZ, layer: 1, pos: 83
type: RSZ, layer: 1, pos: 136
type: RSZ, layer: 1, pos: 198
type: RSZ, layer: 1, pos: 52

Time for candidate selection: 0.00 seconds

### Candidate
type: RSZ, layer: 1, pos: 83

### Relational analysis RSZ of RS_RSZ2_RSZ1_RSZ2_RSZ1_RSZ1_RSZ1

#### Relational analysis RSZ result of RS_RSZ2_RSZ1_RSZ2_RSZ1_RSZ1_RSZ1
Status: Status.UNKNOWN
Output dim: 1, lower bound: -0.0007566, upper bound: 0.0007568
time: 0.59 seconds

### Relational analysis RSZ of RS_RSZ2_RSZ1_RSZ2_RSZ1_RSZ1_RSZ2

#### Relational analysis RSZ result of RS_RSZ2_RSZ1_RSZ2_RSZ1_RSZ1_RSZ2
Status: Status.UNKNOWN
Output dim: 1, lower bound: -0.0007566, upper bound: 0.0007569
time: 0.66 seconds

## BFS RS instance: RS_RSZ2_RSZ1_RSZ2_RSZ1_RSZ2

### Backsubstitution after applying RS history:
0: 0.0007184, 0.0011172, 0.0007184, 0.0011172, -0.0003988, 0.0003988
1: 0.9934075, 0.9943579, 0.9934075, 0.9943579, -0.0009505, 0.0009505
2: -0.0087946, -0.0055215, -0.0087946, -0.0055215, -0.0029863, 0.0029867
3: 0.0035868, 0.0041619, 0.0035868, 0.0041619, -0.0005751, 0.0005751
4: 0.0027809, 0.0053678, 0.0027809, 0.0053678, -0.0025869, 0.0025869
5: 0.0050646, 0.0064029, 0.0050646, 0.0064029, -0.0013382, 0.0013382
6: -0.0021738, -0.0008562, -0.0021738, -0.0008562, -0.0013176, 0.0013176
7: -0.0082275, -0.0074685, -0.0082275, -0.0074685, -0.0007590, 0.0007590
8: 0.0055062, 0.0098066, 0.0055062, 0.0098066, -0.0042224, 0.0042224
9: -0.0036825, -0.0031002, -0.0036825, -0.0031002, -0.0005823, 0.0005823

### Unstable ReLU Count (Linear/Conv2D Layers)
- layer_idx=0, type=LayerType.Linear, total=256, inp1_unstable=9, inp2_unstable=9, delta_unstable=251
- layer_idx=2, type=LayerType.Linear, total=256, inp1_unstable=6, inp2_unstable=6, delta_unstable=256
- layer_idx=4, type=LayerType.Linear, total=256, inp1_unstable=0, inp2_unstable=0, delta_unstable=256
- layer_idx=6, type=LayerType.Linear, total=256, inp1_unstable=2, inp2_unstable=2, delta_unstable=256
- layer_idx=8, type=LayerType.Linear, total=10, inp1_unstable=0, inp2_unstable=0, delta_unstable=10

Time for backsubstitution: 1.61 seconds

### RS candidates at layer 1
type: RSZ, layer: 1, pos: 198
type: RSZ, layer: 1, pos: 52
type: RSZ, layer: 1, pos: 136
type: RSZ, layer: 1, pos: 83

Time for candidate selection: 0.00 seconds

### Candidate
type: RSZ, layer: 1, pos: 198

### Relational analysis RSZ of RS_RSZ2_RSZ1_RSZ2_RSZ1_RSZ2_RSZ1

#### Relational analysis RSZ result of RS_RSZ2_RSZ1_RSZ2_RSZ1_RSZ2_RSZ1
Status: Status.UNKNOWN
Output dim: 1, lower bound: -0.0007131, upper bound: 0.0007131
time: 0.58 seconds

### Relational analysis RSZ of RS_RSZ2_RSZ1_RSZ2_RSZ1_RSZ2_RSZ2

#### Relational analysis RSZ result of RS_RSZ2_RSZ1_RSZ2_RSZ1_RSZ2_RSZ2
Status: Status.UNKNOWN
Output dim: 1, lower bound: -0.0007131, upper bound: 0.0007131
time: 0.59 seconds

## BFS RS instance: RS_RSZ2_RSZ1_RSZ2_RSZ2_RSZ1

### Backsubstitution after applying RS history:
0: 0.0007184, 0.0011172, 0.0007184, 0.0011172, -0.0003988, 0.0003988
1: 0.9934075, 0.9943579, 0.9934075, 0.9943579, -0.0009505, 0.0009505
2: -0.0087946, -0.0055215, -0.0087946, -0.0055215, -0.0029848, 0.0029840
3: 0.0035868, 0.0041619, 0.0035868, 0.0041619, -0.0005751, 0.0005751
4: 0.0027809, 0.0053678, 0.0027809, 0.0053678, -0.0025869, 0.0025869
5: 0.0050646, 0.0064029, 0.0050646, 0.0064029, -0.0013382, 0.0013382
6: -0.0021738, -0.0008562, -0.0021738, -0.0008562, -0.0013176, 0.0013176
7: -0.0082275, -0.0074685, -0.0082275, -0.0074685, -0.0007590, 0.0007590
8: 0.0055062, 0.0098066, 0.0055062, 0.0098066, -0.0042218, 0.0042220
9: -0.0036825, -0.0031002, -0.0036825, -0.0031002, -0.0005823, 0.0005823

### Unstable ReLU Count (Linear/Conv2D Layers)
- layer_idx=0, type=LayerType.Linear, total=256, inp1_unstable=9, inp2_unstable=9, delta_unstable=251
- layer_idx=2, type=LayerType.Linear, total=256, inp1_unstable=6, inp2_unstable=6, delta_unstable=256
- layer_idx=4, type=LayerType.Linear, total=256, inp1_unstable=0, inp2_unstable=0, delta_unstable=256
- layer_idx=6, type=LayerType.Linear, total=256, inp1_unstable=2, inp2_unstable=2, delta_unstable=256
- layer_idx=8, type=LayerType.Linear, total=10, inp1_unstable=0, inp2_unstable=0, delta_unstable=10

Time for backsubstitution: 1.59 seconds

### RS candidates at layer 1
type: RSZ, layer: 1, pos: 136
type: RSZ, layer: 1, pos: 199
type: RSZ, layer: 1, pos: 198
type: RSZ, layer: 1, pos: 52

Time for candidate selection: 0.00 seconds

### Candidate
type: RSZ, layer: 1, pos: 136

### Relational analysis RSZ of RS_RSZ2_RSZ1_RSZ2_RSZ2_RSZ1_RSZ1

#### Relational analysis RSZ result of RS_RSZ2_RSZ1_RSZ2_RSZ2_RSZ1_RSZ1
Status: Status.UNKNOWN
Output dim: 1, lower bound: -0.0007940, upper bound: 0.0007952
time: 0.74 seconds

### Relational analysis RSZ of RS_RSZ2_RSZ1_RSZ2_RSZ2_RSZ1_RSZ2

#### Relational analysis RSZ result of RS_RSZ2_RSZ1_RSZ2_RSZ2_RSZ1_RSZ2
Status: Status.UNKNOWN
Output dim: 1, lower bound: -0.0007939, upper bound: 0.0007954
time: 0.68 seconds

## BFS RS instance: RS_RSZ2_RSZ1_RSZ2_RSZ2_RSZ2

### Backsubstitution after applying RS history:
0: 0.0007184, 0.0011172, 0.0007184, 0.0011172, -0.0003988, 0.0003988
1: 0.9934075, 0.9943579, 0.9934075, 0.9943579, -0.0009505, 0.0009505
2: -0.0087946, -0.0055215, -0.0087946, -0.0055215, -0.0029846, 0.0029843
3: 0.0035868, 0.0041619, 0.0035868, 0.0041619, -0.0005751, 0.0005751
4: 0.0027809, 0.0053678, 0.0027809, 0.0053678, -0.0025869, 0.0025869
5: 0.0050646, 0.0064029, 0.0050646, 0.0064029, -0.0013382, 0.0013382
6: -0.0021738, -0.0008562, -0.0021738, -0.0008562, -0.0013176, 0.0013176
7: -0.0082275, -0.0074685, -0.0082275, -0.0074685, -0.0007590, 0.0007590
8: 0.0055062, 0.0098066, 0.0055062, 0.0098066, -0.0042219, 0.0042220
9: -0.0036825, -0.0031002, -0.0036825, -0.0031002, -0.0005823, 0.0005823

### Unstable ReLU Count (Linear/Conv2D Layers)
- layer_idx=0, type=LayerType.Linear, total=256, inp1_unstable=9, inp2_unstable=9, delta_unstable=251
- layer_idx=2, type=LayerType.Linear, total=256, inp1_unstable=6, inp2_unstable=6, delta_unstable=256
- layer_idx=4, type=LayerType.Linear, total=256, inp1_unstable=0, inp2_unstable=0, delta_unstable=256
- layer_idx=6, type=LayerType.Linear, total=256, inp1_unstable=2, inp2_unstable=2, delta_unstable=256
- layer_idx=8, type=LayerType.Linear, total=10, inp1_unstable=0, inp2_unstable=0, delta_unstable=10

Time for backsubstitution: 1.65 seconds

### RS candidates at layer 1
type: RSZ, layer: 1, pos: 136
type: RSZ, layer: 1, pos: 199
type: RSZ, layer: 1, pos: 52
type: RSZ, layer: 1, pos: 198

Time for candidate selection: 0.00 seconds

### Candidate
type: RSZ, layer: 1, pos: 136

### Relational analysis RSZ of RS_RSZ2_RSZ1_RSZ2_RSZ2_RSZ2_RSZ1

#### Relational analysis RSZ result of RS_RSZ2_RSZ1_RSZ2_RSZ2_RSZ2_RSZ1
Status: Status.UNKNOWN
Output dim: 1, lower bound: -0.0007937, upper bound: 0.0007953
time: 0.72 seconds

### Relational analysis RSZ of RS_RSZ2_RSZ1_RSZ2_RSZ2_RSZ2_RSZ2

#### Relational analysis RSZ result of RS_RSZ2_RSZ1_RSZ2_RSZ2_RSZ2_RSZ2
Status: Status.UNKNOWN
Output dim: 1, lower bound: -0.0007937, upper bound: 0.0007954
time: 0.64 seconds

## BFS RS instance: RS_RSZ2_RSZ2_RSZ1_RSZ1_RSZ1

### Backsubstitution after applying RS history:
0: 0.0007184, 0.0011172, 0.0007184, 0.0011172, -0.0003988, 0.0003988
1: 0.9934075, 0.9943579, 0.9934075, 0.9943579, -0.0009505, 0.0009505
2: -0.0087946, -0.0055215, -0.0087946, -0.0055215, -0.0029797, 0.0029839
3: 0.0035868, 0.0041619, 0.0035868, 0.0041619, -0.0005751, 0.0005751
4: 0.0027809, 0.0053678, 0.0027809, 0.0053678, -0.0025869, 0.0025869
5: 0.0050646, 0.0064029, 0.0050646, 0.0064029, -0.0013382, 0.0013382
6: -0.0021738, -0.0008562, -0.0021738, -0.0008562, -0.0013176, 0.0013176
7: -0.0082275, -0.0074685, -0.0082275, -0.0074685, -0.0007590, 0.0007590
8: 0.0055062, 0.0098066, 0.0055062, 0.0098066, -0.0042220, 0.0042209
9: -0.0036825, -0.0031002, -0.0036825, -0.0031002, -0.0005823, 0.0005823

### Unstable ReLU Count (Linear/Conv2D Layers)
- layer_idx=0, type=LayerType.Linear, total=256, inp1_unstable=9, inp2_unstable=9, delta_unstable=251
- layer_idx=2, type=LayerType.Linear, total=256, inp1_unstable=6, inp2_unstable=6, delta_unstable=256
- layer_idx=4, type=LayerType.Linear, total=256, inp1_unstable=0, inp2_unstable=0, delta_unstable=256
- layer_idx=6, type=LayerType.Linear, total=256, inp1_unstable=2, inp2_unstable=2, delta_unstable=256
- layer_idx=8, type=LayerType.Linear, total=10, inp1_unstable=0, inp2_unstable=0, delta_unstable=10

Time for backsubstitution: 1.61 seconds

### RS candidates at layer 1
type: RSZ, layer: 1, pos: 83
type: RSZ, layer: 1, pos: 198
type: RSZ, layer: 1, pos: 199
type: RSZ, layer: 1, pos: 176

Time for candidate selection: 0.00 seconds

### Candidate
type: RSZ, layer: 1, pos: 83

### Relational analysis RSZ of RS_RSZ2_RSZ2_RSZ1_RSZ1_RSZ1_RSZ1

#### Relational analysis RSZ result of RS_RSZ2_RSZ2_RSZ1_RSZ1_RSZ1_RSZ1
Status: Status.UNKNOWN
Output dim: 1, lower bound: -0.0007975, upper bound: 0.0007935
time: 0.70 seconds

### Relational analysis RSZ of RS_RSZ2_RSZ2_RSZ1_RSZ1_RSZ1_RSZ2

#### Relational analysis RSZ result of RS_RSZ2_RSZ2_RSZ1_RSZ1_RSZ1_RSZ2
Status: Status.UNKNOWN
Output dim: 1, lower bound: -0.0007973, upper bound: 0.0007937
time: 0.67 seconds

## BFS RS instance: RS_RSZ2_RSZ2_RSZ1_RSZ1_RSZ2

### Backsubstitution after applying RS history:
0: 0.0007184, 0.0011172, 0.0007184, 0.0011172, -0.0003988, 0.0003988
1: 0.9934075, 0.9943579, 0.9934075, 0.9943579, -0.0009505, 0.0009505
2: -0.0087946, -0.0055215, -0.0087946, -0.0055215, -0.0029819, 0.0029816
3: 0.0035868, 0.0041619, 0.0035868, 0.0041619, -0.0005751, 0.0005751
4: 0.0027809, 0.0053678, 0.0027809, 0.0053678, -0.0025869, 0.0025869
5: 0.0050646, 0.0064029, 0.0050646, 0.0064029, -0.0013382, 0.0013382
6: -0.0021738, -0.0008562, -0.0021738, -0.0008562, -0.0013176, 0.0013176
7: -0.0082275, -0.0074685, -0.0082275, -0.0074685, -0.0007590, 0.0007590
8: 0.0055062, 0.0098066, 0.0055062, 0.0098066, -0.0042214, 0.0042215
9: -0.0036825, -0.0031002, -0.0036825, -0.0031002, -0.0005823, 0.0005823

### Unstable ReLU Count (Linear/Conv2D Layers)
- layer_idx=0, type=LayerType.Linear, total=256, inp1_unstable=9, inp2_unstable=9, delta_unstable=251
- layer_idx=2, type=LayerType.Linear, total=256, inp1_unstable=6, inp2_unstable=6, delta_unstable=256
- layer_idx=4, type=LayerType.Linear, total=256, inp1_unstable=0, inp2_unstable=0, delta_unstable=256
- layer_idx=6, type=LayerType.Linear, total=256, inp1_unstable=2, inp2_unstable=2, delta_unstable=256
- layer_idx=8, type=LayerType.Linear, total=10, inp1_unstable=0, inp2_unstable=0, delta_unstable=10

Time for backsubstitution: 1.60 seconds

### RS candidates at layer 1
type: RSZ, layer: 1, pos: 198
type: RSZ, layer: 1, pos: 176
type: RSZ, layer: 1, pos: 83
type: RSZ, layer: 1, pos: 199

Time for candidate selection: 0.00 seconds

### Candidate
type: RSZ, layer: 1, pos: 198

### Relational analysis RSZ of RS_RSZ2_RSZ2_RSZ1_RSZ1_RSZ2_RSZ1

#### Relational analysis RSZ result of RS_RSZ2_RSZ2_RSZ1_RSZ1_RSZ2_RSZ1
Status: Status.UNKNOWN
Output dim: 1, lower bound: -0.0007800, upper bound: 0.0007785
time: 0.70 seconds

### Relational analysis RSZ of RS_RSZ2_RSZ2_RSZ1_RSZ1_RSZ2_RSZ2

#### Relational analysis RSZ result of RS_RSZ2_RSZ2_RSZ1_RSZ1_RSZ2_RSZ2
Status: Status.UNKNOWN
Output dim: 1, lower bound: -0.0007800, upper bound: 0.0007785
time: 0.69 seconds

## BFS RS instance: RS_RSZ2_RSZ2_RSZ1_RSZ2_RSZ1

### Backsubstitution after applying RS history:
0: 0.0007184, 0.0011172, 0.0007184, 0.0011172, -0.0003988, 0.0003988
1: 0.9934075, 0.9943579, 0.9934075, 0.9943579, -0.0009505, 0.0009505
2: -0.0087946, -0.0055215, -0.0087946, -0.0055215, -0.0029791, 0.0029845
3: 0.0035868, 0.0041619, 0.0035868, 0.0041619, -0.0005751, 0.0005751
4: 0.0027809, 0.0053678, 0.0027809, 0.0053678, -0.0025869, 0.0025869
5: 0.0050646, 0.0064029, 0.0050646, 0.0064029, -0.0013382, 0.0013382
6: -0.0021738, -0.0008562, -0.0021738, -0.0008562, -0.0013176, 0.0013176
7: -0.0082275, -0.0074685, -0.0082275, -0.0074685, -0.0007590, 0.0007590
8: 0.0055062, 0.0098066, 0.0055062, 0.0098066, -0.0042222, 0.0042208
9: -0.0036825, -0.0031002, -0.0036825, -0.0031002, -0.0005823, 0.0005823

### Unstable ReLU Count (Linear/Conv2D Layers)
- layer_idx=0, type=LayerType.Linear, total=256, inp1_unstable=9, inp2_unstable=9, delta_unstable=251
- layer_idx=2, type=LayerType.Linear, total=256, inp1_unstable=6, inp2_unstable=6, delta_unstable=256
- layer_idx=4, type=LayerType.Linear, total=256, inp1_unstable=0, inp2_unstable=0, delta_unstable=256
- layer_idx=6, type=LayerType.Linear, total=256, inp1_unstable=2, inp2_unstable=2, delta_unstable=256
- layer_idx=8, type=LayerType.Linear, total=10, inp1_unstable=0, inp2_unstable=0, delta_unstable=10

Time for backsubstitution: 1.60 seconds

### RS candidates at layer 1
type: RSZ, layer: 1, pos: 83
type: RSZ, layer: 1, pos: 198
type: RSZ, layer: 1, pos: 199
type: RSZ, layer: 1, pos: 176

Time for candidate selection: 0.00 seconds

### Candidate
type: RSZ, layer: 1, pos: 83

### Relational analysis RSZ of RS_RSZ2_RSZ2_RSZ1_RSZ2_RSZ1_RSZ1

#### Relational analysis RSZ result of RS_RSZ2_RSZ2_RSZ1_RSZ2_RSZ1_RSZ1
Status: Status.UNKNOWN
Output dim: 1, lower bound: -0.0007985, upper bound: 0.0007925
time: 0.66 seconds

### Relational analysis RSZ of RS_RSZ2_RSZ2_RSZ1_RSZ2_RSZ1_RSZ2

#### Relational analysis RSZ result of RS_RSZ2_RSZ2_RSZ1_RSZ2_RSZ1_RSZ2
Status: Status.UNKNOWN
Output dim: 1, lower bound: -0.0007983, upper bound: 0.0007925
time: 0.61 seconds

## BFS RS instance: RS_RSZ2_RSZ2_RSZ1_RSZ2_RSZ2

### Backsubstitution after applying RS history:
0: 0.0007184, 0.0011172, 0.0007184, 0.0011172, -0.0003988, 0.0003988
1: 0.9934075, 0.9943579, 0.9934075, 0.9943579, -0.0009505, 0.0009505
2: -0.0087946, -0.0055215, -0.0087946, -0.0055215, -0.0029813, 0.0029823
3: 0.0035868, 0.0041619, 0.0035868, 0.0041619, -0.0005751, 0.0005751
4: 0.0027809, 0.0053678, 0.0027809, 0.0053678, -0.0025869, 0.0025869
5: 0.0050646, 0.0064029, 0.0050646, 0.0064029, -0.0013382, 0.0013382
6: -0.0021738, -0.0008562, -0.0021738, -0.0008562, -0.0013176, 0.0013176
7: -0.0082275, -0.0074685, -0.0082275, -0.0074685, -0.0007590, 0.0007590
8: 0.0055062, 0.0098066, 0.0055062, 0.0098066, -0.0042216, 0.0042213
9: -0.0036825, -0.0031002, -0.0036825, -0.0031002, -0.0005823, 0.0005823

### Unstable ReLU Count (Linear/Conv2D Layers)
- layer_idx=0, type=LayerType.Linear, total=256, inp1_unstable=9, inp2_unstable=9, delta_unstable=251
- layer_idx=2, type=LayerType.Linear, total=256, inp1_unstable=6, inp2_unstable=6, delta_unstable=256
- layer_idx=4, type=LayerType.Linear, total=256, inp1_unstable=0, inp2_unstable=0, delta_unstable=256
- layer_idx=6, type=LayerType.Linear, total=256, inp1_unstable=2, inp2_unstable=2, delta_unstable=256
- layer_idx=8, type=LayerType.Linear, total=10, inp1_unstable=0, inp2_unstable=0, delta_unstable=10

Time for backsubstitution: 1.71 seconds

### RS candidates at layer 1
type: RSZ, layer: 1, pos: 83
type: RSZ, layer: 1, pos: 198
type: RSZ, layer: 1, pos: 176
type: RSZ, layer: 1, pos: 199

Time for candidate selection: 0.00 seconds

### Candidate
type: RSZ, layer: 1, pos: 83

### Relational analysis RSZ of RS_RSZ2_RSZ2_RSZ1_RSZ2_RSZ2_RSZ1

#### Relational analysis RSZ result of RS_RSZ2_RSZ2_RSZ1_RSZ2_RSZ2_RSZ1
Status: Status.UNKNOWN
Output dim: 1, lower bound: -0.0007984, upper bound: 0.0007925
time: 0.67 seconds

### Relational analysis RSZ of RS_RSZ2_RSZ2_RSZ1_RSZ2_RSZ2_RSZ2

#### Relational analysis RSZ result of RS_RSZ2_RSZ2_RSZ1_RSZ2_RSZ2_RSZ2
Status: Status.UNKNOWN
Output dim: 1, lower bound: -0.0007980, upper bound: 0.0007925
time: 0.67 seconds

## BFS RS instance: RS_RSZ2_RSZ2_RSZ2_RSZ1_RSZ1

### Backsubstitution after applying RS history:
0: 0.0007184, 0.0011172, 0.0007184, 0.0011172, -0.0003988, 0.0003988
1: 0.9934075, 0.9943579, 0.9934075, 0.9943579, -0.0009505, 0.0009505
2: -0.0087946, -0.0055215, -0.0087946, -0.0055215, -0.0029795, 0.0029804
3: 0.0035868, 0.0041619, 0.0035868, 0.0041619, -0.0005751, 0.0005751
4: 0.0027809, 0.0053678, 0.0027809, 0.0053678, -0.0025869, 0.0025869
5: 0.0050646, 0.0064029, 0.0050646, 0.0064029, -0.0013382, 0.0013382
6: -0.0021738, -0.0008562, -0.0021738, -0.0008562, -0.0013176, 0.0013176
7: -0.0082275, -0.0074685, -0.0082275, -0.0074685, -0.0007590, 0.0007590
8: 0.0055062, 0.0098066, 0.0055062, 0.0098066, -0.0042209, 0.0042207
9: -0.0036825, -0.0031002, -0.0036825, -0.0031002, -0.0005823, 0.0005823

### Unstable ReLU Count (Linear/Conv2D Layers)
- layer_idx=0, type=LayerType.Linear, total=256, inp1_unstable=9, inp2_unstable=9, delta_unstable=251
- layer_idx=2, type=LayerType.Linear, total=256, inp1_unstable=6, inp2_unstable=6, delta_unstable=256
- layer_idx=4, type=LayerType.Linear, total=256, inp1_unstable=0, inp2_unstable=0, delta_unstable=256
- layer_idx=6, type=LayerType.Linear, total=256, inp1_unstable=2, inp2_unstable=2, delta_unstable=256
- layer_idx=8, type=LayerType.Linear, total=10, inp1_unstable=0, inp2_unstable=0, delta_unstable=10

Time for backsubstitution: 1.61 seconds

### RS candidates at layer 1
type: RSZ, layer: 1, pos: 199
type: RSZ, layer: 1, pos: 83
type: RSZ, layer: 1, pos: 204
type: RSZ, layer: 1, pos: 136

Time for candidate selection: 0.00 seconds

### Candidate
type: RSZ, layer: 1, pos: 199

### Relational analysis RSZ of RS_RSZ2_RSZ2_RSZ2_RSZ1_RSZ1_RSZ1

#### Relational analysis RSZ result of RS_RSZ2_RSZ2_RSZ2_RSZ1_RSZ1_RSZ1
Status: Status.UNKNOWN
Output dim: 1, lower bound: -0.0007131, upper bound: 0.0007131
time: 0.60 seconds

### Relational analysis RSZ of RS_RSZ2_RSZ2_RSZ2_RSZ1_RSZ1_RSZ2

#### Relational analysis RSZ result of RS_RSZ2_RSZ2_RSZ2_RSZ1_RSZ1_RSZ2
Status: Status.UNKNOWN
Output dim: 1, lower bound: -0.0007131, upper bound: 0.0007131
time: 0.65 seconds

## BFS RS instance: RS_RSZ2_RSZ2_RSZ2_RSZ1_RSZ2

### Backsubstitution after applying RS history:
0: 0.0007184, 0.0011172, 0.0007184, 0.0011172, -0.0003988, 0.0003988
1: 0.9934075, 0.9943579, 0.9934075, 0.9943579, -0.0009505, 0.0009505
2: -0.0087946, -0.0055215, -0.0087946, -0.0055215, -0.0029790, 0.0029810
3: 0.0035868, 0.0041619, 0.0035868, 0.0041619, -0.0005751, 0.0005751
4: 0.0027809, 0.0053678, 0.0027809, 0.0053678, -0.0025869, 0.0025869
5: 0.0050646, 0.0064029, 0.0050646, 0.0064029, -0.0013382, 0.0013382
6: -0.0021738, -0.0008562, -0.0021738, -0.0008562, -0.0013176, 0.0013176
7: -0.0082275, -0.0074685, -0.0082275, -0.0074685, -0.0007590, 0.0007590
8: 0.0055062, 0.0098066, 0.0055062, 0.0098066, -0.0042211, 0.0042205
9: -0.0036825, -0.0031002, -0.0036825, -0.0031002, -0.0005823, 0.0005823

### Unstable ReLU Count (Linear/Conv2D Layers)
- layer_idx=0, type=LayerType.Linear, total=256, inp1_unstable=9, inp2_unstable=9, delta_unstable=251
- layer_idx=2, type=LayerType.Linear, total=256, inp1_unstable=6, inp2_unstable=6, delta_unstable=256
- layer_idx=4, type=LayerType.Linear, total=256, inp1_unstable=0, inp2_unstable=0, delta_unstable=256
- layer_idx=6, type=LayerType.Linear, total=256, inp1_unstable=2, inp2_unstable=2, delta_unstable=256
- layer_idx=8, type=LayerType.Linear, total=10, inp1_unstable=0, inp2_unstable=0, delta_unstable=10

Time for backsubstitution: 1.62 seconds

### RS candidates at layer 1
type: RSZ, layer: 1, pos: 199
type: RSZ, layer: 1, pos: 83
type: RSZ, layer: 1, pos: 204
type: RSZ, layer: 1, pos: 136

Time for candidate selection: 0.00 seconds

### Candidate
type: RSZ, layer: 1, pos: 199

### Relational analysis RSZ of RS_RSZ2_RSZ2_RSZ2_RSZ1_RSZ2_RSZ1

#### Relational analysis RSZ result of RS_RSZ2_RSZ2_RSZ2_RSZ1_RSZ2_RSZ1
Status: Status.UNKNOWN
Output dim: 1, lower bound: -0.0007131, upper bound: 0.0007131
time: 0.71 seconds

### Relational analysis RSZ of RS_RSZ2_RSZ2_RSZ2_RSZ1_RSZ2_RSZ2

#### Relational analysis RSZ result of RS_RSZ2_RSZ2_RSZ2_RSZ1_RSZ2_RSZ2
Status: Status.UNKNOWN
Output dim: 1, lower bound: -0.0007131, upper bound: 0.0007131
time: 0.72 seconds

## BFS RS instance: RS_RSZ2_RSZ2_RSZ2_RSZ2_RSZ1

### Backsubstitution after applying RS history:
0: 0.0007184, 0.0011172, 0.0007184, 0.0011172, -0.0003988, 0.0003988
1: 0.9934075, 0.9943579, 0.9934075, 0.9943579, -0.0009505, 0.0009505
2: -0.0087946, -0.0055215, -0.0087946, -0.0055215, -0.0029862, 0.0029784
3: 0.0035868, 0.0041619, 0.0035868, 0.0041619, -0.0005751, 0.0005751
4: 0.0027809, 0.0053678, 0.0027809, 0.0053678, -0.0025869, 0.0025869
5: 0.0050646, 0.0064029, 0.0050646, 0.0064029, -0.0013382, 0.0013382
6: -0.0021738, -0.0008562, -0.0021738, -0.0008562, -0.0013176, 0.0013176
7: -0.0082275, -0.0074685, -0.0082275, -0.0074685, -0.0007590, 0.0007590
8: 0.0055062, 0.0098066, 0.0055062, 0.0098066, -0.0042208, 0.0042226
9: -0.0036825, -0.0031002, -0.0036825, -0.0031002, -0.0005823, 0.0005823

### Unstable ReLU Count (Linear/Conv2D Layers)
- layer_idx=0, type=LayerType.Linear, total=256, inp1_unstable=9, inp2_unstable=9, delta_unstable=251
- layer_idx=2, type=LayerType.Linear, total=256, inp1_unstable=6, inp2_unstable=6, delta_unstable=256
- layer_idx=4, type=LayerType.Linear, total=256, inp1_unstable=0, inp2_unstable=0, delta_unstable=256
- layer_idx=6, type=LayerType.Linear, total=256, inp1_unstable=2, inp2_unstable=2, delta_unstable=256
- layer_idx=8, type=LayerType.Linear, total=10, inp1_unstable=0, inp2_unstable=0, delta_unstable=10

Time for backsubstitution: 1.83 seconds

### RS candidates at layer 1
type: RSZ, layer: 1, pos: 204
type: RSZ, layer: 1, pos: 176
type: RSZ, layer: 1, pos: 199
type: RSZ, layer: 1, pos: 83

Time for candidate selection: 0.00 seconds

### Candidate
type: RSZ, layer: 1, pos: 204

### Relational analysis RSZ of RS_RSZ2_RSZ2_RSZ2_RSZ2_RSZ1_RSZ1

#### Relational analysis RSZ result of RS_RSZ2_RSZ2_RSZ2_RSZ2_RSZ1_RSZ1
Status: Status.UNKNOWN
Output dim: 1, lower bound: -0.0007796, upper bound: 0.0007786
time: 0.79 seconds

### Relational analysis RSZ of RS_RSZ2_RSZ2_RSZ2_RSZ2_RSZ1_RSZ2

#### Relational analysis RSZ result of RS_RSZ2_RSZ2_RSZ2_RSZ2_RSZ1_RSZ2
Status: Status.UNKNOWN
Output dim: 1, lower bound: -0.0007799, upper bound: 0.0007775
time: 0.77 seconds

## BFS RS instance: RS_RSZ2_RSZ2_RSZ2_RSZ2_RSZ2

### Backsubstitution after applying RS history:
0: 0.0007184, 0.0011172, 0.0007184, 0.0011172, -0.0003988, 0.0003988
1: 0.9934075, 0.9943579, 0.9934075, 0.9943579, -0.0009505, 0.0009505
2: -0.0087946, -0.0055215, -0.0087946, -0.0055215, -0.0029884, 0.0029762
3: 0.0035868, 0.0041619, 0.0035868, 0.0041619, -0.0005751, 0.0005751
4: 0.0027809, 0.0053678, 0.0027809, 0.0053678, -0.0025869, 0.0025869
5: 0.0050646, 0.0064029, 0.0050646, 0.0064029, -0.0013382, 0.0013382
6: -0.0021738, -0.0008562, -0.0021738, -0.0008562, -0.0013176, 0.0013176
7: -0.0082275, -0.0074685, -0.0082275, -0.0074685, -0.0007590, 0.0007590
8: 0.0055062, 0.0098066, 0.0055062, 0.0098066, -0.0042202, 0.0042232
9: -0.0036825, -0.0031002, -0.0036825, -0.0031002, -0.0005823, 0.0005823

### Unstable ReLU Count (Linear/Conv2D Layers)
- layer_idx=0, type=LayerType.Linear, total=256, inp1_unstable=9, inp2_unstable=9, delta_unstable=251
- layer_idx=2, type=LayerType.Linear, total=256, inp1_unstable=6, inp2_unstable=6, delta_unstable=256
- layer_idx=4, type=LayerType.Linear, total=256, inp1_unstable=0, inp2_unstable=0, delta_unstable=256
- layer_idx=6, type=LayerType.Linear, total=256, inp1_unstable=2, inp2_unstable=2, delta_unstable=256
- layer_idx=8, type=LayerType.Linear, total=10, inp1_unstable=0, inp2_unstable=0, delta_unstable=10

Time for backsubstitution: 1.69 seconds

### RS candidates at layer 1
type: RSZ, layer: 1, pos: 204
type: RSZ, layer: 1, pos: 176
type: RSZ, layer: 1, pos: 199
type: RSZ, layer: 1, pos: 83

Time for candidate selection: 0.00 seconds

### Candidate
type: RSZ, layer: 1, pos: 204

### Relational analysis RSZ of RS_RSZ2_RSZ2_RSZ2_RSZ2_RSZ2_RSZ1

#### Relational analysis RSZ result of RS_RSZ2_RSZ2_RSZ2_RSZ2_RSZ2_RSZ1
Status: Status.UNKNOWN
Output dim: 1, lower bound: -0.0007796, upper bound: 0.0007787
time: 0.79 seconds

### Relational analysis RSZ of RS_RSZ2_RSZ2_RSZ2_RSZ2_RSZ2_RSZ2

#### Relational analysis RSZ result of RS_RSZ2_RSZ2_RSZ2_RSZ2_RSZ2_RSZ2
Status: Status.UNKNOWN
Output dim: 1, lower bound: -0.0007800, upper bound: 0.0007774
time: 0.68 seconds

## Summary of splitting (split count: 5)
- Time for RS candidates: 3.17 seconds
RS_RSZ1_RSZ1_RSZ1_RSZ1_RSZ1_RSZ1, status: Status.UNKNOWN, split count: 6, time: 3.17
Output dim: 1, lower bound: -0.0007589, upper bound: 0.0007582
RS_RSZ1_RSZ1_RSZ1_RSZ1_RSZ1_RSZ2, status: Status.UNKNOWN, split count: 6, time: 3.17
Output dim: 1, lower bound: -0.0007589, upper bound: 0.0007582
RS_RSZ1_RSZ1_RSZ1_RSZ1_RSZ2_RSZ1, status: Status.UNKNOWN, split count: 6, time: 3.17
Output dim: 1, lower bound: -0.0007749, upper bound: 0.0007751
RS_RSZ1_RSZ1_RSZ1_RSZ1_RSZ2_RSZ2, status: Status.UNKNOWN, split count: 6, time: 3.17
Output dim: 1, lower bound: -0.0007749, upper bound: 0.0007751
RS_RSZ1_RSZ1_RSZ1_RSZ2_RSZ1_RSZ1, status: Status.UNKNOWN, split count: 6, time: 3.17
Output dim: 1, lower bound: -0.0007916, upper bound: 0.0007958
RS_RSZ1_RSZ1_RSZ1_RSZ2_RSZ1_RSZ2, status: Status.UNKNOWN, split count: 6, time: 3.17
Output dim: 1, lower bound: -0.0007953, upper bound: 0.0007937
RS_RSZ1_RSZ1_RSZ1_RSZ2_RSZ2_RSZ1, status: Status.UNKNOWN, split count: 6, time: 3.17
Output dim: 1, lower bound: -0.0007734, upper bound: 0.0007735
RS_RSZ1_RSZ1_RSZ1_RSZ2_RSZ2_RSZ2, status: Status.UNKNOWN, split count: 6, time: 3.17
Output dim: 1, lower bound: -0.0007734, upper bound: 0.0007735
RS_RSZ1_RSZ1_RSZ2_RSZ1_RSZ1_RSZ1, status: Status.UNKNOWN, split count: 6, time: 3.17
Output dim: 1, lower bound: -0.0007740, upper bound: 0.0007739
RS_RSZ1_RSZ1_RSZ2_RSZ1_RSZ1_RSZ2, status: Status.UNKNOWN, split count: 6, time: 3.17
Output dim: 1, lower bound: -0.0007740, upper bound: 0.0007739
RS_RSZ1_RSZ1_RSZ2_RSZ1_RSZ2_RSZ1, status: Status.UNKNOWN, split count: 6, time: 3.17
Output dim: 1, lower bound: -0.0007131, upper bound: 0.0007131
RS_RSZ1_RSZ1_RSZ2_RSZ1_RSZ2_RSZ2, status: Status.UNKNOWN, split count: 6, time: 3.17
Output dim: 1, lower bound: -0.0007131, upper bound: 0.0007131
RS_RSZ1_RSZ1_RSZ2_RSZ2_RSZ1_RSZ1, status: Status.UNKNOWN, split count: 6, time: 3.17
Output dim: 1, lower bound: -0.0007739, upper bound: 0.0007748
RS_RSZ1_RSZ1_RSZ2_RSZ2_RSZ1_RSZ2, status: Status.UNKNOWN, split count: 6, time: 3.17
Output dim: 1, lower bound: -0.0007739, upper bound: 0.0007748
RS_RSZ1_RSZ1_RSZ2_RSZ2_RSZ2_RSZ1, status: Status.UNKNOWN, split count: 6, time: 3.17
Output dim: 1, lower bound: -0.0007765, upper bound: 0.0007759
RS_RSZ1_RSZ1_RSZ2_RSZ2_RSZ2_RSZ2, status: Status.UNKNOWN, split count: 6, time: 3.17
Output dim: 1, lower bound: -0.0007765, upper bound: 0.0007760
RS_RSZ1_RSZ2_RSZ1_RSZ1_RSZ1_RSZ1, status: Status.UNKNOWN, split count: 6, time: 3.17
Output dim: 1, lower bound: -0.0007773, upper bound: 0.0007771
RS_RSZ1_RSZ2_RSZ1_RSZ1_RSZ1_RSZ2, status: Status.UNKNOWN, split count: 6, time: 3.17
Output dim: 1, lower bound: -0.0007772, upper bound: 0.0007772
RS_RSZ1_RSZ2_RSZ1_RSZ1_RSZ2_RSZ1, status: Status.UNKNOWN, split count: 6, time: 3.17
Output dim: 1, lower bound: -0.0007749, upper bound: 0.0007750
RS_RSZ1_RSZ2_RSZ1_RSZ1_RSZ2_RSZ2, status: Status.UNKNOWN, split count: 6, time: 3.17
Output dim: 1, lower bound: -0.0007750, upper bound: 0.0007750
RS_RSZ1_RSZ2_RSZ1_RSZ2_RSZ1_RSZ1, status: Status.UNKNOWN, split count: 6, time: 3.17
Output dim: 1, lower bound: -0.0007731, upper bound: 0.0007748
RS_RSZ1_RSZ2_RSZ1_RSZ2_RSZ1_RSZ2, status: Status.UNKNOWN, split count: 6, time: 3.17
Output dim: 1, lower bound: -0.0007748, upper bound: 0.0007738
RS_RSZ1_RSZ2_RSZ1_RSZ2_RSZ2_RSZ1, status: Status.UNKNOWN, split count: 6, time: 3.17
Output dim: 1, lower bound: -0.0007752, upper bound: 0.0007749
RS_RSZ1_RSZ2_RSZ1_RSZ2_RSZ2_RSZ2, status: Status.UNKNOWN, split count: 6, time: 3.17
Output dim: 1, lower bound: -0.0007750, upper bound: 0.0007749
RS_RSZ1_RSZ2_RSZ2_RSZ1_RSZ1_RSZ1, status: Status.UNKNOWN, split count: 6, time: 3.17
Output dim: 1, lower bound: -0.0007558, upper bound: 0.0007564
RS_RSZ1_RSZ2_RSZ2_RSZ1_RSZ1_RSZ2, status: Status.UNKNOWN, split count: 6, time: 3.17
Output dim: 1, lower bound: -0.0007562, upper bound: 0.0007560
RS_RSZ1_RSZ2_RSZ2_RSZ1_RSZ2_RSZ1, status: Status.UNKNOWN, split count: 6, time: 3.17
Output dim: 1, lower bound: -0.0007557, upper bound: 0.0007564
RS_RSZ1_RSZ2_RSZ2_RSZ1_RSZ2_RSZ2, status: Status.UNKNOWN, split count: 6, time: 3.17
Output dim: 1, lower bound: -0.0007562, upper bound: 0.0007561
RS_RSZ1_RSZ2_RSZ2_RSZ2_RSZ1_RSZ1, status: Status.UNKNOWN, split count: 6, time: 3.17
Output dim: 1, lower bound: -0.0007131, upper bound: 0.0007131
RS_RSZ1_RSZ2_RSZ2_RSZ2_RSZ1_RSZ2, status: Status.UNKNOWN, split count: 6, time: 3.17
Output dim: 1, lower bound: -0.0007131, upper bound: 0.0007131
RS_RSZ1_RSZ2_RSZ2_RSZ2_RSZ2_RSZ1, status: Status.UNKNOWN, split count: 6, time: 3.17
Output dim: 1, lower bound: -0.0007131, upper bound: 0.0007131
RS_RSZ1_RSZ2_RSZ2_RSZ2_RSZ2_RSZ2, status: Status.UNKNOWN, split count: 6, time: 3.17
Output dim: 1, lower bound: -0.0007131, upper bound: 0.0007131
RS_RSZ2_RSZ1_RSZ1_RSZ1_RSZ1_RSZ1, status: Status.UNKNOWN, split count: 6, time: 3.17
Output dim: 1, lower bound: -0.0007966, upper bound: 0.0007988
RS_RSZ2_RSZ1_RSZ1_RSZ1_RSZ1_RSZ2, status: Status.UNKNOWN, split count: 6, time: 3.17
Output dim: 1, lower bound: -0.0007966, upper bound: 0.0007988
RS_RSZ2_RSZ1_RSZ1_RSZ1_RSZ2_RSZ1, status: Status.UNKNOWN, split count: 6, time: 3.17
Output dim: 1, lower bound: -0.0007954, upper bound: 0.0008023
RS_RSZ2_RSZ1_RSZ1_RSZ1_RSZ2_RSZ2, status: Status.UNKNOWN, split count: 6, time: 3.17
Output dim: 1, lower bound: -0.0007955, upper bound: 0.0008020
RS_RSZ2_RSZ1_RSZ1_RSZ2_RSZ1_RSZ1, status: Status.UNKNOWN, split count: 6, time: 3.17
Output dim: 1, lower bound: -0.0007738, upper bound: 0.0007740
RS_RSZ2_RSZ1_RSZ1_RSZ2_RSZ1_RSZ2, status: Status.UNKNOWN, split count: 6, time: 3.17
Output dim: 1, lower bound: -0.0007738, upper bound: 0.0007740
RS_RSZ2_RSZ1_RSZ1_RSZ2_RSZ2_RSZ1, status: Status.UNKNOWN, split count: 6, time: 3.17
Output dim: 1, lower bound: -0.0007801, upper bound: 0.0007810
RS_RSZ2_RSZ1_RSZ1_RSZ2_RSZ2_RSZ2, status: Status.UNKNOWN, split count: 6, time: 3.17
Output dim: 1, lower bound: -0.0007800, upper bound: 0.0007810
RS_RSZ2_RSZ1_RSZ2_RSZ1_RSZ1_RSZ1, status: Status.UNKNOWN, split count: 6, time: 3.17
Output dim: 1, lower bound: -0.0007566, upper bound: 0.0007568
RS_RSZ2_RSZ1_RSZ2_RSZ1_RSZ1_RSZ2, status: Status.UNKNOWN, split count: 6, time: 3.17
Output dim: 1, lower bound: -0.0007566, upper bound: 0.0007569
RS_RSZ2_RSZ1_RSZ2_RSZ1_RSZ2_RSZ1, status: Status.UNKNOWN, split count: 6, time: 3.17
Output dim: 1, lower bound: -0.0007131, upper bound: 0.0007131
RS_RSZ2_RSZ1_RSZ2_RSZ1_RSZ2_RSZ2, status: Status.UNKNOWN, split count: 6, time: 3.17
Output dim: 1, lower bound: -0.0007131, upper bound: 0.0007131
RS_RSZ2_RSZ1_RSZ2_RSZ2_RSZ1_RSZ1, status: Status.UNKNOWN, split count: 6, time: 3.17
Output dim: 1, lower bound: -0.0007940, upper bound: 0.0007952
RS_RSZ2_RSZ1_RSZ2_RSZ2_RSZ1_RSZ2, status: Status.UNKNOWN, split count: 6, time: 3.17
Output dim: 1, lower bound: -0.0007939, upper bound: 0.0007954
RS_RSZ2_RSZ1_RSZ2_RSZ2_RSZ2_RSZ1, status: Status.UNKNOWN, split count: 6, time: 3.17
Output dim: 1, lower bound: -0.0007937, upper bound: 0.0007953
RS_RSZ2_RSZ1_RSZ2_RSZ2_RSZ2_RSZ2, status: Status.UNKNOWN, split count: 6, time: 3.17
Output dim: 1, lower bound: -0.0007937, upper bound: 0.0007954
RS_RSZ2_RSZ2_RSZ1_RSZ1_RSZ1_RSZ1, status: Status.UNKNOWN, split count: 6, time: 3.17
Output dim: 1, lower bound: -0.0007975, upper bound: 0.0007935
RS_RSZ2_RSZ2_RSZ1_RSZ1_RSZ1_RSZ2, status: Status.UNKNOWN, split count: 6, time: 3.17
Output dim: 1, lower bound: -0.0007973, upper bound: 0.0007937
RS_RSZ2_RSZ2_RSZ1_RSZ1_RSZ2_RSZ1, status: Status.UNKNOWN, split count: 6, time: 3.17
Output dim: 1, lower bound: -0.0007800, upper bound: 0.0007785
RS_RSZ2_RSZ2_RSZ1_RSZ1_RSZ2_RSZ2, status: Status.UNKNOWN, split count: 6, time: 3.17
Output dim: 1, lower bound: -0.0007800, upper bound: 0.0007785
RS_RSZ2_RSZ2_RSZ1_RSZ2_RSZ1_RSZ1, status: Status.UNKNOWN, split count: 6, time: 3.17
Output dim: 1, lower bound: -0.0007985, upper bound: 0.0007925
RS_RSZ2_RSZ2_RSZ1_RSZ2_RSZ1_RSZ2, status: Status.UNKNOWN, split count: 6, time: 3.17
Output dim: 1, lower bound: -0.0007983, upper bound: 0.0007925
RS_RSZ2_RSZ2_RSZ1_RSZ2_RSZ2_RSZ1, status: Status.UNKNOWN, split count: 6, time: 3.17
Output dim: 1, lower bound: -0.0007984, upper bound: 0.0007925
RS_RSZ2_RSZ2_RSZ1_RSZ2_RSZ2_RSZ2, status: Status.UNKNOWN, split count: 6, time: 3.17
Output dim: 1, lower bound: -0.0007980, upper bound: 0.0007925
RS_RSZ2_RSZ2_RSZ2_RSZ1_RSZ1_RSZ1, status: Status.UNKNOWN, split count: 6, time: 3.17
Output dim: 1, lower bound: -0.0007131, upper bound: 0.0007131
RS_RSZ2_RSZ2_RSZ2_RSZ1_RSZ1_RSZ2, status: Status.UNKNOWN, split count: 6, time: 3.17
Output dim: 1, lower bound: -0.0007131, upper bound: 0.0007131
RS_RSZ2_RSZ2_RSZ2_RSZ1_RSZ2_RSZ1, status: Status.UNKNOWN, split count: 6, time: 3.17
Output dim: 1, lower bound: -0.0007131, upper bound: 0.0007131
RS_RSZ2_RSZ2_RSZ2_RSZ1_RSZ2_RSZ2, status: Status.UNKNOWN, split count: 6, time: 3.17
Output dim: 1, lower bound: -0.0007131, upper bound: 0.0007131
RS_RSZ2_RSZ2_RSZ2_RSZ2_RSZ1_RSZ1, status: Status.UNKNOWN, split count: 6, time: 3.17
Output dim: 1, lower bound: -0.0007796, upper bound: 0.0007786
RS_RSZ2_RSZ2_RSZ2_RSZ2_RSZ1_RSZ2, status: Status.UNKNOWN, split count: 6, time: 3.17
Output dim: 1, lower bound: -0.0007799, upper bound: 0.0007775
RS_RSZ2_RSZ2_RSZ2_RSZ2_RSZ2_RSZ1, status: Status.UNKNOWN, split count: 6, time: 3.17
Output dim: 1, lower bound: -0.0007796, upper bound: 0.0007787
RS_RSZ2_RSZ2_RSZ2_RSZ2_RSZ2_RSZ2, status: Status.UNKNOWN, split count: 6, time: 3.17
Output dim: 1, lower bound: -0.0007800, upper bound: 0.0007774

## BFS RS instance: RS_RSZ1_RSZ1_RSZ1_RSZ1_RSZ1_RSZ1

### Backsubstitution after applying RS history:
0: 0.0007184, 0.0011172, 0.0007184, 0.0011172, -0.0003988, 0.0003988
1: 0.9934075, 0.9943579, 0.9934075, 0.9943579, -0.0009505, 0.0009505
2: -0.0087946, -0.0055215, -0.0087946, -0.0055215, -0.0029836, 0.0029860
3: 0.0035868, 0.0041619, 0.0035868, 0.0041619, -0.0005751, 0.0005751
4: 0.0027809, 0.0053678, 0.0027809, 0.0053678, -0.0025869, 0.0025869
5: 0.0050646, 0.0064029, 0.0050646, 0.0064029, -0.0013382, 0.0013382
6: -0.0021738, -0.0008562, -0.0021738, -0.0008562, -0.0013176, 0.0013176
7: -0.0082275, -0.0074685, -0.0082275, -0.0074685, -0.0007590, 0.0007590
8: 0.0055062, 0.0098066, 0.0055062, 0.0098066, -0.0042231, 0.0042224
9: -0.0036825, -0.0031002, -0.0036825, -0.0031002, -0.0005823, 0.0005823

### Unstable ReLU Count (Linear/Conv2D Layers)
- layer_idx=0, type=LayerType.Linear, total=256, inp1_unstable=9, inp2_unstable=9, delta_unstable=250
- layer_idx=2, type=LayerType.Linear, total=256, inp1_unstable=6, inp2_unstable=6, delta_unstable=256
- layer_idx=4, type=LayerType.Linear, total=256, inp1_unstable=0, inp2_unstable=0, delta_unstable=256
- layer_idx=6, type=LayerType.Linear, total=256, inp1_unstable=2, inp2_unstable=2, delta_unstable=256
- layer_idx=8, type=LayerType.Linear, total=10, inp1_unstable=0, inp2_unstable=0, delta_unstable=10

Time for backsubstitution: 1.64 seconds

### RS candidates at layer 1
type: RSZ, layer: 1, pos: 83
type: RSZ, layer: 1, pos: 198
type: RSZ, layer: 1, pos: 68

Time for candidate selection: 0.00 seconds

### Candidate
type: RSZ, layer: 1, pos: 83

### Relational analysis RSZ of RS_RSZ1_RSZ1_RSZ1_RSZ1_RSZ1_RSZ1_RSZ1

#### Relational analysis RSZ result of RS_RSZ1_RSZ1_RSZ1_RSZ1_RSZ1_RSZ1_RSZ1
Status: Status.UNKNOWN
Output dim: 1, lower bound: -0.0007554, upper bound: 0.0007548
time: 0.65 seconds

### Relational analysis RSZ of RS_RSZ1_RSZ1_RSZ1_RSZ1_RSZ1_RSZ1_RSZ2

#### Relational analysis RSZ result of RS_RSZ1_RSZ1_RSZ1_RSZ1_RSZ1_RSZ1_RSZ2
Status: Status.UNKNOWN
Output dim: 1, lower bound: -0.0007554, upper bound: 0.0007548
time: 0.61 seconds

## BFS RS instance: RS_RSZ1_RSZ1_RSZ1_RSZ1_RSZ1_RSZ2

### Backsubstitution after applying RS history:
0: 0.0007184, 0.0011172, 0.0007184, 0.0011172, -0.0003988, 0.0003988
1: 0.9934075, 0.9943579, 0.9934075, 0.9943579, -0.0009505, 0.0009505
2: -0.0087946, -0.0055215, -0.0087946, -0.0055215, -0.0029844, 0.0029880
3: 0.0035868, 0.0041619, 0.0035868, 0.0041619, -0.0005751, 0.0005751
4: 0.0027809, 0.0053678, 0.0027809, 0.0053678, -0.0025869, 0.0025869
5: 0.0050646, 0.0064029, 0.0050646, 0.0064029, -0.0013382, 0.0013382
6: -0.0021738, -0.0008562, -0.0021738, -0.0008562, -0.0013176, 0.0013176
7: -0.0082275, -0.0074685, -0.0082275, -0.0074685, -0.0007590, 0.0007590
8: 0.0055062, 0.0098066, 0.0055062, 0.0098066, -0.0042236, 0.0042226
9: -0.0036825, -0.0031002, -0.0036825, -0.0031002, -0.0005823, 0.0005823

### Unstable ReLU Count (Linear/Conv2D Layers)
- layer_idx=0, type=LayerType.Linear, total=256, inp1_unstable=9, inp2_unstable=9, delta_unstable=250
- layer_idx=2, type=LayerType.Linear, total=256, inp1_unstable=6, inp2_unstable=6, delta_unstable=256
- layer_idx=4, type=LayerType.Linear, total=256, inp1_unstable=0, inp2_unstable=0, delta_unstable=256
- layer_idx=6, type=LayerType.Linear, total=256, inp1_unstable=2, inp2_unstable=2, delta_unstable=256
- layer_idx=8, type=LayerType.Linear, total=10, inp1_unstable=0, inp2_unstable=0, delta_unstable=10

Time for backsubstitution: 1.60 seconds

### RS candidates at layer 1
type: RSZ, layer: 1, pos: 68
type: RSZ, layer: 1, pos: 198
type: RSZ, layer: 1, pos: 83

Time for candidate selection: 0.00 seconds

### Candidate
type: RSZ, layer: 1, pos: 68

### Relational analysis RSZ of RS_RSZ1_RSZ1_RSZ1_RSZ1_RSZ1_RSZ2_RSZ1

#### Relational analysis RSZ result of RS_RSZ1_RSZ1_RSZ1_RSZ1_RSZ1_RSZ2_RSZ1
Status: Status.UNKNOWN
Output dim: 1, lower bound: -0.0007568, upper bound: 0.0007572
time: 0.76 seconds

### Relational analysis RSZ of RS_RSZ1_RSZ1_RSZ1_RSZ1_RSZ1_RSZ2_RSZ2

#### Relational analysis RSZ result of RS_RSZ1_RSZ1_RSZ1_RSZ1_RSZ1_RSZ2_RSZ2
Status: Status.UNKNOWN
Output dim: 1, lower bound: -0.0007578, upper bound: 0.0007566
time: 0.69 seconds

## BFS RS instance: RS_RSZ1_RSZ1_RSZ1_RSZ1_RSZ2_RSZ1

### Backsubstitution after applying RS history:
0: 0.0007184, 0.0011172, 0.0007184, 0.0011172, -0.0003988, 0.0003988
1: 0.9934075, 0.9943579, 0.9934075, 0.9943579, -0.0009505, 0.0009505
2: -0.0087946, -0.0055215, -0.0087946, -0.0055215, -0.0029766, 0.0029771
3: 0.0035868, 0.0041619, 0.0035868, 0.0041619, -0.0005751, 0.0005751
4: 0.0027809, 0.0053678, 0.0027809, 0.0053678, -0.0025869, 0.0025869
5: 0.0050646, 0.0064029, 0.0050646, 0.0064029, -0.0013382, 0.0013382
6: -0.0021738, -0.0008562, -0.0021738, -0.0008562, -0.0013176, 0.0013176
7: -0.0082275, -0.0074685, -0.0082275, -0.0074685, -0.0007590, 0.0007590
8: 0.0055062, 0.0098066, 0.0055062, 0.0098066, -0.0042210, 0.0042208
9: -0.0036825, -0.0031002, -0.0036825, -0.0031002, -0.0005823, 0.0005823

### Unstable ReLU Count (Linear/Conv2D Layers)
- layer_idx=0, type=LayerType.Linear, total=256, inp1_unstable=9, inp2_unstable=9, delta_unstable=250
- layer_idx=2, type=LayerType.Linear, total=256, inp1_unstable=6, inp2_unstable=6, delta_unstable=256
- layer_idx=4, type=LayerType.Linear, total=256, inp1_unstable=0, inp2_unstable=0, delta_unstable=256
- layer_idx=6, type=LayerType.Linear, total=256, inp1_unstable=2, inp2_unstable=2, delta_unstable=256
- layer_idx=8, type=LayerType.Linear, total=10, inp1_unstable=0, inp2_unstable=0, delta_unstable=10

Time for backsubstitution: 1.74 seconds

### RS candidates at layer 1
type: RSZ, layer: 1, pos: 68
type: RSZ, layer: 1, pos: 83
type: RSZ, layer: 1, pos: 199

Time for candidate selection: 0.00 seconds

### Candidate
type: RSZ, layer: 1, pos: 68

### Relational analysis RSZ of RS_RSZ1_RSZ1_RSZ1_RSZ1_RSZ2_RSZ1_RSZ1

#### Relational analysis RSZ result of RS_RSZ1_RSZ1_RSZ1_RSZ1_RSZ2_RSZ1_RSZ1
Status: Status.UNKNOWN
Output dim: 1, lower bound: -0.0007708, upper bound: 0.0007733
time: 0.78 seconds

### Relational analysis RSZ of RS_RSZ1_RSZ1_RSZ1_RSZ1_RSZ2_RSZ1_RSZ2

#### Relational analysis RSZ result of RS_RSZ1_RSZ1_RSZ1_RSZ1_RSZ2_RSZ1_RSZ2
Status: Status.UNKNOWN
Output dim: 1, lower bound: -0.0007727, upper bound: 0.0007718
time: 0.74 seconds

## BFS RS instance: RS_RSZ1_RSZ1_RSZ1_RSZ1_RSZ2_RSZ2

### Backsubstitution after applying RS history:
0: 0.0007184, 0.0011172, 0.0007184, 0.0011172, -0.0003988, 0.0003988
1: 0.9934075, 0.9943579, 0.9934075, 0.9943579, -0.0009505, 0.0009505
2: -0.0087946, -0.0055215, -0.0087946, -0.0055215, -0.0029871, 0.0029765
3: 0.0035868, 0.0041619, 0.0035868, 0.0041619, -0.0005751, 0.0005751
4: 0.0027809, 0.0053678, 0.0027809, 0.0053678, -0.0025869, 0.0025869
5: 0.0050646, 0.0064029, 0.0050646, 0.0064029, -0.0013382, 0.0013382
6: -0.0021738, -0.0008562, -0.0021738, -0.0008562, -0.0013176, 0.0013176
7: -0.0082275, -0.0074685, -0.0082275, -0.0074685, -0.0007590, 0.0007590
8: 0.0055062, 0.0098066, 0.0055062, 0.0098066, -0.0042208, 0.0042234
9: -0.0036825, -0.0031002, -0.0036825, -0.0031002, -0.0005823, 0.0005823

### Unstable ReLU Count (Linear/Conv2D Layers)
- layer_idx=0, type=LayerType.Linear, total=256, inp1_unstable=9, inp2_unstable=9, delta_unstable=250
- layer_idx=2, type=LayerType.Linear, total=256, inp1_unstable=6, inp2_unstable=6, delta_unstable=256
- layer_idx=4, type=LayerType.Linear, total=256, inp1_unstable=0, inp2_unstable=0, delta_unstable=256
- layer_idx=6, type=LayerType.Linear, total=256, inp1_unstable=2, inp2_unstable=2, delta_unstable=256
- layer_idx=8, type=LayerType.Linear, total=10, inp1_unstable=0, inp2_unstable=0, delta_unstable=10

Time for backsubstitution: 1.80 seconds

### RS candidates at layer 1
type: RSZ, layer: 1, pos: 199
type: RSZ, layer: 1, pos: 68
type: RSZ, layer: 1, pos: 83

Time for candidate selection: 0.00 seconds

### Candidate
type: RSZ, layer: 1, pos: 199

### Relational analysis RSZ of RS_RSZ1_RSZ1_RSZ1_RSZ1_RSZ2_RSZ2_RSZ1

#### Relational analysis RSZ result of RS_RSZ1_RSZ1_RSZ1_RSZ1_RSZ2_RSZ2_RSZ1
Status: Status.UNKNOWN
Output dim: 1, lower bound: -0.0007131, upper bound: 0.0007131
time: 0.66 seconds

### Relational analysis RSZ of RS_RSZ1_RSZ1_RSZ1_RSZ1_RSZ2_RSZ2_RSZ2

#### Relational analysis RSZ result of RS_RSZ1_RSZ1_RSZ1_RSZ1_RSZ2_RSZ2_RSZ2
Status: Status.UNKNOWN
Output dim: 1, lower bound: -0.0007131, upper bound: 0.0007131
time: 0.66 seconds

## BFS RS instance: RS_RSZ1_RSZ1_RSZ1_RSZ2_RSZ1_RSZ1

### Backsubstitution after applying RS history:
0: 0.0007184, 0.0011172, 0.0007184, 0.0011172, -0.0003988, 0.0003988
1: 0.9934075, 0.9943579, 0.9934075, 0.9943579, -0.0009505, 0.0009505
2: -0.0087946, -0.0055215, -0.0087946, -0.0055215, -0.0029814, 0.0029761
3: 0.0035868, 0.0041619, 0.0035868, 0.0041619, -0.0005751, 0.0005751
4: 0.0027809, 0.0053678, 0.0027809, 0.0053678, -0.0025869, 0.0025869
5: 0.0050646, 0.0064029, 0.0050646, 0.0064029, -0.0013382, 0.0013382
6: -0.0021738, -0.0008562, -0.0021738, -0.0008562, -0.0013176, 0.0013176
7: -0.0082275, -0.0074685, -0.0082275, -0.0074685, -0.0007590, 0.0007590
8: 0.0055062, 0.0098066, 0.0055062, 0.0098066, -0.0042202, 0.0042216
9: -0.0036825, -0.0031002, -0.0036825, -0.0031002, -0.0005823, 0.0005823

### Unstable ReLU Count (Linear/Conv2D Layers)
- layer_idx=0, type=LayerType.Linear, total=256, inp1_unstable=9, inp2_unstable=9, delta_unstable=250
- layer_idx=2, type=LayerType.Linear, total=256, inp1_unstable=6, inp2_unstable=6, delta_unstable=256
- layer_idx=4, type=LayerType.Linear, total=256, inp1_unstable=0, inp2_unstable=0, delta_unstable=256
- layer_idx=6, type=LayerType.Linear, total=256, inp1_unstable=2, inp2_unstable=2, delta_unstable=256
- layer_idx=8, type=LayerType.Linear, total=10, inp1_unstable=0, inp2_unstable=0, delta_unstable=10

Time for backsubstitution: 1.64 seconds

### RS candidates at layer 1
type: RSZ, layer: 1, pos: 198
type: RSZ, layer: 1, pos: 52
type: RSZ, layer: 1, pos: 199

Time for candidate selection: 0.00 seconds

### Candidate
type: RSZ, layer: 1, pos: 198

### Relational analysis RSZ of RS_RSZ1_RSZ1_RSZ1_RSZ2_RSZ1_RSZ1_RSZ1

#### Relational analysis RSZ result of RS_RSZ1_RSZ1_RSZ1_RSZ2_RSZ1_RSZ1_RSZ1
Status: Status.UNKNOWN
Output dim: 1, lower bound: -0.0007703, upper bound: 0.0007721
time: 0.66 seconds

### Relational analysis RSZ of RS_RSZ1_RSZ1_RSZ1_RSZ2_RSZ1_RSZ1_RSZ2

#### Relational analysis RSZ result of RS_RSZ1_RSZ1_RSZ1_RSZ2_RSZ1_RSZ1_RSZ2
Status: Status.UNKNOWN
Output dim: 1, lower bound: -0.0007703, upper bound: 0.0007721
time: 0.68 seconds

## BFS RS instance: RS_RSZ1_RSZ1_RSZ1_RSZ2_RSZ1_RSZ2

### Backsubstitution after applying RS history:
0: 0.0007184, 0.0011172, 0.0007184, 0.0011172, -0.0003988, 0.0003988
1: 0.9934075, 0.9943579, 0.9934075, 0.9943579, -0.0009505, 0.0009505
2: -0.0087946, -0.0055215, -0.0087946, -0.0055215, -0.0029798, 0.0029779
3: 0.0035868, 0.0041619, 0.0035868, 0.0041619, -0.0005751, 0.0005751
4: 0.0027809, 0.0053678, 0.0027809, 0.0053678, -0.0025869, 0.0025869
5: 0.0050646, 0.0064029, 0.0050646, 0.0064029, -0.0013382, 0.0013382
6: -0.0021738, -0.0008562, -0.0021738, -0.0008562, -0.0013176, 0.0013176
7: -0.0082275, -0.0074685, -0.0082275, -0.0074685, -0.0007590, 0.0007590
8: 0.0055062, 0.0098066, 0.0055062, 0.0098066, -0.0042206, 0.0042212
9: -0.0036825, -0.0031002, -0.0036825, -0.0031002, -0.0005823, 0.0005823

### Unstable ReLU Count (Linear/Conv2D Layers)
- layer_idx=0, type=LayerType.Linear, total=256, inp1_unstable=9, inp2_unstable=9, delta_unstable=250
- layer_idx=2, type=LayerType.Linear, total=256, inp1_unstable=6, inp2_unstable=6, delta_unstable=256
- layer_idx=4, type=LayerType.Linear, total=256, inp1_unstable=0, inp2_unstable=0, delta_unstable=256
- layer_idx=6, type=LayerType.Linear, total=256, inp1_unstable=2, inp2_unstable=2, delta_unstable=256
- layer_idx=8, type=LayerType.Linear, total=10, inp1_unstable=0, inp2_unstable=0, delta_unstable=10

Time for backsubstitution: 1.72 seconds

### RS candidates at layer 1
type: RSZ, layer: 1, pos: 52
type: RSZ, layer: 1, pos: 198
type: RSZ, layer: 1, pos: 199

Time for candidate selection: 0.00 seconds

### Candidate
type: RSZ, layer: 1, pos: 52

### Relational analysis RSZ of RS_RSZ1_RSZ1_RSZ1_RSZ2_RSZ1_RSZ2_RSZ1

#### Relational analysis RSZ result of RS_RSZ1_RSZ1_RSZ1_RSZ2_RSZ1_RSZ2_RSZ1
Status: Status.UNKNOWN
Output dim: 1, lower bound: -0.0007912, upper bound: 0.0007873
time: 0.63 seconds

### Relational analysis RSZ of RS_RSZ1_RSZ1_RSZ1_RSZ2_RSZ1_RSZ2_RSZ2

#### Relational analysis RSZ result of RS_RSZ1_RSZ1_RSZ1_RSZ2_RSZ1_RSZ2_RSZ2
Status: Status.UNKNOWN
Output dim: 1, lower bound: -0.0007889, upper bound: 0.0007897
time: 0.64 seconds

## BFS RS instance: RS_RSZ1_RSZ1_RSZ1_RSZ2_RSZ2_RSZ1

### Backsubstitution after applying RS history:
0: 0.0007184, 0.0011172, 0.0007184, 0.0011172, -0.0003988, 0.0003988
1: 0.9934075, 0.9943579, 0.9934075, 0.9943579, -0.0009505, 0.0009505
2: -0.0087946, -0.0055215, -0.0087946, -0.0055215, -0.0029781, 0.0029761
3: 0.0035868, 0.0041619, 0.0035868, 0.0041619, -0.0005751, 0.0005751
4: 0.0027809, 0.0053678, 0.0027809, 0.0053678, -0.0025869, 0.0025869
5: 0.0050646, 0.0064029, 0.0050646, 0.0064029, -0.0013382, 0.0013382
6: -0.0021738, -0.0008562, -0.0021738, -0.0008562, -0.0013176, 0.0013176
7: -0.0082275, -0.0074685, -0.0082275, -0.0074685, -0.0007590, 0.0007590
8: 0.0055062, 0.0098066, 0.0055062, 0.0098066, -0.0042208, 0.0042214
9: -0.0036825, -0.0031002, -0.0036825, -0.0031002, -0.0005823, 0.0005823

### Unstable ReLU Count (Linear/Conv2D Layers)
- layer_idx=0, type=LayerType.Linear, total=256, inp1_unstable=9, inp2_unstable=9, delta_unstable=250
- layer_idx=2, type=LayerType.Linear, total=256, inp1_unstable=6, inp2_unstable=6, delta_unstable=256
- layer_idx=4, type=LayerType.Linear, total=256, inp1_unstable=0, inp2_unstable=0, delta_unstable=256
- layer_idx=6, type=LayerType.Linear, total=256, inp1_unstable=2, inp2_unstable=2, delta_unstable=256
- layer_idx=8, type=LayerType.Linear, total=10, inp1_unstable=0, inp2_unstable=0, delta_unstable=10

Time for backsubstitution: 1.71 seconds

### RS candidates at layer 1
type: RSZ, layer: 1, pos: 52
type: RSZ, layer: 1, pos: 68
type: RSZ, layer: 1, pos: 199

Time for candidate selection: 0.00 seconds

### Candidate
type: RSZ, layer: 1, pos: 52

### Relational analysis RSZ of RS_RSZ1_RSZ1_RSZ1_RSZ2_RSZ2_RSZ1_RSZ1

#### Relational analysis RSZ result of RS_RSZ1_RSZ1_RSZ1_RSZ2_RSZ2_RSZ1_RSZ1
Status: Status.UNKNOWN
Output dim: 1, lower bound: -0.0007721, upper bound: 0.0007720
time: 0.68 seconds

### Relational analysis RSZ of RS_RSZ1_RSZ1_RSZ1_RSZ2_RSZ2_RSZ1_RSZ2

#### Relational analysis RSZ result of RS_RSZ1_RSZ1_RSZ1_RSZ2_RSZ2_RSZ1_RSZ2
Status: Status.UNKNOWN
Output dim: 1, lower bound: -0.0007720, upper bound: 0.0007722
time: 0.80 seconds

## BFS RS instance: RS_RSZ1_RSZ1_RSZ1_RSZ2_RSZ2_RSZ2

### Backsubstitution after applying RS history:
0: 0.0007184, 0.0011172, 0.0007184, 0.0011172, -0.0003988, 0.0003988
1: 0.9934075, 0.9943579, 0.9934075, 0.9943579, -0.0009505, 0.0009505
2: -0.0087946, -0.0055215, -0.0087946, -0.0055215, -0.0029886, 0.0029751
3: 0.0035868, 0.0041619, 0.0035868, 0.0041619, -0.0005751, 0.0005751
4: 0.0027809, 0.0053678, 0.0027809, 0.0053678, -0.0025869, 0.0025869
5: 0.0050646, 0.0064029, 0.0050646, 0.0064029, -0.0013382, 0.0013382
6: -0.0021738, -0.0008562, -0.0021738, -0.0008562, -0.0013176, 0.0013176
7: -0.0082275, -0.0074685, -0.0082275, -0.0074685, -0.0007590, 0.0007590
8: 0.0055062, 0.0098066, 0.0055062, 0.0098066, -0.0042206, 0.0042239
9: -0.0036825, -0.0031002, -0.0036825, -0.0031002, -0.0005823, 0.0005823

### Unstable ReLU Count (Linear/Conv2D Layers)
- layer_idx=0, type=LayerType.Linear, total=256, inp1_unstable=9, inp2_unstable=9, delta_unstable=250
- layer_idx=2, type=LayerType.Linear, total=256, inp1_unstable=6, inp2_unstable=6, delta_unstable=256
- layer_idx=4, type=LayerType.Linear, total=256, inp1_unstable=0, inp2_unstable=0, delta_unstable=256
- layer_idx=6, type=LayerType.Linear, total=256, inp1_unstable=2, inp2_unstable=2, delta_unstable=256
- layer_idx=8, type=LayerType.Linear, total=10, inp1_unstable=0, inp2_unstable=0, delta_unstable=10

Time for backsubstitution: 1.63 seconds

### RS candidates at layer 1
type: RSZ, layer: 1, pos: 199
type: RSZ, layer: 1, pos: 52
type: RSZ, layer: 1, pos: 68

Time for candidate selection: 0.00 seconds

### Candidate
type: RSZ, layer: 1, pos: 199

### Relational analysis RSZ of RS_RSZ1_RSZ1_RSZ1_RSZ2_RSZ2_RSZ2_RSZ1

#### Relational analysis RSZ result of RS_RSZ1_RSZ1_RSZ1_RSZ2_RSZ2_RSZ2_RSZ1
Status: Status.UNKNOWN
Output dim: 1, lower bound: -0.0007098, upper bound: 0.0007098
time: 0.63 seconds

### Relational analysis RSZ of RS_RSZ1_RSZ1_RSZ1_RSZ2_RSZ2_RSZ2_RSZ2

#### Relational analysis RSZ result of RS_RSZ1_RSZ1_RSZ1_RSZ2_RSZ2_RSZ2_RSZ2
Status: Status.UNKNOWN
Output dim: 1, lower bound: -0.0007098, upper bound: 0.0007098
time: 0.64 seconds

## BFS RS instance: RS_RSZ1_RSZ1_RSZ2_RSZ1_RSZ1_RSZ1

### Backsubstitution after applying RS history:
0: 0.0007184, 0.0011172, 0.0007184, 0.0011172, -0.0003988, 0.0003988
1: 0.9934075, 0.9943579, 0.9934075, 0.9943579, -0.0009505, 0.0009505
2: -0.0087946, -0.0055215, -0.0087946, -0.0055215, -0.0029785, 0.0029801
3: 0.0035868, 0.0041619, 0.0035868, 0.0041619, -0.0005751, 0.0005751
4: 0.0027809, 0.0053678, 0.0027809, 0.0053678, -0.0025869, 0.0025869
5: 0.0050646, 0.0064029, 0.0050646, 0.0064029, -0.0013382, 0.0013382
6: -0.0021738, -0.0008562, -0.0021738, -0.0008562, -0.0013176, 0.0013176
7: -0.0082275, -0.0074685, -0.0082275, -0.0074685, -0.0007590, 0.0007590
8: 0.0055062, 0.0098066, 0.0055062, 0.0098066, -0.0042214, 0.0042210
9: -0.0036825, -0.0031002, -0.0036825, -0.0031002, -0.0005823, 0.0005823

### Unstable ReLU Count (Linear/Conv2D Layers)
- layer_idx=0, type=LayerType.Linear, total=256, inp1_unstable=9, inp2_unstable=9, delta_unstable=250
- layer_idx=2, type=LayerType.Linear, total=256, inp1_unstable=6, inp2_unstable=6, delta_unstable=256
- layer_idx=4, type=LayerType.Linear, total=256, inp1_unstable=0, inp2_unstable=0, delta_unstable=256
- layer_idx=6, type=LayerType.Linear, total=256, inp1_unstable=2, inp2_unstable=2, delta_unstable=256
- layer_idx=8, type=LayerType.Linear, total=10, inp1_unstable=0, inp2_unstable=0, delta_unstable=10

Time for backsubstitution: 1.79 seconds

### RS candidates at layer 1
type: RSZ, layer: 1, pos: 68
type: RSZ, layer: 1, pos: 199
type: RSZ, layer: 1, pos: 136

Time for candidate selection: 0.00 seconds

### Candidate
type: RSZ, layer: 1, pos: 68

### Relational analysis RSZ of RS_RSZ1_RSZ1_RSZ2_RSZ1_RSZ1_RSZ1_RSZ1

#### Relational analysis RSZ result of RS_RSZ1_RSZ1_RSZ2_RSZ1_RSZ1_RSZ1_RSZ1
Status: Status.UNKNOWN
Output dim: 1, lower bound: -0.0007734, upper bound: 0.0007737
time: 0.64 seconds

### Relational analysis RSZ of RS_RSZ1_RSZ1_RSZ2_RSZ1_RSZ1_RSZ1_RSZ2

#### Relational analysis RSZ result of RS_RSZ1_RSZ1_RSZ2_RSZ1_RSZ1_RSZ1_RSZ2
Status: Status.UNKNOWN
Output dim: 1, lower bound: -0.0007738, upper bound: 0.0007730
time: 0.67 seconds

## BFS RS instance: RS_RSZ1_RSZ1_RSZ2_RSZ1_RSZ1_RSZ2

### Backsubstitution after applying RS history:
0: 0.0007184, 0.0011172, 0.0007184, 0.0011172, -0.0003988, 0.0003988
1: 0.9934075, 0.9943579, 0.9934075, 0.9943579, -0.0009505, 0.0009505
2: -0.0087946, -0.0055215, -0.0087946, -0.0055215, -0.0029782, 0.0029803
3: 0.0035868, 0.0041619, 0.0035868, 0.0041619, -0.0005751, 0.0005751
4: 0.0027809, 0.0053678, 0.0027809, 0.0053678, -0.0025869, 0.0025869
5: 0.0050646, 0.0064029, 0.0050646, 0.0064029, -0.0013382, 0.0013382
6: -0.0021738, -0.0008562, -0.0021738, -0.0008562, -0.0013176, 0.0013176
7: -0.0082275, -0.0074685, -0.0082275, -0.0074685, -0.0007590, 0.0007590
8: 0.0055062, 0.0098066, 0.0055062, 0.0098066, -0.0042215, 0.0042209
9: -0.0036825, -0.0031002, -0.0036825, -0.0031002, -0.0005823, 0.0005823

### Unstable ReLU Count (Linear/Conv2D Layers)
- layer_idx=0, type=LayerType.Linear, total=256, inp1_unstable=9, inp2_unstable=9, delta_unstable=250
- layer_idx=2, type=LayerType.Linear, total=256, inp1_unstable=6, inp2_unstable=6, delta_unstable=256
- layer_idx=4, type=LayerType.Linear, total=256, inp1_unstable=0, inp2_unstable=0, delta_unstable=256
- layer_idx=6, type=LayerType.Linear, total=256, inp1_unstable=2, inp2_unstable=2, delta_unstable=256
- layer_idx=8, type=LayerType.Linear, total=10, inp1_unstable=0, inp2_unstable=0, delta_unstable=10

Time for backsubstitution: 1.70 seconds

### RS candidates at layer 1
type: RSZ, layer: 1, pos: 68
type: RSZ, layer: 1, pos: 136
type: RSZ, layer: 1, pos: 199

Time for candidate selection: 0.00 seconds

### Candidate
type: RSZ, layer: 1, pos: 68

### Relational analysis RSZ of RS_RSZ1_RSZ1_RSZ2_RSZ1_RSZ1_RSZ2_RSZ1

#### Relational analysis RSZ result of RS_RSZ1_RSZ1_RSZ2_RSZ1_RSZ1_RSZ2_RSZ1
Status: Status.UNKNOWN
Output dim: 1, lower bound: -0.0007734, upper bound: 0.0007737
time: 0.64 seconds

### Relational analysis RSZ of RS_RSZ1_RSZ1_RSZ2_RSZ1_RSZ1_RSZ2_RSZ2

#### Relational analysis RSZ result of RS_RSZ1_RSZ1_RSZ2_RSZ1_RSZ1_RSZ2_RSZ2
Status: Status.UNKNOWN
Output dim: 1, lower bound: -0.0007737, upper bound: 0.0007731
time: 0.70 seconds

## BFS RS instance: RS_RSZ1_RSZ1_RSZ2_RSZ1_RSZ2_RSZ1

### Backsubstitution after applying RS history:
0: 0.0007184, 0.0011172, 0.0007184, 0.0011172, -0.0003988, 0.0003988
1: 0.9934075, 0.9943579, 0.9934075, 0.9943579, -0.0009505, 0.0009505
2: -0.0087946, -0.0055215, -0.0087946, -0.0055215, -0.0029796, 0.0029802
3: 0.0035868, 0.0041619, 0.0035868, 0.0041619, -0.0005751, 0.0005751
4: 0.0027809, 0.0053678, 0.0027809, 0.0053678, -0.0025869, 0.0025869
5: 0.0050646, 0.0064029, 0.0050646, 0.0064029, -0.0013382, 0.0013382
6: -0.0021738, -0.0008562, -0.0021738, -0.0008562, -0.0013176, 0.0013176
7: -0.0082275, -0.0074685, -0.0082275, -0.0074685, -0.0007590, 0.0007590
8: 0.0055062, 0.0098066, 0.0055062, 0.0098066, -0.0042214, 0.0042212
9: -0.0036825, -0.0031002, -0.0036825, -0.0031002, -0.0005823, 0.0005823

### Unstable ReLU Count (Linear/Conv2D Layers)
- layer_idx=0, type=LayerType.Linear, total=256, inp1_unstable=9, inp2_unstable=9, delta_unstable=250
- layer_idx=2, type=LayerType.Linear, total=256, inp1_unstable=6, inp2_unstable=6, delta_unstable=256
- layer_idx=4, type=LayerType.Linear, total=256, inp1_unstable=0, inp2_unstable=0, delta_unstable=256
- layer_idx=6, type=LayerType.Linear, total=256, inp1_unstable=2, inp2_unstable=2, delta_unstable=256
- layer_idx=8, type=LayerType.Linear, total=10, inp1_unstable=0, inp2_unstable=0, delta_unstable=10

Time for backsubstitution: 1.60 seconds

### RS candidates at layer 1
type: RSZ, layer: 1, pos: 83
type: RSZ, layer: 1, pos: 136
type: RSZ, layer: 1, pos: 68

Time for candidate selection: 0.00 seconds

### Candidate
type: RSZ, layer: 1, pos: 83

### Relational analysis RSZ of RS_RSZ1_RSZ1_RSZ2_RSZ1_RSZ2_RSZ1_RSZ1

#### Relational analysis RSZ result of RS_RSZ1_RSZ1_RSZ2_RSZ1_RSZ2_RSZ1_RSZ1
Status: Status.UNKNOWN
Output dim: 1, lower bound: -0.0007098, upper bound: 0.0007098
time: 0.59 seconds

### Relational analysis RSZ of RS_RSZ1_RSZ1_RSZ2_RSZ1_RSZ2_RSZ1_RSZ2

#### Relational analysis RSZ result of RS_RSZ1_RSZ1_RSZ2_RSZ1_RSZ2_RSZ1_RSZ2
Status: Status.UNKNOWN
Output dim: 1, lower bound: -0.0007098, upper bound: 0.0007098
time: 0.64 seconds

## BFS RS instance: RS_RSZ1_RSZ1_RSZ2_RSZ1_RSZ2_RSZ2

### Backsubstitution after applying RS history:
0: 0.0007184, 0.0011172, 0.0007184, 0.0011172, -0.0003988, 0.0003988
1: 0.9934075, 0.9943579, 0.9934075, 0.9943579, -0.0009505, 0.0009505
2: -0.0087946, -0.0055215, -0.0087946, -0.0055215, -0.0029804, 0.0029823
3: 0.0035868, 0.0041619, 0.0035868, 0.0041619, -0.0005751, 0.0005751
4: 0.0027809, 0.0053678, 0.0027809, 0.0053678, -0.0025869, 0.0025869
5: 0.0050646, 0.0064029, 0.0050646, 0.0064029, -0.0013382, 0.0013382
6: -0.0021738, -0.0008562, -0.0021738, -0.0008562, -0.0013176, 0.0013176
7: -0.0082275, -0.0074685, -0.0082275, -0.0074685, -0.0007590, 0.0007590
8: 0.0055062, 0.0098066, 0.0055062, 0.0098066, -0.0042219, 0.0042214
9: -0.0036825, -0.0031002, -0.0036825, -0.0031002, -0.0005823, 0.0005823

### Unstable ReLU Count (Linear/Conv2D Layers)
- layer_idx=0, type=LayerType.Linear, total=256, inp1_unstable=9, inp2_unstable=9, delta_unstable=250
- layer_idx=2, type=LayerType.Linear, total=256, inp1_unstable=6, inp2_unstable=6, delta_unstable=256
- layer_idx=4, type=LayerType.Linear, total=256, inp1_unstable=0, inp2_unstable=0, delta_unstable=256
- layer_idx=6, type=LayerType.Linear, total=256, inp1_unstable=2, inp2_unstable=2, delta_unstable=256
- layer_idx=8, type=LayerType.Linear, total=10, inp1_unstable=0, inp2_unstable=0, delta_unstable=10

Time for backsubstitution: 1.70 seconds

### RS candidates at layer 1
type: RSZ, layer: 1, pos: 68
type: RSZ, layer: 1, pos: 83
type: RSZ, layer: 1, pos: 136

Time for candidate selection: 0.00 seconds

### Candidate
type: RSZ, layer: 1, pos: 68

### Relational analysis RSZ of RS_RSZ1_RSZ1_RSZ2_RSZ1_RSZ2_RSZ2_RSZ1

#### Relational analysis RSZ result of RS_RSZ1_RSZ1_RSZ2_RSZ1_RSZ2_RSZ2_RSZ1
Status: Status.UNKNOWN
Output dim: 1, lower bound: -0.0007131, upper bound: 0.0007131
time: 0.59 seconds

### Relational analysis RSZ of RS_RSZ1_RSZ1_RSZ2_RSZ1_RSZ2_RSZ2_RSZ2

#### Relational analysis RSZ result of RS_RSZ1_RSZ1_RSZ2_RSZ1_RSZ2_RSZ2_RSZ2
Status: Status.UNKNOWN
Output dim: 1, lower bound: -0.0007131, upper bound: 0.0007131
time: 0.64 seconds

## BFS RS instance: RS_RSZ1_RSZ1_RSZ2_RSZ2_RSZ1_RSZ1

### Backsubstitution after applying RS history:
0: 0.0007184, 0.0011172, 0.0007184, 0.0011172, -0.0003988, 0.0003988
1: 0.9934075, 0.9943579, 0.9934075, 0.9943579, -0.0009505, 0.0009505
2: -0.0087946, -0.0055215, -0.0087946, -0.0055215, -0.0029814, 0.0029716
3: 0.0035868, 0.0041619, 0.0035868, 0.0041619, -0.0005751, 0.0005751
4: 0.0027809, 0.0053678, 0.0027809, 0.0053678, -0.0025869, 0.0025869
5: 0.0050646, 0.0064029, 0.0050646, 0.0064029, -0.0013382, 0.0013382
6: -0.0021738, -0.0008562, -0.0021738, -0.0008562, -0.0013176, 0.0013176
7: -0.0082275, -0.0074685, -0.0082275, -0.0074685, -0.0007590, 0.0007590
8: 0.0055062, 0.0098066, 0.0055062, 0.0098066, -0.0042191, 0.0042215
9: -0.0036825, -0.0031002, -0.0036825, -0.0031002, -0.0005823, 0.0005823

### Unstable ReLU Count (Linear/Conv2D Layers)
- layer_idx=0, type=LayerType.Linear, total=256, inp1_unstable=9, inp2_unstable=9, delta_unstable=250
- layer_idx=2, type=LayerType.Linear, total=256, inp1_unstable=6, inp2_unstable=6, delta_unstable=256
- layer_idx=4, type=LayerType.Linear, total=256, inp1_unstable=0, inp2_unstable=0, delta_unstable=256
- layer_idx=6, type=LayerType.Linear, total=256, inp1_unstable=2, inp2_unstable=2, delta_unstable=256
- layer_idx=8, type=LayerType.Linear, total=10, inp1_unstable=0, inp2_unstable=0, delta_unstable=10

Time for backsubstitution: 1.67 seconds

### RS candidates at layer 1
type: RSZ, layer: 1, pos: 199
type: RSZ, layer: 1, pos: 52
type: RSZ, layer: 1, pos: 83

Time for candidate selection: 0.00 seconds

### Candidate
type: RSZ, layer: 1, pos: 199

### Relational analysis RSZ of RS_RSZ1_RSZ1_RSZ2_RSZ2_RSZ1_RSZ1_RSZ1

#### Relational analysis RSZ result of RS_RSZ1_RSZ1_RSZ2_RSZ2_RSZ1_RSZ1_RSZ1
Status: Status.UNKNOWN
Output dim: 1, lower bound: -0.0007131, upper bound: 0.0007131
time: 0.60 seconds

### Relational analysis RSZ of RS_RSZ1_RSZ1_RSZ2_RSZ2_RSZ1_RSZ1_RSZ2

#### Relational analysis RSZ result of RS_RSZ1_RSZ1_RSZ2_RSZ2_RSZ1_RSZ1_RSZ2
Status: Status.UNKNOWN
Output dim: 1, lower bound: -0.0007131, upper bound: 0.0007131
time: 0.59 seconds

## BFS RS instance: RS_RSZ1_RSZ1_RSZ2_RSZ2_RSZ1_RSZ2

### Backsubstitution after applying RS history:
0: 0.0007184, 0.0011172, 0.0007184, 0.0011172, -0.0003988, 0.0003988
1: 0.9934075, 0.9943579, 0.9934075, 0.9943579, -0.0009505, 0.0009505
2: -0.0087946, -0.0055215, -0.0087946, -0.0055215, -0.0029836, 0.0029693
3: 0.0035868, 0.0041619, 0.0035868, 0.0041619, -0.0005751, 0.0005751
4: 0.0027809, 0.0053678, 0.0027809, 0.0053678, -0.0025869, 0.0025869
5: 0.0050646, 0.0064029, 0.0050646, 0.0064029, -0.0013382, 0.0013382
6: -0.0021738, -0.0008562, -0.0021738, -0.0008562, -0.0013176, 0.0013176
7: -0.0082275, -0.0074685, -0.0082275, -0.0074685, -0.0007590, 0.0007590
8: 0.0055062, 0.0098066, 0.0055062, 0.0098066, -0.0042185, 0.0042221
9: -0.0036825, -0.0031002, -0.0036825, -0.0031002, -0.0005823, 0.0005823

### Unstable ReLU Count (Linear/Conv2D Layers)
- layer_idx=0, type=LayerType.Linear, total=256, inp1_unstable=9, inp2_unstable=9, delta_unstable=250
- layer_idx=2, type=LayerType.Linear, total=256, inp1_unstable=6, inp2_unstable=6, delta_unstable=256
- layer_idx=4, type=LayerType.Linear, total=256, inp1_unstable=0, inp2_unstable=0, delta_unstable=256
- layer_idx=6, type=LayerType.Linear, total=256, inp1_unstable=2, inp2_unstable=2, delta_unstable=256
- layer_idx=8, type=LayerType.Linear, total=10, inp1_unstable=0, inp2_unstable=0, delta_unstable=10

Time for backsubstitution: 1.59 seconds

### RS candidates at layer 1
type: RSZ, layer: 1, pos: 83
type: RSZ, layer: 1, pos: 52
type: RSZ, layer: 1, pos: 199

Time for candidate selection: 0.00 seconds

### Candidate
type: RSZ, layer: 1, pos: 83

### Relational analysis RSZ of RS_RSZ1_RSZ1_RSZ2_RSZ2_RSZ1_RSZ2_RSZ1

#### Relational analysis RSZ result of RS_RSZ1_RSZ1_RSZ2_RSZ2_RSZ1_RSZ2_RSZ1
Status: Status.UNKNOWN
Output dim: 1, lower bound: -0.0007711, upper bound: 0.0007718
time: 0.63 seconds

### Relational analysis RSZ of RS_RSZ1_RSZ1_RSZ2_RSZ2_RSZ1_RSZ2_RSZ2

#### Relational analysis RSZ result of RS_RSZ1_RSZ1_RSZ2_RSZ2_RSZ1_RSZ2_RSZ2
Status: Status.UNKNOWN
Output dim: 1, lower bound: -0.0007707, upper bound: 0.0007718
time: 0.66 seconds

## BFS RS instance: RS_RSZ1_RSZ1_RSZ2_RSZ2_RSZ2_RSZ1

### Backsubstitution after applying RS history:
0: 0.0007184, 0.0011172, 0.0007184, 0.0011172, -0.0003988, 0.0003988
1: 0.9934075, 0.9943579, 0.9934075, 0.9943579, -0.0009505, 0.0009505
2: -0.0087946, -0.0055215, -0.0087946, -0.0055215, -0.0029828, 0.0029753
3: 0.0035868, 0.0041619, 0.0035868, 0.0041619, -0.0005751, 0.0005751
4: 0.0027809, 0.0053678, 0.0027809, 0.0053678, -0.0025869, 0.0025869
5: 0.0050646, 0.0064029, 0.0050646, 0.0064029, -0.0013382, 0.0013382
6: -0.0021738, -0.0008562, -0.0021738, -0.0008562, -0.0013176, 0.0013176
7: -0.0082275, -0.0074685, -0.0082275, -0.0074685, -0.0007590, 0.0007590
8: 0.0055062, 0.0098066, 0.0055062, 0.0098066, -0.0042196, 0.0042213
9: -0.0036825, -0.0031002, -0.0036825, -0.0031002, -0.0005823, 0.0005823

### Unstable ReLU Count (Linear/Conv2D Layers)
- layer_idx=0, type=LayerType.Linear, total=256, inp1_unstable=9, inp2_unstable=9, delta_unstable=250
- layer_idx=2, type=LayerType.Linear, total=256, inp1_unstable=6, inp2_unstable=6, delta_unstable=256
- layer_idx=4, type=LayerType.Linear, total=256, inp1_unstable=0, inp2_unstable=0, delta_unstable=256
- layer_idx=6, type=LayerType.Linear, total=256, inp1_unstable=2, inp2_unstable=2, delta_unstable=256
- layer_idx=8, type=LayerType.Linear, total=10, inp1_unstable=0, inp2_unstable=0, delta_unstable=10

Time for backsubstitution: 1.66 seconds

### RS candidates at layer 1
type: RSZ, layer: 1, pos: 83
type: RSZ, layer: 1, pos: 199
type: RSZ, layer: 1, pos: 136

Time for candidate selection: 0.00 seconds

### Candidate
type: RSZ, layer: 1, pos: 83

### Relational analysis RSZ of RS_RSZ1_RSZ1_RSZ2_RSZ2_RSZ2_RSZ1_RSZ1

#### Relational analysis RSZ result of RS_RSZ1_RSZ1_RSZ2_RSZ2_RSZ2_RSZ1_RSZ1
Status: Status.UNKNOWN
Output dim: 1, lower bound: -0.0007738, upper bound: 0.0007730
time: 0.65 seconds

### Relational analysis RSZ of RS_RSZ1_RSZ1_RSZ2_RSZ2_RSZ2_RSZ1_RSZ2

#### Relational analysis RSZ result of RS_RSZ1_RSZ1_RSZ2_RSZ2_RSZ2_RSZ1_RSZ2
Status: Status.UNKNOWN
Output dim: 1, lower bound: -0.0007737, upper bound: 0.0007731
time: 0.69 seconds

## BFS RS instance: RS_RSZ1_RSZ1_RSZ2_RSZ2_RSZ2_RSZ2

### Backsubstitution after applying RS history:
0: 0.0007184, 0.0011172, 0.0007184, 0.0011172, -0.0003988, 0.0003988
1: 0.9934075, 0.9943579, 0.9934075, 0.9943579, -0.0009505, 0.0009505
2: -0.0087946, -0.0055215, -0.0087946, -0.0055215, -0.0029838, 0.0029743
3: 0.0035868, 0.0041619, 0.0035868, 0.0041619, -0.0005751, 0.0005751
4: 0.0027809, 0.0053678, 0.0027809, 0.0053678, -0.0025869, 0.0025869
5: 0.0050646, 0.0064029, 0.0050646, 0.0064029, -0.0013382, 0.0013382
6: -0.0021738, -0.0008562, -0.0021738, -0.0008562, -0.0013176, 0.0013176
7: -0.0082275, -0.0074685, -0.0082275, -0.0074685, -0.0007590, 0.0007590
8: 0.0055062, 0.0098066, 0.0055062, 0.0098066, -0.0042194, 0.0042216
9: -0.0036825, -0.0031002, -0.0036825, -0.0031002, -0.0005823, 0.0005823

### Unstable ReLU Count (Linear/Conv2D Layers)
- layer_idx=0, type=LayerType.Linear, total=256, inp1_unstable=9, inp2_unstable=9, delta_unstable=250
- layer_idx=2, type=LayerType.Linear, total=256, inp1_unstable=6, inp2_unstable=6, delta_unstable=256
- layer_idx=4, type=LayerType.Linear, total=256, inp1_unstable=0, inp2_unstable=0, delta_unstable=256
- layer_idx=6, type=LayerType.Linear, total=256, inp1_unstable=2, inp2_unstable=2, delta_unstable=256
- layer_idx=8, type=LayerType.Linear, total=10, inp1_unstable=0, inp2_unstable=0, delta_unstable=10

Time for backsubstitution: 1.61 seconds

### RS candidates at layer 1
type: RSZ, layer: 1, pos: 199
type: RSZ, layer: 1, pos: 83
type: RSZ, layer: 1, pos: 136

Time for candidate selection: 0.00 seconds

### Candidate
type: RSZ, layer: 1, pos: 199

### Relational analysis RSZ of RS_RSZ1_RSZ1_RSZ2_RSZ2_RSZ2_RSZ2_RSZ1

#### Relational analysis RSZ result of RS_RSZ1_RSZ1_RSZ2_RSZ2_RSZ2_RSZ2_RSZ1
Status: Status.UNKNOWN
Output dim: 1, lower bound: -0.0007131, upper bound: 0.0007131
time: 0.60 seconds

### Relational analysis RSZ of RS_RSZ1_RSZ1_RSZ2_RSZ2_RSZ2_RSZ2_RSZ2

#### Relational analysis RSZ result of RS_RSZ1_RSZ1_RSZ2_RSZ2_RSZ2_RSZ2_RSZ2
Status: Status.UNKNOWN
Output dim: 1, lower bound: -0.0007131, upper bound: 0.0007131
time: 0.69 seconds

## BFS RS instance: RS_RSZ1_RSZ2_RSZ1_RSZ1_RSZ1_RSZ1

### Backsubstitution after applying RS history:
0: 0.0007184, 0.0011172, 0.0007184, 0.0011172, -0.0003988, 0.0003988
1: 0.9934075, 0.9943579, 0.9934075, 0.9943579, -0.0009505, 0.0009505
2: -0.0087946, -0.0055215, -0.0087946, -0.0055215, -0.0029776, 0.0029817
3: 0.0035868, 0.0041619, 0.0035868, 0.0041619, -0.0005751, 0.0005751
4: 0.0027809, 0.0053678, 0.0027809, 0.0053678, -0.0025869, 0.0025869
5: 0.0050646, 0.0064029, 0.0050646, 0.0064029, -0.0013382, 0.0013382
6: -0.0021738, -0.0008562, -0.0021738, -0.0008562, -0.0013176, 0.0013176
7: -0.0082275, -0.0074685, -0.0082275, -0.0074685, -0.0007590, 0.0007590
8: 0.0055062, 0.0098066, 0.0055062, 0.0098066, -0.0042222, 0.0042211
9: -0.0036825, -0.0031002, -0.0036825, -0.0031002, -0.0005823, 0.0005823

### Unstable ReLU Count (Linear/Conv2D Layers)
- layer_idx=0, type=LayerType.Linear, total=256, inp1_unstable=9, inp2_unstable=9, delta_unstable=250
- layer_idx=2, type=LayerType.Linear, total=256, inp1_unstable=6, inp2_unstable=6, delta_unstable=256
- layer_idx=4, type=LayerType.Linear, total=256, inp1_unstable=0, inp2_unstable=0, delta_unstable=256
- layer_idx=6, type=LayerType.Linear, total=256, inp1_unstable=2, inp2_unstable=2, delta_unstable=256
- layer_idx=8, type=LayerType.Linear, total=10, inp1_unstable=0, inp2_unstable=0, delta_unstable=10

Time for backsubstitution: 1.72 seconds

### RS candidates at layer 1
type: RSZ, layer: 1, pos: 68
type: RSZ, layer: 1, pos: 204
type: RSZ, layer: 1, pos: 199

Time for candidate selection: 0.00 seconds

### Candidate
type: RSZ, layer: 1, pos: 68

### Relational analysis RSZ of RS_RSZ1_RSZ2_RSZ1_RSZ1_RSZ1_RSZ1_RSZ1

#### Relational analysis RSZ result of RS_RSZ1_RSZ2_RSZ1_RSZ1_RSZ1_RSZ1_RSZ1
Status: Status.UNKNOWN
Output dim: 1, lower bound: -0.0007740, upper bound: 0.0007753
time: 0.64 seconds

### Relational analysis RSZ of RS_RSZ1_RSZ2_RSZ1_RSZ1_RSZ1_RSZ1_RSZ2

#### Relational analysis RSZ result of RS_RSZ1_RSZ2_RSZ1_RSZ1_RSZ1_RSZ1_RSZ2
Status: Status.UNKNOWN
Output dim: 1, lower bound: -0.0007755, upper bound: 0.0007734
time: 0.70 seconds

## BFS RS instance: RS_RSZ1_RSZ2_RSZ1_RSZ1_RSZ1_RSZ2

### Backsubstitution after applying RS history:
0: 0.0007184, 0.0011172, 0.0007184, 0.0011172, -0.0003988, 0.0003988
1: 0.9934075, 0.9943579, 0.9934075, 0.9943579, -0.0009505, 0.0009505
2: -0.0087946, -0.0055215, -0.0087946, -0.0055215, -0.0029773, 0.0029820
3: 0.0035868, 0.0041619, 0.0035868, 0.0041619, -0.0005751, 0.0005751
4: 0.0027809, 0.0053678, 0.0027809, 0.0053678, -0.0025869, 0.0025869
5: 0.0050646, 0.0064029, 0.0050646, 0.0064029, -0.0013382, 0.0013382
6: -0.0021738, -0.0008562, -0.0021738, -0.0008562, -0.0013176, 0.0013176
7: -0.0082275, -0.0074685, -0.0082275, -0.0074685, -0.0007590, 0.0007590
8: 0.0055062, 0.0098066, 0.0055062, 0.0098066, -0.0042223, 0.0042210
9: -0.0036825, -0.0031002, -0.0036825, -0.0031002, -0.0005823, 0.0005823

### Unstable ReLU Count (Linear/Conv2D Layers)
- layer_idx=0, type=LayerType.Linear, total=256, inp1_unstable=9, inp2_unstable=9, delta_unstable=250
- layer_idx=2, type=LayerType.Linear, total=256, inp1_unstable=6, inp2_unstable=6, delta_unstable=256
- layer_idx=4, type=LayerType.Linear, total=256, inp1_unstable=0, inp2_unstable=0, delta_unstable=256
- layer_idx=6, type=LayerType.Linear, total=256, inp1_unstable=2, inp2_unstable=2, delta_unstable=256
- layer_idx=8, type=LayerType.Linear, total=10, inp1_unstable=0, inp2_unstable=0, delta_unstable=10

Time for backsubstitution: 1.64 seconds

### RS candidates at layer 1
type: RSZ, layer: 1, pos: 199
type: RSZ, layer: 1, pos: 68
type: RSZ, layer: 1, pos: 204

Time for candidate selection: 0.00 seconds

### Candidate
type: RSZ, layer: 1, pos: 199

### Relational analysis RSZ of RS_RSZ1_RSZ2_RSZ1_RSZ1_RSZ1_RSZ2_RSZ1

#### Relational analysis RSZ result of RS_RSZ1_RSZ2_RSZ1_RSZ1_RSZ1_RSZ2_RSZ1
Status: Status.UNKNOWN
Output dim: 1, lower bound: -0.0007098, upper bound: 0.0007098
time: 0.65 seconds

### Relational analysis RSZ of RS_RSZ1_RSZ2_RSZ1_RSZ1_RSZ1_RSZ2_RSZ2

#### Relational analysis RSZ result of RS_RSZ1_RSZ2_RSZ1_RSZ1_RSZ1_RSZ2_RSZ2
Status: Status.UNKNOWN
Output dim: 1, lower bound: -0.0007098, upper bound: 0.0007098
time: 0.66 seconds

## BFS RS instance: RS_RSZ1_RSZ2_RSZ1_RSZ1_RSZ2_RSZ1

### Backsubstitution after applying RS history:
0: 0.0007184, 0.0011172, 0.0007184, 0.0011172, -0.0003988, 0.0003988
1: 0.9934075, 0.9943579, 0.9934075, 0.9943579, -0.0009505, 0.0009505
2: -0.0087946, -0.0055215, -0.0087946, -0.0055215, -0.0029762, 0.0029777
3: 0.0035868, 0.0041619, 0.0035868, 0.0041619, -0.0005751, 0.0005751
4: 0.0027809, 0.0053678, 0.0027809, 0.0053678, -0.0025869, 0.0025869
5: 0.0050646, 0.0064029, 0.0050646, 0.0064029, -0.0013382, 0.0013382
6: -0.0021738, -0.0008562, -0.0021738, -0.0008562, -0.0013176, 0.0013176
7: -0.0082275, -0.0074685, -0.0082275, -0.0074685, -0.0007590, 0.0007590
8: 0.0055062, 0.0098066, 0.0055062, 0.0098066, -0.0042211, 0.0042207
9: -0.0036825, -0.0031002, -0.0036825, -0.0031002, -0.0005823, 0.0005823

### Unstable ReLU Count (Linear/Conv2D Layers)
- layer_idx=0, type=LayerType.Linear, total=256, inp1_unstable=9, inp2_unstable=9, delta_unstable=250
- layer_idx=2, type=LayerType.Linear, total=256, inp1_unstable=6, inp2_unstable=6, delta_unstable=256
- layer_idx=4, type=LayerType.Linear, total=256, inp1_unstable=0, inp2_unstable=0, delta_unstable=256
- layer_idx=6, type=LayerType.Linear, total=256, inp1_unstable=2, inp2_unstable=2, delta_unstable=256
- layer_idx=8, type=LayerType.Linear, total=10, inp1_unstable=0, inp2_unstable=0, delta_unstable=10

Time for backsubstitution: 1.84 seconds

### RS candidates at layer 1
type: RSZ, layer: 1, pos: 68
type: RSZ, layer: 1, pos: 199
type: RSZ, layer: 1, pos: 83

Time for candidate selection: 0.00 seconds

### Candidate
type: RSZ, layer: 1, pos: 68

### Relational analysis RSZ of RS_RSZ1_RSZ2_RSZ1_RSZ1_RSZ2_RSZ1_RSZ1

#### Relational analysis RSZ result of RS_RSZ1_RSZ2_RSZ1_RSZ1_RSZ2_RSZ1_RSZ1
Status: Status.UNKNOWN
Output dim: 1, lower bound: -0.0007707, upper bound: 0.0007731
time: 0.83 seconds

### Relational analysis RSZ of RS_RSZ1_RSZ2_RSZ1_RSZ1_RSZ2_RSZ1_RSZ2

#### Relational analysis RSZ result of RS_RSZ1_RSZ2_RSZ1_RSZ1_RSZ2_RSZ1_RSZ2
Status: Status.UNKNOWN
Output dim: 1, lower bound: -0.0007727, upper bound: 0.0007718
time: 0.66 seconds

## BFS RS instance: RS_RSZ1_RSZ2_RSZ1_RSZ1_RSZ2_RSZ2

### Backsubstitution after applying RS history:
0: 0.0007184, 0.0011172, 0.0007184, 0.0011172, -0.0003988, 0.0003988
1: 0.9934075, 0.9943579, 0.9934075, 0.9943579, -0.0009505, 0.0009505
2: -0.0087946, -0.0055215, -0.0087946, -0.0055215, -0.0029751, 0.0029783
3: 0.0035868, 0.0041619, 0.0035868, 0.0041619, -0.0005751, 0.0005751
4: 0.0027809, 0.0053678, 0.0027809, 0.0053678, -0.0025869, 0.0025869
5: 0.0050646, 0.0064029, 0.0050646, 0.0064029, -0.0013382, 0.0013382
6: -0.0021738, -0.0008562, -0.0021738, -0.0008562, -0.0013176, 0.0013176
7: -0.0082275, -0.0074685, -0.0082275, -0.0074685, -0.0007590, 0.0007590
8: 0.0055062, 0.0098066, 0.0055062, 0.0098066, -0.0042213, 0.0042204
9: -0.0036825, -0.0031002, -0.0036825, -0.0031002, -0.0005823, 0.0005823

### Unstable ReLU Count (Linear/Conv2D Layers)
- layer_idx=0, type=LayerType.Linear, total=256, inp1_unstable=9, inp2_unstable=9, delta_unstable=250
- layer_idx=2, type=LayerType.Linear, total=256, inp1_unstable=6, inp2_unstable=6, delta_unstable=256
- layer_idx=4, type=LayerType.Linear, total=256, inp1_unstable=0, inp2_unstable=0, delta_unstable=256
- layer_idx=6, type=LayerType.Linear, total=256, inp1_unstable=2, inp2_unstable=2, delta_unstable=256
- layer_idx=8, type=LayerType.Linear, total=10, inp1_unstable=0, inp2_unstable=0, delta_unstable=10

Time for backsubstitution: 1.71 seconds

### RS candidates at layer 1
type: RSZ, layer: 1, pos: 199
type: RSZ, layer: 1, pos: 83
type: RSZ, layer: 1, pos: 68

Time for candidate selection: 0.00 seconds

### Candidate
type: RSZ, layer: 1, pos: 199

### Relational analysis RSZ of RS_RSZ1_RSZ2_RSZ1_RSZ1_RSZ2_RSZ2_RSZ1

#### Relational analysis RSZ result of RS_RSZ1_RSZ2_RSZ1_RSZ1_RSZ2_RSZ2_RSZ1
Status: Status.UNKNOWN
Output dim: 1, lower bound: -0.0007131, upper bound: 0.0007131
time: 0.60 seconds

### Relational analysis RSZ of RS_RSZ1_RSZ2_RSZ1_RSZ1_RSZ2_RSZ2_RSZ2

#### Relational analysis RSZ result of RS_RSZ1_RSZ2_RSZ1_RSZ1_RSZ2_RSZ2_RSZ2
Status: Status.UNKNOWN
Output dim: 1, lower bound: -0.0007131, upper bound: 0.0007131
time: 0.59 seconds

## BFS RS instance: RS_RSZ1_RSZ2_RSZ1_RSZ2_RSZ1_RSZ1

### Backsubstitution after applying RS history:
0: 0.0007184, 0.0011172, 0.0007184, 0.0011172, -0.0003988, 0.0003988
1: 0.9934075, 0.9943579, 0.9934075, 0.9943579, -0.0009505, 0.0009505
2: -0.0087946, -0.0055215, -0.0087946, -0.0055215, -0.0029814, 0.0029714
3: 0.0035868, 0.0041619, 0.0035868, 0.0041619, -0.0005751, 0.0005751
4: 0.0027809, 0.0053678, 0.0027809, 0.0053678, -0.0025869, 0.0025869
5: 0.0050646, 0.0064029, 0.0050646, 0.0064029, -0.0013382, 0.0013382
6: -0.0021738, -0.0008562, -0.0021738, -0.0008562, -0.0013176, 0.0013176
7: -0.0082275, -0.0074685, -0.0082275, -0.0074685, -0.0007590, 0.0007590
8: 0.0055062, 0.0098066, 0.0055062, 0.0098066, -0.0042191, 0.0042215
9: -0.0036825, -0.0031002, -0.0036825, -0.0031002, -0.0005823, 0.0005823

### Unstable ReLU Count (Linear/Conv2D Layers)
- layer_idx=0, type=LayerType.Linear, total=256, inp1_unstable=9, inp2_unstable=9, delta_unstable=250
- layer_idx=2, type=LayerType.Linear, total=256, inp1_unstable=6, inp2_unstable=6, delta_unstable=256
- layer_idx=4, type=LayerType.Linear, total=256, inp1_unstable=0, inp2_unstable=0, delta_unstable=256
- layer_idx=6, type=LayerType.Linear, total=256, inp1_unstable=2, inp2_unstable=2, delta_unstable=256
- layer_idx=8, type=LayerType.Linear, total=10, inp1_unstable=0, inp2_unstable=0, delta_unstable=10

Time for backsubstitution: 1.70 seconds

### RS candidates at layer 1
type: RSZ, layer: 1, pos: 52
type: RSZ, layer: 1, pos: 83
type: RSZ, layer: 1, pos: 199

Time for candidate selection: 0.00 seconds

### Candidate
type: RSZ, layer: 1, pos: 52

### Relational analysis RSZ of RS_RSZ1_RSZ2_RSZ1_RSZ2_RSZ1_RSZ1_RSZ1

#### Relational analysis RSZ result of RS_RSZ1_RSZ2_RSZ1_RSZ2_RSZ1_RSZ1_RSZ1
Status: Status.UNKNOWN
Output dim: 1, lower bound: -0.0007706, upper bound: 0.0007730
time: 0.68 seconds

### Relational analysis RSZ of RS_RSZ1_RSZ2_RSZ1_RSZ2_RSZ1_RSZ1_RSZ2

#### Relational analysis RSZ result of RS_RSZ1_RSZ2_RSZ1_RSZ2_RSZ1_RSZ1_RSZ2
Status: Status.UNKNOWN
Output dim: 1, lower bound: -0.0007706, upper bound: 0.0007731
time: 0.73 seconds

## BFS RS instance: RS_RSZ1_RSZ2_RSZ1_RSZ2_RSZ1_RSZ2

### Backsubstitution after applying RS history:
0: 0.0007184, 0.0011172, 0.0007184, 0.0011172, -0.0003988, 0.0003988
1: 0.9934075, 0.9943579, 0.9934075, 0.9943579, -0.0009505, 0.0009505
2: -0.0087946, -0.0055215, -0.0087946, -0.0055215, -0.0029802, 0.0029733
3: 0.0035868, 0.0041619, 0.0035868, 0.0041619, -0.0005751, 0.0005751
4: 0.0027809, 0.0053678, 0.0027809, 0.0053678, -0.0025869, 0.0025869
5: 0.0050646, 0.0064029, 0.0050646, 0.0064029, -0.0013382, 0.0013382
6: -0.0021738, -0.0008562, -0.0021738, -0.0008562, -0.0013176, 0.0013176
7: -0.0082275, -0.0074685, -0.0082275, -0.0074685, -0.0007590, 0.0007590
8: 0.0055062, 0.0098066, 0.0055062, 0.0098066, -0.0042196, 0.0042211
9: -0.0036825, -0.0031002, -0.0036825, -0.0031002, -0.0005823, 0.0005823

### Unstable ReLU Count (Linear/Conv2D Layers)
- layer_idx=0, type=LayerType.Linear, total=256, inp1_unstable=9, inp2_unstable=9, delta_unstable=250
- layer_idx=2, type=LayerType.Linear, total=256, inp1_unstable=6, inp2_unstable=6, delta_unstable=256
- layer_idx=4, type=LayerType.Linear, total=256, inp1_unstable=0, inp2_unstable=0, delta_unstable=256
- layer_idx=6, type=LayerType.Linear, total=256, inp1_unstable=2, inp2_unstable=2, delta_unstable=256
- layer_idx=8, type=LayerType.Linear, total=10, inp1_unstable=0, inp2_unstable=0, delta_unstable=10

Time for backsubstitution: 1.63 seconds

### RS candidates at layer 1
type: RSZ, layer: 1, pos: 83
type: RSZ, layer: 1, pos: 199
type: RSZ, layer: 1, pos: 52

Time for candidate selection: 0.00 seconds

### Candidate
type: RSZ, layer: 1, pos: 83

### Relational analysis RSZ of RS_RSZ1_RSZ2_RSZ1_RSZ2_RSZ1_RSZ2_RSZ1

#### Relational analysis RSZ result of RS_RSZ1_RSZ2_RSZ1_RSZ2_RSZ1_RSZ2_RSZ1
Status: Status.UNKNOWN
Output dim: 1, lower bound: -0.0007718, upper bound: 0.0007706
time: 0.65 seconds

### Relational analysis RSZ of RS_RSZ1_RSZ2_RSZ1_RSZ2_RSZ1_RSZ2_RSZ2

#### Relational analysis RSZ result of RS_RSZ1_RSZ2_RSZ1_RSZ2_RSZ1_RSZ2_RSZ2
Status: Status.UNKNOWN
Output dim: 1, lower bound: -0.0007718, upper bound: 0.0007709
time: 0.72 seconds

## BFS RS instance: RS_RSZ1_RSZ2_RSZ1_RSZ2_RSZ2_RSZ1

### Backsubstitution after applying RS history:
0: 0.0007184, 0.0011172, 0.0007184, 0.0011172, -0.0003988, 0.0003988
1: 0.9934075, 0.9943579, 0.9934075, 0.9943579, -0.0009505, 0.0009505
2: -0.0087946, -0.0055215, -0.0087946, -0.0055215, -0.0029849, 0.0029786
3: 0.0035868, 0.0041619, 0.0035868, 0.0041619, -0.0005751, 0.0005751
4: 0.0027809, 0.0053678, 0.0027809, 0.0053678, -0.0025869, 0.0025869
5: 0.0050646, 0.0064029, 0.0050646, 0.0064029, -0.0013382, 0.0013382
6: -0.0021738, -0.0008562, -0.0021738, -0.0008562, -0.0013176, 0.0013176
7: -0.0082275, -0.0074685, -0.0082275, -0.0074685, -0.0007590, 0.0007590
8: 0.0055062, 0.0098066, 0.0055062, 0.0098066, -0.0042214, 0.0042228
9: -0.0036825, -0.0031002, -0.0036825, -0.0031002, -0.0005823, 0.0005823

### Unstable ReLU Count (Linear/Conv2D Layers)
- layer_idx=0, type=LayerType.Linear, total=256, inp1_unstable=9, inp2_unstable=9, delta_unstable=250
- layer_idx=2, type=LayerType.Linear, total=256, inp1_unstable=6, inp2_unstable=6, delta_unstable=256
- layer_idx=4, type=LayerType.Linear, total=256, inp1_unstable=0, inp2_unstable=0, delta_unstable=256
- layer_idx=6, type=LayerType.Linear, total=256, inp1_unstable=2, inp2_unstable=2, delta_unstable=256
- layer_idx=8, type=LayerType.Linear, total=10, inp1_unstable=0, inp2_unstable=0, delta_unstable=10

Time for backsubstitution: 1.65 seconds

### RS candidates at layer 1
type: RSZ, layer: 1, pos: 199
type: RSZ, layer: 1, pos: 68
type: RSZ, layer: 1, pos: 83

Time for candidate selection: 0.00 seconds

### Candidate
type: RSZ, layer: 1, pos: 199

### Relational analysis RSZ of RS_RSZ1_RSZ2_RSZ1_RSZ2_RSZ2_RSZ1_RSZ1

#### Relational analysis RSZ result of RS_RSZ1_RSZ2_RSZ1_RSZ2_RSZ2_RSZ1_RSZ1
Status: Status.UNKNOWN
Output dim: 1, lower bound: -0.0007131, upper bound: 0.0007131
time: 0.60 seconds

### Relational analysis RSZ of RS_RSZ1_RSZ2_RSZ1_RSZ2_RSZ2_RSZ1_RSZ2

#### Relational analysis RSZ result of RS_RSZ1_RSZ2_RSZ1_RSZ2_RSZ2_RSZ1_RSZ2
Status: Status.UNKNOWN
Output dim: 1, lower bound: -0.0007131, upper bound: 0.0007131
time: 0.60 seconds

## BFS RS instance: RS_RSZ1_RSZ2_RSZ1_RSZ2_RSZ2_RSZ2

### Backsubstitution after applying RS history:
0: 0.0007184, 0.0011172, 0.0007184, 0.0011172, -0.0003988, 0.0003988
1: 0.9934075, 0.9943579, 0.9934075, 0.9943579, -0.0009505, 0.0009505
2: -0.0087946, -0.0055215, -0.0087946, -0.0055215, -0.0029857, 0.0029776
3: 0.0035868, 0.0041619, 0.0035868, 0.0041619, -0.0005751, 0.0005751
4: 0.0027809, 0.0053678, 0.0027809, 0.0053678, -0.0025869, 0.0025869
5: 0.0050646, 0.0064029, 0.0050646, 0.0064029, -0.0013382, 0.0013382
6: -0.0021738, -0.0008562, -0.0021738, -0.0008562, -0.0013176, 0.0013176
7: -0.0082275, -0.0074685, -0.0082275, -0.0074685, -0.0007590, 0.0007590
8: 0.0055062, 0.0098066, 0.0055062, 0.0098066, -0.0042211, 0.0042230
9: -0.0036825, -0.0031002, -0.0036825, -0.0031002, -0.0005823, 0.0005823

### Unstable ReLU Count (Linear/Conv2D Layers)
- layer_idx=0, type=LayerType.Linear, total=256, inp1_unstable=9, inp2_unstable=9, delta_unstable=250
- layer_idx=2, type=LayerType.Linear, total=256, inp1_unstable=6, inp2_unstable=6, delta_unstable=256
- layer_idx=4, type=LayerType.Linear, total=256, inp1_unstable=0, inp2_unstable=0, delta_unstable=256
- layer_idx=6, type=LayerType.Linear, total=256, inp1_unstable=2, inp2_unstable=2, delta_unstable=256
- layer_idx=8, type=LayerType.Linear, total=10, inp1_unstable=0, inp2_unstable=0, delta_unstable=10

Time for backsubstitution: 1.61 seconds

### RS candidates at layer 1
type: RSZ, layer: 1, pos: 199
type: RSZ, layer: 1, pos: 83
type: RSZ, layer: 1, pos: 68

Time for candidate selection: 0.00 seconds

### Candidate
type: RSZ, layer: 1, pos: 199

### Relational analysis RSZ of RS_RSZ1_RSZ2_RSZ1_RSZ2_RSZ2_RSZ2_RSZ1

#### Relational analysis RSZ result of RS_RSZ1_RSZ2_RSZ1_RSZ2_RSZ2_RSZ2_RSZ1
Status: Status.UNKNOWN
Output dim: 1, lower bound: -0.0007131, upper bound: 0.0007131
time: 0.60 seconds

### Relational analysis RSZ of RS_RSZ1_RSZ2_RSZ1_RSZ2_RSZ2_RSZ2_RSZ2

#### Relational analysis RSZ result of RS_RSZ1_RSZ2_RSZ1_RSZ2_RSZ2_RSZ2_RSZ2
Status: Status.UNKNOWN
Output dim: 1, lower bound: -0.0007131, upper bound: 0.0007131
time: 0.59 seconds

## BFS RS instance: RS_RSZ1_RSZ2_RSZ2_RSZ1_RSZ1_RSZ1

### Backsubstitution after applying RS history:
0: 0.0007184, 0.0011172, 0.0007184, 0.0011172, -0.0003988, 0.0003988
1: 0.9934075, 0.9943579, 0.9934075, 0.9943579, -0.0009505, 0.0009505
2: -0.0087946, -0.0055215, -0.0087946, -0.0055215, -0.0029835, 0.0029806
3: 0.0035868, 0.0041619, 0.0035868, 0.0041619, -0.0005751, 0.0005751
4: 0.0027809, 0.0053678, 0.0027809, 0.0053678, -0.0025869, 0.0025869
5: 0.0050646, 0.0064029, 0.0050646, 0.0064029, -0.0013382, 0.0013382
6: -0.0021738, -0.0008562, -0.0021738, -0.0008562, -0.0013176, 0.0013176
7: -0.0082275, -0.0074685, -0.0082275, -0.0074685, -0.0007590, 0.0007590
8: 0.0055062, 0.0098066, 0.0055062, 0.0098066, -0.0042213, 0.0042221
9: -0.0036825, -0.0031002, -0.0036825, -0.0031002, -0.0005823, 0.0005823

### Unstable ReLU Count (Linear/Conv2D Layers)
- layer_idx=0, type=LayerType.Linear, total=256, inp1_unstable=9, inp2_unstable=9, delta_unstable=250
- layer_idx=2, type=LayerType.Linear, total=256, inp1_unstable=6, inp2_unstable=6, delta_unstable=256
- layer_idx=4, type=LayerType.Linear, total=256, inp1_unstable=0, inp2_unstable=0, delta_unstable=256
- layer_idx=6, type=LayerType.Linear, total=256, inp1_unstable=2, inp2_unstable=2, delta_unstable=256
- layer_idx=8, type=LayerType.Linear, total=10, inp1_unstable=0, inp2_unstable=0, delta_unstable=10

Time for backsubstitution: 1.63 seconds

### RS candidates at layer 1
type: RSZ, layer: 1, pos: 52
type: RSZ, layer: 1, pos: 198
type: RSZ, layer: 1, pos: 204

Time for candidate selection: 0.00 seconds

### Candidate
type: RSZ, layer: 1, pos: 52

### Relational analysis RSZ of RS_RSZ1_RSZ2_RSZ2_RSZ1_RSZ1_RSZ1_RSZ1

#### Relational analysis RSZ result of RS_RSZ1_RSZ2_RSZ2_RSZ1_RSZ1_RSZ1_RSZ1
Status: Status.UNKNOWN
Output dim: 1, lower bound: -0.0007550, upper bound: 0.0007554
time: 0.65 seconds

### Relational analysis RSZ of RS_RSZ1_RSZ2_RSZ2_RSZ1_RSZ1_RSZ1_RSZ2

#### Relational analysis RSZ result of RS_RSZ1_RSZ2_RSZ2_RSZ1_RSZ1_RSZ1_RSZ2
Status: Status.UNKNOWN
Output dim: 1, lower bound: -0.0007546, upper bound: 0.0007556
time: 0.60 seconds

## BFS RS instance: RS_RSZ1_RSZ2_RSZ2_RSZ1_RSZ1_RSZ2

### Backsubstitution after applying RS history:
0: 0.0007184, 0.0011172, 0.0007184, 0.0011172, -0.0003988, 0.0003988
1: 0.9934075, 0.9943579, 0.9934075, 0.9943579, -0.0009505, 0.0009505
2: -0.0087946, -0.0055215, -0.0087946, -0.0055215, -0.0029817, 0.0029824
3: 0.0035868, 0.0041619, 0.0035868, 0.0041619, -0.0005751, 0.0005751
4: 0.0027809, 0.0053678, 0.0027809, 0.0053678, -0.0025869, 0.0025869
5: 0.0050646, 0.0064029, 0.0050646, 0.0064029, -0.0013382, 0.0013382
6: -0.0021738, -0.0008562, -0.0021738, -0.0008562, -0.0013176, 0.0013176
7: -0.0082275, -0.0074685, -0.0082275, -0.0074685, -0.0007590, 0.0007590
8: 0.0055062, 0.0098066, 0.0055062, 0.0098066, -0.0042218, 0.0042216
9: -0.0036825, -0.0031002, -0.0036825, -0.0031002, -0.0005823, 0.0005823

### Unstable ReLU Count (Linear/Conv2D Layers)
- layer_idx=0, type=LayerType.Linear, total=256, inp1_unstable=9, inp2_unstable=9, delta_unstable=250
- layer_idx=2, type=LayerType.Linear, total=256, inp1_unstable=6, inp2_unstable=6, delta_unstable=256
- layer_idx=4, type=LayerType.Linear, total=256, inp1_unstable=0, inp2_unstable=0, delta_unstable=256
- layer_idx=6, type=LayerType.Linear, total=256, inp1_unstable=2, inp2_unstable=2, delta_unstable=256
- layer_idx=8, type=LayerType.Linear, total=10, inp1_unstable=0, inp2_unstable=0, delta_unstable=10

Time for backsubstitution: 1.64 seconds

### RS candidates at layer 1
type: RSZ, layer: 1, pos: 198
type: RSZ, layer: 1, pos: 52
type: RSZ, layer: 1, pos: 204

Time for candidate selection: 0.00 seconds

### Candidate
type: RSZ, layer: 1, pos: 198

### Relational analysis RSZ of RS_RSZ1_RSZ2_RSZ2_RSZ1_RSZ1_RSZ2_RSZ1

#### Relational analysis RSZ result of RS_RSZ1_RSZ2_RSZ2_RSZ1_RSZ1_RSZ2_RSZ1
Status: Status.UNKNOWN
Output dim: 1, lower bound: -0.0007098, upper bound: 0.0007098
time: 0.54 seconds

### Relational analysis RSZ of RS_RSZ1_RSZ2_RSZ2_RSZ1_RSZ1_RSZ2_RSZ2

#### Relational analysis RSZ result of RS_RSZ1_RSZ2_RSZ2_RSZ1_RSZ1_RSZ2_RSZ2
Status: Status.UNKNOWN
Output dim: 1, lower bound: -0.0007098, upper bound: 0.0007098
time: 0.54 seconds

## BFS RS instance: RS_RSZ1_RSZ2_RSZ2_RSZ1_RSZ2_RSZ1

### Backsubstitution after applying RS history:
0: 0.0007184, 0.0011172, 0.0007184, 0.0011172, -0.0003988, 0.0003988
1: 0.9934075, 0.9943579, 0.9934075, 0.9943579, -0.0009505, 0.0009505
2: -0.0087946, -0.0055215, -0.0087946, -0.0055215, -0.0029832, 0.0029810
3: 0.0035868, 0.0041619, 0.0035868, 0.0041619, -0.0005751, 0.0005751
4: 0.0027809, 0.0053678, 0.0027809, 0.0053678, -0.0025869, 0.0025869
5: 0.0050646, 0.0064029, 0.0050646, 0.0064029, -0.0013382, 0.0013382
6: -0.0021738, -0.0008562, -0.0021738, -0.0008562, -0.0013176, 0.0013176
7: -0.0082275, -0.0074685, -0.0082275, -0.0074685, -0.0007590, 0.0007590
8: 0.0055062, 0.0098066, 0.0055062, 0.0098066, -0.0042214, 0.0042220
9: -0.0036825, -0.0031002, -0.0036825, -0.0031002, -0.0005823, 0.0005823

### Unstable ReLU Count (Linear/Conv2D Layers)
- layer_idx=0, type=LayerType.Linear, total=256, inp1_unstable=9, inp2_unstable=9, delta_unstable=250
- layer_idx=2, type=LayerType.Linear, total=256, inp1_unstable=6, inp2_unstable=6, delta_unstable=256
- layer_idx=4, type=LayerType.Linear, total=256, inp1_unstable=0, inp2_unstable=0, delta_unstable=256
- layer_idx=6, type=LayerType.Linear, total=256, inp1_unstable=2, inp2_unstable=2, delta_unstable=256
- layer_idx=8, type=LayerType.Linear, total=10, inp1_unstable=0, inp2_unstable=0, delta_unstable=10

Time for backsubstitution: 1.75 seconds

### RS candidates at layer 1
type: RSZ, layer: 1, pos: 198
type: RSZ, layer: 1, pos: 52
type: RSZ, layer: 1, pos: 204

Time for candidate selection: 0.00 seconds

### Candidate
type: RSZ, layer: 1, pos: 198

### Relational analysis RSZ of RS_RSZ1_RSZ2_RSZ2_RSZ1_RSZ2_RSZ1_RSZ1

#### Relational analysis RSZ result of RS_RSZ1_RSZ2_RSZ2_RSZ1_RSZ2_RSZ1_RSZ1
Status: Status.UNKNOWN
Output dim: 1, lower bound: -0.0007098, upper bound: 0.0007098
time: 0.57 seconds

### Relational analysis RSZ of RS_RSZ1_RSZ2_RSZ2_RSZ1_RSZ2_RSZ1_RSZ2

#### Relational analysis RSZ result of RS_RSZ1_RSZ2_RSZ2_RSZ1_RSZ2_RSZ1_RSZ2
Status: Status.UNKNOWN
Output dim: 1, lower bound: -0.0007098, upper bound: 0.0007098
time: 0.55 seconds

## BFS RS instance: RS_RSZ1_RSZ2_RSZ2_RSZ1_RSZ2_RSZ2

### Backsubstitution after applying RS history:
0: 0.0007184, 0.0011172, 0.0007184, 0.0011172, -0.0003988, 0.0003988
1: 0.9934075, 0.9943579, 0.9934075, 0.9943579, -0.0009505, 0.0009505
2: -0.0087946, -0.0055215, -0.0087946, -0.0055215, -0.0029814, 0.0029827
3: 0.0035868, 0.0041619, 0.0035868, 0.0041619, -0.0005751, 0.0005751
4: 0.0027809, 0.0053678, 0.0027809, 0.0053678, -0.0025869, 0.0025869
5: 0.0050646, 0.0064029, 0.0050646, 0.0064029, -0.0013382, 0.0013382
6: -0.0021738, -0.0008562, -0.0021738, -0.0008562, -0.0013176, 0.0013176
7: -0.0082275, -0.0074685, -0.0082275, -0.0074685, -0.0007590, 0.0007590
8: 0.0055062, 0.0098066, 0.0055062, 0.0098066, -0.0042219, 0.0042216
9: -0.0036825, -0.0031002, -0.0036825, -0.0031002, -0.0005823, 0.0005823

### Unstable ReLU Count (Linear/Conv2D Layers)
- layer_idx=0, type=LayerType.Linear, total=256, inp1_unstable=9, inp2_unstable=9, delta_unstable=250
- layer_idx=2, type=LayerType.Linear, total=256, inp1_unstable=6, inp2_unstable=6, delta_unstable=256
- layer_idx=4, type=LayerType.Linear, total=256, inp1_unstable=0, inp2_unstable=0, delta_unstable=256
- layer_idx=6, type=LayerType.Linear, total=256, inp1_unstable=2, inp2_unstable=2, delta_unstable=256
- layer_idx=8, type=LayerType.Linear, total=10, inp1_unstable=0, inp2_unstable=0, delta_unstable=10

Time for backsubstitution: 1.69 seconds

### RS candidates at layer 1
type: RSZ, layer: 1, pos: 204
type: RSZ, layer: 1, pos: 52
type: RSZ, layer: 1, pos: 198

Time for candidate selection: 0.00 seconds

### Candidate
type: RSZ, layer: 1, pos: 204

### Relational analysis RSZ of RS_RSZ1_RSZ2_RSZ2_RSZ1_RSZ2_RSZ2_RSZ1

#### Relational analysis RSZ result of RS_RSZ1_RSZ2_RSZ2_RSZ1_RSZ2_RSZ2_RSZ1
Status: Status.UNKNOWN
Output dim: 1, lower bound: -0.0007550, upper bound: 0.0007549
time: 0.64 seconds

### Relational analysis RSZ of RS_RSZ1_RSZ2_RSZ2_RSZ1_RSZ2_RSZ2_RSZ2

#### Relational analysis RSZ result of RS_RSZ1_RSZ2_RSZ2_RSZ1_RSZ2_RSZ2_RSZ2
Status: Status.UNKNOWN
Output dim: 1, lower bound: -0.0007552, upper bound: 0.0007547
time: 0.61 seconds

## BFS RS instance: RS_RSZ1_RSZ2_RSZ2_RSZ2_RSZ1_RSZ1

### Backsubstitution after applying RS history:
0: 0.0007184, 0.0011172, 0.0007184, 0.0011172, -0.0003988, 0.0003988
1: 0.9934075, 0.9943579, 0.9934075, 0.9943579, -0.0009505, 0.0009505
2: -0.0087946, -0.0055215, -0.0087946, -0.0055215, -0.0029765, 0.0029762
3: 0.0035868, 0.0041619, 0.0035868, 0.0041619, -0.0005751, 0.0005751
4: 0.0027809, 0.0053678, 0.0027809, 0.0053678, -0.0025869, 0.0025869
5: 0.0050646, 0.0064029, 0.0050646, 0.0064029, -0.0013382, 0.0013382
6: -0.0021738, -0.0008562, -0.0021738, -0.0008562, -0.0013176, 0.0013176
7: -0.0082275, -0.0074685, -0.0082275, -0.0074685, -0.0007590, 0.0007590
8: 0.0055062, 0.0098066, 0.0055062, 0.0098066, -0.0042203, 0.0042204
9: -0.0036825, -0.0031002, -0.0036825, -0.0031002, -0.0005823, 0.0005823

### Unstable ReLU Count (Linear/Conv2D Layers)
- layer_idx=0, type=LayerType.Linear, total=256, inp1_unstable=9, inp2_unstable=9, delta_unstable=250
- layer_idx=2, type=LayerType.Linear, total=256, inp1_unstable=6, inp2_unstable=6, delta_unstable=256
- layer_idx=4, type=LayerType.Linear, total=256, inp1_unstable=0, inp2_unstable=0, delta_unstable=256
- layer_idx=6, type=LayerType.Linear, total=256, inp1_unstable=2, inp2_unstable=2, delta_unstable=256
- layer_idx=8, type=LayerType.Linear, total=10, inp1_unstable=0, inp2_unstable=0, delta_unstable=10

Time for backsubstitution: 1.72 seconds

### RS candidates at layer 1
type: RSZ, layer: 1, pos: 204
type: RSZ, layer: 1, pos: 83
type: RSZ, layer: 1, pos: 52

Time for candidate selection: 0.00 seconds

### Candidate
type: RSZ, layer: 1, pos: 204

### Relational analysis RSZ of RS_RSZ1_RSZ2_RSZ2_RSZ2_RSZ1_RSZ1_RSZ1

#### Relational analysis RSZ result of RS_RSZ1_RSZ2_RSZ2_RSZ2_RSZ1_RSZ1_RSZ1
Status: Status.UNKNOWN
Output dim: 1, lower bound: -0.0007131, upper bound: 0.0007131
time: 0.62 seconds

### Relational analysis RSZ of RS_RSZ1_RSZ2_RSZ2_RSZ2_RSZ1_RSZ1_RSZ2

#### Relational analysis RSZ result of RS_RSZ1_RSZ2_RSZ2_RSZ2_RSZ1_RSZ1_RSZ2
Status: Status.UNKNOWN
Output dim: 1, lower bound: -0.0007131, upper bound: 0.0007131
time: 0.66 seconds

## BFS RS instance: RS_RSZ1_RSZ2_RSZ2_RSZ2_RSZ1_RSZ2

### Backsubstitution after applying RS history:
0: 0.0007184, 0.0011172, 0.0007184, 0.0011172, -0.0003988, 0.0003988
1: 0.9934075, 0.9943579, 0.9934075, 0.9943579, -0.0009505, 0.0009505
2: -0.0087946, -0.0055215, -0.0087946, -0.0055215, -0.0029748, 0.0029784
3: 0.0035868, 0.0041619, 0.0035868, 0.0041619, -0.0005751, 0.0005751
4: 0.0027809, 0.0053678, 0.0027809, 0.0053678, -0.0025869, 0.0025869
5: 0.0050646, 0.0064029, 0.0050646, 0.0064029, -0.0013382, 0.0013382
6: -0.0021738, -0.0008562, -0.0021738, -0.0008562, -0.0013176, 0.0013176
7: -0.0082275, -0.0074685, -0.0082275, -0.0074685, -0.0007590, 0.0007590
8: 0.0055062, 0.0098066, 0.0055062, 0.0098066, -0.0042209, 0.0042200
9: -0.0036825, -0.0031002, -0.0036825, -0.0031002, -0.0005823, 0.0005823

### Unstable ReLU Count (Linear/Conv2D Layers)
- layer_idx=0, type=LayerType.Linear, total=256, inp1_unstable=9, inp2_unstable=9, delta_unstable=250
- layer_idx=2, type=LayerType.Linear, total=256, inp1_unstable=6, inp2_unstable=6, delta_unstable=256
- layer_idx=4, type=LayerType.Linear, total=256, inp1_unstable=0, inp2_unstable=0, delta_unstable=256
- layer_idx=6, type=LayerType.Linear, total=256, inp1_unstable=2, inp2_unstable=2, delta_unstable=256
- layer_idx=8, type=LayerType.Linear, total=10, inp1_unstable=0, inp2_unstable=0, delta_unstable=10

Time for backsubstitution: 1.74 seconds

### RS candidates at layer 1
type: RSZ, layer: 1, pos: 204
type: RSZ, layer: 1, pos: 52
type: RSZ, layer: 1, pos: 83

Time for candidate selection: 0.00 seconds

### Candidate
type: RSZ, layer: 1, pos: 204

### Relational analysis RSZ of RS_RSZ1_RSZ2_RSZ2_RSZ2_RSZ1_RSZ2_RSZ1

#### Relational analysis RSZ result of RS_RSZ1_RSZ2_RSZ2_RSZ2_RSZ1_RSZ2_RSZ1
Status: Status.UNKNOWN
Output dim: 1, lower bound: -0.0007130, upper bound: 0.0007131
time: 0.65 seconds

### Relational analysis RSZ of RS_RSZ1_RSZ2_RSZ2_RSZ2_RSZ1_RSZ2_RSZ2

#### Relational analysis RSZ result of RS_RSZ1_RSZ2_RSZ2_RSZ2_RSZ1_RSZ2_RSZ2
Status: Status.UNKNOWN
Output dim: 1, lower bound: -0.0007131, upper bound: 0.0007131
time: 0.73 seconds

## BFS RS instance: RS_RSZ1_RSZ2_RSZ2_RSZ2_RSZ2_RSZ1

### Backsubstitution after applying RS history:
0: 0.0007184, 0.0011172, 0.0007184, 0.0011172, -0.0003988, 0.0003988
1: 0.9934075, 0.9943579, 0.9934075, 0.9943579, -0.0009505, 0.0009505
2: -0.0087946, -0.0055215, -0.0087946, -0.0055215, -0.0029910, 0.0029820
3: 0.0035868, 0.0041619, 0.0035868, 0.0041619, -0.0005751, 0.0005751
4: 0.0027809, 0.0053678, 0.0027809, 0.0053678, -0.0025869, 0.0025869
5: 0.0050646, 0.0064029, 0.0050646, 0.0064029, -0.0013382, 0.0013382
6: -0.0021738, -0.0008562, -0.0021738, -0.0008562, -0.0013176, 0.0013176
7: -0.0082275, -0.0074685, -0.0082275, -0.0074685, -0.0007590, 0.0007590
8: 0.0055062, 0.0098066, 0.0055062, 0.0098066, -0.0042222, 0.0042243
9: -0.0036825, -0.0031002, -0.0036825, -0.0031002, -0.0005823, 0.0005823

### Unstable ReLU Count (Linear/Conv2D Layers)
- layer_idx=0, type=LayerType.Linear, total=256, inp1_unstable=9, inp2_unstable=9, delta_unstable=250
- layer_idx=2, type=LayerType.Linear, total=256, inp1_unstable=6, inp2_unstable=6, delta_unstable=256
- layer_idx=4, type=LayerType.Linear, total=256, inp1_unstable=0, inp2_unstable=0, delta_unstable=256
- layer_idx=6, type=LayerType.Linear, total=256, inp1_unstable=2, inp2_unstable=2, delta_unstable=256
- layer_idx=8, type=LayerType.Linear, total=10, inp1_unstable=0, inp2_unstable=0, delta_unstable=10

Time for backsubstitution: 1.77 seconds

### RS candidates at layer 1
type: RSZ, layer: 1, pos: 68
type: RSZ, layer: 1, pos: 204
type: RSZ, layer: 1, pos: 83

Time for candidate selection: 0.00 seconds

### Candidate
type: RSZ, layer: 1, pos: 68

### Relational analysis RSZ of RS_RSZ1_RSZ2_RSZ2_RSZ2_RSZ2_RSZ1_RSZ1

#### Relational analysis RSZ result of RS_RSZ1_RSZ2_RSZ2_RSZ2_RSZ2_RSZ1_RSZ1
Status: Status.UNKNOWN
Output dim: 1, lower bound: -0.0007131, upper bound: 0.0007131
time: 0.66 seconds

### Relational analysis RSZ of RS_RSZ1_RSZ2_RSZ2_RSZ2_RSZ2_RSZ1_RSZ2

#### Relational analysis RSZ result of RS_RSZ1_RSZ2_RSZ2_RSZ2_RSZ2_RSZ1_RSZ2
Status: Status.UNKNOWN
Output dim: 1, lower bound: -0.0007131, upper bound: 0.0007131
time: 0.69 seconds

## BFS RS instance: RS_RSZ1_RSZ2_RSZ2_RSZ2_RSZ2_RSZ2

### Backsubstitution after applying RS history:
0: 0.0007184, 0.0011172, 0.0007184, 0.0011172, -0.0003988, 0.0003988
1: 0.9934075, 0.9943579, 0.9934075, 0.9943579, -0.0009505, 0.0009505
2: -0.0087946, -0.0055215, -0.0087946, -0.0055215, -0.0029919, 0.0029810
3: 0.0035868, 0.0041619, 0.0035868, 0.0041619, -0.0005751, 0.0005751
4: 0.0027809, 0.0053678, 0.0027809, 0.0053678, -0.0025869, 0.0025869
5: 0.0050646, 0.0064029, 0.0050646, 0.0064029, -0.0013382, 0.0013382
6: -0.0021738, -0.0008562, -0.0021738, -0.0008562, -0.0013176, 0.0013176
7: -0.0082275, -0.0074685, -0.0082275, -0.0074685, -0.0007590, 0.0007590
8: 0.0055062, 0.0098066, 0.0055062, 0.0098066, -0.0042219, 0.0042246
9: -0.0036825, -0.0031002, -0.0036825, -0.0031002, -0.0005823, 0.0005823

### Unstable ReLU Count (Linear/Conv2D Layers)
- layer_idx=0, type=LayerType.Linear, total=256, inp1_unstable=9, inp2_unstable=9, delta_unstable=250
- layer_idx=2, type=LayerType.Linear, total=256, inp1_unstable=6, inp2_unstable=6, delta_unstable=256
- layer_idx=4, type=LayerType.Linear, total=256, inp1_unstable=0, inp2_unstable=0, delta_unstable=256
- layer_idx=6, type=LayerType.Linear, total=256, inp1_unstable=2, inp2_unstable=2, delta_unstable=256
- layer_idx=8, type=LayerType.Linear, total=10, inp1_unstable=0, inp2_unstable=0, delta_unstable=10

Time for backsubstitution: 1.82 seconds

### RS candidates at layer 1
type: RSZ, layer: 1, pos: 204
type: RSZ, layer: 1, pos: 68
type: RSZ, layer: 1, pos: 83

Time for candidate selection: 0.00 seconds

### Candidate
type: RSZ, layer: 1, pos: 204

### Relational analysis RSZ of RS_RSZ1_RSZ2_RSZ2_RSZ2_RSZ2_RSZ2_RSZ1

#### Relational analysis RSZ result of RS_RSZ1_RSZ2_RSZ2_RSZ2_RSZ2_RSZ2_RSZ1
Status: Status.UNKNOWN
Output dim: 1, lower bound: -0.0007130, upper bound: 0.0007131
time: 0.75 seconds

### Relational analysis RSZ of RS_RSZ1_RSZ2_RSZ2_RSZ2_RSZ2_RSZ2_RSZ2

#### Relational analysis RSZ result of RS_RSZ1_RSZ2_RSZ2_RSZ2_RSZ2_RSZ2_RSZ2
Status: Status.UNKNOWN
Output dim: 1, lower bound: -0.0007130, upper bound: 0.0007131
time: 0.63 seconds

## BFS RS instance: RS_RSZ2_RSZ1_RSZ1_RSZ1_RSZ1_RSZ1

### Backsubstitution after applying RS history:
0: 0.0007184, 0.0011172, 0.0007184, 0.0011172, -0.0003988, 0.0003988
1: 0.9934075, 0.9943579, 0.9934075, 0.9943579, -0.0009505, 0.0009505
2: -0.0087946, -0.0055215, -0.0087946, -0.0055215, -0.0029824, 0.0029802
3: 0.0035868, 0.0041619, 0.0035868, 0.0041619, -0.0005751, 0.0005751
4: 0.0027809, 0.0053678, 0.0027809, 0.0053678, -0.0025869, 0.0025869
5: 0.0050646, 0.0064029, 0.0050646, 0.0064029, -0.0013382, 0.0013382
6: -0.0021738, -0.0008562, -0.0021738, -0.0008562, -0.0013176, 0.0013176
7: -0.0082275, -0.0074685, -0.0082275, -0.0074685, -0.0007590, 0.0007590
8: 0.0055062, 0.0098066, 0.0055062, 0.0098066, -0.0042207, 0.0042213
9: -0.0036825, -0.0031002, -0.0036825, -0.0031002, -0.0005823, 0.0005823

### Unstable ReLU Count (Linear/Conv2D Layers)
- layer_idx=0, type=LayerType.Linear, total=256, inp1_unstable=9, inp2_unstable=9, delta_unstable=250
- layer_idx=2, type=LayerType.Linear, total=256, inp1_unstable=6, inp2_unstable=6, delta_unstable=256
- layer_idx=4, type=LayerType.Linear, total=256, inp1_unstable=0, inp2_unstable=0, delta_unstable=256
- layer_idx=6, type=LayerType.Linear, total=256, inp1_unstable=2, inp2_unstable=2, delta_unstable=256
- layer_idx=8, type=LayerType.Linear, total=10, inp1_unstable=0, inp2_unstable=0, delta_unstable=10

Time for backsubstitution: 1.78 seconds

### RS candidates at layer 1
type: RSZ, layer: 1, pos: 198
type: RSZ, layer: 1, pos: 199
type: RSZ, layer: 1, pos: 136

Time for candidate selection: 0.00 seconds

### Candidate
type: RSZ, layer: 1, pos: 198

### Relational analysis RSZ of RS_RSZ2_RSZ1_RSZ1_RSZ1_RSZ1_RSZ1_RSZ1

#### Relational analysis RSZ result of RS_RSZ2_RSZ1_RSZ1_RSZ1_RSZ1_RSZ1_RSZ1
Status: Status.UNKNOWN
Output dim: 1, lower bound: -0.0007733, upper bound: 0.0007737
time: 0.73 seconds

### Relational analysis RSZ of RS_RSZ2_RSZ1_RSZ1_RSZ1_RSZ1_RSZ1_RSZ2

#### Relational analysis RSZ result of RS_RSZ2_RSZ1_RSZ1_RSZ1_RSZ1_RSZ1_RSZ2
Status: Status.UNKNOWN
Output dim: 1, lower bound: -0.0007733, upper bound: 0.0007737
time: 0.74 seconds

## BFS RS instance: RS_RSZ2_RSZ1_RSZ1_RSZ1_RSZ1_RSZ2

### Backsubstitution after applying RS history:
0: 0.0007184, 0.0011172, 0.0007184, 0.0011172, -0.0003988, 0.0003988
1: 0.9934075, 0.9943579, 0.9934075, 0.9943579, -0.0009505, 0.0009505
2: -0.0087946, -0.0055215, -0.0087946, -0.0055215, -0.0029820, 0.0029807
3: 0.0035868, 0.0041619, 0.0035868, 0.0041619, -0.0005751, 0.0005751
4: 0.0027809, 0.0053678, 0.0027809, 0.0053678, -0.0025869, 0.0025869
5: 0.0050646, 0.0064029, 0.0050646, 0.0064029, -0.0013382, 0.0013382
6: -0.0021738, -0.0008562, -0.0021738, -0.0008562, -0.0013176, 0.0013176
7: -0.0082275, -0.0074685, -0.0082275, -0.0074685, -0.0007590, 0.0007590
8: 0.0055062, 0.0098066, 0.0055062, 0.0098066, -0.0042209, 0.0042212
9: -0.0036825, -0.0031002, -0.0036825, -0.0031002, -0.0005823, 0.0005823

### Unstable ReLU Count (Linear/Conv2D Layers)
- layer_idx=0, type=LayerType.Linear, total=256, inp1_unstable=9, inp2_unstable=9, delta_unstable=250
- layer_idx=2, type=LayerType.Linear, total=256, inp1_unstable=6, inp2_unstable=6, delta_unstable=256
- layer_idx=4, type=LayerType.Linear, total=256, inp1_unstable=0, inp2_unstable=0, delta_unstable=256
- layer_idx=6, type=LayerType.Linear, total=256, inp1_unstable=2, inp2_unstable=2, delta_unstable=256
- layer_idx=8, type=LayerType.Linear, total=10, inp1_unstable=0, inp2_unstable=0, delta_unstable=10

Time for backsubstitution: 1.75 seconds

### RS candidates at layer 1
type: RSZ, layer: 1, pos: 136
type: RSZ, layer: 1, pos: 198
type: RSZ, layer: 1, pos: 199

Time for candidate selection: 0.00 seconds

### Candidate
type: RSZ, layer: 1, pos: 136

### Relational analysis RSZ of RS_RSZ2_RSZ1_RSZ1_RSZ1_RSZ1_RSZ2_RSZ1

#### Relational analysis RSZ result of RS_RSZ2_RSZ1_RSZ1_RSZ1_RSZ1_RSZ2_RSZ1
Status: Status.UNKNOWN
Output dim: 1, lower bound: -0.0007867, upper bound: 0.0007888
time: 0.67 seconds

### Relational analysis RSZ of RS_RSZ2_RSZ1_RSZ1_RSZ1_RSZ1_RSZ2_RSZ2

#### Relational analysis RSZ result of RS_RSZ2_RSZ1_RSZ1_RSZ1_RSZ1_RSZ2_RSZ2
Status: Status.UNKNOWN
Output dim: 1, lower bound: -0.0007866, upper bound: 0.0007888
time: 0.78 seconds

## BFS RS instance: RS_RSZ2_RSZ1_RSZ1_RSZ1_RSZ2_RSZ1

### Backsubstitution after applying RS history:
0: 0.0007184, 0.0011172, 0.0007184, 0.0011172, -0.0003988, 0.0003988
1: 0.9934075, 0.9943579, 0.9934075, 0.9943579, -0.0009505, 0.0009505
2: -0.0087946, -0.0055215, -0.0087946, -0.0055215, -0.0029835, 0.0029793
3: 0.0035868, 0.0041619, 0.0035868, 0.0041619, -0.0005751, 0.0005751
4: 0.0027809, 0.0053678, 0.0027809, 0.0053678, -0.0025869, 0.0025869
5: 0.0050646, 0.0064029, 0.0050646, 0.0064029, -0.0013382, 0.0013382
6: -0.0021738, -0.0008562, -0.0021738, -0.0008562, -0.0013176, 0.0013176
7: -0.0082275, -0.0074685, -0.0082275, -0.0074685, -0.0007590, 0.0007590
8: 0.0055062, 0.0098066, 0.0055062, 0.0098066, -0.0042205, 0.0042216
9: -0.0036825, -0.0031002, -0.0036825, -0.0031002, -0.0005823, 0.0005823

### Unstable ReLU Count (Linear/Conv2D Layers)
- layer_idx=0, type=LayerType.Linear, total=256, inp1_unstable=9, inp2_unstable=9, delta_unstable=250
- layer_idx=2, type=LayerType.Linear, total=256, inp1_unstable=6, inp2_unstable=6, delta_unstable=256
- layer_idx=4, type=LayerType.Linear, total=256, inp1_unstable=0, inp2_unstable=0, delta_unstable=256
- layer_idx=6, type=LayerType.Linear, total=256, inp1_unstable=2, inp2_unstable=2, delta_unstable=256
- layer_idx=8, type=LayerType.Linear, total=10, inp1_unstable=0, inp2_unstable=0, delta_unstable=10

Time for backsubstitution: 1.78 seconds

### RS candidates at layer 1
type: RSZ, layer: 1, pos: 199
type: RSZ, layer: 1, pos: 136
type: RSZ, layer: 1, pos: 198

Time for candidate selection: 0.00 seconds

### Candidate
type: RSZ, layer: 1, pos: 199

### Relational analysis RSZ of RS_RSZ2_RSZ1_RSZ1_RSZ1_RSZ2_RSZ1_RSZ1

#### Relational analysis RSZ result of RS_RSZ2_RSZ1_RSZ1_RSZ1_RSZ2_RSZ1_RSZ1
Status: Status.UNKNOWN
Output dim: 1, lower bound: -0.0007556, upper bound: 0.0007565
time: 0.64 seconds

### Relational analysis RSZ of RS_RSZ2_RSZ1_RSZ1_RSZ1_RSZ2_RSZ1_RSZ2

#### Relational analysis RSZ result of RS_RSZ2_RSZ1_RSZ1_RSZ1_RSZ2_RSZ1_RSZ2
Status: Status.UNKNOWN
Output dim: 1, lower bound: -0.0007556, upper bound: 0.0007565
time: 0.64 seconds

## BFS RS instance: RS_RSZ2_RSZ1_RSZ1_RSZ1_RSZ2_RSZ2

### Backsubstitution after applying RS history:
0: 0.0007184, 0.0011172, 0.0007184, 0.0011172, -0.0003988, 0.0003988
1: 0.9934075, 0.9943579, 0.9934075, 0.9943579, -0.0009505, 0.0009505
2: -0.0087946, -0.0055215, -0.0087946, -0.0055215, -0.0029830, 0.0029796
3: 0.0035868, 0.0041619, 0.0035868, 0.0041619, -0.0005751, 0.0005751
4: 0.0027809, 0.0053678, 0.0027809, 0.0053678, -0.0025869, 0.0025869
5: 0.0050646, 0.0064029, 0.0050646, 0.0064029, -0.0013382, 0.0013382
6: -0.0021738, -0.0008562, -0.0021738, -0.0008562, -0.0013176, 0.0013176
7: -0.0082275, -0.0074685, -0.0082275, -0.0074685, -0.0007590, 0.0007590
8: 0.0055062, 0.0098066, 0.0055062, 0.0098066, -0.0042206, 0.0042215
9: -0.0036825, -0.0031002, -0.0036825, -0.0031002, -0.0005823, 0.0005823

### Unstable ReLU Count (Linear/Conv2D Layers)
- layer_idx=0, type=LayerType.Linear, total=256, inp1_unstable=9, inp2_unstable=9, delta_unstable=250
- layer_idx=2, type=LayerType.Linear, total=256, inp1_unstable=6, inp2_unstable=6, delta_unstable=256
- layer_idx=4, type=LayerType.Linear, total=256, inp1_unstable=0, inp2_unstable=0, delta_unstable=256
- layer_idx=6, type=LayerType.Linear, total=256, inp1_unstable=2, inp2_unstable=2, delta_unstable=256
- layer_idx=8, type=LayerType.Linear, total=10, inp1_unstable=0, inp2_unstable=0, delta_unstable=10

Time for backsubstitution: 1.65 seconds

### RS candidates at layer 1
type: RSZ, layer: 1, pos: 198
type: RSZ, layer: 1, pos: 136
type: RSZ, layer: 1, pos: 199

Time for candidate selection: 0.00 seconds

### Candidate
type: RSZ, layer: 1, pos: 198

### Relational analysis RSZ of RS_RSZ2_RSZ1_RSZ1_RSZ1_RSZ2_RSZ2_RSZ1

#### Relational analysis RSZ result of RS_RSZ2_RSZ1_RSZ1_RSZ1_RSZ2_RSZ2_RSZ1
Status: Status.UNKNOWN
Output dim: 1, lower bound: -0.0007731, upper bound: 0.0007738
time: 0.74 seconds

### Relational analysis RSZ of RS_RSZ2_RSZ1_RSZ1_RSZ1_RSZ2_RSZ2_RSZ2

#### Relational analysis RSZ result of RS_RSZ2_RSZ1_RSZ1_RSZ1_RSZ2_RSZ2_RSZ2
Status: Status.UNKNOWN
Output dim: 1, lower bound: -0.0007731, upper bound: 0.0007738
time: 0.71 seconds

## BFS RS instance: RS_RSZ2_RSZ1_RSZ1_RSZ2_RSZ1_RSZ1

### Backsubstitution after applying RS history:
0: 0.0007184, 0.0011172, 0.0007184, 0.0011172, -0.0003988, 0.0003988
1: 0.9934075, 0.9943579, 0.9934075, 0.9943579, -0.0009505, 0.0009505
2: -0.0087946, -0.0055215, -0.0087946, -0.0055215, -0.0029751, 0.0029732
3: 0.0035868, 0.0041619, 0.0035868, 0.0041619, -0.0005751, 0.0005751
4: 0.0027809, 0.0053678, 0.0027809, 0.0053678, -0.0025869, 0.0025869
5: 0.0050646, 0.0064029, 0.0050646, 0.0064029, -0.0013382, 0.0013382
6: -0.0021738, -0.0008562, -0.0021738, -0.0008562, -0.0013176, 0.0013176
7: -0.0082275, -0.0074685, -0.0082275, -0.0074685, -0.0007590, 0.0007590
8: 0.0055062, 0.0098066, 0.0055062, 0.0098066, -0.0042192, 0.0042197
9: -0.0036825, -0.0031002, -0.0036825, -0.0031002, -0.0005823, 0.0005823

### Unstable ReLU Count (Linear/Conv2D Layers)
- layer_idx=0, type=LayerType.Linear, total=256, inp1_unstable=9, inp2_unstable=9, delta_unstable=250
- layer_idx=2, type=LayerType.Linear, total=256, inp1_unstable=6, inp2_unstable=6, delta_unstable=256
- layer_idx=4, type=LayerType.Linear, total=256, inp1_unstable=0, inp2_unstable=0, delta_unstable=256
- layer_idx=6, type=LayerType.Linear, total=256, inp1_unstable=2, inp2_unstable=2, delta_unstable=256
- layer_idx=8, type=LayerType.Linear, total=10, inp1_unstable=0, inp2_unstable=0, delta_unstable=10

Time for backsubstitution: 1.64 seconds

### RS candidates at layer 1
type: RSZ, layer: 1, pos: 52
type: RSZ, layer: 1, pos: 136
type: RSZ, layer: 1, pos: 199

Time for candidate selection: 0.00 seconds

### Candidate
type: RSZ, layer: 1, pos: 52

### Relational analysis RSZ of RS_RSZ2_RSZ1_RSZ1_RSZ2_RSZ1_RSZ1_RSZ1

#### Relational analysis RSZ result of RS_RSZ2_RSZ1_RSZ1_RSZ2_RSZ1_RSZ1_RSZ1
Status: Status.UNKNOWN
Output dim: 1, lower bound: -0.0007732, upper bound: 0.0007737
time: 0.74 seconds

### Relational analysis RSZ of RS_RSZ2_RSZ1_RSZ1_RSZ2_RSZ1_RSZ1_RSZ2

#### Relational analysis RSZ result of RS_RSZ2_RSZ1_RSZ1_RSZ2_RSZ1_RSZ1_RSZ2
Status: Status.UNKNOWN
Output dim: 1, lower bound: -0.0007732, upper bound: 0.0007738
time: 0.68 seconds

## BFS RS instance: RS_RSZ2_RSZ1_RSZ1_RSZ2_RSZ1_RSZ2

### Backsubstitution after applying RS history:
0: 0.0007184, 0.0011172, 0.0007184, 0.0011172, -0.0003988, 0.0003988
1: 0.9934075, 0.9943579, 0.9934075, 0.9943579, -0.0009505, 0.0009505
2: -0.0087946, -0.0055215, -0.0087946, -0.0055215, -0.0029747, 0.0029738
3: 0.0035868, 0.0041619, 0.0035868, 0.0041619, -0.0005751, 0.0005751
4: 0.0027809, 0.0053678, 0.0027809, 0.0053678, -0.0025869, 0.0025869
5: 0.0050646, 0.0064029, 0.0050646, 0.0064029, -0.0013382, 0.0013382
6: -0.0021738, -0.0008562, -0.0021738, -0.0008562, -0.0013176, 0.0013176
7: -0.0082275, -0.0074685, -0.0082275, -0.0074685, -0.0007590, 0.0007590
8: 0.0055062, 0.0098066, 0.0055062, 0.0098066, -0.0042194, 0.0042196
9: -0.0036825, -0.0031002, -0.0036825, -0.0031002, -0.0005823, 0.0005823

### Unstable ReLU Count (Linear/Conv2D Layers)
- layer_idx=0, type=LayerType.Linear, total=256, inp1_unstable=9, inp2_unstable=9, delta_unstable=250
- layer_idx=2, type=LayerType.Linear, total=256, inp1_unstable=6, inp2_unstable=6, delta_unstable=256
- layer_idx=4, type=LayerType.Linear, total=256, inp1_unstable=0, inp2_unstable=0, delta_unstable=256
- layer_idx=6, type=LayerType.Linear, total=256, inp1_unstable=2, inp2_unstable=2, delta_unstable=256
- layer_idx=8, type=LayerType.Linear, total=10, inp1_unstable=0, inp2_unstable=0, delta_unstable=10

Time for backsubstitution: 1.90 seconds

### RS candidates at layer 1
type: RSZ, layer: 1, pos: 136
type: RSZ, layer: 1, pos: 199
type: RSZ, layer: 1, pos: 52

Time for candidate selection: 0.00 seconds

### Candidate
type: RSZ, layer: 1, pos: 136

### Relational analysis RSZ of RS_RSZ2_RSZ1_RSZ1_RSZ2_RSZ1_RSZ2_RSZ1

#### Relational analysis RSZ result of RS_RSZ2_RSZ1_RSZ1_RSZ2_RSZ1_RSZ2_RSZ1
Status: Status.UNKNOWN
Output dim: 1, lower bound: -0.0007701, upper bound: 0.0007720
time: 0.75 seconds

### Relational analysis RSZ of RS_RSZ2_RSZ1_RSZ1_RSZ2_RSZ1_RSZ2_RSZ2

#### Relational analysis RSZ result of RS_RSZ2_RSZ1_RSZ1_RSZ2_RSZ1_RSZ2_RSZ2
Status: Status.UNKNOWN
Output dim: 1, lower bound: -0.0007700, upper bound: 0.0007719
time: 0.75 seconds

## BFS RS instance: RS_RSZ2_RSZ1_RSZ1_RSZ2_RSZ2_RSZ1

### Backsubstitution after applying RS history:
0: 0.0007184, 0.0011172, 0.0007184, 0.0011172, -0.0003988, 0.0003988
1: 0.9934075, 0.9943579, 0.9934075, 0.9943579, -0.0009505, 0.0009505
2: -0.0087946, -0.0055215, -0.0087946, -0.0055215, -0.0029850, 0.0029733
3: 0.0035868, 0.0041619, 0.0035868, 0.0041619, -0.0005751, 0.0005751
4: 0.0027809, 0.0053678, 0.0027809, 0.0053678, -0.0025869, 0.0025869
5: 0.0050646, 0.0064029, 0.0050646, 0.0064029, -0.0013382, 0.0013382
6: -0.0021738, -0.0008562, -0.0021738, -0.0008562, -0.0013176, 0.0013176
7: -0.0082275, -0.0074685, -0.0082275, -0.0074685, -0.0007590, 0.0007590
8: 0.0055062, 0.0098066, 0.0055062, 0.0098066, -0.0042191, 0.0042220
9: -0.0036825, -0.0031002, -0.0036825, -0.0031002, -0.0005823, 0.0005823

### Unstable ReLU Count (Linear/Conv2D Layers)
- layer_idx=0, type=LayerType.Linear, total=256, inp1_unstable=9, inp2_unstable=9, delta_unstable=250
- layer_idx=2, type=LayerType.Linear, total=256, inp1_unstable=6, inp2_unstable=6, delta_unstable=256
- layer_idx=4, type=LayerType.Linear, total=256, inp1_unstable=0, inp2_unstable=0, delta_unstable=256
- layer_idx=6, type=LayerType.Linear, total=256, inp1_unstable=2, inp2_unstable=2, delta_unstable=256
- layer_idx=8, type=LayerType.Linear, total=10, inp1_unstable=0, inp2_unstable=0, delta_unstable=10

Time for backsubstitution: 1.83 seconds

### RS candidates at layer 1
type: RSZ, layer: 1, pos: 176
type: RSZ, layer: 1, pos: 199
type: RSZ, layer: 1, pos: 136

Time for candidate selection: 0.00 seconds

### Candidate
type: RSZ, layer: 1, pos: 176

### Relational analysis RSZ of RS_RSZ2_RSZ1_RSZ1_RSZ2_RSZ2_RSZ1_RSZ1

#### Relational analysis RSZ result of RS_RSZ2_RSZ1_RSZ1_RSZ2_RSZ2_RSZ1_RSZ1
Status: Status.UNKNOWN
Output dim: 1, lower bound: -0.0007732, upper bound: 0.0007737
time: 0.72 seconds

### Relational analysis RSZ of RS_RSZ2_RSZ1_RSZ1_RSZ2_RSZ2_RSZ1_RSZ2

#### Relational analysis RSZ result of RS_RSZ2_RSZ1_RSZ1_RSZ2_RSZ2_RSZ1_RSZ2
Status: Status.UNKNOWN
Output dim: 1, lower bound: -0.0007730, upper bound: 0.0007737
time: 0.76 seconds

## BFS RS instance: RS_RSZ2_RSZ1_RSZ1_RSZ2_RSZ2_RSZ2

### Backsubstitution after applying RS history:
0: 0.0007184, 0.0011172, 0.0007184, 0.0011172, -0.0003988, 0.0003988
1: 0.9934075, 0.9943579, 0.9934075, 0.9943579, -0.0009505, 0.0009505
2: -0.0087946, -0.0055215, -0.0087946, -0.0055215, -0.0029860, 0.0029722
3: 0.0035868, 0.0041619, 0.0035868, 0.0041619, -0.0005751, 0.0005751
4: 0.0027809, 0.0053678, 0.0027809, 0.0053678, -0.0025869, 0.0025869
5: 0.0050646, 0.0064029, 0.0050646, 0.0064029, -0.0013382, 0.0013382
6: -0.0021738, -0.0008562, -0.0021738, -0.0008562, -0.0013176, 0.0013176
7: -0.0082275, -0.0074685, -0.0082275, -0.0074685, -0.0007590, 0.0007590
8: 0.0055062, 0.0098066, 0.0055062, 0.0098066, -0.0042189, 0.0042222
9: -0.0036825, -0.0031002, -0.0036825, -0.0031002, -0.0005823, 0.0005823

### Unstable ReLU Count (Linear/Conv2D Layers)
- layer_idx=0, type=LayerType.Linear, total=256, inp1_unstable=9, inp2_unstable=9, delta_unstable=250
- layer_idx=2, type=LayerType.Linear, total=256, inp1_unstable=6, inp2_unstable=6, delta_unstable=256
- layer_idx=4, type=LayerType.Linear, total=256, inp1_unstable=0, inp2_unstable=0, delta_unstable=256
- layer_idx=6, type=LayerType.Linear, total=256, inp1_unstable=2, inp2_unstable=2, delta_unstable=256
- layer_idx=8, type=LayerType.Linear, total=10, inp1_unstable=0, inp2_unstable=0, delta_unstable=10

Time for backsubstitution: 1.84 seconds

### RS candidates at layer 1
type: RSZ, layer: 1, pos: 199
type: RSZ, layer: 1, pos: 176
type: RSZ, layer: 1, pos: 136

Time for candidate selection: 0.00 seconds

### Candidate
type: RSZ, layer: 1, pos: 199

### Relational analysis RSZ of RS_RSZ2_RSZ1_RSZ1_RSZ2_RSZ2_RSZ2_RSZ1

#### Relational analysis RSZ result of RS_RSZ2_RSZ1_RSZ1_RSZ2_RSZ2_RSZ2_RSZ1
Status: Status.UNKNOWN
Output dim: 1, lower bound: -0.0007165, upper bound: 0.0007165
time: 0.66 seconds

### Relational analysis RSZ of RS_RSZ2_RSZ1_RSZ1_RSZ2_RSZ2_RSZ2_RSZ2

#### Relational analysis RSZ result of RS_RSZ2_RSZ1_RSZ1_RSZ2_RSZ2_RSZ2_RSZ2
Status: Status.UNKNOWN
Output dim: 1, lower bound: -0.0007165, upper bound: 0.0007165
time: 0.68 seconds

## BFS RS instance: RS_RSZ2_RSZ1_RSZ2_RSZ1_RSZ1_RSZ1

### Backsubstitution after applying RS history:
0: 0.0007184, 0.0011172, 0.0007184, 0.0011172, -0.0003988, 0.0003988
1: 0.9934075, 0.9943579, 0.9934075, 0.9943579, -0.0009505, 0.0009505
2: -0.0087946, -0.0055215, -0.0087946, -0.0055215, -0.0029828, 0.0029815
3: 0.0035868, 0.0041619, 0.0035868, 0.0041619, -0.0005751, 0.0005751
4: 0.0027809, 0.0053678, 0.0027809, 0.0053678, -0.0025869, 0.0025869
5: 0.0050646, 0.0064029, 0.0050646, 0.0064029, -0.0013382, 0.0013382
6: -0.0021738, -0.0008562, -0.0021738, -0.0008562, -0.0013176, 0.0013176
7: -0.0082275, -0.0074685, -0.0082275, -0.0074685, -0.0007590, 0.0007590
8: 0.0055062, 0.0098066, 0.0055062, 0.0098066, -0.0042212, 0.0042215
9: -0.0036825, -0.0031002, -0.0036825, -0.0031002, -0.0005823, 0.0005823

### Unstable ReLU Count (Linear/Conv2D Layers)
- layer_idx=0, type=LayerType.Linear, total=256, inp1_unstable=9, inp2_unstable=9, delta_unstable=250
- layer_idx=2, type=LayerType.Linear, total=256, inp1_unstable=6, inp2_unstable=6, delta_unstable=256
- layer_idx=4, type=LayerType.Linear, total=256, inp1_unstable=0, inp2_unstable=0, delta_unstable=256
- layer_idx=6, type=LayerType.Linear, total=256, inp1_unstable=2, inp2_unstable=2, delta_unstable=256
- layer_idx=8, type=LayerType.Linear, total=10, inp1_unstable=0, inp2_unstable=0, delta_unstable=10

Time for backsubstitution: 1.72 seconds

### RS candidates at layer 1
type: RSZ, layer: 1, pos: 136
type: RSZ, layer: 1, pos: 198
type: RSZ, layer: 1, pos: 52

Time for candidate selection: 0.00 seconds

### Candidate
type: RSZ, layer: 1, pos: 136

### Relational analysis RSZ of RS_RSZ2_RSZ1_RSZ2_RSZ1_RSZ1_RSZ1_RSZ1

#### Relational analysis RSZ result of RS_RSZ2_RSZ1_RSZ2_RSZ1_RSZ1_RSZ1_RSZ1
Status: Status.UNKNOWN
Output dim: 1, lower bound: -0.0007549, upper bound: 0.0007550
time: 0.65 seconds

### Relational analysis RSZ of RS_RSZ2_RSZ1_RSZ2_RSZ1_RSZ1_RSZ1_RSZ2

#### Relational analysis RSZ result of RS_RSZ2_RSZ1_RSZ2_RSZ1_RSZ1_RSZ1_RSZ2
Status: Status.UNKNOWN
Output dim: 1, lower bound: -0.0007547, upper bound: 0.0007555
time: 0.63 seconds

## BFS RS instance: RS_RSZ2_RSZ1_RSZ2_RSZ1_RSZ1_RSZ2

### Backsubstitution after applying RS history:
0: 0.0007184, 0.0011172, 0.0007184, 0.0011172, -0.0003988, 0.0003988
1: 0.9934075, 0.9943579, 0.9934075, 0.9943579, -0.0009505, 0.0009505
2: -0.0087946, -0.0055215, -0.0087946, -0.0055215, -0.0029823, 0.0029818
3: 0.0035868, 0.0041619, 0.0035868, 0.0041619, -0.0005751, 0.0005751
4: 0.0027809, 0.0053678, 0.0027809, 0.0053678, -0.0025869, 0.0025869
5: 0.0050646, 0.0064029, 0.0050646, 0.0064029, -0.0013382, 0.0013382
6: -0.0021738, -0.0008562, -0.0021738, -0.0008562, -0.0013176, 0.0013176
7: -0.0082275, -0.0074685, -0.0082275, -0.0074685, -0.0007590, 0.0007590
8: 0.0055062, 0.0098066, 0.0055062, 0.0098066, -0.0042213, 0.0042214
9: -0.0036825, -0.0031002, -0.0036825, -0.0031002, -0.0005823, 0.0005823

### Unstable ReLU Count (Linear/Conv2D Layers)
- layer_idx=0, type=LayerType.Linear, total=256, inp1_unstable=9, inp2_unstable=9, delta_unstable=250
- layer_idx=2, type=LayerType.Linear, total=256, inp1_unstable=6, inp2_unstable=6, delta_unstable=256
- layer_idx=4, type=LayerType.Linear, total=256, inp1_unstable=0, inp2_unstable=0, delta_unstable=256
- layer_idx=6, type=LayerType.Linear, total=256, inp1_unstable=2, inp2_unstable=2, delta_unstable=256
- layer_idx=8, type=LayerType.Linear, total=10, inp1_unstable=0, inp2_unstable=0, delta_unstable=10

Time for backsubstitution: 1.68 seconds

### RS candidates at layer 1
type: RSZ, layer: 1, pos: 198
type: RSZ, layer: 1, pos: 52
type: RSZ, layer: 1, pos: 136

Time for candidate selection: 0.00 seconds

### Candidate
type: RSZ, layer: 1, pos: 198

### Relational analysis RSZ of RS_RSZ2_RSZ1_RSZ2_RSZ1_RSZ1_RSZ2_RSZ1

#### Relational analysis RSZ result of RS_RSZ2_RSZ1_RSZ2_RSZ1_RSZ1_RSZ2_RSZ1
Status: Status.UNKNOWN
Output dim: 1, lower bound: -0.0007098, upper bound: 0.0007098
time: 0.61 seconds

### Relational analysis RSZ of RS_RSZ2_RSZ1_RSZ2_RSZ1_RSZ1_RSZ2_RSZ2

#### Relational analysis RSZ result of RS_RSZ2_RSZ1_RSZ2_RSZ1_RSZ1_RSZ2_RSZ2
Status: Status.UNKNOWN
Output dim: 1, lower bound: -0.0007098, upper bound: 0.0007098
time: 0.59 seconds

## BFS RS instance: RS_RSZ2_RSZ1_RSZ2_RSZ1_RSZ2_RSZ1

### Backsubstitution after applying RS history:
0: 0.0007184, 0.0011172, 0.0007184, 0.0011172, -0.0003988, 0.0003988
1: 0.9934075, 0.9943579, 0.9934075, 0.9943579, -0.0009505, 0.0009505
2: -0.0087946, -0.0055215, -0.0087946, -0.0055215, -0.0029757, 0.0029765
3: 0.0035868, 0.0041619, 0.0035868, 0.0041619, -0.0005751, 0.0005751
4: 0.0027809, 0.0053678, 0.0027809, 0.0053678, -0.0025869, 0.0025869
5: 0.0050646, 0.0064029, 0.0050646, 0.0064029, -0.0013382, 0.0013382
6: -0.0021738, -0.0008562, -0.0021738, -0.0008562, -0.0013176, 0.0013176
7: -0.0082275, -0.0074685, -0.0082275, -0.0074685, -0.0007590, 0.0007590
8: 0.0055062, 0.0098066, 0.0055062, 0.0098066, -0.0042200, 0.0042198
9: -0.0036825, -0.0031002, -0.0036825, -0.0031002, -0.0005823, 0.0005823

### Unstable ReLU Count (Linear/Conv2D Layers)
- layer_idx=0, type=LayerType.Linear, total=256, inp1_unstable=9, inp2_unstable=9, delta_unstable=250
- layer_idx=2, type=LayerType.Linear, total=256, inp1_unstable=6, inp2_unstable=6, delta_unstable=256
- layer_idx=4, type=LayerType.Linear, total=256, inp1_unstable=0, inp2_unstable=0, delta_unstable=256
- layer_idx=6, type=LayerType.Linear, total=256, inp1_unstable=2, inp2_unstable=2, delta_unstable=256
- layer_idx=8, type=LayerType.Linear, total=10, inp1_unstable=0, inp2_unstable=0, delta_unstable=10

Time for backsubstitution: 1.69 seconds

### RS candidates at layer 1
type: RSZ, layer: 1, pos: 136
type: RSZ, layer: 1, pos: 83
type: RSZ, layer: 1, pos: 52

Time for candidate selection: 0.00 seconds

### Candidate
type: RSZ, layer: 1, pos: 136

### Relational analysis RSZ of RS_RSZ2_RSZ1_RSZ2_RSZ1_RSZ2_RSZ1_RSZ1

#### Relational analysis RSZ result of RS_RSZ2_RSZ1_RSZ2_RSZ1_RSZ2_RSZ1_RSZ1
Status: Status.UNKNOWN
Output dim: 1, lower bound: -0.0007131, upper bound: 0.0007131
time: 0.65 seconds

### Relational analysis RSZ of RS_RSZ2_RSZ1_RSZ2_RSZ1_RSZ2_RSZ1_RSZ2

#### Relational analysis RSZ result of RS_RSZ2_RSZ1_RSZ2_RSZ1_RSZ2_RSZ1_RSZ2
Status: Status.UNKNOWN
Output dim: 1, lower bound: -0.0007131, upper bound: 0.0007131
time: 0.63 seconds

## BFS RS instance: RS_RSZ2_RSZ1_RSZ2_RSZ1_RSZ2_RSZ2

### Backsubstitution after applying RS history:
0: 0.0007184, 0.0011172, 0.0007184, 0.0011172, -0.0003988, 0.0003988
1: 0.9934075, 0.9943579, 0.9934075, 0.9943579, -0.0009505, 0.0009505
2: -0.0087946, -0.0055215, -0.0087946, -0.0055215, -0.0029863, 0.0029761
3: 0.0035868, 0.0041619, 0.0035868, 0.0041619, -0.0005751, 0.0005751
4: 0.0027809, 0.0053678, 0.0027809, 0.0053678, -0.0025869, 0.0025869
5: 0.0050646, 0.0064029, 0.0050646, 0.0064029, -0.0013382, 0.0013382
6: -0.0021738, -0.0008562, -0.0021738, -0.0008562, -0.0013176, 0.0013176
7: -0.0082275, -0.0074685, -0.0082275, -0.0074685, -0.0007590, 0.0007590
8: 0.0055062, 0.0098066, 0.0055062, 0.0098066, -0.0042199, 0.0042224
9: -0.0036825, -0.0031002, -0.0036825, -0.0031002, -0.0005823, 0.0005823

### Unstable ReLU Count (Linear/Conv2D Layers)
- layer_idx=0, type=LayerType.Linear, total=256, inp1_unstable=9, inp2_unstable=9, delta_unstable=250
- layer_idx=2, type=LayerType.Linear, total=256, inp1_unstable=6, inp2_unstable=6, delta_unstable=256
- layer_idx=4, type=LayerType.Linear, total=256, inp1_unstable=0, inp2_unstable=0, delta_unstable=256
- layer_idx=6, type=LayerType.Linear, total=256, inp1_unstable=2, inp2_unstable=2, delta_unstable=256
- layer_idx=8, type=LayerType.Linear, total=10, inp1_unstable=0, inp2_unstable=0, delta_unstable=10

Time for backsubstitution: 1.74 seconds

### RS candidates at layer 1
type: RSZ, layer: 1, pos: 136
type: RSZ, layer: 1, pos: 52
type: RSZ, layer: 1, pos: 83

Time for candidate selection: 0.00 seconds

### Candidate
type: RSZ, layer: 1, pos: 136

### Relational analysis RSZ of RS_RSZ2_RSZ1_RSZ2_RSZ1_RSZ2_RSZ2_RSZ1

#### Relational analysis RSZ result of RS_RSZ2_RSZ1_RSZ2_RSZ1_RSZ2_RSZ2_RSZ1
Status: Status.UNKNOWN
Output dim: 1, lower bound: -0.0007131, upper bound: 0.0007131
time: 0.62 seconds

### Relational analysis RSZ of RS_RSZ2_RSZ1_RSZ2_RSZ1_RSZ2_RSZ2_RSZ2

#### Relational analysis RSZ result of RS_RSZ2_RSZ1_RSZ2_RSZ1_RSZ2_RSZ2_RSZ2
Status: Status.UNKNOWN
Output dim: 1, lower bound: -0.0007131, upper bound: 0.0007131
time: 0.62 seconds

## BFS RS instance: RS_RSZ2_RSZ1_RSZ2_RSZ2_RSZ1_RSZ1

### Backsubstitution after applying RS history:
0: 0.0007184, 0.0011172, 0.0007184, 0.0011172, -0.0003988, 0.0003988
1: 0.9934075, 0.9943579, 0.9934075, 0.9943579, -0.0009505, 0.0009505
2: -0.0087946, -0.0055215, -0.0087946, -0.0055215, -0.0029781, 0.0029794
3: 0.0035868, 0.0041619, 0.0035868, 0.0041619, -0.0005751, 0.0005751
4: 0.0027809, 0.0053678, 0.0027809, 0.0053678, -0.0025869, 0.0025869
5: 0.0050646, 0.0064029, 0.0050646, 0.0064029, -0.0013382, 0.0013382
6: -0.0021738, -0.0008562, -0.0021738, -0.0008562, -0.0013176, 0.0013176
7: -0.0082275, -0.0074685, -0.0082275, -0.0074685, -0.0007590, 0.0007590
8: 0.0055062, 0.0098066, 0.0055062, 0.0098066, -0.0042211, 0.0042207
9: -0.0036825, -0.0031002, -0.0036825, -0.0031002, -0.0005823, 0.0005823

### Unstable ReLU Count (Linear/Conv2D Layers)
- layer_idx=0, type=LayerType.Linear, total=256, inp1_unstable=9, inp2_unstable=9, delta_unstable=250
- layer_idx=2, type=LayerType.Linear, total=256, inp1_unstable=6, inp2_unstable=6, delta_unstable=256
- layer_idx=4, type=LayerType.Linear, total=256, inp1_unstable=0, inp2_unstable=0, delta_unstable=256
- layer_idx=6, type=LayerType.Linear, total=256, inp1_unstable=2, inp2_unstable=2, delta_unstable=256
- layer_idx=8, type=LayerType.Linear, total=10, inp1_unstable=0, inp2_unstable=0, delta_unstable=10

Time for backsubstitution: 1.69 seconds

### RS candidates at layer 1
type: RSZ, layer: 1, pos: 52
type: RSZ, layer: 1, pos: 198
type: RSZ, layer: 1, pos: 199

Time for candidate selection: 0.00 seconds

### Candidate
type: RSZ, layer: 1, pos: 52

### Relational analysis RSZ of RS_RSZ2_RSZ1_RSZ2_RSZ2_RSZ1_RSZ1_RSZ1

#### Relational analysis RSZ result of RS_RSZ2_RSZ1_RSZ2_RSZ2_RSZ1_RSZ1_RSZ1
Status: Status.UNKNOWN
Output dim: 1, lower bound: -0.0007900, upper bound: 0.0007888
time: 0.65 seconds

### Relational analysis RSZ of RS_RSZ2_RSZ1_RSZ2_RSZ2_RSZ1_RSZ1_RSZ2

#### Relational analysis RSZ result of RS_RSZ2_RSZ1_RSZ2_RSZ2_RSZ1_RSZ1_RSZ2
Status: Status.UNKNOWN
Output dim: 1, lower bound: -0.0007876, upper bound: 0.0007912
time: 0.74 seconds

## BFS RS instance: RS_RSZ2_RSZ1_RSZ2_RSZ2_RSZ1_RSZ2

### Backsubstitution after applying RS history:
0: 0.0007184, 0.0011172, 0.0007184, 0.0011172, -0.0003988, 0.0003988
1: 0.9934075, 0.9943579, 0.9934075, 0.9943579, -0.0009505, 0.0009505
2: -0.0087946, -0.0055215, -0.0087946, -0.0055215, -0.0029803, 0.0029775
3: 0.0035868, 0.0041619, 0.0035868, 0.0041619, -0.0005751, 0.0005751
4: 0.0027809, 0.0053678, 0.0027809, 0.0053678, -0.0025869, 0.0025869
5: 0.0050646, 0.0064029, 0.0050646, 0.0064029, -0.0013382, 0.0013382
6: -0.0021738, -0.0008562, -0.0021738, -0.0008562, -0.0013176, 0.0013176
7: -0.0082275, -0.0074685, -0.0082275, -0.0074685, -0.0007590, 0.0007590
8: 0.0055062, 0.0098066, 0.0055062, 0.0098066, -0.0042205, 0.0042213
9: -0.0036825, -0.0031002, -0.0036825, -0.0031002, -0.0005823, 0.0005823

### Unstable ReLU Count (Linear/Conv2D Layers)
- layer_idx=0, type=LayerType.Linear, total=256, inp1_unstable=9, inp2_unstable=9, delta_unstable=250
- layer_idx=2, type=LayerType.Linear, total=256, inp1_unstable=6, inp2_unstable=6, delta_unstable=256
- layer_idx=4, type=LayerType.Linear, total=256, inp1_unstable=0, inp2_unstable=0, delta_unstable=256
- layer_idx=6, type=LayerType.Linear, total=256, inp1_unstable=2, inp2_unstable=2, delta_unstable=256
- layer_idx=8, type=LayerType.Linear, total=10, inp1_unstable=0, inp2_unstable=0, delta_unstable=10

Time for backsubstitution: 1.71 seconds

### RS candidates at layer 1
type: RSZ, layer: 1, pos: 52
type: RSZ, layer: 1, pos: 199
type: RSZ, layer: 1, pos: 198

Time for candidate selection: 0.00 seconds

### Candidate
type: RSZ, layer: 1, pos: 52

### Relational analysis RSZ of RS_RSZ2_RSZ1_RSZ2_RSZ2_RSZ1_RSZ2_RSZ1

#### Relational analysis RSZ result of RS_RSZ2_RSZ1_RSZ2_RSZ2_RSZ1_RSZ2_RSZ1
Status: Status.UNKNOWN
Output dim: 1, lower bound: -0.0007899, upper bound: 0.0007887
time: 0.66 seconds

### Relational analysis RSZ of RS_RSZ2_RSZ1_RSZ2_RSZ2_RSZ1_RSZ2_RSZ2

#### Relational analysis RSZ result of RS_RSZ2_RSZ1_RSZ2_RSZ2_RSZ1_RSZ2_RSZ2
Status: Status.UNKNOWN
Output dim: 1, lower bound: -0.0007876, upper bound: 0.0007913
time: 0.75 seconds

## BFS RS instance: RS_RSZ2_RSZ1_RSZ2_RSZ2_RSZ2_RSZ1

### Backsubstitution after applying RS history:
0: 0.0007184, 0.0011172, 0.0007184, 0.0011172, -0.0003988, 0.0003988
1: 0.9934075, 0.9943579, 0.9934075, 0.9943579, -0.0009505, 0.0009505
2: -0.0087946, -0.0055215, -0.0087946, -0.0055215, -0.0029779, 0.0029798
3: 0.0035868, 0.0041619, 0.0035868, 0.0041619, -0.0005751, 0.0005751
4: 0.0027809, 0.0053678, 0.0027809, 0.0053678, -0.0025869, 0.0025869
5: 0.0050646, 0.0064029, 0.0050646, 0.0064029, -0.0013382, 0.0013382
6: -0.0021738, -0.0008562, -0.0021738, -0.0008562, -0.0013176, 0.0013176
7: -0.0082275, -0.0074685, -0.0082275, -0.0074685, -0.0007590, 0.0007590
8: 0.0055062, 0.0098066, 0.0055062, 0.0098066, -0.0042212, 0.0042206
9: -0.0036825, -0.0031002, -0.0036825, -0.0031002, -0.0005823, 0.0005823

### Unstable ReLU Count (Linear/Conv2D Layers)
- layer_idx=0, type=LayerType.Linear, total=256, inp1_unstable=9, inp2_unstable=9, delta_unstable=250
- layer_idx=2, type=LayerType.Linear, total=256, inp1_unstable=6, inp2_unstable=6, delta_unstable=256
- layer_idx=4, type=LayerType.Linear, total=256, inp1_unstable=0, inp2_unstable=0, delta_unstable=256
- layer_idx=6, type=LayerType.Linear, total=256, inp1_unstable=2, inp2_unstable=2, delta_unstable=256
- layer_idx=8, type=LayerType.Linear, total=10, inp1_unstable=0, inp2_unstable=0, delta_unstable=10

Time for backsubstitution: 1.68 seconds

### RS candidates at layer 1
type: RSZ, layer: 1, pos: 198
type: RSZ, layer: 1, pos: 52
type: RSZ, layer: 1, pos: 199

Time for candidate selection: 0.00 seconds

### Candidate
type: RSZ, layer: 1, pos: 198

### Relational analysis RSZ of RS_RSZ2_RSZ1_RSZ2_RSZ2_RSZ2_RSZ1_RSZ1

#### Relational analysis RSZ result of RS_RSZ2_RSZ1_RSZ2_RSZ2_RSZ2_RSZ1_RSZ1
Status: Status.UNKNOWN
Output dim: 1, lower bound: -0.0007707, upper bound: 0.0007719
time: 0.85 seconds

### Relational analysis RSZ of RS_RSZ2_RSZ1_RSZ2_RSZ2_RSZ2_RSZ1_RSZ2

#### Relational analysis RSZ result of RS_RSZ2_RSZ1_RSZ2_RSZ2_RSZ2_RSZ1_RSZ2
Status: Status.UNKNOWN
Output dim: 1, lower bound: -0.0007707, upper bound: 0.0007719
time: 0.82 seconds

## BFS RS instance: RS_RSZ2_RSZ1_RSZ2_RSZ2_RSZ2_RSZ2

### Backsubstitution after applying RS history:
0: 0.0007184, 0.0011172, 0.0007184, 0.0011172, -0.0003988, 0.0003988
1: 0.9934075, 0.9943579, 0.9934075, 0.9943579, -0.0009505, 0.0009505
2: -0.0087946, -0.0055215, -0.0087946, -0.0055215, -0.0029801, 0.0029777
3: 0.0035868, 0.0041619, 0.0035868, 0.0041619, -0.0005751, 0.0005751
4: 0.0027809, 0.0053678, 0.0027809, 0.0053678, -0.0025869, 0.0025869
5: 0.0050646, 0.0064029, 0.0050646, 0.0064029, -0.0013382, 0.0013382
6: -0.0021738, -0.0008562, -0.0021738, -0.0008562, -0.0013176, 0.0013176
7: -0.0082275, -0.0074685, -0.0082275, -0.0074685, -0.0007590, 0.0007590
8: 0.0055062, 0.0098066, 0.0055062, 0.0098066, -0.0042206, 0.0042212
9: -0.0036825, -0.0031002, -0.0036825, -0.0031002, -0.0005823, 0.0005823

### Unstable ReLU Count (Linear/Conv2D Layers)
- layer_idx=0, type=LayerType.Linear, total=256, inp1_unstable=9, inp2_unstable=9, delta_unstable=250
- layer_idx=2, type=LayerType.Linear, total=256, inp1_unstable=6, inp2_unstable=6, delta_unstable=256
- layer_idx=4, type=LayerType.Linear, total=256, inp1_unstable=0, inp2_unstable=0, delta_unstable=256
- layer_idx=6, type=LayerType.Linear, total=256, inp1_unstable=2, inp2_unstable=2, delta_unstable=256
- layer_idx=8, type=LayerType.Linear, total=10, inp1_unstable=0, inp2_unstable=0, delta_unstable=10

Time for backsubstitution: 1.77 seconds

### RS candidates at layer 1
type: RSZ, layer: 1, pos: 199
type: RSZ, layer: 1, pos: 198
type: RSZ, layer: 1, pos: 52

Time for candidate selection: 0.00 seconds

### Candidate
type: RSZ, layer: 1, pos: 199

### Relational analysis RSZ of RS_RSZ2_RSZ1_RSZ2_RSZ2_RSZ2_RSZ2_RSZ1

#### Relational analysis RSZ result of RS_RSZ2_RSZ1_RSZ2_RSZ2_RSZ2_RSZ2_RSZ1
Status: Status.UNKNOWN
Output dim: 1, lower bound: -0.0007546, upper bound: 0.0007554
time: 0.65 seconds

### Relational analysis RSZ of RS_RSZ2_RSZ1_RSZ2_RSZ2_RSZ2_RSZ2_RSZ2

#### Relational analysis RSZ result of RS_RSZ2_RSZ1_RSZ2_RSZ2_RSZ2_RSZ2_RSZ2
Status: Status.UNKNOWN
Output dim: 1, lower bound: -0.0007546, upper bound: 0.0007554
time: 0.65 seconds

## BFS RS instance: RS_RSZ2_RSZ2_RSZ1_RSZ1_RSZ1_RSZ1

### Backsubstitution after applying RS history:
0: 0.0007184, 0.0011172, 0.0007184, 0.0011172, -0.0003988, 0.0003988
1: 0.9934075, 0.9943579, 0.9934075, 0.9943579, -0.0009505, 0.0009505
2: -0.0087946, -0.0055215, -0.0087946, -0.0055215, -0.0029768, 0.0029807
3: 0.0035868, 0.0041619, 0.0035868, 0.0041619, -0.0005751, 0.0005751
4: 0.0027809, 0.0053678, 0.0027809, 0.0053678, -0.0025869, 0.0025869
5: 0.0050646, 0.0064029, 0.0050646, 0.0064029, -0.0013382, 0.0013382
6: -0.0021738, -0.0008562, -0.0021738, -0.0008562, -0.0013176, 0.0013176
7: -0.0082275, -0.0074685, -0.0082275, -0.0074685, -0.0007590, 0.0007590
8: 0.0055062, 0.0098066, 0.0055062, 0.0098066, -0.0042214, 0.0042203
9: -0.0036825, -0.0031002, -0.0036825, -0.0031002, -0.0005823, 0.0005823

### Unstable ReLU Count (Linear/Conv2D Layers)
- layer_idx=0, type=LayerType.Linear, total=256, inp1_unstable=9, inp2_unstable=9, delta_unstable=250
- layer_idx=2, type=LayerType.Linear, total=256, inp1_unstable=6, inp2_unstable=6, delta_unstable=256
- layer_idx=4, type=LayerType.Linear, total=256, inp1_unstable=0, inp2_unstable=0, delta_unstable=256
- layer_idx=6, type=LayerType.Linear, total=256, inp1_unstable=2, inp2_unstable=2, delta_unstable=256
- layer_idx=8, type=LayerType.Linear, total=10, inp1_unstable=0, inp2_unstable=0, delta_unstable=10

Time for backsubstitution: 1.76 seconds

### RS candidates at layer 1
type: RSZ, layer: 1, pos: 176
type: RSZ, layer: 1, pos: 199
type: RSZ, layer: 1, pos: 198

Time for candidate selection: 0.00 seconds

### Candidate
type: RSZ, layer: 1, pos: 176

### Relational analysis RSZ of RS_RSZ2_RSZ2_RSZ1_RSZ1_RSZ1_RSZ1_RSZ1

#### Relational analysis RSZ result of RS_RSZ2_RSZ2_RSZ1_RSZ1_RSZ1_RSZ1_RSZ1
Status: Status.UNKNOWN
Output dim: 1, lower bound: -0.0007903, upper bound: 0.0007873
time: 0.65 seconds

### Relational analysis RSZ of RS_RSZ2_RSZ2_RSZ1_RSZ1_RSZ1_RSZ1_RSZ2

#### Relational analysis RSZ result of RS_RSZ2_RSZ2_RSZ1_RSZ1_RSZ1_RSZ1_RSZ2
Status: Status.UNKNOWN
Output dim: 1, lower bound: -0.0007908, upper bound: 0.0007875
time: 0.65 seconds

## BFS RS instance: RS_RSZ2_RSZ2_RSZ1_RSZ1_RSZ1_RSZ2

### Backsubstitution after applying RS history:
0: 0.0007184, 0.0011172, 0.0007184, 0.0011172, -0.0003988, 0.0003988
1: 0.9934075, 0.9943579, 0.9934075, 0.9943579, -0.0009505, 0.0009505
2: -0.0087946, -0.0055215, -0.0087946, -0.0055215, -0.0029765, 0.0029811
3: 0.0035868, 0.0041619, 0.0035868, 0.0041619, -0.0005751, 0.0005751
4: 0.0027809, 0.0053678, 0.0027809, 0.0053678, -0.0025869, 0.0025869
5: 0.0050646, 0.0064029, 0.0050646, 0.0064029, -0.0013382, 0.0013382
6: -0.0021738, -0.0008562, -0.0021738, -0.0008562, -0.0013176, 0.0013176
7: -0.0082275, -0.0074685, -0.0082275, -0.0074685, -0.0007590, 0.0007590
8: 0.0055062, 0.0098066, 0.0055062, 0.0098066, -0.0042215, 0.0042203
9: -0.0036825, -0.0031002, -0.0036825, -0.0031002, -0.0005823, 0.0005823

### Unstable ReLU Count (Linear/Conv2D Layers)
- layer_idx=0, type=LayerType.Linear, total=256, inp1_unstable=9, inp2_unstable=9, delta_unstable=250
- layer_idx=2, type=LayerType.Linear, total=256, inp1_unstable=6, inp2_unstable=6, delta_unstable=256
- layer_idx=4, type=LayerType.Linear, total=256, inp1_unstable=0, inp2_unstable=0, delta_unstable=256
- layer_idx=6, type=LayerType.Linear, total=256, inp1_unstable=2, inp2_unstable=2, delta_unstable=256
- layer_idx=8, type=LayerType.Linear, total=10, inp1_unstable=0, inp2_unstable=0, delta_unstable=10

Time for backsubstitution: 1.73 seconds

### RS candidates at layer 1
type: RSZ, layer: 1, pos: 199
type: RSZ, layer: 1, pos: 176
type: RSZ, layer: 1, pos: 198

Time for candidate selection: 0.00 seconds

### Candidate
type: RSZ, layer: 1, pos: 199

### Relational analysis RSZ of RS_RSZ2_RSZ2_RSZ1_RSZ1_RSZ1_RSZ2_RSZ1

#### Relational analysis RSZ result of RS_RSZ2_RSZ2_RSZ1_RSZ1_RSZ1_RSZ2_RSZ1
Status: Status.UNKNOWN
Output dim: 1, lower bound: -0.0007617, upper bound: 0.0007603
time: 0.69 seconds

### Relational analysis RSZ of RS_RSZ2_RSZ2_RSZ1_RSZ1_RSZ1_RSZ2_RSZ2

#### Relational analysis RSZ result of RS_RSZ2_RSZ2_RSZ1_RSZ1_RSZ1_RSZ2_RSZ2
Status: Status.UNKNOWN
Output dim: 1, lower bound: -0.0007617, upper bound: 0.0007603
time: 0.63 seconds

## BFS RS instance: RS_RSZ2_RSZ2_RSZ1_RSZ1_RSZ2_RSZ1

### Backsubstitution after applying RS history:
0: 0.0007184, 0.0011172, 0.0007184, 0.0011172, -0.0003988, 0.0003988
1: 0.9934075, 0.9943579, 0.9934075, 0.9943579, -0.0009505, 0.0009505
2: -0.0087946, -0.0055215, -0.0087946, -0.0055215, -0.0029713, 0.0029724
3: 0.0035868, 0.0041619, 0.0035868, 0.0041619, -0.0005751, 0.0005751
4: 0.0027809, 0.0053678, 0.0027809, 0.0053678, -0.0025869, 0.0025869
5: 0.0050646, 0.0064029, 0.0050646, 0.0064029, -0.0013382, 0.0013382
6: -0.0021738, -0.0008562, -0.0021738, -0.0008562, -0.0013176, 0.0013176
7: -0.0082275, -0.0074685, -0.0082275, -0.0074685, -0.0007590, 0.0007590
8: 0.0055062, 0.0098066, 0.0055062, 0.0098066, -0.0042193, 0.0042190
9: -0.0036825, -0.0031002, -0.0036825, -0.0031002, -0.0005823, 0.0005823

### Unstable ReLU Count (Linear/Conv2D Layers)
- layer_idx=0, type=LayerType.Linear, total=256, inp1_unstable=9, inp2_unstable=9, delta_unstable=250
- layer_idx=2, type=LayerType.Linear, total=256, inp1_unstable=6, inp2_unstable=6, delta_unstable=256
- layer_idx=4, type=LayerType.Linear, total=256, inp1_unstable=0, inp2_unstable=0, delta_unstable=256
- layer_idx=6, type=LayerType.Linear, total=256, inp1_unstable=2, inp2_unstable=2, delta_unstable=256
- layer_idx=8, type=LayerType.Linear, total=10, inp1_unstable=0, inp2_unstable=0, delta_unstable=10

Time for backsubstitution: 1.78 seconds

### RS candidates at layer 1
type: RSZ, layer: 1, pos: 83
type: RSZ, layer: 1, pos: 199
type: RSZ, layer: 1, pos: 176

Time for candidate selection: 0.00 seconds

### Candidate
type: RSZ, layer: 1, pos: 83

### Relational analysis RSZ of RS_RSZ2_RSZ2_RSZ1_RSZ1_RSZ2_RSZ1_RSZ1

#### Relational analysis RSZ result of RS_RSZ2_RSZ2_RSZ1_RSZ1_RSZ2_RSZ1_RSZ1
Status: Status.UNKNOWN
Output dim: 1, lower bound: -0.0007771, upper bound: 0.0007754
time: 0.82 seconds

### Relational analysis RSZ of RS_RSZ2_RSZ2_RSZ1_RSZ1_RSZ2_RSZ1_RSZ2

#### Relational analysis RSZ result of RS_RSZ2_RSZ2_RSZ1_RSZ1_RSZ2_RSZ1_RSZ2
Status: Status.UNKNOWN
Output dim: 1, lower bound: -0.0007771, upper bound: 0.0007757
time: 0.68 seconds

## BFS RS instance: RS_RSZ2_RSZ2_RSZ1_RSZ1_RSZ2_RSZ2

### Backsubstitution after applying RS history:
0: 0.0007184, 0.0011172, 0.0007184, 0.0011172, -0.0003988, 0.0003988
1: 0.9934075, 0.9943579, 0.9934075, 0.9943579, -0.0009505, 0.0009505
2: -0.0087946, -0.0055215, -0.0087946, -0.0055215, -0.0029819, 0.0029711
3: 0.0035868, 0.0041619, 0.0035868, 0.0041619, -0.0005751, 0.0005751
4: 0.0027809, 0.0053678, 0.0027809, 0.0053678, -0.0025869, 0.0025869
5: 0.0050646, 0.0064029, 0.0050646, 0.0064029, -0.0013382, 0.0013382
6: -0.0021738, -0.0008562, -0.0021738, -0.0008562, -0.0013176, 0.0013176
7: -0.0082275, -0.0074685, -0.0082275, -0.0074685, -0.0007590, 0.0007590
8: 0.0055062, 0.0098066, 0.0055062, 0.0098066, -0.0042189, 0.0042215
9: -0.0036825, -0.0031002, -0.0036825, -0.0031002, -0.0005823, 0.0005823

### Unstable ReLU Count (Linear/Conv2D Layers)
- layer_idx=0, type=LayerType.Linear, total=256, inp1_unstable=9, inp2_unstable=9, delta_unstable=250
- layer_idx=2, type=LayerType.Linear, total=256, inp1_unstable=6, inp2_unstable=6, delta_unstable=256
- layer_idx=4, type=LayerType.Linear, total=256, inp1_unstable=0, inp2_unstable=0, delta_unstable=256
- layer_idx=6, type=LayerType.Linear, total=256, inp1_unstable=2, inp2_unstable=2, delta_unstable=256
- layer_idx=8, type=LayerType.Linear, total=10, inp1_unstable=0, inp2_unstable=0, delta_unstable=10

Time for backsubstitution: 1.77 seconds

### RS candidates at layer 1
type: RSZ, layer: 1, pos: 176
type: RSZ, layer: 1, pos: 199
type: RSZ, layer: 1, pos: 83

Time for candidate selection: 0.00 seconds

### Candidate
type: RSZ, layer: 1, pos: 176

### Relational analysis RSZ of RS_RSZ2_RSZ2_RSZ1_RSZ1_RSZ2_RSZ2_RSZ1

#### Relational analysis RSZ result of RS_RSZ2_RSZ2_RSZ1_RSZ1_RSZ2_RSZ2_RSZ1
Status: Status.UNKNOWN
Output dim: 1, lower bound: -0.0007730, upper bound: 0.0007719
time: 0.70 seconds

### Relational analysis RSZ of RS_RSZ2_RSZ2_RSZ1_RSZ1_RSZ2_RSZ2_RSZ2

#### Relational analysis RSZ result of RS_RSZ2_RSZ2_RSZ1_RSZ1_RSZ2_RSZ2_RSZ2
Status: Status.UNKNOWN
Output dim: 1, lower bound: -0.0007731, upper bound: 0.0007719
time: 0.70 seconds

## BFS RS instance: RS_RSZ2_RSZ2_RSZ1_RSZ2_RSZ1_RSZ1

### Backsubstitution after applying RS history:
0: 0.0007184, 0.0011172, 0.0007184, 0.0011172, -0.0003988, 0.0003988
1: 0.9934075, 0.9943579, 0.9934075, 0.9943579, -0.0009505, 0.0009505
2: -0.0087946, -0.0055215, -0.0087946, -0.0055215, -0.0029763, 0.0029813
3: 0.0035868, 0.0041619, 0.0035868, 0.0041619, -0.0005751, 0.0005751
4: 0.0027809, 0.0053678, 0.0027809, 0.0053678, -0.0025869, 0.0025869
5: 0.0050646, 0.0064029, 0.0050646, 0.0064029, -0.0013382, 0.0013382
6: -0.0021738, -0.0008562, -0.0021738, -0.0008562, -0.0013176, 0.0013176
7: -0.0082275, -0.0074685, -0.0082275, -0.0074685, -0.0007590, 0.0007590
8: 0.0055062, 0.0098066, 0.0055062, 0.0098066, -0.0042215, 0.0042202
9: -0.0036825, -0.0031002, -0.0036825, -0.0031002, -0.0005823, 0.0005823

### Unstable ReLU Count (Linear/Conv2D Layers)
- layer_idx=0, type=LayerType.Linear, total=256, inp1_unstable=9, inp2_unstable=9, delta_unstable=250
- layer_idx=2, type=LayerType.Linear, total=256, inp1_unstable=6, inp2_unstable=6, delta_unstable=256
- layer_idx=4, type=LayerType.Linear, total=256, inp1_unstable=0, inp2_unstable=0, delta_unstable=256
- layer_idx=6, type=LayerType.Linear, total=256, inp1_unstable=2, inp2_unstable=2, delta_unstable=256
- layer_idx=8, type=LayerType.Linear, total=10, inp1_unstable=0, inp2_unstable=0, delta_unstable=10

Time for backsubstitution: 1.74 seconds

### RS candidates at layer 1
type: RSZ, layer: 1, pos: 176
type: RSZ, layer: 1, pos: 198
type: RSZ, layer: 1, pos: 199

Time for candidate selection: 0.00 seconds

### Candidate
type: RSZ, layer: 1, pos: 176

### Relational analysis RSZ of RS_RSZ2_RSZ2_RSZ1_RSZ2_RSZ1_RSZ1_RSZ1

#### Relational analysis RSZ result of RS_RSZ2_RSZ2_RSZ1_RSZ2_RSZ1_RSZ1_RSZ1
Status: Status.UNKNOWN
Output dim: 1, lower bound: -0.0007915, upper bound: 0.0007862
time: 0.75 seconds

### Relational analysis RSZ of RS_RSZ2_RSZ2_RSZ1_RSZ2_RSZ1_RSZ1_RSZ2

#### Relational analysis RSZ result of RS_RSZ2_RSZ2_RSZ1_RSZ2_RSZ1_RSZ1_RSZ2
Status: Status.UNKNOWN
Output dim: 1, lower bound: -0.0007919, upper bound: 0.0007864
time: 0.65 seconds

## BFS RS instance: RS_RSZ2_RSZ2_RSZ1_RSZ2_RSZ1_RSZ2

### Backsubstitution after applying RS history:
0: 0.0007184, 0.0011172, 0.0007184, 0.0011172, -0.0003988, 0.0003988
1: 0.9934075, 0.9943579, 0.9934075, 0.9943579, -0.0009505, 0.0009505
2: -0.0087946, -0.0055215, -0.0087946, -0.0055215, -0.0029759, 0.0029817
3: 0.0035868, 0.0041619, 0.0035868, 0.0041619, -0.0005751, 0.0005751
4: 0.0027809, 0.0053678, 0.0027809, 0.0053678, -0.0025869, 0.0025869
5: 0.0050646, 0.0064029, 0.0050646, 0.0064029, -0.0013382, 0.0013382
6: -0.0021738, -0.0008562, -0.0021738, -0.0008562, -0.0013176, 0.0013176
7: -0.0082275, -0.0074685, -0.0082275, -0.0074685, -0.0007590, 0.0007590
8: 0.0055062, 0.0098066, 0.0055062, 0.0098066, -0.0042216, 0.0042201
9: -0.0036825, -0.0031002, -0.0036825, -0.0031002, -0.0005823, 0.0005823

### Unstable ReLU Count (Linear/Conv2D Layers)
- layer_idx=0, type=LayerType.Linear, total=256, inp1_unstable=9, inp2_unstable=9, delta_unstable=250
- layer_idx=2, type=LayerType.Linear, total=256, inp1_unstable=6, inp2_unstable=6, delta_unstable=256
- layer_idx=4, type=LayerType.Linear, total=256, inp1_unstable=0, inp2_unstable=0, delta_unstable=256
- layer_idx=6, type=LayerType.Linear, total=256, inp1_unstable=2, inp2_unstable=2, delta_unstable=256
- layer_idx=8, type=LayerType.Linear, total=10, inp1_unstable=0, inp2_unstable=0, delta_unstable=10

Time for backsubstitution: 1.83 seconds

### RS candidates at layer 1
type: RSZ, layer: 1, pos: 198
type: RSZ, layer: 1, pos: 199
type: RSZ, layer: 1, pos: 176

Time for candidate selection: 0.00 seconds

### Candidate
type: RSZ, layer: 1, pos: 198

### Relational analysis RSZ of RS_RSZ2_RSZ2_RSZ1_RSZ2_RSZ1_RSZ2_RSZ1

#### Relational analysis RSZ result of RS_RSZ2_RSZ2_RSZ1_RSZ2_RSZ1_RSZ2_RSZ1
Status: Status.UNKNOWN
Output dim: 1, lower bound: -0.0007774, upper bound: 0.0007744
time: 0.79 seconds

### Relational analysis RSZ of RS_RSZ2_RSZ2_RSZ1_RSZ2_RSZ1_RSZ2_RSZ2

#### Relational analysis RSZ result of RS_RSZ2_RSZ2_RSZ1_RSZ2_RSZ1_RSZ2_RSZ2
Status: Status.UNKNOWN
Output dim: 1, lower bound: -0.0007774, upper bound: 0.0007746
time: 0.79 seconds

## BFS RS instance: RS_RSZ2_RSZ2_RSZ1_RSZ2_RSZ2_RSZ1

### Backsubstitution after applying RS history:
0: 0.0007184, 0.0011172, 0.0007184, 0.0011172, -0.0003988, 0.0003988
1: 0.9934075, 0.9943579, 0.9934075, 0.9943579, -0.0009505, 0.0009505
2: -0.0087946, -0.0055215, -0.0087946, -0.0055215, -0.0029784, 0.0029792
3: 0.0035868, 0.0041619, 0.0035868, 0.0041619, -0.0005751, 0.0005751
4: 0.0027809, 0.0053678, 0.0027809, 0.0053678, -0.0025869, 0.0025869
5: 0.0050646, 0.0064029, 0.0050646, 0.0064029, -0.0013382, 0.0013382
6: -0.0021738, -0.0008562, -0.0021738, -0.0008562, -0.0013176, 0.0013176
7: -0.0082275, -0.0074685, -0.0082275, -0.0074685, -0.0007590, 0.0007590
8: 0.0055062, 0.0098066, 0.0055062, 0.0098066, -0.0042210, 0.0042208
9: -0.0036825, -0.0031002, -0.0036825, -0.0031002, -0.0005823, 0.0005823

### Unstable ReLU Count (Linear/Conv2D Layers)
- layer_idx=0, type=LayerType.Linear, total=256, inp1_unstable=9, inp2_unstable=9, delta_unstable=250
- layer_idx=2, type=LayerType.Linear, total=256, inp1_unstable=6, inp2_unstable=6, delta_unstable=256
- layer_idx=4, type=LayerType.Linear, total=256, inp1_unstable=0, inp2_unstable=0, delta_unstable=256
- layer_idx=6, type=LayerType.Linear, total=256, inp1_unstable=2, inp2_unstable=2, delta_unstable=256
- layer_idx=8, type=LayerType.Linear, total=10, inp1_unstable=0, inp2_unstable=0, delta_unstable=10

Time for backsubstitution: 1.80 seconds

### RS candidates at layer 1
type: RSZ, layer: 1, pos: 176
type: RSZ, layer: 1, pos: 198
type: RSZ, layer: 1, pos: 199

Time for candidate selection: 0.00 seconds

### Candidate
type: RSZ, layer: 1, pos: 176

### Relational analysis RSZ of RS_RSZ2_RSZ2_RSZ1_RSZ2_RSZ2_RSZ1_RSZ1

#### Relational analysis RSZ result of RS_RSZ2_RSZ2_RSZ1_RSZ2_RSZ2_RSZ1_RSZ1
Status: Status.UNKNOWN
Output dim: 1, lower bound: -0.0007913, upper bound: 0.0007862
time: 0.72 seconds

### Relational analysis RSZ of RS_RSZ2_RSZ2_RSZ1_RSZ2_RSZ2_RSZ1_RSZ2

#### Relational analysis RSZ result of RS_RSZ2_RSZ2_RSZ1_RSZ2_RSZ2_RSZ1_RSZ2
Status: Status.UNKNOWN
Output dim: 1, lower bound: -0.0007916, upper bound: 0.0007864
time: 0.66 seconds

## BFS RS instance: RS_RSZ2_RSZ2_RSZ1_RSZ2_RSZ2_RSZ2

### Backsubstitution after applying RS history:
0: 0.0007184, 0.0011172, 0.0007184, 0.0011172, -0.0003988, 0.0003988
1: 0.9934075, 0.9943579, 0.9934075, 0.9943579, -0.0009505, 0.0009505
2: -0.0087946, -0.0055215, -0.0087946, -0.0055215, -0.0029781, 0.0029794
3: 0.0035868, 0.0041619, 0.0035868, 0.0041619, -0.0005751, 0.0005751
4: 0.0027809, 0.0053678, 0.0027809, 0.0053678, -0.0025869, 0.0025869
5: 0.0050646, 0.0064029, 0.0050646, 0.0064029, -0.0013382, 0.0013382
6: -0.0021738, -0.0008562, -0.0021738, -0.0008562, -0.0013176, 0.0013176
7: -0.0082275, -0.0074685, -0.0082275, -0.0074685, -0.0007590, 0.0007590
8: 0.0055062, 0.0098066, 0.0055062, 0.0098066, -0.0042210, 0.0042207
9: -0.0036825, -0.0031002, -0.0036825, -0.0031002, -0.0005823, 0.0005823

### Unstable ReLU Count (Linear/Conv2D Layers)
- layer_idx=0, type=LayerType.Linear, total=256, inp1_unstable=9, inp2_unstable=9, delta_unstable=250
- layer_idx=2, type=LayerType.Linear, total=256, inp1_unstable=6, inp2_unstable=6, delta_unstable=256
- layer_idx=4, type=LayerType.Linear, total=256, inp1_unstable=0, inp2_unstable=0, delta_unstable=256
- layer_idx=6, type=LayerType.Linear, total=256, inp1_unstable=2, inp2_unstable=2, delta_unstable=256
- layer_idx=8, type=LayerType.Linear, total=10, inp1_unstable=0, inp2_unstable=0, delta_unstable=10

Time for backsubstitution: 1.76 seconds

### RS candidates at layer 1
type: RSZ, layer: 1, pos: 199
type: RSZ, layer: 1, pos: 176
type: RSZ, layer: 1, pos: 198

Time for candidate selection: 0.00 seconds

### Candidate
type: RSZ, layer: 1, pos: 199

### Relational analysis RSZ of RS_RSZ2_RSZ2_RSZ1_RSZ2_RSZ2_RSZ2_RSZ1

#### Relational analysis RSZ result of RS_RSZ2_RSZ2_RSZ1_RSZ2_RSZ2_RSZ2_RSZ1
Status: Status.UNKNOWN
Output dim: 1, lower bound: -0.0007615, upper bound: 0.0007599
time: 0.78 seconds

### Relational analysis RSZ of RS_RSZ2_RSZ2_RSZ1_RSZ2_RSZ2_RSZ2_RSZ2

#### Relational analysis RSZ result of RS_RSZ2_RSZ2_RSZ1_RSZ2_RSZ2_RSZ2_RSZ2
Status: Status.UNKNOWN
Output dim: 1, lower bound: -0.0007615, upper bound: 0.0007599
time: 0.73 seconds

## BFS RS instance: RS_RSZ2_RSZ2_RSZ2_RSZ1_RSZ1_RSZ1

### Backsubstitution after applying RS history:
0: 0.0007184, 0.0011172, 0.0007184, 0.0011172, -0.0003988, 0.0003988
1: 0.9934075, 0.9943579, 0.9934075, 0.9943579, -0.0009505, 0.0009505
2: -0.0087946, -0.0055215, -0.0087946, -0.0055215, -0.0029768, 0.0029784
3: 0.0035868, 0.0041619, 0.0035868, 0.0041619, -0.0005751, 0.0005751
4: 0.0027809, 0.0053678, 0.0027809, 0.0053678, -0.0025869, 0.0025869
5: 0.0050646, 0.0064029, 0.0050646, 0.0064029, -0.0013382, 0.0013382
6: -0.0021738, -0.0008562, -0.0021738, -0.0008562, -0.0013176, 0.0013176
7: -0.0082275, -0.0074685, -0.0082275, -0.0074685, -0.0007590, 0.0007590
8: 0.0055062, 0.0098066, 0.0055062, 0.0098066, -0.0042204, 0.0042200
9: -0.0036825, -0.0031002, -0.0036825, -0.0031002, -0.0005823, 0.0005823

### Unstable ReLU Count (Linear/Conv2D Layers)
- layer_idx=0, type=LayerType.Linear, total=256, inp1_unstable=9, inp2_unstable=9, delta_unstable=250
- layer_idx=2, type=LayerType.Linear, total=256, inp1_unstable=6, inp2_unstable=6, delta_unstable=256
- layer_idx=4, type=LayerType.Linear, total=256, inp1_unstable=0, inp2_unstable=0, delta_unstable=256
- layer_idx=6, type=LayerType.Linear, total=256, inp1_unstable=2, inp2_unstable=2, delta_unstable=256
- layer_idx=8, type=LayerType.Linear, total=10, inp1_unstable=0, inp2_unstable=0, delta_unstable=10

Time for backsubstitution: 1.96 seconds

### RS candidates at layer 1
type: RSZ, layer: 1, pos: 136
type: RSZ, layer: 1, pos: 83
type: RSZ, layer: 1, pos: 204

Time for candidate selection: 0.00 seconds

### Candidate
type: RSZ, layer: 1, pos: 136

### Relational analysis RSZ of RS_RSZ2_RSZ2_RSZ2_RSZ1_RSZ1_RSZ1_RSZ1

#### Relational analysis RSZ result of RS_RSZ2_RSZ2_RSZ2_RSZ1_RSZ1_RSZ1_RSZ1
Status: Status.UNKNOWN
Output dim: 1, lower bound: -0.0007131, upper bound: 0.0007131
time: 0.68 seconds

### Relational analysis RSZ of RS_RSZ2_RSZ2_RSZ2_RSZ1_RSZ1_RSZ1_RSZ2

#### Relational analysis RSZ result of RS_RSZ2_RSZ2_RSZ2_RSZ1_RSZ1_RSZ1_RSZ2
Status: Status.UNKNOWN
Output dim: 1, lower bound: -0.0007131, upper bound: 0.0007131
time: 0.72 seconds

## BFS RS instance: RS_RSZ2_RSZ2_RSZ2_RSZ1_RSZ1_RSZ2

### Backsubstitution after applying RS history:
0: 0.0007184, 0.0011172, 0.0007184, 0.0011172, -0.0003988, 0.0003988
1: 0.9934075, 0.9943579, 0.9934075, 0.9943579, -0.0009505, 0.0009505
2: -0.0087946, -0.0055215, -0.0087946, -0.0055215, -0.0029775, 0.0029804
3: 0.0035868, 0.0041619, 0.0035868, 0.0041619, -0.0005751, 0.0005751
4: 0.0027809, 0.0053678, 0.0027809, 0.0053678, -0.0025869, 0.0025869
5: 0.0050646, 0.0064029, 0.0050646, 0.0064029, -0.0013382, 0.0013382
6: -0.0021738, -0.0008562, -0.0021738, -0.0008562, -0.0013176, 0.0013176
7: -0.0082275, -0.0074685, -0.0082275, -0.0074685, -0.0007590, 0.0007590
8: 0.0055062, 0.0098066, 0.0055062, 0.0098066, -0.0042209, 0.0042202
9: -0.0036825, -0.0031002, -0.0036825, -0.0031002, -0.0005823, 0.0005823

### Unstable ReLU Count (Linear/Conv2D Layers)
- layer_idx=0, type=LayerType.Linear, total=256, inp1_unstable=9, inp2_unstable=9, delta_unstable=250
- layer_idx=2, type=LayerType.Linear, total=256, inp1_unstable=6, inp2_unstable=6, delta_unstable=256
- layer_idx=4, type=LayerType.Linear, total=256, inp1_unstable=0, inp2_unstable=0, delta_unstable=256
- layer_idx=6, type=LayerType.Linear, total=256, inp1_unstable=2, inp2_unstable=2, delta_unstable=256
- layer_idx=8, type=LayerType.Linear, total=10, inp1_unstable=0, inp2_unstable=0, delta_unstable=10

Time for backsubstitution: 1.83 seconds

### RS candidates at layer 1
type: RSZ, layer: 1, pos: 204
type: RSZ, layer: 1, pos: 83
type: RSZ, layer: 1, pos: 136

Time for candidate selection: 0.00 seconds

### Candidate
type: RSZ, layer: 1, pos: 204

### Relational analysis RSZ of RS_RSZ2_RSZ2_RSZ2_RSZ1_RSZ1_RSZ2_RSZ1

#### Relational analysis RSZ result of RS_RSZ2_RSZ2_RSZ2_RSZ1_RSZ1_RSZ2_RSZ1
Status: Status.UNKNOWN
Output dim: 1, lower bound: -0.0007131, upper bound: 0.0007131
time: 0.61 seconds

### Relational analysis RSZ of RS_RSZ2_RSZ2_RSZ2_RSZ1_RSZ1_RSZ2_RSZ2

#### Relational analysis RSZ result of RS_RSZ2_RSZ2_RSZ2_RSZ1_RSZ1_RSZ2_RSZ2
Status: Status.UNKNOWN
Output dim: 1, lower bound: -0.0007131, upper bound: 0.0007131
time: 0.64 seconds

## BFS RS instance: RS_RSZ2_RSZ2_RSZ2_RSZ1_RSZ2_RSZ1

### Backsubstitution after applying RS history:
0: 0.0007184, 0.0011172, 0.0007184, 0.0011172, -0.0003988, 0.0003988
1: 0.9934075, 0.9943579, 0.9934075, 0.9943579, -0.0009505, 0.0009505
2: -0.0087946, -0.0055215, -0.0087946, -0.0055215, -0.0029762, 0.0029789
3: 0.0035868, 0.0041619, 0.0035868, 0.0041619, -0.0005751, 0.0005751
4: 0.0027809, 0.0053678, 0.0027809, 0.0053678, -0.0025869, 0.0025869
5: 0.0050646, 0.0064029, 0.0050646, 0.0064029, -0.0013382, 0.0013382
6: -0.0021738, -0.0008562, -0.0021738, -0.0008562, -0.0013176, 0.0013176
7: -0.0082275, -0.0074685, -0.0082275, -0.0074685, -0.0007590, 0.0007590
8: 0.0055062, 0.0098066, 0.0055062, 0.0098066, -0.0042206, 0.0042198
9: -0.0036825, -0.0031002, -0.0036825, -0.0031002, -0.0005823, 0.0005823

### Unstable ReLU Count (Linear/Conv2D Layers)
- layer_idx=0, type=LayerType.Linear, total=256, inp1_unstable=9, inp2_unstable=9, delta_unstable=250
- layer_idx=2, type=LayerType.Linear, total=256, inp1_unstable=6, inp2_unstable=6, delta_unstable=256
- layer_idx=4, type=LayerType.Linear, total=256, inp1_unstable=0, inp2_unstable=0, delta_unstable=256
- layer_idx=6, type=LayerType.Linear, total=256, inp1_unstable=2, inp2_unstable=2, delta_unstable=256
- layer_idx=8, type=LayerType.Linear, total=10, inp1_unstable=0, inp2_unstable=0, delta_unstable=10

Time for backsubstitution: 1.77 seconds

### RS candidates at layer 1
type: RSZ, layer: 1, pos: 83
type: RSZ, layer: 1, pos: 204
type: RSZ, layer: 1, pos: 136

Time for candidate selection: 0.00 seconds

### Candidate
type: RSZ, layer: 1, pos: 83

### Relational analysis RSZ of RS_RSZ2_RSZ2_RSZ2_RSZ1_RSZ2_RSZ1_RSZ1

#### Relational analysis RSZ result of RS_RSZ2_RSZ2_RSZ2_RSZ1_RSZ2_RSZ1_RSZ1
Status: Status.UNKNOWN
Output dim: 1, lower bound: -0.0007098, upper bound: 0.0007098
time: 0.69 seconds

### Relational analysis RSZ of RS_RSZ2_RSZ2_RSZ2_RSZ1_RSZ2_RSZ1_RSZ2

#### Relational analysis RSZ result of RS_RSZ2_RSZ2_RSZ2_RSZ1_RSZ2_RSZ1_RSZ2
Status: Status.UNKNOWN
Output dim: 1, lower bound: -0.0007098, upper bound: 0.0007098
time: 0.60 seconds

## BFS RS instance: RS_RSZ2_RSZ2_RSZ2_RSZ1_RSZ2_RSZ2

### Backsubstitution after applying RS history:
0: 0.0007184, 0.0011172, 0.0007184, 0.0011172, -0.0003988, 0.0003988
1: 0.9934075, 0.9943579, 0.9934075, 0.9943579, -0.0009505, 0.0009505
2: -0.0087946, -0.0055215, -0.0087946, -0.0055215, -0.0029769, 0.0029810
3: 0.0035868, 0.0041619, 0.0035868, 0.0041619, -0.0005751, 0.0005751
4: 0.0027809, 0.0053678, 0.0027809, 0.0053678, -0.0025869, 0.0025869
5: 0.0050646, 0.0064029, 0.0050646, 0.0064029, -0.0013382, 0.0013382
6: -0.0021738, -0.0008562, -0.0021738, -0.0008562, -0.0013176, 0.0013176
7: -0.0082275, -0.0074685, -0.0082275, -0.0074685, -0.0007590, 0.0007590
8: 0.0055062, 0.0098066, 0.0055062, 0.0098066, -0.0042211, 0.0042200
9: -0.0036825, -0.0031002, -0.0036825, -0.0031002, -0.0005823, 0.0005823

### Unstable ReLU Count (Linear/Conv2D Layers)
- layer_idx=0, type=LayerType.Linear, total=256, inp1_unstable=9, inp2_unstable=9, delta_unstable=250
- layer_idx=2, type=LayerType.Linear, total=256, inp1_unstable=6, inp2_unstable=6, delta_unstable=256
- layer_idx=4, type=LayerType.Linear, total=256, inp1_unstable=0, inp2_unstable=0, delta_unstable=256
- layer_idx=6, type=LayerType.Linear, total=256, inp1_unstable=2, inp2_unstable=2, delta_unstable=256
- layer_idx=8, type=LayerType.Linear, total=10, inp1_unstable=0, inp2_unstable=0, delta_unstable=10

Time for backsubstitution: 1.82 seconds

### RS candidates at layer 1
type: RSZ, layer: 1, pos: 83
type: RSZ, layer: 1, pos: 204
type: RSZ, layer: 1, pos: 136

Time for candidate selection: 0.00 seconds

### Candidate
type: RSZ, layer: 1, pos: 83

### Relational analysis RSZ of RS_RSZ2_RSZ2_RSZ2_RSZ1_RSZ2_RSZ2_RSZ1

#### Relational analysis RSZ result of RS_RSZ2_RSZ2_RSZ2_RSZ1_RSZ2_RSZ2_RSZ1
Status: Status.UNKNOWN
Output dim: 1, lower bound: -0.0007098, upper bound: 0.0007098
time: 0.68 seconds

### Relational analysis RSZ of RS_RSZ2_RSZ2_RSZ2_RSZ1_RSZ2_RSZ2_RSZ2

#### Relational analysis RSZ result of RS_RSZ2_RSZ2_RSZ2_RSZ1_RSZ2_RSZ2_RSZ2
Status: Status.UNKNOWN
Output dim: 1, lower bound: -0.0007098, upper bound: 0.0007098
time: 0.60 seconds

## BFS RS instance: RS_RSZ2_RSZ2_RSZ2_RSZ2_RSZ1_RSZ1

### Backsubstitution after applying RS history:
0: 0.0007184, 0.0011172, 0.0007184, 0.0011172, -0.0003988, 0.0003988
1: 0.9934075, 0.9943579, 0.9934075, 0.9943579, -0.0009505, 0.0009505
2: -0.0087946, -0.0055215, -0.0087946, -0.0055215, -0.0029806, 0.0029723
3: 0.0035868, 0.0041619, 0.0035868, 0.0041619, -0.0005751, 0.0005751
4: 0.0027809, 0.0053678, 0.0027809, 0.0053678, -0.0025869, 0.0025869
5: 0.0050646, 0.0064029, 0.0050646, 0.0064029, -0.0013382, 0.0013382
6: -0.0021738, -0.0008562, -0.0021738, -0.0008562, -0.0013176, 0.0013176
7: -0.0082275, -0.0074685, -0.0082275, -0.0074685, -0.0007590, 0.0007590
8: 0.0055062, 0.0098066, 0.0055062, 0.0098066, -0.0042192, 0.0042212
9: -0.0036825, -0.0031002, -0.0036825, -0.0031002, -0.0005823, 0.0005823

### Unstable ReLU Count (Linear/Conv2D Layers)
- layer_idx=0, type=LayerType.Linear, total=256, inp1_unstable=9, inp2_unstable=9, delta_unstable=250
- layer_idx=2, type=LayerType.Linear, total=256, inp1_unstable=6, inp2_unstable=6, delta_unstable=256
- layer_idx=4, type=LayerType.Linear, total=256, inp1_unstable=0, inp2_unstable=0, delta_unstable=256
- layer_idx=6, type=LayerType.Linear, total=256, inp1_unstable=2, inp2_unstable=2, delta_unstable=256
- layer_idx=8, type=LayerType.Linear, total=10, inp1_unstable=0, inp2_unstable=0, delta_unstable=10

Time for backsubstitution: 1.76 seconds

### RS candidates at layer 1
type: RSZ, layer: 1, pos: 199
type: RSZ, layer: 1, pos: 83
type: RSZ, layer: 1, pos: 176

Time for candidate selection: 0.00 seconds

### Candidate
type: RSZ, layer: 1, pos: 199

### Relational analysis RSZ of RS_RSZ2_RSZ2_RSZ2_RSZ2_RSZ1_RSZ1_RSZ1

#### Relational analysis RSZ result of RS_RSZ2_RSZ2_RSZ2_RSZ2_RSZ1_RSZ1_RSZ1
Status: Status.UNKNOWN
Output dim: 1, lower bound: -0.0007198, upper bound: 0.0007197
time: 0.64 seconds

### Relational analysis RSZ of RS_RSZ2_RSZ2_RSZ2_RSZ2_RSZ1_RSZ1_RSZ2

#### Relational analysis RSZ result of RS_RSZ2_RSZ2_RSZ2_RSZ2_RSZ1_RSZ1_RSZ2
Status: Status.UNKNOWN
Output dim: 1, lower bound: -0.0007198, upper bound: 0.0007197
time: 0.62 seconds

## BFS RS instance: RS_RSZ2_RSZ2_RSZ2_RSZ2_RSZ1_RSZ2

### Backsubstitution after applying RS history:
0: 0.0007184, 0.0011172, 0.0007184, 0.0011172, -0.0003988, 0.0003988
1: 0.9934075, 0.9943579, 0.9934075, 0.9943579, -0.0009505, 0.0009505
2: -0.0087946, -0.0055215, -0.0087946, -0.0055215, -0.0029801, 0.0029729
3: 0.0035868, 0.0041619, 0.0035868, 0.0041619, -0.0005751, 0.0005751
4: 0.0027809, 0.0053678, 0.0027809, 0.0053678, -0.0025869, 0.0025869
5: 0.0050646, 0.0064029, 0.0050646, 0.0064029, -0.0013382, 0.0013382
6: -0.0021738, -0.0008562, -0.0021738, -0.0008562, -0.0013176, 0.0013176
7: -0.0082275, -0.0074685, -0.0082275, -0.0074685, -0.0007590, 0.0007590
8: 0.0055062, 0.0098066, 0.0055062, 0.0098066, -0.0042194, 0.0042210
9: -0.0036825, -0.0031002, -0.0036825, -0.0031002, -0.0005823, 0.0005823

### Unstable ReLU Count (Linear/Conv2D Layers)
- layer_idx=0, type=LayerType.Linear, total=256, inp1_unstable=9, inp2_unstable=9, delta_unstable=250
- layer_idx=2, type=LayerType.Linear, total=256, inp1_unstable=6, inp2_unstable=6, delta_unstable=256
- layer_idx=4, type=LayerType.Linear, total=256, inp1_unstable=0, inp2_unstable=0, delta_unstable=256
- layer_idx=6, type=LayerType.Linear, total=256, inp1_unstable=2, inp2_unstable=2, delta_unstable=256
- layer_idx=8, type=LayerType.Linear, total=10, inp1_unstable=0, inp2_unstable=0, delta_unstable=10

Time for backsubstitution: 1.72 seconds

### RS candidates at layer 1
type: RSZ, layer: 1, pos: 199
type: RSZ, layer: 1, pos: 176
type: RSZ, layer: 1, pos: 83

Time for candidate selection: 0.00 seconds

### Candidate
type: RSZ, layer: 1, pos: 199

### Relational analysis RSZ of RS_RSZ2_RSZ2_RSZ2_RSZ2_RSZ1_RSZ2_RSZ1

#### Relational analysis RSZ result of RS_RSZ2_RSZ2_RSZ2_RSZ2_RSZ1_RSZ2_RSZ1
Status: Status.UNKNOWN
Output dim: 1, lower bound: -0.0007198, upper bound: 0.0007197
time: 0.62 seconds

### Relational analysis RSZ of RS_RSZ2_RSZ2_RSZ2_RSZ2_RSZ1_RSZ2_RSZ2

#### Relational analysis RSZ result of RS_RSZ2_RSZ2_RSZ2_RSZ2_RSZ1_RSZ2_RSZ2
Status: Status.UNKNOWN
Output dim: 1, lower bound: -0.0007198, upper bound: 0.0007197
time: 0.62 seconds

## BFS RS instance: RS_RSZ2_RSZ2_RSZ2_RSZ2_RSZ2_RSZ1

### Backsubstitution after applying RS history:
0: 0.0007184, 0.0011172, 0.0007184, 0.0011172, -0.0003988, 0.0003988
1: 0.9934075, 0.9943579, 0.9934075, 0.9943579, -0.0009505, 0.0009505
2: -0.0087946, -0.0055215, -0.0087946, -0.0055215, -0.0029829, 0.0029701
3: 0.0035868, 0.0041619, 0.0035868, 0.0041619, -0.0005751, 0.0005751
4: 0.0027809, 0.0053678, 0.0027809, 0.0053678, -0.0025869, 0.0025869
5: 0.0050646, 0.0064029, 0.0050646, 0.0064029, -0.0013382, 0.0013382
6: -0.0021738, -0.0008562, -0.0021738, -0.0008562, -0.0013176, 0.0013176
7: -0.0082275, -0.0074685, -0.0082275, -0.0074685, -0.0007590, 0.0007590
8: 0.0055062, 0.0098066, 0.0055062, 0.0098066, -0.0042186, 0.0042218
9: -0.0036825, -0.0031002, -0.0036825, -0.0031002, -0.0005823, 0.0005823

### Unstable ReLU Count (Linear/Conv2D Layers)
- layer_idx=0, type=LayerType.Linear, total=256, inp1_unstable=9, inp2_unstable=9, delta_unstable=250
- layer_idx=2, type=LayerType.Linear, total=256, inp1_unstable=6, inp2_unstable=6, delta_unstable=256
- layer_idx=4, type=LayerType.Linear, total=256, inp1_unstable=0, inp2_unstable=0, delta_unstable=256
- layer_idx=6, type=LayerType.Linear, total=256, inp1_unstable=2, inp2_unstable=2, delta_unstable=256
- layer_idx=8, type=LayerType.Linear, total=10, inp1_unstable=0, inp2_unstable=0, delta_unstable=10

Time for backsubstitution: 1.75 seconds

### RS candidates at layer 1
type: RSZ, layer: 1, pos: 199
type: RSZ, layer: 1, pos: 83
type: RSZ, layer: 1, pos: 176

Time for candidate selection: 0.00 seconds

### Candidate
type: RSZ, layer: 1, pos: 199

### Relational analysis RSZ of RS_RSZ2_RSZ2_RSZ2_RSZ2_RSZ2_RSZ1_RSZ1

#### Relational analysis RSZ result of RS_RSZ2_RSZ2_RSZ2_RSZ2_RSZ2_RSZ1_RSZ1
Status: Status.UNKNOWN
Output dim: 1, lower bound: -0.0007197, upper bound: 0.0007198
time: 0.65 seconds

### Relational analysis RSZ of RS_RSZ2_RSZ2_RSZ2_RSZ2_RSZ2_RSZ1_RSZ2

#### Relational analysis RSZ result of RS_RSZ2_RSZ2_RSZ2_RSZ2_RSZ2_RSZ1_RSZ2
Status: Status.UNKNOWN
Output dim: 1, lower bound: -0.0007197, upper bound: 0.0007198
time: 0.66 seconds

## BFS RS instance: RS_RSZ2_RSZ2_RSZ2_RSZ2_RSZ2_RSZ2

### Backsubstitution after applying RS history:
0: 0.0007184, 0.0011172, 0.0007184, 0.0011172, -0.0003988, 0.0003988
1: 0.9934075, 0.9943579, 0.9934075, 0.9943579, -0.0009505, 0.0009505
2: -0.0087946, -0.0055215, -0.0087946, -0.0055215, -0.0029823, 0.0029707
3: 0.0035868, 0.0041619, 0.0035868, 0.0041619, -0.0005751, 0.0005751
4: 0.0027809, 0.0053678, 0.0027809, 0.0053678, -0.0025869, 0.0025869
5: 0.0050646, 0.0064029, 0.0050646, 0.0064029, -0.0013382, 0.0013382
6: -0.0021738, -0.0008562, -0.0021738, -0.0008562, -0.0013176, 0.0013176
7: -0.0082275, -0.0074685, -0.0082275, -0.0074685, -0.0007590, 0.0007590
8: 0.0055062, 0.0098066, 0.0055062, 0.0098066, -0.0042188, 0.0042216
9: -0.0036825, -0.0031002, -0.0036825, -0.0031002, -0.0005823, 0.0005823

### Unstable ReLU Count (Linear/Conv2D Layers)
- layer_idx=0, type=LayerType.Linear, total=256, inp1_unstable=9, inp2_unstable=9, delta_unstable=250
- layer_idx=2, type=LayerType.Linear, total=256, inp1_unstable=6, inp2_unstable=6, delta_unstable=256
- layer_idx=4, type=LayerType.Linear, total=256, inp1_unstable=0, inp2_unstable=0, delta_unstable=256
- layer_idx=6, type=LayerType.Linear, total=256, inp1_unstable=2, inp2_unstable=2, delta_unstable=256
- layer_idx=8, type=LayerType.Linear, total=10, inp1_unstable=0, inp2_unstable=0, delta_unstable=10

Time for backsubstitution: 1.75 seconds

### RS candidates at layer 1
type: RSZ, layer: 1, pos: 83
type: RSZ, layer: 1, pos: 176
type: RSZ, layer: 1, pos: 199

Time for candidate selection: 0.00 seconds

### Candidate
type: RSZ, layer: 1, pos: 83

### Relational analysis RSZ of RS_RSZ2_RSZ2_RSZ2_RSZ2_RSZ2_RSZ2_RSZ1

#### Relational analysis RSZ result of RS_RSZ2_RSZ2_RSZ2_RSZ2_RSZ2_RSZ2_RSZ1
Status: Status.UNKNOWN
Output dim: 1, lower bound: -0.0007772, upper bound: 0.0007743
time: 0.74 seconds

### Relational analysis RSZ of RS_RSZ2_RSZ2_RSZ2_RSZ2_RSZ2_RSZ2_RSZ2

#### Relational analysis RSZ result of RS_RSZ2_RSZ2_RSZ2_RSZ2_RSZ2_RSZ2_RSZ2
Status: Status.UNKNOWN
Output dim: 1, lower bound: -0.0007771, upper bound: 0.0007745
time: 0.72 seconds

## Summary of splitting (split count: 6)
- Time for RS candidates: 3.22 seconds
RS_RSZ1_RSZ1_RSZ1_RSZ1_RSZ1_RSZ1_RSZ1, status: Status.UNKNOWN, split count: 7, time: 3.22
Output dim: 1, lower bound: -0.0007554, upper bound: 0.0007548
RS_RSZ1_RSZ1_RSZ1_RSZ1_RSZ1_RSZ1_RSZ2, status: Status.UNKNOWN, split count: 7, time: 3.22
Output dim: 1, lower bound: -0.0007554, upper bound: 0.0007548
RS_RSZ1_RSZ1_RSZ1_RSZ1_RSZ1_RSZ2_RSZ1, status: Status.UNKNOWN, split count: 7, time: 3.22
Output dim: 1, lower bound: -0.0007568, upper bound: 0.0007572
RS_RSZ1_RSZ1_RSZ1_RSZ1_RSZ1_RSZ2_RSZ2, status: Status.UNKNOWN, split count: 7, time: 3.22
Output dim: 1, lower bound: -0.0007578, upper bound: 0.0007566
RS_RSZ1_RSZ1_RSZ1_RSZ1_RSZ2_RSZ1_RSZ1, status: Status.UNKNOWN, split count: 7, time: 3.22
Output dim: 1, lower bound: -0.0007708, upper bound: 0.0007733
RS_RSZ1_RSZ1_RSZ1_RSZ1_RSZ2_RSZ1_RSZ2, status: Status.UNKNOWN, split count: 7, time: 3.22
Output dim: 1, lower bound: -0.0007727, upper bound: 0.0007718
RS_RSZ1_RSZ1_RSZ1_RSZ1_RSZ2_RSZ2_RSZ1, status: Status.UNKNOWN, split count: 7, time: 3.22
Output dim: 1, lower bound: -0.0007131, upper bound: 0.0007131
RS_RSZ1_RSZ1_RSZ1_RSZ1_RSZ2_RSZ2_RSZ2, status: Status.UNKNOWN, split count: 7, time: 3.22
Output dim: 1, lower bound: -0.0007131, upper bound: 0.0007131
RS_RSZ1_RSZ1_RSZ1_RSZ2_RSZ1_RSZ1_RSZ1, status: Status.UNKNOWN, split count: 7, time: 3.22
Output dim: 1, lower bound: -0.0007703, upper bound: 0.0007721
RS_RSZ1_RSZ1_RSZ1_RSZ2_RSZ1_RSZ1_RSZ2, status: Status.UNKNOWN, split count: 7, time: 3.22
Output dim: 1, lower bound: -0.0007703, upper bound: 0.0007721
RS_RSZ1_RSZ1_RSZ1_RSZ2_RSZ1_RSZ2_RSZ1, status: Status.UNKNOWN, split count: 7, time: 3.22
Output dim: 1, lower bound: -0.0007912, upper bound: 0.0007873
RS_RSZ1_RSZ1_RSZ1_RSZ2_RSZ1_RSZ2_RSZ2, status: Status.UNKNOWN, split count: 7, time: 3.22
Output dim: 1, lower bound: -0.0007889, upper bound: 0.0007897
RS_RSZ1_RSZ1_RSZ1_RSZ2_RSZ2_RSZ1_RSZ1, status: Status.UNKNOWN, split count: 7, time: 3.22
Output dim: 1, lower bound: -0.0007721, upper bound: 0.0007720
RS_RSZ1_RSZ1_RSZ1_RSZ2_RSZ2_RSZ1_RSZ2, status: Status.UNKNOWN, split count: 7, time: 3.22
Output dim: 1, lower bound: -0.0007720, upper bound: 0.0007722
RS_RSZ1_RSZ1_RSZ1_RSZ2_RSZ2_RSZ2_RSZ1, status: Status.UNKNOWN, split count: 7, time: 3.22
Output dim: 1, lower bound: -0.0007098, upper bound: 0.0007098
RS_RSZ1_RSZ1_RSZ1_RSZ2_RSZ2_RSZ2_RSZ2, status: Status.UNKNOWN, split count: 7, time: 3.22
Output dim: 1, lower bound: -0.0007098, upper bound: 0.0007098
RS_RSZ1_RSZ1_RSZ2_RSZ1_RSZ1_RSZ1_RSZ1, status: Status.UNKNOWN, split count: 7, time: 3.22
Output dim: 1, lower bound: -0.0007734, upper bound: 0.0007737
RS_RSZ1_RSZ1_RSZ2_RSZ1_RSZ1_RSZ1_RSZ2, status: Status.UNKNOWN, split count: 7, time: 3.22
Output dim: 1, lower bound: -0.0007738, upper bound: 0.0007730
RS_RSZ1_RSZ1_RSZ2_RSZ1_RSZ1_RSZ2_RSZ1, status: Status.UNKNOWN, split count: 7, time: 3.22
Output dim: 1, lower bound: -0.0007734, upper bound: 0.0007737
RS_RSZ1_RSZ1_RSZ2_RSZ1_RSZ1_RSZ2_RSZ2, status: Status.UNKNOWN, split count: 7, time: 3.22
Output dim: 1, lower bound: -0.0007737, upper bound: 0.0007731
RS_RSZ1_RSZ1_RSZ2_RSZ1_RSZ2_RSZ1_RSZ1, status: Status.UNKNOWN, split count: 7, time: 3.22
Output dim: 1, lower bound: -0.0007098, upper bound: 0.0007098
RS_RSZ1_RSZ1_RSZ2_RSZ1_RSZ2_RSZ1_RSZ2, status: Status.UNKNOWN, split count: 7, time: 3.22
Output dim: 1, lower bound: -0.0007098, upper bound: 0.0007098
RS_RSZ1_RSZ1_RSZ2_RSZ1_RSZ2_RSZ2_RSZ1, status: Status.UNKNOWN, split count: 7, time: 3.22
Output dim: 1, lower bound: -0.0007131, upper bound: 0.0007131
RS_RSZ1_RSZ1_RSZ2_RSZ1_RSZ2_RSZ2_RSZ2, status: Status.UNKNOWN, split count: 7, time: 3.22
Output dim: 1, lower bound: -0.0007131, upper bound: 0.0007131
RS_RSZ1_RSZ1_RSZ2_RSZ2_RSZ1_RSZ1_RSZ1, status: Status.UNKNOWN, split count: 7, time: 3.22
Output dim: 1, lower bound: -0.0007131, upper bound: 0.0007131
RS_RSZ1_RSZ1_RSZ2_RSZ2_RSZ1_RSZ1_RSZ2, status: Status.UNKNOWN, split count: 7, time: 3.22
Output dim: 1, lower bound: -0.0007131, upper bound: 0.0007131
RS_RSZ1_RSZ1_RSZ2_RSZ2_RSZ1_RSZ2_RSZ1, status: Status.UNKNOWN, split count: 7, time: 3.22
Output dim: 1, lower bound: -0.0007711, upper bound: 0.0007718
RS_RSZ1_RSZ1_RSZ2_RSZ2_RSZ1_RSZ2_RSZ2, status: Status.UNKNOWN, split count: 7, time: 3.22
Output dim: 1, lower bound: -0.0007707, upper bound: 0.0007718
RS_RSZ1_RSZ1_RSZ2_RSZ2_RSZ2_RSZ1_RSZ1, status: Status.UNKNOWN, split count: 7, time: 3.22
Output dim: 1, lower bound: -0.0007738, upper bound: 0.0007730
RS_RSZ1_RSZ1_RSZ2_RSZ2_RSZ2_RSZ1_RSZ2, status: Status.UNKNOWN, split count: 7, time: 3.22
Output dim: 1, lower bound: -0.0007737, upper bound: 0.0007731
RS_RSZ1_RSZ1_RSZ2_RSZ2_RSZ2_RSZ2_RSZ1, status: Status.UNKNOWN, split count: 7, time: 3.22
Output dim: 1, lower bound: -0.0007131, upper bound: 0.0007131
RS_RSZ1_RSZ1_RSZ2_RSZ2_RSZ2_RSZ2_RSZ2, status: Status.UNKNOWN, split count: 7, time: 3.22
Output dim: 1, lower bound: -0.0007131, upper bound: 0.0007131
RS_RSZ1_RSZ2_RSZ1_RSZ1_RSZ1_RSZ1_RSZ1, status: Status.UNKNOWN, split count: 7, time: 3.22
Output dim: 1, lower bound: -0.0007740, upper bound: 0.0007753
RS_RSZ1_RSZ2_RSZ1_RSZ1_RSZ1_RSZ1_RSZ2, status: Status.UNKNOWN, split count: 7, time: 3.22
Output dim: 1, lower bound: -0.0007755, upper bound: 0.0007734
RS_RSZ1_RSZ2_RSZ1_RSZ1_RSZ1_RSZ2_RSZ1, status: Status.UNKNOWN, split count: 7, time: 3.22
Output dim: 1, lower bound: -0.0007098, upper bound: 0.0007098
RS_RSZ1_RSZ2_RSZ1_RSZ1_RSZ1_RSZ2_RSZ2, status: Status.UNKNOWN, split count: 7, time: 3.22
Output dim: 1, lower bound: -0.0007098, upper bound: 0.0007098
RS_RSZ1_RSZ2_RSZ1_RSZ1_RSZ2_RSZ1_RSZ1, status: Status.UNKNOWN, split count: 7, time: 3.22
Output dim: 1, lower bound: -0.0007707, upper bound: 0.0007731
RS_RSZ1_RSZ2_RSZ1_RSZ1_RSZ2_RSZ1_RSZ2, status: Status.UNKNOWN, split count: 7, time: 3.22
Output dim: 1, lower bound: -0.0007727, upper bound: 0.0007718
RS_RSZ1_RSZ2_RSZ1_RSZ1_RSZ2_RSZ2_RSZ1, status: Status.UNKNOWN, split count: 7, time: 3.22
Output dim: 1, lower bound: -0.0007131, upper bound: 0.0007131
RS_RSZ1_RSZ2_RSZ1_RSZ1_RSZ2_RSZ2_RSZ2, status: Status.UNKNOWN, split count: 7, time: 3.22
Output dim: 1, lower bound: -0.0007131, upper bound: 0.0007131
RS_RSZ1_RSZ2_RSZ1_RSZ2_RSZ1_RSZ1_RSZ1, status: Status.UNKNOWN, split count: 7, time: 3.22
Output dim: 1, lower bound: -0.0007706, upper bound: 0.0007730
RS_RSZ1_RSZ2_RSZ1_RSZ2_RSZ1_RSZ1_RSZ2, status: Status.UNKNOWN, split count: 7, time: 3.22
Output dim: 1, lower bound: -0.0007706, upper bound: 0.0007731
RS_RSZ1_RSZ2_RSZ1_RSZ2_RSZ1_RSZ2_RSZ1, status: Status.UNKNOWN, split count: 7, time: 3.22
Output dim: 1, lower bound: -0.0007718, upper bound: 0.0007706
RS_RSZ1_RSZ2_RSZ1_RSZ2_RSZ1_RSZ2_RSZ2, status: Status.UNKNOWN, split count: 7, time: 3.22
Output dim: 1, lower bound: -0.0007718, upper bound: 0.0007709
RS_RSZ1_RSZ2_RSZ1_RSZ2_RSZ2_RSZ1_RSZ1, status: Status.UNKNOWN, split count: 7, time: 3.22
Output dim: 1, lower bound: -0.0007131, upper bound: 0.0007131
RS_RSZ1_RSZ2_RSZ1_RSZ2_RSZ2_RSZ1_RSZ2, status: Status.UNKNOWN, split count: 7, time: 3.22
Output dim: 1, lower bound: -0.0007131, upper bound: 0.0007131
RS_RSZ1_RSZ2_RSZ1_RSZ2_RSZ2_RSZ2_RSZ1, status: Status.UNKNOWN, split count: 7, time: 3.22
Output dim: 1, lower bound: -0.0007131, upper bound: 0.0007131
RS_RSZ1_RSZ2_RSZ1_RSZ2_RSZ2_RSZ2_RSZ2, status: Status.UNKNOWN, split count: 7, time: 3.22
Output dim: 1, lower bound: -0.0007131, upper bound: 0.0007131
RS_RSZ1_RSZ2_RSZ2_RSZ1_RSZ1_RSZ1_RSZ1, status: Status.UNKNOWN, split count: 7, time: 3.22
Output dim: 1, lower bound: -0.0007550, upper bound: 0.0007554
RS_RSZ1_RSZ2_RSZ2_RSZ1_RSZ1_RSZ1_RSZ2, status: Status.UNKNOWN, split count: 7, time: 3.22
Output dim: 1, lower bound: -0.0007546, upper bound: 0.0007556
RS_RSZ1_RSZ2_RSZ2_RSZ1_RSZ1_RSZ2_RSZ1, status: Status.UNKNOWN, split count: 7, time: 3.22
Output dim: 1, lower bound: -0.0007098, upper bound: 0.0007098
RS_RSZ1_RSZ2_RSZ2_RSZ1_RSZ1_RSZ2_RSZ2, status: Status.UNKNOWN, split count: 7, time: 3.22
Output dim: 1, lower bound: -0.0007098, upper bound: 0.0007098
RS_RSZ1_RSZ2_RSZ2_RSZ1_RSZ2_RSZ1_RSZ1, status: Status.UNKNOWN, split count: 7, time: 3.22
Output dim: 1, lower bound: -0.0007098, upper bound: 0.0007098
RS_RSZ1_RSZ2_RSZ2_RSZ1_RSZ2_RSZ1_RSZ2, status: Status.UNKNOWN, split count: 7, time: 3.22
Output dim: 1, lower bound: -0.0007098, upper bound: 0.0007098
RS_RSZ1_RSZ2_RSZ2_RSZ1_RSZ2_RSZ2_RSZ1, status: Status.UNKNOWN, split count: 7, time: 3.22
Output dim: 1, lower bound: -0.0007550, upper bound: 0.0007549
RS_RSZ1_RSZ2_RSZ2_RSZ1_RSZ2_RSZ2_RSZ2, status: Status.UNKNOWN, split count: 7, time: 3.22
Output dim: 1, lower bound: -0.0007552, upper bound: 0.0007547
RS_RSZ1_RSZ2_RSZ2_RSZ2_RSZ1_RSZ1_RSZ1, status: Status.UNKNOWN, split count: 7, time: 3.22
Output dim: 1, lower bound: -0.0007131, upper bound: 0.0007131
RS_RSZ1_RSZ2_RSZ2_RSZ2_RSZ1_RSZ1_RSZ2, status: Status.UNKNOWN, split count: 7, time: 3.22
Output dim: 1, lower bound: -0.0007131, upper bound: 0.0007131
RS_RSZ1_RSZ2_RSZ2_RSZ2_RSZ1_RSZ2_RSZ1, status: Status.UNKNOWN, split count: 7, time: 3.22
Output dim: 1, lower bound: -0.0007130, upper bound: 0.0007131
RS_RSZ1_RSZ2_RSZ2_RSZ2_RSZ1_RSZ2_RSZ2, status: Status.UNKNOWN, split count: 7, time: 3.22
Output dim: 1, lower bound: -0.0007131, upper bound: 0.0007131
RS_RSZ1_RSZ2_RSZ2_RSZ2_RSZ2_RSZ1_RSZ1, status: Status.UNKNOWN, split count: 7, time: 3.22
Output dim: 1, lower bound: -0.0007131, upper bound: 0.0007131
RS_RSZ1_RSZ2_RSZ2_RSZ2_RSZ2_RSZ1_RSZ2, status: Status.UNKNOWN, split count: 7, time: 3.22
Output dim: 1, lower bound: -0.0007131, upper bound: 0.0007131
RS_RSZ1_RSZ2_RSZ2_RSZ2_RSZ2_RSZ2_RSZ1, status: Status.UNKNOWN, split count: 7, time: 3.22
Output dim: 1, lower bound: -0.0007130, upper bound: 0.0007131
RS_RSZ1_RSZ2_RSZ2_RSZ2_RSZ2_RSZ2_RSZ2, status: Status.UNKNOWN, split count: 7, time: 3.22
Output dim: 1, lower bound: -0.0007130, upper bound: 0.0007131
RS_RSZ2_RSZ1_RSZ1_RSZ1_RSZ1_RSZ1_RSZ1, status: Status.UNKNOWN, split count: 7, time: 3.22
Output dim: 1, lower bound: -0.0007733, upper bound: 0.0007737
RS_RSZ2_RSZ1_RSZ1_RSZ1_RSZ1_RSZ1_RSZ2, status: Status.UNKNOWN, split count: 7, time: 3.22
Output dim: 1, lower bound: -0.0007733, upper bound: 0.0007737
RS_RSZ2_RSZ1_RSZ1_RSZ1_RSZ1_RSZ2_RSZ1, status: Status.UNKNOWN, split count: 7, time: 3.22
Output dim: 1, lower bound: -0.0007867, upper bound: 0.0007888
RS_RSZ2_RSZ1_RSZ1_RSZ1_RSZ1_RSZ2_RSZ2, status: Status.UNKNOWN, split count: 7, time: 3.22
Output dim: 1, lower bound: -0.0007866, upper bound: 0.0007888
RS_RSZ2_RSZ1_RSZ1_RSZ1_RSZ2_RSZ1_RSZ1, status: Status.UNKNOWN, split count: 7, time: 3.22
Output dim: 1, lower bound: -0.0007556, upper bound: 0.0007565
RS_RSZ2_RSZ1_RSZ1_RSZ1_RSZ2_RSZ1_RSZ2, status: Status.UNKNOWN, split count: 7, time: 3.22
Output dim: 1, lower bound: -0.0007556, upper bound: 0.0007565
RS_RSZ2_RSZ1_RSZ1_RSZ1_RSZ2_RSZ2_RSZ1, status: Status.UNKNOWN, split count: 7, time: 3.22
Output dim: 1, lower bound: -0.0007731, upper bound: 0.0007738
RS_RSZ2_RSZ1_RSZ1_RSZ1_RSZ2_RSZ2_RSZ2, status: Status.UNKNOWN, split count: 7, time: 3.22
Output dim: 1, lower bound: -0.0007731, upper bound: 0.0007738
RS_RSZ2_RSZ1_RSZ1_RSZ2_RSZ1_RSZ1_RSZ1, status: Status.UNKNOWN, split count: 7, time: 3.22
Output dim: 1, lower bound: -0.0007732, upper bound: 0.0007737
RS_RSZ2_RSZ1_RSZ1_RSZ2_RSZ1_RSZ1_RSZ2, status: Status.UNKNOWN, split count: 7, time: 3.22
Output dim: 1, lower bound: -0.0007732, upper bound: 0.0007738
RS_RSZ2_RSZ1_RSZ1_RSZ2_RSZ1_RSZ2_RSZ1, status: Status.UNKNOWN, split count: 7, time: 3.22
Output dim: 1, lower bound: -0.0007701, upper bound: 0.0007720
RS_RSZ2_RSZ1_RSZ1_RSZ2_RSZ1_RSZ2_RSZ2, status: Status.UNKNOWN, split count: 7, time: 3.22
Output dim: 1, lower bound: -0.0007700, upper bound: 0.0007719
RS_RSZ2_RSZ1_RSZ1_RSZ2_RSZ2_RSZ1_RSZ1, status: Status.UNKNOWN, split count: 7, time: 3.22
Output dim: 1, lower bound: -0.0007732, upper bound: 0.0007737
RS_RSZ2_RSZ1_RSZ1_RSZ2_RSZ2_RSZ1_RSZ2, status: Status.UNKNOWN, split count: 7, time: 3.22
Output dim: 1, lower bound: -0.0007730, upper bound: 0.0007737
RS_RSZ2_RSZ1_RSZ1_RSZ2_RSZ2_RSZ2_RSZ1, status: Status.UNKNOWN, split count: 7, time: 3.22
Output dim: 1, lower bound: -0.0007165, upper bound: 0.0007165
RS_RSZ2_RSZ1_RSZ1_RSZ2_RSZ2_RSZ2_RSZ2, status: Status.UNKNOWN, split count: 7, time: 3.22
Output dim: 1, lower bound: -0.0007165, upper bound: 0.0007165
RS_RSZ2_RSZ1_RSZ2_RSZ1_RSZ1_RSZ1_RSZ1, status: Status.UNKNOWN, split count: 7, time: 3.22
Output dim: 1, lower bound: -0.0007549, upper bound: 0.0007550
RS_RSZ2_RSZ1_RSZ2_RSZ1_RSZ1_RSZ1_RSZ2, status: Status.UNKNOWN, split count: 7, time: 3.22
Output dim: 1, lower bound: -0.0007547, upper bound: 0.0007555
RS_RSZ2_RSZ1_RSZ2_RSZ1_RSZ1_RSZ2_RSZ1, status: Status.UNKNOWN, split count: 7, time: 3.22
Output dim: 1, lower bound: -0.0007098, upper bound: 0.0007098
RS_RSZ2_RSZ1_RSZ2_RSZ1_RSZ1_RSZ2_RSZ2, status: Status.UNKNOWN, split count: 7, time: 3.22
Output dim: 1, lower bound: -0.0007098, upper bound: 0.0007098
RS_RSZ2_RSZ1_RSZ2_RSZ1_RSZ2_RSZ1_RSZ1, status: Status.UNKNOWN, split count: 7, time: 3.22
Output dim: 1, lower bound: -0.0007131, upper bound: 0.0007131
RS_RSZ2_RSZ1_RSZ2_RSZ1_RSZ2_RSZ1_RSZ2, status: Status.UNKNOWN, split count: 7, time: 3.22
Output dim: 1, lower bound: -0.0007131, upper bound: 0.0007131
RS_RSZ2_RSZ1_RSZ2_RSZ1_RSZ2_RSZ2_RSZ1, status: Status.UNKNOWN, split count: 7, time: 3.22
Output dim: 1, lower bound: -0.0007131, upper bound: 0.0007131
RS_RSZ2_RSZ1_RSZ2_RSZ1_RSZ2_RSZ2_RSZ2, status: Status.UNKNOWN, split count: 7, time: 3.22
Output dim: 1, lower bound: -0.0007131, upper bound: 0.0007131
RS_RSZ2_RSZ1_RSZ2_RSZ2_RSZ1_RSZ1_RSZ1, status: Status.UNKNOWN, split count: 7, time: 3.22
Output dim: 1, lower bound: -0.0007900, upper bound: 0.0007888
RS_RSZ2_RSZ1_RSZ2_RSZ2_RSZ1_RSZ1_RSZ2, status: Status.UNKNOWN, split count: 7, time: 3.22
Output dim: 1, lower bound: -0.0007876, upper bound: 0.0007912
RS_RSZ2_RSZ1_RSZ2_RSZ2_RSZ1_RSZ2_RSZ1, status: Status.UNKNOWN, split count: 7, time: 3.22
Output dim: 1, lower bound: -0.0007899, upper bound: 0.0007887
RS_RSZ2_RSZ1_RSZ2_RSZ2_RSZ1_RSZ2_RSZ2, status: Status.UNKNOWN, split count: 7, time: 3.22
Output dim: 1, lower bound: -0.0007876, upper bound: 0.0007913
RS_RSZ2_RSZ1_RSZ2_RSZ2_RSZ2_RSZ1_RSZ1, status: Status.UNKNOWN, split count: 7, time: 3.22
Output dim: 1, lower bound: -0.0007707, upper bound: 0.0007719
RS_RSZ2_RSZ1_RSZ2_RSZ2_RSZ2_RSZ1_RSZ2, status: Status.UNKNOWN, split count: 7, time: 3.22
Output dim: 1, lower bound: -0.0007707, upper bound: 0.0007719
RS_RSZ2_RSZ1_RSZ2_RSZ2_RSZ2_RSZ2_RSZ1, status: Status.UNKNOWN, split count: 7, time: 3.22
Output dim: 1, lower bound: -0.0007546, upper bound: 0.0007554
RS_RSZ2_RSZ1_RSZ2_RSZ2_RSZ2_RSZ2_RSZ2, status: Status.UNKNOWN, split count: 7, time: 3.22
Output dim: 1, lower bound: -0.0007546, upper bound: 0.0007554
RS_RSZ2_RSZ2_RSZ1_RSZ1_RSZ1_RSZ1_RSZ1, status: Status.UNKNOWN, split count: 7, time: 3.22
Output dim: 1, lower bound: -0.0007903, upper bound: 0.0007873
RS_RSZ2_RSZ2_RSZ1_RSZ1_RSZ1_RSZ1_RSZ2, status: Status.UNKNOWN, split count: 7, time: 3.22
Output dim: 1, lower bound: -0.0007908, upper bound: 0.0007875
RS_RSZ2_RSZ2_RSZ1_RSZ1_RSZ1_RSZ2_RSZ1, status: Status.UNKNOWN, split count: 7, time: 3.22
Output dim: 1, lower bound: -0.0007617, upper bound: 0.0007603
RS_RSZ2_RSZ2_RSZ1_RSZ1_RSZ1_RSZ2_RSZ2, status: Status.UNKNOWN, split count: 7, time: 3.22
Output dim: 1, lower bound: -0.0007617, upper bound: 0.0007603
RS_RSZ2_RSZ2_RSZ1_RSZ1_RSZ2_RSZ1_RSZ1, status: Status.UNKNOWN, split count: 7, time: 3.22
Output dim: 1, lower bound: -0.0007771, upper bound: 0.0007754
RS_RSZ2_RSZ2_RSZ1_RSZ1_RSZ2_RSZ1_RSZ2, status: Status.UNKNOWN, split count: 7, time: 3.22
Output dim: 1, lower bound: -0.0007771, upper bound: 0.0007757
RS_RSZ2_RSZ2_RSZ1_RSZ1_RSZ2_RSZ2_RSZ1, status: Status.UNKNOWN, split count: 7, time: 3.22
Output dim: 1, lower bound: -0.0007730, upper bound: 0.0007719
RS_RSZ2_RSZ2_RSZ1_RSZ1_RSZ2_RSZ2_RSZ2, status: Status.UNKNOWN, split count: 7, time: 3.22
Output dim: 1, lower bound: -0.0007731, upper bound: 0.0007719
RS_RSZ2_RSZ2_RSZ1_RSZ2_RSZ1_RSZ1_RSZ1, status: Status.UNKNOWN, split count: 7, time: 3.22
Output dim: 1, lower bound: -0.0007915, upper bound: 0.0007862
RS_RSZ2_RSZ2_RSZ1_RSZ2_RSZ1_RSZ1_RSZ2, status: Status.UNKNOWN, split count: 7, time: 3.22
Output dim: 1, lower bound: -0.0007919, upper bound: 0.0007864
RS_RSZ2_RSZ2_RSZ1_RSZ2_RSZ1_RSZ2_RSZ1, status: Status.UNKNOWN, split count: 7, time: 3.22
Output dim: 1, lower bound: -0.0007774, upper bound: 0.0007744
RS_RSZ2_RSZ2_RSZ1_RSZ2_RSZ1_RSZ2_RSZ2, status: Status.UNKNOWN, split count: 7, time: 3.22
Output dim: 1, lower bound: -0.0007774, upper bound: 0.0007746
RS_RSZ2_RSZ2_RSZ1_RSZ2_RSZ2_RSZ1_RSZ1, status: Status.UNKNOWN, split count: 7, time: 3.22
Output dim: 1, lower bound: -0.0007913, upper bound: 0.0007862
RS_RSZ2_RSZ2_RSZ1_RSZ2_RSZ2_RSZ1_RSZ2, status: Status.UNKNOWN, split count: 7, time: 3.22
Output dim: 1, lower bound: -0.0007916, upper bound: 0.0007864
RS_RSZ2_RSZ2_RSZ1_RSZ2_RSZ2_RSZ2_RSZ1, status: Status.UNKNOWN, split count: 7, time: 3.22
Output dim: 1, lower bound: -0.0007615, upper bound: 0.0007599
RS_RSZ2_RSZ2_RSZ1_RSZ2_RSZ2_RSZ2_RSZ2, status: Status.UNKNOWN, split count: 7, time: 3.22
Output dim: 1, lower bound: -0.0007615, upper bound: 0.0007599
RS_RSZ2_RSZ2_RSZ2_RSZ1_RSZ1_RSZ1_RSZ1, status: Status.UNKNOWN, split count: 7, time: 3.22
Output dim: 1, lower bound: -0.0007131, upper bound: 0.0007131
RS_RSZ2_RSZ2_RSZ2_RSZ1_RSZ1_RSZ1_RSZ2, status: Status.UNKNOWN, split count: 7, time: 3.22
Output dim: 1, lower bound: -0.0007131, upper bound: 0.0007131
RS_RSZ2_RSZ2_RSZ2_RSZ1_RSZ1_RSZ2_RSZ1, status: Status.UNKNOWN, split count: 7, time: 3.22
Output dim: 1, lower bound: -0.0007131, upper bound: 0.0007131
RS_RSZ2_RSZ2_RSZ2_RSZ1_RSZ1_RSZ2_RSZ2, status: Status.UNKNOWN, split count: 7, time: 3.22
Output dim: 1, lower bound: -0.0007131, upper bound: 0.0007131
RS_RSZ2_RSZ2_RSZ2_RSZ1_RSZ2_RSZ1_RSZ1, status: Status.UNKNOWN, split count: 7, time: 3.22
Output dim: 1, lower bound: -0.0007098, upper bound: 0.0007098
RS_RSZ2_RSZ2_RSZ2_RSZ1_RSZ2_RSZ1_RSZ2, status: Status.UNKNOWN, split count: 7, time: 3.22
Output dim: 1, lower bound: -0.0007098, upper bound: 0.0007098
RS_RSZ2_RSZ2_RSZ2_RSZ1_RSZ2_RSZ2_RSZ1, status: Status.UNKNOWN, split count: 7, time: 3.22
Output dim: 1, lower bound: -0.0007098, upper bound: 0.0007098
RS_RSZ2_RSZ2_RSZ2_RSZ1_RSZ2_RSZ2_RSZ2, status: Status.UNKNOWN, split count: 7, time: 3.22
Output dim: 1, lower bound: -0.0007098, upper bound: 0.0007098
RS_RSZ2_RSZ2_RSZ2_RSZ2_RSZ1_RSZ1_RSZ1, status: Status.UNKNOWN, split count: 7, time: 3.22
Output dim: 1, lower bound: -0.0007198, upper bound: 0.0007197
RS_RSZ2_RSZ2_RSZ2_RSZ2_RSZ1_RSZ1_RSZ2, status: Status.UNKNOWN, split count: 7, time: 3.22
Output dim: 1, lower bound: -0.0007198, upper bound: 0.0007197
RS_RSZ2_RSZ2_RSZ2_RSZ2_RSZ1_RSZ2_RSZ1, status: Status.UNKNOWN, split count: 7, time: 3.22
Output dim: 1, lower bound: -0.0007198, upper bound: 0.0007197
RS_RSZ2_RSZ2_RSZ2_RSZ2_RSZ1_RSZ2_RSZ2, status: Status.UNKNOWN, split count: 7, time: 3.22
Output dim: 1, lower bound: -0.0007198, upper bound: 0.0007197
RS_RSZ2_RSZ2_RSZ2_RSZ2_RSZ2_RSZ1_RSZ1, status: Status.UNKNOWN, split count: 7, time: 3.22
Output dim: 1, lower bound: -0.0007197, upper bound: 0.0007198
RS_RSZ2_RSZ2_RSZ2_RSZ2_RSZ2_RSZ1_RSZ2, status: Status.UNKNOWN, split count: 7, time: 3.22
Output dim: 1, lower bound: -0.0007197, upper bound: 0.0007198
RS_RSZ2_RSZ2_RSZ2_RSZ2_RSZ2_RSZ2_RSZ1, status: Status.UNKNOWN, split count: 7, time: 3.22
Output dim: 1, lower bound: -0.0007772, upper bound: 0.0007743
RS_RSZ2_RSZ2_RSZ2_RSZ2_RSZ2_RSZ2_RSZ2, status: Status.UNKNOWN, split count: 7, time: 3.22
Output dim: 1, lower bound: -0.0007771, upper bound: 0.0007745

## BFS RS instance: RS_RSZ1_RSZ1_RSZ1_RSZ1_RSZ1_RSZ1_RSZ1

### Backsubstitution after applying RS history:
0: 0.0007184, 0.0011172, 0.0007184, 0.0011172, -0.0003988, 0.0003988
1: 0.9934075, 0.9943579, 0.9934075, 0.9943579, -0.0009505, 0.0009505
2: -0.0087946, -0.0055215, -0.0087946, -0.0055215, -0.0029809, 0.0029832
3: 0.0035868, 0.0041619, 0.0035868, 0.0041619, -0.0005751, 0.0005751
4: 0.0027809, 0.0053678, 0.0027809, 0.0053678, -0.0025869, 0.0025869
5: 0.0050646, 0.0064029, 0.0050646, 0.0064029, -0.0013382, 0.0013382
6: -0.0021738, -0.0008562, -0.0021738, -0.0008562, -0.0013176, 0.0013176
7: -0.0082275, -0.0074685, -0.0082275, -0.0074685, -0.0007590, 0.0007590
8: 0.0055062, 0.0098066, 0.0055062, 0.0098066, -0.0042224, 0.0042218
9: -0.0036825, -0.0031002, -0.0036825, -0.0031002, -0.0005823, 0.0005823

### Unstable ReLU Count (Linear/Conv2D Layers)
- layer_idx=0, type=LayerType.Linear, total=256, inp1_unstable=9, inp2_unstable=9, delta_unstable=249
- layer_idx=2, type=LayerType.Linear, total=256, inp1_unstable=6, inp2_unstable=6, delta_unstable=256
- layer_idx=4, type=LayerType.Linear, total=256, inp1_unstable=0, inp2_unstable=0, delta_unstable=256
- layer_idx=6, type=LayerType.Linear, total=256, inp1_unstable=2, inp2_unstable=2, delta_unstable=256
- layer_idx=8, type=LayerType.Linear, total=10, inp1_unstable=0, inp2_unstable=0, delta_unstable=10

Time for backsubstitution: 1.69 seconds

### RS candidates at layer 1
type: RSZ, layer: 1, pos: 198
type: RSZ, layer: 1, pos: 68

Time for candidate selection: 0.00 seconds

### Candidate
type: RSZ, layer: 1, pos: 198

### Relational analysis RSZ of RS_RSZ1_RSZ1_RSZ1_RSZ1_RSZ1_RSZ1_RSZ1_RSZ1

#### Relational analysis RSZ result of RS_RSZ1_RSZ1_RSZ1_RSZ1_RSZ1_RSZ1_RSZ1_RSZ1
Status: Status.UNKNOWN
Output dim: 1, lower bound: -0.0007098, upper bound: 0.0007098
time: 0.62 seconds

### Relational analysis RSZ of RS_RSZ1_RSZ1_RSZ1_RSZ1_RSZ1_RSZ1_RSZ1_RSZ2

#### Relational analysis RSZ result of RS_RSZ1_RSZ1_RSZ1_RSZ1_RSZ1_RSZ1_RSZ1_RSZ2
Status: Status.UNKNOWN
Output dim: 1, lower bound: -0.0007098, upper bound: 0.0007098
time: 0.61 seconds

## BFS RS instance: RS_RSZ1_RSZ1_RSZ1_RSZ1_RSZ1_RSZ1_RSZ2

### Backsubstitution after applying RS history:
0: 0.0007184, 0.0011172, 0.0007184, 0.0011172, -0.0003988, 0.0003988
1: 0.9934075, 0.9943579, 0.9934075, 0.9943579, -0.0009505, 0.0009505
2: -0.0087946, -0.0055215, -0.0087946, -0.0055215, -0.0029807, 0.0029834
3: 0.0035868, 0.0041619, 0.0035868, 0.0041619, -0.0005751, 0.0005751
4: 0.0027809, 0.0053678, 0.0027809, 0.0053678, -0.0025869, 0.0025869
5: 0.0050646, 0.0064029, 0.0050646, 0.0064029, -0.0013382, 0.0013382
6: -0.0021738, -0.0008562, -0.0021738, -0.0008562, -0.0013176, 0.0013176
7: -0.0082275, -0.0074685, -0.0082275, -0.0074685, -0.0007590, 0.0007590
8: 0.0055062, 0.0098066, 0.0055062, 0.0098066, -0.0042224, 0.0042217
9: -0.0036825, -0.0031002, -0.0036825, -0.0031002, -0.0005823, 0.0005823

### Unstable ReLU Count (Linear/Conv2D Layers)
- layer_idx=0, type=LayerType.Linear, total=256, inp1_unstable=9, inp2_unstable=9, delta_unstable=249
- layer_idx=2, type=LayerType.Linear, total=256, inp1_unstable=6, inp2_unstable=6, delta_unstable=256
- layer_idx=4, type=LayerType.Linear, total=256, inp1_unstable=0, inp2_unstable=0, delta_unstable=256
- layer_idx=6, type=LayerType.Linear, total=256, inp1_unstable=2, inp2_unstable=2, delta_unstable=256
- layer_idx=8, type=LayerType.Linear, total=10, inp1_unstable=0, inp2_unstable=0, delta_unstable=10

Time for backsubstitution: 1.76 seconds

### RS candidates at layer 1
type: RSZ, layer: 1, pos: 68
type: RSZ, layer: 1, pos: 198

Time for candidate selection: 0.00 seconds

### Candidate
type: RSZ, layer: 1, pos: 68

### Relational analysis RSZ of RS_RSZ1_RSZ1_RSZ1_RSZ1_RSZ1_RSZ1_RSZ2_RSZ1

#### Relational analysis RSZ result of RS_RSZ1_RSZ1_RSZ1_RSZ1_RSZ1_RSZ1_RSZ2_RSZ1
Status: Status.UNKNOWN
Output dim: 1, lower bound: -0.0007534, upper bound: 0.0007538
time: 0.69 seconds

### Relational analysis RSZ of RS_RSZ1_RSZ1_RSZ1_RSZ1_RSZ1_RSZ1_RSZ2_RSZ2

#### Relational analysis RSZ result of RS_RSZ1_RSZ1_RSZ1_RSZ1_RSZ1_RSZ1_RSZ2_RSZ2
Status: Status.UNKNOWN
Output dim: 1, lower bound: -0.0007544, upper bound: 0.0007532
time: 0.69 seconds

## BFS RS instance: RS_RSZ1_RSZ1_RSZ1_RSZ1_RSZ1_RSZ2_RSZ1

### Backsubstitution after applying RS history:
0: 0.0007184, 0.0011172, 0.0007184, 0.0011172, -0.0003988, 0.0003988
1: 0.9934075, 0.9943579, 0.9934075, 0.9943579, -0.0009505, 0.0009505
2: -0.0087946, -0.0055215, -0.0087946, -0.0055215, -0.0029764, 0.0029789
3: 0.0035868, 0.0041619, 0.0035868, 0.0041619, -0.0005751, 0.0005751
4: 0.0027809, 0.0053678, 0.0027809, 0.0053678, -0.0025869, 0.0025869
5: 0.0050646, 0.0064029, 0.0050646, 0.0064029, -0.0013382, 0.0013382
6: -0.0021738, -0.0008562, -0.0021738, -0.0008562, -0.0013176, 0.0013176
7: -0.0082275, -0.0074685, -0.0082275, -0.0074685, -0.0007590, 0.0007590
8: 0.0055062, 0.0098066, 0.0055062, 0.0098066, -0.0042208, 0.0042201
9: -0.0036825, -0.0031002, -0.0036825, -0.0031002, -0.0005823, 0.0005823

### Unstable ReLU Count (Linear/Conv2D Layers)
- layer_idx=0, type=LayerType.Linear, total=256, inp1_unstable=9, inp2_unstable=9, delta_unstable=249
- layer_idx=2, type=LayerType.Linear, total=256, inp1_unstable=6, inp2_unstable=6, delta_unstable=256
- layer_idx=4, type=LayerType.Linear, total=256, inp1_unstable=0, inp2_unstable=0, delta_unstable=256
- layer_idx=6, type=LayerType.Linear, total=256, inp1_unstable=2, inp2_unstable=2, delta_unstable=256
- layer_idx=8, type=LayerType.Linear, total=10, inp1_unstable=0, inp2_unstable=0, delta_unstable=10

Time for backsubstitution: 1.76 seconds

### RS candidates at layer 1
type: RSZ, layer: 1, pos: 83
type: RSZ, layer: 1, pos: 198

Time for candidate selection: 0.00 seconds

### Candidate
type: RSZ, layer: 1, pos: 83

### Relational analysis RSZ of RS_RSZ1_RSZ1_RSZ1_RSZ1_RSZ1_RSZ2_RSZ1_RSZ1

#### Relational analysis RSZ result of RS_RSZ1_RSZ1_RSZ1_RSZ1_RSZ1_RSZ2_RSZ1_RSZ1
Status: Status.UNKNOWN
Output dim: 1, lower bound: -0.0007535, upper bound: 0.0007538
time: 0.78 seconds

### Relational analysis RSZ of RS_RSZ1_RSZ1_RSZ1_RSZ1_RSZ1_RSZ2_RSZ1_RSZ2

#### Relational analysis RSZ result of RS_RSZ1_RSZ1_RSZ1_RSZ1_RSZ1_RSZ2_RSZ1_RSZ2
Status: Status.UNKNOWN
Output dim: 1, lower bound: -0.0007534, upper bound: 0.0007538
time: 0.69 seconds

## BFS RS instance: RS_RSZ1_RSZ1_RSZ1_RSZ1_RSZ1_RSZ2_RSZ2

### Backsubstitution after applying RS history:
0: 0.0007184, 0.0011172, 0.0007184, 0.0011172, -0.0003988, 0.0003988
1: 0.9934075, 0.9943579, 0.9934075, 0.9943579, -0.0009505, 0.0009505
2: -0.0087946, -0.0055215, -0.0087946, -0.0055215, -0.0029751, 0.0029806
3: 0.0035868, 0.0041619, 0.0035868, 0.0041619, -0.0005751, 0.0005751
4: 0.0027809, 0.0053678, 0.0027809, 0.0053678, -0.0025869, 0.0025869
5: 0.0050646, 0.0064029, 0.0050646, 0.0064029, -0.0013382, 0.0013382
6: -0.0021738, -0.0008562, -0.0021738, -0.0008562, -0.0013176, 0.0013176
7: -0.0082275, -0.0074685, -0.0082275, -0.0074685, -0.0007590, 0.0007590
8: 0.0055062, 0.0098066, 0.0055062, 0.0098066, -0.0042212, 0.0042198
9: -0.0036825, -0.0031002, -0.0036825, -0.0031002, -0.0005823, 0.0005823

### Unstable ReLU Count (Linear/Conv2D Layers)
- layer_idx=0, type=LayerType.Linear, total=256, inp1_unstable=9, inp2_unstable=9, delta_unstable=249
- layer_idx=2, type=LayerType.Linear, total=256, inp1_unstable=6, inp2_unstable=6, delta_unstable=256
- layer_idx=4, type=LayerType.Linear, total=256, inp1_unstable=0, inp2_unstable=0, delta_unstable=256
- layer_idx=6, type=LayerType.Linear, total=256, inp1_unstable=2, inp2_unstable=2, delta_unstable=256
- layer_idx=8, type=LayerType.Linear, total=10, inp1_unstable=0, inp2_unstable=0, delta_unstable=10

Time for backsubstitution: 1.66 seconds

### RS candidates at layer 1
type: RSZ, layer: 1, pos: 83
type: RSZ, layer: 1, pos: 198

Time for candidate selection: 0.00 seconds

### Candidate
type: RSZ, layer: 1, pos: 83

### Relational analysis RSZ of RS_RSZ1_RSZ1_RSZ1_RSZ1_RSZ1_RSZ2_RSZ2_RSZ1

#### Relational analysis RSZ result of RS_RSZ1_RSZ1_RSZ1_RSZ1_RSZ1_RSZ2_RSZ2_RSZ1
Status: Status.UNKNOWN
Output dim: 1, lower bound: -0.0007544, upper bound: 0.0007532
time: 0.71 seconds

### Relational analysis RSZ of RS_RSZ1_RSZ1_RSZ1_RSZ1_RSZ1_RSZ2_RSZ2_RSZ2

#### Relational analysis RSZ result of RS_RSZ1_RSZ1_RSZ1_RSZ1_RSZ1_RSZ2_RSZ2_RSZ2
Status: Status.UNKNOWN
Output dim: 1, lower bound: -0.0007544, upper bound: 0.0007532
time: 0.69 seconds

## BFS RS instance: RS_RSZ1_RSZ1_RSZ1_RSZ1_RSZ2_RSZ1_RSZ1

### Backsubstitution after applying RS history:
0: 0.0007184, 0.0011172, 0.0007184, 0.0011172, -0.0003988, 0.0003988
1: 0.9934075, 0.9943579, 0.9934075, 0.9943579, -0.0009505, 0.0009505
2: -0.0087946, -0.0055215, -0.0087946, -0.0055215, -0.0029689, 0.0029680
3: 0.0035868, 0.0041619, 0.0035868, 0.0041619, -0.0005751, 0.0005751
4: 0.0027809, 0.0053678, 0.0027809, 0.0053678, -0.0025869, 0.0025869
5: 0.0050646, 0.0064029, 0.0050646, 0.0064029, -0.0013382, 0.0013382
6: -0.0021738, -0.0008562, -0.0021738, -0.0008562, -0.0013176, 0.0013176
7: -0.0082275, -0.0074685, -0.0082275, -0.0074685, -0.0007590, 0.0007590
8: 0.0055062, 0.0098066, 0.0055062, 0.0098066, -0.0042181, 0.0042184
9: -0.0036825, -0.0031002, -0.0036825, -0.0031002, -0.0005823, 0.0005823

### Unstable ReLU Count (Linear/Conv2D Layers)
- layer_idx=0, type=LayerType.Linear, total=256, inp1_unstable=9, inp2_unstable=9, delta_unstable=249
- layer_idx=2, type=LayerType.Linear, total=256, inp1_unstable=6, inp2_unstable=6, delta_unstable=256
- layer_idx=4, type=LayerType.Linear, total=256, inp1_unstable=0, inp2_unstable=0, delta_unstable=256
- layer_idx=6, type=LayerType.Linear, total=256, inp1_unstable=2, inp2_unstable=2, delta_unstable=256
- layer_idx=8, type=LayerType.Linear, total=10, inp1_unstable=0, inp2_unstable=0, delta_unstable=10

Time for backsubstitution: 1.72 seconds

### RS candidates at layer 1
type: RSZ, layer: 1, pos: 83
type: RSZ, layer: 1, pos: 199

Time for candidate selection: 0.00 seconds

### Candidate
type: RSZ, layer: 1, pos: 83

### Relational analysis RSZ of RS_RSZ1_RSZ1_RSZ1_RSZ1_RSZ2_RSZ1_RSZ1_RSZ1

#### Relational analysis RSZ result of RS_RSZ1_RSZ1_RSZ1_RSZ1_RSZ2_RSZ1_RSZ1_RSZ1
Status: Status.UNKNOWN
Output dim: 1, lower bound: -0.0007679, upper bound: 0.0007704
time: 0.67 seconds

### Relational analysis RSZ of RS_RSZ1_RSZ1_RSZ1_RSZ1_RSZ2_RSZ1_RSZ1_RSZ2

#### Relational analysis RSZ result of RS_RSZ1_RSZ1_RSZ1_RSZ1_RSZ2_RSZ1_RSZ1_RSZ2
Status: Status.UNKNOWN
Output dim: 1, lower bound: -0.0007676, upper bound: 0.0007704
time: 0.86 seconds

## BFS RS instance: RS_RSZ1_RSZ1_RSZ1_RSZ1_RSZ2_RSZ1_RSZ2

### Backsubstitution after applying RS history:
0: 0.0007184, 0.0011172, 0.0007184, 0.0011172, -0.0003988, 0.0003988
1: 0.9934075, 0.9943579, 0.9934075, 0.9943579, -0.0009505, 0.0009505
2: -0.0087946, -0.0055215, -0.0087946, -0.0055215, -0.0029675, 0.0029699
3: 0.0035868, 0.0041619, 0.0035868, 0.0041619, -0.0005751, 0.0005751
4: 0.0027809, 0.0053678, 0.0027809, 0.0053678, -0.0025869, 0.0025869
5: 0.0050646, 0.0064029, 0.0050646, 0.0064029, -0.0013382, 0.0013382
6: -0.0021738, -0.0008562, -0.0021738, -0.0008562, -0.0013176, 0.0013176
7: -0.0082275, -0.0074685, -0.0082275, -0.0074685, -0.0007590, 0.0007590
8: 0.0055062, 0.0098066, 0.0055062, 0.0098066, -0.0042186, 0.0042180
9: -0.0036825, -0.0031002, -0.0036825, -0.0031002, -0.0005823, 0.0005823

### Unstable ReLU Count (Linear/Conv2D Layers)
- layer_idx=0, type=LayerType.Linear, total=256, inp1_unstable=9, inp2_unstable=9, delta_unstable=249
- layer_idx=2, type=LayerType.Linear, total=256, inp1_unstable=6, inp2_unstable=6, delta_unstable=256
- layer_idx=4, type=LayerType.Linear, total=256, inp1_unstable=0, inp2_unstable=0, delta_unstable=256
- layer_idx=6, type=LayerType.Linear, total=256, inp1_unstable=2, inp2_unstable=2, delta_unstable=256
- layer_idx=8, type=LayerType.Linear, total=10, inp1_unstable=0, inp2_unstable=0, delta_unstable=10

Time for backsubstitution: 1.72 seconds

### RS candidates at layer 1
type: RSZ, layer: 1, pos: 199
type: RSZ, layer: 1, pos: 83

Time for candidate selection: 0.00 seconds

### Candidate
type: RSZ, layer: 1, pos: 199

### Relational analysis RSZ of RS_RSZ1_RSZ1_RSZ1_RSZ1_RSZ2_RSZ1_RSZ2_RSZ1

#### Relational analysis RSZ result of RS_RSZ1_RSZ1_RSZ1_RSZ1_RSZ2_RSZ1_RSZ2_RSZ1
Status: Status.UNKNOWN
Output dim: 1, lower bound: -0.0007131, upper bound: 0.0007131
time: 0.62 seconds

### Relational analysis RSZ of RS_RSZ1_RSZ1_RSZ1_RSZ1_RSZ2_RSZ1_RSZ2_RSZ2

#### Relational analysis RSZ result of RS_RSZ1_RSZ1_RSZ1_RSZ1_RSZ2_RSZ1_RSZ2_RSZ2
Status: Status.UNKNOWN
Output dim: 1, lower bound: -0.0007131, upper bound: 0.0007131
time: 0.62 seconds

## BFS RS instance: RS_RSZ1_RSZ1_RSZ1_RSZ1_RSZ2_RSZ2_RSZ1

### Backsubstitution after applying RS history:
0: 0.0007184, 0.0011172, 0.0007184, 0.0011172, -0.0003988, 0.0003988
1: 0.9934075, 0.9943579, 0.9934075, 0.9943579, -0.0009505, 0.0009505
2: -0.0087946, -0.0055215, -0.0087946, -0.0055215, -0.0029844, 0.0029746
3: 0.0035868, 0.0041619, 0.0035868, 0.0041619, -0.0005751, 0.0005751
4: 0.0027809, 0.0053678, 0.0027809, 0.0053678, -0.0025869, 0.0025869
5: 0.0050646, 0.0064029, 0.0050646, 0.0064029, -0.0013382, 0.0013382
6: -0.0021738, -0.0008562, -0.0021738, -0.0008562, -0.0013176, 0.0013176
7: -0.0082275, -0.0074685, -0.0082275, -0.0074685, -0.0007590, 0.0007590
8: 0.0055062, 0.0098066, 0.0055062, 0.0098066, -0.0042203, 0.0042226
9: -0.0036825, -0.0031002, -0.0036825, -0.0031002, -0.0005823, 0.0005823

### Unstable ReLU Count (Linear/Conv2D Layers)
- layer_idx=0, type=LayerType.Linear, total=256, inp1_unstable=9, inp2_unstable=9, delta_unstable=249
- layer_idx=2, type=LayerType.Linear, total=256, inp1_unstable=6, inp2_unstable=6, delta_unstable=256
- layer_idx=4, type=LayerType.Linear, total=256, inp1_unstable=0, inp2_unstable=0, delta_unstable=256
- layer_idx=6, type=LayerType.Linear, total=256, inp1_unstable=2, inp2_unstable=2, delta_unstable=256
- layer_idx=8, type=LayerType.Linear, total=10, inp1_unstable=0, inp2_unstable=0, delta_unstable=10

Time for backsubstitution: 1.72 seconds

### RS candidates at layer 1
type: RSZ, layer: 1, pos: 68
type: RSZ, layer: 1, pos: 83

Time for candidate selection: 0.00 seconds

### Candidate
type: RSZ, layer: 1, pos: 68

### Relational analysis RSZ of RS_RSZ1_RSZ1_RSZ1_RSZ1_RSZ2_RSZ2_RSZ1_RSZ1

#### Relational analysis RSZ result of RS_RSZ1_RSZ1_RSZ1_RSZ1_RSZ2_RSZ2_RSZ1_RSZ1
Status: Status.UNKNOWN
Output dim: 1, lower bound: -0.0007131, upper bound: 0.0007131
time: 0.62 seconds

### Relational analysis RSZ of RS_RSZ1_RSZ1_RSZ1_RSZ1_RSZ2_RSZ2_RSZ1_RSZ2

#### Relational analysis RSZ result of RS_RSZ1_RSZ1_RSZ1_RSZ1_RSZ2_RSZ2_RSZ1_RSZ2
Status: Status.UNKNOWN
Output dim: 1, lower bound: -0.0007131, upper bound: 0.0007131
time: 0.60 seconds

## BFS RS instance: RS_RSZ1_RSZ1_RSZ1_RSZ1_RSZ2_RSZ2_RSZ2

### Backsubstitution after applying RS history:
0: 0.0007184, 0.0011172, 0.0007184, 0.0011172, -0.0003988, 0.0003988
1: 0.9934075, 0.9943579, 0.9934075, 0.9943579, -0.0009505, 0.0009505
2: -0.0087946, -0.0055215, -0.0087946, -0.0055215, -0.0029852, 0.0029765
3: 0.0035868, 0.0041619, 0.0035868, 0.0041619, -0.0005751, 0.0005751
4: 0.0027809, 0.0053678, 0.0027809, 0.0053678, -0.0025869, 0.0025869
5: 0.0050646, 0.0064029, 0.0050646, 0.0064029, -0.0013382, 0.0013382
6: -0.0021738, -0.0008562, -0.0021738, -0.0008562, -0.0013176, 0.0013176
7: -0.0082275, -0.0074685, -0.0082275, -0.0074685, -0.0007590, 0.0007590
8: 0.0055062, 0.0098066, 0.0055062, 0.0098066, -0.0042208, 0.0042228
9: -0.0036825, -0.0031002, -0.0036825, -0.0031002, -0.0005823, 0.0005823

### Unstable ReLU Count (Linear/Conv2D Layers)
- layer_idx=0, type=LayerType.Linear, total=256, inp1_unstable=9, inp2_unstable=9, delta_unstable=249
- layer_idx=2, type=LayerType.Linear, total=256, inp1_unstable=6, inp2_unstable=6, delta_unstable=256
- layer_idx=4, type=LayerType.Linear, total=256, inp1_unstable=0, inp2_unstable=0, delta_unstable=256
- layer_idx=6, type=LayerType.Linear, total=256, inp1_unstable=2, inp2_unstable=2, delta_unstable=256
- layer_idx=8, type=LayerType.Linear, total=10, inp1_unstable=0, inp2_unstable=0, delta_unstable=10

Time for backsubstitution: 1.68 seconds

### RS candidates at layer 1
type: RSZ, layer: 1, pos: 83
type: RSZ, layer: 1, pos: 68

Time for candidate selection: 0.00 seconds

### Candidate
type: RSZ, layer: 1, pos: 83

### Relational analysis RSZ of RS_RSZ1_RSZ1_RSZ1_RSZ1_RSZ2_RSZ2_RSZ2_RSZ1

#### Relational analysis RSZ result of RS_RSZ1_RSZ1_RSZ1_RSZ1_RSZ2_RSZ2_RSZ2_RSZ1
Status: Status.UNKNOWN
Output dim: 1, lower bound: -0.0007098, upper bound: 0.0007098
time: 0.65 seconds

### Relational analysis RSZ of RS_RSZ1_RSZ1_RSZ1_RSZ1_RSZ2_RSZ2_RSZ2_RSZ2

#### Relational analysis RSZ result of RS_RSZ1_RSZ1_RSZ1_RSZ1_RSZ2_RSZ2_RSZ2_RSZ2
Status: Status.UNKNOWN
Output dim: 1, lower bound: -0.0007098, upper bound: 0.0007098
time: 0.61 seconds

## BFS RS instance: RS_RSZ1_RSZ1_RSZ1_RSZ2_RSZ1_RSZ1_RSZ1

### Backsubstitution after applying RS history:
0: 0.0007184, 0.0011172, 0.0007184, 0.0011172, -0.0003988, 0.0003988
1: 0.9934075, 0.9943579, 0.9934075, 0.9943579, -0.0009505, 0.0009505
2: -0.0087946, -0.0055215, -0.0087946, -0.0055215, -0.0029708, 0.0029668
3: 0.0035868, 0.0041619, 0.0035868, 0.0041619, -0.0005751, 0.0005751
4: 0.0027809, 0.0053678, 0.0027809, 0.0053678, -0.0025869, 0.0025869
5: 0.0050646, 0.0064029, 0.0050646, 0.0064029, -0.0013382, 0.0013382
6: -0.0021738, -0.0008562, -0.0021738, -0.0008562, -0.0013176, 0.0013176
7: -0.0082275, -0.0074685, -0.0082275, -0.0074685, -0.0007590, 0.0007590
8: 0.0055062, 0.0098066, 0.0055062, 0.0098066, -0.0042180, 0.0042190
9: -0.0036825, -0.0031002, -0.0036825, -0.0031002, -0.0005823, 0.0005823

### Unstable ReLU Count (Linear/Conv2D Layers)
- layer_idx=0, type=LayerType.Linear, total=256, inp1_unstable=9, inp2_unstable=9, delta_unstable=249
- layer_idx=2, type=LayerType.Linear, total=256, inp1_unstable=6, inp2_unstable=6, delta_unstable=256
- layer_idx=4, type=LayerType.Linear, total=256, inp1_unstable=0, inp2_unstable=0, delta_unstable=256
- layer_idx=6, type=LayerType.Linear, total=256, inp1_unstable=2, inp2_unstable=2, delta_unstable=256
- layer_idx=8, type=LayerType.Linear, total=10, inp1_unstable=0, inp2_unstable=0, delta_unstable=10

Time for backsubstitution: 1.66 seconds

### RS candidates at layer 1
type: RSZ, layer: 1, pos: 52
type: RSZ, layer: 1, pos: 199

Time for candidate selection: 0.00 seconds

### Candidate
type: RSZ, layer: 1, pos: 52

### Relational analysis RSZ of RS_RSZ1_RSZ1_RSZ1_RSZ2_RSZ1_RSZ1_RSZ1_RSZ1

#### Relational analysis RSZ result of RS_RSZ1_RSZ1_RSZ1_RSZ2_RSZ1_RSZ1_RSZ1_RSZ1
Status: Status.UNKNOWN
Output dim: 1, lower bound: -0.0007679, upper bound: 0.0007701
time: 0.73 seconds

### Relational analysis RSZ of RS_RSZ1_RSZ1_RSZ1_RSZ2_RSZ1_RSZ1_RSZ1_RSZ2

#### Relational analysis RSZ result of RS_RSZ1_RSZ1_RSZ1_RSZ2_RSZ1_RSZ1_RSZ1_RSZ2
Status: Status.UNKNOWN
Output dim: 1, lower bound: -0.0007679, upper bound: 0.0007704
time: 0.70 seconds

## BFS RS instance: RS_RSZ1_RSZ1_RSZ1_RSZ2_RSZ1_RSZ1_RSZ2

### Backsubstitution after applying RS history:
0: 0.0007184, 0.0011172, 0.0007184, 0.0011172, -0.0003988, 0.0003988
1: 0.9934075, 0.9943579, 0.9934075, 0.9943579, -0.0009505, 0.0009505
2: -0.0087946, -0.0055215, -0.0087946, -0.0055215, -0.0029814, 0.0029655
3: 0.0035868, 0.0041619, 0.0035868, 0.0041619, -0.0005751, 0.0005751
4: 0.0027809, 0.0053678, 0.0027809, 0.0053678, -0.0025869, 0.0025869
5: 0.0050646, 0.0064029, 0.0050646, 0.0064029, -0.0013382, 0.0013382
6: -0.0021738, -0.0008562, -0.0021738, -0.0008562, -0.0013176, 0.0013176
7: -0.0082275, -0.0074685, -0.0082275, -0.0074685, -0.0007590, 0.0007590
8: 0.0055062, 0.0098066, 0.0055062, 0.0098066, -0.0042176, 0.0042216
9: -0.0036825, -0.0031002, -0.0036825, -0.0031002, -0.0005823, 0.0005823

### Unstable ReLU Count (Linear/Conv2D Layers)
- layer_idx=0, type=LayerType.Linear, total=256, inp1_unstable=9, inp2_unstable=9, delta_unstable=249
- layer_idx=2, type=LayerType.Linear, total=256, inp1_unstable=6, inp2_unstable=6, delta_unstable=256
- layer_idx=4, type=LayerType.Linear, total=256, inp1_unstable=0, inp2_unstable=0, delta_unstable=256
- layer_idx=6, type=LayerType.Linear, total=256, inp1_unstable=2, inp2_unstable=2, delta_unstable=256
- layer_idx=8, type=LayerType.Linear, total=10, inp1_unstable=0, inp2_unstable=0, delta_unstable=10

Time for backsubstitution: 1.70 seconds

### RS candidates at layer 1
type: RSZ, layer: 1, pos: 52
type: RSZ, layer: 1, pos: 199

Time for candidate selection: 0.00 seconds

### Candidate
type: RSZ, layer: 1, pos: 52

### Relational analysis RSZ of RS_RSZ1_RSZ1_RSZ1_RSZ2_RSZ1_RSZ1_RSZ2_RSZ1

#### Relational analysis RSZ result of RS_RSZ1_RSZ1_RSZ1_RSZ2_RSZ1_RSZ1_RSZ2_RSZ1
Status: Status.UNKNOWN
Output dim: 1, lower bound: -0.0007679, upper bound: 0.0007701
time: 0.69 seconds

### Relational analysis RSZ of RS_RSZ1_RSZ1_RSZ1_RSZ2_RSZ1_RSZ1_RSZ2_RSZ2

#### Relational analysis RSZ result of RS_RSZ1_RSZ1_RSZ1_RSZ2_RSZ1_RSZ1_RSZ2_RSZ2
Status: Status.UNKNOWN
Output dim: 1, lower bound: -0.0007679, upper bound: 0.0007704
time: 0.69 seconds

## BFS RS instance: RS_RSZ1_RSZ1_RSZ1_RSZ2_RSZ1_RSZ2_RSZ1

### Backsubstitution after applying RS history:
0: 0.0007184, 0.0011172, 0.0007184, 0.0011172, -0.0003988, 0.0003988
1: 0.9934075, 0.9943579, 0.9934075, 0.9943579, -0.0009505, 0.0009505
2: -0.0087946, -0.0055215, -0.0087946, -0.0055215, -0.0029763, 0.0029753
3: 0.0035868, 0.0041619, 0.0035868, 0.0041619, -0.0005751, 0.0005751
4: 0.0027809, 0.0053678, 0.0027809, 0.0053678, -0.0025869, 0.0025869
5: 0.0050646, 0.0064029, 0.0050646, 0.0064029, -0.0013382, 0.0013382
6: -0.0021738, -0.0008562, -0.0021738, -0.0008562, -0.0013176, 0.0013176
7: -0.0082275, -0.0074685, -0.0082275, -0.0074685, -0.0007590, 0.0007590
8: 0.0055062, 0.0098066, 0.0055062, 0.0098066, -0.0042199, 0.0042202
9: -0.0036825, -0.0031002, -0.0036825, -0.0031002, -0.0005823, 0.0005823

### Unstable ReLU Count (Linear/Conv2D Layers)
- layer_idx=0, type=LayerType.Linear, total=256, inp1_unstable=9, inp2_unstable=9, delta_unstable=249
- layer_idx=2, type=LayerType.Linear, total=256, inp1_unstable=6, inp2_unstable=6, delta_unstable=256
- layer_idx=4, type=LayerType.Linear, total=256, inp1_unstable=0, inp2_unstable=0, delta_unstable=256
- layer_idx=6, type=LayerType.Linear, total=256, inp1_unstable=2, inp2_unstable=2, delta_unstable=256
- layer_idx=8, type=LayerType.Linear, total=10, inp1_unstable=0, inp2_unstable=0, delta_unstable=10

Time for backsubstitution: 1.67 seconds

### RS candidates at layer 1
type: RSZ, layer: 1, pos: 199
type: RSZ, layer: 1, pos: 198

Time for candidate selection: 0.00 seconds

### Candidate
type: RSZ, layer: 1, pos: 199

### Relational analysis RSZ of RS_RSZ1_RSZ1_RSZ1_RSZ2_RSZ1_RSZ2_RSZ1_RSZ1

#### Relational analysis RSZ result of RS_RSZ1_RSZ1_RSZ1_RSZ2_RSZ1_RSZ2_RSZ1_RSZ1
Status: Status.UNKNOWN
Output dim: 1, lower bound: -0.0007540, upper bound: 0.0007534
time: 0.68 seconds

### Relational analysis RSZ of RS_RSZ1_RSZ1_RSZ1_RSZ2_RSZ1_RSZ2_RSZ1_RSZ2

#### Relational analysis RSZ result of RS_RSZ1_RSZ1_RSZ1_RSZ2_RSZ1_RSZ2_RSZ1_RSZ2
Status: Status.UNKNOWN
Output dim: 1, lower bound: -0.0007540, upper bound: 0.0007534
time: 0.67 seconds

## BFS RS instance: RS_RSZ1_RSZ1_RSZ1_RSZ2_RSZ1_RSZ2_RSZ2

### Backsubstitution after applying RS history:
0: 0.0007184, 0.0011172, 0.0007184, 0.0011172, -0.0003988, 0.0003988
1: 0.9934075, 0.9943579, 0.9934075, 0.9943579, -0.0009505, 0.0009505
2: -0.0087946, -0.0055215, -0.0087946, -0.0055215, -0.0029772, 0.0029743
3: 0.0035868, 0.0041619, 0.0035868, 0.0041619, -0.0005751, 0.0005751
4: 0.0027809, 0.0053678, 0.0027809, 0.0053678, -0.0025869, 0.0025869
5: 0.0050646, 0.0064029, 0.0050646, 0.0064029, -0.0013382, 0.0013382
6: -0.0021738, -0.0008562, -0.0021738, -0.0008562, -0.0013176, 0.0013176
7: -0.0082275, -0.0074685, -0.0082275, -0.0074685, -0.0007590, 0.0007590
8: 0.0055062, 0.0098066, 0.0055062, 0.0098066, -0.0042196, 0.0042204
9: -0.0036825, -0.0031002, -0.0036825, -0.0031002, -0.0005823, 0.0005823

### Unstable ReLU Count (Linear/Conv2D Layers)
- layer_idx=0, type=LayerType.Linear, total=256, inp1_unstable=9, inp2_unstable=9, delta_unstable=249
- layer_idx=2, type=LayerType.Linear, total=256, inp1_unstable=6, inp2_unstable=6, delta_unstable=256
- layer_idx=4, type=LayerType.Linear, total=256, inp1_unstable=0, inp2_unstable=0, delta_unstable=256
- layer_idx=6, type=LayerType.Linear, total=256, inp1_unstable=2, inp2_unstable=2, delta_unstable=256
- layer_idx=8, type=LayerType.Linear, total=10, inp1_unstable=0, inp2_unstable=0, delta_unstable=10

Time for backsubstitution: 1.68 seconds

### RS candidates at layer 1
type: RSZ, layer: 1, pos: 198
type: RSZ, layer: 1, pos: 199

Time for candidate selection: 0.00 seconds

### Candidate
type: RSZ, layer: 1, pos: 198

### Relational analysis RSZ of RS_RSZ1_RSZ1_RSZ1_RSZ2_RSZ1_RSZ2_RSZ2_RSZ1

#### Relational analysis RSZ result of RS_RSZ1_RSZ1_RSZ1_RSZ2_RSZ1_RSZ2_RSZ2_RSZ1
Status: Status.UNKNOWN
Output dim: 1, lower bound: -0.0007697, upper bound: 0.0007687
time: 0.91 seconds

### Relational analysis RSZ of RS_RSZ1_RSZ1_RSZ1_RSZ2_RSZ1_RSZ2_RSZ2_RSZ2

#### Relational analysis RSZ result of RS_RSZ1_RSZ1_RSZ1_RSZ2_RSZ1_RSZ2_RSZ2_RSZ2
Status: Status.UNKNOWN
Output dim: 1, lower bound: -0.0007697, upper bound: 0.0007686
time: 0.95 seconds

## BFS RS instance: RS_RSZ1_RSZ1_RSZ1_RSZ2_RSZ2_RSZ1_RSZ1

### Backsubstitution after applying RS history:
0: 0.0007184, 0.0011172, 0.0007184, 0.0011172, -0.0003988, 0.0003988
1: 0.9934075, 0.9943579, 0.9934075, 0.9943579, -0.0009505, 0.0009505
2: -0.0087946, -0.0055215, -0.0087946, -0.0055215, -0.0029747, 0.0029736
3: 0.0035868, 0.0041619, 0.0035868, 0.0041619, -0.0005751, 0.0005751
4: 0.0027809, 0.0053678, 0.0027809, 0.0053678, -0.0025869, 0.0025869
5: 0.0050646, 0.0064029, 0.0050646, 0.0064029, -0.0013382, 0.0013382
6: -0.0021738, -0.0008562, -0.0021738, -0.0008562, -0.0013176, 0.0013176
7: -0.0082275, -0.0074685, -0.0082275, -0.0074685, -0.0007590, 0.0007590
8: 0.0055062, 0.0098066, 0.0055062, 0.0098066, -0.0042201, 0.0042204
9: -0.0036825, -0.0031002, -0.0036825, -0.0031002, -0.0005823, 0.0005823

### Unstable ReLU Count (Linear/Conv2D Layers)
- layer_idx=0, type=LayerType.Linear, total=256, inp1_unstable=9, inp2_unstable=9, delta_unstable=249
- layer_idx=2, type=LayerType.Linear, total=256, inp1_unstable=6, inp2_unstable=6, delta_unstable=256
- layer_idx=4, type=LayerType.Linear, total=256, inp1_unstable=0, inp2_unstable=0, delta_unstable=256
- layer_idx=6, type=LayerType.Linear, total=256, inp1_unstable=2, inp2_unstable=2, delta_unstable=256
- layer_idx=8, type=LayerType.Linear, total=10, inp1_unstable=0, inp2_unstable=0, delta_unstable=10

Time for backsubstitution: 1.76 seconds

### RS candidates at layer 1
type: RSZ, layer: 1, pos: 199
type: RSZ, layer: 1, pos: 68

Time for candidate selection: 0.00 seconds

### Candidate
type: RSZ, layer: 1, pos: 199

### Relational analysis RSZ of RS_RSZ1_RSZ1_RSZ1_RSZ2_RSZ2_RSZ1_RSZ1_RSZ1

#### Relational analysis RSZ result of RS_RSZ1_RSZ1_RSZ1_RSZ2_RSZ2_RSZ1_RSZ1_RSZ1
Status: Status.UNKNOWN
Output dim: 1, lower bound: -0.0007098, upper bound: 0.0007098
time: 0.63 seconds

### Relational analysis RSZ of RS_RSZ1_RSZ1_RSZ1_RSZ2_RSZ2_RSZ1_RSZ1_RSZ2

#### Relational analysis RSZ result of RS_RSZ1_RSZ1_RSZ1_RSZ2_RSZ2_RSZ1_RSZ1_RSZ2
Status: Status.UNKNOWN
Output dim: 1, lower bound: -0.0007098, upper bound: 0.0007098
time: 0.61 seconds

## BFS RS instance: RS_RSZ1_RSZ1_RSZ1_RSZ2_RSZ2_RSZ1_RSZ2

### Backsubstitution after applying RS history:
0: 0.0007184, 0.0011172, 0.0007184, 0.0011172, -0.0003988, 0.0003988
1: 0.9934075, 0.9943579, 0.9934075, 0.9943579, -0.0009505, 0.0009505
2: -0.0087946, -0.0055215, -0.0087946, -0.0055215, -0.0029756, 0.0029724
3: 0.0035868, 0.0041619, 0.0035868, 0.0041619, -0.0005751, 0.0005751
4: 0.0027809, 0.0053678, 0.0027809, 0.0053678, -0.0025869, 0.0025869
5: 0.0050646, 0.0064029, 0.0050646, 0.0064029, -0.0013382, 0.0013382
6: -0.0021738, -0.0008562, -0.0021738, -0.0008562, -0.0013176, 0.0013176
7: -0.0082275, -0.0074685, -0.0082275, -0.0074685, -0.0007590, 0.0007590
8: 0.0055062, 0.0098066, 0.0055062, 0.0098066, -0.0042198, 0.0042206
9: -0.0036825, -0.0031002, -0.0036825, -0.0031002, -0.0005823, 0.0005823

### Unstable ReLU Count (Linear/Conv2D Layers)
- layer_idx=0, type=LayerType.Linear, total=256, inp1_unstable=9, inp2_unstable=9, delta_unstable=249
- layer_idx=2, type=LayerType.Linear, total=256, inp1_unstable=6, inp2_unstable=6, delta_unstable=256
- layer_idx=4, type=LayerType.Linear, total=256, inp1_unstable=0, inp2_unstable=0, delta_unstable=256
- layer_idx=6, type=LayerType.Linear, total=256, inp1_unstable=2, inp2_unstable=2, delta_unstable=256
- layer_idx=8, type=LayerType.Linear, total=10, inp1_unstable=0, inp2_unstable=0, delta_unstable=10

Time for backsubstitution: 1.76 seconds

### RS candidates at layer 1
type: RSZ, layer: 1, pos: 199
type: RSZ, layer: 1, pos: 68

Time for candidate selection: 0.00 seconds

### Candidate
type: RSZ, layer: 1, pos: 199

### Relational analysis RSZ of RS_RSZ1_RSZ1_RSZ1_RSZ2_RSZ2_RSZ1_RSZ2_RSZ1

#### Relational analysis RSZ result of RS_RSZ1_RSZ1_RSZ1_RSZ2_RSZ2_RSZ1_RSZ2_RSZ1
Status: Status.UNKNOWN
Output dim: 1, lower bound: -0.0007098, upper bound: 0.0007098
time: 0.60 seconds

### Relational analysis RSZ of RS_RSZ1_RSZ1_RSZ1_RSZ2_RSZ2_RSZ1_RSZ2_RSZ2

#### Relational analysis RSZ result of RS_RSZ1_RSZ1_RSZ1_RSZ2_RSZ2_RSZ1_RSZ2_RSZ2
Status: Status.UNKNOWN
Output dim: 1, lower bound: -0.0007098, upper bound: 0.0007098
time: 0.60 seconds

## BFS RS instance: RS_RSZ1_RSZ1_RSZ1_RSZ2_RSZ2_RSZ2_RSZ1

### Backsubstitution after applying RS history:
0: 0.0007184, 0.0011172, 0.0007184, 0.0011172, -0.0003988, 0.0003988
1: 0.9934075, 0.9943579, 0.9934075, 0.9943579, -0.0009505, 0.0009505
2: -0.0087946, -0.0055215, -0.0087946, -0.0055215, -0.0029859, 0.0029731
3: 0.0035868, 0.0041619, 0.0035868, 0.0041619, -0.0005751, 0.0005751
4: 0.0027809, 0.0053678, 0.0027809, 0.0053678, -0.0025869, 0.0025869
5: 0.0050646, 0.0064029, 0.0050646, 0.0064029, -0.0013382, 0.0013382
6: -0.0021738, -0.0008562, -0.0021738, -0.0008562, -0.0013176, 0.0013176
7: -0.0082275, -0.0074685, -0.0082275, -0.0074685, -0.0007590, 0.0007590
8: 0.0055062, 0.0098066, 0.0055062, 0.0098066, -0.0042201, 0.0042232
9: -0.0036825, -0.0031002, -0.0036825, -0.0031002, -0.0005823, 0.0005823

### Unstable ReLU Count (Linear/Conv2D Layers)
- layer_idx=0, type=LayerType.Linear, total=256, inp1_unstable=9, inp2_unstable=9, delta_unstable=249
- layer_idx=2, type=LayerType.Linear, total=256, inp1_unstable=6, inp2_unstable=6, delta_unstable=256
- layer_idx=4, type=LayerType.Linear, total=256, inp1_unstable=0, inp2_unstable=0, delta_unstable=256
- layer_idx=6, type=LayerType.Linear, total=256, inp1_unstable=2, inp2_unstable=2, delta_unstable=256
- layer_idx=8, type=LayerType.Linear, total=10, inp1_unstable=0, inp2_unstable=0, delta_unstable=10

Time for backsubstitution: 1.75 seconds

### RS candidates at layer 1
type: RSZ, layer: 1, pos: 68
type: RSZ, layer: 1, pos: 52

Time for candidate selection: 0.00 seconds

### Candidate
type: RSZ, layer: 1, pos: 68

### Relational analysis RSZ of RS_RSZ1_RSZ1_RSZ1_RSZ2_RSZ2_RSZ2_RSZ1_RSZ1

#### Relational analysis RSZ result of RS_RSZ1_RSZ1_RSZ1_RSZ2_RSZ2_RSZ2_RSZ1_RSZ1
Status: Status.UNKNOWN
Output dim: 1, lower bound: -0.0007098, upper bound: 0.0007098
time: 0.63 seconds

### Relational analysis RSZ of RS_RSZ1_RSZ1_RSZ1_RSZ2_RSZ2_RSZ2_RSZ1_RSZ2

#### Relational analysis RSZ result of RS_RSZ1_RSZ1_RSZ1_RSZ2_RSZ2_RSZ2_RSZ1_RSZ2
Status: Status.UNKNOWN
Output dim: 1, lower bound: -0.0007098, upper bound: 0.0007098
time: 0.62 seconds

## BFS RS instance: RS_RSZ1_RSZ1_RSZ1_RSZ2_RSZ2_RSZ2_RSZ2

### Backsubstitution after applying RS history:
0: 0.0007184, 0.0011172, 0.0007184, 0.0011172, -0.0003988, 0.0003988
1: 0.9934075, 0.9943579, 0.9934075, 0.9943579, -0.0009505, 0.0009505
2: -0.0087946, -0.0055215, -0.0087946, -0.0055215, -0.0029867, 0.0029751
3: 0.0035868, 0.0041619, 0.0035868, 0.0041619, -0.0005751, 0.0005751
4: 0.0027809, 0.0053678, 0.0027809, 0.0053678, -0.0025869, 0.0025869
5: 0.0050646, 0.0064029, 0.0050646, 0.0064029, -0.0013382, 0.0013382
6: -0.0021738, -0.0008562, -0.0021738, -0.0008562, -0.0013176, 0.0013176
7: -0.0082275, -0.0074685, -0.0082275, -0.0074685, -0.0007590, 0.0007590
8: 0.0055062, 0.0098066, 0.0055062, 0.0098066, -0.0042206, 0.0042234
9: -0.0036825, -0.0031002, -0.0036825, -0.0031002, -0.0005823, 0.0005823

### Unstable ReLU Count (Linear/Conv2D Layers)
- layer_idx=0, type=LayerType.Linear, total=256, inp1_unstable=9, inp2_unstable=9, delta_unstable=249
- layer_idx=2, type=LayerType.Linear, total=256, inp1_unstable=6, inp2_unstable=6, delta_unstable=256
- layer_idx=4, type=LayerType.Linear, total=256, inp1_unstable=0, inp2_unstable=0, delta_unstable=256
- layer_idx=6, type=LayerType.Linear, total=256, inp1_unstable=2, inp2_unstable=2, delta_unstable=256
- layer_idx=8, type=LayerType.Linear, total=10, inp1_unstable=0, inp2_unstable=0, delta_unstable=10

Time for backsubstitution: 1.74 seconds

### RS candidates at layer 1
type: RSZ, layer: 1, pos: 52
type: RSZ, layer: 1, pos: 68

Time for candidate selection: 0.00 seconds

### Candidate
type: RSZ, layer: 1, pos: 52

### Relational analysis RSZ of RS_RSZ1_RSZ1_RSZ1_RSZ2_RSZ2_RSZ2_RSZ2_RSZ1

#### Relational analysis RSZ result of RS_RSZ1_RSZ1_RSZ1_RSZ2_RSZ2_RSZ2_RSZ2_RSZ1
Status: Status.UNKNOWN
Output dim: 1, lower bound: -0.0007098, upper bound: 0.0007098
time: 0.60 seconds

### Relational analysis RSZ of RS_RSZ1_RSZ1_RSZ1_RSZ2_RSZ2_RSZ2_RSZ2_RSZ2

#### Relational analysis RSZ result of RS_RSZ1_RSZ1_RSZ1_RSZ2_RSZ2_RSZ2_RSZ2_RSZ2
Status: Status.UNKNOWN
Output dim: 1, lower bound: -0.0007098, upper bound: 0.0007098
time: 0.60 seconds

## BFS RS instance: RS_RSZ1_RSZ1_RSZ2_RSZ1_RSZ1_RSZ1_RSZ1

### Backsubstitution after applying RS history:
0: 0.0007184, 0.0011172, 0.0007184, 0.0011172, -0.0003988, 0.0003988
1: 0.9934075, 0.9943579, 0.9934075, 0.9943579, -0.0009505, 0.0009505
2: -0.0087946, -0.0055215, -0.0087946, -0.0055215, -0.0029712, 0.0029710
3: 0.0035868, 0.0041619, 0.0035868, 0.0041619, -0.0005751, 0.0005751
4: 0.0027809, 0.0053678, 0.0027809, 0.0053678, -0.0025869, 0.0025869
5: 0.0050646, 0.0064029, 0.0050646, 0.0064029, -0.0013382, 0.0013382
6: -0.0021738, -0.0008562, -0.0021738, -0.0008562, -0.0013176, 0.0013176
7: -0.0082275, -0.0074685, -0.0082275, -0.0074685, -0.0007590, 0.0007590
8: 0.0055062, 0.0098066, 0.0055062, 0.0098066, -0.0042186, 0.0042186
9: -0.0036825, -0.0031002, -0.0036825, -0.0031002, -0.0005823, 0.0005823

### Unstable ReLU Count (Linear/Conv2D Layers)
- layer_idx=0, type=LayerType.Linear, total=256, inp1_unstable=9, inp2_unstable=9, delta_unstable=249
- layer_idx=2, type=LayerType.Linear, total=256, inp1_unstable=6, inp2_unstable=6, delta_unstable=256
- layer_idx=4, type=LayerType.Linear, total=256, inp1_unstable=0, inp2_unstable=0, delta_unstable=256
- layer_idx=6, type=LayerType.Linear, total=256, inp1_unstable=2, inp2_unstable=2, delta_unstable=256
- layer_idx=8, type=LayerType.Linear, total=10, inp1_unstable=0, inp2_unstable=0, delta_unstable=10

Time for backsubstitution: 1.71 seconds

### RS candidates at layer 1
type: RSZ, layer: 1, pos: 199
type: RSZ, layer: 1, pos: 136

Time for candidate selection: 0.00 seconds

### Candidate
type: RSZ, layer: 1, pos: 199

### Relational analysis RSZ of RS_RSZ1_RSZ1_RSZ2_RSZ1_RSZ1_RSZ1_RSZ1_RSZ1

#### Relational analysis RSZ result of RS_RSZ1_RSZ1_RSZ2_RSZ1_RSZ1_RSZ1_RSZ1_RSZ1
Status: Status.UNKNOWN
Output dim: 1, lower bound: -0.0007098, upper bound: 0.0007098
time: 0.60 seconds

### Relational analysis RSZ of RS_RSZ1_RSZ1_RSZ2_RSZ1_RSZ1_RSZ1_RSZ1_RSZ2

#### Relational analysis RSZ result of RS_RSZ1_RSZ1_RSZ2_RSZ1_RSZ1_RSZ1_RSZ1_RSZ2
Status: Status.UNKNOWN
Output dim: 1, lower bound: -0.0007098, upper bound: 0.0007098
time: 0.61 seconds

## BFS RS instance: RS_RSZ1_RSZ1_RSZ2_RSZ1_RSZ1_RSZ1_RSZ2

### Backsubstitution after applying RS history:
0: 0.0007184, 0.0011172, 0.0007184, 0.0011172, -0.0003988, 0.0003988
1: 0.9934075, 0.9943579, 0.9934075, 0.9943579, -0.0009505, 0.0009505
2: -0.0087946, -0.0055215, -0.0087946, -0.0055215, -0.0029694, 0.0029728
3: 0.0035868, 0.0041619, 0.0035868, 0.0041619, -0.0005751, 0.0005751
4: 0.0027809, 0.0053678, 0.0027809, 0.0053678, -0.0025869, 0.0025869
5: 0.0050646, 0.0064029, 0.0050646, 0.0064029, -0.0013382, 0.0013382
6: -0.0021738, -0.0008562, -0.0021738, -0.0008562, -0.0013176, 0.0013176
7: -0.0082275, -0.0074685, -0.0082275, -0.0074685, -0.0007590, 0.0007590
8: 0.0055062, 0.0098066, 0.0055062, 0.0098066, -0.0042191, 0.0042181
9: -0.0036825, -0.0031002, -0.0036825, -0.0031002, -0.0005823, 0.0005823

### Unstable ReLU Count (Linear/Conv2D Layers)
- layer_idx=0, type=LayerType.Linear, total=256, inp1_unstable=9, inp2_unstable=9, delta_unstable=249
- layer_idx=2, type=LayerType.Linear, total=256, inp1_unstable=6, inp2_unstable=6, delta_unstable=256
- layer_idx=4, type=LayerType.Linear, total=256, inp1_unstable=0, inp2_unstable=0, delta_unstable=256
- layer_idx=6, type=LayerType.Linear, total=256, inp1_unstable=2, inp2_unstable=2, delta_unstable=256
- layer_idx=8, type=LayerType.Linear, total=10, inp1_unstable=0, inp2_unstable=0, delta_unstable=10

Time for backsubstitution: 1.68 seconds

### RS candidates at layer 1
type: RSZ, layer: 1, pos: 136
type: RSZ, layer: 1, pos: 199

Time for candidate selection: 0.00 seconds

### Candidate
type: RSZ, layer: 1, pos: 136

### Relational analysis RSZ of RS_RSZ1_RSZ1_RSZ2_RSZ1_RSZ1_RSZ1_RSZ2_RSZ1

#### Relational analysis RSZ result of RS_RSZ1_RSZ1_RSZ2_RSZ1_RSZ1_RSZ1_RSZ2_RSZ1
Status: Status.UNKNOWN
Output dim: 1, lower bound: -0.0007702, upper bound: 0.0007675
time: 0.74 seconds

### Relational analysis RSZ of RS_RSZ1_RSZ1_RSZ2_RSZ1_RSZ1_RSZ1_RSZ2_RSZ2

#### Relational analysis RSZ result of RS_RSZ1_RSZ1_RSZ2_RSZ1_RSZ1_RSZ1_RSZ2_RSZ2
Status: Status.UNKNOWN
Output dim: 1, lower bound: -0.0007703, upper bound: 0.0007675
time: 0.80 seconds

## BFS RS instance: RS_RSZ1_RSZ1_RSZ2_RSZ1_RSZ1_RSZ2_RSZ1

### Backsubstitution after applying RS history:
0: 0.0007184, 0.0011172, 0.0007184, 0.0011172, -0.0003988, 0.0003988
1: 0.9934075, 0.9943579, 0.9934075, 0.9943579, -0.0009505, 0.0009505
2: -0.0087946, -0.0055215, -0.0087946, -0.0055215, -0.0029708, 0.0029712
3: 0.0035868, 0.0041619, 0.0035868, 0.0041619, -0.0005751, 0.0005751
4: 0.0027809, 0.0053678, 0.0027809, 0.0053678, -0.0025869, 0.0025869
5: 0.0050646, 0.0064029, 0.0050646, 0.0064029, -0.0013382, 0.0013382
6: -0.0021738, -0.0008562, -0.0021738, -0.0008562, -0.0013176, 0.0013176
7: -0.0082275, -0.0074685, -0.0082275, -0.0074685, -0.0007590, 0.0007590
8: 0.0055062, 0.0098066, 0.0055062, 0.0098066, -0.0042186, 0.0042185
9: -0.0036825, -0.0031002, -0.0036825, -0.0031002, -0.0005823, 0.0005823

### Unstable ReLU Count (Linear/Conv2D Layers)
- layer_idx=0, type=LayerType.Linear, total=256, inp1_unstable=9, inp2_unstable=9, delta_unstable=249
- layer_idx=2, type=LayerType.Linear, total=256, inp1_unstable=6, inp2_unstable=6, delta_unstable=256
- layer_idx=4, type=LayerType.Linear, total=256, inp1_unstable=0, inp2_unstable=0, delta_unstable=256
- layer_idx=6, type=LayerType.Linear, total=256, inp1_unstable=2, inp2_unstable=2, delta_unstable=256
- layer_idx=8, type=LayerType.Linear, total=10, inp1_unstable=0, inp2_unstable=0, delta_unstable=10

Time for backsubstitution: 1.75 seconds

### RS candidates at layer 1
type: RSZ, layer: 1, pos: 199
type: RSZ, layer: 1, pos: 136

Time for candidate selection: 0.00 seconds

### Candidate
type: RSZ, layer: 1, pos: 199

### Relational analysis RSZ of RS_RSZ1_RSZ1_RSZ2_RSZ1_RSZ1_RSZ2_RSZ1_RSZ1

#### Relational analysis RSZ result of RS_RSZ1_RSZ1_RSZ2_RSZ1_RSZ1_RSZ2_RSZ1_RSZ1
Status: Status.UNKNOWN
Output dim: 1, lower bound: -0.0007098, upper bound: 0.0007098
time: 0.58 seconds

### Relational analysis RSZ of RS_RSZ1_RSZ1_RSZ2_RSZ1_RSZ1_RSZ2_RSZ1_RSZ2

#### Relational analysis RSZ result of RS_RSZ1_RSZ1_RSZ2_RSZ1_RSZ1_RSZ2_RSZ1_RSZ2
Status: Status.UNKNOWN
Output dim: 1, lower bound: -0.0007098, upper bound: 0.0007098
time: 0.57 seconds

## BFS RS instance: RS_RSZ1_RSZ1_RSZ2_RSZ1_RSZ1_RSZ2_RSZ2

### Backsubstitution after applying RS history:
0: 0.0007184, 0.0011172, 0.0007184, 0.0011172, -0.0003988, 0.0003988
1: 0.9934075, 0.9943579, 0.9934075, 0.9943579, -0.0009505, 0.0009505
2: -0.0087946, -0.0055215, -0.0087946, -0.0055215, -0.0029691, 0.0029731
3: 0.0035868, 0.0041619, 0.0035868, 0.0041619, -0.0005751, 0.0005751
4: 0.0027809, 0.0053678, 0.0027809, 0.0053678, -0.0025869, 0.0025869
5: 0.0050646, 0.0064029, 0.0050646, 0.0064029, -0.0013382, 0.0013382
6: -0.0021738, -0.0008562, -0.0021738, -0.0008562, -0.0013176, 0.0013176
7: -0.0082275, -0.0074685, -0.0082275, -0.0074685, -0.0007590, 0.0007590
8: 0.0055062, 0.0098066, 0.0055062, 0.0098066, -0.0042191, 0.0042181
9: -0.0036825, -0.0031002, -0.0036825, -0.0031002, -0.0005823, 0.0005823

### Unstable ReLU Count (Linear/Conv2D Layers)
- layer_idx=0, type=LayerType.Linear, total=256, inp1_unstable=9, inp2_unstable=9, delta_unstable=249
- layer_idx=2, type=LayerType.Linear, total=256, inp1_unstable=6, inp2_unstable=6, delta_unstable=256
- layer_idx=4, type=LayerType.Linear, total=256, inp1_unstable=0, inp2_unstable=0, delta_unstable=256
- layer_idx=6, type=LayerType.Linear, total=256, inp1_unstable=2, inp2_unstable=2, delta_unstable=256
- layer_idx=8, type=LayerType.Linear, total=10, inp1_unstable=0, inp2_unstable=0, delta_unstable=10

Time for backsubstitution: 1.70 seconds

### RS candidates at layer 1
type: RSZ, layer: 1, pos: 199
type: RSZ, layer: 1, pos: 136

Time for candidate selection: 0.00 seconds

### Candidate
type: RSZ, layer: 1, pos: 199

### Relational analysis RSZ of RS_RSZ1_RSZ1_RSZ2_RSZ1_RSZ1_RSZ2_RSZ2_RSZ1

#### Relational analysis RSZ result of RS_RSZ1_RSZ1_RSZ2_RSZ1_RSZ1_RSZ2_RSZ2_RSZ1
Status: Status.UNKNOWN
Output dim: 1, lower bound: -0.0007098, upper bound: 0.0007098
time: 0.61 seconds

### Relational analysis RSZ of RS_RSZ1_RSZ1_RSZ2_RSZ1_RSZ1_RSZ2_RSZ2_RSZ2

#### Relational analysis RSZ result of RS_RSZ1_RSZ1_RSZ2_RSZ1_RSZ1_RSZ2_RSZ2_RSZ2
Status: Status.UNKNOWN
Output dim: 1, lower bound: -0.0007098, upper bound: 0.0007098
time: 0.62 seconds

## BFS RS instance: RS_RSZ1_RSZ1_RSZ2_RSZ1_RSZ2_RSZ1_RSZ1

### Backsubstitution after applying RS history:
0: 0.0007184, 0.0011172, 0.0007184, 0.0011172, -0.0003988, 0.0003988
1: 0.9934075, 0.9943579, 0.9934075, 0.9943579, -0.0009505, 0.0009505
2: -0.0087946, -0.0055215, -0.0087946, -0.0055215, -0.0029769, 0.0029773
3: 0.0035868, 0.0041619, 0.0035868, 0.0041619, -0.0005751, 0.0005751
4: 0.0027809, 0.0053678, 0.0027809, 0.0053678, -0.0025869, 0.0025869
5: 0.0050646, 0.0064029, 0.0050646, 0.0064029, -0.0013382, 0.0013382
6: -0.0021738, -0.0008562, -0.0021738, -0.0008562, -0.0013176, 0.0013176
7: -0.0082275, -0.0074685, -0.0082275, -0.0074685, -0.0007590, 0.0007590
8: 0.0055062, 0.0098066, 0.0055062, 0.0098066, -0.0042207, 0.0042206
9: -0.0036825, -0.0031002, -0.0036825, -0.0031002, -0.0005823, 0.0005823

### Unstable ReLU Count (Linear/Conv2D Layers)
- layer_idx=0, type=LayerType.Linear, total=256, inp1_unstable=9, inp2_unstable=9, delta_unstable=249
- layer_idx=2, type=LayerType.Linear, total=256, inp1_unstable=6, inp2_unstable=6, delta_unstable=256
- layer_idx=4, type=LayerType.Linear, total=256, inp1_unstable=0, inp2_unstable=0, delta_unstable=256
- layer_idx=6, type=LayerType.Linear, total=256, inp1_unstable=2, inp2_unstable=2, delta_unstable=256
- layer_idx=8, type=LayerType.Linear, total=10, inp1_unstable=0, inp2_unstable=0, delta_unstable=10

Time for backsubstitution: 1.76 seconds

### RS candidates at layer 1
type: RSZ, layer: 1, pos: 136
type: RSZ, layer: 1, pos: 68

Time for candidate selection: 0.00 seconds

### Candidate
type: RSZ, layer: 1, pos: 136

### Relational analysis RSZ of RS_RSZ1_RSZ1_RSZ2_RSZ1_RSZ2_RSZ1_RSZ1_RSZ1

#### Relational analysis RSZ result of RS_RSZ1_RSZ1_RSZ2_RSZ1_RSZ2_RSZ1_RSZ1_RSZ1
Status: Status.UNKNOWN
Output dim: 1, lower bound: -0.0007098, upper bound: 0.0007098
time: 0.56 seconds

### Relational analysis RSZ of RS_RSZ1_RSZ1_RSZ2_RSZ1_RSZ2_RSZ1_RSZ1_RSZ2

#### Relational analysis RSZ result of RS_RSZ1_RSZ1_RSZ2_RSZ1_RSZ2_RSZ1_RSZ1_RSZ2
Status: Status.UNKNOWN
Output dim: 1, lower bound: -0.0007098, upper bound: 0.0007098
time: 0.61 seconds

## BFS RS instance: RS_RSZ1_RSZ1_RSZ2_RSZ1_RSZ2_RSZ1_RSZ2

### Backsubstitution after applying RS history:
0: 0.0007184, 0.0011172, 0.0007184, 0.0011172, -0.0003988, 0.0003988
1: 0.9934075, 0.9943579, 0.9934075, 0.9943579, -0.0009505, 0.0009505
2: -0.0087946, -0.0055215, -0.0087946, -0.0055215, -0.0029766, 0.0029774
3: 0.0035868, 0.0041619, 0.0035868, 0.0041619, -0.0005751, 0.0005751
4: 0.0027809, 0.0053678, 0.0027809, 0.0053678, -0.0025869, 0.0025869
5: 0.0050646, 0.0064029, 0.0050646, 0.0064029, -0.0013382, 0.0013382
6: -0.0021738, -0.0008562, -0.0021738, -0.0008562, -0.0013176, 0.0013176
7: -0.0082275, -0.0074685, -0.0082275, -0.0074685, -0.0007590, 0.0007590
8: 0.0055062, 0.0098066, 0.0055062, 0.0098066, -0.0042207, 0.0042205
9: -0.0036825, -0.0031002, -0.0036825, -0.0031002, -0.0005823, 0.0005823

### Unstable ReLU Count (Linear/Conv2D Layers)
- layer_idx=0, type=LayerType.Linear, total=256, inp1_unstable=9, inp2_unstable=9, delta_unstable=249
- layer_idx=2, type=LayerType.Linear, total=256, inp1_unstable=6, inp2_unstable=6, delta_unstable=256
- layer_idx=4, type=LayerType.Linear, total=256, inp1_unstable=0, inp2_unstable=0, delta_unstable=256
- layer_idx=6, type=LayerType.Linear, total=256, inp1_unstable=2, inp2_unstable=2, delta_unstable=256
- layer_idx=8, type=LayerType.Linear, total=10, inp1_unstable=0, inp2_unstable=0, delta_unstable=10

Time for backsubstitution: 1.70 seconds

### RS candidates at layer 1
type: RSZ, layer: 1, pos: 68
type: RSZ, layer: 1, pos: 136

Time for candidate selection: 0.00 seconds

### Candidate
type: RSZ, layer: 1, pos: 68

### Relational analysis RSZ of RS_RSZ1_RSZ1_RSZ2_RSZ1_RSZ2_RSZ1_RSZ2_RSZ1

#### Relational analysis RSZ result of RS_RSZ1_RSZ1_RSZ2_RSZ1_RSZ2_RSZ1_RSZ2_RSZ1
Status: Status.UNKNOWN
Output dim: 1, lower bound: -0.0007098, upper bound: 0.0007098
time: 0.61 seconds

### Relational analysis RSZ of RS_RSZ1_RSZ1_RSZ2_RSZ1_RSZ2_RSZ1_RSZ2_RSZ2

#### Relational analysis RSZ result of RS_RSZ1_RSZ1_RSZ2_RSZ1_RSZ2_RSZ1_RSZ2_RSZ2
Status: Status.UNKNOWN
Output dim: 1, lower bound: -0.0007098, upper bound: 0.0007098
time: 0.62 seconds

## BFS RS instance: RS_RSZ1_RSZ1_RSZ2_RSZ1_RSZ2_RSZ2_RSZ1

### Backsubstitution after applying RS history:
0: 0.0007184, 0.0011172, 0.0007184, 0.0011172, -0.0003988, 0.0003988
1: 0.9934075, 0.9943579, 0.9934075, 0.9943579, -0.0009505, 0.0009505
2: -0.0087946, -0.0055215, -0.0087946, -0.0055215, -0.0029730, 0.0029731
3: 0.0035868, 0.0041619, 0.0035868, 0.0041619, -0.0005751, 0.0005751
4: 0.0027809, 0.0053678, 0.0027809, 0.0053678, -0.0025869, 0.0025869
5: 0.0050646, 0.0064029, 0.0050646, 0.0064029, -0.0013382, 0.0013382
6: -0.0021738, -0.0008562, -0.0021738, -0.0008562, -0.0013176, 0.0013176
7: -0.0082275, -0.0074685, -0.0082275, -0.0074685, -0.0007590, 0.0007590
8: 0.0055062, 0.0098066, 0.0055062, 0.0098066, -0.0042190, 0.0042190
9: -0.0036825, -0.0031002, -0.0036825, -0.0031002, -0.0005823, 0.0005823

### Unstable ReLU Count (Linear/Conv2D Layers)
- layer_idx=0, type=LayerType.Linear, total=256, inp1_unstable=9, inp2_unstable=9, delta_unstable=249
- layer_idx=2, type=LayerType.Linear, total=256, inp1_unstable=6, inp2_unstable=6, delta_unstable=256
- layer_idx=4, type=LayerType.Linear, total=256, inp1_unstable=0, inp2_unstable=0, delta_unstable=256
- layer_idx=6, type=LayerType.Linear, total=256, inp1_unstable=2, inp2_unstable=2, delta_unstable=256
- layer_idx=8, type=LayerType.Linear, total=10, inp1_unstable=0, inp2_unstable=0, delta_unstable=10

Time for backsubstitution: 1.69 seconds

### RS candidates at layer 1
type: RSZ, layer: 1, pos: 83
type: RSZ, layer: 1, pos: 136

Time for candidate selection: 0.00 seconds

### Candidate
type: RSZ, layer: 1, pos: 83

### Relational analysis RSZ of RS_RSZ1_RSZ1_RSZ2_RSZ1_RSZ2_RSZ2_RSZ1_RSZ1

#### Relational analysis RSZ result of RS_RSZ1_RSZ1_RSZ2_RSZ1_RSZ2_RSZ2_RSZ1_RSZ1
Status: Status.UNKNOWN
Output dim: 1, lower bound: -0.0007098, upper bound: 0.0007098
time: 0.61 seconds

### Relational analysis RSZ of RS_RSZ1_RSZ1_RSZ2_RSZ1_RSZ2_RSZ2_RSZ1_RSZ2

#### Relational analysis RSZ result of RS_RSZ1_RSZ1_RSZ2_RSZ1_RSZ2_RSZ2_RSZ1_RSZ2
Status: Status.UNKNOWN
Output dim: 1, lower bound: -0.0007098, upper bound: 0.0007098
time: 0.61 seconds

## BFS RS instance: RS_RSZ1_RSZ1_RSZ2_RSZ1_RSZ2_RSZ2_RSZ2

### Backsubstitution after applying RS history:
0: 0.0007184, 0.0011172, 0.0007184, 0.0011172, -0.0003988, 0.0003988
1: 0.9934075, 0.9943579, 0.9934075, 0.9943579, -0.0009505, 0.0009505
2: -0.0087946, -0.0055215, -0.0087946, -0.0055215, -0.0029712, 0.0029749
3: 0.0035868, 0.0041619, 0.0035868, 0.0041619, -0.0005751, 0.0005751
4: 0.0027809, 0.0053678, 0.0027809, 0.0053678, -0.0025869, 0.0025869
5: 0.0050646, 0.0064029, 0.0050646, 0.0064029, -0.0013382, 0.0013382
6: -0.0021738, -0.0008562, -0.0021738, -0.0008562, -0.0013176, 0.0013176
7: -0.0082275, -0.0074685, -0.0082275, -0.0074685, -0.0007590, 0.0007590
8: 0.0055062, 0.0098066, 0.0055062, 0.0098066, -0.0042195, 0.0042186
9: -0.0036825, -0.0031002, -0.0036825, -0.0031002, -0.0005823, 0.0005823

### Unstable ReLU Count (Linear/Conv2D Layers)
- layer_idx=0, type=LayerType.Linear, total=256, inp1_unstable=9, inp2_unstable=9, delta_unstable=249
- layer_idx=2, type=LayerType.Linear, total=256, inp1_unstable=6, inp2_unstable=6, delta_unstable=256
- layer_idx=4, type=LayerType.Linear, total=256, inp1_unstable=0, inp2_unstable=0, delta_unstable=256
- layer_idx=6, type=LayerType.Linear, total=256, inp1_unstable=2, inp2_unstable=2, delta_unstable=256
- layer_idx=8, type=LayerType.Linear, total=10, inp1_unstable=0, inp2_unstable=0, delta_unstable=10

Time for backsubstitution: 1.85 seconds

### RS candidates at layer 1
type: RSZ, layer: 1, pos: 83
type: RSZ, layer: 1, pos: 136

Time for candidate selection: 0.00 seconds

### Candidate
type: RSZ, layer: 1, pos: 83

### Relational analysis RSZ of RS_RSZ1_RSZ1_RSZ2_RSZ1_RSZ2_RSZ2_RSZ2_RSZ1

#### Relational analysis RSZ result of RS_RSZ1_RSZ1_RSZ2_RSZ1_RSZ2_RSZ2_RSZ2_RSZ1
Status: Status.UNKNOWN
Output dim: 1, lower bound: -0.0007098, upper bound: 0.0007098
time: 0.62 seconds

### Relational analysis RSZ of RS_RSZ1_RSZ1_RSZ2_RSZ1_RSZ2_RSZ2_RSZ2_RSZ2

#### Relational analysis RSZ result of RS_RSZ1_RSZ1_RSZ2_RSZ1_RSZ2_RSZ2_RSZ2_RSZ2
Status: Status.UNKNOWN
Output dim: 1, lower bound: -0.0007098, upper bound: 0.0007098
time: 0.59 seconds

## BFS RS instance: RS_RSZ1_RSZ1_RSZ2_RSZ2_RSZ1_RSZ1_RSZ1

### Backsubstitution after applying RS history:
0: 0.0007184, 0.0011172, 0.0007184, 0.0011172, -0.0003988, 0.0003988
1: 0.9934075, 0.9943579, 0.9934075, 0.9943579, -0.0009505, 0.0009505
2: -0.0087946, -0.0055215, -0.0087946, -0.0055215, -0.0029787, 0.0029695
3: 0.0035868, 0.0041619, 0.0035868, 0.0041619, -0.0005751, 0.0005751
4: 0.0027809, 0.0053678, 0.0027809, 0.0053678, -0.0025869, 0.0025869
5: 0.0050646, 0.0064029, 0.0050646, 0.0064029, -0.0013382, 0.0013382
6: -0.0021738, -0.0008562, -0.0021738, -0.0008562, -0.0013176, 0.0013176
7: -0.0082275, -0.0074685, -0.0082275, -0.0074685, -0.0007590, 0.0007590
8: 0.0055062, 0.0098066, 0.0055062, 0.0098066, -0.0042186, 0.0042208
9: -0.0036825, -0.0031002, -0.0036825, -0.0031002, -0.0005823, 0.0005823

### Unstable ReLU Count (Linear/Conv2D Layers)
- layer_idx=0, type=LayerType.Linear, total=256, inp1_unstable=9, inp2_unstable=9, delta_unstable=249
- layer_idx=2, type=LayerType.Linear, total=256, inp1_unstable=6, inp2_unstable=6, delta_unstable=256
- layer_idx=4, type=LayerType.Linear, total=256, inp1_unstable=0, inp2_unstable=0, delta_unstable=256
- layer_idx=6, type=LayerType.Linear, total=256, inp1_unstable=2, inp2_unstable=2, delta_unstable=256
- layer_idx=8, type=LayerType.Linear, total=10, inp1_unstable=0, inp2_unstable=0, delta_unstable=10

Time for backsubstitution: 1.69 seconds

### RS candidates at layer 1
type: RSZ, layer: 1, pos: 83
type: RSZ, layer: 1, pos: 52

Time for candidate selection: 0.00 seconds

### Candidate
type: RSZ, layer: 1, pos: 83

### Relational analysis RSZ of RS_RSZ1_RSZ1_RSZ2_RSZ2_RSZ1_RSZ1_RSZ1_RSZ1

#### Relational analysis RSZ result of RS_RSZ1_RSZ1_RSZ2_RSZ2_RSZ1_RSZ1_RSZ1_RSZ1
Status: Status.UNKNOWN
Output dim: 1, lower bound: -0.0007098, upper bound: 0.0007098
time: 0.60 seconds

### Relational analysis RSZ of RS_RSZ1_RSZ1_RSZ2_RSZ2_RSZ1_RSZ1_RSZ1_RSZ2

#### Relational analysis RSZ result of RS_RSZ1_RSZ1_RSZ2_RSZ2_RSZ1_RSZ1_RSZ1_RSZ2
Status: Status.UNKNOWN
Output dim: 1, lower bound: -0.0007098, upper bound: 0.0007098
time: 0.63 seconds

## BFS RS instance: RS_RSZ1_RSZ1_RSZ2_RSZ2_RSZ1_RSZ1_RSZ2

### Backsubstitution after applying RS history:
0: 0.0007184, 0.0011172, 0.0007184, 0.0011172, -0.0003988, 0.0003988
1: 0.9934075, 0.9943579, 0.9934075, 0.9943579, -0.0009505, 0.0009505
2: -0.0087946, -0.0055215, -0.0087946, -0.0055215, -0.0029793, 0.0029716
3: 0.0035868, 0.0041619, 0.0035868, 0.0041619, -0.0005751, 0.0005751
4: 0.0027809, 0.0053678, 0.0027809, 0.0053678, -0.0025869, 0.0025869
5: 0.0050646, 0.0064029, 0.0050646, 0.0064029, -0.0013382, 0.0013382
6: -0.0021738, -0.0008562, -0.0021738, -0.0008562, -0.0013176, 0.0013176
7: -0.0082275, -0.0074685, -0.0082275, -0.0074685, -0.0007590, 0.0007590
8: 0.0055062, 0.0098066, 0.0055062, 0.0098066, -0.0042191, 0.0042210
9: -0.0036825, -0.0031002, -0.0036825, -0.0031002, -0.0005823, 0.0005823

### Unstable ReLU Count (Linear/Conv2D Layers)
- layer_idx=0, type=LayerType.Linear, total=256, inp1_unstable=9, inp2_unstable=9, delta_unstable=249
- layer_idx=2, type=LayerType.Linear, total=256, inp1_unstable=6, inp2_unstable=6, delta_unstable=256
- layer_idx=4, type=LayerType.Linear, total=256, inp1_unstable=0, inp2_unstable=0, delta_unstable=256
- layer_idx=6, type=LayerType.Linear, total=256, inp1_unstable=2, inp2_unstable=2, delta_unstable=256
- layer_idx=8, type=LayerType.Linear, total=10, inp1_unstable=0, inp2_unstable=0, delta_unstable=10

Time for backsubstitution: 1.76 seconds

### RS candidates at layer 1
type: RSZ, layer: 1, pos: 52
type: RSZ, layer: 1, pos: 83

Time for candidate selection: 0.00 seconds

### Candidate
type: RSZ, layer: 1, pos: 52

### Relational analysis RSZ of RS_RSZ1_RSZ1_RSZ2_RSZ2_RSZ1_RSZ1_RSZ2_RSZ1

#### Relational analysis RSZ result of RS_RSZ1_RSZ1_RSZ2_RSZ2_RSZ1_RSZ1_RSZ2_RSZ1
Status: Status.UNKNOWN
Output dim: 1, lower bound: -0.0007131, upper bound: 0.0007131
time: 0.63 seconds

### Relational analysis RSZ of RS_RSZ1_RSZ1_RSZ2_RSZ2_RSZ1_RSZ1_RSZ2_RSZ2

#### Relational analysis RSZ result of RS_RSZ1_RSZ1_RSZ2_RSZ2_RSZ1_RSZ1_RSZ2_RSZ2
Status: Status.UNKNOWN
Output dim: 1, lower bound: -0.0007131, upper bound: 0.0007131
time: 0.61 seconds

## BFS RS instance: RS_RSZ1_RSZ1_RSZ2_RSZ2_RSZ1_RSZ2_RSZ1

### Backsubstitution after applying RS history:
0: 0.0007184, 0.0011172, 0.0007184, 0.0011172, -0.0003988, 0.0003988
1: 0.9934075, 0.9943579, 0.9934075, 0.9943579, -0.0009505, 0.0009505
2: -0.0087946, -0.0055215, -0.0087946, -0.0055215, -0.0029807, 0.0029662
3: 0.0035868, 0.0041619, 0.0035868, 0.0041619, -0.0005751, 0.0005751
4: 0.0027809, 0.0053678, 0.0027809, 0.0053678, -0.0025869, 0.0025869
5: 0.0050646, 0.0064029, 0.0050646, 0.0064029, -0.0013382, 0.0013382
6: -0.0021738, -0.0008562, -0.0021738, -0.0008562, -0.0013176, 0.0013176
7: -0.0082275, -0.0074685, -0.0082275, -0.0074685, -0.0007590, 0.0007590
8: 0.0055062, 0.0098066, 0.0055062, 0.0098066, -0.0042178, 0.0042214
9: -0.0036825, -0.0031002, -0.0036825, -0.0031002, -0.0005823, 0.0005823

### Unstable ReLU Count (Linear/Conv2D Layers)
- layer_idx=0, type=LayerType.Linear, total=256, inp1_unstable=9, inp2_unstable=9, delta_unstable=249
- layer_idx=2, type=LayerType.Linear, total=256, inp1_unstable=6, inp2_unstable=6, delta_unstable=256
- layer_idx=4, type=LayerType.Linear, total=256, inp1_unstable=0, inp2_unstable=0, delta_unstable=256
- layer_idx=6, type=LayerType.Linear, total=256, inp1_unstable=2, inp2_unstable=2, delta_unstable=256
- layer_idx=8, type=LayerType.Linear, total=10, inp1_unstable=0, inp2_unstable=0, delta_unstable=10

Time for backsubstitution: 1.70 seconds

### RS candidates at layer 1
type: RSZ, layer: 1, pos: 199
type: RSZ, layer: 1, pos: 52

Time for candidate selection: 0.00 seconds

### Candidate
type: RSZ, layer: 1, pos: 199

### Relational analysis RSZ of RS_RSZ1_RSZ1_RSZ2_RSZ2_RSZ1_RSZ2_RSZ1_RSZ1

#### Relational analysis RSZ result of RS_RSZ1_RSZ1_RSZ2_RSZ2_RSZ1_RSZ2_RSZ1_RSZ1
Status: Status.UNKNOWN
Output dim: 1, lower bound: -0.0007097, upper bound: 0.0007098
time: 0.65 seconds

### Relational analysis RSZ of RS_RSZ1_RSZ1_RSZ2_RSZ2_RSZ1_RSZ2_RSZ1_RSZ2

#### Relational analysis RSZ result of RS_RSZ1_RSZ1_RSZ2_RSZ2_RSZ1_RSZ2_RSZ1_RSZ2
Status: Status.UNKNOWN
Output dim: 1, lower bound: -0.0007097, upper bound: 0.0007098
time: 0.63 seconds

## BFS RS instance: RS_RSZ1_RSZ1_RSZ2_RSZ2_RSZ1_RSZ2_RSZ2

### Backsubstitution after applying RS history:
0: 0.0007184, 0.0011172, 0.0007184, 0.0011172, -0.0003988, 0.0003988
1: 0.9934075, 0.9943579, 0.9934075, 0.9943579, -0.0009505, 0.0009505
2: -0.0087946, -0.0055215, -0.0087946, -0.0055215, -0.0029804, 0.0029664
3: 0.0035868, 0.0041619, 0.0035868, 0.0041619, -0.0005751, 0.0005751
4: 0.0027809, 0.0053678, 0.0027809, 0.0053678, -0.0025869, 0.0025869
5: 0.0050646, 0.0064029, 0.0050646, 0.0064029, -0.0013382, 0.0013382
6: -0.0021738, -0.0008562, -0.0021738, -0.0008562, -0.0013176, 0.0013176
7: -0.0082275, -0.0074685, -0.0082275, -0.0074685, -0.0007590, 0.0007590
8: 0.0055062, 0.0098066, 0.0055062, 0.0098066, -0.0042179, 0.0042213
9: -0.0036825, -0.0031002, -0.0036825, -0.0031002, -0.0005823, 0.0005823

### Unstable ReLU Count (Linear/Conv2D Layers)
- layer_idx=0, type=LayerType.Linear, total=256, inp1_unstable=9, inp2_unstable=9, delta_unstable=249
- layer_idx=2, type=LayerType.Linear, total=256, inp1_unstable=6, inp2_unstable=6, delta_unstable=256
- layer_idx=4, type=LayerType.Linear, total=256, inp1_unstable=0, inp2_unstable=0, delta_unstable=256
- layer_idx=6, type=LayerType.Linear, total=256, inp1_unstable=2, inp2_unstable=2, delta_unstable=256
- layer_idx=8, type=LayerType.Linear, total=10, inp1_unstable=0, inp2_unstable=0, delta_unstable=10

Time for backsubstitution: 1.73 seconds

### RS candidates at layer 1
type: RSZ, layer: 1, pos: 52
type: RSZ, layer: 1, pos: 199

Time for candidate selection: 0.00 seconds

### Candidate
type: RSZ, layer: 1, pos: 52

### Relational analysis RSZ of RS_RSZ1_RSZ1_RSZ2_RSZ2_RSZ1_RSZ2_RSZ2_RSZ1

#### Relational analysis RSZ result of RS_RSZ1_RSZ1_RSZ2_RSZ2_RSZ1_RSZ2_RSZ2_RSZ1
Status: Status.UNKNOWN
Output dim: 1, lower bound: -0.0007685, upper bound: 0.0007698
time: 0.71 seconds

### Relational analysis RSZ of RS_RSZ1_RSZ1_RSZ2_RSZ2_RSZ1_RSZ2_RSZ2_RSZ2

#### Relational analysis RSZ result of RS_RSZ1_RSZ1_RSZ2_RSZ2_RSZ1_RSZ2_RSZ2_RSZ2
Status: Status.UNKNOWN
Output dim: 1, lower bound: -0.0007685, upper bound: 0.0007700
time: 0.64 seconds

## BFS RS instance: RS_RSZ1_RSZ1_RSZ2_RSZ2_RSZ2_RSZ1_RSZ1

### Backsubstitution after applying RS history:
0: 0.0007184, 0.0011172, 0.0007184, 0.0011172, -0.0003988, 0.0003988
1: 0.9934075, 0.9943579, 0.9934075, 0.9943579, -0.0009505, 0.0009505
2: -0.0087946, -0.0055215, -0.0087946, -0.0055215, -0.0029800, 0.0029722
3: 0.0035868, 0.0041619, 0.0035868, 0.0041619, -0.0005751, 0.0005751
4: 0.0027809, 0.0053678, 0.0027809, 0.0053678, -0.0025869, 0.0025869
5: 0.0050646, 0.0064029, 0.0050646, 0.0064029, -0.0013382, 0.0013382
6: -0.0021738, -0.0008562, -0.0021738, -0.0008562, -0.0013176, 0.0013176
7: -0.0082275, -0.0074685, -0.0082275, -0.0074685, -0.0007590, 0.0007590
8: 0.0055062, 0.0098066, 0.0055062, 0.0098066, -0.0042189, 0.0042207
9: -0.0036825, -0.0031002, -0.0036825, -0.0031002, -0.0005823, 0.0005823

### Unstable ReLU Count (Linear/Conv2D Layers)
- layer_idx=0, type=LayerType.Linear, total=256, inp1_unstable=9, inp2_unstable=9, delta_unstable=249
- layer_idx=2, type=LayerType.Linear, total=256, inp1_unstable=6, inp2_unstable=6, delta_unstable=256
- layer_idx=4, type=LayerType.Linear, total=256, inp1_unstable=0, inp2_unstable=0, delta_unstable=256
- layer_idx=6, type=LayerType.Linear, total=256, inp1_unstable=2, inp2_unstable=2, delta_unstable=256
- layer_idx=8, type=LayerType.Linear, total=10, inp1_unstable=0, inp2_unstable=0, delta_unstable=10

Time for backsubstitution: 1.72 seconds

### RS candidates at layer 1
type: RSZ, layer: 1, pos: 199
type: RSZ, layer: 1, pos: 136

Time for candidate selection: 0.00 seconds

### Candidate
type: RSZ, layer: 1, pos: 199

### Relational analysis RSZ of RS_RSZ1_RSZ1_RSZ2_RSZ2_RSZ2_RSZ1_RSZ1_RSZ1

#### Relational analysis RSZ result of RS_RSZ1_RSZ1_RSZ2_RSZ2_RSZ2_RSZ1_RSZ1_RSZ1
Status: Status.UNKNOWN
Output dim: 1, lower bound: -0.0007098, upper bound: 0.0007098
time: 0.61 seconds

### Relational analysis RSZ of RS_RSZ1_RSZ1_RSZ2_RSZ2_RSZ2_RSZ1_RSZ1_RSZ2

#### Relational analysis RSZ result of RS_RSZ1_RSZ1_RSZ2_RSZ2_RSZ2_RSZ1_RSZ1_RSZ2
Status: Status.UNKNOWN
Output dim: 1, lower bound: -0.0007098, upper bound: 0.0007098
time: 0.61 seconds

## BFS RS instance: RS_RSZ1_RSZ1_RSZ2_RSZ2_RSZ2_RSZ1_RSZ2

### Backsubstitution after applying RS history:
0: 0.0007184, 0.0011172, 0.0007184, 0.0011172, -0.0003988, 0.0003988
1: 0.9934075, 0.9943579, 0.9934075, 0.9943579, -0.0009505, 0.0009505
2: -0.0087946, -0.0055215, -0.0087946, -0.0055215, -0.0029796, 0.0029724
3: 0.0035868, 0.0041619, 0.0035868, 0.0041619, -0.0005751, 0.0005751
4: 0.0027809, 0.0053678, 0.0027809, 0.0053678, -0.0025869, 0.0025869
5: 0.0050646, 0.0064029, 0.0050646, 0.0064029, -0.0013382, 0.0013382
6: -0.0021738, -0.0008562, -0.0021738, -0.0008562, -0.0013176, 0.0013176
7: -0.0082275, -0.0074685, -0.0082275, -0.0074685, -0.0007590, 0.0007590
8: 0.0055062, 0.0098066, 0.0055062, 0.0098066, -0.0042189, 0.0042206
9: -0.0036825, -0.0031002, -0.0036825, -0.0031002, -0.0005823, 0.0005823

### Unstable ReLU Count (Linear/Conv2D Layers)
- layer_idx=0, type=LayerType.Linear, total=256, inp1_unstable=9, inp2_unstable=9, delta_unstable=249
- layer_idx=2, type=LayerType.Linear, total=256, inp1_unstable=6, inp2_unstable=6, delta_unstable=256
- layer_idx=4, type=LayerType.Linear, total=256, inp1_unstable=0, inp2_unstable=0, delta_unstable=256
- layer_idx=6, type=LayerType.Linear, total=256, inp1_unstable=2, inp2_unstable=2, delta_unstable=256
- layer_idx=8, type=LayerType.Linear, total=10, inp1_unstable=0, inp2_unstable=0, delta_unstable=10

Time for backsubstitution: 1.69 seconds

### RS candidates at layer 1
type: RSZ, layer: 1, pos: 199
type: RSZ, layer: 1, pos: 136

Time for candidate selection: 0.00 seconds

### Candidate
type: RSZ, layer: 1, pos: 199

### Relational analysis RSZ of RS_RSZ1_RSZ1_RSZ2_RSZ2_RSZ2_RSZ1_RSZ2_RSZ1

#### Relational analysis RSZ result of RS_RSZ1_RSZ1_RSZ2_RSZ2_RSZ2_RSZ1_RSZ2_RSZ1
Status: Status.UNKNOWN
Output dim: 1, lower bound: -0.0007098, upper bound: 0.0007098
time: 0.62 seconds

### Relational analysis RSZ of RS_RSZ1_RSZ1_RSZ2_RSZ2_RSZ2_RSZ1_RSZ2_RSZ2

#### Relational analysis RSZ result of RS_RSZ1_RSZ1_RSZ2_RSZ2_RSZ2_RSZ1_RSZ2_RSZ2
Status: Status.UNKNOWN
Output dim: 1, lower bound: -0.0007098, upper bound: 0.0007098
time: 0.61 seconds

## BFS RS instance: RS_RSZ1_RSZ1_RSZ2_RSZ2_RSZ2_RSZ2_RSZ1

### Backsubstitution after applying RS history:
0: 0.0007184, 0.0011172, 0.0007184, 0.0011172, -0.0003988, 0.0003988
1: 0.9934075, 0.9943579, 0.9934075, 0.9943579, -0.0009505, 0.0009505
2: -0.0087946, -0.0055215, -0.0087946, -0.0055215, -0.0029811, 0.0029722
3: 0.0035868, 0.0041619, 0.0035868, 0.0041619, -0.0005751, 0.0005751
4: 0.0027809, 0.0053678, 0.0027809, 0.0053678, -0.0025869, 0.0025869
5: 0.0050646, 0.0064029, 0.0050646, 0.0064029, -0.0013382, 0.0013382
6: -0.0021738, -0.0008562, -0.0021738, -0.0008562, -0.0013176, 0.0013176
7: -0.0082275, -0.0074685, -0.0082275, -0.0074685, -0.0007590, 0.0007590
8: 0.0055062, 0.0098066, 0.0055062, 0.0098066, -0.0042188, 0.0042209
9: -0.0036825, -0.0031002, -0.0036825, -0.0031002, -0.0005823, 0.0005823

### Unstable ReLU Count (Linear/Conv2D Layers)
- layer_idx=0, type=LayerType.Linear, total=256, inp1_unstable=9, inp2_unstable=9, delta_unstable=249
- layer_idx=2, type=LayerType.Linear, total=256, inp1_unstable=6, inp2_unstable=6, delta_unstable=256
- layer_idx=4, type=LayerType.Linear, total=256, inp1_unstable=0, inp2_unstable=0, delta_unstable=256
- layer_idx=6, type=LayerType.Linear, total=256, inp1_unstable=2, inp2_unstable=2, delta_unstable=256
- layer_idx=8, type=LayerType.Linear, total=10, inp1_unstable=0, inp2_unstable=0, delta_unstable=10

Time for backsubstitution: 1.67 seconds

### RS candidates at layer 1
type: RSZ, layer: 1, pos: 136
type: RSZ, layer: 1, pos: 83

Time for candidate selection: 0.00 seconds

### Candidate
type: RSZ, layer: 1, pos: 136

### Relational analysis RSZ of RS_RSZ1_RSZ1_RSZ2_RSZ2_RSZ2_RSZ2_RSZ1_RSZ1

#### Relational analysis RSZ result of RS_RSZ1_RSZ1_RSZ2_RSZ2_RSZ2_RSZ2_RSZ1_RSZ1
Status: Status.UNKNOWN
Output dim: 1, lower bound: -0.0007131, upper bound: 0.0007131
time: 0.61 seconds

### Relational analysis RSZ of RS_RSZ1_RSZ1_RSZ2_RSZ2_RSZ2_RSZ2_RSZ1_RSZ2

#### Relational analysis RSZ result of RS_RSZ1_RSZ1_RSZ2_RSZ2_RSZ2_RSZ2_RSZ1_RSZ2
Status: Status.UNKNOWN
Output dim: 1, lower bound: -0.0007131, upper bound: 0.0007131
time: 0.62 seconds

## BFS RS instance: RS_RSZ1_RSZ1_RSZ2_RSZ2_RSZ2_RSZ2_RSZ2

### Backsubstitution after applying RS history:
0: 0.0007184, 0.0011172, 0.0007184, 0.0011172, -0.0003988, 0.0003988
1: 0.9934075, 0.9943579, 0.9934075, 0.9943579, -0.0009505, 0.0009505
2: -0.0087946, -0.0055215, -0.0087946, -0.0055215, -0.0029818, 0.0029743
3: 0.0035868, 0.0041619, 0.0035868, 0.0041619, -0.0005751, 0.0005751
4: 0.0027809, 0.0053678, 0.0027809, 0.0053678, -0.0025869, 0.0025869
5: 0.0050646, 0.0064029, 0.0050646, 0.0064029, -0.0013382, 0.0013382
6: -0.0021738, -0.0008562, -0.0021738, -0.0008562, -0.0013176, 0.0013176
7: -0.0082275, -0.0074685, -0.0082275, -0.0074685, -0.0007590, 0.0007590
8: 0.0055062, 0.0098066, 0.0055062, 0.0098066, -0.0042194, 0.0042211
9: -0.0036825, -0.0031002, -0.0036825, -0.0031002, -0.0005823, 0.0005823

### Unstable ReLU Count (Linear/Conv2D Layers)
- layer_idx=0, type=LayerType.Linear, total=256, inp1_unstable=9, inp2_unstable=9, delta_unstable=249
- layer_idx=2, type=LayerType.Linear, total=256, inp1_unstable=6, inp2_unstable=6, delta_unstable=256
- layer_idx=4, type=LayerType.Linear, total=256, inp1_unstable=0, inp2_unstable=0, delta_unstable=256
- layer_idx=6, type=LayerType.Linear, total=256, inp1_unstable=2, inp2_unstable=2, delta_unstable=256
- layer_idx=8, type=LayerType.Linear, total=10, inp1_unstable=0, inp2_unstable=0, delta_unstable=10

Time for backsubstitution: 1.74 seconds

### RS candidates at layer 1
type: RSZ, layer: 1, pos: 83
type: RSZ, layer: 1, pos: 136

Time for candidate selection: 0.00 seconds

### Candidate
type: RSZ, layer: 1, pos: 83

### Relational analysis RSZ of RS_RSZ1_RSZ1_RSZ2_RSZ2_RSZ2_RSZ2_RSZ2_RSZ1

#### Relational analysis RSZ result of RS_RSZ1_RSZ1_RSZ2_RSZ2_RSZ2_RSZ2_RSZ2_RSZ1
Status: Status.UNKNOWN
Output dim: 1, lower bound: -0.0007098, upper bound: 0.0007098
time: 0.67 seconds

### Relational analysis RSZ of RS_RSZ1_RSZ1_RSZ2_RSZ2_RSZ2_RSZ2_RSZ2_RSZ2

#### Relational analysis RSZ result of RS_RSZ1_RSZ1_RSZ2_RSZ2_RSZ2_RSZ2_RSZ2_RSZ2
Status: Status.UNKNOWN
Output dim: 1, lower bound: -0.0007098, upper bound: 0.0007098
time: 0.66 seconds

## BFS RS instance: RS_RSZ1_RSZ2_RSZ1_RSZ1_RSZ1_RSZ1_RSZ1

### Backsubstitution after applying RS history:
0: 0.0007184, 0.0011172, 0.0007184, 0.0011172, -0.0003988, 0.0003988
1: 0.9934075, 0.9943579, 0.9934075, 0.9943579, -0.0009505, 0.0009505
2: -0.0087946, -0.0055215, -0.0087946, -0.0055215, -0.0029700, 0.0029726
3: 0.0035868, 0.0041619, 0.0035868, 0.0041619, -0.0005751, 0.0005751
4: 0.0027809, 0.0053678, 0.0027809, 0.0053678, -0.0025869, 0.0025869
5: 0.0050646, 0.0064029, 0.0050646, 0.0064029, -0.0013382, 0.0013382
6: -0.0021738, -0.0008562, -0.0021738, -0.0008562, -0.0013176, 0.0013176
7: -0.0082275, -0.0074685, -0.0082275, -0.0074685, -0.0007590, 0.0007590
8: 0.0055062, 0.0098066, 0.0055062, 0.0098066, -0.0042194, 0.0042187
9: -0.0036825, -0.0031002, -0.0036825, -0.0031002, -0.0005823, 0.0005823

### Unstable ReLU Count (Linear/Conv2D Layers)
- layer_idx=0, type=LayerType.Linear, total=256, inp1_unstable=9, inp2_unstable=9, delta_unstable=249
- layer_idx=2, type=LayerType.Linear, total=256, inp1_unstable=6, inp2_unstable=6, delta_unstable=256
- layer_idx=4, type=LayerType.Linear, total=256, inp1_unstable=0, inp2_unstable=0, delta_unstable=256
- layer_idx=6, type=LayerType.Linear, total=256, inp1_unstable=2, inp2_unstable=2, delta_unstable=256
- layer_idx=8, type=LayerType.Linear, total=10, inp1_unstable=0, inp2_unstable=0, delta_unstable=10

Time for backsubstitution: 1.92 seconds

### RS candidates at layer 1
type: RSZ, layer: 1, pos: 199
type: RSZ, layer: 1, pos: 204

Time for candidate selection: 0.00 seconds

### Candidate
type: RSZ, layer: 1, pos: 199

### Relational analysis RSZ of RS_RSZ1_RSZ2_RSZ1_RSZ1_RSZ1_RSZ1_RSZ1_RSZ1

#### Relational analysis RSZ result of RS_RSZ1_RSZ2_RSZ1_RSZ1_RSZ1_RSZ1_RSZ1_RSZ1
Status: Status.UNKNOWN
Output dim: 1, lower bound: -0.0007098, upper bound: 0.0007098
time: 0.63 seconds

### Relational analysis RSZ of RS_RSZ1_RSZ2_RSZ1_RSZ1_RSZ1_RSZ1_RSZ1_RSZ2

#### Relational analysis RSZ result of RS_RSZ1_RSZ2_RSZ1_RSZ1_RSZ1_RSZ1_RSZ1_RSZ2
Status: Status.UNKNOWN
Output dim: 1, lower bound: -0.0007098, upper bound: 0.0007098
time: 0.63 seconds

## BFS RS instance: RS_RSZ1_RSZ2_RSZ1_RSZ1_RSZ1_RSZ1_RSZ2

### Backsubstitution after applying RS history:
0: 0.0007184, 0.0011172, 0.0007184, 0.0011172, -0.0003988, 0.0003988
1: 0.9934075, 0.9943579, 0.9934075, 0.9943579, -0.0009505, 0.0009505
2: -0.0087946, -0.0055215, -0.0087946, -0.0055215, -0.0029685, 0.0029744
3: 0.0035868, 0.0041619, 0.0035868, 0.0041619, -0.0005751, 0.0005751
4: 0.0027809, 0.0053678, 0.0027809, 0.0053678, -0.0025869, 0.0025869
5: 0.0050646, 0.0064029, 0.0050646, 0.0064029, -0.0013382, 0.0013382
6: -0.0021738, -0.0008562, -0.0021738, -0.0008562, -0.0013176, 0.0013176
7: -0.0082275, -0.0074685, -0.0082275, -0.0074685, -0.0007590, 0.0007590
8: 0.0055062, 0.0098066, 0.0055062, 0.0098066, -0.0042199, 0.0042183
9: -0.0036825, -0.0031002, -0.0036825, -0.0031002, -0.0005823, 0.0005823

### Unstable ReLU Count (Linear/Conv2D Layers)
- layer_idx=0, type=LayerType.Linear, total=256, inp1_unstable=9, inp2_unstable=9, delta_unstable=249
- layer_idx=2, type=LayerType.Linear, total=256, inp1_unstable=6, inp2_unstable=6, delta_unstable=256
- layer_idx=4, type=LayerType.Linear, total=256, inp1_unstable=0, inp2_unstable=0, delta_unstable=256
- layer_idx=6, type=LayerType.Linear, total=256, inp1_unstable=2, inp2_unstable=2, delta_unstable=256
- layer_idx=8, type=LayerType.Linear, total=10, inp1_unstable=0, inp2_unstable=0, delta_unstable=10

Time for backsubstitution: 1.89 seconds

### RS candidates at layer 1
type: RSZ, layer: 1, pos: 199
type: RSZ, layer: 1, pos: 204

Time for candidate selection: 0.00 seconds

### Candidate
type: RSZ, layer: 1, pos: 199

### Relational analysis RSZ of RS_RSZ1_RSZ2_RSZ1_RSZ1_RSZ1_RSZ1_RSZ2_RSZ1

#### Relational analysis RSZ result of RS_RSZ1_RSZ2_RSZ1_RSZ1_RSZ1_RSZ1_RSZ2_RSZ1
Status: Status.UNKNOWN
Output dim: 1, lower bound: -0.0007098, upper bound: 0.0007098
time: 0.62 seconds

### Relational analysis RSZ of RS_RSZ1_RSZ2_RSZ1_RSZ1_RSZ1_RSZ1_RSZ2_RSZ2

#### Relational analysis RSZ result of RS_RSZ1_RSZ2_RSZ1_RSZ1_RSZ1_RSZ1_RSZ2_RSZ2
Status: Status.UNKNOWN
Output dim: 1, lower bound: -0.0007098, upper bound: 0.0007098
time: 0.61 seconds

## BFS RS instance: RS_RSZ1_RSZ2_RSZ1_RSZ1_RSZ1_RSZ2_RSZ1

### Backsubstitution after applying RS history:
0: 0.0007184, 0.0011172, 0.0007184, 0.0011172, -0.0003988, 0.0003988
1: 0.9934075, 0.9943579, 0.9934075, 0.9943579, -0.0009505, 0.0009505
2: -0.0087946, -0.0055215, -0.0087946, -0.0055215, -0.0029744, 0.0029800
3: 0.0035868, 0.0041619, 0.0035868, 0.0041619, -0.0005751, 0.0005751
4: 0.0027809, 0.0053678, 0.0027809, 0.0053678, -0.0025869, 0.0025869
5: 0.0050646, 0.0064029, 0.0050646, 0.0064029, -0.0013382, 0.0013382
6: -0.0021738, -0.0008562, -0.0021738, -0.0008562, -0.0013176, 0.0013176
7: -0.0082275, -0.0074685, -0.0082275, -0.0074685, -0.0007590, 0.0007590
8: 0.0055062, 0.0098066, 0.0055062, 0.0098066, -0.0042218, 0.0042203
9: -0.0036825, -0.0031002, -0.0036825, -0.0031002, -0.0005823, 0.0005823

### Unstable ReLU Count (Linear/Conv2D Layers)
- layer_idx=0, type=LayerType.Linear, total=256, inp1_unstable=9, inp2_unstable=9, delta_unstable=249
- layer_idx=2, type=LayerType.Linear, total=256, inp1_unstable=6, inp2_unstable=6, delta_unstable=256
- layer_idx=4, type=LayerType.Linear, total=256, inp1_unstable=0, inp2_unstable=0, delta_unstable=256
- layer_idx=6, type=LayerType.Linear, total=256, inp1_unstable=2, inp2_unstable=2, delta_unstable=256
- layer_idx=8, type=LayerType.Linear, total=10, inp1_unstable=0, inp2_unstable=0, delta_unstable=10

Time for backsubstitution: 1.81 seconds

### RS candidates at layer 1
type: RSZ, layer: 1, pos: 204
type: RSZ, layer: 1, pos: 68

Time for candidate selection: 0.00 seconds

### Candidate
type: RSZ, layer: 1, pos: 204

### Relational analysis RSZ of RS_RSZ1_RSZ2_RSZ1_RSZ1_RSZ1_RSZ2_RSZ1_RSZ1

#### Relational analysis RSZ result of RS_RSZ1_RSZ2_RSZ1_RSZ1_RSZ1_RSZ2_RSZ1_RSZ1
Status: Status.UNKNOWN
Output dim: 1, lower bound: -0.0007098, upper bound: 0.0007098
time: 0.62 seconds

### Relational analysis RSZ of RS_RSZ1_RSZ2_RSZ1_RSZ1_RSZ1_RSZ2_RSZ1_RSZ2

#### Relational analysis RSZ result of RS_RSZ1_RSZ2_RSZ1_RSZ1_RSZ1_RSZ2_RSZ1_RSZ2
Status: Status.UNKNOWN
Output dim: 1, lower bound: -0.0007098, upper bound: 0.0007098
time: 0.63 seconds

## BFS RS instance: RS_RSZ1_RSZ2_RSZ1_RSZ1_RSZ1_RSZ2_RSZ2

### Backsubstitution after applying RS history:
0: 0.0007184, 0.0011172, 0.0007184, 0.0011172, -0.0003988, 0.0003988
1: 0.9934075, 0.9943579, 0.9934075, 0.9943579, -0.0009505, 0.0009505
2: -0.0087946, -0.0055215, -0.0087946, -0.0055215, -0.0029753, 0.0029820
3: 0.0035868, 0.0041619, 0.0035868, 0.0041619, -0.0005751, 0.0005751
4: 0.0027809, 0.0053678, 0.0027809, 0.0053678, -0.0025869, 0.0025869
5: 0.0050646, 0.0064029, 0.0050646, 0.0064029, -0.0013382, 0.0013382
6: -0.0021738, -0.0008562, -0.0021738, -0.0008562, -0.0013176, 0.0013176
7: -0.0082275, -0.0074685, -0.0082275, -0.0074685, -0.0007590, 0.0007590
8: 0.0055062, 0.0098066, 0.0055062, 0.0098066, -0.0042223, 0.0042205
9: -0.0036825, -0.0031002, -0.0036825, -0.0031002, -0.0005823, 0.0005823

### Unstable ReLU Count (Linear/Conv2D Layers)
- layer_idx=0, type=LayerType.Linear, total=256, inp1_unstable=9, inp2_unstable=9, delta_unstable=249
- layer_idx=2, type=LayerType.Linear, total=256, inp1_unstable=6, inp2_unstable=6, delta_unstable=256
- layer_idx=4, type=LayerType.Linear, total=256, inp1_unstable=0, inp2_unstable=0, delta_unstable=256
- layer_idx=6, type=LayerType.Linear, total=256, inp1_unstable=2, inp2_unstable=2, delta_unstable=256
- layer_idx=8, type=LayerType.Linear, total=10, inp1_unstable=0, inp2_unstable=0, delta_unstable=10

Time for backsubstitution: 1.77 seconds

### RS candidates at layer 1
type: RSZ, layer: 1, pos: 68
type: RSZ, layer: 1, pos: 204

Time for candidate selection: 0.00 seconds

### Candidate
type: RSZ, layer: 1, pos: 68

### Relational analysis RSZ of RS_RSZ1_RSZ2_RSZ1_RSZ1_RSZ1_RSZ2_RSZ2_RSZ1

#### Relational analysis RSZ result of RS_RSZ1_RSZ2_RSZ1_RSZ1_RSZ1_RSZ2_RSZ2_RSZ1
Status: Status.UNKNOWN
Output dim: 1, lower bound: -0.0007098, upper bound: 0.0007098
time: 0.58 seconds

### Relational analysis RSZ of RS_RSZ1_RSZ2_RSZ1_RSZ1_RSZ1_RSZ2_RSZ2_RSZ2

#### Relational analysis RSZ result of RS_RSZ1_RSZ2_RSZ1_RSZ1_RSZ1_RSZ2_RSZ2_RSZ2
Status: Status.UNKNOWN
Output dim: 1, lower bound: -0.0007098, upper bound: 0.0007098
time: 0.61 seconds

## BFS RS instance: RS_RSZ1_RSZ2_RSZ1_RSZ1_RSZ2_RSZ1_RSZ1

### Backsubstitution after applying RS history:
0: 0.0007184, 0.0011172, 0.0007184, 0.0011172, -0.0003988, 0.0003988
1: 0.9934075, 0.9943579, 0.9934075, 0.9943579, -0.0009505, 0.0009505
2: -0.0087946, -0.0055215, -0.0087946, -0.0055215, -0.0029683, 0.0029686
3: 0.0035868, 0.0041619, 0.0035868, 0.0041619, -0.0005751, 0.0005751
4: 0.0027809, 0.0053678, 0.0027809, 0.0053678, -0.0025869, 0.0025869
5: 0.0050646, 0.0064029, 0.0050646, 0.0064029, -0.0013382, 0.0013382
6: -0.0021738, -0.0008562, -0.0021738, -0.0008562, -0.0013176, 0.0013176
7: -0.0082275, -0.0074685, -0.0082275, -0.0074685, -0.0007590, 0.0007590
8: 0.0055062, 0.0098066, 0.0055062, 0.0098066, -0.0042183, 0.0042182
9: -0.0036825, -0.0031002, -0.0036825, -0.0031002, -0.0005823, 0.0005823

### Unstable ReLU Count (Linear/Conv2D Layers)
- layer_idx=0, type=LayerType.Linear, total=256, inp1_unstable=9, inp2_unstable=9, delta_unstable=249
- layer_idx=2, type=LayerType.Linear, total=256, inp1_unstable=6, inp2_unstable=6, delta_unstable=256
- layer_idx=4, type=LayerType.Linear, total=256, inp1_unstable=0, inp2_unstable=0, delta_unstable=256
- layer_idx=6, type=LayerType.Linear, total=256, inp1_unstable=2, inp2_unstable=2, delta_unstable=256
- layer_idx=8, type=LayerType.Linear, total=10, inp1_unstable=0, inp2_unstable=0, delta_unstable=10

Time for backsubstitution: 1.71 seconds

### RS candidates at layer 1
type: RSZ, layer: 1, pos: 199
type: RSZ, layer: 1, pos: 83

Time for candidate selection: 0.00 seconds

### Candidate
type: RSZ, layer: 1, pos: 199

### Relational analysis RSZ of RS_RSZ1_RSZ2_RSZ1_RSZ1_RSZ2_RSZ1_RSZ1_RSZ1

#### Relational analysis RSZ result of RS_RSZ1_RSZ2_RSZ1_RSZ1_RSZ2_RSZ1_RSZ1_RSZ1
Status: Status.UNKNOWN
Output dim: 1, lower bound: -0.0007131, upper bound: 0.0007131
time: 0.63 seconds

### Relational analysis RSZ of RS_RSZ1_RSZ2_RSZ1_RSZ1_RSZ2_RSZ1_RSZ1_RSZ2

#### Relational analysis RSZ result of RS_RSZ1_RSZ2_RSZ1_RSZ1_RSZ2_RSZ1_RSZ1_RSZ2
Status: Status.UNKNOWN
Output dim: 1, lower bound: -0.0007131, upper bound: 0.0007131
time: 0.64 seconds

## BFS RS instance: RS_RSZ1_RSZ2_RSZ1_RSZ1_RSZ2_RSZ1_RSZ2

### Backsubstitution after applying RS history:
0: 0.0007184, 0.0011172, 0.0007184, 0.0011172, -0.0003988, 0.0003988
1: 0.9934075, 0.9943579, 0.9934075, 0.9943579, -0.0009505, 0.0009505
2: -0.0087946, -0.0055215, -0.0087946, -0.0055215, -0.0029671, 0.0029704
3: 0.0035868, 0.0041619, 0.0035868, 0.0041619, -0.0005751, 0.0005751
4: 0.0027809, 0.0053678, 0.0027809, 0.0053678, -0.0025869, 0.0025869
5: 0.0050646, 0.0064029, 0.0050646, 0.0064029, -0.0013382, 0.0013382
6: -0.0021738, -0.0008562, -0.0021738, -0.0008562, -0.0013176, 0.0013176
7: -0.0082275, -0.0074685, -0.0082275, -0.0074685, -0.0007590, 0.0007590
8: 0.0055062, 0.0098066, 0.0055062, 0.0098066, -0.0042188, 0.0042179
9: -0.0036825, -0.0031002, -0.0036825, -0.0031002, -0.0005823, 0.0005823

### Unstable ReLU Count (Linear/Conv2D Layers)
- layer_idx=0, type=LayerType.Linear, total=256, inp1_unstable=9, inp2_unstable=9, delta_unstable=249
- layer_idx=2, type=LayerType.Linear, total=256, inp1_unstable=6, inp2_unstable=6, delta_unstable=256
- layer_idx=4, type=LayerType.Linear, total=256, inp1_unstable=0, inp2_unstable=0, delta_unstable=256
- layer_idx=6, type=LayerType.Linear, total=256, inp1_unstable=2, inp2_unstable=2, delta_unstable=256
- layer_idx=8, type=LayerType.Linear, total=10, inp1_unstable=0, inp2_unstable=0, delta_unstable=10

Time for backsubstitution: 1.81 seconds

### RS candidates at layer 1
type: RSZ, layer: 1, pos: 199
type: RSZ, layer: 1, pos: 83

Time for candidate selection: 0.00 seconds

### Candidate
type: RSZ, layer: 1, pos: 199

### Relational analysis RSZ of RS_RSZ1_RSZ2_RSZ1_RSZ1_RSZ2_RSZ1_RSZ2_RSZ1

#### Relational analysis RSZ result of RS_RSZ1_RSZ2_RSZ1_RSZ1_RSZ2_RSZ1_RSZ2_RSZ1
Status: Status.UNKNOWN
Output dim: 1, lower bound: -0.0007131, upper bound: 0.0007131
time: 0.62 seconds

### Relational analysis RSZ of RS_RSZ1_RSZ2_RSZ1_RSZ1_RSZ2_RSZ1_RSZ2_RSZ2

#### Relational analysis RSZ result of RS_RSZ1_RSZ2_RSZ1_RSZ1_RSZ2_RSZ1_RSZ2_RSZ2
Status: Status.UNKNOWN
Output dim: 1, lower bound: -0.0007131, upper bound: 0.0007131
time: 0.63 seconds

## BFS RS instance: RS_RSZ1_RSZ2_RSZ1_RSZ1_RSZ2_RSZ2_RSZ1

### Backsubstitution after applying RS history:
0: 0.0007184, 0.0011172, 0.0007184, 0.0011172, -0.0003988, 0.0003988
1: 0.9934075, 0.9943579, 0.9934075, 0.9943579, -0.0009505, 0.0009505
2: -0.0087946, -0.0055215, -0.0087946, -0.0055215, -0.0029725, 0.0029763
3: 0.0035868, 0.0041619, 0.0035868, 0.0041619, -0.0005751, 0.0005751
4: 0.0027809, 0.0053678, 0.0027809, 0.0053678, -0.0025869, 0.0025869
5: 0.0050646, 0.0064029, 0.0050646, 0.0064029, -0.0013382, 0.0013382
6: -0.0021738, -0.0008562, -0.0021738, -0.0008562, -0.0013176, 0.0013176
7: -0.0082275, -0.0074685, -0.0082275, -0.0074685, -0.0007590, 0.0007590
8: 0.0055062, 0.0098066, 0.0055062, 0.0098066, -0.0042208, 0.0042197
9: -0.0036825, -0.0031002, -0.0036825, -0.0031002, -0.0005823, 0.0005823

### Unstable ReLU Count (Linear/Conv2D Layers)
- layer_idx=0, type=LayerType.Linear, total=256, inp1_unstable=9, inp2_unstable=9, delta_unstable=249
- layer_idx=2, type=LayerType.Linear, total=256, inp1_unstable=6, inp2_unstable=6, delta_unstable=256
- layer_idx=4, type=LayerType.Linear, total=256, inp1_unstable=0, inp2_unstable=0, delta_unstable=256
- layer_idx=6, type=LayerType.Linear, total=256, inp1_unstable=2, inp2_unstable=2, delta_unstable=256
- layer_idx=8, type=LayerType.Linear, total=10, inp1_unstable=0, inp2_unstable=0, delta_unstable=10

Time for backsubstitution: 1.77 seconds

### RS candidates at layer 1
type: RSZ, layer: 1, pos: 68
type: RSZ, layer: 1, pos: 83

Time for candidate selection: 0.00 seconds

### Candidate
type: RSZ, layer: 1, pos: 68

### Relational analysis RSZ of RS_RSZ1_RSZ2_RSZ1_RSZ1_RSZ2_RSZ2_RSZ1_RSZ1

#### Relational analysis RSZ result of RS_RSZ1_RSZ2_RSZ1_RSZ1_RSZ2_RSZ2_RSZ1_RSZ1
Status: Status.UNKNOWN
Output dim: 1, lower bound: -0.0007131, upper bound: 0.0007131
time: 0.61 seconds

### Relational analysis RSZ of RS_RSZ1_RSZ2_RSZ1_RSZ1_RSZ2_RSZ2_RSZ1_RSZ2

#### Relational analysis RSZ result of RS_RSZ1_RSZ2_RSZ1_RSZ1_RSZ2_RSZ2_RSZ1_RSZ2
Status: Status.UNKNOWN
Output dim: 1, lower bound: -0.0007131, upper bound: 0.0007131
time: 0.61 seconds

## BFS RS instance: RS_RSZ1_RSZ2_RSZ1_RSZ1_RSZ2_RSZ2_RSZ2

### Backsubstitution after applying RS history:
0: 0.0007184, 0.0011172, 0.0007184, 0.0011172, -0.0003988, 0.0003988
1: 0.9934075, 0.9943579, 0.9934075, 0.9943579, -0.0009505, 0.0009505
2: -0.0087946, -0.0055215, -0.0087946, -0.0055215, -0.0029732, 0.0029783
3: 0.0035868, 0.0041619, 0.0035868, 0.0041619, -0.0005751, 0.0005751
4: 0.0027809, 0.0053678, 0.0027809, 0.0053678, -0.0025869, 0.0025869
5: 0.0050646, 0.0064029, 0.0050646, 0.0064029, -0.0013382, 0.0013382
6: -0.0021738, -0.0008562, -0.0021738, -0.0008562, -0.0013176, 0.0013176
7: -0.0082275, -0.0074685, -0.0082275, -0.0074685, -0.0007590, 0.0007590
8: 0.0055062, 0.0098066, 0.0055062, 0.0098066, -0.0042213, 0.0042199
9: -0.0036825, -0.0031002, -0.0036825, -0.0031002, -0.0005823, 0.0005823

### Unstable ReLU Count (Linear/Conv2D Layers)
- layer_idx=0, type=LayerType.Linear, total=256, inp1_unstable=9, inp2_unstable=9, delta_unstable=249
- layer_idx=2, type=LayerType.Linear, total=256, inp1_unstable=6, inp2_unstable=6, delta_unstable=256
- layer_idx=4, type=LayerType.Linear, total=256, inp1_unstable=0, inp2_unstable=0, delta_unstable=256
- layer_idx=6, type=LayerType.Linear, total=256, inp1_unstable=2, inp2_unstable=2, delta_unstable=256
- layer_idx=8, type=LayerType.Linear, total=10, inp1_unstable=0, inp2_unstable=0, delta_unstable=10

Time for backsubstitution: 1.76 seconds

### RS candidates at layer 1
type: RSZ, layer: 1, pos: 83
type: RSZ, layer: 1, pos: 68

Time for candidate selection: 0.00 seconds

### Candidate
type: RSZ, layer: 1, pos: 83

### Relational analysis RSZ of RS_RSZ1_RSZ2_RSZ1_RSZ1_RSZ2_RSZ2_RSZ2_RSZ1

#### Relational analysis RSZ result of RS_RSZ1_RSZ2_RSZ1_RSZ1_RSZ2_RSZ2_RSZ2_RSZ1
Status: Status.UNKNOWN
Output dim: 1, lower bound: -0.0007098, upper bound: 0.0007098
time: 0.63 seconds

### Relational analysis RSZ of RS_RSZ1_RSZ2_RSZ1_RSZ1_RSZ2_RSZ2_RSZ2_RSZ2

#### Relational analysis RSZ result of RS_RSZ1_RSZ2_RSZ1_RSZ1_RSZ2_RSZ2_RSZ2_RSZ2
Status: Status.UNKNOWN
Output dim: 1, lower bound: -0.0007098, upper bound: 0.0007098
time: 0.63 seconds

## BFS RS instance: RS_RSZ1_RSZ2_RSZ1_RSZ2_RSZ1_RSZ1_RSZ1

### Backsubstitution after applying RS history:
0: 0.0007184, 0.0011172, 0.0007184, 0.0011172, -0.0003988, 0.0003988
1: 0.9934075, 0.9943579, 0.9934075, 0.9943579, -0.0009505, 0.0009505
2: -0.0087946, -0.0055215, -0.0087946, -0.0055215, -0.0029780, 0.0029689
3: 0.0035868, 0.0041619, 0.0035868, 0.0041619, -0.0005751, 0.0005751
4: 0.0027809, 0.0053678, 0.0027809, 0.0053678, -0.0025869, 0.0025869
5: 0.0050646, 0.0064029, 0.0050646, 0.0064029, -0.0013382, 0.0013382
6: -0.0021738, -0.0008562, -0.0021738, -0.0008562, -0.0013176, 0.0013176
7: -0.0082275, -0.0074685, -0.0082275, -0.0074685, -0.0007590, 0.0007590
8: 0.0055062, 0.0098066, 0.0055062, 0.0098066, -0.0042184, 0.0042205
9: -0.0036825, -0.0031002, -0.0036825, -0.0031002, -0.0005823, 0.0005823

### Unstable ReLU Count (Linear/Conv2D Layers)
- layer_idx=0, type=LayerType.Linear, total=256, inp1_unstable=9, inp2_unstable=9, delta_unstable=249
- layer_idx=2, type=LayerType.Linear, total=256, inp1_unstable=6, inp2_unstable=6, delta_unstable=256
- layer_idx=4, type=LayerType.Linear, total=256, inp1_unstable=0, inp2_unstable=0, delta_unstable=256
- layer_idx=6, type=LayerType.Linear, total=256, inp1_unstable=2, inp2_unstable=2, delta_unstable=256
- layer_idx=8, type=LayerType.Linear, total=10, inp1_unstable=0, inp2_unstable=0, delta_unstable=10

Time for backsubstitution: 1.85 seconds

### RS candidates at layer 1
type: RSZ, layer: 1, pos: 83
type: RSZ, layer: 1, pos: 199

Time for candidate selection: 0.00 seconds

### Candidate
type: RSZ, layer: 1, pos: 83

### Relational analysis RSZ of RS_RSZ1_RSZ2_RSZ1_RSZ2_RSZ1_RSZ1_RSZ1_RSZ1

#### Relational analysis RSZ result of RS_RSZ1_RSZ2_RSZ1_RSZ2_RSZ1_RSZ1_RSZ1_RSZ1
Status: Status.UNKNOWN
Output dim: 1, lower bound: -0.0007678, upper bound: 0.0007701
time: 0.72 seconds

### Relational analysis RSZ of RS_RSZ1_RSZ2_RSZ1_RSZ2_RSZ1_RSZ1_RSZ1_RSZ2

#### Relational analysis RSZ result of RS_RSZ1_RSZ2_RSZ1_RSZ2_RSZ1_RSZ1_RSZ1_RSZ2
Status: Status.UNKNOWN
Output dim: 1, lower bound: -0.0007675, upper bound: 0.0007701
time: 0.75 seconds

## BFS RS instance: RS_RSZ1_RSZ2_RSZ1_RSZ2_RSZ1_RSZ1_RSZ2

### Backsubstitution after applying RS history:
0: 0.0007184, 0.0011172, 0.0007184, 0.0011172, -0.0003988, 0.0003988
1: 0.9934075, 0.9943579, 0.9934075, 0.9943579, -0.0009505, 0.0009505
2: -0.0087946, -0.0055215, -0.0087946, -0.0055215, -0.0029789, 0.0029679
3: 0.0035868, 0.0041619, 0.0035868, 0.0041619, -0.0005751, 0.0005751
4: 0.0027809, 0.0053678, 0.0027809, 0.0053678, -0.0025869, 0.0025869
5: 0.0050646, 0.0064029, 0.0050646, 0.0064029, -0.0013382, 0.0013382
6: -0.0021738, -0.0008562, -0.0021738, -0.0008562, -0.0013176, 0.0013176
7: -0.0082275, -0.0074685, -0.0082275, -0.0074685, -0.0007590, 0.0007590
8: 0.0055062, 0.0098066, 0.0055062, 0.0098066, -0.0042181, 0.0042207
9: -0.0036825, -0.0031002, -0.0036825, -0.0031002, -0.0005823, 0.0005823

### Unstable ReLU Count (Linear/Conv2D Layers)
- layer_idx=0, type=LayerType.Linear, total=256, inp1_unstable=9, inp2_unstable=9, delta_unstable=249
- layer_idx=2, type=LayerType.Linear, total=256, inp1_unstable=6, inp2_unstable=6, delta_unstable=256
- layer_idx=4, type=LayerType.Linear, total=256, inp1_unstable=0, inp2_unstable=0, delta_unstable=256
- layer_idx=6, type=LayerType.Linear, total=256, inp1_unstable=2, inp2_unstable=2, delta_unstable=256
- layer_idx=8, type=LayerType.Linear, total=10, inp1_unstable=0, inp2_unstable=0, delta_unstable=10

Time for backsubstitution: 1.70 seconds

### RS candidates at layer 1
type: RSZ, layer: 1, pos: 199
type: RSZ, layer: 1, pos: 83

Time for candidate selection: 0.00 seconds

### Candidate
type: RSZ, layer: 1, pos: 199

### Relational analysis RSZ of RS_RSZ1_RSZ2_RSZ1_RSZ2_RSZ1_RSZ1_RSZ2_RSZ1

#### Relational analysis RSZ result of RS_RSZ1_RSZ2_RSZ1_RSZ2_RSZ1_RSZ1_RSZ2_RSZ1
Status: Status.UNKNOWN
Output dim: 1, lower bound: -0.0007131, upper bound: 0.0007131
time: 0.63 seconds

### Relational analysis RSZ of RS_RSZ1_RSZ2_RSZ1_RSZ2_RSZ1_RSZ1_RSZ2_RSZ2

#### Relational analysis RSZ result of RS_RSZ1_RSZ2_RSZ1_RSZ2_RSZ1_RSZ1_RSZ2_RSZ2
Status: Status.UNKNOWN
Output dim: 1, lower bound: -0.0007131, upper bound: 0.0007131
time: 0.65 seconds

## BFS RS instance: RS_RSZ1_RSZ2_RSZ1_RSZ2_RSZ1_RSZ2_RSZ1

### Backsubstitution after applying RS history:
0: 0.0007184, 0.0011172, 0.0007184, 0.0011172, -0.0003988, 0.0003988
1: 0.9934075, 0.9943579, 0.9934075, 0.9943579, -0.0009505, 0.0009505
2: -0.0087946, -0.0055215, -0.0087946, -0.0055215, -0.0029773, 0.0029700
3: 0.0035868, 0.0041619, 0.0035868, 0.0041619, -0.0005751, 0.0005751
4: 0.0027809, 0.0053678, 0.0027809, 0.0053678, -0.0025869, 0.0025869
5: 0.0050646, 0.0064029, 0.0050646, 0.0064029, -0.0013382, 0.0013382
6: -0.0021738, -0.0008562, -0.0021738, -0.0008562, -0.0013176, 0.0013176
7: -0.0082275, -0.0074685, -0.0082275, -0.0074685, -0.0007590, 0.0007590
8: 0.0055062, 0.0098066, 0.0055062, 0.0098066, -0.0042188, 0.0042205
9: -0.0036825, -0.0031002, -0.0036825, -0.0031002, -0.0005823, 0.0005823

### Unstable ReLU Count (Linear/Conv2D Layers)
- layer_idx=0, type=LayerType.Linear, total=256, inp1_unstable=9, inp2_unstable=9, delta_unstable=249
- layer_idx=2, type=LayerType.Linear, total=256, inp1_unstable=6, inp2_unstable=6, delta_unstable=256
- layer_idx=4, type=LayerType.Linear, total=256, inp1_unstable=0, inp2_unstable=0, delta_unstable=256
- layer_idx=6, type=LayerType.Linear, total=256, inp1_unstable=2, inp2_unstable=2, delta_unstable=256
- layer_idx=8, type=LayerType.Linear, total=10, inp1_unstable=0, inp2_unstable=0, delta_unstable=10

Time for backsubstitution: 1.84 seconds

### RS candidates at layer 1
type: RSZ, layer: 1, pos: 199
type: RSZ, layer: 1, pos: 52

Time for candidate selection: 0.00 seconds

### Candidate
type: RSZ, layer: 1, pos: 199

### Relational analysis RSZ of RS_RSZ1_RSZ2_RSZ1_RSZ2_RSZ1_RSZ2_RSZ1_RSZ1

#### Relational analysis RSZ result of RS_RSZ1_RSZ2_RSZ1_RSZ2_RSZ1_RSZ2_RSZ1_RSZ1
Status: Status.UNKNOWN
Output dim: 1, lower bound: -0.0007098, upper bound: 0.0007098
time: 0.63 seconds

### Relational analysis RSZ of RS_RSZ1_RSZ2_RSZ1_RSZ2_RSZ1_RSZ2_RSZ1_RSZ2

#### Relational analysis RSZ result of RS_RSZ1_RSZ2_RSZ1_RSZ2_RSZ1_RSZ2_RSZ1_RSZ2
Status: Status.UNKNOWN
Output dim: 1, lower bound: -0.0007098, upper bound: 0.0007098
time: 0.64 seconds

## BFS RS instance: RS_RSZ1_RSZ2_RSZ1_RSZ2_RSZ1_RSZ2_RSZ2

### Backsubstitution after applying RS history:
0: 0.0007184, 0.0011172, 0.0007184, 0.0011172, -0.0003988, 0.0003988
1: 0.9934075, 0.9943579, 0.9934075, 0.9943579, -0.0009505, 0.0009505
2: -0.0087946, -0.0055215, -0.0087946, -0.0055215, -0.0029771, 0.0029704
3: 0.0035868, 0.0041619, 0.0035868, 0.0041619, -0.0005751, 0.0005751
4: 0.0027809, 0.0053678, 0.0027809, 0.0053678, -0.0025869, 0.0025869
5: 0.0050646, 0.0064029, 0.0050646, 0.0064029, -0.0013382, 0.0013382
6: -0.0021738, -0.0008562, -0.0021738, -0.0008562, -0.0013176, 0.0013176
7: -0.0082275, -0.0074685, -0.0082275, -0.0074685, -0.0007590, 0.0007590
8: 0.0055062, 0.0098066, 0.0055062, 0.0098066, -0.0042189, 0.0042204
9: -0.0036825, -0.0031002, -0.0036825, -0.0031002, -0.0005823, 0.0005823

### Unstable ReLU Count (Linear/Conv2D Layers)
- layer_idx=0, type=LayerType.Linear, total=256, inp1_unstable=9, inp2_unstable=9, delta_unstable=249
- layer_idx=2, type=LayerType.Linear, total=256, inp1_unstable=6, inp2_unstable=6, delta_unstable=256
- layer_idx=4, type=LayerType.Linear, total=256, inp1_unstable=0, inp2_unstable=0, delta_unstable=256
- layer_idx=6, type=LayerType.Linear, total=256, inp1_unstable=2, inp2_unstable=2, delta_unstable=256
- layer_idx=8, type=LayerType.Linear, total=10, inp1_unstable=0, inp2_unstable=0, delta_unstable=10

Time for backsubstitution: 1.86 seconds

### RS candidates at layer 1
type: RSZ, layer: 1, pos: 199
type: RSZ, layer: 1, pos: 52

Time for candidate selection: 0.00 seconds

### Candidate
type: RSZ, layer: 1, pos: 199

### Relational analysis RSZ of RS_RSZ1_RSZ2_RSZ1_RSZ2_RSZ1_RSZ2_RSZ2_RSZ1

#### Relational analysis RSZ result of RS_RSZ1_RSZ2_RSZ1_RSZ2_RSZ1_RSZ2_RSZ2_RSZ1
Status: Status.UNKNOWN
Output dim: 1, lower bound: -0.0007098, upper bound: 0.0007098
time: 0.67 seconds

### Relational analysis RSZ of RS_RSZ1_RSZ2_RSZ1_RSZ2_RSZ1_RSZ2_RSZ2_RSZ2

#### Relational analysis RSZ result of RS_RSZ1_RSZ2_RSZ1_RSZ2_RSZ1_RSZ2_RSZ2_RSZ2
Status: Status.UNKNOWN
Output dim: 1, lower bound: -0.0007098, upper bound: 0.0007098
time: 0.66 seconds

## BFS RS instance: RS_RSZ1_RSZ2_RSZ1_RSZ2_RSZ2_RSZ1_RSZ1

### Backsubstitution after applying RS history:
0: 0.0007184, 0.0011172, 0.0007184, 0.0011172, -0.0003988, 0.0003988
1: 0.9934075, 0.9943579, 0.9934075, 0.9943579, -0.0009505, 0.0009505
2: -0.0087946, -0.0055215, -0.0087946, -0.0055215, -0.0029820, 0.0029767
3: 0.0035868, 0.0041619, 0.0035868, 0.0041619, -0.0005751, 0.0005751
4: 0.0027809, 0.0053678, 0.0027809, 0.0053678, -0.0025869, 0.0025869
5: 0.0050646, 0.0064029, 0.0050646, 0.0064029, -0.0013382, 0.0013382
6: -0.0021738, -0.0008562, -0.0021738, -0.0008562, -0.0013176, 0.0013176
7: -0.0082275, -0.0074685, -0.0082275, -0.0074685, -0.0007590, 0.0007590
8: 0.0055062, 0.0098066, 0.0055062, 0.0098066, -0.0042209, 0.0042220
9: -0.0036825, -0.0031002, -0.0036825, -0.0031002, -0.0005823, 0.0005823

### Unstable ReLU Count (Linear/Conv2D Layers)
- layer_idx=0, type=LayerType.Linear, total=256, inp1_unstable=9, inp2_unstable=9, delta_unstable=249
- layer_idx=2, type=LayerType.Linear, total=256, inp1_unstable=6, inp2_unstable=6, delta_unstable=256
- layer_idx=4, type=LayerType.Linear, total=256, inp1_unstable=0, inp2_unstable=0, delta_unstable=256
- layer_idx=6, type=LayerType.Linear, total=256, inp1_unstable=2, inp2_unstable=2, delta_unstable=256
- layer_idx=8, type=LayerType.Linear, total=10, inp1_unstable=0, inp2_unstable=0, delta_unstable=10

Time for backsubstitution: 1.71 seconds

### RS candidates at layer 1
type: RSZ, layer: 1, pos: 68
type: RSZ, layer: 1, pos: 83

Time for candidate selection: 0.00 seconds

### Candidate
type: RSZ, layer: 1, pos: 68

### Relational analysis RSZ of RS_RSZ1_RSZ2_RSZ1_RSZ2_RSZ2_RSZ1_RSZ1_RSZ1

#### Relational analysis RSZ result of RS_RSZ1_RSZ2_RSZ1_RSZ2_RSZ2_RSZ1_RSZ1_RSZ1
Status: Status.UNKNOWN
Output dim: 1, lower bound: -0.0007131, upper bound: 0.0007131
time: 0.61 seconds

### Relational analysis RSZ of RS_RSZ1_RSZ2_RSZ1_RSZ2_RSZ2_RSZ1_RSZ1_RSZ2

#### Relational analysis RSZ result of RS_RSZ1_RSZ2_RSZ1_RSZ2_RSZ2_RSZ1_RSZ1_RSZ2
Status: Status.UNKNOWN
Output dim: 1, lower bound: -0.0007131, upper bound: 0.0007131
time: 0.60 seconds

## BFS RS instance: RS_RSZ1_RSZ2_RSZ1_RSZ2_RSZ2_RSZ1_RSZ2

### Backsubstitution after applying RS history:
0: 0.0007184, 0.0011172, 0.0007184, 0.0011172, -0.0003988, 0.0003988
1: 0.9934075, 0.9943579, 0.9934075, 0.9943579, -0.0009505, 0.0009505
2: -0.0087946, -0.0055215, -0.0087946, -0.0055215, -0.0029829, 0.0029786
3: 0.0035868, 0.0041619, 0.0035868, 0.0041619, -0.0005751, 0.0005751
4: 0.0027809, 0.0053678, 0.0027809, 0.0053678, -0.0025869, 0.0025869
5: 0.0050646, 0.0064029, 0.0050646, 0.0064029, -0.0013382, 0.0013382
6: -0.0021738, -0.0008562, -0.0021738, -0.0008562, -0.0013176, 0.0013176
7: -0.0082275, -0.0074685, -0.0082275, -0.0074685, -0.0007590, 0.0007590
8: 0.0055062, 0.0098066, 0.0055062, 0.0098066, -0.0042214, 0.0042222
9: -0.0036825, -0.0031002, -0.0036825, -0.0031002, -0.0005823, 0.0005823

### Unstable ReLU Count (Linear/Conv2D Layers)
- layer_idx=0, type=LayerType.Linear, total=256, inp1_unstable=9, inp2_unstable=9, delta_unstable=249
- layer_idx=2, type=LayerType.Linear, total=256, inp1_unstable=6, inp2_unstable=6, delta_unstable=256
- layer_idx=4, type=LayerType.Linear, total=256, inp1_unstable=0, inp2_unstable=0, delta_unstable=256
- layer_idx=6, type=LayerType.Linear, total=256, inp1_unstable=2, inp2_unstable=2, delta_unstable=256
- layer_idx=8, type=LayerType.Linear, total=10, inp1_unstable=0, inp2_unstable=0, delta_unstable=10

Time for backsubstitution: 1.82 seconds

### RS candidates at layer 1
type: RSZ, layer: 1, pos: 68
type: RSZ, layer: 1, pos: 83

Time for candidate selection: 0.00 seconds

### Candidate
type: RSZ, layer: 1, pos: 68

### Relational analysis RSZ of RS_RSZ1_RSZ2_RSZ1_RSZ2_RSZ2_RSZ1_RSZ2_RSZ1

#### Relational analysis RSZ result of RS_RSZ1_RSZ2_RSZ1_RSZ2_RSZ2_RSZ1_RSZ2_RSZ1
Status: Status.UNKNOWN
Output dim: 1, lower bound: -0.0007131, upper bound: 0.0007131
time: 0.63 seconds

### Relational analysis RSZ of RS_RSZ1_RSZ2_RSZ1_RSZ2_RSZ2_RSZ1_RSZ2_RSZ2

#### Relational analysis RSZ result of RS_RSZ1_RSZ2_RSZ1_RSZ2_RSZ2_RSZ1_RSZ2_RSZ2
Status: Status.UNKNOWN
Output dim: 1, lower bound: -0.0007131, upper bound: 0.0007131
time: 0.62 seconds

## BFS RS instance: RS_RSZ1_RSZ2_RSZ1_RSZ2_RSZ2_RSZ2_RSZ1

### Backsubstitution after applying RS history:
0: 0.0007184, 0.0011172, 0.0007184, 0.0011172, -0.0003988, 0.0003988
1: 0.9934075, 0.9943579, 0.9934075, 0.9943579, -0.0009505, 0.0009505
2: -0.0087946, -0.0055215, -0.0087946, -0.0055215, -0.0029830, 0.0029757
3: 0.0035868, 0.0041619, 0.0035868, 0.0041619, -0.0005751, 0.0005751
4: 0.0027809, 0.0053678, 0.0027809, 0.0053678, -0.0025869, 0.0025869
5: 0.0050646, 0.0064029, 0.0050646, 0.0064029, -0.0013382, 0.0013382
6: -0.0021738, -0.0008562, -0.0021738, -0.0008562, -0.0013176, 0.0013176
7: -0.0082275, -0.0074685, -0.0082275, -0.0074685, -0.0007590, 0.0007590
8: 0.0055062, 0.0098066, 0.0055062, 0.0098066, -0.0042206, 0.0042223
9: -0.0036825, -0.0031002, -0.0036825, -0.0031002, -0.0005823, 0.0005823

### Unstable ReLU Count (Linear/Conv2D Layers)
- layer_idx=0, type=LayerType.Linear, total=256, inp1_unstable=9, inp2_unstable=9, delta_unstable=249
- layer_idx=2, type=LayerType.Linear, total=256, inp1_unstable=6, inp2_unstable=6, delta_unstable=256
- layer_idx=4, type=LayerType.Linear, total=256, inp1_unstable=0, inp2_unstable=0, delta_unstable=256
- layer_idx=6, type=LayerType.Linear, total=256, inp1_unstable=2, inp2_unstable=2, delta_unstable=256
- layer_idx=8, type=LayerType.Linear, total=10, inp1_unstable=0, inp2_unstable=0, delta_unstable=10

Time for backsubstitution: 1.86 seconds

### RS candidates at layer 1
type: RSZ, layer: 1, pos: 83
type: RSZ, layer: 1, pos: 68

Time for candidate selection: 0.00 seconds

### Candidate
type: RSZ, layer: 1, pos: 83

### Relational analysis RSZ of RS_RSZ1_RSZ2_RSZ1_RSZ2_RSZ2_RSZ2_RSZ1_RSZ1

#### Relational analysis RSZ result of RS_RSZ1_RSZ2_RSZ1_RSZ2_RSZ2_RSZ2_RSZ1_RSZ1
Status: Status.UNKNOWN
Output dim: 1, lower bound: -0.0007098, upper bound: 0.0007098
time: 0.62 seconds

### Relational analysis RSZ of RS_RSZ1_RSZ2_RSZ1_RSZ2_RSZ2_RSZ2_RSZ1_RSZ2

#### Relational analysis RSZ result of RS_RSZ1_RSZ2_RSZ1_RSZ2_RSZ2_RSZ2_RSZ1_RSZ2
Status: Status.UNKNOWN
Output dim: 1, lower bound: -0.0007098, upper bound: 0.0007098
time: 0.67 seconds

## BFS RS instance: RS_RSZ1_RSZ2_RSZ1_RSZ2_RSZ2_RSZ2_RSZ2

### Backsubstitution after applying RS history:
0: 0.0007184, 0.0011172, 0.0007184, 0.0011172, -0.0003988, 0.0003988
1: 0.9934075, 0.9943579, 0.9934075, 0.9943579, -0.0009505, 0.0009505
2: -0.0087946, -0.0055215, -0.0087946, -0.0055215, -0.0029837, 0.0029776
3: 0.0035868, 0.0041619, 0.0035868, 0.0041619, -0.0005751, 0.0005751
4: 0.0027809, 0.0053678, 0.0027809, 0.0053678, -0.0025869, 0.0025869
5: 0.0050646, 0.0064029, 0.0050646, 0.0064029, -0.0013382, 0.0013382
6: -0.0021738, -0.0008562, -0.0021738, -0.0008562, -0.0013176, 0.0013176
7: -0.0082275, -0.0074685, -0.0082275, -0.0074685, -0.0007590, 0.0007590
8: 0.0055062, 0.0098066, 0.0055062, 0.0098066, -0.0042211, 0.0042225
9: -0.0036825, -0.0031002, -0.0036825, -0.0031002, -0.0005823, 0.0005823

### Unstable ReLU Count (Linear/Conv2D Layers)
- layer_idx=0, type=LayerType.Linear, total=256, inp1_unstable=9, inp2_unstable=9, delta_unstable=249
- layer_idx=2, type=LayerType.Linear, total=256, inp1_unstable=6, inp2_unstable=6, delta_unstable=256
- layer_idx=4, type=LayerType.Linear, total=256, inp1_unstable=0, inp2_unstable=0, delta_unstable=256
- layer_idx=6, type=LayerType.Linear, total=256, inp1_unstable=2, inp2_unstable=2, delta_unstable=256
- layer_idx=8, type=LayerType.Linear, total=10, inp1_unstable=0, inp2_unstable=0, delta_unstable=10

Time for backsubstitution: 1.82 seconds

### RS candidates at layer 1
type: RSZ, layer: 1, pos: 68
type: RSZ, layer: 1, pos: 83

Time for candidate selection: 0.00 seconds

### Candidate
type: RSZ, layer: 1, pos: 68

### Relational analysis RSZ of RS_RSZ1_RSZ2_RSZ1_RSZ2_RSZ2_RSZ2_RSZ2_RSZ1

#### Relational analysis RSZ result of RS_RSZ1_RSZ2_RSZ1_RSZ2_RSZ2_RSZ2_RSZ2_RSZ1
Status: Status.UNKNOWN
Output dim: 1, lower bound: -0.0007131, upper bound: 0.0007131
time: 0.66 seconds

### Relational analysis RSZ of RS_RSZ1_RSZ2_RSZ1_RSZ2_RSZ2_RSZ2_RSZ2_RSZ2

#### Relational analysis RSZ result of RS_RSZ1_RSZ2_RSZ1_RSZ2_RSZ2_RSZ2_RSZ2_RSZ2
Status: Status.UNKNOWN
Output dim: 1, lower bound: -0.0007131, upper bound: 0.0007131
time: 0.63 seconds

## BFS RS instance: RS_RSZ1_RSZ2_RSZ2_RSZ1_RSZ1_RSZ1_RSZ1

### Backsubstitution after applying RS history:
0: 0.0007184, 0.0011172, 0.0007184, 0.0011172, -0.0003988, 0.0003988
1: 0.9934075, 0.9943579, 0.9934075, 0.9943579, -0.0009505, 0.0009505
2: -0.0087946, -0.0055215, -0.0087946, -0.0055215, -0.0029799, 0.0029781
3: 0.0035868, 0.0041619, 0.0035868, 0.0041619, -0.0005751, 0.0005751
4: 0.0027809, 0.0053678, 0.0027809, 0.0053678, -0.0025869, 0.0025869
5: 0.0050646, 0.0064029, 0.0050646, 0.0064029, -0.0013382, 0.0013382
6: -0.0021738, -0.0008562, -0.0021738, -0.0008562, -0.0013176, 0.0013176
7: -0.0082275, -0.0074685, -0.0082275, -0.0074685, -0.0007590, 0.0007590
8: 0.0055062, 0.0098066, 0.0055062, 0.0098066, -0.0042206, 0.0042211
9: -0.0036825, -0.0031002, -0.0036825, -0.0031002, -0.0005823, 0.0005823

### Unstable ReLU Count (Linear/Conv2D Layers)
- layer_idx=0, type=LayerType.Linear, total=256, inp1_unstable=9, inp2_unstable=9, delta_unstable=249
- layer_idx=2, type=LayerType.Linear, total=256, inp1_unstable=6, inp2_unstable=6, delta_unstable=256
- layer_idx=4, type=LayerType.Linear, total=256, inp1_unstable=0, inp2_unstable=0, delta_unstable=256
- layer_idx=6, type=LayerType.Linear, total=256, inp1_unstable=2, inp2_unstable=2, delta_unstable=256
- layer_idx=8, type=LayerType.Linear, total=10, inp1_unstable=0, inp2_unstable=0, delta_unstable=10

Time for backsubstitution: 1.83 seconds

### RS candidates at layer 1
type: RSZ, layer: 1, pos: 198
type: RSZ, layer: 1, pos: 204

Time for candidate selection: 0.00 seconds

### Candidate
type: RSZ, layer: 1, pos: 198

### Relational analysis RSZ of RS_RSZ1_RSZ2_RSZ2_RSZ1_RSZ1_RSZ1_RSZ1_RSZ1

#### Relational analysis RSZ result of RS_RSZ1_RSZ2_RSZ2_RSZ1_RSZ1_RSZ1_RSZ1_RSZ1
Status: Status.UNKNOWN
Output dim: 1, lower bound: -0.0007098, upper bound: 0.0007098
time: 0.68 seconds

### Relational analysis RSZ of RS_RSZ1_RSZ2_RSZ2_RSZ1_RSZ1_RSZ1_RSZ1_RSZ2

#### Relational analysis RSZ result of RS_RSZ1_RSZ2_RSZ2_RSZ1_RSZ1_RSZ1_RSZ1_RSZ2
Status: Status.UNKNOWN
Output dim: 1, lower bound: -0.0007098, upper bound: 0.0007098
time: 0.70 seconds

## BFS RS instance: RS_RSZ1_RSZ2_RSZ2_RSZ1_RSZ1_RSZ1_RSZ2

### Backsubstitution after applying RS history:
0: 0.0007184, 0.0011172, 0.0007184, 0.0011172, -0.0003988, 0.0003988
1: 0.9934075, 0.9943579, 0.9934075, 0.9943579, -0.0009505, 0.0009505
2: -0.0087946, -0.0055215, -0.0087946, -0.0055215, -0.0029810, 0.0029771
3: 0.0035868, 0.0041619, 0.0035868, 0.0041619, -0.0005751, 0.0005751
4: 0.0027809, 0.0053678, 0.0027809, 0.0053678, -0.0025869, 0.0025869
5: 0.0050646, 0.0064029, 0.0050646, 0.0064029, -0.0013382, 0.0013382
6: -0.0021738, -0.0008562, -0.0021738, -0.0008562, -0.0013176, 0.0013176
7: -0.0082275, -0.0074685, -0.0082275, -0.0074685, -0.0007590, 0.0007590
8: 0.0055062, 0.0098066, 0.0055062, 0.0098066, -0.0042203, 0.0042214
9: -0.0036825, -0.0031002, -0.0036825, -0.0031002, -0.0005823, 0.0005823

### Unstable ReLU Count (Linear/Conv2D Layers)
- layer_idx=0, type=LayerType.Linear, total=256, inp1_unstable=9, inp2_unstable=9, delta_unstable=249
- layer_idx=2, type=LayerType.Linear, total=256, inp1_unstable=6, inp2_unstable=6, delta_unstable=256
- layer_idx=4, type=LayerType.Linear, total=256, inp1_unstable=0, inp2_unstable=0, delta_unstable=256
- layer_idx=6, type=LayerType.Linear, total=256, inp1_unstable=2, inp2_unstable=2, delta_unstable=256
- layer_idx=8, type=LayerType.Linear, total=10, inp1_unstable=0, inp2_unstable=0, delta_unstable=10

Time for backsubstitution: 1.77 seconds

### RS candidates at layer 1
type: RSZ, layer: 1, pos: 198
type: RSZ, layer: 1, pos: 204

Time for candidate selection: 0.00 seconds

### Candidate
type: RSZ, layer: 1, pos: 198

### Relational analysis RSZ of RS_RSZ1_RSZ2_RSZ2_RSZ1_RSZ1_RSZ1_RSZ2_RSZ1

#### Relational analysis RSZ result of RS_RSZ1_RSZ2_RSZ2_RSZ1_RSZ1_RSZ1_RSZ2_RSZ1
Status: Status.UNKNOWN
Output dim: 1, lower bound: -0.0007097, upper bound: 0.0007098
time: 0.63 seconds

### Relational analysis RSZ of RS_RSZ1_RSZ2_RSZ2_RSZ1_RSZ1_RSZ1_RSZ2_RSZ2

#### Relational analysis RSZ result of RS_RSZ1_RSZ2_RSZ2_RSZ1_RSZ1_RSZ1_RSZ2_RSZ2
Status: Status.UNKNOWN
Output dim: 1, lower bound: -0.0007097, upper bound: 0.0007098
time: 0.64 seconds

## BFS RS instance: RS_RSZ1_RSZ2_RSZ2_RSZ1_RSZ1_RSZ2_RSZ1

### Backsubstitution after applying RS history:
0: 0.0007184, 0.0011172, 0.0007184, 0.0011172, -0.0003988, 0.0003988
1: 0.9934075, 0.9943579, 0.9934075, 0.9943579, -0.0009505, 0.0009505
2: -0.0087946, -0.0055215, -0.0087946, -0.0055215, -0.0029712, 0.0029732
3: 0.0035868, 0.0041619, 0.0035868, 0.0041619, -0.0005751, 0.0005751
4: 0.0027809, 0.0053678, 0.0027809, 0.0053678, -0.0025869, 0.0025869
5: 0.0050646, 0.0064029, 0.0050646, 0.0064029, -0.0013382, 0.0013382
6: -0.0021738, -0.0008562, -0.0021738, -0.0008562, -0.0013176, 0.0013176
7: -0.0082275, -0.0074685, -0.0082275, -0.0074685, -0.0007590, 0.0007590
8: 0.0055062, 0.0098066, 0.0055062, 0.0098066, -0.0042196, 0.0042191
9: -0.0036825, -0.0031002, -0.0036825, -0.0031002, -0.0005823, 0.0005823

### Unstable ReLU Count (Linear/Conv2D Layers)
- layer_idx=0, type=LayerType.Linear, total=256, inp1_unstable=9, inp2_unstable=9, delta_unstable=249
- layer_idx=2, type=LayerType.Linear, total=256, inp1_unstable=6, inp2_unstable=6, delta_unstable=256
- layer_idx=4, type=LayerType.Linear, total=256, inp1_unstable=0, inp2_unstable=0, delta_unstable=256
- layer_idx=6, type=LayerType.Linear, total=256, inp1_unstable=2, inp2_unstable=2, delta_unstable=256
- layer_idx=8, type=LayerType.Linear, total=10, inp1_unstable=0, inp2_unstable=0, delta_unstable=10

Time for backsubstitution: 1.74 seconds

### RS candidates at layer 1
type: RSZ, layer: 1, pos: 204
type: RSZ, layer: 1, pos: 52

Time for candidate selection: 0.00 seconds

### Candidate
type: RSZ, layer: 1, pos: 204

### Relational analysis RSZ of RS_RSZ1_RSZ2_RSZ2_RSZ1_RSZ1_RSZ2_RSZ1_RSZ1

#### Relational analysis RSZ result of RS_RSZ1_RSZ2_RSZ2_RSZ1_RSZ1_RSZ2_RSZ1_RSZ1
Status: Status.UNKNOWN
Output dim: 1, lower bound: -0.0007098, upper bound: 0.0007098
time: 0.68 seconds

### Relational analysis RSZ of RS_RSZ1_RSZ2_RSZ2_RSZ1_RSZ1_RSZ2_RSZ1_RSZ2

#### Relational analysis RSZ result of RS_RSZ1_RSZ2_RSZ2_RSZ1_RSZ1_RSZ2_RSZ1_RSZ2
Status: Status.UNKNOWN
Output dim: 1, lower bound: -0.0007097, upper bound: 0.0007098
time: 0.68 seconds

## BFS RS instance: RS_RSZ1_RSZ2_RSZ2_RSZ1_RSZ1_RSZ2_RSZ2

### Backsubstitution after applying RS history:
0: 0.0007184, 0.0011172, 0.0007184, 0.0011172, -0.0003988, 0.0003988
1: 0.9934075, 0.9943579, 0.9934075, 0.9943579, -0.0009505, 0.0009505
2: -0.0087946, -0.0055215, -0.0087946, -0.0055215, -0.0029817, 0.0029719
3: 0.0035868, 0.0041619, 0.0035868, 0.0041619, -0.0005751, 0.0005751
4: 0.0027809, 0.0053678, 0.0027809, 0.0053678, -0.0025869, 0.0025869
5: 0.0050646, 0.0064029, 0.0050646, 0.0064029, -0.0013382, 0.0013382
6: -0.0021738, -0.0008562, -0.0021738, -0.0008562, -0.0013176, 0.0013176
7: -0.0082275, -0.0074685, -0.0082275, -0.0074685, -0.0007590, 0.0007590
8: 0.0055062, 0.0098066, 0.0055062, 0.0098066, -0.0042193, 0.0042216
9: -0.0036825, -0.0031002, -0.0036825, -0.0031002, -0.0005823, 0.0005823

### Unstable ReLU Count (Linear/Conv2D Layers)
- layer_idx=0, type=LayerType.Linear, total=256, inp1_unstable=9, inp2_unstable=9, delta_unstable=249
- layer_idx=2, type=LayerType.Linear, total=256, inp1_unstable=6, inp2_unstable=6, delta_unstable=256
- layer_idx=4, type=LayerType.Linear, total=256, inp1_unstable=0, inp2_unstable=0, delta_unstable=256
- layer_idx=6, type=LayerType.Linear, total=256, inp1_unstable=2, inp2_unstable=2, delta_unstable=256
- layer_idx=8, type=LayerType.Linear, total=10, inp1_unstable=0, inp2_unstable=0, delta_unstable=10

Time for backsubstitution: 2.08 seconds

### RS candidates at layer 1
type: RSZ, layer: 1, pos: 52
type: RSZ, layer: 1, pos: 204

Time for candidate selection: 0.00 seconds

### Candidate
type: RSZ, layer: 1, pos: 52

### Relational analysis RSZ of RS_RSZ1_RSZ2_RSZ2_RSZ1_RSZ1_RSZ2_RSZ2_RSZ1

#### Relational analysis RSZ result of RS_RSZ1_RSZ2_RSZ2_RSZ1_RSZ1_RSZ2_RSZ2_RSZ1
Status: Status.UNKNOWN
Output dim: 1, lower bound: -0.0007097, upper bound: 0.0007098
time: 0.74 seconds

### Relational analysis RSZ of RS_RSZ1_RSZ2_RSZ2_RSZ1_RSZ1_RSZ2_RSZ2_RSZ2

#### Relational analysis RSZ result of RS_RSZ1_RSZ2_RSZ2_RSZ1_RSZ1_RSZ2_RSZ2_RSZ2
Status: Status.UNKNOWN
Output dim: 1, lower bound: -0.0007098, upper bound: 0.0007098
time: 0.66 seconds

## BFS RS instance: RS_RSZ1_RSZ2_RSZ2_RSZ1_RSZ2_RSZ1_RSZ1

### Backsubstitution after applying RS history:
0: 0.0007184, 0.0011172, 0.0007184, 0.0011172, -0.0003988, 0.0003988
1: 0.9934075, 0.9943579, 0.9934075, 0.9943579, -0.0009505, 0.0009505
2: -0.0087946, -0.0055215, -0.0087946, -0.0055215, -0.0029727, 0.0029713
3: 0.0035868, 0.0041619, 0.0035868, 0.0041619, -0.0005751, 0.0005751
4: 0.0027809, 0.0053678, 0.0027809, 0.0053678, -0.0025869, 0.0025869
5: 0.0050646, 0.0064029, 0.0050646, 0.0064029, -0.0013382, 0.0013382
6: -0.0021738, -0.0008562, -0.0021738, -0.0008562, -0.0013176, 0.0013176
7: -0.0082275, -0.0074685, -0.0082275, -0.0074685, -0.0007590, 0.0007590
8: 0.0055062, 0.0098066, 0.0055062, 0.0098066, -0.0042191, 0.0042195
9: -0.0036825, -0.0031002, -0.0036825, -0.0031002, -0.0005823, 0.0005823

### Unstable ReLU Count (Linear/Conv2D Layers)
- layer_idx=0, type=LayerType.Linear, total=256, inp1_unstable=9, inp2_unstable=9, delta_unstable=249
- layer_idx=2, type=LayerType.Linear, total=256, inp1_unstable=6, inp2_unstable=6, delta_unstable=256
- layer_idx=4, type=LayerType.Linear, total=256, inp1_unstable=0, inp2_unstable=0, delta_unstable=256
- layer_idx=6, type=LayerType.Linear, total=256, inp1_unstable=2, inp2_unstable=2, delta_unstable=256
- layer_idx=8, type=LayerType.Linear, total=10, inp1_unstable=0, inp2_unstable=0, delta_unstable=10

Time for backsubstitution: 2.08 seconds

### RS candidates at layer 1
type: RSZ, layer: 1, pos: 52
type: RSZ, layer: 1, pos: 204

Time for candidate selection: 0.00 seconds

### Candidate
type: RSZ, layer: 1, pos: 52

### Relational analysis RSZ of RS_RSZ1_RSZ2_RSZ2_RSZ1_RSZ2_RSZ1_RSZ1_RSZ1

#### Relational analysis RSZ result of RS_RSZ1_RSZ2_RSZ2_RSZ1_RSZ2_RSZ1_RSZ1_RSZ1
Status: Status.UNKNOWN
Output dim: 1, lower bound: -0.0007098, upper bound: 0.0007098
time: 0.65 seconds

### Relational analysis RSZ of RS_RSZ1_RSZ2_RSZ2_RSZ1_RSZ2_RSZ1_RSZ1_RSZ2

#### Relational analysis RSZ result of RS_RSZ1_RSZ2_RSZ2_RSZ1_RSZ2_RSZ1_RSZ1_RSZ2
Status: Status.UNKNOWN
Output dim: 1, lower bound: -0.0007097, upper bound: 0.0007098
time: 0.70 seconds

## BFS RS instance: RS_RSZ1_RSZ2_RSZ2_RSZ1_RSZ2_RSZ1_RSZ2

### Backsubstitution after applying RS history:
0: 0.0007184, 0.0011172, 0.0007184, 0.0011172, -0.0003988, 0.0003988
1: 0.9934075, 0.9943579, 0.9934075, 0.9943579, -0.0009505, 0.0009505
2: -0.0087946, -0.0055215, -0.0087946, -0.0055215, -0.0029832, 0.0029705
3: 0.0035868, 0.0041619, 0.0035868, 0.0041619, -0.0005751, 0.0005751
4: 0.0027809, 0.0053678, 0.0027809, 0.0053678, -0.0025869, 0.0025869
5: 0.0050646, 0.0064029, 0.0050646, 0.0064029, -0.0013382, 0.0013382
6: -0.0021738, -0.0008562, -0.0021738, -0.0008562, -0.0013176, 0.0013176
7: -0.0082275, -0.0074685, -0.0082275, -0.0074685, -0.0007590, 0.0007590
8: 0.0055062, 0.0098066, 0.0055062, 0.0098066, -0.0042189, 0.0042220
9: -0.0036825, -0.0031002, -0.0036825, -0.0031002, -0.0005823, 0.0005823

### Unstable ReLU Count (Linear/Conv2D Layers)
- layer_idx=0, type=LayerType.Linear, total=256, inp1_unstable=9, inp2_unstable=9, delta_unstable=249
- layer_idx=2, type=LayerType.Linear, total=256, inp1_unstable=6, inp2_unstable=6, delta_unstable=256
- layer_idx=4, type=LayerType.Linear, total=256, inp1_unstable=0, inp2_unstable=0, delta_unstable=256
- layer_idx=6, type=LayerType.Linear, total=256, inp1_unstable=2, inp2_unstable=2, delta_unstable=256
- layer_idx=8, type=LayerType.Linear, total=10, inp1_unstable=0, inp2_unstable=0, delta_unstable=10

Time for backsubstitution: 1.95 seconds

### RS candidates at layer 1
type: RSZ, layer: 1, pos: 52
type: RSZ, layer: 1, pos: 204

Time for candidate selection: 0.00 seconds

### Candidate
type: RSZ, layer: 1, pos: 52

### Relational analysis RSZ of RS_RSZ1_RSZ2_RSZ2_RSZ1_RSZ2_RSZ1_RSZ2_RSZ1

#### Relational analysis RSZ result of RS_RSZ1_RSZ2_RSZ2_RSZ1_RSZ2_RSZ1_RSZ2_RSZ1
Status: Status.UNKNOWN
Output dim: 1, lower bound: -0.0007098, upper bound: 0.0007098
time: 0.62 seconds

### Relational analysis RSZ of RS_RSZ1_RSZ2_RSZ2_RSZ1_RSZ2_RSZ1_RSZ2_RSZ2

#### Relational analysis RSZ result of RS_RSZ1_RSZ2_RSZ2_RSZ1_RSZ2_RSZ1_RSZ2_RSZ2
Status: Status.UNKNOWN
Output dim: 1, lower bound: -0.0007097, upper bound: 0.0007098
time: 0.67 seconds

## BFS RS instance: RS_RSZ1_RSZ2_RSZ2_RSZ1_RSZ2_RSZ2_RSZ1

### Backsubstitution after applying RS history:
0: 0.0007184, 0.0011172, 0.0007184, 0.0011172, -0.0003988, 0.0003988
1: 0.9934075, 0.9943579, 0.9934075, 0.9943579, -0.0009505, 0.0009505
2: -0.0087946, -0.0055215, -0.0087946, -0.0055215, -0.0029761, 0.0029765
3: 0.0035868, 0.0041619, 0.0035868, 0.0041619, -0.0005751, 0.0005751
4: 0.0027809, 0.0053678, 0.0027809, 0.0053678, -0.0025869, 0.0025869
5: 0.0050646, 0.0064029, 0.0050646, 0.0064029, -0.0013382, 0.0013382
6: -0.0021738, -0.0008562, -0.0021738, -0.0008562, -0.0013176, 0.0013176
7: -0.0082275, -0.0074685, -0.0082275, -0.0074685, -0.0007590, 0.0007590
8: 0.0055062, 0.0098066, 0.0055062, 0.0098066, -0.0042203, 0.0042202
9: -0.0036825, -0.0031002, -0.0036825, -0.0031002, -0.0005823, 0.0005823

### Unstable ReLU Count (Linear/Conv2D Layers)
- layer_idx=0, type=LayerType.Linear, total=256, inp1_unstable=9, inp2_unstable=9, delta_unstable=249
- layer_idx=2, type=LayerType.Linear, total=256, inp1_unstable=6, inp2_unstable=6, delta_unstable=256
- layer_idx=4, type=LayerType.Linear, total=256, inp1_unstable=0, inp2_unstable=0, delta_unstable=256
- layer_idx=6, type=LayerType.Linear, total=256, inp1_unstable=2, inp2_unstable=2, delta_unstable=256
- layer_idx=8, type=LayerType.Linear, total=10, inp1_unstable=0, inp2_unstable=0, delta_unstable=10

Time for backsubstitution: 1.84 seconds

### RS candidates at layer 1
type: RSZ, layer: 1, pos: 52
type: RSZ, layer: 1, pos: 198

Time for candidate selection: 0.00 seconds

### Candidate
type: RSZ, layer: 1, pos: 52

### Relational analysis RSZ of RS_RSZ1_RSZ2_RSZ2_RSZ1_RSZ2_RSZ2_RSZ1_RSZ1

#### Relational analysis RSZ result of RS_RSZ1_RSZ2_RSZ2_RSZ1_RSZ2_RSZ2_RSZ1_RSZ1
Status: Status.UNKNOWN
Output dim: 1, lower bound: -0.0007542, upper bound: 0.0007534
time: 0.66 seconds

### Relational analysis RSZ of RS_RSZ1_RSZ2_RSZ2_RSZ1_RSZ2_RSZ2_RSZ1_RSZ2

#### Relational analysis RSZ result of RS_RSZ1_RSZ2_RSZ2_RSZ1_RSZ2_RSZ2_RSZ1_RSZ2
Status: Status.UNKNOWN
Output dim: 1, lower bound: -0.0007536, upper bound: 0.0007537
time: 0.70 seconds

## BFS RS instance: RS_RSZ1_RSZ2_RSZ2_RSZ1_RSZ2_RSZ2_RSZ2

### Backsubstitution after applying RS history:
0: 0.0007184, 0.0011172, 0.0007184, 0.0011172, -0.0003988, 0.0003988
1: 0.9934075, 0.9943579, 0.9934075, 0.9943579, -0.0009505, 0.0009505
2: -0.0087946, -0.0055215, -0.0087946, -0.0055215, -0.0029753, 0.0029771
3: 0.0035868, 0.0041619, 0.0035868, 0.0041619, -0.0005751, 0.0005751
4: 0.0027809, 0.0053678, 0.0027809, 0.0053678, -0.0025869, 0.0025869
5: 0.0050646, 0.0064029, 0.0050646, 0.0064029, -0.0013382, 0.0013382
6: -0.0021738, -0.0008562, -0.0021738, -0.0008562, -0.0013176, 0.0013176
7: -0.0082275, -0.0074685, -0.0082275, -0.0074685, -0.0007590, 0.0007590
8: 0.0055062, 0.0098066, 0.0055062, 0.0098066, -0.0042205, 0.0042200
9: -0.0036825, -0.0031002, -0.0036825, -0.0031002, -0.0005823, 0.0005823

### Unstable ReLU Count (Linear/Conv2D Layers)
- layer_idx=0, type=LayerType.Linear, total=256, inp1_unstable=9, inp2_unstable=9, delta_unstable=249
- layer_idx=2, type=LayerType.Linear, total=256, inp1_unstable=6, inp2_unstable=6, delta_unstable=256
- layer_idx=4, type=LayerType.Linear, total=256, inp1_unstable=0, inp2_unstable=0, delta_unstable=256
- layer_idx=6, type=LayerType.Linear, total=256, inp1_unstable=2, inp2_unstable=2, delta_unstable=256
- layer_idx=8, type=LayerType.Linear, total=10, inp1_unstable=0, inp2_unstable=0, delta_unstable=10

Time for backsubstitution: 1.80 seconds

### RS candidates at layer 1
type: RSZ, layer: 1, pos: 198
type: RSZ, layer: 1, pos: 52

Time for candidate selection: 0.00 seconds

### Candidate
type: RSZ, layer: 1, pos: 198

### Relational analysis RSZ of RS_RSZ1_RSZ2_RSZ2_RSZ1_RSZ2_RSZ2_RSZ2_RSZ1

#### Relational analysis RSZ result of RS_RSZ1_RSZ2_RSZ2_RSZ1_RSZ2_RSZ2_RSZ2_RSZ1
Status: Status.UNKNOWN
Output dim: 1, lower bound: -0.0007098, upper bound: 0.0007098
time: 0.73 seconds

### Relational analysis RSZ of RS_RSZ1_RSZ2_RSZ2_RSZ1_RSZ2_RSZ2_RSZ2_RSZ2

#### Relational analysis RSZ result of RS_RSZ1_RSZ2_RSZ2_RSZ1_RSZ2_RSZ2_RSZ2_RSZ2
Status: Status.UNKNOWN
Output dim: 1, lower bound: -0.0007098, upper bound: 0.0007098
time: 0.69 seconds

## BFS RS instance: RS_RSZ1_RSZ2_RSZ2_RSZ2_RSZ1_RSZ1_RSZ1

### Backsubstitution after applying RS history:
0: 0.0007184, 0.0011172, 0.0007184, 0.0011172, -0.0003988, 0.0003988
1: 0.9934075, 0.9943579, 0.9934075, 0.9943579, -0.0009505, 0.0009505
2: -0.0087946, -0.0055215, -0.0087946, -0.0055215, -0.0029711, 0.0029701
3: 0.0035868, 0.0041619, 0.0035868, 0.0041619, -0.0005751, 0.0005751
4: 0.0027809, 0.0053678, 0.0027809, 0.0053678, -0.0025869, 0.0025869
5: 0.0050646, 0.0064029, 0.0050646, 0.0064029, -0.0013382, 0.0013382
6: -0.0021738, -0.0008562, -0.0021738, -0.0008562, -0.0013176, 0.0013176
7: -0.0082275, -0.0074685, -0.0082275, -0.0074685, -0.0007590, 0.0007590
8: 0.0055062, 0.0098066, 0.0055062, 0.0098066, -0.0042187, 0.0042190
9: -0.0036825, -0.0031002, -0.0036825, -0.0031002, -0.0005823, 0.0005823

### Unstable ReLU Count (Linear/Conv2D Layers)
- layer_idx=0, type=LayerType.Linear, total=256, inp1_unstable=9, inp2_unstable=9, delta_unstable=249
- layer_idx=2, type=LayerType.Linear, total=256, inp1_unstable=6, inp2_unstable=6, delta_unstable=256
- layer_idx=4, type=LayerType.Linear, total=256, inp1_unstable=0, inp2_unstable=0, delta_unstable=256
- layer_idx=6, type=LayerType.Linear, total=256, inp1_unstable=2, inp2_unstable=2, delta_unstable=256
- layer_idx=8, type=LayerType.Linear, total=10, inp1_unstable=0, inp2_unstable=0, delta_unstable=10

Time for backsubstitution: 1.80 seconds

### RS candidates at layer 1
type: RSZ, layer: 1, pos: 83
type: RSZ, layer: 1, pos: 52

Time for candidate selection: 0.00 seconds

### Candidate
type: RSZ, layer: 1, pos: 83

### Relational analysis RSZ of RS_RSZ1_RSZ2_RSZ2_RSZ2_RSZ1_RSZ1_RSZ1_RSZ1

#### Relational analysis RSZ result of RS_RSZ1_RSZ2_RSZ2_RSZ2_RSZ1_RSZ1_RSZ1_RSZ1
Status: Status.UNKNOWN
Output dim: 1, lower bound: -0.0007098, upper bound: 0.0007098
time: 0.71 seconds

### Relational analysis RSZ of RS_RSZ1_RSZ2_RSZ2_RSZ2_RSZ1_RSZ1_RSZ1_RSZ2

#### Relational analysis RSZ result of RS_RSZ1_RSZ2_RSZ2_RSZ2_RSZ1_RSZ1_RSZ1_RSZ2
Status: Status.UNKNOWN
Output dim: 1, lower bound: -0.0007097, upper bound: 0.0007098
time: 0.67 seconds

## BFS RS instance: RS_RSZ1_RSZ2_RSZ2_RSZ2_RSZ1_RSZ1_RSZ2

### Backsubstitution after applying RS history:
0: 0.0007184, 0.0011172, 0.0007184, 0.0011172, -0.0003988, 0.0003988
1: 0.9934075, 0.9943579, 0.9934075, 0.9943579, -0.0009505, 0.0009505
2: -0.0087946, -0.0055215, -0.0087946, -0.0055215, -0.0029704, 0.0029707
3: 0.0035868, 0.0041619, 0.0035868, 0.0041619, -0.0005751, 0.0005751
4: 0.0027809, 0.0053678, 0.0027809, 0.0053678, -0.0025869, 0.0025869
5: 0.0050646, 0.0064029, 0.0050646, 0.0064029, -0.0013382, 0.0013382
6: -0.0021738, -0.0008562, -0.0021738, -0.0008562, -0.0013176, 0.0013176
7: -0.0082275, -0.0074685, -0.0082275, -0.0074685, -0.0007590, 0.0007590
8: 0.0055062, 0.0098066, 0.0055062, 0.0098066, -0.0042189, 0.0042188
9: -0.0036825, -0.0031002, -0.0036825, -0.0031002, -0.0005823, 0.0005823

### Unstable ReLU Count (Linear/Conv2D Layers)
- layer_idx=0, type=LayerType.Linear, total=256, inp1_unstable=9, inp2_unstable=9, delta_unstable=249
- layer_idx=2, type=LayerType.Linear, total=256, inp1_unstable=6, inp2_unstable=6, delta_unstable=256
- layer_idx=4, type=LayerType.Linear, total=256, inp1_unstable=0, inp2_unstable=0, delta_unstable=256
- layer_idx=6, type=LayerType.Linear, total=256, inp1_unstable=2, inp2_unstable=2, delta_unstable=256
- layer_idx=8, type=LayerType.Linear, total=10, inp1_unstable=0, inp2_unstable=0, delta_unstable=10

Time for backsubstitution: 1.85 seconds

### RS candidates at layer 1
type: RSZ, layer: 1, pos: 52
type: RSZ, layer: 1, pos: 83

Time for candidate selection: 0.00 seconds

### Candidate
type: RSZ, layer: 1, pos: 52

### Relational analysis RSZ of RS_RSZ1_RSZ2_RSZ2_RSZ2_RSZ1_RSZ1_RSZ2_RSZ1

#### Relational analysis RSZ result of RS_RSZ1_RSZ2_RSZ2_RSZ2_RSZ1_RSZ1_RSZ2_RSZ1
Status: Status.UNKNOWN
Output dim: 1, lower bound: -0.0007131, upper bound: 0.0007131
time: 0.66 seconds

### Relational analysis RSZ of RS_RSZ1_RSZ2_RSZ2_RSZ2_RSZ1_RSZ1_RSZ2_RSZ2

#### Relational analysis RSZ result of RS_RSZ1_RSZ2_RSZ2_RSZ2_RSZ1_RSZ1_RSZ2_RSZ2
Status: Status.UNKNOWN
Output dim: 1, lower bound: -0.0007131, upper bound: 0.0007131
time: 0.63 seconds

## BFS RS instance: RS_RSZ1_RSZ2_RSZ2_RSZ2_RSZ1_RSZ2_RSZ1

### Backsubstitution after applying RS history:
0: 0.0007184, 0.0011172, 0.0007184, 0.0011172, -0.0003988, 0.0003988
1: 0.9934075, 0.9943579, 0.9934075, 0.9943579, -0.0009505, 0.0009505
2: -0.0087946, -0.0055215, -0.0087946, -0.0055215, -0.0029696, 0.0029723
3: 0.0035868, 0.0041619, 0.0035868, 0.0041619, -0.0005751, 0.0005751
4: 0.0027809, 0.0053678, 0.0027809, 0.0053678, -0.0025869, 0.0025869
5: 0.0050646, 0.0064029, 0.0050646, 0.0064029, -0.0013382, 0.0013382
6: -0.0021738, -0.0008562, -0.0021738, -0.0008562, -0.0013176, 0.0013176
7: -0.0082275, -0.0074685, -0.0082275, -0.0074685, -0.0007590, 0.0007590
8: 0.0055062, 0.0098066, 0.0055062, 0.0098066, -0.0042193, 0.0042186
9: -0.0036825, -0.0031002, -0.0036825, -0.0031002, -0.0005823, 0.0005823

### Unstable ReLU Count (Linear/Conv2D Layers)
- layer_idx=0, type=LayerType.Linear, total=256, inp1_unstable=9, inp2_unstable=9, delta_unstable=249
- layer_idx=2, type=LayerType.Linear, total=256, inp1_unstable=6, inp2_unstable=6, delta_unstable=256
- layer_idx=4, type=LayerType.Linear, total=256, inp1_unstable=0, inp2_unstable=0, delta_unstable=256
- layer_idx=6, type=LayerType.Linear, total=256, inp1_unstable=2, inp2_unstable=2, delta_unstable=256
- layer_idx=8, type=LayerType.Linear, total=10, inp1_unstable=0, inp2_unstable=0, delta_unstable=10

Time for backsubstitution: 1.81 seconds

### RS candidates at layer 1
type: RSZ, layer: 1, pos: 52
type: RSZ, layer: 1, pos: 83

Time for candidate selection: 0.00 seconds

### Candidate
type: RSZ, layer: 1, pos: 52

### Relational analysis RSZ of RS_RSZ1_RSZ2_RSZ2_RSZ2_RSZ1_RSZ2_RSZ1_RSZ1

#### Relational analysis RSZ result of RS_RSZ1_RSZ2_RSZ2_RSZ2_RSZ1_RSZ2_RSZ1_RSZ1
Status: Status.UNKNOWN
Output dim: 1, lower bound: -0.0007131, upper bound: 0.0007131
time: 0.73 seconds

### Relational analysis RSZ of RS_RSZ1_RSZ2_RSZ2_RSZ2_RSZ1_RSZ2_RSZ1_RSZ2

#### Relational analysis RSZ result of RS_RSZ1_RSZ2_RSZ2_RSZ2_RSZ1_RSZ2_RSZ1_RSZ2
Status: Status.UNKNOWN
Output dim: 1, lower bound: -0.0007131, upper bound: 0.0007131
time: 0.68 seconds

## BFS RS instance: RS_RSZ1_RSZ2_RSZ2_RSZ2_RSZ1_RSZ2_RSZ2

### Backsubstitution after applying RS history:
0: 0.0007184, 0.0011172, 0.0007184, 0.0011172, -0.0003988, 0.0003988
1: 0.9934075, 0.9943579, 0.9934075, 0.9943579, -0.0009505, 0.0009505
2: -0.0087946, -0.0055215, -0.0087946, -0.0055215, -0.0029686, 0.0029727
3: 0.0035868, 0.0041619, 0.0035868, 0.0041619, -0.0005751, 0.0005751
4: 0.0027809, 0.0053678, 0.0027809, 0.0053678, -0.0025869, 0.0025869
5: 0.0050646, 0.0064029, 0.0050646, 0.0064029, -0.0013382, 0.0013382
6: -0.0021738, -0.0008562, -0.0021738, -0.0008562, -0.0013176, 0.0013176
7: -0.0082275, -0.0074685, -0.0082275, -0.0074685, -0.0007590, 0.0007590
8: 0.0055062, 0.0098066, 0.0055062, 0.0098066, -0.0042194, 0.0042184
9: -0.0036825, -0.0031002, -0.0036825, -0.0031002, -0.0005823, 0.0005823

### Unstable ReLU Count (Linear/Conv2D Layers)
- layer_idx=0, type=LayerType.Linear, total=256, inp1_unstable=9, inp2_unstable=9, delta_unstable=249
- layer_idx=2, type=LayerType.Linear, total=256, inp1_unstable=6, inp2_unstable=6, delta_unstable=256
- layer_idx=4, type=LayerType.Linear, total=256, inp1_unstable=0, inp2_unstable=0, delta_unstable=256
- layer_idx=6, type=LayerType.Linear, total=256, inp1_unstable=2, inp2_unstable=2, delta_unstable=256
- layer_idx=8, type=LayerType.Linear, total=10, inp1_unstable=0, inp2_unstable=0, delta_unstable=10

Time for backsubstitution: 1.80 seconds

### RS candidates at layer 1
type: RSZ, layer: 1, pos: 83
type: RSZ, layer: 1, pos: 52

Time for candidate selection: 0.00 seconds

### Candidate
type: RSZ, layer: 1, pos: 83

### Relational analysis RSZ of RS_RSZ1_RSZ2_RSZ2_RSZ2_RSZ1_RSZ2_RSZ2_RSZ1

#### Relational analysis RSZ result of RS_RSZ1_RSZ2_RSZ2_RSZ2_RSZ1_RSZ2_RSZ2_RSZ1
Status: Status.UNKNOWN
Output dim: 1, lower bound: -0.0007097, upper bound: 0.0007098
time: 0.68 seconds

### Relational analysis RSZ of RS_RSZ1_RSZ2_RSZ2_RSZ2_RSZ1_RSZ2_RSZ2_RSZ2

#### Relational analysis RSZ result of RS_RSZ1_RSZ2_RSZ2_RSZ2_RSZ1_RSZ2_RSZ2_RSZ2
Status: Status.UNKNOWN
Output dim: 1, lower bound: -0.0007098, upper bound: 0.0007098
time: 0.71 seconds

## BFS RS instance: RS_RSZ1_RSZ2_RSZ2_RSZ2_RSZ2_RSZ1_RSZ1

### Backsubstitution after applying RS history:
0: 0.0007184, 0.0011172, 0.0007184, 0.0011172, -0.0003988, 0.0003988
1: 0.9934075, 0.9943579, 0.9934075, 0.9943579, -0.0009505, 0.0009505
2: -0.0087946, -0.0055215, -0.0087946, -0.0055215, -0.0029835, 0.0029729
3: 0.0035868, 0.0041619, 0.0035868, 0.0041619, -0.0005751, 0.0005751
4: 0.0027809, 0.0053678, 0.0027809, 0.0053678, -0.0025869, 0.0025869
5: 0.0050646, 0.0064029, 0.0050646, 0.0064029, -0.0013382, 0.0013382
6: -0.0021738, -0.0008562, -0.0021738, -0.0008562, -0.0013176, 0.0013176
7: -0.0082275, -0.0074685, -0.0082275, -0.0074685, -0.0007590, 0.0007590
8: 0.0055062, 0.0098066, 0.0055062, 0.0098066, -0.0042194, 0.0042220
9: -0.0036825, -0.0031002, -0.0036825, -0.0031002, -0.0005823, 0.0005823

### Unstable ReLU Count (Linear/Conv2D Layers)
- layer_idx=0, type=LayerType.Linear, total=256, inp1_unstable=9, inp2_unstable=9, delta_unstable=249
- layer_idx=2, type=LayerType.Linear, total=256, inp1_unstable=6, inp2_unstable=6, delta_unstable=256
- layer_idx=4, type=LayerType.Linear, total=256, inp1_unstable=0, inp2_unstable=0, delta_unstable=256
- layer_idx=6, type=LayerType.Linear, total=256, inp1_unstable=2, inp2_unstable=2, delta_unstable=256
- layer_idx=8, type=LayerType.Linear, total=10, inp1_unstable=0, inp2_unstable=0, delta_unstable=10

Time for backsubstitution: 1.81 seconds

### RS candidates at layer 1
type: RSZ, layer: 1, pos: 83
type: RSZ, layer: 1, pos: 204

Time for candidate selection: 0.00 seconds

### Candidate
type: RSZ, layer: 1, pos: 83

### Relational analysis RSZ of RS_RSZ1_RSZ2_RSZ2_RSZ2_RSZ2_RSZ1_RSZ1_RSZ1

#### Relational analysis RSZ result of RS_RSZ1_RSZ2_RSZ2_RSZ2_RSZ2_RSZ1_RSZ1_RSZ1
Status: Status.UNKNOWN
Output dim: 1, lower bound: -0.0007098, upper bound: 0.0007098
time: 0.71 seconds

### Relational analysis RSZ of RS_RSZ1_RSZ2_RSZ2_RSZ2_RSZ2_RSZ1_RSZ1_RSZ2

#### Relational analysis RSZ result of RS_RSZ1_RSZ2_RSZ2_RSZ2_RSZ2_RSZ1_RSZ1_RSZ2
Status: Status.UNKNOWN
Output dim: 1, lower bound: -0.0007098, upper bound: 0.0007098
time: 0.62 seconds

## BFS RS instance: RS_RSZ1_RSZ2_RSZ2_RSZ2_RSZ2_RSZ1_RSZ2

### Backsubstitution after applying RS history:
0: 0.0007184, 0.0011172, 0.0007184, 0.0011172, -0.0003988, 0.0003988
1: 0.9934075, 0.9943579, 0.9934075, 0.9943579, -0.0009505, 0.0009505
2: -0.0087946, -0.0055215, -0.0087946, -0.0055215, -0.0029818, 0.0029745
3: 0.0035868, 0.0041619, 0.0035868, 0.0041619, -0.0005751, 0.0005751
4: 0.0027809, 0.0053678, 0.0027809, 0.0053678, -0.0025869, 0.0025869
5: 0.0050646, 0.0064029, 0.0050646, 0.0064029, -0.0013382, 0.0013382
6: -0.0021738, -0.0008562, -0.0021738, -0.0008562, -0.0013176, 0.0013176
7: -0.0082275, -0.0074685, -0.0082275, -0.0074685, -0.0007590, 0.0007590
8: 0.0055062, 0.0098066, 0.0055062, 0.0098066, -0.0042198, 0.0042215
9: -0.0036825, -0.0031002, -0.0036825, -0.0031002, -0.0005823, 0.0005823

### Unstable ReLU Count (Linear/Conv2D Layers)
- layer_idx=0, type=LayerType.Linear, total=256, inp1_unstable=9, inp2_unstable=9, delta_unstable=249
- layer_idx=2, type=LayerType.Linear, total=256, inp1_unstable=6, inp2_unstable=6, delta_unstable=256
- layer_idx=4, type=LayerType.Linear, total=256, inp1_unstable=0, inp2_unstable=0, delta_unstable=256
- layer_idx=6, type=LayerType.Linear, total=256, inp1_unstable=2, inp2_unstable=2, delta_unstable=256
- layer_idx=8, type=LayerType.Linear, total=10, inp1_unstable=0, inp2_unstable=0, delta_unstable=10

Time for backsubstitution: 1.75 seconds

### RS candidates at layer 1
type: RSZ, layer: 1, pos: 83
type: RSZ, layer: 1, pos: 204

Time for candidate selection: 0.00 seconds

### Candidate
type: RSZ, layer: 1, pos: 83

### Relational analysis RSZ of RS_RSZ1_RSZ2_RSZ2_RSZ2_RSZ2_RSZ1_RSZ2_RSZ1

#### Relational analysis RSZ result of RS_RSZ1_RSZ2_RSZ2_RSZ2_RSZ2_RSZ1_RSZ2_RSZ1
Status: Status.UNKNOWN
Output dim: 1, lower bound: -0.0007097, upper bound: 0.0007098
time: 0.70 seconds

### Relational analysis RSZ of RS_RSZ1_RSZ2_RSZ2_RSZ2_RSZ2_RSZ1_RSZ2_RSZ2

#### Relational analysis RSZ result of RS_RSZ1_RSZ2_RSZ2_RSZ2_RSZ2_RSZ1_RSZ2_RSZ2
Status: Status.UNKNOWN
Output dim: 1, lower bound: -0.0007097, upper bound: 0.0007098
time: 0.66 seconds

## BFS RS instance: RS_RSZ1_RSZ2_RSZ2_RSZ2_RSZ2_RSZ2_RSZ1

### Backsubstitution after applying RS history:
0: 0.0007184, 0.0011172, 0.0007184, 0.0011172, -0.0003988, 0.0003988
1: 0.9934075, 0.9943579, 0.9934075, 0.9943579, -0.0009505, 0.0009505
2: -0.0087946, -0.0055215, -0.0087946, -0.0055215, -0.0029868, 0.0029749
3: 0.0035868, 0.0041619, 0.0035868, 0.0041619, -0.0005751, 0.0005751
4: 0.0027809, 0.0053678, 0.0027809, 0.0053678, -0.0025869, 0.0025869
5: 0.0050646, 0.0064029, 0.0050646, 0.0064029, -0.0013382, 0.0013382
6: -0.0021738, -0.0008562, -0.0021738, -0.0008562, -0.0013176, 0.0013176
7: -0.0082275, -0.0074685, -0.0082275, -0.0074685, -0.0007590, 0.0007590
8: 0.0055062, 0.0098066, 0.0055062, 0.0098066, -0.0042204, 0.0042233
9: -0.0036825, -0.0031002, -0.0036825, -0.0031002, -0.0005823, 0.0005823

### Unstable ReLU Count (Linear/Conv2D Layers)
- layer_idx=0, type=LayerType.Linear, total=256, inp1_unstable=9, inp2_unstable=9, delta_unstable=249
- layer_idx=2, type=LayerType.Linear, total=256, inp1_unstable=6, inp2_unstable=6, delta_unstable=256
- layer_idx=4, type=LayerType.Linear, total=256, inp1_unstable=0, inp2_unstable=0, delta_unstable=256
- layer_idx=6, type=LayerType.Linear, total=256, inp1_unstable=2, inp2_unstable=2, delta_unstable=256
- layer_idx=8, type=LayerType.Linear, total=10, inp1_unstable=0, inp2_unstable=0, delta_unstable=10

Time for backsubstitution: 1.85 seconds

### RS candidates at layer 1
type: RSZ, layer: 1, pos: 68
type: RSZ, layer: 1, pos: 83

Time for candidate selection: 0.00 seconds

### Candidate
type: RSZ, layer: 1, pos: 68

### Relational analysis RSZ of RS_RSZ1_RSZ2_RSZ2_RSZ2_RSZ2_RSZ2_RSZ1_RSZ1

#### Relational analysis RSZ result of RS_RSZ1_RSZ2_RSZ2_RSZ2_RSZ2_RSZ2_RSZ1_RSZ1
Status: Status.UNKNOWN
Output dim: 1, lower bound: -0.0007131, upper bound: 0.0007131
time: 0.65 seconds

### Relational analysis RSZ of RS_RSZ1_RSZ2_RSZ2_RSZ2_RSZ2_RSZ2_RSZ1_RSZ2

#### Relational analysis RSZ result of RS_RSZ1_RSZ2_RSZ2_RSZ2_RSZ2_RSZ2_RSZ1_RSZ2
Status: Status.UNKNOWN
Output dim: 1, lower bound: -0.0007131, upper bound: 0.0007131
time: 0.70 seconds

## BFS RS instance: RS_RSZ1_RSZ2_RSZ2_RSZ2_RSZ2_RSZ2_RSZ2

### Backsubstitution after applying RS history:
0: 0.0007184, 0.0011172, 0.0007184, 0.0011172, -0.0003988, 0.0003988
1: 0.9934075, 0.9943579, 0.9934075, 0.9943579, -0.0009505, 0.0009505
2: -0.0087946, -0.0055215, -0.0087946, -0.0055215, -0.0029859, 0.0029754
3: 0.0035868, 0.0041619, 0.0035868, 0.0041619, -0.0005751, 0.0005751
4: 0.0027809, 0.0053678, 0.0027809, 0.0053678, -0.0025869, 0.0025869
5: 0.0050646, 0.0064029, 0.0050646, 0.0064029, -0.0013382, 0.0013382
6: -0.0021738, -0.0008562, -0.0021738, -0.0008562, -0.0013176, 0.0013176
7: -0.0082275, -0.0074685, -0.0082275, -0.0074685, -0.0007590, 0.0007590
8: 0.0055062, 0.0098066, 0.0055062, 0.0098066, -0.0042205, 0.0042230
9: -0.0036825, -0.0031002, -0.0036825, -0.0031002, -0.0005823, 0.0005823

### Unstable ReLU Count (Linear/Conv2D Layers)
- layer_idx=0, type=LayerType.Linear, total=256, inp1_unstable=9, inp2_unstable=9, delta_unstable=249
- layer_idx=2, type=LayerType.Linear, total=256, inp1_unstable=6, inp2_unstable=6, delta_unstable=256
- layer_idx=4, type=LayerType.Linear, total=256, inp1_unstable=0, inp2_unstable=0, delta_unstable=256
- layer_idx=6, type=LayerType.Linear, total=256, inp1_unstable=2, inp2_unstable=2, delta_unstable=256
- layer_idx=8, type=LayerType.Linear, total=10, inp1_unstable=0, inp2_unstable=0, delta_unstable=10

Time for backsubstitution: 1.84 seconds

### RS candidates at layer 1
type: RSZ, layer: 1, pos: 68
type: RSZ, layer: 1, pos: 83

Time for candidate selection: 0.00 seconds

### Candidate
type: RSZ, layer: 1, pos: 68

### Relational analysis RSZ of RS_RSZ1_RSZ2_RSZ2_RSZ2_RSZ2_RSZ2_RSZ2_RSZ1

#### Relational analysis RSZ result of RS_RSZ1_RSZ2_RSZ2_RSZ2_RSZ2_RSZ2_RSZ2_RSZ1
Status: Status.UNKNOWN
Output dim: 1, lower bound: -0.0007131, upper bound: 0.0007131
time: 0.68 seconds

### Relational analysis RSZ of RS_RSZ1_RSZ2_RSZ2_RSZ2_RSZ2_RSZ2_RSZ2_RSZ2

#### Relational analysis RSZ result of RS_RSZ1_RSZ2_RSZ2_RSZ2_RSZ2_RSZ2_RSZ2_RSZ2
Status: Status.UNKNOWN
Output dim: 1, lower bound: -0.0007131, upper bound: 0.0007131
time: 0.66 seconds

## BFS RS instance: RS_RSZ2_RSZ1_RSZ1_RSZ1_RSZ1_RSZ1_RSZ1

### Backsubstitution after applying RS history:
0: 0.0007184, 0.0011172, 0.0007184, 0.0011172, -0.0003988, 0.0003988
1: 0.9934075, 0.9943579, 0.9934075, 0.9943579, -0.0009505, 0.0009505
2: -0.0087946, -0.0055215, -0.0087946, -0.0055215, -0.0029719, 0.0029704
3: 0.0035868, 0.0041619, 0.0035868, 0.0041619, -0.0005751, 0.0005751
4: 0.0027809, 0.0053678, 0.0027809, 0.0053678, -0.0025869, 0.0025869
5: 0.0050646, 0.0064029, 0.0050646, 0.0064029, -0.0013382, 0.0013382
6: -0.0021738, -0.0008562, -0.0021738, -0.0008562, -0.0013176, 0.0013176
7: -0.0082275, -0.0074685, -0.0082275, -0.0074685, -0.0007590, 0.0007590
8: 0.0055062, 0.0098066, 0.0055062, 0.0098066, -0.0042184, 0.0042188
9: -0.0036825, -0.0031002, -0.0036825, -0.0031002, -0.0005823, 0.0005823

### Unstable ReLU Count (Linear/Conv2D Layers)
- layer_idx=0, type=LayerType.Linear, total=256, inp1_unstable=9, inp2_unstable=9, delta_unstable=249
- layer_idx=2, type=LayerType.Linear, total=256, inp1_unstable=6, inp2_unstable=6, delta_unstable=256
- layer_idx=4, type=LayerType.Linear, total=256, inp1_unstable=0, inp2_unstable=0, delta_unstable=256
- layer_idx=6, type=LayerType.Linear, total=256, inp1_unstable=2, inp2_unstable=2, delta_unstable=256
- layer_idx=8, type=LayerType.Linear, total=10, inp1_unstable=0, inp2_unstable=0, delta_unstable=10

Time for backsubstitution: 1.82 seconds

### RS candidates at layer 1
type: RSZ, layer: 1, pos: 136
type: RSZ, layer: 1, pos: 199

Time for candidate selection: 0.00 seconds

### Candidate
type: RSZ, layer: 1, pos: 136

### Relational analysis RSZ of RS_RSZ2_RSZ1_RSZ1_RSZ1_RSZ1_RSZ1_RSZ1_RSZ1

#### Relational analysis RSZ result of RS_RSZ2_RSZ1_RSZ1_RSZ1_RSZ1_RSZ1_RSZ1_RSZ1
Status: Status.UNKNOWN
Output dim: 1, lower bound: -0.0007679, upper bound: 0.0007702
time: 0.73 seconds

### Relational analysis RSZ of RS_RSZ2_RSZ1_RSZ1_RSZ1_RSZ1_RSZ1_RSZ1_RSZ2

#### Relational analysis RSZ result of RS_RSZ2_RSZ1_RSZ1_RSZ1_RSZ1_RSZ1_RSZ1_RSZ2
Status: Status.UNKNOWN
Output dim: 1, lower bound: -0.0007679, upper bound: 0.0007701
time: 0.67 seconds

## BFS RS instance: RS_RSZ2_RSZ1_RSZ1_RSZ1_RSZ1_RSZ1_RSZ2

### Backsubstitution after applying RS history:
0: 0.0007184, 0.0011172, 0.0007184, 0.0011172, -0.0003988, 0.0003988
1: 0.9934075, 0.9943579, 0.9934075, 0.9943579, -0.0009505, 0.0009505
2: -0.0087946, -0.0055215, -0.0087946, -0.0055215, -0.0029824, 0.0029696
3: 0.0035868, 0.0041619, 0.0035868, 0.0041619, -0.0005751, 0.0005751
4: 0.0027809, 0.0053678, 0.0027809, 0.0053678, -0.0025869, 0.0025869
5: 0.0050646, 0.0064029, 0.0050646, 0.0064029, -0.0013382, 0.0013382
6: -0.0021738, -0.0008562, -0.0021738, -0.0008562, -0.0013176, 0.0013176
7: -0.0082275, -0.0074685, -0.0082275, -0.0074685, -0.0007590, 0.0007590
8: 0.0055062, 0.0098066, 0.0055062, 0.0098066, -0.0042182, 0.0042213
9: -0.0036825, -0.0031002, -0.0036825, -0.0031002, -0.0005823, 0.0005823

### Unstable ReLU Count (Linear/Conv2D Layers)
- layer_idx=0, type=LayerType.Linear, total=256, inp1_unstable=9, inp2_unstable=9, delta_unstable=249
- layer_idx=2, type=LayerType.Linear, total=256, inp1_unstable=6, inp2_unstable=6, delta_unstable=256
- layer_idx=4, type=LayerType.Linear, total=256, inp1_unstable=0, inp2_unstable=0, delta_unstable=256
- layer_idx=6, type=LayerType.Linear, total=256, inp1_unstable=2, inp2_unstable=2, delta_unstable=256
- layer_idx=8, type=LayerType.Linear, total=10, inp1_unstable=0, inp2_unstable=0, delta_unstable=10

Time for backsubstitution: 1.83 seconds

### RS candidates at layer 1
type: RSZ, layer: 1, pos: 136
type: RSZ, layer: 1, pos: 199

Time for candidate selection: 0.00 seconds

### Candidate
type: RSZ, layer: 1, pos: 136

### Relational analysis RSZ of RS_RSZ2_RSZ1_RSZ1_RSZ1_RSZ1_RSZ1_RSZ2_RSZ1

#### Relational analysis RSZ result of RS_RSZ2_RSZ1_RSZ1_RSZ1_RSZ1_RSZ1_RSZ2_RSZ1
Status: Status.UNKNOWN
Output dim: 1, lower bound: -0.0007679, upper bound: 0.0007702
time: 0.73 seconds

### Relational analysis RSZ of RS_RSZ2_RSZ1_RSZ1_RSZ1_RSZ1_RSZ1_RSZ2_RSZ2

#### Relational analysis RSZ result of RS_RSZ2_RSZ1_RSZ1_RSZ1_RSZ1_RSZ1_RSZ2_RSZ2
Status: Status.UNKNOWN
Output dim: 1, lower bound: -0.0007679, upper bound: 0.0007701
time: 0.76 seconds

## BFS RS instance: RS_RSZ2_RSZ1_RSZ1_RSZ1_RSZ1_RSZ2_RSZ1

### Backsubstitution after applying RS history:
0: 0.0007184, 0.0011172, 0.0007184, 0.0011172, -0.0003988, 0.0003988
1: 0.9934075, 0.9943579, 0.9934075, 0.9943579, -0.0009505, 0.0009505
2: -0.0087946, -0.0055215, -0.0087946, -0.0055215, -0.0029751, 0.0029761
3: 0.0035868, 0.0041619, 0.0035868, 0.0041619, -0.0005751, 0.0005751
4: 0.0027809, 0.0053678, 0.0027809, 0.0053678, -0.0025869, 0.0025869
5: 0.0050646, 0.0064029, 0.0050646, 0.0064029, -0.0013382, 0.0013382
6: -0.0021738, -0.0008562, -0.0021738, -0.0008562, -0.0013176, 0.0013176
7: -0.0082275, -0.0074685, -0.0082275, -0.0074685, -0.0007590, 0.0007590
8: 0.0055062, 0.0098066, 0.0055062, 0.0098066, -0.0042201, 0.0042198
9: -0.0036825, -0.0031002, -0.0036825, -0.0031002, -0.0005823, 0.0005823

### Unstable ReLU Count (Linear/Conv2D Layers)
- layer_idx=0, type=LayerType.Linear, total=256, inp1_unstable=9, inp2_unstable=9, delta_unstable=249
- layer_idx=2, type=LayerType.Linear, total=256, inp1_unstable=6, inp2_unstable=6, delta_unstable=256
- layer_idx=4, type=LayerType.Linear, total=256, inp1_unstable=0, inp2_unstable=0, delta_unstable=256
- layer_idx=6, type=LayerType.Linear, total=256, inp1_unstable=2, inp2_unstable=2, delta_unstable=256
- layer_idx=8, type=LayerType.Linear, total=10, inp1_unstable=0, inp2_unstable=0, delta_unstable=10

Time for backsubstitution: 1.78 seconds

### RS candidates at layer 1
type: RSZ, layer: 1, pos: 199
type: RSZ, layer: 1, pos: 198

Time for candidate selection: 0.00 seconds

### Candidate
type: RSZ, layer: 1, pos: 199

### Relational analysis RSZ of RS_RSZ2_RSZ1_RSZ1_RSZ1_RSZ1_RSZ2_RSZ1_RSZ1

#### Relational analysis RSZ result of RS_RSZ2_RSZ1_RSZ1_RSZ1_RSZ1_RSZ2_RSZ1_RSZ1
Status: Status.UNKNOWN
Output dim: 1, lower bound: -0.0007533, upper bound: 0.0007537
time: 0.72 seconds

### Relational analysis RSZ of RS_RSZ2_RSZ1_RSZ1_RSZ1_RSZ1_RSZ2_RSZ1_RSZ2

#### Relational analysis RSZ result of RS_RSZ2_RSZ1_RSZ1_RSZ1_RSZ1_RSZ2_RSZ1_RSZ2
Status: Status.UNKNOWN
Output dim: 1, lower bound: -0.0007533, upper bound: 0.0007537
time: 0.70 seconds

## BFS RS instance: RS_RSZ2_RSZ1_RSZ1_RSZ1_RSZ1_RSZ2_RSZ2

### Backsubstitution after applying RS history:
0: 0.0007184, 0.0011172, 0.0007184, 0.0011172, -0.0003988, 0.0003988
1: 0.9934075, 0.9943579, 0.9934075, 0.9943579, -0.0009505, 0.0009505
2: -0.0087946, -0.0055215, -0.0087946, -0.0055215, -0.0029775, 0.0029739
3: 0.0035868, 0.0041619, 0.0035868, 0.0041619, -0.0005751, 0.0005751
4: 0.0027809, 0.0053678, 0.0027809, 0.0053678, -0.0025869, 0.0025869
5: 0.0050646, 0.0064029, 0.0050646, 0.0064029, -0.0013382, 0.0013382
6: -0.0021738, -0.0008562, -0.0021738, -0.0008562, -0.0013176, 0.0013176
7: -0.0082275, -0.0074685, -0.0082275, -0.0074685, -0.0007590, 0.0007590
8: 0.0055062, 0.0098066, 0.0055062, 0.0098066, -0.0042195, 0.0042205
9: -0.0036825, -0.0031002, -0.0036825, -0.0031002, -0.0005823, 0.0005823

### Unstable ReLU Count (Linear/Conv2D Layers)
- layer_idx=0, type=LayerType.Linear, total=256, inp1_unstable=9, inp2_unstable=9, delta_unstable=249
- layer_idx=2, type=LayerType.Linear, total=256, inp1_unstable=6, inp2_unstable=6, delta_unstable=256
- layer_idx=4, type=LayerType.Linear, total=256, inp1_unstable=0, inp2_unstable=0, delta_unstable=256
- layer_idx=6, type=LayerType.Linear, total=256, inp1_unstable=2, inp2_unstable=2, delta_unstable=256
- layer_idx=8, type=LayerType.Linear, total=10, inp1_unstable=0, inp2_unstable=0, delta_unstable=10

Time for backsubstitution: 1.79 seconds

### RS candidates at layer 1
type: RSZ, layer: 1, pos: 199
type: RSZ, layer: 1, pos: 198

Time for candidate selection: 0.00 seconds

### Candidate
type: RSZ, layer: 1, pos: 199

### Relational analysis RSZ of RS_RSZ2_RSZ1_RSZ1_RSZ1_RSZ1_RSZ2_RSZ2_RSZ1

#### Relational analysis RSZ result of RS_RSZ2_RSZ1_RSZ1_RSZ1_RSZ1_RSZ2_RSZ2_RSZ1
Status: Status.UNKNOWN
Output dim: 1, lower bound: -0.0007531, upper bound: 0.0007542
time: 0.70 seconds

### Relational analysis RSZ of RS_RSZ2_RSZ1_RSZ1_RSZ1_RSZ1_RSZ2_RSZ2_RSZ2

#### Relational analysis RSZ result of RS_RSZ2_RSZ1_RSZ1_RSZ1_RSZ1_RSZ2_RSZ2_RSZ2
Status: Status.UNKNOWN
Output dim: 1, lower bound: -0.0007531, upper bound: 0.0007542
time: 0.70 seconds

## BFS RS instance: RS_RSZ2_RSZ1_RSZ1_RSZ1_RSZ2_RSZ1_RSZ1

### Backsubstitution after applying RS history:
0: 0.0007184, 0.0011172, 0.0007184, 0.0011172, -0.0003988, 0.0003988
1: 0.9934075, 0.9943579, 0.9934075, 0.9943579, -0.0009505, 0.0009505
2: -0.0087946, -0.0055215, -0.0087946, -0.0055215, -0.0029808, 0.0029772
3: 0.0035868, 0.0041619, 0.0035868, 0.0041619, -0.0005751, 0.0005751
4: 0.0027809, 0.0053678, 0.0027809, 0.0053678, -0.0025869, 0.0025869
5: 0.0050646, 0.0064029, 0.0050646, 0.0064029, -0.0013382, 0.0013382
6: -0.0021738, -0.0008562, -0.0021738, -0.0008562, -0.0013176, 0.0013176
7: -0.0082275, -0.0074685, -0.0082275, -0.0074685, -0.0007590, 0.0007590
8: 0.0055062, 0.0098066, 0.0055062, 0.0098066, -0.0042200, 0.0042210
9: -0.0036825, -0.0031002, -0.0036825, -0.0031002, -0.0005823, 0.0005823

### Unstable ReLU Count (Linear/Conv2D Layers)
- layer_idx=0, type=LayerType.Linear, total=256, inp1_unstable=9, inp2_unstable=9, delta_unstable=249
- layer_idx=2, type=LayerType.Linear, total=256, inp1_unstable=6, inp2_unstable=6, delta_unstable=256
- layer_idx=4, type=LayerType.Linear, total=256, inp1_unstable=0, inp2_unstable=0, delta_unstable=256
- layer_idx=6, type=LayerType.Linear, total=256, inp1_unstable=2, inp2_unstable=2, delta_unstable=256
- layer_idx=8, type=LayerType.Linear, total=10, inp1_unstable=0, inp2_unstable=0, delta_unstable=10

Time for backsubstitution: 1.87 seconds

## RS Result
status: Status.UNKNOWN
execution time: (base) + (rs) = 3.22 + 598.38 = 601.60 seconds
