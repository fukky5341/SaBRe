## Execution arguments:
Dataset: Dataset.MNIST
Network: onnx/mnist-net_256x4.onnx
Relational property: GLOBAL_ROBUSTNESS
LP Analysis: True
Epsilon: 0.03515625
Delta epsilon: 0.01171875
execution index: (3, 3, 0)
Time budget: 600 seconds
Split limit: 100
Threshold: 0.00079709


## IAR start

### Unstable ReLU Count (Linear/Conv2D Layers)
- layer_idx=0, type=LayerType.Linear, total=256, inp1_unstable=10, inp2_unstable=10, delta_unstable=256
- layer_idx=2, type=LayerType.Linear, total=256, inp1_unstable=10, inp2_unstable=10, delta_unstable=256
- layer_idx=4, type=LayerType.Linear, total=256, inp1_unstable=0, inp2_unstable=0, delta_unstable=256
- layer_idx=6, type=LayerType.Linear, total=256, inp1_unstable=0, inp2_unstable=0, delta_unstable=256
- layer_idx=8, type=LayerType.Linear, total=10, inp1_unstable=0, inp2_unstable=0, delta_unstable=10

### BASE IAR bounds
Layer (inp1_lb, inp1_ub, inp2_lb, inp2_ub, d_lb, d_ub)
0: (-0.0027939, -0.0016242, -0.0027939, -0.0016242, -0.0008194, 0.0008194)
1: (-0.0114005, -0.0084323, -0.0114005, -0.0084323, -0.0020794, 0.0020794)
2: (0.0279571, 0.0297986, 0.0279571, 0.0297986, -0.0012901, 0.0012901)
3: (0.0039210, 0.0073596, 0.0039210, 0.0073596, -0.0024089, 0.0024089)
4: (-0.0104893, -0.0074701, -0.0104893, -0.0074701, -0.0021151, 0.0021151)
5: (0.0097651, 0.0109087, 0.0097651, 0.0109087, -0.0008011, 0.0008011)
6: (0.0053643, 0.0097284, 0.0053643, 0.0097284, -0.0030572, 0.0030572)
7: (0.9818129, 0.9848667, 0.9818129, 0.9848667, -0.0021393, 0.0021393)
8: (-0.0060636, -0.0027895, -0.0060636, -0.0027895, -0.0022936, 0.0022936)
9: (-0.0031570, -0.0009942, -0.0031570, -0.0009942, -0.0015151, 0.0015151)

## BASE Relational Analysis

## BASE Result
execution time: IAR + RelationalAnalysis = 1.64 + 1.49 = 3.13 seconds
status: Status.UNKNOWN
relational distance
Output dim: 7, lower bound: -0.0012007, upper bound: 0.0012006

# Relational Split (RS) starts

## BFS RS instance: RS

Time for backsubstitution: 0.01 seconds

### RS candidates at layer 1
type: RSZ, layer: 1, pos: 35
type: RSZ, layer: 1, pos: 81
type: RSZ, layer: 1, pos: 85
type: RSZ, layer: 1, pos: 86
type: RSZ, layer: 1, pos: 105
type: RSZ, layer: 1, pos: 123
type: RSZ, layer: 1, pos: 140
type: RSZ, layer: 1, pos: 167
type: RSZ, layer: 1, pos: 213
type: RSZ, layer: 1, pos: 219

Time for candidate selection: 0.15 seconds

### Candidate
type: RSZ, layer: 1, pos: 35

### Relational analysis RSZ of RS_RSZ1

#### Relational analysis RSZ result of RS_RSZ1
Status: Status.UNKNOWN
Output dim: 7, lower bound: -0.0011909, upper bound: 0.0011695
time: 0.58 seconds

### Relational analysis RSZ of RS_RSZ2

#### Relational analysis RSZ result of RS_RSZ2
Status: Status.UNKNOWN
Output dim: 7, lower bound: -0.0011695, upper bound: 0.0011909
time: 0.63 seconds

## Summary of splitting (split count: 0)
- Time for RS candidates: 1.38 seconds
RS_RSZ1, status: Status.UNKNOWN, split count: 1, time: 1.38
Output dim: 7, lower bound: -0.0011909, upper bound: 0.0011695
RS_RSZ2, status: Status.UNKNOWN, split count: 1, time: 1.38
Output dim: 7, lower bound: -0.0011695, upper bound: 0.0011909

## BFS RS instance: RS_RSZ1

### Backsubstitution after applying RS history:
0: -0.0027939, -0.0016242, -0.0027939, -0.0016242, -0.0008094, 0.0008116
1: -0.0114005, -0.0084323, -0.0114005, -0.0084323, -0.0020541, 0.0020596
2: 0.0279571, 0.0297986, 0.0279571, 0.0297986, -0.0012744, 0.0012778
3: 0.0039210, 0.0073596, 0.0039210, 0.0073596, -0.0023860, 0.0023796
4: -0.0104893, -0.0074701, -0.0104893, -0.0074701, -0.0020893, 0.0020950
5: 0.0097651, 0.0109087, 0.0097651, 0.0109087, -0.0007914, 0.0007935
6: 0.0053643, 0.0097284, 0.0053643, 0.0097284, -0.0030281, 0.0030200
7: 0.9818129, 0.9848667, 0.9818129, 0.9848667, -0.0021189, 0.0021132
8: -0.0060636, -0.0027895, -0.0060636, -0.0027895, -0.0022718, 0.0022657
9: -0.0031570, -0.0009942, -0.0031570, -0.0009942, -0.0014966, 0.0015007

### Unstable ReLU Count (Linear/Conv2D Layers)
- layer_idx=0, type=LayerType.Linear, total=256, inp1_unstable=10, inp2_unstable=10, delta_unstable=255
- layer_idx=2, type=LayerType.Linear, total=256, inp1_unstable=10, inp2_unstable=10, delta_unstable=256
- layer_idx=4, type=LayerType.Linear, total=256, inp1_unstable=0, inp2_unstable=0, delta_unstable=256
- layer_idx=6, type=LayerType.Linear, total=256, inp1_unstable=0, inp2_unstable=0, delta_unstable=256
- layer_idx=8, type=LayerType.Linear, total=10, inp1_unstable=0, inp2_unstable=0, delta_unstable=10

Time for backsubstitution: 1.47 seconds

### RS candidates at layer 1
type: RSZ, layer: 1, pos: 81
type: RSZ, layer: 1, pos: 85
type: RSZ, layer: 1, pos: 86
type: RSZ, layer: 1, pos: 105
type: RSZ, layer: 1, pos: 123
type: RSZ, layer: 1, pos: 140
type: RSZ, layer: 1, pos: 167
type: RSZ, layer: 1, pos: 213
type: RSZ, layer: 1, pos: 219

Time for candidate selection: 0.16 seconds

### Candidate
type: RSZ, layer: 1, pos: 81

### Relational analysis RSZ of RS_RSZ1_RSZ1

#### Relational analysis RSZ result of RS_RSZ1_RSZ1
Status: Status.UNKNOWN
Output dim: 7, lower bound: -0.0011825, upper bound: 0.0011413
time: 0.62 seconds

### Relational analysis RSZ of RS_RSZ1_RSZ2

#### Relational analysis RSZ result of RS_RSZ1_RSZ2
Status: Status.UNKNOWN
Output dim: 7, lower bound: -0.0011597, upper bound: 0.0011613
time: 0.65 seconds

## BFS RS instance: RS_RSZ2

### Backsubstitution after applying RS history:
0: -0.0027939, -0.0016242, -0.0027939, -0.0016242, -0.0008116, 0.0008094
1: -0.0114005, -0.0084323, -0.0114005, -0.0084323, -0.0020596, 0.0020541
2: 0.0279571, 0.0297986, 0.0279571, 0.0297986, -0.0012778, 0.0012744
3: 0.0039210, 0.0073596, 0.0039210, 0.0073596, -0.0023796, 0.0023860
4: -0.0104893, -0.0074701, -0.0104893, -0.0074701, -0.0020950, 0.0020893
5: 0.0097651, 0.0109087, 0.0097651, 0.0109087, -0.0007935, 0.0007914
6: 0.0053643, 0.0097284, 0.0053643, 0.0097284, -0.0030200, 0.0030281
7: 0.9818129, 0.9848667, 0.9818129, 0.9848667, -0.0021132, 0.0021189
8: -0.0060636, -0.0027895, -0.0060636, -0.0027895, -0.0022657, 0.0022718
9: -0.0031570, -0.0009942, -0.0031570, -0.0009942, -0.0015007, 0.0014966

### Unstable ReLU Count (Linear/Conv2D Layers)
- layer_idx=0, type=LayerType.Linear, total=256, inp1_unstable=10, inp2_unstable=10, delta_unstable=255
- layer_idx=2, type=LayerType.Linear, total=256, inp1_unstable=10, inp2_unstable=10, delta_unstable=256
- layer_idx=4, type=LayerType.Linear, total=256, inp1_unstable=0, inp2_unstable=0, delta_unstable=256
- layer_idx=6, type=LayerType.Linear, total=256, inp1_unstable=0, inp2_unstable=0, delta_unstable=256
- layer_idx=8, type=LayerType.Linear, total=10, inp1_unstable=0, inp2_unstable=0, delta_unstable=10

Time for backsubstitution: 1.49 seconds

### RS candidates at layer 1
type: RSZ, layer: 1, pos: 81
type: RSZ, layer: 1, pos: 85
type: RSZ, layer: 1, pos: 86
type: RSZ, layer: 1, pos: 105
type: RSZ, layer: 1, pos: 123
type: RSZ, layer: 1, pos: 140
type: RSZ, layer: 1, pos: 167
type: RSZ, layer: 1, pos: 213
type: RSZ, layer: 1, pos: 219

Time for candidate selection: 0.15 seconds

### Candidate
type: RSZ, layer: 1, pos: 81

### Relational analysis RSZ of RS_RSZ2_RSZ1

#### Relational analysis RSZ result of RS_RSZ2_RSZ1
Status: Status.UNKNOWN
Output dim: 7, lower bound: -0.0011613, upper bound: 0.0011597
time: 0.61 seconds

### Relational analysis RSZ of RS_RSZ2_RSZ2

#### Relational analysis RSZ result of RS_RSZ2_RSZ2
Status: Status.UNKNOWN
Output dim: 7, lower bound: -0.0011412, upper bound: 0.0011826
time: 0.61 seconds

## Summary of splitting (split count: 1)
- Time for RS candidates: 2.86 seconds
RS_RSZ1_RSZ1, status: Status.UNKNOWN, split count: 2, time: 2.86
Output dim: 7, lower bound: -0.0011825, upper bound: 0.0011413
RS_RSZ1_RSZ2, status: Status.UNKNOWN, split count: 2, time: 2.86
Output dim: 7, lower bound: -0.0011597, upper bound: 0.0011613
RS_RSZ2_RSZ1, status: Status.UNKNOWN, split count: 2, time: 2.86
Output dim: 7, lower bound: -0.0011613, upper bound: 0.0011597
RS_RSZ2_RSZ2, status: Status.UNKNOWN, split count: 2, time: 2.86
Output dim: 7, lower bound: -0.0011412, upper bound: 0.0011826

## BFS RS instance: RS_RSZ1_RSZ1

### Backsubstitution after applying RS history:
0: -0.0027939, -0.0016242, -0.0027939, -0.0016242, -0.0008069, 0.0008141
1: -0.0114005, -0.0084323, -0.0114005, -0.0084323, -0.0020476, 0.0020658
2: 0.0279571, 0.0297986, 0.0279571, 0.0297986, -0.0012703, 0.0012816
3: 0.0039210, 0.0073596, 0.0039210, 0.0073596, -0.0023931, 0.0023720
4: -0.0104893, -0.0074701, -0.0104893, -0.0074701, -0.0020827, 0.0021013
5: 0.0097651, 0.0109087, 0.0097651, 0.0109087, -0.0007889, 0.0007959
6: 0.0053643, 0.0097284, 0.0053643, 0.0097284, -0.0030372, 0.0030104
7: 0.9818129, 0.9848667, 0.9818129, 0.9848667, -0.0021253, 0.0021065
8: -0.0060636, -0.0027895, -0.0060636, -0.0027895, -0.0022786, 0.0022585
9: -0.0031570, -0.0009942, -0.0031570, -0.0009942, -0.0014919, 0.0015052

### Unstable ReLU Count (Linear/Conv2D Layers)
- layer_idx=0, type=LayerType.Linear, total=256, inp1_unstable=10, inp2_unstable=10, delta_unstable=254
- layer_idx=2, type=LayerType.Linear, total=256, inp1_unstable=10, inp2_unstable=10, delta_unstable=256
- layer_idx=4, type=LayerType.Linear, total=256, inp1_unstable=0, inp2_unstable=0, delta_unstable=256
- layer_idx=6, type=LayerType.Linear, total=256, inp1_unstable=0, inp2_unstable=0, delta_unstable=256
- layer_idx=8, type=LayerType.Linear, total=10, inp1_unstable=0, inp2_unstable=0, delta_unstable=10

Time for backsubstitution: 1.46 seconds

### RS candidates at layer 1
type: RSZ, layer: 1, pos: 85
type: RSZ, layer: 1, pos: 86
type: RSZ, layer: 1, pos: 105
type: RSZ, layer: 1, pos: 123
type: RSZ, layer: 1, pos: 140
type: RSZ, layer: 1, pos: 167
type: RSZ, layer: 1, pos: 213
type: RSZ, layer: 1, pos: 219

Time for candidate selection: 0.15 seconds

### Candidate
type: RSZ, layer: 1, pos: 85

### Relational analysis RSZ of RS_RSZ1_RSZ1_RSZ1

#### Relational analysis RSZ result of RS_RSZ1_RSZ1_RSZ1
Status: Status.UNKNOWN
Output dim: 7, lower bound: -0.0010619, upper bound: 0.0010358
time: 0.60 seconds

### Relational analysis RSZ of RS_RSZ1_RSZ1_RSZ2

#### Relational analysis RSZ result of RS_RSZ1_RSZ1_RSZ2
Status: Status.UNKNOWN
Output dim: 7, lower bound: -0.0010619, upper bound: 0.0010358
time: 0.60 seconds

## BFS RS instance: RS_RSZ1_RSZ2

### Backsubstitution after applying RS history:
0: -0.0027939, -0.0016242, -0.0027939, -0.0016242, -0.0008119, 0.0008091
1: -0.0114005, -0.0084323, -0.0114005, -0.0084323, -0.0020602, 0.0020532
2: 0.0279571, 0.0297986, 0.0279571, 0.0297986, -0.0012782, 0.0012738
3: 0.0039210, 0.0073596, 0.0039210, 0.0073596, -0.0023785, 0.0023867
4: -0.0104893, -0.0074701, -0.0104893, -0.0074701, -0.0020956, 0.0020884
5: 0.0097651, 0.0109087, 0.0097651, 0.0109087, -0.0007938, 0.0007910
6: 0.0053643, 0.0097284, 0.0053643, 0.0097284, -0.0030186, 0.0030290
7: 0.9818129, 0.9848667, 0.9818129, 0.9848667, -0.0021123, 0.0021195
8: -0.0060636, -0.0027895, -0.0060636, -0.0027895, -0.0022647, 0.0022725
9: -0.0031570, -0.0009942, -0.0031570, -0.0009942, -0.0015011, 0.0014960

### Unstable ReLU Count (Linear/Conv2D Layers)
- layer_idx=0, type=LayerType.Linear, total=256, inp1_unstable=10, inp2_unstable=10, delta_unstable=254
- layer_idx=2, type=LayerType.Linear, total=256, inp1_unstable=10, inp2_unstable=10, delta_unstable=256
- layer_idx=4, type=LayerType.Linear, total=256, inp1_unstable=0, inp2_unstable=0, delta_unstable=256
- layer_idx=6, type=LayerType.Linear, total=256, inp1_unstable=0, inp2_unstable=0, delta_unstable=256
- layer_idx=8, type=LayerType.Linear, total=10, inp1_unstable=0, inp2_unstable=0, delta_unstable=10

Time for backsubstitution: 1.47 seconds

### RS candidates at layer 1
type: RSZ, layer: 1, pos: 85
type: RSZ, layer: 1, pos: 86
type: RSZ, layer: 1, pos: 105
type: RSZ, layer: 1, pos: 123
type: RSZ, layer: 1, pos: 140
type: RSZ, layer: 1, pos: 167
type: RSZ, layer: 1, pos: 213
type: RSZ, layer: 1, pos: 219

Time for candidate selection: 0.15 seconds

### Candidate
type: RSZ, layer: 1, pos: 85

### Relational analysis RSZ of RS_RSZ1_RSZ2_RSZ1

#### Relational analysis RSZ result of RS_RSZ1_RSZ2_RSZ1
Status: Status.UNKNOWN
Output dim: 7, lower bound: -0.0010444, upper bound: 0.0010515
time: 0.62 seconds

### Relational analysis RSZ of RS_RSZ1_RSZ2_RSZ2

#### Relational analysis RSZ result of RS_RSZ1_RSZ2_RSZ2
Status: Status.UNKNOWN
Output dim: 7, lower bound: -0.0010444, upper bound: 0.0010515
time: 0.62 seconds

## BFS RS instance: RS_RSZ2_RSZ1

### Backsubstitution after applying RS history:
0: -0.0027939, -0.0016242, -0.0027939, -0.0016242, -0.0008091, 0.0008119
1: -0.0114005, -0.0084323, -0.0114005, -0.0084323, -0.0020532, 0.0020602
2: 0.0279571, 0.0297986, 0.0279571, 0.0297986, -0.0012738, 0.0012782
3: 0.0039210, 0.0073596, 0.0039210, 0.0073596, -0.0023867, 0.0023785
4: -0.0104893, -0.0074701, -0.0104893, -0.0074701, -0.0020884, 0.0020956
5: 0.0097651, 0.0109087, 0.0097651, 0.0109087, -0.0007910, 0.0007938
6: 0.0053643, 0.0097284, 0.0053643, 0.0097284, -0.0030290, 0.0030186
7: 0.9818129, 0.9848667, 0.9818129, 0.9848667, -0.0021195, 0.0021123
8: -0.0060636, -0.0027895, -0.0060636, -0.0027895, -0.0022725, 0.0022647
9: -0.0031570, -0.0009942, -0.0031570, -0.0009942, -0.0014960, 0.0015011

### Unstable ReLU Count (Linear/Conv2D Layers)
- layer_idx=0, type=LayerType.Linear, total=256, inp1_unstable=10, inp2_unstable=10, delta_unstable=254
- layer_idx=2, type=LayerType.Linear, total=256, inp1_unstable=10, inp2_unstable=10, delta_unstable=256
- layer_idx=4, type=LayerType.Linear, total=256, inp1_unstable=0, inp2_unstable=0, delta_unstable=256
- layer_idx=6, type=LayerType.Linear, total=256, inp1_unstable=0, inp2_unstable=0, delta_unstable=256
- layer_idx=8, type=LayerType.Linear, total=10, inp1_unstable=0, inp2_unstable=0, delta_unstable=10

Time for backsubstitution: 1.47 seconds

### RS candidates at layer 1
type: RSZ, layer: 1, pos: 85
type: RSZ, layer: 1, pos: 86
type: RSZ, layer: 1, pos: 105
type: RSZ, layer: 1, pos: 123
type: RSZ, layer: 1, pos: 140
type: RSZ, layer: 1, pos: 167
type: RSZ, layer: 1, pos: 213
type: RSZ, layer: 1, pos: 219

Time for candidate selection: 0.15 seconds

### Candidate
type: RSZ, layer: 1, pos: 85

### Relational analysis RSZ of RS_RSZ2_RSZ1_RSZ1

#### Relational analysis RSZ result of RS_RSZ2_RSZ1_RSZ1
Status: Status.UNKNOWN
Output dim: 7, lower bound: -0.0010515, upper bound: 0.0010444
time: 0.63 seconds

### Relational analysis RSZ of RS_RSZ2_RSZ1_RSZ2

#### Relational analysis RSZ result of RS_RSZ2_RSZ1_RSZ2
Status: Status.UNKNOWN
Output dim: 7, lower bound: -0.0010515, upper bound: 0.0010444
time: 0.62 seconds

## BFS RS instance: RS_RSZ2_RSZ2

### Backsubstitution after applying RS history:
0: -0.0027939, -0.0016242, -0.0027939, -0.0016242, -0.0008141, 0.0008069
1: -0.0114005, -0.0084323, -0.0114005, -0.0084323, -0.0020658, 0.0020476
2: 0.0279571, 0.0297986, 0.0279571, 0.0297986, -0.0012816, 0.0012703
3: 0.0039210, 0.0073596, 0.0039210, 0.0073596, -0.0023720, 0.0023931
4: -0.0104893, -0.0074701, -0.0104893, -0.0074701, -0.0021013, 0.0020827
5: 0.0097651, 0.0109087, 0.0097651, 0.0109087, -0.0007959, 0.0007889
6: 0.0053643, 0.0097284, 0.0053643, 0.0097284, -0.0030104, 0.0030372
7: 0.9818129, 0.9848667, 0.9818129, 0.9848667, -0.0021065, 0.0021253
8: -0.0060636, -0.0027895, -0.0060636, -0.0027895, -0.0022585, 0.0022786
9: -0.0031570, -0.0009942, -0.0031570, -0.0009942, -0.0015052, 0.0014919

### Unstable ReLU Count (Linear/Conv2D Layers)
- layer_idx=0, type=LayerType.Linear, total=256, inp1_unstable=10, inp2_unstable=10, delta_unstable=254
- layer_idx=2, type=LayerType.Linear, total=256, inp1_unstable=10, inp2_unstable=10, delta_unstable=256
- layer_idx=4, type=LayerType.Linear, total=256, inp1_unstable=0, inp2_unstable=0, delta_unstable=256
- layer_idx=6, type=LayerType.Linear, total=256, inp1_unstable=0, inp2_unstable=0, delta_unstable=256
- layer_idx=8, type=LayerType.Linear, total=10, inp1_unstable=0, inp2_unstable=0, delta_unstable=10

Time for backsubstitution: 1.49 seconds

### RS candidates at layer 1
type: RSZ, layer: 1, pos: 85
type: RSZ, layer: 1, pos: 86
type: RSZ, layer: 1, pos: 105
type: RSZ, layer: 1, pos: 123
type: RSZ, layer: 1, pos: 140
type: RSZ, layer: 1, pos: 167
type: RSZ, layer: 1, pos: 213
type: RSZ, layer: 1, pos: 219

Time for candidate selection: 0.16 seconds

### Candidate
type: RSZ, layer: 1, pos: 85

### Relational analysis RSZ of RS_RSZ2_RSZ2_RSZ1

#### Relational analysis RSZ result of RS_RSZ2_RSZ2_RSZ1
Status: Status.UNKNOWN
Output dim: 7, lower bound: -0.0010358, upper bound: 0.0010619
time: 0.60 seconds

### Relational analysis RSZ of RS_RSZ2_RSZ2_RSZ2

#### Relational analysis RSZ result of RS_RSZ2_RSZ2_RSZ2
Status: Status.UNKNOWN
Output dim: 7, lower bound: -0.0010358, upper bound: 0.0010619
time: 0.59 seconds

## Summary of splitting (split count: 2)
- Time for RS candidates: 2.85 seconds
RS_RSZ1_RSZ1_RSZ1, status: Status.UNKNOWN, split count: 3, time: 2.85
Output dim: 7, lower bound: -0.0010619, upper bound: 0.0010358
RS_RSZ1_RSZ1_RSZ2, status: Status.UNKNOWN, split count: 3, time: 2.85
Output dim: 7, lower bound: -0.0010619, upper bound: 0.0010358
RS_RSZ1_RSZ2_RSZ1, status: Status.UNKNOWN, split count: 3, time: 2.85
Output dim: 7, lower bound: -0.0010444, upper bound: 0.0010515
RS_RSZ1_RSZ2_RSZ2, status: Status.UNKNOWN, split count: 3, time: 2.85
Output dim: 7, lower bound: -0.0010444, upper bound: 0.0010515
RS_RSZ2_RSZ1_RSZ1, status: Status.UNKNOWN, split count: 3, time: 2.85
Output dim: 7, lower bound: -0.0010515, upper bound: 0.0010444
RS_RSZ2_RSZ1_RSZ2, status: Status.UNKNOWN, split count: 3, time: 2.85
Output dim: 7, lower bound: -0.0010515, upper bound: 0.0010444
RS_RSZ2_RSZ2_RSZ1, status: Status.UNKNOWN, split count: 3, time: 2.85
Output dim: 7, lower bound: -0.0010358, upper bound: 0.0010619
RS_RSZ2_RSZ2_RSZ2, status: Status.UNKNOWN, split count: 3, time: 2.85
Output dim: 7, lower bound: -0.0010358, upper bound: 0.0010619

## BFS RS instance: RS_RSZ1_RSZ1_RSZ1

### Backsubstitution after applying RS history:
0: -0.0027939, -0.0016242, -0.0027939, -0.0016242, -0.0008051, 0.0008145
1: -0.0114005, -0.0084323, -0.0114005, -0.0084323, -0.0020431, 0.0020668
2: 0.0279571, 0.0297986, 0.0279571, 0.0297986, -0.0012675, 0.0012823
3: 0.0039210, 0.0073596, 0.0039210, 0.0073596, -0.0023943, 0.0023668
4: -0.0104893, -0.0074701, -0.0104893, -0.0074701, -0.0020782, 0.0021023
5: 0.0097651, 0.0109087, 0.0097651, 0.0109087, -0.0007872, 0.0007963
6: 0.0053643, 0.0097284, 0.0053643, 0.0097284, -0.0030387, 0.0030038
7: 0.9818129, 0.9848667, 0.9818129, 0.9848667, -0.0021263, 0.0021019
8: -0.0060636, -0.0027895, -0.0060636, -0.0027895, -0.0022797, 0.0022536
9: -0.0031570, -0.0009942, -0.0031570, -0.0009942, -0.0014886, 0.0015059

### Unstable ReLU Count (Linear/Conv2D Layers)
- layer_idx=0, type=LayerType.Linear, total=256, inp1_unstable=10, inp2_unstable=10, delta_unstable=253
- layer_idx=2, type=LayerType.Linear, total=256, inp1_unstable=10, inp2_unstable=10, delta_unstable=256
- layer_idx=4, type=LayerType.Linear, total=256, inp1_unstable=0, inp2_unstable=0, delta_unstable=256
- layer_idx=6, type=LayerType.Linear, total=256, inp1_unstable=0, inp2_unstable=0, delta_unstable=256
- layer_idx=8, type=LayerType.Linear, total=10, inp1_unstable=0, inp2_unstable=0, delta_unstable=10

Time for backsubstitution: 1.49 seconds

### RS candidates at layer 1
type: RSZ, layer: 1, pos: 86
type: RSZ, layer: 1, pos: 105
type: RSZ, layer: 1, pos: 123
type: RSZ, layer: 1, pos: 140
type: RSZ, layer: 1, pos: 167
type: RSZ, layer: 1, pos: 213
type: RSZ, layer: 1, pos: 219

Time for candidate selection: 0.15 seconds

### Candidate
type: RSZ, layer: 1, pos: 86

### Relational analysis RSZ of RS_RSZ1_RSZ1_RSZ1_RSZ1

#### Relational analysis RSZ result of RS_RSZ1_RSZ1_RSZ1_RSZ1
Status: Status.UNKNOWN
Output dim: 7, lower bound: -0.0010211, upper bound: 0.0009988
time: 0.62 seconds

### Relational analysis RSZ of RS_RSZ1_RSZ1_RSZ1_RSZ2

#### Relational analysis RSZ result of RS_RSZ1_RSZ1_RSZ1_RSZ2
Status: Status.UNKNOWN
Output dim: 7, lower bound: -0.0010215, upper bound: 0.0009938
time: 0.59 seconds

## BFS RS instance: RS_RSZ1_RSZ1_RSZ2

### Backsubstitution after applying RS history:
0: -0.0027939, -0.0016242, -0.0027939, -0.0016242, -0.0008069, 0.0008123
1: -0.0114005, -0.0084323, -0.0114005, -0.0084323, -0.0020476, 0.0020613
2: 0.0279571, 0.0297986, 0.0279571, 0.0297986, -0.0012703, 0.0012788
3: 0.0039210, 0.0073596, 0.0039210, 0.0073596, -0.0023879, 0.0023720
4: -0.0104893, -0.0074701, -0.0104893, -0.0074701, -0.0020827, 0.0020967
5: 0.0097651, 0.0109087, 0.0097651, 0.0109087, -0.0007889, 0.0007942
6: 0.0053643, 0.0097284, 0.0053643, 0.0097284, -0.0030306, 0.0030104
7: 0.9818129, 0.9848667, 0.9818129, 0.9848667, -0.0021206, 0.0021065
8: -0.0060636, -0.0027895, -0.0060636, -0.0027895, -0.0022737, 0.0022585
9: -0.0031570, -0.0009942, -0.0031570, -0.0009942, -0.0014919, 0.0015019

### Unstable ReLU Count (Linear/Conv2D Layers)
- layer_idx=0, type=LayerType.Linear, total=256, inp1_unstable=10, inp2_unstable=10, delta_unstable=253
- layer_idx=2, type=LayerType.Linear, total=256, inp1_unstable=10, inp2_unstable=10, delta_unstable=256
- layer_idx=4, type=LayerType.Linear, total=256, inp1_unstable=0, inp2_unstable=0, delta_unstable=256
- layer_idx=6, type=LayerType.Linear, total=256, inp1_unstable=0, inp2_unstable=0, delta_unstable=256
- layer_idx=8, type=LayerType.Linear, total=10, inp1_unstable=0, inp2_unstable=0, delta_unstable=10

Time for backsubstitution: 1.55 seconds

### RS candidates at layer 1
type: RSZ, layer: 1, pos: 86
type: RSZ, layer: 1, pos: 105
type: RSZ, layer: 1, pos: 123
type: RSZ, layer: 1, pos: 140
type: RSZ, layer: 1, pos: 167
type: RSZ, layer: 1, pos: 213
type: RSZ, layer: 1, pos: 219

Time for candidate selection: 0.16 seconds

### Candidate
type: RSZ, layer: 1, pos: 86

### Relational analysis RSZ of RS_RSZ1_RSZ1_RSZ2_RSZ1

#### Relational analysis RSZ result of RS_RSZ1_RSZ1_RSZ2_RSZ1
Status: Status.UNKNOWN
Output dim: 7, lower bound: -0.0010211, upper bound: 0.0009988
time: 0.62 seconds

### Relational analysis RSZ of RS_RSZ1_RSZ1_RSZ2_RSZ2

#### Relational analysis RSZ result of RS_RSZ1_RSZ1_RSZ2_RSZ2
Status: Status.UNKNOWN
Output dim: 7, lower bound: -0.0010215, upper bound: 0.0009938
time: 0.59 seconds

## BFS RS instance: RS_RSZ1_RSZ2_RSZ1

### Backsubstitution after applying RS history:
0: -0.0027939, -0.0016242, -0.0027939, -0.0016242, -0.0008101, 0.0008093
1: -0.0114005, -0.0084323, -0.0114005, -0.0084323, -0.0020557, 0.0020537
2: 0.0279571, 0.0297986, 0.0279571, 0.0297986, -0.0012754, 0.0012741
3: 0.0039210, 0.0073596, 0.0039210, 0.0073596, -0.0023791, 0.0023815
4: -0.0104893, -0.0074701, -0.0104893, -0.0074701, -0.0020910, 0.0020889
5: 0.0097651, 0.0109087, 0.0097651, 0.0109087, -0.0007920, 0.0007912
6: 0.0053643, 0.0097284, 0.0053643, 0.0097284, -0.0030193, 0.0030224
7: 0.9818129, 0.9848667, 0.9818129, 0.9848667, -0.0021128, 0.0021149
8: -0.0060636, -0.0027895, -0.0060636, -0.0027895, -0.0022652, 0.0022675
9: -0.0031570, -0.0009942, -0.0031570, -0.0009942, -0.0014978, 0.0014963

### Unstable ReLU Count (Linear/Conv2D Layers)
- layer_idx=0, type=LayerType.Linear, total=256, inp1_unstable=10, inp2_unstable=10, delta_unstable=253
- layer_idx=2, type=LayerType.Linear, total=256, inp1_unstable=10, inp2_unstable=10, delta_unstable=256
- layer_idx=4, type=LayerType.Linear, total=256, inp1_unstable=0, inp2_unstable=0, delta_unstable=256
- layer_idx=6, type=LayerType.Linear, total=256, inp1_unstable=0, inp2_unstable=0, delta_unstable=256
- layer_idx=8, type=LayerType.Linear, total=10, inp1_unstable=0, inp2_unstable=0, delta_unstable=10

Time for backsubstitution: 1.44 seconds

### RS candidates at layer 1
type: RSZ, layer: 1, pos: 86
type: RSZ, layer: 1, pos: 105
type: RSZ, layer: 1, pos: 123
type: RSZ, layer: 1, pos: 140
type: RSZ, layer: 1, pos: 167
type: RSZ, layer: 1, pos: 213
type: RSZ, layer: 1, pos: 219

Time for candidate selection: 0.15 seconds

### Candidate
type: RSZ, layer: 1, pos: 86

### Relational analysis RSZ of RS_RSZ1_RSZ2_RSZ1_RSZ1

#### Relational analysis RSZ result of RS_RSZ1_RSZ2_RSZ1_RSZ1
Status: Status.UNKNOWN
Output dim: 7, lower bound: -0.0010038, upper bound: 0.0010146
time: 0.62 seconds

### Relational analysis RSZ of RS_RSZ1_RSZ2_RSZ1_RSZ2

#### Relational analysis RSZ result of RS_RSZ1_RSZ2_RSZ1_RSZ2
Status: Status.UNKNOWN
Output dim: 7, lower bound: -0.0010042, upper bound: 0.0010089
time: 0.62 seconds

## BFS RS instance: RS_RSZ1_RSZ2_RSZ2

### Backsubstitution after applying RS history:
0: -0.0027939, -0.0016242, -0.0027939, -0.0016242, -0.0008119, 0.0008073
1: -0.0114005, -0.0084323, -0.0114005, -0.0084323, -0.0020602, 0.0020487
2: 0.0279571, 0.0297986, 0.0279571, 0.0297986, -0.0012782, 0.0012710
3: 0.0039210, 0.0073596, 0.0039210, 0.0073596, -0.0023733, 0.0023867
4: -0.0104893, -0.0074701, -0.0104893, -0.0074701, -0.0020956, 0.0020838
5: 0.0097651, 0.0109087, 0.0097651, 0.0109087, -0.0007938, 0.0007893
6: 0.0053643, 0.0097284, 0.0053643, 0.0097284, -0.0030120, 0.0030290
7: 0.9818129, 0.9848667, 0.9818129, 0.9848667, -0.0021077, 0.0021195
8: -0.0060636, -0.0027895, -0.0060636, -0.0027895, -0.0022597, 0.0022725
9: -0.0031570, -0.0009942, -0.0031570, -0.0009942, -0.0015011, 0.0014927

### Unstable ReLU Count (Linear/Conv2D Layers)
- layer_idx=0, type=LayerType.Linear, total=256, inp1_unstable=10, inp2_unstable=10, delta_unstable=253
- layer_idx=2, type=LayerType.Linear, total=256, inp1_unstable=10, inp2_unstable=10, delta_unstable=256
- layer_idx=4, type=LayerType.Linear, total=256, inp1_unstable=0, inp2_unstable=0, delta_unstable=256
- layer_idx=6, type=LayerType.Linear, total=256, inp1_unstable=0, inp2_unstable=0, delta_unstable=256
- layer_idx=8, type=LayerType.Linear, total=10, inp1_unstable=0, inp2_unstable=0, delta_unstable=10

Time for backsubstitution: 1.61 seconds

### RS candidates at layer 1
type: RSZ, layer: 1, pos: 86
type: RSZ, layer: 1, pos: 105
type: RSZ, layer: 1, pos: 123
type: RSZ, layer: 1, pos: 140
type: RSZ, layer: 1, pos: 167
type: RSZ, layer: 1, pos: 213
type: RSZ, layer: 1, pos: 219

Time for candidate selection: 0.21 seconds

### Candidate
type: RSZ, layer: 1, pos: 86

### Relational analysis RSZ of RS_RSZ1_RSZ2_RSZ2_RSZ1

#### Relational analysis RSZ result of RS_RSZ1_RSZ2_RSZ2_RSZ1
Status: Status.UNKNOWN
Output dim: 7, lower bound: -0.0010038, upper bound: 0.0010146
time: 0.64 seconds

### Relational analysis RSZ of RS_RSZ1_RSZ2_RSZ2_RSZ2

#### Relational analysis RSZ result of RS_RSZ1_RSZ2_RSZ2_RSZ2
Status: Status.UNKNOWN
Output dim: 7, lower bound: -0.0010042, upper bound: 0.0010089
time: 0.64 seconds

## BFS RS instance: RS_RSZ2_RSZ1_RSZ1

### Backsubstitution after applying RS history:
0: -0.0027939, -0.0016242, -0.0027939, -0.0016242, -0.0008073, 0.0008122
1: -0.0114005, -0.0084323, -0.0114005, -0.0084323, -0.0020487, 0.0020611
2: 0.0279571, 0.0297986, 0.0279571, 0.0297986, -0.0012710, 0.0012787
3: 0.0039210, 0.0073596, 0.0039210, 0.0073596, -0.0023877, 0.0023733
4: -0.0104893, -0.0074701, -0.0104893, -0.0074701, -0.0020838, 0.0020965
5: 0.0097651, 0.0109087, 0.0097651, 0.0109087, -0.0007893, 0.0007941
6: 0.0053643, 0.0097284, 0.0053643, 0.0097284, -0.0030303, 0.0030120
7: 0.9818129, 0.9848667, 0.9818129, 0.9848667, -0.0021205, 0.0021077
8: -0.0060636, -0.0027895, -0.0060636, -0.0027895, -0.0022735, 0.0022597
9: -0.0031570, -0.0009942, -0.0031570, -0.0009942, -0.0014927, 0.0015018

### Unstable ReLU Count (Linear/Conv2D Layers)
- layer_idx=0, type=LayerType.Linear, total=256, inp1_unstable=10, inp2_unstable=10, delta_unstable=253
- layer_idx=2, type=LayerType.Linear, total=256, inp1_unstable=10, inp2_unstable=10, delta_unstable=256
- layer_idx=4, type=LayerType.Linear, total=256, inp1_unstable=0, inp2_unstable=0, delta_unstable=256
- layer_idx=6, type=LayerType.Linear, total=256, inp1_unstable=0, inp2_unstable=0, delta_unstable=256
- layer_idx=8, type=LayerType.Linear, total=10, inp1_unstable=0, inp2_unstable=0, delta_unstable=10

Time for backsubstitution: 1.47 seconds

### RS candidates at layer 1
type: RSZ, layer: 1, pos: 86
type: RSZ, layer: 1, pos: 105
type: RSZ, layer: 1, pos: 123
type: RSZ, layer: 1, pos: 140
type: RSZ, layer: 1, pos: 167
type: RSZ, layer: 1, pos: 213
type: RSZ, layer: 1, pos: 219

Time for candidate selection: 0.17 seconds

### Candidate
type: RSZ, layer: 1, pos: 86

### Relational analysis RSZ of RS_RSZ2_RSZ1_RSZ1_RSZ1

#### Relational analysis RSZ result of RS_RSZ2_RSZ1_RSZ1_RSZ1
Status: Status.UNKNOWN
Output dim: 7, lower bound: -0.0010089, upper bound: 0.0010042
time: 0.62 seconds

### Relational analysis RSZ of RS_RSZ2_RSZ1_RSZ1_RSZ2

#### Relational analysis RSZ result of RS_RSZ2_RSZ1_RSZ1_RSZ2
Status: Status.UNKNOWN
Output dim: 7, lower bound: -0.0010146, upper bound: 0.0010038
time: 0.62 seconds

## BFS RS instance: RS_RSZ2_RSZ1_RSZ2

### Backsubstitution after applying RS history:
0: -0.0027939, -0.0016242, -0.0027939, -0.0016242, -0.0008091, 0.0008101
1: -0.0114005, -0.0084323, -0.0114005, -0.0084323, -0.0020532, 0.0020557
2: 0.0279571, 0.0297986, 0.0279571, 0.0297986, -0.0012738, 0.0012754
3: 0.0039210, 0.0073596, 0.0039210, 0.0073596, -0.0023815, 0.0023785
4: -0.0104893, -0.0074701, -0.0104893, -0.0074701, -0.0020884, 0.0020910
5: 0.0097651, 0.0109087, 0.0097651, 0.0109087, -0.0007910, 0.0007920
6: 0.0053643, 0.0097284, 0.0053643, 0.0097284, -0.0030224, 0.0030186
7: 0.9818129, 0.9848667, 0.9818129, 0.9848667, -0.0021149, 0.0021123
8: -0.0060636, -0.0027895, -0.0060636, -0.0027895, -0.0022675, 0.0022647
9: -0.0031570, -0.0009942, -0.0031570, -0.0009942, -0.0014960, 0.0014978

### Unstable ReLU Count (Linear/Conv2D Layers)
- layer_idx=0, type=LayerType.Linear, total=256, inp1_unstable=10, inp2_unstable=10, delta_unstable=253
- layer_idx=2, type=LayerType.Linear, total=256, inp1_unstable=10, inp2_unstable=10, delta_unstable=256
- layer_idx=4, type=LayerType.Linear, total=256, inp1_unstable=0, inp2_unstable=0, delta_unstable=256
- layer_idx=6, type=LayerType.Linear, total=256, inp1_unstable=0, inp2_unstable=0, delta_unstable=256
- layer_idx=8, type=LayerType.Linear, total=10, inp1_unstable=0, inp2_unstable=0, delta_unstable=10

Time for backsubstitution: 1.53 seconds

### RS candidates at layer 1
type: RSZ, layer: 1, pos: 86
type: RSZ, layer: 1, pos: 105
type: RSZ, layer: 1, pos: 123
type: RSZ, layer: 1, pos: 140
type: RSZ, layer: 1, pos: 167
type: RSZ, layer: 1, pos: 213
type: RSZ, layer: 1, pos: 219

Time for candidate selection: 0.16 seconds

### Candidate
type: RSZ, layer: 1, pos: 86

### Relational analysis RSZ of RS_RSZ2_RSZ1_RSZ2_RSZ1

#### Relational analysis RSZ result of RS_RSZ2_RSZ1_RSZ2_RSZ1
Status: Status.UNKNOWN
Output dim: 7, lower bound: -0.0010089, upper bound: 0.0010042
time: 0.62 seconds

### Relational analysis RSZ of RS_RSZ2_RSZ1_RSZ2_RSZ2

#### Relational analysis RSZ result of RS_RSZ2_RSZ1_RSZ2_RSZ2
Status: Status.UNKNOWN
Output dim: 7, lower bound: -0.0010146, upper bound: 0.0010038
time: 0.62 seconds

## BFS RS instance: RS_RSZ2_RSZ2_RSZ1

### Backsubstitution after applying RS history:
0: -0.0027939, -0.0016242, -0.0027939, -0.0016242, -0.0008123, 0.0008071
1: -0.0114005, -0.0084323, -0.0114005, -0.0084323, -0.0020613, 0.0020480
2: 0.0279571, 0.0297986, 0.0279571, 0.0297986, -0.0012788, 0.0012706
3: 0.0039210, 0.0073596, 0.0039210, 0.0073596, -0.0023725, 0.0023879
4: -0.0104893, -0.0074701, -0.0104893, -0.0074701, -0.0020967, 0.0020832
5: 0.0097651, 0.0109087, 0.0097651, 0.0109087, -0.0007942, 0.0007891
6: 0.0053643, 0.0097284, 0.0053643, 0.0097284, -0.0030111, 0.0030306
7: 0.9818129, 0.9848667, 0.9818129, 0.9848667, -0.0021070, 0.0021206
8: -0.0060636, -0.0027895, -0.0060636, -0.0027895, -0.0022590, 0.0022737
9: -0.0031570, -0.0009942, -0.0031570, -0.0009942, -0.0015019, 0.0014922

### Unstable ReLU Count (Linear/Conv2D Layers)
- layer_idx=0, type=LayerType.Linear, total=256, inp1_unstable=10, inp2_unstable=10, delta_unstable=253
- layer_idx=2, type=LayerType.Linear, total=256, inp1_unstable=10, inp2_unstable=10, delta_unstable=256
- layer_idx=4, type=LayerType.Linear, total=256, inp1_unstable=0, inp2_unstable=0, delta_unstable=256
- layer_idx=6, type=LayerType.Linear, total=256, inp1_unstable=0, inp2_unstable=0, delta_unstable=256
- layer_idx=8, type=LayerType.Linear, total=10, inp1_unstable=0, inp2_unstable=0, delta_unstable=10

Time for backsubstitution: 1.47 seconds

### RS candidates at layer 1
type: RSZ, layer: 1, pos: 86
type: RSZ, layer: 1, pos: 105
type: RSZ, layer: 1, pos: 123
type: RSZ, layer: 1, pos: 140
type: RSZ, layer: 1, pos: 167
type: RSZ, layer: 1, pos: 213
type: RSZ, layer: 1, pos: 219

Time for candidate selection: 0.16 seconds

### Candidate
type: RSZ, layer: 1, pos: 86

### Relational analysis RSZ of RS_RSZ2_RSZ2_RSZ1_RSZ1

#### Relational analysis RSZ result of RS_RSZ2_RSZ2_RSZ1_RSZ1
Status: Status.UNKNOWN
Output dim: 7, lower bound: -0.0009938, upper bound: 0.0010215
time: 0.63 seconds

### Relational analysis RSZ of RS_RSZ2_RSZ2_RSZ1_RSZ2

#### Relational analysis RSZ result of RS_RSZ2_RSZ2_RSZ1_RSZ2
Status: Status.UNKNOWN
Output dim: 7, lower bound: -0.0009988, upper bound: 0.0010211
time: 0.62 seconds

## BFS RS instance: RS_RSZ2_RSZ2_RSZ2

### Backsubstitution after applying RS history:
0: -0.0027939, -0.0016242, -0.0027939, -0.0016242, -0.0008141, 0.0008051
1: -0.0114005, -0.0084323, -0.0114005, -0.0084323, -0.0020658, 0.0020431
2: 0.0279571, 0.0297986, 0.0279571, 0.0297986, -0.0012816, 0.0012675
3: 0.0039210, 0.0073596, 0.0039210, 0.0073596, -0.0023668, 0.0023931
4: -0.0104893, -0.0074701, -0.0104893, -0.0074701, -0.0021013, 0.0020782
5: 0.0097651, 0.0109087, 0.0097651, 0.0109087, -0.0007959, 0.0007872
6: 0.0053643, 0.0097284, 0.0053643, 0.0097284, -0.0030038, 0.0030372
7: 0.9818129, 0.9848667, 0.9818129, 0.9848667, -0.0021019, 0.0021253
8: -0.0060636, -0.0027895, -0.0060636, -0.0027895, -0.0022536, 0.0022786
9: -0.0031570, -0.0009942, -0.0031570, -0.0009942, -0.0015052, 0.0014886

### Unstable ReLU Count (Linear/Conv2D Layers)
- layer_idx=0, type=LayerType.Linear, total=256, inp1_unstable=10, inp2_unstable=10, delta_unstable=253
- layer_idx=2, type=LayerType.Linear, total=256, inp1_unstable=10, inp2_unstable=10, delta_unstable=256
- layer_idx=4, type=LayerType.Linear, total=256, inp1_unstable=0, inp2_unstable=0, delta_unstable=256
- layer_idx=6, type=LayerType.Linear, total=256, inp1_unstable=0, inp2_unstable=0, delta_unstable=256
- layer_idx=8, type=LayerType.Linear, total=10, inp1_unstable=0, inp2_unstable=0, delta_unstable=10

Time for backsubstitution: 1.48 seconds

### RS candidates at layer 1
type: RSZ, layer: 1, pos: 86
type: RSZ, layer: 1, pos: 105
type: RSZ, layer: 1, pos: 123
type: RSZ, layer: 1, pos: 140
type: RSZ, layer: 1, pos: 167
type: RSZ, layer: 1, pos: 213
type: RSZ, layer: 1, pos: 219

Time for candidate selection: 0.14 seconds

### Candidate
type: RSZ, layer: 1, pos: 86

### Relational analysis RSZ of RS_RSZ2_RSZ2_RSZ2_RSZ1

#### Relational analysis RSZ result of RS_RSZ2_RSZ2_RSZ2_RSZ1
Status: Status.UNKNOWN
Output dim: 7, lower bound: -0.0009938, upper bound: 0.0010215
time: 0.63 seconds

### Relational analysis RSZ of RS_RSZ2_RSZ2_RSZ2_RSZ2

#### Relational analysis RSZ result of RS_RSZ2_RSZ2_RSZ2_RSZ2
Status: Status.UNKNOWN
Output dim: 7, lower bound: -0.0009988, upper bound: 0.0010211
time: 0.62 seconds

## Summary of splitting (split count: 3)
- Time for RS candidates: 2.88 seconds
RS_RSZ1_RSZ1_RSZ1_RSZ1, status: Status.UNKNOWN, split count: 4, time: 2.88
Output dim: 7, lower bound: -0.0010211, upper bound: 0.0009988
RS_RSZ1_RSZ1_RSZ1_RSZ2, status: Status.UNKNOWN, split count: 4, time: 2.88
Output dim: 7, lower bound: -0.0010215, upper bound: 0.0009938
RS_RSZ1_RSZ1_RSZ2_RSZ1, status: Status.UNKNOWN, split count: 4, time: 2.88
Output dim: 7, lower bound: -0.0010211, upper bound: 0.0009988
RS_RSZ1_RSZ1_RSZ2_RSZ2, status: Status.UNKNOWN, split count: 4, time: 2.88
Output dim: 7, lower bound: -0.0010215, upper bound: 0.0009938
RS_RSZ1_RSZ2_RSZ1_RSZ1, status: Status.UNKNOWN, split count: 4, time: 2.88
Output dim: 7, lower bound: -0.0010038, upper bound: 0.0010146
RS_RSZ1_RSZ2_RSZ1_RSZ2, status: Status.UNKNOWN, split count: 4, time: 2.88
Output dim: 7, lower bound: -0.0010042, upper bound: 0.0010089
RS_RSZ1_RSZ2_RSZ2_RSZ1, status: Status.UNKNOWN, split count: 4, time: 2.88
Output dim: 7, lower bound: -0.0010038, upper bound: 0.0010146
RS_RSZ1_RSZ2_RSZ2_RSZ2, status: Status.UNKNOWN, split count: 4, time: 2.88
Output dim: 7, lower bound: -0.0010042, upper bound: 0.0010089
RS_RSZ2_RSZ1_RSZ1_RSZ1, status: Status.UNKNOWN, split count: 4, time: 2.88
Output dim: 7, lower bound: -0.0010089, upper bound: 0.0010042
RS_RSZ2_RSZ1_RSZ1_RSZ2, status: Status.UNKNOWN, split count: 4, time: 2.88
Output dim: 7, lower bound: -0.0010146, upper bound: 0.0010038
RS_RSZ2_RSZ1_RSZ2_RSZ1, status: Status.UNKNOWN, split count: 4, time: 2.88
Output dim: 7, lower bound: -0.0010089, upper bound: 0.0010042
RS_RSZ2_RSZ1_RSZ2_RSZ2, status: Status.UNKNOWN, split count: 4, time: 2.88
Output dim: 7, lower bound: -0.0010146, upper bound: 0.0010038
RS_RSZ2_RSZ2_RSZ1_RSZ1, status: Status.UNKNOWN, split count: 4, time: 2.88
Output dim: 7, lower bound: -0.0009938, upper bound: 0.0010215
RS_RSZ2_RSZ2_RSZ1_RSZ2, status: Status.UNKNOWN, split count: 4, time: 2.88
Output dim: 7, lower bound: -0.0009988, upper bound: 0.0010211
RS_RSZ2_RSZ2_RSZ2_RSZ1, status: Status.UNKNOWN, split count: 4, time: 2.88
Output dim: 7, lower bound: -0.0009938, upper bound: 0.0010215
RS_RSZ2_RSZ2_RSZ2_RSZ2, status: Status.UNKNOWN, split count: 4, time: 2.88
Output dim: 7, lower bound: -0.0009988, upper bound: 0.0010211

## BFS RS instance: RS_RSZ1_RSZ1_RSZ1_RSZ1

### Backsubstitution after applying RS history:
0: -0.0027939, -0.0016242, -0.0027939, -0.0016242, -0.0007386, 0.0007555
1: -0.0114005, -0.0084323, -0.0114005, -0.0084323, -0.0018744, 0.0019172
2: 0.0279571, 0.0297986, 0.0279571, 0.0297986, -0.0011629, 0.0011895
3: 0.0039210, 0.0073596, 0.0039210, 0.0073596, -0.0022210, 0.0021714
4: -0.0104893, -0.0074701, -0.0104893, -0.0074701, -0.0019066, 0.0019502
5: 0.0097651, 0.0109087, 0.0097651, 0.0109087, -0.0007222, 0.0007387
6: 0.0053643, 0.0097284, 0.0053643, 0.0097284, -0.0028188, 0.0027558
7: 0.9818129, 0.9848667, 0.9818129, 0.9848667, -0.0019724, 0.0019284
8: -0.0060636, -0.0027895, -0.0060636, -0.0027895, -0.0021148, 0.0020675
9: -0.0031570, -0.0009942, -0.0031570, -0.0009942, -0.0013657, 0.0013969

### Unstable ReLU Count (Linear/Conv2D Layers)
- layer_idx=0, type=LayerType.Linear, total=256, inp1_unstable=10, inp2_unstable=10, delta_unstable=252
- layer_idx=2, type=LayerType.Linear, total=256, inp1_unstable=10, inp2_unstable=10, delta_unstable=256
- layer_idx=4, type=LayerType.Linear, total=256, inp1_unstable=0, inp2_unstable=0, delta_unstable=256
- layer_idx=6, type=LayerType.Linear, total=256, inp1_unstable=0, inp2_unstable=0, delta_unstable=256
- layer_idx=8, type=LayerType.Linear, total=10, inp1_unstable=0, inp2_unstable=0, delta_unstable=10

Time for backsubstitution: 1.46 seconds

### RS candidates at layer 1
type: RSZ, layer: 1, pos: 105
type: RSZ, layer: 1, pos: 123
type: RSZ, layer: 1, pos: 140
type: RSZ, layer: 1, pos: 167
type: RSZ, layer: 1, pos: 213
type: RSZ, layer: 1, pos: 219

Time for candidate selection: 0.15 seconds

### Candidate
type: RSZ, layer: 1, pos: 105

### Relational analysis RSZ of RS_RSZ1_RSZ1_RSZ1_RSZ1_RSZ1

#### Relational analysis RSZ result of RS_RSZ1_RSZ1_RSZ1_RSZ1_RSZ1
Status: Status.UNKNOWN
Output dim: 7, lower bound: -0.0009747, upper bound: 0.0009613
time: 0.62 seconds

### Relational analysis RSZ of RS_RSZ1_RSZ1_RSZ1_RSZ1_RSZ2

#### Relational analysis RSZ result of RS_RSZ1_RSZ1_RSZ1_RSZ1_RSZ2
Status: Status.UNKNOWN
Output dim: 7, lower bound: -0.0009855, upper bound: 0.0009545
time: 0.63 seconds

## BFS RS instance: RS_RSZ1_RSZ1_RSZ1_RSZ2

### Backsubstitution after applying RS history:
0: -0.0027939, -0.0016242, -0.0027939, -0.0016242, -0.0007462, 0.0007479
1: -0.0114005, -0.0084323, -0.0114005, -0.0084323, -0.0018935, 0.0018978
2: 0.0279571, 0.0297986, 0.0279571, 0.0297986, -0.0011748, 0.0011774
3: 0.0039210, 0.0073596, 0.0039210, 0.0073596, -0.0021986, 0.0021936
4: -0.0104893, -0.0074701, -0.0104893, -0.0074701, -0.0019260, 0.0019304
5: 0.0097651, 0.0109087, 0.0097651, 0.0109087, -0.0007295, 0.0007312
6: 0.0053643, 0.0097284, 0.0053643, 0.0097284, -0.0027903, 0.0027839
7: 0.9818129, 0.9848667, 0.9818129, 0.9848667, -0.0019525, 0.0019480
8: -0.0060636, -0.0027895, -0.0060636, -0.0027895, -0.0020934, 0.0020886
9: -0.0031570, -0.0009942, -0.0031570, -0.0009942, -0.0013796, 0.0013828

### Unstable ReLU Count (Linear/Conv2D Layers)
- layer_idx=0, type=LayerType.Linear, total=256, inp1_unstable=10, inp2_unstable=10, delta_unstable=252
- layer_idx=2, type=LayerType.Linear, total=256, inp1_unstable=10, inp2_unstable=10, delta_unstable=256
- layer_idx=4, type=LayerType.Linear, total=256, inp1_unstable=0, inp2_unstable=0, delta_unstable=256
- layer_idx=6, type=LayerType.Linear, total=256, inp1_unstable=0, inp2_unstable=0, delta_unstable=256
- layer_idx=8, type=LayerType.Linear, total=10, inp1_unstable=0, inp2_unstable=0, delta_unstable=10

Time for backsubstitution: 1.48 seconds

### RS candidates at layer 1
type: RSZ, layer: 1, pos: 105
type: RSZ, layer: 1, pos: 123
type: RSZ, layer: 1, pos: 140
type: RSZ, layer: 1, pos: 167
type: RSZ, layer: 1, pos: 213
type: RSZ, layer: 1, pos: 219

Time for candidate selection: 0.15 seconds

### Candidate
type: RSZ, layer: 1, pos: 105

### Relational analysis RSZ of RS_RSZ1_RSZ1_RSZ1_RSZ2_RSZ1

#### Relational analysis RSZ result of RS_RSZ1_RSZ1_RSZ1_RSZ2_RSZ1
Status: Status.UNKNOWN
Output dim: 7, lower bound: -0.0009748, upper bound: 0.0009576
time: 0.64 seconds

### Relational analysis RSZ of RS_RSZ1_RSZ1_RSZ1_RSZ2_RSZ2

#### Relational analysis RSZ result of RS_RSZ1_RSZ1_RSZ1_RSZ2_RSZ2
Status: Status.UNKNOWN
Output dim: 7, lower bound: -0.0009862, upper bound: 0.0009495
time: 0.63 seconds

## BFS RS instance: RS_RSZ1_RSZ1_RSZ2_RSZ1

### Backsubstitution after applying RS history:
0: -0.0027939, -0.0016242, -0.0027939, -0.0016242, -0.0007406, 0.0007533
1: -0.0114005, -0.0084323, -0.0114005, -0.0084323, -0.0018794, 0.0019117
2: 0.0279571, 0.0297986, 0.0279571, 0.0297986, -0.0011660, 0.0011860
3: 0.0039210, 0.0073596, 0.0039210, 0.0073596, -0.0022146, 0.0021772
4: -0.0104893, -0.0074701, -0.0104893, -0.0074701, -0.0019116, 0.0019445
5: 0.0097651, 0.0109087, 0.0097651, 0.0109087, -0.0007241, 0.0007365
6: 0.0053643, 0.0097284, 0.0053643, 0.0097284, -0.0028107, 0.0027631
7: 0.9818129, 0.9848667, 0.9818129, 0.9848667, -0.0019668, 0.0019335
8: -0.0060636, -0.0027895, -0.0060636, -0.0027895, -0.0021087, 0.0020730
9: -0.0031570, -0.0009942, -0.0031570, -0.0009942, -0.0013693, 0.0013929

### Unstable ReLU Count (Linear/Conv2D Layers)
- layer_idx=0, type=LayerType.Linear, total=256, inp1_unstable=10, inp2_unstable=10, delta_unstable=252
- layer_idx=2, type=LayerType.Linear, total=256, inp1_unstable=10, inp2_unstable=10, delta_unstable=256
- layer_idx=4, type=LayerType.Linear, total=256, inp1_unstable=0, inp2_unstable=0, delta_unstable=256
- layer_idx=6, type=LayerType.Linear, total=256, inp1_unstable=0, inp2_unstable=0, delta_unstable=256
- layer_idx=8, type=LayerType.Linear, total=10, inp1_unstable=0, inp2_unstable=0, delta_unstable=10

Time for backsubstitution: 1.54 seconds

### RS candidates at layer 1
type: RSZ, layer: 1, pos: 105
type: RSZ, layer: 1, pos: 123
type: RSZ, layer: 1, pos: 140
type: RSZ, layer: 1, pos: 167
type: RSZ, layer: 1, pos: 213
type: RSZ, layer: 1, pos: 219

Time for candidate selection: 0.17 seconds

### Candidate
type: RSZ, layer: 1, pos: 105

### Relational analysis RSZ of RS_RSZ1_RSZ1_RSZ2_RSZ1_RSZ1

#### Relational analysis RSZ result of RS_RSZ1_RSZ1_RSZ2_RSZ1_RSZ1
Status: Status.UNKNOWN
Output dim: 7, lower bound: -0.0009747, upper bound: 0.0009613
time: 0.63 seconds

### Relational analysis RSZ of RS_RSZ1_RSZ1_RSZ2_RSZ1_RSZ2

#### Relational analysis RSZ result of RS_RSZ1_RSZ1_RSZ2_RSZ1_RSZ2
Status: Status.UNKNOWN
Output dim: 7, lower bound: -0.0009855, upper bound: 0.0009545
time: 0.62 seconds

## BFS RS instance: RS_RSZ1_RSZ1_RSZ2_RSZ2

### Backsubstitution after applying RS history:
0: -0.0027939, -0.0016242, -0.0027939, -0.0016242, -0.0007481, 0.0007464
1: -0.0114005, -0.0084323, -0.0114005, -0.0084323, -0.0018985, 0.0018942
2: 0.0279571, 0.0297986, 0.0279571, 0.0297986, -0.0011778, 0.0011751
3: 0.0039210, 0.0073596, 0.0039210, 0.0073596, -0.0021943, 0.0021993
4: -0.0104893, -0.0074701, -0.0104893, -0.0074701, -0.0019311, 0.0019267
5: 0.0097651, 0.0109087, 0.0097651, 0.0109087, -0.0007314, 0.0007298
6: 0.0053643, 0.0097284, 0.0053643, 0.0097284, -0.0027848, 0.0027912
7: 0.9818129, 0.9848667, 0.9818129, 0.9848667, -0.0019487, 0.0019532
8: -0.0060636, -0.0027895, -0.0060636, -0.0027895, -0.0020893, 0.0020941
9: -0.0031570, -0.0009942, -0.0031570, -0.0009942, -0.0013833, 0.0013801

### Unstable ReLU Count (Linear/Conv2D Layers)
- layer_idx=0, type=LayerType.Linear, total=256, inp1_unstable=10, inp2_unstable=10, delta_unstable=252
- layer_idx=2, type=LayerType.Linear, total=256, inp1_unstable=10, inp2_unstable=10, delta_unstable=256
- layer_idx=4, type=LayerType.Linear, total=256, inp1_unstable=0, inp2_unstable=0, delta_unstable=256
- layer_idx=6, type=LayerType.Linear, total=256, inp1_unstable=0, inp2_unstable=0, delta_unstable=256
- layer_idx=8, type=LayerType.Linear, total=10, inp1_unstable=0, inp2_unstable=0, delta_unstable=10

Time for backsubstitution: 1.45 seconds

### RS candidates at layer 1
type: RSZ, layer: 1, pos: 105
type: RSZ, layer: 1, pos: 123
type: RSZ, layer: 1, pos: 140
type: RSZ, layer: 1, pos: 167
type: RSZ, layer: 1, pos: 213
type: RSZ, layer: 1, pos: 219

Time for candidate selection: 0.17 seconds

### Candidate
type: RSZ, layer: 1, pos: 105

### Relational analysis RSZ of RS_RSZ1_RSZ1_RSZ2_RSZ2_RSZ1

#### Relational analysis RSZ result of RS_RSZ1_RSZ1_RSZ2_RSZ2_RSZ1
Status: Status.UNKNOWN
Output dim: 7, lower bound: -0.0009748, upper bound: 0.0009576
time: 0.64 seconds

### Relational analysis RSZ of RS_RSZ1_RSZ1_RSZ2_RSZ2_RSZ2

#### Relational analysis RSZ result of RS_RSZ1_RSZ1_RSZ2_RSZ2_RSZ2
Status: Status.UNKNOWN
Output dim: 7, lower bound: -0.0009862, upper bound: 0.0009495
time: 0.68 seconds

## BFS RS instance: RS_RSZ1_RSZ2_RSZ1_RSZ1

### Backsubstitution after applying RS history:
0: -0.0027939, -0.0016242, -0.0027939, -0.0016242, -0.0007452, 0.0007503
1: -0.0114005, -0.0084323, -0.0114005, -0.0084323, -0.0018911, 0.0019041
2: 0.0279571, 0.0297986, 0.0279571, 0.0297986, -0.0011732, 0.0011813
3: 0.0039210, 0.0073596, 0.0039210, 0.0073596, -0.0022058, 0.0021907
4: -0.0104893, -0.0074701, -0.0104893, -0.0074701, -0.0019235, 0.0019368
5: 0.0097651, 0.0109087, 0.0097651, 0.0109087, -0.0007286, 0.0007336
6: 0.0053643, 0.0097284, 0.0053643, 0.0097284, -0.0027994, 0.0027803
7: 0.9818129, 0.9848667, 0.9818129, 0.9848667, -0.0019589, 0.0019455
8: -0.0060636, -0.0027895, -0.0060636, -0.0027895, -0.0021003, 0.0020859
9: -0.0031570, -0.0009942, -0.0031570, -0.0009942, -0.0013778, 0.0013873

### Unstable ReLU Count (Linear/Conv2D Layers)
- layer_idx=0, type=LayerType.Linear, total=256, inp1_unstable=10, inp2_unstable=10, delta_unstable=252
- layer_idx=2, type=LayerType.Linear, total=256, inp1_unstable=10, inp2_unstable=10, delta_unstable=256
- layer_idx=4, type=LayerType.Linear, total=256, inp1_unstable=0, inp2_unstable=0, delta_unstable=256
- layer_idx=6, type=LayerType.Linear, total=256, inp1_unstable=0, inp2_unstable=0, delta_unstable=256
- layer_idx=8, type=LayerType.Linear, total=10, inp1_unstable=0, inp2_unstable=0, delta_unstable=10

Time for backsubstitution: 1.53 seconds

### RS candidates at layer 1
type: RSZ, layer: 1, pos: 105
type: RSZ, layer: 1, pos: 123
type: RSZ, layer: 1, pos: 140
type: RSZ, layer: 1, pos: 167
type: RSZ, layer: 1, pos: 213
type: RSZ, layer: 1, pos: 219

Time for candidate selection: 0.15 seconds

### Candidate
type: RSZ, layer: 1, pos: 105

### Relational analysis RSZ of RS_RSZ1_RSZ2_RSZ1_RSZ1_RSZ1

#### Relational analysis RSZ result of RS_RSZ1_RSZ2_RSZ1_RSZ1_RSZ1
Status: Status.UNKNOWN
Output dim: 7, lower bound: -0.0009596, upper bound: 0.0009800
time: 0.62 seconds

### Relational analysis RSZ of RS_RSZ1_RSZ2_RSZ1_RSZ1_RSZ2

#### Relational analysis RSZ result of RS_RSZ1_RSZ2_RSZ1_RSZ1_RSZ2
Status: Status.UNKNOWN
Output dim: 7, lower bound: -0.0009670, upper bound: 0.0009667
time: 0.64 seconds

## BFS RS instance: RS_RSZ1_RSZ2_RSZ1_RSZ2

### Backsubstitution after applying RS history:
0: -0.0027939, -0.0016242, -0.0027939, -0.0016242, -0.0007512, 0.0007417
1: -0.0114005, -0.0084323, -0.0114005, -0.0084323, -0.0019062, 0.0018821
2: 0.0279571, 0.0297986, 0.0279571, 0.0297986, -0.0011826, 0.0011676
3: 0.0039210, 0.0073596, 0.0039210, 0.0073596, -0.0021803, 0.0022082
4: -0.0104893, -0.0074701, -0.0104893, -0.0074701, -0.0019389, 0.0019144
5: 0.0097651, 0.0109087, 0.0097651, 0.0109087, -0.0007344, 0.0007251
6: 0.0053643, 0.0097284, 0.0053643, 0.0097284, -0.0027671, 0.0028025
7: 0.9818129, 0.9848667, 0.9818129, 0.9848667, -0.0019363, 0.0019610
8: -0.0060636, -0.0027895, -0.0060636, -0.0027895, -0.0020760, 0.0021025
9: -0.0031570, -0.0009942, -0.0031570, -0.0009942, -0.0013888, 0.0013713

### Unstable ReLU Count (Linear/Conv2D Layers)
- layer_idx=0, type=LayerType.Linear, total=256, inp1_unstable=10, inp2_unstable=10, delta_unstable=252
- layer_idx=2, type=LayerType.Linear, total=256, inp1_unstable=10, inp2_unstable=10, delta_unstable=256
- layer_idx=4, type=LayerType.Linear, total=256, inp1_unstable=0, inp2_unstable=0, delta_unstable=256
- layer_idx=6, type=LayerType.Linear, total=256, inp1_unstable=0, inp2_unstable=0, delta_unstable=256
- layer_idx=8, type=LayerType.Linear, total=10, inp1_unstable=0, inp2_unstable=0, delta_unstable=10

Time for backsubstitution: 1.51 seconds

### RS candidates at layer 1
type: RSZ, layer: 1, pos: 105
type: RSZ, layer: 1, pos: 123
type: RSZ, layer: 1, pos: 140
type: RSZ, layer: 1, pos: 167
type: RSZ, layer: 1, pos: 213
type: RSZ, layer: 1, pos: 219

Time for candidate selection: 0.16 seconds

### Candidate
type: RSZ, layer: 1, pos: 105

### Relational analysis RSZ of RS_RSZ1_RSZ2_RSZ1_RSZ2_RSZ1

#### Relational analysis RSZ result of RS_RSZ1_RSZ2_RSZ1_RSZ2_RSZ1
Status: Status.UNKNOWN
Output dim: 7, lower bound: -0.0009599, upper bound: 0.0009757
time: 0.62 seconds

### Relational analysis RSZ of RS_RSZ1_RSZ2_RSZ1_RSZ2_RSZ2

#### Relational analysis RSZ result of RS_RSZ1_RSZ2_RSZ1_RSZ2_RSZ2
Status: Status.UNKNOWN
Output dim: 7, lower bound: -0.0009671, upper bound: 0.0009578
time: 0.63 seconds

## BFS RS instance: RS_RSZ1_RSZ2_RSZ2_RSZ1

### Backsubstitution after applying RS history:
0: -0.0027939, -0.0016242, -0.0027939, -0.0016242, -0.0007472, 0.0007484
1: -0.0114005, -0.0084323, -0.0114005, -0.0084323, -0.0018960, 0.0018991
2: 0.0279571, 0.0297986, 0.0279571, 0.0297986, -0.0011763, 0.0011782
3: 0.0039210, 0.0073596, 0.0039210, 0.0073596, -0.0022000, 0.0021964
4: -0.0104893, -0.0074701, -0.0104893, -0.0074701, -0.0019286, 0.0019317
5: 0.0097651, 0.0109087, 0.0097651, 0.0109087, -0.0007305, 0.0007317
6: 0.0053643, 0.0097284, 0.0053643, 0.0097284, -0.0027921, 0.0027876
7: 0.9818129, 0.9848667, 0.9818129, 0.9848667, -0.0019538, 0.0019506
8: -0.0060636, -0.0027895, -0.0060636, -0.0027895, -0.0020948, 0.0020914
9: -0.0031570, -0.0009942, -0.0031570, -0.0009942, -0.0013815, 0.0013837

### Unstable ReLU Count (Linear/Conv2D Layers)
- layer_idx=0, type=LayerType.Linear, total=256, inp1_unstable=10, inp2_unstable=10, delta_unstable=252
- layer_idx=2, type=LayerType.Linear, total=256, inp1_unstable=10, inp2_unstable=10, delta_unstable=256
- layer_idx=4, type=LayerType.Linear, total=256, inp1_unstable=0, inp2_unstable=0, delta_unstable=256
- layer_idx=6, type=LayerType.Linear, total=256, inp1_unstable=0, inp2_unstable=0, delta_unstable=256
- layer_idx=8, type=LayerType.Linear, total=10, inp1_unstable=0, inp2_unstable=0, delta_unstable=10

Time for backsubstitution: 1.57 seconds

### RS candidates at layer 1
type: RSZ, layer: 1, pos: 105
type: RSZ, layer: 1, pos: 123
type: RSZ, layer: 1, pos: 140
type: RSZ, layer: 1, pos: 167
type: RSZ, layer: 1, pos: 213
type: RSZ, layer: 1, pos: 219

Time for candidate selection: 0.16 seconds

### Candidate
type: RSZ, layer: 1, pos: 105

### Relational analysis RSZ of RS_RSZ1_RSZ2_RSZ2_RSZ1_RSZ1

#### Relational analysis RSZ result of RS_RSZ1_RSZ2_RSZ2_RSZ1_RSZ1
Status: Status.UNKNOWN
Output dim: 7, lower bound: -0.0009596, upper bound: 0.0009800
time: 0.64 seconds

### Relational analysis RSZ of RS_RSZ1_RSZ2_RSZ2_RSZ1_RSZ2

#### Relational analysis RSZ result of RS_RSZ1_RSZ2_RSZ2_RSZ1_RSZ2
Status: Status.UNKNOWN
Output dim: 7, lower bound: -0.0009670, upper bound: 0.0009667
time: 0.65 seconds

## BFS RS instance: RS_RSZ1_RSZ2_RSZ2_RSZ2

### Backsubstitution after applying RS history:
0: -0.0027939, -0.0016242, -0.0027939, -0.0016242, -0.0007531, 0.0007398
1: -0.0114005, -0.0084323, -0.0114005, -0.0084323, -0.0019111, 0.0018773
2: 0.0279571, 0.0297986, 0.0279571, 0.0297986, -0.0011857, 0.0011647
3: 0.0039210, 0.0073596, 0.0039210, 0.0073596, -0.0021748, 0.0022139
4: -0.0104893, -0.0074701, -0.0104893, -0.0074701, -0.0019439, 0.0019095
5: 0.0097651, 0.0109087, 0.0097651, 0.0109087, -0.0007363, 0.0007233
6: 0.0053643, 0.0097284, 0.0053643, 0.0097284, -0.0027601, 0.0028098
7: 0.9818129, 0.9848667, 0.9818129, 0.9848667, -0.0019314, 0.0019661
8: -0.0060636, -0.0027895, -0.0060636, -0.0027895, -0.0020707, 0.0021080
9: -0.0031570, -0.0009942, -0.0031570, -0.0009942, -0.0013925, 0.0013678

### Unstable ReLU Count (Linear/Conv2D Layers)
- layer_idx=0, type=LayerType.Linear, total=256, inp1_unstable=10, inp2_unstable=10, delta_unstable=252
- layer_idx=2, type=LayerType.Linear, total=256, inp1_unstable=10, inp2_unstable=10, delta_unstable=256
- layer_idx=4, type=LayerType.Linear, total=256, inp1_unstable=0, inp2_unstable=0, delta_unstable=256
- layer_idx=6, type=LayerType.Linear, total=256, inp1_unstable=0, inp2_unstable=0, delta_unstable=256
- layer_idx=8, type=LayerType.Linear, total=10, inp1_unstable=0, inp2_unstable=0, delta_unstable=10

Time for backsubstitution: 1.51 seconds

### RS candidates at layer 1
type: RSZ, layer: 1, pos: 105
type: RSZ, layer: 1, pos: 123
type: RSZ, layer: 1, pos: 140
type: RSZ, layer: 1, pos: 167
type: RSZ, layer: 1, pos: 213
type: RSZ, layer: 1, pos: 219

Time for candidate selection: 0.19 seconds

### Candidate
type: RSZ, layer: 1, pos: 105

### Relational analysis RSZ of RS_RSZ1_RSZ2_RSZ2_RSZ2_RSZ1

#### Relational analysis RSZ result of RS_RSZ1_RSZ2_RSZ2_RSZ2_RSZ1
Status: Status.UNKNOWN
Output dim: 7, lower bound: -0.0009599, upper bound: 0.0009757
time: 0.62 seconds

### Relational analysis RSZ of RS_RSZ1_RSZ2_RSZ2_RSZ2_RSZ2

#### Relational analysis RSZ result of RS_RSZ1_RSZ2_RSZ2_RSZ2_RSZ2
Status: Status.UNKNOWN
Output dim: 7, lower bound: -0.0009671, upper bound: 0.0009578
time: 0.64 seconds

## BFS RS instance: RS_RSZ2_RSZ1_RSZ1_RSZ1

### Backsubstitution after applying RS history:
0: -0.0027939, -0.0016242, -0.0027939, -0.0016242, -0.0007398, 0.0007533
1: -0.0114005, -0.0084323, -0.0114005, -0.0084323, -0.0018773, 0.0019116
2: 0.0279571, 0.0297986, 0.0279571, 0.0297986, -0.0011647, 0.0011859
3: 0.0039210, 0.0073596, 0.0039210, 0.0073596, -0.0022145, 0.0021748
4: -0.0104893, -0.0074701, -0.0104893, -0.0074701, -0.0019095, 0.0019444
5: 0.0097651, 0.0109087, 0.0097651, 0.0109087, -0.0007233, 0.0007365
6: 0.0053643, 0.0097284, 0.0053643, 0.0097284, -0.0028104, 0.0027601
7: 0.9818129, 0.9848667, 0.9818129, 0.9848667, -0.0019666, 0.0019314
8: -0.0060636, -0.0027895, -0.0060636, -0.0027895, -0.0021085, 0.0020707
9: -0.0031570, -0.0009942, -0.0031570, -0.0009942, -0.0013678, 0.0013928

### Unstable ReLU Count (Linear/Conv2D Layers)
- layer_idx=0, type=LayerType.Linear, total=256, inp1_unstable=10, inp2_unstable=10, delta_unstable=252
- layer_idx=2, type=LayerType.Linear, total=256, inp1_unstable=10, inp2_unstable=10, delta_unstable=256
- layer_idx=4, type=LayerType.Linear, total=256, inp1_unstable=0, inp2_unstable=0, delta_unstable=256
- layer_idx=6, type=LayerType.Linear, total=256, inp1_unstable=0, inp2_unstable=0, delta_unstable=256
- layer_idx=8, type=LayerType.Linear, total=10, inp1_unstable=0, inp2_unstable=0, delta_unstable=10

Time for backsubstitution: 1.49 seconds

### RS candidates at layer 1
type: RSZ, layer: 1, pos: 105
type: RSZ, layer: 1, pos: 123
type: RSZ, layer: 1, pos: 140
type: RSZ, layer: 1, pos: 167
type: RSZ, layer: 1, pos: 213
type: RSZ, layer: 1, pos: 219

Time for candidate selection: 0.16 seconds

### Candidate
type: RSZ, layer: 1, pos: 105

### Relational analysis RSZ of RS_RSZ2_RSZ1_RSZ1_RSZ1_RSZ1

#### Relational analysis RSZ result of RS_RSZ2_RSZ1_RSZ1_RSZ1_RSZ1
Status: Status.UNKNOWN
Output dim: 7, lower bound: -0.0009578, upper bound: 0.0009671
time: 0.65 seconds

### Relational analysis RSZ of RS_RSZ2_RSZ1_RSZ1_RSZ1_RSZ2

#### Relational analysis RSZ result of RS_RSZ2_RSZ1_RSZ1_RSZ1_RSZ2
Status: Status.UNKNOWN
Output dim: 7, lower bound: -0.0009757, upper bound: 0.0009599
time: 0.64 seconds

## BFS RS instance: RS_RSZ2_RSZ1_RSZ1_RSZ2

### Backsubstitution after applying RS history:
0: -0.0027939, -0.0016242, -0.0027939, -0.0016242, -0.0007484, 0.0007467
1: -0.0114005, -0.0084323, -0.0114005, -0.0084323, -0.0018991, 0.0018949
2: 0.0279571, 0.0297986, 0.0279571, 0.0297986, -0.0011782, 0.0011756
3: 0.0039210, 0.0073596, 0.0039210, 0.0073596, -0.0021952, 0.0022000
4: -0.0104893, -0.0074701, -0.0104893, -0.0074701, -0.0019317, 0.0019274
5: 0.0097651, 0.0109087, 0.0097651, 0.0109087, -0.0007317, 0.0007301
6: 0.0053643, 0.0097284, 0.0053643, 0.0097284, -0.0027860, 0.0027921
7: 0.9818129, 0.9848667, 0.9818129, 0.9848667, -0.0019495, 0.0019538
8: -0.0060636, -0.0027895, -0.0060636, -0.0027895, -0.0020901, 0.0020948
9: -0.0031570, -0.0009942, -0.0031570, -0.0009942, -0.0013837, 0.0013807

### Unstable ReLU Count (Linear/Conv2D Layers)
- layer_idx=0, type=LayerType.Linear, total=256, inp1_unstable=10, inp2_unstable=10, delta_unstable=252
- layer_idx=2, type=LayerType.Linear, total=256, inp1_unstable=10, inp2_unstable=10, delta_unstable=256
- layer_idx=4, type=LayerType.Linear, total=256, inp1_unstable=0, inp2_unstable=0, delta_unstable=256
- layer_idx=6, type=LayerType.Linear, total=256, inp1_unstable=0, inp2_unstable=0, delta_unstable=256
- layer_idx=8, type=LayerType.Linear, total=10, inp1_unstable=0, inp2_unstable=0, delta_unstable=10

Time for backsubstitution: 1.55 seconds

### RS candidates at layer 1
type: RSZ, layer: 1, pos: 105
type: RSZ, layer: 1, pos: 123
type: RSZ, layer: 1, pos: 140
type: RSZ, layer: 1, pos: 167
type: RSZ, layer: 1, pos: 213
type: RSZ, layer: 1, pos: 219

Time for candidate selection: 0.16 seconds

### Candidate
type: RSZ, layer: 1, pos: 105

### Relational analysis RSZ of RS_RSZ2_RSZ1_RSZ1_RSZ2_RSZ1

#### Relational analysis RSZ result of RS_RSZ2_RSZ1_RSZ1_RSZ2_RSZ1
Status: Status.UNKNOWN
Output dim: 7, lower bound: -0.0009667, upper bound: 0.0009670
time: 0.64 seconds

### Relational analysis RSZ of RS_RSZ2_RSZ1_RSZ1_RSZ2_RSZ2

#### Relational analysis RSZ result of RS_RSZ2_RSZ1_RSZ1_RSZ2_RSZ2
Status: Status.UNKNOWN
Output dim: 7, lower bound: -0.0009800, upper bound: 0.0009596
time: 0.64 seconds

## BFS RS instance: RS_RSZ2_RSZ1_RSZ2_RSZ1

### Backsubstitution after applying RS history:
0: -0.0027939, -0.0016242, -0.0027939, -0.0016242, -0.0007417, 0.0007512
1: -0.0114005, -0.0084323, -0.0114005, -0.0084323, -0.0018823, 0.0019062
2: 0.0279571, 0.0297986, 0.0279571, 0.0297986, -0.0011678, 0.0011826
3: 0.0039210, 0.0073596, 0.0039210, 0.0073596, -0.0022082, 0.0021805
4: -0.0104893, -0.0074701, -0.0104893, -0.0074701, -0.0019146, 0.0019389
5: 0.0097651, 0.0109087, 0.0097651, 0.0109087, -0.0007252, 0.0007344
6: 0.0053643, 0.0097284, 0.0053643, 0.0097284, -0.0028025, 0.0027674
7: 0.9818129, 0.9848667, 0.9818129, 0.9848667, -0.0019610, 0.0019365
8: -0.0060636, -0.0027895, -0.0060636, -0.0027895, -0.0021025, 0.0020762
9: -0.0031570, -0.0009942, -0.0031570, -0.0009942, -0.0013714, 0.0013888

### Unstable ReLU Count (Linear/Conv2D Layers)
- layer_idx=0, type=LayerType.Linear, total=256, inp1_unstable=10, inp2_unstable=10, delta_unstable=252
- layer_idx=2, type=LayerType.Linear, total=256, inp1_unstable=10, inp2_unstable=10, delta_unstable=256
- layer_idx=4, type=LayerType.Linear, total=256, inp1_unstable=0, inp2_unstable=0, delta_unstable=256
- layer_idx=6, type=LayerType.Linear, total=256, inp1_unstable=0, inp2_unstable=0, delta_unstable=256
- layer_idx=8, type=LayerType.Linear, total=10, inp1_unstable=0, inp2_unstable=0, delta_unstable=10

Time for backsubstitution: 1.55 seconds

### RS candidates at layer 1
type: RSZ, layer: 1, pos: 105
type: RSZ, layer: 1, pos: 123
type: RSZ, layer: 1, pos: 140
type: RSZ, layer: 1, pos: 167
type: RSZ, layer: 1, pos: 213
type: RSZ, layer: 1, pos: 219

Time for candidate selection: 0.17 seconds

### Candidate
type: RSZ, layer: 1, pos: 105

### Relational analysis RSZ of RS_RSZ2_RSZ1_RSZ2_RSZ1_RSZ1

#### Relational analysis RSZ result of RS_RSZ2_RSZ1_RSZ2_RSZ1_RSZ1
Status: Status.UNKNOWN
Output dim: 7, lower bound: -0.0009578, upper bound: 0.0009671
time: 0.64 seconds

### Relational analysis RSZ of RS_RSZ2_RSZ1_RSZ2_RSZ1_RSZ2

#### Relational analysis RSZ result of RS_RSZ2_RSZ1_RSZ2_RSZ1_RSZ2
Status: Status.UNKNOWN
Output dim: 7, lower bound: -0.0009757, upper bound: 0.0009599
time: 0.64 seconds

## BFS RS instance: RS_RSZ2_RSZ1_RSZ2_RSZ2

### Backsubstitution after applying RS history:
0: -0.0027939, -0.0016242, -0.0027939, -0.0016242, -0.0007503, 0.0007452
1: -0.0114005, -0.0084323, -0.0114005, -0.0084323, -0.0019041, 0.0018911
2: 0.0279571, 0.0297986, 0.0279571, 0.0297986, -0.0011813, 0.0011732
3: 0.0039210, 0.0073596, 0.0039210, 0.0073596, -0.0021907, 0.0022058
4: -0.0104893, -0.0074701, -0.0104893, -0.0074701, -0.0019367, 0.0019235
5: 0.0097651, 0.0109087, 0.0097651, 0.0109087, -0.0007336, 0.0007286
6: 0.0053643, 0.0097284, 0.0053643, 0.0097284, -0.0027803, 0.0027994
7: 0.9818129, 0.9848667, 0.9818129, 0.9848667, -0.0019455, 0.0019589
8: -0.0060636, -0.0027895, -0.0060636, -0.0027895, -0.0020859, 0.0021002
9: -0.0031570, -0.0009942, -0.0031570, -0.0009942, -0.0013873, 0.0013778

### Unstable ReLU Count (Linear/Conv2D Layers)
- layer_idx=0, type=LayerType.Linear, total=256, inp1_unstable=10, inp2_unstable=10, delta_unstable=252
- layer_idx=2, type=LayerType.Linear, total=256, inp1_unstable=10, inp2_unstable=10, delta_unstable=256
- layer_idx=4, type=LayerType.Linear, total=256, inp1_unstable=0, inp2_unstable=0, delta_unstable=256
- layer_idx=6, type=LayerType.Linear, total=256, inp1_unstable=0, inp2_unstable=0, delta_unstable=256
- layer_idx=8, type=LayerType.Linear, total=10, inp1_unstable=0, inp2_unstable=0, delta_unstable=10

Time for backsubstitution: 1.51 seconds

### RS candidates at layer 1
type: RSZ, layer: 1, pos: 105
type: RSZ, layer: 1, pos: 123
type: RSZ, layer: 1, pos: 140
type: RSZ, layer: 1, pos: 167
type: RSZ, layer: 1, pos: 213
type: RSZ, layer: 1, pos: 219

Time for candidate selection: 0.17 seconds

### Candidate
type: RSZ, layer: 1, pos: 105

### Relational analysis RSZ of RS_RSZ2_RSZ1_RSZ2_RSZ2_RSZ1

#### Relational analysis RSZ result of RS_RSZ2_RSZ1_RSZ2_RSZ2_RSZ1
Status: Status.UNKNOWN
Output dim: 7, lower bound: -0.0009667, upper bound: 0.0009670
time: 0.62 seconds

### Relational analysis RSZ of RS_RSZ2_RSZ1_RSZ2_RSZ2_RSZ2

#### Relational analysis RSZ result of RS_RSZ2_RSZ1_RSZ2_RSZ2_RSZ2
Status: Status.UNKNOWN
Output dim: 7, lower bound: -0.0009800, upper bound: 0.0009596
time: 0.64 seconds

## BFS RS instance: RS_RSZ2_RSZ2_RSZ1_RSZ1

### Backsubstitution after applying RS history:
0: -0.0027939, -0.0016242, -0.0027939, -0.0016242, -0.0007464, 0.0007481
1: -0.0114005, -0.0084323, -0.0114005, -0.0084323, -0.0018942, 0.0018984
2: 0.0279571, 0.0297986, 0.0279571, 0.0297986, -0.0011751, 0.0011778
3: 0.0039210, 0.0073596, 0.0039210, 0.0073596, -0.0021993, 0.0021943
4: -0.0104893, -0.0074701, -0.0104893, -0.0074701, -0.0019267, 0.0019310
5: 0.0097651, 0.0109087, 0.0097651, 0.0109087, -0.0007298, 0.0007314
6: 0.0053643, 0.0097284, 0.0053643, 0.0097284, -0.0027911, 0.0027848
7: 0.9818129, 0.9848667, 0.9818129, 0.9848667, -0.0019531, 0.0019487
8: -0.0060636, -0.0027895, -0.0060636, -0.0027895, -0.0020940, 0.0020893
9: -0.0031570, -0.0009942, -0.0031570, -0.0009942, -0.0013801, 0.0013832

### Unstable ReLU Count (Linear/Conv2D Layers)
- layer_idx=0, type=LayerType.Linear, total=256, inp1_unstable=10, inp2_unstable=10, delta_unstable=252
- layer_idx=2, type=LayerType.Linear, total=256, inp1_unstable=10, inp2_unstable=10, delta_unstable=256
- layer_idx=4, type=LayerType.Linear, total=256, inp1_unstable=0, inp2_unstable=0, delta_unstable=256
- layer_idx=6, type=LayerType.Linear, total=256, inp1_unstable=0, inp2_unstable=0, delta_unstable=256
- layer_idx=8, type=LayerType.Linear, total=10, inp1_unstable=0, inp2_unstable=0, delta_unstable=10

Time for backsubstitution: 1.53 seconds

### RS candidates at layer 1
type: RSZ, layer: 1, pos: 105
type: RSZ, layer: 1, pos: 123
type: RSZ, layer: 1, pos: 140
type: RSZ, layer: 1, pos: 167
type: RSZ, layer: 1, pos: 213
type: RSZ, layer: 1, pos: 219

Time for candidate selection: 0.18 seconds

### Candidate
type: RSZ, layer: 1, pos: 105

### Relational analysis RSZ of RS_RSZ2_RSZ2_RSZ1_RSZ1_RSZ1

#### Relational analysis RSZ result of RS_RSZ2_RSZ2_RSZ1_RSZ1_RSZ1
Status: Status.UNKNOWN
Output dim: 7, lower bound: -0.0009495, upper bound: 0.0009862
time: 0.64 seconds

### Relational analysis RSZ of RS_RSZ2_RSZ2_RSZ1_RSZ1_RSZ2

#### Relational analysis RSZ result of RS_RSZ2_RSZ2_RSZ1_RSZ1_RSZ2
Status: Status.UNKNOWN
Output dim: 7, lower bound: -0.0009576, upper bound: 0.0009748
time: 0.63 seconds

## BFS RS instance: RS_RSZ2_RSZ2_RSZ1_RSZ2

### Backsubstitution after applying RS history:
0: -0.0027939, -0.0016242, -0.0027939, -0.0016242, -0.0007533, 0.0007405
1: -0.0114005, -0.0084323, -0.0114005, -0.0084323, -0.0019117, 0.0018792
2: 0.0279571, 0.0297986, 0.0279571, 0.0297986, -0.0011860, 0.0011658
3: 0.0039210, 0.0073596, 0.0039210, 0.0073596, -0.0021769, 0.0022146
4: -0.0104893, -0.0074701, -0.0104893, -0.0074701, -0.0019445, 0.0019114
5: 0.0097651, 0.0109087, 0.0097651, 0.0109087, -0.0007365, 0.0007240
6: 0.0053643, 0.0097284, 0.0053643, 0.0097284, -0.0027628, 0.0028107
7: 0.9818129, 0.9848667, 0.9818129, 0.9848667, -0.0019333, 0.0019668
8: -0.0060636, -0.0027895, -0.0060636, -0.0027895, -0.0020728, 0.0021087
9: -0.0031570, -0.0009942, -0.0031570, -0.0009942, -0.0013929, 0.0013692

### Unstable ReLU Count (Linear/Conv2D Layers)
- layer_idx=0, type=LayerType.Linear, total=256, inp1_unstable=10, inp2_unstable=10, delta_unstable=252
- layer_idx=2, type=LayerType.Linear, total=256, inp1_unstable=10, inp2_unstable=10, delta_unstable=256
- layer_idx=4, type=LayerType.Linear, total=256, inp1_unstable=0, inp2_unstable=0, delta_unstable=256
- layer_idx=6, type=LayerType.Linear, total=256, inp1_unstable=0, inp2_unstable=0, delta_unstable=256
- layer_idx=8, type=LayerType.Linear, total=10, inp1_unstable=0, inp2_unstable=0, delta_unstable=10

Time for backsubstitution: 1.51 seconds

### RS candidates at layer 1
type: RSZ, layer: 1, pos: 105
type: RSZ, layer: 1, pos: 123
type: RSZ, layer: 1, pos: 140
type: RSZ, layer: 1, pos: 167
type: RSZ, layer: 1, pos: 213
type: RSZ, layer: 1, pos: 219

Time for candidate selection: 0.16 seconds

### Candidate
type: RSZ, layer: 1, pos: 105

### Relational analysis RSZ of RS_RSZ2_RSZ2_RSZ1_RSZ2_RSZ1

#### Relational analysis RSZ result of RS_RSZ2_RSZ2_RSZ1_RSZ2_RSZ1
Status: Status.UNKNOWN
Output dim: 7, lower bound: -0.0009545, upper bound: 0.0009855
time: 0.65 seconds

### Relational analysis RSZ of RS_RSZ2_RSZ2_RSZ1_RSZ2_RSZ2

#### Relational analysis RSZ result of RS_RSZ2_RSZ2_RSZ1_RSZ2_RSZ2
Status: Status.UNKNOWN
Output dim: 7, lower bound: -0.0009613, upper bound: 0.0009747
time: 0.64 seconds

## BFS RS instance: RS_RSZ2_RSZ2_RSZ2_RSZ1

### Backsubstitution after applying RS history:
0: -0.0027939, -0.0016242, -0.0027939, -0.0016242, -0.0007484, 0.0007462
1: -0.0114005, -0.0084323, -0.0114005, -0.0084323, -0.0018991, 0.0018935
2: 0.0279571, 0.0297986, 0.0279571, 0.0297986, -0.0011782, 0.0011748
3: 0.0039210, 0.0073596, 0.0039210, 0.0073596, -0.0021936, 0.0022000
4: -0.0104893, -0.0074701, -0.0104893, -0.0074701, -0.0019317, 0.0019260
5: 0.0097651, 0.0109087, 0.0097651, 0.0109087, -0.0007317, 0.0007295
6: 0.0053643, 0.0097284, 0.0053643, 0.0097284, -0.0027839, 0.0027921
7: 0.9818129, 0.9848667, 0.9818129, 0.9848667, -0.0019480, 0.0019538
8: -0.0060636, -0.0027895, -0.0060636, -0.0027895, -0.0020886, 0.0020948
9: -0.0031570, -0.0009942, -0.0031570, -0.0009942, -0.0013837, 0.0013796

### Unstable ReLU Count (Linear/Conv2D Layers)
- layer_idx=0, type=LayerType.Linear, total=256, inp1_unstable=10, inp2_unstable=10, delta_unstable=252
- layer_idx=2, type=LayerType.Linear, total=256, inp1_unstable=10, inp2_unstable=10, delta_unstable=256
- layer_idx=4, type=LayerType.Linear, total=256, inp1_unstable=0, inp2_unstable=0, delta_unstable=256
- layer_idx=6, type=LayerType.Linear, total=256, inp1_unstable=0, inp2_unstable=0, delta_unstable=256
- layer_idx=8, type=LayerType.Linear, total=10, inp1_unstable=0, inp2_unstable=0, delta_unstable=10

Time for backsubstitution: 1.63 seconds

### RS candidates at layer 1
type: RSZ, layer: 1, pos: 105
type: RSZ, layer: 1, pos: 123
type: RSZ, layer: 1, pos: 140
type: RSZ, layer: 1, pos: 167
type: RSZ, layer: 1, pos: 213
type: RSZ, layer: 1, pos: 219

Time for candidate selection: 0.17 seconds

### Candidate
type: RSZ, layer: 1, pos: 105

### Relational analysis RSZ of RS_RSZ2_RSZ2_RSZ2_RSZ1_RSZ1

#### Relational analysis RSZ result of RS_RSZ2_RSZ2_RSZ2_RSZ1_RSZ1
Status: Status.UNKNOWN
Output dim: 7, lower bound: -0.0009495, upper bound: 0.0009862
time: 0.63 seconds

### Relational analysis RSZ of RS_RSZ2_RSZ2_RSZ2_RSZ1_RSZ2

#### Relational analysis RSZ result of RS_RSZ2_RSZ2_RSZ2_RSZ1_RSZ2
Status: Status.UNKNOWN
Output dim: 7, lower bound: -0.0009576, upper bound: 0.0009748
time: 0.64 seconds

## BFS RS instance: RS_RSZ2_RSZ2_RSZ2_RSZ2

### Backsubstitution after applying RS history:
0: -0.0027939, -0.0016242, -0.0027939, -0.0016242, -0.0007553, 0.0007386
1: -0.0114005, -0.0084323, -0.0114005, -0.0084323, -0.0019167, 0.0018744
2: 0.0279571, 0.0297986, 0.0279571, 0.0297986, -0.0011891, 0.0011629
3: 0.0039210, 0.0073596, 0.0039210, 0.0073596, -0.0021714, 0.0022204
4: -0.0104893, -0.0074701, -0.0104893, -0.0074701, -0.0019496, 0.0019066
5: 0.0097651, 0.0109087, 0.0097651, 0.0109087, -0.0007385, 0.0007222
6: 0.0053643, 0.0097284, 0.0053643, 0.0097284, -0.0027558, 0.0028180
7: 0.9818129, 0.9848667, 0.9818129, 0.9848667, -0.0019284, 0.0019719
8: -0.0060636, -0.0027895, -0.0060636, -0.0027895, -0.0020675, 0.0021142
9: -0.0031570, -0.0009942, -0.0031570, -0.0009942, -0.0013965, 0.0013657

### Unstable ReLU Count (Linear/Conv2D Layers)
- layer_idx=0, type=LayerType.Linear, total=256, inp1_unstable=10, inp2_unstable=10, delta_unstable=252
- layer_idx=2, type=LayerType.Linear, total=256, inp1_unstable=10, inp2_unstable=10, delta_unstable=256
- layer_idx=4, type=LayerType.Linear, total=256, inp1_unstable=0, inp2_unstable=0, delta_unstable=256
- layer_idx=6, type=LayerType.Linear, total=256, inp1_unstable=0, inp2_unstable=0, delta_unstable=256
- layer_idx=8, type=LayerType.Linear, total=10, inp1_unstable=0, inp2_unstable=0, delta_unstable=10

Time for backsubstitution: 1.52 seconds

### RS candidates at layer 1
type: RSZ, layer: 1, pos: 105
type: RSZ, layer: 1, pos: 123
type: RSZ, layer: 1, pos: 140
type: RSZ, layer: 1, pos: 167
type: RSZ, layer: 1, pos: 213
type: RSZ, layer: 1, pos: 219

Time for candidate selection: 0.20 seconds

### Candidate
type: RSZ, layer: 1, pos: 105

### Relational analysis RSZ of RS_RSZ2_RSZ2_RSZ2_RSZ2_RSZ1

#### Relational analysis RSZ result of RS_RSZ2_RSZ2_RSZ2_RSZ2_RSZ1
Status: Status.UNKNOWN
Output dim: 7, lower bound: -0.0009545, upper bound: 0.0009855
time: 0.69 seconds

### Relational analysis RSZ of RS_RSZ2_RSZ2_RSZ2_RSZ2_RSZ2

#### Relational analysis RSZ result of RS_RSZ2_RSZ2_RSZ2_RSZ2_RSZ2
Status: Status.UNKNOWN
Output dim: 7, lower bound: -0.0009613, upper bound: 0.0009747
time: 0.67 seconds

## Summary of splitting (split count: 4)
- Time for RS candidates: 3.09 seconds
RS_RSZ1_RSZ1_RSZ1_RSZ1_RSZ1, status: Status.UNKNOWN, split count: 5, time: 3.09
Output dim: 7, lower bound: -0.0009747, upper bound: 0.0009613
RS_RSZ1_RSZ1_RSZ1_RSZ1_RSZ2, status: Status.UNKNOWN, split count: 5, time: 3.09
Output dim: 7, lower bound: -0.0009855, upper bound: 0.0009545
RS_RSZ1_RSZ1_RSZ1_RSZ2_RSZ1, status: Status.UNKNOWN, split count: 5, time: 3.09
Output dim: 7, lower bound: -0.0009748, upper bound: 0.0009576
RS_RSZ1_RSZ1_RSZ1_RSZ2_RSZ2, status: Status.UNKNOWN, split count: 5, time: 3.09
Output dim: 7, lower bound: -0.0009862, upper bound: 0.0009495
RS_RSZ1_RSZ1_RSZ2_RSZ1_RSZ1, status: Status.UNKNOWN, split count: 5, time: 3.09
Output dim: 7, lower bound: -0.0009747, upper bound: 0.0009613
RS_RSZ1_RSZ1_RSZ2_RSZ1_RSZ2, status: Status.UNKNOWN, split count: 5, time: 3.09
Output dim: 7, lower bound: -0.0009855, upper bound: 0.0009545
RS_RSZ1_RSZ1_RSZ2_RSZ2_RSZ1, status: Status.UNKNOWN, split count: 5, time: 3.09
Output dim: 7, lower bound: -0.0009748, upper bound: 0.0009576
RS_RSZ1_RSZ1_RSZ2_RSZ2_RSZ2, status: Status.UNKNOWN, split count: 5, time: 3.09
Output dim: 7, lower bound: -0.0009862, upper bound: 0.0009495
RS_RSZ1_RSZ2_RSZ1_RSZ1_RSZ1, status: Status.UNKNOWN, split count: 5, time: 3.09
Output dim: 7, lower bound: -0.0009596, upper bound: 0.0009800
RS_RSZ1_RSZ2_RSZ1_RSZ1_RSZ2, status: Status.UNKNOWN, split count: 5, time: 3.09
Output dim: 7, lower bound: -0.0009670, upper bound: 0.0009667
RS_RSZ1_RSZ2_RSZ1_RSZ2_RSZ1, status: Status.UNKNOWN, split count: 5, time: 3.09
Output dim: 7, lower bound: -0.0009599, upper bound: 0.0009757
RS_RSZ1_RSZ2_RSZ1_RSZ2_RSZ2, status: Status.UNKNOWN, split count: 5, time: 3.09
Output dim: 7, lower bound: -0.0009671, upper bound: 0.0009578
RS_RSZ1_RSZ2_RSZ2_RSZ1_RSZ1, status: Status.UNKNOWN, split count: 5, time: 3.09
Output dim: 7, lower bound: -0.0009596, upper bound: 0.0009800
RS_RSZ1_RSZ2_RSZ2_RSZ1_RSZ2, status: Status.UNKNOWN, split count: 5, time: 3.09
Output dim: 7, lower bound: -0.0009670, upper bound: 0.0009667
RS_RSZ1_RSZ2_RSZ2_RSZ2_RSZ1, status: Status.UNKNOWN, split count: 5, time: 3.09
Output dim: 7, lower bound: -0.0009599, upper bound: 0.0009757
RS_RSZ1_RSZ2_RSZ2_RSZ2_RSZ2, status: Status.UNKNOWN, split count: 5, time: 3.09
Output dim: 7, lower bound: -0.0009671, upper bound: 0.0009578
RS_RSZ2_RSZ1_RSZ1_RSZ1_RSZ1, status: Status.UNKNOWN, split count: 5, time: 3.09
Output dim: 7, lower bound: -0.0009578, upper bound: 0.0009671
RS_RSZ2_RSZ1_RSZ1_RSZ1_RSZ2, status: Status.UNKNOWN, split count: 5, time: 3.09
Output dim: 7, lower bound: -0.0009757, upper bound: 0.0009599
RS_RSZ2_RSZ1_RSZ1_RSZ2_RSZ1, status: Status.UNKNOWN, split count: 5, time: 3.09
Output dim: 7, lower bound: -0.0009667, upper bound: 0.0009670
RS_RSZ2_RSZ1_RSZ1_RSZ2_RSZ2, status: Status.UNKNOWN, split count: 5, time: 3.09
Output dim: 7, lower bound: -0.0009800, upper bound: 0.0009596
RS_RSZ2_RSZ1_RSZ2_RSZ1_RSZ1, status: Status.UNKNOWN, split count: 5, time: 3.09
Output dim: 7, lower bound: -0.0009578, upper bound: 0.0009671
RS_RSZ2_RSZ1_RSZ2_RSZ1_RSZ2, status: Status.UNKNOWN, split count: 5, time: 3.09
Output dim: 7, lower bound: -0.0009757, upper bound: 0.0009599
RS_RSZ2_RSZ1_RSZ2_RSZ2_RSZ1, status: Status.UNKNOWN, split count: 5, time: 3.09
Output dim: 7, lower bound: -0.0009667, upper bound: 0.0009670
RS_RSZ2_RSZ1_RSZ2_RSZ2_RSZ2, status: Status.UNKNOWN, split count: 5, time: 3.09
Output dim: 7, lower bound: -0.0009800, upper bound: 0.0009596
RS_RSZ2_RSZ2_RSZ1_RSZ1_RSZ1, status: Status.UNKNOWN, split count: 5, time: 3.09
Output dim: 7, lower bound: -0.0009495, upper bound: 0.0009862
RS_RSZ2_RSZ2_RSZ1_RSZ1_RSZ2, status: Status.UNKNOWN, split count: 5, time: 3.09
Output dim: 7, lower bound: -0.0009576, upper bound: 0.0009748
RS_RSZ2_RSZ2_RSZ1_RSZ2_RSZ1, status: Status.UNKNOWN, split count: 5, time: 3.09
Output dim: 7, lower bound: -0.0009545, upper bound: 0.0009855
RS_RSZ2_RSZ2_RSZ1_RSZ2_RSZ2, status: Status.UNKNOWN, split count: 5, time: 3.09
Output dim: 7, lower bound: -0.0009613, upper bound: 0.0009747
RS_RSZ2_RSZ2_RSZ2_RSZ1_RSZ1, status: Status.UNKNOWN, split count: 5, time: 3.09
Output dim: 7, lower bound: -0.0009495, upper bound: 0.0009862
RS_RSZ2_RSZ2_RSZ2_RSZ1_RSZ2, status: Status.UNKNOWN, split count: 5, time: 3.09
Output dim: 7, lower bound: -0.0009576, upper bound: 0.0009748
RS_RSZ2_RSZ2_RSZ2_RSZ2_RSZ1, status: Status.UNKNOWN, split count: 5, time: 3.09
Output dim: 7, lower bound: -0.0009545, upper bound: 0.0009855
RS_RSZ2_RSZ2_RSZ2_RSZ2_RSZ2, status: Status.UNKNOWN, split count: 5, time: 3.09
Output dim: 7, lower bound: -0.0009613, upper bound: 0.0009747

## BFS RS instance: RS_RSZ1_RSZ1_RSZ1_RSZ1_RSZ1

### Backsubstitution after applying RS history:
0: -0.0027939, -0.0016242, -0.0027939, -0.0016242, -0.0007504, 0.0007524
1: -0.0114005, -0.0084323, -0.0114005, -0.0084323, -0.0019042, 0.0019093
2: 0.0279571, 0.0297986, 0.0279571, 0.0297986, -0.0011813, 0.0011845
3: 0.0039210, 0.0073596, 0.0039210, 0.0073596, -0.0022118, 0.0022059
4: -0.0104893, -0.0074701, -0.0104893, -0.0074701, -0.0019369, 0.0019421
5: 0.0097651, 0.0109087, 0.0097651, 0.0109087, -0.0007336, 0.0007356
6: 0.0053643, 0.0097284, 0.0053643, 0.0097284, -0.0028071, 0.0027995
7: 0.9818129, 0.9848667, 0.9818129, 0.9848667, -0.0019643, 0.0019590
8: -0.0060636, -0.0027895, -0.0060636, -0.0027895, -0.0021060, 0.0021003
9: -0.0031570, -0.0009942, -0.0031570, -0.0009942, -0.0013874, 0.0013911

### Unstable ReLU Count (Linear/Conv2D Layers)
- layer_idx=0, type=LayerType.Linear, total=256, inp1_unstable=10, inp2_unstable=10, delta_unstable=251
- layer_idx=2, type=LayerType.Linear, total=256, inp1_unstable=10, inp2_unstable=10, delta_unstable=256
- layer_idx=4, type=LayerType.Linear, total=256, inp1_unstable=0, inp2_unstable=0, delta_unstable=256
- layer_idx=6, type=LayerType.Linear, total=256, inp1_unstable=0, inp2_unstable=0, delta_unstable=256
- layer_idx=8, type=LayerType.Linear, total=10, inp1_unstable=0, inp2_unstable=0, delta_unstable=10

Time for backsubstitution: 1.52 seconds

### RS candidates at layer 1
type: RSZ, layer: 1, pos: 123
type: RSZ, layer: 1, pos: 140
type: RSZ, layer: 1, pos: 167
type: RSZ, layer: 1, pos: 213
type: RSZ, layer: 1, pos: 219

Time for candidate selection: 0.16 seconds

### Candidate
type: RSZ, layer: 1, pos: 123

### Relational analysis RSZ of RS_RSZ1_RSZ1_RSZ1_RSZ1_RSZ1_RSZ1

#### Relational analysis RSZ result of RS_RSZ1_RSZ1_RSZ1_RSZ1_RSZ1_RSZ1
Status: Status.UNKNOWN
Output dim: 7, lower bound: -0.0009701, upper bound: 0.0009568
time: 0.65 seconds

### Relational analysis RSZ of RS_RSZ1_RSZ1_RSZ1_RSZ1_RSZ1_RSZ2

#### Relational analysis RSZ result of RS_RSZ1_RSZ1_RSZ1_RSZ1_RSZ1_RSZ2
Status: Status.UNKNOWN
Output dim: 7, lower bound: -0.0009701, upper bound: 0.0009566
time: 0.64 seconds

## BFS RS instance: RS_RSZ1_RSZ1_RSZ1_RSZ1_RSZ2

### Backsubstitution after applying RS history:
0: -0.0027939, -0.0016242, -0.0027939, -0.0016242, -0.0007355, 0.0007677
1: -0.0114005, -0.0084323, -0.0114005, -0.0084323, -0.0018665, 0.0019481
2: 0.0279571, 0.0297986, 0.0279571, 0.0297986, -0.0011580, 0.0012086
3: 0.0039210, 0.0073596, 0.0039210, 0.0073596, -0.0022568, 0.0021622
4: -0.0104893, -0.0074701, -0.0104893, -0.0074701, -0.0018985, 0.0019815
5: 0.0097651, 0.0109087, 0.0097651, 0.0109087, -0.0007191, 0.0007505
6: 0.0053643, 0.0097284, 0.0053643, 0.0097284, -0.0028641, 0.0027441
7: 0.9818129, 0.9848667, 0.9818129, 0.9848667, -0.0020042, 0.0019202
8: -0.0060636, -0.0027895, -0.0060636, -0.0027895, -0.0021488, 0.0020588
9: -0.0031570, -0.0009942, -0.0031570, -0.0009942, -0.0013599, 0.0014194

### Unstable ReLU Count (Linear/Conv2D Layers)
- layer_idx=0, type=LayerType.Linear, total=256, inp1_unstable=10, inp2_unstable=10, delta_unstable=251
- layer_idx=2, type=LayerType.Linear, total=256, inp1_unstable=10, inp2_unstable=10, delta_unstable=256
- layer_idx=4, type=LayerType.Linear, total=256, inp1_unstable=0, inp2_unstable=0, delta_unstable=256
- layer_idx=6, type=LayerType.Linear, total=256, inp1_unstable=0, inp2_unstable=0, delta_unstable=256
- layer_idx=8, type=LayerType.Linear, total=10, inp1_unstable=0, inp2_unstable=0, delta_unstable=10

Time for backsubstitution: 1.55 seconds

### RS candidates at layer 1
type: RSZ, layer: 1, pos: 123
type: RSZ, layer: 1, pos: 140
type: RSZ, layer: 1, pos: 167
type: RSZ, layer: 1, pos: 213
type: RSZ, layer: 1, pos: 219

Time for candidate selection: 0.20 seconds

### Candidate
type: RSZ, layer: 1, pos: 123

### Relational analysis RSZ of RS_RSZ1_RSZ1_RSZ1_RSZ1_RSZ2_RSZ1

#### Relational analysis RSZ result of RS_RSZ1_RSZ1_RSZ1_RSZ1_RSZ2_RSZ1
Status: Status.UNKNOWN
Output dim: 7, lower bound: -0.0009809, upper bound: 0.0009497
time: 0.66 seconds

### Relational analysis RSZ of RS_RSZ1_RSZ1_RSZ1_RSZ1_RSZ2_RSZ2

#### Relational analysis RSZ result of RS_RSZ1_RSZ1_RSZ1_RSZ1_RSZ2_RSZ2
Status: Status.UNKNOWN
Output dim: 7, lower bound: -0.0009804, upper bound: 0.0009500
time: 0.62 seconds

## BFS RS instance: RS_RSZ1_RSZ1_RSZ1_RSZ2_RSZ1

### Backsubstitution after applying RS history:
0: -0.0027939, -0.0016242, -0.0027939, -0.0016242, -0.0007575, 0.0007447
1: -0.0114005, -0.0084323, -0.0114005, -0.0084323, -0.0019222, 0.0018899
2: 0.0279571, 0.0297986, 0.0279571, 0.0297986, -0.0011926, 0.0011725
3: 0.0039210, 0.0073596, 0.0039210, 0.0073596, -0.0021894, 0.0022268
4: -0.0104893, -0.0074701, -0.0104893, -0.0074701, -0.0019552, 0.0019223
5: 0.0097651, 0.0109087, 0.0097651, 0.0109087, -0.0007406, 0.0007281
6: 0.0053643, 0.0097284, 0.0053643, 0.0097284, -0.0027786, 0.0028261
7: 0.9818129, 0.9848667, 0.9818129, 0.9848667, -0.0019443, 0.0019776
8: -0.0060636, -0.0027895, -0.0060636, -0.0027895, -0.0020846, 0.0021203
9: -0.0031570, -0.0009942, -0.0031570, -0.0009942, -0.0014006, 0.0013770

### Unstable ReLU Count (Linear/Conv2D Layers)
- layer_idx=0, type=LayerType.Linear, total=256, inp1_unstable=10, inp2_unstable=10, delta_unstable=251
- layer_idx=2, type=LayerType.Linear, total=256, inp1_unstable=10, inp2_unstable=10, delta_unstable=256
- layer_idx=4, type=LayerType.Linear, total=256, inp1_unstable=0, inp2_unstable=0, delta_unstable=256
- layer_idx=6, type=LayerType.Linear, total=256, inp1_unstable=0, inp2_unstable=0, delta_unstable=256
- layer_idx=8, type=LayerType.Linear, total=10, inp1_unstable=0, inp2_unstable=0, delta_unstable=10

Time for backsubstitution: 1.55 seconds

### RS candidates at layer 1
type: RSZ, layer: 1, pos: 123
type: RSZ, layer: 1, pos: 140
type: RSZ, layer: 1, pos: 167
type: RSZ, layer: 1, pos: 213
type: RSZ, layer: 1, pos: 219

Time for candidate selection: 0.15 seconds

### Candidate
type: RSZ, layer: 1, pos: 123

### Relational analysis RSZ of RS_RSZ1_RSZ1_RSZ1_RSZ2_RSZ1_RSZ1

#### Relational analysis RSZ result of RS_RSZ1_RSZ1_RSZ1_RSZ2_RSZ1_RSZ1
Status: Status.UNKNOWN
Output dim: 7, lower bound: -0.0009702, upper bound: 0.0009529
time: 0.65 seconds

### Relational analysis RSZ of RS_RSZ1_RSZ1_RSZ1_RSZ2_RSZ1_RSZ2

#### Relational analysis RSZ result of RS_RSZ1_RSZ1_RSZ1_RSZ2_RSZ1_RSZ2
Status: Status.UNKNOWN
Output dim: 7, lower bound: -0.0009702, upper bound: 0.0009530
time: 0.65 seconds

## BFS RS instance: RS_RSZ1_RSZ1_RSZ1_RSZ2_RSZ2

### Backsubstitution after applying RS history:
0: -0.0027939, -0.0016242, -0.0027939, -0.0016242, -0.0007430, 0.0007599
1: -0.0114005, -0.0084323, -0.0114005, -0.0084323, -0.0018856, 0.0019283
2: 0.0279571, 0.0297986, 0.0279571, 0.0297986, -0.0011698, 0.0011963
3: 0.0039210, 0.0073596, 0.0039210, 0.0073596, -0.0022338, 0.0021844
4: -0.0104893, -0.0074701, -0.0104893, -0.0074701, -0.0019180, 0.0019614
5: 0.0097651, 0.0109087, 0.0097651, 0.0109087, -0.0007265, 0.0007429
6: 0.0053643, 0.0097284, 0.0053643, 0.0097284, -0.0028350, 0.0027722
7: 0.9818129, 0.9848667, 0.9818129, 0.9848667, -0.0019838, 0.0019399
8: -0.0060636, -0.0027895, -0.0060636, -0.0027895, -0.0021269, 0.0020799
9: -0.0031570, -0.0009942, -0.0031570, -0.0009942, -0.0013739, 0.0014050

### Unstable ReLU Count (Linear/Conv2D Layers)
- layer_idx=0, type=LayerType.Linear, total=256, inp1_unstable=10, inp2_unstable=10, delta_unstable=251
- layer_idx=2, type=LayerType.Linear, total=256, inp1_unstable=10, inp2_unstable=10, delta_unstable=256
- layer_idx=4, type=LayerType.Linear, total=256, inp1_unstable=0, inp2_unstable=0, delta_unstable=256
- layer_idx=6, type=LayerType.Linear, total=256, inp1_unstable=0, inp2_unstable=0, delta_unstable=256
- layer_idx=8, type=LayerType.Linear, total=10, inp1_unstable=0, inp2_unstable=0, delta_unstable=10

Time for backsubstitution: 1.52 seconds

### RS candidates at layer 1
type: RSZ, layer: 1, pos: 123
type: RSZ, layer: 1, pos: 140
type: RSZ, layer: 1, pos: 167
type: RSZ, layer: 1, pos: 213
type: RSZ, layer: 1, pos: 219

Time for candidate selection: 0.16 seconds

### Candidate
type: RSZ, layer: 1, pos: 123

### Relational analysis RSZ of RS_RSZ1_RSZ1_RSZ1_RSZ2_RSZ2_RSZ1

#### Relational analysis RSZ result of RS_RSZ1_RSZ1_RSZ1_RSZ2_RSZ2_RSZ1
Status: Status.UNKNOWN
Output dim: 7, lower bound: -0.0009816, upper bound: 0.0009450
time: 0.64 seconds

### Relational analysis RSZ of RS_RSZ1_RSZ1_RSZ1_RSZ2_RSZ2_RSZ2

#### Relational analysis RSZ result of RS_RSZ1_RSZ1_RSZ1_RSZ2_RSZ2_RSZ2
Status: Status.UNKNOWN
Output dim: 7, lower bound: -0.0009808, upper bound: 0.0009450
time: 0.65 seconds

## BFS RS instance: RS_RSZ1_RSZ1_RSZ2_RSZ1_RSZ1

### Backsubstitution after applying RS history:
0: -0.0027939, -0.0016242, -0.0027939, -0.0016242, -0.0007520, 0.0007502
1: -0.0114005, -0.0084323, -0.0114005, -0.0084323, -0.0019084, 0.0019038
2: 0.0279571, 0.0297986, 0.0279571, 0.0297986, -0.0011840, 0.0011811
3: 0.0039210, 0.0073596, 0.0039210, 0.0073596, -0.0022054, 0.0022108
4: -0.0104893, -0.0074701, -0.0104893, -0.0074701, -0.0019412, 0.0019365
5: 0.0097651, 0.0109087, 0.0097651, 0.0109087, -0.0007353, 0.0007335
6: 0.0053643, 0.0097284, 0.0053643, 0.0097284, -0.0027990, 0.0028058
7: 0.9818129, 0.9848667, 0.9818129, 0.9848667, -0.0019586, 0.0019634
8: -0.0060636, -0.0027895, -0.0060636, -0.0027895, -0.0020999, 0.0021050
9: -0.0031570, -0.0009942, -0.0031570, -0.0009942, -0.0013905, 0.0013871

### Unstable ReLU Count (Linear/Conv2D Layers)
- layer_idx=0, type=LayerType.Linear, total=256, inp1_unstable=10, inp2_unstable=10, delta_unstable=251
- layer_idx=2, type=LayerType.Linear, total=256, inp1_unstable=10, inp2_unstable=10, delta_unstable=256
- layer_idx=4, type=LayerType.Linear, total=256, inp1_unstable=0, inp2_unstable=0, delta_unstable=256
- layer_idx=6, type=LayerType.Linear, total=256, inp1_unstable=0, inp2_unstable=0, delta_unstable=256
- layer_idx=8, type=LayerType.Linear, total=10, inp1_unstable=0, inp2_unstable=0, delta_unstable=10

Time for backsubstitution: 1.52 seconds

### RS candidates at layer 1
type: RSZ, layer: 1, pos: 123
type: RSZ, layer: 1, pos: 140
type: RSZ, layer: 1, pos: 167
type: RSZ, layer: 1, pos: 213
type: RSZ, layer: 1, pos: 219

Time for candidate selection: 0.16 seconds

### Candidate
type: RSZ, layer: 1, pos: 123

### Relational analysis RSZ of RS_RSZ1_RSZ1_RSZ2_RSZ1_RSZ1_RSZ1

#### Relational analysis RSZ result of RS_RSZ1_RSZ1_RSZ2_RSZ1_RSZ1_RSZ1
Status: Status.UNKNOWN
Output dim: 7, lower bound: -0.0009701, upper bound: 0.0009568
time: 0.66 seconds

### Relational analysis RSZ of RS_RSZ1_RSZ1_RSZ2_RSZ1_RSZ1_RSZ2

#### Relational analysis RSZ result of RS_RSZ1_RSZ1_RSZ2_RSZ1_RSZ1_RSZ2
Status: Status.UNKNOWN
Output dim: 7, lower bound: -0.0009701, upper bound: 0.0009566
time: 0.64 seconds

## BFS RS instance: RS_RSZ1_RSZ1_RSZ2_RSZ1_RSZ2

### Backsubstitution after applying RS history:
0: -0.0027939, -0.0016242, -0.0027939, -0.0016242, -0.0007372, 0.0007655
1: -0.0114005, -0.0084323, -0.0114005, -0.0084323, -0.0018707, 0.0019426
2: 0.0279571, 0.0297986, 0.0279571, 0.0297986, -0.0011606, 0.0012052
3: 0.0039210, 0.0073596, 0.0039210, 0.0073596, -0.0022504, 0.0021671
4: -0.0104893, -0.0074701, -0.0104893, -0.0074701, -0.0019028, 0.0019759
5: 0.0097651, 0.0109087, 0.0097651, 0.0109087, -0.0007207, 0.0007484
6: 0.0053643, 0.0097284, 0.0053643, 0.0097284, -0.0028560, 0.0027504
7: 0.9818129, 0.9848667, 0.9818129, 0.9848667, -0.0019985, 0.0019246
8: -0.0060636, -0.0027895, -0.0060636, -0.0027895, -0.0021427, 0.0020634
9: -0.0031570, -0.0009942, -0.0031570, -0.0009942, -0.0013630, 0.0014154

### Unstable ReLU Count (Linear/Conv2D Layers)
- layer_idx=0, type=LayerType.Linear, total=256, inp1_unstable=10, inp2_unstable=10, delta_unstable=251
- layer_idx=2, type=LayerType.Linear, total=256, inp1_unstable=10, inp2_unstable=10, delta_unstable=256
- layer_idx=4, type=LayerType.Linear, total=256, inp1_unstable=0, inp2_unstable=0, delta_unstable=256
- layer_idx=6, type=LayerType.Linear, total=256, inp1_unstable=0, inp2_unstable=0, delta_unstable=256
- layer_idx=8, type=LayerType.Linear, total=10, inp1_unstable=0, inp2_unstable=0, delta_unstable=10

Time for backsubstitution: 1.48 seconds

### RS candidates at layer 1
type: RSZ, layer: 1, pos: 123
type: RSZ, layer: 1, pos: 140
type: RSZ, layer: 1, pos: 167
type: RSZ, layer: 1, pos: 213
type: RSZ, layer: 1, pos: 219

Time for candidate selection: 0.15 seconds

### Candidate
type: RSZ, layer: 1, pos: 123

### Relational analysis RSZ of RS_RSZ1_RSZ1_RSZ2_RSZ1_RSZ2_RSZ1

#### Relational analysis RSZ result of RS_RSZ1_RSZ1_RSZ2_RSZ1_RSZ2_RSZ1
Status: Status.UNKNOWN
Output dim: 7, lower bound: -0.0009809, upper bound: 0.0009497
time: 0.65 seconds

### Relational analysis RSZ of RS_RSZ1_RSZ1_RSZ2_RSZ1_RSZ2_RSZ2

#### Relational analysis RSZ result of RS_RSZ1_RSZ1_RSZ2_RSZ1_RSZ2_RSZ2
Status: Status.UNKNOWN
Output dim: 7, lower bound: -0.0009804, upper bound: 0.0009500
time: 0.61 seconds

## BFS RS instance: RS_RSZ1_RSZ1_RSZ2_RSZ2_RSZ1

### Backsubstitution after applying RS history:
0: -0.0027939, -0.0016242, -0.0027939, -0.0016242, -0.0007592, 0.0007433
1: -0.0114005, -0.0084323, -0.0114005, -0.0084323, -0.0019265, 0.0018862
2: 0.0279571, 0.0297986, 0.0279571, 0.0297986, -0.0011952, 0.0011702
3: 0.0039210, 0.0073596, 0.0039210, 0.0073596, -0.0021851, 0.0022317
4: -0.0104893, -0.0074701, -0.0104893, -0.0074701, -0.0019595, 0.0019186
5: 0.0097651, 0.0109087, 0.0097651, 0.0109087, -0.0007422, 0.0007267
6: 0.0053643, 0.0097284, 0.0053643, 0.0097284, -0.0027732, 0.0028323
7: 0.9818129, 0.9848667, 0.9818129, 0.9848667, -0.0019405, 0.0019819
8: -0.0060636, -0.0027895, -0.0060636, -0.0027895, -0.0020805, 0.0021249
9: -0.0031570, -0.0009942, -0.0031570, -0.0009942, -0.0014036, 0.0013743

### Unstable ReLU Count (Linear/Conv2D Layers)
- layer_idx=0, type=LayerType.Linear, total=256, inp1_unstable=10, inp2_unstable=10, delta_unstable=251
- layer_idx=2, type=LayerType.Linear, total=256, inp1_unstable=10, inp2_unstable=10, delta_unstable=256
- layer_idx=4, type=LayerType.Linear, total=256, inp1_unstable=0, inp2_unstable=0, delta_unstable=256
- layer_idx=6, type=LayerType.Linear, total=256, inp1_unstable=0, inp2_unstable=0, delta_unstable=256
- layer_idx=8, type=LayerType.Linear, total=10, inp1_unstable=0, inp2_unstable=0, delta_unstable=10

Time for backsubstitution: 1.49 seconds

### RS candidates at layer 1
type: RSZ, layer: 1, pos: 123
type: RSZ, layer: 1, pos: 140
type: RSZ, layer: 1, pos: 167
type: RSZ, layer: 1, pos: 213
type: RSZ, layer: 1, pos: 219

Time for candidate selection: 0.19 seconds

### Candidate
type: RSZ, layer: 1, pos: 123

### Relational analysis RSZ of RS_RSZ1_RSZ1_RSZ2_RSZ2_RSZ1_RSZ1

#### Relational analysis RSZ result of RS_RSZ1_RSZ1_RSZ2_RSZ2_RSZ1_RSZ1
Status: Status.UNKNOWN
Output dim: 7, lower bound: -0.0009702, upper bound: 0.0009529
time: 0.65 seconds

### Relational analysis RSZ of RS_RSZ1_RSZ1_RSZ2_RSZ2_RSZ1_RSZ2

#### Relational analysis RSZ result of RS_RSZ1_RSZ1_RSZ2_RSZ2_RSZ1_RSZ2
Status: Status.UNKNOWN
Output dim: 7, lower bound: -0.0009702, upper bound: 0.0009530
time: 0.64 seconds

## BFS RS instance: RS_RSZ1_RSZ1_RSZ2_RSZ2_RSZ2

### Backsubstitution after applying RS history:
0: -0.0027939, -0.0016242, -0.0027939, -0.0016242, -0.0007447, 0.0007577
1: -0.0114005, -0.0084323, -0.0114005, -0.0084323, -0.0018898, 0.0019229
2: 0.0279571, 0.0297986, 0.0279571, 0.0297986, -0.0011725, 0.0011930
3: 0.0039210, 0.0073596, 0.0039210, 0.0073596, -0.0022276, 0.0021893
4: -0.0104893, -0.0074701, -0.0104893, -0.0074701, -0.0019223, 0.0019559
5: 0.0097651, 0.0109087, 0.0097651, 0.0109087, -0.0007281, 0.0007408
6: 0.0053643, 0.0097284, 0.0053643, 0.0097284, -0.0028271, 0.0027785
7: 0.9818129, 0.9848667, 0.9818129, 0.9848667, -0.0019782, 0.0019442
8: -0.0060636, -0.0027895, -0.0060636, -0.0027895, -0.0021210, 0.0020845
9: -0.0031570, -0.0009942, -0.0031570, -0.0009942, -0.0013770, 0.0014010

### Unstable ReLU Count (Linear/Conv2D Layers)
- layer_idx=0, type=LayerType.Linear, total=256, inp1_unstable=10, inp2_unstable=10, delta_unstable=251
- layer_idx=2, type=LayerType.Linear, total=256, inp1_unstable=10, inp2_unstable=10, delta_unstable=256
- layer_idx=4, type=LayerType.Linear, total=256, inp1_unstable=0, inp2_unstable=0, delta_unstable=256
- layer_idx=6, type=LayerType.Linear, total=256, inp1_unstable=0, inp2_unstable=0, delta_unstable=256
- layer_idx=8, type=LayerType.Linear, total=10, inp1_unstable=0, inp2_unstable=0, delta_unstable=10

Time for backsubstitution: 1.51 seconds

### RS candidates at layer 1
type: RSZ, layer: 1, pos: 123
type: RSZ, layer: 1, pos: 140
type: RSZ, layer: 1, pos: 167
type: RSZ, layer: 1, pos: 213
type: RSZ, layer: 1, pos: 219

Time for candidate selection: 0.16 seconds

### Candidate
type: RSZ, layer: 1, pos: 123

### Relational analysis RSZ of RS_RSZ1_RSZ1_RSZ2_RSZ2_RSZ2_RSZ1

#### Relational analysis RSZ result of RS_RSZ1_RSZ1_RSZ2_RSZ2_RSZ2_RSZ1
Status: Status.UNKNOWN
Output dim: 7, lower bound: -0.0009816, upper bound: 0.0009450
time: 0.63 seconds

### Relational analysis RSZ of RS_RSZ1_RSZ1_RSZ2_RSZ2_RSZ2_RSZ2

#### Relational analysis RSZ result of RS_RSZ1_RSZ1_RSZ2_RSZ2_RSZ2_RSZ2
Status: Status.UNKNOWN
Output dim: 7, lower bound: -0.0009808, upper bound: 0.0009450
time: 0.65 seconds

## BFS RS instance: RS_RSZ1_RSZ2_RSZ1_RSZ1_RSZ1

### Backsubstitution after applying RS history:
0: -0.0027939, -0.0016242, -0.0027939, -0.0016242, -0.0007566, 0.0007472
1: -0.0114005, -0.0084323, -0.0114005, -0.0084323, -0.0019199, 0.0018961
2: 0.0279571, 0.0297986, 0.0279571, 0.0297986, -0.0011911, 0.0011764
3: 0.0039210, 0.0073596, 0.0039210, 0.0073596, -0.0021966, 0.0022241
4: -0.0104893, -0.0074701, -0.0104893, -0.0074701, -0.0019529, 0.0019287
5: 0.0097651, 0.0109087, 0.0097651, 0.0109087, -0.0007397, 0.0007305
6: 0.0053643, 0.0097284, 0.0053643, 0.0097284, -0.0027877, 0.0028227
7: 0.9818129, 0.9848667, 0.9818129, 0.9848667, -0.0019507, 0.0019752
8: -0.0060636, -0.0027895, -0.0060636, -0.0027895, -0.0020915, 0.0021177
9: -0.0031570, -0.0009942, -0.0031570, -0.0009942, -0.0013989, 0.0013815

### Unstable ReLU Count (Linear/Conv2D Layers)
- layer_idx=0, type=LayerType.Linear, total=256, inp1_unstable=10, inp2_unstable=10, delta_unstable=251
- layer_idx=2, type=LayerType.Linear, total=256, inp1_unstable=10, inp2_unstable=10, delta_unstable=256
- layer_idx=4, type=LayerType.Linear, total=256, inp1_unstable=0, inp2_unstable=0, delta_unstable=256
- layer_idx=6, type=LayerType.Linear, total=256, inp1_unstable=0, inp2_unstable=0, delta_unstable=256
- layer_idx=8, type=LayerType.Linear, total=10, inp1_unstable=0, inp2_unstable=0, delta_unstable=10

Time for backsubstitution: 1.56 seconds

### RS candidates at layer 1
type: RSZ, layer: 1, pos: 123
type: RSZ, layer: 1, pos: 140
type: RSZ, layer: 1, pos: 167
type: RSZ, layer: 1, pos: 213
type: RSZ, layer: 1, pos: 219

Time for candidate selection: 0.16 seconds

### Candidate
type: RSZ, layer: 1, pos: 123

### Relational analysis RSZ of RS_RSZ1_RSZ2_RSZ1_RSZ1_RSZ1_RSZ1

#### Relational analysis RSZ result of RS_RSZ1_RSZ2_RSZ1_RSZ1_RSZ1_RSZ1
Status: Status.UNKNOWN
Output dim: 7, lower bound: -0.0009551, upper bound: 0.0009742
time: 0.64 seconds

### Relational analysis RSZ of RS_RSZ1_RSZ2_RSZ1_RSZ1_RSZ1_RSZ2

#### Relational analysis RSZ result of RS_RSZ1_RSZ2_RSZ1_RSZ1_RSZ1_RSZ2
Status: Status.UNKNOWN
Output dim: 7, lower bound: -0.0009551, upper bound: 0.0009754
time: 0.68 seconds

## BFS RS instance: RS_RSZ1_RSZ2_RSZ1_RSZ1_RSZ2

### Backsubstitution after applying RS history:
0: -0.0027939, -0.0016242, -0.0027939, -0.0016242, -0.0007421, 0.0007617
1: -0.0114005, -0.0084323, -0.0114005, -0.0084323, -0.0018831, 0.0019328
2: 0.0279571, 0.0297986, 0.0279571, 0.0297986, -0.0011683, 0.0011991
3: 0.0039210, 0.0073596, 0.0039210, 0.0073596, -0.0022391, 0.0021815
4: -0.0104893, -0.0074701, -0.0104893, -0.0074701, -0.0019154, 0.0019660
5: 0.0097651, 0.0109087, 0.0097651, 0.0109087, -0.0007255, 0.0007447
6: 0.0053643, 0.0097284, 0.0053643, 0.0097284, -0.0028417, 0.0027686
7: 0.9818129, 0.9848667, 0.9818129, 0.9848667, -0.0019885, 0.0019373
8: -0.0060636, -0.0027895, -0.0060636, -0.0027895, -0.0021319, 0.0020771
9: -0.0031570, -0.0009942, -0.0031570, -0.0009942, -0.0013721, 0.0014083

### Unstable ReLU Count (Linear/Conv2D Layers)
- layer_idx=0, type=LayerType.Linear, total=256, inp1_unstable=10, inp2_unstable=10, delta_unstable=251
- layer_idx=2, type=LayerType.Linear, total=256, inp1_unstable=10, inp2_unstable=10, delta_unstable=256
- layer_idx=4, type=LayerType.Linear, total=256, inp1_unstable=0, inp2_unstable=0, delta_unstable=256
- layer_idx=6, type=LayerType.Linear, total=256, inp1_unstable=0, inp2_unstable=0, delta_unstable=256
- layer_idx=8, type=LayerType.Linear, total=10, inp1_unstable=0, inp2_unstable=0, delta_unstable=10

Time for backsubstitution: 1.54 seconds

### RS candidates at layer 1
type: RSZ, layer: 1, pos: 123
type: RSZ, layer: 1, pos: 140
type: RSZ, layer: 1, pos: 167
type: RSZ, layer: 1, pos: 213
type: RSZ, layer: 1, pos: 219

Time for candidate selection: 0.16 seconds

### Candidate
type: RSZ, layer: 1, pos: 123

### Relational analysis RSZ of RS_RSZ1_RSZ2_RSZ1_RSZ1_RSZ2_RSZ1

#### Relational analysis RSZ result of RS_RSZ1_RSZ2_RSZ1_RSZ1_RSZ2_RSZ1
Status: Status.UNKNOWN
Output dim: 7, lower bound: -0.0009624, upper bound: 0.0009615
time: 0.66 seconds

### Relational analysis RSZ of RS_RSZ1_RSZ2_RSZ1_RSZ1_RSZ2_RSZ2

#### Relational analysis RSZ result of RS_RSZ1_RSZ2_RSZ1_RSZ1_RSZ2_RSZ2
Status: Status.UNKNOWN
Output dim: 7, lower bound: -0.0009625, upper bound: 0.0009620
time: 0.67 seconds

## BFS RS instance: RS_RSZ1_RSZ2_RSZ1_RSZ2_RSZ1

### Backsubstitution after applying RS history:
0: -0.0027939, -0.0016242, -0.0027939, -0.0016242, -0.0007634, 0.0007385
1: -0.0114005, -0.0084323, -0.0114005, -0.0084323, -0.0019372, 0.0018741
2: 0.0279571, 0.0297986, 0.0279571, 0.0297986, -0.0012019, 0.0011627
3: 0.0039210, 0.0073596, 0.0039210, 0.0073596, -0.0021711, 0.0022442
4: -0.0104893, -0.0074701, -0.0104893, -0.0074701, -0.0019705, 0.0019063
5: 0.0097651, 0.0109087, 0.0097651, 0.0109087, -0.0007464, 0.0007221
6: 0.0053643, 0.0097284, 0.0053643, 0.0097284, -0.0027554, 0.0028482
7: 0.9818129, 0.9848667, 0.9818129, 0.9848667, -0.0019281, 0.0019930
8: -0.0060636, -0.0027895, -0.0060636, -0.0027895, -0.0020672, 0.0021368
9: -0.0031570, -0.0009942, -0.0031570, -0.0009942, -0.0014115, 0.0013655

### Unstable ReLU Count (Linear/Conv2D Layers)
- layer_idx=0, type=LayerType.Linear, total=256, inp1_unstable=10, inp2_unstable=10, delta_unstable=251
- layer_idx=2, type=LayerType.Linear, total=256, inp1_unstable=10, inp2_unstable=10, delta_unstable=256
- layer_idx=4, type=LayerType.Linear, total=256, inp1_unstable=0, inp2_unstable=0, delta_unstable=256
- layer_idx=6, type=LayerType.Linear, total=256, inp1_unstable=0, inp2_unstable=0, delta_unstable=256
- layer_idx=8, type=LayerType.Linear, total=10, inp1_unstable=0, inp2_unstable=0, delta_unstable=10

Time for backsubstitution: 1.51 seconds

### RS candidates at layer 1
type: RSZ, layer: 1, pos: 123
type: RSZ, layer: 1, pos: 140
type: RSZ, layer: 1, pos: 167
type: RSZ, layer: 1, pos: 213
type: RSZ, layer: 1, pos: 219

Time for candidate selection: 0.16 seconds

### Candidate
type: RSZ, layer: 1, pos: 123

### Relational analysis RSZ of RS_RSZ1_RSZ2_RSZ1_RSZ2_RSZ1_RSZ1

#### Relational analysis RSZ result of RS_RSZ1_RSZ2_RSZ1_RSZ2_RSZ1_RSZ1
Status: Status.UNKNOWN
Output dim: 7, lower bound: -0.0009554, upper bound: 0.0009691
time: 0.68 seconds

### Relational analysis RSZ of RS_RSZ1_RSZ2_RSZ1_RSZ2_RSZ1_RSZ2

#### Relational analysis RSZ result of RS_RSZ1_RSZ2_RSZ1_RSZ2_RSZ1_RSZ2
Status: Status.UNKNOWN
Output dim: 7, lower bound: -0.0009554, upper bound: 0.0009710
time: 0.63 seconds

## BFS RS instance: RS_RSZ1_RSZ2_RSZ1_RSZ2_RSZ2

### Backsubstitution after applying RS history:
0: -0.0027939, -0.0016242, -0.0027939, -0.0016242, -0.0007480, 0.0007536
1: -0.0114005, -0.0084323, -0.0114005, -0.0084323, -0.0018982, 0.0019124
2: 0.0279571, 0.0297986, 0.0279571, 0.0297986, -0.0011777, 0.0011864
3: 0.0039210, 0.0073596, 0.0039210, 0.0073596, -0.0022154, 0.0021990
4: -0.0104893, -0.0074701, -0.0104893, -0.0074701, -0.0019308, 0.0019452
5: 0.0097651, 0.0109087, 0.0097651, 0.0109087, -0.0007313, 0.0007368
6: 0.0053643, 0.0097284, 0.0053643, 0.0097284, -0.0028116, 0.0027908
7: 0.9818129, 0.9848667, 0.9818129, 0.9848667, -0.0019674, 0.0019529
8: -0.0060636, -0.0027895, -0.0060636, -0.0027895, -0.0021094, 0.0020938
9: -0.0031570, -0.0009942, -0.0031570, -0.0009942, -0.0013831, 0.0013934

### Unstable ReLU Count (Linear/Conv2D Layers)
- layer_idx=0, type=LayerType.Linear, total=256, inp1_unstable=10, inp2_unstable=10, delta_unstable=251
- layer_idx=2, type=LayerType.Linear, total=256, inp1_unstable=10, inp2_unstable=10, delta_unstable=256
- layer_idx=4, type=LayerType.Linear, total=256, inp1_unstable=0, inp2_unstable=0, delta_unstable=256
- layer_idx=6, type=LayerType.Linear, total=256, inp1_unstable=0, inp2_unstable=0, delta_unstable=256
- layer_idx=8, type=LayerType.Linear, total=10, inp1_unstable=0, inp2_unstable=0, delta_unstable=10

Time for backsubstitution: 1.52 seconds

### RS candidates at layer 1
type: RSZ, layer: 1, pos: 123
type: RSZ, layer: 1, pos: 140
type: RSZ, layer: 1, pos: 167
type: RSZ, layer: 1, pos: 213
type: RSZ, layer: 1, pos: 219

Time for candidate selection: 0.19 seconds

### Candidate
type: RSZ, layer: 1, pos: 123

### Relational analysis RSZ of RS_RSZ1_RSZ2_RSZ1_RSZ2_RSZ2_RSZ1

#### Relational analysis RSZ result of RS_RSZ1_RSZ2_RSZ1_RSZ2_RSZ2_RSZ1
Status: Status.UNKNOWN
Output dim: 7, lower bound: -0.0009624, upper bound: 0.0009526
time: 0.65 seconds

### Relational analysis RSZ of RS_RSZ1_RSZ2_RSZ1_RSZ2_RSZ2_RSZ2

#### Relational analysis RSZ result of RS_RSZ1_RSZ2_RSZ1_RSZ2_RSZ2_RSZ2
Status: Status.UNKNOWN
Output dim: 7, lower bound: -0.0009626, upper bound: 0.0009532
time: 0.65 seconds

## BFS RS instance: RS_RSZ1_RSZ2_RSZ2_RSZ1_RSZ1

### Backsubstitution after applying RS history:
0: -0.0027939, -0.0016242, -0.0027939, -0.0016242, -0.0007583, 0.0007452
1: -0.0114005, -0.0084323, -0.0114005, -0.0084323, -0.0019242, 0.0018911
2: 0.0279571, 0.0297986, 0.0279571, 0.0297986, -0.0011938, 0.0011733
3: 0.0039210, 0.0073596, 0.0039210, 0.0073596, -0.0021908, 0.0022291
4: -0.0104893, -0.0074701, -0.0104893, -0.0074701, -0.0019572, 0.0019236
5: 0.0097651, 0.0109087, 0.0097651, 0.0109087, -0.0007413, 0.0007286
6: 0.0053643, 0.0097284, 0.0053643, 0.0097284, -0.0027804, 0.0028290
7: 0.9818129, 0.9848667, 0.9818129, 0.9848667, -0.0019456, 0.0019796
8: -0.0060636, -0.0027895, -0.0060636, -0.0027895, -0.0020860, 0.0021224
9: -0.0031570, -0.0009942, -0.0031570, -0.0009942, -0.0014020, 0.0013779

### Unstable ReLU Count (Linear/Conv2D Layers)
- layer_idx=0, type=LayerType.Linear, total=256, inp1_unstable=10, inp2_unstable=10, delta_unstable=251
- layer_idx=2, type=LayerType.Linear, total=256, inp1_unstable=10, inp2_unstable=10, delta_unstable=256
- layer_idx=4, type=LayerType.Linear, total=256, inp1_unstable=0, inp2_unstable=0, delta_unstable=256
- layer_idx=6, type=LayerType.Linear, total=256, inp1_unstable=0, inp2_unstable=0, delta_unstable=256
- layer_idx=8, type=LayerType.Linear, total=10, inp1_unstable=0, inp2_unstable=0, delta_unstable=10

Time for backsubstitution: 1.49 seconds

### RS candidates at layer 1
type: RSZ, layer: 1, pos: 123
type: RSZ, layer: 1, pos: 140
type: RSZ, layer: 1, pos: 167
type: RSZ, layer: 1, pos: 213
type: RSZ, layer: 1, pos: 219

Time for candidate selection: 0.15 seconds

### Candidate
type: RSZ, layer: 1, pos: 123

### Relational analysis RSZ of RS_RSZ1_RSZ2_RSZ2_RSZ1_RSZ1_RSZ1

#### Relational analysis RSZ result of RS_RSZ1_RSZ2_RSZ2_RSZ1_RSZ1_RSZ1
Status: Status.UNKNOWN
Output dim: 7, lower bound: -0.0009551, upper bound: 0.0009742
time: 0.64 seconds

### Relational analysis RSZ of RS_RSZ1_RSZ2_RSZ2_RSZ1_RSZ1_RSZ2

#### Relational analysis RSZ result of RS_RSZ1_RSZ2_RSZ2_RSZ1_RSZ1_RSZ2
Status: Status.UNKNOWN
Output dim: 7, lower bound: -0.0009551, upper bound: 0.0009754
time: 0.65 seconds

## BFS RS instance: RS_RSZ1_RSZ2_RSZ2_RSZ1_RSZ2

### Backsubstitution after applying RS history:
0: -0.0027939, -0.0016242, -0.0027939, -0.0016242, -0.0007437, 0.0007595
1: -0.0114005, -0.0084323, -0.0114005, -0.0084323, -0.0018873, 0.0019274
2: 0.0279571, 0.0297986, 0.0279571, 0.0297986, -0.0011709, 0.0011958
3: 0.0039210, 0.0073596, 0.0039210, 0.0073596, -0.0022328, 0.0021864
4: -0.0104893, -0.0074701, -0.0104893, -0.0074701, -0.0019198, 0.0019605
5: 0.0097651, 0.0109087, 0.0097651, 0.0109087, -0.0007271, 0.0007426
6: 0.0053643, 0.0097284, 0.0053643, 0.0097284, -0.0028337, 0.0027748
7: 0.9818129, 0.9848667, 0.9818129, 0.9848667, -0.0019829, 0.0019417
8: -0.0060636, -0.0027895, -0.0060636, -0.0027895, -0.0021260, 0.0020818
9: -0.0031570, -0.0009942, -0.0031570, -0.0009942, -0.0013751, 0.0014043

### Unstable ReLU Count (Linear/Conv2D Layers)
- layer_idx=0, type=LayerType.Linear, total=256, inp1_unstable=10, inp2_unstable=10, delta_unstable=251
- layer_idx=2, type=LayerType.Linear, total=256, inp1_unstable=10, inp2_unstable=10, delta_unstable=256
- layer_idx=4, type=LayerType.Linear, total=256, inp1_unstable=0, inp2_unstable=0, delta_unstable=256
- layer_idx=6, type=LayerType.Linear, total=256, inp1_unstable=0, inp2_unstable=0, delta_unstable=256
- layer_idx=8, type=LayerType.Linear, total=10, inp1_unstable=0, inp2_unstable=0, delta_unstable=10

Time for backsubstitution: 1.54 seconds

### RS candidates at layer 1
type: RSZ, layer: 1, pos: 123
type: RSZ, layer: 1, pos: 140
type: RSZ, layer: 1, pos: 167
type: RSZ, layer: 1, pos: 213
type: RSZ, layer: 1, pos: 219

Time for candidate selection: 0.16 seconds

### Candidate
type: RSZ, layer: 1, pos: 123

### Relational analysis RSZ of RS_RSZ1_RSZ2_RSZ2_RSZ1_RSZ2_RSZ1

#### Relational analysis RSZ result of RS_RSZ1_RSZ2_RSZ2_RSZ1_RSZ2_RSZ1
Status: Status.UNKNOWN
Output dim: 7, lower bound: -0.0009624, upper bound: 0.0009615
time: 0.65 seconds

### Relational analysis RSZ of RS_RSZ1_RSZ2_RSZ2_RSZ1_RSZ2_RSZ2

#### Relational analysis RSZ result of RS_RSZ1_RSZ2_RSZ2_RSZ1_RSZ2_RSZ2
Status: Status.UNKNOWN
Output dim: 7, lower bound: -0.0009625, upper bound: 0.0009620
time: 0.66 seconds

## BFS RS instance: RS_RSZ1_RSZ2_RSZ2_RSZ2_RSZ1

### Backsubstitution after applying RS history:
0: -0.0027939, -0.0016242, -0.0027939, -0.0016242, -0.0007651, 0.0007367
1: -0.0114005, -0.0084323, -0.0114005, -0.0084323, -0.0019415, 0.0018694
2: 0.0279571, 0.0297986, 0.0279571, 0.0297986, -0.0012045, 0.0011598
3: 0.0039210, 0.0073596, 0.0039210, 0.0073596, -0.0021656, 0.0022491
4: -0.0104893, -0.0074701, -0.0104893, -0.0074701, -0.0019748, 0.0019015
5: 0.0097651, 0.0109087, 0.0097651, 0.0109087, -0.0007480, 0.0007202
6: 0.0053643, 0.0097284, 0.0053643, 0.0097284, -0.0027484, 0.0028544
7: 0.9818129, 0.9848667, 0.9818129, 0.9848667, -0.0019232, 0.0019974
8: -0.0060636, -0.0027895, -0.0060636, -0.0027895, -0.0020620, 0.0021415
9: -0.0031570, -0.0009942, -0.0031570, -0.0009942, -0.0014146, 0.0013620

### Unstable ReLU Count (Linear/Conv2D Layers)
- layer_idx=0, type=LayerType.Linear, total=256, inp1_unstable=10, inp2_unstable=10, delta_unstable=251
- layer_idx=2, type=LayerType.Linear, total=256, inp1_unstable=10, inp2_unstable=10, delta_unstable=256
- layer_idx=4, type=LayerType.Linear, total=256, inp1_unstable=0, inp2_unstable=0, delta_unstable=256
- layer_idx=6, type=LayerType.Linear, total=256, inp1_unstable=0, inp2_unstable=0, delta_unstable=256
- layer_idx=8, type=LayerType.Linear, total=10, inp1_unstable=0, inp2_unstable=0, delta_unstable=10

Time for backsubstitution: 1.56 seconds

### RS candidates at layer 1
type: RSZ, layer: 1, pos: 123
type: RSZ, layer: 1, pos: 140
type: RSZ, layer: 1, pos: 167
type: RSZ, layer: 1, pos: 213
type: RSZ, layer: 1, pos: 219

Time for candidate selection: 0.16 seconds

### Candidate
type: RSZ, layer: 1, pos: 123

### Relational analysis RSZ of RS_RSZ1_RSZ2_RSZ2_RSZ2_RSZ1_RSZ1

#### Relational analysis RSZ result of RS_RSZ1_RSZ2_RSZ2_RSZ2_RSZ1_RSZ1
Status: Status.UNKNOWN
Output dim: 7, lower bound: -0.0009554, upper bound: 0.0009691
time: 0.63 seconds

### Relational analysis RSZ of RS_RSZ1_RSZ2_RSZ2_RSZ2_RSZ1_RSZ2

#### Relational analysis RSZ result of RS_RSZ1_RSZ2_RSZ2_RSZ2_RSZ1_RSZ2
Status: Status.UNKNOWN
Output dim: 7, lower bound: -0.0009554, upper bound: 0.0009710
time: 0.63 seconds

## BFS RS instance: RS_RSZ1_RSZ2_RSZ2_RSZ2_RSZ2

### Backsubstitution after applying RS history:
0: -0.0027939, -0.0016242, -0.0027939, -0.0016242, -0.0007497, 0.0007515
1: -0.0114005, -0.0084323, -0.0114005, -0.0084323, -0.0019025, 0.0019070
2: 0.0279571, 0.0297986, 0.0279571, 0.0297986, -0.0011803, 0.0011831
3: 0.0039210, 0.0073596, 0.0039210, 0.0073596, -0.0022091, 0.0022039
4: -0.0104893, -0.0074701, -0.0104893, -0.0074701, -0.0019351, 0.0019397
5: 0.0097651, 0.0109087, 0.0097651, 0.0109087, -0.0007330, 0.0007347
6: 0.0053643, 0.0097284, 0.0053643, 0.0097284, -0.0028037, 0.0027970
7: 0.9818129, 0.9848667, 0.9818129, 0.9848667, -0.0019619, 0.0019572
8: -0.0060636, -0.0027895, -0.0060636, -0.0027895, -0.0021034, 0.0020985
9: -0.0031570, -0.0009942, -0.0031570, -0.0009942, -0.0013862, 0.0013894

### Unstable ReLU Count (Linear/Conv2D Layers)
- layer_idx=0, type=LayerType.Linear, total=256, inp1_unstable=10, inp2_unstable=10, delta_unstable=251
- layer_idx=2, type=LayerType.Linear, total=256, inp1_unstable=10, inp2_unstable=10, delta_unstable=256
- layer_idx=4, type=LayerType.Linear, total=256, inp1_unstable=0, inp2_unstable=0, delta_unstable=256
- layer_idx=6, type=LayerType.Linear, total=256, inp1_unstable=0, inp2_unstable=0, delta_unstable=256
- layer_idx=8, type=LayerType.Linear, total=10, inp1_unstable=0, inp2_unstable=0, delta_unstable=10

Time for backsubstitution: 1.51 seconds

### RS candidates at layer 1
type: RSZ, layer: 1, pos: 123
type: RSZ, layer: 1, pos: 140
type: RSZ, layer: 1, pos: 167
type: RSZ, layer: 1, pos: 213
type: RSZ, layer: 1, pos: 219

Time for candidate selection: 0.15 seconds

### Candidate
type: RSZ, layer: 1, pos: 123

### Relational analysis RSZ of RS_RSZ1_RSZ2_RSZ2_RSZ2_RSZ2_RSZ1

#### Relational analysis RSZ result of RS_RSZ1_RSZ2_RSZ2_RSZ2_RSZ2_RSZ1
Status: Status.UNKNOWN
Output dim: 7, lower bound: -0.0009624, upper bound: 0.0009526
time: 0.64 seconds

### Relational analysis RSZ of RS_RSZ1_RSZ2_RSZ2_RSZ2_RSZ2_RSZ2

#### Relational analysis RSZ result of RS_RSZ1_RSZ2_RSZ2_RSZ2_RSZ2_RSZ2
Status: Status.UNKNOWN
Output dim: 7, lower bound: -0.0009626, upper bound: 0.0009532
time: 0.63 seconds

## BFS RS instance: RS_RSZ2_RSZ1_RSZ1_RSZ1_RSZ1

### Backsubstitution after applying RS history:
0: -0.0027939, -0.0016242, -0.0027939, -0.0016242, -0.0007515, 0.0007502
1: -0.0114005, -0.0084323, -0.0114005, -0.0084323, -0.0019070, 0.0019036
2: 0.0279571, 0.0297986, 0.0279571, 0.0297986, -0.0011831, 0.0011810
3: 0.0039210, 0.0073596, 0.0039210, 0.0073596, -0.0022053, 0.0022091
4: -0.0104893, -0.0074701, -0.0104893, -0.0074701, -0.0019397, 0.0019363
5: 0.0097651, 0.0109087, 0.0097651, 0.0109087, -0.0007347, 0.0007334
6: 0.0053643, 0.0097284, 0.0053643, 0.0097284, -0.0027987, 0.0028037
7: 0.9818129, 0.9848667, 0.9818129, 0.9848667, -0.0019584, 0.0019619
8: -0.0060636, -0.0027895, -0.0060636, -0.0027895, -0.0020997, 0.0021034
9: -0.0031570, -0.0009942, -0.0031570, -0.0009942, -0.0013894, 0.0013870

### Unstable ReLU Count (Linear/Conv2D Layers)
- layer_idx=0, type=LayerType.Linear, total=256, inp1_unstable=10, inp2_unstable=10, delta_unstable=251
- layer_idx=2, type=LayerType.Linear, total=256, inp1_unstable=10, inp2_unstable=10, delta_unstable=256
- layer_idx=4, type=LayerType.Linear, total=256, inp1_unstable=0, inp2_unstable=0, delta_unstable=256
- layer_idx=6, type=LayerType.Linear, total=256, inp1_unstable=0, inp2_unstable=0, delta_unstable=256
- layer_idx=8, type=LayerType.Linear, total=10, inp1_unstable=0, inp2_unstable=0, delta_unstable=10

Time for backsubstitution: 1.49 seconds

### RS candidates at layer 1
type: RSZ, layer: 1, pos: 123
type: RSZ, layer: 1, pos: 140
type: RSZ, layer: 1, pos: 167
type: RSZ, layer: 1, pos: 213
type: RSZ, layer: 1, pos: 219

Time for candidate selection: 0.17 seconds

### Candidate
type: RSZ, layer: 1, pos: 123

### Relational analysis RSZ of RS_RSZ2_RSZ1_RSZ1_RSZ1_RSZ1_RSZ1

#### Relational analysis RSZ result of RS_RSZ2_RSZ1_RSZ1_RSZ1_RSZ1_RSZ1
Status: Status.UNKNOWN
Output dim: 7, lower bound: -0.0009532, upper bound: 0.0009626
time: 0.65 seconds

### Relational analysis RSZ of RS_RSZ2_RSZ1_RSZ1_RSZ1_RSZ1_RSZ2

#### Relational analysis RSZ result of RS_RSZ2_RSZ1_RSZ1_RSZ1_RSZ1_RSZ2
Status: Status.UNKNOWN
Output dim: 7, lower bound: -0.0009526, upper bound: 0.0009624
time: 0.65 seconds

## BFS RS instance: RS_RSZ2_RSZ1_RSZ1_RSZ1_RSZ2

### Backsubstitution after applying RS history:
0: -0.0027939, -0.0016242, -0.0027939, -0.0016242, -0.0007367, 0.0007656
1: -0.0114005, -0.0084323, -0.0114005, -0.0084323, -0.0018694, 0.0019429
2: 0.0279571, 0.0297986, 0.0279571, 0.0297986, -0.0011598, 0.0012054
3: 0.0039210, 0.0073596, 0.0039210, 0.0073596, -0.0022507, 0.0021656
4: -0.0104893, -0.0074701, -0.0104893, -0.0074701, -0.0019015, 0.0019762
5: 0.0097651, 0.0109087, 0.0097651, 0.0109087, -0.0007202, 0.0007485
6: 0.0053643, 0.0097284, 0.0053643, 0.0097284, -0.0028564, 0.0027484
7: 0.9818129, 0.9848667, 0.9818129, 0.9848667, -0.0019988, 0.0019232
8: -0.0060636, -0.0027895, -0.0060636, -0.0027895, -0.0021430, 0.0020620
9: -0.0031570, -0.0009942, -0.0031570, -0.0009942, -0.0013620, 0.0014156

### Unstable ReLU Count (Linear/Conv2D Layers)
- layer_idx=0, type=LayerType.Linear, total=256, inp1_unstable=10, inp2_unstable=10, delta_unstable=251
- layer_idx=2, type=LayerType.Linear, total=256, inp1_unstable=10, inp2_unstable=10, delta_unstable=256
- layer_idx=4, type=LayerType.Linear, total=256, inp1_unstable=0, inp2_unstable=0, delta_unstable=256
- layer_idx=6, type=LayerType.Linear, total=256, inp1_unstable=0, inp2_unstable=0, delta_unstable=256
- layer_idx=8, type=LayerType.Linear, total=10, inp1_unstable=0, inp2_unstable=0, delta_unstable=10

Time for backsubstitution: 1.50 seconds

### RS candidates at layer 1
type: RSZ, layer: 1, pos: 123
type: RSZ, layer: 1, pos: 140
type: RSZ, layer: 1, pos: 167
type: RSZ, layer: 1, pos: 213
type: RSZ, layer: 1, pos: 219

Time for candidate selection: 0.16 seconds

### Candidate
type: RSZ, layer: 1, pos: 123

### Relational analysis RSZ of RS_RSZ2_RSZ1_RSZ1_RSZ1_RSZ2_RSZ1

#### Relational analysis RSZ result of RS_RSZ2_RSZ1_RSZ1_RSZ1_RSZ2_RSZ1
Status: Status.UNKNOWN
Output dim: 7, lower bound: -0.0009710, upper bound: 0.0009554
time: 0.65 seconds

### Relational analysis RSZ of RS_RSZ2_RSZ1_RSZ1_RSZ1_RSZ2_RSZ2

#### Relational analysis RSZ result of RS_RSZ2_RSZ1_RSZ1_RSZ1_RSZ2_RSZ2
Status: Status.UNKNOWN
Output dim: 7, lower bound: -0.0009691, upper bound: 0.0009554
time: 0.66 seconds

## BFS RS instance: RS_RSZ2_RSZ1_RSZ1_RSZ2_RSZ1

### Backsubstitution after applying RS history:
0: -0.0027939, -0.0016242, -0.0027939, -0.0016242, -0.0007595, 0.0007436
1: -0.0114005, -0.0084323, -0.0114005, -0.0084323, -0.0019274, 0.0018870
2: 0.0279571, 0.0297986, 0.0279571, 0.0297986, -0.0011958, 0.0011707
3: 0.0039210, 0.0073596, 0.0039210, 0.0073596, -0.0021860, 0.0022328
4: -0.0104893, -0.0074701, -0.0104893, -0.0074701, -0.0019605, 0.0019194
5: 0.0097651, 0.0109087, 0.0097651, 0.0109087, -0.0007426, 0.0007270
6: 0.0053643, 0.0097284, 0.0053643, 0.0097284, -0.0027743, 0.0028337
7: 0.9818129, 0.9848667, 0.9818129, 0.9848667, -0.0019413, 0.0019829
8: -0.0060636, -0.0027895, -0.0060636, -0.0027895, -0.0020814, 0.0021260
9: -0.0031570, -0.0009942, -0.0031570, -0.0009942, -0.0014043, 0.0013749

### Unstable ReLU Count (Linear/Conv2D Layers)
- layer_idx=0, type=LayerType.Linear, total=256, inp1_unstable=10, inp2_unstable=10, delta_unstable=251
- layer_idx=2, type=LayerType.Linear, total=256, inp1_unstable=10, inp2_unstable=10, delta_unstable=256
- layer_idx=4, type=LayerType.Linear, total=256, inp1_unstable=0, inp2_unstable=0, delta_unstable=256
- layer_idx=6, type=LayerType.Linear, total=256, inp1_unstable=0, inp2_unstable=0, delta_unstable=256
- layer_idx=8, type=LayerType.Linear, total=10, inp1_unstable=0, inp2_unstable=0, delta_unstable=10

Time for backsubstitution: 1.52 seconds

### RS candidates at layer 1
type: RSZ, layer: 1, pos: 123
type: RSZ, layer: 1, pos: 140
type: RSZ, layer: 1, pos: 167
type: RSZ, layer: 1, pos: 213
type: RSZ, layer: 1, pos: 219

Time for candidate selection: 0.16 seconds

### Candidate
type: RSZ, layer: 1, pos: 123

### Relational analysis RSZ of RS_RSZ2_RSZ1_RSZ1_RSZ2_RSZ1_RSZ1

#### Relational analysis RSZ result of RS_RSZ2_RSZ1_RSZ1_RSZ2_RSZ1_RSZ1
Status: Status.UNKNOWN
Output dim: 7, lower bound: -0.0009620, upper bound: 0.0009625
time: 0.64 seconds

### Relational analysis RSZ of RS_RSZ2_RSZ1_RSZ1_RSZ2_RSZ1_RSZ2

#### Relational analysis RSZ result of RS_RSZ2_RSZ1_RSZ1_RSZ2_RSZ1_RSZ2
Status: Status.UNKNOWN
Output dim: 7, lower bound: -0.0009615, upper bound: 0.0009624
time: 0.66 seconds

## BFS RS instance: RS_RSZ2_RSZ1_RSZ1_RSZ2_RSZ2

### Backsubstitution after applying RS history:
0: -0.0027939, -0.0016242, -0.0027939, -0.0016242, -0.0007452, 0.0007588
1: -0.0114005, -0.0084323, -0.0114005, -0.0084323, -0.0018911, 0.0019254
2: 0.0279571, 0.0297986, 0.0279571, 0.0297986, -0.0011733, 0.0011946
3: 0.0039210, 0.0073596, 0.0039210, 0.0073596, -0.0022305, 0.0021908
4: -0.0104893, -0.0074701, -0.0104893, -0.0074701, -0.0019236, 0.0019585
5: 0.0097651, 0.0109087, 0.0097651, 0.0109087, -0.0007286, 0.0007418
6: 0.0053643, 0.0097284, 0.0053643, 0.0097284, -0.0028308, 0.0027804
7: 0.9818129, 0.9848667, 0.9818129, 0.9848667, -0.0019809, 0.0019456
8: -0.0060636, -0.0027895, -0.0060636, -0.0027895, -0.0021238, 0.0020860
9: -0.0031570, -0.0009942, -0.0031570, -0.0009942, -0.0013779, 0.0014029

### Unstable ReLU Count (Linear/Conv2D Layers)
- layer_idx=0, type=LayerType.Linear, total=256, inp1_unstable=10, inp2_unstable=10, delta_unstable=251
- layer_idx=2, type=LayerType.Linear, total=256, inp1_unstable=10, inp2_unstable=10, delta_unstable=256
- layer_idx=4, type=LayerType.Linear, total=256, inp1_unstable=0, inp2_unstable=0, delta_unstable=256
- layer_idx=6, type=LayerType.Linear, total=256, inp1_unstable=0, inp2_unstable=0, delta_unstable=256
- layer_idx=8, type=LayerType.Linear, total=10, inp1_unstable=0, inp2_unstable=0, delta_unstable=10

Time for backsubstitution: 1.71 seconds

### RS candidates at layer 1
type: RSZ, layer: 1, pos: 123
type: RSZ, layer: 1, pos: 140
type: RSZ, layer: 1, pos: 167
type: RSZ, layer: 1, pos: 213
type: RSZ, layer: 1, pos: 219

Time for candidate selection: 0.20 seconds

### Candidate
type: RSZ, layer: 1, pos: 123

### Relational analysis RSZ of RS_RSZ2_RSZ1_RSZ1_RSZ2_RSZ2_RSZ1

#### Relational analysis RSZ result of RS_RSZ2_RSZ1_RSZ1_RSZ2_RSZ2_RSZ1
Status: Status.UNKNOWN
Output dim: 7, lower bound: -0.0009754, upper bound: 0.0009551
time: 0.67 seconds

### Relational analysis RSZ of RS_RSZ2_RSZ1_RSZ1_RSZ2_RSZ2_RSZ2

#### Relational analysis RSZ result of RS_RSZ2_RSZ1_RSZ1_RSZ2_RSZ2_RSZ2
Status: Status.UNKNOWN
Output dim: 7, lower bound: -0.0009742, upper bound: 0.0009551
time: 0.68 seconds

## BFS RS instance: RS_RSZ2_RSZ1_RSZ2_RSZ1_RSZ1

### Backsubstitution after applying RS history:
0: -0.0027939, -0.0016242, -0.0027939, -0.0016242, -0.0007531, 0.0007480
1: -0.0114005, -0.0084323, -0.0114005, -0.0084323, -0.0019112, 0.0018982
2: 0.0279571, 0.0297986, 0.0279571, 0.0297986, -0.0011857, 0.0011777
3: 0.0039210, 0.0073596, 0.0039210, 0.0073596, -0.0021990, 0.0022140
4: -0.0104893, -0.0074701, -0.0104893, -0.0074701, -0.0019440, 0.0019308
5: 0.0097651, 0.0109087, 0.0097651, 0.0109087, -0.0007363, 0.0007313
6: 0.0053643, 0.0097284, 0.0053643, 0.0097284, -0.0027908, 0.0028099
7: 0.9818129, 0.9848667, 0.9818129, 0.9848667, -0.0019529, 0.0019662
8: -0.0060636, -0.0027895, -0.0060636, -0.0027895, -0.0020938, 0.0021081
9: -0.0031570, -0.0009942, -0.0031570, -0.0009942, -0.0013925, 0.0013831

### Unstable ReLU Count (Linear/Conv2D Layers)
- layer_idx=0, type=LayerType.Linear, total=256, inp1_unstable=10, inp2_unstable=10, delta_unstable=251
- layer_idx=2, type=LayerType.Linear, total=256, inp1_unstable=10, inp2_unstable=10, delta_unstable=256
- layer_idx=4, type=LayerType.Linear, total=256, inp1_unstable=0, inp2_unstable=0, delta_unstable=256
- layer_idx=6, type=LayerType.Linear, total=256, inp1_unstable=0, inp2_unstable=0, delta_unstable=256
- layer_idx=8, type=LayerType.Linear, total=10, inp1_unstable=0, inp2_unstable=0, delta_unstable=10

Time for backsubstitution: 1.50 seconds

### RS candidates at layer 1
type: RSZ, layer: 1, pos: 123
type: RSZ, layer: 1, pos: 140
type: RSZ, layer: 1, pos: 167
type: RSZ, layer: 1, pos: 213
type: RSZ, layer: 1, pos: 219

Time for candidate selection: 0.16 seconds

### Candidate
type: RSZ, layer: 1, pos: 123

### Relational analysis RSZ of RS_RSZ2_RSZ1_RSZ2_RSZ1_RSZ1_RSZ1

#### Relational analysis RSZ result of RS_RSZ2_RSZ1_RSZ2_RSZ1_RSZ1_RSZ1
Status: Status.UNKNOWN
Output dim: 7, lower bound: -0.0009532, upper bound: 0.0009626
time: 0.65 seconds

### Relational analysis RSZ of RS_RSZ2_RSZ1_RSZ2_RSZ1_RSZ1_RSZ2

#### Relational analysis RSZ result of RS_RSZ2_RSZ1_RSZ2_RSZ1_RSZ1_RSZ2
Status: Status.UNKNOWN
Output dim: 7, lower bound: -0.0009526, upper bound: 0.0009624
time: 0.64 seconds

## BFS RS instance: RS_RSZ2_RSZ1_RSZ2_RSZ1_RSZ2

### Backsubstitution after applying RS history:
0: -0.0027939, -0.0016242, -0.0027939, -0.0016242, -0.0007383, 0.0007634
1: -0.0114005, -0.0084323, -0.0114005, -0.0084323, -0.0018736, 0.0019372
2: 0.0279571, 0.0297986, 0.0279571, 0.0297986, -0.0011624, 0.0012019
3: 0.0039210, 0.0073596, 0.0039210, 0.0073596, -0.0022442, 0.0021705
4: -0.0104893, -0.0074701, -0.0104893, -0.0074701, -0.0019058, 0.0019705
5: 0.0097651, 0.0109087, 0.0097651, 0.0109087, -0.0007219, 0.0007464
6: 0.0053643, 0.0097284, 0.0053643, 0.0097284, -0.0028482, 0.0027546
7: 0.9818129, 0.9848667, 0.9818129, 0.9848667, -0.0019930, 0.0019275
8: -0.0060636, -0.0027895, -0.0060636, -0.0027895, -0.0021368, 0.0020666
9: -0.0031570, -0.0009942, -0.0031570, -0.0009942, -0.0013651, 0.0014115

### Unstable ReLU Count (Linear/Conv2D Layers)
- layer_idx=0, type=LayerType.Linear, total=256, inp1_unstable=10, inp2_unstable=10, delta_unstable=251
- layer_idx=2, type=LayerType.Linear, total=256, inp1_unstable=10, inp2_unstable=10, delta_unstable=256
- layer_idx=4, type=LayerType.Linear, total=256, inp1_unstable=0, inp2_unstable=0, delta_unstable=256
- layer_idx=6, type=LayerType.Linear, total=256, inp1_unstable=0, inp2_unstable=0, delta_unstable=256
- layer_idx=8, type=LayerType.Linear, total=10, inp1_unstable=0, inp2_unstable=0, delta_unstable=10

Time for backsubstitution: 1.55 seconds

### RS candidates at layer 1
type: RSZ, layer: 1, pos: 123
type: RSZ, layer: 1, pos: 140
type: RSZ, layer: 1, pos: 167
type: RSZ, layer: 1, pos: 213
type: RSZ, layer: 1, pos: 219

Time for candidate selection: 0.20 seconds

### Candidate
type: RSZ, layer: 1, pos: 123

### Relational analysis RSZ of RS_RSZ2_RSZ1_RSZ2_RSZ1_RSZ2_RSZ1

#### Relational analysis RSZ result of RS_RSZ2_RSZ1_RSZ2_RSZ1_RSZ2_RSZ1
Status: Status.UNKNOWN
Output dim: 7, lower bound: -0.0009710, upper bound: 0.0009554
time: 0.64 seconds

### Relational analysis RSZ of RS_RSZ2_RSZ1_RSZ2_RSZ1_RSZ2_RSZ2

#### Relational analysis RSZ result of RS_RSZ2_RSZ1_RSZ2_RSZ1_RSZ2_RSZ2
Status: Status.UNKNOWN
Output dim: 7, lower bound: -0.0009691, upper bound: 0.0009554
time: 0.66 seconds

## BFS RS instance: RS_RSZ2_RSZ1_RSZ2_RSZ2_RSZ1

### Backsubstitution after applying RS history:
0: -0.0027939, -0.0016242, -0.0027939, -0.0016242, -0.0007612, 0.0007421
1: -0.0114005, -0.0084323, -0.0114005, -0.0084323, -0.0019316, 0.0018831
2: 0.0279571, 0.0297986, 0.0279571, 0.0297986, -0.0011984, 0.0011683
3: 0.0039210, 0.0073596, 0.0039210, 0.0073596, -0.0021815, 0.0022377
4: -0.0104893, -0.0074701, -0.0104893, -0.0074701, -0.0019648, 0.0019154
5: 0.0097651, 0.0109087, 0.0097651, 0.0109087, -0.0007442, 0.0007255
6: 0.0053643, 0.0097284, 0.0053643, 0.0097284, -0.0027686, 0.0028399
7: 0.9818129, 0.9848667, 0.9818129, 0.9848667, -0.0019373, 0.0019872
8: -0.0060636, -0.0027895, -0.0060636, -0.0027895, -0.0020771, 0.0021306
9: -0.0031570, -0.0009942, -0.0031570, -0.0009942, -0.0014074, 0.0013721

### Unstable ReLU Count (Linear/Conv2D Layers)
- layer_idx=0, type=LayerType.Linear, total=256, inp1_unstable=10, inp2_unstable=10, delta_unstable=251
- layer_idx=2, type=LayerType.Linear, total=256, inp1_unstable=10, inp2_unstable=10, delta_unstable=256
- layer_idx=4, type=LayerType.Linear, total=256, inp1_unstable=0, inp2_unstable=0, delta_unstable=256
- layer_idx=6, type=LayerType.Linear, total=256, inp1_unstable=0, inp2_unstable=0, delta_unstable=256
- layer_idx=8, type=LayerType.Linear, total=10, inp1_unstable=0, inp2_unstable=0, delta_unstable=10

Time for backsubstitution: 1.54 seconds

### RS candidates at layer 1
type: RSZ, layer: 1, pos: 123
type: RSZ, layer: 1, pos: 140
type: RSZ, layer: 1, pos: 167
type: RSZ, layer: 1, pos: 213
type: RSZ, layer: 1, pos: 219

Time for candidate selection: 0.15 seconds

### Candidate
type: RSZ, layer: 1, pos: 123

### Relational analysis RSZ of RS_RSZ2_RSZ1_RSZ2_RSZ2_RSZ1_RSZ1

#### Relational analysis RSZ result of RS_RSZ2_RSZ1_RSZ2_RSZ2_RSZ1_RSZ1
Status: Status.UNKNOWN
Output dim: 7, lower bound: -0.0009620, upper bound: 0.0009625
time: 0.65 seconds

### Relational analysis RSZ of RS_RSZ2_RSZ1_RSZ2_RSZ2_RSZ1_RSZ2

#### Relational analysis RSZ result of RS_RSZ2_RSZ1_RSZ2_RSZ2_RSZ1_RSZ2
Status: Status.UNKNOWN
Output dim: 7, lower bound: -0.0009615, upper bound: 0.0009624
time: 0.63 seconds

## BFS RS instance: RS_RSZ2_RSZ1_RSZ2_RSZ2_RSZ2

### Backsubstitution after applying RS history:
0: -0.0027939, -0.0016242, -0.0027939, -0.0016242, -0.0007469, 0.0007566
1: -0.0114005, -0.0084323, -0.0114005, -0.0084323, -0.0018954, 0.0019199
2: 0.0279571, 0.0297986, 0.0279571, 0.0297986, -0.0011759, 0.0011911
3: 0.0039210, 0.0073596, 0.0039210, 0.0073596, -0.0022241, 0.0021957
4: -0.0104893, -0.0074701, -0.0104893, -0.0074701, -0.0019279, 0.0019529
5: 0.0097651, 0.0109087, 0.0097651, 0.0109087, -0.0007302, 0.0007397
6: 0.0053643, 0.0097284, 0.0053643, 0.0097284, -0.0028227, 0.0027867
7: 0.9818129, 0.9848667, 0.9818129, 0.9848667, -0.0019752, 0.0019500
8: -0.0060636, -0.0027895, -0.0060636, -0.0027895, -0.0021177, 0.0020907
9: -0.0031570, -0.0009942, -0.0031570, -0.0009942, -0.0013810, 0.0013989

### Unstable ReLU Count (Linear/Conv2D Layers)
- layer_idx=0, type=LayerType.Linear, total=256, inp1_unstable=10, inp2_unstable=10, delta_unstable=251
- layer_idx=2, type=LayerType.Linear, total=256, inp1_unstable=10, inp2_unstable=10, delta_unstable=256
- layer_idx=4, type=LayerType.Linear, total=256, inp1_unstable=0, inp2_unstable=0, delta_unstable=256
- layer_idx=6, type=LayerType.Linear, total=256, inp1_unstable=0, inp2_unstable=0, delta_unstable=256
- layer_idx=8, type=LayerType.Linear, total=10, inp1_unstable=0, inp2_unstable=0, delta_unstable=10

Time for backsubstitution: 1.51 seconds

### RS candidates at layer 1
type: RSZ, layer: 1, pos: 123
type: RSZ, layer: 1, pos: 140
type: RSZ, layer: 1, pos: 167
type: RSZ, layer: 1, pos: 213
type: RSZ, layer: 1, pos: 219

Time for candidate selection: 0.15 seconds

### Candidate
type: RSZ, layer: 1, pos: 123

### Relational analysis RSZ of RS_RSZ2_RSZ1_RSZ2_RSZ2_RSZ2_RSZ1

#### Relational analysis RSZ result of RS_RSZ2_RSZ1_RSZ2_RSZ2_RSZ2_RSZ1
Status: Status.UNKNOWN
Output dim: 7, lower bound: -0.0009754, upper bound: 0.0009551
time: 0.61 seconds

### Relational analysis RSZ of RS_RSZ2_RSZ1_RSZ2_RSZ2_RSZ2_RSZ2

#### Relational analysis RSZ result of RS_RSZ2_RSZ1_RSZ2_RSZ2_RSZ2_RSZ2
Status: Status.UNKNOWN
Output dim: 7, lower bound: -0.0009742, upper bound: 0.0009551
time: 0.64 seconds

## BFS RS instance: RS_RSZ2_RSZ2_RSZ1_RSZ1_RSZ1

### Backsubstitution after applying RS history:
0: -0.0027939, -0.0016242, -0.0027939, -0.0016242, -0.0007577, 0.0007450
1: -0.0114005, -0.0084323, -0.0114005, -0.0084323, -0.0019229, 0.0018905
2: 0.0279571, 0.0297986, 0.0279571, 0.0297986, -0.0011930, 0.0011729
3: 0.0039210, 0.0073596, 0.0039210, 0.0073596, -0.0021901, 0.0022276
4: -0.0104893, -0.0074701, -0.0104893, -0.0074701, -0.0019559, 0.0019230
5: 0.0097651, 0.0109087, 0.0097651, 0.0109087, -0.0007408, 0.0007284
6: 0.0053643, 0.0097284, 0.0053643, 0.0097284, -0.0027795, 0.0028271
7: 0.9818129, 0.9848667, 0.9818129, 0.9848667, -0.0019449, 0.0019782
8: -0.0060636, -0.0027895, -0.0060636, -0.0027895, -0.0020853, 0.0021210
9: -0.0031570, -0.0009942, -0.0031570, -0.0009942, -0.0014010, 0.0013774

### Unstable ReLU Count (Linear/Conv2D Layers)
- layer_idx=0, type=LayerType.Linear, total=256, inp1_unstable=10, inp2_unstable=10, delta_unstable=251
- layer_idx=2, type=LayerType.Linear, total=256, inp1_unstable=10, inp2_unstable=10, delta_unstable=256
- layer_idx=4, type=LayerType.Linear, total=256, inp1_unstable=0, inp2_unstable=0, delta_unstable=256
- layer_idx=6, type=LayerType.Linear, total=256, inp1_unstable=0, inp2_unstable=0, delta_unstable=256
- layer_idx=8, type=LayerType.Linear, total=10, inp1_unstable=0, inp2_unstable=0, delta_unstable=10

Time for backsubstitution: 1.49 seconds

### RS candidates at layer 1
type: RSZ, layer: 1, pos: 123
type: RSZ, layer: 1, pos: 140
type: RSZ, layer: 1, pos: 167
type: RSZ, layer: 1, pos: 213
type: RSZ, layer: 1, pos: 219

Time for candidate selection: 0.15 seconds

### Candidate
type: RSZ, layer: 1, pos: 123

### Relational analysis RSZ of RS_RSZ2_RSZ2_RSZ1_RSZ1_RSZ1_RSZ1

#### Relational analysis RSZ result of RS_RSZ2_RSZ2_RSZ1_RSZ1_RSZ1_RSZ1
Status: Status.UNKNOWN
Output dim: 7, lower bound: -0.0009450, upper bound: 0.0009808
time: 0.66 seconds

### Relational analysis RSZ of RS_RSZ2_RSZ2_RSZ1_RSZ1_RSZ1_RSZ2

#### Relational analysis RSZ result of RS_RSZ2_RSZ2_RSZ1_RSZ1_RSZ1_RSZ2
Status: Status.UNKNOWN
Output dim: 7, lower bound: -0.0009450, upper bound: 0.0009816
time: 0.64 seconds

## BFS RS instance: RS_RSZ2_RSZ2_RSZ1_RSZ1_RSZ2

### Backsubstitution after applying RS history:
0: -0.0027939, -0.0016242, -0.0027939, -0.0016242, -0.0007433, 0.0007596
1: -0.0114005, -0.0084323, -0.0114005, -0.0084323, -0.0018862, 0.0019276
2: 0.0279571, 0.0297986, 0.0279571, 0.0297986, -0.0011702, 0.0011959
3: 0.0039210, 0.0073596, 0.0039210, 0.0073596, -0.0022330, 0.0021851
4: -0.0104893, -0.0074701, -0.0104893, -0.0074701, -0.0019186, 0.0019606
5: 0.0097651, 0.0109087, 0.0097651, 0.0109087, -0.0007267, 0.0007426
6: 0.0053643, 0.0097284, 0.0053643, 0.0097284, -0.0028339, 0.0027732
7: 0.9818129, 0.9848667, 0.9818129, 0.9848667, -0.0019831, 0.0019405
8: -0.0060636, -0.0027895, -0.0060636, -0.0027895, -0.0021262, 0.0020805
9: -0.0031570, -0.0009942, -0.0031570, -0.0009942, -0.0013743, 0.0014044

### Unstable ReLU Count (Linear/Conv2D Layers)
- layer_idx=0, type=LayerType.Linear, total=256, inp1_unstable=10, inp2_unstable=10, delta_unstable=251
- layer_idx=2, type=LayerType.Linear, total=256, inp1_unstable=10, inp2_unstable=10, delta_unstable=256
- layer_idx=4, type=LayerType.Linear, total=256, inp1_unstable=0, inp2_unstable=0, delta_unstable=256
- layer_idx=6, type=LayerType.Linear, total=256, inp1_unstable=0, inp2_unstable=0, delta_unstable=256
- layer_idx=8, type=LayerType.Linear, total=10, inp1_unstable=0, inp2_unstable=0, delta_unstable=10

Time for backsubstitution: 1.48 seconds

### RS candidates at layer 1
type: RSZ, layer: 1, pos: 123
type: RSZ, layer: 1, pos: 140
type: RSZ, layer: 1, pos: 167
type: RSZ, layer: 1, pos: 213
type: RSZ, layer: 1, pos: 219

Time for candidate selection: 0.15 seconds

### Candidate
type: RSZ, layer: 1, pos: 123

### Relational analysis RSZ of RS_RSZ2_RSZ2_RSZ1_RSZ1_RSZ2_RSZ1

#### Relational analysis RSZ result of RS_RSZ2_RSZ2_RSZ1_RSZ1_RSZ2_RSZ1
Status: Status.UNKNOWN
Output dim: 7, lower bound: -0.0009530, upper bound: 0.0009702
time: 0.64 seconds

### Relational analysis RSZ of RS_RSZ2_RSZ2_RSZ1_RSZ1_RSZ2_RSZ2

#### Relational analysis RSZ result of RS_RSZ2_RSZ2_RSZ1_RSZ1_RSZ2_RSZ2
Status: Status.UNKNOWN
Output dim: 7, lower bound: -0.0009529, upper bound: 0.0009702
time: 0.64 seconds

## BFS RS instance: RS_RSZ2_RSZ2_RSZ1_RSZ2_RSZ1

### Backsubstitution after applying RS history:
0: -0.0027939, -0.0016242, -0.0027939, -0.0016242, -0.0007655, 0.0007374
1: -0.0114005, -0.0084323, -0.0114005, -0.0084323, -0.0019426, 0.0018712
2: 0.0279571, 0.0297986, 0.0279571, 0.0297986, -0.0012052, 0.0011609
3: 0.0039210, 0.0073596, 0.0039210, 0.0073596, -0.0021677, 0.0022504
4: -0.0104893, -0.0074701, -0.0104893, -0.0074701, -0.0019759, 0.0019033
5: 0.0097651, 0.0109087, 0.0097651, 0.0109087, -0.0007484, 0.0007209
6: 0.0053643, 0.0097284, 0.0053643, 0.0097284, -0.0027511, 0.0028560
7: 0.9818129, 0.9848667, 0.9818129, 0.9848667, -0.0019251, 0.0019985
8: -0.0060636, -0.0027895, -0.0060636, -0.0027895, -0.0020640, 0.0021427
9: -0.0031570, -0.0009942, -0.0031570, -0.0009942, -0.0014154, 0.0013634

### Unstable ReLU Count (Linear/Conv2D Layers)
- layer_idx=0, type=LayerType.Linear, total=256, inp1_unstable=10, inp2_unstable=10, delta_unstable=251
- layer_idx=2, type=LayerType.Linear, total=256, inp1_unstable=10, inp2_unstable=10, delta_unstable=256
- layer_idx=4, type=LayerType.Linear, total=256, inp1_unstable=0, inp2_unstable=0, delta_unstable=256
- layer_idx=6, type=LayerType.Linear, total=256, inp1_unstable=0, inp2_unstable=0, delta_unstable=256
- layer_idx=8, type=LayerType.Linear, total=10, inp1_unstable=0, inp2_unstable=0, delta_unstable=10

Time for backsubstitution: 1.52 seconds

### RS candidates at layer 1
type: RSZ, layer: 1, pos: 123
type: RSZ, layer: 1, pos: 140
type: RSZ, layer: 1, pos: 167
type: RSZ, layer: 1, pos: 213
type: RSZ, layer: 1, pos: 219

Time for candidate selection: 0.17 seconds

### Candidate
type: RSZ, layer: 1, pos: 123

### Relational analysis RSZ of RS_RSZ2_RSZ2_RSZ1_RSZ2_RSZ1_RSZ1

#### Relational analysis RSZ result of RS_RSZ2_RSZ2_RSZ1_RSZ2_RSZ1_RSZ1
Status: Status.UNKNOWN
Output dim: 7, lower bound: -0.0009500, upper bound: 0.0009804
time: 0.64 seconds

### Relational analysis RSZ of RS_RSZ2_RSZ2_RSZ1_RSZ2_RSZ1_RSZ2

#### Relational analysis RSZ result of RS_RSZ2_RSZ2_RSZ1_RSZ2_RSZ1_RSZ2
Status: Status.UNKNOWN
Output dim: 7, lower bound: -0.0009497, upper bound: 0.0009809
time: 0.64 seconds

## BFS RS instance: RS_RSZ2_RSZ2_RSZ1_RSZ2_RSZ2

### Backsubstitution after applying RS history:
0: -0.0027939, -0.0016242, -0.0027939, -0.0016242, -0.0007502, 0.0007526
1: -0.0114005, -0.0084323, -0.0114005, -0.0084323, -0.0019038, 0.0019097
2: 0.0279571, 0.0297986, 0.0279571, 0.0297986, -0.0011811, 0.0011848
3: 0.0039210, 0.0073596, 0.0039210, 0.0073596, -0.0022123, 0.0022054
4: -0.0104893, -0.0074701, -0.0104893, -0.0074701, -0.0019365, 0.0019425
5: 0.0097651, 0.0109087, 0.0097651, 0.0109087, -0.0007335, 0.0007358
6: 0.0053643, 0.0097284, 0.0053643, 0.0097284, -0.0028077, 0.0027990
7: 0.9818129, 0.9848667, 0.9818129, 0.9848667, -0.0019647, 0.0019586
8: -0.0060636, -0.0027895, -0.0060636, -0.0027895, -0.0021065, 0.0020999
9: -0.0031570, -0.0009942, -0.0031570, -0.0009942, -0.0013871, 0.0013915

### Unstable ReLU Count (Linear/Conv2D Layers)
- layer_idx=0, type=LayerType.Linear, total=256, inp1_unstable=10, inp2_unstable=10, delta_unstable=251
- layer_idx=2, type=LayerType.Linear, total=256, inp1_unstable=10, inp2_unstable=10, delta_unstable=256
- layer_idx=4, type=LayerType.Linear, total=256, inp1_unstable=0, inp2_unstable=0, delta_unstable=256
- layer_idx=6, type=LayerType.Linear, total=256, inp1_unstable=0, inp2_unstable=0, delta_unstable=256
- layer_idx=8, type=LayerType.Linear, total=10, inp1_unstable=0, inp2_unstable=0, delta_unstable=10

Time for backsubstitution: 1.49 seconds

### RS candidates at layer 1
type: RSZ, layer: 1, pos: 123
type: RSZ, layer: 1, pos: 140
type: RSZ, layer: 1, pos: 167
type: RSZ, layer: 1, pos: 213
type: RSZ, layer: 1, pos: 219

Time for candidate selection: 0.16 seconds

### Candidate
type: RSZ, layer: 1, pos: 123

### Relational analysis RSZ of RS_RSZ2_RSZ2_RSZ1_RSZ2_RSZ2_RSZ1

#### Relational analysis RSZ result of RS_RSZ2_RSZ2_RSZ1_RSZ2_RSZ2_RSZ1
Status: Status.UNKNOWN
Output dim: 7, lower bound: -0.0009566, upper bound: 0.0009701
time: 0.65 seconds

### Relational analysis RSZ of RS_RSZ2_RSZ2_RSZ1_RSZ2_RSZ2_RSZ2

#### Relational analysis RSZ result of RS_RSZ2_RSZ2_RSZ1_RSZ2_RSZ2_RSZ2
Status: Status.UNKNOWN
Output dim: 7, lower bound: -0.0009568, upper bound: 0.0009701
time: 0.66 seconds

## BFS RS instance: RS_RSZ2_RSZ2_RSZ2_RSZ1_RSZ1

### Backsubstitution after applying RS history:
0: -0.0027939, -0.0016242, -0.0027939, -0.0016242, -0.0007594, 0.0007430
1: -0.0114005, -0.0084323, -0.0114005, -0.0084323, -0.0019271, 0.0018856
2: 0.0279571, 0.0297986, 0.0279571, 0.0297986, -0.0011956, 0.0011698
3: 0.0039210, 0.0073596, 0.0039210, 0.0073596, -0.0021844, 0.0022325
4: -0.0104893, -0.0074701, -0.0104893, -0.0074701, -0.0019602, 0.0019180
5: 0.0097651, 0.0109087, 0.0097651, 0.0109087, -0.0007425, 0.0007265
6: 0.0053643, 0.0097284, 0.0053643, 0.0097284, -0.0027722, 0.0028333
7: 0.9818129, 0.9848667, 0.9818129, 0.9848667, -0.0019399, 0.0019826
8: -0.0060636, -0.0027895, -0.0060636, -0.0027895, -0.0020799, 0.0021257
9: -0.0031570, -0.0009942, -0.0031570, -0.0009942, -0.0014041, 0.0013739

### Unstable ReLU Count (Linear/Conv2D Layers)
- layer_idx=0, type=LayerType.Linear, total=256, inp1_unstable=10, inp2_unstable=10, delta_unstable=251
- layer_idx=2, type=LayerType.Linear, total=256, inp1_unstable=10, inp2_unstable=10, delta_unstable=256
- layer_idx=4, type=LayerType.Linear, total=256, inp1_unstable=0, inp2_unstable=0, delta_unstable=256
- layer_idx=6, type=LayerType.Linear, total=256, inp1_unstable=0, inp2_unstable=0, delta_unstable=256
- layer_idx=8, type=LayerType.Linear, total=10, inp1_unstable=0, inp2_unstable=0, delta_unstable=10

Time for backsubstitution: 1.59 seconds

### RS candidates at layer 1
type: RSZ, layer: 1, pos: 123
type: RSZ, layer: 1, pos: 140
type: RSZ, layer: 1, pos: 167
type: RSZ, layer: 1, pos: 213
type: RSZ, layer: 1, pos: 219

Time for candidate selection: 0.17 seconds

### Candidate
type: RSZ, layer: 1, pos: 123

### Relational analysis RSZ of RS_RSZ2_RSZ2_RSZ2_RSZ1_RSZ1_RSZ1

#### Relational analysis RSZ result of RS_RSZ2_RSZ2_RSZ2_RSZ1_RSZ1_RSZ1
Status: Status.UNKNOWN
Output dim: 7, lower bound: -0.0009450, upper bound: 0.0009808
time: 0.67 seconds

### Relational analysis RSZ of RS_RSZ2_RSZ2_RSZ2_RSZ1_RSZ1_RSZ2

#### Relational analysis RSZ result of RS_RSZ2_RSZ2_RSZ2_RSZ1_RSZ1_RSZ2
Status: Status.UNKNOWN
Output dim: 7, lower bound: -0.0009450, upper bound: 0.0009816
time: 0.65 seconds

## BFS RS instance: RS_RSZ2_RSZ2_RSZ2_RSZ1_RSZ2

### Backsubstitution after applying RS history:
0: -0.0027939, -0.0016242, -0.0027939, -0.0016242, -0.0007450, 0.0007575
1: -0.0114005, -0.0084323, -0.0114005, -0.0084323, -0.0018905, 0.0019222
2: 0.0279571, 0.0297986, 0.0279571, 0.0297986, -0.0011728, 0.0011926
3: 0.0039210, 0.0073596, 0.0039210, 0.0073596, -0.0022268, 0.0021900
4: -0.0104893, -0.0074701, -0.0104893, -0.0074701, -0.0019229, 0.0019552
5: 0.0097651, 0.0109087, 0.0097651, 0.0109087, -0.0007283, 0.0007406
6: 0.0053643, 0.0097284, 0.0053643, 0.0097284, -0.0028261, 0.0027794
7: 0.9818129, 0.9848667, 0.9818129, 0.9848667, -0.0019776, 0.0019449
8: -0.0060636, -0.0027895, -0.0060636, -0.0027895, -0.0021203, 0.0020852
9: -0.0031570, -0.0009942, -0.0031570, -0.0009942, -0.0013774, 0.0014006

### Unstable ReLU Count (Linear/Conv2D Layers)
- layer_idx=0, type=LayerType.Linear, total=256, inp1_unstable=10, inp2_unstable=10, delta_unstable=251
- layer_idx=2, type=LayerType.Linear, total=256, inp1_unstable=10, inp2_unstable=10, delta_unstable=256
- layer_idx=4, type=LayerType.Linear, total=256, inp1_unstable=0, inp2_unstable=0, delta_unstable=256
- layer_idx=6, type=LayerType.Linear, total=256, inp1_unstable=0, inp2_unstable=0, delta_unstable=256
- layer_idx=8, type=LayerType.Linear, total=10, inp1_unstable=0, inp2_unstable=0, delta_unstable=10

Time for backsubstitution: 1.61 seconds

### RS candidates at layer 1
type: RSZ, layer: 1, pos: 123
type: RSZ, layer: 1, pos: 140
type: RSZ, layer: 1, pos: 167
type: RSZ, layer: 1, pos: 213
type: RSZ, layer: 1, pos: 219

Time for candidate selection: 0.16 seconds

### Candidate
type: RSZ, layer: 1, pos: 123

### Relational analysis RSZ of RS_RSZ2_RSZ2_RSZ2_RSZ1_RSZ2_RSZ1

#### Relational analysis RSZ result of RS_RSZ2_RSZ2_RSZ2_RSZ1_RSZ2_RSZ1
Status: Status.UNKNOWN
Output dim: 7, lower bound: -0.0009530, upper bound: 0.0009702
time: 0.69 seconds

### Relational analysis RSZ of RS_RSZ2_RSZ2_RSZ2_RSZ1_RSZ2_RSZ2

#### Relational analysis RSZ result of RS_RSZ2_RSZ2_RSZ2_RSZ1_RSZ2_RSZ2
Status: Status.UNKNOWN
Output dim: 7, lower bound: -0.0009529, upper bound: 0.0009702
time: 0.66 seconds

## BFS RS instance: RS_RSZ2_RSZ2_RSZ2_RSZ2_RSZ1

### Backsubstitution after applying RS history:
0: -0.0027939, -0.0016242, -0.0027939, -0.0016242, -0.0007672, 0.0007355
1: -0.0114005, -0.0084323, -0.0114005, -0.0084323, -0.0019468, 0.0018665
2: 0.0279571, 0.0297986, 0.0279571, 0.0297986, -0.0012078, 0.0011580
3: 0.0039210, 0.0073596, 0.0039210, 0.0073596, -0.0021622, 0.0022553
4: -0.0104893, -0.0074701, -0.0104893, -0.0074701, -0.0019802, 0.0018985
5: 0.0097651, 0.0109087, 0.0097651, 0.0109087, -0.0007501, 0.0007191
6: 0.0053643, 0.0097284, 0.0053643, 0.0097284, -0.0027441, 0.0028622
7: 0.9818129, 0.9848667, 0.9818129, 0.9848667, -0.0019202, 0.0020029
8: -0.0060636, -0.0027895, -0.0060636, -0.0027895, -0.0020588, 0.0021474
9: -0.0031570, -0.0009942, -0.0031570, -0.0009942, -0.0014185, 0.0013599

### Unstable ReLU Count (Linear/Conv2D Layers)
- layer_idx=0, type=LayerType.Linear, total=256, inp1_unstable=10, inp2_unstable=10, delta_unstable=251
- layer_idx=2, type=LayerType.Linear, total=256, inp1_unstable=10, inp2_unstable=10, delta_unstable=256
- layer_idx=4, type=LayerType.Linear, total=256, inp1_unstable=0, inp2_unstable=0, delta_unstable=256
- layer_idx=6, type=LayerType.Linear, total=256, inp1_unstable=0, inp2_unstable=0, delta_unstable=256
- layer_idx=8, type=LayerType.Linear, total=10, inp1_unstable=0, inp2_unstable=0, delta_unstable=10

Time for backsubstitution: 1.52 seconds

### RS candidates at layer 1
type: RSZ, layer: 1, pos: 123
type: RSZ, layer: 1, pos: 140
type: RSZ, layer: 1, pos: 167
type: RSZ, layer: 1, pos: 213
type: RSZ, layer: 1, pos: 219

Time for candidate selection: 0.16 seconds

### Candidate
type: RSZ, layer: 1, pos: 123

### Relational analysis RSZ of RS_RSZ2_RSZ2_RSZ2_RSZ2_RSZ1_RSZ1

#### Relational analysis RSZ result of RS_RSZ2_RSZ2_RSZ2_RSZ2_RSZ1_RSZ1
Status: Status.UNKNOWN
Output dim: 7, lower bound: -0.0009500, upper bound: 0.0009804
time: 0.64 seconds

### Relational analysis RSZ of RS_RSZ2_RSZ2_RSZ2_RSZ2_RSZ1_RSZ2

#### Relational analysis RSZ result of RS_RSZ2_RSZ2_RSZ2_RSZ2_RSZ1_RSZ2
Status: Status.UNKNOWN
Output dim: 7, lower bound: -0.0009497, upper bound: 0.0009809
time: 0.64 seconds

## BFS RS instance: RS_RSZ2_RSZ2_RSZ2_RSZ2_RSZ2

### Backsubstitution after applying RS history:
0: -0.0027939, -0.0016242, -0.0027939, -0.0016242, -0.0007519, 0.0007504
1: -0.0114005, -0.0084323, -0.0114005, -0.0084323, -0.0019080, 0.0019042
2: 0.0279571, 0.0297986, 0.0279571, 0.0297986, -0.0011837, 0.0011813
3: 0.0039210, 0.0073596, 0.0039210, 0.0073596, -0.0022059, 0.0022104
4: -0.0104893, -0.0074701, -0.0104893, -0.0074701, -0.0019408, 0.0019369
5: 0.0097651, 0.0109087, 0.0097651, 0.0109087, -0.0007351, 0.0007336
6: 0.0053643, 0.0097284, 0.0053643, 0.0097284, -0.0027995, 0.0028052
7: 0.9818129, 0.9848667, 0.9818129, 0.9848667, -0.0019590, 0.0019630
8: -0.0060636, -0.0027895, -0.0060636, -0.0027895, -0.0021003, 0.0021046
9: -0.0031570, -0.0009942, -0.0031570, -0.0009942, -0.0013902, 0.0013874

### Unstable ReLU Count (Linear/Conv2D Layers)
- layer_idx=0, type=LayerType.Linear, total=256, inp1_unstable=10, inp2_unstable=10, delta_unstable=251
- layer_idx=2, type=LayerType.Linear, total=256, inp1_unstable=10, inp2_unstable=10, delta_unstable=256
- layer_idx=4, type=LayerType.Linear, total=256, inp1_unstable=0, inp2_unstable=0, delta_unstable=256
- layer_idx=6, type=LayerType.Linear, total=256, inp1_unstable=0, inp2_unstable=0, delta_unstable=256
- layer_idx=8, type=LayerType.Linear, total=10, inp1_unstable=0, inp2_unstable=0, delta_unstable=10

Time for backsubstitution: 1.54 seconds

### RS candidates at layer 1
type: RSZ, layer: 1, pos: 123
type: RSZ, layer: 1, pos: 140
type: RSZ, layer: 1, pos: 167
type: RSZ, layer: 1, pos: 213
type: RSZ, layer: 1, pos: 219

Time for candidate selection: 0.20 seconds

### Candidate
type: RSZ, layer: 1, pos: 123

### Relational analysis RSZ of RS_RSZ2_RSZ2_RSZ2_RSZ2_RSZ2_RSZ1

#### Relational analysis RSZ result of RS_RSZ2_RSZ2_RSZ2_RSZ2_RSZ2_RSZ1
Status: Status.UNKNOWN
Output dim: 7, lower bound: -0.0009566, upper bound: 0.0009701
time: 0.65 seconds

### Relational analysis RSZ of RS_RSZ2_RSZ2_RSZ2_RSZ2_RSZ2_RSZ2

#### Relational analysis RSZ result of RS_RSZ2_RSZ2_RSZ2_RSZ2_RSZ2_RSZ2
Status: Status.UNKNOWN
Output dim: 7, lower bound: -0.0009568, upper bound: 0.0009701
time: 0.67 seconds

## Summary of splitting (split count: 5)
- Time for RS candidates: 3.07 seconds
RS_RSZ1_RSZ1_RSZ1_RSZ1_RSZ1_RSZ1, status: Status.UNKNOWN, split count: 6, time: 3.07
Output dim: 7, lower bound: -0.0009701, upper bound: 0.0009568
RS_RSZ1_RSZ1_RSZ1_RSZ1_RSZ1_RSZ2, status: Status.UNKNOWN, split count: 6, time: 3.07
Output dim: 7, lower bound: -0.0009701, upper bound: 0.0009566
RS_RSZ1_RSZ1_RSZ1_RSZ1_RSZ2_RSZ1, status: Status.UNKNOWN, split count: 6, time: 3.07
Output dim: 7, lower bound: -0.0009809, upper bound: 0.0009497
RS_RSZ1_RSZ1_RSZ1_RSZ1_RSZ2_RSZ2, status: Status.UNKNOWN, split count: 6, time: 3.07
Output dim: 7, lower bound: -0.0009804, upper bound: 0.0009500
RS_RSZ1_RSZ1_RSZ1_RSZ2_RSZ1_RSZ1, status: Status.UNKNOWN, split count: 6, time: 3.07
Output dim: 7, lower bound: -0.0009702, upper bound: 0.0009529
RS_RSZ1_RSZ1_RSZ1_RSZ2_RSZ1_RSZ2, status: Status.UNKNOWN, split count: 6, time: 3.07
Output dim: 7, lower bound: -0.0009702, upper bound: 0.0009530
RS_RSZ1_RSZ1_RSZ1_RSZ2_RSZ2_RSZ1, status: Status.UNKNOWN, split count: 6, time: 3.07
Output dim: 7, lower bound: -0.0009816, upper bound: 0.0009450
RS_RSZ1_RSZ1_RSZ1_RSZ2_RSZ2_RSZ2, status: Status.UNKNOWN, split count: 6, time: 3.07
Output dim: 7, lower bound: -0.0009808, upper bound: 0.0009450
RS_RSZ1_RSZ1_RSZ2_RSZ1_RSZ1_RSZ1, status: Status.UNKNOWN, split count: 6, time: 3.07
Output dim: 7, lower bound: -0.0009701, upper bound: 0.0009568
RS_RSZ1_RSZ1_RSZ2_RSZ1_RSZ1_RSZ2, status: Status.UNKNOWN, split count: 6, time: 3.07
Output dim: 7, lower bound: -0.0009701, upper bound: 0.0009566
RS_RSZ1_RSZ1_RSZ2_RSZ1_RSZ2_RSZ1, status: Status.UNKNOWN, split count: 6, time: 3.07
Output dim: 7, lower bound: -0.0009809, upper bound: 0.0009497
RS_RSZ1_RSZ1_RSZ2_RSZ1_RSZ2_RSZ2, status: Status.UNKNOWN, split count: 6, time: 3.07
Output dim: 7, lower bound: -0.0009804, upper bound: 0.0009500
RS_RSZ1_RSZ1_RSZ2_RSZ2_RSZ1_RSZ1, status: Status.UNKNOWN, split count: 6, time: 3.07
Output dim: 7, lower bound: -0.0009702, upper bound: 0.0009529
RS_RSZ1_RSZ1_RSZ2_RSZ2_RSZ1_RSZ2, status: Status.UNKNOWN, split count: 6, time: 3.07
Output dim: 7, lower bound: -0.0009702, upper bound: 0.0009530
RS_RSZ1_RSZ1_RSZ2_RSZ2_RSZ2_RSZ1, status: Status.UNKNOWN, split count: 6, time: 3.07
Output dim: 7, lower bound: -0.0009816, upper bound: 0.0009450
RS_RSZ1_RSZ1_RSZ2_RSZ2_RSZ2_RSZ2, status: Status.UNKNOWN, split count: 6, time: 3.07
Output dim: 7, lower bound: -0.0009808, upper bound: 0.0009450
RS_RSZ1_RSZ2_RSZ1_RSZ1_RSZ1_RSZ1, status: Status.UNKNOWN, split count: 6, time: 3.07
Output dim: 7, lower bound: -0.0009551, upper bound: 0.0009742
RS_RSZ1_RSZ2_RSZ1_RSZ1_RSZ1_RSZ2, status: Status.UNKNOWN, split count: 6, time: 3.07
Output dim: 7, lower bound: -0.0009551, upper bound: 0.0009754
RS_RSZ1_RSZ2_RSZ1_RSZ1_RSZ2_RSZ1, status: Status.UNKNOWN, split count: 6, time: 3.07
Output dim: 7, lower bound: -0.0009624, upper bound: 0.0009615
RS_RSZ1_RSZ2_RSZ1_RSZ1_RSZ2_RSZ2, status: Status.UNKNOWN, split count: 6, time: 3.07
Output dim: 7, lower bound: -0.0009625, upper bound: 0.0009620
RS_RSZ1_RSZ2_RSZ1_RSZ2_RSZ1_RSZ1, status: Status.UNKNOWN, split count: 6, time: 3.07
Output dim: 7, lower bound: -0.0009554, upper bound: 0.0009691
RS_RSZ1_RSZ2_RSZ1_RSZ2_RSZ1_RSZ2, status: Status.UNKNOWN, split count: 6, time: 3.07
Output dim: 7, lower bound: -0.0009554, upper bound: 0.0009710
RS_RSZ1_RSZ2_RSZ1_RSZ2_RSZ2_RSZ1, status: Status.UNKNOWN, split count: 6, time: 3.07
Output dim: 7, lower bound: -0.0009624, upper bound: 0.0009526
RS_RSZ1_RSZ2_RSZ1_RSZ2_RSZ2_RSZ2, status: Status.UNKNOWN, split count: 6, time: 3.07
Output dim: 7, lower bound: -0.0009626, upper bound: 0.0009532
RS_RSZ1_RSZ2_RSZ2_RSZ1_RSZ1_RSZ1, status: Status.UNKNOWN, split count: 6, time: 3.07
Output dim: 7, lower bound: -0.0009551, upper bound: 0.0009742
RS_RSZ1_RSZ2_RSZ2_RSZ1_RSZ1_RSZ2, status: Status.UNKNOWN, split count: 6, time: 3.07
Output dim: 7, lower bound: -0.0009551, upper bound: 0.0009754
RS_RSZ1_RSZ2_RSZ2_RSZ1_RSZ2_RSZ1, status: Status.UNKNOWN, split count: 6, time: 3.07
Output dim: 7, lower bound: -0.0009624, upper bound: 0.0009615
RS_RSZ1_RSZ2_RSZ2_RSZ1_RSZ2_RSZ2, status: Status.UNKNOWN, split count: 6, time: 3.07
Output dim: 7, lower bound: -0.0009625, upper bound: 0.0009620
RS_RSZ1_RSZ2_RSZ2_RSZ2_RSZ1_RSZ1, status: Status.UNKNOWN, split count: 6, time: 3.07
Output dim: 7, lower bound: -0.0009554, upper bound: 0.0009691
RS_RSZ1_RSZ2_RSZ2_RSZ2_RSZ1_RSZ2, status: Status.UNKNOWN, split count: 6, time: 3.07
Output dim: 7, lower bound: -0.0009554, upper bound: 0.0009710
RS_RSZ1_RSZ2_RSZ2_RSZ2_RSZ2_RSZ1, status: Status.UNKNOWN, split count: 6, time: 3.07
Output dim: 7, lower bound: -0.0009624, upper bound: 0.0009526
RS_RSZ1_RSZ2_RSZ2_RSZ2_RSZ2_RSZ2, status: Status.UNKNOWN, split count: 6, time: 3.07
Output dim: 7, lower bound: -0.0009626, upper bound: 0.0009532
RS_RSZ2_RSZ1_RSZ1_RSZ1_RSZ1_RSZ1, status: Status.UNKNOWN, split count: 6, time: 3.07
Output dim: 7, lower bound: -0.0009532, upper bound: 0.0009626
RS_RSZ2_RSZ1_RSZ1_RSZ1_RSZ1_RSZ2, status: Status.UNKNOWN, split count: 6, time: 3.07
Output dim: 7, lower bound: -0.0009526, upper bound: 0.0009624
RS_RSZ2_RSZ1_RSZ1_RSZ1_RSZ2_RSZ1, status: Status.UNKNOWN, split count: 6, time: 3.07
Output dim: 7, lower bound: -0.0009710, upper bound: 0.0009554
RS_RSZ2_RSZ1_RSZ1_RSZ1_RSZ2_RSZ2, status: Status.UNKNOWN, split count: 6, time: 3.07
Output dim: 7, lower bound: -0.0009691, upper bound: 0.0009554
RS_RSZ2_RSZ1_RSZ1_RSZ2_RSZ1_RSZ1, status: Status.UNKNOWN, split count: 6, time: 3.07
Output dim: 7, lower bound: -0.0009620, upper bound: 0.0009625
RS_RSZ2_RSZ1_RSZ1_RSZ2_RSZ1_RSZ2, status: Status.UNKNOWN, split count: 6, time: 3.07
Output dim: 7, lower bound: -0.0009615, upper bound: 0.0009624
RS_RSZ2_RSZ1_RSZ1_RSZ2_RSZ2_RSZ1, status: Status.UNKNOWN, split count: 6, time: 3.07
Output dim: 7, lower bound: -0.0009754, upper bound: 0.0009551
RS_RSZ2_RSZ1_RSZ1_RSZ2_RSZ2_RSZ2, status: Status.UNKNOWN, split count: 6, time: 3.07
Output dim: 7, lower bound: -0.0009742, upper bound: 0.0009551
RS_RSZ2_RSZ1_RSZ2_RSZ1_RSZ1_RSZ1, status: Status.UNKNOWN, split count: 6, time: 3.07
Output dim: 7, lower bound: -0.0009532, upper bound: 0.0009626
RS_RSZ2_RSZ1_RSZ2_RSZ1_RSZ1_RSZ2, status: Status.UNKNOWN, split count: 6, time: 3.07
Output dim: 7, lower bound: -0.0009526, upper bound: 0.0009624
RS_RSZ2_RSZ1_RSZ2_RSZ1_RSZ2_RSZ1, status: Status.UNKNOWN, split count: 6, time: 3.07
Output dim: 7, lower bound: -0.0009710, upper bound: 0.0009554
RS_RSZ2_RSZ1_RSZ2_RSZ1_RSZ2_RSZ2, status: Status.UNKNOWN, split count: 6, time: 3.07
Output dim: 7, lower bound: -0.0009691, upper bound: 0.0009554
RS_RSZ2_RSZ1_RSZ2_RSZ2_RSZ1_RSZ1, status: Status.UNKNOWN, split count: 6, time: 3.07
Output dim: 7, lower bound: -0.0009620, upper bound: 0.0009625
RS_RSZ2_RSZ1_RSZ2_RSZ2_RSZ1_RSZ2, status: Status.UNKNOWN, split count: 6, time: 3.07
Output dim: 7, lower bound: -0.0009615, upper bound: 0.0009624
RS_RSZ2_RSZ1_RSZ2_RSZ2_RSZ2_RSZ1, status: Status.UNKNOWN, split count: 6, time: 3.07
Output dim: 7, lower bound: -0.0009754, upper bound: 0.0009551
RS_RSZ2_RSZ1_RSZ2_RSZ2_RSZ2_RSZ2, status: Status.UNKNOWN, split count: 6, time: 3.07
Output dim: 7, lower bound: -0.0009742, upper bound: 0.0009551
RS_RSZ2_RSZ2_RSZ1_RSZ1_RSZ1_RSZ1, status: Status.UNKNOWN, split count: 6, time: 3.07
Output dim: 7, lower bound: -0.0009450, upper bound: 0.0009808
RS_RSZ2_RSZ2_RSZ1_RSZ1_RSZ1_RSZ2, status: Status.UNKNOWN, split count: 6, time: 3.07
Output dim: 7, lower bound: -0.0009450, upper bound: 0.0009816
RS_RSZ2_RSZ2_RSZ1_RSZ1_RSZ2_RSZ1, status: Status.UNKNOWN, split count: 6, time: 3.07
Output dim: 7, lower bound: -0.0009530, upper bound: 0.0009702
RS_RSZ2_RSZ2_RSZ1_RSZ1_RSZ2_RSZ2, status: Status.UNKNOWN, split count: 6, time: 3.07
Output dim: 7, lower bound: -0.0009529, upper bound: 0.0009702
RS_RSZ2_RSZ2_RSZ1_RSZ2_RSZ1_RSZ1, status: Status.UNKNOWN, split count: 6, time: 3.07
Output dim: 7, lower bound: -0.0009500, upper bound: 0.0009804
RS_RSZ2_RSZ2_RSZ1_RSZ2_RSZ1_RSZ2, status: Status.UNKNOWN, split count: 6, time: 3.07
Output dim: 7, lower bound: -0.0009497, upper bound: 0.0009809
RS_RSZ2_RSZ2_RSZ1_RSZ2_RSZ2_RSZ1, status: Status.UNKNOWN, split count: 6, time: 3.07
Output dim: 7, lower bound: -0.0009566, upper bound: 0.0009701
RS_RSZ2_RSZ2_RSZ1_RSZ2_RSZ2_RSZ2, status: Status.UNKNOWN, split count: 6, time: 3.07
Output dim: 7, lower bound: -0.0009568, upper bound: 0.0009701
RS_RSZ2_RSZ2_RSZ2_RSZ1_RSZ1_RSZ1, status: Status.UNKNOWN, split count: 6, time: 3.07
Output dim: 7, lower bound: -0.0009450, upper bound: 0.0009808
RS_RSZ2_RSZ2_RSZ2_RSZ1_RSZ1_RSZ2, status: Status.UNKNOWN, split count: 6, time: 3.07
Output dim: 7, lower bound: -0.0009450, upper bound: 0.0009816
RS_RSZ2_RSZ2_RSZ2_RSZ1_RSZ2_RSZ1, status: Status.UNKNOWN, split count: 6, time: 3.07
Output dim: 7, lower bound: -0.0009530, upper bound: 0.0009702
RS_RSZ2_RSZ2_RSZ2_RSZ1_RSZ2_RSZ2, status: Status.UNKNOWN, split count: 6, time: 3.07
Output dim: 7, lower bound: -0.0009529, upper bound: 0.0009702
RS_RSZ2_RSZ2_RSZ2_RSZ2_RSZ1_RSZ1, status: Status.UNKNOWN, split count: 6, time: 3.07
Output dim: 7, lower bound: -0.0009500, upper bound: 0.0009804
RS_RSZ2_RSZ2_RSZ2_RSZ2_RSZ1_RSZ2, status: Status.UNKNOWN, split count: 6, time: 3.07
Output dim: 7, lower bound: -0.0009497, upper bound: 0.0009809
RS_RSZ2_RSZ2_RSZ2_RSZ2_RSZ2_RSZ1, status: Status.UNKNOWN, split count: 6, time: 3.07
Output dim: 7, lower bound: -0.0009566, upper bound: 0.0009701
RS_RSZ2_RSZ2_RSZ2_RSZ2_RSZ2_RSZ2, status: Status.UNKNOWN, split count: 6, time: 3.07
Output dim: 7, lower bound: -0.0009568, upper bound: 0.0009701

## BFS RS instance: RS_RSZ1_RSZ1_RSZ1_RSZ1_RSZ1_RSZ1

### Backsubstitution after applying RS history:
0: -0.0027939, -0.0016242, -0.0027939, -0.0016242, -0.0007503, 0.0007525
1: -0.0114005, -0.0084323, -0.0114005, -0.0084323, -0.0019039, 0.0019095
2: 0.0279571, 0.0297986, 0.0279571, 0.0297986, -0.0011812, 0.0011846
3: 0.0039210, 0.0073596, 0.0039210, 0.0073596, -0.0022120, 0.0022056
4: -0.0104893, -0.0074701, -0.0104893, -0.0074701, -0.0019366, 0.0019423
5: 0.0097651, 0.0109087, 0.0097651, 0.0109087, -0.0007335, 0.0007357
6: 0.0053643, 0.0097284, 0.0053643, 0.0097284, -0.0028073, 0.0027992
7: 0.9818129, 0.9848667, 0.9818129, 0.9848667, -0.0019644, 0.0019588
8: -0.0060636, -0.0027895, -0.0060636, -0.0027895, -0.0021062, 0.0021001
9: -0.0031570, -0.0009942, -0.0031570, -0.0009942, -0.0013872, 0.0013913

### Unstable ReLU Count (Linear/Conv2D Layers)
- layer_idx=0, type=LayerType.Linear, total=256, inp1_unstable=10, inp2_unstable=10, delta_unstable=250
- layer_idx=2, type=LayerType.Linear, total=256, inp1_unstable=10, inp2_unstable=10, delta_unstable=256
- layer_idx=4, type=LayerType.Linear, total=256, inp1_unstable=0, inp2_unstable=0, delta_unstable=256
- layer_idx=6, type=LayerType.Linear, total=256, inp1_unstable=0, inp2_unstable=0, delta_unstable=256
- layer_idx=8, type=LayerType.Linear, total=10, inp1_unstable=0, inp2_unstable=0, delta_unstable=10

Time for backsubstitution: 1.50 seconds

### RS candidates at layer 1
type: RSZ, layer: 1, pos: 140
type: RSZ, layer: 1, pos: 167
type: RSZ, layer: 1, pos: 213
type: RSZ, layer: 1, pos: 219

Time for candidate selection: 0.15 seconds

### Candidate
type: RSZ, layer: 1, pos: 140

### Relational analysis RSZ of RS_RSZ1_RSZ1_RSZ1_RSZ1_RSZ1_RSZ1_RSZ1
Optimization infeasible because this subproblem isn't reachable.

### Candidate
type: RSZ, layer: 1, pos: 167

### Relational analysis RSZ of RS_RSZ1_RSZ1_RSZ1_RSZ1_RSZ1_RSZ1_RSZ1

#### Relational analysis RSZ result of RS_RSZ1_RSZ1_RSZ1_RSZ1_RSZ1_RSZ1_RSZ1
Status: Status.UNKNOWN
Output dim: 7, lower bound: -0.0009496, upper bound: 0.0009333
time: 0.62 seconds

### Relational analysis RSZ of RS_RSZ1_RSZ1_RSZ1_RSZ1_RSZ1_RSZ1_RSZ2

#### Relational analysis RSZ result of RS_RSZ1_RSZ1_RSZ1_RSZ1_RSZ1_RSZ1_RSZ2
Status: Status.UNKNOWN
Output dim: 7, lower bound: -0.0009454, upper bound: 0.0009365
time: 0.64 seconds

## BFS RS instance: RS_RSZ1_RSZ1_RSZ1_RSZ1_RSZ1_RSZ2

### Backsubstitution after applying RS history:
0: -0.0027939, -0.0016242, -0.0027939, -0.0016242, -0.0007504, 0.0007523
1: -0.0114005, -0.0084323, -0.0114005, -0.0084323, -0.0019043, 0.0019091
2: 0.0279571, 0.0297986, 0.0279571, 0.0297986, -0.0011815, 0.0011844
3: 0.0039210, 0.0073596, 0.0039210, 0.0073596, -0.0022116, 0.0022061
4: -0.0104893, -0.0074701, -0.0104893, -0.0074701, -0.0019370, 0.0019418
5: 0.0097651, 0.0109087, 0.0097651, 0.0109087, -0.0007337, 0.0007355
6: 0.0053643, 0.0097284, 0.0053643, 0.0097284, -0.0028067, 0.0027998
7: 0.9818129, 0.9848667, 0.9818129, 0.9848667, -0.0019640, 0.0019592
8: -0.0060636, -0.0027895, -0.0060636, -0.0027895, -0.0021057, 0.0021005
9: -0.0031570, -0.0009942, -0.0031570, -0.0009942, -0.0013875, 0.0013910

### Unstable ReLU Count (Linear/Conv2D Layers)
- layer_idx=0, type=LayerType.Linear, total=256, inp1_unstable=10, inp2_unstable=10, delta_unstable=250
- layer_idx=2, type=LayerType.Linear, total=256, inp1_unstable=10, inp2_unstable=10, delta_unstable=256
- layer_idx=4, type=LayerType.Linear, total=256, inp1_unstable=0, inp2_unstable=0, delta_unstable=256
- layer_idx=6, type=LayerType.Linear, total=256, inp1_unstable=0, inp2_unstable=0, delta_unstable=256
- layer_idx=8, type=LayerType.Linear, total=10, inp1_unstable=0, inp2_unstable=0, delta_unstable=10

Time for backsubstitution: 1.64 seconds

### RS candidates at layer 1
type: RSZ, layer: 1, pos: 140
type: RSZ, layer: 1, pos: 167
type: RSZ, layer: 1, pos: 213
type: RSZ, layer: 1, pos: 219

Time for candidate selection: 0.19 seconds

### Candidate
type: RSZ, layer: 1, pos: 140

### Relational analysis RSZ of RS_RSZ1_RSZ1_RSZ1_RSZ1_RSZ1_RSZ2_RSZ1
Optimization infeasible because this subproblem isn't reachable.

### Candidate
type: RSZ, layer: 1, pos: 167

### Relational analysis RSZ of RS_RSZ1_RSZ1_RSZ1_RSZ1_RSZ1_RSZ2_RSZ1

#### Relational analysis RSZ result of RS_RSZ1_RSZ1_RSZ1_RSZ1_RSZ1_RSZ2_RSZ1
Status: Status.UNKNOWN
Output dim: 7, lower bound: -0.0009496, upper bound: 0.0009330
time: 0.63 seconds

### Relational analysis RSZ of RS_RSZ1_RSZ1_RSZ1_RSZ1_RSZ1_RSZ2_RSZ2

#### Relational analysis RSZ result of RS_RSZ1_RSZ1_RSZ1_RSZ1_RSZ1_RSZ2_RSZ2
Status: Status.UNKNOWN
Output dim: 7, lower bound: -0.0009452, upper bound: 0.0009362
time: 0.66 seconds

## BFS RS instance: RS_RSZ1_RSZ1_RSZ1_RSZ1_RSZ2_RSZ1

### Backsubstitution after applying RS history:
0: -0.0027939, -0.0016242, -0.0027939, -0.0016242, -0.0007354, 0.0007677
1: -0.0114005, -0.0084323, -0.0114005, -0.0084323, -0.0018662, 0.0019483
2: 0.0279571, 0.0297986, 0.0279571, 0.0297986, -0.0011578, 0.0012087
3: 0.0039210, 0.0073596, 0.0039210, 0.0073596, -0.0022570, 0.0021619
4: -0.0104893, -0.0074701, -0.0104893, -0.0074701, -0.0018983, 0.0019817
5: 0.0097651, 0.0109087, 0.0097651, 0.0109087, -0.0007190, 0.0007506
6: 0.0053643, 0.0097284, 0.0053643, 0.0097284, -0.0028644, 0.0027438
7: 0.9818129, 0.9848667, 0.9818129, 0.9848667, -0.0020044, 0.0019200
8: -0.0060636, -0.0027895, -0.0060636, -0.0027895, -0.0021490, 0.0020585
9: -0.0031570, -0.0009942, -0.0031570, -0.0009942, -0.0013598, 0.0014195

### Unstable ReLU Count (Linear/Conv2D Layers)
- layer_idx=0, type=LayerType.Linear, total=256, inp1_unstable=10, inp2_unstable=10, delta_unstable=250
- layer_idx=2, type=LayerType.Linear, total=256, inp1_unstable=10, inp2_unstable=10, delta_unstable=256
- layer_idx=4, type=LayerType.Linear, total=256, inp1_unstable=0, inp2_unstable=0, delta_unstable=256
- layer_idx=6, type=LayerType.Linear, total=256, inp1_unstable=0, inp2_unstable=0, delta_unstable=256
- layer_idx=8, type=LayerType.Linear, total=10, inp1_unstable=0, inp2_unstable=0, delta_unstable=10

Time for backsubstitution: 1.50 seconds

### RS candidates at layer 1
type: RSZ, layer: 1, pos: 140
type: RSZ, layer: 1, pos: 167
type: RSZ, layer: 1, pos: 213
type: RSZ, layer: 1, pos: 219

Time for candidate selection: 0.15 seconds

### Candidate
type: RSZ, layer: 1, pos: 140

### Relational analysis RSZ of RS_RSZ1_RSZ1_RSZ1_RSZ1_RSZ2_RSZ1_RSZ1
Optimization infeasible because this subproblem isn't reachable.

### Candidate
type: RSZ, layer: 1, pos: 167

### Relational analysis RSZ of RS_RSZ1_RSZ1_RSZ1_RSZ1_RSZ2_RSZ1_RSZ1

#### Relational analysis RSZ result of RS_RSZ1_RSZ1_RSZ1_RSZ1_RSZ2_RSZ1_RSZ1
Status: Status.UNKNOWN
Output dim: 7, lower bound: -0.0009605, upper bound: 0.0009247
time: 0.62 seconds

### Relational analysis RSZ of RS_RSZ1_RSZ1_RSZ1_RSZ1_RSZ2_RSZ1_RSZ2

#### Relational analysis RSZ result of RS_RSZ1_RSZ1_RSZ1_RSZ1_RSZ2_RSZ1_RSZ2
Status: Status.UNKNOWN
Output dim: 7, lower bound: -0.0009552, upper bound: 0.0009294
time: 0.61 seconds

## BFS RS instance: RS_RSZ1_RSZ1_RSZ1_RSZ1_RSZ2_RSZ2

### Backsubstitution after applying RS history:
0: -0.0027939, -0.0016242, -0.0027939, -0.0016242, -0.0007356, 0.0007677
1: -0.0114005, -0.0084323, -0.0114005, -0.0084323, -0.0018666, 0.0019482
2: 0.0279571, 0.0297986, 0.0279571, 0.0297986, -0.0011581, 0.0012087
3: 0.0039210, 0.0073596, 0.0039210, 0.0073596, -0.0022569, 0.0021624
4: -0.0104893, -0.0074701, -0.0104893, -0.0074701, -0.0018987, 0.0019816
5: 0.0097651, 0.0109087, 0.0097651, 0.0109087, -0.0007192, 0.0007506
6: 0.0053643, 0.0097284, 0.0053643, 0.0097284, -0.0028643, 0.0027444
7: 0.9818129, 0.9848667, 0.9818129, 0.9848667, -0.0020043, 0.0019204
8: -0.0060636, -0.0027895, -0.0060636, -0.0027895, -0.0021489, 0.0020589
9: -0.0031570, -0.0009942, -0.0031570, -0.0009942, -0.0013600, 0.0014195

### Unstable ReLU Count (Linear/Conv2D Layers)
- layer_idx=0, type=LayerType.Linear, total=256, inp1_unstable=10, inp2_unstable=10, delta_unstable=250
- layer_idx=2, type=LayerType.Linear, total=256, inp1_unstable=10, inp2_unstable=10, delta_unstable=256
- layer_idx=4, type=LayerType.Linear, total=256, inp1_unstable=0, inp2_unstable=0, delta_unstable=256
- layer_idx=6, type=LayerType.Linear, total=256, inp1_unstable=0, inp2_unstable=0, delta_unstable=256
- layer_idx=8, type=LayerType.Linear, total=10, inp1_unstable=0, inp2_unstable=0, delta_unstable=10

Time for backsubstitution: 1.48 seconds

### RS candidates at layer 1
type: RSZ, layer: 1, pos: 140
type: RSZ, layer: 1, pos: 167
type: RSZ, layer: 1, pos: 213
type: RSZ, layer: 1, pos: 219

Time for candidate selection: 0.15 seconds

### Candidate
type: RSZ, layer: 1, pos: 140

### Relational analysis RSZ of RS_RSZ1_RSZ1_RSZ1_RSZ1_RSZ2_RSZ2_RSZ1
Optimization infeasible because this subproblem isn't reachable.

### Candidate
type: RSZ, layer: 1, pos: 167

### Relational analysis RSZ of RS_RSZ1_RSZ1_RSZ1_RSZ1_RSZ2_RSZ2_RSZ1

#### Relational analysis RSZ result of RS_RSZ1_RSZ1_RSZ1_RSZ1_RSZ2_RSZ2_RSZ1
Status: Status.UNKNOWN
Output dim: 7, lower bound: -0.0009601, upper bound: 0.0009248
time: 0.63 seconds

### Relational analysis RSZ of RS_RSZ1_RSZ1_RSZ1_RSZ1_RSZ2_RSZ2_RSZ2

#### Relational analysis RSZ result of RS_RSZ1_RSZ1_RSZ1_RSZ1_RSZ2_RSZ2_RSZ2
Status: Status.UNKNOWN
Output dim: 7, lower bound: -0.0009546, upper bound: 0.0009297
time: 0.64 seconds

## BFS RS instance: RS_RSZ1_RSZ1_RSZ1_RSZ2_RSZ1_RSZ1

### Backsubstitution after applying RS history:
0: -0.0027939, -0.0016242, -0.0027939, -0.0016242, -0.0007575, 0.0007448
1: -0.0114005, -0.0084323, -0.0114005, -0.0084323, -0.0019222, 0.0018901
2: 0.0279571, 0.0297986, 0.0279571, 0.0297986, -0.0011925, 0.0011726
3: 0.0039210, 0.0073596, 0.0039210, 0.0073596, -0.0021896, 0.0022267
4: -0.0104893, -0.0074701, -0.0104893, -0.0074701, -0.0019552, 0.0019225
5: 0.0097651, 0.0109087, 0.0097651, 0.0109087, -0.0007406, 0.0007282
6: 0.0053643, 0.0097284, 0.0053643, 0.0097284, -0.0027788, 0.0028260
7: 0.9818129, 0.9848667, 0.9818129, 0.9848667, -0.0019445, 0.0019775
8: -0.0060636, -0.0027895, -0.0060636, -0.0027895, -0.0020848, 0.0021202
9: -0.0031570, -0.0009942, -0.0031570, -0.0009942, -0.0014005, 0.0013771

### Unstable ReLU Count (Linear/Conv2D Layers)
- layer_idx=0, type=LayerType.Linear, total=256, inp1_unstable=10, inp2_unstable=10, delta_unstable=250
- layer_idx=2, type=LayerType.Linear, total=256, inp1_unstable=10, inp2_unstable=10, delta_unstable=256
- layer_idx=4, type=LayerType.Linear, total=256, inp1_unstable=0, inp2_unstable=0, delta_unstable=256
- layer_idx=6, type=LayerType.Linear, total=256, inp1_unstable=0, inp2_unstable=0, delta_unstable=256
- layer_idx=8, type=LayerType.Linear, total=10, inp1_unstable=0, inp2_unstable=0, delta_unstable=10

Time for backsubstitution: 1.52 seconds

### RS candidates at layer 1
type: RSZ, layer: 1, pos: 140
type: RSZ, layer: 1, pos: 167
type: RSZ, layer: 1, pos: 213
type: RSZ, layer: 1, pos: 219

Time for candidate selection: 0.16 seconds

### Candidate
type: RSZ, layer: 1, pos: 140

### Relational analysis RSZ of RS_RSZ1_RSZ1_RSZ1_RSZ2_RSZ1_RSZ1_RSZ1
Optimization infeasible because this subproblem isn't reachable.

### Candidate
type: RSZ, layer: 1, pos: 167

### Relational analysis RSZ of RS_RSZ1_RSZ1_RSZ1_RSZ2_RSZ1_RSZ1_RSZ1

#### Relational analysis RSZ result of RS_RSZ1_RSZ1_RSZ1_RSZ2_RSZ1_RSZ1_RSZ1
Status: Status.UNKNOWN
Output dim: 7, lower bound: -0.0009498, upper bound: 0.0009292
time: 0.69 seconds

### Relational analysis RSZ of RS_RSZ1_RSZ1_RSZ1_RSZ2_RSZ1_RSZ1_RSZ2

#### Relational analysis RSZ result of RS_RSZ1_RSZ1_RSZ1_RSZ2_RSZ1_RSZ1_RSZ2
Status: Status.UNKNOWN
Output dim: 7, lower bound: -0.0009461, upper bound: 0.0009326
time: 0.74 seconds

## BFS RS instance: RS_RSZ1_RSZ1_RSZ1_RSZ2_RSZ1_RSZ2

### Backsubstitution after applying RS history:
0: -0.0027939, -0.0016242, -0.0027939, -0.0016242, -0.0007576, 0.0007446
1: -0.0114005, -0.0084323, -0.0114005, -0.0084323, -0.0019224, 0.0018896
2: 0.0279571, 0.0297986, 0.0279571, 0.0297986, -0.0011927, 0.0011723
3: 0.0039210, 0.0073596, 0.0039210, 0.0073596, -0.0021890, 0.0022270
4: -0.0104893, -0.0074701, -0.0104893, -0.0074701, -0.0019554, 0.0019221
5: 0.0097651, 0.0109087, 0.0097651, 0.0109087, -0.0007407, 0.0007280
6: 0.0053643, 0.0097284, 0.0053643, 0.0097284, -0.0027782, 0.0028264
7: 0.9818129, 0.9848667, 0.9818129, 0.9848667, -0.0019440, 0.0019777
8: -0.0060636, -0.0027895, -0.0060636, -0.0027895, -0.0020843, 0.0021205
9: -0.0031570, -0.0009942, -0.0031570, -0.0009942, -0.0014007, 0.0013768

### Unstable ReLU Count (Linear/Conv2D Layers)
- layer_idx=0, type=LayerType.Linear, total=256, inp1_unstable=10, inp2_unstable=10, delta_unstable=250
- layer_idx=2, type=LayerType.Linear, total=256, inp1_unstable=10, inp2_unstable=10, delta_unstable=256
- layer_idx=4, type=LayerType.Linear, total=256, inp1_unstable=0, inp2_unstable=0, delta_unstable=256
- layer_idx=6, type=LayerType.Linear, total=256, inp1_unstable=0, inp2_unstable=0, delta_unstable=256
- layer_idx=8, type=LayerType.Linear, total=10, inp1_unstable=0, inp2_unstable=0, delta_unstable=10

Time for backsubstitution: 1.88 seconds

### RS candidates at layer 1
type: RSZ, layer: 1, pos: 140
type: RSZ, layer: 1, pos: 167
type: RSZ, layer: 1, pos: 213
type: RSZ, layer: 1, pos: 219

Time for candidate selection: 0.22 seconds

### Candidate
type: RSZ, layer: 1, pos: 140

### Relational analysis RSZ of RS_RSZ1_RSZ1_RSZ1_RSZ2_RSZ1_RSZ2_RSZ1
Optimization infeasible because this subproblem isn't reachable.

### Candidate
type: RSZ, layer: 1, pos: 167

### Relational analysis RSZ of RS_RSZ1_RSZ1_RSZ1_RSZ2_RSZ1_RSZ2_RSZ1

#### Relational analysis RSZ result of RS_RSZ1_RSZ1_RSZ1_RSZ2_RSZ1_RSZ2_RSZ1
Status: Status.UNKNOWN
Output dim: 7, lower bound: -0.0009497, upper bound: 0.0009291
time: 0.77 seconds

### Relational analysis RSZ of RS_RSZ1_RSZ1_RSZ1_RSZ2_RSZ1_RSZ2_RSZ2

#### Relational analysis RSZ result of RS_RSZ1_RSZ1_RSZ1_RSZ2_RSZ1_RSZ2_RSZ2
Status: Status.UNKNOWN
Output dim: 7, lower bound: -0.0009458, upper bound: 0.0009327
time: 0.63 seconds

## BFS RS instance: RS_RSZ1_RSZ1_RSZ1_RSZ2_RSZ2_RSZ1

### Backsubstitution after applying RS history:
0: -0.0027939, -0.0016242, -0.0027939, -0.0016242, -0.0007430, 0.0007599
1: -0.0114005, -0.0084323, -0.0114005, -0.0084323, -0.0018854, 0.0019285
2: 0.0279571, 0.0297986, 0.0279571, 0.0297986, -0.0011697, 0.0011964
3: 0.0039210, 0.0073596, 0.0039210, 0.0073596, -0.0022340, 0.0021841
4: -0.0104893, -0.0074701, -0.0104893, -0.0074701, -0.0019177, 0.0019616
5: 0.0097651, 0.0109087, 0.0097651, 0.0109087, -0.0007264, 0.0007430
6: 0.0053643, 0.0097284, 0.0053643, 0.0097284, -0.0028353, 0.0027719
7: 0.9818129, 0.9848667, 0.9818129, 0.9848667, -0.0019840, 0.0019396
8: -0.0060636, -0.0027895, -0.0060636, -0.0027895, -0.0021271, 0.0020796
9: -0.0031570, -0.0009942, -0.0031570, -0.0009942, -0.0013737, 0.0014051

### Unstable ReLU Count (Linear/Conv2D Layers)
- layer_idx=0, type=LayerType.Linear, total=256, inp1_unstable=10, inp2_unstable=10, delta_unstable=250
- layer_idx=2, type=LayerType.Linear, total=256, inp1_unstable=10, inp2_unstable=10, delta_unstable=256
- layer_idx=4, type=LayerType.Linear, total=256, inp1_unstable=0, inp2_unstable=0, delta_unstable=256
- layer_idx=6, type=LayerType.Linear, total=256, inp1_unstable=0, inp2_unstable=0, delta_unstable=256
- layer_idx=8, type=LayerType.Linear, total=10, inp1_unstable=0, inp2_unstable=0, delta_unstable=10

Time for backsubstitution: 1.55 seconds

### RS candidates at layer 1
type: RSZ, layer: 1, pos: 140
type: RSZ, layer: 1, pos: 167
type: RSZ, layer: 1, pos: 213
type: RSZ, layer: 1, pos: 219

Time for candidate selection: 0.16 seconds

### Candidate
type: RSZ, layer: 1, pos: 140

### Relational analysis RSZ of RS_RSZ1_RSZ1_RSZ1_RSZ2_RSZ2_RSZ1_RSZ1
Optimization infeasible because this subproblem isn't reachable.

### Candidate
type: RSZ, layer: 1, pos: 167

### Relational analysis RSZ of RS_RSZ1_RSZ1_RSZ1_RSZ2_RSZ2_RSZ1_RSZ1

#### Relational analysis RSZ result of RS_RSZ1_RSZ1_RSZ1_RSZ2_RSZ2_RSZ1_RSZ1
Status: Status.UNKNOWN
Output dim: 7, lower bound: -0.0009614, upper bound: 0.0009189
time: 0.67 seconds

### Relational analysis RSZ of RS_RSZ1_RSZ1_RSZ1_RSZ2_RSZ2_RSZ1_RSZ2

#### Relational analysis RSZ result of RS_RSZ1_RSZ1_RSZ1_RSZ2_RSZ2_RSZ1_RSZ2
Status: Status.UNKNOWN
Output dim: 7, lower bound: -0.0009566, upper bound: 0.0009243
time: 0.64 seconds

## BFS RS instance: RS_RSZ1_RSZ1_RSZ1_RSZ2_RSZ2_RSZ2

### Backsubstitution after applying RS history:
0: -0.0027939, -0.0016242, -0.0027939, -0.0016242, -0.0007431, 0.0007598
1: -0.0114005, -0.0084323, -0.0114005, -0.0084323, -0.0018858, 0.0019282
2: 0.0279571, 0.0297986, 0.0279571, 0.0297986, -0.0011699, 0.0011962
3: 0.0039210, 0.0073596, 0.0039210, 0.0073596, -0.0022337, 0.0021846
4: -0.0104893, -0.0074701, -0.0104893, -0.0074701, -0.0019181, 0.0019613
5: 0.0097651, 0.0109087, 0.0097651, 0.0109087, -0.0007265, 0.0007429
6: 0.0053643, 0.0097284, 0.0053643, 0.0097284, -0.0028348, 0.0027725
7: 0.9818129, 0.9848667, 0.9818129, 0.9848667, -0.0019837, 0.0019401
8: -0.0060636, -0.0027895, -0.0060636, -0.0027895, -0.0021268, 0.0020800
9: -0.0031570, -0.0009942, -0.0031570, -0.0009942, -0.0013740, 0.0014049

### Unstable ReLU Count (Linear/Conv2D Layers)
- layer_idx=0, type=LayerType.Linear, total=256, inp1_unstable=10, inp2_unstable=10, delta_unstable=250
- layer_idx=2, type=LayerType.Linear, total=256, inp1_unstable=10, inp2_unstable=10, delta_unstable=256
- layer_idx=4, type=LayerType.Linear, total=256, inp1_unstable=0, inp2_unstable=0, delta_unstable=256
- layer_idx=6, type=LayerType.Linear, total=256, inp1_unstable=0, inp2_unstable=0, delta_unstable=256
- layer_idx=8, type=LayerType.Linear, total=10, inp1_unstable=0, inp2_unstable=0, delta_unstable=10

Time for backsubstitution: 1.56 seconds

### RS candidates at layer 1
type: RSZ, layer: 1, pos: 140
type: RSZ, layer: 1, pos: 167
type: RSZ, layer: 1, pos: 213
type: RSZ, layer: 1, pos: 219

Time for candidate selection: 0.17 seconds

### Candidate
type: RSZ, layer: 1, pos: 140

### Relational analysis RSZ of RS_RSZ1_RSZ1_RSZ1_RSZ2_RSZ2_RSZ2_RSZ1
Optimization infeasible because this subproblem isn't reachable.

### Candidate
type: RSZ, layer: 1, pos: 167

### Relational analysis RSZ of RS_RSZ1_RSZ1_RSZ1_RSZ2_RSZ2_RSZ2_RSZ1

#### Relational analysis RSZ result of RS_RSZ1_RSZ1_RSZ1_RSZ2_RSZ2_RSZ2_RSZ1
Status: Status.UNKNOWN
Output dim: 7, lower bound: -0.0009607, upper bound: 0.0009189
time: 0.65 seconds

### Relational analysis RSZ of RS_RSZ1_RSZ1_RSZ1_RSZ2_RSZ2_RSZ2_RSZ2

#### Relational analysis RSZ result of RS_RSZ1_RSZ1_RSZ1_RSZ2_RSZ2_RSZ2_RSZ2
Status: Status.UNKNOWN
Output dim: 7, lower bound: -0.0009559, upper bound: 0.0009244
time: 0.65 seconds

## BFS RS instance: RS_RSZ1_RSZ1_RSZ2_RSZ1_RSZ1_RSZ1

### Backsubstitution after applying RS history:
0: -0.0027939, -0.0016242, -0.0027939, -0.0016242, -0.0007519, 0.0007503
1: -0.0114005, -0.0084323, -0.0114005, -0.0084323, -0.0019081, 0.0019040
2: 0.0279571, 0.0297986, 0.0279571, 0.0297986, -0.0011838, 0.0011812
3: 0.0039210, 0.0073596, 0.0039210, 0.0073596, -0.0022056, 0.0022104
4: -0.0104893, -0.0074701, -0.0104893, -0.0074701, -0.0019409, 0.0019366
5: 0.0097651, 0.0109087, 0.0097651, 0.0109087, -0.0007351, 0.0007335
6: 0.0053643, 0.0097284, 0.0053643, 0.0097284, -0.0027992, 0.0028053
7: 0.9818129, 0.9848667, 0.9818129, 0.9848667, -0.0019588, 0.0019630
8: -0.0060636, -0.0027895, -0.0060636, -0.0027895, -0.0021001, 0.0021047
9: -0.0031570, -0.0009942, -0.0031570, -0.0009942, -0.0013903, 0.0013872

### Unstable ReLU Count (Linear/Conv2D Layers)
- layer_idx=0, type=LayerType.Linear, total=256, inp1_unstable=10, inp2_unstable=10, delta_unstable=250
- layer_idx=2, type=LayerType.Linear, total=256, inp1_unstable=10, inp2_unstable=10, delta_unstable=256
- layer_idx=4, type=LayerType.Linear, total=256, inp1_unstable=0, inp2_unstable=0, delta_unstable=256
- layer_idx=6, type=LayerType.Linear, total=256, inp1_unstable=0, inp2_unstable=0, delta_unstable=256
- layer_idx=8, type=LayerType.Linear, total=10, inp1_unstable=0, inp2_unstable=0, delta_unstable=10

Time for backsubstitution: 1.53 seconds

### RS candidates at layer 1
type: RSZ, layer: 1, pos: 140
type: RSZ, layer: 1, pos: 167
type: RSZ, layer: 1, pos: 213
type: RSZ, layer: 1, pos: 219

Time for candidate selection: 0.17 seconds

### Candidate
type: RSZ, layer: 1, pos: 140

### Relational analysis RSZ of RS_RSZ1_RSZ1_RSZ2_RSZ1_RSZ1_RSZ1_RSZ1
Optimization infeasible because this subproblem isn't reachable.

### Candidate
type: RSZ, layer: 1, pos: 167

### Relational analysis RSZ of RS_RSZ1_RSZ1_RSZ2_RSZ1_RSZ1_RSZ1_RSZ1

#### Relational analysis RSZ result of RS_RSZ1_RSZ1_RSZ2_RSZ1_RSZ1_RSZ1_RSZ1
Status: Status.UNKNOWN
Output dim: 7, lower bound: -0.0009496, upper bound: 0.0009333
time: 0.62 seconds

### Relational analysis RSZ of RS_RSZ1_RSZ1_RSZ2_RSZ1_RSZ1_RSZ1_RSZ2

#### Relational analysis RSZ result of RS_RSZ1_RSZ1_RSZ2_RSZ1_RSZ1_RSZ1_RSZ2
Status: Status.UNKNOWN
Output dim: 7, lower bound: -0.0009454, upper bound: 0.0009365
time: 0.64 seconds

## BFS RS instance: RS_RSZ1_RSZ1_RSZ2_RSZ1_RSZ1_RSZ2

### Backsubstitution after applying RS history:
0: -0.0027939, -0.0016242, -0.0027939, -0.0016242, -0.0007521, 0.0007501
1: -0.0114005, -0.0084323, -0.0114005, -0.0084323, -0.0019085, 0.0019034
2: 0.0279571, 0.0297986, 0.0279571, 0.0297986, -0.0011840, 0.0011809
3: 0.0039210, 0.0073596, 0.0039210, 0.0073596, -0.0022050, 0.0022109
4: -0.0104893, -0.0074701, -0.0104893, -0.0074701, -0.0019413, 0.0019361
5: 0.0097651, 0.0109087, 0.0097651, 0.0109087, -0.0007353, 0.0007333
6: 0.0053643, 0.0097284, 0.0053643, 0.0097284, -0.0027984, 0.0028059
7: 0.9818129, 0.9848667, 0.9818129, 0.9848667, -0.0019582, 0.0019634
8: -0.0060636, -0.0027895, -0.0060636, -0.0027895, -0.0020995, 0.0021051
9: -0.0031570, -0.0009942, -0.0031570, -0.0009942, -0.0013906, 0.0013868

### Unstable ReLU Count (Linear/Conv2D Layers)
- layer_idx=0, type=LayerType.Linear, total=256, inp1_unstable=10, inp2_unstable=10, delta_unstable=250
- layer_idx=2, type=LayerType.Linear, total=256, inp1_unstable=10, inp2_unstable=10, delta_unstable=256
- layer_idx=4, type=LayerType.Linear, total=256, inp1_unstable=0, inp2_unstable=0, delta_unstable=256
- layer_idx=6, type=LayerType.Linear, total=256, inp1_unstable=0, inp2_unstable=0, delta_unstable=256
- layer_idx=8, type=LayerType.Linear, total=10, inp1_unstable=0, inp2_unstable=0, delta_unstable=10

Time for backsubstitution: 1.50 seconds

### RS candidates at layer 1
type: RSZ, layer: 1, pos: 140
type: RSZ, layer: 1, pos: 167
type: RSZ, layer: 1, pos: 213
type: RSZ, layer: 1, pos: 219

Time for candidate selection: 0.17 seconds

### Candidate
type: RSZ, layer: 1, pos: 140

### Relational analysis RSZ of RS_RSZ1_RSZ1_RSZ2_RSZ1_RSZ1_RSZ2_RSZ1
Optimization infeasible because this subproblem isn't reachable.

### Candidate
type: RSZ, layer: 1, pos: 167

### Relational analysis RSZ of RS_RSZ1_RSZ1_RSZ2_RSZ1_RSZ1_RSZ2_RSZ1

#### Relational analysis RSZ result of RS_RSZ1_RSZ1_RSZ2_RSZ1_RSZ1_RSZ2_RSZ1
Status: Status.UNKNOWN
Output dim: 7, lower bound: -0.0009496, upper bound: 0.0009330
time: 0.65 seconds

### Relational analysis RSZ of RS_RSZ1_RSZ1_RSZ2_RSZ1_RSZ1_RSZ2_RSZ2

#### Relational analysis RSZ result of RS_RSZ1_RSZ1_RSZ2_RSZ1_RSZ1_RSZ2_RSZ2
Status: Status.UNKNOWN
Output dim: 7, lower bound: -0.0009452, upper bound: 0.0009362
time: 0.66 seconds

## BFS RS instance: RS_RSZ1_RSZ1_RSZ2_RSZ1_RSZ2_RSZ1

### Backsubstitution after applying RS history:
0: -0.0027939, -0.0016242, -0.0027939, -0.0016242, -0.0007371, 0.0007656
1: -0.0114005, -0.0084323, -0.0114005, -0.0084323, -0.0018704, 0.0019427
2: 0.0279571, 0.0297986, 0.0279571, 0.0297986, -0.0011604, 0.0012053
3: 0.0039210, 0.0073596, 0.0039210, 0.0073596, -0.0022506, 0.0021667
4: -0.0104893, -0.0074701, -0.0104893, -0.0074701, -0.0019025, 0.0019761
5: 0.0097651, 0.0109087, 0.0097651, 0.0109087, -0.0007206, 0.0007485
6: 0.0053643, 0.0097284, 0.0053643, 0.0097284, -0.0028563, 0.0027499
7: 0.9818129, 0.9848667, 0.9818129, 0.9848667, -0.0019987, 0.0019242
8: -0.0060636, -0.0027895, -0.0060636, -0.0027895, -0.0021429, 0.0020631
9: -0.0031570, -0.0009942, -0.0031570, -0.0009942, -0.0013628, 0.0014155

### Unstable ReLU Count (Linear/Conv2D Layers)
- layer_idx=0, type=LayerType.Linear, total=256, inp1_unstable=10, inp2_unstable=10, delta_unstable=250
- layer_idx=2, type=LayerType.Linear, total=256, inp1_unstable=10, inp2_unstable=10, delta_unstable=256
- layer_idx=4, type=LayerType.Linear, total=256, inp1_unstable=0, inp2_unstable=0, delta_unstable=256
- layer_idx=6, type=LayerType.Linear, total=256, inp1_unstable=0, inp2_unstable=0, delta_unstable=256
- layer_idx=8, type=LayerType.Linear, total=10, inp1_unstable=0, inp2_unstable=0, delta_unstable=10

Time for backsubstitution: 1.53 seconds

### RS candidates at layer 1
type: RSZ, layer: 1, pos: 140
type: RSZ, layer: 1, pos: 167
type: RSZ, layer: 1, pos: 213
type: RSZ, layer: 1, pos: 219

Time for candidate selection: 0.16 seconds

### Candidate
type: RSZ, layer: 1, pos: 140

### Relational analysis RSZ of RS_RSZ1_RSZ1_RSZ2_RSZ1_RSZ2_RSZ1_RSZ1
Optimization infeasible because this subproblem isn't reachable.

### Candidate
type: RSZ, layer: 1, pos: 167

### Relational analysis RSZ of RS_RSZ1_RSZ1_RSZ2_RSZ1_RSZ2_RSZ1_RSZ1

#### Relational analysis RSZ result of RS_RSZ1_RSZ1_RSZ2_RSZ1_RSZ2_RSZ1_RSZ1
Status: Status.UNKNOWN
Output dim: 7, lower bound: -0.0009605, upper bound: 0.0009247
time: 0.62 seconds

### Relational analysis RSZ of RS_RSZ1_RSZ1_RSZ2_RSZ1_RSZ2_RSZ1_RSZ2

#### Relational analysis RSZ result of RS_RSZ1_RSZ1_RSZ2_RSZ1_RSZ2_RSZ1_RSZ2
Status: Status.UNKNOWN
Output dim: 7, lower bound: -0.0009552, upper bound: 0.0009294
time: 0.62 seconds

## BFS RS instance: RS_RSZ1_RSZ1_RSZ2_RSZ1_RSZ2_RSZ2

### Backsubstitution after applying RS history:
0: -0.0027939, -0.0016242, -0.0027939, -0.0016242, -0.0007372, 0.0007655
1: -0.0114005, -0.0084323, -0.0114005, -0.0084323, -0.0018708, 0.0019426
2: 0.0279571, 0.0297986, 0.0279571, 0.0297986, -0.0011606, 0.0012052
3: 0.0039210, 0.0073596, 0.0039210, 0.0073596, -0.0022504, 0.0021672
4: -0.0104893, -0.0074701, -0.0104893, -0.0074701, -0.0019029, 0.0019759
5: 0.0097651, 0.0109087, 0.0097651, 0.0109087, -0.0007208, 0.0007484
6: 0.0053643, 0.0097284, 0.0053643, 0.0097284, -0.0028561, 0.0027505
7: 0.9818129, 0.9848667, 0.9818129, 0.9848667, -0.0019985, 0.0019247
8: -0.0060636, -0.0027895, -0.0060636, -0.0027895, -0.0021427, 0.0020635
9: -0.0031570, -0.0009942, -0.0031570, -0.0009942, -0.0013631, 0.0014154

### Unstable ReLU Count (Linear/Conv2D Layers)
- layer_idx=0, type=LayerType.Linear, total=256, inp1_unstable=10, inp2_unstable=10, delta_unstable=250
- layer_idx=2, type=LayerType.Linear, total=256, inp1_unstable=10, inp2_unstable=10, delta_unstable=256
- layer_idx=4, type=LayerType.Linear, total=256, inp1_unstable=0, inp2_unstable=0, delta_unstable=256
- layer_idx=6, type=LayerType.Linear, total=256, inp1_unstable=0, inp2_unstable=0, delta_unstable=256
- layer_idx=8, type=LayerType.Linear, total=10, inp1_unstable=0, inp2_unstable=0, delta_unstable=10

Time for backsubstitution: 1.62 seconds

### RS candidates at layer 1
type: RSZ, layer: 1, pos: 140
type: RSZ, layer: 1, pos: 167
type: RSZ, layer: 1, pos: 213
type: RSZ, layer: 1, pos: 219

Time for candidate selection: 0.18 seconds

### Candidate
type: RSZ, layer: 1, pos: 140

### Relational analysis RSZ of RS_RSZ1_RSZ1_RSZ2_RSZ1_RSZ2_RSZ2_RSZ1
Optimization infeasible because this subproblem isn't reachable.

### Candidate
type: RSZ, layer: 1, pos: 167

### Relational analysis RSZ of RS_RSZ1_RSZ1_RSZ2_RSZ1_RSZ2_RSZ2_RSZ1

#### Relational analysis RSZ result of RS_RSZ1_RSZ1_RSZ2_RSZ1_RSZ2_RSZ2_RSZ1
Status: Status.UNKNOWN
Output dim: 7, lower bound: -0.0009601, upper bound: 0.0009248
time: 0.64 seconds

### Relational analysis RSZ of RS_RSZ1_RSZ1_RSZ2_RSZ1_RSZ2_RSZ2_RSZ2

#### Relational analysis RSZ result of RS_RSZ1_RSZ1_RSZ2_RSZ1_RSZ2_RSZ2_RSZ2
Status: Status.UNKNOWN
Output dim: 7, lower bound: -0.0009546, upper bound: 0.0009297
time: 0.65 seconds

## BFS RS instance: RS_RSZ1_RSZ1_RSZ2_RSZ2_RSZ1_RSZ1

### Backsubstitution after applying RS history:
0: -0.0027939, -0.0016242, -0.0027939, -0.0016242, -0.0007591, 0.0007434
1: -0.0114005, -0.0084323, -0.0114005, -0.0084323, -0.0019263, 0.0018864
2: 0.0279571, 0.0297986, 0.0279571, 0.0297986, -0.0011951, 0.0011703
3: 0.0039210, 0.0073596, 0.0039210, 0.0073596, -0.0021853, 0.0022316
4: -0.0104893, -0.0074701, -0.0104893, -0.0074701, -0.0019594, 0.0019188
5: 0.0097651, 0.0109087, 0.0097651, 0.0109087, -0.0007422, 0.0007268
6: 0.0053643, 0.0097284, 0.0053643, 0.0097284, -0.0027734, 0.0028321
7: 0.9818129, 0.9848667, 0.9818129, 0.9848667, -0.0019407, 0.0019818
8: -0.0060636, -0.0027895, -0.0060636, -0.0027895, -0.0020807, 0.0021248
9: -0.0031570, -0.0009942, -0.0031570, -0.0009942, -0.0014036, 0.0013744

### Unstable ReLU Count (Linear/Conv2D Layers)
- layer_idx=0, type=LayerType.Linear, total=256, inp1_unstable=10, inp2_unstable=10, delta_unstable=250
- layer_idx=2, type=LayerType.Linear, total=256, inp1_unstable=10, inp2_unstable=10, delta_unstable=256
- layer_idx=4, type=LayerType.Linear, total=256, inp1_unstable=0, inp2_unstable=0, delta_unstable=256
- layer_idx=6, type=LayerType.Linear, total=256, inp1_unstable=0, inp2_unstable=0, delta_unstable=256
- layer_idx=8, type=LayerType.Linear, total=10, inp1_unstable=0, inp2_unstable=0, delta_unstable=10

Time for backsubstitution: 1.55 seconds

### RS candidates at layer 1
type: RSZ, layer: 1, pos: 140
type: RSZ, layer: 1, pos: 167
type: RSZ, layer: 1, pos: 213
type: RSZ, layer: 1, pos: 219

Time for candidate selection: 0.17 seconds

### Candidate
type: RSZ, layer: 1, pos: 140

### Relational analysis RSZ of RS_RSZ1_RSZ1_RSZ2_RSZ2_RSZ1_RSZ1_RSZ1
Optimization infeasible because this subproblem isn't reachable.

### Candidate
type: RSZ, layer: 1, pos: 167

### Relational analysis RSZ of RS_RSZ1_RSZ1_RSZ2_RSZ2_RSZ1_RSZ1_RSZ1

#### Relational analysis RSZ result of RS_RSZ1_RSZ1_RSZ2_RSZ2_RSZ1_RSZ1_RSZ1
Status: Status.UNKNOWN
Output dim: 7, lower bound: -0.0009498, upper bound: 0.0009292
time: 0.63 seconds

### Relational analysis RSZ of RS_RSZ1_RSZ1_RSZ2_RSZ2_RSZ1_RSZ1_RSZ2

#### Relational analysis RSZ result of RS_RSZ1_RSZ1_RSZ2_RSZ2_RSZ1_RSZ1_RSZ2
Status: Status.UNKNOWN
Output dim: 7, lower bound: -0.0009461, upper bound: 0.0009326
time: 0.66 seconds

## BFS RS instance: RS_RSZ1_RSZ1_RSZ2_RSZ2_RSZ1_RSZ2

### Backsubstitution after applying RS history:
0: -0.0027939, -0.0016242, -0.0027939, -0.0016242, -0.0007592, 0.0007432
1: -0.0114005, -0.0084323, -0.0114005, -0.0084323, -0.0019266, 0.0018860
2: 0.0279571, 0.0297986, 0.0279571, 0.0297986, -0.0011952, 0.0011701
3: 0.0039210, 0.0073596, 0.0039210, 0.0073596, -0.0021848, 0.0022318
4: -0.0104893, -0.0074701, -0.0104893, -0.0074701, -0.0019596, 0.0019183
5: 0.0097651, 0.0109087, 0.0097651, 0.0109087, -0.0007423, 0.0007266
6: 0.0053643, 0.0097284, 0.0053643, 0.0097284, -0.0027728, 0.0028325
7: 0.9818129, 0.9848667, 0.9818129, 0.9848667, -0.0019403, 0.0019820
8: -0.0060636, -0.0027895, -0.0060636, -0.0027895, -0.0020803, 0.0021250
9: -0.0031570, -0.0009942, -0.0031570, -0.0009942, -0.0014037, 0.0013741

### Unstable ReLU Count (Linear/Conv2D Layers)
- layer_idx=0, type=LayerType.Linear, total=256, inp1_unstable=10, inp2_unstable=10, delta_unstable=250
- layer_idx=2, type=LayerType.Linear, total=256, inp1_unstable=10, inp2_unstable=10, delta_unstable=256
- layer_idx=4, type=LayerType.Linear, total=256, inp1_unstable=0, inp2_unstable=0, delta_unstable=256
- layer_idx=6, type=LayerType.Linear, total=256, inp1_unstable=0, inp2_unstable=0, delta_unstable=256
- layer_idx=8, type=LayerType.Linear, total=10, inp1_unstable=0, inp2_unstable=0, delta_unstable=10

Time for backsubstitution: 1.54 seconds

### RS candidates at layer 1
type: RSZ, layer: 1, pos: 140
type: RSZ, layer: 1, pos: 167
type: RSZ, layer: 1, pos: 213
type: RSZ, layer: 1, pos: 219

Time for candidate selection: 0.18 seconds

### Candidate
type: RSZ, layer: 1, pos: 140

### Relational analysis RSZ of RS_RSZ1_RSZ1_RSZ2_RSZ2_RSZ1_RSZ2_RSZ1
Optimization infeasible because this subproblem isn't reachable.

### Candidate
type: RSZ, layer: 1, pos: 167

### Relational analysis RSZ of RS_RSZ1_RSZ1_RSZ2_RSZ2_RSZ1_RSZ2_RSZ1

#### Relational analysis RSZ result of RS_RSZ1_RSZ1_RSZ2_RSZ2_RSZ1_RSZ2_RSZ1
Status: Status.UNKNOWN
Output dim: 7, lower bound: -0.0009497, upper bound: 0.0009291
time: 0.77 seconds

### Relational analysis RSZ of RS_RSZ1_RSZ1_RSZ2_RSZ2_RSZ1_RSZ2_RSZ2

#### Relational analysis RSZ result of RS_RSZ1_RSZ1_RSZ2_RSZ2_RSZ1_RSZ2_RSZ2
Status: Status.UNKNOWN
Output dim: 7, lower bound: -0.0009458, upper bound: 0.0009327
time: 0.64 seconds

## BFS RS instance: RS_RSZ1_RSZ1_RSZ2_RSZ2_RSZ2_RSZ1

### Backsubstitution after applying RS history:
0: -0.0027939, -0.0016242, -0.0027939, -0.0016242, -0.0007446, 0.0007578
1: -0.0114005, -0.0084323, -0.0114005, -0.0084323, -0.0018895, 0.0019231
2: 0.0279571, 0.0297986, 0.0279571, 0.0297986, -0.0011723, 0.0011931
3: 0.0039210, 0.0073596, 0.0039210, 0.0073596, -0.0022278, 0.0021889
4: -0.0104893, -0.0074701, -0.0104893, -0.0074701, -0.0019220, 0.0019561
5: 0.0097651, 0.0109087, 0.0097651, 0.0109087, -0.0007280, 0.0007409
6: 0.0053643, 0.0097284, 0.0053643, 0.0097284, -0.0028273, 0.0027780
7: 0.9818129, 0.9848667, 0.9818129, 0.9848667, -0.0019784, 0.0019439
8: -0.0060636, -0.0027895, -0.0060636, -0.0027895, -0.0021212, 0.0020842
9: -0.0031570, -0.0009942, -0.0031570, -0.0009942, -0.0013767, 0.0014012

### Unstable ReLU Count (Linear/Conv2D Layers)
- layer_idx=0, type=LayerType.Linear, total=256, inp1_unstable=10, inp2_unstable=10, delta_unstable=250
- layer_idx=2, type=LayerType.Linear, total=256, inp1_unstable=10, inp2_unstable=10, delta_unstable=256
- layer_idx=4, type=LayerType.Linear, total=256, inp1_unstable=0, inp2_unstable=0, delta_unstable=256
- layer_idx=6, type=LayerType.Linear, total=256, inp1_unstable=0, inp2_unstable=0, delta_unstable=256
- layer_idx=8, type=LayerType.Linear, total=10, inp1_unstable=0, inp2_unstable=0, delta_unstable=10

Time for backsubstitution: 1.57 seconds

### RS candidates at layer 1
type: RSZ, layer: 1, pos: 140
type: RSZ, layer: 1, pos: 167
type: RSZ, layer: 1, pos: 213
type: RSZ, layer: 1, pos: 219

Time for candidate selection: 0.16 seconds

### Candidate
type: RSZ, layer: 1, pos: 140

### Relational analysis RSZ of RS_RSZ1_RSZ1_RSZ2_RSZ2_RSZ2_RSZ1_RSZ1
Optimization infeasible because this subproblem isn't reachable.

### Candidate
type: RSZ, layer: 1, pos: 167

### Relational analysis RSZ of RS_RSZ1_RSZ1_RSZ2_RSZ2_RSZ2_RSZ1_RSZ1

#### Relational analysis RSZ result of RS_RSZ1_RSZ1_RSZ2_RSZ2_RSZ2_RSZ1_RSZ1
Status: Status.UNKNOWN
Output dim: 7, lower bound: -0.0009614, upper bound: 0.0009189
time: 0.66 seconds

### Relational analysis RSZ of RS_RSZ1_RSZ1_RSZ2_RSZ2_RSZ2_RSZ1_RSZ2

#### Relational analysis RSZ result of RS_RSZ1_RSZ1_RSZ2_RSZ2_RSZ2_RSZ1_RSZ2
Status: Status.UNKNOWN
Output dim: 7, lower bound: -0.0009566, upper bound: 0.0009243
time: 0.66 seconds

## BFS RS instance: RS_RSZ1_RSZ1_RSZ2_RSZ2_RSZ2_RSZ2

### Backsubstitution after applying RS history:
0: -0.0027939, -0.0016242, -0.0027939, -0.0016242, -0.0007448, 0.0007577
1: -0.0114005, -0.0084323, -0.0114005, -0.0084323, -0.0018899, 0.0019227
2: 0.0279571, 0.0297986, 0.0279571, 0.0297986, -0.0011725, 0.0011928
3: 0.0039210, 0.0073596, 0.0039210, 0.0073596, -0.0022273, 0.0021894
4: -0.0104893, -0.0074701, -0.0104893, -0.0074701, -0.0019224, 0.0019557
5: 0.0097651, 0.0109087, 0.0097651, 0.0109087, -0.0007281, 0.0007408
6: 0.0053643, 0.0097284, 0.0053643, 0.0097284, -0.0028267, 0.0027786
7: 0.9818129, 0.9848667, 0.9818129, 0.9848667, -0.0019780, 0.0019443
8: -0.0060636, -0.0027895, -0.0060636, -0.0027895, -0.0021207, 0.0020846
9: -0.0031570, -0.0009942, -0.0031570, -0.0009942, -0.0013770, 0.0014009

### Unstable ReLU Count (Linear/Conv2D Layers)
- layer_idx=0, type=LayerType.Linear, total=256, inp1_unstable=10, inp2_unstable=10, delta_unstable=250
- layer_idx=2, type=LayerType.Linear, total=256, inp1_unstable=10, inp2_unstable=10, delta_unstable=256
- layer_idx=4, type=LayerType.Linear, total=256, inp1_unstable=0, inp2_unstable=0, delta_unstable=256
- layer_idx=6, type=LayerType.Linear, total=256, inp1_unstable=0, inp2_unstable=0, delta_unstable=256
- layer_idx=8, type=LayerType.Linear, total=10, inp1_unstable=0, inp2_unstable=0, delta_unstable=10

Time for backsubstitution: 1.53 seconds

### RS candidates at layer 1
type: RSZ, layer: 1, pos: 140
type: RSZ, layer: 1, pos: 167
type: RSZ, layer: 1, pos: 213
type: RSZ, layer: 1, pos: 219

Time for candidate selection: 0.16 seconds

### Candidate
type: RSZ, layer: 1, pos: 140

### Relational analysis RSZ of RS_RSZ1_RSZ1_RSZ2_RSZ2_RSZ2_RSZ2_RSZ1
Optimization infeasible because this subproblem isn't reachable.

### Candidate
type: RSZ, layer: 1, pos: 167

### Relational analysis RSZ of RS_RSZ1_RSZ1_RSZ2_RSZ2_RSZ2_RSZ2_RSZ1

#### Relational analysis RSZ result of RS_RSZ1_RSZ1_RSZ2_RSZ2_RSZ2_RSZ2_RSZ1
Status: Status.UNKNOWN
Output dim: 7, lower bound: -0.0009607, upper bound: 0.0009189
time: 0.62 seconds

### Relational analysis RSZ of RS_RSZ1_RSZ1_RSZ2_RSZ2_RSZ2_RSZ2_RSZ2

#### Relational analysis RSZ result of RS_RSZ1_RSZ1_RSZ2_RSZ2_RSZ2_RSZ2_RSZ2
Status: Status.UNKNOWN
Output dim: 7, lower bound: -0.0009559, upper bound: 0.0009244
time: 0.64 seconds

## BFS RS instance: RS_RSZ1_RSZ2_RSZ1_RSZ1_RSZ1_RSZ1

### Backsubstitution after applying RS history:
0: -0.0027939, -0.0016242, -0.0027939, -0.0016242, -0.0007565, 0.0007473
1: -0.0114005, -0.0084323, -0.0114005, -0.0084323, -0.0019197, 0.0018963
2: 0.0279571, 0.0297986, 0.0279571, 0.0297986, -0.0011910, 0.0011765
3: 0.0039210, 0.0073596, 0.0039210, 0.0073596, -0.0021968, 0.0022239
4: -0.0104893, -0.0074701, -0.0104893, -0.0074701, -0.0019527, 0.0019289
5: 0.0097651, 0.0109087, 0.0097651, 0.0109087, -0.0007396, 0.0007306
6: 0.0053643, 0.0097284, 0.0053643, 0.0097284, -0.0027880, 0.0028224
7: 0.9818129, 0.9848667, 0.9818129, 0.9848667, -0.0019509, 0.0019750
8: -0.0060636, -0.0027895, -0.0060636, -0.0027895, -0.0020917, 0.0021175
9: -0.0031570, -0.0009942, -0.0031570, -0.0009942, -0.0013987, 0.0013817

### Unstable ReLU Count (Linear/Conv2D Layers)
- layer_idx=0, type=LayerType.Linear, total=256, inp1_unstable=10, inp2_unstable=10, delta_unstable=250
- layer_idx=2, type=LayerType.Linear, total=256, inp1_unstable=10, inp2_unstable=10, delta_unstable=256
- layer_idx=4, type=LayerType.Linear, total=256, inp1_unstable=0, inp2_unstable=0, delta_unstable=256
- layer_idx=6, type=LayerType.Linear, total=256, inp1_unstable=0, inp2_unstable=0, delta_unstable=256
- layer_idx=8, type=LayerType.Linear, total=10, inp1_unstable=0, inp2_unstable=0, delta_unstable=10

Time for backsubstitution: 1.56 seconds

### RS candidates at layer 1
type: RSZ, layer: 1, pos: 140
type: RSZ, layer: 1, pos: 167
type: RSZ, layer: 1, pos: 213
type: RSZ, layer: 1, pos: 219

Time for candidate selection: 0.17 seconds

### Candidate
type: RSZ, layer: 1, pos: 140

### Relational analysis RSZ of RS_RSZ1_RSZ2_RSZ1_RSZ1_RSZ1_RSZ1_RSZ1
Optimization infeasible because this subproblem isn't reachable.

### Candidate
type: RSZ, layer: 1, pos: 167

### Relational analysis RSZ of RS_RSZ1_RSZ2_RSZ1_RSZ1_RSZ1_RSZ1_RSZ1

#### Relational analysis RSZ result of RS_RSZ1_RSZ2_RSZ1_RSZ1_RSZ1_RSZ1_RSZ1
Status: Status.UNKNOWN
Output dim: 7, lower bound: -0.0009336, upper bound: 0.0009512
time: 0.63 seconds

### Relational analysis RSZ of RS_RSZ1_RSZ2_RSZ1_RSZ1_RSZ1_RSZ1_RSZ2

#### Relational analysis RSZ result of RS_RSZ1_RSZ2_RSZ1_RSZ1_RSZ1_RSZ1_RSZ2
Status: Status.UNKNOWN
Output dim: 7, lower bound: -0.0009246, upper bound: 0.0009541
time: 0.65 seconds

## BFS RS instance: RS_RSZ1_RSZ2_RSZ1_RSZ1_RSZ1_RSZ2

### Backsubstitution after applying RS history:
0: -0.0027939, -0.0016242, -0.0027939, -0.0016242, -0.0007566, 0.0007471
1: -0.0114005, -0.0084323, -0.0114005, -0.0084323, -0.0019201, 0.0018959
2: 0.0279571, 0.0297986, 0.0279571, 0.0297986, -0.0011912, 0.0011762
3: 0.0039210, 0.0073596, 0.0039210, 0.0073596, -0.0021963, 0.0022243
4: -0.0104893, -0.0074701, -0.0104893, -0.0074701, -0.0019531, 0.0019284
5: 0.0097651, 0.0109087, 0.0097651, 0.0109087, -0.0007398, 0.0007304
6: 0.0053643, 0.0097284, 0.0053643, 0.0097284, -0.0027873, 0.0028230
7: 0.9818129, 0.9848667, 0.9818129, 0.9848667, -0.0019505, 0.0019754
8: -0.0060636, -0.0027895, -0.0060636, -0.0027895, -0.0020912, 0.0021179
9: -0.0031570, -0.0009942, -0.0031570, -0.0009942, -0.0013990, 0.0013813

### Unstable ReLU Count (Linear/Conv2D Layers)
- layer_idx=0, type=LayerType.Linear, total=256, inp1_unstable=10, inp2_unstable=10, delta_unstable=250
- layer_idx=2, type=LayerType.Linear, total=256, inp1_unstable=10, inp2_unstable=10, delta_unstable=256
- layer_idx=4, type=LayerType.Linear, total=256, inp1_unstable=0, inp2_unstable=0, delta_unstable=256
- layer_idx=6, type=LayerType.Linear, total=256, inp1_unstable=0, inp2_unstable=0, delta_unstable=256
- layer_idx=8, type=LayerType.Linear, total=10, inp1_unstable=0, inp2_unstable=0, delta_unstable=10

Time for backsubstitution: 1.67 seconds

### RS candidates at layer 1
type: RSZ, layer: 1, pos: 140
type: RSZ, layer: 1, pos: 167
type: RSZ, layer: 1, pos: 213
type: RSZ, layer: 1, pos: 219

Time for candidate selection: 0.17 seconds

### Candidate
type: RSZ, layer: 1, pos: 140

### Relational analysis RSZ of RS_RSZ1_RSZ2_RSZ1_RSZ1_RSZ1_RSZ2_RSZ1
Optimization infeasible because this subproblem isn't reachable.

### Candidate
type: RSZ, layer: 1, pos: 167

### Relational analysis RSZ of RS_RSZ1_RSZ2_RSZ1_RSZ1_RSZ1_RSZ2_RSZ1

#### Relational analysis RSZ result of RS_RSZ1_RSZ2_RSZ1_RSZ1_RSZ1_RSZ2_RSZ1
Status: Status.UNKNOWN
Output dim: 7, lower bound: -0.0009336, upper bound: 0.0009521
time: 0.66 seconds

### Relational analysis RSZ of RS_RSZ1_RSZ2_RSZ1_RSZ1_RSZ1_RSZ2_RSZ2

#### Relational analysis RSZ result of RS_RSZ1_RSZ2_RSZ1_RSZ1_RSZ1_RSZ2_RSZ2
Status: Status.UNKNOWN
Output dim: 7, lower bound: -0.0009246, upper bound: 0.0009553
time: 0.67 seconds

## BFS RS instance: RS_RSZ1_RSZ2_RSZ1_RSZ1_RSZ2_RSZ1

### Backsubstitution after applying RS history:
0: -0.0027939, -0.0016242, -0.0027939, -0.0016242, -0.0007419, 0.0007617
1: -0.0114005, -0.0084323, -0.0114005, -0.0084323, -0.0018828, 0.0019330
2: 0.0279571, 0.0297986, 0.0279571, 0.0297986, -0.0011681, 0.0011992
3: 0.0039210, 0.0073596, 0.0039210, 0.0073596, -0.0022393, 0.0021811
4: -0.0104893, -0.0074701, -0.0104893, -0.0074701, -0.0019151, 0.0019662
5: 0.0097651, 0.0109087, 0.0097651, 0.0109087, -0.0007254, 0.0007447
6: 0.0053643, 0.0097284, 0.0053643, 0.0097284, -0.0028419, 0.0027681
7: 0.9818129, 0.9848667, 0.9818129, 0.9848667, -0.0019886, 0.0019370
8: -0.0060636, -0.0027895, -0.0060636, -0.0027895, -0.0021321, 0.0020768
9: -0.0031570, -0.0009942, -0.0031570, -0.0009942, -0.0013718, 0.0014084

### Unstable ReLU Count (Linear/Conv2D Layers)
- layer_idx=0, type=LayerType.Linear, total=256, inp1_unstable=10, inp2_unstable=10, delta_unstable=250
- layer_idx=2, type=LayerType.Linear, total=256, inp1_unstable=10, inp2_unstable=10, delta_unstable=256
- layer_idx=4, type=LayerType.Linear, total=256, inp1_unstable=0, inp2_unstable=0, delta_unstable=256
- layer_idx=6, type=LayerType.Linear, total=256, inp1_unstable=0, inp2_unstable=0, delta_unstable=256
- layer_idx=8, type=LayerType.Linear, total=10, inp1_unstable=0, inp2_unstable=0, delta_unstable=10

Time for backsubstitution: 1.60 seconds

### RS candidates at layer 1
type: RSZ, layer: 1, pos: 140
type: RSZ, layer: 1, pos: 167
type: RSZ, layer: 1, pos: 213
type: RSZ, layer: 1, pos: 219

Time for candidate selection: 0.17 seconds

### Candidate
type: RSZ, layer: 1, pos: 140

### Relational analysis RSZ of RS_RSZ1_RSZ2_RSZ1_RSZ1_RSZ2_RSZ1_RSZ1
Optimization infeasible because this subproblem isn't reachable.

### Candidate
type: RSZ, layer: 1, pos: 167

### Relational analysis RSZ of RS_RSZ1_RSZ2_RSZ1_RSZ1_RSZ2_RSZ1_RSZ1

#### Relational analysis RSZ result of RS_RSZ1_RSZ2_RSZ1_RSZ1_RSZ2_RSZ1_RSZ1
Status: Status.UNKNOWN
Output dim: 7, lower bound: -0.0009414, upper bound: 0.0009401
time: 0.66 seconds

### Relational analysis RSZ of RS_RSZ1_RSZ2_RSZ1_RSZ1_RSZ2_RSZ1_RSZ2

#### Relational analysis RSZ result of RS_RSZ1_RSZ2_RSZ1_RSZ1_RSZ2_RSZ1_RSZ2
Status: Status.UNKNOWN
Output dim: 7, lower bound: -0.0009337, upper bound: 0.0009419
time: 0.66 seconds

## BFS RS instance: RS_RSZ1_RSZ2_RSZ1_RSZ1_RSZ2_RSZ2

### Backsubstitution after applying RS history:
0: -0.0027939, -0.0016242, -0.0027939, -0.0016242, -0.0007421, 0.0007617
1: -0.0114005, -0.0084323, -0.0114005, -0.0084323, -0.0018833, 0.0019329
2: 0.0279571, 0.0297986, 0.0279571, 0.0297986, -0.0011684, 0.0011992
3: 0.0039210, 0.0073596, 0.0039210, 0.0073596, -0.0022391, 0.0021817
4: -0.0104893, -0.0074701, -0.0104893, -0.0074701, -0.0019156, 0.0019660
5: 0.0097651, 0.0109087, 0.0097651, 0.0109087, -0.0007256, 0.0007447
6: 0.0053643, 0.0097284, 0.0053643, 0.0097284, -0.0028417, 0.0027688
7: 0.9818129, 0.9848667, 0.9818129, 0.9848667, -0.0019885, 0.0019375
8: -0.0060636, -0.0027895, -0.0060636, -0.0027895, -0.0021320, 0.0020773
9: -0.0031570, -0.0009942, -0.0031570, -0.0009942, -0.0013722, 0.0014083

### Unstable ReLU Count (Linear/Conv2D Layers)
- layer_idx=0, type=LayerType.Linear, total=256, inp1_unstable=10, inp2_unstable=10, delta_unstable=250
- layer_idx=2, type=LayerType.Linear, total=256, inp1_unstable=10, inp2_unstable=10, delta_unstable=256
- layer_idx=4, type=LayerType.Linear, total=256, inp1_unstable=0, inp2_unstable=0, delta_unstable=256
- layer_idx=6, type=LayerType.Linear, total=256, inp1_unstable=0, inp2_unstable=0, delta_unstable=256
- layer_idx=8, type=LayerType.Linear, total=10, inp1_unstable=0, inp2_unstable=0, delta_unstable=10

Time for backsubstitution: 1.60 seconds

### RS candidates at layer 1
type: RSZ, layer: 1, pos: 140
type: RSZ, layer: 1, pos: 167
type: RSZ, layer: 1, pos: 213
type: RSZ, layer: 1, pos: 219

Time for candidate selection: 0.16 seconds

### Candidate
type: RSZ, layer: 1, pos: 140

### Relational analysis RSZ of RS_RSZ1_RSZ2_RSZ1_RSZ1_RSZ2_RSZ2_RSZ1
Optimization infeasible because this subproblem isn't reachable.

### Candidate
type: RSZ, layer: 1, pos: 167

### Relational analysis RSZ of RS_RSZ1_RSZ2_RSZ1_RSZ1_RSZ2_RSZ2_RSZ1

#### Relational analysis RSZ result of RS_RSZ1_RSZ2_RSZ1_RSZ1_RSZ2_RSZ2_RSZ1
Status: Status.UNKNOWN
Output dim: 7, lower bound: -0.0009415, upper bound: 0.0009410
time: 0.63 seconds

### Relational analysis RSZ of RS_RSZ1_RSZ2_RSZ1_RSZ1_RSZ2_RSZ2_RSZ2

#### Relational analysis RSZ result of RS_RSZ1_RSZ2_RSZ1_RSZ1_RSZ2_RSZ2_RSZ2
Status: Status.UNKNOWN
Output dim: 7, lower bound: -0.0009341, upper bound: 0.0009426
time: 0.65 seconds

## BFS RS instance: RS_RSZ1_RSZ2_RSZ1_RSZ2_RSZ1_RSZ1

### Backsubstitution after applying RS history:
0: -0.0027939, -0.0016242, -0.0027939, -0.0016242, -0.0007634, 0.0007386
1: -0.0114005, -0.0084323, -0.0114005, -0.0084323, -0.0019373, 0.0018743
2: 0.0279571, 0.0297986, 0.0279571, 0.0297986, -0.0012019, 0.0011628
3: 0.0039210, 0.0073596, 0.0039210, 0.0073596, -0.0021713, 0.0022442
4: -0.0104893, -0.0074701, -0.0104893, -0.0074701, -0.0019705, 0.0019065
5: 0.0097651, 0.0109087, 0.0097651, 0.0109087, -0.0007464, 0.0007221
6: 0.0053643, 0.0097284, 0.0053643, 0.0097284, -0.0027557, 0.0028482
7: 0.9818129, 0.9848667, 0.9818129, 0.9848667, -0.0019283, 0.0019930
8: -0.0060636, -0.0027895, -0.0060636, -0.0027895, -0.0020674, 0.0021368
9: -0.0031570, -0.0009942, -0.0031570, -0.0009942, -0.0014115, 0.0013656

### Unstable ReLU Count (Linear/Conv2D Layers)
- layer_idx=0, type=LayerType.Linear, total=256, inp1_unstable=10, inp2_unstable=10, delta_unstable=250
- layer_idx=2, type=LayerType.Linear, total=256, inp1_unstable=10, inp2_unstable=10, delta_unstable=256
- layer_idx=4, type=LayerType.Linear, total=256, inp1_unstable=0, inp2_unstable=0, delta_unstable=256
- layer_idx=6, type=LayerType.Linear, total=256, inp1_unstable=0, inp2_unstable=0, delta_unstable=256
- layer_idx=8, type=LayerType.Linear, total=10, inp1_unstable=0, inp2_unstable=0, delta_unstable=10

Time for backsubstitution: 1.55 seconds

### RS candidates at layer 1
type: RSZ, layer: 1, pos: 140
type: RSZ, layer: 1, pos: 167
type: RSZ, layer: 1, pos: 213
type: RSZ, layer: 1, pos: 219

Time for candidate selection: 0.16 seconds

### Candidate
type: RSZ, layer: 1, pos: 140

### Relational analysis RSZ of RS_RSZ1_RSZ2_RSZ1_RSZ2_RSZ1_RSZ1_RSZ1
Optimization infeasible because this subproblem isn't reachable.

### Candidate
type: RSZ, layer: 1, pos: 167

### Relational analysis RSZ of RS_RSZ1_RSZ2_RSZ1_RSZ2_RSZ1_RSZ1_RSZ1

#### Relational analysis RSZ result of RS_RSZ1_RSZ2_RSZ1_RSZ2_RSZ1_RSZ1_RSZ1
Status: Status.UNKNOWN
Output dim: 7, lower bound: -0.0009341, upper bound: 0.0009428
time: 0.63 seconds

### Relational analysis RSZ of RS_RSZ1_RSZ2_RSZ1_RSZ2_RSZ1_RSZ1_RSZ2

#### Relational analysis RSZ result of RS_RSZ1_RSZ2_RSZ1_RSZ2_RSZ1_RSZ1_RSZ2
Status: Status.UNKNOWN
Output dim: 7, lower bound: -0.0009266, upper bound: 0.0009485
time: 0.65 seconds

## BFS RS instance: RS_RSZ1_RSZ2_RSZ1_RSZ2_RSZ1_RSZ2

### Backsubstitution after applying RS history:
0: -0.0027939, -0.0016242, -0.0027939, -0.0016242, -0.0007635, 0.0007385
1: -0.0114005, -0.0084323, -0.0114005, -0.0084323, -0.0019374, 0.0018740
2: 0.0279571, 0.0297986, 0.0279571, 0.0297986, -0.0012020, 0.0011626
3: 0.0039210, 0.0073596, 0.0039210, 0.0073596, -0.0021709, 0.0022444
4: -0.0104893, -0.0074701, -0.0104893, -0.0074701, -0.0019707, 0.0019061
5: 0.0097651, 0.0109087, 0.0097651, 0.0109087, -0.0007464, 0.0007220
6: 0.0053643, 0.0097284, 0.0053643, 0.0097284, -0.0027552, 0.0028484
7: 0.9818129, 0.9848667, 0.9818129, 0.9848667, -0.0019279, 0.0019932
8: -0.0060636, -0.0027895, -0.0060636, -0.0027895, -0.0020670, 0.0021370
9: -0.0031570, -0.0009942, -0.0031570, -0.0009942, -0.0014116, 0.0013654

### Unstable ReLU Count (Linear/Conv2D Layers)
- layer_idx=0, type=LayerType.Linear, total=256, inp1_unstable=10, inp2_unstable=10, delta_unstable=250
- layer_idx=2, type=LayerType.Linear, total=256, inp1_unstable=10, inp2_unstable=10, delta_unstable=256
- layer_idx=4, type=LayerType.Linear, total=256, inp1_unstable=0, inp2_unstable=0, delta_unstable=256
- layer_idx=6, type=LayerType.Linear, total=256, inp1_unstable=0, inp2_unstable=0, delta_unstable=256
- layer_idx=8, type=LayerType.Linear, total=10, inp1_unstable=0, inp2_unstable=0, delta_unstable=10

Time for backsubstitution: 1.63 seconds

### RS candidates at layer 1
type: RSZ, layer: 1, pos: 140
type: RSZ, layer: 1, pos: 167
type: RSZ, layer: 1, pos: 213
type: RSZ, layer: 1, pos: 219

Time for candidate selection: 0.17 seconds

### Candidate
type: RSZ, layer: 1, pos: 140

### Relational analysis RSZ of RS_RSZ1_RSZ2_RSZ1_RSZ2_RSZ1_RSZ2_RSZ1
Optimization infeasible because this subproblem isn't reachable.

### Candidate
type: RSZ, layer: 1, pos: 167

### Relational analysis RSZ of RS_RSZ1_RSZ2_RSZ1_RSZ2_RSZ1_RSZ2_RSZ1

#### Relational analysis RSZ result of RS_RSZ1_RSZ2_RSZ1_RSZ2_RSZ1_RSZ2_RSZ1
Status: Status.UNKNOWN
Output dim: 7, lower bound: -0.0009342, upper bound: 0.0009445
time: 0.63 seconds

### Relational analysis RSZ of RS_RSZ1_RSZ2_RSZ1_RSZ2_RSZ1_RSZ2_RSZ2

#### Relational analysis RSZ result of RS_RSZ1_RSZ2_RSZ1_RSZ2_RSZ1_RSZ2_RSZ2
Status: Status.UNKNOWN
Output dim: 7, lower bound: -0.0009266, upper bound: 0.0009504
time: 0.62 seconds

## BFS RS instance: RS_RSZ1_RSZ2_RSZ1_RSZ2_RSZ2_RSZ1

### Backsubstitution after applying RS history:
0: -0.0027939, -0.0016242, -0.0027939, -0.0016242, -0.0007479, 0.0007537
1: -0.0114005, -0.0084323, -0.0114005, -0.0084323, -0.0018978, 0.0019125
2: 0.0279571, 0.0297986, 0.0279571, 0.0297986, -0.0011774, 0.0011866
3: 0.0039210, 0.0073596, 0.0039210, 0.0073596, -0.0022156, 0.0021985
4: -0.0104893, -0.0074701, -0.0104893, -0.0074701, -0.0019304, 0.0019454
5: 0.0097651, 0.0109087, 0.0097651, 0.0109087, -0.0007312, 0.0007369
6: 0.0053643, 0.0097284, 0.0053643, 0.0097284, -0.0028119, 0.0027902
7: 0.9818129, 0.9848667, 0.9818129, 0.9848667, -0.0019676, 0.0019525
8: -0.0060636, -0.0027895, -0.0060636, -0.0027895, -0.0021096, 0.0020933
9: -0.0031570, -0.0009942, -0.0031570, -0.0009942, -0.0013828, 0.0013935

### Unstable ReLU Count (Linear/Conv2D Layers)
- layer_idx=0, type=LayerType.Linear, total=256, inp1_unstable=10, inp2_unstable=10, delta_unstable=250
- layer_idx=2, type=LayerType.Linear, total=256, inp1_unstable=10, inp2_unstable=10, delta_unstable=256
- layer_idx=4, type=LayerType.Linear, total=256, inp1_unstable=0, inp2_unstable=0, delta_unstable=256
- layer_idx=6, type=LayerType.Linear, total=256, inp1_unstable=0, inp2_unstable=0, delta_unstable=256
- layer_idx=8, type=LayerType.Linear, total=10, inp1_unstable=0, inp2_unstable=0, delta_unstable=10

Time for backsubstitution: 1.56 seconds

### RS candidates at layer 1
type: RSZ, layer: 1, pos: 140
type: RSZ, layer: 1, pos: 167
type: RSZ, layer: 1, pos: 213
type: RSZ, layer: 1, pos: 219

Time for candidate selection: 0.17 seconds

### Candidate
type: RSZ, layer: 1, pos: 140

### Relational analysis RSZ of RS_RSZ1_RSZ2_RSZ1_RSZ2_RSZ2_RSZ1_RSZ1
Optimization infeasible because this subproblem isn't reachable.

### Candidate
type: RSZ, layer: 1, pos: 167

### Relational analysis RSZ of RS_RSZ1_RSZ2_RSZ1_RSZ2_RSZ2_RSZ1_RSZ1

#### Relational analysis RSZ result of RS_RSZ1_RSZ2_RSZ1_RSZ2_RSZ2_RSZ1_RSZ1
Status: Status.UNKNOWN
Output dim: 7, lower bound: -0.0009414, upper bound: 0.0009284
time: 0.65 seconds

### Relational analysis RSZ of RS_RSZ1_RSZ2_RSZ1_RSZ2_RSZ2_RSZ1_RSZ2

#### Relational analysis RSZ result of RS_RSZ1_RSZ2_RSZ1_RSZ2_RSZ2_RSZ1_RSZ2
Status: Status.UNKNOWN
Output dim: 7, lower bound: -0.0009349, upper bound: 0.0009322
time: 0.63 seconds

## BFS RS instance: RS_RSZ1_RSZ2_RSZ1_RSZ2_RSZ2_RSZ2

### Backsubstitution after applying RS history:
0: -0.0027939, -0.0016242, -0.0027939, -0.0016242, -0.0007481, 0.0007535
1: -0.0114005, -0.0084323, -0.0114005, -0.0084323, -0.0018984, 0.0019122
2: 0.0279571, 0.0297986, 0.0279571, 0.0297986, -0.0011778, 0.0011863
3: 0.0039210, 0.0073596, 0.0039210, 0.0073596, -0.0022152, 0.0021992
4: -0.0104893, -0.0074701, -0.0104893, -0.0074701, -0.0019310, 0.0019450
5: 0.0097651, 0.0109087, 0.0097651, 0.0109087, -0.0007314, 0.0007367
6: 0.0053643, 0.0097284, 0.0053643, 0.0097284, -0.0028113, 0.0027911
7: 0.9818129, 0.9848667, 0.9818129, 0.9848667, -0.0019672, 0.0019530
8: -0.0060636, -0.0027895, -0.0060636, -0.0027895, -0.0021092, 0.0020940
9: -0.0031570, -0.0009942, -0.0031570, -0.0009942, -0.0013832, 0.0013932

### Unstable ReLU Count (Linear/Conv2D Layers)
- layer_idx=0, type=LayerType.Linear, total=256, inp1_unstable=10, inp2_unstable=10, delta_unstable=250
- layer_idx=2, type=LayerType.Linear, total=256, inp1_unstable=10, inp2_unstable=10, delta_unstable=256
- layer_idx=4, type=LayerType.Linear, total=256, inp1_unstable=0, inp2_unstable=0, delta_unstable=256
- layer_idx=6, type=LayerType.Linear, total=256, inp1_unstable=0, inp2_unstable=0, delta_unstable=256
- layer_idx=8, type=LayerType.Linear, total=10, inp1_unstable=0, inp2_unstable=0, delta_unstable=10

Time for backsubstitution: 1.52 seconds

### RS candidates at layer 1
type: RSZ, layer: 1, pos: 140
type: RSZ, layer: 1, pos: 167
type: RSZ, layer: 1, pos: 213
type: RSZ, layer: 1, pos: 219

Time for candidate selection: 0.19 seconds

### Candidate
type: RSZ, layer: 1, pos: 140

### Relational analysis RSZ of RS_RSZ1_RSZ2_RSZ1_RSZ2_RSZ2_RSZ2_RSZ1
Optimization infeasible because this subproblem isn't reachable.

### Candidate
type: RSZ, layer: 1, pos: 167

### Relational analysis RSZ of RS_RSZ1_RSZ2_RSZ1_RSZ2_RSZ2_RSZ2_RSZ1

#### Relational analysis RSZ result of RS_RSZ1_RSZ2_RSZ1_RSZ2_RSZ2_RSZ2_RSZ1
Status: Status.UNKNOWN
Output dim: 7, lower bound: -0.0009420, upper bound: 0.0009291
time: 0.64 seconds

### Relational analysis RSZ of RS_RSZ1_RSZ2_RSZ1_RSZ2_RSZ2_RSZ2_RSZ2

#### Relational analysis RSZ result of RS_RSZ1_RSZ2_RSZ1_RSZ2_RSZ2_RSZ2_RSZ2
Status: Status.UNKNOWN
Output dim: 7, lower bound: -0.0009355, upper bound: 0.0009327
time: 0.65 seconds

## BFS RS instance: RS_RSZ1_RSZ2_RSZ2_RSZ1_RSZ1_RSZ1

### Backsubstitution after applying RS history:
0: -0.0027939, -0.0016242, -0.0027939, -0.0016242, -0.0007581, 0.0007453
1: -0.0114005, -0.0084323, -0.0114005, -0.0084323, -0.0019239, 0.0018913
2: 0.0279571, 0.0297986, 0.0279571, 0.0297986, -0.0011936, 0.0011734
3: 0.0039210, 0.0073596, 0.0039210, 0.0073596, -0.0021910, 0.0022287
4: -0.0104893, -0.0074701, -0.0104893, -0.0074701, -0.0019569, 0.0019238
5: 0.0097651, 0.0109087, 0.0097651, 0.0109087, -0.0007412, 0.0007287
6: 0.0053643, 0.0097284, 0.0053643, 0.0097284, -0.0027807, 0.0028285
7: 0.9818129, 0.9848667, 0.9818129, 0.9848667, -0.0019458, 0.0019793
8: -0.0060636, -0.0027895, -0.0060636, -0.0027895, -0.0020862, 0.0021221
9: -0.0031570, -0.0009942, -0.0031570, -0.0009942, -0.0014018, 0.0013780

### Unstable ReLU Count (Linear/Conv2D Layers)
- layer_idx=0, type=LayerType.Linear, total=256, inp1_unstable=10, inp2_unstable=10, delta_unstable=250
- layer_idx=2, type=LayerType.Linear, total=256, inp1_unstable=10, inp2_unstable=10, delta_unstable=256
- layer_idx=4, type=LayerType.Linear, total=256, inp1_unstable=0, inp2_unstable=0, delta_unstable=256
- layer_idx=6, type=LayerType.Linear, total=256, inp1_unstable=0, inp2_unstable=0, delta_unstable=256
- layer_idx=8, type=LayerType.Linear, total=10, inp1_unstable=0, inp2_unstable=0, delta_unstable=10

Time for backsubstitution: 1.57 seconds

### RS candidates at layer 1
type: RSZ, layer: 1, pos: 140
type: RSZ, layer: 1, pos: 167
type: RSZ, layer: 1, pos: 213
type: RSZ, layer: 1, pos: 219

Time for candidate selection: 0.21 seconds

### Candidate
type: RSZ, layer: 1, pos: 140

### Relational analysis RSZ of RS_RSZ1_RSZ2_RSZ2_RSZ1_RSZ1_RSZ1_RSZ1
Optimization infeasible because this subproblem isn't reachable.

### Candidate
type: RSZ, layer: 1, pos: 167

### Relational analysis RSZ of RS_RSZ1_RSZ2_RSZ2_RSZ1_RSZ1_RSZ1_RSZ1

#### Relational analysis RSZ result of RS_RSZ1_RSZ2_RSZ2_RSZ1_RSZ1_RSZ1_RSZ1
Status: Status.UNKNOWN
Output dim: 7, lower bound: -0.0009336, upper bound: 0.0009512
time: 0.61 seconds

### Relational analysis RSZ of RS_RSZ1_RSZ2_RSZ2_RSZ1_RSZ1_RSZ1_RSZ2

#### Relational analysis RSZ result of RS_RSZ1_RSZ2_RSZ2_RSZ1_RSZ1_RSZ1_RSZ2
Status: Status.UNKNOWN
Output dim: 7, lower bound: -0.0009246, upper bound: 0.0009541
time: 0.68 seconds

## BFS RS instance: RS_RSZ1_RSZ2_RSZ2_RSZ1_RSZ1_RSZ2

### Backsubstitution after applying RS history:
0: -0.0027939, -0.0016242, -0.0027939, -0.0016242, -0.0007583, 0.0007451
1: -0.0114005, -0.0084323, -0.0114005, -0.0084323, -0.0019243, 0.0018909
2: 0.0279571, 0.0297986, 0.0279571, 0.0297986, -0.0011938, 0.0011731
3: 0.0039210, 0.0073596, 0.0039210, 0.0073596, -0.0021905, 0.0022292
4: -0.0104893, -0.0074701, -0.0104893, -0.0074701, -0.0019573, 0.0019234
5: 0.0097651, 0.0109087, 0.0097651, 0.0109087, -0.0007414, 0.0007285
6: 0.0053643, 0.0097284, 0.0053643, 0.0097284, -0.0027801, 0.0028291
7: 0.9818129, 0.9848667, 0.9818129, 0.9848667, -0.0019454, 0.0019797
8: -0.0060636, -0.0027895, -0.0060636, -0.0027895, -0.0020857, 0.0021225
9: -0.0031570, -0.0009942, -0.0031570, -0.0009942, -0.0014020, 0.0013777

### Unstable ReLU Count (Linear/Conv2D Layers)
- layer_idx=0, type=LayerType.Linear, total=256, inp1_unstable=10, inp2_unstable=10, delta_unstable=250
- layer_idx=2, type=LayerType.Linear, total=256, inp1_unstable=10, inp2_unstable=10, delta_unstable=256
- layer_idx=4, type=LayerType.Linear, total=256, inp1_unstable=0, inp2_unstable=0, delta_unstable=256
- layer_idx=6, type=LayerType.Linear, total=256, inp1_unstable=0, inp2_unstable=0, delta_unstable=256
- layer_idx=8, type=LayerType.Linear, total=10, inp1_unstable=0, inp2_unstable=0, delta_unstable=10

Time for backsubstitution: 1.57 seconds

### RS candidates at layer 1
type: RSZ, layer: 1, pos: 140
type: RSZ, layer: 1, pos: 167
type: RSZ, layer: 1, pos: 213
type: RSZ, layer: 1, pos: 219

Time for candidate selection: 0.16 seconds

### Candidate
type: RSZ, layer: 1, pos: 140

### Relational analysis RSZ of RS_RSZ1_RSZ2_RSZ2_RSZ1_RSZ1_RSZ2_RSZ1
Optimization infeasible because this subproblem isn't reachable.

### Candidate
type: RSZ, layer: 1, pos: 167

### Relational analysis RSZ of RS_RSZ1_RSZ2_RSZ2_RSZ1_RSZ1_RSZ2_RSZ1

#### Relational analysis RSZ result of RS_RSZ1_RSZ2_RSZ2_RSZ1_RSZ1_RSZ2_RSZ1
Status: Status.UNKNOWN
Output dim: 7, lower bound: -0.0009336, upper bound: 0.0009521
time: 0.66 seconds

### Relational analysis RSZ of RS_RSZ1_RSZ2_RSZ2_RSZ1_RSZ1_RSZ2_RSZ2

#### Relational analysis RSZ result of RS_RSZ1_RSZ2_RSZ2_RSZ1_RSZ1_RSZ2_RSZ2
Status: Status.UNKNOWN
Output dim: 7, lower bound: -0.0009246, upper bound: 0.0009553
time: 0.65 seconds

## BFS RS instance: RS_RSZ1_RSZ2_RSZ2_RSZ1_RSZ2_RSZ1

### Backsubstitution after applying RS history:
0: -0.0027939, -0.0016242, -0.0027939, -0.0016242, -0.0007436, 0.0007596
1: -0.0114005, -0.0084323, -0.0114005, -0.0084323, -0.0018869, 0.0019276
2: 0.0279571, 0.0297986, 0.0279571, 0.0297986, -0.0011707, 0.0011959
3: 0.0039210, 0.0073596, 0.0039210, 0.0073596, -0.0022330, 0.0021859
4: -0.0104893, -0.0074701, -0.0104893, -0.0074701, -0.0019193, 0.0019606
5: 0.0097651, 0.0109087, 0.0097651, 0.0109087, -0.0007270, 0.0007426
6: 0.0053643, 0.0097284, 0.0053643, 0.0097284, -0.0028339, 0.0027742
7: 0.9818129, 0.9848667, 0.9818129, 0.9848667, -0.0019831, 0.0019413
8: -0.0060636, -0.0027895, -0.0060636, -0.0027895, -0.0021261, 0.0020813
9: -0.0031570, -0.0009942, -0.0031570, -0.0009942, -0.0013748, 0.0014044

### Unstable ReLU Count (Linear/Conv2D Layers)
- layer_idx=0, type=LayerType.Linear, total=256, inp1_unstable=10, inp2_unstable=10, delta_unstable=250
- layer_idx=2, type=LayerType.Linear, total=256, inp1_unstable=10, inp2_unstable=10, delta_unstable=256
- layer_idx=4, type=LayerType.Linear, total=256, inp1_unstable=0, inp2_unstable=0, delta_unstable=256
- layer_idx=6, type=LayerType.Linear, total=256, inp1_unstable=0, inp2_unstable=0, delta_unstable=256
- layer_idx=8, type=LayerType.Linear, total=10, inp1_unstable=0, inp2_unstable=0, delta_unstable=10

Time for backsubstitution: 1.55 seconds

### RS candidates at layer 1
type: RSZ, layer: 1, pos: 140
type: RSZ, layer: 1, pos: 167
type: RSZ, layer: 1, pos: 213
type: RSZ, layer: 1, pos: 219

Time for candidate selection: 0.16 seconds

### Candidate
type: RSZ, layer: 1, pos: 140

### Relational analysis RSZ of RS_RSZ1_RSZ2_RSZ2_RSZ1_RSZ2_RSZ1_RSZ1
Optimization infeasible because this subproblem isn't reachable.

### Candidate
type: RSZ, layer: 1, pos: 167

### Relational analysis RSZ of RS_RSZ1_RSZ2_RSZ2_RSZ1_RSZ2_RSZ1_RSZ1

#### Relational analysis RSZ result of RS_RSZ1_RSZ2_RSZ2_RSZ1_RSZ2_RSZ1_RSZ1
Status: Status.UNKNOWN
Output dim: 7, lower bound: -0.0009414, upper bound: 0.0009401
time: 0.63 seconds

### Relational analysis RSZ of RS_RSZ1_RSZ2_RSZ2_RSZ1_RSZ2_RSZ1_RSZ2

#### Relational analysis RSZ result of RS_RSZ1_RSZ2_RSZ2_RSZ1_RSZ2_RSZ1_RSZ2
Status: Status.UNKNOWN
Output dim: 7, lower bound: -0.0009337, upper bound: 0.0009419
time: 0.66 seconds

## BFS RS instance: RS_RSZ1_RSZ2_RSZ2_RSZ1_RSZ2_RSZ2

### Backsubstitution after applying RS history:
0: -0.0027939, -0.0016242, -0.0027939, -0.0016242, -0.0007438, 0.0007595
1: -0.0114005, -0.0084323, -0.0114005, -0.0084323, -0.0018874, 0.0019273
2: 0.0279571, 0.0297986, 0.0279571, 0.0297986, -0.0011710, 0.0011957
3: 0.0039210, 0.0073596, 0.0039210, 0.0073596, -0.0022327, 0.0021865
4: -0.0104893, -0.0074701, -0.0104893, -0.0074701, -0.0019198, 0.0019604
5: 0.0097651, 0.0109087, 0.0097651, 0.0109087, -0.0007272, 0.0007426
6: 0.0053643, 0.0097284, 0.0053643, 0.0097284, -0.0028336, 0.0027750
7: 0.9818129, 0.9848667, 0.9818129, 0.9848667, -0.0019828, 0.0019418
8: -0.0060636, -0.0027895, -0.0060636, -0.0027895, -0.0021259, 0.0020819
9: -0.0031570, -0.0009942, -0.0031570, -0.0009942, -0.0013752, 0.0014043

### Unstable ReLU Count (Linear/Conv2D Layers)
- layer_idx=0, type=LayerType.Linear, total=256, inp1_unstable=10, inp2_unstable=10, delta_unstable=250
- layer_idx=2, type=LayerType.Linear, total=256, inp1_unstable=10, inp2_unstable=10, delta_unstable=256
- layer_idx=4, type=LayerType.Linear, total=256, inp1_unstable=0, inp2_unstable=0, delta_unstable=256
- layer_idx=6, type=LayerType.Linear, total=256, inp1_unstable=0, inp2_unstable=0, delta_unstable=256
- layer_idx=8, type=LayerType.Linear, total=10, inp1_unstable=0, inp2_unstable=0, delta_unstable=10

Time for backsubstitution: 1.56 seconds

### RS candidates at layer 1
type: RSZ, layer: 1, pos: 140
type: RSZ, layer: 1, pos: 167
type: RSZ, layer: 1, pos: 213
type: RSZ, layer: 1, pos: 219

Time for candidate selection: 0.16 seconds

### Candidate
type: RSZ, layer: 1, pos: 140

### Relational analysis RSZ of RS_RSZ1_RSZ2_RSZ2_RSZ1_RSZ2_RSZ2_RSZ1
Optimization infeasible because this subproblem isn't reachable.

### Candidate
type: RSZ, layer: 1, pos: 167

### Relational analysis RSZ of RS_RSZ1_RSZ2_RSZ2_RSZ1_RSZ2_RSZ2_RSZ1

#### Relational analysis RSZ result of RS_RSZ1_RSZ2_RSZ2_RSZ1_RSZ2_RSZ2_RSZ1
Status: Status.UNKNOWN
Output dim: 7, lower bound: -0.0009415, upper bound: 0.0009410
time: 0.64 seconds

### Relational analysis RSZ of RS_RSZ1_RSZ2_RSZ2_RSZ1_RSZ2_RSZ2_RSZ2

#### Relational analysis RSZ result of RS_RSZ1_RSZ2_RSZ2_RSZ1_RSZ2_RSZ2_RSZ2
Status: Status.UNKNOWN
Output dim: 7, lower bound: -0.0009341, upper bound: 0.0009426
time: 0.64 seconds

## BFS RS instance: RS_RSZ1_RSZ2_RSZ2_RSZ2_RSZ1_RSZ1

### Backsubstitution after applying RS history:
0: -0.0027939, -0.0016242, -0.0027939, -0.0016242, -0.0007650, 0.0007367
1: -0.0114005, -0.0084323, -0.0114005, -0.0084323, -0.0019414, 0.0018695
2: 0.0279571, 0.0297986, 0.0279571, 0.0297986, -0.0012045, 0.0011599
3: 0.0039210, 0.0073596, 0.0039210, 0.0073596, -0.0021658, 0.0022490
4: -0.0104893, -0.0074701, -0.0104893, -0.0074701, -0.0019747, 0.0019016
5: 0.0097651, 0.0109087, 0.0097651, 0.0109087, -0.0007480, 0.0007203
6: 0.0053643, 0.0097284, 0.0053643, 0.0097284, -0.0027486, 0.0028543
7: 0.9818129, 0.9848667, 0.9818129, 0.9848667, -0.0019234, 0.0019973
8: -0.0060636, -0.0027895, -0.0060636, -0.0027895, -0.0020621, 0.0021414
9: -0.0031570, -0.0009942, -0.0031570, -0.0009942, -0.0014145, 0.0013622

### Unstable ReLU Count (Linear/Conv2D Layers)
- layer_idx=0, type=LayerType.Linear, total=256, inp1_unstable=10, inp2_unstable=10, delta_unstable=250
- layer_idx=2, type=LayerType.Linear, total=256, inp1_unstable=10, inp2_unstable=10, delta_unstable=256
- layer_idx=4, type=LayerType.Linear, total=256, inp1_unstable=0, inp2_unstable=0, delta_unstable=256
- layer_idx=6, type=LayerType.Linear, total=256, inp1_unstable=0, inp2_unstable=0, delta_unstable=256
- layer_idx=8, type=LayerType.Linear, total=10, inp1_unstable=0, inp2_unstable=0, delta_unstable=10

Time for backsubstitution: 1.60 seconds

### RS candidates at layer 1
type: RSZ, layer: 1, pos: 140
type: RSZ, layer: 1, pos: 167
type: RSZ, layer: 1, pos: 213
type: RSZ, layer: 1, pos: 219

Time for candidate selection: 0.22 seconds

### Candidate
type: RSZ, layer: 1, pos: 140

### Relational analysis RSZ of RS_RSZ1_RSZ2_RSZ2_RSZ2_RSZ1_RSZ1_RSZ1
Optimization infeasible because this subproblem isn't reachable.

### Candidate
type: RSZ, layer: 1, pos: 167

### Relational analysis RSZ of RS_RSZ1_RSZ2_RSZ2_RSZ2_RSZ1_RSZ1_RSZ1

#### Relational analysis RSZ result of RS_RSZ1_RSZ2_RSZ2_RSZ2_RSZ1_RSZ1_RSZ1
Status: Status.UNKNOWN
Output dim: 7, lower bound: -0.0009341, upper bound: 0.0009428
time: 0.62 seconds

### Relational analysis RSZ of RS_RSZ1_RSZ2_RSZ2_RSZ2_RSZ1_RSZ1_RSZ2

#### Relational analysis RSZ result of RS_RSZ1_RSZ2_RSZ2_RSZ2_RSZ1_RSZ1_RSZ2
Status: Status.UNKNOWN
Output dim: 7, lower bound: -0.0009266, upper bound: 0.0009485
time: 0.66 seconds

## BFS RS instance: RS_RSZ1_RSZ2_RSZ2_RSZ2_RSZ1_RSZ2

### Backsubstitution after applying RS history:
0: -0.0027939, -0.0016242, -0.0027939, -0.0016242, -0.0007651, 0.0007366
1: -0.0114005, -0.0084323, -0.0114005, -0.0084323, -0.0019416, 0.0018692
2: 0.0279571, 0.0297986, 0.0279571, 0.0297986, -0.0012045, 0.0011597
3: 0.0039210, 0.0073596, 0.0039210, 0.0073596, -0.0021654, 0.0022492
4: -0.0104893, -0.0074701, -0.0104893, -0.0074701, -0.0019749, 0.0019013
5: 0.0097651, 0.0109087, 0.0097651, 0.0109087, -0.0007480, 0.0007202
6: 0.0053643, 0.0097284, 0.0053643, 0.0097284, -0.0027482, 0.0028545
7: 0.9818129, 0.9848667, 0.9818129, 0.9848667, -0.0019231, 0.0019975
8: -0.0060636, -0.0027895, -0.0060636, -0.0027895, -0.0020618, 0.0021416
9: -0.0031570, -0.0009942, -0.0031570, -0.0009942, -0.0014146, 0.0013620

### Unstable ReLU Count (Linear/Conv2D Layers)
- layer_idx=0, type=LayerType.Linear, total=256, inp1_unstable=10, inp2_unstable=10, delta_unstable=250
- layer_idx=2, type=LayerType.Linear, total=256, inp1_unstable=10, inp2_unstable=10, delta_unstable=256
- layer_idx=4, type=LayerType.Linear, total=256, inp1_unstable=0, inp2_unstable=0, delta_unstable=256
- layer_idx=6, type=LayerType.Linear, total=256, inp1_unstable=0, inp2_unstable=0, delta_unstable=256
- layer_idx=8, type=LayerType.Linear, total=10, inp1_unstable=0, inp2_unstable=0, delta_unstable=10

Time for backsubstitution: 1.63 seconds

### RS candidates at layer 1
type: RSZ, layer: 1, pos: 140
type: RSZ, layer: 1, pos: 167
type: RSZ, layer: 1, pos: 213
type: RSZ, layer: 1, pos: 219

Time for candidate selection: 0.18 seconds

### Candidate
type: RSZ, layer: 1, pos: 140

### Relational analysis RSZ of RS_RSZ1_RSZ2_RSZ2_RSZ2_RSZ1_RSZ2_RSZ1
Optimization infeasible because this subproblem isn't reachable.

### Candidate
type: RSZ, layer: 1, pos: 167

### Relational analysis RSZ of RS_RSZ1_RSZ2_RSZ2_RSZ2_RSZ1_RSZ2_RSZ1

#### Relational analysis RSZ result of RS_RSZ1_RSZ2_RSZ2_RSZ2_RSZ1_RSZ2_RSZ1
Status: Status.UNKNOWN
Output dim: 7, lower bound: -0.0009342, upper bound: 0.0009445
time: 0.64 seconds

### Relational analysis RSZ of RS_RSZ1_RSZ2_RSZ2_RSZ2_RSZ1_RSZ2_RSZ2

#### Relational analysis RSZ result of RS_RSZ1_RSZ2_RSZ2_RSZ2_RSZ1_RSZ2_RSZ2
Status: Status.UNKNOWN
Output dim: 7, lower bound: -0.0009266, upper bound: 0.0009504
time: 0.63 seconds

## BFS RS instance: RS_RSZ1_RSZ2_RSZ2_RSZ2_RSZ2_RSZ1

### Backsubstitution after applying RS history:
0: -0.0027939, -0.0016242, -0.0027939, -0.0016242, -0.0007495, 0.0007515
1: -0.0114005, -0.0084323, -0.0114005, -0.0084323, -0.0019020, 0.0019071
2: 0.0279571, 0.0297986, 0.0279571, 0.0297986, -0.0011800, 0.0011832
3: 0.0039210, 0.0073596, 0.0039210, 0.0073596, -0.0022093, 0.0022034
4: -0.0104893, -0.0074701, -0.0104893, -0.0074701, -0.0019346, 0.0019399
5: 0.0097651, 0.0109087, 0.0097651, 0.0109087, -0.0007328, 0.0007348
6: 0.0053643, 0.0097284, 0.0053643, 0.0097284, -0.0028039, 0.0027963
7: 0.9818129, 0.9848667, 0.9818129, 0.9848667, -0.0019620, 0.0019567
8: -0.0060636, -0.0027895, -0.0060636, -0.0027895, -0.0021036, 0.0020979
9: -0.0031570, -0.0009942, -0.0031570, -0.0009942, -0.0013858, 0.0013896

### Unstable ReLU Count (Linear/Conv2D Layers)
- layer_idx=0, type=LayerType.Linear, total=256, inp1_unstable=10, inp2_unstable=10, delta_unstable=250
- layer_idx=2, type=LayerType.Linear, total=256, inp1_unstable=10, inp2_unstable=10, delta_unstable=256
- layer_idx=4, type=LayerType.Linear, total=256, inp1_unstable=0, inp2_unstable=0, delta_unstable=256
- layer_idx=6, type=LayerType.Linear, total=256, inp1_unstable=0, inp2_unstable=0, delta_unstable=256
- layer_idx=8, type=LayerType.Linear, total=10, inp1_unstable=0, inp2_unstable=0, delta_unstable=10

Time for backsubstitution: 1.56 seconds

### RS candidates at layer 1
type: RSZ, layer: 1, pos: 140
type: RSZ, layer: 1, pos: 167
type: RSZ, layer: 1, pos: 213
type: RSZ, layer: 1, pos: 219

Time for candidate selection: 0.16 seconds

### Candidate
type: RSZ, layer: 1, pos: 140

### Relational analysis RSZ of RS_RSZ1_RSZ2_RSZ2_RSZ2_RSZ2_RSZ1_RSZ1
Optimization infeasible because this subproblem isn't reachable.

### Candidate
type: RSZ, layer: 1, pos: 167

### Relational analysis RSZ of RS_RSZ1_RSZ2_RSZ2_RSZ2_RSZ2_RSZ1_RSZ1

#### Relational analysis RSZ result of RS_RSZ1_RSZ2_RSZ2_RSZ2_RSZ2_RSZ1_RSZ1
Status: Status.UNKNOWN
Output dim: 7, lower bound: -0.0009414, upper bound: 0.0009284
time: 0.64 seconds

### Relational analysis RSZ of RS_RSZ1_RSZ2_RSZ2_RSZ2_RSZ2_RSZ1_RSZ2

#### Relational analysis RSZ result of RS_RSZ1_RSZ2_RSZ2_RSZ2_RSZ2_RSZ1_RSZ2
Status: Status.UNKNOWN
Output dim: 7, lower bound: -0.0009349, upper bound: 0.0009321
time: 0.66 seconds

## BFS RS instance: RS_RSZ1_RSZ2_RSZ2_RSZ2_RSZ2_RSZ2

### Backsubstitution after applying RS history:
0: -0.0027939, -0.0016242, -0.0027939, -0.0016242, -0.0007497, 0.0007514
1: -0.0114005, -0.0084323, -0.0114005, -0.0084323, -0.0019025, 0.0019067
2: 0.0279571, 0.0297986, 0.0279571, 0.0297986, -0.0011803, 0.0011829
3: 0.0039210, 0.0073596, 0.0039210, 0.0073596, -0.0022089, 0.0022040
4: -0.0104893, -0.0074701, -0.0104893, -0.0074701, -0.0019352, 0.0019395
5: 0.0097651, 0.0109087, 0.0097651, 0.0109087, -0.0007330, 0.0007346
6: 0.0053643, 0.0097284, 0.0053643, 0.0097284, -0.0028033, 0.0027972
7: 0.9818129, 0.9848667, 0.9818129, 0.9848667, -0.0019616, 0.0019573
8: -0.0060636, -0.0027895, -0.0060636, -0.0027895, -0.0021032, 0.0020986
9: -0.0031570, -0.0009942, -0.0031570, -0.0009942, -0.0013862, 0.0013893

### Unstable ReLU Count (Linear/Conv2D Layers)
- layer_idx=0, type=LayerType.Linear, total=256, inp1_unstable=10, inp2_unstable=10, delta_unstable=250
- layer_idx=2, type=LayerType.Linear, total=256, inp1_unstable=10, inp2_unstable=10, delta_unstable=256
- layer_idx=4, type=LayerType.Linear, total=256, inp1_unstable=0, inp2_unstable=0, delta_unstable=256
- layer_idx=6, type=LayerType.Linear, total=256, inp1_unstable=0, inp2_unstable=0, delta_unstable=256
- layer_idx=8, type=LayerType.Linear, total=10, inp1_unstable=0, inp2_unstable=0, delta_unstable=10

Time for backsubstitution: 1.58 seconds

### RS candidates at layer 1
type: RSZ, layer: 1, pos: 140
type: RSZ, layer: 1, pos: 167
type: RSZ, layer: 1, pos: 213
type: RSZ, layer: 1, pos: 219

Time for candidate selection: 0.16 seconds

### Candidate
type: RSZ, layer: 1, pos: 140

### Relational analysis RSZ of RS_RSZ1_RSZ2_RSZ2_RSZ2_RSZ2_RSZ2_RSZ1
Optimization infeasible because this subproblem isn't reachable.

### Candidate
type: RSZ, layer: 1, pos: 167

### Relational analysis RSZ of RS_RSZ1_RSZ2_RSZ2_RSZ2_RSZ2_RSZ2_RSZ1

#### Relational analysis RSZ result of RS_RSZ1_RSZ2_RSZ2_RSZ2_RSZ2_RSZ2_RSZ1
Status: Status.UNKNOWN
Output dim: 7, lower bound: -0.0009420, upper bound: 0.0009291
time: 0.63 seconds

### Relational analysis RSZ of RS_RSZ1_RSZ2_RSZ2_RSZ2_RSZ2_RSZ2_RSZ2

#### Relational analysis RSZ result of RS_RSZ1_RSZ2_RSZ2_RSZ2_RSZ2_RSZ2_RSZ2
Status: Status.UNKNOWN
Output dim: 7, lower bound: -0.0009355, upper bound: 0.0009327
time: 0.65 seconds

## BFS RS instance: RS_RSZ2_RSZ1_RSZ1_RSZ1_RSZ1_RSZ1

### Backsubstitution after applying RS history:
0: -0.0027939, -0.0016242, -0.0027939, -0.0016242, -0.0007514, 0.0007502
1: -0.0114005, -0.0084323, -0.0114005, -0.0084323, -0.0019067, 0.0019038
2: 0.0279571, 0.0297986, 0.0279571, 0.0297986, -0.0011829, 0.0011811
3: 0.0039210, 0.0073596, 0.0039210, 0.0073596, -0.0022055, 0.0022089
4: -0.0104893, -0.0074701, -0.0104893, -0.0074701, -0.0019395, 0.0019365
5: 0.0097651, 0.0109087, 0.0097651, 0.0109087, -0.0007346, 0.0007335
6: 0.0053643, 0.0097284, 0.0053643, 0.0097284, -0.0027990, 0.0028033
7: 0.9818129, 0.9848667, 0.9818129, 0.9848667, -0.0019586, 0.0019616
8: -0.0060636, -0.0027895, -0.0060636, -0.0027895, -0.0020999, 0.0021032
9: -0.0031570, -0.0009942, -0.0031570, -0.0009942, -0.0013893, 0.0013871

### Unstable ReLU Count (Linear/Conv2D Layers)
- layer_idx=0, type=LayerType.Linear, total=256, inp1_unstable=10, inp2_unstable=10, delta_unstable=250
- layer_idx=2, type=LayerType.Linear, total=256, inp1_unstable=10, inp2_unstable=10, delta_unstable=256
- layer_idx=4, type=LayerType.Linear, total=256, inp1_unstable=0, inp2_unstable=0, delta_unstable=256
- layer_idx=6, type=LayerType.Linear, total=256, inp1_unstable=0, inp2_unstable=0, delta_unstable=256
- layer_idx=8, type=LayerType.Linear, total=10, inp1_unstable=0, inp2_unstable=0, delta_unstable=10

Time for backsubstitution: 1.59 seconds

### RS candidates at layer 1
type: RSZ, layer: 1, pos: 140
type: RSZ, layer: 1, pos: 167
type: RSZ, layer: 1, pos: 213
type: RSZ, layer: 1, pos: 219

Time for candidate selection: 0.18 seconds

### Candidate
type: RSZ, layer: 1, pos: 140

### Relational analysis RSZ of RS_RSZ2_RSZ1_RSZ1_RSZ1_RSZ1_RSZ1_RSZ1
Optimization infeasible because this subproblem isn't reachable.

### Candidate
type: RSZ, layer: 1, pos: 167

### Relational analysis RSZ of RS_RSZ2_RSZ1_RSZ1_RSZ1_RSZ1_RSZ1_RSZ1

#### Relational analysis RSZ result of RS_RSZ2_RSZ1_RSZ1_RSZ1_RSZ1_RSZ1_RSZ1
Status: Status.UNKNOWN
Output dim: 7, lower bound: -0.0009327, upper bound: 0.0009355
time: 0.67 seconds

### Relational analysis RSZ of RS_RSZ2_RSZ1_RSZ1_RSZ1_RSZ1_RSZ1_RSZ2

#### Relational analysis RSZ result of RS_RSZ2_RSZ1_RSZ1_RSZ1_RSZ1_RSZ1_RSZ2
Status: Status.UNKNOWN
Output dim: 7, lower bound: -0.0009291, upper bound: 0.0009420
time: 0.63 seconds

## BFS RS instance: RS_RSZ2_RSZ1_RSZ1_RSZ1_RSZ1_RSZ2

### Backsubstitution after applying RS history:
0: -0.0027939, -0.0016242, -0.0027939, -0.0016242, -0.0007515, 0.0007501
1: -0.0114005, -0.0084323, -0.0114005, -0.0084323, -0.0019071, 0.0019034
2: 0.0279571, 0.0297986, 0.0279571, 0.0297986, -0.0011832, 0.0011809
3: 0.0039210, 0.0073596, 0.0039210, 0.0073596, -0.0022050, 0.0022093
4: -0.0104893, -0.0074701, -0.0104893, -0.0074701, -0.0019399, 0.0019361
5: 0.0097651, 0.0109087, 0.0097651, 0.0109087, -0.0007348, 0.0007333
6: 0.0053643, 0.0097284, 0.0053643, 0.0097284, -0.0027984, 0.0028039
7: 0.9818129, 0.9848667, 0.9818129, 0.9848667, -0.0019582, 0.0019620
8: -0.0060636, -0.0027895, -0.0060636, -0.0027895, -0.0020995, 0.0021036
9: -0.0031570, -0.0009942, -0.0031570, -0.0009942, -0.0013896, 0.0013868

### Unstable ReLU Count (Linear/Conv2D Layers)
- layer_idx=0, type=LayerType.Linear, total=256, inp1_unstable=10, inp2_unstable=10, delta_unstable=250
- layer_idx=2, type=LayerType.Linear, total=256, inp1_unstable=10, inp2_unstable=10, delta_unstable=256
- layer_idx=4, type=LayerType.Linear, total=256, inp1_unstable=0, inp2_unstable=0, delta_unstable=256
- layer_idx=6, type=LayerType.Linear, total=256, inp1_unstable=0, inp2_unstable=0, delta_unstable=256
- layer_idx=8, type=LayerType.Linear, total=10, inp1_unstable=0, inp2_unstable=0, delta_unstable=10

Time for backsubstitution: 1.55 seconds

### RS candidates at layer 1
type: RSZ, layer: 1, pos: 140
type: RSZ, layer: 1, pos: 167
type: RSZ, layer: 1, pos: 213
type: RSZ, layer: 1, pos: 219

Time for candidate selection: 0.16 seconds

### Candidate
type: RSZ, layer: 1, pos: 140

### Relational analysis RSZ of RS_RSZ2_RSZ1_RSZ1_RSZ1_RSZ1_RSZ2_RSZ1
Optimization infeasible because this subproblem isn't reachable.

### Candidate
type: RSZ, layer: 1, pos: 167

### Relational analysis RSZ of RS_RSZ2_RSZ1_RSZ1_RSZ1_RSZ1_RSZ2_RSZ1

#### Relational analysis RSZ result of RS_RSZ2_RSZ1_RSZ1_RSZ1_RSZ1_RSZ2_RSZ1
Status: Status.UNKNOWN
Output dim: 7, lower bound: -0.0009321, upper bound: 0.0009349
time: 0.65 seconds

### Relational analysis RSZ of RS_RSZ2_RSZ1_RSZ1_RSZ1_RSZ1_RSZ2_RSZ2

#### Relational analysis RSZ result of RS_RSZ2_RSZ1_RSZ1_RSZ1_RSZ1_RSZ2_RSZ2
Status: Status.UNKNOWN
Output dim: 7, lower bound: -0.0009284, upper bound: 0.0009414
time: 0.67 seconds

## BFS RS instance: RS_RSZ2_RSZ1_RSZ1_RSZ1_RSZ2_RSZ1

### Backsubstitution after applying RS history:
0: -0.0027939, -0.0016242, -0.0027939, -0.0016242, -0.0007366, 0.0007657
1: -0.0114005, -0.0084323, -0.0114005, -0.0084323, -0.0018692, 0.0019430
2: 0.0279571, 0.0297986, 0.0279571, 0.0297986, -0.0011597, 0.0012055
3: 0.0039210, 0.0073596, 0.0039210, 0.0073596, -0.0022509, 0.0021654
4: -0.0104893, -0.0074701, -0.0104893, -0.0074701, -0.0019013, 0.0019764
5: 0.0097651, 0.0109087, 0.0097651, 0.0109087, -0.0007202, 0.0007486
6: 0.0053643, 0.0097284, 0.0053643, 0.0097284, -0.0028567, 0.0027482
7: 0.9818129, 0.9848667, 0.9818129, 0.9848667, -0.0019990, 0.0019231
8: -0.0060636, -0.0027895, -0.0060636, -0.0027895, -0.0021432, 0.0020618
9: -0.0031570, -0.0009942, -0.0031570, -0.0009942, -0.0013620, 0.0014157

### Unstable ReLU Count (Linear/Conv2D Layers)
- layer_idx=0, type=LayerType.Linear, total=256, inp1_unstable=10, inp2_unstable=10, delta_unstable=250
- layer_idx=2, type=LayerType.Linear, total=256, inp1_unstable=10, inp2_unstable=10, delta_unstable=256
- layer_idx=4, type=LayerType.Linear, total=256, inp1_unstable=0, inp2_unstable=0, delta_unstable=256
- layer_idx=6, type=LayerType.Linear, total=256, inp1_unstable=0, inp2_unstable=0, delta_unstable=256
- layer_idx=8, type=LayerType.Linear, total=10, inp1_unstable=0, inp2_unstable=0, delta_unstable=10

Time for backsubstitution: 1.56 seconds

### RS candidates at layer 1
type: RSZ, layer: 1, pos: 140
type: RSZ, layer: 1, pos: 167
type: RSZ, layer: 1, pos: 213
type: RSZ, layer: 1, pos: 219

Time for candidate selection: 0.17 seconds

### Candidate
type: RSZ, layer: 1, pos: 140

### Relational analysis RSZ of RS_RSZ2_RSZ1_RSZ1_RSZ1_RSZ2_RSZ1_RSZ1
Optimization infeasible because this subproblem isn't reachable.

### Candidate
type: RSZ, layer: 1, pos: 167

### Relational analysis RSZ of RS_RSZ2_RSZ1_RSZ1_RSZ1_RSZ2_RSZ1_RSZ1

#### Relational analysis RSZ result of RS_RSZ2_RSZ1_RSZ1_RSZ1_RSZ2_RSZ1_RSZ1
Status: Status.UNKNOWN
Output dim: 7, lower bound: -0.0009504, upper bound: 0.0009266
time: 0.69 seconds

### Relational analysis RSZ of RS_RSZ2_RSZ1_RSZ1_RSZ1_RSZ2_RSZ1_RSZ2

#### Relational analysis RSZ result of RS_RSZ2_RSZ1_RSZ1_RSZ1_RSZ2_RSZ1_RSZ2
Status: Status.UNKNOWN
Output dim: 7, lower bound: -0.0009445, upper bound: 0.0009342
time: 0.67 seconds

## BFS RS instance: RS_RSZ2_RSZ1_RSZ1_RSZ1_RSZ2_RSZ2

### Backsubstitution after applying RS history:
0: -0.0027939, -0.0016242, -0.0027939, -0.0016242, -0.0007367, 0.0007656
1: -0.0114005, -0.0084323, -0.0114005, -0.0084323, -0.0018695, 0.0019429
2: 0.0279571, 0.0297986, 0.0279571, 0.0297986, -0.0011599, 0.0012054
3: 0.0039210, 0.0073596, 0.0039210, 0.0073596, -0.0022508, 0.0021658
4: -0.0104893, -0.0074701, -0.0104893, -0.0074701, -0.0019016, 0.0019763
5: 0.0097651, 0.0109087, 0.0097651, 0.0109087, -0.0007203, 0.0007486
6: 0.0053643, 0.0097284, 0.0053643, 0.0097284, -0.0028565, 0.0027486
7: 0.9818129, 0.9848667, 0.9818129, 0.9848667, -0.0019989, 0.0019234
8: -0.0060636, -0.0027895, -0.0060636, -0.0027895, -0.0021431, 0.0020621
9: -0.0031570, -0.0009942, -0.0031570, -0.0009942, -0.0013622, 0.0014156

### Unstable ReLU Count (Linear/Conv2D Layers)
- layer_idx=0, type=LayerType.Linear, total=256, inp1_unstable=10, inp2_unstable=10, delta_unstable=250
- layer_idx=2, type=LayerType.Linear, total=256, inp1_unstable=10, inp2_unstable=10, delta_unstable=256
- layer_idx=4, type=LayerType.Linear, total=256, inp1_unstable=0, inp2_unstable=0, delta_unstable=256
- layer_idx=6, type=LayerType.Linear, total=256, inp1_unstable=0, inp2_unstable=0, delta_unstable=256
- layer_idx=8, type=LayerType.Linear, total=10, inp1_unstable=0, inp2_unstable=0, delta_unstable=10

Time for backsubstitution: 1.58 seconds

### RS candidates at layer 1
type: RSZ, layer: 1, pos: 140
type: RSZ, layer: 1, pos: 167
type: RSZ, layer: 1, pos: 213
type: RSZ, layer: 1, pos: 219

Time for candidate selection: 0.21 seconds

### Candidate
type: RSZ, layer: 1, pos: 140

### Relational analysis RSZ of RS_RSZ2_RSZ1_RSZ1_RSZ1_RSZ2_RSZ2_RSZ1
Optimization infeasible because this subproblem isn't reachable.

### Candidate
type: RSZ, layer: 1, pos: 167

### Relational analysis RSZ of RS_RSZ2_RSZ1_RSZ1_RSZ1_RSZ2_RSZ2_RSZ1

#### Relational analysis RSZ result of RS_RSZ2_RSZ1_RSZ1_RSZ1_RSZ2_RSZ2_RSZ1
Status: Status.UNKNOWN
Output dim: 7, lower bound: -0.0009485, upper bound: 0.0009266
time: 0.67 seconds

### Relational analysis RSZ of RS_RSZ2_RSZ1_RSZ1_RSZ1_RSZ2_RSZ2_RSZ2

#### Relational analysis RSZ result of RS_RSZ2_RSZ1_RSZ1_RSZ1_RSZ2_RSZ2_RSZ2
Status: Status.UNKNOWN
Output dim: 7, lower bound: -0.0009428, upper bound: 0.0009341
time: 0.66 seconds

## BFS RS instance: RS_RSZ2_RSZ1_RSZ1_RSZ2_RSZ1_RSZ1

### Backsubstitution after applying RS history:
0: -0.0027939, -0.0016242, -0.0027939, -0.0016242, -0.0007595, 0.0007437
1: -0.0114005, -0.0084323, -0.0114005, -0.0084323, -0.0019273, 0.0018871
2: 0.0279571, 0.0297986, 0.0279571, 0.0297986, -0.0011957, 0.0011708
3: 0.0039210, 0.0073596, 0.0039210, 0.0073596, -0.0021862, 0.0022327
4: -0.0104893, -0.0074701, -0.0104893, -0.0074701, -0.0019604, 0.0019195
5: 0.0097651, 0.0109087, 0.0097651, 0.0109087, -0.0007426, 0.0007271
6: 0.0053643, 0.0097284, 0.0053643, 0.0097284, -0.0027745, 0.0028336
7: 0.9818129, 0.9848667, 0.9818129, 0.9848667, -0.0019415, 0.0019828
8: -0.0060636, -0.0027895, -0.0060636, -0.0027895, -0.0020816, 0.0021259
9: -0.0031570, -0.0009942, -0.0031570, -0.0009942, -0.0014043, 0.0013750

### Unstable ReLU Count (Linear/Conv2D Layers)
- layer_idx=0, type=LayerType.Linear, total=256, inp1_unstable=10, inp2_unstable=10, delta_unstable=250
- layer_idx=2, type=LayerType.Linear, total=256, inp1_unstable=10, inp2_unstable=10, delta_unstable=256
- layer_idx=4, type=LayerType.Linear, total=256, inp1_unstable=0, inp2_unstable=0, delta_unstable=256
- layer_idx=6, type=LayerType.Linear, total=256, inp1_unstable=0, inp2_unstable=0, delta_unstable=256
- layer_idx=8, type=LayerType.Linear, total=10, inp1_unstable=0, inp2_unstable=0, delta_unstable=10

Time for backsubstitution: 1.61 seconds

### RS candidates at layer 1
type: RSZ, layer: 1, pos: 140
type: RSZ, layer: 1, pos: 167
type: RSZ, layer: 1, pos: 213
type: RSZ, layer: 1, pos: 219

Time for candidate selection: 0.22 seconds

### Candidate
type: RSZ, layer: 1, pos: 140

### Relational analysis RSZ of RS_RSZ2_RSZ1_RSZ1_RSZ2_RSZ1_RSZ1_RSZ1
Optimization infeasible because this subproblem isn't reachable.

### Candidate
type: RSZ, layer: 1, pos: 167

### Relational analysis RSZ of RS_RSZ2_RSZ1_RSZ1_RSZ2_RSZ1_RSZ1_RSZ1

#### Relational analysis RSZ result of RS_RSZ2_RSZ1_RSZ1_RSZ2_RSZ1_RSZ1_RSZ1
Status: Status.UNKNOWN
Output dim: 7, lower bound: -0.0009426, upper bound: 0.0009341
time: 0.66 seconds

### Relational analysis RSZ of RS_RSZ2_RSZ1_RSZ1_RSZ2_RSZ1_RSZ1_RSZ2

#### Relational analysis RSZ result of RS_RSZ2_RSZ1_RSZ1_RSZ2_RSZ1_RSZ1_RSZ2
Status: Status.UNKNOWN
Output dim: 7, lower bound: -0.0009410, upper bound: 0.0009415
time: 0.69 seconds

## BFS RS instance: RS_RSZ2_RSZ1_RSZ1_RSZ2_RSZ1_RSZ2

### Backsubstitution after applying RS history:
0: -0.0027939, -0.0016242, -0.0027939, -0.0016242, -0.0007596, 0.0007435
1: -0.0114005, -0.0084323, -0.0114005, -0.0084323, -0.0019276, 0.0018867
2: 0.0279571, 0.0297986, 0.0279571, 0.0297986, -0.0011959, 0.0011705
3: 0.0039210, 0.0073596, 0.0039210, 0.0073596, -0.0021856, 0.0022330
4: -0.0104893, -0.0074701, -0.0104893, -0.0074701, -0.0019606, 0.0019191
5: 0.0097651, 0.0109087, 0.0097651, 0.0109087, -0.0007426, 0.0007269
6: 0.0053643, 0.0097284, 0.0053643, 0.0097284, -0.0027739, 0.0028339
7: 0.9818129, 0.9848667, 0.9818129, 0.9848667, -0.0019410, 0.0019831
8: -0.0060636, -0.0027895, -0.0060636, -0.0027895, -0.0020811, 0.0021261
9: -0.0031570, -0.0009942, -0.0031570, -0.0009942, -0.0014044, 0.0013747

### Unstable ReLU Count (Linear/Conv2D Layers)
- layer_idx=0, type=LayerType.Linear, total=256, inp1_unstable=10, inp2_unstable=10, delta_unstable=250
- layer_idx=2, type=LayerType.Linear, total=256, inp1_unstable=10, inp2_unstable=10, delta_unstable=256
- layer_idx=4, type=LayerType.Linear, total=256, inp1_unstable=0, inp2_unstable=0, delta_unstable=256
- layer_idx=6, type=LayerType.Linear, total=256, inp1_unstable=0, inp2_unstable=0, delta_unstable=256
- layer_idx=8, type=LayerType.Linear, total=10, inp1_unstable=0, inp2_unstable=0, delta_unstable=10

Time for backsubstitution: 1.62 seconds

### RS candidates at layer 1
type: RSZ, layer: 1, pos: 140
type: RSZ, layer: 1, pos: 167
type: RSZ, layer: 1, pos: 213
type: RSZ, layer: 1, pos: 219

Time for candidate selection: 0.17 seconds

### Candidate
type: RSZ, layer: 1, pos: 140

### Relational analysis RSZ of RS_RSZ2_RSZ1_RSZ1_RSZ2_RSZ1_RSZ2_RSZ1
Optimization infeasible because this subproblem isn't reachable.

### Candidate
type: RSZ, layer: 1, pos: 167

### Relational analysis RSZ of RS_RSZ2_RSZ1_RSZ1_RSZ2_RSZ1_RSZ2_RSZ1

#### Relational analysis RSZ result of RS_RSZ2_RSZ1_RSZ1_RSZ2_RSZ1_RSZ2_RSZ1
Status: Status.UNKNOWN
Output dim: 7, lower bound: -0.0009419, upper bound: 0.0009337
time: 0.65 seconds

### Relational analysis RSZ of RS_RSZ2_RSZ1_RSZ1_RSZ2_RSZ1_RSZ2_RSZ2

#### Relational analysis RSZ result of RS_RSZ2_RSZ1_RSZ1_RSZ2_RSZ1_RSZ2_RSZ2
Status: Status.UNKNOWN
Output dim: 7, lower bound: -0.0009400, upper bound: 0.0009414
time: 0.68 seconds

## BFS RS instance: RS_RSZ2_RSZ1_RSZ1_RSZ2_RSZ2_RSZ1

### Backsubstitution after applying RS history:
0: -0.0027939, -0.0016242, -0.0027939, -0.0016242, -0.0007451, 0.0007588
1: -0.0114005, -0.0084323, -0.0114005, -0.0084323, -0.0018909, 0.0019256
2: 0.0279571, 0.0297986, 0.0279571, 0.0297986, -0.0011731, 0.0011947
3: 0.0039210, 0.0073596, 0.0039210, 0.0073596, -0.0022307, 0.0021905
4: -0.0104893, -0.0074701, -0.0104893, -0.0074701, -0.0019234, 0.0019587
5: 0.0097651, 0.0109087, 0.0097651, 0.0109087, -0.0007285, 0.0007419
6: 0.0053643, 0.0097284, 0.0053643, 0.0097284, -0.0028311, 0.0027801
7: 0.9818129, 0.9848667, 0.9818129, 0.9848667, -0.0019811, 0.0019454
8: -0.0060636, -0.0027895, -0.0060636, -0.0027895, -0.0021240, 0.0020857
9: -0.0031570, -0.0009942, -0.0031570, -0.0009942, -0.0013777, 0.0014030

### Unstable ReLU Count (Linear/Conv2D Layers)
- layer_idx=0, type=LayerType.Linear, total=256, inp1_unstable=10, inp2_unstable=10, delta_unstable=250
- layer_idx=2, type=LayerType.Linear, total=256, inp1_unstable=10, inp2_unstable=10, delta_unstable=256
- layer_idx=4, type=LayerType.Linear, total=256, inp1_unstable=0, inp2_unstable=0, delta_unstable=256
- layer_idx=6, type=LayerType.Linear, total=256, inp1_unstable=0, inp2_unstable=0, delta_unstable=256
- layer_idx=8, type=LayerType.Linear, total=10, inp1_unstable=0, inp2_unstable=0, delta_unstable=10

Time for backsubstitution: 1.62 seconds

### RS candidates at layer 1
type: RSZ, layer: 1, pos: 140
type: RSZ, layer: 1, pos: 167
type: RSZ, layer: 1, pos: 213
type: RSZ, layer: 1, pos: 219

Time for candidate selection: 0.17 seconds

### Candidate
type: RSZ, layer: 1, pos: 140

### Relational analysis RSZ of RS_RSZ2_RSZ1_RSZ1_RSZ2_RSZ2_RSZ1_RSZ1
Optimization infeasible because this subproblem isn't reachable.

### Candidate
type: RSZ, layer: 1, pos: 167

### Relational analysis RSZ of RS_RSZ2_RSZ1_RSZ1_RSZ2_RSZ2_RSZ1_RSZ1

#### Relational analysis RSZ result of RS_RSZ2_RSZ1_RSZ1_RSZ2_RSZ2_RSZ1_RSZ1
Status: Status.UNKNOWN
Output dim: 7, lower bound: -0.0009553, upper bound: 0.0009246
time: 0.70 seconds

### Relational analysis RSZ of RS_RSZ2_RSZ1_RSZ1_RSZ2_RSZ2_RSZ1_RSZ2

#### Relational analysis RSZ result of RS_RSZ2_RSZ1_RSZ1_RSZ2_RSZ2_RSZ1_RSZ2
Status: Status.UNKNOWN
Output dim: 7, lower bound: -0.0009521, upper bound: 0.0009336
time: 0.66 seconds

## BFS RS instance: RS_RSZ2_RSZ1_RSZ1_RSZ2_RSZ2_RSZ2

### Backsubstitution after applying RS history:
0: -0.0027939, -0.0016242, -0.0027939, -0.0016242, -0.0007453, 0.0007587
1: -0.0114005, -0.0084323, -0.0114005, -0.0084323, -0.0018913, 0.0019253
2: 0.0279571, 0.0297986, 0.0279571, 0.0297986, -0.0011734, 0.0011945
3: 0.0039210, 0.0073596, 0.0039210, 0.0073596, -0.0022304, 0.0021910
4: -0.0104893, -0.0074701, -0.0104893, -0.0074701, -0.0019238, 0.0019584
5: 0.0097651, 0.0109087, 0.0097651, 0.0109087, -0.0007287, 0.0007418
6: 0.0053643, 0.0097284, 0.0053643, 0.0097284, -0.0028307, 0.0027807
7: 0.9818129, 0.9848667, 0.9818129, 0.9848667, -0.0019808, 0.0019458
8: -0.0060636, -0.0027895, -0.0060636, -0.0027895, -0.0021237, 0.0020862
9: -0.0031570, -0.0009942, -0.0031570, -0.0009942, -0.0013780, 0.0014028

### Unstable ReLU Count (Linear/Conv2D Layers)
- layer_idx=0, type=LayerType.Linear, total=256, inp1_unstable=10, inp2_unstable=10, delta_unstable=250
- layer_idx=2, type=LayerType.Linear, total=256, inp1_unstable=10, inp2_unstable=10, delta_unstable=256
- layer_idx=4, type=LayerType.Linear, total=256, inp1_unstable=0, inp2_unstable=0, delta_unstable=256
- layer_idx=6, type=LayerType.Linear, total=256, inp1_unstable=0, inp2_unstable=0, delta_unstable=256
- layer_idx=8, type=LayerType.Linear, total=10, inp1_unstable=0, inp2_unstable=0, delta_unstable=10

Time for backsubstitution: 1.59 seconds

### RS candidates at layer 1
type: RSZ, layer: 1, pos: 140
type: RSZ, layer: 1, pos: 167
type: RSZ, layer: 1, pos: 213
type: RSZ, layer: 1, pos: 219

Time for candidate selection: 0.16 seconds

### Candidate
type: RSZ, layer: 1, pos: 140

### Relational analysis RSZ of RS_RSZ2_RSZ1_RSZ1_RSZ2_RSZ2_RSZ2_RSZ1
Optimization infeasible because this subproblem isn't reachable.

### Candidate
type: RSZ, layer: 1, pos: 167

### Relational analysis RSZ of RS_RSZ2_RSZ1_RSZ1_RSZ2_RSZ2_RSZ2_RSZ1

#### Relational analysis RSZ result of RS_RSZ2_RSZ1_RSZ1_RSZ2_RSZ2_RSZ2_RSZ1
Status: Status.UNKNOWN
Output dim: 7, lower bound: -0.0009541, upper bound: 0.0009246
time: 0.64 seconds

### Relational analysis RSZ of RS_RSZ2_RSZ1_RSZ1_RSZ2_RSZ2_RSZ2_RSZ2

#### Relational analysis RSZ result of RS_RSZ2_RSZ1_RSZ1_RSZ2_RSZ2_RSZ2_RSZ2
Status: Status.UNKNOWN
Output dim: 7, lower bound: -0.0009512, upper bound: 0.0009336
time: 0.67 seconds

## BFS RS instance: RS_RSZ2_RSZ1_RSZ2_RSZ1_RSZ1_RSZ1

### Backsubstitution after applying RS history:
0: -0.0027939, -0.0016242, -0.0027939, -0.0016242, -0.0007530, 0.0007481
1: -0.0114005, -0.0084323, -0.0114005, -0.0084323, -0.0019109, 0.0018984
2: 0.0279571, 0.0297986, 0.0279571, 0.0297986, -0.0011855, 0.0011778
3: 0.0039210, 0.0073596, 0.0039210, 0.0073596, -0.0021992, 0.0022137
4: -0.0104893, -0.0074701, -0.0104893, -0.0074701, -0.0019437, 0.0019310
5: 0.0097651, 0.0109087, 0.0097651, 0.0109087, -0.0007362, 0.0007314
6: 0.0053643, 0.0097284, 0.0053643, 0.0097284, -0.0027911, 0.0028095
7: 0.9818129, 0.9848667, 0.9818129, 0.9848667, -0.0019530, 0.0019659
8: -0.0060636, -0.0027895, -0.0060636, -0.0027895, -0.0020940, 0.0021078
9: -0.0031570, -0.0009942, -0.0031570, -0.0009942, -0.0013923, 0.0013832

### Unstable ReLU Count (Linear/Conv2D Layers)
- layer_idx=0, type=LayerType.Linear, total=256, inp1_unstable=10, inp2_unstable=10, delta_unstable=250
- layer_idx=2, type=LayerType.Linear, total=256, inp1_unstable=10, inp2_unstable=10, delta_unstable=256
- layer_idx=4, type=LayerType.Linear, total=256, inp1_unstable=0, inp2_unstable=0, delta_unstable=256
- layer_idx=6, type=LayerType.Linear, total=256, inp1_unstable=0, inp2_unstable=0, delta_unstable=256
- layer_idx=8, type=LayerType.Linear, total=10, inp1_unstable=0, inp2_unstable=0, delta_unstable=10

Time for backsubstitution: 1.82 seconds

### RS candidates at layer 1
type: RSZ, layer: 1, pos: 140
type: RSZ, layer: 1, pos: 167
type: RSZ, layer: 1, pos: 213
type: RSZ, layer: 1, pos: 219

Time for candidate selection: 0.22 seconds

### Candidate
type: RSZ, layer: 1, pos: 140

### Relational analysis RSZ of RS_RSZ2_RSZ1_RSZ2_RSZ1_RSZ1_RSZ1_RSZ1
Optimization infeasible because this subproblem isn't reachable.

### Candidate
type: RSZ, layer: 1, pos: 167

### Relational analysis RSZ of RS_RSZ2_RSZ1_RSZ2_RSZ1_RSZ1_RSZ1_RSZ1

#### Relational analysis RSZ result of RS_RSZ2_RSZ1_RSZ2_RSZ1_RSZ1_RSZ1_RSZ1
Status: Status.UNKNOWN
Output dim: 7, lower bound: -0.0009327, upper bound: 0.0009355
time: 0.67 seconds

### Relational analysis RSZ of RS_RSZ2_RSZ1_RSZ2_RSZ1_RSZ1_RSZ1_RSZ2

#### Relational analysis RSZ result of RS_RSZ2_RSZ1_RSZ2_RSZ1_RSZ1_RSZ1_RSZ2
Status: Status.UNKNOWN
Output dim: 7, lower bound: -0.0009291, upper bound: 0.0009420
time: 0.64 seconds

## BFS RS instance: RS_RSZ2_RSZ1_RSZ2_RSZ1_RSZ1_RSZ2

### Backsubstitution after applying RS history:
0: -0.0027939, -0.0016242, -0.0027939, -0.0016242, -0.0007532, 0.0007479
1: -0.0114005, -0.0084323, -0.0114005, -0.0084323, -0.0019113, 0.0018978
2: 0.0279571, 0.0297986, 0.0279571, 0.0297986, -0.0011858, 0.0011774
3: 0.0039210, 0.0073596, 0.0039210, 0.0073596, -0.0021985, 0.0022141
4: -0.0104893, -0.0074701, -0.0104893, -0.0074701, -0.0019441, 0.0019304
5: 0.0097651, 0.0109087, 0.0097651, 0.0109087, -0.0007364, 0.0007312
6: 0.0053643, 0.0097284, 0.0053643, 0.0097284, -0.0027902, 0.0028100
7: 0.9818129, 0.9848667, 0.9818129, 0.9848667, -0.0019525, 0.0019663
8: -0.0060636, -0.0027895, -0.0060636, -0.0027895, -0.0020933, 0.0021082
9: -0.0031570, -0.0009942, -0.0031570, -0.0009942, -0.0013926, 0.0013828

### Unstable ReLU Count (Linear/Conv2D Layers)
- layer_idx=0, type=LayerType.Linear, total=256, inp1_unstable=10, inp2_unstable=10, delta_unstable=250
- layer_idx=2, type=LayerType.Linear, total=256, inp1_unstable=10, inp2_unstable=10, delta_unstable=256
- layer_idx=4, type=LayerType.Linear, total=256, inp1_unstable=0, inp2_unstable=0, delta_unstable=256
- layer_idx=6, type=LayerType.Linear, total=256, inp1_unstable=0, inp2_unstable=0, delta_unstable=256
- layer_idx=8, type=LayerType.Linear, total=10, inp1_unstable=0, inp2_unstable=0, delta_unstable=10

Time for backsubstitution: 1.65 seconds

### RS candidates at layer 1
type: RSZ, layer: 1, pos: 140
type: RSZ, layer: 1, pos: 167
type: RSZ, layer: 1, pos: 213
type: RSZ, layer: 1, pos: 219

Time for candidate selection: 0.18 seconds

### Candidate
type: RSZ, layer: 1, pos: 140

### Relational analysis RSZ of RS_RSZ2_RSZ1_RSZ2_RSZ1_RSZ1_RSZ2_RSZ1
Optimization infeasible because this subproblem isn't reachable.

### Candidate
type: RSZ, layer: 1, pos: 167

### Relational analysis RSZ of RS_RSZ2_RSZ1_RSZ2_RSZ1_RSZ1_RSZ2_RSZ1

#### Relational analysis RSZ result of RS_RSZ2_RSZ1_RSZ2_RSZ1_RSZ1_RSZ2_RSZ1
Status: Status.UNKNOWN
Output dim: 7, lower bound: -0.0009321, upper bound: 0.0009349
time: 0.68 seconds

### Relational analysis RSZ of RS_RSZ2_RSZ1_RSZ2_RSZ1_RSZ1_RSZ2_RSZ2

#### Relational analysis RSZ result of RS_RSZ2_RSZ1_RSZ2_RSZ1_RSZ1_RSZ2_RSZ2
Status: Status.UNKNOWN
Output dim: 7, lower bound: -0.0009284, upper bound: 0.0009414
time: 0.67 seconds

## BFS RS instance: RS_RSZ2_RSZ1_RSZ2_RSZ1_RSZ2_RSZ1

### Backsubstitution after applying RS history:
0: -0.0027939, -0.0016242, -0.0027939, -0.0016242, -0.0007382, 0.0007635
1: -0.0114005, -0.0084323, -0.0114005, -0.0084323, -0.0018734, 0.0019374
2: 0.0279571, 0.0297986, 0.0279571, 0.0297986, -0.0011623, 0.0012020
3: 0.0039210, 0.0073596, 0.0039210, 0.0073596, -0.0022444, 0.0021702
4: -0.0104893, -0.0074701, -0.0104893, -0.0074701, -0.0019056, 0.0019707
5: 0.0097651, 0.0109087, 0.0097651, 0.0109087, -0.0007218, 0.0007464
6: 0.0053643, 0.0097284, 0.0053643, 0.0097284, -0.0028484, 0.0027543
7: 0.9818129, 0.9848667, 0.9818129, 0.9848667, -0.0019932, 0.0019273
8: -0.0060636, -0.0027895, -0.0060636, -0.0027895, -0.0021370, 0.0020664
9: -0.0031570, -0.0009942, -0.0031570, -0.0009942, -0.0013650, 0.0014116

### Unstable ReLU Count (Linear/Conv2D Layers)
- layer_idx=0, type=LayerType.Linear, total=256, inp1_unstable=10, inp2_unstable=10, delta_unstable=250
- layer_idx=2, type=LayerType.Linear, total=256, inp1_unstable=10, inp2_unstable=10, delta_unstable=256
- layer_idx=4, type=LayerType.Linear, total=256, inp1_unstable=0, inp2_unstable=0, delta_unstable=256
- layer_idx=6, type=LayerType.Linear, total=256, inp1_unstable=0, inp2_unstable=0, delta_unstable=256
- layer_idx=8, type=LayerType.Linear, total=10, inp1_unstable=0, inp2_unstable=0, delta_unstable=10

Time for backsubstitution: 1.57 seconds

### RS candidates at layer 1
type: RSZ, layer: 1, pos: 140
type: RSZ, layer: 1, pos: 167
type: RSZ, layer: 1, pos: 213
type: RSZ, layer: 1, pos: 219

Time for candidate selection: 0.18 seconds

### Candidate
type: RSZ, layer: 1, pos: 140

### Relational analysis RSZ of RS_RSZ2_RSZ1_RSZ2_RSZ1_RSZ2_RSZ1_RSZ1
Optimization infeasible because this subproblem isn't reachable.

### Candidate
type: RSZ, layer: 1, pos: 167

### Relational analysis RSZ of RS_RSZ2_RSZ1_RSZ2_RSZ1_RSZ2_RSZ1_RSZ1

#### Relational analysis RSZ result of RS_RSZ2_RSZ1_RSZ2_RSZ1_RSZ2_RSZ1_RSZ1
Status: Status.UNKNOWN
Output dim: 7, lower bound: -0.0009504, upper bound: 0.0009266
time: 0.70 seconds

### Relational analysis RSZ of RS_RSZ2_RSZ1_RSZ2_RSZ1_RSZ2_RSZ1_RSZ2

#### Relational analysis RSZ result of RS_RSZ2_RSZ1_RSZ2_RSZ1_RSZ2_RSZ1_RSZ2
Status: Status.UNKNOWN
Output dim: 7, lower bound: -0.0009445, upper bound: 0.0009342
time: 0.67 seconds

## BFS RS instance: RS_RSZ2_RSZ1_RSZ2_RSZ1_RSZ2_RSZ2

### Backsubstitution after applying RS history:
0: -0.0027939, -0.0016242, -0.0027939, -0.0016242, -0.0007384, 0.0007634
1: -0.0114005, -0.0084323, -0.0114005, -0.0084323, -0.0018737, 0.0019373
2: 0.0279571, 0.0297986, 0.0279571, 0.0297986, -0.0011624, 0.0012019
3: 0.0039210, 0.0073596, 0.0039210, 0.0073596, -0.0022442, 0.0021706
4: -0.0104893, -0.0074701, -0.0104893, -0.0074701, -0.0019059, 0.0019705
5: 0.0097651, 0.0109087, 0.0097651, 0.0109087, -0.0007219, 0.0007464
6: 0.0053643, 0.0097284, 0.0053643, 0.0097284, -0.0028482, 0.0027548
7: 0.9818129, 0.9848667, 0.9818129, 0.9848667, -0.0019930, 0.0019276
8: -0.0060636, -0.0027895, -0.0060636, -0.0027895, -0.0021368, 0.0020667
9: -0.0031570, -0.0009942, -0.0031570, -0.0009942, -0.0013652, 0.0014115

### Unstable ReLU Count (Linear/Conv2D Layers)
- layer_idx=0, type=LayerType.Linear, total=256, inp1_unstable=10, inp2_unstable=10, delta_unstable=250
- layer_idx=2, type=LayerType.Linear, total=256, inp1_unstable=10, inp2_unstable=10, delta_unstable=256
- layer_idx=4, type=LayerType.Linear, total=256, inp1_unstable=0, inp2_unstable=0, delta_unstable=256
- layer_idx=6, type=LayerType.Linear, total=256, inp1_unstable=0, inp2_unstable=0, delta_unstable=256
- layer_idx=8, type=LayerType.Linear, total=10, inp1_unstable=0, inp2_unstable=0, delta_unstable=10

Time for backsubstitution: 1.62 seconds

### RS candidates at layer 1
type: RSZ, layer: 1, pos: 140
type: RSZ, layer: 1, pos: 167
type: RSZ, layer: 1, pos: 213
type: RSZ, layer: 1, pos: 219

Time for candidate selection: 0.16 seconds

### Candidate
type: RSZ, layer: 1, pos: 140

### Relational analysis RSZ of RS_RSZ2_RSZ1_RSZ2_RSZ1_RSZ2_RSZ2_RSZ1
Optimization infeasible because this subproblem isn't reachable.

### Candidate
type: RSZ, layer: 1, pos: 167

### Relational analysis RSZ of RS_RSZ2_RSZ1_RSZ2_RSZ1_RSZ2_RSZ2_RSZ1

#### Relational analysis RSZ result of RS_RSZ2_RSZ1_RSZ2_RSZ1_RSZ2_RSZ2_RSZ1
Status: Status.UNKNOWN
Output dim: 7, lower bound: -0.0009485, upper bound: 0.0009266
time: 0.65 seconds

### Relational analysis RSZ of RS_RSZ2_RSZ1_RSZ2_RSZ1_RSZ2_RSZ2_RSZ2

#### Relational analysis RSZ result of RS_RSZ2_RSZ1_RSZ2_RSZ1_RSZ2_RSZ2_RSZ2
Status: Status.UNKNOWN
Output dim: 7, lower bound: -0.0009428, upper bound: 0.0009341
time: 0.66 seconds

## BFS RS instance: RS_RSZ2_RSZ1_RSZ2_RSZ2_RSZ1_RSZ1

### Backsubstitution after applying RS history:
0: -0.0027939, -0.0016242, -0.0027939, -0.0016242, -0.0007611, 0.0007421
1: -0.0114005, -0.0084323, -0.0114005, -0.0084323, -0.0019315, 0.0018833
2: 0.0279571, 0.0297986, 0.0279571, 0.0297986, -0.0011983, 0.0011684
3: 0.0039210, 0.0073596, 0.0039210, 0.0073596, -0.0021817, 0.0022375
4: -0.0104893, -0.0074701, -0.0104893, -0.0074701, -0.0019647, 0.0019156
5: 0.0097651, 0.0109087, 0.0097651, 0.0109087, -0.0007442, 0.0007256
6: 0.0053643, 0.0097284, 0.0053643, 0.0097284, -0.0027688, 0.0028397
7: 0.9818129, 0.9848667, 0.9818129, 0.9848667, -0.0019375, 0.0019871
8: -0.0060636, -0.0027895, -0.0060636, -0.0027895, -0.0020773, 0.0021305
9: -0.0031570, -0.0009942, -0.0031570, -0.0009942, -0.0014073, 0.0013722

### Unstable ReLU Count (Linear/Conv2D Layers)
- layer_idx=0, type=LayerType.Linear, total=256, inp1_unstable=10, inp2_unstable=10, delta_unstable=250
- layer_idx=2, type=LayerType.Linear, total=256, inp1_unstable=10, inp2_unstable=10, delta_unstable=256
- layer_idx=4, type=LayerType.Linear, total=256, inp1_unstable=0, inp2_unstable=0, delta_unstable=256
- layer_idx=6, type=LayerType.Linear, total=256, inp1_unstable=0, inp2_unstable=0, delta_unstable=256
- layer_idx=8, type=LayerType.Linear, total=10, inp1_unstable=0, inp2_unstable=0, delta_unstable=10

Time for backsubstitution: 1.60 seconds

### RS candidates at layer 1
type: RSZ, layer: 1, pos: 140
type: RSZ, layer: 1, pos: 167
type: RSZ, layer: 1, pos: 213
type: RSZ, layer: 1, pos: 219

Time for candidate selection: 0.19 seconds

### Candidate
type: RSZ, layer: 1, pos: 140

### Relational analysis RSZ of RS_RSZ2_RSZ1_RSZ2_RSZ2_RSZ1_RSZ1_RSZ1
Optimization infeasible because this subproblem isn't reachable.

### Candidate
type: RSZ, layer: 1, pos: 167

### Relational analysis RSZ of RS_RSZ2_RSZ1_RSZ2_RSZ2_RSZ1_RSZ1_RSZ1

#### Relational analysis RSZ result of RS_RSZ2_RSZ1_RSZ2_RSZ2_RSZ1_RSZ1_RSZ1
Status: Status.UNKNOWN
Output dim: 7, lower bound: -0.0009426, upper bound: 0.0009341
time: 0.64 seconds

### Relational analysis RSZ of RS_RSZ2_RSZ1_RSZ2_RSZ2_RSZ1_RSZ1_RSZ2

#### Relational analysis RSZ result of RS_RSZ2_RSZ1_RSZ2_RSZ2_RSZ1_RSZ1_RSZ2
Status: Status.UNKNOWN
Output dim: 7, lower bound: -0.0009410, upper bound: 0.0009415
time: 0.67 seconds

## BFS RS instance: RS_RSZ2_RSZ1_RSZ2_RSZ2_RSZ1_RSZ2

### Backsubstitution after applying RS history:
0: -0.0027939, -0.0016242, -0.0027939, -0.0016242, -0.0007612, 0.0007419
1: -0.0114005, -0.0084323, -0.0114005, -0.0084323, -0.0019317, 0.0018828
2: 0.0279571, 0.0297986, 0.0279571, 0.0297986, -0.0011984, 0.0011681
3: 0.0039210, 0.0073596, 0.0039210, 0.0073596, -0.0021811, 0.0022378
4: -0.0104893, -0.0074701, -0.0104893, -0.0074701, -0.0019649, 0.0019151
5: 0.0097651, 0.0109087, 0.0097651, 0.0109087, -0.0007442, 0.0007254
6: 0.0053643, 0.0097284, 0.0053643, 0.0097284, -0.0027681, 0.0028401
7: 0.9818129, 0.9848667, 0.9818129, 0.9848667, -0.0019370, 0.0019873
8: -0.0060636, -0.0027895, -0.0060636, -0.0027895, -0.0020768, 0.0021307
9: -0.0031570, -0.0009942, -0.0031570, -0.0009942, -0.0014075, 0.0013718

### Unstable ReLU Count (Linear/Conv2D Layers)
- layer_idx=0, type=LayerType.Linear, total=256, inp1_unstable=10, inp2_unstable=10, delta_unstable=250
- layer_idx=2, type=LayerType.Linear, total=256, inp1_unstable=10, inp2_unstable=10, delta_unstable=256
- layer_idx=4, type=LayerType.Linear, total=256, inp1_unstable=0, inp2_unstable=0, delta_unstable=256
- layer_idx=6, type=LayerType.Linear, total=256, inp1_unstable=0, inp2_unstable=0, delta_unstable=256
- layer_idx=8, type=LayerType.Linear, total=10, inp1_unstable=0, inp2_unstable=0, delta_unstable=10

Time for backsubstitution: 1.63 seconds

### RS candidates at layer 1
type: RSZ, layer: 1, pos: 140
type: RSZ, layer: 1, pos: 167
type: RSZ, layer: 1, pos: 213
type: RSZ, layer: 1, pos: 219

Time for candidate selection: 0.17 seconds

### Candidate
type: RSZ, layer: 1, pos: 140

### Relational analysis RSZ of RS_RSZ2_RSZ1_RSZ2_RSZ2_RSZ1_RSZ2_RSZ1
Optimization infeasible because this subproblem isn't reachable.

### Candidate
type: RSZ, layer: 1, pos: 167

### Relational analysis RSZ of RS_RSZ2_RSZ1_RSZ2_RSZ2_RSZ1_RSZ2_RSZ1

#### Relational analysis RSZ result of RS_RSZ2_RSZ1_RSZ2_RSZ2_RSZ1_RSZ2_RSZ1
Status: Status.UNKNOWN
Output dim: 7, lower bound: -0.0009419, upper bound: 0.0009337
time: 0.65 seconds

### Relational analysis RSZ of RS_RSZ2_RSZ1_RSZ2_RSZ2_RSZ1_RSZ2_RSZ2

#### Relational analysis RSZ result of RS_RSZ2_RSZ1_RSZ2_RSZ2_RSZ1_RSZ2_RSZ2
Status: Status.UNKNOWN
Output dim: 7, lower bound: -0.0009400, upper bound: 0.0009414
time: 0.69 seconds

## BFS RS instance: RS_RSZ2_RSZ1_RSZ2_RSZ2_RSZ2_RSZ1

### Backsubstitution after applying RS history:
0: -0.0027939, -0.0016242, -0.0027939, -0.0016242, -0.0007468, 0.0007566
1: -0.0114005, -0.0084323, -0.0114005, -0.0084323, -0.0018951, 0.0019201
2: 0.0279571, 0.0297986, 0.0279571, 0.0297986, -0.0011757, 0.0011912
3: 0.0039210, 0.0073596, 0.0039210, 0.0073596, -0.0022243, 0.0021954
4: -0.0104893, -0.0074701, -0.0104893, -0.0074701, -0.0019276, 0.0019531
5: 0.0097651, 0.0109087, 0.0097651, 0.0109087, -0.0007301, 0.0007398
6: 0.0053643, 0.0097284, 0.0053643, 0.0097284, -0.0028230, 0.0027862
7: 0.9818129, 0.9848667, 0.9818129, 0.9848667, -0.0019754, 0.0019496
8: -0.0060636, -0.0027895, -0.0060636, -0.0027895, -0.0021179, 0.0020903
9: -0.0031570, -0.0009942, -0.0031570, -0.0009942, -0.0013808, 0.0013990

### Unstable ReLU Count (Linear/Conv2D Layers)
- layer_idx=0, type=LayerType.Linear, total=256, inp1_unstable=10, inp2_unstable=10, delta_unstable=250
- layer_idx=2, type=LayerType.Linear, total=256, inp1_unstable=10, inp2_unstable=10, delta_unstable=256
- layer_idx=4, type=LayerType.Linear, total=256, inp1_unstable=0, inp2_unstable=0, delta_unstable=256
- layer_idx=6, type=LayerType.Linear, total=256, inp1_unstable=0, inp2_unstable=0, delta_unstable=256
- layer_idx=8, type=LayerType.Linear, total=10, inp1_unstable=0, inp2_unstable=0, delta_unstable=10

Time for backsubstitution: 1.66 seconds

### RS candidates at layer 1
type: RSZ, layer: 1, pos: 140
type: RSZ, layer: 1, pos: 167
type: RSZ, layer: 1, pos: 213
type: RSZ, layer: 1, pos: 219

Time for candidate selection: 0.17 seconds

### Candidate
type: RSZ, layer: 1, pos: 140

### Relational analysis RSZ of RS_RSZ2_RSZ1_RSZ2_RSZ2_RSZ2_RSZ1_RSZ1
Optimization infeasible because this subproblem isn't reachable.

### Candidate
type: RSZ, layer: 1, pos: 167

### Relational analysis RSZ of RS_RSZ2_RSZ1_RSZ2_RSZ2_RSZ2_RSZ1_RSZ1

#### Relational analysis RSZ result of RS_RSZ2_RSZ1_RSZ2_RSZ2_RSZ2_RSZ1_RSZ1
Status: Status.UNKNOWN
Output dim: 7, lower bound: -0.0009553, upper bound: 0.0009246
time: 0.67 seconds

### Relational analysis RSZ of RS_RSZ2_RSZ1_RSZ2_RSZ2_RSZ2_RSZ1_RSZ2

#### Relational analysis RSZ result of RS_RSZ2_RSZ1_RSZ2_RSZ2_RSZ2_RSZ1_RSZ2
Status: Status.UNKNOWN
Output dim: 7, lower bound: -0.0009521, upper bound: 0.0009336
time: 0.68 seconds

## BFS RS instance: RS_RSZ2_RSZ1_RSZ2_RSZ2_RSZ2_RSZ2

### Backsubstitution after applying RS history:
0: -0.0027939, -0.0016242, -0.0027939, -0.0016242, -0.0007469, 0.0007565
1: -0.0114005, -0.0084323, -0.0114005, -0.0084323, -0.0018955, 0.0019197
2: 0.0279571, 0.0297986, 0.0279571, 0.0297986, -0.0011760, 0.0011910
3: 0.0039210, 0.0073596, 0.0039210, 0.0073596, -0.0022239, 0.0021958
4: -0.0104893, -0.0074701, -0.0104893, -0.0074701, -0.0019280, 0.0019527
5: 0.0097651, 0.0109087, 0.0097651, 0.0109087, -0.0007303, 0.0007396
6: 0.0053643, 0.0097284, 0.0053643, 0.0097284, -0.0028224, 0.0027868
7: 0.9818129, 0.9848667, 0.9818129, 0.9848667, -0.0019750, 0.0019501
8: -0.0060636, -0.0027895, -0.0060636, -0.0027895, -0.0021175, 0.0020908
9: -0.0031570, -0.0009942, -0.0031570, -0.0009942, -0.0013811, 0.0013987

### Unstable ReLU Count (Linear/Conv2D Layers)
- layer_idx=0, type=LayerType.Linear, total=256, inp1_unstable=10, inp2_unstable=10, delta_unstable=250
- layer_idx=2, type=LayerType.Linear, total=256, inp1_unstable=10, inp2_unstable=10, delta_unstable=256
- layer_idx=4, type=LayerType.Linear, total=256, inp1_unstable=0, inp2_unstable=0, delta_unstable=256
- layer_idx=6, type=LayerType.Linear, total=256, inp1_unstable=0, inp2_unstable=0, delta_unstable=256
- layer_idx=8, type=LayerType.Linear, total=10, inp1_unstable=0, inp2_unstable=0, delta_unstable=10

Time for backsubstitution: 1.61 seconds

### RS candidates at layer 1
type: RSZ, layer: 1, pos: 140
type: RSZ, layer: 1, pos: 167
type: RSZ, layer: 1, pos: 213
type: RSZ, layer: 1, pos: 219

Time for candidate selection: 0.18 seconds

### Candidate
type: RSZ, layer: 1, pos: 140

### Relational analysis RSZ of RS_RSZ2_RSZ1_RSZ2_RSZ2_RSZ2_RSZ2_RSZ1
Optimization infeasible because this subproblem isn't reachable.

### Candidate
type: RSZ, layer: 1, pos: 167

### Relational analysis RSZ of RS_RSZ2_RSZ1_RSZ2_RSZ2_RSZ2_RSZ2_RSZ1

#### Relational analysis RSZ result of RS_RSZ2_RSZ1_RSZ2_RSZ2_RSZ2_RSZ2_RSZ1
Status: Status.UNKNOWN
Output dim: 7, lower bound: -0.0009541, upper bound: 0.0009246
time: 0.65 seconds

### Relational analysis RSZ of RS_RSZ2_RSZ1_RSZ2_RSZ2_RSZ2_RSZ2_RSZ2

#### Relational analysis RSZ result of RS_RSZ2_RSZ1_RSZ2_RSZ2_RSZ2_RSZ2_RSZ2
Status: Status.UNKNOWN
Output dim: 7, lower bound: -0.0009512, upper bound: 0.0009336
time: 0.67 seconds

## BFS RS instance: RS_RSZ2_RSZ2_RSZ1_RSZ1_RSZ1_RSZ1

### Backsubstitution after applying RS history:
0: -0.0027939, -0.0016242, -0.0027939, -0.0016242, -0.0007577, 0.0007451
1: -0.0114005, -0.0084323, -0.0114005, -0.0084323, -0.0019227, 0.0018907
2: 0.0279571, 0.0297986, 0.0279571, 0.0297986, -0.0011928, 0.0011730
3: 0.0039210, 0.0073596, 0.0039210, 0.0073596, -0.0021903, 0.0022273
4: -0.0104893, -0.0074701, -0.0104893, -0.0074701, -0.0019557, 0.0019231
5: 0.0097651, 0.0109087, 0.0097651, 0.0109087, -0.0007408, 0.0007284
6: 0.0053643, 0.0097284, 0.0053643, 0.0097284, -0.0027797, 0.0028267
7: 0.9818129, 0.9848667, 0.9818129, 0.9848667, -0.0019451, 0.0019780
8: -0.0060636, -0.0027895, -0.0060636, -0.0027895, -0.0020855, 0.0021207
9: -0.0031570, -0.0009942, -0.0031570, -0.0009942, -0.0014009, 0.0013776

### Unstable ReLU Count (Linear/Conv2D Layers)
- layer_idx=0, type=LayerType.Linear, total=256, inp1_unstable=10, inp2_unstable=10, delta_unstable=250
- layer_idx=2, type=LayerType.Linear, total=256, inp1_unstable=10, inp2_unstable=10, delta_unstable=256
- layer_idx=4, type=LayerType.Linear, total=256, inp1_unstable=0, inp2_unstable=0, delta_unstable=256
- layer_idx=6, type=LayerType.Linear, total=256, inp1_unstable=0, inp2_unstable=0, delta_unstable=256
- layer_idx=8, type=LayerType.Linear, total=10, inp1_unstable=0, inp2_unstable=0, delta_unstable=10

Time for backsubstitution: 1.60 seconds

### RS candidates at layer 1
type: RSZ, layer: 1, pos: 140
type: RSZ, layer: 1, pos: 167
type: RSZ, layer: 1, pos: 213
type: RSZ, layer: 1, pos: 219

Time for candidate selection: 0.17 seconds

### Candidate
type: RSZ, layer: 1, pos: 140

### Relational analysis RSZ of RS_RSZ2_RSZ2_RSZ1_RSZ1_RSZ1_RSZ1_RSZ1
Optimization infeasible because this subproblem isn't reachable.

### Candidate
type: RSZ, layer: 1, pos: 167

### Relational analysis RSZ of RS_RSZ2_RSZ2_RSZ1_RSZ1_RSZ1_RSZ1_RSZ1

#### Relational analysis RSZ result of RS_RSZ2_RSZ2_RSZ1_RSZ1_RSZ1_RSZ1_RSZ1
Status: Status.UNKNOWN
Output dim: 7, lower bound: -0.0009244, upper bound: 0.0009559
time: 0.68 seconds

### Relational analysis RSZ of RS_RSZ2_RSZ2_RSZ1_RSZ1_RSZ1_RSZ1_RSZ2

#### Relational analysis RSZ result of RS_RSZ2_RSZ2_RSZ1_RSZ1_RSZ1_RSZ1_RSZ2
Status: Status.UNKNOWN
Output dim: 7, lower bound: -0.0009189, upper bound: 0.0009607
time: 0.67 seconds

## BFS RS instance: RS_RSZ2_RSZ2_RSZ1_RSZ1_RSZ1_RSZ2

### Backsubstitution after applying RS history:
0: -0.0027939, -0.0016242, -0.0027939, -0.0016242, -0.0007578, 0.0007449
1: -0.0114005, -0.0084323, -0.0114005, -0.0084323, -0.0019231, 0.0018902
2: 0.0279571, 0.0297986, 0.0279571, 0.0297986, -0.0011931, 0.0011727
3: 0.0039210, 0.0073596, 0.0039210, 0.0073596, -0.0021897, 0.0022278
4: -0.0104893, -0.0074701, -0.0104893, -0.0074701, -0.0019561, 0.0019227
5: 0.0097651, 0.0109087, 0.0097651, 0.0109087, -0.0007409, 0.0007283
6: 0.0053643, 0.0097284, 0.0053643, 0.0097284, -0.0027791, 0.0028273
7: 0.9818129, 0.9848667, 0.9818129, 0.9848667, -0.0019447, 0.0019784
8: -0.0060636, -0.0027895, -0.0060636, -0.0027895, -0.0020850, 0.0021212
9: -0.0031570, -0.0009942, -0.0031570, -0.0009942, -0.0014012, 0.0013772

### Unstable ReLU Count (Linear/Conv2D Layers)
- layer_idx=0, type=LayerType.Linear, total=256, inp1_unstable=10, inp2_unstable=10, delta_unstable=250
- layer_idx=2, type=LayerType.Linear, total=256, inp1_unstable=10, inp2_unstable=10, delta_unstable=256
- layer_idx=4, type=LayerType.Linear, total=256, inp1_unstable=0, inp2_unstable=0, delta_unstable=256
- layer_idx=6, type=LayerType.Linear, total=256, inp1_unstable=0, inp2_unstable=0, delta_unstable=256
- layer_idx=8, type=LayerType.Linear, total=10, inp1_unstable=0, inp2_unstable=0, delta_unstable=10

Time for backsubstitution: 1.56 seconds

### RS candidates at layer 1
type: RSZ, layer: 1, pos: 140
type: RSZ, layer: 1, pos: 167
type: RSZ, layer: 1, pos: 213
type: RSZ, layer: 1, pos: 219

Time for candidate selection: 0.17 seconds

### Candidate
type: RSZ, layer: 1, pos: 140

### Relational analysis RSZ of RS_RSZ2_RSZ2_RSZ1_RSZ1_RSZ1_RSZ2_RSZ1
Optimization infeasible because this subproblem isn't reachable.

### Candidate
type: RSZ, layer: 1, pos: 167

### Relational analysis RSZ of RS_RSZ2_RSZ2_RSZ1_RSZ1_RSZ1_RSZ2_RSZ1

#### Relational analysis RSZ result of RS_RSZ2_RSZ2_RSZ1_RSZ1_RSZ1_RSZ2_RSZ1
Status: Status.UNKNOWN
Output dim: 7, lower bound: -0.0009243, upper bound: 0.0009566
time: 0.67 seconds

### Relational analysis RSZ of RS_RSZ2_RSZ2_RSZ1_RSZ1_RSZ1_RSZ2_RSZ2

#### Relational analysis RSZ result of RS_RSZ2_RSZ2_RSZ1_RSZ1_RSZ1_RSZ2_RSZ2
Status: Status.UNKNOWN
Output dim: 7, lower bound: -0.0009189, upper bound: 0.0009614
time: 0.66 seconds

## BFS RS instance: RS_RSZ2_RSZ2_RSZ1_RSZ1_RSZ2_RSZ1

### Backsubstitution after applying RS history:
0: -0.0027939, -0.0016242, -0.0027939, -0.0016242, -0.0007432, 0.0007597
1: -0.0114005, -0.0084323, -0.0114005, -0.0084323, -0.0018860, 0.0019277
2: 0.0279571, 0.0297986, 0.0279571, 0.0297986, -0.0011701, 0.0011960
3: 0.0039210, 0.0073596, 0.0039210, 0.0073596, -0.0022332, 0.0021848
4: -0.0104893, -0.0074701, -0.0104893, -0.0074701, -0.0019183, 0.0019608
5: 0.0097651, 0.0109087, 0.0097651, 0.0109087, -0.0007266, 0.0007427
6: 0.0053643, 0.0097284, 0.0053643, 0.0097284, -0.0028342, 0.0027728
7: 0.9818129, 0.9848667, 0.9818129, 0.9848667, -0.0019832, 0.0019403
8: -0.0060636, -0.0027895, -0.0060636, -0.0027895, -0.0021263, 0.0020803
9: -0.0031570, -0.0009942, -0.0031570, -0.0009942, -0.0013741, 0.0014046

### Unstable ReLU Count (Linear/Conv2D Layers)
- layer_idx=0, type=LayerType.Linear, total=256, inp1_unstable=10, inp2_unstable=10, delta_unstable=250
- layer_idx=2, type=LayerType.Linear, total=256, inp1_unstable=10, inp2_unstable=10, delta_unstable=256
- layer_idx=4, type=LayerType.Linear, total=256, inp1_unstable=0, inp2_unstable=0, delta_unstable=256
- layer_idx=6, type=LayerType.Linear, total=256, inp1_unstable=0, inp2_unstable=0, delta_unstable=256
- layer_idx=8, type=LayerType.Linear, total=10, inp1_unstable=0, inp2_unstable=0, delta_unstable=10

Time for backsubstitution: 1.61 seconds

### RS candidates at layer 1
type: RSZ, layer: 1, pos: 140
type: RSZ, layer: 1, pos: 167
type: RSZ, layer: 1, pos: 213
type: RSZ, layer: 1, pos: 219

Time for candidate selection: 0.17 seconds

### Candidate
type: RSZ, layer: 1, pos: 140

### Relational analysis RSZ of RS_RSZ2_RSZ2_RSZ1_RSZ1_RSZ2_RSZ1_RSZ1
Optimization infeasible because this subproblem isn't reachable.

### Candidate
type: RSZ, layer: 1, pos: 167

### Relational analysis RSZ of RS_RSZ2_RSZ2_RSZ1_RSZ1_RSZ2_RSZ1_RSZ1

#### Relational analysis RSZ result of RS_RSZ2_RSZ2_RSZ1_RSZ1_RSZ2_RSZ1_RSZ1
Status: Status.UNKNOWN
Output dim: 7, lower bound: -0.0009327, upper bound: 0.0009458
time: 0.66 seconds

### Relational analysis RSZ of RS_RSZ2_RSZ2_RSZ1_RSZ1_RSZ2_RSZ1_RSZ2

#### Relational analysis RSZ result of RS_RSZ2_RSZ2_RSZ1_RSZ1_RSZ2_RSZ1_RSZ2
Status: Status.UNKNOWN
Output dim: 7, lower bound: -0.0009291, upper bound: 0.0009498
time: 0.68 seconds

## BFS RS instance: RS_RSZ2_RSZ2_RSZ1_RSZ1_RSZ2_RSZ2

### Backsubstitution after applying RS history:
0: -0.0027939, -0.0016242, -0.0027939, -0.0016242, -0.0007434, 0.0007596
1: -0.0114005, -0.0084323, -0.0114005, -0.0084323, -0.0018864, 0.0019276
2: 0.0279571, 0.0297986, 0.0279571, 0.0297986, -0.0011703, 0.0011959
3: 0.0039210, 0.0073596, 0.0039210, 0.0073596, -0.0022330, 0.0021853
4: -0.0104893, -0.0074701, -0.0104893, -0.0074701, -0.0019188, 0.0019607
5: 0.0097651, 0.0109087, 0.0097651, 0.0109087, -0.0007268, 0.0007427
6: 0.0053643, 0.0097284, 0.0053643, 0.0097284, -0.0028340, 0.0027734
7: 0.9818129, 0.9848667, 0.9818129, 0.9848667, -0.0019831, 0.0019407
8: -0.0060636, -0.0027895, -0.0060636, -0.0027895, -0.0021262, 0.0020807
9: -0.0031570, -0.0009942, -0.0031570, -0.0009942, -0.0013744, 0.0014045

### Unstable ReLU Count (Linear/Conv2D Layers)
- layer_idx=0, type=LayerType.Linear, total=256, inp1_unstable=10, inp2_unstable=10, delta_unstable=250
- layer_idx=2, type=LayerType.Linear, total=256, inp1_unstable=10, inp2_unstable=10, delta_unstable=256
- layer_idx=4, type=LayerType.Linear, total=256, inp1_unstable=0, inp2_unstable=0, delta_unstable=256
- layer_idx=6, type=LayerType.Linear, total=256, inp1_unstable=0, inp2_unstable=0, delta_unstable=256
- layer_idx=8, type=LayerType.Linear, total=10, inp1_unstable=0, inp2_unstable=0, delta_unstable=10

Time for backsubstitution: 1.60 seconds

### RS candidates at layer 1
type: RSZ, layer: 1, pos: 140
type: RSZ, layer: 1, pos: 167
type: RSZ, layer: 1, pos: 213
type: RSZ, layer: 1, pos: 219

Time for candidate selection: 0.17 seconds

### Candidate
type: RSZ, layer: 1, pos: 140

### Relational analysis RSZ of RS_RSZ2_RSZ2_RSZ1_RSZ1_RSZ2_RSZ2_RSZ1
Optimization infeasible because this subproblem isn't reachable.

### Candidate
type: RSZ, layer: 1, pos: 167

### Relational analysis RSZ of RS_RSZ2_RSZ2_RSZ1_RSZ1_RSZ2_RSZ2_RSZ1

#### Relational analysis RSZ result of RS_RSZ2_RSZ2_RSZ1_RSZ1_RSZ2_RSZ2_RSZ1
Status: Status.UNKNOWN
Output dim: 7, lower bound: -0.0009326, upper bound: 0.0009461
time: 0.69 seconds

### Relational analysis RSZ of RS_RSZ2_RSZ2_RSZ1_RSZ1_RSZ2_RSZ2_RSZ2

#### Relational analysis RSZ result of RS_RSZ2_RSZ2_RSZ1_RSZ1_RSZ2_RSZ2_RSZ2
Status: Status.UNKNOWN
Output dim: 7, lower bound: -0.0009292, upper bound: 0.0009498
time: 0.72 seconds

## BFS RS instance: RS_RSZ2_RSZ2_RSZ1_RSZ2_RSZ1_RSZ1

### Backsubstitution after applying RS history:
0: -0.0027939, -0.0016242, -0.0027939, -0.0016242, -0.0007655, 0.0007375
1: -0.0114005, -0.0084323, -0.0114005, -0.0084323, -0.0019426, 0.0018714
2: 0.0279571, 0.0297986, 0.0279571, 0.0297986, -0.0012052, 0.0011610
3: 0.0039210, 0.0073596, 0.0039210, 0.0073596, -0.0021679, 0.0022504
4: -0.0104893, -0.0074701, -0.0104893, -0.0074701, -0.0019759, 0.0019035
5: 0.0097651, 0.0109087, 0.0097651, 0.0109087, -0.0007484, 0.0007210
6: 0.0053643, 0.0097284, 0.0053643, 0.0097284, -0.0027514, 0.0028561
7: 0.9818129, 0.9848667, 0.9818129, 0.9848667, -0.0019253, 0.0019985
8: -0.0060636, -0.0027895, -0.0060636, -0.0027895, -0.0020642, 0.0021427
9: -0.0031570, -0.0009942, -0.0031570, -0.0009942, -0.0014154, 0.0013635

### Unstable ReLU Count (Linear/Conv2D Layers)
- layer_idx=0, type=LayerType.Linear, total=256, inp1_unstable=10, inp2_unstable=10, delta_unstable=250
- layer_idx=2, type=LayerType.Linear, total=256, inp1_unstable=10, inp2_unstable=10, delta_unstable=256
- layer_idx=4, type=LayerType.Linear, total=256, inp1_unstable=0, inp2_unstable=0, delta_unstable=256
- layer_idx=6, type=LayerType.Linear, total=256, inp1_unstable=0, inp2_unstable=0, delta_unstable=256
- layer_idx=8, type=LayerType.Linear, total=10, inp1_unstable=0, inp2_unstable=0, delta_unstable=10

Time for backsubstitution: 1.72 seconds

### RS candidates at layer 1
type: RSZ, layer: 1, pos: 140
type: RSZ, layer: 1, pos: 167
type: RSZ, layer: 1, pos: 213
type: RSZ, layer: 1, pos: 219

Time for candidate selection: 0.18 seconds

### Candidate
type: RSZ, layer: 1, pos: 140

### Relational analysis RSZ of RS_RSZ2_RSZ2_RSZ1_RSZ2_RSZ1_RSZ1_RSZ1
Optimization infeasible because this subproblem isn't reachable.

### Candidate
type: RSZ, layer: 1, pos: 167

### Relational analysis RSZ of RS_RSZ2_RSZ2_RSZ1_RSZ2_RSZ1_RSZ1_RSZ1

#### Relational analysis RSZ result of RS_RSZ2_RSZ2_RSZ1_RSZ2_RSZ1_RSZ1_RSZ1
Status: Status.UNKNOWN
Output dim: 7, lower bound: -0.0009297, upper bound: 0.0009546
time: 0.68 seconds

### Relational analysis RSZ of RS_RSZ2_RSZ2_RSZ1_RSZ2_RSZ1_RSZ1_RSZ2

#### Relational analysis RSZ result of RS_RSZ2_RSZ2_RSZ1_RSZ2_RSZ1_RSZ1_RSZ2
Status: Status.UNKNOWN
Output dim: 7, lower bound: -0.0009248, upper bound: 0.0009601
time: 0.68 seconds

## BFS RS instance: RS_RSZ2_RSZ2_RSZ1_RSZ2_RSZ1_RSZ2

### Backsubstitution after applying RS history:
0: -0.0027939, -0.0016242, -0.0027939, -0.0016242, -0.0007656, 0.0007373
1: -0.0114005, -0.0084323, -0.0114005, -0.0084323, -0.0019427, 0.0018710
2: 0.0279571, 0.0297986, 0.0279571, 0.0297986, -0.0012053, 0.0011608
3: 0.0039210, 0.0073596, 0.0039210, 0.0073596, -0.0021674, 0.0022506
4: -0.0104893, -0.0074701, -0.0104893, -0.0074701, -0.0019761, 0.0019031
5: 0.0097651, 0.0109087, 0.0097651, 0.0109087, -0.0007485, 0.0007208
6: 0.0053643, 0.0097284, 0.0053643, 0.0097284, -0.0027508, 0.0028563
7: 0.9818129, 0.9848667, 0.9818129, 0.9848667, -0.0019249, 0.0019987
8: -0.0060636, -0.0027895, -0.0060636, -0.0027895, -0.0020637, 0.0021429
9: -0.0031570, -0.0009942, -0.0031570, -0.0009942, -0.0014155, 0.0013632

### Unstable ReLU Count (Linear/Conv2D Layers)
- layer_idx=0, type=LayerType.Linear, total=256, inp1_unstable=10, inp2_unstable=10, delta_unstable=250
- layer_idx=2, type=LayerType.Linear, total=256, inp1_unstable=10, inp2_unstable=10, delta_unstable=256
- layer_idx=4, type=LayerType.Linear, total=256, inp1_unstable=0, inp2_unstable=0, delta_unstable=256
- layer_idx=6, type=LayerType.Linear, total=256, inp1_unstable=0, inp2_unstable=0, delta_unstable=256
- layer_idx=8, type=LayerType.Linear, total=10, inp1_unstable=0, inp2_unstable=0, delta_unstable=10

Time for backsubstitution: 1.67 seconds

### RS candidates at layer 1
type: RSZ, layer: 1, pos: 140
type: RSZ, layer: 1, pos: 167
type: RSZ, layer: 1, pos: 213
type: RSZ, layer: 1, pos: 219

Time for candidate selection: 0.19 seconds

### Candidate
type: RSZ, layer: 1, pos: 140

### Relational analysis RSZ of RS_RSZ2_RSZ2_RSZ1_RSZ2_RSZ1_RSZ2_RSZ1
Optimization infeasible because this subproblem isn't reachable.

### Candidate
type: RSZ, layer: 1, pos: 167

### Relational analysis RSZ of RS_RSZ2_RSZ2_RSZ1_RSZ2_RSZ1_RSZ2_RSZ1

#### Relational analysis RSZ result of RS_RSZ2_RSZ2_RSZ1_RSZ2_RSZ1_RSZ2_RSZ1
Status: Status.UNKNOWN
Output dim: 7, lower bound: -0.0009294, upper bound: 0.0009552
time: 0.66 seconds

### Relational analysis RSZ of RS_RSZ2_RSZ2_RSZ1_RSZ2_RSZ1_RSZ2_RSZ2

#### Relational analysis RSZ result of RS_RSZ2_RSZ2_RSZ1_RSZ2_RSZ1_RSZ2_RSZ2
Status: Status.UNKNOWN
Output dim: 7, lower bound: -0.0009247, upper bound: 0.0009605
time: 0.71 seconds

## BFS RS instance: RS_RSZ2_RSZ2_RSZ1_RSZ2_RSZ2_RSZ1

### Backsubstitution after applying RS history:
0: -0.0027939, -0.0016242, -0.0027939, -0.0016242, -0.0007501, 0.0007526
1: -0.0114005, -0.0084323, -0.0114005, -0.0084323, -0.0019034, 0.0019099
2: 0.0279571, 0.0297986, 0.0279571, 0.0297986, -0.0011809, 0.0011849
3: 0.0039210, 0.0073596, 0.0039210, 0.0073596, -0.0022125, 0.0022050
4: -0.0104893, -0.0074701, -0.0104893, -0.0074701, -0.0019361, 0.0019427
5: 0.0097651, 0.0109087, 0.0097651, 0.0109087, -0.0007333, 0.0007358
6: 0.0053643, 0.0097284, 0.0053643, 0.0097284, -0.0028080, 0.0027984
7: 0.9818129, 0.9848667, 0.9818129, 0.9848667, -0.0019649, 0.0019582
8: -0.0060636, -0.0027895, -0.0060636, -0.0027895, -0.0021067, 0.0020995
9: -0.0031570, -0.0009942, -0.0031570, -0.0009942, -0.0013868, 0.0013916

### Unstable ReLU Count (Linear/Conv2D Layers)
- layer_idx=0, type=LayerType.Linear, total=256, inp1_unstable=10, inp2_unstable=10, delta_unstable=250
- layer_idx=2, type=LayerType.Linear, total=256, inp1_unstable=10, inp2_unstable=10, delta_unstable=256
- layer_idx=4, type=LayerType.Linear, total=256, inp1_unstable=0, inp2_unstable=0, delta_unstable=256
- layer_idx=6, type=LayerType.Linear, total=256, inp1_unstable=0, inp2_unstable=0, delta_unstable=256
- layer_idx=8, type=LayerType.Linear, total=10, inp1_unstable=0, inp2_unstable=0, delta_unstable=10

Time for backsubstitution: 1.63 seconds

### RS candidates at layer 1
type: RSZ, layer: 1, pos: 140
type: RSZ, layer: 1, pos: 167
type: RSZ, layer: 1, pos: 213
type: RSZ, layer: 1, pos: 219

Time for candidate selection: 0.17 seconds

### Candidate
type: RSZ, layer: 1, pos: 140

### Relational analysis RSZ of RS_RSZ2_RSZ2_RSZ1_RSZ2_RSZ2_RSZ1_RSZ1
Optimization infeasible because this subproblem isn't reachable.

### Candidate
type: RSZ, layer: 1, pos: 167

### Relational analysis RSZ of RS_RSZ2_RSZ2_RSZ1_RSZ2_RSZ2_RSZ1_RSZ1

#### Relational analysis RSZ result of RS_RSZ2_RSZ2_RSZ1_RSZ2_RSZ2_RSZ1_RSZ1
Status: Status.UNKNOWN
Output dim: 7, lower bound: -0.0009362, upper bound: 0.0009452
time: 0.65 seconds

### Relational analysis RSZ of RS_RSZ2_RSZ2_RSZ1_RSZ2_RSZ2_RSZ1_RSZ2

#### Relational analysis RSZ result of RS_RSZ2_RSZ2_RSZ1_RSZ2_RSZ2_RSZ1_RSZ2
Status: Status.UNKNOWN
Output dim: 7, lower bound: -0.0009330, upper bound: 0.0009496
time: 0.65 seconds

## BFS RS instance: RS_RSZ2_RSZ2_RSZ1_RSZ2_RSZ2_RSZ2

### Backsubstitution after applying RS history:
0: -0.0027939, -0.0016242, -0.0027939, -0.0016242, -0.0007503, 0.0007525
1: -0.0114005, -0.0084323, -0.0114005, -0.0084323, -0.0019040, 0.0019095
2: 0.0279571, 0.0297986, 0.0279571, 0.0297986, -0.0011812, 0.0011847
3: 0.0039210, 0.0073596, 0.0039210, 0.0073596, -0.0022121, 0.0022056
4: -0.0104893, -0.0074701, -0.0104893, -0.0074701, -0.0019366, 0.0019423
5: 0.0097651, 0.0109087, 0.0097651, 0.0109087, -0.0007335, 0.0007357
6: 0.0053643, 0.0097284, 0.0053643, 0.0097284, -0.0028075, 0.0027992
7: 0.9818129, 0.9848667, 0.9818129, 0.9848667, -0.0019645, 0.0019588
8: -0.0060636, -0.0027895, -0.0060636, -0.0027895, -0.0021063, 0.0021001
9: -0.0031570, -0.0009942, -0.0031570, -0.0009942, -0.0013872, 0.0013913

### Unstable ReLU Count (Linear/Conv2D Layers)
- layer_idx=0, type=LayerType.Linear, total=256, inp1_unstable=10, inp2_unstable=10, delta_unstable=250
- layer_idx=2, type=LayerType.Linear, total=256, inp1_unstable=10, inp2_unstable=10, delta_unstable=256
- layer_idx=4, type=LayerType.Linear, total=256, inp1_unstable=0, inp2_unstable=0, delta_unstable=256
- layer_idx=6, type=LayerType.Linear, total=256, inp1_unstable=0, inp2_unstable=0, delta_unstable=256
- layer_idx=8, type=LayerType.Linear, total=10, inp1_unstable=0, inp2_unstable=0, delta_unstable=10

Time for backsubstitution: 1.64 seconds

### RS candidates at layer 1
type: RSZ, layer: 1, pos: 140
type: RSZ, layer: 1, pos: 167
type: RSZ, layer: 1, pos: 213
type: RSZ, layer: 1, pos: 219

Time for candidate selection: 0.17 seconds

### Candidate
type: RSZ, layer: 1, pos: 140

### Relational analysis RSZ of RS_RSZ2_RSZ2_RSZ1_RSZ2_RSZ2_RSZ2_RSZ1
Optimization infeasible because this subproblem isn't reachable.

### Candidate
type: RSZ, layer: 1, pos: 167

### Relational analysis RSZ of RS_RSZ2_RSZ2_RSZ1_RSZ2_RSZ2_RSZ2_RSZ1

#### Relational analysis RSZ result of RS_RSZ2_RSZ2_RSZ1_RSZ2_RSZ2_RSZ2_RSZ1
Status: Status.UNKNOWN
Output dim: 7, lower bound: -0.0009365, upper bound: 0.0009454
time: 0.65 seconds

### Relational analysis RSZ of RS_RSZ2_RSZ2_RSZ1_RSZ2_RSZ2_RSZ2_RSZ2

#### Relational analysis RSZ result of RS_RSZ2_RSZ2_RSZ1_RSZ2_RSZ2_RSZ2_RSZ2
Status: Status.UNKNOWN
Output dim: 7, lower bound: -0.0009333, upper bound: 0.0009496
time: 0.66 seconds

## BFS RS instance: RS_RSZ2_RSZ2_RSZ2_RSZ1_RSZ1_RSZ1

### Backsubstitution after applying RS history:
0: -0.0027939, -0.0016242, -0.0027939, -0.0016242, -0.0007593, 0.0007431
1: -0.0114005, -0.0084323, -0.0114005, -0.0084323, -0.0019268, 0.0018858
2: 0.0279571, 0.0297986, 0.0279571, 0.0297986, -0.0011954, 0.0011699
3: 0.0039210, 0.0073596, 0.0039210, 0.0073596, -0.0021846, 0.0022321
4: -0.0104893, -0.0074701, -0.0104893, -0.0074701, -0.0019599, 0.0019181
5: 0.0097651, 0.0109087, 0.0097651, 0.0109087, -0.0007424, 0.0007265
6: 0.0053643, 0.0097284, 0.0053643, 0.0097284, -0.0027725, 0.0028329
7: 0.9818129, 0.9848667, 0.9818129, 0.9848667, -0.0019401, 0.0019823
8: -0.0060636, -0.0027895, -0.0060636, -0.0027895, -0.0020800, 0.0021253
9: -0.0031570, -0.0009942, -0.0031570, -0.0009942, -0.0014039, 0.0013740

### Unstable ReLU Count (Linear/Conv2D Layers)
- layer_idx=0, type=LayerType.Linear, total=256, inp1_unstable=10, inp2_unstable=10, delta_unstable=250
- layer_idx=2, type=LayerType.Linear, total=256, inp1_unstable=10, inp2_unstable=10, delta_unstable=256
- layer_idx=4, type=LayerType.Linear, total=256, inp1_unstable=0, inp2_unstable=0, delta_unstable=256
- layer_idx=6, type=LayerType.Linear, total=256, inp1_unstable=0, inp2_unstable=0, delta_unstable=256
- layer_idx=8, type=LayerType.Linear, total=10, inp1_unstable=0, inp2_unstable=0, delta_unstable=10

Time for backsubstitution: 1.62 seconds

### RS candidates at layer 1
type: RSZ, layer: 1, pos: 140
type: RSZ, layer: 1, pos: 167
type: RSZ, layer: 1, pos: 213
type: RSZ, layer: 1, pos: 219

Time for candidate selection: 0.17 seconds

### Candidate
type: RSZ, layer: 1, pos: 140

### Relational analysis RSZ of RS_RSZ2_RSZ2_RSZ2_RSZ1_RSZ1_RSZ1_RSZ1
Optimization infeasible because this subproblem isn't reachable.

### Candidate
type: RSZ, layer: 1, pos: 167

### Relational analysis RSZ of RS_RSZ2_RSZ2_RSZ2_RSZ1_RSZ1_RSZ1_RSZ1

#### Relational analysis RSZ result of RS_RSZ2_RSZ2_RSZ2_RSZ1_RSZ1_RSZ1_RSZ1
Status: Status.UNKNOWN
Output dim: 7, lower bound: -0.0009244, upper bound: 0.0009559
time: 0.68 seconds

### Relational analysis RSZ of RS_RSZ2_RSZ2_RSZ2_RSZ1_RSZ1_RSZ1_RSZ2

#### Relational analysis RSZ result of RS_RSZ2_RSZ2_RSZ2_RSZ1_RSZ1_RSZ1_RSZ2
Status: Status.UNKNOWN
Output dim: 7, lower bound: -0.0009189, upper bound: 0.0009607
time: 0.66 seconds

## BFS RS instance: RS_RSZ2_RSZ2_RSZ2_RSZ1_RSZ1_RSZ2

### Backsubstitution after applying RS history:
0: -0.0027939, -0.0016242, -0.0027939, -0.0016242, -0.0007595, 0.0007430
1: -0.0114005, -0.0084323, -0.0114005, -0.0084323, -0.0019272, 0.0018854
2: 0.0279571, 0.0297986, 0.0279571, 0.0297986, -0.0011957, 0.0011697
3: 0.0039210, 0.0073596, 0.0039210, 0.0073596, -0.0021841, 0.0022326
4: -0.0104893, -0.0074701, -0.0104893, -0.0074701, -0.0019603, 0.0019177
5: 0.0097651, 0.0109087, 0.0097651, 0.0109087, -0.0007425, 0.0007264
6: 0.0053643, 0.0097284, 0.0053643, 0.0097284, -0.0027719, 0.0028334
7: 0.9818129, 0.9848667, 0.9818129, 0.9848667, -0.0019396, 0.0019827
8: -0.0060636, -0.0027895, -0.0060636, -0.0027895, -0.0020796, 0.0021258
9: -0.0031570, -0.0009942, -0.0031570, -0.0009942, -0.0014042, 0.0013737

### Unstable ReLU Count (Linear/Conv2D Layers)
- layer_idx=0, type=LayerType.Linear, total=256, inp1_unstable=10, inp2_unstable=10, delta_unstable=250
- layer_idx=2, type=LayerType.Linear, total=256, inp1_unstable=10, inp2_unstable=10, delta_unstable=256
- layer_idx=4, type=LayerType.Linear, total=256, inp1_unstable=0, inp2_unstable=0, delta_unstable=256
- layer_idx=6, type=LayerType.Linear, total=256, inp1_unstable=0, inp2_unstable=0, delta_unstable=256
- layer_idx=8, type=LayerType.Linear, total=10, inp1_unstable=0, inp2_unstable=0, delta_unstable=10

Time for backsubstitution: 1.61 seconds

### RS candidates at layer 1
type: RSZ, layer: 1, pos: 140
type: RSZ, layer: 1, pos: 167
type: RSZ, layer: 1, pos: 213
type: RSZ, layer: 1, pos: 219

Time for candidate selection: 0.16 seconds

### Candidate
type: RSZ, layer: 1, pos: 140

### Relational analysis RSZ of RS_RSZ2_RSZ2_RSZ2_RSZ1_RSZ1_RSZ2_RSZ1
Optimization infeasible because this subproblem isn't reachable.

### Candidate
type: RSZ, layer: 1, pos: 167

### Relational analysis RSZ of RS_RSZ2_RSZ2_RSZ2_RSZ1_RSZ1_RSZ2_RSZ1

#### Relational analysis RSZ result of RS_RSZ2_RSZ2_RSZ2_RSZ1_RSZ1_RSZ2_RSZ1
Status: Status.UNKNOWN
Output dim: 7, lower bound: -0.0009243, upper bound: 0.0009566
time: 0.65 seconds

### Relational analysis RSZ of RS_RSZ2_RSZ2_RSZ2_RSZ1_RSZ1_RSZ2_RSZ2

#### Relational analysis RSZ result of RS_RSZ2_RSZ2_RSZ2_RSZ1_RSZ1_RSZ2_RSZ2
Status: Status.UNKNOWN
Output dim: 7, lower bound: -0.0009189, upper bound: 0.0009614
time: 0.66 seconds

## BFS RS instance: RS_RSZ2_RSZ2_RSZ2_RSZ1_RSZ2_RSZ1

### Backsubstitution after applying RS history:
0: -0.0027939, -0.0016242, -0.0027939, -0.0016242, -0.0007448, 0.0007576
1: -0.0114005, -0.0084323, -0.0114005, -0.0084323, -0.0018901, 0.0019224
2: 0.0279571, 0.0297986, 0.0279571, 0.0297986, -0.0011726, 0.0011927
3: 0.0039210, 0.0073596, 0.0039210, 0.0073596, -0.0022270, 0.0021896
4: -0.0104893, -0.0074701, -0.0104893, -0.0074701, -0.0019226, 0.0019554
5: 0.0097651, 0.0109087, 0.0097651, 0.0109087, -0.0007282, 0.0007407
6: 0.0053643, 0.0097284, 0.0053643, 0.0097284, -0.0028264, 0.0027789
7: 0.9818129, 0.9848667, 0.9818129, 0.9848667, -0.0019777, 0.0019445
8: -0.0060636, -0.0027895, -0.0060636, -0.0027895, -0.0021205, 0.0020849
9: -0.0031570, -0.0009942, -0.0031570, -0.0009942, -0.0013772, 0.0014007

### Unstable ReLU Count (Linear/Conv2D Layers)
- layer_idx=0, type=LayerType.Linear, total=256, inp1_unstable=10, inp2_unstable=10, delta_unstable=250
- layer_idx=2, type=LayerType.Linear, total=256, inp1_unstable=10, inp2_unstable=10, delta_unstable=256
- layer_idx=4, type=LayerType.Linear, total=256, inp1_unstable=0, inp2_unstable=0, delta_unstable=256
- layer_idx=6, type=LayerType.Linear, total=256, inp1_unstable=0, inp2_unstable=0, delta_unstable=256
- layer_idx=8, type=LayerType.Linear, total=10, inp1_unstable=0, inp2_unstable=0, delta_unstable=10

Time for backsubstitution: 1.62 seconds

### RS candidates at layer 1
type: RSZ, layer: 1, pos: 140
type: RSZ, layer: 1, pos: 167
type: RSZ, layer: 1, pos: 213
type: RSZ, layer: 1, pos: 219

Time for candidate selection: 0.17 seconds

### Candidate
type: RSZ, layer: 1, pos: 140

### Relational analysis RSZ of RS_RSZ2_RSZ2_RSZ2_RSZ1_RSZ2_RSZ1_RSZ1
Optimization infeasible because this subproblem isn't reachable.

### Candidate
type: RSZ, layer: 1, pos: 167

### Relational analysis RSZ of RS_RSZ2_RSZ2_RSZ2_RSZ1_RSZ2_RSZ1_RSZ1

#### Relational analysis RSZ result of RS_RSZ2_RSZ2_RSZ2_RSZ1_RSZ2_RSZ1_RSZ1
Status: Status.UNKNOWN
Output dim: 7, lower bound: -0.0009327, upper bound: 0.0009458
time: 0.67 seconds

### Relational analysis RSZ of RS_RSZ2_RSZ2_RSZ2_RSZ1_RSZ2_RSZ1_RSZ2

#### Relational analysis RSZ result of RS_RSZ2_RSZ2_RSZ2_RSZ1_RSZ2_RSZ1_RSZ2
Status: Status.UNKNOWN
Output dim: 7, lower bound: -0.0009291, upper bound: 0.0009498
time: 0.69 seconds

## BFS RS instance: RS_RSZ2_RSZ2_RSZ2_RSZ1_RSZ2_RSZ2

### Backsubstitution after applying RS history:
0: -0.0027939, -0.0016242, -0.0027939, -0.0016242, -0.0007450, 0.0007575
1: -0.0114005, -0.0084323, -0.0114005, -0.0084323, -0.0018905, 0.0019222
2: 0.0279571, 0.0297986, 0.0279571, 0.0297986, -0.0011729, 0.0011925
3: 0.0039210, 0.0073596, 0.0039210, 0.0073596, -0.0022267, 0.0021901
4: -0.0104893, -0.0074701, -0.0104893, -0.0074701, -0.0019230, 0.0019552
5: 0.0097651, 0.0109087, 0.0097651, 0.0109087, -0.0007284, 0.0007406
6: 0.0053643, 0.0097284, 0.0053643, 0.0097284, -0.0028260, 0.0027795
7: 0.9818129, 0.9848667, 0.9818129, 0.9848667, -0.0019775, 0.0019450
8: -0.0060636, -0.0027895, -0.0060636, -0.0027895, -0.0021202, 0.0020853
9: -0.0031570, -0.0009942, -0.0031570, -0.0009942, -0.0013775, 0.0014005

### Unstable ReLU Count (Linear/Conv2D Layers)
- layer_idx=0, type=LayerType.Linear, total=256, inp1_unstable=10, inp2_unstable=10, delta_unstable=250
- layer_idx=2, type=LayerType.Linear, total=256, inp1_unstable=10, inp2_unstable=10, delta_unstable=256
- layer_idx=4, type=LayerType.Linear, total=256, inp1_unstable=0, inp2_unstable=0, delta_unstable=256
- layer_idx=6, type=LayerType.Linear, total=256, inp1_unstable=0, inp2_unstable=0, delta_unstable=256
- layer_idx=8, type=LayerType.Linear, total=10, inp1_unstable=0, inp2_unstable=0, delta_unstable=10

Time for backsubstitution: 1.61 seconds

### RS candidates at layer 1
type: RSZ, layer: 1, pos: 140
type: RSZ, layer: 1, pos: 167
type: RSZ, layer: 1, pos: 213
type: RSZ, layer: 1, pos: 219

Time for candidate selection: 0.16 seconds

### Candidate
type: RSZ, layer: 1, pos: 140

### Relational analysis RSZ of RS_RSZ2_RSZ2_RSZ2_RSZ1_RSZ2_RSZ2_RSZ1
Optimization infeasible because this subproblem isn't reachable.

### Candidate
type: RSZ, layer: 1, pos: 167

### Relational analysis RSZ of RS_RSZ2_RSZ2_RSZ2_RSZ1_RSZ2_RSZ2_RSZ1

#### Relational analysis RSZ result of RS_RSZ2_RSZ2_RSZ2_RSZ1_RSZ2_RSZ2_RSZ1
Status: Status.UNKNOWN
Output dim: 7, lower bound: -0.0009326, upper bound: 0.0009461
time: 0.66 seconds

### Relational analysis RSZ of RS_RSZ2_RSZ2_RSZ2_RSZ1_RSZ2_RSZ2_RSZ2

#### Relational analysis RSZ result of RS_RSZ2_RSZ2_RSZ2_RSZ1_RSZ2_RSZ2_RSZ2
Status: Status.UNKNOWN
Output dim: 7, lower bound: -0.0009292, upper bound: 0.0009498
time: 0.70 seconds

## BFS RS instance: RS_RSZ2_RSZ2_RSZ2_RSZ2_RSZ1_RSZ1

### Backsubstitution after applying RS history:
0: -0.0027939, -0.0016242, -0.0027939, -0.0016242, -0.0007672, 0.0007356
1: -0.0114005, -0.0084323, -0.0114005, -0.0084323, -0.0019468, 0.0018666
2: 0.0279571, 0.0297986, 0.0279571, 0.0297986, -0.0012078, 0.0011581
3: 0.0039210, 0.0073596, 0.0039210, 0.0073596, -0.0021624, 0.0022552
4: -0.0104893, -0.0074701, -0.0104893, -0.0074701, -0.0019802, 0.0018987
5: 0.0097651, 0.0109087, 0.0097651, 0.0109087, -0.0007500, 0.0007192
6: 0.0053643, 0.0097284, 0.0053643, 0.0097284, -0.0027444, 0.0028622
7: 0.9818129, 0.9848667, 0.9818129, 0.9848667, -0.0019204, 0.0020028
8: -0.0060636, -0.0027895, -0.0060636, -0.0027895, -0.0020589, 0.0021473
9: -0.0031570, -0.0009942, -0.0031570, -0.0009942, -0.0014184, 0.0013600

### Unstable ReLU Count (Linear/Conv2D Layers)
- layer_idx=0, type=LayerType.Linear, total=256, inp1_unstable=10, inp2_unstable=10, delta_unstable=250
- layer_idx=2, type=LayerType.Linear, total=256, inp1_unstable=10, inp2_unstable=10, delta_unstable=256
- layer_idx=4, type=LayerType.Linear, total=256, inp1_unstable=0, inp2_unstable=0, delta_unstable=256
- layer_idx=6, type=LayerType.Linear, total=256, inp1_unstable=0, inp2_unstable=0, delta_unstable=256
- layer_idx=8, type=LayerType.Linear, total=10, inp1_unstable=0, inp2_unstable=0, delta_unstable=10

Time for backsubstitution: 1.72 seconds

### RS candidates at layer 1
type: RSZ, layer: 1, pos: 140
type: RSZ, layer: 1, pos: 167
type: RSZ, layer: 1, pos: 213
type: RSZ, layer: 1, pos: 219

Time for candidate selection: 0.18 seconds

### Candidate
type: RSZ, layer: 1, pos: 140

### Relational analysis RSZ of RS_RSZ2_RSZ2_RSZ2_RSZ2_RSZ1_RSZ1_RSZ1
Optimization infeasible because this subproblem isn't reachable.

### Candidate
type: RSZ, layer: 1, pos: 167

### Relational analysis RSZ of RS_RSZ2_RSZ2_RSZ2_RSZ2_RSZ1_RSZ1_RSZ1

#### Relational analysis RSZ result of RS_RSZ2_RSZ2_RSZ2_RSZ2_RSZ1_RSZ1_RSZ1
Status: Status.UNKNOWN
Output dim: 7, lower bound: -0.0009297, upper bound: 0.0009546
time: 0.65 seconds

### Relational analysis RSZ of RS_RSZ2_RSZ2_RSZ2_RSZ2_RSZ1_RSZ1_RSZ2

#### Relational analysis RSZ result of RS_RSZ2_RSZ2_RSZ2_RSZ2_RSZ1_RSZ1_RSZ2
Status: Status.UNKNOWN
Output dim: 7, lower bound: -0.0009248, upper bound: 0.0009601
time: 0.65 seconds

## BFS RS instance: RS_RSZ2_RSZ2_RSZ2_RSZ2_RSZ1_RSZ2

### Backsubstitution after applying RS history:
0: -0.0027939, -0.0016242, -0.0027939, -0.0016242, -0.0007672, 0.0007354
1: -0.0114005, -0.0084323, -0.0114005, -0.0084323, -0.0019469, 0.0018662
2: 0.0279571, 0.0297986, 0.0279571, 0.0297986, -0.0012079, 0.0011578
3: 0.0039210, 0.0073596, 0.0039210, 0.0073596, -0.0021619, 0.0022554
4: -0.0104893, -0.0074701, -0.0104893, -0.0074701, -0.0019803, 0.0018983
5: 0.0097651, 0.0109087, 0.0097651, 0.0109087, -0.0007501, 0.0007190
6: 0.0053643, 0.0097284, 0.0053643, 0.0097284, -0.0027438, 0.0028624
7: 0.9818129, 0.9848667, 0.9818129, 0.9848667, -0.0019200, 0.0020030
8: -0.0060636, -0.0027895, -0.0060636, -0.0027895, -0.0020585, 0.0021475
9: -0.0031570, -0.0009942, -0.0031570, -0.0009942, -0.0014185, 0.0013598

### Unstable ReLU Count (Linear/Conv2D Layers)
- layer_idx=0, type=LayerType.Linear, total=256, inp1_unstable=10, inp2_unstable=10, delta_unstable=250
- layer_idx=2, type=LayerType.Linear, total=256, inp1_unstable=10, inp2_unstable=10, delta_unstable=256
- layer_idx=4, type=LayerType.Linear, total=256, inp1_unstable=0, inp2_unstable=0, delta_unstable=256
- layer_idx=6, type=LayerType.Linear, total=256, inp1_unstable=0, inp2_unstable=0, delta_unstable=256
- layer_idx=8, type=LayerType.Linear, total=10, inp1_unstable=0, inp2_unstable=0, delta_unstable=10

Time for backsubstitution: 1.62 seconds

### RS candidates at layer 1
type: RSZ, layer: 1, pos: 140
type: RSZ, layer: 1, pos: 167
type: RSZ, layer: 1, pos: 213
type: RSZ, layer: 1, pos: 219

Time for candidate selection: 0.18 seconds

### Candidate
type: RSZ, layer: 1, pos: 140

### Relational analysis RSZ of RS_RSZ2_RSZ2_RSZ2_RSZ2_RSZ1_RSZ2_RSZ1
Optimization infeasible because this subproblem isn't reachable.

### Candidate
type: RSZ, layer: 1, pos: 167

### Relational analysis RSZ of RS_RSZ2_RSZ2_RSZ2_RSZ2_RSZ1_RSZ2_RSZ1

#### Relational analysis RSZ result of RS_RSZ2_RSZ2_RSZ2_RSZ2_RSZ1_RSZ2_RSZ1
Status: Status.UNKNOWN
Output dim: 7, lower bound: -0.0009294, upper bound: 0.0009552
time: 0.64 seconds

### Relational analysis RSZ of RS_RSZ2_RSZ2_RSZ2_RSZ2_RSZ1_RSZ2_RSZ2

#### Relational analysis RSZ result of RS_RSZ2_RSZ2_RSZ2_RSZ2_RSZ1_RSZ2_RSZ2
Status: Status.UNKNOWN
Output dim: 7, lower bound: -0.0009247, upper bound: 0.0009605
time: 0.66 seconds

## BFS RS instance: RS_RSZ2_RSZ2_RSZ2_RSZ2_RSZ2_RSZ1

### Backsubstitution after applying RS history:
0: -0.0027939, -0.0016242, -0.0027939, -0.0016242, -0.0007517, 0.0007504
1: -0.0114005, -0.0084323, -0.0114005, -0.0084323, -0.0019075, 0.0019043
2: 0.0279571, 0.0297986, 0.0279571, 0.0297986, -0.0011834, 0.0011815
3: 0.0039210, 0.0073596, 0.0039210, 0.0073596, -0.0022061, 0.0022098
4: -0.0104893, -0.0074701, -0.0104893, -0.0074701, -0.0019403, 0.0019370
5: 0.0097651, 0.0109087, 0.0097651, 0.0109087, -0.0007349, 0.0007337
6: 0.0053643, 0.0097284, 0.0053643, 0.0097284, -0.0027998, 0.0028045
7: 0.9818129, 0.9848667, 0.9818129, 0.9848667, -0.0019592, 0.0019625
8: -0.0060636, -0.0027895, -0.0060636, -0.0027895, -0.0021005, 0.0021041
9: -0.0031570, -0.0009942, -0.0031570, -0.0009942, -0.0013899, 0.0013875

### Unstable ReLU Count (Linear/Conv2D Layers)
- layer_idx=0, type=LayerType.Linear, total=256, inp1_unstable=10, inp2_unstable=10, delta_unstable=250
- layer_idx=2, type=LayerType.Linear, total=256, inp1_unstable=10, inp2_unstable=10, delta_unstable=256
- layer_idx=4, type=LayerType.Linear, total=256, inp1_unstable=0, inp2_unstable=0, delta_unstable=256
- layer_idx=6, type=LayerType.Linear, total=256, inp1_unstable=0, inp2_unstable=0, delta_unstable=256
- layer_idx=8, type=LayerType.Linear, total=10, inp1_unstable=0, inp2_unstable=0, delta_unstable=10

Time for backsubstitution: 1.65 seconds

### RS candidates at layer 1
type: RSZ, layer: 1, pos: 140
type: RSZ, layer: 1, pos: 167
type: RSZ, layer: 1, pos: 213
type: RSZ, layer: 1, pos: 219

Time for candidate selection: 0.20 seconds

### Candidate
type: RSZ, layer: 1, pos: 140

### Relational analysis RSZ of RS_RSZ2_RSZ2_RSZ2_RSZ2_RSZ2_RSZ1_RSZ1
Optimization infeasible because this subproblem isn't reachable.

### Candidate
type: RSZ, layer: 1, pos: 167

### Relational analysis RSZ of RS_RSZ2_RSZ2_RSZ2_RSZ2_RSZ2_RSZ1_RSZ1

#### Relational analysis RSZ result of RS_RSZ2_RSZ2_RSZ2_RSZ2_RSZ2_RSZ1_RSZ1
Status: Status.UNKNOWN
Output dim: 7, lower bound: -0.0009362, upper bound: 0.0009452
time: 0.70 seconds

### Relational analysis RSZ of RS_RSZ2_RSZ2_RSZ2_RSZ2_RSZ2_RSZ1_RSZ2

#### Relational analysis RSZ result of RS_RSZ2_RSZ2_RSZ2_RSZ2_RSZ2_RSZ1_RSZ2
Status: Status.UNKNOWN
Output dim: 7, lower bound: -0.0009330, upper bound: 0.0009496
time: 0.66 seconds

## BFS RS instance: RS_RSZ2_RSZ2_RSZ2_RSZ2_RSZ2_RSZ2

### Backsubstitution after applying RS history:
0: -0.0027939, -0.0016242, -0.0027939, -0.0016242, -0.0007519, 0.0007503
1: -0.0114005, -0.0084323, -0.0114005, -0.0084323, -0.0019081, 0.0019039
2: 0.0279571, 0.0297986, 0.0279571, 0.0297986, -0.0011838, 0.0011812
3: 0.0039210, 0.0073596, 0.0039210, 0.0073596, -0.0022056, 0.0022105
4: -0.0104893, -0.0074701, -0.0104893, -0.0074701, -0.0019409, 0.0019366
5: 0.0097651, 0.0109087, 0.0097651, 0.0109087, -0.0007352, 0.0007335
6: 0.0053643, 0.0097284, 0.0053643, 0.0097284, -0.0027992, 0.0028054
7: 0.9818129, 0.9848667, 0.9818129, 0.9848667, -0.0019588, 0.0019631
8: -0.0060636, -0.0027895, -0.0060636, -0.0027895, -0.0021001, 0.0021047
9: -0.0031570, -0.0009942, -0.0031570, -0.0009942, -0.0013903, 0.0013872

### Unstable ReLU Count (Linear/Conv2D Layers)
- layer_idx=0, type=LayerType.Linear, total=256, inp1_unstable=10, inp2_unstable=10, delta_unstable=250
- layer_idx=2, type=LayerType.Linear, total=256, inp1_unstable=10, inp2_unstable=10, delta_unstable=256
- layer_idx=4, type=LayerType.Linear, total=256, inp1_unstable=0, inp2_unstable=0, delta_unstable=256
- layer_idx=6, type=LayerType.Linear, total=256, inp1_unstable=0, inp2_unstable=0, delta_unstable=256
- layer_idx=8, type=LayerType.Linear, total=10, inp1_unstable=0, inp2_unstable=0, delta_unstable=10

Time for backsubstitution: 1.73 seconds

### RS candidates at layer 1
type: RSZ, layer: 1, pos: 140
type: RSZ, layer: 1, pos: 167
type: RSZ, layer: 1, pos: 213
type: RSZ, layer: 1, pos: 219

Time for candidate selection: 0.23 seconds

### Candidate
type: RSZ, layer: 1, pos: 140

### Relational analysis RSZ of RS_RSZ2_RSZ2_RSZ2_RSZ2_RSZ2_RSZ2_RSZ1
Optimization infeasible because this subproblem isn't reachable.

### Candidate
type: RSZ, layer: 1, pos: 167

### Relational analysis RSZ of RS_RSZ2_RSZ2_RSZ2_RSZ2_RSZ2_RSZ2_RSZ1

#### Relational analysis RSZ result of RS_RSZ2_RSZ2_RSZ2_RSZ2_RSZ2_RSZ2_RSZ1
Status: Status.UNKNOWN
Output dim: 7, lower bound: -0.0009365, upper bound: 0.0009454
time: 0.65 seconds

### Relational analysis RSZ of RS_RSZ2_RSZ2_RSZ2_RSZ2_RSZ2_RSZ2_RSZ2

#### Relational analysis RSZ result of RS_RSZ2_RSZ2_RSZ2_RSZ2_RSZ2_RSZ2_RSZ2
Status: Status.UNKNOWN
Output dim: 7, lower bound: -0.0009333, upper bound: 0.0009496
time: 0.66 seconds

## Summary of splitting (split count: 6)
- Time for RS candidates: 5.24 seconds
RS_RSZ1_RSZ1_RSZ1_RSZ1_RSZ1_RSZ1_RSZ1, status: Status.UNKNOWN, split count: 7, time: 5.24
Output dim: 7, lower bound: -0.0009496, upper bound: 0.0009333
RS_RSZ1_RSZ1_RSZ1_RSZ1_RSZ1_RSZ1_RSZ2, status: Status.UNKNOWN, split count: 7, time: 5.24
Output dim: 7, lower bound: -0.0009454, upper bound: 0.0009365
RS_RSZ1_RSZ1_RSZ1_RSZ1_RSZ1_RSZ2_RSZ1, status: Status.UNKNOWN, split count: 7, time: 5.24
Output dim: 7, lower bound: -0.0009496, upper bound: 0.0009330
RS_RSZ1_RSZ1_RSZ1_RSZ1_RSZ1_RSZ2_RSZ2, status: Status.UNKNOWN, split count: 7, time: 5.24
Output dim: 7, lower bound: -0.0009452, upper bound: 0.0009362
RS_RSZ1_RSZ1_RSZ1_RSZ1_RSZ2_RSZ1_RSZ1, status: Status.UNKNOWN, split count: 7, time: 5.24
Output dim: 7, lower bound: -0.0009605, upper bound: 0.0009247
RS_RSZ1_RSZ1_RSZ1_RSZ1_RSZ2_RSZ1_RSZ2, status: Status.UNKNOWN, split count: 7, time: 5.24
Output dim: 7, lower bound: -0.0009552, upper bound: 0.0009294
RS_RSZ1_RSZ1_RSZ1_RSZ1_RSZ2_RSZ2_RSZ1, status: Status.UNKNOWN, split count: 7, time: 5.24
Output dim: 7, lower bound: -0.0009601, upper bound: 0.0009248
RS_RSZ1_RSZ1_RSZ1_RSZ1_RSZ2_RSZ2_RSZ2, status: Status.UNKNOWN, split count: 7, time: 5.24
Output dim: 7, lower bound: -0.0009546, upper bound: 0.0009297
RS_RSZ1_RSZ1_RSZ1_RSZ2_RSZ1_RSZ1_RSZ1, status: Status.UNKNOWN, split count: 7, time: 5.24
Output dim: 7, lower bound: -0.0009498, upper bound: 0.0009292
RS_RSZ1_RSZ1_RSZ1_RSZ2_RSZ1_RSZ1_RSZ2, status: Status.UNKNOWN, split count: 7, time: 5.24
Output dim: 7, lower bound: -0.0009461, upper bound: 0.0009326
RS_RSZ1_RSZ1_RSZ1_RSZ2_RSZ1_RSZ2_RSZ1, status: Status.UNKNOWN, split count: 7, time: 5.24
Output dim: 7, lower bound: -0.0009497, upper bound: 0.0009291
RS_RSZ1_RSZ1_RSZ1_RSZ2_RSZ1_RSZ2_RSZ2, status: Status.UNKNOWN, split count: 7, time: 5.24
Output dim: 7, lower bound: -0.0009458, upper bound: 0.0009327
RS_RSZ1_RSZ1_RSZ1_RSZ2_RSZ2_RSZ1_RSZ1, status: Status.UNKNOWN, split count: 7, time: 5.24
Output dim: 7, lower bound: -0.0009614, upper bound: 0.0009189
RS_RSZ1_RSZ1_RSZ1_RSZ2_RSZ2_RSZ1_RSZ2, status: Status.UNKNOWN, split count: 7, time: 5.24
Output dim: 7, lower bound: -0.0009566, upper bound: 0.0009243
RS_RSZ1_RSZ1_RSZ1_RSZ2_RSZ2_RSZ2_RSZ1, status: Status.UNKNOWN, split count: 7, time: 5.24
Output dim: 7, lower bound: -0.0009607, upper bound: 0.0009189
RS_RSZ1_RSZ1_RSZ1_RSZ2_RSZ2_RSZ2_RSZ2, status: Status.UNKNOWN, split count: 7, time: 5.24
Output dim: 7, lower bound: -0.0009559, upper bound: 0.0009244
RS_RSZ1_RSZ1_RSZ2_RSZ1_RSZ1_RSZ1_RSZ1, status: Status.UNKNOWN, split count: 7, time: 5.24
Output dim: 7, lower bound: -0.0009496, upper bound: 0.0009333
RS_RSZ1_RSZ1_RSZ2_RSZ1_RSZ1_RSZ1_RSZ2, status: Status.UNKNOWN, split count: 7, time: 5.24
Output dim: 7, lower bound: -0.0009454, upper bound: 0.0009365
RS_RSZ1_RSZ1_RSZ2_RSZ1_RSZ1_RSZ2_RSZ1, status: Status.UNKNOWN, split count: 7, time: 5.24
Output dim: 7, lower bound: -0.0009496, upper bound: 0.0009330
RS_RSZ1_RSZ1_RSZ2_RSZ1_RSZ1_RSZ2_RSZ2, status: Status.UNKNOWN, split count: 7, time: 5.24
Output dim: 7, lower bound: -0.0009452, upper bound: 0.0009362
RS_RSZ1_RSZ1_RSZ2_RSZ1_RSZ2_RSZ1_RSZ1, status: Status.UNKNOWN, split count: 7, time: 5.24
Output dim: 7, lower bound: -0.0009605, upper bound: 0.0009247
RS_RSZ1_RSZ1_RSZ2_RSZ1_RSZ2_RSZ1_RSZ2, status: Status.UNKNOWN, split count: 7, time: 5.24
Output dim: 7, lower bound: -0.0009552, upper bound: 0.0009294
RS_RSZ1_RSZ1_RSZ2_RSZ1_RSZ2_RSZ2_RSZ1, status: Status.UNKNOWN, split count: 7, time: 5.24
Output dim: 7, lower bound: -0.0009601, upper bound: 0.0009248
RS_RSZ1_RSZ1_RSZ2_RSZ1_RSZ2_RSZ2_RSZ2, status: Status.UNKNOWN, split count: 7, time: 5.24
Output dim: 7, lower bound: -0.0009546, upper bound: 0.0009297
RS_RSZ1_RSZ1_RSZ2_RSZ2_RSZ1_RSZ1_RSZ1, status: Status.UNKNOWN, split count: 7, time: 5.24
Output dim: 7, lower bound: -0.0009498, upper bound: 0.0009292
RS_RSZ1_RSZ1_RSZ2_RSZ2_RSZ1_RSZ1_RSZ2, status: Status.UNKNOWN, split count: 7, time: 5.24
Output dim: 7, lower bound: -0.0009461, upper bound: 0.0009326
RS_RSZ1_RSZ1_RSZ2_RSZ2_RSZ1_RSZ2_RSZ1, status: Status.UNKNOWN, split count: 7, time: 5.24
Output dim: 7, lower bound: -0.0009497, upper bound: 0.0009291
RS_RSZ1_RSZ1_RSZ2_RSZ2_RSZ1_RSZ2_RSZ2, status: Status.UNKNOWN, split count: 7, time: 5.24
Output dim: 7, lower bound: -0.0009458, upper bound: 0.0009327
RS_RSZ1_RSZ1_RSZ2_RSZ2_RSZ2_RSZ1_RSZ1, status: Status.UNKNOWN, split count: 7, time: 5.24
Output dim: 7, lower bound: -0.0009614, upper bound: 0.0009189
RS_RSZ1_RSZ1_RSZ2_RSZ2_RSZ2_RSZ1_RSZ2, status: Status.UNKNOWN, split count: 7, time: 5.24
Output dim: 7, lower bound: -0.0009566, upper bound: 0.0009243
RS_RSZ1_RSZ1_RSZ2_RSZ2_RSZ2_RSZ2_RSZ1, status: Status.UNKNOWN, split count: 7, time: 5.24
Output dim: 7, lower bound: -0.0009607, upper bound: 0.0009189
RS_RSZ1_RSZ1_RSZ2_RSZ2_RSZ2_RSZ2_RSZ2, status: Status.UNKNOWN, split count: 7, time: 5.24
Output dim: 7, lower bound: -0.0009559, upper bound: 0.0009244
RS_RSZ1_RSZ2_RSZ1_RSZ1_RSZ1_RSZ1_RSZ1, status: Status.UNKNOWN, split count: 7, time: 5.24
Output dim: 7, lower bound: -0.0009336, upper bound: 0.0009512
RS_RSZ1_RSZ2_RSZ1_RSZ1_RSZ1_RSZ1_RSZ2, status: Status.UNKNOWN, split count: 7, time: 5.24
Output dim: 7, lower bound: -0.0009246, upper bound: 0.0009541
RS_RSZ1_RSZ2_RSZ1_RSZ1_RSZ1_RSZ2_RSZ1, status: Status.UNKNOWN, split count: 7, time: 5.24
Output dim: 7, lower bound: -0.0009336, upper bound: 0.0009521
RS_RSZ1_RSZ2_RSZ1_RSZ1_RSZ1_RSZ2_RSZ2, status: Status.UNKNOWN, split count: 7, time: 5.24
Output dim: 7, lower bound: -0.0009246, upper bound: 0.0009553
RS_RSZ1_RSZ2_RSZ1_RSZ1_RSZ2_RSZ1_RSZ1, status: Status.UNKNOWN, split count: 7, time: 5.24
Output dim: 7, lower bound: -0.0009414, upper bound: 0.0009401
RS_RSZ1_RSZ2_RSZ1_RSZ1_RSZ2_RSZ1_RSZ2, status: Status.UNKNOWN, split count: 7, time: 5.24
Output dim: 7, lower bound: -0.0009337, upper bound: 0.0009419
RS_RSZ1_RSZ2_RSZ1_RSZ1_RSZ2_RSZ2_RSZ1, status: Status.UNKNOWN, split count: 7, time: 5.24
Output dim: 7, lower bound: -0.0009415, upper bound: 0.0009410
RS_RSZ1_RSZ2_RSZ1_RSZ1_RSZ2_RSZ2_RSZ2, status: Status.UNKNOWN, split count: 7, time: 5.24
Output dim: 7, lower bound: -0.0009341, upper bound: 0.0009426
RS_RSZ1_RSZ2_RSZ1_RSZ2_RSZ1_RSZ1_RSZ1, status: Status.UNKNOWN, split count: 7, time: 5.24
Output dim: 7, lower bound: -0.0009341, upper bound: 0.0009428
RS_RSZ1_RSZ2_RSZ1_RSZ2_RSZ1_RSZ1_RSZ2, status: Status.UNKNOWN, split count: 7, time: 5.24
Output dim: 7, lower bound: -0.0009266, upper bound: 0.0009485
RS_RSZ1_RSZ2_RSZ1_RSZ2_RSZ1_RSZ2_RSZ1, status: Status.UNKNOWN, split count: 7, time: 5.24
Output dim: 7, lower bound: -0.0009342, upper bound: 0.0009445
RS_RSZ1_RSZ2_RSZ1_RSZ2_RSZ1_RSZ2_RSZ2, status: Status.UNKNOWN, split count: 7, time: 5.24
Output dim: 7, lower bound: -0.0009266, upper bound: 0.0009504
RS_RSZ1_RSZ2_RSZ1_RSZ2_RSZ2_RSZ1_RSZ1, status: Status.UNKNOWN, split count: 7, time: 5.24
Output dim: 7, lower bound: -0.0009414, upper bound: 0.0009284
RS_RSZ1_RSZ2_RSZ1_RSZ2_RSZ2_RSZ1_RSZ2, status: Status.UNKNOWN, split count: 7, time: 5.24
Output dim: 7, lower bound: -0.0009349, upper bound: 0.0009322
RS_RSZ1_RSZ2_RSZ1_RSZ2_RSZ2_RSZ2_RSZ1, status: Status.UNKNOWN, split count: 7, time: 5.24
Output dim: 7, lower bound: -0.0009420, upper bound: 0.0009291
RS_RSZ1_RSZ2_RSZ1_RSZ2_RSZ2_RSZ2_RSZ2, status: Status.UNKNOWN, split count: 7, time: 5.24
Output dim: 7, lower bound: -0.0009355, upper bound: 0.0009327
RS_RSZ1_RSZ2_RSZ2_RSZ1_RSZ1_RSZ1_RSZ1, status: Status.UNKNOWN, split count: 7, time: 5.24
Output dim: 7, lower bound: -0.0009336, upper bound: 0.0009512
RS_RSZ1_RSZ2_RSZ2_RSZ1_RSZ1_RSZ1_RSZ2, status: Status.UNKNOWN, split count: 7, time: 5.24
Output dim: 7, lower bound: -0.0009246, upper bound: 0.0009541
RS_RSZ1_RSZ2_RSZ2_RSZ1_RSZ1_RSZ2_RSZ1, status: Status.UNKNOWN, split count: 7, time: 5.24
Output dim: 7, lower bound: -0.0009336, upper bound: 0.0009521
RS_RSZ1_RSZ2_RSZ2_RSZ1_RSZ1_RSZ2_RSZ2, status: Status.UNKNOWN, split count: 7, time: 5.24
Output dim: 7, lower bound: -0.0009246, upper bound: 0.0009553
RS_RSZ1_RSZ2_RSZ2_RSZ1_RSZ2_RSZ1_RSZ1, status: Status.UNKNOWN, split count: 7, time: 5.24
Output dim: 7, lower bound: -0.0009414, upper bound: 0.0009401
RS_RSZ1_RSZ2_RSZ2_RSZ1_RSZ2_RSZ1_RSZ2, status: Status.UNKNOWN, split count: 7, time: 5.24
Output dim: 7, lower bound: -0.0009337, upper bound: 0.0009419
RS_RSZ1_RSZ2_RSZ2_RSZ1_RSZ2_RSZ2_RSZ1, status: Status.UNKNOWN, split count: 7, time: 5.24
Output dim: 7, lower bound: -0.0009415, upper bound: 0.0009410
RS_RSZ1_RSZ2_RSZ2_RSZ1_RSZ2_RSZ2_RSZ2, status: Status.UNKNOWN, split count: 7, time: 5.24
Output dim: 7, lower bound: -0.0009341, upper bound: 0.0009426
RS_RSZ1_RSZ2_RSZ2_RSZ2_RSZ1_RSZ1_RSZ1, status: Status.UNKNOWN, split count: 7, time: 5.24
Output dim: 7, lower bound: -0.0009341, upper bound: 0.0009428
RS_RSZ1_RSZ2_RSZ2_RSZ2_RSZ1_RSZ1_RSZ2, status: Status.UNKNOWN, split count: 7, time: 5.24
Output dim: 7, lower bound: -0.0009266, upper bound: 0.0009485
RS_RSZ1_RSZ2_RSZ2_RSZ2_RSZ1_RSZ2_RSZ1, status: Status.UNKNOWN, split count: 7, time: 5.24
Output dim: 7, lower bound: -0.0009342, upper bound: 0.0009445
RS_RSZ1_RSZ2_RSZ2_RSZ2_RSZ1_RSZ2_RSZ2, status: Status.UNKNOWN, split count: 7, time: 5.24
Output dim: 7, lower bound: -0.0009266, upper bound: 0.0009504
RS_RSZ1_RSZ2_RSZ2_RSZ2_RSZ2_RSZ1_RSZ1, status: Status.UNKNOWN, split count: 7, time: 5.24
Output dim: 7, lower bound: -0.0009414, upper bound: 0.0009284
RS_RSZ1_RSZ2_RSZ2_RSZ2_RSZ2_RSZ1_RSZ2, status: Status.UNKNOWN, split count: 7, time: 5.24
Output dim: 7, lower bound: -0.0009349, upper bound: 0.0009321
RS_RSZ1_RSZ2_RSZ2_RSZ2_RSZ2_RSZ2_RSZ1, status: Status.UNKNOWN, split count: 7, time: 5.24
Output dim: 7, lower bound: -0.0009420, upper bound: 0.0009291
RS_RSZ1_RSZ2_RSZ2_RSZ2_RSZ2_RSZ2_RSZ2, status: Status.UNKNOWN, split count: 7, time: 5.24
Output dim: 7, lower bound: -0.0009355, upper bound: 0.0009327
RS_RSZ2_RSZ1_RSZ1_RSZ1_RSZ1_RSZ1_RSZ1, status: Status.UNKNOWN, split count: 7, time: 5.24
Output dim: 7, lower bound: -0.0009327, upper bound: 0.0009355
RS_RSZ2_RSZ1_RSZ1_RSZ1_RSZ1_RSZ1_RSZ2, status: Status.UNKNOWN, split count: 7, time: 5.24
Output dim: 7, lower bound: -0.0009291, upper bound: 0.0009420
RS_RSZ2_RSZ1_RSZ1_RSZ1_RSZ1_RSZ2_RSZ1, status: Status.UNKNOWN, split count: 7, time: 5.24
Output dim: 7, lower bound: -0.0009321, upper bound: 0.0009349
RS_RSZ2_RSZ1_RSZ1_RSZ1_RSZ1_RSZ2_RSZ2, status: Status.UNKNOWN, split count: 7, time: 5.24
Output dim: 7, lower bound: -0.0009284, upper bound: 0.0009414
RS_RSZ2_RSZ1_RSZ1_RSZ1_RSZ2_RSZ1_RSZ1, status: Status.UNKNOWN, split count: 7, time: 5.24
Output dim: 7, lower bound: -0.0009504, upper bound: 0.0009266
RS_RSZ2_RSZ1_RSZ1_RSZ1_RSZ2_RSZ1_RSZ2, status: Status.UNKNOWN, split count: 7, time: 5.24
Output dim: 7, lower bound: -0.0009445, upper bound: 0.0009342
RS_RSZ2_RSZ1_RSZ1_RSZ1_RSZ2_RSZ2_RSZ1, status: Status.UNKNOWN, split count: 7, time: 5.24
Output dim: 7, lower bound: -0.0009485, upper bound: 0.0009266
RS_RSZ2_RSZ1_RSZ1_RSZ1_RSZ2_RSZ2_RSZ2, status: Status.UNKNOWN, split count: 7, time: 5.24
Output dim: 7, lower bound: -0.0009428, upper bound: 0.0009341
RS_RSZ2_RSZ1_RSZ1_RSZ2_RSZ1_RSZ1_RSZ1, status: Status.UNKNOWN, split count: 7, time: 5.24
Output dim: 7, lower bound: -0.0009426, upper bound: 0.0009341
RS_RSZ2_RSZ1_RSZ1_RSZ2_RSZ1_RSZ1_RSZ2, status: Status.UNKNOWN, split count: 7, time: 5.24
Output dim: 7, lower bound: -0.0009410, upper bound: 0.0009415
RS_RSZ2_RSZ1_RSZ1_RSZ2_RSZ1_RSZ2_RSZ1, status: Status.UNKNOWN, split count: 7, time: 5.24
Output dim: 7, lower bound: -0.0009419, upper bound: 0.0009337
RS_RSZ2_RSZ1_RSZ1_RSZ2_RSZ1_RSZ2_RSZ2, status: Status.UNKNOWN, split count: 7, time: 5.24
Output dim: 7, lower bound: -0.0009400, upper bound: 0.0009414
RS_RSZ2_RSZ1_RSZ1_RSZ2_RSZ2_RSZ1_RSZ1, status: Status.UNKNOWN, split count: 7, time: 5.24
Output dim: 7, lower bound: -0.0009553, upper bound: 0.0009246
RS_RSZ2_RSZ1_RSZ1_RSZ2_RSZ2_RSZ1_RSZ2, status: Status.UNKNOWN, split count: 7, time: 5.24
Output dim: 7, lower bound: -0.0009521, upper bound: 0.0009336
RS_RSZ2_RSZ1_RSZ1_RSZ2_RSZ2_RSZ2_RSZ1, status: Status.UNKNOWN, split count: 7, time: 5.24
Output dim: 7, lower bound: -0.0009541, upper bound: 0.0009246
RS_RSZ2_RSZ1_RSZ1_RSZ2_RSZ2_RSZ2_RSZ2, status: Status.UNKNOWN, split count: 7, time: 5.24
Output dim: 7, lower bound: -0.0009512, upper bound: 0.0009336
RS_RSZ2_RSZ1_RSZ2_RSZ1_RSZ1_RSZ1_RSZ1, status: Status.UNKNOWN, split count: 7, time: 5.24
Output dim: 7, lower bound: -0.0009327, upper bound: 0.0009355
RS_RSZ2_RSZ1_RSZ2_RSZ1_RSZ1_RSZ1_RSZ2, status: Status.UNKNOWN, split count: 7, time: 5.24
Output dim: 7, lower bound: -0.0009291, upper bound: 0.0009420
RS_RSZ2_RSZ1_RSZ2_RSZ1_RSZ1_RSZ2_RSZ1, status: Status.UNKNOWN, split count: 7, time: 5.24
Output dim: 7, lower bound: -0.0009321, upper bound: 0.0009349
RS_RSZ2_RSZ1_RSZ2_RSZ1_RSZ1_RSZ2_RSZ2, status: Status.UNKNOWN, split count: 7, time: 5.24
Output dim: 7, lower bound: -0.0009284, upper bound: 0.0009414
RS_RSZ2_RSZ1_RSZ2_RSZ1_RSZ2_RSZ1_RSZ1, status: Status.UNKNOWN, split count: 7, time: 5.24
Output dim: 7, lower bound: -0.0009504, upper bound: 0.0009266
RS_RSZ2_RSZ1_RSZ2_RSZ1_RSZ2_RSZ1_RSZ2, status: Status.UNKNOWN, split count: 7, time: 5.24
Output dim: 7, lower bound: -0.0009445, upper bound: 0.0009342
RS_RSZ2_RSZ1_RSZ2_RSZ1_RSZ2_RSZ2_RSZ1, status: Status.UNKNOWN, split count: 7, time: 5.24
Output dim: 7, lower bound: -0.0009485, upper bound: 0.0009266
RS_RSZ2_RSZ1_RSZ2_RSZ1_RSZ2_RSZ2_RSZ2, status: Status.UNKNOWN, split count: 7, time: 5.24
Output dim: 7, lower bound: -0.0009428, upper bound: 0.0009341
RS_RSZ2_RSZ1_RSZ2_RSZ2_RSZ1_RSZ1_RSZ1, status: Status.UNKNOWN, split count: 7, time: 5.24
Output dim: 7, lower bound: -0.0009426, upper bound: 0.0009341
RS_RSZ2_RSZ1_RSZ2_RSZ2_RSZ1_RSZ1_RSZ2, status: Status.UNKNOWN, split count: 7, time: 5.24
Output dim: 7, lower bound: -0.0009410, upper bound: 0.0009415
RS_RSZ2_RSZ1_RSZ2_RSZ2_RSZ1_RSZ2_RSZ1, status: Status.UNKNOWN, split count: 7, time: 5.24
Output dim: 7, lower bound: -0.0009419, upper bound: 0.0009337
RS_RSZ2_RSZ1_RSZ2_RSZ2_RSZ1_RSZ2_RSZ2, status: Status.UNKNOWN, split count: 7, time: 5.24
Output dim: 7, lower bound: -0.0009400, upper bound: 0.0009414
RS_RSZ2_RSZ1_RSZ2_RSZ2_RSZ2_RSZ1_RSZ1, status: Status.UNKNOWN, split count: 7, time: 5.24
Output dim: 7, lower bound: -0.0009553, upper bound: 0.0009246
RS_RSZ2_RSZ1_RSZ2_RSZ2_RSZ2_RSZ1_RSZ2, status: Status.UNKNOWN, split count: 7, time: 5.24
Output dim: 7, lower bound: -0.0009521, upper bound: 0.0009336
RS_RSZ2_RSZ1_RSZ2_RSZ2_RSZ2_RSZ2_RSZ1, status: Status.UNKNOWN, split count: 7, time: 5.24
Output dim: 7, lower bound: -0.0009541, upper bound: 0.0009246
RS_RSZ2_RSZ1_RSZ2_RSZ2_RSZ2_RSZ2_RSZ2, status: Status.UNKNOWN, split count: 7, time: 5.24
Output dim: 7, lower bound: -0.0009512, upper bound: 0.0009336
RS_RSZ2_RSZ2_RSZ1_RSZ1_RSZ1_RSZ1_RSZ1, status: Status.UNKNOWN, split count: 7, time: 5.24
Output dim: 7, lower bound: -0.0009244, upper bound: 0.0009559
RS_RSZ2_RSZ2_RSZ1_RSZ1_RSZ1_RSZ1_RSZ2, status: Status.UNKNOWN, split count: 7, time: 5.24
Output dim: 7, lower bound: -0.0009189, upper bound: 0.0009607
RS_RSZ2_RSZ2_RSZ1_RSZ1_RSZ1_RSZ2_RSZ1, status: Status.UNKNOWN, split count: 7, time: 5.24
Output dim: 7, lower bound: -0.0009243, upper bound: 0.0009566
RS_RSZ2_RSZ2_RSZ1_RSZ1_RSZ1_RSZ2_RSZ2, status: Status.UNKNOWN, split count: 7, time: 5.24
Output dim: 7, lower bound: -0.0009189, upper bound: 0.0009614
RS_RSZ2_RSZ2_RSZ1_RSZ1_RSZ2_RSZ1_RSZ1, status: Status.UNKNOWN, split count: 7, time: 5.24
Output dim: 7, lower bound: -0.0009327, upper bound: 0.0009458
RS_RSZ2_RSZ2_RSZ1_RSZ1_RSZ2_RSZ1_RSZ2, status: Status.UNKNOWN, split count: 7, time: 5.24
Output dim: 7, lower bound: -0.0009291, upper bound: 0.0009498
RS_RSZ2_RSZ2_RSZ1_RSZ1_RSZ2_RSZ2_RSZ1, status: Status.UNKNOWN, split count: 7, time: 5.24
Output dim: 7, lower bound: -0.0009326, upper bound: 0.0009461
RS_RSZ2_RSZ2_RSZ1_RSZ1_RSZ2_RSZ2_RSZ2, status: Status.UNKNOWN, split count: 7, time: 5.24
Output dim: 7, lower bound: -0.0009292, upper bound: 0.0009498
RS_RSZ2_RSZ2_RSZ1_RSZ2_RSZ1_RSZ1_RSZ1, status: Status.UNKNOWN, split count: 7, time: 5.24
Output dim: 7, lower bound: -0.0009297, upper bound: 0.0009546
RS_RSZ2_RSZ2_RSZ1_RSZ2_RSZ1_RSZ1_RSZ2, status: Status.UNKNOWN, split count: 7, time: 5.24
Output dim: 7, lower bound: -0.0009248, upper bound: 0.0009601
RS_RSZ2_RSZ2_RSZ1_RSZ2_RSZ1_RSZ2_RSZ1, status: Status.UNKNOWN, split count: 7, time: 5.24
Output dim: 7, lower bound: -0.0009294, upper bound: 0.0009552
RS_RSZ2_RSZ2_RSZ1_RSZ2_RSZ1_RSZ2_RSZ2, status: Status.UNKNOWN, split count: 7, time: 5.24
Output dim: 7, lower bound: -0.0009247, upper bound: 0.0009605
RS_RSZ2_RSZ2_RSZ1_RSZ2_RSZ2_RSZ1_RSZ1, status: Status.UNKNOWN, split count: 7, time: 5.24
Output dim: 7, lower bound: -0.0009362, upper bound: 0.0009452
RS_RSZ2_RSZ2_RSZ1_RSZ2_RSZ2_RSZ1_RSZ2, status: Status.UNKNOWN, split count: 7, time: 5.24
Output dim: 7, lower bound: -0.0009330, upper bound: 0.0009496
RS_RSZ2_RSZ2_RSZ1_RSZ2_RSZ2_RSZ2_RSZ1, status: Status.UNKNOWN, split count: 7, time: 5.24
Output dim: 7, lower bound: -0.0009365, upper bound: 0.0009454
RS_RSZ2_RSZ2_RSZ1_RSZ2_RSZ2_RSZ2_RSZ2, status: Status.UNKNOWN, split count: 7, time: 5.24
Output dim: 7, lower bound: -0.0009333, upper bound: 0.0009496
RS_RSZ2_RSZ2_RSZ2_RSZ1_RSZ1_RSZ1_RSZ1, status: Status.UNKNOWN, split count: 7, time: 5.24
Output dim: 7, lower bound: -0.0009244, upper bound: 0.0009559
RS_RSZ2_RSZ2_RSZ2_RSZ1_RSZ1_RSZ1_RSZ2, status: Status.UNKNOWN, split count: 7, time: 5.24
Output dim: 7, lower bound: -0.0009189, upper bound: 0.0009607
RS_RSZ2_RSZ2_RSZ2_RSZ1_RSZ1_RSZ2_RSZ1, status: Status.UNKNOWN, split count: 7, time: 5.24
Output dim: 7, lower bound: -0.0009243, upper bound: 0.0009566
RS_RSZ2_RSZ2_RSZ2_RSZ1_RSZ1_RSZ2_RSZ2, status: Status.UNKNOWN, split count: 7, time: 5.24
Output dim: 7, lower bound: -0.0009189, upper bound: 0.0009614
RS_RSZ2_RSZ2_RSZ2_RSZ1_RSZ2_RSZ1_RSZ1, status: Status.UNKNOWN, split count: 7, time: 5.24
Output dim: 7, lower bound: -0.0009327, upper bound: 0.0009458
RS_RSZ2_RSZ2_RSZ2_RSZ1_RSZ2_RSZ1_RSZ2, status: Status.UNKNOWN, split count: 7, time: 5.24
Output dim: 7, lower bound: -0.0009291, upper bound: 0.0009498
RS_RSZ2_RSZ2_RSZ2_RSZ1_RSZ2_RSZ2_RSZ1, status: Status.UNKNOWN, split count: 7, time: 5.24
Output dim: 7, lower bound: -0.0009326, upper bound: 0.0009461
RS_RSZ2_RSZ2_RSZ2_RSZ1_RSZ2_RSZ2_RSZ2, status: Status.UNKNOWN, split count: 7, time: 5.24
Output dim: 7, lower bound: -0.0009292, upper bound: 0.0009498
RS_RSZ2_RSZ2_RSZ2_RSZ2_RSZ1_RSZ1_RSZ1, status: Status.UNKNOWN, split count: 7, time: 5.24
Output dim: 7, lower bound: -0.0009297, upper bound: 0.0009546
RS_RSZ2_RSZ2_RSZ2_RSZ2_RSZ1_RSZ1_RSZ2, status: Status.UNKNOWN, split count: 7, time: 5.24
Output dim: 7, lower bound: -0.0009248, upper bound: 0.0009601
RS_RSZ2_RSZ2_RSZ2_RSZ2_RSZ1_RSZ2_RSZ1, status: Status.UNKNOWN, split count: 7, time: 5.24
Output dim: 7, lower bound: -0.0009294, upper bound: 0.0009552
RS_RSZ2_RSZ2_RSZ2_RSZ2_RSZ1_RSZ2_RSZ2, status: Status.UNKNOWN, split count: 7, time: 5.24
Output dim: 7, lower bound: -0.0009247, upper bound: 0.0009605
RS_RSZ2_RSZ2_RSZ2_RSZ2_RSZ2_RSZ1_RSZ1, status: Status.UNKNOWN, split count: 7, time: 5.24
Output dim: 7, lower bound: -0.0009362, upper bound: 0.0009452
RS_RSZ2_RSZ2_RSZ2_RSZ2_RSZ2_RSZ1_RSZ2, status: Status.UNKNOWN, split count: 7, time: 5.24
Output dim: 7, lower bound: -0.0009330, upper bound: 0.0009496
RS_RSZ2_RSZ2_RSZ2_RSZ2_RSZ2_RSZ2_RSZ1, status: Status.UNKNOWN, split count: 7, time: 5.24
Output dim: 7, lower bound: -0.0009365, upper bound: 0.0009454
RS_RSZ2_RSZ2_RSZ2_RSZ2_RSZ2_RSZ2_RSZ2, status: Status.UNKNOWN, split count: 7, time: 5.24
Output dim: 7, lower bound: -0.0009333, upper bound: 0.0009496

## BFS RS instance: RS_RSZ1_RSZ1_RSZ1_RSZ1_RSZ1_RSZ1_RSZ1

### Backsubstitution after applying RS history:
0: -0.0027939, -0.0016242, -0.0027939, -0.0016242, -0.0007322, 0.0007359
1: -0.0114005, -0.0084323, -0.0114005, -0.0084323, -0.0018579, 0.0018676
2: 0.0279571, 0.0297986, 0.0279571, 0.0297986, -0.0011527, 0.0011586
3: 0.0039210, 0.0073596, 0.0039210, 0.0073596, -0.0021635, 0.0021523
4: -0.0104893, -0.0074701, -0.0104893, -0.0074701, -0.0018898, 0.0018996
5: 0.0097651, 0.0109087, 0.0097651, 0.0109087, -0.0007158, 0.0007195
6: 0.0053643, 0.0097284, 0.0053643, 0.0097284, -0.0027457, 0.0027316
7: 0.9818129, 0.9848667, 0.9818129, 0.9848667, -0.0019213, 0.0019114
8: -0.0060636, -0.0027895, -0.0060636, -0.0027895, -0.0020600, 0.0020494
9: -0.0031570, -0.0009942, -0.0031570, -0.0009942, -0.0013537, 0.0013607

### Unstable ReLU Count (Linear/Conv2D Layers)
- layer_idx=0, type=LayerType.Linear, total=256, inp1_unstable=10, inp2_unstable=10, delta_unstable=249
- layer_idx=2, type=LayerType.Linear, total=256, inp1_unstable=10, inp2_unstable=10, delta_unstable=256
- layer_idx=4, type=LayerType.Linear, total=256, inp1_unstable=0, inp2_unstable=0, delta_unstable=256
- layer_idx=6, type=LayerType.Linear, total=256, inp1_unstable=0, inp2_unstable=0, delta_unstable=256
- layer_idx=8, type=LayerType.Linear, total=10, inp1_unstable=0, inp2_unstable=0, delta_unstable=10

Time for backsubstitution: 1.62 seconds

### RS candidates at layer 1
type: RSZ, layer: 1, pos: 140
type: RSZ, layer: 1, pos: 213
type: RSZ, layer: 1, pos: 219

Time for candidate selection: 0.17 seconds

### Candidate
type: RSZ, layer: 1, pos: 140

### Relational analysis RSZ of RS_RSZ1_RSZ1_RSZ1_RSZ1_RSZ1_RSZ1_RSZ1_RSZ1
Optimization infeasible because this subproblem isn't reachable.

### Candidate
type: RSZ, layer: 1, pos: 213

### Relational analysis RSZ of RS_RSZ1_RSZ1_RSZ1_RSZ1_RSZ1_RSZ1_RSZ1_RSZ1

#### Relational analysis RSZ result of RS_RSZ1_RSZ1_RSZ1_RSZ1_RSZ1_RSZ1_RSZ1_RSZ1
Status: Status.UNKNOWN
Output dim: 7, lower bound: -0.0008022, upper bound: 0.0007873
time: 0.69 seconds

### Relational analysis RSZ of RS_RSZ1_RSZ1_RSZ1_RSZ1_RSZ1_RSZ1_RSZ1_RSZ2

#### Relational analysis RSZ result of RS_RSZ1_RSZ1_RSZ1_RSZ1_RSZ1_RSZ1_RSZ1_RSZ2
Status: Status.UNKNOWN
Output dim: 7, lower bound: -0.0008022, upper bound: 0.0007873
time: 0.68 seconds

## BFS RS instance: RS_RSZ1_RSZ1_RSZ1_RSZ1_RSZ1_RSZ1_RSZ2

### Backsubstitution after applying RS history:
0: -0.0027939, -0.0016242, -0.0027939, -0.0016242, -0.0007338, 0.0007327
1: -0.0114005, -0.0084323, -0.0114005, -0.0084323, -0.0018620, 0.0018594
2: 0.0279571, 0.0297986, 0.0279571, 0.0297986, -0.0011552, 0.0011536
3: 0.0039210, 0.0073596, 0.0039210, 0.0073596, -0.0021540, 0.0021571
4: -0.0104893, -0.0074701, -0.0104893, -0.0074701, -0.0018940, 0.0018913
5: 0.0097651, 0.0109087, 0.0097651, 0.0109087, -0.0007174, 0.0007164
6: 0.0053643, 0.0097284, 0.0053643, 0.0097284, -0.0027337, 0.0027376
7: 0.9818129, 0.9848667, 0.9818129, 0.9848667, -0.0019129, 0.0019156
8: -0.0060636, -0.0027895, -0.0060636, -0.0027895, -0.0020509, 0.0020539
9: -0.0031570, -0.0009942, -0.0031570, -0.0009942, -0.0013567, 0.0013548

### Unstable ReLU Count (Linear/Conv2D Layers)
- layer_idx=0, type=LayerType.Linear, total=256, inp1_unstable=10, inp2_unstable=10, delta_unstable=249
- layer_idx=2, type=LayerType.Linear, total=256, inp1_unstable=10, inp2_unstable=10, delta_unstable=256
- layer_idx=4, type=LayerType.Linear, total=256, inp1_unstable=0, inp2_unstable=0, delta_unstable=256
- layer_idx=6, type=LayerType.Linear, total=256, inp1_unstable=0, inp2_unstable=0, delta_unstable=256
- layer_idx=8, type=LayerType.Linear, total=10, inp1_unstable=0, inp2_unstable=0, delta_unstable=10

Time for backsubstitution: 1.60 seconds

### RS candidates at layer 1
type: RSZ, layer: 1, pos: 140
type: RSZ, layer: 1, pos: 213
type: RSZ, layer: 1, pos: 219

Time for candidate selection: 0.17 seconds

### Candidate
type: RSZ, layer: 1, pos: 140

### Relational analysis RSZ of RS_RSZ1_RSZ1_RSZ1_RSZ1_RSZ1_RSZ1_RSZ2_RSZ1
Optimization infeasible because this subproblem isn't reachable.

### Candidate
type: RSZ, layer: 1, pos: 213

### Relational analysis RSZ of RS_RSZ1_RSZ1_RSZ1_RSZ1_RSZ1_RSZ1_RSZ2_RSZ1

#### Relational analysis RSZ result of RS_RSZ1_RSZ1_RSZ1_RSZ1_RSZ1_RSZ1_RSZ2_RSZ1
Status: Status.VERIFIED
Output dim: 7, lower bound: -0.0007937, upper bound: 0.0007934
time: 0.70 seconds

### Relational analysis RSZ of RS_RSZ1_RSZ1_RSZ1_RSZ1_RSZ1_RSZ1_RSZ2_RSZ2

#### Relational analysis RSZ result of RS_RSZ1_RSZ1_RSZ1_RSZ1_RSZ1_RSZ1_RSZ2_RSZ2
Status: Status.VERIFIED
Output dim: 7, lower bound: -0.0007937, upper bound: 0.0007934
time: 0.68 seconds

## BFS RS instance: RS_RSZ1_RSZ1_RSZ1_RSZ1_RSZ1_RSZ2_RSZ1

### Backsubstitution after applying RS history:
0: -0.0027939, -0.0016242, -0.0027939, -0.0016242, -0.0007323, 0.0007358
1: -0.0114005, -0.0084323, -0.0114005, -0.0084323, -0.0018583, 0.0018671
2: 0.0279571, 0.0297986, 0.0279571, 0.0297986, -0.0011529, 0.0011584
3: 0.0039210, 0.0073596, 0.0039210, 0.0073596, -0.0021630, 0.0021528
4: -0.0104893, -0.0074701, -0.0104893, -0.0074701, -0.0018902, 0.0018992
5: 0.0097651, 0.0109087, 0.0097651, 0.0109087, -0.0007160, 0.0007194
6: 0.0053643, 0.0097284, 0.0053643, 0.0097284, -0.0027451, 0.0027322
7: 0.9818129, 0.9848667, 0.9818129, 0.9848667, -0.0019209, 0.0019118
8: -0.0060636, -0.0027895, -0.0060636, -0.0027895, -0.0020595, 0.0020498
9: -0.0031570, -0.0009942, -0.0031570, -0.0009942, -0.0013540, 0.0013604

### Unstable ReLU Count (Linear/Conv2D Layers)
- layer_idx=0, type=LayerType.Linear, total=256, inp1_unstable=10, inp2_unstable=10, delta_unstable=249
- layer_idx=2, type=LayerType.Linear, total=256, inp1_unstable=10, inp2_unstable=10, delta_unstable=256
- layer_idx=4, type=LayerType.Linear, total=256, inp1_unstable=0, inp2_unstable=0, delta_unstable=256
- layer_idx=6, type=LayerType.Linear, total=256, inp1_unstable=0, inp2_unstable=0, delta_unstable=256
- layer_idx=8, type=LayerType.Linear, total=10, inp1_unstable=0, inp2_unstable=0, delta_unstable=10

Time for backsubstitution: 1.72 seconds

### RS candidates at layer 1
type: RSZ, layer: 1, pos: 140
type: RSZ, layer: 1, pos: 213
type: RSZ, layer: 1, pos: 219

Time for candidate selection: 0.18 seconds

### Candidate
type: RSZ, layer: 1, pos: 140

### Relational analysis RSZ of RS_RSZ1_RSZ1_RSZ1_RSZ1_RSZ1_RSZ2_RSZ1_RSZ1
Optimization infeasible because this subproblem isn't reachable.

### Candidate
type: RSZ, layer: 1, pos: 213

### Relational analysis RSZ of RS_RSZ1_RSZ1_RSZ1_RSZ1_RSZ1_RSZ2_RSZ1_RSZ1

#### Relational analysis RSZ result of RS_RSZ1_RSZ1_RSZ1_RSZ1_RSZ1_RSZ2_RSZ1_RSZ1
Status: Status.UNKNOWN
Output dim: 7, lower bound: -0.0008040, upper bound: 0.0007859
time: 0.74 seconds

### Relational analysis RSZ of RS_RSZ1_RSZ1_RSZ1_RSZ1_RSZ1_RSZ2_RSZ1_RSZ2

#### Relational analysis RSZ result of RS_RSZ1_RSZ1_RSZ1_RSZ1_RSZ1_RSZ2_RSZ1_RSZ2
Status: Status.UNKNOWN
Output dim: 7, lower bound: -0.0008040, upper bound: 0.0007859
time: 0.73 seconds

## BFS RS instance: RS_RSZ1_RSZ1_RSZ1_RSZ1_RSZ1_RSZ2_RSZ2

### Backsubstitution after applying RS history:
0: -0.0027939, -0.0016242, -0.0027939, -0.0016242, -0.0007339, 0.0007325
1: -0.0114005, -0.0084323, -0.0114005, -0.0084323, -0.0018624, 0.0018588
2: 0.0279571, 0.0297986, 0.0279571, 0.0297986, -0.0011555, 0.0011532
3: 0.0039210, 0.0073596, 0.0039210, 0.0073596, -0.0021533, 0.0021575
4: -0.0104893, -0.0074701, -0.0104893, -0.0074701, -0.0018944, 0.0018907
5: 0.0097651, 0.0109087, 0.0097651, 0.0109087, -0.0007175, 0.0007161
6: 0.0053643, 0.0097284, 0.0053643, 0.0097284, -0.0027328, 0.0027382
7: 0.9818129, 0.9848667, 0.9818129, 0.9848667, -0.0019123, 0.0019161
8: -0.0060636, -0.0027895, -0.0060636, -0.0027895, -0.0020503, 0.0020543
9: -0.0031570, -0.0009942, -0.0031570, -0.0009942, -0.0013570, 0.0013543

### Unstable ReLU Count (Linear/Conv2D Layers)
- layer_idx=0, type=LayerType.Linear, total=256, inp1_unstable=10, inp2_unstable=10, delta_unstable=249
- layer_idx=2, type=LayerType.Linear, total=256, inp1_unstable=10, inp2_unstable=10, delta_unstable=256
- layer_idx=4, type=LayerType.Linear, total=256, inp1_unstable=0, inp2_unstable=0, delta_unstable=256
- layer_idx=6, type=LayerType.Linear, total=256, inp1_unstable=0, inp2_unstable=0, delta_unstable=256
- layer_idx=8, type=LayerType.Linear, total=10, inp1_unstable=0, inp2_unstable=0, delta_unstable=10

Time for backsubstitution: 1.57 seconds

### RS candidates at layer 1
type: RSZ, layer: 1, pos: 140
type: RSZ, layer: 1, pos: 213
type: RSZ, layer: 1, pos: 219

Time for candidate selection: 0.17 seconds

### Candidate
type: RSZ, layer: 1, pos: 140

### Relational analysis RSZ of RS_RSZ1_RSZ1_RSZ1_RSZ1_RSZ1_RSZ2_RSZ2_RSZ1
Optimization infeasible because this subproblem isn't reachable.

### Candidate
type: RSZ, layer: 1, pos: 213

### Relational analysis RSZ of RS_RSZ1_RSZ1_RSZ1_RSZ1_RSZ1_RSZ2_RSZ2_RSZ1

#### Relational analysis RSZ result of RS_RSZ1_RSZ1_RSZ1_RSZ1_RSZ1_RSZ2_RSZ2_RSZ1
Status: Status.VERIFIED
Output dim: 7, lower bound: -0.0007960, upper bound: 0.0007916
time: 0.66 seconds

### Relational analysis RSZ of RS_RSZ1_RSZ1_RSZ1_RSZ1_RSZ1_RSZ2_RSZ2_RSZ2

#### Relational analysis RSZ result of RS_RSZ1_RSZ1_RSZ1_RSZ1_RSZ1_RSZ2_RSZ2_RSZ2
Status: Status.VERIFIED
Output dim: 7, lower bound: -0.0007960, upper bound: 0.0007916
time: 0.66 seconds

## BFS RS instance: RS_RSZ1_RSZ1_RSZ1_RSZ1_RSZ2_RSZ1_RSZ1

### Backsubstitution after applying RS history:
0: -0.0027939, -0.0016242, -0.0027939, -0.0016242, -0.0007172, 0.0007512
1: -0.0114005, -0.0084323, -0.0114005, -0.0084323, -0.0018200, 0.0019064
2: 0.0279571, 0.0297986, 0.0279571, 0.0297986, -0.0011291, 0.0011827
3: 0.0039210, 0.0073596, 0.0039210, 0.0073596, -0.0022084, 0.0021084
4: -0.0104893, -0.0074701, -0.0104893, -0.0074701, -0.0018512, 0.0019391
5: 0.0097651, 0.0109087, 0.0097651, 0.0109087, -0.0007012, 0.0007345
6: 0.0053643, 0.0097284, 0.0053643, 0.0097284, -0.0028028, 0.0026758
7: 0.9818129, 0.9848667, 0.9818129, 0.9848667, -0.0019612, 0.0018724
8: -0.0060636, -0.0027895, -0.0060636, -0.0027895, -0.0021028, 0.0020075
9: -0.0031570, -0.0009942, -0.0031570, -0.0009942, -0.0013261, 0.0013890

### Unstable ReLU Count (Linear/Conv2D Layers)
- layer_idx=0, type=LayerType.Linear, total=256, inp1_unstable=10, inp2_unstable=10, delta_unstable=249
- layer_idx=2, type=LayerType.Linear, total=256, inp1_unstable=10, inp2_unstable=10, delta_unstable=256
- layer_idx=4, type=LayerType.Linear, total=256, inp1_unstable=0, inp2_unstable=0, delta_unstable=256
- layer_idx=6, type=LayerType.Linear, total=256, inp1_unstable=0, inp2_unstable=0, delta_unstable=256
- layer_idx=8, type=LayerType.Linear, total=10, inp1_unstable=0, inp2_unstable=0, delta_unstable=10

Time for backsubstitution: 1.58 seconds

### RS candidates at layer 1
type: RSZ, layer: 1, pos: 140
type: RSZ, layer: 1, pos: 213
type: RSZ, layer: 1, pos: 219

Time for candidate selection: 0.18 seconds

### Candidate
type: RSZ, layer: 1, pos: 140

### Relational analysis RSZ of RS_RSZ1_RSZ1_RSZ1_RSZ1_RSZ2_RSZ1_RSZ1_RSZ1
Optimization infeasible because this subproblem isn't reachable.

### Candidate
type: RSZ, layer: 1, pos: 213

### Relational analysis RSZ of RS_RSZ1_RSZ1_RSZ1_RSZ1_RSZ2_RSZ1_RSZ1_RSZ1

#### Relational analysis RSZ result of RS_RSZ1_RSZ1_RSZ1_RSZ1_RSZ2_RSZ1_RSZ1_RSZ1
Status: Status.UNKNOWN
Output dim: 7, lower bound: -0.0008041, upper bound: 0.0007866
time: 0.68 seconds

### Relational analysis RSZ of RS_RSZ1_RSZ1_RSZ1_RSZ1_RSZ2_RSZ1_RSZ1_RSZ2

#### Relational analysis RSZ result of RS_RSZ1_RSZ1_RSZ1_RSZ1_RSZ2_RSZ1_RSZ1_RSZ2
Status: Status.UNKNOWN
Output dim: 7, lower bound: -0.0008041, upper bound: 0.0007866
time: 0.67 seconds

## BFS RS instance: RS_RSZ1_RSZ1_RSZ1_RSZ1_RSZ2_RSZ1_RSZ2

### Backsubstitution after applying RS history:
0: -0.0027939, -0.0016242, -0.0027939, -0.0016242, -0.0007189, 0.0007483
1: -0.0114005, -0.0084323, -0.0114005, -0.0084323, -0.0018243, 0.0018990
2: 0.0279571, 0.0297986, 0.0279571, 0.0297986, -0.0011318, 0.0011782
3: 0.0039210, 0.0073596, 0.0039210, 0.0073596, -0.0021999, 0.0021134
4: -0.0104893, -0.0074701, -0.0104893, -0.0074701, -0.0018556, 0.0019316
5: 0.0097651, 0.0109087, 0.0097651, 0.0109087, -0.0007029, 0.0007316
6: 0.0053643, 0.0097284, 0.0053643, 0.0097284, -0.0027920, 0.0026821
7: 0.9818129, 0.9848667, 0.9818129, 0.9848667, -0.0019537, 0.0018768
8: -0.0060636, -0.0027895, -0.0060636, -0.0027895, -0.0020947, 0.0020123
9: -0.0031570, -0.0009942, -0.0031570, -0.0009942, -0.0013292, 0.0013836

### Unstable ReLU Count (Linear/Conv2D Layers)
- layer_idx=0, type=LayerType.Linear, total=256, inp1_unstable=10, inp2_unstable=10, delta_unstable=249
- layer_idx=2, type=LayerType.Linear, total=256, inp1_unstable=10, inp2_unstable=10, delta_unstable=256
- layer_idx=4, type=LayerType.Linear, total=256, inp1_unstable=0, inp2_unstable=0, delta_unstable=256
- layer_idx=6, type=LayerType.Linear, total=256, inp1_unstable=0, inp2_unstable=0, delta_unstable=256
- layer_idx=8, type=LayerType.Linear, total=10, inp1_unstable=0, inp2_unstable=0, delta_unstable=10

Time for backsubstitution: 1.58 seconds

### RS candidates at layer 1
type: RSZ, layer: 1, pos: 140
type: RSZ, layer: 1, pos: 213
type: RSZ, layer: 1, pos: 219

Time for candidate selection: 0.18 seconds

### Candidate
type: RSZ, layer: 1, pos: 140

### Relational analysis RSZ of RS_RSZ1_RSZ1_RSZ1_RSZ1_RSZ2_RSZ1_RSZ2_RSZ1
Optimization infeasible because this subproblem isn't reachable.

### Candidate
type: RSZ, layer: 1, pos: 213

### Relational analysis RSZ of RS_RSZ1_RSZ1_RSZ1_RSZ1_RSZ2_RSZ1_RSZ2_RSZ1

#### Relational analysis RSZ result of RS_RSZ1_RSZ1_RSZ1_RSZ1_RSZ2_RSZ1_RSZ2_RSZ1
Status: Status.VERIFIED
Output dim: 7, lower bound: -0.0007951, upper bound: 0.0007917
time: 0.82 seconds

### Relational analysis RSZ of RS_RSZ1_RSZ1_RSZ1_RSZ1_RSZ2_RSZ1_RSZ2_RSZ2

#### Relational analysis RSZ result of RS_RSZ1_RSZ1_RSZ1_RSZ1_RSZ2_RSZ1_RSZ2_RSZ2
Status: Status.VERIFIED
Output dim: 7, lower bound: -0.0007951, upper bound: 0.0007917
time: 0.85 seconds

## BFS RS instance: RS_RSZ1_RSZ1_RSZ1_RSZ1_RSZ2_RSZ2_RSZ1

### Backsubstitution after applying RS history:
0: -0.0027939, -0.0016242, -0.0027939, -0.0016242, -0.0007174, 0.0007512
1: -0.0114005, -0.0084323, -0.0114005, -0.0084323, -0.0018205, 0.0019063
2: 0.0279571, 0.0297986, 0.0279571, 0.0297986, -0.0011294, 0.0011827
3: 0.0039210, 0.0073596, 0.0039210, 0.0073596, -0.0022083, 0.0021089
4: -0.0104893, -0.0074701, -0.0104893, -0.0074701, -0.0018517, 0.0019390
5: 0.0097651, 0.0109087, 0.0097651, 0.0109087, -0.0007014, 0.0007344
6: 0.0053643, 0.0097284, 0.0053643, 0.0097284, -0.0028027, 0.0026765
7: 0.9818129, 0.9848667, 0.9818129, 0.9848667, -0.0019612, 0.0018729
8: -0.0060636, -0.0027895, -0.0060636, -0.0027895, -0.0021027, 0.0020080
9: -0.0031570, -0.0009942, -0.0031570, -0.0009942, -0.0013264, 0.0013889

### Unstable ReLU Count (Linear/Conv2D Layers)
- layer_idx=0, type=LayerType.Linear, total=256, inp1_unstable=10, inp2_unstable=10, delta_unstable=249
- layer_idx=2, type=LayerType.Linear, total=256, inp1_unstable=10, inp2_unstable=10, delta_unstable=256
- layer_idx=4, type=LayerType.Linear, total=256, inp1_unstable=0, inp2_unstable=0, delta_unstable=256
- layer_idx=6, type=LayerType.Linear, total=256, inp1_unstable=0, inp2_unstable=0, delta_unstable=256
- layer_idx=8, type=LayerType.Linear, total=10, inp1_unstable=0, inp2_unstable=0, delta_unstable=10

Time for backsubstitution: 1.72 seconds

### RS candidates at layer 1
type: RSZ, layer: 1, pos: 140
type: RSZ, layer: 1, pos: 213
type: RSZ, layer: 1, pos: 219

Time for candidate selection: 0.20 seconds

### Candidate
type: RSZ, layer: 1, pos: 140

### Relational analysis RSZ of RS_RSZ1_RSZ1_RSZ1_RSZ1_RSZ2_RSZ2_RSZ1_RSZ1
Optimization infeasible because this subproblem isn't reachable.

### Candidate
type: RSZ, layer: 1, pos: 213

### Relational analysis RSZ of RS_RSZ1_RSZ1_RSZ1_RSZ1_RSZ2_RSZ2_RSZ1_RSZ1

#### Relational analysis RSZ result of RS_RSZ1_RSZ1_RSZ1_RSZ1_RSZ2_RSZ2_RSZ1_RSZ1
Status: Status.UNKNOWN
Output dim: 7, lower bound: -0.0008052, upper bound: 0.0007852
time: 0.65 seconds

### Relational analysis RSZ of RS_RSZ1_RSZ1_RSZ1_RSZ1_RSZ2_RSZ2_RSZ1_RSZ2

#### Relational analysis RSZ result of RS_RSZ1_RSZ1_RSZ1_RSZ1_RSZ2_RSZ2_RSZ1_RSZ2
Status: Status.UNKNOWN
Output dim: 7, lower bound: -0.0008052, upper bound: 0.0007852
time: 0.72 seconds

## BFS RS instance: RS_RSZ1_RSZ1_RSZ1_RSZ1_RSZ2_RSZ2_RSZ2

### Backsubstitution after applying RS history:
0: -0.0027939, -0.0016242, -0.0027939, -0.0016242, -0.0007191, 0.0007482
1: -0.0114005, -0.0084323, -0.0114005, -0.0084323, -0.0018247, 0.0018986
2: 0.0279571, 0.0297986, 0.0279571, 0.0297986, -0.0011321, 0.0011779
3: 0.0039210, 0.0073596, 0.0039210, 0.0073596, -0.0021994, 0.0021139
4: -0.0104893, -0.0074701, -0.0104893, -0.0074701, -0.0018560, 0.0019312
5: 0.0097651, 0.0109087, 0.0097651, 0.0109087, -0.0007030, 0.0007315
6: 0.0053643, 0.0097284, 0.0053643, 0.0097284, -0.0027913, 0.0026828
7: 0.9818129, 0.9848667, 0.9818129, 0.9848667, -0.0019532, 0.0018773
8: -0.0060636, -0.0027895, -0.0060636, -0.0027895, -0.0020942, 0.0020127
9: -0.0031570, -0.0009942, -0.0031570, -0.0009942, -0.0013295, 0.0013833

### Unstable ReLU Count (Linear/Conv2D Layers)
- layer_idx=0, type=LayerType.Linear, total=256, inp1_unstable=10, inp2_unstable=10, delta_unstable=249
- layer_idx=2, type=LayerType.Linear, total=256, inp1_unstable=10, inp2_unstable=10, delta_unstable=256
- layer_idx=4, type=LayerType.Linear, total=256, inp1_unstable=0, inp2_unstable=0, delta_unstable=256
- layer_idx=6, type=LayerType.Linear, total=256, inp1_unstable=0, inp2_unstable=0, delta_unstable=256
- layer_idx=8, type=LayerType.Linear, total=10, inp1_unstable=0, inp2_unstable=0, delta_unstable=10

Time for backsubstitution: 1.75 seconds

### RS candidates at layer 1
type: RSZ, layer: 1, pos: 140
type: RSZ, layer: 1, pos: 213
type: RSZ, layer: 1, pos: 219

Time for candidate selection: 0.20 seconds

### Candidate
type: RSZ, layer: 1, pos: 140

### Relational analysis RSZ of RS_RSZ1_RSZ1_RSZ1_RSZ1_RSZ2_RSZ2_RSZ2_RSZ1
Optimization infeasible because this subproblem isn't reachable.

### Candidate
type: RSZ, layer: 1, pos: 213

### Relational analysis RSZ of RS_RSZ1_RSZ1_RSZ1_RSZ1_RSZ2_RSZ2_RSZ2_RSZ1

#### Relational analysis RSZ result of RS_RSZ1_RSZ1_RSZ1_RSZ1_RSZ2_RSZ2_RSZ2_RSZ1
Status: Status.VERIFIED
Output dim: 7, lower bound: -0.0007971, upper bound: 0.0007899
time: 0.69 seconds

### Relational analysis RSZ of RS_RSZ1_RSZ1_RSZ1_RSZ1_RSZ2_RSZ2_RSZ2_RSZ2

#### Relational analysis RSZ result of RS_RSZ1_RSZ1_RSZ1_RSZ1_RSZ2_RSZ2_RSZ2_RSZ2
Status: Status.VERIFIED
Output dim: 7, lower bound: -0.0007971, upper bound: 0.0007899
time: 0.73 seconds

## BFS RS instance: RS_RSZ1_RSZ1_RSZ1_RSZ2_RSZ1_RSZ1_RSZ1

### Backsubstitution after applying RS history:
0: -0.0027939, -0.0016242, -0.0027939, -0.0016242, -0.0007376, 0.0007283
1: -0.0114005, -0.0084323, -0.0114005, -0.0084323, -0.0018718, 0.0018482
2: 0.0279571, 0.0297986, 0.0279571, 0.0297986, -0.0011613, 0.0011466
3: 0.0039210, 0.0073596, 0.0039210, 0.0073596, -0.0021410, 0.0021684
4: -0.0104893, -0.0074701, -0.0104893, -0.0074701, -0.0019039, 0.0018799
5: 0.0097651, 0.0109087, 0.0097651, 0.0109087, -0.0007212, 0.0007121
6: 0.0053643, 0.0097284, 0.0053643, 0.0097284, -0.0027172, 0.0027520
7: 0.9818129, 0.9848667, 0.9818129, 0.9848667, -0.0019014, 0.0019257
8: -0.0060636, -0.0027895, -0.0060636, -0.0027895, -0.0020386, 0.0020647
9: -0.0031570, -0.0009942, -0.0031570, -0.0009942, -0.0013638, 0.0013466

### Unstable ReLU Count (Linear/Conv2D Layers)
- layer_idx=0, type=LayerType.Linear, total=256, inp1_unstable=10, inp2_unstable=10, delta_unstable=249
- layer_idx=2, type=LayerType.Linear, total=256, inp1_unstable=10, inp2_unstable=10, delta_unstable=256
- layer_idx=4, type=LayerType.Linear, total=256, inp1_unstable=0, inp2_unstable=0, delta_unstable=256
- layer_idx=6, type=LayerType.Linear, total=256, inp1_unstable=0, inp2_unstable=0, delta_unstable=256
- layer_idx=8, type=LayerType.Linear, total=10, inp1_unstable=0, inp2_unstable=0, delta_unstable=10

Time for backsubstitution: 1.69 seconds

### RS candidates at layer 1
type: RSZ, layer: 1, pos: 140
type: RSZ, layer: 1, pos: 213
type: RSZ, layer: 1, pos: 219

Time for candidate selection: 0.18 seconds

### Candidate
type: RSZ, layer: 1, pos: 140

### Relational analysis RSZ of RS_RSZ1_RSZ1_RSZ1_RSZ2_RSZ1_RSZ1_RSZ1_RSZ1
Optimization infeasible because this subproblem isn't reachable.

### Candidate
type: RSZ, layer: 1, pos: 213

### Relational analysis RSZ of RS_RSZ1_RSZ1_RSZ1_RSZ2_RSZ1_RSZ1_RSZ1_RSZ1

#### Relational analysis RSZ result of RS_RSZ1_RSZ1_RSZ1_RSZ2_RSZ1_RSZ1_RSZ1_RSZ1
Status: Status.UNKNOWN
Output dim: 7, lower bound: -0.0008026, upper bound: 0.0007811
time: 0.68 seconds

### Relational analysis RSZ of RS_RSZ1_RSZ1_RSZ1_RSZ2_RSZ1_RSZ1_RSZ1_RSZ2

#### Relational analysis RSZ result of RS_RSZ1_RSZ1_RSZ1_RSZ2_RSZ1_RSZ1_RSZ1_RSZ2
Status: Status.UNKNOWN
Output dim: 7, lower bound: -0.0008026, upper bound: 0.0007811
time: 0.71 seconds

## BFS RS instance: RS_RSZ1_RSZ1_RSZ1_RSZ2_RSZ1_RSZ1_RSZ2

### Backsubstitution after applying RS history:
0: -0.0027939, -0.0016242, -0.0027939, -0.0016242, -0.0007410, 0.0007270
1: -0.0114005, -0.0084323, -0.0114005, -0.0084323, -0.0018803, 0.0018449
2: 0.0279571, 0.0297986, 0.0279571, 0.0297986, -0.0011665, 0.0011446
3: 0.0039210, 0.0073596, 0.0039210, 0.0073596, -0.0021372, 0.0021782
4: -0.0104893, -0.0074701, -0.0104893, -0.0074701, -0.0019125, 0.0018765
5: 0.0097651, 0.0109087, 0.0097651, 0.0109087, -0.0007244, 0.0007108
6: 0.0053643, 0.0097284, 0.0053643, 0.0097284, -0.0027124, 0.0027644
7: 0.9818129, 0.9848667, 0.9818129, 0.9848667, -0.0018980, 0.0019344
8: -0.0060636, -0.0027895, -0.0060636, -0.0027895, -0.0020349, 0.0020740
9: -0.0031570, -0.0009942, -0.0031570, -0.0009942, -0.0013700, 0.0013442

### Unstable ReLU Count (Linear/Conv2D Layers)
- layer_idx=0, type=LayerType.Linear, total=256, inp1_unstable=10, inp2_unstable=10, delta_unstable=249
- layer_idx=2, type=LayerType.Linear, total=256, inp1_unstable=10, inp2_unstable=10, delta_unstable=256
- layer_idx=4, type=LayerType.Linear, total=256, inp1_unstable=0, inp2_unstable=0, delta_unstable=256
- layer_idx=6, type=LayerType.Linear, total=256, inp1_unstable=0, inp2_unstable=0, delta_unstable=256
- layer_idx=8, type=LayerType.Linear, total=10, inp1_unstable=0, inp2_unstable=0, delta_unstable=10

Time for backsubstitution: 1.62 seconds

### RS candidates at layer 1
type: RSZ, layer: 1, pos: 140
type: RSZ, layer: 1, pos: 213
type: RSZ, layer: 1, pos: 219

Time for candidate selection: 0.18 seconds

### Candidate
type: RSZ, layer: 1, pos: 140

### Relational analysis RSZ of RS_RSZ1_RSZ1_RSZ1_RSZ2_RSZ1_RSZ1_RSZ2_RSZ1
Optimization infeasible because this subproblem isn't reachable.

### Candidate
type: RSZ, layer: 1, pos: 213

### Relational analysis RSZ of RS_RSZ1_RSZ1_RSZ1_RSZ2_RSZ1_RSZ1_RSZ2_RSZ1

#### Relational analysis RSZ result of RS_RSZ1_RSZ1_RSZ1_RSZ2_RSZ1_RSZ1_RSZ2_RSZ1
Status: Status.VERIFIED
Output dim: 7, lower bound: -0.0007944, upper bound: 0.0007889
time: 0.66 seconds

### Relational analysis RSZ of RS_RSZ1_RSZ1_RSZ1_RSZ2_RSZ1_RSZ1_RSZ2_RSZ2

#### Relational analysis RSZ result of RS_RSZ1_RSZ1_RSZ1_RSZ2_RSZ1_RSZ1_RSZ2_RSZ2
Status: Status.VERIFIED
Output dim: 7, lower bound: -0.0007944, upper bound: 0.0007889
time: 0.66 seconds

## BFS RS instance: RS_RSZ1_RSZ1_RSZ1_RSZ2_RSZ1_RSZ2_RSZ1

### Backsubstitution after applying RS history:
0: -0.0027939, -0.0016242, -0.0027939, -0.0016242, -0.0007378, 0.0007281
1: -0.0114005, -0.0084323, -0.0114005, -0.0084323, -0.0018723, 0.0018477
2: 0.0279571, 0.0297986, 0.0279571, 0.0297986, -0.0011616, 0.0011463
3: 0.0039210, 0.0073596, 0.0039210, 0.0073596, -0.0021405, 0.0021690
4: -0.0104893, -0.0074701, -0.0104893, -0.0074701, -0.0019045, 0.0018794
5: 0.0097651, 0.0109087, 0.0097651, 0.0109087, -0.0007214, 0.0007119
6: 0.0053643, 0.0097284, 0.0053643, 0.0097284, -0.0027166, 0.0027527
7: 0.9818129, 0.9848667, 0.9818129, 0.9848667, -0.0019009, 0.0019262
8: -0.0060636, -0.0027895, -0.0060636, -0.0027895, -0.0020381, 0.0020652
9: -0.0031570, -0.0009942, -0.0031570, -0.0009942, -0.0013642, 0.0013463

### Unstable ReLU Count (Linear/Conv2D Layers)
- layer_idx=0, type=LayerType.Linear, total=256, inp1_unstable=10, inp2_unstable=10, delta_unstable=249
- layer_idx=2, type=LayerType.Linear, total=256, inp1_unstable=10, inp2_unstable=10, delta_unstable=256
- layer_idx=4, type=LayerType.Linear, total=256, inp1_unstable=0, inp2_unstable=0, delta_unstable=256
- layer_idx=6, type=LayerType.Linear, total=256, inp1_unstable=0, inp2_unstable=0, delta_unstable=256
- layer_idx=8, type=LayerType.Linear, total=10, inp1_unstable=0, inp2_unstable=0, delta_unstable=10

Time for backsubstitution: 1.59 seconds

### RS candidates at layer 1
type: RSZ, layer: 1, pos: 140
type: RSZ, layer: 1, pos: 213
type: RSZ, layer: 1, pos: 219

Time for candidate selection: 0.16 seconds

### Candidate
type: RSZ, layer: 1, pos: 140

### Relational analysis RSZ of RS_RSZ1_RSZ1_RSZ1_RSZ2_RSZ1_RSZ2_RSZ1_RSZ1
Optimization infeasible because this subproblem isn't reachable.

### Candidate
type: RSZ, layer: 1, pos: 213

### Relational analysis RSZ of RS_RSZ1_RSZ1_RSZ1_RSZ2_RSZ1_RSZ2_RSZ1_RSZ1

#### Relational analysis RSZ result of RS_RSZ1_RSZ1_RSZ1_RSZ2_RSZ1_RSZ2_RSZ1_RSZ1
Status: Status.UNKNOWN
Output dim: 7, lower bound: -0.0008040, upper bound: 0.0007799
time: 0.73 seconds

### Relational analysis RSZ of RS_RSZ1_RSZ1_RSZ1_RSZ2_RSZ1_RSZ2_RSZ1_RSZ2

#### Relational analysis RSZ result of RS_RSZ1_RSZ1_RSZ1_RSZ2_RSZ1_RSZ2_RSZ1_RSZ2
Status: Status.UNKNOWN
Output dim: 7, lower bound: -0.0008040, upper bound: 0.0007799
time: 0.74 seconds

## BFS RS instance: RS_RSZ1_RSZ1_RSZ1_RSZ2_RSZ1_RSZ2_RSZ2

### Backsubstitution after applying RS history:
0: -0.0027939, -0.0016242, -0.0027939, -0.0016242, -0.0007410, 0.0007268
1: -0.0114005, -0.0084323, -0.0114005, -0.0084323, -0.0018805, 0.0018444
2: 0.0279571, 0.0297986, 0.0279571, 0.0297986, -0.0011667, 0.0011443
3: 0.0039210, 0.0073596, 0.0039210, 0.0073596, -0.0021367, 0.0021785
4: -0.0104893, -0.0074701, -0.0104893, -0.0074701, -0.0019128, 0.0018761
5: 0.0097651, 0.0109087, 0.0097651, 0.0109087, -0.0007245, 0.0007106
6: 0.0053643, 0.0097284, 0.0053643, 0.0097284, -0.0027117, 0.0027647
7: 0.9818129, 0.9848667, 0.9818129, 0.9848667, -0.0018975, 0.0019346
8: -0.0060636, -0.0027895, -0.0060636, -0.0027895, -0.0020344, 0.0020742
9: -0.0031570, -0.0009942, -0.0031570, -0.0009942, -0.0013701, 0.0013439

### Unstable ReLU Count (Linear/Conv2D Layers)
- layer_idx=0, type=LayerType.Linear, total=256, inp1_unstable=10, inp2_unstable=10, delta_unstable=249
- layer_idx=2, type=LayerType.Linear, total=256, inp1_unstable=10, inp2_unstable=10, delta_unstable=256
- layer_idx=4, type=LayerType.Linear, total=256, inp1_unstable=0, inp2_unstable=0, delta_unstable=256
- layer_idx=6, type=LayerType.Linear, total=256, inp1_unstable=0, inp2_unstable=0, delta_unstable=256
- layer_idx=8, type=LayerType.Linear, total=10, inp1_unstable=0, inp2_unstable=0, delta_unstable=10

Time for backsubstitution: 1.73 seconds

### RS candidates at layer 1
type: RSZ, layer: 1, pos: 140
type: RSZ, layer: 1, pos: 213
type: RSZ, layer: 1, pos: 219

Time for candidate selection: 0.18 seconds

### Candidate
type: RSZ, layer: 1, pos: 140

### Relational analysis RSZ of RS_RSZ1_RSZ1_RSZ1_RSZ2_RSZ1_RSZ2_RSZ2_RSZ1
Optimization infeasible because this subproblem isn't reachable.

### Candidate
type: RSZ, layer: 1, pos: 213

### Relational analysis RSZ of RS_RSZ1_RSZ1_RSZ1_RSZ2_RSZ1_RSZ2_RSZ2_RSZ1

#### Relational analysis RSZ result of RS_RSZ1_RSZ1_RSZ1_RSZ2_RSZ1_RSZ2_RSZ2_RSZ1
Status: Status.VERIFIED
Output dim: 7, lower bound: -0.0007961, upper bound: 0.0007884
time: 0.68 seconds

### Relational analysis RSZ of RS_RSZ1_RSZ1_RSZ1_RSZ2_RSZ1_RSZ2_RSZ2_RSZ2

#### Relational analysis RSZ result of RS_RSZ1_RSZ1_RSZ1_RSZ2_RSZ1_RSZ2_RSZ2_RSZ2
Status: Status.VERIFIED
Output dim: 7, lower bound: -0.0007961, upper bound: 0.0007884
time: 0.67 seconds

## BFS RS instance: RS_RSZ1_RSZ1_RSZ1_RSZ2_RSZ2_RSZ1_RSZ1

### Backsubstitution after applying RS history:
0: -0.0027939, -0.0016242, -0.0027939, -0.0016242, -0.0007228, 0.0007434
1: -0.0114005, -0.0084323, -0.0114005, -0.0084323, -0.0018342, 0.0018865
2: 0.0279571, 0.0297986, 0.0279571, 0.0297986, -0.0011379, 0.0011704
3: 0.0039210, 0.0073596, 0.0039210, 0.0073596, -0.0021855, 0.0021248
4: -0.0104893, -0.0074701, -0.0104893, -0.0074701, -0.0018657, 0.0019189
5: 0.0097651, 0.0109087, 0.0097651, 0.0109087, -0.0007067, 0.0007268
6: 0.0053643, 0.0097284, 0.0053643, 0.0097284, -0.0027736, 0.0026967
7: 0.9818129, 0.9848667, 0.9818129, 0.9848667, -0.0019409, 0.0018870
8: -0.0060636, -0.0027895, -0.0060636, -0.0027895, -0.0020809, 0.0020232
9: -0.0031570, -0.0009942, -0.0031570, -0.0009942, -0.0013364, 0.0013746

### Unstable ReLU Count (Linear/Conv2D Layers)
- layer_idx=0, type=LayerType.Linear, total=256, inp1_unstable=10, inp2_unstable=10, delta_unstable=249
- layer_idx=2, type=LayerType.Linear, total=256, inp1_unstable=10, inp2_unstable=10, delta_unstable=256
- layer_idx=4, type=LayerType.Linear, total=256, inp1_unstable=0, inp2_unstable=0, delta_unstable=256
- layer_idx=6, type=LayerType.Linear, total=256, inp1_unstable=0, inp2_unstable=0, delta_unstable=256
- layer_idx=8, type=LayerType.Linear, total=10, inp1_unstable=0, inp2_unstable=0, delta_unstable=10

Time for backsubstitution: 1.72 seconds

### RS candidates at layer 1
type: RSZ, layer: 1, pos: 140
type: RSZ, layer: 1, pos: 213
type: RSZ, layer: 1, pos: 219

Time for candidate selection: 0.19 seconds

### Candidate
type: RSZ, layer: 1, pos: 140

### Relational analysis RSZ of RS_RSZ1_RSZ1_RSZ1_RSZ2_RSZ2_RSZ1_RSZ1_RSZ1
Optimization infeasible because this subproblem isn't reachable.

### Candidate
type: RSZ, layer: 1, pos: 213

### Relational analysis RSZ of RS_RSZ1_RSZ1_RSZ1_RSZ2_RSZ2_RSZ1_RSZ1_RSZ1

#### Relational analysis RSZ result of RS_RSZ1_RSZ1_RSZ1_RSZ2_RSZ2_RSZ1_RSZ1_RSZ1
Status: Status.UNKNOWN
Output dim: 7, lower bound: -0.0008041, upper bound: 0.0007781
time: 0.66 seconds

### Relational analysis RSZ of RS_RSZ1_RSZ1_RSZ1_RSZ2_RSZ2_RSZ1_RSZ1_RSZ2

#### Relational analysis RSZ result of RS_RSZ1_RSZ1_RSZ1_RSZ2_RSZ2_RSZ1_RSZ1_RSZ2
Status: Status.UNKNOWN
Output dim: 7, lower bound: -0.0008041, upper bound: 0.0007781
time: 0.67 seconds

## BFS RS instance: RS_RSZ1_RSZ1_RSZ1_RSZ2_RSZ2_RSZ1_RSZ2

### Backsubstitution after applying RS history:
0: -0.0027939, -0.0016242, -0.0027939, -0.0016242, -0.0007264, 0.0007421
1: -0.0114005, -0.0084323, -0.0114005, -0.0084323, -0.0018434, 0.0018833
2: 0.0279571, 0.0297986, 0.0279571, 0.0297986, -0.0011437, 0.0011684
3: 0.0039210, 0.0073596, 0.0039210, 0.0073596, -0.0021817, 0.0021355
4: -0.0104893, -0.0074701, -0.0104893, -0.0074701, -0.0018751, 0.0019156
5: 0.0097651, 0.0109087, 0.0097651, 0.0109087, -0.0007102, 0.0007256
6: 0.0053643, 0.0097284, 0.0053643, 0.0097284, -0.0027689, 0.0027103
7: 0.9818129, 0.9848667, 0.9818129, 0.9848667, -0.0019375, 0.0018965
8: -0.0060636, -0.0027895, -0.0060636, -0.0027895, -0.0020773, 0.0020334
9: -0.0031570, -0.0009942, -0.0031570, -0.0009942, -0.0013432, 0.0013722

### Unstable ReLU Count (Linear/Conv2D Layers)
- layer_idx=0, type=LayerType.Linear, total=256, inp1_unstable=10, inp2_unstable=10, delta_unstable=249
- layer_idx=2, type=LayerType.Linear, total=256, inp1_unstable=10, inp2_unstable=10, delta_unstable=256
- layer_idx=4, type=LayerType.Linear, total=256, inp1_unstable=0, inp2_unstable=0, delta_unstable=256
- layer_idx=6, type=LayerType.Linear, total=256, inp1_unstable=0, inp2_unstable=0, delta_unstable=256
- layer_idx=8, type=LayerType.Linear, total=10, inp1_unstable=0, inp2_unstable=0, delta_unstable=10

Time for backsubstitution: 1.61 seconds

### RS candidates at layer 1
type: RSZ, layer: 1, pos: 140
type: RSZ, layer: 1, pos: 213
type: RSZ, layer: 1, pos: 219

Time for candidate selection: 0.18 seconds

### Candidate
type: RSZ, layer: 1, pos: 140

### Relational analysis RSZ of RS_RSZ1_RSZ1_RSZ1_RSZ2_RSZ2_RSZ1_RSZ2_RSZ1
Optimization infeasible because this subproblem isn't reachable.

### Candidate
type: RSZ, layer: 1, pos: 213

### Relational analysis RSZ of RS_RSZ1_RSZ1_RSZ1_RSZ2_RSZ2_RSZ1_RSZ2_RSZ1

#### Relational analysis RSZ result of RS_RSZ1_RSZ1_RSZ1_RSZ2_RSZ2_RSZ1_RSZ2_RSZ1
Status: Status.VERIFIED
Output dim: 7, lower bound: -0.0007952, upper bound: 0.0007851
time: 0.75 seconds

### Relational analysis RSZ of RS_RSZ1_RSZ1_RSZ1_RSZ2_RSZ2_RSZ1_RSZ2_RSZ2

#### Relational analysis RSZ result of RS_RSZ1_RSZ1_RSZ1_RSZ2_RSZ2_RSZ1_RSZ2_RSZ2
Status: Status.VERIFIED
Output dim: 7, lower bound: -0.0007952, upper bound: 0.0007851
time: 0.71 seconds

## BFS RS instance: RS_RSZ1_RSZ1_RSZ1_RSZ2_RSZ2_RSZ2_RSZ1

### Backsubstitution after applying RS history:
0: -0.0027939, -0.0016242, -0.0027939, -0.0016242, -0.0007230, 0.0007433
1: -0.0114005, -0.0084323, -0.0114005, -0.0084323, -0.0018348, 0.0018862
2: 0.0279571, 0.0297986, 0.0279571, 0.0297986, -0.0011383, 0.0011702
3: 0.0039210, 0.0073596, 0.0039210, 0.0073596, -0.0021851, 0.0021255
4: -0.0104893, -0.0074701, -0.0104893, -0.0074701, -0.0018663, 0.0019186
5: 0.0097651, 0.0109087, 0.0097651, 0.0109087, -0.0007069, 0.0007267
6: 0.0053643, 0.0097284, 0.0053643, 0.0097284, -0.0027732, 0.0026975
7: 0.9818129, 0.9848667, 0.9818129, 0.9848667, -0.0019406, 0.0018876
8: -0.0060636, -0.0027895, -0.0060636, -0.0027895, -0.0020806, 0.0020238
9: -0.0031570, -0.0009942, -0.0031570, -0.0009942, -0.0013368, 0.0013743

### Unstable ReLU Count (Linear/Conv2D Layers)
- layer_idx=0, type=LayerType.Linear, total=256, inp1_unstable=10, inp2_unstable=10, delta_unstable=249
- layer_idx=2, type=LayerType.Linear, total=256, inp1_unstable=10, inp2_unstable=10, delta_unstable=256
- layer_idx=4, type=LayerType.Linear, total=256, inp1_unstable=0, inp2_unstable=0, delta_unstable=256
- layer_idx=6, type=LayerType.Linear, total=256, inp1_unstable=0, inp2_unstable=0, delta_unstable=256
- layer_idx=8, type=LayerType.Linear, total=10, inp1_unstable=0, inp2_unstable=0, delta_unstable=10

Time for backsubstitution: 1.55 seconds

### RS candidates at layer 1
type: RSZ, layer: 1, pos: 140
type: RSZ, layer: 1, pos: 213
type: RSZ, layer: 1, pos: 219

Time for candidate selection: 0.16 seconds

### Candidate
type: RSZ, layer: 1, pos: 140

### Relational analysis RSZ of RS_RSZ1_RSZ1_RSZ1_RSZ2_RSZ2_RSZ2_RSZ1_RSZ1
Optimization infeasible because this subproblem isn't reachable.

### Candidate
type: RSZ, layer: 1, pos: 213

### Relational analysis RSZ of RS_RSZ1_RSZ1_RSZ1_RSZ2_RSZ2_RSZ2_RSZ1_RSZ1

#### Relational analysis RSZ result of RS_RSZ1_RSZ1_RSZ1_RSZ2_RSZ2_RSZ2_RSZ1_RSZ1
Status: Status.UNKNOWN
Output dim: 7, lower bound: -0.0008052, upper bound: 0.0007771
time: 0.67 seconds

### Relational analysis RSZ of RS_RSZ1_RSZ1_RSZ1_RSZ2_RSZ2_RSZ2_RSZ1_RSZ2

#### Relational analysis RSZ result of RS_RSZ1_RSZ1_RSZ1_RSZ2_RSZ2_RSZ2_RSZ1_RSZ2
Status: Status.UNKNOWN
Output dim: 7, lower bound: -0.0008052, upper bound: 0.0007771
time: 0.66 seconds

## BFS RS instance: RS_RSZ1_RSZ1_RSZ1_RSZ2_RSZ2_RSZ2_RSZ2

### Backsubstitution after applying RS history:
0: -0.0027939, -0.0016242, -0.0027939, -0.0016242, -0.0007266, 0.0007420
1: -0.0114005, -0.0084323, -0.0114005, -0.0084323, -0.0018438, 0.0018828
2: 0.0279571, 0.0297986, 0.0279571, 0.0297986, -0.0011439, 0.0011681
3: 0.0039210, 0.0073596, 0.0039210, 0.0073596, -0.0021812, 0.0021360
4: -0.0104893, -0.0074701, -0.0104893, -0.0074701, -0.0018755, 0.0019152
5: 0.0097651, 0.0109087, 0.0097651, 0.0109087, -0.0007104, 0.0007254
6: 0.0053643, 0.0097284, 0.0053643, 0.0097284, -0.0027682, 0.0027109
7: 0.9818129, 0.9848667, 0.9818129, 0.9848667, -0.0019371, 0.0018969
8: -0.0060636, -0.0027895, -0.0060636, -0.0027895, -0.0020768, 0.0020338
9: -0.0031570, -0.0009942, -0.0031570, -0.0009942, -0.0013435, 0.0013719

### Unstable ReLU Count (Linear/Conv2D Layers)
- layer_idx=0, type=LayerType.Linear, total=256, inp1_unstable=10, inp2_unstable=10, delta_unstable=249
- layer_idx=2, type=LayerType.Linear, total=256, inp1_unstable=10, inp2_unstable=10, delta_unstable=256
- layer_idx=4, type=LayerType.Linear, total=256, inp1_unstable=0, inp2_unstable=0, delta_unstable=256
- layer_idx=6, type=LayerType.Linear, total=256, inp1_unstable=0, inp2_unstable=0, delta_unstable=256
- layer_idx=8, type=LayerType.Linear, total=10, inp1_unstable=0, inp2_unstable=0, delta_unstable=10

Time for backsubstitution: 1.58 seconds

### RS candidates at layer 1
type: RSZ, layer: 1, pos: 140
type: RSZ, layer: 1, pos: 213
type: RSZ, layer: 1, pos: 219

Time for candidate selection: 0.16 seconds

### Candidate
type: RSZ, layer: 1, pos: 140

### Relational analysis RSZ of RS_RSZ1_RSZ1_RSZ1_RSZ2_RSZ2_RSZ2_RSZ2_RSZ1
Optimization infeasible because this subproblem isn't reachable.

### Candidate
type: RSZ, layer: 1, pos: 213

### Relational analysis RSZ of RS_RSZ1_RSZ1_RSZ1_RSZ2_RSZ2_RSZ2_RSZ2_RSZ1

#### Relational analysis RSZ result of RS_RSZ1_RSZ1_RSZ1_RSZ2_RSZ2_RSZ2_RSZ2_RSZ1
Status: Status.VERIFIED
Output dim: 7, lower bound: -0.0007971, upper bound: 0.0007846
time: 0.69 seconds

### Relational analysis RSZ of RS_RSZ1_RSZ1_RSZ1_RSZ2_RSZ2_RSZ2_RSZ2_RSZ2

#### Relational analysis RSZ result of RS_RSZ1_RSZ1_RSZ1_RSZ2_RSZ2_RSZ2_RSZ2_RSZ2
Status: Status.VERIFIED
Output dim: 7, lower bound: -0.0007971, upper bound: 0.0007846
time: 0.67 seconds

## BFS RS instance: RS_RSZ1_RSZ1_RSZ2_RSZ1_RSZ1_RSZ1_RSZ1

### Backsubstitution after applying RS history:
0: -0.0027939, -0.0016242, -0.0027939, -0.0016242, -0.0007337, 0.0007338
1: -0.0114005, -0.0084323, -0.0114005, -0.0084323, -0.0018619, 0.0018620
2: 0.0279571, 0.0297986, 0.0279571, 0.0297986, -0.0011551, 0.0011552
3: 0.0039210, 0.0073596, 0.0039210, 0.0073596, -0.0021571, 0.0021569
4: -0.0104893, -0.0074701, -0.0104893, -0.0074701, -0.0018939, 0.0018940
5: 0.0097651, 0.0109087, 0.0097651, 0.0109087, -0.0007173, 0.0007174
6: 0.0053643, 0.0097284, 0.0053643, 0.0097284, -0.0027376, 0.0027374
7: 0.9818129, 0.9848667, 0.9818129, 0.9848667, -0.0019157, 0.0019155
8: -0.0060636, -0.0027895, -0.0060636, -0.0027895, -0.0020539, 0.0020537
9: -0.0031570, -0.0009942, -0.0031570, -0.0009942, -0.0013566, 0.0013567

### Unstable ReLU Count (Linear/Conv2D Layers)
- layer_idx=0, type=LayerType.Linear, total=256, inp1_unstable=10, inp2_unstable=10, delta_unstable=249
- layer_idx=2, type=LayerType.Linear, total=256, inp1_unstable=10, inp2_unstable=10, delta_unstable=256
- layer_idx=4, type=LayerType.Linear, total=256, inp1_unstable=0, inp2_unstable=0, delta_unstable=256
- layer_idx=6, type=LayerType.Linear, total=256, inp1_unstable=0, inp2_unstable=0, delta_unstable=256
- layer_idx=8, type=LayerType.Linear, total=10, inp1_unstable=0, inp2_unstable=0, delta_unstable=10

Time for backsubstitution: 1.56 seconds

## RS Result
status: Status.UNKNOWN
execution time: (base) + (rs) = 3.13 + 597.05 = 600.18 seconds
