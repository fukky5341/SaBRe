## Execution arguments:
Dataset: Dataset.MNIST
Network: onnx/mnist-net_256x4.onnx
Relational property: GLOBAL_ROBUSTNESS
LP Analysis: True
Epsilon: 0.0234375
Delta epsilon: 0.0078125
execution index: (2, 3, 4)
Time budget: 600 seconds
Split limit: 100
Threshold: 0.00506088


## IAR start

### Unstable ReLU Count (Linear/Conv2D Layers)
- layer_idx=0, type=LayerType.Linear, total=256, inp1_unstable=7, inp2_unstable=7, delta_unstable=256
- layer_idx=2, type=LayerType.Linear, total=256, inp1_unstable=10, inp2_unstable=10, delta_unstable=256
- layer_idx=4, type=LayerType.Linear, total=256, inp1_unstable=1, inp2_unstable=1, delta_unstable=256
- layer_idx=6, type=LayerType.Linear, total=256, inp1_unstable=0, inp2_unstable=0, delta_unstable=256
- layer_idx=8, type=LayerType.Linear, total=10, inp1_unstable=4, inp2_unstable=4, delta_unstable=10

### BASE IAR bounds
Layer (inp1_lb, inp1_ub, inp2_lb, inp2_ub, d_lb, d_ub)
0: (-0.0039095, -0.0003029, -0.0039095, -0.0003029, -0.0036066, 0.0036066)
1: (-0.0044907, 0.0040812, -0.0044907, 0.0040812, -0.0065954, 0.0065954)
2: (0.0029569, 0.0099431, 0.0029569, 0.0099431, -0.0050909, 0.0050909)
3: (-0.0045519, -0.0034923, -0.0045519, -0.0034923, -0.0010595, 0.0010595)
4: (0.0017018, 0.0078700, 0.0017018, 0.0078700, -0.0056105, 0.0056105)
5: (-0.0030341, 0.0031953, -0.0030341, 0.0031953, -0.0054272, 0.0054272)
6: (-0.0068516, -0.0033211, -0.0068516, -0.0033211, -0.0034094, 0.0034094)
7: (-0.0021570, 0.0038259, -0.0021570, 0.0038259, -0.0057421, 0.0057421)
8: (-0.0008533, 0.0000782, -0.0008533, 0.0000782, -0.0009315, 0.0009315)
9: (0.9973557, 1.0121032, 0.9973557, 1.0121032, -0.0112477, 0.0112477)

## BASE Relational Analysis

## BASE Result
execution time: IAR + RelationalAnalysis = 1.45 + 1.57 = 3.02 seconds
status: Status.UNKNOWN
relational distance
Output dim: 9, lower bound: -0.0064393, upper bound: 0.0064394

# Relational Split (RS) starts

## BFS RS instance: RS

Time for backsubstitution: 0.01 seconds

### RS candidates at layer 1
type: RSZ, layer: 1, pos: 20
type: RSZ, layer: 1, pos: 29
type: RSZ, layer: 1, pos: 104
type: RSZ, layer: 1, pos: 114
type: RSZ, layer: 1, pos: 140
type: RSZ, layer: 1, pos: 148
type: RSZ, layer: 1, pos: 219

Time for candidate selection: 0.13 seconds

### Candidate
type: RSZ, layer: 1, pos: 20

### Relational analysis RSZ of RS_RSZ1

#### Relational analysis RSZ result of RS_RSZ1
Status: Status.UNKNOWN
Output dim: 9, lower bound: -0.0063320, upper bound: 0.0062862
time: 0.74 seconds

### Relational analysis RSZ of RS_RSZ2

#### Relational analysis RSZ result of RS_RSZ2
Status: Status.UNKNOWN
Output dim: 9, lower bound: -0.0062862, upper bound: 0.0063320
time: 0.72 seconds

## Summary of splitting (split count: 0)
- Time for RS candidates: 1.60 seconds
RS_RSZ1, status: Status.UNKNOWN, split count: 1, time: 1.60
Output dim: 9, lower bound: -0.0063320, upper bound: 0.0062862
RS_RSZ2, status: Status.UNKNOWN, split count: 1, time: 1.60
Output dim: 9, lower bound: -0.0062862, upper bound: 0.0063320

## BFS RS instance: RS_RSZ1

### Backsubstitution after applying RS history:
0: -0.0039095, -0.0003029, -0.0039095, -0.0003029, -0.0036066, 0.0036066
1: -0.0044907, 0.0040812, -0.0044907, 0.0040812, -0.0065515, 0.0065604
2: 0.0029569, 0.0099431, 0.0029569, 0.0099431, -0.0050505, 0.0050584
3: -0.0045519, -0.0034923, -0.0045519, -0.0034923, -0.0010595, 0.0010595
4: 0.0017018, 0.0078700, 0.0017018, 0.0078700, -0.0056031, 0.0056018
5: -0.0030341, 0.0031953, -0.0030341, 0.0031953, -0.0054226, 0.0054233
6: -0.0068516, -0.0033211, -0.0068516, -0.0033211, -0.0034058, 0.0034064
7: -0.0021570, 0.0038259, -0.0021570, 0.0038259, -0.0057173, 0.0057107
8: -0.0008533, 0.0000782, -0.0008533, 0.0000782, -0.0009315, 0.0009315
9: 0.9973557, 1.0121032, 0.9973557, 1.0121032, -0.0112072, 0.0111990

### Unstable ReLU Count (Linear/Conv2D Layers)
- layer_idx=0, type=LayerType.Linear, total=256, inp1_unstable=7, inp2_unstable=7, delta_unstable=255
- layer_idx=2, type=LayerType.Linear, total=256, inp1_unstable=10, inp2_unstable=10, delta_unstable=256
- layer_idx=4, type=LayerType.Linear, total=256, inp1_unstable=1, inp2_unstable=1, delta_unstable=256
- layer_idx=6, type=LayerType.Linear, total=256, inp1_unstable=0, inp2_unstable=0, delta_unstable=256
- layer_idx=8, type=LayerType.Linear, total=10, inp1_unstable=4, inp2_unstable=4, delta_unstable=10

Time for backsubstitution: 1.24 seconds

### RS candidates at layer 1
type: RSZ, layer: 1, pos: 29
type: RSZ, layer: 1, pos: 104
type: RSZ, layer: 1, pos: 114
type: RSZ, layer: 1, pos: 140
type: RSZ, layer: 1, pos: 148
type: RSZ, layer: 1, pos: 219

Time for candidate selection: 0.13 seconds

### Candidate
type: RSZ, layer: 1, pos: 29

### Relational analysis RSZ of RS_RSZ1_RSZ1

#### Relational analysis RSZ result of RS_RSZ1_RSZ1
Status: Status.UNKNOWN
Output dim: 9, lower bound: -0.0062978, upper bound: 0.0061468
time: 0.78 seconds

### Relational analysis RSZ of RS_RSZ1_RSZ2

#### Relational analysis RSZ result of RS_RSZ1_RSZ2
Status: Status.UNKNOWN
Output dim: 9, lower bound: -0.0062097, upper bound: 0.0062514
time: 0.75 seconds

## BFS RS instance: RS_RSZ2

### Backsubstitution after applying RS history:
0: -0.0039095, -0.0003029, -0.0039095, -0.0003029, -0.0036066, 0.0036066
1: -0.0044907, 0.0040812, -0.0044907, 0.0040812, -0.0065604, 0.0065515
2: 0.0029569, 0.0099431, 0.0029569, 0.0099431, -0.0050584, 0.0050505
3: -0.0045519, -0.0034923, -0.0045519, -0.0034923, -0.0010595, 0.0010595
4: 0.0017018, 0.0078700, 0.0017018, 0.0078700, -0.0056018, 0.0056031
5: -0.0030341, 0.0031953, -0.0030341, 0.0031953, -0.0054233, 0.0054226
6: -0.0068516, -0.0033211, -0.0068516, -0.0033211, -0.0034064, 0.0034058
7: -0.0021570, 0.0038259, -0.0021570, 0.0038259, -0.0057107, 0.0057173
8: -0.0008533, 0.0000782, -0.0008533, 0.0000782, -0.0009315, 0.0009315
9: 0.9973557, 1.0121032, 0.9973557, 1.0121032, -0.0111990, 0.0112072

### Unstable ReLU Count (Linear/Conv2D Layers)
- layer_idx=0, type=LayerType.Linear, total=256, inp1_unstable=7, inp2_unstable=7, delta_unstable=255
- layer_idx=2, type=LayerType.Linear, total=256, inp1_unstable=10, inp2_unstable=10, delta_unstable=256
- layer_idx=4, type=LayerType.Linear, total=256, inp1_unstable=1, inp2_unstable=1, delta_unstable=256
- layer_idx=6, type=LayerType.Linear, total=256, inp1_unstable=0, inp2_unstable=0, delta_unstable=256
- layer_idx=8, type=LayerType.Linear, total=10, inp1_unstable=4, inp2_unstable=4, delta_unstable=10

Time for backsubstitution: 1.23 seconds

### RS candidates at layer 1
type: RSZ, layer: 1, pos: 29
type: RSZ, layer: 1, pos: 104
type: RSZ, layer: 1, pos: 114
type: RSZ, layer: 1, pos: 140
type: RSZ, layer: 1, pos: 148
type: RSZ, layer: 1, pos: 219

Time for candidate selection: 0.13 seconds

### Candidate
type: RSZ, layer: 1, pos: 29

### Relational analysis RSZ of RS_RSZ2_RSZ1

#### Relational analysis RSZ result of RS_RSZ2_RSZ1
Status: Status.UNKNOWN
Output dim: 9, lower bound: -0.0062514, upper bound: 0.0062097
time: 0.72 seconds

### Relational analysis RSZ of RS_RSZ2_RSZ2

#### Relational analysis RSZ result of RS_RSZ2_RSZ2
Status: Status.UNKNOWN
Output dim: 9, lower bound: -0.0061468, upper bound: 0.0062979
time: 0.73 seconds

## Summary of splitting (split count: 1)
- Time for RS candidates: 2.81 seconds
RS_RSZ1_RSZ1, status: Status.UNKNOWN, split count: 2, time: 2.81
Output dim: 9, lower bound: -0.0062978, upper bound: 0.0061468
RS_RSZ1_RSZ2, status: Status.UNKNOWN, split count: 2, time: 2.81
Output dim: 9, lower bound: -0.0062097, upper bound: 0.0062514
RS_RSZ2_RSZ1, status: Status.UNKNOWN, split count: 2, time: 2.81
Output dim: 9, lower bound: -0.0062514, upper bound: 0.0062097
RS_RSZ2_RSZ2, status: Status.UNKNOWN, split count: 2, time: 2.81
Output dim: 9, lower bound: -0.0061468, upper bound: 0.0062979

## BFS RS instance: RS_RSZ1_RSZ1

### Backsubstitution after applying RS history:
0: -0.0039095, -0.0003029, -0.0039095, -0.0003029, -0.0036066, 0.0036066
1: -0.0044907, 0.0040812, -0.0044907, 0.0040812, -0.0063632, 0.0063983
2: 0.0029569, 0.0099431, 0.0029569, 0.0099431, -0.0051038, 0.0051327
3: -0.0045519, -0.0034923, -0.0045519, -0.0034923, -0.0010595, 0.0010595
4: 0.0017018, 0.0078700, 0.0017018, 0.0078700, -0.0056049, 0.0056034
5: -0.0030341, 0.0031953, -0.0030341, 0.0031953, -0.0053310, 0.0053444
6: -0.0068516, -0.0033211, -0.0068516, -0.0033211, -0.0034164, 0.0034188
7: -0.0021570, 0.0038259, -0.0021570, 0.0038259, -0.0059788, 0.0059829
8: -0.0008533, 0.0000782, -0.0008533, 0.0000782, -0.0009315, 0.0009315
9: 0.9973557, 1.0121032, 0.9973557, 1.0121032, -0.0113008, 0.0112726

### Unstable ReLU Count (Linear/Conv2D Layers)
- layer_idx=0, type=LayerType.Linear, total=256, inp1_unstable=7, inp2_unstable=7, delta_unstable=254
- layer_idx=2, type=LayerType.Linear, total=256, inp1_unstable=10, inp2_unstable=10, delta_unstable=256
- layer_idx=4, type=LayerType.Linear, total=256, inp1_unstable=1, inp2_unstable=1, delta_unstable=256
- layer_idx=6, type=LayerType.Linear, total=256, inp1_unstable=0, inp2_unstable=0, delta_unstable=256
- layer_idx=8, type=LayerType.Linear, total=10, inp1_unstable=4, inp2_unstable=4, delta_unstable=10

Time for backsubstitution: 1.24 seconds

### RS candidates at layer 1
type: RSZ, layer: 1, pos: 104
type: RSZ, layer: 1, pos: 114
type: RSZ, layer: 1, pos: 140
type: RSZ, layer: 1, pos: 148
type: RSZ, layer: 1, pos: 219

Time for candidate selection: 0.12 seconds

### Candidate
type: RSZ, layer: 1, pos: 104

### Relational analysis RSZ of RS_RSZ1_RSZ1_RSZ1

#### Relational analysis RSZ result of RS_RSZ1_RSZ1_RSZ1
Status: Status.UNKNOWN
Output dim: 9, lower bound: -0.0061678, upper bound: 0.0059912
time: 0.74 seconds

### Relational analysis RSZ of RS_RSZ1_RSZ1_RSZ2

#### Relational analysis RSZ result of RS_RSZ1_RSZ1_RSZ2
Status: Status.UNKNOWN
Output dim: 9, lower bound: -0.0061137, upper bound: 0.0060237
time: 0.74 seconds

## BFS RS instance: RS_RSZ1_RSZ2

### Backsubstitution after applying RS history:
0: -0.0039095, -0.0003029, -0.0039095, -0.0003029, -0.0036066, 0.0036066
1: -0.0044907, 0.0040812, -0.0044907, 0.0040812, -0.0063893, 0.0063715
2: 0.0029569, 0.0099431, 0.0029569, 0.0099431, -0.0051249, 0.0051121
3: -0.0045519, -0.0034923, -0.0045519, -0.0034923, -0.0010595, 0.0010595
4: 0.0017018, 0.0078700, 0.0017018, 0.0078700, -0.0056046, 0.0056036
5: -0.0030341, 0.0031953, -0.0030341, 0.0031953, -0.0053435, 0.0053318
6: -0.0068516, -0.0033211, -0.0068516, -0.0033211, -0.0034181, 0.0034170
7: -0.0021570, 0.0038259, -0.0021570, 0.0038259, -0.0059829, 0.0059722
8: -0.0008533, 0.0000782, -0.0008533, 0.0000782, -0.0009315, 0.0009315
9: 0.9973557, 1.0121032, 0.9973557, 1.0121032, -0.0112810, 0.0112926

### Unstable ReLU Count (Linear/Conv2D Layers)
- layer_idx=0, type=LayerType.Linear, total=256, inp1_unstable=7, inp2_unstable=7, delta_unstable=254
- layer_idx=2, type=LayerType.Linear, total=256, inp1_unstable=10, inp2_unstable=10, delta_unstable=256
- layer_idx=4, type=LayerType.Linear, total=256, inp1_unstable=1, inp2_unstable=1, delta_unstable=256
- layer_idx=6, type=LayerType.Linear, total=256, inp1_unstable=0, inp2_unstable=0, delta_unstable=256
- layer_idx=8, type=LayerType.Linear, total=10, inp1_unstable=4, inp2_unstable=4, delta_unstable=10

Time for backsubstitution: 1.24 seconds

### RS candidates at layer 1
type: RSZ, layer: 1, pos: 104
type: RSZ, layer: 1, pos: 114
type: RSZ, layer: 1, pos: 140
type: RSZ, layer: 1, pos: 148
type: RSZ, layer: 1, pos: 219

Time for candidate selection: 0.12 seconds

### Candidate
type: RSZ, layer: 1, pos: 104

### Relational analysis RSZ of RS_RSZ1_RSZ2_RSZ1

#### Relational analysis RSZ result of RS_RSZ1_RSZ2_RSZ1
Status: Status.UNKNOWN
Output dim: 9, lower bound: -0.0060825, upper bound: 0.0060808
time: 0.76 seconds

### Relational analysis RSZ of RS_RSZ1_RSZ2_RSZ2

#### Relational analysis RSZ result of RS_RSZ1_RSZ2_RSZ2
Status: Status.UNKNOWN
Output dim: 9, lower bound: -0.0060401, upper bound: 0.0061257
time: 0.76 seconds

## BFS RS instance: RS_RSZ2_RSZ1

### Backsubstitution after applying RS history:
0: -0.0039095, -0.0003029, -0.0039095, -0.0003029, -0.0036066, 0.0036066
1: -0.0044907, 0.0040812, -0.0044907, 0.0040812, -0.0063715, 0.0063893
2: 0.0029569, 0.0099431, 0.0029569, 0.0099431, -0.0051121, 0.0051249
3: -0.0045519, -0.0034923, -0.0045519, -0.0034923, -0.0010595, 0.0010595
4: 0.0017018, 0.0078700, 0.0017018, 0.0078700, -0.0056036, 0.0056046
5: -0.0030341, 0.0031953, -0.0030341, 0.0031953, -0.0053318, 0.0053435
6: -0.0068516, -0.0033211, -0.0068516, -0.0033211, -0.0034170, 0.0034181
7: -0.0021570, 0.0038259, -0.0021570, 0.0038259, -0.0059722, 0.0059829
8: -0.0008533, 0.0000782, -0.0008533, 0.0000782, -0.0009315, 0.0009315
9: 0.9973557, 1.0121032, 0.9973557, 1.0121032, -0.0112926, 0.0112810

### Unstable ReLU Count (Linear/Conv2D Layers)
- layer_idx=0, type=LayerType.Linear, total=256, inp1_unstable=7, inp2_unstable=7, delta_unstable=254
- layer_idx=2, type=LayerType.Linear, total=256, inp1_unstable=10, inp2_unstable=10, delta_unstable=256
- layer_idx=4, type=LayerType.Linear, total=256, inp1_unstable=1, inp2_unstable=1, delta_unstable=256
- layer_idx=6, type=LayerType.Linear, total=256, inp1_unstable=0, inp2_unstable=0, delta_unstable=256
- layer_idx=8, type=LayerType.Linear, total=10, inp1_unstable=4, inp2_unstable=4, delta_unstable=10

Time for backsubstitution: 1.24 seconds

### RS candidates at layer 1
type: RSZ, layer: 1, pos: 104
type: RSZ, layer: 1, pos: 114
type: RSZ, layer: 1, pos: 140
type: RSZ, layer: 1, pos: 148
type: RSZ, layer: 1, pos: 219

Time for candidate selection: 0.13 seconds

### Candidate
type: RSZ, layer: 1, pos: 104

### Relational analysis RSZ of RS_RSZ2_RSZ1_RSZ1

#### Relational analysis RSZ result of RS_RSZ2_RSZ1_RSZ1
Status: Status.UNKNOWN
Output dim: 9, lower bound: -0.0061257, upper bound: 0.0060401
time: 0.72 seconds

### Relational analysis RSZ of RS_RSZ2_RSZ1_RSZ2

#### Relational analysis RSZ result of RS_RSZ2_RSZ1_RSZ2
Status: Status.UNKNOWN
Output dim: 9, lower bound: -0.0060808, upper bound: 0.0060825
time: 0.74 seconds

## BFS RS instance: RS_RSZ2_RSZ2

### Backsubstitution after applying RS history:
0: -0.0039095, -0.0003029, -0.0039095, -0.0003029, -0.0036066, 0.0036066
1: -0.0044907, 0.0040812, -0.0044907, 0.0040812, -0.0063983, 0.0063632
2: 0.0029569, 0.0099431, 0.0029569, 0.0099431, -0.0051327, 0.0051038
3: -0.0045519, -0.0034923, -0.0045519, -0.0034923, -0.0010595, 0.0010595
4: 0.0017018, 0.0078700, 0.0017018, 0.0078700, -0.0056034, 0.0056049
5: -0.0030341, 0.0031953, -0.0030341, 0.0031953, -0.0053444, 0.0053310
6: -0.0068516, -0.0033211, -0.0068516, -0.0033211, -0.0034188, 0.0034164
7: -0.0021570, 0.0038259, -0.0021570, 0.0038259, -0.0059829, 0.0059788
8: -0.0008533, 0.0000782, -0.0008533, 0.0000782, -0.0009315, 0.0009315
9: 0.9973557, 1.0121032, 0.9973557, 1.0121032, -0.0112726, 0.0113008

### Unstable ReLU Count (Linear/Conv2D Layers)
- layer_idx=0, type=LayerType.Linear, total=256, inp1_unstable=7, inp2_unstable=7, delta_unstable=254
- layer_idx=2, type=LayerType.Linear, total=256, inp1_unstable=10, inp2_unstable=10, delta_unstable=256
- layer_idx=4, type=LayerType.Linear, total=256, inp1_unstable=1, inp2_unstable=1, delta_unstable=256
- layer_idx=6, type=LayerType.Linear, total=256, inp1_unstable=0, inp2_unstable=0, delta_unstable=256
- layer_idx=8, type=LayerType.Linear, total=10, inp1_unstable=4, inp2_unstable=4, delta_unstable=10

Time for backsubstitution: 1.24 seconds

### RS candidates at layer 1
type: RSZ, layer: 1, pos: 104
type: RSZ, layer: 1, pos: 114
type: RSZ, layer: 1, pos: 140
type: RSZ, layer: 1, pos: 148
type: RSZ, layer: 1, pos: 219

Time for candidate selection: 0.12 seconds

### Candidate
type: RSZ, layer: 1, pos: 104

### Relational analysis RSZ of RS_RSZ2_RSZ2_RSZ1

#### Relational analysis RSZ result of RS_RSZ2_RSZ2_RSZ1
Status: Status.UNKNOWN
Output dim: 9, lower bound: -0.0060237, upper bound: 0.0061137
time: 0.79 seconds

### Relational analysis RSZ of RS_RSZ2_RSZ2_RSZ2

#### Relational analysis RSZ result of RS_RSZ2_RSZ2_RSZ2
Status: Status.UNKNOWN
Output dim: 9, lower bound: -0.0059912, upper bound: 0.0061679
time: 0.77 seconds

## Summary of splitting (split count: 2)
- Time for RS candidates: 2.93 seconds
RS_RSZ1_RSZ1_RSZ1, status: Status.UNKNOWN, split count: 3, time: 2.93
Output dim: 9, lower bound: -0.0061678, upper bound: 0.0059912
RS_RSZ1_RSZ1_RSZ2, status: Status.UNKNOWN, split count: 3, time: 2.93
Output dim: 9, lower bound: -0.0061137, upper bound: 0.0060237
RS_RSZ1_RSZ2_RSZ1, status: Status.UNKNOWN, split count: 3, time: 2.93
Output dim: 9, lower bound: -0.0060825, upper bound: 0.0060808
RS_RSZ1_RSZ2_RSZ2, status: Status.UNKNOWN, split count: 3, time: 2.93
Output dim: 9, lower bound: -0.0060401, upper bound: 0.0061257
RS_RSZ2_RSZ1_RSZ1, status: Status.UNKNOWN, split count: 3, time: 2.93
Output dim: 9, lower bound: -0.0061257, upper bound: 0.0060401
RS_RSZ2_RSZ1_RSZ2, status: Status.UNKNOWN, split count: 3, time: 2.93
Output dim: 9, lower bound: -0.0060808, upper bound: 0.0060825
RS_RSZ2_RSZ2_RSZ1, status: Status.UNKNOWN, split count: 3, time: 2.93
Output dim: 9, lower bound: -0.0060237, upper bound: 0.0061137
RS_RSZ2_RSZ2_RSZ2, status: Status.UNKNOWN, split count: 3, time: 2.93
Output dim: 9, lower bound: -0.0059912, upper bound: 0.0061679

## BFS RS instance: RS_RSZ1_RSZ1_RSZ1

### Backsubstitution after applying RS history:
0: -0.0039095, -0.0003029, -0.0039095, -0.0003029, -0.0036066, 0.0036066
1: -0.0044907, 0.0040812, -0.0044907, 0.0040812, -0.0062172, 0.0062852
2: 0.0029569, 0.0099431, 0.0029569, 0.0099431, -0.0050400, 0.0050860
3: -0.0045519, -0.0034923, -0.0045519, -0.0034923, -0.0010595, 0.0010595
4: 0.0017018, 0.0078700, 0.0017018, 0.0078700, -0.0055758, 0.0055613
5: -0.0030341, 0.0031953, -0.0030341, 0.0031953, -0.0052570, 0.0052873
6: -0.0068516, -0.0033211, -0.0068516, -0.0033211, -0.0034071, 0.0034127
7: -0.0021570, 0.0038259, -0.0021570, 0.0038259, -0.0059669, 0.0059727
8: -0.0008533, 0.0000782, -0.0008533, 0.0000782, -0.0009315, 0.0009315
9: 0.9973557, 1.0121032, 0.9973557, 1.0121032, -0.0111642, 0.0110841

### Unstable ReLU Count (Linear/Conv2D Layers)
- layer_idx=0, type=LayerType.Linear, total=256, inp1_unstable=7, inp2_unstable=7, delta_unstable=253
- layer_idx=2, type=LayerType.Linear, total=256, inp1_unstable=10, inp2_unstable=10, delta_unstable=256
- layer_idx=4, type=LayerType.Linear, total=256, inp1_unstable=1, inp2_unstable=1, delta_unstable=256
- layer_idx=6, type=LayerType.Linear, total=256, inp1_unstable=0, inp2_unstable=0, delta_unstable=256
- layer_idx=8, type=LayerType.Linear, total=10, inp1_unstable=4, inp2_unstable=4, delta_unstable=10

Time for backsubstitution: 1.24 seconds

### RS candidates at layer 1
type: RSZ, layer: 1, pos: 114
type: RSZ, layer: 1, pos: 140
type: RSZ, layer: 1, pos: 148
type: RSZ, layer: 1, pos: 219

Time for candidate selection: 0.13 seconds

### Candidate
type: RSZ, layer: 1, pos: 114

### Relational analysis RSZ of RS_RSZ1_RSZ1_RSZ1_RSZ1

#### Relational analysis RSZ result of RS_RSZ1_RSZ1_RSZ1_RSZ1
Status: Status.UNKNOWN
Output dim: 9, lower bound: -0.0059769, upper bound: 0.0053263
time: 0.77 seconds

### Relational analysis RSZ of RS_RSZ1_RSZ1_RSZ1_RSZ2

#### Relational analysis RSZ result of RS_RSZ1_RSZ1_RSZ1_RSZ2
Status: Status.UNKNOWN
Output dim: 9, lower bound: -0.0055618, upper bound: 0.0058160
time: 0.77 seconds

## BFS RS instance: RS_RSZ1_RSZ1_RSZ2

### Backsubstitution after applying RS history:
0: -0.0039095, -0.0003029, -0.0039095, -0.0003029, -0.0036066, 0.0036066
1: -0.0044907, 0.0040812, -0.0044907, 0.0040812, -0.0062501, 0.0062500
2: 0.0029569, 0.0099431, 0.0029569, 0.0099431, -0.0050570, 0.0050660
3: -0.0045519, -0.0034923, -0.0045519, -0.0034923, -0.0010595, 0.0010595
4: 0.0017018, 0.0078700, 0.0017018, 0.0078700, -0.0055630, 0.0055743
5: -0.0030341, 0.0031953, -0.0030341, 0.0031953, -0.0052739, 0.0052712
6: -0.0068516, -0.0033211, -0.0068516, -0.0033211, -0.0034103, 0.0034094
7: -0.0021570, 0.0038259, -0.0021570, 0.0038259, -0.0059593, 0.0059786
8: -0.0008533, 0.0000782, -0.0008533, 0.0000782, -0.0009315, 0.0009315
9: 0.9973557, 1.0121032, 0.9973557, 1.0121032, -0.0111125, 0.0111360

### Unstable ReLU Count (Linear/Conv2D Layers)
- layer_idx=0, type=LayerType.Linear, total=256, inp1_unstable=7, inp2_unstable=7, delta_unstable=253
- layer_idx=2, type=LayerType.Linear, total=256, inp1_unstable=10, inp2_unstable=10, delta_unstable=256
- layer_idx=4, type=LayerType.Linear, total=256, inp1_unstable=1, inp2_unstable=1, delta_unstable=256
- layer_idx=6, type=LayerType.Linear, total=256, inp1_unstable=0, inp2_unstable=0, delta_unstable=256
- layer_idx=8, type=LayerType.Linear, total=10, inp1_unstable=4, inp2_unstable=4, delta_unstable=10

Time for backsubstitution: 1.24 seconds

### RS candidates at layer 1
type: RSZ, layer: 1, pos: 114
type: RSZ, layer: 1, pos: 140
type: RSZ, layer: 1, pos: 148
type: RSZ, layer: 1, pos: 219

Time for candidate selection: 0.13 seconds

### Candidate
type: RSZ, layer: 1, pos: 114

### Relational analysis RSZ of RS_RSZ1_RSZ1_RSZ2_RSZ1

#### Relational analysis RSZ result of RS_RSZ1_RSZ1_RSZ2_RSZ1
Status: Status.UNKNOWN
Output dim: 9, lower bound: -0.0059461, upper bound: 0.0053722
time: 0.77 seconds

### Relational analysis RSZ of RS_RSZ1_RSZ1_RSZ2_RSZ2

#### Relational analysis RSZ result of RS_RSZ1_RSZ1_RSZ2_RSZ2
Status: Status.UNKNOWN
Output dim: 9, lower bound: -0.0054856, upper bound: 0.0058321
time: 0.75 seconds

## BFS RS instance: RS_RSZ1_RSZ2_RSZ1

### Backsubstitution after applying RS history:
0: -0.0039095, -0.0003029, -0.0039095, -0.0003029, -0.0036066, 0.0036066
1: -0.0044907, 0.0040812, -0.0044907, 0.0040812, -0.0062421, 0.0062584
2: 0.0029569, 0.0099431, 0.0029569, 0.0099431, -0.0050590, 0.0050653
3: -0.0045519, -0.0034923, -0.0045519, -0.0034923, -0.0010595, 0.0010595
4: 0.0017018, 0.0078700, 0.0017018, 0.0078700, -0.0055755, 0.0055616
5: -0.0030341, 0.0031953, -0.0030341, 0.0031953, -0.0052703, 0.0052747
6: -0.0068516, -0.0033211, -0.0068516, -0.0033211, -0.0034088, 0.0034109
7: -0.0021570, 0.0038259, -0.0021570, 0.0038259, -0.0059829, 0.0059528
8: -0.0008533, 0.0000782, -0.0008533, 0.0000782, -0.0009315, 0.0009315
9: 0.9973557, 1.0121032, 0.9973557, 1.0121032, -0.0111444, 0.0111035

### Unstable ReLU Count (Linear/Conv2D Layers)
- layer_idx=0, type=LayerType.Linear, total=256, inp1_unstable=7, inp2_unstable=7, delta_unstable=253
- layer_idx=2, type=LayerType.Linear, total=256, inp1_unstable=10, inp2_unstable=10, delta_unstable=256
- layer_idx=4, type=LayerType.Linear, total=256, inp1_unstable=1, inp2_unstable=1, delta_unstable=256
- layer_idx=6, type=LayerType.Linear, total=256, inp1_unstable=0, inp2_unstable=0, delta_unstable=256
- layer_idx=8, type=LayerType.Linear, total=10, inp1_unstable=4, inp2_unstable=4, delta_unstable=10

Time for backsubstitution: 1.39 seconds

### RS candidates at layer 1
type: RSZ, layer: 1, pos: 114
type: RSZ, layer: 1, pos: 140
type: RSZ, layer: 1, pos: 148
type: RSZ, layer: 1, pos: 219

Time for candidate selection: 0.15 seconds

### Candidate
type: RSZ, layer: 1, pos: 114

### Relational analysis RSZ of RS_RSZ1_RSZ2_RSZ1_RSZ1

#### Relational analysis RSZ result of RS_RSZ1_RSZ2_RSZ1_RSZ1
Status: Status.UNKNOWN
Output dim: 9, lower bound: -0.0059016, upper bound: 0.0054484
time: 0.85 seconds

### Relational analysis RSZ of RS_RSZ1_RSZ2_RSZ1_RSZ2

#### Relational analysis RSZ result of RS_RSZ1_RSZ2_RSZ1_RSZ2
Status: Status.UNKNOWN
Output dim: 9, lower bound: -0.0054621, upper bound: 0.0059111
time: 0.87 seconds

## BFS RS instance: RS_RSZ1_RSZ2_RSZ2

### Backsubstitution after applying RS history:
0: -0.0039095, -0.0003029, -0.0039095, -0.0003029, -0.0036066, 0.0036066
1: -0.0044907, 0.0040812, -0.0044907, 0.0040812, -0.0062762, 0.0062244
2: 0.0029569, 0.0099431, 0.0029569, 0.0099431, -0.0050781, 0.0050467
3: -0.0045519, -0.0034923, -0.0045519, -0.0034923, -0.0010595, 0.0010595
4: 0.0017018, 0.0078700, 0.0017018, 0.0078700, -0.0055627, 0.0055745
5: -0.0030341, 0.0031953, -0.0030341, 0.0031953, -0.0052864, 0.0052579
6: -0.0068516, -0.0033211, -0.0068516, -0.0033211, -0.0034121, 0.0034077
7: -0.0021570, 0.0038259, -0.0021570, 0.0038259, -0.0059799, 0.0059600
8: -0.0008533, 0.0000782, -0.0008533, 0.0000782, -0.0009315, 0.0009315
9: 0.9973557, 1.0121032, 0.9973557, 1.0121032, -0.0110930, 0.0111560

### Unstable ReLU Count (Linear/Conv2D Layers)
- layer_idx=0, type=LayerType.Linear, total=256, inp1_unstable=7, inp2_unstable=7, delta_unstable=253
- layer_idx=2, type=LayerType.Linear, total=256, inp1_unstable=10, inp2_unstable=10, delta_unstable=256
- layer_idx=4, type=LayerType.Linear, total=256, inp1_unstable=1, inp2_unstable=1, delta_unstable=256
- layer_idx=6, type=LayerType.Linear, total=256, inp1_unstable=0, inp2_unstable=0, delta_unstable=256
- layer_idx=8, type=LayerType.Linear, total=10, inp1_unstable=4, inp2_unstable=4, delta_unstable=10

Time for backsubstitution: 1.38 seconds

### RS candidates at layer 1
type: RSZ, layer: 1, pos: 114
type: RSZ, layer: 1, pos: 140
type: RSZ, layer: 1, pos: 148
type: RSZ, layer: 1, pos: 219

Time for candidate selection: 0.15 seconds

### Candidate
type: RSZ, layer: 1, pos: 114

### Relational analysis RSZ of RS_RSZ1_RSZ2_RSZ2_RSZ1

#### Relational analysis RSZ result of RS_RSZ1_RSZ2_RSZ2_RSZ1
Status: Status.UNKNOWN
Output dim: 9, lower bound: -0.0058774, upper bound: 0.0054982
time: 0.81 seconds

### Relational analysis RSZ of RS_RSZ1_RSZ2_RSZ2_RSZ2

#### Relational analysis RSZ result of RS_RSZ1_RSZ2_RSZ2_RSZ2
Status: Status.UNKNOWN
Output dim: 9, lower bound: -0.0053852, upper bound: 0.0059450
time: 0.86 seconds

## BFS RS instance: RS_RSZ2_RSZ1_RSZ1

### Backsubstitution after applying RS history:
0: -0.0039095, -0.0003029, -0.0039095, -0.0003029, -0.0036066, 0.0036066
1: -0.0044907, 0.0040812, -0.0044907, 0.0040812, -0.0062244, 0.0062762
2: 0.0029569, 0.0099431, 0.0029569, 0.0099431, -0.0050467, 0.0050781
3: -0.0045519, -0.0034923, -0.0045519, -0.0034923, -0.0010595, 0.0010595
4: 0.0017018, 0.0078700, 0.0017018, 0.0078700, -0.0055745, 0.0055627
5: -0.0030341, 0.0031953, -0.0030341, 0.0031953, -0.0052579, 0.0052864
6: -0.0068516, -0.0033211, -0.0068516, -0.0033211, -0.0034077, 0.0034121
7: -0.0021570, 0.0038259, -0.0021570, 0.0038259, -0.0059600, 0.0059799
8: -0.0008533, 0.0000782, -0.0008533, 0.0000782, -0.0009315, 0.0009315
9: 0.9973557, 1.0121032, 0.9973557, 1.0121032, -0.0111560, 0.0110930

### Unstable ReLU Count (Linear/Conv2D Layers)
- layer_idx=0, type=LayerType.Linear, total=256, inp1_unstable=7, inp2_unstable=7, delta_unstable=253
- layer_idx=2, type=LayerType.Linear, total=256, inp1_unstable=10, inp2_unstable=10, delta_unstable=256
- layer_idx=4, type=LayerType.Linear, total=256, inp1_unstable=1, inp2_unstable=1, delta_unstable=256
- layer_idx=6, type=LayerType.Linear, total=256, inp1_unstable=0, inp2_unstable=0, delta_unstable=256
- layer_idx=8, type=LayerType.Linear, total=10, inp1_unstable=4, inp2_unstable=4, delta_unstable=10

Time for backsubstitution: 1.30 seconds

### RS candidates at layer 1
type: RSZ, layer: 1, pos: 114
type: RSZ, layer: 1, pos: 140
type: RSZ, layer: 1, pos: 148
type: RSZ, layer: 1, pos: 219

Time for candidate selection: 0.14 seconds

### Candidate
type: RSZ, layer: 1, pos: 114

### Relational analysis RSZ of RS_RSZ2_RSZ1_RSZ1_RSZ1

#### Relational analysis RSZ result of RS_RSZ2_RSZ1_RSZ1_RSZ1
Status: Status.UNKNOWN
Output dim: 9, lower bound: -0.0059450, upper bound: 0.0053852
time: 0.85 seconds

### Relational analysis RSZ of RS_RSZ2_RSZ1_RSZ1_RSZ2

#### Relational analysis RSZ result of RS_RSZ2_RSZ1_RSZ1_RSZ2
Status: Status.UNKNOWN
Output dim: 9, lower bound: -0.0054982, upper bound: 0.0058774
time: 0.78 seconds

## BFS RS instance: RS_RSZ2_RSZ1_RSZ2

### Backsubstitution after applying RS history:
0: -0.0039095, -0.0003029, -0.0039095, -0.0003029, -0.0036066, 0.0036066
1: -0.0044907, 0.0040812, -0.0044907, 0.0040812, -0.0062584, 0.0062421
2: 0.0029569, 0.0099431, 0.0029569, 0.0099431, -0.0050653, 0.0050590
3: -0.0045519, -0.0034923, -0.0045519, -0.0034923, -0.0010595, 0.0010595
4: 0.0017018, 0.0078700, 0.0017018, 0.0078700, -0.0055616, 0.0055755
5: -0.0030341, 0.0031953, -0.0030341, 0.0031953, -0.0052747, 0.0052703
6: -0.0068516, -0.0033211, -0.0068516, -0.0033211, -0.0034109, 0.0034088
7: -0.0021570, 0.0038259, -0.0021570, 0.0038259, -0.0059528, 0.0059829
8: -0.0008533, 0.0000782, -0.0008533, 0.0000782, -0.0009315, 0.0009315
9: 0.9973557, 1.0121032, 0.9973557, 1.0121032, -0.0111035, 0.0111444

### Unstable ReLU Count (Linear/Conv2D Layers)
- layer_idx=0, type=LayerType.Linear, total=256, inp1_unstable=7, inp2_unstable=7, delta_unstable=253
- layer_idx=2, type=LayerType.Linear, total=256, inp1_unstable=10, inp2_unstable=10, delta_unstable=256
- layer_idx=4, type=LayerType.Linear, total=256, inp1_unstable=1, inp2_unstable=1, delta_unstable=256
- layer_idx=6, type=LayerType.Linear, total=256, inp1_unstable=0, inp2_unstable=0, delta_unstable=256
- layer_idx=8, type=LayerType.Linear, total=10, inp1_unstable=4, inp2_unstable=4, delta_unstable=10

Time for backsubstitution: 1.24 seconds

### RS candidates at layer 1
type: RSZ, layer: 1, pos: 114
type: RSZ, layer: 1, pos: 140
type: RSZ, layer: 1, pos: 148
type: RSZ, layer: 1, pos: 219

Time for candidate selection: 0.14 seconds

### Candidate
type: RSZ, layer: 1, pos: 114

### Relational analysis RSZ of RS_RSZ2_RSZ1_RSZ2_RSZ1

#### Relational analysis RSZ result of RS_RSZ2_RSZ1_RSZ2_RSZ1
Status: Status.UNKNOWN
Output dim: 9, lower bound: -0.0059111, upper bound: 0.0054621
time: 0.84 seconds

### Relational analysis RSZ of RS_RSZ2_RSZ1_RSZ2_RSZ2

#### Relational analysis RSZ result of RS_RSZ2_RSZ1_RSZ2_RSZ2
Status: Status.UNKNOWN
Output dim: 9, lower bound: -0.0054484, upper bound: 0.0059016
time: 0.81 seconds

## BFS RS instance: RS_RSZ2_RSZ2_RSZ1

### Backsubstitution after applying RS history:
0: -0.0039095, -0.0003029, -0.0039095, -0.0003029, -0.0036066, 0.0036066
1: -0.0044907, 0.0040812, -0.0044907, 0.0040812, -0.0062500, 0.0062501
2: 0.0029569, 0.0099431, 0.0029569, 0.0099431, -0.0050660, 0.0050570
3: -0.0045519, -0.0034923, -0.0045519, -0.0034923, -0.0010595, 0.0010595
4: 0.0017018, 0.0078700, 0.0017018, 0.0078700, -0.0055743, 0.0055630
5: -0.0030341, 0.0031953, -0.0030341, 0.0031953, -0.0052712, 0.0052739
6: -0.0068516, -0.0033211, -0.0068516, -0.0033211, -0.0034094, 0.0034103
7: -0.0021570, 0.0038259, -0.0021570, 0.0038259, -0.0059786, 0.0059593
8: -0.0008533, 0.0000782, -0.0008533, 0.0000782, -0.0009315, 0.0009315
9: 0.9973557, 1.0121032, 0.9973557, 1.0121032, -0.0111360, 0.0111125

### Unstable ReLU Count (Linear/Conv2D Layers)
- layer_idx=0, type=LayerType.Linear, total=256, inp1_unstable=7, inp2_unstable=7, delta_unstable=253
- layer_idx=2, type=LayerType.Linear, total=256, inp1_unstable=10, inp2_unstable=10, delta_unstable=256
- layer_idx=4, type=LayerType.Linear, total=256, inp1_unstable=1, inp2_unstable=1, delta_unstable=256
- layer_idx=6, type=LayerType.Linear, total=256, inp1_unstable=0, inp2_unstable=0, delta_unstable=256
- layer_idx=8, type=LayerType.Linear, total=10, inp1_unstable=4, inp2_unstable=4, delta_unstable=10

Time for backsubstitution: 1.26 seconds

### RS candidates at layer 1
type: RSZ, layer: 1, pos: 114
type: RSZ, layer: 1, pos: 140
type: RSZ, layer: 1, pos: 148
type: RSZ, layer: 1, pos: 219

Time for candidate selection: 0.13 seconds

### Candidate
type: RSZ, layer: 1, pos: 114

### Relational analysis RSZ of RS_RSZ2_RSZ2_RSZ1_RSZ1

#### Relational analysis RSZ result of RS_RSZ2_RSZ2_RSZ1_RSZ1
Status: Status.UNKNOWN
Output dim: 9, lower bound: -0.0058321, upper bound: 0.0054856
time: 0.81 seconds

### Relational analysis RSZ of RS_RSZ2_RSZ2_RSZ1_RSZ2

#### Relational analysis RSZ result of RS_RSZ2_RSZ2_RSZ1_RSZ2
Status: Status.UNKNOWN
Output dim: 9, lower bound: -0.0053722, upper bound: 0.0059461
time: 0.85 seconds

## BFS RS instance: RS_RSZ2_RSZ2_RSZ2

### Backsubstitution after applying RS history:
0: -0.0039095, -0.0003029, -0.0039095, -0.0003029, -0.0036066, 0.0036066
1: -0.0044907, 0.0040812, -0.0044907, 0.0040812, -0.0062852, 0.0062172
2: 0.0029569, 0.0099431, 0.0029569, 0.0099431, -0.0050860, 0.0050400
3: -0.0045519, -0.0034923, -0.0045519, -0.0034923, -0.0010595, 0.0010595
4: 0.0017018, 0.0078700, 0.0017018, 0.0078700, -0.0055613, 0.0055758
5: -0.0030341, 0.0031953, -0.0030341, 0.0031953, -0.0052873, 0.0052570
6: -0.0068516, -0.0033211, -0.0068516, -0.0033211, -0.0034127, 0.0034071
7: -0.0021570, 0.0038259, -0.0021570, 0.0038259, -0.0059727, 0.0059669
8: -0.0008533, 0.0000782, -0.0008533, 0.0000782, -0.0009315, 0.0009315
9: 0.9973557, 1.0121032, 0.9973557, 1.0121032, -0.0110841, 0.0111642

### Unstable ReLU Count (Linear/Conv2D Layers)
- layer_idx=0, type=LayerType.Linear, total=256, inp1_unstable=7, inp2_unstable=7, delta_unstable=253
- layer_idx=2, type=LayerType.Linear, total=256, inp1_unstable=10, inp2_unstable=10, delta_unstable=256
- layer_idx=4, type=LayerType.Linear, total=256, inp1_unstable=1, inp2_unstable=1, delta_unstable=256
- layer_idx=6, type=LayerType.Linear, total=256, inp1_unstable=0, inp2_unstable=0, delta_unstable=256
- layer_idx=8, type=LayerType.Linear, total=10, inp1_unstable=4, inp2_unstable=4, delta_unstable=10

Time for backsubstitution: 1.24 seconds

### RS candidates at layer 1
type: RSZ, layer: 1, pos: 114
type: RSZ, layer: 1, pos: 140
type: RSZ, layer: 1, pos: 148
type: RSZ, layer: 1, pos: 219

Time for candidate selection: 0.15 seconds

### Candidate
type: RSZ, layer: 1, pos: 114

### Relational analysis RSZ of RS_RSZ2_RSZ2_RSZ2_RSZ1

#### Relational analysis RSZ result of RS_RSZ2_RSZ2_RSZ2_RSZ1
Status: Status.UNKNOWN
Output dim: 9, lower bound: -0.0058160, upper bound: 0.0055618
time: 0.80 seconds

### Relational analysis RSZ of RS_RSZ2_RSZ2_RSZ2_RSZ2

#### Relational analysis RSZ result of RS_RSZ2_RSZ2_RSZ2_RSZ2
Status: Status.UNKNOWN
Output dim: 9, lower bound: -0.0053263, upper bound: 0.0059769
time: 0.84 seconds

## Summary of splitting (split count: 3)
- Time for RS candidates: 3.04 seconds
RS_RSZ1_RSZ1_RSZ1_RSZ1, status: Status.UNKNOWN, split count: 4, time: 3.04
Output dim: 9, lower bound: -0.0059769, upper bound: 0.0053263
RS_RSZ1_RSZ1_RSZ1_RSZ2, status: Status.UNKNOWN, split count: 4, time: 3.04
Output dim: 9, lower bound: -0.0055618, upper bound: 0.0058160
RS_RSZ1_RSZ1_RSZ2_RSZ1, status: Status.UNKNOWN, split count: 4, time: 3.04
Output dim: 9, lower bound: -0.0059461, upper bound: 0.0053722
RS_RSZ1_RSZ1_RSZ2_RSZ2, status: Status.UNKNOWN, split count: 4, time: 3.04
Output dim: 9, lower bound: -0.0054856, upper bound: 0.0058321
RS_RSZ1_RSZ2_RSZ1_RSZ1, status: Status.UNKNOWN, split count: 4, time: 3.04
Output dim: 9, lower bound: -0.0059016, upper bound: 0.0054484
RS_RSZ1_RSZ2_RSZ1_RSZ2, status: Status.UNKNOWN, split count: 4, time: 3.04
Output dim: 9, lower bound: -0.0054621, upper bound: 0.0059111
RS_RSZ1_RSZ2_RSZ2_RSZ1, status: Status.UNKNOWN, split count: 4, time: 3.04
Output dim: 9, lower bound: -0.0058774, upper bound: 0.0054982
RS_RSZ1_RSZ2_RSZ2_RSZ2, status: Status.UNKNOWN, split count: 4, time: 3.04
Output dim: 9, lower bound: -0.0053852, upper bound: 0.0059450
RS_RSZ2_RSZ1_RSZ1_RSZ1, status: Status.UNKNOWN, split count: 4, time: 3.04
Output dim: 9, lower bound: -0.0059450, upper bound: 0.0053852
RS_RSZ2_RSZ1_RSZ1_RSZ2, status: Status.UNKNOWN, split count: 4, time: 3.04
Output dim: 9, lower bound: -0.0054982, upper bound: 0.0058774
RS_RSZ2_RSZ1_RSZ2_RSZ1, status: Status.UNKNOWN, split count: 4, time: 3.04
Output dim: 9, lower bound: -0.0059111, upper bound: 0.0054621
RS_RSZ2_RSZ1_RSZ2_RSZ2, status: Status.UNKNOWN, split count: 4, time: 3.04
Output dim: 9, lower bound: -0.0054484, upper bound: 0.0059016
RS_RSZ2_RSZ2_RSZ1_RSZ1, status: Status.UNKNOWN, split count: 4, time: 3.04
Output dim: 9, lower bound: -0.0058321, upper bound: 0.0054856
RS_RSZ2_RSZ2_RSZ1_RSZ2, status: Status.UNKNOWN, split count: 4, time: 3.04
Output dim: 9, lower bound: -0.0053722, upper bound: 0.0059461
RS_RSZ2_RSZ2_RSZ2_RSZ1, status: Status.UNKNOWN, split count: 4, time: 3.04
Output dim: 9, lower bound: -0.0058160, upper bound: 0.0055618
RS_RSZ2_RSZ2_RSZ2_RSZ2, status: Status.UNKNOWN, split count: 4, time: 3.04
Output dim: 9, lower bound: -0.0053263, upper bound: 0.0059769

## BFS RS instance: RS_RSZ1_RSZ1_RSZ1_RSZ1

### Backsubstitution after applying RS history:
0: -0.0039095, -0.0003029, -0.0039095, -0.0003029, -0.0036066, 0.0035705
1: -0.0044907, 0.0040812, -0.0044907, 0.0040812, -0.0055162, 0.0057425
2: 0.0029569, 0.0099431, 0.0029569, 0.0099431, -0.0044626, 0.0046108
3: -0.0045519, -0.0034923, -0.0045519, -0.0034923, -0.0010595, 0.0010595
4: 0.0017018, 0.0078700, 0.0017018, 0.0078700, -0.0053787, 0.0053125
5: -0.0030341, 0.0031953, -0.0030341, 0.0031953, -0.0048568, 0.0049635
6: -0.0068516, -0.0033211, -0.0068516, -0.0033211, -0.0033439, 0.0033637
7: -0.0021570, 0.0038259, -0.0021570, 0.0038259, -0.0058689, 0.0058514
8: -0.0008533, 0.0000782, -0.0008533, 0.0000782, -0.0008967, 0.0009273
9: 0.9973557, 1.0121032, 0.9973557, 1.0121032, -0.0102090, 0.0099115

### Unstable ReLU Count (Linear/Conv2D Layers)
- layer_idx=0, type=LayerType.Linear, total=256, inp1_unstable=7, inp2_unstable=7, delta_unstable=252
- layer_idx=2, type=LayerType.Linear, total=256, inp1_unstable=10, inp2_unstable=10, delta_unstable=256
- layer_idx=4, type=LayerType.Linear, total=256, inp1_unstable=1, inp2_unstable=1, delta_unstable=256
- layer_idx=6, type=LayerType.Linear, total=256, inp1_unstable=0, inp2_unstable=0, delta_unstable=256
- layer_idx=8, type=LayerType.Linear, total=10, inp1_unstable=4, inp2_unstable=4, delta_unstable=10

Time for backsubstitution: 1.25 seconds

### RS candidates at layer 1
type: RSZ, layer: 1, pos: 140
type: RSZ, layer: 1, pos: 148
type: RSZ, layer: 1, pos: 219

Time for candidate selection: 0.14 seconds

### Candidate
type: RSZ, layer: 1, pos: 140

### Relational analysis RSZ of RS_RSZ1_RSZ1_RSZ1_RSZ1_RSZ1

#### Relational analysis RSZ result of RS_RSZ1_RSZ1_RSZ1_RSZ1_RSZ1
Status: Status.UNKNOWN
Output dim: 9, lower bound: -0.0055349, upper bound: 0.0050108
time: 0.78 seconds

### Relational analysis RSZ of RS_RSZ1_RSZ1_RSZ1_RSZ1_RSZ2

#### Relational analysis RSZ result of RS_RSZ1_RSZ1_RSZ1_RSZ1_RSZ2
Status: Status.UNKNOWN
Output dim: 9, lower bound: -0.0056671, upper bound: 0.0048847
time: 0.76 seconds

## BFS RS instance: RS_RSZ1_RSZ1_RSZ1_RSZ2

### Backsubstitution after applying RS history:
0: -0.0039095, -0.0003029, -0.0039095, -0.0003029, -0.0035903, 0.0036066
1: -0.0044907, 0.0040812, -0.0044907, 0.0040812, -0.0056923, 0.0055843
2: 0.0029569, 0.0099431, 0.0029569, 0.0099431, -0.0045777, 0.0045086
3: -0.0045519, -0.0034923, -0.0045519, -0.0034923, -0.0010595, 0.0010595
4: 0.0017018, 0.0078700, 0.0017018, 0.0078700, -0.0053269, 0.0053675
5: -0.0030341, 0.0031953, -0.0030341, 0.0031953, -0.0049345, 0.0048871
6: -0.0068516, -0.0033211, -0.0068516, -0.0033211, -0.0033594, 0.0033495
7: -0.0021570, 0.0038259, -0.0021570, 0.0038259, -0.0058456, 0.0058781
8: -0.0008533, 0.0000782, -0.0008533, 0.0000782, -0.0009071, 0.0009191
9: 0.9973557, 1.0121032, 0.9973557, 1.0121032, -0.0099916, 0.0101484

### Unstable ReLU Count (Linear/Conv2D Layers)
- layer_idx=0, type=LayerType.Linear, total=256, inp1_unstable=7, inp2_unstable=7, delta_unstable=252
- layer_idx=2, type=LayerType.Linear, total=256, inp1_unstable=10, inp2_unstable=10, delta_unstable=256
- layer_idx=4, type=LayerType.Linear, total=256, inp1_unstable=1, inp2_unstable=1, delta_unstable=256
- layer_idx=6, type=LayerType.Linear, total=256, inp1_unstable=0, inp2_unstable=0, delta_unstable=256
- layer_idx=8, type=LayerType.Linear, total=10, inp1_unstable=4, inp2_unstable=4, delta_unstable=10

Time for backsubstitution: 1.28 seconds

### RS candidates at layer 1
type: RSZ, layer: 1, pos: 140
type: RSZ, layer: 1, pos: 148
type: RSZ, layer: 1, pos: 219

Time for candidate selection: 0.14 seconds

### Candidate
type: RSZ, layer: 1, pos: 140

### Relational analysis RSZ of RS_RSZ1_RSZ1_RSZ1_RSZ2_RSZ1

#### Relational analysis RSZ result of RS_RSZ1_RSZ1_RSZ1_RSZ2_RSZ1
Status: Status.UNKNOWN
Output dim: 9, lower bound: -0.0051328, upper bound: 0.0055004
time: 0.78 seconds

### Relational analysis RSZ of RS_RSZ1_RSZ1_RSZ1_RSZ2_RSZ2

#### Relational analysis RSZ result of RS_RSZ1_RSZ1_RSZ1_RSZ2_RSZ2
Status: Status.UNKNOWN
Output dim: 9, lower bound: -0.0052212, upper bound: 0.0053588
time: 0.79 seconds

## BFS RS instance: RS_RSZ1_RSZ1_RSZ2_RSZ1

### Backsubstitution after applying RS history:
0: -0.0039095, -0.0003029, -0.0039095, -0.0003029, -0.0036066, 0.0035874
1: -0.0044907, 0.0040812, -0.0044907, 0.0040812, -0.0055492, 0.0057182
2: 0.0029569, 0.0099431, 0.0029569, 0.0099431, -0.0044797, 0.0045964
3: -0.0045519, -0.0034923, -0.0045519, -0.0034923, -0.0010595, 0.0010595
4: 0.0017018, 0.0078700, 0.0017018, 0.0078700, -0.0053690, 0.0053254
5: -0.0030341, 0.0031953, -0.0030341, 0.0031953, -0.0048737, 0.0049503
6: -0.0068516, -0.0033211, -0.0068516, -0.0033211, -0.0033471, 0.0033614
7: -0.0021570, 0.0038259, -0.0021570, 0.0038259, -0.0058630, 0.0058573
8: -0.0008533, 0.0000782, -0.0008533, 0.0000782, -0.0008881, 0.0009315
9: 0.9973557, 1.0121032, 0.9973557, 1.0121032, -0.0101721, 0.0099635

### Unstable ReLU Count (Linear/Conv2D Layers)
- layer_idx=0, type=LayerType.Linear, total=256, inp1_unstable=7, inp2_unstable=7, delta_unstable=252
- layer_idx=2, type=LayerType.Linear, total=256, inp1_unstable=10, inp2_unstable=10, delta_unstable=256
- layer_idx=4, type=LayerType.Linear, total=256, inp1_unstable=1, inp2_unstable=1, delta_unstable=256
- layer_idx=6, type=LayerType.Linear, total=256, inp1_unstable=0, inp2_unstable=0, delta_unstable=256
- layer_idx=8, type=LayerType.Linear, total=10, inp1_unstable=4, inp2_unstable=4, delta_unstable=10

Time for backsubstitution: 1.26 seconds

### RS candidates at layer 1
type: RSZ, layer: 1, pos: 140
type: RSZ, layer: 1, pos: 148
type: RSZ, layer: 1, pos: 219

Time for candidate selection: 0.14 seconds

### Candidate
type: RSZ, layer: 1, pos: 140

### Relational analysis RSZ of RS_RSZ1_RSZ1_RSZ2_RSZ1_RSZ1

#### Relational analysis RSZ result of RS_RSZ1_RSZ1_RSZ2_RSZ1_RSZ1
Status: Status.UNKNOWN
Output dim: 9, lower bound: -0.0054781, upper bound: 0.0050420
time: 0.84 seconds

### Relational analysis RSZ of RS_RSZ1_RSZ1_RSZ2_RSZ1_RSZ2

#### Relational analysis RSZ result of RS_RSZ1_RSZ1_RSZ2_RSZ1_RSZ2
Status: Status.UNKNOWN
Output dim: 9, lower bound: -0.0056423, upper bound: 0.0049505
time: 0.79 seconds

## BFS RS instance: RS_RSZ1_RSZ1_RSZ2_RSZ2

### Backsubstitution after applying RS history:
0: -0.0039095, -0.0003029, -0.0039095, -0.0003029, -0.0035732, 0.0036066
1: -0.0044907, 0.0040812, -0.0044907, 0.0040812, -0.0057132, 0.0055490
2: 0.0029569, 0.0099431, 0.0029569, 0.0099431, -0.0045911, 0.0044887
3: -0.0045519, -0.0034923, -0.0045519, -0.0034923, -0.0010595, 0.0010595
4: 0.0017018, 0.0078700, 0.0017018, 0.0078700, -0.0053141, 0.0053772
5: -0.0030341, 0.0031953, -0.0030341, 0.0031953, -0.0049491, 0.0048710
6: -0.0068516, -0.0033211, -0.0068516, -0.0033211, -0.0033616, 0.0033462
7: -0.0021570, 0.0038259, -0.0021570, 0.0038259, -0.0058380, 0.0058836
8: -0.0008533, 0.0000782, -0.0008533, 0.0000782, -0.0008960, 0.0009295
9: 0.9973557, 1.0121032, 0.9973557, 1.0121032, -0.0099399, 0.0101834

### Unstable ReLU Count (Linear/Conv2D Layers)
- layer_idx=0, type=LayerType.Linear, total=256, inp1_unstable=7, inp2_unstable=7, delta_unstable=252
- layer_idx=2, type=LayerType.Linear, total=256, inp1_unstable=10, inp2_unstable=10, delta_unstable=256
- layer_idx=4, type=LayerType.Linear, total=256, inp1_unstable=1, inp2_unstable=1, delta_unstable=256
- layer_idx=6, type=LayerType.Linear, total=256, inp1_unstable=0, inp2_unstable=0, delta_unstable=256
- layer_idx=8, type=LayerType.Linear, total=10, inp1_unstable=4, inp2_unstable=4, delta_unstable=10

Time for backsubstitution: 1.28 seconds

### RS candidates at layer 1
type: RSZ, layer: 1, pos: 140
type: RSZ, layer: 1, pos: 148
type: RSZ, layer: 1, pos: 219

Time for candidate selection: 0.15 seconds

### Candidate
type: RSZ, layer: 1, pos: 140

### Relational analysis RSZ of RS_RSZ1_RSZ1_RSZ2_RSZ2_RSZ1

#### Relational analysis RSZ result of RS_RSZ1_RSZ1_RSZ2_RSZ2_RSZ1
Status: Status.UNKNOWN
Output dim: 9, lower bound: -0.0050475, upper bound: 0.0055089
time: 0.82 seconds

### Relational analysis RSZ of RS_RSZ1_RSZ1_RSZ2_RSZ2_RSZ2

#### Relational analysis RSZ result of RS_RSZ1_RSZ1_RSZ2_RSZ2_RSZ2
Status: Status.UNKNOWN
Output dim: 9, lower bound: -0.0051694, upper bound: 0.0053966
time: 0.77 seconds

## BFS RS instance: RS_RSZ1_RSZ2_RSZ1_RSZ1

### Backsubstitution after applying RS history:
0: -0.0039095, -0.0003029, -0.0039095, -0.0003029, -0.0036066, 0.0035719
1: -0.0044907, 0.0040812, -0.0044907, 0.0040812, -0.0055412, 0.0057217
2: 0.0029569, 0.0099431, 0.0029569, 0.0099431, -0.0044817, 0.0046002
3: -0.0045519, -0.0034923, -0.0045519, -0.0034923, -0.0010595, 0.0010595
4: 0.0017018, 0.0078700, 0.0017018, 0.0078700, -0.0053784, 0.0053128
5: -0.0030341, 0.0031953, -0.0030341, 0.0031953, -0.0048701, 0.0049498
6: -0.0068516, -0.0033211, -0.0068516, -0.0033211, -0.0033456, 0.0033622
7: -0.0021570, 0.0038259, -0.0021570, 0.0038259, -0.0058903, 0.0058315
8: -0.0008533, 0.0000782, -0.0008533, 0.0000782, -0.0009301, 0.0008965
9: 0.9973557, 1.0121032, 0.9973557, 1.0121032, -0.0101918, 0.0099309

### Unstable ReLU Count (Linear/Conv2D Layers)
- layer_idx=0, type=LayerType.Linear, total=256, inp1_unstable=7, inp2_unstable=7, delta_unstable=252
- layer_idx=2, type=LayerType.Linear, total=256, inp1_unstable=10, inp2_unstable=10, delta_unstable=256
- layer_idx=4, type=LayerType.Linear, total=256, inp1_unstable=1, inp2_unstable=1, delta_unstable=256
- layer_idx=6, type=LayerType.Linear, total=256, inp1_unstable=0, inp2_unstable=0, delta_unstable=256
- layer_idx=8, type=LayerType.Linear, total=10, inp1_unstable=4, inp2_unstable=4, delta_unstable=10

Time for backsubstitution: 1.29 seconds

### RS candidates at layer 1
type: RSZ, layer: 1, pos: 140
type: RSZ, layer: 1, pos: 148
type: RSZ, layer: 1, pos: 219

Time for candidate selection: 0.13 seconds

### Candidate
type: RSZ, layer: 1, pos: 140

### Relational analysis RSZ of RS_RSZ1_RSZ2_RSZ1_RSZ1_RSZ1

#### Relational analysis RSZ result of RS_RSZ1_RSZ2_RSZ1_RSZ1_RSZ1
Status: Status.UNKNOWN
Output dim: 9, lower bound: -0.0054645, upper bound: 0.0051332
time: 0.76 seconds

### Relational analysis RSZ of RS_RSZ1_RSZ2_RSZ1_RSZ1_RSZ2

#### Relational analysis RSZ result of RS_RSZ1_RSZ2_RSZ1_RSZ1_RSZ2
Status: Status.UNKNOWN
Output dim: 9, lower bound: -0.0055684, upper bound: 0.0050013
time: 0.77 seconds

## BFS RS instance: RS_RSZ1_RSZ2_RSZ1_RSZ2

### Backsubstitution after applying RS history:
0: -0.0039095, -0.0003029, -0.0039095, -0.0003029, -0.0035890, 0.0036066
1: -0.0044907, 0.0040812, -0.0044907, 0.0040812, -0.0057109, 0.0055575
2: 0.0029569, 0.0099431, 0.0029569, 0.0099431, -0.0045908, 0.0044880
3: -0.0045519, -0.0034923, -0.0045519, -0.0034923, -0.0010595, 0.0010595
4: 0.0017018, 0.0078700, 0.0017018, 0.0078700, -0.0053267, 0.0053678
5: -0.0030341, 0.0031953, -0.0030341, 0.0031953, -0.0049498, 0.0048745
6: -0.0068516, -0.0033211, -0.0068516, -0.0033211, -0.0033608, 0.0033477
7: -0.0021570, 0.0038259, -0.0021570, 0.0038259, -0.0058649, 0.0058565
8: -0.0008533, 0.0000782, -0.0008533, 0.0000782, -0.0009315, 0.0008881
9: 0.9973557, 1.0121032, 0.9973557, 1.0121032, -0.0099718, 0.0101644

### Unstable ReLU Count (Linear/Conv2D Layers)
- layer_idx=0, type=LayerType.Linear, total=256, inp1_unstable=7, inp2_unstable=7, delta_unstable=252
- layer_idx=2, type=LayerType.Linear, total=256, inp1_unstable=10, inp2_unstable=10, delta_unstable=256
- layer_idx=4, type=LayerType.Linear, total=256, inp1_unstable=1, inp2_unstable=1, delta_unstable=256
- layer_idx=6, type=LayerType.Linear, total=256, inp1_unstable=0, inp2_unstable=0, delta_unstable=256
- layer_idx=8, type=LayerType.Linear, total=10, inp1_unstable=4, inp2_unstable=4, delta_unstable=10

Time for backsubstitution: 1.29 seconds

### RS candidates at layer 1
type: RSZ, layer: 1, pos: 140
type: RSZ, layer: 1, pos: 148
type: RSZ, layer: 1, pos: 219

Time for candidate selection: 0.16 seconds

### Candidate
type: RSZ, layer: 1, pos: 140

### Relational analysis RSZ of RS_RSZ1_RSZ2_RSZ1_RSZ2_RSZ1

#### Relational analysis RSZ result of RS_RSZ1_RSZ2_RSZ1_RSZ2_RSZ1
Status: Status.UNKNOWN
Output dim: 9, lower bound: -0.0050358, upper bound: 0.0056121
time: 0.82 seconds

### Relational analysis RSZ of RS_RSZ1_RSZ2_RSZ1_RSZ2_RSZ2

#### Relational analysis RSZ result of RS_RSZ1_RSZ2_RSZ1_RSZ2_RSZ2
Status: Status.UNKNOWN
Output dim: 9, lower bound: -0.0051218, upper bound: 0.0054378
time: 0.77 seconds

## BFS RS instance: RS_RSZ1_RSZ2_RSZ2_RSZ1

### Backsubstitution after applying RS history:
0: -0.0039095, -0.0003029, -0.0039095, -0.0003029, -0.0036066, 0.0035887
1: -0.0044907, 0.0040812, -0.0044907, 0.0040812, -0.0055753, 0.0056998
2: 0.0029569, 0.0099431, 0.0029569, 0.0099431, -0.0045007, 0.0045862
3: -0.0045519, -0.0034923, -0.0045519, -0.0034923, -0.0010595, 0.0010595
4: 0.0017018, 0.0078700, 0.0017018, 0.0078700, -0.0053688, 0.0053257
5: -0.0030341, 0.0031953, -0.0030341, 0.0031953, -0.0048862, 0.0049352
6: -0.0068516, -0.0033211, -0.0068516, -0.0033211, -0.0033489, 0.0033600
7: -0.0021570, 0.0038259, -0.0021570, 0.0038259, -0.0058840, 0.0058387
8: -0.0008533, 0.0000782, -0.0008533, 0.0000782, -0.0009194, 0.0009065
9: 0.9973557, 1.0121032, 0.9973557, 1.0121032, -0.0101558, 0.0099834

### Unstable ReLU Count (Linear/Conv2D Layers)
- layer_idx=0, type=LayerType.Linear, total=256, inp1_unstable=7, inp2_unstable=7, delta_unstable=252
- layer_idx=2, type=LayerType.Linear, total=256, inp1_unstable=10, inp2_unstable=10, delta_unstable=256
- layer_idx=4, type=LayerType.Linear, total=256, inp1_unstable=1, inp2_unstable=1, delta_unstable=256
- layer_idx=6, type=LayerType.Linear, total=256, inp1_unstable=0, inp2_unstable=0, delta_unstable=256
- layer_idx=8, type=LayerType.Linear, total=10, inp1_unstable=4, inp2_unstable=4, delta_unstable=10

Time for backsubstitution: 1.27 seconds

### RS candidates at layer 1
type: RSZ, layer: 1, pos: 140
type: RSZ, layer: 1, pos: 148
type: RSZ, layer: 1, pos: 219

Time for candidate selection: 0.13 seconds

### Candidate
type: RSZ, layer: 1, pos: 140

### Relational analysis RSZ of RS_RSZ1_RSZ2_RSZ2_RSZ1_RSZ1

#### Relational analysis RSZ result of RS_RSZ1_RSZ2_RSZ2_RSZ1_RSZ1
Status: Status.UNKNOWN
Output dim: 9, lower bound: -0.0054290, upper bound: 0.0051660
time: 0.76 seconds

### Relational analysis RSZ of RS_RSZ1_RSZ2_RSZ2_RSZ1_RSZ2

#### Relational analysis RSZ result of RS_RSZ1_RSZ2_RSZ2_RSZ1_RSZ2
Status: Status.UNKNOWN
Output dim: 9, lower bound: -0.0055543, upper bound: 0.0050725
time: 0.75 seconds

## BFS RS instance: RS_RSZ1_RSZ2_RSZ2_RSZ2

### Backsubstitution after applying RS history:
0: -0.0039095, -0.0003029, -0.0039095, -0.0003029, -0.0035719, 0.0036066
1: -0.0044907, 0.0040812, -0.0044907, 0.0040812, -0.0057356, 0.0055235
2: 0.0029569, 0.0099431, 0.0029569, 0.0099431, -0.0046052, 0.0044693
3: -0.0045519, -0.0034923, -0.0045519, -0.0034923, -0.0010595, 0.0010595
4: 0.0017018, 0.0078700, 0.0017018, 0.0078700, -0.0053138, 0.0053775
5: -0.0030341, 0.0031953, -0.0030341, 0.0031953, -0.0049631, 0.0048577
6: -0.0068516, -0.0033211, -0.0068516, -0.0033211, -0.0033631, 0.0033445
7: -0.0021570, 0.0038259, -0.0021570, 0.0038259, -0.0058586, 0.0058620
8: -0.0008533, 0.0000782, -0.0008533, 0.0000782, -0.0009269, 0.0008959
9: 0.9973557, 1.0121032, 0.9973557, 1.0121032, -0.0099204, 0.0102011

### Unstable ReLU Count (Linear/Conv2D Layers)
- layer_idx=0, type=LayerType.Linear, total=256, inp1_unstable=7, inp2_unstable=7, delta_unstable=252
- layer_idx=2, type=LayerType.Linear, total=256, inp1_unstable=10, inp2_unstable=10, delta_unstable=256
- layer_idx=4, type=LayerType.Linear, total=256, inp1_unstable=1, inp2_unstable=1, delta_unstable=256
- layer_idx=6, type=LayerType.Linear, total=256, inp1_unstable=0, inp2_unstable=0, delta_unstable=256
- layer_idx=8, type=LayerType.Linear, total=10, inp1_unstable=4, inp2_unstable=4, delta_unstable=10

Time for backsubstitution: 1.26 seconds

### RS candidates at layer 1
type: RSZ, layer: 1, pos: 140
type: RSZ, layer: 1, pos: 148
type: RSZ, layer: 1, pos: 219

Time for candidate selection: 0.13 seconds

### Candidate
type: RSZ, layer: 1, pos: 140

### Relational analysis RSZ of RS_RSZ1_RSZ2_RSZ2_RSZ2_RSZ1

#### Relational analysis RSZ result of RS_RSZ1_RSZ2_RSZ2_RSZ2_RSZ1
Status: Status.UNKNOWN
Output dim: 9, lower bound: -0.0049516, upper bound: 0.0056326
time: 0.82 seconds

### Relational analysis RSZ of RS_RSZ1_RSZ2_RSZ2_RSZ2_RSZ2

#### Relational analysis RSZ result of RS_RSZ1_RSZ2_RSZ2_RSZ2_RSZ2
Status: Status.UNKNOWN
Output dim: 9, lower bound: -0.0050636, upper bound: 0.0054938
time: 0.79 seconds

## BFS RS instance: RS_RSZ2_RSZ1_RSZ1_RSZ1

### Backsubstitution after applying RS history:
0: -0.0039095, -0.0003029, -0.0039095, -0.0003029, -0.0036066, 0.0035719
1: -0.0044907, 0.0040812, -0.0044907, 0.0040812, -0.0055235, 0.0057356
2: 0.0029569, 0.0099431, 0.0029569, 0.0099431, -0.0044693, 0.0046052
3: -0.0045519, -0.0034923, -0.0045519, -0.0034923, -0.0010595, 0.0010595
4: 0.0017018, 0.0078700, 0.0017018, 0.0078700, -0.0053775, 0.0053138
5: -0.0030341, 0.0031953, -0.0030341, 0.0031953, -0.0048577, 0.0049631
6: -0.0068516, -0.0033211, -0.0068516, -0.0033211, -0.0033445, 0.0033631
7: -0.0021570, 0.0038259, -0.0021570, 0.0038259, -0.0058620, 0.0058586
8: -0.0008533, 0.0000782, -0.0008533, 0.0000782, -0.0008959, 0.0009269
9: 0.9973557, 1.0121032, 0.9973557, 1.0121032, -0.0102011, 0.0099204

### Unstable ReLU Count (Linear/Conv2D Layers)
- layer_idx=0, type=LayerType.Linear, total=256, inp1_unstable=7, inp2_unstable=7, delta_unstable=252
- layer_idx=2, type=LayerType.Linear, total=256, inp1_unstable=10, inp2_unstable=10, delta_unstable=256
- layer_idx=4, type=LayerType.Linear, total=256, inp1_unstable=1, inp2_unstable=1, delta_unstable=256
- layer_idx=6, type=LayerType.Linear, total=256, inp1_unstable=0, inp2_unstable=0, delta_unstable=256
- layer_idx=8, type=LayerType.Linear, total=10, inp1_unstable=4, inp2_unstable=4, delta_unstable=10

Time for backsubstitution: 1.30 seconds

### RS candidates at layer 1
type: RSZ, layer: 1, pos: 140
type: RSZ, layer: 1, pos: 148
type: RSZ, layer: 1, pos: 219

Time for candidate selection: 0.14 seconds

### Candidate
type: RSZ, layer: 1, pos: 140

### Relational analysis RSZ of RS_RSZ2_RSZ1_RSZ1_RSZ1_RSZ1

#### Relational analysis RSZ result of RS_RSZ2_RSZ1_RSZ1_RSZ1_RSZ1
Status: Status.UNKNOWN
Output dim: 9, lower bound: -0.0054938, upper bound: 0.0050636
time: 0.79 seconds

### Relational analysis RSZ of RS_RSZ2_RSZ1_RSZ1_RSZ1_RSZ2

#### Relational analysis RSZ result of RS_RSZ2_RSZ1_RSZ1_RSZ1_RSZ2
Status: Status.UNKNOWN
Output dim: 9, lower bound: -0.0056326, upper bound: 0.0049517
time: 0.74 seconds

## BFS RS instance: RS_RSZ2_RSZ1_RSZ1_RSZ2

### Backsubstitution after applying RS history:
0: -0.0039095, -0.0003029, -0.0039095, -0.0003029, -0.0035887, 0.0036066
1: -0.0044907, 0.0040812, -0.0044907, 0.0040812, -0.0056998, 0.0055753
2: 0.0029569, 0.0099431, 0.0029569, 0.0099431, -0.0045862, 0.0045007
3: -0.0045519, -0.0034923, -0.0045519, -0.0034923, -0.0010595, 0.0010595
4: 0.0017018, 0.0078700, 0.0017018, 0.0078700, -0.0053257, 0.0053688
5: -0.0030341, 0.0031953, -0.0030341, 0.0031953, -0.0049352, 0.0048862
6: -0.0068516, -0.0033211, -0.0068516, -0.0033211, -0.0033600, 0.0033489
7: -0.0021570, 0.0038259, -0.0021570, 0.0038259, -0.0058387, 0.0058840
8: -0.0008533, 0.0000782, -0.0008533, 0.0000782, -0.0009065, 0.0009194
9: 0.9973557, 1.0121032, 0.9973557, 1.0121032, -0.0099834, 0.0101558

### Unstable ReLU Count (Linear/Conv2D Layers)
- layer_idx=0, type=LayerType.Linear, total=256, inp1_unstable=7, inp2_unstable=7, delta_unstable=252
- layer_idx=2, type=LayerType.Linear, total=256, inp1_unstable=10, inp2_unstable=10, delta_unstable=256
- layer_idx=4, type=LayerType.Linear, total=256, inp1_unstable=1, inp2_unstable=1, delta_unstable=256
- layer_idx=6, type=LayerType.Linear, total=256, inp1_unstable=0, inp2_unstable=0, delta_unstable=256
- layer_idx=8, type=LayerType.Linear, total=10, inp1_unstable=4, inp2_unstable=4, delta_unstable=10

Time for backsubstitution: 1.25 seconds

### RS candidates at layer 1
type: RSZ, layer: 1, pos: 140
type: RSZ, layer: 1, pos: 148
type: RSZ, layer: 1, pos: 219

Time for candidate selection: 0.14 seconds

### Candidate
type: RSZ, layer: 1, pos: 140

### Relational analysis RSZ of RS_RSZ2_RSZ1_RSZ1_RSZ2_RSZ1

#### Relational analysis RSZ result of RS_RSZ2_RSZ1_RSZ1_RSZ2_RSZ1
Status: Status.UNKNOWN
Output dim: 9, lower bound: -0.0050725, upper bound: 0.0055543
time: 0.80 seconds

### Relational analysis RSZ of RS_RSZ2_RSZ1_RSZ1_RSZ2_RSZ2

#### Relational analysis RSZ result of RS_RSZ2_RSZ1_RSZ1_RSZ2_RSZ2
Status: Status.UNKNOWN
Output dim: 9, lower bound: -0.0051660, upper bound: 0.0054290
time: 0.77 seconds

## BFS RS instance: RS_RSZ2_RSZ1_RSZ2_RSZ1

### Backsubstitution after applying RS history:
0: -0.0039095, -0.0003029, -0.0039095, -0.0003029, -0.0036066, 0.0035890
1: -0.0044907, 0.0040812, -0.0044907, 0.0040812, -0.0055575, 0.0057109
2: 0.0029569, 0.0099431, 0.0029569, 0.0099431, -0.0044880, 0.0045908
3: -0.0045519, -0.0034923, -0.0045519, -0.0034923, -0.0010595, 0.0010595
4: 0.0017018, 0.0078700, 0.0017018, 0.0078700, -0.0053678, 0.0053267
5: -0.0030341, 0.0031953, -0.0030341, 0.0031953, -0.0048745, 0.0049498
6: -0.0068516, -0.0033211, -0.0068516, -0.0033211, -0.0033477, 0.0033608
7: -0.0021570, 0.0038259, -0.0021570, 0.0038259, -0.0058565, 0.0058649
8: -0.0008533, 0.0000782, -0.0008533, 0.0000782, -0.0008881, 0.0009315
9: 0.9973557, 1.0121032, 0.9973557, 1.0121032, -0.0101644, 0.0099718

### Unstable ReLU Count (Linear/Conv2D Layers)
- layer_idx=0, type=LayerType.Linear, total=256, inp1_unstable=7, inp2_unstable=7, delta_unstable=252
- layer_idx=2, type=LayerType.Linear, total=256, inp1_unstable=10, inp2_unstable=10, delta_unstable=256
- layer_idx=4, type=LayerType.Linear, total=256, inp1_unstable=1, inp2_unstable=1, delta_unstable=256
- layer_idx=6, type=LayerType.Linear, total=256, inp1_unstable=0, inp2_unstable=0, delta_unstable=256
- layer_idx=8, type=LayerType.Linear, total=10, inp1_unstable=4, inp2_unstable=4, delta_unstable=10

Time for backsubstitution: 1.27 seconds

### RS candidates at layer 1
type: RSZ, layer: 1, pos: 140
type: RSZ, layer: 1, pos: 148
type: RSZ, layer: 1, pos: 219

Time for candidate selection: 0.12 seconds

### Candidate
type: RSZ, layer: 1, pos: 140

### Relational analysis RSZ of RS_RSZ2_RSZ1_RSZ2_RSZ1_RSZ1

#### Relational analysis RSZ result of RS_RSZ2_RSZ1_RSZ2_RSZ1_RSZ1
Status: Status.UNKNOWN
Output dim: 9, lower bound: -0.0054378, upper bound: 0.0051218
time: 0.82 seconds

### Relational analysis RSZ of RS_RSZ2_RSZ1_RSZ2_RSZ1_RSZ2

#### Relational analysis RSZ result of RS_RSZ2_RSZ1_RSZ2_RSZ1_RSZ2
Status: Status.UNKNOWN
Output dim: 9, lower bound: -0.0056121, upper bound: 0.0050358
time: 0.76 seconds

## BFS RS instance: RS_RSZ2_RSZ1_RSZ2_RSZ2

### Backsubstitution after applying RS history:
0: -0.0039095, -0.0003029, -0.0039095, -0.0003029, -0.0035719, 0.0036066
1: -0.0044907, 0.0040812, -0.0044907, 0.0040812, -0.0057217, 0.0055412
2: 0.0029569, 0.0099431, 0.0029569, 0.0099431, -0.0046002, 0.0044817
3: -0.0045519, -0.0034923, -0.0045519, -0.0034923, -0.0010595, 0.0010595
4: 0.0017018, 0.0078700, 0.0017018, 0.0078700, -0.0053128, 0.0053784
5: -0.0030341, 0.0031953, -0.0030341, 0.0031953, -0.0049498, 0.0048701
6: -0.0068516, -0.0033211, -0.0068516, -0.0033211, -0.0033622, 0.0033456
7: -0.0021570, 0.0038259, -0.0021570, 0.0038259, -0.0058315, 0.0058903
8: -0.0008533, 0.0000782, -0.0008533, 0.0000782, -0.0008965, 0.0009301
9: 0.9973557, 1.0121032, 0.9973557, 1.0121032, -0.0099309, 0.0101918

### Unstable ReLU Count (Linear/Conv2D Layers)
- layer_idx=0, type=LayerType.Linear, total=256, inp1_unstable=7, inp2_unstable=7, delta_unstable=252
- layer_idx=2, type=LayerType.Linear, total=256, inp1_unstable=10, inp2_unstable=10, delta_unstable=256
- layer_idx=4, type=LayerType.Linear, total=256, inp1_unstable=1, inp2_unstable=1, delta_unstable=256
- layer_idx=6, type=LayerType.Linear, total=256, inp1_unstable=0, inp2_unstable=0, delta_unstable=256
- layer_idx=8, type=LayerType.Linear, total=10, inp1_unstable=4, inp2_unstable=4, delta_unstable=10

Time for backsubstitution: 1.28 seconds

### RS candidates at layer 1
type: RSZ, layer: 1, pos: 140
type: RSZ, layer: 1, pos: 148
type: RSZ, layer: 1, pos: 219

Time for candidate selection: 0.14 seconds

### Candidate
type: RSZ, layer: 1, pos: 140

### Relational analysis RSZ of RS_RSZ2_RSZ1_RSZ2_RSZ2_RSZ1

#### Relational analysis RSZ result of RS_RSZ2_RSZ1_RSZ2_RSZ2_RSZ1
Status: Status.UNKNOWN
Output dim: 9, lower bound: -0.0050013, upper bound: 0.0055684
time: 0.83 seconds

### Relational analysis RSZ of RS_RSZ2_RSZ1_RSZ2_RSZ2_RSZ2

#### Relational analysis RSZ result of RS_RSZ2_RSZ1_RSZ2_RSZ2_RSZ2
Status: Status.UNKNOWN
Output dim: 9, lower bound: -0.0051332, upper bound: 0.0054645
time: 0.80 seconds

## BFS RS instance: RS_RSZ2_RSZ2_RSZ1_RSZ1

### Backsubstitution after applying RS history:
0: -0.0039095, -0.0003029, -0.0039095, -0.0003029, -0.0036066, 0.0035732
1: -0.0044907, 0.0040812, -0.0044907, 0.0040812, -0.0055490, 0.0057132
2: 0.0029569, 0.0099431, 0.0029569, 0.0099431, -0.0044887, 0.0045911
3: -0.0045519, -0.0034923, -0.0045519, -0.0034923, -0.0010595, 0.0010595
4: 0.0017018, 0.0078700, 0.0017018, 0.0078700, -0.0053772, 0.0053141
5: -0.0030341, 0.0031953, -0.0030341, 0.0031953, -0.0048710, 0.0049491
6: -0.0068516, -0.0033211, -0.0068516, -0.0033211, -0.0033462, 0.0033616
7: -0.0021570, 0.0038259, -0.0021570, 0.0038259, -0.0058836, 0.0058380
8: -0.0008533, 0.0000782, -0.0008533, 0.0000782, -0.0009295, 0.0008960
9: 0.9973557, 1.0121032, 0.9973557, 1.0121032, -0.0101834, 0.0099399

### Unstable ReLU Count (Linear/Conv2D Layers)
- layer_idx=0, type=LayerType.Linear, total=256, inp1_unstable=7, inp2_unstable=7, delta_unstable=252
- layer_idx=2, type=LayerType.Linear, total=256, inp1_unstable=10, inp2_unstable=10, delta_unstable=256
- layer_idx=4, type=LayerType.Linear, total=256, inp1_unstable=1, inp2_unstable=1, delta_unstable=256
- layer_idx=6, type=LayerType.Linear, total=256, inp1_unstable=0, inp2_unstable=0, delta_unstable=256
- layer_idx=8, type=LayerType.Linear, total=10, inp1_unstable=4, inp2_unstable=4, delta_unstable=10

Time for backsubstitution: 1.33 seconds

### RS candidates at layer 1
type: RSZ, layer: 1, pos: 140
type: RSZ, layer: 1, pos: 148
type: RSZ, layer: 1, pos: 219

Time for candidate selection: 0.15 seconds

### Candidate
type: RSZ, layer: 1, pos: 140

### Relational analysis RSZ of RS_RSZ2_RSZ2_RSZ1_RSZ1_RSZ1

#### Relational analysis RSZ result of RS_RSZ2_RSZ2_RSZ1_RSZ1_RSZ1
Status: Status.UNKNOWN
Output dim: 9, lower bound: -0.0053966, upper bound: 0.0051694
time: 0.82 seconds

### Relational analysis RSZ of RS_RSZ2_RSZ2_RSZ1_RSZ1_RSZ2

#### Relational analysis RSZ result of RS_RSZ2_RSZ2_RSZ1_RSZ1_RSZ2
Status: Status.UNKNOWN
Output dim: 9, lower bound: -0.0055089, upper bound: 0.0050476
time: 0.79 seconds

## BFS RS instance: RS_RSZ2_RSZ2_RSZ1_RSZ2

### Backsubstitution after applying RS history:
0: -0.0039095, -0.0003029, -0.0039095, -0.0003029, -0.0035874, 0.0036066
1: -0.0044907, 0.0040812, -0.0044907, 0.0040812, -0.0057182, 0.0055492
2: 0.0029569, 0.0099431, 0.0029569, 0.0099431, -0.0045964, 0.0044797
3: -0.0045519, -0.0034923, -0.0045519, -0.0034923, -0.0010595, 0.0010595
4: 0.0017018, 0.0078700, 0.0017018, 0.0078700, -0.0053254, 0.0053690
5: -0.0030341, 0.0031953, -0.0030341, 0.0031953, -0.0049503, 0.0048737
6: -0.0068516, -0.0033211, -0.0068516, -0.0033211, -0.0033614, 0.0033471
7: -0.0021570, 0.0038259, -0.0021570, 0.0038259, -0.0058573, 0.0058630
8: -0.0008533, 0.0000782, -0.0008533, 0.0000782, -0.0009315, 0.0008881
9: 0.9973557, 1.0121032, 0.9973557, 1.0121032, -0.0099635, 0.0101721

### Unstable ReLU Count (Linear/Conv2D Layers)
- layer_idx=0, type=LayerType.Linear, total=256, inp1_unstable=7, inp2_unstable=7, delta_unstable=252
- layer_idx=2, type=LayerType.Linear, total=256, inp1_unstable=10, inp2_unstable=10, delta_unstable=256
- layer_idx=4, type=LayerType.Linear, total=256, inp1_unstable=1, inp2_unstable=1, delta_unstable=256
- layer_idx=6, type=LayerType.Linear, total=256, inp1_unstable=0, inp2_unstable=0, delta_unstable=256
- layer_idx=8, type=LayerType.Linear, total=10, inp1_unstable=4, inp2_unstable=4, delta_unstable=10

Time for backsubstitution: 1.29 seconds

### RS candidates at layer 1
type: RSZ, layer: 1, pos: 140
type: RSZ, layer: 1, pos: 148
type: RSZ, layer: 1, pos: 219

Time for candidate selection: 0.13 seconds

### Candidate
type: RSZ, layer: 1, pos: 140

### Relational analysis RSZ of RS_RSZ2_RSZ2_RSZ1_RSZ2_RSZ1

#### Relational analysis RSZ result of RS_RSZ2_RSZ2_RSZ1_RSZ2_RSZ1
Status: Status.UNKNOWN
Output dim: 9, lower bound: -0.0049505, upper bound: 0.0056423
time: 0.80 seconds

### Relational analysis RSZ of RS_RSZ2_RSZ2_RSZ1_RSZ2_RSZ2

#### Relational analysis RSZ result of RS_RSZ2_RSZ2_RSZ1_RSZ2_RSZ2
Status: Status.UNKNOWN
Output dim: 9, lower bound: -0.0050420, upper bound: 0.0054782
time: 0.82 seconds

## BFS RS instance: RS_RSZ2_RSZ2_RSZ2_RSZ1

### Backsubstitution after applying RS history:
0: -0.0039095, -0.0003029, -0.0039095, -0.0003029, -0.0036066, 0.0035903
1: -0.0044907, 0.0040812, -0.0044907, 0.0040812, -0.0055843, 0.0056923
2: 0.0029569, 0.0099431, 0.0029569, 0.0099431, -0.0045086, 0.0045777
3: -0.0045519, -0.0034923, -0.0045519, -0.0034923, -0.0010595, 0.0010595
4: 0.0017018, 0.0078700, 0.0017018, 0.0078700, -0.0053675, 0.0053269
5: -0.0030341, 0.0031953, -0.0030341, 0.0031953, -0.0048871, 0.0049345
6: -0.0068516, -0.0033211, -0.0068516, -0.0033211, -0.0033495, 0.0033594
7: -0.0021570, 0.0038259, -0.0021570, 0.0038259, -0.0058781, 0.0058456
8: -0.0008533, 0.0000782, -0.0008533, 0.0000782, -0.0009191, 0.0009071
9: 0.9973557, 1.0121032, 0.9973557, 1.0121032, -0.0101484, 0.0099916

### Unstable ReLU Count (Linear/Conv2D Layers)
- layer_idx=0, type=LayerType.Linear, total=256, inp1_unstable=7, inp2_unstable=7, delta_unstable=252
- layer_idx=2, type=LayerType.Linear, total=256, inp1_unstable=10, inp2_unstable=10, delta_unstable=256
- layer_idx=4, type=LayerType.Linear, total=256, inp1_unstable=1, inp2_unstable=1, delta_unstable=256
- layer_idx=6, type=LayerType.Linear, total=256, inp1_unstable=0, inp2_unstable=0, delta_unstable=256
- layer_idx=8, type=LayerType.Linear, total=10, inp1_unstable=4, inp2_unstable=4, delta_unstable=10

Time for backsubstitution: 1.43 seconds

### RS candidates at layer 1
type: RSZ, layer: 1, pos: 140
type: RSZ, layer: 1, pos: 148
type: RSZ, layer: 1, pos: 219

Time for candidate selection: 0.16 seconds

### Candidate
type: RSZ, layer: 1, pos: 140

### Relational analysis RSZ of RS_RSZ2_RSZ2_RSZ2_RSZ1_RSZ1

#### Relational analysis RSZ result of RS_RSZ2_RSZ2_RSZ2_RSZ1_RSZ1
Status: Status.UNKNOWN
Output dim: 9, lower bound: -0.0053588, upper bound: 0.0052212
time: 0.81 seconds

### Relational analysis RSZ of RS_RSZ2_RSZ2_RSZ2_RSZ1_RSZ2

#### Relational analysis RSZ result of RS_RSZ2_RSZ2_RSZ2_RSZ1_RSZ2
Status: Status.UNKNOWN
Output dim: 9, lower bound: -0.0055004, upper bound: 0.0051328
time: 0.73 seconds

## BFS RS instance: RS_RSZ2_RSZ2_RSZ2_RSZ2

### Backsubstitution after applying RS history:
0: -0.0039095, -0.0003029, -0.0039095, -0.0003029, -0.0035705, 0.0036066
1: -0.0044907, 0.0040812, -0.0044907, 0.0040812, -0.0057425, 0.0055162
2: 0.0029569, 0.0099431, 0.0029569, 0.0099431, -0.0046108, 0.0044626
3: -0.0045519, -0.0034923, -0.0045519, -0.0034923, -0.0010595, 0.0010595
4: 0.0017018, 0.0078700, 0.0017018, 0.0078700, -0.0053125, 0.0053787
5: -0.0030341, 0.0031953, -0.0030341, 0.0031953, -0.0049635, 0.0048568
6: -0.0068516, -0.0033211, -0.0068516, -0.0033211, -0.0033637, 0.0033439
7: -0.0021570, 0.0038259, -0.0021570, 0.0038259, -0.0058514, 0.0058689
8: -0.0008533, 0.0000782, -0.0008533, 0.0000782, -0.0009273, 0.0008967
9: 0.9973557, 1.0121032, 0.9973557, 1.0121032, -0.0099115, 0.0102090

### Unstable ReLU Count (Linear/Conv2D Layers)
- layer_idx=0, type=LayerType.Linear, total=256, inp1_unstable=7, inp2_unstable=7, delta_unstable=252
- layer_idx=2, type=LayerType.Linear, total=256, inp1_unstable=10, inp2_unstable=10, delta_unstable=256
- layer_idx=4, type=LayerType.Linear, total=256, inp1_unstable=1, inp2_unstable=1, delta_unstable=256
- layer_idx=6, type=LayerType.Linear, total=256, inp1_unstable=0, inp2_unstable=0, delta_unstable=256
- layer_idx=8, type=LayerType.Linear, total=10, inp1_unstable=4, inp2_unstable=4, delta_unstable=10

Time for backsubstitution: 1.29 seconds

### RS candidates at layer 1
type: RSZ, layer: 1, pos: 140
type: RSZ, layer: 1, pos: 148
type: RSZ, layer: 1, pos: 219

Time for candidate selection: 0.14 seconds

### Candidate
type: RSZ, layer: 1, pos: 140

### Relational analysis RSZ of RS_RSZ2_RSZ2_RSZ2_RSZ2_RSZ1

#### Relational analysis RSZ result of RS_RSZ2_RSZ2_RSZ2_RSZ2_RSZ1
Status: Status.UNKNOWN
Output dim: 9, lower bound: -0.0048847, upper bound: 0.0056671
time: 0.88 seconds

### Relational analysis RSZ of RS_RSZ2_RSZ2_RSZ2_RSZ2_RSZ2

#### Relational analysis RSZ result of RS_RSZ2_RSZ2_RSZ2_RSZ2_RSZ2
Status: Status.UNKNOWN
Output dim: 9, lower bound: -0.0050108, upper bound: 0.0055349
time: 0.80 seconds

## Summary of splitting (split count: 4)
- Time for RS candidates: 3.12 seconds
RS_RSZ1_RSZ1_RSZ1_RSZ1_RSZ1, status: Status.UNKNOWN, split count: 5, time: 3.12
Output dim: 9, lower bound: -0.0055349, upper bound: 0.0050108
RS_RSZ1_RSZ1_RSZ1_RSZ1_RSZ2, status: Status.UNKNOWN, split count: 5, time: 3.12
Output dim: 9, lower bound: -0.0056671, upper bound: 0.0048847
RS_RSZ1_RSZ1_RSZ1_RSZ2_RSZ1, status: Status.UNKNOWN, split count: 5, time: 3.12
Output dim: 9, lower bound: -0.0051328, upper bound: 0.0055004
RS_RSZ1_RSZ1_RSZ1_RSZ2_RSZ2, status: Status.UNKNOWN, split count: 5, time: 3.12
Output dim: 9, lower bound: -0.0052212, upper bound: 0.0053588
RS_RSZ1_RSZ1_RSZ2_RSZ1_RSZ1, status: Status.UNKNOWN, split count: 5, time: 3.12
Output dim: 9, lower bound: -0.0054781, upper bound: 0.0050420
RS_RSZ1_RSZ1_RSZ2_RSZ1_RSZ2, status: Status.UNKNOWN, split count: 5, time: 3.12
Output dim: 9, lower bound: -0.0056423, upper bound: 0.0049505
RS_RSZ1_RSZ1_RSZ2_RSZ2_RSZ1, status: Status.UNKNOWN, split count: 5, time: 3.12
Output dim: 9, lower bound: -0.0050475, upper bound: 0.0055089
RS_RSZ1_RSZ1_RSZ2_RSZ2_RSZ2, status: Status.UNKNOWN, split count: 5, time: 3.12
Output dim: 9, lower bound: -0.0051694, upper bound: 0.0053966
RS_RSZ1_RSZ2_RSZ1_RSZ1_RSZ1, status: Status.UNKNOWN, split count: 5, time: 3.12
Output dim: 9, lower bound: -0.0054645, upper bound: 0.0051332
RS_RSZ1_RSZ2_RSZ1_RSZ1_RSZ2, status: Status.UNKNOWN, split count: 5, time: 3.12
Output dim: 9, lower bound: -0.0055684, upper bound: 0.0050013
RS_RSZ1_RSZ2_RSZ1_RSZ2_RSZ1, status: Status.UNKNOWN, split count: 5, time: 3.12
Output dim: 9, lower bound: -0.0050358, upper bound: 0.0056121
RS_RSZ1_RSZ2_RSZ1_RSZ2_RSZ2, status: Status.UNKNOWN, split count: 5, time: 3.12
Output dim: 9, lower bound: -0.0051218, upper bound: 0.0054378
RS_RSZ1_RSZ2_RSZ2_RSZ1_RSZ1, status: Status.UNKNOWN, split count: 5, time: 3.12
Output dim: 9, lower bound: -0.0054290, upper bound: 0.0051660
RS_RSZ1_RSZ2_RSZ2_RSZ1_RSZ2, status: Status.UNKNOWN, split count: 5, time: 3.12
Output dim: 9, lower bound: -0.0055543, upper bound: 0.0050725
RS_RSZ1_RSZ2_RSZ2_RSZ2_RSZ1, status: Status.UNKNOWN, split count: 5, time: 3.12
Output dim: 9, lower bound: -0.0049516, upper bound: 0.0056326
RS_RSZ1_RSZ2_RSZ2_RSZ2_RSZ2, status: Status.UNKNOWN, split count: 5, time: 3.12
Output dim: 9, lower bound: -0.0050636, upper bound: 0.0054938
RS_RSZ2_RSZ1_RSZ1_RSZ1_RSZ1, status: Status.UNKNOWN, split count: 5, time: 3.12
Output dim: 9, lower bound: -0.0054938, upper bound: 0.0050636
RS_RSZ2_RSZ1_RSZ1_RSZ1_RSZ2, status: Status.UNKNOWN, split count: 5, time: 3.12
Output dim: 9, lower bound: -0.0056326, upper bound: 0.0049517
RS_RSZ2_RSZ1_RSZ1_RSZ2_RSZ1, status: Status.UNKNOWN, split count: 5, time: 3.12
Output dim: 9, lower bound: -0.0050725, upper bound: 0.0055543
RS_RSZ2_RSZ1_RSZ1_RSZ2_RSZ2, status: Status.UNKNOWN, split count: 5, time: 3.12
Output dim: 9, lower bound: -0.0051660, upper bound: 0.0054290
RS_RSZ2_RSZ1_RSZ2_RSZ1_RSZ1, status: Status.UNKNOWN, split count: 5, time: 3.12
Output dim: 9, lower bound: -0.0054378, upper bound: 0.0051218
RS_RSZ2_RSZ1_RSZ2_RSZ1_RSZ2, status: Status.UNKNOWN, split count: 5, time: 3.12
Output dim: 9, lower bound: -0.0056121, upper bound: 0.0050358
RS_RSZ2_RSZ1_RSZ2_RSZ2_RSZ1, status: Status.UNKNOWN, split count: 5, time: 3.12
Output dim: 9, lower bound: -0.0050013, upper bound: 0.0055684
RS_RSZ2_RSZ1_RSZ2_RSZ2_RSZ2, status: Status.UNKNOWN, split count: 5, time: 3.12
Output dim: 9, lower bound: -0.0051332, upper bound: 0.0054645
RS_RSZ2_RSZ2_RSZ1_RSZ1_RSZ1, status: Status.UNKNOWN, split count: 5, time: 3.12
Output dim: 9, lower bound: -0.0053966, upper bound: 0.0051694
RS_RSZ2_RSZ2_RSZ1_RSZ1_RSZ2, status: Status.UNKNOWN, split count: 5, time: 3.12
Output dim: 9, lower bound: -0.0055089, upper bound: 0.0050476
RS_RSZ2_RSZ2_RSZ1_RSZ2_RSZ1, status: Status.UNKNOWN, split count: 5, time: 3.12
Output dim: 9, lower bound: -0.0049505, upper bound: 0.0056423
RS_RSZ2_RSZ2_RSZ1_RSZ2_RSZ2, status: Status.UNKNOWN, split count: 5, time: 3.12
Output dim: 9, lower bound: -0.0050420, upper bound: 0.0054782
RS_RSZ2_RSZ2_RSZ2_RSZ1_RSZ1, status: Status.UNKNOWN, split count: 5, time: 3.12
Output dim: 9, lower bound: -0.0053588, upper bound: 0.0052212
RS_RSZ2_RSZ2_RSZ2_RSZ1_RSZ2, status: Status.UNKNOWN, split count: 5, time: 3.12
Output dim: 9, lower bound: -0.0055004, upper bound: 0.0051328
RS_RSZ2_RSZ2_RSZ2_RSZ2_RSZ1, status: Status.UNKNOWN, split count: 5, time: 3.12
Output dim: 9, lower bound: -0.0048847, upper bound: 0.0056671
RS_RSZ2_RSZ2_RSZ2_RSZ2_RSZ2, status: Status.UNKNOWN, split count: 5, time: 3.12
Output dim: 9, lower bound: -0.0050108, upper bound: 0.0055349

## BFS RS instance: RS_RSZ1_RSZ1_RSZ1_RSZ1_RSZ1

### Backsubstitution after applying RS history:
0: -0.0039095, -0.0003029, -0.0039095, -0.0003029, -0.0032025, 0.0031721
1: -0.0044907, 0.0040812, -0.0044907, 0.0040812, -0.0038948, 0.0039817
2: 0.0029569, 0.0099431, 0.0029569, 0.0099431, -0.0038746, 0.0039270
3: -0.0045519, -0.0034923, -0.0045519, -0.0034923, -0.0010595, 0.0010595
4: 0.0017018, 0.0078700, 0.0017018, 0.0078700, -0.0053073, 0.0052848
5: -0.0030341, 0.0031953, -0.0030341, 0.0031953, -0.0047482, 0.0047933
6: -0.0068516, -0.0033211, -0.0068516, -0.0033211, -0.0033497, 0.0033575
7: -0.0021570, 0.0038259, -0.0021570, 0.0038259, -0.0057976, 0.0058091
8: -0.0008533, 0.0000782, -0.0008533, 0.0000782, -0.0008644, 0.0008882
9: 0.9973557, 1.0121032, 0.9973557, 1.0121032, -0.0092222, 0.0091078

### Unstable ReLU Count (Linear/Conv2D Layers)
- layer_idx=0, type=LayerType.Linear, total=256, inp1_unstable=7, inp2_unstable=7, delta_unstable=251
- layer_idx=2, type=LayerType.Linear, total=256, inp1_unstable=10, inp2_unstable=10, delta_unstable=256
- layer_idx=4, type=LayerType.Linear, total=256, inp1_unstable=1, inp2_unstable=1, delta_unstable=256
- layer_idx=6, type=LayerType.Linear, total=256, inp1_unstable=0, inp2_unstable=0, delta_unstable=256
- layer_idx=8, type=LayerType.Linear, total=10, inp1_unstable=4, inp2_unstable=4, delta_unstable=10

Time for backsubstitution: 1.28 seconds

### RS candidates at layer 1
type: RSZ, layer: 1, pos: 148
type: RSZ, layer: 1, pos: 219

Time for candidate selection: 0.15 seconds

### Candidate
type: RSZ, layer: 1, pos: 148

### Relational analysis RSZ of RS_RSZ1_RSZ1_RSZ1_RSZ1_RSZ1_RSZ1

#### Relational analysis RSZ result of RS_RSZ1_RSZ1_RSZ1_RSZ1_RSZ1_RSZ1
Status: Status.UNKNOWN
Output dim: 9, lower bound: -0.0053739, upper bound: 0.0048870
time: 0.85 seconds

### Relational analysis RSZ of RS_RSZ1_RSZ1_RSZ1_RSZ1_RSZ1_RSZ2

#### Relational analysis RSZ result of RS_RSZ1_RSZ1_RSZ1_RSZ1_RSZ1_RSZ2
Status: Status.UNKNOWN
Output dim: 9, lower bound: -0.0053739, upper bound: 0.0048870
time: 0.82 seconds

## BFS RS instance: RS_RSZ1_RSZ1_RSZ1_RSZ1_RSZ2

### Backsubstitution after applying RS history:
0: -0.0039095, -0.0003029, -0.0039095, -0.0003029, -0.0032511, 0.0031152
1: -0.0044907, 0.0040812, -0.0044907, 0.0040812, -0.0037555, 0.0041021
2: 0.0029569, 0.0099431, 0.0029569, 0.0099431, -0.0037788, 0.0040102
3: -0.0045519, -0.0034923, -0.0045519, -0.0034923, -0.0010595, 0.0010595
4: 0.0017018, 0.0078700, 0.0017018, 0.0078700, -0.0053441, 0.0052411
5: -0.0030341, 0.0031953, -0.0030341, 0.0031953, -0.0046865, 0.0048522
6: -0.0068516, -0.0033211, -0.0068516, -0.0033211, -0.0033377, 0.0033678
7: -0.0021570, 0.0038259, -0.0021570, 0.0038259, -0.0058264, 0.0057801
8: -0.0008533, 0.0000782, -0.0008533, 0.0000782, -0.0008576, 0.0008948
9: 0.9973557, 1.0121032, 0.9973557, 1.0121032, -0.0093791, 0.0089247

### Unstable ReLU Count (Linear/Conv2D Layers)
- layer_idx=0, type=LayerType.Linear, total=256, inp1_unstable=7, inp2_unstable=7, delta_unstable=251
- layer_idx=2, type=LayerType.Linear, total=256, inp1_unstable=10, inp2_unstable=10, delta_unstable=256
- layer_idx=4, type=LayerType.Linear, total=256, inp1_unstable=1, inp2_unstable=1, delta_unstable=256
- layer_idx=6, type=LayerType.Linear, total=256, inp1_unstable=0, inp2_unstable=0, delta_unstable=256
- layer_idx=8, type=LayerType.Linear, total=10, inp1_unstable=4, inp2_unstable=4, delta_unstable=10

Time for backsubstitution: 1.27 seconds

### RS candidates at layer 1
type: RSZ, layer: 1, pos: 148
type: RSZ, layer: 1, pos: 219

Time for candidate selection: 0.14 seconds

### Candidate
type: RSZ, layer: 1, pos: 148

### Relational analysis RSZ of RS_RSZ1_RSZ1_RSZ1_RSZ1_RSZ2_RSZ1

#### Relational analysis RSZ result of RS_RSZ1_RSZ1_RSZ1_RSZ1_RSZ2_RSZ1
Status: Status.UNKNOWN
Output dim: 9, lower bound: -0.0055117, upper bound: 0.0047578
time: 0.77 seconds

### Relational analysis RSZ of RS_RSZ1_RSZ1_RSZ1_RSZ1_RSZ2_RSZ2

#### Relational analysis RSZ result of RS_RSZ1_RSZ1_RSZ1_RSZ1_RSZ2_RSZ2
Status: Status.UNKNOWN
Output dim: 9, lower bound: -0.0055117, upper bound: 0.0047578
time: 0.80 seconds

## BFS RS instance: RS_RSZ1_RSZ1_RSZ1_RSZ2_RSZ1

### Backsubstitution after applying RS history:
0: -0.0039095, -0.0003029, -0.0039095, -0.0003029, -0.0031349, 0.0032424
1: -0.0044907, 0.0040812, -0.0044907, 0.0040812, -0.0040571, 0.0038235
2: 0.0029569, 0.0099431, 0.0029569, 0.0099431, -0.0039782, 0.0038248
3: -0.0045519, -0.0034923, -0.0045519, -0.0034923, -0.0010595, 0.0010595
4: 0.0017018, 0.0078700, 0.0017018, 0.0078700, -0.0052556, 0.0053385
5: -0.0030341, 0.0031953, -0.0030341, 0.0031953, -0.0048277, 0.0047168
6: -0.0068516, -0.0033211, -0.0068516, -0.0033211, -0.0033644, 0.0033433
7: -0.0021570, 0.0038259, -0.0021570, 0.0038259, -0.0057744, 0.0058335
8: -0.0008533, 0.0000782, -0.0008533, 0.0000782, -0.0008754, 0.0008800
9: 0.9973557, 1.0121032, 0.9973557, 1.0121032, -0.0090048, 0.0093329

### Unstable ReLU Count (Linear/Conv2D Layers)
- layer_idx=0, type=LayerType.Linear, total=256, inp1_unstable=7, inp2_unstable=7, delta_unstable=251
- layer_idx=2, type=LayerType.Linear, total=256, inp1_unstable=10, inp2_unstable=10, delta_unstable=256
- layer_idx=4, type=LayerType.Linear, total=256, inp1_unstable=1, inp2_unstable=1, delta_unstable=256
- layer_idx=6, type=LayerType.Linear, total=256, inp1_unstable=0, inp2_unstable=0, delta_unstable=256
- layer_idx=8, type=LayerType.Linear, total=10, inp1_unstable=4, inp2_unstable=4, delta_unstable=10

Time for backsubstitution: 1.32 seconds

### RS candidates at layer 1
type: RSZ, layer: 1, pos: 148
type: RSZ, layer: 1, pos: 219

Time for candidate selection: 0.14 seconds

### Candidate
type: RSZ, layer: 1, pos: 148

### Relational analysis RSZ of RS_RSZ1_RSZ1_RSZ1_RSZ2_RSZ1_RSZ1

#### Relational analysis RSZ result of RS_RSZ1_RSZ1_RSZ1_RSZ2_RSZ1_RSZ1
Status: Status.UNKNOWN
Output dim: 9, lower bound: -0.0049980, upper bound: 0.0053686
time: 0.86 seconds

### Relational analysis RSZ of RS_RSZ1_RSZ1_RSZ1_RSZ2_RSZ1_RSZ2

#### Relational analysis RSZ result of RS_RSZ1_RSZ1_RSZ1_RSZ2_RSZ1_RSZ2
Status: Status.UNKNOWN
Output dim: 9, lower bound: -0.0049980, upper bound: 0.0053686
time: 0.86 seconds

## BFS RS instance: RS_RSZ1_RSZ1_RSZ1_RSZ2_RSZ2

### Backsubstitution after applying RS history:
0: -0.0039095, -0.0003029, -0.0039095, -0.0003029, -0.0031820, 0.0031868
1: -0.0044907, 0.0040812, -0.0044907, 0.0040812, -0.0039315, 0.0039497
2: 0.0029569, 0.0099431, 0.0029569, 0.0099431, -0.0038939, 0.0039121
3: -0.0045519, -0.0034923, -0.0045519, -0.0034923, -0.0010595, 0.0010595
4: 0.0017018, 0.0078700, 0.0017018, 0.0078700, -0.0052916, 0.0052961
5: -0.0030341, 0.0031953, -0.0030341, 0.0031953, -0.0047642, 0.0047755
6: -0.0068516, -0.0033211, -0.0068516, -0.0033211, -0.0033532, 0.0033541
7: -0.0021570, 0.0038259, -0.0021570, 0.0038259, -0.0058042, 0.0058068
8: -0.0008533, 0.0000782, -0.0008533, 0.0000782, -0.0008680, 0.0008848
9: 0.9973557, 1.0121032, 0.9973557, 1.0121032, -0.0091655, 0.0091616

### Unstable ReLU Count (Linear/Conv2D Layers)
- layer_idx=0, type=LayerType.Linear, total=256, inp1_unstable=7, inp2_unstable=7, delta_unstable=251
- layer_idx=2, type=LayerType.Linear, total=256, inp1_unstable=10, inp2_unstable=10, delta_unstable=256
- layer_idx=4, type=LayerType.Linear, total=256, inp1_unstable=1, inp2_unstable=1, delta_unstable=256
- layer_idx=6, type=LayerType.Linear, total=256, inp1_unstable=0, inp2_unstable=0, delta_unstable=256
- layer_idx=8, type=LayerType.Linear, total=10, inp1_unstable=4, inp2_unstable=4, delta_unstable=10

Time for backsubstitution: 1.26 seconds

### RS candidates at layer 1
type: RSZ, layer: 1, pos: 148
type: RSZ, layer: 1, pos: 219

Time for candidate selection: 0.13 seconds

### Candidate
type: RSZ, layer: 1, pos: 148

### Relational analysis RSZ of RS_RSZ1_RSZ1_RSZ1_RSZ2_RSZ2_RSZ1

#### Relational analysis RSZ result of RS_RSZ1_RSZ1_RSZ1_RSZ2_RSZ2_RSZ1
Status: Status.UNKNOWN
Output dim: 9, lower bound: -0.0050919, upper bound: 0.0052293
time: 0.81 seconds

### Relational analysis RSZ of RS_RSZ1_RSZ1_RSZ1_RSZ2_RSZ2_RSZ2

#### Relational analysis RSZ result of RS_RSZ1_RSZ1_RSZ1_RSZ2_RSZ2_RSZ2
Status: Status.UNKNOWN
Output dim: 9, lower bound: -0.0050919, upper bound: 0.0052293
time: 0.82 seconds

## BFS RS instance: RS_RSZ1_RSZ1_RSZ2_RSZ1_RSZ1

### Backsubstitution after applying RS history:
0: -0.0039095, -0.0003029, -0.0039095, -0.0003029, -0.0031893, 0.0031783
1: -0.0044907, 0.0040812, -0.0044907, 0.0040812, -0.0039074, 0.0039574
2: 0.0029569, 0.0099431, 0.0029569, 0.0099431, -0.0038830, 0.0039125
3: -0.0045519, -0.0034923, -0.0045519, -0.0034923, -0.0010595, 0.0010595
4: 0.0017018, 0.0078700, 0.0017018, 0.0078700, -0.0052976, 0.0052894
5: -0.0030341, 0.0031953, -0.0030341, 0.0031953, -0.0047555, 0.0047801
6: -0.0068516, -0.0033211, -0.0068516, -0.0033211, -0.0033511, 0.0033551
7: -0.0021570, 0.0038259, -0.0021570, 0.0038259, -0.0057917, 0.0058131
8: -0.0008533, 0.0000782, -0.0008533, 0.0000782, -0.0008551, 0.0008994
9: 0.9973557, 1.0121032, 0.9973557, 1.0121032, -0.0091852, 0.0091295

### Unstable ReLU Count (Linear/Conv2D Layers)
- layer_idx=0, type=LayerType.Linear, total=256, inp1_unstable=7, inp2_unstable=7, delta_unstable=251
- layer_idx=2, type=LayerType.Linear, total=256, inp1_unstable=10, inp2_unstable=10, delta_unstable=256
- layer_idx=4, type=LayerType.Linear, total=256, inp1_unstable=1, inp2_unstable=1, delta_unstable=256
- layer_idx=6, type=LayerType.Linear, total=256, inp1_unstable=0, inp2_unstable=0, delta_unstable=256
- layer_idx=8, type=LayerType.Linear, total=10, inp1_unstable=4, inp2_unstable=4, delta_unstable=10

Time for backsubstitution: 1.31 seconds

### RS candidates at layer 1
type: RSZ, layer: 1, pos: 148
type: RSZ, layer: 1, pos: 219

Time for candidate selection: 0.15 seconds

### Candidate
type: RSZ, layer: 1, pos: 148

### Relational analysis RSZ of RS_RSZ1_RSZ1_RSZ2_RSZ1_RSZ1_RSZ1

#### Relational analysis RSZ result of RS_RSZ1_RSZ1_RSZ2_RSZ1_RSZ1_RSZ1
Status: Status.UNKNOWN
Output dim: 9, lower bound: -0.0053411, upper bound: 0.0049153
time: 0.83 seconds

### Relational analysis RSZ of RS_RSZ1_RSZ1_RSZ2_RSZ1_RSZ1_RSZ2

#### Relational analysis RSZ result of RS_RSZ1_RSZ1_RSZ2_RSZ1_RSZ1_RSZ2
Status: Status.UNKNOWN
Output dim: 9, lower bound: -0.0053411, upper bound: 0.0049153
time: 0.82 seconds

## BFS RS instance: RS_RSZ1_RSZ1_RSZ2_RSZ1_RSZ2

### Backsubstitution after applying RS history:
0: -0.0039095, -0.0003029, -0.0039095, -0.0003029, -0.0032454, 0.0031320
1: -0.0044907, 0.0040812, -0.0044907, 0.0040812, -0.0037884, 0.0040875
2: 0.0029569, 0.0099431, 0.0029569, 0.0099431, -0.0037958, 0.0040022
3: -0.0045519, -0.0034923, -0.0045519, -0.0034923, -0.0010595, 0.0010595
4: 0.0017018, 0.0078700, 0.0017018, 0.0078700, -0.0053399, 0.0052540
5: -0.0030341, 0.0031953, -0.0030341, 0.0031953, -0.0047035, 0.0048478
6: -0.0068516, -0.0033211, -0.0068516, -0.0033211, -0.0033409, 0.0033666
7: -0.0021570, 0.0038259, -0.0021570, 0.0038259, -0.0058222, 0.0057861
8: -0.0008533, 0.0000782, -0.0008533, 0.0000782, -0.0008490, 0.0009060
9: 0.9973557, 1.0121032, 0.9973557, 1.0121032, -0.0093615, 0.0089766

### Unstable ReLU Count (Linear/Conv2D Layers)
- layer_idx=0, type=LayerType.Linear, total=256, inp1_unstable=7, inp2_unstable=7, delta_unstable=251
- layer_idx=2, type=LayerType.Linear, total=256, inp1_unstable=10, inp2_unstable=10, delta_unstable=256
- layer_idx=4, type=LayerType.Linear, total=256, inp1_unstable=1, inp2_unstable=1, delta_unstable=256
- layer_idx=6, type=LayerType.Linear, total=256, inp1_unstable=0, inp2_unstable=0, delta_unstable=256
- layer_idx=8, type=LayerType.Linear, total=10, inp1_unstable=4, inp2_unstable=4, delta_unstable=10

Time for backsubstitution: 1.32 seconds

### RS candidates at layer 1
type: RSZ, layer: 1, pos: 148
type: RSZ, layer: 1, pos: 219

Time for candidate selection: 0.13 seconds

### Candidate
type: RSZ, layer: 1, pos: 148

### Relational analysis RSZ of RS_RSZ1_RSZ1_RSZ2_RSZ1_RSZ2_RSZ1

#### Relational analysis RSZ result of RS_RSZ1_RSZ1_RSZ2_RSZ1_RSZ2_RSZ1
Status: Status.UNKNOWN
Output dim: 9, lower bound: -0.0055022, upper bound: 0.0048223
time: 0.80 seconds

### Relational analysis RSZ of RS_RSZ1_RSZ1_RSZ2_RSZ1_RSZ2_RSZ2

#### Relational analysis RSZ result of RS_RSZ1_RSZ1_RSZ2_RSZ1_RSZ2_RSZ2
Status: Status.UNKNOWN
Output dim: 9, lower bound: -0.0055022, upper bound: 0.0048223
time: 0.79 seconds

## BFS RS instance: RS_RSZ1_RSZ1_RSZ2_RSZ2_RSZ1

### Backsubstitution after applying RS history:
0: -0.0039095, -0.0003029, -0.0039095, -0.0003029, -0.0031179, 0.0032479
1: -0.0044907, 0.0040812, -0.0044907, 0.0040812, -0.0040665, 0.0037883
2: 0.0029569, 0.0099431, 0.0029569, 0.0099431, -0.0039850, 0.0038048
3: -0.0045519, -0.0034923, -0.0045519, -0.0034923, -0.0010595, 0.0010595
4: 0.0017018, 0.0078700, 0.0017018, 0.0078700, -0.0052427, 0.0053427
5: -0.0030341, 0.0031953, -0.0030341, 0.0031953, -0.0048320, 0.0047007
6: -0.0068516, -0.0033211, -0.0068516, -0.0033211, -0.0033654, 0.0033400
7: -0.0021570, 0.0038259, -0.0021570, 0.0038259, -0.0057668, 0.0058361
8: -0.0008533, 0.0000782, -0.0008533, 0.0000782, -0.0008640, 0.0008904
9: 0.9973557, 1.0121032, 0.9973557, 1.0121032, -0.0089531, 0.0093472

### Unstable ReLU Count (Linear/Conv2D Layers)
- layer_idx=0, type=LayerType.Linear, total=256, inp1_unstable=7, inp2_unstable=7, delta_unstable=251
- layer_idx=2, type=LayerType.Linear, total=256, inp1_unstable=10, inp2_unstable=10, delta_unstable=256
- layer_idx=4, type=LayerType.Linear, total=256, inp1_unstable=1, inp2_unstable=1, delta_unstable=256
- layer_idx=6, type=LayerType.Linear, total=256, inp1_unstable=0, inp2_unstable=0, delta_unstable=256
- layer_idx=8, type=LayerType.Linear, total=10, inp1_unstable=4, inp2_unstable=4, delta_unstable=10

Time for backsubstitution: 1.24 seconds

### RS candidates at layer 1
type: RSZ, layer: 1, pos: 148
type: RSZ, layer: 1, pos: 219

Time for candidate selection: 0.14 seconds

### Candidate
type: RSZ, layer: 1, pos: 148

### Relational analysis RSZ of RS_RSZ1_RSZ1_RSZ2_RSZ2_RSZ1_RSZ1

#### Relational analysis RSZ result of RS_RSZ1_RSZ1_RSZ2_RSZ2_RSZ1_RSZ1
Status: Status.UNKNOWN
Output dim: 9, lower bound: -0.0049285, upper bound: 0.0053725
time: 0.83 seconds

### Relational analysis RSZ of RS_RSZ1_RSZ1_RSZ2_RSZ2_RSZ1_RSZ2

#### Relational analysis RSZ result of RS_RSZ1_RSZ1_RSZ2_RSZ2_RSZ1_RSZ2
Status: Status.UNKNOWN
Output dim: 9, lower bound: -0.0049285, upper bound: 0.0053725
time: 0.81 seconds

## BFS RS instance: RS_RSZ1_RSZ1_RSZ2_RSZ2_RSZ2

### Backsubstitution after applying RS history:
0: -0.0039095, -0.0003029, -0.0039095, -0.0003029, -0.0031758, 0.0031999
1: -0.0044907, 0.0040812, -0.0044907, 0.0040812, -0.0039525, 0.0039353
2: 0.0029569, 0.0099431, 0.0029569, 0.0099431, -0.0039073, 0.0039034
3: -0.0045519, -0.0034923, -0.0045519, -0.0034923, -0.0010595, 0.0010595
4: 0.0017018, 0.0078700, 0.0017018, 0.0078700, -0.0052868, 0.0053058
5: -0.0030341, 0.0031953, -0.0030341, 0.0031953, -0.0047789, 0.0047691
6: -0.0068516, -0.0033211, -0.0068516, -0.0033211, -0.0033554, 0.0033527
7: -0.0021570, 0.0038259, -0.0021570, 0.0038259, -0.0057992, 0.0058123
8: -0.0008533, 0.0000782, -0.0008533, 0.0000782, -0.0008570, 0.0008954
9: 0.9973557, 1.0121032, 0.9973557, 1.0121032, -0.0091432, 0.0091965

### Unstable ReLU Count (Linear/Conv2D Layers)
- layer_idx=0, type=LayerType.Linear, total=256, inp1_unstable=7, inp2_unstable=7, delta_unstable=251
- layer_idx=2, type=LayerType.Linear, total=256, inp1_unstable=10, inp2_unstable=10, delta_unstable=256
- layer_idx=4, type=LayerType.Linear, total=256, inp1_unstable=1, inp2_unstable=1, delta_unstable=256
- layer_idx=6, type=LayerType.Linear, total=256, inp1_unstable=0, inp2_unstable=0, delta_unstable=256
- layer_idx=8, type=LayerType.Linear, total=10, inp1_unstable=4, inp2_unstable=4, delta_unstable=10

Time for backsubstitution: 1.26 seconds

### RS candidates at layer 1
type: RSZ, layer: 1, pos: 148
type: RSZ, layer: 1, pos: 219

Time for candidate selection: 0.12 seconds

### Candidate
type: RSZ, layer: 1, pos: 148

### Relational analysis RSZ of RS_RSZ1_RSZ1_RSZ2_RSZ2_RSZ2_RSZ1

#### Relational analysis RSZ result of RS_RSZ1_RSZ1_RSZ2_RSZ2_RSZ2_RSZ1
Status: Status.UNKNOWN
Output dim: 9, lower bound: -0.0050524, upper bound: 0.0052487
time: 0.79 seconds

### Relational analysis RSZ of RS_RSZ1_RSZ1_RSZ2_RSZ2_RSZ2_RSZ2

#### Relational analysis RSZ result of RS_RSZ1_RSZ1_RSZ2_RSZ2_RSZ2_RSZ2
Status: Status.UNKNOWN
Output dim: 9, lower bound: -0.0050524, upper bound: 0.0052487
time: 0.80 seconds

## BFS RS instance: RS_RSZ1_RSZ2_RSZ1_RSZ1_RSZ1

### Backsubstitution after applying RS history:
0: -0.0039095, -0.0003029, -0.0039095, -0.0003029, -0.0032013, 0.0031739
1: -0.0044907, 0.0040812, -0.0044907, 0.0040812, -0.0039250, 0.0039610
2: 0.0029569, 0.0099431, 0.0029569, 0.0099431, -0.0038950, 0.0039163
3: -0.0045519, -0.0034923, -0.0045519, -0.0034923, -0.0010595, 0.0010595
4: 0.0017018, 0.0078700, 0.0017018, 0.0078700, -0.0053071, 0.0052851
5: -0.0030341, 0.0031953, -0.0030341, 0.0031953, -0.0047685, 0.0047795
6: -0.0068516, -0.0033211, -0.0068516, -0.0033211, -0.0033519, 0.0033560
7: -0.0021570, 0.0038259, -0.0021570, 0.0038259, -0.0058191, 0.0057957
8: -0.0008533, 0.0000782, -0.0008533, 0.0000782, -0.0008957, 0.0008574
9: 0.9973557, 1.0121032, 0.9973557, 1.0121032, -0.0092050, 0.0091334

### Unstable ReLU Count (Linear/Conv2D Layers)
- layer_idx=0, type=LayerType.Linear, total=256, inp1_unstable=7, inp2_unstable=7, delta_unstable=251
- layer_idx=2, type=LayerType.Linear, total=256, inp1_unstable=10, inp2_unstable=10, delta_unstable=256
- layer_idx=4, type=LayerType.Linear, total=256, inp1_unstable=1, inp2_unstable=1, delta_unstable=256
- layer_idx=6, type=LayerType.Linear, total=256, inp1_unstable=0, inp2_unstable=0, delta_unstable=256
- layer_idx=8, type=LayerType.Linear, total=10, inp1_unstable=4, inp2_unstable=4, delta_unstable=10

Time for backsubstitution: 1.28 seconds

### RS candidates at layer 1
type: RSZ, layer: 1, pos: 148
type: RSZ, layer: 1, pos: 219

Time for candidate selection: 0.14 seconds

### Candidate
type: RSZ, layer: 1, pos: 148

### Relational analysis RSZ of RS_RSZ1_RSZ2_RSZ1_RSZ1_RSZ1_RSZ1

#### Relational analysis RSZ result of RS_RSZ1_RSZ2_RSZ1_RSZ1_RSZ1_RSZ1
Status: Status.UNKNOWN
Output dim: 9, lower bound: -0.0053004, upper bound: 0.0050235
time: 1.01 seconds

### Relational analysis RSZ of RS_RSZ1_RSZ2_RSZ1_RSZ1_RSZ1_RSZ2

#### Relational analysis RSZ result of RS_RSZ1_RSZ2_RSZ1_RSZ1_RSZ1_RSZ2
Status: Status.UNKNOWN
Output dim: 9, lower bound: -0.0053004, upper bound: 0.0050235
time: 0.93 seconds

## BFS RS instance: RS_RSZ1_RSZ2_RSZ1_RSZ1_RSZ2

### Backsubstitution after applying RS history:
0: -0.0039095, -0.0003029, -0.0039095, -0.0003029, -0.0032491, 0.0031165
1: -0.0044907, 0.0040812, -0.0044907, 0.0040812, -0.0037804, 0.0040721
2: 0.0029569, 0.0099431, 0.0029569, 0.0099431, -0.0037978, 0.0039909
3: -0.0045519, -0.0034923, -0.0045519, -0.0034923, -0.0010595, 0.0010595
4: 0.0017018, 0.0078700, 0.0017018, 0.0078700, -0.0053436, 0.0052414
5: -0.0030341, 0.0031953, -0.0030341, 0.0031953, -0.0046998, 0.0048326
6: -0.0068516, -0.0033211, -0.0068516, -0.0033211, -0.0033394, 0.0033657
7: -0.0021570, 0.0038259, -0.0021570, 0.0038259, -0.0058414, 0.0057602
8: -0.0008533, 0.0000782, -0.0008533, 0.0000782, -0.0008910, 0.0008645
9: 0.9973557, 1.0121032, 0.9973557, 1.0121032, -0.0093525, 0.0089441

### Unstable ReLU Count (Linear/Conv2D Layers)
- layer_idx=0, type=LayerType.Linear, total=256, inp1_unstable=7, inp2_unstable=7, delta_unstable=251
- layer_idx=2, type=LayerType.Linear, total=256, inp1_unstable=10, inp2_unstable=10, delta_unstable=256
- layer_idx=4, type=LayerType.Linear, total=256, inp1_unstable=1, inp2_unstable=1, delta_unstable=256
- layer_idx=6, type=LayerType.Linear, total=256, inp1_unstable=0, inp2_unstable=0, delta_unstable=256
- layer_idx=8, type=LayerType.Linear, total=10, inp1_unstable=4, inp2_unstable=4, delta_unstable=10

Time for backsubstitution: 1.26 seconds

### RS candidates at layer 1
type: RSZ, layer: 1, pos: 148
type: RSZ, layer: 1, pos: 219

Time for candidate selection: 0.14 seconds

### Candidate
type: RSZ, layer: 1, pos: 148

### Relational analysis RSZ of RS_RSZ1_RSZ2_RSZ1_RSZ1_RSZ2_RSZ1

#### Relational analysis RSZ result of RS_RSZ1_RSZ2_RSZ1_RSZ1_RSZ2_RSZ1
Status: Status.UNKNOWN
Output dim: 9, lower bound: -0.0054043, upper bound: 0.0048824
time: 0.87 seconds

### Relational analysis RSZ of RS_RSZ1_RSZ2_RSZ1_RSZ1_RSZ2_RSZ2

#### Relational analysis RSZ result of RS_RSZ1_RSZ2_RSZ1_RSZ1_RSZ2_RSZ2
Status: Status.UNKNOWN
Output dim: 9, lower bound: -0.0054043, upper bound: 0.0048833
time: 0.90 seconds

## BFS RS instance: RS_RSZ1_RSZ2_RSZ1_RSZ2_RSZ1

### Backsubstitution after applying RS history:
0: -0.0039095, -0.0003029, -0.0039095, -0.0003029, -0.0031337, 0.0032442
1: -0.0044907, 0.0040812, -0.0044907, 0.0040812, -0.0040812, 0.0037967
2: 0.0029569, 0.0099431, 0.0029569, 0.0099431, -0.0039975, 0.0038041
3: -0.0045519, -0.0034923, -0.0045519, -0.0034923, -0.0010595, 0.0010595
4: 0.0017018, 0.0078700, 0.0017018, 0.0078700, -0.0052553, 0.0053389
5: -0.0030341, 0.0031953, -0.0030341, 0.0031953, -0.0048473, 0.0047042
6: -0.0068516, -0.0033211, -0.0068516, -0.0033211, -0.0033662, 0.0033415
7: -0.0021570, 0.0038259, -0.0021570, 0.0038259, -0.0057936, 0.0058176
8: -0.0008533, 0.0000782, -0.0008533, 0.0000782, -0.0009062, 0.0008490
9: 0.9973557, 1.0121032, 0.9973557, 1.0121032, -0.0089850, 0.0093549

### Unstable ReLU Count (Linear/Conv2D Layers)
- layer_idx=0, type=LayerType.Linear, total=256, inp1_unstable=7, inp2_unstable=7, delta_unstable=251
- layer_idx=2, type=LayerType.Linear, total=256, inp1_unstable=10, inp2_unstable=10, delta_unstable=256
- layer_idx=4, type=LayerType.Linear, total=256, inp1_unstable=1, inp2_unstable=1, delta_unstable=256
- layer_idx=6, type=LayerType.Linear, total=256, inp1_unstable=0, inp2_unstable=0, delta_unstable=256
- layer_idx=8, type=LayerType.Linear, total=10, inp1_unstable=4, inp2_unstable=4, delta_unstable=10

Time for backsubstitution: 1.29 seconds

### RS candidates at layer 1
type: RSZ, layer: 1, pos: 148
type: RSZ, layer: 1, pos: 219

Time for candidate selection: 0.14 seconds

### Candidate
type: RSZ, layer: 1, pos: 148

### Relational analysis RSZ of RS_RSZ1_RSZ2_RSZ1_RSZ2_RSZ1_RSZ1

#### Relational analysis RSZ result of RS_RSZ1_RSZ2_RSZ1_RSZ2_RSZ1_RSZ1
Status: Status.UNKNOWN
Output dim: 9, lower bound: -0.0048879, upper bound: 0.0054862
time: 0.90 seconds

### Relational analysis RSZ of RS_RSZ1_RSZ2_RSZ1_RSZ2_RSZ1_RSZ2

#### Relational analysis RSZ result of RS_RSZ1_RSZ2_RSZ1_RSZ2_RSZ1_RSZ2
Status: Status.UNKNOWN
Output dim: 9, lower bound: -0.0048879, upper bound: 0.0054862
time: 0.89 seconds

## BFS RS instance: RS_RSZ1_RSZ2_RSZ1_RSZ2_RSZ2

### Backsubstitution after applying RS history:
0: -0.0039095, -0.0003029, -0.0039095, -0.0003029, -0.0031804, 0.0031880
1: -0.0044907, 0.0040812, -0.0044907, 0.0040812, -0.0039501, 0.0039191
2: 0.0029569, 0.0099431, 0.0029569, 0.0099431, -0.0039070, 0.0038904
3: -0.0045519, -0.0034923, -0.0045519, -0.0034923, -0.0010595, 0.0010595
4: 0.0017018, 0.0078700, 0.0017018, 0.0078700, -0.0052913, 0.0052964
5: -0.0030341, 0.0031953, -0.0030341, 0.0031953, -0.0047795, 0.0047563
6: -0.0068516, -0.0033211, -0.0068516, -0.0033211, -0.0033546, 0.0033520
7: -0.0021570, 0.0038259, -0.0021570, 0.0038259, -0.0058177, 0.0057853
8: -0.0008533, 0.0000782, -0.0008533, 0.0000782, -0.0009000, 0.0008554
9: 0.9973557, 1.0121032, 0.9973557, 1.0121032, -0.0091414, 0.0091776

### Unstable ReLU Count (Linear/Conv2D Layers)
- layer_idx=0, type=LayerType.Linear, total=256, inp1_unstable=7, inp2_unstable=7, delta_unstable=251
- layer_idx=2, type=LayerType.Linear, total=256, inp1_unstable=10, inp2_unstable=10, delta_unstable=256
- layer_idx=4, type=LayerType.Linear, total=256, inp1_unstable=1, inp2_unstable=1, delta_unstable=256
- layer_idx=6, type=LayerType.Linear, total=256, inp1_unstable=0, inp2_unstable=0, delta_unstable=256
- layer_idx=8, type=LayerType.Linear, total=10, inp1_unstable=4, inp2_unstable=4, delta_unstable=10

Time for backsubstitution: 1.31 seconds

### RS candidates at layer 1
type: RSZ, layer: 1, pos: 148
type: RSZ, layer: 1, pos: 219

Time for candidate selection: 0.15 seconds

### Candidate
type: RSZ, layer: 1, pos: 148

### Relational analysis RSZ of RS_RSZ1_RSZ2_RSZ1_RSZ2_RSZ2_RSZ1

#### Relational analysis RSZ result of RS_RSZ1_RSZ2_RSZ1_RSZ2_RSZ2_RSZ1
Status: Status.UNKNOWN
Output dim: 9, lower bound: -0.0049834, upper bound: 0.0053090
time: 0.89 seconds

### Relational analysis RSZ of RS_RSZ1_RSZ2_RSZ1_RSZ2_RSZ2_RSZ2

#### Relational analysis RSZ result of RS_RSZ1_RSZ2_RSZ1_RSZ2_RSZ2_RSZ2
Status: Status.UNKNOWN
Output dim: 9, lower bound: -0.0049834, upper bound: 0.0053090
time: 0.88 seconds

## BFS RS instance: RS_RSZ1_RSZ2_RSZ2_RSZ1_RSZ1

### Backsubstitution after applying RS history:
0: -0.0039095, -0.0003029, -0.0039095, -0.0003029, -0.0031881, 0.0031800
1: -0.0044907, 0.0040812, -0.0044907, 0.0040812, -0.0039378, 0.0039390
2: 0.0029569, 0.0099431, 0.0029569, 0.0099431, -0.0039035, 0.0039024
3: -0.0045519, -0.0034923, -0.0045519, -0.0034923, -0.0010595, 0.0010595
4: 0.0017018, 0.0078700, 0.0017018, 0.0078700, -0.0052974, 0.0052898
5: -0.0030341, 0.0031953, -0.0030341, 0.0031953, -0.0047745, 0.0047649
6: -0.0068516, -0.0033211, -0.0068516, -0.0033211, -0.0033532, 0.0033538
7: -0.0021570, 0.0038259, -0.0021570, 0.0038259, -0.0058128, 0.0058006
8: -0.0008533, 0.0000782, -0.0008533, 0.0000782, -0.0008847, 0.0008674
9: 0.9973557, 1.0121032, 0.9973557, 1.0121032, -0.0091690, 0.0091534

### Unstable ReLU Count (Linear/Conv2D Layers)
- layer_idx=0, type=LayerType.Linear, total=256, inp1_unstable=7, inp2_unstable=7, delta_unstable=251
- layer_idx=2, type=LayerType.Linear, total=256, inp1_unstable=10, inp2_unstable=10, delta_unstable=256
- layer_idx=4, type=LayerType.Linear, total=256, inp1_unstable=1, inp2_unstable=1, delta_unstable=256
- layer_idx=6, type=LayerType.Linear, total=256, inp1_unstable=0, inp2_unstable=0, delta_unstable=256
- layer_idx=8, type=LayerType.Linear, total=10, inp1_unstable=4, inp2_unstable=4, delta_unstable=10

Time for backsubstitution: 1.28 seconds

### RS candidates at layer 1
type: RSZ, layer: 1, pos: 148
type: RSZ, layer: 1, pos: 219

Time for candidate selection: 0.14 seconds

### Candidate
type: RSZ, layer: 1, pos: 148

### Relational analysis RSZ of RS_RSZ1_RSZ2_RSZ2_RSZ1_RSZ1_RSZ1

#### Relational analysis RSZ result of RS_RSZ1_RSZ2_RSZ2_RSZ1_RSZ1_RSZ1
Status: Status.UNKNOWN
Output dim: 9, lower bound: -0.0052807, upper bound: 0.0050511
time: 0.84 seconds

### Relational analysis RSZ of RS_RSZ1_RSZ2_RSZ2_RSZ1_RSZ1_RSZ2

#### Relational analysis RSZ result of RS_RSZ1_RSZ2_RSZ2_RSZ1_RSZ1_RSZ2
Status: Status.UNKNOWN
Output dim: 9, lower bound: -0.0052807, upper bound: 0.0050511
time: 0.84 seconds

## BFS RS instance: RS_RSZ1_RSZ2_RSZ2_RSZ1_RSZ2

### Backsubstitution after applying RS history:
0: -0.0039095, -0.0003029, -0.0039095, -0.0003029, -0.0032436, 0.0031334
1: -0.0044907, 0.0040812, -0.0044907, 0.0040812, -0.0038145, 0.0040626
2: 0.0029569, 0.0099431, 0.0029569, 0.0099431, -0.0038169, 0.0039842
3: -0.0045519, -0.0034923, -0.0045519, -0.0034923, -0.0010595, 0.0010595
4: 0.0017018, 0.0078700, 0.0017018, 0.0078700, -0.0053395, 0.0052543
5: -0.0030341, 0.0031953, -0.0030341, 0.0031953, -0.0047159, 0.0048282
6: -0.0068516, -0.0033211, -0.0068516, -0.0033211, -0.0033427, 0.0033648
7: -0.0021570, 0.0038259, -0.0021570, 0.0038259, -0.0058384, 0.0057675
8: -0.0008533, 0.0000782, -0.0008533, 0.0000782, -0.0008803, 0.0008747
9: 0.9973557, 1.0121032, 0.9973557, 1.0121032, -0.0093386, 0.0089965

### Unstable ReLU Count (Linear/Conv2D Layers)
- layer_idx=0, type=LayerType.Linear, total=256, inp1_unstable=7, inp2_unstable=7, delta_unstable=251
- layer_idx=2, type=LayerType.Linear, total=256, inp1_unstable=10, inp2_unstable=10, delta_unstable=256
- layer_idx=4, type=LayerType.Linear, total=256, inp1_unstable=1, inp2_unstable=1, delta_unstable=256
- layer_idx=6, type=LayerType.Linear, total=256, inp1_unstable=0, inp2_unstable=0, delta_unstable=256
- layer_idx=8, type=LayerType.Linear, total=10, inp1_unstable=4, inp2_unstable=4, delta_unstable=10

Time for backsubstitution: 1.28 seconds

### RS candidates at layer 1
type: RSZ, layer: 1, pos: 148
type: RSZ, layer: 1, pos: 219

Time for candidate selection: 0.13 seconds

### Candidate
type: RSZ, layer: 1, pos: 148

### Relational analysis RSZ of RS_RSZ1_RSZ2_RSZ2_RSZ1_RSZ2_RSZ1

#### Relational analysis RSZ result of RS_RSZ1_RSZ2_RSZ2_RSZ1_RSZ2_RSZ1
Status: Status.UNKNOWN
Output dim: 9, lower bound: -0.0054016, upper bound: 0.0049486
time: 0.88 seconds

### Relational analysis RSZ of RS_RSZ1_RSZ2_RSZ2_RSZ1_RSZ2_RSZ2

#### Relational analysis RSZ result of RS_RSZ1_RSZ2_RSZ2_RSZ1_RSZ2_RSZ2
Status: Status.UNKNOWN
Output dim: 9, lower bound: -0.0054016, upper bound: 0.0049486
time: 0.90 seconds

## BFS RS instance: RS_RSZ1_RSZ2_RSZ2_RSZ2_RSZ1

### Backsubstitution after applying RS history:
0: -0.0039095, -0.0003029, -0.0039095, -0.0003029, -0.0031165, 0.0032499
1: -0.0044907, 0.0040812, -0.0044907, 0.0040812, -0.0040961, 0.0037627
2: 0.0029569, 0.0099431, 0.0029569, 0.0099431, -0.0040069, 0.0037855
3: -0.0045519, -0.0034923, -0.0045519, -0.0034923, -0.0010595, 0.0010595
4: 0.0017018, 0.0078700, 0.0017018, 0.0078700, -0.0052425, 0.0053431
5: -0.0030341, 0.0031953, -0.0030341, 0.0031953, -0.0048514, 0.0046874
6: -0.0068516, -0.0033211, -0.0068516, -0.0033211, -0.0033675, 0.0033383
7: -0.0021570, 0.0038259, -0.0021570, 0.0038259, -0.0057873, 0.0058214
8: -0.0008533, 0.0000782, -0.0008533, 0.0000782, -0.0008942, 0.0008568
9: 0.9973557, 1.0121032, 0.9973557, 1.0121032, -0.0089336, 0.0093735

### Unstable ReLU Count (Linear/Conv2D Layers)
- layer_idx=0, type=LayerType.Linear, total=256, inp1_unstable=7, inp2_unstable=7, delta_unstable=251
- layer_idx=2, type=LayerType.Linear, total=256, inp1_unstable=10, inp2_unstable=10, delta_unstable=256
- layer_idx=4, type=LayerType.Linear, total=256, inp1_unstable=1, inp2_unstable=1, delta_unstable=256
- layer_idx=6, type=LayerType.Linear, total=256, inp1_unstable=0, inp2_unstable=0, delta_unstable=256
- layer_idx=8, type=LayerType.Linear, total=10, inp1_unstable=4, inp2_unstable=4, delta_unstable=10

Time for backsubstitution: 1.28 seconds

### RS candidates at layer 1
type: RSZ, layer: 1, pos: 148
type: RSZ, layer: 1, pos: 219

Time for candidate selection: 0.14 seconds

### Candidate
type: RSZ, layer: 1, pos: 148

### Relational analysis RSZ of RS_RSZ1_RSZ2_RSZ2_RSZ2_RSZ1_RSZ1

#### Relational analysis RSZ result of RS_RSZ1_RSZ2_RSZ2_RSZ2_RSZ1_RSZ1
Status: Status.UNKNOWN
Output dim: 9, lower bound: -0.0048224, upper bound: 0.0054973
time: 0.91 seconds

### Relational analysis RSZ of RS_RSZ1_RSZ2_RSZ2_RSZ2_RSZ1_RSZ2

#### Relational analysis RSZ result of RS_RSZ1_RSZ2_RSZ2_RSZ2_RSZ1_RSZ2
Status: Status.UNKNOWN
Output dim: 9, lower bound: -0.0048224, upper bound: 0.0054973
time: 0.89 seconds

## BFS RS instance: RS_RSZ1_RSZ2_RSZ2_RSZ2_RSZ2

### Backsubstitution after applying RS history:
0: -0.0039095, -0.0003029, -0.0039095, -0.0003029, -0.0031741, 0.0032011
1: -0.0044907, 0.0040812, -0.0044907, 0.0040812, -0.0039748, 0.0039053
2: 0.0029569, 0.0099431, 0.0029569, 0.0099431, -0.0039214, 0.0038822
3: -0.0045519, -0.0034923, -0.0045519, -0.0034923, -0.0010595, 0.0010595
4: 0.0017018, 0.0078700, 0.0017018, 0.0078700, -0.0052864, 0.0053061
5: -0.0030341, 0.0031953, -0.0030341, 0.0031953, -0.0047929, 0.0047488
6: -0.0068516, -0.0033211, -0.0068516, -0.0033211, -0.0033569, 0.0033504
7: -0.0021570, 0.0038259, -0.0021570, 0.0038259, -0.0058137, 0.0057908
8: -0.0008533, 0.0000782, -0.0008533, 0.0000782, -0.0008878, 0.0008640
9: 0.9973557, 1.0121032, 0.9973557, 1.0121032, -0.0091177, 0.0092142

### Unstable ReLU Count (Linear/Conv2D Layers)
- layer_idx=0, type=LayerType.Linear, total=256, inp1_unstable=7, inp2_unstable=7, delta_unstable=251
- layer_idx=2, type=LayerType.Linear, total=256, inp1_unstable=10, inp2_unstable=10, delta_unstable=256
- layer_idx=4, type=LayerType.Linear, total=256, inp1_unstable=1, inp2_unstable=1, delta_unstable=256
- layer_idx=6, type=LayerType.Linear, total=256, inp1_unstable=0, inp2_unstable=0, delta_unstable=256
- layer_idx=8, type=LayerType.Linear, total=10, inp1_unstable=4, inp2_unstable=4, delta_unstable=10

Time for backsubstitution: 1.22 seconds

### RS candidates at layer 1
type: RSZ, layer: 1, pos: 148
type: RSZ, layer: 1, pos: 219

Time for candidate selection: 0.12 seconds

### Candidate
type: RSZ, layer: 1, pos: 148

### Relational analysis RSZ of RS_RSZ1_RSZ2_RSZ2_RSZ2_RSZ2_RSZ1

#### Relational analysis RSZ result of RS_RSZ1_RSZ2_RSZ2_RSZ2_RSZ2_RSZ1
Status: Status.UNKNOWN
Output dim: 9, lower bound: -0.0049340, upper bound: 0.0053480
time: 0.88 seconds

### Relational analysis RSZ of RS_RSZ1_RSZ2_RSZ2_RSZ2_RSZ2_RSZ2

#### Relational analysis RSZ result of RS_RSZ1_RSZ2_RSZ2_RSZ2_RSZ2_RSZ2
Status: Status.UNKNOWN
Output dim: 9, lower bound: -0.0049340, upper bound: 0.0053480
time: 0.92 seconds

## BFS RS instance: RS_RSZ2_RSZ1_RSZ1_RSZ1_RSZ1

### Backsubstitution after applying RS history:
0: -0.0039095, -0.0003029, -0.0039095, -0.0003029, -0.0032011, 0.0031741
1: -0.0044907, 0.0040812, -0.0044907, 0.0040812, -0.0039053, 0.0039748
2: 0.0029569, 0.0099431, 0.0029569, 0.0099431, -0.0038822, 0.0039214
3: -0.0045519, -0.0034923, -0.0045519, -0.0034923, -0.0010595, 0.0010595
4: 0.0017018, 0.0078700, 0.0017018, 0.0078700, -0.0053061, 0.0052864
5: -0.0030341, 0.0031953, -0.0030341, 0.0031953, -0.0047488, 0.0047929
6: -0.0068516, -0.0033211, -0.0068516, -0.0033211, -0.0033504, 0.0033569
7: -0.0021570, 0.0038259, -0.0021570, 0.0038259, -0.0057908, 0.0058137
8: -0.0008533, 0.0000782, -0.0008533, 0.0000782, -0.0008640, 0.0008878
9: 0.9973557, 1.0121032, 0.9973557, 1.0121032, -0.0092142, 0.0091177

### Unstable ReLU Count (Linear/Conv2D Layers)
- layer_idx=0, type=LayerType.Linear, total=256, inp1_unstable=7, inp2_unstable=7, delta_unstable=251
- layer_idx=2, type=LayerType.Linear, total=256, inp1_unstable=10, inp2_unstable=10, delta_unstable=256
- layer_idx=4, type=LayerType.Linear, total=256, inp1_unstable=1, inp2_unstable=1, delta_unstable=256
- layer_idx=6, type=LayerType.Linear, total=256, inp1_unstable=0, inp2_unstable=0, delta_unstable=256
- layer_idx=8, type=LayerType.Linear, total=10, inp1_unstable=4, inp2_unstable=4, delta_unstable=10

Time for backsubstitution: 1.23 seconds

### RS candidates at layer 1
type: RSZ, layer: 1, pos: 148
type: RSZ, layer: 1, pos: 219

Time for candidate selection: 0.12 seconds

### Candidate
type: RSZ, layer: 1, pos: 148

### Relational analysis RSZ of RS_RSZ2_RSZ1_RSZ1_RSZ1_RSZ1_RSZ1

#### Relational analysis RSZ result of RS_RSZ2_RSZ1_RSZ1_RSZ1_RSZ1_RSZ1
Status: Status.UNKNOWN
Output dim: 9, lower bound: -0.0053480, upper bound: 0.0049340
time: 0.83 seconds

### Relational analysis RSZ of RS_RSZ2_RSZ1_RSZ1_RSZ1_RSZ1_RSZ2

#### Relational analysis RSZ result of RS_RSZ2_RSZ1_RSZ1_RSZ1_RSZ1_RSZ2
Status: Status.UNKNOWN
Output dim: 9, lower bound: -0.0053480, upper bound: 0.0049340
time: 0.87 seconds

## BFS RS instance: RS_RSZ2_RSZ1_RSZ1_RSZ1_RSZ2

### Backsubstitution after applying RS history:
0: -0.0039095, -0.0003029, -0.0039095, -0.0003029, -0.0032499, 0.0031165
1: -0.0044907, 0.0040812, -0.0044907, 0.0040812, -0.0037627, 0.0040961
2: 0.0029569, 0.0099431, 0.0029569, 0.0099431, -0.0037855, 0.0040069
3: -0.0045519, -0.0034923, -0.0045519, -0.0034923, -0.0010595, 0.0010595
4: 0.0017018, 0.0078700, 0.0017018, 0.0078700, -0.0053431, 0.0052425
5: -0.0030341, 0.0031953, -0.0030341, 0.0031953, -0.0046874, 0.0048514
6: -0.0068516, -0.0033211, -0.0068516, -0.0033211, -0.0033383, 0.0033675
7: -0.0021570, 0.0038259, -0.0021570, 0.0038259, -0.0058214, 0.0057873
8: -0.0008533, 0.0000782, -0.0008533, 0.0000782, -0.0008568, 0.0008942
9: 0.9973557, 1.0121032, 0.9973557, 1.0121032, -0.0093735, 0.0089336

### Unstable ReLU Count (Linear/Conv2D Layers)
- layer_idx=0, type=LayerType.Linear, total=256, inp1_unstable=7, inp2_unstable=7, delta_unstable=251
- layer_idx=2, type=LayerType.Linear, total=256, inp1_unstable=10, inp2_unstable=10, delta_unstable=256
- layer_idx=4, type=LayerType.Linear, total=256, inp1_unstable=1, inp2_unstable=1, delta_unstable=256
- layer_idx=6, type=LayerType.Linear, total=256, inp1_unstable=0, inp2_unstable=0, delta_unstable=256
- layer_idx=8, type=LayerType.Linear, total=10, inp1_unstable=4, inp2_unstable=4, delta_unstable=10

Time for backsubstitution: 1.42 seconds

### RS candidates at layer 1
type: RSZ, layer: 1, pos: 148
type: RSZ, layer: 1, pos: 219

Time for candidate selection: 0.16 seconds

### Candidate
type: RSZ, layer: 1, pos: 148

### Relational analysis RSZ of RS_RSZ2_RSZ1_RSZ1_RSZ1_RSZ2_RSZ1

#### Relational analysis RSZ result of RS_RSZ2_RSZ1_RSZ1_RSZ1_RSZ2_RSZ1
Status: Status.UNKNOWN
Output dim: 9, lower bound: -0.0054973, upper bound: 0.0048224
time: 0.80 seconds

### Relational analysis RSZ of RS_RSZ2_RSZ1_RSZ1_RSZ1_RSZ2_RSZ2

#### Relational analysis RSZ result of RS_RSZ2_RSZ1_RSZ1_RSZ1_RSZ2_RSZ2
Status: Status.UNKNOWN
Output dim: 9, lower bound: -0.0054973, upper bound: 0.0048224
time: 0.80 seconds

## BFS RS instance: RS_RSZ2_RSZ1_RSZ1_RSZ2_RSZ1

### Backsubstitution after applying RS history:
0: -0.0039095, -0.0003029, -0.0039095, -0.0003029, -0.0031334, 0.0032436
1: -0.0044907, 0.0040812, -0.0044907, 0.0040812, -0.0040626, 0.0038145
2: 0.0029569, 0.0099431, 0.0029569, 0.0099431, -0.0039842, 0.0038169
3: -0.0045519, -0.0034923, -0.0045519, -0.0034923, -0.0010595, 0.0010595
4: 0.0017018, 0.0078700, 0.0017018, 0.0078700, -0.0052543, 0.0053395
5: -0.0030341, 0.0031953, -0.0030341, 0.0031953, -0.0048282, 0.0047159
6: -0.0068516, -0.0033211, -0.0068516, -0.0033211, -0.0033648, 0.0033427
7: -0.0021570, 0.0038259, -0.0021570, 0.0038259, -0.0057675, 0.0058384
8: -0.0008533, 0.0000782, -0.0008533, 0.0000782, -0.0008747, 0.0008803
9: 0.9973557, 1.0121032, 0.9973557, 1.0121032, -0.0089965, 0.0093386

### Unstable ReLU Count (Linear/Conv2D Layers)
- layer_idx=0, type=LayerType.Linear, total=256, inp1_unstable=7, inp2_unstable=7, delta_unstable=251
- layer_idx=2, type=LayerType.Linear, total=256, inp1_unstable=10, inp2_unstable=10, delta_unstable=256
- layer_idx=4, type=LayerType.Linear, total=256, inp1_unstable=1, inp2_unstable=1, delta_unstable=256
- layer_idx=6, type=LayerType.Linear, total=256, inp1_unstable=0, inp2_unstable=0, delta_unstable=256
- layer_idx=8, type=LayerType.Linear, total=10, inp1_unstable=4, inp2_unstable=4, delta_unstable=10

Time for backsubstitution: 1.31 seconds

### RS candidates at layer 1
type: RSZ, layer: 1, pos: 148
type: RSZ, layer: 1, pos: 219

Time for candidate selection: 0.14 seconds

### Candidate
type: RSZ, layer: 1, pos: 148

### Relational analysis RSZ of RS_RSZ2_RSZ1_RSZ1_RSZ2_RSZ1_RSZ1

#### Relational analysis RSZ result of RS_RSZ2_RSZ1_RSZ1_RSZ2_RSZ1_RSZ1
Status: Status.UNKNOWN
Output dim: 9, lower bound: -0.0049486, upper bound: 0.0054016
time: 0.85 seconds

### Relational analysis RSZ of RS_RSZ2_RSZ1_RSZ1_RSZ2_RSZ1_RSZ2

#### Relational analysis RSZ result of RS_RSZ2_RSZ1_RSZ1_RSZ2_RSZ1_RSZ2
Status: Status.UNKNOWN
Output dim: 9, lower bound: -0.0049486, upper bound: 0.0054016
time: 0.84 seconds

## BFS RS instance: RS_RSZ2_RSZ1_RSZ1_RSZ2_RSZ2

### Backsubstitution after applying RS history:
0: -0.0039095, -0.0003029, -0.0039095, -0.0003029, -0.0031800, 0.0031881
1: -0.0044907, 0.0040812, -0.0044907, 0.0040812, -0.0039390, 0.0039378
2: 0.0029569, 0.0099431, 0.0029569, 0.0099431, -0.0039024, 0.0039035
3: -0.0045519, -0.0034923, -0.0045519, -0.0034923, -0.0010595, 0.0010595
4: 0.0017018, 0.0078700, 0.0017018, 0.0078700, -0.0052898, 0.0052974
5: -0.0030341, 0.0031953, -0.0030341, 0.0031953, -0.0047649, 0.0047745
6: -0.0068516, -0.0033211, -0.0068516, -0.0033211, -0.0033538, 0.0033532
7: -0.0021570, 0.0038259, -0.0021570, 0.0038259, -0.0058006, 0.0058128
8: -0.0008533, 0.0000782, -0.0008533, 0.0000782, -0.0008674, 0.0008847
9: 0.9973557, 1.0121032, 0.9973557, 1.0121032, -0.0091534, 0.0091690

### Unstable ReLU Count (Linear/Conv2D Layers)
- layer_idx=0, type=LayerType.Linear, total=256, inp1_unstable=7, inp2_unstable=7, delta_unstable=251
- layer_idx=2, type=LayerType.Linear, total=256, inp1_unstable=10, inp2_unstable=10, delta_unstable=256
- layer_idx=4, type=LayerType.Linear, total=256, inp1_unstable=1, inp2_unstable=1, delta_unstable=256
- layer_idx=6, type=LayerType.Linear, total=256, inp1_unstable=0, inp2_unstable=0, delta_unstable=256
- layer_idx=8, type=LayerType.Linear, total=10, inp1_unstable=4, inp2_unstable=4, delta_unstable=10

Time for backsubstitution: 1.33 seconds

### RS candidates at layer 1
type: RSZ, layer: 1, pos: 148
type: RSZ, layer: 1, pos: 219

Time for candidate selection: 0.14 seconds

### Candidate
type: RSZ, layer: 1, pos: 148

### Relational analysis RSZ of RS_RSZ2_RSZ1_RSZ1_RSZ2_RSZ2_RSZ1

#### Relational analysis RSZ result of RS_RSZ2_RSZ1_RSZ1_RSZ2_RSZ2_RSZ1
Status: Status.UNKNOWN
Output dim: 9, lower bound: -0.0050511, upper bound: 0.0052807
time: 0.82 seconds

### Relational analysis RSZ of RS_RSZ2_RSZ1_RSZ1_RSZ2_RSZ2_RSZ2

#### Relational analysis RSZ result of RS_RSZ2_RSZ1_RSZ1_RSZ2_RSZ2_RSZ2
Status: Status.UNKNOWN
Output dim: 9, lower bound: -0.0050511, upper bound: 0.0052807
time: 0.84 seconds

## BFS RS instance: RS_RSZ2_RSZ1_RSZ2_RSZ1_RSZ1

### Backsubstitution after applying RS history:
0: -0.0039095, -0.0003029, -0.0039095, -0.0003029, -0.0031880, 0.0031804
1: -0.0044907, 0.0040812, -0.0044907, 0.0040812, -0.0039191, 0.0039501
2: 0.0029569, 0.0099431, 0.0029569, 0.0099431, -0.0038904, 0.0039070
3: -0.0045519, -0.0034923, -0.0045519, -0.0034923, -0.0010595, 0.0010595
4: 0.0017018, 0.0078700, 0.0017018, 0.0078700, -0.0052964, 0.0052913
5: -0.0030341, 0.0031953, -0.0030341, 0.0031953, -0.0047563, 0.0047795
6: -0.0068516, -0.0033211, -0.0068516, -0.0033211, -0.0033520, 0.0033546
7: -0.0021570, 0.0038259, -0.0021570, 0.0038259, -0.0057853, 0.0058177
8: -0.0008533, 0.0000782, -0.0008533, 0.0000782, -0.0008554, 0.0009000
9: 0.9973557, 1.0121032, 0.9973557, 1.0121032, -0.0091776, 0.0091414

### Unstable ReLU Count (Linear/Conv2D Layers)
- layer_idx=0, type=LayerType.Linear, total=256, inp1_unstable=7, inp2_unstable=7, delta_unstable=251
- layer_idx=2, type=LayerType.Linear, total=256, inp1_unstable=10, inp2_unstable=10, delta_unstable=256
- layer_idx=4, type=LayerType.Linear, total=256, inp1_unstable=1, inp2_unstable=1, delta_unstable=256
- layer_idx=6, type=LayerType.Linear, total=256, inp1_unstable=0, inp2_unstable=0, delta_unstable=256
- layer_idx=8, type=LayerType.Linear, total=10, inp1_unstable=4, inp2_unstable=4, delta_unstable=10

Time for backsubstitution: 1.23 seconds

### RS candidates at layer 1
type: RSZ, layer: 1, pos: 148
type: RSZ, layer: 1, pos: 219

Time for candidate selection: 0.12 seconds

### Candidate
type: RSZ, layer: 1, pos: 148

### Relational analysis RSZ of RS_RSZ2_RSZ1_RSZ2_RSZ1_RSZ1_RSZ1

#### Relational analysis RSZ result of RS_RSZ2_RSZ1_RSZ2_RSZ1_RSZ1_RSZ1
Status: Status.UNKNOWN
Output dim: 9, lower bound: -0.0053090, upper bound: 0.0049834
time: 0.80 seconds

### Relational analysis RSZ of RS_RSZ2_RSZ1_RSZ2_RSZ1_RSZ1_RSZ2

#### Relational analysis RSZ result of RS_RSZ2_RSZ1_RSZ2_RSZ1_RSZ1_RSZ2
Status: Status.UNKNOWN
Output dim: 9, lower bound: -0.0053090, upper bound: 0.0049834
time: 0.80 seconds

## BFS RS instance: RS_RSZ2_RSZ1_RSZ2_RSZ1_RSZ2

### Backsubstitution after applying RS history:
0: -0.0039095, -0.0003029, -0.0039095, -0.0003029, -0.0032442, 0.0031337
1: -0.0044907, 0.0040812, -0.0044907, 0.0040812, -0.0037967, 0.0040812
2: 0.0029569, 0.0099431, 0.0029569, 0.0099431, -0.0038041, 0.0039975
3: -0.0045519, -0.0034923, -0.0045519, -0.0034923, -0.0010595, 0.0010595
4: 0.0017018, 0.0078700, 0.0017018, 0.0078700, -0.0053389, 0.0052553
5: -0.0030341, 0.0031953, -0.0030341, 0.0031953, -0.0047042, 0.0048473
6: -0.0068516, -0.0033211, -0.0068516, -0.0033211, -0.0033415, 0.0033662
7: -0.0021570, 0.0038259, -0.0021570, 0.0038259, -0.0058176, 0.0057936
8: -0.0008533, 0.0000782, -0.0008533, 0.0000782, -0.0008490, 0.0009062
9: 0.9973557, 1.0121032, 0.9973557, 1.0121032, -0.0093549, 0.0089850

### Unstable ReLU Count (Linear/Conv2D Layers)
- layer_idx=0, type=LayerType.Linear, total=256, inp1_unstable=7, inp2_unstable=7, delta_unstable=251
- layer_idx=2, type=LayerType.Linear, total=256, inp1_unstable=10, inp2_unstable=10, delta_unstable=256
- layer_idx=4, type=LayerType.Linear, total=256, inp1_unstable=1, inp2_unstable=1, delta_unstable=256
- layer_idx=6, type=LayerType.Linear, total=256, inp1_unstable=0, inp2_unstable=0, delta_unstable=256
- layer_idx=8, type=LayerType.Linear, total=10, inp1_unstable=4, inp2_unstable=4, delta_unstable=10

Time for backsubstitution: 1.25 seconds

### RS candidates at layer 1
type: RSZ, layer: 1, pos: 148
type: RSZ, layer: 1, pos: 219

Time for candidate selection: 0.13 seconds

### Candidate
type: RSZ, layer: 1, pos: 148

### Relational analysis RSZ of RS_RSZ2_RSZ1_RSZ2_RSZ1_RSZ2_RSZ1

#### Relational analysis RSZ result of RS_RSZ2_RSZ1_RSZ2_RSZ1_RSZ2_RSZ1
Status: Status.UNKNOWN
Output dim: 9, lower bound: -0.0054862, upper bound: 0.0048879
time: 0.81 seconds

### Relational analysis RSZ of RS_RSZ2_RSZ1_RSZ2_RSZ1_RSZ2_RSZ2

#### Relational analysis RSZ result of RS_RSZ2_RSZ1_RSZ2_RSZ1_RSZ2_RSZ2
Status: Status.UNKNOWN
Output dim: 9, lower bound: -0.0054862, upper bound: 0.0048879
time: 0.82 seconds

## BFS RS instance: RS_RSZ2_RSZ1_RSZ2_RSZ2_RSZ1

### Backsubstitution after applying RS history:
0: -0.0039095, -0.0003029, -0.0039095, -0.0003029, -0.0031165, 0.0032491
1: -0.0044907, 0.0040812, -0.0044907, 0.0040812, -0.0040721, 0.0037804
2: 0.0029569, 0.0099431, 0.0029569, 0.0099431, -0.0039909, 0.0037978
3: -0.0045519, -0.0034923, -0.0045519, -0.0034923, -0.0010595, 0.0010595
4: 0.0017018, 0.0078700, 0.0017018, 0.0078700, -0.0052414, 0.0053436
5: -0.0030341, 0.0031953, -0.0030341, 0.0031953, -0.0048326, 0.0046998
6: -0.0068516, -0.0033211, -0.0068516, -0.0033211, -0.0033657, 0.0033394
7: -0.0021570, 0.0038259, -0.0021570, 0.0038259, -0.0057602, 0.0058414
8: -0.0008533, 0.0000782, -0.0008533, 0.0000782, -0.0008645, 0.0008910
9: 0.9973557, 1.0121032, 0.9973557, 1.0121032, -0.0089441, 0.0093525

### Unstable ReLU Count (Linear/Conv2D Layers)
- layer_idx=0, type=LayerType.Linear, total=256, inp1_unstable=7, inp2_unstable=7, delta_unstable=251
- layer_idx=2, type=LayerType.Linear, total=256, inp1_unstable=10, inp2_unstable=10, delta_unstable=256
- layer_idx=4, type=LayerType.Linear, total=256, inp1_unstable=1, inp2_unstable=1, delta_unstable=256
- layer_idx=6, type=LayerType.Linear, total=256, inp1_unstable=0, inp2_unstable=0, delta_unstable=256
- layer_idx=8, type=LayerType.Linear, total=10, inp1_unstable=4, inp2_unstable=4, delta_unstable=10

Time for backsubstitution: 1.29 seconds

### RS candidates at layer 1
type: RSZ, layer: 1, pos: 148
type: RSZ, layer: 1, pos: 219

Time for candidate selection: 0.13 seconds

### Candidate
type: RSZ, layer: 1, pos: 148

### Relational analysis RSZ of RS_RSZ2_RSZ1_RSZ2_RSZ2_RSZ1_RSZ1

#### Relational analysis RSZ result of RS_RSZ2_RSZ1_RSZ2_RSZ2_RSZ1_RSZ1
Status: Status.UNKNOWN
Output dim: 9, lower bound: -0.0048833, upper bound: 0.0054043
time: 0.83 seconds

### Relational analysis RSZ of RS_RSZ2_RSZ1_RSZ2_RSZ2_RSZ1_RSZ2

#### Relational analysis RSZ result of RS_RSZ2_RSZ1_RSZ2_RSZ2_RSZ1_RSZ2
Status: Status.UNKNOWN
Output dim: 9, lower bound: -0.0048824, upper bound: 0.0054043
time: 0.82 seconds

## BFS RS instance: RS_RSZ2_RSZ1_RSZ2_RSZ2_RSZ2

### Backsubstitution after applying RS history:
0: -0.0039095, -0.0003029, -0.0039095, -0.0003029, -0.0031739, 0.0032013
1: -0.0044907, 0.0040812, -0.0044907, 0.0040812, -0.0039610, 0.0039250
2: 0.0029569, 0.0099431, 0.0029569, 0.0099431, -0.0039163, 0.0038950
3: -0.0045519, -0.0034923, -0.0045519, -0.0034923, -0.0010595, 0.0010595
4: 0.0017018, 0.0078700, 0.0017018, 0.0078700, -0.0052851, 0.0053071
5: -0.0030341, 0.0031953, -0.0030341, 0.0031953, -0.0047795, 0.0047685
6: -0.0068516, -0.0033211, -0.0068516, -0.0033211, -0.0033560, 0.0033519
7: -0.0021570, 0.0038259, -0.0021570, 0.0038259, -0.0057957, 0.0058191
8: -0.0008533, 0.0000782, -0.0008533, 0.0000782, -0.0008574, 0.0008957
9: 0.9973557, 1.0121032, 0.9973557, 1.0121032, -0.0091334, 0.0092050

### Unstable ReLU Count (Linear/Conv2D Layers)
- layer_idx=0, type=LayerType.Linear, total=256, inp1_unstable=7, inp2_unstable=7, delta_unstable=251
- layer_idx=2, type=LayerType.Linear, total=256, inp1_unstable=10, inp2_unstable=10, delta_unstable=256
- layer_idx=4, type=LayerType.Linear, total=256, inp1_unstable=1, inp2_unstable=1, delta_unstable=256
- layer_idx=6, type=LayerType.Linear, total=256, inp1_unstable=0, inp2_unstable=0, delta_unstable=256
- layer_idx=8, type=LayerType.Linear, total=10, inp1_unstable=4, inp2_unstable=4, delta_unstable=10

Time for backsubstitution: 1.28 seconds

### RS candidates at layer 1
type: RSZ, layer: 1, pos: 148
type: RSZ, layer: 1, pos: 219

Time for candidate selection: 0.13 seconds

### Candidate
type: RSZ, layer: 1, pos: 148

### Relational analysis RSZ of RS_RSZ2_RSZ1_RSZ2_RSZ2_RSZ2_RSZ1

#### Relational analysis RSZ result of RS_RSZ2_RSZ1_RSZ2_RSZ2_RSZ2_RSZ1
Status: Status.UNKNOWN
Output dim: 9, lower bound: -0.0050235, upper bound: 0.0053004
time: 0.81 seconds

### Relational analysis RSZ of RS_RSZ2_RSZ1_RSZ2_RSZ2_RSZ2_RSZ2

#### Relational analysis RSZ result of RS_RSZ2_RSZ1_RSZ2_RSZ2_RSZ2_RSZ2
Status: Status.UNKNOWN
Output dim: 9, lower bound: -0.0050235, upper bound: 0.0053004
time: 0.81 seconds

## BFS RS instance: RS_RSZ2_RSZ2_RSZ1_RSZ1_RSZ1

### Backsubstitution after applying RS history:
0: -0.0039095, -0.0003029, -0.0039095, -0.0003029, -0.0031999, 0.0031758
1: -0.0044907, 0.0040812, -0.0044907, 0.0040812, -0.0039353, 0.0039525
2: 0.0029569, 0.0099431, 0.0029569, 0.0099431, -0.0039034, 0.0039073
3: -0.0045519, -0.0034923, -0.0045519, -0.0034923, -0.0010595, 0.0010595
4: 0.0017018, 0.0078700, 0.0017018, 0.0078700, -0.0053058, 0.0052868
5: -0.0030341, 0.0031953, -0.0030341, 0.0031953, -0.0047691, 0.0047789
6: -0.0068516, -0.0033211, -0.0068516, -0.0033211, -0.0033527, 0.0033554
7: -0.0021570, 0.0038259, -0.0021570, 0.0038259, -0.0058123, 0.0057992
8: -0.0008533, 0.0000782, -0.0008533, 0.0000782, -0.0008954, 0.0008570
9: 0.9973557, 1.0121032, 0.9973557, 1.0121032, -0.0091965, 0.0091432

### Unstable ReLU Count (Linear/Conv2D Layers)
- layer_idx=0, type=LayerType.Linear, total=256, inp1_unstable=7, inp2_unstable=7, delta_unstable=251
- layer_idx=2, type=LayerType.Linear, total=256, inp1_unstable=10, inp2_unstable=10, delta_unstable=256
- layer_idx=4, type=LayerType.Linear, total=256, inp1_unstable=1, inp2_unstable=1, delta_unstable=256
- layer_idx=6, type=LayerType.Linear, total=256, inp1_unstable=0, inp2_unstable=0, delta_unstable=256
- layer_idx=8, type=LayerType.Linear, total=10, inp1_unstable=4, inp2_unstable=4, delta_unstable=10

Time for backsubstitution: 1.26 seconds

### RS candidates at layer 1
type: RSZ, layer: 1, pos: 148
type: RSZ, layer: 1, pos: 219

Time for candidate selection: 0.16 seconds

### Candidate
type: RSZ, layer: 1, pos: 148

### Relational analysis RSZ of RS_RSZ2_RSZ2_RSZ1_RSZ1_RSZ1_RSZ1

#### Relational analysis RSZ result of RS_RSZ2_RSZ2_RSZ1_RSZ1_RSZ1_RSZ1
Status: Status.UNKNOWN
Output dim: 9, lower bound: -0.0052487, upper bound: 0.0050524
time: 0.89 seconds

### Relational analysis RSZ of RS_RSZ2_RSZ2_RSZ1_RSZ1_RSZ1_RSZ2

#### Relational analysis RSZ result of RS_RSZ2_RSZ2_RSZ1_RSZ1_RSZ1_RSZ2
Status: Status.UNKNOWN
Output dim: 9, lower bound: -0.0052487, upper bound: 0.0050524
time: 0.88 seconds

## BFS RS instance: RS_RSZ2_RSZ2_RSZ1_RSZ1_RSZ2

### Backsubstitution after applying RS history:
0: -0.0039095, -0.0003029, -0.0039095, -0.0003029, -0.0032479, 0.0031179
1: -0.0044907, 0.0040812, -0.0044907, 0.0040812, -0.0037883, 0.0040665
2: 0.0029569, 0.0099431, 0.0029569, 0.0099431, -0.0038048, 0.0039850
3: -0.0045519, -0.0034923, -0.0045519, -0.0034923, -0.0010595, 0.0010595
4: 0.0017018, 0.0078700, 0.0017018, 0.0078700, -0.0053427, 0.0052427
5: -0.0030341, 0.0031953, -0.0030341, 0.0031953, -0.0047007, 0.0048320
6: -0.0068516, -0.0033211, -0.0068516, -0.0033211, -0.0033400, 0.0033654
7: -0.0021570, 0.0038259, -0.0021570, 0.0038259, -0.0058361, 0.0057668
8: -0.0008533, 0.0000782, -0.0008533, 0.0000782, -0.0008904, 0.0008640
9: 0.9973557, 1.0121032, 0.9973557, 1.0121032, -0.0093472, 0.0089531

### Unstable ReLU Count (Linear/Conv2D Layers)
- layer_idx=0, type=LayerType.Linear, total=256, inp1_unstable=7, inp2_unstable=7, delta_unstable=251
- layer_idx=2, type=LayerType.Linear, total=256, inp1_unstable=10, inp2_unstable=10, delta_unstable=256
- layer_idx=4, type=LayerType.Linear, total=256, inp1_unstable=1, inp2_unstable=1, delta_unstable=256
- layer_idx=6, type=LayerType.Linear, total=256, inp1_unstable=0, inp2_unstable=0, delta_unstable=256
- layer_idx=8, type=LayerType.Linear, total=10, inp1_unstable=4, inp2_unstable=4, delta_unstable=10

Time for backsubstitution: 1.27 seconds

### RS candidates at layer 1
type: RSZ, layer: 1, pos: 148
type: RSZ, layer: 1, pos: 219

Time for candidate selection: 0.14 seconds

### Candidate
type: RSZ, layer: 1, pos: 148

### Relational analysis RSZ of RS_RSZ2_RSZ2_RSZ1_RSZ1_RSZ2_RSZ1

#### Relational analysis RSZ result of RS_RSZ2_RSZ2_RSZ1_RSZ1_RSZ2_RSZ1
Status: Status.UNKNOWN
Output dim: 9, lower bound: -0.0053725, upper bound: 0.0049285
time: 0.88 seconds

### Relational analysis RSZ of RS_RSZ2_RSZ2_RSZ1_RSZ1_RSZ2_RSZ2

#### Relational analysis RSZ result of RS_RSZ2_RSZ2_RSZ1_RSZ1_RSZ2_RSZ2
Status: Status.UNKNOWN
Output dim: 9, lower bound: -0.0053725, upper bound: 0.0049285
time: 0.88 seconds

## BFS RS instance: RS_RSZ2_RSZ2_RSZ1_RSZ2_RSZ1

### Backsubstitution after applying RS history:
0: -0.0039095, -0.0003029, -0.0039095, -0.0003029, -0.0031320, 0.0032454
1: -0.0044907, 0.0040812, -0.0044907, 0.0040812, -0.0040875, 0.0037884
2: 0.0029569, 0.0099431, 0.0029569, 0.0099431, -0.0040022, 0.0037958
3: -0.0045519, -0.0034923, -0.0045519, -0.0034923, -0.0010595, 0.0010595
4: 0.0017018, 0.0078700, 0.0017018, 0.0078700, -0.0052540, 0.0053399
5: -0.0030341, 0.0031953, -0.0030341, 0.0031953, -0.0048478, 0.0047035
6: -0.0068516, -0.0033211, -0.0068516, -0.0033211, -0.0033666, 0.0033409
7: -0.0021570, 0.0038259, -0.0021570, 0.0038259, -0.0057861, 0.0058222
8: -0.0008533, 0.0000782, -0.0008533, 0.0000782, -0.0009060, 0.0008490
9: 0.9973557, 1.0121032, 0.9973557, 1.0121032, -0.0089766, 0.0093615

### Unstable ReLU Count (Linear/Conv2D Layers)
- layer_idx=0, type=LayerType.Linear, total=256, inp1_unstable=7, inp2_unstable=7, delta_unstable=251
- layer_idx=2, type=LayerType.Linear, total=256, inp1_unstable=10, inp2_unstable=10, delta_unstable=256
- layer_idx=4, type=LayerType.Linear, total=256, inp1_unstable=1, inp2_unstable=1, delta_unstable=256
- layer_idx=6, type=LayerType.Linear, total=256, inp1_unstable=0, inp2_unstable=0, delta_unstable=256
- layer_idx=8, type=LayerType.Linear, total=10, inp1_unstable=4, inp2_unstable=4, delta_unstable=10

Time for backsubstitution: 1.28 seconds

### RS candidates at layer 1
type: RSZ, layer: 1, pos: 148
type: RSZ, layer: 1, pos: 219

Time for candidate selection: 0.13 seconds

### Candidate
type: RSZ, layer: 1, pos: 148

### Relational analysis RSZ of RS_RSZ2_RSZ2_RSZ1_RSZ2_RSZ1_RSZ1

#### Relational analysis RSZ result of RS_RSZ2_RSZ2_RSZ1_RSZ2_RSZ1_RSZ1
Status: Status.UNKNOWN
Output dim: 9, lower bound: -0.0048224, upper bound: 0.0055022
time: 0.88 seconds

### Relational analysis RSZ of RS_RSZ2_RSZ2_RSZ1_RSZ2_RSZ1_RSZ2

#### Relational analysis RSZ result of RS_RSZ2_RSZ2_RSZ1_RSZ2_RSZ1_RSZ2
Status: Status.UNKNOWN
Output dim: 9, lower bound: -0.0048224, upper bound: 0.0055022
time: 0.88 seconds

## BFS RS instance: RS_RSZ2_RSZ2_RSZ1_RSZ2_RSZ2

### Backsubstitution after applying RS history:
0: -0.0039095, -0.0003029, -0.0039095, -0.0003029, -0.0031783, 0.0031893
1: -0.0044907, 0.0040812, -0.0044907, 0.0040812, -0.0039574, 0.0039074
2: 0.0029569, 0.0099431, 0.0029569, 0.0099431, -0.0039125, 0.0038830
3: -0.0045519, -0.0034923, -0.0045519, -0.0034923, -0.0010595, 0.0010595
4: 0.0017018, 0.0078700, 0.0017018, 0.0078700, -0.0052894, 0.0052976
5: -0.0030341, 0.0031953, -0.0030341, 0.0031953, -0.0047801, 0.0047555
6: -0.0068516, -0.0033211, -0.0068516, -0.0033211, -0.0033551, 0.0033511
7: -0.0021570, 0.0038259, -0.0021570, 0.0038259, -0.0058131, 0.0057917
8: -0.0008533, 0.0000782, -0.0008533, 0.0000782, -0.0008994, 0.0008551
9: 0.9973557, 1.0121032, 0.9973557, 1.0121032, -0.0091295, 0.0091852

### Unstable ReLU Count (Linear/Conv2D Layers)
- layer_idx=0, type=LayerType.Linear, total=256, inp1_unstable=7, inp2_unstable=7, delta_unstable=251
- layer_idx=2, type=LayerType.Linear, total=256, inp1_unstable=10, inp2_unstable=10, delta_unstable=256
- layer_idx=4, type=LayerType.Linear, total=256, inp1_unstable=1, inp2_unstable=1, delta_unstable=256
- layer_idx=6, type=LayerType.Linear, total=256, inp1_unstable=0, inp2_unstable=0, delta_unstable=256
- layer_idx=8, type=LayerType.Linear, total=10, inp1_unstable=4, inp2_unstable=4, delta_unstable=10

Time for backsubstitution: 1.30 seconds

### RS candidates at layer 1
type: RSZ, layer: 1, pos: 148
type: RSZ, layer: 1, pos: 219

Time for candidate selection: 0.12 seconds

### Candidate
type: RSZ, layer: 1, pos: 148

### Relational analysis RSZ of RS_RSZ2_RSZ2_RSZ1_RSZ2_RSZ2_RSZ1

#### Relational analysis RSZ result of RS_RSZ2_RSZ2_RSZ1_RSZ2_RSZ2_RSZ1
Status: Status.UNKNOWN
Output dim: 9, lower bound: -0.0049153, upper bound: 0.0053411
time: 0.86 seconds

### Relational analysis RSZ of RS_RSZ2_RSZ2_RSZ1_RSZ2_RSZ2_RSZ2

#### Relational analysis RSZ result of RS_RSZ2_RSZ2_RSZ1_RSZ2_RSZ2_RSZ2
Status: Status.UNKNOWN
Output dim: 9, lower bound: -0.0049153, upper bound: 0.0053411
time: 0.86 seconds

## BFS RS instance: RS_RSZ2_RSZ2_RSZ2_RSZ1_RSZ1

### Backsubstitution after applying RS history:
0: -0.0039095, -0.0003029, -0.0039095, -0.0003029, -0.0031868, 0.0031820
1: -0.0044907, 0.0040812, -0.0044907, 0.0040812, -0.0039497, 0.0039315
2: 0.0029569, 0.0099431, 0.0029569, 0.0099431, -0.0039121, 0.0038939
3: -0.0045519, -0.0034923, -0.0045519, -0.0034923, -0.0010595, 0.0010595
4: 0.0017018, 0.0078700, 0.0017018, 0.0078700, -0.0052961, 0.0052916
5: -0.0030341, 0.0031953, -0.0030341, 0.0031953, -0.0047755, 0.0047642
6: -0.0068516, -0.0033211, -0.0068516, -0.0033211, -0.0033541, 0.0033532
7: -0.0021570, 0.0038259, -0.0021570, 0.0038259, -0.0058068, 0.0058042
8: -0.0008533, 0.0000782, -0.0008533, 0.0000782, -0.0008848, 0.0008680
9: 0.9973557, 1.0121032, 0.9973557, 1.0121032, -0.0091616, 0.0091655

### Unstable ReLU Count (Linear/Conv2D Layers)
- layer_idx=0, type=LayerType.Linear, total=256, inp1_unstable=7, inp2_unstable=7, delta_unstable=251
- layer_idx=2, type=LayerType.Linear, total=256, inp1_unstable=10, inp2_unstable=10, delta_unstable=256
- layer_idx=4, type=LayerType.Linear, total=256, inp1_unstable=1, inp2_unstable=1, delta_unstable=256
- layer_idx=6, type=LayerType.Linear, total=256, inp1_unstable=0, inp2_unstable=0, delta_unstable=256
- layer_idx=8, type=LayerType.Linear, total=10, inp1_unstable=4, inp2_unstable=4, delta_unstable=10

Time for backsubstitution: 1.31 seconds

### RS candidates at layer 1
type: RSZ, layer: 1, pos: 148
type: RSZ, layer: 1, pos: 219

Time for candidate selection: 0.14 seconds

### Candidate
type: RSZ, layer: 1, pos: 148

### Relational analysis RSZ of RS_RSZ2_RSZ2_RSZ2_RSZ1_RSZ1_RSZ1

#### Relational analysis RSZ result of RS_RSZ2_RSZ2_RSZ2_RSZ1_RSZ1_RSZ1
Status: Status.UNKNOWN
Output dim: 9, lower bound: -0.0052293, upper bound: 0.0050919
time: 0.88 seconds

### Relational analysis RSZ of RS_RSZ2_RSZ2_RSZ2_RSZ1_RSZ1_RSZ2

#### Relational analysis RSZ result of RS_RSZ2_RSZ2_RSZ2_RSZ1_RSZ1_RSZ2
Status: Status.UNKNOWN
Output dim: 9, lower bound: -0.0052293, upper bound: 0.0050919
time: 0.87 seconds

## BFS RS instance: RS_RSZ2_RSZ2_RSZ2_RSZ1_RSZ2

### Backsubstitution after applying RS history:
0: -0.0039095, -0.0003029, -0.0039095, -0.0003029, -0.0032424, 0.0031349
1: -0.0044907, 0.0040812, -0.0044907, 0.0040812, -0.0038235, 0.0040571
2: 0.0029569, 0.0099431, 0.0029569, 0.0099431, -0.0038248, 0.0039782
3: -0.0045519, -0.0034923, -0.0045519, -0.0034923, -0.0010595, 0.0010595
4: 0.0017018, 0.0078700, 0.0017018, 0.0078700, -0.0053385, 0.0052556
5: -0.0030341, 0.0031953, -0.0030341, 0.0031953, -0.0047168, 0.0048277
6: -0.0068516, -0.0033211, -0.0068516, -0.0033211, -0.0033433, 0.0033644
7: -0.0021570, 0.0038259, -0.0021570, 0.0038259, -0.0058335, 0.0057744
8: -0.0008533, 0.0000782, -0.0008533, 0.0000782, -0.0008800, 0.0008754
9: 0.9973557, 1.0121032, 0.9973557, 1.0121032, -0.0093329, 0.0090048

### Unstable ReLU Count (Linear/Conv2D Layers)
- layer_idx=0, type=LayerType.Linear, total=256, inp1_unstable=7, inp2_unstable=7, delta_unstable=251
- layer_idx=2, type=LayerType.Linear, total=256, inp1_unstable=10, inp2_unstable=10, delta_unstable=256
- layer_idx=4, type=LayerType.Linear, total=256, inp1_unstable=1, inp2_unstable=1, delta_unstable=256
- layer_idx=6, type=LayerType.Linear, total=256, inp1_unstable=0, inp2_unstable=0, delta_unstable=256
- layer_idx=8, type=LayerType.Linear, total=10, inp1_unstable=4, inp2_unstable=4, delta_unstable=10

Time for backsubstitution: 1.30 seconds

### RS candidates at layer 1
type: RSZ, layer: 1, pos: 148
type: RSZ, layer: 1, pos: 219

Time for candidate selection: 0.13 seconds

### Candidate
type: RSZ, layer: 1, pos: 148

### Relational analysis RSZ of RS_RSZ2_RSZ2_RSZ2_RSZ1_RSZ2_RSZ1

#### Relational analysis RSZ result of RS_RSZ2_RSZ2_RSZ2_RSZ1_RSZ2_RSZ1
Status: Status.UNKNOWN
Output dim: 9, lower bound: -0.0053686, upper bound: 0.0049980
time: 0.94 seconds

### Relational analysis RSZ of RS_RSZ2_RSZ2_RSZ2_RSZ1_RSZ2_RSZ2

#### Relational analysis RSZ result of RS_RSZ2_RSZ2_RSZ2_RSZ1_RSZ2_RSZ2
Status: Status.UNKNOWN
Output dim: 9, lower bound: -0.0053686, upper bound: 0.0049980
time: 0.96 seconds

## BFS RS instance: RS_RSZ2_RSZ2_RSZ2_RSZ2_RSZ1

### Backsubstitution after applying RS history:
0: -0.0039095, -0.0003029, -0.0039095, -0.0003029, -0.0031152, 0.0032511
1: -0.0044907, 0.0040812, -0.0044907, 0.0040812, -0.0041021, 0.0037555
2: 0.0029569, 0.0099431, 0.0029569, 0.0099431, -0.0040102, 0.0037788
3: -0.0045519, -0.0034923, -0.0045519, -0.0034923, -0.0010595, 0.0010595
4: 0.0017018, 0.0078700, 0.0017018, 0.0078700, -0.0052411, 0.0053441
5: -0.0030341, 0.0031953, -0.0030341, 0.0031953, -0.0048522, 0.0046865
6: -0.0068516, -0.0033211, -0.0068516, -0.0033211, -0.0033678, 0.0033377
7: -0.0021570, 0.0038259, -0.0021570, 0.0038259, -0.0057801, 0.0058264
8: -0.0008533, 0.0000782, -0.0008533, 0.0000782, -0.0008948, 0.0008576
9: 0.9973557, 1.0121032, 0.9973557, 1.0121032, -0.0089247, 0.0093791

### Unstable ReLU Count (Linear/Conv2D Layers)
- layer_idx=0, type=LayerType.Linear, total=256, inp1_unstable=7, inp2_unstable=7, delta_unstable=251
- layer_idx=2, type=LayerType.Linear, total=256, inp1_unstable=10, inp2_unstable=10, delta_unstable=256
- layer_idx=4, type=LayerType.Linear, total=256, inp1_unstable=1, inp2_unstable=1, delta_unstable=256
- layer_idx=6, type=LayerType.Linear, total=256, inp1_unstable=0, inp2_unstable=0, delta_unstable=256
- layer_idx=8, type=LayerType.Linear, total=10, inp1_unstable=4, inp2_unstable=4, delta_unstable=10

Time for backsubstitution: 1.36 seconds

### RS candidates at layer 1
type: RSZ, layer: 1, pos: 148
type: RSZ, layer: 1, pos: 219

Time for candidate selection: 0.14 seconds

### Candidate
type: RSZ, layer: 1, pos: 148

### Relational analysis RSZ of RS_RSZ2_RSZ2_RSZ2_RSZ2_RSZ1_RSZ1

#### Relational analysis RSZ result of RS_RSZ2_RSZ2_RSZ2_RSZ2_RSZ1_RSZ1
Status: Status.UNKNOWN
Output dim: 9, lower bound: -0.0047578, upper bound: 0.0055118
time: 0.92 seconds

### Relational analysis RSZ of RS_RSZ2_RSZ2_RSZ2_RSZ2_RSZ1_RSZ2

#### Relational analysis RSZ result of RS_RSZ2_RSZ2_RSZ2_RSZ2_RSZ1_RSZ2
Status: Status.UNKNOWN
Output dim: 9, lower bound: -0.0047578, upper bound: 0.0055118
time: 0.90 seconds

## BFS RS instance: RS_RSZ2_RSZ2_RSZ2_RSZ2_RSZ2

### Backsubstitution after applying RS history:
0: -0.0039095, -0.0003029, -0.0039095, -0.0003029, -0.0031721, 0.0032025
1: -0.0044907, 0.0040812, -0.0044907, 0.0040812, -0.0039817, 0.0038948
2: 0.0029569, 0.0099431, 0.0029569, 0.0099431, -0.0039270, 0.0038746
3: -0.0045519, -0.0034923, -0.0045519, -0.0034923, -0.0010595, 0.0010595
4: 0.0017018, 0.0078700, 0.0017018, 0.0078700, -0.0052848, 0.0053073
5: -0.0030341, 0.0031953, -0.0030341, 0.0031953, -0.0047933, 0.0047482
6: -0.0068516, -0.0033211, -0.0068516, -0.0033211, -0.0033575, 0.0033497
7: -0.0021570, 0.0038259, -0.0021570, 0.0038259, -0.0058091, 0.0057976
8: -0.0008533, 0.0000782, -0.0008533, 0.0000782, -0.0008882, 0.0008644
9: 0.9973557, 1.0121032, 0.9973557, 1.0121032, -0.0091078, 0.0092222

### Unstable ReLU Count (Linear/Conv2D Layers)
- layer_idx=0, type=LayerType.Linear, total=256, inp1_unstable=7, inp2_unstable=7, delta_unstable=251
- layer_idx=2, type=LayerType.Linear, total=256, inp1_unstable=10, inp2_unstable=10, delta_unstable=256
- layer_idx=4, type=LayerType.Linear, total=256, inp1_unstable=1, inp2_unstable=1, delta_unstable=256
- layer_idx=6, type=LayerType.Linear, total=256, inp1_unstable=0, inp2_unstable=0, delta_unstable=256
- layer_idx=8, type=LayerType.Linear, total=10, inp1_unstable=4, inp2_unstable=4, delta_unstable=10

Time for backsubstitution: 1.29 seconds

### RS candidates at layer 1
type: RSZ, layer: 1, pos: 148
type: RSZ, layer: 1, pos: 219

Time for candidate selection: 0.13 seconds

### Candidate
type: RSZ, layer: 1, pos: 148

### Relational analysis RSZ of RS_RSZ2_RSZ2_RSZ2_RSZ2_RSZ2_RSZ1

#### Relational analysis RSZ result of RS_RSZ2_RSZ2_RSZ2_RSZ2_RSZ2_RSZ1
Status: Status.UNKNOWN
Output dim: 9, lower bound: -0.0048870, upper bound: 0.0053739
time: 0.88 seconds

### Relational analysis RSZ of RS_RSZ2_RSZ2_RSZ2_RSZ2_RSZ2_RSZ2

#### Relational analysis RSZ result of RS_RSZ2_RSZ2_RSZ2_RSZ2_RSZ2_RSZ2
Status: Status.UNKNOWN
Output dim: 9, lower bound: -0.0048870, upper bound: 0.0053739
time: 0.88 seconds

## Summary of splitting (split count: 5)
- Time for RS candidates: 3.18 seconds
RS_RSZ1_RSZ1_RSZ1_RSZ1_RSZ1_RSZ1, status: Status.UNKNOWN, split count: 6, time: 3.18
Output dim: 9, lower bound: -0.0053739, upper bound: 0.0048870
RS_RSZ1_RSZ1_RSZ1_RSZ1_RSZ1_RSZ2, status: Status.UNKNOWN, split count: 6, time: 3.18
Output dim: 9, lower bound: -0.0053739, upper bound: 0.0048870
RS_RSZ1_RSZ1_RSZ1_RSZ1_RSZ2_RSZ1, status: Status.UNKNOWN, split count: 6, time: 3.18
Output dim: 9, lower bound: -0.0055117, upper bound: 0.0047578
RS_RSZ1_RSZ1_RSZ1_RSZ1_RSZ2_RSZ2, status: Status.UNKNOWN, split count: 6, time: 3.18
Output dim: 9, lower bound: -0.0055117, upper bound: 0.0047578
RS_RSZ1_RSZ1_RSZ1_RSZ2_RSZ1_RSZ1, status: Status.UNKNOWN, split count: 6, time: 3.18
Output dim: 9, lower bound: -0.0049980, upper bound: 0.0053686
RS_RSZ1_RSZ1_RSZ1_RSZ2_RSZ1_RSZ2, status: Status.UNKNOWN, split count: 6, time: 3.18
Output dim: 9, lower bound: -0.0049980, upper bound: 0.0053686
RS_RSZ1_RSZ1_RSZ1_RSZ2_RSZ2_RSZ1, status: Status.UNKNOWN, split count: 6, time: 3.18
Output dim: 9, lower bound: -0.0050919, upper bound: 0.0052293
RS_RSZ1_RSZ1_RSZ1_RSZ2_RSZ2_RSZ2, status: Status.UNKNOWN, split count: 6, time: 3.18
Output dim: 9, lower bound: -0.0050919, upper bound: 0.0052293
RS_RSZ1_RSZ1_RSZ2_RSZ1_RSZ1_RSZ1, status: Status.UNKNOWN, split count: 6, time: 3.18
Output dim: 9, lower bound: -0.0053411, upper bound: 0.0049153
RS_RSZ1_RSZ1_RSZ2_RSZ1_RSZ1_RSZ2, status: Status.UNKNOWN, split count: 6, time: 3.18
Output dim: 9, lower bound: -0.0053411, upper bound: 0.0049153
RS_RSZ1_RSZ1_RSZ2_RSZ1_RSZ2_RSZ1, status: Status.UNKNOWN, split count: 6, time: 3.18
Output dim: 9, lower bound: -0.0055022, upper bound: 0.0048223
RS_RSZ1_RSZ1_RSZ2_RSZ1_RSZ2_RSZ2, status: Status.UNKNOWN, split count: 6, time: 3.18
Output dim: 9, lower bound: -0.0055022, upper bound: 0.0048223
RS_RSZ1_RSZ1_RSZ2_RSZ2_RSZ1_RSZ1, status: Status.UNKNOWN, split count: 6, time: 3.18
Output dim: 9, lower bound: -0.0049285, upper bound: 0.0053725
RS_RSZ1_RSZ1_RSZ2_RSZ2_RSZ1_RSZ2, status: Status.UNKNOWN, split count: 6, time: 3.18
Output dim: 9, lower bound: -0.0049285, upper bound: 0.0053725
RS_RSZ1_RSZ1_RSZ2_RSZ2_RSZ2_RSZ1, status: Status.UNKNOWN, split count: 6, time: 3.18
Output dim: 9, lower bound: -0.0050524, upper bound: 0.0052487
RS_RSZ1_RSZ1_RSZ2_RSZ2_RSZ2_RSZ2, status: Status.UNKNOWN, split count: 6, time: 3.18
Output dim: 9, lower bound: -0.0050524, upper bound: 0.0052487
RS_RSZ1_RSZ2_RSZ1_RSZ1_RSZ1_RSZ1, status: Status.UNKNOWN, split count: 6, time: 3.18
Output dim: 9, lower bound: -0.0053004, upper bound: 0.0050235
RS_RSZ1_RSZ2_RSZ1_RSZ1_RSZ1_RSZ2, status: Status.UNKNOWN, split count: 6, time: 3.18
Output dim: 9, lower bound: -0.0053004, upper bound: 0.0050235
RS_RSZ1_RSZ2_RSZ1_RSZ1_RSZ2_RSZ1, status: Status.UNKNOWN, split count: 6, time: 3.18
Output dim: 9, lower bound: -0.0054043, upper bound: 0.0048824
RS_RSZ1_RSZ2_RSZ1_RSZ1_RSZ2_RSZ2, status: Status.UNKNOWN, split count: 6, time: 3.18
Output dim: 9, lower bound: -0.0054043, upper bound: 0.0048833
RS_RSZ1_RSZ2_RSZ1_RSZ2_RSZ1_RSZ1, status: Status.UNKNOWN, split count: 6, time: 3.18
Output dim: 9, lower bound: -0.0048879, upper bound: 0.0054862
RS_RSZ1_RSZ2_RSZ1_RSZ2_RSZ1_RSZ2, status: Status.UNKNOWN, split count: 6, time: 3.18
Output dim: 9, lower bound: -0.0048879, upper bound: 0.0054862
RS_RSZ1_RSZ2_RSZ1_RSZ2_RSZ2_RSZ1, status: Status.UNKNOWN, split count: 6, time: 3.18
Output dim: 9, lower bound: -0.0049834, upper bound: 0.0053090
RS_RSZ1_RSZ2_RSZ1_RSZ2_RSZ2_RSZ2, status: Status.UNKNOWN, split count: 6, time: 3.18
Output dim: 9, lower bound: -0.0049834, upper bound: 0.0053090
RS_RSZ1_RSZ2_RSZ2_RSZ1_RSZ1_RSZ1, status: Status.UNKNOWN, split count: 6, time: 3.18
Output dim: 9, lower bound: -0.0052807, upper bound: 0.0050511
RS_RSZ1_RSZ2_RSZ2_RSZ1_RSZ1_RSZ2, status: Status.UNKNOWN, split count: 6, time: 3.18
Output dim: 9, lower bound: -0.0052807, upper bound: 0.0050511
RS_RSZ1_RSZ2_RSZ2_RSZ1_RSZ2_RSZ1, status: Status.UNKNOWN, split count: 6, time: 3.18
Output dim: 9, lower bound: -0.0054016, upper bound: 0.0049486
RS_RSZ1_RSZ2_RSZ2_RSZ1_RSZ2_RSZ2, status: Status.UNKNOWN, split count: 6, time: 3.18
Output dim: 9, lower bound: -0.0054016, upper bound: 0.0049486
RS_RSZ1_RSZ2_RSZ2_RSZ2_RSZ1_RSZ1, status: Status.UNKNOWN, split count: 6, time: 3.18
Output dim: 9, lower bound: -0.0048224, upper bound: 0.0054973
RS_RSZ1_RSZ2_RSZ2_RSZ2_RSZ1_RSZ2, status: Status.UNKNOWN, split count: 6, time: 3.18
Output dim: 9, lower bound: -0.0048224, upper bound: 0.0054973
RS_RSZ1_RSZ2_RSZ2_RSZ2_RSZ2_RSZ1, status: Status.UNKNOWN, split count: 6, time: 3.18
Output dim: 9, lower bound: -0.0049340, upper bound: 0.0053480
RS_RSZ1_RSZ2_RSZ2_RSZ2_RSZ2_RSZ2, status: Status.UNKNOWN, split count: 6, time: 3.18
Output dim: 9, lower bound: -0.0049340, upper bound: 0.0053480
RS_RSZ2_RSZ1_RSZ1_RSZ1_RSZ1_RSZ1, status: Status.UNKNOWN, split count: 6, time: 3.18
Output dim: 9, lower bound: -0.0053480, upper bound: 0.0049340
RS_RSZ2_RSZ1_RSZ1_RSZ1_RSZ1_RSZ2, status: Status.UNKNOWN, split count: 6, time: 3.18
Output dim: 9, lower bound: -0.0053480, upper bound: 0.0049340
RS_RSZ2_RSZ1_RSZ1_RSZ1_RSZ2_RSZ1, status: Status.UNKNOWN, split count: 6, time: 3.18
Output dim: 9, lower bound: -0.0054973, upper bound: 0.0048224
RS_RSZ2_RSZ1_RSZ1_RSZ1_RSZ2_RSZ2, status: Status.UNKNOWN, split count: 6, time: 3.18
Output dim: 9, lower bound: -0.0054973, upper bound: 0.0048224
RS_RSZ2_RSZ1_RSZ1_RSZ2_RSZ1_RSZ1, status: Status.UNKNOWN, split count: 6, time: 3.18
Output dim: 9, lower bound: -0.0049486, upper bound: 0.0054016
RS_RSZ2_RSZ1_RSZ1_RSZ2_RSZ1_RSZ2, status: Status.UNKNOWN, split count: 6, time: 3.18
Output dim: 9, lower bound: -0.0049486, upper bound: 0.0054016
RS_RSZ2_RSZ1_RSZ1_RSZ2_RSZ2_RSZ1, status: Status.UNKNOWN, split count: 6, time: 3.18
Output dim: 9, lower bound: -0.0050511, upper bound: 0.0052807
RS_RSZ2_RSZ1_RSZ1_RSZ2_RSZ2_RSZ2, status: Status.UNKNOWN, split count: 6, time: 3.18
Output dim: 9, lower bound: -0.0050511, upper bound: 0.0052807
RS_RSZ2_RSZ1_RSZ2_RSZ1_RSZ1_RSZ1, status: Status.UNKNOWN, split count: 6, time: 3.18
Output dim: 9, lower bound: -0.0053090, upper bound: 0.0049834
RS_RSZ2_RSZ1_RSZ2_RSZ1_RSZ1_RSZ2, status: Status.UNKNOWN, split count: 6, time: 3.18
Output dim: 9, lower bound: -0.0053090, upper bound: 0.0049834
RS_RSZ2_RSZ1_RSZ2_RSZ1_RSZ2_RSZ1, status: Status.UNKNOWN, split count: 6, time: 3.18
Output dim: 9, lower bound: -0.0054862, upper bound: 0.0048879
RS_RSZ2_RSZ1_RSZ2_RSZ1_RSZ2_RSZ2, status: Status.UNKNOWN, split count: 6, time: 3.18
Output dim: 9, lower bound: -0.0054862, upper bound: 0.0048879
RS_RSZ2_RSZ1_RSZ2_RSZ2_RSZ1_RSZ1, status: Status.UNKNOWN, split count: 6, time: 3.18
Output dim: 9, lower bound: -0.0048833, upper bound: 0.0054043
RS_RSZ2_RSZ1_RSZ2_RSZ2_RSZ1_RSZ2, status: Status.UNKNOWN, split count: 6, time: 3.18
Output dim: 9, lower bound: -0.0048824, upper bound: 0.0054043
RS_RSZ2_RSZ1_RSZ2_RSZ2_RSZ2_RSZ1, status: Status.UNKNOWN, split count: 6, time: 3.18
Output dim: 9, lower bound: -0.0050235, upper bound: 0.0053004
RS_RSZ2_RSZ1_RSZ2_RSZ2_RSZ2_RSZ2, status: Status.UNKNOWN, split count: 6, time: 3.18
Output dim: 9, lower bound: -0.0050235, upper bound: 0.0053004
RS_RSZ2_RSZ2_RSZ1_RSZ1_RSZ1_RSZ1, status: Status.UNKNOWN, split count: 6, time: 3.18
Output dim: 9, lower bound: -0.0052487, upper bound: 0.0050524
RS_RSZ2_RSZ2_RSZ1_RSZ1_RSZ1_RSZ2, status: Status.UNKNOWN, split count: 6, time: 3.18
Output dim: 9, lower bound: -0.0052487, upper bound: 0.0050524
RS_RSZ2_RSZ2_RSZ1_RSZ1_RSZ2_RSZ1, status: Status.UNKNOWN, split count: 6, time: 3.18
Output dim: 9, lower bound: -0.0053725, upper bound: 0.0049285
RS_RSZ2_RSZ2_RSZ1_RSZ1_RSZ2_RSZ2, status: Status.UNKNOWN, split count: 6, time: 3.18
Output dim: 9, lower bound: -0.0053725, upper bound: 0.0049285
RS_RSZ2_RSZ2_RSZ1_RSZ2_RSZ1_RSZ1, status: Status.UNKNOWN, split count: 6, time: 3.18
Output dim: 9, lower bound: -0.0048224, upper bound: 0.0055022
RS_RSZ2_RSZ2_RSZ1_RSZ2_RSZ1_RSZ2, status: Status.UNKNOWN, split count: 6, time: 3.18
Output dim: 9, lower bound: -0.0048224, upper bound: 0.0055022
RS_RSZ2_RSZ2_RSZ1_RSZ2_RSZ2_RSZ1, status: Status.UNKNOWN, split count: 6, time: 3.18
Output dim: 9, lower bound: -0.0049153, upper bound: 0.0053411
RS_RSZ2_RSZ2_RSZ1_RSZ2_RSZ2_RSZ2, status: Status.UNKNOWN, split count: 6, time: 3.18
Output dim: 9, lower bound: -0.0049153, upper bound: 0.0053411
RS_RSZ2_RSZ2_RSZ2_RSZ1_RSZ1_RSZ1, status: Status.UNKNOWN, split count: 6, time: 3.18
Output dim: 9, lower bound: -0.0052293, upper bound: 0.0050919
RS_RSZ2_RSZ2_RSZ2_RSZ1_RSZ1_RSZ2, status: Status.UNKNOWN, split count: 6, time: 3.18
Output dim: 9, lower bound: -0.0052293, upper bound: 0.0050919
RS_RSZ2_RSZ2_RSZ2_RSZ1_RSZ2_RSZ1, status: Status.UNKNOWN, split count: 6, time: 3.18
Output dim: 9, lower bound: -0.0053686, upper bound: 0.0049980
RS_RSZ2_RSZ2_RSZ2_RSZ1_RSZ2_RSZ2, status: Status.UNKNOWN, split count: 6, time: 3.18
Output dim: 9, lower bound: -0.0053686, upper bound: 0.0049980
RS_RSZ2_RSZ2_RSZ2_RSZ2_RSZ1_RSZ1, status: Status.UNKNOWN, split count: 6, time: 3.18
Output dim: 9, lower bound: -0.0047578, upper bound: 0.0055118
RS_RSZ2_RSZ2_RSZ2_RSZ2_RSZ1_RSZ2, status: Status.UNKNOWN, split count: 6, time: 3.18
Output dim: 9, lower bound: -0.0047578, upper bound: 0.0055118
RS_RSZ2_RSZ2_RSZ2_RSZ2_RSZ2_RSZ1, status: Status.UNKNOWN, split count: 6, time: 3.18
Output dim: 9, lower bound: -0.0048870, upper bound: 0.0053739
RS_RSZ2_RSZ2_RSZ2_RSZ2_RSZ2_RSZ2, status: Status.UNKNOWN, split count: 6, time: 3.18
Output dim: 9, lower bound: -0.0048870, upper bound: 0.0053739

## BFS RS instance: RS_RSZ1_RSZ1_RSZ1_RSZ1_RSZ1_RSZ1

### Backsubstitution after applying RS history:
0: -0.0039095, -0.0003029, -0.0039095, -0.0003029, -0.0031922, 0.0031614
1: -0.0044907, 0.0040812, -0.0044907, 0.0040812, -0.0038918, 0.0039786
2: 0.0029569, 0.0099431, 0.0029569, 0.0099431, -0.0038623, 0.0039147
3: -0.0045519, -0.0034923, -0.0045519, -0.0034923, -0.0010595, 0.0010595
4: 0.0017018, 0.0078700, 0.0017018, 0.0078700, -0.0053033, 0.0052804
5: -0.0030341, 0.0031953, -0.0030341, 0.0031953, -0.0047346, 0.0047797
6: -0.0068516, -0.0033211, -0.0068516, -0.0033211, -0.0033495, 0.0033573
7: -0.0021570, 0.0038259, -0.0021570, 0.0038259, -0.0057675, 0.0057782
8: -0.0008533, 0.0000782, -0.0008533, 0.0000782, -0.0008482, 0.0008702
9: 0.9973557, 1.0121032, 0.9973557, 1.0121032, -0.0092151, 0.0091005

### Unstable ReLU Count (Linear/Conv2D Layers)
- layer_idx=0, type=LayerType.Linear, total=256, inp1_unstable=7, inp2_unstable=7, delta_unstable=250
- layer_idx=2, type=LayerType.Linear, total=256, inp1_unstable=10, inp2_unstable=10, delta_unstable=256
- layer_idx=4, type=LayerType.Linear, total=256, inp1_unstable=1, inp2_unstable=1, delta_unstable=256
- layer_idx=6, type=LayerType.Linear, total=256, inp1_unstable=0, inp2_unstable=0, delta_unstable=256
- layer_idx=8, type=LayerType.Linear, total=10, inp1_unstable=4, inp2_unstable=4, delta_unstable=10

Time for backsubstitution: 1.30 seconds

### RS candidates at layer 1
type: RSZ, layer: 1, pos: 219

Time for candidate selection: 0.14 seconds

### Candidate
type: RSZ, layer: 1, pos: 219

### Relational analysis RSZ of RS_RSZ1_RSZ1_RSZ1_RSZ1_RSZ1_RSZ1_RSZ1

#### Relational analysis RSZ result of RS_RSZ1_RSZ1_RSZ1_RSZ1_RSZ1_RSZ1_RSZ1
Status: Status.UNKNOWN
Output dim: 9, lower bound: -0.0053505, upper bound: 0.0048081
time: 0.74 seconds

### Relational analysis RSZ of RS_RSZ1_RSZ1_RSZ1_RSZ1_RSZ1_RSZ1_RSZ2

#### Relational analysis RSZ result of RS_RSZ1_RSZ1_RSZ1_RSZ1_RSZ1_RSZ1_RSZ2
Status: Status.UNKNOWN
Output dim: 9, lower bound: -0.0051431, upper bound: 0.0048608
time: 0.84 seconds

## BFS RS instance: RS_RSZ1_RSZ1_RSZ1_RSZ1_RSZ1_RSZ2

### Backsubstitution after applying RS history:
0: -0.0039095, -0.0003029, -0.0039095, -0.0003029, -0.0031918, 0.0031620
1: -0.0044907, 0.0040812, -0.0044907, 0.0040812, -0.0038917, 0.0039784
2: 0.0029569, 0.0099431, 0.0029569, 0.0099431, -0.0038624, 0.0039144
3: -0.0045519, -0.0034923, -0.0045519, -0.0034923, -0.0010595, 0.0010595
4: 0.0017018, 0.0078700, 0.0017018, 0.0078700, -0.0053030, 0.0052806
5: -0.0030341, 0.0031953, -0.0030341, 0.0031953, -0.0047359, 0.0047797
6: -0.0068516, -0.0033211, -0.0068516, -0.0033211, -0.0033495, 0.0033573
7: -0.0021570, 0.0038259, -0.0021570, 0.0038259, -0.0057678, 0.0057790
8: -0.0008533, 0.0000782, -0.0008533, 0.0000782, -0.0008493, 0.0008720
9: 0.9973557, 1.0121032, 0.9973557, 1.0121032, -0.0092149, 0.0091006

### Unstable ReLU Count (Linear/Conv2D Layers)
- layer_idx=0, type=LayerType.Linear, total=256, inp1_unstable=7, inp2_unstable=7, delta_unstable=250
- layer_idx=2, type=LayerType.Linear, total=256, inp1_unstable=10, inp2_unstable=10, delta_unstable=256
- layer_idx=4, type=LayerType.Linear, total=256, inp1_unstable=1, inp2_unstable=1, delta_unstable=256
- layer_idx=6, type=LayerType.Linear, total=256, inp1_unstable=0, inp2_unstable=0, delta_unstable=256
- layer_idx=8, type=LayerType.Linear, total=10, inp1_unstable=4, inp2_unstable=4, delta_unstable=10

Time for backsubstitution: 1.27 seconds

### RS candidates at layer 1
type: RSZ, layer: 1, pos: 219

Time for candidate selection: 0.13 seconds

### Candidate
type: RSZ, layer: 1, pos: 219

### Relational analysis RSZ of RS_RSZ1_RSZ1_RSZ1_RSZ1_RSZ1_RSZ2_RSZ1

#### Relational analysis RSZ result of RS_RSZ1_RSZ1_RSZ1_RSZ1_RSZ1_RSZ2_RSZ1
Status: Status.UNKNOWN
Output dim: 9, lower bound: -0.0053505, upper bound: 0.0048081
time: 0.75 seconds

### Relational analysis RSZ of RS_RSZ1_RSZ1_RSZ1_RSZ1_RSZ1_RSZ2_RSZ2

#### Relational analysis RSZ result of RS_RSZ1_RSZ1_RSZ1_RSZ1_RSZ1_RSZ2_RSZ2
Status: Status.UNKNOWN
Output dim: 9, lower bound: -0.0051431, upper bound: 0.0048608
time: 0.83 seconds

## BFS RS instance: RS_RSZ1_RSZ1_RSZ1_RSZ1_RSZ2_RSZ1

### Backsubstitution after applying RS history:
0: -0.0039095, -0.0003029, -0.0039095, -0.0003029, -0.0032404, 0.0031045
1: -0.0044907, 0.0040812, -0.0044907, 0.0040812, -0.0037520, 0.0040990
2: 0.0029569, 0.0099431, 0.0029569, 0.0099431, -0.0037662, 0.0039979
3: -0.0045519, -0.0034923, -0.0045519, -0.0034923, -0.0010595, 0.0010595
4: 0.0017018, 0.0078700, 0.0017018, 0.0078700, -0.0053397, 0.0052368
5: -0.0030341, 0.0031953, -0.0030341, 0.0031953, -0.0046729, 0.0048383
6: -0.0068516, -0.0033211, -0.0068516, -0.0033211, -0.0033375, 0.0033677
7: -0.0021570, 0.0038259, -0.0021570, 0.0038259, -0.0057962, 0.0057493
8: -0.0008533, 0.0000782, -0.0008533, 0.0000782, -0.0008413, 0.0008767
9: 0.9973557, 1.0121032, 0.9973557, 1.0121032, -0.0093716, 0.0089173

### Unstable ReLU Count (Linear/Conv2D Layers)
- layer_idx=0, type=LayerType.Linear, total=256, inp1_unstable=7, inp2_unstable=7, delta_unstable=250
- layer_idx=2, type=LayerType.Linear, total=256, inp1_unstable=10, inp2_unstable=10, delta_unstable=256
- layer_idx=4, type=LayerType.Linear, total=256, inp1_unstable=1, inp2_unstable=1, delta_unstable=256
- layer_idx=6, type=LayerType.Linear, total=256, inp1_unstable=0, inp2_unstable=0, delta_unstable=256
- layer_idx=8, type=LayerType.Linear, total=10, inp1_unstable=4, inp2_unstable=4, delta_unstable=10

Time for backsubstitution: 1.27 seconds

### RS candidates at layer 1
type: RSZ, layer: 1, pos: 219

Time for candidate selection: 0.13 seconds

### Candidate
type: RSZ, layer: 1, pos: 219

### Relational analysis RSZ of RS_RSZ1_RSZ1_RSZ1_RSZ1_RSZ2_RSZ1_RSZ1

#### Relational analysis RSZ result of RS_RSZ1_RSZ1_RSZ1_RSZ1_RSZ2_RSZ1_RSZ1
Status: Status.UNKNOWN
Output dim: 9, lower bound: -0.0054877, upper bound: 0.0046830
time: 0.77 seconds

### Relational analysis RSZ of RS_RSZ1_RSZ1_RSZ1_RSZ1_RSZ2_RSZ1_RSZ2

#### Relational analysis RSZ result of RS_RSZ1_RSZ1_RSZ1_RSZ1_RSZ2_RSZ1_RSZ2
Status: Status.UNKNOWN
Output dim: 9, lower bound: -0.0053148, upper bound: 0.0047324
time: 0.81 seconds

## BFS RS instance: RS_RSZ1_RSZ1_RSZ1_RSZ1_RSZ2_RSZ2

### Backsubstitution after applying RS history:
0: -0.0039095, -0.0003029, -0.0039095, -0.0003029, -0.0032404, 0.0031053
1: -0.0044907, 0.0040812, -0.0044907, 0.0040812, -0.0037524, 0.0040991
2: 0.0029569, 0.0099431, 0.0029569, 0.0099431, -0.0037666, 0.0039978
3: -0.0045519, -0.0034923, -0.0045519, -0.0034923, -0.0010595, 0.0010595
4: 0.0017018, 0.0078700, 0.0017018, 0.0078700, -0.0053397, 0.0052373
5: -0.0030341, 0.0031953, -0.0030341, 0.0031953, -0.0046760, 0.0048386
6: -0.0068516, -0.0033211, -0.0068516, -0.0033211, -0.0033375, 0.0033677
7: -0.0021570, 0.0038259, -0.0021570, 0.0038259, -0.0057974, 0.0057500
8: -0.0008533, 0.0000782, -0.0008533, 0.0000782, -0.0008423, 0.0008786
9: 0.9973557, 1.0121032, 0.9973557, 1.0121032, -0.0093718, 0.0089183

### Unstable ReLU Count (Linear/Conv2D Layers)
- layer_idx=0, type=LayerType.Linear, total=256, inp1_unstable=7, inp2_unstable=7, delta_unstable=250
- layer_idx=2, type=LayerType.Linear, total=256, inp1_unstable=10, inp2_unstable=10, delta_unstable=256
- layer_idx=4, type=LayerType.Linear, total=256, inp1_unstable=1, inp2_unstable=1, delta_unstable=256
- layer_idx=6, type=LayerType.Linear, total=256, inp1_unstable=0, inp2_unstable=0, delta_unstable=256
- layer_idx=8, type=LayerType.Linear, total=10, inp1_unstable=4, inp2_unstable=4, delta_unstable=10

Time for backsubstitution: 1.31 seconds

### RS candidates at layer 1
type: RSZ, layer: 1, pos: 219

Time for candidate selection: 0.14 seconds

### Candidate
type: RSZ, layer: 1, pos: 219

### Relational analysis RSZ of RS_RSZ1_RSZ1_RSZ1_RSZ1_RSZ2_RSZ2_RSZ1

#### Relational analysis RSZ result of RS_RSZ1_RSZ1_RSZ1_RSZ1_RSZ2_RSZ2_RSZ1
Status: Status.UNKNOWN
Output dim: 9, lower bound: -0.0054877, upper bound: 0.0046830
time: 0.78 seconds

### Relational analysis RSZ of RS_RSZ1_RSZ1_RSZ1_RSZ1_RSZ2_RSZ2_RSZ2

#### Relational analysis RSZ result of RS_RSZ1_RSZ1_RSZ1_RSZ1_RSZ2_RSZ2_RSZ2
Status: Status.UNKNOWN
Output dim: 9, lower bound: -0.0053154, upper bound: 0.0047324
time: 0.82 seconds

## BFS RS instance: RS_RSZ1_RSZ1_RSZ1_RSZ2_RSZ1_RSZ1

### Backsubstitution after applying RS history:
0: -0.0039095, -0.0003029, -0.0039095, -0.0003029, -0.0031248, 0.0032317
1: -0.0044907, 0.0040812, -0.0044907, 0.0040812, -0.0040541, 0.0038204
2: 0.0029569, 0.0099431, 0.0029569, 0.0099431, -0.0039664, 0.0038126
3: -0.0045519, -0.0034923, -0.0045519, -0.0034923, -0.0010595, 0.0010595
4: 0.0017018, 0.0078700, 0.0017018, 0.0078700, -0.0052516, 0.0053342
5: -0.0030341, 0.0031953, -0.0030341, 0.0031953, -0.0048141, 0.0047048
6: -0.0068516, -0.0033211, -0.0068516, -0.0033211, -0.0033642, 0.0033432
7: -0.0021570, 0.0038259, -0.0021570, 0.0038259, -0.0057442, 0.0058026
8: -0.0008533, 0.0000782, -0.0008533, 0.0000782, -0.0008591, 0.0008628
9: 0.9973557, 1.0121032, 0.9973557, 1.0121032, -0.0089980, 0.0093256

### Unstable ReLU Count (Linear/Conv2D Layers)
- layer_idx=0, type=LayerType.Linear, total=256, inp1_unstable=7, inp2_unstable=7, delta_unstable=250
- layer_idx=2, type=LayerType.Linear, total=256, inp1_unstable=10, inp2_unstable=10, delta_unstable=256
- layer_idx=4, type=LayerType.Linear, total=256, inp1_unstable=1, inp2_unstable=1, delta_unstable=256
- layer_idx=6, type=LayerType.Linear, total=256, inp1_unstable=0, inp2_unstable=0, delta_unstable=256
- layer_idx=8, type=LayerType.Linear, total=10, inp1_unstable=4, inp2_unstable=4, delta_unstable=10

Time for backsubstitution: 1.43 seconds

### RS candidates at layer 1
type: RSZ, layer: 1, pos: 219

Time for candidate selection: 0.16 seconds

### Candidate
type: RSZ, layer: 1, pos: 219

### Relational analysis RSZ of RS_RSZ1_RSZ1_RSZ1_RSZ2_RSZ1_RSZ1_RSZ1

#### Relational analysis RSZ result of RS_RSZ1_RSZ1_RSZ1_RSZ2_RSZ1_RSZ1_RSZ1
Status: Status.UNKNOWN
Output dim: 9, lower bound: -0.0049733, upper bound: 0.0051592
time: 0.82 seconds

### Relational analysis RSZ of RS_RSZ1_RSZ1_RSZ1_RSZ2_RSZ1_RSZ1_RSZ2

#### Relational analysis RSZ result of RS_RSZ1_RSZ1_RSZ1_RSZ2_RSZ1_RSZ1_RSZ2
Status: Status.UNKNOWN
Output dim: 9, lower bound: -0.0048917, upper bound: 0.0053453
time: 0.85 seconds

## BFS RS instance: RS_RSZ1_RSZ1_RSZ1_RSZ2_RSZ1_RSZ2

### Backsubstitution after applying RS history:
0: -0.0039095, -0.0003029, -0.0039095, -0.0003029, -0.0031242, 0.0032320
1: -0.0044907, 0.0040812, -0.0044907, 0.0040812, -0.0040540, 0.0038202
2: 0.0029569, 0.0099431, 0.0029569, 0.0099431, -0.0039660, 0.0038120
3: -0.0045519, -0.0034923, -0.0045519, -0.0034923, -0.0010595, 0.0010595
4: 0.0017018, 0.0078700, 0.0017018, 0.0078700, -0.0052512, 0.0053343
5: -0.0030341, 0.0031953, -0.0030341, 0.0031953, -0.0048146, 0.0047032
6: -0.0068516, -0.0033211, -0.0068516, -0.0033211, -0.0033642, 0.0033431
7: -0.0021570, 0.0038259, -0.0021570, 0.0038259, -0.0057451, 0.0058033
8: -0.0008533, 0.0000782, -0.0008533, 0.0000782, -0.0008592, 0.0008638
9: 0.9973557, 1.0121032, 0.9973557, 1.0121032, -0.0089974, 0.0093255

### Unstable ReLU Count (Linear/Conv2D Layers)
- layer_idx=0, type=LayerType.Linear, total=256, inp1_unstable=7, inp2_unstable=7, delta_unstable=250
- layer_idx=2, type=LayerType.Linear, total=256, inp1_unstable=10, inp2_unstable=10, delta_unstable=256
- layer_idx=4, type=LayerType.Linear, total=256, inp1_unstable=1, inp2_unstable=1, delta_unstable=256
- layer_idx=6, type=LayerType.Linear, total=256, inp1_unstable=0, inp2_unstable=0, delta_unstable=256
- layer_idx=8, type=LayerType.Linear, total=10, inp1_unstable=4, inp2_unstable=4, delta_unstable=10

Time for backsubstitution: 1.31 seconds

### RS candidates at layer 1
type: RSZ, layer: 1, pos: 219

Time for candidate selection: 0.14 seconds

### Candidate
type: RSZ, layer: 1, pos: 219

### Relational analysis RSZ of RS_RSZ1_RSZ1_RSZ1_RSZ2_RSZ1_RSZ2_RSZ1

#### Relational analysis RSZ result of RS_RSZ1_RSZ1_RSZ1_RSZ2_RSZ1_RSZ2_RSZ1
Status: Status.UNKNOWN
Output dim: 9, lower bound: -0.0049733, upper bound: 0.0051584
time: 0.77 seconds

### Relational analysis RSZ of RS_RSZ1_RSZ1_RSZ1_RSZ2_RSZ1_RSZ2_RSZ2

#### Relational analysis RSZ result of RS_RSZ1_RSZ1_RSZ1_RSZ2_RSZ1_RSZ2_RSZ2
Status: Status.UNKNOWN
Output dim: 9, lower bound: -0.0048917, upper bound: 0.0053453
time: 0.81 seconds

## BFS RS instance: RS_RSZ1_RSZ1_RSZ1_RSZ2_RSZ2_RSZ1

### Backsubstitution after applying RS history:
0: -0.0039095, -0.0003029, -0.0039095, -0.0003029, -0.0031716, 0.0031761
1: -0.0044907, 0.0040812, -0.0044907, 0.0040812, -0.0039283, 0.0039466
2: 0.0029569, 0.0099431, 0.0029569, 0.0099431, -0.0038821, 0.0038999
3: -0.0045519, -0.0034923, -0.0045519, -0.0034923, -0.0010595, 0.0010595
4: 0.0017018, 0.0078700, 0.0017018, 0.0078700, -0.0052876, 0.0052918
5: -0.0030341, 0.0031953, -0.0030341, 0.0031953, -0.0047506, 0.0047625
6: -0.0068516, -0.0033211, -0.0068516, -0.0033211, -0.0033530, 0.0033539
7: -0.0021570, 0.0038259, -0.0021570, 0.0038259, -0.0057741, 0.0057752
8: -0.0008533, 0.0000782, -0.0008533, 0.0000782, -0.0008518, 0.0008663
9: 0.9973557, 1.0121032, 0.9973557, 1.0121032, -0.0091584, 0.0091543

### Unstable ReLU Count (Linear/Conv2D Layers)
- layer_idx=0, type=LayerType.Linear, total=256, inp1_unstable=7, inp2_unstable=7, delta_unstable=250
- layer_idx=2, type=LayerType.Linear, total=256, inp1_unstable=10, inp2_unstable=10, delta_unstable=256
- layer_idx=4, type=LayerType.Linear, total=256, inp1_unstable=1, inp2_unstable=1, delta_unstable=256
- layer_idx=6, type=LayerType.Linear, total=256, inp1_unstable=0, inp2_unstable=0, delta_unstable=256
- layer_idx=8, type=LayerType.Linear, total=10, inp1_unstable=4, inp2_unstable=4, delta_unstable=10

Time for backsubstitution: 1.28 seconds

### RS candidates at layer 1
type: RSZ, layer: 1, pos: 219

Time for candidate selection: 0.14 seconds

### Candidate
type: RSZ, layer: 1, pos: 219

### Relational analysis RSZ of RS_RSZ1_RSZ1_RSZ1_RSZ2_RSZ2_RSZ1_RSZ1

#### Relational analysis RSZ result of RS_RSZ1_RSZ1_RSZ1_RSZ2_RSZ2_RSZ1_RSZ1
Status: Status.UNKNOWN
Output dim: 9, lower bound: -0.0050645, upper bound: 0.0050105
time: 0.76 seconds

### Relational analysis RSZ of RS_RSZ1_RSZ1_RSZ1_RSZ2_RSZ2_RSZ1_RSZ2

#### Relational analysis RSZ result of RS_RSZ1_RSZ1_RSZ1_RSZ2_RSZ2_RSZ1_RSZ2
Status: Status.UNKNOWN
Output dim: 9, lower bound: -0.0050176, upper bound: 0.0052068
time: 0.82 seconds

## BFS RS instance: RS_RSZ1_RSZ1_RSZ1_RSZ2_RSZ2_RSZ2

### Backsubstitution after applying RS history:
0: -0.0039095, -0.0003029, -0.0039095, -0.0003029, -0.0031713, 0.0031764
1: -0.0044907, 0.0040812, -0.0044907, 0.0040812, -0.0039285, 0.0039467
2: 0.0029569, 0.0099431, 0.0029569, 0.0099431, -0.0038817, 0.0038999
3: -0.0045519, -0.0034923, -0.0045519, -0.0034923, -0.0010595, 0.0010595
4: 0.0017018, 0.0078700, 0.0017018, 0.0078700, -0.0052873, 0.0052920
5: -0.0030341, 0.0031953, -0.0030341, 0.0031953, -0.0047517, 0.0047619
6: -0.0068516, -0.0033211, -0.0068516, -0.0033211, -0.0033530, 0.0033539
7: -0.0021570, 0.0038259, -0.0021570, 0.0038259, -0.0057745, 0.0057767
8: -0.0008533, 0.0000782, -0.0008533, 0.0000782, -0.0008513, 0.0008686
9: 0.9973557, 1.0121032, 0.9973557, 1.0121032, -0.0091582, 0.0091547

### Unstable ReLU Count (Linear/Conv2D Layers)
- layer_idx=0, type=LayerType.Linear, total=256, inp1_unstable=7, inp2_unstable=7, delta_unstable=250
- layer_idx=2, type=LayerType.Linear, total=256, inp1_unstable=10, inp2_unstable=10, delta_unstable=256
- layer_idx=4, type=LayerType.Linear, total=256, inp1_unstable=1, inp2_unstable=1, delta_unstable=256
- layer_idx=6, type=LayerType.Linear, total=256, inp1_unstable=0, inp2_unstable=0, delta_unstable=256
- layer_idx=8, type=LayerType.Linear, total=10, inp1_unstable=4, inp2_unstable=4, delta_unstable=10

Time for backsubstitution: 1.32 seconds

### RS candidates at layer 1
type: RSZ, layer: 1, pos: 219

Time for candidate selection: 0.14 seconds

### Candidate
type: RSZ, layer: 1, pos: 219

### Relational analysis RSZ of RS_RSZ1_RSZ1_RSZ1_RSZ2_RSZ2_RSZ2_RSZ1

#### Relational analysis RSZ result of RS_RSZ1_RSZ1_RSZ1_RSZ2_RSZ2_RSZ2_RSZ1
Status: Status.UNKNOWN
Output dim: 9, lower bound: -0.0050645, upper bound: 0.0050102
time: 0.77 seconds

### Relational analysis RSZ of RS_RSZ1_RSZ1_RSZ1_RSZ2_RSZ2_RSZ2_RSZ2

#### Relational analysis RSZ result of RS_RSZ1_RSZ1_RSZ1_RSZ2_RSZ2_RSZ2_RSZ2
Status: Status.UNKNOWN
Output dim: 9, lower bound: -0.0050176, upper bound: 0.0052068
time: 0.84 seconds

## BFS RS instance: RS_RSZ1_RSZ1_RSZ2_RSZ1_RSZ1_RSZ1

### Backsubstitution after applying RS history:
0: -0.0039095, -0.0003029, -0.0039095, -0.0003029, -0.0031789, 0.0031676
1: -0.0044907, 0.0040812, -0.0044907, 0.0040812, -0.0039043, 0.0039543
2: 0.0029569, 0.0099431, 0.0029569, 0.0099431, -0.0038706, 0.0039003
3: -0.0045519, -0.0034923, -0.0045519, -0.0034923, -0.0010595, 0.0010595
4: 0.0017018, 0.0078700, 0.0017018, 0.0078700, -0.0052935, 0.0052851
5: -0.0030341, 0.0031953, -0.0030341, 0.0031953, -0.0047419, 0.0047671
6: -0.0068516, -0.0033211, -0.0068516, -0.0033211, -0.0033509, 0.0033550
7: -0.0021570, 0.0038259, -0.0021570, 0.0038259, -0.0057616, 0.0057821
8: -0.0008533, 0.0000782, -0.0008533, 0.0000782, -0.0008389, 0.0008816
9: 0.9973557, 1.0121032, 0.9973557, 1.0121032, -0.0091783, 0.0091222

### Unstable ReLU Count (Linear/Conv2D Layers)
- layer_idx=0, type=LayerType.Linear, total=256, inp1_unstable=7, inp2_unstable=7, delta_unstable=250
- layer_idx=2, type=LayerType.Linear, total=256, inp1_unstable=10, inp2_unstable=10, delta_unstable=256
- layer_idx=4, type=LayerType.Linear, total=256, inp1_unstable=1, inp2_unstable=1, delta_unstable=256
- layer_idx=6, type=LayerType.Linear, total=256, inp1_unstable=0, inp2_unstable=0, delta_unstable=256
- layer_idx=8, type=LayerType.Linear, total=10, inp1_unstable=4, inp2_unstable=4, delta_unstable=10

Time for backsubstitution: 1.30 seconds

### RS candidates at layer 1
type: RSZ, layer: 1, pos: 219

Time for candidate selection: 0.13 seconds

### Candidate
type: RSZ, layer: 1, pos: 219

### Relational analysis RSZ of RS_RSZ1_RSZ1_RSZ2_RSZ1_RSZ1_RSZ1_RSZ1

#### Relational analysis RSZ result of RS_RSZ1_RSZ1_RSZ2_RSZ1_RSZ1_RSZ1_RSZ1
Status: Status.UNKNOWN
Output dim: 9, lower bound: -0.0053174, upper bound: 0.0048282
time: 0.78 seconds

### Relational analysis RSZ of RS_RSZ1_RSZ1_RSZ2_RSZ1_RSZ1_RSZ1_RSZ2

#### Relational analysis RSZ result of RS_RSZ1_RSZ1_RSZ2_RSZ1_RSZ1_RSZ1_RSZ2
Status: Status.UNKNOWN
Output dim: 9, lower bound: -0.0051236, upper bound: 0.0048890
time: 0.77 seconds

## BFS RS instance: RS_RSZ1_RSZ1_RSZ2_RSZ1_RSZ1_RSZ2

### Backsubstitution after applying RS history:
0: -0.0039095, -0.0003029, -0.0039095, -0.0003029, -0.0031786, 0.0031679
1: -0.0044907, 0.0040812, -0.0044907, 0.0040812, -0.0039043, 0.0039542
2: 0.0029569, 0.0099431, 0.0029569, 0.0099431, -0.0038708, 0.0039004
3: -0.0045519, -0.0034923, -0.0045519, -0.0034923, -0.0010595, 0.0010595
4: 0.0017018, 0.0078700, 0.0017018, 0.0078700, -0.0052933, 0.0052853
5: -0.0030341, 0.0031953, -0.0030341, 0.0031953, -0.0047425, 0.0047665
6: -0.0068516, -0.0033211, -0.0068516, -0.0033211, -0.0033509, 0.0033550
7: -0.0021570, 0.0038259, -0.0021570, 0.0038259, -0.0057619, 0.0057829
8: -0.0008533, 0.0000782, -0.0008533, 0.0000782, -0.0008395, 0.0008832
9: 0.9973557, 1.0121032, 0.9973557, 1.0121032, -0.0091779, 0.0091221

### Unstable ReLU Count (Linear/Conv2D Layers)
- layer_idx=0, type=LayerType.Linear, total=256, inp1_unstable=7, inp2_unstable=7, delta_unstable=250
- layer_idx=2, type=LayerType.Linear, total=256, inp1_unstable=10, inp2_unstable=10, delta_unstable=256
- layer_idx=4, type=LayerType.Linear, total=256, inp1_unstable=1, inp2_unstable=1, delta_unstable=256
- layer_idx=6, type=LayerType.Linear, total=256, inp1_unstable=0, inp2_unstable=0, delta_unstable=256
- layer_idx=8, type=LayerType.Linear, total=10, inp1_unstable=4, inp2_unstable=4, delta_unstable=10

Time for backsubstitution: 1.33 seconds

### RS candidates at layer 1
type: RSZ, layer: 1, pos: 219

Time for candidate selection: 0.13 seconds

### Candidate
type: RSZ, layer: 1, pos: 219

### Relational analysis RSZ of RS_RSZ1_RSZ1_RSZ2_RSZ1_RSZ1_RSZ2_RSZ1

#### Relational analysis RSZ result of RS_RSZ1_RSZ1_RSZ2_RSZ1_RSZ1_RSZ2_RSZ1
Status: Status.UNKNOWN
Output dim: 9, lower bound: -0.0053174, upper bound: 0.0048282
time: 0.81 seconds

### Relational analysis RSZ of RS_RSZ1_RSZ1_RSZ2_RSZ1_RSZ1_RSZ2_RSZ2

#### Relational analysis RSZ result of RS_RSZ1_RSZ1_RSZ2_RSZ1_RSZ1_RSZ2_RSZ2
Status: Status.UNKNOWN
Output dim: 9, lower bound: -0.0051239, upper bound: 0.0048890
time: 0.75 seconds

## BFS RS instance: RS_RSZ1_RSZ1_RSZ2_RSZ1_RSZ2_RSZ1

### Backsubstitution after applying RS history:
0: -0.0039095, -0.0003029, -0.0039095, -0.0003029, -0.0032349, 0.0031213
1: -0.0044907, 0.0040812, -0.0044907, 0.0040812, -0.0037851, 0.0040844
2: 0.0029569, 0.0099431, 0.0029569, 0.0099431, -0.0037835, 0.0039900
3: -0.0045519, -0.0034923, -0.0045519, -0.0034923, -0.0010595, 0.0010595
4: 0.0017018, 0.0078700, 0.0017018, 0.0078700, -0.0053356, 0.0052497
5: -0.0030341, 0.0031953, -0.0030341, 0.0031953, -0.0046899, 0.0048342
6: -0.0068516, -0.0033211, -0.0068516, -0.0033211, -0.0033407, 0.0033665
7: -0.0021570, 0.0038259, -0.0021570, 0.0038259, -0.0057921, 0.0057558
8: -0.0008533, 0.0000782, -0.0008533, 0.0000782, -0.0008327, 0.0008879
9: 0.9973557, 1.0121032, 0.9973557, 1.0121032, -0.0093541, 0.0089693

### Unstable ReLU Count (Linear/Conv2D Layers)
- layer_idx=0, type=LayerType.Linear, total=256, inp1_unstable=7, inp2_unstable=7, delta_unstable=250
- layer_idx=2, type=LayerType.Linear, total=256, inp1_unstable=10, inp2_unstable=10, delta_unstable=256
- layer_idx=4, type=LayerType.Linear, total=256, inp1_unstable=1, inp2_unstable=1, delta_unstable=256
- layer_idx=6, type=LayerType.Linear, total=256, inp1_unstable=0, inp2_unstable=0, delta_unstable=256
- layer_idx=8, type=LayerType.Linear, total=10, inp1_unstable=4, inp2_unstable=4, delta_unstable=10

Time for backsubstitution: 1.28 seconds

### RS candidates at layer 1
type: RSZ, layer: 1, pos: 219

Time for candidate selection: 0.13 seconds

### Candidate
type: RSZ, layer: 1, pos: 219

### Relational analysis RSZ of RS_RSZ1_RSZ1_RSZ2_RSZ1_RSZ2_RSZ1_RSZ1

#### Relational analysis RSZ result of RS_RSZ1_RSZ1_RSZ2_RSZ1_RSZ2_RSZ1_RSZ1
Status: Status.UNKNOWN
Output dim: 9, lower bound: -0.0054782, upper bound: 0.0047392
time: 0.78 seconds

### Relational analysis RSZ of RS_RSZ1_RSZ1_RSZ2_RSZ1_RSZ2_RSZ1_RSZ2

#### Relational analysis RSZ result of RS_RSZ1_RSZ1_RSZ2_RSZ1_RSZ2_RSZ1_RSZ2
Status: Status.UNKNOWN
Output dim: 9, lower bound: -0.0053094, upper bound: 0.0047969
time: 0.82 seconds

## BFS RS instance: RS_RSZ1_RSZ1_RSZ2_RSZ1_RSZ2_RSZ2

### Backsubstitution after applying RS history:
0: -0.0039095, -0.0003029, -0.0039095, -0.0003029, -0.0032347, 0.0031220
1: -0.0044907, 0.0040812, -0.0044907, 0.0040812, -0.0037853, 0.0040847
2: 0.0029569, 0.0099431, 0.0029569, 0.0099431, -0.0037836, 0.0039903
3: -0.0045519, -0.0034923, -0.0045519, -0.0034923, -0.0010595, 0.0010595
4: 0.0017018, 0.0078700, 0.0017018, 0.0078700, -0.0053355, 0.0052502
5: -0.0030341, 0.0031953, -0.0030341, 0.0031953, -0.0046918, 0.0048342
6: -0.0068516, -0.0033211, -0.0068516, -0.0033211, -0.0033407, 0.0033665
7: -0.0021570, 0.0038259, -0.0021570, 0.0038259, -0.0057932, 0.0057559
8: -0.0008533, 0.0000782, -0.0008533, 0.0000782, -0.0008329, 0.0008897
9: 0.9973557, 1.0121032, 0.9973557, 1.0121032, -0.0093542, 0.0089700

### Unstable ReLU Count (Linear/Conv2D Layers)
- layer_idx=0, type=LayerType.Linear, total=256, inp1_unstable=7, inp2_unstable=7, delta_unstable=250
- layer_idx=2, type=LayerType.Linear, total=256, inp1_unstable=10, inp2_unstable=10, delta_unstable=256
- layer_idx=4, type=LayerType.Linear, total=256, inp1_unstable=1, inp2_unstable=1, delta_unstable=256
- layer_idx=6, type=LayerType.Linear, total=256, inp1_unstable=0, inp2_unstable=0, delta_unstable=256
- layer_idx=8, type=LayerType.Linear, total=10, inp1_unstable=4, inp2_unstable=4, delta_unstable=10

Time for backsubstitution: 1.27 seconds

### RS candidates at layer 1
type: RSZ, layer: 1, pos: 219

Time for candidate selection: 0.14 seconds

### Candidate
type: RSZ, layer: 1, pos: 219

### Relational analysis RSZ of RS_RSZ1_RSZ1_RSZ2_RSZ1_RSZ2_RSZ2_RSZ1

#### Relational analysis RSZ result of RS_RSZ1_RSZ1_RSZ2_RSZ1_RSZ2_RSZ2_RSZ1
Status: Status.UNKNOWN
Output dim: 9, lower bound: -0.0054783, upper bound: 0.0047392
time: 0.78 seconds

### Relational analysis RSZ of RS_RSZ1_RSZ1_RSZ2_RSZ1_RSZ2_RSZ2_RSZ2

#### Relational analysis RSZ result of RS_RSZ1_RSZ1_RSZ2_RSZ1_RSZ2_RSZ2_RSZ2
Status: Status.UNKNOWN
Output dim: 9, lower bound: -0.0053095, upper bound: 0.0047969
time: 0.82 seconds

## BFS RS instance: RS_RSZ1_RSZ1_RSZ2_RSZ2_RSZ1_RSZ1

### Backsubstitution after applying RS history:
0: -0.0039095, -0.0003029, -0.0039095, -0.0003029, -0.0031080, 0.0032372
1: -0.0044907, 0.0040812, -0.0044907, 0.0040812, -0.0040637, 0.0037852
2: 0.0029569, 0.0099431, 0.0029569, 0.0099431, -0.0039730, 0.0037926
3: -0.0045519, -0.0034923, -0.0045519, -0.0034923, -0.0010595, 0.0010595
4: 0.0017018, 0.0078700, 0.0017018, 0.0078700, -0.0052389, 0.0053383
5: -0.0030341, 0.0031953, -0.0030341, 0.0031953, -0.0048184, 0.0046892
6: -0.0068516, -0.0033211, -0.0068516, -0.0033211, -0.0033652, 0.0033398
7: -0.0021570, 0.0038259, -0.0021570, 0.0038259, -0.0057367, 0.0058053
8: -0.0008533, 0.0000782, -0.0008533, 0.0000782, -0.0008477, 0.0008737
9: 0.9973557, 1.0121032, 0.9973557, 1.0121032, -0.0089466, 0.0093398

### Unstable ReLU Count (Linear/Conv2D Layers)
- layer_idx=0, type=LayerType.Linear, total=256, inp1_unstable=7, inp2_unstable=7, delta_unstable=250
- layer_idx=2, type=LayerType.Linear, total=256, inp1_unstable=10, inp2_unstable=10, delta_unstable=256
- layer_idx=4, type=LayerType.Linear, total=256, inp1_unstable=1, inp2_unstable=1, delta_unstable=256
- layer_idx=6, type=LayerType.Linear, total=256, inp1_unstable=0, inp2_unstable=0, delta_unstable=256
- layer_idx=8, type=LayerType.Linear, total=10, inp1_unstable=4, inp2_unstable=4, delta_unstable=10

Time for backsubstitution: 1.27 seconds

### RS candidates at layer 1
type: RSZ, layer: 1, pos: 219

Time for candidate selection: 0.14 seconds

### Candidate
type: RSZ, layer: 1, pos: 219

### Relational analysis RSZ of RS_RSZ1_RSZ1_RSZ2_RSZ2_RSZ1_RSZ1_RSZ1

#### Relational analysis RSZ result of RS_RSZ1_RSZ1_RSZ2_RSZ2_RSZ1_RSZ1_RSZ1
Status: Status.UNKNOWN
Output dim: 9, lower bound: -0.0049024, upper bound: 0.0051598
time: 0.85 seconds

### Relational analysis RSZ of RS_RSZ1_RSZ1_RSZ2_RSZ2_RSZ1_RSZ1_RSZ2

#### Relational analysis RSZ result of RS_RSZ1_RSZ1_RSZ2_RSZ2_RSZ1_RSZ1_RSZ2
Status: Status.UNKNOWN
Output dim: 9, lower bound: -0.0048358, upper bound: 0.0053493
time: 0.80 seconds

## BFS RS instance: RS_RSZ1_RSZ1_RSZ2_RSZ2_RSZ1_RSZ2

### Backsubstitution after applying RS history:
0: -0.0039095, -0.0003029, -0.0039095, -0.0003029, -0.0031072, 0.0032375
1: -0.0044907, 0.0040812, -0.0044907, 0.0040812, -0.0040634, 0.0037849
2: 0.0029569, 0.0099431, 0.0029569, 0.0099431, -0.0039728, 0.0037915
3: -0.0045519, -0.0034923, -0.0045519, -0.0034923, -0.0010595, 0.0010595
4: 0.0017018, 0.0078700, 0.0017018, 0.0078700, -0.0052384, 0.0053384
5: -0.0030341, 0.0031953, -0.0030341, 0.0031953, -0.0048192, 0.0046871
6: -0.0068516, -0.0033211, -0.0068516, -0.0033211, -0.0033652, 0.0033398
7: -0.0021570, 0.0038259, -0.0021570, 0.0038259, -0.0057373, 0.0058059
8: -0.0008533, 0.0000782, -0.0008533, 0.0000782, -0.0008482, 0.0008742
9: 0.9973557, 1.0121032, 0.9973557, 1.0121032, -0.0089458, 0.0093398

### Unstable ReLU Count (Linear/Conv2D Layers)
- layer_idx=0, type=LayerType.Linear, total=256, inp1_unstable=7, inp2_unstable=7, delta_unstable=250
- layer_idx=2, type=LayerType.Linear, total=256, inp1_unstable=10, inp2_unstable=10, delta_unstable=256
- layer_idx=4, type=LayerType.Linear, total=256, inp1_unstable=1, inp2_unstable=1, delta_unstable=256
- layer_idx=6, type=LayerType.Linear, total=256, inp1_unstable=0, inp2_unstable=0, delta_unstable=256
- layer_idx=8, type=LayerType.Linear, total=10, inp1_unstable=4, inp2_unstable=4, delta_unstable=10

Time for backsubstitution: 1.32 seconds

### RS candidates at layer 1
type: RSZ, layer: 1, pos: 219

Time for candidate selection: 0.13 seconds

### Candidate
type: RSZ, layer: 1, pos: 219

### Relational analysis RSZ of RS_RSZ1_RSZ1_RSZ2_RSZ2_RSZ1_RSZ2_RSZ1

#### Relational analysis RSZ result of RS_RSZ1_RSZ1_RSZ2_RSZ2_RSZ1_RSZ2_RSZ1
Status: Status.UNKNOWN
Output dim: 9, lower bound: -0.0049024, upper bound: 0.0051587
time: 0.82 seconds

### Relational analysis RSZ of RS_RSZ1_RSZ1_RSZ2_RSZ2_RSZ1_RSZ2_RSZ2

#### Relational analysis RSZ result of RS_RSZ1_RSZ1_RSZ2_RSZ2_RSZ1_RSZ2_RSZ2
Status: Status.UNKNOWN
Output dim: 9, lower bound: -0.0048358, upper bound: 0.0053493
time: 0.82 seconds

## BFS RS instance: RS_RSZ1_RSZ1_RSZ2_RSZ2_RSZ2_RSZ1

### Backsubstitution after applying RS history:
0: -0.0039095, -0.0003029, -0.0039095, -0.0003029, -0.0031654, 0.0031892
1: -0.0044907, 0.0040812, -0.0044907, 0.0040812, -0.0039491, 0.0039323
2: 0.0029569, 0.0099431, 0.0029569, 0.0099431, -0.0038954, 0.0038912
3: -0.0045519, -0.0034923, -0.0045519, -0.0034923, -0.0010595, 0.0010595
4: 0.0017018, 0.0078700, 0.0017018, 0.0078700, -0.0052828, 0.0053015
5: -0.0030341, 0.0031953, -0.0030341, 0.0031953, -0.0047653, 0.0047564
6: -0.0068516, -0.0033211, -0.0068516, -0.0033211, -0.0033553, 0.0033525
7: -0.0021570, 0.0038259, -0.0021570, 0.0038259, -0.0057690, 0.0057806
8: -0.0008533, 0.0000782, -0.0008533, 0.0000782, -0.0008407, 0.0008784
9: 0.9973557, 1.0121032, 0.9973557, 1.0121032, -0.0091361, 0.0091892

### Unstable ReLU Count (Linear/Conv2D Layers)
- layer_idx=0, type=LayerType.Linear, total=256, inp1_unstable=7, inp2_unstable=7, delta_unstable=250
- layer_idx=2, type=LayerType.Linear, total=256, inp1_unstable=10, inp2_unstable=10, delta_unstable=256
- layer_idx=4, type=LayerType.Linear, total=256, inp1_unstable=1, inp2_unstable=1, delta_unstable=256
- layer_idx=6, type=LayerType.Linear, total=256, inp1_unstable=0, inp2_unstable=0, delta_unstable=256
- layer_idx=8, type=LayerType.Linear, total=10, inp1_unstable=4, inp2_unstable=4, delta_unstable=10

Time for backsubstitution: 1.28 seconds

### RS candidates at layer 1
type: RSZ, layer: 1, pos: 219

Time for candidate selection: 0.13 seconds

### Candidate
type: RSZ, layer: 1, pos: 219

### Relational analysis RSZ of RS_RSZ1_RSZ1_RSZ2_RSZ2_RSZ2_RSZ1_RSZ1

#### Relational analysis RSZ result of RS_RSZ1_RSZ1_RSZ2_RSZ2_RSZ2_RSZ1_RSZ1
Status: Status.VERIFIED
Output dim: 9, lower bound: -0.0050255, upper bound: 0.0050265
time: 0.82 seconds

### Relational analysis RSZ of RS_RSZ1_RSZ1_RSZ2_RSZ2_RSZ2_RSZ1_RSZ2

#### Relational analysis RSZ result of RS_RSZ1_RSZ1_RSZ2_RSZ2_RSZ2_RSZ1_RSZ2
Status: Status.UNKNOWN
Output dim: 9, lower bound: -0.0049917, upper bound: 0.0052257
time: 0.78 seconds

## BFS RS instance: RS_RSZ1_RSZ1_RSZ2_RSZ2_RSZ2_RSZ2

### Backsubstitution after applying RS history:
0: -0.0039095, -0.0003029, -0.0039095, -0.0003029, -0.0031651, 0.0031894
1: -0.0044907, 0.0040812, -0.0044907, 0.0040812, -0.0039494, 0.0039322
2: 0.0029569, 0.0099431, 0.0029569, 0.0099431, -0.0038951, 0.0038910
3: -0.0045519, -0.0034923, -0.0045519, -0.0034923, -0.0010595, 0.0010595
4: 0.0017018, 0.0078700, 0.0017018, 0.0078700, -0.0052825, 0.0053018
5: -0.0030341, 0.0031953, -0.0030341, 0.0031953, -0.0047658, 0.0047555
6: -0.0068516, -0.0033211, -0.0068516, -0.0033211, -0.0033553, 0.0033525
7: -0.0021570, 0.0038259, -0.0021570, 0.0038259, -0.0057696, 0.0057822
8: -0.0008533, 0.0000782, -0.0008533, 0.0000782, -0.0008410, 0.0008792
9: 0.9973557, 1.0121032, 0.9973557, 1.0121032, -0.0091359, 0.0091896

### Unstable ReLU Count (Linear/Conv2D Layers)
- layer_idx=0, type=LayerType.Linear, total=256, inp1_unstable=7, inp2_unstable=7, delta_unstable=250
- layer_idx=2, type=LayerType.Linear, total=256, inp1_unstable=10, inp2_unstable=10, delta_unstable=256
- layer_idx=4, type=LayerType.Linear, total=256, inp1_unstable=1, inp2_unstable=1, delta_unstable=256
- layer_idx=6, type=LayerType.Linear, total=256, inp1_unstable=0, inp2_unstable=0, delta_unstable=256
- layer_idx=8, type=LayerType.Linear, total=10, inp1_unstable=4, inp2_unstable=4, delta_unstable=10

Time for backsubstitution: 1.27 seconds

### RS candidates at layer 1
type: RSZ, layer: 1, pos: 219

Time for candidate selection: 0.14 seconds

### Candidate
type: RSZ, layer: 1, pos: 219

### Relational analysis RSZ of RS_RSZ1_RSZ1_RSZ2_RSZ2_RSZ2_RSZ2_RSZ1

#### Relational analysis RSZ result of RS_RSZ1_RSZ1_RSZ2_RSZ2_RSZ2_RSZ2_RSZ1
Status: Status.VERIFIED
Output dim: 9, lower bound: -0.0050255, upper bound: 0.0050264
time: 0.85 seconds

### Relational analysis RSZ of RS_RSZ1_RSZ1_RSZ2_RSZ2_RSZ2_RSZ2_RSZ2

#### Relational analysis RSZ result of RS_RSZ1_RSZ1_RSZ2_RSZ2_RSZ2_RSZ2_RSZ2
Status: Status.UNKNOWN
Output dim: 9, lower bound: -0.0049917, upper bound: 0.0052257
time: 0.78 seconds

## BFS RS instance: RS_RSZ1_RSZ2_RSZ1_RSZ1_RSZ1_RSZ1

### Backsubstitution after applying RS history:
0: -0.0039095, -0.0003029, -0.0039095, -0.0003029, -0.0031910, 0.0031632
1: -0.0044907, 0.0040812, -0.0044907, 0.0040812, -0.0039218, 0.0039579
2: 0.0029569, 0.0099431, 0.0029569, 0.0099431, -0.0038827, 0.0039041
3: -0.0045519, -0.0034923, -0.0045519, -0.0034923, -0.0010595, 0.0010595
4: 0.0017018, 0.0078700, 0.0017018, 0.0078700, -0.0053030, 0.0052808
5: -0.0030341, 0.0031953, -0.0030341, 0.0031953, -0.0047549, 0.0047664
6: -0.0068516, -0.0033211, -0.0068516, -0.0033211, -0.0033517, 0.0033558
7: -0.0021570, 0.0038259, -0.0021570, 0.0038259, -0.0057889, 0.0057648
8: -0.0008533, 0.0000782, -0.0008533, 0.0000782, -0.0008795, 0.0008408
9: 0.9973557, 1.0121032, 0.9973557, 1.0121032, -0.0091980, 0.0091260

### Unstable ReLU Count (Linear/Conv2D Layers)
- layer_idx=0, type=LayerType.Linear, total=256, inp1_unstable=7, inp2_unstable=7, delta_unstable=250
- layer_idx=2, type=LayerType.Linear, total=256, inp1_unstable=10, inp2_unstable=10, delta_unstable=256
- layer_idx=4, type=LayerType.Linear, total=256, inp1_unstable=1, inp2_unstable=1, delta_unstable=256
- layer_idx=6, type=LayerType.Linear, total=256, inp1_unstable=0, inp2_unstable=0, delta_unstable=256
- layer_idx=8, type=LayerType.Linear, total=10, inp1_unstable=4, inp2_unstable=4, delta_unstable=10

Time for backsubstitution: 1.27 seconds

### RS candidates at layer 1
type: RSZ, layer: 1, pos: 219

Time for candidate selection: 0.14 seconds

### Candidate
type: RSZ, layer: 1, pos: 219

### Relational analysis RSZ of RS_RSZ1_RSZ2_RSZ1_RSZ1_RSZ1_RSZ1_RSZ1

#### Relational analysis RSZ result of RS_RSZ1_RSZ2_RSZ1_RSZ1_RSZ1_RSZ1_RSZ1
Status: Status.UNKNOWN
Output dim: 9, lower bound: -0.0052773, upper bound: 0.0049534
time: 0.77 seconds

### Relational analysis RSZ of RS_RSZ1_RSZ2_RSZ1_RSZ1_RSZ1_RSZ1_RSZ2

#### Relational analysis RSZ result of RS_RSZ1_RSZ2_RSZ1_RSZ1_RSZ1_RSZ1_RSZ2
Status: Status.VERIFIED
Output dim: 9, lower bound: -0.0050522, upper bound: 0.0049978
time: 0.86 seconds

## BFS RS instance: RS_RSZ1_RSZ2_RSZ1_RSZ1_RSZ1_RSZ2

### Backsubstitution after applying RS history:
0: -0.0039095, -0.0003029, -0.0039095, -0.0003029, -0.0031906, 0.0031636
1: -0.0044907, 0.0040812, -0.0044907, 0.0040812, -0.0039219, 0.0039577
2: 0.0029569, 0.0099431, 0.0029569, 0.0099431, -0.0038828, 0.0039044
3: -0.0045519, -0.0034923, -0.0045519, -0.0034923, -0.0010595, 0.0010595
4: 0.0017018, 0.0078700, 0.0017018, 0.0078700, -0.0053027, 0.0052810
5: -0.0030341, 0.0031953, -0.0030341, 0.0031953, -0.0047559, 0.0047659
6: -0.0068516, -0.0033211, -0.0068516, -0.0033211, -0.0033517, 0.0033558
7: -0.0021570, 0.0038259, -0.0021570, 0.0038259, -0.0057875, 0.0057655
8: -0.0008533, 0.0000782, -0.0008533, 0.0000782, -0.0008790, 0.0008412
9: 0.9973557, 1.0121032, 0.9973557, 1.0121032, -0.0091977, 0.0091260

### Unstable ReLU Count (Linear/Conv2D Layers)
- layer_idx=0, type=LayerType.Linear, total=256, inp1_unstable=7, inp2_unstable=7, delta_unstable=250
- layer_idx=2, type=LayerType.Linear, total=256, inp1_unstable=10, inp2_unstable=10, delta_unstable=256
- layer_idx=4, type=LayerType.Linear, total=256, inp1_unstable=1, inp2_unstable=1, delta_unstable=256
- layer_idx=6, type=LayerType.Linear, total=256, inp1_unstable=0, inp2_unstable=0, delta_unstable=256
- layer_idx=8, type=LayerType.Linear, total=10, inp1_unstable=4, inp2_unstable=4, delta_unstable=10

Time for backsubstitution: 1.26 seconds

### RS candidates at layer 1
type: RSZ, layer: 1, pos: 219

Time for candidate selection: 0.13 seconds

### Candidate
type: RSZ, layer: 1, pos: 219

### Relational analysis RSZ of RS_RSZ1_RSZ2_RSZ1_RSZ1_RSZ1_RSZ2_RSZ1

#### Relational analysis RSZ result of RS_RSZ1_RSZ2_RSZ1_RSZ1_RSZ1_RSZ2_RSZ1
Status: Status.UNKNOWN
Output dim: 9, lower bound: -0.0052773, upper bound: 0.0049534
time: 0.76 seconds

### Relational analysis RSZ of RS_RSZ1_RSZ2_RSZ1_RSZ1_RSZ1_RSZ2_RSZ2

#### Relational analysis RSZ result of RS_RSZ1_RSZ2_RSZ1_RSZ1_RSZ1_RSZ2_RSZ2
Status: Status.VERIFIED
Output dim: 9, lower bound: -0.0050529, upper bound: 0.0049978
time: 0.87 seconds

## BFS RS instance: RS_RSZ1_RSZ2_RSZ1_RSZ1_RSZ2_RSZ1

### Backsubstitution after applying RS history:
0: -0.0039095, -0.0003029, -0.0039095, -0.0003029, -0.0032384, 0.0031058
1: -0.0044907, 0.0040812, -0.0044907, 0.0040812, -0.0037770, 0.0040691
2: 0.0029569, 0.0099431, 0.0029569, 0.0099431, -0.0037846, 0.0039787
3: -0.0045519, -0.0034923, -0.0045519, -0.0034923, -0.0010595, 0.0010595
4: 0.0017018, 0.0078700, 0.0017018, 0.0078700, -0.0053393, 0.0052371
5: -0.0030341, 0.0031953, -0.0030341, 0.0031953, -0.0046862, 0.0048197
6: -0.0068516, -0.0033211, -0.0068516, -0.0033211, -0.0033392, 0.0033655
7: -0.0021570, 0.0038259, -0.0021570, 0.0038259, -0.0058112, 0.0057298
8: -0.0008533, 0.0000782, -0.0008533, 0.0000782, -0.0008748, 0.0008484
9: 0.9973557, 1.0121032, 0.9973557, 1.0121032, -0.0093450, 0.0089368

### Unstable ReLU Count (Linear/Conv2D Layers)
- layer_idx=0, type=LayerType.Linear, total=256, inp1_unstable=7, inp2_unstable=7, delta_unstable=250
- layer_idx=2, type=LayerType.Linear, total=256, inp1_unstable=10, inp2_unstable=10, delta_unstable=256
- layer_idx=4, type=LayerType.Linear, total=256, inp1_unstable=1, inp2_unstable=1, delta_unstable=256
- layer_idx=6, type=LayerType.Linear, total=256, inp1_unstable=0, inp2_unstable=0, delta_unstable=256
- layer_idx=8, type=LayerType.Linear, total=10, inp1_unstable=4, inp2_unstable=4, delta_unstable=10

Time for backsubstitution: 1.33 seconds

### RS candidates at layer 1
type: RSZ, layer: 1, pos: 219

Time for candidate selection: 0.13 seconds

### Candidate
type: RSZ, layer: 1, pos: 219

### Relational analysis RSZ of RS_RSZ1_RSZ2_RSZ1_RSZ1_RSZ2_RSZ1_RSZ1

#### Relational analysis RSZ result of RS_RSZ1_RSZ2_RSZ1_RSZ1_RSZ2_RSZ1_RSZ1
Status: Status.UNKNOWN
Output dim: 9, lower bound: -0.0053802, upper bound: 0.0047963
time: 0.75 seconds

### Relational analysis RSZ of RS_RSZ1_RSZ2_RSZ1_RSZ1_RSZ2_RSZ1_RSZ2

#### Relational analysis RSZ result of RS_RSZ1_RSZ2_RSZ1_RSZ1_RSZ2_RSZ1_RSZ2
Status: Status.UNKNOWN
Output dim: 9, lower bound: -0.0051935, upper bound: 0.0048579
time: 0.86 seconds

## BFS RS instance: RS_RSZ1_RSZ2_RSZ1_RSZ1_RSZ2_RSZ2

### Backsubstitution after applying RS history:
0: -0.0039095, -0.0003029, -0.0039095, -0.0003029, -0.0032384, 0.0031067
1: -0.0044907, 0.0040812, -0.0044907, 0.0040812, -0.0037774, 0.0040693
2: 0.0029569, 0.0099431, 0.0029569, 0.0099431, -0.0037856, 0.0039787
3: -0.0045519, -0.0034923, -0.0045519, -0.0034923, -0.0010595, 0.0010595
4: 0.0017018, 0.0078700, 0.0017018, 0.0078700, -0.0053393, 0.0052376
5: -0.0030341, 0.0031953, -0.0030341, 0.0031953, -0.0046883, 0.0048190
6: -0.0068516, -0.0033211, -0.0068516, -0.0033211, -0.0033392, 0.0033656
7: -0.0021570, 0.0038259, -0.0021570, 0.0038259, -0.0058109, 0.0057301
8: -0.0008533, 0.0000782, -0.0008533, 0.0000782, -0.0008744, 0.0008483
9: 0.9973557, 1.0121032, 0.9973557, 1.0121032, -0.0093452, 0.0089376

### Unstable ReLU Count (Linear/Conv2D Layers)
- layer_idx=0, type=LayerType.Linear, total=256, inp1_unstable=7, inp2_unstable=7, delta_unstable=250
- layer_idx=2, type=LayerType.Linear, total=256, inp1_unstable=10, inp2_unstable=10, delta_unstable=256
- layer_idx=4, type=LayerType.Linear, total=256, inp1_unstable=1, inp2_unstable=1, delta_unstable=256
- layer_idx=6, type=LayerType.Linear, total=256, inp1_unstable=0, inp2_unstable=0, delta_unstable=256
- layer_idx=8, type=LayerType.Linear, total=10, inp1_unstable=4, inp2_unstable=4, delta_unstable=10

Time for backsubstitution: 1.26 seconds

### RS candidates at layer 1
type: RSZ, layer: 1, pos: 219

Time for candidate selection: 0.14 seconds

### Candidate
type: RSZ, layer: 1, pos: 219

### Relational analysis RSZ of RS_RSZ1_RSZ2_RSZ1_RSZ1_RSZ2_RSZ2_RSZ1

#### Relational analysis RSZ result of RS_RSZ1_RSZ2_RSZ1_RSZ1_RSZ2_RSZ2_RSZ1
Status: Status.UNKNOWN
Output dim: 9, lower bound: -0.0053802, upper bound: 0.0047963
time: 0.75 seconds

### Relational analysis RSZ of RS_RSZ1_RSZ2_RSZ1_RSZ1_RSZ2_RSZ2_RSZ2

#### Relational analysis RSZ result of RS_RSZ1_RSZ2_RSZ1_RSZ1_RSZ2_RSZ2_RSZ2
Status: Status.UNKNOWN
Output dim: 9, lower bound: -0.0051947, upper bound: 0.0048588
time: 0.85 seconds

## BFS RS instance: RS_RSZ1_RSZ2_RSZ1_RSZ2_RSZ1_RSZ1

### Backsubstitution after applying RS history:
0: -0.0039095, -0.0003029, -0.0039095, -0.0003029, -0.0031235, 0.0032335
1: -0.0044907, 0.0040812, -0.0044907, 0.0040812, -0.0040783, 0.0037936
2: 0.0029569, 0.0099431, 0.0029569, 0.0099431, -0.0039858, 0.0037919
3: -0.0045519, -0.0034923, -0.0045519, -0.0034923, -0.0010595, 0.0010595
4: 0.0017018, 0.0078700, 0.0017018, 0.0078700, -0.0052514, 0.0053345
5: -0.0030341, 0.0031953, -0.0030341, 0.0031953, -0.0048337, 0.0046925
6: -0.0068516, -0.0033211, -0.0068516, -0.0033211, -0.0033660, 0.0033414
7: -0.0021570, 0.0038259, -0.0021570, 0.0038259, -0.0057635, 0.0057879
8: -0.0008533, 0.0000782, -0.0008533, 0.0000782, -0.0008900, 0.0008328
9: 0.9973557, 1.0121032, 0.9973557, 1.0121032, -0.0089784, 0.0093475

### Unstable ReLU Count (Linear/Conv2D Layers)
- layer_idx=0, type=LayerType.Linear, total=256, inp1_unstable=7, inp2_unstable=7, delta_unstable=250
- layer_idx=2, type=LayerType.Linear, total=256, inp1_unstable=10, inp2_unstable=10, delta_unstable=256
- layer_idx=4, type=LayerType.Linear, total=256, inp1_unstable=1, inp2_unstable=1, delta_unstable=256
- layer_idx=6, type=LayerType.Linear, total=256, inp1_unstable=0, inp2_unstable=0, delta_unstable=256
- layer_idx=8, type=LayerType.Linear, total=10, inp1_unstable=4, inp2_unstable=4, delta_unstable=10

Time for backsubstitution: 1.28 seconds

### RS candidates at layer 1
type: RSZ, layer: 1, pos: 219

Time for candidate selection: 0.13 seconds

### Candidate
type: RSZ, layer: 1, pos: 219

### Relational analysis RSZ of RS_RSZ1_RSZ2_RSZ1_RSZ2_RSZ1_RSZ1_RSZ1

#### Relational analysis RSZ result of RS_RSZ1_RSZ2_RSZ1_RSZ2_RSZ1_RSZ1_RSZ1
Status: Status.UNKNOWN
Output dim: 9, lower bound: -0.0048628, upper bound: 0.0052874
time: 0.80 seconds

### Relational analysis RSZ of RS_RSZ1_RSZ2_RSZ1_RSZ2_RSZ1_RSZ1_RSZ2

#### Relational analysis RSZ result of RS_RSZ1_RSZ2_RSZ1_RSZ2_RSZ1_RSZ1_RSZ2
Status: Status.UNKNOWN
Output dim: 9, lower bound: -0.0047878, upper bound: 0.0054625
time: 0.87 seconds

## BFS RS instance: RS_RSZ1_RSZ2_RSZ1_RSZ2_RSZ1_RSZ2

### Backsubstitution after applying RS history:
0: -0.0039095, -0.0003029, -0.0039095, -0.0003029, -0.0031230, 0.0032338
1: -0.0044907, 0.0040812, -0.0044907, 0.0040812, -0.0040782, 0.0037934
2: 0.0029569, 0.0099431, 0.0029569, 0.0099431, -0.0039853, 0.0037926
3: -0.0045519, -0.0034923, -0.0045519, -0.0034923, -0.0010595, 0.0010595
4: 0.0017018, 0.0078700, 0.0017018, 0.0078700, -0.0052509, 0.0053346
5: -0.0030341, 0.0031953, -0.0030341, 0.0031953, -0.0048337, 0.0046906
6: -0.0068516, -0.0033211, -0.0068516, -0.0033211, -0.0033660, 0.0033413
7: -0.0021570, 0.0038259, -0.0021570, 0.0038259, -0.0057631, 0.0057874
8: -0.0008533, 0.0000782, -0.0008533, 0.0000782, -0.0008885, 0.0008328
9: 0.9973557, 1.0121032, 0.9973557, 1.0121032, -0.0089777, 0.0093477

### Unstable ReLU Count (Linear/Conv2D Layers)
- layer_idx=0, type=LayerType.Linear, total=256, inp1_unstable=7, inp2_unstable=7, delta_unstable=250
- layer_idx=2, type=LayerType.Linear, total=256, inp1_unstable=10, inp2_unstable=10, delta_unstable=256
- layer_idx=4, type=LayerType.Linear, total=256, inp1_unstable=1, inp2_unstable=1, delta_unstable=256
- layer_idx=6, type=LayerType.Linear, total=256, inp1_unstable=0, inp2_unstable=0, delta_unstable=256
- layer_idx=8, type=LayerType.Linear, total=10, inp1_unstable=4, inp2_unstable=4, delta_unstable=10

Time for backsubstitution: 1.27 seconds

### RS candidates at layer 1
type: RSZ, layer: 1, pos: 219

Time for candidate selection: 0.13 seconds

### Candidate
type: RSZ, layer: 1, pos: 219

### Relational analysis RSZ of RS_RSZ1_RSZ2_RSZ1_RSZ2_RSZ1_RSZ2_RSZ1

#### Relational analysis RSZ result of RS_RSZ1_RSZ2_RSZ1_RSZ2_RSZ1_RSZ2_RSZ1
Status: Status.UNKNOWN
Output dim: 9, lower bound: -0.0048628, upper bound: 0.0052874
time: 0.79 seconds

### Relational analysis RSZ of RS_RSZ1_RSZ2_RSZ1_RSZ2_RSZ1_RSZ2_RSZ2

#### Relational analysis RSZ result of RS_RSZ1_RSZ2_RSZ1_RSZ2_RSZ1_RSZ2_RSZ2
Status: Status.UNKNOWN
Output dim: 9, lower bound: -0.0047878, upper bound: 0.0054625
time: 0.87 seconds

## BFS RS instance: RS_RSZ1_RSZ2_RSZ1_RSZ2_RSZ2_RSZ1

### Backsubstitution after applying RS history:
0: -0.0039095, -0.0003029, -0.0039095, -0.0003029, -0.0031700, 0.0031773
1: -0.0044907, 0.0040812, -0.0044907, 0.0040812, -0.0039468, 0.0039161
2: 0.0029569, 0.0099431, 0.0029569, 0.0099431, -0.0038943, 0.0038781
3: -0.0045519, -0.0034923, -0.0045519, -0.0034923, -0.0010595, 0.0010595
4: 0.0017018, 0.0078700, 0.0017018, 0.0078700, -0.0052873, 0.0052920
5: -0.0030341, 0.0031953, -0.0030341, 0.0031953, -0.0047659, 0.0047434
6: -0.0068516, -0.0033211, -0.0068516, -0.0033211, -0.0033544, 0.0033518
7: -0.0021570, 0.0038259, -0.0021570, 0.0038259, -0.0057876, 0.0057544
8: -0.0008533, 0.0000782, -0.0008533, 0.0000782, -0.0008837, 0.0008397
9: 0.9973557, 1.0121032, 0.9973557, 1.0121032, -0.0091342, 0.0091702

### Unstable ReLU Count (Linear/Conv2D Layers)
- layer_idx=0, type=LayerType.Linear, total=256, inp1_unstable=7, inp2_unstable=7, delta_unstable=250
- layer_idx=2, type=LayerType.Linear, total=256, inp1_unstable=10, inp2_unstable=10, delta_unstable=256
- layer_idx=4, type=LayerType.Linear, total=256, inp1_unstable=1, inp2_unstable=1, delta_unstable=256
- layer_idx=6, type=LayerType.Linear, total=256, inp1_unstable=0, inp2_unstable=0, delta_unstable=256
- layer_idx=8, type=LayerType.Linear, total=10, inp1_unstable=4, inp2_unstable=4, delta_unstable=10

Time for backsubstitution: 1.28 seconds

### RS candidates at layer 1
type: RSZ, layer: 1, pos: 219

Time for candidate selection: 0.13 seconds

### Candidate
type: RSZ, layer: 1, pos: 219

### Relational analysis RSZ of RS_RSZ1_RSZ2_RSZ1_RSZ2_RSZ2_RSZ1_RSZ1

#### Relational analysis RSZ result of RS_RSZ1_RSZ2_RSZ1_RSZ2_RSZ2_RSZ1_RSZ1
Status: Status.UNKNOWN
Output dim: 9, lower bound: -0.0049568, upper bound: 0.0050997
time: 0.76 seconds

### Relational analysis RSZ of RS_RSZ1_RSZ2_RSZ1_RSZ2_RSZ2_RSZ1_RSZ2

#### Relational analysis RSZ result of RS_RSZ1_RSZ2_RSZ1_RSZ2_RSZ2_RSZ1_RSZ2
Status: Status.UNKNOWN
Output dim: 9, lower bound: -0.0048928, upper bound: 0.0052867
time: 0.89 seconds

## BFS RS instance: RS_RSZ1_RSZ2_RSZ1_RSZ2_RSZ2_RSZ2

### Backsubstitution after applying RS history:
0: -0.0039095, -0.0003029, -0.0039095, -0.0003029, -0.0031697, 0.0031776
1: -0.0044907, 0.0040812, -0.0044907, 0.0040812, -0.0039470, 0.0039161
2: 0.0029569, 0.0099431, 0.0029569, 0.0099431, -0.0038948, 0.0038784
3: -0.0045519, -0.0034923, -0.0045519, -0.0034923, -0.0010595, 0.0010595
4: 0.0017018, 0.0078700, 0.0017018, 0.0078700, -0.0052869, 0.0052922
5: -0.0030341, 0.0031953, -0.0030341, 0.0031953, -0.0047666, 0.0047427
6: -0.0068516, -0.0033211, -0.0068516, -0.0033211, -0.0033544, 0.0033518
7: -0.0021570, 0.0038259, -0.0021570, 0.0038259, -0.0057870, 0.0057551
8: -0.0008533, 0.0000782, -0.0008533, 0.0000782, -0.0008824, 0.0008392
9: 0.9973557, 1.0121032, 0.9973557, 1.0121032, -0.0091341, 0.0091706

### Unstable ReLU Count (Linear/Conv2D Layers)
- layer_idx=0, type=LayerType.Linear, total=256, inp1_unstable=7, inp2_unstable=7, delta_unstable=250
- layer_idx=2, type=LayerType.Linear, total=256, inp1_unstable=10, inp2_unstable=10, delta_unstable=256
- layer_idx=4, type=LayerType.Linear, total=256, inp1_unstable=1, inp2_unstable=1, delta_unstable=256
- layer_idx=6, type=LayerType.Linear, total=256, inp1_unstable=0, inp2_unstable=0, delta_unstable=256
- layer_idx=8, type=LayerType.Linear, total=10, inp1_unstable=4, inp2_unstable=4, delta_unstable=10

Time for backsubstitution: 1.31 seconds

### RS candidates at layer 1
type: RSZ, layer: 1, pos: 219

Time for candidate selection: 0.13 seconds

### Candidate
type: RSZ, layer: 1, pos: 219

### Relational analysis RSZ of RS_RSZ1_RSZ2_RSZ1_RSZ2_RSZ2_RSZ2_RSZ1

#### Relational analysis RSZ result of RS_RSZ1_RSZ2_RSZ1_RSZ2_RSZ2_RSZ2_RSZ1
Status: Status.UNKNOWN
Output dim: 9, lower bound: -0.0049568, upper bound: 0.0050997
time: 0.78 seconds

### Relational analysis RSZ of RS_RSZ1_RSZ2_RSZ1_RSZ2_RSZ2_RSZ2_RSZ2

#### Relational analysis RSZ result of RS_RSZ1_RSZ2_RSZ1_RSZ2_RSZ2_RSZ2_RSZ2
Status: Status.UNKNOWN
Output dim: 9, lower bound: -0.0048928, upper bound: 0.0052867
time: 0.91 seconds

## BFS RS instance: RS_RSZ1_RSZ2_RSZ2_RSZ1_RSZ1_RSZ1

### Backsubstitution after applying RS history:
0: -0.0039095, -0.0003029, -0.0039095, -0.0003029, -0.0031778, 0.0031693
1: -0.0044907, 0.0040812, -0.0044907, 0.0040812, -0.0039346, 0.0039359
2: 0.0029569, 0.0099431, 0.0029569, 0.0099431, -0.0038910, 0.0038901
3: -0.0045519, -0.0034923, -0.0045519, -0.0034923, -0.0010595, 0.0010595
4: 0.0017018, 0.0078700, 0.0017018, 0.0078700, -0.0052933, 0.0052854
5: -0.0030341, 0.0031953, -0.0030341, 0.0031953, -0.0047609, 0.0047523
6: -0.0068516, -0.0033211, -0.0068516, -0.0033211, -0.0033530, 0.0033536
7: -0.0021570, 0.0038259, -0.0021570, 0.0038259, -0.0057826, 0.0057700
8: -0.0008533, 0.0000782, -0.0008533, 0.0000782, -0.0008684, 0.0008504
9: 0.9973557, 1.0121032, 0.9973557, 1.0121032, -0.0091620, 0.0091461

### Unstable ReLU Count (Linear/Conv2D Layers)
- layer_idx=0, type=LayerType.Linear, total=256, inp1_unstable=7, inp2_unstable=7, delta_unstable=250
- layer_idx=2, type=LayerType.Linear, total=256, inp1_unstable=10, inp2_unstable=10, delta_unstable=256
- layer_idx=4, type=LayerType.Linear, total=256, inp1_unstable=1, inp2_unstable=1, delta_unstable=256
- layer_idx=6, type=LayerType.Linear, total=256, inp1_unstable=0, inp2_unstable=0, delta_unstable=256
- layer_idx=8, type=LayerType.Linear, total=10, inp1_unstable=4, inp2_unstable=4, delta_unstable=10

Time for backsubstitution: 1.39 seconds

### RS candidates at layer 1
type: RSZ, layer: 1, pos: 219

Time for candidate selection: 0.14 seconds

### Candidate
type: RSZ, layer: 1, pos: 219

### Relational analysis RSZ of RS_RSZ1_RSZ2_RSZ2_RSZ1_RSZ1_RSZ1_RSZ1

#### Relational analysis RSZ result of RS_RSZ1_RSZ2_RSZ2_RSZ1_RSZ1_RSZ1_RSZ1
Status: Status.UNKNOWN
Output dim: 9, lower bound: -0.0052573, upper bound: 0.0049775
time: 0.78 seconds

### Relational analysis RSZ of RS_RSZ1_RSZ2_RSZ2_RSZ1_RSZ1_RSZ1_RSZ2

#### Relational analysis RSZ result of RS_RSZ1_RSZ2_RSZ2_RSZ1_RSZ1_RSZ1_RSZ2
Status: Status.VERIFIED
Output dim: 9, lower bound: -0.0050422, upper bound: 0.0050256
time: 0.87 seconds

## BFS RS instance: RS_RSZ1_RSZ2_RSZ2_RSZ1_RSZ1_RSZ2

### Backsubstitution after applying RS history:
0: -0.0039095, -0.0003029, -0.0039095, -0.0003029, -0.0031774, 0.0031695
1: -0.0044907, 0.0040812, -0.0044907, 0.0040812, -0.0039347, 0.0039358
2: 0.0029569, 0.0099431, 0.0029569, 0.0099431, -0.0038913, 0.0038903
3: -0.0045519, -0.0034923, -0.0045519, -0.0034923, -0.0010595, 0.0010595
4: 0.0017018, 0.0078700, 0.0017018, 0.0078700, -0.0052930, 0.0052856
5: -0.0030341, 0.0031953, -0.0030341, 0.0031953, -0.0047616, 0.0047513
6: -0.0068516, -0.0033211, -0.0068516, -0.0033211, -0.0033530, 0.0033536
7: -0.0021570, 0.0038259, -0.0021570, 0.0038259, -0.0057811, 0.0057705
8: -0.0008533, 0.0000782, -0.0008533, 0.0000782, -0.0008666, 0.0008511
9: 0.9973557, 1.0121032, 0.9973557, 1.0121032, -0.0091617, 0.0091461

### Unstable ReLU Count (Linear/Conv2D Layers)
- layer_idx=0, type=LayerType.Linear, total=256, inp1_unstable=7, inp2_unstable=7, delta_unstable=250
- layer_idx=2, type=LayerType.Linear, total=256, inp1_unstable=10, inp2_unstable=10, delta_unstable=256
- layer_idx=4, type=LayerType.Linear, total=256, inp1_unstable=1, inp2_unstable=1, delta_unstable=256
- layer_idx=6, type=LayerType.Linear, total=256, inp1_unstable=0, inp2_unstable=0, delta_unstable=256
- layer_idx=8, type=LayerType.Linear, total=10, inp1_unstable=4, inp2_unstable=4, delta_unstable=10

Time for backsubstitution: 1.31 seconds

### RS candidates at layer 1
type: RSZ, layer: 1, pos: 219

Time for candidate selection: 0.14 seconds

### Candidate
type: RSZ, layer: 1, pos: 219

### Relational analysis RSZ of RS_RSZ1_RSZ2_RSZ2_RSZ1_RSZ1_RSZ2_RSZ1

#### Relational analysis RSZ result of RS_RSZ1_RSZ2_RSZ2_RSZ1_RSZ1_RSZ2_RSZ1
Status: Status.UNKNOWN
Output dim: 9, lower bound: -0.0052573, upper bound: 0.0049775
time: 0.80 seconds

### Relational analysis RSZ of RS_RSZ1_RSZ2_RSZ2_RSZ1_RSZ1_RSZ2_RSZ2

#### Relational analysis RSZ result of RS_RSZ1_RSZ2_RSZ2_RSZ1_RSZ1_RSZ2_RSZ2
Status: Status.VERIFIED
Output dim: 9, lower bound: -0.0050426, upper bound: 0.0050256
time: 0.88 seconds

## BFS RS instance: RS_RSZ1_RSZ2_RSZ2_RSZ1_RSZ2_RSZ1

### Backsubstitution after applying RS history:
0: -0.0039095, -0.0003029, -0.0039095, -0.0003029, -0.0032331, 0.0031227
1: -0.0044907, 0.0040812, -0.0044907, 0.0040812, -0.0038112, 0.0040596
2: 0.0029569, 0.0099431, 0.0029569, 0.0099431, -0.0038036, 0.0039720
3: -0.0045519, -0.0034923, -0.0045519, -0.0034923, -0.0010595, 0.0010595
4: 0.0017018, 0.0078700, 0.0017018, 0.0078700, -0.0053352, 0.0052500
5: -0.0030341, 0.0031953, -0.0030341, 0.0031953, -0.0047023, 0.0048151
6: -0.0068516, -0.0033211, -0.0068516, -0.0033211, -0.0033425, 0.0033646
7: -0.0021570, 0.0038259, -0.0021570, 0.0038259, -0.0058083, 0.0057371
8: -0.0008533, 0.0000782, -0.0008533, 0.0000782, -0.0008641, 0.0008584
9: 0.9973557, 1.0121032, 0.9973557, 1.0121032, -0.0093313, 0.0089892

### Unstable ReLU Count (Linear/Conv2D Layers)
- layer_idx=0, type=LayerType.Linear, total=256, inp1_unstable=7, inp2_unstable=7, delta_unstable=250
- layer_idx=2, type=LayerType.Linear, total=256, inp1_unstable=10, inp2_unstable=10, delta_unstable=256
- layer_idx=4, type=LayerType.Linear, total=256, inp1_unstable=1, inp2_unstable=1, delta_unstable=256
- layer_idx=6, type=LayerType.Linear, total=256, inp1_unstable=0, inp2_unstable=0, delta_unstable=256
- layer_idx=8, type=LayerType.Linear, total=10, inp1_unstable=4, inp2_unstable=4, delta_unstable=10

Time for backsubstitution: 1.33 seconds

### RS candidates at layer 1
type: RSZ, layer: 1, pos: 219

Time for candidate selection: 0.15 seconds

### Candidate
type: RSZ, layer: 1, pos: 219

### Relational analysis RSZ of RS_RSZ1_RSZ2_RSZ2_RSZ1_RSZ2_RSZ1_RSZ1

#### Relational analysis RSZ result of RS_RSZ1_RSZ2_RSZ2_RSZ1_RSZ2_RSZ1_RSZ1
Status: Status.UNKNOWN
Output dim: 9, lower bound: -0.0053776, upper bound: 0.0048599
time: 0.80 seconds

### Relational analysis RSZ of RS_RSZ1_RSZ2_RSZ2_RSZ1_RSZ2_RSZ1_RSZ2

#### Relational analysis RSZ result of RS_RSZ1_RSZ2_RSZ2_RSZ1_RSZ2_RSZ1_RSZ2
Status: Status.UNKNOWN
Output dim: 9, lower bound: -0.0051929, upper bound: 0.0049236
time: 0.83 seconds

## BFS RS instance: RS_RSZ1_RSZ2_RSZ2_RSZ1_RSZ2_RSZ2

### Backsubstitution after applying RS history:
0: -0.0039095, -0.0003029, -0.0039095, -0.0003029, -0.0032329, 0.0031233
1: -0.0044907, 0.0040812, -0.0044907, 0.0040812, -0.0038115, 0.0040597
2: 0.0029569, 0.0099431, 0.0029569, 0.0099431, -0.0038047, 0.0039721
3: -0.0045519, -0.0034923, -0.0045519, -0.0034923, -0.0010595, 0.0010595
4: 0.0017018, 0.0078700, 0.0017018, 0.0078700, -0.0053351, 0.0052504
5: -0.0030341, 0.0031953, -0.0030341, 0.0031953, -0.0047039, 0.0048146
6: -0.0068516, -0.0033211, -0.0068516, -0.0033211, -0.0033425, 0.0033646
7: -0.0021570, 0.0038259, -0.0021570, 0.0038259, -0.0058075, 0.0057374
8: -0.0008533, 0.0000782, -0.0008533, 0.0000782, -0.0008633, 0.0008585
9: 0.9973557, 1.0121032, 0.9973557, 1.0121032, -0.0093313, 0.0089899

### Unstable ReLU Count (Linear/Conv2D Layers)
- layer_idx=0, type=LayerType.Linear, total=256, inp1_unstable=7, inp2_unstable=7, delta_unstable=250
- layer_idx=2, type=LayerType.Linear, total=256, inp1_unstable=10, inp2_unstable=10, delta_unstable=256
- layer_idx=4, type=LayerType.Linear, total=256, inp1_unstable=1, inp2_unstable=1, delta_unstable=256
- layer_idx=6, type=LayerType.Linear, total=256, inp1_unstable=0, inp2_unstable=0, delta_unstable=256
- layer_idx=8, type=LayerType.Linear, total=10, inp1_unstable=4, inp2_unstable=4, delta_unstable=10

Time for backsubstitution: 1.28 seconds

### RS candidates at layer 1
type: RSZ, layer: 1, pos: 219

Time for candidate selection: 0.14 seconds

### Candidate
type: RSZ, layer: 1, pos: 219

### Relational analysis RSZ of RS_RSZ1_RSZ2_RSZ2_RSZ1_RSZ2_RSZ2_RSZ1

#### Relational analysis RSZ result of RS_RSZ1_RSZ2_RSZ2_RSZ1_RSZ2_RSZ2_RSZ1
Status: Status.UNKNOWN
Output dim: 9, lower bound: -0.0053776, upper bound: 0.0048599
time: 0.79 seconds

### Relational analysis RSZ of RS_RSZ1_RSZ2_RSZ2_RSZ1_RSZ2_RSZ2_RSZ2

#### Relational analysis RSZ result of RS_RSZ1_RSZ2_RSZ2_RSZ1_RSZ2_RSZ2_RSZ2
Status: Status.UNKNOWN
Output dim: 9, lower bound: -0.0051936, upper bound: 0.0049236
time: 0.81 seconds

## BFS RS instance: RS_RSZ1_RSZ2_RSZ2_RSZ2_RSZ1_RSZ1

### Backsubstitution after applying RS history:
0: -0.0039095, -0.0003029, -0.0039095, -0.0003029, -0.0031067, 0.0032392
1: -0.0044907, 0.0040812, -0.0044907, 0.0040812, -0.0040930, 0.0037596
2: 0.0029569, 0.0099431, 0.0029569, 0.0099431, -0.0039946, 0.0037733
3: -0.0045519, -0.0034923, -0.0045519, -0.0034923, -0.0010595, 0.0010595
4: 0.0017018, 0.0078700, 0.0017018, 0.0078700, -0.0052386, 0.0053388
5: -0.0030341, 0.0031953, -0.0030341, 0.0031953, -0.0048378, 0.0046768
6: -0.0068516, -0.0033211, -0.0068516, -0.0033211, -0.0033673, 0.0033381
7: -0.0021570, 0.0038259, -0.0021570, 0.0038259, -0.0057572, 0.0057920
8: -0.0008533, 0.0000782, -0.0008533, 0.0000782, -0.0008780, 0.0008415
9: 0.9973557, 1.0121032, 0.9973557, 1.0121032, -0.0089271, 0.0093661

### Unstable ReLU Count (Linear/Conv2D Layers)
- layer_idx=0, type=LayerType.Linear, total=256, inp1_unstable=7, inp2_unstable=7, delta_unstable=250
- layer_idx=2, type=LayerType.Linear, total=256, inp1_unstable=10, inp2_unstable=10, delta_unstable=256
- layer_idx=4, type=LayerType.Linear, total=256, inp1_unstable=1, inp2_unstable=1, delta_unstable=256
- layer_idx=6, type=LayerType.Linear, total=256, inp1_unstable=0, inp2_unstable=0, delta_unstable=256
- layer_idx=8, type=LayerType.Linear, total=10, inp1_unstable=4, inp2_unstable=4, delta_unstable=10

Time for backsubstitution: 1.36 seconds

### RS candidates at layer 1
type: RSZ, layer: 1, pos: 219

Time for candidate selection: 0.13 seconds

### Candidate
type: RSZ, layer: 1, pos: 219

### Relational analysis RSZ of RS_RSZ1_RSZ2_RSZ2_RSZ2_RSZ1_RSZ1_RSZ1

#### Relational analysis RSZ result of RS_RSZ1_RSZ2_RSZ2_RSZ2_RSZ1_RSZ1_RSZ1
Status: Status.UNKNOWN
Output dim: 9, lower bound: -0.0047958, upper bound: 0.0052953
time: 0.78 seconds

### Relational analysis RSZ of RS_RSZ1_RSZ2_RSZ2_RSZ2_RSZ1_RSZ1_RSZ2

#### Relational analysis RSZ result of RS_RSZ1_RSZ2_RSZ2_RSZ2_RSZ1_RSZ1_RSZ2
Status: Status.UNKNOWN
Output dim: 9, lower bound: -0.0047313, upper bound: 0.0054735
time: 0.89 seconds

## BFS RS instance: RS_RSZ1_RSZ2_RSZ2_RSZ2_RSZ1_RSZ2

### Backsubstitution after applying RS history:
0: -0.0039095, -0.0003029, -0.0039095, -0.0003029, -0.0031058, 0.0032394
1: -0.0044907, 0.0040812, -0.0044907, 0.0040812, -0.0040930, 0.0037592
2: 0.0029569, 0.0099431, 0.0029569, 0.0099431, -0.0039947, 0.0037734
3: -0.0045519, -0.0034923, -0.0045519, -0.0034923, -0.0010595, 0.0010595
4: 0.0017018, 0.0078700, 0.0017018, 0.0078700, -0.0052381, 0.0053388
5: -0.0030341, 0.0031953, -0.0030341, 0.0031953, -0.0048377, 0.0046738
6: -0.0068516, -0.0033211, -0.0068516, -0.0033211, -0.0033673, 0.0033381
7: -0.0021570, 0.0038259, -0.0021570, 0.0038259, -0.0057562, 0.0057912
8: -0.0008533, 0.0000782, -0.0008533, 0.0000782, -0.0008767, 0.0008405
9: 0.9973557, 1.0121032, 0.9973557, 1.0121032, -0.0089263, 0.0093661

### Unstable ReLU Count (Linear/Conv2D Layers)
- layer_idx=0, type=LayerType.Linear, total=256, inp1_unstable=7, inp2_unstable=7, delta_unstable=250
- layer_idx=2, type=LayerType.Linear, total=256, inp1_unstable=10, inp2_unstable=10, delta_unstable=256
- layer_idx=4, type=LayerType.Linear, total=256, inp1_unstable=1, inp2_unstable=1, delta_unstable=256
- layer_idx=6, type=LayerType.Linear, total=256, inp1_unstable=0, inp2_unstable=0, delta_unstable=256
- layer_idx=8, type=LayerType.Linear, total=10, inp1_unstable=4, inp2_unstable=4, delta_unstable=10

Time for backsubstitution: 1.29 seconds

### RS candidates at layer 1
type: RSZ, layer: 1, pos: 219

Time for candidate selection: 0.13 seconds

### Candidate
type: RSZ, layer: 1, pos: 219

### Relational analysis RSZ of RS_RSZ1_RSZ2_RSZ2_RSZ2_RSZ1_RSZ2_RSZ1

#### Relational analysis RSZ result of RS_RSZ1_RSZ2_RSZ2_RSZ2_RSZ1_RSZ2_RSZ1
Status: Status.UNKNOWN
Output dim: 9, lower bound: -0.0047958, upper bound: 0.0052950
time: 0.78 seconds

### Relational analysis RSZ of RS_RSZ1_RSZ2_RSZ2_RSZ2_RSZ1_RSZ2_RSZ2

#### Relational analysis RSZ result of RS_RSZ1_RSZ2_RSZ2_RSZ2_RSZ1_RSZ2_RSZ2
Status: Status.UNKNOWN
Output dim: 9, lower bound: -0.0047313, upper bound: 0.0054735
time: 0.85 seconds

## BFS RS instance: RS_RSZ1_RSZ2_RSZ2_RSZ2_RSZ2_RSZ1

### Backsubstitution after applying RS history:
0: -0.0039095, -0.0003029, -0.0039095, -0.0003029, -0.0031637, 0.0031904
1: -0.0044907, 0.0040812, -0.0044907, 0.0040812, -0.0039714, 0.0039022
2: 0.0029569, 0.0099431, 0.0029569, 0.0099431, -0.0039086, 0.0038700
3: -0.0045519, -0.0034923, -0.0045519, -0.0034923, -0.0010595, 0.0010595
4: 0.0017018, 0.0078700, 0.0017018, 0.0078700, -0.0052824, 0.0053017
5: -0.0030341, 0.0031953, -0.0030341, 0.0031953, -0.0047792, 0.0047365
6: -0.0068516, -0.0033211, -0.0068516, -0.0033211, -0.0033567, 0.0033503
7: -0.0021570, 0.0038259, -0.0021570, 0.0038259, -0.0057836, 0.0057600
8: -0.0008533, 0.0000782, -0.0008533, 0.0000782, -0.0008716, 0.0008490
9: 0.9973557, 1.0121032, 0.9973557, 1.0121032, -0.0091104, 0.0092069

### Unstable ReLU Count (Linear/Conv2D Layers)
- layer_idx=0, type=LayerType.Linear, total=256, inp1_unstable=7, inp2_unstable=7, delta_unstable=250
- layer_idx=2, type=LayerType.Linear, total=256, inp1_unstable=10, inp2_unstable=10, delta_unstable=256
- layer_idx=4, type=LayerType.Linear, total=256, inp1_unstable=1, inp2_unstable=1, delta_unstable=256
- layer_idx=6, type=LayerType.Linear, total=256, inp1_unstable=0, inp2_unstable=0, delta_unstable=256
- layer_idx=8, type=LayerType.Linear, total=10, inp1_unstable=4, inp2_unstable=4, delta_unstable=10

Time for backsubstitution: 1.32 seconds

### RS candidates at layer 1
type: RSZ, layer: 1, pos: 219

Time for candidate selection: 0.14 seconds

### Candidate
type: RSZ, layer: 1, pos: 219

### Relational analysis RSZ of RS_RSZ1_RSZ2_RSZ2_RSZ2_RSZ2_RSZ1_RSZ1

#### Relational analysis RSZ result of RS_RSZ1_RSZ2_RSZ2_RSZ2_RSZ2_RSZ1_RSZ1
Status: Status.UNKNOWN
Output dim: 9, lower bound: -0.0049073, upper bound: 0.0051222
time: 0.77 seconds

### Relational analysis RSZ of RS_RSZ1_RSZ2_RSZ2_RSZ2_RSZ2_RSZ1_RSZ2

#### Relational analysis RSZ result of RS_RSZ1_RSZ2_RSZ2_RSZ2_RSZ2_RSZ1_RSZ2
Status: Status.UNKNOWN
Output dim: 9, lower bound: -0.0048586, upper bound: 0.0053248
time: 0.81 seconds

## BFS RS instance: RS_RSZ1_RSZ2_RSZ2_RSZ2_RSZ2_RSZ2

### Backsubstitution after applying RS history:
0: -0.0039095, -0.0003029, -0.0039095, -0.0003029, -0.0031634, 0.0031907
1: -0.0044907, 0.0040812, -0.0044907, 0.0040812, -0.0039717, 0.0039024
2: 0.0029569, 0.0099431, 0.0029569, 0.0099431, -0.0039092, 0.0038700
3: -0.0045519, -0.0034923, -0.0045519, -0.0034923, -0.0010595, 0.0010595
4: 0.0017018, 0.0078700, 0.0017018, 0.0078700, -0.0052821, 0.0053021
5: -0.0030341, 0.0031953, -0.0030341, 0.0031953, -0.0047795, 0.0047352
6: -0.0068516, -0.0033211, -0.0068516, -0.0033211, -0.0033568, 0.0033503
7: -0.0021570, 0.0038259, -0.0021570, 0.0038259, -0.0057830, 0.0057606
8: -0.0008533, 0.0000782, -0.0008533, 0.0000782, -0.0008703, 0.0008478
9: 0.9973557, 1.0121032, 0.9973557, 1.0121032, -0.0091104, 0.0092074

### Unstable ReLU Count (Linear/Conv2D Layers)
- layer_idx=0, type=LayerType.Linear, total=256, inp1_unstable=7, inp2_unstable=7, delta_unstable=250
- layer_idx=2, type=LayerType.Linear, total=256, inp1_unstable=10, inp2_unstable=10, delta_unstable=256
- layer_idx=4, type=LayerType.Linear, total=256, inp1_unstable=1, inp2_unstable=1, delta_unstable=256
- layer_idx=6, type=LayerType.Linear, total=256, inp1_unstable=0, inp2_unstable=0, delta_unstable=256
- layer_idx=8, type=LayerType.Linear, total=10, inp1_unstable=4, inp2_unstable=4, delta_unstable=10

Time for backsubstitution: 1.29 seconds

### RS candidates at layer 1
type: RSZ, layer: 1, pos: 219

Time for candidate selection: 0.13 seconds

### Candidate
type: RSZ, layer: 1, pos: 219

### Relational analysis RSZ of RS_RSZ1_RSZ2_RSZ2_RSZ2_RSZ2_RSZ2_RSZ1

#### Relational analysis RSZ result of RS_RSZ1_RSZ2_RSZ2_RSZ2_RSZ2_RSZ2_RSZ1
Status: Status.UNKNOWN
Output dim: 9, lower bound: -0.0049073, upper bound: 0.0051221
time: 0.76 seconds

### Relational analysis RSZ of RS_RSZ1_RSZ2_RSZ2_RSZ2_RSZ2_RSZ2_RSZ2

#### Relational analysis RSZ result of RS_RSZ1_RSZ2_RSZ2_RSZ2_RSZ2_RSZ2_RSZ2
Status: Status.UNKNOWN
Output dim: 9, lower bound: -0.0048586, upper bound: 0.0053248
time: 0.80 seconds

## BFS RS instance: RS_RSZ2_RSZ1_RSZ1_RSZ1_RSZ1_RSZ1

### Backsubstitution after applying RS history:
0: -0.0039095, -0.0003029, -0.0039095, -0.0003029, -0.0031907, 0.0031634
1: -0.0044907, 0.0040812, -0.0044907, 0.0040812, -0.0039024, 0.0039717
2: 0.0029569, 0.0099431, 0.0029569, 0.0099431, -0.0038700, 0.0039092
3: -0.0045519, -0.0034923, -0.0045519, -0.0034923, -0.0010595, 0.0010595
4: 0.0017018, 0.0078700, 0.0017018, 0.0078700, -0.0053021, 0.0052821
5: -0.0030341, 0.0031953, -0.0030341, 0.0031953, -0.0047352, 0.0047795
6: -0.0068516, -0.0033211, -0.0068516, -0.0033211, -0.0033503, 0.0033568
7: -0.0021570, 0.0038259, -0.0021570, 0.0038259, -0.0057606, 0.0057830
8: -0.0008533, 0.0000782, -0.0008533, 0.0000782, -0.0008478, 0.0008703
9: 0.9973557, 1.0121032, 0.9973557, 1.0121032, -0.0092074, 0.0091104

### Unstable ReLU Count (Linear/Conv2D Layers)
- layer_idx=0, type=LayerType.Linear, total=256, inp1_unstable=7, inp2_unstable=7, delta_unstable=250
- layer_idx=2, type=LayerType.Linear, total=256, inp1_unstable=10, inp2_unstable=10, delta_unstable=256
- layer_idx=4, type=LayerType.Linear, total=256, inp1_unstable=1, inp2_unstable=1, delta_unstable=256
- layer_idx=6, type=LayerType.Linear, total=256, inp1_unstable=0, inp2_unstable=0, delta_unstable=256
- layer_idx=8, type=LayerType.Linear, total=10, inp1_unstable=4, inp2_unstable=4, delta_unstable=10

Time for backsubstitution: 1.34 seconds

### RS candidates at layer 1
type: RSZ, layer: 1, pos: 219

Time for candidate selection: 0.13 seconds

### Candidate
type: RSZ, layer: 1, pos: 219

### Relational analysis RSZ of RS_RSZ2_RSZ1_RSZ1_RSZ1_RSZ1_RSZ1_RSZ1

#### Relational analysis RSZ result of RS_RSZ2_RSZ1_RSZ1_RSZ1_RSZ1_RSZ1_RSZ1
Status: Status.UNKNOWN
Output dim: 9, lower bound: -0.0053248, upper bound: 0.0048586
time: 0.75 seconds

### Relational analysis RSZ of RS_RSZ2_RSZ1_RSZ1_RSZ1_RSZ1_RSZ1_RSZ2

#### Relational analysis RSZ result of RS_RSZ2_RSZ1_RSZ1_RSZ1_RSZ1_RSZ1_RSZ2
Status: Status.UNKNOWN
Output dim: 9, lower bound: -0.0051221, upper bound: 0.0049073
time: 0.80 seconds

## BFS RS instance: RS_RSZ2_RSZ1_RSZ1_RSZ1_RSZ1_RSZ2

### Backsubstitution after applying RS history:
0: -0.0039095, -0.0003029, -0.0039095, -0.0003029, -0.0031904, 0.0031637
1: -0.0044907, 0.0040812, -0.0044907, 0.0040812, -0.0039022, 0.0039714
2: 0.0029569, 0.0099431, 0.0029569, 0.0099431, -0.0038700, 0.0039086
3: -0.0045519, -0.0034923, -0.0045519, -0.0034923, -0.0010595, 0.0010595
4: 0.0017018, 0.0078700, 0.0017018, 0.0078700, -0.0053017, 0.0052824
5: -0.0030341, 0.0031953, -0.0030341, 0.0031953, -0.0047365, 0.0047792
6: -0.0068516, -0.0033211, -0.0068516, -0.0033211, -0.0033503, 0.0033567
7: -0.0021570, 0.0038259, -0.0021570, 0.0038259, -0.0057600, 0.0057836
8: -0.0008533, 0.0000782, -0.0008533, 0.0000782, -0.0008490, 0.0008716
9: 0.9973557, 1.0121032, 0.9973557, 1.0121032, -0.0092069, 0.0091104

### Unstable ReLU Count (Linear/Conv2D Layers)
- layer_idx=0, type=LayerType.Linear, total=256, inp1_unstable=7, inp2_unstable=7, delta_unstable=250
- layer_idx=2, type=LayerType.Linear, total=256, inp1_unstable=10, inp2_unstable=10, delta_unstable=256
- layer_idx=4, type=LayerType.Linear, total=256, inp1_unstable=1, inp2_unstable=1, delta_unstable=256
- layer_idx=6, type=LayerType.Linear, total=256, inp1_unstable=0, inp2_unstable=0, delta_unstable=256
- layer_idx=8, type=LayerType.Linear, total=10, inp1_unstable=4, inp2_unstable=4, delta_unstable=10

Time for backsubstitution: 1.35 seconds

### RS candidates at layer 1
type: RSZ, layer: 1, pos: 219

Time for candidate selection: 0.14 seconds

### Candidate
type: RSZ, layer: 1, pos: 219

### Relational analysis RSZ of RS_RSZ2_RSZ1_RSZ1_RSZ1_RSZ1_RSZ2_RSZ1

#### Relational analysis RSZ result of RS_RSZ2_RSZ1_RSZ1_RSZ1_RSZ1_RSZ2_RSZ1
Status: Status.UNKNOWN
Output dim: 9, lower bound: -0.0053248, upper bound: 0.0048586
time: 0.75 seconds

### Relational analysis RSZ of RS_RSZ2_RSZ1_RSZ1_RSZ1_RSZ1_RSZ2_RSZ2

#### Relational analysis RSZ result of RS_RSZ2_RSZ1_RSZ1_RSZ1_RSZ1_RSZ2_RSZ2
Status: Status.UNKNOWN
Output dim: 9, lower bound: -0.0051222, upper bound: 0.0049073
time: 0.86 seconds

## BFS RS instance: RS_RSZ2_RSZ1_RSZ1_RSZ1_RSZ2_RSZ1

### Backsubstitution after applying RS history:
0: -0.0039095, -0.0003029, -0.0039095, -0.0003029, -0.0032394, 0.0031058
1: -0.0044907, 0.0040812, -0.0044907, 0.0040812, -0.0037592, 0.0040930
2: 0.0029569, 0.0099431, 0.0029569, 0.0099431, -0.0037734, 0.0039947
3: -0.0045519, -0.0034923, -0.0045519, -0.0034923, -0.0010595, 0.0010595
4: 0.0017018, 0.0078700, 0.0017018, 0.0078700, -0.0053388, 0.0052381
5: -0.0030341, 0.0031953, -0.0030341, 0.0031953, -0.0046738, 0.0048377
6: -0.0068516, -0.0033211, -0.0068516, -0.0033211, -0.0033381, 0.0033673
7: -0.0021570, 0.0038259, -0.0021570, 0.0038259, -0.0057912, 0.0057562
8: -0.0008533, 0.0000782, -0.0008533, 0.0000782, -0.0008405, 0.0008767
9: 0.9973557, 1.0121032, 0.9973557, 1.0121032, -0.0093661, 0.0089263

### Unstable ReLU Count (Linear/Conv2D Layers)
- layer_idx=0, type=LayerType.Linear, total=256, inp1_unstable=7, inp2_unstable=7, delta_unstable=250
- layer_idx=2, type=LayerType.Linear, total=256, inp1_unstable=10, inp2_unstable=10, delta_unstable=256
- layer_idx=4, type=LayerType.Linear, total=256, inp1_unstable=1, inp2_unstable=1, delta_unstable=256
- layer_idx=6, type=LayerType.Linear, total=256, inp1_unstable=0, inp2_unstable=0, delta_unstable=256
- layer_idx=8, type=LayerType.Linear, total=10, inp1_unstable=4, inp2_unstable=4, delta_unstable=10

Time for backsubstitution: 1.33 seconds

### RS candidates at layer 1
type: RSZ, layer: 1, pos: 219

Time for candidate selection: 0.14 seconds

### Candidate
type: RSZ, layer: 1, pos: 219

### Relational analysis RSZ of RS_RSZ2_RSZ1_RSZ1_RSZ1_RSZ2_RSZ1_RSZ1

#### Relational analysis RSZ result of RS_RSZ2_RSZ1_RSZ1_RSZ1_RSZ2_RSZ1_RSZ1
Status: Status.UNKNOWN
Output dim: 9, lower bound: -0.0054735, upper bound: 0.0047313
time: 0.78 seconds

### Relational analysis RSZ of RS_RSZ2_RSZ1_RSZ1_RSZ1_RSZ2_RSZ1_RSZ2

#### Relational analysis RSZ result of RS_RSZ2_RSZ1_RSZ1_RSZ1_RSZ2_RSZ1_RSZ2
Status: Status.UNKNOWN
Output dim: 9, lower bound: -0.0052949, upper bound: 0.0047958
time: 0.81 seconds

## BFS RS instance: RS_RSZ2_RSZ1_RSZ1_RSZ1_RSZ2_RSZ2

### Backsubstitution after applying RS history:
0: -0.0039095, -0.0003029, -0.0039095, -0.0003029, -0.0032392, 0.0031067
1: -0.0044907, 0.0040812, -0.0044907, 0.0040812, -0.0037596, 0.0040930
2: 0.0029569, 0.0099431, 0.0029569, 0.0099431, -0.0037733, 0.0039946
3: -0.0045519, -0.0034923, -0.0045519, -0.0034923, -0.0010595, 0.0010595
4: 0.0017018, 0.0078700, 0.0017018, 0.0078700, -0.0053388, 0.0052386
5: -0.0030341, 0.0031953, -0.0030341, 0.0031953, -0.0046768, 0.0048378
6: -0.0068516, -0.0033211, -0.0068516, -0.0033211, -0.0033381, 0.0033673
7: -0.0021570, 0.0038259, -0.0021570, 0.0038259, -0.0057920, 0.0057572
8: -0.0008533, 0.0000782, -0.0008533, 0.0000782, -0.0008415, 0.0008780
9: 0.9973557, 1.0121032, 0.9973557, 1.0121032, -0.0093661, 0.0089271

### Unstable ReLU Count (Linear/Conv2D Layers)
- layer_idx=0, type=LayerType.Linear, total=256, inp1_unstable=7, inp2_unstable=7, delta_unstable=250
- layer_idx=2, type=LayerType.Linear, total=256, inp1_unstable=10, inp2_unstable=10, delta_unstable=256
- layer_idx=4, type=LayerType.Linear, total=256, inp1_unstable=1, inp2_unstable=1, delta_unstable=256
- layer_idx=6, type=LayerType.Linear, total=256, inp1_unstable=0, inp2_unstable=0, delta_unstable=256
- layer_idx=8, type=LayerType.Linear, total=10, inp1_unstable=4, inp2_unstable=4, delta_unstable=10

Time for backsubstitution: 1.32 seconds

### RS candidates at layer 1
type: RSZ, layer: 1, pos: 219

Time for candidate selection: 0.13 seconds

### Candidate
type: RSZ, layer: 1, pos: 219

### Relational analysis RSZ of RS_RSZ2_RSZ1_RSZ1_RSZ1_RSZ2_RSZ2_RSZ1

#### Relational analysis RSZ result of RS_RSZ2_RSZ1_RSZ1_RSZ1_RSZ2_RSZ2_RSZ1
Status: Status.UNKNOWN
Output dim: 9, lower bound: -0.0054735, upper bound: 0.0047313
time: 0.76 seconds

### Relational analysis RSZ of RS_RSZ2_RSZ1_RSZ1_RSZ1_RSZ2_RSZ2_RSZ2

#### Relational analysis RSZ result of RS_RSZ2_RSZ1_RSZ1_RSZ1_RSZ2_RSZ2_RSZ2
Status: Status.UNKNOWN
Output dim: 9, lower bound: -0.0052953, upper bound: 0.0047958
time: 0.82 seconds

## BFS RS instance: RS_RSZ2_RSZ1_RSZ1_RSZ2_RSZ1_RSZ1

### Backsubstitution after applying RS history:
0: -0.0039095, -0.0003029, -0.0039095, -0.0003029, -0.0031233, 0.0032329
1: -0.0044907, 0.0040812, -0.0044907, 0.0040812, -0.0040597, 0.0038115
2: 0.0029569, 0.0099431, 0.0029569, 0.0099431, -0.0039721, 0.0038047
3: -0.0045519, -0.0034923, -0.0045519, -0.0034923, -0.0010595, 0.0010595
4: 0.0017018, 0.0078700, 0.0017018, 0.0078700, -0.0052504, 0.0053351
5: -0.0030341, 0.0031953, -0.0030341, 0.0031953, -0.0048146, 0.0047039
6: -0.0068516, -0.0033211, -0.0068516, -0.0033211, -0.0033646, 0.0033425
7: -0.0021570, 0.0038259, -0.0021570, 0.0038259, -0.0057374, 0.0058075
8: -0.0008533, 0.0000782, -0.0008533, 0.0000782, -0.0008585, 0.0008633
9: 0.9973557, 1.0121032, 0.9973557, 1.0121032, -0.0089899, 0.0093313

### Unstable ReLU Count (Linear/Conv2D Layers)
- layer_idx=0, type=LayerType.Linear, total=256, inp1_unstable=7, inp2_unstable=7, delta_unstable=250
- layer_idx=2, type=LayerType.Linear, total=256, inp1_unstable=10, inp2_unstable=10, delta_unstable=256
- layer_idx=4, type=LayerType.Linear, total=256, inp1_unstable=1, inp2_unstable=1, delta_unstable=256
- layer_idx=6, type=LayerType.Linear, total=256, inp1_unstable=0, inp2_unstable=0, delta_unstable=256
- layer_idx=8, type=LayerType.Linear, total=10, inp1_unstable=4, inp2_unstable=4, delta_unstable=10

Time for backsubstitution: 1.31 seconds

### RS candidates at layer 1
type: RSZ, layer: 1, pos: 219

Time for candidate selection: 0.14 seconds

### Candidate
type: RSZ, layer: 1, pos: 219

### Relational analysis RSZ of RS_RSZ2_RSZ1_RSZ1_RSZ2_RSZ1_RSZ1_RSZ1

#### Relational analysis RSZ result of RS_RSZ2_RSZ1_RSZ1_RSZ2_RSZ1_RSZ1_RSZ1
Status: Status.UNKNOWN
Output dim: 9, lower bound: -0.0049236, upper bound: 0.0051936
time: 0.80 seconds

### Relational analysis RSZ of RS_RSZ2_RSZ1_RSZ1_RSZ2_RSZ1_RSZ1_RSZ2

#### Relational analysis RSZ result of RS_RSZ2_RSZ1_RSZ1_RSZ2_RSZ1_RSZ1_RSZ2
Status: Status.UNKNOWN
Output dim: 9, lower bound: -0.0048599, upper bound: 0.0053776
time: 0.80 seconds

## BFS RS instance: RS_RSZ2_RSZ1_RSZ1_RSZ2_RSZ1_RSZ2

### Backsubstitution after applying RS history:
0: -0.0039095, -0.0003029, -0.0039095, -0.0003029, -0.0031227, 0.0032331
1: -0.0044907, 0.0040812, -0.0044907, 0.0040812, -0.0040596, 0.0038112
2: 0.0029569, 0.0099431, 0.0029569, 0.0099431, -0.0039720, 0.0038036
3: -0.0045519, -0.0034923, -0.0045519, -0.0034923, -0.0010595, 0.0010595
4: 0.0017018, 0.0078700, 0.0017018, 0.0078700, -0.0052500, 0.0053352
5: -0.0030341, 0.0031953, -0.0030341, 0.0031953, -0.0048151, 0.0047023
6: -0.0068516, -0.0033211, -0.0068516, -0.0033211, -0.0033646, 0.0033425
7: -0.0021570, 0.0038259, -0.0021570, 0.0038259, -0.0057371, 0.0058083
8: -0.0008533, 0.0000782, -0.0008533, 0.0000782, -0.0008584, 0.0008641
9: 0.9973557, 1.0121032, 0.9973557, 1.0121032, -0.0089892, 0.0093313

### Unstable ReLU Count (Linear/Conv2D Layers)
- layer_idx=0, type=LayerType.Linear, total=256, inp1_unstable=7, inp2_unstable=7, delta_unstable=250
- layer_idx=2, type=LayerType.Linear, total=256, inp1_unstable=10, inp2_unstable=10, delta_unstable=256
- layer_idx=4, type=LayerType.Linear, total=256, inp1_unstable=1, inp2_unstable=1, delta_unstable=256
- layer_idx=6, type=LayerType.Linear, total=256, inp1_unstable=0, inp2_unstable=0, delta_unstable=256
- layer_idx=8, type=LayerType.Linear, total=10, inp1_unstable=4, inp2_unstable=4, delta_unstable=10

Time for backsubstitution: 1.36 seconds

### RS candidates at layer 1
type: RSZ, layer: 1, pos: 219

Time for candidate selection: 0.15 seconds

### Candidate
type: RSZ, layer: 1, pos: 219

### Relational analysis RSZ of RS_RSZ2_RSZ1_RSZ1_RSZ2_RSZ1_RSZ2_RSZ1

#### Relational analysis RSZ result of RS_RSZ2_RSZ1_RSZ1_RSZ2_RSZ1_RSZ2_RSZ1
Status: Status.UNKNOWN
Output dim: 9, lower bound: -0.0049236, upper bound: 0.0051929
time: 0.79 seconds

### Relational analysis RSZ of RS_RSZ2_RSZ1_RSZ1_RSZ2_RSZ1_RSZ2_RSZ2

#### Relational analysis RSZ result of RS_RSZ2_RSZ1_RSZ1_RSZ2_RSZ1_RSZ2_RSZ2
Status: Status.UNKNOWN
Output dim: 9, lower bound: -0.0048599, upper bound: 0.0053776
time: 0.82 seconds

## BFS RS instance: RS_RSZ2_RSZ1_RSZ1_RSZ2_RSZ2_RSZ1

### Backsubstitution after applying RS history:
0: -0.0039095, -0.0003029, -0.0039095, -0.0003029, -0.0031695, 0.0031774
1: -0.0044907, 0.0040812, -0.0044907, 0.0040812, -0.0039358, 0.0039347
2: 0.0029569, 0.0099431, 0.0029569, 0.0099431, -0.0038903, 0.0038913
3: -0.0045519, -0.0034923, -0.0045519, -0.0034923, -0.0010595, 0.0010595
4: 0.0017018, 0.0078700, 0.0017018, 0.0078700, -0.0052856, 0.0052930
5: -0.0030341, 0.0031953, -0.0030341, 0.0031953, -0.0047513, 0.0047616
6: -0.0068516, -0.0033211, -0.0068516, -0.0033211, -0.0033536, 0.0033530
7: -0.0021570, 0.0038259, -0.0021570, 0.0038259, -0.0057705, 0.0057811
8: -0.0008533, 0.0000782, -0.0008533, 0.0000782, -0.0008511, 0.0008666
9: 0.9973557, 1.0121032, 0.9973557, 1.0121032, -0.0091461, 0.0091617

### Unstable ReLU Count (Linear/Conv2D Layers)
- layer_idx=0, type=LayerType.Linear, total=256, inp1_unstable=7, inp2_unstable=7, delta_unstable=250
- layer_idx=2, type=LayerType.Linear, total=256, inp1_unstable=10, inp2_unstable=10, delta_unstable=256
- layer_idx=4, type=LayerType.Linear, total=256, inp1_unstable=1, inp2_unstable=1, delta_unstable=256
- layer_idx=6, type=LayerType.Linear, total=256, inp1_unstable=0, inp2_unstable=0, delta_unstable=256
- layer_idx=8, type=LayerType.Linear, total=10, inp1_unstable=4, inp2_unstable=4, delta_unstable=10

Time for backsubstitution: 1.42 seconds

### RS candidates at layer 1
type: RSZ, layer: 1, pos: 219

Time for candidate selection: 0.14 seconds

### Candidate
type: RSZ, layer: 1, pos: 219

### Relational analysis RSZ of RS_RSZ2_RSZ1_RSZ1_RSZ2_RSZ2_RSZ1_RSZ1

#### Relational analysis RSZ result of RS_RSZ2_RSZ1_RSZ1_RSZ2_RSZ2_RSZ1_RSZ1
Status: Status.VERIFIED
Output dim: 9, lower bound: -0.0050256, upper bound: 0.0050426
time: 0.77 seconds

### Relational analysis RSZ of RS_RSZ2_RSZ1_RSZ1_RSZ2_RSZ2_RSZ1_RSZ2

#### Relational analysis RSZ result of RS_RSZ2_RSZ1_RSZ1_RSZ2_RSZ2_RSZ1_RSZ2
Status: Status.UNKNOWN
Output dim: 9, lower bound: -0.0049775, upper bound: 0.0052573
time: 0.80 seconds

## BFS RS instance: RS_RSZ2_RSZ1_RSZ1_RSZ2_RSZ2_RSZ2

### Backsubstitution after applying RS history:
0: -0.0039095, -0.0003029, -0.0039095, -0.0003029, -0.0031693, 0.0031778
1: -0.0044907, 0.0040812, -0.0044907, 0.0040812, -0.0039359, 0.0039346
2: 0.0029569, 0.0099431, 0.0029569, 0.0099431, -0.0038901, 0.0038910
3: -0.0045519, -0.0034923, -0.0045519, -0.0034923, -0.0010595, 0.0010595
4: 0.0017018, 0.0078700, 0.0017018, 0.0078700, -0.0052854, 0.0052933
5: -0.0030341, 0.0031953, -0.0030341, 0.0031953, -0.0047523, 0.0047609
6: -0.0068516, -0.0033211, -0.0068516, -0.0033211, -0.0033536, 0.0033530
7: -0.0021570, 0.0038259, -0.0021570, 0.0038259, -0.0057700, 0.0057826
8: -0.0008533, 0.0000782, -0.0008533, 0.0000782, -0.0008504, 0.0008684
9: 0.9973557, 1.0121032, 0.9973557, 1.0121032, -0.0091461, 0.0091620

### Unstable ReLU Count (Linear/Conv2D Layers)
- layer_idx=0, type=LayerType.Linear, total=256, inp1_unstable=7, inp2_unstable=7, delta_unstable=250
- layer_idx=2, type=LayerType.Linear, total=256, inp1_unstable=10, inp2_unstable=10, delta_unstable=256
- layer_idx=4, type=LayerType.Linear, total=256, inp1_unstable=1, inp2_unstable=1, delta_unstable=256
- layer_idx=6, type=LayerType.Linear, total=256, inp1_unstable=0, inp2_unstable=0, delta_unstable=256
- layer_idx=8, type=LayerType.Linear, total=10, inp1_unstable=4, inp2_unstable=4, delta_unstable=10

Time for backsubstitution: 1.32 seconds

### RS candidates at layer 1
type: RSZ, layer: 1, pos: 219

Time for candidate selection: 0.17 seconds

### Candidate
type: RSZ, layer: 1, pos: 219

### Relational analysis RSZ of RS_RSZ2_RSZ1_RSZ1_RSZ2_RSZ2_RSZ2_RSZ1

#### Relational analysis RSZ result of RS_RSZ2_RSZ1_RSZ1_RSZ2_RSZ2_RSZ2_RSZ1
Status: Status.VERIFIED
Output dim: 9, lower bound: -0.0050256, upper bound: 0.0050422
time: 0.76 seconds

### Relational analysis RSZ of RS_RSZ2_RSZ1_RSZ1_RSZ2_RSZ2_RSZ2_RSZ2

#### Relational analysis RSZ result of RS_RSZ2_RSZ1_RSZ1_RSZ2_RSZ2_RSZ2_RSZ2
Status: Status.UNKNOWN
Output dim: 9, lower bound: -0.0049775, upper bound: 0.0052573
time: 0.81 seconds

## BFS RS instance: RS_RSZ2_RSZ1_RSZ2_RSZ1_RSZ1_RSZ1

### Backsubstitution after applying RS history:
0: -0.0039095, -0.0003029, -0.0039095, -0.0003029, -0.0031776, 0.0031697
1: -0.0044907, 0.0040812, -0.0044907, 0.0040812, -0.0039161, 0.0039470
2: 0.0029569, 0.0099431, 0.0029569, 0.0099431, -0.0038784, 0.0038948
3: -0.0045519, -0.0034923, -0.0045519, -0.0034923, -0.0010595, 0.0010595
4: 0.0017018, 0.0078700, 0.0017018, 0.0078700, -0.0052922, 0.0052869
5: -0.0030341, 0.0031953, -0.0030341, 0.0031953, -0.0047427, 0.0047666
6: -0.0068516, -0.0033211, -0.0068516, -0.0033211, -0.0033518, 0.0033544
7: -0.0021570, 0.0038259, -0.0021570, 0.0038259, -0.0057551, 0.0057870
8: -0.0008533, 0.0000782, -0.0008533, 0.0000782, -0.0008392, 0.0008824
9: 0.9973557, 1.0121032, 0.9973557, 1.0121032, -0.0091706, 0.0091341

### Unstable ReLU Count (Linear/Conv2D Layers)
- layer_idx=0, type=LayerType.Linear, total=256, inp1_unstable=7, inp2_unstable=7, delta_unstable=250
- layer_idx=2, type=LayerType.Linear, total=256, inp1_unstable=10, inp2_unstable=10, delta_unstable=256
- layer_idx=4, type=LayerType.Linear, total=256, inp1_unstable=1, inp2_unstable=1, delta_unstable=256
- layer_idx=6, type=LayerType.Linear, total=256, inp1_unstable=0, inp2_unstable=0, delta_unstable=256
- layer_idx=8, type=LayerType.Linear, total=10, inp1_unstable=4, inp2_unstable=4, delta_unstable=10

Time for backsubstitution: 1.33 seconds

### RS candidates at layer 1
type: RSZ, layer: 1, pos: 219

Time for candidate selection: 0.13 seconds

### Candidate
type: RSZ, layer: 1, pos: 219

### Relational analysis RSZ of RS_RSZ2_RSZ1_RSZ2_RSZ1_RSZ1_RSZ1_RSZ1

#### Relational analysis RSZ result of RS_RSZ2_RSZ1_RSZ2_RSZ1_RSZ1_RSZ1_RSZ1
Status: Status.UNKNOWN
Output dim: 9, lower bound: -0.0052867, upper bound: 0.0048928
time: 0.81 seconds

### Relational analysis RSZ of RS_RSZ2_RSZ1_RSZ2_RSZ1_RSZ1_RSZ1_RSZ2

#### Relational analysis RSZ result of RS_RSZ2_RSZ1_RSZ2_RSZ1_RSZ1_RSZ1_RSZ2
Status: Status.UNKNOWN
Output dim: 9, lower bound: -0.0050997, upper bound: 0.0049568
time: 0.79 seconds

## BFS RS instance: RS_RSZ2_RSZ1_RSZ2_RSZ1_RSZ1_RSZ2

### Backsubstitution after applying RS history:
0: -0.0039095, -0.0003029, -0.0039095, -0.0003029, -0.0031773, 0.0031700
1: -0.0044907, 0.0040812, -0.0044907, 0.0040812, -0.0039161, 0.0039468
2: 0.0029569, 0.0099431, 0.0029569, 0.0099431, -0.0038781, 0.0038943
3: -0.0045519, -0.0034923, -0.0045519, -0.0034923, -0.0010595, 0.0010595
4: 0.0017018, 0.0078700, 0.0017018, 0.0078700, -0.0052920, 0.0052873
5: -0.0030341, 0.0031953, -0.0030341, 0.0031953, -0.0047434, 0.0047659
6: -0.0068516, -0.0033211, -0.0068516, -0.0033211, -0.0033518, 0.0033544
7: -0.0021570, 0.0038259, -0.0021570, 0.0038259, -0.0057544, 0.0057876
8: -0.0008533, 0.0000782, -0.0008533, 0.0000782, -0.0008397, 0.0008837
9: 0.9973557, 1.0121032, 0.9973557, 1.0121032, -0.0091702, 0.0091342

### Unstable ReLU Count (Linear/Conv2D Layers)
- layer_idx=0, type=LayerType.Linear, total=256, inp1_unstable=7, inp2_unstable=7, delta_unstable=250
- layer_idx=2, type=LayerType.Linear, total=256, inp1_unstable=10, inp2_unstable=10, delta_unstable=256
- layer_idx=4, type=LayerType.Linear, total=256, inp1_unstable=1, inp2_unstable=1, delta_unstable=256
- layer_idx=6, type=LayerType.Linear, total=256, inp1_unstable=0, inp2_unstable=0, delta_unstable=256
- layer_idx=8, type=LayerType.Linear, total=10, inp1_unstable=4, inp2_unstable=4, delta_unstable=10

Time for backsubstitution: 1.32 seconds

### RS candidates at layer 1
type: RSZ, layer: 1, pos: 219

Time for candidate selection: 0.15 seconds

### Candidate
type: RSZ, layer: 1, pos: 219

### Relational analysis RSZ of RS_RSZ2_RSZ1_RSZ2_RSZ1_RSZ1_RSZ2_RSZ1

#### Relational analysis RSZ result of RS_RSZ2_RSZ1_RSZ2_RSZ1_RSZ1_RSZ2_RSZ1
Status: Status.UNKNOWN
Output dim: 9, lower bound: -0.0052867, upper bound: 0.0048928
time: 0.80 seconds

### Relational analysis RSZ of RS_RSZ2_RSZ1_RSZ2_RSZ1_RSZ1_RSZ2_RSZ2

#### Relational analysis RSZ result of RS_RSZ2_RSZ1_RSZ2_RSZ1_RSZ1_RSZ2_RSZ2
Status: Status.UNKNOWN
Output dim: 9, lower bound: -0.0050997, upper bound: 0.0049568
time: 0.79 seconds

## BFS RS instance: RS_RSZ2_RSZ1_RSZ2_RSZ1_RSZ2_RSZ1

### Backsubstitution after applying RS history:
0: -0.0039095, -0.0003029, -0.0039095, -0.0003029, -0.0032338, 0.0031230
1: -0.0044907, 0.0040812, -0.0044907, 0.0040812, -0.0037934, 0.0040782
2: 0.0029569, 0.0099431, 0.0029569, 0.0099431, -0.0037926, 0.0039853
3: -0.0045519, -0.0034923, -0.0045519, -0.0034923, -0.0010595, 0.0010595
4: 0.0017018, 0.0078700, 0.0017018, 0.0078700, -0.0053346, 0.0052509
5: -0.0030341, 0.0031953, -0.0030341, 0.0031953, -0.0046906, 0.0048337
6: -0.0068516, -0.0033211, -0.0068516, -0.0033211, -0.0033413, 0.0033660
7: -0.0021570, 0.0038259, -0.0021570, 0.0038259, -0.0057874, 0.0057631
8: -0.0008533, 0.0000782, -0.0008533, 0.0000782, -0.0008328, 0.0008885
9: 0.9973557, 1.0121032, 0.9973557, 1.0121032, -0.0093477, 0.0089777

### Unstable ReLU Count (Linear/Conv2D Layers)
- layer_idx=0, type=LayerType.Linear, total=256, inp1_unstable=7, inp2_unstable=7, delta_unstable=250
- layer_idx=2, type=LayerType.Linear, total=256, inp1_unstable=10, inp2_unstable=10, delta_unstable=256
- layer_idx=4, type=LayerType.Linear, total=256, inp1_unstable=1, inp2_unstable=1, delta_unstable=256
- layer_idx=6, type=LayerType.Linear, total=256, inp1_unstable=0, inp2_unstable=0, delta_unstable=256
- layer_idx=8, type=LayerType.Linear, total=10, inp1_unstable=4, inp2_unstable=4, delta_unstable=10

Time for backsubstitution: 1.32 seconds

### RS candidates at layer 1
type: RSZ, layer: 1, pos: 219

Time for candidate selection: 0.13 seconds

### Candidate
type: RSZ, layer: 1, pos: 219

### Relational analysis RSZ of RS_RSZ2_RSZ1_RSZ2_RSZ1_RSZ2_RSZ1_RSZ1

#### Relational analysis RSZ result of RS_RSZ2_RSZ1_RSZ2_RSZ1_RSZ2_RSZ1_RSZ1
Status: Status.UNKNOWN
Output dim: 9, lower bound: -0.0054625, upper bound: 0.0047878
time: 0.78 seconds

### Relational analysis RSZ of RS_RSZ2_RSZ1_RSZ2_RSZ1_RSZ2_RSZ1_RSZ2

#### Relational analysis RSZ result of RS_RSZ2_RSZ1_RSZ2_RSZ1_RSZ2_RSZ1_RSZ2
Status: Status.UNKNOWN
Output dim: 9, lower bound: -0.0052874, upper bound: 0.0048628
time: 0.84 seconds

## BFS RS instance: RS_RSZ2_RSZ1_RSZ2_RSZ1_RSZ2_RSZ2

### Backsubstitution after applying RS history:
0: -0.0039095, -0.0003029, -0.0039095, -0.0003029, -0.0032335, 0.0031235
1: -0.0044907, 0.0040812, -0.0044907, 0.0040812, -0.0037936, 0.0040783
2: 0.0029569, 0.0099431, 0.0029569, 0.0099431, -0.0037919, 0.0039858
3: -0.0045519, -0.0034923, -0.0045519, -0.0034923, -0.0010595, 0.0010595
4: 0.0017018, 0.0078700, 0.0017018, 0.0078700, -0.0053345, 0.0052514
5: -0.0030341, 0.0031953, -0.0030341, 0.0031953, -0.0046925, 0.0048337
6: -0.0068516, -0.0033211, -0.0068516, -0.0033211, -0.0033414, 0.0033660
7: -0.0021570, 0.0038259, -0.0021570, 0.0038259, -0.0057879, 0.0057635
8: -0.0008533, 0.0000782, -0.0008533, 0.0000782, -0.0008328, 0.0008900
9: 0.9973557, 1.0121032, 0.9973557, 1.0121032, -0.0093475, 0.0089784

### Unstable ReLU Count (Linear/Conv2D Layers)
- layer_idx=0, type=LayerType.Linear, total=256, inp1_unstable=7, inp2_unstable=7, delta_unstable=250
- layer_idx=2, type=LayerType.Linear, total=256, inp1_unstable=10, inp2_unstable=10, delta_unstable=256
- layer_idx=4, type=LayerType.Linear, total=256, inp1_unstable=1, inp2_unstable=1, delta_unstable=256
- layer_idx=6, type=LayerType.Linear, total=256, inp1_unstable=0, inp2_unstable=0, delta_unstable=256
- layer_idx=8, type=LayerType.Linear, total=10, inp1_unstable=4, inp2_unstable=4, delta_unstable=10

Time for backsubstitution: 1.46 seconds

### RS candidates at layer 1
type: RSZ, layer: 1, pos: 219

Time for candidate selection: 0.17 seconds

### Candidate
type: RSZ, layer: 1, pos: 219

### Relational analysis RSZ of RS_RSZ2_RSZ1_RSZ2_RSZ1_RSZ2_RSZ2_RSZ1

#### Relational analysis RSZ result of RS_RSZ2_RSZ1_RSZ2_RSZ1_RSZ2_RSZ2_RSZ1
Status: Status.UNKNOWN
Output dim: 9, lower bound: -0.0054625, upper bound: 0.0047878
time: 0.84 seconds

### Relational analysis RSZ of RS_RSZ2_RSZ1_RSZ2_RSZ1_RSZ2_RSZ2_RSZ2

#### Relational analysis RSZ result of RS_RSZ2_RSZ1_RSZ2_RSZ1_RSZ2_RSZ2_RSZ2
Status: Status.UNKNOWN
Output dim: 9, lower bound: -0.0052874, upper bound: 0.0048628
time: 0.84 seconds

## BFS RS instance: RS_RSZ2_RSZ1_RSZ2_RSZ2_RSZ1_RSZ1

### Backsubstitution after applying RS history:
0: -0.0039095, -0.0003029, -0.0039095, -0.0003029, -0.0031067, 0.0032384
1: -0.0044907, 0.0040812, -0.0044907, 0.0040812, -0.0040693, 0.0037774
2: 0.0029569, 0.0099431, 0.0029569, 0.0099431, -0.0039787, 0.0037856
3: -0.0045519, -0.0034923, -0.0045519, -0.0034923, -0.0010595, 0.0010595
4: 0.0017018, 0.0078700, 0.0017018, 0.0078700, -0.0052376, 0.0053393
5: -0.0030341, 0.0031953, -0.0030341, 0.0031953, -0.0048190, 0.0046883
6: -0.0068516, -0.0033211, -0.0068516, -0.0033211, -0.0033656, 0.0033392
7: -0.0021570, 0.0038259, -0.0021570, 0.0038259, -0.0057301, 0.0058109
8: -0.0008533, 0.0000782, -0.0008533, 0.0000782, -0.0008483, 0.0008744
9: 0.9973557, 1.0121032, 0.9973557, 1.0121032, -0.0089376, 0.0093452

### Unstable ReLU Count (Linear/Conv2D Layers)
- layer_idx=0, type=LayerType.Linear, total=256, inp1_unstable=7, inp2_unstable=7, delta_unstable=250
- layer_idx=2, type=LayerType.Linear, total=256, inp1_unstable=10, inp2_unstable=10, delta_unstable=256
- layer_idx=4, type=LayerType.Linear, total=256, inp1_unstable=1, inp2_unstable=1, delta_unstable=256
- layer_idx=6, type=LayerType.Linear, total=256, inp1_unstable=0, inp2_unstable=0, delta_unstable=256
- layer_idx=8, type=LayerType.Linear, total=10, inp1_unstable=4, inp2_unstable=4, delta_unstable=10

Time for backsubstitution: 1.33 seconds

### RS candidates at layer 1
type: RSZ, layer: 1, pos: 219

Time for candidate selection: 0.14 seconds

### Candidate
type: RSZ, layer: 1, pos: 219

### Relational analysis RSZ of RS_RSZ2_RSZ1_RSZ2_RSZ2_RSZ1_RSZ1_RSZ1

#### Relational analysis RSZ result of RS_RSZ2_RSZ1_RSZ2_RSZ2_RSZ1_RSZ1_RSZ1
Status: Status.UNKNOWN
Output dim: 9, lower bound: -0.0048588, upper bound: 0.0051947
time: 0.85 seconds

### Relational analysis RSZ of RS_RSZ2_RSZ1_RSZ2_RSZ2_RSZ1_RSZ1_RSZ2

#### Relational analysis RSZ result of RS_RSZ2_RSZ1_RSZ2_RSZ2_RSZ1_RSZ1_RSZ2
Status: Status.UNKNOWN
Output dim: 9, lower bound: -0.0047963, upper bound: 0.0053802
time: 0.83 seconds

## BFS RS instance: RS_RSZ2_RSZ1_RSZ2_RSZ2_RSZ1_RSZ2

### Backsubstitution after applying RS history:
0: -0.0039095, -0.0003029, -0.0039095, -0.0003029, -0.0031058, 0.0032384
1: -0.0044907, 0.0040812, -0.0044907, 0.0040812, -0.0040691, 0.0037770
2: 0.0029569, 0.0099431, 0.0029569, 0.0099431, -0.0039787, 0.0037846
3: -0.0045519, -0.0034923, -0.0045519, -0.0034923, -0.0010595, 0.0010595
4: 0.0017018, 0.0078700, 0.0017018, 0.0078700, -0.0052371, 0.0053393
5: -0.0030341, 0.0031953, -0.0030341, 0.0031953, -0.0048197, 0.0046862
6: -0.0068516, -0.0033211, -0.0068516, -0.0033211, -0.0033655, 0.0033392
7: -0.0021570, 0.0038259, -0.0021570, 0.0038259, -0.0057298, 0.0058112
8: -0.0008533, 0.0000782, -0.0008533, 0.0000782, -0.0008484, 0.0008748
9: 0.9973557, 1.0121032, 0.9973557, 1.0121032, -0.0089368, 0.0093450

### Unstable ReLU Count (Linear/Conv2D Layers)
- layer_idx=0, type=LayerType.Linear, total=256, inp1_unstable=7, inp2_unstable=7, delta_unstable=250
- layer_idx=2, type=LayerType.Linear, total=256, inp1_unstable=10, inp2_unstable=10, delta_unstable=256
- layer_idx=4, type=LayerType.Linear, total=256, inp1_unstable=1, inp2_unstable=1, delta_unstable=256
- layer_idx=6, type=LayerType.Linear, total=256, inp1_unstable=0, inp2_unstable=0, delta_unstable=256
- layer_idx=8, type=LayerType.Linear, total=10, inp1_unstable=4, inp2_unstable=4, delta_unstable=10

Time for backsubstitution: 1.34 seconds

### RS candidates at layer 1
type: RSZ, layer: 1, pos: 219

Time for candidate selection: 0.15 seconds

### Candidate
type: RSZ, layer: 1, pos: 219

### Relational analysis RSZ of RS_RSZ2_RSZ1_RSZ2_RSZ2_RSZ1_RSZ2_RSZ1

#### Relational analysis RSZ result of RS_RSZ2_RSZ1_RSZ2_RSZ2_RSZ1_RSZ2_RSZ1
Status: Status.UNKNOWN
Output dim: 9, lower bound: -0.0048579, upper bound: 0.0051935
time: 0.84 seconds

### Relational analysis RSZ of RS_RSZ2_RSZ1_RSZ2_RSZ2_RSZ1_RSZ2_RSZ2

#### Relational analysis RSZ result of RS_RSZ2_RSZ1_RSZ2_RSZ2_RSZ1_RSZ2_RSZ2
Status: Status.UNKNOWN
Output dim: 9, lower bound: -0.0047963, upper bound: 0.0053802
time: 0.81 seconds

## BFS RS instance: RS_RSZ2_RSZ1_RSZ2_RSZ2_RSZ2_RSZ1

### Backsubstitution after applying RS history:
0: -0.0039095, -0.0003029, -0.0039095, -0.0003029, -0.0031636, 0.0031906
1: -0.0044907, 0.0040812, -0.0044907, 0.0040812, -0.0039577, 0.0039219
2: 0.0029569, 0.0099431, 0.0029569, 0.0099431, -0.0039044, 0.0038828
3: -0.0045519, -0.0034923, -0.0045519, -0.0034923, -0.0010595, 0.0010595
4: 0.0017018, 0.0078700, 0.0017018, 0.0078700, -0.0052810, 0.0053027
5: -0.0030341, 0.0031953, -0.0030341, 0.0031953, -0.0047659, 0.0047559
6: -0.0068516, -0.0033211, -0.0068516, -0.0033211, -0.0033558, 0.0033517
7: -0.0021570, 0.0038259, -0.0021570, 0.0038259, -0.0057655, 0.0057875
8: -0.0008533, 0.0000782, -0.0008533, 0.0000782, -0.0008412, 0.0008790
9: 0.9973557, 1.0121032, 0.9973557, 1.0121032, -0.0091260, 0.0091977

### Unstable ReLU Count (Linear/Conv2D Layers)
- layer_idx=0, type=LayerType.Linear, total=256, inp1_unstable=7, inp2_unstable=7, delta_unstable=250
- layer_idx=2, type=LayerType.Linear, total=256, inp1_unstable=10, inp2_unstable=10, delta_unstable=256
- layer_idx=4, type=LayerType.Linear, total=256, inp1_unstable=1, inp2_unstable=1, delta_unstable=256
- layer_idx=6, type=LayerType.Linear, total=256, inp1_unstable=0, inp2_unstable=0, delta_unstable=256
- layer_idx=8, type=LayerType.Linear, total=10, inp1_unstable=4, inp2_unstable=4, delta_unstable=10

Time for backsubstitution: 1.34 seconds

### RS candidates at layer 1
type: RSZ, layer: 1, pos: 219

Time for candidate selection: 0.14 seconds

### Candidate
type: RSZ, layer: 1, pos: 219

### Relational analysis RSZ of RS_RSZ2_RSZ1_RSZ2_RSZ2_RSZ2_RSZ1_RSZ1

#### Relational analysis RSZ result of RS_RSZ2_RSZ1_RSZ2_RSZ2_RSZ2_RSZ1_RSZ1
Status: Status.VERIFIED
Output dim: 9, lower bound: -0.0049978, upper bound: 0.0050529
time: 0.83 seconds

### Relational analysis RSZ of RS_RSZ2_RSZ1_RSZ2_RSZ2_RSZ2_RSZ1_RSZ2

#### Relational analysis RSZ result of RS_RSZ2_RSZ1_RSZ2_RSZ2_RSZ2_RSZ1_RSZ2
Status: Status.UNKNOWN
Output dim: 9, lower bound: -0.0049534, upper bound: 0.0052773
time: 0.80 seconds

## BFS RS instance: RS_RSZ2_RSZ1_RSZ2_RSZ2_RSZ2_RSZ2

### Backsubstitution after applying RS history:
0: -0.0039095, -0.0003029, -0.0039095, -0.0003029, -0.0031632, 0.0031910
1: -0.0044907, 0.0040812, -0.0044907, 0.0040812, -0.0039579, 0.0039218
2: 0.0029569, 0.0099431, 0.0029569, 0.0099431, -0.0039041, 0.0038827
3: -0.0045519, -0.0034923, -0.0045519, -0.0034923, -0.0010595, 0.0010595
4: 0.0017018, 0.0078700, 0.0017018, 0.0078700, -0.0052808, 0.0053030
5: -0.0030341, 0.0031953, -0.0030341, 0.0031953, -0.0047664, 0.0047549
6: -0.0068516, -0.0033211, -0.0068516, -0.0033211, -0.0033558, 0.0033517
7: -0.0021570, 0.0038259, -0.0021570, 0.0038259, -0.0057648, 0.0057889
8: -0.0008533, 0.0000782, -0.0008533, 0.0000782, -0.0008408, 0.0008795
9: 0.9973557, 1.0121032, 0.9973557, 1.0121032, -0.0091260, 0.0091980

### Unstable ReLU Count (Linear/Conv2D Layers)
- layer_idx=0, type=LayerType.Linear, total=256, inp1_unstable=7, inp2_unstable=7, delta_unstable=250
- layer_idx=2, type=LayerType.Linear, total=256, inp1_unstable=10, inp2_unstable=10, delta_unstable=256
- layer_idx=4, type=LayerType.Linear, total=256, inp1_unstable=1, inp2_unstable=1, delta_unstable=256
- layer_idx=6, type=LayerType.Linear, total=256, inp1_unstable=0, inp2_unstable=0, delta_unstable=256
- layer_idx=8, type=LayerType.Linear, total=10, inp1_unstable=4, inp2_unstable=4, delta_unstable=10

Time for backsubstitution: 1.31 seconds

### RS candidates at layer 1
type: RSZ, layer: 1, pos: 219

Time for candidate selection: 0.14 seconds

### Candidate
type: RSZ, layer: 1, pos: 219

### Relational analysis RSZ of RS_RSZ2_RSZ1_RSZ2_RSZ2_RSZ2_RSZ2_RSZ1

#### Relational analysis RSZ result of RS_RSZ2_RSZ1_RSZ2_RSZ2_RSZ2_RSZ2_RSZ1
Status: Status.VERIFIED
Output dim: 9, lower bound: -0.0049978, upper bound: 0.0050522
time: 0.83 seconds

### Relational analysis RSZ of RS_RSZ2_RSZ1_RSZ2_RSZ2_RSZ2_RSZ2_RSZ2

#### Relational analysis RSZ result of RS_RSZ2_RSZ1_RSZ2_RSZ2_RSZ2_RSZ2_RSZ2
Status: Status.UNKNOWN
Output dim: 9, lower bound: -0.0049534, upper bound: 0.0052773
time: 0.82 seconds

## BFS RS instance: RS_RSZ2_RSZ2_RSZ1_RSZ1_RSZ1_RSZ1

### Backsubstitution after applying RS history:
0: -0.0039095, -0.0003029, -0.0039095, -0.0003029, -0.0031894, 0.0031651
1: -0.0044907, 0.0040812, -0.0044907, 0.0040812, -0.0039322, 0.0039494
2: 0.0029569, 0.0099431, 0.0029569, 0.0099431, -0.0038910, 0.0038951
3: -0.0045519, -0.0034923, -0.0045519, -0.0034923, -0.0010595, 0.0010595
4: 0.0017018, 0.0078700, 0.0017018, 0.0078700, -0.0053018, 0.0052825
5: -0.0030341, 0.0031953, -0.0030341, 0.0031953, -0.0047555, 0.0047658
6: -0.0068516, -0.0033211, -0.0068516, -0.0033211, -0.0033525, 0.0033553
7: -0.0021570, 0.0038259, -0.0021570, 0.0038259, -0.0057822, 0.0057696
8: -0.0008533, 0.0000782, -0.0008533, 0.0000782, -0.0008792, 0.0008410
9: 0.9973557, 1.0121032, 0.9973557, 1.0121032, -0.0091896, 0.0091359

### Unstable ReLU Count (Linear/Conv2D Layers)
- layer_idx=0, type=LayerType.Linear, total=256, inp1_unstable=7, inp2_unstable=7, delta_unstable=250
- layer_idx=2, type=LayerType.Linear, total=256, inp1_unstable=10, inp2_unstable=10, delta_unstable=256
- layer_idx=4, type=LayerType.Linear, total=256, inp1_unstable=1, inp2_unstable=1, delta_unstable=256
- layer_idx=6, type=LayerType.Linear, total=256, inp1_unstable=0, inp2_unstable=0, delta_unstable=256
- layer_idx=8, type=LayerType.Linear, total=10, inp1_unstable=4, inp2_unstable=4, delta_unstable=10

Time for backsubstitution: 1.44 seconds

### RS candidates at layer 1
type: RSZ, layer: 1, pos: 219

Time for candidate selection: 0.14 seconds

### Candidate
type: RSZ, layer: 1, pos: 219

### Relational analysis RSZ of RS_RSZ2_RSZ2_RSZ1_RSZ1_RSZ1_RSZ1_RSZ1

#### Relational analysis RSZ result of RS_RSZ2_RSZ2_RSZ1_RSZ1_RSZ1_RSZ1_RSZ1
Status: Status.UNKNOWN
Output dim: 9, lower bound: -0.0052257, upper bound: 0.0049917
time: 0.78 seconds

### Relational analysis RSZ of RS_RSZ2_RSZ2_RSZ1_RSZ1_RSZ1_RSZ1_RSZ2

#### Relational analysis RSZ result of RS_RSZ2_RSZ2_RSZ1_RSZ1_RSZ1_RSZ1_RSZ2
Status: Status.VERIFIED
Output dim: 9, lower bound: -0.0050264, upper bound: 0.0050255
time: 0.89 seconds

## BFS RS instance: RS_RSZ2_RSZ2_RSZ1_RSZ1_RSZ1_RSZ2

### Backsubstitution after applying RS history:
0: -0.0039095, -0.0003029, -0.0039095, -0.0003029, -0.0031892, 0.0031654
1: -0.0044907, 0.0040812, -0.0044907, 0.0040812, -0.0039323, 0.0039491
2: 0.0029569, 0.0099431, 0.0029569, 0.0099431, -0.0038912, 0.0038954
3: -0.0045519, -0.0034923, -0.0045519, -0.0034923, -0.0010595, 0.0010595
4: 0.0017018, 0.0078700, 0.0017018, 0.0078700, -0.0053015, 0.0052828
5: -0.0030341, 0.0031953, -0.0030341, 0.0031953, -0.0047564, 0.0047653
6: -0.0068516, -0.0033211, -0.0068516, -0.0033211, -0.0033525, 0.0033553
7: -0.0021570, 0.0038259, -0.0021570, 0.0038259, -0.0057806, 0.0057690
8: -0.0008533, 0.0000782, -0.0008533, 0.0000782, -0.0008784, 0.0008407
9: 0.9973557, 1.0121032, 0.9973557, 1.0121032, -0.0091892, 0.0091361

### Unstable ReLU Count (Linear/Conv2D Layers)
- layer_idx=0, type=LayerType.Linear, total=256, inp1_unstable=7, inp2_unstable=7, delta_unstable=250
- layer_idx=2, type=LayerType.Linear, total=256, inp1_unstable=10, inp2_unstable=10, delta_unstable=256
- layer_idx=4, type=LayerType.Linear, total=256, inp1_unstable=1, inp2_unstable=1, delta_unstable=256
- layer_idx=6, type=LayerType.Linear, total=256, inp1_unstable=0, inp2_unstable=0, delta_unstable=256
- layer_idx=8, type=LayerType.Linear, total=10, inp1_unstable=4, inp2_unstable=4, delta_unstable=10

Time for backsubstitution: 1.32 seconds

### RS candidates at layer 1
type: RSZ, layer: 1, pos: 219

Time for candidate selection: 0.14 seconds

### Candidate
type: RSZ, layer: 1, pos: 219

### Relational analysis RSZ of RS_RSZ2_RSZ2_RSZ1_RSZ1_RSZ1_RSZ2_RSZ1

#### Relational analysis RSZ result of RS_RSZ2_RSZ2_RSZ1_RSZ1_RSZ1_RSZ2_RSZ1
Status: Status.UNKNOWN
Output dim: 9, lower bound: -0.0052257, upper bound: 0.0049917
time: 0.78 seconds

### Relational analysis RSZ of RS_RSZ2_RSZ2_RSZ1_RSZ1_RSZ1_RSZ2_RSZ2

#### Relational analysis RSZ result of RS_RSZ2_RSZ2_RSZ1_RSZ1_RSZ1_RSZ2_RSZ2
Status: Status.VERIFIED
Output dim: 9, lower bound: -0.0050265, upper bound: 0.0050255
time: 0.89 seconds

## BFS RS instance: RS_RSZ2_RSZ2_RSZ1_RSZ1_RSZ2_RSZ1

### Backsubstitution after applying RS history:
0: -0.0039095, -0.0003029, -0.0039095, -0.0003029, -0.0032375, 0.0031072
1: -0.0044907, 0.0040812, -0.0044907, 0.0040812, -0.0037849, 0.0040634
2: 0.0029569, 0.0099431, 0.0029569, 0.0099431, -0.0037915, 0.0039728
3: -0.0045519, -0.0034923, -0.0045519, -0.0034923, -0.0010595, 0.0010595
4: 0.0017018, 0.0078700, 0.0017018, 0.0078700, -0.0053384, 0.0052384
5: -0.0030341, 0.0031953, -0.0030341, 0.0031953, -0.0046871, 0.0048192
6: -0.0068516, -0.0033211, -0.0068516, -0.0033211, -0.0033398, 0.0033652
7: -0.0021570, 0.0038259, -0.0021570, 0.0038259, -0.0058059, 0.0057373
8: -0.0008533, 0.0000782, -0.0008533, 0.0000782, -0.0008742, 0.0008482
9: 0.9973557, 1.0121032, 0.9973557, 1.0121032, -0.0093398, 0.0089458

### Unstable ReLU Count (Linear/Conv2D Layers)
- layer_idx=0, type=LayerType.Linear, total=256, inp1_unstable=7, inp2_unstable=7, delta_unstable=250
- layer_idx=2, type=LayerType.Linear, total=256, inp1_unstable=10, inp2_unstable=10, delta_unstable=256
- layer_idx=4, type=LayerType.Linear, total=256, inp1_unstable=1, inp2_unstable=1, delta_unstable=256
- layer_idx=6, type=LayerType.Linear, total=256, inp1_unstable=0, inp2_unstable=0, delta_unstable=256
- layer_idx=8, type=LayerType.Linear, total=10, inp1_unstable=4, inp2_unstable=4, delta_unstable=10

Time for backsubstitution: 1.34 seconds

### RS candidates at layer 1
type: RSZ, layer: 1, pos: 219

Time for candidate selection: 0.14 seconds

### Candidate
type: RSZ, layer: 1, pos: 219

### Relational analysis RSZ of RS_RSZ2_RSZ2_RSZ1_RSZ1_RSZ2_RSZ1_RSZ1

#### Relational analysis RSZ result of RS_RSZ2_RSZ2_RSZ1_RSZ1_RSZ2_RSZ1_RSZ1
Status: Status.UNKNOWN
Output dim: 9, lower bound: -0.0053493, upper bound: 0.0048358
time: 0.76 seconds

### Relational analysis RSZ of RS_RSZ2_RSZ2_RSZ1_RSZ1_RSZ2_RSZ1_RSZ2

#### Relational analysis RSZ result of RS_RSZ2_RSZ2_RSZ1_RSZ1_RSZ2_RSZ1_RSZ2
Status: Status.UNKNOWN
Output dim: 9, lower bound: -0.0051587, upper bound: 0.0049024
time: 0.82 seconds

## BFS RS instance: RS_RSZ2_RSZ2_RSZ1_RSZ1_RSZ2_RSZ2

### Backsubstitution after applying RS history:
0: -0.0039095, -0.0003029, -0.0039095, -0.0003029, -0.0032372, 0.0031080
1: -0.0044907, 0.0040812, -0.0044907, 0.0040812, -0.0037852, 0.0040637
2: 0.0029569, 0.0099431, 0.0029569, 0.0099431, -0.0037926, 0.0039730
3: -0.0045519, -0.0034923, -0.0045519, -0.0034923, -0.0010595, 0.0010595
4: 0.0017018, 0.0078700, 0.0017018, 0.0078700, -0.0053383, 0.0052389
5: -0.0030341, 0.0031953, -0.0030341, 0.0031953, -0.0046892, 0.0048184
6: -0.0068516, -0.0033211, -0.0068516, -0.0033211, -0.0033398, 0.0033652
7: -0.0021570, 0.0038259, -0.0021570, 0.0038259, -0.0058053, 0.0057367
8: -0.0008533, 0.0000782, -0.0008533, 0.0000782, -0.0008737, 0.0008477
9: 0.9973557, 1.0121032, 0.9973557, 1.0121032, -0.0093398, 0.0089466

### Unstable ReLU Count (Linear/Conv2D Layers)
- layer_idx=0, type=LayerType.Linear, total=256, inp1_unstable=7, inp2_unstable=7, delta_unstable=250
- layer_idx=2, type=LayerType.Linear, total=256, inp1_unstable=10, inp2_unstable=10, delta_unstable=256
- layer_idx=4, type=LayerType.Linear, total=256, inp1_unstable=1, inp2_unstable=1, delta_unstable=256
- layer_idx=6, type=LayerType.Linear, total=256, inp1_unstable=0, inp2_unstable=0, delta_unstable=256
- layer_idx=8, type=LayerType.Linear, total=10, inp1_unstable=4, inp2_unstable=4, delta_unstable=10

Time for backsubstitution: 1.35 seconds

### RS candidates at layer 1
type: RSZ, layer: 1, pos: 219

Time for candidate selection: 0.14 seconds

### Candidate
type: RSZ, layer: 1, pos: 219

### Relational analysis RSZ of RS_RSZ2_RSZ2_RSZ1_RSZ1_RSZ2_RSZ2_RSZ1

#### Relational analysis RSZ result of RS_RSZ2_RSZ2_RSZ1_RSZ1_RSZ2_RSZ2_RSZ1
Status: Status.UNKNOWN
Output dim: 9, lower bound: -0.0053493, upper bound: 0.0048358
time: 0.78 seconds

### Relational analysis RSZ of RS_RSZ2_RSZ2_RSZ1_RSZ1_RSZ2_RSZ2_RSZ2

#### Relational analysis RSZ result of RS_RSZ2_RSZ2_RSZ1_RSZ1_RSZ2_RSZ2_RSZ2
Status: Status.UNKNOWN
Output dim: 9, lower bound: -0.0051598, upper bound: 0.0049024
time: 0.81 seconds

## BFS RS instance: RS_RSZ2_RSZ2_RSZ1_RSZ2_RSZ1_RSZ1

### Backsubstitution after applying RS history:
0: -0.0039095, -0.0003029, -0.0039095, -0.0003029, -0.0031220, 0.0032347
1: -0.0044907, 0.0040812, -0.0044907, 0.0040812, -0.0040847, 0.0037853
2: 0.0029569, 0.0099431, 0.0029569, 0.0099431, -0.0039903, 0.0037836
3: -0.0045519, -0.0034923, -0.0045519, -0.0034923, -0.0010595, 0.0010595
4: 0.0017018, 0.0078700, 0.0017018, 0.0078700, -0.0052502, 0.0053355
5: -0.0030341, 0.0031953, -0.0030341, 0.0031953, -0.0048342, 0.0046918
6: -0.0068516, -0.0033211, -0.0068516, -0.0033211, -0.0033665, 0.0033407
7: -0.0021570, 0.0038259, -0.0021570, 0.0038259, -0.0057559, 0.0057932
8: -0.0008533, 0.0000782, -0.0008533, 0.0000782, -0.0008897, 0.0008329
9: 0.9973557, 1.0121032, 0.9973557, 1.0121032, -0.0089700, 0.0093542

### Unstable ReLU Count (Linear/Conv2D Layers)
- layer_idx=0, type=LayerType.Linear, total=256, inp1_unstable=7, inp2_unstable=7, delta_unstable=250
- layer_idx=2, type=LayerType.Linear, total=256, inp1_unstable=10, inp2_unstable=10, delta_unstable=256
- layer_idx=4, type=LayerType.Linear, total=256, inp1_unstable=1, inp2_unstable=1, delta_unstable=256
- layer_idx=6, type=LayerType.Linear, total=256, inp1_unstable=0, inp2_unstable=0, delta_unstable=256
- layer_idx=8, type=LayerType.Linear, total=10, inp1_unstable=4, inp2_unstable=4, delta_unstable=10

Time for backsubstitution: 1.38 seconds

### RS candidates at layer 1
type: RSZ, layer: 1, pos: 219

Time for candidate selection: 0.14 seconds

### Candidate
type: RSZ, layer: 1, pos: 219

### Relational analysis RSZ of RS_RSZ2_RSZ2_RSZ1_RSZ2_RSZ1_RSZ1_RSZ1

#### Relational analysis RSZ result of RS_RSZ2_RSZ2_RSZ1_RSZ2_RSZ1_RSZ1_RSZ1
Status: Status.UNKNOWN
Output dim: 9, lower bound: -0.0047969, upper bound: 0.0053095
time: 0.79 seconds

### Relational analysis RSZ of RS_RSZ2_RSZ2_RSZ1_RSZ2_RSZ1_RSZ1_RSZ2

#### Relational analysis RSZ result of RS_RSZ2_RSZ2_RSZ1_RSZ2_RSZ1_RSZ1_RSZ2
Status: Status.UNKNOWN
Output dim: 9, lower bound: -0.0047392, upper bound: 0.0054783
time: 0.87 seconds

## BFS RS instance: RS_RSZ2_RSZ2_RSZ1_RSZ2_RSZ1_RSZ2

### Backsubstitution after applying RS history:
0: -0.0039095, -0.0003029, -0.0039095, -0.0003029, -0.0031213, 0.0032349
1: -0.0044907, 0.0040812, -0.0044907, 0.0040812, -0.0040844, 0.0037851
2: 0.0029569, 0.0099431, 0.0029569, 0.0099431, -0.0039900, 0.0037835
3: -0.0045519, -0.0034923, -0.0045519, -0.0034923, -0.0010595, 0.0010595
4: 0.0017018, 0.0078700, 0.0017018, 0.0078700, -0.0052497, 0.0053356
5: -0.0030341, 0.0031953, -0.0030341, 0.0031953, -0.0048342, 0.0046899
6: -0.0068516, -0.0033211, -0.0068516, -0.0033211, -0.0033665, 0.0033407
7: -0.0021570, 0.0038259, -0.0021570, 0.0038259, -0.0057558, 0.0057921
8: -0.0008533, 0.0000782, -0.0008533, 0.0000782, -0.0008879, 0.0008327
9: 0.9973557, 1.0121032, 0.9973557, 1.0121032, -0.0089693, 0.0093541

### Unstable ReLU Count (Linear/Conv2D Layers)
- layer_idx=0, type=LayerType.Linear, total=256, inp1_unstable=7, inp2_unstable=7, delta_unstable=250
- layer_idx=2, type=LayerType.Linear, total=256, inp1_unstable=10, inp2_unstable=10, delta_unstable=256
- layer_idx=4, type=LayerType.Linear, total=256, inp1_unstable=1, inp2_unstable=1, delta_unstable=256
- layer_idx=6, type=LayerType.Linear, total=256, inp1_unstable=0, inp2_unstable=0, delta_unstable=256
- layer_idx=8, type=LayerType.Linear, total=10, inp1_unstable=4, inp2_unstable=4, delta_unstable=10

Time for backsubstitution: 1.33 seconds

### RS candidates at layer 1
type: RSZ, layer: 1, pos: 219

Time for candidate selection: 0.13 seconds

### Candidate
type: RSZ, layer: 1, pos: 219

### Relational analysis RSZ of RS_RSZ2_RSZ2_RSZ1_RSZ2_RSZ1_RSZ2_RSZ1

#### Relational analysis RSZ result of RS_RSZ2_RSZ2_RSZ1_RSZ2_RSZ1_RSZ2_RSZ1
Status: Status.UNKNOWN
Output dim: 9, lower bound: -0.0047969, upper bound: 0.0053094
time: 0.79 seconds

### Relational analysis RSZ of RS_RSZ2_RSZ2_RSZ1_RSZ2_RSZ1_RSZ2_RSZ2

#### Relational analysis RSZ result of RS_RSZ2_RSZ2_RSZ1_RSZ2_RSZ1_RSZ2_RSZ2
Status: Status.UNKNOWN
Output dim: 9, lower bound: -0.0047392, upper bound: 0.0054783
time: 0.90 seconds

## BFS RS instance: RS_RSZ2_RSZ2_RSZ1_RSZ2_RSZ2_RSZ1

### Backsubstitution after applying RS history:
0: -0.0039095, -0.0003029, -0.0039095, -0.0003029, -0.0031679, 0.0031786
1: -0.0044907, 0.0040812, -0.0044907, 0.0040812, -0.0039542, 0.0039043
2: 0.0029569, 0.0099431, 0.0029569, 0.0099431, -0.0039004, 0.0038708
3: -0.0045519, -0.0034923, -0.0045519, -0.0034923, -0.0010595, 0.0010595
4: 0.0017018, 0.0078700, 0.0017018, 0.0078700, -0.0052853, 0.0052933
5: -0.0030341, 0.0031953, -0.0030341, 0.0031953, -0.0047665, 0.0047425
6: -0.0068516, -0.0033211, -0.0068516, -0.0033211, -0.0033550, 0.0033509
7: -0.0021570, 0.0038259, -0.0021570, 0.0038259, -0.0057829, 0.0057619
8: -0.0008533, 0.0000782, -0.0008533, 0.0000782, -0.0008832, 0.0008395
9: 0.9973557, 1.0121032, 0.9973557, 1.0121032, -0.0091221, 0.0091779

### Unstable ReLU Count (Linear/Conv2D Layers)
- layer_idx=0, type=LayerType.Linear, total=256, inp1_unstable=7, inp2_unstable=7, delta_unstable=250
- layer_idx=2, type=LayerType.Linear, total=256, inp1_unstable=10, inp2_unstable=10, delta_unstable=256
- layer_idx=4, type=LayerType.Linear, total=256, inp1_unstable=1, inp2_unstable=1, delta_unstable=256
- layer_idx=6, type=LayerType.Linear, total=256, inp1_unstable=0, inp2_unstable=0, delta_unstable=256
- layer_idx=8, type=LayerType.Linear, total=10, inp1_unstable=4, inp2_unstable=4, delta_unstable=10

Time for backsubstitution: 1.44 seconds

### RS candidates at layer 1
type: RSZ, layer: 1, pos: 219

Time for candidate selection: 0.15 seconds

### Candidate
type: RSZ, layer: 1, pos: 219

### Relational analysis RSZ of RS_RSZ2_RSZ2_RSZ1_RSZ2_RSZ2_RSZ1_RSZ1

#### Relational analysis RSZ result of RS_RSZ2_RSZ2_RSZ1_RSZ2_RSZ2_RSZ1_RSZ1
Status: Status.UNKNOWN
Output dim: 9, lower bound: -0.0048890, upper bound: 0.0051239
time: 0.77 seconds

### Relational analysis RSZ of RS_RSZ2_RSZ2_RSZ1_RSZ2_RSZ2_RSZ1_RSZ2

#### Relational analysis RSZ result of RS_RSZ2_RSZ2_RSZ1_RSZ2_RSZ2_RSZ1_RSZ2
Status: Status.UNKNOWN
Output dim: 9, lower bound: -0.0048282, upper bound: 0.0053174
time: 0.91 seconds

## BFS RS instance: RS_RSZ2_RSZ2_RSZ1_RSZ2_RSZ2_RSZ2

### Backsubstitution after applying RS history:
0: -0.0039095, -0.0003029, -0.0039095, -0.0003029, -0.0031676, 0.0031789
1: -0.0044907, 0.0040812, -0.0044907, 0.0040812, -0.0039543, 0.0039043
2: 0.0029569, 0.0099431, 0.0029569, 0.0099431, -0.0039003, 0.0038706
3: -0.0045519, -0.0034923, -0.0045519, -0.0034923, -0.0010595, 0.0010595
4: 0.0017018, 0.0078700, 0.0017018, 0.0078700, -0.0052851, 0.0052935
5: -0.0030341, 0.0031953, -0.0030341, 0.0031953, -0.0047671, 0.0047419
6: -0.0068516, -0.0033211, -0.0068516, -0.0033211, -0.0033550, 0.0033509
7: -0.0021570, 0.0038259, -0.0021570, 0.0038259, -0.0057821, 0.0057616
8: -0.0008533, 0.0000782, -0.0008533, 0.0000782, -0.0008816, 0.0008389
9: 0.9973557, 1.0121032, 0.9973557, 1.0121032, -0.0091222, 0.0091783

### Unstable ReLU Count (Linear/Conv2D Layers)
- layer_idx=0, type=LayerType.Linear, total=256, inp1_unstable=7, inp2_unstable=7, delta_unstable=250
- layer_idx=2, type=LayerType.Linear, total=256, inp1_unstable=10, inp2_unstable=10, delta_unstable=256
- layer_idx=4, type=LayerType.Linear, total=256, inp1_unstable=1, inp2_unstable=1, delta_unstable=256
- layer_idx=6, type=LayerType.Linear, total=256, inp1_unstable=0, inp2_unstable=0, delta_unstable=256
- layer_idx=8, type=LayerType.Linear, total=10, inp1_unstable=4, inp2_unstable=4, delta_unstable=10

Time for backsubstitution: 1.41 seconds

### RS candidates at layer 1
type: RSZ, layer: 1, pos: 219

Time for candidate selection: 0.14 seconds

### Candidate
type: RSZ, layer: 1, pos: 219

### Relational analysis RSZ of RS_RSZ2_RSZ2_RSZ1_RSZ2_RSZ2_RSZ2_RSZ1

#### Relational analysis RSZ result of RS_RSZ2_RSZ2_RSZ1_RSZ2_RSZ2_RSZ2_RSZ1
Status: Status.UNKNOWN
Output dim: 9, lower bound: -0.0048890, upper bound: 0.0051236
time: 0.75 seconds

### Relational analysis RSZ of RS_RSZ2_RSZ2_RSZ1_RSZ2_RSZ2_RSZ2_RSZ2

#### Relational analysis RSZ result of RS_RSZ2_RSZ2_RSZ1_RSZ2_RSZ2_RSZ2_RSZ2
Status: Status.UNKNOWN
Output dim: 9, lower bound: -0.0048282, upper bound: 0.0053174
time: 0.90 seconds

## BFS RS instance: RS_RSZ2_RSZ2_RSZ2_RSZ1_RSZ1_RSZ1

### Backsubstitution after applying RS history:
0: -0.0039095, -0.0003029, -0.0039095, -0.0003029, -0.0031764, 0.0031713
1: -0.0044907, 0.0040812, -0.0044907, 0.0040812, -0.0039467, 0.0039285
2: 0.0029569, 0.0099431, 0.0029569, 0.0099431, -0.0038999, 0.0038817
3: -0.0045519, -0.0034923, -0.0045519, -0.0034923, -0.0010595, 0.0010595
4: 0.0017018, 0.0078700, 0.0017018, 0.0078700, -0.0052920, 0.0052873
5: -0.0030341, 0.0031953, -0.0030341, 0.0031953, -0.0047619, 0.0047517
6: -0.0068516, -0.0033211, -0.0068516, -0.0033211, -0.0033539, 0.0033530
7: -0.0021570, 0.0038259, -0.0021570, 0.0038259, -0.0057767, 0.0057745
8: -0.0008533, 0.0000782, -0.0008533, 0.0000782, -0.0008686, 0.0008513
9: 0.9973557, 1.0121032, 0.9973557, 1.0121032, -0.0091547, 0.0091582

### Unstable ReLU Count (Linear/Conv2D Layers)
- layer_idx=0, type=LayerType.Linear, total=256, inp1_unstable=7, inp2_unstable=7, delta_unstable=250
- layer_idx=2, type=LayerType.Linear, total=256, inp1_unstable=10, inp2_unstable=10, delta_unstable=256
- layer_idx=4, type=LayerType.Linear, total=256, inp1_unstable=1, inp2_unstable=1, delta_unstable=256
- layer_idx=6, type=LayerType.Linear, total=256, inp1_unstable=0, inp2_unstable=0, delta_unstable=256
- layer_idx=8, type=LayerType.Linear, total=10, inp1_unstable=4, inp2_unstable=4, delta_unstable=10

Time for backsubstitution: 1.39 seconds

### RS candidates at layer 1
type: RSZ, layer: 1, pos: 219

Time for candidate selection: 0.15 seconds

### Candidate
type: RSZ, layer: 1, pos: 219

### Relational analysis RSZ of RS_RSZ2_RSZ2_RSZ2_RSZ1_RSZ1_RSZ1_RSZ1

#### Relational analysis RSZ result of RS_RSZ2_RSZ2_RSZ2_RSZ1_RSZ1_RSZ1_RSZ1
Status: Status.UNKNOWN
Output dim: 9, lower bound: -0.0052068, upper bound: 0.0050176
time: 0.79 seconds

### Relational analysis RSZ of RS_RSZ2_RSZ2_RSZ2_RSZ1_RSZ1_RSZ1_RSZ2

#### Relational analysis RSZ result of RS_RSZ2_RSZ2_RSZ2_RSZ1_RSZ1_RSZ1_RSZ2
Status: Status.UNKNOWN
Output dim: 9, lower bound: -0.0050102, upper bound: 0.0050645
time: 0.92 seconds

## BFS RS instance: RS_RSZ2_RSZ2_RSZ2_RSZ1_RSZ1_RSZ2

### Backsubstitution after applying RS history:
0: -0.0039095, -0.0003029, -0.0039095, -0.0003029, -0.0031761, 0.0031716
1: -0.0044907, 0.0040812, -0.0044907, 0.0040812, -0.0039466, 0.0039283
2: 0.0029569, 0.0099431, 0.0029569, 0.0099431, -0.0038999, 0.0038821
3: -0.0045519, -0.0034923, -0.0045519, -0.0034923, -0.0010595, 0.0010595
4: 0.0017018, 0.0078700, 0.0017018, 0.0078700, -0.0052918, 0.0052876
5: -0.0030341, 0.0031953, -0.0030341, 0.0031953, -0.0047625, 0.0047506
6: -0.0068516, -0.0033211, -0.0068516, -0.0033211, -0.0033539, 0.0033530
7: -0.0021570, 0.0038259, -0.0021570, 0.0038259, -0.0057752, 0.0057741
8: -0.0008533, 0.0000782, -0.0008533, 0.0000782, -0.0008663, 0.0008518
9: 0.9973557, 1.0121032, 0.9973557, 1.0121032, -0.0091543, 0.0091584

### Unstable ReLU Count (Linear/Conv2D Layers)
- layer_idx=0, type=LayerType.Linear, total=256, inp1_unstable=7, inp2_unstable=7, delta_unstable=250
- layer_idx=2, type=LayerType.Linear, total=256, inp1_unstable=10, inp2_unstable=10, delta_unstable=256
- layer_idx=4, type=LayerType.Linear, total=256, inp1_unstable=1, inp2_unstable=1, delta_unstable=256
- layer_idx=6, type=LayerType.Linear, total=256, inp1_unstable=0, inp2_unstable=0, delta_unstable=256
- layer_idx=8, type=LayerType.Linear, total=10, inp1_unstable=4, inp2_unstable=4, delta_unstable=10

Time for backsubstitution: 1.38 seconds

### RS candidates at layer 1
type: RSZ, layer: 1, pos: 219

Time for candidate selection: 0.14 seconds

### Candidate
type: RSZ, layer: 1, pos: 219

### Relational analysis RSZ of RS_RSZ2_RSZ2_RSZ2_RSZ1_RSZ1_RSZ2_RSZ1

#### Relational analysis RSZ result of RS_RSZ2_RSZ2_RSZ2_RSZ1_RSZ1_RSZ2_RSZ1
Status: Status.UNKNOWN
Output dim: 9, lower bound: -0.0052068, upper bound: 0.0050176
time: 0.79 seconds

### Relational analysis RSZ of RS_RSZ2_RSZ2_RSZ2_RSZ1_RSZ1_RSZ2_RSZ2

#### Relational analysis RSZ result of RS_RSZ2_RSZ2_RSZ2_RSZ1_RSZ1_RSZ2_RSZ2
Status: Status.UNKNOWN
Output dim: 9, lower bound: -0.0050105, upper bound: 0.0050645
time: 0.96 seconds

## BFS RS instance: RS_RSZ2_RSZ2_RSZ2_RSZ1_RSZ2_RSZ1

### Backsubstitution after applying RS history:
0: -0.0039095, -0.0003029, -0.0039095, -0.0003029, -0.0032320, 0.0031242
1: -0.0044907, 0.0040812, -0.0044907, 0.0040812, -0.0038202, 0.0040540
2: 0.0029569, 0.0099431, 0.0029569, 0.0099431, -0.0038120, 0.0039660
3: -0.0045519, -0.0034923, -0.0045519, -0.0034923, -0.0010595, 0.0010595
4: 0.0017018, 0.0078700, 0.0017018, 0.0078700, -0.0053343, 0.0052512
5: -0.0030341, 0.0031953, -0.0030341, 0.0031953, -0.0047032, 0.0048146
6: -0.0068516, -0.0033211, -0.0068516, -0.0033211, -0.0033431, 0.0033642
7: -0.0021570, 0.0038259, -0.0021570, 0.0038259, -0.0058033, 0.0057451
8: -0.0008533, 0.0000782, -0.0008533, 0.0000782, -0.0008638, 0.0008592
9: 0.9973557, 1.0121032, 0.9973557, 1.0121032, -0.0093255, 0.0089974

### Unstable ReLU Count (Linear/Conv2D Layers)
- layer_idx=0, type=LayerType.Linear, total=256, inp1_unstable=7, inp2_unstable=7, delta_unstable=250
- layer_idx=2, type=LayerType.Linear, total=256, inp1_unstable=10, inp2_unstable=10, delta_unstable=256
- layer_idx=4, type=LayerType.Linear, total=256, inp1_unstable=1, inp2_unstable=1, delta_unstable=256
- layer_idx=6, type=LayerType.Linear, total=256, inp1_unstable=0, inp2_unstable=0, delta_unstable=256
- layer_idx=8, type=LayerType.Linear, total=10, inp1_unstable=4, inp2_unstable=4, delta_unstable=10

Time for backsubstitution: 1.46 seconds

### RS candidates at layer 1
type: RSZ, layer: 1, pos: 219

Time for candidate selection: 0.15 seconds

### Candidate
type: RSZ, layer: 1, pos: 219

### Relational analysis RSZ of RS_RSZ2_RSZ2_RSZ2_RSZ1_RSZ2_RSZ1_RSZ1

#### Relational analysis RSZ result of RS_RSZ2_RSZ2_RSZ2_RSZ1_RSZ2_RSZ1_RSZ1
Status: Status.UNKNOWN
Output dim: 9, lower bound: -0.0053453, upper bound: 0.0048916
time: 0.83 seconds

### Relational analysis RSZ of RS_RSZ2_RSZ2_RSZ2_RSZ1_RSZ2_RSZ1_RSZ2

#### Relational analysis RSZ result of RS_RSZ2_RSZ2_RSZ2_RSZ1_RSZ2_RSZ1_RSZ2
Status: Status.UNKNOWN
Output dim: 9, lower bound: -0.0051584, upper bound: 0.0049733
time: 0.83 seconds

## BFS RS instance: RS_RSZ2_RSZ2_RSZ2_RSZ1_RSZ2_RSZ2

### Backsubstitution after applying RS history:
0: -0.0039095, -0.0003029, -0.0039095, -0.0003029, -0.0032317, 0.0031248
1: -0.0044907, 0.0040812, -0.0044907, 0.0040812, -0.0038204, 0.0040541
2: 0.0029569, 0.0099431, 0.0029569, 0.0099431, -0.0038126, 0.0039664
3: -0.0045519, -0.0034923, -0.0045519, -0.0034923, -0.0010595, 0.0010595
4: 0.0017018, 0.0078700, 0.0017018, 0.0078700, -0.0053342, 0.0052516
5: -0.0030341, 0.0031953, -0.0030341, 0.0031953, -0.0047048, 0.0048141
6: -0.0068516, -0.0033211, -0.0068516, -0.0033211, -0.0033432, 0.0033642
7: -0.0021570, 0.0038259, -0.0021570, 0.0038259, -0.0058026, 0.0057442
8: -0.0008533, 0.0000782, -0.0008533, 0.0000782, -0.0008628, 0.0008591
9: 0.9973557, 1.0121032, 0.9973557, 1.0121032, -0.0093256, 0.0089980

### Unstable ReLU Count (Linear/Conv2D Layers)
- layer_idx=0, type=LayerType.Linear, total=256, inp1_unstable=7, inp2_unstable=7, delta_unstable=250
- layer_idx=2, type=LayerType.Linear, total=256, inp1_unstable=10, inp2_unstable=10, delta_unstable=256
- layer_idx=4, type=LayerType.Linear, total=256, inp1_unstable=1, inp2_unstable=1, delta_unstable=256
- layer_idx=6, type=LayerType.Linear, total=256, inp1_unstable=0, inp2_unstable=0, delta_unstable=256
- layer_idx=8, type=LayerType.Linear, total=10, inp1_unstable=4, inp2_unstable=4, delta_unstable=10

Time for backsubstitution: 1.36 seconds

### RS candidates at layer 1
type: RSZ, layer: 1, pos: 219

Time for candidate selection: 0.14 seconds

### Candidate
type: RSZ, layer: 1, pos: 219

### Relational analysis RSZ of RS_RSZ2_RSZ2_RSZ2_RSZ1_RSZ2_RSZ2_RSZ1

#### Relational analysis RSZ result of RS_RSZ2_RSZ2_RSZ2_RSZ1_RSZ2_RSZ2_RSZ1
Status: Status.UNKNOWN
Output dim: 9, lower bound: -0.0053453, upper bound: 0.0048916
time: 0.82 seconds

### Relational analysis RSZ of RS_RSZ2_RSZ2_RSZ2_RSZ1_RSZ2_RSZ2_RSZ2

#### Relational analysis RSZ result of RS_RSZ2_RSZ2_RSZ2_RSZ1_RSZ2_RSZ2_RSZ2
Status: Status.UNKNOWN
Output dim: 9, lower bound: -0.0051592, upper bound: 0.0049733
time: 0.82 seconds

## BFS RS instance: RS_RSZ2_RSZ2_RSZ2_RSZ2_RSZ1_RSZ1

### Backsubstitution after applying RS history:
0: -0.0039095, -0.0003029, -0.0039095, -0.0003029, -0.0031053, 0.0032404
1: -0.0044907, 0.0040812, -0.0044907, 0.0040812, -0.0040991, 0.0037524
2: 0.0029569, 0.0099431, 0.0029569, 0.0099431, -0.0039978, 0.0037666
3: -0.0045519, -0.0034923, -0.0045519, -0.0034923, -0.0010595, 0.0010595
4: 0.0017018, 0.0078700, 0.0017018, 0.0078700, -0.0052373, 0.0053397
5: -0.0030341, 0.0031953, -0.0030341, 0.0031953, -0.0048386, 0.0046760
6: -0.0068516, -0.0033211, -0.0068516, -0.0033211, -0.0033677, 0.0033375
7: -0.0021570, 0.0038259, -0.0021570, 0.0038259, -0.0057500, 0.0057974
8: -0.0008533, 0.0000782, -0.0008533, 0.0000782, -0.0008786, 0.0008423
9: 0.9973557, 1.0121032, 0.9973557, 1.0121032, -0.0089183, 0.0093718

### Unstable ReLU Count (Linear/Conv2D Layers)
- layer_idx=0, type=LayerType.Linear, total=256, inp1_unstable=7, inp2_unstable=7, delta_unstable=250
- layer_idx=2, type=LayerType.Linear, total=256, inp1_unstable=10, inp2_unstable=10, delta_unstable=256
- layer_idx=4, type=LayerType.Linear, total=256, inp1_unstable=1, inp2_unstable=1, delta_unstable=256
- layer_idx=6, type=LayerType.Linear, total=256, inp1_unstable=0, inp2_unstable=0, delta_unstable=256
- layer_idx=8, type=LayerType.Linear, total=10, inp1_unstable=4, inp2_unstable=4, delta_unstable=10

Time for backsubstitution: 1.38 seconds

### RS candidates at layer 1
type: RSZ, layer: 1, pos: 219

Time for candidate selection: 0.16 seconds

### Candidate
type: RSZ, layer: 1, pos: 219

### Relational analysis RSZ of RS_RSZ2_RSZ2_RSZ2_RSZ2_RSZ1_RSZ1_RSZ1

#### Relational analysis RSZ result of RS_RSZ2_RSZ2_RSZ2_RSZ2_RSZ1_RSZ1_RSZ1
Status: Status.UNKNOWN
Output dim: 9, lower bound: -0.0047324, upper bound: 0.0053154
time: 0.78 seconds

### Relational analysis RSZ of RS_RSZ2_RSZ2_RSZ2_RSZ2_RSZ1_RSZ1_RSZ2

#### Relational analysis RSZ result of RS_RSZ2_RSZ2_RSZ2_RSZ2_RSZ1_RSZ1_RSZ2
Status: Status.UNKNOWN
Output dim: 9, lower bound: -0.0046830, upper bound: 0.0054877
time: 0.84 seconds

## BFS RS instance: RS_RSZ2_RSZ2_RSZ2_RSZ2_RSZ1_RSZ2

### Backsubstitution after applying RS history:
0: -0.0039095, -0.0003029, -0.0039095, -0.0003029, -0.0031045, 0.0032404
1: -0.0044907, 0.0040812, -0.0044907, 0.0040812, -0.0040990, 0.0037520
2: 0.0029569, 0.0099431, 0.0029569, 0.0099431, -0.0039979, 0.0037662
3: -0.0045519, -0.0034923, -0.0045519, -0.0034923, -0.0010595, 0.0010595
4: 0.0017018, 0.0078700, 0.0017018, 0.0078700, -0.0052368, 0.0053397
5: -0.0030341, 0.0031953, -0.0030341, 0.0031953, -0.0048383, 0.0046729
6: -0.0068516, -0.0033211, -0.0068516, -0.0033211, -0.0033677, 0.0033375
7: -0.0021570, 0.0038259, -0.0021570, 0.0038259, -0.0057493, 0.0057962
8: -0.0008533, 0.0000782, -0.0008533, 0.0000782, -0.0008767, 0.0008413
9: 0.9973557, 1.0121032, 0.9973557, 1.0121032, -0.0089173, 0.0093716

### Unstable ReLU Count (Linear/Conv2D Layers)
- layer_idx=0, type=LayerType.Linear, total=256, inp1_unstable=7, inp2_unstable=7, delta_unstable=250
- layer_idx=2, type=LayerType.Linear, total=256, inp1_unstable=10, inp2_unstable=10, delta_unstable=256
- layer_idx=4, type=LayerType.Linear, total=256, inp1_unstable=1, inp2_unstable=1, delta_unstable=256
- layer_idx=6, type=LayerType.Linear, total=256, inp1_unstable=0, inp2_unstable=0, delta_unstable=256
- layer_idx=8, type=LayerType.Linear, total=10, inp1_unstable=4, inp2_unstable=4, delta_unstable=10

Time for backsubstitution: 1.36 seconds

### RS candidates at layer 1
type: RSZ, layer: 1, pos: 219

Time for candidate selection: 0.14 seconds

### Candidate
type: RSZ, layer: 1, pos: 219

### Relational analysis RSZ of RS_RSZ2_RSZ2_RSZ2_RSZ2_RSZ1_RSZ2_RSZ1

#### Relational analysis RSZ result of RS_RSZ2_RSZ2_RSZ2_RSZ2_RSZ1_RSZ2_RSZ1
Status: Status.UNKNOWN
Output dim: 9, lower bound: -0.0047324, upper bound: 0.0053148
time: 0.79 seconds

### Relational analysis RSZ of RS_RSZ2_RSZ2_RSZ2_RSZ2_RSZ1_RSZ2_RSZ2

#### Relational analysis RSZ result of RS_RSZ2_RSZ2_RSZ2_RSZ2_RSZ1_RSZ2_RSZ2
Status: Status.UNKNOWN
Output dim: 9, lower bound: -0.0046830, upper bound: 0.0054877
time: 0.89 seconds

## BFS RS instance: RS_RSZ2_RSZ2_RSZ2_RSZ2_RSZ2_RSZ1

### Backsubstitution after applying RS history:
0: -0.0039095, -0.0003029, -0.0039095, -0.0003029, -0.0031620, 0.0031918
1: -0.0044907, 0.0040812, -0.0044907, 0.0040812, -0.0039784, 0.0038917
2: 0.0029569, 0.0099431, 0.0029569, 0.0099431, -0.0039144, 0.0038624
3: -0.0045519, -0.0034923, -0.0045519, -0.0034923, -0.0010595, 0.0010595
4: 0.0017018, 0.0078700, 0.0017018, 0.0078700, -0.0052806, 0.0053030
5: -0.0030341, 0.0031953, -0.0030341, 0.0031953, -0.0047797, 0.0047359
6: -0.0068516, -0.0033211, -0.0068516, -0.0033211, -0.0033573, 0.0033495
7: -0.0021570, 0.0038259, -0.0021570, 0.0038259, -0.0057790, 0.0057678
8: -0.0008533, 0.0000782, -0.0008533, 0.0000782, -0.0008720, 0.0008493
9: 0.9973557, 1.0121032, 0.9973557, 1.0121032, -0.0091006, 0.0092149

### Unstable ReLU Count (Linear/Conv2D Layers)
- layer_idx=0, type=LayerType.Linear, total=256, inp1_unstable=7, inp2_unstable=7, delta_unstable=250
- layer_idx=2, type=LayerType.Linear, total=256, inp1_unstable=10, inp2_unstable=10, delta_unstable=256
- layer_idx=4, type=LayerType.Linear, total=256, inp1_unstable=1, inp2_unstable=1, delta_unstable=256
- layer_idx=6, type=LayerType.Linear, total=256, inp1_unstable=0, inp2_unstable=0, delta_unstable=256
- layer_idx=8, type=LayerType.Linear, total=10, inp1_unstable=4, inp2_unstable=4, delta_unstable=10

Time for backsubstitution: 1.56 seconds

### RS candidates at layer 1
type: RSZ, layer: 1, pos: 219

Time for candidate selection: 0.18 seconds

### Candidate
type: RSZ, layer: 1, pos: 219

### Relational analysis RSZ of RS_RSZ2_RSZ2_RSZ2_RSZ2_RSZ2_RSZ1_RSZ1

#### Relational analysis RSZ result of RS_RSZ2_RSZ2_RSZ2_RSZ2_RSZ2_RSZ1_RSZ1
Status: Status.UNKNOWN
Output dim: 9, lower bound: -0.0048608, upper bound: 0.0051431
time: 0.88 seconds

### Relational analysis RSZ of RS_RSZ2_RSZ2_RSZ2_RSZ2_RSZ2_RSZ1_RSZ2

#### Relational analysis RSZ result of RS_RSZ2_RSZ2_RSZ2_RSZ2_RSZ2_RSZ1_RSZ2
Status: Status.UNKNOWN
Output dim: 9, lower bound: -0.0048081, upper bound: 0.0053505
time: 0.94 seconds

## BFS RS instance: RS_RSZ2_RSZ2_RSZ2_RSZ2_RSZ2_RSZ2

### Backsubstitution after applying RS history:
0: -0.0039095, -0.0003029, -0.0039095, -0.0003029, -0.0031614, 0.0031922
1: -0.0044907, 0.0040812, -0.0044907, 0.0040812, -0.0039786, 0.0038918
2: 0.0029569, 0.0099431, 0.0029569, 0.0099431, -0.0039147, 0.0038623
3: -0.0045519, -0.0034923, -0.0045519, -0.0034923, -0.0010595, 0.0010595
4: 0.0017018, 0.0078700, 0.0017018, 0.0078700, -0.0052804, 0.0053033
5: -0.0030341, 0.0031953, -0.0030341, 0.0031953, -0.0047797, 0.0047346
6: -0.0068516, -0.0033211, -0.0068516, -0.0033211, -0.0033573, 0.0033495
7: -0.0021570, 0.0038259, -0.0021570, 0.0038259, -0.0057782, 0.0057675
8: -0.0008533, 0.0000782, -0.0008533, 0.0000782, -0.0008702, 0.0008482
9: 0.9973557, 1.0121032, 0.9973557, 1.0121032, -0.0091005, 0.0092151

### Unstable ReLU Count (Linear/Conv2D Layers)
- layer_idx=0, type=LayerType.Linear, total=256, inp1_unstable=7, inp2_unstable=7, delta_unstable=250
- layer_idx=2, type=LayerType.Linear, total=256, inp1_unstable=10, inp2_unstable=10, delta_unstable=256
- layer_idx=4, type=LayerType.Linear, total=256, inp1_unstable=1, inp2_unstable=1, delta_unstable=256
- layer_idx=6, type=LayerType.Linear, total=256, inp1_unstable=0, inp2_unstable=0, delta_unstable=256
- layer_idx=8, type=LayerType.Linear, total=10, inp1_unstable=4, inp2_unstable=4, delta_unstable=10

Time for backsubstitution: 1.54 seconds

### RS candidates at layer 1
type: RSZ, layer: 1, pos: 219

Time for candidate selection: 0.17 seconds

### Candidate
type: RSZ, layer: 1, pos: 219

### Relational analysis RSZ of RS_RSZ2_RSZ2_RSZ2_RSZ2_RSZ2_RSZ2_RSZ1

#### Relational analysis RSZ result of RS_RSZ2_RSZ2_RSZ2_RSZ2_RSZ2_RSZ2_RSZ1
Status: Status.UNKNOWN
Output dim: 9, lower bound: -0.0048608, upper bound: 0.0051431
time: 0.83 seconds

### Relational analysis RSZ of RS_RSZ2_RSZ2_RSZ2_RSZ2_RSZ2_RSZ2_RSZ2

#### Relational analysis RSZ result of RS_RSZ2_RSZ2_RSZ2_RSZ2_RSZ2_RSZ2_RSZ2
Status: Status.UNKNOWN
Output dim: 9, lower bound: -0.0048081, upper bound: 0.0053505
time: 0.86 seconds

## Summary of splitting (split count: 6)
- Time for RS candidates: 3.42 seconds
RS_RSZ1_RSZ1_RSZ1_RSZ1_RSZ1_RSZ1_RSZ1, status: Status.UNKNOWN, split count: 7, time: 3.42
Output dim: 9, lower bound: -0.0053505, upper bound: 0.0048081
RS_RSZ1_RSZ1_RSZ1_RSZ1_RSZ1_RSZ1_RSZ2, status: Status.UNKNOWN, split count: 7, time: 3.42
Output dim: 9, lower bound: -0.0051431, upper bound: 0.0048608
RS_RSZ1_RSZ1_RSZ1_RSZ1_RSZ1_RSZ2_RSZ1, status: Status.UNKNOWN, split count: 7, time: 3.42
Output dim: 9, lower bound: -0.0053505, upper bound: 0.0048081
RS_RSZ1_RSZ1_RSZ1_RSZ1_RSZ1_RSZ2_RSZ2, status: Status.UNKNOWN, split count: 7, time: 3.42
Output dim: 9, lower bound: -0.0051431, upper bound: 0.0048608
RS_RSZ1_RSZ1_RSZ1_RSZ1_RSZ2_RSZ1_RSZ1, status: Status.UNKNOWN, split count: 7, time: 3.42
Output dim: 9, lower bound: -0.0054877, upper bound: 0.0046830
RS_RSZ1_RSZ1_RSZ1_RSZ1_RSZ2_RSZ1_RSZ2, status: Status.UNKNOWN, split count: 7, time: 3.42
Output dim: 9, lower bound: -0.0053148, upper bound: 0.0047324
RS_RSZ1_RSZ1_RSZ1_RSZ1_RSZ2_RSZ2_RSZ1, status: Status.UNKNOWN, split count: 7, time: 3.42
Output dim: 9, lower bound: -0.0054877, upper bound: 0.0046830
RS_RSZ1_RSZ1_RSZ1_RSZ1_RSZ2_RSZ2_RSZ2, status: Status.UNKNOWN, split count: 7, time: 3.42
Output dim: 9, lower bound: -0.0053154, upper bound: 0.0047324
RS_RSZ1_RSZ1_RSZ1_RSZ2_RSZ1_RSZ1_RSZ1, status: Status.UNKNOWN, split count: 7, time: 3.42
Output dim: 9, lower bound: -0.0049733, upper bound: 0.0051592
RS_RSZ1_RSZ1_RSZ1_RSZ2_RSZ1_RSZ1_RSZ2, status: Status.UNKNOWN, split count: 7, time: 3.42
Output dim: 9, lower bound: -0.0048917, upper bound: 0.0053453
RS_RSZ1_RSZ1_RSZ1_RSZ2_RSZ1_RSZ2_RSZ1, status: Status.UNKNOWN, split count: 7, time: 3.42
Output dim: 9, lower bound: -0.0049733, upper bound: 0.0051584
RS_RSZ1_RSZ1_RSZ1_RSZ2_RSZ1_RSZ2_RSZ2, status: Status.UNKNOWN, split count: 7, time: 3.42
Output dim: 9, lower bound: -0.0048917, upper bound: 0.0053453
RS_RSZ1_RSZ1_RSZ1_RSZ2_RSZ2_RSZ1_RSZ1, status: Status.UNKNOWN, split count: 7, time: 3.42
Output dim: 9, lower bound: -0.0050645, upper bound: 0.0050105
RS_RSZ1_RSZ1_RSZ1_RSZ2_RSZ2_RSZ1_RSZ2, status: Status.UNKNOWN, split count: 7, time: 3.42
Output dim: 9, lower bound: -0.0050176, upper bound: 0.0052068
RS_RSZ1_RSZ1_RSZ1_RSZ2_RSZ2_RSZ2_RSZ1, status: Status.UNKNOWN, split count: 7, time: 3.42
Output dim: 9, lower bound: -0.0050645, upper bound: 0.0050102
RS_RSZ1_RSZ1_RSZ1_RSZ2_RSZ2_RSZ2_RSZ2, status: Status.UNKNOWN, split count: 7, time: 3.42
Output dim: 9, lower bound: -0.0050176, upper bound: 0.0052068
RS_RSZ1_RSZ1_RSZ2_RSZ1_RSZ1_RSZ1_RSZ1, status: Status.UNKNOWN, split count: 7, time: 3.42
Output dim: 9, lower bound: -0.0053174, upper bound: 0.0048282
RS_RSZ1_RSZ1_RSZ2_RSZ1_RSZ1_RSZ1_RSZ2, status: Status.UNKNOWN, split count: 7, time: 3.42
Output dim: 9, lower bound: -0.0051236, upper bound: 0.0048890
RS_RSZ1_RSZ1_RSZ2_RSZ1_RSZ1_RSZ2_RSZ1, status: Status.UNKNOWN, split count: 7, time: 3.42
Output dim: 9, lower bound: -0.0053174, upper bound: 0.0048282
RS_RSZ1_RSZ1_RSZ2_RSZ1_RSZ1_RSZ2_RSZ2, status: Status.UNKNOWN, split count: 7, time: 3.42
Output dim: 9, lower bound: -0.0051239, upper bound: 0.0048890
RS_RSZ1_RSZ1_RSZ2_RSZ1_RSZ2_RSZ1_RSZ1, status: Status.UNKNOWN, split count: 7, time: 3.42
Output dim: 9, lower bound: -0.0054782, upper bound: 0.0047392
RS_RSZ1_RSZ1_RSZ2_RSZ1_RSZ2_RSZ1_RSZ2, status: Status.UNKNOWN, split count: 7, time: 3.42
Output dim: 9, lower bound: -0.0053094, upper bound: 0.0047969
RS_RSZ1_RSZ1_RSZ2_RSZ1_RSZ2_RSZ2_RSZ1, status: Status.UNKNOWN, split count: 7, time: 3.42
Output dim: 9, lower bound: -0.0054783, upper bound: 0.0047392
RS_RSZ1_RSZ1_RSZ2_RSZ1_RSZ2_RSZ2_RSZ2, status: Status.UNKNOWN, split count: 7, time: 3.42
Output dim: 9, lower bound: -0.0053095, upper bound: 0.0047969
RS_RSZ1_RSZ1_RSZ2_RSZ2_RSZ1_RSZ1_RSZ1, status: Status.UNKNOWN, split count: 7, time: 3.42
Output dim: 9, lower bound: -0.0049024, upper bound: 0.0051598
RS_RSZ1_RSZ1_RSZ2_RSZ2_RSZ1_RSZ1_RSZ2, status: Status.UNKNOWN, split count: 7, time: 3.42
Output dim: 9, lower bound: -0.0048358, upper bound: 0.0053493
RS_RSZ1_RSZ1_RSZ2_RSZ2_RSZ1_RSZ2_RSZ1, status: Status.UNKNOWN, split count: 7, time: 3.42
Output dim: 9, lower bound: -0.0049024, upper bound: 0.0051587
RS_RSZ1_RSZ1_RSZ2_RSZ2_RSZ1_RSZ2_RSZ2, status: Status.UNKNOWN, split count: 7, time: 3.42
Output dim: 9, lower bound: -0.0048358, upper bound: 0.0053493
RS_RSZ1_RSZ1_RSZ2_RSZ2_RSZ2_RSZ1_RSZ1, status: Status.VERIFIED, split count: 7, time: 3.42
Output dim: 9, lower bound: -0.0050255, upper bound: 0.0050265
RS_RSZ1_RSZ1_RSZ2_RSZ2_RSZ2_RSZ1_RSZ2, status: Status.UNKNOWN, split count: 7, time: 3.42
Output dim: 9, lower bound: -0.0049917, upper bound: 0.0052257
RS_RSZ1_RSZ1_RSZ2_RSZ2_RSZ2_RSZ2_RSZ1, status: Status.VERIFIED, split count: 7, time: 3.42
Output dim: 9, lower bound: -0.0050255, upper bound: 0.0050264
RS_RSZ1_RSZ1_RSZ2_RSZ2_RSZ2_RSZ2_RSZ2, status: Status.UNKNOWN, split count: 7, time: 3.42
Output dim: 9, lower bound: -0.0049917, upper bound: 0.0052257
RS_RSZ1_RSZ2_RSZ1_RSZ1_RSZ1_RSZ1_RSZ1, status: Status.UNKNOWN, split count: 7, time: 3.42
Output dim: 9, lower bound: -0.0052773, upper bound: 0.0049534
RS_RSZ1_RSZ2_RSZ1_RSZ1_RSZ1_RSZ1_RSZ2, status: Status.VERIFIED, split count: 7, time: 3.42
Output dim: 9, lower bound: -0.0050522, upper bound: 0.0049978
RS_RSZ1_RSZ2_RSZ1_RSZ1_RSZ1_RSZ2_RSZ1, status: Status.UNKNOWN, split count: 7, time: 3.42
Output dim: 9, lower bound: -0.0052773, upper bound: 0.0049534
RS_RSZ1_RSZ2_RSZ1_RSZ1_RSZ1_RSZ2_RSZ2, status: Status.VERIFIED, split count: 7, time: 3.42
Output dim: 9, lower bound: -0.0050529, upper bound: 0.0049978
RS_RSZ1_RSZ2_RSZ1_RSZ1_RSZ2_RSZ1_RSZ1, status: Status.UNKNOWN, split count: 7, time: 3.42
Output dim: 9, lower bound: -0.0053802, upper bound: 0.0047963
RS_RSZ1_RSZ2_RSZ1_RSZ1_RSZ2_RSZ1_RSZ2, status: Status.UNKNOWN, split count: 7, time: 3.42
Output dim: 9, lower bound: -0.0051935, upper bound: 0.0048579
RS_RSZ1_RSZ2_RSZ1_RSZ1_RSZ2_RSZ2_RSZ1, status: Status.UNKNOWN, split count: 7, time: 3.42
Output dim: 9, lower bound: -0.0053802, upper bound: 0.0047963
RS_RSZ1_RSZ2_RSZ1_RSZ1_RSZ2_RSZ2_RSZ2, status: Status.UNKNOWN, split count: 7, time: 3.42
Output dim: 9, lower bound: -0.0051947, upper bound: 0.0048588
RS_RSZ1_RSZ2_RSZ1_RSZ2_RSZ1_RSZ1_RSZ1, status: Status.UNKNOWN, split count: 7, time: 3.42
Output dim: 9, lower bound: -0.0048628, upper bound: 0.0052874
RS_RSZ1_RSZ2_RSZ1_RSZ2_RSZ1_RSZ1_RSZ2, status: Status.UNKNOWN, split count: 7, time: 3.42
Output dim: 9, lower bound: -0.0047878, upper bound: 0.0054625
RS_RSZ1_RSZ2_RSZ1_RSZ2_RSZ1_RSZ2_RSZ1, status: Status.UNKNOWN, split count: 7, time: 3.42
Output dim: 9, lower bound: -0.0048628, upper bound: 0.0052874
RS_RSZ1_RSZ2_RSZ1_RSZ2_RSZ1_RSZ2_RSZ2, status: Status.UNKNOWN, split count: 7, time: 3.42
Output dim: 9, lower bound: -0.0047878, upper bound: 0.0054625
RS_RSZ1_RSZ2_RSZ1_RSZ2_RSZ2_RSZ1_RSZ1, status: Status.UNKNOWN, split count: 7, time: 3.42
Output dim: 9, lower bound: -0.0049568, upper bound: 0.0050997
RS_RSZ1_RSZ2_RSZ1_RSZ2_RSZ2_RSZ1_RSZ2, status: Status.UNKNOWN, split count: 7, time: 3.42
Output dim: 9, lower bound: -0.0048928, upper bound: 0.0052867
RS_RSZ1_RSZ2_RSZ1_RSZ2_RSZ2_RSZ2_RSZ1, status: Status.UNKNOWN, split count: 7, time: 3.42
Output dim: 9, lower bound: -0.0049568, upper bound: 0.0050997
RS_RSZ1_RSZ2_RSZ1_RSZ2_RSZ2_RSZ2_RSZ2, status: Status.UNKNOWN, split count: 7, time: 3.42
Output dim: 9, lower bound: -0.0048928, upper bound: 0.0052867
RS_RSZ1_RSZ2_RSZ2_RSZ1_RSZ1_RSZ1_RSZ1, status: Status.UNKNOWN, split count: 7, time: 3.42
Output dim: 9, lower bound: -0.0052573, upper bound: 0.0049775
RS_RSZ1_RSZ2_RSZ2_RSZ1_RSZ1_RSZ1_RSZ2, status: Status.VERIFIED, split count: 7, time: 3.42
Output dim: 9, lower bound: -0.0050422, upper bound: 0.0050256
RS_RSZ1_RSZ2_RSZ2_RSZ1_RSZ1_RSZ2_RSZ1, status: Status.UNKNOWN, split count: 7, time: 3.42
Output dim: 9, lower bound: -0.0052573, upper bound: 0.0049775
RS_RSZ1_RSZ2_RSZ2_RSZ1_RSZ1_RSZ2_RSZ2, status: Status.VERIFIED, split count: 7, time: 3.42
Output dim: 9, lower bound: -0.0050426, upper bound: 0.0050256
RS_RSZ1_RSZ2_RSZ2_RSZ1_RSZ2_RSZ1_RSZ1, status: Status.UNKNOWN, split count: 7, time: 3.42
Output dim: 9, lower bound: -0.0053776, upper bound: 0.0048599
RS_RSZ1_RSZ2_RSZ2_RSZ1_RSZ2_RSZ1_RSZ2, status: Status.UNKNOWN, split count: 7, time: 3.42
Output dim: 9, lower bound: -0.0051929, upper bound: 0.0049236
RS_RSZ1_RSZ2_RSZ2_RSZ1_RSZ2_RSZ2_RSZ1, status: Status.UNKNOWN, split count: 7, time: 3.42
Output dim: 9, lower bound: -0.0053776, upper bound: 0.0048599
RS_RSZ1_RSZ2_RSZ2_RSZ1_RSZ2_RSZ2_RSZ2, status: Status.UNKNOWN, split count: 7, time: 3.42
Output dim: 9, lower bound: -0.0051936, upper bound: 0.0049236
RS_RSZ1_RSZ2_RSZ2_RSZ2_RSZ1_RSZ1_RSZ1, status: Status.UNKNOWN, split count: 7, time: 3.42
Output dim: 9, lower bound: -0.0047958, upper bound: 0.0052953
RS_RSZ1_RSZ2_RSZ2_RSZ2_RSZ1_RSZ1_RSZ2, status: Status.UNKNOWN, split count: 7, time: 3.42
Output dim: 9, lower bound: -0.0047313, upper bound: 0.0054735
RS_RSZ1_RSZ2_RSZ2_RSZ2_RSZ1_RSZ2_RSZ1, status: Status.UNKNOWN, split count: 7, time: 3.42
Output dim: 9, lower bound: -0.0047958, upper bound: 0.0052950
RS_RSZ1_RSZ2_RSZ2_RSZ2_RSZ1_RSZ2_RSZ2, status: Status.UNKNOWN, split count: 7, time: 3.42
Output dim: 9, lower bound: -0.0047313, upper bound: 0.0054735
RS_RSZ1_RSZ2_RSZ2_RSZ2_RSZ2_RSZ1_RSZ1, status: Status.UNKNOWN, split count: 7, time: 3.42
Output dim: 9, lower bound: -0.0049073, upper bound: 0.0051222
RS_RSZ1_RSZ2_RSZ2_RSZ2_RSZ2_RSZ1_RSZ2, status: Status.UNKNOWN, split count: 7, time: 3.42
Output dim: 9, lower bound: -0.0048586, upper bound: 0.0053248
RS_RSZ1_RSZ2_RSZ2_RSZ2_RSZ2_RSZ2_RSZ1, status: Status.UNKNOWN, split count: 7, time: 3.42
Output dim: 9, lower bound: -0.0049073, upper bound: 0.0051221
RS_RSZ1_RSZ2_RSZ2_RSZ2_RSZ2_RSZ2_RSZ2, status: Status.UNKNOWN, split count: 7, time: 3.42
Output dim: 9, lower bound: -0.0048586, upper bound: 0.0053248
RS_RSZ2_RSZ1_RSZ1_RSZ1_RSZ1_RSZ1_RSZ1, status: Status.UNKNOWN, split count: 7, time: 3.42
Output dim: 9, lower bound: -0.0053248, upper bound: 0.0048586
RS_RSZ2_RSZ1_RSZ1_RSZ1_RSZ1_RSZ1_RSZ2, status: Status.UNKNOWN, split count: 7, time: 3.42
Output dim: 9, lower bound: -0.0051221, upper bound: 0.0049073
RS_RSZ2_RSZ1_RSZ1_RSZ1_RSZ1_RSZ2_RSZ1, status: Status.UNKNOWN, split count: 7, time: 3.42
Output dim: 9, lower bound: -0.0053248, upper bound: 0.0048586
RS_RSZ2_RSZ1_RSZ1_RSZ1_RSZ1_RSZ2_RSZ2, status: Status.UNKNOWN, split count: 7, time: 3.42
Output dim: 9, lower bound: -0.0051222, upper bound: 0.0049073
RS_RSZ2_RSZ1_RSZ1_RSZ1_RSZ2_RSZ1_RSZ1, status: Status.UNKNOWN, split count: 7, time: 3.42
Output dim: 9, lower bound: -0.0054735, upper bound: 0.0047313
RS_RSZ2_RSZ1_RSZ1_RSZ1_RSZ2_RSZ1_RSZ2, status: Status.UNKNOWN, split count: 7, time: 3.42
Output dim: 9, lower bound: -0.0052949, upper bound: 0.0047958
RS_RSZ2_RSZ1_RSZ1_RSZ1_RSZ2_RSZ2_RSZ1, status: Status.UNKNOWN, split count: 7, time: 3.42
Output dim: 9, lower bound: -0.0054735, upper bound: 0.0047313
RS_RSZ2_RSZ1_RSZ1_RSZ1_RSZ2_RSZ2_RSZ2, status: Status.UNKNOWN, split count: 7, time: 3.42
Output dim: 9, lower bound: -0.0052953, upper bound: 0.0047958
RS_RSZ2_RSZ1_RSZ1_RSZ2_RSZ1_RSZ1_RSZ1, status: Status.UNKNOWN, split count: 7, time: 3.42
Output dim: 9, lower bound: -0.0049236, upper bound: 0.0051936
RS_RSZ2_RSZ1_RSZ1_RSZ2_RSZ1_RSZ1_RSZ2, status: Status.UNKNOWN, split count: 7, time: 3.42
Output dim: 9, lower bound: -0.0048599, upper bound: 0.0053776
RS_RSZ2_RSZ1_RSZ1_RSZ2_RSZ1_RSZ2_RSZ1, status: Status.UNKNOWN, split count: 7, time: 3.42
Output dim: 9, lower bound: -0.0049236, upper bound: 0.0051929
RS_RSZ2_RSZ1_RSZ1_RSZ2_RSZ1_RSZ2_RSZ2, status: Status.UNKNOWN, split count: 7, time: 3.42
Output dim: 9, lower bound: -0.0048599, upper bound: 0.0053776
RS_RSZ2_RSZ1_RSZ1_RSZ2_RSZ2_RSZ1_RSZ1, status: Status.VERIFIED, split count: 7, time: 3.42
Output dim: 9, lower bound: -0.0050256, upper bound: 0.0050426
RS_RSZ2_RSZ1_RSZ1_RSZ2_RSZ2_RSZ1_RSZ2, status: Status.UNKNOWN, split count: 7, time: 3.42
Output dim: 9, lower bound: -0.0049775, upper bound: 0.0052573
RS_RSZ2_RSZ1_RSZ1_RSZ2_RSZ2_RSZ2_RSZ1, status: Status.VERIFIED, split count: 7, time: 3.42
Output dim: 9, lower bound: -0.0050256, upper bound: 0.0050422
RS_RSZ2_RSZ1_RSZ1_RSZ2_RSZ2_RSZ2_RSZ2, status: Status.UNKNOWN, split count: 7, time: 3.42
Output dim: 9, lower bound: -0.0049775, upper bound: 0.0052573
RS_RSZ2_RSZ1_RSZ2_RSZ1_RSZ1_RSZ1_RSZ1, status: Status.UNKNOWN, split count: 7, time: 3.42
Output dim: 9, lower bound: -0.0052867, upper bound: 0.0048928
RS_RSZ2_RSZ1_RSZ2_RSZ1_RSZ1_RSZ1_RSZ2, status: Status.UNKNOWN, split count: 7, time: 3.42
Output dim: 9, lower bound: -0.0050997, upper bound: 0.0049568
RS_RSZ2_RSZ1_RSZ2_RSZ1_RSZ1_RSZ2_RSZ1, status: Status.UNKNOWN, split count: 7, time: 3.42
Output dim: 9, lower bound: -0.0052867, upper bound: 0.0048928
RS_RSZ2_RSZ1_RSZ2_RSZ1_RSZ1_RSZ2_RSZ2, status: Status.UNKNOWN, split count: 7, time: 3.42
Output dim: 9, lower bound: -0.0050997, upper bound: 0.0049568
RS_RSZ2_RSZ1_RSZ2_RSZ1_RSZ2_RSZ1_RSZ1, status: Status.UNKNOWN, split count: 7, time: 3.42
Output dim: 9, lower bound: -0.0054625, upper bound: 0.0047878
RS_RSZ2_RSZ1_RSZ2_RSZ1_RSZ2_RSZ1_RSZ2, status: Status.UNKNOWN, split count: 7, time: 3.42
Output dim: 9, lower bound: -0.0052874, upper bound: 0.0048628
RS_RSZ2_RSZ1_RSZ2_RSZ1_RSZ2_RSZ2_RSZ1, status: Status.UNKNOWN, split count: 7, time: 3.42
Output dim: 9, lower bound: -0.0054625, upper bound: 0.0047878
RS_RSZ2_RSZ1_RSZ2_RSZ1_RSZ2_RSZ2_RSZ2, status: Status.UNKNOWN, split count: 7, time: 3.42
Output dim: 9, lower bound: -0.0052874, upper bound: 0.0048628
RS_RSZ2_RSZ1_RSZ2_RSZ2_RSZ1_RSZ1_RSZ1, status: Status.UNKNOWN, split count: 7, time: 3.42
Output dim: 9, lower bound: -0.0048588, upper bound: 0.0051947
RS_RSZ2_RSZ1_RSZ2_RSZ2_RSZ1_RSZ1_RSZ2, status: Status.UNKNOWN, split count: 7, time: 3.42
Output dim: 9, lower bound: -0.0047963, upper bound: 0.0053802
RS_RSZ2_RSZ1_RSZ2_RSZ2_RSZ1_RSZ2_RSZ1, status: Status.UNKNOWN, split count: 7, time: 3.42
Output dim: 9, lower bound: -0.0048579, upper bound: 0.0051935
RS_RSZ2_RSZ1_RSZ2_RSZ2_RSZ1_RSZ2_RSZ2, status: Status.UNKNOWN, split count: 7, time: 3.42
Output dim: 9, lower bound: -0.0047963, upper bound: 0.0053802
RS_RSZ2_RSZ1_RSZ2_RSZ2_RSZ2_RSZ1_RSZ1, status: Status.VERIFIED, split count: 7, time: 3.42
Output dim: 9, lower bound: -0.0049978, upper bound: 0.0050529
RS_RSZ2_RSZ1_RSZ2_RSZ2_RSZ2_RSZ1_RSZ2, status: Status.UNKNOWN, split count: 7, time: 3.42
Output dim: 9, lower bound: -0.0049534, upper bound: 0.0052773
RS_RSZ2_RSZ1_RSZ2_RSZ2_RSZ2_RSZ2_RSZ1, status: Status.VERIFIED, split count: 7, time: 3.42
Output dim: 9, lower bound: -0.0049978, upper bound: 0.0050522
RS_RSZ2_RSZ1_RSZ2_RSZ2_RSZ2_RSZ2_RSZ2, status: Status.UNKNOWN, split count: 7, time: 3.42
Output dim: 9, lower bound: -0.0049534, upper bound: 0.0052773
RS_RSZ2_RSZ2_RSZ1_RSZ1_RSZ1_RSZ1_RSZ1, status: Status.UNKNOWN, split count: 7, time: 3.42
Output dim: 9, lower bound: -0.0052257, upper bound: 0.0049917
RS_RSZ2_RSZ2_RSZ1_RSZ1_RSZ1_RSZ1_RSZ2, status: Status.VERIFIED, split count: 7, time: 3.42
Output dim: 9, lower bound: -0.0050264, upper bound: 0.0050255
RS_RSZ2_RSZ2_RSZ1_RSZ1_RSZ1_RSZ2_RSZ1, status: Status.UNKNOWN, split count: 7, time: 3.42
Output dim: 9, lower bound: -0.0052257, upper bound: 0.0049917
RS_RSZ2_RSZ2_RSZ1_RSZ1_RSZ1_RSZ2_RSZ2, status: Status.VERIFIED, split count: 7, time: 3.42
Output dim: 9, lower bound: -0.0050265, upper bound: 0.0050255
RS_RSZ2_RSZ2_RSZ1_RSZ1_RSZ2_RSZ1_RSZ1, status: Status.UNKNOWN, split count: 7, time: 3.42
Output dim: 9, lower bound: -0.0053493, upper bound: 0.0048358
RS_RSZ2_RSZ2_RSZ1_RSZ1_RSZ2_RSZ1_RSZ2, status: Status.UNKNOWN, split count: 7, time: 3.42
Output dim: 9, lower bound: -0.0051587, upper bound: 0.0049024
RS_RSZ2_RSZ2_RSZ1_RSZ1_RSZ2_RSZ2_RSZ1, status: Status.UNKNOWN, split count: 7, time: 3.42
Output dim: 9, lower bound: -0.0053493, upper bound: 0.0048358
RS_RSZ2_RSZ2_RSZ1_RSZ1_RSZ2_RSZ2_RSZ2, status: Status.UNKNOWN, split count: 7, time: 3.42
Output dim: 9, lower bound: -0.0051598, upper bound: 0.0049024
RS_RSZ2_RSZ2_RSZ1_RSZ2_RSZ1_RSZ1_RSZ1, status: Status.UNKNOWN, split count: 7, time: 3.42
Output dim: 9, lower bound: -0.0047969, upper bound: 0.0053095
RS_RSZ2_RSZ2_RSZ1_RSZ2_RSZ1_RSZ1_RSZ2, status: Status.UNKNOWN, split count: 7, time: 3.42
Output dim: 9, lower bound: -0.0047392, upper bound: 0.0054783
RS_RSZ2_RSZ2_RSZ1_RSZ2_RSZ1_RSZ2_RSZ1, status: Status.UNKNOWN, split count: 7, time: 3.42
Output dim: 9, lower bound: -0.0047969, upper bound: 0.0053094
RS_RSZ2_RSZ2_RSZ1_RSZ2_RSZ1_RSZ2_RSZ2, status: Status.UNKNOWN, split count: 7, time: 3.42
Output dim: 9, lower bound: -0.0047392, upper bound: 0.0054783
RS_RSZ2_RSZ2_RSZ1_RSZ2_RSZ2_RSZ1_RSZ1, status: Status.UNKNOWN, split count: 7, time: 3.42
Output dim: 9, lower bound: -0.0048890, upper bound: 0.0051239
RS_RSZ2_RSZ2_RSZ1_RSZ2_RSZ2_RSZ1_RSZ2, status: Status.UNKNOWN, split count: 7, time: 3.42
Output dim: 9, lower bound: -0.0048282, upper bound: 0.0053174
RS_RSZ2_RSZ2_RSZ1_RSZ2_RSZ2_RSZ2_RSZ1, status: Status.UNKNOWN, split count: 7, time: 3.42
Output dim: 9, lower bound: -0.0048890, upper bound: 0.0051236
RS_RSZ2_RSZ2_RSZ1_RSZ2_RSZ2_RSZ2_RSZ2, status: Status.UNKNOWN, split count: 7, time: 3.42
Output dim: 9, lower bound: -0.0048282, upper bound: 0.0053174
RS_RSZ2_RSZ2_RSZ2_RSZ1_RSZ1_RSZ1_RSZ1, status: Status.UNKNOWN, split count: 7, time: 3.42
Output dim: 9, lower bound: -0.0052068, upper bound: 0.0050176
RS_RSZ2_RSZ2_RSZ2_RSZ1_RSZ1_RSZ1_RSZ2, status: Status.UNKNOWN, split count: 7, time: 3.42
Output dim: 9, lower bound: -0.0050102, upper bound: 0.0050645
RS_RSZ2_RSZ2_RSZ2_RSZ1_RSZ1_RSZ2_RSZ1, status: Status.UNKNOWN, split count: 7, time: 3.42
Output dim: 9, lower bound: -0.0052068, upper bound: 0.0050176
RS_RSZ2_RSZ2_RSZ2_RSZ1_RSZ1_RSZ2_RSZ2, status: Status.UNKNOWN, split count: 7, time: 3.42
Output dim: 9, lower bound: -0.0050105, upper bound: 0.0050645
RS_RSZ2_RSZ2_RSZ2_RSZ1_RSZ2_RSZ1_RSZ1, status: Status.UNKNOWN, split count: 7, time: 3.42
Output dim: 9, lower bound: -0.0053453, upper bound: 0.0048916
RS_RSZ2_RSZ2_RSZ2_RSZ1_RSZ2_RSZ1_RSZ2, status: Status.UNKNOWN, split count: 7, time: 3.42
Output dim: 9, lower bound: -0.0051584, upper bound: 0.0049733
RS_RSZ2_RSZ2_RSZ2_RSZ1_RSZ2_RSZ2_RSZ1, status: Status.UNKNOWN, split count: 7, time: 3.42
Output dim: 9, lower bound: -0.0053453, upper bound: 0.0048916
RS_RSZ2_RSZ2_RSZ2_RSZ1_RSZ2_RSZ2_RSZ2, status: Status.UNKNOWN, split count: 7, time: 3.42
Output dim: 9, lower bound: -0.0051592, upper bound: 0.0049733
RS_RSZ2_RSZ2_RSZ2_RSZ2_RSZ1_RSZ1_RSZ1, status: Status.UNKNOWN, split count: 7, time: 3.42
Output dim: 9, lower bound: -0.0047324, upper bound: 0.0053154
RS_RSZ2_RSZ2_RSZ2_RSZ2_RSZ1_RSZ1_RSZ2, status: Status.UNKNOWN, split count: 7, time: 3.42
Output dim: 9, lower bound: -0.0046830, upper bound: 0.0054877
RS_RSZ2_RSZ2_RSZ2_RSZ2_RSZ1_RSZ2_RSZ1, status: Status.UNKNOWN, split count: 7, time: 3.42
Output dim: 9, lower bound: -0.0047324, upper bound: 0.0053148
RS_RSZ2_RSZ2_RSZ2_RSZ2_RSZ1_RSZ2_RSZ2, status: Status.UNKNOWN, split count: 7, time: 3.42
Output dim: 9, lower bound: -0.0046830, upper bound: 0.0054877
RS_RSZ2_RSZ2_RSZ2_RSZ2_RSZ2_RSZ1_RSZ1, status: Status.UNKNOWN, split count: 7, time: 3.42
Output dim: 9, lower bound: -0.0048608, upper bound: 0.0051431
RS_RSZ2_RSZ2_RSZ2_RSZ2_RSZ2_RSZ1_RSZ2, status: Status.UNKNOWN, split count: 7, time: 3.42
Output dim: 9, lower bound: -0.0048081, upper bound: 0.0053505
RS_RSZ2_RSZ2_RSZ2_RSZ2_RSZ2_RSZ2_RSZ1, status: Status.UNKNOWN, split count: 7, time: 3.42
Output dim: 9, lower bound: -0.0048608, upper bound: 0.0051431
RS_RSZ2_RSZ2_RSZ2_RSZ2_RSZ2_RSZ2_RSZ2, status: Status.UNKNOWN, split count: 7, time: 3.42
Output dim: 9, lower bound: -0.0048081, upper bound: 0.0053505

## BFS RS instance: RS_RSZ1_RSZ1_RSZ1_RSZ1_RSZ1_RSZ1_RSZ1

### Backsubstitution after applying RS history:
0: -0.0039095, -0.0003029, -0.0039095, -0.0003029, -0.0032473, 0.0031990
1: -0.0044907, 0.0040812, -0.0044907, 0.0040812, -0.0039310, 0.0040728
2: 0.0029569, 0.0099431, 0.0029569, 0.0099431, -0.0036995, 0.0037915
3: -0.0045519, -0.0034923, -0.0045519, -0.0034923, -0.0010595, 0.0010595
4: 0.0017018, 0.0078700, 0.0017018, 0.0078700, -0.0052635, 0.0052269
5: -0.0030341, 0.0031953, -0.0030341, 0.0031953, -0.0046788, 0.0047391
6: -0.0068516, -0.0033211, -0.0068516, -0.0033211, -0.0033327, 0.0033451
7: -0.0021570, 0.0038259, -0.0021570, 0.0038259, -0.0057045, 0.0056986
8: -0.0008533, 0.0000782, -0.0008533, 0.0000782, -0.0008465, 0.0008685
9: 0.9973557, 1.0121032, 0.9973557, 1.0121032, -0.0090093, 0.0088281

### Unstable ReLU Count (Linear/Conv2D Layers)
- layer_idx=0, type=LayerType.Linear, total=256, inp1_unstable=7, inp2_unstable=7, delta_unstable=249
- layer_idx=2, type=LayerType.Linear, total=256, inp1_unstable=10, inp2_unstable=10, delta_unstable=256
- layer_idx=4, type=LayerType.Linear, total=256, inp1_unstable=1, inp2_unstable=1, delta_unstable=256
- layer_idx=6, type=LayerType.Linear, total=256, inp1_unstable=0, inp2_unstable=0, delta_unstable=256
- layer_idx=8, type=LayerType.Linear, total=10, inp1_unstable=4, inp2_unstable=4, delta_unstable=10

Time for backsubstitution: 1.38 seconds

### RS candidates at layer 3
type: RSZ, layer: 3, pos: 8
type: RSZ, layer: 3, pos: 13
type: RSZ, layer: 3, pos: 20
type: RSZ, layer: 3, pos: 35
type: RSZ, layer: 3, pos: 43
type: RSZ, layer: 3, pos: 48
type: RSZ, layer: 3, pos: 103
type: RSZ, layer: 3, pos: 133
type: RSZ, layer: 3, pos: 180
type: RSZ, layer: 3, pos: 183

Time for candidate selection: 0.29 seconds

### Candidate
type: RSZ, layer: 3, pos: 8

### Relational analysis RSZ of RS_RSZ1_RSZ1_RSZ1_RSZ1_RSZ1_RSZ1_RSZ1_RSZ1

#### Relational analysis RSZ result of RS_RSZ1_RSZ1_RSZ1_RSZ1_RSZ1_RSZ1_RSZ1_RSZ1
Status: Status.VERIFIED
Output dim: 9, lower bound: -0.0046521, upper bound: 0.0042557
time: 0.86 seconds

### Relational analysis RSZ of RS_RSZ1_RSZ1_RSZ1_RSZ1_RSZ1_RSZ1_RSZ1_RSZ2

#### Relational analysis RSZ result of RS_RSZ1_RSZ1_RSZ1_RSZ1_RSZ1_RSZ1_RSZ1_RSZ2
Status: Status.VERIFIED
Output dim: 9, lower bound: -0.0046521, upper bound: 0.0042548
time: 0.84 seconds

## BFS RS instance: RS_RSZ1_RSZ1_RSZ1_RSZ1_RSZ1_RSZ1_RSZ2

### Backsubstitution after applying RS history:
0: -0.0039095, -0.0003029, -0.0039095, -0.0003029, -0.0032298, 0.0032096
1: -0.0044907, 0.0040812, -0.0044907, 0.0040812, -0.0039618, 0.0040178
2: 0.0029569, 0.0099431, 0.0029569, 0.0099431, -0.0037247, 0.0037520
3: -0.0045519, -0.0034923, -0.0045519, -0.0034923, -0.0010595, 0.0010595
4: 0.0017018, 0.0078700, 0.0017018, 0.0078700, -0.0052498, 0.0052354
5: -0.0030341, 0.0031953, -0.0030341, 0.0031953, -0.0046868, 0.0047240
6: -0.0068516, -0.0033211, -0.0068516, -0.0033211, -0.0033353, 0.0033405
7: -0.0021570, 0.0038259, -0.0021570, 0.0038259, -0.0056879, 0.0057090
8: -0.0008533, 0.0000782, -0.0008533, 0.0000782, -0.0008465, 0.0008679
9: 0.9973557, 1.0121032, 0.9973557, 1.0121032, -0.0089428, 0.0088666

### Unstable ReLU Count (Linear/Conv2D Layers)
- layer_idx=0, type=LayerType.Linear, total=256, inp1_unstable=7, inp2_unstable=7, delta_unstable=249
- layer_idx=2, type=LayerType.Linear, total=256, inp1_unstable=10, inp2_unstable=10, delta_unstable=256
- layer_idx=4, type=LayerType.Linear, total=256, inp1_unstable=1, inp2_unstable=1, delta_unstable=256
- layer_idx=6, type=LayerType.Linear, total=256, inp1_unstable=0, inp2_unstable=0, delta_unstable=256
- layer_idx=8, type=LayerType.Linear, total=10, inp1_unstable=4, inp2_unstable=4, delta_unstable=10

Time for backsubstitution: 1.39 seconds

### RS candidates at layer 3
type: RSZ, layer: 3, pos: 8
type: RSZ, layer: 3, pos: 13
type: RSZ, layer: 3, pos: 20
type: RSZ, layer: 3, pos: 35
type: RSZ, layer: 3, pos: 43
type: RSZ, layer: 3, pos: 48
type: RSZ, layer: 3, pos: 103
type: RSZ, layer: 3, pos: 133
type: RSZ, layer: 3, pos: 180
type: RSZ, layer: 3, pos: 183

Time for candidate selection: 0.27 seconds

### Candidate
type: RSZ, layer: 3, pos: 8

### Relational analysis RSZ of RS_RSZ1_RSZ1_RSZ1_RSZ1_RSZ1_RSZ1_RSZ2_RSZ1

#### Relational analysis RSZ result of RS_RSZ1_RSZ1_RSZ1_RSZ1_RSZ1_RSZ1_RSZ2_RSZ1
Status: Status.VERIFIED
Output dim: 9, lower bound: -0.0045036, upper bound: 0.0042992
time: 0.81 seconds

### Relational analysis RSZ of RS_RSZ1_RSZ1_RSZ1_RSZ1_RSZ1_RSZ1_RSZ2_RSZ2

#### Relational analysis RSZ result of RS_RSZ1_RSZ1_RSZ1_RSZ1_RSZ1_RSZ1_RSZ2_RSZ2
Status: Status.VERIFIED
Output dim: 9, lower bound: -0.0045036, upper bound: 0.0042958
time: 0.82 seconds

## BFS RS instance: RS_RSZ1_RSZ1_RSZ1_RSZ1_RSZ1_RSZ2_RSZ1

### Backsubstitution after applying RS history:
0: -0.0039095, -0.0003029, -0.0039095, -0.0003029, -0.0032468, 0.0031995
1: -0.0044907, 0.0040812, -0.0044907, 0.0040812, -0.0039309, 0.0040726
2: 0.0029569, 0.0099431, 0.0029569, 0.0099431, -0.0036996, 0.0037908
3: -0.0045519, -0.0034923, -0.0045519, -0.0034923, -0.0010595, 0.0010595
4: 0.0017018, 0.0078700, 0.0017018, 0.0078700, -0.0052632, 0.0052271
5: -0.0030341, 0.0031953, -0.0030341, 0.0031953, -0.0046802, 0.0047389
6: -0.0068516, -0.0033211, -0.0068516, -0.0033211, -0.0033327, 0.0033451
7: -0.0021570, 0.0038259, -0.0021570, 0.0038259, -0.0057038, 0.0056993
8: -0.0008533, 0.0000782, -0.0008533, 0.0000782, -0.0008476, 0.0008703
9: 0.9973557, 1.0121032, 0.9973557, 1.0121032, -0.0090088, 0.0088282

### Unstable ReLU Count (Linear/Conv2D Layers)
- layer_idx=0, type=LayerType.Linear, total=256, inp1_unstable=7, inp2_unstable=7, delta_unstable=249
- layer_idx=2, type=LayerType.Linear, total=256, inp1_unstable=10, inp2_unstable=10, delta_unstable=256
- layer_idx=4, type=LayerType.Linear, total=256, inp1_unstable=1, inp2_unstable=1, delta_unstable=256
- layer_idx=6, type=LayerType.Linear, total=256, inp1_unstable=0, inp2_unstable=0, delta_unstable=256
- layer_idx=8, type=LayerType.Linear, total=10, inp1_unstable=4, inp2_unstable=4, delta_unstable=10

Time for backsubstitution: 1.36 seconds

### RS candidates at layer 3
type: RSZ, layer: 3, pos: 8
type: RSZ, layer: 3, pos: 13
type: RSZ, layer: 3, pos: 20
type: RSZ, layer: 3, pos: 35
type: RSZ, layer: 3, pos: 43
type: RSZ, layer: 3, pos: 48
type: RSZ, layer: 3, pos: 103
type: RSZ, layer: 3, pos: 133
type: RSZ, layer: 3, pos: 180
type: RSZ, layer: 3, pos: 183

Time for candidate selection: 0.29 seconds

### Candidate
type: RSZ, layer: 3, pos: 8

### Relational analysis RSZ of RS_RSZ1_RSZ1_RSZ1_RSZ1_RSZ1_RSZ2_RSZ1_RSZ1

#### Relational analysis RSZ result of RS_RSZ1_RSZ1_RSZ1_RSZ1_RSZ1_RSZ2_RSZ1_RSZ1
Status: Status.VERIFIED
Output dim: 9, lower bound: -0.0046521, upper bound: 0.0042557
time: 0.86 seconds

### Relational analysis RSZ of RS_RSZ1_RSZ1_RSZ1_RSZ1_RSZ1_RSZ2_RSZ1_RSZ2

#### Relational analysis RSZ result of RS_RSZ1_RSZ1_RSZ1_RSZ1_RSZ1_RSZ2_RSZ1_RSZ2
Status: Status.VERIFIED
Output dim: 9, lower bound: -0.0046521, upper bound: 0.0042548
time: 0.84 seconds

## BFS RS instance: RS_RSZ1_RSZ1_RSZ1_RSZ1_RSZ1_RSZ2_RSZ2

### Backsubstitution after applying RS history:
0: -0.0039095, -0.0003029, -0.0039095, -0.0003029, -0.0032294, 0.0032099
1: -0.0044907, 0.0040812, -0.0044907, 0.0040812, -0.0039619, 0.0040176
2: 0.0029569, 0.0099431, 0.0029569, 0.0099431, -0.0037250, 0.0037516
3: -0.0045519, -0.0034923, -0.0045519, -0.0034923, -0.0010595, 0.0010595
4: 0.0017018, 0.0078700, 0.0017018, 0.0078700, -0.0052494, 0.0052356
5: -0.0030341, 0.0031953, -0.0030341, 0.0031953, -0.0046884, 0.0047239
6: -0.0068516, -0.0033211, -0.0068516, -0.0033211, -0.0033353, 0.0033405
7: -0.0021570, 0.0038259, -0.0021570, 0.0038259, -0.0056882, 0.0057101
8: -0.0008533, 0.0000782, -0.0008533, 0.0000782, -0.0008476, 0.0008697
9: 0.9973557, 1.0121032, 0.9973557, 1.0121032, -0.0089425, 0.0088669

### Unstable ReLU Count (Linear/Conv2D Layers)
- layer_idx=0, type=LayerType.Linear, total=256, inp1_unstable=7, inp2_unstable=7, delta_unstable=249
- layer_idx=2, type=LayerType.Linear, total=256, inp1_unstable=10, inp2_unstable=10, delta_unstable=256
- layer_idx=4, type=LayerType.Linear, total=256, inp1_unstable=1, inp2_unstable=1, delta_unstable=256
- layer_idx=6, type=LayerType.Linear, total=256, inp1_unstable=0, inp2_unstable=0, delta_unstable=256
- layer_idx=8, type=LayerType.Linear, total=10, inp1_unstable=4, inp2_unstable=4, delta_unstable=10

Time for backsubstitution: 1.30 seconds

### RS candidates at layer 3
type: RSZ, layer: 3, pos: 8
type: RSZ, layer: 3, pos: 13
type: RSZ, layer: 3, pos: 20
type: RSZ, layer: 3, pos: 35
type: RSZ, layer: 3, pos: 43
type: RSZ, layer: 3, pos: 48
type: RSZ, layer: 3, pos: 103
type: RSZ, layer: 3, pos: 133
type: RSZ, layer: 3, pos: 180
type: RSZ, layer: 3, pos: 183

Time for candidate selection: 0.28 seconds

### Candidate
type: RSZ, layer: 3, pos: 8

### Relational analysis RSZ of RS_RSZ1_RSZ1_RSZ1_RSZ1_RSZ1_RSZ2_RSZ2_RSZ1

#### Relational analysis RSZ result of RS_RSZ1_RSZ1_RSZ1_RSZ1_RSZ1_RSZ2_RSZ2_RSZ1
Status: Status.VERIFIED
Output dim: 9, lower bound: -0.0045036, upper bound: 0.0042992
time: 0.83 seconds

### Relational analysis RSZ of RS_RSZ1_RSZ1_RSZ1_RSZ1_RSZ1_RSZ2_RSZ2_RSZ2

#### Relational analysis RSZ result of RS_RSZ1_RSZ1_RSZ1_RSZ1_RSZ1_RSZ2_RSZ2_RSZ2
Status: Status.VERIFIED
Output dim: 9, lower bound: -0.0045036, upper bound: 0.0042958
time: 0.83 seconds

## BFS RS instance: RS_RSZ1_RSZ1_RSZ1_RSZ1_RSZ2_RSZ1_RSZ1

### Backsubstitution after applying RS history:
0: -0.0039095, -0.0003029, -0.0039095, -0.0003029, -0.0032903, 0.0031421
1: -0.0044907, 0.0040812, -0.0044907, 0.0040812, -0.0037912, 0.0041802
2: 0.0029569, 0.0099431, 0.0029569, 0.0099431, -0.0036035, 0.0038661
3: -0.0045519, -0.0034923, -0.0045519, -0.0034923, -0.0010595, 0.0010595
4: 0.0017018, 0.0078700, 0.0017018, 0.0078700, -0.0052959, 0.0051832
5: -0.0030341, 0.0031953, -0.0030341, 0.0031953, -0.0046172, 0.0047937
6: -0.0068516, -0.0033211, -0.0068516, -0.0033211, -0.0033207, 0.0033544
7: -0.0021570, 0.0038259, -0.0021570, 0.0038259, -0.0057289, 0.0056697
8: -0.0008533, 0.0000782, -0.0008533, 0.0000782, -0.0008395, 0.0008750
9: 0.9973557, 1.0121032, 0.9973557, 1.0121032, -0.0091483, 0.0086450

### Unstable ReLU Count (Linear/Conv2D Layers)
- layer_idx=0, type=LayerType.Linear, total=256, inp1_unstable=7, inp2_unstable=7, delta_unstable=249
- layer_idx=2, type=LayerType.Linear, total=256, inp1_unstable=10, inp2_unstable=10, delta_unstable=256
- layer_idx=4, type=LayerType.Linear, total=256, inp1_unstable=1, inp2_unstable=1, delta_unstable=256
- layer_idx=6, type=LayerType.Linear, total=256, inp1_unstable=0, inp2_unstable=0, delta_unstable=256
- layer_idx=8, type=LayerType.Linear, total=10, inp1_unstable=4, inp2_unstable=4, delta_unstable=10

Time for backsubstitution: 1.37 seconds

### RS candidates at layer 3
type: RSZ, layer: 3, pos: 8
type: RSZ, layer: 3, pos: 13
type: RSZ, layer: 3, pos: 20
type: RSZ, layer: 3, pos: 35
type: RSZ, layer: 3, pos: 43
type: RSZ, layer: 3, pos: 48
type: RSZ, layer: 3, pos: 103
type: RSZ, layer: 3, pos: 133
type: RSZ, layer: 3, pos: 180
type: RSZ, layer: 3, pos: 183

Time for candidate selection: 0.30 seconds

### Candidate
type: RSZ, layer: 3, pos: 8

### Relational analysis RSZ of RS_RSZ1_RSZ1_RSZ1_RSZ1_RSZ2_RSZ1_RSZ1_RSZ1

#### Relational analysis RSZ result of RS_RSZ1_RSZ1_RSZ1_RSZ1_RSZ2_RSZ1_RSZ1_RSZ1
Status: Status.VERIFIED
Output dim: 9, lower bound: -0.0047745, upper bound: 0.0041353
time: 0.85 seconds

### Relational analysis RSZ of RS_RSZ1_RSZ1_RSZ1_RSZ1_RSZ2_RSZ1_RSZ1_RSZ2

#### Relational analysis RSZ result of RS_RSZ1_RSZ1_RSZ1_RSZ1_RSZ2_RSZ1_RSZ1_RSZ2
Status: Status.VERIFIED
Output dim: 9, lower bound: -0.0047745, upper bound: 0.0041353
time: 0.86 seconds

## BFS RS instance: RS_RSZ1_RSZ1_RSZ1_RSZ1_RSZ2_RSZ1_RSZ2

### Backsubstitution after applying RS history:
0: -0.0039095, -0.0003029, -0.0039095, -0.0003029, -0.0032780, 0.0031556
1: -0.0044907, 0.0040812, -0.0044907, 0.0040812, -0.0038355, 0.0041382
2: 0.0029569, 0.0099431, 0.0029569, 0.0099431, -0.0036365, 0.0038352
3: -0.0045519, -0.0034923, -0.0045519, -0.0034923, -0.0010595, 0.0010595
4: 0.0017018, 0.0078700, 0.0017018, 0.0078700, -0.0052862, 0.0051943
5: -0.0030341, 0.0031953, -0.0030341, 0.0031953, -0.0046281, 0.0047825
6: -0.0068516, -0.0033211, -0.0068516, -0.0033211, -0.0033244, 0.0033509
7: -0.0021570, 0.0038259, -0.0021570, 0.0038259, -0.0057166, 0.0056834
8: -0.0008533, 0.0000782, -0.0008533, 0.0000782, -0.0008396, 0.0008744
9: 0.9973557, 1.0121032, 0.9973557, 1.0121032, -0.0090993, 0.0086980

### Unstable ReLU Count (Linear/Conv2D Layers)
- layer_idx=0, type=LayerType.Linear, total=256, inp1_unstable=7, inp2_unstable=7, delta_unstable=249
- layer_idx=2, type=LayerType.Linear, total=256, inp1_unstable=10, inp2_unstable=10, delta_unstable=256
- layer_idx=4, type=LayerType.Linear, total=256, inp1_unstable=1, inp2_unstable=1, delta_unstable=256
- layer_idx=6, type=LayerType.Linear, total=256, inp1_unstable=0, inp2_unstable=0, delta_unstable=256
- layer_idx=8, type=LayerType.Linear, total=10, inp1_unstable=4, inp2_unstable=4, delta_unstable=10

Time for backsubstitution: 1.36 seconds

### RS candidates at layer 3
type: RSZ, layer: 3, pos: 8
type: RSZ, layer: 3, pos: 13
type: RSZ, layer: 3, pos: 20
type: RSZ, layer: 3, pos: 35
type: RSZ, layer: 3, pos: 43
type: RSZ, layer: 3, pos: 48
type: RSZ, layer: 3, pos: 103
type: RSZ, layer: 3, pos: 133
type: RSZ, layer: 3, pos: 180
type: RSZ, layer: 3, pos: 183

Time for candidate selection: 0.31 seconds

### Candidate
type: RSZ, layer: 3, pos: 8

### Relational analysis RSZ of RS_RSZ1_RSZ1_RSZ1_RSZ1_RSZ2_RSZ1_RSZ2_RSZ1

#### Relational analysis RSZ result of RS_RSZ1_RSZ1_RSZ1_RSZ1_RSZ2_RSZ1_RSZ2_RSZ1
Status: Status.VERIFIED
Output dim: 9, lower bound: -0.0046577, upper bound: 0.0041857
time: 0.90 seconds

### Relational analysis RSZ of RS_RSZ1_RSZ1_RSZ1_RSZ1_RSZ2_RSZ1_RSZ2_RSZ2

#### Relational analysis RSZ result of RS_RSZ1_RSZ1_RSZ1_RSZ1_RSZ2_RSZ1_RSZ2_RSZ2
Status: Status.VERIFIED
Output dim: 9, lower bound: -0.0046577, upper bound: 0.0041857
time: 0.89 seconds

## BFS RS instance: RS_RSZ1_RSZ1_RSZ1_RSZ1_RSZ2_RSZ2_RSZ1

### Backsubstitution after applying RS history:
0: -0.0039095, -0.0003029, -0.0039095, -0.0003029, -0.0032900, 0.0031429
1: -0.0044907, 0.0040812, -0.0044907, 0.0040812, -0.0037916, 0.0041801
2: 0.0029569, 0.0099431, 0.0029569, 0.0099431, -0.0036038, 0.0038661
3: -0.0045519, -0.0034923, -0.0045519, -0.0034923, -0.0010595, 0.0010595
4: 0.0017018, 0.0078700, 0.0017018, 0.0078700, -0.0052959, 0.0051838
5: -0.0030341, 0.0031953, -0.0030341, 0.0031953, -0.0046202, 0.0047933
6: -0.0068516, -0.0033211, -0.0068516, -0.0033211, -0.0033207, 0.0033544
7: -0.0021570, 0.0038259, -0.0021570, 0.0038259, -0.0057287, 0.0056704
8: -0.0008533, 0.0000782, -0.0008533, 0.0000782, -0.0008404, 0.0008769
9: 0.9973557, 1.0121032, 0.9973557, 1.0121032, -0.0091482, 0.0086459

### Unstable ReLU Count (Linear/Conv2D Layers)
- layer_idx=0, type=LayerType.Linear, total=256, inp1_unstable=7, inp2_unstable=7, delta_unstable=249
- layer_idx=2, type=LayerType.Linear, total=256, inp1_unstable=10, inp2_unstable=10, delta_unstable=256
- layer_idx=4, type=LayerType.Linear, total=256, inp1_unstable=1, inp2_unstable=1, delta_unstable=256
- layer_idx=6, type=LayerType.Linear, total=256, inp1_unstable=0, inp2_unstable=0, delta_unstable=256
- layer_idx=8, type=LayerType.Linear, total=10, inp1_unstable=4, inp2_unstable=4, delta_unstable=10

Time for backsubstitution: 1.33 seconds

### RS candidates at layer 3
type: RSZ, layer: 3, pos: 8
type: RSZ, layer: 3, pos: 13
type: RSZ, layer: 3, pos: 20
type: RSZ, layer: 3, pos: 35
type: RSZ, layer: 3, pos: 43
type: RSZ, layer: 3, pos: 48
type: RSZ, layer: 3, pos: 103
type: RSZ, layer: 3, pos: 133
type: RSZ, layer: 3, pos: 180
type: RSZ, layer: 3, pos: 183

Time for candidate selection: 0.28 seconds

### Candidate
type: RSZ, layer: 3, pos: 8

### Relational analysis RSZ of RS_RSZ1_RSZ1_RSZ1_RSZ1_RSZ2_RSZ2_RSZ1_RSZ1

#### Relational analysis RSZ result of RS_RSZ1_RSZ1_RSZ1_RSZ1_RSZ2_RSZ2_RSZ1_RSZ1
Status: Status.VERIFIED
Output dim: 9, lower bound: -0.0047745, upper bound: 0.0041353
time: 0.88 seconds

### Relational analysis RSZ of RS_RSZ1_RSZ1_RSZ1_RSZ1_RSZ2_RSZ2_RSZ1_RSZ2

#### Relational analysis RSZ result of RS_RSZ1_RSZ1_RSZ1_RSZ1_RSZ2_RSZ2_RSZ1_RSZ2
Status: Status.VERIFIED
Output dim: 9, lower bound: -0.0047745, upper bound: 0.0041353
time: 0.87 seconds

## BFS RS instance: RS_RSZ1_RSZ1_RSZ1_RSZ1_RSZ2_RSZ2_RSZ2

### Backsubstitution after applying RS history:
0: -0.0039095, -0.0003029, -0.0039095, -0.0003029, -0.0032780, 0.0031563
1: -0.0044907, 0.0040812, -0.0044907, 0.0040812, -0.0038363, 0.0041383
2: 0.0029569, 0.0099431, 0.0029569, 0.0099431, -0.0036375, 0.0038351
3: -0.0045519, -0.0034923, -0.0045519, -0.0034923, -0.0010595, 0.0010595
4: 0.0017018, 0.0078700, 0.0017018, 0.0078700, -0.0052862, 0.0051947
5: -0.0030341, 0.0031953, -0.0030341, 0.0031953, -0.0046306, 0.0047828
6: -0.0068516, -0.0033211, -0.0068516, -0.0033211, -0.0033244, 0.0033509
7: -0.0021570, 0.0038259, -0.0021570, 0.0038259, -0.0057178, 0.0056870
8: -0.0008533, 0.0000782, -0.0008533, 0.0000782, -0.0008406, 0.0008765
9: 0.9973557, 1.0121032, 0.9973557, 1.0121032, -0.0090994, 0.0086990

### Unstable ReLU Count (Linear/Conv2D Layers)
- layer_idx=0, type=LayerType.Linear, total=256, inp1_unstable=7, inp2_unstable=7, delta_unstable=249
- layer_idx=2, type=LayerType.Linear, total=256, inp1_unstable=10, inp2_unstable=10, delta_unstable=256
- layer_idx=4, type=LayerType.Linear, total=256, inp1_unstable=1, inp2_unstable=1, delta_unstable=256
- layer_idx=6, type=LayerType.Linear, total=256, inp1_unstable=0, inp2_unstable=0, delta_unstable=256
- layer_idx=8, type=LayerType.Linear, total=10, inp1_unstable=4, inp2_unstable=4, delta_unstable=10

Time for backsubstitution: 1.38 seconds

### RS candidates at layer 3
type: RSZ, layer: 3, pos: 8
type: RSZ, layer: 3, pos: 13
type: RSZ, layer: 3, pos: 20
type: RSZ, layer: 3, pos: 35
type: RSZ, layer: 3, pos: 43
type: RSZ, layer: 3, pos: 48
type: RSZ, layer: 3, pos: 103
type: RSZ, layer: 3, pos: 133
type: RSZ, layer: 3, pos: 180
type: RSZ, layer: 3, pos: 183

Time for candidate selection: 0.27 seconds

### Candidate
type: RSZ, layer: 3, pos: 8

### Relational analysis RSZ of RS_RSZ1_RSZ1_RSZ1_RSZ1_RSZ2_RSZ2_RSZ2_RSZ1

#### Relational analysis RSZ result of RS_RSZ1_RSZ1_RSZ1_RSZ1_RSZ2_RSZ2_RSZ2_RSZ1
Status: Status.VERIFIED
Output dim: 9, lower bound: -0.0046577, upper bound: 0.0041857
time: 0.88 seconds

### Relational analysis RSZ of RS_RSZ1_RSZ1_RSZ1_RSZ1_RSZ2_RSZ2_RSZ2_RSZ2

#### Relational analysis RSZ result of RS_RSZ1_RSZ1_RSZ1_RSZ1_RSZ2_RSZ2_RSZ2_RSZ2
Status: Status.VERIFIED
Output dim: 9, lower bound: -0.0046577, upper bound: 0.0041857
time: 0.89 seconds

## BFS RS instance: RS_RSZ1_RSZ1_RSZ1_RSZ2_RSZ1_RSZ1_RSZ1

### Backsubstitution after applying RS history:
0: -0.0039095, -0.0003029, -0.0039095, -0.0003029, -0.0031764, 0.0032693
1: -0.0044907, 0.0040812, -0.0044907, 0.0040812, -0.0040933, 0.0039022
2: 0.0029569, 0.0099431, 0.0029569, 0.0099431, -0.0038037, 0.0036805
3: -0.0045519, -0.0034923, -0.0045519, -0.0034923, -0.0010595, 0.0010595
4: 0.0017018, 0.0078700, 0.0017018, 0.0078700, -0.0052094, 0.0052806
5: -0.0030341, 0.0031953, -0.0030341, 0.0031953, -0.0047583, 0.0046593
6: -0.0068516, -0.0033211, -0.0068516, -0.0033211, -0.0033474, 0.0033299
7: -0.0021570, 0.0038259, -0.0021570, 0.0038259, -0.0056786, 0.0057230
8: -0.0008533, 0.0000782, -0.0008533, 0.0000782, -0.0008571, 0.0008611
9: 0.9973557, 1.0121032, 0.9973557, 1.0121032, -0.0087778, 0.0090532

### Unstable ReLU Count (Linear/Conv2D Layers)
- layer_idx=0, type=LayerType.Linear, total=256, inp1_unstable=7, inp2_unstable=7, delta_unstable=249
- layer_idx=2, type=LayerType.Linear, total=256, inp1_unstable=10, inp2_unstable=10, delta_unstable=256
- layer_idx=4, type=LayerType.Linear, total=256, inp1_unstable=1, inp2_unstable=1, delta_unstable=256
- layer_idx=6, type=LayerType.Linear, total=256, inp1_unstable=0, inp2_unstable=0, delta_unstable=256
- layer_idx=8, type=LayerType.Linear, total=10, inp1_unstable=4, inp2_unstable=4, delta_unstable=10

Time for backsubstitution: 1.37 seconds

### RS candidates at layer 3
type: RSZ, layer: 3, pos: 8
type: RSZ, layer: 3, pos: 13
type: RSZ, layer: 3, pos: 20
type: RSZ, layer: 3, pos: 35
type: RSZ, layer: 3, pos: 43
type: RSZ, layer: 3, pos: 48
type: RSZ, layer: 3, pos: 103
type: RSZ, layer: 3, pos: 133
type: RSZ, layer: 3, pos: 180
type: RSZ, layer: 3, pos: 183

Time for candidate selection: 0.31 seconds

### Candidate
type: RSZ, layer: 3, pos: 8

### Relational analysis RSZ of RS_RSZ1_RSZ1_RSZ1_RSZ2_RSZ1_RSZ1_RSZ1_RSZ1

#### Relational analysis RSZ result of RS_RSZ1_RSZ1_RSZ1_RSZ2_RSZ1_RSZ1_RSZ1_RSZ1
Status: Status.VERIFIED
Output dim: 9, lower bound: -0.0043844, upper bound: 0.0045505
time: 0.79 seconds

### Relational analysis RSZ of RS_RSZ1_RSZ1_RSZ1_RSZ2_RSZ1_RSZ1_RSZ1_RSZ2

#### Relational analysis RSZ result of RS_RSZ1_RSZ1_RSZ1_RSZ2_RSZ1_RSZ1_RSZ1_RSZ2
Status: Status.VERIFIED
Output dim: 9, lower bound: -0.0043844, upper bound: 0.0045505
time: 0.80 seconds

## BFS RS instance: RS_RSZ1_RSZ1_RSZ1_RSZ2_RSZ1_RSZ1_RSZ2

### Backsubstitution after applying RS history:
0: -0.0039095, -0.0003029, -0.0039095, -0.0003029, -0.0031624, 0.0032821
1: -0.0044907, 0.0040812, -0.0044907, 0.0040812, -0.0041334, 0.0038596
2: 0.0029569, 0.0099431, 0.0029569, 0.0099431, -0.0038337, 0.0036498
3: -0.0045519, -0.0034923, -0.0045519, -0.0034923, -0.0010595, 0.0010595
4: 0.0017018, 0.0078700, 0.0017018, 0.0078700, -0.0051981, 0.0052907
5: -0.0030341, 0.0031953, -0.0030341, 0.0031953, -0.0047693, 0.0046490
6: -0.0068516, -0.0033211, -0.0068516, -0.0033211, -0.0033509, 0.0033264
7: -0.0021570, 0.0038259, -0.0021570, 0.0038259, -0.0056646, 0.0057359
8: -0.0008533, 0.0000782, -0.0008533, 0.0000782, -0.0008574, 0.0008606
9: 0.9973557, 1.0121032, 0.9973557, 1.0121032, -0.0087257, 0.0091030

### Unstable ReLU Count (Linear/Conv2D Layers)
- layer_idx=0, type=LayerType.Linear, total=256, inp1_unstable=7, inp2_unstable=7, delta_unstable=249
- layer_idx=2, type=LayerType.Linear, total=256, inp1_unstable=10, inp2_unstable=10, delta_unstable=256
- layer_idx=4, type=LayerType.Linear, total=256, inp1_unstable=1, inp2_unstable=1, delta_unstable=256
- layer_idx=6, type=LayerType.Linear, total=256, inp1_unstable=0, inp2_unstable=0, delta_unstable=256
- layer_idx=8, type=LayerType.Linear, total=10, inp1_unstable=4, inp2_unstable=4, delta_unstable=10

Time for backsubstitution: 1.31 seconds

### RS candidates at layer 3
type: RSZ, layer: 3, pos: 8
type: RSZ, layer: 3, pos: 13
type: RSZ, layer: 3, pos: 20
type: RSZ, layer: 3, pos: 35
type: RSZ, layer: 3, pos: 43
type: RSZ, layer: 3, pos: 48
type: RSZ, layer: 3, pos: 103
type: RSZ, layer: 3, pos: 133
type: RSZ, layer: 3, pos: 180
type: RSZ, layer: 3, pos: 183

Time for candidate selection: 0.26 seconds

### Candidate
type: RSZ, layer: 3, pos: 8

### Relational analysis RSZ of RS_RSZ1_RSZ1_RSZ1_RSZ2_RSZ1_RSZ1_RSZ2_RSZ1

#### Relational analysis RSZ result of RS_RSZ1_RSZ1_RSZ1_RSZ2_RSZ1_RSZ1_RSZ2_RSZ1
Status: Status.VERIFIED
Output dim: 9, lower bound: -0.0043047, upper bound: 0.0046810
time: 0.81 seconds

### Relational analysis RSZ of RS_RSZ1_RSZ1_RSZ1_RSZ2_RSZ1_RSZ1_RSZ2_RSZ2

#### Relational analysis RSZ result of RS_RSZ1_RSZ1_RSZ1_RSZ2_RSZ1_RSZ1_RSZ2_RSZ2
Status: Status.VERIFIED
Output dim: 9, lower bound: -0.0043047, upper bound: 0.0046810
time: 0.81 seconds

## BFS RS instance: RS_RSZ1_RSZ1_RSZ1_RSZ2_RSZ1_RSZ2_RSZ1

### Backsubstitution after applying RS history:
0: -0.0039095, -0.0003029, -0.0039095, -0.0003029, -0.0031757, 0.0032696
1: -0.0044907, 0.0040812, -0.0044907, 0.0040812, -0.0040932, 0.0039018
2: 0.0029569, 0.0099431, 0.0029569, 0.0099431, -0.0038033, 0.0036798
3: -0.0045519, -0.0034923, -0.0045519, -0.0034923, -0.0010595, 0.0010595
4: 0.0017018, 0.0078700, 0.0017018, 0.0078700, -0.0052089, 0.0052807
5: -0.0030341, 0.0031953, -0.0030341, 0.0031953, -0.0047588, 0.0046581
6: -0.0068516, -0.0033211, -0.0068516, -0.0033211, -0.0033474, 0.0033299
7: -0.0021570, 0.0038259, -0.0021570, 0.0038259, -0.0056777, 0.0057237
8: -0.0008533, 0.0000782, -0.0008533, 0.0000782, -0.0008572, 0.0008621
9: 0.9973557, 1.0121032, 0.9973557, 1.0121032, -0.0087770, 0.0090531

### Unstable ReLU Count (Linear/Conv2D Layers)
- layer_idx=0, type=LayerType.Linear, total=256, inp1_unstable=7, inp2_unstable=7, delta_unstable=249
- layer_idx=2, type=LayerType.Linear, total=256, inp1_unstable=10, inp2_unstable=10, delta_unstable=256
- layer_idx=4, type=LayerType.Linear, total=256, inp1_unstable=1, inp2_unstable=1, delta_unstable=256
- layer_idx=6, type=LayerType.Linear, total=256, inp1_unstable=0, inp2_unstable=0, delta_unstable=256
- layer_idx=8, type=LayerType.Linear, total=10, inp1_unstable=4, inp2_unstable=4, delta_unstable=10

Time for backsubstitution: 1.36 seconds

### RS candidates at layer 3
type: RSZ, layer: 3, pos: 8
type: RSZ, layer: 3, pos: 13
type: RSZ, layer: 3, pos: 20
type: RSZ, layer: 3, pos: 35
type: RSZ, layer: 3, pos: 43
type: RSZ, layer: 3, pos: 48
type: RSZ, layer: 3, pos: 103
type: RSZ, layer: 3, pos: 133
type: RSZ, layer: 3, pos: 180
type: RSZ, layer: 3, pos: 183

Time for candidate selection: 0.29 seconds

### Candidate
type: RSZ, layer: 3, pos: 8

### Relational analysis RSZ of RS_RSZ1_RSZ1_RSZ1_RSZ2_RSZ1_RSZ2_RSZ1_RSZ1

#### Relational analysis RSZ result of RS_RSZ1_RSZ1_RSZ1_RSZ2_RSZ1_RSZ2_RSZ1_RSZ1
Status: Status.VERIFIED
Output dim: 9, lower bound: -0.0043844, upper bound: 0.0045505
time: 0.80 seconds

### Relational analysis RSZ of RS_RSZ1_RSZ1_RSZ1_RSZ2_RSZ1_RSZ2_RSZ1_RSZ2

#### Relational analysis RSZ result of RS_RSZ1_RSZ1_RSZ1_RSZ2_RSZ1_RSZ2_RSZ1_RSZ2
Status: Status.VERIFIED
Output dim: 9, lower bound: -0.0043844, upper bound: 0.0045505
time: 0.79 seconds

## BFS RS instance: RS_RSZ1_RSZ1_RSZ1_RSZ2_RSZ1_RSZ2_RSZ2

### Backsubstitution after applying RS history:
0: -0.0039095, -0.0003029, -0.0039095, -0.0003029, -0.0031618, 0.0032827
1: -0.0044907, 0.0040812, -0.0044907, 0.0040812, -0.0041335, 0.0038594
2: 0.0029569, 0.0099431, 0.0029569, 0.0099431, -0.0038343, 0.0036492
3: -0.0045519, -0.0034923, -0.0045519, -0.0034923, -0.0010595, 0.0010595
4: 0.0017018, 0.0078700, 0.0017018, 0.0078700, -0.0051977, 0.0052911
5: -0.0030341, 0.0031953, -0.0030341, 0.0031953, -0.0047703, 0.0046474
6: -0.0068516, -0.0033211, -0.0068516, -0.0033211, -0.0033509, 0.0033263
7: -0.0021570, 0.0038259, -0.0021570, 0.0038259, -0.0056654, 0.0057371
8: -0.0008533, 0.0000782, -0.0008533, 0.0000782, -0.0008574, 0.0008618
9: 0.9973557, 1.0121032, 0.9973557, 1.0121032, -0.0087251, 0.0091033

### Unstable ReLU Count (Linear/Conv2D Layers)
- layer_idx=0, type=LayerType.Linear, total=256, inp1_unstable=7, inp2_unstable=7, delta_unstable=249
- layer_idx=2, type=LayerType.Linear, total=256, inp1_unstable=10, inp2_unstable=10, delta_unstable=256
- layer_idx=4, type=LayerType.Linear, total=256, inp1_unstable=1, inp2_unstable=1, delta_unstable=256
- layer_idx=6, type=LayerType.Linear, total=256, inp1_unstable=0, inp2_unstable=0, delta_unstable=256
- layer_idx=8, type=LayerType.Linear, total=10, inp1_unstable=4, inp2_unstable=4, delta_unstable=10

Time for backsubstitution: 1.39 seconds

### RS candidates at layer 3
type: RSZ, layer: 3, pos: 8
type: RSZ, layer: 3, pos: 13
type: RSZ, layer: 3, pos: 20
type: RSZ, layer: 3, pos: 35
type: RSZ, layer: 3, pos: 43
type: RSZ, layer: 3, pos: 48
type: RSZ, layer: 3, pos: 103
type: RSZ, layer: 3, pos: 133
type: RSZ, layer: 3, pos: 180
type: RSZ, layer: 3, pos: 183

Time for candidate selection: 0.33 seconds

### Candidate
type: RSZ, layer: 3, pos: 8

### Relational analysis RSZ of RS_RSZ1_RSZ1_RSZ1_RSZ2_RSZ1_RSZ2_RSZ2_RSZ1

#### Relational analysis RSZ result of RS_RSZ1_RSZ1_RSZ1_RSZ2_RSZ1_RSZ2_RSZ2_RSZ1
Status: Status.VERIFIED
Output dim: 9, lower bound: -0.0043047, upper bound: 0.0046810
time: 0.82 seconds

### Relational analysis RSZ of RS_RSZ1_RSZ1_RSZ1_RSZ2_RSZ1_RSZ2_RSZ2_RSZ2

#### Relational analysis RSZ result of RS_RSZ1_RSZ1_RSZ1_RSZ2_RSZ1_RSZ2_RSZ2_RSZ2
Status: Status.VERIFIED
Output dim: 9, lower bound: -0.0043047, upper bound: 0.0046810
time: 0.83 seconds

## BFS RS instance: RS_RSZ1_RSZ1_RSZ1_RSZ2_RSZ2_RSZ1_RSZ1

### Backsubstitution after applying RS history:
0: -0.0039095, -0.0003029, -0.0039095, -0.0003029, -0.0032191, 0.0032137
1: -0.0044907, 0.0040812, -0.0044907, 0.0040812, -0.0039675, 0.0040184
2: 0.0029569, 0.0099431, 0.0029569, 0.0099431, -0.0037193, 0.0037620
3: -0.0045519, -0.0034923, -0.0045519, -0.0034923, -0.0010595, 0.0010595
4: 0.0017018, 0.0078700, 0.0017018, 0.0078700, -0.0052420, 0.0052382
5: -0.0030341, 0.0031953, -0.0030341, 0.0031953, -0.0046948, 0.0047143
6: -0.0068516, -0.0033211, -0.0068516, -0.0033211, -0.0033362, 0.0033397
7: -0.0021570, 0.0038259, -0.0021570, 0.0038259, -0.0057042, 0.0056956
8: -0.0008533, 0.0000782, -0.0008533, 0.0000782, -0.0008497, 0.0008646
9: 0.9973557, 1.0121032, 0.9973557, 1.0121032, -0.0089227, 0.0088819

### Unstable ReLU Count (Linear/Conv2D Layers)
- layer_idx=0, type=LayerType.Linear, total=256, inp1_unstable=7, inp2_unstable=7, delta_unstable=249
- layer_idx=2, type=LayerType.Linear, total=256, inp1_unstable=10, inp2_unstable=10, delta_unstable=256
- layer_idx=4, type=LayerType.Linear, total=256, inp1_unstable=1, inp2_unstable=1, delta_unstable=256
- layer_idx=6, type=LayerType.Linear, total=256, inp1_unstable=0, inp2_unstable=0, delta_unstable=256
- layer_idx=8, type=LayerType.Linear, total=10, inp1_unstable=4, inp2_unstable=4, delta_unstable=10

Time for backsubstitution: 1.35 seconds

### RS candidates at layer 3
type: RSZ, layer: 3, pos: 8
type: RSZ, layer: 3, pos: 13
type: RSZ, layer: 3, pos: 20
type: RSZ, layer: 3, pos: 35
type: RSZ, layer: 3, pos: 43
type: RSZ, layer: 3, pos: 48
type: RSZ, layer: 3, pos: 103
type: RSZ, layer: 3, pos: 133
type: RSZ, layer: 3, pos: 180
type: RSZ, layer: 3, pos: 183

Time for candidate selection: 0.28 seconds

### Candidate
type: RSZ, layer: 3, pos: 8

### Relational analysis RSZ of RS_RSZ1_RSZ1_RSZ1_RSZ2_RSZ2_RSZ1_RSZ1_RSZ1

#### Relational analysis RSZ result of RS_RSZ1_RSZ1_RSZ1_RSZ2_RSZ2_RSZ1_RSZ1_RSZ1
Status: Status.VERIFIED
Output dim: 9, lower bound: -0.0044883, upper bound: 0.0044006
time: 0.78 seconds

### Relational analysis RSZ of RS_RSZ1_RSZ1_RSZ1_RSZ2_RSZ2_RSZ1_RSZ1_RSZ2

#### Relational analysis RSZ result of RS_RSZ1_RSZ1_RSZ1_RSZ2_RSZ2_RSZ1_RSZ1_RSZ2
Status: Status.VERIFIED
Output dim: 9, lower bound: -0.0044883, upper bound: 0.0044006
time: 0.80 seconds

## BFS RS instance: RS_RSZ1_RSZ1_RSZ1_RSZ2_RSZ2_RSZ1_RSZ2

### Backsubstitution after applying RS history:
0: -0.0039095, -0.0003029, -0.0039095, -0.0003029, -0.0032092, 0.0032312
1: -0.0044907, 0.0040812, -0.0044907, 0.0040812, -0.0040218, 0.0039858
2: 0.0029569, 0.0099431, 0.0029569, 0.0099431, -0.0037581, 0.0037371
3: -0.0045519, -0.0034923, -0.0045519, -0.0034923, -0.0010595, 0.0010595
4: 0.0017018, 0.0078700, 0.0017018, 0.0078700, -0.0052341, 0.0052522
5: -0.0030341, 0.0031953, -0.0030341, 0.0031953, -0.0047087, 0.0047067
6: -0.0068516, -0.0033211, -0.0068516, -0.0033211, -0.0033407, 0.0033371
7: -0.0021570, 0.0038259, -0.0021570, 0.0038259, -0.0056945, 0.0057122
8: -0.0008533, 0.0000782, -0.0008533, 0.0000782, -0.0008500, 0.0008644
9: 0.9973557, 1.0121032, 0.9973557, 1.0121032, -0.0088861, 0.0089466

### Unstable ReLU Count (Linear/Conv2D Layers)
- layer_idx=0, type=LayerType.Linear, total=256, inp1_unstable=7, inp2_unstable=7, delta_unstable=249
- layer_idx=2, type=LayerType.Linear, total=256, inp1_unstable=10, inp2_unstable=10, delta_unstable=256
- layer_idx=4, type=LayerType.Linear, total=256, inp1_unstable=1, inp2_unstable=1, delta_unstable=256
- layer_idx=6, type=LayerType.Linear, total=256, inp1_unstable=0, inp2_unstable=0, delta_unstable=256
- layer_idx=8, type=LayerType.Linear, total=10, inp1_unstable=4, inp2_unstable=4, delta_unstable=10

Time for backsubstitution: 1.39 seconds

### RS candidates at layer 3
type: RSZ, layer: 3, pos: 8
type: RSZ, layer: 3, pos: 13
type: RSZ, layer: 3, pos: 20
type: RSZ, layer: 3, pos: 35
type: RSZ, layer: 3, pos: 43
type: RSZ, layer: 3, pos: 48
type: RSZ, layer: 3, pos: 103
type: RSZ, layer: 3, pos: 133
type: RSZ, layer: 3, pos: 180
type: RSZ, layer: 3, pos: 183

Time for candidate selection: 0.27 seconds

### Candidate
type: RSZ, layer: 3, pos: 8

### Relational analysis RSZ of RS_RSZ1_RSZ1_RSZ1_RSZ2_RSZ2_RSZ1_RSZ2_RSZ1

#### Relational analysis RSZ result of RS_RSZ1_RSZ1_RSZ1_RSZ2_RSZ2_RSZ1_RSZ2_RSZ1
Status: Status.VERIFIED
Output dim: 9, lower bound: -0.0044282, upper bound: 0.0045563
time: 0.88 seconds

### Relational analysis RSZ of RS_RSZ1_RSZ1_RSZ1_RSZ2_RSZ2_RSZ1_RSZ2_RSZ2

#### Relational analysis RSZ result of RS_RSZ1_RSZ1_RSZ1_RSZ2_RSZ2_RSZ1_RSZ2_RSZ2
Status: Status.VERIFIED
Output dim: 9, lower bound: -0.0044284, upper bound: 0.0045563
time: 0.87 seconds

## BFS RS instance: RS_RSZ1_RSZ1_RSZ1_RSZ2_RSZ2_RSZ2_RSZ1

### Backsubstitution after applying RS history:
0: -0.0039095, -0.0003029, -0.0039095, -0.0003029, -0.0032189, 0.0032140
1: -0.0044907, 0.0040812, -0.0044907, 0.0040812, -0.0039677, 0.0040184
2: 0.0029569, 0.0099431, 0.0029569, 0.0099431, -0.0037189, 0.0037614
3: -0.0045519, -0.0034923, -0.0045519, -0.0034923, -0.0010595, 0.0010595
4: 0.0017018, 0.0078700, 0.0017018, 0.0078700, -0.0052418, 0.0052385
5: -0.0030341, 0.0031953, -0.0030341, 0.0031953, -0.0046960, 0.0047135
6: -0.0068516, -0.0033211, -0.0068516, -0.0033211, -0.0033362, 0.0033397
7: -0.0021570, 0.0038259, -0.0021570, 0.0038259, -0.0057037, 0.0056971
8: -0.0008533, 0.0000782, -0.0008533, 0.0000782, -0.0008492, 0.0008669
9: 0.9973557, 1.0121032, 0.9973557, 1.0121032, -0.0089225, 0.0088823

### Unstable ReLU Count (Linear/Conv2D Layers)
- layer_idx=0, type=LayerType.Linear, total=256, inp1_unstable=7, inp2_unstable=7, delta_unstable=249
- layer_idx=2, type=LayerType.Linear, total=256, inp1_unstable=10, inp2_unstable=10, delta_unstable=256
- layer_idx=4, type=LayerType.Linear, total=256, inp1_unstable=1, inp2_unstable=1, delta_unstable=256
- layer_idx=6, type=LayerType.Linear, total=256, inp1_unstable=0, inp2_unstable=0, delta_unstable=256
- layer_idx=8, type=LayerType.Linear, total=10, inp1_unstable=4, inp2_unstable=4, delta_unstable=10

Time for backsubstitution: 1.35 seconds

### RS candidates at layer 3
type: RSZ, layer: 3, pos: 8
type: RSZ, layer: 3, pos: 13
type: RSZ, layer: 3, pos: 20
type: RSZ, layer: 3, pos: 35
type: RSZ, layer: 3, pos: 43
type: RSZ, layer: 3, pos: 48
type: RSZ, layer: 3, pos: 103
type: RSZ, layer: 3, pos: 133
type: RSZ, layer: 3, pos: 180
type: RSZ, layer: 3, pos: 183

Time for candidate selection: 0.31 seconds

### Candidate
type: RSZ, layer: 3, pos: 8

### Relational analysis RSZ of RS_RSZ1_RSZ1_RSZ1_RSZ2_RSZ2_RSZ2_RSZ1_RSZ1

#### Relational analysis RSZ result of RS_RSZ1_RSZ1_RSZ1_RSZ2_RSZ2_RSZ2_RSZ1_RSZ1
Status: Status.VERIFIED
Output dim: 9, lower bound: -0.0044883, upper bound: 0.0044006
time: 0.79 seconds

### Relational analysis RSZ of RS_RSZ1_RSZ1_RSZ1_RSZ2_RSZ2_RSZ2_RSZ1_RSZ2

#### Relational analysis RSZ result of RS_RSZ1_RSZ1_RSZ1_RSZ2_RSZ2_RSZ2_RSZ1_RSZ2
Status: Status.VERIFIED
Output dim: 9, lower bound: -0.0044883, upper bound: 0.0044006
time: 0.77 seconds

## BFS RS instance: RS_RSZ1_RSZ1_RSZ1_RSZ2_RSZ2_RSZ2_RSZ2

### Backsubstitution after applying RS history:
0: -0.0039095, -0.0003029, -0.0039095, -0.0003029, -0.0032089, 0.0032319
1: -0.0044907, 0.0040812, -0.0044907, 0.0040812, -0.0040219, 0.0039859
2: 0.0029569, 0.0099431, 0.0029569, 0.0099431, -0.0037584, 0.0037372
3: -0.0045519, -0.0034923, -0.0045519, -0.0034923, -0.0010595, 0.0010595
4: 0.0017018, 0.0078700, 0.0017018, 0.0078700, -0.0052337, 0.0052526
5: -0.0030341, 0.0031953, -0.0030341, 0.0031953, -0.0047104, 0.0047061
6: -0.0068516, -0.0033211, -0.0068516, -0.0033211, -0.0033407, 0.0033371
7: -0.0021570, 0.0038259, -0.0021570, 0.0038259, -0.0056949, 0.0057139
8: -0.0008533, 0.0000782, -0.0008533, 0.0000782, -0.0008496, 0.0008670
9: 0.9973557, 1.0121032, 0.9973557, 1.0121032, -0.0088859, 0.0089474

### Unstable ReLU Count (Linear/Conv2D Layers)
- layer_idx=0, type=LayerType.Linear, total=256, inp1_unstable=7, inp2_unstable=7, delta_unstable=249
- layer_idx=2, type=LayerType.Linear, total=256, inp1_unstable=10, inp2_unstable=10, delta_unstable=256
- layer_idx=4, type=LayerType.Linear, total=256, inp1_unstable=1, inp2_unstable=1, delta_unstable=256
- layer_idx=6, type=LayerType.Linear, total=256, inp1_unstable=0, inp2_unstable=0, delta_unstable=256
- layer_idx=8, type=LayerType.Linear, total=10, inp1_unstable=4, inp2_unstable=4, delta_unstable=10

Time for backsubstitution: 1.34 seconds

### RS candidates at layer 3
type: RSZ, layer: 3, pos: 8
type: RSZ, layer: 3, pos: 13
type: RSZ, layer: 3, pos: 20
type: RSZ, layer: 3, pos: 35
type: RSZ, layer: 3, pos: 43
type: RSZ, layer: 3, pos: 48
type: RSZ, layer: 3, pos: 103
type: RSZ, layer: 3, pos: 133
type: RSZ, layer: 3, pos: 180
type: RSZ, layer: 3, pos: 183

Time for candidate selection: 0.29 seconds

### Candidate
type: RSZ, layer: 3, pos: 8

### Relational analysis RSZ of RS_RSZ1_RSZ1_RSZ1_RSZ2_RSZ2_RSZ2_RSZ2_RSZ1

#### Relational analysis RSZ result of RS_RSZ1_RSZ1_RSZ1_RSZ2_RSZ2_RSZ2_RSZ2_RSZ1
Status: Status.VERIFIED
Output dim: 9, lower bound: -0.0044282, upper bound: 0.0045563
time: 0.87 seconds

### Relational analysis RSZ of RS_RSZ1_RSZ1_RSZ1_RSZ2_RSZ2_RSZ2_RSZ2_RSZ2

#### Relational analysis RSZ result of RS_RSZ1_RSZ1_RSZ1_RSZ2_RSZ2_RSZ2_RSZ2_RSZ2
Status: Status.VERIFIED
Output dim: 9, lower bound: -0.0044284, upper bound: 0.0045563
time: 0.91 seconds

## BFS RS instance: RS_RSZ1_RSZ1_RSZ2_RSZ1_RSZ1_RSZ1_RSZ1

### Backsubstitution after applying RS history:
0: -0.0039095, -0.0003029, -0.0039095, -0.0003029, -0.0032344, 0.0032052
1: -0.0044907, 0.0040812, -0.0044907, 0.0040812, -0.0039435, 0.0040479
2: 0.0029569, 0.0099431, 0.0029569, 0.0099431, -0.0037079, 0.0037768
3: -0.0045519, -0.0034923, -0.0045519, -0.0034923, -0.0010595, 0.0010595
4: 0.0017018, 0.0078700, 0.0017018, 0.0078700, -0.0052541, 0.0052315
5: -0.0030341, 0.0031953, -0.0030341, 0.0031953, -0.0046862, 0.0047251
6: -0.0068516, -0.0033211, -0.0068516, -0.0033211, -0.0033342, 0.0033427
7: -0.0021570, 0.0038259, -0.0021570, 0.0038259, -0.0056981, 0.0057025
8: -0.0008533, 0.0000782, -0.0008533, 0.0000782, -0.0008369, 0.0008799
9: 0.9973557, 1.0121032, 0.9973557, 1.0121032, -0.0089718, 0.0088499

### Unstable ReLU Count (Linear/Conv2D Layers)
- layer_idx=0, type=LayerType.Linear, total=256, inp1_unstable=7, inp2_unstable=7, delta_unstable=249
- layer_idx=2, type=LayerType.Linear, total=256, inp1_unstable=10, inp2_unstable=10, delta_unstable=256
- layer_idx=4, type=LayerType.Linear, total=256, inp1_unstable=1, inp2_unstable=1, delta_unstable=256
- layer_idx=6, type=LayerType.Linear, total=256, inp1_unstable=0, inp2_unstable=0, delta_unstable=256
- layer_idx=8, type=LayerType.Linear, total=10, inp1_unstable=4, inp2_unstable=4, delta_unstable=10

Time for backsubstitution: 1.50 seconds

### RS candidates at layer 3
type: RSZ, layer: 3, pos: 8
type: RSZ, layer: 3, pos: 13
type: RSZ, layer: 3, pos: 20
type: RSZ, layer: 3, pos: 35
type: RSZ, layer: 3, pos: 43
type: RSZ, layer: 3, pos: 48
type: RSZ, layer: 3, pos: 103
type: RSZ, layer: 3, pos: 133
type: RSZ, layer: 3, pos: 180
type: RSZ, layer: 3, pos: 183

Time for candidate selection: 0.29 seconds

### Candidate
type: RSZ, layer: 3, pos: 8

### Relational analysis RSZ of RS_RSZ1_RSZ1_RSZ2_RSZ1_RSZ1_RSZ1_RSZ1_RSZ1

#### Relational analysis RSZ result of RS_RSZ1_RSZ1_RSZ2_RSZ1_RSZ1_RSZ1_RSZ1_RSZ1
Status: Status.VERIFIED
Output dim: 9, lower bound: -0.0046266, upper bound: 0.0042737
time: 0.84 seconds

### Relational analysis RSZ of RS_RSZ1_RSZ1_RSZ2_RSZ1_RSZ1_RSZ1_RSZ1_RSZ2

#### Relational analysis RSZ result of RS_RSZ1_RSZ1_RSZ2_RSZ1_RSZ1_RSZ1_RSZ1_RSZ2
Status: Status.VERIFIED
Output dim: 9, lower bound: -0.0046266, upper bound: 0.0042737
time: 0.84 seconds

## BFS RS instance: RS_RSZ1_RSZ1_RSZ2_RSZ1_RSZ1_RSZ1_RSZ2

### Backsubstitution after applying RS history:
0: -0.0039095, -0.0003029, -0.0039095, -0.0003029, -0.0032165, 0.0032155
1: -0.0044907, 0.0040812, -0.0044907, 0.0040812, -0.0039763, 0.0039935
2: 0.0029569, 0.0099431, 0.0029569, 0.0099431, -0.0037327, 0.0037375
3: -0.0045519, -0.0034923, -0.0045519, -0.0034923, -0.0010595, 0.0010595
4: 0.0017018, 0.0078700, 0.0017018, 0.0078700, -0.0052400, 0.0052398
5: -0.0030341, 0.0031953, -0.0030341, 0.0031953, -0.0046941, 0.0047113
6: -0.0068516, -0.0033211, -0.0068516, -0.0033211, -0.0033368, 0.0033382
7: -0.0021570, 0.0038259, -0.0021570, 0.0038259, -0.0056820, 0.0057128
8: -0.0008533, 0.0000782, -0.0008533, 0.0000782, -0.0008372, 0.0008794
9: 0.9973557, 1.0121032, 0.9973557, 1.0121032, -0.0089059, 0.0088874

### Unstable ReLU Count (Linear/Conv2D Layers)
- layer_idx=0, type=LayerType.Linear, total=256, inp1_unstable=7, inp2_unstable=7, delta_unstable=249
- layer_idx=2, type=LayerType.Linear, total=256, inp1_unstable=10, inp2_unstable=10, delta_unstable=256
- layer_idx=4, type=LayerType.Linear, total=256, inp1_unstable=1, inp2_unstable=1, delta_unstable=256
- layer_idx=6, type=LayerType.Linear, total=256, inp1_unstable=0, inp2_unstable=0, delta_unstable=256
- layer_idx=8, type=LayerType.Linear, total=10, inp1_unstable=4, inp2_unstable=4, delta_unstable=10

Time for backsubstitution: 1.43 seconds

### RS candidates at layer 3
type: RSZ, layer: 3, pos: 8
type: RSZ, layer: 3, pos: 13
type: RSZ, layer: 3, pos: 20
type: RSZ, layer: 3, pos: 35
type: RSZ, layer: 3, pos: 43
type: RSZ, layer: 3, pos: 48
type: RSZ, layer: 3, pos: 103
type: RSZ, layer: 3, pos: 133
type: RSZ, layer: 3, pos: 180
type: RSZ, layer: 3, pos: 183

Time for candidate selection: 0.34 seconds

### Candidate
type: RSZ, layer: 3, pos: 8

### Relational analysis RSZ of RS_RSZ1_RSZ1_RSZ2_RSZ1_RSZ1_RSZ1_RSZ2_RSZ1

#### Relational analysis RSZ result of RS_RSZ1_RSZ1_RSZ2_RSZ1_RSZ1_RSZ1_RSZ2_RSZ1
Status: Status.VERIFIED
Output dim: 9, lower bound: -0.0044941, upper bound: 0.0043335
time: 0.81 seconds

### Relational analysis RSZ of RS_RSZ1_RSZ1_RSZ2_RSZ1_RSZ1_RSZ1_RSZ2_RSZ2

#### Relational analysis RSZ result of RS_RSZ1_RSZ1_RSZ2_RSZ1_RSZ1_RSZ1_RSZ2_RSZ2
Status: Status.VERIFIED
Output dim: 9, lower bound: -0.0044941, upper bound: 0.0043335
time: 0.80 seconds

## BFS RS instance: RS_RSZ1_RSZ1_RSZ2_RSZ1_RSZ1_RSZ2_RSZ1

### Backsubstitution after applying RS history:
0: -0.0039095, -0.0003029, -0.0039095, -0.0003029, -0.0032339, 0.0032055
1: -0.0044907, 0.0040812, -0.0044907, 0.0040812, -0.0039435, 0.0040477
2: 0.0029569, 0.0099431, 0.0029569, 0.0099431, -0.0037080, 0.0037762
3: -0.0045519, -0.0034923, -0.0045519, -0.0034923, -0.0010595, 0.0010595
4: 0.0017018, 0.0078700, 0.0017018, 0.0078700, -0.0052537, 0.0052317
5: -0.0030341, 0.0031953, -0.0030341, 0.0031953, -0.0046867, 0.0047236
6: -0.0068516, -0.0033211, -0.0068516, -0.0033211, -0.0033342, 0.0033427
7: -0.0021570, 0.0038259, -0.0021570, 0.0038259, -0.0056968, 0.0057033
8: -0.0008533, 0.0000782, -0.0008533, 0.0000782, -0.0008374, 0.0008815
9: 0.9973557, 1.0121032, 0.9973557, 1.0121032, -0.0089714, 0.0088498

### Unstable ReLU Count (Linear/Conv2D Layers)
- layer_idx=0, type=LayerType.Linear, total=256, inp1_unstable=7, inp2_unstable=7, delta_unstable=249
- layer_idx=2, type=LayerType.Linear, total=256, inp1_unstable=10, inp2_unstable=10, delta_unstable=256
- layer_idx=4, type=LayerType.Linear, total=256, inp1_unstable=1, inp2_unstable=1, delta_unstable=256
- layer_idx=6, type=LayerType.Linear, total=256, inp1_unstable=0, inp2_unstable=0, delta_unstable=256
- layer_idx=8, type=LayerType.Linear, total=10, inp1_unstable=4, inp2_unstable=4, delta_unstable=10

Time for backsubstitution: 1.38 seconds

### RS candidates at layer 3
type: RSZ, layer: 3, pos: 8
type: RSZ, layer: 3, pos: 13
type: RSZ, layer: 3, pos: 20
type: RSZ, layer: 3, pos: 35
type: RSZ, layer: 3, pos: 43
type: RSZ, layer: 3, pos: 48
type: RSZ, layer: 3, pos: 103
type: RSZ, layer: 3, pos: 133
type: RSZ, layer: 3, pos: 180
type: RSZ, layer: 3, pos: 183

Time for candidate selection: 0.30 seconds

### Candidate
type: RSZ, layer: 3, pos: 8

### Relational analysis RSZ of RS_RSZ1_RSZ1_RSZ2_RSZ1_RSZ1_RSZ2_RSZ1_RSZ1

#### Relational analysis RSZ result of RS_RSZ1_RSZ1_RSZ2_RSZ1_RSZ1_RSZ2_RSZ1_RSZ1
Status: Status.VERIFIED
Output dim: 9, lower bound: -0.0046266, upper bound: 0.0042737
time: 0.84 seconds

### Relational analysis RSZ of RS_RSZ1_RSZ1_RSZ2_RSZ1_RSZ1_RSZ2_RSZ1_RSZ2

#### Relational analysis RSZ result of RS_RSZ1_RSZ1_RSZ2_RSZ1_RSZ1_RSZ2_RSZ1_RSZ2
Status: Status.VERIFIED
Output dim: 9, lower bound: -0.0046266, upper bound: 0.0042737
time: 0.82 seconds

## BFS RS instance: RS_RSZ1_RSZ1_RSZ2_RSZ1_RSZ1_RSZ2_RSZ2

### Backsubstitution after applying RS history:
0: -0.0039095, -0.0003029, -0.0039095, -0.0003029, -0.0032162, 0.0032156
1: -0.0044907, 0.0040812, -0.0044907, 0.0040812, -0.0039763, 0.0039934
2: 0.0029569, 0.0099431, 0.0029569, 0.0099431, -0.0037332, 0.0037377
3: -0.0045519, -0.0034923, -0.0045519, -0.0034923, -0.0010595, 0.0010595
4: 0.0017018, 0.0078700, 0.0017018, 0.0078700, -0.0052397, 0.0052399
5: -0.0030341, 0.0031953, -0.0030341, 0.0031953, -0.0046949, 0.0047107
6: -0.0068516, -0.0033211, -0.0068516, -0.0033211, -0.0033368, 0.0033382
7: -0.0021570, 0.0038259, -0.0021570, 0.0038259, -0.0056822, 0.0057139
8: -0.0008533, 0.0000782, -0.0008533, 0.0000782, -0.0008378, 0.0008811
9: 0.9973557, 1.0121032, 0.9973557, 1.0121032, -0.0089055, 0.0088877

### Unstable ReLU Count (Linear/Conv2D Layers)
- layer_idx=0, type=LayerType.Linear, total=256, inp1_unstable=7, inp2_unstable=7, delta_unstable=249
- layer_idx=2, type=LayerType.Linear, total=256, inp1_unstable=10, inp2_unstable=10, delta_unstable=256
- layer_idx=4, type=LayerType.Linear, total=256, inp1_unstable=1, inp2_unstable=1, delta_unstable=256
- layer_idx=6, type=LayerType.Linear, total=256, inp1_unstable=0, inp2_unstable=0, delta_unstable=256
- layer_idx=8, type=LayerType.Linear, total=10, inp1_unstable=4, inp2_unstable=4, delta_unstable=10

Time for backsubstitution: 1.34 seconds

### RS candidates at layer 3
type: RSZ, layer: 3, pos: 8
type: RSZ, layer: 3, pos: 13
type: RSZ, layer: 3, pos: 20
type: RSZ, layer: 3, pos: 35
type: RSZ, layer: 3, pos: 43
type: RSZ, layer: 3, pos: 48
type: RSZ, layer: 3, pos: 103
type: RSZ, layer: 3, pos: 133
type: RSZ, layer: 3, pos: 180
type: RSZ, layer: 3, pos: 183

Time for candidate selection: 0.28 seconds

### Candidate
type: RSZ, layer: 3, pos: 8

### Relational analysis RSZ of RS_RSZ1_RSZ1_RSZ2_RSZ1_RSZ1_RSZ2_RSZ2_RSZ1

#### Relational analysis RSZ result of RS_RSZ1_RSZ1_RSZ2_RSZ1_RSZ1_RSZ2_RSZ2_RSZ1
Status: Status.VERIFIED
Output dim: 9, lower bound: -0.0044941, upper bound: 0.0043335
time: 0.78 seconds

### Relational analysis RSZ of RS_RSZ1_RSZ1_RSZ2_RSZ1_RSZ1_RSZ2_RSZ2_RSZ2

#### Relational analysis RSZ result of RS_RSZ1_RSZ1_RSZ2_RSZ1_RSZ1_RSZ2_RSZ2_RSZ2
Status: Status.VERIFIED
Output dim: 9, lower bound: -0.0044941, upper bound: 0.0043335
time: 0.79 seconds

## BFS RS instance: RS_RSZ1_RSZ1_RSZ2_RSZ1_RSZ2_RSZ1_RSZ1

### Backsubstitution after applying RS history:
0: -0.0039095, -0.0003029, -0.0039095, -0.0003029, -0.0032854, 0.0031589
1: -0.0044907, 0.0040812, -0.0044907, 0.0040812, -0.0038243, 0.0041649
2: 0.0029569, 0.0099431, 0.0029569, 0.0099431, -0.0036207, 0.0038568
3: -0.0045519, -0.0034923, -0.0045519, -0.0034923, -0.0010595, 0.0010595
4: 0.0017018, 0.0078700, 0.0017018, 0.0078700, -0.0052923, 0.0051962
5: -0.0030341, 0.0031953, -0.0030341, 0.0031953, -0.0046341, 0.0047894
6: -0.0068516, -0.0033211, -0.0068516, -0.0033211, -0.0033239, 0.0033530
7: -0.0021570, 0.0038259, -0.0021570, 0.0038259, -0.0057248, 0.0056762
8: -0.0008533, 0.0000782, -0.0008533, 0.0000782, -0.0008305, 0.0008862
9: 0.9973557, 1.0121032, 0.9973557, 1.0121032, -0.0091302, 0.0086970

### Unstable ReLU Count (Linear/Conv2D Layers)
- layer_idx=0, type=LayerType.Linear, total=256, inp1_unstable=7, inp2_unstable=7, delta_unstable=249
- layer_idx=2, type=LayerType.Linear, total=256, inp1_unstable=10, inp2_unstable=10, delta_unstable=256
- layer_idx=4, type=LayerType.Linear, total=256, inp1_unstable=1, inp2_unstable=1, delta_unstable=256
- layer_idx=6, type=LayerType.Linear, total=256, inp1_unstable=0, inp2_unstable=0, delta_unstable=256
- layer_idx=8, type=LayerType.Linear, total=10, inp1_unstable=4, inp2_unstable=4, delta_unstable=10

Time for backsubstitution: 1.34 seconds

### RS candidates at layer 3
type: RSZ, layer: 3, pos: 8
type: RSZ, layer: 3, pos: 13
type: RSZ, layer: 3, pos: 20
type: RSZ, layer: 3, pos: 35
type: RSZ, layer: 3, pos: 43
type: RSZ, layer: 3, pos: 48
type: RSZ, layer: 3, pos: 103
type: RSZ, layer: 3, pos: 133
type: RSZ, layer: 3, pos: 180
type: RSZ, layer: 3, pos: 183

Time for candidate selection: 0.28 seconds

### Candidate
type: RSZ, layer: 3, pos: 8

### Relational analysis RSZ of RS_RSZ1_RSZ1_RSZ2_RSZ1_RSZ2_RSZ1_RSZ1_RSZ1

#### Relational analysis RSZ result of RS_RSZ1_RSZ1_RSZ2_RSZ1_RSZ2_RSZ1_RSZ1_RSZ1
Status: Status.VERIFIED
Output dim: 9, lower bound: -0.0047614, upper bound: 0.0041783
time: 0.85 seconds

### Relational analysis RSZ of RS_RSZ1_RSZ1_RSZ2_RSZ1_RSZ2_RSZ1_RSZ1_RSZ2

#### Relational analysis RSZ result of RS_RSZ1_RSZ1_RSZ2_RSZ1_RSZ2_RSZ1_RSZ1_RSZ2
Status: Status.VERIFIED
Output dim: 9, lower bound: -0.0047614, upper bound: 0.0041783
time: 0.84 seconds

## BFS RS instance: RS_RSZ1_RSZ1_RSZ2_RSZ1_RSZ2_RSZ1_RSZ2

### Backsubstitution after applying RS history:
0: -0.0039095, -0.0003029, -0.0039095, -0.0003029, -0.0032724, 0.0031728
1: -0.0044907, 0.0040812, -0.0044907, 0.0040812, -0.0038688, 0.0041236
2: 0.0029569, 0.0099431, 0.0029569, 0.0099431, -0.0036526, 0.0038272
3: -0.0045519, -0.0034923, -0.0045519, -0.0034923, -0.0010595, 0.0010595
4: 0.0017018, 0.0078700, 0.0017018, 0.0078700, -0.0052821, 0.0052073
5: -0.0030341, 0.0031953, -0.0030341, 0.0031953, -0.0046453, 0.0047784
6: -0.0068516, -0.0033211, -0.0068516, -0.0033211, -0.0033276, 0.0033497
7: -0.0021570, 0.0038259, -0.0021570, 0.0038259, -0.0057125, 0.0056892
8: -0.0008533, 0.0000782, -0.0008533, 0.0000782, -0.0008310, 0.0008859
9: 0.9973557, 1.0121032, 0.9973557, 1.0121032, -0.0090818, 0.0087509

### Unstable ReLU Count (Linear/Conv2D Layers)
- layer_idx=0, type=LayerType.Linear, total=256, inp1_unstable=7, inp2_unstable=7, delta_unstable=249
- layer_idx=2, type=LayerType.Linear, total=256, inp1_unstable=10, inp2_unstable=10, delta_unstable=256
- layer_idx=4, type=LayerType.Linear, total=256, inp1_unstable=1, inp2_unstable=1, delta_unstable=256
- layer_idx=6, type=LayerType.Linear, total=256, inp1_unstable=0, inp2_unstable=0, delta_unstable=256
- layer_idx=8, type=LayerType.Linear, total=10, inp1_unstable=4, inp2_unstable=4, delta_unstable=10

Time for backsubstitution: 1.36 seconds

### RS candidates at layer 3
type: RSZ, layer: 3, pos: 8
type: RSZ, layer: 3, pos: 13
type: RSZ, layer: 3, pos: 20
type: RSZ, layer: 3, pos: 35
type: RSZ, layer: 3, pos: 43
type: RSZ, layer: 3, pos: 48
type: RSZ, layer: 3, pos: 103
type: RSZ, layer: 3, pos: 133
type: RSZ, layer: 3, pos: 180
type: RSZ, layer: 3, pos: 183

Time for candidate selection: 0.28 seconds

### Candidate
type: RSZ, layer: 3, pos: 8

### Relational analysis RSZ of RS_RSZ1_RSZ1_RSZ2_RSZ1_RSZ2_RSZ1_RSZ2_RSZ1

#### Relational analysis RSZ result of RS_RSZ1_RSZ1_RSZ2_RSZ1_RSZ2_RSZ1_RSZ2_RSZ1
Status: Status.VERIFIED
Output dim: 9, lower bound: -0.0046537, upper bound: 0.0042358
time: 0.88 seconds

### Relational analysis RSZ of RS_RSZ1_RSZ1_RSZ2_RSZ1_RSZ2_RSZ1_RSZ2_RSZ2

#### Relational analysis RSZ result of RS_RSZ1_RSZ1_RSZ2_RSZ1_RSZ2_RSZ1_RSZ2_RSZ2
Status: Status.VERIFIED
Output dim: 9, lower bound: -0.0046537, upper bound: 0.0042358
time: 0.91 seconds

## BFS RS instance: RS_RSZ1_RSZ1_RSZ2_RSZ1_RSZ2_RSZ2_RSZ1

### Backsubstitution after applying RS history:
0: -0.0039095, -0.0003029, -0.0039095, -0.0003029, -0.0032850, 0.0031596
1: -0.0044907, 0.0040812, -0.0044907, 0.0040812, -0.0038245, 0.0041648
2: 0.0029569, 0.0099431, 0.0029569, 0.0099431, -0.0036209, 0.0038569
3: -0.0045519, -0.0034923, -0.0045519, -0.0034923, -0.0010595, 0.0010595
4: 0.0017018, 0.0078700, 0.0017018, 0.0078700, -0.0052920, 0.0051966
5: -0.0030341, 0.0031953, -0.0030341, 0.0031953, -0.0046360, 0.0047894
6: -0.0068516, -0.0033211, -0.0068516, -0.0033211, -0.0033240, 0.0033530
7: -0.0021570, 0.0038259, -0.0021570, 0.0038259, -0.0057247, 0.0056763
8: -0.0008533, 0.0000782, -0.0008533, 0.0000782, -0.0008308, 0.0008880
9: 0.9973557, 1.0121032, 0.9973557, 1.0121032, -0.0091299, 0.0086976

### Unstable ReLU Count (Linear/Conv2D Layers)
- layer_idx=0, type=LayerType.Linear, total=256, inp1_unstable=7, inp2_unstable=7, delta_unstable=249
- layer_idx=2, type=LayerType.Linear, total=256, inp1_unstable=10, inp2_unstable=10, delta_unstable=256
- layer_idx=4, type=LayerType.Linear, total=256, inp1_unstable=1, inp2_unstable=1, delta_unstable=256
- layer_idx=6, type=LayerType.Linear, total=256, inp1_unstable=0, inp2_unstable=0, delta_unstable=256
- layer_idx=8, type=LayerType.Linear, total=10, inp1_unstable=4, inp2_unstable=4, delta_unstable=10

Time for backsubstitution: 1.46 seconds

### RS candidates at layer 3
type: RSZ, layer: 3, pos: 8
type: RSZ, layer: 3, pos: 13
type: RSZ, layer: 3, pos: 20
type: RSZ, layer: 3, pos: 35
type: RSZ, layer: 3, pos: 43
type: RSZ, layer: 3, pos: 48
type: RSZ, layer: 3, pos: 103
type: RSZ, layer: 3, pos: 133
type: RSZ, layer: 3, pos: 180
type: RSZ, layer: 3, pos: 183

Time for candidate selection: 0.28 seconds

### Candidate
type: RSZ, layer: 3, pos: 8

### Relational analysis RSZ of RS_RSZ1_RSZ1_RSZ2_RSZ1_RSZ2_RSZ2_RSZ1_RSZ1

#### Relational analysis RSZ result of RS_RSZ1_RSZ1_RSZ2_RSZ1_RSZ2_RSZ2_RSZ1_RSZ1
Status: Status.VERIFIED
Output dim: 9, lower bound: -0.0047614, upper bound: 0.0041783
time: 0.84 seconds

### Relational analysis RSZ of RS_RSZ1_RSZ1_RSZ2_RSZ1_RSZ2_RSZ2_RSZ1_RSZ2

#### Relational analysis RSZ result of RS_RSZ1_RSZ1_RSZ2_RSZ1_RSZ2_RSZ2_RSZ1_RSZ2
Status: Status.VERIFIED
Output dim: 9, lower bound: -0.0047614, upper bound: 0.0041783
time: 0.85 seconds

## BFS RS instance: RS_RSZ1_RSZ1_RSZ2_RSZ1_RSZ2_RSZ2_RSZ2

### Backsubstitution after applying RS history:
0: -0.0039095, -0.0003029, -0.0039095, -0.0003029, -0.0032723, 0.0031737
1: -0.0044907, 0.0040812, -0.0044907, 0.0040812, -0.0038692, 0.0041239
2: 0.0029569, 0.0099431, 0.0029569, 0.0099431, -0.0036532, 0.0038275
3: -0.0045519, -0.0034923, -0.0045519, -0.0034923, -0.0010595, 0.0010595
4: 0.0017018, 0.0078700, 0.0017018, 0.0078700, -0.0052820, 0.0052079
5: -0.0030341, 0.0031953, -0.0030341, 0.0031953, -0.0046472, 0.0047785
6: -0.0068516, -0.0033211, -0.0068516, -0.0033211, -0.0033277, 0.0033497
7: -0.0021570, 0.0038259, -0.0021570, 0.0038259, -0.0057136, 0.0056930
8: -0.0008533, 0.0000782, -0.0008533, 0.0000782, -0.0008312, 0.0008878
9: 0.9973557, 1.0121032, 0.9973557, 1.0121032, -0.0090819, 0.0087520

### Unstable ReLU Count (Linear/Conv2D Layers)
- layer_idx=0, type=LayerType.Linear, total=256, inp1_unstable=7, inp2_unstable=7, delta_unstable=249
- layer_idx=2, type=LayerType.Linear, total=256, inp1_unstable=10, inp2_unstable=10, delta_unstable=256
- layer_idx=4, type=LayerType.Linear, total=256, inp1_unstable=1, inp2_unstable=1, delta_unstable=256
- layer_idx=6, type=LayerType.Linear, total=256, inp1_unstable=0, inp2_unstable=0, delta_unstable=256
- layer_idx=8, type=LayerType.Linear, total=10, inp1_unstable=4, inp2_unstable=4, delta_unstable=10

Time for backsubstitution: 1.42 seconds

### RS candidates at layer 3
type: RSZ, layer: 3, pos: 8
type: RSZ, layer: 3, pos: 13
type: RSZ, layer: 3, pos: 20
type: RSZ, layer: 3, pos: 35
type: RSZ, layer: 3, pos: 43
type: RSZ, layer: 3, pos: 48
type: RSZ, layer: 3, pos: 103
type: RSZ, layer: 3, pos: 133
type: RSZ, layer: 3, pos: 180
type: RSZ, layer: 3, pos: 183

Time for candidate selection: 0.28 seconds

### Candidate
type: RSZ, layer: 3, pos: 8

### Relational analysis RSZ of RS_RSZ1_RSZ1_RSZ2_RSZ1_RSZ2_RSZ2_RSZ2_RSZ1

#### Relational analysis RSZ result of RS_RSZ1_RSZ1_RSZ2_RSZ1_RSZ2_RSZ2_RSZ2_RSZ1
Status: Status.VERIFIED
Output dim: 9, lower bound: -0.0046537, upper bound: 0.0042358
time: 0.89 seconds

### Relational analysis RSZ of RS_RSZ1_RSZ1_RSZ2_RSZ1_RSZ2_RSZ2_RSZ2_RSZ2

#### Relational analysis RSZ result of RS_RSZ1_RSZ1_RSZ2_RSZ1_RSZ2_RSZ2_RSZ2_RSZ2
Status: Status.VERIFIED
Output dim: 9, lower bound: -0.0046537, upper bound: 0.0042358
time: 0.89 seconds

## BFS RS instance: RS_RSZ1_RSZ1_RSZ2_RSZ2_RSZ1_RSZ1_RSZ1

### Backsubstitution after applying RS history:
0: -0.0039095, -0.0003029, -0.0039095, -0.0003029, -0.0031587, 0.0032748
1: -0.0044907, 0.0040812, -0.0044907, 0.0040812, -0.0041029, 0.0038682
2: 0.0029569, 0.0099431, 0.0029569, 0.0099431, -0.0038103, 0.0036614
3: -0.0045519, -0.0034923, -0.0045519, -0.0034923, -0.0010595, 0.0010595
4: 0.0017018, 0.0078700, 0.0017018, 0.0078700, -0.0051961, 0.0052848
5: -0.0030341, 0.0031953, -0.0030341, 0.0031953, -0.0047626, 0.0046440
6: -0.0068516, -0.0033211, -0.0068516, -0.0033211, -0.0033484, 0.0033266
7: -0.0021570, 0.0038259, -0.0021570, 0.0038259, -0.0056711, 0.0057257
8: -0.0008533, 0.0000782, -0.0008533, 0.0000782, -0.0008454, 0.0008720
9: 0.9973557, 1.0121032, 0.9973557, 1.0121032, -0.0087249, 0.0090675

### Unstable ReLU Count (Linear/Conv2D Layers)
- layer_idx=0, type=LayerType.Linear, total=256, inp1_unstable=7, inp2_unstable=7, delta_unstable=249
- layer_idx=2, type=LayerType.Linear, total=256, inp1_unstable=10, inp2_unstable=10, delta_unstable=256
- layer_idx=4, type=LayerType.Linear, total=256, inp1_unstable=1, inp2_unstable=1, delta_unstable=256
- layer_idx=6, type=LayerType.Linear, total=256, inp1_unstable=0, inp2_unstable=0, delta_unstable=256
- layer_idx=8, type=LayerType.Linear, total=10, inp1_unstable=4, inp2_unstable=4, delta_unstable=10

Time for backsubstitution: 1.37 seconds

### RS candidates at layer 3
type: RSZ, layer: 3, pos: 8
type: RSZ, layer: 3, pos: 13
type: RSZ, layer: 3, pos: 20
type: RSZ, layer: 3, pos: 35
type: RSZ, layer: 3, pos: 43
type: RSZ, layer: 3, pos: 48
type: RSZ, layer: 3, pos: 103
type: RSZ, layer: 3, pos: 133
type: RSZ, layer: 3, pos: 180
type: RSZ, layer: 3, pos: 183

Time for candidate selection: 0.28 seconds

### Candidate
type: RSZ, layer: 3, pos: 8

### Relational analysis RSZ of RS_RSZ1_RSZ1_RSZ2_RSZ2_RSZ1_RSZ1_RSZ1_RSZ1

#### Relational analysis RSZ result of RS_RSZ1_RSZ1_RSZ2_RSZ2_RSZ1_RSZ1_RSZ1_RSZ1
Status: Status.VERIFIED
Output dim: 9, lower bound: -0.0043022, upper bound: 0.0045506
time: 0.81 seconds

### Relational analysis RSZ of RS_RSZ1_RSZ1_RSZ2_RSZ2_RSZ1_RSZ1_RSZ1_RSZ2

#### Relational analysis RSZ result of RS_RSZ1_RSZ1_RSZ2_RSZ2_RSZ1_RSZ1_RSZ1_RSZ2
Status: Status.VERIFIED
Output dim: 9, lower bound: -0.0043042, upper bound: 0.0045506
time: 0.84 seconds

## BFS RS instance: RS_RSZ1_RSZ1_RSZ2_RSZ2_RSZ1_RSZ1_RSZ2

### Backsubstitution after applying RS history:
0: -0.0039095, -0.0003029, -0.0039095, -0.0003029, -0.0031455, 0.0032868
1: -0.0044907, 0.0040812, -0.0044907, 0.0040812, -0.0041438, 0.0038244
2: 0.0029569, 0.0099431, 0.0029569, 0.0099431, -0.0038398, 0.0036299
3: -0.0045519, -0.0034923, -0.0045519, -0.0034923, -0.0010595, 0.0010595
4: 0.0017018, 0.0078700, 0.0017018, 0.0078700, -0.0051854, 0.0052944
5: -0.0030341, 0.0031953, -0.0030341, 0.0031953, -0.0047736, 0.0046334
6: -0.0068516, -0.0033211, -0.0068516, -0.0033211, -0.0033518, 0.0033231
7: -0.0021570, 0.0038259, -0.0021570, 0.0038259, -0.0056570, 0.0057385
8: -0.0008533, 0.0000782, -0.0008533, 0.0000782, -0.0008460, 0.0008719
9: 0.9973557, 1.0121032, 0.9973557, 1.0121032, -0.0086742, 0.0091175

### Unstable ReLU Count (Linear/Conv2D Layers)
- layer_idx=0, type=LayerType.Linear, total=256, inp1_unstable=7, inp2_unstable=7, delta_unstable=249
- layer_idx=2, type=LayerType.Linear, total=256, inp1_unstable=10, inp2_unstable=10, delta_unstable=256
- layer_idx=4, type=LayerType.Linear, total=256, inp1_unstable=1, inp2_unstable=1, delta_unstable=256
- layer_idx=6, type=LayerType.Linear, total=256, inp1_unstable=0, inp2_unstable=0, delta_unstable=256
- layer_idx=8, type=LayerType.Linear, total=10, inp1_unstable=4, inp2_unstable=4, delta_unstable=10

Time for backsubstitution: 1.42 seconds

### RS candidates at layer 3
type: RSZ, layer: 3, pos: 8
type: RSZ, layer: 3, pos: 13
type: RSZ, layer: 3, pos: 20
type: RSZ, layer: 3, pos: 35
type: RSZ, layer: 3, pos: 43
type: RSZ, layer: 3, pos: 48
type: RSZ, layer: 3, pos: 103
type: RSZ, layer: 3, pos: 133
type: RSZ, layer: 3, pos: 180
type: RSZ, layer: 3, pos: 183

Time for candidate selection: 0.28 seconds

### Candidate
type: RSZ, layer: 3, pos: 8

### Relational analysis RSZ of RS_RSZ1_RSZ1_RSZ2_RSZ2_RSZ1_RSZ1_RSZ2_RSZ1

#### Relational analysis RSZ result of RS_RSZ1_RSZ1_RSZ2_RSZ2_RSZ1_RSZ1_RSZ2_RSZ1
Status: Status.VERIFIED
Output dim: 9, lower bound: -0.0042473, upper bound: 0.0046823
time: 0.93 seconds

### Relational analysis RSZ of RS_RSZ1_RSZ1_RSZ2_RSZ2_RSZ1_RSZ1_RSZ2_RSZ2

#### Relational analysis RSZ result of RS_RSZ1_RSZ1_RSZ2_RSZ2_RSZ1_RSZ1_RSZ2_RSZ2
Status: Status.VERIFIED
Output dim: 9, lower bound: -0.0042482, upper bound: 0.0046823
time: 0.91 seconds

## BFS RS instance: RS_RSZ1_RSZ1_RSZ2_RSZ2_RSZ1_RSZ2_RSZ1

### Backsubstitution after applying RS history:
0: -0.0039095, -0.0003029, -0.0039095, -0.0003029, -0.0031580, 0.0032750
1: -0.0044907, 0.0040812, -0.0044907, 0.0040812, -0.0041026, 0.0038676
2: 0.0029569, 0.0099431, 0.0029569, 0.0099431, -0.0038100, 0.0036602
3: -0.0045519, -0.0034923, -0.0045519, -0.0034923, -0.0010595, 0.0010595
4: 0.0017018, 0.0078700, 0.0017018, 0.0078700, -0.0051956, 0.0052849
5: -0.0030341, 0.0031953, -0.0030341, 0.0031953, -0.0047635, 0.0046416
6: -0.0068516, -0.0033211, -0.0068516, -0.0033211, -0.0033484, 0.0033265
7: -0.0021570, 0.0038259, -0.0021570, 0.0038259, -0.0056698, 0.0057263
8: -0.0008533, 0.0000782, -0.0008533, 0.0000782, -0.0008456, 0.0008724
9: 0.9973557, 1.0121032, 0.9973557, 1.0121032, -0.0087240, 0.0090674

### Unstable ReLU Count (Linear/Conv2D Layers)
- layer_idx=0, type=LayerType.Linear, total=256, inp1_unstable=7, inp2_unstable=7, delta_unstable=249
- layer_idx=2, type=LayerType.Linear, total=256, inp1_unstable=10, inp2_unstable=10, delta_unstable=256
- layer_idx=4, type=LayerType.Linear, total=256, inp1_unstable=1, inp2_unstable=1, delta_unstable=256
- layer_idx=6, type=LayerType.Linear, total=256, inp1_unstable=0, inp2_unstable=0, delta_unstable=256
- layer_idx=8, type=LayerType.Linear, total=10, inp1_unstable=4, inp2_unstable=4, delta_unstable=10

Time for backsubstitution: 1.38 seconds

### RS candidates at layer 3
type: RSZ, layer: 3, pos: 8
type: RSZ, layer: 3, pos: 13
type: RSZ, layer: 3, pos: 20
type: RSZ, layer: 3, pos: 35
type: RSZ, layer: 3, pos: 43
type: RSZ, layer: 3, pos: 48
type: RSZ, layer: 3, pos: 103
type: RSZ, layer: 3, pos: 133
type: RSZ, layer: 3, pos: 180
type: RSZ, layer: 3, pos: 183

Time for candidate selection: 0.29 seconds

### Candidate
type: RSZ, layer: 3, pos: 8

### Relational analysis RSZ of RS_RSZ1_RSZ1_RSZ2_RSZ2_RSZ1_RSZ2_RSZ1_RSZ1

#### Relational analysis RSZ result of RS_RSZ1_RSZ1_RSZ2_RSZ2_RSZ1_RSZ2_RSZ1_RSZ1
Status: Status.VERIFIED
Output dim: 9, lower bound: -0.0043022, upper bound: 0.0045506
time: 0.81 seconds

### Relational analysis RSZ of RS_RSZ1_RSZ1_RSZ2_RSZ2_RSZ1_RSZ2_RSZ1_RSZ2

#### Relational analysis RSZ result of RS_RSZ1_RSZ1_RSZ2_RSZ2_RSZ1_RSZ2_RSZ1_RSZ2
Status: Status.VERIFIED
Output dim: 9, lower bound: -0.0043042, upper bound: 0.0045506
time: 0.84 seconds

## BFS RS instance: RS_RSZ1_RSZ1_RSZ2_RSZ2_RSZ1_RSZ2_RSZ2

### Backsubstitution after applying RS history:
0: -0.0039095, -0.0003029, -0.0039095, -0.0003029, -0.0031448, 0.0032872
1: -0.0044907, 0.0040812, -0.0044907, 0.0040812, -0.0041437, 0.0038241
2: 0.0029569, 0.0099431, 0.0029569, 0.0099431, -0.0038403, 0.0036288
3: -0.0045519, -0.0034923, -0.0045519, -0.0034923, -0.0010595, 0.0010595
4: 0.0017018, 0.0078700, 0.0017018, 0.0078700, -0.0051849, 0.0052946
5: -0.0030341, 0.0031953, -0.0030341, 0.0031953, -0.0047744, 0.0046313
6: -0.0068516, -0.0033211, -0.0068516, -0.0033211, -0.0033518, 0.0033230
7: -0.0021570, 0.0038259, -0.0021570, 0.0038259, -0.0056577, 0.0057394
8: -0.0008533, 0.0000782, -0.0008533, 0.0000782, -0.0008465, 0.0008724
9: 0.9973557, 1.0121032, 0.9973557, 1.0121032, -0.0086734, 0.0091179

### Unstable ReLU Count (Linear/Conv2D Layers)
- layer_idx=0, type=LayerType.Linear, total=256, inp1_unstable=7, inp2_unstable=7, delta_unstable=249
- layer_idx=2, type=LayerType.Linear, total=256, inp1_unstable=10, inp2_unstable=10, delta_unstable=256
- layer_idx=4, type=LayerType.Linear, total=256, inp1_unstable=1, inp2_unstable=1, delta_unstable=256
- layer_idx=6, type=LayerType.Linear, total=256, inp1_unstable=0, inp2_unstable=0, delta_unstable=256
- layer_idx=8, type=LayerType.Linear, total=10, inp1_unstable=4, inp2_unstable=4, delta_unstable=10

Time for backsubstitution: 1.35 seconds

### RS candidates at layer 3
type: RSZ, layer: 3, pos: 8
type: RSZ, layer: 3, pos: 13
type: RSZ, layer: 3, pos: 20
type: RSZ, layer: 3, pos: 35
type: RSZ, layer: 3, pos: 43
type: RSZ, layer: 3, pos: 48
type: RSZ, layer: 3, pos: 103
type: RSZ, layer: 3, pos: 133
type: RSZ, layer: 3, pos: 180
type: RSZ, layer: 3, pos: 183

Time for candidate selection: 0.29 seconds

### Candidate
type: RSZ, layer: 3, pos: 8

### Relational analysis RSZ of RS_RSZ1_RSZ1_RSZ2_RSZ2_RSZ1_RSZ2_RSZ2_RSZ1

#### Relational analysis RSZ result of RS_RSZ1_RSZ1_RSZ2_RSZ2_RSZ1_RSZ2_RSZ2_RSZ1
Status: Status.VERIFIED
Output dim: 9, lower bound: -0.0042473, upper bound: 0.0046823
time: 0.92 seconds

### Relational analysis RSZ of RS_RSZ1_RSZ1_RSZ2_RSZ2_RSZ1_RSZ2_RSZ2_RSZ2

#### Relational analysis RSZ result of RS_RSZ1_RSZ1_RSZ2_RSZ2_RSZ1_RSZ2_RSZ2_RSZ2
Status: Status.VERIFIED
Output dim: 9, lower bound: -0.0042482, upper bound: 0.0046823
time: 0.92 seconds

## BFS RS instance: RS_RSZ1_RSZ1_RSZ2_RSZ2_RSZ2_RSZ1_RSZ2

### Backsubstitution after applying RS history:
0: -0.0039095, -0.0003029, -0.0039095, -0.0003029, -0.0032030, 0.0032442
1: -0.0044907, 0.0040812, -0.0044907, 0.0040812, -0.0040416, 0.0039715
2: 0.0029569, 0.0099431, 0.0029569, 0.0099431, -0.0037696, 0.0037285
3: -0.0045519, -0.0034923, -0.0045519, -0.0034923, -0.0010595, 0.0010595
4: 0.0017018, 0.0078700, 0.0017018, 0.0078700, -0.0052293, 0.0052618
5: -0.0030341, 0.0031953, -0.0030341, 0.0031953, -0.0047247, 0.0047006
6: -0.0068516, -0.0033211, -0.0068516, -0.0033211, -0.0033429, 0.0033357
7: -0.0021570, 0.0038259, -0.0021570, 0.0038259, -0.0056894, 0.0057170
8: -0.0008533, 0.0000782, -0.0008533, 0.0000782, -0.0008390, 0.0008767
9: 0.9973557, 1.0121032, 0.9973557, 1.0121032, -0.0088638, 0.0089826

### Unstable ReLU Count (Linear/Conv2D Layers)
- layer_idx=0, type=LayerType.Linear, total=256, inp1_unstable=7, inp2_unstable=7, delta_unstable=249
- layer_idx=2, type=LayerType.Linear, total=256, inp1_unstable=10, inp2_unstable=10, delta_unstable=256
- layer_idx=4, type=LayerType.Linear, total=256, inp1_unstable=1, inp2_unstable=1, delta_unstable=256
- layer_idx=6, type=LayerType.Linear, total=256, inp1_unstable=0, inp2_unstable=0, delta_unstable=256
- layer_idx=8, type=LayerType.Linear, total=10, inp1_unstable=4, inp2_unstable=4, delta_unstable=10

Time for backsubstitution: 1.37 seconds

### RS candidates at layer 3
type: RSZ, layer: 3, pos: 8
type: RSZ, layer: 3, pos: 13
type: RSZ, layer: 3, pos: 20
type: RSZ, layer: 3, pos: 35
type: RSZ, layer: 3, pos: 43
type: RSZ, layer: 3, pos: 48
type: RSZ, layer: 3, pos: 103
type: RSZ, layer: 3, pos: 133
type: RSZ, layer: 3, pos: 180
type: RSZ, layer: 3, pos: 183

Time for candidate selection: 0.29 seconds

### Candidate
type: RSZ, layer: 3, pos: 8

### Relational analysis RSZ of RS_RSZ1_RSZ1_RSZ2_RSZ2_RSZ2_RSZ1_RSZ2_RSZ1

#### Relational analysis RSZ result of RS_RSZ1_RSZ1_RSZ2_RSZ2_RSZ2_RSZ1_RSZ2_RSZ1
Status: Status.VERIFIED
Output dim: 9, lower bound: -0.0043773, upper bound: 0.0045630
time: 0.88 seconds

### Relational analysis RSZ of RS_RSZ1_RSZ1_RSZ2_RSZ2_RSZ2_RSZ1_RSZ2_RSZ2

#### Relational analysis RSZ result of RS_RSZ1_RSZ1_RSZ2_RSZ2_RSZ2_RSZ1_RSZ2_RSZ2
Status: Status.VERIFIED
Output dim: 9, lower bound: -0.0043890, upper bound: 0.0045630
time: 0.85 seconds

## BFS RS instance: RS_RSZ1_RSZ1_RSZ2_RSZ2_RSZ2_RSZ2_RSZ2

### Backsubstitution after applying RS history:
0: -0.0039095, -0.0003029, -0.0039095, -0.0003029, -0.0032027, 0.0032447
1: -0.0044907, 0.0040812, -0.0044907, 0.0040812, -0.0040419, 0.0039714
2: 0.0029569, 0.0099431, 0.0029569, 0.0099431, -0.0037699, 0.0037283
3: -0.0045519, -0.0034923, -0.0045519, -0.0034923, -0.0010595, 0.0010595
4: 0.0017018, 0.0078700, 0.0017018, 0.0078700, -0.0052289, 0.0052621
5: -0.0030341, 0.0031953, -0.0030341, 0.0031953, -0.0047256, 0.0046997
6: -0.0068516, -0.0033211, -0.0068516, -0.0033211, -0.0033429, 0.0033357
7: -0.0021570, 0.0038259, -0.0021570, 0.0038259, -0.0056900, 0.0057188
8: -0.0008533, 0.0000782, -0.0008533, 0.0000782, -0.0008393, 0.0008773
9: 0.9973557, 1.0121032, 0.9973557, 1.0121032, -0.0088635, 0.0089831

### Unstable ReLU Count (Linear/Conv2D Layers)
- layer_idx=0, type=LayerType.Linear, total=256, inp1_unstable=7, inp2_unstable=7, delta_unstable=249
- layer_idx=2, type=LayerType.Linear, total=256, inp1_unstable=10, inp2_unstable=10, delta_unstable=256
- layer_idx=4, type=LayerType.Linear, total=256, inp1_unstable=1, inp2_unstable=1, delta_unstable=256
- layer_idx=6, type=LayerType.Linear, total=256, inp1_unstable=0, inp2_unstable=0, delta_unstable=256
- layer_idx=8, type=LayerType.Linear, total=10, inp1_unstable=4, inp2_unstable=4, delta_unstable=10

Time for backsubstitution: 1.36 seconds

### RS candidates at layer 3
type: RSZ, layer: 3, pos: 8
type: RSZ, layer: 3, pos: 13
type: RSZ, layer: 3, pos: 20
type: RSZ, layer: 3, pos: 35
type: RSZ, layer: 3, pos: 43
type: RSZ, layer: 3, pos: 48
type: RSZ, layer: 3, pos: 103
type: RSZ, layer: 3, pos: 133
type: RSZ, layer: 3, pos: 180
type: RSZ, layer: 3, pos: 183

Time for candidate selection: 0.27 seconds

### Candidate
type: RSZ, layer: 3, pos: 8

### Relational analysis RSZ of RS_RSZ1_RSZ1_RSZ2_RSZ2_RSZ2_RSZ2_RSZ2_RSZ1

#### Relational analysis RSZ result of RS_RSZ1_RSZ1_RSZ2_RSZ2_RSZ2_RSZ2_RSZ2_RSZ1
Status: Status.VERIFIED
Output dim: 9, lower bound: -0.0043773, upper bound: 0.0045630
time: 0.87 seconds

### Relational analysis RSZ of RS_RSZ1_RSZ1_RSZ2_RSZ2_RSZ2_RSZ2_RSZ2_RSZ2

#### Relational analysis RSZ result of RS_RSZ1_RSZ1_RSZ2_RSZ2_RSZ2_RSZ2_RSZ2_RSZ2
Status: Status.VERIFIED
Output dim: 9, lower bound: -0.0043890, upper bound: 0.0045630
time: 0.84 seconds

## BFS RS instance: RS_RSZ1_RSZ2_RSZ1_RSZ1_RSZ1_RSZ1_RSZ1

### Backsubstitution after applying RS history:
0: -0.0039095, -0.0003029, -0.0039095, -0.0003029, -0.0032460, 0.0032008
1: -0.0044907, 0.0040812, -0.0044907, 0.0040812, -0.0039610, 0.0040498
2: 0.0029569, 0.0099431, 0.0029569, 0.0099431, -0.0037199, 0.0037784
3: -0.0045519, -0.0034923, -0.0045519, -0.0034923, -0.0010595, 0.0010595
4: 0.0017018, 0.0078700, 0.0017018, 0.0078700, -0.0052633, 0.0052273
5: -0.0030341, 0.0031953, -0.0030341, 0.0031953, -0.0046992, 0.0047262
6: -0.0068516, -0.0033211, -0.0068516, -0.0033211, -0.0033349, 0.0033435
7: -0.0021570, 0.0038259, -0.0021570, 0.0038259, -0.0057259, 0.0056852
8: -0.0008533, 0.0000782, -0.0008533, 0.0000782, -0.0008775, 0.0008390
9: 0.9973557, 1.0121032, 0.9973557, 1.0121032, -0.0089900, 0.0088537

### Unstable ReLU Count (Linear/Conv2D Layers)
- layer_idx=0, type=LayerType.Linear, total=256, inp1_unstable=7, inp2_unstable=7, delta_unstable=249
- layer_idx=2, type=LayerType.Linear, total=256, inp1_unstable=10, inp2_unstable=10, delta_unstable=256
- layer_idx=4, type=LayerType.Linear, total=256, inp1_unstable=1, inp2_unstable=1, delta_unstable=256
- layer_idx=6, type=LayerType.Linear, total=256, inp1_unstable=0, inp2_unstable=0, delta_unstable=256
- layer_idx=8, type=LayerType.Linear, total=10, inp1_unstable=4, inp2_unstable=4, delta_unstable=10

Time for backsubstitution: 1.36 seconds

### RS candidates at layer 3
type: RSZ, layer: 3, pos: 8
type: RSZ, layer: 3, pos: 13
type: RSZ, layer: 3, pos: 20
type: RSZ, layer: 3, pos: 35
type: RSZ, layer: 3, pos: 43
type: RSZ, layer: 3, pos: 48
type: RSZ, layer: 3, pos: 103
type: RSZ, layer: 3, pos: 133
type: RSZ, layer: 3, pos: 180
type: RSZ, layer: 3, pos: 183

Time for candidate selection: 0.27 seconds

### Candidate
type: RSZ, layer: 3, pos: 8

### Relational analysis RSZ of RS_RSZ1_RSZ2_RSZ1_RSZ1_RSZ1_RSZ1_RSZ1_RSZ1

#### Relational analysis RSZ result of RS_RSZ1_RSZ2_RSZ1_RSZ1_RSZ1_RSZ1_RSZ1_RSZ1
Status: Status.VERIFIED
Output dim: 9, lower bound: -0.0045717, upper bound: 0.0043565
time: 0.87 seconds

### Relational analysis RSZ of RS_RSZ1_RSZ2_RSZ1_RSZ1_RSZ1_RSZ1_RSZ1_RSZ2

#### Relational analysis RSZ result of RS_RSZ1_RSZ2_RSZ1_RSZ1_RSZ1_RSZ1_RSZ1_RSZ2
Status: Status.VERIFIED
Output dim: 9, lower bound: -0.0045717, upper bound: 0.0043436
time: 0.89 seconds

## BFS RS instance: RS_RSZ1_RSZ2_RSZ1_RSZ1_RSZ1_RSZ2_RSZ1

### Backsubstitution after applying RS history:
0: -0.0039095, -0.0003029, -0.0039095, -0.0003029, -0.0032455, 0.0032012
1: -0.0044907, 0.0040812, -0.0044907, 0.0040812, -0.0039611, 0.0040495
2: 0.0029569, 0.0099431, 0.0029569, 0.0099431, -0.0037200, 0.0037783
3: -0.0045519, -0.0034923, -0.0045519, -0.0034923, -0.0010595, 0.0010595
4: 0.0017018, 0.0078700, 0.0017018, 0.0078700, -0.0052630, 0.0052275
5: -0.0030341, 0.0031953, -0.0030341, 0.0031953, -0.0047001, 0.0047252
6: -0.0068516, -0.0033211, -0.0068516, -0.0033211, -0.0033349, 0.0033435
7: -0.0021570, 0.0038259, -0.0021570, 0.0038259, -0.0057238, 0.0056859
8: -0.0008533, 0.0000782, -0.0008533, 0.0000782, -0.0008773, 0.0008395
9: 0.9973557, 1.0121032, 0.9973557, 1.0121032, -0.0089896, 0.0088536

### Unstable ReLU Count (Linear/Conv2D Layers)
- layer_idx=0, type=LayerType.Linear, total=256, inp1_unstable=7, inp2_unstable=7, delta_unstable=249
- layer_idx=2, type=LayerType.Linear, total=256, inp1_unstable=10, inp2_unstable=10, delta_unstable=256
- layer_idx=4, type=LayerType.Linear, total=256, inp1_unstable=1, inp2_unstable=1, delta_unstable=256
- layer_idx=6, type=LayerType.Linear, total=256, inp1_unstable=0, inp2_unstable=0, delta_unstable=256
- layer_idx=8, type=LayerType.Linear, total=10, inp1_unstable=4, inp2_unstable=4, delta_unstable=10

Time for backsubstitution: 1.45 seconds

### RS candidates at layer 3
type: RSZ, layer: 3, pos: 8
type: RSZ, layer: 3, pos: 13
type: RSZ, layer: 3, pos: 20
type: RSZ, layer: 3, pos: 35
type: RSZ, layer: 3, pos: 43
type: RSZ, layer: 3, pos: 48
type: RSZ, layer: 3, pos: 103
type: RSZ, layer: 3, pos: 133
type: RSZ, layer: 3, pos: 180
type: RSZ, layer: 3, pos: 183

Time for candidate selection: 0.30 seconds

### Candidate
type: RSZ, layer: 3, pos: 8

### Relational analysis RSZ of RS_RSZ1_RSZ2_RSZ1_RSZ1_RSZ1_RSZ2_RSZ1_RSZ1

#### Relational analysis RSZ result of RS_RSZ1_RSZ2_RSZ1_RSZ1_RSZ1_RSZ2_RSZ1_RSZ1
Status: Status.VERIFIED
Output dim: 9, lower bound: -0.0045717, upper bound: 0.0043565
time: 0.87 seconds

### Relational analysis RSZ of RS_RSZ1_RSZ2_RSZ1_RSZ1_RSZ1_RSZ2_RSZ1_RSZ2

#### Relational analysis RSZ result of RS_RSZ1_RSZ2_RSZ1_RSZ1_RSZ1_RSZ2_RSZ1_RSZ2
Status: Status.VERIFIED
Output dim: 9, lower bound: -0.0045717, upper bound: 0.0043436
time: 0.89 seconds

## BFS RS instance: RS_RSZ1_RSZ2_RSZ1_RSZ1_RSZ2_RSZ1_RSZ1

### Backsubstitution after applying RS history:
0: -0.0039095, -0.0003029, -0.0039095, -0.0003029, -0.0032883, 0.0031434
1: -0.0044907, 0.0040812, -0.0044907, 0.0040812, -0.0038162, 0.0041502
2: 0.0029569, 0.0099431, 0.0029569, 0.0099431, -0.0036219, 0.0038454
3: -0.0045519, -0.0034923, -0.0045519, -0.0034923, -0.0010595, 0.0010595
4: 0.0017018, 0.0078700, 0.0017018, 0.0078700, -0.0052955, 0.0051835
5: -0.0030341, 0.0031953, -0.0030341, 0.0031953, -0.0046305, 0.0047749
6: -0.0068516, -0.0033211, -0.0068516, -0.0033211, -0.0033224, 0.0033523
7: -0.0021570, 0.0038259, -0.0021570, 0.0038259, -0.0057462, 0.0056501
8: -0.0008533, 0.0000782, -0.0008533, 0.0000782, -0.0008731, 0.0008466
9: 0.9973557, 1.0121032, 0.9973557, 1.0121032, -0.0091225, 0.0086644

### Unstable ReLU Count (Linear/Conv2D Layers)
- layer_idx=0, type=LayerType.Linear, total=256, inp1_unstable=7, inp2_unstable=7, delta_unstable=249
- layer_idx=2, type=LayerType.Linear, total=256, inp1_unstable=10, inp2_unstable=10, delta_unstable=256
- layer_idx=4, type=LayerType.Linear, total=256, inp1_unstable=1, inp2_unstable=1, delta_unstable=256
- layer_idx=6, type=LayerType.Linear, total=256, inp1_unstable=0, inp2_unstable=0, delta_unstable=256
- layer_idx=8, type=LayerType.Linear, total=10, inp1_unstable=4, inp2_unstable=4, delta_unstable=10

Time for backsubstitution: 1.41 seconds

### RS candidates at layer 3
type: RSZ, layer: 3, pos: 8
type: RSZ, layer: 3, pos: 13
type: RSZ, layer: 3, pos: 20
type: RSZ, layer: 3, pos: 35
type: RSZ, layer: 3, pos: 43
type: RSZ, layer: 3, pos: 48
type: RSZ, layer: 3, pos: 103
type: RSZ, layer: 3, pos: 133
type: RSZ, layer: 3, pos: 180
type: RSZ, layer: 3, pos: 183

Time for candidate selection: 0.31 seconds

### Candidate
type: RSZ, layer: 3, pos: 8

### Relational analysis RSZ of RS_RSZ1_RSZ2_RSZ1_RSZ1_RSZ2_RSZ1_RSZ1_RSZ1

#### Relational analysis RSZ result of RS_RSZ1_RSZ2_RSZ1_RSZ1_RSZ2_RSZ1_RSZ1_RSZ1
Status: Status.VERIFIED
Output dim: 9, lower bound: -0.0047022, upper bound: 0.0042183
time: 0.84 seconds

### Relational analysis RSZ of RS_RSZ1_RSZ2_RSZ1_RSZ1_RSZ2_RSZ1_RSZ1_RSZ2

#### Relational analysis RSZ result of RS_RSZ1_RSZ2_RSZ1_RSZ1_RSZ2_RSZ1_RSZ1_RSZ2
Status: Status.VERIFIED
Output dim: 9, lower bound: -0.0047022, upper bound: 0.0042133
time: 0.82 seconds

## BFS RS instance: RS_RSZ1_RSZ2_RSZ1_RSZ1_RSZ2_RSZ1_RSZ2

### Backsubstitution after applying RS history:
0: -0.0039095, -0.0003029, -0.0039095, -0.0003029, -0.0032760, 0.0031569
1: -0.0044907, 0.0040812, -0.0044907, 0.0040812, -0.0038601, 0.0041083
2: 0.0029569, 0.0099431, 0.0029569, 0.0099431, -0.0036541, 0.0038160
3: -0.0045519, -0.0034923, -0.0045519, -0.0034923, -0.0010595, 0.0010595
4: 0.0017018, 0.0078700, 0.0017018, 0.0078700, -0.0052858, 0.0051946
5: -0.0030341, 0.0031953, -0.0030341, 0.0031953, -0.0046408, 0.0047639
6: -0.0068516, -0.0033211, -0.0068516, -0.0033211, -0.0033260, 0.0033488
7: -0.0021570, 0.0038259, -0.0021570, 0.0038259, -0.0057316, 0.0056626
8: -0.0008533, 0.0000782, -0.0008533, 0.0000782, -0.0008731, 0.0008459
9: 0.9973557, 1.0121032, 0.9973557, 1.0121032, -0.0090726, 0.0087168

### Unstable ReLU Count (Linear/Conv2D Layers)
- layer_idx=0, type=LayerType.Linear, total=256, inp1_unstable=7, inp2_unstable=7, delta_unstable=249
- layer_idx=2, type=LayerType.Linear, total=256, inp1_unstable=10, inp2_unstable=10, delta_unstable=256
- layer_idx=4, type=LayerType.Linear, total=256, inp1_unstable=1, inp2_unstable=1, delta_unstable=256
- layer_idx=6, type=LayerType.Linear, total=256, inp1_unstable=0, inp2_unstable=0, delta_unstable=256
- layer_idx=8, type=LayerType.Linear, total=10, inp1_unstable=4, inp2_unstable=4, delta_unstable=10

Time for backsubstitution: 1.39 seconds

### RS candidates at layer 3
type: RSZ, layer: 3, pos: 8
type: RSZ, layer: 3, pos: 13
type: RSZ, layer: 3, pos: 20
type: RSZ, layer: 3, pos: 35
type: RSZ, layer: 3, pos: 43
type: RSZ, layer: 3, pos: 48
type: RSZ, layer: 3, pos: 103
type: RSZ, layer: 3, pos: 133
type: RSZ, layer: 3, pos: 180
type: RSZ, layer: 3, pos: 183

Time for candidate selection: 0.31 seconds

### Candidate
type: RSZ, layer: 3, pos: 8

### Relational analysis RSZ of RS_RSZ1_RSZ2_RSZ1_RSZ1_RSZ2_RSZ1_RSZ2_RSZ1

#### Relational analysis RSZ result of RS_RSZ1_RSZ2_RSZ1_RSZ1_RSZ2_RSZ1_RSZ2_RSZ1
Status: Status.VERIFIED
Output dim: 9, lower bound: -0.0045761, upper bound: 0.0042685
time: 0.87 seconds

### Relational analysis RSZ of RS_RSZ1_RSZ2_RSZ1_RSZ1_RSZ2_RSZ1_RSZ2_RSZ2

#### Relational analysis RSZ result of RS_RSZ1_RSZ2_RSZ1_RSZ1_RSZ2_RSZ1_RSZ2_RSZ2
Status: Status.VERIFIED
Output dim: 9, lower bound: -0.0045761, upper bound: 0.0042613
time: 0.89 seconds

## BFS RS instance: RS_RSZ1_RSZ2_RSZ1_RSZ1_RSZ2_RSZ2_RSZ1

### Backsubstitution after applying RS history:
0: -0.0039095, -0.0003029, -0.0039095, -0.0003029, -0.0032880, 0.0031443
1: -0.0044907, 0.0040812, -0.0044907, 0.0040812, -0.0038166, 0.0041501
2: 0.0029569, 0.0099431, 0.0029569, 0.0099431, -0.0036228, 0.0038454
3: -0.0045519, -0.0034923, -0.0045519, -0.0034923, -0.0010595, 0.0010595
4: 0.0017018, 0.0078700, 0.0017018, 0.0078700, -0.0052954, 0.0051841
5: -0.0030341, 0.0031953, -0.0030341, 0.0031953, -0.0046326, 0.0047742
6: -0.0068516, -0.0033211, -0.0068516, -0.0033211, -0.0033224, 0.0033523
7: -0.0021570, 0.0038259, -0.0021570, 0.0038259, -0.0057449, 0.0056505
8: -0.0008533, 0.0000782, -0.0008533, 0.0000782, -0.0008725, 0.0008466
9: 0.9973557, 1.0121032, 0.9973557, 1.0121032, -0.0091226, 0.0086653

### Unstable ReLU Count (Linear/Conv2D Layers)
- layer_idx=0, type=LayerType.Linear, total=256, inp1_unstable=7, inp2_unstable=7, delta_unstable=249
- layer_idx=2, type=LayerType.Linear, total=256, inp1_unstable=10, inp2_unstable=10, delta_unstable=256
- layer_idx=4, type=LayerType.Linear, total=256, inp1_unstable=1, inp2_unstable=1, delta_unstable=256
- layer_idx=6, type=LayerType.Linear, total=256, inp1_unstable=0, inp2_unstable=0, delta_unstable=256
- layer_idx=8, type=LayerType.Linear, total=10, inp1_unstable=4, inp2_unstable=4, delta_unstable=10

Time for backsubstitution: 1.44 seconds

### RS candidates at layer 3
type: RSZ, layer: 3, pos: 8
type: RSZ, layer: 3, pos: 13
type: RSZ, layer: 3, pos: 20
type: RSZ, layer: 3, pos: 35
type: RSZ, layer: 3, pos: 43
type: RSZ, layer: 3, pos: 48
type: RSZ, layer: 3, pos: 103
type: RSZ, layer: 3, pos: 133
type: RSZ, layer: 3, pos: 180
type: RSZ, layer: 3, pos: 183

Time for candidate selection: 0.31 seconds

### Candidate
type: RSZ, layer: 3, pos: 8

### Relational analysis RSZ of RS_RSZ1_RSZ2_RSZ1_RSZ1_RSZ2_RSZ2_RSZ1_RSZ1

#### Relational analysis RSZ result of RS_RSZ1_RSZ2_RSZ1_RSZ1_RSZ2_RSZ2_RSZ1_RSZ1
Status: Status.VERIFIED
Output dim: 9, lower bound: -0.0047022, upper bound: 0.0042183
time: 0.85 seconds

### Relational analysis RSZ of RS_RSZ1_RSZ2_RSZ1_RSZ1_RSZ2_RSZ2_RSZ1_RSZ2

#### Relational analysis RSZ result of RS_RSZ1_RSZ2_RSZ1_RSZ1_RSZ2_RSZ2_RSZ1_RSZ2
Status: Status.VERIFIED
Output dim: 9, lower bound: -0.0047022, upper bound: 0.0042133
time: 0.82 seconds

## BFS RS instance: RS_RSZ1_RSZ2_RSZ1_RSZ1_RSZ2_RSZ2_RSZ2

### Backsubstitution after applying RS history:
0: -0.0039095, -0.0003029, -0.0039095, -0.0003029, -0.0032759, 0.0031576
1: -0.0044907, 0.0040812, -0.0044907, 0.0040812, -0.0038607, 0.0041085
2: 0.0029569, 0.0099431, 0.0029569, 0.0099431, -0.0036553, 0.0038159
3: -0.0045519, -0.0034923, -0.0045519, -0.0034923, -0.0010595, 0.0010595
4: 0.0017018, 0.0078700, 0.0017018, 0.0078700, -0.0052858, 0.0051950
5: -0.0030341, 0.0031953, -0.0030341, 0.0031953, -0.0046432, 0.0047632
6: -0.0068516, -0.0033211, -0.0068516, -0.0033211, -0.0033261, 0.0033488
7: -0.0021570, 0.0038259, -0.0021570, 0.0038259, -0.0057313, 0.0056649
8: -0.0008533, 0.0000782, -0.0008533, 0.0000782, -0.0008727, 0.0008459
9: 0.9973557, 1.0121032, 0.9973557, 1.0121032, -0.0090729, 0.0087179

### Unstable ReLU Count (Linear/Conv2D Layers)
- layer_idx=0, type=LayerType.Linear, total=256, inp1_unstable=7, inp2_unstable=7, delta_unstable=249
- layer_idx=2, type=LayerType.Linear, total=256, inp1_unstable=10, inp2_unstable=10, delta_unstable=256
- layer_idx=4, type=LayerType.Linear, total=256, inp1_unstable=1, inp2_unstable=1, delta_unstable=256
- layer_idx=6, type=LayerType.Linear, total=256, inp1_unstable=0, inp2_unstable=0, delta_unstable=256
- layer_idx=8, type=LayerType.Linear, total=10, inp1_unstable=4, inp2_unstable=4, delta_unstable=10

Time for backsubstitution: 1.46 seconds

### RS candidates at layer 3
type: RSZ, layer: 3, pos: 8
type: RSZ, layer: 3, pos: 13
type: RSZ, layer: 3, pos: 20
type: RSZ, layer: 3, pos: 35
type: RSZ, layer: 3, pos: 43
type: RSZ, layer: 3, pos: 48
type: RSZ, layer: 3, pos: 103
type: RSZ, layer: 3, pos: 133
type: RSZ, layer: 3, pos: 180
type: RSZ, layer: 3, pos: 183

Time for candidate selection: 0.29 seconds

### Candidate
type: RSZ, layer: 3, pos: 8

### Relational analysis RSZ of RS_RSZ1_RSZ2_RSZ1_RSZ1_RSZ2_RSZ2_RSZ2_RSZ1

#### Relational analysis RSZ result of RS_RSZ1_RSZ2_RSZ1_RSZ1_RSZ2_RSZ2_RSZ2_RSZ1
Status: Status.VERIFIED
Output dim: 9, lower bound: -0.0045761, upper bound: 0.0042685
time: 0.83 seconds

### Relational analysis RSZ of RS_RSZ1_RSZ2_RSZ1_RSZ1_RSZ2_RSZ2_RSZ2_RSZ2

#### Relational analysis RSZ result of RS_RSZ1_RSZ2_RSZ1_RSZ1_RSZ2_RSZ2_RSZ2_RSZ2
Status: Status.VERIFIED
Output dim: 9, lower bound: -0.0045761, upper bound: 0.0042613
time: 0.83 seconds

## BFS RS instance: RS_RSZ1_RSZ2_RSZ1_RSZ2_RSZ1_RSZ1_RSZ1

### Backsubstitution after applying RS history:
0: -0.0039095, -0.0003029, -0.0039095, -0.0003029, -0.0031752, 0.0032710
1: -0.0044907, 0.0040812, -0.0044907, 0.0040812, -0.0041175, 0.0038770
2: 0.0029569, 0.0099431, 0.0029569, 0.0099431, -0.0038230, 0.0036600
3: -0.0045519, -0.0034923, -0.0045519, -0.0034923, -0.0010595, 0.0010595
4: 0.0017018, 0.0078700, 0.0017018, 0.0078700, -0.0052091, 0.0052810
5: -0.0030341, 0.0031953, -0.0030341, 0.0031953, -0.0047779, 0.0046479
6: -0.0068516, -0.0033211, -0.0068516, -0.0033211, -0.0033492, 0.0033282
7: -0.0021570, 0.0038259, -0.0021570, 0.0038259, -0.0056992, 0.0057083
8: -0.0008533, 0.0000782, -0.0008533, 0.0000782, -0.0008881, 0.0008311
9: 0.9973557, 1.0121032, 0.9973557, 1.0121032, -0.0087588, 0.0090752

### Unstable ReLU Count (Linear/Conv2D Layers)
- layer_idx=0, type=LayerType.Linear, total=256, inp1_unstable=7, inp2_unstable=7, delta_unstable=249
- layer_idx=2, type=LayerType.Linear, total=256, inp1_unstable=10, inp2_unstable=10, delta_unstable=256
- layer_idx=4, type=LayerType.Linear, total=256, inp1_unstable=1, inp2_unstable=1, delta_unstable=256
- layer_idx=6, type=LayerType.Linear, total=256, inp1_unstable=0, inp2_unstable=0, delta_unstable=256
- layer_idx=8, type=LayerType.Linear, total=10, inp1_unstable=4, inp2_unstable=4, delta_unstable=10

Time for backsubstitution: 1.41 seconds

### RS candidates at layer 3
type: RSZ, layer: 3, pos: 8
type: RSZ, layer: 3, pos: 13
type: RSZ, layer: 3, pos: 20
type: RSZ, layer: 3, pos: 35
type: RSZ, layer: 3, pos: 43
type: RSZ, layer: 3, pos: 48
type: RSZ, layer: 3, pos: 103
type: RSZ, layer: 3, pos: 133
type: RSZ, layer: 3, pos: 180
type: RSZ, layer: 3, pos: 183

Time for candidate selection: 0.29 seconds

### Candidate
type: RSZ, layer: 3, pos: 8

### Relational analysis RSZ of RS_RSZ1_RSZ2_RSZ1_RSZ2_RSZ1_RSZ1_RSZ1_RSZ1

#### Relational analysis RSZ result of RS_RSZ1_RSZ2_RSZ1_RSZ2_RSZ1_RSZ1_RSZ1_RSZ1
Status: Status.VERIFIED
Output dim: 9, lower bound: -0.0042838, upper bound: 0.0046346
time: 0.85 seconds

### Relational analysis RSZ of RS_RSZ1_RSZ2_RSZ1_RSZ2_RSZ1_RSZ1_RSZ1_RSZ2

#### Relational analysis RSZ result of RS_RSZ1_RSZ2_RSZ1_RSZ2_RSZ1_RSZ1_RSZ1_RSZ2
Status: Status.VERIFIED
Output dim: 9, lower bound: -0.0042838, upper bound: 0.0046333
time: 0.86 seconds

## BFS RS instance: RS_RSZ1_RSZ2_RSZ1_RSZ2_RSZ1_RSZ1_RSZ2

### Backsubstitution after applying RS history:
0: -0.0039095, -0.0003029, -0.0039095, -0.0003029, -0.0031611, 0.0032840
1: -0.0044907, 0.0040812, -0.0044907, 0.0040812, -0.0041582, 0.0038328
2: 0.0029569, 0.0099431, 0.0029569, 0.0099431, -0.0038513, 0.0036292
3: -0.0045519, -0.0034923, -0.0045519, -0.0034923, -0.0010595, 0.0010595
4: 0.0017018, 0.0078700, 0.0017018, 0.0078700, -0.0051978, 0.0052911
5: -0.0030341, 0.0031953, -0.0030341, 0.0031953, -0.0047890, 0.0046367
6: -0.0068516, -0.0033211, -0.0068516, -0.0033211, -0.0033526, 0.0033246
7: -0.0021570, 0.0038259, -0.0021570, 0.0038259, -0.0056839, 0.0057200
8: -0.0008533, 0.0000782, -0.0008533, 0.0000782, -0.0008883, 0.0008307
9: 0.9973557, 1.0121032, 0.9973557, 1.0121032, -0.0087061, 0.0091237

### Unstable ReLU Count (Linear/Conv2D Layers)
- layer_idx=0, type=LayerType.Linear, total=256, inp1_unstable=7, inp2_unstable=7, delta_unstable=249
- layer_idx=2, type=LayerType.Linear, total=256, inp1_unstable=10, inp2_unstable=10, delta_unstable=256
- layer_idx=4, type=LayerType.Linear, total=256, inp1_unstable=1, inp2_unstable=1, delta_unstable=256
- layer_idx=6, type=LayerType.Linear, total=256, inp1_unstable=0, inp2_unstable=0, delta_unstable=256
- layer_idx=8, type=LayerType.Linear, total=10, inp1_unstable=4, inp2_unstable=4, delta_unstable=10

Time for backsubstitution: 1.40 seconds

### RS candidates at layer 3
type: RSZ, layer: 3, pos: 8
type: RSZ, layer: 3, pos: 13
type: RSZ, layer: 3, pos: 20
type: RSZ, layer: 3, pos: 35
type: RSZ, layer: 3, pos: 43
type: RSZ, layer: 3, pos: 48
type: RSZ, layer: 3, pos: 103
type: RSZ, layer: 3, pos: 133
type: RSZ, layer: 3, pos: 180
type: RSZ, layer: 3, pos: 183

Time for candidate selection: 0.35 seconds

### Candidate
type: RSZ, layer: 3, pos: 8

### Relational analysis RSZ of RS_RSZ1_RSZ2_RSZ1_RSZ2_RSZ1_RSZ1_RSZ2_RSZ1

#### Relational analysis RSZ result of RS_RSZ1_RSZ2_RSZ1_RSZ2_RSZ1_RSZ1_RSZ2_RSZ1
Status: Status.VERIFIED
Output dim: 9, lower bound: -0.0042118, upper bound: 0.0047462
time: 0.89 seconds

### Relational analysis RSZ of RS_RSZ1_RSZ2_RSZ1_RSZ2_RSZ1_RSZ1_RSZ2_RSZ2

#### Relational analysis RSZ result of RS_RSZ1_RSZ2_RSZ1_RSZ2_RSZ1_RSZ1_RSZ2_RSZ2
Status: Status.VERIFIED
Output dim: 9, lower bound: -0.0042118, upper bound: 0.0047462
time: 0.89 seconds

## BFS RS instance: RS_RSZ1_RSZ2_RSZ1_RSZ2_RSZ1_RSZ2_RSZ1

### Backsubstitution after applying RS history:
0: -0.0039095, -0.0003029, -0.0039095, -0.0003029, -0.0031744, 0.0032714
1: -0.0044907, 0.0040812, -0.0044907, 0.0040812, -0.0041174, 0.0038765
2: 0.0029569, 0.0099431, 0.0029569, 0.0099431, -0.0038226, 0.0036594
3: -0.0045519, -0.0034923, -0.0045519, -0.0034923, -0.0010595, 0.0010595
4: 0.0017018, 0.0078700, 0.0017018, 0.0078700, -0.0052086, 0.0052811
5: -0.0030341, 0.0031953, -0.0030341, 0.0031953, -0.0047780, 0.0046461
6: -0.0068516, -0.0033211, -0.0068516, -0.0033211, -0.0033492, 0.0033282
7: -0.0021570, 0.0038259, -0.0021570, 0.0038259, -0.0056962, 0.0057078
8: -0.0008533, 0.0000782, -0.0008533, 0.0000782, -0.0008865, 0.0008311
9: 0.9973557, 1.0121032, 0.9973557, 1.0121032, -0.0087578, 0.0090754

### Unstable ReLU Count (Linear/Conv2D Layers)
- layer_idx=0, type=LayerType.Linear, total=256, inp1_unstable=7, inp2_unstable=7, delta_unstable=249
- layer_idx=2, type=LayerType.Linear, total=256, inp1_unstable=10, inp2_unstable=10, delta_unstable=256
- layer_idx=4, type=LayerType.Linear, total=256, inp1_unstable=1, inp2_unstable=1, delta_unstable=256
- layer_idx=6, type=LayerType.Linear, total=256, inp1_unstable=0, inp2_unstable=0, delta_unstable=256
- layer_idx=8, type=LayerType.Linear, total=10, inp1_unstable=4, inp2_unstable=4, delta_unstable=10

Time for backsubstitution: 1.36 seconds

### RS candidates at layer 3
type: RSZ, layer: 3, pos: 8
type: RSZ, layer: 3, pos: 13
type: RSZ, layer: 3, pos: 20
type: RSZ, layer: 3, pos: 35
type: RSZ, layer: 3, pos: 43
type: RSZ, layer: 3, pos: 48
type: RSZ, layer: 3, pos: 103
type: RSZ, layer: 3, pos: 133
type: RSZ, layer: 3, pos: 180
type: RSZ, layer: 3, pos: 183

Time for candidate selection: 0.27 seconds

### Candidate
type: RSZ, layer: 3, pos: 8

### Relational analysis RSZ of RS_RSZ1_RSZ2_RSZ1_RSZ2_RSZ1_RSZ2_RSZ1_RSZ1

#### Relational analysis RSZ result of RS_RSZ1_RSZ2_RSZ1_RSZ2_RSZ1_RSZ2_RSZ1_RSZ1
Status: Status.VERIFIED
Output dim: 9, lower bound: -0.0042838, upper bound: 0.0046346
time: 0.83 seconds

### Relational analysis RSZ of RS_RSZ1_RSZ2_RSZ1_RSZ2_RSZ1_RSZ2_RSZ1_RSZ2

#### Relational analysis RSZ result of RS_RSZ1_RSZ2_RSZ1_RSZ2_RSZ1_RSZ2_RSZ1_RSZ2
Status: Status.VERIFIED
Output dim: 9, lower bound: -0.0042838, upper bound: 0.0046333
time: 0.85 seconds

## BFS RS instance: RS_RSZ1_RSZ2_RSZ1_RSZ2_RSZ1_RSZ2_RSZ2

### Backsubstitution after applying RS history:
0: -0.0039095, -0.0003029, -0.0039095, -0.0003029, -0.0031605, 0.0032846
1: -0.0044907, 0.0040812, -0.0044907, 0.0040812, -0.0041583, 0.0038327
2: 0.0029569, 0.0099431, 0.0029569, 0.0099431, -0.0038515, 0.0036298
3: -0.0045519, -0.0034923, -0.0045519, -0.0034923, -0.0010595, 0.0010595
4: 0.0017018, 0.0078700, 0.0017018, 0.0078700, -0.0051974, 0.0052915
5: -0.0030341, 0.0031953, -0.0030341, 0.0031953, -0.0047890, 0.0046348
6: -0.0068516, -0.0033211, -0.0068516, -0.0033211, -0.0033526, 0.0033246
7: -0.0021570, 0.0038259, -0.0021570, 0.0038259, -0.0056834, 0.0057204
8: -0.0008533, 0.0000782, -0.0008533, 0.0000782, -0.0008868, 0.0008305
9: 0.9973557, 1.0121032, 0.9973557, 1.0121032, -0.0087053, 0.0091240

### Unstable ReLU Count (Linear/Conv2D Layers)
- layer_idx=0, type=LayerType.Linear, total=256, inp1_unstable=7, inp2_unstable=7, delta_unstable=249
- layer_idx=2, type=LayerType.Linear, total=256, inp1_unstable=10, inp2_unstable=10, delta_unstable=256
- layer_idx=4, type=LayerType.Linear, total=256, inp1_unstable=1, inp2_unstable=1, delta_unstable=256
- layer_idx=6, type=LayerType.Linear, total=256, inp1_unstable=0, inp2_unstable=0, delta_unstable=256
- layer_idx=8, type=LayerType.Linear, total=10, inp1_unstable=4, inp2_unstable=4, delta_unstable=10

Time for backsubstitution: 1.38 seconds

### RS candidates at layer 3
type: RSZ, layer: 3, pos: 8
type: RSZ, layer: 3, pos: 13
type: RSZ, layer: 3, pos: 20
type: RSZ, layer: 3, pos: 35
type: RSZ, layer: 3, pos: 43
type: RSZ, layer: 3, pos: 48
type: RSZ, layer: 3, pos: 103
type: RSZ, layer: 3, pos: 133
type: RSZ, layer: 3, pos: 180
type: RSZ, layer: 3, pos: 183

Time for candidate selection: 0.30 seconds

### Candidate
type: RSZ, layer: 3, pos: 8

### Relational analysis RSZ of RS_RSZ1_RSZ2_RSZ1_RSZ2_RSZ1_RSZ2_RSZ2_RSZ1

#### Relational analysis RSZ result of RS_RSZ1_RSZ2_RSZ1_RSZ2_RSZ1_RSZ2_RSZ2_RSZ1
Status: Status.VERIFIED
Output dim: 9, lower bound: -0.0042118, upper bound: 0.0047462
time: 0.90 seconds

### Relational analysis RSZ of RS_RSZ1_RSZ2_RSZ1_RSZ2_RSZ1_RSZ2_RSZ2_RSZ2

#### Relational analysis RSZ result of RS_RSZ1_RSZ2_RSZ1_RSZ2_RSZ1_RSZ2_RSZ2_RSZ2
Status: Status.VERIFIED
Output dim: 9, lower bound: -0.0042118, upper bound: 0.0047462
time: 0.91 seconds

## BFS RS instance: RS_RSZ1_RSZ2_RSZ1_RSZ2_RSZ2_RSZ1_RSZ1

### Backsubstitution after applying RS history:
0: -0.0039095, -0.0003029, -0.0039095, -0.0003029, -0.0032175, 0.0032149
1: -0.0044907, 0.0040812, -0.0044907, 0.0040812, -0.0039861, 0.0039862
2: 0.0029569, 0.0099431, 0.0029569, 0.0099431, -0.0037315, 0.0037410
3: -0.0045519, -0.0034923, -0.0045519, -0.0034923, -0.0010595, 0.0010595
4: 0.0017018, 0.0078700, 0.0017018, 0.0078700, -0.0052416, 0.0052385
5: -0.0030341, 0.0031953, -0.0030341, 0.0031953, -0.0047102, 0.0046956
6: -0.0068516, -0.0033211, -0.0068516, -0.0033211, -0.0033376, 0.0033375
7: -0.0021570, 0.0038259, -0.0021570, 0.0038259, -0.0057205, 0.0056748
8: -0.0008533, 0.0000782, -0.0008533, 0.0000782, -0.0008817, 0.0008380
9: 0.9973557, 1.0121032, 0.9973557, 1.0121032, -0.0088988, 0.0088979

### Unstable ReLU Count (Linear/Conv2D Layers)
- layer_idx=0, type=LayerType.Linear, total=256, inp1_unstable=7, inp2_unstable=7, delta_unstable=249
- layer_idx=2, type=LayerType.Linear, total=256, inp1_unstable=10, inp2_unstable=10, delta_unstable=256
- layer_idx=4, type=LayerType.Linear, total=256, inp1_unstable=1, inp2_unstable=1, delta_unstable=256
- layer_idx=6, type=LayerType.Linear, total=256, inp1_unstable=0, inp2_unstable=0, delta_unstable=256
- layer_idx=8, type=LayerType.Linear, total=10, inp1_unstable=4, inp2_unstable=4, delta_unstable=10

Time for backsubstitution: 1.42 seconds

### RS candidates at layer 3
type: RSZ, layer: 3, pos: 8
type: RSZ, layer: 3, pos: 13
type: RSZ, layer: 3, pos: 20
type: RSZ, layer: 3, pos: 35
type: RSZ, layer: 3, pos: 43
type: RSZ, layer: 3, pos: 48
type: RSZ, layer: 3, pos: 103
type: RSZ, layer: 3, pos: 133
type: RSZ, layer: 3, pos: 180
type: RSZ, layer: 3, pos: 183

Time for candidate selection: 0.33 seconds

### Candidate
type: RSZ, layer: 3, pos: 8

### Relational analysis RSZ of RS_RSZ1_RSZ2_RSZ1_RSZ2_RSZ2_RSZ1_RSZ1_RSZ1

#### Relational analysis RSZ result of RS_RSZ1_RSZ2_RSZ1_RSZ2_RSZ2_RSZ1_RSZ1_RSZ1
Status: Status.VERIFIED
Output dim: 9, lower bound: -0.0043991, upper bound: 0.0044823
time: 0.80 seconds

### Relational analysis RSZ of RS_RSZ1_RSZ2_RSZ1_RSZ2_RSZ2_RSZ1_RSZ1_RSZ2

#### Relational analysis RSZ result of RS_RSZ1_RSZ2_RSZ1_RSZ2_RSZ2_RSZ1_RSZ1_RSZ2
Status: Status.VERIFIED
Output dim: 9, lower bound: -0.0043991, upper bound: 0.0044823
time: 0.81 seconds

## BFS RS instance: RS_RSZ1_RSZ2_RSZ1_RSZ2_RSZ2_RSZ1_RSZ2

### Backsubstitution after applying RS history:
0: -0.0039095, -0.0003029, -0.0039095, -0.0003029, -0.0032076, 0.0032324
1: -0.0044907, 0.0040812, -0.0044907, 0.0040812, -0.0040400, 0.0039553
2: 0.0029569, 0.0099431, 0.0029569, 0.0099431, -0.0037693, 0.0037154
3: -0.0045519, -0.0034923, -0.0045519, -0.0034923, -0.0010595, 0.0010595
4: 0.0017018, 0.0078700, 0.0017018, 0.0078700, -0.0052338, 0.0052524
5: -0.0030341, 0.0031953, -0.0030341, 0.0031953, -0.0047232, 0.0046876
6: -0.0068516, -0.0033211, -0.0068516, -0.0033211, -0.0033421, 0.0033351
7: -0.0021570, 0.0038259, -0.0021570, 0.0038259, -0.0057079, 0.0056906
8: -0.0008533, 0.0000782, -0.0008533, 0.0000782, -0.0008820, 0.0008378
9: 0.9973557, 1.0121032, 0.9973557, 1.0121032, -0.0088619, 0.0089628

### Unstable ReLU Count (Linear/Conv2D Layers)
- layer_idx=0, type=LayerType.Linear, total=256, inp1_unstable=7, inp2_unstable=7, delta_unstable=249
- layer_idx=2, type=LayerType.Linear, total=256, inp1_unstable=10, inp2_unstable=10, delta_unstable=256
- layer_idx=4, type=LayerType.Linear, total=256, inp1_unstable=1, inp2_unstable=1, delta_unstable=256
- layer_idx=6, type=LayerType.Linear, total=256, inp1_unstable=0, inp2_unstable=0, delta_unstable=256
- layer_idx=8, type=LayerType.Linear, total=10, inp1_unstable=4, inp2_unstable=4, delta_unstable=10

Time for backsubstitution: 1.49 seconds

### RS candidates at layer 3
type: RSZ, layer: 3, pos: 8
type: RSZ, layer: 3, pos: 13
type: RSZ, layer: 3, pos: 20
type: RSZ, layer: 3, pos: 35
type: RSZ, layer: 3, pos: 43
type: RSZ, layer: 3, pos: 48
type: RSZ, layer: 3, pos: 103
type: RSZ, layer: 3, pos: 133
type: RSZ, layer: 3, pos: 180
type: RSZ, layer: 3, pos: 183

Time for candidate selection: 0.31 seconds

### Candidate
type: RSZ, layer: 3, pos: 8

### Relational analysis RSZ of RS_RSZ1_RSZ2_RSZ1_RSZ2_RSZ2_RSZ1_RSZ2_RSZ1

#### Relational analysis RSZ result of RS_RSZ1_RSZ2_RSZ1_RSZ2_RSZ2_RSZ1_RSZ2_RSZ1
Status: Status.VERIFIED
Output dim: 9, lower bound: -0.0043315, upper bound: 0.0046113
time: 0.86 seconds

### Relational analysis RSZ of RS_RSZ1_RSZ2_RSZ1_RSZ2_RSZ2_RSZ1_RSZ2_RSZ2

#### Relational analysis RSZ result of RS_RSZ1_RSZ2_RSZ1_RSZ2_RSZ2_RSZ1_RSZ2_RSZ2
Status: Status.VERIFIED
Output dim: 9, lower bound: -0.0043315, upper bound: 0.0046112
time: 0.86 seconds

## BFS RS instance: RS_RSZ1_RSZ2_RSZ1_RSZ2_RSZ2_RSZ2_RSZ1

### Backsubstitution after applying RS history:
0: -0.0039095, -0.0003029, -0.0039095, -0.0003029, -0.0032173, 0.0032151
1: -0.0044907, 0.0040812, -0.0044907, 0.0040812, -0.0039862, 0.0039861
2: 0.0029569, 0.0099431, 0.0029569, 0.0099431, -0.0037320, 0.0037407
3: -0.0045519, -0.0034923, -0.0045519, -0.0034923, -0.0010595, 0.0010595
4: 0.0017018, 0.0078700, 0.0017018, 0.0078700, -0.0052415, 0.0052387
5: -0.0030341, 0.0031953, -0.0030341, 0.0031953, -0.0047108, 0.0046947
6: -0.0068516, -0.0033211, -0.0068516, -0.0033211, -0.0033376, 0.0033375
7: -0.0021570, 0.0038259, -0.0021570, 0.0038259, -0.0057195, 0.0056755
8: -0.0008533, 0.0000782, -0.0008533, 0.0000782, -0.0008802, 0.0008375
9: 0.9973557, 1.0121032, 0.9973557, 1.0121032, -0.0088985, 0.0088983

### Unstable ReLU Count (Linear/Conv2D Layers)
- layer_idx=0, type=LayerType.Linear, total=256, inp1_unstable=7, inp2_unstable=7, delta_unstable=249
- layer_idx=2, type=LayerType.Linear, total=256, inp1_unstable=10, inp2_unstable=10, delta_unstable=256
- layer_idx=4, type=LayerType.Linear, total=256, inp1_unstable=1, inp2_unstable=1, delta_unstable=256
- layer_idx=6, type=LayerType.Linear, total=256, inp1_unstable=0, inp2_unstable=0, delta_unstable=256
- layer_idx=8, type=LayerType.Linear, total=10, inp1_unstable=4, inp2_unstable=4, delta_unstable=10

Time for backsubstitution: 1.41 seconds

### RS candidates at layer 3
type: RSZ, layer: 3, pos: 8
type: RSZ, layer: 3, pos: 13
type: RSZ, layer: 3, pos: 20
type: RSZ, layer: 3, pos: 35
type: RSZ, layer: 3, pos: 43
type: RSZ, layer: 3, pos: 48
type: RSZ, layer: 3, pos: 103
type: RSZ, layer: 3, pos: 133
type: RSZ, layer: 3, pos: 180
type: RSZ, layer: 3, pos: 183

Time for candidate selection: 0.32 seconds

### Candidate
type: RSZ, layer: 3, pos: 8

### Relational analysis RSZ of RS_RSZ1_RSZ2_RSZ1_RSZ2_RSZ2_RSZ2_RSZ1_RSZ1

#### Relational analysis RSZ result of RS_RSZ1_RSZ2_RSZ1_RSZ2_RSZ2_RSZ2_RSZ1_RSZ1
Status: Status.VERIFIED
Output dim: 9, lower bound: -0.0043991, upper bound: 0.0044823
time: 0.79 seconds

### Relational analysis RSZ of RS_RSZ1_RSZ2_RSZ1_RSZ2_RSZ2_RSZ2_RSZ1_RSZ2

#### Relational analysis RSZ result of RS_RSZ1_RSZ2_RSZ1_RSZ2_RSZ2_RSZ2_RSZ1_RSZ2
Status: Status.VERIFIED
Output dim: 9, lower bound: -0.0043991, upper bound: 0.0044823
time: 0.82 seconds

## BFS RS instance: RS_RSZ1_RSZ2_RSZ1_RSZ2_RSZ2_RSZ2_RSZ2

### Backsubstitution after applying RS history:
0: -0.0039095, -0.0003029, -0.0039095, -0.0003029, -0.0032072, 0.0032330
1: -0.0044907, 0.0040812, -0.0044907, 0.0040812, -0.0040402, 0.0039553
2: 0.0029569, 0.0099431, 0.0029569, 0.0099431, -0.0037698, 0.0037157
3: -0.0045519, -0.0034923, -0.0045519, -0.0034923, -0.0010595, 0.0010595
4: 0.0017018, 0.0078700, 0.0017018, 0.0078700, -0.0052334, 0.0052529
5: -0.0030341, 0.0031953, -0.0030341, 0.0031953, -0.0047246, 0.0046870
6: -0.0068516, -0.0033211, -0.0068516, -0.0033211, -0.0033421, 0.0033350
7: -0.0021570, 0.0038259, -0.0021570, 0.0038259, -0.0057074, 0.0056925
8: -0.0008533, 0.0000782, -0.0008533, 0.0000782, -0.0008806, 0.0008374
9: 0.9973557, 1.0121032, 0.9973557, 1.0121032, -0.0088618, 0.0089635

### Unstable ReLU Count (Linear/Conv2D Layers)
- layer_idx=0, type=LayerType.Linear, total=256, inp1_unstable=7, inp2_unstable=7, delta_unstable=249
- layer_idx=2, type=LayerType.Linear, total=256, inp1_unstable=10, inp2_unstable=10, delta_unstable=256
- layer_idx=4, type=LayerType.Linear, total=256, inp1_unstable=1, inp2_unstable=1, delta_unstable=256
- layer_idx=6, type=LayerType.Linear, total=256, inp1_unstable=0, inp2_unstable=0, delta_unstable=256
- layer_idx=8, type=LayerType.Linear, total=10, inp1_unstable=4, inp2_unstable=4, delta_unstable=10

Time for backsubstitution: 1.37 seconds

### RS candidates at layer 3
type: RSZ, layer: 3, pos: 8
type: RSZ, layer: 3, pos: 13
type: RSZ, layer: 3, pos: 20
type: RSZ, layer: 3, pos: 35
type: RSZ, layer: 3, pos: 43
type: RSZ, layer: 3, pos: 48
type: RSZ, layer: 3, pos: 103
type: RSZ, layer: 3, pos: 133
type: RSZ, layer: 3, pos: 180
type: RSZ, layer: 3, pos: 183

Time for candidate selection: 0.28 seconds

### Candidate
type: RSZ, layer: 3, pos: 8

### Relational analysis RSZ of RS_RSZ1_RSZ2_RSZ1_RSZ2_RSZ2_RSZ2_RSZ2_RSZ1

#### Relational analysis RSZ result of RS_RSZ1_RSZ2_RSZ1_RSZ2_RSZ2_RSZ2_RSZ2_RSZ1
Status: Status.VERIFIED
Output dim: 9, lower bound: -0.0043315, upper bound: 0.0046113
time: 0.85 seconds

### Relational analysis RSZ of RS_RSZ1_RSZ2_RSZ1_RSZ2_RSZ2_RSZ2_RSZ2_RSZ2

#### Relational analysis RSZ result of RS_RSZ1_RSZ2_RSZ1_RSZ2_RSZ2_RSZ2_RSZ2_RSZ2
Status: Status.VERIFIED
Output dim: 9, lower bound: -0.0043315, upper bound: 0.0046112
time: 0.86 seconds

## BFS RS instance: RS_RSZ1_RSZ2_RSZ2_RSZ1_RSZ1_RSZ1_RSZ1

### Backsubstitution after applying RS history:
0: -0.0039095, -0.0003029, -0.0039095, -0.0003029, -0.0032333, 0.0032069
1: -0.0044907, 0.0040812, -0.0044907, 0.0040812, -0.0039738, 0.0040292
2: 0.0029569, 0.0099431, 0.0029569, 0.0099431, -0.0037283, 0.0037664
3: -0.0045519, -0.0034923, -0.0045519, -0.0034923, -0.0010595, 0.0010595
4: 0.0017018, 0.0078700, 0.0017018, 0.0078700, -0.0052538, 0.0052319
5: -0.0030341, 0.0031953, -0.0030341, 0.0031953, -0.0047052, 0.0047110
6: -0.0068516, -0.0033211, -0.0068516, -0.0033211, -0.0033362, 0.0033413
7: -0.0021570, 0.0038259, -0.0021570, 0.0038259, -0.0057205, 0.0056904
8: -0.0008533, 0.0000782, -0.0008533, 0.0000782, -0.0008668, 0.0008487
9: 0.9973557, 1.0121032, 0.9973557, 1.0121032, -0.0089552, 0.0088737

### Unstable ReLU Count (Linear/Conv2D Layers)
- layer_idx=0, type=LayerType.Linear, total=256, inp1_unstable=7, inp2_unstable=7, delta_unstable=249
- layer_idx=2, type=LayerType.Linear, total=256, inp1_unstable=10, inp2_unstable=10, delta_unstable=256
- layer_idx=4, type=LayerType.Linear, total=256, inp1_unstable=1, inp2_unstable=1, delta_unstable=256
- layer_idx=6, type=LayerType.Linear, total=256, inp1_unstable=0, inp2_unstable=0, delta_unstable=256
- layer_idx=8, type=LayerType.Linear, total=10, inp1_unstable=4, inp2_unstable=4, delta_unstable=10

Time for backsubstitution: 1.41 seconds

### RS candidates at layer 3
type: RSZ, layer: 3, pos: 8
type: RSZ, layer: 3, pos: 13
type: RSZ, layer: 3, pos: 20
type: RSZ, layer: 3, pos: 35
type: RSZ, layer: 3, pos: 43
type: RSZ, layer: 3, pos: 48
type: RSZ, layer: 3, pos: 103
type: RSZ, layer: 3, pos: 133
type: RSZ, layer: 3, pos: 180
type: RSZ, layer: 3, pos: 183

Time for candidate selection: 0.29 seconds

### Candidate
type: RSZ, layer: 3, pos: 8

### Relational analysis RSZ of RS_RSZ1_RSZ2_RSZ2_RSZ1_RSZ1_RSZ1_RSZ1_RSZ1

#### Relational analysis RSZ result of RS_RSZ1_RSZ2_RSZ2_RSZ1_RSZ1_RSZ1_RSZ1_RSZ1
Status: Status.VERIFIED
Output dim: 9, lower bound: -0.0045672, upper bound: 0.0043763
time: 0.85 seconds

### Relational analysis RSZ of RS_RSZ1_RSZ2_RSZ2_RSZ1_RSZ1_RSZ1_RSZ1_RSZ2

#### Relational analysis RSZ result of RS_RSZ1_RSZ2_RSZ2_RSZ1_RSZ1_RSZ1_RSZ1_RSZ2
Status: Status.VERIFIED
Output dim: 9, lower bound: -0.0045672, upper bound: 0.0043743
time: 0.87 seconds

## BFS RS instance: RS_RSZ1_RSZ2_RSZ2_RSZ1_RSZ1_RSZ2_RSZ1

### Backsubstitution after applying RS history:
0: -0.0039095, -0.0003029, -0.0039095, -0.0003029, -0.0032327, 0.0032071
1: -0.0044907, 0.0040812, -0.0044907, 0.0040812, -0.0039739, 0.0040291
2: 0.0029569, 0.0099431, 0.0029569, 0.0099431, -0.0037286, 0.0037663
3: -0.0045519, -0.0034923, -0.0045519, -0.0034923, -0.0010595, 0.0010595
4: 0.0017018, 0.0078700, 0.0017018, 0.0078700, -0.0052534, 0.0052321
5: -0.0030341, 0.0031953, -0.0030341, 0.0031953, -0.0047058, 0.0047093
6: -0.0068516, -0.0033211, -0.0068516, -0.0033211, -0.0033362, 0.0033413
7: -0.0021570, 0.0038259, -0.0021570, 0.0038259, -0.0057185, 0.0056909
8: -0.0008533, 0.0000782, -0.0008533, 0.0000782, -0.0008647, 0.0008494
9: 0.9973557, 1.0121032, 0.9973557, 1.0121032, -0.0089547, 0.0088737

### Unstable ReLU Count (Linear/Conv2D Layers)
- layer_idx=0, type=LayerType.Linear, total=256, inp1_unstable=7, inp2_unstable=7, delta_unstable=249
- layer_idx=2, type=LayerType.Linear, total=256, inp1_unstable=10, inp2_unstable=10, delta_unstable=256
- layer_idx=4, type=LayerType.Linear, total=256, inp1_unstable=1, inp2_unstable=1, delta_unstable=256
- layer_idx=6, type=LayerType.Linear, total=256, inp1_unstable=0, inp2_unstable=0, delta_unstable=256
- layer_idx=8, type=LayerType.Linear, total=10, inp1_unstable=4, inp2_unstable=4, delta_unstable=10

Time for backsubstitution: 1.54 seconds

### RS candidates at layer 3
type: RSZ, layer: 3, pos: 8
type: RSZ, layer: 3, pos: 13
type: RSZ, layer: 3, pos: 20
type: RSZ, layer: 3, pos: 35
type: RSZ, layer: 3, pos: 43
type: RSZ, layer: 3, pos: 48
type: RSZ, layer: 3, pos: 103
type: RSZ, layer: 3, pos: 133
type: RSZ, layer: 3, pos: 180
type: RSZ, layer: 3, pos: 183

Time for candidate selection: 0.32 seconds

### Candidate
type: RSZ, layer: 3, pos: 8

### Relational analysis RSZ of RS_RSZ1_RSZ2_RSZ2_RSZ1_RSZ1_RSZ2_RSZ1_RSZ1

#### Relational analysis RSZ result of RS_RSZ1_RSZ2_RSZ2_RSZ1_RSZ1_RSZ2_RSZ1_RSZ1
Status: Status.VERIFIED
Output dim: 9, lower bound: -0.0045672, upper bound: 0.0043763
time: 0.87 seconds

### Relational analysis RSZ of RS_RSZ1_RSZ2_RSZ2_RSZ1_RSZ1_RSZ2_RSZ1_RSZ2

#### Relational analysis RSZ result of RS_RSZ1_RSZ2_RSZ2_RSZ1_RSZ1_RSZ2_RSZ1_RSZ2
Status: Status.VERIFIED
Output dim: 9, lower bound: -0.0045672, upper bound: 0.0043743
time: 0.86 seconds

## BFS RS instance: RS_RSZ1_RSZ2_RSZ2_RSZ1_RSZ2_RSZ1_RSZ1

### Backsubstitution after applying RS history:
0: -0.0039095, -0.0003029, -0.0039095, -0.0003029, -0.0032836, 0.0031603
1: -0.0044907, 0.0040812, -0.0044907, 0.0040812, -0.0038504, 0.0041402
2: 0.0029569, 0.0099431, 0.0029569, 0.0099431, -0.0036408, 0.0038394
3: -0.0045519, -0.0034923, -0.0045519, -0.0034923, -0.0010595, 0.0010595
4: 0.0017018, 0.0078700, 0.0017018, 0.0078700, -0.0052919, 0.0051964
5: -0.0030341, 0.0031953, -0.0030341, 0.0031953, -0.0046466, 0.0047707
6: -0.0068516, -0.0033211, -0.0068516, -0.0033211, -0.0033257, 0.0033512
7: -0.0021570, 0.0038259, -0.0021570, 0.0038259, -0.0057435, 0.0056575
8: -0.0008533, 0.0000782, -0.0008533, 0.0000782, -0.0008621, 0.0008567
9: 0.9973557, 1.0121032, 0.9973557, 1.0121032, -0.0091086, 0.0087169

### Unstable ReLU Count (Linear/Conv2D Layers)
- layer_idx=0, type=LayerType.Linear, total=256, inp1_unstable=7, inp2_unstable=7, delta_unstable=249
- layer_idx=2, type=LayerType.Linear, total=256, inp1_unstable=10, inp2_unstable=10, delta_unstable=256
- layer_idx=4, type=LayerType.Linear, total=256, inp1_unstable=1, inp2_unstable=1, delta_unstable=256
- layer_idx=6, type=LayerType.Linear, total=256, inp1_unstable=0, inp2_unstable=0, delta_unstable=256
- layer_idx=8, type=LayerType.Linear, total=10, inp1_unstable=4, inp2_unstable=4, delta_unstable=10

Time for backsubstitution: 1.40 seconds

### RS candidates at layer 3
type: RSZ, layer: 3, pos: 8
type: RSZ, layer: 3, pos: 13
type: RSZ, layer: 3, pos: 20
type: RSZ, layer: 3, pos: 35
type: RSZ, layer: 3, pos: 43
type: RSZ, layer: 3, pos: 48
type: RSZ, layer: 3, pos: 103
type: RSZ, layer: 3, pos: 133
type: RSZ, layer: 3, pos: 180
type: RSZ, layer: 3, pos: 183

Time for candidate selection: 0.29 seconds

### Candidate
type: RSZ, layer: 3, pos: 8

### Relational analysis RSZ of RS_RSZ1_RSZ2_RSZ2_RSZ1_RSZ2_RSZ1_RSZ1_RSZ1

#### Relational analysis RSZ result of RS_RSZ1_RSZ2_RSZ2_RSZ1_RSZ2_RSZ1_RSZ1_RSZ1
Status: Status.VERIFIED
Output dim: 9, lower bound: -0.0047008, upper bound: 0.0042658
time: 0.83 seconds

### Relational analysis RSZ of RS_RSZ1_RSZ2_RSZ2_RSZ1_RSZ2_RSZ1_RSZ1_RSZ2

#### Relational analysis RSZ result of RS_RSZ1_RSZ2_RSZ2_RSZ1_RSZ2_RSZ1_RSZ1_RSZ2
Status: Status.VERIFIED
Output dim: 9, lower bound: -0.0047008, upper bound: 0.0042658
time: 0.83 seconds

## BFS RS instance: RS_RSZ1_RSZ2_RSZ2_RSZ1_RSZ2_RSZ1_RSZ2

### Backsubstitution after applying RS history:
0: -0.0039095, -0.0003029, -0.0039095, -0.0003029, -0.0032707, 0.0031740
1: -0.0044907, 0.0040812, -0.0044907, 0.0040812, -0.0038937, 0.0040988
2: 0.0029569, 0.0099431, 0.0029569, 0.0099431, -0.0036727, 0.0038093
3: -0.0045519, -0.0034923, -0.0045519, -0.0034923, -0.0010595, 0.0010595
4: 0.0017018, 0.0078700, 0.0017018, 0.0078700, -0.0052817, 0.0052076
5: -0.0030341, 0.0031953, -0.0030341, 0.0031953, -0.0046571, 0.0047593
6: -0.0068516, -0.0033211, -0.0068516, -0.0033211, -0.0033293, 0.0033478
7: -0.0021570, 0.0038259, -0.0021570, 0.0038259, -0.0057287, 0.0056705
8: -0.0008533, 0.0000782, -0.0008533, 0.0000782, -0.0008624, 0.0008564
9: 0.9973557, 1.0121032, 0.9973557, 1.0121032, -0.0090590, 0.0087696

### Unstable ReLU Count (Linear/Conv2D Layers)
- layer_idx=0, type=LayerType.Linear, total=256, inp1_unstable=7, inp2_unstable=7, delta_unstable=249
- layer_idx=2, type=LayerType.Linear, total=256, inp1_unstable=10, inp2_unstable=10, delta_unstable=256
- layer_idx=4, type=LayerType.Linear, total=256, inp1_unstable=1, inp2_unstable=1, delta_unstable=256
- layer_idx=6, type=LayerType.Linear, total=256, inp1_unstable=0, inp2_unstable=0, delta_unstable=256
- layer_idx=8, type=LayerType.Linear, total=10, inp1_unstable=4, inp2_unstable=4, delta_unstable=10

Time for backsubstitution: 1.38 seconds

### RS candidates at layer 3
type: RSZ, layer: 3, pos: 8
type: RSZ, layer: 3, pos: 13
type: RSZ, layer: 3, pos: 20
type: RSZ, layer: 3, pos: 35
type: RSZ, layer: 3, pos: 43
type: RSZ, layer: 3, pos: 48
type: RSZ, layer: 3, pos: 103
type: RSZ, layer: 3, pos: 133
type: RSZ, layer: 3, pos: 180
type: RSZ, layer: 3, pos: 183

Time for candidate selection: 0.28 seconds

### Candidate
type: RSZ, layer: 3, pos: 8

### Relational analysis RSZ of RS_RSZ1_RSZ2_RSZ2_RSZ1_RSZ2_RSZ1_RSZ2_RSZ1

#### Relational analysis RSZ result of RS_RSZ1_RSZ2_RSZ2_RSZ1_RSZ2_RSZ1_RSZ2_RSZ1
Status: Status.VERIFIED
Output dim: 9, lower bound: -0.0045760, upper bound: 0.0043311
time: 0.90 seconds

### Relational analysis RSZ of RS_RSZ1_RSZ2_RSZ2_RSZ1_RSZ2_RSZ1_RSZ2_RSZ2

#### Relational analysis RSZ result of RS_RSZ1_RSZ2_RSZ2_RSZ1_RSZ2_RSZ1_RSZ2_RSZ2
Status: Status.VERIFIED
Output dim: 9, lower bound: -0.0045760, upper bound: 0.0043311
time: 0.91 seconds

## BFS RS instance: RS_RSZ1_RSZ2_RSZ2_RSZ1_RSZ2_RSZ2_RSZ1

### Backsubstitution after applying RS history:
0: -0.0039095, -0.0003029, -0.0039095, -0.0003029, -0.0032831, 0.0031609
1: -0.0044907, 0.0040812, -0.0044907, 0.0040812, -0.0038507, 0.0041402
2: 0.0029569, 0.0099431, 0.0029569, 0.0099431, -0.0036419, 0.0038391
3: -0.0045519, -0.0034923, -0.0045519, -0.0034923, -0.0010595, 0.0010595
4: 0.0017018, 0.0078700, 0.0017018, 0.0078700, -0.0052916, 0.0051969
5: -0.0030341, 0.0031953, -0.0030341, 0.0031953, -0.0046481, 0.0047698
6: -0.0068516, -0.0033211, -0.0068516, -0.0033211, -0.0033257, 0.0033512
7: -0.0021570, 0.0038259, -0.0021570, 0.0038259, -0.0057414, 0.0056577
8: -0.0008533, 0.0000782, -0.0008533, 0.0000782, -0.0008611, 0.0008568
9: 0.9973557, 1.0121032, 0.9973557, 1.0121032, -0.0091084, 0.0087175

### Unstable ReLU Count (Linear/Conv2D Layers)
- layer_idx=0, type=LayerType.Linear, total=256, inp1_unstable=7, inp2_unstable=7, delta_unstable=249
- layer_idx=2, type=LayerType.Linear, total=256, inp1_unstable=10, inp2_unstable=10, delta_unstable=256
- layer_idx=4, type=LayerType.Linear, total=256, inp1_unstable=1, inp2_unstable=1, delta_unstable=256
- layer_idx=6, type=LayerType.Linear, total=256, inp1_unstable=0, inp2_unstable=0, delta_unstable=256
- layer_idx=8, type=LayerType.Linear, total=10, inp1_unstable=4, inp2_unstable=4, delta_unstable=10

Time for backsubstitution: 1.45 seconds

### RS candidates at layer 3
type: RSZ, layer: 3, pos: 8
type: RSZ, layer: 3, pos: 13
type: RSZ, layer: 3, pos: 20
type: RSZ, layer: 3, pos: 35
type: RSZ, layer: 3, pos: 43
type: RSZ, layer: 3, pos: 48
type: RSZ, layer: 3, pos: 103
type: RSZ, layer: 3, pos: 133
type: RSZ, layer: 3, pos: 180
type: RSZ, layer: 3, pos: 183

Time for candidate selection: 0.28 seconds

### Candidate
type: RSZ, layer: 3, pos: 8

### Relational analysis RSZ of RS_RSZ1_RSZ2_RSZ2_RSZ1_RSZ2_RSZ2_RSZ1_RSZ1

#### Relational analysis RSZ result of RS_RSZ1_RSZ2_RSZ2_RSZ1_RSZ2_RSZ2_RSZ1_RSZ1
Status: Status.VERIFIED
Output dim: 9, lower bound: -0.0047008, upper bound: 0.0042658
time: 0.82 seconds

### Relational analysis RSZ of RS_RSZ1_RSZ2_RSZ2_RSZ1_RSZ2_RSZ2_RSZ1_RSZ2

#### Relational analysis RSZ result of RS_RSZ1_RSZ2_RSZ2_RSZ1_RSZ2_RSZ2_RSZ1_RSZ2
Status: Status.VERIFIED
Output dim: 9, lower bound: -0.0047008, upper bound: 0.0042658
time: 0.81 seconds

## BFS RS instance: RS_RSZ1_RSZ2_RSZ2_RSZ1_RSZ2_RSZ2_RSZ2

### Backsubstitution after applying RS history:
0: -0.0039095, -0.0003029, -0.0039095, -0.0003029, -0.0032704, 0.0031749
1: -0.0044907, 0.0040812, -0.0044907, 0.0040812, -0.0038941, 0.0040989
2: 0.0029569, 0.0099431, 0.0029569, 0.0099431, -0.0036731, 0.0038093
3: -0.0045519, -0.0034923, -0.0045519, -0.0034923, -0.0010595, 0.0010595
4: 0.0017018, 0.0078700, 0.0017018, 0.0078700, -0.0052816, 0.0052082
5: -0.0030341, 0.0031953, -0.0030341, 0.0031953, -0.0046583, 0.0047588
6: -0.0068516, -0.0033211, -0.0068516, -0.0033211, -0.0033293, 0.0033478
7: -0.0021570, 0.0038259, -0.0021570, 0.0038259, -0.0057279, 0.0056725
8: -0.0008533, 0.0000782, -0.0008533, 0.0000782, -0.0008616, 0.0008566
9: 0.9973557, 1.0121032, 0.9973557, 1.0121032, -0.0090590, 0.0087704

### Unstable ReLU Count (Linear/Conv2D Layers)
- layer_idx=0, type=LayerType.Linear, total=256, inp1_unstable=7, inp2_unstable=7, delta_unstable=249
- layer_idx=2, type=LayerType.Linear, total=256, inp1_unstable=10, inp2_unstable=10, delta_unstable=256
- layer_idx=4, type=LayerType.Linear, total=256, inp1_unstable=1, inp2_unstable=1, delta_unstable=256
- layer_idx=6, type=LayerType.Linear, total=256, inp1_unstable=0, inp2_unstable=0, delta_unstable=256
- layer_idx=8, type=LayerType.Linear, total=10, inp1_unstable=4, inp2_unstable=4, delta_unstable=10

Time for backsubstitution: 1.41 seconds

### RS candidates at layer 3
type: RSZ, layer: 3, pos: 8
type: RSZ, layer: 3, pos: 13
type: RSZ, layer: 3, pos: 20
type: RSZ, layer: 3, pos: 35
type: RSZ, layer: 3, pos: 43
type: RSZ, layer: 3, pos: 48
type: RSZ, layer: 3, pos: 103
type: RSZ, layer: 3, pos: 133
type: RSZ, layer: 3, pos: 180
type: RSZ, layer: 3, pos: 183

Time for candidate selection: 0.34 seconds

### Candidate
type: RSZ, layer: 3, pos: 8

### Relational analysis RSZ of RS_RSZ1_RSZ2_RSZ2_RSZ1_RSZ2_RSZ2_RSZ2_RSZ1

#### Relational analysis RSZ result of RS_RSZ1_RSZ2_RSZ2_RSZ1_RSZ2_RSZ2_RSZ2_RSZ1
Status: Status.VERIFIED
Output dim: 9, lower bound: -0.0045760, upper bound: 0.0043311
time: 0.90 seconds

### Relational analysis RSZ of RS_RSZ1_RSZ2_RSZ2_RSZ1_RSZ2_RSZ2_RSZ2_RSZ2

#### Relational analysis RSZ result of RS_RSZ1_RSZ2_RSZ2_RSZ1_RSZ2_RSZ2_RSZ2_RSZ2
Status: Status.VERIFIED
Output dim: 9, lower bound: -0.0045760, upper bound: 0.0043311
time: 0.89 seconds

## BFS RS instance: RS_RSZ1_RSZ2_RSZ2_RSZ2_RSZ1_RSZ1_RSZ1

### Backsubstitution after applying RS history:
0: -0.0039095, -0.0003029, -0.0039095, -0.0003029, -0.0031576, 0.0032768
1: -0.0044907, 0.0040812, -0.0044907, 0.0040812, -0.0041322, 0.0038443
2: 0.0029569, 0.0099431, 0.0029569, 0.0099431, -0.0038319, 0.0036438
3: -0.0045519, -0.0034923, -0.0045519, -0.0034923, -0.0010595, 0.0010595
4: 0.0017018, 0.0078700, 0.0017018, 0.0078700, -0.0051958, 0.0052852
5: -0.0030341, 0.0031953, -0.0030341, 0.0031953, -0.0047821, 0.0046315
6: -0.0068516, -0.0033211, -0.0068516, -0.0033211, -0.0033505, 0.0033250
7: -0.0021570, 0.0038259, -0.0021570, 0.0038259, -0.0056934, 0.0057124
8: -0.0008533, 0.0000782, -0.0008533, 0.0000782, -0.0008760, 0.0008398
9: 0.9973557, 1.0121032, 0.9973557, 1.0121032, -0.0087070, 0.0090938

### Unstable ReLU Count (Linear/Conv2D Layers)
- layer_idx=0, type=LayerType.Linear, total=256, inp1_unstable=7, inp2_unstable=7, delta_unstable=249
- layer_idx=2, type=LayerType.Linear, total=256, inp1_unstable=10, inp2_unstable=10, delta_unstable=256
- layer_idx=4, type=LayerType.Linear, total=256, inp1_unstable=1, inp2_unstable=1, delta_unstable=256
- layer_idx=6, type=LayerType.Linear, total=256, inp1_unstable=0, inp2_unstable=0, delta_unstable=256
- layer_idx=8, type=LayerType.Linear, total=10, inp1_unstable=4, inp2_unstable=4, delta_unstable=10

Time for backsubstitution: 1.39 seconds

### RS candidates at layer 3
type: RSZ, layer: 3, pos: 8
type: RSZ, layer: 3, pos: 13
type: RSZ, layer: 3, pos: 20
type: RSZ, layer: 3, pos: 35
type: RSZ, layer: 3, pos: 43
type: RSZ, layer: 3, pos: 48
type: RSZ, layer: 3, pos: 103
type: RSZ, layer: 3, pos: 133
type: RSZ, layer: 3, pos: 180
type: RSZ, layer: 3, pos: 183

Time for candidate selection: 0.30 seconds

### Candidate
type: RSZ, layer: 3, pos: 8

### Relational analysis RSZ of RS_RSZ1_RSZ2_RSZ2_RSZ2_RSZ1_RSZ1_RSZ1_RSZ1

#### Relational analysis RSZ result of RS_RSZ1_RSZ2_RSZ2_RSZ2_RSZ1_RSZ1_RSZ1_RSZ1
Status: Status.VERIFIED
Output dim: 9, lower bound: -0.0042269, upper bound: 0.0046439
time: 0.93 seconds

### Relational analysis RSZ of RS_RSZ1_RSZ2_RSZ2_RSZ2_RSZ1_RSZ1_RSZ1_RSZ2

#### Relational analysis RSZ result of RS_RSZ1_RSZ2_RSZ2_RSZ2_RSZ1_RSZ1_RSZ1_RSZ2
Status: Status.VERIFIED
Output dim: 9, lower bound: -0.0042269, upper bound: 0.0046436
time: 0.93 seconds

## BFS RS instance: RS_RSZ1_RSZ2_RSZ2_RSZ2_RSZ1_RSZ1_RSZ2

### Backsubstitution after applying RS history:
0: -0.0039095, -0.0003029, -0.0039095, -0.0003029, -0.0031443, 0.0032888
1: -0.0044907, 0.0040812, -0.0044907, 0.0040812, -0.0041735, 0.0037988
2: 0.0029569, 0.0099431, 0.0029569, 0.0099431, -0.0038603, 0.0036105
3: -0.0045519, -0.0034923, -0.0045519, -0.0034923, -0.0010595, 0.0010595
4: 0.0017018, 0.0078700, 0.0017018, 0.0078700, -0.0051851, 0.0052948
5: -0.0030341, 0.0031953, -0.0030341, 0.0031953, -0.0047927, 0.0046211
6: -0.0068516, -0.0033211, -0.0068516, -0.0033211, -0.0033539, 0.0033214
7: -0.0021570, 0.0038259, -0.0021570, 0.0038259, -0.0056776, 0.0057239
8: -0.0008533, 0.0000782, -0.0008533, 0.0000782, -0.0008763, 0.0008396
9: 0.9973557, 1.0121032, 0.9973557, 1.0121032, -0.0086547, 0.0091427

### Unstable ReLU Count (Linear/Conv2D Layers)
- layer_idx=0, type=LayerType.Linear, total=256, inp1_unstable=7, inp2_unstable=7, delta_unstable=249
- layer_idx=2, type=LayerType.Linear, total=256, inp1_unstable=10, inp2_unstable=10, delta_unstable=256
- layer_idx=4, type=LayerType.Linear, total=256, inp1_unstable=1, inp2_unstable=1, delta_unstable=256
- layer_idx=6, type=LayerType.Linear, total=256, inp1_unstable=0, inp2_unstable=0, delta_unstable=256
- layer_idx=8, type=LayerType.Linear, total=10, inp1_unstable=4, inp2_unstable=4, delta_unstable=10

Time for backsubstitution: 1.49 seconds

### RS candidates at layer 3
type: RSZ, layer: 3, pos: 8
type: RSZ, layer: 3, pos: 13
type: RSZ, layer: 3, pos: 20
type: RSZ, layer: 3, pos: 35
type: RSZ, layer: 3, pos: 43
type: RSZ, layer: 3, pos: 48
type: RSZ, layer: 3, pos: 103
type: RSZ, layer: 3, pos: 133
type: RSZ, layer: 3, pos: 180
type: RSZ, layer: 3, pos: 183

Time for candidate selection: 0.35 seconds

### Candidate
type: RSZ, layer: 3, pos: 8

### Relational analysis RSZ of RS_RSZ1_RSZ2_RSZ2_RSZ2_RSZ1_RSZ1_RSZ2_RSZ1

#### Relational analysis RSZ result of RS_RSZ1_RSZ2_RSZ2_RSZ2_RSZ1_RSZ1_RSZ2_RSZ1
Status: Status.VERIFIED
Output dim: 9, lower bound: -0.0041728, upper bound: 0.0047617
time: 0.90 seconds

### Relational analysis RSZ of RS_RSZ1_RSZ2_RSZ2_RSZ2_RSZ1_RSZ1_RSZ2_RSZ2

#### Relational analysis RSZ result of RS_RSZ1_RSZ2_RSZ2_RSZ2_RSZ1_RSZ1_RSZ2_RSZ2
Status: Status.VERIFIED
Output dim: 9, lower bound: -0.0041728, upper bound: 0.0047617
time: 0.90 seconds

## BFS RS instance: RS_RSZ1_RSZ2_RSZ2_RSZ2_RSZ1_RSZ2_RSZ1

### Backsubstitution after applying RS history:
0: -0.0039095, -0.0003029, -0.0039095, -0.0003029, -0.0031568, 0.0032770
1: -0.0044907, 0.0040812, -0.0044907, 0.0040812, -0.0041322, 0.0038437
2: 0.0029569, 0.0099431, 0.0029569, 0.0099431, -0.0038319, 0.0036426
3: -0.0045519, -0.0034923, -0.0045519, -0.0034923, -0.0010595, 0.0010595
4: 0.0017018, 0.0078700, 0.0017018, 0.0078700, -0.0051953, 0.0052853
5: -0.0030341, 0.0031953, -0.0030341, 0.0031953, -0.0047819, 0.0046288
6: -0.0068516, -0.0033211, -0.0068516, -0.0033211, -0.0033505, 0.0033250
7: -0.0021570, 0.0038259, -0.0021570, 0.0038259, -0.0056902, 0.0057116
8: -0.0008533, 0.0000782, -0.0008533, 0.0000782, -0.0008744, 0.0008388
9: 0.9973557, 1.0121032, 0.9973557, 1.0121032, -0.0087059, 0.0090937

### Unstable ReLU Count (Linear/Conv2D Layers)
- layer_idx=0, type=LayerType.Linear, total=256, inp1_unstable=7, inp2_unstable=7, delta_unstable=249
- layer_idx=2, type=LayerType.Linear, total=256, inp1_unstable=10, inp2_unstable=10, delta_unstable=256
- layer_idx=4, type=LayerType.Linear, total=256, inp1_unstable=1, inp2_unstable=1, delta_unstable=256
- layer_idx=6, type=LayerType.Linear, total=256, inp1_unstable=0, inp2_unstable=0, delta_unstable=256
- layer_idx=8, type=LayerType.Linear, total=10, inp1_unstable=4, inp2_unstable=4, delta_unstable=10

Time for backsubstitution: 1.37 seconds

### RS candidates at layer 3
type: RSZ, layer: 3, pos: 8
type: RSZ, layer: 3, pos: 13
type: RSZ, layer: 3, pos: 20
type: RSZ, layer: 3, pos: 35
type: RSZ, layer: 3, pos: 43
type: RSZ, layer: 3, pos: 48
type: RSZ, layer: 3, pos: 103
type: RSZ, layer: 3, pos: 133
type: RSZ, layer: 3, pos: 180
type: RSZ, layer: 3, pos: 183

Time for candidate selection: 0.34 seconds

### Candidate
type: RSZ, layer: 3, pos: 8

### Relational analysis RSZ of RS_RSZ1_RSZ2_RSZ2_RSZ2_RSZ1_RSZ2_RSZ1_RSZ1

#### Relational analysis RSZ result of RS_RSZ1_RSZ2_RSZ2_RSZ2_RSZ1_RSZ2_RSZ1_RSZ1
Status: Status.VERIFIED
Output dim: 9, lower bound: -0.0042269, upper bound: 0.0046439
time: 0.94 seconds

### Relational analysis RSZ of RS_RSZ1_RSZ2_RSZ2_RSZ2_RSZ1_RSZ2_RSZ1_RSZ2

#### Relational analysis RSZ result of RS_RSZ1_RSZ2_RSZ2_RSZ2_RSZ1_RSZ2_RSZ1_RSZ2
Status: Status.VERIFIED
Output dim: 9, lower bound: -0.0042269, upper bound: 0.0046436
time: 0.89 seconds

## BFS RS instance: RS_RSZ1_RSZ2_RSZ2_RSZ2_RSZ1_RSZ2_RSZ2

### Backsubstitution after applying RS history:
0: -0.0039095, -0.0003029, -0.0039095, -0.0003029, -0.0031434, 0.0032891
1: -0.0044907, 0.0040812, -0.0044907, 0.0040812, -0.0041736, 0.0037984
2: 0.0029569, 0.0099431, 0.0029569, 0.0099431, -0.0038605, 0.0036106
3: -0.0045519, -0.0034923, -0.0045519, -0.0034923, -0.0010595, 0.0010595
4: 0.0017018, 0.0078700, 0.0017018, 0.0078700, -0.0051846, 0.0052950
5: -0.0030341, 0.0031953, -0.0030341, 0.0031953, -0.0047932, 0.0046180
6: -0.0068516, -0.0033211, -0.0068516, -0.0033211, -0.0033539, 0.0033213
7: -0.0021570, 0.0038259, -0.0021570, 0.0038259, -0.0056766, 0.0057241
8: -0.0008533, 0.0000782, -0.0008533, 0.0000782, -0.0008750, 0.0008388
9: 0.9973557, 1.0121032, 0.9973557, 1.0121032, -0.0086539, 0.0091427

### Unstable ReLU Count (Linear/Conv2D Layers)
- layer_idx=0, type=LayerType.Linear, total=256, inp1_unstable=7, inp2_unstable=7, delta_unstable=249
- layer_idx=2, type=LayerType.Linear, total=256, inp1_unstable=10, inp2_unstable=10, delta_unstable=256
- layer_idx=4, type=LayerType.Linear, total=256, inp1_unstable=1, inp2_unstable=1, delta_unstable=256
- layer_idx=6, type=LayerType.Linear, total=256, inp1_unstable=0, inp2_unstable=0, delta_unstable=256
- layer_idx=8, type=LayerType.Linear, total=10, inp1_unstable=4, inp2_unstable=4, delta_unstable=10

Time for backsubstitution: 1.44 seconds

### RS candidates at layer 3
type: RSZ, layer: 3, pos: 8
type: RSZ, layer: 3, pos: 13
type: RSZ, layer: 3, pos: 20
type: RSZ, layer: 3, pos: 35
type: RSZ, layer: 3, pos: 43
type: RSZ, layer: 3, pos: 48
type: RSZ, layer: 3, pos: 103
type: RSZ, layer: 3, pos: 133
type: RSZ, layer: 3, pos: 180
type: RSZ, layer: 3, pos: 183

Time for candidate selection: 0.28 seconds

### Candidate
type: RSZ, layer: 3, pos: 8

### Relational analysis RSZ of RS_RSZ1_RSZ2_RSZ2_RSZ2_RSZ1_RSZ2_RSZ2_RSZ1

#### Relational analysis RSZ result of RS_RSZ1_RSZ2_RSZ2_RSZ2_RSZ1_RSZ2_RSZ2_RSZ1
Status: Status.VERIFIED
Output dim: 9, lower bound: -0.0041728, upper bound: 0.0047617
time: 0.89 seconds

### Relational analysis RSZ of RS_RSZ1_RSZ2_RSZ2_RSZ2_RSZ1_RSZ2_RSZ2_RSZ2

#### Relational analysis RSZ result of RS_RSZ1_RSZ2_RSZ2_RSZ2_RSZ1_RSZ2_RSZ2_RSZ2
Status: Status.VERIFIED
Output dim: 9, lower bound: -0.0041728, upper bound: 0.0047617
time: 0.91 seconds

## BFS RS instance: RS_RSZ1_RSZ2_RSZ2_RSZ2_RSZ2_RSZ1_RSZ1

### Backsubstitution after applying RS history:
0: -0.0039095, -0.0003029, -0.0039095, -0.0003029, -0.0032112, 0.0032280
1: -0.0044907, 0.0040812, -0.0044907, 0.0040812, -0.0040106, 0.0039711
2: 0.0029569, 0.0099431, 0.0029569, 0.0099431, -0.0037459, 0.0037317
3: -0.0045519, -0.0034923, -0.0045519, -0.0034923, -0.0010595, 0.0010595
4: 0.0017018, 0.0078700, 0.0017018, 0.0078700, -0.0052368, 0.0052482
5: -0.0030341, 0.0031953, -0.0030341, 0.0031953, -0.0047235, 0.0046889
6: -0.0068516, -0.0033211, -0.0068516, -0.0033211, -0.0033400, 0.0033359
7: -0.0021570, 0.0038259, -0.0021570, 0.0038259, -0.0057166, 0.0056804
8: -0.0008533, 0.0000782, -0.0008533, 0.0000782, -0.0008692, 0.0008473
9: 0.9973557, 1.0121032, 0.9973557, 1.0121032, -0.0088761, 0.0089345

### Unstable ReLU Count (Linear/Conv2D Layers)
- layer_idx=0, type=LayerType.Linear, total=256, inp1_unstable=7, inp2_unstable=7, delta_unstable=249
- layer_idx=2, type=LayerType.Linear, total=256, inp1_unstable=10, inp2_unstable=10, delta_unstable=256
- layer_idx=4, type=LayerType.Linear, total=256, inp1_unstable=1, inp2_unstable=1, delta_unstable=256
- layer_idx=6, type=LayerType.Linear, total=256, inp1_unstable=0, inp2_unstable=0, delta_unstable=256
- layer_idx=8, type=LayerType.Linear, total=10, inp1_unstable=4, inp2_unstable=4, delta_unstable=10

Time for backsubstitution: 1.41 seconds

### RS candidates at layer 3
type: RSZ, layer: 3, pos: 8
type: RSZ, layer: 3, pos: 13
type: RSZ, layer: 3, pos: 20
type: RSZ, layer: 3, pos: 35
type: RSZ, layer: 3, pos: 43
type: RSZ, layer: 3, pos: 48
type: RSZ, layer: 3, pos: 103
type: RSZ, layer: 3, pos: 133
type: RSZ, layer: 3, pos: 180
type: RSZ, layer: 3, pos: 183

Time for candidate selection: 0.28 seconds

### Candidate
type: RSZ, layer: 3, pos: 8

### Relational analysis RSZ of RS_RSZ1_RSZ2_RSZ2_RSZ2_RSZ2_RSZ1_RSZ1_RSZ1

#### Relational analysis RSZ result of RS_RSZ1_RSZ2_RSZ2_RSZ2_RSZ2_RSZ1_RSZ1_RSZ1
Status: Status.VERIFIED
Output dim: 9, lower bound: -0.0043521, upper bound: 0.0044960
time: 0.80 seconds

### Relational analysis RSZ of RS_RSZ1_RSZ2_RSZ2_RSZ2_RSZ2_RSZ1_RSZ1_RSZ2

#### Relational analysis RSZ result of RS_RSZ1_RSZ2_RSZ2_RSZ2_RSZ2_RSZ1_RSZ1_RSZ2
Status: Status.VERIFIED
Output dim: 9, lower bound: -0.0043524, upper bound: 0.0044960
time: 0.79 seconds

## BFS RS instance: RS_RSZ1_RSZ2_RSZ2_RSZ2_RSZ2_RSZ1_RSZ2

### Backsubstitution after applying RS history:
0: -0.0039095, -0.0003029, -0.0039095, -0.0003029, -0.0032013, 0.0032454
1: -0.0044907, 0.0040812, -0.0044907, 0.0040812, -0.0040649, 0.0039414
2: 0.0029569, 0.0099431, 0.0029569, 0.0099431, -0.0037847, 0.0037072
3: -0.0045519, -0.0034923, -0.0045519, -0.0034923, -0.0010595, 0.0010595
4: 0.0017018, 0.0078700, 0.0017018, 0.0078700, -0.0052289, 0.0052620
5: -0.0030341, 0.0031953, -0.0030341, 0.0031953, -0.0047385, 0.0046808
6: -0.0068516, -0.0033211, -0.0068516, -0.0033211, -0.0033445, 0.0033335
7: -0.0021570, 0.0038259, -0.0021570, 0.0038259, -0.0057040, 0.0056970
8: -0.0008533, 0.0000782, -0.0008533, 0.0000782, -0.0008699, 0.0008472
9: 0.9973557, 1.0121032, 0.9973557, 1.0121032, -0.0088380, 0.0090013

### Unstable ReLU Count (Linear/Conv2D Layers)
- layer_idx=0, type=LayerType.Linear, total=256, inp1_unstable=7, inp2_unstable=7, delta_unstable=249
- layer_idx=2, type=LayerType.Linear, total=256, inp1_unstable=10, inp2_unstable=10, delta_unstable=256
- layer_idx=4, type=LayerType.Linear, total=256, inp1_unstable=1, inp2_unstable=1, delta_unstable=256
- layer_idx=6, type=LayerType.Linear, total=256, inp1_unstable=0, inp2_unstable=0, delta_unstable=256
- layer_idx=8, type=LayerType.Linear, total=10, inp1_unstable=4, inp2_unstable=4, delta_unstable=10

Time for backsubstitution: 1.35 seconds

### RS candidates at layer 3
type: RSZ, layer: 3, pos: 8
type: RSZ, layer: 3, pos: 13
type: RSZ, layer: 3, pos: 20
type: RSZ, layer: 3, pos: 35
type: RSZ, layer: 3, pos: 43
type: RSZ, layer: 3, pos: 48
type: RSZ, layer: 3, pos: 103
type: RSZ, layer: 3, pos: 133
type: RSZ, layer: 3, pos: 180
type: RSZ, layer: 3, pos: 183

Time for candidate selection: 0.28 seconds

### Candidate
type: RSZ, layer: 3, pos: 8

### Relational analysis RSZ of RS_RSZ1_RSZ2_RSZ2_RSZ2_RSZ2_RSZ1_RSZ2_RSZ1

#### Relational analysis RSZ result of RS_RSZ1_RSZ2_RSZ2_RSZ2_RSZ2_RSZ1_RSZ2_RSZ1
Status: Status.VERIFIED
Output dim: 9, lower bound: -0.0043099, upper bound: 0.0046410
time: 0.84 seconds

### Relational analysis RSZ of RS_RSZ1_RSZ2_RSZ2_RSZ2_RSZ2_RSZ1_RSZ2_RSZ2

#### Relational analysis RSZ result of RS_RSZ1_RSZ2_RSZ2_RSZ2_RSZ2_RSZ1_RSZ2_RSZ2
Status: Status.VERIFIED
Output dim: 9, lower bound: -0.0043105, upper bound: 0.0046410
time: 0.88 seconds

## BFS RS instance: RS_RSZ1_RSZ2_RSZ2_RSZ2_RSZ2_RSZ2_RSZ1

### Backsubstitution after applying RS history:
0: -0.0039095, -0.0003029, -0.0039095, -0.0003029, -0.0032110, 0.0032283
1: -0.0044907, 0.0040812, -0.0044907, 0.0040812, -0.0040109, 0.0039710
2: 0.0029569, 0.0099431, 0.0029569, 0.0099431, -0.0037464, 0.0037316
3: -0.0045519, -0.0034923, -0.0045519, -0.0034923, -0.0010595, 0.0010595
4: 0.0017018, 0.0078700, 0.0017018, 0.0078700, -0.0052366, 0.0052485
5: -0.0030341, 0.0031953, -0.0030341, 0.0031953, -0.0047237, 0.0046875
6: -0.0068516, -0.0033211, -0.0068516, -0.0033211, -0.0033400, 0.0033360
7: -0.0021570, 0.0038259, -0.0021570, 0.0038259, -0.0057154, 0.0056810
8: -0.0008533, 0.0000782, -0.0008533, 0.0000782, -0.0008680, 0.0008461
9: 0.9973557, 1.0121032, 0.9973557, 1.0121032, -0.0088756, 0.0089351

### Unstable ReLU Count (Linear/Conv2D Layers)
- layer_idx=0, type=LayerType.Linear, total=256, inp1_unstable=7, inp2_unstable=7, delta_unstable=249
- layer_idx=2, type=LayerType.Linear, total=256, inp1_unstable=10, inp2_unstable=10, delta_unstable=256
- layer_idx=4, type=LayerType.Linear, total=256, inp1_unstable=1, inp2_unstable=1, delta_unstable=256
- layer_idx=6, type=LayerType.Linear, total=256, inp1_unstable=0, inp2_unstable=0, delta_unstable=256
- layer_idx=8, type=LayerType.Linear, total=10, inp1_unstable=4, inp2_unstable=4, delta_unstable=10

Time for backsubstitution: 1.39 seconds

### RS candidates at layer 3
type: RSZ, layer: 3, pos: 8
type: RSZ, layer: 3, pos: 13
type: RSZ, layer: 3, pos: 20
type: RSZ, layer: 3, pos: 35
type: RSZ, layer: 3, pos: 43
type: RSZ, layer: 3, pos: 48
type: RSZ, layer: 3, pos: 103
type: RSZ, layer: 3, pos: 133
type: RSZ, layer: 3, pos: 180
type: RSZ, layer: 3, pos: 183

Time for candidate selection: 0.32 seconds

### Candidate
type: RSZ, layer: 3, pos: 8

### Relational analysis RSZ of RS_RSZ1_RSZ2_RSZ2_RSZ2_RSZ2_RSZ2_RSZ1_RSZ1

#### Relational analysis RSZ result of RS_RSZ1_RSZ2_RSZ2_RSZ2_RSZ2_RSZ2_RSZ1_RSZ1
Status: Status.VERIFIED
Output dim: 9, lower bound: -0.0043521, upper bound: 0.0044960
time: 0.78 seconds

### Relational analysis RSZ of RS_RSZ1_RSZ2_RSZ2_RSZ2_RSZ2_RSZ2_RSZ1_RSZ2

#### Relational analysis RSZ result of RS_RSZ1_RSZ2_RSZ2_RSZ2_RSZ2_RSZ2_RSZ1_RSZ2
Status: Status.VERIFIED
Output dim: 9, lower bound: -0.0043524, upper bound: 0.0044960
time: 0.80 seconds

## BFS RS instance: RS_RSZ1_RSZ2_RSZ2_RSZ2_RSZ2_RSZ2_RSZ2

### Backsubstitution after applying RS history:
0: -0.0039095, -0.0003029, -0.0039095, -0.0003029, -0.0032010, 0.0032460
1: -0.0044907, 0.0040812, -0.0044907, 0.0040812, -0.0040652, 0.0039416
2: 0.0029569, 0.0099431, 0.0029569, 0.0099431, -0.0037855, 0.0037072
3: -0.0045519, -0.0034923, -0.0045519, -0.0034923, -0.0010595, 0.0010595
4: 0.0017018, 0.0078700, 0.0017018, 0.0078700, -0.0052286, 0.0052624
5: -0.0030341, 0.0031953, -0.0030341, 0.0031953, -0.0047387, 0.0046794
6: -0.0068516, -0.0033211, -0.0068516, -0.0033211, -0.0033445, 0.0033335
7: -0.0021570, 0.0038259, -0.0021570, 0.0038259, -0.0057034, 0.0056984
8: -0.0008533, 0.0000782, -0.0008533, 0.0000782, -0.0008685, 0.0008461
9: 0.9973557, 1.0121032, 0.9973557, 1.0121032, -0.0088380, 0.0090020

### Unstable ReLU Count (Linear/Conv2D Layers)
- layer_idx=0, type=LayerType.Linear, total=256, inp1_unstable=7, inp2_unstable=7, delta_unstable=249
- layer_idx=2, type=LayerType.Linear, total=256, inp1_unstable=10, inp2_unstable=10, delta_unstable=256
- layer_idx=4, type=LayerType.Linear, total=256, inp1_unstable=1, inp2_unstable=1, delta_unstable=256
- layer_idx=6, type=LayerType.Linear, total=256, inp1_unstable=0, inp2_unstable=0, delta_unstable=256
- layer_idx=8, type=LayerType.Linear, total=10, inp1_unstable=4, inp2_unstable=4, delta_unstable=10

Time for backsubstitution: 1.39 seconds

### RS candidates at layer 3
type: RSZ, layer: 3, pos: 8
type: RSZ, layer: 3, pos: 13
type: RSZ, layer: 3, pos: 20
type: RSZ, layer: 3, pos: 35
type: RSZ, layer: 3, pos: 43
type: RSZ, layer: 3, pos: 48
type: RSZ, layer: 3, pos: 103
type: RSZ, layer: 3, pos: 133
type: RSZ, layer: 3, pos: 180
type: RSZ, layer: 3, pos: 183

Time for candidate selection: 0.27 seconds

### Candidate
type: RSZ, layer: 3, pos: 8

### Relational analysis RSZ of RS_RSZ1_RSZ2_RSZ2_RSZ2_RSZ2_RSZ2_RSZ2_RSZ1

#### Relational analysis RSZ result of RS_RSZ1_RSZ2_RSZ2_RSZ2_RSZ2_RSZ2_RSZ2_RSZ1
Status: Status.VERIFIED
Output dim: 9, lower bound: -0.0043099, upper bound: 0.0046410
time: 0.88 seconds

### Relational analysis RSZ of RS_RSZ1_RSZ2_RSZ2_RSZ2_RSZ2_RSZ2_RSZ2_RSZ2

#### Relational analysis RSZ result of RS_RSZ1_RSZ2_RSZ2_RSZ2_RSZ2_RSZ2_RSZ2_RSZ2
Status: Status.VERIFIED
Output dim: 9, lower bound: -0.0043105, upper bound: 0.0046410
time: 0.86 seconds

## BFS RS instance: RS_RSZ2_RSZ1_RSZ1_RSZ1_RSZ1_RSZ1_RSZ1

### Backsubstitution after applying RS history:
0: -0.0039095, -0.0003029, -0.0039095, -0.0003029, -0.0032460, 0.0032010
1: -0.0044907, 0.0040812, -0.0044907, 0.0040812, -0.0039416, 0.0040652
2: 0.0029569, 0.0099431, 0.0029569, 0.0099431, -0.0037072, 0.0037855
3: -0.0045519, -0.0034923, -0.0045519, -0.0034923, -0.0010595, 0.0010595
4: 0.0017018, 0.0078700, 0.0017018, 0.0078700, -0.0052624, 0.0052286
5: -0.0030341, 0.0031953, -0.0030341, 0.0031953, -0.0046794, 0.0047387
6: -0.0068516, -0.0033211, -0.0068516, -0.0033211, -0.0033335, 0.0033445
7: -0.0021570, 0.0038259, -0.0021570, 0.0038259, -0.0056984, 0.0057034
8: -0.0008533, 0.0000782, -0.0008533, 0.0000782, -0.0008461, 0.0008685
9: 0.9973557, 1.0121032, 0.9973557, 1.0121032, -0.0090020, 0.0088380

### Unstable ReLU Count (Linear/Conv2D Layers)
- layer_idx=0, type=LayerType.Linear, total=256, inp1_unstable=7, inp2_unstable=7, delta_unstable=249
- layer_idx=2, type=LayerType.Linear, total=256, inp1_unstable=10, inp2_unstable=10, delta_unstable=256
- layer_idx=4, type=LayerType.Linear, total=256, inp1_unstable=1, inp2_unstable=1, delta_unstable=256
- layer_idx=6, type=LayerType.Linear, total=256, inp1_unstable=0, inp2_unstable=0, delta_unstable=256
- layer_idx=8, type=LayerType.Linear, total=10, inp1_unstable=4, inp2_unstable=4, delta_unstable=10

Time for backsubstitution: 1.42 seconds

### RS candidates at layer 3
type: RSZ, layer: 3, pos: 8
type: RSZ, layer: 3, pos: 13
type: RSZ, layer: 3, pos: 20
type: RSZ, layer: 3, pos: 35
type: RSZ, layer: 3, pos: 43
type: RSZ, layer: 3, pos: 48
type: RSZ, layer: 3, pos: 103
type: RSZ, layer: 3, pos: 133
type: RSZ, layer: 3, pos: 180
type: RSZ, layer: 3, pos: 183

Time for candidate selection: 0.35 seconds

### Candidate
type: RSZ, layer: 3, pos: 8

### Relational analysis RSZ of RS_RSZ2_RSZ1_RSZ1_RSZ1_RSZ1_RSZ1_RSZ1_RSZ1

#### Relational analysis RSZ result of RS_RSZ2_RSZ1_RSZ1_RSZ1_RSZ1_RSZ1_RSZ1_RSZ1
Status: Status.VERIFIED
Output dim: 9, lower bound: -0.0046410, upper bound: 0.0043105
time: 0.84 seconds

### Relational analysis RSZ of RS_RSZ2_RSZ1_RSZ1_RSZ1_RSZ1_RSZ1_RSZ1_RSZ2

#### Relational analysis RSZ result of RS_RSZ2_RSZ1_RSZ1_RSZ1_RSZ1_RSZ1_RSZ1_RSZ2
Status: Status.VERIFIED
Output dim: 9, lower bound: -0.0046410, upper bound: 0.0043099
time: 0.83 seconds

## BFS RS instance: RS_RSZ2_RSZ1_RSZ1_RSZ1_RSZ1_RSZ1_RSZ2

### Backsubstitution after applying RS history:
0: -0.0039095, -0.0003029, -0.0039095, -0.0003029, -0.0032283, 0.0032110
1: -0.0044907, 0.0040812, -0.0044907, 0.0040812, -0.0039710, 0.0040109
2: 0.0029569, 0.0099431, 0.0029569, 0.0099431, -0.0037316, 0.0037464
3: -0.0045519, -0.0034923, -0.0045519, -0.0034923, -0.0010595, 0.0010595
4: 0.0017018, 0.0078700, 0.0017018, 0.0078700, -0.0052485, 0.0052366
5: -0.0030341, 0.0031953, -0.0030341, 0.0031953, -0.0046875, 0.0047237
6: -0.0068516, -0.0033211, -0.0068516, -0.0033211, -0.0033360, 0.0033400
7: -0.0021570, 0.0038259, -0.0021570, 0.0038259, -0.0056810, 0.0057154
8: -0.0008533, 0.0000782, -0.0008533, 0.0000782, -0.0008461, 0.0008680
9: 0.9973557, 1.0121032, 0.9973557, 1.0121032, -0.0089351, 0.0088756

### Unstable ReLU Count (Linear/Conv2D Layers)
- layer_idx=0, type=LayerType.Linear, total=256, inp1_unstable=7, inp2_unstable=7, delta_unstable=249
- layer_idx=2, type=LayerType.Linear, total=256, inp1_unstable=10, inp2_unstable=10, delta_unstable=256
- layer_idx=4, type=LayerType.Linear, total=256, inp1_unstable=1, inp2_unstable=1, delta_unstable=256
- layer_idx=6, type=LayerType.Linear, total=256, inp1_unstable=0, inp2_unstable=0, delta_unstable=256
- layer_idx=8, type=LayerType.Linear, total=10, inp1_unstable=4, inp2_unstable=4, delta_unstable=10

Time for backsubstitution: 1.40 seconds

### RS candidates at layer 3
type: RSZ, layer: 3, pos: 8
type: RSZ, layer: 3, pos: 13
type: RSZ, layer: 3, pos: 20
type: RSZ, layer: 3, pos: 35
type: RSZ, layer: 3, pos: 43
type: RSZ, layer: 3, pos: 48
type: RSZ, layer: 3, pos: 103
type: RSZ, layer: 3, pos: 133
type: RSZ, layer: 3, pos: 180
type: RSZ, layer: 3, pos: 183

Time for candidate selection: 0.30 seconds

### Candidate
type: RSZ, layer: 3, pos: 8

### Relational analysis RSZ of RS_RSZ2_RSZ1_RSZ1_RSZ1_RSZ1_RSZ1_RSZ2_RSZ1

#### Relational analysis RSZ result of RS_RSZ2_RSZ1_RSZ1_RSZ1_RSZ1_RSZ1_RSZ2_RSZ1
Status: Status.VERIFIED
Output dim: 9, lower bound: -0.0044960, upper bound: 0.0043524
time: 0.78 seconds

### Relational analysis RSZ of RS_RSZ2_RSZ1_RSZ1_RSZ1_RSZ1_RSZ1_RSZ2_RSZ2

#### Relational analysis RSZ result of RS_RSZ2_RSZ1_RSZ1_RSZ1_RSZ1_RSZ1_RSZ2_RSZ2
Status: Status.VERIFIED
Output dim: 9, lower bound: -0.0044960, upper bound: 0.0043521
time: 0.79 seconds

## BFS RS instance: RS_RSZ2_RSZ1_RSZ1_RSZ1_RSZ1_RSZ2_RSZ1

### Backsubstitution after applying RS history:
0: -0.0039095, -0.0003029, -0.0039095, -0.0003029, -0.0032454, 0.0032013
1: -0.0044907, 0.0040812, -0.0044907, 0.0040812, -0.0039414, 0.0040649
2: 0.0029569, 0.0099431, 0.0029569, 0.0099431, -0.0037072, 0.0037847
3: -0.0045519, -0.0034923, -0.0045519, -0.0034923, -0.0010595, 0.0010595
4: 0.0017018, 0.0078700, 0.0017018, 0.0078700, -0.0052620, 0.0052289
5: -0.0030341, 0.0031953, -0.0030341, 0.0031953, -0.0046808, 0.0047385
6: -0.0068516, -0.0033211, -0.0068516, -0.0033211, -0.0033335, 0.0033445
7: -0.0021570, 0.0038259, -0.0021570, 0.0038259, -0.0056970, 0.0057040
8: -0.0008533, 0.0000782, -0.0008533, 0.0000782, -0.0008472, 0.0008699
9: 0.9973557, 1.0121032, 0.9973557, 1.0121032, -0.0090013, 0.0088380

### Unstable ReLU Count (Linear/Conv2D Layers)
- layer_idx=0, type=LayerType.Linear, total=256, inp1_unstable=7, inp2_unstable=7, delta_unstable=249
- layer_idx=2, type=LayerType.Linear, total=256, inp1_unstable=10, inp2_unstable=10, delta_unstable=256
- layer_idx=4, type=LayerType.Linear, total=256, inp1_unstable=1, inp2_unstable=1, delta_unstable=256
- layer_idx=6, type=LayerType.Linear, total=256, inp1_unstable=0, inp2_unstable=0, delta_unstable=256
- layer_idx=8, type=LayerType.Linear, total=10, inp1_unstable=4, inp2_unstable=4, delta_unstable=10

Time for backsubstitution: 1.41 seconds

## RS Result
status: Status.UNKNOWN
execution time: (base) + (rs) = 3.02 + 597.48 = 600.50 seconds
