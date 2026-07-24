## Execution arguments:
Dataset: Dataset.MNIST
Network: onnx/mnist-net_256x4.onnx
Relational property: GLOBAL_ROBUSTNESS
LP Analysis: True
Epsilon: 0.0078125
Delta epsilon: 0.00390625
execution index: (1, 2, 0)
Time budget: 600 seconds
Split limit: 100
Threshold: 0.000280174


## IAR start

### Unstable ReLU Count (Linear/Conv2D Layers)
- layer_idx=0, type=LayerType.Linear, total=256, inp1_unstable=5, inp2_unstable=5, delta_unstable=256
- layer_idx=2, type=LayerType.Linear, total=256, inp1_unstable=4, inp2_unstable=4, delta_unstable=256
- layer_idx=4, type=LayerType.Linear, total=256, inp1_unstable=0, inp2_unstable=0, delta_unstable=256
- layer_idx=6, type=LayerType.Linear, total=256, inp1_unstable=1, inp2_unstable=1, delta_unstable=256
- layer_idx=8, type=LayerType.Linear, total=10, inp1_unstable=0, inp2_unstable=0, delta_unstable=10

### BASE IAR bounds
Layer (inp1_lb, inp1_ub, inp2_lb, inp2_ub, d_lb, d_ub)
0: (0.0009457, 0.0011003, 0.0009457, 0.0011003, -0.0001509, 0.0001509)
1: (0.9936197, 0.9940771, 0.9936197, 0.9940771, -0.0004126, 0.0004126)
2: (-0.0065606, -0.0050208, -0.0065606, -0.0050208, -0.0013707, 0.0013707)
3: (0.0038004, 0.0040140, 0.0038004, 0.0040140, -0.0001829, 0.0001829)
4: (0.0023852, 0.0036022, 0.0023852, 0.0036022, -0.0011669, 0.0011669)
5: (0.0060086, 0.0065111, 0.0060086, 0.0065111, -0.0005024, 0.0005024)
6: (-0.0013983, -0.0008639, -0.0013983, -0.0008639, -0.0004708, 0.0004708)
7: (-0.0083221, -0.0079387, -0.0083221, -0.0079387, -0.0003835, 0.0003835)
8: (0.0048484, 0.0068715, 0.0048484, 0.0068715, -0.0015712, 0.0015712)
9: (-0.0036877, -0.0032079, -0.0036877, -0.0032079, -0.0004797, 0.0004797)

## BASE Relational Analysis

## BASE Result
execution time: IAR + RelationalAnalysis = 1.26 + 1.40 = 2.66 seconds
status: Status.UNKNOWN
relational distance
Output dim: 1, lower bound: -0.0003914, upper bound: 0.0003914

# Relational Split (RS) starts

## BFS RS instance: RS

Time for backsubstitution: 0.01 seconds

### RS candidates at layer 1
type: RSZ, layer: 1, pos: 64
type: RSZ, layer: 1, pos: 127
type: RSZ, layer: 1, pos: 167
type: RSZ, layer: 1, pos: 178
type: RSZ, layer: 1, pos: 251

Time for candidate selection: 0.11 seconds

### Candidate
type: RSZ, layer: 1, pos: 64

### Relational analysis RSZ of RS_RSZ1

#### Relational analysis RSZ result of RS_RSZ1
Status: Status.UNKNOWN
Output dim: 1, lower bound: -0.0003550, upper bound: 0.0003550
time: 0.51 seconds

### Relational analysis RSZ of RS_RSZ2

#### Relational analysis RSZ result of RS_RSZ2
Status: Status.UNKNOWN
Output dim: 1, lower bound: -0.0003550, upper bound: 0.0003550
time: 0.51 seconds

## Summary of splitting (split count: 0)
- Time for RS candidates: 1.14 seconds
RS_RSZ1, status: Status.UNKNOWN, split count: 1, time: 1.14
Output dim: 1, lower bound: -0.0003550, upper bound: 0.0003550
RS_RSZ2, status: Status.UNKNOWN, split count: 1, time: 1.14
Output dim: 1, lower bound: -0.0003550, upper bound: 0.0003550

## BFS RS instance: RS_RSZ1

### Backsubstitution after applying RS history:
0: 0.0009457, 0.0011003, 0.0009457, 0.0011003, -0.0001508, 0.0001509
1: 0.9936197, 0.9940771, 0.9936197, 0.9940771, -0.0004115, 0.0004109
2: -0.0065606, -0.0050208, -0.0065606, -0.0050208, -0.0013593, 0.0013524
3: 0.0038004, 0.0040140, 0.0038004, 0.0040140, -0.0001817, 0.0001822
4: 0.0023852, 0.0036022, 0.0023852, 0.0036022, -0.0011524, 0.0011578
5: 0.0060086, 0.0065111, 0.0060086, 0.0065111, -0.0005024, 0.0005024
6: -0.0013983, -0.0008639, -0.0013983, -0.0008639, -0.0004668, 0.0004645
7: -0.0083221, -0.0079387, -0.0083221, -0.0079387, -0.0003835, 0.0003835
8: 0.0048484, 0.0068715, 0.0048484, 0.0068715, -0.0015480, 0.0015570
9: -0.0036877, -0.0032079, -0.0036877, -0.0032079, -0.0004797, 0.0004797

### Unstable ReLU Count (Linear/Conv2D Layers)
- layer_idx=0, type=LayerType.Linear, total=256, inp1_unstable=5, inp2_unstable=5, delta_unstable=255
- layer_idx=2, type=LayerType.Linear, total=256, inp1_unstable=4, inp2_unstable=4, delta_unstable=256
- layer_idx=4, type=LayerType.Linear, total=256, inp1_unstable=0, inp2_unstable=0, delta_unstable=256
- layer_idx=6, type=LayerType.Linear, total=256, inp1_unstable=1, inp2_unstable=1, delta_unstable=256
- layer_idx=8, type=LayerType.Linear, total=10, inp1_unstable=0, inp2_unstable=0, delta_unstable=10

Time for backsubstitution: 1.08 seconds

### RS candidates at layer 1
type: RSZ, layer: 1, pos: 127
type: RSZ, layer: 1, pos: 167
type: RSZ, layer: 1, pos: 178
type: RSZ, layer: 1, pos: 251

Time for candidate selection: 0.11 seconds

### Candidate
type: RSZ, layer: 1, pos: 127

### Relational analysis RSZ of RS_RSZ1_RSZ1

#### Relational analysis RSZ result of RS_RSZ1_RSZ1
Status: Status.UNKNOWN
Output dim: 1, lower bound: -0.0003508, upper bound: 0.0003508
time: 0.51 seconds

### Relational analysis RSZ of RS_RSZ1_RSZ2

#### Relational analysis RSZ result of RS_RSZ1_RSZ2
Status: Status.UNKNOWN
Output dim: 1, lower bound: -0.0003508, upper bound: 0.0003508
time: 0.53 seconds

## BFS RS instance: RS_RSZ2

### Backsubstitution after applying RS history:
0: 0.0009457, 0.0011003, 0.0009457, 0.0011003, -0.0001509, 0.0001509
1: 0.9936197, 0.9940771, 0.9936197, 0.9940771, -0.0004126, 0.0004115
2: -0.0065606, -0.0050208, -0.0065606, -0.0050208, -0.0013707, 0.0013593
3: 0.0038004, 0.0040140, 0.0038004, 0.0040140, -0.0001822, 0.0001829
4: 0.0023852, 0.0036022, 0.0023852, 0.0036022, -0.0011578, 0.0011669
5: 0.0060086, 0.0065111, 0.0060086, 0.0065111, -0.0005024, 0.0005024
6: -0.0013983, -0.0008639, -0.0013983, -0.0008639, -0.0004708, 0.0004668
7: -0.0083221, -0.0079387, -0.0083221, -0.0079387, -0.0003835, 0.0003835
8: 0.0048484, 0.0068715, 0.0048484, 0.0068715, -0.0015570, 0.0015712
9: -0.0036877, -0.0032079, -0.0036877, -0.0032079, -0.0004797, 0.0004797

### Unstable ReLU Count (Linear/Conv2D Layers)
- layer_idx=0, type=LayerType.Linear, total=256, inp1_unstable=5, inp2_unstable=5, delta_unstable=255
- layer_idx=2, type=LayerType.Linear, total=256, inp1_unstable=4, inp2_unstable=4, delta_unstable=256
- layer_idx=4, type=LayerType.Linear, total=256, inp1_unstable=0, inp2_unstable=0, delta_unstable=256
- layer_idx=6, type=LayerType.Linear, total=256, inp1_unstable=1, inp2_unstable=1, delta_unstable=256
- layer_idx=8, type=LayerType.Linear, total=10, inp1_unstable=0, inp2_unstable=0, delta_unstable=10

Time for backsubstitution: 1.10 seconds

### RS candidates at layer 1
type: RSZ, layer: 1, pos: 127
type: RSZ, layer: 1, pos: 167
type: RSZ, layer: 1, pos: 178
type: RSZ, layer: 1, pos: 251

Time for candidate selection: 0.11 seconds

### Candidate
type: RSZ, layer: 1, pos: 127

### Relational analysis RSZ of RS_RSZ2_RSZ1

#### Relational analysis RSZ result of RS_RSZ2_RSZ1
Status: Status.UNKNOWN
Output dim: 1, lower bound: -0.0003508, upper bound: 0.0003508
time: 0.51 seconds

### Relational analysis RSZ of RS_RSZ2_RSZ2

#### Relational analysis RSZ result of RS_RSZ2_RSZ2
Status: Status.UNKNOWN
Output dim: 1, lower bound: -0.0003508, upper bound: 0.0003508
time: 0.53 seconds

## Summary of splitting (split count: 1)
- Time for RS candidates: 2.27 seconds
RS_RSZ1_RSZ1, status: Status.UNKNOWN, split count: 2, time: 2.27
Output dim: 1, lower bound: -0.0003508, upper bound: 0.0003508
RS_RSZ1_RSZ2, status: Status.UNKNOWN, split count: 2, time: 2.27
Output dim: 1, lower bound: -0.0003508, upper bound: 0.0003508
RS_RSZ2_RSZ1, status: Status.UNKNOWN, split count: 2, time: 2.27
Output dim: 1, lower bound: -0.0003508, upper bound: 0.0003508
RS_RSZ2_RSZ2, status: Status.UNKNOWN, split count: 2, time: 2.27
Output dim: 1, lower bound: -0.0003508, upper bound: 0.0003508

## BFS RS instance: RS_RSZ1_RSZ1

### Backsubstitution after applying RS history:
0: 0.0009457, 0.0011003, 0.0009457, 0.0011003, -0.0001507, 0.0001508
1: 0.9936197, 0.9940771, 0.9936197, 0.9940771, -0.0004106, 0.0004102
2: -0.0065606, -0.0050208, -0.0065606, -0.0050208, -0.0013496, 0.0013453
3: 0.0038004, 0.0040140, 0.0038004, 0.0040140, -0.0001812, 0.0001815
4: 0.0023852, 0.0036022, 0.0023852, 0.0036022, -0.0011468, 0.0011502
5: 0.0060086, 0.0065111, 0.0060086, 0.0065111, -0.0005024, 0.0005024
6: -0.0013983, -0.0008639, -0.0013983, -0.0008639, -0.0004635, 0.0004620
7: -0.0083221, -0.0079387, -0.0083221, -0.0079387, -0.0003835, 0.0003835
8: 0.0048484, 0.0068715, 0.0048484, 0.0068715, -0.0015379, 0.0015435
9: -0.0036877, -0.0032079, -0.0036877, -0.0032079, -0.0004797, 0.0004797

### Unstable ReLU Count (Linear/Conv2D Layers)
- layer_idx=0, type=LayerType.Linear, total=256, inp1_unstable=5, inp2_unstable=5, delta_unstable=254
- layer_idx=2, type=LayerType.Linear, total=256, inp1_unstable=4, inp2_unstable=4, delta_unstable=256
- layer_idx=4, type=LayerType.Linear, total=256, inp1_unstable=0, inp2_unstable=0, delta_unstable=256
- layer_idx=6, type=LayerType.Linear, total=256, inp1_unstable=1, inp2_unstable=1, delta_unstable=256
- layer_idx=8, type=LayerType.Linear, total=10, inp1_unstable=0, inp2_unstable=0, delta_unstable=10

Time for backsubstitution: 1.09 seconds

### RS candidates at layer 1
type: RSZ, layer: 1, pos: 167
type: RSZ, layer: 1, pos: 178
type: RSZ, layer: 1, pos: 251

Time for candidate selection: 0.11 seconds

### Candidate
type: RSZ, layer: 1, pos: 167

### Relational analysis RSZ of RS_RSZ1_RSZ1_RSZ1

#### Relational analysis RSZ result of RS_RSZ1_RSZ1_RSZ1
Status: Status.UNKNOWN
Output dim: 1, lower bound: -0.0003467, upper bound: 0.0003473
time: 0.53 seconds

### Relational analysis RSZ of RS_RSZ1_RSZ1_RSZ2

#### Relational analysis RSZ result of RS_RSZ1_RSZ1_RSZ2
Status: Status.UNKNOWN
Output dim: 1, lower bound: -0.0003473, upper bound: 0.0003467
time: 0.58 seconds

## BFS RS instance: RS_RSZ1_RSZ2

### Backsubstitution after applying RS history:
0: 0.0009457, 0.0011003, 0.0009457, 0.0011003, -0.0001507, 0.0001509
1: 0.9936197, 0.9940771, 0.9936197, 0.9940771, -0.0004115, 0.0004099
2: -0.0065606, -0.0050208, -0.0065606, -0.0050208, -0.0013593, 0.0013427
3: 0.0038004, 0.0040140, 0.0038004, 0.0040140, -0.0001810, 0.0001822
4: 0.0023852, 0.0036022, 0.0023852, 0.0036022, -0.0011448, 0.0011578
5: 0.0060086, 0.0065111, 0.0060086, 0.0065111, -0.0005024, 0.0005024
6: -0.0013983, -0.0008639, -0.0013983, -0.0008639, -0.0004668, 0.0004611
7: -0.0083221, -0.0079387, -0.0083221, -0.0079387, -0.0003835, 0.0003835
8: 0.0048484, 0.0068715, 0.0048484, 0.0068715, -0.0015345, 0.0015570
9: -0.0036877, -0.0032079, -0.0036877, -0.0032079, -0.0004797, 0.0004797

### Unstable ReLU Count (Linear/Conv2D Layers)
- layer_idx=0, type=LayerType.Linear, total=256, inp1_unstable=5, inp2_unstable=5, delta_unstable=254
- layer_idx=2, type=LayerType.Linear, total=256, inp1_unstable=4, inp2_unstable=4, delta_unstable=256
- layer_idx=4, type=LayerType.Linear, total=256, inp1_unstable=0, inp2_unstable=0, delta_unstable=256
- layer_idx=6, type=LayerType.Linear, total=256, inp1_unstable=1, inp2_unstable=1, delta_unstable=256
- layer_idx=8, type=LayerType.Linear, total=10, inp1_unstable=0, inp2_unstable=0, delta_unstable=10

Time for backsubstitution: 1.09 seconds

### RS candidates at layer 1
type: RSZ, layer: 1, pos: 167
type: RSZ, layer: 1, pos: 178
type: RSZ, layer: 1, pos: 251

Time for candidate selection: 0.11 seconds

### Candidate
type: RSZ, layer: 1, pos: 167

### Relational analysis RSZ of RS_RSZ1_RSZ2_RSZ1

#### Relational analysis RSZ result of RS_RSZ1_RSZ2_RSZ1
Status: Status.UNKNOWN
Output dim: 1, lower bound: -0.0003467, upper bound: 0.0003473
time: 0.59 seconds

### Relational analysis RSZ of RS_RSZ1_RSZ2_RSZ2

#### Relational analysis RSZ result of RS_RSZ1_RSZ2_RSZ2
Status: Status.UNKNOWN
Output dim: 1, lower bound: -0.0003473, upper bound: 0.0003467
time: 0.55 seconds

## BFS RS instance: RS_RSZ2_RSZ1

### Backsubstitution after applying RS history:
0: 0.0009457, 0.0011003, 0.0009457, 0.0011003, -0.0001508, 0.0001509
1: 0.9936197, 0.9940771, 0.9936197, 0.9940771, -0.0004116, 0.0004109
2: -0.0065606, -0.0050208, -0.0065606, -0.0050208, -0.0013610, 0.0013527
3: 0.0038004, 0.0040140, 0.0038004, 0.0040140, -0.0001817, 0.0001822
4: 0.0023852, 0.0036022, 0.0023852, 0.0036022, -0.0011527, 0.0011593
5: 0.0060086, 0.0065111, 0.0060086, 0.0065111, -0.0005024, 0.0005024
6: -0.0013983, -0.0008639, -0.0013983, -0.0008639, -0.0004674, 0.0004646
7: -0.0083221, -0.0079387, -0.0083221, -0.0079387, -0.0003835, 0.0003835
8: 0.0048484, 0.0068715, 0.0048484, 0.0068715, -0.0015476, 0.0015574
9: -0.0036877, -0.0032079, -0.0036877, -0.0032079, -0.0004797, 0.0004797

### Unstable ReLU Count (Linear/Conv2D Layers)
- layer_idx=0, type=LayerType.Linear, total=256, inp1_unstable=5, inp2_unstable=5, delta_unstable=254
- layer_idx=2, type=LayerType.Linear, total=256, inp1_unstable=4, inp2_unstable=4, delta_unstable=256
- layer_idx=4, type=LayerType.Linear, total=256, inp1_unstable=0, inp2_unstable=0, delta_unstable=256
- layer_idx=6, type=LayerType.Linear, total=256, inp1_unstable=1, inp2_unstable=1, delta_unstable=256
- layer_idx=8, type=LayerType.Linear, total=10, inp1_unstable=0, inp2_unstable=0, delta_unstable=10

Time for backsubstitution: 1.09 seconds

### RS candidates at layer 1
type: RSZ, layer: 1, pos: 167
type: RSZ, layer: 1, pos: 178
type: RSZ, layer: 1, pos: 251

Time for candidate selection: 0.11 seconds

### Candidate
type: RSZ, layer: 1, pos: 167

### Relational analysis RSZ of RS_RSZ2_RSZ1_RSZ1

#### Relational analysis RSZ result of RS_RSZ2_RSZ1_RSZ1
Status: Status.UNKNOWN
Output dim: 1, lower bound: -0.0003467, upper bound: 0.0003473
time: 0.52 seconds

### Relational analysis RSZ of RS_RSZ2_RSZ1_RSZ2

#### Relational analysis RSZ result of RS_RSZ2_RSZ1_RSZ2
Status: Status.UNKNOWN
Output dim: 1, lower bound: -0.0003473, upper bound: 0.0003467
time: 0.58 seconds

## BFS RS instance: RS_RSZ2_RSZ2

### Backsubstitution after applying RS history:
0: 0.0009457, 0.0011003, 0.0009457, 0.0011003, -0.0001508, 0.0001509
1: 0.9936197, 0.9940771, 0.9936197, 0.9940771, -0.0004126, 0.0004106
2: -0.0065606, -0.0050208, -0.0065606, -0.0050208, -0.0013707, 0.0013496
3: 0.0038004, 0.0040140, 0.0038004, 0.0040140, -0.0001815, 0.0001829
4: 0.0023852, 0.0036022, 0.0023852, 0.0036022, -0.0011502, 0.0011669
5: 0.0060086, 0.0065111, 0.0060086, 0.0065111, -0.0005024, 0.0005024
6: -0.0013983, -0.0008639, -0.0013983, -0.0008639, -0.0004708, 0.0004635
7: -0.0083221, -0.0079387, -0.0083221, -0.0079387, -0.0003835, 0.0003835
8: 0.0048484, 0.0068715, 0.0048484, 0.0068715, -0.0015435, 0.0015712
9: -0.0036877, -0.0032079, -0.0036877, -0.0032079, -0.0004797, 0.0004797

### Unstable ReLU Count (Linear/Conv2D Layers)
- layer_idx=0, type=LayerType.Linear, total=256, inp1_unstable=5, inp2_unstable=5, delta_unstable=254
- layer_idx=2, type=LayerType.Linear, total=256, inp1_unstable=4, inp2_unstable=4, delta_unstable=256
- layer_idx=4, type=LayerType.Linear, total=256, inp1_unstable=0, inp2_unstable=0, delta_unstable=256
- layer_idx=6, type=LayerType.Linear, total=256, inp1_unstable=1, inp2_unstable=1, delta_unstable=256
- layer_idx=8, type=LayerType.Linear, total=10, inp1_unstable=0, inp2_unstable=0, delta_unstable=10

Time for backsubstitution: 1.08 seconds

### RS candidates at layer 1
type: RSZ, layer: 1, pos: 167
type: RSZ, layer: 1, pos: 178
type: RSZ, layer: 1, pos: 251

Time for candidate selection: 0.10 seconds

### Candidate
type: RSZ, layer: 1, pos: 167

### Relational analysis RSZ of RS_RSZ2_RSZ2_RSZ1

#### Relational analysis RSZ result of RS_RSZ2_RSZ2_RSZ1
Status: Status.UNKNOWN
Output dim: 1, lower bound: -0.0003467, upper bound: 0.0003473
time: 0.59 seconds

### Relational analysis RSZ of RS_RSZ2_RSZ2_RSZ2

#### Relational analysis RSZ result of RS_RSZ2_RSZ2_RSZ2
Status: Status.UNKNOWN
Output dim: 1, lower bound: -0.0003473, upper bound: 0.0003467
time: 0.53 seconds

## Summary of splitting (split count: 2)
- Time for RS candidates: 2.30 seconds
RS_RSZ1_RSZ1_RSZ1, status: Status.UNKNOWN, split count: 3, time: 2.30
Output dim: 1, lower bound: -0.0003467, upper bound: 0.0003473
RS_RSZ1_RSZ1_RSZ2, status: Status.UNKNOWN, split count: 3, time: 2.30
Output dim: 1, lower bound: -0.0003473, upper bound: 0.0003467
RS_RSZ1_RSZ2_RSZ1, status: Status.UNKNOWN, split count: 3, time: 2.30
Output dim: 1, lower bound: -0.0003467, upper bound: 0.0003473
RS_RSZ1_RSZ2_RSZ2, status: Status.UNKNOWN, split count: 3, time: 2.30
Output dim: 1, lower bound: -0.0003473, upper bound: 0.0003467
RS_RSZ2_RSZ1_RSZ1, status: Status.UNKNOWN, split count: 3, time: 2.30
Output dim: 1, lower bound: -0.0003467, upper bound: 0.0003473
RS_RSZ2_RSZ1_RSZ2, status: Status.UNKNOWN, split count: 3, time: 2.30
Output dim: 1, lower bound: -0.0003473, upper bound: 0.0003467
RS_RSZ2_RSZ2_RSZ1, status: Status.UNKNOWN, split count: 3, time: 2.30
Output dim: 1, lower bound: -0.0003467, upper bound: 0.0003473
RS_RSZ2_RSZ2_RSZ2, status: Status.UNKNOWN, split count: 3, time: 2.30
Output dim: 1, lower bound: -0.0003473, upper bound: 0.0003467

## BFS RS instance: RS_RSZ1_RSZ1_RSZ1

### Backsubstitution after applying RS history:
0: 0.0009457, 0.0011003, 0.0009457, 0.0011003, -0.0001507, 0.0001507
1: 0.9936197, 0.9940771, 0.9936197, 0.9940771, -0.0004103, 0.0004099
2: -0.0065606, -0.0050208, -0.0065606, -0.0050208, -0.0013394, 0.0013355
3: 0.0038004, 0.0040140, 0.0038004, 0.0040140, -0.0001810, 0.0001813
4: 0.0023852, 0.0036022, 0.0023852, 0.0036022, -0.0011396, 0.0011427
5: 0.0060086, 0.0065111, 0.0060086, 0.0065111, -0.0005024, 0.0005024
6: -0.0013983, -0.0008639, -0.0013983, -0.0008639, -0.0004601, 0.0004588
7: -0.0083221, -0.0079387, -0.0083221, -0.0079387, -0.0003835, 0.0003835
8: 0.0048484, 0.0068715, 0.0048484, 0.0068715, -0.0015294, 0.0015346
9: -0.0036877, -0.0032079, -0.0036877, -0.0032079, -0.0004797, 0.0004797

### Unstable ReLU Count (Linear/Conv2D Layers)
- layer_idx=0, type=LayerType.Linear, total=256, inp1_unstable=5, inp2_unstable=5, delta_unstable=253
- layer_idx=2, type=LayerType.Linear, total=256, inp1_unstable=4, inp2_unstable=4, delta_unstable=256
- layer_idx=4, type=LayerType.Linear, total=256, inp1_unstable=0, inp2_unstable=0, delta_unstable=256
- layer_idx=6, type=LayerType.Linear, total=256, inp1_unstable=1, inp2_unstable=1, delta_unstable=256
- layer_idx=8, type=LayerType.Linear, total=10, inp1_unstable=0, inp2_unstable=0, delta_unstable=10

Time for backsubstitution: 1.08 seconds

### RS candidates at layer 1
type: RSZ, layer: 1, pos: 178
type: RSZ, layer: 1, pos: 251

Time for candidate selection: 0.11 seconds

### Candidate
type: RSZ, layer: 1, pos: 178

### Relational analysis RSZ of RS_RSZ1_RSZ1_RSZ1_RSZ1

#### Relational analysis RSZ result of RS_RSZ1_RSZ1_RSZ1_RSZ1
Status: Status.UNKNOWN
Output dim: 1, lower bound: -0.0003443, upper bound: 0.0003443
time: 0.63 seconds

### Relational analysis RSZ of RS_RSZ1_RSZ1_RSZ1_RSZ2

#### Relational analysis RSZ result of RS_RSZ1_RSZ1_RSZ1_RSZ2
Status: Status.UNKNOWN
Output dim: 1, lower bound: -0.0003435, upper bound: 0.0003450
time: 0.56 seconds

## BFS RS instance: RS_RSZ1_RSZ1_RSZ2

### Backsubstitution after applying RS history:
0: 0.0009457, 0.0011003, 0.0009457, 0.0011003, -0.0001507, 0.0001507
1: 0.9936197, 0.9940771, 0.9936197, 0.9940771, -0.0004104, 0.0004099
2: -0.0065606, -0.0050208, -0.0065606, -0.0050208, -0.0013398, 0.0013348
3: 0.0038004, 0.0040140, 0.0038004, 0.0040140, -0.0001810, 0.0001813
4: 0.0023852, 0.0036022, 0.0023852, 0.0036022, -0.0011390, 0.0011430
5: 0.0060086, 0.0065111, 0.0060086, 0.0065111, -0.0005024, 0.0005024
6: -0.0013983, -0.0008639, -0.0013983, -0.0008639, -0.0004603, 0.0004585
7: -0.0083221, -0.0079387, -0.0083221, -0.0079387, -0.0003835, 0.0003835
8: 0.0048484, 0.0068715, 0.0048484, 0.0068715, -0.0015284, 0.0015350
9: -0.0036877, -0.0032079, -0.0036877, -0.0032079, -0.0004797, 0.0004797

### Unstable ReLU Count (Linear/Conv2D Layers)
- layer_idx=0, type=LayerType.Linear, total=256, inp1_unstable=5, inp2_unstable=5, delta_unstable=253
- layer_idx=2, type=LayerType.Linear, total=256, inp1_unstable=4, inp2_unstable=4, delta_unstable=256
- layer_idx=4, type=LayerType.Linear, total=256, inp1_unstable=0, inp2_unstable=0, delta_unstable=256
- layer_idx=6, type=LayerType.Linear, total=256, inp1_unstable=1, inp2_unstable=1, delta_unstable=256
- layer_idx=8, type=LayerType.Linear, total=10, inp1_unstable=0, inp2_unstable=0, delta_unstable=10

Time for backsubstitution: 1.16 seconds

### RS candidates at layer 1
type: RSZ, layer: 1, pos: 178
type: RSZ, layer: 1, pos: 251

Time for candidate selection: 0.11 seconds

### Candidate
type: RSZ, layer: 1, pos: 178

### Relational analysis RSZ of RS_RSZ1_RSZ1_RSZ2_RSZ1

#### Relational analysis RSZ result of RS_RSZ1_RSZ1_RSZ2_RSZ1
Status: Status.UNKNOWN
Output dim: 1, lower bound: -0.0003449, upper bound: 0.0003436
time: 0.55 seconds

### Relational analysis RSZ of RS_RSZ1_RSZ1_RSZ2_RSZ2

#### Relational analysis RSZ result of RS_RSZ1_RSZ1_RSZ2_RSZ2
Status: Status.UNKNOWN
Output dim: 1, lower bound: -0.0003441, upper bound: 0.0003443
time: 0.55 seconds

## BFS RS instance: RS_RSZ1_RSZ2_RSZ1

### Backsubstitution after applying RS history:
0: 0.0009457, 0.0011003, 0.0009457, 0.0011003, -0.0001507, 0.0001508
1: 0.9936197, 0.9940771, 0.9936197, 0.9940771, -0.0004112, 0.0004097
2: -0.0065606, -0.0050208, -0.0065606, -0.0050208, -0.0013496, 0.0013329
3: 0.0038004, 0.0040140, 0.0038004, 0.0040140, -0.0001809, 0.0001819
4: 0.0023852, 0.0036022, 0.0023852, 0.0036022, -0.0011375, 0.0011504
5: 0.0060086, 0.0065111, 0.0060086, 0.0065111, -0.0005024, 0.0005024
6: -0.0013983, -0.0008639, -0.0013983, -0.0008639, -0.0004636, 0.0004579
7: -0.0083221, -0.0079387, -0.0083221, -0.0079387, -0.0003835, 0.0003835
8: 0.0048484, 0.0068715, 0.0048484, 0.0068715, -0.0015260, 0.0015478
9: -0.0036877, -0.0032079, -0.0036877, -0.0032079, -0.0004797, 0.0004797

### Unstable ReLU Count (Linear/Conv2D Layers)
- layer_idx=0, type=LayerType.Linear, total=256, inp1_unstable=5, inp2_unstable=5, delta_unstable=253
- layer_idx=2, type=LayerType.Linear, total=256, inp1_unstable=4, inp2_unstable=4, delta_unstable=256
- layer_idx=4, type=LayerType.Linear, total=256, inp1_unstable=0, inp2_unstable=0, delta_unstable=256
- layer_idx=6, type=LayerType.Linear, total=256, inp1_unstable=1, inp2_unstable=1, delta_unstable=256
- layer_idx=8, type=LayerType.Linear, total=10, inp1_unstable=0, inp2_unstable=0, delta_unstable=10

Time for backsubstitution: 1.09 seconds

### RS candidates at layer 1
type: RSZ, layer: 1, pos: 178
type: RSZ, layer: 1, pos: 251

Time for candidate selection: 0.11 seconds

### Candidate
type: RSZ, layer: 1, pos: 178

### Relational analysis RSZ of RS_RSZ1_RSZ2_RSZ1_RSZ1

#### Relational analysis RSZ result of RS_RSZ1_RSZ2_RSZ1_RSZ1
Status: Status.UNKNOWN
Output dim: 1, lower bound: -0.0003443, upper bound: 0.0003441
time: 0.54 seconds

### Relational analysis RSZ of RS_RSZ1_RSZ2_RSZ1_RSZ2

#### Relational analysis RSZ result of RS_RSZ1_RSZ2_RSZ1_RSZ2
Status: Status.UNKNOWN
Output dim: 1, lower bound: -0.0003436, upper bound: 0.0003449
time: 0.59 seconds

## BFS RS instance: RS_RSZ1_RSZ2_RSZ2

### Backsubstitution after applying RS history:
0: 0.0009457, 0.0011003, 0.0009457, 0.0011003, -0.0001507, 0.0001508
1: 0.9936197, 0.9940771, 0.9936197, 0.9940771, -0.0004113, 0.0004096
2: -0.0065606, -0.0050208, -0.0065606, -0.0050208, -0.0013500, 0.0013322
3: 0.0038004, 0.0040140, 0.0038004, 0.0040140, -0.0001808, 0.0001820
4: 0.0023852, 0.0036022, 0.0023852, 0.0036022, -0.0011370, 0.0011507
5: 0.0060086, 0.0065111, 0.0060086, 0.0065111, -0.0005024, 0.0005024
6: -0.0013983, -0.0008639, -0.0013983, -0.0008639, -0.0004638, 0.0004576
7: -0.0083221, -0.0079387, -0.0083221, -0.0079387, -0.0003835, 0.0003835
8: 0.0048484, 0.0068715, 0.0048484, 0.0068715, -0.0015251, 0.0015483
9: -0.0036877, -0.0032079, -0.0036877, -0.0032079, -0.0004797, 0.0004797

### Unstable ReLU Count (Linear/Conv2D Layers)
- layer_idx=0, type=LayerType.Linear, total=256, inp1_unstable=5, inp2_unstable=5, delta_unstable=253
- layer_idx=2, type=LayerType.Linear, total=256, inp1_unstable=4, inp2_unstable=4, delta_unstable=256
- layer_idx=4, type=LayerType.Linear, total=256, inp1_unstable=0, inp2_unstable=0, delta_unstable=256
- layer_idx=6, type=LayerType.Linear, total=256, inp1_unstable=1, inp2_unstable=1, delta_unstable=256
- layer_idx=8, type=LayerType.Linear, total=10, inp1_unstable=0, inp2_unstable=0, delta_unstable=10

Time for backsubstitution: 1.15 seconds

### RS candidates at layer 1
type: RSZ, layer: 1, pos: 178
type: RSZ, layer: 1, pos: 251

Time for candidate selection: 0.11 seconds

### Candidate
type: RSZ, layer: 1, pos: 178

### Relational analysis RSZ of RS_RSZ1_RSZ2_RSZ2_RSZ1

#### Relational analysis RSZ result of RS_RSZ1_RSZ2_RSZ2_RSZ1
Status: Status.UNKNOWN
Output dim: 1, lower bound: -0.0003450, upper bound: 0.0003435
time: 0.54 seconds

### Relational analysis RSZ of RS_RSZ1_RSZ2_RSZ2_RSZ2

#### Relational analysis RSZ result of RS_RSZ1_RSZ2_RSZ2_RSZ2
Status: Status.UNKNOWN
Output dim: 1, lower bound: -0.0003443, upper bound: 0.0003443
time: 0.64 seconds

## BFS RS instance: RS_RSZ2_RSZ1_RSZ1

### Backsubstitution after applying RS history:
0: 0.0009457, 0.0011003, 0.0009457, 0.0011003, -0.0001507, 0.0001508
1: 0.9936197, 0.9940771, 0.9936197, 0.9940771, -0.0004113, 0.0004107
2: -0.0065606, -0.0050208, -0.0065606, -0.0050208, -0.0013506, 0.0013429
3: 0.0038004, 0.0040140, 0.0038004, 0.0040140, -0.0001815, 0.0001820
4: 0.0023852, 0.0036022, 0.0023852, 0.0036022, -0.0011454, 0.0011518
5: 0.0060086, 0.0065111, 0.0060086, 0.0065111, -0.0005024, 0.0005024
6: -0.0013983, -0.0008639, -0.0013983, -0.0008639, -0.0004640, 0.0004614
7: -0.0083221, -0.0079387, -0.0083221, -0.0079387, -0.0003835, 0.0003835
8: 0.0048484, 0.0068715, 0.0048484, 0.0068715, -0.0015391, 0.0015495
9: -0.0036877, -0.0032079, -0.0036877, -0.0032079, -0.0004797, 0.0004797

### Unstable ReLU Count (Linear/Conv2D Layers)
- layer_idx=0, type=LayerType.Linear, total=256, inp1_unstable=5, inp2_unstable=5, delta_unstable=253
- layer_idx=2, type=LayerType.Linear, total=256, inp1_unstable=4, inp2_unstable=4, delta_unstable=256
- layer_idx=4, type=LayerType.Linear, total=256, inp1_unstable=0, inp2_unstable=0, delta_unstable=256
- layer_idx=6, type=LayerType.Linear, total=256, inp1_unstable=1, inp2_unstable=1, delta_unstable=256
- layer_idx=8, type=LayerType.Linear, total=10, inp1_unstable=0, inp2_unstable=0, delta_unstable=10

Time for backsubstitution: 1.09 seconds

### RS candidates at layer 1
type: RSZ, layer: 1, pos: 178
type: RSZ, layer: 1, pos: 251

Time for candidate selection: 0.11 seconds

### Candidate
type: RSZ, layer: 1, pos: 178

### Relational analysis RSZ of RS_RSZ2_RSZ1_RSZ1_RSZ1

#### Relational analysis RSZ result of RS_RSZ2_RSZ1_RSZ1_RSZ1
Status: Status.UNKNOWN
Output dim: 1, lower bound: -0.0003443, upper bound: 0.0003443
time: 0.56 seconds

### Relational analysis RSZ of RS_RSZ2_RSZ1_RSZ1_RSZ2

#### Relational analysis RSZ result of RS_RSZ2_RSZ1_RSZ1_RSZ2
Status: Status.UNKNOWN
Output dim: 1, lower bound: -0.0003435, upper bound: 0.0003450
time: 0.54 seconds

## BFS RS instance: RS_RSZ2_RSZ1_RSZ2

### Backsubstitution after applying RS history:
0: 0.0009457, 0.0011003, 0.0009457, 0.0011003, -0.0001507, 0.0001508
1: 0.9936197, 0.9940771, 0.9936197, 0.9940771, -0.0004114, 0.0004105
2: -0.0065606, -0.0050208, -0.0065606, -0.0050208, -0.0013509, 0.0013418
3: 0.0038004, 0.0040140, 0.0038004, 0.0040140, -0.0001814, 0.0001820
4: 0.0023852, 0.0036022, 0.0023852, 0.0036022, -0.0011446, 0.0011520
5: 0.0060086, 0.0065111, 0.0060086, 0.0065111, -0.0005024, 0.0005024
6: -0.0013983, -0.0008639, -0.0013983, -0.0008639, -0.0004641, 0.0004610
7: -0.0083221, -0.0079387, -0.0083221, -0.0079387, -0.0003835, 0.0003835
8: 0.0048484, 0.0068715, 0.0048484, 0.0068715, -0.0015377, 0.0015499
9: -0.0036877, -0.0032079, -0.0036877, -0.0032079, -0.0004797, 0.0004797

### Unstable ReLU Count (Linear/Conv2D Layers)
- layer_idx=0, type=LayerType.Linear, total=256, inp1_unstable=5, inp2_unstable=5, delta_unstable=253
- layer_idx=2, type=LayerType.Linear, total=256, inp1_unstable=4, inp2_unstable=4, delta_unstable=256
- layer_idx=4, type=LayerType.Linear, total=256, inp1_unstable=0, inp2_unstable=0, delta_unstable=256
- layer_idx=6, type=LayerType.Linear, total=256, inp1_unstable=1, inp2_unstable=1, delta_unstable=256
- layer_idx=8, type=LayerType.Linear, total=10, inp1_unstable=0, inp2_unstable=0, delta_unstable=10

Time for backsubstitution: 1.10 seconds

### RS candidates at layer 1
type: RSZ, layer: 1, pos: 178
type: RSZ, layer: 1, pos: 251

Time for candidate selection: 0.12 seconds

### Candidate
type: RSZ, layer: 1, pos: 178

### Relational analysis RSZ of RS_RSZ2_RSZ1_RSZ2_RSZ1

#### Relational analysis RSZ result of RS_RSZ2_RSZ1_RSZ2_RSZ1
Status: Status.UNKNOWN
Output dim: 1, lower bound: -0.0003449, upper bound: 0.0003436
time: 0.56 seconds

### Relational analysis RSZ of RS_RSZ2_RSZ1_RSZ2_RSZ2

#### Relational analysis RSZ result of RS_RSZ2_RSZ1_RSZ2_RSZ2
Status: Status.UNKNOWN
Output dim: 1, lower bound: -0.0003441, upper bound: 0.0003443
time: 0.59 seconds

## BFS RS instance: RS_RSZ2_RSZ2_RSZ1

### Backsubstitution after applying RS history:
0: 0.0009457, 0.0011003, 0.0009457, 0.0011003, -0.0001507, 0.0001509
1: 0.9936197, 0.9940771, 0.9936197, 0.9940771, -0.0004123, 0.0004104
2: -0.0065606, -0.0050208, -0.0065606, -0.0050208, -0.0013608, 0.0013398
3: 0.0038004, 0.0040140, 0.0038004, 0.0040140, -0.0001813, 0.0001827
4: 0.0023852, 0.0036022, 0.0023852, 0.0036022, -0.0011430, 0.0011595
5: 0.0060086, 0.0065111, 0.0060086, 0.0065111, -0.0005024, 0.0005024
6: -0.0013983, -0.0008639, -0.0013983, -0.0008639, -0.0004676, 0.0004603
7: -0.0083221, -0.0079387, -0.0083221, -0.0079387, -0.0003835, 0.0003835
8: 0.0048484, 0.0068715, 0.0048484, 0.0068715, -0.0015350, 0.0015629
9: -0.0036877, -0.0032079, -0.0036877, -0.0032079, -0.0004797, 0.0004797

### Unstable ReLU Count (Linear/Conv2D Layers)
- layer_idx=0, type=LayerType.Linear, total=256, inp1_unstable=5, inp2_unstable=5, delta_unstable=253
- layer_idx=2, type=LayerType.Linear, total=256, inp1_unstable=4, inp2_unstable=4, delta_unstable=256
- layer_idx=4, type=LayerType.Linear, total=256, inp1_unstable=0, inp2_unstable=0, delta_unstable=256
- layer_idx=6, type=LayerType.Linear, total=256, inp1_unstable=1, inp2_unstable=1, delta_unstable=256
- layer_idx=8, type=LayerType.Linear, total=10, inp1_unstable=0, inp2_unstable=0, delta_unstable=10

Time for backsubstitution: 1.09 seconds

### RS candidates at layer 1
type: RSZ, layer: 1, pos: 178
type: RSZ, layer: 1, pos: 251

Time for candidate selection: 0.13 seconds

### Candidate
type: RSZ, layer: 1, pos: 178

### Relational analysis RSZ of RS_RSZ2_RSZ2_RSZ1_RSZ1

#### Relational analysis RSZ result of RS_RSZ2_RSZ2_RSZ1_RSZ1
Status: Status.UNKNOWN
Output dim: 1, lower bound: -0.0003443, upper bound: 0.0003441
time: 0.55 seconds

### Relational analysis RSZ of RS_RSZ2_RSZ2_RSZ1_RSZ2

#### Relational analysis RSZ result of RS_RSZ2_RSZ2_RSZ1_RSZ2
Status: Status.UNKNOWN
Output dim: 1, lower bound: -0.0003436, upper bound: 0.0003449
time: 0.59 seconds

## BFS RS instance: RS_RSZ2_RSZ2_RSZ2

### Backsubstitution after applying RS history:
0: 0.0009457, 0.0011003, 0.0009457, 0.0011003, -0.0001507, 0.0001509
1: 0.9936197, 0.9940771, 0.9936197, 0.9940771, -0.0004123, 0.0004103
2: -0.0065606, -0.0050208, -0.0065606, -0.0050208, -0.0013611, 0.0013394
3: 0.0038004, 0.0040140, 0.0038004, 0.0040140, -0.0001813, 0.0001827
4: 0.0023852, 0.0036022, 0.0023852, 0.0036022, -0.0011427, 0.0011598
5: 0.0060086, 0.0065111, 0.0060086, 0.0065111, -0.0005024, 0.0005024
6: -0.0013983, -0.0008639, -0.0013983, -0.0008639, -0.0004677, 0.0004601
7: -0.0083221, -0.0079387, -0.0083221, -0.0079387, -0.0003835, 0.0003835
8: 0.0048484, 0.0068715, 0.0048484, 0.0068715, -0.0015346, 0.0015634
9: -0.0036877, -0.0032079, -0.0036877, -0.0032079, -0.0004797, 0.0004797

### Unstable ReLU Count (Linear/Conv2D Layers)
- layer_idx=0, type=LayerType.Linear, total=256, inp1_unstable=5, inp2_unstable=5, delta_unstable=253
- layer_idx=2, type=LayerType.Linear, total=256, inp1_unstable=4, inp2_unstable=4, delta_unstable=256
- layer_idx=4, type=LayerType.Linear, total=256, inp1_unstable=0, inp2_unstable=0, delta_unstable=256
- layer_idx=6, type=LayerType.Linear, total=256, inp1_unstable=1, inp2_unstable=1, delta_unstable=256
- layer_idx=8, type=LayerType.Linear, total=10, inp1_unstable=0, inp2_unstable=0, delta_unstable=10

Time for backsubstitution: 1.10 seconds

### RS candidates at layer 1
type: RSZ, layer: 1, pos: 178
type: RSZ, layer: 1, pos: 251

Time for candidate selection: 0.11 seconds

### Candidate
type: RSZ, layer: 1, pos: 178

### Relational analysis RSZ of RS_RSZ2_RSZ2_RSZ2_RSZ1

#### Relational analysis RSZ result of RS_RSZ2_RSZ2_RSZ2_RSZ1
Status: Status.UNKNOWN
Output dim: 1, lower bound: -0.0003450, upper bound: 0.0003435
time: 0.54 seconds

### Relational analysis RSZ of RS_RSZ2_RSZ2_RSZ2_RSZ2

#### Relational analysis RSZ result of RS_RSZ2_RSZ2_RSZ2_RSZ2
Status: Status.UNKNOWN
Output dim: 1, lower bound: -0.0003443, upper bound: 0.0003443
time: 0.65 seconds

## Summary of splitting (split count: 3)
- Time for RS candidates: 2.41 seconds
RS_RSZ1_RSZ1_RSZ1_RSZ1, status: Status.UNKNOWN, split count: 4, time: 2.41
Output dim: 1, lower bound: -0.0003443, upper bound: 0.0003443
RS_RSZ1_RSZ1_RSZ1_RSZ2, status: Status.UNKNOWN, split count: 4, time: 2.41
Output dim: 1, lower bound: -0.0003435, upper bound: 0.0003450
RS_RSZ1_RSZ1_RSZ2_RSZ1, status: Status.UNKNOWN, split count: 4, time: 2.41
Output dim: 1, lower bound: -0.0003449, upper bound: 0.0003436
RS_RSZ1_RSZ1_RSZ2_RSZ2, status: Status.UNKNOWN, split count: 4, time: 2.41
Output dim: 1, lower bound: -0.0003441, upper bound: 0.0003443
RS_RSZ1_RSZ2_RSZ1_RSZ1, status: Status.UNKNOWN, split count: 4, time: 2.41
Output dim: 1, lower bound: -0.0003443, upper bound: 0.0003441
RS_RSZ1_RSZ2_RSZ1_RSZ2, status: Status.UNKNOWN, split count: 4, time: 2.41
Output dim: 1, lower bound: -0.0003436, upper bound: 0.0003449
RS_RSZ1_RSZ2_RSZ2_RSZ1, status: Status.UNKNOWN, split count: 4, time: 2.41
Output dim: 1, lower bound: -0.0003450, upper bound: 0.0003435
RS_RSZ1_RSZ2_RSZ2_RSZ2, status: Status.UNKNOWN, split count: 4, time: 2.41
Output dim: 1, lower bound: -0.0003443, upper bound: 0.0003443
RS_RSZ2_RSZ1_RSZ1_RSZ1, status: Status.UNKNOWN, split count: 4, time: 2.41
Output dim: 1, lower bound: -0.0003443, upper bound: 0.0003443
RS_RSZ2_RSZ1_RSZ1_RSZ2, status: Status.UNKNOWN, split count: 4, time: 2.41
Output dim: 1, lower bound: -0.0003435, upper bound: 0.0003450
RS_RSZ2_RSZ1_RSZ2_RSZ1, status: Status.UNKNOWN, split count: 4, time: 2.41
Output dim: 1, lower bound: -0.0003449, upper bound: 0.0003436
RS_RSZ2_RSZ1_RSZ2_RSZ2, status: Status.UNKNOWN, split count: 4, time: 2.41
Output dim: 1, lower bound: -0.0003441, upper bound: 0.0003443
RS_RSZ2_RSZ2_RSZ1_RSZ1, status: Status.UNKNOWN, split count: 4, time: 2.41
Output dim: 1, lower bound: -0.0003443, upper bound: 0.0003441
RS_RSZ2_RSZ2_RSZ1_RSZ2, status: Status.UNKNOWN, split count: 4, time: 2.41
Output dim: 1, lower bound: -0.0003436, upper bound: 0.0003449
RS_RSZ2_RSZ2_RSZ2_RSZ1, status: Status.UNKNOWN, split count: 4, time: 2.41
Output dim: 1, lower bound: -0.0003450, upper bound: 0.0003435
RS_RSZ2_RSZ2_RSZ2_RSZ2, status: Status.UNKNOWN, split count: 4, time: 2.41
Output dim: 1, lower bound: -0.0003443, upper bound: 0.0003443

## BFS RS instance: RS_RSZ1_RSZ1_RSZ1_RSZ1

### Backsubstitution after applying RS history:
0: 0.0009457, 0.0011003, 0.0009457, 0.0011003, -0.0001503, 0.0001503
1: 0.9936197, 0.9940771, 0.9936197, 0.9940771, -0.0004047, 0.0004046
2: -0.0065606, -0.0050208, -0.0065606, -0.0050208, -0.0012841, 0.0012833
3: 0.0038004, 0.0040140, 0.0038004, 0.0040140, -0.0001772, 0.0001772
4: 0.0023852, 0.0036022, 0.0023852, 0.0036022, -0.0011010, 0.0011017
5: 0.0060086, 0.0065111, 0.0060086, 0.0065111, -0.0005024, 0.0005024
6: -0.0013983, -0.0008639, -0.0013983, -0.0008639, -0.0004407, 0.0004404
7: -0.0083221, -0.0079387, -0.0083221, -0.0079387, -0.0003835, 0.0003835
8: 0.0048484, 0.0068715, 0.0048484, 0.0068715, -0.0014472, 0.0014484
9: -0.0036877, -0.0032079, -0.0036877, -0.0032079, -0.0004797, 0.0004797

### Unstable ReLU Count (Linear/Conv2D Layers)
- layer_idx=0, type=LayerType.Linear, total=256, inp1_unstable=5, inp2_unstable=5, delta_unstable=252
- layer_idx=2, type=LayerType.Linear, total=256, inp1_unstable=4, inp2_unstable=4, delta_unstable=256
- layer_idx=4, type=LayerType.Linear, total=256, inp1_unstable=0, inp2_unstable=0, delta_unstable=256
- layer_idx=6, type=LayerType.Linear, total=256, inp1_unstable=1, inp2_unstable=1, delta_unstable=256
- layer_idx=8, type=LayerType.Linear, total=10, inp1_unstable=0, inp2_unstable=0, delta_unstable=10

Time for backsubstitution: 1.09 seconds

### RS candidates at layer 1
type: RSZ, layer: 1, pos: 251

Time for candidate selection: 0.11 seconds

### Candidate
type: RSZ, layer: 1, pos: 251

### Relational analysis RSZ of RS_RSZ1_RSZ1_RSZ1_RSZ1_RSZ1

#### Relational analysis RSZ result of RS_RSZ1_RSZ1_RSZ1_RSZ1_RSZ1
Status: Status.UNKNOWN
Output dim: 1, lower bound: -0.0003435, upper bound: 0.0003434
time: 0.56 seconds

### Relational analysis RSZ of RS_RSZ1_RSZ1_RSZ1_RSZ1_RSZ2

#### Relational analysis RSZ result of RS_RSZ1_RSZ1_RSZ1_RSZ1_RSZ2
Status: Status.UNKNOWN
Output dim: 1, lower bound: -0.0003434, upper bound: 0.0003434
time: 0.56 seconds

## BFS RS instance: RS_RSZ1_RSZ1_RSZ1_RSZ2

### Backsubstitution after applying RS history:
0: 0.0009457, 0.0011003, 0.0009457, 0.0011003, -0.0001503, 0.0001503
1: 0.9936197, 0.9940771, 0.9936197, 0.9940771, -0.0004049, 0.0004046
2: -0.0065606, -0.0050208, -0.0065606, -0.0050208, -0.0012872, 0.0012837
3: 0.0038004, 0.0040140, 0.0038004, 0.0040140, -0.0001772, 0.0001774
4: 0.0023852, 0.0036022, 0.0023852, 0.0036022, -0.0011014, 0.0011041
5: 0.0060086, 0.0065111, 0.0060086, 0.0065111, -0.0005024, 0.0005024
6: -0.0013983, -0.0008639, -0.0013983, -0.0008639, -0.0004417, 0.0004405
7: -0.0083221, -0.0079387, -0.0083221, -0.0079387, -0.0003835, 0.0003835
8: 0.0048484, 0.0068715, 0.0048484, 0.0068715, -0.0014479, 0.0014524
9: -0.0036877, -0.0032079, -0.0036877, -0.0032079, -0.0004797, 0.0004797

### Unstable ReLU Count (Linear/Conv2D Layers)
- layer_idx=0, type=LayerType.Linear, total=256, inp1_unstable=5, inp2_unstable=5, delta_unstable=252
- layer_idx=2, type=LayerType.Linear, total=256, inp1_unstable=4, inp2_unstable=4, delta_unstable=256
- layer_idx=4, type=LayerType.Linear, total=256, inp1_unstable=0, inp2_unstable=0, delta_unstable=256
- layer_idx=6, type=LayerType.Linear, total=256, inp1_unstable=1, inp2_unstable=1, delta_unstable=256
- layer_idx=8, type=LayerType.Linear, total=10, inp1_unstable=0, inp2_unstable=0, delta_unstable=10

Time for backsubstitution: 1.12 seconds

### RS candidates at layer 1
type: RSZ, layer: 1, pos: 251

Time for candidate selection: 0.11 seconds

### Candidate
type: RSZ, layer: 1, pos: 251

### Relational analysis RSZ of RS_RSZ1_RSZ1_RSZ1_RSZ2_RSZ1

#### Relational analysis RSZ result of RS_RSZ1_RSZ1_RSZ1_RSZ2_RSZ1
Status: Status.UNKNOWN
Output dim: 1, lower bound: -0.0003427, upper bound: 0.0003441
time: 0.53 seconds

### Relational analysis RSZ of RS_RSZ1_RSZ1_RSZ1_RSZ2_RSZ2

#### Relational analysis RSZ result of RS_RSZ1_RSZ1_RSZ1_RSZ2_RSZ2
Status: Status.UNKNOWN
Output dim: 1, lower bound: -0.0003427, upper bound: 0.0003442
time: 0.57 seconds

## BFS RS instance: RS_RSZ1_RSZ1_RSZ2_RSZ1

### Backsubstitution after applying RS history:
0: 0.0009457, 0.0011003, 0.0009457, 0.0011003, -0.0001503, 0.0001503
1: 0.9936197, 0.9940771, 0.9936197, 0.9940771, -0.0004050, 0.0004045
2: -0.0065606, -0.0050208, -0.0065606, -0.0050208, -0.0012878, 0.0012825
3: 0.0038004, 0.0040140, 0.0038004, 0.0040140, -0.0001771, 0.0001775
4: 0.0023852, 0.0036022, 0.0023852, 0.0036022, -0.0011004, 0.0011046
5: 0.0060086, 0.0065111, 0.0060086, 0.0065111, -0.0005024, 0.0005024
6: -0.0013983, -0.0008639, -0.0013983, -0.0008639, -0.0004420, 0.0004401
7: -0.0083221, -0.0079387, -0.0083221, -0.0079387, -0.0003835, 0.0003835
8: 0.0048484, 0.0068715, 0.0048484, 0.0068715, -0.0014462, 0.0014532
9: -0.0036877, -0.0032079, -0.0036877, -0.0032079, -0.0004797, 0.0004797

### Unstable ReLU Count (Linear/Conv2D Layers)
- layer_idx=0, type=LayerType.Linear, total=256, inp1_unstable=5, inp2_unstable=5, delta_unstable=252
- layer_idx=2, type=LayerType.Linear, total=256, inp1_unstable=4, inp2_unstable=4, delta_unstable=256
- layer_idx=4, type=LayerType.Linear, total=256, inp1_unstable=0, inp2_unstable=0, delta_unstable=256
- layer_idx=6, type=LayerType.Linear, total=256, inp1_unstable=1, inp2_unstable=1, delta_unstable=256
- layer_idx=8, type=LayerType.Linear, total=10, inp1_unstable=0, inp2_unstable=0, delta_unstable=10

Time for backsubstitution: 1.10 seconds

### RS candidates at layer 1
type: RSZ, layer: 1, pos: 251

Time for candidate selection: 0.11 seconds

### Candidate
type: RSZ, layer: 1, pos: 251

### Relational analysis RSZ of RS_RSZ1_RSZ1_RSZ2_RSZ1_RSZ1

#### Relational analysis RSZ result of RS_RSZ1_RSZ1_RSZ2_RSZ1_RSZ1
Status: Status.UNKNOWN
Output dim: 1, lower bound: -0.0003442, upper bound: 0.0003427
time: 0.55 seconds

### Relational analysis RSZ of RS_RSZ1_RSZ1_RSZ2_RSZ1_RSZ2

#### Relational analysis RSZ result of RS_RSZ1_RSZ1_RSZ2_RSZ1_RSZ2
Status: Status.UNKNOWN
Output dim: 1, lower bound: -0.0003441, upper bound: 0.0003428
time: 0.55 seconds

## BFS RS instance: RS_RSZ1_RSZ1_RSZ2_RSZ2

### Backsubstitution after applying RS history:
0: 0.0009457, 0.0011003, 0.0009457, 0.0011003, -0.0001503, 0.0001503
1: 0.9936197, 0.9940771, 0.9936197, 0.9940771, -0.0004050, 0.0004041
2: -0.0065606, -0.0050208, -0.0065606, -0.0050208, -0.0012875, 0.0012784
3: 0.0038004, 0.0040140, 0.0038004, 0.0040140, -0.0001768, 0.0001774
4: 0.0023852, 0.0036022, 0.0023852, 0.0036022, -0.0010971, 0.0011043
5: 0.0060086, 0.0065111, 0.0060086, 0.0065111, -0.0005024, 0.0005024
6: -0.0013983, -0.0008639, -0.0013983, -0.0008639, -0.0004419, 0.0004387
7: -0.0083221, -0.0079387, -0.0083221, -0.0079387, -0.0003835, 0.0003835
8: 0.0048484, 0.0068715, 0.0048484, 0.0068715, -0.0014408, 0.0014528
9: -0.0036877, -0.0032079, -0.0036877, -0.0032079, -0.0004797, 0.0004797

### Unstable ReLU Count (Linear/Conv2D Layers)
- layer_idx=0, type=LayerType.Linear, total=256, inp1_unstable=5, inp2_unstable=5, delta_unstable=252
- layer_idx=2, type=LayerType.Linear, total=256, inp1_unstable=4, inp2_unstable=4, delta_unstable=256
- layer_idx=4, type=LayerType.Linear, total=256, inp1_unstable=0, inp2_unstable=0, delta_unstable=256
- layer_idx=6, type=LayerType.Linear, total=256, inp1_unstable=1, inp2_unstable=1, delta_unstable=256
- layer_idx=8, type=LayerType.Linear, total=10, inp1_unstable=0, inp2_unstable=0, delta_unstable=10

Time for backsubstitution: 1.12 seconds

### RS candidates at layer 1
type: RSZ, layer: 1, pos: 251

Time for candidate selection: 0.11 seconds

### Candidate
type: RSZ, layer: 1, pos: 251

### Relational analysis RSZ of RS_RSZ1_RSZ1_RSZ2_RSZ2_RSZ1

#### Relational analysis RSZ result of RS_RSZ1_RSZ1_RSZ2_RSZ2_RSZ1
Status: Status.UNKNOWN
Output dim: 1, lower bound: -0.0003433, upper bound: 0.0003434
time: 0.58 seconds

### Relational analysis RSZ of RS_RSZ1_RSZ1_RSZ2_RSZ2_RSZ2

#### Relational analysis RSZ result of RS_RSZ1_RSZ1_RSZ2_RSZ2_RSZ2
Status: Status.UNKNOWN
Output dim: 1, lower bound: -0.0003433, upper bound: 0.0003436
time: 0.53 seconds

## BFS RS instance: RS_RSZ1_RSZ2_RSZ1_RSZ1

### Backsubstitution after applying RS history:
0: 0.0009457, 0.0011003, 0.0009457, 0.0011003, -0.0001503, 0.0001504
1: 0.9936197, 0.9940771, 0.9936197, 0.9940771, -0.0004058, 0.0004043
2: -0.0065606, -0.0050208, -0.0065606, -0.0050208, -0.0012958, 0.0012807
3: 0.0038004, 0.0040140, 0.0038004, 0.0040140, -0.0001770, 0.0001780
4: 0.0023852, 0.0036022, 0.0023852, 0.0036022, -0.0010989, 0.0011101
5: 0.0060086, 0.0065111, 0.0060086, 0.0065111, -0.0005024, 0.0005024
6: -0.0013983, -0.0008639, -0.0013983, -0.0008639, -0.0004446, 0.0004395
7: -0.0083221, -0.0079387, -0.0083221, -0.0079387, -0.0003835, 0.0003835
8: 0.0048484, 0.0068715, 0.0048484, 0.0068715, -0.0014438, 0.0014639
9: -0.0036877, -0.0032079, -0.0036877, -0.0032079, -0.0004797, 0.0004797

### Unstable ReLU Count (Linear/Conv2D Layers)
- layer_idx=0, type=LayerType.Linear, total=256, inp1_unstable=5, inp2_unstable=5, delta_unstable=252
- layer_idx=2, type=LayerType.Linear, total=256, inp1_unstable=4, inp2_unstable=4, delta_unstable=256
- layer_idx=4, type=LayerType.Linear, total=256, inp1_unstable=0, inp2_unstable=0, delta_unstable=256
- layer_idx=6, type=LayerType.Linear, total=256, inp1_unstable=1, inp2_unstable=1, delta_unstable=256
- layer_idx=8, type=LayerType.Linear, total=10, inp1_unstable=0, inp2_unstable=0, delta_unstable=10

Time for backsubstitution: 1.11 seconds

### RS candidates at layer 1
type: RSZ, layer: 1, pos: 251

Time for candidate selection: 0.11 seconds

### Candidate
type: RSZ, layer: 1, pos: 251

### Relational analysis RSZ of RS_RSZ1_RSZ2_RSZ1_RSZ1_RSZ1

#### Relational analysis RSZ result of RS_RSZ1_RSZ2_RSZ1_RSZ1_RSZ1
Status: Status.UNKNOWN
Output dim: 1, lower bound: -0.0003436, upper bound: 0.0003433
time: 0.56 seconds

### Relational analysis RSZ of RS_RSZ1_RSZ2_RSZ1_RSZ1_RSZ2

#### Relational analysis RSZ result of RS_RSZ1_RSZ2_RSZ1_RSZ1_RSZ2
Status: Status.UNKNOWN
Output dim: 1, lower bound: -0.0003434, upper bound: 0.0003433
time: 0.56 seconds

## BFS RS instance: RS_RSZ1_RSZ2_RSZ1_RSZ2

### Backsubstitution after applying RS history:
0: 0.0009457, 0.0011003, 0.0009457, 0.0011003, -0.0001503, 0.0001504
1: 0.9936197, 0.9940771, 0.9936197, 0.9940771, -0.0004061, 0.0004045
2: -0.0065606, -0.0050208, -0.0065606, -0.0050208, -0.0012988, 0.0012829
3: 0.0038004, 0.0040140, 0.0038004, 0.0040140, -0.0001771, 0.0001782
4: 0.0023852, 0.0036022, 0.0023852, 0.0036022, -0.0011007, 0.0011125
5: 0.0060086, 0.0065111, 0.0060086, 0.0065111, -0.0005024, 0.0005024
6: -0.0013983, -0.0008639, -0.0013983, -0.0008639, -0.0004457, 0.0004402
7: -0.0083221, -0.0079387, -0.0083221, -0.0079387, -0.0003835, 0.0003835
8: 0.0048484, 0.0068715, 0.0048484, 0.0068715, -0.0014468, 0.0014679
9: -0.0036877, -0.0032079, -0.0036877, -0.0032079, -0.0004797, 0.0004797

### Unstable ReLU Count (Linear/Conv2D Layers)
- layer_idx=0, type=LayerType.Linear, total=256, inp1_unstable=5, inp2_unstable=5, delta_unstable=252
- layer_idx=2, type=LayerType.Linear, total=256, inp1_unstable=4, inp2_unstable=4, delta_unstable=256
- layer_idx=4, type=LayerType.Linear, total=256, inp1_unstable=0, inp2_unstable=0, delta_unstable=256
- layer_idx=6, type=LayerType.Linear, total=256, inp1_unstable=1, inp2_unstable=1, delta_unstable=256
- layer_idx=8, type=LayerType.Linear, total=10, inp1_unstable=0, inp2_unstable=0, delta_unstable=10

Time for backsubstitution: 1.10 seconds

### RS candidates at layer 1
type: RSZ, layer: 1, pos: 251

Time for candidate selection: 0.11 seconds

### Candidate
type: RSZ, layer: 1, pos: 251

### Relational analysis RSZ of RS_RSZ1_RSZ2_RSZ1_RSZ2_RSZ1

#### Relational analysis RSZ result of RS_RSZ1_RSZ2_RSZ1_RSZ2_RSZ1
Status: Status.UNKNOWN
Output dim: 1, lower bound: -0.0003428, upper bound: 0.0003441
time: 0.55 seconds

### Relational analysis RSZ of RS_RSZ1_RSZ2_RSZ1_RSZ2_RSZ2

#### Relational analysis RSZ result of RS_RSZ1_RSZ2_RSZ1_RSZ2_RSZ2
Status: Status.UNKNOWN
Output dim: 1, lower bound: -0.0003427, upper bound: 0.0003442
time: 0.57 seconds

## BFS RS instance: RS_RSZ1_RSZ2_RSZ2_RSZ1

### Backsubstitution after applying RS history:
0: 0.0009457, 0.0011003, 0.0009457, 0.0011003, -0.0001503, 0.0001504
1: 0.9936197, 0.9940771, 0.9936197, 0.9940771, -0.0004061, 0.0004043
2: -0.0065606, -0.0050208, -0.0065606, -0.0050208, -0.0012995, 0.0012800
3: 0.0038004, 0.0040140, 0.0038004, 0.0040140, -0.0001769, 0.0001782
4: 0.0023852, 0.0036022, 0.0023852, 0.0036022, -0.0010984, 0.0011130
5: 0.0060086, 0.0065111, 0.0060086, 0.0065111, -0.0005024, 0.0005024
6: -0.0013983, -0.0008639, -0.0013983, -0.0008639, -0.0004459, 0.0004392
7: -0.0083221, -0.0079387, -0.0083221, -0.0079387, -0.0003835, 0.0003835
8: 0.0048484, 0.0068715, 0.0048484, 0.0068715, -0.0014429, 0.0014688
9: -0.0036877, -0.0032079, -0.0036877, -0.0032079, -0.0004797, 0.0004797

### Unstable ReLU Count (Linear/Conv2D Layers)
- layer_idx=0, type=LayerType.Linear, total=256, inp1_unstable=5, inp2_unstable=5, delta_unstable=252
- layer_idx=2, type=LayerType.Linear, total=256, inp1_unstable=4, inp2_unstable=4, delta_unstable=256
- layer_idx=4, type=LayerType.Linear, total=256, inp1_unstable=0, inp2_unstable=0, delta_unstable=256
- layer_idx=6, type=LayerType.Linear, total=256, inp1_unstable=1, inp2_unstable=1, delta_unstable=256
- layer_idx=8, type=LayerType.Linear, total=10, inp1_unstable=0, inp2_unstable=0, delta_unstable=10

Time for backsubstitution: 1.09 seconds

### RS candidates at layer 1
type: RSZ, layer: 1, pos: 251

Time for candidate selection: 0.11 seconds

### Candidate
type: RSZ, layer: 1, pos: 251

### Relational analysis RSZ of RS_RSZ1_RSZ2_RSZ2_RSZ1_RSZ1

#### Relational analysis RSZ result of RS_RSZ1_RSZ2_RSZ2_RSZ1_RSZ1
Status: Status.UNKNOWN
Output dim: 1, lower bound: -0.0003442, upper bound: 0.0003427
time: 0.53 seconds

### Relational analysis RSZ of RS_RSZ1_RSZ2_RSZ2_RSZ1_RSZ2

#### Relational analysis RSZ result of RS_RSZ1_RSZ2_RSZ2_RSZ1_RSZ2
Status: Status.UNKNOWN
Output dim: 1, lower bound: -0.0003441, upper bound: 0.0003427
time: 0.53 seconds

## BFS RS instance: RS_RSZ1_RSZ2_RSZ2_RSZ2

### Backsubstitution after applying RS history:
0: 0.0009457, 0.0011003, 0.0009457, 0.0011003, -0.0001503, 0.0001504
1: 0.9936197, 0.9940771, 0.9936197, 0.9940771, -0.0004061, 0.0004040
2: -0.0065606, -0.0050208, -0.0065606, -0.0050208, -0.0012992, 0.0012771
3: 0.0038004, 0.0040140, 0.0038004, 0.0040140, -0.0001767, 0.0001782
4: 0.0023852, 0.0036022, 0.0023852, 0.0036022, -0.0010961, 0.0011128
5: 0.0060086, 0.0065111, 0.0060086, 0.0065111, -0.0005024, 0.0005024
6: -0.0013983, -0.0008639, -0.0013983, -0.0008639, -0.0004458, 0.0004382
7: -0.0083221, -0.0079387, -0.0083221, -0.0079387, -0.0003835, 0.0003835
8: 0.0048484, 0.0068715, 0.0048484, 0.0068715, -0.0014392, 0.0014684
9: -0.0036877, -0.0032079, -0.0036877, -0.0032079, -0.0004797, 0.0004797

### Unstable ReLU Count (Linear/Conv2D Layers)
- layer_idx=0, type=LayerType.Linear, total=256, inp1_unstable=5, inp2_unstable=5, delta_unstable=252
- layer_idx=2, type=LayerType.Linear, total=256, inp1_unstable=4, inp2_unstable=4, delta_unstable=256
- layer_idx=4, type=LayerType.Linear, total=256, inp1_unstable=0, inp2_unstable=0, delta_unstable=256
- layer_idx=6, type=LayerType.Linear, total=256, inp1_unstable=1, inp2_unstable=1, delta_unstable=256
- layer_idx=8, type=LayerType.Linear, total=10, inp1_unstable=0, inp2_unstable=0, delta_unstable=10

Time for backsubstitution: 1.10 seconds

### RS candidates at layer 1
type: RSZ, layer: 1, pos: 251

Time for candidate selection: 0.11 seconds

### Candidate
type: RSZ, layer: 1, pos: 251

### Relational analysis RSZ of RS_RSZ1_RSZ2_RSZ2_RSZ2_RSZ1

#### Relational analysis RSZ result of RS_RSZ1_RSZ2_RSZ2_RSZ2_RSZ1
Status: Status.UNKNOWN
Output dim: 1, lower bound: -0.0003434, upper bound: 0.0003433
time: 0.53 seconds

### Relational analysis RSZ of RS_RSZ1_RSZ2_RSZ2_RSZ2_RSZ2

#### Relational analysis RSZ result of RS_RSZ1_RSZ2_RSZ2_RSZ2_RSZ2
Status: Status.UNKNOWN
Output dim: 1, lower bound: -0.0003434, upper bound: 0.0003435
time: 0.56 seconds

## BFS RS instance: RS_RSZ2_RSZ1_RSZ1_RSZ1

### Backsubstitution after applying RS history:
0: 0.0009457, 0.0011003, 0.0009457, 0.0011003, -0.0001504, 0.0001504
1: 0.9936197, 0.9940771, 0.9936197, 0.9940771, -0.0004057, 0.0004053
2: -0.0065606, -0.0050208, -0.0065606, -0.0050208, -0.0012951, 0.0012907
3: 0.0038004, 0.0040140, 0.0038004, 0.0040140, -0.0001776, 0.0001780
4: 0.0023852, 0.0036022, 0.0023852, 0.0036022, -0.0011068, 0.0011104
5: 0.0060086, 0.0065111, 0.0060086, 0.0065111, -0.0005024, 0.0005024
6: -0.0013983, -0.0008639, -0.0013983, -0.0008639, -0.0004445, 0.0004429
7: -0.0083221, -0.0079387, -0.0083221, -0.0079387, -0.0003835, 0.0003835
8: 0.0048484, 0.0068715, 0.0048484, 0.0068715, -0.0014570, 0.0014634
9: -0.0036877, -0.0032079, -0.0036877, -0.0032079, -0.0004797, 0.0004797

### Unstable ReLU Count (Linear/Conv2D Layers)
- layer_idx=0, type=LayerType.Linear, total=256, inp1_unstable=5, inp2_unstable=5, delta_unstable=252
- layer_idx=2, type=LayerType.Linear, total=256, inp1_unstable=4, inp2_unstable=4, delta_unstable=256
- layer_idx=4, type=LayerType.Linear, total=256, inp1_unstable=0, inp2_unstable=0, delta_unstable=256
- layer_idx=6, type=LayerType.Linear, total=256, inp1_unstable=1, inp2_unstable=1, delta_unstable=256
- layer_idx=8, type=LayerType.Linear, total=10, inp1_unstable=0, inp2_unstable=0, delta_unstable=10

Time for backsubstitution: 1.09 seconds

### RS candidates at layer 1
type: RSZ, layer: 1, pos: 251

Time for candidate selection: 0.11 seconds

### Candidate
type: RSZ, layer: 1, pos: 251

### Relational analysis RSZ of RS_RSZ2_RSZ1_RSZ1_RSZ1_RSZ1

#### Relational analysis RSZ result of RS_RSZ2_RSZ1_RSZ1_RSZ1_RSZ1
Status: Status.UNKNOWN
Output dim: 1, lower bound: -0.0003435, upper bound: 0.0003434
time: 0.55 seconds

### Relational analysis RSZ of RS_RSZ2_RSZ1_RSZ1_RSZ1_RSZ2

#### Relational analysis RSZ result of RS_RSZ2_RSZ1_RSZ1_RSZ1_RSZ2
Status: Status.UNKNOWN
Output dim: 1, lower bound: -0.0003434, upper bound: 0.0003434
time: 0.57 seconds

## BFS RS instance: RS_RSZ2_RSZ1_RSZ1_RSZ2

### Backsubstitution after applying RS history:
0: 0.0009457, 0.0011003, 0.0009457, 0.0011003, -0.0001503, 0.0001504
1: 0.9936197, 0.9940771, 0.9936197, 0.9940771, -0.0004060, 0.0004051
2: -0.0065606, -0.0050208, -0.0065606, -0.0050208, -0.0012981, 0.0012892
3: 0.0038004, 0.0040140, 0.0038004, 0.0040140, -0.0001775, 0.0001782
4: 0.0023852, 0.0036022, 0.0023852, 0.0036022, -0.0011057, 0.0011128
5: 0.0060086, 0.0065111, 0.0060086, 0.0065111, -0.0005024, 0.0005024
6: -0.0013983, -0.0008639, -0.0013983, -0.0008639, -0.0004456, 0.0004424
7: -0.0083221, -0.0079387, -0.0083221, -0.0079387, -0.0003835, 0.0003835
8: 0.0048484, 0.0068715, 0.0048484, 0.0068715, -0.0014551, 0.0014674
9: -0.0036877, -0.0032079, -0.0036877, -0.0032079, -0.0004797, 0.0004797

### Unstable ReLU Count (Linear/Conv2D Layers)
- layer_idx=0, type=LayerType.Linear, total=256, inp1_unstable=5, inp2_unstable=5, delta_unstable=252
- layer_idx=2, type=LayerType.Linear, total=256, inp1_unstable=4, inp2_unstable=4, delta_unstable=256
- layer_idx=4, type=LayerType.Linear, total=256, inp1_unstable=0, inp2_unstable=0, delta_unstable=256
- layer_idx=6, type=LayerType.Linear, total=256, inp1_unstable=1, inp2_unstable=1, delta_unstable=256
- layer_idx=8, type=LayerType.Linear, total=10, inp1_unstable=0, inp2_unstable=0, delta_unstable=10

Time for backsubstitution: 1.09 seconds

### RS candidates at layer 1
type: RSZ, layer: 1, pos: 251

Time for candidate selection: 0.11 seconds

### Candidate
type: RSZ, layer: 1, pos: 251

### Relational analysis RSZ of RS_RSZ2_RSZ1_RSZ1_RSZ2_RSZ1

#### Relational analysis RSZ result of RS_RSZ2_RSZ1_RSZ1_RSZ2_RSZ1
Status: Status.UNKNOWN
Output dim: 1, lower bound: -0.0003427, upper bound: 0.0003441
time: 0.54 seconds

### Relational analysis RSZ of RS_RSZ2_RSZ1_RSZ1_RSZ2_RSZ2

#### Relational analysis RSZ result of RS_RSZ2_RSZ1_RSZ1_RSZ2_RSZ2
Status: Status.UNKNOWN
Output dim: 1, lower bound: -0.0003427, upper bound: 0.0003442
time: 0.56 seconds

## BFS RS instance: RS_RSZ2_RSZ1_RSZ2_RSZ1

### Backsubstitution after applying RS history:
0: 0.0009457, 0.0011003, 0.0009457, 0.0011003, -0.0001503, 0.0001504
1: 0.9936197, 0.9940771, 0.9936197, 0.9940771, -0.0004061, 0.0004052
2: -0.0065606, -0.0050208, -0.0065606, -0.0050208, -0.0012988, 0.0012895
3: 0.0038004, 0.0040140, 0.0038004, 0.0040140, -0.0001776, 0.0001782
4: 0.0023852, 0.0036022, 0.0023852, 0.0036022, -0.0011059, 0.0011134
5: 0.0060086, 0.0065111, 0.0060086, 0.0065111, -0.0005024, 0.0005024
6: -0.0013983, -0.0008639, -0.0013983, -0.0008639, -0.0004458, 0.0004426
7: -0.0083221, -0.0079387, -0.0083221, -0.0079387, -0.0003835, 0.0003835
8: 0.0048484, 0.0068715, 0.0048484, 0.0068715, -0.0014555, 0.0014682
9: -0.0036877, -0.0032079, -0.0036877, -0.0032079, -0.0004797, 0.0004797

### Unstable ReLU Count (Linear/Conv2D Layers)
- layer_idx=0, type=LayerType.Linear, total=256, inp1_unstable=5, inp2_unstable=5, delta_unstable=252
- layer_idx=2, type=LayerType.Linear, total=256, inp1_unstable=4, inp2_unstable=4, delta_unstable=256
- layer_idx=4, type=LayerType.Linear, total=256, inp1_unstable=0, inp2_unstable=0, delta_unstable=256
- layer_idx=6, type=LayerType.Linear, total=256, inp1_unstable=1, inp2_unstable=1, delta_unstable=256
- layer_idx=8, type=LayerType.Linear, total=10, inp1_unstable=0, inp2_unstable=0, delta_unstable=10

Time for backsubstitution: 1.13 seconds

### RS candidates at layer 1
type: RSZ, layer: 1, pos: 251

Time for candidate selection: 0.12 seconds

### Candidate
type: RSZ, layer: 1, pos: 251

### Relational analysis RSZ of RS_RSZ2_RSZ1_RSZ2_RSZ1_RSZ1

#### Relational analysis RSZ result of RS_RSZ2_RSZ1_RSZ2_RSZ1_RSZ1
Status: Status.UNKNOWN
Output dim: 1, lower bound: -0.0003442, upper bound: 0.0003427
time: 0.56 seconds

### Relational analysis RSZ of RS_RSZ2_RSZ1_RSZ2_RSZ1_RSZ2

#### Relational analysis RSZ result of RS_RSZ2_RSZ1_RSZ2_RSZ1_RSZ2
Status: Status.UNKNOWN
Output dim: 1, lower bound: -0.0003441, upper bound: 0.0003428
time: 0.57 seconds

## BFS RS instance: RS_RSZ2_RSZ1_RSZ2_RSZ2

### Backsubstitution after applying RS history:
0: 0.0009457, 0.0011003, 0.0009457, 0.0011003, -0.0001503, 0.0001504
1: 0.9936197, 0.9940771, 0.9936197, 0.9940771, -0.0004060, 0.0004048
2: -0.0065606, -0.0050208, -0.0065606, -0.0050208, -0.0012985, 0.0012855
3: 0.0038004, 0.0040140, 0.0038004, 0.0040140, -0.0001773, 0.0001782
4: 0.0023852, 0.0036022, 0.0023852, 0.0036022, -0.0011027, 0.0011131
5: 0.0060086, 0.0065111, 0.0060086, 0.0065111, -0.0005024, 0.0005024
6: -0.0013983, -0.0008639, -0.0013983, -0.0008639, -0.0004457, 0.0004411
7: -0.0083221, -0.0079387, -0.0083221, -0.0079387, -0.0003835, 0.0003835
8: 0.0048484, 0.0068715, 0.0048484, 0.0068715, -0.0014501, 0.0014678
9: -0.0036877, -0.0032079, -0.0036877, -0.0032079, -0.0004797, 0.0004797

### Unstable ReLU Count (Linear/Conv2D Layers)
- layer_idx=0, type=LayerType.Linear, total=256, inp1_unstable=5, inp2_unstable=5, delta_unstable=252
- layer_idx=2, type=LayerType.Linear, total=256, inp1_unstable=4, inp2_unstable=4, delta_unstable=256
- layer_idx=4, type=LayerType.Linear, total=256, inp1_unstable=0, inp2_unstable=0, delta_unstable=256
- layer_idx=6, type=LayerType.Linear, total=256, inp1_unstable=1, inp2_unstable=1, delta_unstable=256
- layer_idx=8, type=LayerType.Linear, total=10, inp1_unstable=0, inp2_unstable=0, delta_unstable=10

Time for backsubstitution: 1.23 seconds

### RS candidates at layer 1
type: RSZ, layer: 1, pos: 251

Time for candidate selection: 0.13 seconds

### Candidate
type: RSZ, layer: 1, pos: 251

### Relational analysis RSZ of RS_RSZ2_RSZ1_RSZ2_RSZ2_RSZ1

#### Relational analysis RSZ result of RS_RSZ2_RSZ1_RSZ2_RSZ2_RSZ1
Status: Status.UNKNOWN
Output dim: 1, lower bound: -0.0003433, upper bound: 0.0003434
time: 0.58 seconds

### Relational analysis RSZ of RS_RSZ2_RSZ1_RSZ2_RSZ2_RSZ2

#### Relational analysis RSZ result of RS_RSZ2_RSZ1_RSZ2_RSZ2_RSZ2
Status: Status.UNKNOWN
Output dim: 1, lower bound: -0.0003433, upper bound: 0.0003436
time: 0.54 seconds

## BFS RS instance: RS_RSZ2_RSZ2_RSZ1_RSZ1

### Backsubstitution after applying RS history:
0: 0.0009457, 0.0011003, 0.0009457, 0.0011003, -0.0001503, 0.0001505
1: 0.9936197, 0.9940771, 0.9936197, 0.9940771, -0.0004068, 0.0004050
2: -0.0065606, -0.0050208, -0.0065606, -0.0050208, -0.0013071, 0.0012875
3: 0.0038004, 0.0040140, 0.0038004, 0.0040140, -0.0001774, 0.0001787
4: 0.0023852, 0.0036022, 0.0023852, 0.0036022, -0.0011043, 0.0011190
5: 0.0060086, 0.0065111, 0.0060086, 0.0065111, -0.0005024, 0.0005024
6: -0.0013983, -0.0008639, -0.0013983, -0.0008639, -0.0004486, 0.0004419
7: -0.0083221, -0.0079387, -0.0083221, -0.0079387, -0.0003835, 0.0003835
8: 0.0048484, 0.0068715, 0.0048484, 0.0068715, -0.0014528, 0.0014788
9: -0.0036877, -0.0032079, -0.0036877, -0.0032079, -0.0004797, 0.0004797

### Unstable ReLU Count (Linear/Conv2D Layers)
- layer_idx=0, type=LayerType.Linear, total=256, inp1_unstable=5, inp2_unstable=5, delta_unstable=252
- layer_idx=2, type=LayerType.Linear, total=256, inp1_unstable=4, inp2_unstable=4, delta_unstable=256
- layer_idx=4, type=LayerType.Linear, total=256, inp1_unstable=0, inp2_unstable=0, delta_unstable=256
- layer_idx=6, type=LayerType.Linear, total=256, inp1_unstable=1, inp2_unstable=1, delta_unstable=256
- layer_idx=8, type=LayerType.Linear, total=10, inp1_unstable=0, inp2_unstable=0, delta_unstable=10

Time for backsubstitution: 1.13 seconds

### RS candidates at layer 1
type: RSZ, layer: 1, pos: 251

Time for candidate selection: 0.11 seconds

### Candidate
type: RSZ, layer: 1, pos: 251

### Relational analysis RSZ of RS_RSZ2_RSZ2_RSZ1_RSZ1_RSZ1

#### Relational analysis RSZ result of RS_RSZ2_RSZ2_RSZ1_RSZ1_RSZ1
Status: Status.UNKNOWN
Output dim: 1, lower bound: -0.0003436, upper bound: 0.0003433
time: 0.56 seconds

### Relational analysis RSZ of RS_RSZ2_RSZ2_RSZ1_RSZ1_RSZ2

#### Relational analysis RSZ result of RS_RSZ2_RSZ2_RSZ1_RSZ1_RSZ2
Status: Status.UNKNOWN
Output dim: 1, lower bound: -0.0003434, upper bound: 0.0003433
time: 0.54 seconds

## BFS RS instance: RS_RSZ2_RSZ2_RSZ1_RSZ2

### Backsubstitution after applying RS history:
0: 0.0009457, 0.0011003, 0.0009457, 0.0011003, -0.0001503, 0.0001505
1: 0.9936197, 0.9940771, 0.9936197, 0.9940771, -0.0004071, 0.0004050
2: -0.0065606, -0.0050208, -0.0065606, -0.0050208, -0.0013101, 0.0012878
3: 0.0038004, 0.0040140, 0.0038004, 0.0040140, -0.0001775, 0.0001789
4: 0.0023852, 0.0036022, 0.0023852, 0.0036022, -0.0011046, 0.0011214
5: 0.0060086, 0.0065111, 0.0060086, 0.0065111, -0.0005024, 0.0005024
6: -0.0013983, -0.0008639, -0.0013983, -0.0008639, -0.0004497, 0.0004420
7: -0.0083221, -0.0079387, -0.0083221, -0.0079387, -0.0003835, 0.0003835
8: 0.0048484, 0.0068715, 0.0048484, 0.0068715, -0.0014532, 0.0014828
9: -0.0036877, -0.0032079, -0.0036877, -0.0032079, -0.0004797, 0.0004797

### Unstable ReLU Count (Linear/Conv2D Layers)
- layer_idx=0, type=LayerType.Linear, total=256, inp1_unstable=5, inp2_unstable=5, delta_unstable=252
- layer_idx=2, type=LayerType.Linear, total=256, inp1_unstable=4, inp2_unstable=4, delta_unstable=256
- layer_idx=4, type=LayerType.Linear, total=256, inp1_unstable=0, inp2_unstable=0, delta_unstable=256
- layer_idx=6, type=LayerType.Linear, total=256, inp1_unstable=1, inp2_unstable=1, delta_unstable=256
- layer_idx=8, type=LayerType.Linear, total=10, inp1_unstable=0, inp2_unstable=0, delta_unstable=10

Time for backsubstitution: 1.14 seconds

### RS candidates at layer 1
type: RSZ, layer: 1, pos: 251

Time for candidate selection: 0.11 seconds

### Candidate
type: RSZ, layer: 1, pos: 251

### Relational analysis RSZ of RS_RSZ2_RSZ2_RSZ1_RSZ2_RSZ1

#### Relational analysis RSZ result of RS_RSZ2_RSZ2_RSZ1_RSZ2_RSZ1
Status: Status.UNKNOWN
Output dim: 1, lower bound: -0.0003428, upper bound: 0.0003441
time: 0.57 seconds

### Relational analysis RSZ of RS_RSZ2_RSZ2_RSZ1_RSZ2_RSZ2

#### Relational analysis RSZ result of RS_RSZ2_RSZ2_RSZ1_RSZ2_RSZ2
Status: Status.UNKNOWN
Output dim: 1, lower bound: -0.0003427, upper bound: 0.0003442
time: 0.56 seconds

## BFS RS instance: RS_RSZ2_RSZ2_RSZ2_RSZ1

### Backsubstitution after applying RS history:
0: 0.0009457, 0.0011003, 0.0009457, 0.0011003, -0.0001503, 0.0001505
1: 0.9936197, 0.9940771, 0.9936197, 0.9940771, -0.0004072, 0.0004049
2: -0.0065606, -0.0050208, -0.0065606, -0.0050208, -0.0013108, 0.0012872
3: 0.0038004, 0.0040140, 0.0038004, 0.0040140, -0.0001774, 0.0001790
4: 0.0023852, 0.0036022, 0.0023852, 0.0036022, -0.0011041, 0.0011219
5: 0.0060086, 0.0065111, 0.0060086, 0.0065111, -0.0005024, 0.0005024
6: -0.0013983, -0.0008639, -0.0013983, -0.0008639, -0.0004499, 0.0004417
7: -0.0083221, -0.0079387, -0.0083221, -0.0079387, -0.0003835, 0.0003835
8: 0.0048484, 0.0068715, 0.0048484, 0.0068715, -0.0014524, 0.0014837
9: -0.0036877, -0.0032079, -0.0036877, -0.0032079, -0.0004797, 0.0004797

### Unstable ReLU Count (Linear/Conv2D Layers)
- layer_idx=0, type=LayerType.Linear, total=256, inp1_unstable=5, inp2_unstable=5, delta_unstable=252
- layer_idx=2, type=LayerType.Linear, total=256, inp1_unstable=4, inp2_unstable=4, delta_unstable=256
- layer_idx=4, type=LayerType.Linear, total=256, inp1_unstable=0, inp2_unstable=0, delta_unstable=256
- layer_idx=6, type=LayerType.Linear, total=256, inp1_unstable=1, inp2_unstable=1, delta_unstable=256
- layer_idx=8, type=LayerType.Linear, total=10, inp1_unstable=0, inp2_unstable=0, delta_unstable=10

Time for backsubstitution: 1.12 seconds

### RS candidates at layer 1
type: RSZ, layer: 1, pos: 251

Time for candidate selection: 0.10 seconds

### Candidate
type: RSZ, layer: 1, pos: 251

### Relational analysis RSZ of RS_RSZ2_RSZ2_RSZ2_RSZ1_RSZ1

#### Relational analysis RSZ result of RS_RSZ2_RSZ2_RSZ2_RSZ1_RSZ1
Status: Status.UNKNOWN
Output dim: 1, lower bound: -0.0003442, upper bound: 0.0003427
time: 0.53 seconds

### Relational analysis RSZ of RS_RSZ2_RSZ2_RSZ2_RSZ1_RSZ2

#### Relational analysis RSZ result of RS_RSZ2_RSZ2_RSZ2_RSZ1_RSZ2
Status: Status.UNKNOWN
Output dim: 1, lower bound: -0.0003441, upper bound: 0.0003427
time: 0.53 seconds

## BFS RS instance: RS_RSZ2_RSZ2_RSZ2_RSZ2

### Backsubstitution after applying RS history:
0: 0.0009457, 0.0011003, 0.0009457, 0.0011003, -0.0001503, 0.0001505
1: 0.9936197, 0.9940771, 0.9936197, 0.9940771, -0.0004071, 0.0004047
2: -0.0065606, -0.0050208, -0.0065606, -0.0050208, -0.0013105, 0.0012841
3: 0.0038004, 0.0040140, 0.0038004, 0.0040140, -0.0001772, 0.0001790
4: 0.0023852, 0.0036022, 0.0023852, 0.0036022, -0.0011017, 0.0011217
5: 0.0060086, 0.0065111, 0.0060086, 0.0065111, -0.0005024, 0.0005024
6: -0.0013983, -0.0008639, -0.0013983, -0.0008639, -0.0004498, 0.0004407
7: -0.0083221, -0.0079387, -0.0083221, -0.0079387, -0.0003835, 0.0003835
8: 0.0048484, 0.0068715, 0.0048484, 0.0068715, -0.0014484, 0.0014833
9: -0.0036877, -0.0032079, -0.0036877, -0.0032079, -0.0004797, 0.0004797

### Unstable ReLU Count (Linear/Conv2D Layers)
- layer_idx=0, type=LayerType.Linear, total=256, inp1_unstable=5, inp2_unstable=5, delta_unstable=252
- layer_idx=2, type=LayerType.Linear, total=256, inp1_unstable=4, inp2_unstable=4, delta_unstable=256
- layer_idx=4, type=LayerType.Linear, total=256, inp1_unstable=0, inp2_unstable=0, delta_unstable=256
- layer_idx=6, type=LayerType.Linear, total=256, inp1_unstable=1, inp2_unstable=1, delta_unstable=256
- layer_idx=8, type=LayerType.Linear, total=10, inp1_unstable=0, inp2_unstable=0, delta_unstable=10

Time for backsubstitution: 1.11 seconds

### RS candidates at layer 1
type: RSZ, layer: 1, pos: 251

Time for candidate selection: 0.11 seconds

### Candidate
type: RSZ, layer: 1, pos: 251

### Relational analysis RSZ of RS_RSZ2_RSZ2_RSZ2_RSZ2_RSZ1

#### Relational analysis RSZ result of RS_RSZ2_RSZ2_RSZ2_RSZ2_RSZ1
Status: Status.UNKNOWN
Output dim: 1, lower bound: -0.0003434, upper bound: 0.0003433
time: 0.53 seconds

### Relational analysis RSZ of RS_RSZ2_RSZ2_RSZ2_RSZ2_RSZ2

#### Relational analysis RSZ result of RS_RSZ2_RSZ2_RSZ2_RSZ2_RSZ2
Status: Status.UNKNOWN
Output dim: 1, lower bound: -0.0003434, upper bound: 0.0003435
time: 0.55 seconds

## Summary of splitting (split count: 4)
- Time for RS candidates: 2.32 seconds
RS_RSZ1_RSZ1_RSZ1_RSZ1_RSZ1, status: Status.UNKNOWN, split count: 5, time: 2.32
Output dim: 1, lower bound: -0.0003435, upper bound: 0.0003434
RS_RSZ1_RSZ1_RSZ1_RSZ1_RSZ2, status: Status.UNKNOWN, split count: 5, time: 2.32
Output dim: 1, lower bound: -0.0003434, upper bound: 0.0003434
RS_RSZ1_RSZ1_RSZ1_RSZ2_RSZ1, status: Status.UNKNOWN, split count: 5, time: 2.32
Output dim: 1, lower bound: -0.0003427, upper bound: 0.0003441
RS_RSZ1_RSZ1_RSZ1_RSZ2_RSZ2, status: Status.UNKNOWN, split count: 5, time: 2.32
Output dim: 1, lower bound: -0.0003427, upper bound: 0.0003442
RS_RSZ1_RSZ1_RSZ2_RSZ1_RSZ1, status: Status.UNKNOWN, split count: 5, time: 2.32
Output dim: 1, lower bound: -0.0003442, upper bound: 0.0003427
RS_RSZ1_RSZ1_RSZ2_RSZ1_RSZ2, status: Status.UNKNOWN, split count: 5, time: 2.32
Output dim: 1, lower bound: -0.0003441, upper bound: 0.0003428
RS_RSZ1_RSZ1_RSZ2_RSZ2_RSZ1, status: Status.UNKNOWN, split count: 5, time: 2.32
Output dim: 1, lower bound: -0.0003433, upper bound: 0.0003434
RS_RSZ1_RSZ1_RSZ2_RSZ2_RSZ2, status: Status.UNKNOWN, split count: 5, time: 2.32
Output dim: 1, lower bound: -0.0003433, upper bound: 0.0003436
RS_RSZ1_RSZ2_RSZ1_RSZ1_RSZ1, status: Status.UNKNOWN, split count: 5, time: 2.32
Output dim: 1, lower bound: -0.0003436, upper bound: 0.0003433
RS_RSZ1_RSZ2_RSZ1_RSZ1_RSZ2, status: Status.UNKNOWN, split count: 5, time: 2.32
Output dim: 1, lower bound: -0.0003434, upper bound: 0.0003433
RS_RSZ1_RSZ2_RSZ1_RSZ2_RSZ1, status: Status.UNKNOWN, split count: 5, time: 2.32
Output dim: 1, lower bound: -0.0003428, upper bound: 0.0003441
RS_RSZ1_RSZ2_RSZ1_RSZ2_RSZ2, status: Status.UNKNOWN, split count: 5, time: 2.32
Output dim: 1, lower bound: -0.0003427, upper bound: 0.0003442
RS_RSZ1_RSZ2_RSZ2_RSZ1_RSZ1, status: Status.UNKNOWN, split count: 5, time: 2.32
Output dim: 1, lower bound: -0.0003442, upper bound: 0.0003427
RS_RSZ1_RSZ2_RSZ2_RSZ1_RSZ2, status: Status.UNKNOWN, split count: 5, time: 2.32
Output dim: 1, lower bound: -0.0003441, upper bound: 0.0003427
RS_RSZ1_RSZ2_RSZ2_RSZ2_RSZ1, status: Status.UNKNOWN, split count: 5, time: 2.32
Output dim: 1, lower bound: -0.0003434, upper bound: 0.0003433
RS_RSZ1_RSZ2_RSZ2_RSZ2_RSZ2, status: Status.UNKNOWN, split count: 5, time: 2.32
Output dim: 1, lower bound: -0.0003434, upper bound: 0.0003435
RS_RSZ2_RSZ1_RSZ1_RSZ1_RSZ1, status: Status.UNKNOWN, split count: 5, time: 2.32
Output dim: 1, lower bound: -0.0003435, upper bound: 0.0003434
RS_RSZ2_RSZ1_RSZ1_RSZ1_RSZ2, status: Status.UNKNOWN, split count: 5, time: 2.32
Output dim: 1, lower bound: -0.0003434, upper bound: 0.0003434
RS_RSZ2_RSZ1_RSZ1_RSZ2_RSZ1, status: Status.UNKNOWN, split count: 5, time: 2.32
Output dim: 1, lower bound: -0.0003427, upper bound: 0.0003441
RS_RSZ2_RSZ1_RSZ1_RSZ2_RSZ2, status: Status.UNKNOWN, split count: 5, time: 2.32
Output dim: 1, lower bound: -0.0003427, upper bound: 0.0003442
RS_RSZ2_RSZ1_RSZ2_RSZ1_RSZ1, status: Status.UNKNOWN, split count: 5, time: 2.32
Output dim: 1, lower bound: -0.0003442, upper bound: 0.0003427
RS_RSZ2_RSZ1_RSZ2_RSZ1_RSZ2, status: Status.UNKNOWN, split count: 5, time: 2.32
Output dim: 1, lower bound: -0.0003441, upper bound: 0.0003428
RS_RSZ2_RSZ1_RSZ2_RSZ2_RSZ1, status: Status.UNKNOWN, split count: 5, time: 2.32
Output dim: 1, lower bound: -0.0003433, upper bound: 0.0003434
RS_RSZ2_RSZ1_RSZ2_RSZ2_RSZ2, status: Status.UNKNOWN, split count: 5, time: 2.32
Output dim: 1, lower bound: -0.0003433, upper bound: 0.0003436
RS_RSZ2_RSZ2_RSZ1_RSZ1_RSZ1, status: Status.UNKNOWN, split count: 5, time: 2.32
Output dim: 1, lower bound: -0.0003436, upper bound: 0.0003433
RS_RSZ2_RSZ2_RSZ1_RSZ1_RSZ2, status: Status.UNKNOWN, split count: 5, time: 2.32
Output dim: 1, lower bound: -0.0003434, upper bound: 0.0003433
RS_RSZ2_RSZ2_RSZ1_RSZ2_RSZ1, status: Status.UNKNOWN, split count: 5, time: 2.32
Output dim: 1, lower bound: -0.0003428, upper bound: 0.0003441
RS_RSZ2_RSZ2_RSZ1_RSZ2_RSZ2, status: Status.UNKNOWN, split count: 5, time: 2.32
Output dim: 1, lower bound: -0.0003427, upper bound: 0.0003442
RS_RSZ2_RSZ2_RSZ2_RSZ1_RSZ1, status: Status.UNKNOWN, split count: 5, time: 2.32
Output dim: 1, lower bound: -0.0003442, upper bound: 0.0003427
RS_RSZ2_RSZ2_RSZ2_RSZ1_RSZ2, status: Status.UNKNOWN, split count: 5, time: 2.32
Output dim: 1, lower bound: -0.0003441, upper bound: 0.0003427
RS_RSZ2_RSZ2_RSZ2_RSZ2_RSZ1, status: Status.UNKNOWN, split count: 5, time: 2.32
Output dim: 1, lower bound: -0.0003434, upper bound: 0.0003433
RS_RSZ2_RSZ2_RSZ2_RSZ2_RSZ2, status: Status.UNKNOWN, split count: 5, time: 2.32
Output dim: 1, lower bound: -0.0003434, upper bound: 0.0003435

## BFS RS instance: RS_RSZ1_RSZ1_RSZ1_RSZ1_RSZ1

### Backsubstitution after applying RS history:
0: 0.0009457, 0.0011003, 0.0009457, 0.0011003, -0.0001502, 0.0001503
1: 0.9936197, 0.9940771, 0.9936197, 0.9940771, -0.0004041, 0.0004037
2: -0.0065606, -0.0050208, -0.0065606, -0.0050208, -0.0012809, 0.0012771
3: 0.0038004, 0.0040140, 0.0038004, 0.0040140, -0.0001765, 0.0001768
4: 0.0023852, 0.0036022, 0.0023852, 0.0036022, -0.0010964, 0.0010994
5: 0.0060086, 0.0065111, 0.0060086, 0.0065111, -0.0005024, 0.0005024
6: -0.0013983, -0.0008639, -0.0013983, -0.0008639, -0.0004395, 0.0004382
7: -0.0083221, -0.0079387, -0.0083221, -0.0079387, -0.0003835, 0.0003835
8: 0.0048484, 0.0068715, 0.0048484, 0.0068715, -0.0014384, 0.0014434
9: -0.0036877, -0.0032079, -0.0036877, -0.0032079, -0.0004797, 0.0004797

### Unstable ReLU Count (Linear/Conv2D Layers)
- layer_idx=0, type=LayerType.Linear, total=256, inp1_unstable=5, inp2_unstable=5, delta_unstable=251
- layer_idx=2, type=LayerType.Linear, total=256, inp1_unstable=4, inp2_unstable=4, delta_unstable=256
- layer_idx=4, type=LayerType.Linear, total=256, inp1_unstable=0, inp2_unstable=0, delta_unstable=256
- layer_idx=6, type=LayerType.Linear, total=256, inp1_unstable=1, inp2_unstable=1, delta_unstable=256
- layer_idx=8, type=LayerType.Linear, total=10, inp1_unstable=0, inp2_unstable=0, delta_unstable=10

Time for backsubstitution: 1.10 seconds

### RS candidates at layer 3
type: RSZ, layer: 3, pos: 73
type: RSZ, layer: 3, pos: 80
type: RSZ, layer: 3, pos: 81
type: RSZ, layer: 3, pos: 136

Time for candidate selection: 0.21 seconds

### Candidate
type: RSZ, layer: 3, pos: 73

### Relational analysis RSZ of RS_RSZ1_RSZ1_RSZ1_RSZ1_RSZ1_RSZ1

#### Relational analysis RSZ result of RS_RSZ1_RSZ1_RSZ1_RSZ1_RSZ1_RSZ1
Status: Status.UNKNOWN
Output dim: 1, lower bound: -0.0003020, upper bound: 0.0002976
time: 0.57 seconds

### Relational analysis RSZ of RS_RSZ1_RSZ1_RSZ1_RSZ1_RSZ1_RSZ2

#### Relational analysis RSZ result of RS_RSZ1_RSZ1_RSZ1_RSZ1_RSZ1_RSZ2
Status: Status.UNKNOWN
Output dim: 1, lower bound: -0.0002977, upper bound: 0.0003021
time: 0.58 seconds

## BFS RS instance: RS_RSZ1_RSZ1_RSZ1_RSZ1_RSZ2

### Backsubstitution after applying RS history:
0: 0.0009457, 0.0011003, 0.0009457, 0.0011003, -0.0001502, 0.0001502
1: 0.9936197, 0.9940771, 0.9936197, 0.9940771, -0.0004038, 0.0004039
2: -0.0065606, -0.0050208, -0.0065606, -0.0050208, -0.0012780, 0.0012790
3: 0.0038004, 0.0040140, 0.0038004, 0.0040140, -0.0001767, 0.0001766
4: 0.0023852, 0.0036022, 0.0023852, 0.0036022, -0.0010978, 0.0010971
5: 0.0060086, 0.0065111, 0.0060086, 0.0065111, -0.0005024, 0.0005024
6: -0.0013983, -0.0008639, -0.0013983, -0.0008639, -0.0004385, 0.0004388
7: -0.0083221, -0.0079387, -0.0083221, -0.0079387, -0.0003835, 0.0003835
8: 0.0048484, 0.0068715, 0.0048484, 0.0068715, -0.0014408, 0.0014395
9: -0.0036877, -0.0032079, -0.0036877, -0.0032079, -0.0004797, 0.0004797

### Unstable ReLU Count (Linear/Conv2D Layers)
- layer_idx=0, type=LayerType.Linear, total=256, inp1_unstable=5, inp2_unstable=5, delta_unstable=251
- layer_idx=2, type=LayerType.Linear, total=256, inp1_unstable=4, inp2_unstable=4, delta_unstable=256
- layer_idx=4, type=LayerType.Linear, total=256, inp1_unstable=0, inp2_unstable=0, delta_unstable=256
- layer_idx=6, type=LayerType.Linear, total=256, inp1_unstable=1, inp2_unstable=1, delta_unstable=256
- layer_idx=8, type=LayerType.Linear, total=10, inp1_unstable=0, inp2_unstable=0, delta_unstable=10

Time for backsubstitution: 1.11 seconds

### RS candidates at layer 3
type: RSZ, layer: 3, pos: 73
type: RSZ, layer: 3, pos: 80
type: RSZ, layer: 3, pos: 81
type: RSZ, layer: 3, pos: 136

Time for candidate selection: 0.22 seconds

### Candidate
type: RSZ, layer: 3, pos: 73

### Relational analysis RSZ of RS_RSZ1_RSZ1_RSZ1_RSZ1_RSZ2_RSZ1

#### Relational analysis RSZ result of RS_RSZ1_RSZ1_RSZ1_RSZ1_RSZ2_RSZ1
Status: Status.UNKNOWN
Output dim: 1, lower bound: -0.0003019, upper bound: 0.0002976
time: 0.56 seconds

### Relational analysis RSZ of RS_RSZ1_RSZ1_RSZ1_RSZ1_RSZ2_RSZ2

#### Relational analysis RSZ result of RS_RSZ1_RSZ1_RSZ1_RSZ1_RSZ2_RSZ2
Status: Status.UNKNOWN
Output dim: 1, lower bound: -0.0002975, upper bound: 0.0003021
time: 0.59 seconds

## BFS RS instance: RS_RSZ1_RSZ1_RSZ1_RSZ2_RSZ1

### Backsubstitution after applying RS history:
0: 0.0009457, 0.0011003, 0.0009457, 0.0011003, -0.0001502, 0.0001503
1: 0.9936197, 0.9940771, 0.9936197, 0.9940771, -0.0004042, 0.0004037
2: -0.0065606, -0.0050208, -0.0065606, -0.0050208, -0.0012826, 0.0012776
3: 0.0038004, 0.0040140, 0.0038004, 0.0040140, -0.0001766, 0.0001769
4: 0.0023852, 0.0036022, 0.0023852, 0.0036022, -0.0010967, 0.0011007
5: 0.0060086, 0.0065111, 0.0060086, 0.0065111, -0.0005024, 0.0005024
6: -0.0013983, -0.0008639, -0.0013983, -0.0008639, -0.0004401, 0.0004384
7: -0.0083221, -0.0079387, -0.0083221, -0.0079387, -0.0003835, 0.0003835
8: 0.0048484, 0.0068715, 0.0048484, 0.0068715, -0.0014390, 0.0014456
9: -0.0036877, -0.0032079, -0.0036877, -0.0032079, -0.0004797, 0.0004797

### Unstable ReLU Count (Linear/Conv2D Layers)
- layer_idx=0, type=LayerType.Linear, total=256, inp1_unstable=5, inp2_unstable=5, delta_unstable=251
- layer_idx=2, type=LayerType.Linear, total=256, inp1_unstable=4, inp2_unstable=4, delta_unstable=256
- layer_idx=4, type=LayerType.Linear, total=256, inp1_unstable=0, inp2_unstable=0, delta_unstable=256
- layer_idx=6, type=LayerType.Linear, total=256, inp1_unstable=1, inp2_unstable=1, delta_unstable=256
- layer_idx=8, type=LayerType.Linear, total=10, inp1_unstable=0, inp2_unstable=0, delta_unstable=10

Time for backsubstitution: 1.26 seconds

### RS candidates at layer 3
type: RSZ, layer: 3, pos: 73
type: RSZ, layer: 3, pos: 80
type: RSZ, layer: 3, pos: 81
type: RSZ, layer: 3, pos: 136

Time for candidate selection: 0.28 seconds

### Candidate
type: RSZ, layer: 3, pos: 73

### Relational analysis RSZ of RS_RSZ1_RSZ1_RSZ1_RSZ2_RSZ1_RSZ1

#### Relational analysis RSZ result of RS_RSZ1_RSZ1_RSZ1_RSZ2_RSZ1_RSZ1
Status: Status.UNKNOWN
Output dim: 1, lower bound: -0.0003011, upper bound: 0.0002983
time: 0.62 seconds

### Relational analysis RSZ of RS_RSZ1_RSZ1_RSZ1_RSZ2_RSZ1_RSZ2

#### Relational analysis RSZ result of RS_RSZ1_RSZ1_RSZ1_RSZ2_RSZ1_RSZ2
Status: Status.UNKNOWN
Output dim: 1, lower bound: -0.0002967, upper bound: 0.0003029
time: 0.60 seconds

## BFS RS instance: RS_RSZ1_RSZ1_RSZ1_RSZ2_RSZ2

### Backsubstitution after applying RS history:
0: 0.0009457, 0.0011003, 0.0009457, 0.0011003, -0.0001502, 0.0001503
1: 0.9936197, 0.9940771, 0.9936197, 0.9940771, -0.0004041, 0.0004040
2: -0.0065606, -0.0050208, -0.0065606, -0.0050208, -0.0012810, 0.0012801
3: 0.0038004, 0.0040140, 0.0038004, 0.0040140, -0.0001767, 0.0001768
4: 0.0023852, 0.0036022, 0.0023852, 0.0036022, -0.0010987, 0.0010995
5: 0.0060086, 0.0065111, 0.0060086, 0.0065111, -0.0005024, 0.0005024
6: -0.0013983, -0.0008639, -0.0013983, -0.0008639, -0.0004396, 0.0004392
7: -0.0083221, -0.0079387, -0.0083221, -0.0079387, -0.0003835, 0.0003835
8: 0.0048484, 0.0068715, 0.0048484, 0.0068715, -0.0014423, 0.0014435
9: -0.0036877, -0.0032079, -0.0036877, -0.0032079, -0.0004797, 0.0004797

### Unstable ReLU Count (Linear/Conv2D Layers)
- layer_idx=0, type=LayerType.Linear, total=256, inp1_unstable=5, inp2_unstable=5, delta_unstable=251
- layer_idx=2, type=LayerType.Linear, total=256, inp1_unstable=4, inp2_unstable=4, delta_unstable=256
- layer_idx=4, type=LayerType.Linear, total=256, inp1_unstable=0, inp2_unstable=0, delta_unstable=256
- layer_idx=6, type=LayerType.Linear, total=256, inp1_unstable=1, inp2_unstable=1, delta_unstable=256
- layer_idx=8, type=LayerType.Linear, total=10, inp1_unstable=0, inp2_unstable=0, delta_unstable=10

Time for backsubstitution: 1.30 seconds

### RS candidates at layer 3
type: RSZ, layer: 3, pos: 73
type: RSZ, layer: 3, pos: 80
type: RSZ, layer: 3, pos: 81
type: RSZ, layer: 3, pos: 136

Time for candidate selection: 0.28 seconds

### Candidate
type: RSZ, layer: 3, pos: 73

### Relational analysis RSZ of RS_RSZ1_RSZ1_RSZ1_RSZ2_RSZ2_RSZ1

#### Relational analysis RSZ result of RS_RSZ1_RSZ1_RSZ1_RSZ2_RSZ2_RSZ1
Status: Status.UNKNOWN
Output dim: 1, lower bound: -0.0003011, upper bound: 0.0002983
time: 0.72 seconds

### Relational analysis RSZ of RS_RSZ1_RSZ1_RSZ1_RSZ2_RSZ2_RSZ2

#### Relational analysis RSZ result of RS_RSZ1_RSZ1_RSZ1_RSZ2_RSZ2_RSZ2
Status: Status.UNKNOWN
Output dim: 1, lower bound: -0.0002967, upper bound: 0.0003029
time: 0.60 seconds

## BFS RS instance: RS_RSZ1_RSZ1_RSZ2_RSZ1_RSZ1

### Backsubstitution after applying RS history:
0: 0.0009457, 0.0011003, 0.0009457, 0.0011003, -0.0001502, 0.0001503
1: 0.9936197, 0.9940771, 0.9936197, 0.9940771, -0.0004044, 0.0004036
2: -0.0065606, -0.0050208, -0.0065606, -0.0050208, -0.0012847, 0.0012764
3: 0.0038004, 0.0040140, 0.0038004, 0.0040140, -0.0001765, 0.0001770
4: 0.0023852, 0.0036022, 0.0023852, 0.0036022, -0.0010958, 0.0011023
5: 0.0060086, 0.0065111, 0.0060086, 0.0065111, -0.0005024, 0.0005024
6: -0.0013983, -0.0008639, -0.0013983, -0.0008639, -0.0004408, 0.0004379
7: -0.0083221, -0.0079387, -0.0083221, -0.0079387, -0.0003835, 0.0003835
8: 0.0048484, 0.0068715, 0.0048484, 0.0068715, -0.0014374, 0.0014483
9: -0.0036877, -0.0032079, -0.0036877, -0.0032079, -0.0004797, 0.0004797

### Unstable ReLU Count (Linear/Conv2D Layers)
- layer_idx=0, type=LayerType.Linear, total=256, inp1_unstable=5, inp2_unstable=5, delta_unstable=251
- layer_idx=2, type=LayerType.Linear, total=256, inp1_unstable=4, inp2_unstable=4, delta_unstable=256
- layer_idx=4, type=LayerType.Linear, total=256, inp1_unstable=0, inp2_unstable=0, delta_unstable=256
- layer_idx=6, type=LayerType.Linear, total=256, inp1_unstable=1, inp2_unstable=1, delta_unstable=256
- layer_idx=8, type=LayerType.Linear, total=10, inp1_unstable=0, inp2_unstable=0, delta_unstable=10

Time for backsubstitution: 1.23 seconds

### RS candidates at layer 3
type: RSZ, layer: 3, pos: 73
type: RSZ, layer: 3, pos: 80
type: RSZ, layer: 3, pos: 81
type: RSZ, layer: 3, pos: 136

Time for candidate selection: 0.24 seconds

### Candidate
type: RSZ, layer: 3, pos: 73

### Relational analysis RSZ of RS_RSZ1_RSZ1_RSZ2_RSZ1_RSZ1_RSZ1

#### Relational analysis RSZ result of RS_RSZ1_RSZ1_RSZ2_RSZ1_RSZ1_RSZ1
Status: Status.UNKNOWN
Output dim: 1, lower bound: -0.0003029, upper bound: 0.0002968
time: 0.55 seconds

### Relational analysis RSZ of RS_RSZ1_RSZ1_RSZ2_RSZ1_RSZ1_RSZ2

#### Relational analysis RSZ result of RS_RSZ1_RSZ1_RSZ2_RSZ1_RSZ1_RSZ2
Status: Status.UNKNOWN
Output dim: 1, lower bound: -0.0002982, upper bound: 0.0003013
time: 0.57 seconds

## BFS RS instance: RS_RSZ1_RSZ1_RSZ2_RSZ1_RSZ2

### Backsubstitution after applying RS history:
0: 0.0009457, 0.0011003, 0.0009457, 0.0011003, -0.0001502, 0.0001503
1: 0.9936197, 0.9940771, 0.9936197, 0.9940771, -0.0004041, 0.0004038
2: -0.0065606, -0.0050208, -0.0065606, -0.0050208, -0.0012817, 0.0012779
3: 0.0038004, 0.0040140, 0.0038004, 0.0040140, -0.0001766, 0.0001769
4: 0.0023852, 0.0036022, 0.0023852, 0.0036022, -0.0010969, 0.0011000
5: 0.0060086, 0.0065111, 0.0060086, 0.0065111, -0.0005024, 0.0005024
6: -0.0013983, -0.0008639, -0.0013983, -0.0008639, -0.0004398, 0.0004385
7: -0.0083221, -0.0079387, -0.0083221, -0.0079387, -0.0003835, 0.0003835
8: 0.0048484, 0.0068715, 0.0048484, 0.0068715, -0.0014393, 0.0014443
9: -0.0036877, -0.0032079, -0.0036877, -0.0032079, -0.0004797, 0.0004797

### Unstable ReLU Count (Linear/Conv2D Layers)
- layer_idx=0, type=LayerType.Linear, total=256, inp1_unstable=5, inp2_unstable=5, delta_unstable=251
- layer_idx=2, type=LayerType.Linear, total=256, inp1_unstable=4, inp2_unstable=4, delta_unstable=256
- layer_idx=4, type=LayerType.Linear, total=256, inp1_unstable=0, inp2_unstable=0, delta_unstable=256
- layer_idx=6, type=LayerType.Linear, total=256, inp1_unstable=1, inp2_unstable=1, delta_unstable=256
- layer_idx=8, type=LayerType.Linear, total=10, inp1_unstable=0, inp2_unstable=0, delta_unstable=10

Time for backsubstitution: 1.14 seconds

### RS candidates at layer 3
type: RSZ, layer: 3, pos: 73
type: RSZ, layer: 3, pos: 80
type: RSZ, layer: 3, pos: 81
type: RSZ, layer: 3, pos: 136

Time for candidate selection: 0.24 seconds

### Candidate
type: RSZ, layer: 3, pos: 73

### Relational analysis RSZ of RS_RSZ1_RSZ1_RSZ2_RSZ1_RSZ2_RSZ1

#### Relational analysis RSZ result of RS_RSZ1_RSZ1_RSZ2_RSZ1_RSZ2_RSZ1
Status: Status.UNKNOWN
Output dim: 1, lower bound: -0.0003029, upper bound: 0.0002968
time: 0.59 seconds

### Relational analysis RSZ of RS_RSZ1_RSZ1_RSZ2_RSZ1_RSZ2_RSZ2

#### Relational analysis RSZ result of RS_RSZ1_RSZ1_RSZ2_RSZ1_RSZ2_RSZ2
Status: Status.UNKNOWN
Output dim: 1, lower bound: -0.0002982, upper bound: 0.0003013
time: 0.59 seconds

## BFS RS instance: RS_RSZ1_RSZ1_RSZ2_RSZ2_RSZ1

### Backsubstitution after applying RS history:
0: 0.0009457, 0.0011003, 0.0009457, 0.0011003, -0.0001502, 0.0001503
1: 0.9936197, 0.9940771, 0.9936197, 0.9940771, -0.0004042, 0.0004032
2: -0.0065606, -0.0050208, -0.0065606, -0.0050208, -0.0012827, 0.0012722
3: 0.0038004, 0.0040140, 0.0038004, 0.0040140, -0.0001762, 0.0001769
4: 0.0023852, 0.0036022, 0.0023852, 0.0036022, -0.0010925, 0.0011007
5: 0.0060086, 0.0065111, 0.0060086, 0.0065111, -0.0005024, 0.0005024
6: -0.0013983, -0.0008639, -0.0013983, -0.0008639, -0.0004401, 0.0004365
7: -0.0083221, -0.0079387, -0.0083221, -0.0079387, -0.0003835, 0.0003835
8: 0.0048484, 0.0068715, 0.0048484, 0.0068715, -0.0014319, 0.0014456
9: -0.0036877, -0.0032079, -0.0036877, -0.0032079, -0.0004797, 0.0004797

### Unstable ReLU Count (Linear/Conv2D Layers)
- layer_idx=0, type=LayerType.Linear, total=256, inp1_unstable=5, inp2_unstable=5, delta_unstable=251
- layer_idx=2, type=LayerType.Linear, total=256, inp1_unstable=4, inp2_unstable=4, delta_unstable=256
- layer_idx=4, type=LayerType.Linear, total=256, inp1_unstable=0, inp2_unstable=0, delta_unstable=256
- layer_idx=6, type=LayerType.Linear, total=256, inp1_unstable=1, inp2_unstable=1, delta_unstable=256
- layer_idx=8, type=LayerType.Linear, total=10, inp1_unstable=0, inp2_unstable=0, delta_unstable=10

Time for backsubstitution: 1.13 seconds

### RS candidates at layer 3
type: RSZ, layer: 3, pos: 73
type: RSZ, layer: 3, pos: 80
type: RSZ, layer: 3, pos: 81
type: RSZ, layer: 3, pos: 136

Time for candidate selection: 0.21 seconds

### Candidate
type: RSZ, layer: 3, pos: 73

### Relational analysis RSZ of RS_RSZ1_RSZ1_RSZ2_RSZ2_RSZ1_RSZ1

#### Relational analysis RSZ result of RS_RSZ1_RSZ1_RSZ2_RSZ2_RSZ1_RSZ1
Status: Status.UNKNOWN
Output dim: 1, lower bound: -0.0003018, upper bound: 0.0002975
time: 0.59 seconds

### Relational analysis RSZ of RS_RSZ1_RSZ1_RSZ2_RSZ2_RSZ1_RSZ2

#### Relational analysis RSZ result of RS_RSZ1_RSZ1_RSZ2_RSZ2_RSZ1_RSZ2
Status: Status.UNKNOWN
Output dim: 1, lower bound: -0.0002972, upper bound: 0.0003019
time: 0.56 seconds

## BFS RS instance: RS_RSZ1_RSZ1_RSZ2_RSZ2_RSZ2

### Backsubstitution after applying RS history:
0: 0.0009457, 0.0011003, 0.0009457, 0.0011003, -0.0001502, 0.0001503
1: 0.9936197, 0.9940771, 0.9936197, 0.9940771, -0.0004041, 0.0004035
2: -0.0065606, -0.0050208, -0.0065606, -0.0050208, -0.0012814, 0.0012746
3: 0.0038004, 0.0040140, 0.0038004, 0.0040140, -0.0001764, 0.0001768
4: 0.0023852, 0.0036022, 0.0023852, 0.0036022, -0.0010944, 0.0010997
5: 0.0060086, 0.0065111, 0.0060086, 0.0065111, -0.0005024, 0.0005024
6: -0.0013983, -0.0008639, -0.0013983, -0.0008639, -0.0004397, 0.0004373
7: -0.0083221, -0.0079387, -0.0083221, -0.0079387, -0.0003835, 0.0003835
8: 0.0048484, 0.0068715, 0.0048484, 0.0068715, -0.0014350, 0.0014440
9: -0.0036877, -0.0032079, -0.0036877, -0.0032079, -0.0004797, 0.0004797

### Unstable ReLU Count (Linear/Conv2D Layers)
- layer_idx=0, type=LayerType.Linear, total=256, inp1_unstable=5, inp2_unstable=5, delta_unstable=251
- layer_idx=2, type=LayerType.Linear, total=256, inp1_unstable=4, inp2_unstable=4, delta_unstable=256
- layer_idx=4, type=LayerType.Linear, total=256, inp1_unstable=0, inp2_unstable=0, delta_unstable=256
- layer_idx=6, type=LayerType.Linear, total=256, inp1_unstable=1, inp2_unstable=1, delta_unstable=256
- layer_idx=8, type=LayerType.Linear, total=10, inp1_unstable=0, inp2_unstable=0, delta_unstable=10

Time for backsubstitution: 1.10 seconds

### RS candidates at layer 3
type: RSZ, layer: 3, pos: 73
type: RSZ, layer: 3, pos: 80
type: RSZ, layer: 3, pos: 81
type: RSZ, layer: 3, pos: 136

Time for candidate selection: 0.23 seconds

### Candidate
type: RSZ, layer: 3, pos: 73

### Relational analysis RSZ of RS_RSZ1_RSZ1_RSZ2_RSZ2_RSZ2_RSZ1

#### Relational analysis RSZ result of RS_RSZ1_RSZ1_RSZ2_RSZ2_RSZ2_RSZ1
Status: Status.UNKNOWN
Output dim: 1, lower bound: -0.0003018, upper bound: 0.0002977
time: 0.61 seconds

### Relational analysis RSZ of RS_RSZ1_RSZ1_RSZ2_RSZ2_RSZ2_RSZ2

#### Relational analysis RSZ result of RS_RSZ1_RSZ1_RSZ2_RSZ2_RSZ2_RSZ2
Status: Status.UNKNOWN
Output dim: 1, lower bound: -0.0002972, upper bound: 0.0003020
time: 0.59 seconds

## BFS RS instance: RS_RSZ1_RSZ2_RSZ1_RSZ1_RSZ1

### Backsubstitution after applying RS history:
0: 0.0009457, 0.0011003, 0.0009457, 0.0011003, -0.0001502, 0.0001504
1: 0.9936197, 0.9940771, 0.9936197, 0.9940771, -0.0004053, 0.0004035
2: -0.0065606, -0.0050208, -0.0065606, -0.0050208, -0.0012923, 0.0012745
3: 0.0038004, 0.0040140, 0.0038004, 0.0040140, -0.0001764, 0.0001776
4: 0.0023852, 0.0036022, 0.0023852, 0.0036022, -0.0010943, 0.0011081
5: 0.0060086, 0.0065111, 0.0060086, 0.0065111, -0.0005024, 0.0005024
6: -0.0013983, -0.0008639, -0.0013983, -0.0008639, -0.0004435, 0.0004373
7: -0.0083221, -0.0079387, -0.0083221, -0.0079387, -0.0003835, 0.0003835
8: 0.0048484, 0.0068715, 0.0048484, 0.0068715, -0.0014350, 0.0014593
9: -0.0036877, -0.0032079, -0.0036877, -0.0032079, -0.0004797, 0.0004797

### Unstable ReLU Count (Linear/Conv2D Layers)
- layer_idx=0, type=LayerType.Linear, total=256, inp1_unstable=5, inp2_unstable=5, delta_unstable=251
- layer_idx=2, type=LayerType.Linear, total=256, inp1_unstable=4, inp2_unstable=4, delta_unstable=256
- layer_idx=4, type=LayerType.Linear, total=256, inp1_unstable=0, inp2_unstable=0, delta_unstable=256
- layer_idx=6, type=LayerType.Linear, total=256, inp1_unstable=1, inp2_unstable=1, delta_unstable=256
- layer_idx=8, type=LayerType.Linear, total=10, inp1_unstable=0, inp2_unstable=0, delta_unstable=10

Time for backsubstitution: 1.17 seconds

### RS candidates at layer 3
type: RSZ, layer: 3, pos: 73
type: RSZ, layer: 3, pos: 80
type: RSZ, layer: 3, pos: 81
type: RSZ, layer: 3, pos: 136

Time for candidate selection: 0.25 seconds

### Candidate
type: RSZ, layer: 3, pos: 73

### Relational analysis RSZ of RS_RSZ1_RSZ2_RSZ1_RSZ1_RSZ1_RSZ1

#### Relational analysis RSZ result of RS_RSZ1_RSZ2_RSZ1_RSZ1_RSZ1_RSZ1
Status: Status.UNKNOWN
Output dim: 1, lower bound: -0.0003020, upper bound: 0.0002972
time: 0.54 seconds

### Relational analysis RSZ of RS_RSZ1_RSZ2_RSZ1_RSZ1_RSZ1_RSZ2

#### Relational analysis RSZ result of RS_RSZ1_RSZ2_RSZ1_RSZ1_RSZ1_RSZ2
Status: Status.UNKNOWN
Output dim: 1, lower bound: -0.0002977, upper bound: 0.0003018
time: 0.57 seconds

## BFS RS instance: RS_RSZ1_RSZ2_RSZ1_RSZ1_RSZ2

### Backsubstitution after applying RS history:
0: 0.0009457, 0.0011003, 0.0009457, 0.0011003, -0.0001502, 0.0001503
1: 0.9936197, 0.9940771, 0.9936197, 0.9940771, -0.0004050, 0.0004036
2: -0.0065606, -0.0050208, -0.0065606, -0.0050208, -0.0012893, 0.0012761
3: 0.0038004, 0.0040140, 0.0038004, 0.0040140, -0.0001765, 0.0001774
4: 0.0023852, 0.0036022, 0.0023852, 0.0036022, -0.0010955, 0.0011058
5: 0.0060086, 0.0065111, 0.0060086, 0.0065111, -0.0005024, 0.0005024
6: -0.0013983, -0.0008639, -0.0013983, -0.0008639, -0.0004425, 0.0004378
7: -0.0083221, -0.0079387, -0.0083221, -0.0079387, -0.0003835, 0.0003835
8: 0.0048484, 0.0068715, 0.0048484, 0.0068715, -0.0014370, 0.0014554
9: -0.0036877, -0.0032079, -0.0036877, -0.0032079, -0.0004797, 0.0004797

### Unstable ReLU Count (Linear/Conv2D Layers)
- layer_idx=0, type=LayerType.Linear, total=256, inp1_unstable=5, inp2_unstable=5, delta_unstable=251
- layer_idx=2, type=LayerType.Linear, total=256, inp1_unstable=4, inp2_unstable=4, delta_unstable=256
- layer_idx=4, type=LayerType.Linear, total=256, inp1_unstable=0, inp2_unstable=0, delta_unstable=256
- layer_idx=6, type=LayerType.Linear, total=256, inp1_unstable=1, inp2_unstable=1, delta_unstable=256
- layer_idx=8, type=LayerType.Linear, total=10, inp1_unstable=0, inp2_unstable=0, delta_unstable=10

Time for backsubstitution: 1.13 seconds

### RS candidates at layer 3
type: RSZ, layer: 3, pos: 73
type: RSZ, layer: 3, pos: 80
type: RSZ, layer: 3, pos: 81
type: RSZ, layer: 3, pos: 136

Time for candidate selection: 0.21 seconds

### Candidate
type: RSZ, layer: 3, pos: 73

### Relational analysis RSZ of RS_RSZ1_RSZ2_RSZ1_RSZ1_RSZ2_RSZ1

#### Relational analysis RSZ result of RS_RSZ1_RSZ2_RSZ1_RSZ1_RSZ2_RSZ1
Status: Status.UNKNOWN
Output dim: 1, lower bound: -0.0003019, upper bound: 0.0002972
time: 0.57 seconds

### Relational analysis RSZ of RS_RSZ1_RSZ2_RSZ1_RSZ1_RSZ2_RSZ2

#### Relational analysis RSZ result of RS_RSZ1_RSZ2_RSZ1_RSZ1_RSZ2_RSZ2
Status: Status.UNKNOWN
Output dim: 1, lower bound: -0.0002975, upper bound: 0.0003018
time: 0.57 seconds

## BFS RS instance: RS_RSZ1_RSZ2_RSZ1_RSZ2_RSZ1

### Backsubstitution after applying RS history:
0: 0.0009457, 0.0011003, 0.0009457, 0.0011003, -0.0001502, 0.0001504
1: 0.9936197, 0.9940771, 0.9936197, 0.9940771, -0.0004054, 0.0004037
2: -0.0065606, -0.0050208, -0.0065606, -0.0050208, -0.0012940, 0.0012768
3: 0.0038004, 0.0040140, 0.0038004, 0.0040140, -0.0001765, 0.0001777
4: 0.0023852, 0.0036022, 0.0023852, 0.0036022, -0.0010961, 0.0011094
5: 0.0060086, 0.0065111, 0.0060086, 0.0065111, -0.0005024, 0.0005024
6: -0.0013983, -0.0008639, -0.0013983, -0.0008639, -0.0004441, 0.0004381
7: -0.0083221, -0.0079387, -0.0083221, -0.0079387, -0.0003835, 0.0003835
8: 0.0048484, 0.0068715, 0.0048484, 0.0068715, -0.0014379, 0.0014615
9: -0.0036877, -0.0032079, -0.0036877, -0.0032079, -0.0004797, 0.0004797

### Unstable ReLU Count (Linear/Conv2D Layers)
- layer_idx=0, type=LayerType.Linear, total=256, inp1_unstable=5, inp2_unstable=5, delta_unstable=251
- layer_idx=2, type=LayerType.Linear, total=256, inp1_unstable=4, inp2_unstable=4, delta_unstable=256
- layer_idx=4, type=LayerType.Linear, total=256, inp1_unstable=0, inp2_unstable=0, delta_unstable=256
- layer_idx=6, type=LayerType.Linear, total=256, inp1_unstable=1, inp2_unstable=1, delta_unstable=256
- layer_idx=8, type=LayerType.Linear, total=10, inp1_unstable=0, inp2_unstable=0, delta_unstable=10

Time for backsubstitution: 1.14 seconds

### RS candidates at layer 3
type: RSZ, layer: 3, pos: 73
type: RSZ, layer: 3, pos: 80
type: RSZ, layer: 3, pos: 81
type: RSZ, layer: 3, pos: 136

Time for candidate selection: 0.23 seconds

### Candidate
type: RSZ, layer: 3, pos: 73

### Relational analysis RSZ of RS_RSZ1_RSZ2_RSZ1_RSZ2_RSZ1_RSZ1

#### Relational analysis RSZ result of RS_RSZ1_RSZ2_RSZ1_RSZ2_RSZ1_RSZ1
Status: Status.UNKNOWN
Output dim: 1, lower bound: -0.0003013, upper bound: 0.0002982
time: 0.56 seconds

### Relational analysis RSZ of RS_RSZ1_RSZ2_RSZ1_RSZ2_RSZ1_RSZ2

#### Relational analysis RSZ result of RS_RSZ1_RSZ2_RSZ1_RSZ2_RSZ1_RSZ2
Status: Status.UNKNOWN
Output dim: 1, lower bound: -0.0002968, upper bound: 0.0003029
time: 0.57 seconds

## BFS RS instance: RS_RSZ1_RSZ2_RSZ1_RSZ2_RSZ2

### Backsubstitution after applying RS history:
0: 0.0009457, 0.0011003, 0.0009457, 0.0011003, -0.0001502, 0.0001504
1: 0.9936197, 0.9940771, 0.9936197, 0.9940771, -0.0004053, 0.0004039
2: -0.0065606, -0.0050208, -0.0065606, -0.0050208, -0.0012924, 0.0012790
3: 0.0038004, 0.0040140, 0.0038004, 0.0040140, -0.0001767, 0.0001776
4: 0.0023852, 0.0036022, 0.0023852, 0.0036022, -0.0010978, 0.0011082
5: 0.0060086, 0.0065111, 0.0060086, 0.0065111, -0.0005024, 0.0005024
6: -0.0013983, -0.0008639, -0.0013983, -0.0008639, -0.0004436, 0.0004388
7: -0.0083221, -0.0079387, -0.0083221, -0.0079387, -0.0003835, 0.0003835
8: 0.0048484, 0.0068715, 0.0048484, 0.0068715, -0.0014408, 0.0014594
9: -0.0036877, -0.0032079, -0.0036877, -0.0032079, -0.0004797, 0.0004797

### Unstable ReLU Count (Linear/Conv2D Layers)
- layer_idx=0, type=LayerType.Linear, total=256, inp1_unstable=5, inp2_unstable=5, delta_unstable=251
- layer_idx=2, type=LayerType.Linear, total=256, inp1_unstable=4, inp2_unstable=4, delta_unstable=256
- layer_idx=4, type=LayerType.Linear, total=256, inp1_unstable=0, inp2_unstable=0, delta_unstable=256
- layer_idx=6, type=LayerType.Linear, total=256, inp1_unstable=1, inp2_unstable=1, delta_unstable=256
- layer_idx=8, type=LayerType.Linear, total=10, inp1_unstable=0, inp2_unstable=0, delta_unstable=10

Time for backsubstitution: 1.14 seconds

### RS candidates at layer 3
type: RSZ, layer: 3, pos: 73
type: RSZ, layer: 3, pos: 80
type: RSZ, layer: 3, pos: 81
type: RSZ, layer: 3, pos: 136

Time for candidate selection: 0.22 seconds

### Candidate
type: RSZ, layer: 3, pos: 73

### Relational analysis RSZ of RS_RSZ1_RSZ2_RSZ1_RSZ2_RSZ2_RSZ1

#### Relational analysis RSZ result of RS_RSZ1_RSZ2_RSZ1_RSZ2_RSZ2_RSZ1
Status: Status.UNKNOWN
Output dim: 1, lower bound: -0.0003013, upper bound: 0.0002982
time: 0.56 seconds

### Relational analysis RSZ of RS_RSZ1_RSZ2_RSZ1_RSZ2_RSZ2_RSZ2

#### Relational analysis RSZ result of RS_RSZ1_RSZ2_RSZ1_RSZ2_RSZ2_RSZ2
Status: Status.UNKNOWN
Output dim: 1, lower bound: -0.0002968, upper bound: 0.0003029
time: 0.58 seconds

## BFS RS instance: RS_RSZ1_RSZ2_RSZ2_RSZ1_RSZ1

### Backsubstitution after applying RS history:
0: 0.0009457, 0.0011003, 0.0009457, 0.0011003, -0.0001502, 0.0001504
1: 0.9936197, 0.9940771, 0.9936197, 0.9940771, -0.0004056, 0.0004034
2: -0.0065606, -0.0050208, -0.0065606, -0.0050208, -0.0012960, 0.0012738
3: 0.0038004, 0.0040140, 0.0038004, 0.0040140, -0.0001763, 0.0001779
4: 0.0023852, 0.0036022, 0.0023852, 0.0036022, -0.0010938, 0.0011110
5: 0.0060086, 0.0065111, 0.0060086, 0.0065111, -0.0005024, 0.0005024
6: -0.0013983, -0.0008639, -0.0013983, -0.0008639, -0.0004448, 0.0004371
7: -0.0083221, -0.0079387, -0.0083221, -0.0079387, -0.0003835, 0.0003835
8: 0.0048484, 0.0068715, 0.0048484, 0.0068715, -0.0014340, 0.0014642
9: -0.0036877, -0.0032079, -0.0036877, -0.0032079, -0.0004797, 0.0004797

### Unstable ReLU Count (Linear/Conv2D Layers)
- layer_idx=0, type=LayerType.Linear, total=256, inp1_unstable=5, inp2_unstable=5, delta_unstable=251
- layer_idx=2, type=LayerType.Linear, total=256, inp1_unstable=4, inp2_unstable=4, delta_unstable=256
- layer_idx=4, type=LayerType.Linear, total=256, inp1_unstable=0, inp2_unstable=0, delta_unstable=256
- layer_idx=6, type=LayerType.Linear, total=256, inp1_unstable=1, inp2_unstable=1, delta_unstable=256
- layer_idx=8, type=LayerType.Linear, total=10, inp1_unstable=0, inp2_unstable=0, delta_unstable=10

Time for backsubstitution: 1.17 seconds

### RS candidates at layer 3
type: RSZ, layer: 3, pos: 73
type: RSZ, layer: 3, pos: 80
type: RSZ, layer: 3, pos: 81
type: RSZ, layer: 3, pos: 136

Time for candidate selection: 0.22 seconds

### Candidate
type: RSZ, layer: 3, pos: 73

### Relational analysis RSZ of RS_RSZ1_RSZ2_RSZ2_RSZ1_RSZ1_RSZ1

#### Relational analysis RSZ result of RS_RSZ1_RSZ2_RSZ2_RSZ1_RSZ1_RSZ1
Status: Status.UNKNOWN
Output dim: 1, lower bound: -0.0003029, upper bound: 0.0002967
time: 0.55 seconds

### Relational analysis RSZ of RS_RSZ1_RSZ2_RSZ2_RSZ1_RSZ1_RSZ2

#### Relational analysis RSZ result of RS_RSZ1_RSZ2_RSZ2_RSZ1_RSZ1_RSZ2
Status: Status.UNKNOWN
Output dim: 1, lower bound: -0.0002983, upper bound: 0.0003011
time: 0.55 seconds

## BFS RS instance: RS_RSZ1_RSZ2_RSZ2_RSZ1_RSZ2

### Backsubstitution after applying RS history:
0: 0.0009457, 0.0011003, 0.0009457, 0.0011003, -0.0001502, 0.0001504
1: 0.9936197, 0.9940771, 0.9936197, 0.9940771, -0.0004053, 0.0004035
2: -0.0065606, -0.0050208, -0.0065606, -0.0050208, -0.0012930, 0.0012754
3: 0.0038004, 0.0040140, 0.0038004, 0.0040140, -0.0001764, 0.0001777
4: 0.0023852, 0.0036022, 0.0023852, 0.0036022, -0.0010950, 0.0011087
5: 0.0060086, 0.0065111, 0.0060086, 0.0065111, -0.0005024, 0.0005024
6: -0.0013983, -0.0008639, -0.0013983, -0.0008639, -0.0004438, 0.0004376
7: -0.0083221, -0.0079387, -0.0083221, -0.0079387, -0.0003835, 0.0003835
8: 0.0048484, 0.0068715, 0.0048484, 0.0068715, -0.0014361, 0.0014603
9: -0.0036877, -0.0032079, -0.0036877, -0.0032079, -0.0004797, 0.0004797

### Unstable ReLU Count (Linear/Conv2D Layers)
- layer_idx=0, type=LayerType.Linear, total=256, inp1_unstable=5, inp2_unstable=5, delta_unstable=251
- layer_idx=2, type=LayerType.Linear, total=256, inp1_unstable=4, inp2_unstable=4, delta_unstable=256
- layer_idx=4, type=LayerType.Linear, total=256, inp1_unstable=0, inp2_unstable=0, delta_unstable=256
- layer_idx=6, type=LayerType.Linear, total=256, inp1_unstable=1, inp2_unstable=1, delta_unstable=256
- layer_idx=8, type=LayerType.Linear, total=10, inp1_unstable=0, inp2_unstable=0, delta_unstable=10

Time for backsubstitution: 1.14 seconds

### RS candidates at layer 3
type: RSZ, layer: 3, pos: 73
type: RSZ, layer: 3, pos: 80
type: RSZ, layer: 3, pos: 81
type: RSZ, layer: 3, pos: 136

Time for candidate selection: 0.24 seconds

### Candidate
type: RSZ, layer: 3, pos: 73

### Relational analysis RSZ of RS_RSZ1_RSZ2_RSZ2_RSZ1_RSZ2_RSZ1

#### Relational analysis RSZ result of RS_RSZ1_RSZ2_RSZ2_RSZ1_RSZ2_RSZ1
Status: Status.UNKNOWN
Output dim: 1, lower bound: -0.0003029, upper bound: 0.0002967
time: 0.55 seconds

### Relational analysis RSZ of RS_RSZ1_RSZ2_RSZ2_RSZ1_RSZ2_RSZ2

#### Relational analysis RSZ result of RS_RSZ1_RSZ2_RSZ2_RSZ1_RSZ2_RSZ2
Status: Status.UNKNOWN
Output dim: 1, lower bound: -0.0002983, upper bound: 0.0003011
time: 0.57 seconds

## BFS RS instance: RS_RSZ1_RSZ2_RSZ2_RSZ2_RSZ1

### Backsubstitution after applying RS history:
0: 0.0009457, 0.0011003, 0.0009457, 0.0011003, -0.0001502, 0.0001504
1: 0.9936197, 0.9940771, 0.9936197, 0.9940771, -0.0004054, 0.0004031
2: -0.0065606, -0.0050208, -0.0065606, -0.0050208, -0.0012940, 0.0012710
3: 0.0038004, 0.0040140, 0.0038004, 0.0040140, -0.0001761, 0.0001777
4: 0.0023852, 0.0036022, 0.0023852, 0.0036022, -0.0010915, 0.0011095
5: 0.0060086, 0.0065111, 0.0060086, 0.0065111, -0.0005024, 0.0005024
6: -0.0013983, -0.0008639, -0.0013983, -0.0008639, -0.0004441, 0.0004361
7: -0.0083221, -0.0079387, -0.0083221, -0.0079387, -0.0003835, 0.0003835
8: 0.0048484, 0.0068715, 0.0048484, 0.0068715, -0.0014303, 0.0014616
9: -0.0036877, -0.0032079, -0.0036877, -0.0032079, -0.0004797, 0.0004797

### Unstable ReLU Count (Linear/Conv2D Layers)
- layer_idx=0, type=LayerType.Linear, total=256, inp1_unstable=5, inp2_unstable=5, delta_unstable=251
- layer_idx=2, type=LayerType.Linear, total=256, inp1_unstable=4, inp2_unstable=4, delta_unstable=256
- layer_idx=4, type=LayerType.Linear, total=256, inp1_unstable=0, inp2_unstable=0, delta_unstable=256
- layer_idx=6, type=LayerType.Linear, total=256, inp1_unstable=1, inp2_unstable=1, delta_unstable=256
- layer_idx=8, type=LayerType.Linear, total=10, inp1_unstable=0, inp2_unstable=0, delta_unstable=10

Time for backsubstitution: 1.13 seconds

### RS candidates at layer 3
type: RSZ, layer: 3, pos: 73
type: RSZ, layer: 3, pos: 80
type: RSZ, layer: 3, pos: 81
type: RSZ, layer: 3, pos: 136

Time for candidate selection: 0.22 seconds

### Candidate
type: RSZ, layer: 3, pos: 73

### Relational analysis RSZ of RS_RSZ1_RSZ2_RSZ2_RSZ2_RSZ1_RSZ1

#### Relational analysis RSZ result of RS_RSZ1_RSZ2_RSZ2_RSZ2_RSZ1_RSZ1
Status: Status.UNKNOWN
Output dim: 1, lower bound: -0.0003021, upper bound: 0.0002975
time: 0.55 seconds

### Relational analysis RSZ of RS_RSZ1_RSZ2_RSZ2_RSZ2_RSZ1_RSZ2

#### Relational analysis RSZ result of RS_RSZ1_RSZ2_RSZ2_RSZ2_RSZ1_RSZ2
Status: Status.UNKNOWN
Output dim: 1, lower bound: -0.0002976, upper bound: 0.0003019
time: 0.53 seconds

## BFS RS instance: RS_RSZ1_RSZ2_RSZ2_RSZ2_RSZ2

### Backsubstitution after applying RS history:
0: 0.0009457, 0.0011003, 0.0009457, 0.0011003, -0.0001502, 0.0001504
1: 0.9936197, 0.9940771, 0.9936197, 0.9940771, -0.0004053, 0.0004034
2: -0.0065606, -0.0050208, -0.0065606, -0.0050208, -0.0012927, 0.0012737
3: 0.0038004, 0.0040140, 0.0038004, 0.0040140, -0.0001763, 0.0001777
4: 0.0023852, 0.0036022, 0.0023852, 0.0036022, -0.0010937, 0.0011084
5: 0.0060086, 0.0065111, 0.0060086, 0.0065111, -0.0005024, 0.0005024
6: -0.0013983, -0.0008639, -0.0013983, -0.0008639, -0.0004437, 0.0004370
7: -0.0083221, -0.0079387, -0.0083221, -0.0079387, -0.0003835, 0.0003835
8: 0.0048484, 0.0068715, 0.0048484, 0.0068715, -0.0014339, 0.0014599
9: -0.0036877, -0.0032079, -0.0036877, -0.0032079, -0.0004797, 0.0004797

### Unstable ReLU Count (Linear/Conv2D Layers)
- layer_idx=0, type=LayerType.Linear, total=256, inp1_unstable=5, inp2_unstable=5, delta_unstable=251
- layer_idx=2, type=LayerType.Linear, total=256, inp1_unstable=4, inp2_unstable=4, delta_unstable=256
- layer_idx=4, type=LayerType.Linear, total=256, inp1_unstable=0, inp2_unstable=0, delta_unstable=256
- layer_idx=6, type=LayerType.Linear, total=256, inp1_unstable=1, inp2_unstable=1, delta_unstable=256
- layer_idx=8, type=LayerType.Linear, total=10, inp1_unstable=0, inp2_unstable=0, delta_unstable=10

Time for backsubstitution: 1.17 seconds

### RS candidates at layer 3
type: RSZ, layer: 3, pos: 73
type: RSZ, layer: 3, pos: 80
type: RSZ, layer: 3, pos: 81
type: RSZ, layer: 3, pos: 136

Time for candidate selection: 0.21 seconds

### Candidate
type: RSZ, layer: 3, pos: 73

### Relational analysis RSZ of RS_RSZ1_RSZ2_RSZ2_RSZ2_RSZ2_RSZ1

#### Relational analysis RSZ result of RS_RSZ1_RSZ2_RSZ2_RSZ2_RSZ2_RSZ1
Status: Status.UNKNOWN
Output dim: 1, lower bound: -0.0003021, upper bound: 0.0002977
time: 0.56 seconds

### Relational analysis RSZ of RS_RSZ1_RSZ2_RSZ2_RSZ2_RSZ2_RSZ2

#### Relational analysis RSZ result of RS_RSZ1_RSZ2_RSZ2_RSZ2_RSZ2_RSZ2
Status: Status.UNKNOWN
Output dim: 1, lower bound: -0.0002976, upper bound: 0.0003020
time: 0.62 seconds

## BFS RS instance: RS_RSZ2_RSZ1_RSZ1_RSZ1_RSZ1

### Backsubstitution after applying RS history:
0: 0.0009457, 0.0011003, 0.0009457, 0.0011003, -0.0001503, 0.0001503
1: 0.9936197, 0.9940771, 0.9936197, 0.9940771, -0.0004051, 0.0004044
2: -0.0065606, -0.0050208, -0.0065606, -0.0050208, -0.0012926, 0.0012845
3: 0.0038004, 0.0040140, 0.0038004, 0.0040140, -0.0001770, 0.0001776
4: 0.0023852, 0.0036022, 0.0023852, 0.0036022, -0.0011022, 0.0011084
5: 0.0060086, 0.0065111, 0.0060086, 0.0065111, -0.0005024, 0.0005024
6: -0.0013983, -0.0008639, -0.0013983, -0.0008639, -0.0004435, 0.0004408
7: -0.0083221, -0.0079387, -0.0083221, -0.0079387, -0.0003835, 0.0003835
8: 0.0048484, 0.0068715, 0.0048484, 0.0068715, -0.0014481, 0.0014586
9: -0.0036877, -0.0032079, -0.0036877, -0.0032079, -0.0004797, 0.0004797

### Unstable ReLU Count (Linear/Conv2D Layers)
- layer_idx=0, type=LayerType.Linear, total=256, inp1_unstable=5, inp2_unstable=5, delta_unstable=251
- layer_idx=2, type=LayerType.Linear, total=256, inp1_unstable=4, inp2_unstable=4, delta_unstable=256
- layer_idx=4, type=LayerType.Linear, total=256, inp1_unstable=0, inp2_unstable=0, delta_unstable=256
- layer_idx=6, type=LayerType.Linear, total=256, inp1_unstable=1, inp2_unstable=1, delta_unstable=256
- layer_idx=8, type=LayerType.Linear, total=10, inp1_unstable=0, inp2_unstable=0, delta_unstable=10

Time for backsubstitution: 1.14 seconds

### RS candidates at layer 3
type: RSZ, layer: 3, pos: 73
type: RSZ, layer: 3, pos: 80
type: RSZ, layer: 3, pos: 81
type: RSZ, layer: 3, pos: 136

Time for candidate selection: 0.21 seconds

### Candidate
type: RSZ, layer: 3, pos: 73

### Relational analysis RSZ of RS_RSZ2_RSZ1_RSZ1_RSZ1_RSZ1_RSZ1

#### Relational analysis RSZ result of RS_RSZ2_RSZ1_RSZ1_RSZ1_RSZ1_RSZ1
Status: Status.UNKNOWN
Output dim: 1, lower bound: -0.0003020, upper bound: 0.0002976
time: 0.58 seconds

### Relational analysis RSZ of RS_RSZ2_RSZ1_RSZ1_RSZ1_RSZ1_RSZ2

#### Relational analysis RSZ result of RS_RSZ2_RSZ1_RSZ1_RSZ1_RSZ1_RSZ2
Status: Status.UNKNOWN
Output dim: 1, lower bound: -0.0002977, upper bound: 0.0003021
time: 0.59 seconds

## BFS RS instance: RS_RSZ2_RSZ1_RSZ1_RSZ1_RSZ2

### Backsubstitution after applying RS history:
0: 0.0009457, 0.0011003, 0.0009457, 0.0011003, -0.0001503, 0.0001503
1: 0.9936197, 0.9940771, 0.9936197, 0.9940771, -0.0004049, 0.0004045
2: -0.0065606, -0.0050208, -0.0065606, -0.0050208, -0.0012897, 0.0012860
3: 0.0038004, 0.0040140, 0.0038004, 0.0040140, -0.0001771, 0.0001774
4: 0.0023852, 0.0036022, 0.0023852, 0.0036022, -0.0011034, 0.0011061
5: 0.0060086, 0.0065111, 0.0060086, 0.0065111, -0.0005024, 0.0005024
6: -0.0013983, -0.0008639, -0.0013983, -0.0008639, -0.0004425, 0.0004413
7: -0.0083221, -0.0079387, -0.0083221, -0.0079387, -0.0003835, 0.0003835
8: 0.0048484, 0.0068715, 0.0048484, 0.0068715, -0.0014500, 0.0014547
9: -0.0036877, -0.0032079, -0.0036877, -0.0032079, -0.0004797, 0.0004797

### Unstable ReLU Count (Linear/Conv2D Layers)
- layer_idx=0, type=LayerType.Linear, total=256, inp1_unstable=5, inp2_unstable=5, delta_unstable=251
- layer_idx=2, type=LayerType.Linear, total=256, inp1_unstable=4, inp2_unstable=4, delta_unstable=256
- layer_idx=4, type=LayerType.Linear, total=256, inp1_unstable=0, inp2_unstable=0, delta_unstable=256
- layer_idx=6, type=LayerType.Linear, total=256, inp1_unstable=1, inp2_unstable=1, delta_unstable=256
- layer_idx=8, type=LayerType.Linear, total=10, inp1_unstable=0, inp2_unstable=0, delta_unstable=10

Time for backsubstitution: 1.14 seconds

### RS candidates at layer 3
type: RSZ, layer: 3, pos: 73
type: RSZ, layer: 3, pos: 80
type: RSZ, layer: 3, pos: 81
type: RSZ, layer: 3, pos: 136

Time for candidate selection: 0.22 seconds

### Candidate
type: RSZ, layer: 3, pos: 73

### Relational analysis RSZ of RS_RSZ2_RSZ1_RSZ1_RSZ1_RSZ2_RSZ1

#### Relational analysis RSZ result of RS_RSZ2_RSZ1_RSZ1_RSZ1_RSZ2_RSZ1
Status: Status.UNKNOWN
Output dim: 1, lower bound: -0.0003019, upper bound: 0.0002976
time: 0.55 seconds

### Relational analysis RSZ of RS_RSZ2_RSZ1_RSZ1_RSZ1_RSZ2_RSZ2

#### Relational analysis RSZ result of RS_RSZ2_RSZ1_RSZ1_RSZ1_RSZ2_RSZ2
Status: Status.UNKNOWN
Output dim: 1, lower bound: -0.0002975, upper bound: 0.0003021
time: 0.56 seconds

## BFS RS instance: RS_RSZ2_RSZ1_RSZ1_RSZ2_RSZ1

### Backsubstitution after applying RS history:
0: 0.0009457, 0.0011003, 0.0009457, 0.0011003, -0.0001503, 0.0001504
1: 0.9936197, 0.9940771, 0.9936197, 0.9940771, -0.0004053, 0.0004043
2: -0.0065606, -0.0050208, -0.0065606, -0.0050208, -0.0012943, 0.0012831
3: 0.0038004, 0.0040140, 0.0038004, 0.0040140, -0.0001769, 0.0001777
4: 0.0023852, 0.0036022, 0.0023852, 0.0036022, -0.0011011, 0.0011097
5: 0.0060086, 0.0065111, 0.0060086, 0.0065111, -0.0005024, 0.0005024
6: -0.0013983, -0.0008639, -0.0013983, -0.0008639, -0.0004441, 0.0004403
7: -0.0083221, -0.0079387, -0.0083221, -0.0079387, -0.0003835, 0.0003835
8: 0.0048484, 0.0068715, 0.0048484, 0.0068715, -0.0014462, 0.0014608
9: -0.0036877, -0.0032079, -0.0036877, -0.0032079, -0.0004797, 0.0004797

### Unstable ReLU Count (Linear/Conv2D Layers)
- layer_idx=0, type=LayerType.Linear, total=256, inp1_unstable=5, inp2_unstable=5, delta_unstable=251
- layer_idx=2, type=LayerType.Linear, total=256, inp1_unstable=4, inp2_unstable=4, delta_unstable=256
- layer_idx=4, type=LayerType.Linear, total=256, inp1_unstable=0, inp2_unstable=0, delta_unstable=256
- layer_idx=6, type=LayerType.Linear, total=256, inp1_unstable=1, inp2_unstable=1, delta_unstable=256
- layer_idx=8, type=LayerType.Linear, total=10, inp1_unstable=0, inp2_unstable=0, delta_unstable=10

Time for backsubstitution: 1.15 seconds

### RS candidates at layer 3
type: RSZ, layer: 3, pos: 73
type: RSZ, layer: 3, pos: 80
type: RSZ, layer: 3, pos: 81
type: RSZ, layer: 3, pos: 136

Time for candidate selection: 0.23 seconds

### Candidate
type: RSZ, layer: 3, pos: 73

### Relational analysis RSZ of RS_RSZ2_RSZ1_RSZ1_RSZ2_RSZ1_RSZ1

#### Relational analysis RSZ result of RS_RSZ2_RSZ1_RSZ1_RSZ2_RSZ1_RSZ1
Status: Status.UNKNOWN
Output dim: 1, lower bound: -0.0003011, upper bound: 0.0002983
time: 0.57 seconds

### Relational analysis RSZ of RS_RSZ2_RSZ1_RSZ1_RSZ2_RSZ1_RSZ2

#### Relational analysis RSZ result of RS_RSZ2_RSZ1_RSZ1_RSZ2_RSZ1_RSZ2
Status: Status.UNKNOWN
Output dim: 1, lower bound: -0.0002967, upper bound: 0.0003029
time: 0.56 seconds

## BFS RS instance: RS_RSZ2_RSZ1_RSZ1_RSZ2_RSZ2

### Backsubstitution after applying RS history:
0: 0.0009457, 0.0011003, 0.0009457, 0.0011003, -0.0001503, 0.0001503
1: 0.9936197, 0.9940771, 0.9936197, 0.9940771, -0.0004052, 0.0004046
2: -0.0065606, -0.0050208, -0.0065606, -0.0050208, -0.0012927, 0.0012863
3: 0.0038004, 0.0040140, 0.0038004, 0.0040140, -0.0001772, 0.0001776
4: 0.0023852, 0.0036022, 0.0023852, 0.0036022, -0.0011036, 0.0011085
5: 0.0060086, 0.0065111, 0.0060086, 0.0065111, -0.0005024, 0.0005024
6: -0.0013983, -0.0008639, -0.0013983, -0.0008639, -0.0004436, 0.0004414
7: -0.0083221, -0.0079387, -0.0083221, -0.0079387, -0.0003835, 0.0003835
8: 0.0048484, 0.0068715, 0.0048484, 0.0068715, -0.0014504, 0.0014587
9: -0.0036877, -0.0032079, -0.0036877, -0.0032079, -0.0004797, 0.0004797

### Unstable ReLU Count (Linear/Conv2D Layers)
- layer_idx=0, type=LayerType.Linear, total=256, inp1_unstable=5, inp2_unstable=5, delta_unstable=251
- layer_idx=2, type=LayerType.Linear, total=256, inp1_unstable=4, inp2_unstable=4, delta_unstable=256
- layer_idx=4, type=LayerType.Linear, total=256, inp1_unstable=0, inp2_unstable=0, delta_unstable=256
- layer_idx=6, type=LayerType.Linear, total=256, inp1_unstable=1, inp2_unstable=1, delta_unstable=256
- layer_idx=8, type=LayerType.Linear, total=10, inp1_unstable=0, inp2_unstable=0, delta_unstable=10

Time for backsubstitution: 1.14 seconds

### RS candidates at layer 3
type: RSZ, layer: 3, pos: 73
type: RSZ, layer: 3, pos: 80
type: RSZ, layer: 3, pos: 81
type: RSZ, layer: 3, pos: 136

Time for candidate selection: 0.23 seconds

### Candidate
type: RSZ, layer: 3, pos: 73

### Relational analysis RSZ of RS_RSZ2_RSZ1_RSZ1_RSZ2_RSZ2_RSZ1

#### Relational analysis RSZ result of RS_RSZ2_RSZ1_RSZ1_RSZ2_RSZ2_RSZ1
Status: Status.UNKNOWN
Output dim: 1, lower bound: -0.0003011, upper bound: 0.0002983
time: 0.64 seconds

### Relational analysis RSZ of RS_RSZ2_RSZ1_RSZ1_RSZ2_RSZ2_RSZ2

#### Relational analysis RSZ result of RS_RSZ2_RSZ1_RSZ1_RSZ2_RSZ2_RSZ2
Status: Status.UNKNOWN
Output dim: 1, lower bound: -0.0002967, upper bound: 0.0003029
time: 0.59 seconds

## BFS RS instance: RS_RSZ2_RSZ1_RSZ2_RSZ1_RSZ1

### Backsubstitution after applying RS history:
0: 0.0009457, 0.0011003, 0.0009457, 0.0011003, -0.0001503, 0.0001504
1: 0.9936197, 0.9940771, 0.9936197, 0.9940771, -0.0004055, 0.0004043
2: -0.0065606, -0.0050208, -0.0065606, -0.0050208, -0.0012963, 0.0012834
3: 0.0038004, 0.0040140, 0.0038004, 0.0040140, -0.0001770, 0.0001778
4: 0.0023852, 0.0036022, 0.0023852, 0.0036022, -0.0011013, 0.0011113
5: 0.0060086, 0.0065111, 0.0060086, 0.0065111, -0.0005024, 0.0005024
6: -0.0013983, -0.0008639, -0.0013983, -0.0008639, -0.0004448, 0.0004404
7: -0.0083221, -0.0079387, -0.0083221, -0.0079387, -0.0003835, 0.0003835
8: 0.0048484, 0.0068715, 0.0048484, 0.0068715, -0.0014466, 0.0014635
9: -0.0036877, -0.0032079, -0.0036877, -0.0032079, -0.0004797, 0.0004797

### Unstable ReLU Count (Linear/Conv2D Layers)
- layer_idx=0, type=LayerType.Linear, total=256, inp1_unstable=5, inp2_unstable=5, delta_unstable=251
- layer_idx=2, type=LayerType.Linear, total=256, inp1_unstable=4, inp2_unstable=4, delta_unstable=256
- layer_idx=4, type=LayerType.Linear, total=256, inp1_unstable=0, inp2_unstable=0, delta_unstable=256
- layer_idx=6, type=LayerType.Linear, total=256, inp1_unstable=1, inp2_unstable=1, delta_unstable=256
- layer_idx=8, type=LayerType.Linear, total=10, inp1_unstable=0, inp2_unstable=0, delta_unstable=10

Time for backsubstitution: 1.22 seconds

### RS candidates at layer 3
type: RSZ, layer: 3, pos: 73
type: RSZ, layer: 3, pos: 80
type: RSZ, layer: 3, pos: 81
type: RSZ, layer: 3, pos: 136

Time for candidate selection: 0.22 seconds

### Candidate
type: RSZ, layer: 3, pos: 73

### Relational analysis RSZ of RS_RSZ2_RSZ1_RSZ2_RSZ1_RSZ1_RSZ1

#### Relational analysis RSZ result of RS_RSZ2_RSZ1_RSZ2_RSZ1_RSZ1_RSZ1
Status: Status.UNKNOWN
Output dim: 1, lower bound: -0.0003029, upper bound: 0.0002968
time: 0.55 seconds

### Relational analysis RSZ of RS_RSZ2_RSZ1_RSZ2_RSZ1_RSZ1_RSZ2

#### Relational analysis RSZ result of RS_RSZ2_RSZ1_RSZ2_RSZ1_RSZ1_RSZ2
Status: Status.UNKNOWN
Output dim: 1, lower bound: -0.0002982, upper bound: 0.0003013
time: 0.56 seconds

## BFS RS instance: RS_RSZ2_RSZ1_RSZ2_RSZ1_RSZ2

### Backsubstitution after applying RS history:
0: 0.0009457, 0.0011003, 0.0009457, 0.0011003, -0.0001503, 0.0001503
1: 0.9936197, 0.9940771, 0.9936197, 0.9940771, -0.0004052, 0.0004045
2: -0.0065606, -0.0050208, -0.0065606, -0.0050208, -0.0012933, 0.0012852
3: 0.0038004, 0.0040140, 0.0038004, 0.0040140, -0.0001771, 0.0001776
4: 0.0023852, 0.0036022, 0.0023852, 0.0036022, -0.0011027, 0.0011090
5: 0.0060086, 0.0065111, 0.0060086, 0.0065111, -0.0005024, 0.0005024
6: -0.0013983, -0.0008639, -0.0013983, -0.0008639, -0.0004438, 0.0004410
7: -0.0083221, -0.0079387, -0.0083221, -0.0079387, -0.0003835, 0.0003835
8: 0.0048484, 0.0068715, 0.0048484, 0.0068715, -0.0014489, 0.0014596
9: -0.0036877, -0.0032079, -0.0036877, -0.0032079, -0.0004797, 0.0004797

### Unstable ReLU Count (Linear/Conv2D Layers)
- layer_idx=0, type=LayerType.Linear, total=256, inp1_unstable=5, inp2_unstable=5, delta_unstable=251
- layer_idx=2, type=LayerType.Linear, total=256, inp1_unstable=4, inp2_unstable=4, delta_unstable=256
- layer_idx=4, type=LayerType.Linear, total=256, inp1_unstable=0, inp2_unstable=0, delta_unstable=256
- layer_idx=6, type=LayerType.Linear, total=256, inp1_unstable=1, inp2_unstable=1, delta_unstable=256
- layer_idx=8, type=LayerType.Linear, total=10, inp1_unstable=0, inp2_unstable=0, delta_unstable=10

Time for backsubstitution: 1.13 seconds

### RS candidates at layer 3
type: RSZ, layer: 3, pos: 73
type: RSZ, layer: 3, pos: 80
type: RSZ, layer: 3, pos: 81
type: RSZ, layer: 3, pos: 136

Time for candidate selection: 0.22 seconds

### Candidate
type: RSZ, layer: 3, pos: 73

### Relational analysis RSZ of RS_RSZ2_RSZ1_RSZ2_RSZ1_RSZ2_RSZ1

#### Relational analysis RSZ result of RS_RSZ2_RSZ1_RSZ2_RSZ1_RSZ2_RSZ1
Status: Status.UNKNOWN
Output dim: 1, lower bound: -0.0003029, upper bound: 0.0002968
time: 0.58 seconds

### Relational analysis RSZ of RS_RSZ2_RSZ1_RSZ2_RSZ1_RSZ2_RSZ2

#### Relational analysis RSZ result of RS_RSZ2_RSZ1_RSZ2_RSZ1_RSZ2_RSZ2
Status: Status.UNKNOWN
Output dim: 1, lower bound: -0.0002982, upper bound: 0.0003013
time: 0.59 seconds

## BFS RS instance: RS_RSZ2_RSZ1_RSZ2_RSZ2_RSZ1

### Backsubstitution after applying RS history:
0: 0.0009457, 0.0011003, 0.0009457, 0.0011003, -0.0001502, 0.0001504
1: 0.9936197, 0.9940771, 0.9936197, 0.9940771, -0.0004053, 0.0004039
2: -0.0065606, -0.0050208, -0.0065606, -0.0050208, -0.0012943, 0.0012793
3: 0.0038004, 0.0040140, 0.0038004, 0.0040140, -0.0001767, 0.0001777
4: 0.0023852, 0.0036022, 0.0023852, 0.0036022, -0.0010981, 0.0011097
5: 0.0060086, 0.0065111, 0.0060086, 0.0065111, -0.0005024, 0.0005024
6: -0.0013983, -0.0008639, -0.0013983, -0.0008639, -0.0004441, 0.0004390
7: -0.0083221, -0.0079387, -0.0083221, -0.0079387, -0.0003835, 0.0003835
8: 0.0048484, 0.0068715, 0.0048484, 0.0068715, -0.0014413, 0.0014609
9: -0.0036877, -0.0032079, -0.0036877, -0.0032079, -0.0004797, 0.0004797

### Unstable ReLU Count (Linear/Conv2D Layers)
- layer_idx=0, type=LayerType.Linear, total=256, inp1_unstable=5, inp2_unstable=5, delta_unstable=251
- layer_idx=2, type=LayerType.Linear, total=256, inp1_unstable=4, inp2_unstable=4, delta_unstable=256
- layer_idx=4, type=LayerType.Linear, total=256, inp1_unstable=0, inp2_unstable=0, delta_unstable=256
- layer_idx=6, type=LayerType.Linear, total=256, inp1_unstable=1, inp2_unstable=1, delta_unstable=256
- layer_idx=8, type=LayerType.Linear, total=10, inp1_unstable=0, inp2_unstable=0, delta_unstable=10

Time for backsubstitution: 1.15 seconds

### RS candidates at layer 3
type: RSZ, layer: 3, pos: 73
type: RSZ, layer: 3, pos: 80
type: RSZ, layer: 3, pos: 81
type: RSZ, layer: 3, pos: 136

Time for candidate selection: 0.22 seconds

### Candidate
type: RSZ, layer: 3, pos: 73

### Relational analysis RSZ of RS_RSZ2_RSZ1_RSZ2_RSZ2_RSZ1_RSZ1

#### Relational analysis RSZ result of RS_RSZ2_RSZ1_RSZ2_RSZ2_RSZ1_RSZ1
Status: Status.UNKNOWN
Output dim: 1, lower bound: -0.0003018, upper bound: 0.0002975
time: 0.59 seconds

### Relational analysis RSZ of RS_RSZ2_RSZ1_RSZ2_RSZ2_RSZ1_RSZ2

#### Relational analysis RSZ result of RS_RSZ2_RSZ1_RSZ2_RSZ2_RSZ1_RSZ2
Status: Status.UNKNOWN
Output dim: 1, lower bound: -0.0002972, upper bound: 0.0003019
time: 0.56 seconds

## BFS RS instance: RS_RSZ2_RSZ1_RSZ2_RSZ2_RSZ2

### Backsubstitution after applying RS history:
0: 0.0009457, 0.0011003, 0.0009457, 0.0011003, -0.0001503, 0.0001503
1: 0.9936197, 0.9940771, 0.9936197, 0.9940771, -0.0004052, 0.0004041
2: -0.0065606, -0.0050208, -0.0065606, -0.0050208, -0.0012930, 0.0012815
3: 0.0038004, 0.0040140, 0.0038004, 0.0040140, -0.0001768, 0.0001776
4: 0.0023852, 0.0036022, 0.0023852, 0.0036022, -0.0010998, 0.0011087
5: 0.0060086, 0.0065111, 0.0060086, 0.0065111, -0.0005024, 0.0005024
6: -0.0013983, -0.0008639, -0.0013983, -0.0008639, -0.0004437, 0.0004397
7: -0.0083221, -0.0079387, -0.0083221, -0.0079387, -0.0003835, 0.0003835
8: 0.0048484, 0.0068715, 0.0048484, 0.0068715, -0.0014441, 0.0014592
9: -0.0036877, -0.0032079, -0.0036877, -0.0032079, -0.0004797, 0.0004797

### Unstable ReLU Count (Linear/Conv2D Layers)
- layer_idx=0, type=LayerType.Linear, total=256, inp1_unstable=5, inp2_unstable=5, delta_unstable=251
- layer_idx=2, type=LayerType.Linear, total=256, inp1_unstable=4, inp2_unstable=4, delta_unstable=256
- layer_idx=4, type=LayerType.Linear, total=256, inp1_unstable=0, inp2_unstable=0, delta_unstable=256
- layer_idx=6, type=LayerType.Linear, total=256, inp1_unstable=1, inp2_unstable=1, delta_unstable=256
- layer_idx=8, type=LayerType.Linear, total=10, inp1_unstable=0, inp2_unstable=0, delta_unstable=10

Time for backsubstitution: 1.12 seconds

### RS candidates at layer 3
type: RSZ, layer: 3, pos: 73
type: RSZ, layer: 3, pos: 80
type: RSZ, layer: 3, pos: 81
type: RSZ, layer: 3, pos: 136

Time for candidate selection: 0.21 seconds

### Candidate
type: RSZ, layer: 3, pos: 73

### Relational analysis RSZ of RS_RSZ2_RSZ1_RSZ2_RSZ2_RSZ2_RSZ1

#### Relational analysis RSZ result of RS_RSZ2_RSZ1_RSZ2_RSZ2_RSZ2_RSZ1
Status: Status.UNKNOWN
Output dim: 1, lower bound: -0.0003018, upper bound: 0.0002977
time: 0.55 seconds

### Relational analysis RSZ of RS_RSZ2_RSZ1_RSZ2_RSZ2_RSZ2_RSZ2

#### Relational analysis RSZ result of RS_RSZ2_RSZ1_RSZ2_RSZ2_RSZ2_RSZ2
Status: Status.UNKNOWN
Output dim: 1, lower bound: -0.0002972, upper bound: 0.0003020
time: 0.59 seconds

## BFS RS instance: RS_RSZ2_RSZ2_RSZ1_RSZ1_RSZ1

### Backsubstitution after applying RS history:
0: 0.0009457, 0.0011003, 0.0009457, 0.0011003, -0.0001503, 0.0001504
1: 0.9936197, 0.9940771, 0.9936197, 0.9940771, -0.0004063, 0.0004041
2: -0.0065606, -0.0050208, -0.0065606, -0.0050208, -0.0013038, 0.0012814
3: 0.0038004, 0.0040140, 0.0038004, 0.0040140, -0.0001768, 0.0001784
4: 0.0023852, 0.0036022, 0.0023852, 0.0036022, -0.0010997, 0.0011173
5: 0.0060086, 0.0065111, 0.0060086, 0.0065111, -0.0005024, 0.0005024
6: -0.0013983, -0.0008639, -0.0013983, -0.0008639, -0.0004476, 0.0004397
7: -0.0083221, -0.0079387, -0.0083221, -0.0079387, -0.0003835, 0.0003835
8: 0.0048484, 0.0068715, 0.0048484, 0.0068715, -0.0014440, 0.0014744
9: -0.0036877, -0.0032079, -0.0036877, -0.0032079, -0.0004797, 0.0004797

### Unstable ReLU Count (Linear/Conv2D Layers)
- layer_idx=0, type=LayerType.Linear, total=256, inp1_unstable=5, inp2_unstable=5, delta_unstable=251
- layer_idx=2, type=LayerType.Linear, total=256, inp1_unstable=4, inp2_unstable=4, delta_unstable=256
- layer_idx=4, type=LayerType.Linear, total=256, inp1_unstable=0, inp2_unstable=0, delta_unstable=256
- layer_idx=6, type=LayerType.Linear, total=256, inp1_unstable=1, inp2_unstable=1, delta_unstable=256
- layer_idx=8, type=LayerType.Linear, total=10, inp1_unstable=0, inp2_unstable=0, delta_unstable=10

Time for backsubstitution: 1.18 seconds

### RS candidates at layer 3
type: RSZ, layer: 3, pos: 73
type: RSZ, layer: 3, pos: 80
type: RSZ, layer: 3, pos: 81
type: RSZ, layer: 3, pos: 136

Time for candidate selection: 0.22 seconds

### Candidate
type: RSZ, layer: 3, pos: 73

### Relational analysis RSZ of RS_RSZ2_RSZ2_RSZ1_RSZ1_RSZ1_RSZ1

#### Relational analysis RSZ result of RS_RSZ2_RSZ2_RSZ1_RSZ1_RSZ1_RSZ1
Status: Status.UNKNOWN
Output dim: 1, lower bound: -0.0003020, upper bound: 0.0002972
time: 0.54 seconds

### Relational analysis RSZ of RS_RSZ2_RSZ2_RSZ1_RSZ1_RSZ1_RSZ2

#### Relational analysis RSZ result of RS_RSZ2_RSZ2_RSZ1_RSZ1_RSZ1_RSZ2
Status: Status.UNKNOWN
Output dim: 1, lower bound: -0.0002977, upper bound: 0.0003018
time: 0.57 seconds

## BFS RS instance: RS_RSZ2_RSZ2_RSZ1_RSZ1_RSZ2

### Backsubstitution after applying RS history:
0: 0.0009457, 0.0011003, 0.0009457, 0.0011003, -0.0001503, 0.0001504
1: 0.9936197, 0.9940771, 0.9936197, 0.9940771, -0.0004060, 0.0004042
2: -0.0065606, -0.0050208, -0.0065606, -0.0050208, -0.0013009, 0.0012827
3: 0.0038004, 0.0040140, 0.0038004, 0.0040140, -0.0001769, 0.0001782
4: 0.0023852, 0.0036022, 0.0023852, 0.0036022, -0.0011007, 0.0011149
5: 0.0060086, 0.0065111, 0.0060086, 0.0065111, -0.0005024, 0.0005024
6: -0.0013983, -0.0008639, -0.0013983, -0.0008639, -0.0004465, 0.0004401
7: -0.0083221, -0.0079387, -0.0083221, -0.0079387, -0.0003835, 0.0003835
8: 0.0048484, 0.0068715, 0.0048484, 0.0068715, -0.0014456, 0.0014706
9: -0.0036877, -0.0032079, -0.0036877, -0.0032079, -0.0004797, 0.0004797

### Unstable ReLU Count (Linear/Conv2D Layers)
- layer_idx=0, type=LayerType.Linear, total=256, inp1_unstable=5, inp2_unstable=5, delta_unstable=251
- layer_idx=2, type=LayerType.Linear, total=256, inp1_unstable=4, inp2_unstable=4, delta_unstable=256
- layer_idx=4, type=LayerType.Linear, total=256, inp1_unstable=0, inp2_unstable=0, delta_unstable=256
- layer_idx=6, type=LayerType.Linear, total=256, inp1_unstable=1, inp2_unstable=1, delta_unstable=256
- layer_idx=8, type=LayerType.Linear, total=10, inp1_unstable=0, inp2_unstable=0, delta_unstable=10

Time for backsubstitution: 1.13 seconds

### RS candidates at layer 3
type: RSZ, layer: 3, pos: 73
type: RSZ, layer: 3, pos: 80
type: RSZ, layer: 3, pos: 81
type: RSZ, layer: 3, pos: 136

Time for candidate selection: 0.22 seconds

### Candidate
type: RSZ, layer: 3, pos: 73

### Relational analysis RSZ of RS_RSZ2_RSZ2_RSZ1_RSZ1_RSZ2_RSZ1

#### Relational analysis RSZ result of RS_RSZ2_RSZ2_RSZ1_RSZ1_RSZ2_RSZ1
Status: Status.UNKNOWN
Output dim: 1, lower bound: -0.0003019, upper bound: 0.0002972
time: 0.57 seconds

### Relational analysis RSZ of RS_RSZ2_RSZ2_RSZ1_RSZ1_RSZ2_RSZ2

#### Relational analysis RSZ result of RS_RSZ2_RSZ2_RSZ1_RSZ1_RSZ2_RSZ2
Status: Status.UNKNOWN
Output dim: 1, lower bound: -0.0002975, upper bound: 0.0003018
time: 0.59 seconds

## BFS RS instance: RS_RSZ2_RSZ2_RSZ1_RSZ2_RSZ1

### Backsubstitution after applying RS history:
0: 0.0009457, 0.0011003, 0.0009457, 0.0011003, -0.0001503, 0.0001504
1: 0.9936197, 0.9940771, 0.9936197, 0.9940771, -0.0004065, 0.0004041
2: -0.0065606, -0.0050208, -0.0065606, -0.0050208, -0.0013055, 0.0012817
3: 0.0038004, 0.0040140, 0.0038004, 0.0040140, -0.0001769, 0.0001785
4: 0.0023852, 0.0036022, 0.0023852, 0.0036022, -0.0011000, 0.0011186
5: 0.0060086, 0.0065111, 0.0060086, 0.0065111, -0.0005024, 0.0005024
6: -0.0013983, -0.0008639, -0.0013983, -0.0008639, -0.0004481, 0.0004398
7: -0.0083221, -0.0079387, -0.0083221, -0.0079387, -0.0003835, 0.0003835
8: 0.0048484, 0.0068715, 0.0048484, 0.0068715, -0.0014444, 0.0014766
9: -0.0036877, -0.0032079, -0.0036877, -0.0032079, -0.0004797, 0.0004797

### Unstable ReLU Count (Linear/Conv2D Layers)
- layer_idx=0, type=LayerType.Linear, total=256, inp1_unstable=5, inp2_unstable=5, delta_unstable=251
- layer_idx=2, type=LayerType.Linear, total=256, inp1_unstable=4, inp2_unstable=4, delta_unstable=256
- layer_idx=4, type=LayerType.Linear, total=256, inp1_unstable=0, inp2_unstable=0, delta_unstable=256
- layer_idx=6, type=LayerType.Linear, total=256, inp1_unstable=1, inp2_unstable=1, delta_unstable=256
- layer_idx=8, type=LayerType.Linear, total=10, inp1_unstable=0, inp2_unstable=0, delta_unstable=10

Time for backsubstitution: 1.25 seconds

### RS candidates at layer 3
type: RSZ, layer: 3, pos: 73
type: RSZ, layer: 3, pos: 80
type: RSZ, layer: 3, pos: 81
type: RSZ, layer: 3, pos: 136

Time for candidate selection: 0.25 seconds

### Candidate
type: RSZ, layer: 3, pos: 73

### Relational analysis RSZ of RS_RSZ2_RSZ2_RSZ1_RSZ2_RSZ1_RSZ1

#### Relational analysis RSZ result of RS_RSZ2_RSZ2_RSZ1_RSZ2_RSZ1_RSZ1
Status: Status.UNKNOWN
Output dim: 1, lower bound: -0.0003013, upper bound: 0.0002982
time: 0.56 seconds

### Relational analysis RSZ of RS_RSZ2_RSZ2_RSZ1_RSZ2_RSZ1_RSZ2

#### Relational analysis RSZ result of RS_RSZ2_RSZ2_RSZ1_RSZ2_RSZ1_RSZ2
Status: Status.UNKNOWN
Output dim: 1, lower bound: -0.0002968, upper bound: 0.0003029
time: 0.56 seconds

## BFS RS instance: RS_RSZ2_RSZ2_RSZ1_RSZ2_RSZ2

### Backsubstitution after applying RS history:
0: 0.0009457, 0.0011003, 0.0009457, 0.0011003, -0.0001503, 0.0001504
1: 0.9936197, 0.9940771, 0.9936197, 0.9940771, -0.0004063, 0.0004044
2: -0.0065606, -0.0050208, -0.0065606, -0.0050208, -0.0013039, 0.0012847
3: 0.0038004, 0.0040140, 0.0038004, 0.0040140, -0.0001770, 0.0001784
4: 0.0023852, 0.0036022, 0.0023852, 0.0036022, -0.0011023, 0.0011173
5: 0.0060086, 0.0065111, 0.0060086, 0.0065111, -0.0005024, 0.0005024
6: -0.0013983, -0.0008639, -0.0013983, -0.0008639, -0.0004476, 0.0004408
7: -0.0083221, -0.0079387, -0.0083221, -0.0079387, -0.0003835, 0.0003835
8: 0.0048484, 0.0068715, 0.0048484, 0.0068715, -0.0014483, 0.0014745
9: -0.0036877, -0.0032079, -0.0036877, -0.0032079, -0.0004797, 0.0004797

### Unstable ReLU Count (Linear/Conv2D Layers)
- layer_idx=0, type=LayerType.Linear, total=256, inp1_unstable=5, inp2_unstable=5, delta_unstable=251
- layer_idx=2, type=LayerType.Linear, total=256, inp1_unstable=4, inp2_unstable=4, delta_unstable=256
- layer_idx=4, type=LayerType.Linear, total=256, inp1_unstable=0, inp2_unstable=0, delta_unstable=256
- layer_idx=6, type=LayerType.Linear, total=256, inp1_unstable=1, inp2_unstable=1, delta_unstable=256
- layer_idx=8, type=LayerType.Linear, total=10, inp1_unstable=0, inp2_unstable=0, delta_unstable=10

Time for backsubstitution: 1.16 seconds

### RS candidates at layer 3
type: RSZ, layer: 3, pos: 73
type: RSZ, layer: 3, pos: 80
type: RSZ, layer: 3, pos: 81
type: RSZ, layer: 3, pos: 136

Time for candidate selection: 0.23 seconds

### Candidate
type: RSZ, layer: 3, pos: 73

### Relational analysis RSZ of RS_RSZ2_RSZ2_RSZ1_RSZ2_RSZ2_RSZ1

#### Relational analysis RSZ result of RS_RSZ2_RSZ2_RSZ1_RSZ2_RSZ2_RSZ1
Status: Status.UNKNOWN
Output dim: 1, lower bound: -0.0003013, upper bound: 0.0002982
time: 0.56 seconds

### Relational analysis RSZ of RS_RSZ2_RSZ2_RSZ1_RSZ2_RSZ2_RSZ2

#### Relational analysis RSZ result of RS_RSZ2_RSZ2_RSZ1_RSZ2_RSZ2_RSZ2
Status: Status.UNKNOWN
Output dim: 1, lower bound: -0.0002968, upper bound: 0.0003029
time: 0.58 seconds

## BFS RS instance: RS_RSZ2_RSZ2_RSZ2_RSZ1_RSZ1

### Backsubstitution after applying RS history:
0: 0.0009457, 0.0011003, 0.0009457, 0.0011003, -0.0001503, 0.0001505
1: 0.9936197, 0.9940771, 0.9936197, 0.9940771, -0.0004067, 0.0004041
2: -0.0065606, -0.0050208, -0.0065606, -0.0050208, -0.0013076, 0.0012810
3: 0.0038004, 0.0040140, 0.0038004, 0.0040140, -0.0001768, 0.0001786
4: 0.0023852, 0.0036022, 0.0023852, 0.0036022, -0.0010995, 0.0011202
5: 0.0060086, 0.0065111, 0.0060086, 0.0065111, -0.0005024, 0.0005024
6: -0.0013983, -0.0008639, -0.0013983, -0.0008639, -0.0004489, 0.0004396
7: -0.0083221, -0.0079387, -0.0083221, -0.0079387, -0.0003835, 0.0003835
8: 0.0048484, 0.0068715, 0.0048484, 0.0068715, -0.0014435, 0.0014793
9: -0.0036877, -0.0032079, -0.0036877, -0.0032079, -0.0004797, 0.0004797

### Unstable ReLU Count (Linear/Conv2D Layers)
- layer_idx=0, type=LayerType.Linear, total=256, inp1_unstable=5, inp2_unstable=5, delta_unstable=251
- layer_idx=2, type=LayerType.Linear, total=256, inp1_unstable=4, inp2_unstable=4, delta_unstable=256
- layer_idx=4, type=LayerType.Linear, total=256, inp1_unstable=0, inp2_unstable=0, delta_unstable=256
- layer_idx=6, type=LayerType.Linear, total=256, inp1_unstable=1, inp2_unstable=1, delta_unstable=256
- layer_idx=8, type=LayerType.Linear, total=10, inp1_unstable=0, inp2_unstable=0, delta_unstable=10

Time for backsubstitution: 1.18 seconds

### RS candidates at layer 3
type: RSZ, layer: 3, pos: 73
type: RSZ, layer: 3, pos: 80
type: RSZ, layer: 3, pos: 81
type: RSZ, layer: 3, pos: 136

Time for candidate selection: 0.25 seconds

### Candidate
type: RSZ, layer: 3, pos: 73

### Relational analysis RSZ of RS_RSZ2_RSZ2_RSZ2_RSZ1_RSZ1_RSZ1

#### Relational analysis RSZ result of RS_RSZ2_RSZ2_RSZ2_RSZ1_RSZ1_RSZ1
Status: Status.UNKNOWN
Output dim: 1, lower bound: -0.0003029, upper bound: 0.0002967
time: 0.57 seconds

### Relational analysis RSZ of RS_RSZ2_RSZ2_RSZ2_RSZ1_RSZ1_RSZ2

#### Relational analysis RSZ result of RS_RSZ2_RSZ2_RSZ2_RSZ1_RSZ1_RSZ2
Status: Status.UNKNOWN
Output dim: 1, lower bound: -0.0002983, upper bound: 0.0003011
time: 0.55 seconds

## BFS RS instance: RS_RSZ2_RSZ2_RSZ2_RSZ1_RSZ2

### Backsubstitution after applying RS history:
0: 0.0009457, 0.0011003, 0.0009457, 0.0011003, -0.0001503, 0.0001504
1: 0.9936197, 0.9940771, 0.9936197, 0.9940771, -0.0004064, 0.0004042
2: -0.0065606, -0.0050208, -0.0065606, -0.0050208, -0.0013046, 0.0012826
3: 0.0038004, 0.0040140, 0.0038004, 0.0040140, -0.0001769, 0.0001784
4: 0.0023852, 0.0036022, 0.0023852, 0.0036022, -0.0011007, 0.0011179
5: 0.0060086, 0.0065111, 0.0060086, 0.0065111, -0.0005024, 0.0005024
6: -0.0013983, -0.0008639, -0.0013983, -0.0008639, -0.0004478, 0.0004401
7: -0.0083221, -0.0079387, -0.0083221, -0.0079387, -0.0003835, 0.0003835
8: 0.0048484, 0.0068715, 0.0048484, 0.0068715, -0.0014456, 0.0014754
9: -0.0036877, -0.0032079, -0.0036877, -0.0032079, -0.0004797, 0.0004797

### Unstable ReLU Count (Linear/Conv2D Layers)
- layer_idx=0, type=LayerType.Linear, total=256, inp1_unstable=5, inp2_unstable=5, delta_unstable=251
- layer_idx=2, type=LayerType.Linear, total=256, inp1_unstable=4, inp2_unstable=4, delta_unstable=256
- layer_idx=4, type=LayerType.Linear, total=256, inp1_unstable=0, inp2_unstable=0, delta_unstable=256
- layer_idx=6, type=LayerType.Linear, total=256, inp1_unstable=1, inp2_unstable=1, delta_unstable=256
- layer_idx=8, type=LayerType.Linear, total=10, inp1_unstable=0, inp2_unstable=0, delta_unstable=10

Time for backsubstitution: 1.17 seconds

### RS candidates at layer 3
type: RSZ, layer: 3, pos: 73
type: RSZ, layer: 3, pos: 80
type: RSZ, layer: 3, pos: 81
type: RSZ, layer: 3, pos: 136

Time for candidate selection: 0.22 seconds

### Candidate
type: RSZ, layer: 3, pos: 73

### Relational analysis RSZ of RS_RSZ2_RSZ2_RSZ2_RSZ1_RSZ2_RSZ1

#### Relational analysis RSZ result of RS_RSZ2_RSZ2_RSZ2_RSZ1_RSZ2_RSZ1
Status: Status.UNKNOWN
Output dim: 1, lower bound: -0.0003029, upper bound: 0.0002967
time: 0.55 seconds

### Relational analysis RSZ of RS_RSZ2_RSZ2_RSZ2_RSZ1_RSZ2_RSZ2

#### Relational analysis RSZ result of RS_RSZ2_RSZ2_RSZ2_RSZ1_RSZ2_RSZ2
Status: Status.UNKNOWN
Output dim: 1, lower bound: -0.0002983, upper bound: 0.0003011
time: 0.56 seconds

## BFS RS instance: RS_RSZ2_RSZ2_RSZ2_RSZ2_RSZ1

### Backsubstitution after applying RS history:
0: 0.0009457, 0.0011003, 0.0009457, 0.0011003, -0.0001502, 0.0001504
1: 0.9936197, 0.9940771, 0.9936197, 0.9940771, -0.0004065, 0.0004038
2: -0.0065606, -0.0050208, -0.0065606, -0.0050208, -0.0013056, 0.0012780
3: 0.0038004, 0.0040140, 0.0038004, 0.0040140, -0.0001766, 0.0001785
4: 0.0023852, 0.0036022, 0.0023852, 0.0036022, -0.0010971, 0.0011186
5: 0.0060086, 0.0065111, 0.0060086, 0.0065111, -0.0005024, 0.0005024
6: -0.0013983, -0.0008639, -0.0013983, -0.0008639, -0.0004482, 0.0004385
7: -0.0083221, -0.0079387, -0.0083221, -0.0079387, -0.0003835, 0.0003835
8: 0.0048484, 0.0068715, 0.0048484, 0.0068715, -0.0014395, 0.0014767
9: -0.0036877, -0.0032079, -0.0036877, -0.0032079, -0.0004797, 0.0004797

### Unstable ReLU Count (Linear/Conv2D Layers)
- layer_idx=0, type=LayerType.Linear, total=256, inp1_unstable=5, inp2_unstable=5, delta_unstable=251
- layer_idx=2, type=LayerType.Linear, total=256, inp1_unstable=4, inp2_unstable=4, delta_unstable=256
- layer_idx=4, type=LayerType.Linear, total=256, inp1_unstable=0, inp2_unstable=0, delta_unstable=256
- layer_idx=6, type=LayerType.Linear, total=256, inp1_unstable=1, inp2_unstable=1, delta_unstable=256
- layer_idx=8, type=LayerType.Linear, total=10, inp1_unstable=0, inp2_unstable=0, delta_unstable=10

Time for backsubstitution: 1.14 seconds

### RS candidates at layer 3
type: RSZ, layer: 3, pos: 73
type: RSZ, layer: 3, pos: 80
type: RSZ, layer: 3, pos: 81
type: RSZ, layer: 3, pos: 136

Time for candidate selection: 0.23 seconds

### Candidate
type: RSZ, layer: 3, pos: 73

### Relational analysis RSZ of RS_RSZ2_RSZ2_RSZ2_RSZ2_RSZ1_RSZ1

#### Relational analysis RSZ result of RS_RSZ2_RSZ2_RSZ2_RSZ2_RSZ1_RSZ1
Status: Status.UNKNOWN
Output dim: 1, lower bound: -0.0003021, upper bound: 0.0002975
time: 0.56 seconds

### Relational analysis RSZ of RS_RSZ2_RSZ2_RSZ2_RSZ2_RSZ1_RSZ2

#### Relational analysis RSZ result of RS_RSZ2_RSZ2_RSZ2_RSZ2_RSZ1_RSZ2
Status: Status.UNKNOWN
Output dim: 1, lower bound: -0.0002976, upper bound: 0.0003019
time: 0.53 seconds

## BFS RS instance: RS_RSZ2_RSZ2_RSZ2_RSZ2_RSZ2

### Backsubstitution after applying RS history:
0: 0.0009457, 0.0011003, 0.0009457, 0.0011003, -0.0001503, 0.0001504
1: 0.9936197, 0.9940771, 0.9936197, 0.9940771, -0.0004063, 0.0004041
2: -0.0065606, -0.0050208, -0.0065606, -0.0050208, -0.0013043, 0.0012809
3: 0.0038004, 0.0040140, 0.0038004, 0.0040140, -0.0001768, 0.0001784
4: 0.0023852, 0.0036022, 0.0023852, 0.0036022, -0.0010994, 0.0011176
5: 0.0060086, 0.0065111, 0.0060086, 0.0065111, -0.0005024, 0.0005024
6: -0.0013983, -0.0008639, -0.0013983, -0.0008639, -0.0004477, 0.0004395
7: -0.0083221, -0.0079387, -0.0083221, -0.0079387, -0.0003835, 0.0003835
8: 0.0048484, 0.0068715, 0.0048484, 0.0068715, -0.0014434, 0.0014750
9: -0.0036877, -0.0032079, -0.0036877, -0.0032079, -0.0004797, 0.0004797

### Unstable ReLU Count (Linear/Conv2D Layers)
- layer_idx=0, type=LayerType.Linear, total=256, inp1_unstable=5, inp2_unstable=5, delta_unstable=251
- layer_idx=2, type=LayerType.Linear, total=256, inp1_unstable=4, inp2_unstable=4, delta_unstable=256
- layer_idx=4, type=LayerType.Linear, total=256, inp1_unstable=0, inp2_unstable=0, delta_unstable=256
- layer_idx=6, type=LayerType.Linear, total=256, inp1_unstable=1, inp2_unstable=1, delta_unstable=256
- layer_idx=8, type=LayerType.Linear, total=10, inp1_unstable=0, inp2_unstable=0, delta_unstable=10

Time for backsubstitution: 1.14 seconds

### RS candidates at layer 3
type: RSZ, layer: 3, pos: 73
type: RSZ, layer: 3, pos: 80
type: RSZ, layer: 3, pos: 81
type: RSZ, layer: 3, pos: 136

Time for candidate selection: 0.22 seconds

### Candidate
type: RSZ, layer: 3, pos: 73

### Relational analysis RSZ of RS_RSZ2_RSZ2_RSZ2_RSZ2_RSZ2_RSZ1

#### Relational analysis RSZ result of RS_RSZ2_RSZ2_RSZ2_RSZ2_RSZ2_RSZ1
Status: Status.UNKNOWN
Output dim: 1, lower bound: -0.0003021, upper bound: 0.0002977
time: 0.56 seconds

### Relational analysis RSZ of RS_RSZ2_RSZ2_RSZ2_RSZ2_RSZ2_RSZ2

#### Relational analysis RSZ result of RS_RSZ2_RSZ2_RSZ2_RSZ2_RSZ2_RSZ2
Status: Status.UNKNOWN
Output dim: 1, lower bound: -0.0002976, upper bound: 0.0003020
time: 0.59 seconds

## Summary of splitting (split count: 5)
- Time for RS candidates: 2.51 seconds
RS_RSZ1_RSZ1_RSZ1_RSZ1_RSZ1_RSZ1, status: Status.UNKNOWN, split count: 6, time: 2.51
Output dim: 1, lower bound: -0.0003020, upper bound: 0.0002976
RS_RSZ1_RSZ1_RSZ1_RSZ1_RSZ1_RSZ2, status: Status.UNKNOWN, split count: 6, time: 2.51
Output dim: 1, lower bound: -0.0002977, upper bound: 0.0003021
RS_RSZ1_RSZ1_RSZ1_RSZ1_RSZ2_RSZ1, status: Status.UNKNOWN, split count: 6, time: 2.51
Output dim: 1, lower bound: -0.0003019, upper bound: 0.0002976
RS_RSZ1_RSZ1_RSZ1_RSZ1_RSZ2_RSZ2, status: Status.UNKNOWN, split count: 6, time: 2.51
Output dim: 1, lower bound: -0.0002975, upper bound: 0.0003021
RS_RSZ1_RSZ1_RSZ1_RSZ2_RSZ1_RSZ1, status: Status.UNKNOWN, split count: 6, time: 2.51
Output dim: 1, lower bound: -0.0003011, upper bound: 0.0002983
RS_RSZ1_RSZ1_RSZ1_RSZ2_RSZ1_RSZ2, status: Status.UNKNOWN, split count: 6, time: 2.51
Output dim: 1, lower bound: -0.0002967, upper bound: 0.0003029
RS_RSZ1_RSZ1_RSZ1_RSZ2_RSZ2_RSZ1, status: Status.UNKNOWN, split count: 6, time: 2.51
Output dim: 1, lower bound: -0.0003011, upper bound: 0.0002983
RS_RSZ1_RSZ1_RSZ1_RSZ2_RSZ2_RSZ2, status: Status.UNKNOWN, split count: 6, time: 2.51
Output dim: 1, lower bound: -0.0002967, upper bound: 0.0003029
RS_RSZ1_RSZ1_RSZ2_RSZ1_RSZ1_RSZ1, status: Status.UNKNOWN, split count: 6, time: 2.51
Output dim: 1, lower bound: -0.0003029, upper bound: 0.0002968
RS_RSZ1_RSZ1_RSZ2_RSZ1_RSZ1_RSZ2, status: Status.UNKNOWN, split count: 6, time: 2.51
Output dim: 1, lower bound: -0.0002982, upper bound: 0.0003013
RS_RSZ1_RSZ1_RSZ2_RSZ1_RSZ2_RSZ1, status: Status.UNKNOWN, split count: 6, time: 2.51
Output dim: 1, lower bound: -0.0003029, upper bound: 0.0002968
RS_RSZ1_RSZ1_RSZ2_RSZ1_RSZ2_RSZ2, status: Status.UNKNOWN, split count: 6, time: 2.51
Output dim: 1, lower bound: -0.0002982, upper bound: 0.0003013
RS_RSZ1_RSZ1_RSZ2_RSZ2_RSZ1_RSZ1, status: Status.UNKNOWN, split count: 6, time: 2.51
Output dim: 1, lower bound: -0.0003018, upper bound: 0.0002975
RS_RSZ1_RSZ1_RSZ2_RSZ2_RSZ1_RSZ2, status: Status.UNKNOWN, split count: 6, time: 2.51
Output dim: 1, lower bound: -0.0002972, upper bound: 0.0003019
RS_RSZ1_RSZ1_RSZ2_RSZ2_RSZ2_RSZ1, status: Status.UNKNOWN, split count: 6, time: 2.51
Output dim: 1, lower bound: -0.0003018, upper bound: 0.0002977
RS_RSZ1_RSZ1_RSZ2_RSZ2_RSZ2_RSZ2, status: Status.UNKNOWN, split count: 6, time: 2.51
Output dim: 1, lower bound: -0.0002972, upper bound: 0.0003020
RS_RSZ1_RSZ2_RSZ1_RSZ1_RSZ1_RSZ1, status: Status.UNKNOWN, split count: 6, time: 2.51
Output dim: 1, lower bound: -0.0003020, upper bound: 0.0002972
RS_RSZ1_RSZ2_RSZ1_RSZ1_RSZ1_RSZ2, status: Status.UNKNOWN, split count: 6, time: 2.51
Output dim: 1, lower bound: -0.0002977, upper bound: 0.0003018
RS_RSZ1_RSZ2_RSZ1_RSZ1_RSZ2_RSZ1, status: Status.UNKNOWN, split count: 6, time: 2.51
Output dim: 1, lower bound: -0.0003019, upper bound: 0.0002972
RS_RSZ1_RSZ2_RSZ1_RSZ1_RSZ2_RSZ2, status: Status.UNKNOWN, split count: 6, time: 2.51
Output dim: 1, lower bound: -0.0002975, upper bound: 0.0003018
RS_RSZ1_RSZ2_RSZ1_RSZ2_RSZ1_RSZ1, status: Status.UNKNOWN, split count: 6, time: 2.51
Output dim: 1, lower bound: -0.0003013, upper bound: 0.0002982
RS_RSZ1_RSZ2_RSZ1_RSZ2_RSZ1_RSZ2, status: Status.UNKNOWN, split count: 6, time: 2.51
Output dim: 1, lower bound: -0.0002968, upper bound: 0.0003029
RS_RSZ1_RSZ2_RSZ1_RSZ2_RSZ2_RSZ1, status: Status.UNKNOWN, split count: 6, time: 2.51
Output dim: 1, lower bound: -0.0003013, upper bound: 0.0002982
RS_RSZ1_RSZ2_RSZ1_RSZ2_RSZ2_RSZ2, status: Status.UNKNOWN, split count: 6, time: 2.51
Output dim: 1, lower bound: -0.0002968, upper bound: 0.0003029
RS_RSZ1_RSZ2_RSZ2_RSZ1_RSZ1_RSZ1, status: Status.UNKNOWN, split count: 6, time: 2.51
Output dim: 1, lower bound: -0.0003029, upper bound: 0.0002967
RS_RSZ1_RSZ2_RSZ2_RSZ1_RSZ1_RSZ2, status: Status.UNKNOWN, split count: 6, time: 2.51
Output dim: 1, lower bound: -0.0002983, upper bound: 0.0003011
RS_RSZ1_RSZ2_RSZ2_RSZ1_RSZ2_RSZ1, status: Status.UNKNOWN, split count: 6, time: 2.51
Output dim: 1, lower bound: -0.0003029, upper bound: 0.0002967
RS_RSZ1_RSZ2_RSZ2_RSZ1_RSZ2_RSZ2, status: Status.UNKNOWN, split count: 6, time: 2.51
Output dim: 1, lower bound: -0.0002983, upper bound: 0.0003011
RS_RSZ1_RSZ2_RSZ2_RSZ2_RSZ1_RSZ1, status: Status.UNKNOWN, split count: 6, time: 2.51
Output dim: 1, lower bound: -0.0003021, upper bound: 0.0002975
RS_RSZ1_RSZ2_RSZ2_RSZ2_RSZ1_RSZ2, status: Status.UNKNOWN, split count: 6, time: 2.51
Output dim: 1, lower bound: -0.0002976, upper bound: 0.0003019
RS_RSZ1_RSZ2_RSZ2_RSZ2_RSZ2_RSZ1, status: Status.UNKNOWN, split count: 6, time: 2.51
Output dim: 1, lower bound: -0.0003021, upper bound: 0.0002977
RS_RSZ1_RSZ2_RSZ2_RSZ2_RSZ2_RSZ2, status: Status.UNKNOWN, split count: 6, time: 2.51
Output dim: 1, lower bound: -0.0002976, upper bound: 0.0003020
RS_RSZ2_RSZ1_RSZ1_RSZ1_RSZ1_RSZ1, status: Status.UNKNOWN, split count: 6, time: 2.51
Output dim: 1, lower bound: -0.0003020, upper bound: 0.0002976
RS_RSZ2_RSZ1_RSZ1_RSZ1_RSZ1_RSZ2, status: Status.UNKNOWN, split count: 6, time: 2.51
Output dim: 1, lower bound: -0.0002977, upper bound: 0.0003021
RS_RSZ2_RSZ1_RSZ1_RSZ1_RSZ2_RSZ1, status: Status.UNKNOWN, split count: 6, time: 2.51
Output dim: 1, lower bound: -0.0003019, upper bound: 0.0002976
RS_RSZ2_RSZ1_RSZ1_RSZ1_RSZ2_RSZ2, status: Status.UNKNOWN, split count: 6, time: 2.51
Output dim: 1, lower bound: -0.0002975, upper bound: 0.0003021
RS_RSZ2_RSZ1_RSZ1_RSZ2_RSZ1_RSZ1, status: Status.UNKNOWN, split count: 6, time: 2.51
Output dim: 1, lower bound: -0.0003011, upper bound: 0.0002983
RS_RSZ2_RSZ1_RSZ1_RSZ2_RSZ1_RSZ2, status: Status.UNKNOWN, split count: 6, time: 2.51
Output dim: 1, lower bound: -0.0002967, upper bound: 0.0003029
RS_RSZ2_RSZ1_RSZ1_RSZ2_RSZ2_RSZ1, status: Status.UNKNOWN, split count: 6, time: 2.51
Output dim: 1, lower bound: -0.0003011, upper bound: 0.0002983
RS_RSZ2_RSZ1_RSZ1_RSZ2_RSZ2_RSZ2, status: Status.UNKNOWN, split count: 6, time: 2.51
Output dim: 1, lower bound: -0.0002967, upper bound: 0.0003029
RS_RSZ2_RSZ1_RSZ2_RSZ1_RSZ1_RSZ1, status: Status.UNKNOWN, split count: 6, time: 2.51
Output dim: 1, lower bound: -0.0003029, upper bound: 0.0002968
RS_RSZ2_RSZ1_RSZ2_RSZ1_RSZ1_RSZ2, status: Status.UNKNOWN, split count: 6, time: 2.51
Output dim: 1, lower bound: -0.0002982, upper bound: 0.0003013
RS_RSZ2_RSZ1_RSZ2_RSZ1_RSZ2_RSZ1, status: Status.UNKNOWN, split count: 6, time: 2.51
Output dim: 1, lower bound: -0.0003029, upper bound: 0.0002968
RS_RSZ2_RSZ1_RSZ2_RSZ1_RSZ2_RSZ2, status: Status.UNKNOWN, split count: 6, time: 2.51
Output dim: 1, lower bound: -0.0002982, upper bound: 0.0003013
RS_RSZ2_RSZ1_RSZ2_RSZ2_RSZ1_RSZ1, status: Status.UNKNOWN, split count: 6, time: 2.51
Output dim: 1, lower bound: -0.0003018, upper bound: 0.0002975
RS_RSZ2_RSZ1_RSZ2_RSZ2_RSZ1_RSZ2, status: Status.UNKNOWN, split count: 6, time: 2.51
Output dim: 1, lower bound: -0.0002972, upper bound: 0.0003019
RS_RSZ2_RSZ1_RSZ2_RSZ2_RSZ2_RSZ1, status: Status.UNKNOWN, split count: 6, time: 2.51
Output dim: 1, lower bound: -0.0003018, upper bound: 0.0002977
RS_RSZ2_RSZ1_RSZ2_RSZ2_RSZ2_RSZ2, status: Status.UNKNOWN, split count: 6, time: 2.51
Output dim: 1, lower bound: -0.0002972, upper bound: 0.0003020
RS_RSZ2_RSZ2_RSZ1_RSZ1_RSZ1_RSZ1, status: Status.UNKNOWN, split count: 6, time: 2.51
Output dim: 1, lower bound: -0.0003020, upper bound: 0.0002972
RS_RSZ2_RSZ2_RSZ1_RSZ1_RSZ1_RSZ2, status: Status.UNKNOWN, split count: 6, time: 2.51
Output dim: 1, lower bound: -0.0002977, upper bound: 0.0003018
RS_RSZ2_RSZ2_RSZ1_RSZ1_RSZ2_RSZ1, status: Status.UNKNOWN, split count: 6, time: 2.51
Output dim: 1, lower bound: -0.0003019, upper bound: 0.0002972
RS_RSZ2_RSZ2_RSZ1_RSZ1_RSZ2_RSZ2, status: Status.UNKNOWN, split count: 6, time: 2.51
Output dim: 1, lower bound: -0.0002975, upper bound: 0.0003018
RS_RSZ2_RSZ2_RSZ1_RSZ2_RSZ1_RSZ1, status: Status.UNKNOWN, split count: 6, time: 2.51
Output dim: 1, lower bound: -0.0003013, upper bound: 0.0002982
RS_RSZ2_RSZ2_RSZ1_RSZ2_RSZ1_RSZ2, status: Status.UNKNOWN, split count: 6, time: 2.51
Output dim: 1, lower bound: -0.0002968, upper bound: 0.0003029
RS_RSZ2_RSZ2_RSZ1_RSZ2_RSZ2_RSZ1, status: Status.UNKNOWN, split count: 6, time: 2.51
Output dim: 1, lower bound: -0.0003013, upper bound: 0.0002982
RS_RSZ2_RSZ2_RSZ1_RSZ2_RSZ2_RSZ2, status: Status.UNKNOWN, split count: 6, time: 2.51
Output dim: 1, lower bound: -0.0002968, upper bound: 0.0003029
RS_RSZ2_RSZ2_RSZ2_RSZ1_RSZ1_RSZ1, status: Status.UNKNOWN, split count: 6, time: 2.51
Output dim: 1, lower bound: -0.0003029, upper bound: 0.0002967
RS_RSZ2_RSZ2_RSZ2_RSZ1_RSZ1_RSZ2, status: Status.UNKNOWN, split count: 6, time: 2.51
Output dim: 1, lower bound: -0.0002983, upper bound: 0.0003011
RS_RSZ2_RSZ2_RSZ2_RSZ1_RSZ2_RSZ1, status: Status.UNKNOWN, split count: 6, time: 2.51
Output dim: 1, lower bound: -0.0003029, upper bound: 0.0002967
RS_RSZ2_RSZ2_RSZ2_RSZ1_RSZ2_RSZ2, status: Status.UNKNOWN, split count: 6, time: 2.51
Output dim: 1, lower bound: -0.0002983, upper bound: 0.0003011
RS_RSZ2_RSZ2_RSZ2_RSZ2_RSZ1_RSZ1, status: Status.UNKNOWN, split count: 6, time: 2.51
Output dim: 1, lower bound: -0.0003021, upper bound: 0.0002975
RS_RSZ2_RSZ2_RSZ2_RSZ2_RSZ1_RSZ2, status: Status.UNKNOWN, split count: 6, time: 2.51
Output dim: 1, lower bound: -0.0002976, upper bound: 0.0003019
RS_RSZ2_RSZ2_RSZ2_RSZ2_RSZ2_RSZ1, status: Status.UNKNOWN, split count: 6, time: 2.51
Output dim: 1, lower bound: -0.0003021, upper bound: 0.0002977
RS_RSZ2_RSZ2_RSZ2_RSZ2_RSZ2_RSZ2, status: Status.UNKNOWN, split count: 6, time: 2.51
Output dim: 1, lower bound: -0.0002976, upper bound: 0.0003020

## BFS RS instance: RS_RSZ1_RSZ1_RSZ1_RSZ1_RSZ1_RSZ1

### Backsubstitution after applying RS history:
0: 0.0009457, 0.0011003, 0.0009457, 0.0011003, -0.0001488, 0.0001493
1: 0.9936197, 0.9940771, 0.9936197, 0.9940771, -0.0003923, 0.0003857
2: -0.0065606, -0.0050208, -0.0065606, -0.0050208, -0.0011302, 0.0010614
3: 0.0038004, 0.0040140, 0.0038004, 0.0040140, -0.0001638, 0.0001683
4: 0.0023852, 0.0036022, 0.0023852, 0.0036022, -0.0009259, 0.0009803
5: 0.0060086, 0.0065111, 0.0060086, 0.0065111, -0.0005024, 0.0005024
6: -0.0013983, -0.0008639, -0.0013983, -0.0008639, -0.0003872, 0.0003633
7: -0.0083221, -0.0079387, -0.0083221, -0.0079387, -0.0003835, 0.0003835
8: 0.0048484, 0.0068715, 0.0048484, 0.0068715, -0.0011549, 0.0012453
9: -0.0036877, -0.0032079, -0.0036877, -0.0032079, -0.0004797, 0.0004797

### Unstable ReLU Count (Linear/Conv2D Layers)
- layer_idx=0, type=LayerType.Linear, total=256, inp1_unstable=5, inp2_unstable=5, delta_unstable=251
- layer_idx=2, type=LayerType.Linear, total=256, inp1_unstable=4, inp2_unstable=4, delta_unstable=255
- layer_idx=4, type=LayerType.Linear, total=256, inp1_unstable=0, inp2_unstable=0, delta_unstable=256
- layer_idx=6, type=LayerType.Linear, total=256, inp1_unstable=1, inp2_unstable=1, delta_unstable=256
- layer_idx=8, type=LayerType.Linear, total=10, inp1_unstable=0, inp2_unstable=0, delta_unstable=10

Time for backsubstitution: 1.19 seconds

### RS candidates at layer 3
type: RSZ, layer: 3, pos: 80
type: RSZ, layer: 3, pos: 81
type: RSZ, layer: 3, pos: 136

Time for candidate selection: 0.15 seconds

### Candidate
type: RSZ, layer: 3, pos: 80

### Relational analysis RSZ of RS_RSZ1_RSZ1_RSZ1_RSZ1_RSZ1_RSZ1_RSZ1

#### Relational analysis RSZ result of RS_RSZ1_RSZ1_RSZ1_RSZ1_RSZ1_RSZ1_RSZ1
Status: Status.UNKNOWN
Output dim: 1, lower bound: -0.0002842, upper bound: 0.0002781
time: 0.56 seconds

### Relational analysis RSZ of RS_RSZ1_RSZ1_RSZ1_RSZ1_RSZ1_RSZ1_RSZ2

#### Relational analysis RSZ result of RS_RSZ1_RSZ1_RSZ1_RSZ1_RSZ1_RSZ1_RSZ2
Status: Status.UNKNOWN
Output dim: 1, lower bound: -0.0002833, upper bound: 0.0002804
time: 0.57 seconds

## BFS RS instance: RS_RSZ1_RSZ1_RSZ1_RSZ1_RSZ1_RSZ2

### Backsubstitution after applying RS history:
0: 0.0009457, 0.0011003, 0.0009457, 0.0011003, -0.0001494, 0.0001489
1: 0.9936197, 0.9940771, 0.9936197, 0.9940771, -0.0003861, 0.0003926
2: -0.0065606, -0.0050208, -0.0065606, -0.0050208, -0.0010652, 0.0011337
3: 0.0038004, 0.0040140, 0.0038004, 0.0040140, -0.0001686, 0.0001640
4: 0.0023852, 0.0036022, 0.0023852, 0.0036022, -0.0009830, 0.0009289
5: 0.0060086, 0.0065111, 0.0060086, 0.0065111, -0.0005024, 0.0005024
6: -0.0013983, -0.0008639, -0.0013983, -0.0008639, -0.0003646, 0.0003884
7: -0.0083221, -0.0079387, -0.0083221, -0.0079387, -0.0003835, 0.0003835
8: 0.0048484, 0.0068715, 0.0048484, 0.0068715, -0.0012499, 0.0011599
9: -0.0036877, -0.0032079, -0.0036877, -0.0032079, -0.0004797, 0.0004797

### Unstable ReLU Count (Linear/Conv2D Layers)
- layer_idx=0, type=LayerType.Linear, total=256, inp1_unstable=5, inp2_unstable=5, delta_unstable=251
- layer_idx=2, type=LayerType.Linear, total=256, inp1_unstable=4, inp2_unstable=4, delta_unstable=255
- layer_idx=4, type=LayerType.Linear, total=256, inp1_unstable=0, inp2_unstable=0, delta_unstable=256
- layer_idx=6, type=LayerType.Linear, total=256, inp1_unstable=1, inp2_unstable=1, delta_unstable=256
- layer_idx=8, type=LayerType.Linear, total=10, inp1_unstable=0, inp2_unstable=0, delta_unstable=10

Time for backsubstitution: 1.15 seconds

### RS candidates at layer 3
type: RSZ, layer: 3, pos: 80
type: RSZ, layer: 3, pos: 81
type: RSZ, layer: 3, pos: 136

Time for candidate selection: 0.12 seconds

### Candidate
type: RSZ, layer: 3, pos: 80

### Relational analysis RSZ of RS_RSZ1_RSZ1_RSZ1_RSZ1_RSZ1_RSZ2_RSZ1

#### Relational analysis RSZ result of RS_RSZ1_RSZ1_RSZ1_RSZ1_RSZ1_RSZ2_RSZ1
Status: Status.UNKNOWN
Output dim: 1, lower bound: -0.0002803, upper bound: 0.0002828
time: 0.55 seconds

### Relational analysis RSZ of RS_RSZ1_RSZ1_RSZ1_RSZ1_RSZ1_RSZ2_RSZ2

#### Relational analysis RSZ result of RS_RSZ1_RSZ1_RSZ1_RSZ1_RSZ1_RSZ2_RSZ2
Status: Status.UNKNOWN
Output dim: 1, lower bound: -0.0002785, upper bound: 0.0002841
time: 0.57 seconds

## BFS RS instance: RS_RSZ1_RSZ1_RSZ1_RSZ1_RSZ2_RSZ1

### Backsubstitution after applying RS history:
0: 0.0009457, 0.0011003, 0.0009457, 0.0011003, -0.0001488, 0.0001493
1: 0.9936197, 0.9940771, 0.9936197, 0.9940771, -0.0003923, 0.0003859
2: -0.0065606, -0.0050208, -0.0065606, -0.0050208, -0.0011302, 0.0010633
3: 0.0038004, 0.0040140, 0.0038004, 0.0040140, -0.0001639, 0.0001683
4: 0.0023852, 0.0036022, 0.0023852, 0.0036022, -0.0009273, 0.0009802
5: 0.0060086, 0.0065111, 0.0060086, 0.0065111, -0.0005024, 0.0005024
6: -0.0013983, -0.0008639, -0.0013983, -0.0008639, -0.0003872, 0.0003640
7: -0.0083221, -0.0079387, -0.0083221, -0.0079387, -0.0003835, 0.0003835
8: 0.0048484, 0.0068715, 0.0048484, 0.0068715, -0.0011574, 0.0012453
9: -0.0036877, -0.0032079, -0.0036877, -0.0032079, -0.0004797, 0.0004797

### Unstable ReLU Count (Linear/Conv2D Layers)
- layer_idx=0, type=LayerType.Linear, total=256, inp1_unstable=5, inp2_unstable=5, delta_unstable=251
- layer_idx=2, type=LayerType.Linear, total=256, inp1_unstable=4, inp2_unstable=4, delta_unstable=255
- layer_idx=4, type=LayerType.Linear, total=256, inp1_unstable=0, inp2_unstable=0, delta_unstable=256
- layer_idx=6, type=LayerType.Linear, total=256, inp1_unstable=1, inp2_unstable=1, delta_unstable=256
- layer_idx=8, type=LayerType.Linear, total=10, inp1_unstable=0, inp2_unstable=0, delta_unstable=10

Time for backsubstitution: 1.18 seconds

### RS candidates at layer 3
type: RSZ, layer: 3, pos: 80
type: RSZ, layer: 3, pos: 81
type: RSZ, layer: 3, pos: 136

Time for candidate selection: 0.13 seconds

### Candidate
type: RSZ, layer: 3, pos: 80

### Relational analysis RSZ of RS_RSZ1_RSZ1_RSZ1_RSZ1_RSZ2_RSZ1_RSZ1

#### Relational analysis RSZ result of RS_RSZ1_RSZ1_RSZ1_RSZ1_RSZ2_RSZ1_RSZ1
Status: Status.UNKNOWN
Output dim: 1, lower bound: -0.0002840, upper bound: 0.0002779
time: 0.57 seconds

### Relational analysis RSZ of RS_RSZ1_RSZ1_RSZ1_RSZ1_RSZ2_RSZ1_RSZ2

#### Relational analysis RSZ result of RS_RSZ1_RSZ1_RSZ1_RSZ1_RSZ2_RSZ1_RSZ2
Status: Status.UNKNOWN
Output dim: 1, lower bound: -0.0002833, upper bound: 0.0002804
time: 0.55 seconds

## BFS RS instance: RS_RSZ1_RSZ1_RSZ1_RSZ1_RSZ2_RSZ2

### Backsubstitution after applying RS history:
0: 0.0009457, 0.0011003, 0.0009457, 0.0011003, -0.0001494, 0.0001488
1: 0.9936197, 0.9940771, 0.9936197, 0.9940771, -0.0003858, 0.0003926
2: -0.0065606, -0.0050208, -0.0065606, -0.0050208, -0.0010623, 0.0011333
3: 0.0038004, 0.0040140, 0.0038004, 0.0040140, -0.0001685, 0.0001638
4: 0.0023852, 0.0036022, 0.0023852, 0.0036022, -0.0009827, 0.0009266
5: 0.0060086, 0.0065111, 0.0060086, 0.0065111, -0.0005024, 0.0005024
6: -0.0013983, -0.0008639, -0.0013983, -0.0008639, -0.0003636, 0.0003883
7: -0.0083221, -0.0079387, -0.0083221, -0.0079387, -0.0003835, 0.0003835
8: 0.0048484, 0.0068715, 0.0048484, 0.0068715, -0.0012494, 0.0011561
9: -0.0036877, -0.0032079, -0.0036877, -0.0032079, -0.0004797, 0.0004797

### Unstable ReLU Count (Linear/Conv2D Layers)
- layer_idx=0, type=LayerType.Linear, total=256, inp1_unstable=5, inp2_unstable=5, delta_unstable=251
- layer_idx=2, type=LayerType.Linear, total=256, inp1_unstable=4, inp2_unstable=4, delta_unstable=255
- layer_idx=4, type=LayerType.Linear, total=256, inp1_unstable=0, inp2_unstable=0, delta_unstable=256
- layer_idx=6, type=LayerType.Linear, total=256, inp1_unstable=1, inp2_unstable=1, delta_unstable=256
- layer_idx=8, type=LayerType.Linear, total=10, inp1_unstable=0, inp2_unstable=0, delta_unstable=10

Time for backsubstitution: 1.15 seconds

### RS candidates at layer 3
type: RSZ, layer: 3, pos: 80
type: RSZ, layer: 3, pos: 81
type: RSZ, layer: 3, pos: 136

Time for candidate selection: 0.12 seconds

### Candidate
type: RSZ, layer: 3, pos: 80

### Relational analysis RSZ of RS_RSZ1_RSZ1_RSZ1_RSZ1_RSZ2_RSZ2_RSZ1

#### Relational analysis RSZ result of RS_RSZ1_RSZ1_RSZ1_RSZ1_RSZ2_RSZ2_RSZ1
Status: Status.UNKNOWN
Output dim: 1, lower bound: -0.0002802, upper bound: 0.0002827
time: 0.61 seconds

### Relational analysis RSZ of RS_RSZ1_RSZ1_RSZ1_RSZ1_RSZ2_RSZ2_RSZ2

#### Relational analysis RSZ result of RS_RSZ1_RSZ1_RSZ1_RSZ1_RSZ2_RSZ2_RSZ2
Status: Status.UNKNOWN
Output dim: 1, lower bound: -0.0002785, upper bound: 0.0002840
time: 0.59 seconds

## BFS RS instance: RS_RSZ1_RSZ1_RSZ1_RSZ2_RSZ1_RSZ1

### Backsubstitution after applying RS history:
0: 0.0009457, 0.0011003, 0.0009457, 0.0011003, -0.0001488, 0.0001493
1: 0.9936197, 0.9940771, 0.9936197, 0.9940771, -0.0003919, 0.0003858
2: -0.0065606, -0.0050208, -0.0065606, -0.0050208, -0.0011265, 0.0010619
3: 0.0038004, 0.0040140, 0.0038004, 0.0040140, -0.0001638, 0.0001681
4: 0.0023852, 0.0036022, 0.0023852, 0.0036022, -0.0009262, 0.0009773
5: 0.0060086, 0.0065111, 0.0060086, 0.0065111, -0.0005024, 0.0005024
6: -0.0013983, -0.0008639, -0.0013983, -0.0008639, -0.0003859, 0.0003635
7: -0.0083221, -0.0079387, -0.0083221, -0.0079387, -0.0003835, 0.0003835
8: 0.0048484, 0.0068715, 0.0048484, 0.0068715, -0.0011556, 0.0012405
9: -0.0036877, -0.0032079, -0.0036877, -0.0032079, -0.0004797, 0.0004797

### Unstable ReLU Count (Linear/Conv2D Layers)
- layer_idx=0, type=LayerType.Linear, total=256, inp1_unstable=5, inp2_unstable=5, delta_unstable=251
- layer_idx=2, type=LayerType.Linear, total=256, inp1_unstable=4, inp2_unstable=4, delta_unstable=255
- layer_idx=4, type=LayerType.Linear, total=256, inp1_unstable=0, inp2_unstable=0, delta_unstable=256
- layer_idx=6, type=LayerType.Linear, total=256, inp1_unstable=1, inp2_unstable=1, delta_unstable=256
- layer_idx=8, type=LayerType.Linear, total=10, inp1_unstable=0, inp2_unstable=0, delta_unstable=10

Time for backsubstitution: 1.20 seconds

### RS candidates at layer 3
type: RSZ, layer: 3, pos: 80
type: RSZ, layer: 3, pos: 81
type: RSZ, layer: 3, pos: 136

Time for candidate selection: 0.12 seconds

### Candidate
type: RSZ, layer: 3, pos: 80

### Relational analysis RSZ of RS_RSZ1_RSZ1_RSZ1_RSZ2_RSZ1_RSZ1_RSZ1

#### Relational analysis RSZ result of RS_RSZ1_RSZ1_RSZ1_RSZ2_RSZ1_RSZ1_RSZ1
Status: Status.UNKNOWN
Output dim: 1, lower bound: -0.0002831, upper bound: 0.0002785
time: 0.64 seconds

### Relational analysis RSZ of RS_RSZ1_RSZ1_RSZ1_RSZ2_RSZ1_RSZ1_RSZ2

#### Relational analysis RSZ result of RS_RSZ1_RSZ1_RSZ1_RSZ2_RSZ1_RSZ1_RSZ2
Status: Status.UNKNOWN
Output dim: 1, lower bound: -0.0002826, upper bound: 0.0002810
time: 0.60 seconds

## BFS RS instance: RS_RSZ1_RSZ1_RSZ1_RSZ2_RSZ1_RSZ2

### Backsubstitution after applying RS history:
0: 0.0009457, 0.0011003, 0.0009457, 0.0011003, -0.0001494, 0.0001489
1: 0.9936197, 0.9940771, 0.9936197, 0.9940771, -0.0003863, 0.0003928
2: -0.0065606, -0.0050208, -0.0065606, -0.0050208, -0.0010669, 0.0011360
3: 0.0038004, 0.0040140, 0.0038004, 0.0040140, -0.0001687, 0.0001642
4: 0.0023852, 0.0036022, 0.0023852, 0.0036022, -0.0009848, 0.0009302
5: 0.0060086, 0.0065111, 0.0060086, 0.0065111, -0.0005024, 0.0005024
6: -0.0013983, -0.0008639, -0.0013983, -0.0008639, -0.0003652, 0.0003892
7: -0.0083221, -0.0079387, -0.0083221, -0.0079387, -0.0003835, 0.0003835
8: 0.0048484, 0.0068715, 0.0048484, 0.0068715, -0.0012529, 0.0011621
9: -0.0036877, -0.0032079, -0.0036877, -0.0032079, -0.0004797, 0.0004797

### Unstable ReLU Count (Linear/Conv2D Layers)
- layer_idx=0, type=LayerType.Linear, total=256, inp1_unstable=5, inp2_unstable=5, delta_unstable=251
- layer_idx=2, type=LayerType.Linear, total=256, inp1_unstable=4, inp2_unstable=4, delta_unstable=255
- layer_idx=4, type=LayerType.Linear, total=256, inp1_unstable=0, inp2_unstable=0, delta_unstable=256
- layer_idx=6, type=LayerType.Linear, total=256, inp1_unstable=1, inp2_unstable=1, delta_unstable=256
- layer_idx=8, type=LayerType.Linear, total=10, inp1_unstable=0, inp2_unstable=0, delta_unstable=10

Time for backsubstitution: 1.15 seconds

### RS candidates at layer 3
type: RSZ, layer: 3, pos: 80
type: RSZ, layer: 3, pos: 81
type: RSZ, layer: 3, pos: 136

Time for candidate selection: 0.11 seconds

### Candidate
type: RSZ, layer: 3, pos: 80

### Relational analysis RSZ of RS_RSZ1_RSZ1_RSZ1_RSZ2_RSZ1_RSZ2_RSZ1

#### Relational analysis RSZ result of RS_RSZ1_RSZ1_RSZ1_RSZ2_RSZ1_RSZ2_RSZ1
Status: Status.UNKNOWN
Output dim: 1, lower bound: -0.0002796, upper bound: 0.0002835
time: 0.57 seconds

### Relational analysis RSZ of RS_RSZ1_RSZ1_RSZ1_RSZ2_RSZ1_RSZ2_RSZ2

#### Relational analysis RSZ result of RS_RSZ1_RSZ1_RSZ1_RSZ2_RSZ1_RSZ2_RSZ2
Status: Status.UNKNOWN
Output dim: 1, lower bound: -0.0002779, upper bound: 0.0002850
time: 0.58 seconds

## BFS RS instance: RS_RSZ1_RSZ1_RSZ1_RSZ2_RSZ2_RSZ1

### Backsubstitution after applying RS history:
0: 0.0009457, 0.0011003, 0.0009457, 0.0011003, -0.0001489, 0.0001493
1: 0.9936197, 0.9940771, 0.9936197, 0.9940771, -0.0003919, 0.0003860
2: -0.0065606, -0.0050208, -0.0065606, -0.0050208, -0.0011266, 0.0010644
3: 0.0038004, 0.0040140, 0.0038004, 0.0040140, -0.0001640, 0.0001681
4: 0.0023852, 0.0036022, 0.0023852, 0.0036022, -0.0009282, 0.0009774
5: 0.0060086, 0.0065111, 0.0060086, 0.0065111, -0.0005024, 0.0005024
6: -0.0013983, -0.0008639, -0.0013983, -0.0008639, -0.0003860, 0.0003644
7: -0.0083221, -0.0079387, -0.0083221, -0.0079387, -0.0003835, 0.0003835
8: 0.0048484, 0.0068715, 0.0048484, 0.0068715, -0.0011589, 0.0012406
9: -0.0036877, -0.0032079, -0.0036877, -0.0032079, -0.0004797, 0.0004797

### Unstable ReLU Count (Linear/Conv2D Layers)
- layer_idx=0, type=LayerType.Linear, total=256, inp1_unstable=5, inp2_unstable=5, delta_unstable=251
- layer_idx=2, type=LayerType.Linear, total=256, inp1_unstable=4, inp2_unstable=4, delta_unstable=255
- layer_idx=4, type=LayerType.Linear, total=256, inp1_unstable=0, inp2_unstable=0, delta_unstable=256
- layer_idx=6, type=LayerType.Linear, total=256, inp1_unstable=1, inp2_unstable=1, delta_unstable=256
- layer_idx=8, type=LayerType.Linear, total=10, inp1_unstable=0, inp2_unstable=0, delta_unstable=10

Time for backsubstitution: 1.18 seconds

### RS candidates at layer 3
type: RSZ, layer: 3, pos: 80
type: RSZ, layer: 3, pos: 81
type: RSZ, layer: 3, pos: 136

Time for candidate selection: 0.12 seconds

### Candidate
type: RSZ, layer: 3, pos: 80

### Relational analysis RSZ of RS_RSZ1_RSZ1_RSZ1_RSZ2_RSZ2_RSZ1_RSZ1

#### Relational analysis RSZ result of RS_RSZ1_RSZ1_RSZ1_RSZ2_RSZ2_RSZ1_RSZ1
Status: Status.UNKNOWN
Output dim: 1, lower bound: -0.0002831, upper bound: 0.0002785
time: 0.62 seconds

### Relational analysis RSZ of RS_RSZ1_RSZ1_RSZ1_RSZ2_RSZ2_RSZ1_RSZ2

#### Relational analysis RSZ result of RS_RSZ1_RSZ1_RSZ1_RSZ2_RSZ2_RSZ1_RSZ2
Status: Status.UNKNOWN
Output dim: 1, lower bound: -0.0002827, upper bound: 0.0002810
time: 0.59 seconds

## BFS RS instance: RS_RSZ1_RSZ1_RSZ1_RSZ2_RSZ2_RSZ2

### Backsubstitution after applying RS history:
0: 0.0009457, 0.0011003, 0.0009457, 0.0011003, -0.0001494, 0.0001489
1: 0.9936197, 0.9940771, 0.9936197, 0.9940771, -0.0003861, 0.0003928
2: -0.0065606, -0.0050208, -0.0065606, -0.0050208, -0.0010653, 0.0011357
3: 0.0038004, 0.0040140, 0.0038004, 0.0040140, -0.0001687, 0.0001640
4: 0.0023852, 0.0036022, 0.0023852, 0.0036022, -0.0009846, 0.0009290
5: 0.0060086, 0.0065111, 0.0060086, 0.0065111, -0.0005024, 0.0005024
6: -0.0013983, -0.0008639, -0.0013983, -0.0008639, -0.0003647, 0.0003891
7: -0.0083221, -0.0079387, -0.0083221, -0.0079387, -0.0003835, 0.0003835
8: 0.0048484, 0.0068715, 0.0048484, 0.0068715, -0.0012526, 0.0011601
9: -0.0036877, -0.0032079, -0.0036877, -0.0032079, -0.0004797, 0.0004797

### Unstable ReLU Count (Linear/Conv2D Layers)
- layer_idx=0, type=LayerType.Linear, total=256, inp1_unstable=5, inp2_unstable=5, delta_unstable=251
- layer_idx=2, type=LayerType.Linear, total=256, inp1_unstable=4, inp2_unstable=4, delta_unstable=255
- layer_idx=4, type=LayerType.Linear, total=256, inp1_unstable=0, inp2_unstable=0, delta_unstable=256
- layer_idx=6, type=LayerType.Linear, total=256, inp1_unstable=1, inp2_unstable=1, delta_unstable=256
- layer_idx=8, type=LayerType.Linear, total=10, inp1_unstable=0, inp2_unstable=0, delta_unstable=10

Time for backsubstitution: 1.18 seconds

### RS candidates at layer 3
type: RSZ, layer: 3, pos: 80
type: RSZ, layer: 3, pos: 81
type: RSZ, layer: 3, pos: 136

Time for candidate selection: 0.15 seconds

### Candidate
type: RSZ, layer: 3, pos: 80

### Relational analysis RSZ of RS_RSZ1_RSZ1_RSZ1_RSZ2_RSZ2_RSZ2_RSZ1

#### Relational analysis RSZ result of RS_RSZ1_RSZ1_RSZ1_RSZ2_RSZ2_RSZ2_RSZ1
Status: Status.UNKNOWN
Output dim: 1, lower bound: -0.0002795, upper bound: 0.0002835
time: 0.57 seconds

### Relational analysis RSZ of RS_RSZ1_RSZ1_RSZ1_RSZ2_RSZ2_RSZ2_RSZ2

#### Relational analysis RSZ result of RS_RSZ1_RSZ1_RSZ1_RSZ2_RSZ2_RSZ2_RSZ2
Status: Status.UNKNOWN
Output dim: 1, lower bound: -0.0002779, upper bound: 0.0002850
time: 0.59 seconds

## BFS RS instance: RS_RSZ1_RSZ1_RSZ2_RSZ1_RSZ1_RSZ1

### Backsubstitution after applying RS history:
0: 0.0009457, 0.0011003, 0.0009457, 0.0011003, -0.0001488, 0.0001494
1: 0.9936197, 0.9940771, 0.9936197, 0.9940771, -0.0003924, 0.0003857
2: -0.0065606, -0.0050208, -0.0065606, -0.0050208, -0.0011318, 0.0010607
3: 0.0038004, 0.0040140, 0.0038004, 0.0040140, -0.0001637, 0.0001685
4: 0.0023852, 0.0036022, 0.0023852, 0.0036022, -0.0009253, 0.0009815
5: 0.0060086, 0.0065111, 0.0060086, 0.0065111, -0.0005024, 0.0005024
6: -0.0013983, -0.0008639, -0.0013983, -0.0008639, -0.0003878, 0.0003631
7: -0.0083221, -0.0079387, -0.0083221, -0.0079387, -0.0003835, 0.0003835
8: 0.0048484, 0.0068715, 0.0048484, 0.0068715, -0.0011539, 0.0012474
9: -0.0036877, -0.0032079, -0.0036877, -0.0032079, -0.0004797, 0.0004797

### Unstable ReLU Count (Linear/Conv2D Layers)
- layer_idx=0, type=LayerType.Linear, total=256, inp1_unstable=5, inp2_unstable=5, delta_unstable=251
- layer_idx=2, type=LayerType.Linear, total=256, inp1_unstable=4, inp2_unstable=4, delta_unstable=255
- layer_idx=4, type=LayerType.Linear, total=256, inp1_unstable=0, inp2_unstable=0, delta_unstable=256
- layer_idx=6, type=LayerType.Linear, total=256, inp1_unstable=1, inp2_unstable=1, delta_unstable=256
- layer_idx=8, type=LayerType.Linear, total=10, inp1_unstable=0, inp2_unstable=0, delta_unstable=10

Time for backsubstitution: 1.15 seconds

### RS candidates at layer 3
type: RSZ, layer: 3, pos: 80
type: RSZ, layer: 3, pos: 81
type: RSZ, layer: 3, pos: 136

Time for candidate selection: 0.12 seconds

### Candidate
type: RSZ, layer: 3, pos: 80

### Relational analysis RSZ of RS_RSZ1_RSZ1_RSZ2_RSZ1_RSZ1_RSZ1_RSZ1

#### Relational analysis RSZ result of RS_RSZ1_RSZ1_RSZ2_RSZ1_RSZ1_RSZ1_RSZ1
Status: Status.UNKNOWN
Output dim: 1, lower bound: -0.0002850, upper bound: 0.0002779
time: 0.62 seconds

### Relational analysis RSZ of RS_RSZ1_RSZ1_RSZ2_RSZ1_RSZ1_RSZ1_RSZ2

#### Relational analysis RSZ result of RS_RSZ1_RSZ1_RSZ2_RSZ1_RSZ1_RSZ1_RSZ2
Status: Status.UNKNOWN
Output dim: 1, lower bound: -0.0002835, upper bound: 0.0002796
time: 0.59 seconds

## BFS RS instance: RS_RSZ1_RSZ1_RSZ2_RSZ1_RSZ1_RSZ2

### Backsubstitution after applying RS history:
0: 0.0009457, 0.0011003, 0.0009457, 0.0011003, -0.0001493, 0.0001489
1: 0.9936197, 0.9940771, 0.9936197, 0.9940771, -0.0003865, 0.0003923
2: -0.0065606, -0.0050208, -0.0065606, -0.0050208, -0.0010689, 0.0011302
3: 0.0038004, 0.0040140, 0.0038004, 0.0040140, -0.0001683, 0.0001643
4: 0.0023852, 0.0036022, 0.0023852, 0.0036022, -0.0009802, 0.0009318
5: 0.0060086, 0.0065111, 0.0060086, 0.0065111, -0.0005024, 0.0005024
6: -0.0013983, -0.0008639, -0.0013983, -0.0008639, -0.0003659, 0.0003872
7: -0.0083221, -0.0079387, -0.0083221, -0.0079387, -0.0003835, 0.0003835
8: 0.0048484, 0.0068715, 0.0048484, 0.0068715, -0.0012453, 0.0011648
9: -0.0036877, -0.0032079, -0.0036877, -0.0032079, -0.0004797, 0.0004797

### Unstable ReLU Count (Linear/Conv2D Layers)
- layer_idx=0, type=LayerType.Linear, total=256, inp1_unstable=5, inp2_unstable=5, delta_unstable=251
- layer_idx=2, type=LayerType.Linear, total=256, inp1_unstable=4, inp2_unstable=4, delta_unstable=255
- layer_idx=4, type=LayerType.Linear, total=256, inp1_unstable=0, inp2_unstable=0, delta_unstable=256
- layer_idx=6, type=LayerType.Linear, total=256, inp1_unstable=1, inp2_unstable=1, delta_unstable=256
- layer_idx=8, type=LayerType.Linear, total=10, inp1_unstable=0, inp2_unstable=0, delta_unstable=10

Time for backsubstitution: 1.15 seconds

### RS candidates at layer 3
type: RSZ, layer: 3, pos: 80
type: RSZ, layer: 3, pos: 81
type: RSZ, layer: 3, pos: 136

Time for candidate selection: 0.12 seconds

### Candidate
type: RSZ, layer: 3, pos: 80

### Relational analysis RSZ of RS_RSZ1_RSZ1_RSZ2_RSZ1_RSZ1_RSZ2_RSZ1

#### Relational analysis RSZ result of RS_RSZ1_RSZ1_RSZ2_RSZ1_RSZ1_RSZ2_RSZ1
Status: Status.UNKNOWN
Output dim: 1, lower bound: -0.0002810, upper bound: 0.0002827
time: 0.58 seconds

### Relational analysis RSZ of RS_RSZ1_RSZ1_RSZ2_RSZ1_RSZ1_RSZ2_RSZ2

#### Relational analysis RSZ result of RS_RSZ1_RSZ1_RSZ2_RSZ1_RSZ1_RSZ2_RSZ2
Status: Status.UNKNOWN
Output dim: 1, lower bound: -0.0002785, upper bound: 0.0002832
time: 0.59 seconds

## BFS RS instance: RS_RSZ1_RSZ1_RSZ2_RSZ1_RSZ2_RSZ1

### Backsubstitution after applying RS history:
0: 0.0009457, 0.0011003, 0.0009457, 0.0011003, -0.0001488, 0.0001494
1: 0.9936197, 0.9940771, 0.9936197, 0.9940771, -0.0003924, 0.0003858
2: -0.0065606, -0.0050208, -0.0065606, -0.0050208, -0.0011319, 0.0010621
3: 0.0038004, 0.0040140, 0.0038004, 0.0040140, -0.0001638, 0.0001685
4: 0.0023852, 0.0036022, 0.0023852, 0.0036022, -0.0009264, 0.0009816
5: 0.0060086, 0.0065111, 0.0060086, 0.0065111, -0.0005024, 0.0005024
6: -0.0013983, -0.0008639, -0.0013983, -0.0008639, -0.0003878, 0.0003636
7: -0.0083221, -0.0079387, -0.0083221, -0.0079387, -0.0003835, 0.0003835
8: 0.0048484, 0.0068715, 0.0048484, 0.0068715, -0.0011559, 0.0012476
9: -0.0036877, -0.0032079, -0.0036877, -0.0032079, -0.0004797, 0.0004797

### Unstable ReLU Count (Linear/Conv2D Layers)
- layer_idx=0, type=LayerType.Linear, total=256, inp1_unstable=5, inp2_unstable=5, delta_unstable=251
- layer_idx=2, type=LayerType.Linear, total=256, inp1_unstable=4, inp2_unstable=4, delta_unstable=255
- layer_idx=4, type=LayerType.Linear, total=256, inp1_unstable=0, inp2_unstable=0, delta_unstable=256
- layer_idx=6, type=LayerType.Linear, total=256, inp1_unstable=1, inp2_unstable=1, delta_unstable=256
- layer_idx=8, type=LayerType.Linear, total=10, inp1_unstable=0, inp2_unstable=0, delta_unstable=10

Time for backsubstitution: 1.17 seconds

### RS candidates at layer 3
type: RSZ, layer: 3, pos: 80
type: RSZ, layer: 3, pos: 81
type: RSZ, layer: 3, pos: 136

Time for candidate selection: 0.13 seconds

### Candidate
type: RSZ, layer: 3, pos: 80

### Relational analysis RSZ of RS_RSZ1_RSZ1_RSZ2_RSZ1_RSZ2_RSZ1_RSZ1

#### Relational analysis RSZ result of RS_RSZ1_RSZ1_RSZ2_RSZ1_RSZ2_RSZ1_RSZ1
Status: Status.UNKNOWN
Output dim: 1, lower bound: -0.0002849, upper bound: 0.0002779
time: 0.59 seconds

### Relational analysis RSZ of RS_RSZ1_RSZ1_RSZ2_RSZ1_RSZ2_RSZ1_RSZ2

#### Relational analysis RSZ result of RS_RSZ1_RSZ1_RSZ2_RSZ1_RSZ2_RSZ1_RSZ2
Status: Status.UNKNOWN
Output dim: 1, lower bound: -0.0002835, upper bound: 0.0002797
time: 0.58 seconds

## BFS RS instance: RS_RSZ1_RSZ1_RSZ2_RSZ1_RSZ2_RSZ2

### Backsubstitution after applying RS history:
0: 0.0009457, 0.0011003, 0.0009457, 0.0011003, -0.0001493, 0.0001489
1: 0.9936197, 0.9940771, 0.9936197, 0.9940771, -0.0003862, 0.0003923
2: -0.0065606, -0.0050208, -0.0065606, -0.0050208, -0.0010660, 0.0011301
3: 0.0038004, 0.0040140, 0.0038004, 0.0040140, -0.0001683, 0.0001641
4: 0.0023852, 0.0036022, 0.0023852, 0.0036022, -0.0009802, 0.0009295
5: 0.0060086, 0.0065111, 0.0060086, 0.0065111, -0.0005024, 0.0005024
6: -0.0013983, -0.0008639, -0.0013983, -0.0008639, -0.0003649, 0.0003872
7: -0.0083221, -0.0079387, -0.0083221, -0.0079387, -0.0003835, 0.0003835
8: 0.0048484, 0.0068715, 0.0048484, 0.0068715, -0.0012452, 0.0011609
9: -0.0036877, -0.0032079, -0.0036877, -0.0032079, -0.0004797, 0.0004797

### Unstable ReLU Count (Linear/Conv2D Layers)
- layer_idx=0, type=LayerType.Linear, total=256, inp1_unstable=5, inp2_unstable=5, delta_unstable=251
- layer_idx=2, type=LayerType.Linear, total=256, inp1_unstable=4, inp2_unstable=4, delta_unstable=255
- layer_idx=4, type=LayerType.Linear, total=256, inp1_unstable=0, inp2_unstable=0, delta_unstable=256
- layer_idx=6, type=LayerType.Linear, total=256, inp1_unstable=1, inp2_unstable=1, delta_unstable=256
- layer_idx=8, type=LayerType.Linear, total=10, inp1_unstable=0, inp2_unstable=0, delta_unstable=10

Time for backsubstitution: 1.13 seconds

### RS candidates at layer 3
type: RSZ, layer: 3, pos: 80
type: RSZ, layer: 3, pos: 81
type: RSZ, layer: 3, pos: 136

Time for candidate selection: 0.11 seconds

### Candidate
type: RSZ, layer: 3, pos: 80

### Relational analysis RSZ of RS_RSZ1_RSZ1_RSZ2_RSZ1_RSZ2_RSZ2_RSZ1

#### Relational analysis RSZ result of RS_RSZ1_RSZ1_RSZ2_RSZ1_RSZ2_RSZ2_RSZ1
Status: Status.UNKNOWN
Output dim: 1, lower bound: -0.0002810, upper bound: 0.0002826
time: 0.57 seconds

### Relational analysis RSZ of RS_RSZ1_RSZ1_RSZ2_RSZ1_RSZ2_RSZ2_RSZ2

#### Relational analysis RSZ result of RS_RSZ1_RSZ1_RSZ2_RSZ1_RSZ2_RSZ2_RSZ2
Status: Status.UNKNOWN
Output dim: 1, lower bound: -0.0002785, upper bound: 0.0002832
time: 0.57 seconds

## BFS RS instance: RS_RSZ1_RSZ1_RSZ2_RSZ2_RSZ1_RSZ1

### Backsubstitution after applying RS history:
0: 0.0009457, 0.0011003, 0.0009457, 0.0011003, -0.0001488, 0.0001493
1: 0.9936197, 0.9940771, 0.9936197, 0.9940771, -0.0003923, 0.0003853
2: -0.0065606, -0.0050208, -0.0065606, -0.0050208, -0.0011300, 0.0010565
3: 0.0038004, 0.0040140, 0.0038004, 0.0040140, -0.0001635, 0.0001683
4: 0.0023852, 0.0036022, 0.0023852, 0.0036022, -0.0009220, 0.0009801
5: 0.0060086, 0.0065111, 0.0060086, 0.0065111, -0.0005024, 0.0005024
6: -0.0013983, -0.0008639, -0.0013983, -0.0008639, -0.0003871, 0.0003616
7: -0.0083221, -0.0079387, -0.0083221, -0.0079387, -0.0003835, 0.0003835
8: 0.0048484, 0.0068715, 0.0048484, 0.0068715, -0.0011485, 0.0012451
9: -0.0036877, -0.0032079, -0.0036877, -0.0032079, -0.0004797, 0.0004797

### Unstable ReLU Count (Linear/Conv2D Layers)
- layer_idx=0, type=LayerType.Linear, total=256, inp1_unstable=5, inp2_unstable=5, delta_unstable=251
- layer_idx=2, type=LayerType.Linear, total=256, inp1_unstable=4, inp2_unstable=4, delta_unstable=255
- layer_idx=4, type=LayerType.Linear, total=256, inp1_unstable=0, inp2_unstable=0, delta_unstable=256
- layer_idx=6, type=LayerType.Linear, total=256, inp1_unstable=1, inp2_unstable=1, delta_unstable=256
- layer_idx=8, type=LayerType.Linear, total=10, inp1_unstable=0, inp2_unstable=0, delta_unstable=10

Time for backsubstitution: 1.15 seconds

### RS candidates at layer 3
type: RSZ, layer: 3, pos: 80
type: RSZ, layer: 3, pos: 81
type: RSZ, layer: 3, pos: 136

Time for candidate selection: 0.11 seconds

### Candidate
type: RSZ, layer: 3, pos: 80

### Relational analysis RSZ of RS_RSZ1_RSZ1_RSZ2_RSZ2_RSZ1_RSZ1_RSZ1

#### Relational analysis RSZ result of RS_RSZ1_RSZ1_RSZ2_RSZ2_RSZ1_RSZ1_RSZ1
Status: Status.UNKNOWN
Output dim: 1, lower bound: -0.0002839, upper bound: 0.0002785
time: 0.59 seconds

### Relational analysis RSZ of RS_RSZ1_RSZ1_RSZ2_RSZ2_RSZ1_RSZ1_RSZ2

#### Relational analysis RSZ result of RS_RSZ1_RSZ1_RSZ2_RSZ2_RSZ1_RSZ1_RSZ2
Status: Status.UNKNOWN
Output dim: 1, lower bound: -0.0002827, upper bound: 0.0002802
time: 0.62 seconds

## BFS RS instance: RS_RSZ1_RSZ1_RSZ2_RSZ2_RSZ1_RSZ2

### Backsubstitution after applying RS history:
0: 0.0009457, 0.0011003, 0.0009457, 0.0011003, -0.0001494, 0.0001489
1: 0.9936197, 0.9940771, 0.9936197, 0.9940771, -0.0003863, 0.0003925
2: -0.0065606, -0.0050208, -0.0065606, -0.0050208, -0.0010669, 0.0011322
3: 0.0038004, 0.0040140, 0.0038004, 0.0040140, -0.0001685, 0.0001642
4: 0.0023852, 0.0036022, 0.0023852, 0.0036022, -0.0009818, 0.0009302
5: 0.0060086, 0.0065111, 0.0060086, 0.0065111, -0.0005024, 0.0005024
6: -0.0013983, -0.0008639, -0.0013983, -0.0008639, -0.0003652, 0.0003879
7: -0.0083221, -0.0079387, -0.0083221, -0.0079387, -0.0003835, 0.0003835
8: 0.0048484, 0.0068715, 0.0048484, 0.0068715, -0.0012479, 0.0011622
9: -0.0036877, -0.0032079, -0.0036877, -0.0032079, -0.0004797, 0.0004797

### Unstable ReLU Count (Linear/Conv2D Layers)
- layer_idx=0, type=LayerType.Linear, total=256, inp1_unstable=5, inp2_unstable=5, delta_unstable=251
- layer_idx=2, type=LayerType.Linear, total=256, inp1_unstable=4, inp2_unstable=4, delta_unstable=255
- layer_idx=4, type=LayerType.Linear, total=256, inp1_unstable=0, inp2_unstable=0, delta_unstable=256
- layer_idx=6, type=LayerType.Linear, total=256, inp1_unstable=1, inp2_unstable=1, delta_unstable=256
- layer_idx=8, type=LayerType.Linear, total=10, inp1_unstable=0, inp2_unstable=0, delta_unstable=10

Time for backsubstitution: 1.23 seconds

### RS candidates at layer 3
type: RSZ, layer: 3, pos: 80
type: RSZ, layer: 3, pos: 81
type: RSZ, layer: 3, pos: 136

Time for candidate selection: 0.11 seconds

### Candidate
type: RSZ, layer: 3, pos: 80

### Relational analysis RSZ of RS_RSZ1_RSZ1_RSZ2_RSZ2_RSZ1_RSZ2_RSZ1

#### Relational analysis RSZ result of RS_RSZ1_RSZ1_RSZ2_RSZ2_RSZ1_RSZ2_RSZ1
Status: Status.UNKNOWN
Output dim: 1, lower bound: -0.0002801, upper bound: 0.0002833
time: 0.57 seconds

### Relational analysis RSZ of RS_RSZ1_RSZ1_RSZ2_RSZ2_RSZ1_RSZ2_RSZ2

#### Relational analysis RSZ result of RS_RSZ1_RSZ1_RSZ2_RSZ2_RSZ1_RSZ2_RSZ2
Status: Status.UNKNOWN
Output dim: 1, lower bound: -0.0002779, upper bound: 0.0002840
time: 0.58 seconds

## BFS RS instance: RS_RSZ1_RSZ1_RSZ2_RSZ2_RSZ2_RSZ1

### Backsubstitution after applying RS history:
0: 0.0009457, 0.0011003, 0.0009457, 0.0011003, -0.0001488, 0.0001494
1: 0.9936197, 0.9940771, 0.9936197, 0.9940771, -0.0003923, 0.0003855
2: -0.0065606, -0.0050208, -0.0065606, -0.0050208, -0.0011304, 0.0010589
3: 0.0038004, 0.0040140, 0.0038004, 0.0040140, -0.0001636, 0.0001684
4: 0.0023852, 0.0036022, 0.0023852, 0.0036022, -0.0009239, 0.0009804
5: 0.0060086, 0.0065111, 0.0060086, 0.0065111, -0.0005024, 0.0005024
6: -0.0013983, -0.0008639, -0.0013983, -0.0008639, -0.0003873, 0.0003624
7: -0.0083221, -0.0079387, -0.0083221, -0.0079387, -0.0003835, 0.0003835
8: 0.0048484, 0.0068715, 0.0048484, 0.0068715, -0.0011516, 0.0012456
9: -0.0036877, -0.0032079, -0.0036877, -0.0032079, -0.0004797, 0.0004797

### Unstable ReLU Count (Linear/Conv2D Layers)
- layer_idx=0, type=LayerType.Linear, total=256, inp1_unstable=5, inp2_unstable=5, delta_unstable=251
- layer_idx=2, type=LayerType.Linear, total=256, inp1_unstable=4, inp2_unstable=4, delta_unstable=255
- layer_idx=4, type=LayerType.Linear, total=256, inp1_unstable=0, inp2_unstable=0, delta_unstable=256
- layer_idx=6, type=LayerType.Linear, total=256, inp1_unstable=1, inp2_unstable=1, delta_unstable=256
- layer_idx=8, type=LayerType.Linear, total=10, inp1_unstable=0, inp2_unstable=0, delta_unstable=10

Time for backsubstitution: 1.15 seconds

### RS candidates at layer 3
type: RSZ, layer: 3, pos: 80
type: RSZ, layer: 3, pos: 81
type: RSZ, layer: 3, pos: 136

Time for candidate selection: 0.12 seconds

### Candidate
type: RSZ, layer: 3, pos: 80

### Relational analysis RSZ of RS_RSZ1_RSZ1_RSZ2_RSZ2_RSZ2_RSZ1_RSZ1

#### Relational analysis RSZ result of RS_RSZ1_RSZ1_RSZ2_RSZ2_RSZ2_RSZ1_RSZ1
Status: Status.UNKNOWN
Output dim: 1, lower bound: -0.0002839, upper bound: 0.0002785
time: 0.57 seconds

### Relational analysis RSZ of RS_RSZ1_RSZ1_RSZ2_RSZ2_RSZ2_RSZ1_RSZ2

#### Relational analysis RSZ result of RS_RSZ1_RSZ1_RSZ2_RSZ2_RSZ2_RSZ1_RSZ2
Status: Status.UNKNOWN
Output dim: 1, lower bound: -0.0002827, upper bound: 0.0002803
time: 0.56 seconds

## BFS RS instance: RS_RSZ1_RSZ1_RSZ2_RSZ2_RSZ2_RSZ2

### Backsubstitution after applying RS history:
0: 0.0009457, 0.0011003, 0.0009457, 0.0011003, -0.0001494, 0.0001489
1: 0.9936197, 0.9940771, 0.9936197, 0.9940771, -0.0003861, 0.0003925
2: -0.0065606, -0.0050208, -0.0065606, -0.0050208, -0.0010657, 0.0011321
3: 0.0038004, 0.0040140, 0.0038004, 0.0040140, -0.0001685, 0.0001641
4: 0.0023852, 0.0036022, 0.0023852, 0.0036022, -0.0009818, 0.0009292
5: 0.0060086, 0.0065111, 0.0060086, 0.0065111, -0.0005024, 0.0005024
6: -0.0013983, -0.0008639, -0.0013983, -0.0008639, -0.0003648, 0.0003879
7: -0.0083221, -0.0079387, -0.0083221, -0.0079387, -0.0003835, 0.0003835
8: 0.0048484, 0.0068715, 0.0048484, 0.0068715, -0.0012479, 0.0011605
9: -0.0036877, -0.0032079, -0.0036877, -0.0032079, -0.0004797, 0.0004797

### Unstable ReLU Count (Linear/Conv2D Layers)
- layer_idx=0, type=LayerType.Linear, total=256, inp1_unstable=5, inp2_unstable=5, delta_unstable=251
- layer_idx=2, type=LayerType.Linear, total=256, inp1_unstable=4, inp2_unstable=4, delta_unstable=255
- layer_idx=4, type=LayerType.Linear, total=256, inp1_unstable=0, inp2_unstable=0, delta_unstable=256
- layer_idx=6, type=LayerType.Linear, total=256, inp1_unstable=1, inp2_unstable=1, delta_unstable=256
- layer_idx=8, type=LayerType.Linear, total=10, inp1_unstable=0, inp2_unstable=0, delta_unstable=10

Time for backsubstitution: 1.15 seconds

### RS candidates at layer 3
type: RSZ, layer: 3, pos: 80
type: RSZ, layer: 3, pos: 81
type: RSZ, layer: 3, pos: 136

Time for candidate selection: 0.12 seconds

### Candidate
type: RSZ, layer: 3, pos: 80

### Relational analysis RSZ of RS_RSZ1_RSZ1_RSZ2_RSZ2_RSZ2_RSZ2_RSZ1

#### Relational analysis RSZ result of RS_RSZ1_RSZ1_RSZ2_RSZ2_RSZ2_RSZ2_RSZ1
Status: Status.UNKNOWN
Output dim: 1, lower bound: -0.0002801, upper bound: 0.0002833
time: 0.58 seconds

### Relational analysis RSZ of RS_RSZ1_RSZ1_RSZ2_RSZ2_RSZ2_RSZ2_RSZ2

#### Relational analysis RSZ result of RS_RSZ1_RSZ1_RSZ2_RSZ2_RSZ2_RSZ2_RSZ2
Status: Status.UNKNOWN
Output dim: 1, lower bound: -0.0002780, upper bound: 0.0002842
time: 0.57 seconds

## BFS RS instance: RS_RSZ1_RSZ2_RSZ1_RSZ1_RSZ1_RSZ1

### Backsubstitution after applying RS history:
0: 0.0009457, 0.0011003, 0.0009457, 0.0011003, -0.0001488, 0.0001495
1: 0.9936197, 0.9940771, 0.9936197, 0.9940771, -0.0003940, 0.0003855
2: -0.0065606, -0.0050208, -0.0065606, -0.0050208, -0.0011471, 0.0010588
3: 0.0038004, 0.0040140, 0.0038004, 0.0040140, -0.0001636, 0.0001696
4: 0.0023852, 0.0036022, 0.0023852, 0.0036022, -0.0009238, 0.0009932
5: 0.0060086, 0.0065111, 0.0060086, 0.0065111, -0.0005024, 0.0005024
6: -0.0013983, -0.0008639, -0.0013983, -0.0008639, -0.0003931, 0.0003624
7: -0.0083221, -0.0079387, -0.0083221, -0.0079387, -0.0003835, 0.0003835
8: 0.0048484, 0.0068715, 0.0048484, 0.0068715, -0.0011515, 0.0012689
9: -0.0036877, -0.0032079, -0.0036877, -0.0032079, -0.0004797, 0.0004797

### Unstable ReLU Count (Linear/Conv2D Layers)
- layer_idx=0, type=LayerType.Linear, total=256, inp1_unstable=5, inp2_unstable=5, delta_unstable=251
- layer_idx=2, type=LayerType.Linear, total=256, inp1_unstable=4, inp2_unstable=4, delta_unstable=255
- layer_idx=4, type=LayerType.Linear, total=256, inp1_unstable=0, inp2_unstable=0, delta_unstable=256
- layer_idx=6, type=LayerType.Linear, total=256, inp1_unstable=1, inp2_unstable=1, delta_unstable=256
- layer_idx=8, type=LayerType.Linear, total=10, inp1_unstable=0, inp2_unstable=0, delta_unstable=10

Time for backsubstitution: 1.16 seconds

### RS candidates at layer 3
type: RSZ, layer: 3, pos: 80
type: RSZ, layer: 3, pos: 81
type: RSZ, layer: 3, pos: 136

Time for candidate selection: 0.12 seconds

### Candidate
type: RSZ, layer: 3, pos: 80

### Relational analysis RSZ of RS_RSZ1_RSZ2_RSZ1_RSZ1_RSZ1_RSZ1_RSZ1

#### Relational analysis RSZ result of RS_RSZ1_RSZ2_RSZ1_RSZ1_RSZ1_RSZ1_RSZ1
Status: Status.UNKNOWN
Output dim: 1, lower bound: -0.0002842, upper bound: 0.0002780
time: 0.58 seconds

### Relational analysis RSZ of RS_RSZ1_RSZ2_RSZ1_RSZ1_RSZ1_RSZ1_RSZ2

#### Relational analysis RSZ result of RS_RSZ1_RSZ2_RSZ1_RSZ1_RSZ1_RSZ1_RSZ2
Status: Status.UNKNOWN
Output dim: 1, lower bound: -0.0002833, upper bound: 0.0002801
time: 0.58 seconds

## BFS RS instance: RS_RSZ1_RSZ2_RSZ1_RSZ1_RSZ1_RSZ2

### Backsubstitution after applying RS history:
0: 0.0009457, 0.0011003, 0.0009457, 0.0011003, -0.0001493, 0.0001490
1: 0.9936197, 0.9940771, 0.9936197, 0.9940771, -0.0003873, 0.0003923
2: -0.0065606, -0.0050208, -0.0065606, -0.0050208, -0.0010766, 0.0011302
3: 0.0038004, 0.0040140, 0.0038004, 0.0040140, -0.0001683, 0.0001649
4: 0.0023852, 0.0036022, 0.0023852, 0.0036022, -0.0009803, 0.0009376
5: 0.0060086, 0.0065111, 0.0060086, 0.0065111, -0.0005024, 0.0005024
6: -0.0013983, -0.0008639, -0.0013983, -0.0008639, -0.0003686, 0.0003872
7: -0.0083221, -0.0079387, -0.0083221, -0.0079387, -0.0003835, 0.0003835
8: 0.0048484, 0.0068715, 0.0048484, 0.0068715, -0.0012453, 0.0011759
9: -0.0036877, -0.0032079, -0.0036877, -0.0032079, -0.0004797, 0.0004797

### Unstable ReLU Count (Linear/Conv2D Layers)
- layer_idx=0, type=LayerType.Linear, total=256, inp1_unstable=5, inp2_unstable=5, delta_unstable=251
- layer_idx=2, type=LayerType.Linear, total=256, inp1_unstable=4, inp2_unstable=4, delta_unstable=255
- layer_idx=4, type=LayerType.Linear, total=256, inp1_unstable=0, inp2_unstable=0, delta_unstable=256
- layer_idx=6, type=LayerType.Linear, total=256, inp1_unstable=1, inp2_unstable=1, delta_unstable=256
- layer_idx=8, type=LayerType.Linear, total=10, inp1_unstable=0, inp2_unstable=0, delta_unstable=10

Time for backsubstitution: 1.17 seconds

### RS candidates at layer 3
type: RSZ, layer: 3, pos: 80
type: RSZ, layer: 3, pos: 81
type: RSZ, layer: 3, pos: 136

Time for candidate selection: 0.12 seconds

### Candidate
type: RSZ, layer: 3, pos: 80

### Relational analysis RSZ of RS_RSZ1_RSZ2_RSZ1_RSZ1_RSZ1_RSZ2_RSZ1

#### Relational analysis RSZ result of RS_RSZ1_RSZ2_RSZ1_RSZ1_RSZ1_RSZ2_RSZ1
Status: Status.UNKNOWN
Output dim: 1, lower bound: -0.0002803, upper bound: 0.0002827
time: 0.59 seconds

### Relational analysis RSZ of RS_RSZ1_RSZ2_RSZ1_RSZ1_RSZ1_RSZ2_RSZ2

#### Relational analysis RSZ result of RS_RSZ1_RSZ2_RSZ1_RSZ1_RSZ1_RSZ2_RSZ2
Status: Status.UNKNOWN
Output dim: 1, lower bound: -0.0002785, upper bound: 0.0002839
time: 0.62 seconds

## BFS RS instance: RS_RSZ1_RSZ2_RSZ1_RSZ1_RSZ2_RSZ1

### Backsubstitution after applying RS history:
0: 0.0009457, 0.0011003, 0.0009457, 0.0011003, -0.0001488, 0.0001495
1: 0.9936197, 0.9940771, 0.9936197, 0.9940771, -0.0003940, 0.0003856
2: -0.0065606, -0.0050208, -0.0065606, -0.0050208, -0.0011471, 0.0010604
3: 0.0038004, 0.0040140, 0.0038004, 0.0040140, -0.0001637, 0.0001696
4: 0.0023852, 0.0036022, 0.0023852, 0.0036022, -0.0009250, 0.0009931
5: 0.0060086, 0.0065111, 0.0060086, 0.0065111, -0.0005024, 0.0005024
6: -0.0013983, -0.0008639, -0.0013983, -0.0008639, -0.0003931, 0.0003630
7: -0.0083221, -0.0079387, -0.0083221, -0.0079387, -0.0003835, 0.0003835
8: 0.0048484, 0.0068715, 0.0048484, 0.0068715, -0.0011535, 0.0012688
9: -0.0036877, -0.0032079, -0.0036877, -0.0032079, -0.0004797, 0.0004797

### Unstable ReLU Count (Linear/Conv2D Layers)
- layer_idx=0, type=LayerType.Linear, total=256, inp1_unstable=5, inp2_unstable=5, delta_unstable=251
- layer_idx=2, type=LayerType.Linear, total=256, inp1_unstable=4, inp2_unstable=4, delta_unstable=255
- layer_idx=4, type=LayerType.Linear, total=256, inp1_unstable=0, inp2_unstable=0, delta_unstable=256
- layer_idx=6, type=LayerType.Linear, total=256, inp1_unstable=1, inp2_unstable=1, delta_unstable=256
- layer_idx=8, type=LayerType.Linear, total=10, inp1_unstable=0, inp2_unstable=0, delta_unstable=10

Time for backsubstitution: 1.31 seconds

### RS candidates at layer 3
type: RSZ, layer: 3, pos: 80
type: RSZ, layer: 3, pos: 81
type: RSZ, layer: 3, pos: 136

Time for candidate selection: 0.14 seconds

### Candidate
type: RSZ, layer: 3, pos: 80

### Relational analysis RSZ of RS_RSZ1_RSZ2_RSZ1_RSZ1_RSZ2_RSZ1_RSZ1

#### Relational analysis RSZ result of RS_RSZ1_RSZ2_RSZ1_RSZ1_RSZ2_RSZ1_RSZ1
Status: Status.UNKNOWN
Output dim: 1, lower bound: -0.0002840, upper bound: 0.0002779
time: 0.60 seconds

### Relational analysis RSZ of RS_RSZ1_RSZ2_RSZ1_RSZ1_RSZ2_RSZ1_RSZ2

#### Relational analysis RSZ result of RS_RSZ1_RSZ2_RSZ1_RSZ1_RSZ2_RSZ1_RSZ2
Status: Status.UNKNOWN
Output dim: 1, lower bound: -0.0002833, upper bound: 0.0002801
time: 0.58 seconds

## BFS RS instance: RS_RSZ1_RSZ2_RSZ1_RSZ1_RSZ2_RSZ2

### Backsubstitution after applying RS history:
0: 0.0009457, 0.0011003, 0.0009457, 0.0011003, -0.0001493, 0.0001489
1: 0.9936197, 0.9940771, 0.9936197, 0.9940771, -0.0003870, 0.0003923
2: -0.0065606, -0.0050208, -0.0065606, -0.0050208, -0.0010736, 0.0011300
3: 0.0038004, 0.0040140, 0.0038004, 0.0040140, -0.0001683, 0.0001647
4: 0.0023852, 0.0036022, 0.0023852, 0.0036022, -0.0009800, 0.0009353
5: 0.0060086, 0.0065111, 0.0060086, 0.0065111, -0.0005024, 0.0005024
6: -0.0013983, -0.0008639, -0.0013983, -0.0008639, -0.0003676, 0.0003871
7: -0.0083221, -0.0079387, -0.0083221, -0.0079387, -0.0003835, 0.0003835
8: 0.0048484, 0.0068715, 0.0048484, 0.0068715, -0.0012450, 0.0011720
9: -0.0036877, -0.0032079, -0.0036877, -0.0032079, -0.0004797, 0.0004797

### Unstable ReLU Count (Linear/Conv2D Layers)
- layer_idx=0, type=LayerType.Linear, total=256, inp1_unstable=5, inp2_unstable=5, delta_unstable=251
- layer_idx=2, type=LayerType.Linear, total=256, inp1_unstable=4, inp2_unstable=4, delta_unstable=255
- layer_idx=4, type=LayerType.Linear, total=256, inp1_unstable=0, inp2_unstable=0, delta_unstable=256
- layer_idx=6, type=LayerType.Linear, total=256, inp1_unstable=1, inp2_unstable=1, delta_unstable=256
- layer_idx=8, type=LayerType.Linear, total=10, inp1_unstable=0, inp2_unstable=0, delta_unstable=10

Time for backsubstitution: 1.18 seconds

### RS candidates at layer 3
type: RSZ, layer: 3, pos: 80
type: RSZ, layer: 3, pos: 81
type: RSZ, layer: 3, pos: 136

Time for candidate selection: 0.12 seconds

### Candidate
type: RSZ, layer: 3, pos: 80

### Relational analysis RSZ of RS_RSZ1_RSZ2_RSZ1_RSZ1_RSZ2_RSZ2_RSZ1

#### Relational analysis RSZ result of RS_RSZ1_RSZ2_RSZ1_RSZ1_RSZ2_RSZ2_RSZ1
Status: Status.UNKNOWN
Output dim: 1, lower bound: -0.0002802, upper bound: 0.0002827
time: 0.58 seconds

### Relational analysis RSZ of RS_RSZ1_RSZ2_RSZ1_RSZ1_RSZ2_RSZ2_RSZ2

#### Relational analysis RSZ result of RS_RSZ1_RSZ2_RSZ1_RSZ1_RSZ2_RSZ2_RSZ2
Status: Status.UNKNOWN
Output dim: 1, lower bound: -0.0002785, upper bound: 0.0002839
time: 0.58 seconds

## BFS RS instance: RS_RSZ1_RSZ2_RSZ1_RSZ2_RSZ1_RSZ1

### Backsubstitution after applying RS history:
0: 0.0009457, 0.0011003, 0.0009457, 0.0011003, -0.0001488, 0.0001495
1: 0.9936197, 0.9940771, 0.9936197, 0.9940771, -0.0003937, 0.0003857
2: -0.0065606, -0.0050208, -0.0065606, -0.0050208, -0.0011434, 0.0010610
3: 0.0038004, 0.0040140, 0.0038004, 0.0040140, -0.0001638, 0.0001693
4: 0.0023852, 0.0036022, 0.0023852, 0.0036022, -0.0009256, 0.0009902
5: 0.0060086, 0.0065111, 0.0060086, 0.0065111, -0.0005024, 0.0005024
6: -0.0013983, -0.0008639, -0.0013983, -0.0008639, -0.0003918, 0.0003632
7: -0.0083221, -0.0079387, -0.0083221, -0.0079387, -0.0003835, 0.0003835
8: 0.0048484, 0.0068715, 0.0048484, 0.0068715, -0.0011544, 0.0012640
9: -0.0036877, -0.0032079, -0.0036877, -0.0032079, -0.0004797, 0.0004797

### Unstable ReLU Count (Linear/Conv2D Layers)
- layer_idx=0, type=LayerType.Linear, total=256, inp1_unstable=5, inp2_unstable=5, delta_unstable=251
- layer_idx=2, type=LayerType.Linear, total=256, inp1_unstable=4, inp2_unstable=4, delta_unstable=255
- layer_idx=4, type=LayerType.Linear, total=256, inp1_unstable=0, inp2_unstable=0, delta_unstable=256
- layer_idx=6, type=LayerType.Linear, total=256, inp1_unstable=1, inp2_unstable=1, delta_unstable=256
- layer_idx=8, type=LayerType.Linear, total=10, inp1_unstable=0, inp2_unstable=0, delta_unstable=10

Time for backsubstitution: 1.20 seconds

### RS candidates at layer 3
type: RSZ, layer: 3, pos: 80
type: RSZ, layer: 3, pos: 81
type: RSZ, layer: 3, pos: 136

Time for candidate selection: 0.12 seconds

### Candidate
type: RSZ, layer: 3, pos: 80

### Relational analysis RSZ of RS_RSZ1_RSZ2_RSZ1_RSZ2_RSZ1_RSZ1_RSZ1

#### Relational analysis RSZ result of RS_RSZ1_RSZ2_RSZ1_RSZ2_RSZ1_RSZ1_RSZ1
Status: Status.UNKNOWN
Output dim: 1, lower bound: -0.0002832, upper bound: 0.0002785
time: 0.59 seconds

### Relational analysis RSZ of RS_RSZ1_RSZ2_RSZ1_RSZ2_RSZ1_RSZ1_RSZ2

#### Relational analysis RSZ result of RS_RSZ1_RSZ2_RSZ1_RSZ2_RSZ1_RSZ1_RSZ2
Status: Status.UNKNOWN
Output dim: 1, lower bound: -0.0002826, upper bound: 0.0002809
time: 0.61 seconds

## BFS RS instance: RS_RSZ1_RSZ2_RSZ1_RSZ2_RSZ1_RSZ2

### Backsubstitution after applying RS history:
0: 0.0009457, 0.0011003, 0.0009457, 0.0011003, -0.0001494, 0.0001490
1: 0.9936197, 0.9940771, 0.9936197, 0.9940771, -0.0003875, 0.0003926
2: -0.0065606, -0.0050208, -0.0065606, -0.0050208, -0.0010782, 0.0011337
3: 0.0038004, 0.0040140, 0.0038004, 0.0040140, -0.0001686, 0.0001650
4: 0.0023852, 0.0036022, 0.0023852, 0.0036022, -0.0009830, 0.0009389
5: 0.0060086, 0.0065111, 0.0060086, 0.0065111, -0.0005024, 0.0005024
6: -0.0013983, -0.0008639, -0.0013983, -0.0008639, -0.0003692, 0.0003884
7: -0.0083221, -0.0079387, -0.0083221, -0.0079387, -0.0003835, 0.0003835
8: 0.0048484, 0.0068715, 0.0048484, 0.0068715, -0.0012500, 0.0011781
9: -0.0036877, -0.0032079, -0.0036877, -0.0032079, -0.0004797, 0.0004797

### Unstable ReLU Count (Linear/Conv2D Layers)
- layer_idx=0, type=LayerType.Linear, total=256, inp1_unstable=5, inp2_unstable=5, delta_unstable=251
- layer_idx=2, type=LayerType.Linear, total=256, inp1_unstable=4, inp2_unstable=4, delta_unstable=255
- layer_idx=4, type=LayerType.Linear, total=256, inp1_unstable=0, inp2_unstable=0, delta_unstable=256
- layer_idx=6, type=LayerType.Linear, total=256, inp1_unstable=1, inp2_unstable=1, delta_unstable=256
- layer_idx=8, type=LayerType.Linear, total=10, inp1_unstable=0, inp2_unstable=0, delta_unstable=10

Time for backsubstitution: 1.19 seconds

### RS candidates at layer 3
type: RSZ, layer: 3, pos: 80
type: RSZ, layer: 3, pos: 81
type: RSZ, layer: 3, pos: 136

Time for candidate selection: 0.12 seconds

### Candidate
type: RSZ, layer: 3, pos: 80

### Relational analysis RSZ of RS_RSZ1_RSZ2_RSZ1_RSZ2_RSZ1_RSZ2_RSZ1

#### Relational analysis RSZ result of RS_RSZ1_RSZ2_RSZ1_RSZ2_RSZ1_RSZ2_RSZ1
Status: Status.UNKNOWN
Output dim: 1, lower bound: -0.0002797, upper bound: 0.0002835
time: 0.57 seconds

### Relational analysis RSZ of RS_RSZ1_RSZ2_RSZ1_RSZ2_RSZ1_RSZ2_RSZ2

#### Relational analysis RSZ result of RS_RSZ1_RSZ2_RSZ1_RSZ2_RSZ1_RSZ2_RSZ2
Status: Status.UNKNOWN
Output dim: 1, lower bound: -0.0002779, upper bound: 0.0002849
time: 0.57 seconds

## BFS RS instance: RS_RSZ1_RSZ2_RSZ1_RSZ2_RSZ2_RSZ1

### Backsubstitution after applying RS history:
0: 0.0009457, 0.0011003, 0.0009457, 0.0011003, -0.0001488, 0.0001495
1: 0.9936197, 0.9940771, 0.9936197, 0.9940771, -0.0003937, 0.0003859
2: -0.0065606, -0.0050208, -0.0065606, -0.0050208, -0.0011435, 0.0010632
3: 0.0038004, 0.0040140, 0.0038004, 0.0040140, -0.0001639, 0.0001693
4: 0.0023852, 0.0036022, 0.0023852, 0.0036022, -0.0009273, 0.0009903
5: 0.0060086, 0.0065111, 0.0060086, 0.0065111, -0.0005024, 0.0005024
6: -0.0013983, -0.0008639, -0.0013983, -0.0008639, -0.0003919, 0.0003640
7: -0.0083221, -0.0079387, -0.0083221, -0.0079387, -0.0003835, 0.0003835
8: 0.0048484, 0.0068715, 0.0048484, 0.0068715, -0.0011573, 0.0012642
9: -0.0036877, -0.0032079, -0.0036877, -0.0032079, -0.0004797, 0.0004797

### Unstable ReLU Count (Linear/Conv2D Layers)
- layer_idx=0, type=LayerType.Linear, total=256, inp1_unstable=5, inp2_unstable=5, delta_unstable=251
- layer_idx=2, type=LayerType.Linear, total=256, inp1_unstable=4, inp2_unstable=4, delta_unstable=255
- layer_idx=4, type=LayerType.Linear, total=256, inp1_unstable=0, inp2_unstable=0, delta_unstable=256
- layer_idx=6, type=LayerType.Linear, total=256, inp1_unstable=1, inp2_unstable=1, delta_unstable=256
- layer_idx=8, type=LayerType.Linear, total=10, inp1_unstable=0, inp2_unstable=0, delta_unstable=10

Time for backsubstitution: 1.17 seconds

### RS candidates at layer 3
type: RSZ, layer: 3, pos: 80
type: RSZ, layer: 3, pos: 81
type: RSZ, layer: 3, pos: 136

Time for candidate selection: 0.12 seconds

### Candidate
type: RSZ, layer: 3, pos: 80

### Relational analysis RSZ of RS_RSZ1_RSZ2_RSZ1_RSZ2_RSZ2_RSZ1_RSZ1

#### Relational analysis RSZ result of RS_RSZ1_RSZ2_RSZ1_RSZ2_RSZ2_RSZ1_RSZ1
Status: Status.UNKNOWN
Output dim: 1, lower bound: -0.0002832, upper bound: 0.0002785
time: 0.59 seconds

### Relational analysis RSZ of RS_RSZ1_RSZ2_RSZ1_RSZ2_RSZ2_RSZ1_RSZ2

#### Relational analysis RSZ result of RS_RSZ1_RSZ2_RSZ1_RSZ2_RSZ2_RSZ1_RSZ2
Status: Status.UNKNOWN
Output dim: 1, lower bound: -0.0002827, upper bound: 0.0002810
time: 0.56 seconds

## BFS RS instance: RS_RSZ1_RSZ2_RSZ1_RSZ2_RSZ2_RSZ2

### Backsubstitution after applying RS history:
0: 0.0009457, 0.0011003, 0.0009457, 0.0011003, -0.0001494, 0.0001490
1: 0.9936197, 0.9940771, 0.9936197, 0.9940771, -0.0003873, 0.0003926
2: -0.0065606, -0.0050208, -0.0065606, -0.0050208, -0.0010767, 0.0011337
3: 0.0038004, 0.0040140, 0.0038004, 0.0040140, -0.0001686, 0.0001649
4: 0.0023852, 0.0036022, 0.0023852, 0.0036022, -0.0009830, 0.0009377
5: 0.0060086, 0.0065111, 0.0060086, 0.0065111, -0.0005024, 0.0005024
6: -0.0013983, -0.0008639, -0.0013983, -0.0008639, -0.0003687, 0.0003884
7: -0.0083221, -0.0079387, -0.0083221, -0.0079387, -0.0003835, 0.0003835
8: 0.0048484, 0.0068715, 0.0048484, 0.0068715, -0.0012499, 0.0011760
9: -0.0036877, -0.0032079, -0.0036877, -0.0032079, -0.0004797, 0.0004797

### Unstable ReLU Count (Linear/Conv2D Layers)
- layer_idx=0, type=LayerType.Linear, total=256, inp1_unstable=5, inp2_unstable=5, delta_unstable=251
- layer_idx=2, type=LayerType.Linear, total=256, inp1_unstable=4, inp2_unstable=4, delta_unstable=255
- layer_idx=4, type=LayerType.Linear, total=256, inp1_unstable=0, inp2_unstable=0, delta_unstable=256
- layer_idx=6, type=LayerType.Linear, total=256, inp1_unstable=1, inp2_unstable=1, delta_unstable=256
- layer_idx=8, type=LayerType.Linear, total=10, inp1_unstable=0, inp2_unstable=0, delta_unstable=10

Time for backsubstitution: 1.22 seconds

### RS candidates at layer 3
type: RSZ, layer: 3, pos: 80
type: RSZ, layer: 3, pos: 81
type: RSZ, layer: 3, pos: 136

Time for candidate selection: 0.12 seconds

### Candidate
type: RSZ, layer: 3, pos: 80

### Relational analysis RSZ of RS_RSZ1_RSZ2_RSZ1_RSZ2_RSZ2_RSZ2_RSZ1

#### Relational analysis RSZ result of RS_RSZ1_RSZ2_RSZ1_RSZ2_RSZ2_RSZ2_RSZ1
Status: Status.UNKNOWN
Output dim: 1, lower bound: -0.0002796, upper bound: 0.0002835
time: 0.57 seconds

### Relational analysis RSZ of RS_RSZ1_RSZ2_RSZ1_RSZ2_RSZ2_RSZ2_RSZ2

#### Relational analysis RSZ result of RS_RSZ1_RSZ2_RSZ1_RSZ2_RSZ2_RSZ2_RSZ2
Status: Status.UNKNOWN
Output dim: 1, lower bound: -0.0002779, upper bound: 0.0002850
time: 0.56 seconds

## BFS RS instance: RS_RSZ1_RSZ2_RSZ2_RSZ1_RSZ1_RSZ1

### Backsubstitution after applying RS history:
0: 0.0009457, 0.0011003, 0.0009457, 0.0011003, -0.0001488, 0.0001495
1: 0.9936197, 0.9940771, 0.9936197, 0.9940771, -0.0003942, 0.0003854
2: -0.0065606, -0.0050208, -0.0065606, -0.0050208, -0.0011487, 0.0010581
3: 0.0038004, 0.0040140, 0.0038004, 0.0040140, -0.0001636, 0.0001697
4: 0.0023852, 0.0036022, 0.0023852, 0.0036022, -0.0009233, 0.0009944
5: 0.0060086, 0.0065111, 0.0060086, 0.0065111, -0.0005024, 0.0005024
6: -0.0013983, -0.0008639, -0.0013983, -0.0008639, -0.0003936, 0.0003622
7: -0.0083221, -0.0079387, -0.0083221, -0.0079387, -0.0003835, 0.0003835
8: 0.0048484, 0.0068715, 0.0048484, 0.0068715, -0.0011506, 0.0012710
9: -0.0036877, -0.0032079, -0.0036877, -0.0032079, -0.0004797, 0.0004797

### Unstable ReLU Count (Linear/Conv2D Layers)
- layer_idx=0, type=LayerType.Linear, total=256, inp1_unstable=5, inp2_unstable=5, delta_unstable=251
- layer_idx=2, type=LayerType.Linear, total=256, inp1_unstable=4, inp2_unstable=4, delta_unstable=255
- layer_idx=4, type=LayerType.Linear, total=256, inp1_unstable=0, inp2_unstable=0, delta_unstable=256
- layer_idx=6, type=LayerType.Linear, total=256, inp1_unstable=1, inp2_unstable=1, delta_unstable=256
- layer_idx=8, type=LayerType.Linear, total=10, inp1_unstable=0, inp2_unstable=0, delta_unstable=10

Time for backsubstitution: 1.18 seconds

### RS candidates at layer 3
type: RSZ, layer: 3, pos: 80
type: RSZ, layer: 3, pos: 81
type: RSZ, layer: 3, pos: 136

Time for candidate selection: 0.13 seconds

### Candidate
type: RSZ, layer: 3, pos: 80

### Relational analysis RSZ of RS_RSZ1_RSZ2_RSZ2_RSZ1_RSZ1_RSZ1_RSZ1

#### Relational analysis RSZ result of RS_RSZ1_RSZ2_RSZ2_RSZ1_RSZ1_RSZ1_RSZ1
Status: Status.UNKNOWN
Output dim: 1, lower bound: -0.0002850, upper bound: 0.0002779
time: 0.58 seconds

### Relational analysis RSZ of RS_RSZ1_RSZ2_RSZ2_RSZ1_RSZ1_RSZ1_RSZ2

#### Relational analysis RSZ result of RS_RSZ1_RSZ2_RSZ2_RSZ1_RSZ1_RSZ1_RSZ2
Status: Status.UNKNOWN
Output dim: 1, lower bound: -0.0002835, upper bound: 0.0002795
time: 0.59 seconds

## BFS RS instance: RS_RSZ1_RSZ2_RSZ2_RSZ1_RSZ1_RSZ2

### Backsubstitution after applying RS history:
0: 0.0009457, 0.0011003, 0.0009457, 0.0011003, -0.0001493, 0.0001490
1: 0.9936197, 0.9940771, 0.9936197, 0.9940771, -0.0003876, 0.0003920
2: -0.0065606, -0.0050208, -0.0065606, -0.0050208, -0.0010803, 0.0011277
3: 0.0038004, 0.0040140, 0.0038004, 0.0040140, -0.0001682, 0.0001651
4: 0.0023852, 0.0036022, 0.0023852, 0.0036022, -0.0009783, 0.0009405
5: 0.0060086, 0.0065111, 0.0060086, 0.0065111, -0.0005024, 0.0005024
6: -0.0013983, -0.0008639, -0.0013983, -0.0008639, -0.0003699, 0.0003863
7: -0.0083221, -0.0079387, -0.0083221, -0.0079387, -0.0003835, 0.0003835
8: 0.0048484, 0.0068715, 0.0048484, 0.0068715, -0.0012421, 0.0011808
9: -0.0036877, -0.0032079, -0.0036877, -0.0032079, -0.0004797, 0.0004797

### Unstable ReLU Count (Linear/Conv2D Layers)
- layer_idx=0, type=LayerType.Linear, total=256, inp1_unstable=5, inp2_unstable=5, delta_unstable=251
- layer_idx=2, type=LayerType.Linear, total=256, inp1_unstable=4, inp2_unstable=4, delta_unstable=255
- layer_idx=4, type=LayerType.Linear, total=256, inp1_unstable=0, inp2_unstable=0, delta_unstable=256
- layer_idx=6, type=LayerType.Linear, total=256, inp1_unstable=1, inp2_unstable=1, delta_unstable=256
- layer_idx=8, type=LayerType.Linear, total=10, inp1_unstable=0, inp2_unstable=0, delta_unstable=10

Time for backsubstitution: 1.16 seconds

### RS candidates at layer 3
type: RSZ, layer: 3, pos: 80
type: RSZ, layer: 3, pos: 81
type: RSZ, layer: 3, pos: 136

Time for candidate selection: 0.14 seconds

### Candidate
type: RSZ, layer: 3, pos: 80

### Relational analysis RSZ of RS_RSZ1_RSZ2_RSZ2_RSZ1_RSZ1_RSZ2_RSZ1

#### Relational analysis RSZ result of RS_RSZ1_RSZ2_RSZ2_RSZ1_RSZ1_RSZ2_RSZ1
Status: Status.UNKNOWN
Output dim: 1, lower bound: -0.0002810, upper bound: 0.0002827
time: 0.64 seconds

### Relational analysis RSZ of RS_RSZ1_RSZ2_RSZ2_RSZ1_RSZ1_RSZ2_RSZ2

#### Relational analysis RSZ result of RS_RSZ1_RSZ2_RSZ2_RSZ1_RSZ1_RSZ2_RSZ2
Status: Status.UNKNOWN
Output dim: 1, lower bound: -0.0002785, upper bound: 0.0002831
time: 0.59 seconds

## BFS RS instance: RS_RSZ1_RSZ2_RSZ2_RSZ1_RSZ2_RSZ1

### Backsubstitution after applying RS history:
0: 0.0009457, 0.0011003, 0.0009457, 0.0011003, -0.0001488, 0.0001495
1: 0.9936197, 0.9940771, 0.9936197, 0.9940771, -0.0003942, 0.0003856
2: -0.0065606, -0.0050208, -0.0065606, -0.0050208, -0.0011488, 0.0010597
3: 0.0038004, 0.0040140, 0.0038004, 0.0040140, -0.0001637, 0.0001697
4: 0.0023852, 0.0036022, 0.0023852, 0.0036022, -0.0009245, 0.0009945
5: 0.0060086, 0.0065111, 0.0060086, 0.0065111, -0.0005024, 0.0005024
6: -0.0013983, -0.0008639, -0.0013983, -0.0008639, -0.0003937, 0.0003627
7: -0.0083221, -0.0079387, -0.0083221, -0.0079387, -0.0003835, 0.0003835
8: 0.0048484, 0.0068715, 0.0048484, 0.0068715, -0.0011527, 0.0012711
9: -0.0036877, -0.0032079, -0.0036877, -0.0032079, -0.0004797, 0.0004797

### Unstable ReLU Count (Linear/Conv2D Layers)
- layer_idx=0, type=LayerType.Linear, total=256, inp1_unstable=5, inp2_unstable=5, delta_unstable=251
- layer_idx=2, type=LayerType.Linear, total=256, inp1_unstable=4, inp2_unstable=4, delta_unstable=255
- layer_idx=4, type=LayerType.Linear, total=256, inp1_unstable=0, inp2_unstable=0, delta_unstable=256
- layer_idx=6, type=LayerType.Linear, total=256, inp1_unstable=1, inp2_unstable=1, delta_unstable=256
- layer_idx=8, type=LayerType.Linear, total=10, inp1_unstable=0, inp2_unstable=0, delta_unstable=10

Time for backsubstitution: 1.19 seconds

### RS candidates at layer 3
type: RSZ, layer: 3, pos: 80
type: RSZ, layer: 3, pos: 81
type: RSZ, layer: 3, pos: 136

Time for candidate selection: 0.12 seconds

### Candidate
type: RSZ, layer: 3, pos: 80

### Relational analysis RSZ of RS_RSZ1_RSZ2_RSZ2_RSZ1_RSZ2_RSZ1_RSZ1

#### Relational analysis RSZ result of RS_RSZ1_RSZ2_RSZ2_RSZ1_RSZ2_RSZ1_RSZ1
Status: Status.UNKNOWN
Output dim: 1, lower bound: -0.0002850, upper bound: 0.0002779
time: 0.61 seconds

### Relational analysis RSZ of RS_RSZ1_RSZ2_RSZ2_RSZ1_RSZ2_RSZ1_RSZ2

#### Relational analysis RSZ result of RS_RSZ1_RSZ2_RSZ2_RSZ1_RSZ2_RSZ1_RSZ2
Status: Status.UNKNOWN
Output dim: 1, lower bound: -0.0002835, upper bound: 0.0002796
time: 0.56 seconds

## BFS RS instance: RS_RSZ1_RSZ2_RSZ2_RSZ1_RSZ2_RSZ2

### Backsubstitution after applying RS history:
0: 0.0009457, 0.0011003, 0.0009457, 0.0011003, -0.0001493, 0.0001490
1: 0.9936197, 0.9940771, 0.9936197, 0.9940771, -0.0003874, 0.0003920
2: -0.0065606, -0.0050208, -0.0065606, -0.0050208, -0.0010773, 0.0011275
3: 0.0038004, 0.0040140, 0.0038004, 0.0040140, -0.0001682, 0.0001649
4: 0.0023852, 0.0036022, 0.0023852, 0.0036022, -0.0009781, 0.0009382
5: 0.0060086, 0.0065111, 0.0060086, 0.0065111, -0.0005024, 0.0005024
6: -0.0013983, -0.0008639, -0.0013983, -0.0008639, -0.0003689, 0.0003863
7: -0.0083221, -0.0079387, -0.0083221, -0.0079387, -0.0003835, 0.0003835
8: 0.0048484, 0.0068715, 0.0048484, 0.0068715, -0.0012418, 0.0011769
9: -0.0036877, -0.0032079, -0.0036877, -0.0032079, -0.0004797, 0.0004797

### Unstable ReLU Count (Linear/Conv2D Layers)
- layer_idx=0, type=LayerType.Linear, total=256, inp1_unstable=5, inp2_unstable=5, delta_unstable=251
- layer_idx=2, type=LayerType.Linear, total=256, inp1_unstable=4, inp2_unstable=4, delta_unstable=255
- layer_idx=4, type=LayerType.Linear, total=256, inp1_unstable=0, inp2_unstable=0, delta_unstable=256
- layer_idx=6, type=LayerType.Linear, total=256, inp1_unstable=1, inp2_unstable=1, delta_unstable=256
- layer_idx=8, type=LayerType.Linear, total=10, inp1_unstable=0, inp2_unstable=0, delta_unstable=10

Time for backsubstitution: 1.18 seconds

### RS candidates at layer 3
type: RSZ, layer: 3, pos: 80
type: RSZ, layer: 3, pos: 81
type: RSZ, layer: 3, pos: 136

Time for candidate selection: 0.13 seconds

### Candidate
type: RSZ, layer: 3, pos: 80

### Relational analysis RSZ of RS_RSZ1_RSZ2_RSZ2_RSZ1_RSZ2_RSZ2_RSZ1

#### Relational analysis RSZ result of RS_RSZ1_RSZ2_RSZ2_RSZ1_RSZ2_RSZ2_RSZ1
Status: Status.UNKNOWN
Output dim: 1, lower bound: -0.0002810, upper bound: 0.0002826
time: 0.58 seconds

### Relational analysis RSZ of RS_RSZ1_RSZ2_RSZ2_RSZ1_RSZ2_RSZ2_RSZ2

#### Relational analysis RSZ result of RS_RSZ1_RSZ2_RSZ2_RSZ1_RSZ2_RSZ2_RSZ2
Status: Status.UNKNOWN
Output dim: 1, lower bound: -0.0002785, upper bound: 0.0002831
time: 0.59 seconds

## BFS RS instance: RS_RSZ1_RSZ2_RSZ2_RSZ2_RSZ1_RSZ1

### Backsubstitution after applying RS history:
0: 0.0009457, 0.0011003, 0.0009457, 0.0011003, -0.0001488, 0.0001495
1: 0.9936197, 0.9940771, 0.9936197, 0.9940771, -0.0003940, 0.0003852
2: -0.0065606, -0.0050208, -0.0065606, -0.0050208, -0.0011469, 0.0010553
3: 0.0038004, 0.0040140, 0.0038004, 0.0040140, -0.0001634, 0.0001696
4: 0.0023852, 0.0036022, 0.0023852, 0.0036022, -0.0009210, 0.0009930
5: 0.0060086, 0.0065111, 0.0060086, 0.0065111, -0.0005024, 0.0005024
6: -0.0013983, -0.0008639, -0.0013983, -0.0008639, -0.0003930, 0.0003612
7: -0.0083221, -0.0079387, -0.0083221, -0.0079387, -0.0003835, 0.0003835
8: 0.0048484, 0.0068715, 0.0048484, 0.0068715, -0.0011468, 0.0012686
9: -0.0036877, -0.0032079, -0.0036877, -0.0032079, -0.0004797, 0.0004797

### Unstable ReLU Count (Linear/Conv2D Layers)
- layer_idx=0, type=LayerType.Linear, total=256, inp1_unstable=5, inp2_unstable=5, delta_unstable=251
- layer_idx=2, type=LayerType.Linear, total=256, inp1_unstable=4, inp2_unstable=4, delta_unstable=255
- layer_idx=4, type=LayerType.Linear, total=256, inp1_unstable=0, inp2_unstable=0, delta_unstable=256
- layer_idx=6, type=LayerType.Linear, total=256, inp1_unstable=1, inp2_unstable=1, delta_unstable=256
- layer_idx=8, type=LayerType.Linear, total=10, inp1_unstable=0, inp2_unstable=0, delta_unstable=10

Time for backsubstitution: 1.21 seconds

### RS candidates at layer 3
type: RSZ, layer: 3, pos: 80
type: RSZ, layer: 3, pos: 81
type: RSZ, layer: 3, pos: 136

Time for candidate selection: 0.12 seconds

### Candidate
type: RSZ, layer: 3, pos: 80

### Relational analysis RSZ of RS_RSZ1_RSZ2_RSZ2_RSZ2_RSZ1_RSZ1_RSZ1

#### Relational analysis RSZ result of RS_RSZ1_RSZ2_RSZ2_RSZ2_RSZ1_RSZ1_RSZ1
Status: Status.UNKNOWN
Output dim: 1, lower bound: -0.0002840, upper bound: 0.0002785
time: 0.60 seconds

### Relational analysis RSZ of RS_RSZ1_RSZ2_RSZ2_RSZ2_RSZ1_RSZ1_RSZ2

#### Relational analysis RSZ result of RS_RSZ1_RSZ2_RSZ2_RSZ2_RSZ1_RSZ1_RSZ2
Status: Status.UNKNOWN
Output dim: 1, lower bound: -0.0002827, upper bound: 0.0002802
time: 0.65 seconds

## BFS RS instance: RS_RSZ1_RSZ2_RSZ2_RSZ2_RSZ1_RSZ2

### Backsubstitution after applying RS history:
0: 0.0009457, 0.0011003, 0.0009457, 0.0011003, -0.0001494, 0.0001490
1: 0.9936197, 0.9940771, 0.9936197, 0.9940771, -0.0003875, 0.0003923
2: -0.0065606, -0.0050208, -0.0065606, -0.0050208, -0.0010783, 0.0011305
3: 0.0038004, 0.0040140, 0.0038004, 0.0040140, -0.0001684, 0.0001650
4: 0.0023852, 0.0036022, 0.0023852, 0.0036022, -0.0009805, 0.0009390
5: 0.0060086, 0.0065111, 0.0060086, 0.0065111, -0.0005024, 0.0005024
6: -0.0013983, -0.0008639, -0.0013983, -0.0008639, -0.0003692, 0.0003873
7: -0.0083221, -0.0079387, -0.0083221, -0.0079387, -0.0003835, 0.0003835
8: 0.0048484, 0.0068715, 0.0048484, 0.0068715, -0.0012457, 0.0011781
9: -0.0036877, -0.0032079, -0.0036877, -0.0032079, -0.0004797, 0.0004797

### Unstable ReLU Count (Linear/Conv2D Layers)
- layer_idx=0, type=LayerType.Linear, total=256, inp1_unstable=5, inp2_unstable=5, delta_unstable=251
- layer_idx=2, type=LayerType.Linear, total=256, inp1_unstable=4, inp2_unstable=4, delta_unstable=255
- layer_idx=4, type=LayerType.Linear, total=256, inp1_unstable=0, inp2_unstable=0, delta_unstable=256
- layer_idx=6, type=LayerType.Linear, total=256, inp1_unstable=1, inp2_unstable=1, delta_unstable=256
- layer_idx=8, type=LayerType.Linear, total=10, inp1_unstable=0, inp2_unstable=0, delta_unstable=10

Time for backsubstitution: 1.16 seconds

### RS candidates at layer 3
type: RSZ, layer: 3, pos: 80
type: RSZ, layer: 3, pos: 81
type: RSZ, layer: 3, pos: 136

Time for candidate selection: 0.11 seconds

### Candidate
type: RSZ, layer: 3, pos: 80

### Relational analysis RSZ of RS_RSZ1_RSZ2_RSZ2_RSZ2_RSZ1_RSZ2_RSZ1

#### Relational analysis RSZ result of RS_RSZ1_RSZ2_RSZ2_RSZ2_RSZ1_RSZ2_RSZ1
Status: Status.UNKNOWN
Output dim: 1, lower bound: -0.0002804, upper bound: 0.0002833
time: 0.56 seconds

### Relational analysis RSZ of RS_RSZ1_RSZ2_RSZ2_RSZ2_RSZ1_RSZ2_RSZ2

#### Relational analysis RSZ result of RS_RSZ1_RSZ2_RSZ2_RSZ2_RSZ1_RSZ2_RSZ2
Status: Status.UNKNOWN
Output dim: 1, lower bound: -0.0002780, upper bound: 0.0002840
time: 0.61 seconds

## BFS RS instance: RS_RSZ1_RSZ2_RSZ2_RSZ2_RSZ2_RSZ1

### Backsubstitution after applying RS history:
0: 0.0009457, 0.0011003, 0.0009457, 0.0011003, -0.0001488, 0.0001495
1: 0.9936197, 0.9940771, 0.9936197, 0.9940771, -0.0003941, 0.0003854
2: -0.0065606, -0.0050208, -0.0065606, -0.0050208, -0.0011473, 0.0010580
3: 0.0038004, 0.0040140, 0.0038004, 0.0040140, -0.0001636, 0.0001696
4: 0.0023852, 0.0036022, 0.0023852, 0.0036022, -0.0009232, 0.0009933
5: 0.0060086, 0.0065111, 0.0060086, 0.0065111, -0.0005024, 0.0005024
6: -0.0013983, -0.0008639, -0.0013983, -0.0008639, -0.0003932, 0.0003621
7: -0.0083221, -0.0079387, -0.0083221, -0.0079387, -0.0003835, 0.0003835
8: 0.0048484, 0.0068715, 0.0048484, 0.0068715, -0.0011504, 0.0012691
9: -0.0036877, -0.0032079, -0.0036877, -0.0032079, -0.0004797, 0.0004797

### Unstable ReLU Count (Linear/Conv2D Layers)
- layer_idx=0, type=LayerType.Linear, total=256, inp1_unstable=5, inp2_unstable=5, delta_unstable=251
- layer_idx=2, type=LayerType.Linear, total=256, inp1_unstable=4, inp2_unstable=4, delta_unstable=255
- layer_idx=4, type=LayerType.Linear, total=256, inp1_unstable=0, inp2_unstable=0, delta_unstable=256
- layer_idx=6, type=LayerType.Linear, total=256, inp1_unstable=1, inp2_unstable=1, delta_unstable=256
- layer_idx=8, type=LayerType.Linear, total=10, inp1_unstable=0, inp2_unstable=0, delta_unstable=10

Time for backsubstitution: 1.19 seconds

### RS candidates at layer 3
type: RSZ, layer: 3, pos: 80
type: RSZ, layer: 3, pos: 81
type: RSZ, layer: 3, pos: 136

Time for candidate selection: 0.12 seconds

### Candidate
type: RSZ, layer: 3, pos: 80

### Relational analysis RSZ of RS_RSZ1_RSZ2_RSZ2_RSZ2_RSZ2_RSZ1_RSZ1

#### Relational analysis RSZ result of RS_RSZ1_RSZ2_RSZ2_RSZ2_RSZ2_RSZ1_RSZ1
Status: Status.UNKNOWN
Output dim: 1, lower bound: -0.0002841, upper bound: 0.0002785
time: 0.55 seconds

### Relational analysis RSZ of RS_RSZ1_RSZ2_RSZ2_RSZ2_RSZ2_RSZ1_RSZ2

#### Relational analysis RSZ result of RS_RSZ1_RSZ2_RSZ2_RSZ2_RSZ2_RSZ1_RSZ2
Status: Status.UNKNOWN
Output dim: 1, lower bound: -0.0002828, upper bound: 0.0002803
time: 0.55 seconds

## BFS RS instance: RS_RSZ1_RSZ2_RSZ2_RSZ2_RSZ2_RSZ2

### Backsubstitution after applying RS history:
0: 0.0009457, 0.0011003, 0.0009457, 0.0011003, -0.0001494, 0.0001490
1: 0.9936197, 0.9940771, 0.9936197, 0.9940771, -0.0003873, 0.0003923
2: -0.0065606, -0.0050208, -0.0065606, -0.0050208, -0.0010770, 0.0011306
3: 0.0038004, 0.0040140, 0.0038004, 0.0040140, -0.0001684, 0.0001649
4: 0.0023852, 0.0036022, 0.0023852, 0.0036022, -0.0009805, 0.0009379
5: 0.0060086, 0.0065111, 0.0060086, 0.0065111, -0.0005024, 0.0005024
6: -0.0013983, -0.0008639, -0.0013983, -0.0008639, -0.0003688, 0.0003873
7: -0.0083221, -0.0079387, -0.0083221, -0.0079387, -0.0003835, 0.0003835
8: 0.0048484, 0.0068715, 0.0048484, 0.0068715, -0.0012458, 0.0011765
9: -0.0036877, -0.0032079, -0.0036877, -0.0032079, -0.0004797, 0.0004797

### Unstable ReLU Count (Linear/Conv2D Layers)
- layer_idx=0, type=LayerType.Linear, total=256, inp1_unstable=5, inp2_unstable=5, delta_unstable=251
- layer_idx=2, type=LayerType.Linear, total=256, inp1_unstable=4, inp2_unstable=4, delta_unstable=255
- layer_idx=4, type=LayerType.Linear, total=256, inp1_unstable=0, inp2_unstable=0, delta_unstable=256
- layer_idx=6, type=LayerType.Linear, total=256, inp1_unstable=1, inp2_unstable=1, delta_unstable=256
- layer_idx=8, type=LayerType.Linear, total=10, inp1_unstable=0, inp2_unstable=0, delta_unstable=10

Time for backsubstitution: 1.23 seconds

### RS candidates at layer 3
type: RSZ, layer: 3, pos: 80
type: RSZ, layer: 3, pos: 81
type: RSZ, layer: 3, pos: 136

Time for candidate selection: 0.13 seconds

### Candidate
type: RSZ, layer: 3, pos: 80

### Relational analysis RSZ of RS_RSZ1_RSZ2_RSZ2_RSZ2_RSZ2_RSZ2_RSZ1

#### Relational analysis RSZ result of RS_RSZ1_RSZ2_RSZ2_RSZ2_RSZ2_RSZ2_RSZ1
Status: Status.UNKNOWN
Output dim: 1, lower bound: -0.0002804, upper bound: 0.0002833
time: 0.55 seconds

### Relational analysis RSZ of RS_RSZ1_RSZ2_RSZ2_RSZ2_RSZ2_RSZ2_RSZ2

#### Relational analysis RSZ result of RS_RSZ1_RSZ2_RSZ2_RSZ2_RSZ2_RSZ2_RSZ2
Status: Status.UNKNOWN
Output dim: 1, lower bound: -0.0002781, upper bound: 0.0002842
time: 0.59 seconds

## BFS RS instance: RS_RSZ2_RSZ1_RSZ1_RSZ1_RSZ1_RSZ1

### Backsubstitution after applying RS history:
0: 0.0009457, 0.0011003, 0.0009457, 0.0011003, -0.0001489, 0.0001495
1: 0.9936197, 0.9940771, 0.9936197, 0.9940771, -0.0003936, 0.0003864
2: -0.0065606, -0.0050208, -0.0065606, -0.0050208, -0.0011441, 0.0010688
3: 0.0038004, 0.0040140, 0.0038004, 0.0040140, -0.0001643, 0.0001693
4: 0.0023852, 0.0036022, 0.0023852, 0.0036022, -0.0009317, 0.0009910
5: 0.0060086, 0.0065111, 0.0060086, 0.0065111, -0.0005024, 0.0005024
6: -0.0013983, -0.0008639, -0.0013983, -0.0008639, -0.0003920, 0.0003659
7: -0.0083221, -0.0079387, -0.0083221, -0.0079387, -0.0003835, 0.0003835
8: 0.0048484, 0.0068715, 0.0048484, 0.0068715, -0.0011646, 0.0012635
9: -0.0036877, -0.0032079, -0.0036877, -0.0032079, -0.0004797, 0.0004797

### Unstable ReLU Count (Linear/Conv2D Layers)
- layer_idx=0, type=LayerType.Linear, total=256, inp1_unstable=5, inp2_unstable=5, delta_unstable=251
- layer_idx=2, type=LayerType.Linear, total=256, inp1_unstable=4, inp2_unstable=4, delta_unstable=255
- layer_idx=4, type=LayerType.Linear, total=256, inp1_unstable=0, inp2_unstable=0, delta_unstable=256
- layer_idx=6, type=LayerType.Linear, total=256, inp1_unstable=1, inp2_unstable=1, delta_unstable=256
- layer_idx=8, type=LayerType.Linear, total=10, inp1_unstable=0, inp2_unstable=0, delta_unstable=10

Time for backsubstitution: 1.21 seconds

### RS candidates at layer 3
type: RSZ, layer: 3, pos: 80
type: RSZ, layer: 3, pos: 81
type: RSZ, layer: 3, pos: 136

Time for candidate selection: 0.12 seconds

### Candidate
type: RSZ, layer: 3, pos: 80

### Relational analysis RSZ of RS_RSZ2_RSZ1_RSZ1_RSZ1_RSZ1_RSZ1_RSZ1

#### Relational analysis RSZ result of RS_RSZ2_RSZ1_RSZ1_RSZ1_RSZ1_RSZ1_RSZ1
Status: Status.UNKNOWN
Output dim: 1, lower bound: -0.0002842, upper bound: 0.0002781
time: 0.57 seconds

### Relational analysis RSZ of RS_RSZ2_RSZ1_RSZ1_RSZ1_RSZ1_RSZ1_RSZ2

#### Relational analysis RSZ result of RS_RSZ2_RSZ1_RSZ1_RSZ1_RSZ1_RSZ1_RSZ2
Status: Status.UNKNOWN
Output dim: 1, lower bound: -0.0002833, upper bound: 0.0002804
time: 0.59 seconds

## BFS RS instance: RS_RSZ2_RSZ1_RSZ1_RSZ1_RSZ1_RSZ2

### Backsubstitution after applying RS history:
0: 0.0009457, 0.0011003, 0.0009457, 0.0011003, -0.0001494, 0.0001489
1: 0.9936197, 0.9940771, 0.9936197, 0.9940771, -0.0003872, 0.0003927
2: -0.0065606, -0.0050208, -0.0065606, -0.0050208, -0.0010769, 0.0011348
3: 0.0038004, 0.0040140, 0.0038004, 0.0040140, -0.0001686, 0.0001648
4: 0.0023852, 0.0036022, 0.0023852, 0.0036022, -0.0009839, 0.0009379
5: 0.0060086, 0.0065111, 0.0060086, 0.0065111, -0.0005024, 0.0005024
6: -0.0013983, -0.0008639, -0.0013983, -0.0008639, -0.0003686, 0.0003888
7: -0.0083221, -0.0079387, -0.0083221, -0.0079387, -0.0003835, 0.0003835
8: 0.0048484, 0.0068715, 0.0048484, 0.0068715, -0.0012513, 0.0011751
9: -0.0036877, -0.0032079, -0.0036877, -0.0032079, -0.0004797, 0.0004797

### Unstable ReLU Count (Linear/Conv2D Layers)
- layer_idx=0, type=LayerType.Linear, total=256, inp1_unstable=5, inp2_unstable=5, delta_unstable=251
- layer_idx=2, type=LayerType.Linear, total=256, inp1_unstable=4, inp2_unstable=4, delta_unstable=255
- layer_idx=4, type=LayerType.Linear, total=256, inp1_unstable=0, inp2_unstable=0, delta_unstable=256
- layer_idx=6, type=LayerType.Linear, total=256, inp1_unstable=1, inp2_unstable=1, delta_unstable=256
- layer_idx=8, type=LayerType.Linear, total=10, inp1_unstable=0, inp2_unstable=0, delta_unstable=10

Time for backsubstitution: 1.16 seconds

### RS candidates at layer 3
type: RSZ, layer: 3, pos: 80
type: RSZ, layer: 3, pos: 81
type: RSZ, layer: 3, pos: 136

Time for candidate selection: 0.12 seconds

### Candidate
type: RSZ, layer: 3, pos: 80

### Relational analysis RSZ of RS_RSZ2_RSZ1_RSZ1_RSZ1_RSZ1_RSZ2_RSZ1

#### Relational analysis RSZ result of RS_RSZ2_RSZ1_RSZ1_RSZ1_RSZ1_RSZ2_RSZ1
Status: Status.UNKNOWN
Output dim: 1, lower bound: -0.0002803, upper bound: 0.0002828
time: 0.55 seconds

### Relational analysis RSZ of RS_RSZ2_RSZ1_RSZ1_RSZ1_RSZ1_RSZ2_RSZ2

#### Relational analysis RSZ result of RS_RSZ2_RSZ1_RSZ1_RSZ1_RSZ1_RSZ2_RSZ2
Status: Status.UNKNOWN
Output dim: 1, lower bound: -0.0002785, upper bound: 0.0002841
time: 0.59 seconds

## BFS RS instance: RS_RSZ2_RSZ1_RSZ1_RSZ1_RSZ2_RSZ1

### Backsubstitution after applying RS history:
0: 0.0009457, 0.0011003, 0.0009457, 0.0011003, -0.0001489, 0.0001495
1: 0.9936197, 0.9940771, 0.9936197, 0.9940771, -0.0003936, 0.0003866
2: -0.0065606, -0.0050208, -0.0065606, -0.0050208, -0.0011441, 0.0010703
3: 0.0038004, 0.0040140, 0.0038004, 0.0040140, -0.0001644, 0.0001692
4: 0.0023852, 0.0036022, 0.0023852, 0.0036022, -0.0009329, 0.0009909
5: 0.0060086, 0.0065111, 0.0060086, 0.0065111, -0.0005024, 0.0005024
6: -0.0013983, -0.0008639, -0.0013983, -0.0008639, -0.0003920, 0.0003664
7: -0.0083221, -0.0079387, -0.0083221, -0.0079387, -0.0003835, 0.0003835
8: 0.0048484, 0.0068715, 0.0048484, 0.0068715, -0.0011666, 0.0012634
9: -0.0036877, -0.0032079, -0.0036877, -0.0032079, -0.0004797, 0.0004797

### Unstable ReLU Count (Linear/Conv2D Layers)
- layer_idx=0, type=LayerType.Linear, total=256, inp1_unstable=5, inp2_unstable=5, delta_unstable=251
- layer_idx=2, type=LayerType.Linear, total=256, inp1_unstable=4, inp2_unstable=4, delta_unstable=255
- layer_idx=4, type=LayerType.Linear, total=256, inp1_unstable=0, inp2_unstable=0, delta_unstable=256
- layer_idx=6, type=LayerType.Linear, total=256, inp1_unstable=1, inp2_unstable=1, delta_unstable=256
- layer_idx=8, type=LayerType.Linear, total=10, inp1_unstable=0, inp2_unstable=0, delta_unstable=10

Time for backsubstitution: 1.26 seconds

### RS candidates at layer 3
type: RSZ, layer: 3, pos: 80
type: RSZ, layer: 3, pos: 81
type: RSZ, layer: 3, pos: 136

Time for candidate selection: 0.12 seconds

### Candidate
type: RSZ, layer: 3, pos: 80

### Relational analysis RSZ of RS_RSZ2_RSZ1_RSZ1_RSZ1_RSZ2_RSZ1_RSZ1

#### Relational analysis RSZ result of RS_RSZ2_RSZ1_RSZ1_RSZ1_RSZ2_RSZ1_RSZ1
Status: Status.UNKNOWN
Output dim: 1, lower bound: -0.0002840, upper bound: 0.0002780
time: 0.56 seconds

### Relational analysis RSZ of RS_RSZ2_RSZ1_RSZ1_RSZ1_RSZ2_RSZ1_RSZ2

#### Relational analysis RSZ result of RS_RSZ2_RSZ1_RSZ1_RSZ1_RSZ2_RSZ1_RSZ2
Status: Status.UNKNOWN
Output dim: 1, lower bound: -0.0002833, upper bound: 0.0002804
time: 0.54 seconds

## BFS RS instance: RS_RSZ2_RSZ1_RSZ1_RSZ1_RSZ2_RSZ2

### Backsubstitution after applying RS history:
0: 0.0009457, 0.0011003, 0.0009457, 0.0011003, -0.0001494, 0.0001489
1: 0.9936197, 0.9940771, 0.9936197, 0.9940771, -0.0003869, 0.0003927
2: -0.0065606, -0.0050208, -0.0065606, -0.0050208, -0.0010739, 0.0011343
3: 0.0038004, 0.0040140, 0.0038004, 0.0040140, -0.0001686, 0.0001646
4: 0.0023852, 0.0036022, 0.0023852, 0.0036022, -0.0009835, 0.0009356
5: 0.0060086, 0.0065111, 0.0060086, 0.0065111, -0.0005024, 0.0005024
6: -0.0013983, -0.0008639, -0.0013983, -0.0008639, -0.0003676, 0.0003886
7: -0.0083221, -0.0079387, -0.0083221, -0.0079387, -0.0003835, 0.0003835
8: 0.0048484, 0.0068715, 0.0048484, 0.0068715, -0.0012507, 0.0011713
9: -0.0036877, -0.0032079, -0.0036877, -0.0032079, -0.0004797, 0.0004797

### Unstable ReLU Count (Linear/Conv2D Layers)
- layer_idx=0, type=LayerType.Linear, total=256, inp1_unstable=5, inp2_unstable=5, delta_unstable=251
- layer_idx=2, type=LayerType.Linear, total=256, inp1_unstable=4, inp2_unstable=4, delta_unstable=255
- layer_idx=4, type=LayerType.Linear, total=256, inp1_unstable=0, inp2_unstable=0, delta_unstable=256
- layer_idx=6, type=LayerType.Linear, total=256, inp1_unstable=1, inp2_unstable=1, delta_unstable=256
- layer_idx=8, type=LayerType.Linear, total=10, inp1_unstable=0, inp2_unstable=0, delta_unstable=10

Time for backsubstitution: 1.17 seconds

### RS candidates at layer 3
type: RSZ, layer: 3, pos: 80
type: RSZ, layer: 3, pos: 81
type: RSZ, layer: 3, pos: 136

Time for candidate selection: 0.12 seconds

### Candidate
type: RSZ, layer: 3, pos: 80

### Relational analysis RSZ of RS_RSZ2_RSZ1_RSZ1_RSZ1_RSZ2_RSZ2_RSZ1

#### Relational analysis RSZ result of RS_RSZ2_RSZ1_RSZ1_RSZ1_RSZ2_RSZ2_RSZ1
Status: Status.UNKNOWN
Output dim: 1, lower bound: -0.0002802, upper bound: 0.0002827
time: 0.61 seconds

### Relational analysis RSZ of RS_RSZ2_RSZ1_RSZ1_RSZ1_RSZ2_RSZ2_RSZ2

#### Relational analysis RSZ result of RS_RSZ2_RSZ1_RSZ1_RSZ1_RSZ2_RSZ2_RSZ2
Status: Status.UNKNOWN
Output dim: 1, lower bound: -0.0002785, upper bound: 0.0002840
time: 0.57 seconds

## BFS RS instance: RS_RSZ2_RSZ1_RSZ1_RSZ2_RSZ1_RSZ1

### Backsubstitution after applying RS history:
0: 0.0009457, 0.0011003, 0.0009457, 0.0011003, -0.0001489, 0.0001494
1: 0.9936197, 0.9940771, 0.9936197, 0.9940771, -0.0003932, 0.0003863
2: -0.0065606, -0.0050208, -0.0065606, -0.0050208, -0.0011404, 0.0010674
3: 0.0038004, 0.0040140, 0.0038004, 0.0040140, -0.0001642, 0.0001690
4: 0.0023852, 0.0036022, 0.0023852, 0.0036022, -0.0009306, 0.0009881
5: 0.0060086, 0.0065111, 0.0060086, 0.0065111, -0.0005024, 0.0005024
6: -0.0013983, -0.0008639, -0.0013983, -0.0008639, -0.0003907, 0.0003654
7: -0.0083221, -0.0079387, -0.0083221, -0.0079387, -0.0003835, 0.0003835
8: 0.0048484, 0.0068715, 0.0048484, 0.0068715, -0.0011628, 0.0012587
9: -0.0036877, -0.0032079, -0.0036877, -0.0032079, -0.0004797, 0.0004797

### Unstable ReLU Count (Linear/Conv2D Layers)
- layer_idx=0, type=LayerType.Linear, total=256, inp1_unstable=5, inp2_unstable=5, delta_unstable=251
- layer_idx=2, type=LayerType.Linear, total=256, inp1_unstable=4, inp2_unstable=4, delta_unstable=255
- layer_idx=4, type=LayerType.Linear, total=256, inp1_unstable=0, inp2_unstable=0, delta_unstable=256
- layer_idx=6, type=LayerType.Linear, total=256, inp1_unstable=1, inp2_unstable=1, delta_unstable=256
- layer_idx=8, type=LayerType.Linear, total=10, inp1_unstable=0, inp2_unstable=0, delta_unstable=10

Time for backsubstitution: 1.18 seconds

### RS candidates at layer 3
type: RSZ, layer: 3, pos: 80
type: RSZ, layer: 3, pos: 81
type: RSZ, layer: 3, pos: 136

Time for candidate selection: 0.12 seconds

### Candidate
type: RSZ, layer: 3, pos: 80

### Relational analysis RSZ of RS_RSZ2_RSZ1_RSZ1_RSZ2_RSZ1_RSZ1_RSZ1

#### Relational analysis RSZ result of RS_RSZ2_RSZ1_RSZ1_RSZ2_RSZ1_RSZ1_RSZ1
Status: Status.UNKNOWN
Output dim: 1, lower bound: -0.0002831, upper bound: 0.0002785
time: 0.64 seconds

### Relational analysis RSZ of RS_RSZ2_RSZ1_RSZ1_RSZ2_RSZ1_RSZ1_RSZ2

#### Relational analysis RSZ result of RS_RSZ2_RSZ1_RSZ1_RSZ2_RSZ1_RSZ1_RSZ2
Status: Status.UNKNOWN
Output dim: 1, lower bound: -0.0002826, upper bound: 0.0002810
time: 0.58 seconds

## BFS RS instance: RS_RSZ2_RSZ1_RSZ1_RSZ2_RSZ1_RSZ2

### Backsubstitution after applying RS history:
0: 0.0009457, 0.0011003, 0.0009457, 0.0011003, -0.0001494, 0.0001490
1: 0.9936197, 0.9940771, 0.9936197, 0.9940771, -0.0003873, 0.0003928
2: -0.0065606, -0.0050208, -0.0065606, -0.0050208, -0.0010786, 0.0011358
3: 0.0038004, 0.0040140, 0.0038004, 0.0040140, -0.0001687, 0.0001649
4: 0.0023852, 0.0036022, 0.0023852, 0.0036022, -0.0009847, 0.0009392
5: 0.0060086, 0.0065111, 0.0060086, 0.0065111, -0.0005024, 0.0005024
6: -0.0013983, -0.0008639, -0.0013983, -0.0008639, -0.0003692, 0.0003891
7: -0.0083221, -0.0079387, -0.0083221, -0.0079387, -0.0003835, 0.0003835
8: 0.0048484, 0.0068715, 0.0048484, 0.0068715, -0.0012527, 0.0011774
9: -0.0036877, -0.0032079, -0.0036877, -0.0032079, -0.0004797, 0.0004797

### Unstable ReLU Count (Linear/Conv2D Layers)
- layer_idx=0, type=LayerType.Linear, total=256, inp1_unstable=5, inp2_unstable=5, delta_unstable=251
- layer_idx=2, type=LayerType.Linear, total=256, inp1_unstable=4, inp2_unstable=4, delta_unstable=255
- layer_idx=4, type=LayerType.Linear, total=256, inp1_unstable=0, inp2_unstable=0, delta_unstable=256
- layer_idx=6, type=LayerType.Linear, total=256, inp1_unstable=1, inp2_unstable=1, delta_unstable=256
- layer_idx=8, type=LayerType.Linear, total=10, inp1_unstable=0, inp2_unstable=0, delta_unstable=10

Time for backsubstitution: 1.21 seconds

### RS candidates at layer 3
type: RSZ, layer: 3, pos: 80
type: RSZ, layer: 3, pos: 81
type: RSZ, layer: 3, pos: 136

Time for candidate selection: 0.12 seconds

### Candidate
type: RSZ, layer: 3, pos: 80

### Relational analysis RSZ of RS_RSZ2_RSZ1_RSZ1_RSZ2_RSZ1_RSZ2_RSZ1

#### Relational analysis RSZ result of RS_RSZ2_RSZ1_RSZ1_RSZ2_RSZ1_RSZ2_RSZ1
Status: Status.UNKNOWN
Output dim: 1, lower bound: -0.0002796, upper bound: 0.0002835
time: 0.56 seconds

### Relational analysis RSZ of RS_RSZ2_RSZ1_RSZ1_RSZ2_RSZ1_RSZ2_RSZ2

#### Relational analysis RSZ result of RS_RSZ2_RSZ1_RSZ1_RSZ2_RSZ1_RSZ2_RSZ2
Status: Status.UNKNOWN
Output dim: 1, lower bound: -0.0002779, upper bound: 0.0002850
time: 0.58 seconds

## BFS RS instance: RS_RSZ2_RSZ1_RSZ1_RSZ2_RSZ2_RSZ1

### Backsubstitution after applying RS history:
0: 0.0009457, 0.0011003, 0.0009457, 0.0011003, -0.0001489, 0.0001494
1: 0.9936197, 0.9940771, 0.9936197, 0.9940771, -0.0003932, 0.0003866
2: -0.0065606, -0.0050208, -0.0065606, -0.0050208, -0.0011406, 0.0010705
3: 0.0038004, 0.0040140, 0.0038004, 0.0040140, -0.0001644, 0.0001690
4: 0.0023852, 0.0036022, 0.0023852, 0.0036022, -0.0009331, 0.0009882
5: 0.0060086, 0.0065111, 0.0060086, 0.0065111, -0.0005024, 0.0005024
6: -0.0013983, -0.0008639, -0.0013983, -0.0008639, -0.0003908, 0.0003665
7: -0.0083221, -0.0079387, -0.0083221, -0.0079387, -0.0003835, 0.0003835
8: 0.0048484, 0.0068715, 0.0048484, 0.0068715, -0.0011669, 0.0012588
9: -0.0036877, -0.0032079, -0.0036877, -0.0032079, -0.0004797, 0.0004797

### Unstable ReLU Count (Linear/Conv2D Layers)
- layer_idx=0, type=LayerType.Linear, total=256, inp1_unstable=5, inp2_unstable=5, delta_unstable=251
- layer_idx=2, type=LayerType.Linear, total=256, inp1_unstable=4, inp2_unstable=4, delta_unstable=255
- layer_idx=4, type=LayerType.Linear, total=256, inp1_unstable=0, inp2_unstable=0, delta_unstable=256
- layer_idx=6, type=LayerType.Linear, total=256, inp1_unstable=1, inp2_unstable=1, delta_unstable=256
- layer_idx=8, type=LayerType.Linear, total=10, inp1_unstable=0, inp2_unstable=0, delta_unstable=10

Time for backsubstitution: 1.27 seconds

### RS candidates at layer 3
type: RSZ, layer: 3, pos: 80
type: RSZ, layer: 3, pos: 81
type: RSZ, layer: 3, pos: 136

Time for candidate selection: 0.12 seconds

### Candidate
type: RSZ, layer: 3, pos: 80

### Relational analysis RSZ of RS_RSZ2_RSZ1_RSZ1_RSZ2_RSZ2_RSZ1_RSZ1

#### Relational analysis RSZ result of RS_RSZ2_RSZ1_RSZ1_RSZ2_RSZ2_RSZ1_RSZ1
Status: Status.UNKNOWN
Output dim: 1, lower bound: -0.0002831, upper bound: 0.0002785
time: 0.76 seconds

### Relational analysis RSZ of RS_RSZ2_RSZ1_RSZ1_RSZ2_RSZ2_RSZ1_RSZ2

#### Relational analysis RSZ result of RS_RSZ2_RSZ1_RSZ1_RSZ2_RSZ2_RSZ1_RSZ2
Status: Status.UNKNOWN
Output dim: 1, lower bound: -0.0002827, upper bound: 0.0002810
time: 0.61 seconds

## BFS RS instance: RS_RSZ2_RSZ1_RSZ1_RSZ2_RSZ2_RSZ2

### Backsubstitution after applying RS history:
0: 0.0009457, 0.0011003, 0.0009457, 0.0011003, -0.0001494, 0.0001489
1: 0.9936197, 0.9940771, 0.9936197, 0.9940771, -0.0003872, 0.0003928
2: -0.0065606, -0.0050208, -0.0065606, -0.0050208, -0.0010770, 0.0011357
3: 0.0038004, 0.0040140, 0.0038004, 0.0040140, -0.0001687, 0.0001648
4: 0.0023852, 0.0036022, 0.0023852, 0.0036022, -0.0009846, 0.0009380
5: 0.0060086, 0.0065111, 0.0060086, 0.0065111, -0.0005024, 0.0005024
6: -0.0013983, -0.0008639, -0.0013983, -0.0008639, -0.0003687, 0.0003891
7: -0.0083221, -0.0079387, -0.0083221, -0.0079387, -0.0003835, 0.0003835
8: 0.0048484, 0.0068715, 0.0048484, 0.0068715, -0.0012525, 0.0011753
9: -0.0036877, -0.0032079, -0.0036877, -0.0032079, -0.0004797, 0.0004797

### Unstable ReLU Count (Linear/Conv2D Layers)
- layer_idx=0, type=LayerType.Linear, total=256, inp1_unstable=5, inp2_unstable=5, delta_unstable=251
- layer_idx=2, type=LayerType.Linear, total=256, inp1_unstable=4, inp2_unstable=4, delta_unstable=255
- layer_idx=4, type=LayerType.Linear, total=256, inp1_unstable=0, inp2_unstable=0, delta_unstable=256
- layer_idx=6, type=LayerType.Linear, total=256, inp1_unstable=1, inp2_unstable=1, delta_unstable=256
- layer_idx=8, type=LayerType.Linear, total=10, inp1_unstable=0, inp2_unstable=0, delta_unstable=10

Time for backsubstitution: 1.22 seconds

### RS candidates at layer 3
type: RSZ, layer: 3, pos: 80
type: RSZ, layer: 3, pos: 81
type: RSZ, layer: 3, pos: 136

Time for candidate selection: 0.14 seconds

### Candidate
type: RSZ, layer: 3, pos: 80

### Relational analysis RSZ of RS_RSZ2_RSZ1_RSZ1_RSZ2_RSZ2_RSZ2_RSZ1

#### Relational analysis RSZ result of RS_RSZ2_RSZ1_RSZ1_RSZ2_RSZ2_RSZ2_RSZ1
Status: Status.UNKNOWN
Output dim: 1, lower bound: -0.0002795, upper bound: 0.0002835
time: 0.56 seconds

### Relational analysis RSZ of RS_RSZ2_RSZ1_RSZ1_RSZ2_RSZ2_RSZ2_RSZ2

#### Relational analysis RSZ result of RS_RSZ2_RSZ1_RSZ1_RSZ2_RSZ2_RSZ2_RSZ2
Status: Status.UNKNOWN
Output dim: 1, lower bound: -0.0002779, upper bound: 0.0002850
time: 0.57 seconds

## BFS RS instance: RS_RSZ2_RSZ1_RSZ2_RSZ1_RSZ1_RSZ1

### Backsubstitution after applying RS history:
0: 0.0009457, 0.0011003, 0.0009457, 0.0011003, -0.0001489, 0.0001495
1: 0.9936197, 0.9940771, 0.9936197, 0.9940771, -0.0003937, 0.0003863
2: -0.0065606, -0.0050208, -0.0065606, -0.0050208, -0.0011457, 0.0010677
3: 0.0038004, 0.0040140, 0.0038004, 0.0040140, -0.0001642, 0.0001694
4: 0.0023852, 0.0036022, 0.0023852, 0.0036022, -0.0009308, 0.0009923
5: 0.0060086, 0.0065111, 0.0060086, 0.0065111, -0.0005024, 0.0005024
6: -0.0013983, -0.0008639, -0.0013983, -0.0008639, -0.0003925, 0.0003655
7: -0.0083221, -0.0079387, -0.0083221, -0.0079387, -0.0003835, 0.0003835
8: 0.0048484, 0.0068715, 0.0048484, 0.0068715, -0.0011632, 0.0012656
9: -0.0036877, -0.0032079, -0.0036877, -0.0032079, -0.0004797, 0.0004797

### Unstable ReLU Count (Linear/Conv2D Layers)
- layer_idx=0, type=LayerType.Linear, total=256, inp1_unstable=5, inp2_unstable=5, delta_unstable=251
- layer_idx=2, type=LayerType.Linear, total=256, inp1_unstable=4, inp2_unstable=4, delta_unstable=255
- layer_idx=4, type=LayerType.Linear, total=256, inp1_unstable=0, inp2_unstable=0, delta_unstable=256
- layer_idx=6, type=LayerType.Linear, total=256, inp1_unstable=1, inp2_unstable=1, delta_unstable=256
- layer_idx=8, type=LayerType.Linear, total=10, inp1_unstable=0, inp2_unstable=0, delta_unstable=10

Time for backsubstitution: 1.20 seconds

### RS candidates at layer 3
type: RSZ, layer: 3, pos: 80
type: RSZ, layer: 3, pos: 81
type: RSZ, layer: 3, pos: 136

Time for candidate selection: 0.12 seconds

### Candidate
type: RSZ, layer: 3, pos: 80

### Relational analysis RSZ of RS_RSZ2_RSZ1_RSZ2_RSZ1_RSZ1_RSZ1_RSZ1

#### Relational analysis RSZ result of RS_RSZ2_RSZ1_RSZ2_RSZ1_RSZ1_RSZ1_RSZ1
Status: Status.UNKNOWN
Output dim: 1, lower bound: -0.0002850, upper bound: 0.0002779
time: 0.58 seconds

### Relational analysis RSZ of RS_RSZ2_RSZ1_RSZ2_RSZ1_RSZ1_RSZ1_RSZ2

#### Relational analysis RSZ result of RS_RSZ2_RSZ1_RSZ2_RSZ1_RSZ1_RSZ1_RSZ2
Status: Status.UNKNOWN
Output dim: 1, lower bound: -0.0002835, upper bound: 0.0002796
time: 0.59 seconds

## BFS RS instance: RS_RSZ2_RSZ1_RSZ2_RSZ1_RSZ1_RSZ2

### Backsubstitution after applying RS history:
0: 0.0009457, 0.0011003, 0.0009457, 0.0011003, -0.0001493, 0.0001490
1: 0.9936197, 0.9940771, 0.9936197, 0.9940771, -0.0003875, 0.0003922
2: -0.0065606, -0.0050208, -0.0065606, -0.0050208, -0.0010806, 0.0011298
3: 0.0038004, 0.0040140, 0.0038004, 0.0040140, -0.0001683, 0.0001650
4: 0.0023852, 0.0036022, 0.0023852, 0.0036022, -0.0009799, 0.0009408
5: 0.0060086, 0.0065111, 0.0060086, 0.0065111, -0.0005024, 0.0005024
6: -0.0013983, -0.0008639, -0.0013983, -0.0008639, -0.0003699, 0.0003870
7: -0.0083221, -0.0079387, -0.0083221, -0.0079387, -0.0003835, 0.0003835
8: 0.0048484, 0.0068715, 0.0048484, 0.0068715, -0.0012447, 0.0011801
9: -0.0036877, -0.0032079, -0.0036877, -0.0032079, -0.0004797, 0.0004797

### Unstable ReLU Count (Linear/Conv2D Layers)
- layer_idx=0, type=LayerType.Linear, total=256, inp1_unstable=5, inp2_unstable=5, delta_unstable=251
- layer_idx=2, type=LayerType.Linear, total=256, inp1_unstable=4, inp2_unstable=4, delta_unstable=255
- layer_idx=4, type=LayerType.Linear, total=256, inp1_unstable=0, inp2_unstable=0, delta_unstable=256
- layer_idx=6, type=LayerType.Linear, total=256, inp1_unstable=1, inp2_unstable=1, delta_unstable=256
- layer_idx=8, type=LayerType.Linear, total=10, inp1_unstable=0, inp2_unstable=0, delta_unstable=10

Time for backsubstitution: 1.19 seconds

### RS candidates at layer 3
type: RSZ, layer: 3, pos: 80
type: RSZ, layer: 3, pos: 81
type: RSZ, layer: 3, pos: 136

Time for candidate selection: 0.15 seconds

### Candidate
type: RSZ, layer: 3, pos: 80

### Relational analysis RSZ of RS_RSZ2_RSZ1_RSZ2_RSZ1_RSZ1_RSZ2_RSZ1

#### Relational analysis RSZ result of RS_RSZ2_RSZ1_RSZ2_RSZ1_RSZ1_RSZ2_RSZ1
Status: Status.UNKNOWN
Output dim: 1, lower bound: -0.0002810, upper bound: 0.0002827
time: 0.60 seconds

### Relational analysis RSZ of RS_RSZ2_RSZ1_RSZ2_RSZ1_RSZ1_RSZ2_RSZ2

#### Relational analysis RSZ result of RS_RSZ2_RSZ1_RSZ2_RSZ1_RSZ1_RSZ2_RSZ2
Status: Status.UNKNOWN
Output dim: 1, lower bound: -0.0002785, upper bound: 0.0002832
time: 0.64 seconds

## BFS RS instance: RS_RSZ2_RSZ1_RSZ2_RSZ1_RSZ2_RSZ1

### Backsubstitution after applying RS history:
0: 0.0009457, 0.0011003, 0.0009457, 0.0011003, -0.0001489, 0.0001495
1: 0.9936197, 0.9940771, 0.9936197, 0.9940771, -0.0003937, 0.0003865
2: -0.0065606, -0.0050208, -0.0065606, -0.0050208, -0.0011458, 0.0010695
3: 0.0038004, 0.0040140, 0.0038004, 0.0040140, -0.0001643, 0.0001694
4: 0.0023852, 0.0036022, 0.0023852, 0.0036022, -0.0009322, 0.0009923
5: 0.0060086, 0.0065111, 0.0060086, 0.0065111, -0.0005024, 0.0005024
6: -0.0013983, -0.0008639, -0.0013983, -0.0008639, -0.0003926, 0.0003661
7: -0.0083221, -0.0079387, -0.0083221, -0.0079387, -0.0003835, 0.0003835
8: 0.0048484, 0.0068715, 0.0048484, 0.0068715, -0.0011655, 0.0012657
9: -0.0036877, -0.0032079, -0.0036877, -0.0032079, -0.0004797, 0.0004797

### Unstable ReLU Count (Linear/Conv2D Layers)
- layer_idx=0, type=LayerType.Linear, total=256, inp1_unstable=5, inp2_unstable=5, delta_unstable=251
- layer_idx=2, type=LayerType.Linear, total=256, inp1_unstable=4, inp2_unstable=4, delta_unstable=255
- layer_idx=4, type=LayerType.Linear, total=256, inp1_unstable=0, inp2_unstable=0, delta_unstable=256
- layer_idx=6, type=LayerType.Linear, total=256, inp1_unstable=1, inp2_unstable=1, delta_unstable=256
- layer_idx=8, type=LayerType.Linear, total=10, inp1_unstable=0, inp2_unstable=0, delta_unstable=10

Time for backsubstitution: 1.25 seconds

### RS candidates at layer 3
type: RSZ, layer: 3, pos: 80
type: RSZ, layer: 3, pos: 81
type: RSZ, layer: 3, pos: 136

Time for candidate selection: 0.12 seconds

### Candidate
type: RSZ, layer: 3, pos: 80

### Relational analysis RSZ of RS_RSZ2_RSZ1_RSZ2_RSZ1_RSZ2_RSZ1_RSZ1

#### Relational analysis RSZ result of RS_RSZ2_RSZ1_RSZ2_RSZ1_RSZ2_RSZ1_RSZ1
Status: Status.UNKNOWN
Output dim: 1, lower bound: -0.0002849, upper bound: 0.0002779
time: 0.57 seconds

### Relational analysis RSZ of RS_RSZ2_RSZ1_RSZ2_RSZ1_RSZ2_RSZ1_RSZ2

#### Relational analysis RSZ result of RS_RSZ2_RSZ1_RSZ2_RSZ1_RSZ2_RSZ1_RSZ2
Status: Status.UNKNOWN
Output dim: 1, lower bound: -0.0002835, upper bound: 0.0002797
time: 0.59 seconds

## BFS RS instance: RS_RSZ2_RSZ1_RSZ2_RSZ1_RSZ2_RSZ2

### Backsubstitution after applying RS history:
0: 0.0009457, 0.0011003, 0.0009457, 0.0011003, -0.0001493, 0.0001489
1: 0.9936197, 0.9940771, 0.9936197, 0.9940771, -0.0003873, 0.0003922
2: -0.0065606, -0.0050208, -0.0065606, -0.0050208, -0.0010776, 0.0011298
3: 0.0038004, 0.0040140, 0.0038004, 0.0040140, -0.0001683, 0.0001648
4: 0.0023852, 0.0036022, 0.0023852, 0.0036022, -0.0009799, 0.0009385
5: 0.0060086, 0.0065111, 0.0060086, 0.0065111, -0.0005024, 0.0005024
6: -0.0013983, -0.0008639, -0.0013983, -0.0008639, -0.0003689, 0.0003870
7: -0.0083221, -0.0079387, -0.0083221, -0.0079387, -0.0003835, 0.0003835
8: 0.0048484, 0.0068715, 0.0048484, 0.0068715, -0.0012447, 0.0011761
9: -0.0036877, -0.0032079, -0.0036877, -0.0032079, -0.0004797, 0.0004797

### Unstable ReLU Count (Linear/Conv2D Layers)
- layer_idx=0, type=LayerType.Linear, total=256, inp1_unstable=5, inp2_unstable=5, delta_unstable=251
- layer_idx=2, type=LayerType.Linear, total=256, inp1_unstable=4, inp2_unstable=4, delta_unstable=255
- layer_idx=4, type=LayerType.Linear, total=256, inp1_unstable=0, inp2_unstable=0, delta_unstable=256
- layer_idx=6, type=LayerType.Linear, total=256, inp1_unstable=1, inp2_unstable=1, delta_unstable=256
- layer_idx=8, type=LayerType.Linear, total=10, inp1_unstable=0, inp2_unstable=0, delta_unstable=10

Time for backsubstitution: 1.20 seconds

### RS candidates at layer 3
type: RSZ, layer: 3, pos: 80
type: RSZ, layer: 3, pos: 81
type: RSZ, layer: 3, pos: 136

Time for candidate selection: 0.13 seconds

### Candidate
type: RSZ, layer: 3, pos: 80

### Relational analysis RSZ of RS_RSZ2_RSZ1_RSZ2_RSZ1_RSZ2_RSZ2_RSZ1

#### Relational analysis RSZ result of RS_RSZ2_RSZ1_RSZ2_RSZ1_RSZ2_RSZ2_RSZ1
Status: Status.UNKNOWN
Output dim: 1, lower bound: -0.0002810, upper bound: 0.0002826
time: 0.58 seconds

### Relational analysis RSZ of RS_RSZ2_RSZ1_RSZ2_RSZ1_RSZ2_RSZ2_RSZ2

#### Relational analysis RSZ result of RS_RSZ2_RSZ1_RSZ2_RSZ1_RSZ2_RSZ2_RSZ2
Status: Status.UNKNOWN
Output dim: 1, lower bound: -0.0002785, upper bound: 0.0002832
time: 0.59 seconds

## BFS RS instance: RS_RSZ2_RSZ1_RSZ2_RSZ2_RSZ1_RSZ1

### Backsubstitution after applying RS history:
0: 0.0009457, 0.0011003, 0.0009457, 0.0011003, -0.0001488, 0.0001495
1: 0.9936197, 0.9940771, 0.9936197, 0.9940771, -0.0003936, 0.0003859
2: -0.0065606, -0.0050208, -0.0065606, -0.0050208, -0.0011439, 0.0010636
3: 0.0038004, 0.0040140, 0.0038004, 0.0040140, -0.0001639, 0.0001692
4: 0.0023852, 0.0036022, 0.0023852, 0.0036022, -0.0009276, 0.0009908
5: 0.0060086, 0.0065111, 0.0060086, 0.0065111, -0.0005024, 0.0005024
6: -0.0013983, -0.0008639, -0.0013983, -0.0008639, -0.0003919, 0.0003641
7: -0.0083221, -0.0079387, -0.0083221, -0.0079387, -0.0003835, 0.0003835
8: 0.0048484, 0.0068715, 0.0048484, 0.0068715, -0.0011578, 0.0012632
9: -0.0036877, -0.0032079, -0.0036877, -0.0032079, -0.0004797, 0.0004797

### Unstable ReLU Count (Linear/Conv2D Layers)
- layer_idx=0, type=LayerType.Linear, total=256, inp1_unstable=5, inp2_unstable=5, delta_unstable=251
- layer_idx=2, type=LayerType.Linear, total=256, inp1_unstable=4, inp2_unstable=4, delta_unstable=255
- layer_idx=4, type=LayerType.Linear, total=256, inp1_unstable=0, inp2_unstable=0, delta_unstable=256
- layer_idx=6, type=LayerType.Linear, total=256, inp1_unstable=1, inp2_unstable=1, delta_unstable=256
- layer_idx=8, type=LayerType.Linear, total=10, inp1_unstable=0, inp2_unstable=0, delta_unstable=10

Time for backsubstitution: 1.23 seconds

### RS candidates at layer 3
type: RSZ, layer: 3, pos: 80
type: RSZ, layer: 3, pos: 81
type: RSZ, layer: 3, pos: 136

Time for candidate selection: 0.13 seconds

### Candidate
type: RSZ, layer: 3, pos: 80

### Relational analysis RSZ of RS_RSZ2_RSZ1_RSZ2_RSZ2_RSZ1_RSZ1_RSZ1

#### Relational analysis RSZ result of RS_RSZ2_RSZ1_RSZ2_RSZ2_RSZ1_RSZ1_RSZ1
Status: Status.UNKNOWN
Output dim: 1, lower bound: -0.0002839, upper bound: 0.0002785
time: 0.58 seconds

### Relational analysis RSZ of RS_RSZ2_RSZ1_RSZ2_RSZ2_RSZ1_RSZ1_RSZ2

#### Relational analysis RSZ result of RS_RSZ2_RSZ1_RSZ2_RSZ2_RSZ1_RSZ1_RSZ2
Status: Status.UNKNOWN
Output dim: 1, lower bound: -0.0002827, upper bound: 0.0002802
time: 0.61 seconds

## BFS RS instance: RS_RSZ2_RSZ1_RSZ2_RSZ2_RSZ1_RSZ2

### Backsubstitution after applying RS history:
0: 0.0009457, 0.0011003, 0.0009457, 0.0011003, -0.0001494, 0.0001490
1: 0.9936197, 0.9940771, 0.9936197, 0.9940771, -0.0003873, 0.0003924
2: -0.0065606, -0.0050208, -0.0065606, -0.0050208, -0.0010786, 0.0011311
3: 0.0038004, 0.0040140, 0.0038004, 0.0040140, -0.0001684, 0.0001649
4: 0.0023852, 0.0036022, 0.0023852, 0.0036022, -0.0009810, 0.0009392
5: 0.0060086, 0.0065111, 0.0060086, 0.0065111, -0.0005024, 0.0005024
6: -0.0013983, -0.0008639, -0.0013983, -0.0008639, -0.0003692, 0.0003875
7: -0.0083221, -0.0079387, -0.0083221, -0.0079387, -0.0003835, 0.0003835
8: 0.0048484, 0.0068715, 0.0048484, 0.0068715, -0.0012465, 0.0011774
9: -0.0036877, -0.0032079, -0.0036877, -0.0032079, -0.0004797, 0.0004797

### Unstable ReLU Count (Linear/Conv2D Layers)
- layer_idx=0, type=LayerType.Linear, total=256, inp1_unstable=5, inp2_unstable=5, delta_unstable=251
- layer_idx=2, type=LayerType.Linear, total=256, inp1_unstable=4, inp2_unstable=4, delta_unstable=255
- layer_idx=4, type=LayerType.Linear, total=256, inp1_unstable=0, inp2_unstable=0, delta_unstable=256
- layer_idx=6, type=LayerType.Linear, total=256, inp1_unstable=1, inp2_unstable=1, delta_unstable=256
- layer_idx=8, type=LayerType.Linear, total=10, inp1_unstable=0, inp2_unstable=0, delta_unstable=10

Time for backsubstitution: 1.21 seconds

### RS candidates at layer 3
type: RSZ, layer: 3, pos: 80
type: RSZ, layer: 3, pos: 81
type: RSZ, layer: 3, pos: 136

Time for candidate selection: 0.12 seconds

### Candidate
type: RSZ, layer: 3, pos: 80

### Relational analysis RSZ of RS_RSZ2_RSZ1_RSZ2_RSZ2_RSZ1_RSZ2_RSZ1

#### Relational analysis RSZ result of RS_RSZ2_RSZ1_RSZ2_RSZ2_RSZ1_RSZ2_RSZ1
Status: Status.UNKNOWN
Output dim: 1, lower bound: -0.0002801, upper bound: 0.0002833
time: 0.58 seconds

### Relational analysis RSZ of RS_RSZ2_RSZ1_RSZ2_RSZ2_RSZ1_RSZ2_RSZ2

#### Relational analysis RSZ result of RS_RSZ2_RSZ1_RSZ2_RSZ2_RSZ1_RSZ2_RSZ2
Status: Status.UNKNOWN
Output dim: 1, lower bound: -0.0002779, upper bound: 0.0002840
time: 0.58 seconds

## BFS RS instance: RS_RSZ2_RSZ1_RSZ2_RSZ2_RSZ2_RSZ1

### Backsubstitution after applying RS history:
0: 0.0009457, 0.0011003, 0.0009457, 0.0011003, -0.0001489, 0.0001495
1: 0.9936197, 0.9940771, 0.9936197, 0.9940771, -0.0003936, 0.0003862
2: -0.0065606, -0.0050208, -0.0065606, -0.0050208, -0.0011443, 0.0010658
3: 0.0038004, 0.0040140, 0.0038004, 0.0040140, -0.0001641, 0.0001693
4: 0.0023852, 0.0036022, 0.0023852, 0.0036022, -0.0009293, 0.0009911
5: 0.0060086, 0.0065111, 0.0060086, 0.0065111, -0.0005024, 0.0005024
6: -0.0013983, -0.0008639, -0.0013983, -0.0008639, -0.0003921, 0.0003648
7: -0.0083221, -0.0079387, -0.0083221, -0.0079387, -0.0003835, 0.0003835
8: 0.0048484, 0.0068715, 0.0048484, 0.0068715, -0.0011607, 0.0012637
9: -0.0036877, -0.0032079, -0.0036877, -0.0032079, -0.0004797, 0.0004797

### Unstable ReLU Count (Linear/Conv2D Layers)
- layer_idx=0, type=LayerType.Linear, total=256, inp1_unstable=5, inp2_unstable=5, delta_unstable=251
- layer_idx=2, type=LayerType.Linear, total=256, inp1_unstable=4, inp2_unstable=4, delta_unstable=255
- layer_idx=4, type=LayerType.Linear, total=256, inp1_unstable=0, inp2_unstable=0, delta_unstable=256
- layer_idx=6, type=LayerType.Linear, total=256, inp1_unstable=1, inp2_unstable=1, delta_unstable=256
- layer_idx=8, type=LayerType.Linear, total=10, inp1_unstable=0, inp2_unstable=0, delta_unstable=10

Time for backsubstitution: 1.19 seconds

### RS candidates at layer 3
type: RSZ, layer: 3, pos: 80
type: RSZ, layer: 3, pos: 81
type: RSZ, layer: 3, pos: 136

Time for candidate selection: 0.12 seconds

### Candidate
type: RSZ, layer: 3, pos: 80

### Relational analysis RSZ of RS_RSZ2_RSZ1_RSZ2_RSZ2_RSZ2_RSZ1_RSZ1

#### Relational analysis RSZ result of RS_RSZ2_RSZ1_RSZ2_RSZ2_RSZ2_RSZ1_RSZ1
Status: Status.UNKNOWN
Output dim: 1, lower bound: -0.0002839, upper bound: 0.0002785
time: 0.55 seconds

### Relational analysis RSZ of RS_RSZ2_RSZ1_RSZ2_RSZ2_RSZ2_RSZ1_RSZ2

#### Relational analysis RSZ result of RS_RSZ2_RSZ1_RSZ2_RSZ2_RSZ2_RSZ1_RSZ2
Status: Status.UNKNOWN
Output dim: 1, lower bound: -0.0002827, upper bound: 0.0002803
time: 0.57 seconds

## BFS RS instance: RS_RSZ2_RSZ1_RSZ2_RSZ2_RSZ2_RSZ2

### Backsubstitution after applying RS history:
0: 0.0009457, 0.0011003, 0.0009457, 0.0011003, -0.0001494, 0.0001489
1: 0.9936197, 0.9940771, 0.9936197, 0.9940771, -0.0003872, 0.0003924
2: -0.0065606, -0.0050208, -0.0065606, -0.0050208, -0.0010773, 0.0011311
3: 0.0038004, 0.0040140, 0.0038004, 0.0040140, -0.0001684, 0.0001648
4: 0.0023852, 0.0036022, 0.0023852, 0.0036022, -0.0009810, 0.0009382
5: 0.0060086, 0.0065111, 0.0060086, 0.0065111, -0.0005024, 0.0005024
6: -0.0013983, -0.0008639, -0.0013983, -0.0008639, -0.0003688, 0.0003875
7: -0.0083221, -0.0079387, -0.0083221, -0.0079387, -0.0003835, 0.0003835
8: 0.0048484, 0.0068715, 0.0048484, 0.0068715, -0.0012465, 0.0011757
9: -0.0036877, -0.0032079, -0.0036877, -0.0032079, -0.0004797, 0.0004797

### Unstable ReLU Count (Linear/Conv2D Layers)
- layer_idx=0, type=LayerType.Linear, total=256, inp1_unstable=5, inp2_unstable=5, delta_unstable=251
- layer_idx=2, type=LayerType.Linear, total=256, inp1_unstable=4, inp2_unstable=4, delta_unstable=255
- layer_idx=4, type=LayerType.Linear, total=256, inp1_unstable=0, inp2_unstable=0, delta_unstable=256
- layer_idx=6, type=LayerType.Linear, total=256, inp1_unstable=1, inp2_unstable=1, delta_unstable=256
- layer_idx=8, type=LayerType.Linear, total=10, inp1_unstable=0, inp2_unstable=0, delta_unstable=10

Time for backsubstitution: 1.21 seconds

### RS candidates at layer 3
type: RSZ, layer: 3, pos: 80
type: RSZ, layer: 3, pos: 81
type: RSZ, layer: 3, pos: 136

Time for candidate selection: 0.12 seconds

### Candidate
type: RSZ, layer: 3, pos: 80

### Relational analysis RSZ of RS_RSZ2_RSZ1_RSZ2_RSZ2_RSZ2_RSZ2_RSZ1

#### Relational analysis RSZ result of RS_RSZ2_RSZ1_RSZ2_RSZ2_RSZ2_RSZ2_RSZ1
Status: Status.UNKNOWN
Output dim: 1, lower bound: -0.0002801, upper bound: 0.0002833
time: 0.59 seconds

### Relational analysis RSZ of RS_RSZ2_RSZ1_RSZ2_RSZ2_RSZ2_RSZ2_RSZ2

#### Relational analysis RSZ result of RS_RSZ2_RSZ1_RSZ2_RSZ2_RSZ2_RSZ2_RSZ2
Status: Status.UNKNOWN
Output dim: 1, lower bound: -0.0002780, upper bound: 0.0002842
time: 0.59 seconds

## BFS RS instance: RS_RSZ2_RSZ2_RSZ1_RSZ1_RSZ1_RSZ1

### Backsubstitution after applying RS history:
0: 0.0009457, 0.0011003, 0.0009457, 0.0011003, -0.0001489, 0.0001496
1: 0.9936197, 0.9940771, 0.9936197, 0.9940771, -0.0003953, 0.0003861
2: -0.0065606, -0.0050208, -0.0065606, -0.0050208, -0.0011612, 0.0010657
3: 0.0038004, 0.0040140, 0.0038004, 0.0040140, -0.0001641, 0.0001705
4: 0.0023852, 0.0036022, 0.0023852, 0.0036022, -0.0009292, 0.0010042
5: 0.0060086, 0.0065111, 0.0060086, 0.0065111, -0.0005024, 0.0005024
6: -0.0013983, -0.0008639, -0.0013983, -0.0008639, -0.0003980, 0.0003648
7: -0.0083221, -0.0079387, -0.0083221, -0.0079387, -0.0003835, 0.0003835
8: 0.0048484, 0.0068715, 0.0048484, 0.0068715, -0.0011605, 0.0012868
9: -0.0036877, -0.0032079, -0.0036877, -0.0032079, -0.0004797, 0.0004797

### Unstable ReLU Count (Linear/Conv2D Layers)
- layer_idx=0, type=LayerType.Linear, total=256, inp1_unstable=5, inp2_unstable=5, delta_unstable=251
- layer_idx=2, type=LayerType.Linear, total=256, inp1_unstable=4, inp2_unstable=4, delta_unstable=255
- layer_idx=4, type=LayerType.Linear, total=256, inp1_unstable=0, inp2_unstable=0, delta_unstable=256
- layer_idx=6, type=LayerType.Linear, total=256, inp1_unstable=1, inp2_unstable=1, delta_unstable=256
- layer_idx=8, type=LayerType.Linear, total=10, inp1_unstable=0, inp2_unstable=0, delta_unstable=10

Time for backsubstitution: 1.20 seconds

### RS candidates at layer 3
type: RSZ, layer: 3, pos: 80
type: RSZ, layer: 3, pos: 81
type: RSZ, layer: 3, pos: 136

Time for candidate selection: 0.12 seconds

### Candidate
type: RSZ, layer: 3, pos: 80

### Relational analysis RSZ of RS_RSZ2_RSZ2_RSZ1_RSZ1_RSZ1_RSZ1_RSZ1

#### Relational analysis RSZ result of RS_RSZ2_RSZ2_RSZ1_RSZ1_RSZ1_RSZ1_RSZ1
Status: Status.UNKNOWN
Output dim: 1, lower bound: -0.0002842, upper bound: 0.0002780
time: 0.57 seconds

### Relational analysis RSZ of RS_RSZ2_RSZ2_RSZ1_RSZ1_RSZ1_RSZ1_RSZ2

#### Relational analysis RSZ result of RS_RSZ2_RSZ2_RSZ1_RSZ1_RSZ1_RSZ1_RSZ2
Status: Status.UNKNOWN
Output dim: 1, lower bound: -0.0002833, upper bound: 0.0002801
time: 0.57 seconds

## BFS RS instance: RS_RSZ2_RSZ2_RSZ1_RSZ1_RSZ1_RSZ2

### Backsubstitution after applying RS history:
0: 0.0009457, 0.0011003, 0.0009457, 0.0011003, -0.0001494, 0.0001490
1: 0.9936197, 0.9940771, 0.9936197, 0.9940771, -0.0003883, 0.0003923
2: -0.0065606, -0.0050208, -0.0065606, -0.0050208, -0.0010881, 0.0011304
3: 0.0038004, 0.0040140, 0.0038004, 0.0040140, -0.0001684, 0.0001656
4: 0.0023852, 0.0036022, 0.0023852, 0.0036022, -0.0009804, 0.0009468
5: 0.0060086, 0.0065111, 0.0060086, 0.0065111, -0.0005024, 0.0005024
6: -0.0013983, -0.0008639, -0.0013983, -0.0008639, -0.0003727, 0.0003873
7: -0.0083221, -0.0079387, -0.0083221, -0.0079387, -0.0003835, 0.0003835
8: 0.0048484, 0.0068715, 0.0048484, 0.0068715, -0.0012456, 0.0011910
9: -0.0036877, -0.0032079, -0.0036877, -0.0032079, -0.0004797, 0.0004797

### Unstable ReLU Count (Linear/Conv2D Layers)
- layer_idx=0, type=LayerType.Linear, total=256, inp1_unstable=5, inp2_unstable=5, delta_unstable=251
- layer_idx=2, type=LayerType.Linear, total=256, inp1_unstable=4, inp2_unstable=4, delta_unstable=255
- layer_idx=4, type=LayerType.Linear, total=256, inp1_unstable=0, inp2_unstable=0, delta_unstable=256
- layer_idx=6, type=LayerType.Linear, total=256, inp1_unstable=1, inp2_unstable=1, delta_unstable=256
- layer_idx=8, type=LayerType.Linear, total=10, inp1_unstable=0, inp2_unstable=0, delta_unstable=10

Time for backsubstitution: 1.23 seconds

### RS candidates at layer 3
type: RSZ, layer: 3, pos: 80
type: RSZ, layer: 3, pos: 81
type: RSZ, layer: 3, pos: 136

Time for candidate selection: 0.12 seconds

### Candidate
type: RSZ, layer: 3, pos: 80

### Relational analysis RSZ of RS_RSZ2_RSZ2_RSZ1_RSZ1_RSZ1_RSZ2_RSZ1

#### Relational analysis RSZ result of RS_RSZ2_RSZ2_RSZ1_RSZ1_RSZ1_RSZ2_RSZ1
Status: Status.UNKNOWN
Output dim: 1, lower bound: -0.0002803, upper bound: 0.0002827
time: 0.57 seconds

### Relational analysis RSZ of RS_RSZ2_RSZ2_RSZ1_RSZ1_RSZ1_RSZ2_RSZ2

#### Relational analysis RSZ result of RS_RSZ2_RSZ2_RSZ1_RSZ1_RSZ1_RSZ2_RSZ2
Status: Status.UNKNOWN
Output dim: 1, lower bound: -0.0002785, upper bound: 0.0002839
time: 0.57 seconds

## BFS RS instance: RS_RSZ2_RSZ2_RSZ1_RSZ1_RSZ2_RSZ1

### Backsubstitution after applying RS history:
0: 0.0009457, 0.0011003, 0.0009457, 0.0011003, -0.0001489, 0.0001496
1: 0.9936197, 0.9940771, 0.9936197, 0.9940771, -0.0003953, 0.0003863
2: -0.0065606, -0.0050208, -0.0065606, -0.0050208, -0.0011611, 0.0010669
3: 0.0038004, 0.0040140, 0.0038004, 0.0040140, -0.0001642, 0.0001705
4: 0.0023852, 0.0036022, 0.0023852, 0.0036022, -0.0009302, 0.0010041
5: 0.0060086, 0.0065111, 0.0060086, 0.0065111, -0.0005024, 0.0005024
6: -0.0013983, -0.0008639, -0.0013983, -0.0008639, -0.0003980, 0.0003652
7: -0.0083221, -0.0079387, -0.0083221, -0.0079387, -0.0003835, 0.0003835
8: 0.0048484, 0.0068715, 0.0048484, 0.0068715, -0.0011622, 0.0012867
9: -0.0036877, -0.0032079, -0.0036877, -0.0032079, -0.0004797, 0.0004797

### Unstable ReLU Count (Linear/Conv2D Layers)
- layer_idx=0, type=LayerType.Linear, total=256, inp1_unstable=5, inp2_unstable=5, delta_unstable=251
- layer_idx=2, type=LayerType.Linear, total=256, inp1_unstable=4, inp2_unstable=4, delta_unstable=255
- layer_idx=4, type=LayerType.Linear, total=256, inp1_unstable=0, inp2_unstable=0, delta_unstable=256
- layer_idx=6, type=LayerType.Linear, total=256, inp1_unstable=1, inp2_unstable=1, delta_unstable=256
- layer_idx=8, type=LayerType.Linear, total=10, inp1_unstable=0, inp2_unstable=0, delta_unstable=10

Time for backsubstitution: 1.22 seconds

### RS candidates at layer 3
type: RSZ, layer: 3, pos: 80
type: RSZ, layer: 3, pos: 81
type: RSZ, layer: 3, pos: 136

Time for candidate selection: 0.12 seconds

### Candidate
type: RSZ, layer: 3, pos: 80

### Relational analysis RSZ of RS_RSZ2_RSZ2_RSZ1_RSZ1_RSZ2_RSZ1_RSZ1

#### Relational analysis RSZ result of RS_RSZ2_RSZ2_RSZ1_RSZ1_RSZ2_RSZ1_RSZ1
Status: Status.UNKNOWN
Output dim: 1, lower bound: -0.0002840, upper bound: 0.0002779
time: 0.61 seconds

### Relational analysis RSZ of RS_RSZ2_RSZ2_RSZ1_RSZ1_RSZ2_RSZ1_RSZ2

#### Relational analysis RSZ result of RS_RSZ2_RSZ2_RSZ1_RSZ1_RSZ2_RSZ1_RSZ2
Status: Status.UNKNOWN
Output dim: 1, lower bound: -0.0002833, upper bound: 0.0002801
time: 0.56 seconds

## BFS RS instance: RS_RSZ2_RSZ2_RSZ1_RSZ1_RSZ2_RSZ2

### Backsubstitution after applying RS history:
0: 0.0009457, 0.0011003, 0.0009457, 0.0011003, -0.0001493, 0.0001490
1: 0.9936197, 0.9940771, 0.9936197, 0.9940771, -0.0003881, 0.0003923
2: -0.0065606, -0.0050208, -0.0065606, -0.0050208, -0.0010852, 0.0011300
3: 0.0038004, 0.0040140, 0.0038004, 0.0040140, -0.0001683, 0.0001654
4: 0.0023852, 0.0036022, 0.0023852, 0.0036022, -0.0009801, 0.0009444
5: 0.0060086, 0.0065111, 0.0060086, 0.0065111, -0.0005024, 0.0005024
6: -0.0013983, -0.0008639, -0.0013983, -0.0008639, -0.0003717, 0.0003871
7: -0.0083221, -0.0079387, -0.0083221, -0.0079387, -0.0003835, 0.0003835
8: 0.0048484, 0.0068715, 0.0048484, 0.0068715, -0.0012451, 0.0011871
9: -0.0036877, -0.0032079, -0.0036877, -0.0032079, -0.0004797, 0.0004797

### Unstable ReLU Count (Linear/Conv2D Layers)
- layer_idx=0, type=LayerType.Linear, total=256, inp1_unstable=5, inp2_unstable=5, delta_unstable=251
- layer_idx=2, type=LayerType.Linear, total=256, inp1_unstable=4, inp2_unstable=4, delta_unstable=255
- layer_idx=4, type=LayerType.Linear, total=256, inp1_unstable=0, inp2_unstable=0, delta_unstable=256
- layer_idx=6, type=LayerType.Linear, total=256, inp1_unstable=1, inp2_unstable=1, delta_unstable=256
- layer_idx=8, type=LayerType.Linear, total=10, inp1_unstable=0, inp2_unstable=0, delta_unstable=10

Time for backsubstitution: 1.22 seconds

### RS candidates at layer 3
type: RSZ, layer: 3, pos: 80
type: RSZ, layer: 3, pos: 81
type: RSZ, layer: 3, pos: 136

Time for candidate selection: 0.12 seconds

### Candidate
type: RSZ, layer: 3, pos: 80

### Relational analysis RSZ of RS_RSZ2_RSZ2_RSZ1_RSZ1_RSZ2_RSZ2_RSZ1

#### Relational analysis RSZ result of RS_RSZ2_RSZ2_RSZ1_RSZ1_RSZ2_RSZ2_RSZ1
Status: Status.UNKNOWN
Output dim: 1, lower bound: -0.0002802, upper bound: 0.0002827
time: 0.58 seconds

### Relational analysis RSZ of RS_RSZ2_RSZ2_RSZ1_RSZ1_RSZ2_RSZ2_RSZ2

#### Relational analysis RSZ result of RS_RSZ2_RSZ2_RSZ1_RSZ1_RSZ2_RSZ2_RSZ2
Status: Status.UNKNOWN
Output dim: 1, lower bound: -0.0002785, upper bound: 0.0002839
time: 0.57 seconds

## BFS RS instance: RS_RSZ2_RSZ2_RSZ1_RSZ2_RSZ1_RSZ1

### Backsubstitution after applying RS history:
0: 0.0009457, 0.0011003, 0.0009457, 0.0011003, -0.0001489, 0.0001496
1: 0.9936197, 0.9940771, 0.9936197, 0.9940771, -0.0003950, 0.0003862
2: -0.0065606, -0.0050208, -0.0065606, -0.0050208, -0.0011575, 0.0010660
3: 0.0038004, 0.0040140, 0.0038004, 0.0040140, -0.0001641, 0.0001702
4: 0.0023852, 0.0036022, 0.0023852, 0.0036022, -0.0009295, 0.0010013
5: 0.0060086, 0.0065111, 0.0060086, 0.0065111, -0.0005024, 0.0005024
6: -0.0013983, -0.0008639, -0.0013983, -0.0008639, -0.0003967, 0.0003649
7: -0.0083221, -0.0079387, -0.0083221, -0.0079387, -0.0003835, 0.0003835
8: 0.0048484, 0.0068715, 0.0048484, 0.0068715, -0.0011609, 0.0012820
9: -0.0036877, -0.0032079, -0.0036877, -0.0032079, -0.0004797, 0.0004797

### Unstable ReLU Count (Linear/Conv2D Layers)
- layer_idx=0, type=LayerType.Linear, total=256, inp1_unstable=5, inp2_unstable=5, delta_unstable=251
- layer_idx=2, type=LayerType.Linear, total=256, inp1_unstable=4, inp2_unstable=4, delta_unstable=255
- layer_idx=4, type=LayerType.Linear, total=256, inp1_unstable=0, inp2_unstable=0, delta_unstable=256
- layer_idx=6, type=LayerType.Linear, total=256, inp1_unstable=1, inp2_unstable=1, delta_unstable=256
- layer_idx=8, type=LayerType.Linear, total=10, inp1_unstable=0, inp2_unstable=0, delta_unstable=10

Time for backsubstitution: 1.24 seconds

### RS candidates at layer 3
type: RSZ, layer: 3, pos: 80
type: RSZ, layer: 3, pos: 81
type: RSZ, layer: 3, pos: 136

Time for candidate selection: 0.12 seconds

### Candidate
type: RSZ, layer: 3, pos: 80

### Relational analysis RSZ of RS_RSZ2_RSZ2_RSZ1_RSZ2_RSZ1_RSZ1_RSZ1

#### Relational analysis RSZ result of RS_RSZ2_RSZ2_RSZ1_RSZ2_RSZ1_RSZ1_RSZ1
Status: Status.UNKNOWN
Output dim: 1, lower bound: -0.0002832, upper bound: 0.0002785
time: 0.60 seconds

### Relational analysis RSZ of RS_RSZ2_RSZ2_RSZ1_RSZ2_RSZ1_RSZ1_RSZ2

#### Relational analysis RSZ result of RS_RSZ2_RSZ2_RSZ1_RSZ2_RSZ1_RSZ1_RSZ2
Status: Status.UNKNOWN
Output dim: 1, lower bound: -0.0002826, upper bound: 0.0002809
time: 0.56 seconds

## BFS RS instance: RS_RSZ2_RSZ2_RSZ1_RSZ2_RSZ1_RSZ2

### Backsubstitution after applying RS history:
0: 0.0009457, 0.0011003, 0.0009457, 0.0011003, -0.0001494, 0.0001490
1: 0.9936197, 0.9940771, 0.9936197, 0.9940771, -0.0003885, 0.0003924
2: -0.0065606, -0.0050208, -0.0065606, -0.0050208, -0.0010898, 0.0011319
3: 0.0038004, 0.0040140, 0.0038004, 0.0040140, -0.0001685, 0.0001657
4: 0.0023852, 0.0036022, 0.0023852, 0.0036022, -0.0009816, 0.0009481
5: 0.0060086, 0.0065111, 0.0060086, 0.0065111, -0.0005024, 0.0005024
6: -0.0013983, -0.0008639, -0.0013983, -0.0008639, -0.0003733, 0.0003878
7: -0.0083221, -0.0079387, -0.0083221, -0.0079387, -0.0003835, 0.0003835
8: 0.0048484, 0.0068715, 0.0048484, 0.0068715, -0.0012476, 0.0011932
9: -0.0036877, -0.0032079, -0.0036877, -0.0032079, -0.0004797, 0.0004797

### Unstable ReLU Count (Linear/Conv2D Layers)
- layer_idx=0, type=LayerType.Linear, total=256, inp1_unstable=5, inp2_unstable=5, delta_unstable=251
- layer_idx=2, type=LayerType.Linear, total=256, inp1_unstable=4, inp2_unstable=4, delta_unstable=255
- layer_idx=4, type=LayerType.Linear, total=256, inp1_unstable=0, inp2_unstable=0, delta_unstable=256
- layer_idx=6, type=LayerType.Linear, total=256, inp1_unstable=1, inp2_unstable=1, delta_unstable=256
- layer_idx=8, type=LayerType.Linear, total=10, inp1_unstable=0, inp2_unstable=0, delta_unstable=10

Time for backsubstitution: 1.21 seconds

### RS candidates at layer 3
type: RSZ, layer: 3, pos: 80
type: RSZ, layer: 3, pos: 81
type: RSZ, layer: 3, pos: 136

Time for candidate selection: 0.13 seconds

### Candidate
type: RSZ, layer: 3, pos: 80

### Relational analysis RSZ of RS_RSZ2_RSZ2_RSZ1_RSZ2_RSZ1_RSZ2_RSZ1

#### Relational analysis RSZ result of RS_RSZ2_RSZ2_RSZ1_RSZ2_RSZ1_RSZ2_RSZ1
Status: Status.UNKNOWN
Output dim: 1, lower bound: -0.0002797, upper bound: 0.0002835
time: 0.58 seconds

### Relational analysis RSZ of RS_RSZ2_RSZ2_RSZ1_RSZ2_RSZ1_RSZ2_RSZ2

#### Relational analysis RSZ result of RS_RSZ2_RSZ2_RSZ1_RSZ2_RSZ1_RSZ2_RSZ2
Status: Status.UNKNOWN
Output dim: 1, lower bound: -0.0002779, upper bound: 0.0002849
time: 0.58 seconds

## BFS RS instance: RS_RSZ2_RSZ2_RSZ1_RSZ2_RSZ2_RSZ1

### Backsubstitution after applying RS history:
0: 0.0009457, 0.0011003, 0.0009457, 0.0011003, -0.0001489, 0.0001496
1: 0.9936197, 0.9940771, 0.9936197, 0.9940771, -0.0003950, 0.0003865
2: -0.0065606, -0.0050208, -0.0065606, -0.0050208, -0.0011576, 0.0010689
3: 0.0038004, 0.0040140, 0.0038004, 0.0040140, -0.0001643, 0.0001702
4: 0.0023852, 0.0036022, 0.0023852, 0.0036022, -0.0009318, 0.0010014
5: 0.0060086, 0.0065111, 0.0060086, 0.0065111, -0.0005024, 0.0005024
6: -0.0013983, -0.0008639, -0.0013983, -0.0008639, -0.0003967, 0.0003659
7: -0.0083221, -0.0079387, -0.0083221, -0.0079387, -0.0003835, 0.0003835
8: 0.0048484, 0.0068715, 0.0048484, 0.0068715, -0.0011648, 0.0012821
9: -0.0036877, -0.0032079, -0.0036877, -0.0032079, -0.0004797, 0.0004797

### Unstable ReLU Count (Linear/Conv2D Layers)
- layer_idx=0, type=LayerType.Linear, total=256, inp1_unstable=5, inp2_unstable=5, delta_unstable=251
- layer_idx=2, type=LayerType.Linear, total=256, inp1_unstable=4, inp2_unstable=4, delta_unstable=255
- layer_idx=4, type=LayerType.Linear, total=256, inp1_unstable=0, inp2_unstable=0, delta_unstable=256
- layer_idx=6, type=LayerType.Linear, total=256, inp1_unstable=1, inp2_unstable=1, delta_unstable=256
- layer_idx=8, type=LayerType.Linear, total=10, inp1_unstable=0, inp2_unstable=0, delta_unstable=10

Time for backsubstitution: 1.21 seconds

### RS candidates at layer 3
type: RSZ, layer: 3, pos: 80
type: RSZ, layer: 3, pos: 81
type: RSZ, layer: 3, pos: 136

Time for candidate selection: 0.13 seconds

### Candidate
type: RSZ, layer: 3, pos: 80

### Relational analysis RSZ of RS_RSZ2_RSZ2_RSZ1_RSZ2_RSZ2_RSZ1_RSZ1

#### Relational analysis RSZ result of RS_RSZ2_RSZ2_RSZ1_RSZ2_RSZ2_RSZ1_RSZ1
Status: Status.UNKNOWN
Output dim: 1, lower bound: -0.0002832, upper bound: 0.0002785
time: 0.60 seconds

### Relational analysis RSZ of RS_RSZ2_RSZ2_RSZ1_RSZ2_RSZ2_RSZ1_RSZ2

#### Relational analysis RSZ result of RS_RSZ2_RSZ2_RSZ1_RSZ2_RSZ2_RSZ1_RSZ2
Status: Status.UNKNOWN
Output dim: 1, lower bound: -0.0002827, upper bound: 0.0002810
time: 0.57 seconds

## BFS RS instance: RS_RSZ2_RSZ2_RSZ1_RSZ2_RSZ2_RSZ2

### Backsubstitution after applying RS history:
0: 0.0009457, 0.0011003, 0.0009457, 0.0011003, -0.0001494, 0.0001490
1: 0.9936197, 0.9940771, 0.9936197, 0.9940771, -0.0003884, 0.0003924
2: -0.0065606, -0.0050208, -0.0065606, -0.0050208, -0.0010882, 0.0011318
3: 0.0038004, 0.0040140, 0.0038004, 0.0040140, -0.0001685, 0.0001656
4: 0.0023852, 0.0036022, 0.0023852, 0.0036022, -0.0009815, 0.0009468
5: 0.0060086, 0.0065111, 0.0060086, 0.0065111, -0.0005024, 0.0005024
6: -0.0013983, -0.0008639, -0.0013983, -0.0008639, -0.0003727, 0.0003878
7: -0.0083221, -0.0079387, -0.0083221, -0.0079387, -0.0003835, 0.0003835
8: 0.0048484, 0.0068715, 0.0048484, 0.0068715, -0.0012474, 0.0011911
9: -0.0036877, -0.0032079, -0.0036877, -0.0032079, -0.0004797, 0.0004797

### Unstable ReLU Count (Linear/Conv2D Layers)
- layer_idx=0, type=LayerType.Linear, total=256, inp1_unstable=5, inp2_unstable=5, delta_unstable=251
- layer_idx=2, type=LayerType.Linear, total=256, inp1_unstable=4, inp2_unstable=4, delta_unstable=255
- layer_idx=4, type=LayerType.Linear, total=256, inp1_unstable=0, inp2_unstable=0, delta_unstable=256
- layer_idx=6, type=LayerType.Linear, total=256, inp1_unstable=1, inp2_unstable=1, delta_unstable=256
- layer_idx=8, type=LayerType.Linear, total=10, inp1_unstable=0, inp2_unstable=0, delta_unstable=10

Time for backsubstitution: 1.21 seconds

### RS candidates at layer 3
type: RSZ, layer: 3, pos: 80
type: RSZ, layer: 3, pos: 81
type: RSZ, layer: 3, pos: 136

Time for candidate selection: 0.13 seconds

### Candidate
type: RSZ, layer: 3, pos: 80

### Relational analysis RSZ of RS_RSZ2_RSZ2_RSZ1_RSZ2_RSZ2_RSZ2_RSZ1

#### Relational analysis RSZ result of RS_RSZ2_RSZ2_RSZ1_RSZ2_RSZ2_RSZ2_RSZ1
Status: Status.UNKNOWN
Output dim: 1, lower bound: -0.0002796, upper bound: 0.0002835
time: 0.59 seconds

### Relational analysis RSZ of RS_RSZ2_RSZ2_RSZ1_RSZ2_RSZ2_RSZ2_RSZ2

#### Relational analysis RSZ result of RS_RSZ2_RSZ2_RSZ1_RSZ2_RSZ2_RSZ2_RSZ2
Status: Status.UNKNOWN
Output dim: 1, lower bound: -0.0002779, upper bound: 0.0002850
time: 0.57 seconds

## BFS RS instance: RS_RSZ2_RSZ2_RSZ2_RSZ1_RSZ1_RSZ1

### Backsubstitution after applying RS history:
0: 0.0009457, 0.0011003, 0.0009457, 0.0011003, -0.0001489, 0.0001496
1: 0.9936197, 0.9940771, 0.9936197, 0.9940771, -0.0003955, 0.0003861
2: -0.0065606, -0.0050208, -0.0065606, -0.0050208, -0.0011627, 0.0010653
3: 0.0038004, 0.0040140, 0.0038004, 0.0040140, -0.0001640, 0.0001706
4: 0.0023852, 0.0036022, 0.0023852, 0.0036022, -0.0009290, 0.0010054
5: 0.0060086, 0.0065111, 0.0060086, 0.0065111, -0.0005024, 0.0005024
6: -0.0013983, -0.0008639, -0.0013983, -0.0008639, -0.0003985, 0.0003647
7: -0.0083221, -0.0079387, -0.0083221, -0.0079387, -0.0003835, 0.0003835
8: 0.0048484, 0.0068715, 0.0048484, 0.0068715, -0.0011601, 0.0012889
9: -0.0036877, -0.0032079, -0.0036877, -0.0032079, -0.0004797, 0.0004797

### Unstable ReLU Count (Linear/Conv2D Layers)
- layer_idx=0, type=LayerType.Linear, total=256, inp1_unstable=5, inp2_unstable=5, delta_unstable=251
- layer_idx=2, type=LayerType.Linear, total=256, inp1_unstable=4, inp2_unstable=4, delta_unstable=255
- layer_idx=4, type=LayerType.Linear, total=256, inp1_unstable=0, inp2_unstable=0, delta_unstable=256
- layer_idx=6, type=LayerType.Linear, total=256, inp1_unstable=1, inp2_unstable=1, delta_unstable=256
- layer_idx=8, type=LayerType.Linear, total=10, inp1_unstable=0, inp2_unstable=0, delta_unstable=10

Time for backsubstitution: 1.21 seconds

### RS candidates at layer 3
type: RSZ, layer: 3, pos: 80
type: RSZ, layer: 3, pos: 81
type: RSZ, layer: 3, pos: 136

Time for candidate selection: 0.12 seconds

### Candidate
type: RSZ, layer: 3, pos: 80

### Relational analysis RSZ of RS_RSZ2_RSZ2_RSZ2_RSZ1_RSZ1_RSZ1_RSZ1

#### Relational analysis RSZ result of RS_RSZ2_RSZ2_RSZ2_RSZ1_RSZ1_RSZ1_RSZ1
Status: Status.UNKNOWN
Output dim: 1, lower bound: -0.0002850, upper bound: 0.0002779
time: 0.56 seconds

### Relational analysis RSZ of RS_RSZ2_RSZ2_RSZ2_RSZ1_RSZ1_RSZ1_RSZ2

#### Relational analysis RSZ result of RS_RSZ2_RSZ2_RSZ2_RSZ1_RSZ1_RSZ1_RSZ2
Status: Status.UNKNOWN
Output dim: 1, lower bound: -0.0002835, upper bound: 0.0002795
time: 0.57 seconds

## BFS RS instance: RS_RSZ2_RSZ2_RSZ2_RSZ1_RSZ1_RSZ2

### Backsubstitution after applying RS history:
0: 0.0009457, 0.0011003, 0.0009457, 0.0011003, -0.0001493, 0.0001491
1: 0.9936197, 0.9940771, 0.9936197, 0.9940771, -0.0003887, 0.0003919
2: -0.0065606, -0.0050208, -0.0065606, -0.0050208, -0.0010918, 0.0011266
3: 0.0038004, 0.0040140, 0.0038004, 0.0040140, -0.0001681, 0.0001659
4: 0.0023852, 0.0036022, 0.0023852, 0.0036022, -0.0009774, 0.0009497
5: 0.0060086, 0.0065111, 0.0060086, 0.0065111, -0.0005024, 0.0005024
6: -0.0013983, -0.0008639, -0.0013983, -0.0008639, -0.0003740, 0.0003860
7: -0.0083221, -0.0079387, -0.0083221, -0.0079387, -0.0003835, 0.0003835
8: 0.0048484, 0.0068715, 0.0048484, 0.0068715, -0.0012406, 0.0011959
9: -0.0036877, -0.0032079, -0.0036877, -0.0032079, -0.0004797, 0.0004797

### Unstable ReLU Count (Linear/Conv2D Layers)
- layer_idx=0, type=LayerType.Linear, total=256, inp1_unstable=5, inp2_unstable=5, delta_unstable=251
- layer_idx=2, type=LayerType.Linear, total=256, inp1_unstable=4, inp2_unstable=4, delta_unstable=255
- layer_idx=4, type=LayerType.Linear, total=256, inp1_unstable=0, inp2_unstable=0, delta_unstable=256
- layer_idx=6, type=LayerType.Linear, total=256, inp1_unstable=1, inp2_unstable=1, delta_unstable=256
- layer_idx=8, type=LayerType.Linear, total=10, inp1_unstable=0, inp2_unstable=0, delta_unstable=10

Time for backsubstitution: 1.24 seconds

### RS candidates at layer 3
type: RSZ, layer: 3, pos: 80
type: RSZ, layer: 3, pos: 81
type: RSZ, layer: 3, pos: 136

Time for candidate selection: 0.13 seconds

### Candidate
type: RSZ, layer: 3, pos: 80

### Relational analysis RSZ of RS_RSZ2_RSZ2_RSZ2_RSZ1_RSZ1_RSZ2_RSZ1

#### Relational analysis RSZ result of RS_RSZ2_RSZ2_RSZ2_RSZ1_RSZ1_RSZ2_RSZ1
Status: Status.UNKNOWN
Output dim: 1, lower bound: -0.0002810, upper bound: 0.0002827
time: 0.58 seconds

### Relational analysis RSZ of RS_RSZ2_RSZ2_RSZ2_RSZ1_RSZ1_RSZ2_RSZ2

#### Relational analysis RSZ result of RS_RSZ2_RSZ2_RSZ2_RSZ1_RSZ1_RSZ2_RSZ2
Status: Status.UNKNOWN
Output dim: 1, lower bound: -0.0002785, upper bound: 0.0002831
time: 0.60 seconds

## BFS RS instance: RS_RSZ2_RSZ2_RSZ2_RSZ1_RSZ2_RSZ1

### Backsubstitution after applying RS history:
0: 0.0009457, 0.0011003, 0.0009457, 0.0011003, -0.0001489, 0.0001496
1: 0.9936197, 0.9940771, 0.9936197, 0.9940771, -0.0003955, 0.0003863
2: -0.0065606, -0.0050208, -0.0065606, -0.0050208, -0.0011628, 0.0010669
3: 0.0038004, 0.0040140, 0.0038004, 0.0040140, -0.0001642, 0.0001706
4: 0.0023852, 0.0036022, 0.0023852, 0.0036022, -0.0009302, 0.0010055
5: 0.0060086, 0.0065111, 0.0060086, 0.0065111, -0.0005024, 0.0005024
6: -0.0013983, -0.0008639, -0.0013983, -0.0008639, -0.0003986, 0.0003652
7: -0.0083221, -0.0079387, -0.0083221, -0.0079387, -0.0003835, 0.0003835
8: 0.0048484, 0.0068715, 0.0048484, 0.0068715, -0.0011621, 0.0012890
9: -0.0036877, -0.0032079, -0.0036877, -0.0032079, -0.0004797, 0.0004797

### Unstable ReLU Count (Linear/Conv2D Layers)
- layer_idx=0, type=LayerType.Linear, total=256, inp1_unstable=5, inp2_unstable=5, delta_unstable=251
- layer_idx=2, type=LayerType.Linear, total=256, inp1_unstable=4, inp2_unstable=4, delta_unstable=255
- layer_idx=4, type=LayerType.Linear, total=256, inp1_unstable=0, inp2_unstable=0, delta_unstable=256
- layer_idx=6, type=LayerType.Linear, total=256, inp1_unstable=1, inp2_unstable=1, delta_unstable=256
- layer_idx=8, type=LayerType.Linear, total=10, inp1_unstable=0, inp2_unstable=0, delta_unstable=10

Time for backsubstitution: 1.24 seconds

### RS candidates at layer 3
type: RSZ, layer: 3, pos: 80
type: RSZ, layer: 3, pos: 81
type: RSZ, layer: 3, pos: 136

Time for candidate selection: 0.13 seconds

### Candidate
type: RSZ, layer: 3, pos: 80

### Relational analysis RSZ of RS_RSZ2_RSZ2_RSZ2_RSZ1_RSZ2_RSZ1_RSZ1

#### Relational analysis RSZ result of RS_RSZ2_RSZ2_RSZ2_RSZ1_RSZ2_RSZ1_RSZ1
Status: Status.UNKNOWN
Output dim: 1, lower bound: -0.0002850, upper bound: 0.0002779
time: 0.56 seconds

### Relational analysis RSZ of RS_RSZ2_RSZ2_RSZ2_RSZ1_RSZ2_RSZ1_RSZ2

#### Relational analysis RSZ result of RS_RSZ2_RSZ2_RSZ2_RSZ1_RSZ2_RSZ1_RSZ2
Status: Status.UNKNOWN
Output dim: 1, lower bound: -0.0002835, upper bound: 0.0002796
time: 0.56 seconds

## BFS RS instance: RS_RSZ2_RSZ2_RSZ2_RSZ1_RSZ2_RSZ2

### Backsubstitution after applying RS history:
0: 0.0009457, 0.0011003, 0.0009457, 0.0011003, -0.0001493, 0.0001490
1: 0.9936197, 0.9940771, 0.9936197, 0.9940771, -0.0003884, 0.0003919
2: -0.0065606, -0.0050208, -0.0065606, -0.0050208, -0.0010889, 0.0011265
3: 0.0038004, 0.0040140, 0.0038004, 0.0040140, -0.0001681, 0.0001657
4: 0.0023852, 0.0036022, 0.0023852, 0.0036022, -0.0009773, 0.0009474
5: 0.0060086, 0.0065111, 0.0060086, 0.0065111, -0.0005024, 0.0005024
6: -0.0013983, -0.0008639, -0.0013983, -0.0008639, -0.0003729, 0.0003859
7: -0.0083221, -0.0079387, -0.0083221, -0.0079387, -0.0003835, 0.0003835
8: 0.0048484, 0.0068715, 0.0048484, 0.0068715, -0.0012405, 0.0011920
9: -0.0036877, -0.0032079, -0.0036877, -0.0032079, -0.0004797, 0.0004797

### Unstable ReLU Count (Linear/Conv2D Layers)
- layer_idx=0, type=LayerType.Linear, total=256, inp1_unstable=5, inp2_unstable=5, delta_unstable=251
- layer_idx=2, type=LayerType.Linear, total=256, inp1_unstable=4, inp2_unstable=4, delta_unstable=255
- layer_idx=4, type=LayerType.Linear, total=256, inp1_unstable=0, inp2_unstable=0, delta_unstable=256
- layer_idx=6, type=LayerType.Linear, total=256, inp1_unstable=1, inp2_unstable=1, delta_unstable=256
- layer_idx=8, type=LayerType.Linear, total=10, inp1_unstable=0, inp2_unstable=0, delta_unstable=10

Time for backsubstitution: 1.21 seconds

### RS candidates at layer 3
type: RSZ, layer: 3, pos: 80
type: RSZ, layer: 3, pos: 81
type: RSZ, layer: 3, pos: 136

Time for candidate selection: 0.12 seconds

### Candidate
type: RSZ, layer: 3, pos: 80

### Relational analysis RSZ of RS_RSZ2_RSZ2_RSZ2_RSZ1_RSZ2_RSZ2_RSZ1

#### Relational analysis RSZ result of RS_RSZ2_RSZ2_RSZ2_RSZ1_RSZ2_RSZ2_RSZ1
Status: Status.UNKNOWN
Output dim: 1, lower bound: -0.0002810, upper bound: 0.0002826
time: 0.59 seconds

### Relational analysis RSZ of RS_RSZ2_RSZ2_RSZ2_RSZ1_RSZ2_RSZ2_RSZ2

#### Relational analysis RSZ result of RS_RSZ2_RSZ2_RSZ2_RSZ1_RSZ2_RSZ2_RSZ2
Status: Status.UNKNOWN
Output dim: 1, lower bound: -0.0002785, upper bound: 0.0002831
time: 0.60 seconds

## BFS RS instance: RS_RSZ2_RSZ2_RSZ2_RSZ2_RSZ1_RSZ1

### Backsubstitution after applying RS history:
0: 0.0009457, 0.0011003, 0.0009457, 0.0011003, -0.0001488, 0.0001496
1: 0.9936197, 0.9940771, 0.9936197, 0.9940771, -0.0003953, 0.0003858
2: -0.0065606, -0.0050208, -0.0065606, -0.0050208, -0.0011609, 0.0010623
3: 0.0038004, 0.0040140, 0.0038004, 0.0040140, -0.0001638, 0.0001704
4: 0.0023852, 0.0036022, 0.0023852, 0.0036022, -0.0009266, 0.0010040
5: 0.0060086, 0.0065111, 0.0060086, 0.0065111, -0.0005024, 0.0005024
6: -0.0013983, -0.0008639, -0.0013983, -0.0008639, -0.0003979, 0.0003636
7: -0.0083221, -0.0079387, -0.0083221, -0.0079387, -0.0003835, 0.0003835
8: 0.0048484, 0.0068715, 0.0048484, 0.0068715, -0.0011561, 0.0012865
9: -0.0036877, -0.0032079, -0.0036877, -0.0032079, -0.0004797, 0.0004797

### Unstable ReLU Count (Linear/Conv2D Layers)
- layer_idx=0, type=LayerType.Linear, total=256, inp1_unstable=5, inp2_unstable=5, delta_unstable=251
- layer_idx=2, type=LayerType.Linear, total=256, inp1_unstable=4, inp2_unstable=4, delta_unstable=255
- layer_idx=4, type=LayerType.Linear, total=256, inp1_unstable=0, inp2_unstable=0, delta_unstable=256
- layer_idx=6, type=LayerType.Linear, total=256, inp1_unstable=1, inp2_unstable=1, delta_unstable=256
- layer_idx=8, type=LayerType.Linear, total=10, inp1_unstable=0, inp2_unstable=0, delta_unstable=10

Time for backsubstitution: 1.23 seconds

### RS candidates at layer 3
type: RSZ, layer: 3, pos: 80
type: RSZ, layer: 3, pos: 81
type: RSZ, layer: 3, pos: 136

Time for candidate selection: 0.13 seconds

### Candidate
type: RSZ, layer: 3, pos: 80

### Relational analysis RSZ of RS_RSZ2_RSZ2_RSZ2_RSZ2_RSZ1_RSZ1_RSZ1

#### Relational analysis RSZ result of RS_RSZ2_RSZ2_RSZ2_RSZ2_RSZ1_RSZ1_RSZ1
Status: Status.UNKNOWN
Output dim: 1, lower bound: -0.0002840, upper bound: 0.0002785
time: 0.61 seconds

### Relational analysis RSZ of RS_RSZ2_RSZ2_RSZ2_RSZ2_RSZ1_RSZ1_RSZ2

#### Relational analysis RSZ result of RS_RSZ2_RSZ2_RSZ2_RSZ2_RSZ1_RSZ1_RSZ2
Status: Status.UNKNOWN
Output dim: 1, lower bound: -0.0002827, upper bound: 0.0002802
time: 0.66 seconds

## BFS RS instance: RS_RSZ2_RSZ2_RSZ2_RSZ2_RSZ1_RSZ2

### Backsubstitution after applying RS history:
0: 0.0009457, 0.0011003, 0.0009457, 0.0011003, -0.0001493, 0.0001490
1: 0.9936197, 0.9940771, 0.9936197, 0.9940771, -0.0003885, 0.0003923
2: -0.0065606, -0.0050208, -0.0065606, -0.0050208, -0.0010898, 0.0011302
3: 0.0038004, 0.0040140, 0.0038004, 0.0040140, -0.0001683, 0.0001657
4: 0.0023852, 0.0036022, 0.0023852, 0.0036022, -0.0009802, 0.0009481
5: 0.0060086, 0.0065111, 0.0060086, 0.0065111, -0.0005024, 0.0005024
6: -0.0013983, -0.0008639, -0.0013983, -0.0008639, -0.0003733, 0.0003872
7: -0.0083221, -0.0079387, -0.0083221, -0.0079387, -0.0003835, 0.0003835
8: 0.0048484, 0.0068715, 0.0048484, 0.0068715, -0.0012453, 0.0011932
9: -0.0036877, -0.0032079, -0.0036877, -0.0032079, -0.0004797, 0.0004797

### Unstable ReLU Count (Linear/Conv2D Layers)
- layer_idx=0, type=LayerType.Linear, total=256, inp1_unstable=5, inp2_unstable=5, delta_unstable=251
- layer_idx=2, type=LayerType.Linear, total=256, inp1_unstable=4, inp2_unstable=4, delta_unstable=255
- layer_idx=4, type=LayerType.Linear, total=256, inp1_unstable=0, inp2_unstable=0, delta_unstable=256
- layer_idx=6, type=LayerType.Linear, total=256, inp1_unstable=1, inp2_unstable=1, delta_unstable=256
- layer_idx=8, type=LayerType.Linear, total=10, inp1_unstable=0, inp2_unstable=0, delta_unstable=10

Time for backsubstitution: 1.23 seconds

### RS candidates at layer 3
type: RSZ, layer: 3, pos: 80
type: RSZ, layer: 3, pos: 81
type: RSZ, layer: 3, pos: 136

Time for candidate selection: 0.12 seconds

### Candidate
type: RSZ, layer: 3, pos: 80

### Relational analysis RSZ of RS_RSZ2_RSZ2_RSZ2_RSZ2_RSZ1_RSZ2_RSZ1

#### Relational analysis RSZ result of RS_RSZ2_RSZ2_RSZ2_RSZ2_RSZ1_RSZ2_RSZ1
Status: Status.UNKNOWN
Output dim: 1, lower bound: -0.0002804, upper bound: 0.0002833
time: 0.60 seconds

### Relational analysis RSZ of RS_RSZ2_RSZ2_RSZ2_RSZ2_RSZ1_RSZ2_RSZ2

#### Relational analysis RSZ result of RS_RSZ2_RSZ2_RSZ2_RSZ2_RSZ1_RSZ2_RSZ2
Status: Status.UNKNOWN
Output dim: 1, lower bound: -0.0002780, upper bound: 0.0002840
time: 0.58 seconds

## BFS RS instance: RS_RSZ2_RSZ2_RSZ2_RSZ2_RSZ2_RSZ1

### Backsubstitution after applying RS history:
0: 0.0009457, 0.0011003, 0.0009457, 0.0011003, -0.0001489, 0.0001496
1: 0.9936197, 0.9940771, 0.9936197, 0.9940771, -0.0003953, 0.0003861
2: -0.0065606, -0.0050208, -0.0065606, -0.0050208, -0.0011613, 0.0010652
3: 0.0038004, 0.0040140, 0.0038004, 0.0040140, -0.0001640, 0.0001705
4: 0.0023852, 0.0036022, 0.0023852, 0.0036022, -0.0009289, 0.0010043
5: 0.0060086, 0.0065111, 0.0060086, 0.0065111, -0.0005024, 0.0005024
6: -0.0013983, -0.0008639, -0.0013983, -0.0008639, -0.0003980, 0.0003646
7: -0.0083221, -0.0079387, -0.0083221, -0.0079387, -0.0003835, 0.0003835
8: 0.0048484, 0.0068715, 0.0048484, 0.0068715, -0.0011599, 0.0012871
9: -0.0036877, -0.0032079, -0.0036877, -0.0032079, -0.0004797, 0.0004797

### Unstable ReLU Count (Linear/Conv2D Layers)
- layer_idx=0, type=LayerType.Linear, total=256, inp1_unstable=5, inp2_unstable=5, delta_unstable=251
- layer_idx=2, type=LayerType.Linear, total=256, inp1_unstable=4, inp2_unstable=4, delta_unstable=255
- layer_idx=4, type=LayerType.Linear, total=256, inp1_unstable=0, inp2_unstable=0, delta_unstable=256
- layer_idx=6, type=LayerType.Linear, total=256, inp1_unstable=1, inp2_unstable=1, delta_unstable=256
- layer_idx=8, type=LayerType.Linear, total=10, inp1_unstable=0, inp2_unstable=0, delta_unstable=10

Time for backsubstitution: 1.26 seconds

### RS candidates at layer 3
type: RSZ, layer: 3, pos: 80
type: RSZ, layer: 3, pos: 81
type: RSZ, layer: 3, pos: 136

Time for candidate selection: 0.12 seconds

### Candidate
type: RSZ, layer: 3, pos: 80

### Relational analysis RSZ of RS_RSZ2_RSZ2_RSZ2_RSZ2_RSZ2_RSZ1_RSZ1

#### Relational analysis RSZ result of RS_RSZ2_RSZ2_RSZ2_RSZ2_RSZ2_RSZ1_RSZ1
Status: Status.UNKNOWN
Output dim: 1, lower bound: -0.0002841, upper bound: 0.0002785
time: 0.56 seconds

### Relational analysis RSZ of RS_RSZ2_RSZ2_RSZ2_RSZ2_RSZ2_RSZ1_RSZ2

#### Relational analysis RSZ result of RS_RSZ2_RSZ2_RSZ2_RSZ2_RSZ2_RSZ1_RSZ2
Status: Status.UNKNOWN
Output dim: 1, lower bound: -0.0002828, upper bound: 0.0002803
time: 0.57 seconds

## BFS RS instance: RS_RSZ2_RSZ2_RSZ2_RSZ2_RSZ2_RSZ2

### Backsubstitution after applying RS history:
0: 0.0009457, 0.0011003, 0.0009457, 0.0011003, -0.0001493, 0.0001490
1: 0.9936197, 0.9940771, 0.9936197, 0.9940771, -0.0003884, 0.0003923
2: -0.0065606, -0.0050208, -0.0065606, -0.0050208, -0.0010886, 0.0011302
3: 0.0038004, 0.0040140, 0.0038004, 0.0040140, -0.0001683, 0.0001656
4: 0.0023852, 0.0036022, 0.0023852, 0.0036022, -0.0009803, 0.0009471
5: 0.0060086, 0.0065111, 0.0060086, 0.0065111, -0.0005024, 0.0005024
6: -0.0013983, -0.0008639, -0.0013983, -0.0008639, -0.0003728, 0.0003872
7: -0.0083221, -0.0079387, -0.0083221, -0.0079387, -0.0003835, 0.0003835
8: 0.0048484, 0.0068715, 0.0048484, 0.0068715, -0.0012453, 0.0011916
9: -0.0036877, -0.0032079, -0.0036877, -0.0032079, -0.0004797, 0.0004797

### Unstable ReLU Count (Linear/Conv2D Layers)
- layer_idx=0, type=LayerType.Linear, total=256, inp1_unstable=5, inp2_unstable=5, delta_unstable=251
- layer_idx=2, type=LayerType.Linear, total=256, inp1_unstable=4, inp2_unstable=4, delta_unstable=255
- layer_idx=4, type=LayerType.Linear, total=256, inp1_unstable=0, inp2_unstable=0, delta_unstable=256
- layer_idx=6, type=LayerType.Linear, total=256, inp1_unstable=1, inp2_unstable=1, delta_unstable=256
- layer_idx=8, type=LayerType.Linear, total=10, inp1_unstable=0, inp2_unstable=0, delta_unstable=10

Time for backsubstitution: 1.27 seconds

### RS candidates at layer 3
type: RSZ, layer: 3, pos: 80
type: RSZ, layer: 3, pos: 81
type: RSZ, layer: 3, pos: 136

Time for candidate selection: 0.13 seconds

### Candidate
type: RSZ, layer: 3, pos: 80

### Relational analysis RSZ of RS_RSZ2_RSZ2_RSZ2_RSZ2_RSZ2_RSZ2_RSZ1

#### Relational analysis RSZ result of RS_RSZ2_RSZ2_RSZ2_RSZ2_RSZ2_RSZ2_RSZ1
Status: Status.UNKNOWN
Output dim: 1, lower bound: -0.0002804, upper bound: 0.0002833
time: 0.56 seconds

### Relational analysis RSZ of RS_RSZ2_RSZ2_RSZ2_RSZ2_RSZ2_RSZ2_RSZ2

#### Relational analysis RSZ result of RS_RSZ2_RSZ2_RSZ2_RSZ2_RSZ2_RSZ2_RSZ2
Status: Status.UNKNOWN
Output dim: 1, lower bound: -0.0002781, upper bound: 0.0002842
time: 0.58 seconds

## Summary of splitting (split count: 6)
- Time for RS candidates: 2.55 seconds
RS_RSZ1_RSZ1_RSZ1_RSZ1_RSZ1_RSZ1_RSZ1, status: Status.UNKNOWN, split count: 7, time: 2.55
Output dim: 1, lower bound: -0.0002842, upper bound: 0.0002781
RS_RSZ1_RSZ1_RSZ1_RSZ1_RSZ1_RSZ1_RSZ2, status: Status.UNKNOWN, split count: 7, time: 2.55
Output dim: 1, lower bound: -0.0002833, upper bound: 0.0002804
RS_RSZ1_RSZ1_RSZ1_RSZ1_RSZ1_RSZ2_RSZ1, status: Status.UNKNOWN, split count: 7, time: 2.55
Output dim: 1, lower bound: -0.0002803, upper bound: 0.0002828
RS_RSZ1_RSZ1_RSZ1_RSZ1_RSZ1_RSZ2_RSZ2, status: Status.UNKNOWN, split count: 7, time: 2.55
Output dim: 1, lower bound: -0.0002785, upper bound: 0.0002841
RS_RSZ1_RSZ1_RSZ1_RSZ1_RSZ2_RSZ1_RSZ1, status: Status.UNKNOWN, split count: 7, time: 2.55
Output dim: 1, lower bound: -0.0002840, upper bound: 0.0002779
RS_RSZ1_RSZ1_RSZ1_RSZ1_RSZ2_RSZ1_RSZ2, status: Status.UNKNOWN, split count: 7, time: 2.55
Output dim: 1, lower bound: -0.0002833, upper bound: 0.0002804
RS_RSZ1_RSZ1_RSZ1_RSZ1_RSZ2_RSZ2_RSZ1, status: Status.UNKNOWN, split count: 7, time: 2.55
Output dim: 1, lower bound: -0.0002802, upper bound: 0.0002827
RS_RSZ1_RSZ1_RSZ1_RSZ1_RSZ2_RSZ2_RSZ2, status: Status.UNKNOWN, split count: 7, time: 2.55
Output dim: 1, lower bound: -0.0002785, upper bound: 0.0002840
RS_RSZ1_RSZ1_RSZ1_RSZ2_RSZ1_RSZ1_RSZ1, status: Status.UNKNOWN, split count: 7, time: 2.55
Output dim: 1, lower bound: -0.0002831, upper bound: 0.0002785
RS_RSZ1_RSZ1_RSZ1_RSZ2_RSZ1_RSZ1_RSZ2, status: Status.UNKNOWN, split count: 7, time: 2.55
Output dim: 1, lower bound: -0.0002826, upper bound: 0.0002810
RS_RSZ1_RSZ1_RSZ1_RSZ2_RSZ1_RSZ2_RSZ1, status: Status.UNKNOWN, split count: 7, time: 2.55
Output dim: 1, lower bound: -0.0002796, upper bound: 0.0002835
RS_RSZ1_RSZ1_RSZ1_RSZ2_RSZ1_RSZ2_RSZ2, status: Status.UNKNOWN, split count: 7, time: 2.55
Output dim: 1, lower bound: -0.0002779, upper bound: 0.0002850
RS_RSZ1_RSZ1_RSZ1_RSZ2_RSZ2_RSZ1_RSZ1, status: Status.UNKNOWN, split count: 7, time: 2.55
Output dim: 1, lower bound: -0.0002831, upper bound: 0.0002785
RS_RSZ1_RSZ1_RSZ1_RSZ2_RSZ2_RSZ1_RSZ2, status: Status.UNKNOWN, split count: 7, time: 2.55
Output dim: 1, lower bound: -0.0002827, upper bound: 0.0002810
RS_RSZ1_RSZ1_RSZ1_RSZ2_RSZ2_RSZ2_RSZ1, status: Status.UNKNOWN, split count: 7, time: 2.55
Output dim: 1, lower bound: -0.0002795, upper bound: 0.0002835
RS_RSZ1_RSZ1_RSZ1_RSZ2_RSZ2_RSZ2_RSZ2, status: Status.UNKNOWN, split count: 7, time: 2.55
Output dim: 1, lower bound: -0.0002779, upper bound: 0.0002850
RS_RSZ1_RSZ1_RSZ2_RSZ1_RSZ1_RSZ1_RSZ1, status: Status.UNKNOWN, split count: 7, time: 2.55
Output dim: 1, lower bound: -0.0002850, upper bound: 0.0002779
RS_RSZ1_RSZ1_RSZ2_RSZ1_RSZ1_RSZ1_RSZ2, status: Status.UNKNOWN, split count: 7, time: 2.55
Output dim: 1, lower bound: -0.0002835, upper bound: 0.0002796
RS_RSZ1_RSZ1_RSZ2_RSZ1_RSZ1_RSZ2_RSZ1, status: Status.UNKNOWN, split count: 7, time: 2.55
Output dim: 1, lower bound: -0.0002810, upper bound: 0.0002827
RS_RSZ1_RSZ1_RSZ2_RSZ1_RSZ1_RSZ2_RSZ2, status: Status.UNKNOWN, split count: 7, time: 2.55
Output dim: 1, lower bound: -0.0002785, upper bound: 0.0002832
RS_RSZ1_RSZ1_RSZ2_RSZ1_RSZ2_RSZ1_RSZ1, status: Status.UNKNOWN, split count: 7, time: 2.55
Output dim: 1, lower bound: -0.0002849, upper bound: 0.0002779
RS_RSZ1_RSZ1_RSZ2_RSZ1_RSZ2_RSZ1_RSZ2, status: Status.UNKNOWN, split count: 7, time: 2.55
Output dim: 1, lower bound: -0.0002835, upper bound: 0.0002797
RS_RSZ1_RSZ1_RSZ2_RSZ1_RSZ2_RSZ2_RSZ1, status: Status.UNKNOWN, split count: 7, time: 2.55
Output dim: 1, lower bound: -0.0002810, upper bound: 0.0002826
RS_RSZ1_RSZ1_RSZ2_RSZ1_RSZ2_RSZ2_RSZ2, status: Status.UNKNOWN, split count: 7, time: 2.55
Output dim: 1, lower bound: -0.0002785, upper bound: 0.0002832
RS_RSZ1_RSZ1_RSZ2_RSZ2_RSZ1_RSZ1_RSZ1, status: Status.UNKNOWN, split count: 7, time: 2.55
Output dim: 1, lower bound: -0.0002839, upper bound: 0.0002785
RS_RSZ1_RSZ1_RSZ2_RSZ2_RSZ1_RSZ1_RSZ2, status: Status.UNKNOWN, split count: 7, time: 2.55
Output dim: 1, lower bound: -0.0002827, upper bound: 0.0002802
RS_RSZ1_RSZ1_RSZ2_RSZ2_RSZ1_RSZ2_RSZ1, status: Status.UNKNOWN, split count: 7, time: 2.55
Output dim: 1, lower bound: -0.0002801, upper bound: 0.0002833
RS_RSZ1_RSZ1_RSZ2_RSZ2_RSZ1_RSZ2_RSZ2, status: Status.UNKNOWN, split count: 7, time: 2.55
Output dim: 1, lower bound: -0.0002779, upper bound: 0.0002840
RS_RSZ1_RSZ1_RSZ2_RSZ2_RSZ2_RSZ1_RSZ1, status: Status.UNKNOWN, split count: 7, time: 2.55
Output dim: 1, lower bound: -0.0002839, upper bound: 0.0002785
RS_RSZ1_RSZ1_RSZ2_RSZ2_RSZ2_RSZ1_RSZ2, status: Status.UNKNOWN, split count: 7, time: 2.55
Output dim: 1, lower bound: -0.0002827, upper bound: 0.0002803
RS_RSZ1_RSZ1_RSZ2_RSZ2_RSZ2_RSZ2_RSZ1, status: Status.UNKNOWN, split count: 7, time: 2.55
Output dim: 1, lower bound: -0.0002801, upper bound: 0.0002833
RS_RSZ1_RSZ1_RSZ2_RSZ2_RSZ2_RSZ2_RSZ2, status: Status.UNKNOWN, split count: 7, time: 2.55
Output dim: 1, lower bound: -0.0002780, upper bound: 0.0002842
RS_RSZ1_RSZ2_RSZ1_RSZ1_RSZ1_RSZ1_RSZ1, status: Status.UNKNOWN, split count: 7, time: 2.55
Output dim: 1, lower bound: -0.0002842, upper bound: 0.0002780
RS_RSZ1_RSZ2_RSZ1_RSZ1_RSZ1_RSZ1_RSZ2, status: Status.UNKNOWN, split count: 7, time: 2.55
Output dim: 1, lower bound: -0.0002833, upper bound: 0.0002801
RS_RSZ1_RSZ2_RSZ1_RSZ1_RSZ1_RSZ2_RSZ1, status: Status.UNKNOWN, split count: 7, time: 2.55
Output dim: 1, lower bound: -0.0002803, upper bound: 0.0002827
RS_RSZ1_RSZ2_RSZ1_RSZ1_RSZ1_RSZ2_RSZ2, status: Status.UNKNOWN, split count: 7, time: 2.55
Output dim: 1, lower bound: -0.0002785, upper bound: 0.0002839
RS_RSZ1_RSZ2_RSZ1_RSZ1_RSZ2_RSZ1_RSZ1, status: Status.UNKNOWN, split count: 7, time: 2.55
Output dim: 1, lower bound: -0.0002840, upper bound: 0.0002779
RS_RSZ1_RSZ2_RSZ1_RSZ1_RSZ2_RSZ1_RSZ2, status: Status.UNKNOWN, split count: 7, time: 2.55
Output dim: 1, lower bound: -0.0002833, upper bound: 0.0002801
RS_RSZ1_RSZ2_RSZ1_RSZ1_RSZ2_RSZ2_RSZ1, status: Status.UNKNOWN, split count: 7, time: 2.55
Output dim: 1, lower bound: -0.0002802, upper bound: 0.0002827
RS_RSZ1_RSZ2_RSZ1_RSZ1_RSZ2_RSZ2_RSZ2, status: Status.UNKNOWN, split count: 7, time: 2.55
Output dim: 1, lower bound: -0.0002785, upper bound: 0.0002839
RS_RSZ1_RSZ2_RSZ1_RSZ2_RSZ1_RSZ1_RSZ1, status: Status.UNKNOWN, split count: 7, time: 2.55
Output dim: 1, lower bound: -0.0002832, upper bound: 0.0002785
RS_RSZ1_RSZ2_RSZ1_RSZ2_RSZ1_RSZ1_RSZ2, status: Status.UNKNOWN, split count: 7, time: 2.55
Output dim: 1, lower bound: -0.0002826, upper bound: 0.0002809
RS_RSZ1_RSZ2_RSZ1_RSZ2_RSZ1_RSZ2_RSZ1, status: Status.UNKNOWN, split count: 7, time: 2.55
Output dim: 1, lower bound: -0.0002797, upper bound: 0.0002835
RS_RSZ1_RSZ2_RSZ1_RSZ2_RSZ1_RSZ2_RSZ2, status: Status.UNKNOWN, split count: 7, time: 2.55
Output dim: 1, lower bound: -0.0002779, upper bound: 0.0002849
RS_RSZ1_RSZ2_RSZ1_RSZ2_RSZ2_RSZ1_RSZ1, status: Status.UNKNOWN, split count: 7, time: 2.55
Output dim: 1, lower bound: -0.0002832, upper bound: 0.0002785
RS_RSZ1_RSZ2_RSZ1_RSZ2_RSZ2_RSZ1_RSZ2, status: Status.UNKNOWN, split count: 7, time: 2.55
Output dim: 1, lower bound: -0.0002827, upper bound: 0.0002810
RS_RSZ1_RSZ2_RSZ1_RSZ2_RSZ2_RSZ2_RSZ1, status: Status.UNKNOWN, split count: 7, time: 2.55
Output dim: 1, lower bound: -0.0002796, upper bound: 0.0002835
RS_RSZ1_RSZ2_RSZ1_RSZ2_RSZ2_RSZ2_RSZ2, status: Status.UNKNOWN, split count: 7, time: 2.55
Output dim: 1, lower bound: -0.0002779, upper bound: 0.0002850
RS_RSZ1_RSZ2_RSZ2_RSZ1_RSZ1_RSZ1_RSZ1, status: Status.UNKNOWN, split count: 7, time: 2.55
Output dim: 1, lower bound: -0.0002850, upper bound: 0.0002779
RS_RSZ1_RSZ2_RSZ2_RSZ1_RSZ1_RSZ1_RSZ2, status: Status.UNKNOWN, split count: 7, time: 2.55
Output dim: 1, lower bound: -0.0002835, upper bound: 0.0002795
RS_RSZ1_RSZ2_RSZ2_RSZ1_RSZ1_RSZ2_RSZ1, status: Status.UNKNOWN, split count: 7, time: 2.55
Output dim: 1, lower bound: -0.0002810, upper bound: 0.0002827
RS_RSZ1_RSZ2_RSZ2_RSZ1_RSZ1_RSZ2_RSZ2, status: Status.UNKNOWN, split count: 7, time: 2.55
Output dim: 1, lower bound: -0.0002785, upper bound: 0.0002831
RS_RSZ1_RSZ2_RSZ2_RSZ1_RSZ2_RSZ1_RSZ1, status: Status.UNKNOWN, split count: 7, time: 2.55
Output dim: 1, lower bound: -0.0002850, upper bound: 0.0002779
RS_RSZ1_RSZ2_RSZ2_RSZ1_RSZ2_RSZ1_RSZ2, status: Status.UNKNOWN, split count: 7, time: 2.55
Output dim: 1, lower bound: -0.0002835, upper bound: 0.0002796
RS_RSZ1_RSZ2_RSZ2_RSZ1_RSZ2_RSZ2_RSZ1, status: Status.UNKNOWN, split count: 7, time: 2.55
Output dim: 1, lower bound: -0.0002810, upper bound: 0.0002826
RS_RSZ1_RSZ2_RSZ2_RSZ1_RSZ2_RSZ2_RSZ2, status: Status.UNKNOWN, split count: 7, time: 2.55
Output dim: 1, lower bound: -0.0002785, upper bound: 0.0002831
RS_RSZ1_RSZ2_RSZ2_RSZ2_RSZ1_RSZ1_RSZ1, status: Status.UNKNOWN, split count: 7, time: 2.55
Output dim: 1, lower bound: -0.0002840, upper bound: 0.0002785
RS_RSZ1_RSZ2_RSZ2_RSZ2_RSZ1_RSZ1_RSZ2, status: Status.UNKNOWN, split count: 7, time: 2.55
Output dim: 1, lower bound: -0.0002827, upper bound: 0.0002802
RS_RSZ1_RSZ2_RSZ2_RSZ2_RSZ1_RSZ2_RSZ1, status: Status.UNKNOWN, split count: 7, time: 2.55
Output dim: 1, lower bound: -0.0002804, upper bound: 0.0002833
RS_RSZ1_RSZ2_RSZ2_RSZ2_RSZ1_RSZ2_RSZ2, status: Status.UNKNOWN, split count: 7, time: 2.55
Output dim: 1, lower bound: -0.0002780, upper bound: 0.0002840
RS_RSZ1_RSZ2_RSZ2_RSZ2_RSZ2_RSZ1_RSZ1, status: Status.UNKNOWN, split count: 7, time: 2.55
Output dim: 1, lower bound: -0.0002841, upper bound: 0.0002785
RS_RSZ1_RSZ2_RSZ2_RSZ2_RSZ2_RSZ1_RSZ2, status: Status.UNKNOWN, split count: 7, time: 2.55
Output dim: 1, lower bound: -0.0002828, upper bound: 0.0002803
RS_RSZ1_RSZ2_RSZ2_RSZ2_RSZ2_RSZ2_RSZ1, status: Status.UNKNOWN, split count: 7, time: 2.55
Output dim: 1, lower bound: -0.0002804, upper bound: 0.0002833
RS_RSZ1_RSZ2_RSZ2_RSZ2_RSZ2_RSZ2_RSZ2, status: Status.UNKNOWN, split count: 7, time: 2.55
Output dim: 1, lower bound: -0.0002781, upper bound: 0.0002842
RS_RSZ2_RSZ1_RSZ1_RSZ1_RSZ1_RSZ1_RSZ1, status: Status.UNKNOWN, split count: 7, time: 2.55
Output dim: 1, lower bound: -0.0002842, upper bound: 0.0002781
RS_RSZ2_RSZ1_RSZ1_RSZ1_RSZ1_RSZ1_RSZ2, status: Status.UNKNOWN, split count: 7, time: 2.55
Output dim: 1, lower bound: -0.0002833, upper bound: 0.0002804
RS_RSZ2_RSZ1_RSZ1_RSZ1_RSZ1_RSZ2_RSZ1, status: Status.UNKNOWN, split count: 7, time: 2.55
Output dim: 1, lower bound: -0.0002803, upper bound: 0.0002828
RS_RSZ2_RSZ1_RSZ1_RSZ1_RSZ1_RSZ2_RSZ2, status: Status.UNKNOWN, split count: 7, time: 2.55
Output dim: 1, lower bound: -0.0002785, upper bound: 0.0002841
RS_RSZ2_RSZ1_RSZ1_RSZ1_RSZ2_RSZ1_RSZ1, status: Status.UNKNOWN, split count: 7, time: 2.55
Output dim: 1, lower bound: -0.0002840, upper bound: 0.0002780
RS_RSZ2_RSZ1_RSZ1_RSZ1_RSZ2_RSZ1_RSZ2, status: Status.UNKNOWN, split count: 7, time: 2.55
Output dim: 1, lower bound: -0.0002833, upper bound: 0.0002804
RS_RSZ2_RSZ1_RSZ1_RSZ1_RSZ2_RSZ2_RSZ1, status: Status.UNKNOWN, split count: 7, time: 2.55
Output dim: 1, lower bound: -0.0002802, upper bound: 0.0002827
RS_RSZ2_RSZ1_RSZ1_RSZ1_RSZ2_RSZ2_RSZ2, status: Status.UNKNOWN, split count: 7, time: 2.55
Output dim: 1, lower bound: -0.0002785, upper bound: 0.0002840
RS_RSZ2_RSZ1_RSZ1_RSZ2_RSZ1_RSZ1_RSZ1, status: Status.UNKNOWN, split count: 7, time: 2.55
Output dim: 1, lower bound: -0.0002831, upper bound: 0.0002785
RS_RSZ2_RSZ1_RSZ1_RSZ2_RSZ1_RSZ1_RSZ2, status: Status.UNKNOWN, split count: 7, time: 2.55
Output dim: 1, lower bound: -0.0002826, upper bound: 0.0002810
RS_RSZ2_RSZ1_RSZ1_RSZ2_RSZ1_RSZ2_RSZ1, status: Status.UNKNOWN, split count: 7, time: 2.55
Output dim: 1, lower bound: -0.0002796, upper bound: 0.0002835
RS_RSZ2_RSZ1_RSZ1_RSZ2_RSZ1_RSZ2_RSZ2, status: Status.UNKNOWN, split count: 7, time: 2.55
Output dim: 1, lower bound: -0.0002779, upper bound: 0.0002850
RS_RSZ2_RSZ1_RSZ1_RSZ2_RSZ2_RSZ1_RSZ1, status: Status.UNKNOWN, split count: 7, time: 2.55
Output dim: 1, lower bound: -0.0002831, upper bound: 0.0002785
RS_RSZ2_RSZ1_RSZ1_RSZ2_RSZ2_RSZ1_RSZ2, status: Status.UNKNOWN, split count: 7, time: 2.55
Output dim: 1, lower bound: -0.0002827, upper bound: 0.0002810
RS_RSZ2_RSZ1_RSZ1_RSZ2_RSZ2_RSZ2_RSZ1, status: Status.UNKNOWN, split count: 7, time: 2.55
Output dim: 1, lower bound: -0.0002795, upper bound: 0.0002835
RS_RSZ2_RSZ1_RSZ1_RSZ2_RSZ2_RSZ2_RSZ2, status: Status.UNKNOWN, split count: 7, time: 2.55
Output dim: 1, lower bound: -0.0002779, upper bound: 0.0002850
RS_RSZ2_RSZ1_RSZ2_RSZ1_RSZ1_RSZ1_RSZ1, status: Status.UNKNOWN, split count: 7, time: 2.55
Output dim: 1, lower bound: -0.0002850, upper bound: 0.0002779
RS_RSZ2_RSZ1_RSZ2_RSZ1_RSZ1_RSZ1_RSZ2, status: Status.UNKNOWN, split count: 7, time: 2.55
Output dim: 1, lower bound: -0.0002835, upper bound: 0.0002796
RS_RSZ2_RSZ1_RSZ2_RSZ1_RSZ1_RSZ2_RSZ1, status: Status.UNKNOWN, split count: 7, time: 2.55
Output dim: 1, lower bound: -0.0002810, upper bound: 0.0002827
RS_RSZ2_RSZ1_RSZ2_RSZ1_RSZ1_RSZ2_RSZ2, status: Status.UNKNOWN, split count: 7, time: 2.55
Output dim: 1, lower bound: -0.0002785, upper bound: 0.0002832
RS_RSZ2_RSZ1_RSZ2_RSZ1_RSZ2_RSZ1_RSZ1, status: Status.UNKNOWN, split count: 7, time: 2.55
Output dim: 1, lower bound: -0.0002849, upper bound: 0.0002779
RS_RSZ2_RSZ1_RSZ2_RSZ1_RSZ2_RSZ1_RSZ2, status: Status.UNKNOWN, split count: 7, time: 2.55
Output dim: 1, lower bound: -0.0002835, upper bound: 0.0002797
RS_RSZ2_RSZ1_RSZ2_RSZ1_RSZ2_RSZ2_RSZ1, status: Status.UNKNOWN, split count: 7, time: 2.55
Output dim: 1, lower bound: -0.0002810, upper bound: 0.0002826
RS_RSZ2_RSZ1_RSZ2_RSZ1_RSZ2_RSZ2_RSZ2, status: Status.UNKNOWN, split count: 7, time: 2.55
Output dim: 1, lower bound: -0.0002785, upper bound: 0.0002832
RS_RSZ2_RSZ1_RSZ2_RSZ2_RSZ1_RSZ1_RSZ1, status: Status.UNKNOWN, split count: 7, time: 2.55
Output dim: 1, lower bound: -0.0002839, upper bound: 0.0002785
RS_RSZ2_RSZ1_RSZ2_RSZ2_RSZ1_RSZ1_RSZ2, status: Status.UNKNOWN, split count: 7, time: 2.55
Output dim: 1, lower bound: -0.0002827, upper bound: 0.0002802
RS_RSZ2_RSZ1_RSZ2_RSZ2_RSZ1_RSZ2_RSZ1, status: Status.UNKNOWN, split count: 7, time: 2.55
Output dim: 1, lower bound: -0.0002801, upper bound: 0.0002833
RS_RSZ2_RSZ1_RSZ2_RSZ2_RSZ1_RSZ2_RSZ2, status: Status.UNKNOWN, split count: 7, time: 2.55
Output dim: 1, lower bound: -0.0002779, upper bound: 0.0002840
RS_RSZ2_RSZ1_RSZ2_RSZ2_RSZ2_RSZ1_RSZ1, status: Status.UNKNOWN, split count: 7, time: 2.55
Output dim: 1, lower bound: -0.0002839, upper bound: 0.0002785
RS_RSZ2_RSZ1_RSZ2_RSZ2_RSZ2_RSZ1_RSZ2, status: Status.UNKNOWN, split count: 7, time: 2.55
Output dim: 1, lower bound: -0.0002827, upper bound: 0.0002803
RS_RSZ2_RSZ1_RSZ2_RSZ2_RSZ2_RSZ2_RSZ1, status: Status.UNKNOWN, split count: 7, time: 2.55
Output dim: 1, lower bound: -0.0002801, upper bound: 0.0002833
RS_RSZ2_RSZ1_RSZ2_RSZ2_RSZ2_RSZ2_RSZ2, status: Status.UNKNOWN, split count: 7, time: 2.55
Output dim: 1, lower bound: -0.0002780, upper bound: 0.0002842
RS_RSZ2_RSZ2_RSZ1_RSZ1_RSZ1_RSZ1_RSZ1, status: Status.UNKNOWN, split count: 7, time: 2.55
Output dim: 1, lower bound: -0.0002842, upper bound: 0.0002780
RS_RSZ2_RSZ2_RSZ1_RSZ1_RSZ1_RSZ1_RSZ2, status: Status.UNKNOWN, split count: 7, time: 2.55
Output dim: 1, lower bound: -0.0002833, upper bound: 0.0002801
RS_RSZ2_RSZ2_RSZ1_RSZ1_RSZ1_RSZ2_RSZ1, status: Status.UNKNOWN, split count: 7, time: 2.55
Output dim: 1, lower bound: -0.0002803, upper bound: 0.0002827
RS_RSZ2_RSZ2_RSZ1_RSZ1_RSZ1_RSZ2_RSZ2, status: Status.UNKNOWN, split count: 7, time: 2.55
Output dim: 1, lower bound: -0.0002785, upper bound: 0.0002839
RS_RSZ2_RSZ2_RSZ1_RSZ1_RSZ2_RSZ1_RSZ1, status: Status.UNKNOWN, split count: 7, time: 2.55
Output dim: 1, lower bound: -0.0002840, upper bound: 0.0002779
RS_RSZ2_RSZ2_RSZ1_RSZ1_RSZ2_RSZ1_RSZ2, status: Status.UNKNOWN, split count: 7, time: 2.55
Output dim: 1, lower bound: -0.0002833, upper bound: 0.0002801
RS_RSZ2_RSZ2_RSZ1_RSZ1_RSZ2_RSZ2_RSZ1, status: Status.UNKNOWN, split count: 7, time: 2.55
Output dim: 1, lower bound: -0.0002802, upper bound: 0.0002827
RS_RSZ2_RSZ2_RSZ1_RSZ1_RSZ2_RSZ2_RSZ2, status: Status.UNKNOWN, split count: 7, time: 2.55
Output dim: 1, lower bound: -0.0002785, upper bound: 0.0002839
RS_RSZ2_RSZ2_RSZ1_RSZ2_RSZ1_RSZ1_RSZ1, status: Status.UNKNOWN, split count: 7, time: 2.55
Output dim: 1, lower bound: -0.0002832, upper bound: 0.0002785
RS_RSZ2_RSZ2_RSZ1_RSZ2_RSZ1_RSZ1_RSZ2, status: Status.UNKNOWN, split count: 7, time: 2.55
Output dim: 1, lower bound: -0.0002826, upper bound: 0.0002809
RS_RSZ2_RSZ2_RSZ1_RSZ2_RSZ1_RSZ2_RSZ1, status: Status.UNKNOWN, split count: 7, time: 2.55
Output dim: 1, lower bound: -0.0002797, upper bound: 0.0002835
RS_RSZ2_RSZ2_RSZ1_RSZ2_RSZ1_RSZ2_RSZ2, status: Status.UNKNOWN, split count: 7, time: 2.55
Output dim: 1, lower bound: -0.0002779, upper bound: 0.0002849
RS_RSZ2_RSZ2_RSZ1_RSZ2_RSZ2_RSZ1_RSZ1, status: Status.UNKNOWN, split count: 7, time: 2.55
Output dim: 1, lower bound: -0.0002832, upper bound: 0.0002785
RS_RSZ2_RSZ2_RSZ1_RSZ2_RSZ2_RSZ1_RSZ2, status: Status.UNKNOWN, split count: 7, time: 2.55
Output dim: 1, lower bound: -0.0002827, upper bound: 0.0002810
RS_RSZ2_RSZ2_RSZ1_RSZ2_RSZ2_RSZ2_RSZ1, status: Status.UNKNOWN, split count: 7, time: 2.55
Output dim: 1, lower bound: -0.0002796, upper bound: 0.0002835
RS_RSZ2_RSZ2_RSZ1_RSZ2_RSZ2_RSZ2_RSZ2, status: Status.UNKNOWN, split count: 7, time: 2.55
Output dim: 1, lower bound: -0.0002779, upper bound: 0.0002850
RS_RSZ2_RSZ2_RSZ2_RSZ1_RSZ1_RSZ1_RSZ1, status: Status.UNKNOWN, split count: 7, time: 2.55
Output dim: 1, lower bound: -0.0002850, upper bound: 0.0002779
RS_RSZ2_RSZ2_RSZ2_RSZ1_RSZ1_RSZ1_RSZ2, status: Status.UNKNOWN, split count: 7, time: 2.55
Output dim: 1, lower bound: -0.0002835, upper bound: 0.0002795
RS_RSZ2_RSZ2_RSZ2_RSZ1_RSZ1_RSZ2_RSZ1, status: Status.UNKNOWN, split count: 7, time: 2.55
Output dim: 1, lower bound: -0.0002810, upper bound: 0.0002827
RS_RSZ2_RSZ2_RSZ2_RSZ1_RSZ1_RSZ2_RSZ2, status: Status.UNKNOWN, split count: 7, time: 2.55
Output dim: 1, lower bound: -0.0002785, upper bound: 0.0002831
RS_RSZ2_RSZ2_RSZ2_RSZ1_RSZ2_RSZ1_RSZ1, status: Status.UNKNOWN, split count: 7, time: 2.55
Output dim: 1, lower bound: -0.0002850, upper bound: 0.0002779
RS_RSZ2_RSZ2_RSZ2_RSZ1_RSZ2_RSZ1_RSZ2, status: Status.UNKNOWN, split count: 7, time: 2.55
Output dim: 1, lower bound: -0.0002835, upper bound: 0.0002796
RS_RSZ2_RSZ2_RSZ2_RSZ1_RSZ2_RSZ2_RSZ1, status: Status.UNKNOWN, split count: 7, time: 2.55
Output dim: 1, lower bound: -0.0002810, upper bound: 0.0002826
RS_RSZ2_RSZ2_RSZ2_RSZ1_RSZ2_RSZ2_RSZ2, status: Status.UNKNOWN, split count: 7, time: 2.55
Output dim: 1, lower bound: -0.0002785, upper bound: 0.0002831
RS_RSZ2_RSZ2_RSZ2_RSZ2_RSZ1_RSZ1_RSZ1, status: Status.UNKNOWN, split count: 7, time: 2.55
Output dim: 1, lower bound: -0.0002840, upper bound: 0.0002785
RS_RSZ2_RSZ2_RSZ2_RSZ2_RSZ1_RSZ1_RSZ2, status: Status.UNKNOWN, split count: 7, time: 2.55
Output dim: 1, lower bound: -0.0002827, upper bound: 0.0002802
RS_RSZ2_RSZ2_RSZ2_RSZ2_RSZ1_RSZ2_RSZ1, status: Status.UNKNOWN, split count: 7, time: 2.55
Output dim: 1, lower bound: -0.0002804, upper bound: 0.0002833
RS_RSZ2_RSZ2_RSZ2_RSZ2_RSZ1_RSZ2_RSZ2, status: Status.UNKNOWN, split count: 7, time: 2.55
Output dim: 1, lower bound: -0.0002780, upper bound: 0.0002840
RS_RSZ2_RSZ2_RSZ2_RSZ2_RSZ2_RSZ1_RSZ1, status: Status.UNKNOWN, split count: 7, time: 2.55
Output dim: 1, lower bound: -0.0002841, upper bound: 0.0002785
RS_RSZ2_RSZ2_RSZ2_RSZ2_RSZ2_RSZ1_RSZ2, status: Status.UNKNOWN, split count: 7, time: 2.55
Output dim: 1, lower bound: -0.0002828, upper bound: 0.0002803
RS_RSZ2_RSZ2_RSZ2_RSZ2_RSZ2_RSZ2_RSZ1, status: Status.UNKNOWN, split count: 7, time: 2.55
Output dim: 1, lower bound: -0.0002804, upper bound: 0.0002833
RS_RSZ2_RSZ2_RSZ2_RSZ2_RSZ2_RSZ2_RSZ2, status: Status.UNKNOWN, split count: 7, time: 2.55
Output dim: 1, lower bound: -0.0002781, upper bound: 0.0002842

## BFS RS instance: RS_RSZ1_RSZ1_RSZ1_RSZ1_RSZ1_RSZ1_RSZ1

### Backsubstitution after applying RS history:
0: 0.0009457, 0.0011003, 0.0009457, 0.0011003, -0.0001486, 0.0001488
1: 0.9936197, 0.9940771, 0.9936197, 0.9940771, -0.0003858, 0.0003832
2: -0.0065606, -0.0050208, -0.0065606, -0.0050208, -0.0011817, 0.0011551
3: 0.0038004, 0.0040140, 0.0038004, 0.0040140, -0.0001623, 0.0001641
4: 0.0023852, 0.0036022, 0.0023852, 0.0036022, -0.0010177, 0.0010387
5: 0.0060086, 0.0065111, 0.0060086, 0.0065111, -0.0005024, 0.0005024
6: -0.0013983, -0.0008639, -0.0013983, -0.0008639, -0.0004037, 0.0003944
7: -0.0083221, -0.0079387, -0.0083221, -0.0079387, -0.0003835, 0.0003835
8: 0.0048484, 0.0068715, 0.0048484, 0.0068715, -0.0011751, 0.0012100
9: -0.0036877, -0.0032079, -0.0036877, -0.0032079, -0.0004797, 0.0004797

### Unstable ReLU Count (Linear/Conv2D Layers)
- layer_idx=0, type=LayerType.Linear, total=256, inp1_unstable=5, inp2_unstable=5, delta_unstable=251
- layer_idx=2, type=LayerType.Linear, total=256, inp1_unstable=4, inp2_unstable=4, delta_unstable=255
- layer_idx=4, type=LayerType.Linear, total=256, inp1_unstable=0, inp2_unstable=0, delta_unstable=256
- layer_idx=6, type=LayerType.Linear, total=256, inp1_unstable=1, inp2_unstable=1, delta_unstable=256
- layer_idx=8, type=LayerType.Linear, total=10, inp1_unstable=0, inp2_unstable=0, delta_unstable=10

Time for backsubstitution: 1.21 seconds

### RS candidates at layer 3
type: RSZ, layer: 3, pos: 73
type: RSZ, layer: 3, pos: 81
type: RSZ, layer: 3, pos: 136

Time for candidate selection: 0.12 seconds

### Candidate
type: RSZ, layer: 3, pos: 73

### Candidate
type: RSZ, layer: 3, pos: 81

### Relational analysis RSZ of RS_RSZ1_RSZ1_RSZ1_RSZ1_RSZ1_RSZ1_RSZ1_RSZ1

#### Relational analysis RSZ result of RS_RSZ1_RSZ1_RSZ1_RSZ1_RSZ1_RSZ1_RSZ1_RSZ1
Status: Status.VERIFIED
Output dim: 1, lower bound: -0.0002493, upper bound: 0.0002462
time: 0.58 seconds

### Relational analysis RSZ of RS_RSZ1_RSZ1_RSZ1_RSZ1_RSZ1_RSZ1_RSZ1_RSZ2

#### Relational analysis RSZ result of RS_RSZ1_RSZ1_RSZ1_RSZ1_RSZ1_RSZ1_RSZ1_RSZ2
Status: Status.VERIFIED
Output dim: 1, lower bound: -0.0002493, upper bound: 0.0002462
time: 0.59 seconds

## BFS RS instance: RS_RSZ1_RSZ1_RSZ1_RSZ1_RSZ1_RSZ1_RSZ2

### Backsubstitution after applying RS history:
0: 0.0009457, 0.0011003, 0.0009457, 0.0011003, -0.0001488, 0.0001486
1: 0.9936197, 0.9940771, 0.9936197, 0.9940771, -0.0003836, 0.0003853
2: -0.0065606, -0.0050208, -0.0065606, -0.0050208, -0.0011589, 0.0011764
3: 0.0038004, 0.0040140, 0.0038004, 0.0040140, -0.0001637, 0.0001626
4: 0.0023852, 0.0036022, 0.0023852, 0.0036022, -0.0010345, 0.0010207
5: 0.0060086, 0.0065111, 0.0060086, 0.0065111, -0.0005024, 0.0005024
6: -0.0013983, -0.0008639, -0.0013983, -0.0008639, -0.0003958, 0.0004018
7: -0.0083221, -0.0079387, -0.0083221, -0.0079387, -0.0003835, 0.0003835
8: 0.0048484, 0.0068715, 0.0048484, 0.0068715, -0.0012030, 0.0011801
9: -0.0036877, -0.0032079, -0.0036877, -0.0032079, -0.0004797, 0.0004797

### Unstable ReLU Count (Linear/Conv2D Layers)
- layer_idx=0, type=LayerType.Linear, total=256, inp1_unstable=5, inp2_unstable=5, delta_unstable=251
- layer_idx=2, type=LayerType.Linear, total=256, inp1_unstable=4, inp2_unstable=4, delta_unstable=255
- layer_idx=4, type=LayerType.Linear, total=256, inp1_unstable=0, inp2_unstable=0, delta_unstable=256
- layer_idx=6, type=LayerType.Linear, total=256, inp1_unstable=1, inp2_unstable=1, delta_unstable=256
- layer_idx=8, type=LayerType.Linear, total=10, inp1_unstable=0, inp2_unstable=0, delta_unstable=10

Time for backsubstitution: 1.24 seconds

### RS candidates at layer 3
type: RSZ, layer: 3, pos: 73
type: RSZ, layer: 3, pos: 81
type: RSZ, layer: 3, pos: 136

Time for candidate selection: 0.15 seconds

### Candidate
type: RSZ, layer: 3, pos: 73

### Candidate
type: RSZ, layer: 3, pos: 81

### Relational analysis RSZ of RS_RSZ1_RSZ1_RSZ1_RSZ1_RSZ1_RSZ1_RSZ2_RSZ1

#### Relational analysis RSZ result of RS_RSZ1_RSZ1_RSZ1_RSZ1_RSZ1_RSZ1_RSZ2_RSZ1
Status: Status.VERIFIED
Output dim: 1, lower bound: -0.0002468, upper bound: 0.0002490
time: 0.61 seconds

### Relational analysis RSZ of RS_RSZ1_RSZ1_RSZ1_RSZ1_RSZ1_RSZ1_RSZ2_RSZ2

#### Relational analysis RSZ result of RS_RSZ1_RSZ1_RSZ1_RSZ1_RSZ1_RSZ1_RSZ2_RSZ2
Status: Status.VERIFIED
Output dim: 1, lower bound: -0.0002468, upper bound: 0.0002490
time: 0.59 seconds

## BFS RS instance: RS_RSZ1_RSZ1_RSZ1_RSZ1_RSZ1_RSZ2_RSZ1

### Backsubstitution after applying RS history:
0: 0.0009457, 0.0011003, 0.0009457, 0.0011003, -0.0001486, 0.0001488
1: 0.9936197, 0.9940771, 0.9936197, 0.9940771, -0.0003858, 0.0003832
2: -0.0065606, -0.0050208, -0.0065606, -0.0050208, -0.0011817, 0.0011551
3: 0.0038004, 0.0040140, 0.0038004, 0.0040140, -0.0001623, 0.0001641
4: 0.0023852, 0.0036022, 0.0023852, 0.0036022, -0.0010177, 0.0010387
5: 0.0060086, 0.0065111, 0.0060086, 0.0065111, -0.0005024, 0.0005024
6: -0.0013983, -0.0008639, -0.0013983, -0.0008639, -0.0004037, 0.0003944
7: -0.0083221, -0.0079387, -0.0083221, -0.0079387, -0.0003835, 0.0003835
8: 0.0048484, 0.0068715, 0.0048484, 0.0068715, -0.0011751, 0.0012100
9: -0.0036877, -0.0032079, -0.0036877, -0.0032079, -0.0004797, 0.0004797

### Unstable ReLU Count (Linear/Conv2D Layers)
- layer_idx=0, type=LayerType.Linear, total=256, inp1_unstable=5, inp2_unstable=5, delta_unstable=251
- layer_idx=2, type=LayerType.Linear, total=256, inp1_unstable=4, inp2_unstable=4, delta_unstable=255
- layer_idx=4, type=LayerType.Linear, total=256, inp1_unstable=0, inp2_unstable=0, delta_unstable=256
- layer_idx=6, type=LayerType.Linear, total=256, inp1_unstable=1, inp2_unstable=1, delta_unstable=256
- layer_idx=8, type=LayerType.Linear, total=10, inp1_unstable=0, inp2_unstable=0, delta_unstable=10

Time for backsubstitution: 1.20 seconds

### RS candidates at layer 3
type: RSZ, layer: 3, pos: 73
type: RSZ, layer: 3, pos: 81
type: RSZ, layer: 3, pos: 136

Time for candidate selection: 0.12 seconds

### Candidate
type: RSZ, layer: 3, pos: 73

### Candidate
type: RSZ, layer: 3, pos: 81

### Relational analysis RSZ of RS_RSZ1_RSZ1_RSZ1_RSZ1_RSZ1_RSZ2_RSZ1_RSZ1

#### Relational analysis RSZ result of RS_RSZ1_RSZ1_RSZ1_RSZ1_RSZ1_RSZ2_RSZ1_RSZ1
Status: Status.VERIFIED
Output dim: 1, lower bound: -0.0002493, upper bound: 0.0002462
time: 0.55 seconds

### Relational analysis RSZ of RS_RSZ1_RSZ1_RSZ1_RSZ1_RSZ1_RSZ2_RSZ1_RSZ2

#### Relational analysis RSZ result of RS_RSZ1_RSZ1_RSZ1_RSZ1_RSZ1_RSZ2_RSZ1_RSZ2
Status: Status.VERIFIED
Output dim: 1, lower bound: -0.0002493, upper bound: 0.0002462
time: 0.55 seconds

## BFS RS instance: RS_RSZ1_RSZ1_RSZ1_RSZ1_RSZ1_RSZ2_RSZ2

### Backsubstitution after applying RS history:
0: 0.0009457, 0.0011003, 0.0009457, 0.0011003, -0.0001488, 0.0001486
1: 0.9936197, 0.9940771, 0.9936197, 0.9940771, -0.0003836, 0.0003853
2: -0.0065606, -0.0050208, -0.0065606, -0.0050208, -0.0011589, 0.0011764
3: 0.0038004, 0.0040140, 0.0038004, 0.0040140, -0.0001637, 0.0001626
4: 0.0023852, 0.0036022, 0.0023852, 0.0036022, -0.0010345, 0.0010207
5: 0.0060086, 0.0065111, 0.0060086, 0.0065111, -0.0005024, 0.0005024
6: -0.0013983, -0.0008639, -0.0013983, -0.0008639, -0.0003958, 0.0004018
7: -0.0083221, -0.0079387, -0.0083221, -0.0079387, -0.0003835, 0.0003835
8: 0.0048484, 0.0068715, 0.0048484, 0.0068715, -0.0012030, 0.0011801
9: -0.0036877, -0.0032079, -0.0036877, -0.0032079, -0.0004797, 0.0004797

### Unstable ReLU Count (Linear/Conv2D Layers)
- layer_idx=0, type=LayerType.Linear, total=256, inp1_unstable=5, inp2_unstable=5, delta_unstable=251
- layer_idx=2, type=LayerType.Linear, total=256, inp1_unstable=4, inp2_unstable=4, delta_unstable=255
- layer_idx=4, type=LayerType.Linear, total=256, inp1_unstable=0, inp2_unstable=0, delta_unstable=256
- layer_idx=6, type=LayerType.Linear, total=256, inp1_unstable=1, inp2_unstable=1, delta_unstable=256
- layer_idx=8, type=LayerType.Linear, total=10, inp1_unstable=0, inp2_unstable=0, delta_unstable=10

Time for backsubstitution: 1.20 seconds

### RS candidates at layer 3
type: RSZ, layer: 3, pos: 73
type: RSZ, layer: 3, pos: 81
type: RSZ, layer: 3, pos: 136

Time for candidate selection: 0.12 seconds

### Candidate
type: RSZ, layer: 3, pos: 73

### Candidate
type: RSZ, layer: 3, pos: 81

### Relational analysis RSZ of RS_RSZ1_RSZ1_RSZ1_RSZ1_RSZ1_RSZ2_RSZ2_RSZ1

#### Relational analysis RSZ result of RS_RSZ1_RSZ1_RSZ1_RSZ1_RSZ1_RSZ2_RSZ2_RSZ1
Status: Status.VERIFIED
Output dim: 1, lower bound: -0.0002468, upper bound: 0.0002490
time: 0.57 seconds

### Relational analysis RSZ of RS_RSZ1_RSZ1_RSZ1_RSZ1_RSZ1_RSZ2_RSZ2_RSZ2

#### Relational analysis RSZ result of RS_RSZ1_RSZ1_RSZ1_RSZ1_RSZ1_RSZ2_RSZ2_RSZ2
Status: Status.VERIFIED
Output dim: 1, lower bound: -0.0002468, upper bound: 0.0002490
time: 0.56 seconds

## BFS RS instance: RS_RSZ1_RSZ1_RSZ1_RSZ1_RSZ2_RSZ1_RSZ1

### Backsubstitution after applying RS history:
0: 0.0009457, 0.0011003, 0.0009457, 0.0011003, -0.0001486, 0.0001488
1: 0.9936197, 0.9940771, 0.9936197, 0.9940771, -0.0003854, 0.0003834
2: -0.0065606, -0.0050208, -0.0065606, -0.0050208, -0.0011780, 0.0011570
3: 0.0038004, 0.0040140, 0.0038004, 0.0040140, -0.0001625, 0.0001638
4: 0.0023852, 0.0036022, 0.0023852, 0.0036022, -0.0010192, 0.0010358
5: 0.0060086, 0.0065111, 0.0060086, 0.0065111, -0.0005024, 0.0005024
6: -0.0013983, -0.0008639, -0.0013983, -0.0008639, -0.0004024, 0.0003951
7: -0.0083221, -0.0079387, -0.0083221, -0.0079387, -0.0003835, 0.0003835
8: 0.0048484, 0.0068715, 0.0048484, 0.0068715, -0.0011775, 0.0012051
9: -0.0036877, -0.0032079, -0.0036877, -0.0032079, -0.0004797, 0.0004797

### Unstable ReLU Count (Linear/Conv2D Layers)
- layer_idx=0, type=LayerType.Linear, total=256, inp1_unstable=5, inp2_unstable=5, delta_unstable=251
- layer_idx=2, type=LayerType.Linear, total=256, inp1_unstable=4, inp2_unstable=4, delta_unstable=255
- layer_idx=4, type=LayerType.Linear, total=256, inp1_unstable=0, inp2_unstable=0, delta_unstable=256
- layer_idx=6, type=LayerType.Linear, total=256, inp1_unstable=1, inp2_unstable=1, delta_unstable=256
- layer_idx=8, type=LayerType.Linear, total=10, inp1_unstable=0, inp2_unstable=0, delta_unstable=10

Time for backsubstitution: 1.22 seconds

### RS candidates at layer 3
type: RSZ, layer: 3, pos: 73
type: RSZ, layer: 3, pos: 81
type: RSZ, layer: 3, pos: 136

Time for candidate selection: 0.14 seconds

### Candidate
type: RSZ, layer: 3, pos: 73

### Candidate
type: RSZ, layer: 3, pos: 81

### Relational analysis RSZ of RS_RSZ1_RSZ1_RSZ1_RSZ1_RSZ2_RSZ1_RSZ1_RSZ1

#### Relational analysis RSZ result of RS_RSZ1_RSZ1_RSZ1_RSZ1_RSZ2_RSZ1_RSZ1_RSZ1
Status: Status.VERIFIED
Output dim: 1, lower bound: -0.0002493, upper bound: 0.0002462
time: 0.61 seconds

### Relational analysis RSZ of RS_RSZ1_RSZ1_RSZ1_RSZ1_RSZ2_RSZ1_RSZ1_RSZ2

#### Relational analysis RSZ result of RS_RSZ1_RSZ1_RSZ1_RSZ1_RSZ2_RSZ1_RSZ1_RSZ2
Status: Status.VERIFIED
Output dim: 1, lower bound: -0.0002493, upper bound: 0.0002462
time: 0.60 seconds

## BFS RS instance: RS_RSZ1_RSZ1_RSZ1_RSZ1_RSZ2_RSZ1_RSZ2

### Backsubstitution after applying RS history:
0: 0.0009457, 0.0011003, 0.0009457, 0.0011003, -0.0001488, 0.0001486
1: 0.9936197, 0.9940771, 0.9936197, 0.9940771, -0.0003833, 0.0003856
2: -0.0065606, -0.0050208, -0.0065606, -0.0050208, -0.0011560, 0.0011797
3: 0.0038004, 0.0040140, 0.0038004, 0.0040140, -0.0001640, 0.0001624
4: 0.0023852, 0.0036022, 0.0023852, 0.0036022, -0.0010371, 0.0010184
5: 0.0060086, 0.0065111, 0.0060086, 0.0065111, -0.0005024, 0.0005024
6: -0.0013983, -0.0008639, -0.0013983, -0.0008639, -0.0003947, 0.0004030
7: -0.0083221, -0.0079387, -0.0083221, -0.0079387, -0.0003835, 0.0003835
8: 0.0048484, 0.0068715, 0.0048484, 0.0068715, -0.0012073, 0.0011762
9: -0.0036877, -0.0032079, -0.0036877, -0.0032079, -0.0004797, 0.0004797

### Unstable ReLU Count (Linear/Conv2D Layers)
- layer_idx=0, type=LayerType.Linear, total=256, inp1_unstable=5, inp2_unstable=5, delta_unstable=251
- layer_idx=2, type=LayerType.Linear, total=256, inp1_unstable=4, inp2_unstable=4, delta_unstable=255
- layer_idx=4, type=LayerType.Linear, total=256, inp1_unstable=0, inp2_unstable=0, delta_unstable=256
- layer_idx=6, type=LayerType.Linear, total=256, inp1_unstable=1, inp2_unstable=1, delta_unstable=256
- layer_idx=8, type=LayerType.Linear, total=10, inp1_unstable=0, inp2_unstable=0, delta_unstable=10

Time for backsubstitution: 1.18 seconds

### RS candidates at layer 3
type: RSZ, layer: 3, pos: 73
type: RSZ, layer: 3, pos: 81
type: RSZ, layer: 3, pos: 136

Time for candidate selection: 0.12 seconds

### Candidate
type: RSZ, layer: 3, pos: 73

### Candidate
type: RSZ, layer: 3, pos: 81

### Relational analysis RSZ of RS_RSZ1_RSZ1_RSZ1_RSZ1_RSZ2_RSZ1_RSZ2_RSZ1

#### Relational analysis RSZ result of RS_RSZ1_RSZ1_RSZ1_RSZ1_RSZ2_RSZ1_RSZ2_RSZ1
Status: Status.VERIFIED
Output dim: 1, lower bound: -0.0002468, upper bound: 0.0002490
time: 0.55 seconds

### Relational analysis RSZ of RS_RSZ1_RSZ1_RSZ1_RSZ1_RSZ2_RSZ1_RSZ2_RSZ2

#### Relational analysis RSZ result of RS_RSZ1_RSZ1_RSZ1_RSZ1_RSZ2_RSZ1_RSZ2_RSZ2
Status: Status.VERIFIED
Output dim: 1, lower bound: -0.0002468, upper bound: 0.0002490
time: 0.56 seconds

## BFS RS instance: RS_RSZ1_RSZ1_RSZ1_RSZ1_RSZ2_RSZ2_RSZ1

### Backsubstitution after applying RS history:
0: 0.0009457, 0.0011003, 0.0009457, 0.0011003, -0.0001486, 0.0001488
1: 0.9936197, 0.9940771, 0.9936197, 0.9940771, -0.0003854, 0.0003834
2: -0.0065606, -0.0050208, -0.0065606, -0.0050208, -0.0011780, 0.0011570
3: 0.0038004, 0.0040140, 0.0038004, 0.0040140, -0.0001625, 0.0001638
4: 0.0023852, 0.0036022, 0.0023852, 0.0036022, -0.0010192, 0.0010358
5: 0.0060086, 0.0065111, 0.0060086, 0.0065111, -0.0005024, 0.0005024
6: -0.0013983, -0.0008639, -0.0013983, -0.0008639, -0.0004024, 0.0003951
7: -0.0083221, -0.0079387, -0.0083221, -0.0079387, -0.0003835, 0.0003835
8: 0.0048484, 0.0068715, 0.0048484, 0.0068715, -0.0011775, 0.0012051
9: -0.0036877, -0.0032079, -0.0036877, -0.0032079, -0.0004797, 0.0004797

### Unstable ReLU Count (Linear/Conv2D Layers)
- layer_idx=0, type=LayerType.Linear, total=256, inp1_unstable=5, inp2_unstable=5, delta_unstable=251
- layer_idx=2, type=LayerType.Linear, total=256, inp1_unstable=4, inp2_unstable=4, delta_unstable=255
- layer_idx=4, type=LayerType.Linear, total=256, inp1_unstable=0, inp2_unstable=0, delta_unstable=256
- layer_idx=6, type=LayerType.Linear, total=256, inp1_unstable=1, inp2_unstable=1, delta_unstable=256
- layer_idx=8, type=LayerType.Linear, total=10, inp1_unstable=0, inp2_unstable=0, delta_unstable=10

Time for backsubstitution: 1.20 seconds

### RS candidates at layer 3
type: RSZ, layer: 3, pos: 73
type: RSZ, layer: 3, pos: 81
type: RSZ, layer: 3, pos: 136

Time for candidate selection: 0.12 seconds

### Candidate
type: RSZ, layer: 3, pos: 73

### Candidate
type: RSZ, layer: 3, pos: 81

### Relational analysis RSZ of RS_RSZ1_RSZ1_RSZ1_RSZ1_RSZ2_RSZ2_RSZ1_RSZ1

#### Relational analysis RSZ result of RS_RSZ1_RSZ1_RSZ1_RSZ1_RSZ2_RSZ2_RSZ1_RSZ1
Status: Status.VERIFIED
Output dim: 1, lower bound: -0.0002493, upper bound: 0.0002462
time: 0.61 seconds

### Relational analysis RSZ of RS_RSZ1_RSZ1_RSZ1_RSZ1_RSZ2_RSZ2_RSZ1_RSZ2

#### Relational analysis RSZ result of RS_RSZ1_RSZ1_RSZ1_RSZ1_RSZ2_RSZ2_RSZ1_RSZ2
Status: Status.VERIFIED
Output dim: 1, lower bound: -0.0002493, upper bound: 0.0002462
time: 0.62 seconds

## BFS RS instance: RS_RSZ1_RSZ1_RSZ1_RSZ1_RSZ2_RSZ2_RSZ2

### Backsubstitution after applying RS history:
0: 0.0009457, 0.0011003, 0.0009457, 0.0011003, -0.0001488, 0.0001486
1: 0.9936197, 0.9940771, 0.9936197, 0.9940771, -0.0003833, 0.0003856
2: -0.0065606, -0.0050208, -0.0065606, -0.0050208, -0.0011560, 0.0011797
3: 0.0038004, 0.0040140, 0.0038004, 0.0040140, -0.0001640, 0.0001624
4: 0.0023852, 0.0036022, 0.0023852, 0.0036022, -0.0010371, 0.0010184
5: 0.0060086, 0.0065111, 0.0060086, 0.0065111, -0.0005024, 0.0005024
6: -0.0013983, -0.0008639, -0.0013983, -0.0008639, -0.0003947, 0.0004030
7: -0.0083221, -0.0079387, -0.0083221, -0.0079387, -0.0003835, 0.0003835
8: 0.0048484, 0.0068715, 0.0048484, 0.0068715, -0.0012073, 0.0011762
9: -0.0036877, -0.0032079, -0.0036877, -0.0032079, -0.0004797, 0.0004797

### Unstable ReLU Count (Linear/Conv2D Layers)
- layer_idx=0, type=LayerType.Linear, total=256, inp1_unstable=5, inp2_unstable=5, delta_unstable=251
- layer_idx=2, type=LayerType.Linear, total=256, inp1_unstable=4, inp2_unstable=4, delta_unstable=255
- layer_idx=4, type=LayerType.Linear, total=256, inp1_unstable=0, inp2_unstable=0, delta_unstable=256
- layer_idx=6, type=LayerType.Linear, total=256, inp1_unstable=1, inp2_unstable=1, delta_unstable=256
- layer_idx=8, type=LayerType.Linear, total=10, inp1_unstable=0, inp2_unstable=0, delta_unstable=10

Time for backsubstitution: 1.20 seconds

### RS candidates at layer 3
type: RSZ, layer: 3, pos: 73
type: RSZ, layer: 3, pos: 81
type: RSZ, layer: 3, pos: 136

Time for candidate selection: 0.12 seconds

### Candidate
type: RSZ, layer: 3, pos: 73

### Candidate
type: RSZ, layer: 3, pos: 81

### Relational analysis RSZ of RS_RSZ1_RSZ1_RSZ1_RSZ1_RSZ2_RSZ2_RSZ2_RSZ1

#### Relational analysis RSZ result of RS_RSZ1_RSZ1_RSZ1_RSZ1_RSZ2_RSZ2_RSZ2_RSZ1
Status: Status.VERIFIED
Output dim: 1, lower bound: -0.0002468, upper bound: 0.0002490
time: 0.60 seconds

### Relational analysis RSZ of RS_RSZ1_RSZ1_RSZ1_RSZ1_RSZ2_RSZ2_RSZ2_RSZ2

#### Relational analysis RSZ result of RS_RSZ1_RSZ1_RSZ1_RSZ1_RSZ2_RSZ2_RSZ2_RSZ2
Status: Status.VERIFIED
Output dim: 1, lower bound: -0.0002468, upper bound: 0.0002490
time: 0.59 seconds

## BFS RS instance: RS_RSZ1_RSZ1_RSZ1_RSZ2_RSZ1_RSZ1_RSZ1

### Backsubstitution after applying RS history:
0: 0.0009457, 0.0011003, 0.0009457, 0.0011003, -0.0001486, 0.0001488
1: 0.9936197, 0.9940771, 0.9936197, 0.9940771, -0.0003853, 0.0003833
2: -0.0065606, -0.0050208, -0.0065606, -0.0050208, -0.0011768, 0.0011556
3: 0.0038004, 0.0040140, 0.0038004, 0.0040140, -0.0001624, 0.0001638
4: 0.0023852, 0.0036022, 0.0023852, 0.0036022, -0.0010181, 0.0010348
5: 0.0060086, 0.0065111, 0.0060086, 0.0065111, -0.0005024, 0.0005024
6: -0.0013983, -0.0008639, -0.0013983, -0.0008639, -0.0004020, 0.0003946
7: -0.0083221, -0.0079387, -0.0083221, -0.0079387, -0.0003835, 0.0003835
8: 0.0048484, 0.0068715, 0.0048484, 0.0068715, -0.0011757, 0.0012035
9: -0.0036877, -0.0032079, -0.0036877, -0.0032079, -0.0004797, 0.0004797

### Unstable ReLU Count (Linear/Conv2D Layers)
- layer_idx=0, type=LayerType.Linear, total=256, inp1_unstable=5, inp2_unstable=5, delta_unstable=251
- layer_idx=2, type=LayerType.Linear, total=256, inp1_unstable=4, inp2_unstable=4, delta_unstable=255
- layer_idx=4, type=LayerType.Linear, total=256, inp1_unstable=0, inp2_unstable=0, delta_unstable=256
- layer_idx=6, type=LayerType.Linear, total=256, inp1_unstable=1, inp2_unstable=1, delta_unstable=256
- layer_idx=8, type=LayerType.Linear, total=10, inp1_unstable=0, inp2_unstable=0, delta_unstable=10

Time for backsubstitution: 1.19 seconds

### RS candidates at layer 3
type: RSZ, layer: 3, pos: 73
type: RSZ, layer: 3, pos: 81
type: RSZ, layer: 3, pos: 136

Time for candidate selection: 0.12 seconds

### Candidate
type: RSZ, layer: 3, pos: 73

### Candidate
type: RSZ, layer: 3, pos: 81

### Relational analysis RSZ of RS_RSZ1_RSZ1_RSZ1_RSZ2_RSZ1_RSZ1_RSZ1_RSZ1

#### Relational analysis RSZ result of RS_RSZ1_RSZ1_RSZ1_RSZ2_RSZ1_RSZ1_RSZ1_RSZ1
Status: Status.VERIFIED
Output dim: 1, lower bound: -0.0002482, upper bound: 0.0002466
time: 0.55 seconds

### Relational analysis RSZ of RS_RSZ1_RSZ1_RSZ1_RSZ2_RSZ1_RSZ1_RSZ1_RSZ2

#### Relational analysis RSZ result of RS_RSZ1_RSZ1_RSZ1_RSZ2_RSZ1_RSZ1_RSZ1_RSZ2
Status: Status.VERIFIED
Output dim: 1, lower bound: -0.0002482, upper bound: 0.0002466
time: 0.56 seconds

## BFS RS instance: RS_RSZ1_RSZ1_RSZ1_RSZ2_RSZ1_RSZ1_RSZ2

### Backsubstitution after applying RS history:
0: 0.0009457, 0.0011003, 0.0009457, 0.0011003, -0.0001488, 0.0001487
1: 0.9936197, 0.9940771, 0.9936197, 0.9940771, -0.0003838, 0.0003857
2: -0.0065606, -0.0050208, -0.0065606, -0.0050208, -0.0011606, 0.0011813
3: 0.0038004, 0.0040140, 0.0038004, 0.0040140, -0.0001641, 0.0001627
4: 0.0023852, 0.0036022, 0.0023852, 0.0036022, -0.0010384, 0.0010220
5: 0.0060086, 0.0065111, 0.0060086, 0.0065111, -0.0005024, 0.0005024
6: -0.0013983, -0.0008639, -0.0013983, -0.0008639, -0.0003963, 0.0004035
7: -0.0083221, -0.0079387, -0.0083221, -0.0079387, -0.0003835, 0.0003835
8: 0.0048484, 0.0068715, 0.0048484, 0.0068715, -0.0012095, 0.0011823
9: -0.0036877, -0.0032079, -0.0036877, -0.0032079, -0.0004797, 0.0004797

### Unstable ReLU Count (Linear/Conv2D Layers)
- layer_idx=0, type=LayerType.Linear, total=256, inp1_unstable=5, inp2_unstable=5, delta_unstable=251
- layer_idx=2, type=LayerType.Linear, total=256, inp1_unstable=4, inp2_unstable=4, delta_unstable=255
- layer_idx=4, type=LayerType.Linear, total=256, inp1_unstable=0, inp2_unstable=0, delta_unstable=256
- layer_idx=6, type=LayerType.Linear, total=256, inp1_unstable=1, inp2_unstable=1, delta_unstable=256
- layer_idx=8, type=LayerType.Linear, total=10, inp1_unstable=0, inp2_unstable=0, delta_unstable=10

Time for backsubstitution: 1.24 seconds

### RS candidates at layer 3
type: RSZ, layer: 3, pos: 73
type: RSZ, layer: 3, pos: 81
type: RSZ, layer: 3, pos: 136

Time for candidate selection: 0.13 seconds

### Candidate
type: RSZ, layer: 3, pos: 73

### Candidate
type: RSZ, layer: 3, pos: 81

### Relational analysis RSZ of RS_RSZ1_RSZ1_RSZ1_RSZ2_RSZ1_RSZ1_RSZ2_RSZ1

#### Relational analysis RSZ result of RS_RSZ1_RSZ1_RSZ1_RSZ2_RSZ1_RSZ1_RSZ2_RSZ1
Status: Status.VERIFIED
Output dim: 1, lower bound: -0.0002463, upper bound: 0.0002496
time: 0.56 seconds

### Relational analysis RSZ of RS_RSZ1_RSZ1_RSZ1_RSZ2_RSZ1_RSZ1_RSZ2_RSZ2

#### Relational analysis RSZ result of RS_RSZ1_RSZ1_RSZ1_RSZ2_RSZ1_RSZ1_RSZ2_RSZ2
Status: Status.VERIFIED
Output dim: 1, lower bound: -0.0002463, upper bound: 0.0002496
time: 0.57 seconds

## BFS RS instance: RS_RSZ1_RSZ1_RSZ1_RSZ2_RSZ1_RSZ2_RSZ1

### Backsubstitution after applying RS history:
0: 0.0009457, 0.0011003, 0.0009457, 0.0011003, -0.0001486, 0.0001488
1: 0.9936197, 0.9940771, 0.9936197, 0.9940771, -0.0003853, 0.0003833
2: -0.0065606, -0.0050208, -0.0065606, -0.0050208, -0.0011768, 0.0011556
3: 0.0038004, 0.0040140, 0.0038004, 0.0040140, -0.0001624, 0.0001638
4: 0.0023852, 0.0036022, 0.0023852, 0.0036022, -0.0010181, 0.0010348
5: 0.0060086, 0.0065111, 0.0060086, 0.0065111, -0.0005024, 0.0005024
6: -0.0013983, -0.0008639, -0.0013983, -0.0008639, -0.0004020, 0.0003946
7: -0.0083221, -0.0079387, -0.0083221, -0.0079387, -0.0003835, 0.0003835
8: 0.0048484, 0.0068715, 0.0048484, 0.0068715, -0.0011757, 0.0012035
9: -0.0036877, -0.0032079, -0.0036877, -0.0032079, -0.0004797, 0.0004797

### Unstable ReLU Count (Linear/Conv2D Layers)
- layer_idx=0, type=LayerType.Linear, total=256, inp1_unstable=5, inp2_unstable=5, delta_unstable=251
- layer_idx=2, type=LayerType.Linear, total=256, inp1_unstable=4, inp2_unstable=4, delta_unstable=255
- layer_idx=4, type=LayerType.Linear, total=256, inp1_unstable=0, inp2_unstable=0, delta_unstable=256
- layer_idx=6, type=LayerType.Linear, total=256, inp1_unstable=1, inp2_unstable=1, delta_unstable=256
- layer_idx=8, type=LayerType.Linear, total=10, inp1_unstable=0, inp2_unstable=0, delta_unstable=10

Time for backsubstitution: 1.21 seconds

### RS candidates at layer 3
type: RSZ, layer: 3, pos: 73
type: RSZ, layer: 3, pos: 81
type: RSZ, layer: 3, pos: 136

Time for candidate selection: 0.12 seconds

### Candidate
type: RSZ, layer: 3, pos: 73

### Candidate
type: RSZ, layer: 3, pos: 81

### Relational analysis RSZ of RS_RSZ1_RSZ1_RSZ1_RSZ2_RSZ1_RSZ2_RSZ1_RSZ1

#### Relational analysis RSZ result of RS_RSZ1_RSZ1_RSZ1_RSZ2_RSZ1_RSZ2_RSZ1_RSZ1
Status: Status.VERIFIED
Output dim: 1, lower bound: -0.0002482, upper bound: 0.0002466
time: 0.55 seconds

### Relational analysis RSZ of RS_RSZ1_RSZ1_RSZ1_RSZ2_RSZ1_RSZ2_RSZ1_RSZ2

#### Relational analysis RSZ result of RS_RSZ1_RSZ1_RSZ1_RSZ2_RSZ1_RSZ2_RSZ1_RSZ2
Status: Status.VERIFIED
Output dim: 1, lower bound: -0.0002482, upper bound: 0.0002466
time: 0.55 seconds

## BFS RS instance: RS_RSZ1_RSZ1_RSZ1_RSZ2_RSZ1_RSZ2_RSZ2

### Backsubstitution after applying RS history:
0: 0.0009457, 0.0011003, 0.0009457, 0.0011003, -0.0001488, 0.0001487
1: 0.9936197, 0.9940771, 0.9936197, 0.9940771, -0.0003838, 0.0003857
2: -0.0065606, -0.0050208, -0.0065606, -0.0050208, -0.0011606, 0.0011813
3: 0.0038004, 0.0040140, 0.0038004, 0.0040140, -0.0001641, 0.0001627
4: 0.0023852, 0.0036022, 0.0023852, 0.0036022, -0.0010384, 0.0010220
5: 0.0060086, 0.0065111, 0.0060086, 0.0065111, -0.0005024, 0.0005024
6: -0.0013983, -0.0008639, -0.0013983, -0.0008639, -0.0003963, 0.0004035
7: -0.0083221, -0.0079387, -0.0083221, -0.0079387, -0.0003835, 0.0003835
8: 0.0048484, 0.0068715, 0.0048484, 0.0068715, -0.0012095, 0.0011823
9: -0.0036877, -0.0032079, -0.0036877, -0.0032079, -0.0004797, 0.0004797

### Unstable ReLU Count (Linear/Conv2D Layers)
- layer_idx=0, type=LayerType.Linear, total=256, inp1_unstable=5, inp2_unstable=5, delta_unstable=251
- layer_idx=2, type=LayerType.Linear, total=256, inp1_unstable=4, inp2_unstable=4, delta_unstable=255
- layer_idx=4, type=LayerType.Linear, total=256, inp1_unstable=0, inp2_unstable=0, delta_unstable=256
- layer_idx=6, type=LayerType.Linear, total=256, inp1_unstable=1, inp2_unstable=1, delta_unstable=256
- layer_idx=8, type=LayerType.Linear, total=10, inp1_unstable=0, inp2_unstable=0, delta_unstable=10

Time for backsubstitution: 1.24 seconds

### RS candidates at layer 3
type: RSZ, layer: 3, pos: 73
type: RSZ, layer: 3, pos: 81
type: RSZ, layer: 3, pos: 136

Time for candidate selection: 0.16 seconds

### Candidate
type: RSZ, layer: 3, pos: 73

### Candidate
type: RSZ, layer: 3, pos: 81

### Relational analysis RSZ of RS_RSZ1_RSZ1_RSZ1_RSZ2_RSZ1_RSZ2_RSZ2_RSZ1

#### Relational analysis RSZ result of RS_RSZ1_RSZ1_RSZ1_RSZ2_RSZ1_RSZ2_RSZ2_RSZ1
Status: Status.VERIFIED
Output dim: 1, lower bound: -0.0002463, upper bound: 0.0002496
time: 0.59 seconds

### Relational analysis RSZ of RS_RSZ1_RSZ1_RSZ1_RSZ2_RSZ1_RSZ2_RSZ2_RSZ2

#### Relational analysis RSZ result of RS_RSZ1_RSZ1_RSZ1_RSZ2_RSZ1_RSZ2_RSZ2_RSZ2
Status: Status.VERIFIED
Output dim: 1, lower bound: -0.0002463, upper bound: 0.0002496
time: 0.60 seconds

## BFS RS instance: RS_RSZ1_RSZ1_RSZ1_RSZ2_RSZ2_RSZ1_RSZ1

### Backsubstitution after applying RS history:
0: 0.0009457, 0.0011003, 0.0009457, 0.0011003, -0.0001486, 0.0001487
1: 0.9936197, 0.9940771, 0.9936197, 0.9940771, -0.0003849, 0.0003835
2: -0.0065606, -0.0050208, -0.0065606, -0.0050208, -0.0011726, 0.0011581
3: 0.0038004, 0.0040140, 0.0038004, 0.0040140, -0.0001625, 0.0001635
4: 0.0023852, 0.0036022, 0.0023852, 0.0036022, -0.0010201, 0.0010315
5: 0.0060086, 0.0065111, 0.0060086, 0.0065111, -0.0005024, 0.0005024
6: -0.0013983, -0.0008639, -0.0013983, -0.0008639, -0.0004005, 0.0003955
7: -0.0083221, -0.0079387, -0.0083221, -0.0079387, -0.0003835, 0.0003835
8: 0.0048484, 0.0068715, 0.0048484, 0.0068715, -0.0011790, 0.0011981
9: -0.0036877, -0.0032079, -0.0036877, -0.0032079, -0.0004797, 0.0004797

### Unstable ReLU Count (Linear/Conv2D Layers)
- layer_idx=0, type=LayerType.Linear, total=256, inp1_unstable=5, inp2_unstable=5, delta_unstable=251
- layer_idx=2, type=LayerType.Linear, total=256, inp1_unstable=4, inp2_unstable=4, delta_unstable=255
- layer_idx=4, type=LayerType.Linear, total=256, inp1_unstable=0, inp2_unstable=0, delta_unstable=256
- layer_idx=6, type=LayerType.Linear, total=256, inp1_unstable=1, inp2_unstable=1, delta_unstable=256
- layer_idx=8, type=LayerType.Linear, total=10, inp1_unstable=0, inp2_unstable=0, delta_unstable=10

Time for backsubstitution: 1.19 seconds

### RS candidates at layer 3
type: RSZ, layer: 3, pos: 73
type: RSZ, layer: 3, pos: 81
type: RSZ, layer: 3, pos: 136

Time for candidate selection: 0.12 seconds

### Candidate
type: RSZ, layer: 3, pos: 73

### Candidate
type: RSZ, layer: 3, pos: 81

### Relational analysis RSZ of RS_RSZ1_RSZ1_RSZ1_RSZ2_RSZ2_RSZ1_RSZ1_RSZ1

#### Relational analysis RSZ result of RS_RSZ1_RSZ1_RSZ1_RSZ2_RSZ2_RSZ1_RSZ1_RSZ1
Status: Status.VERIFIED
Output dim: 1, lower bound: -0.0002481, upper bound: 0.0002467
time: 0.60 seconds

### Relational analysis RSZ of RS_RSZ1_RSZ1_RSZ1_RSZ2_RSZ2_RSZ1_RSZ1_RSZ2

#### Relational analysis RSZ result of RS_RSZ1_RSZ1_RSZ1_RSZ2_RSZ2_RSZ1_RSZ1_RSZ2
Status: Status.VERIFIED
Output dim: 1, lower bound: -0.0002481, upper bound: 0.0002467
time: 0.59 seconds

## BFS RS instance: RS_RSZ1_RSZ1_RSZ1_RSZ2_RSZ2_RSZ1_RSZ2

### Backsubstitution after applying RS history:
0: 0.0009457, 0.0011003, 0.0009457, 0.0011003, -0.0001488, 0.0001486
1: 0.9936197, 0.9940771, 0.9936197, 0.9940771, -0.0003836, 0.0003861
2: -0.0065606, -0.0050208, -0.0065606, -0.0050208, -0.0011590, 0.0011850
3: 0.0038004, 0.0040140, 0.0038004, 0.0040140, -0.0001643, 0.0001626
4: 0.0023852, 0.0036022, 0.0023852, 0.0036022, -0.0010413, 0.0010208
5: 0.0060086, 0.0065111, 0.0060086, 0.0065111, -0.0005024, 0.0005024
6: -0.0013983, -0.0008639, -0.0013983, -0.0008639, -0.0003958, 0.0004048
7: -0.0083221, -0.0079387, -0.0083221, -0.0079387, -0.0003835, 0.0003835
8: 0.0048484, 0.0068715, 0.0048484, 0.0068715, -0.0012143, 0.0011802
9: -0.0036877, -0.0032079, -0.0036877, -0.0032079, -0.0004797, 0.0004797

### Unstable ReLU Count (Linear/Conv2D Layers)
- layer_idx=0, type=LayerType.Linear, total=256, inp1_unstable=5, inp2_unstable=5, delta_unstable=251
- layer_idx=2, type=LayerType.Linear, total=256, inp1_unstable=4, inp2_unstable=4, delta_unstable=255
- layer_idx=4, type=LayerType.Linear, total=256, inp1_unstable=0, inp2_unstable=0, delta_unstable=256
- layer_idx=6, type=LayerType.Linear, total=256, inp1_unstable=1, inp2_unstable=1, delta_unstable=256
- layer_idx=8, type=LayerType.Linear, total=10, inp1_unstable=0, inp2_unstable=0, delta_unstable=10

Time for backsubstitution: 1.20 seconds

### RS candidates at layer 3
type: RSZ, layer: 3, pos: 73
type: RSZ, layer: 3, pos: 81
type: RSZ, layer: 3, pos: 136

Time for candidate selection: 0.12 seconds

### Candidate
type: RSZ, layer: 3, pos: 73

### Candidate
type: RSZ, layer: 3, pos: 81

### Relational analysis RSZ of RS_RSZ1_RSZ1_RSZ1_RSZ2_RSZ2_RSZ1_RSZ2_RSZ1

#### Relational analysis RSZ result of RS_RSZ1_RSZ1_RSZ1_RSZ2_RSZ2_RSZ1_RSZ2_RSZ1
Status: Status.VERIFIED
Output dim: 1, lower bound: -0.0002463, upper bound: 0.0002496
time: 0.59 seconds

### Relational analysis RSZ of RS_RSZ1_RSZ1_RSZ1_RSZ2_RSZ2_RSZ1_RSZ2_RSZ2

#### Relational analysis RSZ result of RS_RSZ1_RSZ1_RSZ1_RSZ2_RSZ2_RSZ1_RSZ2_RSZ2
Status: Status.VERIFIED
Output dim: 1, lower bound: -0.0002463, upper bound: 0.0002496
time: 0.60 seconds

## BFS RS instance: RS_RSZ1_RSZ1_RSZ1_RSZ2_RSZ2_RSZ2_RSZ1

### Backsubstitution after applying RS history:
0: 0.0009457, 0.0011003, 0.0009457, 0.0011003, -0.0001486, 0.0001487
1: 0.9936197, 0.9940771, 0.9936197, 0.9940771, -0.0003849, 0.0003835
2: -0.0065606, -0.0050208, -0.0065606, -0.0050208, -0.0011726, 0.0011581
3: 0.0038004, 0.0040140, 0.0038004, 0.0040140, -0.0001625, 0.0001635
4: 0.0023852, 0.0036022, 0.0023852, 0.0036022, -0.0010201, 0.0010315
5: 0.0060086, 0.0065111, 0.0060086, 0.0065111, -0.0005024, 0.0005024
6: -0.0013983, -0.0008639, -0.0013983, -0.0008639, -0.0004005, 0.0003955
7: -0.0083221, -0.0079387, -0.0083221, -0.0079387, -0.0003835, 0.0003835
8: 0.0048484, 0.0068715, 0.0048484, 0.0068715, -0.0011790, 0.0011981
9: -0.0036877, -0.0032079, -0.0036877, -0.0032079, -0.0004797, 0.0004797

### Unstable ReLU Count (Linear/Conv2D Layers)
- layer_idx=0, type=LayerType.Linear, total=256, inp1_unstable=5, inp2_unstable=5, delta_unstable=251
- layer_idx=2, type=LayerType.Linear, total=256, inp1_unstable=4, inp2_unstable=4, delta_unstable=255
- layer_idx=4, type=LayerType.Linear, total=256, inp1_unstable=0, inp2_unstable=0, delta_unstable=256
- layer_idx=6, type=LayerType.Linear, total=256, inp1_unstable=1, inp2_unstable=1, delta_unstable=256
- layer_idx=8, type=LayerType.Linear, total=10, inp1_unstable=0, inp2_unstable=0, delta_unstable=10

Time for backsubstitution: 1.24 seconds

### RS candidates at layer 3
type: RSZ, layer: 3, pos: 73
type: RSZ, layer: 3, pos: 81
type: RSZ, layer: 3, pos: 136

Time for candidate selection: 0.12 seconds

### Candidate
type: RSZ, layer: 3, pos: 73

### Candidate
type: RSZ, layer: 3, pos: 81

### Relational analysis RSZ of RS_RSZ1_RSZ1_RSZ1_RSZ2_RSZ2_RSZ2_RSZ1_RSZ1

#### Relational analysis RSZ result of RS_RSZ1_RSZ1_RSZ1_RSZ2_RSZ2_RSZ2_RSZ1_RSZ1
Status: Status.VERIFIED
Output dim: 1, lower bound: -0.0002481, upper bound: 0.0002467
time: 0.66 seconds

### Relational analysis RSZ of RS_RSZ1_RSZ1_RSZ1_RSZ2_RSZ2_RSZ2_RSZ1_RSZ2

#### Relational analysis RSZ result of RS_RSZ1_RSZ1_RSZ1_RSZ2_RSZ2_RSZ2_RSZ1_RSZ2
Status: Status.VERIFIED
Output dim: 1, lower bound: -0.0002481, upper bound: 0.0002467
time: 0.66 seconds

## BFS RS instance: RS_RSZ1_RSZ1_RSZ1_RSZ2_RSZ2_RSZ2_RSZ2

### Backsubstitution after applying RS history:
0: 0.0009457, 0.0011003, 0.0009457, 0.0011003, -0.0001488, 0.0001486
1: 0.9936197, 0.9940771, 0.9936197, 0.9940771, -0.0003836, 0.0003861
2: -0.0065606, -0.0050208, -0.0065606, -0.0050208, -0.0011590, 0.0011850
3: 0.0038004, 0.0040140, 0.0038004, 0.0040140, -0.0001643, 0.0001626
4: 0.0023852, 0.0036022, 0.0023852, 0.0036022, -0.0010413, 0.0010208
5: 0.0060086, 0.0065111, 0.0060086, 0.0065111, -0.0005024, 0.0005024
6: -0.0013983, -0.0008639, -0.0013983, -0.0008639, -0.0003958, 0.0004048
7: -0.0083221, -0.0079387, -0.0083221, -0.0079387, -0.0003835, 0.0003835
8: 0.0048484, 0.0068715, 0.0048484, 0.0068715, -0.0012143, 0.0011802
9: -0.0036877, -0.0032079, -0.0036877, -0.0032079, -0.0004797, 0.0004797

### Unstable ReLU Count (Linear/Conv2D Layers)
- layer_idx=0, type=LayerType.Linear, total=256, inp1_unstable=5, inp2_unstable=5, delta_unstable=251
- layer_idx=2, type=LayerType.Linear, total=256, inp1_unstable=4, inp2_unstable=4, delta_unstable=255
- layer_idx=4, type=LayerType.Linear, total=256, inp1_unstable=0, inp2_unstable=0, delta_unstable=256
- layer_idx=6, type=LayerType.Linear, total=256, inp1_unstable=1, inp2_unstable=1, delta_unstable=256
- layer_idx=8, type=LayerType.Linear, total=10, inp1_unstable=0, inp2_unstable=0, delta_unstable=10

Time for backsubstitution: 1.22 seconds

### RS candidates at layer 3
type: RSZ, layer: 3, pos: 73
type: RSZ, layer: 3, pos: 81
type: RSZ, layer: 3, pos: 136

Time for candidate selection: 0.12 seconds

### Candidate
type: RSZ, layer: 3, pos: 73

### Candidate
type: RSZ, layer: 3, pos: 81

### Relational analysis RSZ of RS_RSZ1_RSZ1_RSZ1_RSZ2_RSZ2_RSZ2_RSZ2_RSZ1

#### Relational analysis RSZ result of RS_RSZ1_RSZ1_RSZ1_RSZ2_RSZ2_RSZ2_RSZ2_RSZ1
Status: Status.VERIFIED
Output dim: 1, lower bound: -0.0002463, upper bound: 0.0002496
time: 0.58 seconds

### Relational analysis RSZ of RS_RSZ1_RSZ1_RSZ1_RSZ2_RSZ2_RSZ2_RSZ2_RSZ2

#### Relational analysis RSZ result of RS_RSZ1_RSZ1_RSZ1_RSZ2_RSZ2_RSZ2_RSZ2_RSZ2
Status: Status.VERIFIED
Output dim: 1, lower bound: -0.0002463, upper bound: 0.0002496
time: 0.58 seconds

## BFS RS instance: RS_RSZ1_RSZ1_RSZ2_RSZ1_RSZ1_RSZ1_RSZ1

### Backsubstitution after applying RS history:
0: 0.0009457, 0.0011003, 0.0009457, 0.0011003, -0.0001486, 0.0001489
1: 0.9936197, 0.9940771, 0.9936197, 0.9940771, -0.0003866, 0.0003832
2: -0.0065606, -0.0050208, -0.0065606, -0.0050208, -0.0011904, 0.0011544
3: 0.0038004, 0.0040140, 0.0038004, 0.0040140, -0.0001623, 0.0001647
4: 0.0023852, 0.0036022, 0.0023852, 0.0036022, -0.0010171, 0.0010456
5: 0.0060086, 0.0065111, 0.0060086, 0.0065111, -0.0005024, 0.0005024
6: -0.0013983, -0.0008639, -0.0013983, -0.0008639, -0.0004067, 0.0003942
7: -0.0083221, -0.0079387, -0.0083221, -0.0079387, -0.0003835, 0.0003835
8: 0.0048484, 0.0068715, 0.0048484, 0.0068715, -0.0011741, 0.0012214
9: -0.0036877, -0.0032079, -0.0036877, -0.0032079, -0.0004797, 0.0004797

### Unstable ReLU Count (Linear/Conv2D Layers)
- layer_idx=0, type=LayerType.Linear, total=256, inp1_unstable=5, inp2_unstable=5, delta_unstable=251
- layer_idx=2, type=LayerType.Linear, total=256, inp1_unstable=4, inp2_unstable=4, delta_unstable=255
- layer_idx=4, type=LayerType.Linear, total=256, inp1_unstable=0, inp2_unstable=0, delta_unstable=256
- layer_idx=6, type=LayerType.Linear, total=256, inp1_unstable=1, inp2_unstable=1, delta_unstable=256
- layer_idx=8, type=LayerType.Linear, total=10, inp1_unstable=0, inp2_unstable=0, delta_unstable=10

Time for backsubstitution: 1.20 seconds

### RS candidates at layer 3
type: RSZ, layer: 3, pos: 73
type: RSZ, layer: 3, pos: 81
type: RSZ, layer: 3, pos: 136

Time for candidate selection: 0.13 seconds

### Candidate
type: RSZ, layer: 3, pos: 73

### Candidate
type: RSZ, layer: 3, pos: 81

### Relational analysis RSZ of RS_RSZ1_RSZ1_RSZ2_RSZ1_RSZ1_RSZ1_RSZ1_RSZ1

#### Relational analysis RSZ result of RS_RSZ1_RSZ1_RSZ2_RSZ1_RSZ1_RSZ1_RSZ1_RSZ1
Status: Status.VERIFIED
Output dim: 1, lower bound: -0.0002495, upper bound: 0.0002463
time: 0.59 seconds

### Relational analysis RSZ of RS_RSZ1_RSZ1_RSZ2_RSZ1_RSZ1_RSZ1_RSZ1_RSZ2

#### Relational analysis RSZ result of RS_RSZ1_RSZ1_RSZ2_RSZ1_RSZ1_RSZ1_RSZ1_RSZ2
Status: Status.VERIFIED
Output dim: 1, lower bound: -0.0002495, upper bound: 0.0002463
time: 0.58 seconds

## BFS RS instance: RS_RSZ1_RSZ1_RSZ2_RSZ1_RSZ1_RSZ1_RSZ2

### Backsubstitution after applying RS history:
0: 0.0009457, 0.0011003, 0.0009457, 0.0011003, -0.0001487, 0.0001487
1: 0.9936197, 0.9940771, 0.9936197, 0.9940771, -0.0003840, 0.0003846
2: -0.0065606, -0.0050208, -0.0065606, -0.0050208, -0.0011626, 0.0011694
3: 0.0038004, 0.0040140, 0.0038004, 0.0040140, -0.0001633, 0.0001628
4: 0.0023852, 0.0036022, 0.0023852, 0.0036022, -0.0010290, 0.0010237
5: 0.0060086, 0.0065111, 0.0060086, 0.0065111, -0.0005024, 0.0005024
6: -0.0013983, -0.0008639, -0.0013983, -0.0008639, -0.0003971, 0.0003994
7: -0.0083221, -0.0079387, -0.0083221, -0.0079387, -0.0003835, 0.0003835
8: 0.0048484, 0.0068715, 0.0048484, 0.0068715, -0.0011938, 0.0011850
9: -0.0036877, -0.0032079, -0.0036877, -0.0032079, -0.0004797, 0.0004797

### Unstable ReLU Count (Linear/Conv2D Layers)
- layer_idx=0, type=LayerType.Linear, total=256, inp1_unstable=5, inp2_unstable=5, delta_unstable=251
- layer_idx=2, type=LayerType.Linear, total=256, inp1_unstable=4, inp2_unstable=4, delta_unstable=255
- layer_idx=4, type=LayerType.Linear, total=256, inp1_unstable=0, inp2_unstable=0, delta_unstable=256
- layer_idx=6, type=LayerType.Linear, total=256, inp1_unstable=1, inp2_unstable=1, delta_unstable=256
- layer_idx=8, type=LayerType.Linear, total=10, inp1_unstable=0, inp2_unstable=0, delta_unstable=10

Time for backsubstitution: 1.20 seconds

### RS candidates at layer 3
type: RSZ, layer: 3, pos: 73
type: RSZ, layer: 3, pos: 81
type: RSZ, layer: 3, pos: 136

Time for candidate selection: 0.12 seconds

### Candidate
type: RSZ, layer: 3, pos: 73

### Candidate
type: RSZ, layer: 3, pos: 81

### Relational analysis RSZ of RS_RSZ1_RSZ1_RSZ2_RSZ1_RSZ1_RSZ1_RSZ2_RSZ1

#### Relational analysis RSZ result of RS_RSZ1_RSZ1_RSZ2_RSZ1_RSZ1_RSZ1_RSZ2_RSZ1
Status: Status.VERIFIED
Output dim: 1, lower bound: -0.0002467, upper bound: 0.0002486
time: 0.56 seconds

### Relational analysis RSZ of RS_RSZ1_RSZ1_RSZ2_RSZ1_RSZ1_RSZ1_RSZ2_RSZ2

#### Relational analysis RSZ result of RS_RSZ1_RSZ1_RSZ2_RSZ1_RSZ1_RSZ1_RSZ2_RSZ2
Status: Status.VERIFIED
Output dim: 1, lower bound: -0.0002467, upper bound: 0.0002486
time: 0.58 seconds

## BFS RS instance: RS_RSZ1_RSZ1_RSZ2_RSZ1_RSZ1_RSZ2_RSZ1

### Backsubstitution after applying RS history:
0: 0.0009457, 0.0011003, 0.0009457, 0.0011003, -0.0001486, 0.0001489
1: 0.9936197, 0.9940771, 0.9936197, 0.9940771, -0.0003866, 0.0003832
2: -0.0065606, -0.0050208, -0.0065606, -0.0050208, -0.0011904, 0.0011544
3: 0.0038004, 0.0040140, 0.0038004, 0.0040140, -0.0001623, 0.0001647
4: 0.0023852, 0.0036022, 0.0023852, 0.0036022, -0.0010171, 0.0010456
5: 0.0060086, 0.0065111, 0.0060086, 0.0065111, -0.0005024, 0.0005024
6: -0.0013983, -0.0008639, -0.0013983, -0.0008639, -0.0004067, 0.0003942
7: -0.0083221, -0.0079387, -0.0083221, -0.0079387, -0.0003835, 0.0003835
8: 0.0048484, 0.0068715, 0.0048484, 0.0068715, -0.0011741, 0.0012214
9: -0.0036877, -0.0032079, -0.0036877, -0.0032079, -0.0004797, 0.0004797

### Unstable ReLU Count (Linear/Conv2D Layers)
- layer_idx=0, type=LayerType.Linear, total=256, inp1_unstable=5, inp2_unstable=5, delta_unstable=251
- layer_idx=2, type=LayerType.Linear, total=256, inp1_unstable=4, inp2_unstable=4, delta_unstable=255
- layer_idx=4, type=LayerType.Linear, total=256, inp1_unstable=0, inp2_unstable=0, delta_unstable=256
- layer_idx=6, type=LayerType.Linear, total=256, inp1_unstable=1, inp2_unstable=1, delta_unstable=256
- layer_idx=8, type=LayerType.Linear, total=10, inp1_unstable=0, inp2_unstable=0, delta_unstable=10

Time for backsubstitution: 1.22 seconds

### RS candidates at layer 3
type: RSZ, layer: 3, pos: 73
type: RSZ, layer: 3, pos: 81
type: RSZ, layer: 3, pos: 136

Time for candidate selection: 0.12 seconds

### Candidate
type: RSZ, layer: 3, pos: 73

### Candidate
type: RSZ, layer: 3, pos: 81

### Relational analysis RSZ of RS_RSZ1_RSZ1_RSZ2_RSZ1_RSZ1_RSZ2_RSZ1_RSZ1

#### Relational analysis RSZ result of RS_RSZ1_RSZ1_RSZ2_RSZ1_RSZ1_RSZ2_RSZ1_RSZ1
Status: Status.VERIFIED
Output dim: 1, lower bound: -0.0002495, upper bound: 0.0002463
time: 0.57 seconds

### Relational analysis RSZ of RS_RSZ1_RSZ1_RSZ2_RSZ1_RSZ1_RSZ2_RSZ1_RSZ2

#### Relational analysis RSZ result of RS_RSZ1_RSZ1_RSZ2_RSZ1_RSZ1_RSZ2_RSZ1_RSZ2
Status: Status.VERIFIED
Output dim: 1, lower bound: -0.0002495, upper bound: 0.0002463
time: 0.57 seconds

## BFS RS instance: RS_RSZ1_RSZ1_RSZ2_RSZ1_RSZ1_RSZ2_RSZ2

### Backsubstitution after applying RS history:
0: 0.0009457, 0.0011003, 0.0009457, 0.0011003, -0.0001487, 0.0001487
1: 0.9936197, 0.9940771, 0.9936197, 0.9940771, -0.0003840, 0.0003846
2: -0.0065606, -0.0050208, -0.0065606, -0.0050208, -0.0011626, 0.0011694
3: 0.0038004, 0.0040140, 0.0038004, 0.0040140, -0.0001633, 0.0001628
4: 0.0023852, 0.0036022, 0.0023852, 0.0036022, -0.0010290, 0.0010237
5: 0.0060086, 0.0065111, 0.0060086, 0.0065111, -0.0005024, 0.0005024
6: -0.0013983, -0.0008639, -0.0013983, -0.0008639, -0.0003971, 0.0003994
7: -0.0083221, -0.0079387, -0.0083221, -0.0079387, -0.0003835, 0.0003835
8: 0.0048484, 0.0068715, 0.0048484, 0.0068715, -0.0011938, 0.0011850
9: -0.0036877, -0.0032079, -0.0036877, -0.0032079, -0.0004797, 0.0004797

### Unstable ReLU Count (Linear/Conv2D Layers)
- layer_idx=0, type=LayerType.Linear, total=256, inp1_unstable=5, inp2_unstable=5, delta_unstable=251
- layer_idx=2, type=LayerType.Linear, total=256, inp1_unstable=4, inp2_unstable=4, delta_unstable=255
- layer_idx=4, type=LayerType.Linear, total=256, inp1_unstable=0, inp2_unstable=0, delta_unstable=256
- layer_idx=6, type=LayerType.Linear, total=256, inp1_unstable=1, inp2_unstable=1, delta_unstable=256
- layer_idx=8, type=LayerType.Linear, total=10, inp1_unstable=0, inp2_unstable=0, delta_unstable=10

Time for backsubstitution: 1.20 seconds

### RS candidates at layer 3
type: RSZ, layer: 3, pos: 73
type: RSZ, layer: 3, pos: 81
type: RSZ, layer: 3, pos: 136

Time for candidate selection: 0.12 seconds

### Candidate
type: RSZ, layer: 3, pos: 73

### Candidate
type: RSZ, layer: 3, pos: 81

### Relational analysis RSZ of RS_RSZ1_RSZ1_RSZ2_RSZ1_RSZ1_RSZ2_RSZ2_RSZ1

#### Relational analysis RSZ result of RS_RSZ1_RSZ1_RSZ2_RSZ1_RSZ1_RSZ2_RSZ2_RSZ1
Status: Status.VERIFIED
Output dim: 1, lower bound: -0.0002467, upper bound: 0.0002486
time: 0.58 seconds

### Relational analysis RSZ of RS_RSZ1_RSZ1_RSZ2_RSZ1_RSZ1_RSZ2_RSZ2_RSZ2

#### Relational analysis RSZ result of RS_RSZ1_RSZ1_RSZ2_RSZ1_RSZ1_RSZ2_RSZ2_RSZ2
Status: Status.VERIFIED
Output dim: 1, lower bound: -0.0002467, upper bound: 0.0002486
time: 0.59 seconds

## BFS RS instance: RS_RSZ1_RSZ1_RSZ2_RSZ1_RSZ2_RSZ1_RSZ1

### Backsubstitution after applying RS history:
0: 0.0009457, 0.0011003, 0.0009457, 0.0011003, -0.0001486, 0.0001489
1: 0.9936197, 0.9940771, 0.9936197, 0.9940771, -0.0003862, 0.0003833
2: -0.0065606, -0.0050208, -0.0065606, -0.0050208, -0.0011865, 0.0011558
3: 0.0038004, 0.0040140, 0.0038004, 0.0040140, -0.0001624, 0.0001644
4: 0.0023852, 0.0036022, 0.0023852, 0.0036022, -0.0010183, 0.0010425
5: 0.0060086, 0.0065111, 0.0060086, 0.0065111, -0.0005024, 0.0005024
6: -0.0013983, -0.0008639, -0.0013983, -0.0008639, -0.0004053, 0.0003947
7: -0.0083221, -0.0079387, -0.0083221, -0.0079387, -0.0003835, 0.0003835
8: 0.0048484, 0.0068715, 0.0048484, 0.0068715, -0.0011760, 0.0012163
9: -0.0036877, -0.0032079, -0.0036877, -0.0032079, -0.0004797, 0.0004797

### Unstable ReLU Count (Linear/Conv2D Layers)
- layer_idx=0, type=LayerType.Linear, total=256, inp1_unstable=5, inp2_unstable=5, delta_unstable=251
- layer_idx=2, type=LayerType.Linear, total=256, inp1_unstable=4, inp2_unstable=4, delta_unstable=255
- layer_idx=4, type=LayerType.Linear, total=256, inp1_unstable=0, inp2_unstable=0, delta_unstable=256
- layer_idx=6, type=LayerType.Linear, total=256, inp1_unstable=1, inp2_unstable=1, delta_unstable=256
- layer_idx=8, type=LayerType.Linear, total=10, inp1_unstable=0, inp2_unstable=0, delta_unstable=10

Time for backsubstitution: 1.22 seconds

### RS candidates at layer 3
type: RSZ, layer: 3, pos: 73
type: RSZ, layer: 3, pos: 81
type: RSZ, layer: 3, pos: 136

Time for candidate selection: 0.13 seconds

### Candidate
type: RSZ, layer: 3, pos: 73

### Candidate
type: RSZ, layer: 3, pos: 81

### Relational analysis RSZ of RS_RSZ1_RSZ1_RSZ2_RSZ1_RSZ2_RSZ1_RSZ1_RSZ1

#### Relational analysis RSZ result of RS_RSZ1_RSZ1_RSZ2_RSZ1_RSZ2_RSZ1_RSZ1_RSZ1
Status: Status.VERIFIED
Output dim: 1, lower bound: -0.0002495, upper bound: 0.0002463
time: 0.58 seconds

### Relational analysis RSZ of RS_RSZ1_RSZ1_RSZ2_RSZ1_RSZ2_RSZ1_RSZ1_RSZ2

#### Relational analysis RSZ result of RS_RSZ1_RSZ1_RSZ2_RSZ1_RSZ2_RSZ1_RSZ1_RSZ2
Status: Status.VERIFIED
Output dim: 1, lower bound: -0.0002495, upper bound: 0.0002463
time: 0.57 seconds

## BFS RS instance: RS_RSZ1_RSZ1_RSZ2_RSZ1_RSZ2_RSZ1_RSZ2

### Backsubstitution after applying RS history:
0: 0.0009457, 0.0011003, 0.0009457, 0.0011003, -0.0001488, 0.0001486
1: 0.9936197, 0.9940771, 0.9936197, 0.9940771, -0.0003837, 0.0003850
2: -0.0065606, -0.0050208, -0.0065606, -0.0050208, -0.0011597, 0.0011732
3: 0.0038004, 0.0040140, 0.0038004, 0.0040140, -0.0001635, 0.0001626
4: 0.0023852, 0.0036022, 0.0023852, 0.0036022, -0.0010320, 0.0010213
5: 0.0060086, 0.0065111, 0.0060086, 0.0065111, -0.0005024, 0.0005024
6: -0.0013983, -0.0008639, -0.0013983, -0.0008639, -0.0003960, 0.0004007
7: -0.0083221, -0.0079387, -0.0083221, -0.0079387, -0.0003835, 0.0003835
8: 0.0048484, 0.0068715, 0.0048484, 0.0068715, -0.0011989, 0.0011811
9: -0.0036877, -0.0032079, -0.0036877, -0.0032079, -0.0004797, 0.0004797

### Unstable ReLU Count (Linear/Conv2D Layers)
- layer_idx=0, type=LayerType.Linear, total=256, inp1_unstable=5, inp2_unstable=5, delta_unstable=251
- layer_idx=2, type=LayerType.Linear, total=256, inp1_unstable=4, inp2_unstable=4, delta_unstable=255
- layer_idx=4, type=LayerType.Linear, total=256, inp1_unstable=0, inp2_unstable=0, delta_unstable=256
- layer_idx=6, type=LayerType.Linear, total=256, inp1_unstable=1, inp2_unstable=1, delta_unstable=256
- layer_idx=8, type=LayerType.Linear, total=10, inp1_unstable=0, inp2_unstable=0, delta_unstable=10

Time for backsubstitution: 1.20 seconds

### RS candidates at layer 3
type: RSZ, layer: 3, pos: 73
type: RSZ, layer: 3, pos: 81
type: RSZ, layer: 3, pos: 136

Time for candidate selection: 0.13 seconds

### Candidate
type: RSZ, layer: 3, pos: 73

### Candidate
type: RSZ, layer: 3, pos: 81

### Relational analysis RSZ of RS_RSZ1_RSZ1_RSZ2_RSZ1_RSZ2_RSZ1_RSZ2_RSZ1

#### Relational analysis RSZ result of RS_RSZ1_RSZ1_RSZ2_RSZ1_RSZ2_RSZ1_RSZ2_RSZ1
Status: Status.VERIFIED
Output dim: 1, lower bound: -0.0002466, upper bound: 0.0002486
time: 0.58 seconds

### Relational analysis RSZ of RS_RSZ1_RSZ1_RSZ2_RSZ1_RSZ2_RSZ1_RSZ2_RSZ2

#### Relational analysis RSZ result of RS_RSZ1_RSZ1_RSZ2_RSZ1_RSZ2_RSZ1_RSZ2_RSZ2
Status: Status.VERIFIED
Output dim: 1, lower bound: -0.0002466, upper bound: 0.0002486
time: 0.58 seconds

## BFS RS instance: RS_RSZ1_RSZ1_RSZ2_RSZ1_RSZ2_RSZ2_RSZ1

### Backsubstitution after applying RS history:
0: 0.0009457, 0.0011003, 0.0009457, 0.0011003, -0.0001486, 0.0001489
1: 0.9936197, 0.9940771, 0.9936197, 0.9940771, -0.0003862, 0.0003833
2: -0.0065606, -0.0050208, -0.0065606, -0.0050208, -0.0011865, 0.0011558
3: 0.0038004, 0.0040140, 0.0038004, 0.0040140, -0.0001624, 0.0001644
4: 0.0023852, 0.0036022, 0.0023852, 0.0036022, -0.0010183, 0.0010425
5: 0.0060086, 0.0065111, 0.0060086, 0.0065111, -0.0005024, 0.0005024
6: -0.0013983, -0.0008639, -0.0013983, -0.0008639, -0.0004053, 0.0003947
7: -0.0083221, -0.0079387, -0.0083221, -0.0079387, -0.0003835, 0.0003835
8: 0.0048484, 0.0068715, 0.0048484, 0.0068715, -0.0011760, 0.0012163
9: -0.0036877, -0.0032079, -0.0036877, -0.0032079, -0.0004797, 0.0004797

### Unstable ReLU Count (Linear/Conv2D Layers)
- layer_idx=0, type=LayerType.Linear, total=256, inp1_unstable=5, inp2_unstable=5, delta_unstable=251
- layer_idx=2, type=LayerType.Linear, total=256, inp1_unstable=4, inp2_unstable=4, delta_unstable=255
- layer_idx=4, type=LayerType.Linear, total=256, inp1_unstable=0, inp2_unstable=0, delta_unstable=256
- layer_idx=6, type=LayerType.Linear, total=256, inp1_unstable=1, inp2_unstable=1, delta_unstable=256
- layer_idx=8, type=LayerType.Linear, total=10, inp1_unstable=0, inp2_unstable=0, delta_unstable=10

Time for backsubstitution: 1.23 seconds

### RS candidates at layer 3
type: RSZ, layer: 3, pos: 73
type: RSZ, layer: 3, pos: 81
type: RSZ, layer: 3, pos: 136

Time for candidate selection: 0.13 seconds

### Candidate
type: RSZ, layer: 3, pos: 73

### Candidate
type: RSZ, layer: 3, pos: 81

### Relational analysis RSZ of RS_RSZ1_RSZ1_RSZ2_RSZ1_RSZ2_RSZ2_RSZ1_RSZ1

#### Relational analysis RSZ result of RS_RSZ1_RSZ1_RSZ2_RSZ1_RSZ2_RSZ2_RSZ1_RSZ1
Status: Status.VERIFIED
Output dim: 1, lower bound: -0.0002495, upper bound: 0.0002463
time: 0.58 seconds

### Relational analysis RSZ of RS_RSZ1_RSZ1_RSZ2_RSZ1_RSZ2_RSZ2_RSZ1_RSZ2

#### Relational analysis RSZ result of RS_RSZ1_RSZ1_RSZ2_RSZ1_RSZ2_RSZ2_RSZ1_RSZ2
Status: Status.VERIFIED
Output dim: 1, lower bound: -0.0002495, upper bound: 0.0002463
time: 0.59 seconds

## BFS RS instance: RS_RSZ1_RSZ1_RSZ2_RSZ1_RSZ2_RSZ2_RSZ2

### Backsubstitution after applying RS history:
0: 0.0009457, 0.0011003, 0.0009457, 0.0011003, -0.0001488, 0.0001486
1: 0.9936197, 0.9940771, 0.9936197, 0.9940771, -0.0003837, 0.0003850
2: -0.0065606, -0.0050208, -0.0065606, -0.0050208, -0.0011597, 0.0011732
3: 0.0038004, 0.0040140, 0.0038004, 0.0040140, -0.0001635, 0.0001626
4: 0.0023852, 0.0036022, 0.0023852, 0.0036022, -0.0010320, 0.0010213
5: 0.0060086, 0.0065111, 0.0060086, 0.0065111, -0.0005024, 0.0005024
6: -0.0013983, -0.0008639, -0.0013983, -0.0008639, -0.0003960, 0.0004007
7: -0.0083221, -0.0079387, -0.0083221, -0.0079387, -0.0003835, 0.0003835
8: 0.0048484, 0.0068715, 0.0048484, 0.0068715, -0.0011989, 0.0011811
9: -0.0036877, -0.0032079, -0.0036877, -0.0032079, -0.0004797, 0.0004797

### Unstable ReLU Count (Linear/Conv2D Layers)
- layer_idx=0, type=LayerType.Linear, total=256, inp1_unstable=5, inp2_unstable=5, delta_unstable=251
- layer_idx=2, type=LayerType.Linear, total=256, inp1_unstable=4, inp2_unstable=4, delta_unstable=255
- layer_idx=4, type=LayerType.Linear, total=256, inp1_unstable=0, inp2_unstable=0, delta_unstable=256
- layer_idx=6, type=LayerType.Linear, total=256, inp1_unstable=1, inp2_unstable=1, delta_unstable=256
- layer_idx=8, type=LayerType.Linear, total=10, inp1_unstable=0, inp2_unstable=0, delta_unstable=10

Time for backsubstitution: 1.25 seconds

### RS candidates at layer 3
type: RSZ, layer: 3, pos: 73
type: RSZ, layer: 3, pos: 81
type: RSZ, layer: 3, pos: 136

Time for candidate selection: 0.13 seconds

### Candidate
type: RSZ, layer: 3, pos: 73

### Candidate
type: RSZ, layer: 3, pos: 81

### Relational analysis RSZ of RS_RSZ1_RSZ1_RSZ2_RSZ1_RSZ2_RSZ2_RSZ2_RSZ1

#### Relational analysis RSZ result of RS_RSZ1_RSZ1_RSZ2_RSZ1_RSZ2_RSZ2_RSZ2_RSZ1
Status: Status.VERIFIED
Output dim: 1, lower bound: -0.0002466, upper bound: 0.0002486
time: 0.61 seconds

### Relational analysis RSZ of RS_RSZ1_RSZ1_RSZ2_RSZ1_RSZ2_RSZ2_RSZ2_RSZ2

#### Relational analysis RSZ result of RS_RSZ1_RSZ1_RSZ2_RSZ1_RSZ2_RSZ2_RSZ2_RSZ2
Status: Status.VERIFIED
Output dim: 1, lower bound: -0.0002466, upper bound: 0.0002486
time: 0.61 seconds

## BFS RS instance: RS_RSZ1_RSZ1_RSZ2_RSZ2_RSZ1_RSZ1_RSZ1

### Backsubstitution after applying RS history:
0: 0.0009457, 0.0011003, 0.0009457, 0.0011003, -0.0001486, 0.0001488
1: 0.9936197, 0.9940771, 0.9936197, 0.9940771, -0.0003860, 0.0003828
2: -0.0065606, -0.0050208, -0.0065606, -0.0050208, -0.0011842, 0.0011502
3: 0.0038004, 0.0040140, 0.0038004, 0.0040140, -0.0001620, 0.0001643
4: 0.0023852, 0.0036022, 0.0023852, 0.0036022, -0.0010138, 0.0010407
5: 0.0060086, 0.0065111, 0.0060086, 0.0065111, -0.0005024, 0.0005024
6: -0.0013983, -0.0008639, -0.0013983, -0.0008639, -0.0004045, 0.0003927
7: -0.0083221, -0.0079387, -0.0083221, -0.0079387, -0.0003835, 0.0003835
8: 0.0048484, 0.0068715, 0.0048484, 0.0068715, -0.0011686, 0.0012132
9: -0.0036877, -0.0032079, -0.0036877, -0.0032079, -0.0004797, 0.0004797

### Unstable ReLU Count (Linear/Conv2D Layers)
- layer_idx=0, type=LayerType.Linear, total=256, inp1_unstable=5, inp2_unstable=5, delta_unstable=251
- layer_idx=2, type=LayerType.Linear, total=256, inp1_unstable=4, inp2_unstable=4, delta_unstable=255
- layer_idx=4, type=LayerType.Linear, total=256, inp1_unstable=0, inp2_unstable=0, delta_unstable=256
- layer_idx=6, type=LayerType.Linear, total=256, inp1_unstable=1, inp2_unstable=1, delta_unstable=256
- layer_idx=8, type=LayerType.Linear, total=10, inp1_unstable=0, inp2_unstable=0, delta_unstable=10

Time for backsubstitution: 1.22 seconds

### RS candidates at layer 3
type: RSZ, layer: 3, pos: 73
type: RSZ, layer: 3, pos: 81
type: RSZ, layer: 3, pos: 136

Time for candidate selection: 0.12 seconds

### Candidate
type: RSZ, layer: 3, pos: 73

### Candidate
type: RSZ, layer: 3, pos: 81

### Relational analysis RSZ of RS_RSZ1_RSZ1_RSZ2_RSZ2_RSZ1_RSZ1_RSZ1_RSZ1

#### Relational analysis RSZ result of RS_RSZ1_RSZ1_RSZ2_RSZ2_RSZ1_RSZ1_RSZ1_RSZ1
Status: Status.VERIFIED
Output dim: 1, lower bound: -0.0002486, upper bound: 0.0002468
time: 0.57 seconds

### Relational analysis RSZ of RS_RSZ1_RSZ1_RSZ2_RSZ2_RSZ1_RSZ1_RSZ1_RSZ2

#### Relational analysis RSZ result of RS_RSZ1_RSZ1_RSZ2_RSZ2_RSZ1_RSZ1_RSZ1_RSZ2
Status: Status.VERIFIED
Output dim: 1, lower bound: -0.0002486, upper bound: 0.0002468
time: 0.57 seconds

## BFS RS instance: RS_RSZ1_RSZ1_RSZ2_RSZ2_RSZ1_RSZ1_RSZ2

### Backsubstitution after applying RS history:
0: 0.0009457, 0.0011003, 0.0009457, 0.0011003, -0.0001487, 0.0001487
1: 0.9936197, 0.9940771, 0.9936197, 0.9940771, -0.0003838, 0.0003849
2: -0.0065606, -0.0050208, -0.0065606, -0.0050208, -0.0011606, 0.0011724
3: 0.0038004, 0.0040140, 0.0038004, 0.0040140, -0.0001635, 0.0001627
4: 0.0023852, 0.0036022, 0.0023852, 0.0036022, -0.0010314, 0.0010221
5: 0.0060086, 0.0065111, 0.0060086, 0.0065111, -0.0005024, 0.0005024
6: -0.0013983, -0.0008639, -0.0013983, -0.0008639, -0.0003964, 0.0004004
7: -0.0083221, -0.0079387, -0.0083221, -0.0079387, -0.0003835, 0.0003835
8: 0.0048484, 0.0068715, 0.0048484, 0.0068715, -0.0011978, 0.0011823
9: -0.0036877, -0.0032079, -0.0036877, -0.0032079, -0.0004797, 0.0004797

### Unstable ReLU Count (Linear/Conv2D Layers)
- layer_idx=0, type=LayerType.Linear, total=256, inp1_unstable=5, inp2_unstable=5, delta_unstable=251
- layer_idx=2, type=LayerType.Linear, total=256, inp1_unstable=4, inp2_unstable=4, delta_unstable=255
- layer_idx=4, type=LayerType.Linear, total=256, inp1_unstable=0, inp2_unstable=0, delta_unstable=256
- layer_idx=6, type=LayerType.Linear, total=256, inp1_unstable=1, inp2_unstable=1, delta_unstable=256
- layer_idx=8, type=LayerType.Linear, total=10, inp1_unstable=0, inp2_unstable=0, delta_unstable=10

Time for backsubstitution: 1.33 seconds

### RS candidates at layer 3
type: RSZ, layer: 3, pos: 73
type: RSZ, layer: 3, pos: 81
type: RSZ, layer: 3, pos: 136

Time for candidate selection: 0.16 seconds

### Candidate
type: RSZ, layer: 3, pos: 73

### Candidate
type: RSZ, layer: 3, pos: 81

### Relational analysis RSZ of RS_RSZ1_RSZ1_RSZ2_RSZ2_RSZ1_RSZ1_RSZ2_RSZ1

#### Relational analysis RSZ result of RS_RSZ1_RSZ1_RSZ2_RSZ2_RSZ1_RSZ1_RSZ2_RSZ1
Status: Status.VERIFIED
Output dim: 1, lower bound: -0.0002462, upper bound: 0.0002493
time: 0.61 seconds

### Relational analysis RSZ of RS_RSZ1_RSZ1_RSZ2_RSZ2_RSZ1_RSZ1_RSZ2_RSZ2

#### Relational analysis RSZ result of RS_RSZ1_RSZ1_RSZ2_RSZ2_RSZ1_RSZ1_RSZ2_RSZ2
Status: Status.VERIFIED
Output dim: 1, lower bound: -0.0002462, upper bound: 0.0002493
time: 0.57 seconds

## BFS RS instance: RS_RSZ1_RSZ1_RSZ2_RSZ2_RSZ1_RSZ2_RSZ1

### Backsubstitution after applying RS history:
0: 0.0009457, 0.0011003, 0.0009457, 0.0011003, -0.0001486, 0.0001488
1: 0.9936197, 0.9940771, 0.9936197, 0.9940771, -0.0003860, 0.0003828
2: -0.0065606, -0.0050208, -0.0065606, -0.0050208, -0.0011842, 0.0011502
3: 0.0038004, 0.0040140, 0.0038004, 0.0040140, -0.0001620, 0.0001643
4: 0.0023852, 0.0036022, 0.0023852, 0.0036022, -0.0010138, 0.0010407
5: 0.0060086, 0.0065111, 0.0060086, 0.0065111, -0.0005024, 0.0005024
6: -0.0013983, -0.0008639, -0.0013983, -0.0008639, -0.0004045, 0.0003927
7: -0.0083221, -0.0079387, -0.0083221, -0.0079387, -0.0003835, 0.0003835
8: 0.0048484, 0.0068715, 0.0048484, 0.0068715, -0.0011686, 0.0012132
9: -0.0036877, -0.0032079, -0.0036877, -0.0032079, -0.0004797, 0.0004797

### Unstable ReLU Count (Linear/Conv2D Layers)
- layer_idx=0, type=LayerType.Linear, total=256, inp1_unstable=5, inp2_unstable=5, delta_unstable=251
- layer_idx=2, type=LayerType.Linear, total=256, inp1_unstable=4, inp2_unstable=4, delta_unstable=255
- layer_idx=4, type=LayerType.Linear, total=256, inp1_unstable=0, inp2_unstable=0, delta_unstable=256
- layer_idx=6, type=LayerType.Linear, total=256, inp1_unstable=1, inp2_unstable=1, delta_unstable=256
- layer_idx=8, type=LayerType.Linear, total=10, inp1_unstable=0, inp2_unstable=0, delta_unstable=10

Time for backsubstitution: 1.22 seconds

### RS candidates at layer 3
type: RSZ, layer: 3, pos: 73
type: RSZ, layer: 3, pos: 81
type: RSZ, layer: 3, pos: 136

Time for candidate selection: 0.13 seconds

### Candidate
type: RSZ, layer: 3, pos: 73

### Candidate
type: RSZ, layer: 3, pos: 81

### Relational analysis RSZ of RS_RSZ1_RSZ1_RSZ2_RSZ2_RSZ1_RSZ2_RSZ1_RSZ1

#### Relational analysis RSZ result of RS_RSZ1_RSZ1_RSZ2_RSZ2_RSZ1_RSZ2_RSZ1_RSZ1
Status: Status.VERIFIED
Output dim: 1, lower bound: -0.0002486, upper bound: 0.0002468
time: 0.55 seconds

### Relational analysis RSZ of RS_RSZ1_RSZ1_RSZ2_RSZ2_RSZ1_RSZ2_RSZ1_RSZ2

#### Relational analysis RSZ result of RS_RSZ1_RSZ1_RSZ2_RSZ2_RSZ1_RSZ2_RSZ1_RSZ2
Status: Status.VERIFIED
Output dim: 1, lower bound: -0.0002486, upper bound: 0.0002468
time: 0.55 seconds

## BFS RS instance: RS_RSZ1_RSZ1_RSZ2_RSZ2_RSZ1_RSZ2_RSZ2

### Backsubstitution after applying RS history:
0: 0.0009457, 0.0011003, 0.0009457, 0.0011003, -0.0001487, 0.0001487
1: 0.9936197, 0.9940771, 0.9936197, 0.9940771, -0.0003838, 0.0003849
2: -0.0065606, -0.0050208, -0.0065606, -0.0050208, -0.0011606, 0.0011724
3: 0.0038004, 0.0040140, 0.0038004, 0.0040140, -0.0001635, 0.0001627
4: 0.0023852, 0.0036022, 0.0023852, 0.0036022, -0.0010314, 0.0010221
5: 0.0060086, 0.0065111, 0.0060086, 0.0065111, -0.0005024, 0.0005024
6: -0.0013983, -0.0008639, -0.0013983, -0.0008639, -0.0003964, 0.0004004
7: -0.0083221, -0.0079387, -0.0083221, -0.0079387, -0.0003835, 0.0003835
8: 0.0048484, 0.0068715, 0.0048484, 0.0068715, -0.0011978, 0.0011823
9: -0.0036877, -0.0032079, -0.0036877, -0.0032079, -0.0004797, 0.0004797

### Unstable ReLU Count (Linear/Conv2D Layers)
- layer_idx=0, type=LayerType.Linear, total=256, inp1_unstable=5, inp2_unstable=5, delta_unstable=251
- layer_idx=2, type=LayerType.Linear, total=256, inp1_unstable=4, inp2_unstable=4, delta_unstable=255
- layer_idx=4, type=LayerType.Linear, total=256, inp1_unstable=0, inp2_unstable=0, delta_unstable=256
- layer_idx=6, type=LayerType.Linear, total=256, inp1_unstable=1, inp2_unstable=1, delta_unstable=256
- layer_idx=8, type=LayerType.Linear, total=10, inp1_unstable=0, inp2_unstable=0, delta_unstable=10

Time for backsubstitution: 1.24 seconds

### RS candidates at layer 3
type: RSZ, layer: 3, pos: 73
type: RSZ, layer: 3, pos: 81
type: RSZ, layer: 3, pos: 136

Time for candidate selection: 0.15 seconds

### Candidate
type: RSZ, layer: 3, pos: 73

### Candidate
type: RSZ, layer: 3, pos: 81

### Relational analysis RSZ of RS_RSZ1_RSZ1_RSZ2_RSZ2_RSZ1_RSZ2_RSZ2_RSZ1

#### Relational analysis RSZ result of RS_RSZ1_RSZ1_RSZ2_RSZ2_RSZ1_RSZ2_RSZ2_RSZ1
Status: Status.VERIFIED
Output dim: 1, lower bound: -0.0002462, upper bound: 0.0002493
time: 0.59 seconds

### Relational analysis RSZ of RS_RSZ1_RSZ1_RSZ2_RSZ2_RSZ1_RSZ2_RSZ2_RSZ2

#### Relational analysis RSZ result of RS_RSZ1_RSZ1_RSZ2_RSZ2_RSZ1_RSZ2_RSZ2_RSZ2
Status: Status.VERIFIED
Output dim: 1, lower bound: -0.0002462, upper bound: 0.0002493
time: 0.58 seconds

## BFS RS instance: RS_RSZ1_RSZ1_RSZ2_RSZ2_RSZ2_RSZ1_RSZ1

### Backsubstitution after applying RS history:
0: 0.0009457, 0.0011003, 0.0009457, 0.0011003, -0.0001486, 0.0001488
1: 0.9936197, 0.9940771, 0.9936197, 0.9940771, -0.0003856, 0.0003830
2: -0.0065606, -0.0050208, -0.0065606, -0.0050208, -0.0011799, 0.0011526
3: 0.0038004, 0.0040140, 0.0038004, 0.0040140, -0.0001622, 0.0001640
4: 0.0023852, 0.0036022, 0.0023852, 0.0036022, -0.0010157, 0.0010373
5: 0.0060086, 0.0065111, 0.0060086, 0.0065111, -0.0005024, 0.0005024
6: -0.0013983, -0.0008639, -0.0013983, -0.0008639, -0.0004030, 0.0003936
7: -0.0083221, -0.0079387, -0.0083221, -0.0079387, -0.0003835, 0.0003835
8: 0.0048484, 0.0068715, 0.0048484, 0.0068715, -0.0011717, 0.0012076
9: -0.0036877, -0.0032079, -0.0036877, -0.0032079, -0.0004797, 0.0004797

### Unstable ReLU Count (Linear/Conv2D Layers)
- layer_idx=0, type=LayerType.Linear, total=256, inp1_unstable=5, inp2_unstable=5, delta_unstable=251
- layer_idx=2, type=LayerType.Linear, total=256, inp1_unstable=4, inp2_unstable=4, delta_unstable=255
- layer_idx=4, type=LayerType.Linear, total=256, inp1_unstable=0, inp2_unstable=0, delta_unstable=256
- layer_idx=6, type=LayerType.Linear, total=256, inp1_unstable=1, inp2_unstable=1, delta_unstable=256
- layer_idx=8, type=LayerType.Linear, total=10, inp1_unstable=0, inp2_unstable=0, delta_unstable=10

Time for backsubstitution: 1.26 seconds

### RS candidates at layer 3
type: RSZ, layer: 3, pos: 73
type: RSZ, layer: 3, pos: 81
type: RSZ, layer: 3, pos: 136

Time for candidate selection: 0.14 seconds

### Candidate
type: RSZ, layer: 3, pos: 73

### Candidate
type: RSZ, layer: 3, pos: 81

### Relational analysis RSZ of RS_RSZ1_RSZ1_RSZ2_RSZ2_RSZ2_RSZ1_RSZ1_RSZ1

#### Relational analysis RSZ result of RS_RSZ1_RSZ1_RSZ2_RSZ2_RSZ2_RSZ1_RSZ1_RSZ1
Status: Status.VERIFIED
Output dim: 1, lower bound: -0.0002485, upper bound: 0.0002468
time: 0.60 seconds

### Relational analysis RSZ of RS_RSZ1_RSZ1_RSZ2_RSZ2_RSZ2_RSZ1_RSZ1_RSZ2

#### Relational analysis RSZ result of RS_RSZ1_RSZ1_RSZ2_RSZ2_RSZ2_RSZ1_RSZ1_RSZ2
Status: Status.VERIFIED
Output dim: 1, lower bound: -0.0002485, upper bound: 0.0002468
time: 0.60 seconds

## BFS RS instance: RS_RSZ1_RSZ1_RSZ2_RSZ2_RSZ2_RSZ1_RSZ2

### Backsubstitution after applying RS history:
0: 0.0009457, 0.0011003, 0.0009457, 0.0011003, -0.0001488, 0.0001486
1: 0.9936197, 0.9940771, 0.9936197, 0.9940771, -0.0003836, 0.0003852
2: -0.0065606, -0.0050208, -0.0065606, -0.0050208, -0.0011594, 0.0011759
3: 0.0038004, 0.0040140, 0.0038004, 0.0040140, -0.0001637, 0.0001626
4: 0.0023852, 0.0036022, 0.0023852, 0.0036022, -0.0010341, 0.0010211
5: 0.0060086, 0.0065111, 0.0060086, 0.0065111, -0.0005024, 0.0005024
6: -0.0013983, -0.0008639, -0.0013983, -0.0008639, -0.0003959, 0.0004016
7: -0.0083221, -0.0079387, -0.0083221, -0.0079387, -0.0003835, 0.0003835
8: 0.0048484, 0.0068715, 0.0048484, 0.0068715, -0.0012023, 0.0011807
9: -0.0036877, -0.0032079, -0.0036877, -0.0032079, -0.0004797, 0.0004797

### Unstable ReLU Count (Linear/Conv2D Layers)
- layer_idx=0, type=LayerType.Linear, total=256, inp1_unstable=5, inp2_unstable=5, delta_unstable=251
- layer_idx=2, type=LayerType.Linear, total=256, inp1_unstable=4, inp2_unstable=4, delta_unstable=255
- layer_idx=4, type=LayerType.Linear, total=256, inp1_unstable=0, inp2_unstable=0, delta_unstable=256
- layer_idx=6, type=LayerType.Linear, total=256, inp1_unstable=1, inp2_unstable=1, delta_unstable=256
- layer_idx=8, type=LayerType.Linear, total=10, inp1_unstable=0, inp2_unstable=0, delta_unstable=10

Time for backsubstitution: 1.23 seconds

### RS candidates at layer 3
type: RSZ, layer: 3, pos: 73
type: RSZ, layer: 3, pos: 81
type: RSZ, layer: 3, pos: 136

Time for candidate selection: 0.12 seconds

### Candidate
type: RSZ, layer: 3, pos: 73

### Candidate
type: RSZ, layer: 3, pos: 81

### Relational analysis RSZ of RS_RSZ1_RSZ1_RSZ2_RSZ2_RSZ2_RSZ1_RSZ2_RSZ1

#### Relational analysis RSZ result of RS_RSZ1_RSZ1_RSZ2_RSZ2_RSZ2_RSZ1_RSZ2_RSZ1
Status: Status.VERIFIED
Output dim: 1, lower bound: -0.0002462, upper bound: 0.0002493
time: 0.57 seconds

### Relational analysis RSZ of RS_RSZ1_RSZ1_RSZ2_RSZ2_RSZ2_RSZ1_RSZ2_RSZ2

#### Relational analysis RSZ result of RS_RSZ1_RSZ1_RSZ2_RSZ2_RSZ2_RSZ1_RSZ2_RSZ2
Status: Status.VERIFIED
Output dim: 1, lower bound: -0.0002462, upper bound: 0.0002493
time: 0.57 seconds

## BFS RS instance: RS_RSZ1_RSZ1_RSZ2_RSZ2_RSZ2_RSZ2_RSZ1

### Backsubstitution after applying RS history:
0: 0.0009457, 0.0011003, 0.0009457, 0.0011003, -0.0001486, 0.0001488
1: 0.9936197, 0.9940771, 0.9936197, 0.9940771, -0.0003856, 0.0003830
2: -0.0065606, -0.0050208, -0.0065606, -0.0050208, -0.0011799, 0.0011526
3: 0.0038004, 0.0040140, 0.0038004, 0.0040140, -0.0001622, 0.0001640
4: 0.0023852, 0.0036022, 0.0023852, 0.0036022, -0.0010157, 0.0010373
5: 0.0060086, 0.0065111, 0.0060086, 0.0065111, -0.0005024, 0.0005024
6: -0.0013983, -0.0008639, -0.0013983, -0.0008639, -0.0004030, 0.0003936
7: -0.0083221, -0.0079387, -0.0083221, -0.0079387, -0.0003835, 0.0003835
8: 0.0048484, 0.0068715, 0.0048484, 0.0068715, -0.0011717, 0.0012076
9: -0.0036877, -0.0032079, -0.0036877, -0.0032079, -0.0004797, 0.0004797

### Unstable ReLU Count (Linear/Conv2D Layers)
- layer_idx=0, type=LayerType.Linear, total=256, inp1_unstable=5, inp2_unstable=5, delta_unstable=251
- layer_idx=2, type=LayerType.Linear, total=256, inp1_unstable=4, inp2_unstable=4, delta_unstable=255
- layer_idx=4, type=LayerType.Linear, total=256, inp1_unstable=0, inp2_unstable=0, delta_unstable=256
- layer_idx=6, type=LayerType.Linear, total=256, inp1_unstable=1, inp2_unstable=1, delta_unstable=256
- layer_idx=8, type=LayerType.Linear, total=10, inp1_unstable=0, inp2_unstable=0, delta_unstable=10

Time for backsubstitution: 1.21 seconds

### RS candidates at layer 3
type: RSZ, layer: 3, pos: 73
type: RSZ, layer: 3, pos: 81
type: RSZ, layer: 3, pos: 136

Time for candidate selection: 0.13 seconds

### Candidate
type: RSZ, layer: 3, pos: 73

### Candidate
type: RSZ, layer: 3, pos: 81

### Relational analysis RSZ of RS_RSZ1_RSZ1_RSZ2_RSZ2_RSZ2_RSZ2_RSZ1_RSZ1

#### Relational analysis RSZ result of RS_RSZ1_RSZ1_RSZ2_RSZ2_RSZ2_RSZ2_RSZ1_RSZ1
Status: Status.VERIFIED
Output dim: 1, lower bound: -0.0002485, upper bound: 0.0002468
time: 0.56 seconds

### Relational analysis RSZ of RS_RSZ1_RSZ1_RSZ2_RSZ2_RSZ2_RSZ2_RSZ1_RSZ2

#### Relational analysis RSZ result of RS_RSZ1_RSZ1_RSZ2_RSZ2_RSZ2_RSZ2_RSZ1_RSZ2
Status: Status.VERIFIED
Output dim: 1, lower bound: -0.0002485, upper bound: 0.0002468
time: 0.56 seconds

## BFS RS instance: RS_RSZ1_RSZ1_RSZ2_RSZ2_RSZ2_RSZ2_RSZ2

### Backsubstitution after applying RS history:
0: 0.0009457, 0.0011003, 0.0009457, 0.0011003, -0.0001488, 0.0001486
1: 0.9936197, 0.9940771, 0.9936197, 0.9940771, -0.0003836, 0.0003852
2: -0.0065606, -0.0050208, -0.0065606, -0.0050208, -0.0011594, 0.0011759
3: 0.0038004, 0.0040140, 0.0038004, 0.0040140, -0.0001637, 0.0001626
4: 0.0023852, 0.0036022, 0.0023852, 0.0036022, -0.0010341, 0.0010211
5: 0.0060086, 0.0065111, 0.0060086, 0.0065111, -0.0005024, 0.0005024
6: -0.0013983, -0.0008639, -0.0013983, -0.0008639, -0.0003959, 0.0004016
7: -0.0083221, -0.0079387, -0.0083221, -0.0079387, -0.0003835, 0.0003835
8: 0.0048484, 0.0068715, 0.0048484, 0.0068715, -0.0012023, 0.0011807
9: -0.0036877, -0.0032079, -0.0036877, -0.0032079, -0.0004797, 0.0004797

### Unstable ReLU Count (Linear/Conv2D Layers)
- layer_idx=0, type=LayerType.Linear, total=256, inp1_unstable=5, inp2_unstable=5, delta_unstable=251
- layer_idx=2, type=LayerType.Linear, total=256, inp1_unstable=4, inp2_unstable=4, delta_unstable=255
- layer_idx=4, type=LayerType.Linear, total=256, inp1_unstable=0, inp2_unstable=0, delta_unstable=256
- layer_idx=6, type=LayerType.Linear, total=256, inp1_unstable=1, inp2_unstable=1, delta_unstable=256
- layer_idx=8, type=LayerType.Linear, total=10, inp1_unstable=0, inp2_unstable=0, delta_unstable=10

Time for backsubstitution: 1.22 seconds

### RS candidates at layer 3
type: RSZ, layer: 3, pos: 73
type: RSZ, layer: 3, pos: 81
type: RSZ, layer: 3, pos: 136

Time for candidate selection: 0.13 seconds

### Candidate
type: RSZ, layer: 3, pos: 73

### Candidate
type: RSZ, layer: 3, pos: 81

### Relational analysis RSZ of RS_RSZ1_RSZ1_RSZ2_RSZ2_RSZ2_RSZ2_RSZ2_RSZ1

#### Relational analysis RSZ result of RS_RSZ1_RSZ1_RSZ2_RSZ2_RSZ2_RSZ2_RSZ2_RSZ1
Status: Status.VERIFIED
Output dim: 1, lower bound: -0.0002462, upper bound: 0.0002493
time: 0.57 seconds

### Relational analysis RSZ of RS_RSZ1_RSZ1_RSZ2_RSZ2_RSZ2_RSZ2_RSZ2_RSZ2

#### Relational analysis RSZ result of RS_RSZ1_RSZ1_RSZ2_RSZ2_RSZ2_RSZ2_RSZ2_RSZ2
Status: Status.VERIFIED
Output dim: 1, lower bound: -0.0002462, upper bound: 0.0002493
time: 0.57 seconds

## BFS RS instance: RS_RSZ1_RSZ2_RSZ1_RSZ1_RSZ1_RSZ1_RSZ1

### Backsubstitution after applying RS history:
0: 0.0009457, 0.0011003, 0.0009457, 0.0011003, -0.0001486, 0.0001490
1: 0.9936197, 0.9940771, 0.9936197, 0.9940771, -0.0003882, 0.0003830
2: -0.0065606, -0.0050208, -0.0065606, -0.0050208, -0.0011981, 0.0011525
3: 0.0038004, 0.0040140, 0.0038004, 0.0040140, -0.0001622, 0.0001658
4: 0.0023852, 0.0036022, 0.0023852, 0.0036022, -0.0010157, 0.0010515
5: 0.0060086, 0.0065111, 0.0060086, 0.0065111, -0.0005024, 0.0005024
6: -0.0013983, -0.0008639, -0.0013983, -0.0008639, -0.0004094, 0.0003935
7: -0.0083221, -0.0079387, -0.0083221, -0.0079387, -0.0003835, 0.0003835
8: 0.0048484, 0.0068715, 0.0048484, 0.0068715, -0.0011717, 0.0012400
9: -0.0036877, -0.0032079, -0.0036877, -0.0032079, -0.0004797, 0.0004797

### Unstable ReLU Count (Linear/Conv2D Layers)
- layer_idx=0, type=LayerType.Linear, total=256, inp1_unstable=5, inp2_unstable=5, delta_unstable=251
- layer_idx=2, type=LayerType.Linear, total=256, inp1_unstable=4, inp2_unstable=4, delta_unstable=255
- layer_idx=4, type=LayerType.Linear, total=256, inp1_unstable=0, inp2_unstable=0, delta_unstable=256
- layer_idx=6, type=LayerType.Linear, total=256, inp1_unstable=1, inp2_unstable=1, delta_unstable=256
- layer_idx=8, type=LayerType.Linear, total=10, inp1_unstable=0, inp2_unstable=0, delta_unstable=10

Time for backsubstitution: 1.20 seconds

### RS candidates at layer 3
type: RSZ, layer: 3, pos: 73
type: RSZ, layer: 3, pos: 81
type: RSZ, layer: 3, pos: 136

Time for candidate selection: 0.12 seconds

### Candidate
type: RSZ, layer: 3, pos: 73

### Candidate
type: RSZ, layer: 3, pos: 81

### Relational analysis RSZ of RS_RSZ1_RSZ2_RSZ1_RSZ1_RSZ1_RSZ1_RSZ1_RSZ1

#### Relational analysis RSZ result of RS_RSZ1_RSZ2_RSZ1_RSZ1_RSZ1_RSZ1_RSZ1_RSZ1
Status: Status.VERIFIED
Output dim: 1, lower bound: -0.0002493, upper bound: 0.0002462
time: 0.62 seconds

### Relational analysis RSZ of RS_RSZ1_RSZ2_RSZ1_RSZ1_RSZ1_RSZ1_RSZ1_RSZ2

#### Relational analysis RSZ result of RS_RSZ1_RSZ2_RSZ1_RSZ1_RSZ1_RSZ1_RSZ1_RSZ2
Status: Status.VERIFIED
Output dim: 1, lower bound: -0.0002493, upper bound: 0.0002462
time: 0.62 seconds

## BFS RS instance: RS_RSZ1_RSZ2_RSZ1_RSZ1_RSZ1_RSZ1_RSZ2

### Backsubstitution after applying RS history:
0: 0.0009457, 0.0011003, 0.0009457, 0.0011003, -0.0001487, 0.0001488
1: 0.9936197, 0.9940771, 0.9936197, 0.9940771, -0.0003851, 0.0003848
2: -0.0065606, -0.0050208, -0.0065606, -0.0050208, -0.0011695, 0.0011716
3: 0.0038004, 0.0040140, 0.0038004, 0.0040140, -0.0001634, 0.0001636
4: 0.0023852, 0.0036022, 0.0023852, 0.0036022, -0.0010307, 0.0010291
5: 0.0060086, 0.0065111, 0.0060086, 0.0065111, -0.0005024, 0.0005024
6: -0.0013983, -0.0008639, -0.0013983, -0.0008639, -0.0003995, 0.0004002
7: -0.0083221, -0.0079387, -0.0083221, -0.0079387, -0.0003835, 0.0003835
8: 0.0048484, 0.0068715, 0.0048484, 0.0068715, -0.0011967, 0.0011989
9: -0.0036877, -0.0032079, -0.0036877, -0.0032079, -0.0004797, 0.0004797

### Unstable ReLU Count (Linear/Conv2D Layers)
- layer_idx=0, type=LayerType.Linear, total=256, inp1_unstable=5, inp2_unstable=5, delta_unstable=251
- layer_idx=2, type=LayerType.Linear, total=256, inp1_unstable=4, inp2_unstable=4, delta_unstable=255
- layer_idx=4, type=LayerType.Linear, total=256, inp1_unstable=0, inp2_unstable=0, delta_unstable=256
- layer_idx=6, type=LayerType.Linear, total=256, inp1_unstable=1, inp2_unstable=1, delta_unstable=256
- layer_idx=8, type=LayerType.Linear, total=10, inp1_unstable=0, inp2_unstable=0, delta_unstable=10

Time for backsubstitution: 1.23 seconds

### RS candidates at layer 3
type: RSZ, layer: 3, pos: 73
type: RSZ, layer: 3, pos: 81
type: RSZ, layer: 3, pos: 136

Time for candidate selection: 0.14 seconds

### Candidate
type: RSZ, layer: 3, pos: 73

### Candidate
type: RSZ, layer: 3, pos: 81

### Relational analysis RSZ of RS_RSZ1_RSZ2_RSZ1_RSZ1_RSZ1_RSZ1_RSZ2_RSZ1

#### Relational analysis RSZ result of RS_RSZ1_RSZ2_RSZ1_RSZ1_RSZ1_RSZ1_RSZ2_RSZ1
Status: Status.VERIFIED
Output dim: 1, lower bound: -0.0002468, upper bound: 0.0002485
time: 0.58 seconds

### Relational analysis RSZ of RS_RSZ1_RSZ2_RSZ1_RSZ1_RSZ1_RSZ1_RSZ2_RSZ2

#### Relational analysis RSZ result of RS_RSZ1_RSZ2_RSZ1_RSZ1_RSZ1_RSZ1_RSZ2_RSZ2
Status: Status.VERIFIED
Output dim: 1, lower bound: -0.0002468, upper bound: 0.0002485
time: 0.57 seconds

## BFS RS instance: RS_RSZ1_RSZ2_RSZ1_RSZ1_RSZ1_RSZ2_RSZ1

### Backsubstitution after applying RS history:
0: 0.0009457, 0.0011003, 0.0009457, 0.0011003, -0.0001486, 0.0001490
1: 0.9936197, 0.9940771, 0.9936197, 0.9940771, -0.0003882, 0.0003830
2: -0.0065606, -0.0050208, -0.0065606, -0.0050208, -0.0011981, 0.0011525
3: 0.0038004, 0.0040140, 0.0038004, 0.0040140, -0.0001622, 0.0001658
4: 0.0023852, 0.0036022, 0.0023852, 0.0036022, -0.0010157, 0.0010515
5: 0.0060086, 0.0065111, 0.0060086, 0.0065111, -0.0005024, 0.0005024
6: -0.0013983, -0.0008639, -0.0013983, -0.0008639, -0.0004094, 0.0003935
7: -0.0083221, -0.0079387, -0.0083221, -0.0079387, -0.0003835, 0.0003835
8: 0.0048484, 0.0068715, 0.0048484, 0.0068715, -0.0011717, 0.0012400
9: -0.0036877, -0.0032079, -0.0036877, -0.0032079, -0.0004797, 0.0004797

### Unstable ReLU Count (Linear/Conv2D Layers)
- layer_idx=0, type=LayerType.Linear, total=256, inp1_unstable=5, inp2_unstable=5, delta_unstable=251
- layer_idx=2, type=LayerType.Linear, total=256, inp1_unstable=4, inp2_unstable=4, delta_unstable=255
- layer_idx=4, type=LayerType.Linear, total=256, inp1_unstable=0, inp2_unstable=0, delta_unstable=256
- layer_idx=6, type=LayerType.Linear, total=256, inp1_unstable=1, inp2_unstable=1, delta_unstable=256
- layer_idx=8, type=LayerType.Linear, total=10, inp1_unstable=0, inp2_unstable=0, delta_unstable=10

Time for backsubstitution: 1.22 seconds

### RS candidates at layer 3
type: RSZ, layer: 3, pos: 73
type: RSZ, layer: 3, pos: 81
type: RSZ, layer: 3, pos: 136

Time for candidate selection: 0.12 seconds

### Candidate
type: RSZ, layer: 3, pos: 73

### Candidate
type: RSZ, layer: 3, pos: 81

### Relational analysis RSZ of RS_RSZ1_RSZ2_RSZ1_RSZ1_RSZ1_RSZ2_RSZ1_RSZ1

#### Relational analysis RSZ result of RS_RSZ1_RSZ2_RSZ1_RSZ1_RSZ1_RSZ2_RSZ1_RSZ1
Status: Status.VERIFIED
Output dim: 1, lower bound: -0.0002493, upper bound: 0.0002462
time: 0.59 seconds

### Relational analysis RSZ of RS_RSZ1_RSZ2_RSZ1_RSZ1_RSZ1_RSZ2_RSZ1_RSZ2

#### Relational analysis RSZ result of RS_RSZ1_RSZ2_RSZ1_RSZ1_RSZ1_RSZ2_RSZ1_RSZ2
Status: Status.VERIFIED
Output dim: 1, lower bound: -0.0002493, upper bound: 0.0002462
time: 0.60 seconds

## BFS RS instance: RS_RSZ1_RSZ2_RSZ1_RSZ1_RSZ1_RSZ2_RSZ2

### Backsubstitution after applying RS history:
0: 0.0009457, 0.0011003, 0.0009457, 0.0011003, -0.0001487, 0.0001488
1: 0.9936197, 0.9940771, 0.9936197, 0.9940771, -0.0003851, 0.0003848
2: -0.0065606, -0.0050208, -0.0065606, -0.0050208, -0.0011695, 0.0011716
3: 0.0038004, 0.0040140, 0.0038004, 0.0040140, -0.0001634, 0.0001636
4: 0.0023852, 0.0036022, 0.0023852, 0.0036022, -0.0010307, 0.0010291
5: 0.0060086, 0.0065111, 0.0060086, 0.0065111, -0.0005024, 0.0005024
6: -0.0013983, -0.0008639, -0.0013983, -0.0008639, -0.0003995, 0.0004002
7: -0.0083221, -0.0079387, -0.0083221, -0.0079387, -0.0003835, 0.0003835
8: 0.0048484, 0.0068715, 0.0048484, 0.0068715, -0.0011967, 0.0011989
9: -0.0036877, -0.0032079, -0.0036877, -0.0032079, -0.0004797, 0.0004797

### Unstable ReLU Count (Linear/Conv2D Layers)
- layer_idx=0, type=LayerType.Linear, total=256, inp1_unstable=5, inp2_unstable=5, delta_unstable=251
- layer_idx=2, type=LayerType.Linear, total=256, inp1_unstable=4, inp2_unstable=4, delta_unstable=255
- layer_idx=4, type=LayerType.Linear, total=256, inp1_unstable=0, inp2_unstable=0, delta_unstable=256
- layer_idx=6, type=LayerType.Linear, total=256, inp1_unstable=1, inp2_unstable=1, delta_unstable=256
- layer_idx=8, type=LayerType.Linear, total=10, inp1_unstable=0, inp2_unstable=0, delta_unstable=10

Time for backsubstitution: 1.23 seconds

### RS candidates at layer 3
type: RSZ, layer: 3, pos: 73
type: RSZ, layer: 3, pos: 81
type: RSZ, layer: 3, pos: 136

Time for candidate selection: 0.13 seconds

### Candidate
type: RSZ, layer: 3, pos: 73

### Candidate
type: RSZ, layer: 3, pos: 81

### Relational analysis RSZ of RS_RSZ1_RSZ2_RSZ1_RSZ1_RSZ1_RSZ2_RSZ2_RSZ1

#### Relational analysis RSZ result of RS_RSZ1_RSZ2_RSZ1_RSZ1_RSZ1_RSZ2_RSZ2_RSZ1
Status: Status.VERIFIED
Output dim: 1, lower bound: -0.0002468, upper bound: 0.0002485
time: 0.62 seconds

### Relational analysis RSZ of RS_RSZ1_RSZ2_RSZ1_RSZ1_RSZ1_RSZ2_RSZ2_RSZ2

#### Relational analysis RSZ result of RS_RSZ1_RSZ2_RSZ1_RSZ1_RSZ1_RSZ2_RSZ2_RSZ2
Status: Status.VERIFIED
Output dim: 1, lower bound: -0.0002468, upper bound: 0.0002485
time: 0.60 seconds

## BFS RS instance: RS_RSZ1_RSZ2_RSZ1_RSZ1_RSZ2_RSZ1_RSZ1

### Backsubstitution after applying RS history:
0: 0.0009457, 0.0011003, 0.0009457, 0.0011003, -0.0001486, 0.0001490
1: 0.9936197, 0.9940771, 0.9936197, 0.9940771, -0.0003879, 0.0003831
2: -0.0065606, -0.0050208, -0.0065606, -0.0050208, -0.0011945, 0.0011541
3: 0.0038004, 0.0040140, 0.0038004, 0.0040140, -0.0001623, 0.0001655
4: 0.0023852, 0.0036022, 0.0023852, 0.0036022, -0.0010169, 0.0010486
5: 0.0060086, 0.0065111, 0.0060086, 0.0065111, -0.0005024, 0.0005024
6: -0.0013983, -0.0008639, -0.0013983, -0.0008639, -0.0004081, 0.0003941
7: -0.0083221, -0.0079387, -0.0083221, -0.0079387, -0.0003835, 0.0003835
8: 0.0048484, 0.0068715, 0.0048484, 0.0068715, -0.0011737, 0.0012351
9: -0.0036877, -0.0032079, -0.0036877, -0.0032079, -0.0004797, 0.0004797

### Unstable ReLU Count (Linear/Conv2D Layers)
- layer_idx=0, type=LayerType.Linear, total=256, inp1_unstable=5, inp2_unstable=5, delta_unstable=251
- layer_idx=2, type=LayerType.Linear, total=256, inp1_unstable=4, inp2_unstable=4, delta_unstable=255
- layer_idx=4, type=LayerType.Linear, total=256, inp1_unstable=0, inp2_unstable=0, delta_unstable=256
- layer_idx=6, type=LayerType.Linear, total=256, inp1_unstable=1, inp2_unstable=1, delta_unstable=256
- layer_idx=8, type=LayerType.Linear, total=10, inp1_unstable=0, inp2_unstable=0, delta_unstable=10

Time for backsubstitution: 1.29 seconds

### RS candidates at layer 3
type: RSZ, layer: 3, pos: 73
type: RSZ, layer: 3, pos: 81
type: RSZ, layer: 3, pos: 136

Time for candidate selection: 0.12 seconds

### Candidate
type: RSZ, layer: 3, pos: 73

### Candidate
type: RSZ, layer: 3, pos: 81

### Relational analysis RSZ of RS_RSZ1_RSZ2_RSZ1_RSZ1_RSZ2_RSZ1_RSZ1_RSZ1

#### Relational analysis RSZ result of RS_RSZ1_RSZ2_RSZ1_RSZ1_RSZ2_RSZ1_RSZ1_RSZ1
Status: Status.VERIFIED
Output dim: 1, lower bound: -0.0002493, upper bound: 0.0002462
time: 0.62 seconds

### Relational analysis RSZ of RS_RSZ1_RSZ2_RSZ1_RSZ1_RSZ2_RSZ1_RSZ1_RSZ2

#### Relational analysis RSZ result of RS_RSZ1_RSZ2_RSZ1_RSZ1_RSZ2_RSZ1_RSZ1_RSZ2
Status: Status.VERIFIED
Output dim: 1, lower bound: -0.0002493, upper bound: 0.0002462
time: 0.60 seconds

## BFS RS instance: RS_RSZ1_RSZ2_RSZ1_RSZ1_RSZ2_RSZ1_RSZ2

### Backsubstitution after applying RS history:
0: 0.0009457, 0.0011003, 0.0009457, 0.0011003, -0.0001488, 0.0001487
1: 0.9936197, 0.9940771, 0.9936197, 0.9940771, -0.0003849, 0.0003852
2: -0.0065606, -0.0050208, -0.0065606, -0.0050208, -0.0011666, 0.0011758
3: 0.0038004, 0.0040140, 0.0038004, 0.0040140, -0.0001637, 0.0001635
4: 0.0023852, 0.0036022, 0.0023852, 0.0036022, -0.0010341, 0.0010268
5: 0.0060086, 0.0065111, 0.0060086, 0.0065111, -0.0005024, 0.0005024
6: -0.0013983, -0.0008639, -0.0013983, -0.0008639, -0.0003984, 0.0004016
7: -0.0083221, -0.0079387, -0.0083221, -0.0079387, -0.0003835, 0.0003835
8: 0.0048484, 0.0068715, 0.0048484, 0.0068715, -0.0012023, 0.0011951
9: -0.0036877, -0.0032079, -0.0036877, -0.0032079, -0.0004797, 0.0004797

### Unstable ReLU Count (Linear/Conv2D Layers)
- layer_idx=0, type=LayerType.Linear, total=256, inp1_unstable=5, inp2_unstable=5, delta_unstable=251
- layer_idx=2, type=LayerType.Linear, total=256, inp1_unstable=4, inp2_unstable=4, delta_unstable=255
- layer_idx=4, type=LayerType.Linear, total=256, inp1_unstable=0, inp2_unstable=0, delta_unstable=256
- layer_idx=6, type=LayerType.Linear, total=256, inp1_unstable=1, inp2_unstable=1, delta_unstable=256
- layer_idx=8, type=LayerType.Linear, total=10, inp1_unstable=0, inp2_unstable=0, delta_unstable=10

Time for backsubstitution: 1.21 seconds

### RS candidates at layer 3
type: RSZ, layer: 3, pos: 73
type: RSZ, layer: 3, pos: 81
type: RSZ, layer: 3, pos: 136

Time for candidate selection: 0.12 seconds

### Candidate
type: RSZ, layer: 3, pos: 73

### Candidate
type: RSZ, layer: 3, pos: 81

### Relational analysis RSZ of RS_RSZ1_RSZ2_RSZ1_RSZ1_RSZ2_RSZ1_RSZ2_RSZ1

#### Relational analysis RSZ result of RS_RSZ1_RSZ2_RSZ1_RSZ1_RSZ2_RSZ1_RSZ2_RSZ1
Status: Status.VERIFIED
Output dim: 1, lower bound: -0.0002468, upper bound: 0.0002486
time: 0.56 seconds

### Relational analysis RSZ of RS_RSZ1_RSZ2_RSZ1_RSZ1_RSZ2_RSZ1_RSZ2_RSZ2

#### Relational analysis RSZ result of RS_RSZ1_RSZ2_RSZ1_RSZ1_RSZ2_RSZ1_RSZ2_RSZ2
Status: Status.VERIFIED
Output dim: 1, lower bound: -0.0002468, upper bound: 0.0002486
time: 0.56 seconds

## BFS RS instance: RS_RSZ1_RSZ2_RSZ1_RSZ1_RSZ2_RSZ2_RSZ1

### Backsubstitution after applying RS history:
0: 0.0009457, 0.0011003, 0.0009457, 0.0011003, -0.0001486, 0.0001490
1: 0.9936197, 0.9940771, 0.9936197, 0.9940771, -0.0003879, 0.0003831
2: -0.0065606, -0.0050208, -0.0065606, -0.0050208, -0.0011945, 0.0011541
3: 0.0038004, 0.0040140, 0.0038004, 0.0040140, -0.0001623, 0.0001655
4: 0.0023852, 0.0036022, 0.0023852, 0.0036022, -0.0010169, 0.0010486
5: 0.0060086, 0.0065111, 0.0060086, 0.0065111, -0.0005024, 0.0005024
6: -0.0013983, -0.0008639, -0.0013983, -0.0008639, -0.0004081, 0.0003941
7: -0.0083221, -0.0079387, -0.0083221, -0.0079387, -0.0003835, 0.0003835
8: 0.0048484, 0.0068715, 0.0048484, 0.0068715, -0.0011737, 0.0012351
9: -0.0036877, -0.0032079, -0.0036877, -0.0032079, -0.0004797, 0.0004797

### Unstable ReLU Count (Linear/Conv2D Layers)
- layer_idx=0, type=LayerType.Linear, total=256, inp1_unstable=5, inp2_unstable=5, delta_unstable=251
- layer_idx=2, type=LayerType.Linear, total=256, inp1_unstable=4, inp2_unstable=4, delta_unstable=255
- layer_idx=4, type=LayerType.Linear, total=256, inp1_unstable=0, inp2_unstable=0, delta_unstable=256
- layer_idx=6, type=LayerType.Linear, total=256, inp1_unstable=1, inp2_unstable=1, delta_unstable=256
- layer_idx=8, type=LayerType.Linear, total=10, inp1_unstable=0, inp2_unstable=0, delta_unstable=10

Time for backsubstitution: 1.28 seconds

### RS candidates at layer 3
type: RSZ, layer: 3, pos: 73
type: RSZ, layer: 3, pos: 81
type: RSZ, layer: 3, pos: 136

Time for candidate selection: 0.14 seconds

### Candidate
type: RSZ, layer: 3, pos: 73

### Candidate
type: RSZ, layer: 3, pos: 81

### Relational analysis RSZ of RS_RSZ1_RSZ2_RSZ1_RSZ1_RSZ2_RSZ2_RSZ1_RSZ1

#### Relational analysis RSZ result of RS_RSZ1_RSZ2_RSZ1_RSZ1_RSZ2_RSZ2_RSZ1_RSZ1
Status: Status.VERIFIED
Output dim: 1, lower bound: -0.0002493, upper bound: 0.0002462
time: 0.54 seconds

### Relational analysis RSZ of RS_RSZ1_RSZ2_RSZ1_RSZ1_RSZ2_RSZ2_RSZ1_RSZ2

#### Relational analysis RSZ result of RS_RSZ1_RSZ2_RSZ1_RSZ1_RSZ2_RSZ2_RSZ1_RSZ2
Status: Status.VERIFIED
Output dim: 1, lower bound: -0.0002493, upper bound: 0.0002462
time: 0.55 seconds

## BFS RS instance: RS_RSZ1_RSZ2_RSZ1_RSZ1_RSZ2_RSZ2_RSZ2

### Backsubstitution after applying RS history:
0: 0.0009457, 0.0011003, 0.0009457, 0.0011003, -0.0001488, 0.0001487
1: 0.9936197, 0.9940771, 0.9936197, 0.9940771, -0.0003849, 0.0003852
2: -0.0065606, -0.0050208, -0.0065606, -0.0050208, -0.0011666, 0.0011758
3: 0.0038004, 0.0040140, 0.0038004, 0.0040140, -0.0001637, 0.0001635
4: 0.0023852, 0.0036022, 0.0023852, 0.0036022, -0.0010341, 0.0010268
5: 0.0060086, 0.0065111, 0.0060086, 0.0065111, -0.0005024, 0.0005024
6: -0.0013983, -0.0008639, -0.0013983, -0.0008639, -0.0003984, 0.0004016
7: -0.0083221, -0.0079387, -0.0083221, -0.0079387, -0.0003835, 0.0003835
8: 0.0048484, 0.0068715, 0.0048484, 0.0068715, -0.0012023, 0.0011951
9: -0.0036877, -0.0032079, -0.0036877, -0.0032079, -0.0004797, 0.0004797

### Unstable ReLU Count (Linear/Conv2D Layers)
- layer_idx=0, type=LayerType.Linear, total=256, inp1_unstable=5, inp2_unstable=5, delta_unstable=251
- layer_idx=2, type=LayerType.Linear, total=256, inp1_unstable=4, inp2_unstable=4, delta_unstable=255
- layer_idx=4, type=LayerType.Linear, total=256, inp1_unstable=0, inp2_unstable=0, delta_unstable=256
- layer_idx=6, type=LayerType.Linear, total=256, inp1_unstable=1, inp2_unstable=1, delta_unstable=256
- layer_idx=8, type=LayerType.Linear, total=10, inp1_unstable=0, inp2_unstable=0, delta_unstable=10

Time for backsubstitution: 1.30 seconds

### RS candidates at layer 3
type: RSZ, layer: 3, pos: 73
type: RSZ, layer: 3, pos: 81
type: RSZ, layer: 3, pos: 136

Time for candidate selection: 0.14 seconds

### Candidate
type: RSZ, layer: 3, pos: 73

### Candidate
type: RSZ, layer: 3, pos: 81

### Relational analysis RSZ of RS_RSZ1_RSZ2_RSZ1_RSZ1_RSZ2_RSZ2_RSZ2_RSZ1

#### Relational analysis RSZ result of RS_RSZ1_RSZ2_RSZ1_RSZ1_RSZ2_RSZ2_RSZ2_RSZ1
Status: Status.VERIFIED
Output dim: 1, lower bound: -0.0002468, upper bound: 0.0002486
time: 0.56 seconds

### Relational analysis RSZ of RS_RSZ1_RSZ2_RSZ1_RSZ1_RSZ2_RSZ2_RSZ2_RSZ2

#### Relational analysis RSZ result of RS_RSZ1_RSZ2_RSZ1_RSZ1_RSZ2_RSZ2_RSZ2_RSZ2
Status: Status.VERIFIED
Output dim: 1, lower bound: -0.0002468, upper bound: 0.0002486
time: 0.57 seconds

## BFS RS instance: RS_RSZ1_RSZ2_RSZ1_RSZ2_RSZ1_RSZ1_RSZ1

### Backsubstitution after applying RS history:
0: 0.0009457, 0.0011003, 0.0009457, 0.0011003, -0.0001486, 0.0001490
1: 0.9936197, 0.9940771, 0.9936197, 0.9940771, -0.0003877, 0.0003832
2: -0.0065606, -0.0050208, -0.0065606, -0.0050208, -0.0011933, 0.0011547
3: 0.0038004, 0.0040140, 0.0038004, 0.0040140, -0.0001623, 0.0001655
4: 0.0023852, 0.0036022, 0.0023852, 0.0036022, -0.0010174, 0.0010476
5: 0.0060086, 0.0065111, 0.0060086, 0.0065111, -0.0005024, 0.0005024
6: -0.0013983, -0.0008639, -0.0013983, -0.0008639, -0.0004077, 0.0003943
7: -0.0083221, -0.0079387, -0.0083221, -0.0079387, -0.0003835, 0.0003835
8: 0.0048484, 0.0068715, 0.0048484, 0.0068715, -0.0011746, 0.0012335
9: -0.0036877, -0.0032079, -0.0036877, -0.0032079, -0.0004797, 0.0004797

### Unstable ReLU Count (Linear/Conv2D Layers)
- layer_idx=0, type=LayerType.Linear, total=256, inp1_unstable=5, inp2_unstable=5, delta_unstable=251
- layer_idx=2, type=LayerType.Linear, total=256, inp1_unstable=4, inp2_unstable=4, delta_unstable=255
- layer_idx=4, type=LayerType.Linear, total=256, inp1_unstable=0, inp2_unstable=0, delta_unstable=256
- layer_idx=6, type=LayerType.Linear, total=256, inp1_unstable=1, inp2_unstable=1, delta_unstable=256
- layer_idx=8, type=LayerType.Linear, total=10, inp1_unstable=0, inp2_unstable=0, delta_unstable=10

Time for backsubstitution: 1.30 seconds

### RS candidates at layer 3
type: RSZ, layer: 3, pos: 73
type: RSZ, layer: 3, pos: 81
type: RSZ, layer: 3, pos: 136

Time for candidate selection: 0.13 seconds

### Candidate
type: RSZ, layer: 3, pos: 73

### Candidate
type: RSZ, layer: 3, pos: 81

### Relational analysis RSZ of RS_RSZ1_RSZ2_RSZ1_RSZ2_RSZ1_RSZ1_RSZ1_RSZ1

#### Relational analysis RSZ result of RS_RSZ1_RSZ2_RSZ1_RSZ2_RSZ1_RSZ1_RSZ1_RSZ1
Status: Status.VERIFIED
Output dim: 1, lower bound: -0.0002486, upper bound: 0.0002466
time: 0.62 seconds

### Relational analysis RSZ of RS_RSZ1_RSZ2_RSZ1_RSZ2_RSZ1_RSZ1_RSZ1_RSZ2

#### Relational analysis RSZ result of RS_RSZ1_RSZ2_RSZ1_RSZ2_RSZ1_RSZ1_RSZ1_RSZ2
Status: Status.VERIFIED
Output dim: 1, lower bound: -0.0002486, upper bound: 0.0002466
time: 0.63 seconds

## BFS RS instance: RS_RSZ1_RSZ2_RSZ1_RSZ2_RSZ1_RSZ1_RSZ2

### Backsubstitution after applying RS history:
0: 0.0009457, 0.0011003, 0.0009457, 0.0011003, -0.0001488, 0.0001488
1: 0.9936197, 0.9940771, 0.9936197, 0.9940771, -0.0003853, 0.0003855
2: -0.0065606, -0.0050208, -0.0065606, -0.0050208, -0.0011712, 0.0011794
3: 0.0038004, 0.0040140, 0.0038004, 0.0040140, -0.0001639, 0.0001638
4: 0.0023852, 0.0036022, 0.0023852, 0.0036022, -0.0010369, 0.0010304
5: 0.0060086, 0.0065111, 0.0060086, 0.0065111, -0.0005024, 0.0005024
6: -0.0013983, -0.0008639, -0.0013983, -0.0008639, -0.0004000, 0.0004029
7: -0.0083221, -0.0079387, -0.0083221, -0.0079387, -0.0003835, 0.0003835
8: 0.0048484, 0.0068715, 0.0048484, 0.0068715, -0.0012069, 0.0012011
9: -0.0036877, -0.0032079, -0.0036877, -0.0032079, -0.0004797, 0.0004797

### Unstable ReLU Count (Linear/Conv2D Layers)
- layer_idx=0, type=LayerType.Linear, total=256, inp1_unstable=5, inp2_unstable=5, delta_unstable=251
- layer_idx=2, type=LayerType.Linear, total=256, inp1_unstable=4, inp2_unstable=4, delta_unstable=255
- layer_idx=4, type=LayerType.Linear, total=256, inp1_unstable=0, inp2_unstable=0, delta_unstable=256
- layer_idx=6, type=LayerType.Linear, total=256, inp1_unstable=1, inp2_unstable=1, delta_unstable=256
- layer_idx=8, type=LayerType.Linear, total=10, inp1_unstable=0, inp2_unstable=0, delta_unstable=10

Time for backsubstitution: 1.29 seconds

### RS candidates at layer 3
type: RSZ, layer: 3, pos: 73
type: RSZ, layer: 3, pos: 81
type: RSZ, layer: 3, pos: 136

Time for candidate selection: 0.15 seconds

### Candidate
type: RSZ, layer: 3, pos: 73

### Candidate
type: RSZ, layer: 3, pos: 81

### Relational analysis RSZ of RS_RSZ1_RSZ2_RSZ1_RSZ2_RSZ1_RSZ1_RSZ2_RSZ1

#### Relational analysis RSZ result of RS_RSZ1_RSZ2_RSZ1_RSZ2_RSZ1_RSZ1_RSZ2_RSZ1
Status: Status.VERIFIED
Output dim: 1, lower bound: -0.0002463, upper bound: 0.0002495
time: 0.58 seconds

### Relational analysis RSZ of RS_RSZ1_RSZ2_RSZ1_RSZ2_RSZ1_RSZ1_RSZ2_RSZ2

#### Relational analysis RSZ result of RS_RSZ1_RSZ2_RSZ1_RSZ2_RSZ1_RSZ1_RSZ2_RSZ2
Status: Status.VERIFIED
Output dim: 1, lower bound: -0.0002463, upper bound: 0.0002495
time: 0.57 seconds

## BFS RS instance: RS_RSZ1_RSZ2_RSZ1_RSZ2_RSZ1_RSZ2_RSZ1

### Backsubstitution after applying RS history:
0: 0.0009457, 0.0011003, 0.0009457, 0.0011003, -0.0001486, 0.0001490
1: 0.9936197, 0.9940771, 0.9936197, 0.9940771, -0.0003877, 0.0003832
2: -0.0065606, -0.0050208, -0.0065606, -0.0050208, -0.0011933, 0.0011547
3: 0.0038004, 0.0040140, 0.0038004, 0.0040140, -0.0001623, 0.0001655
4: 0.0023852, 0.0036022, 0.0023852, 0.0036022, -0.0010174, 0.0010476
5: 0.0060086, 0.0065111, 0.0060086, 0.0065111, -0.0005024, 0.0005024
6: -0.0013983, -0.0008639, -0.0013983, -0.0008639, -0.0004077, 0.0003943
7: -0.0083221, -0.0079387, -0.0083221, -0.0079387, -0.0003835, 0.0003835
8: 0.0048484, 0.0068715, 0.0048484, 0.0068715, -0.0011746, 0.0012335
9: -0.0036877, -0.0032079, -0.0036877, -0.0032079, -0.0004797, 0.0004797

### Unstable ReLU Count (Linear/Conv2D Layers)
- layer_idx=0, type=LayerType.Linear, total=256, inp1_unstable=5, inp2_unstable=5, delta_unstable=251
- layer_idx=2, type=LayerType.Linear, total=256, inp1_unstable=4, inp2_unstable=4, delta_unstable=255
- layer_idx=4, type=LayerType.Linear, total=256, inp1_unstable=0, inp2_unstable=0, delta_unstable=256
- layer_idx=6, type=LayerType.Linear, total=256, inp1_unstable=1, inp2_unstable=1, delta_unstable=256
- layer_idx=8, type=LayerType.Linear, total=10, inp1_unstable=0, inp2_unstable=0, delta_unstable=10

Time for backsubstitution: 1.30 seconds

### RS candidates at layer 3
type: RSZ, layer: 3, pos: 73
type: RSZ, layer: 3, pos: 81
type: RSZ, layer: 3, pos: 136

Time for candidate selection: 0.14 seconds

### Candidate
type: RSZ, layer: 3, pos: 73

### Candidate
type: RSZ, layer: 3, pos: 81

### Relational analysis RSZ of RS_RSZ1_RSZ2_RSZ1_RSZ2_RSZ1_RSZ2_RSZ1_RSZ1

#### Relational analysis RSZ result of RS_RSZ1_RSZ2_RSZ1_RSZ2_RSZ1_RSZ2_RSZ1_RSZ1
Status: Status.VERIFIED
Output dim: 1, lower bound: -0.0002486, upper bound: 0.0002466
time: 0.60 seconds

### Relational analysis RSZ of RS_RSZ1_RSZ2_RSZ1_RSZ2_RSZ1_RSZ2_RSZ1_RSZ2

#### Relational analysis RSZ result of RS_RSZ1_RSZ2_RSZ1_RSZ2_RSZ1_RSZ2_RSZ1_RSZ2
Status: Status.VERIFIED
Output dim: 1, lower bound: -0.0002486, upper bound: 0.0002466
time: 0.60 seconds

## BFS RS instance: RS_RSZ1_RSZ2_RSZ1_RSZ2_RSZ1_RSZ2_RSZ2

### Backsubstitution after applying RS history:
0: 0.0009457, 0.0011003, 0.0009457, 0.0011003, -0.0001488, 0.0001488
1: 0.9936197, 0.9940771, 0.9936197, 0.9940771, -0.0003853, 0.0003855
2: -0.0065606, -0.0050208, -0.0065606, -0.0050208, -0.0011712, 0.0011794
3: 0.0038004, 0.0040140, 0.0038004, 0.0040140, -0.0001639, 0.0001638
4: 0.0023852, 0.0036022, 0.0023852, 0.0036022, -0.0010369, 0.0010304
5: 0.0060086, 0.0065111, 0.0060086, 0.0065111, -0.0005024, 0.0005024
6: -0.0013983, -0.0008639, -0.0013983, -0.0008639, -0.0004000, 0.0004029
7: -0.0083221, -0.0079387, -0.0083221, -0.0079387, -0.0003835, 0.0003835
8: 0.0048484, 0.0068715, 0.0048484, 0.0068715, -0.0012069, 0.0012011
9: -0.0036877, -0.0032079, -0.0036877, -0.0032079, -0.0004797, 0.0004797

### Unstable ReLU Count (Linear/Conv2D Layers)
- layer_idx=0, type=LayerType.Linear, total=256, inp1_unstable=5, inp2_unstable=5, delta_unstable=251
- layer_idx=2, type=LayerType.Linear, total=256, inp1_unstable=4, inp2_unstable=4, delta_unstable=255
- layer_idx=4, type=LayerType.Linear, total=256, inp1_unstable=0, inp2_unstable=0, delta_unstable=256
- layer_idx=6, type=LayerType.Linear, total=256, inp1_unstable=1, inp2_unstable=1, delta_unstable=256
- layer_idx=8, type=LayerType.Linear, total=10, inp1_unstable=0, inp2_unstable=0, delta_unstable=10

Time for backsubstitution: 1.29 seconds

### RS candidates at layer 3
type: RSZ, layer: 3, pos: 73
type: RSZ, layer: 3, pos: 81
type: RSZ, layer: 3, pos: 136

Time for candidate selection: 0.14 seconds

### Candidate
type: RSZ, layer: 3, pos: 73

### Candidate
type: RSZ, layer: 3, pos: 81

### Relational analysis RSZ of RS_RSZ1_RSZ2_RSZ1_RSZ2_RSZ1_RSZ2_RSZ2_RSZ1

#### Relational analysis RSZ result of RS_RSZ1_RSZ2_RSZ1_RSZ2_RSZ1_RSZ2_RSZ2_RSZ1
Status: Status.VERIFIED
Output dim: 1, lower bound: -0.0002463, upper bound: 0.0002495
time: 0.58 seconds

### Relational analysis RSZ of RS_RSZ1_RSZ2_RSZ1_RSZ2_RSZ1_RSZ2_RSZ2_RSZ2

#### Relational analysis RSZ result of RS_RSZ1_RSZ2_RSZ1_RSZ2_RSZ1_RSZ2_RSZ2_RSZ2
Status: Status.VERIFIED
Output dim: 1, lower bound: -0.0002463, upper bound: 0.0002495
time: 0.58 seconds

## BFS RS instance: RS_RSZ1_RSZ2_RSZ1_RSZ2_RSZ2_RSZ1_RSZ1

### Backsubstitution after applying RS history:
0: 0.0009457, 0.0011003, 0.0009457, 0.0011003, -0.0001486, 0.0001489
1: 0.9936197, 0.9940771, 0.9936197, 0.9940771, -0.0003874, 0.0003834
2: -0.0065606, -0.0050208, -0.0065606, -0.0050208, -0.0011891, 0.0011569
3: 0.0038004, 0.0040140, 0.0038004, 0.0040140, -0.0001624, 0.0001652
4: 0.0023852, 0.0036022, 0.0023852, 0.0036022, -0.0010192, 0.0010443
5: 0.0060086, 0.0065111, 0.0060086, 0.0065111, -0.0005024, 0.0005024
6: -0.0013983, -0.0008639, -0.0013983, -0.0008639, -0.0004062, 0.0003951
7: -0.0083221, -0.0079387, -0.0083221, -0.0079387, -0.0003835, 0.0003835
8: 0.0048484, 0.0068715, 0.0048484, 0.0068715, -0.0011775, 0.0012281
9: -0.0036877, -0.0032079, -0.0036877, -0.0032079, -0.0004797, 0.0004797

### Unstable ReLU Count (Linear/Conv2D Layers)
- layer_idx=0, type=LayerType.Linear, total=256, inp1_unstable=5, inp2_unstable=5, delta_unstable=251
- layer_idx=2, type=LayerType.Linear, total=256, inp1_unstable=4, inp2_unstable=4, delta_unstable=255
- layer_idx=4, type=LayerType.Linear, total=256, inp1_unstable=0, inp2_unstable=0, delta_unstable=256
- layer_idx=6, type=LayerType.Linear, total=256, inp1_unstable=1, inp2_unstable=1, delta_unstable=256
- layer_idx=8, type=LayerType.Linear, total=10, inp1_unstable=0, inp2_unstable=0, delta_unstable=10

Time for backsubstitution: 1.26 seconds

### RS candidates at layer 3
type: RSZ, layer: 3, pos: 73
type: RSZ, layer: 3, pos: 81
type: RSZ, layer: 3, pos: 136

Time for candidate selection: 0.14 seconds

### Candidate
type: RSZ, layer: 3, pos: 73

### Candidate
type: RSZ, layer: 3, pos: 81

### Relational analysis RSZ of RS_RSZ1_RSZ2_RSZ1_RSZ2_RSZ2_RSZ1_RSZ1_RSZ1

#### Relational analysis RSZ result of RS_RSZ1_RSZ2_RSZ1_RSZ2_RSZ2_RSZ1_RSZ1_RSZ1
Status: Status.VERIFIED
Output dim: 1, lower bound: -0.0002486, upper bound: 0.0002467
time: 0.59 seconds

### Relational analysis RSZ of RS_RSZ1_RSZ2_RSZ1_RSZ2_RSZ2_RSZ1_RSZ1_RSZ2

#### Relational analysis RSZ result of RS_RSZ1_RSZ2_RSZ1_RSZ2_RSZ2_RSZ1_RSZ1_RSZ2
Status: Status.VERIFIED
Output dim: 1, lower bound: -0.0002486, upper bound: 0.0002467
time: 0.60 seconds

## BFS RS instance: RS_RSZ1_RSZ2_RSZ1_RSZ2_RSZ2_RSZ1_RSZ2

### Backsubstitution after applying RS history:
0: 0.0009457, 0.0011003, 0.0009457, 0.0011003, -0.0001488, 0.0001488
1: 0.9936197, 0.9940771, 0.9936197, 0.9940771, -0.0003851, 0.0003859
2: -0.0065606, -0.0050208, -0.0065606, -0.0050208, -0.0011696, 0.0011830
3: 0.0038004, 0.0040140, 0.0038004, 0.0040140, -0.0001642, 0.0001637
4: 0.0023852, 0.0036022, 0.0023852, 0.0036022, -0.0010397, 0.0010292
5: 0.0060086, 0.0065111, 0.0060086, 0.0065111, -0.0005024, 0.0005024
6: -0.0013983, -0.0008639, -0.0013983, -0.0008639, -0.0003995, 0.0004041
7: -0.0083221, -0.0079387, -0.0083221, -0.0079387, -0.0003835, 0.0003835
8: 0.0048484, 0.0068715, 0.0048484, 0.0068715, -0.0012117, 0.0011990
9: -0.0036877, -0.0032079, -0.0036877, -0.0032079, -0.0004797, 0.0004797

### Unstable ReLU Count (Linear/Conv2D Layers)
- layer_idx=0, type=LayerType.Linear, total=256, inp1_unstable=5, inp2_unstable=5, delta_unstable=251
- layer_idx=2, type=LayerType.Linear, total=256, inp1_unstable=4, inp2_unstable=4, delta_unstable=255
- layer_idx=4, type=LayerType.Linear, total=256, inp1_unstable=0, inp2_unstable=0, delta_unstable=256
- layer_idx=6, type=LayerType.Linear, total=256, inp1_unstable=1, inp2_unstable=1, delta_unstable=256
- layer_idx=8, type=LayerType.Linear, total=10, inp1_unstable=0, inp2_unstable=0, delta_unstable=10

Time for backsubstitution: 1.24 seconds

### RS candidates at layer 3
type: RSZ, layer: 3, pos: 73
type: RSZ, layer: 3, pos: 81
type: RSZ, layer: 3, pos: 136

Time for candidate selection: 0.12 seconds

### Candidate
type: RSZ, layer: 3, pos: 73

### Candidate
type: RSZ, layer: 3, pos: 81

### Relational analysis RSZ of RS_RSZ1_RSZ2_RSZ1_RSZ2_RSZ2_RSZ1_RSZ2_RSZ1

#### Relational analysis RSZ result of RS_RSZ1_RSZ2_RSZ1_RSZ2_RSZ2_RSZ1_RSZ2_RSZ1
Status: Status.VERIFIED
Output dim: 1, lower bound: -0.0002463, upper bound: 0.0002495
time: 0.61 seconds

### Relational analysis RSZ of RS_RSZ1_RSZ2_RSZ1_RSZ2_RSZ2_RSZ1_RSZ2_RSZ2

#### Relational analysis RSZ result of RS_RSZ1_RSZ2_RSZ1_RSZ2_RSZ2_RSZ1_RSZ2_RSZ2
Status: Status.VERIFIED
Output dim: 1, lower bound: -0.0002463, upper bound: 0.0002495
time: 0.61 seconds

## BFS RS instance: RS_RSZ1_RSZ2_RSZ1_RSZ2_RSZ2_RSZ2_RSZ1

### Backsubstitution after applying RS history:
0: 0.0009457, 0.0011003, 0.0009457, 0.0011003, -0.0001486, 0.0001489
1: 0.9936197, 0.9940771, 0.9936197, 0.9940771, -0.0003874, 0.0003834
2: -0.0065606, -0.0050208, -0.0065606, -0.0050208, -0.0011891, 0.0011569
3: 0.0038004, 0.0040140, 0.0038004, 0.0040140, -0.0001624, 0.0001652
4: 0.0023852, 0.0036022, 0.0023852, 0.0036022, -0.0010192, 0.0010443
5: 0.0060086, 0.0065111, 0.0060086, 0.0065111, -0.0005024, 0.0005024
6: -0.0013983, -0.0008639, -0.0013983, -0.0008639, -0.0004062, 0.0003951
7: -0.0083221, -0.0079387, -0.0083221, -0.0079387, -0.0003835, 0.0003835
8: 0.0048484, 0.0068715, 0.0048484, 0.0068715, -0.0011775, 0.0012281
9: -0.0036877, -0.0032079, -0.0036877, -0.0032079, -0.0004797, 0.0004797

### Unstable ReLU Count (Linear/Conv2D Layers)
- layer_idx=0, type=LayerType.Linear, total=256, inp1_unstable=5, inp2_unstable=5, delta_unstable=251
- layer_idx=2, type=LayerType.Linear, total=256, inp1_unstable=4, inp2_unstable=4, delta_unstable=255
- layer_idx=4, type=LayerType.Linear, total=256, inp1_unstable=0, inp2_unstable=0, delta_unstable=256
- layer_idx=6, type=LayerType.Linear, total=256, inp1_unstable=1, inp2_unstable=1, delta_unstable=256
- layer_idx=8, type=LayerType.Linear, total=10, inp1_unstable=0, inp2_unstable=0, delta_unstable=10

Time for backsubstitution: 1.24 seconds

### RS candidates at layer 3
type: RSZ, layer: 3, pos: 73
type: RSZ, layer: 3, pos: 81
type: RSZ, layer: 3, pos: 136

Time for candidate selection: 0.12 seconds

### Candidate
type: RSZ, layer: 3, pos: 73

### Candidate
type: RSZ, layer: 3, pos: 81

### Relational analysis RSZ of RS_RSZ1_RSZ2_RSZ1_RSZ2_RSZ2_RSZ2_RSZ1_RSZ1

#### Relational analysis RSZ result of RS_RSZ1_RSZ2_RSZ1_RSZ2_RSZ2_RSZ2_RSZ1_RSZ1
Status: Status.VERIFIED
Output dim: 1, lower bound: -0.0002486, upper bound: 0.0002467
time: 0.57 seconds

### Relational analysis RSZ of RS_RSZ1_RSZ2_RSZ1_RSZ2_RSZ2_RSZ2_RSZ1_RSZ2

#### Relational analysis RSZ result of RS_RSZ1_RSZ2_RSZ1_RSZ2_RSZ2_RSZ2_RSZ1_RSZ2
Status: Status.VERIFIED
Output dim: 1, lower bound: -0.0002486, upper bound: 0.0002467
time: 0.55 seconds

## BFS RS instance: RS_RSZ1_RSZ2_RSZ1_RSZ2_RSZ2_RSZ2_RSZ2

### Backsubstitution after applying RS history:
0: 0.0009457, 0.0011003, 0.0009457, 0.0011003, -0.0001488, 0.0001488
1: 0.9936197, 0.9940771, 0.9936197, 0.9940771, -0.0003851, 0.0003859
2: -0.0065606, -0.0050208, -0.0065606, -0.0050208, -0.0011696, 0.0011830
3: 0.0038004, 0.0040140, 0.0038004, 0.0040140, -0.0001642, 0.0001637
4: 0.0023852, 0.0036022, 0.0023852, 0.0036022, -0.0010397, 0.0010292
5: 0.0060086, 0.0065111, 0.0060086, 0.0065111, -0.0005024, 0.0005024
6: -0.0013983, -0.0008639, -0.0013983, -0.0008639, -0.0003995, 0.0004041
7: -0.0083221, -0.0079387, -0.0083221, -0.0079387, -0.0003835, 0.0003835
8: 0.0048484, 0.0068715, 0.0048484, 0.0068715, -0.0012117, 0.0011990
9: -0.0036877, -0.0032079, -0.0036877, -0.0032079, -0.0004797, 0.0004797

### Unstable ReLU Count (Linear/Conv2D Layers)
- layer_idx=0, type=LayerType.Linear, total=256, inp1_unstable=5, inp2_unstable=5, delta_unstable=251
- layer_idx=2, type=LayerType.Linear, total=256, inp1_unstable=4, inp2_unstable=4, delta_unstable=255
- layer_idx=4, type=LayerType.Linear, total=256, inp1_unstable=0, inp2_unstable=0, delta_unstable=256
- layer_idx=6, type=LayerType.Linear, total=256, inp1_unstable=1, inp2_unstable=1, delta_unstable=256
- layer_idx=8, type=LayerType.Linear, total=10, inp1_unstable=0, inp2_unstable=0, delta_unstable=10

Time for backsubstitution: 1.22 seconds

### RS candidates at layer 3
type: RSZ, layer: 3, pos: 73
type: RSZ, layer: 3, pos: 81
type: RSZ, layer: 3, pos: 136

Time for candidate selection: 0.13 seconds

### Candidate
type: RSZ, layer: 3, pos: 73

### Candidate
type: RSZ, layer: 3, pos: 81

### Relational analysis RSZ of RS_RSZ1_RSZ2_RSZ1_RSZ2_RSZ2_RSZ2_RSZ2_RSZ1

#### Relational analysis RSZ result of RS_RSZ1_RSZ2_RSZ1_RSZ2_RSZ2_RSZ2_RSZ2_RSZ1
Status: Status.VERIFIED
Output dim: 1, lower bound: -0.0002463, upper bound: 0.0002495
time: 0.58 seconds

### Relational analysis RSZ of RS_RSZ1_RSZ2_RSZ1_RSZ2_RSZ2_RSZ2_RSZ2_RSZ2

#### Relational analysis RSZ result of RS_RSZ1_RSZ2_RSZ1_RSZ2_RSZ2_RSZ2_RSZ2_RSZ2
Status: Status.VERIFIED
Output dim: 1, lower bound: -0.0002463, upper bound: 0.0002495
time: 0.58 seconds

## BFS RS instance: RS_RSZ1_RSZ2_RSZ2_RSZ1_RSZ1_RSZ1_RSZ1

### Backsubstitution after applying RS history:
0: 0.0009457, 0.0011003, 0.0009457, 0.0011003, -0.0001486, 0.0001491
1: 0.9936197, 0.9940771, 0.9936197, 0.9940771, -0.0003890, 0.0003829
2: -0.0065606, -0.0050208, -0.0065606, -0.0050208, -0.0012069, 0.0011518
3: 0.0038004, 0.0040140, 0.0038004, 0.0040140, -0.0001621, 0.0001664
4: 0.0023852, 0.0036022, 0.0023852, 0.0036022, -0.0010151, 0.0010584
5: 0.0060086, 0.0065111, 0.0060086, 0.0065111, -0.0005024, 0.0005024
6: -0.0013983, -0.0008639, -0.0013983, -0.0008639, -0.0004124, 0.0003933
7: -0.0083221, -0.0079387, -0.0083221, -0.0079387, -0.0003835, 0.0003835
8: 0.0048484, 0.0068715, 0.0048484, 0.0068715, -0.0011708, 0.0012514
9: -0.0036877, -0.0032079, -0.0036877, -0.0032079, -0.0004797, 0.0004797

### Unstable ReLU Count (Linear/Conv2D Layers)
- layer_idx=0, type=LayerType.Linear, total=256, inp1_unstable=5, inp2_unstable=5, delta_unstable=251
- layer_idx=2, type=LayerType.Linear, total=256, inp1_unstable=4, inp2_unstable=4, delta_unstable=255
- layer_idx=4, type=LayerType.Linear, total=256, inp1_unstable=0, inp2_unstable=0, delta_unstable=256
- layer_idx=6, type=LayerType.Linear, total=256, inp1_unstable=1, inp2_unstable=1, delta_unstable=256
- layer_idx=8, type=LayerType.Linear, total=10, inp1_unstable=0, inp2_unstable=0, delta_unstable=10

Time for backsubstitution: 1.32 seconds

### RS candidates at layer 3
type: RSZ, layer: 3, pos: 73
type: RSZ, layer: 3, pos: 81
type: RSZ, layer: 3, pos: 136

Time for candidate selection: 0.13 seconds

### Candidate
type: RSZ, layer: 3, pos: 73

### Candidate
type: RSZ, layer: 3, pos: 81

### Relational analysis RSZ of RS_RSZ1_RSZ2_RSZ2_RSZ1_RSZ1_RSZ1_RSZ1_RSZ1

#### Relational analysis RSZ result of RS_RSZ1_RSZ2_RSZ2_RSZ1_RSZ1_RSZ1_RSZ1_RSZ1
Status: Status.VERIFIED
Output dim: 1, lower bound: -0.0002496, upper bound: 0.0002463
time: 0.60 seconds

### Relational analysis RSZ of RS_RSZ1_RSZ2_RSZ2_RSZ1_RSZ1_RSZ1_RSZ1_RSZ2

#### Relational analysis RSZ result of RS_RSZ1_RSZ2_RSZ2_RSZ1_RSZ1_RSZ1_RSZ1_RSZ2
Status: Status.VERIFIED
Output dim: 1, lower bound: -0.0002496, upper bound: 0.0002463
time: 0.61 seconds

## BFS RS instance: RS_RSZ1_RSZ2_RSZ2_RSZ1_RSZ1_RSZ1_RSZ2

### Backsubstitution after applying RS history:
0: 0.0009457, 0.0011003, 0.0009457, 0.0011003, -0.0001487, 0.0001488
1: 0.9936197, 0.9940771, 0.9936197, 0.9940771, -0.0003855, 0.0003841
2: -0.0065606, -0.0050208, -0.0065606, -0.0050208, -0.0011733, 0.0011641
3: 0.0038004, 0.0040140, 0.0038004, 0.0040140, -0.0001629, 0.0001639
4: 0.0023852, 0.0036022, 0.0023852, 0.0036022, -0.0010248, 0.0010321
5: 0.0060086, 0.0065111, 0.0060086, 0.0065111, -0.0005024, 0.0005024
6: -0.0013983, -0.0008639, -0.0013983, -0.0008639, -0.0004008, 0.0003976
7: -0.0083221, -0.0079387, -0.0083221, -0.0079387, -0.0003835, 0.0003835
8: 0.0048484, 0.0068715, 0.0048484, 0.0068715, -0.0011869, 0.0012038
9: -0.0036877, -0.0032079, -0.0036877, -0.0032079, -0.0004797, 0.0004797

### Unstable ReLU Count (Linear/Conv2D Layers)
- layer_idx=0, type=LayerType.Linear, total=256, inp1_unstable=5, inp2_unstable=5, delta_unstable=251
- layer_idx=2, type=LayerType.Linear, total=256, inp1_unstable=4, inp2_unstable=4, delta_unstable=255
- layer_idx=4, type=LayerType.Linear, total=256, inp1_unstable=0, inp2_unstable=0, delta_unstable=256
- layer_idx=6, type=LayerType.Linear, total=256, inp1_unstable=1, inp2_unstable=1, delta_unstable=256
- layer_idx=8, type=LayerType.Linear, total=10, inp1_unstable=0, inp2_unstable=0, delta_unstable=10

Time for backsubstitution: 1.39 seconds

### RS candidates at layer 3
type: RSZ, layer: 3, pos: 73
type: RSZ, layer: 3, pos: 81
type: RSZ, layer: 3, pos: 136

Time for candidate selection: 0.14 seconds

### Candidate
type: RSZ, layer: 3, pos: 73

### Candidate
type: RSZ, layer: 3, pos: 81

### Relational analysis RSZ of RS_RSZ1_RSZ2_RSZ2_RSZ1_RSZ1_RSZ1_RSZ2_RSZ1

#### Relational analysis RSZ result of RS_RSZ1_RSZ2_RSZ2_RSZ1_RSZ1_RSZ1_RSZ2_RSZ1
Status: Status.VERIFIED
Output dim: 1, lower bound: -0.0002467, upper bound: 0.0002481
time: 0.58 seconds

### Relational analysis RSZ of RS_RSZ1_RSZ2_RSZ2_RSZ1_RSZ1_RSZ1_RSZ2_RSZ2

#### Relational analysis RSZ result of RS_RSZ1_RSZ2_RSZ2_RSZ1_RSZ1_RSZ1_RSZ2_RSZ2
Status: Status.VERIFIED
Output dim: 1, lower bound: -0.0002467, upper bound: 0.0002481
time: 0.58 seconds

## BFS RS instance: RS_RSZ1_RSZ2_RSZ2_RSZ1_RSZ1_RSZ2_RSZ1

### Backsubstitution after applying RS history:
0: 0.0009457, 0.0011003, 0.0009457, 0.0011003, -0.0001486, 0.0001491
1: 0.9936197, 0.9940771, 0.9936197, 0.9940771, -0.0003890, 0.0003829
2: -0.0065606, -0.0050208, -0.0065606, -0.0050208, -0.0012069, 0.0011518
3: 0.0038004, 0.0040140, 0.0038004, 0.0040140, -0.0001621, 0.0001664
4: 0.0023852, 0.0036022, 0.0023852, 0.0036022, -0.0010151, 0.0010584
5: 0.0060086, 0.0065111, 0.0060086, 0.0065111, -0.0005024, 0.0005024
6: -0.0013983, -0.0008639, -0.0013983, -0.0008639, -0.0004124, 0.0003933
7: -0.0083221, -0.0079387, -0.0083221, -0.0079387, -0.0003835, 0.0003835
8: 0.0048484, 0.0068715, 0.0048484, 0.0068715, -0.0011708, 0.0012514
9: -0.0036877, -0.0032079, -0.0036877, -0.0032079, -0.0004797, 0.0004797

### Unstable ReLU Count (Linear/Conv2D Layers)
- layer_idx=0, type=LayerType.Linear, total=256, inp1_unstable=5, inp2_unstable=5, delta_unstable=251
- layer_idx=2, type=LayerType.Linear, total=256, inp1_unstable=4, inp2_unstable=4, delta_unstable=255
- layer_idx=4, type=LayerType.Linear, total=256, inp1_unstable=0, inp2_unstable=0, delta_unstable=256
- layer_idx=6, type=LayerType.Linear, total=256, inp1_unstable=1, inp2_unstable=1, delta_unstable=256
- layer_idx=8, type=LayerType.Linear, total=10, inp1_unstable=0, inp2_unstable=0, delta_unstable=10

Time for backsubstitution: 1.28 seconds

### RS candidates at layer 3
type: RSZ, layer: 3, pos: 73
type: RSZ, layer: 3, pos: 81
type: RSZ, layer: 3, pos: 136

Time for candidate selection: 0.14 seconds

### Candidate
type: RSZ, layer: 3, pos: 73

### Candidate
type: RSZ, layer: 3, pos: 81

### Relational analysis RSZ of RS_RSZ1_RSZ2_RSZ2_RSZ1_RSZ1_RSZ2_RSZ1_RSZ1

#### Relational analysis RSZ result of RS_RSZ1_RSZ2_RSZ2_RSZ1_RSZ1_RSZ2_RSZ1_RSZ1
Status: Status.VERIFIED
Output dim: 1, lower bound: -0.0002496, upper bound: 0.0002463
time: 0.58 seconds

### Relational analysis RSZ of RS_RSZ1_RSZ2_RSZ2_RSZ1_RSZ1_RSZ2_RSZ1_RSZ2

#### Relational analysis RSZ result of RS_RSZ1_RSZ2_RSZ2_RSZ1_RSZ1_RSZ2_RSZ1_RSZ2
Status: Status.VERIFIED
Output dim: 1, lower bound: -0.0002496, upper bound: 0.0002463
time: 0.59 seconds

## BFS RS instance: RS_RSZ1_RSZ2_RSZ2_RSZ1_RSZ1_RSZ2_RSZ2

### Backsubstitution after applying RS history:
0: 0.0009457, 0.0011003, 0.0009457, 0.0011003, -0.0001487, 0.0001488
1: 0.9936197, 0.9940771, 0.9936197, 0.9940771, -0.0003855, 0.0003841
2: -0.0065606, -0.0050208, -0.0065606, -0.0050208, -0.0011733, 0.0011641
3: 0.0038004, 0.0040140, 0.0038004, 0.0040140, -0.0001629, 0.0001639
4: 0.0023852, 0.0036022, 0.0023852, 0.0036022, -0.0010248, 0.0010321
5: 0.0060086, 0.0065111, 0.0060086, 0.0065111, -0.0005024, 0.0005024
6: -0.0013983, -0.0008639, -0.0013983, -0.0008639, -0.0004008, 0.0003976
7: -0.0083221, -0.0079387, -0.0083221, -0.0079387, -0.0003835, 0.0003835
8: 0.0048484, 0.0068715, 0.0048484, 0.0068715, -0.0011869, 0.0012038
9: -0.0036877, -0.0032079, -0.0036877, -0.0032079, -0.0004797, 0.0004797

### Unstable ReLU Count (Linear/Conv2D Layers)
- layer_idx=0, type=LayerType.Linear, total=256, inp1_unstable=5, inp2_unstable=5, delta_unstable=251
- layer_idx=2, type=LayerType.Linear, total=256, inp1_unstable=4, inp2_unstable=4, delta_unstable=255
- layer_idx=4, type=LayerType.Linear, total=256, inp1_unstable=0, inp2_unstable=0, delta_unstable=256
- layer_idx=6, type=LayerType.Linear, total=256, inp1_unstable=1, inp2_unstable=1, delta_unstable=256
- layer_idx=8, type=LayerType.Linear, total=10, inp1_unstable=0, inp2_unstable=0, delta_unstable=10

Time for backsubstitution: 1.30 seconds

### RS candidates at layer 3
type: RSZ, layer: 3, pos: 73
type: RSZ, layer: 3, pos: 81
type: RSZ, layer: 3, pos: 136

Time for candidate selection: 0.14 seconds

### Candidate
type: RSZ, layer: 3, pos: 73

### Candidate
type: RSZ, layer: 3, pos: 81

### Relational analysis RSZ of RS_RSZ1_RSZ2_RSZ2_RSZ1_RSZ1_RSZ2_RSZ2_RSZ1

#### Relational analysis RSZ result of RS_RSZ1_RSZ2_RSZ2_RSZ1_RSZ1_RSZ2_RSZ2_RSZ1
Status: Status.VERIFIED
Output dim: 1, lower bound: -0.0002467, upper bound: 0.0002481
time: 0.66 seconds

### Relational analysis RSZ of RS_RSZ1_RSZ2_RSZ2_RSZ1_RSZ1_RSZ2_RSZ2_RSZ2

#### Relational analysis RSZ result of RS_RSZ1_RSZ2_RSZ2_RSZ1_RSZ1_RSZ2_RSZ2_RSZ2
Status: Status.VERIFIED
Output dim: 1, lower bound: -0.0002467, upper bound: 0.0002481
time: 0.64 seconds

## BFS RS instance: RS_RSZ1_RSZ2_RSZ2_RSZ1_RSZ2_RSZ1_RSZ1

### Backsubstitution after applying RS history:
0: 0.0009457, 0.0011003, 0.0009457, 0.0011003, -0.0001486, 0.0001490
1: 0.9936197, 0.9940771, 0.9936197, 0.9940771, -0.0003887, 0.0003831
2: -0.0065606, -0.0050208, -0.0065606, -0.0050208, -0.0012030, 0.0011534
3: 0.0038004, 0.0040140, 0.0038004, 0.0040140, -0.0001622, 0.0001661
4: 0.0023852, 0.0036022, 0.0023852, 0.0036022, -0.0010164, 0.0010553
5: 0.0060086, 0.0065111, 0.0060086, 0.0065111, -0.0005024, 0.0005024
6: -0.0013983, -0.0008639, -0.0013983, -0.0008639, -0.0004111, 0.0003938
7: -0.0083221, -0.0079387, -0.0083221, -0.0079387, -0.0003835, 0.0003835
8: 0.0048484, 0.0068715, 0.0048484, 0.0068715, -0.0011728, 0.0012463
9: -0.0036877, -0.0032079, -0.0036877, -0.0032079, -0.0004797, 0.0004797

### Unstable ReLU Count (Linear/Conv2D Layers)
- layer_idx=0, type=LayerType.Linear, total=256, inp1_unstable=5, inp2_unstable=5, delta_unstable=251
- layer_idx=2, type=LayerType.Linear, total=256, inp1_unstable=4, inp2_unstable=4, delta_unstable=255
- layer_idx=4, type=LayerType.Linear, total=256, inp1_unstable=0, inp2_unstable=0, delta_unstable=256
- layer_idx=6, type=LayerType.Linear, total=256, inp1_unstable=1, inp2_unstable=1, delta_unstable=256
- layer_idx=8, type=LayerType.Linear, total=10, inp1_unstable=0, inp2_unstable=0, delta_unstable=10

Time for backsubstitution: 1.29 seconds

### RS candidates at layer 3
type: RSZ, layer: 3, pos: 73
type: RSZ, layer: 3, pos: 81
type: RSZ, layer: 3, pos: 136

Time for candidate selection: 0.15 seconds

### Candidate
type: RSZ, layer: 3, pos: 73

### Candidate
type: RSZ, layer: 3, pos: 81

### Relational analysis RSZ of RS_RSZ1_RSZ2_RSZ2_RSZ1_RSZ2_RSZ1_RSZ1_RSZ1

#### Relational analysis RSZ result of RS_RSZ1_RSZ2_RSZ2_RSZ1_RSZ2_RSZ1_RSZ1_RSZ1
Status: Status.VERIFIED
Output dim: 1, lower bound: -0.0002496, upper bound: 0.0002463
time: 0.59 seconds

### Relational analysis RSZ of RS_RSZ1_RSZ2_RSZ2_RSZ1_RSZ2_RSZ1_RSZ1_RSZ2

#### Relational analysis RSZ result of RS_RSZ1_RSZ2_RSZ2_RSZ1_RSZ2_RSZ1_RSZ1_RSZ2
Status: Status.VERIFIED
Output dim: 1, lower bound: -0.0002496, upper bound: 0.0002463
time: 0.60 seconds

## BFS RS instance: RS_RSZ1_RSZ2_RSZ2_RSZ1_RSZ2_RSZ1_RSZ2

### Backsubstitution after applying RS history:
0: 0.0009457, 0.0011003, 0.0009457, 0.0011003, -0.0001487, 0.0001488
1: 0.9936197, 0.9940771, 0.9936197, 0.9940771, -0.0003852, 0.0003845
2: -0.0065606, -0.0050208, -0.0065606, -0.0050208, -0.0011703, 0.0011685
3: 0.0038004, 0.0040140, 0.0038004, 0.0040140, -0.0001632, 0.0001637
4: 0.0023852, 0.0036022, 0.0023852, 0.0036022, -0.0010283, 0.0010297
5: 0.0060086, 0.0065111, 0.0060086, 0.0065111, -0.0005024, 0.0005024
6: -0.0013983, -0.0008639, -0.0013983, -0.0008639, -0.0003997, 0.0003991
7: -0.0083221, -0.0079387, -0.0083221, -0.0079387, -0.0003835, 0.0003835
8: 0.0048484, 0.0068715, 0.0048484, 0.0068715, -0.0011927, 0.0011999
9: -0.0036877, -0.0032079, -0.0036877, -0.0032079, -0.0004797, 0.0004797

### Unstable ReLU Count (Linear/Conv2D Layers)
- layer_idx=0, type=LayerType.Linear, total=256, inp1_unstable=5, inp2_unstable=5, delta_unstable=251
- layer_idx=2, type=LayerType.Linear, total=256, inp1_unstable=4, inp2_unstable=4, delta_unstable=255
- layer_idx=4, type=LayerType.Linear, total=256, inp1_unstable=0, inp2_unstable=0, delta_unstable=256
- layer_idx=6, type=LayerType.Linear, total=256, inp1_unstable=1, inp2_unstable=1, delta_unstable=256
- layer_idx=8, type=LayerType.Linear, total=10, inp1_unstable=0, inp2_unstable=0, delta_unstable=10

Time for backsubstitution: 1.30 seconds

### RS candidates at layer 3
type: RSZ, layer: 3, pos: 73
type: RSZ, layer: 3, pos: 81
type: RSZ, layer: 3, pos: 136

Time for candidate selection: 0.14 seconds

### Candidate
type: RSZ, layer: 3, pos: 73

### Candidate
type: RSZ, layer: 3, pos: 81

### Relational analysis RSZ of RS_RSZ1_RSZ2_RSZ2_RSZ1_RSZ2_RSZ1_RSZ2_RSZ1

#### Relational analysis RSZ result of RS_RSZ1_RSZ2_RSZ2_RSZ1_RSZ2_RSZ1_RSZ2_RSZ1
Status: Status.VERIFIED
Output dim: 1, lower bound: -0.0002466, upper bound: 0.0002482
time: 0.61 seconds

### Relational analysis RSZ of RS_RSZ1_RSZ2_RSZ2_RSZ1_RSZ2_RSZ1_RSZ2_RSZ2

#### Relational analysis RSZ result of RS_RSZ1_RSZ2_RSZ2_RSZ1_RSZ2_RSZ1_RSZ2_RSZ2
Status: Status.VERIFIED
Output dim: 1, lower bound: -0.0002466, upper bound: 0.0002482
time: 0.60 seconds

## BFS RS instance: RS_RSZ1_RSZ2_RSZ2_RSZ1_RSZ2_RSZ2_RSZ1

### Backsubstitution after applying RS history:
0: 0.0009457, 0.0011003, 0.0009457, 0.0011003, -0.0001486, 0.0001490
1: 0.9936197, 0.9940771, 0.9936197, 0.9940771, -0.0003887, 0.0003831
2: -0.0065606, -0.0050208, -0.0065606, -0.0050208, -0.0012030, 0.0011534
3: 0.0038004, 0.0040140, 0.0038004, 0.0040140, -0.0001622, 0.0001661
4: 0.0023852, 0.0036022, 0.0023852, 0.0036022, -0.0010164, 0.0010553
5: 0.0060086, 0.0065111, 0.0060086, 0.0065111, -0.0005024, 0.0005024
6: -0.0013983, -0.0008639, -0.0013983, -0.0008639, -0.0004111, 0.0003938
7: -0.0083221, -0.0079387, -0.0083221, -0.0079387, -0.0003835, 0.0003835
8: 0.0048484, 0.0068715, 0.0048484, 0.0068715, -0.0011728, 0.0012463
9: -0.0036877, -0.0032079, -0.0036877, -0.0032079, -0.0004797, 0.0004797

### Unstable ReLU Count (Linear/Conv2D Layers)
- layer_idx=0, type=LayerType.Linear, total=256, inp1_unstable=5, inp2_unstable=5, delta_unstable=251
- layer_idx=2, type=LayerType.Linear, total=256, inp1_unstable=4, inp2_unstable=4, delta_unstable=255
- layer_idx=4, type=LayerType.Linear, total=256, inp1_unstable=0, inp2_unstable=0, delta_unstable=256
- layer_idx=6, type=LayerType.Linear, total=256, inp1_unstable=1, inp2_unstable=1, delta_unstable=256
- layer_idx=8, type=LayerType.Linear, total=10, inp1_unstable=0, inp2_unstable=0, delta_unstable=10

Time for backsubstitution: 1.26 seconds

### RS candidates at layer 3
type: RSZ, layer: 3, pos: 73
type: RSZ, layer: 3, pos: 81
type: RSZ, layer: 3, pos: 136

Time for candidate selection: 0.15 seconds

### Candidate
type: RSZ, layer: 3, pos: 73

### Candidate
type: RSZ, layer: 3, pos: 81

### Relational analysis RSZ of RS_RSZ1_RSZ2_RSZ2_RSZ1_RSZ2_RSZ2_RSZ1_RSZ1

#### Relational analysis RSZ result of RS_RSZ1_RSZ2_RSZ2_RSZ1_RSZ2_RSZ2_RSZ1_RSZ1
Status: Status.VERIFIED
Output dim: 1, lower bound: -0.0002496, upper bound: 0.0002463
time: 0.60 seconds

### Relational analysis RSZ of RS_RSZ1_RSZ2_RSZ2_RSZ1_RSZ2_RSZ2_RSZ1_RSZ2

#### Relational analysis RSZ result of RS_RSZ1_RSZ2_RSZ2_RSZ1_RSZ2_RSZ2_RSZ1_RSZ2
Status: Status.VERIFIED
Output dim: 1, lower bound: -0.0002496, upper bound: 0.0002463
time: 0.60 seconds

## BFS RS instance: RS_RSZ1_RSZ2_RSZ2_RSZ1_RSZ2_RSZ2_RSZ2

### Backsubstitution after applying RS history:
0: 0.0009457, 0.0011003, 0.0009457, 0.0011003, -0.0001487, 0.0001488
1: 0.9936197, 0.9940771, 0.9936197, 0.9940771, -0.0003852, 0.0003845
2: -0.0065606, -0.0050208, -0.0065606, -0.0050208, -0.0011703, 0.0011685
3: 0.0038004, 0.0040140, 0.0038004, 0.0040140, -0.0001632, 0.0001637
4: 0.0023852, 0.0036022, 0.0023852, 0.0036022, -0.0010283, 0.0010297
5: 0.0060086, 0.0065111, 0.0060086, 0.0065111, -0.0005024, 0.0005024
6: -0.0013983, -0.0008639, -0.0013983, -0.0008639, -0.0003997, 0.0003991
7: -0.0083221, -0.0079387, -0.0083221, -0.0079387, -0.0003835, 0.0003835
8: 0.0048484, 0.0068715, 0.0048484, 0.0068715, -0.0011927, 0.0011999
9: -0.0036877, -0.0032079, -0.0036877, -0.0032079, -0.0004797, 0.0004797

### Unstable ReLU Count (Linear/Conv2D Layers)
- layer_idx=0, type=LayerType.Linear, total=256, inp1_unstable=5, inp2_unstable=5, delta_unstable=251
- layer_idx=2, type=LayerType.Linear, total=256, inp1_unstable=4, inp2_unstable=4, delta_unstable=255
- layer_idx=4, type=LayerType.Linear, total=256, inp1_unstable=0, inp2_unstable=0, delta_unstable=256
- layer_idx=6, type=LayerType.Linear, total=256, inp1_unstable=1, inp2_unstable=1, delta_unstable=256
- layer_idx=8, type=LayerType.Linear, total=10, inp1_unstable=0, inp2_unstable=0, delta_unstable=10

Time for backsubstitution: 1.29 seconds

### RS candidates at layer 3
type: RSZ, layer: 3, pos: 73
type: RSZ, layer: 3, pos: 81
type: RSZ, layer: 3, pos: 136

Time for candidate selection: 0.13 seconds

### Candidate
type: RSZ, layer: 3, pos: 73

### Candidate
type: RSZ, layer: 3, pos: 81

### Relational analysis RSZ of RS_RSZ1_RSZ2_RSZ2_RSZ1_RSZ2_RSZ2_RSZ2_RSZ1

#### Relational analysis RSZ result of RS_RSZ1_RSZ2_RSZ2_RSZ1_RSZ2_RSZ2_RSZ2_RSZ1
Status: Status.VERIFIED
Output dim: 1, lower bound: -0.0002466, upper bound: 0.0002482
time: 0.63 seconds

### Relational analysis RSZ of RS_RSZ1_RSZ2_RSZ2_RSZ1_RSZ2_RSZ2_RSZ2_RSZ2

#### Relational analysis RSZ result of RS_RSZ1_RSZ2_RSZ2_RSZ1_RSZ2_RSZ2_RSZ2_RSZ2
Status: Status.VERIFIED
Output dim: 1, lower bound: -0.0002466, upper bound: 0.0002482
time: 0.63 seconds

## BFS RS instance: RS_RSZ1_RSZ2_RSZ2_RSZ2_RSZ1_RSZ1_RSZ1

### Backsubstitution after applying RS history:
0: 0.0009457, 0.0011003, 0.0009457, 0.0011003, -0.0001486, 0.0001490
1: 0.9936197, 0.9940771, 0.9936197, 0.9940771, -0.0003885, 0.0003827
2: -0.0065606, -0.0050208, -0.0065606, -0.0050208, -0.0012007, 0.0011490
3: 0.0038004, 0.0040140, 0.0038004, 0.0040140, -0.0001619, 0.0001660
4: 0.0023852, 0.0036022, 0.0023852, 0.0036022, -0.0010128, 0.0010535
5: 0.0060086, 0.0065111, 0.0060086, 0.0065111, -0.0005024, 0.0005024
6: -0.0013983, -0.0008639, -0.0013983, -0.0008639, -0.0004103, 0.0003923
7: -0.0083221, -0.0079387, -0.0083221, -0.0079387, -0.0003835, 0.0003835
8: 0.0048484, 0.0068715, 0.0048484, 0.0068715, -0.0011670, 0.0012432
9: -0.0036877, -0.0032079, -0.0036877, -0.0032079, -0.0004797, 0.0004797

### Unstable ReLU Count (Linear/Conv2D Layers)
- layer_idx=0, type=LayerType.Linear, total=256, inp1_unstable=5, inp2_unstable=5, delta_unstable=251
- layer_idx=2, type=LayerType.Linear, total=256, inp1_unstable=4, inp2_unstable=4, delta_unstable=255
- layer_idx=4, type=LayerType.Linear, total=256, inp1_unstable=0, inp2_unstable=0, delta_unstable=256
- layer_idx=6, type=LayerType.Linear, total=256, inp1_unstable=1, inp2_unstable=1, delta_unstable=256
- layer_idx=8, type=LayerType.Linear, total=10, inp1_unstable=0, inp2_unstable=0, delta_unstable=10

Time for backsubstitution: 1.30 seconds

### RS candidates at layer 3
type: RSZ, layer: 3, pos: 73
type: RSZ, layer: 3, pos: 81
type: RSZ, layer: 3, pos: 136

Time for candidate selection: 0.13 seconds

### Candidate
type: RSZ, layer: 3, pos: 73

### Candidate
type: RSZ, layer: 3, pos: 81

### Relational analysis RSZ of RS_RSZ1_RSZ2_RSZ2_RSZ2_RSZ1_RSZ1_RSZ1_RSZ1

#### Relational analysis RSZ result of RS_RSZ1_RSZ2_RSZ2_RSZ2_RSZ1_RSZ1_RSZ1_RSZ1
Status: Status.VERIFIED
Output dim: 1, lower bound: -0.0002490, upper bound: 0.0002468
time: 0.60 seconds

### Relational analysis RSZ of RS_RSZ1_RSZ2_RSZ2_RSZ2_RSZ1_RSZ1_RSZ1_RSZ2

#### Relational analysis RSZ result of RS_RSZ1_RSZ2_RSZ2_RSZ2_RSZ1_RSZ1_RSZ1_RSZ2
Status: Status.VERIFIED
Output dim: 1, lower bound: -0.0002490, upper bound: 0.0002468
time: 0.61 seconds

## BFS RS instance: RS_RSZ1_RSZ2_RSZ2_RSZ2_RSZ1_RSZ1_RSZ2

### Backsubstitution after applying RS history:
0: 0.0009457, 0.0011003, 0.0009457, 0.0011003, -0.0001487, 0.0001488
1: 0.9936197, 0.9940771, 0.9936197, 0.9940771, -0.0003853, 0.0003847
2: -0.0065606, -0.0050208, -0.0065606, -0.0050208, -0.0011713, 0.0011700
3: 0.0038004, 0.0040140, 0.0038004, 0.0040140, -0.0001633, 0.0001638
4: 0.0023852, 0.0036022, 0.0023852, 0.0036022, -0.0010295, 0.0010305
5: 0.0060086, 0.0065111, 0.0060086, 0.0065111, -0.0005024, 0.0005024
6: -0.0013983, -0.0008639, -0.0013983, -0.0008639, -0.0004001, 0.0003996
7: -0.0083221, -0.0079387, -0.0083221, -0.0079387, -0.0003835, 0.0003835
8: 0.0048484, 0.0068715, 0.0048484, 0.0068715, -0.0011946, 0.0012012
9: -0.0036877, -0.0032079, -0.0036877, -0.0032079, -0.0004797, 0.0004797

### Unstable ReLU Count (Linear/Conv2D Layers)
- layer_idx=0, type=LayerType.Linear, total=256, inp1_unstable=5, inp2_unstable=5, delta_unstable=251
- layer_idx=2, type=LayerType.Linear, total=256, inp1_unstable=4, inp2_unstable=4, delta_unstable=255
- layer_idx=4, type=LayerType.Linear, total=256, inp1_unstable=0, inp2_unstable=0, delta_unstable=256
- layer_idx=6, type=LayerType.Linear, total=256, inp1_unstable=1, inp2_unstable=1, delta_unstable=256
- layer_idx=8, type=LayerType.Linear, total=10, inp1_unstable=0, inp2_unstable=0, delta_unstable=10

Time for backsubstitution: 1.31 seconds

### RS candidates at layer 3
type: RSZ, layer: 3, pos: 73
type: RSZ, layer: 3, pos: 81
type: RSZ, layer: 3, pos: 136

Time for candidate selection: 0.15 seconds

### Candidate
type: RSZ, layer: 3, pos: 73

### Candidate
type: RSZ, layer: 3, pos: 81

### Relational analysis RSZ of RS_RSZ1_RSZ2_RSZ2_RSZ2_RSZ1_RSZ1_RSZ2_RSZ1

#### Relational analysis RSZ result of RS_RSZ1_RSZ2_RSZ2_RSZ2_RSZ1_RSZ1_RSZ2_RSZ1
Status: Status.VERIFIED
Output dim: 1, lower bound: -0.0002462, upper bound: 0.0002493
time: 0.59 seconds

### Relational analysis RSZ of RS_RSZ1_RSZ2_RSZ2_RSZ2_RSZ1_RSZ1_RSZ2_RSZ2

#### Relational analysis RSZ result of RS_RSZ1_RSZ2_RSZ2_RSZ2_RSZ1_RSZ1_RSZ2_RSZ2
Status: Status.VERIFIED
Output dim: 1, lower bound: -0.0002462, upper bound: 0.0002493
time: 0.60 seconds

## BFS RS instance: RS_RSZ1_RSZ2_RSZ2_RSZ2_RSZ1_RSZ2_RSZ1

### Backsubstitution after applying RS history:
0: 0.0009457, 0.0011003, 0.0009457, 0.0011003, -0.0001486, 0.0001490
1: 0.9936197, 0.9940771, 0.9936197, 0.9940771, -0.0003885, 0.0003827
2: -0.0065606, -0.0050208, -0.0065606, -0.0050208, -0.0012007, 0.0011490
3: 0.0038004, 0.0040140, 0.0038004, 0.0040140, -0.0001619, 0.0001660
4: 0.0023852, 0.0036022, 0.0023852, 0.0036022, -0.0010128, 0.0010535
5: 0.0060086, 0.0065111, 0.0060086, 0.0065111, -0.0005024, 0.0005024
6: -0.0013983, -0.0008639, -0.0013983, -0.0008639, -0.0004103, 0.0003923
7: -0.0083221, -0.0079387, -0.0083221, -0.0079387, -0.0003835, 0.0003835
8: 0.0048484, 0.0068715, 0.0048484, 0.0068715, -0.0011670, 0.0012432
9: -0.0036877, -0.0032079, -0.0036877, -0.0032079, -0.0004797, 0.0004797

### Unstable ReLU Count (Linear/Conv2D Layers)
- layer_idx=0, type=LayerType.Linear, total=256, inp1_unstable=5, inp2_unstable=5, delta_unstable=251
- layer_idx=2, type=LayerType.Linear, total=256, inp1_unstable=4, inp2_unstable=4, delta_unstable=255
- layer_idx=4, type=LayerType.Linear, total=256, inp1_unstable=0, inp2_unstable=0, delta_unstable=256
- layer_idx=6, type=LayerType.Linear, total=256, inp1_unstable=1, inp2_unstable=1, delta_unstable=256
- layer_idx=8, type=LayerType.Linear, total=10, inp1_unstable=0, inp2_unstable=0, delta_unstable=10

Time for backsubstitution: 1.31 seconds

### RS candidates at layer 3
type: RSZ, layer: 3, pos: 73
type: RSZ, layer: 3, pos: 81
type: RSZ, layer: 3, pos: 136

Time for candidate selection: 0.14 seconds

### Candidate
type: RSZ, layer: 3, pos: 73

### Candidate
type: RSZ, layer: 3, pos: 81

### Relational analysis RSZ of RS_RSZ1_RSZ2_RSZ2_RSZ2_RSZ1_RSZ2_RSZ1_RSZ1

#### Relational analysis RSZ result of RS_RSZ1_RSZ2_RSZ2_RSZ2_RSZ1_RSZ2_RSZ1_RSZ1
Status: Status.VERIFIED
Output dim: 1, lower bound: -0.0002490, upper bound: 0.0002468
time: 0.59 seconds

### Relational analysis RSZ of RS_RSZ1_RSZ2_RSZ2_RSZ2_RSZ1_RSZ2_RSZ1_RSZ2

#### Relational analysis RSZ result of RS_RSZ1_RSZ2_RSZ2_RSZ2_RSZ1_RSZ2_RSZ1_RSZ2
Status: Status.VERIFIED
Output dim: 1, lower bound: -0.0002490, upper bound: 0.0002468
time: 0.59 seconds

## BFS RS instance: RS_RSZ1_RSZ2_RSZ2_RSZ2_RSZ1_RSZ2_RSZ2

### Backsubstitution after applying RS history:
0: 0.0009457, 0.0011003, 0.0009457, 0.0011003, -0.0001487, 0.0001488
1: 0.9936197, 0.9940771, 0.9936197, 0.9940771, -0.0003853, 0.0003847
2: -0.0065606, -0.0050208, -0.0065606, -0.0050208, -0.0011713, 0.0011700
3: 0.0038004, 0.0040140, 0.0038004, 0.0040140, -0.0001633, 0.0001638
4: 0.0023852, 0.0036022, 0.0023852, 0.0036022, -0.0010295, 0.0010305
5: 0.0060086, 0.0065111, 0.0060086, 0.0065111, -0.0005024, 0.0005024
6: -0.0013983, -0.0008639, -0.0013983, -0.0008639, -0.0004001, 0.0003996
7: -0.0083221, -0.0079387, -0.0083221, -0.0079387, -0.0003835, 0.0003835
8: 0.0048484, 0.0068715, 0.0048484, 0.0068715, -0.0011946, 0.0012012
9: -0.0036877, -0.0032079, -0.0036877, -0.0032079, -0.0004797, 0.0004797

### Unstable ReLU Count (Linear/Conv2D Layers)
- layer_idx=0, type=LayerType.Linear, total=256, inp1_unstable=5, inp2_unstable=5, delta_unstable=251
- layer_idx=2, type=LayerType.Linear, total=256, inp1_unstable=4, inp2_unstable=4, delta_unstable=255
- layer_idx=4, type=LayerType.Linear, total=256, inp1_unstable=0, inp2_unstable=0, delta_unstable=256
- layer_idx=6, type=LayerType.Linear, total=256, inp1_unstable=1, inp2_unstable=1, delta_unstable=256
- layer_idx=8, type=LayerType.Linear, total=10, inp1_unstable=0, inp2_unstable=0, delta_unstable=10

Time for backsubstitution: 1.28 seconds

### RS candidates at layer 3
type: RSZ, layer: 3, pos: 73
type: RSZ, layer: 3, pos: 81
type: RSZ, layer: 3, pos: 136

Time for candidate selection: 0.14 seconds

### Candidate
type: RSZ, layer: 3, pos: 73

### Candidate
type: RSZ, layer: 3, pos: 81

### Relational analysis RSZ of RS_RSZ1_RSZ2_RSZ2_RSZ2_RSZ1_RSZ2_RSZ2_RSZ1

#### Relational analysis RSZ result of RS_RSZ1_RSZ2_RSZ2_RSZ2_RSZ1_RSZ2_RSZ2_RSZ1
Status: Status.VERIFIED
Output dim: 1, lower bound: -0.0002462, upper bound: 0.0002493
time: 0.60 seconds

### Relational analysis RSZ of RS_RSZ1_RSZ2_RSZ2_RSZ2_RSZ1_RSZ2_RSZ2_RSZ2

#### Relational analysis RSZ result of RS_RSZ1_RSZ2_RSZ2_RSZ2_RSZ1_RSZ2_RSZ2_RSZ2
Status: Status.VERIFIED
Output dim: 1, lower bound: -0.0002462, upper bound: 0.0002493
time: 0.61 seconds

## BFS RS instance: RS_RSZ1_RSZ2_RSZ2_RSZ2_RSZ2_RSZ1_RSZ1

### Backsubstitution after applying RS history:
0: 0.0009457, 0.0011003, 0.0009457, 0.0011003, -0.0001486, 0.0001490
1: 0.9936197, 0.9940771, 0.9936197, 0.9940771, -0.0003880, 0.0003829
2: -0.0065606, -0.0050208, -0.0065606, -0.0050208, -0.0011964, 0.0011517
3: 0.0038004, 0.0040140, 0.0038004, 0.0040140, -0.0001621, 0.0001657
4: 0.0023852, 0.0036022, 0.0023852, 0.0036022, -0.0010150, 0.0010501
5: 0.0060086, 0.0065111, 0.0060086, 0.0065111, -0.0005024, 0.0005024
6: -0.0013983, -0.0008639, -0.0013983, -0.0008639, -0.0004088, 0.0003932
7: -0.0083221, -0.0079387, -0.0083221, -0.0079387, -0.0003835, 0.0003835
8: 0.0048484, 0.0068715, 0.0048484, 0.0068715, -0.0011706, 0.0012376
9: -0.0036877, -0.0032079, -0.0036877, -0.0032079, -0.0004797, 0.0004797

### Unstable ReLU Count (Linear/Conv2D Layers)
- layer_idx=0, type=LayerType.Linear, total=256, inp1_unstable=5, inp2_unstable=5, delta_unstable=251
- layer_idx=2, type=LayerType.Linear, total=256, inp1_unstable=4, inp2_unstable=4, delta_unstable=255
- layer_idx=4, type=LayerType.Linear, total=256, inp1_unstable=0, inp2_unstable=0, delta_unstable=256
- layer_idx=6, type=LayerType.Linear, total=256, inp1_unstable=1, inp2_unstable=1, delta_unstable=256
- layer_idx=8, type=LayerType.Linear, total=10, inp1_unstable=0, inp2_unstable=0, delta_unstable=10

Time for backsubstitution: 1.35 seconds

### RS candidates at layer 3
type: RSZ, layer: 3, pos: 73
type: RSZ, layer: 3, pos: 81
type: RSZ, layer: 3, pos: 136

Time for candidate selection: 0.13 seconds

### Candidate
type: RSZ, layer: 3, pos: 73

### Candidate
type: RSZ, layer: 3, pos: 81

### Relational analysis RSZ of RS_RSZ1_RSZ2_RSZ2_RSZ2_RSZ2_RSZ1_RSZ1_RSZ1

#### Relational analysis RSZ result of RS_RSZ1_RSZ2_RSZ2_RSZ2_RSZ2_RSZ1_RSZ1_RSZ1
Status: Status.VERIFIED
Output dim: 1, lower bound: -0.0002490, upper bound: 0.0002468
time: 0.62 seconds

### Relational analysis RSZ of RS_RSZ1_RSZ2_RSZ2_RSZ2_RSZ2_RSZ1_RSZ1_RSZ2

#### Relational analysis RSZ result of RS_RSZ1_RSZ2_RSZ2_RSZ2_RSZ2_RSZ1_RSZ1_RSZ2
Status: Status.VERIFIED
Output dim: 1, lower bound: -0.0002490, upper bound: 0.0002468
time: 0.62 seconds

## BFS RS instance: RS_RSZ1_RSZ2_RSZ2_RSZ2_RSZ2_RSZ1_RSZ2

### Backsubstitution after applying RS history:
0: 0.0009457, 0.0011003, 0.0009457, 0.0011003, -0.0001488, 0.0001488
1: 0.9936197, 0.9940771, 0.9936197, 0.9940771, -0.0003852, 0.0003850
2: -0.0065606, -0.0050208, -0.0065606, -0.0050208, -0.0011700, 0.0011734
3: 0.0038004, 0.0040140, 0.0038004, 0.0040140, -0.0001635, 0.0001637
4: 0.0023852, 0.0036022, 0.0023852, 0.0036022, -0.0010322, 0.0010295
5: 0.0060086, 0.0065111, 0.0060086, 0.0065111, -0.0005024, 0.0005024
6: -0.0013983, -0.0008639, -0.0013983, -0.0008639, -0.0003996, 0.0004008
7: -0.0083221, -0.0079387, -0.0083221, -0.0079387, -0.0003835, 0.0003835
8: 0.0048484, 0.0068715, 0.0048484, 0.0068715, -0.0011991, 0.0011995
9: -0.0036877, -0.0032079, -0.0036877, -0.0032079, -0.0004797, 0.0004797

### Unstable ReLU Count (Linear/Conv2D Layers)
- layer_idx=0, type=LayerType.Linear, total=256, inp1_unstable=5, inp2_unstable=5, delta_unstable=251
- layer_idx=2, type=LayerType.Linear, total=256, inp1_unstable=4, inp2_unstable=4, delta_unstable=255
- layer_idx=4, type=LayerType.Linear, total=256, inp1_unstable=0, inp2_unstable=0, delta_unstable=256
- layer_idx=6, type=LayerType.Linear, total=256, inp1_unstable=1, inp2_unstable=1, delta_unstable=256
- layer_idx=8, type=LayerType.Linear, total=10, inp1_unstable=0, inp2_unstable=0, delta_unstable=10

Time for backsubstitution: 1.31 seconds

### RS candidates at layer 3
type: RSZ, layer: 3, pos: 73
type: RSZ, layer: 3, pos: 81
type: RSZ, layer: 3, pos: 136

Time for candidate selection: 0.14 seconds

### Candidate
type: RSZ, layer: 3, pos: 73

### Candidate
type: RSZ, layer: 3, pos: 81

### Relational analysis RSZ of RS_RSZ1_RSZ2_RSZ2_RSZ2_RSZ2_RSZ1_RSZ2_RSZ1

#### Relational analysis RSZ result of RS_RSZ1_RSZ2_RSZ2_RSZ2_RSZ2_RSZ1_RSZ2_RSZ1
Status: Status.VERIFIED
Output dim: 1, lower bound: -0.0002462, upper bound: 0.0002493
time: 0.60 seconds

### Relational analysis RSZ of RS_RSZ1_RSZ2_RSZ2_RSZ2_RSZ2_RSZ1_RSZ2_RSZ2

#### Relational analysis RSZ result of RS_RSZ1_RSZ2_RSZ2_RSZ2_RSZ2_RSZ1_RSZ2_RSZ2
Status: Status.VERIFIED
Output dim: 1, lower bound: -0.0002462, upper bound: 0.0002493
time: 0.61 seconds

## BFS RS instance: RS_RSZ1_RSZ2_RSZ2_RSZ2_RSZ2_RSZ2_RSZ1

### Backsubstitution after applying RS history:
0: 0.0009457, 0.0011003, 0.0009457, 0.0011003, -0.0001486, 0.0001490
1: 0.9936197, 0.9940771, 0.9936197, 0.9940771, -0.0003880, 0.0003829
2: -0.0065606, -0.0050208, -0.0065606, -0.0050208, -0.0011964, 0.0011517
3: 0.0038004, 0.0040140, 0.0038004, 0.0040140, -0.0001621, 0.0001657
4: 0.0023852, 0.0036022, 0.0023852, 0.0036022, -0.0010150, 0.0010501
5: 0.0060086, 0.0065111, 0.0060086, 0.0065111, -0.0005024, 0.0005024
6: -0.0013983, -0.0008639, -0.0013983, -0.0008639, -0.0004088, 0.0003932
7: -0.0083221, -0.0079387, -0.0083221, -0.0079387, -0.0003835, 0.0003835
8: 0.0048484, 0.0068715, 0.0048484, 0.0068715, -0.0011706, 0.0012376
9: -0.0036877, -0.0032079, -0.0036877, -0.0032079, -0.0004797, 0.0004797

### Unstable ReLU Count (Linear/Conv2D Layers)
- layer_idx=0, type=LayerType.Linear, total=256, inp1_unstable=5, inp2_unstable=5, delta_unstable=251
- layer_idx=2, type=LayerType.Linear, total=256, inp1_unstable=4, inp2_unstable=4, delta_unstable=255
- layer_idx=4, type=LayerType.Linear, total=256, inp1_unstable=0, inp2_unstable=0, delta_unstable=256
- layer_idx=6, type=LayerType.Linear, total=256, inp1_unstable=1, inp2_unstable=1, delta_unstable=256
- layer_idx=8, type=LayerType.Linear, total=10, inp1_unstable=0, inp2_unstable=0, delta_unstable=10

Time for backsubstitution: 1.30 seconds

### RS candidates at layer 3
type: RSZ, layer: 3, pos: 73
type: RSZ, layer: 3, pos: 81
type: RSZ, layer: 3, pos: 136

Time for candidate selection: 0.13 seconds

### Candidate
type: RSZ, layer: 3, pos: 73

### Candidate
type: RSZ, layer: 3, pos: 81

### Relational analysis RSZ of RS_RSZ1_RSZ2_RSZ2_RSZ2_RSZ2_RSZ2_RSZ1_RSZ1

#### Relational analysis RSZ result of RS_RSZ1_RSZ2_RSZ2_RSZ2_RSZ2_RSZ2_RSZ1_RSZ1
Status: Status.VERIFIED
Output dim: 1, lower bound: -0.0002490, upper bound: 0.0002468
time: 0.60 seconds

### Relational analysis RSZ of RS_RSZ1_RSZ2_RSZ2_RSZ2_RSZ2_RSZ2_RSZ1_RSZ2

#### Relational analysis RSZ result of RS_RSZ1_RSZ2_RSZ2_RSZ2_RSZ2_RSZ2_RSZ1_RSZ2
Status: Status.VERIFIED
Output dim: 1, lower bound: -0.0002490, upper bound: 0.0002468
time: 0.61 seconds

## BFS RS instance: RS_RSZ1_RSZ2_RSZ2_RSZ2_RSZ2_RSZ2_RSZ2

### Backsubstitution after applying RS history:
0: 0.0009457, 0.0011003, 0.0009457, 0.0011003, -0.0001488, 0.0001488
1: 0.9936197, 0.9940771, 0.9936197, 0.9940771, -0.0003852, 0.0003850
2: -0.0065606, -0.0050208, -0.0065606, -0.0050208, -0.0011700, 0.0011734
3: 0.0038004, 0.0040140, 0.0038004, 0.0040140, -0.0001635, 0.0001637
4: 0.0023852, 0.0036022, 0.0023852, 0.0036022, -0.0010322, 0.0010295
5: 0.0060086, 0.0065111, 0.0060086, 0.0065111, -0.0005024, 0.0005024
6: -0.0013983, -0.0008639, -0.0013983, -0.0008639, -0.0003996, 0.0004008
7: -0.0083221, -0.0079387, -0.0083221, -0.0079387, -0.0003835, 0.0003835
8: 0.0048484, 0.0068715, 0.0048484, 0.0068715, -0.0011991, 0.0011995
9: -0.0036877, -0.0032079, -0.0036877, -0.0032079, -0.0004797, 0.0004797

### Unstable ReLU Count (Linear/Conv2D Layers)
- layer_idx=0, type=LayerType.Linear, total=256, inp1_unstable=5, inp2_unstable=5, delta_unstable=251
- layer_idx=2, type=LayerType.Linear, total=256, inp1_unstable=4, inp2_unstable=4, delta_unstable=255
- layer_idx=4, type=LayerType.Linear, total=256, inp1_unstable=0, inp2_unstable=0, delta_unstable=256
- layer_idx=6, type=LayerType.Linear, total=256, inp1_unstable=1, inp2_unstable=1, delta_unstable=256
- layer_idx=8, type=LayerType.Linear, total=10, inp1_unstable=0, inp2_unstable=0, delta_unstable=10

Time for backsubstitution: 1.31 seconds

### RS candidates at layer 3
type: RSZ, layer: 3, pos: 73
type: RSZ, layer: 3, pos: 81
type: RSZ, layer: 3, pos: 136

Time for candidate selection: 0.14 seconds

### Candidate
type: RSZ, layer: 3, pos: 73

### Candidate
type: RSZ, layer: 3, pos: 81

### Relational analysis RSZ of RS_RSZ1_RSZ2_RSZ2_RSZ2_RSZ2_RSZ2_RSZ2_RSZ1

#### Relational analysis RSZ result of RS_RSZ1_RSZ2_RSZ2_RSZ2_RSZ2_RSZ2_RSZ2_RSZ1
Status: Status.VERIFIED
Output dim: 1, lower bound: -0.0002462, upper bound: 0.0002493
time: 0.63 seconds

### Relational analysis RSZ of RS_RSZ1_RSZ2_RSZ2_RSZ2_RSZ2_RSZ2_RSZ2_RSZ2

#### Relational analysis RSZ result of RS_RSZ1_RSZ2_RSZ2_RSZ2_RSZ2_RSZ2_RSZ2_RSZ2
Status: Status.VERIFIED
Output dim: 1, lower bound: -0.0002462, upper bound: 0.0002493
time: 0.65 seconds

## BFS RS instance: RS_RSZ2_RSZ1_RSZ1_RSZ1_RSZ1_RSZ1_RSZ1

### Backsubstitution after applying RS history:
0: 0.0009457, 0.0011003, 0.0009457, 0.0011003, -0.0001487, 0.0001489
1: 0.9936197, 0.9940771, 0.9936197, 0.9940771, -0.0003866, 0.0003839
2: -0.0065606, -0.0050208, -0.0065606, -0.0050208, -0.0011894, 0.0011625
3: 0.0038004, 0.0040140, 0.0038004, 0.0040140, -0.0001628, 0.0001647
4: 0.0023852, 0.0036022, 0.0023852, 0.0036022, -0.0010235, 0.0010448
5: 0.0060086, 0.0065111, 0.0060086, 0.0065111, -0.0005024, 0.0005024
6: -0.0013983, -0.0008639, -0.0013983, -0.0008639, -0.0004063, 0.0003970
7: -0.0083221, -0.0079387, -0.0083221, -0.0079387, -0.0003835, 0.0003835
8: 0.0048484, 0.0068715, 0.0048484, 0.0068715, -0.0011848, 0.0012215
9: -0.0036877, -0.0032079, -0.0036877, -0.0032079, -0.0004797, 0.0004797

### Unstable ReLU Count (Linear/Conv2D Layers)
- layer_idx=0, type=LayerType.Linear, total=256, inp1_unstable=5, inp2_unstable=5, delta_unstable=251
- layer_idx=2, type=LayerType.Linear, total=256, inp1_unstable=4, inp2_unstable=4, delta_unstable=255
- layer_idx=4, type=LayerType.Linear, total=256, inp1_unstable=0, inp2_unstable=0, delta_unstable=256
- layer_idx=6, type=LayerType.Linear, total=256, inp1_unstable=1, inp2_unstable=1, delta_unstable=256
- layer_idx=8, type=LayerType.Linear, total=10, inp1_unstable=0, inp2_unstable=0, delta_unstable=10

Time for backsubstitution: 1.35 seconds

### RS candidates at layer 3
type: RSZ, layer: 3, pos: 73
type: RSZ, layer: 3, pos: 81
type: RSZ, layer: 3, pos: 136

Time for candidate selection: 0.14 seconds

### Candidate
type: RSZ, layer: 3, pos: 73

### Candidate
type: RSZ, layer: 3, pos: 81

### Relational analysis RSZ of RS_RSZ2_RSZ1_RSZ1_RSZ1_RSZ1_RSZ1_RSZ1_RSZ1

#### Relational analysis RSZ result of RS_RSZ2_RSZ1_RSZ1_RSZ1_RSZ1_RSZ1_RSZ1_RSZ1
Status: Status.VERIFIED
Output dim: 1, lower bound: -0.0002493, upper bound: 0.0002462
time: 0.62 seconds

### Relational analysis RSZ of RS_RSZ2_RSZ1_RSZ1_RSZ1_RSZ1_RSZ1_RSZ1_RSZ2

#### Relational analysis RSZ result of RS_RSZ2_RSZ1_RSZ1_RSZ1_RSZ1_RSZ1_RSZ1_RSZ2
Status: Status.VERIFIED
Output dim: 1, lower bound: -0.0002493, upper bound: 0.0002462
time: 0.61 seconds

## BFS RS instance: RS_RSZ2_RSZ1_RSZ1_RSZ1_RSZ1_RSZ1_RSZ2

### Backsubstitution after applying RS history:
0: 0.0009457, 0.0011003, 0.0009457, 0.0011003, -0.0001488, 0.0001487
1: 0.9936197, 0.9940771, 0.9936197, 0.9940771, -0.0003849, 0.0003860
2: -0.0065606, -0.0050208, -0.0065606, -0.0050208, -0.0011696, 0.0011844
3: 0.0038004, 0.0040140, 0.0038004, 0.0040140, -0.0001643, 0.0001635
4: 0.0023852, 0.0036022, 0.0023852, 0.0036022, -0.0010408, 0.0010292
5: 0.0060086, 0.0065111, 0.0060086, 0.0065111, -0.0005024, 0.0005024
6: -0.0013983, -0.0008639, -0.0013983, -0.0008639, -0.0003995, 0.0004046
7: -0.0083221, -0.0079387, -0.0083221, -0.0079387, -0.0003835, 0.0003835
8: 0.0048484, 0.0068715, 0.0048484, 0.0068715, -0.0012135, 0.0011968
9: -0.0036877, -0.0032079, -0.0036877, -0.0032079, -0.0004797, 0.0004797

### Unstable ReLU Count (Linear/Conv2D Layers)
- layer_idx=0, type=LayerType.Linear, total=256, inp1_unstable=5, inp2_unstable=5, delta_unstable=251
- layer_idx=2, type=LayerType.Linear, total=256, inp1_unstable=4, inp2_unstable=4, delta_unstable=255
- layer_idx=4, type=LayerType.Linear, total=256, inp1_unstable=0, inp2_unstable=0, delta_unstable=256
- layer_idx=6, type=LayerType.Linear, total=256, inp1_unstable=1, inp2_unstable=1, delta_unstable=256
- layer_idx=8, type=LayerType.Linear, total=10, inp1_unstable=0, inp2_unstable=0, delta_unstable=10

Time for backsubstitution: 1.32 seconds

### RS candidates at layer 3
type: RSZ, layer: 3, pos: 73
type: RSZ, layer: 3, pos: 81
type: RSZ, layer: 3, pos: 136

Time for candidate selection: 0.14 seconds

### Candidate
type: RSZ, layer: 3, pos: 73

### Candidate
type: RSZ, layer: 3, pos: 81

### Relational analysis RSZ of RS_RSZ2_RSZ1_RSZ1_RSZ1_RSZ1_RSZ1_RSZ2_RSZ1

#### Relational analysis RSZ result of RS_RSZ2_RSZ1_RSZ1_RSZ1_RSZ1_RSZ1_RSZ2_RSZ1
Status: Status.VERIFIED
Output dim: 1, lower bound: -0.0002468, upper bound: 0.0002490
time: 0.59 seconds

### Relational analysis RSZ of RS_RSZ2_RSZ1_RSZ1_RSZ1_RSZ1_RSZ1_RSZ2_RSZ2

#### Relational analysis RSZ result of RS_RSZ2_RSZ1_RSZ1_RSZ1_RSZ1_RSZ1_RSZ2_RSZ2
Status: Status.VERIFIED
Output dim: 1, lower bound: -0.0002468, upper bound: 0.0002490
time: 0.60 seconds

## BFS RS instance: RS_RSZ2_RSZ1_RSZ1_RSZ1_RSZ1_RSZ2_RSZ1

### Backsubstitution after applying RS history:
0: 0.0009457, 0.0011003, 0.0009457, 0.0011003, -0.0001487, 0.0001489
1: 0.9936197, 0.9940771, 0.9936197, 0.9940771, -0.0003866, 0.0003839
2: -0.0065606, -0.0050208, -0.0065606, -0.0050208, -0.0011894, 0.0011625
3: 0.0038004, 0.0040140, 0.0038004, 0.0040140, -0.0001628, 0.0001647
4: 0.0023852, 0.0036022, 0.0023852, 0.0036022, -0.0010235, 0.0010448
5: 0.0060086, 0.0065111, 0.0060086, 0.0065111, -0.0005024, 0.0005024
6: -0.0013983, -0.0008639, -0.0013983, -0.0008639, -0.0004063, 0.0003970
7: -0.0083221, -0.0079387, -0.0083221, -0.0079387, -0.0003835, 0.0003835
8: 0.0048484, 0.0068715, 0.0048484, 0.0068715, -0.0011848, 0.0012215
9: -0.0036877, -0.0032079, -0.0036877, -0.0032079, -0.0004797, 0.0004797

### Unstable ReLU Count (Linear/Conv2D Layers)
- layer_idx=0, type=LayerType.Linear, total=256, inp1_unstable=5, inp2_unstable=5, delta_unstable=251
- layer_idx=2, type=LayerType.Linear, total=256, inp1_unstable=4, inp2_unstable=4, delta_unstable=255
- layer_idx=4, type=LayerType.Linear, total=256, inp1_unstable=0, inp2_unstable=0, delta_unstable=256
- layer_idx=6, type=LayerType.Linear, total=256, inp1_unstable=1, inp2_unstable=1, delta_unstable=256
- layer_idx=8, type=LayerType.Linear, total=10, inp1_unstable=0, inp2_unstable=0, delta_unstable=10

Time for backsubstitution: 1.32 seconds

### RS candidates at layer 3
type: RSZ, layer: 3, pos: 73
type: RSZ, layer: 3, pos: 81
type: RSZ, layer: 3, pos: 136

Time for candidate selection: 0.15 seconds

### Candidate
type: RSZ, layer: 3, pos: 73

### Candidate
type: RSZ, layer: 3, pos: 81

### Relational analysis RSZ of RS_RSZ2_RSZ1_RSZ1_RSZ1_RSZ1_RSZ2_RSZ1_RSZ1

#### Relational analysis RSZ result of RS_RSZ2_RSZ1_RSZ1_RSZ1_RSZ1_RSZ2_RSZ1_RSZ1
Status: Status.VERIFIED
Output dim: 1, lower bound: -0.0002493, upper bound: 0.0002462
time: 0.58 seconds

### Relational analysis RSZ of RS_RSZ2_RSZ1_RSZ1_RSZ1_RSZ1_RSZ2_RSZ1_RSZ2

#### Relational analysis RSZ result of RS_RSZ2_RSZ1_RSZ1_RSZ1_RSZ1_RSZ2_RSZ1_RSZ2
Status: Status.VERIFIED
Output dim: 1, lower bound: -0.0002493, upper bound: 0.0002462
time: 0.59 seconds

## BFS RS instance: RS_RSZ2_RSZ1_RSZ1_RSZ1_RSZ1_RSZ2_RSZ2

### Backsubstitution after applying RS history:
0: 0.0009457, 0.0011003, 0.0009457, 0.0011003, -0.0001488, 0.0001487
1: 0.9936197, 0.9940771, 0.9936197, 0.9940771, -0.0003849, 0.0003860
2: -0.0065606, -0.0050208, -0.0065606, -0.0050208, -0.0011696, 0.0011844
3: 0.0038004, 0.0040140, 0.0038004, 0.0040140, -0.0001643, 0.0001635
4: 0.0023852, 0.0036022, 0.0023852, 0.0036022, -0.0010408, 0.0010292
5: 0.0060086, 0.0065111, 0.0060086, 0.0065111, -0.0005024, 0.0005024
6: -0.0013983, -0.0008639, -0.0013983, -0.0008639, -0.0003995, 0.0004046
7: -0.0083221, -0.0079387, -0.0083221, -0.0079387, -0.0003835, 0.0003835
8: 0.0048484, 0.0068715, 0.0048484, 0.0068715, -0.0012135, 0.0011968
9: -0.0036877, -0.0032079, -0.0036877, -0.0032079, -0.0004797, 0.0004797

### Unstable ReLU Count (Linear/Conv2D Layers)
- layer_idx=0, type=LayerType.Linear, total=256, inp1_unstable=5, inp2_unstable=5, delta_unstable=251
- layer_idx=2, type=LayerType.Linear, total=256, inp1_unstable=4, inp2_unstable=4, delta_unstable=255
- layer_idx=4, type=LayerType.Linear, total=256, inp1_unstable=0, inp2_unstable=0, delta_unstable=256
- layer_idx=6, type=LayerType.Linear, total=256, inp1_unstable=1, inp2_unstable=1, delta_unstable=256
- layer_idx=8, type=LayerType.Linear, total=10, inp1_unstable=0, inp2_unstable=0, delta_unstable=10

Time for backsubstitution: 1.31 seconds

### RS candidates at layer 3
type: RSZ, layer: 3, pos: 73
type: RSZ, layer: 3, pos: 81
type: RSZ, layer: 3, pos: 136

Time for candidate selection: 0.13 seconds

### Candidate
type: RSZ, layer: 3, pos: 73

### Candidate
type: RSZ, layer: 3, pos: 81

### Relational analysis RSZ of RS_RSZ2_RSZ1_RSZ1_RSZ1_RSZ1_RSZ2_RSZ2_RSZ1

#### Relational analysis RSZ result of RS_RSZ2_RSZ1_RSZ1_RSZ1_RSZ1_RSZ2_RSZ2_RSZ1
Status: Status.VERIFIED
Output dim: 1, lower bound: -0.0002468, upper bound: 0.0002490
time: 0.61 seconds

### Relational analysis RSZ of RS_RSZ2_RSZ1_RSZ1_RSZ1_RSZ1_RSZ2_RSZ2_RSZ2

#### Relational analysis RSZ result of RS_RSZ2_RSZ1_RSZ1_RSZ1_RSZ1_RSZ2_RSZ2_RSZ2
Status: Status.VERIFIED
Output dim: 1, lower bound: -0.0002468, upper bound: 0.0002490
time: 0.59 seconds

## BFS RS instance: RS_RSZ2_RSZ1_RSZ1_RSZ1_RSZ2_RSZ1_RSZ1

### Backsubstitution after applying RS history:
0: 0.0009457, 0.0011003, 0.0009457, 0.0011003, -0.0001487, 0.0001489
1: 0.9936197, 0.9940771, 0.9936197, 0.9940771, -0.0003863, 0.0003841
2: -0.0065606, -0.0050208, -0.0065606, -0.0050208, -0.0011857, 0.0011640
3: 0.0038004, 0.0040140, 0.0038004, 0.0040140, -0.0001629, 0.0001644
4: 0.0023852, 0.0036022, 0.0023852, 0.0036022, -0.0010247, 0.0010419
5: 0.0060086, 0.0065111, 0.0060086, 0.0065111, -0.0005024, 0.0005024
6: -0.0013983, -0.0008639, -0.0013983, -0.0008639, -0.0004051, 0.0003975
7: -0.0083221, -0.0079387, -0.0083221, -0.0079387, -0.0003835, 0.0003835
8: 0.0048484, 0.0068715, 0.0048484, 0.0068715, -0.0011867, 0.0012166
9: -0.0036877, -0.0032079, -0.0036877, -0.0032079, -0.0004797, 0.0004797

### Unstable ReLU Count (Linear/Conv2D Layers)
- layer_idx=0, type=LayerType.Linear, total=256, inp1_unstable=5, inp2_unstable=5, delta_unstable=251
- layer_idx=2, type=LayerType.Linear, total=256, inp1_unstable=4, inp2_unstable=4, delta_unstable=255
- layer_idx=4, type=LayerType.Linear, total=256, inp1_unstable=0, inp2_unstable=0, delta_unstable=256
- layer_idx=6, type=LayerType.Linear, total=256, inp1_unstable=1, inp2_unstable=1, delta_unstable=256
- layer_idx=8, type=LayerType.Linear, total=10, inp1_unstable=0, inp2_unstable=0, delta_unstable=10

Time for backsubstitution: 1.32 seconds

### RS candidates at layer 3
type: RSZ, layer: 3, pos: 73
type: RSZ, layer: 3, pos: 81
type: RSZ, layer: 3, pos: 136

Time for candidate selection: 0.14 seconds

### Candidate
type: RSZ, layer: 3, pos: 73

### Candidate
type: RSZ, layer: 3, pos: 81

### Relational analysis RSZ of RS_RSZ2_RSZ1_RSZ1_RSZ1_RSZ2_RSZ1_RSZ1_RSZ1

#### Relational analysis RSZ result of RS_RSZ2_RSZ1_RSZ1_RSZ1_RSZ2_RSZ1_RSZ1_RSZ1
Status: Status.VERIFIED
Output dim: 1, lower bound: -0.0002493, upper bound: 0.0002462
time: 0.64 seconds

### Relational analysis RSZ of RS_RSZ2_RSZ1_RSZ1_RSZ1_RSZ2_RSZ1_RSZ1_RSZ2

#### Relational analysis RSZ result of RS_RSZ2_RSZ1_RSZ1_RSZ1_RSZ2_RSZ1_RSZ1_RSZ2
Status: Status.VERIFIED
Output dim: 1, lower bound: -0.0002493, upper bound: 0.0002462
time: 0.64 seconds

## BFS RS instance: RS_RSZ2_RSZ1_RSZ1_RSZ1_RSZ2_RSZ1_RSZ2

### Backsubstitution after applying RS history:
0: 0.0009457, 0.0011003, 0.0009457, 0.0011003, -0.0001489, 0.0001487
1: 0.9936197, 0.9940771, 0.9936197, 0.9940771, -0.0003846, 0.0003864
2: -0.0065606, -0.0050208, -0.0065606, -0.0050208, -0.0011667, 0.0011880
3: 0.0038004, 0.0040140, 0.0038004, 0.0040140, -0.0001645, 0.0001633
4: 0.0023852, 0.0036022, 0.0023852, 0.0036022, -0.0010437, 0.0010269
5: 0.0060086, 0.0065111, 0.0060086, 0.0065111, -0.0005024, 0.0005024
6: -0.0013983, -0.0008639, -0.0013983, -0.0008639, -0.0003984, 0.0004059
7: -0.0083221, -0.0079387, -0.0083221, -0.0079387, -0.0003835, 0.0003835
8: 0.0048484, 0.0068715, 0.0048484, 0.0068715, -0.0012183, 0.0011930
9: -0.0036877, -0.0032079, -0.0036877, -0.0032079, -0.0004797, 0.0004797

### Unstable ReLU Count (Linear/Conv2D Layers)
- layer_idx=0, type=LayerType.Linear, total=256, inp1_unstable=5, inp2_unstable=5, delta_unstable=251
- layer_idx=2, type=LayerType.Linear, total=256, inp1_unstable=4, inp2_unstable=4, delta_unstable=255
- layer_idx=4, type=LayerType.Linear, total=256, inp1_unstable=0, inp2_unstable=0, delta_unstable=256
- layer_idx=6, type=LayerType.Linear, total=256, inp1_unstable=1, inp2_unstable=1, delta_unstable=256
- layer_idx=8, type=LayerType.Linear, total=10, inp1_unstable=0, inp2_unstable=0, delta_unstable=10

Time for backsubstitution: 1.32 seconds

### RS candidates at layer 3
type: RSZ, layer: 3, pos: 73
type: RSZ, layer: 3, pos: 81
type: RSZ, layer: 3, pos: 136

Time for candidate selection: 0.14 seconds

### Candidate
type: RSZ, layer: 3, pos: 73

### Candidate
type: RSZ, layer: 3, pos: 81

### Relational analysis RSZ of RS_RSZ2_RSZ1_RSZ1_RSZ1_RSZ2_RSZ1_RSZ2_RSZ1

#### Relational analysis RSZ result of RS_RSZ2_RSZ1_RSZ1_RSZ1_RSZ2_RSZ1_RSZ2_RSZ1
Status: Status.VERIFIED
Output dim: 1, lower bound: -0.0002468, upper bound: 0.0002490
time: 0.60 seconds

### Relational analysis RSZ of RS_RSZ2_RSZ1_RSZ1_RSZ1_RSZ2_RSZ1_RSZ2_RSZ2

#### Relational analysis RSZ result of RS_RSZ2_RSZ1_RSZ1_RSZ1_RSZ2_RSZ1_RSZ2_RSZ2
Status: Status.VERIFIED
Output dim: 1, lower bound: -0.0002468, upper bound: 0.0002490
time: 0.62 seconds

## BFS RS instance: RS_RSZ2_RSZ1_RSZ1_RSZ1_RSZ2_RSZ2_RSZ1

### Backsubstitution after applying RS history:
0: 0.0009457, 0.0011003, 0.0009457, 0.0011003, -0.0001487, 0.0001489
1: 0.9936197, 0.9940771, 0.9936197, 0.9940771, -0.0003863, 0.0003841
2: -0.0065606, -0.0050208, -0.0065606, -0.0050208, -0.0011857, 0.0011640
3: 0.0038004, 0.0040140, 0.0038004, 0.0040140, -0.0001629, 0.0001644
4: 0.0023852, 0.0036022, 0.0023852, 0.0036022, -0.0010247, 0.0010419
5: 0.0060086, 0.0065111, 0.0060086, 0.0065111, -0.0005024, 0.0005024
6: -0.0013983, -0.0008639, -0.0013983, -0.0008639, -0.0004051, 0.0003975
7: -0.0083221, -0.0079387, -0.0083221, -0.0079387, -0.0003835, 0.0003835
8: 0.0048484, 0.0068715, 0.0048484, 0.0068715, -0.0011867, 0.0012166
9: -0.0036877, -0.0032079, -0.0036877, -0.0032079, -0.0004797, 0.0004797

### Unstable ReLU Count (Linear/Conv2D Layers)
- layer_idx=0, type=LayerType.Linear, total=256, inp1_unstable=5, inp2_unstable=5, delta_unstable=251
- layer_idx=2, type=LayerType.Linear, total=256, inp1_unstable=4, inp2_unstable=4, delta_unstable=255
- layer_idx=4, type=LayerType.Linear, total=256, inp1_unstable=0, inp2_unstable=0, delta_unstable=256
- layer_idx=6, type=LayerType.Linear, total=256, inp1_unstable=1, inp2_unstable=1, delta_unstable=256
- layer_idx=8, type=LayerType.Linear, total=10, inp1_unstable=0, inp2_unstable=0, delta_unstable=10

Time for backsubstitution: 1.34 seconds

### RS candidates at layer 3
type: RSZ, layer: 3, pos: 73
type: RSZ, layer: 3, pos: 81
type: RSZ, layer: 3, pos: 136

Time for candidate selection: 0.15 seconds

### Candidate
type: RSZ, layer: 3, pos: 73

### Candidate
type: RSZ, layer: 3, pos: 81

### Relational analysis RSZ of RS_RSZ2_RSZ1_RSZ1_RSZ1_RSZ2_RSZ2_RSZ1_RSZ1

#### Relational analysis RSZ result of RS_RSZ2_RSZ1_RSZ1_RSZ1_RSZ2_RSZ2_RSZ1_RSZ1
Status: Status.VERIFIED
Output dim: 1, lower bound: -0.0002493, upper bound: 0.0002462
time: 0.62 seconds

### Relational analysis RSZ of RS_RSZ2_RSZ1_RSZ1_RSZ1_RSZ2_RSZ2_RSZ1_RSZ2

#### Relational analysis RSZ result of RS_RSZ2_RSZ1_RSZ1_RSZ1_RSZ2_RSZ2_RSZ1_RSZ2
Status: Status.VERIFIED
Output dim: 1, lower bound: -0.0002493, upper bound: 0.0002462
time: 0.66 seconds

## BFS RS instance: RS_RSZ2_RSZ1_RSZ1_RSZ1_RSZ2_RSZ2_RSZ2

### Backsubstitution after applying RS history:
0: 0.0009457, 0.0011003, 0.0009457, 0.0011003, -0.0001489, 0.0001487
1: 0.9936197, 0.9940771, 0.9936197, 0.9940771, -0.0003846, 0.0003864
2: -0.0065606, -0.0050208, -0.0065606, -0.0050208, -0.0011667, 0.0011880
3: 0.0038004, 0.0040140, 0.0038004, 0.0040140, -0.0001645, 0.0001633
4: 0.0023852, 0.0036022, 0.0023852, 0.0036022, -0.0010437, 0.0010269
5: 0.0060086, 0.0065111, 0.0060086, 0.0065111, -0.0005024, 0.0005024
6: -0.0013983, -0.0008639, -0.0013983, -0.0008639, -0.0003984, 0.0004059
7: -0.0083221, -0.0079387, -0.0083221, -0.0079387, -0.0003835, 0.0003835
8: 0.0048484, 0.0068715, 0.0048484, 0.0068715, -0.0012183, 0.0011930
9: -0.0036877, -0.0032079, -0.0036877, -0.0032079, -0.0004797, 0.0004797

### Unstable ReLU Count (Linear/Conv2D Layers)
- layer_idx=0, type=LayerType.Linear, total=256, inp1_unstable=5, inp2_unstable=5, delta_unstable=251
- layer_idx=2, type=LayerType.Linear, total=256, inp1_unstable=4, inp2_unstable=4, delta_unstable=255
- layer_idx=4, type=LayerType.Linear, total=256, inp1_unstable=0, inp2_unstable=0, delta_unstable=256
- layer_idx=6, type=LayerType.Linear, total=256, inp1_unstable=1, inp2_unstable=1, delta_unstable=256
- layer_idx=8, type=LayerType.Linear, total=10, inp1_unstable=0, inp2_unstable=0, delta_unstable=10

Time for backsubstitution: 1.47 seconds

### RS candidates at layer 3
type: RSZ, layer: 3, pos: 73
type: RSZ, layer: 3, pos: 81
type: RSZ, layer: 3, pos: 136

Time for candidate selection: 0.17 seconds

### Candidate
type: RSZ, layer: 3, pos: 73

### Candidate
type: RSZ, layer: 3, pos: 81

### Relational analysis RSZ of RS_RSZ2_RSZ1_RSZ1_RSZ1_RSZ2_RSZ2_RSZ2_RSZ1

#### Relational analysis RSZ result of RS_RSZ2_RSZ1_RSZ1_RSZ1_RSZ2_RSZ2_RSZ2_RSZ1
Status: Status.VERIFIED
Output dim: 1, lower bound: -0.0002468, upper bound: 0.0002490
time: 0.67 seconds

### Relational analysis RSZ of RS_RSZ2_RSZ1_RSZ1_RSZ1_RSZ2_RSZ2_RSZ2_RSZ2

#### Relational analysis RSZ result of RS_RSZ2_RSZ1_RSZ1_RSZ1_RSZ2_RSZ2_RSZ2_RSZ2
Status: Status.VERIFIED
Output dim: 1, lower bound: -0.0002468, upper bound: 0.0002490
time: 0.63 seconds

## BFS RS instance: RS_RSZ2_RSZ1_RSZ1_RSZ2_RSZ1_RSZ1_RSZ1

### Backsubstitution after applying RS history:
0: 0.0009457, 0.0011003, 0.0009457, 0.0011003, -0.0001487, 0.0001488
1: 0.9936197, 0.9940771, 0.9936197, 0.9940771, -0.0003862, 0.0003838
2: -0.0065606, -0.0050208, -0.0065606, -0.0050208, -0.0011845, 0.0011611
3: 0.0038004, 0.0040140, 0.0038004, 0.0040140, -0.0001627, 0.0001644
4: 0.0023852, 0.0036022, 0.0023852, 0.0036022, -0.0010224, 0.0010410
5: 0.0060086, 0.0065111, 0.0060086, 0.0065111, -0.0005024, 0.0005024
6: -0.0013983, -0.0008639, -0.0013983, -0.0008639, -0.0004046, 0.0003965
7: -0.0083221, -0.0079387, -0.0083221, -0.0079387, -0.0003835, 0.0003835
8: 0.0048484, 0.0068715, 0.0048484, 0.0068715, -0.0011829, 0.0012151
9: -0.0036877, -0.0032079, -0.0036877, -0.0032079, -0.0004797, 0.0004797

### Unstable ReLU Count (Linear/Conv2D Layers)
- layer_idx=0, type=LayerType.Linear, total=256, inp1_unstable=5, inp2_unstable=5, delta_unstable=251
- layer_idx=2, type=LayerType.Linear, total=256, inp1_unstable=4, inp2_unstable=4, delta_unstable=255
- layer_idx=4, type=LayerType.Linear, total=256, inp1_unstable=0, inp2_unstable=0, delta_unstable=256
- layer_idx=6, type=LayerType.Linear, total=256, inp1_unstable=1, inp2_unstable=1, delta_unstable=256
- layer_idx=8, type=LayerType.Linear, total=10, inp1_unstable=0, inp2_unstable=0, delta_unstable=10

Time for backsubstitution: 1.35 seconds

### RS candidates at layer 3
type: RSZ, layer: 3, pos: 73
type: RSZ, layer: 3, pos: 81
type: RSZ, layer: 3, pos: 136

Time for candidate selection: 0.15 seconds

### Candidate
type: RSZ, layer: 3, pos: 73

### Candidate
type: RSZ, layer: 3, pos: 81

### Relational analysis RSZ of RS_RSZ2_RSZ1_RSZ1_RSZ2_RSZ1_RSZ1_RSZ1_RSZ1

#### Relational analysis RSZ result of RS_RSZ2_RSZ1_RSZ1_RSZ2_RSZ1_RSZ1_RSZ1_RSZ1
Status: Status.VERIFIED
Output dim: 1, lower bound: -0.0002482, upper bound: 0.0002466
time: 0.58 seconds

### Relational analysis RSZ of RS_RSZ2_RSZ1_RSZ1_RSZ2_RSZ1_RSZ1_RSZ1_RSZ2

#### Relational analysis RSZ result of RS_RSZ2_RSZ1_RSZ1_RSZ2_RSZ1_RSZ1_RSZ1_RSZ2
Status: Status.VERIFIED
Output dim: 1, lower bound: -0.0002482, upper bound: 0.0002466
time: 0.59 seconds

## BFS RS instance: RS_RSZ2_RSZ1_RSZ1_RSZ2_RSZ1_RSZ1_RSZ2

### Backsubstitution after applying RS history:
0: 0.0009457, 0.0011003, 0.0009457, 0.0011003, -0.0001489, 0.0001488
1: 0.9936197, 0.9940771, 0.9936197, 0.9940771, -0.0003850, 0.0003864
2: -0.0065606, -0.0050208, -0.0065606, -0.0050208, -0.0011713, 0.0011887
3: 0.0038004, 0.0040140, 0.0038004, 0.0040140, -0.0001646, 0.0001636
4: 0.0023852, 0.0036022, 0.0023852, 0.0036022, -0.0010443, 0.0010305
5: 0.0060086, 0.0065111, 0.0060086, 0.0065111, -0.0005024, 0.0005024
6: -0.0013983, -0.0008639, -0.0013983, -0.0008639, -0.0004000, 0.0004061
7: -0.0083221, -0.0079387, -0.0083221, -0.0079387, -0.0003835, 0.0003835
8: 0.0048484, 0.0068715, 0.0048484, 0.0068715, -0.0012192, 0.0011990
9: -0.0036877, -0.0032079, -0.0036877, -0.0032079, -0.0004797, 0.0004797

### Unstable ReLU Count (Linear/Conv2D Layers)
- layer_idx=0, type=LayerType.Linear, total=256, inp1_unstable=5, inp2_unstable=5, delta_unstable=251
- layer_idx=2, type=LayerType.Linear, total=256, inp1_unstable=4, inp2_unstable=4, delta_unstable=255
- layer_idx=4, type=LayerType.Linear, total=256, inp1_unstable=0, inp2_unstable=0, delta_unstable=256
- layer_idx=6, type=LayerType.Linear, total=256, inp1_unstable=1, inp2_unstable=1, delta_unstable=256
- layer_idx=8, type=LayerType.Linear, total=10, inp1_unstable=0, inp2_unstable=0, delta_unstable=10

Time for backsubstitution: 1.33 seconds

### RS candidates at layer 3
type: RSZ, layer: 3, pos: 73
type: RSZ, layer: 3, pos: 81
type: RSZ, layer: 3, pos: 136

Time for candidate selection: 0.14 seconds

### Candidate
type: RSZ, layer: 3, pos: 73

### Candidate
type: RSZ, layer: 3, pos: 81

### Relational analysis RSZ of RS_RSZ2_RSZ1_RSZ1_RSZ2_RSZ1_RSZ1_RSZ2_RSZ1

#### Relational analysis RSZ result of RS_RSZ2_RSZ1_RSZ1_RSZ2_RSZ1_RSZ1_RSZ2_RSZ1
Status: Status.VERIFIED
Output dim: 1, lower bound: -0.0002463, upper bound: 0.0002496
time: 0.58 seconds

### Relational analysis RSZ of RS_RSZ2_RSZ1_RSZ1_RSZ2_RSZ1_RSZ1_RSZ2_RSZ2

#### Relational analysis RSZ result of RS_RSZ2_RSZ1_RSZ1_RSZ2_RSZ1_RSZ1_RSZ2_RSZ2
Status: Status.VERIFIED
Output dim: 1, lower bound: -0.0002463, upper bound: 0.0002496
time: 0.59 seconds

## BFS RS instance: RS_RSZ2_RSZ1_RSZ1_RSZ2_RSZ1_RSZ2_RSZ1

### Backsubstitution after applying RS history:
0: 0.0009457, 0.0011003, 0.0009457, 0.0011003, -0.0001487, 0.0001488
1: 0.9936197, 0.9940771, 0.9936197, 0.9940771, -0.0003862, 0.0003838
2: -0.0065606, -0.0050208, -0.0065606, -0.0050208, -0.0011845, 0.0011611
3: 0.0038004, 0.0040140, 0.0038004, 0.0040140, -0.0001627, 0.0001644
4: 0.0023852, 0.0036022, 0.0023852, 0.0036022, -0.0010224, 0.0010410
5: 0.0060086, 0.0065111, 0.0060086, 0.0065111, -0.0005024, 0.0005024
6: -0.0013983, -0.0008639, -0.0013983, -0.0008639, -0.0004046, 0.0003965
7: -0.0083221, -0.0079387, -0.0083221, -0.0079387, -0.0003835, 0.0003835
8: 0.0048484, 0.0068715, 0.0048484, 0.0068715, -0.0011829, 0.0012151
9: -0.0036877, -0.0032079, -0.0036877, -0.0032079, -0.0004797, 0.0004797

### Unstable ReLU Count (Linear/Conv2D Layers)
- layer_idx=0, type=LayerType.Linear, total=256, inp1_unstable=5, inp2_unstable=5, delta_unstable=251
- layer_idx=2, type=LayerType.Linear, total=256, inp1_unstable=4, inp2_unstable=4, delta_unstable=255
- layer_idx=4, type=LayerType.Linear, total=256, inp1_unstable=0, inp2_unstable=0, delta_unstable=256
- layer_idx=6, type=LayerType.Linear, total=256, inp1_unstable=1, inp2_unstable=1, delta_unstable=256
- layer_idx=8, type=LayerType.Linear, total=10, inp1_unstable=0, inp2_unstable=0, delta_unstable=10

Time for backsubstitution: 1.37 seconds

### RS candidates at layer 3
type: RSZ, layer: 3, pos: 73
type: RSZ, layer: 3, pos: 81
type: RSZ, layer: 3, pos: 136

Time for candidate selection: 0.15 seconds

### Candidate
type: RSZ, layer: 3, pos: 73

### Candidate
type: RSZ, layer: 3, pos: 81

### Relational analysis RSZ of RS_RSZ2_RSZ1_RSZ1_RSZ2_RSZ1_RSZ2_RSZ1_RSZ1

#### Relational analysis RSZ result of RS_RSZ2_RSZ1_RSZ1_RSZ2_RSZ1_RSZ2_RSZ1_RSZ1
Status: Status.VERIFIED
Output dim: 1, lower bound: -0.0002482, upper bound: 0.0002466
time: 0.58 seconds

### Relational analysis RSZ of RS_RSZ2_RSZ1_RSZ1_RSZ2_RSZ1_RSZ2_RSZ1_RSZ2

#### Relational analysis RSZ result of RS_RSZ2_RSZ1_RSZ1_RSZ2_RSZ1_RSZ2_RSZ1_RSZ2
Status: Status.VERIFIED
Output dim: 1, lower bound: -0.0002482, upper bound: 0.0002466
time: 0.59 seconds

## BFS RS instance: RS_RSZ2_RSZ1_RSZ1_RSZ2_RSZ1_RSZ2_RSZ2

### Backsubstitution after applying RS history:
0: 0.0009457, 0.0011003, 0.0009457, 0.0011003, -0.0001489, 0.0001488
1: 0.9936197, 0.9940771, 0.9936197, 0.9940771, -0.0003850, 0.0003864
2: -0.0065606, -0.0050208, -0.0065606, -0.0050208, -0.0011713, 0.0011887
3: 0.0038004, 0.0040140, 0.0038004, 0.0040140, -0.0001646, 0.0001636
4: 0.0023852, 0.0036022, 0.0023852, 0.0036022, -0.0010443, 0.0010305
5: 0.0060086, 0.0065111, 0.0060086, 0.0065111, -0.0005024, 0.0005024
6: -0.0013983, -0.0008639, -0.0013983, -0.0008639, -0.0004000, 0.0004061
7: -0.0083221, -0.0079387, -0.0083221, -0.0079387, -0.0003835, 0.0003835
8: 0.0048484, 0.0068715, 0.0048484, 0.0068715, -0.0012192, 0.0011990
9: -0.0036877, -0.0032079, -0.0036877, -0.0032079, -0.0004797, 0.0004797

### Unstable ReLU Count (Linear/Conv2D Layers)
- layer_idx=0, type=LayerType.Linear, total=256, inp1_unstable=5, inp2_unstable=5, delta_unstable=251
- layer_idx=2, type=LayerType.Linear, total=256, inp1_unstable=4, inp2_unstable=4, delta_unstable=255
- layer_idx=4, type=LayerType.Linear, total=256, inp1_unstable=0, inp2_unstable=0, delta_unstable=256
- layer_idx=6, type=LayerType.Linear, total=256, inp1_unstable=1, inp2_unstable=1, delta_unstable=256
- layer_idx=8, type=LayerType.Linear, total=10, inp1_unstable=0, inp2_unstable=0, delta_unstable=10

Time for backsubstitution: 1.34 seconds

### RS candidates at layer 3
type: RSZ, layer: 3, pos: 73
type: RSZ, layer: 3, pos: 81
type: RSZ, layer: 3, pos: 136

Time for candidate selection: 0.14 seconds

### Candidate
type: RSZ, layer: 3, pos: 73

### Candidate
type: RSZ, layer: 3, pos: 81

### Relational analysis RSZ of RS_RSZ2_RSZ1_RSZ1_RSZ2_RSZ1_RSZ2_RSZ2_RSZ1

#### Relational analysis RSZ result of RS_RSZ2_RSZ1_RSZ1_RSZ2_RSZ1_RSZ2_RSZ2_RSZ1
Status: Status.VERIFIED
Output dim: 1, lower bound: -0.0002463, upper bound: 0.0002496
time: 0.61 seconds

### Relational analysis RSZ of RS_RSZ2_RSZ1_RSZ1_RSZ2_RSZ1_RSZ2_RSZ2_RSZ2

#### Relational analysis RSZ result of RS_RSZ2_RSZ1_RSZ1_RSZ2_RSZ1_RSZ2_RSZ2_RSZ2
Status: Status.VERIFIED
Output dim: 1, lower bound: -0.0002463, upper bound: 0.0002496
time: 0.61 seconds

## BFS RS instance: RS_RSZ2_RSZ1_RSZ1_RSZ2_RSZ2_RSZ1_RSZ1

### Backsubstitution after applying RS history:
0: 0.0009457, 0.0011003, 0.0009457, 0.0011003, -0.0001487, 0.0001488
1: 0.9936197, 0.9940771, 0.9936197, 0.9940771, -0.0003858, 0.0003841
2: -0.0065606, -0.0050208, -0.0065606, -0.0050208, -0.0011804, 0.0011642
3: 0.0038004, 0.0040140, 0.0038004, 0.0040140, -0.0001629, 0.0001641
4: 0.0023852, 0.0036022, 0.0023852, 0.0036022, -0.0010249, 0.0010377
5: 0.0060086, 0.0065111, 0.0060086, 0.0065111, -0.0005024, 0.0005024
6: -0.0013983, -0.0008639, -0.0013983, -0.0008639, -0.0004032, 0.0003976
7: -0.0083221, -0.0079387, -0.0083221, -0.0079387, -0.0003835, 0.0003835
8: 0.0048484, 0.0068715, 0.0048484, 0.0068715, -0.0011871, 0.0012096
9: -0.0036877, -0.0032079, -0.0036877, -0.0032079, -0.0004797, 0.0004797

### Unstable ReLU Count (Linear/Conv2D Layers)
- layer_idx=0, type=LayerType.Linear, total=256, inp1_unstable=5, inp2_unstable=5, delta_unstable=251
- layer_idx=2, type=LayerType.Linear, total=256, inp1_unstable=4, inp2_unstable=4, delta_unstable=255
- layer_idx=4, type=LayerType.Linear, total=256, inp1_unstable=0, inp2_unstable=0, delta_unstable=256
- layer_idx=6, type=LayerType.Linear, total=256, inp1_unstable=1, inp2_unstable=1, delta_unstable=256
- layer_idx=8, type=LayerType.Linear, total=10, inp1_unstable=0, inp2_unstable=0, delta_unstable=10

Time for backsubstitution: 1.33 seconds

### RS candidates at layer 3
type: RSZ, layer: 3, pos: 73
type: RSZ, layer: 3, pos: 81
type: RSZ, layer: 3, pos: 136

Time for candidate selection: 0.14 seconds

### Candidate
type: RSZ, layer: 3, pos: 73

### Candidate
type: RSZ, layer: 3, pos: 81

### Relational analysis RSZ of RS_RSZ2_RSZ1_RSZ1_RSZ2_RSZ2_RSZ1_RSZ1_RSZ1

#### Relational analysis RSZ result of RS_RSZ2_RSZ1_RSZ1_RSZ2_RSZ2_RSZ1_RSZ1_RSZ1
Status: Status.VERIFIED
Output dim: 1, lower bound: -0.0002481, upper bound: 0.0002467
time: 0.64 seconds

### Relational analysis RSZ of RS_RSZ2_RSZ1_RSZ1_RSZ2_RSZ2_RSZ1_RSZ1_RSZ2

#### Relational analysis RSZ result of RS_RSZ2_RSZ1_RSZ1_RSZ2_RSZ2_RSZ1_RSZ1_RSZ2
Status: Status.VERIFIED
Output dim: 1, lower bound: -0.0002481, upper bound: 0.0002467
time: 0.64 seconds

## BFS RS instance: RS_RSZ2_RSZ1_RSZ1_RSZ2_RSZ2_RSZ1_RSZ2

### Backsubstitution after applying RS history:
0: 0.0009457, 0.0011003, 0.0009457, 0.0011003, -0.0001489, 0.0001487
1: 0.9936197, 0.9940771, 0.9936197, 0.9940771, -0.0003849, 0.0003867
2: -0.0065606, -0.0050208, -0.0065606, -0.0050208, -0.0011697, 0.0011916
3: 0.0038004, 0.0040140, 0.0038004, 0.0040140, -0.0001647, 0.0001635
4: 0.0023852, 0.0036022, 0.0023852, 0.0036022, -0.0010465, 0.0010293
5: 0.0060086, 0.0065111, 0.0060086, 0.0065111, -0.0005024, 0.0005024
6: -0.0013983, -0.0008639, -0.0013983, -0.0008639, -0.0003995, 0.0004071
7: -0.0083221, -0.0079387, -0.0083221, -0.0079387, -0.0003835, 0.0003835
8: 0.0048484, 0.0068715, 0.0048484, 0.0068715, -0.0012230, 0.0011969
9: -0.0036877, -0.0032079, -0.0036877, -0.0032079, -0.0004797, 0.0004797

### Unstable ReLU Count (Linear/Conv2D Layers)
- layer_idx=0, type=LayerType.Linear, total=256, inp1_unstable=5, inp2_unstable=5, delta_unstable=251
- layer_idx=2, type=LayerType.Linear, total=256, inp1_unstable=4, inp2_unstable=4, delta_unstable=255
- layer_idx=4, type=LayerType.Linear, total=256, inp1_unstable=0, inp2_unstable=0, delta_unstable=256
- layer_idx=6, type=LayerType.Linear, total=256, inp1_unstable=1, inp2_unstable=1, delta_unstable=256
- layer_idx=8, type=LayerType.Linear, total=10, inp1_unstable=0, inp2_unstable=0, delta_unstable=10

Time for backsubstitution: 1.32 seconds

### RS candidates at layer 3
type: RSZ, layer: 3, pos: 73
type: RSZ, layer: 3, pos: 81
type: RSZ, layer: 3, pos: 136

Time for candidate selection: 0.15 seconds

### Candidate
type: RSZ, layer: 3, pos: 73

### Candidate
type: RSZ, layer: 3, pos: 81

### Relational analysis RSZ of RS_RSZ2_RSZ1_RSZ1_RSZ2_RSZ2_RSZ1_RSZ2_RSZ1

#### Relational analysis RSZ result of RS_RSZ2_RSZ1_RSZ1_RSZ2_RSZ2_RSZ1_RSZ2_RSZ1
Status: Status.VERIFIED
Output dim: 1, lower bound: -0.0002463, upper bound: 0.0002496
time: 0.60 seconds

### Relational analysis RSZ of RS_RSZ2_RSZ1_RSZ1_RSZ2_RSZ2_RSZ1_RSZ2_RSZ2

#### Relational analysis RSZ result of RS_RSZ2_RSZ1_RSZ1_RSZ2_RSZ2_RSZ1_RSZ2_RSZ2
Status: Status.VERIFIED
Output dim: 1, lower bound: -0.0002463, upper bound: 0.0002496
time: 0.62 seconds

## BFS RS instance: RS_RSZ2_RSZ1_RSZ1_RSZ2_RSZ2_RSZ2_RSZ1

### Backsubstitution after applying RS history:
0: 0.0009457, 0.0011003, 0.0009457, 0.0011003, -0.0001487, 0.0001488
1: 0.9936197, 0.9940771, 0.9936197, 0.9940771, -0.0003858, 0.0003841
2: -0.0065606, -0.0050208, -0.0065606, -0.0050208, -0.0011804, 0.0011642
3: 0.0038004, 0.0040140, 0.0038004, 0.0040140, -0.0001629, 0.0001641
4: 0.0023852, 0.0036022, 0.0023852, 0.0036022, -0.0010249, 0.0010377
5: 0.0060086, 0.0065111, 0.0060086, 0.0065111, -0.0005024, 0.0005024
6: -0.0013983, -0.0008639, -0.0013983, -0.0008639, -0.0004032, 0.0003976
7: -0.0083221, -0.0079387, -0.0083221, -0.0079387, -0.0003835, 0.0003835
8: 0.0048484, 0.0068715, 0.0048484, 0.0068715, -0.0011871, 0.0012096
9: -0.0036877, -0.0032079, -0.0036877, -0.0032079, -0.0004797, 0.0004797

### Unstable ReLU Count (Linear/Conv2D Layers)
- layer_idx=0, type=LayerType.Linear, total=256, inp1_unstable=5, inp2_unstable=5, delta_unstable=251
- layer_idx=2, type=LayerType.Linear, total=256, inp1_unstable=4, inp2_unstable=4, delta_unstable=255
- layer_idx=4, type=LayerType.Linear, total=256, inp1_unstable=0, inp2_unstable=0, delta_unstable=256
- layer_idx=6, type=LayerType.Linear, total=256, inp1_unstable=1, inp2_unstable=1, delta_unstable=256
- layer_idx=8, type=LayerType.Linear, total=10, inp1_unstable=0, inp2_unstable=0, delta_unstable=10

Time for backsubstitution: 1.35 seconds

### RS candidates at layer 3
type: RSZ, layer: 3, pos: 73
type: RSZ, layer: 3, pos: 81
type: RSZ, layer: 3, pos: 136

Time for candidate selection: 0.14 seconds

### Candidate
type: RSZ, layer: 3, pos: 73

### Candidate
type: RSZ, layer: 3, pos: 81

### Relational analysis RSZ of RS_RSZ2_RSZ1_RSZ1_RSZ2_RSZ2_RSZ2_RSZ1_RSZ1

#### Relational analysis RSZ result of RS_RSZ2_RSZ1_RSZ1_RSZ2_RSZ2_RSZ2_RSZ1_RSZ1
Status: Status.VERIFIED
Output dim: 1, lower bound: -0.0002481, upper bound: 0.0002467
time: 0.67 seconds

### Relational analysis RSZ of RS_RSZ2_RSZ1_RSZ1_RSZ2_RSZ2_RSZ2_RSZ1_RSZ2

#### Relational analysis RSZ result of RS_RSZ2_RSZ1_RSZ1_RSZ2_RSZ2_RSZ2_RSZ1_RSZ2
Status: Status.VERIFIED
Output dim: 1, lower bound: -0.0002481, upper bound: 0.0002467
time: 0.69 seconds

## BFS RS instance: RS_RSZ2_RSZ1_RSZ1_RSZ2_RSZ2_RSZ2_RSZ2

### Backsubstitution after applying RS history:
0: 0.0009457, 0.0011003, 0.0009457, 0.0011003, -0.0001489, 0.0001487
1: 0.9936197, 0.9940771, 0.9936197, 0.9940771, -0.0003849, 0.0003867
2: -0.0065606, -0.0050208, -0.0065606, -0.0050208, -0.0011697, 0.0011916
3: 0.0038004, 0.0040140, 0.0038004, 0.0040140, -0.0001647, 0.0001635
4: 0.0023852, 0.0036022, 0.0023852, 0.0036022, -0.0010465, 0.0010293
5: 0.0060086, 0.0065111, 0.0060086, 0.0065111, -0.0005024, 0.0005024
6: -0.0013983, -0.0008639, -0.0013983, -0.0008639, -0.0003995, 0.0004071
7: -0.0083221, -0.0079387, -0.0083221, -0.0079387, -0.0003835, 0.0003835
8: 0.0048484, 0.0068715, 0.0048484, 0.0068715, -0.0012230, 0.0011969
9: -0.0036877, -0.0032079, -0.0036877, -0.0032079, -0.0004797, 0.0004797

### Unstable ReLU Count (Linear/Conv2D Layers)
- layer_idx=0, type=LayerType.Linear, total=256, inp1_unstable=5, inp2_unstable=5, delta_unstable=251
- layer_idx=2, type=LayerType.Linear, total=256, inp1_unstable=4, inp2_unstable=4, delta_unstable=255
- layer_idx=4, type=LayerType.Linear, total=256, inp1_unstable=0, inp2_unstable=0, delta_unstable=256
- layer_idx=6, type=LayerType.Linear, total=256, inp1_unstable=1, inp2_unstable=1, delta_unstable=256
- layer_idx=8, type=LayerType.Linear, total=10, inp1_unstable=0, inp2_unstable=0, delta_unstable=10

Time for backsubstitution: 1.32 seconds

### RS candidates at layer 3
type: RSZ, layer: 3, pos: 73
type: RSZ, layer: 3, pos: 81
type: RSZ, layer: 3, pos: 136

Time for candidate selection: 0.13 seconds

### Candidate
type: RSZ, layer: 3, pos: 73

### Candidate
type: RSZ, layer: 3, pos: 81

### Relational analysis RSZ of RS_RSZ2_RSZ1_RSZ1_RSZ2_RSZ2_RSZ2_RSZ2_RSZ1

#### Relational analysis RSZ result of RS_RSZ2_RSZ1_RSZ1_RSZ2_RSZ2_RSZ2_RSZ2_RSZ1
Status: Status.VERIFIED
Output dim: 1, lower bound: -0.0002463, upper bound: 0.0002496
time: 0.61 seconds

### Relational analysis RSZ of RS_RSZ2_RSZ1_RSZ1_RSZ2_RSZ2_RSZ2_RSZ2_RSZ2

#### Relational analysis RSZ result of RS_RSZ2_RSZ1_RSZ1_RSZ2_RSZ2_RSZ2_RSZ2_RSZ2
Status: Status.VERIFIED
Output dim: 1, lower bound: -0.0002463, upper bound: 0.0002496
time: 0.62 seconds

## BFS RS instance: RS_RSZ2_RSZ1_RSZ2_RSZ1_RSZ1_RSZ1_RSZ1

### Backsubstitution after applying RS history:
0: 0.0009457, 0.0011003, 0.0009457, 0.0011003, -0.0001487, 0.0001489
1: 0.9936197, 0.9940771, 0.9936197, 0.9940771, -0.0003874, 0.0003838
2: -0.0065606, -0.0050208, -0.0065606, -0.0050208, -0.0011981, 0.0011614
3: 0.0038004, 0.0040140, 0.0038004, 0.0040140, -0.0001627, 0.0001653
4: 0.0023852, 0.0036022, 0.0023852, 0.0036022, -0.0010227, 0.0010517
5: 0.0060086, 0.0065111, 0.0060086, 0.0065111, -0.0005024, 0.0005024
6: -0.0013983, -0.0008639, -0.0013983, -0.0008639, -0.0004094, 0.0003966
7: -0.0083221, -0.0079387, -0.0083221, -0.0079387, -0.0003835, 0.0003835
8: 0.0048484, 0.0068715, 0.0048484, 0.0068715, -0.0011833, 0.0012330
9: -0.0036877, -0.0032079, -0.0036877, -0.0032079, -0.0004797, 0.0004797

### Unstable ReLU Count (Linear/Conv2D Layers)
- layer_idx=0, type=LayerType.Linear, total=256, inp1_unstable=5, inp2_unstable=5, delta_unstable=251
- layer_idx=2, type=LayerType.Linear, total=256, inp1_unstable=4, inp2_unstable=4, delta_unstable=255
- layer_idx=4, type=LayerType.Linear, total=256, inp1_unstable=0, inp2_unstable=0, delta_unstable=256
- layer_idx=6, type=LayerType.Linear, total=256, inp1_unstable=1, inp2_unstable=1, delta_unstable=256
- layer_idx=8, type=LayerType.Linear, total=10, inp1_unstable=0, inp2_unstable=0, delta_unstable=10

Time for backsubstitution: 1.27 seconds

### RS candidates at layer 3
type: RSZ, layer: 3, pos: 73
type: RSZ, layer: 3, pos: 81
type: RSZ, layer: 3, pos: 136

Time for candidate selection: 0.14 seconds

### Candidate
type: RSZ, layer: 3, pos: 73

### Candidate
type: RSZ, layer: 3, pos: 81

### Relational analysis RSZ of RS_RSZ2_RSZ1_RSZ2_RSZ1_RSZ1_RSZ1_RSZ1_RSZ1

#### Relational analysis RSZ result of RS_RSZ2_RSZ1_RSZ2_RSZ1_RSZ1_RSZ1_RSZ1_RSZ1
Status: Status.VERIFIED
Output dim: 1, lower bound: -0.0002495, upper bound: 0.0002463
time: 0.62 seconds

### Relational analysis RSZ of RS_RSZ2_RSZ1_RSZ2_RSZ1_RSZ1_RSZ1_RSZ1_RSZ2

#### Relational analysis RSZ result of RS_RSZ2_RSZ1_RSZ2_RSZ1_RSZ1_RSZ1_RSZ1_RSZ2
Status: Status.VERIFIED
Output dim: 1, lower bound: -0.0002495, upper bound: 0.0002463
time: 0.61 seconds

## BFS RS instance: RS_RSZ2_RSZ1_RSZ2_RSZ1_RSZ1_RSZ1_RSZ2

### Backsubstitution after applying RS history:
0: 0.0009457, 0.0011003, 0.0009457, 0.0011003, -0.0001488, 0.0001488
1: 0.9936197, 0.9940771, 0.9936197, 0.9940771, -0.0003852, 0.0003854
2: -0.0065606, -0.0050208, -0.0065606, -0.0050208, -0.0011733, 0.0011779
3: 0.0038004, 0.0040140, 0.0038004, 0.0040140, -0.0001638, 0.0001637
4: 0.0023852, 0.0036022, 0.0023852, 0.0036022, -0.0010357, 0.0010321
5: 0.0060086, 0.0065111, 0.0060086, 0.0065111, -0.0005024, 0.0005024
6: -0.0013983, -0.0008639, -0.0013983, -0.0008639, -0.0004008, 0.0004023
7: -0.0083221, -0.0079387, -0.0083221, -0.0079387, -0.0003835, 0.0003835
8: 0.0048484, 0.0068715, 0.0048484, 0.0068715, -0.0012050, 0.0012017
9: -0.0036877, -0.0032079, -0.0036877, -0.0032079, -0.0004797, 0.0004797

### Unstable ReLU Count (Linear/Conv2D Layers)
- layer_idx=0, type=LayerType.Linear, total=256, inp1_unstable=5, inp2_unstable=5, delta_unstable=251
- layer_idx=2, type=LayerType.Linear, total=256, inp1_unstable=4, inp2_unstable=4, delta_unstable=255
- layer_idx=4, type=LayerType.Linear, total=256, inp1_unstable=0, inp2_unstable=0, delta_unstable=256
- layer_idx=6, type=LayerType.Linear, total=256, inp1_unstable=1, inp2_unstable=1, delta_unstable=256
- layer_idx=8, type=LayerType.Linear, total=10, inp1_unstable=0, inp2_unstable=0, delta_unstable=10

Time for backsubstitution: 1.32 seconds

### RS candidates at layer 3
type: RSZ, layer: 3, pos: 73
type: RSZ, layer: 3, pos: 81
type: RSZ, layer: 3, pos: 136

Time for candidate selection: 0.13 seconds

### Candidate
type: RSZ, layer: 3, pos: 73

### Candidate
type: RSZ, layer: 3, pos: 81

### Relational analysis RSZ of RS_RSZ2_RSZ1_RSZ2_RSZ1_RSZ1_RSZ1_RSZ2_RSZ1

#### Relational analysis RSZ result of RS_RSZ2_RSZ1_RSZ2_RSZ1_RSZ1_RSZ1_RSZ2_RSZ1
Status: Status.VERIFIED
Output dim: 1, lower bound: -0.0002467, upper bound: 0.0002486
time: 0.62 seconds

### Relational analysis RSZ of RS_RSZ2_RSZ1_RSZ2_RSZ1_RSZ1_RSZ1_RSZ2_RSZ2

#### Relational analysis RSZ result of RS_RSZ2_RSZ1_RSZ2_RSZ1_RSZ1_RSZ1_RSZ2_RSZ2
Status: Status.VERIFIED
Output dim: 1, lower bound: -0.0002467, upper bound: 0.0002486
time: 0.63 seconds

## BFS RS instance: RS_RSZ2_RSZ1_RSZ2_RSZ1_RSZ1_RSZ2_RSZ1

### Backsubstitution after applying RS history:
0: 0.0009457, 0.0011003, 0.0009457, 0.0011003, -0.0001487, 0.0001489
1: 0.9936197, 0.9940771, 0.9936197, 0.9940771, -0.0003874, 0.0003838
2: -0.0065606, -0.0050208, -0.0065606, -0.0050208, -0.0011981, 0.0011614
3: 0.0038004, 0.0040140, 0.0038004, 0.0040140, -0.0001627, 0.0001653
4: 0.0023852, 0.0036022, 0.0023852, 0.0036022, -0.0010227, 0.0010517
5: 0.0060086, 0.0065111, 0.0060086, 0.0065111, -0.0005024, 0.0005024
6: -0.0013983, -0.0008639, -0.0013983, -0.0008639, -0.0004094, 0.0003966
7: -0.0083221, -0.0079387, -0.0083221, -0.0079387, -0.0003835, 0.0003835
8: 0.0048484, 0.0068715, 0.0048484, 0.0068715, -0.0011833, 0.0012330
9: -0.0036877, -0.0032079, -0.0036877, -0.0032079, -0.0004797, 0.0004797

### Unstable ReLU Count (Linear/Conv2D Layers)
- layer_idx=0, type=LayerType.Linear, total=256, inp1_unstable=5, inp2_unstable=5, delta_unstable=251
- layer_idx=2, type=LayerType.Linear, total=256, inp1_unstable=4, inp2_unstable=4, delta_unstable=255
- layer_idx=4, type=LayerType.Linear, total=256, inp1_unstable=0, inp2_unstable=0, delta_unstable=256
- layer_idx=6, type=LayerType.Linear, total=256, inp1_unstable=1, inp2_unstable=1, delta_unstable=256
- layer_idx=8, type=LayerType.Linear, total=10, inp1_unstable=0, inp2_unstable=0, delta_unstable=10

Time for backsubstitution: 1.31 seconds

### RS candidates at layer 3
type: RSZ, layer: 3, pos: 73
type: RSZ, layer: 3, pos: 81
type: RSZ, layer: 3, pos: 136

Time for candidate selection: 0.15 seconds

### Candidate
type: RSZ, layer: 3, pos: 73

### Candidate
type: RSZ, layer: 3, pos: 81

### Relational analysis RSZ of RS_RSZ2_RSZ1_RSZ2_RSZ1_RSZ1_RSZ2_RSZ1_RSZ1

#### Relational analysis RSZ result of RS_RSZ2_RSZ1_RSZ2_RSZ1_RSZ1_RSZ2_RSZ1_RSZ1
Status: Status.VERIFIED
Output dim: 1, lower bound: -0.0002495, upper bound: 0.0002463
time: 0.60 seconds

### Relational analysis RSZ of RS_RSZ2_RSZ1_RSZ2_RSZ1_RSZ1_RSZ2_RSZ1_RSZ2

#### Relational analysis RSZ result of RS_RSZ2_RSZ1_RSZ2_RSZ1_RSZ1_RSZ2_RSZ1_RSZ2
Status: Status.VERIFIED
Output dim: 1, lower bound: -0.0002495, upper bound: 0.0002463
time: 0.59 seconds

## BFS RS instance: RS_RSZ2_RSZ1_RSZ2_RSZ1_RSZ1_RSZ2_RSZ2

### Backsubstitution after applying RS history:
0: 0.0009457, 0.0011003, 0.0009457, 0.0011003, -0.0001488, 0.0001488
1: 0.9936197, 0.9940771, 0.9936197, 0.9940771, -0.0003852, 0.0003854
2: -0.0065606, -0.0050208, -0.0065606, -0.0050208, -0.0011733, 0.0011779
3: 0.0038004, 0.0040140, 0.0038004, 0.0040140, -0.0001638, 0.0001637
4: 0.0023852, 0.0036022, 0.0023852, 0.0036022, -0.0010357, 0.0010321
5: 0.0060086, 0.0065111, 0.0060086, 0.0065111, -0.0005024, 0.0005024
6: -0.0013983, -0.0008639, -0.0013983, -0.0008639, -0.0004008, 0.0004023
7: -0.0083221, -0.0079387, -0.0083221, -0.0079387, -0.0003835, 0.0003835
8: 0.0048484, 0.0068715, 0.0048484, 0.0068715, -0.0012050, 0.0012017
9: -0.0036877, -0.0032079, -0.0036877, -0.0032079, -0.0004797, 0.0004797

### Unstable ReLU Count (Linear/Conv2D Layers)
- layer_idx=0, type=LayerType.Linear, total=256, inp1_unstable=5, inp2_unstable=5, delta_unstable=251
- layer_idx=2, type=LayerType.Linear, total=256, inp1_unstable=4, inp2_unstable=4, delta_unstable=255
- layer_idx=4, type=LayerType.Linear, total=256, inp1_unstable=0, inp2_unstable=0, delta_unstable=256
- layer_idx=6, type=LayerType.Linear, total=256, inp1_unstable=1, inp2_unstable=1, delta_unstable=256
- layer_idx=8, type=LayerType.Linear, total=10, inp1_unstable=0, inp2_unstable=0, delta_unstable=10

Time for backsubstitution: 1.37 seconds

### RS candidates at layer 3
type: RSZ, layer: 3, pos: 73
type: RSZ, layer: 3, pos: 81
type: RSZ, layer: 3, pos: 136

Time for candidate selection: 0.15 seconds

### Candidate
type: RSZ, layer: 3, pos: 73

### Candidate
type: RSZ, layer: 3, pos: 81

### Relational analysis RSZ of RS_RSZ2_RSZ1_RSZ2_RSZ1_RSZ1_RSZ2_RSZ2_RSZ1

#### Relational analysis RSZ result of RS_RSZ2_RSZ1_RSZ2_RSZ1_RSZ1_RSZ2_RSZ2_RSZ1
Status: Status.VERIFIED
Output dim: 1, lower bound: -0.0002467, upper bound: 0.0002486
time: 0.60 seconds

### Relational analysis RSZ of RS_RSZ2_RSZ1_RSZ2_RSZ1_RSZ1_RSZ2_RSZ2_RSZ2

#### Relational analysis RSZ result of RS_RSZ2_RSZ1_RSZ2_RSZ1_RSZ1_RSZ2_RSZ2_RSZ2
Status: Status.VERIFIED
Output dim: 1, lower bound: -0.0002467, upper bound: 0.0002486
time: 0.61 seconds

## BFS RS instance: RS_RSZ2_RSZ1_RSZ2_RSZ1_RSZ2_RSZ1_RSZ1

### Backsubstitution after applying RS history:
0: 0.0009457, 0.0011003, 0.0009457, 0.0011003, -0.0001487, 0.0001489
1: 0.9936197, 0.9940771, 0.9936197, 0.9940771, -0.0003871, 0.0003840
2: -0.0065606, -0.0050208, -0.0065606, -0.0050208, -0.0011943, 0.0011632
3: 0.0038004, 0.0040140, 0.0038004, 0.0040140, -0.0001629, 0.0001650
4: 0.0023852, 0.0036022, 0.0023852, 0.0036022, -0.0010241, 0.0010487
5: 0.0060086, 0.0065111, 0.0060086, 0.0065111, -0.0005024, 0.0005024
6: -0.0013983, -0.0008639, -0.0013983, -0.0008639, -0.0004080, 0.0003972
7: -0.0083221, -0.0079387, -0.0083221, -0.0079387, -0.0003835, 0.0003835
8: 0.0048484, 0.0068715, 0.0048484, 0.0068715, -0.0011857, 0.0012279
9: -0.0036877, -0.0032079, -0.0036877, -0.0032079, -0.0004797, 0.0004797

### Unstable ReLU Count (Linear/Conv2D Layers)
- layer_idx=0, type=LayerType.Linear, total=256, inp1_unstable=5, inp2_unstable=5, delta_unstable=251
- layer_idx=2, type=LayerType.Linear, total=256, inp1_unstable=4, inp2_unstable=4, delta_unstable=255
- layer_idx=4, type=LayerType.Linear, total=256, inp1_unstable=0, inp2_unstable=0, delta_unstable=256
- layer_idx=6, type=LayerType.Linear, total=256, inp1_unstable=1, inp2_unstable=1, delta_unstable=256
- layer_idx=8, type=LayerType.Linear, total=10, inp1_unstable=0, inp2_unstable=0, delta_unstable=10

Time for backsubstitution: 1.33 seconds

### RS candidates at layer 3
type: RSZ, layer: 3, pos: 73
type: RSZ, layer: 3, pos: 81
type: RSZ, layer: 3, pos: 136

Time for candidate selection: 0.13 seconds

### Candidate
type: RSZ, layer: 3, pos: 73

### Candidate
type: RSZ, layer: 3, pos: 81

### Relational analysis RSZ of RS_RSZ2_RSZ1_RSZ2_RSZ1_RSZ2_RSZ1_RSZ1_RSZ1

#### Relational analysis RSZ result of RS_RSZ2_RSZ1_RSZ2_RSZ1_RSZ2_RSZ1_RSZ1_RSZ1
Status: Status.VERIFIED
Output dim: 1, lower bound: -0.0002495, upper bound: 0.0002463
time: 0.59 seconds

### Relational analysis RSZ of RS_RSZ2_RSZ1_RSZ2_RSZ1_RSZ2_RSZ1_RSZ1_RSZ2

#### Relational analysis RSZ result of RS_RSZ2_RSZ1_RSZ2_RSZ1_RSZ2_RSZ1_RSZ1_RSZ2
Status: Status.VERIFIED
Output dim: 1, lower bound: -0.0002495, upper bound: 0.0002463
time: 0.60 seconds

## BFS RS instance: RS_RSZ2_RSZ1_RSZ2_RSZ1_RSZ2_RSZ1_RSZ2

### Backsubstitution after applying RS history:
0: 0.0009457, 0.0011003, 0.0009457, 0.0011003, -0.0001488, 0.0001487
1: 0.9936197, 0.9940771, 0.9936197, 0.9940771, -0.0003849, 0.0003858
2: -0.0065606, -0.0050208, -0.0065606, -0.0050208, -0.0011703, 0.0011817
3: 0.0038004, 0.0040140, 0.0038004, 0.0040140, -0.0001641, 0.0001635
4: 0.0023852, 0.0036022, 0.0023852, 0.0036022, -0.0010387, 0.0010298
5: 0.0060086, 0.0065111, 0.0060086, 0.0065111, -0.0005024, 0.0005024
6: -0.0013983, -0.0008639, -0.0013983, -0.0008639, -0.0003997, 0.0004037
7: -0.0083221, -0.0079387, -0.0083221, -0.0079387, -0.0003835, 0.0003835
8: 0.0048484, 0.0068715, 0.0048484, 0.0068715, -0.0012100, 0.0011978
9: -0.0036877, -0.0032079, -0.0036877, -0.0032079, -0.0004797, 0.0004797

### Unstable ReLU Count (Linear/Conv2D Layers)
- layer_idx=0, type=LayerType.Linear, total=256, inp1_unstable=5, inp2_unstable=5, delta_unstable=251
- layer_idx=2, type=LayerType.Linear, total=256, inp1_unstable=4, inp2_unstable=4, delta_unstable=255
- layer_idx=4, type=LayerType.Linear, total=256, inp1_unstable=0, inp2_unstable=0, delta_unstable=256
- layer_idx=6, type=LayerType.Linear, total=256, inp1_unstable=1, inp2_unstable=1, delta_unstable=256
- layer_idx=8, type=LayerType.Linear, total=10, inp1_unstable=0, inp2_unstable=0, delta_unstable=10

Time for backsubstitution: 1.27 seconds

### RS candidates at layer 3
type: RSZ, layer: 3, pos: 73
type: RSZ, layer: 3, pos: 81
type: RSZ, layer: 3, pos: 136

Time for candidate selection: 0.13 seconds

### Candidate
type: RSZ, layer: 3, pos: 73

### Candidate
type: RSZ, layer: 3, pos: 81

### Relational analysis RSZ of RS_RSZ2_RSZ1_RSZ2_RSZ1_RSZ2_RSZ1_RSZ2_RSZ1

#### Relational analysis RSZ result of RS_RSZ2_RSZ1_RSZ2_RSZ1_RSZ2_RSZ1_RSZ2_RSZ1
Status: Status.VERIFIED
Output dim: 1, lower bound: -0.0002466, upper bound: 0.0002486
time: 0.58 seconds

### Relational analysis RSZ of RS_RSZ2_RSZ1_RSZ2_RSZ1_RSZ2_RSZ1_RSZ2_RSZ2

#### Relational analysis RSZ result of RS_RSZ2_RSZ1_RSZ2_RSZ1_RSZ2_RSZ1_RSZ2_RSZ2
Status: Status.VERIFIED
Output dim: 1, lower bound: -0.0002466, upper bound: 0.0002486
time: 0.59 seconds

## BFS RS instance: RS_RSZ2_RSZ1_RSZ2_RSZ1_RSZ2_RSZ2_RSZ1

### Backsubstitution after applying RS history:
0: 0.0009457, 0.0011003, 0.0009457, 0.0011003, -0.0001487, 0.0001489
1: 0.9936197, 0.9940771, 0.9936197, 0.9940771, -0.0003871, 0.0003840
2: -0.0065606, -0.0050208, -0.0065606, -0.0050208, -0.0011943, 0.0011632
3: 0.0038004, 0.0040140, 0.0038004, 0.0040140, -0.0001629, 0.0001650
4: 0.0023852, 0.0036022, 0.0023852, 0.0036022, -0.0010241, 0.0010487
5: 0.0060086, 0.0065111, 0.0060086, 0.0065111, -0.0005024, 0.0005024
6: -0.0013983, -0.0008639, -0.0013983, -0.0008639, -0.0004080, 0.0003972
7: -0.0083221, -0.0079387, -0.0083221, -0.0079387, -0.0003835, 0.0003835
8: 0.0048484, 0.0068715, 0.0048484, 0.0068715, -0.0011857, 0.0012279
9: -0.0036877, -0.0032079, -0.0036877, -0.0032079, -0.0004797, 0.0004797

### Unstable ReLU Count (Linear/Conv2D Layers)
- layer_idx=0, type=LayerType.Linear, total=256, inp1_unstable=5, inp2_unstable=5, delta_unstable=251
- layer_idx=2, type=LayerType.Linear, total=256, inp1_unstable=4, inp2_unstable=4, delta_unstable=255
- layer_idx=4, type=LayerType.Linear, total=256, inp1_unstable=0, inp2_unstable=0, delta_unstable=256
- layer_idx=6, type=LayerType.Linear, total=256, inp1_unstable=1, inp2_unstable=1, delta_unstable=256
- layer_idx=8, type=LayerType.Linear, total=10, inp1_unstable=0, inp2_unstable=0, delta_unstable=10

Time for backsubstitution: 1.30 seconds

### RS candidates at layer 3
type: RSZ, layer: 3, pos: 73
type: RSZ, layer: 3, pos: 81
type: RSZ, layer: 3, pos: 136

Time for candidate selection: 0.14 seconds

### Candidate
type: RSZ, layer: 3, pos: 73

### Candidate
type: RSZ, layer: 3, pos: 81

### Relational analysis RSZ of RS_RSZ2_RSZ1_RSZ2_RSZ1_RSZ2_RSZ2_RSZ1_RSZ1

#### Relational analysis RSZ result of RS_RSZ2_RSZ1_RSZ2_RSZ1_RSZ2_RSZ2_RSZ1_RSZ1
Status: Status.VERIFIED
Output dim: 1, lower bound: -0.0002495, upper bound: 0.0002463
time: 0.61 seconds

### Relational analysis RSZ of RS_RSZ2_RSZ1_RSZ2_RSZ1_RSZ2_RSZ2_RSZ1_RSZ2

#### Relational analysis RSZ result of RS_RSZ2_RSZ1_RSZ2_RSZ1_RSZ2_RSZ2_RSZ1_RSZ2
Status: Status.VERIFIED
Output dim: 1, lower bound: -0.0002495, upper bound: 0.0002463
time: 0.61 seconds

## BFS RS instance: RS_RSZ2_RSZ1_RSZ2_RSZ1_RSZ2_RSZ2_RSZ2

### Backsubstitution after applying RS history:
0: 0.0009457, 0.0011003, 0.0009457, 0.0011003, -0.0001488, 0.0001487
1: 0.9936197, 0.9940771, 0.9936197, 0.9940771, -0.0003849, 0.0003858
2: -0.0065606, -0.0050208, -0.0065606, -0.0050208, -0.0011703, 0.0011817
3: 0.0038004, 0.0040140, 0.0038004, 0.0040140, -0.0001641, 0.0001635
4: 0.0023852, 0.0036022, 0.0023852, 0.0036022, -0.0010387, 0.0010298
5: 0.0060086, 0.0065111, 0.0060086, 0.0065111, -0.0005024, 0.0005024
6: -0.0013983, -0.0008639, -0.0013983, -0.0008639, -0.0003997, 0.0004037
7: -0.0083221, -0.0079387, -0.0083221, -0.0079387, -0.0003835, 0.0003835
8: 0.0048484, 0.0068715, 0.0048484, 0.0068715, -0.0012100, 0.0011978
9: -0.0036877, -0.0032079, -0.0036877, -0.0032079, -0.0004797, 0.0004797

### Unstable ReLU Count (Linear/Conv2D Layers)
- layer_idx=0, type=LayerType.Linear, total=256, inp1_unstable=5, inp2_unstable=5, delta_unstable=251
- layer_idx=2, type=LayerType.Linear, total=256, inp1_unstable=4, inp2_unstable=4, delta_unstable=255
- layer_idx=4, type=LayerType.Linear, total=256, inp1_unstable=0, inp2_unstable=0, delta_unstable=256
- layer_idx=6, type=LayerType.Linear, total=256, inp1_unstable=1, inp2_unstable=1, delta_unstable=256
- layer_idx=8, type=LayerType.Linear, total=10, inp1_unstable=0, inp2_unstable=0, delta_unstable=10

Time for backsubstitution: 1.29 seconds

### RS candidates at layer 3
type: RSZ, layer: 3, pos: 73
type: RSZ, layer: 3, pos: 81
type: RSZ, layer: 3, pos: 136

Time for candidate selection: 0.13 seconds

### Candidate
type: RSZ, layer: 3, pos: 73

### Candidate
type: RSZ, layer: 3, pos: 81

### Relational analysis RSZ of RS_RSZ2_RSZ1_RSZ2_RSZ1_RSZ2_RSZ2_RSZ2_RSZ1

#### Relational analysis RSZ result of RS_RSZ2_RSZ1_RSZ2_RSZ1_RSZ2_RSZ2_RSZ2_RSZ1
Status: Status.VERIFIED
Output dim: 1, lower bound: -0.0002466, upper bound: 0.0002486
time: 0.62 seconds

### Relational analysis RSZ of RS_RSZ2_RSZ1_RSZ2_RSZ1_RSZ2_RSZ2_RSZ2_RSZ2

#### Relational analysis RSZ result of RS_RSZ2_RSZ1_RSZ2_RSZ1_RSZ2_RSZ2_RSZ2_RSZ2
Status: Status.VERIFIED
Output dim: 1, lower bound: -0.0002466, upper bound: 0.0002486
time: 0.63 seconds

## BFS RS instance: RS_RSZ2_RSZ1_RSZ2_RSZ2_RSZ1_RSZ1_RSZ1

### Backsubstitution after applying RS history:
0: 0.0009457, 0.0011003, 0.0009457, 0.0011003, -0.0001486, 0.0001489
1: 0.9936197, 0.9940771, 0.9936197, 0.9940771, -0.0003869, 0.0003834
2: -0.0065606, -0.0050208, -0.0065606, -0.0050208, -0.0011919, 0.0011573
3: 0.0038004, 0.0040140, 0.0038004, 0.0040140, -0.0001625, 0.0001648
4: 0.0023852, 0.0036022, 0.0023852, 0.0036022, -0.0010195, 0.0010468
5: 0.0060086, 0.0065111, 0.0060086, 0.0065111, -0.0005024, 0.0005024
6: -0.0013983, -0.0008639, -0.0013983, -0.0008639, -0.0004072, 0.0003952
7: -0.0083221, -0.0079387, -0.0083221, -0.0079387, -0.0003835, 0.0003835
8: 0.0048484, 0.0068715, 0.0048484, 0.0068715, -0.0011780, 0.0012248
9: -0.0036877, -0.0032079, -0.0036877, -0.0032079, -0.0004797, 0.0004797

### Unstable ReLU Count (Linear/Conv2D Layers)
- layer_idx=0, type=LayerType.Linear, total=256, inp1_unstable=5, inp2_unstable=5, delta_unstable=251
- layer_idx=2, type=LayerType.Linear, total=256, inp1_unstable=4, inp2_unstable=4, delta_unstable=255
- layer_idx=4, type=LayerType.Linear, total=256, inp1_unstable=0, inp2_unstable=0, delta_unstable=256
- layer_idx=6, type=LayerType.Linear, total=256, inp1_unstable=1, inp2_unstable=1, delta_unstable=256
- layer_idx=8, type=LayerType.Linear, total=10, inp1_unstable=0, inp2_unstable=0, delta_unstable=10

Time for backsubstitution: 1.29 seconds

### RS candidates at layer 3
type: RSZ, layer: 3, pos: 73
type: RSZ, layer: 3, pos: 81
type: RSZ, layer: 3, pos: 136

Time for candidate selection: 0.13 seconds

### Candidate
type: RSZ, layer: 3, pos: 73

### Candidate
type: RSZ, layer: 3, pos: 81

### Relational analysis RSZ of RS_RSZ2_RSZ1_RSZ2_RSZ2_RSZ1_RSZ1_RSZ1_RSZ1

#### Relational analysis RSZ result of RS_RSZ2_RSZ1_RSZ2_RSZ2_RSZ1_RSZ1_RSZ1_RSZ1
Status: Status.VERIFIED
Output dim: 1, lower bound: -0.0002486, upper bound: 0.0002468
time: 0.59 seconds

### Relational analysis RSZ of RS_RSZ2_RSZ1_RSZ2_RSZ2_RSZ1_RSZ1_RSZ1_RSZ2

#### Relational analysis RSZ result of RS_RSZ2_RSZ1_RSZ2_RSZ2_RSZ1_RSZ1_RSZ1_RSZ2
Status: Status.VERIFIED
Output dim: 1, lower bound: -0.0002486, upper bound: 0.0002468
time: 0.60 seconds

## BFS RS instance: RS_RSZ2_RSZ1_RSZ2_RSZ2_RSZ1_RSZ1_RSZ2

### Backsubstitution after applying RS history:
0: 0.0009457, 0.0011003, 0.0009457, 0.0011003, -0.0001488, 0.0001488
1: 0.9936197, 0.9940771, 0.9936197, 0.9940771, -0.0003850, 0.0003856
2: -0.0065606, -0.0050208, -0.0065606, -0.0050208, -0.0011713, 0.0011797
3: 0.0038004, 0.0040140, 0.0038004, 0.0040140, -0.0001640, 0.0001636
4: 0.0023852, 0.0036022, 0.0023852, 0.0036022, -0.0010371, 0.0010306
5: 0.0060086, 0.0065111, 0.0060086, 0.0065111, -0.0005024, 0.0005024
6: -0.0013983, -0.0008639, -0.0013983, -0.0008639, -0.0004001, 0.0004030
7: -0.0083221, -0.0079387, -0.0083221, -0.0079387, -0.0003835, 0.0003835
8: 0.0048484, 0.0068715, 0.0048484, 0.0068715, -0.0012073, 0.0011991
9: -0.0036877, -0.0032079, -0.0036877, -0.0032079, -0.0004797, 0.0004797

### Unstable ReLU Count (Linear/Conv2D Layers)
- layer_idx=0, type=LayerType.Linear, total=256, inp1_unstable=5, inp2_unstable=5, delta_unstable=251
- layer_idx=2, type=LayerType.Linear, total=256, inp1_unstable=4, inp2_unstable=4, delta_unstable=255
- layer_idx=4, type=LayerType.Linear, total=256, inp1_unstable=0, inp2_unstable=0, delta_unstable=256
- layer_idx=6, type=LayerType.Linear, total=256, inp1_unstable=1, inp2_unstable=1, delta_unstable=256
- layer_idx=8, type=LayerType.Linear, total=10, inp1_unstable=0, inp2_unstable=0, delta_unstable=10

Time for backsubstitution: 1.30 seconds

### RS candidates at layer 3
type: RSZ, layer: 3, pos: 73
type: RSZ, layer: 3, pos: 81
type: RSZ, layer: 3, pos: 136

Time for candidate selection: 0.13 seconds

### Candidate
type: RSZ, layer: 3, pos: 73

### Candidate
type: RSZ, layer: 3, pos: 81

### Relational analysis RSZ of RS_RSZ2_RSZ1_RSZ2_RSZ2_RSZ1_RSZ1_RSZ2_RSZ1

#### Relational analysis RSZ result of RS_RSZ2_RSZ1_RSZ2_RSZ2_RSZ1_RSZ1_RSZ2_RSZ1
Status: Status.VERIFIED
Output dim: 1, lower bound: -0.0002462, upper bound: 0.0002493
time: 0.62 seconds

### Relational analysis RSZ of RS_RSZ2_RSZ1_RSZ2_RSZ2_RSZ1_RSZ1_RSZ2_RSZ2

#### Relational analysis RSZ result of RS_RSZ2_RSZ1_RSZ2_RSZ2_RSZ1_RSZ1_RSZ2_RSZ2
Status: Status.VERIFIED
Output dim: 1, lower bound: -0.0002462, upper bound: 0.0002493
time: 0.62 seconds

## BFS RS instance: RS_RSZ2_RSZ1_RSZ2_RSZ2_RSZ1_RSZ2_RSZ1

### Backsubstitution after applying RS history:
0: 0.0009457, 0.0011003, 0.0009457, 0.0011003, -0.0001486, 0.0001489
1: 0.9936197, 0.9940771, 0.9936197, 0.9940771, -0.0003869, 0.0003834
2: -0.0065606, -0.0050208, -0.0065606, -0.0050208, -0.0011919, 0.0011573
3: 0.0038004, 0.0040140, 0.0038004, 0.0040140, -0.0001625, 0.0001648
4: 0.0023852, 0.0036022, 0.0023852, 0.0036022, -0.0010195, 0.0010468
5: 0.0060086, 0.0065111, 0.0060086, 0.0065111, -0.0005024, 0.0005024
6: -0.0013983, -0.0008639, -0.0013983, -0.0008639, -0.0004072, 0.0003952
7: -0.0083221, -0.0079387, -0.0083221, -0.0079387, -0.0003835, 0.0003835
8: 0.0048484, 0.0068715, 0.0048484, 0.0068715, -0.0011780, 0.0012248
9: -0.0036877, -0.0032079, -0.0036877, -0.0032079, -0.0004797, 0.0004797

### Unstable ReLU Count (Linear/Conv2D Layers)
- layer_idx=0, type=LayerType.Linear, total=256, inp1_unstable=5, inp2_unstable=5, delta_unstable=251
- layer_idx=2, type=LayerType.Linear, total=256, inp1_unstable=4, inp2_unstable=4, delta_unstable=255
- layer_idx=4, type=LayerType.Linear, total=256, inp1_unstable=0, inp2_unstable=0, delta_unstable=256
- layer_idx=6, type=LayerType.Linear, total=256, inp1_unstable=1, inp2_unstable=1, delta_unstable=256
- layer_idx=8, type=LayerType.Linear, total=10, inp1_unstable=0, inp2_unstable=0, delta_unstable=10

Time for backsubstitution: 1.32 seconds

### RS candidates at layer 3
type: RSZ, layer: 3, pos: 73
type: RSZ, layer: 3, pos: 81
type: RSZ, layer: 3, pos: 136

Time for candidate selection: 0.13 seconds

### Candidate
type: RSZ, layer: 3, pos: 73

### Candidate
type: RSZ, layer: 3, pos: 81

### Relational analysis RSZ of RS_RSZ2_RSZ1_RSZ2_RSZ2_RSZ1_RSZ2_RSZ1_RSZ1

#### Relational analysis RSZ result of RS_RSZ2_RSZ1_RSZ2_RSZ2_RSZ1_RSZ2_RSZ1_RSZ1
Status: Status.VERIFIED
Output dim: 1, lower bound: -0.0002486, upper bound: 0.0002468
time: 0.58 seconds

### Relational analysis RSZ of RS_RSZ2_RSZ1_RSZ2_RSZ2_RSZ1_RSZ2_RSZ1_RSZ2

#### Relational analysis RSZ result of RS_RSZ2_RSZ1_RSZ2_RSZ2_RSZ1_RSZ2_RSZ1_RSZ2
Status: Status.VERIFIED
Output dim: 1, lower bound: -0.0002486, upper bound: 0.0002468
time: 0.58 seconds

## BFS RS instance: RS_RSZ2_RSZ1_RSZ2_RSZ2_RSZ1_RSZ2_RSZ2

### Backsubstitution after applying RS history:
0: 0.0009457, 0.0011003, 0.0009457, 0.0011003, -0.0001488, 0.0001488
1: 0.9936197, 0.9940771, 0.9936197, 0.9940771, -0.0003850, 0.0003856
2: -0.0065606, -0.0050208, -0.0065606, -0.0050208, -0.0011713, 0.0011797
3: 0.0038004, 0.0040140, 0.0038004, 0.0040140, -0.0001640, 0.0001636
4: 0.0023852, 0.0036022, 0.0023852, 0.0036022, -0.0010371, 0.0010306
5: 0.0060086, 0.0065111, 0.0060086, 0.0065111, -0.0005024, 0.0005024
6: -0.0013983, -0.0008639, -0.0013983, -0.0008639, -0.0004001, 0.0004030
7: -0.0083221, -0.0079387, -0.0083221, -0.0079387, -0.0003835, 0.0003835
8: 0.0048484, 0.0068715, 0.0048484, 0.0068715, -0.0012073, 0.0011991
9: -0.0036877, -0.0032079, -0.0036877, -0.0032079, -0.0004797, 0.0004797

### Unstable ReLU Count (Linear/Conv2D Layers)
- layer_idx=0, type=LayerType.Linear, total=256, inp1_unstable=5, inp2_unstable=5, delta_unstable=251
- layer_idx=2, type=LayerType.Linear, total=256, inp1_unstable=4, inp2_unstable=4, delta_unstable=255
- layer_idx=4, type=LayerType.Linear, total=256, inp1_unstable=0, inp2_unstable=0, delta_unstable=256
- layer_idx=6, type=LayerType.Linear, total=256, inp1_unstable=1, inp2_unstable=1, delta_unstable=256
- layer_idx=8, type=LayerType.Linear, total=10, inp1_unstable=0, inp2_unstable=0, delta_unstable=10

Time for backsubstitution: 1.35 seconds

### RS candidates at layer 3
type: RSZ, layer: 3, pos: 73
type: RSZ, layer: 3, pos: 81
type: RSZ, layer: 3, pos: 136

Time for candidate selection: 0.14 seconds

### Candidate
type: RSZ, layer: 3, pos: 73

### Candidate
type: RSZ, layer: 3, pos: 81

### Relational analysis RSZ of RS_RSZ2_RSZ1_RSZ2_RSZ2_RSZ1_RSZ2_RSZ2_RSZ1

#### Relational analysis RSZ result of RS_RSZ2_RSZ1_RSZ2_RSZ2_RSZ1_RSZ2_RSZ2_RSZ1
Status: Status.VERIFIED
Output dim: 1, lower bound: -0.0002462, upper bound: 0.0002493
time: 0.61 seconds

### Relational analysis RSZ of RS_RSZ2_RSZ1_RSZ2_RSZ2_RSZ1_RSZ2_RSZ2_RSZ2

#### Relational analysis RSZ result of RS_RSZ2_RSZ1_RSZ2_RSZ2_RSZ1_RSZ2_RSZ2_RSZ2
Status: Status.VERIFIED
Output dim: 1, lower bound: -0.0002462, upper bound: 0.0002493
time: 0.62 seconds

## BFS RS instance: RS_RSZ2_RSZ1_RSZ2_RSZ2_RSZ2_RSZ1_RSZ1

### Backsubstitution after applying RS history:
0: 0.0009457, 0.0011003, 0.0009457, 0.0011003, -0.0001486, 0.0001489
1: 0.9936197, 0.9940771, 0.9936197, 0.9940771, -0.0003864, 0.0003837
2: -0.0065606, -0.0050208, -0.0065606, -0.0050208, -0.0011876, 0.0011595
3: 0.0038004, 0.0040140, 0.0038004, 0.0040140, -0.0001626, 0.0001646
4: 0.0023852, 0.0036022, 0.0023852, 0.0036022, -0.0010212, 0.0010434
5: 0.0060086, 0.0065111, 0.0060086, 0.0065111, -0.0005024, 0.0005024
6: -0.0013983, -0.0008639, -0.0013983, -0.0008639, -0.0004057, 0.0003960
7: -0.0083221, -0.0079387, -0.0083221, -0.0079387, -0.0003835, 0.0003835
8: 0.0048484, 0.0068715, 0.0048484, 0.0068715, -0.0011808, 0.0012192
9: -0.0036877, -0.0032079, -0.0036877, -0.0032079, -0.0004797, 0.0004797

### Unstable ReLU Count (Linear/Conv2D Layers)
- layer_idx=0, type=LayerType.Linear, total=256, inp1_unstable=5, inp2_unstable=5, delta_unstable=251
- layer_idx=2, type=LayerType.Linear, total=256, inp1_unstable=4, inp2_unstable=4, delta_unstable=255
- layer_idx=4, type=LayerType.Linear, total=256, inp1_unstable=0, inp2_unstable=0, delta_unstable=256
- layer_idx=6, type=LayerType.Linear, total=256, inp1_unstable=1, inp2_unstable=1, delta_unstable=256
- layer_idx=8, type=LayerType.Linear, total=10, inp1_unstable=0, inp2_unstable=0, delta_unstable=10

Time for backsubstitution: 1.40 seconds

### RS candidates at layer 3
type: RSZ, layer: 3, pos: 73
type: RSZ, layer: 3, pos: 81
type: RSZ, layer: 3, pos: 136

Time for candidate selection: 0.15 seconds

### Candidate
type: RSZ, layer: 3, pos: 73

### Candidate
type: RSZ, layer: 3, pos: 81

### Relational analysis RSZ of RS_RSZ2_RSZ1_RSZ2_RSZ2_RSZ2_RSZ1_RSZ1_RSZ1

#### Relational analysis RSZ result of RS_RSZ2_RSZ1_RSZ2_RSZ2_RSZ2_RSZ1_RSZ1_RSZ1
Status: Status.VERIFIED
Output dim: 1, lower bound: -0.0002485, upper bound: 0.0002468
time: 0.62 seconds

### Relational analysis RSZ of RS_RSZ2_RSZ1_RSZ2_RSZ2_RSZ2_RSZ1_RSZ1_RSZ2

#### Relational analysis RSZ result of RS_RSZ2_RSZ1_RSZ2_RSZ2_RSZ2_RSZ1_RSZ1_RSZ2
Status: Status.VERIFIED
Output dim: 1, lower bound: -0.0002485, upper bound: 0.0002468
time: 0.64 seconds

## BFS RS instance: RS_RSZ2_RSZ1_RSZ2_RSZ2_RSZ2_RSZ1_RSZ2

### Backsubstitution after applying RS history:
0: 0.0009457, 0.0011003, 0.0009457, 0.0011003, -0.0001488, 0.0001487
1: 0.9936197, 0.9940771, 0.9936197, 0.9940771, -0.0003849, 0.0003860
2: -0.0065606, -0.0050208, -0.0065606, -0.0050208, -0.0011700, 0.0011839
3: 0.0038004, 0.0040140, 0.0038004, 0.0040140, -0.0001642, 0.0001635
4: 0.0023852, 0.0036022, 0.0023852, 0.0036022, -0.0010405, 0.0010295
5: 0.0060086, 0.0065111, 0.0060086, 0.0065111, -0.0005024, 0.0005024
6: -0.0013983, -0.0008639, -0.0013983, -0.0008639, -0.0003996, 0.0004044
7: -0.0083221, -0.0079387, -0.0083221, -0.0079387, -0.0003835, 0.0003835
8: 0.0048484, 0.0068715, 0.0048484, 0.0068715, -0.0012129, 0.0011974
9: -0.0036877, -0.0032079, -0.0036877, -0.0032079, -0.0004797, 0.0004797

### Unstable ReLU Count (Linear/Conv2D Layers)
- layer_idx=0, type=LayerType.Linear, total=256, inp1_unstable=5, inp2_unstable=5, delta_unstable=251
- layer_idx=2, type=LayerType.Linear, total=256, inp1_unstable=4, inp2_unstable=4, delta_unstable=255
- layer_idx=4, type=LayerType.Linear, total=256, inp1_unstable=0, inp2_unstable=0, delta_unstable=256
- layer_idx=6, type=LayerType.Linear, total=256, inp1_unstable=1, inp2_unstable=1, delta_unstable=256
- layer_idx=8, type=LayerType.Linear, total=10, inp1_unstable=0, inp2_unstable=0, delta_unstable=10

Time for backsubstitution: 1.48 seconds

### RS candidates at layer 3
type: RSZ, layer: 3, pos: 73
type: RSZ, layer: 3, pos: 81
type: RSZ, layer: 3, pos: 136

Time for candidate selection: 0.17 seconds

### Candidate
type: RSZ, layer: 3, pos: 73

### Candidate
type: RSZ, layer: 3, pos: 81

### Relational analysis RSZ of RS_RSZ2_RSZ1_RSZ2_RSZ2_RSZ2_RSZ1_RSZ2_RSZ1

#### Relational analysis RSZ result of RS_RSZ2_RSZ1_RSZ2_RSZ2_RSZ2_RSZ1_RSZ2_RSZ1
Status: Status.VERIFIED
Output dim: 1, lower bound: -0.0002462, upper bound: 0.0002493
time: 0.62 seconds

### Relational analysis RSZ of RS_RSZ2_RSZ1_RSZ2_RSZ2_RSZ2_RSZ1_RSZ2_RSZ2

#### Relational analysis RSZ result of RS_RSZ2_RSZ1_RSZ2_RSZ2_RSZ2_RSZ1_RSZ2_RSZ2
Status: Status.VERIFIED
Output dim: 1, lower bound: -0.0002462, upper bound: 0.0002493
time: 0.61 seconds

## BFS RS instance: RS_RSZ2_RSZ1_RSZ2_RSZ2_RSZ2_RSZ2_RSZ1

### Backsubstitution after applying RS history:
0: 0.0009457, 0.0011003, 0.0009457, 0.0011003, -0.0001486, 0.0001489
1: 0.9936197, 0.9940771, 0.9936197, 0.9940771, -0.0003864, 0.0003837
2: -0.0065606, -0.0050208, -0.0065606, -0.0050208, -0.0011876, 0.0011595
3: 0.0038004, 0.0040140, 0.0038004, 0.0040140, -0.0001626, 0.0001646
4: 0.0023852, 0.0036022, 0.0023852, 0.0036022, -0.0010212, 0.0010434
5: 0.0060086, 0.0065111, 0.0060086, 0.0065111, -0.0005024, 0.0005024
6: -0.0013983, -0.0008639, -0.0013983, -0.0008639, -0.0004057, 0.0003960
7: -0.0083221, -0.0079387, -0.0083221, -0.0079387, -0.0003835, 0.0003835
8: 0.0048484, 0.0068715, 0.0048484, 0.0068715, -0.0011808, 0.0012192
9: -0.0036877, -0.0032079, -0.0036877, -0.0032079, -0.0004797, 0.0004797

### Unstable ReLU Count (Linear/Conv2D Layers)
- layer_idx=0, type=LayerType.Linear, total=256, inp1_unstable=5, inp2_unstable=5, delta_unstable=251
- layer_idx=2, type=LayerType.Linear, total=256, inp1_unstable=4, inp2_unstable=4, delta_unstable=255
- layer_idx=4, type=LayerType.Linear, total=256, inp1_unstable=0, inp2_unstable=0, delta_unstable=256
- layer_idx=6, type=LayerType.Linear, total=256, inp1_unstable=1, inp2_unstable=1, delta_unstable=256
- layer_idx=8, type=LayerType.Linear, total=10, inp1_unstable=0, inp2_unstable=0, delta_unstable=10

Time for backsubstitution: 1.30 seconds

### RS candidates at layer 3
type: RSZ, layer: 3, pos: 73
type: RSZ, layer: 3, pos: 81
type: RSZ, layer: 3, pos: 136

Time for candidate selection: 0.14 seconds

### Candidate
type: RSZ, layer: 3, pos: 73

### Candidate
type: RSZ, layer: 3, pos: 81

### Relational analysis RSZ of RS_RSZ2_RSZ1_RSZ2_RSZ2_RSZ2_RSZ2_RSZ1_RSZ1

#### Relational analysis RSZ result of RS_RSZ2_RSZ1_RSZ2_RSZ2_RSZ2_RSZ2_RSZ1_RSZ1
Status: Status.VERIFIED
Output dim: 1, lower bound: -0.0002485, upper bound: 0.0002468
time: 0.61 seconds

### Relational analysis RSZ of RS_RSZ2_RSZ1_RSZ2_RSZ2_RSZ2_RSZ2_RSZ1_RSZ2

#### Relational analysis RSZ result of RS_RSZ2_RSZ1_RSZ2_RSZ2_RSZ2_RSZ2_RSZ1_RSZ2
Status: Status.VERIFIED
Output dim: 1, lower bound: -0.0002485, upper bound: 0.0002468
time: 0.61 seconds

## BFS RS instance: RS_RSZ2_RSZ1_RSZ2_RSZ2_RSZ2_RSZ2_RSZ2

### Backsubstitution after applying RS history:
0: 0.0009457, 0.0011003, 0.0009457, 0.0011003, -0.0001488, 0.0001487
1: 0.9936197, 0.9940771, 0.9936197, 0.9940771, -0.0003849, 0.0003860
2: -0.0065606, -0.0050208, -0.0065606, -0.0050208, -0.0011700, 0.0011839
3: 0.0038004, 0.0040140, 0.0038004, 0.0040140, -0.0001642, 0.0001635
4: 0.0023852, 0.0036022, 0.0023852, 0.0036022, -0.0010405, 0.0010295
5: 0.0060086, 0.0065111, 0.0060086, 0.0065111, -0.0005024, 0.0005024
6: -0.0013983, -0.0008639, -0.0013983, -0.0008639, -0.0003996, 0.0004044
7: -0.0083221, -0.0079387, -0.0083221, -0.0079387, -0.0003835, 0.0003835
8: 0.0048484, 0.0068715, 0.0048484, 0.0068715, -0.0012129, 0.0011974
9: -0.0036877, -0.0032079, -0.0036877, -0.0032079, -0.0004797, 0.0004797

### Unstable ReLU Count (Linear/Conv2D Layers)
- layer_idx=0, type=LayerType.Linear, total=256, inp1_unstable=5, inp2_unstable=5, delta_unstable=251
- layer_idx=2, type=LayerType.Linear, total=256, inp1_unstable=4, inp2_unstable=4, delta_unstable=255
- layer_idx=4, type=LayerType.Linear, total=256, inp1_unstable=0, inp2_unstable=0, delta_unstable=256
- layer_idx=6, type=LayerType.Linear, total=256, inp1_unstable=1, inp2_unstable=1, delta_unstable=256
- layer_idx=8, type=LayerType.Linear, total=10, inp1_unstable=0, inp2_unstable=0, delta_unstable=10

Time for backsubstitution: 1.37 seconds

### RS candidates at layer 3
type: RSZ, layer: 3, pos: 73
type: RSZ, layer: 3, pos: 81
type: RSZ, layer: 3, pos: 136

Time for candidate selection: 0.15 seconds

### Candidate
type: RSZ, layer: 3, pos: 73

### Candidate
type: RSZ, layer: 3, pos: 81

### Relational analysis RSZ of RS_RSZ2_RSZ1_RSZ2_RSZ2_RSZ2_RSZ2_RSZ2_RSZ1

#### Relational analysis RSZ result of RS_RSZ2_RSZ1_RSZ2_RSZ2_RSZ2_RSZ2_RSZ2_RSZ1
Status: Status.VERIFIED
Output dim: 1, lower bound: -0.0002462, upper bound: 0.0002493
time: 0.63 seconds

### Relational analysis RSZ of RS_RSZ2_RSZ1_RSZ2_RSZ2_RSZ2_RSZ2_RSZ2_RSZ2

#### Relational analysis RSZ result of RS_RSZ2_RSZ1_RSZ2_RSZ2_RSZ2_RSZ2_RSZ2_RSZ2
Status: Status.VERIFIED
Output dim: 1, lower bound: -0.0002462, upper bound: 0.0002493
time: 0.64 seconds

## BFS RS instance: RS_RSZ2_RSZ2_RSZ1_RSZ1_RSZ1_RSZ1_RSZ1

### Backsubstitution after applying RS history:
0: 0.0009457, 0.0011003, 0.0009457, 0.0011003, -0.0001486, 0.0001491
1: 0.9936197, 0.9940771, 0.9936197, 0.9940771, -0.0003890, 0.0003836
2: -0.0065606, -0.0050208, -0.0065606, -0.0050208, -0.0012059, 0.0011594
3: 0.0038004, 0.0040140, 0.0038004, 0.0040140, -0.0001626, 0.0001663
4: 0.0023852, 0.0036022, 0.0023852, 0.0036022, -0.0010211, 0.0010577
5: 0.0060086, 0.0065111, 0.0060086, 0.0065111, -0.0005024, 0.0005024
6: -0.0013983, -0.0008639, -0.0013983, -0.0008639, -0.0004121, 0.0003959
7: -0.0083221, -0.0079387, -0.0083221, -0.0079387, -0.0003835, 0.0003835
8: 0.0048484, 0.0068715, 0.0048484, 0.0068715, -0.0011807, 0.0012513
9: -0.0036877, -0.0032079, -0.0036877, -0.0032079, -0.0004797, 0.0004797

### Unstable ReLU Count (Linear/Conv2D Layers)
- layer_idx=0, type=LayerType.Linear, total=256, inp1_unstable=5, inp2_unstable=5, delta_unstable=251
- layer_idx=2, type=LayerType.Linear, total=256, inp1_unstable=4, inp2_unstable=4, delta_unstable=255
- layer_idx=4, type=LayerType.Linear, total=256, inp1_unstable=0, inp2_unstable=0, delta_unstable=256
- layer_idx=6, type=LayerType.Linear, total=256, inp1_unstable=1, inp2_unstable=1, delta_unstable=256
- layer_idx=8, type=LayerType.Linear, total=10, inp1_unstable=0, inp2_unstable=0, delta_unstable=10

Time for backsubstitution: 1.34 seconds

### RS candidates at layer 3
type: RSZ, layer: 3, pos: 73
type: RSZ, layer: 3, pos: 81
type: RSZ, layer: 3, pos: 136

Time for candidate selection: 0.13 seconds

### Candidate
type: RSZ, layer: 3, pos: 73

### Candidate
type: RSZ, layer: 3, pos: 81

### Relational analysis RSZ of RS_RSZ2_RSZ2_RSZ1_RSZ1_RSZ1_RSZ1_RSZ1_RSZ1

#### Relational analysis RSZ result of RS_RSZ2_RSZ2_RSZ1_RSZ1_RSZ1_RSZ1_RSZ1_RSZ1
Status: Status.VERIFIED
Output dim: 1, lower bound: -0.0002493, upper bound: 0.0002462
time: 0.63 seconds

### Relational analysis RSZ of RS_RSZ2_RSZ2_RSZ1_RSZ1_RSZ1_RSZ1_RSZ1_RSZ2

#### Relational analysis RSZ result of RS_RSZ2_RSZ2_RSZ1_RSZ1_RSZ1_RSZ1_RSZ1_RSZ2
Status: Status.VERIFIED
Output dim: 1, lower bound: -0.0002493, upper bound: 0.0002462
time: 0.64 seconds

## BFS RS instance: RS_RSZ2_RSZ2_RSZ1_RSZ1_RSZ1_RSZ1_RSZ2

### Backsubstitution after applying RS history:
0: 0.0009457, 0.0011003, 0.0009457, 0.0011003, -0.0001488, 0.0001489
1: 0.9936197, 0.9940771, 0.9936197, 0.9940771, -0.0003863, 0.0003856
2: -0.0065606, -0.0050208, -0.0065606, -0.0050208, -0.0011803, 0.0011799
3: 0.0038004, 0.0040140, 0.0038004, 0.0040140, -0.0001640, 0.0001645
4: 0.0023852, 0.0036022, 0.0023852, 0.0036022, -0.0010373, 0.0010376
5: 0.0060086, 0.0065111, 0.0060086, 0.0065111, -0.0005024, 0.0005024
6: -0.0013983, -0.0008639, -0.0013983, -0.0008639, -0.0004032, 0.0004030
7: -0.0083221, -0.0079387, -0.0083221, -0.0079387, -0.0003835, 0.0003835
8: 0.0048484, 0.0068715, 0.0048484, 0.0068715, -0.0012076, 0.0012154
9: -0.0036877, -0.0032079, -0.0036877, -0.0032079, -0.0004797, 0.0004797

### Unstable ReLU Count (Linear/Conv2D Layers)
- layer_idx=0, type=LayerType.Linear, total=256, inp1_unstable=5, inp2_unstable=5, delta_unstable=251
- layer_idx=2, type=LayerType.Linear, total=256, inp1_unstable=4, inp2_unstable=4, delta_unstable=255
- layer_idx=4, type=LayerType.Linear, total=256, inp1_unstable=0, inp2_unstable=0, delta_unstable=256
- layer_idx=6, type=LayerType.Linear, total=256, inp1_unstable=1, inp2_unstable=1, delta_unstable=256
- layer_idx=8, type=LayerType.Linear, total=10, inp1_unstable=0, inp2_unstable=0, delta_unstable=10

Time for backsubstitution: 1.30 seconds

### RS candidates at layer 3
type: RSZ, layer: 3, pos: 73
type: RSZ, layer: 3, pos: 81
type: RSZ, layer: 3, pos: 136

Time for candidate selection: 0.14 seconds

### Candidate
type: RSZ, layer: 3, pos: 73

### Candidate
type: RSZ, layer: 3, pos: 81

### Relational analysis RSZ of RS_RSZ2_RSZ2_RSZ1_RSZ1_RSZ1_RSZ1_RSZ2_RSZ1

#### Relational analysis RSZ result of RS_RSZ2_RSZ2_RSZ1_RSZ1_RSZ1_RSZ1_RSZ2_RSZ1
Status: Status.VERIFIED
Output dim: 1, lower bound: -0.0002468, upper bound: 0.0002485
time: 0.58 seconds

### Relational analysis RSZ of RS_RSZ2_RSZ2_RSZ1_RSZ1_RSZ1_RSZ1_RSZ2_RSZ2

#### Relational analysis RSZ result of RS_RSZ2_RSZ2_RSZ1_RSZ1_RSZ1_RSZ1_RSZ2_RSZ2
Status: Status.VERIFIED
Output dim: 1, lower bound: -0.0002468, upper bound: 0.0002485
time: 0.59 seconds

## BFS RS instance: RS_RSZ2_RSZ2_RSZ1_RSZ1_RSZ1_RSZ2_RSZ1

### Backsubstitution after applying RS history:
0: 0.0009457, 0.0011003, 0.0009457, 0.0011003, -0.0001486, 0.0001491
1: 0.9936197, 0.9940771, 0.9936197, 0.9940771, -0.0003890, 0.0003836
2: -0.0065606, -0.0050208, -0.0065606, -0.0050208, -0.0012059, 0.0011594
3: 0.0038004, 0.0040140, 0.0038004, 0.0040140, -0.0001626, 0.0001663
4: 0.0023852, 0.0036022, 0.0023852, 0.0036022, -0.0010211, 0.0010577
5: 0.0060086, 0.0065111, 0.0060086, 0.0065111, -0.0005024, 0.0005024
6: -0.0013983, -0.0008639, -0.0013983, -0.0008639, -0.0004121, 0.0003959
7: -0.0083221, -0.0079387, -0.0083221, -0.0079387, -0.0003835, 0.0003835
8: 0.0048484, 0.0068715, 0.0048484, 0.0068715, -0.0011807, 0.0012513
9: -0.0036877, -0.0032079, -0.0036877, -0.0032079, -0.0004797, 0.0004797

### Unstable ReLU Count (Linear/Conv2D Layers)
- layer_idx=0, type=LayerType.Linear, total=256, inp1_unstable=5, inp2_unstable=5, delta_unstable=251
- layer_idx=2, type=LayerType.Linear, total=256, inp1_unstable=4, inp2_unstable=4, delta_unstable=255
- layer_idx=4, type=LayerType.Linear, total=256, inp1_unstable=0, inp2_unstable=0, delta_unstable=256
- layer_idx=6, type=LayerType.Linear, total=256, inp1_unstable=1, inp2_unstable=1, delta_unstable=256
- layer_idx=8, type=LayerType.Linear, total=10, inp1_unstable=0, inp2_unstable=0, delta_unstable=10

Time for backsubstitution: 1.30 seconds

### RS candidates at layer 3
type: RSZ, layer: 3, pos: 73
type: RSZ, layer: 3, pos: 81
type: RSZ, layer: 3, pos: 136

Time for candidate selection: 0.13 seconds

### Candidate
type: RSZ, layer: 3, pos: 73

### Candidate
type: RSZ, layer: 3, pos: 81

### Relational analysis RSZ of RS_RSZ2_RSZ2_RSZ1_RSZ1_RSZ1_RSZ2_RSZ1_RSZ1

#### Relational analysis RSZ result of RS_RSZ2_RSZ2_RSZ1_RSZ1_RSZ1_RSZ2_RSZ1_RSZ1
Status: Status.VERIFIED
Output dim: 1, lower bound: -0.0002493, upper bound: 0.0002462
time: 0.61 seconds

### Relational analysis RSZ of RS_RSZ2_RSZ2_RSZ1_RSZ1_RSZ1_RSZ2_RSZ1_RSZ2

#### Relational analysis RSZ result of RS_RSZ2_RSZ2_RSZ1_RSZ1_RSZ1_RSZ2_RSZ1_RSZ2
Status: Status.VERIFIED
Output dim: 1, lower bound: -0.0002493, upper bound: 0.0002462
time: 0.62 seconds

## BFS RS instance: RS_RSZ2_RSZ2_RSZ1_RSZ1_RSZ1_RSZ2_RSZ2

### Backsubstitution after applying RS history:
0: 0.0009457, 0.0011003, 0.0009457, 0.0011003, -0.0001488, 0.0001489
1: 0.9936197, 0.9940771, 0.9936197, 0.9940771, -0.0003863, 0.0003856
2: -0.0065606, -0.0050208, -0.0065606, -0.0050208, -0.0011803, 0.0011799
3: 0.0038004, 0.0040140, 0.0038004, 0.0040140, -0.0001640, 0.0001645
4: 0.0023852, 0.0036022, 0.0023852, 0.0036022, -0.0010373, 0.0010376
5: 0.0060086, 0.0065111, 0.0060086, 0.0065111, -0.0005024, 0.0005024
6: -0.0013983, -0.0008639, -0.0013983, -0.0008639, -0.0004032, 0.0004030
7: -0.0083221, -0.0079387, -0.0083221, -0.0079387, -0.0003835, 0.0003835
8: 0.0048484, 0.0068715, 0.0048484, 0.0068715, -0.0012076, 0.0012154
9: -0.0036877, -0.0032079, -0.0036877, -0.0032079, -0.0004797, 0.0004797

### Unstable ReLU Count (Linear/Conv2D Layers)
- layer_idx=0, type=LayerType.Linear, total=256, inp1_unstable=5, inp2_unstable=5, delta_unstable=251
- layer_idx=2, type=LayerType.Linear, total=256, inp1_unstable=4, inp2_unstable=4, delta_unstable=255
- layer_idx=4, type=LayerType.Linear, total=256, inp1_unstable=0, inp2_unstable=0, delta_unstable=256
- layer_idx=6, type=LayerType.Linear, total=256, inp1_unstable=1, inp2_unstable=1, delta_unstable=256
- layer_idx=8, type=LayerType.Linear, total=10, inp1_unstable=0, inp2_unstable=0, delta_unstable=10

Time for backsubstitution: 1.29 seconds

### RS candidates at layer 3
type: RSZ, layer: 3, pos: 73
type: RSZ, layer: 3, pos: 81
type: RSZ, layer: 3, pos: 136

Time for candidate selection: 0.14 seconds

### Candidate
type: RSZ, layer: 3, pos: 73

### Candidate
type: RSZ, layer: 3, pos: 81

### Relational analysis RSZ of RS_RSZ2_RSZ2_RSZ1_RSZ1_RSZ1_RSZ2_RSZ2_RSZ1

#### Relational analysis RSZ result of RS_RSZ2_RSZ2_RSZ1_RSZ1_RSZ1_RSZ2_RSZ2_RSZ1
Status: Status.VERIFIED
Output dim: 1, lower bound: -0.0002468, upper bound: 0.0002485
time: 0.62 seconds

### Relational analysis RSZ of RS_RSZ2_RSZ2_RSZ1_RSZ1_RSZ1_RSZ2_RSZ2_RSZ2

#### Relational analysis RSZ result of RS_RSZ2_RSZ2_RSZ1_RSZ1_RSZ1_RSZ2_RSZ2_RSZ2
Status: Status.VERIFIED
Output dim: 1, lower bound: -0.0002468, upper bound: 0.0002485
time: 0.63 seconds

## BFS RS instance: RS_RSZ2_RSZ2_RSZ1_RSZ1_RSZ2_RSZ1_RSZ1

### Backsubstitution after applying RS history:
0: 0.0009457, 0.0011003, 0.0009457, 0.0011003, -0.0001487, 0.0001490
1: 0.9936197, 0.9940771, 0.9936197, 0.9940771, -0.0003886, 0.0003838
2: -0.0065606, -0.0050208, -0.0065606, -0.0050208, -0.0012023, 0.0011606
3: 0.0038004, 0.0040140, 0.0038004, 0.0040140, -0.0001627, 0.0001661
4: 0.0023852, 0.0036022, 0.0023852, 0.0036022, -0.0010221, 0.0010548
5: 0.0060086, 0.0065111, 0.0060086, 0.0065111, -0.0005024, 0.0005024
6: -0.0013983, -0.0008639, -0.0013983, -0.0008639, -0.0004108, 0.0003964
7: -0.0083221, -0.0079387, -0.0083221, -0.0079387, -0.0003835, 0.0003835
8: 0.0048484, 0.0068715, 0.0048484, 0.0068715, -0.0011823, 0.0012465
9: -0.0036877, -0.0032079, -0.0036877, -0.0032079, -0.0004797, 0.0004797

### Unstable ReLU Count (Linear/Conv2D Layers)
- layer_idx=0, type=LayerType.Linear, total=256, inp1_unstable=5, inp2_unstable=5, delta_unstable=251
- layer_idx=2, type=LayerType.Linear, total=256, inp1_unstable=4, inp2_unstable=4, delta_unstable=255
- layer_idx=4, type=LayerType.Linear, total=256, inp1_unstable=0, inp2_unstable=0, delta_unstable=256
- layer_idx=6, type=LayerType.Linear, total=256, inp1_unstable=1, inp2_unstable=1, delta_unstable=256
- layer_idx=8, type=LayerType.Linear, total=10, inp1_unstable=0, inp2_unstable=0, delta_unstable=10

Time for backsubstitution: 1.31 seconds

### RS candidates at layer 3
type: RSZ, layer: 3, pos: 73
type: RSZ, layer: 3, pos: 81
type: RSZ, layer: 3, pos: 136

Time for candidate selection: 0.13 seconds

### Candidate
type: RSZ, layer: 3, pos: 73

### Candidate
type: RSZ, layer: 3, pos: 81

### Relational analysis RSZ of RS_RSZ2_RSZ2_RSZ1_RSZ1_RSZ2_RSZ1_RSZ1_RSZ1

#### Relational analysis RSZ result of RS_RSZ2_RSZ2_RSZ1_RSZ1_RSZ2_RSZ1_RSZ1_RSZ1
Status: Status.VERIFIED
Output dim: 1, lower bound: -0.0002493, upper bound: 0.0002462
time: 0.62 seconds

### Relational analysis RSZ of RS_RSZ2_RSZ2_RSZ1_RSZ1_RSZ2_RSZ1_RSZ1_RSZ2

#### Relational analysis RSZ result of RS_RSZ2_RSZ2_RSZ1_RSZ1_RSZ2_RSZ1_RSZ1_RSZ2
Status: Status.VERIFIED
Output dim: 1, lower bound: -0.0002493, upper bound: 0.0002462
time: 0.63 seconds

## BFS RS instance: RS_RSZ2_RSZ2_RSZ1_RSZ1_RSZ2_RSZ1_RSZ2

### Backsubstitution after applying RS history:
0: 0.0009457, 0.0011003, 0.0009457, 0.0011003, -0.0001488, 0.0001488
1: 0.9936197, 0.9940771, 0.9936197, 0.9940771, -0.0003860, 0.0003860
2: -0.0065606, -0.0050208, -0.0065606, -0.0050208, -0.0011773, 0.0011842
3: 0.0038004, 0.0040140, 0.0038004, 0.0040140, -0.0001643, 0.0001643
4: 0.0023852, 0.0036022, 0.0023852, 0.0036022, -0.0010407, 0.0010353
5: 0.0060086, 0.0065111, 0.0060086, 0.0065111, -0.0005024, 0.0005024
6: -0.0013983, -0.0008639, -0.0013983, -0.0008639, -0.0004022, 0.0004045
7: -0.0083221, -0.0079387, -0.0083221, -0.0079387, -0.0003835, 0.0003835
8: 0.0048484, 0.0068715, 0.0048484, 0.0068715, -0.0012132, 0.0012115
9: -0.0036877, -0.0032079, -0.0036877, -0.0032079, -0.0004797, 0.0004797

### Unstable ReLU Count (Linear/Conv2D Layers)
- layer_idx=0, type=LayerType.Linear, total=256, inp1_unstable=5, inp2_unstable=5, delta_unstable=251
- layer_idx=2, type=LayerType.Linear, total=256, inp1_unstable=4, inp2_unstable=4, delta_unstable=255
- layer_idx=4, type=LayerType.Linear, total=256, inp1_unstable=0, inp2_unstable=0, delta_unstable=256
- layer_idx=6, type=LayerType.Linear, total=256, inp1_unstable=1, inp2_unstable=1, delta_unstable=256
- layer_idx=8, type=LayerType.Linear, total=10, inp1_unstable=0, inp2_unstable=0, delta_unstable=10

Time for backsubstitution: 1.31 seconds

### RS candidates at layer 3
type: RSZ, layer: 3, pos: 73
type: RSZ, layer: 3, pos: 81
type: RSZ, layer: 3, pos: 136

Time for candidate selection: 0.13 seconds

### Candidate
type: RSZ, layer: 3, pos: 73

### Candidate
type: RSZ, layer: 3, pos: 81

### Relational analysis RSZ of RS_RSZ2_RSZ2_RSZ1_RSZ1_RSZ2_RSZ1_RSZ2_RSZ1

#### Relational analysis RSZ result of RS_RSZ2_RSZ2_RSZ1_RSZ1_RSZ2_RSZ1_RSZ2_RSZ1
Status: Status.VERIFIED
Output dim: 1, lower bound: -0.0002468, upper bound: 0.0002486
time: 0.59 seconds

### Relational analysis RSZ of RS_RSZ2_RSZ2_RSZ1_RSZ1_RSZ2_RSZ1_RSZ2_RSZ2

#### Relational analysis RSZ result of RS_RSZ2_RSZ2_RSZ1_RSZ1_RSZ2_RSZ1_RSZ2_RSZ2
Status: Status.VERIFIED
Output dim: 1, lower bound: -0.0002468, upper bound: 0.0002486
time: 0.60 seconds

## BFS RS instance: RS_RSZ2_RSZ2_RSZ1_RSZ1_RSZ2_RSZ2_RSZ1

### Backsubstitution after applying RS history:
0: 0.0009457, 0.0011003, 0.0009457, 0.0011003, -0.0001487, 0.0001490
1: 0.9936197, 0.9940771, 0.9936197, 0.9940771, -0.0003886, 0.0003838
2: -0.0065606, -0.0050208, -0.0065606, -0.0050208, -0.0012023, 0.0011606
3: 0.0038004, 0.0040140, 0.0038004, 0.0040140, -0.0001627, 0.0001661
4: 0.0023852, 0.0036022, 0.0023852, 0.0036022, -0.0010221, 0.0010548
5: 0.0060086, 0.0065111, 0.0060086, 0.0065111, -0.0005024, 0.0005024
6: -0.0013983, -0.0008639, -0.0013983, -0.0008639, -0.0004108, 0.0003964
7: -0.0083221, -0.0079387, -0.0083221, -0.0079387, -0.0003835, 0.0003835
8: 0.0048484, 0.0068715, 0.0048484, 0.0068715, -0.0011823, 0.0012465
9: -0.0036877, -0.0032079, -0.0036877, -0.0032079, -0.0004797, 0.0004797

### Unstable ReLU Count (Linear/Conv2D Layers)
- layer_idx=0, type=LayerType.Linear, total=256, inp1_unstable=5, inp2_unstable=5, delta_unstable=251
- layer_idx=2, type=LayerType.Linear, total=256, inp1_unstable=4, inp2_unstable=4, delta_unstable=255
- layer_idx=4, type=LayerType.Linear, total=256, inp1_unstable=0, inp2_unstable=0, delta_unstable=256
- layer_idx=6, type=LayerType.Linear, total=256, inp1_unstable=1, inp2_unstable=1, delta_unstable=256
- layer_idx=8, type=LayerType.Linear, total=10, inp1_unstable=0, inp2_unstable=0, delta_unstable=10

Time for backsubstitution: 1.31 seconds

### RS candidates at layer 3
type: RSZ, layer: 3, pos: 73
type: RSZ, layer: 3, pos: 81
type: RSZ, layer: 3, pos: 136

Time for candidate selection: 0.13 seconds

### Candidate
type: RSZ, layer: 3, pos: 73

### Candidate
type: RSZ, layer: 3, pos: 81

### Relational analysis RSZ of RS_RSZ2_RSZ2_RSZ1_RSZ1_RSZ2_RSZ2_RSZ1_RSZ1

#### Relational analysis RSZ result of RS_RSZ2_RSZ2_RSZ1_RSZ1_RSZ2_RSZ2_RSZ1_RSZ1
Status: Status.VERIFIED
Output dim: 1, lower bound: -0.0002493, upper bound: 0.0002462
time: 0.56 seconds

### Relational analysis RSZ of RS_RSZ2_RSZ2_RSZ1_RSZ1_RSZ2_RSZ2_RSZ1_RSZ2

#### Relational analysis RSZ result of RS_RSZ2_RSZ2_RSZ1_RSZ1_RSZ2_RSZ2_RSZ1_RSZ2
Status: Status.VERIFIED
Output dim: 1, lower bound: -0.0002493, upper bound: 0.0002462
time: 0.60 seconds

## BFS RS instance: RS_RSZ2_RSZ2_RSZ1_RSZ1_RSZ2_RSZ2_RSZ2

### Backsubstitution after applying RS history:
0: 0.0009457, 0.0011003, 0.0009457, 0.0011003, -0.0001488, 0.0001488
1: 0.9936197, 0.9940771, 0.9936197, 0.9940771, -0.0003860, 0.0003860
2: -0.0065606, -0.0050208, -0.0065606, -0.0050208, -0.0011773, 0.0011842
3: 0.0038004, 0.0040140, 0.0038004, 0.0040140, -0.0001643, 0.0001643
4: 0.0023852, 0.0036022, 0.0023852, 0.0036022, -0.0010407, 0.0010353
5: 0.0060086, 0.0065111, 0.0060086, 0.0065111, -0.0005024, 0.0005024
6: -0.0013983, -0.0008639, -0.0013983, -0.0008639, -0.0004022, 0.0004045
7: -0.0083221, -0.0079387, -0.0083221, -0.0079387, -0.0003835, 0.0003835
8: 0.0048484, 0.0068715, 0.0048484, 0.0068715, -0.0012132, 0.0012115
9: -0.0036877, -0.0032079, -0.0036877, -0.0032079, -0.0004797, 0.0004797

### Unstable ReLU Count (Linear/Conv2D Layers)
- layer_idx=0, type=LayerType.Linear, total=256, inp1_unstable=5, inp2_unstable=5, delta_unstable=251
- layer_idx=2, type=LayerType.Linear, total=256, inp1_unstable=4, inp2_unstable=4, delta_unstable=255
- layer_idx=4, type=LayerType.Linear, total=256, inp1_unstable=0, inp2_unstable=0, delta_unstable=256
- layer_idx=6, type=LayerType.Linear, total=256, inp1_unstable=1, inp2_unstable=1, delta_unstable=256
- layer_idx=8, type=LayerType.Linear, total=10, inp1_unstable=0, inp2_unstable=0, delta_unstable=10

Time for backsubstitution: 1.34 seconds

### RS candidates at layer 3
type: RSZ, layer: 3, pos: 73
type: RSZ, layer: 3, pos: 81
type: RSZ, layer: 3, pos: 136

Time for candidate selection: 0.13 seconds

### Candidate
type: RSZ, layer: 3, pos: 73

### Candidate
type: RSZ, layer: 3, pos: 81

### Relational analysis RSZ of RS_RSZ2_RSZ2_RSZ1_RSZ1_RSZ2_RSZ2_RSZ2_RSZ1

#### Relational analysis RSZ result of RS_RSZ2_RSZ2_RSZ1_RSZ1_RSZ2_RSZ2_RSZ2_RSZ1
Status: Status.VERIFIED
Output dim: 1, lower bound: -0.0002468, upper bound: 0.0002486
time: 0.59 seconds

### Relational analysis RSZ of RS_RSZ2_RSZ2_RSZ1_RSZ1_RSZ2_RSZ2_RSZ2_RSZ2

#### Relational analysis RSZ result of RS_RSZ2_RSZ2_RSZ1_RSZ1_RSZ2_RSZ2_RSZ2_RSZ2
Status: Status.VERIFIED
Output dim: 1, lower bound: -0.0002468, upper bound: 0.0002486
time: 0.59 seconds

## BFS RS instance: RS_RSZ2_RSZ2_RSZ1_RSZ2_RSZ1_RSZ1_RSZ1

### Backsubstitution after applying RS history:
0: 0.0009457, 0.0011003, 0.0009457, 0.0011003, -0.0001486, 0.0001490
1: 0.9936197, 0.9940771, 0.9936197, 0.9940771, -0.0003885, 0.0003837
2: -0.0065606, -0.0050208, -0.0065606, -0.0050208, -0.0012011, 0.0011597
3: 0.0038004, 0.0040140, 0.0038004, 0.0040140, -0.0001626, 0.0001660
4: 0.0023852, 0.0036022, 0.0023852, 0.0036022, -0.0010213, 0.0010538
5: 0.0060086, 0.0065111, 0.0060086, 0.0065111, -0.0005024, 0.0005024
6: -0.0013983, -0.0008639, -0.0013983, -0.0008639, -0.0004104, 0.0003960
7: -0.0083221, -0.0079387, -0.0083221, -0.0079387, -0.0003835, 0.0003835
8: 0.0048484, 0.0068715, 0.0048484, 0.0068715, -0.0011811, 0.0012449
9: -0.0036877, -0.0032079, -0.0036877, -0.0032079, -0.0004797, 0.0004797

### Unstable ReLU Count (Linear/Conv2D Layers)
- layer_idx=0, type=LayerType.Linear, total=256, inp1_unstable=5, inp2_unstable=5, delta_unstable=251
- layer_idx=2, type=LayerType.Linear, total=256, inp1_unstable=4, inp2_unstable=4, delta_unstable=255
- layer_idx=4, type=LayerType.Linear, total=256, inp1_unstable=0, inp2_unstable=0, delta_unstable=256
- layer_idx=6, type=LayerType.Linear, total=256, inp1_unstable=1, inp2_unstable=1, delta_unstable=256
- layer_idx=8, type=LayerType.Linear, total=10, inp1_unstable=0, inp2_unstable=0, delta_unstable=10

Time for backsubstitution: 1.31 seconds

### RS candidates at layer 3
type: RSZ, layer: 3, pos: 73
type: RSZ, layer: 3, pos: 81
type: RSZ, layer: 3, pos: 136

Time for candidate selection: 0.13 seconds

### Candidate
type: RSZ, layer: 3, pos: 73

### Candidate
type: RSZ, layer: 3, pos: 81

### Relational analysis RSZ of RS_RSZ2_RSZ2_RSZ1_RSZ2_RSZ1_RSZ1_RSZ1_RSZ1

#### Relational analysis RSZ result of RS_RSZ2_RSZ2_RSZ1_RSZ2_RSZ1_RSZ1_RSZ1_RSZ1
Status: Status.VERIFIED
Output dim: 1, lower bound: -0.0002486, upper bound: 0.0002466
time: 0.65 seconds

### Relational analysis RSZ of RS_RSZ2_RSZ2_RSZ1_RSZ2_RSZ1_RSZ1_RSZ1_RSZ2

#### Relational analysis RSZ result of RS_RSZ2_RSZ2_RSZ1_RSZ2_RSZ1_RSZ1_RSZ1_RSZ2
Status: Status.VERIFIED
Output dim: 1, lower bound: -0.0002486, upper bound: 0.0002466
time: 0.65 seconds

## BFS RS instance: RS_RSZ2_RSZ2_RSZ1_RSZ2_RSZ1_RSZ1_RSZ2

### Backsubstitution after applying RS history:
0: 0.0009457, 0.0011003, 0.0009457, 0.0011003, -0.0001489, 0.0001489
1: 0.9936197, 0.9940771, 0.9936197, 0.9940771, -0.0003865, 0.0003862
2: -0.0065606, -0.0050208, -0.0065606, -0.0050208, -0.0011820, 0.0011865
3: 0.0038004, 0.0040140, 0.0038004, 0.0040140, -0.0001644, 0.0001646
4: 0.0023852, 0.0036022, 0.0023852, 0.0036022, -0.0010425, 0.0010389
5: 0.0060086, 0.0065111, 0.0060086, 0.0065111, -0.0005024, 0.0005024
6: -0.0013983, -0.0008639, -0.0013983, -0.0008639, -0.0004038, 0.0004053
7: -0.0083221, -0.0079387, -0.0083221, -0.0079387, -0.0003835, 0.0003835
8: 0.0048484, 0.0068715, 0.0048484, 0.0068715, -0.0012163, 0.0012176
9: -0.0036877, -0.0032079, -0.0036877, -0.0032079, -0.0004797, 0.0004797

### Unstable ReLU Count (Linear/Conv2D Layers)
- layer_idx=0, type=LayerType.Linear, total=256, inp1_unstable=5, inp2_unstable=5, delta_unstable=251
- layer_idx=2, type=LayerType.Linear, total=256, inp1_unstable=4, inp2_unstable=4, delta_unstable=255
- layer_idx=4, type=LayerType.Linear, total=256, inp1_unstable=0, inp2_unstable=0, delta_unstable=256
- layer_idx=6, type=LayerType.Linear, total=256, inp1_unstable=1, inp2_unstable=1, delta_unstable=256
- layer_idx=8, type=LayerType.Linear, total=10, inp1_unstable=0, inp2_unstable=0, delta_unstable=10

Time for backsubstitution: 1.30 seconds

### RS candidates at layer 3
type: RSZ, layer: 3, pos: 73
type: RSZ, layer: 3, pos: 81
type: RSZ, layer: 3, pos: 136

Time for candidate selection: 0.13 seconds

### Candidate
type: RSZ, layer: 3, pos: 73

### Candidate
type: RSZ, layer: 3, pos: 81

### Relational analysis RSZ of RS_RSZ2_RSZ2_RSZ1_RSZ2_RSZ1_RSZ1_RSZ2_RSZ1

#### Relational analysis RSZ result of RS_RSZ2_RSZ2_RSZ1_RSZ2_RSZ1_RSZ1_RSZ2_RSZ1
Status: Status.VERIFIED
Output dim: 1, lower bound: -0.0002463, upper bound: 0.0002495
time: 0.59 seconds

### Relational analysis RSZ of RS_RSZ2_RSZ2_RSZ1_RSZ2_RSZ1_RSZ1_RSZ2_RSZ2

#### Relational analysis RSZ result of RS_RSZ2_RSZ2_RSZ1_RSZ2_RSZ1_RSZ1_RSZ2_RSZ2
Status: Status.VERIFIED
Output dim: 1, lower bound: -0.0002463, upper bound: 0.0002495
time: 0.59 seconds

## BFS RS instance: RS_RSZ2_RSZ2_RSZ1_RSZ2_RSZ1_RSZ2_RSZ1

### Backsubstitution after applying RS history:
0: 0.0009457, 0.0011003, 0.0009457, 0.0011003, -0.0001486, 0.0001490
1: 0.9936197, 0.9940771, 0.9936197, 0.9940771, -0.0003885, 0.0003837
2: -0.0065606, -0.0050208, -0.0065606, -0.0050208, -0.0012011, 0.0011597
3: 0.0038004, 0.0040140, 0.0038004, 0.0040140, -0.0001626, 0.0001660
4: 0.0023852, 0.0036022, 0.0023852, 0.0036022, -0.0010213, 0.0010538
5: 0.0060086, 0.0065111, 0.0060086, 0.0065111, -0.0005024, 0.0005024
6: -0.0013983, -0.0008639, -0.0013983, -0.0008639, -0.0004104, 0.0003960
7: -0.0083221, -0.0079387, -0.0083221, -0.0079387, -0.0003835, 0.0003835
8: 0.0048484, 0.0068715, 0.0048484, 0.0068715, -0.0011811, 0.0012449
9: -0.0036877, -0.0032079, -0.0036877, -0.0032079, -0.0004797, 0.0004797

### Unstable ReLU Count (Linear/Conv2D Layers)
- layer_idx=0, type=LayerType.Linear, total=256, inp1_unstable=5, inp2_unstable=5, delta_unstable=251
- layer_idx=2, type=LayerType.Linear, total=256, inp1_unstable=4, inp2_unstable=4, delta_unstable=255
- layer_idx=4, type=LayerType.Linear, total=256, inp1_unstable=0, inp2_unstable=0, delta_unstable=256
- layer_idx=6, type=LayerType.Linear, total=256, inp1_unstable=1, inp2_unstable=1, delta_unstable=256
- layer_idx=8, type=LayerType.Linear, total=10, inp1_unstable=0, inp2_unstable=0, delta_unstable=10

Time for backsubstitution: 1.37 seconds

### RS candidates at layer 3
type: RSZ, layer: 3, pos: 73
type: RSZ, layer: 3, pos: 81
type: RSZ, layer: 3, pos: 136

Time for candidate selection: 0.18 seconds

### Candidate
type: RSZ, layer: 3, pos: 73

### Candidate
type: RSZ, layer: 3, pos: 81

### Relational analysis RSZ of RS_RSZ2_RSZ2_RSZ1_RSZ2_RSZ1_RSZ2_RSZ1_RSZ1

#### Relational analysis RSZ result of RS_RSZ2_RSZ2_RSZ1_RSZ2_RSZ1_RSZ2_RSZ1_RSZ1
Status: Status.VERIFIED
Output dim: 1, lower bound: -0.0002486, upper bound: 0.0002466
time: 0.64 seconds

### Relational analysis RSZ of RS_RSZ2_RSZ2_RSZ1_RSZ2_RSZ1_RSZ2_RSZ1_RSZ2

#### Relational analysis RSZ result of RS_RSZ2_RSZ2_RSZ1_RSZ2_RSZ1_RSZ2_RSZ1_RSZ2
Status: Status.VERIFIED
Output dim: 1, lower bound: -0.0002486, upper bound: 0.0002466
time: 0.63 seconds

## BFS RS instance: RS_RSZ2_RSZ2_RSZ1_RSZ2_RSZ1_RSZ2_RSZ2

### Backsubstitution after applying RS history:
0: 0.0009457, 0.0011003, 0.0009457, 0.0011003, -0.0001489, 0.0001489
1: 0.9936197, 0.9940771, 0.9936197, 0.9940771, -0.0003865, 0.0003862
2: -0.0065606, -0.0050208, -0.0065606, -0.0050208, -0.0011820, 0.0011865
3: 0.0038004, 0.0040140, 0.0038004, 0.0040140, -0.0001644, 0.0001646
4: 0.0023852, 0.0036022, 0.0023852, 0.0036022, -0.0010425, 0.0010389
5: 0.0060086, 0.0065111, 0.0060086, 0.0065111, -0.0005024, 0.0005024
6: -0.0013983, -0.0008639, -0.0013983, -0.0008639, -0.0004038, 0.0004053
7: -0.0083221, -0.0079387, -0.0083221, -0.0079387, -0.0003835, 0.0003835
8: 0.0048484, 0.0068715, 0.0048484, 0.0068715, -0.0012163, 0.0012176
9: -0.0036877, -0.0032079, -0.0036877, -0.0032079, -0.0004797, 0.0004797

### Unstable ReLU Count (Linear/Conv2D Layers)
- layer_idx=0, type=LayerType.Linear, total=256, inp1_unstable=5, inp2_unstable=5, delta_unstable=251
- layer_idx=2, type=LayerType.Linear, total=256, inp1_unstable=4, inp2_unstable=4, delta_unstable=255
- layer_idx=4, type=LayerType.Linear, total=256, inp1_unstable=0, inp2_unstable=0, delta_unstable=256
- layer_idx=6, type=LayerType.Linear, total=256, inp1_unstable=1, inp2_unstable=1, delta_unstable=256
- layer_idx=8, type=LayerType.Linear, total=10, inp1_unstable=0, inp2_unstable=0, delta_unstable=10

Time for backsubstitution: 1.31 seconds

### RS candidates at layer 3
type: RSZ, layer: 3, pos: 73
type: RSZ, layer: 3, pos: 81
type: RSZ, layer: 3, pos: 136

Time for candidate selection: 0.13 seconds

### Candidate
type: RSZ, layer: 3, pos: 73

### Candidate
type: RSZ, layer: 3, pos: 81

### Relational analysis RSZ of RS_RSZ2_RSZ2_RSZ1_RSZ2_RSZ1_RSZ2_RSZ2_RSZ1

#### Relational analysis RSZ result of RS_RSZ2_RSZ2_RSZ1_RSZ2_RSZ1_RSZ2_RSZ2_RSZ1
Status: Status.VERIFIED
Output dim: 1, lower bound: -0.0002463, upper bound: 0.0002495
time: 0.62 seconds

### Relational analysis RSZ of RS_RSZ2_RSZ2_RSZ1_RSZ2_RSZ1_RSZ2_RSZ2_RSZ2

#### Relational analysis RSZ result of RS_RSZ2_RSZ2_RSZ1_RSZ2_RSZ1_RSZ2_RSZ2_RSZ2
Status: Status.VERIFIED
Output dim: 1, lower bound: -0.0002463, upper bound: 0.0002495
time: 0.62 seconds

## BFS RS instance: RS_RSZ2_RSZ2_RSZ1_RSZ2_RSZ2_RSZ1_RSZ1

### Backsubstitution after applying RS history:
0: 0.0009457, 0.0011003, 0.0009457, 0.0011003, -0.0001487, 0.0001490
1: 0.9936197, 0.9940771, 0.9936197, 0.9940771, -0.0003881, 0.0003840
2: -0.0065606, -0.0050208, -0.0065606, -0.0050208, -0.0011969, 0.0011626
3: 0.0038004, 0.0040140, 0.0038004, 0.0040140, -0.0001628, 0.0001657
4: 0.0023852, 0.0036022, 0.0023852, 0.0036022, -0.0010237, 0.0010505
5: 0.0060086, 0.0065111, 0.0060086, 0.0065111, -0.0005024, 0.0005024
6: -0.0013983, -0.0008639, -0.0013983, -0.0008639, -0.0004090, 0.0003971
7: -0.0083221, -0.0079387, -0.0083221, -0.0079387, -0.0003835, 0.0003835
8: 0.0048484, 0.0068715, 0.0048484, 0.0068715, -0.0011850, 0.0012394
9: -0.0036877, -0.0032079, -0.0036877, -0.0032079, -0.0004797, 0.0004797

### Unstable ReLU Count (Linear/Conv2D Layers)
- layer_idx=0, type=LayerType.Linear, total=256, inp1_unstable=5, inp2_unstable=5, delta_unstable=251
- layer_idx=2, type=LayerType.Linear, total=256, inp1_unstable=4, inp2_unstable=4, delta_unstable=255
- layer_idx=4, type=LayerType.Linear, total=256, inp1_unstable=0, inp2_unstable=0, delta_unstable=256
- layer_idx=6, type=LayerType.Linear, total=256, inp1_unstable=1, inp2_unstable=1, delta_unstable=256
- layer_idx=8, type=LayerType.Linear, total=10, inp1_unstable=0, inp2_unstable=0, delta_unstable=10

Time for backsubstitution: 1.31 seconds

## RS Result
status: Status.UNKNOWN
execution time: (base) + (rs) = 2.66 + 597.79 = 600.45 seconds
