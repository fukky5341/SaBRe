## Execution arguments:
Dataset: Dataset.MNIST
Network: onnx/mnist-net_256x4.onnx
Relational property: GLOBAL_ROBUSTNESS
LP Analysis: True
Epsilon: 0.015625
Delta epsilon: 0.0078125
execution index: (2, 2, 0)
Time budget: 600 seconds
Split limit: 100
Threshold: 0.0018531149999999998


## IAR start

### Unstable ReLU Count (Linear/Conv2D Layers)
- layer_idx=0, type=LayerType.Linear, total=256, inp1_unstable=17, inp2_unstable=17, delta_unstable=256
- layer_idx=2, type=LayerType.Linear, total=256, inp1_unstable=27, inp2_unstable=27, delta_unstable=256
- layer_idx=4, type=LayerType.Linear, total=256, inp1_unstable=0, inp2_unstable=0, delta_unstable=256
- layer_idx=6, type=LayerType.Linear, total=256, inp1_unstable=0, inp2_unstable=0, delta_unstable=256
- layer_idx=8, type=LayerType.Linear, total=10, inp1_unstable=2, inp2_unstable=2, delta_unstable=10

### BASE IAR bounds
Layer (inp1_lb, inp1_ub, inp2_lb, inp2_ub, d_lb, d_ub)
0: (-0.0041910, -0.0041119, -0.0041910, -0.0041119, -0.0000791, 0.0000791)
1: (-0.0096089, -0.0066455, -0.0096089, -0.0066455, -0.0029634, 0.0029634)
2: (0.9649323, 0.9684886, 0.9649323, 0.9684886, -0.0035563, 0.0035563)
3: (-0.0123469, 0.0138834, -0.0123469, 0.0138834, -0.0262303, 0.0262303)
4: (-0.0017489, 0.0002460, -0.0017489, 0.0002460, -0.0019950, 0.0019950)
5: (0.0155027, 0.0175190, 0.0155027, 0.0175190, -0.0020163, 0.0020163)
6: (0.0033856, 0.0043663, 0.0033856, 0.0043663, -0.0009807, 0.0009807)
7: (-0.0113762, -0.0045784, -0.0113762, -0.0045784, -0.0067978, 0.0067978)
8: (0.0077038, 0.0130968, 0.0077038, 0.0130968, -0.0053931, 0.0053931)
9: (0.0115806, 0.0212805, 0.0115806, 0.0212805, -0.0096999, 0.0096999)

## BASE Relational Analysis

## BASE Result
execution time: IAR + RelationalAnalysis = 1.26 + 2.01 = 3.27 seconds
status: Status.UNKNOWN
relational distance
Output dim: 2, lower bound: -0.0022462, upper bound: 0.0022461

# Relational Split (RS) starts

## BFS RS instance: RS

Time for backsubstitution: 0.00 seconds

### RS candidates at layer 1
type: RSZ, layer: 1, pos: 122
type: RSZ, layer: 1, pos: 1
type: RSZ, layer: 1, pos: 195
type: RSZ, layer: 1, pos: 114
type: RSZ, layer: 1, pos: 137
type: RSZ, layer: 1, pos: 133
type: RSZ, layer: 1, pos: 230
type: RSZ, layer: 1, pos: 112
type: RSZ, layer: 1, pos: 25
type: RSZ, layer: 1, pos: 84
type: RSZ, layer: 1, pos: 240
type: RSZ, layer: 1, pos: 250
type: RSZ, layer: 1, pos: 173
type: RSZ, layer: 1, pos: 176
type: RSZ, layer: 1, pos: 174
type: RSZ, layer: 1, pos: 29
type: RSZ, layer: 1, pos: 181

Time for candidate selection: 0.00 seconds

### Candidate
type: RSZ, layer: 1, pos: 122

### Relational analysis RSZ of RS_RSZ1

#### Relational analysis RSZ result of RS_RSZ1
Status: Status.UNKNOWN
Output dim: 2, lower bound: -0.0021875, upper bound: 0.0021881
time: 1.15 seconds

### Relational analysis RSZ of RS_RSZ2

#### Relational analysis RSZ result of RS_RSZ2
Status: Status.UNKNOWN
Output dim: 2, lower bound: -0.0021881, upper bound: 0.0021875
time: 1.54 seconds

## Summary of splitting (split count: 0)
- Time for RS candidates: 2.70 seconds
RS_RSZ1, status: Status.UNKNOWN, split count: 1, time: 2.70
Output dim: 2, lower bound: -0.0021875, upper bound: 0.0021881
RS_RSZ2, status: Status.UNKNOWN, split count: 1, time: 2.70
Output dim: 2, lower bound: -0.0021881, upper bound: 0.0021875

## BFS RS instance: RS_RSZ1

### Backsubstitution after applying RS history:
0: -0.0041910, -0.0041119, -0.0041910, -0.0041119, -0.0000791, 0.0000791
1: -0.0096089, -0.0066455, -0.0096089, -0.0066455, -0.0029634, 0.0029634
2: 0.9649323, 0.9684886, 0.9649323, 0.9684886, -0.0035563, 0.0035563
3: -0.0123469, 0.0138834, -0.0123469, 0.0138834, -0.0262303, 0.0262303
4: -0.0017489, 0.0002460, -0.0017489, 0.0002460, -0.0019950, 0.0019950
5: 0.0155027, 0.0175190, 0.0155027, 0.0175190, -0.0020163, 0.0020163
6: 0.0033856, 0.0043663, 0.0033856, 0.0043663, -0.0009807, 0.0009807
7: -0.0113762, -0.0045784, -0.0113762, -0.0045784, -0.0067978, 0.0067978
8: 0.0077038, 0.0130968, 0.0077038, 0.0130968, -0.0053931, 0.0053931
9: 0.0115806, 0.0212805, 0.0115806, 0.0212805, -0.0096999, 0.0096999

### Unstable ReLU Count (Linear/Conv2D Layers)
- layer_idx=0, type=LayerType.Linear, total=256, inp1_unstable=17, inp2_unstable=17, delta_unstable=255
- layer_idx=2, type=LayerType.Linear, total=256, inp1_unstable=27, inp2_unstable=27, delta_unstable=256
- layer_idx=4, type=LayerType.Linear, total=256, inp1_unstable=0, inp2_unstable=0, delta_unstable=256
- layer_idx=6, type=LayerType.Linear, total=256, inp1_unstable=0, inp2_unstable=0, delta_unstable=256
- layer_idx=8, type=LayerType.Linear, total=10, inp1_unstable=2, inp2_unstable=2, delta_unstable=10

Time for backsubstitution: 1.14 seconds

### RS candidates at layer 1
type: RSZ, layer: 1, pos: 114
type: RSZ, layer: 1, pos: 137
type: RSZ, layer: 1, pos: 240
type: RSZ, layer: 1, pos: 29
type: RSZ, layer: 1, pos: 112
type: RSZ, layer: 1, pos: 173
type: RSZ, layer: 1, pos: 84
type: RSZ, layer: 1, pos: 176
type: RSZ, layer: 1, pos: 174
type: RSZ, layer: 1, pos: 230
type: RSZ, layer: 1, pos: 195
type: RSZ, layer: 1, pos: 181
type: RSZ, layer: 1, pos: 25
type: RSZ, layer: 1, pos: 1
type: RSZ, layer: 1, pos: 250
type: RSZ, layer: 1, pos: 133

Time for candidate selection: 0.00 seconds

### Candidate
type: RSZ, layer: 1, pos: 114

### Relational analysis RSZ of RS_RSZ1_RSZ1

#### Relational analysis RSZ result of RS_RSZ1_RSZ1
Status: Status.VERIFIED
Output dim: 2, lower bound: -0.0016162, upper bound: 0.0016162
time: 1.00 seconds

### Relational analysis RSZ of RS_RSZ1_RSZ2

#### Relational analysis RSZ result of RS_RSZ1_RSZ2
Status: Status.VERIFIED
Output dim: 2, lower bound: -0.0016162, upper bound: 0.0016162
time: 1.00 seconds

## BFS RS instance: RS_RSZ2

### Backsubstitution after applying RS history:
0: -0.0041910, -0.0041119, -0.0041910, -0.0041119, -0.0000791, 0.0000791
1: -0.0096089, -0.0066455, -0.0096089, -0.0066455, -0.0029634, 0.0029634
2: 0.9649323, 0.9684886, 0.9649323, 0.9684886, -0.0035563, 0.0035563
3: -0.0123469, 0.0138834, -0.0123469, 0.0138834, -0.0262303, 0.0262303
4: -0.0017489, 0.0002460, -0.0017489, 0.0002460, -0.0019950, 0.0019950
5: 0.0155027, 0.0175190, 0.0155027, 0.0175190, -0.0020163, 0.0020163
6: 0.0033856, 0.0043663, 0.0033856, 0.0043663, -0.0009807, 0.0009807
7: -0.0113762, -0.0045784, -0.0113762, -0.0045784, -0.0067978, 0.0067978
8: 0.0077038, 0.0130968, 0.0077038, 0.0130968, -0.0053931, 0.0053931
9: 0.0115806, 0.0212805, 0.0115806, 0.0212805, -0.0096999, 0.0096999

### Unstable ReLU Count (Linear/Conv2D Layers)
- layer_idx=0, type=LayerType.Linear, total=256, inp1_unstable=17, inp2_unstable=17, delta_unstable=255
- layer_idx=2, type=LayerType.Linear, total=256, inp1_unstable=27, inp2_unstable=27, delta_unstable=256
- layer_idx=4, type=LayerType.Linear, total=256, inp1_unstable=0, inp2_unstable=0, delta_unstable=256
- layer_idx=6, type=LayerType.Linear, total=256, inp1_unstable=0, inp2_unstable=0, delta_unstable=256
- layer_idx=8, type=LayerType.Linear, total=10, inp1_unstable=2, inp2_unstable=2, delta_unstable=10

Time for backsubstitution: 1.13 seconds

### RS candidates at layer 1
type: RSZ, layer: 1, pos: 133
type: RSZ, layer: 1, pos: 1
type: RSZ, layer: 1, pos: 137
type: RSZ, layer: 1, pos: 25
type: RSZ, layer: 1, pos: 181
type: RSZ, layer: 1, pos: 29
type: RSZ, layer: 1, pos: 84
type: RSZ, layer: 1, pos: 250
type: RSZ, layer: 1, pos: 173
type: RSZ, layer: 1, pos: 176
type: RSZ, layer: 1, pos: 230
type: RSZ, layer: 1, pos: 174
type: RSZ, layer: 1, pos: 195
type: RSZ, layer: 1, pos: 112
type: RSZ, layer: 1, pos: 114
type: RSZ, layer: 1, pos: 240

Time for candidate selection: 0.00 seconds

### Candidate
type: RSZ, layer: 1, pos: 133

### Relational analysis RSZ of RS_RSZ2_RSZ1

#### Relational analysis RSZ result of RS_RSZ2_RSZ1
Status: Status.UNKNOWN
Output dim: 2, lower bound: -0.0020783, upper bound: 0.0020762
time: 1.69 seconds

### Relational analysis RSZ of RS_RSZ2_RSZ2

#### Relational analysis RSZ result of RS_RSZ2_RSZ2
Status: Status.UNKNOWN
Output dim: 2, lower bound: -0.0020777, upper bound: 0.0020766
time: 1.56 seconds

## Summary of splitting (split count: 1)
- Time for RS candidates: 4.39 seconds
RS_RSZ1_RSZ1, status: Status.VERIFIED, split count: 2, time: 4.39
Output dim: 2, lower bound: -0.0016162, upper bound: 0.0016162
RS_RSZ1_RSZ2, status: Status.VERIFIED, split count: 2, time: 4.39
Output dim: 2, lower bound: -0.0016162, upper bound: 0.0016162
RS_RSZ2_RSZ1, status: Status.UNKNOWN, split count: 2, time: 4.39
Output dim: 2, lower bound: -0.0020783, upper bound: 0.0020762
RS_RSZ2_RSZ2, status: Status.UNKNOWN, split count: 2, time: 4.39
Output dim: 2, lower bound: -0.0020777, upper bound: 0.0020766

## BFS RS instance: RS_RSZ2_RSZ1

### Backsubstitution after applying RS history:
0: -0.0041910, -0.0041119, -0.0041910, -0.0041119, -0.0000768, 0.0000768
1: -0.0096089, -0.0066455, -0.0096089, -0.0066455, -0.0028774, 0.0028758
2: 0.9649323, 0.9684886, 0.9649323, 0.9684886, -0.0034530, 0.0034511
3: -0.0123469, 0.0138834, -0.0123469, 0.0138834, -0.0254688, 0.0254547
4: -0.0017489, 0.0002460, -0.0017489, 0.0002460, -0.0019360, 0.0019371
5: 0.0155027, 0.0175190, 0.0155027, 0.0175190, -0.0019566, 0.0019577
6: 0.0033856, 0.0043663, 0.0033856, 0.0043663, -0.0009522, 0.0009517
7: -0.0113762, -0.0045784, -0.0113762, -0.0045784, -0.0065968, 0.0066005
8: 0.0077038, 0.0130968, 0.0077038, 0.0130968, -0.0052336, 0.0052365
9: 0.0115806, 0.0212805, 0.0115806, 0.0212805, -0.0094131, 0.0094183

### Unstable ReLU Count (Linear/Conv2D Layers)
- layer_idx=0, type=LayerType.Linear, total=256, inp1_unstable=17, inp2_unstable=17, delta_unstable=254
- layer_idx=2, type=LayerType.Linear, total=256, inp1_unstable=27, inp2_unstable=27, delta_unstable=256
- layer_idx=4, type=LayerType.Linear, total=256, inp1_unstable=0, inp2_unstable=0, delta_unstable=256
- layer_idx=6, type=LayerType.Linear, total=256, inp1_unstable=0, inp2_unstable=0, delta_unstable=256
- layer_idx=8, type=LayerType.Linear, total=10, inp1_unstable=2, inp2_unstable=2, delta_unstable=10

Time for backsubstitution: 1.14 seconds

### RS candidates at layer 1
type: RSZ, layer: 1, pos: 112
type: RSZ, layer: 1, pos: 250
type: RSZ, layer: 1, pos: 1
type: RSZ, layer: 1, pos: 173
type: RSZ, layer: 1, pos: 176
type: RSZ, layer: 1, pos: 195
type: RSZ, layer: 1, pos: 240
type: RSZ, layer: 1, pos: 114
type: RSZ, layer: 1, pos: 174
type: RSZ, layer: 1, pos: 84
type: RSZ, layer: 1, pos: 230
type: RSZ, layer: 1, pos: 137
type: RSZ, layer: 1, pos: 25
type: RSZ, layer: 1, pos: 29
type: RSZ, layer: 1, pos: 181

Time for candidate selection: 0.00 seconds

### Candidate
type: RSZ, layer: 1, pos: 112

### Relational analysis RSZ of RS_RSZ2_RSZ1_RSZ1

#### Relational analysis RSZ result of RS_RSZ2_RSZ1_RSZ1
Status: Status.UNKNOWN
Output dim: 2, lower bound: -0.0018983, upper bound: 0.0018983
time: 1.38 seconds

### Relational analysis RSZ of RS_RSZ2_RSZ1_RSZ2

#### Relational analysis RSZ result of RS_RSZ2_RSZ1_RSZ2
Status: Status.UNKNOWN
Output dim: 2, lower bound: -0.0018983, upper bound: 0.0018983
time: 1.36 seconds

## BFS RS instance: RS_RSZ2_RSZ2

### Backsubstitution after applying RS history:
0: -0.0041910, -0.0041119, -0.0041910, -0.0041119, -0.0000774, 0.0000762
1: -0.0096089, -0.0066455, -0.0096089, -0.0066455, -0.0028991, 0.0028541
2: 0.9649323, 0.9684886, 0.9649323, 0.9684886, -0.0034791, 0.0034250
3: -0.0123469, 0.0138834, -0.0123469, 0.0138834, -0.0256611, 0.0252624
4: -0.0017489, 0.0002460, -0.0017489, 0.0002460, -0.0019214, 0.0019517
5: 0.0155027, 0.0175190, 0.0155027, 0.0175190, -0.0019419, 0.0019725
6: 0.0033856, 0.0043663, 0.0033856, 0.0043663, -0.0009594, 0.0009445
7: -0.0113762, -0.0045784, -0.0113762, -0.0045784, -0.0065470, 0.0066503
8: 0.0077038, 0.0130968, 0.0077038, 0.0130968, -0.0051941, 0.0052760
9: 0.0115806, 0.0212805, 0.0115806, 0.0212805, -0.0093420, 0.0094894

### Unstable ReLU Count (Linear/Conv2D Layers)
- layer_idx=0, type=LayerType.Linear, total=256, inp1_unstable=17, inp2_unstable=17, delta_unstable=254
- layer_idx=2, type=LayerType.Linear, total=256, inp1_unstable=27, inp2_unstable=27, delta_unstable=256
- layer_idx=4, type=LayerType.Linear, total=256, inp1_unstable=0, inp2_unstable=0, delta_unstable=256
- layer_idx=6, type=LayerType.Linear, total=256, inp1_unstable=0, inp2_unstable=0, delta_unstable=256
- layer_idx=8, type=LayerType.Linear, total=10, inp1_unstable=2, inp2_unstable=2, delta_unstable=10

Time for backsubstitution: 1.13 seconds

### RS candidates at layer 1
type: RSZ, layer: 1, pos: 29
type: RSZ, layer: 1, pos: 181
type: RSZ, layer: 1, pos: 230
type: RSZ, layer: 1, pos: 114
type: RSZ, layer: 1, pos: 176
type: RSZ, layer: 1, pos: 240
type: RSZ, layer: 1, pos: 25
type: RSZ, layer: 1, pos: 250
type: RSZ, layer: 1, pos: 1
type: RSZ, layer: 1, pos: 173
type: RSZ, layer: 1, pos: 112
type: RSZ, layer: 1, pos: 137
type: RSZ, layer: 1, pos: 84
type: RSZ, layer: 1, pos: 174
type: RSZ, layer: 1, pos: 195

Time for candidate selection: 0.00 seconds

### Candidate
type: RSZ, layer: 1, pos: 29

### Relational analysis RSZ of RS_RSZ2_RSZ2_RSZ1

#### Relational analysis RSZ result of RS_RSZ2_RSZ2_RSZ1
Status: Status.UNKNOWN
Output dim: 2, lower bound: -0.0019815, upper bound: 0.0019107
time: 1.11 seconds

### Relational analysis RSZ of RS_RSZ2_RSZ2_RSZ2

#### Relational analysis RSZ result of RS_RSZ2_RSZ2_RSZ2
Status: Status.UNKNOWN
Output dim: 2, lower bound: -0.0019139, upper bound: 0.0019812
time: 1.10 seconds

## Summary of splitting (split count: 2)
- Time for RS candidates: 3.35 seconds
RS_RSZ2_RSZ1_RSZ1, status: Status.UNKNOWN, split count: 3, time: 3.35
Output dim: 2, lower bound: -0.0018983, upper bound: 0.0018983
RS_RSZ2_RSZ1_RSZ2, status: Status.UNKNOWN, split count: 3, time: 3.35
Output dim: 2, lower bound: -0.0018983, upper bound: 0.0018983
RS_RSZ2_RSZ2_RSZ1, status: Status.UNKNOWN, split count: 3, time: 3.35
Output dim: 2, lower bound: -0.0019815, upper bound: 0.0019107
RS_RSZ2_RSZ2_RSZ2, status: Status.UNKNOWN, split count: 3, time: 3.35
Output dim: 2, lower bound: -0.0019139, upper bound: 0.0019812

## BFS RS instance: RS_RSZ2_RSZ1_RSZ1

### Backsubstitution after applying RS history:
0: -0.0041910, -0.0041119, -0.0041910, -0.0041119, -0.0000764, 0.0000765
1: -0.0096089, -0.0066455, -0.0096089, -0.0066455, -0.0028618, 0.0028640
2: 0.9649323, 0.9684886, 0.9649323, 0.9684886, -0.0034343, 0.0034369
3: -0.0123469, 0.0138834, -0.0123469, 0.0138834, -0.0253307, 0.0253500
4: -0.0017489, 0.0002460, -0.0017489, 0.0002460, -0.0019280, 0.0019265
5: 0.0155027, 0.0175190, 0.0155027, 0.0175190, -0.0019486, 0.0019471
6: 0.0033856, 0.0043663, 0.0033856, 0.0043663, -0.0009471, 0.0009478
7: -0.0113762, -0.0045784, -0.0113762, -0.0045784, -0.0065697, 0.0065647
8: 0.0077038, 0.0130968, 0.0077038, 0.0130968, -0.0052121, 0.0052081
9: 0.0115806, 0.0212805, 0.0115806, 0.0212805, -0.0093744, 0.0093673

### Unstable ReLU Count (Linear/Conv2D Layers)
- layer_idx=0, type=LayerType.Linear, total=256, inp1_unstable=17, inp2_unstable=17, delta_unstable=253
- layer_idx=2, type=LayerType.Linear, total=256, inp1_unstable=27, inp2_unstable=27, delta_unstable=256
- layer_idx=4, type=LayerType.Linear, total=256, inp1_unstable=0, inp2_unstable=0, delta_unstable=256
- layer_idx=6, type=LayerType.Linear, total=256, inp1_unstable=0, inp2_unstable=0, delta_unstable=256
- layer_idx=8, type=LayerType.Linear, total=10, inp1_unstable=2, inp2_unstable=2, delta_unstable=10

Time for backsubstitution: 1.15 seconds

### RS candidates at layer 1
type: RSZ, layer: 1, pos: 174
type: RSZ, layer: 1, pos: 137
type: RSZ, layer: 1, pos: 195
type: RSZ, layer: 1, pos: 176
type: RSZ, layer: 1, pos: 240
type: RSZ, layer: 1, pos: 84
type: RSZ, layer: 1, pos: 173
type: RSZ, layer: 1, pos: 181
type: RSZ, layer: 1, pos: 25
type: RSZ, layer: 1, pos: 250
type: RSZ, layer: 1, pos: 230
type: RSZ, layer: 1, pos: 1
type: RSZ, layer: 1, pos: 29
type: RSZ, layer: 1, pos: 114

Time for candidate selection: 0.00 seconds

### Candidate
type: RSZ, layer: 1, pos: 174

### Relational analysis RSZ of RS_RSZ2_RSZ1_RSZ1_RSZ1
Optimization infeasible because this subproblem isn't reachable.

### Candidate
type: RSZ, layer: 1, pos: 137

### Relational analysis RSZ of RS_RSZ2_RSZ1_RSZ1_RSZ1

#### Relational analysis RSZ result of RS_RSZ2_RSZ1_RSZ1_RSZ1
Status: Status.UNKNOWN
Output dim: 2, lower bound: -0.0018883, upper bound: 0.0018887
time: 1.02 seconds

### Relational analysis RSZ of RS_RSZ2_RSZ1_RSZ1_RSZ2

#### Relational analysis RSZ result of RS_RSZ2_RSZ1_RSZ1_RSZ2
Status: Status.UNKNOWN
Output dim: 2, lower bound: -0.0018887, upper bound: 0.0018883
time: 1.43 seconds

## BFS RS instance: RS_RSZ2_RSZ1_RSZ2

### Backsubstitution after applying RS history:
0: -0.0041910, -0.0041119, -0.0041910, -0.0041119, -0.0000768, 0.0000764
1: -0.0096089, -0.0066455, -0.0096089, -0.0066455, -0.0028774, 0.0028604
2: 0.9649323, 0.9684886, 0.9649323, 0.9684886, -0.0034530, 0.0034326
3: -0.0123469, 0.0138834, -0.0123469, 0.0138834, -0.0254688, 0.0253181
4: -0.0017489, 0.0002460, -0.0017489, 0.0002460, -0.0019256, 0.0019371
5: 0.0155027, 0.0175190, 0.0155027, 0.0175190, -0.0019461, 0.0019577
6: 0.0033856, 0.0043663, 0.0033856, 0.0043663, -0.0009522, 0.0009466
7: -0.0113762, -0.0045784, -0.0113762, -0.0045784, -0.0065614, 0.0066005
8: 0.0077038, 0.0130968, 0.0077038, 0.0130968, -0.0052055, 0.0052365
9: 0.0115806, 0.0212805, 0.0115806, 0.0212805, -0.0093626, 0.0094183

### Unstable ReLU Count (Linear/Conv2D Layers)
- layer_idx=0, type=LayerType.Linear, total=256, inp1_unstable=17, inp2_unstable=17, delta_unstable=253
- layer_idx=2, type=LayerType.Linear, total=256, inp1_unstable=27, inp2_unstable=27, delta_unstable=256
- layer_idx=4, type=LayerType.Linear, total=256, inp1_unstable=0, inp2_unstable=0, delta_unstable=256
- layer_idx=6, type=LayerType.Linear, total=256, inp1_unstable=0, inp2_unstable=0, delta_unstable=256
- layer_idx=8, type=LayerType.Linear, total=10, inp1_unstable=2, inp2_unstable=2, delta_unstable=10

Time for backsubstitution: 1.13 seconds

### RS candidates at layer 1
type: RSZ, layer: 1, pos: 250
type: RSZ, layer: 1, pos: 181
type: RSZ, layer: 1, pos: 25
type: RSZ, layer: 1, pos: 230
type: RSZ, layer: 1, pos: 174
type: RSZ, layer: 1, pos: 176
type: RSZ, layer: 1, pos: 195
type: RSZ, layer: 1, pos: 1
type: RSZ, layer: 1, pos: 173
type: RSZ, layer: 1, pos: 114
type: RSZ, layer: 1, pos: 137
type: RSZ, layer: 1, pos: 240
type: RSZ, layer: 1, pos: 84
type: RSZ, layer: 1, pos: 29

Time for candidate selection: 0.00 seconds

### Candidate
type: RSZ, layer: 1, pos: 250

### Relational analysis RSZ of RS_RSZ2_RSZ1_RSZ2_RSZ1
Optimization infeasible because this subproblem isn't reachable.

### Candidate
type: RSZ, layer: 1, pos: 181

### Relational analysis RSZ of RS_RSZ2_RSZ1_RSZ2_RSZ1

#### Relational analysis RSZ result of RS_RSZ2_RSZ1_RSZ2_RSZ1
Status: Status.UNKNOWN
Output dim: 2, lower bound: -0.0018975, upper bound: 0.0018974
time: 1.04 seconds

### Relational analysis RSZ of RS_RSZ2_RSZ1_RSZ2_RSZ2

#### Relational analysis RSZ result of RS_RSZ2_RSZ1_RSZ2_RSZ2
Status: Status.UNKNOWN
Output dim: 2, lower bound: -0.0018974, upper bound: 0.0018975
time: 1.04 seconds

## BFS RS instance: RS_RSZ2_RSZ2_RSZ1

### Backsubstitution after applying RS history:
0: -0.0041910, -0.0041119, -0.0041910, -0.0041119, -0.0000697, 0.0000676
1: -0.0096089, -0.0066455, -0.0096089, -0.0066455, -0.0026092, 0.0025306
2: 0.9649323, 0.9684886, 0.9649323, 0.9684886, -0.0031311, 0.0030368
3: -0.0123469, 0.0138834, -0.0123469, 0.0138834, -0.0230945, 0.0223988
4: -0.0017489, 0.0002460, -0.0017489, 0.0002460, -0.0017036, 0.0017565
5: 0.0155027, 0.0175190, 0.0155027, 0.0175190, -0.0017218, 0.0017752
6: 0.0033856, 0.0043663, 0.0033856, 0.0043663, -0.0008635, 0.0008375
7: -0.0113762, -0.0045784, -0.0113762, -0.0045784, -0.0058049, 0.0059851
8: 0.0077038, 0.0130968, 0.0077038, 0.0130968, -0.0046053, 0.0047483
9: 0.0115806, 0.0212805, 0.0115806, 0.0212805, -0.0082831, 0.0085403

### Unstable ReLU Count (Linear/Conv2D Layers)
- layer_idx=0, type=LayerType.Linear, total=256, inp1_unstable=17, inp2_unstable=17, delta_unstable=253
- layer_idx=2, type=LayerType.Linear, total=256, inp1_unstable=27, inp2_unstable=27, delta_unstable=256
- layer_idx=4, type=LayerType.Linear, total=256, inp1_unstable=0, inp2_unstable=0, delta_unstable=256
- layer_idx=6, type=LayerType.Linear, total=256, inp1_unstable=0, inp2_unstable=0, delta_unstable=256
- layer_idx=8, type=LayerType.Linear, total=10, inp1_unstable=2, inp2_unstable=2, delta_unstable=10

Time for backsubstitution: 1.18 seconds

### RS candidates at layer 1
type: RSZ, layer: 1, pos: 84
type: RSZ, layer: 1, pos: 112
type: RSZ, layer: 1, pos: 174
type: RSZ, layer: 1, pos: 137
type: RSZ, layer: 1, pos: 195
type: RSZ, layer: 1, pos: 173
type: RSZ, layer: 1, pos: 230
type: RSZ, layer: 1, pos: 181
type: RSZ, layer: 1, pos: 114
type: RSZ, layer: 1, pos: 1
type: RSZ, layer: 1, pos: 240
type: RSZ, layer: 1, pos: 25
type: RSZ, layer: 1, pos: 250
type: RSZ, layer: 1, pos: 176

Time for candidate selection: 0.00 seconds

### Candidate
type: RSZ, layer: 1, pos: 84

### Relational analysis RSZ of RS_RSZ2_RSZ2_RSZ1_RSZ1

#### Relational analysis RSZ result of RS_RSZ2_RSZ2_RSZ1_RSZ1
Status: Status.UNKNOWN
Output dim: 2, lower bound: -0.0019349, upper bound: 0.0018518
time: 1.09 seconds

### Relational analysis RSZ of RS_RSZ2_RSZ2_RSZ1_RSZ2

#### Relational analysis RSZ result of RS_RSZ2_RSZ2_RSZ1_RSZ2
Status: Status.UNKNOWN
Output dim: 2, lower bound: -0.0019208, upper bound: 0.0018632
time: 1.11 seconds

## BFS RS instance: RS_RSZ2_RSZ2_RSZ2

### Backsubstitution after applying RS history:
0: -0.0041910, -0.0041119, -0.0041910, -0.0041119, -0.0000688, 0.0000686
1: -0.0096089, -0.0066455, -0.0096089, -0.0066455, -0.0025760, 0.0025685
2: 0.9649323, 0.9684886, 0.9649323, 0.9684886, -0.0030913, 0.0030823
3: -0.0123469, 0.0138834, -0.0123469, 0.0138834, -0.0228006, 0.0227345
4: -0.0017489, 0.0002460, -0.0017489, 0.0002460, -0.0017291, 0.0017341
5: 0.0155027, 0.0175190, 0.0155027, 0.0175190, -0.0017476, 0.0017526
6: 0.0033856, 0.0043663, 0.0033856, 0.0043663, -0.0008525, 0.0008500
7: -0.0113762, -0.0045784, -0.0113762, -0.0045784, -0.0058918, 0.0059090
8: 0.0077038, 0.0130968, 0.0077038, 0.0130968, -0.0046743, 0.0046879
9: 0.0115806, 0.0212805, 0.0115806, 0.0212805, -0.0084072, 0.0084316

### Unstable ReLU Count (Linear/Conv2D Layers)
- layer_idx=0, type=LayerType.Linear, total=256, inp1_unstable=17, inp2_unstable=17, delta_unstable=253
- layer_idx=2, type=LayerType.Linear, total=256, inp1_unstable=27, inp2_unstable=27, delta_unstable=256
- layer_idx=4, type=LayerType.Linear, total=256, inp1_unstable=0, inp2_unstable=0, delta_unstable=256
- layer_idx=6, type=LayerType.Linear, total=256, inp1_unstable=0, inp2_unstable=0, delta_unstable=256
- layer_idx=8, type=LayerType.Linear, total=10, inp1_unstable=2, inp2_unstable=2, delta_unstable=10

Time for backsubstitution: 1.13 seconds

### RS candidates at layer 1
type: RSZ, layer: 1, pos: 112
type: RSZ, layer: 1, pos: 137
type: RSZ, layer: 1, pos: 173
type: RSZ, layer: 1, pos: 240
type: RSZ, layer: 1, pos: 84
type: RSZ, layer: 1, pos: 1
type: RSZ, layer: 1, pos: 174
type: RSZ, layer: 1, pos: 250
type: RSZ, layer: 1, pos: 176
type: RSZ, layer: 1, pos: 181
type: RSZ, layer: 1, pos: 195
type: RSZ, layer: 1, pos: 230
type: RSZ, layer: 1, pos: 114
type: RSZ, layer: 1, pos: 25

Time for candidate selection: 0.00 seconds

### Candidate
type: RSZ, layer: 1, pos: 112

### Relational analysis RSZ of RS_RSZ2_RSZ2_RSZ2_RSZ1

#### Relational analysis RSZ result of RS_RSZ2_RSZ2_RSZ2_RSZ1
Status: Status.VERIFIED
Output dim: 2, lower bound: -0.0017337, upper bound: 0.0017939
time: 1.35 seconds

### Relational analysis RSZ of RS_RSZ2_RSZ2_RSZ2_RSZ2

#### Relational analysis RSZ result of RS_RSZ2_RSZ2_RSZ2_RSZ2
Status: Status.VERIFIED
Output dim: 2, lower bound: -0.0017337, upper bound: 0.0017939
time: 1.35 seconds

## Summary of splitting (split count: 3)
- Time for RS candidates: 3.85 seconds
RS_RSZ2_RSZ1_RSZ1_RSZ1, status: Status.UNKNOWN, split count: 4, time: 3.85
Output dim: 2, lower bound: -0.0018883, upper bound: 0.0018887
RS_RSZ2_RSZ1_RSZ1_RSZ2, status: Status.UNKNOWN, split count: 4, time: 3.85
Output dim: 2, lower bound: -0.0018887, upper bound: 0.0018883
RS_RSZ2_RSZ1_RSZ2_RSZ1, status: Status.UNKNOWN, split count: 4, time: 3.85
Output dim: 2, lower bound: -0.0018975, upper bound: 0.0018974
RS_RSZ2_RSZ1_RSZ2_RSZ2, status: Status.UNKNOWN, split count: 4, time: 3.85
Output dim: 2, lower bound: -0.0018974, upper bound: 0.0018975
RS_RSZ2_RSZ2_RSZ1_RSZ1, status: Status.UNKNOWN, split count: 4, time: 3.85
Output dim: 2, lower bound: -0.0019349, upper bound: 0.0018518
RS_RSZ2_RSZ2_RSZ1_RSZ2, status: Status.UNKNOWN, split count: 4, time: 3.85
Output dim: 2, lower bound: -0.0019208, upper bound: 0.0018632
RS_RSZ2_RSZ2_RSZ2_RSZ1, status: Status.VERIFIED, split count: 4, time: 3.85
Output dim: 2, lower bound: -0.0017337, upper bound: 0.0017939
RS_RSZ2_RSZ2_RSZ2_RSZ2, status: Status.VERIFIED, split count: 4, time: 3.85
Output dim: 2, lower bound: -0.0017337, upper bound: 0.0017939

## BFS RS instance: RS_RSZ2_RSZ1_RSZ1_RSZ1

### Backsubstitution after applying RS history:
0: -0.0041910, -0.0041119, -0.0041910, -0.0041119, -0.0000765, 0.0000766
1: -0.0096089, -0.0066455, -0.0096089, -0.0066455, -0.0028635, 0.0028669
2: 0.9649323, 0.9684886, 0.9649323, 0.9684886, -0.0034363, 0.0034404
3: -0.0123469, 0.0138834, -0.0123469, 0.0138834, -0.0253454, 0.0253758
4: -0.0017489, 0.0002460, -0.0017489, 0.0002460, -0.0019300, 0.0019277
5: 0.0155027, 0.0175190, 0.0155027, 0.0175190, -0.0019506, 0.0019482
6: 0.0033856, 0.0043663, 0.0033856, 0.0043663, -0.0009476, 0.0009488
7: -0.0113762, -0.0045784, -0.0113762, -0.0045784, -0.0065764, 0.0065685
8: 0.0077038, 0.0130968, 0.0077038, 0.0130968, -0.0052174, 0.0052111
9: 0.0115806, 0.0212805, 0.0115806, 0.0212805, -0.0093839, 0.0093727

### Unstable ReLU Count (Linear/Conv2D Layers)
- layer_idx=0, type=LayerType.Linear, total=256, inp1_unstable=17, inp2_unstable=17, delta_unstable=252
- layer_idx=2, type=LayerType.Linear, total=256, inp1_unstable=27, inp2_unstable=27, delta_unstable=256
- layer_idx=4, type=LayerType.Linear, total=256, inp1_unstable=0, inp2_unstable=0, delta_unstable=256
- layer_idx=6, type=LayerType.Linear, total=256, inp1_unstable=0, inp2_unstable=0, delta_unstable=256
- layer_idx=8, type=LayerType.Linear, total=10, inp1_unstable=2, inp2_unstable=2, delta_unstable=10

Time for backsubstitution: 1.14 seconds

### RS candidates at layer 1
type: RSZ, layer: 1, pos: 25
type: RSZ, layer: 1, pos: 174
type: RSZ, layer: 1, pos: 1
type: RSZ, layer: 1, pos: 195
type: RSZ, layer: 1, pos: 181
type: RSZ, layer: 1, pos: 176
type: RSZ, layer: 1, pos: 173
type: RSZ, layer: 1, pos: 240
type: RSZ, layer: 1, pos: 114
type: RSZ, layer: 1, pos: 29
type: RSZ, layer: 1, pos: 84
type: RSZ, layer: 1, pos: 250
type: RSZ, layer: 1, pos: 230

Time for candidate selection: 0.00 seconds

### Candidate
type: RSZ, layer: 1, pos: 25

### Relational analysis RSZ of RS_RSZ2_RSZ1_RSZ1_RSZ1_RSZ1
Optimization infeasible because this subproblem isn't reachable.

### Candidate
type: RSZ, layer: 1, pos: 174

### Relational analysis RSZ of RS_RSZ2_RSZ1_RSZ1_RSZ1_RSZ1
Optimization infeasible because this subproblem isn't reachable.

### Candidate
type: RSZ, layer: 1, pos: 1

### Relational analysis RSZ of RS_RSZ2_RSZ1_RSZ1_RSZ1_RSZ1

#### Relational analysis RSZ result of RS_RSZ2_RSZ1_RSZ1_RSZ1_RSZ1
Status: Status.UNKNOWN
Output dim: 2, lower bound: -0.0018658, upper bound: 0.0018734
time: 1.04 seconds

### Relational analysis RSZ of RS_RSZ2_RSZ1_RSZ1_RSZ1_RSZ2

#### Relational analysis RSZ result of RS_RSZ2_RSZ1_RSZ1_RSZ1_RSZ2
Status: Status.UNKNOWN
Output dim: 2, lower bound: -0.0018731, upper bound: 0.0018658
time: 1.41 seconds

## BFS RS instance: RS_RSZ2_RSZ1_RSZ1_RSZ2

### Backsubstitution after applying RS history:
0: -0.0041910, -0.0041119, -0.0041910, -0.0041119, -0.0000765, 0.0000765
1: -0.0096089, -0.0066455, -0.0096089, -0.0066455, -0.0028647, 0.0028655
2: 0.9649323, 0.9684886, 0.9649323, 0.9684886, -0.0034377, 0.0034387
3: -0.0123469, 0.0138834, -0.0123469, 0.0138834, -0.0253560, 0.0253634
4: -0.0017489, 0.0002460, -0.0017489, 0.0002460, -0.0019290, 0.0019285
5: 0.0155027, 0.0175190, 0.0155027, 0.0175190, -0.0019496, 0.0019491
6: 0.0033856, 0.0043663, 0.0033856, 0.0043663, -0.0009480, 0.0009483
7: -0.0113762, -0.0045784, -0.0113762, -0.0045784, -0.0065732, 0.0065712
8: 0.0077038, 0.0130968, 0.0077038, 0.0130968, -0.0052148, 0.0052133
9: 0.0115806, 0.0212805, 0.0115806, 0.0212805, -0.0093794, 0.0093766

### Unstable ReLU Count (Linear/Conv2D Layers)
- layer_idx=0, type=LayerType.Linear, total=256, inp1_unstable=17, inp2_unstable=17, delta_unstable=252
- layer_idx=2, type=LayerType.Linear, total=256, inp1_unstable=27, inp2_unstable=27, delta_unstable=256
- layer_idx=4, type=LayerType.Linear, total=256, inp1_unstable=0, inp2_unstable=0, delta_unstable=256
- layer_idx=6, type=LayerType.Linear, total=256, inp1_unstable=0, inp2_unstable=0, delta_unstable=256
- layer_idx=8, type=LayerType.Linear, total=10, inp1_unstable=2, inp2_unstable=2, delta_unstable=10

Time for backsubstitution: 1.13 seconds

### RS candidates at layer 1
type: RSZ, layer: 1, pos: 1
type: RSZ, layer: 1, pos: 174
type: RSZ, layer: 1, pos: 195
type: RSZ, layer: 1, pos: 84
type: RSZ, layer: 1, pos: 176
type: RSZ, layer: 1, pos: 173
type: RSZ, layer: 1, pos: 29
type: RSZ, layer: 1, pos: 25
type: RSZ, layer: 1, pos: 181
type: RSZ, layer: 1, pos: 230
type: RSZ, layer: 1, pos: 240
type: RSZ, layer: 1, pos: 114
type: RSZ, layer: 1, pos: 250

Time for candidate selection: 0.00 seconds

### Candidate
type: RSZ, layer: 1, pos: 1

### Relational analysis RSZ of RS_RSZ2_RSZ1_RSZ1_RSZ2_RSZ1

#### Relational analysis RSZ result of RS_RSZ2_RSZ1_RSZ1_RSZ2_RSZ1
Status: Status.UNKNOWN
Output dim: 2, lower bound: -0.0018658, upper bound: 0.0018731
time: 1.05 seconds

### Relational analysis RSZ of RS_RSZ2_RSZ1_RSZ1_RSZ2_RSZ2

#### Relational analysis RSZ result of RS_RSZ2_RSZ1_RSZ1_RSZ2_RSZ2
Status: Status.UNKNOWN
Output dim: 2, lower bound: -0.0018734, upper bound: 0.0018658
time: 1.03 seconds

## BFS RS instance: RS_RSZ2_RSZ1_RSZ2_RSZ1

### Backsubstitution after applying RS history:
0: -0.0041910, -0.0041119, -0.0041910, -0.0041119, -0.0000773, 0.0000768
1: -0.0096089, -0.0066455, -0.0096089, -0.0066455, -0.0028963, 0.0028753
2: 0.9649323, 0.9684886, 0.9649323, 0.9684886, -0.0034757, 0.0034505
3: -0.0123469, 0.0138834, -0.0123469, 0.0138834, -0.0256362, 0.0254499
4: -0.0017489, 0.0002460, -0.0017489, 0.0002460, -0.0019356, 0.0019498
5: 0.0155027, 0.0175190, 0.0155027, 0.0175190, -0.0019563, 0.0019706
6: 0.0033856, 0.0043663, 0.0033856, 0.0043663, -0.0009585, 0.0009515
7: -0.0113762, -0.0045784, -0.0113762, -0.0045784, -0.0065956, 0.0066438
8: 0.0077038, 0.0130968, 0.0077038, 0.0130968, -0.0052326, 0.0052709
9: 0.0115806, 0.0212805, 0.0115806, 0.0212805, -0.0094114, 0.0094802

### Unstable ReLU Count (Linear/Conv2D Layers)
- layer_idx=0, type=LayerType.Linear, total=256, inp1_unstable=17, inp2_unstable=17, delta_unstable=252
- layer_idx=2, type=LayerType.Linear, total=256, inp1_unstable=27, inp2_unstable=27, delta_unstable=256
- layer_idx=4, type=LayerType.Linear, total=256, inp1_unstable=0, inp2_unstable=0, delta_unstable=256
- layer_idx=6, type=LayerType.Linear, total=256, inp1_unstable=0, inp2_unstable=0, delta_unstable=256
- layer_idx=8, type=LayerType.Linear, total=10, inp1_unstable=2, inp2_unstable=2, delta_unstable=10

Time for backsubstitution: 1.13 seconds

### RS candidates at layer 1
type: RSZ, layer: 1, pos: 250
type: RSZ, layer: 1, pos: 195
type: RSZ, layer: 1, pos: 230
type: RSZ, layer: 1, pos: 176
type: RSZ, layer: 1, pos: 173
type: RSZ, layer: 1, pos: 137
type: RSZ, layer: 1, pos: 174
type: RSZ, layer: 1, pos: 114
type: RSZ, layer: 1, pos: 240
type: RSZ, layer: 1, pos: 1
type: RSZ, layer: 1, pos: 29
type: RSZ, layer: 1, pos: 25
type: RSZ, layer: 1, pos: 84

Time for candidate selection: 0.00 seconds

### Candidate
type: RSZ, layer: 1, pos: 250

### Relational analysis RSZ of RS_RSZ2_RSZ1_RSZ2_RSZ1_RSZ1
Optimization infeasible because this subproblem isn't reachable.

### Candidate
type: RSZ, layer: 1, pos: 195

### Relational analysis RSZ of RS_RSZ2_RSZ1_RSZ2_RSZ1_RSZ1

#### Relational analysis RSZ result of RS_RSZ2_RSZ1_RSZ2_RSZ1_RSZ1
Status: Status.VERIFIED
Output dim: 2, lower bound: -0.0017565, upper bound: 0.0017567
time: 1.37 seconds

### Relational analysis RSZ of RS_RSZ2_RSZ1_RSZ2_RSZ1_RSZ2

#### Relational analysis RSZ result of RS_RSZ2_RSZ1_RSZ2_RSZ1_RSZ2
Status: Status.VERIFIED
Output dim: 2, lower bound: -0.0017565, upper bound: 0.0017567
time: 1.36 seconds

## BFS RS instance: RS_RSZ2_RSZ1_RSZ2_RSZ2

### Backsubstitution after applying RS history:
0: -0.0041910, -0.0041119, -0.0041910, -0.0041119, -0.0000772, 0.0000769
1: -0.0096089, -0.0066455, -0.0096089, -0.0066455, -0.0028927, 0.0028780
2: 0.9649323, 0.9684886, 0.9649323, 0.9684886, -0.0034714, 0.0034537
3: -0.0123469, 0.0138834, -0.0123469, 0.0138834, -0.0256046, 0.0254737
4: -0.0017489, 0.0002460, -0.0017489, 0.0002460, -0.0019374, 0.0019474
5: 0.0155027, 0.0175190, 0.0155027, 0.0175190, -0.0019581, 0.0019682
6: 0.0033856, 0.0043663, 0.0033856, 0.0043663, -0.0009573, 0.0009524
7: -0.0113762, -0.0045784, -0.0113762, -0.0045784, -0.0066017, 0.0066357
8: 0.0077038, 0.0130968, 0.0077038, 0.0130968, -0.0052375, 0.0052644
9: 0.0115806, 0.0212805, 0.0115806, 0.0212805, -0.0094202, 0.0094686

### Unstable ReLU Count (Linear/Conv2D Layers)
- layer_idx=0, type=LayerType.Linear, total=256, inp1_unstable=17, inp2_unstable=17, delta_unstable=252
- layer_idx=2, type=LayerType.Linear, total=256, inp1_unstable=27, inp2_unstable=27, delta_unstable=256
- layer_idx=4, type=LayerType.Linear, total=256, inp1_unstable=0, inp2_unstable=0, delta_unstable=256
- layer_idx=6, type=LayerType.Linear, total=256, inp1_unstable=0, inp2_unstable=0, delta_unstable=256
- layer_idx=8, type=LayerType.Linear, total=10, inp1_unstable=2, inp2_unstable=2, delta_unstable=10

Time for backsubstitution: 1.18 seconds

### RS candidates at layer 1
type: RSZ, layer: 1, pos: 240
type: RSZ, layer: 1, pos: 25
type: RSZ, layer: 1, pos: 230
type: RSZ, layer: 1, pos: 250
type: RSZ, layer: 1, pos: 84
type: RSZ, layer: 1, pos: 1
type: RSZ, layer: 1, pos: 176
type: RSZ, layer: 1, pos: 114
type: RSZ, layer: 1, pos: 137
type: RSZ, layer: 1, pos: 195
type: RSZ, layer: 1, pos: 173
type: RSZ, layer: 1, pos: 174
type: RSZ, layer: 1, pos: 29

Time for candidate selection: 0.00 seconds

### Candidate
type: RSZ, layer: 1, pos: 240

### Relational analysis RSZ of RS_RSZ2_RSZ1_RSZ2_RSZ2_RSZ1
Optimization infeasible because this subproblem isn't reachable.

### Candidate
type: RSZ, layer: 1, pos: 25

### Relational analysis RSZ of RS_RSZ2_RSZ1_RSZ2_RSZ2_RSZ1
Optimization infeasible because this subproblem isn't reachable.

### Candidate
type: RSZ, layer: 1, pos: 230

### Relational analysis RSZ of RS_RSZ2_RSZ1_RSZ2_RSZ2_RSZ1

#### Relational analysis RSZ result of RS_RSZ2_RSZ1_RSZ2_RSZ2_RSZ1
Status: Status.VERIFIED
Output dim: 2, lower bound: -0.0013020, upper bound: 0.0013021
time: 0.80 seconds

### Relational analysis RSZ of RS_RSZ2_RSZ1_RSZ2_RSZ2_RSZ2

#### Relational analysis RSZ result of RS_RSZ2_RSZ1_RSZ2_RSZ2_RSZ2
Status: Status.VERIFIED
Output dim: 2, lower bound: -0.0013020, upper bound: 0.0013021
time: 0.80 seconds

## BFS RS instance: RS_RSZ2_RSZ2_RSZ1_RSZ1

### Backsubstitution after applying RS history:
0: -0.0041910, -0.0041119, -0.0041910, -0.0041119, -0.0000683, 0.0000662
1: -0.0096089, -0.0066455, -0.0096089, -0.0066455, -0.0025560, 0.0024779
2: 0.9649323, 0.9684886, 0.9649323, 0.9684886, -0.0030673, 0.0029736
3: -0.0123469, 0.0138834, -0.0123469, 0.0138834, -0.0226240, 0.0219327
4: -0.0017489, 0.0002460, -0.0017489, 0.0002460, -0.0016681, 0.0017207
5: 0.0155027, 0.0175190, 0.0155027, 0.0175190, -0.0016859, 0.0017391
6: 0.0033856, 0.0043663, 0.0033856, 0.0043663, -0.0008459, 0.0008200
7: -0.0113762, -0.0045784, -0.0113762, -0.0045784, -0.0056840, 0.0058632
8: 0.0077038, 0.0130968, 0.0077038, 0.0130968, -0.0045094, 0.0046516
9: 0.0115806, 0.0212805, 0.0115806, 0.0212805, -0.0081107, 0.0083663

### Unstable ReLU Count (Linear/Conv2D Layers)
- layer_idx=0, type=LayerType.Linear, total=256, inp1_unstable=17, inp2_unstable=17, delta_unstable=252
- layer_idx=2, type=LayerType.Linear, total=256, inp1_unstable=27, inp2_unstable=27, delta_unstable=256
- layer_idx=4, type=LayerType.Linear, total=256, inp1_unstable=0, inp2_unstable=0, delta_unstable=256
- layer_idx=6, type=LayerType.Linear, total=256, inp1_unstable=0, inp2_unstable=0, delta_unstable=256
- layer_idx=8, type=LayerType.Linear, total=10, inp1_unstable=2, inp2_unstable=2, delta_unstable=10

Time for backsubstitution: 1.15 seconds

### RS candidates at layer 1
type: RSZ, layer: 1, pos: 1
type: RSZ, layer: 1, pos: 176
type: RSZ, layer: 1, pos: 181
type: RSZ, layer: 1, pos: 173
type: RSZ, layer: 1, pos: 240
type: RSZ, layer: 1, pos: 137
type: RSZ, layer: 1, pos: 25
type: RSZ, layer: 1, pos: 112
type: RSZ, layer: 1, pos: 230
type: RSZ, layer: 1, pos: 114
type: RSZ, layer: 1, pos: 250
type: RSZ, layer: 1, pos: 174
type: RSZ, layer: 1, pos: 195

Time for candidate selection: 0.00 seconds

### Candidate
type: RSZ, layer: 1, pos: 1

### Relational analysis RSZ of RS_RSZ2_RSZ2_RSZ1_RSZ1_RSZ1

#### Relational analysis RSZ result of RS_RSZ2_RSZ2_RSZ1_RSZ1_RSZ1
Status: Status.UNKNOWN
Output dim: 2, lower bound: -0.0019037, upper bound: 0.0018324
time: 1.14 seconds

### Relational analysis RSZ of RS_RSZ2_RSZ2_RSZ1_RSZ1_RSZ2

#### Relational analysis RSZ result of RS_RSZ2_RSZ2_RSZ1_RSZ1_RSZ2
Status: Status.UNKNOWN
Output dim: 2, lower bound: -0.0019156, upper bound: 0.0018258
time: 1.10 seconds

## BFS RS instance: RS_RSZ2_RSZ2_RSZ1_RSZ2

### Backsubstitution after applying RS history:
0: -0.0041910, -0.0041119, -0.0041910, -0.0041119, -0.0000683, 0.0000661
1: -0.0096089, -0.0066455, -0.0096089, -0.0066455, -0.0025565, 0.0024759
2: 0.9649323, 0.9684886, 0.9649323, 0.9684886, -0.0030680, 0.0029712
3: -0.0123469, 0.0138834, -0.0123469, 0.0138834, -0.0226288, 0.0219153
4: -0.0017489, 0.0002460, -0.0017489, 0.0002460, -0.0016668, 0.0017211
5: 0.0155027, 0.0175190, 0.0155027, 0.0175190, -0.0016846, 0.0017394
6: 0.0033856, 0.0043663, 0.0033856, 0.0043663, -0.0008461, 0.0008194
7: -0.0113762, -0.0045784, -0.0113762, -0.0045784, -0.0056795, 0.0058645
8: 0.0077038, 0.0130968, 0.0077038, 0.0130968, -0.0045059, 0.0046526
9: 0.0115806, 0.0212805, 0.0115806, 0.0212805, -0.0081043, 0.0083681

### Unstable ReLU Count (Linear/Conv2D Layers)
- layer_idx=0, type=LayerType.Linear, total=256, inp1_unstable=17, inp2_unstable=17, delta_unstable=252
- layer_idx=2, type=LayerType.Linear, total=256, inp1_unstable=27, inp2_unstable=27, delta_unstable=256
- layer_idx=4, type=LayerType.Linear, total=256, inp1_unstable=0, inp2_unstable=0, delta_unstable=256
- layer_idx=6, type=LayerType.Linear, total=256, inp1_unstable=0, inp2_unstable=0, delta_unstable=256
- layer_idx=8, type=LayerType.Linear, total=10, inp1_unstable=2, inp2_unstable=2, delta_unstable=10

Time for backsubstitution: 1.13 seconds

### RS candidates at layer 1
type: RSZ, layer: 1, pos: 137
type: RSZ, layer: 1, pos: 195
type: RSZ, layer: 1, pos: 176
type: RSZ, layer: 1, pos: 114
type: RSZ, layer: 1, pos: 181
type: RSZ, layer: 1, pos: 240
type: RSZ, layer: 1, pos: 1
type: RSZ, layer: 1, pos: 174
type: RSZ, layer: 1, pos: 173
type: RSZ, layer: 1, pos: 112
type: RSZ, layer: 1, pos: 230
type: RSZ, layer: 1, pos: 25
type: RSZ, layer: 1, pos: 250

Time for candidate selection: 0.00 seconds

### Candidate
type: RSZ, layer: 1, pos: 137

### Relational analysis RSZ of RS_RSZ2_RSZ2_RSZ1_RSZ2_RSZ1

#### Relational analysis RSZ result of RS_RSZ2_RSZ2_RSZ1_RSZ2_RSZ1
Status: Status.UNKNOWN
Output dim: 2, lower bound: -0.0019064, upper bound: 0.0018534
time: 1.13 seconds

### Relational analysis RSZ of RS_RSZ2_RSZ2_RSZ1_RSZ2_RSZ2

#### Relational analysis RSZ result of RS_RSZ2_RSZ2_RSZ1_RSZ2_RSZ2
Status: Status.UNKNOWN
Output dim: 2, lower bound: -0.0019108, upper bound: 0.0018527
time: 1.13 seconds

## Summary of splitting (split count: 4)
- Time for RS candidates: 3.41 seconds
RS_RSZ2_RSZ1_RSZ1_RSZ1_RSZ1, status: Status.UNKNOWN, split count: 5, time: 3.41
Output dim: 2, lower bound: -0.0018658, upper bound: 0.0018734
RS_RSZ2_RSZ1_RSZ1_RSZ1_RSZ2, status: Status.UNKNOWN, split count: 5, time: 3.41
Output dim: 2, lower bound: -0.0018731, upper bound: 0.0018658
RS_RSZ2_RSZ1_RSZ1_RSZ2_RSZ1, status: Status.UNKNOWN, split count: 5, time: 3.41
Output dim: 2, lower bound: -0.0018658, upper bound: 0.0018731
RS_RSZ2_RSZ1_RSZ1_RSZ2_RSZ2, status: Status.UNKNOWN, split count: 5, time: 3.41
Output dim: 2, lower bound: -0.0018734, upper bound: 0.0018658
RS_RSZ2_RSZ1_RSZ2_RSZ1_RSZ1, status: Status.VERIFIED, split count: 5, time: 3.41
Output dim: 2, lower bound: -0.0017565, upper bound: 0.0017567
RS_RSZ2_RSZ1_RSZ2_RSZ1_RSZ2, status: Status.VERIFIED, split count: 5, time: 3.41
Output dim: 2, lower bound: -0.0017565, upper bound: 0.0017567
RS_RSZ2_RSZ1_RSZ2_RSZ2_RSZ1, status: Status.VERIFIED, split count: 5, time: 3.41
Output dim: 2, lower bound: -0.0013020, upper bound: 0.0013021
RS_RSZ2_RSZ1_RSZ2_RSZ2_RSZ2, status: Status.VERIFIED, split count: 5, time: 3.41
Output dim: 2, lower bound: -0.0013020, upper bound: 0.0013021
RS_RSZ2_RSZ2_RSZ1_RSZ1_RSZ1, status: Status.UNKNOWN, split count: 5, time: 3.41
Output dim: 2, lower bound: -0.0019037, upper bound: 0.0018324
RS_RSZ2_RSZ2_RSZ1_RSZ1_RSZ2, status: Status.UNKNOWN, split count: 5, time: 3.41
Output dim: 2, lower bound: -0.0019156, upper bound: 0.0018258
RS_RSZ2_RSZ2_RSZ1_RSZ2_RSZ1, status: Status.UNKNOWN, split count: 5, time: 3.41
Output dim: 2, lower bound: -0.0019064, upper bound: 0.0018534
RS_RSZ2_RSZ2_RSZ1_RSZ2_RSZ2, status: Status.UNKNOWN, split count: 5, time: 3.41
Output dim: 2, lower bound: -0.0019108, upper bound: 0.0018527

## BFS RS instance: RS_RSZ2_RSZ1_RSZ1_RSZ1_RSZ1

### Backsubstitution after applying RS history:
0: -0.0041910, -0.0041119, -0.0041910, -0.0041119, -0.0000763, 0.0000765
1: -0.0096089, -0.0066455, -0.0096089, -0.0066455, -0.0028590, 0.0028634
2: 0.9649323, 0.9684886, 0.9649323, 0.9684886, -0.0034309, 0.0034362
3: -0.0123469, 0.0138834, -0.0123469, 0.0138834, -0.0253058, 0.0253447
4: -0.0017489, 0.0002460, -0.0017489, 0.0002460, -0.0019276, 0.0019247
5: 0.0155027, 0.0175190, 0.0155027, 0.0175190, -0.0019482, 0.0019452
6: 0.0033856, 0.0043663, 0.0033856, 0.0043663, -0.0009461, 0.0009476
7: -0.0113762, -0.0045784, -0.0113762, -0.0045784, -0.0065683, 0.0065582
8: 0.0077038, 0.0130968, 0.0077038, 0.0130968, -0.0052110, 0.0052030
9: 0.0115806, 0.0212805, 0.0115806, 0.0212805, -0.0093724, 0.0093580

### Unstable ReLU Count (Linear/Conv2D Layers)
- layer_idx=0, type=LayerType.Linear, total=256, inp1_unstable=17, inp2_unstable=17, delta_unstable=251
- layer_idx=2, type=LayerType.Linear, total=256, inp1_unstable=27, inp2_unstable=27, delta_unstable=256
- layer_idx=4, type=LayerType.Linear, total=256, inp1_unstable=0, inp2_unstable=0, delta_unstable=256
- layer_idx=6, type=LayerType.Linear, total=256, inp1_unstable=0, inp2_unstable=0, delta_unstable=256
- layer_idx=8, type=LayerType.Linear, total=10, inp1_unstable=2, inp2_unstable=2, delta_unstable=10

Time for backsubstitution: 1.14 seconds

### RS candidates at layer 1
type: RSZ, layer: 1, pos: 195
type: RSZ, layer: 1, pos: 250
type: RSZ, layer: 1, pos: 114
type: RSZ, layer: 1, pos: 176
type: RSZ, layer: 1, pos: 29
type: RSZ, layer: 1, pos: 84
type: RSZ, layer: 1, pos: 174
type: RSZ, layer: 1, pos: 230
type: RSZ, layer: 1, pos: 181
type: RSZ, layer: 1, pos: 173
type: RSZ, layer: 1, pos: 240
type: RSZ, layer: 1, pos: 25

Time for candidate selection: 0.00 seconds

### Candidate
type: RSZ, layer: 1, pos: 195

### Relational analysis RSZ of RS_RSZ2_RSZ1_RSZ1_RSZ1_RSZ1_RSZ1

#### Relational analysis RSZ result of RS_RSZ2_RSZ1_RSZ1_RSZ1_RSZ1_RSZ1
Status: Status.VERIFIED
Output dim: 2, lower bound: -0.0017240, upper bound: 0.0017361
time: 1.39 seconds

### Relational analysis RSZ of RS_RSZ2_RSZ1_RSZ1_RSZ1_RSZ1_RSZ2

#### Relational analysis RSZ result of RS_RSZ2_RSZ1_RSZ1_RSZ1_RSZ1_RSZ2
Status: Status.VERIFIED
Output dim: 2, lower bound: -0.0017240, upper bound: 0.0017361
time: 1.39 seconds

## BFS RS instance: RS_RSZ2_RSZ1_RSZ1_RSZ1_RSZ2

### Backsubstitution after applying RS history:
0: -0.0041910, -0.0041119, -0.0041910, -0.0041119, -0.0000764, 0.0000764
1: -0.0096089, -0.0066455, -0.0096089, -0.0066455, -0.0028600, 0.0028624
2: 0.9649323, 0.9684886, 0.9649323, 0.9684886, -0.0034321, 0.0034350
3: -0.0123469, 0.0138834, -0.0123469, 0.0138834, -0.0253143, 0.0253362
4: -0.0017489, 0.0002460, -0.0017489, 0.0002460, -0.0019270, 0.0019253
5: 0.0155027, 0.0175190, 0.0155027, 0.0175190, -0.0019475, 0.0019459
6: 0.0033856, 0.0043663, 0.0033856, 0.0043663, -0.0009465, 0.0009473
7: -0.0113762, -0.0045784, -0.0113762, -0.0045784, -0.0065661, 0.0065604
8: 0.0077038, 0.0130968, 0.0077038, 0.0130968, -0.0052092, 0.0052047
9: 0.0115806, 0.0212805, 0.0115806, 0.0212805, -0.0093693, 0.0093612

### Unstable ReLU Count (Linear/Conv2D Layers)
- layer_idx=0, type=LayerType.Linear, total=256, inp1_unstable=17, inp2_unstable=17, delta_unstable=251
- layer_idx=2, type=LayerType.Linear, total=256, inp1_unstable=27, inp2_unstable=27, delta_unstable=256
- layer_idx=4, type=LayerType.Linear, total=256, inp1_unstable=0, inp2_unstable=0, delta_unstable=256
- layer_idx=6, type=LayerType.Linear, total=256, inp1_unstable=0, inp2_unstable=0, delta_unstable=256
- layer_idx=8, type=LayerType.Linear, total=10, inp1_unstable=2, inp2_unstable=2, delta_unstable=10

Time for backsubstitution: 1.14 seconds

### RS candidates at layer 1
type: RSZ, layer: 1, pos: 181
type: RSZ, layer: 1, pos: 176
type: RSZ, layer: 1, pos: 173
type: RSZ, layer: 1, pos: 240
type: RSZ, layer: 1, pos: 114
type: RSZ, layer: 1, pos: 25
type: RSZ, layer: 1, pos: 29
type: RSZ, layer: 1, pos: 84
type: RSZ, layer: 1, pos: 174
type: RSZ, layer: 1, pos: 250
type: RSZ, layer: 1, pos: 230
type: RSZ, layer: 1, pos: 195

Time for candidate selection: 0.00 seconds

### Candidate
type: RSZ, layer: 1, pos: 181

### Relational analysis RSZ of RS_RSZ2_RSZ1_RSZ1_RSZ1_RSZ2_RSZ1

#### Relational analysis RSZ result of RS_RSZ2_RSZ1_RSZ1_RSZ1_RSZ2_RSZ1
Status: Status.UNKNOWN
Output dim: 2, lower bound: -0.0018723, upper bound: 0.0018650
time: 1.06 seconds

### Relational analysis RSZ of RS_RSZ2_RSZ1_RSZ1_RSZ1_RSZ2_RSZ2

#### Relational analysis RSZ result of RS_RSZ2_RSZ1_RSZ1_RSZ1_RSZ2_RSZ2
Status: Status.UNKNOWN
Output dim: 2, lower bound: -0.0018723, upper bound: 0.0018649
time: 1.04 seconds

## BFS RS instance: RS_RSZ2_RSZ1_RSZ1_RSZ2_RSZ1

### Backsubstitution after applying RS history:
0: -0.0041910, -0.0041119, -0.0041910, -0.0041119, -0.0000764, 0.0000764
1: -0.0096089, -0.0066455, -0.0096089, -0.0066455, -0.0028602, 0.0028620
2: 0.9649323, 0.9684886, 0.9649323, 0.9684886, -0.0034323, 0.0034345
3: -0.0123469, 0.0138834, -0.0123469, 0.0138834, -0.0253162, 0.0253323
4: -0.0017489, 0.0002460, -0.0017489, 0.0002460, -0.0019267, 0.0019254
5: 0.0155027, 0.0175190, 0.0155027, 0.0175190, -0.0019472, 0.0019460
6: 0.0033856, 0.0043663, 0.0033856, 0.0043663, -0.0009465, 0.0009471
7: -0.0113762, -0.0045784, -0.0113762, -0.0045784, -0.0065651, 0.0065609
8: 0.0077038, 0.0130968, 0.0077038, 0.0130968, -0.0052084, 0.0052051
9: 0.0115806, 0.0212805, 0.0115806, 0.0212805, -0.0093679, 0.0093619

### Unstable ReLU Count (Linear/Conv2D Layers)
- layer_idx=0, type=LayerType.Linear, total=256, inp1_unstable=17, inp2_unstable=17, delta_unstable=251
- layer_idx=2, type=LayerType.Linear, total=256, inp1_unstable=27, inp2_unstable=27, delta_unstable=256
- layer_idx=4, type=LayerType.Linear, total=256, inp1_unstable=0, inp2_unstable=0, delta_unstable=256
- layer_idx=6, type=LayerType.Linear, total=256, inp1_unstable=0, inp2_unstable=0, delta_unstable=256
- layer_idx=8, type=LayerType.Linear, total=10, inp1_unstable=2, inp2_unstable=2, delta_unstable=10

Time for backsubstitution: 1.14 seconds

### RS candidates at layer 1
type: RSZ, layer: 1, pos: 25
type: RSZ, layer: 1, pos: 230
type: RSZ, layer: 1, pos: 29
type: RSZ, layer: 1, pos: 176
type: RSZ, layer: 1, pos: 250
type: RSZ, layer: 1, pos: 181
type: RSZ, layer: 1, pos: 240
type: RSZ, layer: 1, pos: 174
type: RSZ, layer: 1, pos: 114
type: RSZ, layer: 1, pos: 173
type: RSZ, layer: 1, pos: 84
type: RSZ, layer: 1, pos: 195

Time for candidate selection: 0.00 seconds

### Candidate
type: RSZ, layer: 1, pos: 25

### Relational analysis RSZ of RS_RSZ2_RSZ1_RSZ1_RSZ2_RSZ1_RSZ1
Optimization infeasible because this subproblem isn't reachable.

### Candidate
type: RSZ, layer: 1, pos: 230

### Relational analysis RSZ of RS_RSZ2_RSZ1_RSZ1_RSZ2_RSZ1_RSZ1

#### Relational analysis RSZ result of RS_RSZ2_RSZ1_RSZ1_RSZ2_RSZ1_RSZ1
Status: Status.VERIFIED
Output dim: 2, lower bound: -0.0012777, upper bound: 0.0012779
time: 1.00 seconds

### Relational analysis RSZ of RS_RSZ2_RSZ1_RSZ1_RSZ2_RSZ1_RSZ2

#### Relational analysis RSZ result of RS_RSZ2_RSZ1_RSZ1_RSZ2_RSZ1_RSZ2
Status: Status.VERIFIED
Output dim: 2, lower bound: -0.0012777, upper bound: 0.0012779
time: 0.99 seconds

## BFS RS instance: RS_RSZ2_RSZ1_RSZ1_RSZ2_RSZ2

### Backsubstitution after applying RS history:
0: -0.0041910, -0.0041119, -0.0041910, -0.0041119, -0.0000764, 0.0000764
1: -0.0096089, -0.0066455, -0.0096089, -0.0066455, -0.0028612, 0.0028610
2: 0.9649323, 0.9684886, 0.9649323, 0.9684886, -0.0034335, 0.0034334
3: -0.0123469, 0.0138834, -0.0123469, 0.0138834, -0.0253250, 0.0253238
4: -0.0017489, 0.0002460, -0.0017489, 0.0002460, -0.0019260, 0.0019261
5: 0.0155027, 0.0175190, 0.0155027, 0.0175190, -0.0019466, 0.0019467
6: 0.0033856, 0.0043663, 0.0033856, 0.0043663, -0.0009469, 0.0009468
7: -0.0113762, -0.0045784, -0.0113762, -0.0045784, -0.0065629, 0.0065632
8: 0.0077038, 0.0130968, 0.0077038, 0.0130968, -0.0052067, 0.0052069
9: 0.0115806, 0.0212805, 0.0115806, 0.0212805, -0.0093647, 0.0093651

### Unstable ReLU Count (Linear/Conv2D Layers)
- layer_idx=0, type=LayerType.Linear, total=256, inp1_unstable=17, inp2_unstable=17, delta_unstable=251
- layer_idx=2, type=LayerType.Linear, total=256, inp1_unstable=27, inp2_unstable=27, delta_unstable=256
- layer_idx=4, type=LayerType.Linear, total=256, inp1_unstable=0, inp2_unstable=0, delta_unstable=256
- layer_idx=6, type=LayerType.Linear, total=256, inp1_unstable=0, inp2_unstable=0, delta_unstable=256
- layer_idx=8, type=LayerType.Linear, total=10, inp1_unstable=2, inp2_unstable=2, delta_unstable=10

Time for backsubstitution: 1.14 seconds

### RS candidates at layer 1
type: RSZ, layer: 1, pos: 29
type: RSZ, layer: 1, pos: 173
type: RSZ, layer: 1, pos: 174
type: RSZ, layer: 1, pos: 25
type: RSZ, layer: 1, pos: 230
type: RSZ, layer: 1, pos: 240
type: RSZ, layer: 1, pos: 250
type: RSZ, layer: 1, pos: 181
type: RSZ, layer: 1, pos: 114
type: RSZ, layer: 1, pos: 84
type: RSZ, layer: 1, pos: 176
type: RSZ, layer: 1, pos: 195

Time for candidate selection: 0.00 seconds

### Candidate
type: RSZ, layer: 1, pos: 29

### Relational analysis RSZ of RS_RSZ2_RSZ1_RSZ1_RSZ2_RSZ2_RSZ1

#### Relational analysis RSZ result of RS_RSZ2_RSZ1_RSZ1_RSZ2_RSZ2_RSZ1
Status: Status.VERIFIED
Output dim: 2, lower bound: -0.0017689, upper bound: 0.0017040
time: 1.40 seconds

### Relational analysis RSZ of RS_RSZ2_RSZ1_RSZ1_RSZ2_RSZ2_RSZ2

#### Relational analysis RSZ result of RS_RSZ2_RSZ1_RSZ1_RSZ2_RSZ2_RSZ2
Status: Status.VERIFIED
Output dim: 2, lower bound: -0.0017093, upper bound: 0.0017591
time: 1.41 seconds

## BFS RS instance: RS_RSZ2_RSZ2_RSZ1_RSZ1_RSZ1

### Backsubstitution after applying RS history:
0: -0.0041910, -0.0041119, -0.0041910, -0.0041119, -0.0000681, 0.0000661
1: -0.0096089, -0.0066455, -0.0096089, -0.0066455, -0.0025517, 0.0024745
2: 0.9649323, 0.9684886, 0.9649323, 0.9684886, -0.0030621, 0.0029695
3: -0.0123469, 0.0138834, -0.0123469, 0.0138834, -0.0225855, 0.0219023
4: -0.0017489, 0.0002460, -0.0017489, 0.0002460, -0.0016658, 0.0017178
5: 0.0155027, 0.0175190, 0.0155027, 0.0175190, -0.0016836, 0.0017361
6: 0.0033856, 0.0043663, 0.0033856, 0.0043663, -0.0008444, 0.0008189
7: -0.0113762, -0.0045784, -0.0113762, -0.0045784, -0.0056762, 0.0058532
8: 0.0077038, 0.0130968, 0.0077038, 0.0130968, -0.0045032, 0.0046437
9: 0.0115806, 0.0212805, 0.0115806, 0.0212805, -0.0080995, 0.0083521

### Unstable ReLU Count (Linear/Conv2D Layers)
- layer_idx=0, type=LayerType.Linear, total=256, inp1_unstable=17, inp2_unstable=17, delta_unstable=251
- layer_idx=2, type=LayerType.Linear, total=256, inp1_unstable=27, inp2_unstable=27, delta_unstable=256
- layer_idx=4, type=LayerType.Linear, total=256, inp1_unstable=0, inp2_unstable=0, delta_unstable=256
- layer_idx=6, type=LayerType.Linear, total=256, inp1_unstable=0, inp2_unstable=0, delta_unstable=256
- layer_idx=8, type=LayerType.Linear, total=10, inp1_unstable=2, inp2_unstable=2, delta_unstable=10

Time for backsubstitution: 1.15 seconds

### RS candidates at layer 1
type: RSZ, layer: 1, pos: 173
type: RSZ, layer: 1, pos: 174
type: RSZ, layer: 1, pos: 250
type: RSZ, layer: 1, pos: 112
type: RSZ, layer: 1, pos: 181
type: RSZ, layer: 1, pos: 114
type: RSZ, layer: 1, pos: 176
type: RSZ, layer: 1, pos: 25
type: RSZ, layer: 1, pos: 240
type: RSZ, layer: 1, pos: 137
type: RSZ, layer: 1, pos: 195
type: RSZ, layer: 1, pos: 230

Time for candidate selection: 0.00 seconds

### Candidate
type: RSZ, layer: 1, pos: 173

### Relational analysis RSZ of RS_RSZ2_RSZ2_RSZ1_RSZ1_RSZ1_RSZ1

#### Relational analysis RSZ result of RS_RSZ2_RSZ2_RSZ1_RSZ1_RSZ1_RSZ1
Status: Status.UNKNOWN
Output dim: 2, lower bound: -0.0018937, upper bound: 0.0018288
time: 1.12 seconds

### Relational analysis RSZ of RS_RSZ2_RSZ2_RSZ1_RSZ1_RSZ1_RSZ2

#### Relational analysis RSZ result of RS_RSZ2_RSZ2_RSZ1_RSZ1_RSZ1_RSZ2
Status: Status.UNKNOWN
Output dim: 2, lower bound: -0.0019005, upper bound: 0.0018171
time: 1.57 seconds

## BFS RS instance: RS_RSZ2_RSZ2_RSZ1_RSZ1_RSZ2

### Backsubstitution after applying RS history:
0: -0.0041910, -0.0041119, -0.0041910, -0.0041119, -0.0000682, 0.0000661
1: -0.0096089, -0.0066455, -0.0096089, -0.0066455, -0.0025526, 0.0024735
2: 0.9649323, 0.9684886, 0.9649323, 0.9684886, -0.0030632, 0.0029683
3: -0.0123469, 0.0138834, -0.0123469, 0.0138834, -0.0225937, 0.0218937
4: -0.0017489, 0.0002460, -0.0017489, 0.0002460, -0.0016651, 0.0017184
5: 0.0155027, 0.0175190, 0.0155027, 0.0175190, -0.0016829, 0.0017367
6: 0.0033856, 0.0043663, 0.0033856, 0.0043663, -0.0008447, 0.0008186
7: -0.0113762, -0.0045784, -0.0113762, -0.0045784, -0.0056740, 0.0058553
8: 0.0077038, 0.0130968, 0.0077038, 0.0130968, -0.0045014, 0.0046454
9: 0.0115806, 0.0212805, 0.0115806, 0.0212805, -0.0080963, 0.0083551

### Unstable ReLU Count (Linear/Conv2D Layers)
- layer_idx=0, type=LayerType.Linear, total=256, inp1_unstable=17, inp2_unstable=17, delta_unstable=251
- layer_idx=2, type=LayerType.Linear, total=256, inp1_unstable=27, inp2_unstable=27, delta_unstable=256
- layer_idx=4, type=LayerType.Linear, total=256, inp1_unstable=0, inp2_unstable=0, delta_unstable=256
- layer_idx=6, type=LayerType.Linear, total=256, inp1_unstable=0, inp2_unstable=0, delta_unstable=256
- layer_idx=8, type=LayerType.Linear, total=10, inp1_unstable=2, inp2_unstable=2, delta_unstable=10

Time for backsubstitution: 1.16 seconds

### RS candidates at layer 1
type: RSZ, layer: 1, pos: 114
type: RSZ, layer: 1, pos: 176
type: RSZ, layer: 1, pos: 173
type: RSZ, layer: 1, pos: 137
type: RSZ, layer: 1, pos: 195
type: RSZ, layer: 1, pos: 112
type: RSZ, layer: 1, pos: 174
type: RSZ, layer: 1, pos: 240
type: RSZ, layer: 1, pos: 250
type: RSZ, layer: 1, pos: 181
type: RSZ, layer: 1, pos: 230
type: RSZ, layer: 1, pos: 25

Time for candidate selection: 0.00 seconds

### Candidate
type: RSZ, layer: 1, pos: 114

### Relational analysis RSZ of RS_RSZ2_RSZ2_RSZ1_RSZ1_RSZ2_RSZ1

#### Relational analysis RSZ result of RS_RSZ2_RSZ2_RSZ1_RSZ1_RSZ2_RSZ1
Status: Status.VERIFIED
Output dim: 2, lower bound: -0.0012539, upper bound: 0.0012121
time: 0.83 seconds

### Relational analysis RSZ of RS_RSZ2_RSZ2_RSZ1_RSZ1_RSZ2_RSZ2

#### Relational analysis RSZ result of RS_RSZ2_RSZ2_RSZ1_RSZ1_RSZ2_RSZ2
Status: Status.VERIFIED
Output dim: 2, lower bound: -0.0012539, upper bound: 0.0012121
time: 0.83 seconds

## BFS RS instance: RS_RSZ2_RSZ2_RSZ1_RSZ2_RSZ1

### Backsubstitution after applying RS history:
0: -0.0041910, -0.0041119, -0.0041910, -0.0041119, -0.0000683, 0.0000662
1: -0.0096089, -0.0066455, -0.0096089, -0.0066455, -0.0025591, 0.0024798
2: 0.9649323, 0.9684886, 0.9649323, 0.9684886, -0.0030711, 0.0029759
3: -0.0123469, 0.0138834, -0.0123469, 0.0138834, -0.0226516, 0.0219495
4: -0.0017489, 0.0002460, -0.0017489, 0.0002460, -0.0016694, 0.0017228
5: 0.0155027, 0.0175190, 0.0155027, 0.0175190, -0.0016872, 0.0017412
6: 0.0033856, 0.0043663, 0.0033856, 0.0043663, -0.0008469, 0.0008207
7: -0.0113762, -0.0045784, -0.0113762, -0.0045784, -0.0056884, 0.0058704
8: 0.0077038, 0.0130968, 0.0077038, 0.0130968, -0.0045129, 0.0046573
9: 0.0115806, 0.0212805, 0.0115806, 0.0212805, -0.0081169, 0.0083766

### Unstable ReLU Count (Linear/Conv2D Layers)
- layer_idx=0, type=LayerType.Linear, total=256, inp1_unstable=17, inp2_unstable=17, delta_unstable=251
- layer_idx=2, type=LayerType.Linear, total=256, inp1_unstable=27, inp2_unstable=27, delta_unstable=256
- layer_idx=4, type=LayerType.Linear, total=256, inp1_unstable=0, inp2_unstable=0, delta_unstable=256
- layer_idx=6, type=LayerType.Linear, total=256, inp1_unstable=0, inp2_unstable=0, delta_unstable=256
- layer_idx=8, type=LayerType.Linear, total=10, inp1_unstable=2, inp2_unstable=2, delta_unstable=10

Time for backsubstitution: 1.19 seconds

### RS candidates at layer 1
type: RSZ, layer: 1, pos: 1
type: RSZ, layer: 1, pos: 176
type: RSZ, layer: 1, pos: 174
type: RSZ, layer: 1, pos: 173
type: RSZ, layer: 1, pos: 240
type: RSZ, layer: 1, pos: 195
type: RSZ, layer: 1, pos: 25
type: RSZ, layer: 1, pos: 114
type: RSZ, layer: 1, pos: 181
type: RSZ, layer: 1, pos: 230
type: RSZ, layer: 1, pos: 112
type: RSZ, layer: 1, pos: 250

Time for candidate selection: 0.00 seconds

### Candidate
type: RSZ, layer: 1, pos: 1

### Relational analysis RSZ of RS_RSZ2_RSZ2_RSZ1_RSZ2_RSZ1_RSZ1

#### Relational analysis RSZ result of RS_RSZ2_RSZ2_RSZ1_RSZ2_RSZ1_RSZ1
Status: Status.UNKNOWN
Output dim: 2, lower bound: -0.0018745, upper bound: 0.0018340
time: 1.18 seconds

### Relational analysis RSZ of RS_RSZ2_RSZ2_RSZ1_RSZ2_RSZ1_RSZ2

#### Relational analysis RSZ result of RS_RSZ2_RSZ2_RSZ1_RSZ2_RSZ1_RSZ2
Status: Status.UNKNOWN
Output dim: 2, lower bound: -0.0018869, upper bound: 0.0018285
time: 1.18 seconds

## BFS RS instance: RS_RSZ2_RSZ2_RSZ1_RSZ2_RSZ2

### Backsubstitution after applying RS history:
0: -0.0041910, -0.0041119, -0.0041910, -0.0041119, -0.0000684, 0.0000662
1: -0.0096089, -0.0066455, -0.0096089, -0.0066455, -0.0025603, 0.0024786
2: 0.9649323, 0.9684886, 0.9649323, 0.9684886, -0.0030724, 0.0029744
3: -0.0123469, 0.0138834, -0.0123469, 0.0138834, -0.0226617, 0.0219388
4: -0.0017489, 0.0002460, -0.0017489, 0.0002460, -0.0016686, 0.0017236
5: 0.0155027, 0.0175190, 0.0155027, 0.0175190, -0.0016864, 0.0017420
6: 0.0033856, 0.0043663, 0.0033856, 0.0043663, -0.0008473, 0.0008203
7: -0.0113762, -0.0045784, -0.0113762, -0.0045784, -0.0056856, 0.0058730
8: 0.0077038, 0.0130968, 0.0077038, 0.0130968, -0.0045107, 0.0046593
9: 0.0115806, 0.0212805, 0.0115806, 0.0212805, -0.0081129, 0.0083803

### Unstable ReLU Count (Linear/Conv2D Layers)
- layer_idx=0, type=LayerType.Linear, total=256, inp1_unstable=17, inp2_unstable=17, delta_unstable=251
- layer_idx=2, type=LayerType.Linear, total=256, inp1_unstable=27, inp2_unstable=27, delta_unstable=256
- layer_idx=4, type=LayerType.Linear, total=256, inp1_unstable=0, inp2_unstable=0, delta_unstable=256
- layer_idx=6, type=LayerType.Linear, total=256, inp1_unstable=0, inp2_unstable=0, delta_unstable=256
- layer_idx=8, type=LayerType.Linear, total=10, inp1_unstable=2, inp2_unstable=2, delta_unstable=10

Time for backsubstitution: 1.17 seconds

### RS candidates at layer 1
type: RSZ, layer: 1, pos: 173
type: RSZ, layer: 1, pos: 250
type: RSZ, layer: 1, pos: 114
type: RSZ, layer: 1, pos: 240
type: RSZ, layer: 1, pos: 112
type: RSZ, layer: 1, pos: 195
type: RSZ, layer: 1, pos: 181
type: RSZ, layer: 1, pos: 176
type: RSZ, layer: 1, pos: 230
type: RSZ, layer: 1, pos: 25
type: RSZ, layer: 1, pos: 174
type: RSZ, layer: 1, pos: 1

Time for candidate selection: 0.00 seconds

### Candidate
type: RSZ, layer: 1, pos: 173

### Relational analysis RSZ of RS_RSZ2_RSZ2_RSZ1_RSZ2_RSZ2_RSZ1

#### Relational analysis RSZ result of RS_RSZ2_RSZ2_RSZ1_RSZ2_RSZ2_RSZ1
Status: Status.UNKNOWN
Output dim: 2, lower bound: -0.0018960, upper bound: 0.0018493
time: 1.11 seconds

### Relational analysis RSZ of RS_RSZ2_RSZ2_RSZ1_RSZ2_RSZ2_RSZ2

#### Relational analysis RSZ result of RS_RSZ2_RSZ2_RSZ1_RSZ2_RSZ2_RSZ2
Status: Status.UNKNOWN
Output dim: 2, lower bound: -0.0019075, upper bound: 0.0018412
time: 1.13 seconds

## Summary of splitting (split count: 5)
- Time for RS candidates: 3.42 seconds
RS_RSZ2_RSZ1_RSZ1_RSZ1_RSZ1_RSZ1, status: Status.VERIFIED, split count: 6, time: 3.42
Output dim: 2, lower bound: -0.0017240, upper bound: 0.0017361
RS_RSZ2_RSZ1_RSZ1_RSZ1_RSZ1_RSZ2, status: Status.VERIFIED, split count: 6, time: 3.42
Output dim: 2, lower bound: -0.0017240, upper bound: 0.0017361
RS_RSZ2_RSZ1_RSZ1_RSZ1_RSZ2_RSZ1, status: Status.UNKNOWN, split count: 6, time: 3.42
Output dim: 2, lower bound: -0.0018723, upper bound: 0.0018650
RS_RSZ2_RSZ1_RSZ1_RSZ1_RSZ2_RSZ2, status: Status.UNKNOWN, split count: 6, time: 3.42
Output dim: 2, lower bound: -0.0018723, upper bound: 0.0018649
RS_RSZ2_RSZ1_RSZ1_RSZ2_RSZ1_RSZ1, status: Status.VERIFIED, split count: 6, time: 3.42
Output dim: 2, lower bound: -0.0012777, upper bound: 0.0012779
RS_RSZ2_RSZ1_RSZ1_RSZ2_RSZ1_RSZ2, status: Status.VERIFIED, split count: 6, time: 3.42
Output dim: 2, lower bound: -0.0012777, upper bound: 0.0012779
RS_RSZ2_RSZ1_RSZ1_RSZ2_RSZ2_RSZ1, status: Status.VERIFIED, split count: 6, time: 3.42
Output dim: 2, lower bound: -0.0017689, upper bound: 0.0017040
RS_RSZ2_RSZ1_RSZ1_RSZ2_RSZ2_RSZ2, status: Status.VERIFIED, split count: 6, time: 3.42
Output dim: 2, lower bound: -0.0017093, upper bound: 0.0017591
RS_RSZ2_RSZ2_RSZ1_RSZ1_RSZ1_RSZ1, status: Status.UNKNOWN, split count: 6, time: 3.42
Output dim: 2, lower bound: -0.0018937, upper bound: 0.0018288
RS_RSZ2_RSZ2_RSZ1_RSZ1_RSZ1_RSZ2, status: Status.UNKNOWN, split count: 6, time: 3.42
Output dim: 2, lower bound: -0.0019005, upper bound: 0.0018171
RS_RSZ2_RSZ2_RSZ1_RSZ1_RSZ2_RSZ1, status: Status.VERIFIED, split count: 6, time: 3.42
Output dim: 2, lower bound: -0.0012539, upper bound: 0.0012121
RS_RSZ2_RSZ2_RSZ1_RSZ1_RSZ2_RSZ2, status: Status.VERIFIED, split count: 6, time: 3.42
Output dim: 2, lower bound: -0.0012539, upper bound: 0.0012121
RS_RSZ2_RSZ2_RSZ1_RSZ2_RSZ1_RSZ1, status: Status.UNKNOWN, split count: 6, time: 3.42
Output dim: 2, lower bound: -0.0018745, upper bound: 0.0018340
RS_RSZ2_RSZ2_RSZ1_RSZ2_RSZ1_RSZ2, status: Status.UNKNOWN, split count: 6, time: 3.42
Output dim: 2, lower bound: -0.0018869, upper bound: 0.0018285
RS_RSZ2_RSZ2_RSZ1_RSZ2_RSZ2_RSZ1, status: Status.UNKNOWN, split count: 6, time: 3.42
Output dim: 2, lower bound: -0.0018960, upper bound: 0.0018493
RS_RSZ2_RSZ2_RSZ1_RSZ2_RSZ2_RSZ2, status: Status.UNKNOWN, split count: 6, time: 3.42
Output dim: 2, lower bound: -0.0019075, upper bound: 0.0018412

## BFS RS instance: RS_RSZ2_RSZ1_RSZ1_RSZ1_RSZ2_RSZ1

### Backsubstitution after applying RS history:
0: -0.0041910, -0.0041119, -0.0041910, -0.0041119, -0.0000769, 0.0000768
1: -0.0096089, -0.0066455, -0.0096089, -0.0066455, -0.0028786, 0.0028771
2: 0.9649323, 0.9684886, 0.9649323, 0.9684886, -0.0034545, 0.0034527
3: -0.0123469, 0.0138834, -0.0123469, 0.0138834, -0.0254794, 0.0254664
4: -0.0017489, 0.0002460, -0.0017489, 0.0002460, -0.0019369, 0.0019379
5: 0.0155027, 0.0175190, 0.0155027, 0.0175190, -0.0019575, 0.0019585
6: 0.0033856, 0.0043663, 0.0033856, 0.0043663, -0.0009526, 0.0009521
7: -0.0113762, -0.0045784, -0.0113762, -0.0045784, -0.0065998, 0.0066032
8: 0.0077038, 0.0130968, 0.0077038, 0.0130968, -0.0052360, 0.0052387
9: 0.0115806, 0.0212805, 0.0115806, 0.0212805, -0.0094175, 0.0094223

### Unstable ReLU Count (Linear/Conv2D Layers)
- layer_idx=0, type=LayerType.Linear, total=256, inp1_unstable=17, inp2_unstable=17, delta_unstable=250
- layer_idx=2, type=LayerType.Linear, total=256, inp1_unstable=27, inp2_unstable=27, delta_unstable=256
- layer_idx=4, type=LayerType.Linear, total=256, inp1_unstable=0, inp2_unstable=0, delta_unstable=256
- layer_idx=6, type=LayerType.Linear, total=256, inp1_unstable=0, inp2_unstable=0, delta_unstable=256
- layer_idx=8, type=LayerType.Linear, total=10, inp1_unstable=2, inp2_unstable=2, delta_unstable=10

Time for backsubstitution: 1.14 seconds

### RS candidates at layer 1
type: RSZ, layer: 1, pos: 195
type: RSZ, layer: 1, pos: 25
type: RSZ, layer: 1, pos: 173
type: RSZ, layer: 1, pos: 230
type: RSZ, layer: 1, pos: 84
type: RSZ, layer: 1, pos: 176
type: RSZ, layer: 1, pos: 240
type: RSZ, layer: 1, pos: 174
type: RSZ, layer: 1, pos: 29
type: RSZ, layer: 1, pos: 114
type: RSZ, layer: 1, pos: 250

Time for candidate selection: 0.00 seconds

### Candidate
type: RSZ, layer: 1, pos: 195

### Relational analysis RSZ of RS_RSZ2_RSZ1_RSZ1_RSZ1_RSZ2_RSZ1_RSZ1

#### Relational analysis RSZ result of RS_RSZ2_RSZ1_RSZ1_RSZ1_RSZ2_RSZ1_RSZ1
Status: Status.VERIFIED
Output dim: 2, lower bound: -0.0017343, upper bound: 0.0017237
time: 1.40 seconds

### Relational analysis RSZ of RS_RSZ2_RSZ1_RSZ1_RSZ1_RSZ2_RSZ1_RSZ2

#### Relational analysis RSZ result of RS_RSZ2_RSZ1_RSZ1_RSZ1_RSZ2_RSZ1_RSZ2
Status: Status.VERIFIED
Output dim: 2, lower bound: -0.0017343, upper bound: 0.0017237
time: 1.41 seconds

## BFS RS instance: RS_RSZ2_RSZ1_RSZ1_RSZ1_RSZ2_RSZ2

### Backsubstitution after applying RS history:
0: -0.0041910, -0.0041119, -0.0041910, -0.0041119, -0.0000768, 0.0000769
1: -0.0096089, -0.0066455, -0.0096089, -0.0066455, -0.0028750, 0.0028795
2: 0.9649323, 0.9684886, 0.9649323, 0.9684886, -0.0034502, 0.0034555
3: -0.0123469, 0.0138834, -0.0123469, 0.0138834, -0.0254478, 0.0254872
4: -0.0017489, 0.0002460, -0.0017489, 0.0002460, -0.0019384, 0.0019355
5: 0.0155027, 0.0175190, 0.0155027, 0.0175190, -0.0019591, 0.0019561
6: 0.0033856, 0.0043663, 0.0033856, 0.0043663, -0.0009515, 0.0009529
7: -0.0113762, -0.0045784, -0.0113762, -0.0045784, -0.0066052, 0.0065950
8: 0.0077038, 0.0130968, 0.0077038, 0.0130968, -0.0052403, 0.0052322
9: 0.0115806, 0.0212805, 0.0115806, 0.0212805, -0.0094251, 0.0094106

### Unstable ReLU Count (Linear/Conv2D Layers)
- layer_idx=0, type=LayerType.Linear, total=256, inp1_unstable=17, inp2_unstable=17, delta_unstable=250
- layer_idx=2, type=LayerType.Linear, total=256, inp1_unstable=27, inp2_unstable=27, delta_unstable=256
- layer_idx=4, type=LayerType.Linear, total=256, inp1_unstable=0, inp2_unstable=0, delta_unstable=256
- layer_idx=6, type=LayerType.Linear, total=256, inp1_unstable=0, inp2_unstable=0, delta_unstable=256
- layer_idx=8, type=LayerType.Linear, total=10, inp1_unstable=2, inp2_unstable=2, delta_unstable=10

Time for backsubstitution: 1.15 seconds

### RS candidates at layer 1
type: RSZ, layer: 1, pos: 250
type: RSZ, layer: 1, pos: 174
type: RSZ, layer: 1, pos: 173
type: RSZ, layer: 1, pos: 25
type: RSZ, layer: 1, pos: 230
type: RSZ, layer: 1, pos: 176
type: RSZ, layer: 1, pos: 114
type: RSZ, layer: 1, pos: 195
type: RSZ, layer: 1, pos: 29
type: RSZ, layer: 1, pos: 84
type: RSZ, layer: 1, pos: 240

Time for candidate selection: 0.00 seconds

### Candidate
type: RSZ, layer: 1, pos: 250

### Relational analysis RSZ of RS_RSZ2_RSZ1_RSZ1_RSZ1_RSZ2_RSZ2_RSZ1
Optimization infeasible because this subproblem isn't reachable.

### Candidate
type: RSZ, layer: 1, pos: 174

### Relational analysis RSZ of RS_RSZ2_RSZ1_RSZ1_RSZ1_RSZ2_RSZ2_RSZ1
Optimization infeasible because this subproblem isn't reachable.

### Candidate
type: RSZ, layer: 1, pos: 173

### Relational analysis RSZ of RS_RSZ2_RSZ1_RSZ1_RSZ1_RSZ2_RSZ2_RSZ1

#### Relational analysis RSZ result of RS_RSZ2_RSZ1_RSZ1_RSZ1_RSZ2_RSZ2_RSZ1
Status: Status.UNKNOWN
Output dim: 2, lower bound: -0.0018603, upper bound: 0.0018607
time: 1.07 seconds

### Relational analysis RSZ of RS_RSZ2_RSZ1_RSZ1_RSZ1_RSZ2_RSZ2_RSZ2

#### Relational analysis RSZ result of RS_RSZ2_RSZ1_RSZ1_RSZ1_RSZ2_RSZ2_RSZ2
Status: Status.UNKNOWN
Output dim: 2, lower bound: -0.0018681, upper bound: 0.0018547
time: 1.08 seconds

## BFS RS instance: RS_RSZ2_RSZ2_RSZ1_RSZ1_RSZ1_RSZ1

### Backsubstitution after applying RS history:
0: -0.0041910, -0.0041119, -0.0041910, -0.0041119, -0.0000676, 0.0000657
1: -0.0096089, -0.0066455, -0.0096089, -0.0066455, -0.0025327, 0.0024600
2: 0.9649323, 0.9684886, 0.9649323, 0.9684886, -0.0030394, 0.0029521
3: -0.0123469, 0.0138834, -0.0123469, 0.0138834, -0.0224179, 0.0217738
4: -0.0017489, 0.0002460, -0.0017489, 0.0002460, -0.0016560, 0.0017050
5: 0.0155027, 0.0175190, 0.0155027, 0.0175190, -0.0016737, 0.0017232
6: 0.0033856, 0.0043663, 0.0033856, 0.0043663, -0.0008382, 0.0008141
7: -0.0113762, -0.0045784, -0.0113762, -0.0045784, -0.0056429, 0.0058098
8: 0.0077038, 0.0130968, 0.0077038, 0.0130968, -0.0044768, 0.0046092
9: 0.0115806, 0.0212805, 0.0115806, 0.0212805, -0.0080519, 0.0082901

### Unstable ReLU Count (Linear/Conv2D Layers)
- layer_idx=0, type=LayerType.Linear, total=256, inp1_unstable=17, inp2_unstable=17, delta_unstable=250
- layer_idx=2, type=LayerType.Linear, total=256, inp1_unstable=27, inp2_unstable=27, delta_unstable=256
- layer_idx=4, type=LayerType.Linear, total=256, inp1_unstable=0, inp2_unstable=0, delta_unstable=256
- layer_idx=6, type=LayerType.Linear, total=256, inp1_unstable=0, inp2_unstable=0, delta_unstable=256
- layer_idx=8, type=LayerType.Linear, total=10, inp1_unstable=2, inp2_unstable=2, delta_unstable=10

Time for backsubstitution: 1.16 seconds

### RS candidates at layer 1
type: RSZ, layer: 1, pos: 25
type: RSZ, layer: 1, pos: 181
type: RSZ, layer: 1, pos: 250
type: RSZ, layer: 1, pos: 195
type: RSZ, layer: 1, pos: 230
type: RSZ, layer: 1, pos: 114
type: RSZ, layer: 1, pos: 176
type: RSZ, layer: 1, pos: 112
type: RSZ, layer: 1, pos: 137
type: RSZ, layer: 1, pos: 240
type: RSZ, layer: 1, pos: 174

Time for candidate selection: 0.00 seconds

### Candidate
type: RSZ, layer: 1, pos: 25

### Relational analysis RSZ of RS_RSZ2_RSZ2_RSZ1_RSZ1_RSZ1_RSZ1_RSZ1

#### Relational analysis RSZ result of RS_RSZ2_RSZ2_RSZ1_RSZ1_RSZ1_RSZ1_RSZ1
Status: Status.VERIFIED
Output dim: 2, lower bound: -0.0016416, upper bound: 0.0015902
time: 1.39 seconds

### Relational analysis RSZ of RS_RSZ2_RSZ2_RSZ1_RSZ1_RSZ1_RSZ1_RSZ2

#### Relational analysis RSZ result of RS_RSZ2_RSZ2_RSZ1_RSZ1_RSZ1_RSZ1_RSZ2
Status: Status.VERIFIED
Output dim: 2, lower bound: -0.0016416, upper bound: 0.0015902
time: 1.39 seconds

## BFS RS instance: RS_RSZ2_RSZ2_RSZ1_RSZ1_RSZ1_RSZ2

### Backsubstitution after applying RS history:
0: -0.0041910, -0.0041119, -0.0041910, -0.0041119, -0.0000677, 0.0000656
1: -0.0096089, -0.0066455, -0.0096089, -0.0066455, -0.0025367, 0.0024554
2: 0.9649323, 0.9684886, 0.9649323, 0.9684886, -0.0030442, 0.0029466
3: -0.0123469, 0.0138834, -0.0123469, 0.0138834, -0.0224536, 0.0217333
4: -0.0017489, 0.0002460, -0.0017489, 0.0002460, -0.0016529, 0.0017077
5: 0.0155027, 0.0175190, 0.0155027, 0.0175190, -0.0016706, 0.0017260
6: 0.0033856, 0.0043663, 0.0033856, 0.0043663, -0.0008395, 0.0008126
7: -0.0113762, -0.0045784, -0.0113762, -0.0045784, -0.0056324, 0.0058190
8: 0.0077038, 0.0130968, 0.0077038, 0.0130968, -0.0044685, 0.0046165
9: 0.0115806, 0.0212805, 0.0115806, 0.0212805, -0.0080370, 0.0083033

### Unstable ReLU Count (Linear/Conv2D Layers)
- layer_idx=0, type=LayerType.Linear, total=256, inp1_unstable=17, inp2_unstable=17, delta_unstable=250
- layer_idx=2, type=LayerType.Linear, total=256, inp1_unstable=27, inp2_unstable=27, delta_unstable=256
- layer_idx=4, type=LayerType.Linear, total=256, inp1_unstable=0, inp2_unstable=0, delta_unstable=256
- layer_idx=6, type=LayerType.Linear, total=256, inp1_unstable=0, inp2_unstable=0, delta_unstable=256
- layer_idx=8, type=LayerType.Linear, total=10, inp1_unstable=2, inp2_unstable=2, delta_unstable=10

Time for backsubstitution: 1.18 seconds

### RS candidates at layer 1
type: RSZ, layer: 1, pos: 230
type: RSZ, layer: 1, pos: 181
type: RSZ, layer: 1, pos: 250
type: RSZ, layer: 1, pos: 137
type: RSZ, layer: 1, pos: 176
type: RSZ, layer: 1, pos: 240
type: RSZ, layer: 1, pos: 195
type: RSZ, layer: 1, pos: 174
type: RSZ, layer: 1, pos: 25
type: RSZ, layer: 1, pos: 112
type: RSZ, layer: 1, pos: 114

Time for candidate selection: 0.00 seconds

### Candidate
type: RSZ, layer: 1, pos: 230

### Relational analysis RSZ of RS_RSZ2_RSZ2_RSZ1_RSZ1_RSZ1_RSZ2_RSZ1

#### Relational analysis RSZ result of RS_RSZ2_RSZ2_RSZ1_RSZ1_RSZ1_RSZ2_RSZ1
Status: Status.VERIFIED
Output dim: 2, lower bound: -0.0016805, upper bound: 0.0016139
time: 1.04 seconds

### Relational analysis RSZ of RS_RSZ2_RSZ2_RSZ1_RSZ1_RSZ1_RSZ2_RSZ2

#### Relational analysis RSZ result of RS_RSZ2_RSZ2_RSZ1_RSZ1_RSZ1_RSZ2_RSZ2
Status: Status.VERIFIED
Output dim: 2, lower bound: -0.0016805, upper bound: 0.0016139
time: 1.03 seconds

## BFS RS instance: RS_RSZ2_RSZ2_RSZ1_RSZ2_RSZ1_RSZ1

### Backsubstitution after applying RS history:
0: -0.0041910, -0.0041119, -0.0041910, -0.0041119, -0.0000682, 0.0000661
1: -0.0096089, -0.0066455, -0.0096089, -0.0066455, -0.0025546, 0.0024764
2: 0.9649323, 0.9684886, 0.9649323, 0.9684886, -0.0030656, 0.0029717
3: -0.0123469, 0.0138834, -0.0123469, 0.0138834, -0.0226115, 0.0219191
4: -0.0017489, 0.0002460, -0.0017489, 0.0002460, -0.0016671, 0.0017197
5: 0.0155027, 0.0175190, 0.0155027, 0.0175190, -0.0016849, 0.0017381
6: 0.0033856, 0.0043663, 0.0033856, 0.0043663, -0.0008454, 0.0008195
7: -0.0113762, -0.0045784, -0.0113762, -0.0045784, -0.0056805, 0.0058600
8: 0.0077038, 0.0130968, 0.0077038, 0.0130968, -0.0045067, 0.0046490
9: 0.0115806, 0.0212805, 0.0115806, 0.0212805, -0.0081057, 0.0083617

### Unstable ReLU Count (Linear/Conv2D Layers)
- layer_idx=0, type=LayerType.Linear, total=256, inp1_unstable=17, inp2_unstable=17, delta_unstable=250
- layer_idx=2, type=LayerType.Linear, total=256, inp1_unstable=27, inp2_unstable=27, delta_unstable=256
- layer_idx=4, type=LayerType.Linear, total=256, inp1_unstable=0, inp2_unstable=0, delta_unstable=256
- layer_idx=6, type=LayerType.Linear, total=256, inp1_unstable=0, inp2_unstable=0, delta_unstable=256
- layer_idx=8, type=LayerType.Linear, total=10, inp1_unstable=2, inp2_unstable=2, delta_unstable=10

Time for backsubstitution: 1.16 seconds

### RS candidates at layer 1
type: RSZ, layer: 1, pos: 112
type: RSZ, layer: 1, pos: 25
type: RSZ, layer: 1, pos: 174
type: RSZ, layer: 1, pos: 176
type: RSZ, layer: 1, pos: 240
type: RSZ, layer: 1, pos: 195
type: RSZ, layer: 1, pos: 114
type: RSZ, layer: 1, pos: 250
type: RSZ, layer: 1, pos: 181
type: RSZ, layer: 1, pos: 230
type: RSZ, layer: 1, pos: 173

Time for candidate selection: 0.00 seconds

### Candidate
type: RSZ, layer: 1, pos: 112

### Relational analysis RSZ of RS_RSZ2_RSZ2_RSZ1_RSZ2_RSZ1_RSZ1_RSZ1

#### Relational analysis RSZ result of RS_RSZ2_RSZ2_RSZ1_RSZ2_RSZ1_RSZ1_RSZ1
Status: Status.VERIFIED
Output dim: 2, lower bound: -0.0016957, upper bound: 0.0016651
time: 1.08 seconds

### Relational analysis RSZ of RS_RSZ2_RSZ2_RSZ1_RSZ2_RSZ1_RSZ1_RSZ2

#### Relational analysis RSZ result of RS_RSZ2_RSZ2_RSZ1_RSZ2_RSZ1_RSZ1_RSZ2
Status: Status.VERIFIED
Output dim: 2, lower bound: -0.0016957, upper bound: 0.0016651
time: 1.08 seconds

## BFS RS instance: RS_RSZ2_RSZ2_RSZ1_RSZ2_RSZ1_RSZ2

### Backsubstitution after applying RS history:
0: -0.0041910, -0.0041119, -0.0041910, -0.0041119, -0.0000682, 0.0000661
1: -0.0096089, -0.0066455, -0.0096089, -0.0066455, -0.0025557, 0.0024754
2: 0.9649323, 0.9684886, 0.9649323, 0.9684886, -0.0030669, 0.0029706
3: -0.0123469, 0.0138834, -0.0123469, 0.0138834, -0.0226212, 0.0219107
4: -0.0017489, 0.0002460, -0.0017489, 0.0002460, -0.0016664, 0.0017205
5: 0.0155027, 0.0175190, 0.0155027, 0.0175190, -0.0016842, 0.0017388
6: 0.0033856, 0.0043663, 0.0033856, 0.0043663, -0.0008458, 0.0008192
7: -0.0113762, -0.0045784, -0.0113762, -0.0045784, -0.0056784, 0.0058625
8: 0.0077038, 0.0130968, 0.0077038, 0.0130968, -0.0045049, 0.0046510
9: 0.0115806, 0.0212805, 0.0115806, 0.0212805, -0.0081026, 0.0083653

### Unstable ReLU Count (Linear/Conv2D Layers)
- layer_idx=0, type=LayerType.Linear, total=256, inp1_unstable=17, inp2_unstable=17, delta_unstable=250
- layer_idx=2, type=LayerType.Linear, total=256, inp1_unstable=27, inp2_unstable=27, delta_unstable=256
- layer_idx=4, type=LayerType.Linear, total=256, inp1_unstable=0, inp2_unstable=0, delta_unstable=256
- layer_idx=6, type=LayerType.Linear, total=256, inp1_unstable=0, inp2_unstable=0, delta_unstable=256
- layer_idx=8, type=LayerType.Linear, total=10, inp1_unstable=2, inp2_unstable=2, delta_unstable=10

Time for backsubstitution: 1.16 seconds

### RS candidates at layer 1
type: RSZ, layer: 1, pos: 181
type: RSZ, layer: 1, pos: 230
type: RSZ, layer: 1, pos: 176
type: RSZ, layer: 1, pos: 195
type: RSZ, layer: 1, pos: 173
type: RSZ, layer: 1, pos: 240
type: RSZ, layer: 1, pos: 174
type: RSZ, layer: 1, pos: 114
type: RSZ, layer: 1, pos: 25
type: RSZ, layer: 1, pos: 112
type: RSZ, layer: 1, pos: 250

Time for candidate selection: 0.00 seconds

### Candidate
type: RSZ, layer: 1, pos: 181

### Relational analysis RSZ of RS_RSZ2_RSZ2_RSZ1_RSZ2_RSZ1_RSZ2_RSZ1

#### Relational analysis RSZ result of RS_RSZ2_RSZ2_RSZ1_RSZ2_RSZ1_RSZ2_RSZ1
Status: Status.UNKNOWN
Output dim: 2, lower bound: -0.0018850, upper bound: 0.0018277
time: 1.17 seconds

### Relational analysis RSZ of RS_RSZ2_RSZ2_RSZ1_RSZ2_RSZ1_RSZ2_RSZ2

#### Relational analysis RSZ result of RS_RSZ2_RSZ2_RSZ1_RSZ2_RSZ1_RSZ2_RSZ2
Status: Status.UNKNOWN
Output dim: 2, lower bound: -0.0018861, upper bound: 0.0018265
time: 1.20 seconds

## BFS RS instance: RS_RSZ2_RSZ2_RSZ1_RSZ2_RSZ2_RSZ1

### Backsubstitution after applying RS history:
0: -0.0041910, -0.0041119, -0.0041910, -0.0041119, -0.0000678, 0.0000658
1: -0.0096089, -0.0066455, -0.0096089, -0.0066455, -0.0025407, 0.0024643
2: 0.9649323, 0.9684886, 0.9649323, 0.9684886, -0.0030490, 0.0029572
3: -0.0123469, 0.0138834, -0.0123469, 0.0138834, -0.0224889, 0.0218121
4: -0.0017489, 0.0002460, -0.0017489, 0.0002460, -0.0016589, 0.0017104
5: 0.0155027, 0.0175190, 0.0155027, 0.0175190, -0.0016766, 0.0017287
6: 0.0033856, 0.0043663, 0.0033856, 0.0043663, -0.0008408, 0.0008155
7: -0.0113762, -0.0045784, -0.0113762, -0.0045784, -0.0056528, 0.0058282
8: 0.0077038, 0.0130968, 0.0077038, 0.0130968, -0.0044847, 0.0046238
9: 0.0115806, 0.0212805, 0.0115806, 0.0212805, -0.0080661, 0.0083164

### Unstable ReLU Count (Linear/Conv2D Layers)
- layer_idx=0, type=LayerType.Linear, total=256, inp1_unstable=17, inp2_unstable=17, delta_unstable=250
- layer_idx=2, type=LayerType.Linear, total=256, inp1_unstable=27, inp2_unstable=27, delta_unstable=256
- layer_idx=4, type=LayerType.Linear, total=256, inp1_unstable=0, inp2_unstable=0, delta_unstable=256
- layer_idx=6, type=LayerType.Linear, total=256, inp1_unstable=0, inp2_unstable=0, delta_unstable=256
- layer_idx=8, type=LayerType.Linear, total=10, inp1_unstable=2, inp2_unstable=2, delta_unstable=10

Time for backsubstitution: 1.28 seconds

### RS candidates at layer 1
type: RSZ, layer: 1, pos: 112
type: RSZ, layer: 1, pos: 174
type: RSZ, layer: 1, pos: 114
type: RSZ, layer: 1, pos: 195
type: RSZ, layer: 1, pos: 1
type: RSZ, layer: 1, pos: 250
type: RSZ, layer: 1, pos: 181
type: RSZ, layer: 1, pos: 25
type: RSZ, layer: 1, pos: 176
type: RSZ, layer: 1, pos: 230
type: RSZ, layer: 1, pos: 240

Time for candidate selection: 0.00 seconds

### Candidate
type: RSZ, layer: 1, pos: 112

### Relational analysis RSZ of RS_RSZ2_RSZ2_RSZ1_RSZ2_RSZ2_RSZ1_RSZ1

#### Relational analysis RSZ result of RS_RSZ2_RSZ2_RSZ1_RSZ2_RSZ2_RSZ1_RSZ1
Status: Status.VERIFIED
Output dim: 2, lower bound: -0.0017135, upper bound: 0.0016752
time: 1.09 seconds

### Relational analysis RSZ of RS_RSZ2_RSZ2_RSZ1_RSZ2_RSZ2_RSZ1_RSZ2

#### Relational analysis RSZ result of RS_RSZ2_RSZ2_RSZ1_RSZ2_RSZ2_RSZ1_RSZ2
Status: Status.VERIFIED
Output dim: 2, lower bound: -0.0017135, upper bound: 0.0016752
time: 1.11 seconds

## BFS RS instance: RS_RSZ2_RSZ2_RSZ1_RSZ2_RSZ2_RSZ2

### Backsubstitution after applying RS history:
0: -0.0041910, -0.0041119, -0.0041910, -0.0041119, -0.0000680, 0.0000657
1: -0.0096089, -0.0066455, -0.0096089, -0.0066455, -0.0025457, 0.0024598
2: 0.9649323, 0.9684886, 0.9649323, 0.9684886, -0.0030549, 0.0029519
3: -0.0123469, 0.0138834, -0.0123469, 0.0138834, -0.0225326, 0.0217723
4: -0.0017489, 0.0002460, -0.0017489, 0.0002460, -0.0016559, 0.0017137
5: 0.0155027, 0.0175190, 0.0155027, 0.0175190, -0.0016736, 0.0017320
6: 0.0033856, 0.0043663, 0.0033856, 0.0043663, -0.0008425, 0.0008140
7: -0.0113762, -0.0045784, -0.0113762, -0.0045784, -0.0056425, 0.0058395
8: 0.0077038, 0.0130968, 0.0077038, 0.0130968, -0.0044765, 0.0046328
9: 0.0115806, 0.0212805, 0.0115806, 0.0212805, -0.0080514, 0.0083325

### Unstable ReLU Count (Linear/Conv2D Layers)
- layer_idx=0, type=LayerType.Linear, total=256, inp1_unstable=17, inp2_unstable=17, delta_unstable=250
- layer_idx=2, type=LayerType.Linear, total=256, inp1_unstable=27, inp2_unstable=27, delta_unstable=256
- layer_idx=4, type=LayerType.Linear, total=256, inp1_unstable=0, inp2_unstable=0, delta_unstable=256
- layer_idx=6, type=LayerType.Linear, total=256, inp1_unstable=0, inp2_unstable=0, delta_unstable=256
- layer_idx=8, type=LayerType.Linear, total=10, inp1_unstable=2, inp2_unstable=2, delta_unstable=10

Time for backsubstitution: 1.28 seconds

### RS candidates at layer 1
type: RSZ, layer: 1, pos: 1
type: RSZ, layer: 1, pos: 181
type: RSZ, layer: 1, pos: 174
type: RSZ, layer: 1, pos: 230
type: RSZ, layer: 1, pos: 176
type: RSZ, layer: 1, pos: 25
type: RSZ, layer: 1, pos: 112
type: RSZ, layer: 1, pos: 250
type: RSZ, layer: 1, pos: 240
type: RSZ, layer: 1, pos: 114
type: RSZ, layer: 1, pos: 195

Time for candidate selection: 0.00 seconds

### Candidate
type: RSZ, layer: 1, pos: 1

### Relational analysis RSZ of RS_RSZ2_RSZ2_RSZ1_RSZ2_RSZ2_RSZ2_RSZ1

#### Relational analysis RSZ result of RS_RSZ2_RSZ2_RSZ1_RSZ2_RSZ2_RSZ2_RSZ1
Status: Status.UNKNOWN
Output dim: 2, lower bound: -0.0018757, upper bound: 0.0018215
time: 1.11 seconds

### Relational analysis RSZ of RS_RSZ2_RSZ2_RSZ1_RSZ2_RSZ2_RSZ2_RSZ2

#### Relational analysis RSZ result of RS_RSZ2_RSZ2_RSZ1_RSZ2_RSZ2_RSZ2_RSZ2
Status: Status.UNKNOWN
Output dim: 2, lower bound: -0.0018880, upper bound: 0.0018152
time: 1.10 seconds

## Summary of splitting (split count: 6)
- Time for RS candidates: 3.49 seconds
RS_RSZ2_RSZ1_RSZ1_RSZ1_RSZ2_RSZ1_RSZ1, status: Status.VERIFIED, split count: 7, time: 3.49
Output dim: 2, lower bound: -0.0017343, upper bound: 0.0017237
RS_RSZ2_RSZ1_RSZ1_RSZ1_RSZ2_RSZ1_RSZ2, status: Status.VERIFIED, split count: 7, time: 3.49
Output dim: 2, lower bound: -0.0017343, upper bound: 0.0017237
RS_RSZ2_RSZ1_RSZ1_RSZ1_RSZ2_RSZ2_RSZ1, status: Status.UNKNOWN, split count: 7, time: 3.49
Output dim: 2, lower bound: -0.0018603, upper bound: 0.0018607
RS_RSZ2_RSZ1_RSZ1_RSZ1_RSZ2_RSZ2_RSZ2, status: Status.UNKNOWN, split count: 7, time: 3.49
Output dim: 2, lower bound: -0.0018681, upper bound: 0.0018547
RS_RSZ2_RSZ2_RSZ1_RSZ1_RSZ1_RSZ1_RSZ1, status: Status.VERIFIED, split count: 7, time: 3.49
Output dim: 2, lower bound: -0.0016416, upper bound: 0.0015902
RS_RSZ2_RSZ2_RSZ1_RSZ1_RSZ1_RSZ1_RSZ2, status: Status.VERIFIED, split count: 7, time: 3.49
Output dim: 2, lower bound: -0.0016416, upper bound: 0.0015902
RS_RSZ2_RSZ2_RSZ1_RSZ1_RSZ1_RSZ2_RSZ1, status: Status.VERIFIED, split count: 7, time: 3.49
Output dim: 2, lower bound: -0.0016805, upper bound: 0.0016139
RS_RSZ2_RSZ2_RSZ1_RSZ1_RSZ1_RSZ2_RSZ2, status: Status.VERIFIED, split count: 7, time: 3.49
Output dim: 2, lower bound: -0.0016805, upper bound: 0.0016139
RS_RSZ2_RSZ2_RSZ1_RSZ2_RSZ1_RSZ1_RSZ1, status: Status.VERIFIED, split count: 7, time: 3.49
Output dim: 2, lower bound: -0.0016957, upper bound: 0.0016651
RS_RSZ2_RSZ2_RSZ1_RSZ2_RSZ1_RSZ1_RSZ2, status: Status.VERIFIED, split count: 7, time: 3.49
Output dim: 2, lower bound: -0.0016957, upper bound: 0.0016651
RS_RSZ2_RSZ2_RSZ1_RSZ2_RSZ1_RSZ2_RSZ1, status: Status.UNKNOWN, split count: 7, time: 3.49
Output dim: 2, lower bound: -0.0018850, upper bound: 0.0018277
RS_RSZ2_RSZ2_RSZ1_RSZ2_RSZ1_RSZ2_RSZ2, status: Status.UNKNOWN, split count: 7, time: 3.49
Output dim: 2, lower bound: -0.0018861, upper bound: 0.0018265
RS_RSZ2_RSZ2_RSZ1_RSZ2_RSZ2_RSZ1_RSZ1, status: Status.VERIFIED, split count: 7, time: 3.49
Output dim: 2, lower bound: -0.0017135, upper bound: 0.0016752
RS_RSZ2_RSZ2_RSZ1_RSZ2_RSZ2_RSZ1_RSZ2, status: Status.VERIFIED, split count: 7, time: 3.49
Output dim: 2, lower bound: -0.0017135, upper bound: 0.0016752
RS_RSZ2_RSZ2_RSZ1_RSZ2_RSZ2_RSZ2_RSZ1, status: Status.UNKNOWN, split count: 7, time: 3.49
Output dim: 2, lower bound: -0.0018757, upper bound: 0.0018215
RS_RSZ2_RSZ2_RSZ1_RSZ2_RSZ2_RSZ2_RSZ2, status: Status.UNKNOWN, split count: 7, time: 3.49
Output dim: 2, lower bound: -0.0018880, upper bound: 0.0018152

## BFS RS instance: RS_RSZ2_RSZ1_RSZ1_RSZ1_RSZ2_RSZ2_RSZ1

### Backsubstitution after applying RS history:
0: -0.0041910, -0.0041119, -0.0041910, -0.0041119, -0.0000762, 0.0000764
1: -0.0096089, -0.0066455, -0.0096089, -0.0066455, -0.0028528, 0.0028619
2: 0.9649323, 0.9684886, 0.9649323, 0.9684886, -0.0034235, 0.0034344
3: -0.0123469, 0.0138834, -0.0123469, 0.0138834, -0.0252514, 0.0253317
4: -0.0017489, 0.0002460, -0.0017489, 0.0002460, -0.0019266, 0.0019205
5: 0.0155027, 0.0175190, 0.0155027, 0.0175190, -0.0019472, 0.0019410
6: 0.0033856, 0.0043663, 0.0033856, 0.0043663, -0.0009441, 0.0009471
7: -0.0113762, -0.0045784, -0.0113762, -0.0045784, -0.0065649, 0.0065441
8: 0.0077038, 0.0130968, 0.0077038, 0.0130968, -0.0052083, 0.0051918
9: 0.0115806, 0.0212805, 0.0115806, 0.0212805, -0.0093676, 0.0093380

### Unstable ReLU Count (Linear/Conv2D Layers)
- layer_idx=0, type=LayerType.Linear, total=256, inp1_unstable=17, inp2_unstable=17, delta_unstable=249
- layer_idx=2, type=LayerType.Linear, total=256, inp1_unstable=27, inp2_unstable=27, delta_unstable=256
- layer_idx=4, type=LayerType.Linear, total=256, inp1_unstable=0, inp2_unstable=0, delta_unstable=256
- layer_idx=6, type=LayerType.Linear, total=256, inp1_unstable=0, inp2_unstable=0, delta_unstable=256
- layer_idx=8, type=LayerType.Linear, total=10, inp1_unstable=2, inp2_unstable=2, delta_unstable=10

Time for backsubstitution: 1.25 seconds

### RS candidates at layer 1
type: RSZ, layer: 1, pos: 195
type: RSZ, layer: 1, pos: 114
type: RSZ, layer: 1, pos: 25
type: RSZ, layer: 1, pos: 29
type: RSZ, layer: 1, pos: 176
type: RSZ, layer: 1, pos: 84
type: RSZ, layer: 1, pos: 250
type: RSZ, layer: 1, pos: 230
type: RSZ, layer: 1, pos: 174
type: RSZ, layer: 1, pos: 240

Time for candidate selection: 0.00 seconds

### Candidate
type: RSZ, layer: 1, pos: 195

### Relational analysis RSZ of RS_RSZ2_RSZ1_RSZ1_RSZ1_RSZ2_RSZ2_RSZ1_RSZ1

#### Relational analysis RSZ result of RS_RSZ2_RSZ1_RSZ1_RSZ1_RSZ2_RSZ2_RSZ1_RSZ1
Status: Status.VERIFIED
Output dim: 2, lower bound: -0.0017198, upper bound: 0.0017183
time: 1.43 seconds

### Relational analysis RSZ of RS_RSZ2_RSZ1_RSZ1_RSZ1_RSZ2_RSZ2_RSZ1_RSZ2

#### Relational analysis RSZ result of RS_RSZ2_RSZ1_RSZ1_RSZ1_RSZ2_RSZ2_RSZ1_RSZ2
Status: Status.VERIFIED
Output dim: 2, lower bound: -0.0017198, upper bound: 0.0017183
time: 1.45 seconds

## BFS RS instance: RS_RSZ2_RSZ1_RSZ1_RSZ1_RSZ2_RSZ2_RSZ2

### Backsubstitution after applying RS history:
0: -0.0041910, -0.0041119, -0.0041910, -0.0041119, -0.0000763, 0.0000763
1: -0.0096089, -0.0066455, -0.0096089, -0.0066455, -0.0028574, 0.0028572
2: 0.9649323, 0.9684886, 0.9649323, 0.9684886, -0.0034290, 0.0034287
3: -0.0123469, 0.0138834, -0.0123469, 0.0138834, -0.0252919, 0.0252898
4: -0.0017489, 0.0002460, -0.0017489, 0.0002460, -0.0019234, 0.0019236
5: 0.0155027, 0.0175190, 0.0155027, 0.0175190, -0.0019440, 0.0019441
6: 0.0033856, 0.0043663, 0.0033856, 0.0043663, -0.0009456, 0.0009455
7: -0.0113762, -0.0045784, -0.0113762, -0.0045784, -0.0065541, 0.0065546
8: 0.0077038, 0.0130968, 0.0077038, 0.0130968, -0.0051997, 0.0052001
9: 0.0115806, 0.0212805, 0.0115806, 0.0212805, -0.0093522, 0.0093529

### Unstable ReLU Count (Linear/Conv2D Layers)
- layer_idx=0, type=LayerType.Linear, total=256, inp1_unstable=17, inp2_unstable=17, delta_unstable=249
- layer_idx=2, type=LayerType.Linear, total=256, inp1_unstable=27, inp2_unstable=27, delta_unstable=256
- layer_idx=4, type=LayerType.Linear, total=256, inp1_unstable=0, inp2_unstable=0, delta_unstable=256
- layer_idx=6, type=LayerType.Linear, total=256, inp1_unstable=0, inp2_unstable=0, delta_unstable=256
- layer_idx=8, type=LayerType.Linear, total=10, inp1_unstable=2, inp2_unstable=2, delta_unstable=10

Time for backsubstitution: 1.16 seconds

### RS candidates at layer 1
type: RSZ, layer: 1, pos: 84
type: RSZ, layer: 1, pos: 114
type: RSZ, layer: 1, pos: 250
type: RSZ, layer: 1, pos: 176
type: RSZ, layer: 1, pos: 174
type: RSZ, layer: 1, pos: 29
type: RSZ, layer: 1, pos: 195
type: RSZ, layer: 1, pos: 240
type: RSZ, layer: 1, pos: 230
type: RSZ, layer: 1, pos: 25

Time for candidate selection: 0.00 seconds

### Candidate
type: RSZ, layer: 1, pos: 84

### Relational analysis RSZ of RS_RSZ2_RSZ1_RSZ1_RSZ1_RSZ2_RSZ2_RSZ2_RSZ1

#### Relational analysis RSZ result of RS_RSZ2_RSZ1_RSZ1_RSZ1_RSZ2_RSZ2_RSZ2_RSZ1
Status: Status.VERIFIED
Output dim: 2, lower bound: -0.0018239, upper bound: 0.0017952
time: 1.08 seconds

### Relational analysis RSZ of RS_RSZ2_RSZ1_RSZ1_RSZ1_RSZ2_RSZ2_RSZ2_RSZ2

#### Relational analysis RSZ result of RS_RSZ2_RSZ1_RSZ1_RSZ1_RSZ2_RSZ2_RSZ2_RSZ2
Status: Status.VERIFIED
Output dim: 2, lower bound: -0.0018091, upper bound: 0.0018107
time: 1.08 seconds

## BFS RS instance: RS_RSZ2_RSZ2_RSZ1_RSZ2_RSZ1_RSZ2_RSZ1

### Backsubstitution after applying RS history:
0: -0.0041910, -0.0041119, -0.0041910, -0.0041119, -0.0000686, 0.0000664
1: -0.0096089, -0.0066455, -0.0096089, -0.0066455, -0.0025707, 0.0024877
2: 0.9649323, 0.9684886, 0.9649323, 0.9684886, -0.0030850, 0.0029854
3: -0.0123469, 0.0138834, -0.0123469, 0.0138834, -0.0227541, 0.0220198
4: -0.0017489, 0.0002460, -0.0017489, 0.0002460, -0.0016747, 0.0017306
5: 0.0155027, 0.0175190, 0.0155027, 0.0175190, -0.0016926, 0.0017491
6: 0.0033856, 0.0043663, 0.0033856, 0.0043663, -0.0008507, 0.0008233
7: -0.0113762, -0.0045784, -0.0113762, -0.0045784, -0.0057066, 0.0058969
8: 0.0077038, 0.0130968, 0.0077038, 0.0130968, -0.0045274, 0.0046783
9: 0.0115806, 0.0212805, 0.0115806, 0.0212805, -0.0081429, 0.0084144

### Unstable ReLU Count (Linear/Conv2D Layers)
- layer_idx=0, type=LayerType.Linear, total=256, inp1_unstable=17, inp2_unstable=17, delta_unstable=249
- layer_idx=2, type=LayerType.Linear, total=256, inp1_unstable=27, inp2_unstable=27, delta_unstable=256
- layer_idx=4, type=LayerType.Linear, total=256, inp1_unstable=0, inp2_unstable=0, delta_unstable=256
- layer_idx=6, type=LayerType.Linear, total=256, inp1_unstable=0, inp2_unstable=0, delta_unstable=256
- layer_idx=8, type=LayerType.Linear, total=10, inp1_unstable=2, inp2_unstable=2, delta_unstable=10

Time for backsubstitution: 1.15 seconds

### RS candidates at layer 1
type: RSZ, layer: 1, pos: 173
type: RSZ, layer: 1, pos: 250
type: RSZ, layer: 1, pos: 195
type: RSZ, layer: 1, pos: 174
type: RSZ, layer: 1, pos: 230
type: RSZ, layer: 1, pos: 176
type: RSZ, layer: 1, pos: 240
type: RSZ, layer: 1, pos: 114
type: RSZ, layer: 1, pos: 25
type: RSZ, layer: 1, pos: 112

Time for candidate selection: 0.00 seconds

### Candidate
type: RSZ, layer: 1, pos: 173

### Relational analysis RSZ of RS_RSZ2_RSZ2_RSZ1_RSZ2_RSZ1_RSZ2_RSZ1_RSZ1

#### Relational analysis RSZ result of RS_RSZ2_RSZ2_RSZ1_RSZ2_RSZ1_RSZ2_RSZ1_RSZ1
Status: Status.UNKNOWN
Output dim: 2, lower bound: -0.0018699, upper bound: 0.0018241
time: 1.13 seconds

### Relational analysis RSZ of RS_RSZ2_RSZ2_RSZ1_RSZ2_RSZ1_RSZ2_RSZ1_RSZ2

#### Relational analysis RSZ result of RS_RSZ2_RSZ2_RSZ1_RSZ2_RSZ1_RSZ2_RSZ1_RSZ2
Status: Status.UNKNOWN
Output dim: 2, lower bound: -0.0018814, upper bound: 0.0018146
time: 1.10 seconds

## BFS RS instance: RS_RSZ2_RSZ2_RSZ1_RSZ2_RSZ1_RSZ2_RSZ2

### Backsubstitution after applying RS history:
0: -0.0041910, -0.0041119, -0.0041910, -0.0041119, -0.0000686, 0.0000665
1: -0.0096089, -0.0066455, -0.0096089, -0.0066455, -0.0025677, 0.0024917
2: 0.9649323, 0.9684886, 0.9649323, 0.9684886, -0.0030814, 0.0029901
3: -0.0123469, 0.0138834, -0.0123469, 0.0138834, -0.0227277, 0.0220546
4: -0.0017489, 0.0002460, -0.0017489, 0.0002460, -0.0016774, 0.0017286
5: 0.0155027, 0.0175190, 0.0155027, 0.0175190, -0.0016953, 0.0017470
6: 0.0033856, 0.0043663, 0.0033856, 0.0043663, -0.0008498, 0.0008246
7: -0.0113762, -0.0045784, -0.0113762, -0.0045784, -0.0057157, 0.0058901
8: 0.0077038, 0.0130968, 0.0077038, 0.0130968, -0.0045345, 0.0046729
9: 0.0115806, 0.0212805, 0.0115806, 0.0212805, -0.0081558, 0.0084047

### Unstable ReLU Count (Linear/Conv2D Layers)
- layer_idx=0, type=LayerType.Linear, total=256, inp1_unstable=17, inp2_unstable=17, delta_unstable=249
- layer_idx=2, type=LayerType.Linear, total=256, inp1_unstable=27, inp2_unstable=27, delta_unstable=256
- layer_idx=4, type=LayerType.Linear, total=256, inp1_unstable=0, inp2_unstable=0, delta_unstable=256
- layer_idx=6, type=LayerType.Linear, total=256, inp1_unstable=0, inp2_unstable=0, delta_unstable=256
- layer_idx=8, type=LayerType.Linear, total=10, inp1_unstable=2, inp2_unstable=2, delta_unstable=10

Time for backsubstitution: 1.15 seconds

### RS candidates at layer 1
type: RSZ, layer: 1, pos: 173
type: RSZ, layer: 1, pos: 25
type: RSZ, layer: 1, pos: 240
type: RSZ, layer: 1, pos: 114
type: RSZ, layer: 1, pos: 176
type: RSZ, layer: 1, pos: 230
type: RSZ, layer: 1, pos: 174
type: RSZ, layer: 1, pos: 112
type: RSZ, layer: 1, pos: 195
type: RSZ, layer: 1, pos: 250

Time for candidate selection: 0.00 seconds

### Candidate
type: RSZ, layer: 1, pos: 173

### Relational analysis RSZ of RS_RSZ2_RSZ2_RSZ1_RSZ2_RSZ1_RSZ2_RSZ2_RSZ1

#### Relational analysis RSZ result of RS_RSZ2_RSZ2_RSZ1_RSZ2_RSZ1_RSZ2_RSZ2_RSZ1
Status: Status.UNKNOWN
Output dim: 2, lower bound: -0.0018720, upper bound: 0.0018229
time: 1.08 seconds

### Relational analysis RSZ of RS_RSZ2_RSZ2_RSZ1_RSZ2_RSZ1_RSZ2_RSZ2_RSZ2

#### Relational analysis RSZ result of RS_RSZ2_RSZ2_RSZ1_RSZ2_RSZ1_RSZ2_RSZ2_RSZ2
Status: Status.UNKNOWN
Output dim: 2, lower bound: -0.0018825, upper bound: 0.0018136
time: 1.09 seconds

## BFS RS instance: RS_RSZ2_RSZ2_RSZ1_RSZ2_RSZ2_RSZ2_RSZ1

### Backsubstitution after applying RS history:
0: -0.0041910, -0.0041119, -0.0041910, -0.0041119, -0.0000679, 0.0000656
1: -0.0096089, -0.0066455, -0.0096089, -0.0066455, -0.0025412, 0.0024564
2: 0.9649323, 0.9684886, 0.9649323, 0.9684886, -0.0030495, 0.0029477
3: -0.0123469, 0.0138834, -0.0123469, 0.0138834, -0.0224927, 0.0217420
4: -0.0017489, 0.0002460, -0.0017489, 0.0002460, -0.0016536, 0.0017107
5: 0.0155027, 0.0175190, 0.0155027, 0.0175190, -0.0016713, 0.0017290
6: 0.0033856, 0.0043663, 0.0033856, 0.0043663, -0.0008410, 0.0008129
7: -0.0113762, -0.0045784, -0.0113762, -0.0045784, -0.0056346, 0.0058292
8: 0.0077038, 0.0130968, 0.0077038, 0.0130968, -0.0044703, 0.0046246
9: 0.0115806, 0.0212805, 0.0115806, 0.0212805, -0.0080402, 0.0083178

### Unstable ReLU Count (Linear/Conv2D Layers)
- layer_idx=0, type=LayerType.Linear, total=256, inp1_unstable=17, inp2_unstable=17, delta_unstable=249
- layer_idx=2, type=LayerType.Linear, total=256, inp1_unstable=27, inp2_unstable=27, delta_unstable=256
- layer_idx=4, type=LayerType.Linear, total=256, inp1_unstable=0, inp2_unstable=0, delta_unstable=256
- layer_idx=6, type=LayerType.Linear, total=256, inp1_unstable=0, inp2_unstable=0, delta_unstable=256
- layer_idx=8, type=LayerType.Linear, total=10, inp1_unstable=2, inp2_unstable=2, delta_unstable=10

Time for backsubstitution: 1.17 seconds

### RS candidates at layer 1
type: RSZ, layer: 1, pos: 240
type: RSZ, layer: 1, pos: 230
type: RSZ, layer: 1, pos: 181
type: RSZ, layer: 1, pos: 25
type: RSZ, layer: 1, pos: 114
type: RSZ, layer: 1, pos: 250
type: RSZ, layer: 1, pos: 174
type: RSZ, layer: 1, pos: 112
type: RSZ, layer: 1, pos: 195
type: RSZ, layer: 1, pos: 176

Time for candidate selection: 0.00 seconds

### Candidate
type: RSZ, layer: 1, pos: 240

### Relational analysis RSZ of RS_RSZ2_RSZ2_RSZ1_RSZ2_RSZ2_RSZ2_RSZ1_RSZ1

#### Relational analysis RSZ result of RS_RSZ2_RSZ2_RSZ1_RSZ2_RSZ2_RSZ2_RSZ1_RSZ1
Status: Status.VERIFIED
Output dim: 2, lower bound: -0.0017535, upper bound: 0.0017005
time: 1.04 seconds

### Relational analysis RSZ of RS_RSZ2_RSZ2_RSZ1_RSZ2_RSZ2_RSZ2_RSZ1_RSZ2

#### Relational analysis RSZ result of RS_RSZ2_RSZ2_RSZ1_RSZ2_RSZ2_RSZ2_RSZ1_RSZ2
Status: Status.VERIFIED
Output dim: 2, lower bound: -0.0017535, upper bound: 0.0017005
time: 1.04 seconds

## BFS RS instance: RS_RSZ2_RSZ2_RSZ1_RSZ2_RSZ2_RSZ2_RSZ2

### Backsubstitution after applying RS history:
0: -0.0041910, -0.0041119, -0.0041910, -0.0041119, -0.0000679, 0.0000656
1: -0.0096089, -0.0066455, -0.0096089, -0.0066455, -0.0025423, 0.0024555
2: 0.9649323, 0.9684886, 0.9649323, 0.9684886, -0.0030508, 0.0029467
3: -0.0123469, 0.0138834, -0.0123469, 0.0138834, -0.0225023, 0.0217344
4: -0.0017489, 0.0002460, -0.0017489, 0.0002460, -0.0016530, 0.0017114
5: 0.0155027, 0.0175190, 0.0155027, 0.0175190, -0.0016707, 0.0017297
6: 0.0033856, 0.0043663, 0.0033856, 0.0043663, -0.0008413, 0.0008126
7: -0.0113762, -0.0045784, -0.0113762, -0.0045784, -0.0056327, 0.0058317
8: 0.0077038, 0.0130968, 0.0077038, 0.0130968, -0.0044687, 0.0046266
9: 0.0115806, 0.0212805, 0.0115806, 0.0212805, -0.0080374, 0.0083213

### Unstable ReLU Count (Linear/Conv2D Layers)
- layer_idx=0, type=LayerType.Linear, total=256, inp1_unstable=17, inp2_unstable=17, delta_unstable=249
- layer_idx=2, type=LayerType.Linear, total=256, inp1_unstable=27, inp2_unstable=27, delta_unstable=256
- layer_idx=4, type=LayerType.Linear, total=256, inp1_unstable=0, inp2_unstable=0, delta_unstable=256
- layer_idx=6, type=LayerType.Linear, total=256, inp1_unstable=0, inp2_unstable=0, delta_unstable=256
- layer_idx=8, type=LayerType.Linear, total=10, inp1_unstable=2, inp2_unstable=2, delta_unstable=10

Time for backsubstitution: 1.16 seconds

### RS candidates at layer 1
type: RSZ, layer: 1, pos: 25
type: RSZ, layer: 1, pos: 181
type: RSZ, layer: 1, pos: 114
type: RSZ, layer: 1, pos: 176
type: RSZ, layer: 1, pos: 250
type: RSZ, layer: 1, pos: 112
type: RSZ, layer: 1, pos: 240
type: RSZ, layer: 1, pos: 230
type: RSZ, layer: 1, pos: 174
type: RSZ, layer: 1, pos: 195

Time for candidate selection: 0.00 seconds

### Candidate
type: RSZ, layer: 1, pos: 25

### Relational analysis RSZ of RS_RSZ2_RSZ2_RSZ1_RSZ2_RSZ2_RSZ2_RSZ2_RSZ1

#### Relational analysis RSZ result of RS_RSZ2_RSZ2_RSZ1_RSZ2_RSZ2_RSZ2_RSZ2_RSZ1
Status: Status.VERIFIED
Output dim: 2, lower bound: -0.0016334, upper bound: 0.0015758
time: 1.02 seconds

### Relational analysis RSZ of RS_RSZ2_RSZ2_RSZ1_RSZ2_RSZ2_RSZ2_RSZ2_RSZ2

#### Relational analysis RSZ result of RS_RSZ2_RSZ2_RSZ1_RSZ2_RSZ2_RSZ2_RSZ2_RSZ2
Status: Status.VERIFIED
Output dim: 2, lower bound: -0.0016334, upper bound: 0.0015758
time: 1.02 seconds

## Summary of splitting (split count: 7)
- Time for RS candidates: 3.22 seconds
RS_RSZ2_RSZ1_RSZ1_RSZ1_RSZ2_RSZ2_RSZ1_RSZ1, status: Status.VERIFIED, split count: 8, time: 3.22
Output dim: 2, lower bound: -0.0017198, upper bound: 0.0017183
RS_RSZ2_RSZ1_RSZ1_RSZ1_RSZ2_RSZ2_RSZ1_RSZ2, status: Status.VERIFIED, split count: 8, time: 3.22
Output dim: 2, lower bound: -0.0017198, upper bound: 0.0017183
RS_RSZ2_RSZ1_RSZ1_RSZ1_RSZ2_RSZ2_RSZ2_RSZ1, status: Status.VERIFIED, split count: 8, time: 3.22
Output dim: 2, lower bound: -0.0018239, upper bound: 0.0017952
RS_RSZ2_RSZ1_RSZ1_RSZ1_RSZ2_RSZ2_RSZ2_RSZ2, status: Status.VERIFIED, split count: 8, time: 3.22
Output dim: 2, lower bound: -0.0018091, upper bound: 0.0018107
RS_RSZ2_RSZ2_RSZ1_RSZ2_RSZ1_RSZ2_RSZ1_RSZ1, status: Status.UNKNOWN, split count: 8, time: 3.22
Output dim: 2, lower bound: -0.0018699, upper bound: 0.0018241
RS_RSZ2_RSZ2_RSZ1_RSZ2_RSZ1_RSZ2_RSZ1_RSZ2, status: Status.UNKNOWN, split count: 8, time: 3.22
Output dim: 2, lower bound: -0.0018814, upper bound: 0.0018146
RS_RSZ2_RSZ2_RSZ1_RSZ2_RSZ1_RSZ2_RSZ2_RSZ1, status: Status.UNKNOWN, split count: 8, time: 3.22
Output dim: 2, lower bound: -0.0018720, upper bound: 0.0018229
RS_RSZ2_RSZ2_RSZ1_RSZ2_RSZ1_RSZ2_RSZ2_RSZ2, status: Status.UNKNOWN, split count: 8, time: 3.22
Output dim: 2, lower bound: -0.0018825, upper bound: 0.0018136
RS_RSZ2_RSZ2_RSZ1_RSZ2_RSZ2_RSZ2_RSZ1_RSZ1, status: Status.VERIFIED, split count: 8, time: 3.22
Output dim: 2, lower bound: -0.0017535, upper bound: 0.0017005
RS_RSZ2_RSZ2_RSZ1_RSZ2_RSZ2_RSZ2_RSZ1_RSZ2, status: Status.VERIFIED, split count: 8, time: 3.22
Output dim: 2, lower bound: -0.0017535, upper bound: 0.0017005
RS_RSZ2_RSZ2_RSZ1_RSZ2_RSZ2_RSZ2_RSZ2_RSZ1, status: Status.VERIFIED, split count: 8, time: 3.22
Output dim: 2, lower bound: -0.0016334, upper bound: 0.0015758
RS_RSZ2_RSZ2_RSZ1_RSZ2_RSZ2_RSZ2_RSZ2_RSZ2, status: Status.VERIFIED, split count: 8, time: 3.22
Output dim: 2, lower bound: -0.0016334, upper bound: 0.0015758

## BFS RS instance: RS_RSZ2_RSZ2_RSZ1_RSZ2_RSZ1_RSZ2_RSZ1_RSZ1

### Backsubstitution after applying RS history:
0: -0.0041910, -0.0041119, -0.0041910, -0.0041119, -0.0000681, 0.0000660
1: -0.0096089, -0.0066455, -0.0096089, -0.0066455, -0.0025506, 0.0024723
2: 0.9649323, 0.9684886, 0.9649323, 0.9684886, -0.0030609, 0.0029669
3: -0.0123469, 0.0138834, -0.0123469, 0.0138834, -0.0225764, 0.0218831
4: -0.0017489, 0.0002460, -0.0017489, 0.0002460, -0.0016643, 0.0017171
5: 0.0155027, 0.0175190, 0.0155027, 0.0175190, -0.0016821, 0.0017354
6: 0.0033856, 0.0043663, 0.0033856, 0.0043663, -0.0008441, 0.0008182
7: -0.0113762, -0.0045784, -0.0113762, -0.0045784, -0.0056712, 0.0058509
8: 0.0077038, 0.0130968, 0.0077038, 0.0130968, -0.0044993, 0.0046418
9: 0.0115806, 0.0212805, 0.0115806, 0.0212805, -0.0080924, 0.0083487

### Unstable ReLU Count (Linear/Conv2D Layers)
- layer_idx=0, type=LayerType.Linear, total=256, inp1_unstable=17, inp2_unstable=17, delta_unstable=248
- layer_idx=2, type=LayerType.Linear, total=256, inp1_unstable=27, inp2_unstable=27, delta_unstable=256
- layer_idx=4, type=LayerType.Linear, total=256, inp1_unstable=0, inp2_unstable=0, delta_unstable=256
- layer_idx=6, type=LayerType.Linear, total=256, inp1_unstable=0, inp2_unstable=0, delta_unstable=256
- layer_idx=8, type=LayerType.Linear, total=10, inp1_unstable=2, inp2_unstable=2, delta_unstable=10

Time for backsubstitution: 1.16 seconds

### RS candidates at layer 1
type: RSZ, layer: 1, pos: 250
type: RSZ, layer: 1, pos: 25
type: RSZ, layer: 1, pos: 114
type: RSZ, layer: 1, pos: 240
type: RSZ, layer: 1, pos: 176
type: RSZ, layer: 1, pos: 112
type: RSZ, layer: 1, pos: 195
type: RSZ, layer: 1, pos: 174
type: RSZ, layer: 1, pos: 230

Time for candidate selection: 0.00 seconds

### Candidate
type: RSZ, layer: 1, pos: 250

### Relational analysis RSZ of RS_RSZ2_RSZ2_RSZ1_RSZ2_RSZ1_RSZ2_RSZ1_RSZ1_RSZ1
Optimization infeasible because this subproblem isn't reachable.

### Candidate
type: RSZ, layer: 1, pos: 25

### Relational analysis RSZ of RS_RSZ2_RSZ2_RSZ1_RSZ2_RSZ1_RSZ2_RSZ1_RSZ1_RSZ1

#### Relational analysis RSZ result of RS_RSZ2_RSZ2_RSZ1_RSZ2_RSZ1_RSZ2_RSZ1_RSZ1_RSZ1
Status: Status.VERIFIED
Output dim: 2, lower bound: -0.0016157, upper bound: 0.0015852
time: 1.06 seconds

### Relational analysis RSZ of RS_RSZ2_RSZ2_RSZ1_RSZ2_RSZ1_RSZ2_RSZ1_RSZ1_RSZ2

#### Relational analysis RSZ result of RS_RSZ2_RSZ2_RSZ1_RSZ2_RSZ1_RSZ2_RSZ1_RSZ1_RSZ2
Status: Status.VERIFIED
Output dim: 2, lower bound: -0.0016157, upper bound: 0.0015852
time: 1.07 seconds

## BFS RS instance: RS_RSZ2_RSZ2_RSZ1_RSZ2_RSZ1_RSZ2_RSZ1_RSZ2

### Backsubstitution after applying RS history:
0: -0.0041910, -0.0041119, -0.0041910, -0.0041119, -0.0000682, 0.0000659
1: -0.0096089, -0.0066455, -0.0096089, -0.0066455, -0.0025554, 0.0024677
2: 0.9649323, 0.9684886, 0.9649323, 0.9684886, -0.0030666, 0.0029614
3: -0.0123469, 0.0138834, -0.0123469, 0.0138834, -0.0226183, 0.0218426
4: -0.0017489, 0.0002460, -0.0017489, 0.0002460, -0.0016613, 0.0017203
5: 0.0155027, 0.0175190, 0.0155027, 0.0175190, -0.0016790, 0.0017386
6: 0.0033856, 0.0043663, 0.0033856, 0.0043663, -0.0008457, 0.0008167
7: -0.0113762, -0.0045784, -0.0113762, -0.0045784, -0.0056607, 0.0058617
8: 0.0077038, 0.0130968, 0.0077038, 0.0130968, -0.0044909, 0.0046504
9: 0.0115806, 0.0212805, 0.0115806, 0.0212805, -0.0080774, 0.0083642

### Unstable ReLU Count (Linear/Conv2D Layers)
- layer_idx=0, type=LayerType.Linear, total=256, inp1_unstable=17, inp2_unstable=17, delta_unstable=248
- layer_idx=2, type=LayerType.Linear, total=256, inp1_unstable=27, inp2_unstable=27, delta_unstable=256
- layer_idx=4, type=LayerType.Linear, total=256, inp1_unstable=0, inp2_unstable=0, delta_unstable=256
- layer_idx=6, type=LayerType.Linear, total=256, inp1_unstable=0, inp2_unstable=0, delta_unstable=256
- layer_idx=8, type=LayerType.Linear, total=10, inp1_unstable=2, inp2_unstable=2, delta_unstable=10

Time for backsubstitution: 1.17 seconds

### RS candidates at layer 1
type: RSZ, layer: 1, pos: 114
type: RSZ, layer: 1, pos: 25
type: RSZ, layer: 1, pos: 195
type: RSZ, layer: 1, pos: 176
type: RSZ, layer: 1, pos: 112
type: RSZ, layer: 1, pos: 230
type: RSZ, layer: 1, pos: 240
type: RSZ, layer: 1, pos: 174
type: RSZ, layer: 1, pos: 250

Time for candidate selection: 0.00 seconds

### Candidate
type: RSZ, layer: 1, pos: 114

### Relational analysis RSZ of RS_RSZ2_RSZ2_RSZ1_RSZ2_RSZ1_RSZ2_RSZ1_RSZ2_RSZ1

#### Relational analysis RSZ result of RS_RSZ2_RSZ2_RSZ1_RSZ2_RSZ1_RSZ2_RSZ1_RSZ2_RSZ1
Status: Status.VERIFIED
Output dim: 2, lower bound: -0.0012260, upper bound: 0.0011980
time: 0.85 seconds

### Relational analysis RSZ of RS_RSZ2_RSZ2_RSZ1_RSZ2_RSZ1_RSZ2_RSZ1_RSZ2_RSZ2

#### Relational analysis RSZ result of RS_RSZ2_RSZ2_RSZ1_RSZ2_RSZ1_RSZ2_RSZ1_RSZ2_RSZ2
Status: Status.VERIFIED
Output dim: 2, lower bound: -0.0012260, upper bound: 0.0011980
time: 0.84 seconds

## BFS RS instance: RS_RSZ2_RSZ2_RSZ1_RSZ2_RSZ1_RSZ2_RSZ2_RSZ1

### Backsubstitution after applying RS history:
0: -0.0041910, -0.0041119, -0.0041910, -0.0041119, -0.0000680, 0.0000661
1: -0.0096089, -0.0066455, -0.0096089, -0.0066455, -0.0025475, 0.0024762
2: 0.9649323, 0.9684886, 0.9649323, 0.9684886, -0.0030571, 0.0029716
3: -0.0123469, 0.0138834, -0.0123469, 0.0138834, -0.0225489, 0.0219180
4: -0.0017489, 0.0002460, -0.0017489, 0.0002460, -0.0016670, 0.0017150
5: 0.0155027, 0.0175190, 0.0155027, 0.0175190, -0.0016848, 0.0017333
6: 0.0033856, 0.0043663, 0.0033856, 0.0043663, -0.0008431, 0.0008195
7: -0.0113762, -0.0045784, -0.0113762, -0.0045784, -0.0056802, 0.0058437
8: 0.0077038, 0.0130968, 0.0077038, 0.0130968, -0.0045064, 0.0046361
9: 0.0115806, 0.0212805, 0.0115806, 0.0212805, -0.0081053, 0.0083386

### Unstable ReLU Count (Linear/Conv2D Layers)
- layer_idx=0, type=LayerType.Linear, total=256, inp1_unstable=17, inp2_unstable=17, delta_unstable=248
- layer_idx=2, type=LayerType.Linear, total=256, inp1_unstable=27, inp2_unstable=27, delta_unstable=256
- layer_idx=4, type=LayerType.Linear, total=256, inp1_unstable=0, inp2_unstable=0, delta_unstable=256
- layer_idx=6, type=LayerType.Linear, total=256, inp1_unstable=0, inp2_unstable=0, delta_unstable=256
- layer_idx=8, type=LayerType.Linear, total=10, inp1_unstable=2, inp2_unstable=2, delta_unstable=10

Time for backsubstitution: 1.16 seconds

### RS candidates at layer 1
type: RSZ, layer: 1, pos: 230
type: RSZ, layer: 1, pos: 114
type: RSZ, layer: 1, pos: 250
type: RSZ, layer: 1, pos: 112
type: RSZ, layer: 1, pos: 176
type: RSZ, layer: 1, pos: 25
type: RSZ, layer: 1, pos: 174
type: RSZ, layer: 1, pos: 195
type: RSZ, layer: 1, pos: 240

Time for candidate selection: 0.00 seconds

### Candidate
type: RSZ, layer: 1, pos: 230

### Relational analysis RSZ of RS_RSZ2_RSZ2_RSZ1_RSZ2_RSZ1_RSZ2_RSZ2_RSZ1_RSZ1

#### Relational analysis RSZ result of RS_RSZ2_RSZ2_RSZ1_RSZ2_RSZ1_RSZ2_RSZ2_RSZ1_RSZ1
Status: Status.VERIFIED
Output dim: 2, lower bound: -0.0016625, upper bound: 0.0016154
time: 1.03 seconds

### Relational analysis RSZ of RS_RSZ2_RSZ2_RSZ1_RSZ2_RSZ1_RSZ2_RSZ2_RSZ1_RSZ2

#### Relational analysis RSZ result of RS_RSZ2_RSZ2_RSZ1_RSZ2_RSZ1_RSZ2_RSZ2_RSZ1_RSZ2
Status: Status.VERIFIED
Output dim: 2, lower bound: -0.0016625, upper bound: 0.0016154
time: 1.04 seconds

## BFS RS instance: RS_RSZ2_RSZ2_RSZ1_RSZ2_RSZ1_RSZ2_RSZ2_RSZ2

### Backsubstitution after applying RS history:
0: -0.0041910, -0.0041119, -0.0041910, -0.0041119, -0.0000682, 0.0000660
1: -0.0096089, -0.0066455, -0.0096089, -0.0066455, -0.0025524, 0.0024718
2: 0.9649323, 0.9684886, 0.9649323, 0.9684886, -0.0030630, 0.0029663
3: -0.0123469, 0.0138834, -0.0123469, 0.0138834, -0.0225919, 0.0218786
4: -0.0017489, 0.0002460, -0.0017489, 0.0002460, -0.0016640, 0.0017182
5: 0.0155027, 0.0175190, 0.0155027, 0.0175190, -0.0016818, 0.0017366
6: 0.0033856, 0.0043663, 0.0033856, 0.0043663, -0.0008447, 0.0008180
7: -0.0113762, -0.0045784, -0.0113762, -0.0045784, -0.0056700, 0.0058549
8: 0.0077038, 0.0130968, 0.0077038, 0.0130968, -0.0044983, 0.0046450
9: 0.0115806, 0.0212805, 0.0115806, 0.0212805, -0.0080907, 0.0083545

### Unstable ReLU Count (Linear/Conv2D Layers)
- layer_idx=0, type=LayerType.Linear, total=256, inp1_unstable=17, inp2_unstable=17, delta_unstable=248
- layer_idx=2, type=LayerType.Linear, total=256, inp1_unstable=27, inp2_unstable=27, delta_unstable=256
- layer_idx=4, type=LayerType.Linear, total=256, inp1_unstable=0, inp2_unstable=0, delta_unstable=256
- layer_idx=6, type=LayerType.Linear, total=256, inp1_unstable=0, inp2_unstable=0, delta_unstable=256
- layer_idx=8, type=LayerType.Linear, total=10, inp1_unstable=2, inp2_unstable=2, delta_unstable=10

Time for backsubstitution: 1.15 seconds

### RS candidates at layer 1
type: RSZ, layer: 1, pos: 240
type: RSZ, layer: 1, pos: 25
type: RSZ, layer: 1, pos: 250
type: RSZ, layer: 1, pos: 195
type: RSZ, layer: 1, pos: 112
type: RSZ, layer: 1, pos: 114
type: RSZ, layer: 1, pos: 174
type: RSZ, layer: 1, pos: 230
type: RSZ, layer: 1, pos: 176

Time for candidate selection: 0.00 seconds

### Candidate
type: RSZ, layer: 1, pos: 240

### Relational analysis RSZ of RS_RSZ2_RSZ2_RSZ1_RSZ2_RSZ1_RSZ2_RSZ2_RSZ2_RSZ1

#### Relational analysis RSZ result of RS_RSZ2_RSZ2_RSZ1_RSZ2_RSZ1_RSZ2_RSZ2_RSZ2_RSZ1
Status: Status.VERIFIED
Output dim: 2, lower bound: -0.0017538, upper bound: 0.0016988
time: 1.07 seconds

### Relational analysis RSZ of RS_RSZ2_RSZ2_RSZ1_RSZ2_RSZ1_RSZ2_RSZ2_RSZ2_RSZ2

#### Relational analysis RSZ result of RS_RSZ2_RSZ2_RSZ1_RSZ2_RSZ1_RSZ2_RSZ2_RSZ2_RSZ2
Status: Status.VERIFIED
Output dim: 2, lower bound: -0.0017538, upper bound: 0.0016988
time: 1.07 seconds

## Summary of splitting (split count: 8)
- Time for RS candidates: 3.30 seconds
RS_RSZ2_RSZ2_RSZ1_RSZ2_RSZ1_RSZ2_RSZ1_RSZ1_RSZ1, status: Status.VERIFIED, split count: 9, time: 3.30
Output dim: 2, lower bound: -0.0016157, upper bound: 0.0015852
RS_RSZ2_RSZ2_RSZ1_RSZ2_RSZ1_RSZ2_RSZ1_RSZ1_RSZ2, status: Status.VERIFIED, split count: 9, time: 3.30
Output dim: 2, lower bound: -0.0016157, upper bound: 0.0015852
RS_RSZ2_RSZ2_RSZ1_RSZ2_RSZ1_RSZ2_RSZ1_RSZ2_RSZ1, status: Status.VERIFIED, split count: 9, time: 3.30
Output dim: 2, lower bound: -0.0012260, upper bound: 0.0011980
RS_RSZ2_RSZ2_RSZ1_RSZ2_RSZ1_RSZ2_RSZ1_RSZ2_RSZ2, status: Status.VERIFIED, split count: 9, time: 3.30
Output dim: 2, lower bound: -0.0012260, upper bound: 0.0011980
RS_RSZ2_RSZ2_RSZ1_RSZ2_RSZ1_RSZ2_RSZ2_RSZ1_RSZ1, status: Status.VERIFIED, split count: 9, time: 3.30
Output dim: 2, lower bound: -0.0016625, upper bound: 0.0016154
RS_RSZ2_RSZ2_RSZ1_RSZ2_RSZ1_RSZ2_RSZ2_RSZ1_RSZ2, status: Status.VERIFIED, split count: 9, time: 3.30
Output dim: 2, lower bound: -0.0016625, upper bound: 0.0016154
RS_RSZ2_RSZ2_RSZ1_RSZ2_RSZ1_RSZ2_RSZ2_RSZ2_RSZ1, status: Status.VERIFIED, split count: 9, time: 3.30
Output dim: 2, lower bound: -0.0017538, upper bound: 0.0016988
RS_RSZ2_RSZ2_RSZ1_RSZ2_RSZ1_RSZ2_RSZ2_RSZ2_RSZ2, status: Status.VERIFIED, split count: 9, time: 3.30
Output dim: 2, lower bound: -0.0017538, upper bound: 0.0016988

## RS Result
status: Status.VERIFIED
execution time: (base) + (rs) = 3.27 + 162.54 = 165.81 seconds
